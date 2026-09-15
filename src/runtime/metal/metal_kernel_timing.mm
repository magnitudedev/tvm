/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "metal_kernel_timing.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <vector>
#include <unordered_map>

#include "metal_common.h"

namespace tvm::runtime::metal {

class KernelCapture;
// Generated host code submits synchronously on Torch's dispatch queue. Carry
// the caller identity across that handoff; never capture another owner's work.
static std::mutex capture_mutex;
static std::unordered_map<uint64_t, ffi::ObjectPtr<KernelCapture>> captures;
static std::atomic<size_t> capture_count{0};
static thread_local uint64_t submitting_thread = 0;

class KernelCapture final : public ffi::ModuleObj {
 public:
  KernelCapture(id<MTLDevice> device, id<MTLCounterSet> counters, int64_t limit)
      : owner_(std::this_thread::get_id()),
        owner_identity_(reinterpret_cast<uintptr_t>(pthread_self())), limit_(limit) {
    auto descriptor = [MTLCounterSampleBufferDescriptor new];
    descriptor.counterSet = counters;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.sampleCount = static_cast<NSUInteger>(2 * limit);
    NSError* error = nil;
    samples_ = [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
    [descriptor release];
    TVM_FFI_ICHECK(samples_ != nil)
        << "Cannot allocate Metal timestamp samples: "
        << (error == nil ? "unknown error" : error.localizedDescription.UTF8String);
    device_ = [device retain];
  }

  ~KernelCapture() {
    for (auto buffer : buffers_) [buffer release];
    [samples_ release];
    [device_ release];
  }

  const char* kind() const final { return "metal_kernel_capture"; }
  int GetPropertyMask() const final { return ffi::Module::kRunnable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    auto self = ffi::GetObjectPtr<KernelCapture>(this);
    if (name == "start") return ffi::Function::FromTyped([self]() { self->Start(); });
    if (name == "finish") return ffi::Function::FromTyped([self]() { return self->Finish(); });
    if (name == "close") return ffi::Function::FromTyped([self]() { self->Close(); });
    return std::nullopt;
  }

  id<MTLComputeCommandEncoder> Encoder(id<MTLCommandBuffer> buffer,
                                       const std::string& kernel_name) {
    TVM_FFI_ICHECK(started_ && !closed_) << "Kernel capture is not active";
    if (buffer.device != device_) {
      failed_ = true;
      TVM_FFI_THROW(RuntimeError) << "Kernel capture crossed device ownership";
    }
    if (names_.size() >= static_cast<size_t>(limit_)) {
      failed_ = true;
      TVM_FFI_THROW(RuntimeError) << "Kernel timestamp capacity exhausted";
    }
    if (std::find(buffers_.begin(), buffers_.end(), buffer) == buffers_.end()) {
      buffers_.push_back([buffer retain]);
      // A failed capture may close before its caller drains the stream. Keep the
      // sampling resource alive without retaining this C++ object in a callback.
      id<MTLCounterSampleBuffer> retained_samples = samples_;
      [buffer addCompletedHandler:^(id<MTLCommandBuffer>) { [retained_samples self]; }];
    }
    auto descriptor = [MTLComputePassDescriptor computePassDescriptor];
    auto attachment = descriptor.sampleBufferAttachments[0];
    attachment.sampleBuffer = samples_;
    attachment.startOfEncoderSampleIndex = 2 * names_.size();
    attachment.endOfEncoderSampleIndex = 2 * names_.size() + 1;
    auto encoder = [buffer computeCommandEncoderWithDescriptor:descriptor];
    if (encoder == nil) failed_ = true;
    TVM_FFI_ICHECK(encoder != nil) << "Cannot create timestamped compute encoder";
    names_.push_back(kernel_name);
    return encoder;
  }

 private:
  void CheckOwner() const {
    TVM_FFI_ICHECK(owner_ == std::this_thread::get_id())
        << "Kernel capture belongs to its creating thread";
  }

  void Start() {
    CheckOwner();
    TVM_FFI_ICHECK(!started_ && !closed_) << "Kernel capture is single-use";
    std::lock_guard<std::mutex> lock(capture_mutex);
    TVM_FFI_ICHECK(captures.count(owner_identity_) == 0) << "A kernel capture is already active";
    [device_ sampleTimestamps:&cpu_start_ gpuTimestamp:&gpu_start_];
    started_ = true;
    captures.emplace(owner_identity_, ffi::GetObjectPtr<KernelCapture>(this));
    capture_count.fetch_add(1, std::memory_order_release);
  }

  ffi::Array<ffi::Map<ffi::String, ffi::Any>> Finish() {
    CheckOwner();
    TVM_FFI_ICHECK(started_ && !closed_) << "Kernel capture is not active";
    Close();
    TVM_FFI_ICHECK(!failed_) << "Kernel capture failed; partial timestamps are not valid";
    // No queue commit or synchronization here. The execution owner supplies the
    // completion boundary; this is a read of already-completed GPU samples.
    for (auto buffer : buffers_) {
      TVM_FFI_ICHECK(buffer.status == MTLCommandBufferStatusCompleted)
          << "Kernel capture requires successfully completed command buffers";
    }
    ffi::Array<ffi::Map<ffi::String, ffi::Any>> result;
    if (names_.empty()) return result;
    MTLTimestamp cpu_end, gpu_end;
    [device_ sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end];
    TVM_FFI_ICHECK(cpu_end > cpu_start_ && gpu_end > gpu_start_)
        << "Invalid CPU/GPU timestamp calibration";
    const long double nanoseconds_per_tick =
        static_cast<long double>(cpu_end - cpu_start_) / (gpu_end - gpu_start_);
    AUTORELEASEPOOL {
      NSData* data = [samples_ resolveCounterRange:NSMakeRange(0, 2 * names_.size())];
      TVM_FFI_ICHECK(data != nil && data.length == 2 * names_.size() * sizeof(MTLCounterResultTimestamp))
          << "Incomplete Metal timestamp results";
      const auto* values = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
      for (size_t index = 0; index < names_.size(); ++index) {
        uint64_t start = values[2 * index].timestamp;
        uint64_t end = values[2 * index + 1].timestamp;
        TVM_FFI_ICHECK(start != MTLCounterErrorValue && end != MTLCounterErrorValue && end >= start)
            << "Invalid Metal kernel timestamps";
        long double elapsed = static_cast<long double>(end - start) * nanoseconds_per_tick;
        TVM_FFI_ICHECK(std::isfinite(elapsed) && elapsed <= std::numeric_limits<int64_t>::max())
            << "Kernel timestamp duration is out of range";
        // Keep endpoints in one capture-relative calibrated GPU clock. They are
        // not host perf_counter timestamps and must not be joined to that clock.
        TVM_FFI_ICHECK(start >= gpu_start_) << "Kernel timestamp precedes capture";
        long double offset = static_cast<long double>(start - gpu_start_) * nanoseconds_per_tick;
        TVM_FFI_ICHECK(std::isfinite(offset) &&
                      offset + elapsed <= std::numeric_limits<int64_t>::max())
            << "Kernel timestamp endpoint is out of range";
        const int64_t started_ns = static_cast<int64_t>(std::llround(offset));
        const int64_t elapsed_ns = static_cast<int64_t>(std::llround(elapsed));
        result.push_back({{"name", ffi::String(names_[index])},
                          {"elapsed_ns", elapsed_ns},
                          {"started_ns", started_ns},
                          {"ended_ns", started_ns + elapsed_ns},
                          {"dispatch", static_cast<int64_t>(index)}});
      }
    };
    return result;
  }

  void Close() {
    CheckOwner();
    std::lock_guard<std::mutex> lock(capture_mutex);
    auto found = captures.find(owner_identity_);
    if (found != captures.end() && found->second.get() == this) {
      captures.erase(found);
      capture_count.fetch_sub(1, std::memory_order_release);
    }
    closed_ = true;
  }

  std::thread::id owner_;
  uint64_t owner_identity_;
  int64_t limit_;
  id<MTLDevice> device_ = nil;
  id<MTLCounterSampleBuffer> samples_ = nil;
  std::vector<id<MTLCommandBuffer>> buffers_;
  std::vector<std::string> names_;
  MTLTimestamp cpu_start_ = 0, gpu_start_ = 0;
  bool started_ = false, closed_ = false, failed_ = false;
};

id<MTLComputeCommandEncoder> CreateKernelEncoder(id<MTLCommandBuffer> buffer,
                                                const std::string& kernel_name) {
  if (capture_count.load(std::memory_order_acquire) == 0) return [buffer computeCommandEncoder];
  ffi::ObjectPtr<KernelCapture> capture;
  {
    std::lock_guard<std::mutex> lock(capture_mutex);
    auto found = captures.find(submitting_thread);
    if (found != captures.end()) capture = found->second;
  }
  return capture ? capture->Encoder(buffer, kernel_name) : [buffer computeCommandEncoder];
}

void SetKernelCaptureSubmittingThread(uint64_t identity) { submitting_thread = identity; }

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "runtime.metal.CreateKernelCapture",
      [](int ordinal, int64_t limit) -> ffi::Optional<ffi::Module> {
        TVM_FFI_ICHECK(ordinal >= 0 && limit > 0 && limit <= 65536)
            << "Invalid device or kernel timestamp capacity";
        ffi::Optional<ffi::Module> result;
        AUTORELEASEPOOL {
          auto device = MetalWorkspace::Global()->GetDevice(Device{kDLMetal, ordinal});
          if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
            return;
          for (id<MTLCounterSet> counters in device.counterSets) {
            if ([counters.name isEqualToString:MTLCommonCounterSetTimestamp]) {
              result = ffi::Module(ffi::make_object<KernelCapture>(device, counters, limit));
              return;
            }
          }
        };
        return result;
      });
}
}  // namespace tvm::runtime::metal
