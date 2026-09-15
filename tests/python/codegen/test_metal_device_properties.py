# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from types import SimpleNamespace

import pytest
import tvm
import tvm.testing
from tvm.target import detect_target


PROPERTIES = {
    "metal_language_version": 31,
    "supports_bfloat16": True,
    "supports_simdgroup_permute": True,
    "supports_simdgroup_reduction": True,
    "supports_simdgroup_matrix": True,
    "supports_metal4": False,
}


def test_metal_target_detection_uses_device_properties(monkeypatch):
    dev = SimpleNamespace(
        max_shared_memory_per_block=65536, max_threads_per_block=512, warp_size=32
    )

    def lookup(name):
        assert name == "device_api.metal.get_target_property"

        def get_property(device, property_name):
            assert device is dev
            return PROPERTIES[property_name]

        return get_property

    monkeypatch.setattr(detect_target, "get_global_func", lookup)
    target = detect_target._detect_metal(dev)
    assert int(target.attrs["max_shared_memory_per_block"]) == 65536
    assert int(target.attrs["max_threads_per_block"]) == 512
    assert int(target.attrs["thread_warp_size"]) == 32
    for name, value in PROPERTIES.items():
        assert target.attrs[name] == value


@tvm.testing.requires_metal
def test_native_metal_device_properties():
    dev = tvm.metal()
    target = tvm.target.Target.from_device(dev)
    get_property = tvm.get_global_func("device_api.metal.get_target_property")
    assert target.attrs["max_shared_memory_per_block"] == dev.max_shared_memory_per_block
    assert target.attrs["max_threads_per_block"] == dev.max_threads_per_block
    assert target.attrs["thread_warp_size"] == dev.warp_size
    for name in PROPERTIES:
        assert target.attrs[name] == get_property(dev, name)
    with pytest.raises(ValueError, match="Unknown Metal target property"):
        get_property(dev, "does_not_exist")
    assert tvm.get_global_func("tl.MetalDeviceCapabilities", allow_missing=True) is None
