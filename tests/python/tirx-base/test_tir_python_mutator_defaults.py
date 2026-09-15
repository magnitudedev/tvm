"""Default Python mutator methods return the native rewritten node."""

import tvm
from tvm import tirx


@tirx.functor.mutator
class Rewrite(tirx.PyStmtExprMutator):
    def visit_int_imm_(self, expr):
        return tirx.IntImm(expr.dtype, int(expr) + 1)

    def visit_add_(self, expr):
        return super().visit_add_(expr)

    def visit_attr_stmt_(self, stmt):
        return super().visit_attr_stmt_(stmt)


def test_default_expression_returns_rewritten_children():
    index = tirx.Var("index", "int32")
    actual = Rewrite().visit_expr(index + 4)
    assert actual is not None
    tvm.ir.assert_structural_equal(actual, index + 5)


def test_default_statement_returns_rewritten_body_and_attribute():
    before = tirx.AttrStmt(0, "preserved", 7, tirx.Evaluate(11))
    actual = Rewrite().visit_stmt(before)
    assert actual is not None
    expected = tirx.AttrStmt(0, "preserved", 8, tirx.Evaluate(12))
    tvm.ir.assert_structural_equal(actual, expected)
