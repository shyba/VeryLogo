from __future__ import annotations

from dataclasses import replace

from stc.passmgr import Pass, PassContext, PassMetrics
from stc.tick_ir import TickIR, Expr, And, Or, Xor


def collect_assoc_leaves(expr: Expr, op_type: type) -> list[Expr]:
    """Collect all leaves of same associative operation type."""
    if isinstance(expr, op_type):
        left_leaves = collect_assoc_leaves(expr.a, op_type)
        right_leaves = collect_assoc_leaves(expr.b, op_type)
        return left_leaves + right_leaves
    else:
        return [expr]


def build_balanced_tree(op_type: type, leaves: list[Expr]) -> Expr:
    """Build a balanced binary tree from leaves."""
    if len(leaves) == 1:
        return leaves[0]

    if len(leaves) == 2:
        return op_type(leaves[0], leaves[1])

    mid = len(leaves) // 2
    left_tree = build_balanced_tree(op_type, leaves[:mid])
    right_tree = build_balanced_tree(op_type, leaves[mid:])
    return op_type(left_tree, right_tree)


class BalanceAssociativePass(Pass):
    """Balance associative operation trees to minimize depth."""

    @property
    def name(self) -> str:
        return "balance"

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        changed = False
        expressions_modified = 0

        def balance_expr(expr: Expr) -> Expr:
            nonlocal changed, expressions_modified

            if not isinstance(expr, (Xor, And, Or)):
                return expr

            leaves = collect_assoc_leaves(expr, type(expr))
            if len(leaves) <= 2:
                return expr

            changed = True
            expressions_modified += 1
            return build_balanced_tree(type(expr), leaves)

        new_output_exprs = {}
        for name, expr in ir.output_exprs.items():
            new_output_exprs[name] = balance_expr(expr)

        new_next_state = {}
        for name, expr in ir.next_state.items():
            new_next_state[name] = balance_expr(expr)

        new_ir = replace(
            ir,
            output_exprs=new_output_exprs,
            next_state=new_next_state,
        )

        return new_ir, PassMetrics(
            changed=changed,
            expressions_modified=expressions_modified,
        )
