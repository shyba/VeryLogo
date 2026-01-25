from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from stc.tick_ir import (
    Expr,
    And,
    Or,
    Xor,
    Not,
    BoolConst,
    BitVecConst,
    SimdConst,
    Var,
    TernaryLut,
    BoolType,
    BitVecType,
    SimdType,
)
from stc.passmgr import Pass, PassContext, PassMetrics


def compute_imm8(func: Callable[[int, int, int], int]) -> int:
    """Compute imm8 truth table from a 3-input boolean function.

    func(a, b, c) -> output_bit for each combination.

    Example:
        # AND(a, b) ignoring c
        imm8 = compute_imm8(lambda a, b, c: a & b)  # = 0x80

        # XOR(a, XOR(b, c))
        imm8 = compute_imm8(lambda a, b, c: a ^ b ^ c)  # = 0x96

        # Majority(a, b, c)
        imm8 = compute_imm8(lambda a, b, c: (a & b) | (b & c) | (a & c))  # = 0xE8
    """
    result = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        if func(a, b, c):
            result |= 1 << i
    return result


def imm8_to_func(imm8: int) -> Callable[[int, int, int], int]:
    """Convert imm8 back to a function (for verification)."""

    def func(a: int, b: int, c: int) -> int:
        idx = (a << 2) | (b << 1) | c
        return (imm8 >> idx) & 1

    return func


IMM8_AND_AB = compute_imm8(lambda a, b, c: a & b)
IMM8_OR_AB = compute_imm8(lambda a, b, c: a | b)
IMM8_XOR_AB = compute_imm8(lambda a, b, c: a ^ b)
IMM8_XOR_ABC = compute_imm8(lambda a, b, c: a ^ b ^ c)
IMM8_MAJ = compute_imm8(lambda a, b, c: (a & b) | (b & c) | (a & c))
IMM8_AND_ABC = compute_imm8(lambda a, b, c: a & b & c)
IMM8_OR_ABC = compute_imm8(lambda a, b, c: a | b | c)


def negate_imm8(imm8: int) -> int:
    """Return imm8 for NOT(f) given imm8 for f."""
    return imm8 ^ 0xFF


def complement_input_a(imm8: int) -> int:
    """Return imm8 with input 'a' complemented (absorbed NOT)."""
    lo = imm8 & 0x0F
    hi = (imm8 >> 4) & 0x0F
    return (lo << 4) | hi


def complement_input_b(imm8: int) -> int:
    """Return imm8 with input 'b' complemented."""
    result = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        src_idx = (a << 2) | ((1 - b) << 1) | c
        if (imm8 >> src_idx) & 1:
            result |= 1 << i
    return result


def complement_input_c(imm8: int) -> int:
    """Return imm8 with input 'c' complemented."""
    result = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        src_idx = (a << 2) | (b << 1) | (1 - c)
        if (imm8 >> src_idx) & 1:
            result |= 1 << i
    return result


@dataclass
class Cone3:
    """A 3-input cone extracted from an expression tree."""

    root: Expr
    inputs: tuple[Expr, Expr, Expr]
    imm8: int
    gate_count: int

    def to_ternary_lut(self) -> TernaryLut:
        return TernaryLut(
            a=self.inputs[0],
            b=self.inputs[1],
            c=self.inputs[2],
            imm8=self.imm8,
        )


def extract_3input_cone(expr: Expr, max_depth: int = 4) -> Cone3 | None:
    """Extract a 3-input boolean cone from expression, if possible.

    Traverses the expression tree up to max_depth, collecting unique
    leaf inputs. If exactly 3 unique inputs are found, computes the
    imm8 truth table and returns a Cone3.

    Returns None if:
    - More than 3 unique inputs
    - Expression contains non-boolean ops
    - Depth exceeds max_depth
    """
    inputs_found: dict[Expr, int] = {}
    gate_count = 0

    def collect_inputs(e: Expr, depth: int) -> bool:
        nonlocal gate_count

        if depth > max_depth:
            return False

        if isinstance(e, (Var, BoolConst, BitVecConst, SimdConst)):
            if e not in inputs_found:
                if len(inputs_found) >= 3:
                    return False
                inputs_found[e] = len(inputs_found)
            return True

        if isinstance(e, Not):
            return collect_inputs(e.x, depth)

        if isinstance(e, (And, Or, Xor)):
            gate_count += 1
            return collect_inputs(e.a, depth + 1) and collect_inputs(e.b, depth + 1)

        return False

    if not collect_inputs(expr, 0):
        return None

    if len(inputs_found) < 1:
        return None

    input_list = list(inputs_found.keys())
    while len(input_list) < 3:
        const_zero = BoolConst(value=False)
        input_list.append(const_zero)
        inputs_found[const_zero] = len(inputs_found)

    def eval_expr(e: Expr, vals: dict[Expr, int]) -> int:
        if isinstance(e, BoolConst):
            return 1 if e.value else 0
        if isinstance(e, (BitVecConst, SimdConst)):
            return e.value & 1
        if e in vals:
            return vals[e]
        if isinstance(e, Not):
            return 1 - eval_expr(e.x, vals)
        if isinstance(e, And):
            return eval_expr(e.a, vals) & eval_expr(e.b, vals)
        if isinstance(e, Or):
            return eval_expr(e.a, vals) | eval_expr(e.b, vals)
        if isinstance(e, Xor):
            return eval_expr(e.a, vals) ^ eval_expr(e.b, vals)
        raise ValueError(f"Cannot evaluate {type(e)}")

    imm8 = 0
    for i in range(8):
        a_val = (i >> 2) & 1
        b_val = (i >> 1) & 1
        c_val = i & 1
        vals = {
            input_list[0]: a_val,
            input_list[1]: b_val,
            input_list[2]: c_val,
        }
        if eval_expr(expr, vals):
            imm8 |= 1 << i

    return Cone3(
        root=expr,
        inputs=(input_list[0], input_list[1], input_list[2]),
        imm8=imm8,
        gate_count=gate_count,
    )


class TernaryMappingPass(Pass):
    """Map eligible 3-input cones to TernaryLut nodes."""

    def __init__(self, min_gate_savings: int = 1):
        self.min_gate_savings = min_gate_savings

    @property
    def name(self) -> str:
        return "ternary-mapping"

    def should_run(self, ctx: PassContext) -> bool:
        prims = {p.name for p in ctx.technology.primitives()}
        return "lop3" in prims or "vpternlog" in prims

    def run(self, ir, ctx: PassContext) -> tuple:
        from stc.tick_ir import TickIR

        changed = False
        mapped_count = 0

        def try_map(expr: Expr) -> Expr:
            nonlocal changed, mapped_count

            if isinstance(expr, TernaryLut):
                return expr

            cone = extract_3input_cone(expr)
            if cone is None:
                return expr

            if cone.gate_count < 1 + self.min_gate_savings:
                return expr

            changed = True
            mapped_count += 1
            return cone.to_ternary_lut()

        new_output_exprs = {
            k: _map_expr(v, try_map) for k, v in ir.output_exprs.items()
        }
        new_next_state = {k: _map_expr(v, try_map) for k, v in ir.next_state.items()}

        new_ir = TickIR(
            name=ir.name,
            inputs=dict(ir.inputs),
            outputs=dict(ir.outputs),
            state=dict(ir.state),
            reset_state=dict(ir.reset_state),
            next_state=new_next_state,
            output_exprs=new_output_exprs,
        )

        return new_ir, PassMetrics(
            changed=changed,
            expressions_modified=mapped_count,
        )


def _map_expr(expr: Expr, f: Callable[[Expr], Expr]) -> Expr:
    """Recursively apply f to expr and all subexpressions."""
    if isinstance(expr, Var):
        return expr
    if isinstance(expr, (BoolConst, BitVecConst, SimdConst)):
        return expr

    mapped_children = {}
    if isinstance(expr, Not):
        mapped_children = {"x": _map_expr(expr.x, f)}
    elif isinstance(expr, (And, Or, Xor)):
        mapped_children = {"a": _map_expr(expr.a, f), "b": _map_expr(expr.b, f)}
    elif isinstance(expr, TernaryLut):
        mapped_children = {
            "a": _map_expr(expr.a, f),
            "b": _map_expr(expr.b, f),
            "c": _map_expr(expr.c, f),
            "imm8": expr.imm8,
        }
    elif hasattr(expr, "__dataclass_fields__"):
        for field_name in expr.__dataclass_fields__:
            field_val = getattr(expr, field_name)
            if isinstance(field_val, Expr):
                mapped_children[field_name] = _map_expr(field_val, f)
            else:
                mapped_children[field_name] = field_val

    if mapped_children:
        mapped_expr = type(expr)(**mapped_children)
    else:
        mapped_expr = expr

    return f(mapped_expr)
