from __future__ import annotations

from dataclasses import dataclass

from stc.tick_ir import (
    Add,
    And,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Eq,
    Mux,
    Not,
    TickIR,
    Var,
    Xor,
)


@dataclass(frozen=True)
class Case:
    name: str
    ir: TickIR
    inputs: list[dict[str, int | bool]]
    expected_outputs: list[dict[str, int | bool]]


def case_combinational_not() -> Case:
    ir = TickIR(
        name="comb_not",
        inputs={"i": BoolType()},
        outputs={"o": BoolType()},
        state={},
        reset_state={},
        next_state={},
        output_exprs={"o": Not(x=Var(name="i"))},
    )
    inputs = [{"i": 0}, {"i": 1}, {"i": 0}]
    expected = [{"o": 1}, {"o": 0}, {"o": 1}]
    return Case(
        name="combinational_not", ir=ir, inputs=inputs, expected_outputs=expected
    )


def case_fsm_toggle() -> Case:
    ir = TickIR(
        name="fsm_toggle",
        inputs={"i": BoolType()},
        outputs={"o": BoolType()},
        state={"s": BoolType()},
        reset_state={"s": BoolConst(value=False)},
        next_state={"s": Xor(a=Var(name="s"), b=Var(name="i"))},
        output_exprs={"o": Var(name="s")},
    )
    inputs = [{"i": 0}, {"i": 1}, {"i": 0}, {"i": 1}]
    expected = [{"o": 0}, {"o": 0}, {"o": 1}, {"o": 1}]
    return Case(name="fsm_toggle", ir=ir, inputs=inputs, expected_outputs=expected)


def case_temporal_pipeline_counter() -> Case:
    ir = TickIR(
        name="counter_pipe",
        inputs={},
        outputs={"o": BitVecType(width=3)},
        state={"c": BitVecType(width=3)},
        reset_state={"c": BitVecConst(width=3, value=0)},
        next_state={"c": Add(a=Var(name="c"), b=BitVecConst(width=3, value=1))},
        output_exprs={"o": Var(name="c")},
    )
    inputs = [{}, {}, {}, {}]
    expected = [{"o": 0}, {"o": 1}, {"o": 2}, {"o": 3}]
    return Case(
        name="temporal_pipeline_counter",
        ir=ir,
        inputs=inputs,
        expected_outputs=expected,
    )


def case_debounce_redundant_state() -> Case:
    ir = TickIR(
        name="debounce_like",
        inputs={"i": BoolType()},
        outputs={"o": BoolType()},
        state={"o_reg": BoolType(), "dead": BoolType()},
        reset_state={"o_reg": BoolConst(value=False), "dead": BoolConst(value=False)},
        next_state={"o_reg": Var(name="i"), "dead": BoolConst(value=False)},
        output_exprs={"o": Var(name="o_reg")},
    )
    inputs = [{"i": 0}, {"i": 1}, {"i": 1}]
    expected = [{"o": 0}, {"o": 0}, {"o": 1}]
    return Case(
        name="debounce_redundant_state", ir=ir, inputs=inputs, expected_outputs=expected
    )


def all_cases() -> list[Case]:
    return [
        case_combinational_not(),
        case_fsm_toggle(),
        case_temporal_pipeline_counter(),
        case_debounce_redundant_state(),
    ]
