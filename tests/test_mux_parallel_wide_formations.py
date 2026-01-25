import os
import random
import unittest

from stc.tick_ir import (
    TickIR,
    BoolType,
    BitVecType,
    Var,
    BoolConst,
    BitVecConst,
    Xor,
    And,
    Or,
    Not,
    Mux,
    Concat,
    Slice,
)
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
from stc.testing import eval_circuitstate_bits, step_tickir


def _bits_le(value: int, width: int) -> list[int]:
    return [((value >> i) & 1) for i in range(width)]


def _set_bits(packed: list[int], lsb: int, width: int, value: int) -> None:
    for i in range(width):
        packed[lsb + i] = (value >> i) & 1


class TestMuxParallelWideFormations(unittest.TestCase):
    def test_tickir_lowering_matches_interpreter_for_muxy_wide_logic(self):
        # Mux-heavy + wide concat/slice/reorder + parallel fanout.
        #
        # This is designed to stress the lowering and scheduling surfaces that
        # frequently go wrong in compiler-like transforms.
        a = Var("a")  # 32-bit
        b = Var("b")  # 32-bit
        sel = Var("sel")  # 1-bit
        reset = Var("reset")  # 1-bit
        st = Var("st")  # 64-bit state

        # A byte reorder / lane shuffle pattern (similar spirit to Keccak's byte swaps).
        def byte(x: Var, idx: int):
            return Slice(x=x, offset=idx * 8, width=8)

        a_swapped = Concat(parts=[byte(a, 0), byte(a, 1), byte(a, 2), byte(a, 3)])
        b_swapped = Concat(parts=[byte(b, 3), byte(b, 2), byte(b, 1), byte(b, 0)])

        # Parallel graph: build two candidates then mux between them.
        cand0 = Xor(a=a_swapped, b=Slice(x=st, offset=0, width=32))
        cand1 = Or(a=b_swapped, b=Slice(x=st, offset=32, width=32))
        out32 = Mux(cond=sel, a=cand1, b=cand0)

        # Next state uses many overlapping slices and muxes.
        st_lo = Slice(x=st, offset=0, width=32)
        st_hi = Slice(x=st, offset=32, width=32)
        mix0 = Xor(a=st_lo, b=out32)
        mix1 = Xor(a=st_hi, b=out32)
        nx = Concat(parts=[Mux(cond=sel, a=mix1, b=mix0), Mux(cond=sel, a=mix0, b=mix1)])

        ir = TickIR(
            name="muxy",
            inputs={"a": BitVecType(32), "b": BitVecType(32), "sel": BoolType(), "reset": BoolType()},
            outputs={"out": BitVecType(32)},
            state={"st": BitVecType(64)},
            reset_state={"st": BitVecConst(width=64, value=0)},
            next_state={"st": nx},
            output_exprs={"out": out32},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        cs = circuit.to_dict()

        rng = random.Random(0 if os.environ.get("STC_SEED") is None else int(os.environ["STC_SEED"]))

        cur_st = 0
        packed = [0] * layout.input_bits

        for t in range(200):
            aval = rng.getrandbits(32)
            bval = rng.getrandbits(32)
            selv = rng.getrandbits(1)
            rst = 1 if (t == 0) else 0

            # CircuitState step (packed inputs+state -> outputs+next_state)
            _set_bits(packed, layout.inputs["a"]["lsb"], 32, aval)
            _set_bits(packed, layout.inputs["b"]["lsb"], 32, bval)
            _set_bits(packed, layout.inputs["sel"]["lsb"], 1, selv)
            _set_bits(packed, layout.inputs["reset"]["lsb"], 1, rst)
            _set_bits(packed, layout.state["st"]["lsb"], 64, cur_st)

            out_bits = eval_circuitstate_bits(cs, packed)
            out_val = sum((out_bits[layout.outputs["out"]["lsb"] + i] & 1) << i for i in range(32))
            nx_st = sum((out_bits[layout.next_state["st"]["lsb"] + i] & 1) << i for i in range(64))

            # TickIR step from the same current state.
            step = step_tickir(
                ir,
                state={"st": cur_st},
                inputs={"a": aval, "b": bval, "sel": bool(selv), "reset": bool(rst)},
            )

            self.assertEqual(out_val & 0xFFFFFFFF, int(step.outputs["out"]) & 0xFFFFFFFF)
            self.assertEqual(nx_st & ((1 << 64) - 1), int(step.next_state["st"]) & ((1 << 64) - 1))

            cur_st = int(step.next_state["st"])
