import platform
import shutil
import unittest

from stc.backend_sched import generate_scheduled_code
from stc.tick_ir import BitVecConst, BitVecType, TickIR, Var, Xor
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state
from stc.testing import eval_circuitstate_bits
from stc.testing.native_x86_runner import (
    compile_shared,
    have_avx512,
    run_avx512_steps_shared,
)


@unittest.skipUnless(
    shutil.which("cc") and platform.machine() == "x86_64" and have_avx512(),
    "requires cc, x86_64, and avx512f",
)
class TestEmitStepsAvx512Optional(unittest.TestCase):
    def test_steps_shared_matches_repeated_single_steps(self) -> None:
        # 1-bit state toggled by input bit: s' = s xor i, o = s.
        ir = TickIR(
            name="toggle",
            inputs={"i": BitVecType(1)},
            outputs={"o": BitVecType(1)},
            state={"s": BitVecType(1)},
            reset_state={"s": BitVecConst(width=1, value=0)},
            next_state={"s": Xor(a=Var("s"), b=Var("i"))},
            output_exprs={"o": Var("s")},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        input_io_bits = sum(int(v["width"]) for v in layout.inputs.values())
        output_io_bits = sum(int(v["width"]) for v in layout.outputs.values())
        state_bits = sum(int(v["width"]) for v in layout.state.values())
        self.assertEqual(input_io_bits, 1)
        self.assertEqual(output_io_bits, 1)
        self.assertEqual(state_bits, 1)

        code = generate_scheduled_code(
            circuit,
            target="avx512",
            scheduler="list",
            function_name="circuit",
            io_split=(input_io_bits, output_io_bits),
        )

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            c_path = td / "circuit.c"
            c_path.write_text(code, encoding="utf-8")
            nb = compile_shared(
                c_path,
                cflags=["-mavx512f", "-mavx512vl", "-mavx512bw", "-mavx512dq"],
            )

            # Input i=1 for all steps, initial s=0.
            K = 9
            out_io, st_out = run_avx512_steps_shared(
                nb.so_path,
                [0xFFFFFFFFFFFFFFFF],
                [0],
                input_io_bits=input_io_bits,
                state_bits=state_bits,
                output_io_bits=output_io_bits,
                steps=K,
            )

            # Python reference using combinational CircuitState stepping.
            cs = circuit.to_dict()
            st = [0]
            in_io = [1]
            for _ in range(K):
                packed_in = [in_io[0], st[0]]
                out_bits = eval_circuitstate_bits(cs, packed_in)
                # next_state is appended after primary outputs.
                st = [out_bits[output_io_bits + 0]]

            self.assertEqual(st_out[0] & 1, st[0] & 1)
            # Output o is pre-state each tick; after K steps, last computed o is s_{K-1}.
            # For toggling with i=1 starting at 0, s alternates. s_{K-1} == (K-1)%2.
            self.assertEqual(out_io[0] & 1, (K - 1) & 1)
