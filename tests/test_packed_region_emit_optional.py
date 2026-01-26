import platform
import unittest
from pathlib import Path

from stc.packed_circuit import eval_packed_circuit_words
from stc.packed_region_emit import EmitRegionsConfig, emit_avx512_u64_regions
from stc.packed_regions import RegionCaps
from stc.testing.native_x86_runner import (
    compile_shared,
    have_avx512,
    run_avx512_steps_shared,
)
from stc.tick_ir import BitVecConst, BitVecType, Slice, TickIR, Var, Xor
from stc.tick_ir_to_packed_circuit_state import lower_tick_ir_to_packed_circuit_state


@unittest.skipUnless(platform.machine() == "x86_64", "x86_64 only")
@unittest.skipUnless(have_avx512(), "requires avx512f")
class TestPackedRegionEmitOptional(unittest.TestCase):
    def test_regions_steps_shared_matches_eval(self) -> None:
        expr = Xor(a=Var("x"), b=Var("s"))
        ir = TickIR(
            name="xor128",
            inputs={"x": BitVecType(128)},
            outputs={"o0": BitVecType(64), "o1": BitVecType(64)},
            state={"s": BitVecType(128)},
            reset_state={"s": BitVecConst(width=128, value=0)},
            next_state={"s": expr},
            output_exprs={
                "o0": Slice(x=expr, offset=0, width=64),
                "o1": Slice(x=expr, offset=64, width=64),
            },
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)

        c_code = emit_avx512_u64_regions(
            circuit,
            layout,
            function_name="circuit",
            config=EmitRegionsConfig(caps=RegionCaps(max_gates=1, max_boundary=64)),
        )

        td = Path("out") / "test_packed_regions"
        td.mkdir(parents=True, exist_ok=True)
        c_path = td / "circuit_avx512_u64_regions.c"
        c_path.write_text(c_code, encoding="utf-8")
        build = compile_shared(c_path, cflags=["-mavx512f"])

        # One-step reference: interpret PackedCircuitState with concrete u64 values.
        x0 = 0x0123456789ABCDEF0123456789ABCDEF
        s0 = 0x0F0E0D0C0B0A09080706050403020100
        in_io = [x0 & ((1 << 64) - 1), (x0 >> 64) & ((1 << 64) - 1)]
        st_in = [s0 & ((1 << 64) - 1), (s0 >> 64) & ((1 << 64) - 1)]

        # circuit input vector = (inputs, state)
        outs_ref = eval_packed_circuit_words(circuit, in_io + st_in)
        out_io_ref = outs_ref[:2]
        st_out_ref = outs_ref[2:]

        out_io, st_out = run_avx512_steps_shared(
            build.so_path,
            in_io,
            st_in,
            input_io_bits=2,
            state_bits=2,
            output_io_bits=2,
            steps=1,
        )
        self.assertEqual(out_io, out_io_ref)
        self.assertEqual(st_out, st_out_ref)


if __name__ == "__main__":
    unittest.main()
