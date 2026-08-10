import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_futhark import (
    CodegenError,
    compute_futhark_source_metrics,
    emit_futhark,
    emit_futhark_manifest,
    resolve_futhark_mode,
)
from stc.tick_ir import (
    Add,
    BitVecConst,
    BitVecType,
    BoolType,
    Concat,
    LShr,
    Lut8,
    Mux,
    Not,
    AShr,
    Shl,
    SimdType,
    Slice,
    Sub,
    TickIR,
    Ult,
    Var,
    Xor,
)


class TestBackendFuthark(unittest.TestCase):
    def test_emits_entries_and_parallel_soac_nodes(self) -> None:
        t8 = BitVecType(width=8)
        ir = TickIR(
            name="top",
            inputs={"a": t8, "b": t8, "sel": BoolType()},
            outputs={"y": t8, "nsel": BoolType()},
            state={"q": t8},
            reset_state={"q": BitVecConst(width=8, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={
                "y": Mux(cond=Var("sel"), a=Var("q"), b=Var("b")),
                "nsel": Not(x=Var("sel")),
            },
        )
        src = emit_futhark(ir)
        self.assertIn("entry init_state", src)
        self.assertIn("entry step_batch", src)
        self.assertIn("entry run_steps_batch", src)
        self.assertIn("map2", src)
        self.assertIn("map3", src)

    def test_emits_shift_helpers_and_concat(self) -> None:
        t8 = BitVecType(width=8)
        ir = TickIR(
            name="concat_shift",
            inputs={"x": t8, "sh": t8},
            outputs={"y": t8, "z": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "y": Concat(
                    parts=[
                        Slice(x=Var("x"), offset=4, width=4),
                        Slice(x=Var("x"), offset=0, width=4),
                    ]
                ),
                "z": Shl(a=Var("x"), b=Var("sh")),
            },
        )
        src = emit_futhark(ir)
        self.assertIn("def shl_w8", src)
        self.assertIn("map2 (\\x y -> shl_w8 x y)", src)

    def test_supports_wide_bitvecs_and_lut8(self) -> None:
        t8 = BitVecType(width=8)
        t128 = BitVecType(width=128)
        table = list(range(256))
        ir = TickIR(
            name="wide",
            inputs={"x": t128},
            outputs={"lo": t8, "swapped": t128, "lt": BoolType()},
            state={"q": t128},
            reset_state={"q": BitVecConst(width=128, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("x"))},
            output_exprs={
                "lo": Lut8(x=Slice(x=Var("q"), offset=0, width=8), table=table),
                "swapped": Concat(
                    parts=[
                        Slice(x=Var("q"), offset=64, width=64),
                        Slice(x=Var("q"), offset=0, width=64),
                    ]
                ),
                "lt": Ult(a=Var("q"), b=Var("x")),
            },
        )
        src = emit_futhark(ir)
        self.assertIn("def add_chunks", src)
        self.assertIn("def ult_chunks", src)
        self.assertIn("def slice_chunks", src)
        self.assertIn("def lut8_t0", src)
        self.assertIn("[n][2]u64", src)
        self.assertNotIn("pack_bits_w128", src)
        self.assertNotIn("unpack_bits_w128", src)

    def test_manifest_shape(self) -> None:
        t8 = BitVecType(width=8)
        ir = TickIR(
            name="m",
            inputs={"z": t8, "a": BoolType()},
            outputs={"o": t8},
            state={"q": t8},
            reset_state={"q": BitVecConst(width=8, value=0)},
            next_state={"q": Var("q")},
            output_exprs={"o": Var("z")},
        )
        manifest = emit_futhark_manifest(ir)
        self.assertEqual(manifest["backend"], "futhark")
        self.assertEqual(manifest["module_name"], "m")
        self.assertEqual([item["name"] for item in manifest["inputs"]], ["a", "z"])
        self.assertEqual(manifest["inputs"][0]["param"], "in_a")
        self.assertEqual(manifest["state"][0]["param"], "st_q")
        self.assertIn("step_batch", manifest["entries"])
        self.assertEqual(manifest["layout"], "soa-batch")
        self.assertEqual(manifest["abi_version"], 2)

    def test_combinational_fast_mode_emits_eval_entries(self) -> None:
        t8 = BitVecType(width=8)
        ir = TickIR(
            name="comb",
            inputs={"a": t8, "b": t8},
            outputs={"y": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Add(a=Var("a"), b=Var("b"))},
        )
        selected, reason = resolve_futhark_mode(ir, "auto")
        self.assertEqual(selected, "combinational_fast")
        self.assertIsNone(reason)
        src = emit_futhark(ir, mode="auto")
        self.assertIn("entry eval_batch", src)
        self.assertIn("entry eval_batch_xor", src)
        self.assertNotIn("entry run_steps_batch", src)
        manifest = emit_futhark_manifest(ir, mode="auto")
        self.assertEqual(manifest["mode"], "combinational_fast")
        self.assertIn("eval_batch", manifest["entries"])
        self.assertIn("eval_batch_xor", manifest["entries"])
        metrics = compute_futhark_source_metrics(src)
        self.assertGreater(metrics["map2"], 0)
        self.assertGreater(metrics["map"], 0)

    def test_auto_mode_sequential_feedback_falls_back(self) -> None:
        t8 = BitVecType(width=8)
        ir = TickIR(
            name="seq_auto",
            inputs={"a": t8},
            outputs={"y": t8},
            state={"q": t8},
            reset_state={"q": BitVecConst(width=8, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={"y": Var("q")},
        )
        mode, reason = resolve_futhark_mode(ir, "auto")
        self.assertEqual(mode, "step_legacy")
        self.assertEqual(reason, "auto_fallback_sequential_feedback_in_next_state")
        manifest = emit_futhark_manifest(ir, mode="auto")
        self.assertEqual(manifest["mode"], "step_legacy")
        self.assertEqual(
            manifest["fallback_reason"],
            "auto_fallback_sequential_feedback_in_next_state",
        )
        self.assertIn("run_steps_batch", manifest["entries"])

    def test_deterministic_emit_and_manifest(self) -> None:
        t128 = BitVecType(width=128)
        ir = TickIR(
            name="det",
            inputs={"a": t128, "b": t128},
            outputs={"o": t128},
            state={"q": t128},
            reset_state={"q": BitVecConst(width=128, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={
                "o": Xor(
                    a=Shl(a=Var("q"), b=Var("b")),
                    b=LShr(a=Var("q"), b=Var("b")),
                )
            },
        )
        src1 = emit_futhark(ir, mode="step_legacy")
        src2 = emit_futhark(ir, mode="step_legacy")
        self.assertEqual(src1, src2)
        man1 = emit_futhark_manifest(ir, mode="step_legacy")
        man2 = emit_futhark_manifest(ir, mode="step_legacy")
        self.assertEqual(man1, man2)

    def test_associative_xor_tree_lifts_to_reduce(self) -> None:
        t8 = BitVecType(width=8)
        expr = Xor(
            a=Var("a"),
            b=Xor(a=Var("b"), b=Xor(a=Var("c"), b=Var("d"))),
        )
        ir = TickIR(
            name="xor_reduce",
            inputs={"a": t8, "b": t8, "c": t8, "d": t8},
            outputs={"y": t8},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": expr},
        )
        src = emit_futhark(ir, mode="combinational_fast")
        self.assertIn("reduce (\\p q -> p ^ q)", src)
        metrics = compute_futhark_source_metrics(src)
        self.assertGreater(metrics["reduce"], 0)

    def test_metrics_detect_plain_map_tokens(self) -> None:
        src = "\n".join(
            [
                "let a = map f xs",
                "let b = map2 g ys zs",
                "let c = map3 h p q r",
                "let d = reduce op z xs",
                "let e = scan op z xs",
            ]
        )
        metrics = compute_futhark_source_metrics(src)
        self.assertEqual(metrics["map"], 1)
        self.assertEqual(metrics["map2"], 1)
        self.assertEqual(metrics["map3"], 1)
        self.assertEqual(metrics["reduce"], 1)
        self.assertEqual(metrics["scan"], 1)

    def test_wide_add_sub_lifts_to_scan_helpers(self) -> None:
        t128 = BitVecType(width=128)
        ir = TickIR(
            name="scan_helpers",
            inputs={"a": t128, "b": t128},
            outputs={"sum": t128, "diff": t128},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "sum": Add(a=Var("a"), b=Var("b")),
                "diff": Sub(a=Var("a"), b=Var("b")),
            },
        )
        src = emit_futhark(ir, mode="combinational_fast")
        self.assertIn("def add_chunks", src)
        self.assertIn("def sub_chunks", src)
        self.assertNotIn("scan gp_compose_add", src)
        self.assertNotIn("scan gp_compose_sub", src)
        metrics = compute_futhark_source_metrics(src)
        self.assertGreater(metrics["map2"], 0)

    def test_wide_shift_helpers_emit(self) -> None:
        t128 = BitVecType(width=128)
        ir = TickIR(
            name="wide_shift",
            inputs={"a": t128, "sh": t128},
            outputs={"l": t128, "r": t128, "ar": t128},
            state={},
            reset_state={},
            next_state={},
            output_exprs={
                "l": Shl(a=Var("a"), b=Var("sh")),
                "r": LShr(a=Var("a"), b=Var("sh")),
                "ar": AShr(a=Var("a"), b=Var("sh")),
            },
        )
        src = emit_futhark(ir, mode="combinational_fast")
        self.assertIn("def shift_amt_chunks", src)
        self.assertIn("def shl_chunks", src)
        self.assertIn("def lshr_chunks", src)
        self.assertIn("def ashr_chunks", src)

    def test_wide_ult_lifts_to_reduce_helper(self) -> None:
        t128 = BitVecType(width=128)
        ir = TickIR(
            name="ult_reduce",
            inputs={"a": t128, "b": t128},
            outputs={"lt": BoolType()},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"lt": Ult(a=Var("a"), b=Var("b"))},
        )
        src = emit_futhark(ir, mode="combinational_fast")
        self.assertIn("def ult_chunks", src)

    def test_wide_no_pack_unpack_roundtrip_in_hot_path(self) -> None:
        t128 = BitVecType(width=128)
        ir = TickIR(
            name="no_roundtrip",
            inputs={"a": t128, "b": t128},
            outputs={"o": t128},
            state={"q": t128},
            reset_state={"q": BitVecConst(width=128, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={"o": Xor(a=Var("q"), b=Var("b"))},
        )
        src = emit_futhark(ir, mode="step_legacy")
        self.assertNotIn("map unpack_bits_w128", src)
        self.assertNotIn("map (\\x -> pack_bits_w128 x)", src)

    def test_rejects_simd_types(self) -> None:
        ir = TickIR(
            name="simd_bad",
            inputs={"x": SimdType(lane_width=8, lanes=16)},
            outputs={"y": SimdType(lane_width=8, lanes=16)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"y": Var("x")},
        )
        with self.assertRaises(CodegenError):
            emit_futhark(ir)

    def test_generated_source_typechecks_when_futhark_available(self) -> None:
        if shutil.which("futhark") is None:
            self.skipTest("futhark not installed")

        t8 = BitVecType(width=8)
        ir = TickIR(
            name="checkable",
            inputs={"a": t8, "b": t8, "sel": BoolType()},
            outputs={"y": t8, "nsel": BoolType()},
            state={"q": t8},
            reset_state={"q": BitVecConst(width=8, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={
                "y": Mux(cond=Var("sel"), a=Var("q"), b=Var("b")),
                "nsel": Not(x=Var("sel")),
            },
        )

        src = emit_futhark(ir)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "circuit.fut"
            path.write_text(src, encoding="utf-8")
            subprocess.run(
                ["futhark", "check", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_wide_source_typechecks_when_futhark_available(self) -> None:
        if shutil.which("futhark") is None:
            self.skipTest("futhark not installed")

        t8 = BitVecType(width=8)
        t128 = BitVecType(width=128)
        ir = TickIR(
            name="wide_checkable",
            inputs={"x": t128},
            outputs={"y": t8},
            state={"q": t128},
            reset_state={"q": BitVecConst(width=128, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("x"))},
            output_exprs={
                "y": Lut8(
                    x=Slice(x=Var("q"), offset=0, width=8),
                    table=list(range(256)),
                )
            },
        )

        src = emit_futhark(ir)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "wide_circuit.fut"
            path.write_text(src, encoding="utf-8")
            subprocess.run(
                ["futhark", "check", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
