import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_ptx import emit_ptx
from stc.tick_ir import Add, BitVecConst, BitVecType, TickIR, Var


@unittest.skipUnless(shutil.which("ptxas"), "requires ptxas")
class TestBackendPtxAssembleOptional(unittest.TestCase):
    def test_ptx_assembles_for_sm61(self) -> None:
        t = BitVecType(width=32)
        ir = TickIR(
            name="t",
            inputs={"a": t, "b": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Add(a=Var("a"), b=Var("b"))},
        )
        ptx = emit_ptx(ir, sm="sm_61")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            ptx_path = out_dir / "kernel.ptx"
            cubin = out_dir / "kernel.cubin"
            ptx_path.write_text(ptx, encoding="utf-8")
            subprocess.run(
                ["ptxas", "-arch=sm_61", "-o", str(cubin), str(ptx_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    def test_ptx_assembles_u64_for_sm61(self) -> None:
        t = BitVecType(width=64)
        ir = TickIR(
            name="t64",
            inputs={"a": t, "b": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Add(a=Var("a"), b=Var("b"))},
        )
        ptx = emit_ptx(ir, sm="sm_61")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            ptx_path = out_dir / "kernel64.ptx"
            cubin = out_dir / "kernel64.cubin"
            ptx_path.write_text(ptx, encoding="utf-8")
            subprocess.run(
                ["ptxas", "-arch=sm_61", "-o", str(cubin), str(ptx_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    def test_ptx_assembles_stateful_for_sm61(self) -> None:
        t = BitVecType(width=8)
        ir = TickIR(
            name="tstate",
            inputs={"a": t},
            outputs={"o": t},
            state={"q": t},
            reset_state={"q": BitVecConst(width=8, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={"o": Var("q")},
        )
        ptx = emit_ptx(ir, sm="sm_61")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            ptx_path = out_dir / "kernel_state.ptx"
            cubin = out_dir / "kernel_state.cubin"
            ptx_path.write_text(ptx, encoding="utf-8")
            subprocess.run(
                ["ptxas", "-arch=sm_61", "-o", str(cubin), str(ptx_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
