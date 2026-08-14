import json
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.backend_x86_gemm import emit_x86_gemm_c
from stc.cli import run_pipeline
from stc.extract import extract_tick_ir
from stc.gemm_lowering import expand_gemm_call
from stc.interp import eval_expr, infer_type
from stc.tick_ir import BitVecType, GemmCall, TickIR, Var
from stc.tick_ir_bin2 import read_tick_ir_bin, write_tick_ir_bin
from stc.tick_ir_to_verilog import emit_verilog
from stc.tick_ir_validate import validate_tick_ir
from stc.yosys_frontend import run_yosys
from stc.yosys_json import load_design


def _has_cpu_flag(flag: str) -> bool:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    aliases = {flag, flag.replace("avx512", "avx512_")}
    return any(
        line.startswith("flags") and aliases.intersection(line.split())
        for line in text.splitlines()
    )


def _verilog(m: int, n: int, k: int) -> str:
    return (
        f"module gemm #(parameter integer M={m}, parameter integer N={n}, parameter integer K={k}) "
        f"(input wire [M*K*8-1:0] a, input wire [K*N*8-1:0] b, "
        f"output wire [M*N*32-1:0] c); "
        "assign c = {M*N*32{1'b0}}; endmodule\n"
        f"module top #(parameter integer M={m}, parameter integer N={n}, parameter integer K={k}) "
        f"(input wire [M*K*8-1:0] a, input wire [K*N*8-1:0] b, "
        f"output wire [M*N*32-1:0] c); "
        "gemm #(.M(M),.N(N),.K(K)) u(.a(a),.b(b),.c(c)); endmodule\n"
    )


class TestGemmPipeline(unittest.TestCase):
    def test_interpreter_generic_expansion_and_binary_roundtrip(self) -> None:
        expr = GemmCall(Var("a"), Var("b"), m=2, n=2, k=2)
        ir = TickIR(
            name="gemm",
            inputs={"a": BitVecType(expr.a_total_width), "b": BitVecType(expr.b_total_width)},
            outputs={"c": BitVecType(expr.result_width)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"c": expr},
        )
        validate_tick_ir(ir)
        a = 1 | (2 << 8) | (3 << 16) | (4 << 24)
        b = 255 | (2 << 8) | (253 << 16) | (4 << 24)
        expected = eval_expr(expr, ir.inputs, {"a": a, "b": b})
        expanded = expand_gemm_call(expr)
        self.assertEqual(infer_type(expanded, ir.inputs), ir.outputs["c"])
        self.assertEqual(eval_expr(expanded, ir.inputs, {"a": a, "b": b}), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gemm.bin"
            write_tick_ir_bin(ir, str(path))
            self.assertEqual(read_tick_ir_bin(str(path)).to_dict(), ir.to_dict())
            if shutil.which("verilator"):
                verilog = Path(directory) / "gemm.sv"
                verilog.write_text(emit_verilog(ir), encoding="utf-8")
                subprocess.run(
                    ["verilator", "--lint-only", "--sv", str(verilog)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

    @unittest.skipUnless(shutil.which("yosys"), "requires yosys")
    def test_parameterized_hierarchy_lifts_to_shaped_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gemm.v"
            normalized = root / "normalized.json"
            source.write_text(_verilog(2, 16, 4), encoding="utf-8")
            run_yosys(source, normalized, top="top")
            ir = extract_tick_ir(load_design(normalized, top="top"))
        self.assertEqual(ir.inputs, {"a": BitVecType(64), "b": BitVecType(512)})
        self.assertEqual(ir.outputs, {"c": BitVecType(1024)})
        self.assertIsInstance(ir.output_exprs["c"], GemmCall)
        self.assertEqual((ir.output_exprs["c"].m, ir.output_exprs["c"].n, ir.output_exprs["c"].k), (2, 16, 4))

    def test_scalar_target_codegen_matches_matrix_reference(self) -> None:
        expr = GemmCall(Var("a"), Var("b"), m=2, n=2, k=2)
        ir = TickIR(
            name="gemm",
            inputs={"a": BitVecType(32), "b": BitVecType(32)},
            outputs={"c": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"c": expr},
        )
        generated = emit_x86_gemm_c(ir)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "impl.c").write_text(generated, encoding="utf-8")
            (root / "main.c").write_text(
                "#include <stdint.h>\n#include <stdio.h>\n"
                "void stc_eval(const uint64_t*, uint64_t*);\n"
                "int main(void) { uint64_t in[2] = {0}, out[2] = {0};\n"
                "  uint8_t* a=(uint8_t*)in; uint8_t* b=(uint8_t*)(in+1);\n"
                "  a[0]=1; a[1]=2; a[2]=3; a[3]=4;\n"
                "  b[0]=255; b[1]=2; b[2]=253; b[3]=4; stc_eval(in,out);\n"
                "  int32_t* c=(int32_t*)out;\n"
                "  return (c[0] == -7 && c[1] == 10 && c[2] == -15 && c[3] == 22) ? 0 : 1; }\n",
                encoding="utf-8",
            )
            executable = root / "runner"
            subprocess.run(
                ["cc", "-std=c99", "-O2", "-o", str(executable), str(root / "main.c"), str(root / "impl.c")],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run([str(executable)], check=True)

    @unittest.skipUnless(shutil.which("yosys"), "requires yosys")
    def test_cli_generic_backend_expands_bounded_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gemm.v"
            out_dir = root / "out"
            source.write_text(_verilog(2, 2, 2), encoding="utf-8")
            run_pipeline(source, out_dir, top="top", backend="generic", no_backend=True)
            reduced = read_tick_ir_bin(out_dir / "reduced_tick_ir.bin")
        self.assertNotIsInstance(reduced.output_exprs["c"], GemmCall)

    @unittest.skipUnless(
        shutil.which("yosys")
        and shutil.which("cc")
        and platform.machine() == "x86_64"
        and _has_cpu_flag("avx512f")
        and _has_cpu_flag("avx512vnni"),
        "requires yosys, cc, x86_64, avx512f, and avx512vnni",
    )
    def test_cli_pipeline_emits_and_runs_vnni_gemm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gemm.v"
            out_dir = root / "out"
            source.write_text(_verilog(2, 16, 4), encoding="utf-8")
            run_pipeline(source, out_dir, top="top", backend="x86-gemm")
            generated = (out_dir / "circuit_x86_gemm.c").read_text(encoding="utf-8")
            self.assertIn("_mm512_dpbusd_epi32", generated)
            (root / "main.c").write_text(
                "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                "void stc_eval(const uint64_t*, uint64_t*);\n"
                "int main(void) { uint64_t in[9]={0}, out[16]={0};\n"
                " uint8_t* a=(uint8_t*)in; int8_t* b=(int8_t*)(in+1);\n"
                " for(int i=0;i<8;i++) a[i]=(uint8_t)(i+1);\n"
                " for(int q=0;q<4;q++) for(int j=0;j<16;j++) b[q*16+j]=(int8_t)((j%5)-2+q);\n"
                " stc_eval(in,out); int32_t* c=(int32_t*)out;\n"
                " for(int i=0;i<2;i++) for(int j=0;j<16;j++){ int32_t r=0;\n"
                "  for(int q=0;q<4;q++) r += (int)a[i*4+q]*(int)b[q*16+j];\n"
                "  if(c[i*16+j]!=r) return 1; } return 0; }\n",
                encoding="utf-8",
            )
            executable = root / "runner"
            subprocess.run(
                ["cc", "-std=c99", "-O2", "-mavx512f", "-mavx512vnni", "-o", str(executable), str(root / "main.c"), str(out_dir / "circuit_x86_gemm.c")],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run([str(executable)], check=True)
            self.assertIsInstance(read_tick_ir_bin(out_dir / "tick_ir.bin").output_exprs["c"], GemmCall)
