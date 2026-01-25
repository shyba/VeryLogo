import os
import shutil
import tempfile
import unittest
from pathlib import Path

from stc.backend_sched import generate_scheduled_code
from stc.circuit_synth import CircuitState
from stc.testing import (
    compile_shared,
    eval_circuitstate_bits,
    have_avx2,
    have_avx512,
    run_avx2_circuit,
    run_avx512_circuit,
)


def _bitmask64(bit: int) -> int:
    return 0xFFFFFFFFFFFFFFFF if bit else 0


class TestNativeEmitOptional(unittest.TestCase):
    def test_avx2_avx512_match_python_on_muxy_parallel_circuit(self):
        if os.environ.get("STC_RUN_NATIVE", "") not in {"1", "true", "TRUE"}:
            self.skipTest("set STC_RUN_NATIVE=1 to enable")
        if shutil.which(os.environ.get("CC", "cc")) is None:
            self.skipTest("missing C compiler")

        # Mux-like structure expressed in ternary gates and wide fanout.
        input_bits = 16
        gates = []
        # build parallel intermediates
        for i in range(64):
            a = i % input_bits
            b = (i + 1) % input_bits
            c = (i + 2) % input_bits
            imm = 0xCA  # arbitrary ternary truth table
            gates.append(("ternary", a, b, c, imm))
        # combine many ways to raise pressure
        for i in range(64, 160):
            gates.append(("xor", input_bits + (i - 64), (i - 64) % input_bits))
        outputs = [(input_bits + len(gates) - 1, False)]

        circuit = CircuitState(
            input_bits=input_bits,
            output_bits=1,
            gates=gates,
            outputs=outputs,
            gate_count=len(gates),
        )
        cs = circuit.to_dict()

        # Single-lane (bit0) oracle via python CircuitState evaluator.
        packed_in_bits = [((i * 7) & 1) for i in range(input_bits)]
        py_out_bits = eval_circuitstate_bits(cs, packed_in_bits)
        py_out_bit = py_out_bits[0] & 1

        # Bitsliced mask inputs (64 lanes) for native execution.
        in_words = [_bitmask64(b) for b in packed_in_bits]

        if have_avx2():
            c_src = generate_scheduled_code(circuit, target="avx2", scheduler="list")
            with tempfile.TemporaryDirectory(prefix="stc_native_avx2_") as td:
                c_path = Path(td) / "circuit.c"
                c_path.write_text(c_src, encoding="utf-8")
                build = compile_shared(c_path, cflags=["-mavx2"])
                out_words = run_avx2_circuit(
                    build.so_path, in_words, input_bits=input_bits, output_bits=1
                )
                got = 1 if out_words[0] == 0xFFFFFFFFFFFFFFFF else 0
                self.assertEqual(got, py_out_bit)
        else:
            self.skipTest("CPU lacks AVX2")

        if have_avx512():
            c_src = generate_scheduled_code(circuit, target="avx512", scheduler="list")
            with tempfile.TemporaryDirectory(prefix="stc_native_avx512_") as td:
                c_path = Path(td) / "circuit.c"
                c_path.write_text(c_src, encoding="utf-8")
                build = compile_shared(
                    c_path,
                    cflags=["-mavx512f", "-mavx512vl", "-mavx512dq", "-mavx512bw"],
                )
                out_words = run_avx512_circuit(
                    build.so_path, in_words, input_bits=input_bits, output_bits=1
                )
                got = 1 if out_words[0] == 0xFFFFFFFFFFFFFFFF else 0
                self.assertEqual(got, py_out_bit)
        else:
            self.skipTest("CPU lacks AVX-512")

