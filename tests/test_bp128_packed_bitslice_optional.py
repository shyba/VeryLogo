import random
import tempfile
import unittest
from pathlib import Path

from scripts.bp_circuit_sbox import build_bp_sbox
from stc.backend_sched import generate_scheduled_code
from stc.packed_bitslice import bitslice_circuit_to_packed
from stc.testing.circuitstate_eval import eval_circuitstate_bits
from stc.testing.native_x86_runner import (
    compile_shared,
    have_avx512,
    run_avx512_steps_shared,
)
from stc.tick_ir_to_circuit_state import PackedLayout


class TestBP128PackedBitsliceOptional(unittest.TestCase):
    def test_bp128_packed_bitslice_correctness(self) -> None:
        if not have_avx512():
            self.skipTest("AVX-512 not available")

        circuit = build_bp_sbox()
        layout = PackedLayout(
            inputs={f"in{i}": {"lsb": i, "width": 1} for i in range(8)},
            state={},
            outputs={f"out{i}": {"lsb": i, "width": 1} for i in range(8)},
            next_state={},
            input_bits=8,
            output_bits=8,
        )
        packed, packed_layout = bitslice_circuit_to_packed(circuit, layout)
        input_io_words, output_io_words = packed_layout.io_words()

        code = generate_scheduled_code(
            packed,
            target="avx512_u64",
            scheduler="list",
            io_split=(input_io_words, output_io_words),
        )

        with tempfile.TemporaryDirectory() as td:
            c_path = Path(td) / "circuit_avx512_u64.c"
            c_path.write_text(code, encoding="utf-8")
            build = compile_shared(
                c_path,
                cflags=["-mavx512f", "-mavx512vl", "-mavx512dq", "-mavx512bw"],
            )

            rng = random.Random(0)
            inputs = [rng.randrange(256) for _ in range(64)]

            in_words = []
            for bit in range(8):
                word = 0
                for lane, value in enumerate(inputs):
                    if (value >> bit) & 1:
                        word |= 1 << lane
                in_words.append(word)

            out_words, _ = run_avx512_steps_shared(
                build.so_path,
                in_words,
                [],
                input_io_bits=input_io_words,
                state_bits=0,
                output_io_bits=output_io_words,
                steps=1,
            )

            circuit_dict = {
                "input_bits": circuit.input_bits,
                "gates": circuit.gates,
                "outputs": circuit.outputs,
            }
            for lane, value in enumerate(inputs):
                out_byte = 0
                for bit in range(8):
                    if (out_words[bit] >> lane) & 1:
                        out_byte |= 1 << bit
                ref_bits = eval_circuitstate_bits(
                    circuit_dict, [(value >> i) & 1 for i in range(8)]
                )
                ref_byte = sum(int(b) << i for i, b in enumerate(ref_bits))
                self.assertEqual(out_byte, ref_byte)


if __name__ == "__main__":
    unittest.main()
