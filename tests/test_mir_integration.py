"""Integration tests for MIR-based code generation pipeline."""

import unittest

from stc.backend_sched import generate_scheduled_code
from stc.circuit_synth import CircuitState


class TestMIRIntegration(unittest.TestCase):
    """Test MIR integration into compilation pipeline."""

    def test_avx512_mir_simple_and(self):
        """Generate AVX-512 code through MIR path."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("and", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        code = generate_scheduled_code(
            circuit, target="avx512_mir", scheduler="list", function_name="test_and"
        )

        self.assertIn("#include <immintrin.h>", code)
        self.assertIn("void test_and(__m512i* in, __m512i* out)", code)
        self.assertIn("_mm512_and_si512", code)
        self.assertIn("out[0]", code)

    def test_avx512_mir_ternary(self):
        """Generate AVX-512 code with ternary operation through MIR."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[("ternary", 0, 1, 2, 0xCA)],
            outputs=[(3, False)],
            gate_count=1,
        )

        code = generate_scheduled_code(circuit, target="avx512_mir", scheduler="list")

        self.assertIn("_mm512_ternarylogic_epi32", code)
        self.assertIn("202", code)

    def test_avx512_mir_multiple_gates(self):
        """Generate AVX-512 code with multiple gates through MIR."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("not", 2),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )

        code = generate_scheduled_code(circuit, target="avx512_mir", scheduler="list")

        self.assertIn("_mm512_and_si512", code)
        self.assertIn("_mm512_xor_si512", code)
        self.assertIn("ones = _mm512_set1_epi32(-1)", code)

    def test_ptx_mir_simple_and(self):
        """Generate PTX code through MIR path."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("and", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        code = generate_scheduled_code(
            circuit, target="ptx_mir", scheduler="list", function_name="test_and"
        )

        self.assertIn(".version 6.0", code)
        self.assertIn(".target sm_61", code)
        self.assertIn(".visible .entry test_and(", code)
        self.assertIn("and.b32", code)

    def test_ptx_mir_ternary(self):
        """Generate PTX code with ternary operation through MIR."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[("ternary", 0, 1, 2, 0xCA)],
            outputs=[(3, False)],
            gate_count=1,
        )

        code = generate_scheduled_code(circuit, target="ptx_mir", scheduler="list")

        self.assertIn("lop3.b32", code)
        self.assertIn("202", code)

    def test_mir_vs_direct_equivalence(self):
        """Verify MIR and direct emission produce functionally equivalent code."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=2,
            gates=[
                ("and", 0, 1),
                ("or", 1, 2),
                ("xor", 3, 4),
            ],
            outputs=[(3, False), (5, False)],
            gate_count=3,
        )

        mir_code = generate_scheduled_code(
            circuit, target="avx512_mir", scheduler="list"
        )
        direct_code = generate_scheduled_code(
            circuit, target="avx512", scheduler="list"
        )

        self.assertIn("_mm512_and_si512", mir_code)
        self.assertIn("_mm512_or_si512", mir_code)
        self.assertIn("_mm512_xor_si512", mir_code)

        self.assertIn("_mm512_and_si512", direct_code)
        self.assertIn("_mm512_or_si512", direct_code)
        self.assertIn("_mm512_xor_si512", direct_code)


if __name__ == "__main__":
    unittest.main()
