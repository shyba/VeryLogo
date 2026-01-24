import os
import unittest

from stc.bitslice import synthesize_bitslice_circuit, AES_SBOX_TABLE
from stc.circuit_synth import (
    synthesize_single_output,
    synthesize_single_output_cegar,
    synthesize_multi_output,
    synthesize_multi_output_shared,
    synthesize_multi_output_anf,
)
from stc.interp import eval_expr
from stc.tick_ir import BitVecType, SimdType


class TestCircuitSynthesis(unittest.TestCase):
    def test_synthesize_identity_bit(self) -> None:
        """Synthesize identity function for bit 0."""
        table = [(i >> 0) & 1 for i in range(256)]
        result = synthesize_single_output(
            table, input_bits=8, max_gates=10, timeout_ms=5000
        )
        self.assertIsNotNone(result)
        expr, gates = result
        self.assertEqual(gates, 0)

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            val = eval_expr(expr, types, {"x": i})
            self.assertEqual(val, (i >> 0) & 1)

    def test_synthesize_xor_two_bits(self) -> None:
        """Synthesize x[0] XOR x[1]."""
        table = [((i >> 0) ^ (i >> 1)) & 1 for i in range(256)]
        result = synthesize_single_output(
            table, input_bits=8, max_gates=10, timeout_ms=5000
        )
        self.assertIsNotNone(result)
        expr, gates = result
        self.assertEqual(gates, 1)

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            val = eval_expr(expr, types, {"x": i})
            expected = ((i >> 0) ^ (i >> 1)) & 1
            self.assertEqual(val, expected)

    def test_synthesize_and_two_bits(self) -> None:
        """Synthesize x[0] AND x[1]."""
        table = [((i >> 0) & (i >> 1)) & 1 for i in range(256)]
        result = synthesize_single_output(
            table, input_bits=8, max_gates=10, timeout_ms=5000
        )
        self.assertIsNotNone(result)
        expr, gates = result
        self.assertEqual(gates, 1)

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            val = eval_expr(expr, types, {"x": i})
            expected = ((i >> 0) & (i >> 1)) & 1
            self.assertEqual(val, expected)

    def test_synthesize_three_input_xor(self) -> None:
        """Synthesize x[0] XOR x[1] XOR x[2]."""
        table = [((i >> 0) ^ (i >> 1) ^ (i >> 2)) & 1 for i in range(256)]
        result = synthesize_single_output(
            table, input_bits=8, max_gates=10, timeout_ms=5000
        )
        self.assertIsNotNone(result)
        expr, gates = result
        self.assertEqual(gates, 2)

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            val = eval_expr(expr, types, {"x": i})
            expected = ((i >> 0) ^ (i >> 1) ^ (i >> 2)) & 1
            self.assertEqual(val, expected)

    def test_synthesize_majority_3bit(self) -> None:
        """Synthesize 3-input majority: (a&b)|(b&c)|(a&c) = (a&b)^(b&c)^(a&c)^(a&b&c)."""

        def majority(x: int) -> int:
            a, b, c = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1
            return 1 if (a + b + c) >= 2 else 0

        table = [majority(i) for i in range(256)]
        result = synthesize_single_output(
            table, input_bits=8, max_gates=20, timeout_ms=10000
        )
        self.assertIsNotNone(result)
        expr, gates = result
        self.assertLessEqual(gates, 5)

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            val = eval_expr(expr, types, {"x": i})
            self.assertEqual(val, majority(i))

    def test_synthesize_4bit_to_4bit_identity(self) -> None:
        """Synthesize 4-bit identity function."""
        table = list(range(16))
        result = synthesize_multi_output(
            table, input_bits=4, output_bits=4, max_gates=10, timeout_ms=5000
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.gate_count, 0)

        types = {"x": BitVecType(width=4)}
        for i in range(16):
            total = 0
            for bit in range(4):
                val = eval_expr(result.circuit[bit], types, {"x": i})
                total |= val << bit
            self.assertEqual(total, i)

    def test_synthesize_4bit_increment(self) -> None:
        """Synthesize 4-bit increment (x + 1) mod 16."""
        table = [(i + 1) % 16 for i in range(16)]
        result = synthesize_multi_output(
            table, input_bits=4, output_bits=4, max_gates=20, timeout_ms=10000
        )
        self.assertIsNotNone(result)

        types = {"x": BitVecType(width=4)}
        for i in range(16):
            total = 0
            for bit in range(4):
                val = eval_expr(result.circuit[bit], types, {"x": i})
                total |= val << bit
            self.assertEqual(total, (i + 1) % 16, f"failed for input {i}")


class TestCircuitSynthesisSmallSbox(unittest.TestCase):
    def test_synthesize_4bit_sbox(self) -> None:
        """Synthesize a simple 4-bit S-box."""
        sbox = [
            0xC,
            0x5,
            0x6,
            0xB,
            0x9,
            0x0,
            0xA,
            0xD,
            0x3,
            0xE,
            0xF,
            0x8,
            0x4,
            0x7,
            0x1,
            0x2,
        ]
        result = synthesize_multi_output(
            sbox, input_bits=4, output_bits=4, max_gates=60, timeout_ms=60000
        )
        self.assertIsNotNone(result)

        types = {"x": BitVecType(width=4)}
        for i in range(16):
            total = 0
            for bit in range(4):
                val = eval_expr(result.circuit[bit], types, {"x": i})
                total |= val << bit
            self.assertEqual(
                total, sbox[i], f"failed for input {i}: got {total}, expected {sbox[i]}"
            )

        print(
            f"4-bit S-box synthesized with {result.gate_count} gates ({result.xor_count} XOR, {result.and_count} AND)"
        )


class TestSharedGateSynthesis(unittest.TestCase):
    def test_shared_4bit_increment(self) -> None:
        """Synthesize 4-bit increment with shared gates."""
        table = [(i + 1) % 16 for i in range(16)]
        result = synthesize_multi_output_shared(
            table, input_bits=4, output_bits=4, max_gates=20, timeout_ms=30000
        )
        self.assertIsNotNone(result)

        types = {"x": BitVecType(width=4)}
        for i in range(16):
            total = 0
            for bit in range(4):
                val = eval_expr(result.circuit[bit], types, {"x": i})
                total |= val << bit
            self.assertEqual(total, (i + 1) % 16, f"failed for input {i}")

        print(f"4-bit increment (shared): {result.gate_count} gates")

    @unittest.skipUnless(
        os.environ.get("STC_RUN_SYNTH_TESTS") == "1",
        "slow test - requires STC_RUN_SYNTH_TESTS=1",
    )
    def test_shared_4bit_sbox(self) -> None:
        """Synthesize 4-bit S-box with shared gates."""
        sbox = [
            0xC,
            0x5,
            0x6,
            0xB,
            0x9,
            0x0,
            0xA,
            0xD,
            0x3,
            0xE,
            0xF,
            0x8,
            0x4,
            0x7,
            0x1,
            0x2,
        ]
        result = synthesize_multi_output_shared(
            sbox,
            input_bits=4,
            output_bits=4,
            max_gates=30,
            timeout_ms=600000,
            min_gates=20,
        )
        self.assertIsNotNone(result)

        types = {"x": BitVecType(width=4)}
        for i in range(16):
            total = 0
            for bit in range(4):
                val = eval_expr(result.circuit[bit], types, {"x": i})
                total |= val << bit
            self.assertEqual(total, sbox[i], f"failed for input {i}")

        print(
            f"4-bit S-box (shared): {result.gate_count} gates ({result.xor_count} XOR, {result.and_count} AND)"
        )


@unittest.skipUnless(
    os.environ.get("STC_RUN_SYNTH_TESTS") == "1",
    "requires STC_RUN_SYNTH_TESTS=1 (long running)",
)
class TestAesSboxSynthesis(unittest.TestCase):
    def test_aes_sbox_shared_synthesis(self) -> None:
        """Synthesize AES S-box with shared gates. Goal: <1000 gates."""
        result = synthesize_multi_output_shared(
            AES_SBOX_TABLE,
            input_bits=8,
            output_bits=8,
            max_gates=1000,
            timeout_ms=3600000,
            min_gates=100,
        )
        self.assertIsNotNone(result, "Failed to synthesize AES S-box with <1000 gates")

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            total = 0
            for bit in range(8):
                val = eval_expr(result.circuit[bit], types, {"x": i})
                total |= val << bit
            self.assertEqual(total, AES_SBOX_TABLE[i], f"S-box mismatch at {i}")

        print(
            f"AES S-box synthesized: {result.gate_count} gates ({result.xor_count} XOR, {result.and_count} AND)"
        )
        self.assertLess(result.gate_count, 1000)


class TestCegarSynthesis(unittest.TestCase):
    def test_cegar_xor_chain(self) -> None:
        """Test CEGAR synthesis on XOR chain function."""
        table = [((i >> 0) ^ (i >> 1) ^ (i >> 2) ^ (i >> 3)) & 1 for i in range(256)]
        result = synthesize_single_output_cegar(
            table, input_bits=8, max_gates=10, timeout_ms_per_round=5000, max_rounds=20
        )
        self.assertIsNotNone(result)
        expr, gates = result
        self.assertEqual(gates, 3)

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            val = eval_expr(expr, types, {"x": i})
            expected = table[i]
            self.assertEqual(val, expected)

    def test_cegar_majority(self) -> None:
        """Test CEGAR synthesis on majority function."""

        def majority(x: int) -> int:
            a, b, c = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1
            return 1 if (a + b + c) >= 2 else 0

        table = [majority(i) for i in range(256)]
        result = synthesize_single_output_cegar(
            table, input_bits=8, max_gates=10, timeout_ms_per_round=5000, max_rounds=20
        )
        self.assertIsNotNone(result)
        expr, gates = result
        self.assertLessEqual(gates, 5)

        types = {"x": BitVecType(width=8)}
        for i in range(256):
            val = eval_expr(expr, types, {"x": i})
            self.assertEqual(val, majority(i))


def _simple_eval_circuit(expr, x):
    """Simple recursive evaluation for testing ANF circuits."""
    from stc.tick_ir import BitVecConst, Slice, Xor, And, Var

    if isinstance(expr, BitVecConst):
        return expr.value
    elif isinstance(expr, Slice) and isinstance(expr.x, Var):
        return (x >> expr.offset) & ((1 << expr.width) - 1)
    elif isinstance(expr, Xor):
        return _simple_eval_circuit(expr.a, x) ^ _simple_eval_circuit(expr.b, x)
    elif isinstance(expr, And):
        return _simple_eval_circuit(expr.a, x) & _simple_eval_circuit(expr.b, x)
    else:
        raise ValueError(f"Unknown expr type: {type(expr)}")


class TestAnfSynthesis(unittest.TestCase):
    def test_anf_4bit_sbox(self) -> None:
        """Test ANF synthesis on 4-bit S-box."""
        import sys

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        try:
            sbox = [
                0xC,
                0x5,
                0x6,
                0xB,
                0x9,
                0x0,
                0xA,
                0xD,
                0x3,
                0xE,
                0xF,
                0x8,
                0x4,
                0x7,
                0x1,
                0x2,
            ]
            result = synthesize_multi_output_anf(sbox, input_bits=4, output_bits=4)

            for i in range(16):
                total = 0
                for bit in range(4):
                    val = _simple_eval_circuit(result.circuit[bit], i)
                    total |= val << bit
                self.assertEqual(total, sbox[i], f"failed for input {i}")

            print(
                f"4-bit S-box (ANF): {result.gate_count} gates ({result.xor_count} XOR, {result.and_count} AND)"
            )
        finally:
            sys.setrecursionlimit(old_limit)

    def test_anf_aes_sbox(self) -> None:
        """Test ANF synthesis on AES S-box - produces working circuit."""
        import sys

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        try:
            result = synthesize_multi_output_anf(
                AES_SBOX_TABLE, input_bits=8, output_bits=8
            )

            for i in range(256):
                total = 0
                for bit in range(8):
                    val = _simple_eval_circuit(result.circuit[bit], i)
                    total |= val << bit
                self.assertEqual(total, AES_SBOX_TABLE[i], f"failed for input {i}")

            self.assertLess(result.gate_count, 2000)
            print(
                f"AES S-box (ANF): {result.gate_count} gates ({result.xor_count} XOR, {result.and_count} AND)"
            )
        finally:
            sys.setrecursionlimit(old_limit)


class TestIncrementalOptimizer(unittest.TestCase):
    def test_anf_circuit_correctness(self) -> None:
        """Test that ANF-synthesized circuit computes correct S-box values."""
        import sys

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        try:
            from stc.circuit_synth import IncrementalOptimizer

            opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)

            for i in range(256):
                got = opt.best_state.evaluate(i)
                self.assertEqual(got, AES_SBOX_TABLE[i], f"S-box mismatch at input {i}")

            self.assertTrue(opt.verify())
            self.assertLess(opt.get_gate_count(), 1200)

            print(f"IncrementalOptimizer AES S-box: {opt.get_gate_count()} gates")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_save_load_roundtrip(self) -> None:
        """Test that circuit can be saved and loaded correctly."""
        import os
        import sys
        import tempfile

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        try:
            from stc.circuit_synth import IncrementalOptimizer

            opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
            original_gates = opt.get_gate_count()

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                temp_path = f.name

            try:
                opt.save(temp_path)
                loaded = IncrementalOptimizer.load(temp_path)

                self.assertEqual(loaded.get_gate_count(), original_gates)
                self.assertTrue(loaded.verify())

                for i in range(256):
                    self.assertEqual(loaded.best_state.evaluate(i), AES_SBOX_TABLE[i])
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        finally:
            sys.setrecursionlimit(old_limit)


class TestSynthesizedBitslice(unittest.TestCase):
    def test_synthesize_simple_sbox_bitslice(self) -> None:
        """Synthesize bitslice circuit for a simple permutation."""
        table = [(i ^ 0x5A) for i in range(256)]

        result = synthesize_bitslice_circuit(
            table, num_bytes=16, max_gates_per_bit=10, timeout_ms_per_bit=5000
        )
        self.assertIsNotNone(result)

        types = {"x": SimdType(lane_width=8, lanes=16)}
        test_inputs = [
            0x000102030405060708090A0B0C0D0E0F,
            0xFFEEDDCCBBAA99887766554433221100,
        ]

        for inp in test_inputs:
            output = eval_expr(result, types, {"x": inp})
            for i in range(16):
                in_byte = (inp >> (i * 8)) & 0xFF
                out_byte = (output >> (i * 8)) & 0xFF
                expected = table[in_byte]
                self.assertEqual(
                    out_byte, expected, f"byte {i}: got {out_byte}, expected {expected}"
                )


class TestCircuitStateMetrics(unittest.TestCase):
    def test_and_count(self) -> None:
        """Test and_count property."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("xor", 0, 1),
                ("and", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )
        self.assertEqual(state.and_count, 2)

    def test_xor_count(self) -> None:
        """Test xor_count property."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 0, 2),
                ("and", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )
        self.assertEqual(state.xor_count, 2)

    def test_depth_simple_chain(self) -> None:
        """Test depth property with a simple gate chain."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 2),
                ("xor", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )
        self.assertEqual(state.depth, 3)

    def test_depth_parallel(self) -> None:
        """Test depth with parallel gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=4,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 3),
                ("and", 4, 5),
            ],
            outputs=[(6, False)],
            gate_count=3,
        )
        self.assertEqual(state.depth, 2)

    def test_multiplicative_depth_xor_only(self) -> None:
        """Test multiplicative_depth with only XOR gates (should be 0)."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        self.assertEqual(state.multiplicative_depth, 0)

    def test_multiplicative_depth_and_chain(self) -> None:
        """Test multiplicative_depth with AND chain."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("and", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        self.assertEqual(state.multiplicative_depth, 2)

    def test_multiplicative_depth_mixed(self) -> None:
        """Test multiplicative_depth with mixed AND/XOR gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 3),
                ("xor", 3, 4),
                ("and", 4, 5),
            ],
            outputs=[(6, False)],
            gate_count=4,
        )
        self.assertEqual(state.multiplicative_depth, 2)

    def test_empty_circuit(self) -> None:
        """Test metrics on circuit with no gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[],
            outputs=[(0, False)],
            gate_count=0,
        )
        self.assertEqual(state.and_count, 0)
        self.assertEqual(state.xor_count, 0)
        self.assertEqual(state.depth, 0)
        self.assertEqual(state.multiplicative_depth, 0)

    def test_not_gate_depth(self) -> None:
        """Test that NOT gates contribute to depth but not multiplicative depth."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=1,
            output_bits=1,
            gates=[
                ("not", 0, 0),
                ("not", 1, 0),
            ],
            outputs=[(2, False)],
            gate_count=2,
        )
        self.assertEqual(state.depth, 2)
        self.assertEqual(state.multiplicative_depth, 0)


class TestDeadCodeElimination(unittest.TestCase):
    def test_dce_removes_unused(self) -> None:
        """Create circuit with gates not connected to output, verify removed."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 2),
                ("xor", 1, 2),
            ],
            outputs=[(3, False)],
            gate_count=3,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 1)
        self.assertEqual(len(cleaned.gates), 1)
        self.assertEqual(cleaned.gates[0], ("xor", 0, 1))
        self.assertEqual(cleaned.outputs, [(3, False)])

    def test_dce_preserves_outputs(self) -> None:
        """Verify evaluate() gives same results after DCE."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=4,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 3),
                ("xor", 0, 2),
                ("and", 4, 5),
            ],
            outputs=[(4, False), (7, True)],
            gate_count=4,
        )
        cleaned = state.eliminate_dead_code()

        for i in range(16):
            original_result = state.evaluate(i)
            cleaned_result = cleaned.evaluate(i)
            self.assertEqual(
                original_result,
                cleaned_result,
                f"Mismatch at input {i}: original={original_result}, cleaned={cleaned_result}",
            )

    def test_dce_chains(self) -> None:
        """Verify it handles chains of unused gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 2),
                ("xor", 2, 3),
                ("and", 3, 4),
                ("xor", 0, 1),
            ],
            outputs=[(6, False)],
            gate_count=5,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 1)
        self.assertEqual(len(cleaned.gates), 1)
        self.assertEqual(cleaned.gates[0], ("xor", 0, 1))

        for i in range(4):
            original_result = state.evaluate(i)
            cleaned_result = cleaned.evaluate(i)
            self.assertEqual(original_result, cleaned_result)

    def test_dce_all_gates_used(self) -> None:
        """Verify no change when all gates are used."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 2),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 2)
        self.assertEqual(len(cleaned.gates), 2)

    def test_dce_no_gates(self) -> None:
        """Verify DCE handles circuit with no gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[],
            outputs=[(0, False)],
            gate_count=0,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 0)
        self.assertEqual(len(cleaned.gates), 0)

    def test_dce_multiple_outputs_shared_cone(self) -> None:
        """Verify DCE preserves gates shared by multiple outputs."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 2),
                ("xor", 1, 2),
                ("or", 0, 1),
            ],
            outputs=[(3, False), (4, False)],
            gate_count=4,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 3)
        self.assertIn(("xor", 0, 1), cleaned.gates)

        for i in range(4):
            original_result = state.evaluate(i)
            cleaned_result = cleaned.evaluate(i)
            self.assertEqual(original_result, cleaned_result)

    def test_dce_shared_intermediate(self) -> None:
        """Verify DCE preserves intermediate gates used by multiple outputs."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 2),
                ("xor", 1, 2),
            ],
            outputs=[(3, False), (4, False)],
            gate_count=3,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 3)
        self.assertIn(("xor", 0, 1), cleaned.gates)

        for i in range(4):
            original_result = state.evaluate(i)
            cleaned_result = cleaned.evaluate(i)
            self.assertEqual(original_result, cleaned_result)

    def test_dce_const_gate(self) -> None:
        """Verify DCE handles const gates correctly."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("const", 1, 1),
                ("xor", 0, 1),
                ("and", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 3)

        for i in range(4):
            original_result = state.evaluate(i)
            cleaned_result = cleaned.evaluate(i)
            self.assertEqual(original_result, cleaned_result)

    def test_dce_not_gate(self) -> None:
        """Verify DCE handles not gates correctly."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("not", 0, 0),
                ("xor", 0, 1),
                ("and", 2, 3),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 3)

        for i in range(4):
            original_result = state.evaluate(i)
            cleaned_result = cleaned.evaluate(i)
            self.assertEqual(original_result, cleaned_result)

    def test_dce_removes_deep_unused_chain(self) -> None:
        """Verify DCE removes a long chain of unused gates."""
        from stc.circuit_synth import CircuitState

        gates = [
            ("xor", 0, 1),
            ("and", 0, 2),
            ("xor", 2, 3),
            ("and", 3, 4),
            ("xor", 4, 5),
            ("and", 5, 6),
            ("xor", 0, 1),
        ]
        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=gates,
            outputs=[(8, False)],
            gate_count=7,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 1)
        self.assertEqual(cleaned.gates[0], ("xor", 0, 1))

    def test_dce_input_as_output(self) -> None:
        """Verify DCE handles input directly used as output."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
            ],
            outputs=[(0, False)],
            gate_count=1,
        )
        cleaned = state.eliminate_dead_code()
        self.assertEqual(cleaned.gate_count, 0)
        self.assertEqual(len(cleaned.gates), 0)
        self.assertEqual(cleaned.outputs, [(0, False)])


class TestSlpExport(unittest.TestCase):
    def test_slp_xor_and_circuit(self) -> None:
        """Test SLP export with XOR and AND gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 0, 2),
            ],
            outputs=[(2, False), (3, True)],
            gate_count=2,
        )
        slp = state.to_slp()
        lines = slp.split("\n")
        self.assertEqual(lines[0], "t0 = x0 ^ x1")
        self.assertEqual(lines[1], "t1 = x0 & t0")
        self.assertEqual(lines[2], "y0 = t0")
        self.assertEqual(lines[3], "y1 = ~t1")

    def test_slp_all_ops(self) -> None:
        """Test SLP export with all operation types."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=4,
            gates=[
                ("xor", 0, 1),
                ("and", 1, 2),
                ("or", 0, 2),
                ("not", 3, 0),
                ("const", 1, 1),
            ],
            outputs=[(3, False), (4, False), (5, True), (7, False)],
            gate_count=5,
        )
        slp = state.to_slp()
        lines = slp.split("\n")
        self.assertEqual(lines[0], "t0 = x0 ^ x1")
        self.assertEqual(lines[1], "t1 = x1 & x2")
        self.assertEqual(lines[2], "t2 = x0 | x2")
        self.assertEqual(lines[3], "t3 = ~t0")
        self.assertEqual(lines[4], "t4 = 1")
        self.assertEqual(lines[5], "y0 = t0")
        self.assertEqual(lines[6], "y1 = t1")
        self.assertEqual(lines[7], "y2 = ~t2")
        self.assertEqual(lines[8], "y3 = t4")

    def test_slp_input_as_output(self) -> None:
        """Test SLP export when input is directly used as output."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[],
            outputs=[(0, False), (1, True)],
            gate_count=0,
        )
        slp = state.to_slp()
        lines = slp.split("\n")
        self.assertEqual(lines[0], "y0 = x0")
        self.assertEqual(lines[1], "y1 = ~x1")

    def test_slp_empty_circuit(self) -> None:
        """Test SLP export with empty circuit (no gates, no outputs)."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=0,
            gates=[],
            outputs=[],
            gate_count=0,
        )
        slp = state.to_slp()
        self.assertEqual(slp, "")

    def test_slp_const_zero(self) -> None:
        """Test SLP export with constant zero gate."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=1,
            output_bits=1,
            gates=[
                ("const", 0, 1),
            ],
            outputs=[(1, False)],
            gate_count=1,
        )
        slp = state.to_slp()
        lines = slp.split("\n")
        self.assertEqual(lines[0], "t0 = 0")
        self.assertEqual(lines[1], "y0 = t0")


class TestAlgebraicRewrites(unittest.TestCase):
    def test_rewrite_xor_self(self) -> None:
        """Test x ^ x -> 0."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 0),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            self.assertEqual(rewritten.evaluate(i), 0)

        self.assertLess(rewritten.gate_count, state.gate_count + 1)

    def test_rewrite_xor_zero(self) -> None:
        """Test x ^ 0 -> x."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("const", 0, 1),
                ("xor", 0, 2),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            expected = i & 1
            self.assertEqual(rewritten.evaluate(i), expected)

        self.assertLessEqual(rewritten.gate_count, 1)

    def test_rewrite_and_self(self) -> None:
        """Test x & x -> x."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 0),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            expected = i & 1
            self.assertEqual(rewritten.evaluate(i), expected)

        self.assertEqual(rewritten.gate_count, 0)

    def test_rewrite_preserves_correctness(self) -> None:
        """Test that rewrites preserve circuit semantics."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 3),
                ("and", 3, 3),
                ("xor", 4, 4),
                ("const", 0, 1),
                ("xor", 0, 7),
            ],
            outputs=[(5, False), (8, False)],
            gate_count=6,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(8):
            original_result = state.evaluate(i)
            rewritten_result = rewritten.evaluate(i)
            self.assertEqual(
                original_result,
                rewritten_result,
                f"Mismatch at input {i}: original={original_result}, rewritten={rewritten_result}",
            )

        self.assertLessEqual(rewritten.gate_count, state.gate_count)

    def test_rewrite_xor_cancel_left(self) -> None:
        """Test (a ^ b) ^ b -> a."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 1),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            a = i & 1
            self.assertEqual(rewritten.evaluate(i), a)

        self.assertLess(rewritten.gate_count, state.gate_count)

    def test_rewrite_xor_cancel_right(self) -> None:
        """Test (a ^ b) ^ a -> b."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            b = (i >> 1) & 1
            self.assertEqual(rewritten.evaluate(i), b)

        self.assertLess(rewritten.gate_count, state.gate_count)

    def test_rewrite_and_zero(self) -> None:
        """Test x & 0 -> 0."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("const", 0, 1),
                ("and", 0, 2),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            self.assertEqual(rewritten.evaluate(i), 0)

    def test_rewrite_and_one(self) -> None:
        """Test x & 1 -> x."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("const", 1, 1),
                ("and", 0, 2),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            expected = i & 1
            self.assertEqual(rewritten.evaluate(i), expected)

        self.assertLessEqual(rewritten.gate_count, 1)

    def test_rewrite_chain(self) -> None:
        """Test that chained rewrites work correctly."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 1),
                ("xor", 3, 3),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )
        rewritten = state.apply_algebraic_rewrites()

        for i in range(4):
            self.assertEqual(rewritten.evaluate(i), 0)

        self.assertLess(rewritten.gate_count, state.gate_count)


class TestCommonSubexpressionElimination(unittest.TestCase):
    def test_cse_removes_duplicates(self) -> None:
        """CSE should merge identical gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("xor", 0, 1),
                ("and", 0, 2),
                ("and", 0, 2),
            ],
            outputs=[(3, False), (5, False)],
            gate_count=4,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 2)
        self.assertEqual(len(optimized.gates), 2)
        self.assertIn(("xor", 0, 1), optimized.gates)
        self.assertIn(("and", 0, 2), optimized.gates)

    def test_cse_preserves_correctness(self) -> None:
        """Verify evaluate() gives same results before/after CSE."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=3,
            gates=[
                ("xor", 0, 1),
                ("xor", 1, 0),
                ("and", 1, 2),
                ("and", 2, 1),
                ("or", 0, 2),
                ("or", 2, 0),
            ],
            outputs=[(3, False), (5, False), (7, True)],
            gate_count=6,
        )
        optimized = state.eliminate_common_subexpressions()

        for i in range(8):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(
                original_result,
                optimized_result,
                f"Mismatch at input {i}: original={original_result}, optimized={optimized_result}",
            )

        self.assertLess(optimized.gate_count, state.gate_count)

    def test_cse_normalizes_commutative_ops(self) -> None:
        """CSE should normalize commutative ops: (and, 5, 3) -> (and, 3, 5)."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=4,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("xor", 1, 0),
                ("and", 2, 3),
                ("and", 3, 2),
            ],
            outputs=[(4, False), (6, False)],
            gate_count=4,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 2)
        self.assertEqual(len(optimized.gates), 2)

        for i in range(16):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

    def test_cse_handles_chains(self) -> None:
        """CSE should handle chains of duplicate gates correctly."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("xor", 0, 1),
                ("and", 2, 0),
                ("and", 3, 0),
            ],
            outputs=[(4, False), (5, False)],
            gate_count=4,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 2)

        for i in range(4):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

    def test_cse_no_duplicates(self) -> None:
        """CSE should not change circuit with no duplicates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 1, 2),
            ],
            outputs=[(3, False), (4, True)],
            gate_count=2,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, state.gate_count)

        for i in range(8):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

    def test_cse_handles_not_gates(self) -> None:
        """CSE should handle NOT gates correctly."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("not", 0, 0),
                ("not", 0, 0),
                ("xor", 2, 1),
                ("xor", 3, 1),
            ],
            outputs=[(4, False), (5, False)],
            gate_count=4,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 2)

        for i in range(4):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

    def test_cse_handles_const_gates(self) -> None:
        """CSE should handle const gates correctly."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("const", 1, 1),
                ("const", 1, 1),
                ("xor", 2, 0),
                ("xor", 3, 0),
            ],
            outputs=[(4, False), (5, False)],
            gate_count=4,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 2)

        for i in range(4):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

    def test_cse_removes_unreferenced_gates(self) -> None:
        """CSE should remove gates that become unreferenced after merging."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 0, 1),
                ("and", 2, 2),
            ],
            outputs=[(4, False)],
            gate_count=3,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 2)

        for i in range(4):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

    def test_cse_preserves_output_inversions(self) -> None:
        """CSE should preserve output inversions."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("xor", 0, 1),
            ],
            outputs=[(2, False), (3, True)],
            gate_count=2,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 1)

        for i in range(4):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

    def test_cse_empty_circuit(self) -> None:
        """CSE should handle empty circuit."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[],
            outputs=[(0, False)],
            gate_count=0,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 0)
        self.assertEqual(optimized.outputs, [(0, False)])

    def test_cse_input_as_output(self) -> None:
        """CSE should handle input directly used as output."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
            ],
            outputs=[(0, False), (2, True)],
            gate_count=1,
        )
        optimized = state.eliminate_common_subexpressions()
        self.assertEqual(optimized.gate_count, 1)

        for i in range(4):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)


class TestXorTreeFlattening(unittest.TestCase):
    def test_flatten_simple_chain(self) -> None:
        """Test x ^ (y ^ z) -> balanced tree."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 1, 2),
                ("xor", 0, 3),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        flattened = state.flatten_xor_trees()

        for i in range(8):
            original_result = state.evaluate(i)
            flattened_result = flattened.evaluate(i)
            self.assertEqual(
                original_result,
                flattened_result,
                f"Mismatch at input {i}: original={original_result}, flattened={flattened_result}",
            )

        self.assertLessEqual(flattened.depth, state.depth)

    def test_flatten_with_cancellation(self) -> None:
        """Test x ^ (y ^ (x ^ z)) -> y ^ z (x cancels)."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=4,
            output_bits=1,
            gates=[
                ("xor", 0, 3),
                ("xor", 1, 4),
                ("xor", 0, 5),
            ],
            outputs=[(6, False)],
            gate_count=3,
        )

        for i in range(16):
            x0 = (i >> 0) & 1
            x1 = (i >> 1) & 1
            x3 = (i >> 3) & 1
            expected = x1 ^ x3
            actual = state.evaluate(i)
            self.assertEqual(
                actual,
                x0 ^ x1 ^ x0 ^ x3,
                f"Input {i}: expected x0^x1^x0^x3={x0 ^ x1 ^ x0 ^ x3}",
            )

        flattened = state.flatten_xor_trees()

        for i in range(16):
            original_result = state.evaluate(i)
            flattened_result = flattened.evaluate(i)
            self.assertEqual(
                original_result,
                flattened_result,
                f"Mismatch at input {i}: original={original_result}, flattened={flattened_result}",
            )

        self.assertLess(
            flattened.gate_count,
            state.gate_count,
            f"Expected fewer gates due to x^x cancellation, got {flattened.gate_count} vs {state.gate_count}",
        )

    def test_flatten_preserves_and_gates(self) -> None:
        """Test that flattening does not flatten through AND gates."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 3),
                ("xor", 3, 4),
            ],
            outputs=[(5, False)],
            gate_count=3,
        )
        flattened = state.flatten_xor_trees()

        for i in range(8):
            original_result = state.evaluate(i)
            flattened_result = flattened.evaluate(i)
            self.assertEqual(
                original_result,
                flattened_result,
                f"Mismatch at input {i}: original={original_result}, flattened={flattened_result}",
            )

        and_count = sum(1 for op, _, _ in flattened.gates if op == "and")
        self.assertEqual(and_count, 1, "AND gate should be preserved")

    def test_flatten_no_change_already_flat(self) -> None:
        """Test that already flat tree returns equivalent circuit."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )
        flattened = state.flatten_xor_trees()

        for i in range(4):
            original_result = state.evaluate(i)
            flattened_result = flattened.evaluate(i)
            self.assertEqual(original_result, flattened_result)

        self.assertEqual(flattened.gate_count, 1)

    def test_flatten_aes_sbox(self) -> None:
        """Verify correctness on AES S-box after flattening."""
        from stc.circuit_synth import (
            CircuitState,
            IncrementalOptimizer,
            circuit_to_state,
            synthesize_multi_output_anf,
        )

        result = synthesize_multi_output_anf(
            AES_SBOX_TABLE, input_bits=8, output_bits=8
        )
        state = circuit_to_state(result.circuit, input_bits=8, output_bits=8)

        flattened = state.flatten_xor_trees()

        for i in range(256):
            expected = AES_SBOX_TABLE[i]
            original_result = state.evaluate(i)
            flattened_result = flattened.evaluate(i)
            self.assertEqual(
                original_result,
                expected,
                f"Original mismatch at {i}: got {original_result}, expected {expected}",
            )
            self.assertEqual(
                flattened_result,
                expected,
                f"Flattened mismatch at {i}: got {flattened_result}, expected {expected}",
            )

        self.assertLessEqual(flattened.gate_count, state.gate_count)


class TestLocalRewrites(unittest.TestCase):
    def test_local_rewrite_xor_cancel(self) -> None:
        """Test (a ^ b) ^ b -> a."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 1),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        rewritten = state.try_local_rewrites()

        for i in range(4):
            a = i & 1
            self.assertEqual(rewritten.evaluate(i), a)

        self.assertLess(rewritten.gate_count, state.gate_count)

    def test_local_rewrite_and_identity(self) -> None:
        """Test (a & b) & a -> a & b."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 1),
                ("and", 2, 0),
            ],
            outputs=[(3, False)],
            gate_count=2,
        )
        rewritten = state.try_local_rewrites()

        for i in range(4):
            a = i & 1
            b = (i >> 1) & 1
            expected = a & b
            self.assertEqual(rewritten.evaluate(i), expected)

        self.assertLessEqual(rewritten.gate_count, state.gate_count)

    def test_local_rewrite_nested_xor(self) -> None:
        """Test ((a ^ b) ^ c) ^ b -> a ^ c."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 3, 2),
                ("xor", 4, 1),
            ],
            outputs=[(5, False)],
            gate_count=3,
        )
        rewritten = state.try_local_rewrites()

        for i in range(8):
            a = i & 1
            b = (i >> 1) & 1
            c = (i >> 2) & 1
            expected = a ^ c
            self.assertEqual(
                rewritten.evaluate(i),
                expected,
                f"Mismatch at {i}: got {rewritten.evaluate(i)}, expected {expected}",
            )

        self.assertLess(rewritten.gate_count, state.gate_count)

    def test_local_rewrite_no_change(self) -> None:
        """Test that already optimal circuit is unchanged."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("and", 3, 2),
            ],
            outputs=[(4, False)],
            gate_count=2,
        )
        rewritten = state.try_local_rewrites()

        for i in range(8):
            original_result = state.evaluate(i)
            rewritten_result = rewritten.evaluate(i)
            self.assertEqual(original_result, rewritten_result)

        self.assertEqual(rewritten.gate_count, state.gate_count)

    def test_local_rewrite_aes_sbox(self) -> None:
        """Verify correctness on AES S-box after local rewrites."""
        import sys

        from stc.circuit_synth import (
            CircuitState,
            circuit_to_state,
            synthesize_multi_output_anf,
        )

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        try:
            result = synthesize_multi_output_anf(
                AES_SBOX_TABLE, input_bits=8, output_bits=8
            )
            state = circuit_to_state(result.circuit, input_bits=8, output_bits=8)

            rewritten = state.try_local_rewrites(max_window=3)

            for i in range(256):
                expected = AES_SBOX_TABLE[i]
                rewritten_result = rewritten.evaluate(i)
                self.assertEqual(
                    rewritten_result,
                    expected,
                    f"S-box mismatch at {i}: got {rewritten_result}, expected {expected}",
                )

            self.assertLessEqual(rewritten.gate_count, state.gate_count)
        finally:
            sys.setrecursionlimit(old_limit)


class TestOptimizeLinearLayers(unittest.TestCase):
    def test_optimize_linear_layers_simple(self) -> None:
        """Simple XOR chain gets optimized."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=4,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 4, 2),
                ("xor", 5, 3),
            ],
            outputs=[(6, False)],
            gate_count=3,
        )

        optimized = state.optimize_linear_layers()

        for i in range(16):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(
                original_result,
                optimized_result,
                f"Mismatch at input {i}: original={original_result}, optimized={optimized_result}",
            )

        self.assertLessEqual(optimized.xor_count, state.xor_count)

    def test_optimize_linear_layers_with_and(self) -> None:
        """AND boundaries are preserved during optimization."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=4,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 2, 3),
                ("and", 4, 5),
                ("xor", 6, 0),
            ],
            outputs=[(7, False)],
            gate_count=4,
        )

        optimized = state.optimize_linear_layers()

        for i in range(16):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(
                original_result,
                optimized_result,
                f"Mismatch at input {i}: original={original_result}, optimized={optimized_result}",
            )

        and_count = sum(1 for op, _, _ in optimized.gates if op == "and")
        self.assertEqual(and_count, 1, "AND gate should be preserved")

    def test_optimize_linear_layers_aes_sbox(self) -> None:
        """Verify correctness AND check gate count reduced on AES S-box."""
        import sys

        from stc.circuit_synth import (
            CircuitState,
            circuit_to_state,
            synthesize_multi_output_anf,
        )

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        try:
            result = synthesize_multi_output_anf(
                AES_SBOX_TABLE, input_bits=8, output_bits=8
            )
            state = circuit_to_state(result.circuit, input_bits=8, output_bits=8)

            original_xor_count = state.xor_count
            original_gate_count = state.gate_count

            optimized = state.optimize_linear_layers()

            for i in range(256):
                expected = AES_SBOX_TABLE[i]
                optimized_result = optimized.evaluate(i)
                self.assertEqual(
                    optimized_result,
                    expected,
                    f"S-box mismatch at {i}: got {optimized_result}, expected {expected}",
                )

            self.assertLessEqual(
                optimized.gate_count,
                original_gate_count,
                f"Gate count increased: {optimized.gate_count} > {original_gate_count}",
            )

            print(
                f"AES S-box linear optimization: {original_gate_count} -> {optimized.gate_count} gates"
            )
            print(f"  XOR gates: {original_xor_count} -> {optimized.xor_count}")
        finally:
            sys.setrecursionlimit(old_limit)

    def test_optimize_linear_layers_no_xor(self) -> None:
        """Circuit with no XOR gates is unchanged."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("and", 0, 1),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )

        optimized = state.optimize_linear_layers()

        for i in range(4):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(original_result, optimized_result)

        self.assertEqual(optimized.gate_count, state.gate_count)

    def test_optimize_linear_layers_shared_xor(self) -> None:
        """Test that shared XOR subexpressions are handled correctly."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("xor", 3, 2),
                ("and", 0, 1),
                ("xor", 5, 3),
            ],
            outputs=[(4, False), (6, False)],
            gate_count=4,
        )

        optimized = state.optimize_linear_layers()

        for i in range(8):
            original_result = state.evaluate(i)
            optimized_result = optimized.evaluate(i)
            self.assertEqual(
                original_result,
                optimized_result,
                f"Mismatch at input {i}: original={original_result}, optimized={optimized_result}",
            )


class TestSatWindowResynthesis(unittest.TestCase):
    def test_sat_window_resynthesis_simple(self) -> None:
        """Simple circuit gets optimized via window resynthesis."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
                ("xor", 3, 2),
                ("xor", 4, 3),
                ("xor", 5, 3),
            ],
            outputs=[(6, False)],
            gate_count=4,
        )

        original_outputs = [state.evaluate(i) for i in range(8)]

        optimized = state.sat_window_resynthesis(
            max_window_inputs=6, timeout_per_window=5000, max_iterations=10
        )

        for i in range(8):
            optimized_result = optimized.evaluate(i)
            self.assertEqual(
                optimized_result,
                original_outputs[i],
                f"Mismatch at input {i}: got {optimized_result}, expected {original_outputs[i]}",
            )

        self.assertLessEqual(
            optimized.gate_count,
            state.gate_count,
            f"Gate count should not increase: {optimized.gate_count} > {state.gate_count}",
        )

    def test_sat_window_resynthesis_no_change(self) -> None:
        """Already optimal circuit unchanged via window resynthesis."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[
                ("xor", 0, 1),
            ],
            outputs=[(2, False)],
            gate_count=1,
        )

        original_outputs = [state.evaluate(i) for i in range(4)]

        optimized = state.sat_window_resynthesis(
            max_window_inputs=6, timeout_per_window=5000, max_iterations=10
        )

        for i in range(4):
            optimized_result = optimized.evaluate(i)
            self.assertEqual(
                optimized_result,
                original_outputs[i],
                f"Mismatch at input {i}: got {optimized_result}, expected {original_outputs[i]}",
            )

        self.assertEqual(
            optimized.gate_count,
            state.gate_count,
            f"Gate count should remain the same: {optimized.gate_count} != {state.gate_count}",
        )

    def test_sat_window_resynthesis_correctness(self) -> None:
        """Output behavior preserved after window resynthesis."""
        from stc.circuit_synth import CircuitState

        state = CircuitState(
            input_bits=4,
            output_bits=2,
            gates=[
                ("xor", 0, 1),
                ("and", 2, 3),
                ("xor", 4, 5),
                ("xor", 6, 0),
                ("and", 6, 1),
                ("xor", 7, 8),
            ],
            outputs=[(7, False), (9, False)],
            gate_count=6,
        )

        original_outputs = [state.evaluate(i) for i in range(16)]

        optimized = state.sat_window_resynthesis(
            max_window_inputs=6, timeout_per_window=5000, max_iterations=20
        )

        for i in range(16):
            optimized_result = optimized.evaluate(i)
            self.assertEqual(
                optimized_result,
                original_outputs[i],
                f"Mismatch at input {i}: got {optimized_result}, expected {original_outputs[i]}",
            )

        self.assertLessEqual(
            optimized.gate_count,
            state.gate_count,
            f"Gate count should not increase: {optimized.gate_count} > {state.gate_count}",
        )
