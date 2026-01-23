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
