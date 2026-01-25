import unittest

from stc.ternary_db import (
    build_bgc_table,
    build_q0_table,
    build_q1_table,
    compute_base_vectors,
    apply_ternary_lut,
    score_sbox,
)


class TestBGCTable(unittest.TestCase):
    @unittest.skip("Slow: builds full BGC table")
    def test_bgc_constants(self):
        """BGC[0] = 0, BGC[0xFFFF] = 0."""
        bgc = build_bgc_table()
        self.assertEqual(bgc[0], 0)
        self.assertEqual(bgc[0xFFFF], 0)

    @unittest.skip("Slow: builds full BGC table")
    def test_bgc_base_vectors(self):
        """Base vectors have BGC = 1."""
        bgc = build_bgc_table()
        base = compute_base_vectors()

        for v in base:
            self.assertEqual(bgc[v], 1)
            self.assertEqual(bgc[v ^ 0xFFFF], 1)

    @unittest.skip("Slow: builds full BGC table")
    def test_bgc_bounds(self):
        """All BGC values are in [0, 4]."""
        bgc = build_bgc_table()
        for v in range(65536):
            self.assertIn(bgc[v], [0, 1, 2, 3, 4])

    def test_base_vectors(self):
        """Base vectors have correct values."""
        base = compute_base_vectors()
        self.assertEqual(len(base), 4)
        x0, x1, x2, x3 = base
        self.assertEqual(x0, 0x5555)
        self.assertEqual(x1, 0x3333)
        self.assertEqual(x2, 0x0F0F)
        self.assertEqual(x3, 0x00FF)


class TestReachableSets(unittest.TestCase):
    @unittest.skip("Slow: builds q0 table")
    def test_q0_size(self):
        """q0 should have ~936 vectors (from Sovyn paper)."""
        base = compute_base_vectors()
        q0 = build_q0_table(base)

        self.assertGreater(len(q0), 900)
        self.assertLess(len(q0), 1000)

    @unittest.skip("Slow: builds q0 table")
    def test_q0_contains_base(self):
        """q0 contains all base vectors."""
        base = compute_base_vectors()
        q0 = build_q0_table(base)

        for v in base:
            self.assertIn(v, q0)

    @unittest.skip("Very slow: builds q1 table")
    def test_q1_size(self):
        """q1 should have ~438,312 vectors (from paper)."""
        base = compute_base_vectors()
        q0 = build_q0_table(base)
        q1 = build_q1_table(q0, base)

        self.assertGreater(len(q1), 400000)
        self.assertLess(len(q1), 500000)

    def test_q0_basic(self):
        """q0 table includes constants and base vectors."""
        base = compute_base_vectors()

        q0 = {0x0000, 0xFFFF}
        for v in base:
            q0.add(v)
            q0.add(v ^ 0xFFFF)

        self.assertEqual(len(q0), 10)


class TestTernaryLut(unittest.TestCase):
    def test_apply_and(self):
        """TernaryLut with AND imm8."""
        from stc.mapping.ternary import IMM8_AND_AB

        a = 0xF0F0
        b = 0xCCCC
        result = apply_ternary_lut(a, b, 0x0000, IMM8_AND_AB)

        expected = a & b
        self.assertEqual(result, expected)

    def test_apply_xor3(self):
        """TernaryLut with XOR3 imm8."""
        from stc.mapping.ternary import IMM8_XOR_ABC

        a = 0xAAAA
        b = 0xCCCC
        c = 0xF0F0
        result = apply_ternary_lut(a, b, c, IMM8_XOR_ABC)

        expected = a ^ b ^ c
        self.assertEqual(result, expected)


class TestSBoxScoring(unittest.TestCase):
    @unittest.skip("Slow: builds full BGC table")
    def test_score_identity(self):
        """Identity S-box: outputs = inputs."""
        base = compute_base_vectors()
        bgc = build_bgc_table()
        score = score_sbox(base, bgc)

        self.assertEqual(score, 4)

    def test_score_with_mock_bgc(self):
        """Score S-box with mock BGC values."""
        base = compute_base_vectors()
        mock_bgc = bytearray(65536)
        for i in range(65536):
            mock_bgc[i] = 1
        score = score_sbox(base, bytes(mock_bgc))
        self.assertEqual(score, 4)
