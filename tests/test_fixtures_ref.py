import unittest
from tests.fixtures_ref import (
    vec_add,
    xor_tree,
    acc_fsm,
    lfsr_table,
    roundN,
    nonlinear_island,
    pmux16,
    pmux32,
    mux_reconverge,
    rotmix,
    slice_concat,
)
from tests.fixtures_ref.sequential_ref import lfsr_next


class TestThroughputRef(unittest.TestCase):
    def test_vec_add_basic(self):
        a = 0x00000001_00000002_00000003_00000004
        b = 0x00000005_00000006_00000007_00000008
        result = vec_add(a, b, N=4)
        expected = 0x00000006_00000008_0000000A_0000000C
        self.assertEqual(result, expected)

    def test_vec_add_overflow(self):
        a = 0xFFFFFFFF_FFFFFFFE
        b = 0x00000001_00000002
        result = vec_add(a, b, N=2)
        expected = 0x00000000_00000000
        self.assertEqual(result, expected)

    def test_xor_tree_depth_0(self):
        inputs = [1, 2, 3, 4]
        result = xor_tree(inputs, depth=0)
        expected = 1 ^ 2 ^ 3 ^ 4
        self.assertEqual(result, expected)

    def test_xor_tree_depth_2(self):
        inputs = [0xA, 0xB, 0xC, 0xD]
        result = xor_tree(inputs, depth=2)
        expected = 0xA ^ 0xB ^ 0xC ^ 0xD
        self.assertEqual(result, expected)


class TestSequentialRef(unittest.TestCase):
    def test_acc_fsm_basic(self):
        states = acc_fsm(limit=3, x=0x100, ticks=5)
        self.assertEqual(len(states), 5)
        self.assertEqual(states[0]["acc"], 0x100)
        self.assertEqual(states[0]["i"], 1)
        self.assertEqual(states[1]["acc"], (0x100 + 0x101) & 0xFFFFFFFF)
        self.assertEqual(states[1]["i"], 2)

    def test_acc_fsm_with_reset(self):
        states = acc_fsm(limit=10, x=0xFF, ticks=5, rst_at=[2])
        self.assertEqual(states[2]["acc"], 0)
        self.assertEqual(states[2]["i"], 0)

    def test_lfsr_next(self):
        state = 0x12345678
        next_state = lfsr_next(state)
        self.assertIsInstance(next_state, int)
        self.assertLessEqual(next_state, 0xFFFFFFFF)

    def test_lfsr_table_basic(self):
        table = [i * 0x11 for i in range(32)]
        states = lfsr_table(seed=1, table=table, ticks=5, table_width=32)
        self.assertEqual(len(states), 5)


class TestOptimizableRef(unittest.TestCase):
    def test_roundN_basic(self):
        states = roundN(in_val=0x123456789ABCDEF0, R=4, ticks=5)
        self.assertEqual(len(states), 5)
        self.assertEqual(states[0]["out"], 0x123456789ABCDEF0)
        self.assertGreater(states[4]["r"], 0)

    def test_roundN_completes(self):
        states = roundN(in_val=0xAAAAAAAAAAAAAAAA, R=8, ticks=10)
        final_state = states[-1]
        self.assertEqual(final_state["r"], 8)

    def test_nonlinear_island_basic(self):
        result = nonlinear_island(0x123456789ABCDEF0, 0xFEDCBA9876543210)
        self.assertIsInstance(result, int)
        self.assertLessEqual(result, 0xFFFFFFFFFFFFFFFF)

    def test_nonlinear_island_zeros(self):
        result = nonlinear_island(0, 0)
        self.assertEqual(result, 0)


class TestControlRef(unittest.TestCase):
    def test_pmux16_all_cases(self):
        inputs = [i * 0x1111111111111111 for i in range(16)]
        for sel in range(16):
            result = pmux16(sel, *inputs)
            expected = inputs[sel] & 0xFFFFFFFFFFFFFFFF
            self.assertEqual(result, expected)

    def test_pmux32_boundary(self):
        inputs = [i for i in range(32)]
        result = pmux32(0, *inputs)
        self.assertEqual(result, 0)
        result = pmux32(31, *inputs)
        self.assertEqual(result, 31)

    def test_mux_reconverge_basic(self):
        result = mux_reconverge(1, 0, 0x100, 0x200, 0x300, 0x400)
        self.assertIsInstance(result, int)

    def test_mux_reconverge_both_conditions(self):
        result = mux_reconverge(1, 1, 0xA, 0xB, 0xC, 0xD)
        self.assertIsInstance(result, int)


class TestBitvectorRef(unittest.TestCase):
    def test_rotmix_basic(self):
        x = 0x123456789ABCDEF0
        result = rotmix(x)
        self.assertIsInstance(result, int)
        self.assertLessEqual(result, 0xFFFFFFFFFFFFFFFF)
        self.assertNotEqual(result, x)

    def test_rotmix_zero(self):
        result = rotmix(0)
        const = 0x9E3779B97F4A7C15
        self.assertEqual(result, const)

    def test_slice_concat_64bit(self):
        x = 0x123456789ABCDEF0
        result = slice_concat(x, width=64)
        self.assertIsInstance(result, int)
        self.assertLessEqual(result, 0xFFFFFFFFFFFFFFFF)

    def test_slice_concat_32bit(self):
        x = 0x12345678
        result = slice_concat(x, width=32)
        self.assertIsInstance(result, int)
        self.assertLessEqual(result, 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
