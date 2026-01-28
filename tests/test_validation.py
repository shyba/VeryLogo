"""Tests for emission validation context."""

import unittest

from stc.sched.regalloc import RegAllocation
from stc.sched.validation import EmitContext


class TestEmitContext(unittest.TestCase):
    """Test validation context for code emission."""

    def test_validate_node_negative(self):
        """Negative node indices should raise ValueError."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=3,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 0, 5: 1, 6: 2}),
        )

        with self.assertRaises(ValueError) as cm:
            ctx.validate_node(-1)
        self.assertIn("negative", str(cm.exception).lower())

    def test_validate_node_exceeds_total(self):
        """Node index beyond total nodes should raise ValueError."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=3,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 0, 5: 1, 6: 2}),
        )

        with self.assertRaises(ValueError) as cm:
            ctx.validate_node(10)
        self.assertIn("exceeds", str(cm.exception).lower())

    def test_validate_node_not_allocated(self):
        """Gate node not in allocation should raise ValueError."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=3,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 0, 5: 1}),
        )

        with self.assertRaises(ValueError) as cm:
            ctx.validate_node(6)
        self.assertIn("not allocated", str(cm.exception).lower())

    def test_validate_input_node(self):
        """Input nodes should always be valid."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=3,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 0, 5: 1, 6: 2}),
        )

        ctx.validate_node(0)
        ctx.validate_node(3)

    def test_get_register_input(self):
        """Input nodes should return their index as register."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=3,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 0, 5: 1, 6: 2}),
        )

        self.assertEqual(ctx.get_register(0), 0)
        self.assertEqual(ctx.get_register(3), 3)

    def test_get_register_gate(self):
        """Gate nodes should return allocated physical register."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=3,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 10, 5: 11, 6: 12}),
        )

        self.assertEqual(ctx.get_register(4), 10)
        self.assertEqual(ctx.get_register(5), 11)
        self.assertEqual(ctx.get_register(6), 12)

    def test_is_spilled(self):
        """Check if node is spilled to memory."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=5,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 0, 5: 1, 6: 2}, spills=[7, 8]),
        )

        self.assertFalse(ctx.is_spilled(4))
        self.assertFalse(ctx.is_spilled(5))
        self.assertTrue(ctx.is_spilled(7))
        self.assertTrue(ctx.is_spilled(8))

    def test_validate_spilled_node(self):
        """Spilled nodes should pass validation."""
        ctx = EmitContext(
            input_bits=4,
            num_gates=5,
            num_physical_regs=8,
            allocation=RegAllocation(reg_assignment={4: 0, 5: 1, 6: 2}, spills=[7, 8]),
        )

        ctx.validate_node(7)
        ctx.validate_node(8)


if __name__ == "__main__":
    unittest.main()
