import re
import unittest

from stc.mir import Binary, Load, MIRFunction, Store, Ternary, VReg
from stc.mir.emit_spirv_proper import emit_vulkan_compute
from stc.sched.regalloc import RegAllocation


class TestEmitSpirvProper(unittest.TestCase):
    def test_tid_mode_guardrails(self) -> None:
        mir = MIRFunction(
            input_regs=[VReg(0)],
            output_regs=[VReg(0)],
            instructions=[],
            num_virtual_regs=1,
        )
        alloc = RegAllocation(reg_assignment={})

        with self.assertRaises(ValueError):
            emit_vulkan_compute(
                mir,
                alloc,
                local_size=(4, 8, 1),
                tid_linear_mode="xonly",
            )
        with self.assertRaises(ValueError):
            emit_vulkan_compute(
                mir,
                alloc,
                local_size=(4, 2, 2),
                tid_linear_mode="xy",
            )

    def test_spill_slots_are_emitted(self) -> None:
        mir = MIRFunction(
            input_regs=[VReg(0)],
            output_regs=[VReg(1)],
            instructions=[
                Store(src=VReg(0), slot=0, dst=None),
                Load(dst=VReg(1), slot=0),
            ],
            num_virtual_regs=2,
        )
        alloc = RegAllocation(reg_assignment={1: 1}, spills=[1])
        spv = emit_vulkan_compute(mir, alloc, local_size=(64, 1, 1))

        self.assertIn("OpTypePointer Function", spv)
        match = re.search(r"(%\d+) = OpVariable %\d+ Function", spv)
        self.assertIsNotNone(match)
        slot_ptr = match.group(1) if match else None
        self.assertIsNotNone(slot_ptr)
        if slot_ptr is not None:
            self.assertIn(f"OpStore {slot_ptr}", spv)
            self.assertRegex(spv, rf"%\d+ = OpLoad %\d+ {re.escape(slot_ptr)}")

    def test_step_loop_state_only_is_emitted(self) -> None:
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1)],
            output_regs=[VReg(2)],
            instructions=[Binary(dst=VReg(2), op="xor", a=VReg(0), b=VReg(1))],
            num_virtual_regs=3,
        )
        alloc = RegAllocation(reg_assignment={2: 2})
        spv = emit_vulkan_compute(
            mir,
            alloc,
            local_size=(64, 1, 1),
            step_count=4,
            loop_state_words=1,
            loop_state_input_offset=0,
            loop_state_output_offset=0,
            loop_output_state_only=True,
        )
        self.assertIn("Step loop enabled: steps=4 state_words=1", spv)
        self.assertIn("OpLoopMerge", spv)
        self.assertIn("OpULessThan", spv)
        self.assertIn("OpStore", spv)
        # State-only mode keeps loop-carried state in function-local storage.
        self.assertGreaterEqual(spv.count("OpVariable %"), 3)
        self.assertIn("OpLoad", spv)
        # Input loads + final output store only (no per-iteration global state ping-pong).
        self.assertLessEqual(spv.count("OpAccessChain"), 3)

    def test_ternary_xor3_uses_fast_path(self) -> None:
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1), VReg(2)],
            output_regs=[VReg(3)],
            instructions=[
                Ternary(dst=VReg(3), a=VReg(0), b=VReg(1), c=VReg(2), imm8=0x96)
            ],
            num_virtual_regs=4,
        )
        alloc = RegAllocation(reg_assignment={3: 3})
        spv = emit_vulkan_compute(mir, alloc, local_size=(64, 1, 1))
        # xor3 should lower to two XORs, not full minterm decomposition.
        self.assertEqual(spv.count("OpBitwiseXor"), 2)
        self.assertNotIn("OpBitwiseAnd", spv)

    def test_step_loop_validates_ranges(self) -> None:
        mir = MIRFunction(
            input_regs=[VReg(0)],
            output_regs=[VReg(1)],
            instructions=[Binary(dst=VReg(1), op="xor", a=VReg(0), b=VReg(0))],
            num_virtual_regs=2,
        )
        alloc = RegAllocation(reg_assignment={1: 1})
        with self.assertRaises(ValueError):
            emit_vulkan_compute(
                mir,
                alloc,
                step_count=2,
                loop_state_words=2,
                loop_state_input_offset=0,
                loop_state_output_offset=0,
            )

    def test_binary_andn_is_emitted(self) -> None:
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1)],
            output_regs=[VReg(2)],
            instructions=[Binary(dst=VReg(2), op="andn", a=VReg(0), b=VReg(1))],
            num_virtual_regs=3,
        )
        alloc = RegAllocation(reg_assignment={2: 2})
        spv = emit_vulkan_compute(mir, alloc, local_size=(64, 1, 1))
        self.assertIn("OpNot", spv)
        self.assertIn("OpBitwiseAnd", spv)


if __name__ == "__main__":
    unittest.main()
