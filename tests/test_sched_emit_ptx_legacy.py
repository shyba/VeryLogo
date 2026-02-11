import unittest

from stc.sched.emit.ptx import PTXEmitter
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


class TestPTXLegacyEmitter(unittest.TestCase):
    def test_emits_andn(self) -> None:
        emitter = PTXEmitter()
        schedule = Schedule(gate_cycle={0: 0})
        allocation = RegAllocation(reg_assignment={2: 2})
        code = emitter.emit(
            schedule=schedule,
            allocation=allocation,
            gates=[("andn", 0, 1)],
            input_bits=2,
            outputs=[(2, False)],
            function_name="circuit",
        )
        self.assertIn("lop3.b32 %r4, %r0, %r1, %r0, 12;", code)


if __name__ == "__main__":
    unittest.main()
