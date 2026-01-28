"""Tests for Machine IR layer."""

import unittest

from stc.circuit_synth import CircuitState
from stc.mir import Binary, Const, MIRFunction, Ternary, Unary, VReg
from stc.mir.emit_avx512 import emit_avx512
from stc.mir.emit_ptx import emit_ptx
from stc.mir.lower import circuit_to_mir
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


class TestMIRLowering(unittest.TestCase):
    """Test CircuitState to MIR lowering."""

    def test_simple_and_gate(self):
        """Lower a simple AND gate to MIR."""
        circuit = CircuitState(
            input_bits=2,
            output_bits=1,
            gates=[("and", 0, 1)],
            outputs=[(2, False)],
            gate_count=1,
        )

        schedule = Schedule(gate_cycle={0: 0})
        allocation = RegAllocation(reg_assignment={2: 2})

        mir = circuit_to_mir(circuit, schedule, allocation)

        self.assertEqual(len(mir.input_regs), 2)
        self.assertEqual(len(mir.output_regs), 1)
        self.assertEqual(len(mir.instructions), 1)

        inst = mir.instructions[0]
        self.assertIsInstance(inst, Binary)
        self.assertEqual(inst.op, "and")
        self.assertEqual(inst.dst, VReg(2))
        self.assertEqual(inst.a, VReg(0))
        self.assertEqual(inst.b, VReg(1))

    def test_ternary_gate(self):
        """Lower a ternary gate to MIR."""
        circuit = CircuitState(
            input_bits=3,
            output_bits=1,
            gates=[("ternary", 0, 1, 2, 0xCA)],
            outputs=[(3, False)],
            gate_count=1,
        )

        schedule = Schedule(gate_cycle={0: 0})
        allocation = RegAllocation(reg_assignment={3: 3})

        mir = circuit_to_mir(circuit, schedule, allocation)

        self.assertEqual(len(mir.instructions), 1)
        inst = mir.instructions[0]
        self.assertIsInstance(inst, Ternary)
        self.assertEqual(inst.imm8, 0xCA)
        self.assertEqual(inst.a, VReg(0))
        self.assertEqual(inst.b, VReg(1))
        self.assertEqual(inst.c, VReg(2))

    def test_not_gate(self):
        """Lower a NOT gate to MIR."""
        circuit = CircuitState(
            input_bits=1,
            output_bits=1,
            gates=[("not", 0)],
            outputs=[(1, False)],
            gate_count=1,
        )

        schedule = Schedule(gate_cycle={0: 0})
        allocation = RegAllocation(reg_assignment={1: 1})

        mir = circuit_to_mir(circuit, schedule, allocation)

        self.assertEqual(len(mir.instructions), 1)
        inst = mir.instructions[0]
        self.assertIsInstance(inst, Unary)
        self.assertEqual(inst.op, "not")
        self.assertEqual(inst.a, VReg(0))

    def test_const_gate(self):
        """Lower a constant gate to MIR."""
        circuit = CircuitState(
            input_bits=0,
            output_bits=1,
            gates=[("const", 1)],
            outputs=[(0, False)],
            gate_count=1,
        )

        schedule = Schedule(gate_cycle={0: 0})
        allocation = RegAllocation(reg_assignment={0: 0})

        mir = circuit_to_mir(circuit, schedule, allocation)

        self.assertEqual(len(mir.instructions), 1)
        inst = mir.instructions[0]
        self.assertIsInstance(inst, Const)
        self.assertEqual(inst.value, 1)

    def test_multiple_gates(self):
        """Lower multiple gates in sequence."""
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

        schedule = Schedule(gate_cycle={0: 0, 1: 1})
        allocation = RegAllocation(reg_assignment={2: 2, 3: 3})

        mir = circuit_to_mir(circuit, schedule, allocation)

        self.assertEqual(len(mir.instructions), 2)
        self.assertIsInstance(mir.instructions[0], Binary)
        self.assertIsInstance(mir.instructions[1], Unary)


class TestAVX512Emission(unittest.TestCase):
    """Test AVX-512 code emission from MIR."""

    def test_emit_simple_and(self):
        """Emit AVX-512 code for a simple AND gate."""
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1)],
            output_regs=[VReg(2)],
            instructions=[Binary(dst=VReg(2), op="and", a=VReg(0), b=VReg(1))],
            num_virtual_regs=3,
        )

        allocation = RegAllocation(reg_assignment={2: 2})
        code = emit_avx512(mir, allocation)

        self.assertIn("#include <immintrin.h>", code)
        self.assertIn("void circuit(__m512i* in, __m512i* out)", code)
        self.assertIn("r2 = _mm512_and_si512(r0, r1);", code)
        self.assertIn("out[0] = r2;", code)

    def test_emit_ternary(self):
        """Emit AVX-512 code for a ternary operation."""
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1), VReg(2)],
            output_regs=[VReg(3)],
            instructions=[
                Ternary(dst=VReg(3), a=VReg(0), b=VReg(1), c=VReg(2), imm8=0xCA)
            ],
            num_virtual_regs=4,
        )

        allocation = RegAllocation(reg_assignment={3: 3})
        code = emit_avx512(mir, allocation)

        self.assertIn("r3 = _mm512_ternarylogic_epi32(r0, r1, r2, 202);", code)

    def test_emit_not(self):
        """Emit AVX-512 code for NOT operation."""
        mir = MIRFunction(
            input_regs=[VReg(0)],
            output_regs=[VReg(1)],
            instructions=[Unary(dst=VReg(1), op="not", a=VReg(0))],
            num_virtual_regs=2,
        )

        allocation = RegAllocation(reg_assignment={1: 1})
        code = emit_avx512(mir, allocation)

        self.assertIn("__m512i ones = _mm512_set1_epi32(-1);", code)
        self.assertIn("r1 = _mm512_xor_si512(r0, ones);", code)

    def test_emit_const(self):
        """Emit AVX-512 code for constant."""
        mir = MIRFunction(
            input_regs=[],
            output_regs=[VReg(0)],
            instructions=[Const(dst=VReg(0), value=0)],
            num_virtual_regs=1,
        )

        allocation = RegAllocation(reg_assignment={0: 0})
        code = emit_avx512(mir, allocation)

        self.assertIn("r0 = _mm512_setzero_si512();", code)


class TestPTXEmission(unittest.TestCase):
    """Test PTX code emission from MIR."""

    def test_emit_simple_and(self):
        """Emit PTX code for a simple AND gate."""
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1)],
            output_regs=[VReg(2)],
            instructions=[Binary(dst=VReg(2), op="and", a=VReg(0), b=VReg(1))],
            num_virtual_regs=3,
        )

        allocation = RegAllocation(reg_assignment={2: 2})
        code = emit_ptx(mir, allocation)

        self.assertIn(".version 6.0", code)
        self.assertIn(".target sm_61", code)
        self.assertIn(".visible .entry circuit(", code)
        self.assertIn("and.b32 %r2, %r0, %r1;", code)
        self.assertIn("st.global.u32 [%out_addr + 0], %r2;", code)

    def test_emit_ternary(self):
        """Emit PTX code for a ternary operation."""
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1), VReg(2)],
            output_regs=[VReg(3)],
            instructions=[
                Ternary(dst=VReg(3), a=VReg(0), b=VReg(1), c=VReg(2), imm8=0xCA)
            ],
            num_virtual_regs=4,
        )

        allocation = RegAllocation(reg_assignment={3: 3})
        code = emit_ptx(mir, allocation)

        self.assertIn("lop3.b32 %r3, %r0, %r1, %r2, 202;", code)

    def test_emit_not(self):
        """Emit PTX code for NOT operation."""
        mir = MIRFunction(
            input_regs=[VReg(0)],
            output_regs=[VReg(1)],
            instructions=[Unary(dst=VReg(1), op="not", a=VReg(0))],
            num_virtual_regs=2,
        )

        allocation = RegAllocation(reg_assignment={1: 1})
        code = emit_ptx(mir, allocation)

        self.assertIn("not.b32 %r1, %r0;", code)


if __name__ == "__main__":
    unittest.main()
