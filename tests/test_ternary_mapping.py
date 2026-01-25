import unittest

from stc.mapping.ternary import (
    compute_imm8,
    imm8_to_func,
    extract_3input_cone,
    negate_imm8,
    complement_input_a,
    complement_input_b,
    complement_input_c,
    IMM8_AND_AB,
    IMM8_XOR_AB,
    IMM8_XOR_ABC,
    IMM8_MAJ,
)
from stc.tick_ir import And, Or, Xor, Not, Var, BoolConst, BoolType, TernaryLut


class TestImm8Computation(unittest.TestCase):
    def test_and(self):
        """AND(a, b) computed correctly."""
        imm8 = compute_imm8(lambda a, b, c: a & b)
        self.assertEqual(imm8, IMM8_AND_AB)

    def test_xor(self):
        """XOR(a, b) computed correctly."""
        imm8 = compute_imm8(lambda a, b, c: a ^ b)
        self.assertEqual(imm8, IMM8_XOR_AB)

    def test_xor3(self):
        """XOR(a, b, c) = 0x96."""
        imm8 = compute_imm8(lambda a, b, c: a ^ b ^ c)
        self.assertEqual(imm8, 0x96)
        self.assertEqual(imm8, IMM8_XOR_ABC)

    def test_majority(self):
        """Majority(a, b, c) = 0xE8."""
        imm8 = compute_imm8(lambda a, b, c: (a & b) | (b & c) | (a & c))
        self.assertEqual(imm8, 0xE8)
        self.assertEqual(imm8, IMM8_MAJ)

    def test_roundtrip(self):
        """imm8 -> func -> imm8 roundtrip."""
        for imm8 in [0x00, 0x80, 0x66, 0x96, 0xE8, 0xFF]:
            func = imm8_to_func(imm8)
            recovered = compute_imm8(func)
            self.assertEqual(recovered, imm8)

    def test_negate(self):
        """NOT(AND(a,b)) via negate_imm8."""
        imm8 = negate_imm8(IMM8_AND_AB)
        nand_expected = compute_imm8(lambda a, b, c: not (a & b))
        self.assertEqual(imm8, nand_expected)


class TestConeExtraction(unittest.TestCase):
    def test_simple_and(self):
        """Extract cone from AND(a, b)."""
        a = Var(name="a")
        b = Var(name="b")
        expr = And(a=a, b=b)

        cone = extract_3input_cone(expr)
        self.assertIsNotNone(cone)
        self.assertEqual(cone.imm8, IMM8_AND_AB)
        self.assertEqual(cone.gate_count, 1)

    def test_xor_chain(self):
        """Extract cone from XOR(XOR(a, b), c)."""
        a = Var(name="a")
        b = Var(name="b")
        c = Var(name="c")
        expr = Xor(a=Xor(a=a, b=b), b=c)

        cone = extract_3input_cone(expr)
        self.assertIsNotNone(cone)
        self.assertEqual(cone.imm8, IMM8_XOR_ABC)
        self.assertEqual(cone.gate_count, 2)

    def test_too_many_inputs(self):
        """Return None if > 3 inputs."""
        a = Var(name="a")
        b = Var(name="b")
        c = Var(name="c")
        d = Var(name="d")
        expr = And(a=And(a=a, b=b), b=And(a=c, b=d))

        cone = extract_3input_cone(expr)
        self.assertIsNone(cone)

    def test_not_absorption(self):
        """NOT absorbed into imm8."""
        a = Var(name="a")
        b = Var(name="b")
        expr = Not(x=And(a=a, b=b))

        cone = extract_3input_cone(expr)
        self.assertIsNotNone(cone)
        nand_expected = negate_imm8(IMM8_AND_AB)
        self.assertEqual(cone.imm8, nand_expected)


class TestComplementInput(unittest.TestCase):
    def test_complement_a(self):
        """complement_input_a swaps low and high nibbles."""
        imm8 = 0xAB
        complemented = complement_input_a(imm8)
        self.assertEqual(complemented, 0xBA)

    def test_complement_roundtrip(self):
        """Complementing an input twice returns original."""
        imm8 = 0x80
        self.assertEqual(complement_input_a(complement_input_a(imm8)), imm8)
        self.assertEqual(complement_input_b(complement_input_b(imm8)), imm8)
        self.assertEqual(complement_input_c(complement_input_c(imm8)), imm8)
