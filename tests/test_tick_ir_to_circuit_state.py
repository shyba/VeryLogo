import unittest

from stc.tick_ir import (
    TickIR,
    BitVecType,
    BoolType,
    Var,
    Slice,
    Concat,
    Xor,
    Shl,
    LShr,
    BitVecConst,
    BoolConst,
    Mux,
)
from stc.tick_ir_to_circuit_state import lower_tick_ir_to_circuit_state


def _eval_circuit_state_once(circuit, in_bits):
    nodes = list(in_bits)
    for gate in circuit.gates:
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
            assert op == "ternary"
            ia = nodes[a] & 1
            ib = nodes[b] & 1
            ic = nodes[c] & 1
            idx = (ic << 2) | (ib << 1) | ia
            nodes.append((imm8 >> idx) & 1)
            continue

        op = gate[0]
        if op == "const":
            nodes.append(gate[1] & 1)
        elif op == "not":
            nodes.append((nodes[gate[1]] ^ 1) & 1)
        elif op == "xor":
            nodes.append((nodes[gate[1]] ^ nodes[gate[2]]) & 1)
        elif op == "and":
            nodes.append((nodes[gate[1]] & nodes[gate[2]]) & 1)
        elif op == "or":
            nodes.append((nodes[gate[1]] | nodes[gate[2]]) & 1)
        else:
            raise AssertionError(f"unexpected gate op: {op}")
    return [nodes[idx] ^ (1 if inv else 0) for idx, inv in circuit.outputs]


class TestTickIrToCircuitState(unittest.TestCase):
    def test_lowering_bitvec_ops_and_state(self):
        # a:4, s:2 -> o:4 and next s:2
        a = Var("a")
        s = Var("s")

        # o = (a << 1) XOR (a >> 1) XOR (s repeated to 4 bits)
        s_lo = Slice(x=s, offset=0, width=1)
        s_hi = Slice(x=s, offset=1, width=1)
        s_rep = Concat(parts=[s_hi, s_lo, s_hi, s_lo])

        o_expr = Xor(
            a=Shl(a=a, b=BitVecConst(width=4, value=1)),
            b=Xor(
                a=LShr(a=a, b=BitVecConst(width=4, value=1)),
                b=s_rep,
            ),
        )

        # next_s = mux(cond=a[0], a=s, b=01)
        cond = Slice(x=a, offset=0, width=1)
        nx_expr = Mux(
            cond=cond, a=s, b=Concat(parts=[BoolConst(False), BoolConst(True)])
        )

        ir = TickIR(
            name="t",
            inputs={"a": BitVecType(4)},
            outputs={"o": BitVecType(4)},
            state={"s": BitVecType(2)},
            reset_state={"s": BitVecConst(width=2, value=0)},
            next_state={"s": nx_expr},
            output_exprs={"o": o_expr},
        )

        circuit, layout = lower_tick_ir_to_circuit_state(ir)
        self.assertEqual(layout.input_bits, 6)
        self.assertEqual(layout.output_bits, 6)
        self.assertEqual(circuit.input_bits, 6)
        self.assertEqual(circuit.output_bits, 6)

        # Exhaustive check for all (a,s) combos.
        for a_val in range(16):
            for s_val in range(4):
                in_bits = []
                for i in range(4):
                    in_bits.append((a_val >> i) & 1)
                for i in range(2):
                    in_bits.append((s_val >> i) & 1)

                out_bits = _eval_circuit_state_once(circuit, in_bits)
                o_val = sum((out_bits[i] & 1) << i for i in range(4))
                nx_val = sum((out_bits[4 + i] & 1) << i for i in range(2))

                exp_o = (
                    ((a_val << 1) & 0xF)
                    ^ ((a_val >> 1) & 0xF)
                    ^ (((s_val & 1) * 0b0101) | (((s_val >> 1) & 1) * 0b1010))
                ) & 0xF
                exp_nx = s_val if (a_val & 1) else 0b01

                self.assertEqual(o_val, exp_o)
                self.assertEqual(nx_val, exp_nx)
