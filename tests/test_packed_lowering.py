import random
import unittest

from stc.tick_ir import (
    BitVecType,
    BitVecConst,
    BoolConst,
    Var,
    Xor,
    Slice,
    Concat,
    Rotl,
    Rotr,
    Eq,
    Mux,
    TickIR,
)
from stc.tick_ir_to_packed_circuit_state import (
    PackedLoweringError,
    lower_tick_ir_to_packed_circuit_state,
)
from stc.packed_circuit import eval_packed_circuit_words


class TestPackedLowering(unittest.TestCase):
    def test_lower_and_eval_xor_128(self) -> None:
        # 128-bit xor, sliced into outputs; next_state mirrors output.
        expr = Xor(a=Var("x"), b=Var("s"))
        ir = TickIR(
            name="xor128",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o0": BitVecType(64), "o1": BitVecType(64)},
            state={"s": BitVecType(128)},
            reset_state={"s": BitVecConst(width=128, value=0)},
            next_state={"s": expr},
            output_exprs={
                "o0": Slice(x=expr, offset=0, width=64),
                "o1": Slice(x=expr, offset=64, width=64),
            },
        )

        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 6)  # x(2) + y(2) + s(2)
        self.assertEqual(layout.output_words, 4)  # o0(1) + o1(1) + next_state(s=2)

        for _ in range(50):
            xv = random.getrandbits(128)
            yv = random.getrandbits(128)
            sv = random.getrandbits(128)
            inputs = [
                xv & ((1 << 64) - 1),
                (xv >> 64) & ((1 << 64) - 1),
                yv & ((1 << 64) - 1),
                (yv >> 64) & ((1 << 64) - 1),
                sv & ((1 << 64) - 1),
                (sv >> 64) & ((1 << 64) - 1),
            ]
            outs = eval_packed_circuit_words(circuit, inputs)
            out_w0, out_w1, ns_w0, ns_w1 = outs
            expect = (xv ^ sv) & ((1 << 128) - 1)
            self.assertEqual(out_w0, expect & ((1 << 64) - 1))
            self.assertEqual(out_w1, (expect >> 64) & ((1 << 64) - 1))
            self.assertEqual(ns_w0, expect & ((1 << 64) - 1))
            self.assertEqual(ns_w1, (expect >> 64) & ((1 << 64) - 1))

    def test_reject_non_const_shift_amount(self) -> None:
        # shift amount is a var (not constant) -> reject for now.
        from stc.tick_ir import Shl

        ir = TickIR(
            name="bad_shift",
            inputs={"x": BitVecType(64)},
            outputs={},
            state={"s": BitVecType(64)},
            reset_state={"s": BitVecConst(width=64, value=0)},
            next_state={"s": Shl(a=Var("s"), b=Var("x"))},
            output_exprs={},
        )
        with self.assertRaises(PackedLoweringError):
            lower_tick_ir_to_packed_circuit_state(ir)

    def test_const_shift_works(self) -> None:
        from stc.tick_ir import Shl

        ir = TickIR(
            name="shl128",
            inputs={"x": BitVecType(128)},
            outputs={},
            state={"s": BitVecType(128)},
            reset_state={"s": BitVecConst(width=128, value=0)},
            next_state={"s": Shl(a=Var("s"), b=BitVecConst(width=32, value=5))},
            output_exprs={},
        )
        circuit, _layout = lower_tick_ir_to_packed_circuit_state(ir)

        for _ in range(50):
            xv = random.getrandbits(128)
            sv = random.getrandbits(128)
            inputs = [
                xv & ((1 << 64) - 1),
                (xv >> 64) & ((1 << 64) - 1),
                sv & ((1 << 64) - 1),
                (sv >> 64) & ((1 << 64) - 1),
            ]
            outs = eval_packed_circuit_words(circuit, inputs)
            ns_w0, ns_w1 = outs
            expect = (sv << 5) & ((1 << 128) - 1)
            self.assertEqual(ns_w0, expect & ((1 << 64) - 1))
            self.assertEqual(ns_w1, (expect >> 64) & ((1 << 64) - 1))

    def test_unaligned_slice_96bits_offset13(self) -> None:
        # 96-bit unaligned slice at offset 13 from a 192-bit input.
        # This spans 3 source words: bits [13:108] of a 192-bit value.
        slice_expr = Slice(x=Var("x"), offset=13, width=96)
        ir = TickIR(
            name="slice96",
            inputs={"x": BitVecType(192)},
            outputs={"o": BitVecType(96)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": slice_expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 3)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            xv = random.getrandbits(192)
            inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(3)]
            outs = eval_packed_circuit_words(circuit, inputs)
            expect = (xv >> 13) & ((1 << 96) - 1)
            out_w0, out_w1 = outs
            self.assertEqual(out_w0, expect & ((1 << 64) - 1))
            self.assertEqual(out_w1, (expect >> 64) & ((1 << 32) - 1))

    def test_unaligned_slice_spans_3plus_words(self) -> None:
        # 200-bit slice at offset 17 from a 256-bit input.
        # This spans 4 source words.
        slice_expr = Slice(x=Var("x"), offset=17, width=200)
        ir = TickIR(
            name="slice200",
            inputs={"x": BitVecType(256)},
            outputs={"o": BitVecType(200)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": slice_expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 4)

        for _ in range(50):
            xv = random.getrandbits(256)
            inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
            outs = eval_packed_circuit_words(circuit, inputs)
            expect = (xv >> 17) & ((1 << 200) - 1)
            reconstructed = 0
            for i, w in enumerate(outs):
                reconstructed |= w << (i * 64)
            reconstructed &= (1 << 200) - 1
            self.assertEqual(reconstructed, expect)

    def test_unaligned_slice_aligned_to_word_boundary(self) -> None:
        # Edge case: offset is word-aligned (offset % 64 == 0) but width is partial.
        # Should still work without gathering from multiple words.
        slice_expr = Slice(x=Var("x"), offset=64, width=80)
        ir = TickIR(
            name="slice_aligned",
            inputs={"x": BitVecType(192)},
            outputs={"o": BitVecType(80)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": slice_expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 3)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            xv = random.getrandbits(192)
            inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(3)]
            outs = eval_packed_circuit_words(circuit, inputs)
            expect = (xv >> 64) & ((1 << 80) - 1)
            out_w0, out_w1 = outs
            reconstructed = out_w0 | (out_w1 << 64)
            reconstructed &= (1 << 80) - 1
            self.assertEqual(reconstructed, expect)

    def test_unaligned_slice_at_word_boundaries(self) -> None:
        # Test various unaligned offsets at and near word boundaries.
        test_cases = [
            (128, 0, 96),
            (128, 13, 96),
            (256, 63, 128),
            (256, 65, 127),
            (192, 32, 128),
        ]
        for src_width, offset, slice_width in test_cases:
            with self.subTest(
                src_width=src_width, offset=offset, slice_width=slice_width
            ):
                slice_expr = Slice(x=Var("x"), offset=offset, width=slice_width)
                ir = TickIR(
                    name="slice_test",
                    inputs={"x": BitVecType(src_width)},
                    outputs={"o": BitVecType(slice_width)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": slice_expr},
                )
                circuit, _layout = lower_tick_ir_to_packed_circuit_state(ir)

                for _ in range(10):
                    xv = random.getrandbits(src_width)
                    nwords = (src_width + 63) // 64
                    inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(nwords)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    expect = (xv >> offset) & ((1 << slice_width) - 1)
                    reconstructed = 0
                    for i, w in enumerate(outs):
                        reconstructed |= w << (i * 64)
                    reconstructed &= (1 << slice_width) - 1
                    self.assertEqual(reconstructed, expect)

    def test_concat_non_word_aligned_simple(self) -> None:
        # Concat of 17-bit + 43-bit -> 60 bits total (fits in 1 word).
        # Test: concat(a[17], b[43]) where a is 17 bits, b is 43 bits.
        # Result should be: a in bits [43:60), b in bits [0:43).
        ir = TickIR(
            name="concat_mixed",
            inputs={"a": BitVecType(17), "b": BitVecType(43)},
            outputs={"o": BitVecType(60)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Concat(parts=[Var("a"), Var("b")])},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)

        for _ in range(50):
            av = random.getrandbits(17)
            bv = random.getrandbits(43)
            inputs = [av, bv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (av << 43) | bv
            self.assertEqual(outs[0], expected & ((1 << 60) - 1))

    def test_concat_non_word_aligned_spanning_words(self) -> None:
        # Concat of 17-bit + 43-bit + 22-bit -> 82 bits total (needs 2 words).
        # Parts: [a=17, b=43, c=22]
        # Layout in bits: c[0:22), b[22:65), a[65:82)
        ir = TickIR(
            name="concat_span",
            inputs={"a": BitVecType(17), "b": BitVecType(43), "c": BitVecType(22)},
            outputs={"o": BitVecType(82)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Concat(parts=[Var("a"), Var("b"), Var("c")])},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 3)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            av = random.getrandbits(17)
            bv = random.getrandbits(43)
            cv = random.getrandbits(22)
            inputs = [av, bv, cv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (av << 65) | (bv << 22) | cv
            self.assertEqual(outs[0], expected & ((1 << 64) - 1))
            self.assertEqual(outs[1], (expected >> 64) & ((1 << 18) - 1))

    def test_concat_mixed_with_word_aligned(self) -> None:
        # Mix of word-aligned (64-bit) and non-aligned parts.
        # Parts: [a=64, b=17, c=11] -> 92 bits total (needs 2 words).
        ir = TickIR(
            name="concat_mixed_aligned",
            inputs={"a": BitVecType(64), "b": BitVecType(17), "c": BitVecType(11)},
            outputs={"o": BitVecType(92)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Concat(parts=[Var("a"), Var("b"), Var("c")])},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 3)
        self.assertEqual(layout.output_words, 2)

        for _ in range(50):
            av = random.getrandbits(64)
            bv = random.getrandbits(17)
            cv = random.getrandbits(11)
            inputs = [av, bv, cv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (av << 28) | (bv << 11) | cv
            self.assertEqual(outs[0], expected & ((1 << 64) - 1))
            self.assertEqual(outs[1], (expected >> 64) & ((1 << 28) - 1))

    def test_concat_large_non_aligned_parts(self) -> None:
        # Concat of two 100-bit parts -> 200 bits total (needs 4 words).
        # Each 100-bit input spans 2 words.
        ir = TickIR(
            name="concat_large",
            inputs={"a": BitVecType(100), "b": BitVecType(100)},
            outputs={"o": BitVecType(200)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Concat(parts=[Var("a"), Var("b")])},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 4)

        for _ in range(50):
            av = random.getrandbits(100)
            bv = random.getrandbits(100)
            inputs = [
                av & ((1 << 64) - 1),
                (av >> 64) & ((1 << 36) - 1),
                bv & ((1 << 64) - 1),
                (bv >> 64) & ((1 << 36) - 1),
            ]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (av << 100) | bv
            self.assertEqual(outs[0], expected & ((1 << 64) - 1))
            self.assertEqual(outs[1], (expected >> 64) & ((1 << 64) - 1))
            self.assertEqual(outs[2], (expected >> 128) & ((1 << 64) - 1))
            self.assertEqual(outs[3], (expected >> 192) & ((1 << 8) - 1))

    def test_concat_single_bit_parts(self) -> None:
        # Concat of three 1-bit parts -> 3 bits total.
        # Parts: [a=1, b=1, c=1] -> a in bit 2, b in bit 1, c in bit 0.
        ir = TickIR(
            name="concat_bits",
            inputs={"a": BitVecType(1), "b": BitVecType(1), "c": BitVecType(1)},
            outputs={"o": BitVecType(3)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Concat(parts=[Var("a"), Var("b"), Var("c")])},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 3)
        self.assertEqual(layout.output_words, 1)

        for _ in range(8):
            av = random.randint(0, 1)
            bv = random.randint(0, 1)
            cv = random.randint(0, 1)
            inputs = [av, bv, cv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (av << 2) | (bv << 1) | cv
            self.assertEqual(outs[0], expected)

    def test_concat_all_odd_widths(self) -> None:
        # Various odd widths: 7, 13, 19, 23 -> 62 bits total (fits in 1 word).
        ir = TickIR(
            name="concat_odd",
            inputs={
                "a": BitVecType(7),
                "b": BitVecType(13),
                "c": BitVecType(19),
                "d": BitVecType(23),
            },
            outputs={"o": BitVecType(62)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Concat(parts=[Var("a"), Var("b"), Var("c"), Var("d")])},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 1)

        for _ in range(50):
            av = random.getrandbits(7)
            bv = random.getrandbits(13)
            cv = random.getrandbits(19)
            dv = random.getrandbits(23)
            inputs = [av, bv, cv, dv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = (av << 55) | (bv << 42) | (cv << 23) | dv
            self.assertEqual(outs[0], expected)

    def test_rotl_single_word_64bit(self) -> None:
        # 64-bit rotate left by various amounts (single word).
        test_amounts = [0, 1, 7, 13, 31, 32, 63]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotl(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotl64",
                    inputs={"x": BitVecType(64)},
                    outputs={"o": BitVecType(64)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 1)
                self.assertEqual(layout.output_words, 1)

                for _ in range(20):
                    xv = random.getrandbits(64)
                    inputs = [xv]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 64) - 1
                    expected = ((xv << sh_amt) | (xv >> (64 - sh_amt))) & mask
                    self.assertEqual(outs[0], expected)

    def test_rotr_single_word_64bit(self) -> None:
        # 64-bit rotate right by various amounts (single word).
        test_amounts = [0, 1, 7, 13, 31, 32, 63]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotr(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotr64",
                    inputs={"x": BitVecType(64)},
                    outputs={"o": BitVecType(64)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 1)
                self.assertEqual(layout.output_words, 1)

                for _ in range(20):
                    xv = random.getrandbits(64)
                    inputs = [xv]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 64) - 1
                    expected = ((xv >> sh_amt) | (xv << (64 - sh_amt))) & mask
                    self.assertEqual(outs[0], expected)

    def test_rotl_multi_word_128bit(self) -> None:
        # 128-bit rotate left (2 words).
        test_amounts = [0, 1, 7, 13, 32, 63, 64, 65, 96, 127]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotl(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotl128",
                    inputs={"x": BitVecType(128)},
                    outputs={"o": BitVecType(128)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 2)
                self.assertEqual(layout.output_words, 2)

                for _ in range(20):
                    xv = random.getrandbits(128)
                    inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 128) - 1
                    expected = ((xv << sh_amt) | (xv >> (128 - sh_amt))) & mask
                    reconstructed = outs[0] | (outs[1] << 64)
                    self.assertEqual(reconstructed, expected)

    def test_rotr_multi_word_128bit(self) -> None:
        # 128-bit rotate right (2 words).
        test_amounts = [0, 1, 7, 13, 32, 63, 64, 65, 96, 127]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotr(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotr128",
                    inputs={"x": BitVecType(128)},
                    outputs={"o": BitVecType(128)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 2)
                self.assertEqual(layout.output_words, 2)

                for _ in range(20):
                    xv = random.getrandbits(128)
                    inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 128) - 1
                    expected = ((xv >> sh_amt) | (xv << (128 - sh_amt))) & mask
                    reconstructed = outs[0] | (outs[1] << 64)
                    self.assertEqual(reconstructed, expected)

    def test_rotl_multi_word_256bit(self) -> None:
        # 256-bit rotate left (4 words).
        test_amounts = [0, 1, 17, 32, 63, 64, 65, 128, 192, 200, 255]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotl(x=Var("x"), sh=BitVecConst(width=16, value=sh_amt))
                ir = TickIR(
                    name="rotl256",
                    inputs={"x": BitVecType(256)},
                    outputs={"o": BitVecType(256)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 4)
                self.assertEqual(layout.output_words, 4)

                for _ in range(20):
                    xv = random.getrandbits(256)
                    inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 256) - 1
                    expected = ((xv << sh_amt) | (xv >> (256 - sh_amt))) & mask
                    reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
                    self.assertEqual(reconstructed, expected)

    def test_rotr_multi_word_256bit(self) -> None:
        # 256-bit rotate right (4 words).
        test_amounts = [0, 1, 17, 32, 63, 64, 65, 128, 192, 200, 255]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotr(x=Var("x"), sh=BitVecConst(width=16, value=sh_amt))
                ir = TickIR(
                    name="rotr256",
                    inputs={"x": BitVecType(256)},
                    outputs={"o": BitVecType(256)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 4)
                self.assertEqual(layout.output_words, 4)

                for _ in range(20):
                    xv = random.getrandbits(256)
                    inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 256) - 1
                    expected = ((xv >> sh_amt) | (xv << (256 - sh_amt))) & mask
                    reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
                    self.assertEqual(reconstructed, expected)

    def test_rotl_non_word_aligned_width(self) -> None:
        # Rotate a 100-bit value (spans 2 words but not word-aligned).
        test_amounts = [0, 1, 7, 32, 63, 64, 99]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotl(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotl100",
                    inputs={"x": BitVecType(100)},
                    outputs={"o": BitVecType(100)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 2)
                self.assertEqual(layout.output_words, 2)

                for _ in range(20):
                    xv = random.getrandbits(100)
                    inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 36) - 1)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 100) - 1
                    expected = ((xv << sh_amt) | (xv >> (100 - sh_amt))) & mask
                    reconstructed = outs[0] | (outs[1] << 64)
                    reconstructed &= mask
                    self.assertEqual(reconstructed, expected)

    def test_rotr_non_word_aligned_width(self) -> None:
        # Rotate a 200-bit value (spans 4 words but not word-aligned).
        test_amounts = [0, 1, 13, 64, 127, 128, 199]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotr(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotr200",
                    inputs={"x": BitVecType(200)},
                    outputs={"o": BitVecType(200)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 4)
                self.assertEqual(layout.output_words, 4)

                for _ in range(20):
                    xv = random.getrandbits(200)
                    inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 200) - 1
                    expected = ((xv >> sh_amt) | (xv << (200 - sh_amt))) & mask
                    reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
                    reconstructed &= mask
                    self.assertEqual(reconstructed, expected)

    def test_rotl_word_boundary_shifts(self) -> None:
        # Test rotate by exact word boundaries and near-word-boundaries.
        # 192-bit value with shifts at 63, 64, 65, 127, 128, 129.
        test_amounts = [63, 64, 65, 127, 128, 129]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotl(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotl192",
                    inputs={"x": BitVecType(192)},
                    outputs={"o": BitVecType(192)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 3)
                self.assertEqual(layout.output_words, 3)

                for _ in range(20):
                    xv = random.getrandbits(192)
                    inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(3)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 192) - 1
                    expected = ((xv << sh_amt) | (xv >> (192 - sh_amt))) & mask
                    reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
                    self.assertEqual(reconstructed, expected)

    def test_rotr_word_boundary_shifts(self) -> None:
        # Test rotate by exact word boundaries and near-word-boundaries.
        # 192-bit value with shifts at 63, 64, 65, 127, 128, 129.
        test_amounts = [63, 64, 65, 127, 128, 129]
        for sh_amt in test_amounts:
            with self.subTest(sh_amt=sh_amt):
                expr = Rotr(x=Var("x"), sh=BitVecConst(width=8, value=sh_amt))
                ir = TickIR(
                    name="rotr192",
                    inputs={"x": BitVecType(192)},
                    outputs={"o": BitVecType(192)},
                    state={},
                    reset_state={},
                    next_state={},
                    output_exprs={"o": expr},
                )
                circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
                self.assertEqual(layout.input_words, 3)
                self.assertEqual(layout.output_words, 3)

                for _ in range(20):
                    xv = random.getrandbits(192)
                    inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(3)]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    mask = (1 << 192) - 1
                    expected = ((xv >> sh_amt) | (xv << (192 - sh_amt))) & mask
                    reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
                    self.assertEqual(reconstructed, expected)

    def test_rotate_rejects_non_const_shift(self) -> None:
        # Dynamic rotation amount should be rejected.
        expr = Rotl(x=Var("x"), sh=Var("s"))
        ir = TickIR(
            name="rotl_dynamic",
            inputs={"x": BitVecType(64), "s": BitVecType(8)},
            outputs={},
            state={"st": BitVecType(64)},
            reset_state={"st": BitVecConst(width=64, value=0)},
            next_state={"st": expr},
            output_exprs={},
        )
        with self.assertRaises(PackedLoweringError):
            lower_tick_ir_to_packed_circuit_state(ir)

    def test_rotate_modulo_width(self) -> None:
        # Test that rotation amounts > width are handled via modulo.
        # 64-bit rotate left by 65 should be equivalent to rotate by 1.
        expr = Rotl(x=Var("x"), sh=BitVecConst(width=8, value=65))
        ir = TickIR(
            name="rotl_mod",
            inputs={"x": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 1)
        self.assertEqual(layout.output_words, 1)

        for _ in range(20):
            xv = random.getrandbits(64)
            inputs = [xv]
            outs = eval_packed_circuit_words(circuit, inputs)
            mask = (1 << 64) - 1
            # 65 % 64 = 1
            expected = ((xv << 1) | (xv >> 63)) & mask
            self.assertEqual(outs[0], expected)

    def test_eq_single_word_equal(self) -> None:
        # Test Eq with single-word (64-bit) values that are equal.
        expr = Eq(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="eq64_equal",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(1)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)
        # Test equal values
        xv = 0x123456789ABCDEF0
        inputs = [xv, xv]
        outs = eval_packed_circuit_words(circuit, inputs)
        self.assertEqual(outs[0], 1)

    def test_eq_single_word_not_equal(self) -> None:
        # Test Eq with single-word (64-bit) values that are different.
        expr = Eq(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="eq64_not_equal",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(1)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)
        # Test different values
        for _ in range(50):
            xv = random.getrandbits(64)
            yv = random.getrandbits(64)
            inputs = [xv, yv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = 1 if xv == yv else 0
            self.assertEqual(outs[0], expected)

    def test_eq_multi_word_128bit(self) -> None:
        # Test Eq with 128-bit values (2 words).
        expr = Eq(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="eq128",
            inputs={"x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(1)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 1)
        for _ in range(50):
            xv = random.getrandbits(128)
            yv = random.getrandbits(128)
            x_inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
            y_inputs = [yv & ((1 << 64) - 1), (yv >> 64) & ((1 << 64) - 1)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = 1 if xv == yv else 0
            self.assertEqual(outs[0], expected)

    def test_eq_multi_word_256bit(self) -> None:
        # Test Eq with 256-bit values (4 words).
        expr = Eq(a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="eq256",
            inputs={"x": BitVecType(256), "y": BitVecType(256)},
            outputs={"o": BitVecType(1)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 8)
        self.assertEqual(layout.output_words, 1)
        for _ in range(50):
            xv = random.getrandbits(256)
            yv = random.getrandbits(256)
            x_inputs = [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
            y_inputs = [(yv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
            inputs = x_inputs + y_inputs
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = 1 if xv == yv else 0
            self.assertEqual(outs[0], expected)

    def test_eq_always_true(self) -> None:
        # Test Eq comparing a value with itself (always true).
        expr = Eq(a=Var("x"), b=Var("x"))
        ir = TickIR(
            name="eq_self",
            inputs={"x": BitVecType(128)},
            outputs={"o": BitVecType(1)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)
        for _ in range(50):
            xv = random.getrandbits(128)
            inputs = [xv & ((1 << 64) - 1), (xv >> 64) & ((1 << 64) - 1)]
            outs = eval_packed_circuit_words(circuit, inputs)
            self.assertEqual(outs[0], 1)

    def test_mux_const_cond_true(self) -> None:
        # Test Mux with constant true condition (selects a).
        expr = Mux(cond=BoolConst(True), a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="mux_true",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)
        for _ in range(50):
            xv = random.getrandbits(64)
            yv = random.getrandbits(64)
            inputs = [xv, yv]
            outs = eval_packed_circuit_words(circuit, inputs)
            self.assertEqual(outs[0], xv)

    def test_mux_const_cond_false(self) -> None:
        # Test Mux with constant false condition (selects b).
        expr = Mux(cond=BoolConst(False), a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="mux_false",
            inputs={"x": BitVecType(64), "y": BitVecType(64)},
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 2)
        self.assertEqual(layout.output_words, 1)
        for _ in range(50):
            xv = random.getrandbits(64)
            yv = random.getrandbits(64)
            inputs = [xv, yv]
            outs = eval_packed_circuit_words(circuit, inputs)
            self.assertEqual(outs[0], yv)

    def test_mux_with_eq_condition(self) -> None:
        # Test Mux with Eq as condition.
        # If x == y, select z; otherwise select w.
        eq_expr = Eq(a=Var("x"), b=Var("y"))
        expr = Mux(cond=eq_expr, a=Var("z"), b=Var("w"))
        ir = TickIR(
            name="mux_eq",
            inputs={
                "x": BitVecType(64),
                "y": BitVecType(64),
                "z": BitVecType(64),
                "w": BitVecType(64),
            },
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 4)
        self.assertEqual(layout.output_words, 1)
        for _ in range(50):
            xv = random.getrandbits(64)
            yv = random.getrandbits(64)
            zv = random.getrandbits(64)
            wv = random.getrandbits(64)
            # Inputs must be in alphabetical order: w, x, y, z
            inputs = [wv, xv, yv, zv]
            outs = eval_packed_circuit_words(circuit, inputs)
            expected = zv if xv == yv else wv
            self.assertEqual(outs[0], expected)

    def test_mux_multi_word_128bit(self) -> None:
        # Test Mux with 128-bit branches (2 words each).
        expr = Mux(cond=Var("c"), a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="mux128",
            inputs={"c": BitVecType(1), "x": BitVecType(128), "y": BitVecType(128)},
            outputs={"o": BitVecType(128)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 5)
        self.assertEqual(layout.output_words, 2)
        for cond in [0, 1]:
            for _ in range(25):
                xv = random.getrandbits(128)
                yv = random.getrandbits(128)
                inputs = [
                    cond,
                    xv & ((1 << 64) - 1),
                    (xv >> 64) & ((1 << 64) - 1),
                    yv & ((1 << 64) - 1),
                    (yv >> 64) & ((1 << 64) - 1),
                ]
                outs = eval_packed_circuit_words(circuit, inputs)
                expected = xv if cond == 1 else yv
                reconstructed = outs[0] | (outs[1] << 64)
                self.assertEqual(reconstructed, expected)

    def test_mux_multi_word_256bit(self) -> None:
        # Test Mux with 256-bit branches (4 words each).
        expr = Mux(cond=Var("c"), a=Var("x"), b=Var("y"))
        ir = TickIR(
            name="mux256",
            inputs={"c": BitVecType(1), "x": BitVecType(256), "y": BitVecType(256)},
            outputs={"o": BitVecType(256)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": expr},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 9)
        self.assertEqual(layout.output_words, 4)
        for cond in [0, 1]:
            for _ in range(25):
                xv = random.getrandbits(256)
                yv = random.getrandbits(256)
                inputs = (
                    [cond]
                    + [(xv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
                    + [(yv >> (i * 64)) & ((1 << 64) - 1) for i in range(4)]
                )
                outs = eval_packed_circuit_words(circuit, inputs)
                expected = xv if cond == 1 else yv
                reconstructed = sum(w << (i * 64) for i, w in enumerate(outs))
                self.assertEqual(reconstructed, expected)

    def test_mux_nested(self) -> None:
        # Test nested Mux: Mux(c1, Mux(c2, a, b), d)
        inner_mux = Mux(cond=Var("c2"), a=Var("a"), b=Var("b"))
        outer_mux = Mux(cond=Var("c1"), a=inner_mux, b=Var("d"))
        ir = TickIR(
            name="mux_nested",
            inputs={
                "c1": BitVecType(1),
                "c2": BitVecType(1),
                "a": BitVecType(64),
                "b": BitVecType(64),
                "d": BitVecType(64),
            },
            outputs={"o": BitVecType(64)},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": outer_mux},
        )
        circuit, layout = lower_tick_ir_to_packed_circuit_state(ir)
        self.assertEqual(layout.input_words, 5)
        self.assertEqual(layout.output_words, 1)
        for c1 in [0, 1]:
            for c2 in [0, 1]:
                for _ in range(10):
                    av = random.getrandbits(64)
                    bv = random.getrandbits(64)
                    dv = random.getrandbits(64)
                    # Inputs must be in alphabetical order: a, b, c1, c2, d
                    inputs = [av, bv, c1, c2, dv]
                    outs = eval_packed_circuit_words(circuit, inputs)
                    inner_result = av if c2 == 1 else bv
                    expected = inner_result if c1 == 1 else dv
                    self.assertEqual(outs[0], expected)


if __name__ == "__main__":
    unittest.main()
