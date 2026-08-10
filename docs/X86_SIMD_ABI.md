# x86 SIMD C ABI

This project’s x86 SIMD backends emit a single C function:

`void stc_eval(const uint64_t* in, uint64_t* out);`

The ABI is word-based and deterministic so the generated C can be tested easily and called from any host code.

## Word Order

- Each SIMD value is stored in little-endian 64-bit words.
- Word 0 contains bits `[63:0]`, word 1 contains bits `[127:64]`, etc.

## Input Order

- Inputs are ordered lexicographically by input name (`sorted(ir.inputs.keys())`).
- Each input consumes a fixed number of 64-bit words based on the backend width:
  - SSE2 (`simd128`): 2 words per input
  - AVX (`simd256`, float ops): 4 words per input
  - AVX2 (`simd256`, integer ops): 4 words per input
  - AVX-512 (`simd512`): 8 words per input

## Output Order

- Outputs are ordered lexicographically by output name (`sorted(ir.outputs.keys())`).
- Each output consumes the same fixed number of 64-bit words as an input for that backend.

### Mask Outputs

Tick-IR mask vectors use `SimdType(lane_width=1, lanes=N)` and are encoded as an `N`-bit packed integer.

- The packed mask value is written to `out[word0]`.
- Remaining words for that output are set to 0.

#### Mask Bit Order

- Lane 0 is encoded as bit 0 (least significant bit).
- Lane `i` is encoded as bit `i`.

#### Mask Producers

- Compare ops (`SimdEq`, `SimdSlt`, `SimdUlt`) are lowered to per-lane compare intrinsics and then packed with `movemask`-style extraction.
- `SimdMaskPack(x)` is treated as the canonical “lane != 0” predicate and produces the same packed layout.

## AVX-512 Mask Notes

AVX-512 compare intrinsics natively return `__mmaskN` values (`N ∈ {8,16,32,64}`), where bit `i` corresponds to lane `i`.

- The backends still write mask results using the same packed convention: `out[word0] = (uint64_t)mask`.
- When `lanes < 64`, only the low `lanes` bits are defined; upper bits in `out[word0]` are ignored.
