# SSE Backend Plan (SSE2 First)

This plan extends the project with an x86 SIMD backend starting with SSE2 integer operations. It is test-driven and keeps `./scripts/test.sh` green.

## Rules

- All changes are TDD-first: add a failing test, implement the smallest change, keep tests green.
- No code comments.
- Black formatting.
- Focus on SSE2 integer semantics first. SSE1 floating point support is blocked until Tick-IR gains `f32/f64` types and IEEE semantics.

## Phase 1 — Infrastructure (SSE2)

- [x] Add `stc/backend_x86_sse2.py` that emits C using `<emmintrin.h>` intrinsics.
- [x] Add an optional unit test that compiles and runs the emitted C on the host (`cc -msse2`) and checks outputs against Tick-IR interpreter results.
- [x] Limit initial backend scope to combinational Tick-IR (no state) and 128-bit SIMD values (`SimdType.total_width == 128`).

## Phase 2 — SSE2 Instruction Coverage (Integer)

Each item is a backend capability plus a test vector set (fixed inputs) verified against interpreter semantics.

### Bitwise

- [x] `pand` (`_mm_and_si128`) for `SimdAnd`
- [x] `por` (`_mm_or_si128`) for `SimdOr`
- [x] `pxor` (`_mm_xor_si128`) for `SimdXor`
- [x] `pandn` (`_mm_andnot_si128`) as a peephole for `SimdAnd(SimdNot(a), b)`

### Add/Sub (wrap)

- [x] `paddb/paddw/paddd/paddq` for `SimdAdd` (lane_width ∈ {8,16,32,64})
- [x] `psubb/psubw/psubd/psubq` for `SimdSub` (lane_width ∈ {8,16,32,64})

### Shifts

- [x] `pslld/psllw/psllq` for `SimdShl` (lane_width ∈ {16,32,64}, constant shift amount only)
- [x] `psrld/psrlw/psrlq` for `SimdLShr` (lane_width ∈ {16,32,64}, constant shift amount only)
- [x] `psrad/psraw` for `SimdAShr` (lane_width ∈ {16,32}, constant shift amount only)

### Compare

SSE2 compares produce per-lane all-ones/zeros, while Tick-IR mask vectors are `SimdType(lane_width=1, lanes=N)` packed into low bits. Mapping requires a mask-pack lowering.

- [x] `pcmpeqb/w/d` for `SimdEq` (mask-pack via `movemask`)
- [x] `pcmpgtb/w/d` for signed compares (`SimdSlt` via swapped `cmpgt`, mask-pack via `movemask`)
- [x] Unsigned compares for {8,16,32}-bit lanes (`SimdUlt` via XOR-bias + signed compare, mask-pack via `movemask`)

### Min/Max, Multiply, Pack/Unpack, Movemask

These require additional Tick-IR ops and/or backend lowering rules:

- [x] Unsigned saturating add/sub for u8/u16 (`paddusb/paddusw`, `psubusb/psubusw`) via `SimdAddSatU` / `SimdSubSatU`
- [x] Signed saturating add/sub for s8/s16 (`paddsb/paddsw`, `psubsb/psubsw`) via `SimdAddSatS` / `SimdSubSatS`
- [x] `pmullw/pmulhw/pmulhuw/pmaddwd` via `SimdMulLo` / `SimdMulHiS` / `SimdMulHiU` / `SimdMaddS16`
- [x] Pack/unpack (`pack*`, `punpck*`) via `SimdUnpackLo` / `SimdUnpackHi` and `SimdPack*` ops
- [x] `pmovmskb` as a first-class Tick-IR op (`SimdMaskPack`) and backend lowering for mask outputs

## Phase 3 — SSSE3/SSE4.1/SSE4.2 (Later)

- [x] SSSE3 `pshufb` (`SimdShuffle` lowering for `simd[8,16]`) (optional, requires `-mssse3`)
- [x] SSE4.1 blend/min/max/sign/zero-extend via `SimdBlend`, `SimdMin*/SimdMax*`, `SimdZExtLo`, `SimdSExtLo`
- [x] SSE4.2 string-compare ops are out of scope for Tick-IR
