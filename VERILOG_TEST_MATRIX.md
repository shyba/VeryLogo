# Verilog → Tick-IR Instruction Test Matrix

This document enumerates every Tick-IR expression node currently defined in `stc/tick_ir.py` and proposes a minimal Verilog test (when possible) that compiles to that node.

Scope notes:

- The current Verilog frontend is intentionally small (see `stc/subset.py` and `stc/extract.py`). Many Tick-IR nodes are IR-only or are only produced by later passes.
- SIMD nodes are generally reached by compiling scalar, lane-wise Verilog and running `stc` with `--infer-simd --autovec` (or future extensions). The AVR backend does not support SIMD; use `--no-backend`.
- Float nodes are not currently representable from the Verilog subset; they require Tick-IR fixtures (or a future float frontend).

Columns:

- `Produced by`: pipeline stage where the node is expected to appear.
- `Verilog trigger`: a 1-line sketch of the construct to place inside a small single-module test.
- `CLI flags`: minimal flags needed for the node to appear.
- `Expected in`: which artifact should contain the node (`tick_ir.bin` vs `reduced_tick_ir.bin`).
- `Status`: whether the case is reachable today or what is missing to make it reachable from Verilog.

| Category | Tick-IR node | kind | Produced by | Verilog trigger | CLI flags | Expected in | Status |
|---|---|---:|---|---|---|---|---|
| Core | `Var` | `var` | Frontend | `assign y = a;` | none | `tick_ir.bin` | Reachable |
| Const | `BoolConst` | `bool_const` | Frontend | `assign y = 1'b1;` | none | `tick_ir.bin` | Reachable |
| Const | `BitVecConst` | `bitvec_const` | Reducer | `assign y = 8'hA5;` | none | `reduced_tick_ir.bin` | Reachable (via constant folding) |
| Const | `FloatConst` | `float_const` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Const | `SimdConst` | `simd_const` | Infer-SIMD | `always_ff ... if (rst) q <= '0;` with SIMD-typed `q` | `--infer-simd` | `tick_ir.bin` | Reachable when SIMD inferred for a register |
| Structural | `Slice` | `slice` | Frontend | `assign y = a[7:0];` | none | `tick_ir.bin` | Reachable |
| Structural | `Concat` | `concat` | Frontend | `assign y = {a[3:0], b[3:0]};` | none | `tick_ir.bin` | Reachable |
| Structural | `Mux` | `mux` | Frontend | `assign y = sel ? a : b;` | none | `tick_ir.bin` | Reachable |
| Structural | `Bitcast` | `bitcast` | Infer-SIMD | lane-wise concat output inferred as SIMD, wrapped by bitcast | `--infer-simd` | `tick_ir.bin` | Reachable (via `infer_simd_types`) |
| Structural | `Delay` | `delay` | IR-only | N/A (no Verilog→`Delay` recognition) | N/A | N/A | Blocked (needs reg-chain→Delay recognition or IR fixture) |
| BV | `Not` | `not` | Frontend | `assign y = ~a;` | none | `tick_ir.bin` | Reachable |
| BV | `And` | `and` | Frontend | `assign y = a & b;` | none | `tick_ir.bin` | Reachable |
| BV | `Or` | `or` | Frontend | `assign y = a \| b;` | none | `tick_ir.bin` | Reachable |
| BV | `Xor` | `xor` | Frontend | `assign y = a ^ b;` | none | `tick_ir.bin` | Reachable |
| BV | `Add` | `add` | Frontend | `assign y = a + b;` | none | `tick_ir.bin` | Reachable |
| BV | `Sub` | `sub` | Frontend | `assign y = a - b;` | none | `tick_ir.bin` | Reachable |
| BV | `Shl` | `shl` | Frontend | `assign y = a << sh;` | none | `tick_ir.bin` | Reachable |
| BV | `LShr` | `lshr` | Frontend | `assign y = a >> sh;` | none | `tick_ir.bin` | Reachable |
| BV | `AShr` | `ashr` | Frontend | `assign y = $signed(a) >>> sh;` | none | `tick_ir.bin` | Reachable |
| BV | `Eq` | `eq` | Frontend | `assign y = (a == b);` | none | `tick_ir.bin` | Reachable |
| BV | `Ult` | `ult` | Frontend | `assign y = (a < b);` | none | `tick_ir.bin` | Reachable (unsigned only) |
| BV | `Ule` | `ule` | Frontend | `assign y = (a <= b);` | none | `tick_ir.bin` | Reachable (unsigned only) |
| BV | `Ugt` | `ugt` | Frontend | `assign y = (a > b);` | none | `tick_ir.bin` | Reachable (unsigned only) |
| BV | `Uge` | `uge` | Frontend | `assign y = (a >= b);` | none | `tick_ir.bin` | Reachable (unsigned only) |
| Float | `FNeg` | `fneg` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FAbs` | `fabs` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FAdd` | `fadd` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FSub` | `fsub` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FMul` | `fmul` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FDiv` | `fdiv` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FSqrt` | `fsqrt` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FEq` | `feq` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FLt` | `flt` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FLe` | `fle` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| Float | `FNe` | `fne` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked (needs float frontend or IR fixture) |
| SIMD-int | `SimdAdd` | `simd_add` | Autovec | lane-wise `+` on `W`-bit lanes, concat into wide output | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable for regular add/sub/bitwise |
| SIMD-int | `SimdSub` | `simd_sub` | Autovec | lane-wise `-` on `W`-bit lanes, concat into wide output | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable |
| SIMD-int | `SimdAddMasked` | `simd_add_masked` | IR-only | `out = mask ? (x+y) : x` (e.g. `SimdBlend(mask, x, SimdAdd(x,y))`) | N/A | N/A | Reachable only via IR fixtures or an explicit lowering from `SimdBlend` patterns |
| SIMD-int | `SimdSubMasked` | `simd_sub_masked` | IR-only | `out = mask ? (x-y) : x` (e.g. `SimdBlend(mask, x, SimdSub(x,y))`) | N/A | N/A | Reachable only via IR fixtures or an explicit lowering from `SimdBlend` patterns |
| SIMD-int | `SimdAnd` | `simd_and` | Autovec | lane-wise `&` on lanes, concat into wide output | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable |
| SIMD-int | `SimdOr` | `simd_or` | Autovec | lane-wise `\|` on lanes, concat into wide output | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable |
| SIMD-int | `SimdXor` | `simd_xor` | Autovec | lane-wise `^` on lanes, concat into wide output | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable |
| SIMD-int | `SimdEq` | `simd_eq` | Autovec | lane-wise `==` producing 1-bit lanes, concat mask bits | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable (mask vectors) |
| SIMD-int | `SimdUlt` | `simd_ult` | Autovec | lane-wise `<` producing 1-bit lanes, concat mask bits | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable (unsigned only) |
| SIMD-int | `SimdUle` | `simd_ule` | Autovec | lane-wise `<=` producing 1-bit lanes, concat mask bits | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable (unsigned only) |
| SIMD-int | `SimdUgt` | `simd_ugt` | Autovec | lane-wise `>` producing 1-bit lanes, concat mask bits | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable (unsigned only) |
| SIMD-int | `SimdUge` | `simd_uge` | Autovec | lane-wise `>=` producing 1-bit lanes, concat mask bits | `--infer-simd --autovec --no-backend` | `reduced_tick_ir.bin` | Reachable (unsigned only) |
| SIMD-int | `SimdNot` | `simd_not` | IR-only | lane-wise `~` (would require new autovec rule) | N/A | N/A | Blocked (extend autovec to emit `SimdNot`) |
| SIMD-int | `SimdShl` | `simd_shl` | IR-only | lane-wise `<<` (would require new autovec rule) | N/A | N/A | Blocked (extend autovec to emit `SimdShl`) |
| SIMD-int | `SimdLShr` | `simd_lshr` | IR-only | lane-wise `>>` (would require new autovec rule) | N/A | N/A | Blocked (extend autovec to emit `SimdLShr`) |
| SIMD-int | `SimdAShr` | `simd_ashr` | IR-only | lane-wise signed `>>>` (would require new autovec rule + signedness spec) | N/A | N/A | Blocked (extend autovec + define signedness) |
| SIMD-int | `SimdSlt` | `simd_slt` | IR-only | lane-wise signed `<` mask (requires signedness spec + autovec rule) | N/A | N/A | Blocked |
| SIMD-int | `SimdSle` | `simd_sle` | IR-only | lane-wise signed `<=` mask (requires signedness spec + autovec rule) | N/A | N/A | Blocked |
| SIMD-int | `SimdSgt` | `simd_sgt` | IR-only | lane-wise signed `>` mask (requires signedness spec + autovec rule) | N/A | N/A | Blocked |
| SIMD-int | `SimdSge` | `simd_sge` | IR-only | lane-wise signed `>=` mask (requires signedness spec + autovec rule) | N/A | N/A | Blocked |
| SIMD-int | `SimdAddSatU` | `simd_add_satu` | IR-only | saturating `u8/u16` lane spec (requires new autovec/superopt support) | N/A | N/A | Blocked (no generator pass) |
| SIMD-int | `SimdSubSatU` | `simd_sub_satu` | IR-only | saturating `u8/u16` lane spec (requires new autovec/superopt support) | N/A | N/A | Blocked (no generator pass) |
| SIMD-int | `SimdAddSatS` | `simd_add_sats` | IR-only | saturating `s8/s16` lane spec (requires new autovec/superopt support) | N/A | N/A | Blocked (no generator pass) |
| SIMD-int | `SimdSubSatS` | `simd_sub_sats` | IR-only | saturating `s8/s16` lane spec (requires new autovec/superopt support) | N/A | N/A | Blocked (no generator pass) |
| SIMD-int | `SimdMulLo` | `simd_mul_lo` | IR-only | lane-wise `*` (blocked: Verilog subset lacks `$mul`, autovec lacks rule) | N/A | N/A | Blocked (add `$mul` + autovec rule) |
| SIMD-int | `SimdMulHiU` | `simd_mul_hi_u` | IR-only | high-half multiply (requires new frontend ops + generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMulHiS` | `simd_mul_hi_s` | IR-only | high-half signed multiply (requires new signedness spec + generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMaddS16` | `simd_madd_s16` | IR-only | `(a*s16)*(b*s16)` pairwise add (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMinU` | `simd_min_u` | IR-only | lane-wise `min` (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMaxU` | `simd_max_u` | IR-only | lane-wise `max` (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMinS` | `simd_min_s` | IR-only | lane-wise signed `min` (requires signedness spec + generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMaxS` | `simd_max_s` | IR-only | lane-wise signed `max` (requires signedness spec + generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdBlend` | `simd_blend` | IR-only | lane-wise `sel ? b : a` on a mask vector (needs explicit mask type + generator pass) | N/A | N/A | Blocked (not produced from Verilog today) |
| SIMD-int | `SimdZExtLo` | `simd_zext_lo` | IR-only | widen low half lanes with zero extend (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdSExtLo` | `simd_sext_lo` | IR-only | widen low half lanes with sign extend (requires signedness spec + generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdUnpackLo` | `simd_unpack_lo` | IR-only | interleave low halves (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdUnpackHi` | `simd_unpack_hi` | IR-only | interleave high halves (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdPackSS16To8` | `simd_pack_ss16_to_8` | IR-only | pack with signed saturation (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdPackUS16To8` | `simd_pack_us16_to_8` | IR-only | pack with unsigned saturation (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdPackSS32To16` | `simd_pack_ss32_to_16` | IR-only | pack with signed saturation (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMaskExpand` | `simd_mask_expand` | IR-only | mask-bitvector → lane mask (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdMaskPack` | `simd_mask_pack` | IR-only | lane mask → mask-bitvector (requires generator pass) | N/A | N/A | Blocked |
| SIMD-int | `SimdSplat` | `simd_splat` | IR-only | `assign y = {LANES{a[LANE-1:0]}};` | N/A | N/A | Blocked (extend autovec to emit `SimdSplat`) |
| SIMD-int | `SimdExtractLane` | `simd_extract_lane` | IR-only | `assign y = a[IDX*W +: W];` | N/A | N/A | Blocked (needs explicit lane-op lowering) |
| SIMD-int | `SimdInsertLane` | `simd_insert_lane` | IR-only | insert a lane into a packed bus | N/A | N/A | Blocked (needs explicit lane-op lowering) |
| SIMD-int | `SimdShuffle` | `simd_shuffle` | IR-only | fixed-permutation concat of lanes (e.g. reverse lanes) | N/A | N/A | Blocked (extend autovec to emit `SimdShuffle`) |
| SIMD-float | `SimdFAdd` | `simd_fadd` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFSub` | `simd_fsub` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFMul` | `simd_fmul` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFDiv` | `simd_fdiv` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFSqrt` | `simd_fsqrt` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFNeg` | `simd_fneg` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFAbs` | `simd_fabs` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFFma` | `simd_ffma` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFCmpEq` | `simd_fcmp_eq` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFCmpLt` | `simd_fcmp_lt` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFCmpLe` | `simd_fcmp_le` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |
| SIMD-float | `SimdFCmpNe` | `simd_fcmp_ne` | IR-only | N/A (no float Verilog subset) | N/A | N/A | Blocked |

## Fixtures and Tests

- `fixtures/verilog_matrix/basic_ops.v`: direct frontend cases for `$not/$and/$or/$xor/$add/$sub/$mux/$eq`.
- `tests/test_verilog_matrix_optional.py`: optional extraction test (requires `yosys`) verifying the expected Tick-IR node classes appear per output.
