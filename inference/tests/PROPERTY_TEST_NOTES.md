# Property-Based Test Notes (AVX-512 kernel stack)

Run: `.venv/bin/python -m pytest tests/test_properties_gemm.py tests/test_properties_simd.py -q`

Requires AVX512-VNNI + AVX512-BF16 (Zen 5); the suites skip on other CPUs.
hypothesis 6.165.3 and pytest 9 were added to `.venv` for this.

## Property inventory

### tests/test_properties_gemm.py
| test | property | status |
|---|---|---|
| test_vnni8_random_shapes | 8x32 vpdpbusd kernel: C == naive uint8-A x int8-B, random M(8..48) N(32..128) K(8..64) and seeds | PASS |
| test_vnni8_edge_shapes | same at boundary shapes (smallest supported K=8, non-square tiles) | PASS |
| test_bf16_random_shapes | 8x32 vdpbf16ps kernel: C == naive (mixed normal/denormal bf16 data), K(4..32) | PASS |
| test_bf16_edge_shapes | same at boundary shapes (K=4 minimum) | PASS |
| test_bf16_transpose_pack | gemm(A, pack_T(B)) == A @ B^T (the QK^T path) | PASS |
| test_multi_nr_shared_bp_incorrect | 8x16 kernel reading a 32-packed Bp must not equal naive | **XFAIL (real bug)** |
| test_aggen_predictions_positive | predicted_gemm_cycles > 0 | PASS |
| test_aggen_predictions_monotone_in_{m,n,k} | predictions grow with each dimension | PASS |
| test_best_tile_acceptable_to_generator | emit_gemm_kernel(*best_tile(family)) does not raise | PASS (fixed) |

### tests/test_properties_simd.py (functions copied verbatim from the decoder template)
| test | property | status |
|---|---|---|
| test_exp_properties | exp(0)=1; exp(a+b)=exp(a)exp(b); monotone; finite for |x|<=80 | PASS |
| test_softmax_properties | rows sum to 1; argmax preserved; shift-invariance softmax(x+c)=softmax(x) | PASS |
| test_layernorm_properties | output rows mean~0, var~1 | PASS |
| test_split_qkv_roundtrip | concat(split(x)) == x bit-exact | PASS |
| test_residual_signal | x + o keeps o's signal | PASS |
| test_f2b_roundtrip | b2f(f2b(x)) within bf16 rounding of x | PASS |

## Property results

- 16 pass, 1 xfail. The two properties below found real bugs; one is fixed,
  one remains latent and is guarded in the benchmark.

### 1. best_tile returned a tile the generator rejects (FIXED)
- Property: `emit_gemm_kernel(*best_tile("vnni8"))` must not raise.
- Failure found: `stc.aggen.best_tile` returned (12, 32) for both vnni8 and
  bf16, but `stc.gemm_asm.emit_gemm_kernel` rejects MR > 8 ("8 GPR row
  pointers"). The model's tile suggestion was not emittable.
- Fix: `inference/aggen.py` best_tile is now constrained to the generator's
  fixed-register allocation (MR <= 8, <= 16 accumulators, B-vector + temp
  budget) and returns 8x32 - the measured best tile. The test is now a
  hard property and passes.
  -> `ValueError: MR > 8 not supported (8 GPR row pointers)`.
- Impact: only the advisory tile-selection API; the benchmarks use explicit
  tiles. Fix would be constraining `best_tile` to MR <= 8.

### 2. Mixed-NR GEMM benchmark shares one Bp buffer packed with the last tile's NR
- Property: the 8x16 kernel run against a Bp packed with the 32-column
  layout must not silently equal the naive result.
- Failure: it does NOT equal naive - the 8x16 kernel reads chunks at the
  16-column stride from a 128-byte-stride buffer, so the second n-tile's
  columns are wrong (verified by hand: row0 col0 got -7814 vs naive
  -20849 with seed 1).
- Repro: `pytest tests/test_properties_gemm.py::test_multi_nr_shared_bp_incorrect -rxX`
  (shows XFAIL with the mismatch). Direct harness:
  `.venv/bin/python -c "import sys; sys.path.insert(0,'tests'); import test_properties_gemm as t; from property_harness import run_check; print(run_check(t._i8_bin(), [8,32,8,1,1]))"`
  -> prints `FAIL 0 0 ...` (the 8x16 kernel on the 32-packed buffer).
- Impact: in `inference/bench/bench_gemm_avx512.py`, `pack_b_i8_%d`/`pack_b_bf16_%d`
  in main pack `Bp8`/`Bp16` once with the LAST tile's NR, while each
  `run_i8_MRxNR` kernel reads its own NR layout. Running mixed-NR tiles
  (e.g. `--tiles 8x32,8x16`) therefore times garbage while verify passes
  (verify re-packs per tile with the correct NR) - i.e. bad=0 with
  meaningless timings. Fix: pack per tile-NR, or restrict runs to a single
  NR.

## Notes
- The checker harness generates its own data from a seed (xorshift64), so
  hypothesis shrinks counterexamples to (shape, seed); the C side guards
  seed 0 (degenerate xorshift).
- bf16 data mixes normals ([-1,1]) with ~6% denormals to exercise the
  smallest exponents.
- The exp/softmax/layernorm functions under test are the CURRENT (fixed)
  implementations; the reversed-Horner exp bug that previously lurked
  would fail test_exp_properties immediately (exp(0) != 1).
- The GEMM suite compiles the asm kernels via `stc.gemm_asm` and a small C
  checker once per process (cached in `tests/property_harness.py`).
