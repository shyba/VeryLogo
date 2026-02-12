# AES10 BP128 CUDA Learnings (2026-02-10 / updated 2026-02-11)

This note records the current CUDA AES10 BP128 status after removing slower experimental paths and introducing a production-style runtime API.

## Modes

- `legacy` / `legacy_tuned`
  - Historical benchmark path.
  - Fixed plaintext + embedded round keys.
  - Fastest raw compute benchmark.
- `legacy_streamed` / `legacy_streamed_tuned`
  - Column-streamed in-place round body (no full `sb/sr/mc` arrays).
  - Keeps legacy ABI and benchmark semantics.
- `replacement` / `replacement_tuned`
  - Runtime `in_ptr/out_ptr` kernel ABI.
  - Round keys uploaded at runtime through CUDA module global `RK_BITS`.
  - Intended integration path for other CUDA projects.
- `replacement_coalesced` / `replacement_coalesced_tuned`
  - Runtime ABI with plane-major bitplane layout:
    - `in_ptr[plane * n_threads + tid]`
    - `out_ptr[plane * n_threads + tid]`
- `replacement_coalesced4` / `replacement_coalesced4_tuned`
  - Runtime ABI with plane-major packed `uint4` groups:
    - 32 groups per AES block (`2 groups/byte`)
    - each group stores 4 bitplanes as `{x,y,z,w}`
- `replacement_streamed` / `replacement_streamed_tuned`
  - Runtime ABI + column-streamed in-place round body.
  - Structural attempt to lower peak liveness while preserving API.
- `replacement_coalesced4_paramrk` / `replacement_coalesced4_paramrk_tuned`
  - Runtime ABI with explicit key pointer (`rk_bits` kernel param).
  - Keeps key material outside module symbols for easier per-key debugging.
- `replacement_coalesced4_paramrk_shared` / `replacement_coalesced4_paramrk_shared_tuned`
  - Explicit key pointer + per-block shared staging of 11x16x8 key bits.
  - Removes per-thread repeated global key fetches from the hot path for one-key-per-block workloads.
- `replacement_coalesced4_paramrk_soa` / `replacement_coalesced4_paramrk_soa_tuned`
  - Explicit key pointer where key bits are provided as SoA-by-thread: `rk_bits[(plane * threads) + tid]`.
  - Supports per-thread distinct keys with coalesced key loads.

## Current Findings

- Legacy remains the fastest microbenchmark because it avoids runtime input loads.
- Replacement is slower (expected) because each launch performs real input loads + output stores while keeping the same compute core.
- All current modes (including streamed variants) remain register-resident at compile time (`255` regs, `0` spills, `0` stack frame on this GPU).
- Streamed in-place did not reduce the register ceiling on sm_61 in current ptxas output; it only shifted instruction scheduling.
- Legacy-streamed is a small win over legacy on this GPU; replacement-streamed regressed vs replacement.
- Slower experimental variants were removed from the main flow to reduce noise.

## Recommended Defaults

- For integration as an AES-NI-style GPU replacement:
  - Use `replacement_coalesced4_tuned`.
  - This is the best current performance among runtime-ABI modes.
- If you need scalar `uint32_t` plane-major buffers (no `uint4` packing):
  - Use `replacement_coalesced_tuned`.
- For pure compute roofline / microbench:
  - Use `legacy_tuned`.
  - This is not an application ABI (fixed internal plaintext).
- Treat `*_streamed*` modes as experimental:
  - Correctness is good, but they do not currently beat the non-streamed replacement path.

## Why the streamed rewrite did not unlock a large win

- Expected benefit: remove full-round intermediates (`sb/sr/mc`) to reduce peak liveness.
- Observed on sm_61:
  - `ptxas -v` still reports `Used 255 registers` with `0 spills, 0 stack`.
  - PTX/SASS instruction mix remains near-identical to non-streamed variants.
- Practical conclusion:
  - Register pressure remains maxed (`255`) across modes.
  - Runtime-ABI throughput was primarily limited by IO layout/coalescing, not spills.
  - Coalesced layouts recover most/all of the runtime-vs-legacy gap without changing S-box logic.

## Benchmarks (GTX 1070, sm_61)

| Mode | Threads / Block / Reps | Throughput | Bandwidth |
|---|---:|---:|---:|
| `legacy_tuned` (best block=64) | `65536 / [64,128,256] / 200` | `1.759B eval/s` | `26845.81 MiB/s` |
| `legacy_streamed_tuned` (best block=64) | `65536 / [64,128,256] / 200` | `1.755B eval/s` | `26777.60 MiB/s` |
| `replacement_tuned` (best block=128) | `65536 / [64,128,256] / 200` | `0.744B eval/s` | `11359.42 MiB/s` |
| `replacement_coalesced_tuned` (best block=256) | `65536 / [64,128,256] / 200` | `0.928B eval/s` | `14157.92 MiB/s` |
| `replacement_coalesced4_tuned` (best block=64) | `65536 / [64,128,256] / 200` | `1.756B eval/s` | `26789.08 MiB/s` |
| `replacement_streamed_tuned` (best block=64) | `65536 / [64,128,256] / 200` | `1.025B eval/s` | `15641.00 MiB/s` |
| `legacy` + KAT | `256 / 128 / 2` | `0.076B eval/s` | `1157.06 MiB/s` |
| `replacement` + KAT | `256 / 128 / 2` | `0.092B eval/s` | `1403.11 MiB/s` |
| `replacement_coalesced4` + KAT | `65536 / 64 / 2` | `1.749B eval/s` | `26684.46 MiB/s` |


## Explicit RK Pointer Mode (small example)

For per-key debugging/integration, there is now an explicit RK kernel-parameter mode:

```bash
PYTHONPATH=. .venv/bin/python scripts/bench_aes10_bp128_cuda.py   --kernel-mode replacement_coalesced4_paramrk --threads 1024 --block 128 --reps 2 --check --ptxas-report
```

Observed on GTX 1070:

- Correctness: `PASS: AES-128 known-answer check matched`
- Resources: `206` registers, `0` spills, `0` stack frame
- Throughput at `65536` threads (`*_tuned`): `~1.369B eval/s` (`~20887 MiB/s`)
- With shared key staging (`replacement_coalesced4_paramrk_shared_tuned`): `~1.725B eval/s` (`~26317 MiB/s`).

Tradeoff:

- Explicit RK parameter is cleaner for per-key correctness tracing.
- Current symbol-based `replacement_coalesced4_tuned` remains faster (`~1.756B eval/s`).


## ParamRK shared staging fix (2026-02-12)

Problem reproduced:

- `replacement_coalesced4_paramrk_tuned` dropped to `~1.369B eval/s` while symbol-key `replacement_coalesced4_tuned` stayed at `~1.749B eval/s`.
- PTX/SASS showed the cause: explicit-key mode replaced broadcast-friendly `ld.const` key reads with many scalar global key loads in the kernel hot path.

Fix implemented:

- Added `replacement_coalesced4_paramrk_shared(_tuned)`.
- Kernel now stages `rk_bits` once per block into dynamic shared memory (`rk4_shared`) and then reads round keys from shared for all rounds.
- Host launch now sets dynamic shared size to `sizeof(h_rk_bits)` (5632 bytes).

Measured result on GTX 1070 (`threads=65536`, autotune blocks `64/128/256`, `reps=200`):

- `replacement_coalesced4_tuned`: `1.749B eval/s` (`26691.39 MiB/s`)
- `replacement_coalesced4_paramrk_tuned`: `1.369B eval/s` (`20887.00 MiB/s`)
- `replacement_coalesced4_paramrk_shared_tuned`: `1.725B eval/s` (`26317.23 MiB/s`)
- `replacement_coalesced4_paramrk_soa_tuned`: `0.702B eval/s` (`10712.20 MiB/s`)

Outcome:

- Explicit-key path is now within ~1.4% of symbol-key throughput on this GPU while keeping key-as-parameter API.
- Use `replacement_coalesced4_paramrk_shared_tuned` for shared-key workloads.
- Use `replacement_coalesced4_paramrk_soa_tuned` when each thread has a distinct key.
- Per-thread distinct keys are bandwidth-heavy in u32-bitplane SoA form; expect lower throughput unless key loads are amortized across more work per launch.

## Replacement ABI (for other CUDA projects)

Kernel entrypoint:

```c
extern "C" __global__ void aes10_bp128_kernel(
    const uint32_t* __restrict__ in_ptr,
    uint32_t* __restrict__ out_ptr,
    uint32_t n_threads);
```

Requirements:

- `in_ptr` and `out_ptr` are bitplane buffers sized `n_threads * 128` `uint32_t` words.
- `RK_BITS` symbol (`uint32_t[11][16][8]`) must be uploaded once per key schedule via `cuModuleGetGlobal` + `cuMemcpyHtoD`.
- One thread computes 32 AES block evaluations (bitplane lanes), same as legacy accounting.

## Recommended usage

```bash
PYTHONPATH=. .venv/bin/python scripts/bench_aes10_bp128_cuda.py \
  --kernel-mode replacement_coalesced4_tuned \
  --threads 65536 --reps 200 --check
```

For FAEST-style integration details (byte/block compatibility wrapper + key conversion + native plane-major4 API), see:

- `docs/FAEST_BP128_COALESCED4_ADAPTER.md`
- `scripts/bench_aes10_bp128_faest_examples.py` for side-by-side mode benchmarking
- `scripts/bench_aes10_bp128_byteio_cuda.py` for byte-API converter overhead benchmarking

```bash
PYTHONPATH=. .venv/bin/python scripts/bench_aes10_bp128_cuda.py \
  --kernel-mode legacy_tuned \
  --threads 65536 --reps 200
```
