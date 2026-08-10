# FAEST Adapter for BP128 `coalesced4` (CUDA)

This document adapts the current `replacement_coalesced4` kernel shape to FAEST-style inputs.

## Goal

Integrate BP128 AES (32-way bitslice per thread) into FAEST while preserving:

- FAEST-facing byte/block workflow where needed.
- GPU-native fast path with `coalesced4` layout for throughput.
- No host-side global transpose in hot path.

## Core Contract (fast path)

### Data layout: `plane-major4`

For each thread `tid`:

- 16 AES bytes
- 8 bitplanes per byte
- packed as 2 `uint4` groups per byte (`bits 0..3`, `bits 4..7`)

Index:

```c
group = byte * 2 + (bit >> 2);   // 0..31
lane  = bit & 3;                 // x/y/z/w
idx   = group * n_threads + tid;
```

Buffer shape:

- `in_planes4[32 * n_threads]` (`uint4`)
- `out_planes4[32 * n_threads]` (`uint4`)

### Round keys: per-thread bitsliced keys

Recommended SoA layout for coalesced key loads:

```c
rk_bits[((round * 16 + byte) * 8 + bit) * n_threads + tid]
```

Total words per thread: `11 * 16 * 8 = 1408` (`uint32_t`).

## CUDA API

### 1) Native high-throughput API (recommended)

```c
extern "C" __global__ void aes128_bp32_encrypt_kernel(
    const uint4* __restrict__ in_planes4,      // [32 * n_threads]
    uint4* __restrict__ out_planes4,           // [32 * n_threads]
    const uint32_t* __restrict__ rk_bits_soa,  // [11*16*8*n_threads]
    uint32_t n_threads);
```

Use this when caller can keep data in `plane-major4` between stages.

### 2) FAEST compatibility API (byte/block inputs)

```c
extern "C" __global__ void aes128_bp32_encrypt_bytes_kernel(
    const uint8_t* __restrict__ in_blocks,     // [n_threads][32][16]
    uint8_t* __restrict__ out_blocks,          // [n_threads][32][16]
    const uint32_t* __restrict__ rk_bits_soa,  // [11*16*8*n_threads]
    uint32_t n_threads);
```

This wrapper does in-kernel transpose:

1. Load 32x16 byte-blocks for `tid`.
2. Build local bitplanes (`st[16][8]`).
3. Run BP128 AES core.
4. Scatter local bitplanes back to 32x16 output bytes.

This avoids host/global transpose kernels while preserving FAEST byte API.

### 3) Key conversion API (one-time per key batch)

FAEST currently has BE word round keys (AES-NI/T0 style). Convert once:

```c
extern "C" __global__ void aes128_rk_be_to_bits_soa_kernel(
    const uint32_t* __restrict__ rk_be_words,  // [n_threads][44]
    uint32_t* __restrict__ rk_bits_soa,        // [11*16*8*n_threads]
    uint32_t n_threads);
```

Notes:

- `rk_be_words` is AES-128 key schedule in BE word format.
- Kernel expands to byte-wise round keys then to bit-sliced planes.
- Reuse `rk_bits_soa` across many CTR batches.

## Migration Path

1. Keep existing FAEST byte API; add `aes128_bp32_encrypt_bytes_kernel`.
2. Add key-prep kernel to produce `rk_bits_soa` once per key.
3. Benchmark and validate correctness.
4. If call graph allows, switch producer/consumer stages to `plane-major4` and use native kernel directly.

## Example + Benchmark

Use the FAEST-oriented comparison runner:

```bash
PYTHONPATH=. .venv/bin/python scripts/bench_aes10_bp128_faest_examples.py \
  --threads 65536 --reps 200 --sm sm_61 --check
```

It benchmarks:

- `legacy_tuned`
- `replacement_tuned`
- `replacement_coalesced_tuned`
- `replacement_coalesced4_tuned`

and prints a side-by-side throughput table plus `coalesced4 vs replacement` speedup.

For byte/block compatibility conversion cost (AoS bytes <-> plane-major4), use:

```bash
PYTHONPATH=. .venv/bin/python scripts/bench_aes10_bp128_byteio_cuda.py \
  --threads 65536 --reps 200 --sm sm_61 --check --autotune-blocks 64,128,256
```

Observed on GTX 1070 (`sm_61`):

- Best full pipeline (`conv_in + aes + conv_out`): `0.821B eval/s` (`12524 MiB/s`, block `128`)
- Converter-only equivalent: `~1.54B eval/s`
- AES-only on coalesced4 planes: `~1.77B eval/s` (`~27024 MiB/s`)

Interpretation:

- The converter is functional and reasonably fast, but conversion still dominates end-to-end byte-API mode.
- Highest throughput remains native `plane-major4` API (`replacement_coalesced4_tuned`).


### Explicit key-pointer variant

If you need per-key debuggability, `bench_aes10_bp128_cuda.py` now supports:

- `replacement_coalesced4_paramrk`
- `replacement_coalesced4_paramrk_tuned`
- `replacement_coalesced4_paramrk_shared`
- `replacement_coalesced4_paramrk_shared_tuned`
- `replacement_coalesced4_paramrk_soa`
- `replacement_coalesced4_paramrk_soa_tuned`
- `replacement_coalesced4_paramrk_soa_packed`
- `replacement_coalesced4_paramrk_soa_packed_tuned`
- `replacement_coalesced4_masterkey_soa`
- `replacement_coalesced4_masterkey_soa_tuned`

This passes KAT and removes module-symbol key binding (keys passed as kernel arg). The basic paramrk mode is slower because it performs repeated global key loads, but `replacement_coalesced4_paramrk_shared_tuned` stages key bits in shared memory and recovers near-symbol throughput for shared-key workloads. For per-thread distinct keys, `replacement_coalesced4_paramrk_soa_tuned` consumes `rk_bits[(plane * n_threads) + tid]` directly with coalesced key loads. `replacement_coalesced4_paramrk_soa_packed_tuned` uses packed key bytes SoA (`rk_bytes[(round*16 + byte) * n_threads + tid]`) to reduce key-memory bandwidth.

Latest GTX 1070 tuned reference (`threads=65536`, `reps=200`): `paramrk_shared` ~1.724B eval/s, `paramrk_soa` ~0.703B eval/s, `paramrk_soa_packed` ~1.347B eval/s.

Latest validation (`threads=65536`, `reps=120`): `paramrk_shared` ~1.675B eval/s, `paramrk_soa` ~0.702B eval/s, `paramrk_soa_packed` ~1.350B eval/s.

### Per-thread master-key mode

`replacement_coalesced4_masterkey_soa(_tuned)` takes only the AES-128 master key per thread:

```text
key_bytes_soa[(byte * n_threads) + tid]   // 16 bytes/thread
```

The kernel expands round keys in-kernel and applies them directly. On GTX 1070 (`threads=65536`, `reps=120`) this mode reached `~1.829B eval/s` (`~27908 MiB/s`, best block `64`) and outperformed the packed per-thread-roundkey mode in this benchmark.

### Packed key-buffer helpers (host side)

For direct FAEST integration without an intermediate `rk_bits` expansion pass, `scripts/bench_aes10_bp128_cuda.py` now provides:

- `pack_rk_soa_packed_shared_key(rk_bytes_11x16, threads)`
- `pack_rk_soa_packed_thread_keys(rk_bytes_per_thread, threads)`
- `pack_masterkey_soa_shared_key(master_key_16, threads)`
- `pack_masterkey_soa_thread_keys(master_keys_per_thread, threads)`

Use these to build the packed SoA key buffer expected by `replacement_coalesced4_paramrk_soa_packed*`:

```text
rk_bytes_soa[(round * 16 + byte) * n_threads + tid]
```

This keeps key material compact (`176` bytes/thread) while preserving coalesced per-thread loads.


## Correctness Requirements

- Bit/byte exact against current FAEST AES path for random and KAT vectors.
- Validate endian handling in key conversion (`rk_be_words` -> bytes -> `rk_bits_soa`).
- Ensure each thread processes exactly 32 blocks.

## Performance Guidance

- Prefer `blockDim.x` in `{64, 128, 256}` and autotune per GPU.
- Keep `rk_bits_soa` in global memory unless key scope is tiny; constant memory is not suitable for per-thread distinct keys.
- For compatibility kernel:
  - Use registers for local transpose.
  - Avoid separate transpose kernels unless profiling proves benefit.

## Mapping to Current VeryLogo Modes

- `replacement_coalesced4` in `scripts/bench_aes10_bp128_cuda.py` is the direct prototype of `aes128_bp32_encrypt_kernel`.
- It already demonstrates:
  - Correct KAT.
  - Coalesced runtime IO layout.
  - Throughput near legacy roofline on GTX 1070.
