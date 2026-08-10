# BP128 Dynamic Variant Family

This family is built directly from the proven high-throughput CUDA kernels used by:

- `replacement_coalesced4_masterkey_soa_tuned`
- `replacement_coalesced4_paramrk_soa_packed_tuned`

in `scripts/bench_aes10_bp128_cuda.py`.

## Goal

Generate variant PTX dynamically from the known-fast BP128 kernel lineage, instead of maintaining a separate generic implementation.

## Post Operation

Generated kernels now support:

- `store` (default)
- `xor-accumulate` (bitplanes path; output is XORed with existing output buffer contents)

## Variant Axes

- `key_bits`: `128`, `192`, `256`
- `ctr_group`: `1`, `2`, `4`
- `key_source`: `masterkey_soa`, `expanded_rk_soa`, `const_key`
- `io_layout`: `plane-major4`, `bytes`

## Current Support

- Fast supported:
  - `io_layout=plane-major4`
  - `key_source=const_key` with `key_bits=128/192/256`
  - `key_source=masterkey_soa` with `key_bits=128/192/256`
  - `key_source=expanded_rk_soa` with `key_bits=128/192/256` (round count patched in source)
- Fallback behavior:
  - `masterkey_soa` with `key_bits=192/256` uses host-expanded round keys with the packed expanded-RK kernel shape.
  - `const_key` with `key_bits=192/256` uses an embedded constant expanded-RK kernel shape.
- Still not supported in this fast family:
  - `io_layout=bytes`

Unsupported variants are still enumerated in metadata with `supported=false` and explanatory notes.

## ABI

Generated kernels keep the fast kernel ABI:

- `masterkey_soa`/`expanded_rk_soa`:
  - `const uint32_t* in_ptr`
  - `uint32_t* out_ptr`
  - `const uint8_t* key_or_rk_soa`
  - `uint32_t n_threads`
- `const_key`:
  - `const uint32_t* in_ptr`
  - `uint32_t* out_ptr`
  - `uint32_t n_threads`

`n_threads` is the effective thread count. For `ctr_group > 1`, host-side code expands:

- effective threads = `logical_threads * ctr_group`
- per-thread key material is replicated per group in SoA form.

## Usage

- Print metadata:
  - `python3 scripts/aes_bp128_variant_family.py manifest`
- Dispatch:
  - `python3 scripts/aes_bp128_variant_family.py dispatch --key-bits 128 --max-ctr-group 4`
- Emit PTX:
  - `python3 scripts/aes_bp128_variant_family.py emit-ptx --key-bits 128 --ctr-group 4 --key-source masterkey_soa --io-layout plane-major4 --post-op store --sm sm_61 --out out/bp128_aes128_master_g4.ptx`
  - `python3 scripts/aes_bp128_variant_family.py emit-ptx --key-bits 128 --ctr-group 4 --key-source masterkey_soa --io-layout plane-major4 --post-op xor-accumulate --sm sm_61 --out out/bp128_aes128_master_g4_xoracc.ptx`
- Benchmark emitted variant:
  - `python3 scripts/aes_bp128_variant_family.py bench --key-bits 128 --ctr-group 1 --key-source masterkey_soa --io-layout plane-major4 --post-op xor-accumulate --threads 65536 --block 64 --reps 1000`
  - (default benchmark block is `64`, matching the tuned fast path)
- Correctness check:
  - `python3 scripts/aes_bp128_variant_family.py check --key-bits 256 --ctr-group 2 --key-source expanded_rk_soa --io-layout plane-major4`

## Axis Sweep

To test the requested axis matrix (`KEY_BITS`, `CTR_GROUP`, `KEY_SOURCE`, input/output layout, `POST_OP`) with throughput + correctness where supported:

- `python3 scripts/bench_aes_bp128_axes.py --threads 65536 --reps 40 --sm sm_61`
- `python3 scripts/bench_aes_bp128_axes.py --threads 65536 --reps 40 --sm sm_61 --autotune-blocks 64,128,256`

This writes:

- `out/bp128_axes_results.json`
- `out/bp128_axes_results.md`

Unsupported combinations are reported with explicit reasons in the output table.

## Register-only status (sm_61)

`ptxas -v` check on generated store kernels (g1 variants):

- `128/masterkey_soa`: `255 regs`, `0 stack`, `0 spill`
- `128/expanded_rk_soa`: `188 regs`, `0 stack`, `0 spill`
- `128/const_key`: `255 regs`, `0 stack`, `0 spill`
- `192/masterkey_soa`: `188 regs`, `0 stack`, `0 spill`
- `192/expanded_rk_soa`: `188 regs`, `0 stack`, `0 spill`
- `192/const_key`: `255 regs`, `0 stack`, `0 spill`
- `256/masterkey_soa`: `188 regs`, `0 stack`, `0 spill`
- `256/expanded_rk_soa`: `188 regs`, `0 stack`, `0 spill`
- `256/const_key`: `255 regs`, `0 stack`, `0 spill`

This keeps round-state in registers and avoids local-memory spill traffic.

## Throughput spot-check (GTX 1070, sm_61)

Store mode, `threads=65536`, `reps=120`:

- `128/g4/masterkey_soa` (`block=64`): `1.981B eval/s` (`30227 MiB/s`)
- `128/g4/expanded_rk_soa` (`block=128`): `1.388B eval/s` (`21181 MiB/s`)
- `128/g4/const_key` (`block=64`): `1.820B eval/s` (`27767 MiB/s`)
- `256/g4/masterkey_soa` (`block=64`): `1.352B eval/s` (`20632 MiB/s`)
- `256/g4/expanded_rk_soa` (`block=64`): `1.338B eval/s` (`20420 MiB/s`)
- `256/g4/const_key` (`block=64`): `1.437B eval/s` (`21928 MiB/s`)

`expanded_rk_soa` remains lower because every round key byte is read from global key SoA; masterkey/const paths avoid that cost.
