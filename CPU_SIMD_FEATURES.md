# CPU SIMD Features

This file is generated for the current machine.
Regenerate with `python3 scripts/cpu_simd_features.py --out CPU_SIMD_FEATURES.md`.

## CPU
- Architecture: `x86_64`
- Model: `AMD Ryzen 9 9950X3D 16-Core Processor`

## SIMD ISA Support (from `/proc/cpuinfo` flags)

### SSE Family
- SSE: yes (`sse`)
- SSE2: yes (`sse2`)
- SSE3: yes (`sse3`|`pni`)
- SSSE3: yes (`ssse3`)
- SSE4.1: yes (`sse4_1`)
- SSE4.2: yes (`sse4_2`)
- SSE4a: yes (`sse4a`)

### AVX Family
- AVX: yes (`avx`)
- AVX2: yes (`avx2`)
- FMA: yes (`fma`)
- AVX-VNNI: yes (`avx_vnni`)

### AVX-512 Family
- Present flags:
  - `avx512_bf16`
  - `avx512_bitalg`
  - `avx512_vbmi2`
  - `avx512_vnni`
  - `avx512_vp2intersect`
  - `avx512_vpopcntdq`
  - `avx512bw`
  - `avx512cd`
  - `avx512dq`
  - `avx512f`
  - `avx512ifma`
  - `avx512vbmi`
  - `avx512vl`

