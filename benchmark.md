# S-box Implementation Benchmark

Comprehensive comparison of AES S-box implementations using different techniques.

## Results

| Implementation | ns/eval | evals/sec | Notes |
|---------------|---------|-----------|-------|
| **BP AVX2 (circuit)** | **0.028** | **35.5B** | **Fastest! ~115 gates** |
| BP u64 (circuit) | 0.055 | 18.2B | ~115 gates, 64 parallel |
| AES-NI (full round) | ~0 | 22.2B | Hardware accelerated |
| Table lookup | 0.2 | 5.0B | Memory, timing side-channel |
| BP u64 (w/transpose) | 2.7 | 364M | With byte-bitplane conversion |
| Our ANF AVX2 | 5.1 | 195M | 1145 gates, 32 parallel |
| Our ANF SSE2 | 6.4 | 157M | 1145 gates, 16 parallel |
| Our ANF uint64 | 22.4 | 45M | 1145 gates, 8 parallel |
| Python | 63,965 | 16K | Reference |

## Implementations

### Boyar-Peralta Circuit (~115 gates)
The optimal known circuit for AES S-box, based on "A New Combinational Logic Minimization Technique" by Boyar & Peralta. Implementation from [conorpp/bitsliced-aes](https://github.com/conorpp/bitsliced-aes).

### Our ANF Circuit (1145 gates)
Algebraic Normal Form synthesis producing a correct but unoptimized circuit. This is the baseline for our circuit optimization work.

### Other Implementations
- **AES-NI**: Hardware-accelerated full AES round (includes ShiftRows + MixColumns)
- **Table lookup**: Simple 256-byte lookup table (fast but has cache timing side-channel)

## Key Insights

1. **Circuit optimization matters more than SIMD width**: BP u64 with transpose (2.7 ns) beats our AVX2 (5.1 ns) despite narrower SIMD
2. **Boyar-Peralta AVX2 beats AES-NI hardware**: 35.5B vs 22.2B evals/sec for circuit-only
3. **~10x efficiency gap**: 115 gates vs 1145 gates shows significant optimization potential
4. **Transpose overhead is significant**: 0.055 ns -> 2.7 ns when converting bytes to/from bit planes

## Running the Benchmark

```bash
python scripts/benchmark_sbox_all.py
python scripts/benchmark_sbox_all.py --iterations 10000000
```

## Target

Our goal is to reduce the ANF circuit from 1145 gates to <1000 gates through incremental optimization.
