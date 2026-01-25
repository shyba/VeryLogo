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
# Clone the Boyar-Peralta reference implementation
git clone https://github.com/conorpp/bitsliced-aes.git external-bitsliced

# Run the benchmark
python scripts/benchmark_sbox_all.py
python scripts/benchmark_sbox_all.py --iterations 10000000
```

## Target

Our goal is to reduce the ANF circuit from 1145 gates to <1000 gates through incremental optimization.

## ✅ Target Achieved: Tower Field Decomposition

Using Boyar-Peralta tower field decomposition:

| Implementation | Gates | AND | XOR | Performance | Speedup |
|---------------|-------|-----|-----|-------------|---------|
| Our ANF | 1145 | 246 | 898 | 5.25 ns/eval | 1x |
| ANF + Ternary | 994 | ~236 | ~606 | 4.79 ns/eval | 1.1x |
| **BP Tower (128)** | **128** | **34** | **94** | **3.17 ns/eval** | **1.66x** |
| BP Optimal (~115) | ~115 | 32 | 83 | 0.028 ns/eval | 188x |

### Tower Field Results

```bash
# Run the benchmark
python scripts/benchmark_bp_128.py
```

**Gate reduction**: 1145 → 128 gates (**88.8% fewer**)

### How Tower Field Works

1. **Field Decomposition**: GF(2^8) → GF((2^4)²) → GF(((2^2)²)²)
2. **Small Field Inversion**: 4-bit and 2-bit inversions need fewer gates
3. **Recomposition**: Build 8-bit result from small field operations

The 128-gate circuit has:
- **34 AND gates** (vs 246 in ANF) - 7.2x fewer
- **94 XOR gates** (vs 898 in ANF) - 9.6x fewer

### Ternary Optimization (Additive)

The TernaryMappingPass can further optimize by replacing 3-input patterns with `vpternlogd`:

```
# Before: 2 XOR gates
t1 = a ^ b
t2 = t1 ^ c

# After: 1 vpternlogd instruction
t2 = vpternlogd(a, b, c, 0x96)  // imm8=0x96 encodes XOR3
```
