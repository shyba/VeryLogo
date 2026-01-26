# AES S-Box: The Boyar-Peralta Optimal Circuit

This document provides a technical analysis of the AES S-box implementation found in `external-bitsliced/`, which uses the Boyar-Peralta circuit achieving the theoretical minimum of 32 AND gates.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [The Boyar-Peralta Circuit Structure](#the-boyar-peralta-circuit-structure)
4. [Gate Counts and Optimality](#gate-counts-and-optimality)
5. [Bitslicing Implementation](#bitslicing-implementation)
6. [Circuit Analysis](#circuit-analysis)

## Introduction

The AES S-box (substitution box) is the only non-linear component in the AES cipher. It operates on 8-bit values and provides the cryptographic strength through confusion. The S-box is defined as:

```
S(x) = A * x^(-1) + c  (in GF(2^8))
```

where:
- `x^(-1)` is the multiplicative inverse in GF(2^8) with irreducible polynomial `x^8 + x^4 + x^3 + x + 1`
- `A` is an 8x8 affine transformation matrix over GF(2)
- `c` is the constant `0x63`
- For `x = 0`, the inverse is defined as 0

The implementation in `external-bitsliced/bs.c` uses the Boyar-Peralta straight-line program, referenced from:
```
http://cs-www.cs.yale.edu/homes/peralta/CircuitStuff/CMT.html
```

## Mathematical Background

### Tower Field Decomposition

The core insight enabling efficient S-box computation is that GF(2^8) can be decomposed into a tower of smaller fields:

```
GF(2^8) = GF((2^2)^2)^2 = GF(2)[x]/(x^2 + x + 1)[y]/(y^2 + y + x)[z]/(z^2 + z + xy)
```

This tower construction allows the GF(2^8) inversion to be computed through:

1. **GF(2^4) operations**: The 8-bit element is viewed as a polynomial `a*z + b` where `a, b` are in GF(2^4)
2. **GF(2^2) operations**: Each GF(2^4) element is viewed as `c*y + d` where `c, d` are in GF(2^2)
3. **GF(2) operations**: Each GF(2^2) element is viewed as `e*x + f` where `e, f` are single bits

### GF(2^8) Inversion Formula

For an element `g = a*z + b` in GF(2^8), where `a, b` are in GF(2^4):

```
g^(-1) = (a*z + b)^(-1) = a*(a^2*N + ab + b^2)^(-1)*z + (a+b)*(a^2*N + ab + b^2)^(-1)
```

where `N` is the norm element. The key insight is that `a^2*N + ab + b^2` is in GF(2^4), so we only need to compute one GF(2^4) inversion instead of a GF(2^8) inversion.

This process recurses: the GF(2^4) inversion becomes a GF(2^2) inversion, and GF(2^2) inversion is trivial (just swap the bits: `(a,b)^(-1) = (b,a)` after norm computation).

### Why 32 AND Gates is Optimal

The theoretical lower bound for AND gates in a GF(2^8) inversion circuit is based on the algebraic degree of the function. Since inversion has algebraic degree 254 (which is 2^8 - 2), and each AND gate can at most double the algebraic degree, we need at least `log_2(254) ~ 8` AND gates just for the degree contribution.

However, the actual bound comes from a more sophisticated analysis:
- The S-box is a composition of GF(2^8) inversion and an affine transformation
- The affine parts (linear + constant) require only XOR and NOT gates
- The inversion requires non-linear operations (AND gates)
- Through careful algebraic analysis and exhaustive search, Boyar and Peralta proved 32 AND gates is the minimum achievable

## The Boyar-Peralta Circuit Structure

The circuit follows a three-layer architecture:

### Layer 1: Top Linear Layer (T variables)

This layer prepares the input for the non-linear operations using only XOR gates:

```c
T1 = U[7] ^ U[4];
T2 = U[7] ^ U[2];
T3 = U[7] ^ U[1];
T4 = U[4] ^ U[2];
T5 = U[3] ^ U[1];
T6 = T1 ^ T5;
T7 = U[6] ^ U[5];
T8 = U[0] ^ T6;
T9 = U[0] ^ T7;
T10 = T6 ^ T7;
// ... continues through T27
```

The input bits `U[0]` through `U[7]` represent the 8-bit input byte. This layer produces 27 intermediate values (T1-T27) that are linear combinations of the input bits.

**Purpose**: Transform the input into a basis that minimizes the number of AND gates needed in the middle layer. This is the result of extensive computer-aided optimization.

### Layer 2: Non-Linear Middle Layer (M variables)

This is where the 32 AND gates reside, computing the GF(2^8) inversion:

```c
// First section: 18 AND gates for forward computation
M1 = T13 & T6;
M2 = T23 & T8;
M3 = T14 ^ M1;
M4 = T19 & U[0];
M5 = M4 ^ M1;
M6 = T3 & T16;
// ... through M23

// Core inversion logic (4 AND gates)
M24 = M22 ^ M23;
M25 = M22 & M20;
M26 = M21 ^ M25;
M27 = M20 ^ M21;
M28 = M23 ^ M25;
M29 = M28 & M27;
M30 = M26 & M24;
M31 = M20 & M23;
M32 = M27 & M31;
M33 = M27 ^ M25;
M34 = M21 & M22;
M35 = M24 & M34;
M36 = M24 ^ M25;
M37 = M21 ^ M29;
M38 = M32 ^ M33;
M39 = M23 ^ M30;
M40 = M35 ^ M36;
M41 = M38 ^ M40;
M42 = M37 ^ M39;
M43 = M37 ^ M38;
M44 = M39 ^ M40;
M45 = M42 ^ M41;

// Second section: 18 AND gates for output expansion
M46 = M44 & T6;
M47 = M40 & T8;
M48 = M39 & U[0];
// ... through M63
```

**Gate count breakdown**:
- M1, M2: 2 AND gates
- M4: 1 AND gate
- M6, M7: 2 AND gates
- M9: 1 AND gate
- M11, M12: 2 AND gates
- M14: 1 AND gate
- M25, M29, M30, M31, M32, M34, M35: 7 AND gates (core inversion)
- M46-M63: 18 AND gates (output expansion)

**Total: 32 AND gates**

The XOR operations (M3, M5, M8, M10, M13, M15-M45) do not count toward the non-linear complexity.

### Layer 3: Bottom Linear Layer (L/S variables)

This layer combines the outputs of the middle layer using only XOR gates:

```c
L0 = M61 ^ M62;
L1 = M50 ^ M56;
L2 = M46 ^ M48;
// ... continues through L29

S[7] = L6 ^ L24;
S[6] = ~(L16 ^ L26);  // NOT + XOR
S[5] = ~(L19 ^ L28);
S[4] = L6 ^ L21;
S[3] = L20 ^ L22;
S[2] = L25 ^ L29;
S[1] = ~(L13 ^ L27);
S[0] = ~(L6 ^ L23);
```

**Purpose**: Apply the affine transformation matrix `A` and add the constant `0x63`. The NOT operations (`~`) implement the addition of the constant vector.

## Gate Counts and Optimality

### Complete Gate Count

| Operation Type | Count | Description |
|---------------|-------|-------------|
| XOR | 83 | Top layer (23) + Middle layer (28) + Bottom layer (32) |
| AND | 32 | Non-linear operations (theoretical minimum) |
| NOT | 4 | Part of affine constant addition |
| **Total** | **119** | Complete S-box circuit |

### Comparison with Other Implementations

| Implementation | AND Gates | XOR Gates | Total | Depth |
|---------------|-----------|-----------|-------|-------|
| Lookup Table | 0 | 0 | N/A | 1 (memory access) |
| Naive Boolean | ~200 | ~500 | ~700 | High |
| Canright 2005 | 36 | 120 | 156 | - |
| Boyar-Peralta 2012 | **32** | 83 | 115 | 16 |

The Boyar-Peralta circuit is provably optimal in AND gate count. The XOR count of 83 is also very competitive, achieved through careful sharing of intermediate results.

### Why AND Gates Matter

1. **Side-channel resistance**: AND gates are more expensive to protect against power analysis and other side-channel attacks
2. **Masking cost**: In masked implementations, each AND gate requires significant overhead
3. **Hardware area**: In some technologies, AND gates are larger or slower than XOR gates
4. **Multi-party computation**: AND gates require communication between parties while XOR gates can be computed locally

## Bitslicing Implementation

### Concept

Bitslicing treats the processor as a SIMD machine operating on individual bits. Instead of computing one S-box at a time, we compute `WORD_SIZE` S-boxes in parallel:

```
Traditional:     byte[0] -> sbox -> byte[0]'
                 byte[1] -> sbox -> byte[1]'
                 ...

Bitsliced:       bit0[0..63] -> sbox_bit0 -> bit0'[0..63]
                 bit1[0..63] -> sbox_bit1 -> bit1'[0..63]
                 ...
```

### Data Layout (bs.h)

```c
#define WORD_SIZE           64      // Bits per processor word
#define BLOCK_SIZE          128     // AES block size in bits
#define BS_BLOCK_SIZE       (BLOCK_SIZE * WORD_SIZE / 8)  // 1024 bytes = 64 blocks

typedef uint64_t word_t;  // Each word holds 1 bit from 64 different blocks
```

For a 64-bit machine:
- Process 64 AES blocks (64 * 16 = 1024 bytes) simultaneously
- State is 128 `word_t` values (one per bit position in AES state)
- Each `word_t` holds bit `i` from all 64 blocks

### Transpose Operations

The transpose converts between normal byte-oriented representation and bitsliced representation:

```c
void bs_transpose(word_t * blocks);      // Normal -> Bitsliced
void bs_transpose_rev(word_t * blocks);  // Bitsliced -> Normal
```

**Algorithm** (simplified):
```
for each bit position k (0 to WORD_SIZE-1):
    for each word i in block:
        for each bit j in word:
            transpose[offset + j] |= (blocks[k*WORDS_PER_BLOCK + i] & (1 << j)) ? (1 << k) : 0
```

### S-box Function Signature

```c
void bs_sbox(word_t U[8]);
```

Takes 8 words representing 8 bit-planes. Each word contains the same bit position from 64 different bytes. The function computes 64 S-box operations in parallel using the Boyar-Peralta circuit.

### Performance Characteristics

From the README:
- **Performance optimized**: 51 cycles/byte (12,150 byte footprint)
- **Footprint optimized**: 81 cycles/byte (8,526 byte footprint)

For 64 blocks processed simultaneously:
- Total operations: 64 * 119 = 7,616 gate operations
- Amortized per block: 119 operations (same as single S-box)
- But now XOR/AND/NOT are single CPU instructions operating on 64 bits

## Circuit Analysis

### Information Flow

```
U[0..7] (8 input bit-planes, each 64 bits wide)
    |
    v
Top Linear (T1-T27)  [23 XOR gates]
    |
    v
Non-linear Core (M1-M63)  [32 AND + 28 XOR gates]
    |
    v
Bottom Linear (L0-L29, S[0..7])  [32 XOR + 4 NOT gates]
    |
    v
S[0..7] (8 output bit-planes)
```

### Critical Path Analysis

The circuit depth (longest path from input to output) is 16 gates. This determines latency in hardware implementations but is less relevant for software bitslicing where all operations are sequential.

### Inverse S-box (bs_sbox_rev)

The inverse S-box (`bs_sbox_rev`) follows a similar structure but with:
- Modified top linear layer (different T variables)
- Same non-linear core (M variables) - inversion is self-inverse
- Modified bottom linear layer (P variables instead of L)

The inverse applies `A^(-1) * (S - c)` then computes the GF(2^8) inverse.

### Integration with AES

The S-box is applied to each byte in the AES state:

```c
void bs_apply_sbox(word_t * input)
{
    int i;
    for(i=0; i < BLOCK_SIZE; i+=8)
    {
        bs_sbox(input+i);  // Process 8 bit-planes at a time
    }
}
```

Since the AES state is 128 bits = 16 bytes, and each `bs_sbox` call processes one byte position across all 64 parallel blocks, we need 16 calls (128/8) to complete SubBytes for all blocks.

## References

1. Boyar, J., Peralta, R.: "A New Combinational Logic Minimization Technique with Applications to Cryptology" (2010)
2. Boyar, J., Peralta, R.: "A Small Depth-16 Circuit for the AES S-Box" (2012)
3. Canright, D.: "A Very Compact S-Box for AES" (2005)
4. Biham, E.: "A Fast New DES Implementation in Software" (1997)
5. Original circuit: http://cs-www.cs.yale.edu/homes/peralta/CircuitStuff/CMT.html

## Appendix: Complete Gate Assignment

### Top Linear Layer (23 XOR operations)

| Variable | Computation |
|----------|-------------|
| T1 | U[7] ^ U[4] |
| T2 | U[7] ^ U[2] |
| T3 | U[7] ^ U[1] |
| T4 | U[4] ^ U[2] |
| T5 | U[3] ^ U[1] |
| T6 | T1 ^ T5 |
| T7 | U[6] ^ U[5] |
| T8 | U[0] ^ T6 |
| T9 | U[0] ^ T7 |
| T10 | T6 ^ T7 |
| T11 | U[6] ^ U[2] |
| T12 | U[5] ^ U[2] |
| T13 | T3 ^ T4 |
| T14 | T6 ^ T11 |
| T15 | T5 ^ T11 |
| T16 | T5 ^ T12 |
| T17 | T9 ^ T16 |
| T18 | U[4] ^ U[0] |
| T19 | T7 ^ T18 |
| T20 | T1 ^ T19 |
| T21 | U[1] ^ U[0] |
| T22 | T7 ^ T21 |
| T23 | T2 ^ T22 |
| T24 | T2 ^ T10 |
| T25 | T20 ^ T17 |
| T26 | T3 ^ T16 |
| T27 | T1 ^ T12 |

### Non-Linear Middle Layer (32 AND + 28 XOR operations)

| Variable | Computation | Type |
|----------|-------------|------|
| M1 | T13 & T6 | AND |
| M2 | T23 & T8 | AND |
| M3 | T14 ^ M1 | XOR |
| M4 | T19 & U[0] | AND |
| M5 | M4 ^ M1 | XOR |
| M6 | T3 & T16 | AND |
| M7 | T22 & T9 | AND |
| M8 | T26 ^ M6 | XOR |
| M9 | T20 & T17 | AND |
| M10 | M9 ^ M6 | XOR |
| M11 | T1 & T15 | AND |
| M12 | T4 & T27 | AND |
| M13 | M12 ^ M11 | XOR |
| M14 | T2 & T10 | AND |
| M15 | M14 ^ M11 | XOR |
| M16 | M3 ^ M2 | XOR |
| M17 | M5 ^ T24 | XOR |
| M18 | M8 ^ M7 | XOR |
| M19 | M10 ^ M15 | XOR |
| M20 | M16 ^ M13 | XOR |
| M21 | M17 ^ M15 | XOR |
| M22 | M18 ^ M13 | XOR |
| M23 | M19 ^ T25 | XOR |
| M24 | M22 ^ M23 | XOR |
| M25 | M22 & M20 | AND |
| M26 | M21 ^ M25 | XOR |
| M27 | M20 ^ M21 | XOR |
| M28 | M23 ^ M25 | XOR |
| M29 | M28 & M27 | AND |
| M30 | M26 & M24 | AND |
| M31 | M20 & M23 | AND |
| M32 | M27 & M31 | AND |
| M33 | M27 ^ M25 | XOR |
| M34 | M21 & M22 | AND |
| M35 | M24 & M34 | AND |
| M36 | M24 ^ M25 | XOR |
| M37 | M21 ^ M29 | XOR |
| M38 | M32 ^ M33 | XOR |
| M39 | M23 ^ M30 | XOR |
| M40 | M35 ^ M36 | XOR |
| M41 | M38 ^ M40 | XOR |
| M42 | M37 ^ M39 | XOR |
| M43 | M37 ^ M38 | XOR |
| M44 | M39 ^ M40 | XOR |
| M45 | M42 ^ M41 | XOR |
| M46 | M44 & T6 | AND |
| M47 | M40 & T8 | AND |
| M48 | M39 & U[0] | AND |
| M49 | M43 & T16 | AND |
| M50 | M38 & T9 | AND |
| M51 | M37 & T17 | AND |
| M52 | M42 & T15 | AND |
| M53 | M45 & T27 | AND |
| M54 | M41 & T10 | AND |
| M55 | M44 & T13 | AND |
| M56 | M40 & T23 | AND |
| M57 | M39 & T19 | AND |
| M58 | M43 & T3 | AND |
| M59 | M38 & T22 | AND |
| M60 | M37 & T20 | AND |
| M61 | M42 & T1 | AND |
| M62 | M45 & T4 | AND |
| M63 | M41 & T2 | AND |

### Bottom Linear Layer (32 XOR + 4 NOT operations)

| Variable | Computation |
|----------|-------------|
| L0 | M61 ^ M62 |
| L1 | M50 ^ M56 |
| L2 | M46 ^ M48 |
| L3 | M47 ^ M55 |
| L4 | M54 ^ M58 |
| L5 | M49 ^ M61 |
| L6 | M62 ^ L5 |
| L7 | M46 ^ L3 |
| L8 | M51 ^ M59 |
| L9 | M52 ^ M53 |
| L10 | M53 ^ L4 |
| L11 | M60 ^ L2 |
| L12 | M48 ^ M51 |
| L13 | M50 ^ L0 |
| L14 | M52 ^ M61 |
| L15 | M55 ^ L1 |
| L16 | M56 ^ L0 |
| L17 | M57 ^ L1 |
| L18 | M58 ^ L8 |
| L19 | M63 ^ L4 |
| L20 | L0 ^ L1 |
| L21 | L1 ^ L7 |
| L22 | L3 ^ L12 |
| L23 | L18 ^ L2 |
| L24 | L15 ^ L9 |
| L25 | L6 ^ L10 |
| L26 | L7 ^ L9 |
| L27 | L8 ^ L10 |
| L28 | L11 ^ L14 |
| L29 | L11 ^ L17 |
| S[7] | L6 ^ L24 |
| S[6] | ~(L16 ^ L26) |
| S[5] | ~(L19 ^ L28) |
| S[4] | L6 ^ L21 |
| S[3] | L20 ^ L22 |
| S[2] | L25 ^ L29 |
| S[1] | ~(L13 ^ L27) |
| S[0] | ~(L6 ^ L23) |
