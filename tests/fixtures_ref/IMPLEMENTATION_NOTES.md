# Implementation Notes

Technical details about the reference model implementations.

## Design Decisions

### Bit-Width Handling

All functions use explicit masking to ensure correct bit-width semantics:

```python
# 32-bit addition with overflow
result = (a + b) & 0xFFFFFFFF

# 64-bit rotation
mask = (1 << 64) - 1
rotated = ((x << n) | (x >> (64 - n))) & mask
```

### Sequential State Representation

Sequential fixtures return lists of state dictionaries:

```python
states = acc_fsm(limit=10, x=0xFF, ticks=20)
# states[i] = {"acc": ..., "i": ...}
```

This allows:
- Per-tick state inspection
- Easy comparison with HDL simulation traces
- Deterministic replay for debugging

### LFSR Implementation

32-bit Galois LFSR with configurable taps:

```python
def lfsr_next(state, taps=None):
    if taps is None:
        taps = [32, 22, 2, 1]  # Default polynomial
    bit = 0
    for tap in taps:
        bit ^= (state >> (tap - 1)) & 1
    return ((state << 1) | bit) & 0xFFFFFFFF
```

Default polynomial: x^32 + x^22 + x^2 + x + 1 (maximum period)

### Round Function Design (roundN)

Matches common cipher round structure:
1. Rotate left by 1 bit
2. Add round constant
3. XOR results

```python
def f(x, k):
    rotated = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
    added = (x + k) & 0xFFFFFFFFFFFFFFFF
    return rotated ^ added
```

State includes:
- `s`: current round state
- `r`: round counter
- `out`: output (always equals `s`)

### Nonlinear Island Pattern

Core nonlinear operations (AND, OR) surrounded by linear mixing (XOR, rotate):

```python
# Nonlinear core
core = (x & y) ^ ((x | y) & ((x ^ y) << 1))

# Linear diffusion (multiple rotations + XOR)
r1 = rotate(core, 13)
r2 = rotate(core, 37)
mixed = core ^ r1 ^ r2 ^ x ^ y
```

This pattern mimics:
- Cryptographic S-boxes (nonlinear substitution)
- ARX constructions (Add-Rotate-XOR)
- Tower field decompositions

### Mux Reconvergence

Creates shared subexpressions and later reconvergence:

```python
m1 = a if cond1 else b  # First mux
m2 = c if cond1 else d  # Second mux (shares cond1)
m3 = m1 if cond2 else m2  # Reconvergence
shared = a ^ c  # Shared computation
result = m3 ^ shared  # Final reconvergence
```

Stresses:
- Common subexpression elimination (CSE)
- Mux tree balancing
- Ternary mapping correctness

### Slice/Concat Normalization

Tests slice extraction and concatenation identity:

```python
# 64-bit: permute quarters
q0, q1, q2, q3 = extract_quarters(x)
h0 = concat(q2, q0)
h1 = concat(q3, q1)
y = concat(h1, h0)

# Re-slice at odd boundaries
slice1 = extract_bits(y, 7, 25)
slice2 = extract_bits(y, 29, 40)
```

Ensures:
- Slice/concat composition is associative
- Bit ordering is preserved
- No semantic drift in normalization

## Parameter Ranges

### Recommended Test Values

| Fixture | Parameter | Small | Medium | Large |
|---------|-----------|-------|--------|-------|
| vec_add | N | 4 | 256 | 65536 |
| xor_tree | depth | 4 | 16 | 24 |
| acc_fsm | limit | 10 | 1000 | 1000000 |
| acc_fsm | ticks | 20 | 2000 | 2000000 |
| lfsr_table | ticks | 32 | 10000 | 1000000 |
| roundN | R | 4 | 8 | 16 |
| roundN | ticks | 5 | 10 | 20 |
| pmux16/32 | - | - | - | - |

### Width Support

| Fixture | Default | Tested | Notes |
|---------|---------|--------|-------|
| vec_add | 32-bit lanes | 32 | Lane width fixed |
| nonlinear_island | 64 | 16,32,64,128 | Generic |
| mux_reconverge | 64 | 32,64,128 | Generic |
| rotmix | 64 | 32,64 | Rotation constants tuned for 64 |
| slice_concat | 64 | 32,64 | Slice offsets width-dependent |

## Correctness Properties

### Overflow Handling

All arithmetic operations wrap at the specified bit-width:

```python
# 32-bit: 0xFFFFFFFF + 1 = 0
# 64-bit: 0xFFFFFFFFFFFFFFFF + 1 = 0
```

### Sign Extension

All operations are unsigned. No sign extension occurs:

```python
# Extract lane 0 (bits 31:0)
lane0 = (packed >> 0) & 0xFFFFFFFF  # Mask, don't sign-extend
```

### Rotation Direction

All rotations are left rotations (rotate toward MSB):

```python
# Rotate left by n bits
rotated = ((x << n) | (x >> (width - n))) & mask
```

### Reset Semantics

Reset sets state to initial values:
- `acc_fsm`: acc=0, i=0
- `lfsr_table`: lfsr=seed, acc=0
- `roundN`: s=in_val, r=0

Reset is synchronous (takes effect on the tick specified in `rst_at`).

## Testing Strategy

### Unit Tests

Each function has:
1. Basic correctness test (known input/output)
2. Boundary condition test (0, max value, overflow)
3. Parameter variation test (different N, R, width)

### Integration Tests

Reference models are used to validate:
1. Compiled HDL simulation (Verilator)
2. Native backend execution (AVX-512, PTX)
3. Optimized transformations (fusion, unrolling)

### Determinism Verification

All tests are deterministic:
- No random number generation
- Fixed seeds for LFSR
- Reproducible across platforms

Run same test 1000x, verify identical output:

```bash
for i in {1..1000}; do
    .venv/bin/python -m unittest tests.test_fixtures_ref.TestThroughputRef.test_vec_add_basic
done
```

## Performance Notes

### Reference Model Speed

These are **reference** implementations, not optimized for speed:

- `vec_add(N=65536)`: ~1ms (Python loop overhead)
- `acc_fsm(ticks=1000000)`: ~100ms (Python loop)
- `lfsr_table(ticks=1000000)`: ~150ms (LFSR + table lookup)

For benchmarking HDL backends, use native execution (compiled C/PTX), not these reference models.

### When to Use Verilator Instead

Use Verilator golden simulation when:
1. HDL has complex `always` block semantics
2. Combinational logic is too complex to hand-translate
3. You need cycle-accurate timing verification

Use Python reference when:
1. Semantics are straightforward (add, xor, rotate)
2. You need fast test iteration
3. You want portable tests (no Verilator dependency)

## Future Extensions

### Planned Additions

1. **Float fixtures**: IEEE-754 operations (fma, sqrt, etc.)
2. **Memory fixtures**: RAM read/write patterns
3. **Vectorized references**: NumPy-based implementations for speed
4. **Parameterized generation**: Auto-generate references from HDL

### Compatibility

All reference models maintain backward compatibility:
- New optional parameters added at end
- Default values preserve existing behavior
- Return types never change

Example:

```python
# Old code still works
result = vec_add(a, b, N=4)

# New optional parameter
result = vec_add(a, b, N=4, saturate=False)  # Future
```
