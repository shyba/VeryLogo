# Fixture Reference Models

This directory contains Python reference implementations for test fixtures specified in `/home/user/repos/VeryLogo/docs/tests.md`.

Each module provides bit-accurate reference models that match the exact Verilog semantics of the corresponding fixtures.

## Modules

### throughput_ref.py
Embarrassingly parallel throughput fixtures:
- `vec_add(a, b, N=1024)` - N lanes of 32-bit addition
- `xor_tree(inputs, depth=24)` - XOR reduction tree with fanout/reconvergence

### sequential_ref.py
Long-horizon sequential fixtures:
- `acc_fsm(limit, x, ticks, rst_at=None)` - Data-dependent accumulator FSM
- `lfsr_table(seed, table, ticks, table_width=32, rst_at=None)` - LFSR-indexed table selection
- `lfsr_next(state, taps=None)` - LFSR state update helper

### optimizable_ref.py
Small fixed-loop fixtures suitable for fusion:
- `roundN(in_val, R=8, ticks=None, rst_at=None)` - Fixed-round round function
- `nonlinear_island(x, y, width=64)` - Nonlinear core with linear mixing

### control_ref.py
Mux-heavy control cone fixtures:
- `pmux16(sel, a0, ..., a15)` - 16-way parallel mux
- `pmux32(sel, a0, ..., a31)` - 32-way parallel mux
- `mux_reconverge(cond1, cond2, a, b, c, d, width=64)` - Mux graph with reconvergence

### bitvector_ref.py
Rotate/slice/concat fixtures:
- `rotmix(x, width=64)` - Rotate-heavy mixing function
- `slice_concat(x, width=64)` - Pathological slice/concat recomposition

## Usage

```python
from tests.fixtures_ref import vec_add, acc_fsm, pmux16

# Combinational fixture
a = 0x00000001_00000002
b = 0x00000003_00000004
result = vec_add(a, b, N=2)

# Sequential fixture
states = acc_fsm(limit=10, x=0xFF, ticks=20)
final_acc = states[-1]["acc"]

# Control fixture
result = pmux16(sel=5, a0=0, a1=1, ..., a15=15)
```

## Design Principles

1. **Bit-accurate**: All operations match Verilog semantics exactly
2. **Deterministic**: No randomness, same inputs always produce same outputs
3. **Self-documenting**: Code structure mirrors fixture specification
4. **Type-consistent**: Uses Python integers for bit vectors, lists for arrays
5. **Testable**: Comprehensive unit tests in `test_fixtures_ref.py`

## Input/Output Conventions

- **Bit vectors**: Python integers with explicit masking
- **Multi-lane vectors**: Single integer with packed lanes (e.g., N lanes of 32 bits)
- **Sequential state**: List of state dictionaries, one per tick
- **Width parameters**: Always specified in bits

## Testing

Run all reference model tests:
```bash
.venv/bin/python -m unittest tests.test_fixtures_ref
```

Run specific category:
```bash
.venv/bin/python -m unittest tests.test_fixtures_ref.TestThroughputRef
```
