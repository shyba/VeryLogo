# Reference Model Examples

Complete usage examples for all fixture reference models.

## Throughput Fixtures

### vec_add - Massive vector addition

```python
from tests.fixtures_ref import vec_add

# 4 lanes of 32-bit addition
a = 0x00000001_00000002_00000003_00000004
b = 0x00000010_00000020_00000030_00000040
result = vec_add(a, b, N=4)
# result = 0x00000011_00000022_00000033_00000044

# Handles overflow correctly (per-lane 32-bit)
a = 0xFFFFFFFF_FFFFFFFE
b = 0x00000001_00000002
result = vec_add(a, b, N=2)
# result = 0x00000000_00000000
```

### xor_tree - Fanout/reconvergence tree

```python
from tests.fixtures_ref import xor_tree

# Flat XOR (depth 0)
inputs = [0xA, 0xB, 0xC, 0xD]
result = xor_tree(inputs, depth=0)
# result = 0xA ^ 0xB ^ 0xC ^ 0xD = 4

# Tree with depth (stress CSE/reconvergence)
inputs = list(range(16))
result = xor_tree(inputs, depth=4)
# Creates diamond-shaped dependency graph
```

## Sequential Fixtures

### acc_fsm - Data-dependent accumulator

```python
from tests.fixtures_ref import acc_fsm

# Run for 10 ticks with limit=5
states = acc_fsm(limit=5, x=0x100, ticks=10)

# Access state at each tick
print(states[0])  # {"acc": 0x100, "i": 1}
print(states[4])  # {"acc": ..., "i": 5}
print(states[5])  # Stopped (i >= limit)

# With reset at tick 3
states = acc_fsm(limit=10, x=0xFF, ticks=8, rst_at=[3])
print(states[3])  # {"acc": 0, "i": 0}
```

### lfsr_table - LFSR-indexed table lookup

```python
from tests.fixtures_ref import lfsr_table, lfsr_next

# Standalone LFSR update
state = 0x12345678
next_state = lfsr_next(state)

# Full LFSR table with 32 entries
table = [i * 0x11111111 for i in range(32)]
states = lfsr_table(seed=1, table=table, ticks=100, table_width=32)

# Each state has lfsr and acc
for s in states[:5]:
    print(f"LFSR: {s['lfsr']:08x}, ACC: {s['acc']:08x}")

# With reset
states = lfsr_table(seed=0xDEADBEEF, table=table, ticks=50,
                    table_width=32, rst_at=[20])
```

## Optimizable Fixtures

### roundN - Fixed-round cipher-like function

```python
from tests.fixtures_ref import roundN

# 8 rounds, run for 10 ticks
states = roundN(in_val=0x123456789ABCDEF0, R=8, ticks=10)

# Initial state
print(states[0])  # {"s": in_val, "r": 0, "out": in_val}

# After all rounds
print(states[8])  # {"s": <final>, "r": 8, "out": <final>}

# With reset during execution
states = roundN(in_val=0xAAAAAAAAAAAAAAAA, R=4, ticks=10, rst_at=[5])
print(states[5])  # Reset: {"s": in_val, "r": 0, ...}
```

### nonlinear_island - Nonlinear core + linear mixing

```python
from tests.fixtures_ref import nonlinear_island

# 64-bit mixing
x = 0x123456789ABCDEF0
y = 0xFEDCBA9876543210
result = nonlinear_island(x, y)

# 32-bit mixing
x = 0x12345678
y = 0xFEDCBA98
result = nonlinear_island(x, y, width=32)

# Custom width
result = nonlinear_island(0xFF, 0xAA, width=16)
```

## Control Fixtures

### pmux16 / pmux32 - Parallel mux

```python
from tests.fixtures_ref import pmux16, pmux32

# 16-way mux
inputs = [i * 0x1111111111111111 for i in range(16)]
result = pmux16(sel=5, *inputs)
# result = inputs[5] = 0x5555555555555555

# 32-way mux
inputs32 = list(range(32))
result = pmux32(sel=15, *inputs32)
# result = 15

# Selector masked to valid range
result = pmux16(sel=0xFF, *inputs)  # sel=0xFF & 0xF = 0xF
# result = inputs[15]
```

### mux_reconverge - Mux graph with shared conditions

```python
from tests.fixtures_ref import mux_reconverge

# Two conditions, four inputs
result = mux_reconverge(
    cond1=1,
    cond2=0,
    a=0x1000,
    b=0x2000,
    c=0x3000,
    d=0x4000
)

# Different condition combinations
result1 = mux_reconverge(1, 1, 0xA, 0xB, 0xC, 0xD)
result2 = mux_reconverge(0, 0, 0xA, 0xB, 0xC, 0xD)
result3 = mux_reconverge(1, 0, 0xA, 0xB, 0xC, 0xD)

# Custom width
result = mux_reconverge(1, 1, 0xFF, 0xAA, 0x55, 0x00, width=32)
```

## Bitvector Fixtures

### rotmix - Rotate-heavy mixing

```python
from tests.fixtures_ref import rotmix

# 64-bit rotate and mix
x = 0x123456789ABCDEF0
result = rotmix(x)

# 32-bit
x = 0x12345678
result = rotmix(x, width=32)

# Zero input (shows constant)
result = rotmix(0, width=64)
# result = 0x9E3779B97F4A7C15 (constant only)
```

### slice_concat - Pathological slice/concat

```python
from tests.fixtures_ref import slice_concat

# 64-bit permutation
x = 0x123456789ABCDEF0
result = slice_concat(x, width=64)
# Permutes quarters then slices odd ranges

# 32-bit permutation
x = 0x12345678
result = slice_concat(x, width=32)
# Permutes bytes then slices

# Custom width (generic algorithm)
result = slice_concat(0xFFFF, width=16)
```

## Integration with Tests

```python
import unittest
from tests.fixtures_ref import vec_add, acc_fsm, pmux16

class TestMyFixture(unittest.TestCase):
    def test_vec_add_matches_hardware(self):
        # Generate test vectors
        a = 0x12345678_9ABCDEF0
        b = 0xFEDCBA98_76543210

        # Python reference
        expected = vec_add(a, b, N=2)

        # Compare with compiled HDL output
        actual = run_hardware_simulation(a, b)

        self.assertEqual(actual, expected)

    def test_sequential_fsm(self):
        # Run reference model
        states = acc_fsm(limit=100, x=0xDEADBEEF, ticks=150)

        # Compare final state
        expected_final = states[-1]["acc"]
        actual_final = run_hdl_for_150_ticks(...)

        self.assertEqual(actual_final, expected_final)
```
