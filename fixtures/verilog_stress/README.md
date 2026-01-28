# Verilog Stress Test Fixtures

This directory contains stress test fixtures for the STC compiler, organized by category.

## Quick Start

```bash
# Run all stress tests
.venv/bin/python -m unittest discover -s tests -k stress -v

# Run specific category
.venv/bin/python -m unittest tests.test_verilog_stress_bitvector_optional -v

# Run single test
.venv/bin/python -m unittest tests.test_verilog_stress_bitvector_optional.TestVerilogStressBitvectorOptional.test_rotmix -v
```

## Current Status

**2/15 tests PASS** - See `/home/user/repos/VeryLogo/docs/TESTS_IMPLEMENTATION_STATUS.md` for details.

## Working Tests

- `bitvector__rotmix.v` - XOR/rotate/slice operations
- `bitvector__slice_concat_patho.v` - Slice/concat normalization

## Categories

### A) Throughput (4 fixtures)
Tests for scale, code size, and occupancy.

- `throughput__vec_add__N256.v` - 256-lane 32-bit vector addition (BLOCKED: no Add support)
- `throughput__vec_add__N65536.v` - 65K-lane stress test
- `throughput__vec_add__N1048576.v` - 1M-lane occupancy test
- `throughput__xor_diamonds__D24.v` - Deep XOR tree with reconvergence

### B) Sequential (2 fixtures)
Tests for long-horizon sequential loops.

- `sequential__acc_fsm.v` - Data-dependent accumulator FSM
- `sequential__lfsr_table__W64.v` - LFSR-indexed table with case statement

### C) Optimizable (4 fixtures)
Tests for fixed-round loops that benefit from fusion.

- `optimizable__roundN__R4.v` - 4-round fixed iteration
- `optimizable__roundN__R8.v` - 8-round fixed iteration
- `optimizable__roundN__R12.v` - 12-round fixed iteration
- `optimizable__nonlinear_island.v` - Small nonlinear core with linear mixing

### D) Control (3 fixtures)
Tests for mux-heavy control cones.

- `control__pmux16__W64.v` - 16-way mux via case statement
- `control__pmux32__W64.v` - 32-way mux via case statement
- `control__mux_reconverge__W64.v` - Multiple muxes with reconvergence

### E) Bitvector (2 fixtures)
Tests for rotate/slice/concat operations.

- `bitvector__rotmix.v` - Rotate-heavy mixing (WORKING)
- `bitvector__slice_concat_patho.v` - Pathological slice/concat (WORKING)

## Known Issues

1. **No Add/Sub support** - Blocks 6 tests (40% of suite)
2. **Test harness bug** - Layout field mismatch blocks 3 tests
3. **AVR GPIO limit** - Sequential tests exceed 8-bit GPIO
4. **Verilog function** - Extraction doesn't support function keyword

## Manual Compilation

```bash
# Compile to TickIR (bypasses lowering)
.venv/bin/python -m stc fixtures/verilog_stress/throughput__vec_add__N256.v \
  --top vec_add --out /tmp/test --bound 1 --no-backend

# Compile working fixtures (full pipeline)
.venv/bin/python -m stc fixtures/verilog_stress/bitvector__rotmix.v \
  --top rotmix --out /tmp/test --bound 1 --backend x86-avx2
```

## Reference Models

Python reference implementations in `/home/user/repos/VeryLogo/tests/fixtures_ref/`:
- `throughput_ref.py`
- `sequential_ref.py`
- `optimizable_ref.py`
- `control_ref.py`
- `bitvector_ref.py`

## Documentation

- `docs/tests.md` - Original test plan specification
- `docs/TESTS_IMPLEMENTATION_STATUS.md` - Detailed validation report
- `docs/TEST_VALIDATION_SUMMARY.txt` - Quick reference summary
