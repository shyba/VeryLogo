# Emit-time stepping (x86 AVX-512 scheduled backend)

VeryLogo’s AVX-512 “scheduled” backend emits a **purely combinational** function (`circuit(in, out)`) where `in` is a packed `(inputs + state)` bit-vector and `out` is a packed `(outputs + next_state)` bit-vector. For sequential designs, host code traditionally advances state by copying `next_state -> state` every tick.

On large stateful designs, that per-tick state copy dominates runtime.

## What changed

When lowering `TickIR -> CircuitState`, the pipeline already produces an `io_layout.json` describing how many packed bits belong to primary inputs, primary outputs, and state.

When that layout information is available, the AVX-512 scheduled emitter now also emits:

- `circuit__core(in_io, st_in, out_io, st_out)` (internal)
- `circuit(in, out)` (backward compatible packed ABI)
- `circuit_steps_shared(in_io, out_io, state_in, state_out, steps)` (new)

`circuit_steps_shared` advances the design by `steps` sequential ticks using **double-buffered state pointers**:

- compute next state into `state_out`
- swap `(state_in, state_out)` pointers
- repeat

This avoids a full `memcpy`/bit-copy per tick. If `steps` is even, the final state would otherwise end up in `state_in`; the helper copies once at the end so callers always read the final state from `state_out`.

## ABI summary

All pointers are arrays of bitsliced vectors (`__m512i`), one element per packed boolean bit:

- `in_io` length = number of primary input bits
- `out_io` length = number of primary output bits
- `state_in/state_out` length = number of state bits

## Using it

- Run the normal pipeline to produce `circuit_avx512.c` and `io_layout.json`.
- Compile to a shared library and call `circuit_steps_shared(...)` via `ctypes` or native C.

The Keccak benchmark script `scripts/bench_keccak_steps_avx512.py` demonstrates this pattern end-to-end.

