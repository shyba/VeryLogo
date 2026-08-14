# Circuit Scheduling Framework

This module provides instruction scheduling for compiling Boolean circuits to SIMD/GPU targets.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CircuitState   │────▶│    Scheduler    │────▶│    Schedule     │
│  (gates, deps)  │     │   (strategy)    │     │ (cycle, reg)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  TargetModel    │
                        │ (regs, latency) │
                        └─────────────────┘
```

## Components

### TargetModel (`target.py`)
Hardware constraints: registers, latencies, throughput per op type.

```python
AVX2 = TargetModel(
    name="avx2",
    registers=16,
    issue_width=4,
    latencies={"xor": 1, "and": 1},
    throughput={"xor": 3, "and": 2},
)
```

### CpuModel (`cpu_model.py`)
`CpuModel` is the machine-description boundary for the floor
planner. `load_agner_forms` preserves every operand-form row (including
unknown timings and eligible pipe sets) instead of collapsing a family to one
representative instruction. `cpu_from_agner_csv` combines those rows with an
explicit machine manifest for issue width, resource capacities, register
files, ISA features, concrete encodings, and optional memory forms. The
existing schedulers still consume the legacy `TargetModel` projection.

### Floor planner v1 (`floor_planner.py`)
The v1 path accepts an explicit value DAG (`FloorProgram`), resolves each
operation to one `InstructionForm`, reserves eligible resources with
augmenting-path matching for one-uop forms, enforces dependency latency and
form throughput, and performs non-spilling allocation in a declared register
file. It rejects unknown timing, ambiguous forms, impossible resources, and
register pressure instead of silently delegating those decisions to GCC.

```python
from stc.sched import FloorOp, FloorProgram, cpu_from_agner_csv, plan_floor

# Supply the Agner table plus the target's explicit machine manifest.
cpu = cpu_from_agner_csv(table_path, issue_width=4, resources=resources,
                         register_files=register_files, features=features)
program = FloorProgram(
    inputs=(0, 1, 2),
    outputs=(3,),
    operations=(FloorOp(0, "bitwise", (0, 1), 3, operands="v,v,v"),),
)
schedule = plan_floor(program, cpu)
```

`stc.mir.lower_floor` provides the explicit bitwise MIR subset adapter, and
`emit_x86_64_asm` emits GNU-as text directly. The v1 emitter covers
non-destructive AVX-512 integer bitwise operations and tied `VPTERNLOG`
forms for NOT/MUX/ternary logic. Constant/copy expansions, spills, memory
scheduling, and automatic instruction selection remain rejected until their
costs are represented in the machine IR.

### Floor planner v2

v2 closes the boundary between an Agner timing row and the opcode that is
actually emitted. `InstructionEncoding` records the concrete mnemonic, source
arity, tied-input and immediate contract, and required ISA features. A
`FloorOp` may carry that encoding; the planner validates it against the
selected `InstructionForm`, and the x86 emitter independently rejects a legacy
opcode override that describes a different form. The target feature manifest
must include the encoding's requirements (for example `avx512f`).

`MemoryModel` adds explicit vector load and store forms. Passing it to
`CpuModel` or `plan_floor` creates a conservative ABI envelope: used inputs
are scheduled as loads, the core DAG starts after load latency, and outputs are
scheduled as stores after the final result latency. `FloorSchedule` exposes
those memory placements and the core offset, so its total-cycle figure no
longer silently excludes the function's fixed I/O traffic. The envelope is
serial by design; load/core overlap remains a later optimization.

```python
from stc.sched import (
    avx512_memory_model,
    cpu_from_agner_csv,
    x86_encoding_specs,
)

cpu = cpu_from_agner_csv(
    table_path,
    issue_width=4,
    resources=resources,
    register_files=register_files,
    features=("avx512f", "avx512vnni"),
    encodings=x86_encoding_specs(),
    memory=avx512_memory_model(),
)
```

### Schedule (`schedule.py`)
Assignment of gates to cycles and registers.

```python
schedule.gate_cycle[g]  # cycle when gate g executes
schedule.gate_reg[g]    # physical register for gate g's output
schedule.validate(...)  # check dependency constraints
schedule.compute_stats(...) # cycles, max_live, parallelism
```

### Scheduler Protocol (`scheduler.py`)
All strategies implement this interface:

```python
class Scheduler(Protocol):
    def schedule(self, gates, input_bits, outputs, target) -> Schedule: ...
```

### ListScheduler (`list_scheduler.py`)
Priority-based list scheduling. Simple, fast, reasonable quality.

## Extensibility Points

### Adding a New Target

```python
# In target.py
RISCV_V = TargetModel(
    name="riscv_v",
    registers=32,
    issue_width=2,
    latencies={"xor": 1, "and": 1},
    throughput={"xor": 2, "and": 1},
)
```

### Adding a New Scheduler

```python
# In stc/sched/my_scheduler.py
class MyScheduler(BaseScheduler):
    @property
    def name(self) -> str:
        return "my_scheduler"

    def schedule(self, gates, input_bits, outputs, target) -> Schedule:
        # Your algorithm here
        ...
```

## Future Strategies

### 1. Pipelined Scheduler
Model instruction pipelining where issue != completion:
- Issue an instruction every cycle
- Track when results become available
- Overlap independent chains

### 2. Register-Aware Scheduler
Integrate register allocation into scheduling:
- Track live ranges during scheduling
- Prefer schedules that minimize max_live
- Insert spills only when necessary

### 3. ILP-Based Scheduler
Optimal scheduling via integer linear programming:
```
minimize: max(cycle[g] for g in gates)
subject to:
  cycle[g2] >= cycle[g1] + latency  (dependencies)
  sum(scheduled[c]) <= throughput   (per-cycle limits)
```

### 4. Modulo Scheduler
For software pipelining of loops (multiple S-box evaluations):
- Find initiation interval (II)
- Overlap iterations
- Maximize throughput over latency

### 5. SAT-Based Scheduler
Encode as Boolean satisfiability:
- Binary search on total cycles
- Find if schedule exists in K cycles
- Optimal for small circuits

### 6. Simulated Annealing
For complex cost functions:
- Start with valid schedule
- Perturb (swap, move) gates
- Accept improvements + occasional worse moves
- Good for multi-objective (cycles + registers + spills)

## Usage Example

```python
from stc.sched import list_schedule, AVX2, compute_depth

# Schedule a circuit
schedule = list_schedule(gates, input_bits, outputs, AVX2)

# Check quality
stats = schedule.compute_stats(gates, input_bits)
print(f"Cycles: {stats.total_cycles}")
print(f"Depth: {compute_depth(gates, input_bits)}")
print(f"Max live: {stats.max_live} (target has {AVX2.registers} regs)")

# Validate correctness
errors = schedule.validate(gates, input_bits, AVX2.latencies)
assert not errors
```

## Testing

```bash
python -m unittest tests.test_sched -v
```

Tests cover:
- ASAP/ALAP computation
- Dependency analysis
- Schedule validation
- Throughput limiting
- Multiple targets and strategies
