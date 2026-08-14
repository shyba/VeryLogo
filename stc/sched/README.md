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
`CpuModel` is the richer machine-description boundary for a future floor
planner. `load_agner_forms` preserves every operand-form row (including
unknown timings and eligible pipe sets) instead of collapsing a family to one
representative instruction. `cpu_from_agner_csv` combines those rows with an
explicit machine manifest for issue width, resource capacities, register
files, and ISA features. The existing schedulers still consume the legacy
`TargetModel` projection; resource reservation and target instruction
selection are intentionally the next layer.

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
