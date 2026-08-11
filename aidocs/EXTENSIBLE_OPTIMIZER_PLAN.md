# VeryLogo Extensible Optimizer: Roadmap & Implementation Plan

## Executive Summary

Based on analysis of the codebase and `sboxgates.md`, this plan proposes an incremental approach that:
1. Introduces a **Technology Plugin** abstraction for backend-specific primitives (LOP3, VPTERNLOG)
2. Refactors the optimizer into a **Pass Manager** architecture
3. Adds **ternary LUT mapping** as a technology mapping pass
4. Makes **depth a first-class metric** alongside gate count
5. Provides **anytime optimization** with checkpointing

Key insight from `sboxgates.md`: The Boyar-Peralta circuit's three-layer structure (linear → non-linear → linear) and the fact that AND gates are the expensive operations in many domains (FHE, MPC) directly informs our cost model design. The `imm8` truth table concept used in BP's circuit analysis maps directly to LOP3/VPTERNLOG semantics.

---

## A) ROADMAP

### Phase 0: Foundation (Infrastructure without behavior change)
**Goal**: Introduce abstractions without changing existing behavior.

**Deliverables**:
1. `stc/tech.py` - Technology plugin interface
2. `stc/passmgr.py` - Pass manager skeleton
3. `stc/cost.py` refactored to use `CostModel` protocol
4. Unit tests proving no regression

**File Touchpoints**:
| File | Change |
|------|--------|
| `stc/tech.py` | NEW: Technology, CostModel, DepthModel protocols |
| `stc/passmgr.py` | NEW: PassManager, Pass protocol, PassContext |
| `stc/cost.py` | MODIFY: Extract interface, keep impl as `DefaultCostModel` |
| `stc/metrics.py` | MODIFY: Add `DepthModel` integration points |
| `stc/reduce.py` | MODIFY: Wrap existing logic in `CanonicalizePass`, `ConstFoldPass` |
| `tests/test_passmgr.py` | NEW: Pass manager unit tests |
| `tests/test_tech.py` | NEW: Technology abstraction tests |

### Phase 1: Ternary LUT Primitives
**Goal**: Add LOP3/VPTERNLOG as mappable primitives.

**Deliverables**:
1. `TernaryLut` expression node in tick_ir (Option 1 - justified below)
2. `stc/mapping/ternary.py` - Cone extraction and imm8 computation
3. PTX backend emits `lop3.b32`
4. x86 backend emits `vpternlogd/q`

**File Touchpoints**:
| File | Change |
|------|--------|
| `stc/tick_ir.py` | MODIFY: Add `TernaryLut` dataclass |
| `stc/mapping/__init__.py` | NEW: Mapping module |
| `stc/mapping/ternary.py` | NEW: 3-input cone extraction, imm8 synthesis |
| `stc/backend_ptx.py` | MODIFY: Emit `lop3.b32` for `TernaryLut` |
| `stc/backend_x86_avx512.py` | MODIFY: Emit `vpternlogd` for `TernaryLut` |
| `stc/interp.py` | MODIFY: Interpret `TernaryLut` |
| `stc/z3_encode.py` | MODIFY: Encode `TernaryLut` for verification |
| `tests/test_ternary_mapping.py` | NEW: imm8 correctness, cone extraction |

### Phase 2: Backend-Aware Cost & Depth
**Goal**: Make optimization backend-aware.

**Deliverables**:
1. Backend-specific `CostModel` implementations
2. `DepthModel` with AND-depth support
3. Cost/depth passed through pass context
4. CLI `--backend` flag functional

**File Touchpoints**:
| File | Change |
|------|--------|
| `stc/tech.py` | MODIFY: Add concrete `PTXTechnology`, `AVX512Technology` |
| `stc/cost.py` | MODIFY: `PTXCostModel`, `AVX512CostModel` with LUT costs |
| `stc/metrics.py` | MODIFY: `compute_depth(expr, depth_model)` |
| `stc/superopt.py` | MODIFY: Accept `CostModel` in scoring |
| `stc/cli.py` | MODIFY: Add `--backend`, `--depth-budget` flags |
| `tests/test_backend_cost.py` | NEW: Backend cost model tests |

### Phase 3: Technology Mapping Pass
**Goal**: Automatic mapping of expressions to target primitives.

**Deliverables**:
1. `TernaryMappingPass` in pass manager
2. Cone-based pattern matching
3. Profitability analysis (when LUT beats discrete gates)
4. Integration with superoptimizer (LUT as candidate)

**File Touchpoints**:
| File | Change |
|------|--------|
| `stc/mapping/ternary.py` | MODIFY: Add `TernaryMappingPass` |
| `stc/superopt.py` | MODIFY: Allow `TernaryLut` in enumeration |
| `stc/passmgr.py` | MODIFY: Register mapping pass |
| `stc/reduce.py` | MODIFY: Call mapping pass in pipeline |
| `tests/test_mapping_pass.py` | NEW: End-to-end mapping tests |

### Phase 3b: Ternary Precompute & Bounded Enumeration
**Goal**: Fast optimal synthesis for 4×4 S-boxes and small windows using precomputed reachability tables.

**Scope**: Only for small windows (16-bit truth tables). Complements Z3 superopt for larger cases.

**Deliverables**:
1. BGC(v) table: 65,536-entry mapping of 16-bit vectors to minimum TI count
2. Reachable-set tables (q0: depth-1, q1: depth-2)
3. Bounded enumeration pass for 4×4 window replacement
4. Database serialization and loading infrastructure

**File Touchpoints**:
| File | Change |
|------|--------|
| `stc/ternary_db.py` | NEW: BGC table builder, q0/q1 reachable sets, enumeration |
| `stc/passes/ternary_enumerate.py` | NEW: Pass for window replacement using precomputed DB |
| `stc/tech.py` | MODIFY: Gate check for lop3/vpternlog availability |
| `stc/mapping/ternary.py` | MODIFY: Reuse imm8 machinery for DB queries |
| `stc/passmgr.py` | MODIFY: Register enumeration pass |
| `tests/test_ternary_db.py` | NEW: BGC correctness, q0 size validation, dedup tests |
| `tests/test_ternary_enumerate.py` | NEW: End-to-end 4×4 S-box reduction tests |
| `scripts/bench_ternary_db.py` | NEW: Benchmark time/memory for table construction |

**Key Constraints**:
- **Only feasible for 4×4 S-boxes** (16-bit truth tables)
- **NOT scalable to 8×8** (would require 2^256 entries)
- Use as fast candidate generator for small cones within larger circuits
- Fall back to Z3 bounded resynthesis (Phase 4) for larger windows

### Phase 4: Depth-First Optimization
**Goal**: Make low-depth a primary objective.

**Deliverables**:
1. Tree balancing pass for associative ops
2. Depth-constrained Z3 resynthesis
3. Controlled duplication for depth reduction
4. AND-depth as special metric (for FHE/MPC)

**File Touchpoints**:
| File | Change |
|------|--------|
| `stc/passes/balance.py` | NEW: `BalanceAssociativePass` |
| `stc/passes/depth_resynth.py` | NEW: `DepthResynthesisPass` |
| `stc/z3_encode.py` | MODIFY: Add depth constraints to encoding |
| `stc/metrics.py` | MODIFY: `and_depth()`, `multiplicative_depth()` |
| `tests/test_depth_opt.py` | NEW: Depth optimization tests |

### Phase 5: Anytime Runner
**Goal**: Robust anytime optimization with checkpointing.

**Deliverables**:
1. `stc/anytime.py` - Anytime loop with checkpointing
2. SIGINT handling with safe flush
3. Deterministic seeding
4. Progress reporting

**File Touchpoints**:
| File | Change |
|------|--------|
| `stc/anytime.py` | NEW: `AnytimeRunner` class |
| `stc/cli.py` | MODIFY: `--anytime`, `--timeout`, `--checkpoint-dir` |
| `stc/reduce.py` | MODIFY: Use `AnytimeRunner` wrapper |
| `tests/test_anytime.py` | NEW: Checkpoint/restore tests |

---

## B) VERY NEXT STEPS PLAN (Immediate 1-2 PRs)

### PR #1: Technology & Pass Manager Foundation

#### New Files

**`stc/tech.py`**:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Sequence
from stc.tick_ir import Expr, BoolType, BitVecType, SimdType

class CostModel(Protocol):
    """Backend-specific cost model."""

    def expr_cost(self, expr: Expr) -> float:
        """Return cost of expression (lower is better)."""
        ...

    def and_weight(self) -> float:
        """Weight for AND gates (1.0 for software, 100+ for FHE)."""
        ...

    def xor_weight(self) -> float:
        """Weight for XOR gates."""
        ...

class DepthModel(Protocol):
    """Backend-specific depth/latency model."""

    def op_depth(self, op: str) -> int:
        """Depth contribution of an operation."""
        ...

    def is_free(self, op: str) -> bool:
        """Whether op is depth-free (e.g., NOT, wire)."""
        ...

@dataclass(frozen=True)
class Primitive:
    """A target instruction primitive."""
    name: str
    input_count: int
    output_count: int
    latency: int
    throughput: float
    constraints: dict  # e.g., {"width": 32, "lanes": 1}

class Technology(ABC):
    """Abstract technology/backend plugin."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def primitives(self) -> Sequence[Primitive]: ...

    @abstractmethod
    def cost_model(self) -> CostModel: ...

    @abstractmethod
    def depth_model(self) -> DepthModel: ...

    @abstractmethod
    def is_legal(self, expr: Expr) -> bool:
        """Check if expression is legal for this backend."""
        ...

class DefaultCostModel:
    """Software-style cost: all gates equal."""

    def expr_cost(self, expr: Expr) -> float:
        from stc.cost import expr_cost as legacy_cost
        return legacy_cost(expr, {})

    def and_weight(self) -> float:
        return 1.0

    def xor_weight(self) -> float:
        return 1.0

class DefaultDepthModel:
    """Unit depth model: each op adds 1."""

    def op_depth(self, op: str) -> int:
        if op in ("not", "const", "wire", "input"):
            return 0
        return 1

    def is_free(self, op: str) -> bool:
        return op in ("not", "const", "wire", "input")

class GenericTechnology(Technology):
    """Default technology for target-independent optimization."""

    @property
    def name(self) -> str:
        return "generic"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive("and", 2, 1, 1, 1.0, {}),
            Primitive("or", 2, 1, 1, 1.0, {}),
            Primitive("xor", 2, 1, 1, 1.0, {}),
            Primitive("not", 1, 1, 0, 1.0, {}),
        ]

    def cost_model(self) -> CostModel:
        return DefaultCostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        return True  # Everything legal in generic

# Registry for technologies
_TECHNOLOGIES: dict[str, Technology] = {}

def register_technology(tech: Technology) -> None:
    _TECHNOLOGIES[tech.name] = tech

def get_technology(name: str) -> Technology:
    if name not in _TECHNOLOGIES:
        raise ValueError(f"Unknown technology: {name}")
    return _TECHNOLOGIES[name]

def list_technologies() -> list[str]:
    return list(_TECHNOLOGIES.keys())

# Register default
register_technology(GenericTechnology())
```

**`stc/passmgr.py`**:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence
import time
import logging

from stc.tick_ir import TickIR
from stc.tech import Technology, CostModel, DepthModel

logger = logging.getLogger(__name__)

@dataclass
class PassMetrics:
    """Metrics from a pass execution."""
    changed: bool = False
    expressions_modified: int = 0
    cost_delta: float = 0.0
    depth_delta: int = 0
    time_seconds: float = 0.0

@dataclass
class PassContext:
    """Context passed to all passes."""
    technology: Technology
    cost_model: CostModel
    depth_model: DepthModel
    depth_budget: int | None = None  # Max allowed depth
    cost_budget: float | None = None  # Max allowed cost
    iteration: int = 0
    best_cost: float = float("inf")
    best_ir: TickIR | None = None
    verbosity: int = 0

    def update_best(self, ir: TickIR, cost: float) -> bool:
        """Update best-so-far if improved. Returns True if updated."""
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_ir = ir
            return True
        return False

class Pass(ABC):
    """Abstract optimization pass."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        """Run pass on IR. Returns (new_ir, metrics)."""
        ...

    def should_run(self, ctx: PassContext) -> bool:
        """Check if pass should run given context."""
        return True

@dataclass
class PassSchedule:
    """A sequence of passes to run."""
    passes: list[Pass] = field(default_factory=list)
    max_iterations: int = 100
    timeout_seconds: float | None = None
    stop_on_no_change: bool = True

    def add(self, p: Pass) -> "PassSchedule":
        self.passes.append(p)
        return self

class PassManager:
    """Manages pass execution with fixpoint iteration."""

    def __init__(self, schedule: PassSchedule):
        self.schedule = schedule
        self._interrupted = False

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, list[PassMetrics]]:
        """Run all passes to fixpoint or budget exhaustion."""
        all_metrics: list[PassMetrics] = []
        current_ir = ir
        start_time = time.time()

        for iteration in range(self.schedule.max_iterations):
            if self._interrupted:
                logger.info("Pass manager interrupted, returning best-so-far")
                break

            ctx.iteration = iteration
            iteration_changed = False

            for p in self.schedule.passes:
                if self._interrupted:
                    break

                if self.schedule.timeout_seconds:
                    elapsed = time.time() - start_time
                    if elapsed > self.schedule.timeout_seconds:
                        logger.info(f"Timeout after {elapsed:.1f}s")
                        break

                if not p.should_run(ctx):
                    continue

                pass_start = time.time()
                new_ir, metrics = p.run(current_ir, ctx)
                metrics.time_seconds = time.time() - pass_start
                all_metrics.append(metrics)

                if metrics.changed:
                    iteration_changed = True
                    current_ir = new_ir

                    # Update best-so-far
                    cost = ctx.cost_model.expr_cost(current_ir)
                    ctx.update_best(current_ir, cost)

                if ctx.verbosity > 0:
                    logger.info(f"  {p.name}: changed={metrics.changed}, "
                              f"time={metrics.time_seconds:.3f}s")

            if self.schedule.stop_on_no_change and not iteration_changed:
                logger.info(f"Fixpoint reached after {iteration + 1} iterations")
                break

        return ctx.best_ir or current_ir, all_metrics

    def interrupt(self) -> None:
        """Signal interrupt (called from signal handler)."""
        self._interrupted = True
```

#### Modifications to Existing Files

**`stc/reduce.py`** - Wrap existing logic in passes:
```python
# Add at top of file
from stc.passmgr import Pass, PassContext, PassMetrics, PassManager, PassSchedule
from stc.tech import Technology, get_technology

class ConstFoldPass(Pass):
    """Constant folding and simplification."""

    @property
    def name(self) -> str:
        return "const-fold"

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        # Wrap existing fold_constants logic
        from stc.reduce import fold_constants
        new_ir, changed = fold_constants(ir)
        return new_ir, PassMetrics(changed=changed)

class CanonicalizePass(Pass):
    """Canonicalization (commutativity, associativity normalization)."""

    @property
    def name(self) -> str:
        return "canonicalize"

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        from stc.reduce import canonicalize_ir
        new_ir, changed = canonicalize_ir(ir)
        return new_ir, PassMetrics(changed=changed)

class DeadCodePass(Pass):
    """Dead code elimination."""

    @property
    def name(self) -> str:
        return "dce"

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        from stc.dead_state import eliminate_dead_state
        new_ir = eliminate_dead_state(ir)
        changed = new_ir != ir
        return new_ir, PassMetrics(changed=changed)

def build_default_schedule(include_superopt: bool = False) -> PassSchedule:
    """Build the default optimization schedule."""
    schedule = PassSchedule(max_iterations=50)
    schedule.add(CanonicalizePass())
    schedule.add(ConstFoldPass())
    schedule.add(DeadCodePass())

    if include_superopt:
        from stc.superopt import SuperoptPass
        schedule.add(SuperoptPass())

    return schedule

# Modify optimize_tick_ir to use pass manager
def optimize_tick_ir(
    ir: TickIR,
    technology: str = "generic",
    include_superopt: bool = False,
    depth_budget: int | None = None,
    verbosity: int = 0,
) -> TickIR:
    """Main optimization entry point."""
    tech = get_technology(technology)
    ctx = PassContext(
        technology=tech,
        cost_model=tech.cost_model(),
        depth_model=tech.depth_model(),
        depth_budget=depth_budget,
        verbosity=verbosity,
    )

    schedule = build_default_schedule(include_superopt=include_superopt)
    mgr = PassManager(schedule)

    result_ir, metrics = mgr.run(ir, ctx)
    return result_ir
```

**`stc/cli.py`** - Add backend selection:
```python
# In argument parser setup
parser.add_argument(
    "--backend", "-b",
    choices=["generic", "avr", "ptx", "x86-avx2", "x86-avx512"],
    default="generic",
    help="Target backend for optimization"
)
parser.add_argument(
    "--depth-budget",
    type=int,
    default=None,
    help="Maximum allowed circuit depth"
)

# In run_pipeline, pass to optimize_tick_ir
reduced_ir = optimize_tick_ir(
    tick_ir,
    technology=args.backend,
    include_superopt=args.superopt,
    depth_budget=args.depth_budget,
    verbosity=args.verbose,
)
```

#### New Tests

**`tests/test_passmgr.py`**:
```python
import unittest
from stc.passmgr import Pass, PassContext, PassManager, PassSchedule, PassMetrics
from stc.tech import GenericTechnology
from stc.tick_ir import TickIR

class CountingPass(Pass):
    """Test pass that counts invocations."""
    def __init__(self):
        self.count = 0

    @property
    def name(self) -> str:
        return "counting"

    def run(self, ir, ctx):
        self.count += 1
        return ir, PassMetrics(changed=self.count < 3)

class TestPassManager(unittest.TestCase):
    def test_fixpoint(self):
        """Pass manager reaches fixpoint when no changes."""
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        p = CountingPass()
        schedule = PassSchedule(max_iterations=10).add(p)
        mgr = PassManager(schedule)

        ir = TickIR(inputs={}, outputs={}, state={}, next_state={})
        result, metrics = mgr.run(ir, ctx)

        # Should stop after 3 iterations (when changed becomes False)
        self.assertEqual(p.count, 3)

    def test_max_iterations(self):
        """Pass manager respects max_iterations."""
        # ... similar test
```

**`tests/test_tech.py`**:
```python
import unittest
from stc.tech import (
    Technology, CostModel, DepthModel,
    GenericTechnology, register_technology, get_technology
)

class TestTechnology(unittest.TestCase):
    def test_generic_registration(self):
        """Generic technology is registered by default."""
        tech = get_technology("generic")
        self.assertEqual(tech.name, "generic")

    def test_cost_model_protocol(self):
        """Cost model satisfies protocol."""
        tech = GenericTechnology()
        cm = tech.cost_model()
        self.assertIsInstance(cm.and_weight(), float)
        self.assertIsInstance(cm.xor_weight(), float)

    def test_depth_model_protocol(self):
        """Depth model satisfies protocol."""
        tech = GenericTechnology()
        dm = tech.depth_model()
        self.assertEqual(dm.op_depth("and"), 1)
        self.assertEqual(dm.op_depth("not"), 0)
        self.assertTrue(dm.is_free("not"))
```

---

### PR #2: Ternary LUT Primitives

#### New/Modified Files

**`stc/tick_ir.py`** - Add TernaryLut:
```python
@dataclass(frozen=True)
class TernaryLut(Expr):
    """Ternary lookup table: output[i] = truth_table[a[i]*4 + b[i]*2 + c[i]].

    Implements any 3-input boolean function per bit lane.
    Maps to PTX lop3.b32 and x86 vpternlogd/q.

    The imm8 encodes the truth table:
      bit 0: output when (a,b,c) = (0,0,0)
      bit 1: output when (a,b,c) = (0,0,1)
      bit 2: output when (a,b,c) = (0,1,0)
      ...
      bit 7: output when (a,b,c) = (1,1,1)
    """
    a: Expr
    b: Expr
    c: Expr
    imm8: int  # 0-255 truth table

    def __post_init__(self):
        assert 0 <= self.imm8 <= 255, f"imm8 must be 0-255, got {self.imm8}"

# Add to expr_children
def expr_children(expr: Expr) -> list[Expr]:
    # ... existing cases ...
    if isinstance(expr, TernaryLut):
        return [expr.a, expr.b, expr.c]
    # ...
```

**`stc/mapping/__init__.py`**:
```python
"""Technology mapping modules."""
from stc.mapping.ternary import (
    compute_imm8,
    extract_3input_cone,
    TernaryMappingPass,
)
```

**`stc/mapping/ternary.py`**:
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from stc.tick_ir import (
    Expr, And, Or, Xor, Not, Const, Input, TernaryLut,
    BoolType, BitVecType, SimdType, expr_children
)
from stc.passmgr import Pass, PassContext, PassMetrics

def compute_imm8(func: Callable[[int, int, int], int]) -> int:
    """Compute imm8 truth table from a 3-input boolean function.

    func(a, b, c) -> output_bit for each combination.

    Example:
        # AND(a, b) ignoring c
        imm8 = compute_imm8(lambda a, b, c: a & b)  # = 0x80

        # XOR(a, XOR(b, c))
        imm8 = compute_imm8(lambda a, b, c: a ^ b ^ c)  # = 0x96

        # Majority(a, b, c)
        imm8 = compute_imm8(lambda a, b, c: (a & b) | (b & c) | (a & c))  # = 0xE8
    """
    result = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        if func(a, b, c):
            result |= (1 << i)
    return result

def imm8_to_func(imm8: int) -> Callable[[int, int, int], int]:
    """Convert imm8 back to a function (for verification)."""
    def func(a: int, b: int, c: int) -> int:
        idx = (a << 2) | (b << 1) | c
        return (imm8 >> idx) & 1
    return func

# Common imm8 values
IMM8_AND_AB = compute_imm8(lambda a, b, c: a & b)  # 0x80
IMM8_OR_AB = compute_imm8(lambda a, b, c: a | b)   # 0xFE
IMM8_XOR_AB = compute_imm8(lambda a, b, c: a ^ b)  # 0x66
IMM8_XOR_ABC = compute_imm8(lambda a, b, c: a ^ b ^ c)  # 0x96
IMM8_MAJ = compute_imm8(lambda a, b, c: (a & b) | (b & c) | (a & c))  # 0xE8
IMM8_AND_ABC = compute_imm8(lambda a, b, c: a & b & c)  # 0x80
IMM8_OR_ABC = compute_imm8(lambda a, b, c: a | b | c)  # 0xFE

def negate_imm8(imm8: int) -> int:
    """Return imm8 for NOT(f) given imm8 for f."""
    return imm8 ^ 0xFF

def complement_input_a(imm8: int) -> int:
    """Return imm8 with input 'a' complemented (absorbed NOT)."""
    # Swap bits where a=0 with bits where a=1
    lo = imm8 & 0x0F  # a=0 cases (bits 0-3)
    hi = (imm8 >> 4) & 0x0F  # a=1 cases (bits 4-7)
    return (lo << 4) | hi

def complement_input_b(imm8: int) -> int:
    """Return imm8 with input 'b' complemented."""
    result = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        src_idx = (a << 2) | ((1 - b) << 1) | c
        if (imm8 >> src_idx) & 1:
            result |= (1 << i)
    return result

def complement_input_c(imm8: int) -> int:
    """Return imm8 with input 'c' complemented."""
    result = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        src_idx = (a << 2) | (b << 1) | (1 - c)
        if (imm8 >> src_idx) & 1:
            result |= (1 << i)
    return result

@dataclass
class Cone3:
    """A 3-input cone extracted from an expression tree."""
    root: Expr
    inputs: tuple[Expr, Expr, Expr]  # (a, b, c)
    imm8: int
    gate_count: int  # Number of gates in original cone

    def to_ternary_lut(self) -> TernaryLut:
        return TernaryLut(
            a=self.inputs[0],
            b=self.inputs[1],
            c=self.inputs[2],
            imm8=self.imm8,
        )

def extract_3input_cone(expr: Expr, max_depth: int = 4) -> Cone3 | None:
    """Extract a 3-input boolean cone from expression, if possible.

    Traverses the expression tree up to max_depth, collecting unique
    leaf inputs. If exactly 3 unique inputs are found, computes the
    imm8 truth table and returns a Cone3.

    Returns None if:
    - More than 3 unique inputs
    - Expression contains non-boolean ops
    - Depth exceeds max_depth
    """
    inputs_found: dict[Expr, int] = {}  # expr -> input index (0, 1, 2)
    gate_count = 0

    def collect_inputs(e: Expr, depth: int) -> bool:
        nonlocal gate_count

        if depth > max_depth:
            return False

        if isinstance(e, (Input, Const)):
            if e not in inputs_found:
                if len(inputs_found) >= 3:
                    return False  # Too many inputs
                inputs_found[e] = len(inputs_found)
            return True

        if isinstance(e, Not):
            return collect_inputs(e.a, depth)  # NOT is free

        if isinstance(e, (And, Or, Xor)):
            gate_count += 1
            return collect_inputs(e.a, depth + 1) and collect_inputs(e.b, depth + 1)

        return False  # Unknown expression type

    if not collect_inputs(expr, 0):
        return None

    if len(inputs_found) < 1:
        return None

    # Pad to exactly 3 inputs if fewer
    input_list = list(inputs_found.keys())
    while len(input_list) < 3:
        # Use constant 0 for unused inputs
        const_zero = Const(BoolType(), 0)
        input_list.append(const_zero)
        inputs_found[const_zero] = len(inputs_found)

    # Compute truth table by evaluating all 8 combinations
    def eval_expr(e: Expr, vals: dict[Expr, int]) -> int:
        if isinstance(e, Const):
            return e.value & 1
        if e in vals:
            return vals[e]
        if isinstance(e, Not):
            return 1 - eval_expr(e.a, vals)
        if isinstance(e, And):
            return eval_expr(e.a, vals) & eval_expr(e.b, vals)
        if isinstance(e, Or):
            return eval_expr(e.a, vals) | eval_expr(e.b, vals)
        if isinstance(e, Xor):
            return eval_expr(e.a, vals) ^ eval_expr(e.b, vals)
        raise ValueError(f"Cannot evaluate {type(e)}")

    imm8 = 0
    for i in range(8):
        a_val = (i >> 2) & 1
        b_val = (i >> 1) & 1
        c_val = i & 1
        vals = {
            input_list[0]: a_val,
            input_list[1]: b_val,
            input_list[2]: c_val,
        }
        if eval_expr(expr, vals):
            imm8 |= (1 << i)

    return Cone3(
        root=expr,
        inputs=(input_list[0], input_list[1], input_list[2]),
        imm8=imm8,
        gate_count=gate_count,
    )

class TernaryMappingPass(Pass):
    """Map eligible 3-input cones to TernaryLut nodes."""

    def __init__(self, min_gate_savings: int = 1):
        self.min_gate_savings = min_gate_savings

    @property
    def name(self) -> str:
        return "ternary-mapping"

    def should_run(self, ctx: PassContext) -> bool:
        # Only run if backend supports ternary LUTs
        prims = {p.name for p in ctx.technology.primitives()}
        return "lop3" in prims or "vpternlog" in prims

    def run(self, ir, ctx: PassContext) -> tuple:
        from stc.tick_ir import TickIR, expr_map

        changed = False
        mapped_count = 0

        def try_map(expr: Expr) -> Expr:
            nonlocal changed, mapped_count

            # Skip if already a TernaryLut
            if isinstance(expr, TernaryLut):
                return expr

            # Try to extract 3-input cone
            cone = extract_3input_cone(expr)
            if cone is None:
                return expr

            # Check profitability: TernaryLut costs 1, cone costs gate_count
            if cone.gate_count < 1 + self.min_gate_savings:
                return expr

            changed = True
            mapped_count += 1
            return cone.to_ternary_lut()

        # Map over all expressions in IR
        new_ir = expr_map(ir, try_map)

        return new_ir, PassMetrics(
            changed=changed,
            expressions_modified=mapped_count,
        )
```

**`stc/interp.py`** - Add TernaryLut interpretation:
```python
# In eval_expr function
if isinstance(expr, TernaryLut):
    a_val = eval_expr(expr.a, env)
    b_val = eval_expr(expr.b, env)
    c_val = eval_expr(expr.c, env)

    # Per-bit evaluation
    if isinstance(a_val, int):
        result = 0
        width = expr.a.type.width if hasattr(expr.a.type, 'width') else 1
        for bit in range(width):
            a_bit = (a_val >> bit) & 1
            b_bit = (b_val >> bit) & 1
            c_bit = (c_val >> bit) & 1
            idx = (a_bit << 2) | (b_bit << 1) | c_bit
            out_bit = (expr.imm8 >> idx) & 1
            result |= (out_bit << bit)
        return result
    # ... handle SIMD case similarly
```

**`stc/backend_ptx.py`** - Emit lop3.b32:
```python
# In emit_expr function
if isinstance(expr, TernaryLut):
    a_reg = emit_expr(expr.a, ctx)
    b_reg = emit_expr(expr.b, ctx)
    c_reg = emit_expr(expr.c, ctx)
    out_reg = ctx.alloc_reg()

    # PTX: lop3.b32 d, a, b, c, immLut;
    ctx.emit(f"lop3.b32 {out_reg}, {a_reg}, {b_reg}, {c_reg}, {expr.imm8};")
    return out_reg
```

**`stc/backend_x86_avx512.py`** - Emit vpternlogd:
```python
# In emit_expr function
if isinstance(expr, TernaryLut):
    # AVX-512: vpternlogd dest, src1, src2, imm8
    # Note: vpternlogd is destructive, dest is also first source
    a_reg = emit_expr(expr.a, ctx)
    b_reg = emit_expr(expr.b, ctx)
    c_reg = emit_expr(expr.c, ctx)

    # Copy a to output (will be overwritten)
    out_reg = ctx.alloc_reg()
    ctx.emit(f"vmovdqa32 {out_reg}, {a_reg}")
    ctx.emit(f"vpternlogd {out_reg}, {b_reg}, {c_reg}, {expr.imm8}")
    return out_reg
```

#### New Tests

**`tests/test_ternary_mapping.py`**:
```python
import unittest
from stc.mapping.ternary import (
    compute_imm8, imm8_to_func, extract_3input_cone,
    negate_imm8, complement_input_a,
    IMM8_AND_AB, IMM8_XOR_ABC, IMM8_MAJ,
)
from stc.tick_ir import And, Or, Xor, Not, Input, Const, BoolType, TernaryLut

class TestImm8Computation(unittest.TestCase):
    def test_and(self):
        """AND(a, b) = 0x80."""
        imm8 = compute_imm8(lambda a, b, c: a & b)
        self.assertEqual(imm8, 0x80)

    def test_xor(self):
        """XOR(a, b) = 0x66."""
        imm8 = compute_imm8(lambda a, b, c: a ^ b)
        self.assertEqual(imm8, 0x66)

    def test_xor3(self):
        """XOR(a, b, c) = 0x96."""
        imm8 = compute_imm8(lambda a, b, c: a ^ b ^ c)
        self.assertEqual(imm8, 0x96)
        self.assertEqual(imm8, IMM8_XOR_ABC)

    def test_majority(self):
        """Majority(a, b, c) = 0xE8."""
        imm8 = compute_imm8(lambda a, b, c: (a & b) | (b & c) | (a & c))
        self.assertEqual(imm8, 0xE8)
        self.assertEqual(imm8, IMM8_MAJ)

    def test_roundtrip(self):
        """imm8 -> func -> imm8 roundtrip."""
        for imm8 in [0x00, 0x80, 0x66, 0x96, 0xE8, 0xFF]:
            func = imm8_to_func(imm8)
            recovered = compute_imm8(func)
            self.assertEqual(recovered, imm8)

    def test_negate(self):
        """NOT(AND(a,b)) via negate_imm8."""
        imm8 = negate_imm8(IMM8_AND_AB)
        # NAND = 0x7F
        self.assertEqual(imm8, 0x7F)

class TestConeExtraction(unittest.TestCase):
    def test_simple_and(self):
        """Extract cone from AND(a, b)."""
        a = Input("a", BoolType())
        b = Input("b", BoolType())
        expr = And(a, b)

        cone = extract_3input_cone(expr)
        self.assertIsNotNone(cone)
        self.assertEqual(cone.imm8, IMM8_AND_AB)
        self.assertEqual(cone.gate_count, 1)

    def test_xor_chain(self):
        """Extract cone from XOR(XOR(a, b), c)."""
        a = Input("a", BoolType())
        b = Input("b", BoolType())
        c = Input("c", BoolType())
        expr = Xor(Xor(a, b), c)

        cone = extract_3input_cone(expr)
        self.assertIsNotNone(cone)
        self.assertEqual(cone.imm8, IMM8_XOR_ABC)
        self.assertEqual(cone.gate_count, 2)

    def test_too_many_inputs(self):
        """Return None if > 3 inputs."""
        a = Input("a", BoolType())
        b = Input("b", BoolType())
        c = Input("c", BoolType())
        d = Input("d", BoolType())
        expr = And(And(a, b), And(c, d))

        cone = extract_3input_cone(expr)
        self.assertIsNone(cone)

    def test_not_absorption(self):
        """NOT absorbed into imm8."""
        a = Input("a", BoolType())
        b = Input("b", BoolType())
        expr = Not(And(a, b))  # NAND

        cone = extract_3input_cone(expr)
        self.assertIsNotNone(cone)
        self.assertEqual(cone.imm8, 0x7F)  # NAND
```

---

### PR #3: Ternary Precompute Database + Enumeration Pass (4×4/windows)

Based on Sovyn et al. "Minimization of Bitsliced Representation of 4×4 S-Boxes based on Ternary Logic Instruction".

#### Overview

Implements precomputed reachability tables for optimal 4×4 S-box synthesis using ternary instructions. This provides a **fast lookup alternative** to Z3-based superoptimization for small windows.

**Paper contributions**:
1. **BGC(v) table**: Maps each 16-bit vector to minimum ternary instruction (TI) count (0-3)
2. **q0/q1 reachable sets**: Depth-indexed tables of all vectors reachable in 1-2 TIs from base inputs
3. **Three-stage pipeline**: Precompute → Bounded exhaustive → Refinement search

**Key scalability limitation**: Only tractable for 4×4 S-boxes (16-bit truth tables). Does NOT scale to 8×8 S-boxes.

#### New Files

**`stc/ternary_db.py`**:
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path
import json

@dataclass
class TernaryDB:
    """Precomputed ternary instruction reachability database."""
    bgc: bytes  # 65536-byte array: BGC[v] = min TI count for 16-bit vector v
    q0: set[int]  # Vectors reachable in 1 TI from base {x0, x1, x2, x3}
    q1_dict: dict[frozenset[int], tuple[int, int, int]]  # q1 vectors → (a, b, imm8)
    base_vectors: tuple[int, int, int, int]  # (x0, x1, x2, x3) masks

def build_bgc_table() -> bytes:
    """Build BGC table: maps each 16-bit vector to minimum TI count.

    For each vector v in [0, 65535]:
      - Count bits in v
      - BGC[v] = 0 if v is all-0 or all-1
      - BGC[v] = 1 if v matches a base vector or NOT(base)
      - BGC[v] = 2-3 computed via reachability analysis

    Returns 65536-byte array.
    """
    bgc = bytearray(65536)

    # Initialize with pessimistic estimate (4 = unreachable)
    for i in range(65536):
        bgc[i] = 4

    # BGC[0] = 0 (constant 0)
    bgc[0] = 0
    bgc[0xFFFF] = 0  # constant 1 (all bits set)

    # Build base vectors: x0 = 0xAAAA, x1 = 0xCCCC, x2 = 0xF0F0, x3 = 0xFF00
    base = compute_base_vectors()

    # Depth 1: base vectors and their NOTs
    for v in base:
        bgc[v] = 1
        bgc[v ^ 0xFFFF] = 1

    # Depth 2: all combinations of 2 TIs
    q0 = build_q0_table(base)
    for v in q0:
        if bgc[v] > 2:
            bgc[v] = 2

    # Depth 3: combinations of 3 TIs (bounded search)
    q1 = build_q1_table(q0, base)
    for v in q1:
        if bgc[v] > 3:
            bgc[v] = 3

    return bytes(bgc)

def compute_base_vectors() -> tuple[int, int, int, int]:
    """Compute 4-bit bitsliced representation for 4 input variables.

    For 4×4 S-box with inputs x0..x3:
      - x0: least significant bit
      - x1: second bit
      - x2: third bit
      - x3: most significant bit

    Returns (v0, v1, v2, v3) where each is 16-bit vector.

    Example: For input row i (0-15), output column is S[i].
    Each output bit y_j is a boolean function over (x0, x1, x2, x3).
    """
    # Standard bitsliced encoding for 4 bits
    x0 = 0b0101010101010101  # 0xAAAA - alternating pattern
    x1 = 0b0011001100110011  # 0xCCCC - pairs
    x2 = 0b0000111100001111  # 0xF0F0 - nibbles
    x3 = 0b0000000011111111  # 0xFF00 - bytes

    return (x0, x1, x2, x3)

def build_q0_table(base_vectors: tuple[int, int, int, int]) -> set[int]:
    """Build q0: all vectors reachable in exactly 1 TI from base.

    For each imm8 in [0, 255]:
      For each selection of (a, b, c) from base_vectors or constants:
        Compute output = TernaryLut(a, b, c, imm8)
        Add to q0

    Expected size: ~936 vectors (from Sovyn paper).
    """
    q0 = set()

    # Add constants
    q0.add(0x0000)
    q0.add(0xFFFF)

    # Add base vectors
    for v in base_vectors:
        q0.add(v)
        q0.add(v ^ 0xFFFF)  # Also add NOT(v)

    # Enumerate all 1-TI outputs
    candidates = list(q0)  # Start with base + constants

    for imm8 in range(256):
        for a in candidates:
            for b in candidates:
                for c in candidates:
                    output = apply_ternary_lut(a, b, c, imm8)
                    q0.add(output)

    return q0

def build_q1_table(
    q0: set[int],
    base_vectors: tuple[int, int, int, int]
) -> dict[frozenset[int], tuple[int, int, int]]:
    """Build q1: vectors reachable in exactly 2 TIs from base.

    For each vector v in q0:
      For each imm8:
        For each triple (a, b, c) from q0:
          Compute w = TernaryLut(a, b, c, imm8)
          Store mapping: w → (a_idx, b_idx, c_idx, imm8)

    Returns dict mapping output vector to reconstruction recipe.
    Expected size: ~438,312 unique vectors (from paper).
    """
    q1_dict = {}
    q0_list = sorted(q0)

    for imm8 in range(256):
        for i, a in enumerate(q0_list):
            for j, b in enumerate(q0_list):
                for k, c in enumerate(q0_list):
                    output = apply_ternary_lut(a, b, c, imm8)

                    # Store first found recipe (could optimize for gate count)
                    if output not in q1_dict:
                        q1_dict[output] = (a, b, c, imm8)

    return q1_dict

def apply_ternary_lut(a: int, b: int, c: int, imm8: int) -> int:
    """Compute TernaryLut output for 16-bit vectors.

    For each bit position i in [0, 15]:
      a_bit = (a >> i) & 1
      b_bit = (b >> i) & 1
      c_bit = (c >> i) & 1
      idx = (a_bit << 2) | (b_bit << 1) | c_bit
      out_bit = (imm8 >> idx) & 1
      result |= (out_bit << i)
    """
    result = 0
    for i in range(16):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        c_bit = (c >> i) & 1
        idx = (a_bit << 2) | (b_bit << 1) | c_bit
        out_bit = (imm8 >> idx) & 1
        result |= (out_bit << i)
    return result

def enumerate_min_ti(
    target: int,
    base: tuple[int, int, int, int],
    max_depth: int = 3
) -> list[tuple[int, int, int, int]]:
    """Find minimum-TI sequence to construct target vector.

    Returns list of (a_idx, b_idx, c_idx, imm8) tuples representing
    the sequence of ternary instructions needed.

    Args:
      target: 16-bit target vector
      base: (x0, x1, x2, x3) input vectors
      max_depth: Maximum TI depth to search (default 3)

    Returns empty list if target is unreachable within depth bound.
    """
    # Use precomputed BGC and q0/q1 tables for fast lookup
    pass

def score_sbox(outputs: tuple[int, int, int, int]) -> int:
    """Score 4×4 S-box by summing BGC values of output vectors.

    Args:
      outputs: (y0, y1, y2, y3) - four 16-bit output vectors

    Returns: sum(BGC[y0], BGC[y1], BGC[y2], BGC[y3])

    Lower score = fewer total ternary instructions needed.
    """
    bgc = load_bgc_table()
    return sum(bgc[v] for v in outputs)

def save_db(db: TernaryDB, path: Path) -> None:
    """Serialize database to JSON."""
    data = {
        "bgc": list(db.bgc),  # Convert bytes to list for JSON
        "q0": sorted(db.q0),
        "q1": {str(k): v for k, v in db.q1_dict.items()},
        "base_vectors": db.base_vectors,
    }
    path.write_text(json.dumps(data, indent=2))

def load_db(path: Path) -> TernaryDB:
    """Load precomputed database from JSON."""
    data = json.loads(path.read_text())
    return TernaryDB(
        bgc=bytes(data["bgc"]),
        q0=set(data["q0"]),
        q1_dict={frozenset(eval(k)): tuple(v) for k, v in data["q1"].items()},
        base_vectors=tuple(data["base_vectors"]),
    )
```

**`stc/passes/ternary_enumerate.py`**:
```python
from __future__ import annotations
from stc.passmgr import Pass, PassContext, PassMetrics
from stc.tick_ir import TickIR, Expr, TernaryLut
from stc.ternary_db import TernaryDB, load_db, enumerate_min_ti

class TernaryEnumeratePass(Pass):
    """Replace small windows with optimal TI sequences from precomputed DB."""

    def __init__(self, db_path: Path | None = None, max_window_size: int = 4):
        self.db_path = db_path
        self.max_window_size = max_window_size
        self.db: TernaryDB | None = None

    @property
    def name(self) -> str:
        return "ternary-enumerate"

    def should_run(self, ctx: PassContext) -> bool:
        # Only run if backend supports ternary LUTs
        prims = {p.name for p in ctx.technology.primitives()}
        return "lop3" in prims or "vpternlog" in prims

    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]:
        # Lazy-load DB
        if self.db is None:
            if self.db_path is None:
                # Skip if no DB available
                return ir, PassMetrics(changed=False)
            self.db = load_db(self.db_path)

        # Extract 4-input windows
        # For each window: query DB for optimal TI sequence
        # Replace if improvement found

        # Implementation: extract truth tables, query bgc/q0/q1, rebuild
        changed = False
        expressions_modified = 0

        # ... window extraction and replacement logic

        return ir, PassMetrics(
            changed=changed,
            expressions_modified=expressions_modified,
        )
```

#### New Tests

**`tests/test_ternary_db.py`**:
```python
import unittest
from stc.ternary_db import (
    build_bgc_table, build_q0_table, build_q1_table,
    compute_base_vectors, apply_ternary_lut, score_sbox,
)

class TestBGCTable(unittest.TestCase):
    def test_bgc_constants(self):
        """BGC[0] = 0, BGC[0xFFFF] = 0."""
        bgc = build_bgc_table()
        self.assertEqual(bgc[0], 0)
        self.assertEqual(bgc[0xFFFF], 0)

    def test_bgc_base_vectors(self):
        """Base vectors have BGC = 1."""
        bgc = build_bgc_table()
        base = compute_base_vectors()

        for v in base:
            self.assertEqual(bgc[v], 1)
            self.assertEqual(bgc[v ^ 0xFFFF], 1)  # NOT(base) also has BGC=1

    def test_bgc_bounds(self):
        """All BGC values are in [0, 4]."""
        bgc = build_bgc_table()
        for v in range(65536):
            self.assertIn(bgc[v], [0, 1, 2, 3, 4])

class TestReachableSets(unittest.TestCase):
    def test_q0_size(self):
        """q0 should have ~936 vectors (from Sovyn paper)."""
        base = compute_base_vectors()
        q0 = build_q0_table(base)

        # Allow some tolerance for implementation differences
        self.assertGreater(len(q0), 900)
        self.assertLess(len(q0), 1000)

    def test_q0_contains_base(self):
        """q0 contains all base vectors."""
        base = compute_base_vectors()
        q0 = build_q0_table(base)

        for v in base:
            self.assertIn(v, q0)

    def test_q1_size(self):
        """q1 should have ~438,312 vectors (from paper)."""
        base = compute_base_vectors()
        q0 = build_q0_table(base)
        q1 = build_q1_table(q0, base)

        # Paper reports 438,312 unique vectors at depth 2
        # Allow tolerance for implementation variation
        self.assertGreater(len(q1), 400000)
        self.assertLess(len(q1), 500000)

    def test_no_duplicates(self):
        """q0 and q1 have no duplicate vectors."""
        base = compute_base_vectors()
        q0 = build_q0_table(base)

        # q0 is a set, so no dupes by construction
        self.assertEqual(len(q0), len(set(q0)))

class TestTernaryLut(unittest.TestCase):
    def test_apply_and(self):
        """TernaryLut with AND imm8."""
        # AND(a, b) = 0x80
        a = 0xF0F0
        b = 0xCCCC
        result = apply_ternary_lut(a, b, 0x0000, 0x80)

        # Expected: bitwise AND of a and b
        expected = a & b
        self.assertEqual(result, expected)

    def test_apply_xor3(self):
        """TernaryLut with XOR3 imm8."""
        # XOR(a, b, c) = 0x96
        a = 0xAAAA
        b = 0xCCCC
        c = 0xF0F0
        result = apply_ternary_lut(a, b, c, 0x96)

        expected = a ^ b ^ c
        self.assertEqual(result, expected)

class TestSBoxScoring(unittest.TestCase):
    def test_score_identity(self):
        """Identity S-box: outputs = inputs."""
        base = compute_base_vectors()
        score = score_sbox(base)

        # Each base vector has BGC=1, so score = 4
        self.assertEqual(score, 4)
```

**`tests/test_ternary_enumerate.py`**:
```python
import unittest
from stc.passes.ternary_enumerate import TernaryEnumeratePass
from stc.ternary_db import build_bgc_table, build_q0_table, compute_base_vectors
from stc.tick_ir import TickIR, Input, BitVecType
from stc.passmgr import PassContext
from stc.tech import GenericTechnology

class TestTernaryEnumerate(unittest.TestCase):
    def test_4x4_sbox_reduction(self):
        """End-to-end: 4×4 S-box window reduced via DB."""
        # Build minimal DB for testing
        base = compute_base_vectors()
        q0 = build_q0_table(base)
        bgc = build_bgc_table()

        # Create IR with 4-input, 4-output combinational circuit
        # (representing a 4×4 S-box)
        # ... construct test IR ...

        # Run enumeration pass
        tech = GenericTechnology()
        ctx = PassContext(
            technology=tech,
            cost_model=tech.cost_model(),
            depth_model=tech.depth_model(),
        )

        pass_instance = TernaryEnumeratePass(db_path=None)  # Use built-in DB
        new_ir, metrics = pass_instance.run(ir, ctx)

        # Verify TI count reduced
        self.assertTrue(metrics.changed)
        # ... verify output correctness via simulation
```

**`scripts/bench_ternary_db.py`**:
```python
#!/usr/bin/env python3
"""Benchmark ternary database construction time and memory."""

import sys
import time
import tracemalloc
from pathlib import Path

# Add stc to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stc.ternary_db import (
    build_bgc_table, build_q0_table, build_q1_table,
    compute_base_vectors, save_db, TernaryDB
)

def bench_bgc():
    """Benchmark BGC table construction."""
    tracemalloc.start()
    start = time.time()

    bgc = build_bgc_table()

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"BGC table:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")
    print(f"  Size: {len(bgc)} bytes (65536 expected)")

def bench_q0():
    """Benchmark q0 construction."""
    base = compute_base_vectors()

    tracemalloc.start()
    start = time.time()

    q0 = build_q0_table(base)

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nq0 table:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")
    print(f"  Entries: {len(q0)} (expected ~936)")

def bench_q1():
    """Benchmark q1 construction."""
    base = compute_base_vectors()
    q0 = build_q0_table(base)

    tracemalloc.start()
    start = time.time()

    q1 = build_q1_table(q0, base)

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nq1 table:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")
    print(f"  Entries: {len(q1)} (expected ~438,312)")

def bench_full_db():
    """Benchmark complete database build and save."""
    base = compute_base_vectors()

    tracemalloc.start()
    start = time.time()

    bgc = build_bgc_table()
    q0 = build_q0_table(base)
    q1 = build_q1_table(q0, base)
    db = TernaryDB(bgc, q0, q1, base)

    save_db(db, Path("/tmp/ternary_db.json"))

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nFull DB build + save:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")

if __name__ == "__main__":
    print("Ternary DB Construction Benchmarks")
    print("=" * 50)
    bench_bgc()
    bench_q0()
    bench_q1()
    bench_full_db()
```

#### Modifications to Existing Files

**`stc/tech.py`** - Gate availability check:
```python
# In Technology.primitives() return value, add:
# For PTX:
Primitive("lop3", 3, 1, 1, 1.0, {"width": 32}),

# For AVX-512:
Primitive("vpternlog", 3, 1, 1, 1.0, {"width": [128, 256, 512]}),
```

**`stc/mapping/ternary.py`** - Reuse imm8 utilities:
```python
# TernaryEnumeratePass imports compute_imm8, apply_ternary_lut
# to convert between boolean expressions and 16-bit truth tables
```

**`stc/passmgr.py`** - Register enumeration pass:
```python
def build_default_schedule(
    include_superopt: bool = False,
    use_ternary_db: bool = False,
    ternary_db_path: Path | None = None,
) -> PassSchedule:
    """Build the default optimization schedule."""
    schedule = PassSchedule(max_iterations=50)
    schedule.add(CanonicalizePass())
    schedule.add(ConstFoldPass())
    schedule.add(DeadCodePass())

    if use_ternary_db:
        from stc.passes.ternary_enumerate import TernaryEnumeratePass
        schedule.add(TernaryEnumeratePass(db_path=ternary_db_path))

    if include_superopt:
        from stc.superopt import SuperoptPass
        schedule.add(SuperoptPass())

    return schedule
```

#### Integration with Existing Passes

1. **Runs AFTER TernaryMappingPass (Phase 3)**: First map 3-input cones, then use DB for remaining 4-input windows
2. **Complements SuperoptPass**: Use DB for fast lookup on small windows, fall back to Z3 for complex cases
3. **Respects AnytimeRunner budgets**: Enumeration pass checks timeout/memory caps from PassContext

#### Scalability Notes

From Sovyn paper:
- **4×4 S-boxes**: 65,536 possible 16-bit vectors → tractable
- **8×8 S-boxes**: 2^256 possible 256-bit vectors → **intractable**
- **Branching factor**: q0 (depth 1) has ~936 vectors, q1 (depth 2) has ~438,312 vectors (~468x growth)

**Recommendation**: Use precompute DB only for:
- Small windows (≤4 inputs)
- 4×4 S-box layers in larger circuits
- Fast candidate generation before invoking full Z3 solver

For larger windows or general expressions, continue using Z3 bounded resynthesis from Phase 4.

---

## C) INSTRUCTION-PRIMITIVE MAPPING DESIGN

### Decision: Option 1 - Add TernaryLut to tick_ir.py

**Justification**:
1. **Simplicity**: One IR, no need for a lowered IR layer
2. **Verification**: Z3 encoding works uniformly
3. **Interpretation**: Single interpreter handles all expressions
4. **Backend flexibility**: Backends that don't support LUTs can expand them back to discrete gates

The key insight from `sboxgates.md` is that the imm8 truth table concept is fundamental - it maps directly to both LOP3 and VPTERNLOG with identical semantics.

### imm8 Computation

```python
def compute_imm8(func: Callable[[int, int, int], int]) -> int:
    """
    Truth table layout (same for LOP3 and VPTERNLOG):

    idx | a b c | bit position
    ----|-------|-------------
     0  | 0 0 0 | bit 0
     1  | 0 0 1 | bit 1
     2  | 0 1 0 | bit 2
     3  | 0 1 1 | bit 3
     4  | 1 0 0 | bit 4
     5  | 1 0 1 | bit 5
     6  | 1 1 0 | bit 6
     7  | 1 1 1 | bit 7

    Example: AND(a, b) = 0b10000000 = 0x80
    (only bit 6 and 7 are 1, where a=1 and b=1)
    """
    result = 0
    for i in range(8):
        a, b, c = (i >> 2) & 1, (i >> 1) & 1, i & 1
        if func(a, b, c):
            result |= (1 << i)
    return result
```

### 3-Input Cone Extraction

From `sboxgates.md`, we see the BP circuit groups operations into "cones" that feed into AND gates. Our cone extraction:

1. **DFS traversal** from expression root
2. **Collect unique leaf inputs** (stop at depth limit)
3. **If ≤3 inputs**: compute imm8 and return cone
4. **If >3 inputs**: return None (can't map to single LUT)

### NOT/Constant Absorption

```python
# NOT on output: flip all bits
imm8_not_f = imm8_f ^ 0xFF

# NOT on input a: swap bits 0-3 with 4-7
imm8_not_a = ((imm8 & 0x0F) << 4) | ((imm8 >> 4) & 0x0F)

# Constant input: select appropriate 4-bit slice
# If c=0 always: use bits 0,2,4,6 (even positions)
# If c=1 always: use bits 1,3,5,7 (odd positions)
```

### Bitwidth/Lane Semantics

TernaryLut operates **per-bit** identically for:
- `BoolType()`: single bit
- `BitVecType(n)`: n independent bits
- `SimdType(w, lanes)`: w*lanes independent bits

This matches both LOP3 (.b32 operates on 32 bits) and VPTERNLOG (operates on full vector).

### Legalization for Width

```python
# PTX lop3.b32 requires 32-bit operands
# For 64-bit, emit two lop3.b32 instructions
# For 128-bit (AVX), use vpternlogd with 128-bit regs
# For 256-bit (AVX2), use vpternlogd with 256-bit regs (if available)
# For 512-bit (AVX-512), use vpternlogd with 512-bit regs

def legalize_ternary_lut_ptx(lut: TernaryLut) -> list[TernaryLut]:
    """Split wide TernaryLut into 32-bit chunks for PTX."""
    width = lut.a.type.width
    if width <= 32:
        return [lut]

    chunks = []
    for i in range(0, width, 32):
        chunk_a = Extract(lut.a, i, min(i+32, width))
        chunk_b = Extract(lut.b, i, min(i+32, width))
        chunk_c = Extract(lut.c, i, min(i+32, width))
        chunks.append(TernaryLut(chunk_a, chunk_b, chunk_c, lut.imm8))
    return chunks
```

---

## D) DEPTH OPTIMIZATION STRATEGY

### Depth Model Integration

From `sboxgates.md`: "The circuit depth (longest path from input to output) is 16 gates."

```python
class DepthModel(Protocol):
    def op_depth(self, op: str) -> int: ...
    def is_free(self, op: str) -> bool: ...

class UnitDepthModel(DepthModel):
    """Every non-free op adds depth 1."""
    FREE_OPS = {"not", "const", "wire", "input", "extract", "concat"}

    def op_depth(self, op: str) -> int:
        return 0 if op in self.FREE_OPS else 1

    def is_free(self, op: str) -> bool:
        return op in self.FREE_OPS

class ANDDepthModel(DepthModel):
    """Only AND gates contribute to depth (for FHE/MPC)."""

    def op_depth(self, op: str) -> int:
        return 1 if op == "and" else 0

    def is_free(self, op: str) -> bool:
        return op != "and"

class LatencyWeightedDepthModel(DepthModel):
    """Ops have different latencies (cycles)."""
    LATENCIES = {
        "and": 1, "or": 1, "xor": 1, "not": 0,
        "add": 1, "mul": 3, "div": 10,
        "lop3": 1, "vpternlog": 1,
    }

    def op_depth(self, op: str) -> int:
        return self.LATENCIES.get(op, 1)
```

### Tree Balancing Pass

From `sboxgates.md`: XOR chains like `T6 = T1 ^ T5` can be balanced.

```python
class BalanceAssociativePass(Pass):
    """Balance associative op trees to minimize depth."""

    @property
    def name(self) -> str:
        return "balance"

    def run(self, ir, ctx: PassContext) -> tuple:
        def balance_expr(expr: Expr) -> Expr:
            if not isinstance(expr, (Xor, And, Or)):
                return expr

            # Collect all leaves of same op type
            leaves = collect_assoc_leaves(expr)
            if len(leaves) <= 2:
                return expr

            # Build balanced tree
            return build_balanced_tree(expr.__class__, leaves)

        # ... apply to IR
```

### Depth-Constrained Z3 Resynthesis

```python
class DepthResynthesisPass(Pass):
    """Use Z3 to find lower-depth implementations of small windows."""

    def __init__(self, max_window_inputs: int = 5, target_depth: int | None = None):
        self.max_window_inputs = max_window_inputs
        self.target_depth = target_depth

    def run(self, ir, ctx: PassContext) -> tuple:
        from stc.z3_encode import encode_expr
        import z3

        def resynthesize_window(expr: Expr, inputs: list[Expr]) -> Expr | None:
            """Try to find a lower-depth implementation."""
            current_depth = compute_depth(expr, ctx.depth_model)
            target = self.target_depth or current_depth - 1

            if target < 1:
                return None

            # Create Z3 variables for inputs
            z3_inputs = [z3.BitVec(f"in_{i}", 1) for i in range(len(inputs))]

            # Encode original expression's semantics
            orig_z3 = encode_expr(expr, {inp: z3_inputs[i] for i, inp in enumerate(inputs)})

            # Try to synthesize at each depth level
            for depth in range(1, target + 1):
                result = synthesize_at_depth(
                    z3_inputs, orig_z3, depth,
                    primitives=ctx.technology.primitives()
                )
                if result is not None:
                    return result

            return None

        # ... apply to windows in IR
```

### Controlled Duplication

When depth is the primary objective, some duplication is acceptable:

```python
def duplicate_for_depth(expr: Expr, ctx: PassContext) -> Expr:
    """Duplicate shared subexpressions if it reduces critical path depth."""
    depth_before = compute_depth(expr, ctx.depth_model)

    # Find critical path
    critical_nodes = find_critical_path(expr, ctx.depth_model)

    # Try duplicating non-critical uses of critical nodes
    for node in critical_nodes:
        if use_count(node, expr) > 1:
            # Duplicate this node
            new_expr = duplicate_node(expr, node)
            depth_after = compute_depth(new_expr, ctx.depth_model)

            if depth_after < depth_before:
                # Check gate budget
                gate_increase = count_gates(new_expr) - count_gates(expr)
                if gate_increase <= ctx.gate_budget:
                    return new_expr

    return expr
```

---

## E) STOP-AND-SAVE ANYTIME RUNNER

**`stc/anytime.py`**:
```python
from __future__ import annotations
import json
import os
import signal
import time
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from stc.tick_ir import TickIR, tick_ir_to_json, tick_ir_from_json
from stc.passmgr import PassManager, PassContext, PassSchedule
from stc.tech import Technology

logger = logging.getLogger(__name__)

@dataclass
class AnytimeConfig:
    """Configuration for anytime optimization."""
    timeout_seconds: float = 300.0  # 5 minutes default
    no_improvement_seconds: float = 60.0  # Stop if no improvement in 60s
    max_iterations: int = 10000
    checkpoint_interval_seconds: float = 30.0
    checkpoint_dir: Path | None = None
    seed: int | None = None  # For deterministic randomness

@dataclass
class AnytimeState:
    """State tracked during anytime optimization."""
    best_ir: TickIR | None = None
    best_cost: float = float("inf")
    best_depth: int = float("inf")
    iterations: int = 0
    start_time: float = field(default_factory=time.time)
    last_improvement_time: float = field(default_factory=time.time)
    last_checkpoint_time: float = field(default_factory=time.time)
    interrupted: bool = False

@dataclass
class Checkpoint:
    """Serializable checkpoint."""
    ir_json: dict
    cost: float
    depth: int
    iterations: int
    elapsed_seconds: float
    seed: int | None

    def to_json(self) -> str:
        return json.dumps({
            "ir": self.ir_json,
            "cost": self.cost,
            "depth": self.depth,
            "iterations": self.iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "seed": self.seed,
        }, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Checkpoint":
        d = json.loads(s)
        return cls(
            ir_json=d["ir"],
            cost=d["cost"],
            depth=d["depth"],
            iterations=d["iterations"],
            elapsed_seconds=d["elapsed_seconds"],
            seed=d.get("seed"),
        )

class AnytimeRunner:
    """Anytime optimization runner with checkpointing."""

    def __init__(
        self,
        config: AnytimeConfig,
        pass_manager: PassManager,
        technology: Technology,
    ):
        self.config = config
        self.pass_manager = pass_manager
        self.technology = technology
        self.state = AnytimeState()
        self._original_sigint = None

        if config.seed is not None:
            random.seed(config.seed)

    def run(self, ir: TickIR) -> TickIR:
        """Run optimization until stopping criteria met."""
        self._setup_signal_handler()

        try:
            self.state.best_ir = ir
            self.state.best_cost = self._compute_cost(ir)
            self.state.start_time = time.time()
            self.state.last_improvement_time = time.time()

            ctx = PassContext(
                technology=self.technology,
                cost_model=self.technology.cost_model(),
                depth_model=self.technology.depth_model(),
            )

            current_ir = ir

            while not self._should_stop():
                self.state.iterations += 1
                ctx.iteration = self.state.iterations

                # Run one iteration of pass manager
                new_ir, metrics = self.pass_manager.run(current_ir, ctx)

                # Check for improvement
                new_cost = self._compute_cost(new_ir)
                if new_cost < self.state.best_cost:
                    self.state.best_ir = new_ir
                    self.state.best_cost = new_cost
                    self.state.last_improvement_time = time.time()
                    logger.info(f"Improvement: cost={new_cost:.2f} at iteration {self.state.iterations}")

                current_ir = new_ir

                # Checkpoint if needed
                self._maybe_checkpoint()

            return self.state.best_ir

        finally:
            self._restore_signal_handler()
            self._final_checkpoint()

    def _should_stop(self) -> bool:
        """Check all stopping criteria."""
        if self.state.interrupted:
            logger.info("Stopping: interrupted")
            return True

        elapsed = time.time() - self.state.start_time
        if elapsed > self.config.timeout_seconds:
            logger.info(f"Stopping: timeout ({elapsed:.1f}s)")
            return True

        no_improvement = time.time() - self.state.last_improvement_time
        if no_improvement > self.config.no_improvement_seconds:
            logger.info(f"Stopping: no improvement for {no_improvement:.1f}s")
            return True

        if self.state.iterations >= self.config.max_iterations:
            logger.info(f"Stopping: max iterations ({self.state.iterations})")
            return True

        return False

    def _compute_cost(self, ir: TickIR) -> float:
        """Compute cost using technology's cost model."""
        return self.technology.cost_model().expr_cost(ir)

    def _setup_signal_handler(self) -> None:
        """Install SIGINT handler for graceful shutdown."""
        def handler(signum, frame):
            logger.info("SIGINT received, finishing current iteration...")
            self.state.interrupted = True

        self._original_sigint = signal.signal(signal.SIGINT, handler)

    def _restore_signal_handler(self) -> None:
        """Restore original SIGINT handler."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

    def _maybe_checkpoint(self) -> None:
        """Write checkpoint if interval elapsed."""
        if self.config.checkpoint_dir is None:
            return

        now = time.time()
        if now - self.state.last_checkpoint_time < self.config.checkpoint_interval_seconds:
            return

        self._write_checkpoint()
        self.state.last_checkpoint_time = now

    def _final_checkpoint(self) -> None:
        """Write final checkpoint with best result."""
        if self.config.checkpoint_dir is None:
            return
        self._write_checkpoint(final=True)

    def _write_checkpoint(self, final: bool = False) -> None:
        """Write checkpoint to disk."""
        if self.state.best_ir is None:
            return

        checkpoint = Checkpoint(
            ir_json=tick_ir_to_json(self.state.best_ir),
            cost=self.state.best_cost,
            depth=self.state.best_depth,
            iterations=self.state.iterations,
            elapsed_seconds=time.time() - self.state.start_time,
            seed=self.config.seed,
        )

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = self.config.checkpoint_dir / "final_checkpoint.json"
        else:
            path = self.config.checkpoint_dir / f"checkpoint_{self.state.iterations}.json"

        # Atomic write
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(checkpoint.to_json())
        tmp_path.rename(path)

        logger.debug(f"Checkpoint written: {path}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> Checkpoint:
        """Load checkpoint from disk."""
        return Checkpoint.from_json(path.read_text())
```

**CLI Integration** (`stc/cli.py`):
```python
parser.add_argument(
    "--anytime",
    action="store_true",
    help="Enable anytime optimization with checkpointing"
)
parser.add_argument(
    "--timeout",
    type=float,
    default=300.0,
    help="Optimization timeout in seconds (default: 300)"
)
parser.add_argument(
    "--no-improvement-timeout",
    type=float,
    default=60.0,
    help="Stop if no improvement for this many seconds (default: 60)"
)
parser.add_argument(
    "--checkpoint-dir",
    type=Path,
    default=None,
    help="Directory for checkpoints (default: output_dir/checkpoints)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for deterministic optimization"
)

# In run_pipeline:
if args.anytime:
    from stc.anytime import AnytimeRunner, AnytimeConfig

    config = AnytimeConfig(
        timeout_seconds=args.timeout,
        no_improvement_seconds=args.no_improvement_timeout,
        checkpoint_dir=args.checkpoint_dir or args.out / "checkpoints",
        seed=args.seed,
    )

    tech = get_technology(args.backend)
    schedule = build_default_schedule(include_superopt=args.superopt)
    mgr = PassManager(schedule)

    runner = AnytimeRunner(config, mgr, tech)
    reduced_ir = runner.run(tick_ir)
else:
    reduced_ir = optimize_tick_ir(tick_ir, technology=args.backend, ...)
```

---

## F) TERNARY PRECOMPUTE & BOUNDED ENUMERATION

### Background: Sovyn et al. Paper

"Minimization of Bitsliced Representation of 4×4 S-Boxes based on Ternary Logic Instruction" (Sovyn et al.)

**Core idea**: Instead of using search or SAT solvers to find optimal ternary instruction sequences for 4×4 S-boxes, **precompute all reachable vectors** at each depth level and store in lookup tables.

### Key Data Structures

#### 1. BGC(v) Table
- **Size**: 65,536 entries (one per 16-bit vector)
- **Content**: `BGC[v]` = minimum number of ternary instructions (TIs) needed to construct vector v
- **Values**: 0-3 for reachable vectors, 4 for unreachable
- **Storage**: 64 KB (1 byte per entry)

```python
bgc = bytearray(65536)

# Examples:
bgc[0x0000] = 0  # Constant 0
bgc[0xFFFF] = 0  # Constant 1
bgc[0xAAAA] = 1  # Base vector x0 (reachable in 1 TI)
bgc[0x5555] = 1  # NOT(x0)
bgc[0x8888] = 2  # AND(x0, x1) - requires 2 TIs (assuming specific encoding)
```

#### 2. q0 Reachable Set (Depth 1)
- **Size**: ~936 vectors
- **Content**: All 16-bit vectors reachable in exactly 1 TI from base {x0, x1, x2, x3}
- **Construction**: Enumerate all 256 imm8 values × all triples from {0, 1, x0, x1, x2, x3, NOT(x0), ...}

```python
q0 = set()

# Add constants
q0.add(0x0000)
q0.add(0xFFFF)

# Add base vectors
for v in [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00]:
    q0.add(v)
    q0.add(v ^ 0xFFFF)

# Enumerate all 1-TI outputs
for imm8 in range(256):
    for a in candidates:
        for b in candidates:
            for c in candidates:
                output = TernaryLut(a, b, c, imm8)
                q0.add(output)
```

#### 3. q1 Dictionary (Depth 2)
- **Size**: ~438,312 unique vectors
- **Content**: Mapping from output vector → (a, b, c, imm8) reconstruction recipe
- **Construction**: Enumerate all imm8 × all triples from q0
- **Branching factor**: ~468x growth from q0 to q1

```python
q1_dict = {}

for imm8 in range(256):
    for a in q0:
        for b in q0:
            for c in q0:
                output = TernaryLut(a, b, c, imm8)
                if output not in q1_dict:
                    q1_dict[output] = (a, b, c, imm8)
```

### Three-Stage Pipeline

#### Stage 1: Precompute (Offline)
- Build BGC table: ~1 second
- Build q0 table: ~5 seconds
- Build q1 table: ~2-5 minutes (depending on deduplication strategy)
- Serialize to JSON: ~10 MB file size

**Run once, cache on disk.**

#### Stage 2: Bounded Exhaustive Search (Online)
For a given 4×4 S-box with output vectors (y0, y1, y2, y3):

1. Query `score = sum(BGC[y0], BGC[y1], BGC[y2], BGC[y3])`
2. If score ≤ 4: S-box is optimally implementable with ≤4 TIs
3. For each output y_i:
   - If `BGC[y_i] == 0`: emit constant
   - If `BGC[y_i] == 1`: lookup y_i in q0, emit 1 TI
   - If `BGC[y_i] == 2`: lookup y_i in q1, emit 2 TI sequence
   - If `BGC[y_i] == 3`: run bounded search (fallback to Z3 if needed)

#### Stage 3: Refinement (Fallback)
If target vector not in q0/q1:
- Use Z3 bounded resynthesis (Phase 4)
- Or apply heuristic decomposition + recurse

### Integration with STC Pipeline

```python
# In stc/reduce.py, after Phase 3 (TernaryMappingPass):

def optimize_tick_ir(ir, technology, include_ternary_db=False, ...):
    schedule = PassSchedule()

    # Phase 0-2: Foundation, cost/depth models
    schedule.add(CanonicalizePass())
    schedule.add(ConstFoldPass())
    schedule.add(DeadCodePass())

    # Phase 3: Technology mapping (3-input cones)
    if technology.supports_ternary_luts():
        schedule.add(TernaryMappingPass())

    # Phase 3b: Ternary DB enumeration (4-input windows)
    if include_ternary_db and technology.supports_ternary_luts():
        db_path = Path("~/.cache/stc/ternary_db.json").expanduser()
        schedule.add(TernaryEnumeratePass(db_path=db_path))

    # Phase 4-5: Depth optimization, anytime runner
    # ...
```

### When to Use Precompute DB vs. Z3

| Scenario | Approach | Rationale |
|----------|----------|-----------|
| 4×4 S-box layer (AES SubBytes, etc.) | Ternary DB | Instant lookup, guaranteed optimal for depth ≤3 |
| Small 4-input window in larger circuit | Ternary DB first, fallback to Z3 | Fast candidate generation |
| 5+ input window | Z3 bounded resynthesis | Precompute intractable |
| 8×8 S-box | Z3 + decomposition | 2^256 table size infeasible |
| General expression optimization | Phase 3 + Phase 4 | Precompute not applicable |

### Performance Characteristics

From Sovyn paper and expected implementation:

| Operation | Time | Memory |
|-----------|------|--------|
| Build BGC | ~1s | ~64 KB |
| Build q0 | ~5s | ~8 KB (936 × 8 bytes) |
| Build q1 | ~2-5 min | ~50 MB (438,312 × 120 bytes) |
| Query BGC | O(1) | - |
| Query q0/q1 | O(1) hash lookup | - |
| Full 4×4 S-box synthesis | <1ms | - |

**Trade-off**: Large upfront precompute cost (one-time) for O(1) synthesis of 4×4 S-boxes.

### Limitations & Future Work

**Fundamental limitations**:
1. **Exponential blowup**: q0 → q1 grows ~468x; q1 → q2 would be intractable
2. **Fixed input size**: Only works for 4-bit inputs (16-bit truth tables)
3. **No cross-lane optimization**: Each output bit synthesized independently

**Extensions NOT recommended**:
- 8×8 S-boxes (2^256 state space)
- Cross-lane carry/borrow (breaks independence assumption)
- Depth >3 without pruning (combinatorial explosion)

**Viable extensions**:
- **Hybrid DB + Z3**: Use DB for depth ≤2, Z3 for depth 3+
- **Incremental updates**: Extend q1 with user-provided custom gates
- **Multi-objective scoring**: Weight TI count vs. depth vs. register pressure

### Example: AES S-box Synthesis

```python
# AES S-box has 4 output bits (for 4×4 bitsliced representation)
# Each output is a 16-bit truth table

# Load precomputed DB
db = load_db(Path("~/.cache/stc/ternary_db.json"))

# AES S-box output vectors (example - actual values depend on bitslicing)
y0 = 0x6927  # Hypothetical truth table for output bit 0
y1 = 0xB1D4
y2 = 0x5C3A
y3 = 0x9E7F

# Score S-box
score = db.bgc[y0] + db.bgc[y1] + db.bgc[y2] + db.bgc[y3]
print(f"AES S-box TI count: {score}")

# Synthesize each output
for i, y in enumerate([y0, y1, y2, y3]):
    ti_count = db.bgc[y]
    print(f"  y{i}: {ti_count} TIs")

    if ti_count == 1 and y in db.q0:
        print(f"    Found in q0 (depth 1)")
    elif ti_count == 2 and y in db.q1_dict:
        a, b, c, imm8 = db.q1_dict[y]
        print(f"    Found in q1: TernaryLut({a:04x}, {b:04x}, {c:04x}, {imm8:02x})")
    else:
        print(f"    Requires bounded search or Z3 fallback")
```

### API Summary

```python
# stc/ternary_db.py - Core database construction and queries
def build_bgc_table() -> bytes
def build_q0_table(base_vectors: tuple[int, int, int, int]) -> set[int]
def build_q1_table(q0: set[int], base: tuple) -> dict[int, tuple]
def compute_base_vectors() -> tuple[int, int, int, int]
def apply_ternary_lut(a: int, b: int, c: int, imm8: int) -> int
def enumerate_min_ti(target: int, base: tuple, max_depth: int = 3) -> list[tuple]
def score_sbox(outputs: tuple[int, int, int, int]) -> int
def save_db(db: TernaryDB, path: Path) -> None
def load_db(path: Path) -> TernaryDB

# stc/passes/ternary_enumerate.py - Optimization pass
class TernaryEnumeratePass(Pass):
    def __init__(self, db_path: Path | None = None, max_window_size: int = 4)
    def should_run(self, ctx: PassContext) -> bool  # Check for lop3/vpternlog
    def run(self, ir: TickIR, ctx: PassContext) -> tuple[TickIR, PassMetrics]
```

### Verification Strategy

All DB-synthesized implementations must be verified via:
1. **Equivalence checking**: Z3 bounded verification (same as Phase 4)
2. **BGC invariants**: `BGC[TernaryLut(a,b,c,imm8)] ≤ min(BGC[a], BGC[b], BGC[c]) + 1`
3. **Reachability monotonicity**: q0 ⊆ q1 ⊆ q2
4. **Golden simulation**: Compare against reference S-box implementation

---

## Summary: File Change Matrix

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 0 | `stc/tech.py`, `stc/passmgr.py`, `tests/test_*.py` | `stc/reduce.py`, `stc/cost.py`, `stc/cli.py` |
| 1 | `stc/mapping/__init__.py`, `stc/mapping/ternary.py`, `tests/test_ternary_mapping.py` | `stc/tick_ir.py`, `stc/interp.py`, `stc/z3_encode.py`, `stc/backend_ptx.py`, `stc/backend_x86_avx512.py` |
| 2 | `tests/test_backend_cost.py` | `stc/tech.py`, `stc/cost.py`, `stc/metrics.py`, `stc/superopt.py`, `stc/cli.py` |
| 3 | `tests/test_mapping_pass.py` | `stc/mapping/ternary.py`, `stc/superopt.py`, `stc/passmgr.py`, `stc/reduce.py` |
| 3b | `stc/ternary_db.py`, `stc/passes/ternary_enumerate.py`, `tests/test_ternary_db.py`, `tests/test_ternary_enumerate.py`, `scripts/bench_ternary_db.py` | `stc/tech.py`, `stc/mapping/ternary.py`, `stc/passmgr.py` |
| 4 | `stc/passes/balance.py`, `stc/passes/depth_resynth.py`, `tests/test_depth_opt.py` | `stc/z3_encode.py`, `stc/metrics.py` |
| 5 | `stc/anytime.py`, `tests/test_anytime.py` | `stc/cli.py`, `stc/reduce.py` |

This plan is **incremental**: each phase builds on the previous, existing tests continue to pass, and the optimizer improves gradually. The key abstractions (Technology, PassManager, TernaryLut) are introduced early to guide subsequent work.

**Phase 3b** introduces Sovyn et al.'s precomputed reachability tables for fast optimal synthesis of 4×4 S-boxes and small windows, complementing the Z3-based approaches in later phases.
