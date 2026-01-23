# AES S-box Circuit Optimization Plan

## Goal
Reduce AES S-box circuit from **1145 gates** (current ANF synthesis) to **<120 gates** (near Boyar-Peralta's 115) using incremental optimization techniques.

**Important**: We do NOT import known optimal circuits as starting points. We use them only as benchmark targets to measure our progress.

---

## Current State (as of 2026-01-23)

### What We Have
- `stc/circuit_synth.py`: CircuitState representation, ANF synthesis, IncrementalOptimizer
- `stc/bitslice_codegen.py`: SIMD C code generation (AVX2/SSE2/uint64)
- `scripts/benchmark_sbox_all.py`: Comprehensive benchmark
- `external-bitsliced/`: Reference Boyar-Peralta implementation (115 gates) for comparison only

### Current Metrics
| Implementation | Gates | Performance |
|---------------|-------|-------------|
| Our ANF | 1145 | 5.1 ns/eval (AVX2) |
| Boyar-Peralta | ~115 | 0.028 ns/eval (AVX2) |

### Key Files
```
stc/circuit_synth.py      # CircuitState, gates=[(op, left, right), ...], outputs=[(idx, inv), ...]
stc/bitslice.py           # AES_SBOX_TABLE (256 entries)
scripts/synthesize_sbox.py # CLI for synthesis and optimization
```

---

## Phase 0: Foundation Improvements

### 0.1 Add Gate Breakdown Metrics
**File**: `stc/circuit_synth.py`

Add to `CircuitState`:
```python
@property
def and_count(self) -> int:
    return sum(1 for op, _, _ in self.gates if op == "and")

@property
def xor_count(self) -> int:
    return sum(1 for op, _, _ in self.gates if op == "xor")

@property
def depth(self) -> int:
    # Compute longest path from inputs to outputs
    pass

@property
def multiplicative_depth(self) -> int:
    # Compute longest path counting only AND gates
    pass
```

**Why**: Literature measures circuits as "32 AND + 83 XOR = 115 gates". We need this breakdown.

### 0.2 Add SLP (Straight-Line Program) Export
**File**: `stc/circuit_synth.py`

Add method to export in standard format:
```python
def to_slp(self) -> str:
    """Export as straight-line program (NIST/literature format).

    Format:
        t0 = x0 ^ x1
        t1 = x2 & t0
        ...
        y0 = t42
    """
```

**Why**: Enables comparison with external tools and published circuits.

### 0.3 Deterministic Optimization Runs
**File**: `stc/circuit_synth.py`

- Add explicit `seed` parameter to all random operations
- Add logging of every accepted transformation
- Add periodic checkpoint saves

```python
def optimize(self, seed: int = 0, checkpoint_interval: int = 100, log_file: str = None):
    random.seed(seed)
    # ... log each improvement to log_file
```

---

## Phase 1: Structural Optimizations (No SAT)

**Target**: 1145 → 400-500 gates

### 1.1 Hash-Consing / CSE (Common Subexpression Elimination)
**File**: `stc/circuit_synth.py`

```python
def eliminate_common_subexpressions(self) -> 'CircuitState':
    """Merge gates with identical (op, left, right) after normalization."""
    seen = {}  # (op, min(l,r), max(l,r)) -> gate_index for commutative ops
    # For each gate, check if equivalent exists, remap references
```

**Details**:
- Normalize commutative ops: `(and, 5, 3)` → `(and, 3, 5)`
- Build hash map of `(op, left, right)` → first occurrence
- Remap all references to use first occurrence
- Remove unreferenced gates

### 1.2 Algebraic Identity Rewrites
**File**: `stc/circuit_synth.py`

```python
def apply_algebraic_rewrites(self) -> 'CircuitState':
    """Apply simplification rules repeatedly until fixed point."""
```

**Rules to implement**:
```
XOR identities:
  x ^ x → 0 (constant)
  x ^ 0 → x
  x ^ 1 → NOT(x)
  (a ^ b) ^ b → a
  (a ^ b) ^ a → b

AND identities:
  x & x → x
  x & 0 → 0
  x & 1 → x
  x & NOT(x) → 0

OR identities (if used):
  x | x → x
  x | 0 → x
  x | 1 → 1

De Morgan:
  NOT(a & b) → NOT(a) | NOT(b)
  NOT(a | b) → NOT(a) & NOT(b)
```

**Implementation approach**:
1. Build use-count for each gate
2. Pattern match each gate against rules
3. When rule matches, create replacement, update references
4. Repeat until no changes

### 1.3 Dead Code Elimination
**File**: `stc/circuit_synth.py`

```python
def eliminate_dead_code(self) -> 'CircuitState':
    """Remove gates not in any output cone."""
    # 1. Mark all output indices as live
    # 2. Backward propagate: if gate is live, its inputs are live
    # 3. Remove non-live gates, renumber
```

**Already partially exists** in `_remove_unused_gates()` - verify it's complete.

### 1.4 XOR Tree Flattening
**File**: `stc/circuit_synth.py`

```python
def flatten_xor_trees(self) -> 'CircuitState':
    """Convert nested XORs to flat representation, then rebuild optimally."""
    # For each XOR gate, collect all leaf inputs (through XOR chain)
    # x ^ (y ^ z) has leaves {x, y, z}
    # Rebuild as balanced tree or linear chain depending on depth target
```

**Why**: ANF produces deeply nested XORs. Flattening enables better CSE and prepares for Phase 2.

### 1.5 Local Rewriting Passes
**File**: `stc/circuit_synth.py`

Enhance existing `IncrementalOptimizer._try_local_rewrite()`:
- Try all 2-gate windows
- Try all 3-gate windows (if affordable)
- For each window, enumerate all equivalent implementations
- Keep if gate count decreases

---

## Phase 2: Linear Layer Optimization (Boyar-Peralta Algorithm)

**Target**: 400-500 → 150-200 gates

### 2.1 Circuit Partitioning
**File**: `stc/circuit_synth.py` (new section or separate file `stc/linear_opt.py`)

```python
def partition_linear_nonlinear(state: CircuitState) -> tuple[list[LinearCone], list[int]]:
    """Separate circuit into linear cones (XOR-only) separated by AND gates.

    Returns:
        linear_cones: List of LinearCone objects (GF(2) matrices)
        and_gates: List of AND gate indices (boundaries)
    """
```

**Key insight**: AES S-box has exactly 32 AND gates. Everything else is XOR (linear over GF(2)).

### 2.2 GF(2) Matrix Representation
**File**: `stc/linear_opt.py` (new)

```python
@dataclass
class LinearCone:
    """Represents a linear (XOR-only) subcircuit as a GF(2) matrix.

    If the cone has n inputs and m outputs, this is an m×n binary matrix
    where entry [i,j] = 1 means output i depends on input j (XOR chain).
    """
    inputs: list[int]      # Input signal indices
    outputs: list[int]     # Output signal indices
    matrix: list[int]      # m integers, each is n-bit mask (rows of matrix)

    @classmethod
    def from_circuit(cls, state: CircuitState, output_indices: list[int],
                     stop_at: set[int]) -> 'LinearCone':
        """Extract linear cone by backward traversal until hitting stop_at signals."""

    def to_xor_circuit(self) -> list[tuple[str, int, int]]:
        """Convert matrix back to XOR gates using BP algorithm."""
```

### 2.3 Boyar-Peralta XOR Minimization
**File**: `stc/linear_opt.py`

```python
def boyar_peralta_minimize(matrix: list[int], n_inputs: int) -> list[tuple[int, int]]:
    """Minimize XOR count to compute matrix rows from unit vectors.

    Algorithm (greedy):
    1. Start with base = {e_0, e_1, ..., e_{n-1}} (unit vectors = inputs)
    2. While not all target rows are in base:
       a. Find pair (a, b) in base where a^b is "most useful"
          - "Useful" = reduces total Hamming distance to remaining targets
       b. Add a^b to base
    3. Return sequence of XOR operations

    Returns: List of (idx_a, idx_b) meaning "new = base[idx_a] ^ base[idx_b]"
    """
```

**Reference**: Boyar & Peralta, "A New Combinational Logic Minimization Technique with Applications to Cryptology"

### 2.4 Depth-Bounded Variant (BPD)
**File**: `stc/linear_opt.py`

```python
def boyar_peralta_depth_bounded(matrix: list[int], n_inputs: int,
                                 max_depth: int) -> list[tuple[int, int]]:
    """BP algorithm with depth constraint.

    Modification: when selecting pair (a,b), also consider depth(a^b).
    May use more gates to achieve lower depth.
    """
```

### 2.5 Integration: Optimize Linear Layers
**File**: `stc/circuit_synth.py`

```python
def optimize_linear_layers(self) -> 'CircuitState':
    """Replace all XOR regions with BP-optimized versions.

    1. Partition into linear cones and AND boundaries
    2. For each linear cone:
       a. Convert to GF(2) matrix
       b. Run BP minimization
       c. Convert back to gates
    3. Stitch together with AND gates
    4. Renumber and return
    """
```

---

## Phase 3: SAT-Based Window Resynthesis

**Target**: 150-200 → 115-125 gates

### 3.1 Structural Indexing
**File**: `stc/circuit_synth.py`

Add cached structural information:
```python
class CircuitState:
    def compute_fanouts(self) -> dict[int, list[int]]:
        """For each node, list of gates that use it as input."""

    def compute_depths(self) -> list[int]:
        """Depth of each node from inputs."""

    def compute_mffc(self, node: int) -> set[int]:
        """Maximum fanout-free cone rooted at node.

        MFFC = all nodes reachable backward from node that have
        no fanout outside the cone (except through node).
        """
```

### 3.2 Window Extraction
**File**: `stc/circuit_synth.py` or `stc/window_synth.py` (new)

```python
@dataclass
class Window:
    inputs: list[int]       # External signals feeding the window
    outputs: list[int]      # Signals used outside the window
    internal: list[int]     # Gate indices inside the window
    truth_tables: list[int] # 2^len(inputs) bit truth table per output

def extract_window(state: CircuitState, root: int,
                   max_inputs: int = 10) -> Window | None:
    """Extract resynthesizable window rooted at given node.

    1. Start with MFFC of root
    2. Expand to include related gates if under input limit
    3. Compute truth table for each output
    4. Return None if too many inputs
    """

def compute_truth_table(state: CircuitState, output: int,
                        inputs: list[int]) -> int:
    """Compute 2^k bit truth table for output in terms of inputs."""
```

### 3.3 SAT-Based Exact Synthesis for Window
**File**: `stc/window_synth.py` (new)

```python
def synthesize_exact(truth_tables: list[int], n_inputs: int,
                     max_gates: int, gate_types: list[str] = ["xor", "and"]
                    ) -> list[tuple[str, int, int]] | None:
    """Find minimum circuit implementing truth tables using SAT.

    Encoding (for g gates):
    - Variables: gate_type[i], gate_left[i], gate_right[i] for i in 0..g-1
    - Constraints:
      1. gate_left[i] < i + n_inputs (can only use previous gates or inputs)
      2. gate_right[i] < i + n_inputs
      3. For commutative ops: gate_left[i] <= gate_right[i] (symmetry breaking)
      4. Correctness: for each input combination, outputs match truth tables

    Strategy: iterative deepening
    - Try g=1, if UNSAT try g=2, etc. until SAT or g > max_gates

    Returns: gate list or None if no solution under max_gates
    """
```

**Implementation notes**:
- Use Z3 or PySAT
- Truth table correctness: can encode as bitvector operations (efficient)
- Add symmetry breaking to prune search space
- Timeout per attempt (e.g., 10 seconds)

### 3.4 Window Splicing
**File**: `stc/window_synth.py`

```python
def splice_window(state: CircuitState, window: Window,
                  new_gates: list[tuple[str, int, int]]) -> CircuitState:
    """Replace window's internal gates with new implementation.

    1. Map new gate inputs to window.inputs indices
    2. Remove old internal gates
    3. Insert new gates
    4. Update all references to old outputs to point to new outputs
    5. Renumber everything
    """
```

### 3.5 Anytime Resynthesis Loop
**File**: `stc/circuit_synth.py`

```python
def sat_window_resynthesis(self,
                           max_window_inputs: int = 10,
                           timeout_per_window: float = 10.0,
                           stop_after_no_improve: float = 300.0,
                           checkpoint_file: str = None) -> 'CircuitState':
    """Repeatedly try to improve circuit via window resynthesis.

    Loop:
    1. Select a window (round-robin, or prioritize large MFFCs)
    2. Try to synthesize with fewer gates than current
    3. If success, splice and update best
    4. If no improvement for stop_after_no_improve seconds, return

    Saves checkpoint_file periodically if provided.
    """
```

---

## Phase 4: Anytime Run Manager

### 4.1 CLI Interface
**File**: `scripts/synthesize_sbox.py`

```bash
# Basic usage
python scripts/synthesize_sbox.py --optimize --iterations 1000

# Anytime mode (runs until stopped or no progress)
python scripts/synthesize_sbox.py --anytime \
    --stop-after-no-improve 600 \
    --checkpoint best_circuit.json \
    --log optimization.log

# Resume from checkpoint
python scripts/synthesize_sbox.py --resume best_circuit.json --anytime
```

### 4.2 Signal Handling
**File**: `scripts/synthesize_sbox.py`

```python
import signal

best_circuit = None

def handle_interrupt(signum, frame):
    """Save best circuit on Ctrl+C."""
    if best_circuit and checkpoint_file:
        best_circuit.save(checkpoint_file)
        print(f"\nInterrupted. Saved best ({best_circuit.gate_count} gates) to {checkpoint_file}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)
```

### 4.3 Progress Logging
**File**: `stc/circuit_synth.py`

```python
@dataclass
class OptimizationLog:
    timestamp: float
    iteration: int
    gate_count: int
    and_count: int
    xor_count: int
    transformation: str  # e.g., "CSE", "BP_linear", "SAT_window_node_42"

def log_improvement(log_file: str, entry: OptimizationLog):
    """Append improvement to log file."""
```

---

## Success Criteria

| Phase | Target Gates | Key Technique | Validation |
|-------|-------------|---------------|------------|
| 0 | 1145 | (baseline) | All 256 inputs correct |
| 1 | 400-500 | Structural rewrites | Equivalence + metrics |
| 2 | 150-200 | BP linear optimization | Equivalence + metrics |
| 3 | 115-125 | SAT window resynthesis | Equivalence + metrics |

**Final validation**:
- Exhaustive test: all 256 S-box inputs produce correct outputs
- Gate count breakdown: ~32 AND + ~83 XOR (matching literature)
- Benchmark: within 2x of Boyar-Peralta performance

---

## Testing Strategy

### Unit Tests
```python
# test_circuit_synth.py additions

def test_cse_removes_duplicates():
    """CSE should merge identical gates."""

def test_algebraic_rewrites():
    """Test each algebraic identity."""

def test_dead_code_elimination():
    """DCE should remove unreachable gates."""

def test_bp_minimize_identity_matrix():
    """BP on identity matrix should return 0 operations."""

def test_bp_minimize_known_case():
    """BP on known matrix should match expected XOR count."""

def test_window_extraction():
    """Window should have correct truth tables."""

def test_sat_synthesis_small():
    """SAT should find optimal 2-input function circuits."""
```

### Integration Tests
```python
def test_full_optimization_4bit_sbox():
    """Optimize 4-bit S-box end-to-end, verify correctness."""

def test_full_optimization_aes_sbox():
    """Optimize 8-bit AES S-box, verify correctness, check gate count < 200."""
```

### Regression Tests
```python
def test_aes_sbox_no_regression():
    """Ensure we don't get worse than previous best."""
    # Load previous best from file, compare
```

---

## Implementation Order

1. **Week 1**: Phase 0 (metrics, SLP export, deterministic runs)
2. **Week 2**: Phase 1.1-1.3 (CSE, algebraic rewrites, DCE)
3. **Week 3**: Phase 1.4-1.5 (XOR flattening, local rewrites)
4. **Week 4**: Phase 2.1-2.2 (partitioning, GF(2) matrices)
5. **Week 5**: Phase 2.3-2.5 (BP algorithm, integration)
6. **Week 6**: Phase 3.1-3.2 (structural indexing, windows)
7. **Week 7**: Phase 3.3-3.4 (SAT synthesis, splicing)
8. **Week 8**: Phase 3.5, Phase 4 (anytime loop, CLI)

---

## Appendix A: Why 32 AND Gates?

The AES S-box computes `S(x) = A * inv(x) + c` in GF(2^8), where:
- `inv(x)` = multiplicative inverse in GF(2^8), with `inv(0) = 0`
- `A` = affine transformation matrix (linear, just XORs)
- `c` = constant vector

The inverse in GF(2^8) requires non-linear operations. The minimum known is **32 AND gates** because:
1. GF(2^8) can be decomposed as tower field: GF(2^8) = GF((2^4)^2) = GF(((2^2)^2)^2)
2. Each level of the tower requires multiplications
3. Inversion uses: `inv(x) = x^254 = x^(2^8-2)` computed via square-and-multiply
4. The 32 ANDs are intrinsic to the algebraic structure

**Implication**: Optimization focuses on XOR count (83 in BP vs hundreds in naive), not AND count.

---

## Appendix B: Boyar-Peralta Algorithm Details

### Problem Statement
Given an m×n binary matrix M, find minimum XOR sequence to compute all m row vectors from n unit vectors (inputs).

### Example
```
Matrix M (2 outputs, 3 inputs):
  [1 1 0]   <- output 0 = x0 ^ x1
  [1 0 1]   <- output 1 = x0 ^ x2

Base vectors start as:
  e0 = [1 0 0] (= x0)
  e1 = [0 1 0] (= x1)
  e2 = [0 0 1] (= x2)

Target: produce [1 1 0] and [1 0 1]

Step 1: e0 ^ e1 = [1 1 0] ✓ (matches output 0)
Step 2: e0 ^ e2 = [1 0 1] ✓ (matches output 1)

Result: 2 XOR operations (optimal for this case)
```

### Greedy Heuristic
```python
def bp_greedy(targets: set[int], base: list[int]) -> list[tuple[int, int]]:
    """
    targets: set of row vectors we need (as integers, bit-packed)
    base: list of vectors we have (starts with unit vectors)
    """
    ops = []
    while not targets.issubset(set(base)):
        best_pair = None
        best_score = -1

        # Try all pairs in base
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                candidate = base[i] ^ base[j]
                if candidate in base:
                    continue  # Already have it

                # Score = how much closer this gets us to targets
                score = sum(
                    hamming_distance(t, candidate) < min(
                        hamming_distance(t, base[i]),
                        hamming_distance(t, base[j])
                    )
                    for t in targets if t not in base
                )

                # Bonus if candidate IS a target
                if candidate in targets:
                    score += 1000

                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            raise RuntimeError("No progress possible")

        i, j = best_pair
        new_vec = base[i] ^ base[j]
        base.append(new_vec)
        ops.append((i, j))

    return ops

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')
```

### Complexity
- Naive: O(m * n) XORs (one per matrix entry)
- BP: Often O(m + k) where k is small (sublinear in entries)
- AES S-box: BP achieves 83 XORs vs ~200+ naive

---

## Appendix C: SAT Encoding for Exact Synthesis

### Variables (for g gates, k inputs)
```
For each gate i in 0..g-1:
  type[i] ∈ {XOR, AND}           # gate operation
  left[i] ∈ 0..k+i-1             # left input (input or previous gate)
  right[i] ∈ 0..k+i-1            # right input

For correctness checking:
  val[i][j] ∈ {0, 1}             # value of signal i under input assignment j
```

### Constraints
```
1. Structural:
   left[i] < k + i               # can only use earlier signals
   right[i] < k + i
   left[i] <= right[i]           # symmetry breaking (commutative ops)

2. Semantics (for each input assignment j):
   If type[i] = XOR:
     val[k+i][j] = val[left[i]][j] XOR val[right[i]][j]
   If type[i] = AND:
     val[k+i][j] = val[left[i]][j] AND val[right[i]][j]

3. Correctness:
   For each output o with truth table T:
     For each input assignment j:
       val[output_gate[o]][j] = T[j]
```

### Bitvector Optimization
Instead of per-assignment constraints, encode truth tables as bitvectors:
```python
# For k inputs, truth table is 2^k bits
# Gate semantics become bitvector operations:
tt[i] = tt[left[i]] ^ tt[right[i]]   # if XOR
tt[i] = tt[left[i]] & tt[right[i]]   # if AND

# Correctness: tt[output_gate] == target_truth_table
```

This reduces constraint count from O(g * 2^k) to O(g).

---

## Appendix D: Common Pitfalls

### 1. Gate Index Corruption
After removing gates, all indices shift. Always rebuild with fresh indices:
```python
# BAD: in-place modification with stale indices
for i, gate in enumerate(gates):
    if should_remove(i):
        del gates[i]  # indices now wrong!

# GOOD: build new list, remap all references
new_gates = []
old_to_new = {}
for i, gate in enumerate(old_gates):
    if not should_remove(i):
        old_to_new[i] = len(new_gates) + n_inputs
        new_gates.append(remap(gate, old_to_new))
```

### 2. Circular Dependencies in Rewrites
When applying `(a ^ b) ^ b → a`, ensure `b` is actually the same signal:
```python
# Must check: right operand of outer XOR == one operand of inner XOR
# Not just structurally similar, but same gate index
```

### 3. Output Inversions
CircuitState has `outputs = [(idx, inverted), ...]`. Don't forget the inversion:
```python
def evaluate(self, input_val: int) -> int:
    # ... compute values ...
    result = 0
    for bit, (idx, inv) in enumerate(self.outputs):
        val = values[idx]
        if inv:
            val = 1 - val
        result |= val << bit
    return result
```

### 4. SAT Timeout Handling
SAT on large windows can hang. Always use timeouts:
```python
solver.set("timeout", 10000)  # 10 seconds
result = solver.check()
if result == z3.unknown:
    return None  # timed out, skip this window
```

### 5. Equivalence Checking After Every Change
```python
def verify_equivalence(old: CircuitState, new: CircuitState, table: list[int]) -> bool:
    for i in range(256):
        if old.evaluate(i) != table[i] or new.evaluate(i) != table[i]:
            return False
    return True

# Call after EVERY transformation
new_state = transform(state)
assert verify_equivalence(state, new_state, AES_SBOX_TABLE)
state = new_state
```

---

## Appendix E: Checkpoint File Format

```json
{
  "version": 1,
  "timestamp": "2026-01-23T12:34:56",
  "gate_count": 423,
  "and_count": 32,
  "xor_count": 391,
  "depth": 28,
  "multiplicative_depth": 4,
  "iterations": 1547,
  "improvements": 23,
  "last_improvement_iteration": 1523,
  "circuit": {
    "input_bits": 8,
    "output_bits": 8,
    "gates": [
      ["xor", 0, 1],
      ["and", 2, 8],
      ...
    ],
    "outputs": [
      [42, false],
      [87, true],
      ...
    ]
  },
  "optimization_log": [
    {"iteration": 100, "gate_count": 1145, "transformation": "initial"},
    {"iteration": 150, "gate_count": 1102, "transformation": "CSE"},
    ...
  ]
}
```

---

## References

1. Boyar & Peralta, "A New Combinational Logic Minimization Technique" - BP algorithm
2. NIST Circuit Complexity project - 113/115 gate benchmarks
3. Stoffelen, "Optimizing S-box Implementations for Several Criteria using SAT Solvers"
4. Berkeley ABC documentation - window-based resynthesis

---

## Notes for Implementer

- **Always verify equivalence** after any transformation (256 inputs is cheap)
- **Keep the Boyar-Peralta circuit in external-bitsliced/ as benchmark only** - do not parse it as input
- **Log everything** - optimization is stochastic, need reproducibility
- **Start with small tests** - verify each phase works on 4-bit S-boxes before AES
- The 32 AND gates are fundamental (GF(2^8) inversion requires them) - focus on reducing XORs
