# Experiments Report (Keccak + Window Resynthesis + Depth/LUT Mapping)

This report summarizes the experiments attempted in the repository, their outcomes, and why they did or did not succeed. It is meant to be a factual record to avoid repeating unproductive directions.

## Scope

Targets:
- Keccak low-throughput and high-throughput Verilog cores (from `external-sha3-verilog`).
- TickIR optimization and mapping (depth + ternary/LUT objectives).
- SAT/Z3 windowed resynthesis.
- AVX-512 and Vulkan backend validation (correctness and performance).

Key goals:
- Reduce circuit depth and LOP3/ternary usage.
- Improve sequential correctness across large stateful cores.
- Reduce code size and compilation time without sacrificing correctness.

---

## 1) Baseline: Yosys/ABC Flow

**What was done**
- Yosys + ABC optimizations prior to TickIR lowering.
- Selective ABC variants (tuning script and flags).

**Result**
- **Only reliable source of significant improvement** on Keccak.
- Produced consistent gate reductions and stable correctness.

**Why it worked**
- ABC has mature technology-independent optimizations and balancing.
- It operates before any TickIR specifics, reducing complexity early.

**Takeaway**
- Keep ABC in the pipeline as a first-line optimizer.
- Other approaches must at least match ABC’s stability and reductions.

---

## 2) TickIR Depth-Aware Ternary Mapping

**What was attempted**
- DepthAwareTernaryPass on TickIR (critical-path ternary insertion).
- Introduced global depth budget and “revert to best-so-far” if exceeded.
- Guarded depth with a consistent depth model.

**Results**
- Low-throughput keccak: depth held at 21 after fixes.
- High-throughput keccak: depth held at 34 with a fast phase-2 mode.
- No depth improvements found; still a plateau.

**Why it didn’t reduce depth**
- Ternary mapping did not find cones that reduce depth on the Keccak cores.
- The pass is limited to small cones and strict improvements.

**Important bugfix**
- The depth-budget pass originally used `_expr_depth` (structural), while the rest of the pipeline used `compute_ir_depth` (depth model). This mismatch allowed depth to increase even when a budget was set.
- **Fix kept**: `DepthBudgetPass` now uses `compute_ir_depth(ir, ctx.depth_model)`.

**Takeaway**
- Depth-aware ternary pass is correct but not sufficient for Keccak.
- Depth budget enforcement is now reliable (bugfix retained).

---

## 3) Windowed SAT/Z3 Resynthesis

**Key features attempted**
1. **Const1 availability** in window synthesis:
   - Added constant-1 as a synthetic boundary input.
   - Goal: allow NOT via XOR with 1, and constant terms in ANF.

2. **Care-set / don’t-care constraints**:
   - Random simulation care masks on boundary inputs.
   - Optionally stateful care masks (bounded sequential traces).
   - Fixed input masks for correlated boundary signals.

3. **MFFC corrections and overlap control**:
   - Improved MFFC construction.
   - Added rejection categories and overlap accounting.

4. **Concurrency improvements**:
   - Keep the pool busy (non-blocking submission).
   - Dynamic per-window timeout (based on avg success time).
   - Resume via checkpoints and time-window summaries.

5. **Sequential correctness checks**:
   - Random-step verification and small exact checks.
   - “Window enum” checks for strict subgraph verification.

**Results**
- Many variants **broke sequential correctness** (especially stateful care).
- The safe variants (pure combinational or very conservative care sets) did not yield meaningful reductions on Keccak.
- Window resynthesis often stalled or reverted to original due to strict checks.

**Why it failed to improve Keccak**
- Keccak’s sequential structure and large state make local windows risky.
- The reachable-care mask approximations were too weak to enable real reductions.
- Enforcing sequential correctness closed the door on the most aggressive transformations.

**Takeaway**
- SAT window resynthesis is currently **not viable** for large sequential Keccak cores in its current form.
- It remains useful for small or purely combinational circuits, but cannot be trusted on full cores.

---

## 4) Sequential Correctness Validation

**What was added**
- Sequential random-step checks for whole-circuit equivalence.
- Window-level full truth table checks for small windows.

**Result**
- These checks prevented unsafe rewrites, which is good for correctness.
- They also **blocked most resynthesis gains** on Keccak.

**Takeaway**
- The correctness checks are necessary but expose that current heuristics are not safe.
- Any future reduction pass must prove correctness at the sequential level or stay within safe cones.

---

## 5) Depth-then-LOP3 Two-Phase Strategy

**What was attempted**
- Phase 1: reduce depth with depth-aware ternary.
- Phase 2: minimize LOP3 count under a depth budget.
- Added fast mode to reduce runtime for large cores.

**Results**
- Correct depth budgeting after fixes.
- No observable depth reduction on Keccak.
- Phase-2 runs were expensive for HT keccak; needed a fast path to finish.

**Why it did not reduce LOP3**
- The IR did not expose a structure where ternary demotion helps.
- Phase-2 needed aggressive demotion logic and decomposition tables not yet implemented.

**Takeaway**
- The framework is correct, but missing a strong demotion pass and better rewrite strategies.

---

## 6) AVX-512 vs Vulkan Observations

**What was attempted**
- AVX-512 pipeline used as a baseline (mature tooling).
- Vulkan (SPIR-V) path tested for parity and looped execution.

**Results**
- AVX-512 was more stable and predictable.
- Vulkan compilation often produced large unrolled code and was harder to debug.

**Takeaway**
- Use AVX-512 as the correctness reference for logic-level changes.
- Vulkan should be re-approached after AVX-512 parity is ensured.

---

## 7) High-Throughput Keccak Gate Count

**Observation**
- High-throughput core gate count ~46k after Yosys/ABC.
- Low-throughput core previously ~28k (baseline).

**Hypothesis**
- Increased gate count is due to a more aggressive pipelining/unrolling strategy in HT core.

**Takeaway**
- Expect larger baselines in HT core; use low-throughput as the baseline for optimization experiments.

---

## Summary of What Actually Worked

✅ **Yosys/ABC** (stable, consistent reductions)  
✅ **Depth budget enforcement fix** (correct behavior; bug fixed)  
⚠️ **TickIR depth-aware ternary** (safe but no improvement yet)  
❌ **SAT window resynthesis on Keccak** (sequential correctness + poor gains)  
❌ **Care-set / stateful windowing for Keccak** (not reliable at scale)  

---

## Key Reasons Attempts Failed

1. **Sequential correctness is the real constraint**  
   Many window optimizations were invalid once sequential behavior was enforced.

2. **Local windowing did not capture global structure**  
   Keccak’s structure makes isolated windows unsafe and unproductive.

3. **Depth and LOP3 goals did not align with available rewrites**  
   We lacked a strong ternary demotion table and multi-output windowing for shared logic.

4. **Pass runtime dominated by TickIR reduction**  
   Canonicalize/const-fold on large HT cores was the major cost, limiting iteration.

---

## What Was Reverted (and Why)

All changes unrelated to the working ABC flow or the depth-budget bugfix were reverted to reduce instability and avoid regressions. This includes:
- Experimental window resynthesis variants.
- Care-mask / stateful variants.
- Pool scheduling and resume logic.
- Added test-only circuits and experimental documents.

The only retained change is the depth-budget bugfix.

---

## Next Likely Productive Directions

1. **Keep ABC + tighten integration**  
   ABC should remain the default for large sequential cores.

2. **Strengthen ternary demotion with a 256-function table**  
   For LOP3 minimization under a depth budget, a deterministic ternary demotion pass is needed.

3. **Introduce safe combinational-only passes**  
   Keccak has combinational subgraphs (e.g., Chi, Theta) that can be optimized safely if isolated.

4. **Better boundary detection in TickIR**  
   Identify stable/clocked separation in a generic way to allow safe local optimization.

---

## Current Repository State

- **Only retained code change**: depth-budget uses the correct depth model in `DepthBudgetPass`.
- All experimental additions were removed to keep the baseline clean.

