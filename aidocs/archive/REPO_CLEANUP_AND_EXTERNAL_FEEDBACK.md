# Repository Cleanup Plan + External Feedback Triage

Date: 2026-02-16

## 1) Cleanup plan for contributor usability + internal experiments

### Objective

Make the repo understandable for external contributors while preserving velocity for internal experiment branches.

### Proposed structure (tiered)

1. **Stable core (`stc/`)**
   - Compiler/runtime primitives with stable APIs.
   - No `stc -> scripts` imports.

2. **Integrations (`integrations/`)**
   - Adapters for downstream projects (FAEST/consumer-specific ABI shims).
   - Depends on stable core only.

3. **Experiments (`experiments/`)**
   - Branch-derived kernels/benchmarks/prototypes.
   - Explicitly unstable namespace.

### Practical repository changes

1. **Reduce root noise**
   - Keep root docs minimal (`README.md`, `CONTRIBUTING.md`, `ARCHITECTURE.md`).
   - Move historical plans/reports to `docs/archive/`.

2. **Split script surfaces**
   - Move reusable libraries out of `scripts/` into importable packages.
   - Keep `scripts/` as thin CLI runners.

3. **Break monoliths**
   - Split `stc/cli.py` orchestration into staged modules (`frontend`, `opt`, `lower`, `emit`, `bench`).

4. **Formalize experiment lifecycle**
   - Add manifest per experiment (`owner`, `status`, `entrypoint`, `deps`, `expected outputs`).
   - States: `draft -> incubating -> supported -> deprecated`.

### Rollout phases

- **Phase 1 (low risk):** docs/index cleanup + boundaries policy.
- **Phase 2:** move reusable experiment logic from `scripts/` into package modules.
- **Phase 3:** CLI modularization + CI split (required fast tests vs optional GPU/ISA matrix).

---

## 2) External feedback triage (other project)

### Confirmed accurate

- Variant axes exist in `stc/aes_bp128_variant_family.py` (`KeySource`, `KEY_BITS_OPTIONS`, `CTR_GROUP_OPTIONS`).
- Manifest/dispatch exist (`family_manifest_json`, `Bp128Dispatcher`).
- `xor-accumulate` patching exists (`_apply_post_op`).
- Family is still plane-major only (`_support_notes` rejects `io_layout != "plane-major4"`).
- Family `check`/`bench` are plane-major only (`check_variant_correctness`, `benchmark_variant`).
- Axes script contains external special cases for `bytes->bytes` and `bitplanes->words`.
- Tail-aware ABI is not present in manifest ABI; ABI is currently `(..., n_threads)` only.

### Needs update / partially stale

- Feedback said `out/bp128_axes_results.json` is `324 total / 63 supported / 60 pass / 3 fail`.
- Current local file now reports `324 total / 63 supported / 63 pass / 0 fail`.
- Likely explanation: results from different run revisions/environment.

### Valid structural risk called out

- `_run_external_const_byteio` uses subprocess without forcing a controlled `PYTHONPATH`, so behavior can vary by invocation context.
- Even when currently passing locally, this is a real portability risk and should be fixed.

---

## 3) Priority fixes from feedback (recommended order)

1. **Make subprocess env explicit in axes bench**
   - In `scripts/bench_aes_bp128_axes.py`, pass controlled `env` (include repo-root `PYTHONPATH`) to `_run_external_const_byteio` subprocess calls.

2. **Add native family bytes/words support**
   - Remove special-case external paths in the axes script.
   - Implement bytes/words in `stc/aes_bp128_variant_family.py` as first-class family modes.

3. **Add tail-aware ABI**
   - Extend generated kernel signature and manifest ABI with explicit output-length/tail arguments.
   - Keep old ABI as compatibility mode during migration.

4. **Remove `stc -> scripts` dependency**
   - Move BP128 kernel template generation helpers out of `scripts/bench_aes10_bp128_cuda.py` into a package module and import from there.

5. **Stabilize contract for external consumers**
   - Publish versioned manifest fields and dispatch contract (variant ID, ABI mode, supported layouts, post-op support).

