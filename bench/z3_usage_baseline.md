# Z3 Usage Baselines

Frozen reference numbers for the Z3-backed paths, produced by
`scripts/bench_z3_usage.py`. Re-check with:

```bash
.venv/bin/python scripts/bench_z3_usage.py --check
```

The `--check` mode runs every workload and fails (non-zero exit) if a median
exceeds its frozen threshold. Thresholds have generous headroom because this
repo is developed on a shared machine with fluctuating load; the check is a
regression tripwire, not a tight performance assertion.

Frozen at commit `61fac36` + `2e23359` (CPU budgets, incremental CEGIS,
synthesis structural constraints). z3 4.13.0, loadavg ~4-5.

| workload | median (CPU) | frozen threshold |
|---|---:|---:|
| equiv proof 128-bit (CPU budget 2000ms) | ~5.5 ms | 50 ms |
| equiv proof 512-bit (CPU budget 2000ms) | ~6.0 ms | 50 ms |
| superopt mux-min pattern n=5 | ~9 ms | 200 ms |
| superopt no-equiv n=5 | ~9 ms | 200 ms |
| synthesize bit1 @ 7 gates (60s budget) | ~6-9 s | 20 s |
| synthesize 4-bit sbox (multi_output) | ~0.7 s | 3 s |
| constant_state counter bound=8 | ~3.5 ms | 50 ms |
| cli --infer-simd --autovec (no backend) | ~100 ms | 1000 ms |
| cli --infer-simd --autovec --superopt | ~105 ms | 1000 ms |

## History

Pre-change (baseline for the `61fac36` work):

| workload | median (CPU) |
|---|---:|
| equiv proof 128-bit / 512-bit | 5.0 / 5.5 ms |
| superopt mux-min pattern / no-equiv | 28.5 / 22.6 ms |
| synthesize bit1 @ 7 gates | 7.21 s (12-rep A/B; mean 10.0 s) |
| constant_state bound=8 | 3.2 ms |
| cli --autovec / +superopt | 104 / 130 ms |

After `61fac36` (CPU budgets, incremental CEGIS, structural constraints):

| workload | median (CPU) |
|---|---:|
| superopt mux-min pattern / no-equiv | ~9 / ~9 ms |
| synthesize bit1 @ 7 gates | ~5.7 s (12-rep A/B) |
| synthesize 4-bit sbox (multi_output) | ~0.7 s (greedy fast path, 13 gates) |

## Greedy fast path

`synthesize_single_output` now tries the reachable-set greedy synthesis first
for `input_bits <= 4` (CPU budget 2.5 s) and falls back to Z3 exact iterative
deepening. For the 4-input cases this is both faster and finds smaller
circuits: the 4-bit sbox synthesizes to 13 gates in ~0.7 s (the Z3 deepening
previously reported 24-27 gates in ~10 s, skipping past true minima when
per-attempt budgets expired). The exact path remains available via
`require_minimal=True` and is still the default for `input_bits > 4` (the
8-input AES S-box case, where greedy is not viable).

## Known variance

The marginal-size synthesis search is inherently hard and highly variable:
12-rep CPU-time ranges were 1.9-22 s (old) and 2.7-15 s (new) on this box.
The structural constraints buy ~25-30% on median/mean, but the variance
dominates. The 20 s synthesis threshold is intentionally loose; a gross
regression (e.g., reverting to the old encoding) is detectable via the
12-rep A/B procedure described in `aidocs/tmp_state_of_the_repo.md`.
