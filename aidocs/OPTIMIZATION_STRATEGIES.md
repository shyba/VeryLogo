# Circuit Optimization Strategies

## Current Optimizer Limitations

The window resynthesis approach finds **local minima** but may miss **global optima** because:
1. Windows are extracted independently (no cross-window optimization)
2. Linear layers are optimized separately from the non-linear core
3. No sharing of intermediate results across different windows

## Gate Weighting

Different application domains have different gate costs:

| Domain | AND Weight | XOR Weight | Notes |
|--------|------------|------------|-------|
| Software (bitslice) | 1.0 | 1.0 | Equal cost, minimize total |
| FHE (homomorphic) | 100+ | 1.0 | AND = expensive bootstrapping |
| MPC (multi-party) | 10-100 | 1.0 | AND = communication round |
| Hardware (area) | 1.5 | 1.0 | AND slightly larger |
| Hardware (depth) | 1.0 | 1.0 | Critical path matters |

Use `circuit.weighted_cost(and_weight, xor_weight)` and pass weights to `sat_window_resynthesis()`.

## Strategy 1: Linear Layer Optimization (Paar/BP Algorithm)

The AES S-box has structure:
```
Input → Top Linear → Non-linear Core → Bottom Linear → Output
```

The linear layers (XOR-only) can be optimized using GF(2) matrix techniques:

1. **Express as matrix multiplication**: Each linear layer is a binary matrix M where output = M × input (mod 2)
2. **Apply Paar's algorithm**: Greedily find pairs of rows with Hamming distance 1
3. **Apply Boyar-Peralta algorithm**: Find minimum XOR program via greedy distance minimization

This is already implemented in `stc/linear_opt.py` but has a bug. When fixed, it should reduce XOR gates.

## Strategy 2: AND Gate Minimization First

Since AND gates are often more expensive:

1. **Find minimum AND circuit**: Use SAT to find circuit with minimum AND gates (ignore XORs)
2. **Optimize XOR regions**: Apply linear optimization between AND gate boundaries
3. **Iterative refinement**: Try to reduce ANDs further with increased XOR budget

```python
# Pseudo-code
for target_ands in range(current_ands, 0, -1):
    circuit = synthesize_with_max_ands(truth_table, target_ands, max_total_gates=200)
    if circuit:
        circuit = optimize_linear_layers(circuit)
        if circuit.weighted_cost() < best.weighted_cost():
            best = circuit
```

## Strategy 3: Tower Field Decomposition

For AES S-box specifically, use algebraic structure:

1. **GF(2^8) → GF((2^4)²)**: Decompose 8-bit field into two 4-bit subfields
2. **GF(2^4) → GF((2^2)²)**: Further decompose to 2-bit subfields
3. **Inversion in small field**: 4-bit inversion needs only ~5-6 AND gates
4. **Compose operations**: Build full 8-bit inversion from small field ops

This is the approach used to get 32 AND gates (theoretical minimum for AES S-box).

## Strategy 4: SAT/SMT with Global Constraints

Instead of window-by-window synthesis:

1. **Encode entire circuit in SAT**: All gates as variables
2. **Add structural constraints**: Topological order, no cycles
3. **Minimize AND count**: Use MaxSAT or optimization modulo theories
4. **Use symmetry breaking**: Reduce search space

This is expensive but can find global optima for small circuits.

## Strategy 5: Evolutionary/Genetic Algorithms

1. **Population of circuits**: Start with multiple random/ANF circuits
2. **Mutation**: Randomly modify gates, swap operations
3. **Crossover**: Combine good subcircuits from different parents
4. **Selection**: Keep circuits with best weighted cost
5. **Local optimization**: Apply SAT synthesis to improve survivors

## Comparing Our Circuit to Optimal

| Metric | Our BP (depth-16) | Optimal BP (depth-?) | Difference |
|--------|-------------------|---------------------|------------|
| Total gates | 128 | 113-115 | +13-15 |
| AND gates | 34 | 32 | +2 |
| XOR gates | 94 | 81-83 | +11-13 |
| Mult. depth | 4 | 4 | 0 |

The difference is primarily in the linear layers, not the non-linear core.

## Implementation Priority

1. **Fix `optimize_linear_layers` bug** - Low-hanging fruit for XOR reduction
2. **Add tower field decomposition** - Reaches theoretical 32-AND minimum
3. **Global SAT synthesis** - For small subcircuits
4. **Evolutionary search** - For large-scale exploration

## Example: Using Weighted Optimization

```python
from stc.circuit_synth import CircuitState

circuit = build_sbox_circuit()

# For FHE applications: minimize ANDs
opt_fhe = circuit.sat_window_resynthesis(
    max_iterations=100,
    and_weight=100.0,  # ANDs are 100x more expensive
    xor_weight=1.0,
)

# For software: minimize total gates
opt_sw = circuit.sat_window_resynthesis(
    max_iterations=100,
    and_weight=1.0,
    xor_weight=1.0,
)
```
