#!/usr/bin/env python3
"""
Analyze AES S-box circuit for ternary instruction optimization opportunities.

This script examines the 1145-gate ANF circuit to find 3-input patterns
that could be replaced with a single ternary instruction (vpternlogd/lop3).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import IncrementalOptimizer


def analyze_circuit(state):
    """Analyze circuit for 3-input pattern opportunities."""
    gates = state.gates
    input_bits = state.input_bits

    # Build usage map
    use_count = {}
    for idx, (op, left, right) in enumerate(gates):
        full_idx = input_bits + idx
        use_count[left] = use_count.get(left, 0) + 1
        if op not in ("const", "not"):
            use_count[right] = use_count.get(right, 0) + 1
    for out_idx, _ in state.outputs:
        use_count[out_idx] = use_count.get(out_idx, 0) + 1

    # Find XOR chains: a ^ b ^ c patterns
    xor_chains = 0
    xor_chain_examples = []

    for idx, (op, left, right) in enumerate(gates):
        if op != "xor":
            continue
        full_idx = input_bits + idx

        # Check if left or right is also an XOR with single use
        for child_idx in (left, right):
            if child_idx < input_bits:
                continue
            gate_idx = child_idx - input_bits
            child_op, child_left, child_right = gates[gate_idx]
            if child_op == "xor" and use_count.get(child_idx, 0) == 1:
                # Found a ^ (b ^ c) pattern
                xor_chains += 1
                if len(xor_chain_examples) < 5:
                    xor_chain_examples.append((idx, gate_idx))
                break

    # Find 3-input AND-OR patterns: (a & b) | (b & c) | (a & c) (majority)
    # This is harder to detect, so we'll count AND gates that share operands
    and_gates = [
        (idx, left, right) for idx, (op, left, right) in enumerate(gates) if op == "and"
    ]

    # Count shared operands between AND gates
    shared_and_operands = 0
    for i, (idx1, l1, r1) in enumerate(and_gates):
        for idx2, l2, r2 in and_gates[i + 1 :]:
            operands1 = {l1, r1}
            operands2 = {l2, r2}
            if operands1 & operands2:
                shared_and_operands += 1

    # Count operation types
    xor_count = sum(1 for op, _, _ in gates if op == "xor")
    and_count = sum(1 for op, _, _ in gates if op == "and")
    not_count = sum(1 for op, _, _ in gates if op == "not")
    or_count = sum(1 for op, _, _ in gates if op == "or")

    return {
        "total_gates": len(gates),
        "xor_count": xor_count,
        "and_count": and_count,
        "not_count": not_count,
        "or_count": or_count,
        "xor_chains": xor_chains,
        "shared_and_operands": shared_and_operands,
        "xor_chain_examples": xor_chain_examples,
    }


def find_actual_3input_cones(state):
    """
    Find actual 3-input cones that can be replaced with a single ternary op.

    A 3-input cone is a subtree rooted at a gate that depends on exactly 3 leaves
    (input variables or gates whose values are used elsewhere) and has >=2 internal gates.
    """
    gates = state.gates
    input_bits = state.input_bits

    # Build usage map: how many times each gate's output is used
    use_count = {}
    for idx, (op, left, right) in enumerate(gates):
        use_count[left] = use_count.get(left, 0) + 1
        if op not in ("const", "not"):
            use_count[right] = use_count.get(right, 0) + 1
    for out_idx, _ in state.outputs:
        use_count[out_idx] = use_count.get(out_idx, 0) + 1

    # For each gate, find its 3-input cone if it exists
    cones = []
    visited_gates = set()

    for root_idx, (root_op, root_left, root_right) in enumerate(gates):
        if root_op == "const":
            continue

        full_root = input_bits + root_idx

        # Try to find a 3-input cone rooted at this gate
        def collect_cone(idx, depth=0):
            """Collect gates in a potential cone, returning (leaves, internal_gates)."""
            if idx < input_bits:
                # Input variable - this is a leaf
                return ({idx}, set())

            gate_idx = idx - input_bits
            if gate_idx >= len(gates):
                return ({idx}, set())

            op, left, right = gates[gate_idx]
            if op == "const":
                return ({idx}, set())

            # If this gate is used multiple times (besides by us), it's a boundary
            if use_count.get(idx, 0) > 1 and idx != full_root:
                return ({idx}, set())

            # Only go 2 levels deep to limit cone size
            if depth > 1:
                return ({idx}, set())

            # Recursively collect from children
            left_leaves, left_internal = collect_cone(left, depth + 1)
            right_leaves, right_internal = set(), set()
            if op not in ("not",):
                right_leaves, right_internal = collect_cone(right, depth + 1)

            all_leaves = left_leaves | right_leaves
            all_internal = left_internal | right_internal | {gate_idx}

            return (all_leaves, all_internal)

        leaves, internal_gates = collect_cone(full_root)

        # Check if this is a valid 3-input cone
        if len(leaves) == 3 and len(internal_gates) >= 2:
            # Avoid overlapping cones
            if not any(g in visited_gates for g in internal_gates):
                cones.append(
                    {
                        "root": root_idx,
                        "leaves": leaves,
                        "internal_gates": internal_gates,
                        "gates_saved": len(internal_gates)
                        - 1,  # Replace N gates with 1
                    }
                )
                visited_gates.update(internal_gates)

    return cones


def estimate_ternary_savings(analysis, state):
    """Estimate gate count reduction from ternary instructions."""
    cones = find_actual_3input_cones(state)

    total_gates_in_cones = sum(len(c["internal_gates"]) for c in cones)
    gates_after_ternary = len(cones)  # Each cone becomes 1 ternary op
    savings = total_gates_in_cones - gates_after_ternary

    return savings, len(cones), cones[:10]  # Return first 10 examples


def main():
    print("Generating ANF circuit for AES S-box...")
    opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
    state = opt.best_state

    print(f"\nCircuit Statistics:")
    print(f"  Total gates: {state.gate_count}")
    print(f"  Depth: {state.depth}")
    print(f"  Multiplicative depth: {state.multiplicative_depth}")

    analysis = analyze_circuit(state)

    print(f"\nOperation Breakdown:")
    print(f"  XOR gates: {analysis['xor_count']}")
    print(f"  AND gates: {analysis['and_count']}")
    print(f"  NOT gates: {analysis['not_count']}")
    print(f"  OR gates: {analysis['or_count']}")

    print(f"\nTernary Optimization Opportunities:")
    print(f"  XOR chains (a^b^c patterns): {analysis['xor_chains']}")
    print(f"  AND gates with shared operands: {analysis['shared_and_operands']}")

    savings, num_cones, example_cones = estimate_ternary_savings(analysis, state)
    optimized_gates = state.gate_count - savings
    reduction_pct = (savings / state.gate_count) * 100 if state.gate_count > 0 else 0

    print(f"\nActual 3-Input Cone Analysis:")
    print(f"  Valid 3-input cones found: {num_cones}")
    print(
        f"  Total gates in cones: {sum(len(c['internal_gates']) for c in example_cones[:num_cones])}"
    )
    print(f"  Gates after ternary: {num_cones} (1 per cone)")
    print(f"  Potential gates saved: {savings}")
    print(f"  Optimized gate count: {optimized_gates}")
    print(f"  Reduction: {reduction_pct:.1f}%")

    if example_cones:
        print(f"\n  Example cones (first 5):")
        for i, cone in enumerate(example_cones[:5]):
            print(
                f"    Cone {i+1}: root=gate[{cone['root']}], "
                f"leaves={cone['leaves']}, internal={len(cone['internal_gates'])} gates"
            )

    target = 1000
    print(f"\nProgress Toward Target:")
    print(f"  Target: {target} gates")
    print(f"  Current: {state.gate_count} gates")
    print(f"  Gap: {state.gate_count - target} gates")
    print(f"  With ternary: {optimized_gates} gates")

    if optimized_gates < target:
        print(f"  ✅ Would achieve target with ternary optimization!")
    else:
        print(f"  ❌ Still {optimized_gates - target} gates above target")
        print(f"  Additional optimization needed (Z3 superopt, etc.)")


if __name__ == "__main__":
    main()
