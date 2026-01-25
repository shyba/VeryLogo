#!/usr/bin/env python3
"""
Run aggressive ternary optimization on ANF S-box circuit targeting AVX-512.

This demonstrates the ternary mapping optimization on a circuit with room
for improvement (1145 gates -> optimized with vpternlogd).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.bitslice import AES_SBOX_TABLE
from stc.circuit_synth import IncrementalOptimizer, CircuitState
from stc.passes.depth_ternary import map_circuit_critical_path_to_ternary
from stc.sched import list_schedule, AVX512
from stc.sched.liveness import compute_live_ranges
from stc.sched.regalloc import allocate_registers
from stc.sched.emit import AVX2Emitter


def count_ternary_gates(circuit: CircuitState) -> int:
    """Count number of ternary (5-tuple) gates in circuit."""
    count = 0
    for gate in circuit.gates:
        if len(gate) == 5:
            count += 1
    return count


def find_3input_cones(state: CircuitState):
    """Find 3-input cones that can be replaced with ternary ops."""
    gates = list(state.gates)
    input_bits = state.input_bits

    # Count how many times each node is used
    use_count = {}
    for idx, gate in enumerate(gates):
        op = gate[0]
        left = gate[1] if len(gate) > 1 else 0
        right = gate[2] if len(gate) > 2 else 0
        use_count[left] = use_count.get(left, 0) + 1
        if op not in ("const", "not"):
            use_count[right] = use_count.get(right, 0) + 1
    for out_idx, _ in state.outputs:
        use_count[out_idx] = use_count.get(out_idx, 0) + 1

    cones = []
    visited_gates = set()

    for root_idx, gate in enumerate(gates):
        if len(gate) == 5:  # Already a ternary gate
            continue
        root_op = gate[0]
        if root_op not in ("xor", "and", "or"):
            continue

        full_root = input_bits + root_idx

        def collect_cone(idx, depth=0):
            if idx < input_bits:
                return ({idx}, set())
            gate_idx = idx - input_bits
            if gate_idx >= len(gates) or gate_idx < 0:
                return ({idx}, set())
            inner_gate = gates[gate_idx]
            if len(inner_gate) == 5:  # Ternary gate
                return ({idx}, set())
            op = inner_gate[0]
            if op == "const":
                return ({idx}, set())
            # If multi-use, treat as a leaf
            if use_count.get(idx, 0) > 1 and idx != full_root:
                return ({idx}, set())
            if depth > 1:
                return ({idx}, set())

            left = inner_gate[1] if len(inner_gate) > 1 else 0
            right = inner_gate[2] if len(inner_gate) > 2 else 0
            left_leaves, left_internal = collect_cone(left, depth + 1)
            right_leaves, right_internal = set(), set()
            if op not in ("not",):
                right_leaves, right_internal = collect_cone(right, depth + 1)
            return (
                left_leaves | right_leaves,
                left_internal | right_internal | {gate_idx},
            )

        leaves, internal_gates = collect_cone(full_root)

        if len(leaves) == 3 and len(internal_gates) >= 2:
            if not any(g in visited_gates for g in internal_gates):
                cones.append(
                    {
                        "root": root_idx,
                        "leaves": sorted(leaves),
                        "internal_gates": internal_gates,
                    }
                )
                visited_gates.update(internal_gates)

    return cones


def compute_truth_table(state: CircuitState, root_idx: int, leaves: list) -> int:
    """Compute the 8-bit truth table for a 3-input cone."""
    gates = state.gates
    input_bits = state.input_bits
    leaves_set = set(leaves)

    def eval_at(assignments: dict) -> int:
        node_vals = {}
        # Initialize inputs
        for i in range(input_bits):
            node_vals[i] = assignments.get(i, 0)

        for idx, gate in enumerate(gates):
            full_idx = input_bits + idx

            # If this node is a leaf, use the forced value instead of computing
            if full_idx in leaves_set:
                node_vals[full_idx] = assignments.get(full_idx, 0)
                if idx == root_idx:
                    return node_vals[full_idx]
                continue

            if len(gate) == 5:
                op, a, b, c, imm8 = gate
                a_val = node_vals.get(a, assignments.get(a, 0))
                b_val = node_vals.get(b, assignments.get(b, 0))
                c_val = node_vals.get(c, assignments.get(c, 0))
                tidx = (a_val << 2) | (b_val << 1) | c_val
                node_vals[full_idx] = (imm8 >> tidx) & 1
            else:
                op = gate[0]
                left = gate[1] if len(gate) > 1 else 0
                right = gate[2] if len(gate) > 2 else 0
                l_val = node_vals.get(left, assignments.get(left, 0))
                r_val = node_vals.get(right, assignments.get(right, 0)) if right >= 0 else 0

                if op == "xor":
                    node_vals[full_idx] = l_val ^ r_val
                elif op == "and":
                    node_vals[full_idx] = l_val & r_val
                elif op == "or":
                    node_vals[full_idx] = l_val | r_val
                elif op == "not":
                    node_vals[full_idx] = 1 - l_val
                elif op == "const":
                    node_vals[full_idx] = left
                else:
                    node_vals[full_idx] = 0

            if idx == root_idx:
                return node_vals[full_idx]
        return 0

    # CircuitState.evaluate uses: idx = (va << 2) | (vb << 1) | vc
    # where va=node_vals[a], vb=node_vals[b], vc=node_vals[c]
    # and gate is ("ternary", a, b, c, imm8) = ("ternary", leaves[0], leaves[1], leaves[2], imm8)
    # So: idx = (node_vals[leaves[0]] << 2) | (node_vals[leaves[1]] << 1) | node_vals[leaves[2]]
    # For truth table bit i to be set when idx=i:
    # - node_vals[leaves[0]] = (i >> 2) & 1
    # - node_vals[leaves[1]] = (i >> 1) & 1
    # - node_vals[leaves[2]] = i & 1
    imm8 = 0
    for i in range(8):
        assignments = {
            leaves[0]: (i >> 2) & 1,
            leaves[1]: (i >> 1) & 1,
            leaves[2]: i & 1,
        }
        if eval_at(assignments):
            imm8 |= 1 << i
    return imm8


def verify_cone_truth_table(circuit: CircuitState, root_idx: int, leaves: list, imm8: int) -> bool:
    """Verify that the truth table correctly captures the cone's function."""
    input_bits = circuit.input_bits
    leaves_set = set(leaves)

    # For each truth table entry, verify the cone output matches
    for i in range(8):
        a_val = (i >> 2) & 1
        b_val = (i >> 1) & 1
        c_val = i & 1

        # Create assignment dict
        assignments = {
            leaves[0]: a_val,
            leaves[1]: b_val,
            leaves[2]: c_val,
        }

        # Evaluate original circuit with these leaf values forced
        node_vals = [0] * input_bits
        for leaf_idx, val in zip(leaves, [a_val, b_val, c_val]):
            if leaf_idx < input_bits:
                node_vals[leaf_idx] = val

        # Evaluate gates up to root, forcing leaf gate values
        for g_idx, gate in enumerate(circuit.gates):
            full_idx = input_bits + g_idx

            # If this is a leaf gate, force its value
            if full_idx in leaves_set:
                node_vals.append(assignments.get(full_idx, 0))
                if g_idx == root_idx:
                    original_result = node_vals[-1]
                    break
                continue

            if len(gate) == 5:
                _, ga, gb, gc, gimm8 = gate
                va = node_vals[ga] if ga < len(node_vals) else assignments.get(ga, 0)
                vb = node_vals[gb] if gb < len(node_vals) else assignments.get(gb, 0)
                vc = node_vals[gc] if gc < len(node_vals) else assignments.get(gc, 0)
                tidx = (va << 2) | (vb << 1) | vc
                node_vals.append((gimm8 >> tidx) & 1)
            else:
                op = gate[0]
                left = gate[1] if len(gate) > 1 else 0
                right = gate[2] if len(gate) > 2 else 0
                lv = node_vals[left] if left < len(node_vals) else assignments.get(left, 0)
                rv = node_vals[right] if right < len(node_vals) else assignments.get(right, 0)

                if op == "xor":
                    node_vals.append(lv ^ rv)
                elif op == "and":
                    node_vals.append(lv & rv)
                elif op == "or":
                    node_vals.append(lv | rv)
                elif op == "not":
                    node_vals.append(1 - lv)
                elif op == "const":
                    node_vals.append(left & 1)
                else:
                    node_vals.append(0)

            if g_idx == root_idx:
                original_result = node_vals[-1]
                break
        else:
            original_result = 0

        # Check truth table
        expected_from_imm8 = (imm8 >> i) & 1

        if original_result != expected_from_imm8:
            return False

    return True


def apply_ternary_mapping(circuit: CircuitState) -> CircuitState:
    """Apply ternary mapping to all eligible 3-input cones."""
    cones = find_3input_cones(circuit)
    if not cones:
        return circuit

    new_gates = list(circuit.gates)
    verified_count = 0
    failed_count = 0

    for cone in cones:
        root_idx = cone["root"]
        leaves = cone["leaves"]
        imm8 = compute_truth_table(circuit, root_idx, leaves)

        # Verify the truth table is correct
        if not verify_cone_truth_table(circuit, root_idx, leaves, imm8):
            failed_count += 1
            continue  # Skip this cone if verification fails

        verified_count += 1
        # Replace root gate with ternary gate
        new_gates[root_idx] = ("ternary", leaves[0], leaves[1], leaves[2], imm8)

    print(f"   Verified: {verified_count}, Failed: {failed_count}")

    result = CircuitState(
        input_bits=circuit.input_bits,
        output_bits=circuit.output_bits,
        gates=new_gates,
        outputs=list(circuit.outputs),
        gate_count=len(new_gates),
    )

    return result.eliminate_dead_code()


def main():
    print("=" * 70)
    print("Aggressive AVX-512 Ternary Optimization: ANF S-box")
    print("=" * 70)

    # Step 1: Build initial ANF circuit
    print("\n1. Building initial ANF circuit...")
    opt = IncrementalOptimizer(AES_SBOX_TABLE, input_bits=8, output_bits=8)
    circuit = opt.best_state
    initial_gates = circuit.gate_count
    print(f"   Initial: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)")
    print(f"   Depth: {circuit.depth}")

    # Step 2: Apply basic optimizations first
    print("\n2. Applying basic optimizations...")
    circuit = circuit.eliminate_common_subexpressions()
    circuit = circuit.apply_algebraic_rewrites()
    circuit = circuit.eliminate_dead_code()
    circuit = circuit.flatten_xor_trees()
    after_basic = circuit.gate_count
    print(f"   After basic opts: {circuit.gate_count} gates ({circuit.and_count} AND, {circuit.xor_count} XOR)")
    print(f"   Depth: {circuit.depth}")

    # Step 3: Apply ternary mapping iteratively
    print("\n3. Applying ternary mapping (iterative)...")

    iteration = 0
    max_iterations = 10
    while iteration < max_iterations:
        cones = find_3input_cones(circuit)
        if not cones:
            print(f"   Iteration {iteration + 1}: No more cones found")
            break

        circuit = apply_ternary_mapping(circuit)
        circuit = circuit.eliminate_dead_code()

        ternary_count = count_ternary_gates(circuit)
        iteration += 1
        print(f"   Iteration {iteration}: {circuit.gate_count} gates, {ternary_count} ternary, depth {circuit.depth}")

        # Verify after each iteration
        errors = sum(1 for i in range(256) if circuit.evaluate(i) != AES_SBOX_TABLE[i])
        if errors > 0:
            print(f"   ERROR: {errors} verification failures!")
            break

    ternary_count = count_ternary_gates(circuit)
    binary_count = circuit.gate_count - ternary_count
    print(f"\n   Final: {circuit.gate_count} gates ({binary_count} binary, {ternary_count} vpternlogd)")
    print(f"   Final depth: {circuit.depth}")

    # Step 4: Verify correctness
    print("\n4. Verifying correctness...")
    errors = 0
    for i in range(256):
        result = circuit.evaluate(i)
        if result != AES_SBOX_TABLE[i]:
            if errors < 5:
                print(f"   ERROR: S-box[{i}] = 0x{result:02x}, expected 0x{AES_SBOX_TABLE[i]:02x}")
            errors += 1
    if errors == 0:
        print("   All 256 values correct")
    else:
        print(f"   {errors} errors found!")
        return 1

    # Step 5: Schedule for AVX-512
    print("\n5. Scheduling for AVX-512...")
    gates = list(circuit.gates)
    input_bits = circuit.input_bits
    outputs = list(circuit.outputs)

    schedule = list_schedule(gates, input_bits, outputs, AVX512, "slack")
    print(f"   Schedule: {schedule.total_cycles} cycles")

    # Step 6: Register allocation
    print("\n6. Allocating registers...")
    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
    allocation = allocate_registers(live_ranges, schedule, AVX512.registers)
    print(f"   Registers: {AVX512.registers} available, {allocation.num_spills} spills")

    # Step 7: Emit AVX-512 code
    print("\n7. Emitting AVX-512 code...")
    # Note: this emits 256-bit vectors and uses `_mm256_ternarylogic_epi32`,
    # which requires AVX-512VL (in addition to AVX-512F) when compiling.
    emitter = AVX2Emitter()
    code = emitter.emit(schedule, allocation, gates, input_bits, outputs, "sbox_anf_avx512")
    print(f"   Generated {len(code)} bytes of C code")

    # Save the generated code
    output_file = Path("out/sbox_anf_avx512.c")
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(code)
    print(f"   Saved to {output_file}")

    # Also save the circuit state for further analysis
    import json
    circuit_file = Path("out/sbox_anf_optimized.json")
    with open(circuit_file, "w") as f:
        json.dump(circuit.to_dict(), f, indent=2)
    print(f"   Circuit saved to {circuit_file}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Original ANF:        {initial_gates} gates")
    print(f"  After basic opts:    {after_basic} gates")
    print(f"  With ternary:        {circuit.gate_count} gates ({ternary_count} vpternlogd)")
    print(f"  Gate reduction:      {initial_gates - circuit.gate_count} ({(initial_gates - circuit.gate_count) / initial_gates * 100:.1f}%)")
    print(f"  Final depth:         {circuit.depth}")
    print(f"  Schedule cycles:     {schedule.total_cycles}")
    print(f"  Register spills:     {allocation.num_spills}")

    # Compare with target
    print("\n  Benchmark targets:")
    print(f"    < 1000 gates:      {'ACHIEVED' if circuit.gate_count < 1000 else 'NOT MET'}")
    print(f"    BP optimal (~115): {circuit.gate_count} vs 115 (room for more optimization)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
