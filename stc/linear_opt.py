from __future__ import annotations

from dataclasses import dataclass

from stc.circuit_synth import CircuitState


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


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
    base = [1 << i for i in range(n_inputs)]
    base_set = set(base)
    targets = set(matrix)
    remaining = targets - base_set
    operations: list[tuple[int, int]] = []

    while remaining:
        best_pair: tuple[int, int] | None = None
        best_score = float("-inf")
        best_candidate = 0

        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                candidate = base[i] ^ base[j]
                if candidate in base_set:
                    continue

                score = 0.0
                for t in remaining:
                    old_dist = min(hamming_distance(t, b) for b in base)
                    new_dist = min(old_dist, hamming_distance(t, candidate))
                    score += old_dist - new_dist

                if candidate in remaining:
                    score += len(remaining) * 10

                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
                    best_candidate = candidate

        assert best_pair is not None
        operations.append(best_pair)
        base.append(best_candidate)
        base_set.add(best_candidate)
        remaining.discard(best_candidate)

    return operations


@dataclass
class LinearCone:
    """Represents a linear (XOR-only) subcircuit as a GF(2) matrix.

    If the cone has n inputs and m outputs, this is an m x n binary matrix
    where entry [i,j] = 1 means output i depends on input j (XOR chain).

    The matrix is stored as a list of m integers, where each integer is an
    n-bit mask representing one row of the matrix.
    """

    inputs: list[int]
    outputs: list[int]
    matrix: list[int]

    @classmethod
    def from_circuit(
        cls, state: CircuitState, output_indices: list[int], stop_at: set[int]
    ) -> "LinearCone":
        """Extract linear cone by backward traversal until hitting stop_at signals.

        Traverses backward from the given output signals, collecting XOR
        dependencies. Stops at:
        - Primary inputs (indices < state.input_bits)
        - AND gates
        - Signals in the stop_at set

        Args:
            state: The circuit state to extract from
            output_indices: List of signal indices to use as outputs of the cone
            stop_at: Set of signal indices to treat as cone inputs (stop points)

        Returns:
            LinearCone representing the XOR-only subcircuit
        """
        cone_inputs_set: set[int] = set()
        output_dependencies: list[set[int]] = []

        for out_idx in output_indices:
            deps = cls._collect_linear_deps(state, out_idx, stop_at, cone_inputs_set)
            output_dependencies.append(deps)

        actual_inputs: set[int] = set()
        for deps in output_dependencies:
            actual_inputs.update(deps)

        inputs = sorted(actual_inputs)
        input_to_bit: dict[int, int] = {inp: i for i, inp in enumerate(inputs)}

        matrix: list[int] = []
        for deps in output_dependencies:
            row = 0
            for dep in deps:
                if dep in input_to_bit:
                    row |= 1 << input_to_bit[dep]
            matrix.append(row)

        return cls(inputs=inputs, outputs=output_indices, matrix=matrix)

    @classmethod
    def _collect_linear_deps(
        cls,
        state: CircuitState,
        signal_idx: int,
        stop_at: set[int],
        cone_inputs: set[int],
    ) -> set[int]:
        """Recursively collect linear (XOR) dependencies for a signal.

        Returns a set of signal indices that are XORed together to produce
        the given signal.
        """
        if signal_idx in stop_at:
            cone_inputs.add(signal_idx)
            return {signal_idx}

        if signal_idx < state.input_bits:
            cone_inputs.add(signal_idx)
            return {signal_idx}

        gate_idx = signal_idx - state.input_bits
        if gate_idx < 0 or gate_idx >= len(state.gates):
            cone_inputs.add(signal_idx)
            return {signal_idx}

        op, left, right = state.gates[gate_idx]

        if op == "xor":
            left_deps = cls._collect_linear_deps(state, left, stop_at, cone_inputs)
            right_deps = cls._collect_linear_deps(state, right, stop_at, cone_inputs)
            return left_deps.symmetric_difference(right_deps)

        if op == "not":
            return cls._collect_linear_deps(state, left, stop_at, cone_inputs)

        cone_inputs.add(signal_idx)
        return {signal_idx}

    def to_xor_circuit(self) -> list[tuple[str, int, int]]:
        """Convert matrix back to XOR gates (naive version).

        Returns a list of (op, left, right) tuples representing gates.
        The gates use indices where:
        - 0 to len(inputs)-1 are the cone inputs
        - len(inputs) onwards are the generated XOR gates

        This is a naive implementation that simply chains XORs for each
        output row. More sophisticated implementations could share
        intermediate results.
        """
        gates: list[tuple[str, int, int]] = []
        num_inputs = len(self.inputs)

        for row in self.matrix:
            if row == 0:
                continue

            bits = []
            for bit_idx in range(num_inputs):
                if row & (1 << bit_idx):
                    bits.append(bit_idx)

            if len(bits) <= 1:
                continue

            current = bits[0]
            for next_bit in bits[1:]:
                gate_idx = num_inputs + len(gates)
                gates.append(("xor", current, next_bit))
                current = gate_idx

        return gates

    def to_xor_circuit_optimized(
        self, max_inputs_for_bp: int = 64
    ) -> list[tuple[str, int, int]]:
        """Convert matrix to XOR gates using BP algorithm for minimum gates.

        Uses Boyar-Peralta greedy algorithm to minimize the number of XOR
        gates needed to compute all outputs from the inputs.

        For large input counts (> max_inputs_for_bp), falls back to the
        naive chained implementation to avoid excessive computation time.

        Args:
            max_inputs_for_bp: Maximum number of inputs to use BP algorithm.
                Falls back to naive version for larger cones.

        Returns a list of (op, left, right) tuples representing gates.
        The gates use indices where:
        - 0 to len(inputs)-1 are the cone inputs
        - len(inputs) onwards are the generated XOR gates
        """
        num_inputs = len(self.inputs)

        if not self.matrix or all(row == 0 for row in self.matrix):
            return []

        non_trivial_targets = [row for row in self.matrix if bin(row).count("1") > 1]
        if not non_trivial_targets:
            return []

        if num_inputs > max_inputs_for_bp:
            return self.to_xor_circuit()

        ops = boyar_peralta_minimize(non_trivial_targets, num_inputs)

        gates: list[tuple[str, int, int]] = []
        for idx_a, idx_b in ops:
            gates.append(("xor", idx_a, idx_b))

        return gates

    def get_output_signals_optimized(
        self, max_inputs_for_bp: int = 64
    ) -> list[int]:
        """Get the signal indices for outputs after to_xor_circuit_optimized.

        Returns indices into the combined input+gates space based on the
        BP-optimized circuit.

        Args:
            max_inputs_for_bp: Maximum number of inputs to use BP algorithm.
                Must match the value used in to_xor_circuit_optimized.
        """
        num_inputs = len(self.inputs)

        if num_inputs > max_inputs_for_bp:
            return self.get_output_signals()

        base = [1 << i for i in range(num_inputs)]
        base_set = set(base)
        non_trivial_targets = [row for row in self.matrix if bin(row).count("1") > 1]
        if not non_trivial_targets:
            pass
        else:
            ops = boyar_peralta_minimize(non_trivial_targets, num_inputs)
            for idx_a, idx_b in ops:
                candidate = base[idx_a] ^ base[idx_b]
                base.append(candidate)
                base_set.add(candidate)

        row_to_signal: dict[int, int] = {}
        for i in range(num_inputs):
            row_to_signal[1 << i] = i

        for i, val in enumerate(base[num_inputs:]):
            row_to_signal[val] = num_inputs + i

        output_signals: list[int] = []
        for row in self.matrix:
            if row == 0:
                output_signals.append(-1)
            elif row in row_to_signal:
                output_signals.append(row_to_signal[row])
            else:
                output_signals.append(-1)

        return output_signals

    def get_output_signals(self) -> list[int]:
        """Get the signal indices for outputs after to_xor_circuit.

        Returns indices into the combined input+gates space.
        """
        num_inputs = len(self.inputs)
        output_signals: list[int] = []
        gate_count = 0

        for row in self.matrix:
            if row == 0:
                output_signals.append(-1)
                continue

            bits = []
            for bit_idx in range(num_inputs):
                if row & (1 << bit_idx):
                    bits.append(bit_idx)

            if len(bits) == 1:
                output_signals.append(bits[0])
                continue

            gate_count += len(bits) - 1
            output_signals.append(num_inputs + gate_count - 1)

        return output_signals


def partition_linear_nonlinear(
    state: CircuitState,
) -> tuple[list[LinearCone], list[int]]:
    """Separate circuit into linear cones (XOR-only) separated by AND gates.

    Identifies all AND gates in the circuit as boundaries, then extracts
    linear (XOR-only) cones that feed into each AND gate and the outputs.

    Returns:
        linear_cones: List of LinearCone objects (GF(2) matrices)
        and_gates: List of AND gate indices (signal indices, not gate indices)
    """
    and_gate_indices: list[int] = []
    for g_idx, (op, _, _) in enumerate(state.gates):
        if op == "and":
            and_gate_indices.append(state.input_bits + g_idx)

    stop_at = set(and_gate_indices)

    linear_cones: list[LinearCone] = []

    for and_idx in and_gate_indices:
        gate_idx = and_idx - state.input_bits
        _, left, right = state.gates[gate_idx]

        for operand in [left, right]:
            if operand in stop_at or operand < state.input_bits:
                continue
            cone = LinearCone.from_circuit(state, [operand], stop_at)
            if cone.matrix[0] != 0:
                linear_cones.append(cone)

    output_indices = [idx for idx, _ in state.outputs]
    for out_idx in output_indices:
        if out_idx in stop_at or out_idx < state.input_bits:
            continue
        cone = LinearCone.from_circuit(state, [out_idx], stop_at)
        if cone.matrix[0] != 0:
            linear_cones.append(cone)

    return linear_cones, and_gate_indices


def evaluate_linear_cone(cone: LinearCone, input_values: dict[int, int]) -> list[int]:
    """Evaluate a linear cone given input values.

    Args:
        cone: The LinearCone to evaluate
        input_values: Dictionary mapping input signal index to value (0 or 1)

    Returns:
        List of output values (0 or 1 for each output)
    """
    outputs = []
    for row in cone.matrix:
        val = 0
        for bit_idx, inp in enumerate(cone.inputs):
            if row & (1 << bit_idx):
                val ^= input_values.get(inp, 0)
        outputs.append(val)
    return outputs
