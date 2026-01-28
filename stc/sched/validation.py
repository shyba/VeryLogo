"""Validation context for code emission.

This module provides validation of node indices and register assignments
before code generation to catch bugs early instead of at C compile time.
"""

from __future__ import annotations

from dataclasses import dataclass

from stc.sched.regalloc import RegAllocation


@dataclass
class EmitContext:
    """Context for validating code emission.

    Tracks input bits, gate count, and register allocation to validate
    that all node references are valid before emitting code.
    """

    input_bits: int
    num_gates: int
    num_physical_regs: int
    allocation: RegAllocation

    @property
    def total_nodes(self) -> int:
        """Total number of valid nodes (inputs + gates)."""
        return self.input_bits + self.num_gates

    def validate_node(self, node: int) -> None:
        """Validate that a node index is valid.

        Args:
            node: Node index to validate

        Raises:
            ValueError: If node is invalid (negative, exceeds count, or not allocated)
        """
        if node < 0:
            raise ValueError(f"Invalid negative node: {node}")

        if node >= self.total_nodes:
            raise ValueError(
                f"Node {node} exceeds total nodes {self.total_nodes} "
                f"(input_bits={self.input_bits}, num_gates={self.num_gates})"
            )

        if node >= self.input_bits:
            if node not in self.allocation.reg_assignment:
                if node not in set(self.allocation.spills):
                    raise ValueError(
                        f"Gate node {node} not allocated and not spilled "
                        f"(available regs: {sorted(self.allocation.reg_assignment.keys())})"
                    )

    def get_register(self, node: int) -> int:
        """Get the physical register for a node.

        Args:
            node: Node index (input or gate)

        Returns:
            Physical register ID

        Raises:
            ValueError: If node is invalid or not allocated to a register
        """
        self.validate_node(node)

        if node < self.input_bits:
            return node

        if node in self.allocation.reg_assignment:
            return self.allocation.reg_assignment[node]

        raise ValueError(
            f"Node {node} not allocated to register "
            f"(spilled nodes: {self.allocation.spills})"
        )

    def is_spilled(self, node: int) -> bool:
        """Check if a node is spilled to memory.

        Args:
            node: Node index to check

        Returns:
            True if node is spilled, False otherwise
        """
        return node in set(self.allocation.spills)
