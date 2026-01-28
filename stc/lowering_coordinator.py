from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stc.circuit_synth import CircuitState
from stc.metrics import compute_metrics
from stc.packed_circuit import PackedCircuitState
from stc.tick_ir import TickIR
from stc.tick_ir_to_circuit_state import PackedLayout, lower_tick_ir_to_circuit_state
from stc.tick_ir_to_packed_circuit_state import (
    PackedLoweringError,
    PackedWordLayout,
    lower_tick_ir_to_packed_circuit_state,
)


@dataclass(frozen=True)
class LoweringChoice:
    path: str
    reason: str
    unsupported: list[str]
    stats: dict[str, int]
    gate_comparison: dict[str, int | float] | None = None

    def to_dict(self) -> dict:
        d = {
            "path": self.path,
            "reason": self.reason,
            "unsupported": self.unsupported,
            "stats": self.stats,
        }
        if self.gate_comparison is not None:
            d["gate_comparison"] = self.gate_comparison
        return d


def coordinate_lowering(
    ir: TickIR,
    *,
    prefer_packed: bool = False,
    force_choice: bool = False,
) -> tuple[
    CircuitState | PackedCircuitState,
    PackedLayout | PackedWordLayout,
    LoweringChoice,
]:
    """
    Coordinate lowering with automatic fallback and choice reporting.

    If prefer_packed is True, tries packed lowering first. On PackedLoweringError,
    falls back to bit-level lowering (unless force_choice=True) and generates a
    detailed report explaining the choice.

    If prefer_packed is False (default), uses bit-level lowering. This is the
    safest choice for backends that assume bit-level CircuitState semantics.

    If force_choice is True, does not fall back on error - raises the exception instead
    (e.g., AVX2/PTX bitsliced emitters).

    Returns:
        - circuit: Either PackedCircuitState (word-level) or CircuitState (bit-level)
        - layout: Either PackedWordLayout or PackedLayout
        - choice: LoweringChoice report with path, reason, unsupported expressions, and stats
    """
    metrics = compute_metrics(ir)
    stats = {
        "input_bits": sum(_width_bits(t) for t in ir.inputs.values() if t is not None),
        "state_bits": metrics.state_bits,
        "output_bits": sum(
            _width_bits(t) for t in ir.outputs.values() if t is not None
        ),
        "gates": metrics.ops_total,
        "expr_nodes": metrics.expr_nodes_total,
        "expr_depth": metrics.expr_depth_max,
    }

    if prefer_packed:
        try:
            packed, layout = lower_tick_ir_to_packed_circuit_state(ir)
            circuit, bit_layout = lower_tick_ir_to_circuit_state(ir)
            packed_gates = len(packed.gates)
            bit_gates = len(circuit.gates)
            ratio = packed_gates / bit_gates if bit_gates > 0 else 0.0

            gate_comparison = {
                "packed_gates": packed_gates,
                "bitsliced_gates": bit_gates,
                "ratio": round(ratio, 2),
            }

            if ratio > 5.0 and not force_choice:
                choice = LoweringChoice(
                    path="bit",
                    reason=f"Packed lowering inefficient (ratio={ratio:.1f}x), automatic fallback to bit-level",
                    unsupported=[],
                    stats=stats,
                    gate_comparison=gate_comparison,
                )
                return (circuit, bit_layout, choice)

            choice = LoweringChoice(
                path="packed",
                reason="Successfully lowered to packed 64-bit word-level representation",
                unsupported=[],
                stats=stats,
                gate_comparison=gate_comparison,
            )
            return (packed, layout, choice)
        except PackedLoweringError as e:
            if force_choice:
                raise
            unsupported_types = []
            if e.expression_type:
                unsupported_types.append(e.expression_type)

            reason_parts = [
                "Packed lowering failed, using bit-level lowering as fallback.",
                f"Error: {e.message}",
            ]
            if e.variable_name:
                reason_parts.append(f"Failed at variable: {e.variable_name}")
            if e.subexpression_context:
                reason_parts.append(f"Context: {e.subexpression_context}")

            circuit, layout = lower_tick_ir_to_circuit_state(ir)
            choice = LoweringChoice(
                path="bit",
                reason=" ".join(reason_parts),
                unsupported=unsupported_types,
                stats=stats,
                gate_comparison=None,
            )
            return (circuit, layout, choice)

    circuit, layout = lower_tick_ir_to_circuit_state(ir)
    choice = LoweringChoice(
        path="bit",
        reason="Bit-level lowering selected (prefer_packed disabled).",
        unsupported=[],
        stats=stats,
        gate_comparison=None,
    )
    return (circuit, layout, choice)


def _width_bits(t) -> int:
    """Compute width in bits for a Type."""
    from stc.tick_ir import BitVecType, BoolType, SimdType

    if isinstance(t, BoolType):
        return 1
    if isinstance(t, BitVecType):
        return int(t.width)
    if isinstance(t, SimdType):
        return int(t.lane_width) * int(t.lanes)
    return 0


def write_lowering_choice_report(choice: LoweringChoice, out_dir: Path) -> None:
    """Write lowering choice report to out_dir/lowering_choice.json."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "lowering_choice.json"
    report_path.write_text(
        json.dumps(choice.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
