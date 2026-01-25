"""Optimization passes."""

from stc.passes.ternary_enumerate import TernaryEnumeratePass
from stc.passes.depth_ternary import (
    DepthAwareTernaryPass,
    DepthBudgetPass,
    map_critical_path_to_ternary,
    map_circuit_critical_path_to_ternary,
)
from stc.passes.balance import BalanceAssociativePass
from stc.passes.depth_resynth import DepthResynthesisPass

__all__ = [
    "TernaryEnumeratePass",
    "DepthAwareTernaryPass",
    "DepthBudgetPass",
    "BalanceAssociativePass",
    "DepthResynthesisPass",
    "map_critical_path_to_ternary",
    "map_circuit_critical_path_to_ternary",
]
