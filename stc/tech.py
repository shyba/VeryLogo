from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from stc.tick_ir import Expr, GemmCall


class CostModel(Protocol):
    """Backend-specific cost model."""

    def expr_cost(self, expr: Expr) -> float:
        """Return cost of expression (lower is better)."""
        ...

    def and_weight(self) -> float:
        """Weight for AND gates (1.0 for software, 100+ for FHE)."""
        ...

    def xor_weight(self) -> float:
        """Weight for XOR gates."""
        ...


class DepthModel(Protocol):
    """Backend-specific depth/latency model."""

    def op_depth(self, op: str) -> int:
        """Depth contribution of an operation."""
        ...

    def is_free(self, op: str) -> bool:
        """Whether op is depth-free (e.g., NOT, wire)."""
        ...


@dataclass(frozen=True)
class Primitive:
    """A target instruction primitive."""

    name: str
    input_count: int
    output_count: int
    latency: int
    throughput: float
    constraints: dict


class Technology(ABC):
    """Abstract technology/backend plugin."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def primitives(self) -> Sequence[Primitive]: ...

    @abstractmethod
    def cost_model(self) -> CostModel: ...

    @abstractmethod
    def depth_model(self) -> DepthModel: ...

    @abstractmethod
    def is_legal(self, expr: Expr) -> bool:
        """Check if expression is legal for this backend."""
        ...

    def lower_expr(self, expr: Expr, *, target: str) -> Expr:
        """Dispatch a target lowering for one expression.

        Technologies historically exposed only ``is_legal``; the explicit
        dispatcher keeps shaped operations from being silently treated as
        ordinary gates and gives each target a named lowering seam.
        """

        return dispatch_lowering(self.name, expr, target=target)


class DefaultCostModel:
    """Software-style cost: all gates equal."""

    def expr_cost(self, expr: Expr) -> float:
        from stc.cost import expr_cost as legacy_cost

        return legacy_cost(expr, {})

    def and_weight(self) -> float:
        return 1.0

    def xor_weight(self) -> float:
        return 1.0


class DefaultDepthModel:
    """Unit depth model: each op adds 1."""

    def op_depth(self, op: str) -> int:
        if op in ("not", "const", "wire", "input"):
            return 0
        return 1

    def is_free(self, op: str) -> bool:
        return op in ("not", "const", "wire", "input")


class GenericTechnology(Technology):
    """Default technology for target-independent optimization."""

    @property
    def name(self) -> str:
        return "generic"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive("and", 2, 1, 1, 1.0, {}),
            Primitive("or", 2, 1, 1, 1.0, {}),
            Primitive("xor", 2, 1, 1, 1.0, {}),
            Primitive("not", 1, 1, 0, 1.0, {}),
        ]

    def cost_model(self) -> CostModel:
        return DefaultCostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        return True


_TECHNOLOGIES: dict[str, Technology] = {}
_LOWERING_DISPATCH: dict[tuple[str, str, type], Callable[[Expr], Expr]] = {}


def register_lowering(
    technology: str, target: str, expr_type: type, lowering: Callable[[Expr], Expr]
) -> None:
    _LOWERING_DISPATCH[(technology, target, expr_type)] = lowering


def dispatch_lowering(technology: str, expr: Expr, *, target: str) -> Expr:
    """Resolve the most-specific registered lowering, preserving unknown nodes."""

    for cls in type(expr).__mro__:
        lowering = _LOWERING_DISPATCH.get((technology, target, cls))
        if lowering is not None:
            return lowering(expr)
    return expr


def register_technology(tech: Technology) -> None:
    _TECHNOLOGIES[tech.name] = tech


def get_technology(name: str) -> Technology:
    if name not in _TECHNOLOGIES:
        raise ValueError(f"Unknown technology: {name}")
    return _TECHNOLOGIES[name]


def list_technologies() -> list[str]:
    return list(_TECHNOLOGIES.keys())


class PTXCostModel:
    """PTX cost model: ternary LUTs are cheap, others standard."""

    def expr_cost(self, expr: Expr) -> float:
        from stc.cost import expr_cost as legacy_cost
        from stc.tick_ir import TernaryLut

        if isinstance(expr, TernaryLut):
            return 1.0

        return legacy_cost(expr, {})

    def and_weight(self) -> float:
        return 1.0

    def xor_weight(self) -> float:
        return 1.0


class PTXTechnology(Technology):
    """NVIDIA PTX backend with lop3 support."""

    @property
    def name(self) -> str:
        return "ptx"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive("and", 2, 1, 1, 1.0, {}),
            Primitive("or", 2, 1, 1, 1.0, {}),
            Primitive("xor", 2, 1, 1, 1.0, {}),
            Primitive("not", 1, 1, 0, 1.0, {}),
            Primitive("lop3", 3, 1, 1, 1.0, {"width": 32}),
        ]

    def cost_model(self) -> CostModel:
        return PTXCostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        return True


class AVX512CostModel:
    """AVX-512 cost model: ternary LUTs and SIMD ops."""

    def expr_cost(self, expr: Expr) -> float:
        from stc.cost import expr_cost as legacy_cost
        from stc.tick_ir import TernaryLut

        if isinstance(expr, TernaryLut):
            return 1.0

        return legacy_cost(expr, {})

    def and_weight(self) -> float:
        return 1.0

    def xor_weight(self) -> float:
        return 0.5


class AVX512Technology(Technology):
    """x86 AVX-512 backend with vpternlog support."""

    @property
    def name(self) -> str:
        return "x86-avx512"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive("and", 2, 1, 1, 1.0, {}),
            Primitive("or", 2, 1, 1, 1.0, {}),
            Primitive("xor", 2, 1, 1, 1.0, {}),
            Primitive("not", 1, 1, 0, 1.0, {}),
            Primitive("vpternlog", 3, 1, 1, 1.0, {"width": [128, 256, 512]}),
        ]

    def cost_model(self) -> CostModel:
        return AVX512CostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        return True


class AVRTechnology(Technology):
    """ATtiny85/AVR backend - basic gates only."""

    @property
    def name(self) -> str:
        return "avr"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive("and", 2, 1, 1, 1.0, {}),
            Primitive("or", 2, 1, 1, 1.0, {}),
            Primitive("xor", 2, 1, 1, 1.0, {}),
            Primitive("not", 1, 1, 0, 1.0, {}),
        ]

    def cost_model(self) -> CostModel:
        return DefaultCostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        return True


class AVX2Technology(Technology):
    """x86 AVX2 backend - no ternary LUTs."""

    @property
    def name(self) -> str:
        return "x86-avx2"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive("and", 2, 1, 1, 1.0, {}),
            Primitive("or", 2, 1, 1, 1.0, {}),
            Primitive("xor", 2, 1, 1, 1.0, {}),
            Primitive("not", 1, 1, 0, 1.0, {}),
        ]

    def cost_model(self) -> CostModel:
        return DefaultCostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        return True


class FutharkTechnology(Technology):
    """Futhark backend - generic bool/bitvec cost model."""

    @property
    def name(self) -> str:
        return "futhark"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive("and", 2, 1, 1, 1.0, {}),
            Primitive("or", 2, 1, 1, 1.0, {}),
            Primitive("xor", 2, 1, 1, 1.0, {}),
            Primitive("not", 1, 1, 0, 1.0, {}),
            Primitive("add", 2, 1, 1, 1.0, {}),
            Primitive("sub", 2, 1, 1, 1.0, {}),
            Primitive("mux", 3, 1, 1, 1.0, {}),
        ]

    def cost_model(self) -> CostModel:
        return DefaultCostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        return True


class X86GemmTechnology(Technology):
    """x86 packed GEMM target with an explicit shaped primitive."""

    @property
    def name(self) -> str:
        return "x86-gemm"

    def primitives(self) -> Sequence[Primitive]:
        return [
            Primitive(
                "gemm_u8s8_i32",
                2,
                1,
                4,
                1.0,
                {"instruction": "vpdpbusd", "layout": "row_major"},
            )
        ]

    def cost_model(self) -> CostModel:
        return DefaultCostModel()

    def depth_model(self) -> DepthModel:
        return DefaultDepthModel()

    def is_legal(self, expr: Expr) -> bool:
        from stc.tick_ir import GemmCall

        return isinstance(expr, GemmCall)


register_technology(GenericTechnology())
register_technology(PTXTechnology())
register_technology(AVX512Technology())
register_technology(AVRTechnology())
register_technology(AVX2Technology())
register_technology(FutharkTechnology())
register_technology(X86GemmTechnology())


# The generic lowering is deliberately bounded; production-shaped calls stay
# intact for the x86 target and are consumed by backend_x86_gemm.py.
def _lower_gemm_generic(expr: Expr) -> Expr:
    from stc.gemm_lowering import expand_gemm_call

    if not isinstance(expr, GemmCall):
        return expr
    return expand_gemm_call(expr)


def _identity_lowering(expr: Expr) -> Expr:
    return expr


register_lowering("generic", "scalar", GemmCall, _lower_gemm_generic)
register_lowering("x86-gemm", "x86-vnni", GemmCall, _identity_lowering)
