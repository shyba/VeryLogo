from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BoolType:
    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bool"}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BoolType:
        if data.get("kind") != "bool":
            raise ValueError("invalid bool type")
        return BoolType()


@dataclass(frozen=True)
class BitVecType:
    width: int

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("bitvec width must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bitvec", "width": self.width}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BitVecType:
        if data.get("kind") != "bitvec":
            raise ValueError("invalid bitvec type")
        return BitVecType(width=int(data["width"]))


@dataclass(frozen=True)
class FloatType:
    width: int

    def __post_init__(self) -> None:
        if self.width not in {32, 64}:
            raise ValueError("float width must be 32 or 64")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "float", "width": self.width}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> FloatType:
        if data.get("kind") != "float":
            raise ValueError("invalid float type")
        return FloatType(width=int(data["width"]))


@dataclass(frozen=True)
class SimdType:
    lane_width: int
    lanes: int

    def __post_init__(self) -> None:
        if self.lane_width < 1:
            raise ValueError("simd lane_width must be >= 1")
        if self.lanes < 1:
            raise ValueError("simd lanes must be >= 1")

    @property
    def total_width(self) -> int:
        return self.lane_width * self.lanes

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd", "lane_width": self.lane_width, "lanes": self.lanes}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SimdType:
        if data.get("kind") != "simd":
            raise ValueError("invalid simd type")
        return SimdType(lane_width=int(data["lane_width"]), lanes=int(data["lanes"]))


Type = BoolType | BitVecType | FloatType | SimdType


def type_from_dict(data: dict[str, Any]) -> Type:
    kind = data.get("kind")
    if kind == "bool":
        return BoolType.from_dict(data)
    if kind == "bitvec":
        return BitVecType.from_dict(data)
    if kind == "float":
        return FloatType.from_dict(data)
    if kind == "simd":
        return SimdType.from_dict(data)
    raise ValueError("unknown type kind")


@dataclass(frozen=True)
class Var:
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "var", "name": self.name}


@dataclass(frozen=True)
class BoolConst:
    value: bool

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bool_const", "value": bool(self.value)}


@dataclass(frozen=True)
class BitVecConst:
    width: int
    value: int

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("bitvec const width must be >= 1")
        mask = (1 << self.width) - 1
        object.__setattr__(self, "value", int(self.value) & mask)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bitvec_const", "width": self.width, "value": self.value}


@dataclass(frozen=True)
class FloatConst:
    width: int
    bits: int

    def __post_init__(self) -> None:
        if self.width not in {32, 64}:
            raise ValueError("float const width must be 32 or 64")
        mask = (1 << self.width) - 1
        object.__setattr__(self, "bits", int(self.bits) & mask)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "float_const", "width": self.width, "bits": self.bits}


@dataclass(frozen=True)
class FNeg:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fneg", "x": self.x.to_dict()}


@dataclass(frozen=True)
class FAbs:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fabs", "x": self.x.to_dict()}


@dataclass(frozen=True)
class SimdConst:
    lane_width: int
    lanes: int
    value: int

    def __post_init__(self) -> None:
        if self.lane_width < 1:
            raise ValueError("simd const lane_width must be >= 1")
        if self.lanes < 1:
            raise ValueError("simd const lanes must be >= 1")
        total = self.lane_width * self.lanes
        mask = (1 << total) - 1
        object.__setattr__(self, "value", int(self.value) & mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_const",
            "lane_width": self.lane_width,
            "lanes": self.lanes,
            "value": self.value,
        }


@dataclass(frozen=True)
class FAdd:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fadd", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class FSub:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fsub", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class FMul:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fmul", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class FDiv:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fdiv", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class FSqrt:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fsqrt", "x": self.x.to_dict()}


@dataclass(frozen=True)
class FEq:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "feq", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class FLt:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "flt", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class FLe:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fle", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class FNe:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fne", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFAdd:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fadd", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFSub:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fsub", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFMul:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fmul", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFDiv:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fdiv", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFFma:
    a: Expr
    b: Expr
    c: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_ffma",
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "c": self.c.to_dict(),
        }


@dataclass(frozen=True)
class SimdFSqrt:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fsqrt", "x": self.x.to_dict()}


@dataclass(frozen=True)
class SimdFNeg:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fneg", "x": self.x.to_dict()}


@dataclass(frozen=True)
class SimdFAbs:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fabs", "x": self.x.to_dict()}


@dataclass(frozen=True)
class SimdFCmpEq:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fcmp_eq", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFCmpLt:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fcmp_lt", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFCmpLe:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fcmp_le", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdFCmpNe:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_fcmp_ne", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Not:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "not", "x": self.x.to_dict()}


@dataclass(frozen=True)
class And:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "and", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Or:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "or", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Xor:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "xor", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Add:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "add", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Sub:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "sub", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Shl:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "shl", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class LShr:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "lshr", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class AShr:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "ashr", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdAdd:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_add", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSub:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_sub", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdAddMasked:
    mask: Expr
    passthru: Expr
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_add_masked",
            "mask": self.mask.to_dict(),
            "passthru": self.passthru.to_dict(),
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdSubMasked:
    mask: Expr
    passthru: Expr
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_sub_masked",
            "mask": self.mask.to_dict(),
            "passthru": self.passthru.to_dict(),
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdAddSatU:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_add_satu", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSubSatU:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_sub_satu", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdAddSatS:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_add_sats", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSubSatS:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_sub_sats", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdMulLo:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_mul_lo", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdMulHiU:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_mul_hi_u", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdMulHiS:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_mul_hi_s", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdMaddS16:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_madd_s16", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdUnpackLo:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_unpack_lo",
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdUnpackHi:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_unpack_hi",
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdPackSS16To8:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_pack_ss16_to_8",
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdPackUS16To8:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_pack_us16_to_8",
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdPackSS32To16:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_pack_ss32_to_16",
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdMaskExpand:
    to: Type
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_mask_expand",
            "to": self.to.to_dict(),
            "x": self.x.to_dict(),
        }


@dataclass(frozen=True)
class SimdMaskPack:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_mask_pack", "x": self.x.to_dict()}


@dataclass(frozen=True)
class SimdMinU:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_min_u", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdMaxU:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_max_u", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdMinS:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_min_s", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdMaxS:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_max_s", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdBlend:
    mask: Expr
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_blend",
            "mask": self.mask.to_dict(),
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class SimdZExtLo:
    to: Type
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_zext_lo",
            "to": self.to.to_dict(),
            "x": self.x.to_dict(),
        }


@dataclass(frozen=True)
class SimdSExtLo:
    to: Type
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_sext_lo",
            "to": self.to.to_dict(),
            "x": self.x.to_dict(),
        }


@dataclass(frozen=True)
class SimdEq:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_eq", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdNot:
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_not", "x": self.x.to_dict()}


@dataclass(frozen=True)
class SimdAnd:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_and", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdOr:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_or", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdXor:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_xor", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdShl:
    a: Expr
    sh: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_shl", "a": self.a.to_dict(), "sh": self.sh.to_dict()}


@dataclass(frozen=True)
class SimdLShr:
    a: Expr
    sh: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_lshr", "a": self.a.to_dict(), "sh": self.sh.to_dict()}


@dataclass(frozen=True)
class SimdAShr:
    a: Expr
    sh: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_ashr", "a": self.a.to_dict(), "sh": self.sh.to_dict()}


@dataclass(frozen=True)
class SimdUlt:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_ult", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdUle:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_ule", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdUgt:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_ugt", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdUge:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_uge", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSlt:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_slt", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSle:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_sle", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSgt:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_sgt", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSge:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_sge", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class SimdSplat:
    to: SimdType
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_splat", "to": self.to.to_dict(), "x": self.x.to_dict()}


@dataclass(frozen=True)
class SimdExtractLane:
    x: Expr
    lane: int

    def __post_init__(self) -> None:
        if self.lane < 0:
            raise ValueError("lane must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_extract_lane", "x": self.x.to_dict(), "lane": self.lane}


@dataclass(frozen=True)
class SimdInsertLane:
    x: Expr
    lane: int
    value: Expr

    def __post_init__(self) -> None:
        if self.lane < 0:
            raise ValueError("lane must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "simd_insert_lane",
            "x": self.x.to_dict(),
            "lane": self.lane,
            "value": self.value.to_dict(),
        }


@dataclass(frozen=True)
class SimdShuffle:
    x: Expr
    indices: list[int]

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("indices must be non-empty")
        if any(i < 0 for i in self.indices):
            raise ValueError("indices must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "simd_shuffle", "x": self.x.to_dict(), "indices": self.indices}


@dataclass(frozen=True)
class Eq:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "eq", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Ult:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "ult", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Ule:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "ule", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Ugt:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "ugt", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Uge:
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "uge", "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass(frozen=True)
class Mux:
    cond: Expr
    a: Expr
    b: Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "mux",
            "cond": self.cond.to_dict(),
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


@dataclass(frozen=True)
class Concat:
    parts: list[Expr]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "concat", "parts": [p.to_dict() for p in self.parts]}


@dataclass(frozen=True)
class Slice:
    x: Expr
    offset: int
    width: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("slice offset must be >= 0")
        if self.width < 1:
            raise ValueError("slice width must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "slice",
            "x": self.x.to_dict(),
            "offset": self.offset,
            "width": self.width,
        }


@dataclass(frozen=True)
class Lut8:
    x: Expr
    table: list[int]

    def __post_init__(self) -> None:
        if len(self.table) != 256:
            raise ValueError("lut8 table must have 256 entries")
        for v in self.table:
            if int(v) < 0 or int(v) > 0xFF:
                raise ValueError("lut8 table entry out of range")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "lut8", "x": self.x.to_dict(), "table": list(self.table)}


@dataclass(frozen=True)
class TernaryLut:
    """Ternary lookup table: output[i] = truth_table[a[i]*4 + b[i]*2 + c[i]].

    Implements any 3-input boolean function per bit lane.
    Maps to PTX lop3.b32 and x86 vpternlogd/q.

    The imm8 encodes the truth table:
      bit 0: output when (a,b,c) = (0,0,0)
      bit 1: output when (a,b,c) = (0,0,1)
      bit 2: output when (a,b,c) = (0,1,0)
      ...
      bit 7: output when (a,b,c) = (1,1,1)
    """

    a: Expr
    b: Expr
    c: Expr
    imm8: int

    def __post_init__(self) -> None:
        if not (0 <= self.imm8 <= 255):
            raise ValueError(f"imm8 must be 0-255, got {self.imm8}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ternary_lut",
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "c": self.c.to_dict(),
            "imm8": self.imm8,
        }


@dataclass(frozen=True)
class Bitcast:
    to: Type
    x: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bitcast", "to": self.to.to_dict(), "x": self.x.to_dict()}


@dataclass(frozen=True)
class BitTranspose:
    x: Expr
    lane_width: int
    lanes: int

    def __post_init__(self) -> None:
        if self.lane_width < 1:
            raise ValueError("lane_width must be >= 1")
        if self.lanes < 1:
            raise ValueError("lanes must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "bit_transpose",
            "x": self.x.to_dict(),
            "lane_width": self.lane_width,
            "lanes": self.lanes,
        }


@dataclass(frozen=True)
class Delay:
    x: Expr
    ticks: int

    def __post_init__(self) -> None:
        if self.ticks < 1:
            raise ValueError("delay ticks must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "delay", "x": self.x.to_dict(), "ticks": int(self.ticks)}


Expr = (
    Var
    | BoolConst
    | BitVecConst
    | FloatConst
    | SimdConst
    | Not
    | And
    | Or
    | Xor
    | Add
    | Sub
    | Shl
    | LShr
    | AShr
    | SimdAdd
    | SimdSub
    | SimdAddMasked
    | SimdSubMasked
    | SimdAddSatU
    | SimdSubSatU
    | SimdAddSatS
    | SimdSubSatS
    | SimdMulLo
    | SimdMulHiU
    | SimdMulHiS
    | SimdMaddS16
    | SimdUnpackLo
    | SimdUnpackHi
    | SimdPackSS16To8
    | SimdPackUS16To8
    | SimdPackSS32To16
    | SimdMaskExpand
    | SimdMaskPack
    | SimdMinU
    | SimdMaxU
    | SimdMinS
    | SimdMaxS
    | SimdBlend
    | SimdZExtLo
    | SimdSExtLo
    | SimdEq
    | SimdNot
    | SimdAnd
    | SimdOr
    | SimdXor
    | SimdShl
    | SimdLShr
    | SimdAShr
    | SimdUlt
    | SimdUle
    | SimdUgt
    | SimdUge
    | SimdSlt
    | SimdSle
    | SimdSgt
    | SimdSge
    | SimdSplat
    | SimdExtractLane
    | SimdInsertLane
    | SimdShuffle
    | Eq
    | Ult
    | Ule
    | Ugt
    | Uge
    | Mux
    | Concat
    | Slice
    | Lut8
    | Bitcast
    | BitTranspose
    | Delay
    | FNeg
    | FAbs
    | FAdd
    | FSub
    | FMul
    | FDiv
    | FSqrt
    | FEq
    | FLt
    | FLe
    | FNe
    | SimdFAdd
    | SimdFSub
    | SimdFMul
    | SimdFDiv
    | SimdFFma
    | SimdFSqrt
    | SimdFNeg
    | SimdFAbs
    | SimdFCmpEq
    | SimdFCmpLt
    | SimdFCmpLe
    | SimdFCmpNe
)
EXPR_CLASSES = (
    Var,
    BoolConst,
    BitVecConst,
    FloatConst,
    SimdConst,
    Not,
    And,
    Or,
    Xor,
    Add,
    Sub,
    Shl,
    LShr,
    AShr,
    SimdAdd,
    SimdSub,
    SimdAddMasked,
    SimdSubMasked,
    SimdAddSatU,
    SimdSubSatU,
    SimdAddSatS,
    SimdSubSatS,
    SimdMulLo,
    SimdMulHiU,
    SimdMulHiS,
    SimdMaddS16,
    SimdUnpackLo,
    SimdUnpackHi,
    SimdPackSS16To8,
    SimdPackUS16To8,
    SimdPackSS32To16,
    SimdMaskExpand,
    SimdMaskPack,
    SimdMinU,
    SimdMaxU,
    SimdMinS,
    SimdMaxS,
    SimdBlend,
    SimdZExtLo,
    SimdSExtLo,
    SimdEq,
    SimdNot,
    SimdAnd,
    SimdOr,
    SimdXor,
    SimdShl,
    SimdLShr,
    SimdAShr,
    SimdUlt,
    SimdUle,
    SimdUgt,
    SimdUge,
    SimdSlt,
    SimdSle,
    SimdSgt,
    SimdSge,
    SimdSplat,
    SimdExtractLane,
    SimdInsertLane,
    SimdShuffle,
    Eq,
    Ult,
    Ule,
    Ugt,
    Uge,
    Mux,
    Concat,
    Slice,
    Lut8,
    TernaryLut,
    Bitcast,
    Delay,
    FNeg,
    FAbs,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FSqrt,
    FEq,
    FLt,
    FLe,
    FNe,
    SimdFAdd,
    SimdFSub,
    SimdFMul,
    SimdFDiv,
    SimdFFma,
    SimdFSqrt,
    SimdFNeg,
    SimdFAbs,
    SimdFCmpEq,
    SimdFCmpLt,
    SimdFCmpLe,
    SimdFCmpNe,
)


def expr_from_dict(data: dict[str, Any]) -> Expr:
    kind = data.get("kind")
    if kind == "var":
        return Var(name=str(data["name"]))
    if kind == "bool_const":
        return BoolConst(value=bool(data["value"]))
    if kind == "bitvec_const":
        return BitVecConst(width=int(data["width"]), value=int(data["value"]))
    if kind == "float_const":
        return FloatConst(width=int(data["width"]), bits=int(data["bits"]))
    if kind == "simd_const":
        return SimdConst(
            lane_width=int(data["lane_width"]),
            lanes=int(data["lanes"]),
            value=int(data["value"]),
        )
    if kind == "not":
        return Not(x=expr_from_dict(data["x"]))
    if kind == "and":
        return And(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "or":
        return Or(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "xor":
        return Xor(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "add":
        return Add(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "sub":
        return Sub(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "shl":
        return Shl(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "lshr":
        return LShr(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "ashr":
        return AShr(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "fneg":
        return FNeg(x=expr_from_dict(data["x"]))
    if kind == "fabs":
        return FAbs(x=expr_from_dict(data["x"]))
    if kind == "fadd":
        return FAdd(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "fsub":
        return FSub(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "fmul":
        return FMul(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "fdiv":
        return FDiv(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "fsqrt":
        return FSqrt(x=expr_from_dict(data["x"]))
    if kind == "feq":
        return FEq(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "flt":
        return FLt(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "fle":
        return FLe(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "fne":
        return FNe(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_add":
        return SimdAdd(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_sub":
        return SimdSub(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_add_masked":
        return SimdAddMasked(
            mask=expr_from_dict(data["mask"]),
            passthru=expr_from_dict(data["passthru"]),
            a=expr_from_dict(data["a"]),
            b=expr_from_dict(data["b"]),
        )
    if kind == "simd_sub_masked":
        return SimdSubMasked(
            mask=expr_from_dict(data["mask"]),
            passthru=expr_from_dict(data["passthru"]),
            a=expr_from_dict(data["a"]),
            b=expr_from_dict(data["b"]),
        )
    if kind == "simd_add_satu":
        return SimdAddSatU(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_sub_satu":
        return SimdSubSatU(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_add_sats":
        return SimdAddSatS(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_sub_sats":
        return SimdSubSatS(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_mul_lo":
        return SimdMulLo(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_mul_hi_u":
        return SimdMulHiU(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_mul_hi_s":
        return SimdMulHiS(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_madd_s16":
        return SimdMaddS16(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_unpack_lo":
        return SimdUnpackLo(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_unpack_hi":
        return SimdUnpackHi(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_pack_ss16_to_8":
        return SimdPackSS16To8(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_pack_us16_to_8":
        return SimdPackUS16To8(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_pack_ss32_to_16":
        return SimdPackSS32To16(
            a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"])
        )
    if kind == "simd_mask_expand":
        return SimdMaskExpand(
            to=type_from_dict(data["to"]), x=expr_from_dict(data["x"])
        )
    if kind == "simd_mask_pack":
        return SimdMaskPack(x=expr_from_dict(data["x"]))
    if kind == "simd_min_u":
        return SimdMinU(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_max_u":
        return SimdMaxU(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_min_s":
        return SimdMinS(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_max_s":
        return SimdMaxS(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_blend":
        return SimdBlend(
            mask=expr_from_dict(data["mask"]),
            a=expr_from_dict(data["a"]),
            b=expr_from_dict(data["b"]),
        )
    if kind == "simd_zext_lo":
        return SimdZExtLo(to=type_from_dict(data["to"]), x=expr_from_dict(data["x"]))
    if kind == "simd_sext_lo":
        return SimdSExtLo(to=type_from_dict(data["to"]), x=expr_from_dict(data["x"]))
    if kind == "simd_eq":
        return SimdEq(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_not":
        return SimdNot(x=expr_from_dict(data["x"]))
    if kind == "simd_and":
        return SimdAnd(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_or":
        return SimdOr(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_xor":
        return SimdXor(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_shl":
        return SimdShl(a=expr_from_dict(data["a"]), sh=expr_from_dict(data["sh"]))
    if kind == "simd_lshr":
        return SimdLShr(a=expr_from_dict(data["a"]), sh=expr_from_dict(data["sh"]))
    if kind == "simd_ashr":
        return SimdAShr(a=expr_from_dict(data["a"]), sh=expr_from_dict(data["sh"]))
    if kind == "simd_ult":
        return SimdUlt(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_ule":
        return SimdUle(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_ugt":
        return SimdUgt(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_uge":
        return SimdUge(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_slt":
        return SimdSlt(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_sle":
        return SimdSle(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_sgt":
        return SimdSgt(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_sge":
        return SimdSge(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_fadd":
        return SimdFAdd(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_fmul":
        return SimdFMul(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_fcmp_eq":
        return SimdFCmpEq(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_fcmp_lt":
        return SimdFCmpLt(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_fsub":
        return SimdFSub(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_fdiv":
        return SimdFDiv(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_ffma":
        return SimdFFma(
            a=expr_from_dict(data["a"]),
            b=expr_from_dict(data["b"]),
            c=expr_from_dict(data["c"]),
        )
    if kind == "simd_fsqrt":
        return SimdFSqrt(x=expr_from_dict(data["x"]))
    if kind == "simd_fneg":
        return SimdFNeg(x=expr_from_dict(data["x"]))
    if kind == "simd_fabs":
        return SimdFAbs(x=expr_from_dict(data["x"]))
    if kind == "simd_fcmp_le":
        return SimdFCmpLe(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_fcmp_ne":
        return SimdFCmpNe(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "simd_splat":
        return SimdSplat(to=SimdType.from_dict(data["to"]), x=expr_from_dict(data["x"]))
    if kind == "simd_extract_lane":
        return SimdExtractLane(x=expr_from_dict(data["x"]), lane=int(data["lane"]))
    if kind == "simd_insert_lane":
        return SimdInsertLane(
            x=expr_from_dict(data["x"]),
            lane=int(data["lane"]),
            value=expr_from_dict(data["value"]),
        )
    if kind == "simd_shuffle":
        indices = data.get("indices")
        if not isinstance(indices, list) or not all(
            isinstance(i, int) for i in indices
        ):
            raise ValueError("indices must be a list[int]")
        return SimdShuffle(
            x=expr_from_dict(data["x"]), indices=[int(i) for i in indices]
        )
    if kind == "eq":
        return Eq(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "ult":
        return Ult(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "ule":
        return Ule(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "ugt":
        return Ugt(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "uge":
        return Uge(a=expr_from_dict(data["a"]), b=expr_from_dict(data["b"]))
    if kind == "mux":
        return Mux(
            cond=expr_from_dict(data["cond"]),
            a=expr_from_dict(data["a"]),
            b=expr_from_dict(data["b"]),
        )
    if kind == "concat":
        parts = data.get("parts")
        if not isinstance(parts, list):
            raise ValueError("concat parts must be a list")
        return Concat(parts=[expr_from_dict(p) for p in parts])
    if kind == "slice":
        return Slice(
            x=expr_from_dict(data["x"]),
            offset=int(data["offset"]),
            width=int(data["width"]),
        )
    if kind == "lut8":
        t = data.get("table")
        if not isinstance(t, list):
            raise ValueError("lut8 table must be a list")
        return Lut8(x=expr_from_dict(data["x"]), table=[int(v) for v in t])
    if kind == "ternary_lut":
        return TernaryLut(
            a=expr_from_dict(data["a"]),
            b=expr_from_dict(data["b"]),
            c=expr_from_dict(data["c"]),
            imm8=int(data["imm8"]),
        )
    if kind == "bitcast":
        return Bitcast(to=type_from_dict(data["to"]), x=expr_from_dict(data["x"]))
    if kind == "delay":
        return Delay(x=expr_from_dict(data["x"]), ticks=int(data["ticks"]))
    raise ValueError("unknown expr kind")


@dataclass(frozen=True)
class TickIR:
    name: str
    inputs: dict[str, Type]
    outputs: dict[str, Type]
    state: dict[str, Type]
    reset_state: dict[str, Expr]
    next_state: dict[str, Expr]
    output_exprs: dict[str, Expr]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "state": {k: v.to_dict() for k, v in self.state.items()},
            "reset_state": {k: v.to_dict() for k, v in self.reset_state.items()},
            "next_state": {k: v.to_dict() for k, v in self.next_state.items()},
            "output_exprs": {k: v.to_dict() for k, v in self.output_exprs.items()},
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TickIR:
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("unsupported Tick-IR schema version")
        return TickIR(
            name=str(data["name"]),
            inputs={k: type_from_dict(v) for k, v in data["inputs"].items()},
            outputs={k: type_from_dict(v) for k, v in data["outputs"].items()},
            state={k: type_from_dict(v) for k, v in data["state"].items()},
            reset_state={
                k: expr_from_dict(v) for k, v in data.get("reset_state", {}).items()
            },
            next_state={k: expr_from_dict(v) for k, v in data["next_state"].items()},
            output_exprs={
                k: expr_from_dict(v) for k, v in data["output_exprs"].items()
            },
        )
