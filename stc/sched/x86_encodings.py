"""Concrete x86-64 encodings used by the floor-planner v2 path."""

from __future__ import annotations

from stc.sched.cpu_model import (
    CpuModelError,
    InstructionEncoding,
    InstructionForm,
    MemoryModel,
)


_AVX512F = frozenset({"avx512f"})


_ENCODINGS = {
    "vpandq": InstructionEncoding(
        name="vpandq",
        mnemonic="vpandq",
        family="bitwise",
        source_count=2,
        operands="v,v,v",
        required_features=_AVX512F,
        target="x86_64_att",
        source="x86-avx512-manifest",
    ),
    "vpandnq": InstructionEncoding(
        name="vpandnq",
        mnemonic="vpandnq",
        family="bitwise",
        source_count=2,
        operands="v,v,v",
        required_features=_AVX512F,
        target="x86_64_att",
        source="x86-avx512-manifest",
    ),
    "vporq": InstructionEncoding(
        name="vporq",
        mnemonic="vporq",
        family="bitwise",
        source_count=2,
        operands="v,v,v",
        required_features=_AVX512F,
        target="x86_64_att",
        source="x86-avx512-manifest",
    ),
    "vpxorq": InstructionEncoding(
        name="vpxorq",
        mnemonic="vpxorq",
        family="bitwise",
        source_count=2,
        operands="v,v,v",
        required_features=_AVX512F,
        target="x86_64_att",
        source="x86-avx512-manifest",
    ),
    "vpternlogq": InstructionEncoding(
        name="vpternlogq",
        mnemonic="vpternlogq",
        family="ternary",
        source_count=3,
        operands="v,v,v",
        required_features=_AVX512F,
        tied_input=0,
        requires_immediate=True,
        target="x86_64_att",
        source="x86-avx512-manifest",
    ),
}

_MEMORY_ENCODINGS = {
    "load": InstructionEncoding(
        name="vmovdqu64.load",
        mnemonic="vmovdqu64",
        family="load",
        source_count=0,
        operands="v,m",
        required_features=_AVX512F,
        target="x86_64_att",
        source="x86-avx512-memory-manifest",
    ),
    "store": InstructionEncoding(
        name="vmovdqu64.store",
        mnemonic="vmovdqu64",
        family="store",
        source_count=1,
        operands="m,v",
        required_features=_AVX512F,
        target="x86_64_att",
        source="x86-avx512-memory-manifest",
    ),
}


def x86_encoding(mnemonic: str) -> InstructionEncoding:
    """Return a known AVX-512 vector encoding by mnemonic."""

    try:
        return _ENCODINGS[mnemonic.casefold()]
    except KeyError as exc:
        raise CpuModelError(f"unsupported x86 floor encoding {mnemonic!r}") from exc


def x86_encoding_specs() -> tuple[InstructionEncoding, ...]:
    """Return the immutable target encoding manifest."""

    return tuple(_ENCODINGS.values())


def x86_memory_encoding(kind: str) -> InstructionEncoding:
    """Return the concrete AVX-512 vector-memory encoding for ``kind``."""

    try:
        return _MEMORY_ENCODINGS[kind.casefold()]
    except KeyError as exc:
        raise CpuModelError(f"unsupported x86 memory encoding {kind!r}") from exc


def avx512_memory_model(
    *,
    load_pipes: tuple[str, ...] = ("P2", "P3"),
    store_pipes: tuple[str, ...] = ("P2", "P3"),
    load_latency: float = 4.0,
    store_latency: float = 1.0,
    reciprocal_throughput: float = 0.5,
) -> MemoryModel:
    """Build an explicit AVX-512 vector-memory manifest.

    The Agner instruction CSV does not provide the machine-wide ABI traffic
    policy.  Callers therefore choose the load/store resources and timings in
    the target manifest instead of inheriting an unverified default.
    """

    return MemoryModel(
        load=InstructionForm(
            mnemonic="VMOVDQU64 load",
            operands="v,m",
            uops=1,
            latency=load_latency,
            reciprocal_throughput=reciprocal_throughput,
            pipes=load_pipes,
            family="load",
            features=_AVX512F,
            source="x86-avx512-memory-manifest",
        ),
        store=InstructionForm(
            mnemonic="VMOVDQU64 store",
            operands="m,v",
            uops=1,
            latency=store_latency,
            reciprocal_throughput=reciprocal_throughput,
            pipes=store_pipes,
            family="store",
            features=_AVX512F,
            source="x86-avx512-memory-manifest",
        ),
        load_encoding=x86_memory_encoding("load"),
        store_encoding=x86_memory_encoding("store"),
    )
