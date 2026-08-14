"""
Target models describing hardware constraints for scheduling.

Each target specifies:
- Available registers
- Instruction latencies (cycles until result ready)
- Throughput limits (max instructions per cycle per operation type)
- Issue width (total instructions per cycle)
"""

from dataclasses import dataclass, field

from stc.sched.cpu_model import CpuModel


@dataclass(frozen=True)
class TargetModel:
    """Legacy scheduler projection plus an optional rich CPU description.

    ``latencies`` and integer ``throughput`` remain for compatibility with
    the existing gate schedulers. ``cpu`` carries operand forms, eligible
    resources, and register-file facts for the resource-aware floor planner;
    the current schedulers do not consume it yet.
    """

    name: str
    registers: int
    issue_width: int
    latencies: dict[str, int] = field(default_factory=dict)
    throughput: dict[str, int] = field(default_factory=dict)
    cpu: CpuModel | None = None

    def latency(self, op: str) -> int:
        return self.latencies.get(op, 1)

    def max_per_cycle(self, op: str) -> int:
        return self.throughput.get(op, self.issue_width)

    def __str__(self) -> str:
        return f"{self.name}(regs={self.registers}, issue={self.issue_width})"


SSE2 = TargetModel(
    name="sse2_32bit",
    registers=8,
    issue_width=3,
    latencies={"xor": 1, "and": 1, "or": 1, "not": 1, "andn": 1},
    throughput={"xor": 2, "and": 1, "or": 1, "not": 1, "andn": 1},
)

SSE2_X64 = TargetModel(
    name="sse2_64bit",
    registers=16,
    issue_width=4,
    latencies={"xor": 1, "and": 1, "or": 1, "not": 1, "andn": 1},
    throughput={"xor": 3, "and": 2, "or": 2, "not": 2, "andn": 1},
)

AVX2 = TargetModel(
    name="avx2",
    registers=16,
    issue_width=4,
    latencies={"xor": 1, "and": 1, "or": 1, "not": 1, "andn": 1, "ternary": 1},
    throughput={"xor": 3, "and": 2, "or": 2, "not": 2, "andn": 2, "ternary": 2},
)


def _avx512_from_agner() -> TargetModel:
    """AVX-512 scheduler target with locally measured Zen 5 corrections.

    The ternary values come from the Zen 5 instruction-table measurements
    (see inference/aggen.py / inference/results/avx512_chains_results.md):
    VPTERNLOG measures latency 2 and ~3.4 ops/cyc on this SKU (Agner lists
    latency 3 / 1 per cyc). Inlined here so stc/ has no dependency on the
    inference module; the machine model lives in inference/aggen.py.
    """
    return TargetModel(
        name="avx512",
        registers=32,
        issue_width=4,
        latencies={"xor": 1, "and": 1, "or": 1, "not": 1, "andn": 1, "ternary": 2},
        throughput={"xor": 2, "and": 2, "or": 2, "not": 2, "andn": 2, "ternary": 3},
    )


def _zen5_avx512() -> TargetModel:
    """Zen 5 AVX-512 scheduler target (full Agner-derived, inline).

    Mirrors inference/aggen.py's machine model for this CPU (measured
    ternary latency 2 / ~3 ops/cyc, bitwise ops at the P0123 3/cyc rate),
    inlined so stc/ stays independent of the inference module.
    """
    return TargetModel(
        name="zen5_avx512",
        registers=32,
        issue_width=4,
        latencies={"xor": 1, "and": 1, "or": 1, "not": 1, "andn": 1, "ternary": 2},
        throughput={"xor": 3, "and": 3, "or": 3, "not": 3, "andn": 3, "ternary": 3},
    )


AVX512 = _avx512_from_agner()


ZEN5_AVX512 = _zen5_avx512()

PTX = TargetModel(
    name="ptx",
    registers=255,
    issue_width=32,
    latencies={"xor": 4, "and": 4, "or": 4, "not": 4, "lop3": 4},
    throughput={"xor": 32, "and": 32, "or": 32, "not": 32, "lop3": 32},
)

TARGETS = {
    "sse2": SSE2,
    "sse2_x64": SSE2_X64,
    "avx2": AVX2,
    "avx512": AVX512,
    "zen5_avx512": ZEN5_AVX512,
    "ptx": PTX,
}


def get_target(name: str) -> TargetModel:
    if name not in TARGETS:
        raise ValueError(f"Unknown target: {name}. Available: {list(TARGETS.keys())}")
    return TARGETS[name]
