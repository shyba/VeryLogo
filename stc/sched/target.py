"""
Target models describing hardware constraints for scheduling.

Each target specifies:
- Available registers
- Instruction latencies (cycles until result ready)
- Throughput limits (max instructions per cycle per operation type)
- Issue width (total instructions per cycle)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetModel:
    """Hardware constraints for instruction scheduling."""

    name: str
    registers: int
    issue_width: int
    latencies: dict[str, int] = field(default_factory=dict)
    throughput: dict[str, int] = field(default_factory=dict)

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

AVX512 = TargetModel(
    name="avx512",
    registers=32,
    issue_width=4,
    latencies={"xor": 1, "and": 1, "or": 1, "not": 1, "andn": 1, "ternary": 1},
    throughput={"xor": 2, "and": 2, "or": 2, "not": 2, "andn": 2, "ternary": 2},
)

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
    "ptx": PTX,
}


def get_target(name: str) -> TargetModel:
    if name not in TARGETS:
        raise ValueError(f"Unknown target: {name}. Available: {list(TARGETS.keys())}")
    return TARGETS[name]
