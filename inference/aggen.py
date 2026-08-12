"""Agner-driven x86 machine model for STC scheduling and codegen.

Loads the Zen 5 AVX-512 instruction table (inference/results/agner_zen5_avx512.csv,
extracted from Agner Fog's instruction_tables.ods, Zen 5 sheet) and applies
the corrections measured locally (see inference/results/avx512_chains_results.md):

  measured                        Agner
  VDPBF16PS  rt 0.5 (2.0/cyc)     rt 3     (too pessimistic)
  VPTERNLOG  rt ~0.29 (>=3.4/cyc) rt 1     (too pessimistic)
  VPTERNLOG  latency 2            latency 3 (too pessimistic)

The rest of the table validates within rounding (see inference/results/avx512_vs_agner_results.md:
29 of 31 rows within +-15%).

This module is the single source of truth for instruction timing used by:
  - stc/sched/target.py (scheduler TargetModel for the AVX-512 backend),
  - inference/inference/results/bench_gemm_avx512.py (GEMM micro-kernel design: tile selection,
    predicted cycles, peak MACs/cyc oracle).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

AGNER_CSV = Path(__file__).resolve().parent / "results" / "agner_zen5_avx512.csv"

# Locally measured corrections: family -> (rt_cycles, latency, note).
# rt_cycles is reciprocal throughput; 0.5 == 2 instructions/cycle.
MEASURED_CORRECTIONS = {
    "bf16": (0.5, 6, "chain sweep 8/12/16 ch: 1.33/1.85/2.00 -> rt 0.5"),
    "ternary": (0.294, 2, "clean inline-asm 12 ch: 3.41/cyc -> rt ~0.29; latency 2"),
    "vnni8": (0.5, 4, "validates Agner (2.0/cyc, lat 4)"),
    "fma": (0.5, 4, "validates Agner (2.0/cyc, lat 4)"),
    "add/sub": (0.5, 2, "validates Agner (2.0/cyc, lat 2)"),
}


@dataclass(frozen=True)
class InstrSpec:
    """Timing spec for one instruction family."""

    instruction: str  # canonical mnemonic from Agner
    rt: float  # reciprocal throughput, cycles/instruction
    latency: int  # cycles until the result is ready
    pipes: str  # Agner pipe assignment, e.g. "P01"
    macs: int = 0  # multiply-accumulates per 512-bit instruction
    notes: str = ""

    @property
    def per_cycle(self) -> float:
        """Instructions per cycle (== 1/rt)."""
        return 1.0 / self.rt

    @property
    def macs_per_cycle(self) -> float:
        return self.per_cycle * self.macs


class Machine:
    """Zen 5 AVX-512 machine description derived from Agner + measurements."""

    def __init__(self, specs: dict[str, InstrSpec]):
        self.specs = specs
        self.name = "zen5_avx512"
        self.registers = 32  # architectural zmm
        self.loads_per_cycle = 2.0  # Zen 5: 2 load pipes
        self.issue_width = 4

    # -- accessors -------------------------------------------------------
    def spec(self, family: str) -> InstrSpec:
        return self.specs[family]

    def rt(self, family: str) -> float:
        return self.specs[family].rt

    def latency(self, family: str) -> int:
        return self.specs[family].latency

    def max_per_cycle(self, family: str) -> float:
        """Max instructions/cycle of this family on its own pipes."""
        return self.specs[family].per_cycle

    # -- GEMM design helpers ---------------------------------------------
    def gemm_peak_macs_per_cycle(self, family: str) -> float:
        """Theoretical peak MACs/cyc for a GEMM built on this op family."""
        return self.specs[family].macs_per_cycle

    def gemm_tile_cycles_per_chunk(
        self,
        family: str,
        mr: int,
        nr: int,
        k_per_chunk: int,
        loads_per_a_row: int = 1,
    ) -> float:
        """Predicted steady-state cycles for one K-chunk of an MR x NR tile.

        Model (validated in inference/results/bench_gemm_avx512_results.md):
        - the FMA-class op saturates its pipes (P01, 2/cyc),
        - broadcast/loads of A operands run on the load pipes (2/cyc),
        - both happen concurrently (independent pipes), so the chunk time is
          the max of the two, plus a small fixed loop overhead term.
        """
        macs_per_chunk = mr * nr * k_per_chunk
        macs_per_instr = self.specs[family].macs
        fma_ops = macs_per_chunk / macs_per_instr
        fma_cycles = fma_ops / self.max_per_cycle(family)

        a_loads = mr * loads_per_a_row
        b_vecs = nr / 16.0
        load_cycles = (a_loads + b_vecs) / self.loads_per_cycle

        return max(fma_cycles, load_cycles)

    def best_tile(self, family: str) -> tuple[int, int]:
        """Largest MR x NR tile that stc.gemm_asm can actually emit.

        Must agree with inference/gemm_asm.py's fixed-register allocation: at most
        8 row pointers (MR <= 8), accumulators on zmm0-7+zmm16-23 (16 max),
        B vectors at zmm24+, and unroll*2 broadcast temps at zmm28+. Any
        tile whose FMA-ops/cycle does not exceed the op's pipe rate and
        whose loads fit the load pipes predicts the same peak MACs/cyc, so
        the optimal choice is the largest such tile: it amortizes the tile
        prologue/epilogue and loop overhead over the most work. Measured
        best on Zen 5 is 8x32 (see inference/results/bench_gemm_avx512_results.md).
        """
        import inference.gemm_asm as ga

        unroll = 2
        k_per_chunk = ga.K_PER_CHUNK[family]
        best: tuple[int, int] = (0, 0)
        best_area = 0
        for nr in (16, 32, 64):
            for mr in range(4, 9):  # 8 GPR row pointers in the emitted kernel
                nb = nr // 16
                nacc = mr * nb
                if nacc > 16:  # zmm0-7 + zmm16-23
                    continue
                if 24 + unroll * nb + unroll > 32:  # B vecs + broadcast temps
                    continue
                macs_per_chunk = mr * nr * k_per_chunk
                fma_ops = macs_per_chunk / self.specs[family].macs
                fma_cycles = fma_ops / self.max_per_cycle(family)
                loads = mr + nr / 16.0
                load_cycles = loads / self.loads_per_cycle
                if fma_cycles < load_cycles:
                    continue  # load-bound: below peak, not optimal
                area = mr * nr
                if area > best_area:
                    best, best_area = (mr, nr), area
        return best

    def predicted_gemm_cycles(
        self,
        family: str,
        m: int,
        n: int,
        k: int,
        mr: int,
        nr: int,
    ) -> float:
        """Predicted total core cycles for a full M x N x K GEMM.

        Matches the micro-kernel structure in inference/inference/results/bench_gemm_avx512.py:
        the K-loop runs one chunk per k_per_chunk elements (INT8: 4, BF16: 2,
        i.e. one per-lane group per instruction), and every chunk issues
        MR*NR/macs FMA-class ops plus MR A-broadcasts and NR/16 B loads.
        """
        k_per_chunk = {  # k elements per chunk per zmm lane-group
            "vnni8": 4,
            "bf16": 2,
        }.get(family, 4)
        chunks = k // k_per_chunk
        chunk_cyc = self.gemm_tile_cycles_per_chunk(family, mr, nr, k_per_chunk)
        tiles = (m / mr) * (n / nr)
        return tiles * chunks * chunk_cyc

    def __repr__(self) -> str:
        return f"<Machine {self.name}: {len(self.specs)} instruction families>"


# -- table parsing -------------------------------------------------------

_AGNER_MACS = {
    "vnni8": 64,  # 16 lanes x 4 i8 dot -> i32
    "vnni16": 64,  # 16 lanes x 2 x 2 i16 dot -> i32
    "bf16": 32,  # 16 lanes x 2 bf16 MACs
    "fma": 32,  # 16 lanes x 2 fp32 MACs
    "add/sub": 16,
    "bitwise": 16,
    "ternary": 16,
}


def load_agner_specs(csv_path: Path | None = None) -> dict[str, InstrSpec]:
    """Parse the Agner CSV into per-family specs, applying local corrections."""
    path = csv_path or AGNER_CSV
    rows_by_fam: dict[str, list[dict[str, str]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_fam.setdefault(row["family"], []).append(row)

    specs: dict[str, InstrSpec] = {}
    for fam, rows in rows_by_fam.items():
        # pick the register-register row (lowest operands weight)
        def reg_weight(r):
            ops = r["operands"]
            if "m" in ops:
                return 2
            return 1

        pick = min(rows, key=reg_weight)
        rt = float(pick["rt_cycles"])
        latency = int(pick["latency"]) if pick["latency"] else 1
        notes = pick.get("notes", "")
        if fam in MEASURED_CORRECTIONS:
            rt, latency, note = MEASURED_CORRECTIONS[fam]
            notes = (notes + "; " if notes else "") + "measured: " + note
        specs[fam] = InstrSpec(
            instruction=pick["instruction"],
            rt=rt,
            latency=latency,
            pipes=pick["pipes"],
            macs=_AGNER_MACS.get(fam, 0),
            notes=notes,
        )
    return specs


_machine: Machine | None = None


def get_machine() -> Machine:
    """Singleton machine (cached)."""
    global _machine
    if _machine is None:
        _machine = Machine(load_agner_specs())
    return _machine


def main() -> None:
    m = get_machine()
    print(f"{m}")
    print(
        f"{'family':10s} {'instr':26s} {'rt':>5s} {'lat':>4s} {'pipes':>6s} "
        f"{'MACs':>5s} {'MACs/cyc':>9s}"
    )
    for fam, s in m.specs.items():
        print(
            f"{fam:10s} {s.instruction:26s} {s.rt:5.2f} {s.latency:4d} "
            f"{s.pipes:>6s} {s.macs:5d} {s.macs_per_cycle:9.1f}"
        )
    print(
        f"\nGEMM peak MACs/cyc: INT8(vnni8)={m.gemm_peak_macs_per_cycle('vnni8'):.0f}, "
        f"BF16={m.gemm_peak_macs_per_cycle('bf16'):.0f}, "
        f"FP32(fma)={m.gemm_peak_macs_per_cycle('fma'):.0f}"
    )
    print(
        f"best INT8 tile: {m.best_tile('vnni8')}, best BF16 tile: {m.best_tile('bf16')}"
    )


if __name__ == "__main__":
    main()
