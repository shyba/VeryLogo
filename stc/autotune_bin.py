from __future__ import annotations

from pathlib import Path

from stc.autotune import (
    AutotuneCandidate,
    AutotuneChoice,
    AutotuneConfig,
    AutotuneResults,
    AutotuneScores,
)


MAGIC_RESULTS = b"ATB1"
MAGIC_CHOICE = b"ATC1"
VERSION = 1


class _BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write_u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def write_u32(self, v: int) -> None:
        self.buf.extend(int(v).to_bytes(4, "little", signed=False))

    def write_f64(self, v: float) -> None:
        import struct

        self.buf.extend(struct.pack("<d", float(v)))

    def write_str(self, s: str) -> None:
        data = s.encode("utf-8")
        self.write_u32(len(data))
        self.buf.extend(data)


class _BinReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.off = 0

    def read_u8(self) -> int:
        v = self.data[self.off]
        self.off += 1
        return v

    def read_u32(self) -> int:
        v = int.from_bytes(self.data[self.off : self.off + 4], "little", signed=False)
        self.off += 4
        return v

    def read_f64(self) -> float:
        import struct

        v = struct.unpack("<d", self.data[self.off : self.off + 8])[0]
        self.off += 8
        return float(v)

    def read_str(self) -> str:
        n = self.read_u32()
        data = self.data[self.off : self.off + n]
        self.off += n
        return data.decode("utf-8")


def _write_config(w: _BinWriter, cfg: AutotuneConfig) -> None:
    w.write_u32(cfg.region_max_gates)
    w.write_u32(cfg.region_max_boundary)
    w.write_str(cfg.scheduler)


def _read_config(r: _BinReader) -> AutotuneConfig:
    return AutotuneConfig(
        region_max_gates=int(r.read_u32()),
        region_max_boundary=int(r.read_u32()),
        scheduler=r.read_str(),
    )


def _write_scores(w: _BinWriter, scores: AutotuneScores) -> None:
    w.write_u32(scores.max_live_estimate)
    w.write_u32(scores.boundary_total)
    w.write_u32(scores.gate_count)
    w.write_u32(scores.critical_path)
    w.write_f64(scores.compile_time_ms)


def _read_scores(r: _BinReader) -> AutotuneScores:
    return AutotuneScores(
        max_live_estimate=int(r.read_u32()),
        boundary_total=int(r.read_u32()),
        gate_count=int(r.read_u32()),
        critical_path=int(r.read_u32()),
        compile_time_ms=float(r.read_f64()),
    )


def write_autotune_results_bin(results: AutotuneResults, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC_RESULTS)
    w.write_u8(VERSION)
    w.write_u32(results.seed)
    w.write_u32(results.budget_ms)
    w.write_u32(len(results.candidates))
    for cand in results.candidates:
        w.write_u32(cand.id)
        _write_config(w, cand.config)
        _write_scores(w, cand.scores)
        w.write_f64(cand.composite_score)
    Path(path).write_bytes(w.buf)


def read_autotune_results_bin(path: str | Path) -> AutotuneResults:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC_RESULTS:
        raise ValueError("bad autotune results bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported autotune results version {version}")
    seed = int(r.read_u32())
    budget_ms = int(r.read_u32())
    count = r.read_u32()
    candidates: list[AutotuneCandidate] = []
    for _ in range(count):
        cand_id = int(r.read_u32())
        config = _read_config(r)
        scores = _read_scores(r)
        composite_score = float(r.read_f64())
        candidates.append(
            AutotuneCandidate(
                id=cand_id,
                config=config,
                scores=scores,
                composite_score=composite_score,
            )
        )
    return AutotuneResults(candidates=candidates, seed=seed, budget_ms=budget_ms)


def write_autotune_choice_bin(choice: AutotuneChoice, path: str | Path) -> None:
    w = _BinWriter()
    w.buf.extend(MAGIC_CHOICE)
    w.write_u8(VERSION)
    w.write_u32(choice.selected_id)
    _write_config(w, choice.config)
    w.write_str(choice.reason)
    _write_scores(w, choice.scores)
    Path(path).write_bytes(w.buf)


def read_autotune_choice_bin(path: str | Path) -> AutotuneChoice:
    data = Path(path).read_bytes()
    if data[:4] != MAGIC_CHOICE:
        raise ValueError("bad autotune choice bin magic")
    r = _BinReader(data[4:])
    version = r.read_u8()
    if version != VERSION:
        raise ValueError(f"unsupported autotune choice version {version}")
    selected_id = int(r.read_u32())
    config = _read_config(r)
    reason = r.read_str()
    scores = _read_scores(r)
    return AutotuneChoice(
        selected_id=selected_id, config=config, reason=reason, scores=scores
    )
