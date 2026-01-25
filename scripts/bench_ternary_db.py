#!/usr/bin/env python3
"""Benchmark ternary database construction time and memory."""

import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stc.ternary_db import (
    build_bgc_table,
    build_q0_table,
    build_q1_table,
    compute_base_vectors,
    save_db,
    TernaryDB,
)


def bench_bgc():
    """Benchmark BGC table construction."""
    tracemalloc.start()
    start = time.time()

    bgc = build_bgc_table()

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("BGC table:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")
    print(f"  Size: {len(bgc)} bytes (65536 expected)")


def bench_q0():
    """Benchmark q0 construction."""
    base = compute_base_vectors()

    tracemalloc.start()
    start = time.time()

    q0 = build_q0_table(base)

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nq0 table:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")
    print(f"  Entries: {len(q0)} (expected ~936)")


def bench_q1():
    """Benchmark q1 construction."""
    base = compute_base_vectors()
    q0 = build_q0_table(base)

    tracemalloc.start()
    start = time.time()

    q1 = build_q1_table(q0, base)

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nq1 table:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")
    print(f"  Entries: {len(q1)} (expected ~438,312)")


def bench_full_db():
    """Benchmark complete database build and save."""
    base = compute_base_vectors()

    tracemalloc.start()
    start = time.time()

    bgc = build_bgc_table()
    q0 = build_q0_table(base)
    q1 = build_q1_table(q0, base)
    db = TernaryDB(bgc, q0, q1, base)

    save_db(db, Path("/tmp/ternary_db.json"))

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nFull DB build + save:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {peak / 1024 / 1024:.2f} MB peak")


if __name__ == "__main__":
    print("Ternary DB Construction Benchmarks")
    print("=" * 50)
    print("\nWarning: This benchmark is very slow (especially q1).")
    print("Consider running individual benchmarks instead.\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--q0":
        bench_q0()
    elif len(sys.argv) > 1 and sys.argv[1] == "--full":
        bench_full_db()
    else:
        print("Usage:")
        print("  bench_ternary_db.py --q0    # Benchmark q0 only")
        print("  bench_ternary_db.py --full  # Benchmark full DB (very slow)")
