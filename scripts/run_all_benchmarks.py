#!/usr/bin/env python3
"""
=============================================================================
MUDRA DECOMPILER: REPRODUCE ALL BENCHMARKS SCRIPT
=============================================================================

Runs the complete empirical verification pipeline:
1. Unit and invariant integration tests
2. Cross-domain DeonticBench generalization
3. Full SARA U.S. IRC benchmark (Tracks 1A, 1B, 1C, 1D, Track 2 Binary & Numeric)
"""

import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(title: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 80)
    print(f"STEP: {title}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    dt = time.time() - t0
    ok = (res.returncode == 0)
    status_str = "PASSED" if ok else "FAILED"
    print(f"[{status_str}] in {dt:.2f}s")
    return ok


def main():
    print("=" * 80)
    print("MUDRA DECOMPILER: FORMAL REPRODUCIBILITY VERIFICATION")
    print("=" * 80)

    stages = [
        ("Unit & Invariant Integration Tests", [sys.executable, "-m", "unittest", "tests/test_decompiler_unit.py"]),
        ("Cross-Domain DeonticBench Generalization", [sys.executable, "benchmarks/test_crossdomain_deonticbench.py"]),
        ("SARA U.S. IRC Benchmark (Zero-Shot Tracks 1 & 2)", [sys.executable, "benchmarks/test_sara_benchmark.py"]),
    ]

    results = []
    all_ok = True
    for title, cmd in stages:
        ok = run_step(title, cmd)
        results.append((title, ok))
        if not ok:
            all_ok = False

    print("\n" + "=" * 80)
    print("FINAL REPRODUCIBILITY SUMMARY")
    print("=" * 80)
    for title, ok in results:
        status_str = "PASS" if ok else "FAIL"
        print(f"  {title:<60s}: {status_str}")
    print("=" * 80)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
