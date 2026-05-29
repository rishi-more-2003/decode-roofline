"""profiling/ncu_kernels.py — Phase 1: per-kernel ncu metric capture.

Wraps the decode workload (profiling/nsys_decode.py) under Nsight Compute,
capturing the metrics that place each decode kernel on the roofline, and writes
the raw CSV to bench/results/ncu_gemv.csv (parsed by profiling/roofline.py).

By default it profiles the cuBLAS weight GEMVs (the dominant decode kernels,
matched by their *demangled* name since their short name is just "kernel").

    python profiling/ncu_kernels.py --model Qwen/Qwen2.5-1.5B

Notes:
  - ncu must wrap the Python process, so this script shells out to `ncu`.
  - On Windows, `ncu` is the .BAT launcher; we invoke via cmd. No MSVC needed
    (the decode workload does not JIT-compile).
  - Requires GPU performance-counter permission (see docs/00_environment.md).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

METRICS = [
    "dram__bytes.sum",
    "gpu__time_duration.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--kernel-regex", default="gemv",
                        help="match against demangled kernel name")
    parser.add_argument("--launch-count", type=int, default=12)
    parser.add_argument("--out", default=str(ROOT / "bench/results/ncu_gemv.csv"))
    args = parser.parse_args()

    workload = [
        sys.executable, "profiling/nsys_decode.py",
        "--model", args.model,
        "--warmup-steps", "2", "--decode-steps", "6", "--no-latency-report",
    ]
    ncu = [
        "ncu", "--kernel-name-base", "demangled",
        "--kernel-name", f"regex:{args.kernel_regex}",
        "--launch-count", str(args.launch_count),
        "--metrics", ",".join(METRICS), "--csv",
    ]
    cmd = ncu + workload

    is_windows = os.name == "nt"
    print("running:", " ".join(cmd))
    with open(args.out, "w", encoding="utf-8") as out:
        if is_windows:
            # Route through cmd so the ncu.BAT launcher resolves.
            full = "ncu " + subprocess.list2cmdline(cmd[1:])
            proc = subprocess.run(["cmd", "/c", full], stdout=out,
                                  stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
        else:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE,
                                  text=True, cwd=str(ROOT))

    if "ERR_NVGPUCTRPERM" in (proc.stderr or ""):
        print("ERROR: no GPU counter permission. See docs/00_environment.md.")
        return 2
    print(f"wrote {args.out} (rc={proc.returncode})")
    print("next: python profiling/roofline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
