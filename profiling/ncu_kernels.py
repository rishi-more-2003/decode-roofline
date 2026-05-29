"""profiling/ncu_kernels.py — Phase 1 (STUB).

For each dominant decode kernel (from nsys_decode.py), capture Nsight Compute
metrics and parse them into a CSV that roofline.py consumes.

Metrics collected (see profiling/metrics.md for the why):
  - dram__bytes.sum / elapsed  -> achieved DRAM throughput (GB/s)
  - sm__throughput.avg.pct_of_peak_sustained_elapsed
  - memory vs compute issue-stall reasons
  - lts__t_sector_hit_rate.pct (L2 hit rate)
  - sm__warps_active.avg.pct_of_peak_sustained_active (achieved occupancy)

Intended usage (wrapped by `make profile-ncu`):

    ncu --set full --metrics <list> --csv \\
        python profiling/ncu_kernels.py --model Qwen/Qwen2.5-1.5B

TODO(Phase 1):
  - drive a single decode step so ncu profiles the relevant kernels,
  - parse ncu --csv output into bench/results/ncu_kernels.csv,
  - confirm counter access works (ERR_NVGPUCTRPERM => fix per docs/00).
"""

from __future__ import annotations

import argparse

NCU_METRICS = [
    "dram__bytes.sum",
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--out", default="bench/results/ncu_kernels.csv")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Phase 1 not implemented yet (model={args.model}); "
        f"will collect metrics: {NCU_METRICS}. See docs/01_roofline.md."
    )


if __name__ == "__main__":
    raise SystemExit(main())
