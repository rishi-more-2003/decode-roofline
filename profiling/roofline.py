"""profiling/roofline.py — Phase 1 (STUB).

Build the RTX 4070 Laptop roofline plot from the ncu metrics CSV and place each
dominant decode kernel on it (arithmetic intensity vs achieved throughput).

The memory ceiling uses the MEASURED DRAM bandwidth from docs/00_environment.md,
NOT the ~256 GB/s datasheet number.

Output: bench/results/roofline.png (checked in so the README renders w/o a GPU).

TODO(Phase 1):
  - read bench/results/ncu_kernels.csv,
  - compute arithmetic intensity (FLOP/byte) per kernel,
  - draw memory ceiling (measured GB/s) + compute ceiling, plot kernel points,
  - save PNG + write the prose conclusion into docs/01_roofline.md.
"""

from __future__ import annotations

import argparse

# Set from docs/00_environment.md once measured (do NOT hardcode datasheet).
MEASURED_PEAK_BW_GBPS: float | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", default="bench/results/ncu_kernels.csv")
    parser.add_argument("--out", default="bench/results/roofline.png")
    parser.parse_args()
    if MEASURED_PEAK_BW_GBPS is None:
        raise NotImplementedError(
            "Set MEASURED_PEAK_BW_GBPS from docs/00_environment.md (measured, "
            "not datasheet) and implement the plot. See docs/01_roofline.md."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
