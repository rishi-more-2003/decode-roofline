"""kernels/tests/test_bench.py — Phase 2 (STUB).

Latency + achieved-bandwidth bench for the fused kernel vs the baseline.

HARD RULE (§7): this bench must FIRST run the correctness check in the same
process and abort if it fails — no timing without a passing correctness test.
Timing uses CUDA events, fixed seeds, warmup + measured iters, and reports
**median + IQR** (not best-case) to expose thermal throttling on this mobile GPU.

TODO(Phase 2):
  - call the correctness check; bail on failure,
  - time baseline vs fused with CUDA events (warmup W, measure M),
  - compute achieved GB/s from the memory-traffic model (docs/02),
  - write rows to bench/results/bench.csv.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 2 not implemented yet — correctness gates this")
def test_bench_batch1():
    raise NotImplementedError
