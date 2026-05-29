"""bench/sweep.py — Phase 3 (STUB).

Sweep batch size {1, 2, 4, 8, 16, 32} and a couple of hidden sizes; measure
latency and achieved bandwidth for baseline vs fused. Produces the data behind
the "win at batch 1, vanishes by batch 32" attribution story (docs/03_results.md).

Writes CSV + plots into bench/results/ (checked in so README renders w/o a GPU).

TODO(Phase 3):
  - for each (batch, hidden): run correctness gate, then time baseline vs fused
    (CUDA events, median + IQR), compute achieved GB/s,
  - save bench/results/sweep.csv + bench/results/sweep.png,
  - tie the regime crossover back to the Phase 1 roofline ridge point.
"""

from __future__ import annotations

BATCHES = [1, 2, 4, 8, 16, 32]
HIDDEN_SIZES = []  # TODO: fill from the chosen model's projection shapes


def main() -> int:
    raise NotImplementedError("Phase 3 sweep not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
