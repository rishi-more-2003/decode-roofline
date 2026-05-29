"""Minimal decode-harness entry point.

The final artifact separates concerns:

- `profiling/nsys_decode.py` is the real batch-1 model decode workload used for
  Phase 1 timeline and roofline measurement.
- `bench/profile_dequant_gemv.py` and `bench/sweep.py` are the standalone
  projection-level harnesses where the fused kernel is evaluated.

This file remains as the spec-level harness entry point and delegates to the
native decode workload so `python harness/decode_harness.py` is directly useful.
"""

from __future__ import annotations

from profiling.nsys_decode import main as decode_main


def main() -> int:
    return decode_main()


if __name__ == "__main__":
    raise SystemExit(main())
