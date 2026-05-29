"""profiling/nsys_decode.py — Phase 1 (STUB).

Capture an Nsight Systems timeline of a vLLM decode loop on the target model
and identify the dominant kernels by total time (attention/PagedAttention,
GEMMs, RMSNorm, sampling).

Intended usage (wrapped by `make profile-nsys`):

    nsys profile -o bench/results/nsys_decode \\
        python profiling/nsys_decode.py --model Qwen/Qwen2.5-1.5B

NOTE: vLLM has no native Windows support — this phase runs under WSL2 (or a
Linux box). See docs/00_environment.md. The custom-kernel phases do not need it.

TODO(Phase 1):
  - load the model in vLLM, run N decode steps (batch 1) inside an nsys capture,
  - emit a kernel-by-total-time summary (parse `nsys stats`),
  - write the dominant-kernel list to bench/results/ for roofline.py.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--out", default="bench/results/nsys_decode")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Phase 1 not implemented yet (model={args.model}). "
        "See docs/01_roofline.md."
    )


if __name__ == "__main__":
    raise SystemExit(main())
