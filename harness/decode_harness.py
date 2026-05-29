"""harness/decode_harness.py — Phase 2 (STUB).

Minimal single-token decode loop using the target model's weights, with NO
vLLM. This is the standalone harness where the custom fused dequant+GEMV kernel
plugs in (per §4: custom-kernel work lives here, not inside vLLM).

TODO(Phase 2):
  - load weights via harness/load_model.py (FP16 + a 4-bit quantized variant),
  - run a batch-1 decode step swapping the projection between:
      * reference (harness/reference_gemv.py), and
      * fused kernel (kernels/load.py),
  - expose hooks for the correctness + bench tests.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Phase 2 decode harness not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
