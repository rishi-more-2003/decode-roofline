"""harness/reference_gemv.py — Phase 2 (STUB).

PyTorch/cuBLAS reference for the fused kernel. This is BOTH:
  1. the correctness oracle (dequant-then-GEMV as two explicit ops), and
  2. the baseline to beat at batch 1.

The reference must be unambiguous and exactly match the fused kernel's intended
math within tolerance, so correctness comparisons are meaningful (§7).

TODO(Phase 2):
  - dequantize packed 4-bit W (+ scales/zeros, group size G) to FP16,
  - y = x @ W.T  via torch.matmul (cuBLAS), batch 1 and batched,
  - return y for correctness comparison; time it for the baseline.
"""

from __future__ import annotations


def dequant_then_gemv(*args, **kwargs):
    """Reference: dequantize, then GEMV via cuBLAS. STUB."""
    raise NotImplementedError("Phase 2 reference not implemented yet.")
