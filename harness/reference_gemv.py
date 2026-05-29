"""harness/reference_gemv.py — Phase 2: correctness oracle + baselines.

Three reference paths for the fused INT4 dequant+GEMV kernel:

  1. reference_fp32   : dequant to FP32, then W @ x in FP32  (the CORRECTNESS
                        ORACLE — high precision, exact dequant formula).
  2. baseline_two_op  : dequant to FP16 (writes W to DRAM), then cuBLAS GEMV in
                        FP16 (reads W back) — the §6 "dequant-then-GEMV as two
                        ops" baseline to beat. Two DRAM round-trips of W.
  3. baseline_fp16_gemv: a plain FP16 GEMV on an already-dequantized W (weights
                        resident in DRAM as FP16) — the Phase 1 baseline, for
                        context on the quant-free comparison.

x is [K] (batch 1) or [B, K]; weight is [N, K]; output is [N] or [B, N].
"""

from __future__ import annotations

from harness.load_model import dequant_int4_groupwise


def reference_fp32(qweight, scales, x, group_size: int):
    """Correctness oracle: dequant FP32 then matmul in FP32. Returns [.., N]."""
    K = x.shape[-1]
    W = dequant_int4_groupwise(qweight, scales, group_size, K)  # [N,K] fp32
    return x.float() @ W.t()


def baseline_two_op(qweight, scales, x, group_size: int):
    """§6 baseline: dequant to FP16 (DRAM write), then cuBLAS GEMV (DRAM read)."""
    import torch

    K = x.shape[-1]
    W = dequant_int4_groupwise(qweight, scales, group_size, K, out_dtype=torch.float16)
    return x @ W.t()


def baseline_fp16_gemv(W_fp16, x):
    """Plain FP16 GEMV on a resident FP16 weight (Phase 1 baseline)."""
    return x @ W_fp16.t()
