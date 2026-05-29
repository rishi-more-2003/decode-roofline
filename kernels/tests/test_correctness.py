"""Correctness gate for Phase 2 fused INT4 dequant+GEMV.

Correctness always runs before any benchmark (§7). The oracle dequantizes with
the exact nibble/scale formula and accumulates in FP32. The fused kernel also
accumulates in FP32 and returns FP16, so we compare against the FP32 oracle after
rounding it to FP16.
"""

from __future__ import annotations

import pytest
import torch

from harness.load_model import SHAPES, make_test_weight, quantize_int4_groupwise
from harness.reference_gemv import reference_fp32
from kernels.load import load_kernel


GROUP_SIZE = 128


@pytest.mark.parametrize(
    ("shape_name", "seed"),
    [
        ("kv_proj", 0),
        ("attn_qo", 1),
        ("mlp_down", 2),
        ("mlp_gate_up", 3),
    ],
)
def test_fused_matches_reference(shape_name: str, seed: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused kernel correctness")

    n, k = SHAPES[shape_name]
    torch.manual_seed(seed)
    W = make_test_weight(n, k, seed=seed)
    qweight, scales = quantize_int4_groupwise(W, GROUP_SIZE)
    x = torch.randn(k, device="cuda", dtype=torch.float16)

    qweight = qweight.to("cuda")
    scales = scales.to("cuda")
    mod = load_kernel(verbose=False)

    fused = mod.dequant_gemv(x, qweight, scales, GROUP_SIZE)
    ref = reference_fp32(qweight, scales, x, GROUP_SIZE).to(torch.float16)
    torch.cuda.synchronize()

    max_abs = (fused - ref).abs().max().item()
    # The fused path and oracle use the same FP32 accumulation, then round to
    # FP16. Any residual should be only roundoff/ordering noise.
    assert torch.allclose(fused, ref, rtol=2e-3, atol=2e-3), (
        shape_name,
        max_abs,
    )


def test_pack_dequant_roundtrip_cpu():
    W = make_test_weight(17, 256, seed=123)
    qweight, scales = quantize_int4_groupwise(W, GROUP_SIZE)
    ref = reference_fp32(qweight, scales, torch.eye(256, dtype=torch.float16), GROUP_SIZE)
    assert ref.shape == (256, 17)
