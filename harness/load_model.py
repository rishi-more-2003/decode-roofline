"""harness/load_model.py — Phase 2: INT4 group-quant weight tensors.

Self-defined symmetric INT4 group quantization (group size 128) used by both the
fused CUDA kernel and the reference (docs/02_kernel_design.md):

  per group of G weights along K:
    scale = max(|W|)/7
    q     = clamp(round(W/scale), -7, 7)        # signed 4-bit
    nib   = q + 8  in [1,15]                     # zero exact at nib=8
    w_hat = (nib - 8) * scale                    # dequant

Packing: 8 nibbles per int32 (nibble t at bits 4t) => qweight [N, K/8] int32.
Scales: [N, K/G] fp16.

For kernel study we use representative random weights of the target shapes
(model quality is irrelevant, §3). `load_qwen_mlp_weight` can pull a real layer
if desired.
"""

from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"

# Representative decode GEMV shapes (N_out, K_in) for Qwen2.5-1.5B.
SHAPES = {
    "mlp_gate_up": (8960, 1536),   # the Phase 1 headline kernel
    "mlp_down": (1536, 8960),
    "attn_qo": (1536, 1536),
    "kv_proj": (256, 1536),        # GQA k/v
}


def make_test_weight(n: int, k: int, seed: int = 0):
    """Representative FP16 weight matrix [N, K] (normal, scaled like a layer)."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=g, dtype=torch.float32) * (1.0 / (k ** 0.5))
    return w.to(torch.float16)


def quantize_int4_groupwise(W, group_size: int = 128):
    """Quantize W [N,K] -> (qweight [N,K/8] int32, scales [N,K/G] fp16)."""
    import torch

    N, K = W.shape
    assert K % group_size == 0, "K must be divisible by group_size"
    assert group_size % 8 == 0, "group_size must be divisible by 8 (packing)"

    Wf = W.detach().float()
    Wg = Wf.view(N, K // group_size, group_size)
    maxabs = Wg.abs().amax(dim=2, keepdim=True)
    scale = (maxabs / 7.0).clamp(min=1e-8)                 # [N, G, 1]
    q = torch.round(Wg / scale).clamp(-7, 7)               # [-7, 7]
    nib = (q + 8).to(torch.int64).view(N, K // 8, 8)       # [1, 15]

    shifts = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    packed = (nib << shifts).sum(dim=2).to(torch.int32)    # wraps to 32-bit pattern
    scales = scale.squeeze(-1).to(torch.float16)           # [N, K/G]
    return packed.contiguous(), scales.contiguous()


def dequant_int4_groupwise(qweight, scales, group_size: int, K: int, out_dtype=None):
    """Inverse of quantize_int4_groupwise -> W [N,K] (fp32 by default)."""
    import torch

    N = qweight.shape[0]
    shifts = (torch.arange(8, device=qweight.device) * 4).view(1, 1, 8)
    # Reinterpret int32 bit-pattern as unsigned for nibble extraction.
    qi = qweight.to(torch.int64) & 0xFFFFFFFF
    nib = (qi.unsqueeze(-1) >> shifts) & 0xF               # [N, K/8, 8]
    q = nib.reshape(N, K).to(torch.float32) - 8.0
    scale = scales.to(torch.float32).repeat_interleave(group_size, dim=1)  # [N,K]
    W = q * scale
    return W if out_dtype is None else W.to(out_dtype)


def load_qwen_mlp_weight(model_name: str = DEFAULT_MODEL, layer: int = 0):
    """Pull a real MLP gate_proj weight [N,K] (FP16) for sanity, if wanted."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    w = model.model.layers[layer].mlp.gate_proj.weight.detach().clone()
    del model
    return w
