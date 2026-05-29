"""harness/load_model.py — Phase 2 (STUB).

Load model weights and build the quantized weight tensors used by the kernel.

Default targets (§3): Qwen2.5-1.5B / Qwen2.5-3B or Llama-3.2-1B / Llama-3.2-3B.
ALWAYS confirm VRAM headroom with nvidia-smi before a run (8 GB budget).

TODO(Phase 2):
  - load FP16 weights for the baseline GEMV,
  - build a 4-bit (AWQ/GPTQ) variant for the fused kernel (packed W, scales,
    zeros, group size G) — or synthesize representative quantized tensors if a
    real checkpoint is overkill for kernel study,
  - return the specific projection layer(s) we benchmark.
"""

from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"


def load(model_name: str = DEFAULT_MODEL):
    """Load weights + build quantized tensors. STUB."""
    raise NotImplementedError("Phase 2 model loading not implemented yet.")
