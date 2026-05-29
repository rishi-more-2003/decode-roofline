# -decode-roofline
Kernel-level profiling of LLM decode on consumer GPUs (RTX 4070, Ada): proving decode is memory-bandwidth-bound against the hardware roofline, then beating the baseline with a fused dequant+GEMV CUDA kernel — with honest, regime-aware attribution.
