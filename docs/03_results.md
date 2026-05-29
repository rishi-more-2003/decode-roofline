# 03 — Results: regime sweep & honest attribution

**Phase 3 result.** The fused kernel is a strong **batch-1 GEMV** primitive, not
a general GEMM replacement. As batch grows, batched GEMM reuses weights while the
current fused path simply launches one independent GEMV per batch row and
therefore rereads quantized weights B times.

Raw data: `bench/results/sweep.csv`. Plot: `bench/results/sweep.png`.

## Method

- Workload: synthetic self-defined INT4 group-quant weights matching
  Qwen2.5-1.5B projection shapes (`G=128`), fixed seeds.
- Fused: `kernels/dequant_gemv.cu`, run once per batch row for B>1.
- Baseline 1 (spec baseline): literal two-op PyTorch dequant→GEMV/GEMM.
- Baseline 2 (context): FP16 GEMM with already-dequantized weights resident in
  DRAM. This is the fair reminder that a batch-1 GEMV kernel should not be
  compared to a batched GEMM primitive at large B.
- Timing: CUDA events, warmup 5, measured iters 20, **median + IQR**. Each row
  is correctness-gated before timing in the same run.

## Regime sweep

![sweep](../bench/results/sweep.png)

### MLP gate/up projection (`N=8960`, `K=1536`)

| batch | two-op median (µs) | fused median (µs) | speedup vs two-op | speedup vs FP16 GEMM | fused effective GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1  | 5066 | 84   | 60.3× | 2.38× | 85 |
| 2  | 5181 | 191  | 27.1× | 0.59× | 75 |
| 4  | 4559 | 183  | 24.9× | 0.75× | 156 |
| 8  | 4673 | 239  | 19.6× | 0.52× | 239 |
| 16 | 4622 | 527  | 8.77× | 0.15× | 216 |
| 32 | 4576 | 1254 | 3.65× | 0.07× | 182 |

### Attention q/o projection (`N=1536`, `K=1536`)

| batch | two-op median (µs) | fused median (µs) | speedup vs two-op | speedup vs FP16 GEMM | fused effective GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1  | 640 | 157  | 4.08× | 0.75× | 8 |
| 2  | 922 | 207  | 4.46× | 0.50× | 12 |
| 4  | 634 | 173  | 3.68× | 0.47× | 28 |
| 8  | 706 | 364  | 1.94× | 0.31× | 27 |
| 16 | 538 | 656  | 0.82× | 0.12× | 30 |
| 32 | 721 | 1251 | 0.58× | 0.09× | 31 |

## Attribution (the honest part)

The Phase 1 roofline showed that the large batch-1 MLP GEMV streams FP16 weights
at ~95% of the RTX 4070 Laptop memory roofline. Phase 2's fused INT4 path moves
~7.1 MB instead of ~27.5 MB (FP16 GEMV) or ~62 MB (literal two-op dequant
baseline), so the batch-1 win is real and byte-driven.

But the win is **regime-specific**:

- Against the literal two-op dequant baseline, the big MLP projection remains
  faster through B=32, but speedup shrinks from **60× → 3.65×** because the fused
  implementation rereads the packed weights once per batch row while the
  baseline dequantizes once and then uses a batched GEMM.
- Against the FP16 GEMM context baseline, the fused batch-1 kernel wins only at
  **B=1** for the large MLP projection (**2.38×**) and loses by **B=2**. This is
  the clean attribution: when batch grows, weight reuse raises arithmetic
  intensity and GEMM becomes the right primitive.
- For the smaller attention q/o projection, the fused kernel is not the right
  primitive even at B=1 versus FP16 GEMM. Phase 1 already hinted why: smaller
  kernels do not saturate the memory roofline as well, so launch overhead and
  per-row work dominate.

**Bottom line:** the headline fused kernel is justified for **batch-1, large
weight-streaming projections** — especially the MLP gate/up/down GEMVs identified
in Phase 1. Its advantage shrinks quickly with batch because the problem stops
being pure GEMV weight streaming and becomes a GEMM/weight-reuse problem.

## Thermal / variance note

The sweep reports IQR rather than best-case. Variance widened notably on some
rows (`mlp_gate_up` two-op at B=4–8: IQR ~230–243 µs; fused at B=32: IQR ~1135 µs),
consistent with normal Windows/mobile-GPU jitter. The qualitative attribution is
robust: speedup vs two-op is monotonic downward with batch, and speedup vs FP16
GEMM is below 1 for all B>1 on the large MLP projection.
