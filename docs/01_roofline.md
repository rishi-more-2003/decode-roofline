# 01 — Roofline (RTX 4070 Laptop) & per-kernel placement

**Phase 1 result.** Profiler data (nsys timeline + ncu per-kernel metrics) on a
batch-1 decode loop of **Qwen2.5-1.5B (FP16)** proves decode is overwhelmingly
**memory-bandwidth-bound**, dominated by weight-streaming GEMVs.

Repro: `profiling/nsys_decode.py` (workload), `profiling/measure_bandwidth.py`
(ceiling), per-kernel ncu capture, then `profiling/roofline.py`.

## The 4070 Laptop roofline

- **Memory ceiling (achievable): ~250 GB/s** — best measured via the ncu
  `dram__bytes.sum` counter on the actual decode GEMV (~247 GB/s), rounded.
  ncu's hardware-theoretical peak is ~259 GB/s (≈ the 256 GB/s datasheet). We
  quote against the achievable 250 and report ncu's %-of-peak alongside.
- **Compute ceiling (approx): ~19 TFLOP/s** FP16 CUDA-core (4608 cores × 2 ×
  ~2.1 GHz). The cuBLAS `gemvx` kernels are CUDA-core, not tensor-core.
- **Ridge point: ~77 FLOP/byte.** A batch-1 GEMV has arithmetic intensity
  ≈ **1 FLOP/byte** (2·N·K FLOPs over ~2·N·K bytes of FP16 weights) — ~77× to
  the left of the ridge, i.e. memory-bound by construction.

![roofline](../bench/results/roofline.png)

## Where decode time goes (nsys, 40 decode steps)

Total GPU kernel time 635 ms; by kernel:

| Kernel class | % of GPU kernel time | What it is |
| --- | --- | --- |
| `internal::gemvx::kernel` (cuBLAS GEMV) | **81.3%** (59.4 + 21.9) | the weight matrix–vector products (q/k/v/o + gate/up/down proj) |
| `cutlass ... f16 gemm` | ~4.2% | lm_head / larger matmuls |
| elementwise / reduce / cat / softmax | ~14% | RMSNorm, RoPE, residual, SDPA glue, sampling |

> Caveat (honest): GPU kernel time (635 ms) is only ~33% of decode wall time
> (~1.9 s over 40 steps). The rest is **kernel-launch / Python overhead** — at
> batch 1 each token fires ~196 tiny kernels. So decode is *both* launch-bound
> (gaps between kernels) *and*, when the GPU is actually working, memory-bound.
> A fused kernel attacks both: fewer launches **and** fewer bytes.

## Per-kernel placement (ncu, batch-1 decode GEMVs)

Measured against the ~250 GB/s achievable ceiling (full data:
`bench/results/ncu_kernels.csv`):

| Kernel (GEMV) | weight read | duration | achieved BW | % of peak | SM (compute) % | L2 hit % | occupancy % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP gate/up/down (1536↔8960) | ~27.5 MB | ~112 µs | **247 GB/s** | **~95%** | 24 | 1 | 24 |
| attn q/o proj (1536×1536) | ~4.74 MB | ~22 µs | 216 GB/s | ~84% | 21 | 4 | 22 |
| KV proj (GQA, 1536×256) | ~0.81 MB | ~6.2 µs | 131 GB/s | ~51% | 14 | 17 | 15 |
| attn score/av (fp32, tiny) | ~0.05 MB | ~2.8 µs | ~20 GB/s | ~8% | 2 | ~46 | 5 |

Reading the metrics:

- **DRAM throughput is the binding constraint.** The big MLP GEMVs run at
  **95% of peak bandwidth** while SM (compute) utilization is only ~24% — the
  hardware is streaming bytes, not crunching FLOPs.
- **L2 hit rate ~1%** on the large GEMVs: weights are read once from DRAM with
  no reuse — pure streaming, exactly the batch-1 signature.
- **Small GEMVs (KV proj) fall off the ceiling (~51%)**: at ~0.8 MB they're too
  small to amortize launch/latency; they're latency-bound, not bandwidth-bound.
  Their higher L2 hit (~17%) reflects the tiny footprint.

## Conclusion (Phase 1 exit criterion)

> **Decode is memory-bandwidth-bound on the RTX 4070 Laptop.** ~81% of
> decode-step GPU time is spent in cuBLAS weight GEMVs, and the dominant ones
> run at **84–95% of the measured ~250 GB/s DRAM roofline** with compute
> utilization of only 14–24% and ~1% L2 reuse. The single largest consumer is
> the **MLP gate/up/down GEMV at batch 1** (~27.5 MB of FP16 weight reads per
> call, ~95% of peak bandwidth). This is the kernel the Phase 2 fused
> dequant+GEMV targets: at batch 1 the win must come from **moving fewer
> bytes**, since the roofline says bytes — not FLOPs — set the time.

## Stretch / not done here

- A WSL2 + vLLM PagedAttention timeline (the spec's original Phase 1 vehicle)
  is deferred — vLLM has no native Windows support and the memory-bound
  conclusion does not depend on it. The native `transformers` loop profiled
  here lives in the same environment as the custom kernel, which makes the
  Phase 2/3 attribution cleaner.
