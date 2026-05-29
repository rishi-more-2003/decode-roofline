# 03 — Results: regime sweep & honest attribution

> **Status: STUB — filled in Phase 3.** The senior move: show the win at
> batch 1, then show it shrink and vanish as the kernel becomes compute-bound,
> and explain why via the Phase 1 roofline.

## Method

- Baseline: separate dequant + GEMV (PyTorch/cuBLAS), the §6 reference.
- Fused: `kernels/dequant_gemv.cu`.
- Timing: CUDA events, fixed seeds, `W` warmup + `M` measured iters,
  report **median + IQR** (not best-case — exposes thermal throttling on this
  mobile part). Each row is gated by a passing correctness check in the same run.
- All raw numbers in `bench/results/*.csv`; plots checked in.

## Regime sweep (batch × hidden)

| batch | hidden | baseline median (µs) | fused median (µs) | speedup | fused achieved BW (GB/s) | % of peak |
| --- | --- | --- | --- | --- | --- | --- |
| 1  | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 2  | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 4  | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 8  | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 16 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 32 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

![sweep placeholder](../bench/results/sweep.png)

## Attribution (the honest part)

> _TODO — e.g.: "≈11% at batch 1, ≈3% by batch 8, noise by batch 32, because as
> batch grows the GEMV becomes a GEMM with rising arithmetic intensity and
> crosses the roofline ridge into compute-bound territory, where eliminating a
> DRAM round-trip no longer matters."_ Tie explicitly to `docs/01_roofline.md`.

## Thermal note

> _TODO — GPU temp trend across the sweep; whether throttling widened the IQR._
