# 02 — Fused dequant + GEMV kernel design

> **Status: STUB — filled in Phase 2.** Design reasoning and the explicit
> memory-traffic arithmetic that *predicts* the speedup before we measure it.

## Problem

At batch 1, decode's matmuls degenerate into matrix–vector products
(`y = W · x`, `W` is `[out, in]`, `x` is `[in]`). This is pure weight streaming
from DRAM — memory-bound. With quantized weights, the naive path is two ops:

1. **dequant:** read packed 4-bit `W_q` + scales → write FP16 `W` to DRAM,
2. **GEMV:** read FP16 `W` back from DRAM → produce `y`.

Step 2's re-read of the full dequantized weight matrix is a wasted DRAM
round-trip. Fusing dequant into the GEMV load path eliminates it.

## Memory-traffic arithmetic (predict before measuring)

Let `W` be `[N, K]`, 4-bit quantized with group size `G`, FP16 scales/zeros.

| Path | Bytes read | Bytes written |
| --- | --- | --- |
| separate dequant | `N·K/2` (packed) + scales | `2·N·K` (FP16 W) |
| separate GEMV | `2·N·K` (FP16 W) + `2K` (x) | `2N` (y) |
| **separate total** | **≈ `4·N·K`** | **≈ `2·N·K`** |
| **fused** | **`N·K/2` + scales + `2K`** | **`2N`** |

Predicted traffic reduction ≈ **_TODO×_** → predicted speedup from the roofline
(memory-bound ⇒ speedup ≈ traffic ratio): **_TODO_**.

> Fill the real `N, K, G` for the chosen model layer and compute the numbers.

## Ada (SM 8.9) tuning notes

- Coalesced, vectorized loads of packed weights (`int4`/`uint32` packing,
  `float4`/`int4` vector loads).
- Block/tile sizes chosen against L2 (and the 4070 Laptop's 128-bit bus).
- One block per output row-tile; reduce `x`-dot in registers/shared mem.
- Keep scales/zeros in registers or shared mem per group.
- _TODO: chosen quant format (AWQ/GPTQ), group size, dtype of accumulation._

## Risks / correctness notes

- Dequant math must match the reference exactly within tolerance (§7).
- Group-boundary handling for `K % G != 0`.
- Numerical accumulation order vs cuBLAS reference (tolerance budget).
