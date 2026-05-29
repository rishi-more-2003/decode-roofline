# 02 — Fused dequant + GEMV kernel design

**Phase 2.** Design reasoning and the explicit memory-traffic arithmetic that
*predicts* the speedup before we measure it. Phase 1 proved decode is
memory-bound (the MLP GEMV runs at ~95% of the ~250 GB/s roofline), so at batch
1 the only lever that matters is **bytes moved**.

## Quantization scheme (self-defined, unambiguous)

Symmetric **INT4 group quantization**, group size **G = 128** along K:

- Per group of 128 weights: `scale = max(|W|)/7`, `q = clamp(round(W/scale), -7, 7)`.
- Stored as a nibble `nib = q + 8 ∈ [1,15]` (zero is exactly representable at
  `nib = 8`). Dequant: `w = (nib - 8) * scale`.
- Packing: **8 nibbles per `int32`** (nibble `t` at bits `4t`), so `qweight` is
  `[N, K/8] int32`; `scales` is `[N, K/G] fp16`.

We define our own format (rather than AWQ/GPTQ) because kernel behavior is the
object of study; model quality is irrelevant (§3), and a self-defined scheme
keeps the kernel and its correctness oracle exact and dependency-free.

## Why fusing helps (the wasted round-trip)

The two-op baseline (`harness/reference_gemv.py`) is what cuBLAS-style code does:

1. **dequant**: read packed INT4 `W_q` + scales → **write FP16 `W` to DRAM**,
2. **GEMV**: **read FP16 `W` back from DRAM** → produce `y`.

Step 2 re-reads the *entire* dequantized weight matrix that step 1 just wrote.
The fused kernel dequantizes inside the GEMV load path, so the FP16 `W` never
touches DRAM at all.

## Memory-traffic arithmetic (predict before measuring)

For `W` of shape `[N, K]`, INT4 + FP16 scales, group `G=128`. Bytes:

| Path | DRAM bytes (read + write) |
| --- | --- |
| separate **dequant** | read `N·K/2` (packed) + `N·K/G·2` (scales); **write `2·N·K`** (fp16 W) |
| separate **GEMV** | read `2·N·K` (fp16 W) + `2·K` (x); write `2·N` (y) |
| **separate total** | **≈ `4·N·K + N·K/2`** |
| **fused** | read `N·K/2` (packed) + `N·K/G·2` (scales) + `2·K` (x); write `2·N` |
| **fused total** | **≈ `N·K/2`** (packed weights dominate) |

### Worked example — MLP gate/up proj (the Phase 1 headline kernel)

Qwen2.5-1.5B: `N = 8960`, `K = 1536`, `G = 128`.

| Path | DRAM traffic |
| --- | --- |
| separate dequant (read packed 6.88 MB + write fp16 27.5 MB) | ~34.4 MB |
| separate GEMV (read fp16 27.5 MB) | ~27.5 MB |
| **separate total** | **~61.9 MB** |
| **fused** (read packed 6.88 MB + scales 0.21 MB + x) | **~7.1 MB** |

**Predicted, from the roofline (memory-bound ⇒ time ∝ bytes):**

- Fused vs the **two-op dequant baseline**: ~61.9 / 7.1 ≈ **8.7× less traffic**.
- Fused vs a plain **FP16 GEMV** (weights already fp16, no quant — the Phase 1
  baseline): ~27.5 / 7.1 ≈ **3.9× less traffic**.

So before measuring we predict roughly **3.9–8.7×** depending on which baseline,
*if* the fused kernel also hits a high fraction of the ~250 GB/s ceiling. The
honest number to beat is the two-op dequant baseline (§6); we also report the
FP16-GEMV comparison for context.

## Ada (SM 8.9) tuning notes

- **One warp per output row**; 4 warps/block. Lanes stride the `[N, K/8] int32`
  packed row by 32, giving fully **coalesced 128-byte loads**.
- `x` (FP16, K elements) is tiny and reused by every row in the block → stays
  hot in **L2** (Phase 1 showed L2 is otherwise idle at batch 1).
- Each `int32` unpacks 8 nibbles; **accumulate in FP32**, warp-reduce with
  `__shfl_down_sync`.
- Scales: one FP16 load per 16 `int32`s (since `G/8 = 16`).
- Future: vectorize to `int4`/`uint4` loads (32 nibbles/lane) if bandwidth-bound
  headroom remains.

## Correctness plan (gates everything — §7)

- Oracle: dequantize to **FP32** with the exact `(nib-8)*scale` formula, then
  `W @ x` in FP32 — the fused FP32 accumulation should match within tight
  tolerance (same dequant values).
- `kernels/tests/test_correctness.py`: multiple shapes (gate/up, down, q/o) and
  seeds; assert `allclose` within tolerance **before any timing is reported**.

## Phase 2 result (correctness first, then speed)

Correctness gate:

```bash
export TORCH_CUDA_ARCH_LIST=8.9
MSYS_NO_PATHCONV=1 cmd.exe /c "scripts\with_msvc.bat python -m pytest kernels\tests\test_correctness.py -v"
```

Result: **5/5 passed** across `kv_proj`, `attn_qo`, `mlp_down`, and
`mlp_gate_up`; max observed absolute error in the benchmark gate:
`2.44e-4` vs the FP32 oracle rounded to FP16.

Benchmark gate (`mlp_gate_up`, `N=8960`, `K=1536`, `G=128`; reproduced by
`scripts/reproduce.sh`, latest data in `bench/results/bench.csv`):

| Path | median | IQR | modeled traffic | effective BW | Speedup |
| --- | --- | --- | --- | --- | --- |
| two-op PyTorch dequant→GEMV baseline | 5069 µs | 214 µs | 62.2 MB | 12.3 GB/s | 1.0× |
| **fused INT4 dequant+GEMV** | **55.8 µs** | 10.2 µs | **7.12 MB** | **128 GB/s** | **90.8×** |

The huge speedup is against an intentionally literal two-op PyTorch baseline:
it materializes the FP16 dequantized matrix through many tensor ops before
cuBLAS reads it back. This is the §6 baseline to beat, but it is not a fair
comparison to a hand-tuned custom dequant kernel. The more important hardware
result is the `ncu` trace of the fused kernel:

```bash
MSYS_NO_PATHCONV=1 cmd.exe /c "scripts\with_msvc.bat ncu --kernel-name regex:dequant_gemv_kernel --launch-count 1 --metrics dram__bytes.sum,gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active --csv python bench\profile_dequant_gemv.py"
```

`ncu` result for `dequant_gemv_kernel`:

| Metric | Value |
| --- | --- |
| correctness before profiled launch | PASS (`max_abs=9.77e-4`) |
| `dram__bytes.sum` | 7.10 MB |
| `gpu__time_duration.sum` | 31.81 µs |
| achieved DRAM throughput | **~223 GB/s** (`7.10 MB / 31.81 µs`) |
| `gpu__dram_throughput...pct_of_peak` | **86.38%** |
| SM throughput | 64.27% |
| L2 hit rate | 5.59% |
| occupancy | 87.65% |

This meets the Phase 2 exit condition: the fused kernel is correct, and `ncu`
shows it operates near the measured roofline (about **89% of the 250 GB/s
achievable ceiling**, or **86% of ncu's hardware peak**) while moving ~8.7× fewer
bytes than the literal two-op baseline.
