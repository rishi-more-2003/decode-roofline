"""Phase 3 regime sweep: where the fused batch-1 win holds and vanishes.

The fused kernel is intentionally a batch-1 GEMV kernel. For B>1 this script
runs it as B independent GEMVs, while the baselines use batched matmul/GEMM.
That is the honest attribution: fusion wins when decode is pure weight
streaming at batch 1, but a specialized batch-1 kernel is not a general GEMM
replacement once batch grows and weight reuse appears.

Each row is correctness-gated before timing, and timing is CUDA-event median +
IQR. Writes:
  - bench/results/sweep.csv
  - bench/results/sweep.png
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from harness.load_model import SHAPES, dequant_int4_groupwise, make_test_weight, quantize_int4_groupwise
from harness.reference_gemv import reference_fp32
from kernels.load import load_kernel

BATCHES = [1, 2, 4, 8, 16, 32]
SHAPE_NAMES = ["mlp_gate_up", "attn_qo"]
GROUP_SIZE = 128
RESULTS = ROOT / "bench" / "results"
ROOFLINE_GBPS = 250.0


def _iqr(xs: list[float]) -> tuple[float, float, float]:
    s = sorted(xs)
    return statistics.median(s), s[len(s) // 4], s[(3 * len(s)) // 4]


def _time_cuda(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def _traffic_bytes(n: int, k: int, batch: int, group_size: int) -> dict[str, int]:
    packed = n * k // 2
    scales = n * (k // group_size) * 2
    x = batch * k * 2
    y = batch * n * 2
    fp16_w = 2 * n * k
    return {
        # Literal dequant once, then GEMM/GEMV: read packed/scales, write fp16 W,
        # read fp16 W once for GEMM. This baseline gets weight reuse as B grows.
        "two_op": packed + scales + fp16_w + fp16_w + x + y,
        # Current fused kernel run as B independent GEMVs reads packed/scales B
        # times. This is deliberate: it exposes when the batch-1 design stops
        # being the right primitive.
        "fused_loop": batch * (packed + scales + k * 2 + n * 2),
        # Context baseline: if dequantized weights already exist in FP16.
        "fp16_gemm": fp16_w + x + y,
    }


def _fused_batched(mod, X, qweight, scales, group_size: int):
    return torch.stack([mod.dequant_gemv(X[i], qweight, scales, group_size) for i in range(X.shape[0])])


def _two_op_batched(X, qweight, scales, group_size: int):
    W = dequant_int4_groupwise(qweight, scales, group_size, X.shape[-1], out_dtype=torch.float16)
    return X @ W.t()


def _fp16_batched(X, W_fp16):
    return X @ W_fp16.t()


def _run_row(shape_name: str, batch: int, warmup: int, iters: int) -> dict[str, str | int | float]:
    n, k = SHAPES[shape_name]
    W = make_test_weight(n, k, seed=1000 + batch + n)
    qweight, scales = quantize_int4_groupwise(W, GROUP_SIZE)
    Wq_fp16 = dequant_int4_groupwise(qweight, scales, GROUP_SIZE, k, out_dtype=torch.float16).to("cuda")
    qweight = qweight.to("cuda")
    scales = scales.to("cuda")
    X = torch.randn(batch, k, device="cuda", dtype=torch.float16)
    mod = load_kernel(verbose=False)

    # Correctness gate before timing.
    fused = _fused_batched(mod, X, qweight, scales, GROUP_SIZE)
    ref = reference_fp32(qweight, scales, X, GROUP_SIZE).to(torch.float16)
    torch.cuda.synchronize()
    max_abs = (fused - ref).abs().max().item()
    if not torch.allclose(fused, ref, rtol=2e-3, atol=2e-3):
        raise AssertionError(f"{shape_name} batch={batch} correctness failed: {max_abs}")

    two_times = _time_cuda(lambda: _two_op_batched(X, qweight, scales, GROUP_SIZE), warmup, iters)
    fused_times = _time_cuda(lambda: _fused_batched(mod, X, qweight, scales, GROUP_SIZE), warmup, iters)
    fp16_times = _time_cuda(lambda: _fp16_batched(X, Wq_fp16), warmup, iters)
    two_med, two_q1, two_q3 = _iqr(two_times)
    fused_med, fused_q1, fused_q3 = _iqr(fused_times)
    fp16_med, fp16_q1, fp16_q3 = _iqr(fp16_times)
    traffic = _traffic_bytes(n, k, batch, GROUP_SIZE)

    fused_gbps = traffic["fused_loop"] / (fused_med / 1e3) / 1e9
    two_gbps = traffic["two_op"] / (two_med / 1e3) / 1e9
    fp16_gbps = traffic["fp16_gemm"] / (fp16_med / 1e3) / 1e9
    return {
        "shape": shape_name,
        "batch": batch,
        "N": n,
        "K": k,
        "correctness_max_abs": max_abs,
        "two_op_median_us": two_med * 1000,
        "two_op_iqr_us": (two_q3 - two_q1) * 1000,
        "two_op_effective_GBps": two_gbps,
        "fused_median_us": fused_med * 1000,
        "fused_iqr_us": (fused_q3 - fused_q1) * 1000,
        "fused_effective_GBps": fused_gbps,
        "fused_pct_roofline": 100 * fused_gbps / ROOFLINE_GBPS,
        "fp16_gemm_median_us": fp16_med * 1000,
        "fp16_gemm_iqr_us": (fp16_q3 - fp16_q1) * 1000,
        "fp16_gemm_effective_GBps": fp16_gbps,
        "speedup_vs_two_op": two_med / fused_med,
        "speedup_vs_fp16_gemm": fp16_med / fused_med,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for shape in SHAPE_NAMES:
        sub = [r for r in rows if r["shape"] == shape]
        xs = [r["batch"] for r in sub]
        axes[0].plot(xs, [r["speedup_vs_two_op"] for r in sub], "o-", label=f"{shape} vs two-op")
        axes[0].plot(xs, [r["speedup_vs_fp16_gemm"] for r in sub], "s--", label=f"{shape} vs fp16 GEMM")
        axes[1].plot(xs, [r["fused_effective_GBps"] for r in sub], "o-", label=shape)

    axes[0].axhline(1.0, color="k", lw=1, alpha=0.4)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Batch")
    axes[0].set_ylabel("Speedup (fused / baseline)")
    axes[0].set_title("Batch-1 fused GEMV is not a GEMM replacement")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].axhline(ROOFLINE_GBPS, color="k", ls="--", lw=1, alpha=0.5, label="~250 GB/s roofline")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Fused effective GB/s (modeled bytes / median)")
    axes[1].set_title("Fused path re-reads weights for each batch row")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--shapes", nargs="+", default=SHAPE_NAMES, choices=sorted(SHAPES))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    rows = []
    for shape in args.shapes:
        for batch in BATCHES:
            row = _run_row(shape, batch, args.warmup, args.iters)
            rows.append(row)
            print(
                f"{shape:11s} B={batch:2d} correctness={row['correctness_max_abs']:.2e} "
                f"fused={row['fused_median_us']:.1f}us "
                f"two_op={row['two_op_median_us']:.1f}us "
                f"speedup_two={row['speedup_vs_two_op']:.2f}x "
                f"speedup_fp16={row['speedup_vs_fp16_gemm']:.2f}x"
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, RESULTS / "sweep.csv")
    _plot(rows, RESULTS / "sweep.png")
    print(f"wrote {RESULTS / 'sweep.csv'}")
    print(f"wrote {RESULTS / 'sweep.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
