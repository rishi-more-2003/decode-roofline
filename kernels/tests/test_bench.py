"""Phase 2 benchmark, correctness-gated.

HARD RULE (§7): correctness runs first in this same test. Only after the fused
kernel matches the FP32 oracle do we report timings. Results are median + IQR
over CUDA-event timings and are written to bench/results/bench.csv.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest
import torch

from harness.load_model import SHAPES, make_test_weight, quantize_int4_groupwise
from harness.reference_gemv import baseline_two_op, reference_fp32
from kernels.load import load_kernel


GROUP_SIZE = 128
RESULTS = Path(__file__).resolve().parents[2] / "bench" / "results"


def _iqr(xs: list[float]) -> tuple[float, float, float]:
    s = sorted(xs)
    return statistics.median(s), s[len(s) // 4], s[(3 * len(s)) // 4]


def _time_cuda(fn, warmup: int = 10, iters: int = 50) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _traffic_bytes(n: int, k: int, group_size: int) -> tuple[int, int]:
    packed = n * k // 2
    scales = n * (k // group_size) * 2
    x = k * 2
    y = n * 2
    fused = packed + scales + x + y
    separate = packed + scales + (2 * n * k) + (2 * n * k) + x + y
    return fused, separate


@pytest.mark.parametrize("shape_name", ["mlp_gate_up"])
def test_bench_batch1(shape_name: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for benchmarking")

    n, k = SHAPES[shape_name]
    W = make_test_weight(n, k, seed=2026)
    qweight, scales = quantize_int4_groupwise(W, GROUP_SIZE)
    qweight = qweight.to("cuda")
    scales = scales.to("cuda")
    x = torch.randn(k, device="cuda", dtype=torch.float16)
    mod = load_kernel(verbose=False)

    # Correctness gate in the same run, before any timing.
    fused = mod.dequant_gemv(x, qweight, scales, GROUP_SIZE)
    ref = reference_fp32(qweight, scales, x, GROUP_SIZE).to(torch.float16)
    torch.cuda.synchronize()
    max_abs = (fused - ref).abs().max().item()
    assert torch.allclose(fused, ref, rtol=2e-3, atol=2e-3), max_abs

    fused_times = _time_cuda(lambda: mod.dequant_gemv(x, qweight, scales, GROUP_SIZE))
    baseline_times = _time_cuda(lambda: baseline_two_op(qweight, scales, x, GROUP_SIZE))

    fused_med, fused_q1, fused_q3 = _iqr(fused_times)
    base_med, base_q1, base_q3 = _iqr(baseline_times)
    fused_bytes, separate_bytes = _traffic_bytes(n, k, GROUP_SIZE)
    fused_gbps = fused_bytes / (fused_med / 1e3) / 1e9
    base_gbps = separate_bytes / (base_med / 1e3) / 1e9
    speedup = base_med / fused_med

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "bench.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "N",
                "K",
                "group_size",
                "correctness_max_abs",
                "baseline_median_us",
                "baseline_iqr_us",
                "baseline_traffic_MB",
                "baseline_effective_GBps",
                "fused_median_us",
                "fused_iqr_us",
                "fused_traffic_MB",
                "fused_effective_GBps",
                "speedup",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "shape": shape_name,
                "N": n,
                "K": k,
                "group_size": GROUP_SIZE,
                "correctness_max_abs": f"{max_abs:.6g}",
                "baseline_median_us": f"{base_med * 1000:.3f}",
                "baseline_iqr_us": f"{(base_q3 - base_q1) * 1000:.3f}",
                "baseline_traffic_MB": f"{separate_bytes / 1e6:.3f}",
                "baseline_effective_GBps": f"{base_gbps:.3f}",
                "fused_median_us": f"{fused_med * 1000:.3f}",
                "fused_iqr_us": f"{(fused_q3 - fused_q1) * 1000:.3f}",
                "fused_traffic_MB": f"{fused_bytes / 1e6:.3f}",
                "fused_effective_GBps": f"{fused_gbps:.3f}",
                "speedup": f"{speedup:.3f}",
            }
        )

    print(
        f"\n{shape_name}: correctness max_abs={max_abs:.3e}; "
        f"baseline {base_med * 1000:.1f} us (IQR {(base_q3-base_q1)*1000:.1f}), "
        f"fused {fused_med * 1000:.1f} us (IQR {(fused_q3-fused_q1)*1000:.1f}), "
        f"speedup {speedup:.2f}x, fused effective {fused_gbps:.1f} GB/s"
    )

