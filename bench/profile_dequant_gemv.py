"""Small correctness-gated launch driver for profiling the fused kernel with ncu.

This is intentionally separate from the pytest benchmark so Nsight Compute can
profile one clean kernel name:

    MSYS_NO_PATHCONV=1 cmd.exe /c "scripts\\with_msvc.bat ncu --kernel-name regex:dequant_gemv_kernel --launch-count 1 --metrics dram__bytes.sum,gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active python bench\\profile_dequant_gemv.py"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.load_model import SHAPES, make_test_weight, quantize_int4_groupwise
from harness.reference_gemv import reference_fp32
from kernels.load import load_kernel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="mlp_gate_up", choices=sorted(SHAPES))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    n, k = SHAPES[args.shape]
    W = make_test_weight(n, k, seed=args.seed)
    qweight, scales = quantize_int4_groupwise(W, args.group_size)
    qweight = qweight.to("cuda")
    scales = scales.to("cuda")
    x = torch.randn(k, device="cuda", dtype=torch.float16)
    mod = load_kernel(verbose=False)

    # Correctness gate before the profiled launch.
    fused = mod.dequant_gemv(x, qweight, scales, args.group_size)
    ref = reference_fp32(qweight, scales, x, args.group_size).to(torch.float16)
    torch.cuda.synchronize()
    max_abs = (fused - ref).abs().max().item()
    if not torch.allclose(fused, ref, rtol=2e-3, atol=2e-3):
        print(f"FAIL correctness max_abs={max_abs:.6g}")
        return 2
    print(f"PASS correctness max_abs={max_abs:.6g}")

    for _ in range(args.warmup):
        mod.dequant_gemv(x, qweight, scales, args.group_size)
    torch.cuda.synchronize()

    y = mod.dequant_gemv(x, qweight, scales, args.group_size)
    torch.cuda.synchronize()
    print(f"profiled fused {args.shape} N={n} K={k}, y0={y[0].item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
