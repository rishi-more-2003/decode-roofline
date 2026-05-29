#!/usr/bin/env python3
"""Phase 0 — CUDA ramp: prove the toolchain end-to-end.

Compiles a trivial `saxpy` (y = a*x + y) CUDA kernel via
torch.utils.cpp_extension, calls it from PyTorch, and validates it against the
torch reference in the SAME run (correctness before anything else, per §7).

This is also the artifact we profile with `ncu` to confirm GPU performance
counters are readable on this machine:

    ncu --metrics dram__bytes.sum,gpu__time_duration.sum \\
        python scripts/phase0_saxpy.py --profile

(`--profile` runs a single fixed launch so ncu has exactly one kernel to grab.)

Exit code is non-zero if the build fails or correctness fails, so it can gate
CI / reproduce.sh.

This file is the optional Phase-0 ramp; it can be deleted once Phase 2's real
kernel proves the same path.
"""

from __future__ import annotations

import argparse
import sys

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void saxpy_kernel(int n, float a, const float* __restrict__ x,
                             const float* __restrict__ y, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

torch::Tensor saxpy(double a, torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                y.scalar_type() == torch::kFloat32, "saxpy expects float32");
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have equal numel");
    auto xc = x.contiguous();
    auto yc = y.contiguous();
    auto out = torch::empty_like(xc);
    const int n = static_cast<int>(xc.numel());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    saxpy_kernel<<<blocks, threads>>>(
        n, static_cast<float>(a),
        xc.data_ptr<float>(), yc.data_ptr<float>(), out.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "saxpy kernel launch failed: ",
                cudaGetErrorString(err));
    return out;
}
"""

CPP_SRC = "torch::Tensor saxpy(double a, torch::Tensor x, torch::Tensor y);"


def build(verbose: bool = True):
    """JIT-compile the saxpy extension and return the module."""
    from torch.utils.cpp_extension import load_inline  # lazy import

    return load_inline(
        name="phase0_saxpy",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["saxpy"],
        # CUDA 13.x CCCL headers require MSVC's conforming preprocessor.
        extra_cflags=["/Zc:preprocessor"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_89,code=sm_89",
            "-Xcompiler",
            "/Zc:preprocessor",
        ],
        verbose=verbose,
    )


def check(n: int = 1 << 20, seed: int = 0) -> bool:
    """Build, run on CUDA, and compare against the torch reference."""
    import torch

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return False

    mod = build()
    torch.manual_seed(seed)
    a = 2.5
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)

    out = mod.saxpy(a, x, y)
    ref = a * x + y
    torch.cuda.synchronize()

    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-6)
    max_abs = (out - ref).abs().max().item()
    print(f"saxpy n={n} a={a}  max_abs_err={max_abs:.3e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def profile_launch(n: int = 1 << 24) -> None:
    """One fixed kernel launch for ncu to profile (no correctness noise)."""
    import torch

    mod = build(verbose=False)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    _ = mod.saxpy(2.0, x, y)  # warm up / build cache
    torch.cuda.synchronize()
    out = mod.saxpy(2.0, x, y)  # the launch ncu captures
    torch.cuda.synchronize()
    print(f"profiled saxpy launch: n={n}, out[0]={out[0].item():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run a single fixed launch for ncu (skip correctness sweep)",
    )
    args = parser.parse_args()

    if args.profile:
        profile_launch()
        return 0

    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
