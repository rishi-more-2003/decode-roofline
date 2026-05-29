"""kernels/load.py — torch cpp_extension JIT loader.

Compiles kernels/dequant_gemv.cu + bindings via torch.utils.cpp_extension.load
and returns the imported module. Running this file directly (`make build`)
proves the toolchain end-to-end (Phase 0 deliverable): nvcc compiles, the
extension imports.

On the target machine (Windows): the MSVC `cl.exe` must be reachable — run from
a "x64 Native Tools Command Prompt for VS" or ensure cl is on PATH; `ninja`
(pinned in requirements.txt) is used as the build backend.
"""

from __future__ import annotations

from pathlib import Path

_KDIR = Path(__file__).resolve().parent
_MODULE = None


def load_kernel(verbose: bool = True):
    """JIT-compile and import the fused-kernel extension. Cached per process."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load  # lazy import

    _MODULE = load(
        name="decode_roofline_dequant_gemv",
        sources=[
            str(_KDIR / "dequant_gemv_bindings.cpp"),
            str(_KDIR / "dequant_gemv.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
        verbose=verbose,
    )
    return _MODULE


if __name__ == "__main__":
    mod = load_kernel(verbose=True)
    print(f"OK: compiled & imported {mod.__name__}")
    print("exported symbols:", [s for s in dir(mod) if not s.startswith("__")])
