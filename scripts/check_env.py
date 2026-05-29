#!/usr/bin/env python3
"""Environment verification for the decode-roofline project.

Prints and logs everything we need to confirm the toolchain is ready BEFORE
any project code is written:
  - GPU name, driver version
  - CUDA toolkit version (nvcc)
  - PyTorch CUDA availability + the CUDA runtime PyTorch was built against
  - measured VRAM free/total (via nvidia-smi and, if available, torch)
  - whether `ncu` (Nsight Compute) and `nsys` (Nsight Systems) are on PATH

Run:  python scripts/check_env.py
A copy of the output is written to scripts/check_env.log.

This script is intentionally dependency-light: it only *optionally* imports
torch, so it still produces useful output even if torch is not yet installed.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "check_env.log"
_log_lines: list[str] = []


def emit(line: str = "") -> None:
    """Print to stdout and buffer for the log file."""
    print(line)
    _log_lines.append(line)


def run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run a command, returning (returncode, combined_output). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip()
    except FileNotFoundError:
        return 127, f"<not found: {cmd[0]}>"
    except subprocess.TimeoutExpired:
        return 124, f"<timeout after {timeout}s: {' '.join(cmd)}>"
    except Exception as exc:  # noqa: BLE001
        return 1, f"<error running {' '.join(cmd)}: {exc}>"


def header(title: str) -> None:
    emit()
    emit("=" * 70)
    emit(title)
    emit("=" * 70)


def section_host() -> None:
    header("HOST")
    emit(f"timestamp     : {_dt.datetime.now().isoformat(timespec='seconds')}")
    emit(f"platform      : {platform.platform()}")
    emit(f"python        : {sys.version.split()[0]} ({sys.executable})")


def section_nvidia_smi() -> dict:
    header("GPU / DRIVER (nvidia-smi)")
    result: dict = {"ok": False}
    if shutil.which("nvidia-smi") is None:
        emit("nvidia-smi    : NOT on PATH  -> cannot read GPU/driver/VRAM")
        return result

    # Structured query so we can parse VRAM reliably.
    rc, out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        emit(f"nvidia-smi    : failed (rc={rc})")
        emit(out)
        return result

    for idx, row in enumerate(out.splitlines()):
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 6:
            continue
        name, drv, mtot, mfree, mused, temp = parts[:6]
        emit(f"GPU[{idx}] name  : {name}")
        emit(f"driver version: {drv}")
        emit(f"VRAM total    : {mtot} MiB")
        emit(f"VRAM free     : {mfree} MiB")
        emit(f"VRAM used     : {mused} MiB")
        emit(f"GPU temp      : {temp} C")
        result.update(
            ok=True,
            name=name,
            driver=drv,
            mem_total_mib=mtot,
            mem_free_mib=mfree,
        )

        # Sanity check against the spec: this should be the LAPTOP 4070 (8 GB),
        # not the desktop 4070 (12 GB).
        try:
            if float(mtot) > 9000:
                emit(
                    "  ** WARNING: >9 GB VRAM detected. The spec targets the "
                    "RTX 4070 *Laptop* GPU (8 GB). Confirm this is the right device."
                )
        except ValueError:
            pass
    return result


def section_cuda_toolkit() -> None:
    header("CUDA TOOLKIT (nvcc)")
    if shutil.which("nvcc") is None:
        emit("nvcc          : NOT on PATH")
        emit("  -> install the CUDA Toolkit (12.x) or add it to PATH if you")
        emit("     intend to build custom kernels (Phase 0+).")
        return
    rc, out = run(["nvcc", "--version"])
    emit(out if out else f"nvcc --version failed (rc={rc})")


def section_torch() -> None:
    header("PYTORCH")
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        emit(f"torch         : NOT importable ({exc})")
        emit("  -> not fatal for env check; install per requirements.txt later.")
        return

    emit(f"torch version : {torch.__version__}")
    emit(f"built for CUDA: {torch.version.cuda}")
    emit(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        emit("  ** torch.cuda.is_available() is False -> CUDA path is NOT usable.")
        emit("     Check that the torch build matches the installed driver/CUDA.")
        return

    try:
        emit(f"cudnn version : {torch.backends.cudnn.version()}")
    except Exception:  # noqa: BLE001
        pass

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cc = f"{props.major}.{props.minor}"
        free_b, total_b = torch.cuda.mem_get_info(i)
        emit(f"device[{i}]     : {props.name}")
        emit(f"  compute cap : {cc}  (spec expects 8.9 / Ada AD106)")
        emit(f"  SMs         : {props.multi_processor_count}")
        emit(
            f"  VRAM (torch): {free_b / 2**30:.2f} GiB free / "
            f"{total_b / 2**30:.2f} GiB total"
        )


def section_nsight() -> None:
    header("NSIGHT TOOLS (ncu / nsys)")
    for tool, label in (("ncu", "Nsight Compute"), ("nsys", "Nsight Systems")):
        path = shutil.which(tool)
        if path is None:
            emit(f"{tool:<5} ({label}): NOT on PATH")
            continue
        rc, out = run([tool, "--version"])
        first = out.splitlines()[0] if out else "(no version output)"
        emit(f"{tool:<5} ({label}): {path}")
        emit(f"      version : {first}")


def section_summary(smi: dict) -> None:
    header("SUMMARY")
    checks = {
        "nvidia-smi present": shutil.which("nvidia-smi") is not None,
        "GPU detected": smi.get("ok", False),
        "nvcc present": shutil.which("nvcc") is not None,
        "ncu present": shutil.which("ncu") is not None,
        "nsys present": shutil.which("nsys") is not None,
    }
    # torch is optional-but-wanted; probe without hard dependency.
    try:
        import torch  # noqa: PLC0415

        checks["torch + CUDA"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        checks["torch + CUDA"] = False

    for k, v in checks.items():
        emit(f"  [{'OK ' if v else 'XX '}] {k}")

    green = all(checks.values())
    emit()
    emit("ENVIRONMENT: " + ("GREEN (all checks passed)" if green else "NOT yet green"))
    if not green:
        emit("  -> resolve the [XX] items above before Step 2 (scaffold).")


def main() -> int:
    section_host()
    smi = section_nvidia_smi()
    section_cuda_toolkit()
    section_torch()
    section_nsight()
    section_summary(smi)

    try:
        LOG_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
        emit()
        emit(f"(log written to {LOG_PATH})")
    except Exception as exc:  # noqa: BLE001
        emit(f"(could not write log: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
