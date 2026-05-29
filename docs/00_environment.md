# 00 — Environment

Exact hardware/software the results were produced on, and how to reproduce it.

## Measured machine (verified 2026-05-29, `scripts/check_env.py` → GREEN)

| Item | Value |
| --- | --- |
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.9 |
| GPU | NVIDIA GeForce RTX 4070 **Laptop** GPU (Ada, AD106) |
| Compute capability | 8.9 |
| SMs | 36 |
| VRAM | 8188 MiB total (~8.0 GiB; ~6.9 GiB free at idle) |
| Driver | 581.95 (reports CUDA 13.0 runtime) |
| CUDA toolkit (`nvcc`) | 13.2 (V13.2.78) — used to compile custom kernels |
| PyTorch | 2.6.0+cu124 (ships its own CUDA 12.4 runtime; `torch.cuda.is_available()` = True) |
| cuDNN | 9.1.0 |
| Nsight Compute (`ncu`) | 2026.1.1 |
| Nsight Systems (`nsys`) | 2025.6.3 |

> Note the two CUDA versions are expected and fine: the **driver** (CUDA 13.0)
> and the standalone **toolkit** (13.2) compile our kernels, while **PyTorch**
> carries its own bundled CUDA 12.4 runtime. SM 8.9 is supported by all of them.

## Run conditions

The measurements were taken on AC power on the native Windows CUDA stack. Clock
pinning was not used because WDDM/laptop controls can restrict `nvidia-smi -lgc`;
results therefore report **median + IQR** rather than best-case timings.

| Field | Value |
| --- | --- |
| Laptop model | RTX 4070 Laptop GPU system (exact chassis not recorded) |
| TGP / power limit | `nvidia-smi` reported 110 W cap at idle |
| **Measured DRAM bandwidth (roofline ceiling)** | **~250 GB/s achievable** (theoretical ~259) |
| Pinned SM clock | Not pinned (native Windows/WDDM run) |

### Measured bandwidth detail (2026-05-29)

| Method | Achieved |
| --- | --- |
| torch device-to-device copy (256 MiB) | median 227 GB/s, max 229 |
| torch triad `a*x+y` (256 MiB) | median 232 GB/s, max 235 |
| custom saxpy via `ncu dram__bytes.sum` (Phase 0) | ~244 GB/s |
| **cuBLAS decode GEMV via `ncu dram__bytes.sum` (Phase 1)** | **~247 GB/s** (best observed) |

**Roofline ceiling used: 250 GB/s** (achievable peak, rounded from the best
measured ~247 GB/s on the actual decode GEMV via the ncu DRAM byte counter).
ncu's internal hardware-theoretical peak is **~259 GB/s** (247 / 0.953 reported
%-of-peak), i.e. the datasheet ~256 GB/s. We quote against the *achievable*
250 GB/s and report ncu's %-of-peak (vs ~259) alongside. Raw numbers in
`bench/results/bandwidth.csv` and `bench/results/ncu_kernels.csv`.

## Reproducing the environment

```bash
# 1. CUDA-enabled torch first (matches this machine)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 2. everything else
pip install -r requirements.txt
# 3. verify
python scripts/check_env.py        # expect: ENVIRONMENT: GREEN
```

## Nsight Compute counter permission (most common blocker)

`ncu` needs OS permission to read GPU performance counters or every metric
comes back empty (`ERR_NVGPUCTRPERM`). On Windows this is **not** sudo — it is:

> NVIDIA Control Panel → Developer → **Manage GPU Performance Counters** →
> "Allow access to the GPU performance counters to all users" → Apply → reboot.

Ref: <https://developer.nvidia.com/ERR_NVGPUCTRPERM>

Verify (once a CUDA program exists, e.g. in Phase 0):

```bash
ncu --metrics dram__bytes.sum python -c "import torch; x=torch.randn(4096,4096,device='cuda'); (x@x).sum().item()"
```

- [x] **Counter read confirmed (2026-05-29, Phase 0).** `ncu` profiled the
      `saxpy_kernel` and returned real metric values (no `ERR_NVGPUCTRPERM`):
      `dram__bytes.sum = 186.17 MB`, `gpu__time_duration.sum = 763.65 us`
      (≈244 GB/s achieved on a 16.7M-element streaming saxpy — ~95% of the
      256 GB/s nominal). GPU performance counters are readable on this machine.

## Building custom kernels on this machine (Windows recipe)

`torch.utils.cpp_extension` needs the MSVC host compiler (`cl.exe`) and a few
Windows-specific flags. Established in Phase 0:

- **MSVC:** Visual Studio 2022 **Build Tools** (toolset 14.43) at
  `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`. `cl.exe` is
  not on PATH by default — activate it via `scripts\with_msvc.bat`, which calls
  `vcvars64.bat` then runs the given command.
- **`/Zc:preprocessor` is required.** CUDA 13.x CCCL headers reject MSVC's
  traditional preprocessor (`fatal error C1189`). Passed as `extra_cflags` and
  `-Xcompiler /Zc:preprocessor` in `kernels/load.py` and `scripts/phase0_saxpy.py`.
- **Pin the arch:** `export TORCH_CUDA_ARCH_LIST=8.9` (Ada) to avoid building
  for all archs.
- **From Git Bash**, disable MSYS path mangling and use `cmd /c`:

```bash
export TORCH_CUDA_ARCH_LIST=8.9
MSYS_NO_PATHCONV=1 cmd.exe /c "scripts\with_msvc.bat python scripts\phase0_saxpy.py"
# profile (also via the wrapper, since torch regenerates build.ninja => needs cl):
MSYS_NO_PATHCONV=1 cmd.exe /c "scripts\with_msvc.bat ncu --kernel-name regex:saxpy --launch-count 1 --metrics dram__bytes.sum,gpu__time_duration.sum python scripts\phase0_saxpy.py --profile"
```

> Benign note: under `ncu`, a relaunched Python child can emit a one-off
> `LookupError: unknown encoding: utf-8-sig` at shutdown; it does not affect the
> captured metrics (the profile completes and exits 0).

## vLLM note

The original project spec suggested vLLM for the Phase 1 timeline. vLLM does not
support native Windows, so this artifact uses a native Hugging Face
`transformers` batch-1 decode loop for profiling. That keeps Phase 1 measurement
and Phases 2/3 custom-kernel work on the same CUDA stack. The memory-bound
roofline conclusion does not depend on PagedAttention.
