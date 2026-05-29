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

## TODO before any timing run (§3 of the spec)

- [ ] Record laptop model + TGP (power limit). Idle showed `2W / 110W` cap via
      `nvidia-smi` — confirm sustained TGP under load.
- [ ] **Measure actual DRAM bandwidth** (do NOT use the 256 GB/s datasheet
      figure as the roofline ceiling). Use a streaming microbenchmark / the
      Phase 1 ncu peak. Record the measured GB/s here. Spec range: ~256–288.
- [ ] Pin clocks where possible: `nvidia-smi -lgc <min>,<max>` (needs admin;
      WDDM may restrict this on a laptop — note if it fails).
- [ ] Keep machine on AC power; log GPU temperature trend across a profiling run.

| Field | Value |
| --- | --- |
| Laptop model | _TODO_ |
| TGP / power limit | _TODO (idle cap seen: 110 W)_ |
| **Measured DRAM bandwidth (roofline ceiling)** | **_TODO GB/s_** |
| Pinned SM clock | _TODO_ |

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

- [ ] Counter read confirmed (prints a metric value, not `ERR_NVGPUCTRPERM`).

## vLLM on this machine (Phase 1 only)

vLLM does not support native Windows. The measurement phase will run under
**WSL2** (or be skipped in favor of a manual decode-timeline capture). The
custom-kernel work (Phases 0/2/3) needs no vLLM and runs natively on Windows.
