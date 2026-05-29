# decode-roofline

**Profiling & beating LLM decode at the CUDA-kernel level on consumer mobile silicon (RTX 4070 Laptop GPU).**

> **Status:** scaffold complete. Results below are placeholders until the phased
> build lands (see [Roadmap](#roadmap)). Headline numbers will appear here as
> Phases 1–3 complete, each backed by a correctness test and a profiler trace.

---

## Why this project exists

Decode-phase inference on a consumer laptop GPU is aggressively **memory-bound**:
at batch size 1 the matrix multiplies degenerate into matrix–vector products
that are pure weight-streaming from DRAM. The RTX 4070 Laptop GPU's ~256 GB/s of
bandwidth (roughly half the desktop 4070's 504 GB/s) makes this *even more*
pronounced — which is a feature, not a bug. It makes the 4070 Laptop an unusually
**clean** place to study inference cost, because the binding constraint is
unmistakably *bytes moved*, not FLOPs, and that's measurable and beatable.

The deliverable is **not** "I made it faster." It's:

> "I can attribute every microsecond of a decode step to a kernel, explain why
> it's memory-bound via the roofline, and land a fusion win in the regime where
> it matters — then honestly show where that win vanishes."

The headline kernel is a **fused dequant + GEMV** for the batch-1 decode regime.

## Headline result (placeholder)

| Metric | Value |
| --- | --- |
| Model | _TODO (default: Qwen2.5-1.5B)_ |
| Measured DRAM bandwidth (roofline ceiling) | _TODO GB/s_ |
| Decode-step time that is memory-bound | _TODO %_ |
| Fused vs baseline @ batch 1 | _TODO % faster_ |
| Fused achieved bandwidth @ batch 1 | _TODO GB/s (TODO % of peak)_ |
| Regime caveat | _TODO ("…shrinks to ~X% by batch 8, noise by batch 32")_ |

![roofline (placeholder)](bench/results/roofline.png)
![regime sweep (placeholder)](bench/results/sweep.png)

## Hardware / environment

Verified GREEN on 2026-05-29 (`python scripts/check_env.py`). Full detail and
reproduction steps in [`docs/00_environment.md`](docs/00_environment.md).

- **GPU:** RTX 4070 **Laptop** GPU (Ada, AD106), 8 GB GDDR6, CC 8.9, 36 SMs, 128-bit bus, ~256 GB/s nominal (**use the *measured* peak as the roofline ceiling**)
- **Driver** 581.95 · **CUDA toolkit** 13.2 (`nvcc`) · **PyTorch** 2.6.0+cu124 · **Nsight Compute** 2026.1.1 · **Nsight Systems** 2025.6.3 · Python 3.12 · Windows 11
- TGP / thermals matter on a mobile part: results report **median + IQR**, on AC power, with GPU temp trend logged.

## Setup

```bash
# CUDA-enabled torch FIRST (matches this machine), then the rest:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/check_env.py          # expect: ENVIRONMENT: GREEN
```

## Usage (Make targets)

```bash
make env           # verify toolchain
make build         # JIT-compile the custom CUDA kernel
make test          # correctness tests (MUST pass before any timing)
make profile-nsys  # nsys decode timeline (Phase 1)
make profile-ncu   # per-kernel ncu metrics (Phase 1)
make roofline      # build the 4070 roofline plot (Phase 1)
make bench         # latency + achieved bandwidth (Phase 2)
make sweep         # batch/hidden regime sweep (Phase 3)
make reproduce     # one-command end-to-end repro
```

## Repository layout

```
docs/        00 environment · 01 roofline · 02 kernel design · 03 results
profiling/   nsys timeline · ncu metrics · roofline plot · metrics.md
harness/     minimal decode loop · reference GEMV · model loading
kernels/     dequant_gemv.cu + bindings · JIT loader · correctness/bench tests
bench/       regime sweep · results/ (CSVs + plots, checked in)
scripts/     check_env.py · reproduce.sh
```

## Engineering standards (non-negotiable)

- **Correctness before speed, always.** No timing number without a passing
  correctness test in the same run, validated vs the PyTorch/cuBLAS reference.
- **Every speedup claim carries** its shape/regime, achieved bandwidth, roofline
  context, and a repro command.
- Median + IQR (not best-case), CUDA-event timing, fixed seeds, reported warmup
  and measurement iters.
- Plots + CSVs are checked into `bench/results/` so this README renders without
  a GPU.

## Roadmap

- [ ] **Phase 0** — trivial custom op compiles, callable from PyTorch, profilable by `ncu`.
- [ ] **Phase 1** — roofline plot + written memory-bound conclusion ([`docs/01_roofline.md`](docs/01_roofline.md)).
- [ ] **Phase 2** — fused dequant+GEMV: correctness first, then `ncu` bandwidth ([`docs/02_kernel_design.md`](docs/02_kernel_design.md)).
- [ ] **Phase 3** — regime sweep + honest attribution ([`docs/03_results.md`](docs/03_results.md)).

See [`project_spec.md`](project_spec.md) for the full authoritative specification.
