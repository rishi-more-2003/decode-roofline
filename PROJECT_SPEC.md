# Project Spec: Decode-Phase Kernel Profiling & Optimization on Consumer Silicon

> **For the agent (Cursor):** This is a from-scratch systems/CUDA project. Read this whole spec before writing code. Build incrementally, validate correctness at every phase before optimizing, and never report a speedup without a correctness check and a profiler trace backing it. When in doubt, prefer a smaller, fully-measured result over a larger unverified one. Ask me before adding heavyweight dependencies.

---

## 1. One-line summary

Profile where time actually goes inside LLM decode at the **CUDA kernel level** on an **RTX 4070 Laptop GPU (Ada, AD106, SM 8.9, ~256 GB/s, 8 GB)**, prove decode is memory-bandwidth-bound against the hardware roofline, then write a custom fused **dequant + GEMV** kernel that beats the baseline in the batch-1 decode regime — with honest, regime-aware attribution of where the win holds and where it vanishes.

## 2. Why this project exists (keep this framing in the README)

Decode-phase inference on a consumer laptop GPU is aggressively memory-bound: at batch size 1 the matrix multiplies degenerate into matrix-vector products that are pure weight-streaming from DRAM. The RTX 4070 Laptop GPU's ~256 GB/s of bandwidth (roughly half the desktop 4070's 504 GB/s) makes this *even more* pronounced, which is a feature, not a bug — it makes the 4070 Laptop an unusually *clean* place to study inference cost, because the binding constraint is unmistakably bytes moved, not FLOPs, and that's measurable and beatable. The deliverable is not "I made it faster"; it's "I can attribute every microsecond of a decode step to a kernel, explain why it's memory-bound via the roofline, and land a fusion win in the regime where it matters." The mobile-part framing — "kernel optimization where memory bandwidth is the hard binding constraint" — is a sharper, more reproducible study than a desktop/datacenter throughput benchmark.

## 3. Hardware / environment assumptions

- **GPU:** single **RTX 4070 Laptop GPU**, **8 GB GDDR6**, Ada Lovelace (**AD106 die**, distinct from the desktop 4070's AD104), compute capability 8.9, **4,608 CUDA cores / 36 SMs**, **128-bit bus, ~256 GB/s peak DRAM bandwidth** (some higher-TGP/faster-memory configs reach ~288 GB/s — measure and record the actual figure for your laptop). No NVLink, single device.
- **TGP / thermals matter on a mobile part.** The 4070 Laptop ships at widely varying power limits (~35–140W) depending on chassis, and sustained profiling loads will throttle. Record the laptop model and TGP, pin clocks where possible (`nvidia-smi -lgc`), keep the machine on AC power, and report **median + IQR over enough runs to expose throttling** rather than a single best-case number. Note the GPU temperature trend across a profiling run in the writeup.
- **VRAM is the real constraint: 8 GB, shared with the OS/display.** A 7–8B model does not fit comfortably once KV cache, activations, and CUDA/PyTorch overhead are included. **Default to a 1.5–3B model**: FP16 weights for the baseline GEMV, a 4-bit (AWQ/GPTQ) variant for the fused dequant kernel. Kernel behavior is the object of study; model quality is irrelevant. Default target for the custom kernel: **Qwen2.5-1.5B / Qwen2.5-3B or Llama-3.2-1B / Llama-3.2-3B**. Confirm headroom with `nvidia-smi` before each run.
- **Toolchain:** CUDA 12.x, PyTorch with CUDA, `torch.utils.cpp_extension` (load/JIT) for custom ops, Nsight Compute (`ncu`) and Nsight Systems (`nsys`), vLLM (recent release) for the measurement phase only.
- Pin all versions in `requirements.txt` / `environment.yml` and record `nvidia-smi`, driver, CUDA version, laptop model, and TGP in the README.

## 4. Scope boundaries (read carefully)

- **Custom-kernel work happens in a standalone minimal decode harness** that we control — NOT inside vLLM's PagedAttention plugin path. Plugging custom CUDA back into vLLM is fiddly and not the point. We use vLLM only to generate the *measurement* story (Phase 1).
- The headline kernel is **fused dequant + GEMV (batch=1 decode)**. RMSNorm+residual+quant fusion and a small-batch attention-tile study are **stretch goals**, attempted only after the headline kernel is correct and benchmarked.
- Single GPU, single precision path at a time. No multi-GPU, no distributed, no training.
- Correctness is non-negotiable: every custom kernel is validated against a reference (PyTorch / cuBLAS) within tolerance before any timing is reported.

## 5. Repository structure

```
.
├── README.md                  # narrative: the framing in §2, headline results, roofline plots
├── requirements.txt
├── environment.yml
├── Makefile                   # build, test, profile, bench targets
├── docs/
│   ├── 00_environment.md       # exact HW/SW, how to reproduce
│   ├── 01_roofline.md          # the 4070 roofline + per-kernel placement
│   ├── 02_kernel_design.md     # fused dequant+GEMV design notes, memory-traffic math
│   └── 03_results.md           # regime sweep tables, attribution, honest limits
├── profiling/
│   ├── nsys_decode.py          # timeline capture of a vLLM decode loop
│   ├── ncu_kernels.py          # per-kernel ncu metric capture + parsing
│   ├── roofline.py             # build roofline plot from ncu metrics
│   └── metrics.md              # which ncu metrics we collect and why
├── harness/
│   ├── decode_harness.py       # minimal single-token decode loop (no vLLM)
│   ├── reference_gemv.py       # PyTorch/cuBLAS reference for correctness
│   └── load_model.py           # load weights, build quantized weight tensors
├── kernels/
│   ├── dequant_gemv.cu         # the headline fused kernel
│   ├── dequant_gemv_bindings.cpp
│   ├── load.py                 # torch cpp_extension JIT loader
│   └── tests/
│       ├── test_correctness.py # vs reference, multiple shapes & seeds
│       └── test_bench.py       # latency + achieved bandwidth sweep
├── bench/
│   ├── sweep.py                # (batch, hidden, seqlen) sweep driver
│   └── results/                # CSVs + plots, checked in
└── scripts/
    └── reproduce.sh            # one-command end-to-end repro
```

## 6. Phased build plan

### Phase 0 — CUDA ramp (OPTIONAL — delete if you've written CUDA before)
Goal: get a trivial custom op compiling and callable from PyTorch so the toolchain is proven before real work.
- Build a `vector_add` or `saxpy` kernel via `torch.utils.cpp_extension.load`, call it from Python, validate against `torch`, and confirm `ncu` can profile it. Deliverable: the toolchain works end-to-end. Do not proceed until this is green.

### Phase 1 — Measurement (the foundation)
Goal: prove, with profiler data, that decode is memory-bound on the 4070.
- `profiling/nsys_decode.py`: run a vLLM decode loop on the target model, capture an `nsys` timeline, identify the dominant kernels by total time (expect: attention/PagedAttention, the GEMMs, RMSNorm, sampling).
- `profiling/ncu_kernels.py`: for each dominant kernel, capture `ncu` metrics:
  - achieved DRAM throughput (GB/s) vs the ~256 GB/s peak (use your measured peak),
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  - memory vs compute issue-stall reasons,
  - L2 hit rate,
  - achieved occupancy.
- `profiling/roofline.py` + `docs/01_roofline.md`: build a **per-kernel roofline plot** for the 4070. Headline artifact: decode kernels pinned against the memory ceiling. Write the prose conclusion: which kernels are memory-bound, by how much, and why.

**Phase 1 exit criteria:** a roofline plot + a written claim like "decode-step time is X% memory-bound; the dominant memory-bound kernel is the weight GEMV/GEMM at batch 1."

### Phase 2 — The custom kernel (the headline)
Goal: write a fused dequant + GEMV CUDA kernel and prove it beats the baseline at batch 1.
- `harness/decode_harness.py`: a minimal single-token decode loop using the target model's weights, no vLLM. This is where the kernel plugs in.
- `harness/reference_gemv.py`: PyTorch/cuBLAS reference (dequant-then-GEMV as two ops) — this is both the correctness oracle and the baseline to beat.
- `kernels/dequant_gemv.cu`: the fused kernel. Design intent (write the reasoning in `docs/02_kernel_design.md`):
  - at batch 1, GEMM → GEMV: pure weight streaming, memory-bound,
  - fusing dequantization into the load path eliminates a full DRAM round-trip of the dequantized weights,
  - tune for Ada: coalesced loads of the quantized weights, block/tile sizes chosen against the L2 and register file, vectorized loads where possible,
  - do the **memory-traffic arithmetic** explicitly: bytes read by (separate dequant + GEMV) vs (fused) and predict the speedup from the bandwidth roofline *before* measuring.
- `kernels/tests/test_correctness.py`: validate vs reference across multiple shapes and seeds, within tolerance. **Must pass before any benchmarking.**

**Phase 2 exit criteria:** fused kernel is correct, and `ncu` confirms its achieved bandwidth is closer to the ~256 GB/s roofline than the baseline's.

### Phase 3 — Regime sweep & honest attribution (what makes it senior)
Goal: show where the win holds and where it disappears.
- `bench/sweep.py`: sweep batch size {1, 2, 4, 8, 16, 32} and a couple of hidden sizes; measure latency and achieved bandwidth for baseline vs fused.
- `docs/03_results.md`: the senior move — report the speedup at batch 1, then show it shrink as the kernel becomes compute-bound, e.g. "11% at batch 1, ~3% by batch 8, noise by batch 32, because [roofline reason]." Tie back to the Phase 1 roofline.
- Update the README with the headline number, the regime caveat, and the roofline plot.

### Stretch goals (only after Phase 3 is solid)
- Fused RMSNorm + residual + quant kernel (kills two DRAM round-trips).
- A PagedAttention-style attention tile study for the small-batch / fragmented-KV regime, Ada-tuned block sizes.
- FP8 path exploration (Ada has limited FP8 — note the caveats).

## 7. Engineering standards

- **Correctness before speed, always.** No timing numbers without a passing correctness test in the same run.
- **Every speedup claim carries:** the shape/regime, the achieved bandwidth, the roofline context, and a repro command.
- Determinism where feasible: fixed seeds, report warmup iters and measurement iters, report median + IQR not just mean, use CUDA events for timing, sync correctly.
- Keep CSV outputs and plots checked into `bench/results/` so the README renders without a GPU.
- `scripts/reproduce.sh` regenerates the headline results end-to-end.

## 8. Definition of done

- [ ] Phase 1 roofline plot + written memory-bound conclusion for the 4070.
- [ ] Correct, tested fused dequant+GEMV kernel beating the two-op baseline at batch 1, confirmed by `ncu` bandwidth.
- [ ] Regime sweep showing where the win holds and where it vanishes, with the roofline explanation.
- [ ] README that tells the §2 story, leads with the headline result and the honest caveat, and renders its plots without a GPU.
- [ ] One-command reproduction.

## 9. Out of scope (do not build)

Multi-GPU / distributed; training or fine-tuning; production serving; a full vLLM kernel plugin; model-quality evaluation; anything that doesn't serve the "attribute and beat decode cost on a 4070" narrative.