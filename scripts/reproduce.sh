#!/usr/bin/env bash
# scripts/reproduce.sh — one-command end-to-end repro.
#
# Regenerates the headline results from a clean checkout. Correctness gates all
# timings (§7): the Phase 2 correctness suite runs before bench/sweep, and the
# fused-kernel ncu driver correctness-checks immediately before the profiled
# launch.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
PYTHON="${PYTHON:-python}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

# Native Windows / Git Bash: make nsys visible even if ~/.bashrc was not sourced.
NSYS_DIR="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64"
if [[ -d "$NSYS_DIR" ]]; then
  export PATH="$PATH:$NSYS_DIR"
fi

run_msvc() {
  # MSYS_NO_PATHCONV prevents Git Bash from rewriting backslash-heavy cmd args.
  MSYS_NO_PATHCONV=1 cmd.exe /c "scripts\\with_msvc.bat $*"
}

echo "[0/9] verify environment"
"$PYTHON" scripts/check_env.py

echo "[1/9] Phase 1: measure DRAM bandwidth ceiling"
"$PYTHON" profiling/measure_bandwidth.py --mib 256 --iters 50

echo "[2/9] Phase 1: native decode latency sanity"
"$PYTHON" profiling/nsys_decode.py --model "$MODEL" --decode-steps 64 --warmup-steps 16

echo "[3/9] Phase 1: nsys timeline"
MSYS_NO_PATHCONV=1 nsys profile --stats=true -f true -o bench/results/nsys_decode \
  "$PYTHON" profiling/nsys_decode.py --model "$MODEL" --decode-steps 32 --warmup-steps 8 --no-latency-report

echo "[4/9] Phase 1: ncu GEMV metrics + roofline"
"$PYTHON" profiling/ncu_kernels.py --model "$MODEL"
"$PYTHON" profiling/roofline.py

echo "[5/9] Phase 2: build fused kernel"
run_msvc "$PYTHON kernels/load.py"

echo "[6/9] Phase 2: correctness gate"
run_msvc "$PYTHON -m pytest kernels\\tests\\test_correctness.py -v"

echo "[7/9] Phase 2: correctness-gated benchmark"
run_msvc "$PYTHON -m pytest kernels\\tests\\test_bench.py -v -s"

echo "[8/9] Phase 2: ncu fused-kernel bandwidth"
run_msvc "ncu --kernel-name regex:dequant_gemv_kernel --launch-count 1 --metrics dram__bytes.sum,gpu__time_duration.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active --csv $PYTHON bench\\profile_dequant_gemv.py > bench\\results\\ncu_fused.csv"

echo "[9/9] Phase 3: regime sweep + plots"
run_msvc "$PYTHON bench\\sweep.py --warmup 5 --iters 20"

echo "reproduce.sh: complete"
