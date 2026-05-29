#!/usr/bin/env bash
# scripts/reproduce.sh — one-command end-to-end repro (STUB).
#
# Regenerates the headline results from a clean checkout. Each phase is gated:
# correctness must pass before any timing is produced (§7).
#
# TODO: wire up as phases land. Kept as a documented stub for now.
set -euo pipefail

echo "[0/5] verify environment"
python scripts/check_env.py

echo "[1/5] Phase 1 — measurement (nsys + ncu + roofline)   [TODO]"
# python profiling/nsys_decode.py --model "${MODEL:-Qwen/Qwen2.5-1.5B}"
# python profiling/ncu_kernels.py --model "${MODEL:-Qwen/Qwen2.5-1.5B}"
# python profiling/roofline.py

echo "[2/5] build the fused kernel                          [TODO]"
# python kernels/load.py

echo "[3/5] correctness (gates everything)                  [TODO]"
# python -m pytest kernels/tests/test_correctness.py -v

echo "[4/5] bench batch-1 + ncu bandwidth                   [TODO]"
# python -m pytest kernels/tests/test_bench.py -v

echo "[5/5] regime sweep + plots                            [TODO]"
# python bench/sweep.py

echo "reproduce.sh: stub complete (phases not yet implemented)"
