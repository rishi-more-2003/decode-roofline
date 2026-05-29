# decode-roofline — task runner
# ---------------------------------------------------------------------------
# Phony targets only; each delegates to a script so the Makefile stays thin.
# On native Windows, run these from Git Bash with `make` (mingw32-make) or
# just copy the underlying command. Targets named per project_spec.md §5.
# ---------------------------------------------------------------------------

PYTHON ?= python
MODEL  ?= Qwen/Qwen2.5-1.5B
MSVC_CMD = MSYS_NO_PATHCONV=1 cmd.exe /c

.PHONY: help env install build test profile-nsys profile-ncu roofline bench sweep reproduce clean

help:
	@echo "decode-roofline targets:"
	@echo "  make env          - verify toolchain (scripts/check_env.py)"
	@echo "  make install      - pip install -r requirements.txt"
	@echo "  make build        - JIT-compile the custom CUDA kernel (Phase 0/2)"
	@echo "  make test         - run correctness tests (MUST pass before bench)"
	@echo "  make profile-nsys - capture an nsys decode timeline (Phase 1)"
	@echo "  make profile-ncu  - capture per-kernel ncu metrics (Phase 1)"
	@echo "  make roofline     - build the 4070 roofline plot (Phase 1)"
	@echo "  make bench        - latency + achieved-bandwidth bench (Phase 2)"
	@echo "  make sweep        - (batch,hidden) regime sweep (Phase 3)"
	@echo "  make reproduce    - one-command end-to-end repro"
	@echo "  make clean        - remove build artifacts & profiler traces"
	@echo "  (override model:  make bench MODEL=meta-llama/Llama-3.2-1B)"

env:
	$(PYTHON) scripts/check_env.py

install:
	$(PYTHON) -m pip install -r requirements.txt

# Phase 0/2: compile the fused dequant+GEMV kernel via cpp_extension JIT.
# On Windows this must run under scripts/with_msvc.bat so nvcc can find cl.exe.
build:
	$(MSVC_CMD) "scripts\\with_msvc.bat $(PYTHON) kernels/load.py"

# Correctness is non-negotiable and gates everything downstream (§7).
test:
	$(MSVC_CMD) "scripts\\with_msvc.bat $(PYTHON) -m pytest kernels\\tests\\test_correctness.py -v"

profile-nsys:
	bash -lc 'export PATH="$$PATH:/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64"; MSYS_NO_PATHCONV=1 nsys profile --stats=true -f true -o bench/results/nsys_decode $(PYTHON) profiling/nsys_decode.py --model $(MODEL) --decode-steps 32 --warmup-steps 8 --no-latency-report'

profile-ncu:
	$(PYTHON) profiling/ncu_kernels.py --model $(MODEL)

roofline:
	$(PYTHON) profiling/roofline.py

# Bench refuses to run unless correctness passes first (enforced in-script).
bench:
	$(MSVC_CMD) "scripts\\with_msvc.bat $(PYTHON) -m pytest kernels\\tests\\test_bench.py -v -s"

sweep:
	$(MSVC_CMD) "scripts\\with_msvc.bat $(PYTHON) bench\\sweep.py"

reproduce:
	bash scripts/reproduce.sh

clean:
	-rm -rf build/ **/__pycache__ .pytest_cache
	-rm -f *.nsys-rep *.ncu-rep *.sqlite
