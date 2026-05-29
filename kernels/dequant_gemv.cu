// kernels/dequant_gemv.cu — Phase 2 (STUB)
//
// The headline fused dequant + GEMV CUDA kernel for batch-1 decode.
//
// Design intent (see docs/02_kernel_design.md):
//   - at batch 1, GEMM -> GEMV: pure weight streaming, memory-bound;
//   - fuse dequantization into the load path to eliminate a full DRAM
//     round-trip of the dequantized FP16 weights;
//   - Ada (SM 8.9) tuning: coalesced/vectorized loads of packed weights,
//     block/tile sizes chosen against L2 and the 128-bit bus.
//
// TODO(Phase 2): implement. Correctness vs harness/reference_gemv.py FIRST,
// then ncu bandwidth. Until implemented, the binding (see bindings .cpp) is the
// only symbol exported; calling it raises from Python.

#include <cuda_runtime.h>

// Placeholder so the translation unit compiles cleanly during scaffold.
// Real entry point (e.g. launch_dequant_gemv(...)) is added in Phase 2.
extern "C" int decode_roofline_dequant_gemv_stub() { return 0; }
