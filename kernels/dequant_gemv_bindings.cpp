// kernels/dequant_gemv_bindings.cpp — Phase 2 (STUB)
//
// pybind11 bindings exposing the fused dequant+GEMV kernel to PyTorch via
// torch.utils.cpp_extension. During scaffold this exports a single stub that
// raises, so `import` works and the toolchain can be proven, but no timing can
// be reported without the real implementation + a passing correctness test.

#include <torch/extension.h>

// Phase 2: replace with the real signature, e.g.
//   torch::Tensor dequant_gemv(torch::Tensor x, torch::Tensor w_packed,
//                              torch::Tensor scales, torch::Tensor zeros,
//                              int64_t group_size);
torch::Tensor dequant_gemv_stub() {
    TORCH_CHECK(false,
        "dequant_gemv not implemented yet (Phase 2). "
        "See docs/02_kernel_design.md.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "decode-roofline fused dequant+GEMV (Phase 2 stub)";
    m.def("dequant_gemv_stub", &dequant_gemv_stub,
          "Stub; raises until Phase 2 implements the fused kernel.");
}
