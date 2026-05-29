// kernels/dequant_gemv_bindings.cpp — Phase 2 bindings.
//
// Exposes the fused INT4 dequant+GEMV kernel to Python via
// torch.utils.cpp_extension.

#include <torch/extension.h>

torch::Tensor dequant_gemv_cuda(torch::Tensor x, torch::Tensor qweight,
                                torch::Tensor scales, int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "decode-roofline fused INT4 dequant+GEMV";
    m.def("dequant_gemv", &dequant_gemv_cuda,
          "Fused symmetric-INT4 dequant + GEMV (batch 1)");
}
