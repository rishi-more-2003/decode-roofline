// kernels/dequant_gemv.cu — Phase 2: fused INT4 dequant + GEMV (batch 1).
//
// y[n] = sum_k W_hat[n,k] * x[k],  with W_hat dequantized on the fly from
// symmetric INT4 group quantization (group size G):
//     w_hat = (nibble - 8) * scale[n, k/G]
//
// Layout (see docs/02_kernel_design.md):
//   x        : FP16  [K]            (batch 1)
//   qweight  : int32 [N, K/8]       (8 nibbles per int32, nibble t at bits 4t)
//   scales   : FP16  [N, K/G]
//   y        : FP16  [N]
//
// One warp per output row; lanes stride the packed row by 32 int32s for fully
// coalesced 128-byte loads. Accumulate in FP32, warp-reduce. x stays hot in L2
// (reused by every row in the block).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void dequant_gemv_kernel(
        const __half* __restrict__ x,
        const uint32_t* __restrict__ qw,
        const __half* __restrict__ scales,
        __half* __restrict__ y,
        int N, int K, int group_size) {
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= N) return;

    const int Kp = K >> 3;                       // int32 per row (K/8)
    const int groups_per_row = K / group_size;
    const int u32_per_group = group_size >> 3;   // G/8
    const uint32_t* qrow = qw + (size_t)row * Kp;
    const __half* srow = scales + (size_t)row * groups_per_row;

    float acc = 0.f;
    for (int j = lane; j < Kp; j += 32) {
        uint32_t packed = qrow[j];
        float scale = __half2float(srow[j / u32_per_group]);
        const int base_k = j << 3;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int nib = (packed >> (4 * t)) & 0xF;
            float w = static_cast<float>(nib - 8) * scale;
            acc += w * __half2float(x[base_k + t]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = __float2half(acc);
}

torch::Tensor dequant_gemv_cuda(torch::Tensor x, torch::Tensor qweight,
                                torch::Tensor scales, int64_t group_size) {
    TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && scales.is_cuda(),
                "x, qweight, scales must be CUDA tensors");
    TORCH_CHECK(x.dim() == 1, "x must be 1D [K] (batch 1)");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(scales.scalar_type() == torch::kHalf, "scales must be float16");

    const int K = static_cast<int>(x.size(0));
    const int N = static_cast<int>(qweight.size(0));
    TORCH_CHECK(qweight.size(1) * 8 == K, "qweight [N,K/8] mismatch with x [K]");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 8 == 0, "group_size must be divisible by 8");

    auto xc = x.contiguous();
    auto qc = qweight.contiguous();
    auto sc = scales.contiguous();
    auto y = torch::empty({N}, x.options());

    const int warps_per_block = 4;
    const int threads = warps_per_block * 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;
    dequant_gemv_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qc.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K, static_cast<int>(group_size));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dequant_gemv launch failed: ",
                cudaGetErrorString(err));
    return y;
}
