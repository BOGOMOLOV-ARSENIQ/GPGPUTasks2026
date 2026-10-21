#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // TODO 020 Это добровольное задание за супер-пупер-баллы престижа сверх нормы
    // группа грузит A 64x32 и B 32x64 и считает кусок C 64x64
    __shared__ __align__(32) half shared_a[64][32];
    __shared__ __align__(32) half shared_b[32][64];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        wmma::fill_fragment(acc[i], 0.0f);
    }

    int w_blocks = (w + 63) / 64;

    // 8 варпов сеткой 2x4, каждый считает кусок C 32x16 = 2 плитки A x 1 плитка B
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 4;
    int warp_col = warp_id % 4;

    for (int i = 0; i < ((k + 31) / 32); ++i) {
        int a_row = (blockIdx.x / w_blocks) * 64;
        int a_col = i * 32;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int e = j * 256 + threadIdx.x;
            shared_a[e / 32][e % 32] = __float2half(a[(a_row + e / 32) * k + a_col + e % 32]);
        }

        int b_row = i * 32;
        int b_col = (blockIdx.x % w_blocks) * 64;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int e = j * 256 + threadIdx.x;
            shared_b[e / 64][e % 64] = __float2half(b[(b_row + e / 64) * w + b_col + e % 64]);
        }

        __syncthreads();

        // срез по k шириной 32 = два шага mma по 16
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
#pragma unroll
            for (int ti = 0; ti < 2; ++ti) {
                wmma::load_matrix_sync(a_frag[ti], &shared_a[warp_row * 32 + ti * 16][kk], 32);
            }

            wmma::load_matrix_sync(b_frag, &shared_b[kk][warp_col * 16], 64);
#pragma unroll
            for (int ti = 0; ti < 2; ++ti) {
                wmma::mma_sync(acc[ti], a_frag[ti], b_frag, acc[ti]);
            }
        }

        __syncthreads();
    }

    int c_row = (blockIdx.x / w_blocks) * 64 + warp_row * 32;
    int c_col = (blockIdx.x % w_blocks) * 64 + warp_col * 16;

#pragma unroll
    for (int ti = 0; ti < 2; ++ti) {
        wmma::store_matrix_sync(c + (c_row + ti * 16) * w + c_col, acc[ti], w, wmma::mem_row_major);
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
