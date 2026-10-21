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

// float -> half, каждый поток переводит 4 соседних числа (одна загрузка float4, одна запись 8 байт)
__global__ void convert_float_to_half(const float* src, half* dst, unsigned int n4)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    float4 v = reinterpret_cast<const float4*>(src)[i];
    reinterpret_cast<half2*>(dst)[2 * i] = __floats2half2_rn(v.x, v.y);
    reinterpret_cast<half2*>(dst)[2 * i + 1] = __floats2half2_rn(v.z, v.w);
}

// группа 256 потоков (8 варпов) считает кусок C 128x128, хотим 2 группы на SM -> не больше 128 регистров на поток
__global__ void __launch_bounds__(256, 2) matrix_multiply_wmma(
                       const half* a, // rows=h x cols=k
                       const half* b, // rows=k x cols=w
                            float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // TODO 020 Это добровольное задание за супер-пупер-баллы престижа сверх нормы
    // два буфера: пока считаем по одному, следующий срез по k пишется в другой. 2 * (4 + 4) КБ = 16 КБ на группу
    __shared__ __align__(32) half shared_a[2][128 * 16]; // башенка A 128x16 построчно
    __shared__ __align__(32) half shared_b[2][16 * 128]; // полоса B 16x128 построчно

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        wmma::fill_fragment(acc[i], 0.0f);
    }

    int h_blocks = h / 128;
    int w_blocks = w / 128;

    // порядок групп для L2: идём полосами по 8 рядов кусков и внутри полосы по столбцам,
    // тогда одновременно работающие группы читают мало разных полос A и B и те чаще лежат в L2
    const int GROUP_ROWS = 8;
    int groups_in_band = GROUP_ROWS * w_blocks;
    int first_row = (blockIdx.x / groups_in_band) * GROUP_ROWS;
    int rows_in_band = min(h_blocks - first_row, GROUP_ROWS);
    int in_band = blockIdx.x % groups_in_band;
    int block_row = first_row + in_band % rows_in_band;
    int block_col = in_band / rows_in_band;

    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    // каждый поток за шаг переносит 8 соседних half из A и 8 из B - ровно по одной 16-байтной загрузке
    int a_r = threadIdx.x / 2;          // строка в башенке A, 0..127
    int a_c = (threadIdx.x % 2) * 8;    // столбец в башенке A, 0 или 8
    int b_r = threadIdx.x / 16;         // строка в полосе B, 0..15
    int b_c = (threadIdx.x % 16) * 8;   // столбец в полосе B, 0..120

    const half* a_src = a + (block_row * 128 + a_r) * k + a_c;
    const half* b_src = b + b_r * w + block_col * 128 + b_c;

    int steps = k / 16;

    // пролог: срез 0 сразу кладём в буфер 0
    uint4 a_next = *reinterpret_cast<const uint4*>(a_src);
    uint4 b_next = *reinterpret_cast<const uint4*>(b_src);
    *reinterpret_cast<uint4*>(&shared_a[0][a_r * 16 + a_c]) = a_next;
    *reinterpret_cast<uint4*>(&shared_b[0][b_r * 128 + b_c]) = b_next;
    __syncthreads();

    for (int i = 0; i < steps; ++i) {
        int cur = i % 2;

        // запросы за следующим срезом уходят до счёта, а ждать ответ придётся только при записи в shared после счёта
        if (i + 1 < steps) {
            a_next = *reinterpret_cast<const uint4*>(a_src + (i + 1) * 16);
            b_next = *reinterpret_cast<const uint4*>(b_src + (i + 1) * 16 * w);
        }

#pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            wmma::load_matrix_sync(a_frag[ti], &shared_a[cur][(warp_row * 32 + ti * 16) * 16], 16);
        }

#pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            wmma::load_matrix_sync(b_frag, &shared_b[cur][warp_col * 64 + tj * 16], 128);
#pragma unroll
            for (int ti = 0; ti < 2; ++ti) {
                wmma::mma_sync(acc[ti * 4 + tj], a_frag[ti], b_frag, acc[ti * 4 + tj]);
            }
        }

        // пишем в другой буфер: его все дочитали на прошлом шаге, до прошлого __syncthreads
        if (i + 1 < steps) {
            *reinterpret_cast<uint4*>(&shared_a[cur ^ 1][a_r * 16 + a_c]) = a_next;
            *reinterpret_cast<uint4*>(&shared_b[cur ^ 1][b_r * 128 + b_c]) = b_next;
        }

        // один барьер на шаг: следующий буфер дописан, текущий все дочитали
        __syncthreads();
    }

    int c_row = block_row * 128 + warp_row * 32;
    int c_col = block_col * 128 + warp_col * 64;

#pragma unroll
    for (int ti = 0; ti < 2; ++ti) {
#pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            wmma::store_matrix_sync(c + (c_row + ti * 16) * w + c_col + tj * 16, acc[ti * 4 + tj], w, wmma::mem_row_major);
        }
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    // кернел не обрабатывает края: куски 128x128 и шаг по k 16
    rassert(w % 128 == 0 && h % 128 == 0 && k % 16 == 0, 2345236245234, w, h, k);
    cudaStream_t stream = context.cudaStream();

    // буферы под half-копии A и B выделяются один раз на всю программу (resizeN ничего не делает, если размер не изменился),
    // а new без delete - чтобы cudaFree не вызвался в статическом деструкторе уже после завершения CUDA
    static gpu::gpu_mem_16u *a_half = new gpu::gpu_mem_16u();
    static gpu::gpu_mem_16u *b_half = new gpu::gpu_mem_16u();
    a_half->resizeN(h * k);
    b_half->resizeN(k * w);

    unsigned int a_n4 = h * k / 4;
    unsigned int b_n4 = k * w / 4;
    ::convert_float_to_half<<<(a_n4 + 255) / 256, 256, 0, stream>>>(a.cuptr(), (half*) a_half->cuptr(), a_n4);
    CUDA_CHECK_KERNEL_ASYNC(stream);
    ::convert_float_to_half<<<(b_n4 + 255) / 256, 256, 0, stream>>>(b.cuptr(), (half*) b_half->cuptr(), b_n4);
    CUDA_CHECK_KERNEL_ASYNC(stream);

    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>((const half*) a_half->cuptr(), (const half*) b_half->cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
