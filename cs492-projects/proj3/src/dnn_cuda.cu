#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define BLOCK_SIZE 16
#define BLOCK_SQ_SIZE 256

#define MAX(x, y) (x >= y ? x : y)

__global__ void h_cuda_matmul(float* imcol, float* kernel, float* result, 
        int m_size, int n_size, int k_size)
{
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_idx < m_size && c_idx < n_size) {
        float res = 0.0f;
        for (int i = 0; i < k_size; ++i) {
            res += imcol[r_idx * k_size + i] * kernel[i * n_size + c_idx];
        }
        result[r_idx * n_size + c_idx] = res;
    }
}

__global__ void h_cuda_batch_norm(float* in_layer, float* alpha, float* beta, float* result,
        int r_size, int od)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < r_size) {
        int d = t_idx % od;
        result[t_idx] = in_layer[t_idx] * alpha[d] - beta[d];
    }
}

void im2col(float* imb_arg,
    float* colb_arg,
    int oh, int ow,
    int ih, int iw, int ic,
    int kh, int kw, 
    int sh, int sw)
{
    float (*imb)[iw][ic] = (float (*)[iw][ic]) imb_arg;
    float (*colb)[ic * kh * kw] = (float (*)[ic * kh * kw]) colb_arg;

    for (int i = 0; i < oh; ++i) {
        for (int j = 0; j < ow; ++j) {
            int patch_i = i * sh;
            int patch_j = j * sw;
            for (int c = 0; c < ic; ++c) {
                int col_i = i * ow + j;
                int col_j = c * (kh * kw);
                for (int k = 0; k < kh * kw; ++k) {
                    colb[col_i][col_j + k] = imb[patch_i + k / kw][patch_j + k % kw][c];
                }
            }
        }
    }
}

extern "C" {

void conv2d_cuda(float* in_layer,
        float* col,
        float* kernel_r, 
        float* result,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw)
{
    for (int b = 0; b < batch; ++b) {
        float* imb = in_layer + b * (oh * ow * od);
        float* colb = col + b * ((oh * ow) * (ic * kh * kw));
        float* resultb = result + b * (oh * ow * od);

        im2col(imb,
                colb,
                oh, ow,
                ih, iw, ic,
                kh, kw,
                sh, sw);

        // colb : (oh * ow) X (ic * kh * kw)
        // kernel_r : (ic * kh * kw) X od

        int m_size = oh * ow;
        int n_size = od;
        int k_size = ic * kh * kw;

        float* d_imcol;
        float* d_kernel;
        float* d_result;
        cudaMalloc((void **) &d_imcol, sizeof(float) * m_size * k_size);
        cudaMalloc((void **) &d_kernel, sizeof(float) * k_size * n_size);
        cudaMalloc((void **) &d_result, sizeof(float) * m_size * k_size);

        cudaMemcpy(d_imcol, colb, sizeof(float) * m_size * k_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel_r, sizeof(float) * k_size * n_size, cudaMemcpyHostToDevice);
        
        unsigned int grid_r = (m_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_c = (k_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid_dim(grid_c, grid_r);
        dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

        h_cuda_matmul<<<grid_dim, block_dim>>>(d_imcol, d_kernel, d_result, m_size, n_size, k_size);
        cudaFree(d_imcol);
        cudaFree(d_kernel);

        cudaMemcpy(resultb, d_result, sizeof(float) * m_size * n_size, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
    }
}


void bias_add(float* in_layer, float* biases, float* result,
    int batch, int oh, int ow, int od) 
{
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    int ri = b * (oh * ow * od) +
                            i * (ow * od) +
                            j * od +
                            d;
                    result[ri] = in_layer[ri] + biases[d];
                }
            }
        }
    }
}

void batch_norm_cuda(float* in_layer,
    float* alpha,
    float* beta,
    float* result,
    int batch, int oh, int ow, int od)
{
    int r_size = batch * oh * ow * od;
    if (r_size < 1e+6) {
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    for (int d = 0; d < od; ++d) {
                        int ri = i * (ow * od) +
                                j * od +
                                d;
                        result[ri] = in_layer[ri] * alpha[d] - beta[d];
                    }
                }
            }
        }

    } else {
        float* d_in_layer;
        float* d_alpha;
        float* d_beta;
        float* d_result;

        cudaMalloc((void **) &d_in_layer, sizeof(float) * r_size);
        cudaMalloc((void **) &d_alpha, sizeof(float) * od);
        cudaMalloc((void **) &d_beta, sizeof(float) * od);
        cudaMalloc((void **) &d_result, sizeof(float) * r_size);

        cudaMemcpy(d_in_layer, in_layer, sizeof(float) * r_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha, sizeof(float) * od, cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta, sizeof(float) * od, cudaMemcpyHostToDevice);
        cudaMemcpy(d_result, result, sizeof(float) * r_size, cudaMemcpyHostToDevice);
        
        unsigned int grid_size = (r_size + BLOCK_SQ_SIZE - 1) / BLOCK_SQ_SIZE;
        dim3 grid_dim(grid_size);
        dim3 block_dim(BLOCK_SQ_SIZE);

        h_cuda_batch_norm<<<grid_dim, block_dim>>>(d_in_layer, d_alpha, d_beta, d_result, r_size, od);
        cudaFree(d_in_layer);
        cudaFree(d_alpha);
        cudaFree(d_beta);

        cudaMemcpy(result, d_result, sizeof(float) * r_size, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
    }
}

void max_pool2d(float* in_layer,
        float* result,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw)
{
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    int in_i = i * sh;
                    int in_j = j * sw;
                    float imax = in_layer[
                            b * (ih * iw * ic) +
                            in_i * (iw * ic) +
                            in_j * ic +
                            d
                    ];
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int ii = b * (ih * iw * ic) +
                                    (in_i + di) * (iw * ic) +
                                    (in_j + dj) * ic +
                                    d;
                            imax = MAX(imax, in_layer[ii]);
                        }
                    }
                    result[
                            b * (oh * ow * od) +
                            i * (ow * od) +
                            j * od +
                            d
                    ] = imax;
                }
            }
        }
    }
}

void leaky_relu(float* in_layer,
        float* result,
        int batch, int oh, int ow, int od)
{
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    int idx = b * (oh * ow * od) +
                            i * (ow * od) +
                            j * od +
                            d;
                    float t = in_layer[idx];
                    result[idx] = t < 0 ? 0.1 * t : t;
                }
            }
        }
    }
}
}