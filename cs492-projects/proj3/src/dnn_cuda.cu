#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define BLOCK_SIZE 16

#define MAX(x, y) (x >= y ? x : y)

__global__ void conv2d_matmul(float* imcol, float* kernel, float* result, int size_m, int size_n, int size_k)
{
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_idx < size_m && c_idx < size_n) {
        float res = 0.0f;
        for (int i = 0; i < size_k; ++i) {
            res += imcol[r_idx * size_k + i] * kernel[i * size_n + c_idx];
        }
        result[r_idx * size_n + c_idx] = res;
    }
}

void cpu_matmul(float* imcol, float* kernel, float* result, int size_m, int size_n, int size_k,
        int r_idx, int c_idx)
{
    float res = 0.0f;
    for (int i = 0; i < size_k; ++i) {
        res += imcol[r_idx * size_k + i] * kernel[i * size_n + c_idx];
    }
    result[r_idx * size_n + c_idx] = res;
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

void conv2d_mul(float* in_layer,
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

        int size_m = oh * ow;
        int size_n = od;
        int size_k = ic * kh * kw;

        float* d_imcol;
        float* d_kernel;
        float* d_result;
        cudaMalloc((void **) &d_imcol, sizeof(float) * size_m * size_k);
        cudaMalloc((void **) &d_kernel, sizeof(float) * size_k * size_n);
        cudaMalloc((void **) &d_result, sizeof(float) * size_m * size_k);

        cudaMemcpy(d_imcol, colb, sizeof(float) * size_m * size_k, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel_r, sizeof(float) * size_k * size_n, cudaMemcpyHostToDevice);
        
        unsigned int grid_r = (size_m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_c = (size_k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_c, grid_r);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        conv2d_matmul<<<dimGrid, dimBlock>>>(d_imcol, d_kernel, d_result, size_m, size_n, size_k);
        cudaFree(d_imcol);
        cudaFree(d_kernel);

        cudaMemcpy(resultb, d_result, sizeof(float) * size_m * size_n, cudaMemcpyDeviceToHost);
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

void batch_norm(float* in_layer,
        float* mean,
        float* variance,
        float* gamma,
        float epsilon,
        float* result,
        int batch, int oh, int ow, int od)
{
    for (int d = 0; d < od; ++d) {
        variance[d] = sqrt(variance[d] + epsilon);
    }

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    int ri = i * (ow * od) +
                            j * od +
                            d;
                    result[ri] = ((in_layer[ri] - mean[d]) / variance[d]) 
                            * gamma[d];
                }
            }
        }
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