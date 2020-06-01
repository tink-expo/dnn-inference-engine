#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define MAX(x, y) (x >= y ? x : y)

void im2col(float* im_b,
    float* col_b,
    int oh, int ow,
    int ih, int iw, int ic,
    int kh, int kw, 
    int sh, int sw)
{
int col_w = ic * kh * kw;
for (int i = 0; i < oh; ++i) {
    for (int j = 0; j < ow; ++j) {
        int patch_i = i * sh;
        int patch_j = j * sw;
        for (int c = 0; c < ic; ++c) {
            int col_i = i * ow + j;
            int col_j = c * (kh * kw);
            for (int di = 0; di < kh; ++di) {
                for (int dj = 0; dj < kw; ++dj) {
                    col_b[col_i * col_w +
                            col_j + (di * kw) + dj] = 
                            im_b[(patch_i + di) * (iw * ic) +
                            (patch_j + dj) * ic +
                            c];
                }
            }
        }
    }
}
}

extern "C" {
void bias_add(float* in_layer, float* biases, float* result,
    int batch, int oh, int ow, int od) {

    cublasHandle_t handle;

    int size = batch * oh * ow * od;
    float ones[oh * ow];
    for (int i = 0; i < oh * ow; ++i) {
        ones[i] = 1.0;
    }

    float* d_in_layer;
    float* d_ones;
    float* d_result;
    cudaMalloc((void**)&d_in_layer, size * sizeof(float));
    cudaMalloc((void**)&d_result, size * sizeof(float));

    cublasCreate(&handle);
    cublasSetVector(size, sizeof(float), in_layer, 1, d_in_layer, 1);

    cublasScopy(handle, size, d_in_layer, 1, d_result, 1);
    cudaFree(d_in_layer);

    cudaMalloc((void**) &d_ones, oh * ow * sizeof(float));
    cublasSetVector(oh * ow, sizeof(float), ones, 1, d_ones, 1);

    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            cublasSaxpy(
                handle,
                oh * ow,
                biases + d,
                d_ones,
                1,
                d_result + d + b * (oh * ow * od),
                od
            );
        }
    }
    cudaFree(d_ones);

    cublasGetVector(size, sizeof(float), d_result, 1, result, 1);
    cudaFree(d_result);

    cublasDestroy(handle);
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

void conv2d_cublas(float* in_layer,
    float* col,
    float* kernel_r, 
    float* result,
    int batch, int oh, int ow, int od,
    int ih, int iw, int ic,
    int kh, int kw,
    int sh, int sw)
{
    cublasHandle_t handle;
    cublasCreate (&handle);
    
    for (int b = 0; b < batch; ++b) {
        float* im_b = in_layer + b * (ih * iw * ic);
        float* col_b = col + b * ((oh * ow) * (ic * kh * kw));
        float* result_b = result + b * (oh * ow * od);


        im2col(im_b,
                col_b,
                oh, ow,
                ih, iw, ic,
                kh, kw,
                sh, sw);

        // col_b : (oh * ow) X (ic * kh * kw)
        // kernel_r : (ic * kh * kw) X od

        int m_size = oh * ow;
        int n_size = od;
        int k_size = ic * kh * kw;

        float* d_col;
        float* d_kernel;
        float* d_result;
        cudaMalloc((void **) &d_col, sizeof(float) * m_size * k_size);
        cudaMalloc((void **) &d_kernel, sizeof(float) * k_size * n_size);
        cudaMalloc((void **) &d_result, sizeof(float) * m_size * k_size);

        cublasSetMatrix(m_size, k_size, sizeof(float), col_b, m_size, d_col, m_size);
        cublasSetMatrix(k_size, n_size, sizeof(float), kernel_r, k_size, d_kernel, k_size);
        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n_size, m_size, k_size,
                &alpha, 
                d_kernel, n_size,
                d_col, k_size,
                &beta,
                d_result,
                n_size);
        cudaFree(d_col);
        cudaFree(d_kernel);

        cublasGetMatrix(m_size, n_size, sizeof(float), d_result, m_size, result_b, m_size);
        cudaFree(d_result);
    }

    cublasDestroy (handle);
}

void conv2d(float* in_layer, 
        float* kernel, 
        float* result,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw)
{
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int c = 0; c < ic; ++c) {
                for (int i = 0; i < oh; ++i) {
                    for (int j = 0; j < ow; ++j) {
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                int ri = b * (oh * ow * od) +
                                        i * (ow * od) +
                                        j * od +
                                        d;
                                int ii = b * (ih * iw * ic) +
                                        (sh * i + di) * (iw * ic) +
                                        (sw * j + dj) * ic +
                                        c;
                                int ki = di * (kw * ic * od) +
                                        dj * (ic * od) +
                                        c * od +
                                        d;
                                
                                result[ri] += in_layer[ii] * kernel[ki];
                            }
                        }
                    }
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