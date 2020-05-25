#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define MAX(x, y) (x >= y ? x : y)

extern "C" {
void bias_add(float* in_layer, float* biases, float* result,
    int batch, int oh, int ow, int od) {

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int size = batch * oh * ow * od;
    float ones[oh * ow];
    for (int i = 0; i < oh * ow; ++i) {
        ones[i] = 1.0;
    }

    float* d_in_layer;
    float* d_ones;
    float* d_result;
    cudaStat = cudaMalloc((void**)&d_in_layer, size * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_result, size * sizeof(float));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(size, sizeof(float), in_layer, 1, d_in_layer, 1);

    stat = cublasScopy(handle, size, d_in_layer, 1, d_result, 1);
    cudaFree(d_in_layer);

    cudaStat = cudaMalloc((void**) &d_ones, oh * ow * sizeof(float));
    stat = cublasSetVector(oh * ow, sizeof(float), ones, 1, d_ones, 1);

    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            stat = cublasSaxpy(
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

    stat = cublasGetVector(size, sizeof(float), d_result, 1, result, 1);
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