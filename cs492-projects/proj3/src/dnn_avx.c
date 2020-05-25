#include "cblas.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "immintrin.h"
#include "time.h"

#define MAX(x, y) (x >= y ? x : y)

void bias_add(float* in_layer, float* biases, float* result,
        int batch, int oh, int ow, int od)
{
    __m256 biases_av[od / 8];
    for (int d = 0; d <= od - 8; d += 8) {
        biases_av[d / 8] = _mm256_loadu_ps(biases + d);
    }
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                int d;
                for (d = 0; d <= od - 8; d += 8) {
                    int ri = b * (oh * ow * od) +
                            i * (ow * od) +
                            j * od +
                            d;
                    __m256 in_av = _mm256_loadu_ps(in_layer + ri);
                    __m256 r_av = _mm256_add_ps(in_av, biases_av[d / 8]);
                    _mm256_storeu_ps(result + ri, r_av);
                }

                if (d < od) {
                    for (; d < od; ++d) {
                        int rri = b * (oh * ow * od) +
                            i * (ow * od) +
                            j * od +
                            d;
                        result[rri] = in_layer[rri] + biases[d];
                    }
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

void conv2d(float* in_layer, 
        float* kernel, 
        float* result,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw)
{
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                __m256 r_av[od / 8]; 
                for (int d = 0; d <= od - 8; d += 8) {
                    r_av[d / 8] = _mm256_setzero_ps();
                }
                int r_idx = b * (oh * ow * od) +
                        i * (ow * od) +
                        j * od;
                for (int c = 0; c < ic; ++c) {
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int i_idx = b * (ih * iw * ic) +
                                    (sh * i + di) * (iw * ic) +
                                    (sw * j + dj) * ic +
                                    c;
                            int k_idx = di * (kw * ic * od) +
                                    dj * (ic * od) +
                                    c * od;
                            int d;
                            for (d = 0; d <= od - 8; d += 8) {                          
                                __m256 in_av = _mm256_set1_ps(*(in_layer + i_idx));
                                __m256 k_av = _mm256_loadu_ps(kernel + k_idx + d);
                                __m256 cr_av = _mm256_mul_ps(in_av, k_av);
                                r_av[d / 8] = _mm256_add_ps(r_av[d / 8], cr_av);
                            }

                            if (d < od) {
                                for (; d < od; ++d) {
                                    result[r_idx + d] += in_layer[i_idx] * kernel[k_idx + d];
                                }
                            }
                        }
                    }
                }
                for (int d = 0; d <= od - 8; d += 8) {
                    _mm256_storeu_ps(result + r_idx + d, r_av[d / 8]);
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
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                int in_i = i * sh;
                int in_j = j * sw;

                int i_idx = b * (ih * iw * ic) +
                        in_i * (iw * ic) +
                        in_j * ic;
                int r_idx = b * (oh * ow * od) +
                        i * (ow * od) +
                        j * od;
                
                int d;
                for (d = 0; d <= od - 8; d += 8) {
                    __m256 imax_av = _mm256_loadu_ps(in_layer + i_idx + d);
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            __m256 icand_av = _mm256_loadu_ps(
                                    in_layer + i_idx +
                                    di * (iw * ic) +
                                    dj * ic +
                                    d);
                            imax_av = _mm256_max_ps(imax_av, icand_av);
                        }
                    }
                    _mm256_storeu_ps(result + r_idx + d, imax_av);
                }

                if (d < od) {
                    for (; d < od; ++d) {
                        float imax = in_layer[i_idx + d];
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                imax = MAX(imax, 
                                        in_layer[i_idx +
                                        di * (iw * ic) +
                                        dj * ic +
                                        d]);
                            }
                            result[r_idx + d] = imax;
                        }
                    }
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