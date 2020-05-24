#include "cblas.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "immintrin.h"
#include "time.h"

#define MAX(x, y) (x >= y ? x : y)

union avx256 {
    __m256 av;
    float r[8];
};

void bias_add(float* in_layer, float* biases, float* result,
        int batch, int oh, int ow, int od)
{
    int dod = od / 8;
    int dod_remain = od % 8 == 0 ? 0 : 1;
    __m256 biases_av[dod];
    for (int dd = 0; dd < dod; ++dd) {
        biases_av[dd] = _mm256_loadu_ps(biases + 8 * dd);
    }

    union avx256 result_av;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int dd = 0; dd < dod; ++dd) {
                    int ri = b * (oh * ow * od) +
                            i * (ow * od) +
                            j * od +
                            8 * dd;
                    __m256 in_av = _mm256_loadu_ps(in_layer + ri);
                    result_av.av = _mm256_add_ps(in_av, biases_av[dd]);
                    memcpy(result + ri, result_av.r, 8 * sizeof(float));
                }

                if (dod_remain) {
                    for (int rd = 8 * dod; rd < od; ++rd) {
                        int rri = b * (oh * ow * od) +
                            i * (ow * od) +
                            j * od +
                            rd;
                        result[rri] = in_layer[rri] + biases[rd];
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