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
    __m256 alpha_av[od / 8];
    __m256 beta_av[od / 8];

    __m256 epsilon_av = _mm256_set1_ps(epsilon);

    int d = 0;
    for (d = 0; d <= od - 8; d += 8) {
        __m256 variance_av = _mm256_loadu_ps(variance + d);
        __m256 gamma_av = _mm256_loadu_ps(gamma + d);
        __m256 mean_av = _mm256_loadu_ps(mean + d);

        variance_av = _mm256_add_ps(variance_av, epsilon_av);
        variance_av = _mm256_sqrt_ps(variance_av);
        alpha_av[d / 8] = _mm256_div_ps(gamma_av, variance_av);
        
        beta_av[d / 8] = _mm256_mul_ps(alpha_av[d / 8], mean_av);
    }
    if (d < od) {
        for (; d < od; ++d) {
            variance[d] = sqrt(variance[d] + epsilon);
            variance[d] = gamma[d] / variance[d];
            mean[d] = -mean[d] * variance[d];
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                int r_idx = i * (ow * od) +
                        j * od;
                for (d = 0; d <= od - 8; d += 8) {
                    __m256 in_av = _mm256_loadu_ps(in_layer + r_idx + d);
                    __m256 r_av = _mm256_mul_ps(in_av, alpha_av[d / 8]);
                    r_av = _mm256_sub_ps(r_av, beta_av[d / 8]);
                    _mm256_storeu_ps(result + r_idx + d, r_av);
                }
                if (d < od) {
                    for (; d < od; ++d) {
                        result[r_idx + d] =
                                in_layer[r_idx + d] * variance[d] - mean[d];
                    }
                }
            }
        }
    }
}

struct conv2d_shape_arg {
    int batch, oh, ow, od;
    int ih, iw, ic;
    int kh, kw;
    int sh, sw;
};

void conv2d(float* in_layer, 
        float* kernel, 
        float* result,
        int* shape_arg_arr)
{
    struct conv2d_shape_arg* shape = (struct conv2d_shape_arg*) shape_arg_arr;
    for (int b = 0; b < shape->batch; ++b) {
        for (int i = 0; i < shape->oh; ++i) {
            for (int j = 0; j < shape->ow; ++j) {
                __m256 r_av[shape->od / 8]; 
                for (int d = 0; d <= shape->od - 8; d += 8) {
                    r_av[d / 8] = _mm256_setzero_ps();
                }
                int r_idx = b * (shape->oh * shape->ow * shape->od) +
                        i * (shape->ow * shape->od) +
                        j * shape->od;
                for (int c = 0; c < shape->ic; ++c) {
                    for (int di = 0; di < shape->kh; ++di) {
                        for (int dj = 0; dj < shape->kw; ++dj) {
                            int i_idx = b * (shape->ih * shape->iw * shape->ic) +
                                    (shape->sh * i + di) * (shape->iw * shape->ic) +
                                    (shape->sw * j + dj) * shape->ic +
                                    c;
                            int k_idx = di * (shape->kw * shape->ic * shape->od) +
                                    dj * (shape->ic * shape->od) +
                                    c * shape->od;
                            int d;
                            for (d = 0; d <= shape->od - 8; d += 8) {                          
                                __m256 in_av = _mm256_set1_ps(*(in_layer + i_idx));
                                __m256 k_av = _mm256_loadu_ps(kernel + k_idx + d);
                                __m256 cr_av = _mm256_mul_ps(in_av, k_av);
                                r_av[d / 8] = _mm256_add_ps(r_av[d / 8], cr_av);
                            }

                            if (d < shape->od) {
                                for (; d < shape->od; ++d) {
                                    result[r_idx + d] += in_layer[i_idx] * kernel[k_idx + d];
                                }
                            }
                        }
                    }
                }
                for (int d = 0; d <= shape->od - 8; d += 8) {
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
    __m256 const_01 = _mm256_set1_ps(0.1);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                int d;
                int idx = b * (oh * ow * od) +
                        i * (ow * od) +
                        j * od;
                for (d = 0; d < od; d += 8) {
                    __m256 in_av = _mm256_loadu_ps(in_layer + idx + d);
                    __m256 in_01 = _mm256_mul_ps(in_av, const_01);
                    __m256 r_av = _mm256_max_ps(in_av, in_01);
                    _mm256_storeu_ps(result + idx + d, r_av);
                }

                if (d < od) {
                    for (; d < od; ++d) {
                        float t = in_layer[idx + d];
                        result[idx + d] = t < 0 ? 0.1 * t : t;
                    } 
                }
            }
        }
    }
}