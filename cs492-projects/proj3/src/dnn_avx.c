#include "cblas.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "immintrin.h"
#include "time.h"
#include "pthread.h"

#define MAX(x, y) (x >= y ? x : y)

#define P_THREADS 4

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
        float* alpha,
        float* beta,
        float* result,
        int batch, int oh, int ow, int od)
{
    __m256 alpha_av[od / 8];
    __m256 beta_av[od / 8];

    int d = 0;
    for (d = 0; d <= od - 8; d += 8) {
        alpha_av[d / 8] = _mm256_loadu_ps(alpha + d);
        beta_av[d / 8] = _mm256_loadu_ps(beta + d);
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
                                in_layer[r_idx + d] * alpha[d] - beta[d];
                    }
                }
            }
        }
    }
}

struct shape_arg {
    int batch, oh, ow, od;
    int ih, iw, ic;
    int kh, kw;
    int sh, sw;
};

struct conv2d_thread_arg {
    float* in_layer;
    float* kernel;
    float* result;
    struct shape_arg* shape;
    int oh_s;
    int oh_e;
};

struct im2col_thread_arg {
    float* im_kernel_b;  // also used for kernel
    float* col_b;
    float* result_b;
    struct shape_arg* shape;
    int oh_s;
    int oh_e;
};

void* conv2d_matmul_thread_func(void* thread_arg)
{
    struct im2col_thread_arg* arg = (struct im2col_thread_arg*) thread_arg;
    struct shape_arg* shape = arg->shape;

    // col_b : (oh * ow) X (ic * kh * kw)
    // kernel_r : (ic * kh * kw) X od

    int n_size = shape->od;
    int k_size = shape->ic * shape->kh * shape->kw;

    // for (int i = arg->oh_s; i < arg->oh_e; ++i) {
    //     for (int j = 0; j < n_size; ++j) {
    //         float res = 0.0f;
    //         for (int k = 0; k < k_size; ++k) {
    //             res += arg->col_b[i * k_size + k] * arg->im_kernel_b[k * n_size + j];
    //         }
    //         arg->result_b[i * n_size + j] = res;
    //     }
    // }

    for (int i = arg->oh_s; i < arg->oh_e; ++i) {
        // __m256 col_av
        int d;
        for (d = 0; d <= n_size - 8; d += 8) {
            __m256 r_av = _mm256_setzero_ps();
            for (int k = 0; k < k_size; ++k) {
                __m256 col_av = _mm256_set1_ps(arg->col_b[i * k_size + k]);
                __m256 ker_av = _mm256_loadu_ps(arg->im_kernel_b + k * n_size + d);
                __m256 cr_av = _mm256_mul_ps(col_av, ker_av);
                r_av = _mm256_add_ps(r_av, cr_av);
            }
            _mm256_storeu_ps(arg->result_b + i * n_size + d, r_av);
        }
        if (d < n_size) {
            for (; d < n_size; ++d) {
                float res = 0.0f;
                for (int k = 0; k < k_size; ++k) {
                    res += arg->col_b[i * k_size + k] * arg->im_kernel_b[k * n_size + d];
                }
                arg->result_b[i * n_size + d] = res;
            }
        }
    }
}

void* im2col_thread_func(void* thread_arg)
{
    struct im2col_thread_arg* arg = (struct im2col_thread_arg*) thread_arg;
    struct shape_arg* shape = arg->shape;

    int col_w = shape->ic * shape->kh * shape->kw;
    for (int i = arg->oh_s; i < arg->oh_e; ++i) {
        for (int j = 0; j < shape->ow; ++j) {
            int patch_i = i * shape->sh;
            int patch_j = j * shape->sw;
            for (int c = 0; c < shape->ic; ++c) {
                int col_i = i * shape->ow + j;
                int col_j = c * (shape->kh * shape->kw);
                for (int di = 0; di < shape->kh; ++di) {
                    for (int dj = 0; dj < shape->kw; ++dj) {
                        arg->col_b[col_i * col_w +
                                col_j + (di * shape->kw) + dj] = 
                                arg->im_kernel_b[(patch_i + di) * (shape->iw * shape->ic) +
                                (patch_j + dj) * shape->ic +
                                c];
                    }
                }
            }
        }
    }

    return 0;
}

void conv2d_matmul(float* in_layer,
        float* col,
        float* kernel_r, 
        float* result,
        int* shape_arg_arr)
{
    struct shape_arg* shape = (struct shape_arg*) shape_arg_arr;
    
    for (int b = 0; b < shape->batch; ++b) {
        float* im_b = in_layer + b * (shape->ih * shape->iw * shape->ic);
        float* col_b = col + b * ((shape->oh * shape->ow) * (shape->ic * shape->kh * shape->kw));
        float* result_b = result + b * (shape->oh * shape->ow * shape->od);

        pthread_t threads[P_THREADS];
        struct im2col_thread_arg t_args[P_THREADS];

        t_args[0].im_kernel_b = im_b;
        t_args[0].col_b = col_b;
        t_args[0].shape = shape;
        int oh_part_size = shape->oh / P_THREADS;

        int t_id;

        for (int t_idx = 0; t_idx < P_THREADS; ++t_idx) {
            if (t_idx > 0) {
                t_args[t_idx] = t_args[0];
            }

            int oh_s = oh_part_size * t_idx;
            int oh_e = t_idx < P_THREADS - 1 ? oh_s + oh_part_size : shape->oh;
            
            t_args[t_idx].oh_s = oh_s;
            t_args[t_idx].oh_e = oh_e;

            t_id = pthread_create(&threads[t_idx], NULL, im2col_thread_func, (void*) &t_args[t_idx]);
            if (t_id < 0) {
                perror("conv2d im2col thread error : ");
                exit(0);
            }
        }

        for (int t_idx = 0; t_idx < P_THREADS; ++t_idx) {
            pthread_join(threads[t_idx], NULL);
        }

        // col_b : (oh * ow) X (ic * kh * kw)
        // kernel_r : (ic * kh * kw) X od

        int m_size = shape->oh * shape->ow;
        int n_size = shape->od;
        int k_size = shape->ic * shape->kh * shape->kw;
        oh_part_size = m_size / P_THREADS;

        for (int t_idx = 0; t_idx < P_THREADS; ++t_idx) {
            t_args[t_idx].im_kernel_b = kernel_r;
            t_args[t_idx].result_b = result_b;

            int oh_s = oh_part_size * t_idx;
            int oh_e = t_idx < P_THREADS - 1 ? oh_s + oh_part_size : m_size;
            
            t_args[t_idx].oh_s = oh_s;
            t_args[t_idx].oh_e = oh_e;

            t_id = pthread_create(&threads[t_idx], NULL, conv2d_matmul_thread_func, (void*) &t_args[t_idx]);
            if (t_id < 0) {
                perror("conv2d im2col thread error : ");
                exit(0);
            }
        }

        for (int t_idx = 0; t_idx < P_THREADS; ++t_idx) {
            pthread_join(threads[t_idx], NULL);
        }
    }
}

void* conv2d_thread_func(void* thread_arg) 
{
    struct conv2d_thread_arg* arg = (struct conv2d_thread_arg*) thread_arg;
    struct shape_arg* shape = arg->shape;
    for (int b = 0; b < shape->batch; ++b) {
        for (int i = arg->oh_s; i < arg->oh_e; ++i) {
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
                                __m256 in_av = _mm256_set1_ps(*(arg->in_layer + i_idx));
                                __m256 k_av = _mm256_loadu_ps(arg->kernel + k_idx + d);
                                __m256 cr_av = _mm256_mul_ps(in_av, k_av);
                                r_av[d / 8] = _mm256_add_ps(r_av[d / 8], cr_av);
                            }

                            if (d < shape->od) {
                                for (; d < shape->od; ++d) {
                                    (arg->result)[r_idx + d] += 
                                            (arg->in_layer)[i_idx] * (arg->kernel)[k_idx + d];
                                }
                            }
                        }
                    }
                }
                for (int d = 0; d <= shape->od - 8; d += 8) {
                    _mm256_storeu_ps(arg->result + r_idx + d, r_av[d / 8]);
                }
            }
        }
    }
}

void conv2d(float* in_layer, 
        float* kernel, 
        float* result,
        int* shape_arg_arr)
{
    struct shape_arg* shape = (struct shape_arg*) shape_arg_arr;
    
    pthread_t threads[P_THREADS];
    struct conv2d_thread_arg t_args[P_THREADS];
    int oh_part_size = shape->oh / P_THREADS;

    t_args[0].in_layer = in_layer;
    t_args[0].kernel = kernel;
    t_args[0].result = result;
    t_args[0].shape = shape;

    int t_id;
    
    for (int t_idx = 0; t_idx < P_THREADS; ++t_idx) {
        if (t_idx > 0) {
            t_args[t_idx] = t_args[0];
        }

        int oh_s = oh_part_size * t_idx;
        int oh_e = t_idx < P_THREADS - 1 ? oh_s + oh_part_size : shape->oh;
        
        t_args[t_idx].oh_s = oh_s;
        t_args[t_idx].oh_e = oh_e;

        t_id = pthread_create(&threads[t_idx], NULL, conv2d_thread_func, (void*) &t_args[t_idx]);
        if (t_id < 0) {
            perror("conv2d thread error : ");
            exit(0);
        }
    }

    for (int t_idx = 0; t_idx < P_THREADS; ++t_idx) {
        pthread_join(threads[t_idx], NULL);
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