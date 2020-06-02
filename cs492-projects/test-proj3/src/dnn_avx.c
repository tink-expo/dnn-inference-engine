#include "cblas.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "immintrin.h"
#include "time.h"
#include "pthread.h"

#define MAX(x, y) (x >= y ? x : y)
#define MIN(x, y) (x <= y ? x : y)

#define P_THREADS 4

// [Conv2d]

struct shape_arg {
    int oh, ow, od;
    int ih, iw, ic;
    int kh, kw;
    int sh, sw;
};

struct conv2d_thread_arg {
    float* im_b;
    float* kernel;
    float* result_b;
    struct shape_arg* shape;
    int oh_s;
    int oh_e;
};

void* conv2d_thread_func(void* thread_arg) 
{
    struct conv2d_thread_arg* arg = (struct conv2d_thread_arg*) thread_arg;
    struct shape_arg* shape = arg->shape;

    for (int i = arg->oh_s; i < arg->oh_e; ++i) {
        for (int j = 0; j < shape->ow; ++j) {
            __m256 r_av[shape->od / 8]; 
            for (int d = 0; d <= shape->od - 8; d += 8) {
                r_av[d / 8] = _mm256_setzero_ps();
            }
            int r_idx = i * (shape->ow * shape->od) +
                    j * shape->od;
            for (int c = 0; c < shape->ic; ++c) {
                for (int di = 0; di < shape->kh; ++di) {
                    for (int dj = 0; dj < shape->kw; ++dj) {
                        int i_idx = (shape->sh * i + di) * (shape->iw * shape->ic) +
                                (shape->sw * j + dj) * shape->ic +
                                c;
                        int k_idx = di * (shape->kw * shape->ic * shape->od) +
                                dj * (shape->ic * shape->od) +
                                c * shape->od;
                        int d;
                        for (d = 0; d <= shape->od - 8; d += 8) {                          
                            __m256 in_av = _mm256_set1_ps(*(arg->im_b + i_idx));
                            __m256 k_av = _mm256_loadu_ps(arg->kernel + k_idx + d);
                            __m256 cr_av = _mm256_mul_ps(in_av, k_av);
                            r_av[d / 8] = _mm256_add_ps(r_av[d / 8], cr_av);
                        }

                        if (d < shape->od) {
                            for (; d < shape->od; ++d) {
                                (arg->result_b)[r_idx + d] += 
                                        (arg->im_b)[i_idx] * (arg->kernel)[k_idx + d];
                            }
                        }
                    }
                }
            }
            for (int d = 0; d <= shape->od - 8; d += 8) {
                _mm256_storeu_ps(arg->result_b + r_idx + d, r_av[d / 8]);
            }
        }
    }
    return 0;
}

void conv2d_pthread(float* in_layer, 
        float* kernel, 
        float* result,
        int batch,
        int* shape_arg_arr)
{
    struct shape_arg* shape = (struct shape_arg*) shape_arg_arr;

    for (int b = 0; b < batch; ++b) {
        float* im_b = in_layer + b * (shape->ih * shape->iw * shape->ic);
        float* result_b = result + b * (shape->oh * shape->ow * shape->od);

        pthread_t threads[P_THREADS];
        struct conv2d_thread_arg t_args[P_THREADS];
        
        int num_threads = MIN(P_THREADS, shape->oh);
        int oh_part_size = shape->oh / num_threads;

        t_args[0].im_b = im_b;
        t_args[0].kernel = kernel;
        t_args[0].result_b = result_b;
        t_args[0].shape = shape;

        int t_id;
        
        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            if (t_idx > 0) {
                t_args[t_idx] = t_args[0];
            }

            int oh_s = oh_part_size * t_idx;
            int oh_e = t_idx < num_threads - 1 ? oh_s + oh_part_size : shape->oh;
            
            t_args[t_idx].oh_s = oh_s;
            t_args[t_idx].oh_e = oh_e;

            t_id = pthread_create(&threads[t_idx], NULL, conv2d_thread_func, (void*) &t_args[t_idx]);
            if (t_id < 0) {
                perror("conv2d thread error : ");
                exit(0);
            }
        }

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            pthread_join(threads[t_idx], NULL);
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





// [BiasAdd]

struct bias_add_thread_arg {
    float* im_b;
    float* biases;
    float* result_b;

    int ow;
    int od;

    int oh_s;
    int oh_e;
};

void* bias_add_thread_func(void* thread_arg)
{
    struct bias_add_thread_arg* arg = (struct bias_add_thread_arg*) thread_arg;

    __m256 biases_av[arg->od / 8];
    for (int d = 0; d <= arg->od - 8; d += 8) {
        biases_av[d / 8] = _mm256_loadu_ps(arg->biases + d);
    }

    for (int i = arg->oh_s; i < arg->oh_e; ++i) {
        for (int j = 0; j < arg->ow; ++j) {
            int d;
            for (d = 0; d <= arg->od - 8; d += 8) {
                int ri = i * (arg->ow * arg->od) +
                        j * arg->od +
                        d;
                __m256 in_av = _mm256_loadu_ps(arg->im_b + ri);
                __m256 r_av = _mm256_add_ps(in_av, biases_av[d / 8]);
                _mm256_storeu_ps(arg->result_b + ri, r_av);
            }

            if (d < arg->od) {
                for (; d < arg->od; ++d) {
                    int rri = i * (arg->ow * arg->od) +
                        j * arg->od +
                        d;
                    arg->result_b[rri] = arg->im_b[rri] + arg->biases[d];
                }
            }
        }
    }

    return 0;
}

void bias_add_pthread(float* in_layer, float* biases, float* result,
        int batch, int oh, int ow, int od)
{
    for (int b = 0; b < batch; ++b) {
        float* im_b = in_layer + b * (oh * ow * od);
        float* result_b = result + b * (oh * ow * od);

        pthread_t threads[P_THREADS];
        struct bias_add_thread_arg t_args[P_THREADS];
        
        int num_threads = MIN(P_THREADS, oh);
        int oh_part_size = oh / num_threads;

        t_args[0].im_b = im_b;
        t_args[0].biases = biases;
        t_args[0].result_b = result_b;
        t_args[0].ow = ow;
        t_args[0].od = od;

        int t_id;

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            if (t_idx > 0) {
                t_args[t_idx] = t_args[0];
            }

            int oh_s = oh_part_size * t_idx;
            int oh_e = t_idx < num_threads - 1 ? oh_s + oh_part_size : oh;
            
            t_args[t_idx].oh_s = oh_s;
            t_args[t_idx].oh_e = oh_e;

            t_id = pthread_create(&threads[t_idx], NULL, bias_add_thread_func, (void*) &t_args[t_idx]);
            if (t_id < 0) {
                perror("bias add thread error : ");
                exit(0);
            }
        }

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            pthread_join(threads[t_idx], NULL);
        }
    }
}

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




// [MaxPool2D]

struct max_pool2d_thread_arg {
    float* im_b;
    float* result_b;
    struct shape_arg* shape;
    int oh_s;
    int oh_e;
};

void* max_pool2d_thread_func(void* thread_arg)
{
    struct max_pool2d_thread_arg* arg = (struct max_pool2d_thread_arg*) thread_arg;
    struct shape_arg* shape = arg->shape;

    for (int i = arg->oh_s; i < arg->oh_e; ++i) {
        for (int j = 0; j < shape->ow; ++j) {
            int in_i = i * shape->sh;
            int in_j = j * shape->sw;

            int i_idx = in_i * (shape->iw * shape->ic) +
                    in_j * shape->ic;
            int r_idx = i * (shape->ow * shape->od) +
                    j * shape->od;

            int d;
            for (d = 0; d <= shape->od - 8; d += 8) {
                __m256 imax_av = _mm256_loadu_ps(arg->im_b + i_idx + d);
                for (int di = 0; di < shape->kh; ++di) {
                    for (int dj = 0; dj < shape->kw; ++dj) {
                        __m256 icand_av = _mm256_loadu_ps(
                                arg->im_b + i_idx +
                                di * (shape->iw * shape->ic) +
                                dj * shape->ic +
                                d);
                        imax_av = _mm256_max_ps(imax_av, icand_av);
                    }
                }
                _mm256_storeu_ps(arg->result_b + r_idx + d, imax_av);
            }

            if (d < shape->od) {
                for (; d < shape->od; ++d) {
                    float imax = arg->im_b[i_idx + d];
                    for (int di = 0; di < shape->kh; ++di) {
                        for (int dj = 0; dj < shape->kw; ++dj) {
                            imax = MAX(imax, 
                                    arg->im_b[i_idx +
                                    di * (shape->iw * shape->ic) +
                                    dj * shape->ic +
                                    d]);
                        }
                        arg->result_b[r_idx + d] = imax;
                    }
                }
            }
        }
    }

    return 0;
}

void max_pool2d_pthread(float* in_layer,
        float* result,
        int batch,
        int* shape_arg_arr)
{
    struct shape_arg* shape = (struct shape_arg*) shape_arg_arr;

    for (int b = 0; b < batch; ++b) {
        float* im_b = in_layer + b * (shape->ih * shape->iw * shape->ic);
        float* result_b = result + b * (shape->oh * shape->ow * shape->od);

        pthread_t threads[P_THREADS];

        struct max_pool2d_thread_arg t_args[P_THREADS];
        
        int num_threads = MIN(P_THREADS, shape->oh);
        int oh_part_size = shape->oh / num_threads;

        t_args[0].im_b = im_b;
        t_args[0].result_b = result_b;
        t_args[0].shape = shape;

        int t_id;

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            if (t_idx > 0) {
                t_args[t_idx] = t_args[0];
            }

            int oh_s = oh_part_size * t_idx;
            int oh_e = t_idx < num_threads - 1 ? oh_s + oh_part_size : shape->oh;
            
            t_args[t_idx].oh_s = oh_s;
            t_args[t_idx].oh_e = oh_e;

            t_id = pthread_create(&threads[t_idx], NULL, max_pool2d_thread_func, (void*) &t_args[t_idx]);
            if (t_id < 0) {
                perror("conv2d thread error : ");
                exit(0);
            }
        }

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            pthread_join(threads[t_idx], NULL);
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




// [BatchNorm]

struct batch_norm_thread_arg {
    float* im_b;
    float* alpha;
    float* beta;
    float* result_b;
    int ow;
    int od;

    int oh_s;
    int oh_e;
};

void* batch_norm_thread_func(void* thread_arg)
{
    struct batch_norm_thread_arg* arg = (struct batch_norm_thread_arg*) thread_arg;

    __m256 alpha_av[arg->od / 8];
    __m256 beta_av[arg->od / 8];

    int d = 0;
    for (d = 0; d <= arg->od - 8; d += 8) {
        alpha_av[d / 8] = _mm256_loadu_ps(arg->alpha + d);
        beta_av[d / 8] = _mm256_loadu_ps(arg->beta + d);
    }

    for (int i = arg->oh_s; i < arg->oh_e; ++i) {
        for (int j = 0; j < arg->ow; ++j) {
            int r_idx = i * (arg->ow * arg->od) +
                    j * arg->od;
            for (d = 0; d <= arg->od - 8; d += 8) {
                __m256 in_av = _mm256_loadu_ps(arg->im_b + r_idx + d);
                __m256 r_av = _mm256_mul_ps(in_av, alpha_av[d / 8]);
                r_av = _mm256_sub_ps(r_av, beta_av[d / 8]);
                _mm256_storeu_ps(arg->result_b + r_idx + d, r_av);
            }
            if (d < arg->od) {
                for (; d < arg->od; ++d) {
                    arg->result_b[r_idx + d] =
                            arg->im_b[r_idx + d] * arg->alpha[d] - arg->beta[d];
                }
            }
        }
    }

    return 0;
}

void batch_norm_pthread(float* in_layer,
        float* alpha,
        float* beta,
        float* result,
        int batch, int oh, int ow, int od)
{
    for (int b = 0; b < batch; ++b) {
        float* im_b = in_layer + b * (oh * ow * od);
        float* result_b = result + b * (oh * ow * od);

        pthread_t threads[P_THREADS];
        struct batch_norm_thread_arg t_args[P_THREADS];
        
        int num_threads = MIN(P_THREADS, oh);
        int oh_part_size = oh / num_threads;

        t_args[0].im_b = im_b;
        t_args[0].alpha = alpha;
        t_args[0].beta = beta;
        t_args[0].result_b = result_b;
        t_args[0].ow = ow;
        t_args[0].od = od;

        int t_id;

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            if (t_idx > 0) {
                t_args[t_idx] = t_args[0];
            }

            int oh_s = oh_part_size * t_idx;
            int oh_e = t_idx < num_threads - 1 ? oh_s + oh_part_size : oh;
            
            t_args[t_idx].oh_s = oh_s;
            t_args[t_idx].oh_e = oh_e;

            t_id = pthread_create(&threads[t_idx], NULL, batch_norm_thread_func, (void*) &t_args[t_idx]);
            if (t_id < 0) {
                perror("bias add thread error : ");
                exit(0);
            }
        }

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            pthread_join(threads[t_idx], NULL);
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
                int r_idx = b * (oh * ow * od) +
                        i * (ow * od) +
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




// [LeakyRelu]

struct leaky_relu_thread_arg {
    float* im_b;
    float* result_b;
    int ow;
    int od;

    int oh_s;
    int oh_e;
};

void* leaky_relu_thread_func(void* thread_arg)
{
    struct leaky_relu_thread_arg* arg = (struct leaky_relu_thread_arg*) thread_arg;

    __m256 const_01 = _mm256_set1_ps(0.1);
    for (int i = arg->oh_s; i < arg->oh_e; ++i) {
        for (int j = 0; j < arg->ow; ++j) {
            int d;
            int idx = i * (arg->ow * arg->od) +
                    j * arg->od;
            for (d = 0; d < arg->od; d += 8) {
                __m256 in_av = _mm256_loadu_ps(arg->im_b + idx + d);
                __m256 in_01 = _mm256_mul_ps(in_av, const_01);
                __m256 r_av = _mm256_max_ps(in_av, in_01);
                _mm256_storeu_ps(arg->result_b + idx + d, r_av);
            }

            if (d < arg->od) {
                for (; d < arg->od; ++d) {
                    float t = arg->im_b[idx + d];
                    arg->result_b[idx + d] = t < 0 ? 0.1 * t : t;
                } 
            }
        }
    }

    return 0;
}

void leaky_relu_pthread(float* in_layer,
        float* result,
        int batch, int oh, int ow, int od)
{
    for (int b = 0; b < batch; ++b) {
        float* im_b = in_layer + b * (oh * ow * od);
        float* result_b = result + b * (oh * ow * od);

        pthread_t threads[P_THREADS];
        struct leaky_relu_thread_arg t_args[P_THREADS];
        
        int num_threads = MIN(P_THREADS, oh);
        int oh_part_size = oh / num_threads;

        t_args[0].im_b = im_b;
        t_args[0].result_b = result_b;
        t_args[0].ow = ow;
        t_args[0].od = od;

        int t_id;

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            if (t_idx > 0) {
                t_args[t_idx] = t_args[0];
            }

            int oh_s = oh_part_size * t_idx;
            int oh_e = t_idx < num_threads - 1 ? oh_s + oh_part_size : oh;
            
            t_args[t_idx].oh_s = oh_s;
            t_args[t_idx].oh_e = oh_e;

            t_id = pthread_create(&threads[t_idx], NULL, leaky_relu_thread_func, (void*) &t_args[t_idx]);
            if (t_id < 0) {
                perror("bias add thread error : ");
                exit(0);
            }
        }

        for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
            pthread_join(threads[t_idx], NULL);
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