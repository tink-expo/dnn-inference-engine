#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"

#define CUDA_THREADS_2D 16
#define CUDA_THREADS_1D 256

#define MAX(x, y) (x >= y ? x : y)

#define P_THREADS 4

__global__ void h_cuda_im2col(float* im_b, float* col_b,
        int oh, int ow,
        int iw, int ic,
        int kh, int kw, 
        int sh, int sw)
{
    int col_w = ic * kh * kw;
    int col_i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_i_idx < oh * ow) {
        int patch_i = (col_i_idx / ow) * sh;
        int patch_j = (col_i_idx % ow) * sw;
        for (int c = 0; c < ic; ++c) {
            int col_j = c * (kh * kw);
            for (int di = 0; di < kh; ++di) {
                for (int dj = 0; dj < kw; ++dj) {
                    col_b[col_i_idx * col_w +
                            col_j + (di * kw) + dj] = 
                            im_b[(patch_i + di) * (iw * ic) +
                            (patch_j + dj) * ic +
                            c];
                }
            }
        }
    }
}

__global__ void h_cuda_matmul(float* imcol, float* kernel, float* result, 
        int m_size, int n_size, int k_size)
{
    int i_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx < m_size && j_idx < n_size) {
        float res = 0.0f;
        for (int k = 0; k < k_size; ++k) {
            res += imcol[i_idx * k_size + k] * kernel[k * n_size + j_idx];
        }
        result[i_idx * n_size + j_idx] = res;
    }
}

__global__ void h_cuda_batch_norm(float* alpha, float* beta, float* result,
        int r_size, int od)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < r_size) {
        int d = t_idx % od;
        result[t_idx] = result[t_idx] * alpha[d] - beta[d];
    }
}

__global__ void h_cuda_batch_norm2(float* in_layer, float* alpha, float* beta, float* result,
    int r_size, int od)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < r_size) {
        int d = t_idx % od;
        result[t_idx] = in_layer[t_idx] * alpha[d] - beta[d];
    }
}

__global__ void h_cuda_max_pool2d(
        float* in_layer, float* result,
        int r_size, 
        int oh, int ow, int od, 
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < r_size) {
        // Calc i, j, d.
        int d = t_idx;
        int i = d / (ow * od);
        d -= i * (ow * od);
        int j = d / od;
        d -= j * od;
        
        int ii = (i * sh) * (iw * ic) + (j * sw) * ic + d;
        
        float imax = in_layer[ii];
        for (int di = 0; di < kh; ++di) {
            for (int dj = 0; dj < kw; ++dj) {
                if (di > 0 || dj > 0) {
                    imax = MAX(imax, 
                            in_layer[ii + di * (iw * ic) + dj * ic]);
                }
            }
        }
        result[t_idx] = imax;
    }
}

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

struct shape_arg {
    int batch, oh, ow, od;
    int ih, iw, ic;
    int kh, kw;
    int sh, sw;
};

struct im2col_thread_arg {
    float* im_b;
    float* col_b;
    struct shape_arg* shape;
    int oh_s;
    int oh_e;
};

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
                                arg->im_b[(patch_i + di) * (shape->iw * shape->ic) +
                                (patch_j + dj) * shape->ic +
                                c];
                    }
                }
            }
        }
    }

    return 0;
}

extern "C" {

void conv2d_cuda_pthread(float* in_layer,
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
        int oh_part_size = shape->oh / P_THREADS;
        

        t_args[0].im_b = im_b;
        t_args[0].col_b = col_b;
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

        float* d_imcol;
        float* d_kernel;
        float* d_result;
        cudaMalloc((void **) &d_imcol, sizeof(float) * m_size * k_size);
        cudaMalloc((void **) &d_kernel, sizeof(float) * k_size * n_size);
        cudaMalloc((void **) &d_result, sizeof(float) * m_size * k_size);

        cudaMemcpy(d_imcol, col_b, sizeof(float) * m_size * k_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel_r, sizeof(float) * k_size * n_size, cudaMemcpyHostToDevice);
        
        // TODO: Optimize here for Yolov2tiny size
        unsigned int grid_r = (m_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
        unsigned int grid_c = (n_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
        dim3 grid_dim(grid_c, grid_r);
        dim3 block_dim(CUDA_THREADS_2D, CUDA_THREADS_2D);

        h_cuda_matmul<<<grid_dim, block_dim>>>(d_imcol, d_kernel, d_result, m_size, n_size, k_size);
        cudaFree(d_imcol);
        cudaFree(d_kernel);

        cudaMemcpy(result_b, d_result, sizeof(float) * m_size * n_size, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
    }
}

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

        float* d_imcol;
        float* d_kernel;
        float* d_result;
        cudaMalloc((void **) &d_imcol, sizeof(float) * m_size * k_size);
        cudaMalloc((void **) &d_kernel, sizeof(float) * k_size * n_size);
        cudaMalloc((void **) &d_result, sizeof(float) * m_size * k_size);

        cudaMemcpy(d_imcol, col_b, sizeof(float) * m_size * k_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel_r, sizeof(float) * k_size * n_size, cudaMemcpyHostToDevice);
        
        // TODO: Optimize here for Yolov2tiny size
        unsigned int grid_r = (m_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
        unsigned int grid_c = (n_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
        dim3 grid_dim(grid_c, grid_r);
        dim3 block_dim(CUDA_THREADS_2D, CUDA_THREADS_2D);

        h_cuda_matmul<<<grid_dim, block_dim>>>(d_imcol, d_kernel, d_result, m_size, n_size, k_size);
        cudaFree(d_imcol);
        cudaFree(d_kernel);

        cudaMemcpy(result_b, d_result, sizeof(float) * m_size * n_size, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
    }
}


void conv2d_cuda_im2col_cuda(float* in_layer,
    float* col,
    float* kernel_r, 
    float* result,
    int batch, int oh, int ow, int od,
    int ih, int iw, int ic,
    int kh, int kw,
    int sh, int sw)
{
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
    
    int im_size = ih * iw * ic;
    int m_size = oh * ow;
    int n_size = od;
    int k_size = ic * kh * kw;
    
    float* d_im;
    float* d_col;
    float* d_kernel;
    float* d_result;
    cudaMalloc((void **) &d_im, sizeof(float) * im_size);
    cudaMalloc((void **) &d_col, sizeof(float) * m_size * k_size);
    cudaMemcpy(d_im, im_b, sizeof(float) * im_size, cudaMemcpyHostToDevice);

    unsigned int grid_m = (m_size + CUDA_THREADS_1D - 1) / CUDA_THREADS_1D;
    dim3 grid_m_dim(grid_m);
    dim3 block_m_dim(CUDA_THREADS_1D);

    h_cuda_im2col<<<grid_m_dim, block_m_dim>>>(d_im, d_col,
            oh, ow, iw, ic, kh, kw, sh, sw);
    cudaFree(d_im);

    cudaMalloc((void **) &d_kernel, sizeof(float) * k_size * n_size);
    cudaMalloc((void **) &d_result, sizeof(float) * m_size * k_size);
    cudaMemcpy(d_kernel, kernel_r, sizeof(float) * k_size * n_size, cudaMemcpyHostToDevice);
    
    // TODO: Optimize here for Yolov2tiny size
    unsigned int grid_r = (m_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
    unsigned int grid_c = (n_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
    dim3 grid_dim(grid_c, grid_r);
    dim3 block_dim(CUDA_THREADS_2D, CUDA_THREADS_2D);

    h_cuda_matmul<<<grid_dim, block_dim>>>(d_col, d_kernel, d_result, m_size, n_size, k_size);
    cudaFree(d_col);
    cudaFree(d_kernel);

    cudaMemcpy(result_b, d_result, sizeof(float) * m_size * n_size, cudaMemcpyDeviceToHost);
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
    memcpy(result, in_layer, sizeof(float) * r_size); 

    float* d_alpha;
    float* d_beta;
    float* d_result;

    cudaMalloc((void **) &d_alpha, sizeof(float) * od);
    cudaMalloc((void **) &d_beta, sizeof(float) * od);
    cudaMalloc((void **) &d_result, sizeof(float) * r_size);

    cudaMemcpy(d_result, result, sizeof(float) * r_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha, sizeof(float) * od, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, sizeof(float) * od, cudaMemcpyHostToDevice);
    
    unsigned int grid_size = (r_size + CUDA_THREADS_1D - 1) / CUDA_THREADS_1D;
    dim3 grid_dim(grid_size);
    dim3 block_dim(CUDA_THREADS_1D);

    h_cuda_batch_norm<<<grid_dim, block_dim>>>(d_alpha, d_beta, d_result, r_size, od);
    cudaFree(d_alpha);
    cudaFree(d_beta);

    cudaMemcpy(result, d_result, sizeof(float) * r_size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

void batch_norm_cuda2(float* in_layer,
    float* alpha,
    float* beta,
    float* result,
    int batch, int oh, int ow, int od)
{
    int r_size = batch * oh * ow * od;

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
    
    unsigned int grid_size = (r_size + CUDA_THREADS_1D - 1) / CUDA_THREADS_1D;
    dim3 grid_dim(grid_size);
    dim3 block_dim(CUDA_THREADS_1D);

    h_cuda_batch_norm2<<<grid_dim, block_dim>>>(d_in_layer, d_alpha, d_beta, d_result, r_size, od);
    cudaFree(d_in_layer);
    cudaFree(d_alpha);
    cudaFree(d_beta);

    cudaMemcpy(result, d_result, sizeof(float) * r_size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
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
                for (int d = 0; d < od; ++d) {
                    int ii = (i * sh) * (iw * ic) + (j * sw) * ic + d;
                    float imax = in_layer[ii];
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            if (di > 0 || dj > 0) {
                                imax = MAX(imax, 
                                        in_layer[ii + di * (iw * ic) + dj * ic]);
                            }
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

void max_pool2d_test(float* in_layer,
    float* result,
    int batch, int oh, int ow, int od,
    int ih, int iw, int ic,
    int kh, int kw,
    int sh, int sw)
{
    for (int b = 0; b < batch; ++b) {
        int r_size = oh * ow * od;

        float* in_layer_b = in_layer + b * (ih * iw * ic);
        float* result_b = result + b * (oh * ow * od);

        for (int t_idx = 0; t_idx < r_size; ++t_idx) {
            int d = t_idx;
            int i = d / (ow * od);
            d -= i * (ow * od);
            int j = d / od;
            d -= j * od;

            int ii = (i * sh) * (iw * ic) + (j * sw) * ic + d;
            float imax = in_layer_b[ii];
            for (int di = 0; di < kh; ++di) {
                for (int dj = 0; dj < kw; ++dj) {
                    if (di > 0 || dj > 0) {
                        imax = MAX(imax, 
                                in_layer_b[ii + di * (iw * ic) + dj * ic]);
                    }
                }
            }
            result_b[t_idx] = imax;
        }
    }
}

void max_pool2d_cuda(float* in_layer,
    float* result,
    int batch, int oh, int ow, int od,
    int ih, int iw, int ic,
    int kh, int kw,
    int sh, int sw)
{
    for (int b = 0; b < batch; ++b) {
        int r_size = oh * ow * od;
        int i_size = ih * iw * ic;

        float* in_layer_b = in_layer + b * (ih * iw * ic);
        float* result_b = result + b * (oh * ow * od);
        
        float* d_in_layer;
        float* d_result;

        cudaMalloc((void **) &d_in_layer, sizeof(float) * i_size);
        cudaMalloc((void **) &d_result, sizeof(float) * r_size);

        cudaMemcpy(d_in_layer, in_layer_b, sizeof(float) * i_size, cudaMemcpyHostToDevice);

        unsigned int grid_size = (r_size + CUDA_THREADS_1D - 1) / CUDA_THREADS_1D;
        dim3 grid_dim(grid_size);
        dim3 block_dim(CUDA_THREADS_1D);

        h_cuda_max_pool2d<<<grid_dim, block_dim>>>(
                d_in_layer, d_result, 
                r_size,
                oh, ow, od, 
                ih, iw, ic, 
                kh, kw,
                sh, sw);
        cudaFree(d_in_layer);

        cudaMemcpy(result_b, d_result, sizeof(float) * r_size, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
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