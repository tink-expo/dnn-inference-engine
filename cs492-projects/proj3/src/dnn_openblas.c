#include "cblas.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define MAX(x, y) (x >= y ? x : y)

void bias_add(float* in_layer, float* biases, float* result,
        int shape_b, int shape_h, int shape_w, int shape_d)
{
    cblas_scopy(
        shape_b * shape_h * shape_w * shape_d,
        in_layer,
        1,
        result,
        1
    );
    
    float bias_d[shape_h * shape_w];
    for (int i = 0; i < shape_h * shape_w; ++i) {
        bias_d[i] = 1.0f;
    }
    for (int b = 0; b < shape_b; ++b) {
        float* result_b = result + b * (shape_h * shape_w * shape_d);

        for (int d = 0; d < shape_d; ++d) {
            cblas_saxpy(
                shape_h * shape_w,
                biases[d],
                bias_d,
                1,
                result_b + d,
                shape_d
            );
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

void conv2d_arr(float* in_layer_arg, 
        float* kernel_arg, 
        float* result_arg,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw)
{
    float (*in_layer)[ih][iw][ic] = (float (*)[ih][iw][ic]) in_layer_arg;
    float (*kernel)[kw][ic][od] = (float (*)[kw][ic][od]) kernel_arg;
    float (*result)[oh][ow][od] = (float (*)[oh][ow][od]) result_arg;
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int c = 0; c < ic; ++c) {
                for (int i = 0; i < oh; ++i) {
                    for (int j = 0; j < ow; ++j) {
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                // int ri = b * (oh * ow * od) +
                                //         i * (ow * od) +
                                //         j * od +
                                //         d;
                                // int ii = b * (ih * iw * ic) +
                                //         (sh * i + di) * (iw * ic) +
                                //         (sw * j + dj) * ic +
                                //         c;
                                // int ki = di * (kw * ic * od) +
                                //         dj * (ic * od) +
                                //         c * od +
                                //         d;
                                
                                result[b][i][j][d] += 
                                        in_layer[b][sh * i + di][sw * j + dj][c] * 
                                        kernel[di][dj][c][d];
                            }
                        }
                    }
                }
            }
        }
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

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            (oh * ow), od, (ic * kh * kw),
            1,
            colb, (ic * kh * kw),
            kernel_r, od,
            0,
            resultb, od
        );
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