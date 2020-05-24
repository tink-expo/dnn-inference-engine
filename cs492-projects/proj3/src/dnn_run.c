#include "cblas.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"

#define MAX(x, y) (x >= y ? x : y)

void bias_add(float* in_layer, float* biases, float* result,
        int ob, int oh, int ow, int od)
{
    for (int b = 0; b < ob; ++b) {
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

void bias_add_cb(float* in_layer, float* biases, float* result,
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

void conv2d(float* in_layer, 
        float* kernel, 
        float* result,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw)
{
    if (od == 125) {
        for (int c = 0; c < 3; ++c) {
            for (int d = 0; d < 3; ++d) {
                printf("%f ", kernel[c * od + d]);
            }
            printf("\n");
        }
    }

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
                // if (od == 125 && c < 4 && d == 0) {
                //     for (int pi = 0; pi < 4; ++pi) {
                //         for (int pj = 0; pj < 4; ++pj) {
                //             printf("%f ", result[pi * (ow * od) + pj * od]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }
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


/*
void max_pool2d(float* in_layer,
        float* result,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kh, int kw,
        int sh, int sw,
        int pt, int pb, int pl, int pr)
{
    int p_ih = pt + pb + ih;
    int p_iw = pl + pr + iw;
    float* pad_in = (float*) calloc(batch * p_ih * p_iw * ic, sizeof(float));

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < ih; ++i) {
            for (int j = 0; j < iw; ++j) {
                for (int c = 0; c < ic; ++c) {
                    pad_in[
                        b * (p_ih * p_iw * ic) +
                        (pt + i) * (p_iw * ic) +
                        (pl + j) * ic +
                        c
                    ] = in_layer[
                        b * (ih * iw * ic) +
                        i * (iw * ic) +
                        j * ic +
                        c
                    ];
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    int in_i = i * sh;
                    int in_j = j * sw;
                    float imax = pad_in[
                            b * (p_ih * p_iw * ic) +
                            in_i * (p_iw * ic) +
                            in_j * ic +
                            d
                    ];
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int ii = b * (p_ih * p_iw * ic) +
                                    (in_i + di) * (p_iw * ic) +
                                    (in_j + dj) * ic +
                                    d;
                            imax = MAX(imax, pad_in[ii]);
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
*/

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