#include "cblas.h"
#include "string.h"
#include "stdio.h"

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

void conv2d(float* in_layer, 
        float* kernel, 
        float* result,
        int batch, int oh, int ow, int od,
        int ih, int iw, int ic,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int RS, int IS, int KS)
{
    printf("CONV2D %d %d %d\n", oh, ow, od);

    printf("%d %d\n", iw, ic);
    for (int i = 0; i < 20; ++i) {
        printf("%f ", in_layer[2 * (iw * ic) + 2 * (ic) + i]);
    }
    printf("\n");
    
    int count = 0;
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int c = 0; c < ic; ++c) {
                for (int i = 0; i < oh; ++i) {
                    for (int j = 0; j < ow; ++j) {
                        for (int di = 0; di < kernel_h; ++di) {
                            for (int dj = 0; dj < kernel_w; ++dj) {
                                int ri = b * (oh * ow * od) +
                                        i * (ow * od) +
                                        j * (od) +
                                        d;
                                if (ri >= RS) {
                                    printf("R\n");
                                    printf("%d %d %d %d\n", b, i, j, d);
                                    return;
                                }
                                int ii = b * (ih * iw * ic) +
                                        (stride_h * i + di) * (iw * ic) +
                                        (stride_w * j + dj) * ic +
                                        c;
                                if (ii >= IS) {
                                    printf("I\n");
                                    printf("%d %d %d %d\n", b,stride_h * i + di, stride_w * j + dj, c);
                                    return;
                                }
                                int ki = di * (kernel_w * ic * od) +
                                        dj * (ic * od) +
                                        c * od +
                                        d;
                                if (ki >= KS) {
                                    printf("K\n");
                                    printf("%d %d %d %d\n", di, dj, c, d);
                                    return;
                                }


                                // if (ri == 0) {
                                //     printf("%f %f %f\n", result[ri], in_layer[ii], kernel[ki]);
                                //     if (in_layer[ii] < -1000.0f) {
                                //         printf("HERE\n");
                                //         printf("%d %d %d %d\n", b,stride_h * i + di, stride_w * j + dj, c);
                                //         printf("%d %d %d %d %d %d\n", stride_h, i, di, stride_w, j, dj);
                                //         return;
                                //     }
                                // }
                                
                                result[
                                        b * (oh * ow * od) +
                                        i * (ow * od) +
                                        j * (od) +
                                        d
                                ] += (
                                in_layer[
                                        b * (ih * iw * ic) +
                                        (stride_h * i + di) * (iw * ic) +
                                        (stride_w * j + dj) * ic +
                                        c
                                ] * kernel[
                                        di * (kernel_w * ic * od) +
                                        dj * (ic * od) +
                                        c * od +
                                        d
                                ]);
                            }
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