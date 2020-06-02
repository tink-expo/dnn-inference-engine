//#include "mav.h"
#include "stdio.h"
#include "immintrin.h"

__global__ void cu_f(float* s)
{
    s[0] = 3.0f;
}

extern "C" {
void mf(float* s)
{
    // float s[8];
    // for (int i = 0; i < 8; ++i) {
    //     s[i] = 0.0f;
    // }
    // avx_f(s);
    for (int i = 0; i < 8; ++i) {
        printf("%f ", s[i]);
    }
    printf("\n");
    __m256 v = _mm256_set1_ps(1.0f);
    _mm256_storeu_ps(s, v);
    for (int i = 0; i < 8; ++i) {
        printf("%f ", s[i]);
    }
    printf("\n");

    float* d;
    cudaMalloc((void**) &d, sizeof(float) * 1);
    dim3 gd(3, 3);
    dim3 bd(3, 3);
    cu_f<<<gd, bd>>>(d);
    cudaMemcpy(s, d, sizeof(float) * 1, cudaMemcpyDeviceToHost);
    cudaFree(d);
    printf("%f\n", s[0]);
}
}

// int main()
// {
//     mf();
// }