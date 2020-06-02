/*
 * Reference:
 * https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
 */

#include <immintrin.h>
#include <stdio.h>
#include <string.h>

union uv256 {
    __m256 av;
    float r[8];
};

void avx(float* a, float* b, float* r, int n)
{
    // __m256 va = _mm256_loadu_ps(a);
    // __m256 vb = _mm256_loadu_ps(b);
    // union uv256 vr;
    // vr.av = _mm256_add_ps(va, vb);
    // memcpy(r, vr.r, 8 * sizeof(float));

    int dn = n / 8;
    __m256 va[dn];
    __m256 vb[dn];
    union uv256 vr[dn];
    for (int i = 0; i < dn; ++i) {
        va[i] = _mm256_loadu_ps(a + 8 * i);
        vb[i] = _mm256_loadu_ps(b + 8 * i);
        vr[i].av = _mm256_add_ps(va[i], vb[i]);
        memcpy(r + 8 * i, vr[i].r, 8 * sizeof(float));
    }
}

int main()
{
    ecuda(4);
    // Multiply 8 floats at a time
    __m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
    __m256 odds  = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
    __m256i zeros = _mm256_setzero_si256();
    __m256 res   = _mm256_set1_ps(1.0);

    printf("evens: ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", *(float *)&evens[i]);
    printf("\nodds:  ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", *(float *)&odds[i]);
    printf("\nres:   ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", *(float *)&res[i]);
    printf("\n");

    // float a[24];
    // float b[24];
    // for (int i = 0; i < 24; ++i) {
    //     a[i] = 1.0;
    //     b[i] = 2.0;
    // }
    // float r[24] = {0};

    // avx(a, b, r, 24);
    // for (int i = 0; i < 24; ++i) {
    //     printf("%.3f ", r[i]);
    // }
    // printf("\n");
}
