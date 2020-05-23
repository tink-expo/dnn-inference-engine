#include "stdio.h"

void add_float_p(float a, float b, float *c)
{
    *c = a + b;
}

void print_p(float* a, int size)
{
    for (int i = 0; i < size; ++i) {
        printf("%.1f ", a[i]);
    }
    printf("\n");
}
