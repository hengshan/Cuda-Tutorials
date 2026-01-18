#include <cstdio>
#include <stdio.h>
#include <memory>

// CUDA runtime
#include <cuda_runtime.h>

__global__ void testKernel(float *d_result, float a, float b)
{
  //所有线程执行相同的操作
  *d_result = a+b;
  printf("GPU: %.1f + %.1f= %.1f \n ", a, b, *d_result);
}

int main(int argc, char **argv)
{
    float a = 5.5f, b =3.2f;
    float h_result;
    printf("a + b: %.1f + %.1f = %.1f\n", a, b, a+b);
    float *d_result = nullptr;
    printf("d_result 的值(未初始化): %p\n", (void*)d_result);

    cudaMalloc(&d_result, sizeof(float));
    printf("d_result 的地址(栈上指针变量): %p\n", (void*)&d_result);
    printf("d_result 的值(GPU的地址): %p\n", (void*)d_result);

    testKernel<<<2,2>>>(d_result, a, b);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("h_result 的地址(CPU): %p\n", (void*)&h_result);
    printf("GPU内存中存储的值传到CPU: %.1f\n", h_result);
    cudaFree(d_result);
    return 0;
}

