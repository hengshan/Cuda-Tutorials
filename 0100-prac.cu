#include <cstdio>
#include <stdio.h>

// 并发计算
__global__ void hello(){
  int idx = blockDim.x * blockIdx.x +  threadIdx.x;
  printf("hello world! %d \n", idx);
}

__global__ void add(float *d_result, float a, float b){
  int idx = blockDim.x * blockIdx.x +  threadIdx.x;
  *d_result = a+b;
  printf("hello world! %d \n", idx);
}

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    c[tid] = a[tid] + b[tid];
  }
}

int main()
{
  const int N = 256 * 1024;
  // const int N = 16;
  int h_a[N], h_b[N], h_c[N];

  // 初始化数组
  for (int i = 0; i < N; ++i)
  {
    h_a[i] = i;
    h_b[i] = i * 10;
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sizeof(int) * N);
  cudaMalloc(&d_b, sizeof(int) * N);
  cudaMalloc(&d_c, sizeof(int) * N);

  cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice);

  // kernel
  vectorAdd<<<256, 1024>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i)
  {
    printf("%d + %d = %d \n", h_a[i], h_b[i], h_c[i]);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
