// SM 利用率（sm__throughput.avg.pct_of_peak_sustained_elapsed）
// 显存带宽利用率（dram__throughput.avg.pct_of_peak_sustained_elapsed）
// Occupancy（占用率）（sm__warps_active.avg.pct_of_peak_sustained_active）
// Warp 执行效率（sm__sass_average_active_warps_per_cycle / sm__sass_average_threads_per_warp）
// L2 缓存利用率（lts__throughput.avg.pct_of_peak_sustained_elapsed）

#include <cstdio>
#include <nvtx3/nvToolsExt.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n){
    c[tid] = a[tid] + b[tid];
  }
}
__global__ void vectorAdd2(int *a, int *b, int *c, int n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n){
    c[tid] = a[tid] + b[tid];
  }
}

int main(){
  const int N = 256*1024;
  int h_a[N], h_b[N], h_c[N];

  nvtxRangePush("init array");
  for (int i =0; i<N; ++i){
    h_a[i] = i;
    h_b[i] = i * 10;
  }
  nvtxRangePop();

  int *d_a, *d_b, *d_c;
  nvtxRangePush("Start cudaMalloc");
  cudaMalloc(&d_a, sizeof(int)*N);
  cudaMalloc(&d_b, sizeof(int)*N);
  cudaMalloc(&d_c, sizeof(int)*N);
  nvtxRangePop();

  nvtxRangePush("Memcpy HtoD");
  cudaMemcpy(d_a, h_a, sizeof(int)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int)*N, cudaMemcpyHostToDevice);
  nvtxRangePop();

  nvtxRangePush("Kernel Launch: vectorAdd");
  vectorAdd<<<1024,256>>>(d_a, d_b, d_c, N);
  vectorAdd<<<1024 * 256 ,1>>>(d_a, d_b, d_c, N);
  vectorAdd<<<256,1024>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
  nvtxRangePop();

  nvtxRangePush("Memcpy DtoH");
  cudaMemcpy(h_c, d_c, sizeof(int)*N, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  nvtxRangePush("print array");
  for( int i =0; i< 100; ++i){
    printf("%d + %d = %d \n", h_a[i], h_b[i], h_c[i]);
  }
  nvtxRangePop();

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  nvtxMark("End of Program");
  return 0;
}

