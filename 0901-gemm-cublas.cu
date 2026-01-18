//增加怎么编译cublas的代码
// nvcc -arch=sm_120 -o lesson09 0901-gemm-cublas.cu -lcublas -lcudart
// ./lesson09
// 硬件要求：
// - Compute Capability 9.0+ (RTX 5090 = 12.0 ✓)
// - CUDA 11.0+

#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>            // fabs
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cublas_v2.h>       // ✅ 新增：cuBLAS

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS错误 %s:%d - status=%d\n", __FILE__, __LINE__, (int)status); \
            exit(1); \
        } \
    } while(0)

// =============================================================================
// CPU版本：用于验证
// =============================================================================
void matmulCPU(float *A, float *B,float *C, int M, int N, int K){
  for (int i=0; i< M; i++){
    for (int j=0; j< N; j++){
      float sum =0.0f;
      for(int k =0; k<K; k++){
        sum+=A[i *K +k] * B[k*N +j];
      }
      C[i*N + j] =sum;
    }
  }
}

// =============================================================================
// GPU版本：matmal naive
// =============================================================================
__global__ void matmalGPU_naive(float *A, float *B,float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col<N){
    float sum = 0.0f;
    for (int k =0; k<K; k++){
      sum += A[row*K +k] * B[k*N +col];
    }
    C[row*N + col] =sum;
  }
}

// =============================================================================
// GPU版本：matmal tiled
// =============================================================================
#define TILE_SIZE 16
__global__ void matmalGPU_tiled(float *A, float *B,float *C, int M, int N, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;

  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  int numTiles = (K + TILE_SIZE -1)/TILE_SIZE;

  for (int t =0; t< numTiles; t++){
    int aCol = t * TILE_SIZE +tx;
    As[ty][tx] = (row< M && aCol<K) ? A[row * K +aCol] : 0.0f;

    int bRow = t * TILE_SIZE +ty;
    Bs[ty][tx] = (col< N && bRow<K) ? B[bRow * N +col] : 0.0f;

    __syncthreads();

    for (int k =0; k<TILE_SIZE; k++){
      sum += As[ty][k]* Bs[k][tx];
    }
    __syncthreads();
  }

  if(row<M && col <N) C[row*N +col] = sum;
}

// =============================================================================
// GPU版本：matmal tiled shared memory length/width*4
// =============================================================================
__global__ void matmalGPU_tiled4(float *A, float *B,float *C, int M, int N, int K) {
  cg::thread_block block = cg::this_thread_block();
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_row = blockIdx.y * TILE_SIZE *4;
  int block_col = blockIdx.x * TILE_SIZE *4;

  __shared__ float As[TILE_SIZE *4][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE*4];

  float sum[4][4] = {{0.0f}};
  int numTiles = (K + TILE_SIZE -1)/TILE_SIZE;

  for (int t =0; t< numTiles; t++){
    int tile_k =t* TILE_SIZE;

    for (int i =0; i< 4; i++){
      int aRow = block_row + ty*4 +i;
      int aCol = tile_k +tx;
      As[ty*4 +i][tx] = (aRow<M && aCol<K)? A[aRow*K + aCol]: 0.0f;
    }

    for (int j =0; j< 4; j++){
      int bRow = tile_k +ty;
      int bCol = block_col + tx*4 +j;
      Bs[ty][tx*4 +j] = (bRow<K && bCol<N)? B[bRow*N + bCol]: 0.0f;
    }

    block.sync();

    for ( int k = 0; k< TILE_SIZE; k++){
      float a_reg[4], b_reg[4];
      for (int i =0; i< 4; i++) a_reg[i] =As[ty*4+i][k];
      for (int j =0; j< 4; j++) b_reg[j] =Bs[k][tx*4 +j];

      for (int i =0; i<4; i++){
        for (int j=0; j<4; j++){
          sum[i][j]+=a_reg[i]*b_reg[j];
        }
      }
    }

    block.sync();
  }

  for (int i =0; i<4; i++){
    for (int j=0; j<4; j++){
      int c_row = block_row + ty*4+i;
      int c_col = block_col + tx*4+j;
      if(c_row<M && c_col <N) C[c_row*N + c_col] = sum[i][j];
    }
  }
}

void initMatRandom(float *mat, int rows, int cols)
{
  for(int i =0; i< rows*cols; i++)
    mat[i] = (float)rand()/RAND_MAX;
}

bool verifyResult(float *C_cpu, float *C_gpu, int M, int N)
{
  const float epsilon = 1e-3;
  int errCount =0;
  for (int i =0; i < M*N; i++){
    float diff = fabs(C_cpu[i]- C_gpu[i]);
    if (diff>epsilon) errCount++;
  }
  if (errCount>0){
    printf("oops we found %d error \n", errCount);
    return false;
  }
  return true;
}

// =============================================================================
// ✅ 新增：cuBLAS SGEMM（适配 row-major 的 A/B/C）
// 解释：用 column-major 计算 C^T = B^T * A^T
// 在 cuBLAS 里等价调用：C(N×M) = B(N×K) * A(K×M) （都按column-major解释）
// 输出 buffer 拷回后，直接当 row-major 的 M×N 读取即可匹配 CPU 结果。
// =============================================================================
void matmulCUBLAS_rowmajor(float *d_A, float *d_B, float *d_C, int M, int N, int K)
{
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // cublasSgemm(handle, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  // 我们要计算：C_col (N×M) = B_col (N×K) * A_col (K×M)
  // 所以 m=N, n=M, k=K
  // B 的 leading dimension = N；A 的 leading dimension = K；C 的 leading dimension = N
  CUBLAS_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      d_B, N,   // A = B_col, lda = N
      d_A, K,   // B = A_col, ldb = K
      &beta,
      d_C, N    // C = C_col, ldc = N
  ));

  CUBLAS_CHECK(cublasDestroy(handle));
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("========================================\n");
    printf("第5课：Naive 矩阵相乘 (使用 cudaMalloc)\n");
    printf("========================================\n\n");

    int M = 1024;
    int K = 1024;
    int N = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_cpu = (float*)malloc(size_C);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_gpu_tiled = (float*)malloc(size_C);
    float *h_C_gpu_tiled4 = (float*)malloc(size_C);
    float *h_C_gpu_cublas = (float*)malloc(size_C);   // ✅ 新增：cuBLAS结果

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu|| !h_C_gpu_tiled|| !h_C_gpu_tiled4 || !h_C_gpu_cublas) {
        fprintf(stderr, "Host内存分配失败!\n");
        exit(1);
    }

    float *d_A, *d_B, *d_C, *d_C_tiled,*d_C_tiled4, *d_C_cublas; // ✅ 新增 d_C_cublas
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_tiled4, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, size_C)); // ✅ 新增

    srand(time(NULL));
    initMatRandom(h_A, M, K);
    initMatRandom(h_B, K, N);

    printf("预计 A[0]: %.2f\n", h_A[0]);
    printf("预计 B[0]: %.2f\n", h_B[0]);

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    printf("CPU计算中...\n");
    clock_t cpu_start = clock();
    matmulCPU(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    printf("CPU耗时: %.2f ms\n\n", cpu_time);

    // -------------------------------------------------
    // GPU naive
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU naive实现: \n");
    printf("----------------------------------------\n");

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    CUDA_CHECK(cudaEventRecord(start1));
    matmalGPU_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop1));
    CUDA_CHECK(cudaEventSynchronize(stop1));

    float time_naive;
    CUDA_CHECK(cudaEventElapsedTime(&time_naive, start1, stop1));

    printf("GPU naive 耗时: %.2f ms\n", time_naive);
    printf("加速比(vs CPU): %.2fx\n\n", cpu_time / time_naive);

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
    if (verifyResult(h_C_cpu, h_C_gpu, M, N)) printf("✓ 结果正确\n");
    else printf("✗ 出错，请检查结果\n");

    // -------------------------------------------------
    // GPU tiled
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU tiled实现: \n");
    printf("----------------------------------------\n");

    cudaEvent_t start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    CUDA_CHECK(cudaEventRecord(start2));
    matmalGPU_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C_tiled, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop2));
    CUDA_CHECK(cudaEventSynchronize(stop2));

    float time_tiled;
    CUDA_CHECK(cudaEventElapsedTime(&time_tiled, start2, stop2));

    printf("GPU tiled 耗时: %.2f ms\n", time_tiled);
    printf("加速比(vs CPU): %.2fx\n", cpu_time / time_tiled);
    printf("加速比(vs naive): %.2fx\n\n", time_naive / time_tiled);

    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled, d_C_tiled, size_C, cudaMemcpyDeviceToHost));
    if (verifyResult(h_C_cpu, h_C_gpu_tiled, M, N)) printf("✓ 结果正确\n");
    else printf("✗ 出错，请检查结果\n");

    // -------------------------------------------------
    // GPU tiled4
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU tiled 4*4实现: \n");
    printf("----------------------------------------\n");

    dim3 blockDim_cg(16, 16);
    dim3 gridDim_cg((N + 63) / 64, (M + 63) / 64);

    cudaEvent_t start3, stop3;
    CUDA_CHECK(cudaEventCreate(&start3));
    CUDA_CHECK(cudaEventCreate(&stop3));

    CUDA_CHECK(cudaEventRecord(start3));
    matmalGPU_tiled4<<<gridDim_cg, blockDim_cg>>>(d_A, d_B, d_C_tiled4, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop3));
    CUDA_CHECK(cudaEventSynchronize(stop3));

    float time_tiled4;
    CUDA_CHECK(cudaEventElapsedTime(&time_tiled4, start3, stop3));

    printf("GPU tiled 4*4 耗时: %.2f ms\n", time_tiled4);
    printf("加速比(vs CPU): %.2fx\n", cpu_time / time_tiled4);
    printf("加速比(vs naive): %.2fx\n", time_naive / time_tiled4);
    printf("加速比(vs tiled): %.2fx\n\n", time_tiled / time_tiled4);

    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled4, d_C_tiled4, size_C, cudaMemcpyDeviceToHost));
    if (verifyResult(h_C_cpu, h_C_gpu_tiled4, M, N)) printf("✓ 结果正确\n");
    else printf("✗ 出错，请检查结果\n");

    // -------------------------------------------------
    // ✅ cuBLAS SGEMM
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("cuBLAS SGEMM实现: \n");
    printf("----------------------------------------\n");
    // 建议：warm-up 一次避免首调用初始化开销影响计时
    matmulCUBLAS_rowmajor(d_A, d_B, d_C_cublas, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start4, stop4;
    CUDA_CHECK(cudaEventCreate(&start4));
    CUDA_CHECK(cudaEventCreate(&stop4));

    CUDA_CHECK(cudaEventRecord(start4));
    matmulCUBLAS_rowmajor(d_A, d_B, d_C_cublas, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop4));
    CUDA_CHECK(cudaEventSynchronize(stop4));

    float time_cublas;
    CUDA_CHECK(cudaEventElapsedTime(&time_cublas, start4, stop4));

    printf("cuBLAS SGEMM 耗时: %.2f ms\n", time_cublas);
    printf("加速比(vs CPU): %.2fx\n", cpu_time / time_cublas);
    printf("加速比(vs naive): %.2fx\n", time_naive / time_cublas);
    printf("加速比(vs tiled): %.2fx\n", time_tiled / time_cublas);
    printf("加速比(vs tiled4): %.2fx\n\n", time_tiled4 / time_cublas);

    CUDA_CHECK(cudaMemcpy(h_C_gpu_cublas, d_C_cublas, size_C, cudaMemcpyDeviceToHost));
    if (verifyResult(h_C_cpu, h_C_gpu_cublas, M, N)) printf("✓ 结果正确\n");
    else printf("✗ 出错，请检查结果\n");

    // -------------------------------------------------
    // 清理
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_tiled));
    CUDA_CHECK(cudaFree(d_C_tiled4));
    CUDA_CHECK(cudaFree(d_C_cublas));

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_C_gpu_tiled);
    free(h_C_gpu_tiled4);
    free(h_C_gpu_cublas);

    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(stop3));
    CUDA_CHECK(cudaEventDestroy(start4));
    CUDA_CHECK(cudaEventDestroy(stop4));

    return 0;
}

