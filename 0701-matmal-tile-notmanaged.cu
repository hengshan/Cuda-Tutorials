#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

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
    //dot product
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

  //shared memory
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  int numTiles = (K + TILE_SIZE -1)/TILE_SIZE;

  for (int t =0; t< numTiles; t++){
    // load A B to shared memory
    int aCol = t * TILE_SIZE +tx;
    if (row< M && aCol<K){
      As[ty][tx] = A[row * K +aCol];
    } else{
      As[ty][tx] = 0.0f;
    }

    int bRow = t * TILE_SIZE +ty;
    if (col< N && bRow<K){
      Bs[ty][tx] = B[bRow * N +col];
    } else{
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();
    //step2: 使用shared memory to calculate tiles value
    for (int k =0; k<TILE_SIZE; k++){
      sum += As[ty][k]* Bs[k][tx];
    }

    __syncthreads();
  }

  //step3: write back
  if(row<M && col <N)
  {
    C[row*N +col] = sum;
  }
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

  //shared memory
  __shared__ float As[TILE_SIZE *4][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE*4];

  float sum[4][4] = {{0.0f}};
  int numTiles = (K + TILE_SIZE -1)/TILE_SIZE;

  for (int t =0; t< numTiles; t++){
    // load A B to shared memory
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

    //step step2
    for ( int k = 0; k< TILE_SIZE; k++){
      float a_reg[4], b_reg[4];
      for (int i =0; i< 4; i++){
        a_reg[i] =As[ty*4+i][k];
      }

      for (int j =0; j< 4; j++){
        b_reg[j] =Bs[k][tx*4 +j];
      }

      for (int i =0; i<4; i++){
        for (int j=0; j<4; j++){
          sum[i][j]+=a_reg[i]*b_reg[j];
        }
      }
    }

    block.sync();
  }

  //step3: write back

  for (int i =0; i<4; i++){
    for (int j=0; j<4; j++){
      int c_row = block_row + ty*4+i;
      int c_col = block_col + tx*4+j;
      if(c_row<M && c_col <N)
        C[c_row*N + c_col] = sum[i][j];
      }
  }
}
void initMatRandom(float *mat, int rows, int cols)
{
  for(int i =0; i< rows*cols; i++)
  {
    //generate a random number between 0-1
    mat[i] = (float)rand()/RAND_MAX;
  }
}

bool verifyResult(float *C_cpu, float *C_gpu, int M, int N)
{
  const float epsilon = 1e-3;
  int errCount =0;
  for (int i =0; i < M*N; i++){
    float diff = fabs(C_cpu[i]- C_gpu[i]);
    if (diff>epsilon){
      errCount++;
    }
  }

  if (errCount>0){
    printf("oops we found %d error \n", errCount);
    return false;
  }

  return true;
}
// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("========================================\n");
    printf("第5课：Naive 矩阵相乘 (使用 cudaMalloc)\n");
    printf("========================================\n\n");

    // -------------------------------------------------
    // 准备数据
    // -------------------------------------------------
    int M = 1024;
    int K = 1024;
    int N = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // -------------------------------------------------
    // 分配 Host 内存
    // -------------------------------------------------
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_cpu = (float*)malloc(size_C);
    float *h_C_gpu = (float*)malloc(size_C);  // 用于从GPU拷贝回来
    float *h_C_gpu_tiled = (float*)malloc(size_C);  // 用于从GPU拷贝回来
    float *h_C_gpu_tiled4 = (float*)malloc(size_C);  // 用于从GPU拷贝回来

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu|| !h_C_gpu_tiled|| !h_C_gpu_tiled4) {
        fprintf(stderr, "Host内存分配失败!\n");
        exit(1);
    }

    // -------------------------------------------------
    // 分配 Device 内存
    // -------------------------------------------------
    float *d_A, *d_B, *d_C, *d_C_tiled,*d_C_tiled4;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_tiled4, size_C));

    // -------------------------------------------------
    // 初始化数据 (在Host上)
    // -------------------------------------------------
    srand(time(NULL));
    initMatRandom(h_A, M, K);
    initMatRandom(h_B, K, N);

    printf("预计 A[0]: %.2f\n", h_A[0]);
    printf("预计 B[0]: %.2f\n", h_B[0]);

    // -------------------------------------------------
    // 拷贝数据到 Device
    // -------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // -------------------------------------------------
    // CPU基准测试
    // -------------------------------------------------
    printf("CPU计算中...\n");
    clock_t cpu_start = clock();
    matmulCPU(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    printf("CPU耗时: %.2f ms\n\n", cpu_time);

    // -------------------------------------------------
    // GPU naive实现
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

    // -------------------------------------------------
    // 拷贝结果回 Host
    // -------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

    // -------------------------------------------------
    // 验证结果
    // -------------------------------------------------
    if (verifyResult(h_C_cpu, h_C_gpu, M, N)) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 出错，请检查结果\n");
    }

    // -------------------------------------------------
    // GPU tiled实现
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
    printf("加速比(vs CPU): %.2fx\n\n", cpu_time / time_tiled);
    printf("加速比(vs naive): %.2fx\n\n", time_naive / time_tiled);

    // -------------------------------------------------
    // 拷贝结果回 Host
    // -------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled, d_C_tiled, size_C, cudaMemcpyDeviceToHost));

    // -------------------------------------------------
    // 验证结果
    // -------------------------------------------------
    if (verifyResult(h_C_cpu, h_C_gpu_tiled, M, N)) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 出错，请检查结果\n");
    }


    // -------------------------------------------------
    // GPU tiled shared memory col or row *4实现
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU tiled 4*4实现: \n");
    printf("----------------------------------------\n");

    dim3 blockDim_cg(16, 16);
    dim3 gridDim_cg((N + 63) / 64,
                 (M + 63) / 64);
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
    printf("加速比(vs CPU): %.2fx\n\n", cpu_time / time_tiled4);
    printf("加速比(vs naive): %.2fx\n\n", time_naive / time_tiled4);
    printf("加速比(vs tiled): %.2fx\n\n", time_tiled / time_tiled4);

    // -------------------------------------------------
    // 拷贝结果回 Host
    // -------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled4, d_C_tiled4, size_C, cudaMemcpyDeviceToHost));

    // -------------------------------------------------
    // 验证结果
    // -------------------------------------------------
    if (verifyResult(h_C_cpu, h_C_gpu_tiled4, M, N)) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 出错，请检查结果\n");
    }
    // -------------------------------------------------
    // 清理 Device 内存
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_tiled));
    CUDA_CHECK(cudaFree(d_C_tiled4));

    // -------------------------------------------------
    // 清理 Host 内存
    // -------------------------------------------------
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_C_gpu_tiled);
    free(h_C_gpu_tiled4);

    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(stop3));
    return 0;
}
