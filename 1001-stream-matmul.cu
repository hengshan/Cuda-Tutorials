//第10课：CUDA Streams 与异步操作与第11课 cuda graph的代码都在这里

#include <cstdio>
#include <cstdlib>
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

//CG优化
__global__ void matmul_tile_4x4(float *A, float *B, float *C, int M, int N, int K) {
    cg::thread_block block = cg::this_thread_block();

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * TILE_SIZE * 4;
    int block_col = blockIdx.x * TILE_SIZE * 4;

    __shared__ float As[TILE_SIZE * 4][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * 4];

    float sum[4][4] = {{0.0f}};

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tile_k = t * TILE_SIZE;

        for (int i = 0; i < 4; i++) {
            int aRow = block_row + ty * 4 + i;
            int aCol = tile_k + tx;
            As[ty * 4 + i][tx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        }

        for (int j = 0; j < 4; j++) {
            int bRow = tile_k + ty;
            int bCol = block_col + tx * 4 + j;
            Bs[ty][tx * 4 + j] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        }

        block.sync();

        for (int k = 0; k < TILE_SIZE; k++) {
            float a_reg[4], b_reg[4];

            for (int i = 0; i < 4; i++) {
                a_reg[i] = As[ty * 4 + i][k];
            }

            for (int j = 0; j < 4; j++) {
                b_reg[j] = Bs[k][tx * 4 + j];
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    sum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        block.sync();
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int c_row = block_row + ty * 4 + i;
            int c_col = block_col + tx * 4 + j;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = sum[i][j];
            }
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
    // int M = 1024;
    // int K = 1024;
    // int N = 1024;

    int M = 128;
    int K = 128;
    int N = 128;
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
    float *h_C_gpu_cg = (float*)malloc(size_C);  // 用于从GPU拷贝回来

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu|| !h_C_gpu_tiled|| !h_C_gpu_cg) {
        fprintf(stderr, "Host内存分配失败!\n");
        exit(1);
    }

    // -------------------------------------------------
    // 分配 Device 内存
    // -------------------------------------------------
    float *d_A, *d_B, *d_C, *d_C_tiled, *d_C_cg;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_cg, size_C));

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
    // GPU cg实现
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU CG实现: \n");
    printf("----------------------------------------\n");

    // 关键修复：CG 4x4 每个 block 处理 64×64 输出
    dim3 blockDim_cg(16, 16);
    dim3 gridDim_cg((N + 63) / 64,    // = 16
                    (M + 63) / 64);   // = 16
    cudaEvent_t start3, stop3;
    CUDA_CHECK(cudaEventCreate(&start3));
    CUDA_CHECK(cudaEventCreate(&stop3));

    CUDA_CHECK(cudaEventRecord(start3));
    matmul_tile_4x4<<<gridDim_cg, blockDim_cg>>>(d_A, d_B, d_C_cg, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop3));
    CUDA_CHECK(cudaEventSynchronize(stop3));

    float time_cg;
    CUDA_CHECK(cudaEventElapsedTime(&time_cg, start3, stop3));

    printf("GPU CG 耗时: %.2f ms\n", time_cg);
    printf("加速比(vs CPU): %.2fx\n\n", cpu_time / time_cg);
    printf("加速比(vs naive): %.2fx\n\n", time_naive / time_cg);

    // -------------------------------------------------
    // 拷贝结果回 Host
    // -------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C_gpu_cg, d_C_cg, size_C, cudaMemcpyDeviceToHost));

    // -------------------------------------------------
    // 验证结果
    // -------------------------------------------------
    if (verifyResult(h_C_cpu, h_C_gpu_cg, M, N)) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 出错，请检查结果\n");
    }

    //批量矩阵的乘法 同步vs 异步的对比
    // const int NUM_BATCHES = 8;
    const int NUM_BATCHES = 1000;

    float *h_A_batch[NUM_BATCHES], *h_C_batch[NUM_BATCHES];
    float *d_A_batch[NUM_BATCHES], *d_C_batch[NUM_BATCHES];

    float *h_B_shared;
    CUDA_CHECK(cudaMallocHost(&h_B_shared, size_B));  // pinned memory
    float *d_B_shared;
    CUDA_CHECK(cudaMalloc(&d_B_shared, size_B));
    initMatRandom(h_B_shared, K, N);

    for(int i =0; i< NUM_BATCHES; i++){
      CUDA_CHECK(cudaMallocHost(&h_A_batch[i], size_A));  // pinned memory
      CUDA_CHECK(cudaMallocHost(&h_C_batch[i], size_C));  // pinned memory

      cudaMalloc(&d_A_batch[i], size_A);
      cudaMalloc(&d_C_batch[i], size_C);

      initMatRandom(h_A_batch[i], M, K);
    }

    //method 1: sync
    cudaEvent_t sync_start, sync_stop;
    CUDA_CHECK(cudaEventCreate(&sync_start));
    CUDA_CHECK(cudaEventCreate(&sync_stop));

    CUDA_CHECK(cudaEventRecord(sync_start));

    cudaMemcpy(d_B_shared, h_B_shared, size_B, cudaMemcpyHostToDevice);

    for(int i =0; i< NUM_BATCHES; i++){
      cudaMemcpy(d_A_batch[i], h_A_batch[i], size_A, cudaMemcpyHostToDevice);
      matmul_tile_4x4<<<gridDim_cg, blockDim_cg>>>(d_A_batch[i], d_B_shared, d_C_batch[i], M, N, K);
      cudaMemcpy(h_C_batch[i], d_C_batch[i], size_C, cudaMemcpyDeviceToHost);
    }
    CUDA_CHECK(cudaEventRecord(sync_stop));
    CUDA_CHECK(cudaEventSynchronize(sync_stop));

    float time_sync;
    CUDA_CHECK(cudaEventElapsedTime(&time_sync, sync_start, sync_stop));

    printf("GPU Batch总 耗时: %.2f ms\n", time_sync);
    printf("GPU Batch each one 耗时: %.2f ms\n", time_sync/NUM_BATCHES);

    //method 2: async
    // cudaStream_t streams[NUM_BATCHES];
    const int STREAM_POOL_SIZE = 8;
    cudaStream_t stream_pool[STREAM_POOL_SIZE];
    for(int i =0; i< STREAM_POOL_SIZE; i++){
      cudaStreamCreate(&stream_pool[i]);
    }

    cudaEvent_t async_start, async_stop;
    CUDA_CHECK(cudaEventCreate(&async_start));
    CUDA_CHECK(cudaEventCreate(&async_stop));

    CUDA_CHECK(cudaEventRecord(async_start));

    cudaMemcpy(d_B_shared, h_B_shared, size_B, cudaMemcpyHostToDevice);

    for(int i =0; i< NUM_BATCHES; i++){
      int stream_idx = i % STREAM_POOL_SIZE;  // 轮流使用stream
      cudaMemcpyAsync(d_A_batch[i], h_A_batch[i], size_A, cudaMemcpyHostToDevice, stream_pool[stream_idx]);
      matmul_tile_4x4<<<gridDim_cg, blockDim_cg, 0, stream_pool[stream_idx]>>>(d_A_batch[i], d_B_shared, d_C_batch[i], M, N, K);
      cudaMemcpyAsync(h_C_batch[i], d_C_batch[i], size_C, cudaMemcpyDeviceToHost, stream_pool[stream_idx]);
    }
    CUDA_CHECK(cudaEventRecord(async_stop));
    CUDA_CHECK(cudaEventSynchronize(async_stop));

    float time_async;
    CUDA_CHECK(cudaEventElapsedTime(&time_async, async_start, async_stop));

    printf("GPU Batch async总 耗时: %.2f ms\n", time_async);
    printf("GPU Batch async each one 耗时: %.2f ms\n", time_async/NUM_BATCHES);

    printf("GPU Batch sync vs async 耗时加速比: %.2f \n", time_sync/time_async);


    for(int i =0; i< STREAM_POOL_SIZE; i++){
        cudaStreamDestroy(stream_pool[i]);
    }

    // -------------------------------------------------
    // Cuda graph
    // -------------------------------------------------
    printf("方法4： Cuda Graph + stream \n\n");

    cudaStream_t graph_stream_pool[STREAM_POOL_SIZE];
    for(int i =0; i<STREAM_POOL_SIZE; i++){
      cudaStreamCreate(&graph_stream_pool[i]);
    }

    cudaGraph_t graph_par;
    cudaGraphExec_t graphExec_par;

    printf("正在capture cuda graph: ");
    // step 1: capture 捕捉stream
    cudaStreamBeginCapture(graph_stream_pool[0], cudaStreamCaptureModeGlobal);

    //step 2: fork-join
    cudaEvent_t fork_event;
    cudaEventCreate(&fork_event);
    cudaEventRecord(fork_event, graph_stream_pool[0]);

    for(int i =1; i< STREAM_POOL_SIZE; i++){
      cudaStreamWaitEvent(graph_stream_pool[i], fork_event, 0);
    }
    // step3: 使用stream 并行处理所有的batch
    for(int i =0; i< NUM_BATCHES; i++){
      int stream_idx = i % STREAM_POOL_SIZE;  // 轮流使用stream
      cudaMemcpyAsync(d_A_batch[i], h_A_batch[i], size_A, cudaMemcpyHostToDevice, graph_stream_pool[stream_idx]);
      matmul_tile_4x4<<<gridDim_cg, blockDim_cg, 0, graph_stream_pool[stream_idx]>>>(d_A_batch[i], d_B_shared, d_C_batch[i], M, N, K);
      cudaMemcpyAsync(h_C_batch[i], d_C_batch[i], size_C, cudaMemcpyDeviceToHost, graph_stream_pool[stream_idx]);
    }
    // step4: 创建join event, 回合所以streams
    cudaEvent_t join_event;
    cudaEventCreate(&join_event);
    for(int i =1; i<STREAM_POOL_SIZE; i++){
      cudaEventRecord(join_event, graph_stream_pool[i]);
      cudaStreamWaitEvent(graph_stream_pool[0], join_event, 0);
    }
    //step 5: 结束捕捉 实例化 graph
    cudaStreamEndCapture(graph_stream_pool[0], &graph_par);
    printf("正在实例化... \n\n");
    cudaGraphInstantiate(&graphExec_par, graph_par, nullptr, nullptr,0 );
    //step: launch， graph内部就是并行的

    cudaEvent_t graph_start, graph_stop;
    CUDA_CHECK(cudaEventCreate(&graph_start));
    CUDA_CHECK(cudaEventCreate(&graph_stop));

    CUDA_CHECK(cudaEventRecord(graph_start));

    cudaMemcpy(d_B_shared, h_B_shared, size_B, cudaMemcpyHostToDevice);

    cudaGraphLaunch(graphExec_par, graph_stream_pool[0]);
    CUDA_CHECK(cudaEventRecord(graph_stop));
    CUDA_CHECK(cudaEventSynchronize(graph_stop));

    float time_graph;
    CUDA_CHECK(cudaEventElapsedTime(&time_graph, graph_start, graph_stop));

    printf("GPU Graph总 耗时: %.2f ms\n", time_graph);
    printf("GPU graph each one 耗时: %.2f ms\n", time_graph/NUM_BATCHES);
    printf("GPU graph vs. stream 耗时: %.2f ms\n", time_async/time_graph);

    for(int i =0; i< STREAM_POOL_SIZE; i++){
        cudaStreamDestroy(graph_stream_pool[i]);
    }

    // -------------------------------------------------
    // 清理 Device 内存
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_tiled));
    CUDA_CHECK(cudaFree(d_C_cg));
    CUDA_CHECK(cudaFree(d_B_shared));

    // -------------------------------------------------
    // 清理 Host 内存
    // -------------------------------------------------
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_C_gpu_tiled);
    free(h_C_gpu_cg);
    // free(h_B_shared);
    cudaFreeHost(h_B_shared);  // 改为cudaFreeHost

    for(int i =0; i< NUM_BATCHES; i++){
      cudaFreeHost(h_A_batch[i]);
      cudaFreeHost(h_C_batch[i]);
      cudaFree(d_A_batch[i]);
      cudaFree(d_C_batch[i]);
    }

    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(stop3));

    CUDA_CHECK(cudaEventDestroy(sync_start));
    CUDA_CHECK(cudaEventDestroy(sync_stop));
    CUDA_CHECK(cudaEventDestroy(async_start));
    CUDA_CHECK(cudaEventDestroy(async_stop));
    return 0;
}
