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
int findMaxCPU(int *data, int n) {
    int max_val = INT_MIN;
    for (int i = 0; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

// =============================================================================
// GPU版本1：第3课的简单实现（用于性能对比）
// =============================================================================
__global__ void matmalGPU_naive(int *data, int n, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_max = INT_MIN;
    for (int i = tid; i < n; i += stride) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    atomicMax(result, local_max);  // 所有线程都竞争这一个位置！
}

__global__ void findMaxGPU_shared(int *data, int n, int *result) {
    // --------------------------------------------
    // 第1步：分配共享内存
    // --------------------------------------------
    // 共享内存：block内所有线程可见，速度快，容量小(48KB)
    extern __shared__ int shared_data[];

    // 线程索引
    int tid = threadIdx.x;  // block内线程索引 (0 ~ blockDim.x-1)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程索引
    int stride = blockDim.x * gridDim.x;

    // --------------------------------------------
    // 第2步：每个线程找自己负责数据的最大值
    // --------------------------------------------
    int local_max = INT_MIN;
    for (int i = gid; i < n; i += stride) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    // 写入共享内存
    shared_data[tid] = local_max;

    // 同步！确保所有线程都写完了shared_data
    // 如果不同步，有的线程可能读到未初始化的数据
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // 比较并更新
            if (shared_data[tid + stride] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + stride];
            }
        }
        // 同步！确保当前层的比较都完成了
        __syncthreads();
    }

    // --------------------------------------------
    // 第4步：block的代表（线程0）更新全局结果
    // --------------------------------------------
    if (tid == 0) {
        // 现在shared_data[0]存储了整个block的最大值
        atomicMax(result, shared_data[0]);

        // 注意：现在只有 gridDim.x 个线程做原子操作
        // 相比第3课的 gridDim.x * blockDim.x 个线程，减少了 blockDim.x 倍！
    }
}

__global__ void findMaxGPU_warp(int *data, int n, int *result){
  int tid =threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int nThreads = blockDim.x * gridDim.x;

  //计算warp的信息
  int lane = tid % 32;
  int warpId = tid/32;

  //step1: each thread finds its max
  int local_max = INT_MIN;
  for (int i = gid; i < n; i += nThreads) {
      if (data[i] > local_max) {
          local_max = data[i];
      }
  }

  //step2: warp shuflle reduce
  for (int offset =16; offset > 0; offset>>=1 ){
    int neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
    local_max = max(local_max, neighbor);
  }

  //step3: collect all warp results to shared memory
  __shared__ int warp_maxes[8];
  if(lane==0){
    warp_maxes[warpId] = local_max;
  }
  __syncthreads();

  //step4: the last warp does the reduce for all warp_maxes
  int block_max = INT_MIN;
  if (warpId==0){
    if (lane < 8){
      block_max = warp_maxes[lane];
    }
    for (int offset =16; offset > 0; offset>>=1 ){
      int neighbor = __shfl_down_sync(0xffffffff, block_max, offset);
      block_max = max(block_max, neighbor);
    }
  }

  //step5: thread 0 update the global max
  if (tid==0){
    atomicMax(result, block_max);
  }
}

#define CLUSTER_SIZE 8
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
findMaxGPU_cluster_dsm(int *data, int n, int *result){

  // cp handle
  cg::cluster_group cluster = cg::this_cluster();
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp =cg::tiled_partition<32>(block);

  // shared memory
  extern __shared__ int smem[];
  int *warp_results = smem;
  int *cluster_results = &smem[block.size()/32];

  int tid = block.thread_rank();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int nThreads = blockDim.x * gridDim.x;

  int warpId = warp.meta_group_rank(); //warp 在当前block的索引 0-7
  int numWarps = warp.meta_group_size(); //线程束数量

  //step1: each thread finds its max
  int local_max = INT_MIN;
  for (int i = gid; i < n; i += nThreads) {
      if (data[i] > local_max) {
          local_max = data[i];
      }
  }

  //step2: each warp reduce
  int warp_max = cg::reduce(warp, local_max, cg::greater<int>());

  //step3: warp lane 0/leader write into shared mem
  if (warp.thread_rank()==0){
    warp_results[warpId] = warp_max;
  }
  block.sync();


  //step4: 8 warps: 第一个warp规约所有的warp的结果
  int block_max = INT_MIN;
  if (warpId==0){
    if (warp.thread_rank() <  numWarps){
      block_max = warp_results[warp.thread_rank()];
    }

    block_max = cg::reduce(warp, block_max, cg::greater<int>());

    if(warp.thread_rank()==0){
      cluster_results[0] = block_max;
    }
  }

  //step5
  cluster.sync();

  //step6 dsm cluster 级别的reduce
  if(tid ==0){
    int cluster_rank = cluster.block_rank();
    int cluster_size = cluster.num_blocks();

    int cluster_max = cluster_results[0];

    for(int i =0; i<cluster_size; i++){
      if(i !=cluster_rank){
        int *remote_sm = cluster.map_shared_rank(cluster_results,i);
        cluster_max = max(cluster_max, remote_sm[0]); //*(remote_sm +0) = remote_sm[0]
      }
    }

    // cluster 0 update block global result
    if(cluster_rank==0){
      atomicMax(result, cluster_max);
    }
  }


}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("========================================\n");
    printf("第4课：共享内存 + 归约优化\n");
    printf("========================================\n\n");
    // -------------------------------------------------
    // 准备数据
    // -------------------------------------------------
    const int N = 10000000;  // 1000万
    printf("数据规模: %d 个整数\n", N);
    printf("数据大小: %.2f MB\n\n", N * sizeof(int) / 1024.0 / 1024.0);

    int *data, *result_naive, *result_shared, *result_warp, *result_cluster;
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_naive, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_shared, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_warp, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_cluster, sizeof(int)));

    // 初始化数据
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100000;
    }
    data[N/2] = 999999;  // 已知最大值

    // -------------------------------------------------
    // CPU基准测试
    // -------------------------------------------------
    printf("CPU计算中...\n");
    clock_t cpu_start = clock();
    int cpu_max = findMaxCPU(data, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    printf("CPU结果: %d (耗时: %.2f ms)\n\n", cpu_max, cpu_time);

    // -------------------------------------------------
    // GPU实现1: 第3课的简单实现（朴素版）
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU实现1: 朴素版（第3课方法）\n");
    printf("----------------------------------------\n");

    *result_naive = INT_MIN;

    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;  // 固定1024个block
    // int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;

    printf("配置: <<<%d blocks, %d threads>>>\n", blocksPerGrid, threadsPerBlock);
    printf("总线程数: %d\n", blocksPerGrid * threadsPerBlock);
    printf("原子操作次数: 约 %d 次\n", blocksPerGrid * threadsPerBlock);

    cudaEvent_t start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    CUDA_CHECK(cudaEventRecord(start1));
    matmalGPU_naive<<<blocksPerGrid, threadsPerBlock>>>(data, N, result_naive);
    CUDA_CHECK(cudaEventRecord(stop1));
    CUDA_CHECK(cudaEventSynchronize(stop1));

    float time_naive;
    CUDA_CHECK(cudaEventElapsedTime(&time_naive, start1, stop1));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("GPU结果: %d\n", *result_naive);
    printf("耗时: %.2f ms\n", time_naive);
    printf("加速比(vs CPU): %.2fx\n\n", cpu_time / time_naive);

    // -------------------------------------------------
    // GPU实现2: 共享内存优化版（今天的重点！）
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU实现2: 共享内存优化版（新！）\n");
    printf("----------------------------------------\n");

    *result_shared = INT_MIN;

    // 注意：需要指定共享内存大小（第三个参数）
    int shared_mem_size = threadsPerBlock * sizeof(int);

    printf("配置: <<<%d blocks, %d threads, %d bytes shared mem>>>\n",
           blocksPerGrid, threadsPerBlock, shared_mem_size);
    printf("总线程数: %d\n", blocksPerGrid * threadsPerBlock);
    printf("原子操作次数: 仅 %d 次（减少了%dx！）\n",
           blocksPerGrid, threadsPerBlock);

    cudaEvent_t start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    CUDA_CHECK(cudaEventRecord(start2));
    findMaxGPU_shared<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>
        (data, N, result_shared);
    CUDA_CHECK(cudaEventRecord(stop2));
    CUDA_CHECK(cudaEventSynchronize(stop2));

    float time_shared;
    CUDA_CHECK(cudaEventElapsedTime(&time_shared, start2, stop2));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("GPU结果: %d\n", *result_shared);
    printf("耗时: %.2f ms\n", time_shared);
    printf("加速比(vs CPU): %.2fx\n", cpu_time / time_shared);
    printf("加速比(vs 朴素版): %.2fx\n\n", time_naive / time_shared);

    // -------------------------------------------------
    // GPU实现3: warp shuflle
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU实现3: warp shuffle get max\n");
    printf("----------------------------------------\n");

    *result_warp = INT_MIN;

    cudaEvent_t start3, stop3;
    CUDA_CHECK(cudaEventCreate(&start3));
    CUDA_CHECK(cudaEventCreate(&stop3));

    CUDA_CHECK(cudaEventRecord(start3));
    findMaxGPU_warp<<<blocksPerGrid, threadsPerBlock>>>
        (data, N, result_warp);
    CUDA_CHECK(cudaEventRecord(stop3));
    CUDA_CHECK(cudaEventSynchronize(stop3));

    float time_warp;
    CUDA_CHECK(cudaEventElapsedTime(&time_warp, start3, stop3));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("GPU warp结果: %d\n", *result_warp);
    printf("耗时: %.2f ms\n", time_warp);
    printf("加速比(vs CPU): %.2fx\n", cpu_time / time_warp);
    printf("加速比(vs 朴素版): %.2fx\n\n", time_naive / time_warp);

    // -------------------------------------------------
    // gpu cg + DSM + cluster sync
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU实现4: cg + DSM + cluster sync\n");
    printf("----------------------------------------\n");


    *result_cluster = INT_MIN;

    // cluster configuration
    int clusterPerGrid = blocksPerGrid/CLUSTER_SIZE;
    int actualBlocks = clusterPerGrid/CLUSTER_SIZE;

    // shared mem
    int cluster_shared_mem_size = (threadsPerBlock/32 + 1) *sizeof(int);

    cudaEvent_t start4, stop4;
    CUDA_CHECK(cudaEventCreate(&start4));
    CUDA_CHECK(cudaEventCreate(&stop4));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(actualBlocks, 1,1);
    config.blockDim = dim3(threadsPerBlock, 1,1);
    config.dynamicSmemBytes = cluster_shared_mem_size;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = CLUSTER_SIZE;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.attrs = attribute;
    config.numAttrs =1;

    CUDA_CHECK(cudaEventRecord(start4));

    // cudaLaunchKernelEx api
    CUDA_CHECK(cudaLaunchKernelEx(&config, findMaxGPU_cluster_dsm, data, (int)N, result_cluster));

    CUDA_CHECK(cudaEventRecord(stop4));
    CUDA_CHECK(cudaEventSynchronize(stop4));

    float time_cluster;
    CUDA_CHECK(cudaEventElapsedTime(&time_cluster, start4, stop4));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("GPU cluster结果: %d\n", *result_cluster);
    printf("耗时: %.2f ms\n", time_cluster);
    printf("加速比(vs CPU): %.2fx\n", cpu_time / time_cluster);
    printf("加速比(vs 朴素版): %.2fx\n\n", time_naive / time_cluster);
    printf("加速比(vs warp): %.2fx\n\n", time_warp / time_cluster);
    // -------------------------------------------------
    // 结果验证
    // -------------------------------------------------
    printf("========================================\n");
    printf("结果验证:\n");
    printf("CPU: %d\n", cpu_max);
    printf("GPU朴素版: %d %s\n", *result_naive,
           (*result_naive == cpu_max) ? "✓" : "✗");
    printf("GPU优化版: %d %s\n", *result_shared,
           (*result_shared == cpu_max) ? "✓" : "✗");
    printf("GPU优化版warp: %d %s\n", *result_warp,
           (*result_warp == cpu_max) ? "✓" : "✗");
    printf("GPU优化版warp: %d %s\n", *result_cluster,
           (*result_cluster == cpu_max) ? "✓" : "✗");

    // -------------------------------------------------
    // 清理
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(result_naive));
    CUDA_CHECK(cudaFree(result_shared));
    CUDA_CHECK(cudaFree(result_warp));
    CUDA_CHECK(cudaFree(result_cluster));
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

