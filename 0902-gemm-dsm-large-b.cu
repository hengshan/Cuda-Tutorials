#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>

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

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

// =============================================================================
// CPU版本：用于验证（只算小部分）
// =============================================================================
void matmulCPU_partial(const float *A, const float *B, float *C, 
                       int M, int N, int K, int check_size)
{
  // 只检查左上角 check_size × check_size
  int m_check = (M < check_size) ? M : check_size;
  int n_check = (N < check_size) ? N : check_size;
  
  for (int i = 0; i < m_check; i++){
    for (int j = 0; j < n_check; j++){
      float sum = 0.0f;
      for (int k = 0; k < K; k++){
        sum += A[i*K + k] * B[k*N + j];
      }
      C[i*n_check + j] = sum;
    }
  }
}

bool verifyPartial(const float* C_cpu, const float* C_gpu,
                   int M, int N, int check_size)
{
  const float eps = 1e-2f;  // 稍微放宽精度
  int m_check = (M < check_size) ? M : check_size;
  int n_check = (N < check_size) ? N : check_size;
  int err = 0;

  for (int i = 0; i < m_check && err < 10; i++){
    for (int j = 0; j < n_check && err < 10; j++){
      float cpu_val = C_cpu[i*n_check + j];
      float gpu_val = C_gpu[i*N + j];  // GPU 是完整矩阵布局
      float d = fabs(cpu_val - gpu_val);
      if (d > eps * fabs(cpu_val) + eps) {
        if (err < 5) {
          printf("  误差 [%d,%d]: CPU=%.4f GPU=%.4f diff=%.4f\n", 
                 i, j, cpu_val, gpu_val, d);
        }
        err++;
      }
    }
  }

  if (err > 0){
    printf("✗ 验证失败：发现 %d 处误差\n", err);
    return false;
  }
  printf("✓ 验证正确 (检查了 %d×%d 区域)\n", m_check, n_check);
  return true;
}

void initMatRandom(float *mat, size_t n)
{
  for (size_t i = 0; i < n; i++){
    mat[i] = (float)rand() / RAND_MAX;
  }
}

// =============================================================================
// GPU Baseline: tiled4
// =============================================================================
#define TILE_SIZE 16

__global__ void matmalGPU_tiled4_batched(const float *A_all, const float *B, float *C_all,
                                        int M, int N, int K)
{
  cg::thread_block block = cg::this_thread_block();

  int b = blockIdx.z;
  const float* A = A_all + (size_t)b * M * K;
  float*       C = C_all + (size_t)b * M * N;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int block_row = blockIdx.y * TILE_SIZE * 4;
  int block_col = blockIdx.x * TILE_SIZE * 4;

  __shared__ float As[TILE_SIZE*4][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE*4];

  float sum[4][4] = {{0.0f}};
  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < numTiles; t++){
    int tile_k = t * TILE_SIZE;

    #pragma unroll
    for (int i = 0; i < 4; i++){
      int aRow = block_row + ty*4 + i;
      int aCol = tile_k + tx;
      As[ty*4 + i][tx] = (aRow < M && aCol < K) ? A[aRow*K + aCol] : 0.0f;
    }

    #pragma unroll
    for (int j = 0; j < 4; j++){
      int bRow = tile_k + ty;
      int bCol = block_col + tx*4 + j;
      Bs[ty][tx*4 + j] = (bRow < K && bCol < N) ? B[bRow*N + bCol] : 0.0f;
    }

    block.sync();

    #pragma unroll
    for (int kk = 0; kk < TILE_SIZE; kk++){
      float a_reg[4], b_reg[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) a_reg[i] = As[ty*4 + i][kk];
      #pragma unroll
      for (int j = 0; j < 4; j++) b_reg[j] = Bs[kk][tx*4 + j];

      #pragma unroll
      for (int i = 0; i < 4; i++){
        #pragma unroll
        for (int j = 0; j < 4; j++){
          sum[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }
    block.sync();
  }

  #pragma unroll
  for (int i = 0; i < 4; i++){
    #pragma unroll
    for (int j = 0; j < 4; j++){
      int c_row = block_row + ty*4 + i;
      int c_col = block_col + tx*4 + j;
      if (c_row < M && c_col < N){
        C[c_row*N + c_col] = sum[i][j];
      }
    }
  }
}

// =============================================================================
// DSM 版本：共享 B tile
// =============================================================================
__global__ void matmalGPU_dsm(const float *A_all, const float *B, float *C_all,
                              int M, int N, int K, int expected_cluster_z)
{
  cg::thread_block block = cg::this_thread_block();
  cg::cluster_group cluster = cg::this_cluster();

  if (cluster.dim_blocks().z != expected_cluster_z) return;

  int b = blockIdx.z;
  const float* A = A_all + (size_t)b * M * K;
  float*       C = C_all + (size_t)b * M * N;

  int tx = threadIdx.x, ty = threadIdx.y;
  int block_row = blockIdx.y * TILE_SIZE * 4;
  int block_col = blockIdx.x * TILE_SIZE * 4;

  __shared__ float As[TILE_SIZE*4][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE*4];

  float sum[4][4] = {{0.0f}};
  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < numTiles; t++){
    int tile_k = t * TILE_SIZE;

    // 每个 block 加载自己的 A tile
    #pragma unroll
    for (int i = 0; i < 4; i++){
      int aRow = block_row + ty*4 + i;
      int aCol = tile_k + tx;
      As[ty*4 + i][tx] = (aRow < M && aCol < K) ? A[aRow*K + aCol] : 0.0f;
    }

    // 只有 rank0 加载 B tile
    if (cluster.block_rank() == 0) {
      #pragma unroll
      for (int j = 0; j < 4; j++){
        int bRow = tile_k + ty;
        int bCol = block_col + tx*4 + j;
        Bs[ty][tx*4 + j] = (bRow < K && bCol < N) ? B[bRow*N + bCol] : 0.0f;
      }
    }

    block.sync();
    cluster.sync();

    // 获取 rank0 的 Bs
    float* Bs_rank0 = (float*)cluster.map_shared_rank(&Bs[0][0], 0);

    #pragma unroll
    for (int kk = 0; kk < TILE_SIZE; kk++){
      float a_reg[4], b_reg[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) a_reg[i] = As[ty*4 + i][kk];

      int base = kk * (TILE_SIZE * 4) + tx * 4;
      #pragma unroll
      for (int j = 0; j < 4; j++) b_reg[j] = Bs_rank0[base + j];

      #pragma unroll
      for (int i = 0; i < 4; i++){
        #pragma unroll
        for (int j = 0; j < 4; j++){
          sum[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }

    block.sync();
    cluster.sync();
  }

  #pragma unroll
  for (int i = 0; i < 4; i++){
    #pragma unroll
    for (int j = 0; j < 4; j++){
      int c_row = block_row + ty*4 + i;
      int c_col = block_col + tx*4 + j;
      if (c_row < M && c_col < N){
        C[c_row*N + c_col] = sum[i][j];
      }
    }
  }
}

// =============================================================================
// 辅助函数
// =============================================================================
bool isClusterLaunchSupported()
{
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int supported = 0;
  cudaError_t err = cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, dev);
  if (err == cudaSuccess) return supported != 0;
  if (err == cudaErrorInvalidValue) { cudaGetLastError(); return false; }
  CUDA_CHECK(err);
  return false;
}

void printGPUInfo() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  printf("GPU: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("SM 数量: %d\n", prop.multiProcessorCount);
  printf("L2 缓存: %d MB\n", prop.l2CacheSize / (1024*1024));
  printf("总显存: %.1f GB\n", prop.totalGlobalMem / (1024.0*1024*1024));
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("========================================\n");
    printf("DSM vs Baseline: 大矩阵测试\n");
    printf("当 B 矩阵超过 L2 缓存时，DSM 可能有优势\n");
    printf("========================================\n\n");

    printGPUInfo();

    // 获取 L2 大小用于配置建议
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int l2_mb = prop.l2CacheSize / (1024*1024);

    printf("\n--- 测试配置 ---\n");

    // 配置：让 B 矩阵超过 L2 缓存
    // B = K × N × 4 bytes
    // 如果 L2 = 64MB，需要 K × N > 16M，即 K=N=4096 时 B=64MB
    // 我们用 K=N=5120，使 B = 100MB > 64MB
    
    const int M = 1024;      // 保持较小
    const int K = 5120;      // 增大
    const int N = 5120;      // 增大
    const int BATCH = 32;    // 减少 batch 以节省显存
    const int CLUSTER_Z = 4; // cluster size

    const size_t size_A_all = (size_t)BATCH * M * K * sizeof(float);
    const size_t size_B     = (size_t)K * N * sizeof(float);
    const size_t size_C_all = (size_t)BATCH * M * N * sizeof(float);
    const size_t total_mem  = size_A_all + size_B + size_C_all;

    printf("M=%d, K=%d, N=%d, BATCH=%d\n", M, K, N, BATCH);
    printf("B 矩阵大小: %.1f MB\n", size_B / (1024.0f * 1024.0f));
    printf("L2 缓存: %d MB\n", l2_mb);
    printf("B 矩阵 %s L2 缓存\n", (size_B > prop.l2CacheSize) ? "超过 ✓" : "未超过 ✗");
    printf("总显存使用: %.1f MB\n", total_mem / (1024.0f * 1024.0f));
    printf("CLUSTER_Z = %d (每 %d 个 batch 共享 B tile)\n", CLUSTER_Z, CLUSTER_Z);

    if (BATCH % CLUSTER_Z != 0) {
      printf("错误：BATCH 必须能被 CLUSTER_Z 整除\n");
      return 1;
    }

    // 分配 Host 内存
    float *h_A_all = (float*)malloc(size_A_all);
    float *h_B     = (float*)malloc(size_B);
    if (!h_A_all || !h_B) {
      printf("Host 内存分配失败\n");
      return 1;
    }

    // CPU 验证只检查一小部分
    const int CHECK_SIZE = 64;
    float *h_C_cpu = (float*)malloc(CHECK_SIZE * CHECK_SIZE * sizeof(float));
    float *h_C_gpu = (float*)malloc((size_t)M * N * sizeof(float));

    printf("\n初始化数据...\n");
    srand(42);
    initMatRandom(h_A_all, (size_t)BATCH * M * K);
    initMatRandom(h_B, (size_t)K * N);

    // CPU 参考（只算 batch 0 的一小部分）
    printf("CPU 参考计算 (只检查 %d×%d 区域)...\n", CHECK_SIZE, CHECK_SIZE);
    matmulCPU_partial(h_A_all, h_B, h_C_cpu, M, N, K, CHECK_SIZE);

    // Device 内存
    float *d_A_all, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A_all, size_A_all));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C_all));
    CUDA_CHECK(cudaMemcpy(d_A_all, h_A_all, size_A_all, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block16(16, 16);
    dim3 grid(div_up(N, 64), div_up(M, 64), BATCH);

    printf("\nGrid: (%d, %d, %d), Block: (16, 16)\n", grid.x, grid.y, grid.z);

    // 测试函数
    auto run_baseline = [&]() -> float {
      CUDA_CHECK(cudaMemset(d_C, 0, size_C_all));

      cudaEvent_t s, e;
      CUDA_CHECK(cudaEventCreate(&s));
      CUDA_CHECK(cudaEventCreate(&e));

      // Warmup
      matmalGPU_tiled4_batched<<<grid, block16>>>(d_A_all, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaDeviceSynchronize());

      float total_ms = 0;
      const int runs = 5;

      for (int r = 0; r < runs; r++) {
        CUDA_CHECK(cudaEventRecord(s));
        matmalGPU_tiled4_batched<<<grid, block16>>>(d_A_all, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
      }

      CUDA_CHECK(cudaEventDestroy(s));
      CUDA_CHECK(cudaEventDestroy(e));
      return total_ms / runs;
    };

    auto run_dsm = [&]() -> float {
      CUDA_CHECK(cudaMemset(d_C, 0, size_C_all));

      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = grid;
      cfg.blockDim = block16;

      cudaLaunchAttribute attr;
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = 1;
      attr.val.clusterDim.y = 1;
      attr.val.clusterDim.z = CLUSTER_Z;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;

      CUDA_CHECK(cudaFuncSetAttribute(matmalGPU_dsm, 
                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

      cudaEvent_t s, e;
      CUDA_CHECK(cudaEventCreate(&s));
      CUDA_CHECK(cudaEventCreate(&e));

      // Warmup
      cudaLaunchKernelEx(&cfg, matmalGPU_dsm, d_A_all, d_B, d_C, M, N, K, CLUSTER_Z);
      CUDA_CHECK(cudaDeviceSynchronize());

      float total_ms = 0;
      const int runs = 5;

      for (int r = 0; r < runs; r++) {
        CUDA_CHECK(cudaEventRecord(s));
        cudaLaunchKernelEx(&cfg, matmalGPU_dsm, d_A_all, d_B, d_C, M, N, K, CLUSTER_Z);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
      }

      CUDA_CHECK(cudaEventDestroy(s));
      CUDA_CHECK(cudaEventDestroy(e));
      return total_ms / runs;
    };

    // 运行测试
    printf("\n========================================\n");
    printf("性能对比\n");
    printf("========================================\n");

    printf("\n--- Baseline (tiled4, 每个 block 都读 B) ---\n");
    float t_baseline = run_baseline();
    printf("耗时: %.3f ms\n", t_baseline);

    // 验证 baseline
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    verifyPartial(h_C_cpu, h_C_gpu, M, N, CHECK_SIZE);

    // 计算理论值
    double flops = 2.0 * BATCH * M * N * K;
    double tflops_baseline = flops / (t_baseline * 1e9);
    printf("吞吐量: %.2f TFLOPS\n", tflops_baseline);

    if (isClusterLaunchSupported()) {
      printf("\n--- DSM (cluster 共享 B tile) ---\n");
      float t_dsm = run_dsm();
      printf("耗时: %.3f ms\n", t_dsm);

      // 验证 DSM
      CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
      verifyPartial(h_C_cpu, h_C_gpu, M, N, CHECK_SIZE);

      double tflops_dsm = flops / (t_dsm * 1e9);
      printf("吞吐量: %.2f TFLOPS\n", tflops_dsm);

      printf("\n--- 对比结果 ---\n");
      float speedup = t_baseline / t_dsm;
      printf("DSM vs Baseline: %.2fx %s\n", speedup, 
             speedup > 1.0 ? "更快 ✓" : "更慢 ✗");

      if (speedup < 1.0) {
        printf("\n分析：即使 B 超过 L2，cluster.sync() 开销仍然太大\n");
        printf("每个 tile 需要 2 次 cluster.sync()\n");
        printf("总 tiles = K/16 = %d\n", K/16);
        printf("总 sync 次数 = %d\n", 2 * K/16);
      }
    } else {
      printf("\n设备不支持 Cluster Launch\n");
    }

    // 清理
    CUDA_CHECK(cudaFree(d_A_all));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A_all);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    printf("\n========================================\n");
    printf("结论\n");
    printf("========================================\n");
    printf("即使 B 矩阵超过 L2 缓存，cluster.sync() 的开销\n");
    printf("可能仍然大于节省的 global memory 带宽。\n");
    printf("\n");
    printf("DSM 更适合的场景：\n");
    printf("1. AllReduce / Ring Reduce（通信密集型）\n");
    printf("2. 邻居数据交换（Halo Exchange）\n");
    printf("3. Persistent Kernels（长时间运行，少量同步）\n");

    return 0;
}
