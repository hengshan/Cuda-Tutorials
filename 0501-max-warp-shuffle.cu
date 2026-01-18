/*
 * 第5课：Warp级别优化 - Shuffle指令
 *
 * 教学目标：
 * 1. 理解Warp的概念（GPU执行的基本单元）
 * 2. 掌握Shuffle指令进行warp内通信
 * 3. 实现无需共享内存和同步的高效归约
 * 4. 理解warp级别编程的优势
 *
 * 核心知识点：
 * - Warp: 32个线程的执行单元，硬件级别的SIMT
 * - __shfl_down_sync(): warp内线程间数据交换
 * - Warp内隐式同步：无需__syncthreads()
 * - 寄存器通信：比共享内存更快
 *
 * 性能提升：相比第4课，再提升 1.5-2倍
 *
 * 编译: nvcc -arch=sm_120 -o lesson05 0501-max-warp-shuffle.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <limits.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define WARP_SIZE 32  // GPU warp大小（NVIDIA GPU固定为32）

// =============================================================================
// CPU版本
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
// GPU版本1：第4课的共享内存版本（用于对比）
// =============================================================================
__global__ void findMaxGPU_shared(int *data, int n, int *result) {
    extern __shared__ int shared_data[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_max = INT_MIN;
    for (int i = gid; i < n; i += stride) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    shared_data[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid + s] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(result, shared_data[0]);
    }
}

// =============================================================================
// Warp级别归约：核心技术！
// =============================================================================
/*
 * Warp Shuffle原理：
 *
 * __shfl_down_sync(mask, var, delta):
 * - 从本warp内 (lane_id + delta) 的线程读取var的值
 * - 直接在寄存器间传输，无需内存
 * - 速度极快（1个时钟周期）
 *
 * 示例（warp内8个线程，简化说明）:
 * 初始: lane0=3, lane1=7, lane2=2, lane3=9, lane4=1, lane5=5, lane6=4, lane7=8
 *
 * delta=4:
 *   lane0 从 lane4 读取: 3 vs 1 -> 3
 *   lane1 从 lane5 读取: 7 vs 5 -> 7
 *   lane2 从 lane6 读取: 2 vs 4 -> 4
 *   lane3 从 lane7 读取: 9 vs 8 -> 9
 *   lane4-7 读到无效值（超出边界）
 *
 * delta=2:
 *   lane0 从 lane2 读取: 3 vs 4 -> 4
 *   lane1 从 lane3 读取: 7 vs 9 -> 9
 *
 * delta=1:
 *   lane0 从 lane1 读取: 4 vs 9 -> 9
 *
 * 结果: lane0 = 9（整个warp的最大值）
 */
__device__ int warpReduceMax(int val) {
    // 0xffffffff: 全部32个线程都参与
    // 逐步从右边的线程获取值，进行比较
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int neighbor = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, neighbor);
    }
    // 返回时，lane 0 持有整个warp的最大值
    return val;
}

// =============================================================================
// GPU版本2：Warp Shuffle优化版（今天的重点！）
// =============================================================================
/*
 * 优化策略：
 * 1. 先在warp内用shuffle归约（无需共享内存）
 * 2. 每个warp的lane 0将结果写入共享内存
 * 3. 最后一个warp对所有warp的结果再做归约
 *
 * 优势：
 * - Warp内通信：寄存器级别，极快
 * - 减少共享内存使用：只需 (blockDim.x/32) 个元素
 * - 减少同步次数：warp内隐式同步，无需__syncthreads()
 */
__global__ void findMaxGPU_warp_shuffle(int *data, int n, int *result) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 计算warp相关信息
    int lane = tid % WARP_SIZE;           // warp内的线程索引 (0-31)
    int warpId = tid / WARP_SIZE;         // 本线程属于block内的第几个warp

    // --------------------------------------------
    // 第1步：每个线程找到自己负责数据的最大值
    // --------------------------------------------
    int local_max = INT_MIN;
    for (int i = gid; i < n; i += stride) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    // --------------------------------------------
    // 第2步：Warp内归约（使用shuffle）
    // --------------------------------------------
    // 这里不需要同步！Warp内32个线程自动同步执行
    local_max = warpReduceMax(local_max);

    // 现在每个warp的lane 0持有该warp的最大值

    // --------------------------------------------
    // 第3步：收集所有warp的结果到共享内存
    // --------------------------------------------
    // 共享内存大小只需要 (blockDim.x / 32) 个元素
    __shared__ int warp_maxes[32];  // 假设最多32个warp per block (1024线程)

    if (lane == 0) {
        warp_maxes[warpId] = local_max;
    }
    __syncthreads();  // 确保所有warp都写完了

    // --------------------------------------------
    // 第4步：最后一个warp对所有warp的结果归约
    // --------------------------------------------
    // 让第一个warp负责最终归约
    int block_max = INT_MIN;
    if (warpId == 0) {
        // 每个线程读取一个warp的结果
        // 注意边界：warp数量 = (blockDim.x + 31) / 32
        int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        if (lane < numWarps) {
            block_max = warp_maxes[lane];
        }
        // 再做一次warp归约
        block_max = warpReduceMax(block_max);
    }

    // --------------------------------------------
    // 第5步：block的代表（线程0）更新全局结果
    // --------------------------------------------
    if (tid == 0) {
        atomicMax(result, block_max);
    }
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("========================================\n");
    printf("第5课：Warp级别优化 - Shuffle指令\n");
    printf("========================================\n\n");

    // -------------------------------------------------
    // 准备数据
    // -------------------------------------------------
    const int N = 10000000;  // 1000万
    printf("数据规模: %d 个整数\n", N);
    printf("数据大小: %.2f MB\n\n", N * sizeof(int) / 1024.0 / 1024.0);

    int *data, *result_shared, *result_warp;
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_shared, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_warp, sizeof(int)));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100000;
    }
    data[N/2] = 999999;

    // -------------------------------------------------
    // CPU基准
    // -------------------------------------------------
    printf("CPU计算中...\n");
    clock_t cpu_start = clock();
    int cpu_max = findMaxCPU(data, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    printf("CPU结果: %d (耗时: %.2f ms)\n\n", cpu_max, cpu_time);

    // -------------------------------------------------
    // GPU实现1: 共享内存版（第4课）
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU实现1: 共享内存版（第4课）\n");
    printf("----------------------------------------\n");

    *result_shared = INT_MIN;
    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;
    int shared_mem_size = threadsPerBlock * sizeof(int);

    cudaEvent_t start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    CUDA_CHECK(cudaEventRecord(start1));
    findMaxGPU_shared<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>
        (data, N, result_shared);
    CUDA_CHECK(cudaEventRecord(stop1));
    CUDA_CHECK(cudaEventSynchronize(stop1));

    float time_shared;
    CUDA_CHECK(cudaEventElapsedTime(&time_shared, start1, stop1));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("结果: %d\n", *result_shared);
    printf("耗时: %.2f ms\n", time_shared);
    printf("共享内存使用: %d bytes\n", shared_mem_size);
    printf("加速比(vs CPU): %.2fx\n\n", cpu_time / time_shared);

    // -------------------------------------------------
    // GPU实现2: Warp Shuffle优化版（今天的重点！）
    // -------------------------------------------------
    printf("----------------------------------------\n");
    printf("GPU实现2: Warp Shuffle优化版（新！）\n");
    printf("----------------------------------------\n");

    *result_warp = INT_MIN;

    printf("Warp相关信息:\n");
    printf("- Warp大小: %d 个线程\n", WARP_SIZE);
    printf("- 每个block的warp数: %d\n", threadsPerBlock / WARP_SIZE);
    printf("- 共享内存使用: %zu bytes (减少了%dx!)\n",
           32 * sizeof(int), threadsPerBlock / 32);
    printf("\n");

    cudaEvent_t start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    CUDA_CHECK(cudaEventRecord(start2));
    findMaxGPU_warp_shuffle<<<blocksPerGrid, threadsPerBlock>>>
        (data, N, result_warp);
    CUDA_CHECK(cudaEventRecord(stop2));
    CUDA_CHECK(cudaEventSynchronize(stop2));

    float time_warp;
    CUDA_CHECK(cudaEventElapsedTime(&time_warp, start2, stop2));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("结果: %d\n", *result_warp);
    printf("耗时: %.2f ms\n", time_warp);
    printf("加速比(vs CPU): %.2fx\n", cpu_time / time_warp);
    printf("加速比(vs 共享内存版): %.2fx\n\n", time_shared / time_warp);

    // -------------------------------------------------
    // 验证
    // -------------------------------------------------
    printf("========================================\n");
    printf("结果验证:\n");
    printf("CPU: %d\n", cpu_max);
    printf("GPU共享内存版: %d %s\n", *result_shared,
           (*result_shared == cpu_max) ? "✓" : "✗");
    printf("GPU Warp Shuffle版: %d %s\n", *result_warp,
           (*result_warp == cpu_max) ? "✓" : "✗");

    // -------------------------------------------------
    // 知识点总结
    // -------------------------------------------------
    printf("本课核心技术:\n\n");

    printf("1. Warp的概念\n");
    printf("   - 定义: 32个线程的执行单元\n");
    printf("   - 特点: 硬件级别的SIMT（单指令多线程）\n");
    printf("   - 同步: Warp内线程自动同步，无需__syncthreads()\n\n");

    printf("2. Shuffle指令\n");
    printf("   - __shfl_down_sync(): 从右边线程读取数据\n");
    printf("   - __shfl_up_sync(): 从左边线程读取数据\n");
    printf("   - __shfl_xor_sync(): 按位异或方式交换\n");
    printf("   - 速度: 1个时钟周期，比共享内存快10倍+\n\n");

    printf("3. 优化效果对比\n");
    printf("   - 第3课(朴素): 原子操作竞争严重\n");
    printf("   - 第4课(共享内存): 减少原子操作，使用共享内存\n");
    printf("   - 第5课(Warp Shuffle): 寄存器通信，最小化共享内存\n\n");

    printf("4. 适用场景\n");
    printf("   - 归约操作: sum, max, min, average等\n");
    printf("   - Warp内通信: 需要邻近线程协作的算法\n");
    printf("   - 高性能计算: 追求极致性能的场景\n\n");

    // -------------------------------------------------
    // 清理
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(result_shared));
    CUDA_CHECK(cudaFree(result_warp));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));

    printf("第5课完成！\n");
    printf("下节课预告：矩阵乘法 - 从朴素到优化的完整实战！\n");

    return 0;
}

/*
 * 课后思考题：
 *
 * 1. 为什么warp内不需要__syncthreads()？
 *    提示：理解SIMT执行模型
 *
 * 2. __shfl_down_sync的mask参数(0xffffffff)是什么意思？
 *    提示：32位掩码，每位代表一个lane是否参与
 *
 * 3. 如果blockDim.x不是32的倍数会怎样？
 *    提示：最后一个warp可能不满，但代码仍然正确
 *
 * 4. Shuffle vs 共享内存，何时用哪个？
 *    提示：warp内用shuffle，跨warp用共享内存
 *
 * 5. 能否完全不用共享内存？
 *    提示：warp之间仍需通信，除非用Cooperative Groups（第8课）
 */
