/*
 * 第4课：共享内存 + 归约优化
 *
 * 教学目标：
 * 1. 理解共享内存的作用和优势
 * 2. 掌握__syncthreads()同步机制
 * 3. 实现高效的block内归约算法
 * 4. 对比第3课，体验性能大幅提升
 *
 * 核心知识点：
 * - __shared__ 共享内存：block内线程共享的高速缓存
 * - __syncthreads(): block内线程同步屏障
 * - 归约树(Reduction Tree): 对数时间复杂度的并行归约
 * - Bank Conflict: 共享内存访问冲突（简介）
 *
 * 性能提升：相比第3课，预期 3-5倍加速
 *
 * 编译: nvcc -o lesson04 0401-max-shared-optimized.cu
 * 运行: ./lesson04
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
__global__ void findMaxGPU_naive(int *data, int n, int *result) {
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

// =============================================================================
// GPU版本2：使用共享内存优化（今天的重点！）
// =============================================================================
/*
 * 优化思路：
 * 1. 每个block分配一块共享内存
 * 2. 在block内做归约，得到block的最大值
 * 3. 只需要每个block的代表线程做一次atomicMax
 *
 * 性能提升原因：
 * - 原子操作次数：从(总线程数)次 减少到 (block数)次
 * - 共享内存速度：比全局内存快 20-30倍
 * - 归约树算法：O(log N)复杂度
 */
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

    // --------------------------------------------
    // 第3步：归约树 - 在block内找最大值
    // --------------------------------------------
    /*
     * 归约树示例（假设blockDim.x=8）:
     *
     * 初始: [3, 7, 2, 9, 1, 5, 4, 8]
     *        0  1  2  3  4  5  6  7  <- 线程索引
     *
     * stride=4: 线程0-3活跃
     *   shared[0] = max(shared[0], shared[4]) = max(3,1)=3
     *   shared[1] = max(shared[1], shared[5]) = max(7,5)=7
     *   shared[2] = max(shared[2], shared[6]) = max(2,4)=4
     *   shared[3] = max(shared[3], shared[7]) = max(9,8)=9
     *   结果: [3, 7, 4, 9, -, -, -, -]
     *
     * stride=2: 线程0-1活跃
     *   shared[0] = max(shared[0], shared[2]) = max(3,4)=4
     *   shared[1] = max(shared[1], shared[3]) = max(7,9)=9
     *   结果: [4, 9, -, -, -, -, -, -]
     *
     * stride=1: 线程0活跃
     *   shared[0] = max(shared[0], shared[1]) = max(4,9)=9
     *   结果: [9, -, -, -, -, -, -, -]
     *
     * 最终答案在 shared[0]！
     */

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

    int *data, *result_naive, *result_shared;
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_naive, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_shared, sizeof(int)));

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

    printf("配置: <<<%d blocks, %d threads>>>\n", blocksPerGrid, threadsPerBlock);
    printf("总线程数: %d\n", blocksPerGrid * threadsPerBlock);
    printf("原子操作次数: 约 %d 次\n", blocksPerGrid * threadsPerBlock);

    cudaEvent_t start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    CUDA_CHECK(cudaEventRecord(start1));
    findMaxGPU_naive<<<blocksPerGrid, threadsPerBlock>>>(data, N, result_naive);
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
    // 结果验证
    // -------------------------------------------------
    printf("========================================\n");
    printf("结果验证:\n");
    printf("CPU: %d\n", cpu_max);
    printf("GPU朴素版: %d %s\n", *result_naive,
           (*result_naive == cpu_max) ? "✓" : "✗");
    printf("GPU优化版: %d %s\n", *result_shared,
           (*result_shared == cpu_max) ? "✓" : "✗");

    // -------------------------------------------------
    // 性能总结
    // -------------------------------------------------
    printf("\n========================================\n");
    printf("性能总结:\n");
    printf("========================================\n");
    printf("%-20s %10.2f ms  (基准)\n", "CPU", cpu_time);
    printf("%-20s %10.2f ms  (%.2fx)\n", "GPU朴素版", time_naive,
           cpu_time/time_naive);
    printf("%-20s %10.2f ms  (%.2fx)\n", "GPU优化版", time_shared,
        time_naive/time_shared);
    printf("----------------------------------------\n");
    printf("共享内存优化提升: %.2fx\n", time_naive / time_shared);
    printf("========================================\n\n");

    // -------------------------------------------------
    // 知识点总结
    // -------------------------------------------------
    printf("本课学到的关键技术:\n");
    printf("1. __shared__ 共享内存声明\n");
    printf("   - 速度: 比全局内存快20-30倍\n");
    printf("   - 范围: block内所有线程共享\n");
    printf("   - 大小: 通常48KB，需要合理利用\n\n");

    printf("2. __syncthreads() 同步屏障\n");
    printf("   - 作用: 等待block内所有线程到达此处\n");
    printf("   - 必要性: 确保共享内存读写顺序正确\n");
    printf("   - 规则: 必须在条件语句外（所有线程都能执行到）\n\n");

    printf("3. 归约树算法\n");
    printf("   - 复杂度: O(log N) vs 串行O(N)\n");
    printf("   - 步数: log2(%d) = %d 步\n",
           threadsPerBlock, (int)(log2((double)threadsPerBlock)));
    printf("   - 优势: 充分利用并行性\n\n");

    printf("4. 减少原子操作\n");
    printf("   - 之前: 每个线程都做atomicMax\n");
    printf("   - 现在: 每个block只做一次atomicMax\n");
    printf("   - 减少: %dx 原子操作竞争\n\n", threadsPerBlock);

    // -------------------------------------------------
    // 清理
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(result_naive));
    CUDA_CHECK(cudaFree(result_shared));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));

    printf("第4课完成！\n");
    printf("下节课预告：Warp Shuffle - 更快的归约方法！\n");

    return 0;
}

/*
 * 课后思考题：
 *
 * 1. __syncthreads()如果放错位置会怎样？
 *    试试把第2个__syncthreads()注释掉，看看结果是否正确
 *
 * 2. 为什么归约树比串行快？
 *    计算: 256个线程，串行需要255次比较，归约树需要几步？
 *
 * 3. 共享内存大小限制？
 *    试试把threadsPerBlock改成1024，2048，观察是否能运行
 *
 * 4. Bank Conflict是什么？
 *    提示：shared_data[tid]和shared_data[tid+stride]的访问模式
 *    （第5课会深入讲解）
 *
 * 5. 能否完全避免原子操作？
 *    思考：如果blocksPerGrid也很大，能否继续归约？
 */
