/*
 * 第3课：统一内存 + 数组求最大值（简单版）
 *
 * 教学目标：
 * 1. 学会使用统一内存（Unified Memory）简化内存管理
 * 2. 理解GPU归约算法的基本思想
 * 3. 对比CPU和GPU的性能差异
 *
 * 核心知识点：
 * - cudaMallocManaged(): 统一内存分配，CPU和GPU都能访问
 * - 归约模式：将N个数据规约为1个结果（这里求最大值）
 * - 原子操作：atomicMax() 保证线程安全
 *
 * 编译: nvcc -o lesson03 0301-max-unified-naive.cu
 * 运行: ./lesson03
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <limits.h>

// CUDA错误检查宏（养成好习惯：每次CUDA调用都要检查）
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
// CPU版本：用于验证GPU结果是否正确
// =============================================================================
int findMaxCPU(int *data, int n) {
    int max_val = INT_MIN;  // 从最小值开始
    for (int i = 0; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

// =============================================================================
// GPU版本：简单的归约算法（使用原子操作）
// =============================================================================
/*
 * Kernel设计思路：
 * - 每个线程负责检查一部分数据
 * - 使用grid-stride loop处理超过线程数的数据
 * - 用atomicMax()将本地最大值原子地更新到全局结果
 *
 * 优点：代码简单，统一内存易用
 * 缺点：原子操作有性能开销（下节课会优化）
 */
__global__ void findMaxGPU_naive(int *data, int n, int *result) {
    // 计算全局线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 总线程数

    // 每个线程的局部最大值
    int local_max = INT_MIN;

    // Grid-stride loop: 处理所有数据
    // 为什么用stride？因为数据量可能远大于线程数
    for (int i = tid; i < n; i += stride) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    // 原子操作：将局部最大值更新到全局结果
    // atomicMax保证线程安全，但会有性能开销
    atomicMax(result, local_max);
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("========================================\n");
    printf("第3课：统一内存 + 数组求最大值\n");
    printf("========================================\n\n");

    // -------------------------------------------------
    // 步骤1: 准备数据
    // -------------------------------------------------
    const int N = 10000000;  // 1000万个数据
    printf("数据规模: %d 个整数\n", N);
    printf("数据大小: %.2f MB\n\n", N * sizeof(int) / 1024.0 / 1024.0);

    // 使用统一内存分配：CPU和GPU都能访问！
    // 对比之前的cudaMalloc + cudaMemcpy，这里简单多了
    int *data, *gpu_result;
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&gpu_result, sizeof(int)));

    // 初始化数据（随机数）
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100000;  // 0-99999的随机数
    }

    // 人为设置一个已知的最大值，方便验证
    int known_max_pos = N / 2;
    data[known_max_pos] = 999999;
    printf("提示：在位置 %d 放置了已知最大值 999999\n\n", known_max_pos);

    // -------------------------------------------------
    // 步骤2: CPU求最大值（用于验证）
    // -------------------------------------------------
    printf("CPU计算中...\n");
    clock_t cpu_start = clock();
    int cpu_max = findMaxCPU(data, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    printf("CPU结果: %d (耗时: %.2f ms)\n\n", cpu_max, cpu_time);

    // -------------------------------------------------
    // 步骤3: GPU求最大值
    // -------------------------------------------------
    printf("GPU计算中...\n");

    // 初始化GPU结果
    *gpu_result = INT_MIN;

    // 配置kernel启动参数
    int threadsPerBlock = 256;  // 每个block 256个线程（常用配置）
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // 限制block数量，避免过多（1024个block足够了）
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    printf("Kernel配置: <<<%d blocks, %d threads>>>\n", blocksPerGrid, threadsPerBlock);
    printf("总线程数: %d\n", blocksPerGrid * threadsPerBlock);

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 开始计时
    CUDA_CHECK(cudaEventRecord(start));

    // 启动kernel
    findMaxGPU_naive<<<blocksPerGrid, threadsPerBlock>>>(data, N, gpu_result);

    // 结束计时
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 计算耗时
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    // 检查kernel是否有错误
    CUDA_CHECK(cudaGetLastError());

    // 等待GPU完成（统一内存需要同步才能在CPU访问）
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("GPU结果: %d (耗时: %.2f ms)\n\n", *gpu_result, gpu_time);

    // -------------------------------------------------
    // 步骤4: 验证结果
    // -------------------------------------------------
    printf("========================================\n");
    printf("结果验证:\n");
    if (cpu_max == *gpu_result) {
        printf("✓ 结果正确！CPU和GPU结果一致\n");
    } else {
        printf("✗ 结果错误！CPU=%d, GPU=%d\n", cpu_max, *gpu_result);
    }

    // -------------------------------------------------
    // 步骤5: 性能分析
    // -------------------------------------------------
    printf("\n性能分析:\n");
    float speedup = cpu_time / gpu_time;
    printf("加速比: %.2fx\n", speedup);

    if (speedup > 1.0) {
        printf("GPU比CPU快 %.2f 倍！\n", speedup);
    } else {
        printf("注意：数据量较小时，GPU可能不如CPU\n");
        printf("      GPU的优势在大规模并行计算\n");
    }

    // 计算吞吐量（每秒处理多少个数据）
    float throughput_gpu = (N / 1e6) / (gpu_time / 1000.0);  // M元素/秒
    printf("GPU吞吐量: %.2f M元素/秒\n", throughput_gpu);

    printf("========================================\n\n");

    // -------------------------------------------------
    // 步骤6: 统一内存的优势展示
    // -------------------------------------------------
    printf("统一内存的优势:\n");
    printf("1. 无需手动cudaMemcpy，自动迁移数据\n");
    printf("2. CPU和GPU可以直接访问同一块内存\n");
    printf("3. 代码更简洁，更容易维护\n");
    printf("4. 对学习CUDA算法非常友好\n\n");

    printf("局限性（下节课会解决）:\n");
    printf("1. 原子操作有性能开销\n");
    printf("2. 未充分利用共享内存\n");
    printf("3. 还有很大的优化空间\n\n");

    // -------------------------------------------------
    // 清理资源
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(gpu_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("第3课完成！下节课我们将使用共享内存优化性能。\n");

    return 0;
}

/*
 * 课后思考题：
 *
 * 1. 为什么要使用grid-stride loop？
 *    提示：思考数据量远大于线程数的情况
 *
 * 2. atomicMax()会影响性能吗？
 *    提示：试试把N改成100万、1000万、1亿，观察加速比变化
 *
 * 3. 统一内存vs手动内存管理的性能差异？
 *    提示：可以在0202-vec-add.cu中看手动管理的例子
 *
 * 4. 如何避免使用原子操作？
 *    提示：能否让每个block先算出local max，再合并？（下节课内容）
 */
