/*
 * 第6课：矩阵乘法 - 朴素实现
 *
 * 教学目标：
 * 1. 实现GPU矩阵乘法的基本算法
 * 2. 理解2D线程网格与矩阵元素的映射
 * 3. 学会计算GFLOPS性能指标
 * 4. 分析全局内存访问的性能瓶颈
 *
 * 核心知识点：
 * - 矩阵乘法: C = A × B，C[i][j] = Σ A[i][k] * B[k][j]
 * - 2D线程网格: blockIdx.x/y, threadIdx.x/y
 * - 行主序存储: matrix[i][j] = array[i * width + j]
 * - GFLOPS计算: (2*M*N*K) / (时间秒数) / 10^9
 *
 * 性能预期：
 * - 朴素实现：~50-100 GFLOPS (取决于硬件)
 * - 理论峰值(RTX 5090 FP32): ~90 TFLOPS
 * - 效率: <1% （全局内存瓶颈）
 *
 * 编译: nvcc -arch=sm_120 -o lesson06 0601-matmul-naive.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

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
// CPU矩阵乘法：用于验证GPU结果
// =============================================================================
/*
 * 矩阵乘法定义：C = A × B
 * A: M × K
 * B: K × N
 * C: M × N
 *
 * C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
 */
void matmulCPU(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// =============================================================================
// GPU矩阵乘法：朴素实现
// =============================================================================
/*
 * 算法思路：
 * 1. 每个线程负责计算C矩阵的一个元素
 * 2. 线程(i,j)计算C[i][j]
 * 3. 需要读取A的第i行和B的第j列
 *
 * 性能问题：
 * - 每个元素需要K次全局内存读取
 * - A和B的数据被重复读取多次
 * - 没有利用数据重用（下节课优化）
 */
__global__ void matmulGPU_naive(float *A, float *B, float *C,
                                int M, int N, int K) {
    // 计算当前线程负责的矩阵元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C矩阵的行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C矩阵的列索引

    // 边界检查：确保不越界
    if (row < M && col < N) {
        float sum = 0.0f;

        // 计算点积：A的第row行 × B的第col列
        for (int k = 0; k < K; k++) {
            // A[row][k]: 行主序存储，索引为 row * K + k
            // B[k][col]: 行主序存储，索引为 k * N + col
            sum += A[row * K + k] * B[k * N + col];
        }

        // 写回结果
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 辅助函数：初始化矩阵
// =============================================================================
void initMatrix(float *mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = value;
    }
}

void initMatrixRandom(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;  // 0-1之间的随机数
    }
}

// =============================================================================
// 辅助函数：验证结果
// =============================================================================
bool verifyResult(float *C_cpu, float *C_gpu, int M, int N) {
    const float epsilon = 1e-3;  // 允许的误差范围
    int errorCount = 0;

    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C_cpu[i] - C_gpu[i]);
        if (diff > epsilon) {
            errorCount++;
            if (errorCount <= 10) {  // 只打印前10个错误
                printf("错误[%d]: CPU=%.6f, GPU=%.6f, diff=%.6f\n",
                       i, C_cpu[i], C_gpu[i], diff);
            }
        }
    }

    if (errorCount > 0) {
        printf("发现 %d 个错误（总共 %d 个元素）\n", errorCount, M * N);
        return false;
    }
    return true;
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("========================================\n");
    printf("第6课：矩阵乘法 - 朴素实现\n");
    printf("========================================\n\n");

    // -------------------------------------------------
    // 矩阵维度设置
    // -------------------------------------------------
    // 为了在30分钟内讲完，使用中等规模的矩阵
    int M = 1024;  // A的行数，C的行数
    int K = 1024;  // A的列数，B的行数
    int N = 1024;  // B的列数，C的列数

    printf("矩阵维度:\n");
    printf("  A: %d × %d\n", M, K);
    printf("  B: %d × %d\n", K, N);
    printf("  C: %d × %d\n", M, N);
    printf("\n");

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    printf("内存占用:\n");
    printf("  A: %.2f MB\n", size_A / 1024.0 / 1024.0);
    printf("  B: %.2f MB\n", size_B / 1024.0 / 1024.0);
    printf("  C: %.2f MB\n", size_C / 1024.0 / 1024.0);
    printf("  总计: %.2f MB\n\n",
           (size_A + size_B + size_C) / 1024.0 / 1024.0);

    // -------------------------------------------------
    // 分配内存（统一内存）
    // -------------------------------------------------
    float *A, *B, *C_cpu, *C_gpu;
    CUDA_CHECK(cudaMallocManaged(&A, size_A));
    CUDA_CHECK(cudaMallocManaged(&B, size_B));
    CUDA_CHECK(cudaMallocManaged(&C_cpu, size_C));
    CUDA_CHECK(cudaMallocManaged(&C_gpu, size_C));

    // -------------------------------------------------
    // 初始化矩阵
    // -------------------------------------------------
    printf("初始化矩阵...\n");
    srand(time(NULL));

    // 使用简单的初始化方便验证
    // 实际应用中可以用随机数: initMatrixRandom(A, M, K);
    initMatrixRandom(A, M, K);
    initMatrixRandom(B, K, N);
    // initMatrix(A, M, K, 1.0f);
    // initMatrix(B, K, N, 2.0f);

    printf("  A: 全1矩阵\n");
    printf("  B: 全2矩阵\n");
    printf("  预期C: 全%d矩阵 (1*2*K = %d)\n\n", 2*K, 2*K);

    // -------------------------------------------------
    // CPU计算（小规模验证用）
    // -------------------------------------------------
    printf("CPU计算中...\n");
    clock_t cpu_start = clock();
    matmulCPU(A, B, C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;

    // 计算GFLOPS
    // 矩阵乘法的浮点操作数：每个输出元素需要 K次乘法 + K次加法 = 2K次操作
    // 总操作数：M * N * 2K
    double gflops_cpu = (2.0 * M * N * K) / (cpu_time / 1000.0) / 1e9;

    printf("CPU完成！\n");
    printf("  耗时: %.2f ms\n", cpu_time);
    printf("  性能: %.2f GFLOPS\n", gflops_cpu);
    printf("  示例结果: C[0][0]=%.2f, C[10][10]=%.2f\n\n",
           C_cpu[0], C_cpu[10 * N + 10]);

    // -------------------------------------------------
    // GPU计算
    // -------------------------------------------------
    printf("========================================\n");
    printf("GPU计算（朴素实现）\n");
    printf("========================================\n");

    // 配置2D线程块和网格
    dim3 blockDim(16, 16);  // 16×16=256个线程per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("Kernel配置:\n");
    printf("  Block: (%d, %d) = %d threads\n",
           blockDim.x, blockDim.y, blockDim.x * blockDim.y);
    printf("  Grid:  (%d, %d) = %d blocks\n",
           gridDim.x, gridDim.y, gridDim.x * gridDim.y);
    printf("  总线程数: %d\n\n",
           blockDim.x * blockDim.y * gridDim.x * gridDim.y);

    // 创建CUDA事件计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热GPU（第一次调用可能有初始化开销）
    matmulGPU_naive<<<gridDim, blockDim>>>(A, B, C_gpu, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 正式计时
    CUDA_CHECK(cudaEventRecord(start));
    matmulGPU_naive<<<gridDim, blockDim>>>(A, B, C_gpu, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    double gflops_gpu = (2.0 * M * N * K) / (gpu_time / 1000.0) / 1e9;

    printf("GPU完成！\n");
    printf("  耗时: %.2f ms\n", gpu_time);
    printf("  性能: %.2f GFLOPS\n", gflops_gpu);
    printf("  加速比: %.2fx\n", cpu_time / gpu_time);
    printf("  示例结果: C[0][0]=%.2f, C[10][10]=%.2f\n\n",
           C_gpu[0], C_gpu[10 * N + 10]);

    // -------------------------------------------------
    // 验证结果
    // -------------------------------------------------
    printf("========================================\n");
    printf("验证结果...\n");
    if (verifyResult(C_cpu, C_gpu, M, N)) {
        printf("✓ 结果正确！GPU计算与CPU一致\n");
    } else {
        printf("✗ 结果错误！请检查代码\n");
    }
    printf("========================================\n\n");

    // -------------------------------------------------
    // 性能分析
    // -------------------------------------------------
    printf("性能分析:\n");
    printf("----------------------------------------\n");
    printf("1. 计算密度\n");
    printf("   浮点操作数: %.2f GFLOP (2*%d*%d*%d)\n",
           (2.0*M*N*K)/1e9, M, N, K);
    printf("   数据量: %.2f MB\n",
           (size_A + size_B + size_C) / 1024.0 / 1024.0);
    printf("   计算密度: %.2f FLOP/Byte\n",
           (2.0*M*N*K) / (size_A + size_B + size_C));
    printf("\n");

    printf("2. 内存带宽利用\n");
    printf("   每个线程读取: %d 个float (从A和B)\n", 2*K);
    printf("   总读取量: %.2f GB\n",
           (M*N*2*K*sizeof(float)) / 1024.0 / 1024.0 / 1024.0);
    printf("   有效带宽: %.2f GB/s\n",
           (M*N*2*K*sizeof(float)) / (gpu_time/1000.0) / 1e9);
    printf("   理论峰值(RTX 5090): 1792 GB/s\n");
    printf("   带宽利用率: %.2f%%\n",
           ((M*N*2*K*sizeof(float))/(gpu_time/1000.0)/1e9) / 1792 * 100);
    printf("\n");

    printf("3. 性能瓶颈\n");
    printf("   ✗ 全局内存访问：A和B被重复读取\n");
    printf("   ✗ 无数据重用：未使用共享内存缓存\n");
    printf("   ✗ B的列访问：跨步访问，缓存不友好\n");
    printf("\n");

    printf("4. 优化方向（下节课）\n");
    printf("   → 使用共享内存缓存tile\n");
    printf("   → 提高数据重用率\n");
    printf("   → 预期加速：10-20倍！\n");
    printf("========================================\n\n");

    // -------------------------------------------------
    // 清理
    // -------------------------------------------------
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C_cpu));
    CUDA_CHECK(cudaFree(C_gpu));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("第6课完成！\n");
    printf("下节课：矩阵乘法 Tiling优化 - 达到接近硬件峰值的性能！\n");

    return 0;
}

/*
 * 课后思考题：
 *
 * 1. 为什么使用2D线程网格？
 *    提示：矩阵是2D结构，映射更自然
 *
 * 2. 访问B[k][col]时的内存模式？
 *    提示：列访问，跨步访问，对缓存不友好
 *
 * 3. 每个线程读取多少次全局内存？
 *    提示：A读K次，B读K次，C写1次，总共2K+1次
 *
 * 4. 为什么性能远低于理论峰值？
 *    提示：内存带宽限制，不是计算限制
 *
 * 5. 如果M, N, K不是blockDim的倍数？
 *    提示：边界检查 (row < M && col < N) 保证正确性
 */
