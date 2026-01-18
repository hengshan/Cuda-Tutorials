// GPU (整个显卡)
// ├── GPC (Graphics Processing Cluster) - 多个
// │   ├── SM (Streaming Multiprocessor) - 每个GPC包含多个SM
// │   │   ├── CUDA Core (计算单元) - 每个SM包含多个CUDA核心
// │   │   ├── Tensor Core - AI计算单元
// │   │   ├── RT Core - 光线追踪单元
// │   │   ├── 共享内存 (Shared Memory)
// │   │   ├── 寄存器文件 (Register File)
// │   │   └── 调度器 (Warp Scheduler)
// │   └── L1缓存、纹理单元等
// └── L2缓存、显存控制器等全局资源
/*
关键理解:
- GPU硬件架构是多层次的：GPU → GPC → SM → 执行单元
- 每个层次都有特定的功能和限制
- Thread Block Cluster利用了GPC级别的硬件保证
- 不同类型的执行单元针对不同类型的计算优化
- 内存层次决定了数据访问的性能特征
*/
// 创建测试文件：test_cuda.cu
#include <iostream>
#include <cuda_runtime.h>

// 简单的CUDA kernel
__global__ void hello_rtx5090() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from RTX 5090! Thread %d in block %d, the id is: %d\n",
           threadIdx.x, blockIdx.x, idx);
}

// 设备信息查询函数
void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "检测到 " << deviceCount << " 个CUDA设备\n\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "=== 设备 " << i << " 信息 ===\n";
        std::cout << "名称: " << prop.name << "\n";
        std::cout << "计算能力: " << prop.major << "." << prop.minor << "\n";
        std::cout << "全局内存: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
        std::cout << "流式多处理器SM数量: " << prop.multiProcessorCount << "\n";
        std::cout << "每个SM的最大线程数: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "每个块的最大线程数: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "warp size: " << prop.warpSize << "\n";
        std::cout << "每个块的最大维度: (" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
        std::cout << "网格的最大维度: (" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
        std::cout << "共享内存每块: " << prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "共享内存每block optin: " << prop.sharedMemPerBlockOptin / 1024 << " KB\n";
        std::cout << "共享内存每SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
        std::cout << "常量内存: " << prop.totalConstMem / 1024 << " KB\n";
        std::cout << "内存总线宽度: " << prop.memoryBusWidth << " bits\n";
        std::cout << "L2缓存大小: " <<  prop.l2CacheSize/(1024*1024) << "M \n";
        std::cout << "支持L2缓存持久化大小: " <<  prop.persistingL2CacheMaxSize/(1024*1024) << "M\n";
        std::cout << "最大的访问策略窗口大小: " <<  prop.accessPolicyMaxWindowSize/(1024*1024)<< "M \n";

        std::cout << "是否支持并发内核执行: " << (prop.concurrentKernels ? "是" : "否") << "\n";
        std::cout << "是否支持在 pinned 系统内存和设备内存间异步拷贝: " << (prop.asyncEngineCount ? "是" : "否") << "\n"; // asyncEngineCount > 0 表示支持
        std::cout << "是否支持映射到设备的 pinned 系统内存: " << (prop.canMapHostMemory ? "是" : "否") << "\n";

        std::cout << "异步引擎数量 (DMA): " << prop.asyncEngineCount << "\n"; // 1=只支持单向，2=支持双向
        std::cout << "是否支持统一虚拟地址 (UVA): " << (prop.unifiedAddressing ? "是" : "否") << "\n";
        std::cout << "是否支持计算抢占: " << (prop.computePreemptionSupported ? "是" : "否") << "\n"; // 重要！影响长时间内核的响应性

        std::cout << "每个线程块的最大32位寄存器数量: " << prop.regsPerBlock << "\n";
        std::cout << "每个SM的最大32位寄存器数量: " << prop.regsPerMultiprocessor << "\n";
        std::cout << "设备是否集成在主板（如集显）: " << (prop.integrated ? "是" : "否") << "\n";
        std::cout << "是否支持ECC内存: " << (prop.ECCEnabled ? "是" : "否") << "\n";
        std::cout << "是否支持 Cooperative Kernel Launch (网格同步): " << (prop.cooperativeLaunch ? "是" : "否") << "\n";
        std::cout << "纹理对齐要求: " << prop.textureAlignment << " bytes\n";
        std::cout << "纹理Pitch最大大小: " << prop.texturePitchAlignment << " bytes\n";
        std::cout << "设备是否支持与特定CPU群体的原子操作: " << (prop.hostNativeAtomicSupported ? "是" : "否") << "\n";

        std::cout << "一维表面的最大宽度: " << prop.maxSurface1D << "\n";
        std::cout << "二维表面的最大维度 (宽 x 高): " << prop.maxSurface2D[0] << " x " << prop.maxSurface2D[1] << "\n";
        std::cout << "三维表面的最大维度 (宽 x 高 x 深): "
                  << prop.maxSurface3D[0] << " x "
                  << prop.maxSurface3D[1] << " x "
                  << prop.maxSurface3D[2] << "\n";
        // RTX 5090特有信息
        if (prop.major == 12 && prop.minor == 0) {  // sm_120
            std::cout << "\n🎉 RTX 5090 Blackwell 架构检测成功！\n";
            std::cout << "✅ 支持 CUDA 13.0 所有新特性\n";
            std::cout << "✅ 支持第5代Tensor Core\n";
            std::cout << "✅ 支持第4代RT Core\n";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "=== CUDA 13 + RTX 5090 安装验证 ===\n\n";

    // 打印设备信息
    print_device_info();

    // 运行简单的kernel
    std::cout << "运行测试kernel...\n";
    hello_rtx5090<<<2, 4>>>();  // 2个块，每个块4个线程

    // 等待GPU完成并检查错误
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "\n✅ CUDA 13 安装验证成功！\n";
    std::cout << "🚀 RTX 5090 已准备就绪，可以开始GPU编程之旅！\n";

    printf("=== 逻辑 vs 物理映射关系 ===\n");
    printf("逻辑概念          ←→  物理硬件\n");
    printf("─────────────────────────────────\n");
    printf("Grid             ←→  整个GPU\n");
    printf("Cluster          ←→  单个GPC内\n");
    printf("Block            ←→  单个SM内\n");
    printf("Warp (32 threads)←→  执行单元群组\n");
    printf("Thread           ←→  单个CUDA Core\n");
    printf("Shared Memory    ←→  SM内共享SRAM\n");
    printf("Global Memory    ←→  HBM/GDDR显存\n\n");

    printf("=== 关键理解要点 ===\n");
    printf("1. ️硬件是固定的：GPC数量、SM数量在制造时确定\n");
    printf("2. 软件是灵活的：threads、blocks、clusters是逻辑概念\n");
    printf("3. 调度是动态的：硬件调度器将逻辑单元映射到物理单元\n");
    printf("4. 并行是分层的：不同层次有不同的并行度和通信能力\n");
    printf("5. 优化需匹配：算法设计要考虑硬件架构特点\n");
    return 0;
}
