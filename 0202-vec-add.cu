#include <stdio.h>

__global__ void vectorAdd1D(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int N = 16; // 向量长度
    int h_a[N], h_b[N], h_c[N];

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 10;
    }

    // 分配设备内存
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 启动 kernel
    vectorAdd1D<<<2, 8>>>(d_a, d_b, d_c, N); // 2 blocks × 8 threads = 16 线程

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
