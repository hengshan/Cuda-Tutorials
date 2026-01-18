# 第8课补充：Cooperative Groups 矩阵乘法优化

## 课程定位 (1分钟)

**这是什么？**
- 第7课的延续：矩阵乘法tiling优化
- 第8课的应用：用Cooperative Groups改进代码
- 从reduce扩展到matmul：CG的通用性展示

**学习路径**：
```
第7课: 矩阵乘法 + Tiling     (传统API)
第8课: Cooperative Groups    (reduce示例)
    ↓
第8课补充: CG + 矩阵乘法     (结合两者)
```

---

## 今天要解决的问题 (1分钟)

**第7课的代码有什么问题？**

```cuda
// 传统方式（第7课）
__global__ void matmul_tiled(float *A, float *B, float *C, ...) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < numTiles; t++) {
        // 加载tile
        As[ty][tx] = A[...];
        Bs[ty][tx] = B[...];
        __syncthreads();  // ← 不够明确

        // 计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // ← 为什么需要第二次？不清楚
    }
}
```

**问题**：
1. `__syncthreads()` 意图不明确
2. 两次同步的原因需要注释说明
3. 无法利用warp级优化
4. 代码可读性差

---

## Cooperative Groups 能带来什么？ (2分钟)

### 优势1：代码意图更清晰

**传统方式**：
```cuda
__syncthreads();  // 等什么？为什么等？
```

**CG方式**：
```cuda
cg::thread_block block = cg::this_thread_block();
block.sync();  // "整个block同步"，意图明确
```

### 优势2：支持多级优化

```
Block级别:  block.sync()           ← 加载tile后同步
Warp级别:   tile32.sync()          ← warp内计算优化
Thread级别: 每个线程独立计算
```

### 优势3：灵活的组织方式

**可以这样做**：
- 用tile32做warp级reduction（减少shared memory压力）
- 用tile.shfl_down()做高效通信
- 用coalesced_group做动态分组

---

## 核心概念速讲 (2分钟)

### 1. CG在矩阵乘法中的应用点

**应用点1：同步语义**
```cuda
// 加载完tile，需要同步
block.sync();  // vs __syncthreads()

// 用完tile，需要同步
block.sync();  // 意图：等所有线程用完再加载下一个
```

**应用点2：Warp级部分和**
```cuda
// 传统：每个线程独立累加
for (int k = 0; k < TILE_SIZE; k++) {
    sum += As[ty][k] * Bs[k][tx];
}

// CG优化：warp协作计算
auto tile = cg::tiled_partition<TILE_SIZE>(block);
for (int k = 0; k < TILE_SIZE; k += WARP_SIZE) {
    // warp级向量化加载和计算
    ...
}
```

**应用点3：Cooperative加载**
```cuda
// 所有线程协作加载tile（更规整的访问模式）
auto tile = cg::tiled_partition<32>(block);
int load_idx = tile.thread_rank();
// 使用tile.thread_rank()让加载模式更清晰
```

---

### 2. 今天的优化思路

**渐进式优化**：

```
Version 1: 基础Tiling（第7课复习）
    ↓
Version 2: CG基础版（替换__syncthreads）
    ↓ 代码清晰度 +20%
Version 3: CG + Warp Tiles（warp级优化）
    ↓ 性能 +5-10%
Version 4: CG + 协作加载（优化访存模式）
    ↓ 性能 +5-10%
```

---

## 今天的代码结构

### Version 1: 基础Tiling（复习）
```cuda
__global__ void matmul_tiled_basic(float *A, float *B, float *C,
                                    int M, int N, int K) {
    // 传统方式：__syncthreads()
    // 作为baseline
}
```

### Version 2: CG基础（同步优化）
```cuda
__global__ void matmul_tiled_cg_basic(float *A, float *B, float *C,
                                       int M, int N, int K) {
    cg::thread_block block = cg::this_thread_block();

    // 加载tile
    As[ty][tx] = A[...];
    Bs[ty][tx] = B[...];
    block.sync();  // ← 清晰：等待tile加载完成

    // 计算
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
    block.sync();  // ← 清晰：等待计算完成，准备下一个tile
}
```

### Version 3: CG + Warp优化
```cuda
__global__ void matmul_tiled_cg_warp(float *A, float *B, float *C,
                                      int M, int N, int K) {
    cg::thread_block block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    // 使用warp级reduction减少shared memory读取
    float partial_sum = 0.0f;

    // 计算部分和
    for (int k = warp.thread_rank(); k < TILE_SIZE; k += warp.size()) {
        partial_sum += As[ty][k] * Bs[k][tx];
    }

    // warp内reduction（可选，如果需要）
    // sum = warpReduceSum(warp, partial_sum);
}
```

### Version 4: CG + 协作加载
```cuda
__global__ void matmul_tiled_cg_coop(float *A, float *B, float *C,
                                      int M, int N, int K) {
    cg::thread_block block = cg::this_thread_block();

    // 使用thread_rank()实现更规整的加载模式
    int block_rank = block.thread_rank();
    int total_threads = block.size();

    // 协作加载As和Bs（向量化访问）
    int tile_elements = TILE_SIZE * TILE_SIZE;
    for (int i = block_rank; i < tile_elements; i += total_threads) {
        int local_row = i / TILE_SIZE;
        int local_col = i % TILE_SIZE;
        // 加载到shared memory
    }
    block.sync();

    // ... 计算 ...
}
```

---

## 性能预期

```
版本                  相对性能    代码清晰度    说明
─────────────────────────────────────────────────────────
V1: 基础Tiling        100%        ⭐⭐         baseline
V2: CG基础            100-102%    ⭐⭐⭐⭐     清晰但性能相近
V3: CG+Warp          105-110%    ⭐⭐⭐       warp优化
V4: CG+协作加载      110-115%    ⭐⭐⭐⭐     最优性能+清晰
```

**关键点**：
- CG的主要优势是**代码清晰度**和**可维护性**
- 性能提升是**次要**收益（5-15%）
- 为未来硬件特性做准备（Thread Block Clusters等）

---

## 关键技术点

### 1. Block同步的语义

```cuda
// 传统
__syncthreads();  // 功能：同步，语义：不明确

// CG
block.sync();     // 功能：同步，语义：明确（整个block）
```

### 2. Warp级操作

```cuda
auto warp = cg::tiled_partition<32>(block);

// 获取warp内位置
int lane_id = warp.thread_rank();  // vs threadIdx.x % 32

// Warp级通信
float neighbor = warp.shfl_down(val, offset);  // vs __shfl_down_sync
```

### 3. 协作加载模式

```cuda
// 传统：每个线程固定位置
As[ty][tx] = A[...];

// CG协作：所有线程一起加载整个tile
int rank = block.thread_rank();
for (int i = rank; i < TILE_SIZE*TILE_SIZE; i += block.size()) {
    // 更灵活的加载模式
}
```

---

## 与第8课的联系

**第8课（reduce）**：
- CG用于reduction（找最大值）
- 展示了tile归约的通用模板
- 重点：`tileReduceMax<TILE_SIZE>`

**第8课补充（matmul）**：
- CG用于matmul（计算密集）
- 展示了CG在复杂算法中的应用
- 重点：多级同步 + warp优化

**共同点**：
- 都用`thread_block`做block级同步
- 都用`tiled_partition`做warp级操作
- 代码清晰度和可维护性提升

---

## 本节课目标

学完后你应该能：
- ✅ 用CG重写第7课的矩阵乘法
- ✅ 理解`block.sync()`的清晰语义
- ✅ 使用warp tiles优化计算
- ✅ 实现协作加载模式
- ✅ 对比CG和传统API的优劣

---

## 接下来：20分钟 Live Coding

重点：
1. 复习第7课的tiling算法
2. 用CG替换`__syncthreads()`
3. 添加warp级优化
4. 实现协作加载
5. 性能和代码质量对比

**准备好探索CG在矩阵乘法中的应用了吗？** 🚀

---

## 下一课预告

第9课：**Thread Block Clusters**
- 跨block协作（RTX 5090新特性）
- Distributed Shared Memory
- Grid级同步
- 更大规模的并行（超越单个block）

CG的终极形态等着你！

---

*第8课补充 | Cooperative Groups矩阵乘法 | 清晰代码 + 性能优化*
