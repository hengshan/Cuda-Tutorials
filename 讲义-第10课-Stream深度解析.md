# 第10课讲义：CUDA Stream 深度解析 - 何时有效，何时无效

## 课前回顾 (2分钟)

### 前9课我们学到了什么？

**优化之路：让单个任务跑得更快**

```
第3课: 找最大值 - 朴素版
       ↓ 原子操作太慢
第4课: 共享内存优化 (3-5x加速)
       ↓ 还有优化空间
第5课: Warp Shuffle (再1.5-2x加速)
       ↓ 总加速60倍！

第6课: 矩阵乘法 - 朴素版
       ↓ 全局内存访问太多
第7课: Tiling技术 (10-20x加速)
       ↓ 共享内存 + 数据复用
```

**今天的问题**：单个kernel已经很快了，还能更快吗？

**答案**：能！但不是优化kernel本身，而是优化**整个系统**！

---

## 🎯 先给结论！(3分钟)

### CUDA Stream 的核心真相

**❌ Stream 不会让单个 kernel 变快！**

**✅ Stream 的唯一作用：在 Copy Engine 与 SM 之间制造时间错位（latency hiding）**

**关键洞察**：
```
没有时间错位 = 没有 Stream 的性能价值

时间错位来源：
1. 多 batch：Batch_i 传输时，Batch_i-1 在计算
2. 连续数据流：数据持续到达，需要不断 H2D
3. 多 kernel stage：Stage_1(Batch_i) 与 Stage_2(Batch_i-1) 并行
```

**工程直觉**：
- 如果你只有**一个大任务**，stream 基本无效
- 如果你有**多个小任务连续到达**，stream 才真正有用

---

## ⚠️ 澄清常见误区 (5分钟)

### 误区1：把大矩阵切块 + stream 会更快

**错误想法**：
```cuda
// 把 4096×4096 矩阵切成 4 块，每块用一个 stream
for (int i = 0; i < 4; i++) {
    cudaMemcpyAsync(..., streams[i]);  // 传输 1/4 数据
    matmul_kernel<<<..., streams[i]>>>(...);  // 计算 1/4 结果
}
```

**为什么通常更慢？**
1. **SM 饱和问题**：单个大矩阵已经充分利用 GPU，切块反而降低并行度
2. **额外开销**：Stream 管理 + 多次 kernel 启动
3. **没有时间错位**：所有数据一开始就就绪，无法隐藏传输延迟

**正确场景**：
- 不是一个大矩阵切 4 块
- 而是 4 个不同的矩阵（4 个 batch）连续到达

---

### 误区2：单次大 GEMM 用 stream 加速

**场景**：
```cuda
// 单次大矩阵乘法
cudaMemcpy(d_A, h_A, ...);  // 传 A
cudaMemcpy(d_B, h_B, ...);  // 传 B
cublasGemm(...);            // 计算
cudaMemcpy(h_C, d_C, ...);  // 传回
```

**能用 stream 加速吗？**
❌ **不能！**

**原因**：
1. 只有一个 batch，无法制造时间错位
2. 所有数据一开始就就绪，无连续数据流
3. 即使用 stream，也是完全串行执行

**改进方向**：
- 不是优化单次调用，而是优化整个应用流程
- 如果有多次 GEMM 调用（batch inference），才考虑 stream

---

### 误区3：数学上的 batch ≠ 系统层面的 pipeline

**数学视角（cuBLAS batched GEMM）**：
```cuda
// 一次性传所有数据
cudaMemcpy(d_A_array, h_A_array, ...);
cublasSgemmBatched(...);  // 内部批处理
```
- 数据一次性就绪
- 内部优化，但无 stream pipeline

**系统视角（Stream pipeline）**：
```cuda
// 数据连续到达
for (int i = 0; i < num_batches; i++) {
    cudaMemcpyAsync(d_A[i], h_A[i], ..., streams[i % 4]);
    kernel<<<..., streams[i % 4]>>>(...);
}
```
- 数据分批到达
- 传输与计算重叠
- **这才是 stream 的价值！**

---

## ✅ Stream 真正有效的三个必要条件 (5分钟)

**至少满足一个，stream 才可能有效：**

### 条件1：多 batch / 连续数据流（pipeline）

**典型场景**：
- **AI 推理**：模型权重固定，输入图像连续到达
- **视频处理**：帧接帧处理
- **流式计算**：数据从 CPU/网络持续到达

**时间线**：
```
Stream_0: [H2D_0][Compute_0]           [H2D_4][Compute_4]
Stream_1:     [H2D_1][Compute_1]           [H2D_5]...
Stream_2:         [H2D_2][Compute_2]           ...
Stream_3:             [H2D_3][Compute_3]
          └────── H2D 与 Compute 重叠 ──────┘
```

---

### 条件2：CPU / IO 持续产生新数据，需要不断 H2D

**典型场景**：
- **实时采集**：传感器数据实时上传 GPU 处理
- **数据增强**：CPU 端生成增强数据，GPU 端训练
- **在线服务**：请求不断到达，每个请求独立处理

**关键点**：
- 数据不是一开始就全部就绪
- CPU 和 GPU 协同工作
- Stream 隐藏 CPU → GPU 传输延迟

---

### 条件3：多 kernel stage + 多 batch，可错位执行

**典型场景**：
```
Pipeline: Preprocess → Compute → Postprocess

Batch_0: [Pre_0] [Comp_0] [Post_0]
Batch_1:     [Pre_1] [Comp_1] [Post_1]
Batch_2:         [Pre_2] [Comp_2] [Post_2]
         └──── 各 stage 错位执行 ────┘
```

**为什么有效？**
- Batch_i 在 Preprocess 时，Batch_i-1 在 Compute
- 多个 kernel 在不同 batch 间流水线
- 类似 CPU 流水线：取指、译码、执行同时进行

---

## 🔧 硬件视角解释 (5分钟)

### GPU 内部的角色分工

```
┌──────────────────────────────────────────────────────────┐
│                      GPU 芯片                             │
│                                                           │
│   ┌──────────────────┐          ┌───────────────────┐   │
│   │  Copy Engine     │          │  SM 计算单元       │   │
│   │  (DMA 引擎)      │          │  (Streaming       │   │
│   │                  │          │   Multiprocessor) │   │
│   │  - H2D 传输      │          │  - Kernel 执行     │   │
│   │  - D2H 传输      │          │  - 并行计算        │   │
│   └──────────────────┘          └───────────────────┘   │
│          ↕                              ↕                 │
│      PCIe 总线                      GPU 显存              │
│                                                           │
└──────────────────────────────────────────────────────────┘

⚠️  关键：Copy Engine 和 SM 是独立硬件，可以并行工作！
```

### 为什么 Stream 能让 Copy 与 Compute 并行？

**单 Stream（或 Default Stream）**：
```
时间轴：
────────────────────────────────────────────►
  [H2D] [Kernel] [D2H] [H2D] [Kernel] [D2H]
   ↑       ↑       ↑
  DMA     SM     DMA    (串行，资源浪费)

Copy Engine 使用率: 33%
SM 使用率:         33%
总使用率:          33%  ❌
```

**多 Stream**：
```
时间轴：
────────────────────────────────────────────►
Stream_0: [H2D_0] [Kernel_0] [D2H_0]
Stream_1:     [H2D_1] [Kernel_1] [D2H_1]
Stream_2:         [H2D_2] [Kernel_2] [D2H_2]
             ↑       ↑
            DMA     SM    (并行，充分利用)

Copy Engine 使用率: 80-90%  ✅
SM 使用率:         80-90%  ✅
总使用率:          80-90%  ✅
```

---

### Copy Engine 数量的影响

**查询 GPU 的 Copy Engine 数量**：
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("异步引擎数: %d\n", prop.asyncEngineCount);
```

**影响**：

| Copy Engine 数 | H2D + D2H 并行？ | 说明 |
|---------------|----------------|------|
| 1 个 | ❌ 不能 | H2D 和 D2H 必须排队 |
| 2 个 | ✅ 能 | H2D 和 D2H 可同时进行 |

**RTX 5090**：通常有 2 个 Copy Engine
- 一个专门做 H2D
- 一个专门做 D2H
- 可以同时传输双向数据

---

### Pinned Memory 的必要性

**为什么异步传输必须用 Pinned Memory？**

| 内存类型 | malloc | cudaMallocHost |
|---------|--------|----------------|
| 物理位置 | 可被 OS 换页 | 锁定在 RAM |
| GPU DMA 访问 | ❌ 需要中间缓冲 | ✅ 直接访问 |
| 带宽 | ~6-12 GB/s | ~25 GB/s (PCIe 4.0) |
| **异步支持** | ❌ **不支持** | ✅ **必需** |

**错误示例**（看起来异步，实际同步）：
```cuda
float *h_data = (float*)malloc(size);  // 普通内存
cudaMemcpyAsync(d_data, h_data, size, ..., stream);
// ⚠️  实际是同步！驱动会等待完成
```

**正确示例**：
```cuda
float *h_data;
cudaMallocHost(&h_data, size);  // Pinned Memory
cudaMemcpyAsync(d_data, h_data, size, ..., stream);
// ✅ 真正异步
```

**硬件原理**：
- Pinned Memory 物理地址固定，DMA 可直接访问
- 普通内存可能被换页，DMA 无法直接访问
- 异步传输需要 DMA 独立工作，不能等 CPU

---

## 💡 典型正例：从代码到时间线 (8分钟)

### 正例1：Batch GEMM Pipeline（AI 推理典型场景）

**应用场景**：
- 模型权重 B 固定在 GPU
- 输入 A 连续到达（用户请求、视频帧等）
- 每个 batch 独立，可流水线

**代码结构**（简化）：
```cuda
// 固定权重 B 在 GPU
float *d_B;
cudaMalloc(&d_B, K * N * sizeof(float));
cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

// 创建 4 个 stream
cudaStream_t streams[4];
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
}

// 每个 stream 需要独立的 A 缓冲区（避免覆盖）
float *d_A[4];
for (int i = 0; i < 4; i++) {
    cudaMalloc(&d_A[i], M * K * sizeof(float));
}

// 流水线发射所有 batch
for (int b = 0; b < num_batches; b++) {
    int sid = b % 4;

    // 异步 H2D：传输输入 A
    cudaMemcpyAsync(d_A[sid], h_A_batches[b], ..., streams[sid]);

    // 异步 Compute：C = A × B
    matmul<<<..., streams[sid]>>>(d_A[sid], d_B, d_C[b], ...);
}

cudaDeviceSynchronize();
```

**时间线解释**（重点！）：

```
时间：0ms     2ms     4ms     6ms     8ms     10ms
     ─────────────────────────────────────────────────►

Stream_0: [H2D_A0][Compute_0]           [H2D_A4][Compute_4]
                     ▲                               ▲
                 ┌───┴───┐                       ┌───┴───┐
Stream_1:        │[H2D_A1]│[Compute_1]           │[H2D_A5]│...
                 └───────┬┘     ▲                └───────┘
                         │  ┌───┴───┐
Stream_2:                └──│[H2D_A2]│[Compute_2]
                            └───────┬┘     ▲
                                    │  ┌───┴───┐
Stream_3:                           └──│[H2D_A3]│[Compute_3]
                                       └───────┘

硬件使用：
  Copy Engine: ████████████████████████████████  (持续工作)
  SM:              ████████████████████████████  (持续工作)
              └─────── 高效重叠！ ────────┘
```

**谁在 Copy，谁在 Compute**：
- **t=0-1ms**：Copy Engine 传输 A0，SM 空闲
- **t=1-3ms**：Copy Engine 传输 A1，**SM 计算 A0** ← 重叠！
- **t=2-4ms**：Copy Engine 传输 A2，**SM 计算 A1** ← 重叠！
- **t=3-5ms**：Copy Engine 传输 A3，**SM 计算 A2** ← 重叠！
- **t=4-6ms**：Copy Engine 传输 A4，**SM 计算 A3** ← 重叠！

**性能结果**（实测 RTX 5090）：
- 同步执行：2.16 ms
- 流水线执行：0.90 ms
- **加速比：2.39x** ✅

**为什么有效？**
1. 满足条件1：多 batch，数据连续到达
2. 传输与计算时间接近，重叠空间大
3. Copy Engine 和 SM 充分并行工作

---

### 正例2：Multi-kernel Pipeline（数据处理流水线）

**应用场景**：
- Stage 1: Preprocess（归一化、数据增强）
- Stage 2: Compute（主计算，如卷积）
- Stage 3: Postprocess（激活函数、NMS）

**代码结构**：
```cuda
cudaStream_t streams[4];
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
}

for (int b = 0; b < num_batches; b++) {
    int sid = b % 4;

    // 异步 H2D
    cudaMemcpyAsync(d_input[sid], h_input[b], ..., streams[sid]);

    // Stage 1: Preprocess
    preprocess_kernel<<<..., streams[sid]>>>(d_input[sid], d_temp1[sid], ...);

    // Stage 2: Compute
    compute_kernel<<<..., streams[sid]>>>(d_temp1[sid], d_temp2[sid], ...);

    // Stage 3: Postprocess
    postprocess_kernel<<<..., streams[sid]>>>(d_temp2[sid], d_output[sid], ...);

    // 异步 D2H
    cudaMemcpyAsync(h_output[b], d_output[sid], ..., streams[sid]);
}
```

**时间线解释**：

```
时间：0ms    2ms    4ms    6ms    8ms    10ms   12ms
     ─────────────────────────────────────────────────────►

Stream_0: [H2D][Pre][Comp][Post][D2H]              [H2D][Pre]...
                  ▲    ▲     ▲    ▲
              ┌───┴────┴─────┴────┴──┐
Stream_1:     │[H2D][Pre][Comp][Post]│[D2H]        [H2D]...
              └──┬────┬─────┬─────┬──┘  ▲
                 │ ┌──┴─────┴─────┴──┐  │
Stream_2:        └─│[H2D][Pre][Comp]│[Post][D2H]  [H2D]...
                   └──┬──────┬──────┬──┘     ▲
                      │   ┌──┴──────┴──┐     │
Stream_3:             └───│[H2D][Pre]│[Comp][Post][D2H]...
                          └──────────┘

观察：Batch_0 在 Post 时，Batch_1 在 Comp，Batch_2 在 Pre
```

**为什么有效？**
1. 满足条件3：多 kernel stage + 多 batch
2. 不同 stage 在不同 batch 间错位执行
3. SM 几乎无空闲（一直有 kernel 在执行）

**性能结果**（实测）：
- 同步执行：3.91 ms
- 流水线执行：3.34 ms
- **加速比：1.17x**

**加速不如场景1的原因**：
- Kernel 太快（每个 < 1ms），传输占比小
- 多 stage 之间有依赖，无法完全并行
- 但仍有效！避免了 GPU 空闲

---

### 正例3：Double Buffering（经典 Ping-Pong 模式）

**应用场景**：
- 生产者-消费者模式
- 数据连续流式到达
- 经典的性能优化模式

**代码结构**：
```cuda
// 两个 stream
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

// 两个缓冲区（ping-pong）
float *d_bufferA, *d_bufferB;
cudaMalloc(&d_bufferA, size);
cudaMalloc(&d_bufferB, size);

for (int i = 0; i < num_batches; i++) {
    cudaStream_t stream = (i % 2 == 0) ? stream0 : stream1;
    float *d_buffer = (i % 2 == 0) ? d_bufferA : d_bufferB;

    // 异步 H2D → Compute → D2H
    cudaMemcpyAsync(d_buffer, h_input[i], ..., stream);
    kernel<<<..., stream>>>(d_buffer, ...);
    cudaMemcpyAsync(h_output[i], d_buffer, ..., stream);
}
```

**时间线**：
```
Buffer_A (Stream_0): [H2D_0][Compute_0][D2H_0]       [H2D_2][Compute_2][D2H_2]
                               ▲         ▲               ▲
                           ┌───┴─────────┴───────────────┴──┐
Buffer_B (Stream_1):       │[H2D_1][Compute_1][D2H_1]       │[H2D_3]...
                           └────────────────────────────────┘

硬件使用：
  H2D:  ███ ███ ███ ███ ███ ███  (交替传输)
  SM:     ███ ███ ███ ███ ███    (持续计算)
  D2H:      ███ ███ ███ ███ ███  (交替传回)
        └──── 三者高度重叠 ────┘
```

**为什么叫 Ping-Pong？**
- Buffer A 在计算时，Buffer B 在传输（Ping）
- Buffer B 在计算时，Buffer A 在传输（Pong）
- 轮流切换，隐藏传输延迟

---

## 🔬 Nsight Systems 观察指南 (4分钟)

### 应该看哪些轨道？

**重点轨道**（按优先级）：
1. **CUDA GPU Kernels** - Kernel 执行时间线
2. **CUDA Memcpy (HtoD)** - Host → Device 传输
3. **CUDA Memcpy (DtoH)** - Device → Host 传输
4. **CUDA Streams** - Stream 活动情况

**操作**：
```bash
# 生成报告
nsys profile -o stream_report ./stream_effective

# 打开 GUI（如果可用）
nsys-ui stream_report.nsys-rep

# 命令行查看
nsys stats --report cuda_api_sum stream_report.nsys-rep
```

---

### 判断 Stream 是否生效的"肉眼规则"

**🟢 有效信号**：
```
时间轴（放大后）：
─────────────────────────────────────────►
CUDA Memcpy:  ███     ███     ███     ███
                 │       │       │
CUDA Kernels:    ███████████████████████
                 └─── 重叠！─────┘
```
**观察点**：
- 某一时刻**同时**存在 Memcpy 和 Kernel
- Kernel 之间几乎无空隙
- 多个 Stream 的操作交错排列

**🔴 无效信号**：
```
时间轴：
─────────────────────────────────────────►
CUDA Memcpy:  ███       ███       ███
                 │         │         │
CUDA Kernels:    │ ███     │ ███     │ ███
                 └─ 串行！──┘
```
**观察点**：
- Memcpy 和 Kernel 完全不重叠
- Kernel 之间有明显空隙
- 虽然用了 Stream，但实际串行

---

### 为什么有时看不到 Overlap？

**原因1：Kernel 太大（计算瓶颈）**
```
CUDA Memcpy:  █           █           █
CUDA Kernels: ████████████████████████████
              └─ Kernel 太长，传输被隐藏 ─┘
```
- Kernel 执行时间 >> 传输时间
- 传输已经完全被隐藏，无法进一步优化
- **结论**：Stream 对这个场景帮助有限

**原因2：Copy 太小（传输可忽略）**
```
CUDA Memcpy:  ▂  ▂  ▂  ▂  ▂  ▂  ▂  ▂
CUDA Kernels: ███ ███ ███ ███ ███ ███
              └─ 传输太快，重叠空间小 ─┘
```
- 传输时间 << 计算时间
- 即使重叠，收益也很小
- **结论**：Stream 收益有限

**原因3：Copy Engine 只有 1 个**
```
Stream_0 H2D: ███     ███
Stream_1 H2D:     ███     ███
              └─ 排队！─┘
```
- H2D 和 D2H 无法同时进行
- 多个 Stream 的 H2D 仍然串行
- **解决**：升级到有 2 个 Copy Engine 的 GPU

**原因4：Default Stream / 隐式同步**
```cuda
// 错误：忘记指定 stream
cudaMemcpyAsync(...);  // 用了 default stream
kernel<<<...>>>(...);  // default stream

// 正确：显式指定
cudaMemcpyAsync(..., stream);
kernel<<<..., stream>>>(...);
```
- Default Stream 会同步所有操作
- 看起来异步，实际串行
- **解决**：显式创建和使用 Stream

---

## 🎯 工程判断口诀 (2分钟)

### ✅ 什么时候"值得用 Stream"？

**满足以下至少一个条件：**

1. **多 batch 处理**
   - 例子：推理服务、视频处理、批量计算
   - 判断：有多个独立任务连续到达

2. **数据连续到达**
   - 例子：实时采集、流式计算、在线服务
   - 判断：数据不是一开始全部就绪

3. **多 kernel stage**
   - 例子：Preprocess → Compute → Postprocess
   - 判断：处理流程有多个阶段

4. **传输时间 ≈ 计算时间**
   - 例子：中等规模 kernel + 较大数据传输
   - 判断：Memcpy 和 Kernel 时间接近（1:1 到 1:3）

---

### ❌ 什么时候"可以直接不用想"？

**以下场景 Stream 基本无效：**

1. **单 batch + 单 kernel**
   - 例子：单次矩阵乘法、单次卷积
   - 原因：无法制造时间错位

2. **Kernel 已饱和 SM**
   - 例子：大规模 GEMM（4096×4096+）
   - 原因：计算 >> 传输，GPU 已满负荷

3. **所有数据已就绪**
   - 例子：数据一次性加载，批量处理
   - 原因：无连续数据流

4. **传输时间 << 计算时间**
   - 例子：小数据 + 复杂计算
   - 原因：传输开销可忽略，重叠收益小

---

### 快速决策树

```
开始
  │
  ├─ 有多个 batch 吗？
  │   ├─ 是 → 用 Stream ✅
  │   └─ 否 → 继续
  │
  ├─ 数据连续到达吗？
  │   ├─ 是 → 用 Stream ✅
  │   └─ 否 → 继续
  │
  ├─ 有多个 kernel stage 吗？
  │   ├─ 是 → 用 Stream ✅
  │   └─ 否 → 继续
  │
  └─ 传输 ≈ 计算时间吗？
      ├─ 是 → 尝试 Stream（可能有小幅提升）
      └─ 否 → 不用 Stream ❌
```

---

## 今天的代码结构 (1分钟)

### 文件：1003-stream.cu

**包含三个有效场景**：

**1. Batch GEMM Pipeline**
```cuda
batchGEMM_synchronous(...)   // 基准：串行执行
batchGEMM_pipelined(...)     // 优化：流水线
// 实测加速：2.39x ✅
```

**2. Multi-kernel Pipeline**
```cuda
multiKernelPipeline_synchronous(...)
multiKernelPipeline_pipelined(...)
// 实测加速：1.17x ✅
```

**3. Double Buffering**
```cuda
doubleBuffering(...)
// 经典模式：Ping-Pong 缓冲
```

**运行结果示例**：
```bash
$ ./stream_effective
================================================================================
场景1：Batch GEMM Pipeline
================================================================================
同步执行: 2.16 ms
流水线:   0.90 ms
加速比:   2.39x  ✅ Stream 有效！
```

---

## 课程总结 (2分钟)

### 从"单点优化"到"系统思维"

**前9课：微观优化**
- 目标：让单个 kernel 跑得更快
- 方法：算法、内存、warp
- 思维：代码级优化

**第10课：宏观优化**
- 目标：让整个应用更高效
- 方法：并发、流水线、隐藏延迟
- 思维：系统级优化

**关键转变**：
```
错误思维：如何让这个 kernel 更快？
正确思维：如何让整个系统更高效？

错误思维：Stream 能加速我的 GEMM 吗？
正确思维：我的应用有多 batch 吗？有连续数据流吗？
```

---

### 核心要点回顾

1. **Stream 不会让单个 kernel 变快**
   - 只能通过时间错位隐藏延迟

2. **三个必要条件（至少满足一个）**
   - 多 batch / 连续数据流
   - CPU/IO 持续产生新数据
   - 多 kernel stage + 多 batch

3. **硬件关键**
   - Copy Engine 和 SM 是独立硬件
   - Pinned Memory 必需
   - 异步引擎数影响 H2D/D2H 并发

4. **工程判断**
   - 单 batch + 单 kernel → 不用
   - 多 batch + 连续到达 → 用
   - 传输 << 计算 → 不用
   - 传输 ≈ 计算 → 用

---

## 附录：快速参考

### CUDA Streams API

```cuda
// 创建和销毁
cudaStreamCreate(&stream);
cudaStreamDestroy(stream);

// 异步操作
cudaMemcpyAsync(..., stream);
kernel<<<grid, block, 0, stream>>>();

// 同步
cudaStreamSynchronize(stream);  // 等待指定 stream
cudaDeviceSynchronize();        // 等待所有 stream

// 查询
cudaStreamQuery(stream);  // 非阻塞查询 stream 状态
```

---

### Pinned Memory API

```cuda
// 分配和释放
cudaMallocHost(&ptr, size);
cudaFreeHost(ptr);

// 高级选项
cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
cudaHostAlloc(&ptr, size, cudaHostAllocPortable);     // 多 GPU
cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined); // 优化写
```

---

### Profiling 命令

```bash
# Nsight Systems（时间线分析）
nsys profile -o report ./app
nsys-ui report.nsys-rep

# 命令行查看 CUDA API 统计
nsys stats --report cuda_api_sum report.nsys-rep

# 查看 GPU trace
nsys stats --report cuda_gpu_trace report.nsys-rep
```

---

**讲义结束**

*CUDA 13.1教程 | RTX 5090 Blackwell | 第10课 Stream 深度解析 | 2025*
