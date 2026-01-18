# 🚀 CUDA基础教程

## 第一步：确认环境

```bash
# 检查CUDA版本
nvcc --version

# 检查GPU信息
nvidia-smi

# 查看GPU计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# gitclone repo到本地
```

## 第二步：选择一节课开始

前面两节课是基础的Kernel的理解。推荐从**第3课**开始：

```bash
cd /your-path-to/Cuda-Tutorials
```

## 第三步：编译并运行

### 如果是RTX 5090:
```bash
nvcc -arch=sm_120 -o lesson03 0301-max-unified-naive.cu
./lesson03
```

### 如果是其他GPU（根据上面查到的compute_cap修改）:
```bash
# RTX 4090
nvcc -arch=sm_89 -o lesson03 0301-max-unified-naive.cu
./lesson03

# RTX 3090
nvcc -arch=sm_86 -o lesson03 0301-max-unified-naive.cu
./lesson03

# RTX 2080
nvcc -arch=sm_75 -o lesson03 0301-max-unified-naive.cu
./lesson03
```

## 第四步：查看结果

你应该看到类似这样的输出：
```
========================================
第3课：统一内存 + 数组求最大值
========================================

数据规模: 10000000 个整数
...
CPU结果: 999999 (耗时: 50.23 ms)
GPU结果: 999999 (耗时: 5.12 ms)

✓ 结果正确！CPU和GPU结果一致
加速比: ??x
```

## 学习流程

### 每节课标准流程：

**1. 阅读讲义（5分钟）**
```bash
# 用任意文本编辑器打开
cat 讲义-第3课.md
# 或
vim 讲义-第3课.md
# 或
code 讲义-第3课.md  # VS Code
```

**2. 看代码（10-15分钟）**
```bash
# 阅读代码，理解每个部分
vim 0301-max-unified-naive.cu
```

**3. 编译运行（5分钟）**
```bash
nvcc -arch=sm_120 -o lesson03 0301-max-unified-naive.cu
./lesson03
```

## 课程顺序

```
第3课 (统一内存)
  ↓
第4课 (共享内存)
  ↓
第5课 (Warp Shuffle)
  ↓
第6课 (矩阵乘法朴素)
  ↓
第7课 (矩阵乘法优化)
  ↓
第8课 (协作组)
  ↓
第8课补充 (矩阵乘法分片Tiling)
  ↓
第9课 (线程块簇)
  ↓
第10课 (Streams异步)
  ↓
第11课 (Cuda Graph与Pipeline思想)
```

### 性能分析（可选）
```bash
# Nsight Systems - 查看时间线
nsys profile -o report ./lesson03
nsys-ui report.nsys-rep

# Nsight Compute - 详细kernel分析
ncu --set full -o report ./lesson03
ncu-ui report.ncu-rep
```

## 常见问题速查

### 编译错误：`nvcc: command not found`
**解决**：CUDA未安装或未加入PATH
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 编译错误：`unsupported GNU version`
**解决**：GCC版本太新，降级或指定旧版本
```bash
nvcc -ccbin gcc-11 -arch=sm_120 -o lesson03 0301-max-unified-naive.cu
```

### 运行错误：`no CUDA-capable device is detected`
**解决**：
1. 检查GPU驱动：`nvidia-smi`
2. 检查CUDA安装：`nvcc --version`
3. 可能在没有GPU的机器上

### 性能比课程说的差很多
**原因**：
1. GPU型号不同（非RTX 5090）
2. 数据规模不同
3. 系统负载

**重点**：关注**加速比**（优化前vs后），不是绝对数字

## 学习建议

✅ **每天一课** - 30分钟讲课 + 30分钟消化
✅ **动手实践** - 必须自己敲代码才能理解
✅ **改变参数** - 试试不同的数据规模、block大小
✅ **做笔记** - 记录每节课的核心收获

---

**准备好了吗？开始你的CUDA之旅！** 🎉

```bash
# 从第3课开始
nvcc -arch=sm_120 -o lesson03 0301-max-unified-naive.cu
./lesson03
```
