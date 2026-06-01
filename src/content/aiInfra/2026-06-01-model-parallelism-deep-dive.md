---
title: "模型并行深度解析：Tensor Parallel、Pipeline Parallel 与 Expert Parallel 的原理与实战"
description: "深入剖析三大模型并行策略的核心原理、通信模式、显存分布与工程实现，结合 Megatron-LM、DeepSpeed、vLLM 讲解大规模模型训练与推理的并行方案选型。"
date: 2026-06-01
author: "RiceBall"
category: "aiInfra"
subCategory: inference
tags: ["模型并行", "Tensor Parallel", "Pipeline Parallel", "Expert Parallel", "Megatron-LM", "DeepSpeed", "分布式训练"]
draft: false
---

# 模型并行深度解析：Tensor Parallel、Pipeline Parallel 与 Expert Parallel

## 引言：为什么需要模型并行？

当模型参数量从几十亿飙升到数百亿甚至万亿级别时，单卡显存早已无法容纳完整的模型参数、梯度和优化器状态。以一个 70B 参数的 LLaMA-3 模型为例：

| 精度 | 参数量 | 参数显存 | Adam 优化器 | 梯度 | 总显存需求 |
|------|--------|----------|-------------|------|------------|
| FP16 | 70B | 140 GB | 560 GB | 140 GB | ~840 GB |

即使使用最激进的混合精度训练（BF16 参数 + FP32 优化器状态），单张 80GB A100 也远远不够。模型并行的核心思想是**将模型的不同部分分布到多张 GPU 上协同计算**，从而突破单卡显存和算力的瓶颈。

但并行策略不是万能药——不同的并行方式带来不同的通信开销、显存分布和吞吐特性。选错策略，轻则浪费算力，重则训练崩溃。本文将深入拆解三大主流并行策略的核心原理与工程实现。

---

## 一、Tensor Parallel（张量并行）：把一层切到多卡上

### 1.1 核心思想

Tensor Parallel 的目标是**将单个算子（如线性层）的计算分摊到多张 GPU 上**。最经典的实现来自 Megatron-LM 对 Transformer 的分析。

对于一个线性层 $Y = XA$，其中 $A \in \mathbb{R}^{d \times h}$，我们可以沿不同维度切分矩阵 $A$：

```
┌─────────────────────────────┐
│  矩阵 A (d × h)             │
│                             │
│  ┌──────┬──────┬──────┐     │
│  │ GPU0 │ GPU1 │ GPU2 │     │  ← 沿输出维度 h 列切分
│  │A[:,0] │A[:,1] │A[:,2] │     │
│  └──────┴──────┴──────┘     │
│                             │
│  每张卡持有 A 的一部分列      │
│  并行计算 Y 的一部分列        │
└─────────────────────────────┘
```

### 1.2 Megatron-LM 的巧妙设计

Megatron-LM 利用 MLP 层中两个连续线性层的结构，设计了**无通信的张量并行**：

```
MLP层: Y = GeLU(XA)B

第一层 A: 列切分（Each GPU holds A_i）
  → Y_i = GeLU(X · A_i)  — 各卡独立计算，无需通信

第二层 B: 行切分（Each GPU holds B_i）
  → Z_i = Y_i · B_i  — 各卡独立计算

最后：所有 GPU 求和
  → Z = Σ Z_i  — 仅需一次 AllReduce
```

这种设计使得**一个 MLP 层只需要一次 AllReduce 通信**，极大减少了通信次数。

### 1.3 注意力层的并行

对于 Multi-Head Attention，天然适合按 head 切分：

```python
# 每张 GPU 负责一部分 head
num_heads = 32
tp_size = 8  # 张量并行度
heads_per_gpu = num_heads // tp_size  # 每卡 4 个 head

# 各卡独立计算自己的 head
# 最后需要一次 AllGather 收集所有 head 的输出
```

### 1.4 通信模式分析

| 操作 | 通信原语 | 通信量 | 发生频率 |
|------|----------|--------|----------|
| MLP 输出 | AllReduce | $O(b \cdot s \cdot d)$ | 每层 1 次 |
| Attention 输出 | AllGather + ReduceScatter | $O(b \cdot s \cdot d)$ | 每层 1 次 |
| LayerNorm | AllGather | $O(b \cdot s \cdot d)$ | 每层 1 次 |

**关键洞察**：Tensor Parallel 的通信量与 batch_size × seq_len × hidden_size 成正比，因此**大 batch 训练时通信占比更低**，效率更高。

### 1.5 适用场景

- ✅ 节点内高带宽互联（NVLink/NVSwitch）的多卡训练
- ✅ 单层参数量特别大的情况
- ❌ 跨节点训练（网络带宽不足以支撑频繁的 AllReduce）
- ❌ 推理场景（batch_size 小，通信开销占比高）

---

## 二、Pipeline Parallel（流水线并行）：把层切到多卡上

### 2.1 核心思想

Pipeline Parallel 将模型按**层**切分，每张 GPU 持有模型的一部分连续层。数据依次流过各 GPU，形成流水线：

```
GPU 0: Layers [0, 1, 2, 3]    → 中间激活传给 GPU 1
GPU 1: Layers [4, 5, 6, 7]    → 中间激活传给 GPU 2
GPU 2: Layers [8, 9, 10, 11]  → 中间激活传给 GPU 3
GPU 3: Layers [12, 13, 14, 15]
```

### 2.2 GPipe：朴素流水线的问题

最简单的 Pipeline Parallel 是 GPipe——将一个 mini-batch 切成多个 micro-batch，依次送入流水线：

```
时间 →
GPU 0: [m1] [m2] [m3] [m4] [  空闲  ] [  空闲  ] [  空闲  ]
GPU 1: [ 空 ] [m1] [m2] [m3] [m4]    [  空闲  ] [  空闲  ]
GPU 2: [ 空 ] [ 空 ] [m1] [m2] [m3]   [m4]    [  空闲  ]
GPU 3: [ 空 ] [ 空 ] [ 空 ] [m1] [m2]   [m3]   [m4]
```

**问题**：大量时间 GPU 处于空闲状态（称为 **bubble**），流水线利用率低。

### 2.3 1F1B 调度：减少显存峰值

1F1B（One Forward One Backward）通过交错执行前向和反向传播，**减少了需要同时保存的激活值数量**：

```
时间 →
GPU 0: F1 F2 F3 F4 B1 B2 B3 B4
GPU 1:    F1 F2 F3 F4 B1 B2 B3 B4
GPU 2:       F1 F2 F3 F4 B1 B2 B3 B4
GPU 3:          F1 F2 F3 F4 B1 B2 B3 B4
```

1F1B 将显存占用从 O(m)（m 为 micro-batch 数量）降低到 O(p)（p 为流水线 stage 数量），这对于大模型训练至关重要。

### 2.4 Interleaved Pipeline：进一步减少 bubble

Megatron-LM 引入了**交错流水线**（Interleaved Pipeline），让每张 GPU 持有**多个非连续的 stage**：

```
传统流水线 (p=4, g=4):
GPU 0: [Stage 0]
GPU 1: [Stage 1]
GPU 2: [Stage 2]
GPU 3: [Stage 3]
Bubble 比例: (p-1) / (p-1+m)

交错流水线 (v=4, g=4):  
GPU 0: [Stage 0] [Stage 4]
GPU 1: [Stage 1] [Stage 5]
GPU 2: [Stage 2] [Stage 6]
GPU 3: [Stage 3] [Stage 7]
Bubble 比例: (p-1) / (p-1+v*m)
```

v 倍的交错使 bubble 减少约 v 倍，但代价是**更多的通信次数**（跨 stage 的激活值传输）。

### 2.5 通信模式分析

| 调度策略 | 通信原语 | 通信次数 | 显存占用 |
|----------|----------|----------|----------|
| GPipe | P2P Send/Recv | O(m × p) | O(m) |
| 1F1B | P2P Send/Recv | O(m × p) | O(p) |
| Interleaved | P2P Send/Recv | O(m × p × v) | O(p) |

**关键洞察**：Pipeline Parallel 的通信只发生在**相邻 stage 之间**，通信量为 $O(b \cdot s \cdot d)$（单个 micro-batch 的激活值），因此**跨节点训练时通信开销可控**。

### 2.6 适用场景

- ✅ 跨节点训练（通信只在相邻节点间）
- ✅ 模型层数特别多的情况
- ✅ 想要最大化显存利用率
- ❌ 需要极低延迟的推理场景

---

## 三、Expert Parallel（专家并行）：MoE 模型的专属并行

### 3.1 MoE 模型的特殊性

Mixture of Experts (MoE) 模型将 MLP 层替换为多个"专家"网络，通过门控网络（Router）动态选择每个 token 激活哪些专家：

```
普通 Transformer MLP:
  Y = MLP(X)  ← 每个 token 经过同一个 MLP

MoE Transformer MLP:
  Y = Σ_i (G(X)_i · Expert_i(X))  ← 每个 token 仅激活 Top-K 个专家
```

以 Mixtral 8x7B 为例：共有 8 个专家，每个 token 仅激活 2 个，总参数量约 46.7B，但计算量仅约 12.9B。

### 3.2 Expert Parallel 的核心思想

Expert Parallel 将**不同的专家分布到不同的 GPU 上**：

```
┌────────────────────────────────────┐
│  8 个 Expert, EP=4 (每卡 2 个专家)    │
│                                    │
│  GPU 0: Expert 0, Expert 1         │
│  GPU 1: Expert 2, Expert 3         │
│  GPU 2: Expert 4, Expert 5         │
│  GPU 3: Expert 6, Expert 7         │
│                                    │
│  Router 决定每个 token 去哪个专家    │
│  需要 All-to-All 通信分发 token     │
└────────────────────────────────────┘
```

### 3.3 All-to-All 通信

Expert Parallel 的核心通信模式是 **All-to-All**：每张 GPU 将自己负责的 token 发送到对应专家所在的 GPU，计算完成后再收回结果。

```
发送阶段（Dispatch）:
  GPU 0 → GPU 1: tokens[Expert 2, Expert 3]
  GPU 0 → GPU 2: tokens[Expert 4, Expert 5]
  GPU 0 → GPU 3: tokens[Expert 6, Expert 7]
  ...（所有 GPU 同时执行）

计算阶段:
  各 GPU 独立计算本地专家

收集阶段（Combine）:
  All-to-All 的逆操作
```

### 3.4 负载均衡的挑战

MoE 模型训练中最大的工程难题之一是**负载均衡**——如果大部分 token 被路由到少数几个专家，会导致：

1. **计算不均衡**：部分 GPU 繁忙，部分 GPU 空闲
2. **通信不均衡**：部分 GPU 发送大量 token，部分 GPU 几乎无通信
3. **训练不稳定**：热门专家的梯度更新过多，冷门专家更新不足

DeepSeek-V3 引入了 **Auxiliary Loss** 和 **Bias-based Load Balancing**：

```python
# 辅助损失：鼓励均匀分配
aux_loss = α * N * Σ_i (f_i * P_i)

# 其中:
# f_i = 分配给专家 i 的 token 比例
# P_i = 所有 token 对专家 i 的路由概率均值
# N = 专家数量
# α = 平衡系数（通常 0.01）
```

### 3.5 通信模式分析

| 操作 | 通信原语 | 通信量 | 特点 |
|------|----------|--------|------|
| Dispatch | All-to-All | $O(b \cdot s \cdot d)$ | 路由决定通信模式 |
| Combine | All-to-All | $O(b \cdot s \cdot d)$ | 与 Dispatch 对称 |
| Expert 梯度同步 | AllReduce | $O(b \cdot s \cdot d)$ | 各专家独立同步 |

**关键洞察**：Expert Parallel 的 All-to-All 通信模式要求**所有 GPU 之间的互联带宽均匀**，因此**特别适合节点内多卡训练**（NVSwitch 提供全互联）。

### 3.6 适用场景

- ✅ MoE 模型（Mixtral、DeepSeek-V3、Qwen-MoE 等）
- ✅ 节点内高带宽互联
- ✅ 需要大参数量但限制计算量的场景
- ❌ Dense 模型（没有专家概念）

---

## 四、三种并行策略的组合与选型

### 4.1 3D 并行：工业级训练的标准方案

实际的大规模模型训练通常采用**多种并行策略的组合**（3D Parallel）：

```
┌─────────────────────────────────────────────────┐
│                  3D Parallel                      │
│                                                   │
│  ┌───────────────────────────────────────────┐   │
│  │         Pipeline Parallel (跨节点)         │   │
│  │   Node 0: Layers [0-7]                    │   │
│  │   Node 1: Layers [8-15]                   │   │
│  │   Node 2: Layers [16-23]                  │   │
│  │   Node 3: Layers [24-31]                  │   │
│  │                                           │   │
│  │   ┌─────────────────────────────────┐     │   │
│  │   │     Tensor Parallel (节点内)     │     │   │
│  │   │  GPU 0-3: TP=4                  │     │   │
│  │   │  GPU 4-7: TP=4                  │     │   │
│  │   └─────────────────────────────────┘     │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  ┌───────────────────────────────────────────┐   │
│  │         Data Parallel (全局)               │   │
│  │  每个 DP group 独立处理不同数据              │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 4.2 LLaMA-70B 训练配置示例

以 Meta 训练 LLaMA-70B 为例，典型的并行配置：

```yaml
# LLaMA-70B 训练并行配置
model:
  parameters: 70B
  layers: 80
  hidden_size: 8192
  num_heads: 64

parallel:
  tp: 8      # 张量并行度 (节点内 8 卡 NVSwitch)
  pp: 4      # 流水线并行度 (4 个节点)
  dp: 2      # 数据并行度 (2 个 DP group)
  # 总 GPU 数: tp × pp × dp = 8 × 4 × 2 = 64 卡

hardware:
  gpus_per_node: 8
  gpu_type: H100 80GB
  inter_node_bandwidth: 400 Gbps  # InfiniBand
  intra_node_bandwidth: 900 GB/s  # NVSwitch
```

### 4.3 选型决策树

```
你的模型是 MoE 还是 Dense？
├── MoE → Expert Parallel 是必须的
│   ├── 配合 Tensor Parallel 处理单层内计算
│   └── 配合 Pipeline Parallel 处理跨层分布
│
└── Dense → 选择 TP + PP 组合
    ├── 节点内互联带宽 > 500 GB/s？
    │   ├── 是 → 优先 Tensor Parallel（通信效率高）
    │   └── 否 → 优先 Pipeline Parallel（通信量小）
    │
    ├── 模型层数 > 100？
    │   └── Pipeline Parallel（天然按层切分）
    │
    └── 单层参数量特别大？
        └── Tensor Parallel（把大矩阵切到多卡）
```

---

## 五、工程实战：DeepSpeed 与 Megatron-LM 的并行实现

### 5.1 DeepSpeed 的 ZeRO + 并行混合

DeepSpeed 的 ZeRO（Zero Redundancy Optimizer）本质上是一种**数据并行的显存优化**，它与模型并行可以叠加使用：

```python
# DeepSpeed 配置示例
ds_config = {
    "zero_optimization": {
        "stage": 3,  # ZeRO-3: 参数、梯度、优化器状态全分片
        "overlap_comm": True,  # 通信计算重叠
        "contiguous_gradients": True,
    },
    "bf16": {"enabled": True},
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
}

# ZeRO-3 + TP + PP 的组合
# ZeRO-3 负责分片优化器状态（数据并行维度）
# TP 负责单层内的计算分片
# PP 负责跨层的计算分片
```

### 5.2 Megatron-LM 的并行配置

```python
# Megatron-LM 并行参数
parallel_args = {
    "tensor_model_parallel_size": 8,   # TP=8
    "pipeline_model_parallel_size": 4,  # PP=4
    "num_layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "seq_length": 4096,
    "micro_batch_size": 1,
    "global_batch_size": 1024,
    "sequence_parallel": True,  # 序列并行（TP 的扩展）
}
```

### 5.3 vLLM 推理的并行策略

对于推理场景，vLLM 的并行策略与训练有所不同：

```python
from vllm import LLM, SamplingParams

# 张量并行推理
llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=8,  # TP=8
    pipeline_parallel_size=1,  # 推理通常不用 PP（延迟敏感）
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# 推理场景的关键差异:
# 1. 不需要梯度和优化器状态，显存需求大幅降低
# 2. Prefill 阶段 compute-bound，Decode 阶段 memory-bound
# 3. TP 适合 Prefill，但 Decode 阶段通信开销占比高
```

---

## 六、性能对比与实战经验

### 6.1 通信开销对比

以 70B 模型、64 卡 H100 为例：

| 并行策略 | 通信模式 | 通信量 (per step) | 适合互联 |
|----------|----------|-------------------|----------|
| TP=8, PP=4 | AllReduce (节点内) + P2P (跨节点) | ~200 GB | NVLink + IB |
| TP=4, PP=8 | AllReduce (节点内) + P2P (跨节点) | ~150 GB | NVLink + IB |
| TP=8, PP=1 | AllReduce (节点内) | ~250 GB | NVLink |
| PP=4, DP=2 | P2P (跨节点) + AllReduce (梯度) | ~100 GB | IB |

### 6.2 显存分布对比

| 并行策略 | 参数显存 (per GPU) | 优化器显存 | 激活显存 |
|----------|-------------------|------------|----------|
| DP=64 | 140 GB ❌ | 560 GB ❌ | ~20 GB |
| TP=8, PP=4 | 17.5 GB ✅ | 70 GB | ~15 GB |
| TP=8, PP=4 + ZeRO-3 | 17.5 GB | 8.75 GB | ~15 GB |

### 6.3 常见踩坑点

**坑 1：TP 配置与硬件不匹配**
```python
# 错误：跨节点配置 TP=8
tp = 8  # 但节点间只有 IB 网络
# 后果：AllReduce 通信成为严重瓶颈，训练速度下降 5-10 倍

# 正确：TP 只在节点内
tp = 8  # 节点内 8 卡 NVSwitch
pp = 4  # 跨节点用 Pipeline Parallel
```

**坑 2：PP 的 micro-batch 太小**
```python
# 错误：micro_batch_size 太小
micro_batch_size = 1  # 导致 bubble 占比过高

# 正确：增大 micro_batch_size 或使用 Interleaved PP
micro_batch_size = 4  # 或使用 v=2 的交错流水线
```

**坑 3：MoE 的 Expert Parallel 与 TP 叠加**
```python
# 错误：同时使用 EP 和 TP，通信模式冲突
ep = 8  # Expert Parallel
tp = 4  # Tensor Parallel
# 后果：All-to-All + AllReduce 的通信冲突导致死锁

# 正确：MoE 模型中 EP 通常替代 TP
ep = 8  # Expert Parallel
tp = 1  # 不使用 TP（专家本身已经分片了）
```

---

## 七、总结：并行策略的本质

| 维度 | Tensor Parallel | Pipeline Parallel | Expert Parallel |
|------|-----------------|-------------------|-----------------|
| 切分粒度 | 矩阵（行/列） | 层 | 专家 |
| 通信模式 | AllReduce / AllGather | P2P Send/Recv | All-to-All |
| 通信频率 | 每层 1-2 次 | 每个 micro-batch 1 次 | 每个 MoE 层 1 次 |
| 通信带宽需求 | 极高（节点内） | 中等（跨节点） | 高（全互联） |
| 适用场景 | 节点内并行 | 跨节点并行 | MoE 模型 |
| 对 batch size 的敏感度 | 大 batch 更高效 | 影响较小 | 中等 |

**核心原则**：

1. **通信带宽决定并行策略**：高带宽用 TP，低带宽用 PP
2. **模型结构决定切分方式**：Dense 用 TP+PP，MoE 用 EP
3. **显存需求决定并行组合**：显存不足用 ZeRO + 模型并行叠加
4. **延迟需求决定推理策略**：推理优先 TP（Prefill），Decode 阶段考虑量化

模型并行不是银弹，而是一个需要根据硬件、模型结构、训练/推理需求综合权衡的工程决策。理解每种策略的通信模式和显存分布，才能在实际项目中做出最优选择。

---

> **参考资料**：
> - [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
> - [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
> - [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://arxiv.org/abs/2010.02043)
> - [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
> - [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
