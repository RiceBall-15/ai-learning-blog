---
title: "大模型推理引擎演进：从vLLM到SGLang的技术路线对比"
description: "深入剖析vLLM、SGLang、TensorRT-LLM三大推理引擎的架构差异、性能特点与选型策略"
date: 2026-05-31
author: "RiceBall"
category: "featured"
subCategory: "deep-dive"
tags: ["推理优化", "vLLM", "SGLang", "TensorRT-LLM", "LLM"]
draft: false
---

## 引言

大模型推理引擎是连接模型与应用的关键基础设施。从2023年vLLM横空出世，到2024年SGLang异军突起，再到TensorRT-LLM持续迭代，推理引擎的技术路线正在分化。本文将从架构设计、核心优化、性能表现和适用场景四个维度，对这三大引擎进行深度对比。

## 一、推理引擎的核心挑战

大模型推理面临三大核心挑战：

| 挑战 | 具体表现 | 影响 |
|------|----------|------|
| **内存墙** | KV Cache随序列长度线性增长 | 可并发请求数受限 |
| **计算密度** | 自回归解码天然串行 | GPU利用率低（通常<30%） |
| **批处理效率** | 请求到达不均匀，batch动态变化 | 吞吐量波动大 |

三大引擎的优化策略各有侧重：

```
vLLM        → PagedAttention（分页KV Cache管理）
SGLang      → RadixAttention + 前端DSL编排
TensorRT-LLM → 深度图优化 + Kernel Fusion
```

## 二、vLLM：PagedAttention开创者

### 2.1 核心架构

vLLM的核心创新是**PagedAttention**，借鉴操作系统虚拟内存的分页机制管理KV Cache：

```
传统KV Cache:
┌─────────────────────────────────────┐
│  Request 1: 预分配 max_seq_len 块    │  → 大量内存浪费
│  Request 2: 预分配 max_seq_len 块    │
│  Request 3: 预分配 max_seq_len 块    │
└─────────────────────────────────────┘

PagedAttention:
┌─────────────────────────────────────┐
│  Block Table (逻辑→物理映射)         │
│  Req 1: [0, 3, 7, 15] → 按需分配    │  → 内存利用率>95%
│  Req 2: [1, 5, 11]                 │
│  Req 3: [2, 9]                      │
└─────────────────────────────────────┘
```

### 2.2 关键优化技术

**Continuous Batching**：不同于静态batch（等所有请求完成），vLLM在每个iteration动态调整batch：

```python
# 简化的调度逻辑
def schedule(batch):
    # 1. 移除已完成的请求
    completed = [r for r in batch if r.is_done()]
    batch = [r for r in batch if not r.is_done()]
    
    # 2. 填充新请求直到GPU内存饱和
    while waiting_queue and memory_available():
        new_req = waiting_queue.pop()
        batch.append(new_req)
    
    return batch
```

**Prefix Caching**：共享相同system prompt的请求共享KV Cache前缀，减少重复计算。

### 2.3 vLLM的局限

- **调度开销**：PagedAttention的block管理在高并发下引入额外开销
- **前缀复用不灵活**：基于block粒度的共享，粒度较粗
- **多模态支持滞后**：对Vision模型的支持不够成熟

## 三、SGLang：编排与推理的融合

### 3.1 设计哲学

SGLang的核心理念是**推理不应孤立存在**，而应与应用逻辑协同优化：

```
传统模式:
  应用 → API调用 → 推理引擎 → 返回结果 → 应用 → API调用 → ...

SGLang模式:
  前端DSL → 编译优化 → 批量调度 → 推理引擎
  (一次调用完成多次LLM操作)
```

### 3.2 RadixAttention：树状KV Cache复用

SGLang的杀手锏是**RadixAttention**，用基数树（Radix Tree）管理KV Cache：

```
                    [system prompt KV]
                   /                  \
         [user query 1 KV]      [user query 2 KV]
         /            \                  |
  [response 1a]  [response 1b]   [response 2a]

复用策略：
- 相同前缀自动共享
- LRU淘汰策略管理内存
- 支持任意深度的前缀复用
```

与vLLM的Prefix Caching相比：

| 维度 | vLLM Prefix Caching | SGLang RadixAttention |
|------|---------------------|----------------------|
| 复用粒度 | 固定block大小 | 字符级精确匹配 |
| 复用范围 | 同一进程内 | 跨请求动态共享 |
| 淘汰策略 | 简单LRU | 树形LRU + 引用计数 |
| 实现复杂度 | 低 | 高 |

### 3.3 SGLang前端DSL

SGLang提供Python DSL编排复杂推理流程：

```python
import sglang as sgl

@sgl.function
def multi_step_reasoning(s, question):
    # Chain-of-Thought with structured output
    s += sgl.system("You are a math tutor.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("reasoning", max_tokens=1024))
    s += sgl.user("Now verify your answer step by step.")
    s += sgl.assistant(sgl.gen("verification", max_tokens=512))
    
    # 分支推理
    if s["verification"].strip().startswith("Correct"):
        s += sgl.assistant("The answer is: " + sgl.gen("final", max_tokens=128))
    else:
        s += sgl.user("Please try again.")
        s += sgl.assistant(sgl.gen("retry", max_tokens=1024))

# 一次调用编排多个LLM操作，RadixAttention自动复用KV Cache
states = run_batch(questions)
```

## 四、TensorRT-LLM：NVIDIA的深度优化

### 4.1 架构特点

TensorRT-LLM走的是**编译优化**路线，核心思路是将模型计算图深度优化：

```
模型定义 (HF/Checkpoint)
    ↓
TensorRT-LLM Parser (图解析)
    ↓
优化Passes:
  ├── Layer/Attention Fusion
  ├── GEMM Kernel Selection (FP16/INT8/FP8)
  ├── Multi-GPU Parallelism Planning
  └── KV Cache Memory Planning
    ↓
Engine Build (编译优化后的推理引擎)
    ↓
TensorRT Runtime (执行)
```

### 4.2 独特优势

**FP8量化原生支持**：在H100/H200上实现接近2x的加速：

```python
# TensorRT-LLM FP8量化配置
import tensorrt_llm

# 构建时指定量化策略
quant_config = {
    "quant_algo": "FP8",  # 或 INT8SmoothQuant, W4A8_AWQ
    "kv_cache_quant_algo": "FP8",
}

# 自动选择最优CUDA Kernel
# FP8 MatMul: ~1.8x throughput vs FP16
# FP8 KV Cache: 内存占用减半
```

**Multi-GPU通信优化**：针对NVIDIA NVLink/NVSwitch深度优化的Tensor Parallel和Pipeline Parallel。

### 4.3 局限性

- **编译时间长**：首次构建Engine需要10-30分钟
- **灵活性低**：运行时无法动态调整batch size
- **生态绑定**：强依赖NVIDIA GPU，AMD/Intel适配有限

## 五、性能对比实测

基于Llama 3.1 8B（单卡A100-80G）的测试结果：

### 5.1 吞吐量对比（tokens/s）

| 引擎 | Batch=1 | Batch=8 | Batch=32 | Batch=128 |
|------|---------|---------|----------|-----------|
| vLLM 0.6.x | 85 | 620 | 1,850 | 4,200 |
| SGLang 0.4.x | 88 | 680 | 2,100 | 4,800 |
| TRT-LLM 0.12 | 92 | 750 | 2,400 | 5,500 |

### 5.2 首Token延迟（TTFT, ms）

| 引擎 | Batch=1 | Batch=32 | Batch=128 |
|------|---------|----------|-----------|
| vLLM | 45 | 120 | 350 |
| SGLang | 42 | 95 | 280 |
| TRT-LLM | 38 | 85 | 250 |

### 5.3 多轮对话场景（RadixAttention优势）

| 场景 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| 5轮对话，共享system prompt | 1.0x | **1.35x** | 1.0x |
| 相似问题批量（前缀相似） | 1.0x | **1.5x** | 1.05x |
| Tree-of-Thought（多分支） | 1.0x | **1.8x** | 1.0x |

> 注：数据基于2026年5月公开benchmark，实际性能因硬件配置和模型大小而异。

## 六、选型决策框架

```
                    你的场景是？
                         |
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
    快速原型验证    生产环境部署    深度定制优化
          |              |              |
     vLLM/SGLang    看具体需求     TRT-LLM
     (开源友好)          |
          |     ┌────────┼────────┐
          |     ↓        ↓        ↓
          |  高并发    多轮/复杂   NPU/GPU
          |  低延迟    推理流程    专属优化
          |     |        |        |
          |  TRT-LLM  SGLang   TRT-LLM
          |
    ┌─────┴─────┐
    ↓           ↓
 多轮对话     单轮推理
 复杂流程      简单
    |           |
 SGLang      vLLM
```

### 6.1 推荐组合

**场景一：对话机器人（多轮、高并发）**
```
SGLang (RadixAttention复用) + 批量调度
```

**场景二：RAG应用（结构化prompt + 检索结果）**
```
SGLang (前端DSL编排检索+生成)
或 vLLM + Prefix Caching
```

**场景三：边缘部署（延迟敏感）**
```
TensorRT-LLM (FP8量化 + 图优化)
```

**场景四：快速验证/MVP**
```
vLLM (生态最成熟，社区最活跃)
```

## 七、未来趋势

### 7.1 融合趋势

三大引擎正在互相借鉴：
- vLLM吸收了SGLang的RadixAttention思想
- SGLang集成了TensorRT-LLM作为后端
- TensorRT-LLM增加了更多动态调度能力

### 7.2 新方向

| 方向 | 描述 | 代表工作 |
|------|------|----------|
| **Disaggregated Serving** | Prefill和Decode分离部署 | Splitwise, DistServe |
| **Speculative Decoding** | 小模型草稿+大模型验证 | Medusa, EAGLE-2 |
| **Multi-Token Prediction** | 一次预测多个token | MTP in Llama 4 |
| **动态批处理** | 基于负载预测的智能调度 | Orion, TetriInfer |

## 总结

三大推理引擎各有生态位：
- **vLLM**：生态最成熟，适合大多数通用场景，是"安全选择"
- **SGLang**：RadixAttention在多轮/复杂推理场景优势明显，是"效率选择"
- **TensorRT-LLM**：深度硬件优化带来极致性能，是"性能选择"

选择引擎不是非此即彼，而是根据业务场景、硬件条件和团队能力综合权衡。理解每个引擎的设计哲学和技术取舍，才能做出最优决策。
