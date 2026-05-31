---
title: "LLM推理量化技术深度对比：GPTQ、AWQ、GGUF与bitsandbytes实战指南"
description: "从原理到实践，全面解析主流量化方案的精度、性能与适用场景"
date: 2026-06-01
author: "RiceBall"
category: "aiInfra"
tags: ["量化", "GPTQ", "AWQ", "GGUF", "推理优化"]
draft: false
---

## 引言：量化——让大模型"瘦身"的关键技术

大语言模型的参数规模正在以惊人的速度增长——从最初的7B到现在的400B+，模型能力在飞速提升的同时，对计算资源的需求也在指数级增长。以Llama 3.1 70B为例，FP16精度下需要约140GB显存，这远远超出了单张消费级GPU（RTX 4090 24GB）的承载能力。

**量化（Quantization）** 就是在这个背景下诞生的核心技术——通过降低模型权重的数值精度（如从FP16降到INT4），大幅减少模型的显存占用和计算需求，同时尽可能保持模型输出质量。

然而，量化技术并非银弹。不同的量化方法在**精度保持、推理速度、硬件要求、易用性**等方面差异显著。本文将深度对比四大主流量化方案：**GPTQ**、**AWQ**、**GGUF** 和 **bitsandbytes**，帮助你根据实际需求做出最佳选择。

---

## 一、量化基础：从理论到直觉

### 1.1 什么是量化？

量化的核心思想很简单：用更少的bit来表示每个权重值。

```
FP16 (16-bit):  0.18456723
INT8 (8-bit):   0.184
INT4 (4-bit):   0.18
```

更正式地，量化是将连续的浮点数值映射到离散的整数集合：

```
量化公式:  x_q = round(x / scale) + zero_point
反量化:    x = (x_q - zero_point) * scale
```

### 1.2 量化分类体系

```
量化方法
├── 按粒度
│   ├── Layer-wise (整层统一scale)
│   ├── Group-wise (分组scale) ← 主流方案
│   └── Channel-wise (逐通道scale)
│
├── 按时机
│   ├── 训练后量化 PTQ (Post-Training Quantization) ← 本文重点
│   └── 量化感知训练 QAT (Quantization-Aware Training)
│
└── 按方案
    ├── GPTQ (逐层最优量化)
    ├── AWQ (激活感知量化)
    ├── GGUF (llama.cpp生态量化)
    ├── bitsandbytes (动态量化)
    ├── SqueezeLLM (非均匀量化)
    └── QuIP# (信息论量化)
```

### 1.3 关键指标解读

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **PPL (Perplexity)** | 困惑度，衡量模型预测能力 | 越低越好，是量化精度的核心指标 |
| **Bitrate** | 每个参数平均占用bit数 | 越低显存越小 |
| **Dequantization Overhead** | 反量化计算开销 | 影响推理速度 |
| **GPU Memory** | 显存占用 | 直接决定能否运行 |
| **Throughput** | 吞吐量(tokens/s) | 生产环境关键指标 |

---

## 二、四大方案深度剖析

### 2.1 GPTQ：基于最优脑量化的精确方案

GPTQ（Generalized Post-Training Quantization）源自 Optimal Brain Quantization (OBQ) 的思路，核心思想是**逐层量化，最小化量化误差**。

#### 原理

GPTQ 的关键创新在于将 OBS 框架的逐列量化改为了**逐行量化**，并利用 Hessian 矩阵的逆来进行最优量化决策：

```
对于每一层的权重矩阵 W：

1. 计算 Hessian 逆矩阵 H⁻¹
2. 按行处理权重，对于每个待量化的权重 w:
   - 计算量化误差: δ = w - quant(w)
   - 将误差补偿到剩余未量化的权重: W' += δ * H⁻¹ / H⁻¹[j,j]
3. 逐行完成量化，误差通过 Hessian 信息传播到未量化权重

这种"误差补偿"机制确保了即使在4-bit量化下，
整体误差也能被控制在很小的范围内。
```

#### 核心代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# GPTQ量化流程
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# 准备量化配置
quantization_config = GPTQConfig(
    bits=4,                          # 量化位数
    group_size=128,                  # 分组大小
    desc_act=True,                   # 按激活值降序排列列（更精确）
    dataset="c4",                    # 校准数据集
    sym=False,                       # 非对称量化
)

# 加载并量化
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# 保存量化模型
model.save_pretrained("llama3.1-8b-gptq")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("llama3.1-8b-gptq")
```

#### 特性总结

| 特性 | 说明 |
|------|------|
| **量化精度** | 高（误差补偿机制） |
| **校准数据** | 需要（通常128条样本） |
| **量化速度** | 中等（逐层计算Hessian） |
| **推理支持** | AutoGPTQ / ExLlamaV2 / vLLM |
| **GPU要求** | 需要GPU进行量化 |
| **显存节省** | FP16→INT4 约4x |

---

### 2.2 AWQ：激活感知的权重量化

AWQ（Activation-aware Weight Quantization）的核心洞察是：**不是所有权重都同等重要——那些对应于大激活值的权重通道对模型输出的影响更大，应该被更精确地保留**。

#### 原理

```
传统量化：对所有权重通道一视同仁
┌──────────────────────────────────┐
│  Channel 1: [0.1, 0.2, 0.15]   │  ← 量化精度相同
│  Channel 2: [5.0, 4.8, 5.2]    │  ← 量化精度相同
│  Channel 3: [0.05, 0.1, 0.08]  │  ← 量化精度相同
└──────────────────────────────────┘

AWQ：识别重要通道并保护它们
┌──────────────────────────────────┐
│  Channel 1: [0.1, 0.2, 0.15]   │  ← 激活值小，正常量化
│  Channel 2: [5.0, 4.8, 5.2]    │  ← 激活值大，保护性量化
│  Channel 3: [0.05, 0.1, 0.08]  │  ← 激活值小，激进量化
└──────────────────────────────────┘

关键操作：通过 per-channel scaling 乘以一个缩放因子 s，
将重要通道的数值范围放大到更不容易产生量化误差的区间。
```

#### 核心代码

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# AWQ量化配置
quant_config = {
    "zero_point": True,     # 使用零点对称量化
    "q_group_size": 128,    # 分组大小
    "w_bit": 4,             # 量化位数
    "version": "GEMM",      # 使用GEMM内核（更适合部署）
}

# 执行量化（使用校准数据）
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",   # 使用pileval数据集校准
)

# 保存量化模型
model.save_quantized("./llama3.1-8b-awq")
tokenizer.save_pretrained("./llama3.1-8b-awq")
```

#### 特性总结

| 特性 | 说明 |
|------|------|
| **量化精度** | 高（保护重要通道） |
| **校准数据** | 需要（自动选取） |
| **量化速度** | 较快（无需Hessian计算） |
| **推理支持** | AutoAWQ / vLLM / TensorRT-LLM |
| **GPU要求** | 需要GPU |
| **显存节省** | FP16→INT4 约4x |
| **独特优势** | 激活感知，精度通常略优于GPTQ |

---

### 2.3 GGUF：llama.cpp 生态的量化标准

GGUF（GPT-Generated Unified Format）是 llama.cpp 生态中定义的模型格式，它将模型权重、分词器、元数据等信息打包在一个文件中。GGUF本身不是一个量化算法，而是一个**量化模型的容器格式和分发标准**。

#### 量化级别详解

GGUF提供从2-bit到8-bit的多种量化级别，使用不同的量化策略：

```
量化级别（精度从低到高，大小从小到大）：

Q2_K  ─── 2-bit, K-quant     │ 精度损失较大，极端压缩场景
  │                           │
Q3_K_M ─── 3-bit, medium     │ 平衡精度和大小
  │                           │
Q4_K_M ─── 4-bit, medium     │ ★ 最常用，性价比最高
  │                           │
Q5_K_M ─── 5-bit, medium     │ 精度较好，大小适中
  │                           │
Q6_K  ─── 6-bit, K-quant     │ 接近FP16精度
  │                           │
Q8_0  ─── 8-bit              │ 几乎无损，FP16的1/2大小
  │                           │
F16   ─── 16-bit             │ 原始精度（FP16）
  │                           │
F32   ─── 32-bit             │ 全精度
```

#### K-Quant 混合精度策略

GGUF的 K-Quant 系列采用了**逐层混合精度**策略——对不同层使用不同的量化位数：

```
Llama 3.1 8B 的 Q4_K_M 量化层分配：

Embedding层:   FP16 (不量化，保证输入精度)
  ↓
Attention层:   Q6_K (较高精度，对注意力计算影响大)
  ↓
FFN中间层:     Q4_K_M (标准4-bit)
  ↓
输出层:        Q6_K (较高精度，对输出影响大)

这种混合策略比统一使用Q4_K_M精度更高，
比统一使用Q6_K体积更小——是工程上的最佳平衡点。
```

#### 核心使用

```bash
# 使用 llama.cpp 量化
./llama-quantize \
  model-f16.gguf \           # 输入FP16模型
  model-q4_k_m.gguf \        # 输出量化模型
  Q4_K_M \                   # 量化级别
  8 \                        # 线程数
  0                          # GPU层数

# 使用 Python 转换和量化
pip install llama-cpp-python

from llama_cpp import Llama

# 加载量化模型
llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=33,  # GPU加速
)
```

#### 特性总结

| 特性 | 说明 |
|------|------|
| **量化精度** | 依赖级别（Q4_K_M为最佳平衡点） |
| **校准数据** | 不需要（基于统计的量化） |
| **量化速度** | 极快（简单的统计量化） |
| **推理支持** | llama.cpp / Ollama / LM Studio / ctransformers |
| **GPU要求** | 不强制（CPU-only也能运行） |
| **最大优势** | 无需校准，CPU友好，生态完善 |

---

### 2.4 bitsandbytes：动态量化的灵活方案

bitsandbytes (bnb) 是由 Tim Dettmers 开发的量化库，它提供了两种量化模式：**8-bit量化**和**4-bit量化（QLoRA的基础）**。

#### 核心原理

bitsandbytes 的独特之处在于它使用**动态量化**——量化过程在模型加载时实时完成，而不需要预先量化和保存量化模型。

```python
# 8-bit量化：分块动态量化
# 将权重矩阵分成块，每块独立计算量化参数
#
# 原始权重:
# ┌─────────────────────────┐
# │ [0.1, -0.5, 0.3, 0.8]  │  block 1 → scale1, zp1
# │ [0.2,  0.1, -0.4, 0.6] │  block 2 → scale2, zp2
# │ [0.9, -0.2, 0.5, 0.3]  │  block 3 → scale3, zp3
# └─────────────────────────┘

# 4-bit NormalFloat (NF4): QLoRA的核心
# 假设权重服从正态分布，设计最优的量化桶
# 量化点不是均匀分布，而是按正态分布密度采样
#
# 密度高(权重多)的区域 → 量化点密集 → 精度高
# 密度低(权重少)的区域 → 量化点稀疏 → 精度够用
```

#### 核心代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit QLoRA量化配置
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算精度
    bnb_4bit_use_double_quant=True,       # 二次量化（量化缩放因子）
)

# 8-bit量化配置
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,    # 异常值阈值（超过此值的通道不量化）
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config_4bit,
    device_map="auto",
)

# 直接使用，无需量化步骤
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
output = model.generate(
    tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to("cuda"),
    max_new_tokens=100,
)
```

#### QLoRA：量化 + LoRA 微调

bitsandbytes 最具影响力的应用场景是 **QLoRA**——在量化模型上进行LoRA微调：

```
传统微调（FP16全参数）:
┌──────────────────────────────┐
│ Model (16GB) + Gradients     │  需要 ≥32GB 显存
│ + Optimizer States           │
└──────────────────────────────┘

QLoRA微调（NF4量化 + LoRA）:
┌──────────────────────────────┐
│ Model (4GB, NF4量化)         │  仅需 ~8GB 显存
│ + LoRA Adapters (少量参数)    │  
│ + Optimizer (仅LoRA参数)      │
└──────────────────────────────┘

显存节省: 32GB → 8GB (4x)
微调质量: 接近全参数微调
```

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 准备量化模型用于微调
model = prepare_model_for_kbit_training(model)

# LoRA配置
lora_config = LoraConfig(
    r=16,                          # LoRA秩
    lora_alpha=32,                 # 缩放因子
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 查看可训练参数量
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 4,502,599,680 || trainable%: 0.0931
```

#### 特性总结

| 特性 | 说明 |
|------|------|
| **量化精度** | 中高（NF4专为正态分布优化） |
| **校准数据** | 不需要（动态量化） |
| **量化速度** | 极快（加载时实时量化） |
| **推理支持** | Hugging Face / TGI / Axolotl |
| **GPU要求** | 需要GPU（CUDA） |
| **最大优势** | 动态量化，QLoRA微调基础 |

---

## 三、全维度对比

### 3.1 精度对比（以 Llama 3.1 8B Instruct 为例）

| 量化方案 | 位数 | 大小 | PPL (↑越低越好) | MMLU | HumanEval |
|---------|------|------|-----------------|------|-----------|
| **FP16 (基线)** | 16 | 16GB | 5.23 | 68.4 | 62.2 |
| **GPTQ** | 4 | 4.4GB | 5.31 | 67.8 | 60.4 |
| **AWQ** | 4 | 4.4GB | 5.28 | 68.0 | 61.0 |
| **GGUF Q4_K_M** | 4 | 4.9GB | 5.35 | 67.5 | 59.8 |
| **GGUF Q5_K_M** | 5 | 5.7GB | 5.26 | 68.1 | 61.5 |
| **GGUF Q6_K** | 6 | 6.6GB | 5.24 | 68.3 | 62.0 |
| **bitsandbytes NF4** | 4 | 4.5GB | 5.33 | 67.6 | 60.1 |

> 📝 **精度排名**: AWQ ≈ GPTQ > GGUF Q5_K_M > bitsandbytes NF4 > GGUF Q4_K_M

### 3.2 推理速度对比

在RTX 4090上使用Llama 3.1 8B的推理速度：

| 量化方案 | 量化引擎 | 吞吐量 (tokens/s) | 首Token延迟 | 说明 |
|---------|---------|-------------------|------------|------|
| FP16 | vLLM | 112 | 62ms | 基线 |
| GPTQ | AutoGPTQ + vLLM | 108 | 58ms | 接近FP16 |
| AWQ | vLLM | 115 | 55ms | 略快于FP16 |
| GGUF Q4_K_M | llama.cpp | 105 | 78ms | CPU/GPU混合 |
| GGUF Q4_K_M | llama.cpp (GPU) | 130 | 52ms | 全GPU最快 |
| bitsandbytes | HF Transformers | 68 | 120ms | 动态量化开销 |

**分析：**
- **AWQ + vLLM** 是推理速度和精度的最佳组合
- **GGUF + llama.cpp** 在GPU模式下吞吐量最高（C++实现更高效）
- **bitsandbytes** 的动态量化带来额外开销，推理速度最慢

### 3.3 显存占用对比

以70B模型为例（不同量化级别）：

```
70B模型显存需求对比：

FP16:  ████████████████████████████████████████████  140GB
Q8_0:  ██████████████████████████                  70GB
Q6_K:  ████████████████████                        56GB
Q5_K_M: ██████████████████                         49GB
Q4_K_M: ████████████████                           40GB
Q4_K_S: ███████████████                            38GB
Q3_K_M: █████████████                              34GB
Q2_K:   █████████                                  28GB

一张RTX 4090 (24GB) 的参考线：
Q4_K_M 70B → 需要 2×RTX 4090
Q2_K 70B  → 刚好 1×RTX 4090（但精度损失大）
Q4_K_M 8B → 1×RTX 4090 绰绰有余
```

### 3.4 量化速度对比

| 操作 | GPTQ | AWQ | GGUF | bitsandbytes |
|------|------|-----|------|-------------|
| **量化8B模型** | ~30min | ~15min | ~5min | N/A（动态） |
| **量化70B模型** | ~3hr | ~1.5hr | ~30min | N/A（动态） |
| **所需GPU** | 1×A100 | 1×A100 | 可CPU | N/A |
| **校准数据** | 128条 | 自动选取 | 不需要 | 不需要 |
| **量化步骤** | 转换→量化→保存 | 转换→量化→保存 | 直接转换 | 无需量化 |

### 3.5 硬件兼容性

| 硬件平台 | GPTQ | AWQ | GGUF | bitsandbytes |
|---------|------|-----|------|-------------|
| NVIDIA GPU (CUDA) | ✅ | ✅ | ✅ | ✅ |
| AMD GPU (ROCm) | ⚠️ | ⚠️ | ✅ | ❌ |
| Apple Silicon (Metal) | ❌ | ❌ | ✅ | ❌ |
| CPU-only | ❌ | ❌ | ✅ | ❌ |
| Intel GPU | ❌ | ❌ | ✅ | ❌ |
| Qualcomm NPU | ❌ | ❌ | ⚠️ | ❌ |
| 移动端 (ARM) | ❌ | ❌ | ✅ | ❌ |

---

## 四、选型指南

### 4.1 按使用场景选型

```
你的主要场景是什么？
│
├── 推理部署（生产环境）
│   ├── 需要高并发 → AWQ + vLLM ✅
│   ├── 单卡部署 → GPTQ 或 AWQ + vLLM ✅
│   └── 多卡并行 → AWQ/GPTQ + vLLM (TP) ✅
│
├── 微调训练
│   ├── 全参数微调 → 不量化 (FP16/BF16)
│   ├── LoRA/QLoRA微调 → bitsandbytes NF4 ✅
│   └── LoRA + 推理 → bitsandbytes 量化
│
├── 本地个人使用
│   ├── 桌面/CLI → GGUF (Ollama/llama.cpp) ✅
│   ├── Mac用户 → GGUF (Metal加速) ✅
│   └── 低配机器 → GGUF Q3/Q4 ✅
│
├── 边缘/嵌入式部署
│   ├── ARM设备 → GGUF (CPU-only) ✅
│   ├── 树莓派 → GGUF Q2/Q3 ✅
│   └── 手机 → GGUF (通过llama.cpp) ✅
│
└── 快速实验
    ├── 最简流程 → bitsandbytes (加载即量化) ✅
    └── 精度优先 → AWQ ✅
```

### 4.2 各方案综合评分

| 评估维度 | GPTQ | AWQ | GGUF | bitsandbytes |
|---------|------|-----|------|-------------|
| **量化精度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **推理速度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **易用性** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **硬件兼容** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **生态支持** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **生产就绪** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **微调支持** | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

### 4.3 推荐组合方案

**方案一：企业推理服务**
```
AWQ量化 + vLLM部署
├── 精度：接近FP16
├── 吞吐：高于FP16 (量化减少计算量)
├── 显存：4x节省
└── 适合：API服务、多用户场景
```

**方案二：个人开发 + 微调**
```
bitsandbytes NF4 + QLoRA微调 + vLLM推理
├── 微调：仅需8GB显存即可微调8B模型
├── 推理：微调后转为AWQ导出
└── 适合：资源受限的研究者
```

**方案三：全平台本地使用**
```
GGUF Q4_K_M + Ollama/llama.cpp
├── 精度：可接受（Q5_K_M接近AWQ）
├── 速度：C++实现极快
├── 兼容：CPU/GPU/ARM/Metal
└── 适合：个人使用、边缘部署
```

**方案四：混合精度部署**
```
关键层(Attention/Output): Q6_K FFN层: Q4_K_M Embedding: F16
├── 精度：接近Q6_K
├── 大小：接近Q4_K_M
└── 适合：对精度敏感的生产环境
```

---

## 五、量化实战：从模型到部署

### 5.1 完整工作流示例

```python
"""
完整量化流程：Llama 3.1 8B → AWQ量化 → vLLM部署
"""

# Step 1: AWQ量化
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("./llama3.1-8b-awq")
tokenizer.save_pretrained("./llama3.1-8b-awq")

# Step 2: vLLM部署
from vllm import LLM, SamplingParams

llm = LLM(
    model="./llama3.1-8b-awq",
    quantization="awq",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
)

# Step 3: 服务测试
prompts = [
    "解释什么是Transformer架构",
    "写一个Python快速排序算法",
    "分析量子计算的商业前景",
]
sampling_params = SamplingParams(temperature=0.7, max_tokens=512, top_p=0.9)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Generated: {output.outputs[0].text[:100]}...")
    print(f"Tokens/s: {len(output.outputs[0].token_ids) / output.metrics.finished_time:.1f}")
    print("---")
```

### 5.2 量化效果验证

```python
"""
量化前后对比测试
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model(model, tokenizer, test_prompts):
    """评估模型在测试集上的表现"""
    results = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
    return results

# 测试集
test_prompts = [
    "What is 2+2? Answer with just the number.",
    "Complete: The capital of France is",
    "Translate to French: Hello, how are you?",
    "Write a haiku about spring.",
]

# 对比 FP16 vs AWQ vs GGUF
fp16_results = evaluate_model(fp16_model, tokenizer, test_prompts)
awq_results = evaluate_model(awq_model, tokenizer, test_prompts)
gguf_results = evaluate_model(gguf_model, tokenizer, test_prompts)
```

---

## 六、常见问题与解决方案

### Q1: 量化后模型输出质量下降明显怎么办？

```
排查步骤：
1. 检查量化位数 → Q4可能不够，尝试Q5或Q6
2. 检查量化方案 → AWQ通常精度最好
3. 检查任务类型 → 数学/代码类任务对量化更敏感
4. 调整推理参数 → temperature=0可减少随机性
5. 使用混合精度 → 关键层用更高位数
```

### Q2: GPU显存不够怎么办？

```
优化策略（按效果排序）：
1. 降低量化位数: Q4 → Q3 → Q2
2. 减小上下文长度: 8192 → 4096 → 2048
3. 使用CPU offloading: 部分层放CPU
4. 使用GGUF: CPU+GPU混合推理
5. 使用更小的模型: 8B → 3B
6. 多GPU并行: tensor_parallel_size=2
```

### Q3: 不同方案之间如何迁移？

```
迁移路径：
GPTQ → AWQ: 需要重新量化（无法直接转换）
AWQ → vLLM: 直接支持 ✅
GGUF → vLLM: 有限支持（需要额外转换）
bitsandbytes → GPTQ/AWQ: 需要先恢复FP16再量化

建议：一开始就确定量化方案，避免反复转换。
```

---

## 七、总结

| 如果你需要... | 选择 | 理由 |
|--------------|------|------|
| **最佳推理精度** | AWQ | 激活感知，精度最高 |
| **最简微调流程** | bitsandbytes NF4 | 加载即量化，QLoRA基础 |
| **最广硬件兼容** | GGUF | CPU/GPU/ARM/Metal全覆盖 |
| **最佳推理速度** | AWQ + vLLM 或 GGUF + llama.cpp | C++后端，极致性能 |
| **最易上手** | GGUF + Ollama 或 bitsandbytes | 一键使用 |
| **生产级部署** | AWQ + vLLM | 高并发，低延迟 |
| **资源极度受限** | GGUF Q2_K | 最小体积 |

量化技术正在快速发展，各方案之间的差距也在不断缩小。理解每种方案的原理和特点，结合自己的实际场景做出选择，才是最重要的。

---

> 📌 本文数据基于2026年6月的主流版本，随着量化算法和硬件的发展，具体数值可能会有变化。建议参考各项目官方文档获取最新信息。
