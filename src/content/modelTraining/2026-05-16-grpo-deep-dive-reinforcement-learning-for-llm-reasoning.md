---
title: "GRPO深度解析：从PPO到群组相对策略优化，LLM推理能力的强化学习革命"
description: "深入剖析GRPO算法原理、与PPO的核心差异、DeepSeek R1的训练策略，以及如何用TRL/Axolotl实战GRPO训练"
date: 2026-05-16
author: RiceBall-15
category: modelTraining
tags: ["GRPO", "强化学习", "DeepSeek R1", "PPO", "RLHF", "LLM推理", "策略优化"]
draft: false
---

# GRPO深度解析：从PPO到群组相对策略优化，LLM推理能力的强化学习革命

## 引言：为什么GRPO改变了游戏规则

2025年1月，DeepSeek R1横空出世，在数学推理和代码生成任务上达到了与OpenAI o1媲美的水平。而支撑这一突破的核心算法，不是传统的PPO（Proximal Policy Optimization），而是一种名为**GRPO（Group Relative Policy Optimization）**的强化学习方法。

GRPO的核心洞察极其简洁：**不需要单独训练一个价值模型（Critic），而是用同一问题下多个采样结果的相对排名作为基线**。这一设计直接砍掉了RLHF训练中最大、最不稳定的组件——价值网络，将训练成本降低了40-50%，同时在推理任务上取得了更好的效果。

本文将从第一性原理出发，剖析GRPO的设计动机、数学推导、与PPO的关键差异，以及在实际训练中的工程细节。

---

## 第一部分：PPO在LLM训练中的困境

### 1.1 经典RLHF的三阶段流水线

传统RLHF（Reinforcement Learning from Human Feedback）训练包含三个阶段：

```
阶段1: SFT（监督微调）
  预训练模型 + 高质量指令数据 → SFT模型

阶段2: 奖励模型训练
  人类偏好数据（回答A > 回答B）→ 奖励模型 RM

阶段3: PPO强化学习
  SFT模型（Actor）+ 奖励模型（Critic）+ KL惩罚 → 优化策略
```

阶段3的PPO训练涉及**四个模型同时在GPU上运行**：

| 模型 | 角色 | 显存占用 |
|------|------|----------|
| Actor（策略模型） | 生成回答，接受梯度更新 | 满参数 + 优化器状态 |
| Critic（价值模型） | 估计状态价值V(s)，计算优势函数 | 满参数 + 优化器状态 |
| Reference Model | 计算KL散度，防止策略偏移 | 满参数（冻结） |
| Reward Model | 提供奖励信号 | 满参数（冻结） |

对于一个7B参数的模型，这四个模型的显存需求：

```
Actor: 7B × (2 + 4 + 4) bytes ≈ 70GB（fp16 + Adam优化器）
Critic: 7B × 10 bytes ≈ 70GB
Reference: 7B × 2 bytes ≈ 14GB
Reward: 7B × 2 bytes ≈ 14GB
总计: ~168GB（仅模型参数，不含激活值和KV-Cache）
```

### 1.2 Critic模型的三大痛点

**痛点一：训练不稳定**

Critic模型需要准确估计"在当前状态下，未来能获得多少奖励"。但LLM的生成空间极其庞大，Critic的估计经常出现高方差，导致Actor的梯度信号噪声很大。

```
实际观察到的问题：
- Critic loss震荡不收敛
- Actor梯度突然爆炸（gradient spike）
- 训练过程中奖励先升后降（reward hacking后崩溃）
```

**痛点二：Critic和Actor的不对称性**

在标准RLHF中，Critic和Actor通常共享同一个基础模型。但两者的训练目标本质不同——Actor优化生成质量，Critic优化价值估计精度。实践中发现：

- 用同一个模型初始化Actor和Critic，Critic的表现往往不佳
- 单独训练一个高质量Critic又增加了额外成本
- Critic的容量如果小于Actor，会成为性能瓶颈

**痛点三：计算开销巨大**

在每个PPO训练步骤中：

```
1. Actor生成N个回答（前向传播）
2. Critic计算每个token的价值估计（前向传播）
3. Reward Model计算奖励（前向传播）
4. Reference计算KL散度（前向传播）
5. 计算GAE优势函数
6. Actor更新（反向传播）
7. Critic更新（反向传播）

7次模型前向/反向传播 × 4个模型
```

---

## 第二部分：GRPO的核心设计

### 2.1 关键洞察：用群组采样替代价值函数

GRPO的核心思想来自一个简单的观察：

> 对于同一个问题，如果我们采样多个回答，那么这些回答之间的**相对排名**本身就包含了"优势"信息。

传统PPO用GAE（Generalized Advantage Estimation）计算优势：

$$A_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，需要一个训练好的 $V(s)$。

GRPO的替代方案：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, ..., r_G\})}{\text{std}(\{r_1, r_2, ..., r_G\})}$$

其中 $G$ 是对同一个问题采样的回答数量，$r_i$ 是第 $i$ 个回答的奖励。

**这个公式意味着什么？**

- 如果一个问题采样了8个回答，奖励分别是 [0.2, 0.5, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4]
- 均值 = 0.475，标准差 = 0.271
- 得分0.9的回答，优势 = (0.9 - 0.475) / 0.271 = 1.57（正向强化）
- 得分0.1的回答，优势 = (0.1 - 0.475) / 0.271 = -1.38（负向强化）

不需要任何Critic模型，群组内的相对比较自然产生了优势信号。

### 2.2 GRPO的损失函数

GRPO的完整损失函数：

$$L_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \rho_t^{(i)} \hat{A}_i, \text{clip}(\rho_t^{(i)}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL} \right) \right]$$

其中：
- $\rho_t^{(i)} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$ 是重要性采样比率
- $\hat{A}_i$ 是群组归一化后的优势
- $\epsilon$ 是PPO裁剪参数（通常0.2）
- $\beta$ 是KL惩罚系数
- $D_{KL}$ 是与Reference模型的KL散度

**与标准PPO的关键差异：**

| 维度 | PPO | GRPO |
|------|-----|------|
| 优势估计 | GAE，需要Critic模型 $V(s)$ | 群组内奖励归一化，无需Critic |
| 计算的模型数 | 4个（Actor + Critic + Ref + RM） | 3个（Actor + Ref + RM） |
| Token级 vs 序列级 | Token级优势（每个token有独立的A_t） | **序列级优势（整个回答共享同一个A_i） |
| 显存需求 | 高（需要加载Critic） | 低（省去Critic） |
| 适用场景 | 通用RL任务 | 推理任务（有明确正确答案） |

### 2.3 序列级优势：一个重要的设计选择

GRPO中，同一个回答内的所有token共享同一个优势值 $\hat{A}_i$。这看似粗糙，但实际上是一个深思熟虑的设计：

**为什么不需要token级优势？**

在推理任务中（数学、代码），最终奖励通常只在序列结束时给出（答案正确/错误）。中间token的贡献很难单独评估。Critic模型在这种稀疏奖励场景下很难学到准确的token级价值估计，反而引入噪声。

**群组归一化天然处理了这个问题**：正确回答的所有token被正向强化，错误回答的所有token被负向强化。虽然粒度粗，但信号干净。

**对比：PPO的token级优势为什么在LLM中问题重重？**

```
问题：Critic需要为每个token估计V(s_t)
- "The answer is 42" 
  V("The") = ? V("answer") = ? V("is") = ? V("42") = ?
- 在开放式生成中，中间状态的价值极难估计
- Critic通常只能学到浅层启发式（如"长回答价值高"）
- 这些启发式对推理任务几乎无用
```

---

## 第三部分：DeepSeek R1的GRPO训练策略

### 3.1 DeepSeek R1-Zero：纯RL训练的惊人发现

DeepSeek R1-Zero是第一个完全通过GRPO训练、不使用任何SFT数据就展现出推理能力的模型。其训练流程：

```
DeepSeek-V3-Base（预训练模型）
    ↓
直接应用GRPO（无SFT阶段）
    ↓
DeepSeek-R1-Zero
```

**奖励设计**（极其简洁）：

| 奖励类型 | 设计 | 说明 |
|----------|------|------|
| 准确性奖励 | 答案正确 → +1，错误 → 0 | 基于规则的二值奖励 |
| 格式奖励 | 正确使用 `<think>...</think>` 标签 | 鼓励模型展示推理过程 |

没有过程奖励模型（PRM），没有人类标注，只有最终答案的对错判断。

### 3.2 涌现的推理行为

R1-Zero在GRPO训练过程中自发涌现了多种推理行为：

**行为一：思维链长度自动增长**

```
训练步数:  1K    5K    10K   20K   50K
平均推理token: 200   500   1200  3000  8000+
```

模型发现更长的推理过程能带来更高的准确率，因此自动学会了"多想一会儿"。

**行为二：自我反思和纠错**

模型在推理过程中出现了类似以下的模式：

```
<think>
Let me solve this step by step...
Wait, that doesn't seem right. Let me reconsider...
Actually, I made an error in step 3. The correct approach is...
Hmm, let me verify this by trying a different method...
Yes, both methods give the same answer. I'm confident now.
</think>
```

这种"Wait"、"Let me reconsider"等模式完全是通过RL训练自发涌现的，没有在任何训练数据中显式出现过。

**行为三：Aha Moment（顿悟时刻）**

训练过程中观察到模型在某些困难问题上突然从"胡乱尝试"转变为"系统性推理"，类似人类的顿悟体验。这个现象在GRPO训练曲线中表现为奖励的突然跳跃。

### 3.3 R1的完整训练流水线

R1-Zero虽然推理能力强，但输出可读性差、语言混用。DeepSeek R1的完整训练流水线：

```
阶段1: 冷启动SFT
  - 使用少量高质量长推理数据（含<think>标签）对V3-Base进行SFT
  - 目的：教会模型基本的推理格式

阶段2: 推理导向的GRPO
  - 使用数学、代码、逻辑推理任务
  - 准确性奖励 + 格式奖励
  - 训练直到推理能力收敛

阶段3: 拒绝采样 + SFT
  - 用阶段2的模型生成大量推理轨迹
  - 筛选正确且可读性好的轨迹
  - 混合通用SFT数据，重新训练
  - 目的：提升输出质量和语言一致性

阶段4: 全场景GRPO
  - 在推理 + 通用任务上进行最终GRPO训练
  - 使用规则奖励 + 奖励模型的混合信号
  - 加入语言一致性奖励
```

**关键设计决策：**

- 阶段2只用规则奖励（答案对错），不引入奖励模型——避免reward hacking
- 阶段3的拒绝采样相当于"从RL策略中提取最好的轨迹做SFT"——这是一种策略蒸馏
- 阶段4混合使用多种奖励信号——平衡推理能力和通用能力

---

## 第四部分：GRPO vs 其他RL方法的系统对比

### 4.1 方法对比全景

| 方法 | Critic | 优势估计 | 奖励来源 | 训练稳定性 | 计算成本 | 适用场景 |
|------|--------|----------|----------|------------|----------|----------|
| **PPO** | ✅ 需要 | GAE（token级） | RM | 中等 | 高（4模型） | 通用RLHF |
| **GRPO** | ❌ 不需要 | 群组归一化（序列级） | 规则/RM | 高 | 低（3模型） | 推理任务 |
| **REINFORCE** | ❌ 不需要 | 单条轨迹回报 | RM | 低（高方差） | 低 | 简单任务 |
| **DPO** | ❌ 不需要 | 隐式（偏好对） | 人类偏好 | 高 | 低（2模型） | 偏好对齐 |
| **RLOO** | ❌ 不需要 | 留一法估计 | RM | 中等 | 中等 | 通用 |
| **ReMax** | ❌ 不需要 | 贪婪解码基线 | RM | 中等 | 低 | 推理任务 |

### 4.2 GRPO vs PPO：核心差异深度对比

**差异一：优势估计的质量**

```
PPO的GAE:
  优势 = Σ(γλ)^l × (r_t + γV(s_{t+1}) - V(s_t))
  质量取决于Critic的准确性
  如果Critic不准 → 优势估计有偏 → Actor学到错误信号

GRPO的群组归一化:
  优势 = (r_i - mean(r)) / std(r)
  质量取决于采样数量G和奖励的区分度
  G越大，估计越稳定
  不需要任何可学习组件
```

**差异二：训练动态**

PPO的训练动态受Critic和Actor的交互影响，可能出现"双人博弈"式的不稳定：

```
Critic估计偏高 → Actor获得虚假正向信号 → Actor生成低质量回答
→ 奖励下降 → Critic需要更新估计 → 振荡
```

GRPO消除了这个耦合——优势信号完全来自真实的奖励值，不存在估计误差的累积。

**差异三：扩展性**

```
模型规模:  1B    7B    70B    671B
PPO显存:   40GB  168GB  1.4TB  13TB+  (需要4个完整模型)
GRPO显存:  30GB  126GB  1.0TB  10TB+  (省去Critic)
节省:      25%   25%    25%    25%
```

对于DeepSeek V3（671B MoE）这样的超大模型，GRPO省去一个Critic模型的显存意味着可以用更少的GPU完成训练。

### 4.3 GRPO vs DPO：不同场景的选择

DPO（Direct Preference Optimization）通过偏好对直接优化策略，不需要奖励模型：

$$L_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

| 维度 | GRPO | DPO |
|------|------|-----|
| 数据需求 | 问题 + 奖励信号（可以是规则） | 偏好对（chosen, rejected） |
| 奖励信号 | 显式（答案对错/奖励模型） | 隐式（偏好比较） |
| 推理能力提升 | ✅ 强（通过试错学习） | ⚠️ 有限（只能从偏好对中学习） |
| 训练效率 | 需要多次采样（G次/问题） | 高效（1次前向/样本） |
| 适用场景 | 有明确正确答案的推理任务 | 风格对齐、主观质量提升 |

**核心结论：GRPO擅长推理（有客观标准），DPO擅长偏好对齐（主观标准）。两者互补，不是替代关系。**

---

## 第五部分：GRPO实战——用TRL训练你的推理模型

### 5.1 环境准备

```bash
pip install trl>=0.15.0 transformers>=4.48.0 peft accelerate
pip install vllm  # 可选，用于加速采样
```

### 5.2 最小化GRPO训练脚本

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 加载模型
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载数学推理数据集
dataset = load_dataset("openai/gsm8k", "main", split="train")

# 定义奖励函数（关键！）
def accuracy_reward(completions, prompts, **kwargs):
    """基于答案正确性的奖励函数"""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # 从completion中提取最终答案
        answer = extract_answer(completion)  # 你的答案提取逻辑
        ground_truth = extract_ground_truth(prompt)  # 你的标准答案提取逻辑
        rewards.append(1.0 if answer == ground_truth else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """格式奖励：鼓励使用<think>标签"""
    rewards = []
    for completion in completions:
        has_think = "<think>" in completion and "</think>" in completion
        rewards.append(0.1 if has_think else 0.0)
    return rewards

# GRPO配置
training_args = GRPOConfig(
    output_dir="./grpo-qwen2.5-7b-math",
    num_generations=8,          # 每个问题采样8个回答（G=8）
    max_completion_length=2048,  # 最大生成长度
    num_train_epochs=1,
    per_device_train_batch_size=2,  # 每个问题算一个"样本"
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    beta=0.04,                   # KL惩罚系数
    epsilon=0.2,                 # PPO裁剪参数
    temperature=0.7,             # 采样温度
    reward_funcs=[accuracy_reward, format_reward],
    logging_steps=10,
    save_steps=500,
    bf16=True,
)

# 训练
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### 5.3 奖励函数设计的艺术

GRPO的效果高度依赖奖励函数的设计。以下是几个关键原则：

**原则一：奖励要稀疏但准确**

```
❌ 错误做法：连续奖励（如输出长度相关的奖励）
  → 模型会学会生成长而无意义的内容

✅ 正确做法：二值奖励（答案正确=1，错误=0）
  → 信号干净，模型专注于提升准确率
```

**原则二：多维度奖励要合理加权**

```python
def combined_reward(completions, prompts, **kwargs):
    acc_reward = accuracy_reward(completions, prompts, **kwargs)
    fmt_reward = format_reward(completions, **kwargs)
    len_reward = length_reward(completions, **kwargs)
    
    # 权重分配：准确性 >> 格式 >> 长度
    return [
        1.0 * a + 0.1 * f + 0.05 * l 
        for a, f, l in zip(acc_reward, fmt_reward, len_reward)
    ]
```

**原则三：避免奖励黑客（Reward Hacking）**

常见的reward hacking模式：

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 模型输出越来越长 | 长回答碰巧得分高 | 不使用长度相关奖励 |
| 模型学会复制题目 | 某些格式匹配得高分 | 奖励函数检查答案而非格式 |
| 模型只输出答案不推理 | 直接输出答案效率更高 | 推理过程单独给小奖励 |
| 模型用markdown表格包装 | 格式奖励被利用 | 奖励函数只检查内容 |

### 5.4 采样数量G的选择

G（每个问题的采样数量）是GRPO最重要的超参数：

| G值 | 优势 | 劣势 | 推荐场景 |
|-----|------|------|----------|
| 4 | 计算快，显存低 | 优势估计方差大 | 快速实验 |
| 8 | 平衡点 | 中等开销 | **默认推荐** |
| 16 | 优势估计稳定 | 显存和计算开销大 | 最终训练 |
| 32+ | 极其稳定 | 开销过大 | 研究用途 |

**经验法则：** G应该大到保证每个问题至少有1个正确和1个错误回答。如果准确率已经是90%，G=4就够了（期望0.4个错误）；如果准确率只有10%，需要G=16以上。

### 5.5 LoRA + GRPO：低资源训练方案

对于显存有限的场景，可以结合LoRA进行参数高效训练：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# 显存对比（7B模型）：
# 全参数GRPO: ~126GB（Actor + Ref + RM）
# LoRA GRPO:  ~50GB（LoRA参数很小，Ref也可用LoRA）
```

---

## 第六部分：GRPO的局限与未来方向

### 6.1 当前局限

**局限一：依赖可验证的奖励信号**

GRPO在有明确正确答案的任务上效果最好（数学、代码、逻辑推理）。对于开放式任务（写作、对话），很难定义群组内的相对优劣。

**局限二：采样效率**

每个训练步需要对同一个问题生成G个完整回答，这比DPO（只需要1个前向传播/样本）的计算开销大得多。

**局限三：优势估计的粒度**

序列级优势无法区分"推理过程正确但最后计算错误"和"推理过程完全错误"两种情况。这导致一些有用的推理模式被负向强化。

### 6.2 前沿改进方向

**方向一：过程奖励模型（PRM）+ GRPO**

将PRM与GRPO结合，为推理过程中的每一步提供奖励信号：

```
传统GRPO: 整个回答 → 最终答案对错 → 一个奖励值
PRM-GRPO: 每个推理步骤 → 步骤正确性 → 每步一个奖励值
```

OpenAI的"Let's Verify Step by Step"论文已经证明了PRM的价值。将PRM集成到GRPO中是自然的下一步。

**方向二：自适应采样**

根据问题难度动态调整G值：
- 简单问题：G=4（节省计算）
- 困难问题：G=16（保证信号质量）

**方向三：多轮GRPO**

当前GRPO是单轮采样。多轮GRPO可以让模型在第一轮采样的基础上进行修正和改进：

```
Round 1: 采样G个回答，选择最好的
Round 2: 以最好的回答为起点，再次采样G个改进版本
Round 3: ...
```

---

## 总结

GRPO通过一个极其简洁的设计——用群组内采样的相对排名替代Critic模型——解决了PPO在LLM训练中的核心痛点：

1. **去掉Critic**：降低40%显存占用，消除训练不稳定的主要来源
2. **序列级优势**：适配LLM推理任务的稀疏奖励特性
3. **简洁的奖励设计**：规则奖励（答案对错）就足够驱动推理能力涌现
4. **工程友好**：实现简单，超参数少，易于复现

GRPO不是PPO的"简化版"，而是针对LLM推理任务特点的**定制化优化**。在DeepSeek R1的成功验证之后，GRPO已经成为LLM强化学习训练的主流方法之一。

如果你正在考虑训练自己的推理模型，GRPO应该是第一个尝试的方法——它的投入产出比远高于PPO，尤其在有明确评估标准的推理任务上。

---

## 参考来源

1. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.
2. Shao, Zhihong, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024.（GRPO首次提出）
3. Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
4. Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS, 2023.
5. Lightman, Hunter, et al. "Let's Verify Step by Step." arXiv:2305.20050, 2023.
6. TRL Documentation. "GRPO Trainer." https://huggingface.co/docs/trl/grpo_trainer
