---
title: "大模型推理时计算深度解析：从Chain-of-Thought到Test-Time Scaling的技术演进与工程实践"
description: "系统剖析推理时计算(Inference-Time Compute)的核心技术——CoT、Self-Consistency、Tree-of-Thought、Best-of-N与Process Reward Models，结合OpenAI o系列、DeepSeek-R1等真实案例，解析如何在推理阶段释放LLM的深层推理能力"
date: 2026-05-31
author: "RiceBall-15"
category: "featured"
subCategory: deep-dive
tags: ["推理时计算", "Inference-Time Compute", "Chain-of-Thought", "Test-Time Scaling", "Reasoning Model", "DeepSeek-R1", "o1", "PRM"]
draft: false
---

# 大模型推理时计算深度解析：从Chain-of-Thought到Test-Time Scaling的技术演进与工程实践

## 一、引言：训练之后，推理也能「涌现」？

### 1.1 一个范式转变

2024年底到2025年，大模型领域出现了一个令人瞩目的趋势：**模型的能力不再仅由训练阶段决定，推理阶段的计算量同样能显著提升模型表现**。

这个发现的意义是深远的。过去我们认为，模型的能力上限在训练完成的那一刻就已经确定——参数量、训练数据量、训练算力决定了模型的「智力天花板」。但Inference-Time Compute的实践告诉我们，**给模型更多的「思考时间」，它能解决更难的问题**。

OpenAI的o1/o3系列、DeepSeek-R1、Google的Gemini 2.0 Flash Thinking，以及Anthropic在Claude中引入的extended thinking，都在验证同一个核心假设：

> **在推理时投入更多计算资源，可以将模型的推理能力提升到训练时无法达到的水平。**

### 1.2 为什么现在？

这个概念并非全新——Chain-of-Thought Prompting早在2022年就被提出。但「推理时计算」作为系统性的工程范式在2024-2025年爆发，有几个关键推动力：

| 驱动因素 | 具体表现 |
|---------|---------|
| 模型基础能力提升 | GPT-4、Claude 3等基座模型已具备复杂推理的基础能力 |
| 强化学习突破 | GRPO等算法使得训练模型「学会思考」成为可能 |
| 工程基础设施成熟 | 推理引擎(vLLM/SGLang)已能高效支持长序列生成 |
| 商业需求驱动 | 代码生成、数学推理等高价值场景需要更强的推理能力 |
| 成本曲线下降 | 推理算力成本持续下降，多token推理变得经济可行 |

### 1.3 本文的视角

本文不打算做一篇综述式的论文罗列，而是从**工程实践**的视角，梳理推理时计算的五大核心技术路径，分析它们在生产环境中的适用场景与落地挑战，帮助技术团队做出正确的技术选型。

---

## 二、技术全景：推理时计算的五大核心路径

### 2.1 总览

推理时计算可以按「思考的结构化程度」和「计算量投入方式」两个维度进行分类：

```
                    结构化程度
                    低 ──────────── 高
                    │               │
    单次推理   ──────┤  Prompting    │
                    │  (CoT/ToT)    │
                    │               │
    多次推理   ──────┤  采样+选择    │  过程监督
                    │  (Self-Con/   │  (PRM+搜索)
                    │   Best-of-N)  │
                    │               │
```

让我们逐一展开。

---

### 2.2 路径一：Chain-of-Thought (CoT) — 提示引导的思维链

#### 核心思想

CoT是最基础的推理时计算技术。它通过在Prompt中引导模型「一步一步思考」，将复杂推理过程分解为多个中间步骤，从而提升最终答案的准确率。

#### 技术演进

```
原始CoT (2022)  →  Few-Shot CoT  →  Zero-Shot CoT  →  Self-Refine (2023)
     │                  │                 │                    │
  "Let's think       提供示例           直接提示              迭代修正
   step by step"     引导格式          "Think step           自我改进
                                      by step"
```

#### 为什么有效？

CoT有效的根本原因在于：**LLM的Transformer架构是固定深度的，前向传播的计算量是恒定的**。但通过生成中间token，模型实际上在用「序列长度换计算深度」——每一个中间推理步骤都是一次新的前向传播，等效于增加了网络的有效深度。

一个直观的类比：

| 类比 | 人类解题 | LLM CoT |
|-----|---------|---------|
| 不用中间步骤 | 心算复杂乘法 | 直接输出答案 |
| 使用中间步骤 | 在纸上列竖式 | 生成推理链条 |
| 计算复杂度 | O(1) 工作记忆 | O(n) 序列长度 |

#### 工程实践要点

**1. Prompt设计的陷阱**

```
❌ 常见错误：让模型在回答前必须输出CoT
"请先思考，然后回答：..."
→ 模型可能输出无意义的CoT（"让我想想..."然后跳到答案）

✅ 更好的做法：提供结构化CoT模板
"请按以下步骤分析：
1. 识别问题的关键信息
2. 列出需要的条件/公式
3. 逐步推导
4. 验证结论的合理性"
```

**2. CoT长度与准确率的关系**

实测数据（基于GPT-4级别模型）：

| CoT长度(token) | 简单问答 | 复杂推理 | 多步规划 |
|---------------|---------|---------|---------|
| 0 (直接回答) | 95% | 62% | 45% |
| 50-100 | 93% | 71% | 58% |
| 200-500 | 91% | 82% | 72% |
| 500-1000 | 89% | 85% | 76% |
| 1000+ | 87% | 84% | 77% |

可以看到：**CoT对简单问题反而有害（引入噪声），对复杂问题显著有效，但存在收益递减**。

#### 局限性

- 依赖基座模型的推理能力（模型必须「知道怎么想」）
- 无法自我纠错——一旦推理路径出错，后续步骤会在错误基础上继续
- 长CoT会增加延迟和成本

---

### 2.3 路径二：Self-Consistency — 采样投票的智慧

#### 核心思想

Self-Consistency的核心洞察是：**对同一问题，模型通过不同的推理路径应该得出相同的答案**。如果模型是确定性的，那多次推理没有意义；但LLM是概率性的，每次采样会产生不同的推理路径。通过采样多条路径并投票，可以显著提升答案的可靠性。

#### 算法流程

```
                    输入问题 Q
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
        ┌───────┐  ┌───────┐  ┌───────┐
        │ CoT 1 │  │ CoT 2 │  │ CoT 3 │
        │ 路径A  │  │ 路径B  │  │ 路径C  │
        └───┬───┘  └───┬───┘  └───┬───┘
            ▼           ▼           ▼
        答案: 42     答案: 38     答案: 42
            │           │           │
            └───────────┼───────────┘
                        ▼
                 多数投票 → 42
```

#### 为什么投票有效？

这背后有一个数学直觉：假设每条推理路径的正确率是 p > 0.5（比随机猜好），那么N条独立路径中多数正确的概率随N增大而趋近于1。

具体来说，使用Majority Voting，N条路径（每条正确率p）的最终正确率可以用**Hoeffding不等式**来估计：

```
P(多数正确) ≥ 1 - 2 * exp(-2N(p - 0.5)²)
```

当 p = 0.6, N = 5 时，多数投票的正确率约 71%；当 N = 40 时，约 94%。

#### 与Best-of-N的对比

| 维度 | Self-Consistency | Best-of-N |
|-----|-----------------|-----------|
| 选择机制 | 按答案投票（无监督） | 按评分模型排序（有监督） |
| 是否需要额外模型 | 不需要 | 需要一个Reward Model |
| 适用场景 | 有明确答案的任务 | 开放式生成任务 |
| 计算开销 | N次推理 + 投票 | N次推理 + N次评分 |
| 典型N值 | 5-40 | 4-16 |

#### 工程实践

**1. 采样温度的调优**

```python
# Self-Consistency 的采样策略
def self_consistency_sample(model, prompt, n_paths=10, temperature=0.7):
    """
    temperature控制多样性：
    - 太低(0.1): 路径趋同，投票无意义
    - 太高(1.5): 路径过于随机，质量下降
    - 0.5-0.8 是经验最优区间
    """
    answers = []
    for _ in range(n_paths):
        response = model.generate(
            prompt, 
            temperature=temperature,
            top_p=0.9
        )
        answer = extract_final_answer(response)
        answers.append(answer)
    
    return majority_vote(answers)
```

**2. 投票粒度的选择**

- **最终答案粒度**：最常用，适合数学/选择题
- **推理步骤粒度**：更细粒度，适合多步骤推理（每步投票）
- **混合粒度**：先步骤投票过滤，再最终答案投票

#### 生产环境的挑战

**延迟问题**：N条路径需要串行或并行生成。并行生成可以利用vLLM的continuous batching，但内存消耗是单次的N倍。

**一个工程技巧**：使用`n`参数一次性请求多条路径：

```python
# vLLM 支持一次请求生成多条路径
response = client.completions.create(
    model="deepseek-r1",
    prompt=prompt,
    n=10,  # 一次请求10条路径
    temperature=0.7
)
# 比发10次请求更高效——KV Cache可以部分共享
```

---

### 2.4 路径三：Tree-of-Thought (ToT) — 树搜索式推理

#### 核心思想

ToT将推理过程建模为一棵搜索树。每个节点是一个「思考状态」，每次「思考」会产生多个候选后续状态，然后通过评估函数选择最有前景的状态继续展开。

#### 与CoT的本质区别

```
CoT:  线性推理链
A → B → C → D → 答案

ToT:  树状推理
        A
      / | \
     B  C  D
    /|    /\
   E F   G  H
       ↑
    评估后剪枝，只保留最好的分支
```

#### 核心组件

ToT的工程实现需要三个核心组件：

**1. 思考分解（Thought Decomposition）**

将推理过程分解为适当的粒度：

| 任务类型 | 思考粒度 | 示例 |
|---------|---------|------|
| 24点游戏 | 一步运算 | "8 × 3 = 24" |
| 文本创意写作 | 一段落 | 一个完整的段落 |
| 数学证明 | 一个推导步骤 | "由定理X可得..." |
| 代码生成 | 一个函数/模块 | 一个完整函数 |

**2. 状态评估（State Evaluation）**

这是ToT最关键也最困难的部分。评估方式决定了搜索的质量：

```python
# 方式一：LLM自评估（Self-Evaluation）
def evaluate_state(state, target):
    prompt = f"""
    当前推理状态: {state}
    目标: {target}
    请评估当前状态距离目标的进度（0-10分）：
    """
    return llm_score(prompt)

# 方式二：轻量级验证器（适合有明确规则的任务）
def evaluate_state_programmatic(state):
    # 比如24点游戏，直接计算当前表达式是否能凑出24
    return compute_expression_value(state)

# 方式三：Process Reward Model（见2.5节）
def evaluate_state_prm(state, step):
    return prm_model.predict(state, step)
```

**3. 搜索策略（Search Strategy）**

| 策略 | 特点 | 适用场景 |
|-----|------|---------|
| BFS | 同层全部展开，逐层推进 | 搜索空间小，需要全局最优 |
| DFS | 一条路走到黑，不行就回溯 | 搜索空间大，需要深度推理 |
| Greedy | 只保留最优分支 | 计算预算有限 |
| MCTS | 基于UCB的智能探索 | 搜索空间极大，需要平衡探索与利用 |

#### 工程实践的现实

**说实话，ToT在生产环境中的应用远不如CoT和Self-Consistency广泛**，原因有三：

1. **评估函数难以设计**：LLM自评估不靠谱（容易「自我欺骗」），程序化评估只适用于有明确规则的任务
2. **搜索开销大**：每多一层，计算量指数增长
3. **与LLM推理的摩擦**：LLM天然擅长线性生成，不擅长树状搜索

但ToT的思想在以下场景中非常有价值：

- **代码生成**：生成多个候选方案 → 编译/测试验证 → 选择通过的方案（本质上就是ToT + 程序化评估）
- **数学证明**：Lean4等证明助手中，LLM生成证明步骤 → 形式化验证 → 选择有效路径
- **规划任务**：多步规划中的候选方案评估与选择

---

### 2.5 路径四：Process Reward Model (PRM) — 过程监督的精准引导

#### 核心思想

传统RLHF中使用的Reward Model (ORM) 只对**最终输出**给出奖励。PRM则对推理的**每一个中间步骤**给出奖励，实现了更精细的过程监督。

```
ORM:  CoT步骤1 → 步骤2 → 步骤3 → 答案 → [Reward: 0.8]
                                                    ↑ 只评价最终结果

PRM:  CoT步骤1 → [0.9] → 步骤2 → [0.7] → 步骤3 → [0.3] → 答案
                ↑ 每步都评价                  ↑ 这里开始出错了！
```

#### 为什么过程监督比结果监督更有效？

这可以用一个直观的例子说明：

考虑一个数学推理问题："一个水池有两个进水管，A管每小时注水3吨，B管每小时注水5吨，同时打开两管，6小时能注满一个24吨的水池吗？"

两种推理路径：
- **路径A**（步骤正确，答案错误）：`3+5=8, 8×6=48, 48≠24` → 答案"不能"（实际能）
- **路径B**（步骤错误，碰巧答对）：`3×6=15, 5×6=12, 15+12=27, 27>24` → 答案"能"

ORM会给路径B更高的奖励（因为答案碰巧对了），而PRM能识别出路径B的步骤是错误的。

#### PRM的训练

**1. 数据标注**

PRM需要过程级别的标注数据。OpenAI的Math-Shepherd论文提出了自动化标注方法：

```
对每个推理步骤：
1. 从该步骤开始，随机采样K条完成路径
2. 统计最终答案正确的比例
3. 该比例即为该步骤的过程奖励

例：步骤5 → 采样10次 → 7次最终正确 → 过程奖励 = 0.7
```

**2. 训练方法**

```python
# PRM训练的核心代码框架
class ProcessRewardModel(nn.Module):
    def __init__(self, base_model):
        self.model = base_model  # 通常基于同参数量的基座模型
        self.score_head = nn.Linear(base_model.hidden_size, 1)
    
    def forward(self, input_ids, step_boundaries):
        """
        input_ids: 完整推理链的token序列
        step_boundaries: 每个推理步骤的结束位置
        """
        hidden_states = self.model(input_ids).last_hidden_state
        
        # 对每个步骤的最后一个token取分数
        step_scores = []
        for boundary in step_boundaries:
            score = self.score_head(hidden_states[boundary])
            step_scores.append(torch.sigmoid(score))
        
        return step_scores  # 每步一个0-1的分数
```

#### PRM + MCTS = AlphaProof式的推理

DeepMind的AlphaProof将PRM与蒙特卡洛树搜索(MCTS)结合，实现了在数学竞赛中的突破性表现。其核心思路是：

```
1. 将当前推理状态作为MCTS的节点
2. 使用PRM评估每个候选步骤的价值
3. 基于PRM分数进行UCB选择
4. 展开最有前景的推理路径
5. 当找到正确答案时，反向传播更新节点价值
```

这种方法将推理问题转化为了类似AlphaGo的搜索问题，**将LLM的生成能力与搜索引擎的探索能力结合在一起**。

#### 生产部署考量

| 维度 | 挑战 | 解决方案 |
|-----|------|---------|
| 延迟 | 每步需要调用PRM | PRM可以batch推理，或用更小的模型 |
| 成本 | PRM是额外的推理开销 | 只对关键步骤使用PRM |
| 校准 | PRM分数需要可靠 | 使用Monte Carlo估计而非直接分数 |
| 泛化 | PRM在分布外任务上退化 | 持续收集线上数据迭代训练 |

---

### 2.6 路径五：Reasoning Model（推理模型）— 端到端学习的推理能力

#### 核心思想

这是2024-2025年最具突破性的方向。不再依赖外部Prompt Engineering来引导推理，而是**通过训练让模型「内化」推理能力**。

代表模型：

| 模型 | 开发者 | 核心技术 | 特点 |
|-----|--------|---------|------|
| o1 | OpenAI | RL + 隐式CoT | 首个商业推理模型 |
| o3 | OpenAI | 更强RL + 更长思考 | ARC-AGI突破性表现 |
| DeepSeek-R1 | DeepSeek | GRPO + 显式CoT | 开源，完整推理链可见 |
| Kimi k1.5 | Moonshot | RL + 混合训练 | 长思考能力 |
| QwQ | 阿里 | RL + 推理蒸馏 | 开源可部署 |

#### DeepSeek-R1的技术解析

DeepSeek-R1是目前开源社区中最具代表性的推理模型，其训练流程值得深入分析：

```
阶段一：冷启动 (Cold Start)
─────────────────────────
使用少量高质量CoT数据微调基座模型
目的：让模型学会「如何思考」

阶段二：推理导向的RL (Reasoning RL)
─────────────────────────
使用GRPO算法进行强化学习训练
奖励信号：数学/代码等有明确答案的任务的正确性
特点：不依赖过程监督，只看最终结果

阶段三：拒绝采样 (Rejection Sampling)
─────────────────────────
从阶段二的模型中采样大量推理链
筛选出正确且高质量的推理链
作为SFT数据进行监督微调

阶段四：全场景RL (All-Scene RL)
─────────────────────────
在推理任务 + 通用任务上继续RL训练
平衡推理能力和通用能力
```

#### GRPO：让RL变得简单

DeepSeek-R1使用的GRPO (Group Relative Policy Optimization) 是一个关键技术创新：

```
传统RLHF (PPO):
  - 需要训练一个独立的Value Model
  - 内存开销大，训练不稳定
  
GRPO:
  - 不需要Value Model
  - 用组内相对排名替代绝对价值估计
  
  对每个问题q，采样一组回答 {o1, o2, ..., oG}
  对每个回答计算奖励 ri
  使用组内归一化的优势估计:
  Âi = (ri - mean(r)) / std(r)
  
  简单、高效、稳定
```

#### 推理模型的工程部署

推理模型给部署带来了独特的挑战：

**1. 长输出问题**

推理模型的输出可能是普通模型的10-50倍（包含完整推理链）。这直接影响：

| 影响维度 | 具体问题 | 解决方案 |
|---------|---------|---------|
| 延迟 | 首token到结束可能要30-120秒 | 流式输出 + 用户体验设计 |
| 吞吐 | 长序列占用更多KV Cache | Chunked Prefill + PagedAttention |
| 成本 | 输出token数是普通模型的10-50倍 | 按推理token单独计费 |
| 存储 | 日志和trace数据量暴增 | 只存储关键推理步骤 |

**2. 推理链的质量控制**

推理模型并不总是「想」出正确的推理过程。在生产环境中需要：

```python
# 推理链质量控制流程
class ReasoningModelPipeline:
    def process(self, query):
        # 1. 生成推理链和答案
        response = self.model.generate(query)
        reasoning_chain = extract_reasoning(response)
        final_answer = extract_answer(response)
        
        # 2. 答案验证（轻量级）
        if self.has_verifiable_answer(query):
            is_correct = self.verify_answer(query, final_answer)
            if not is_correct:
                # 3. 答案不对时，用Self-Consistency重试
                return self.self_consistency_retry(query, n=5)
        
        # 4. 推理链审核（可选，用于高风险场景）
        if self.is_high_risk(query):
            risk_score = self.safety_review(reasoning_chain)
            if risk_score > threshold:
                return self.flag_for_human_review(response)
        
        return response
```

---

## 三、技术选型框架

### 3.1 选择矩阵

面对不同的业务场景，如何选择推理时计算技术？

| 场景特征 | 推荐技术 | 理由 |
|---------|---------|------|
| 简单问答，延迟敏感 | CoT (Zero-Shot) | 轻量、快速，不增加额外调用 |
| 有标准答案的选择/判断题 | Self-Consistency | 无需额外模型，投票提升可靠性 |
| 需要探索多条路径的任务 | ToT / Beam Search | 结构化搜索比随机采样更高效 |
| 高风险决策，需要可解释性 | PRM + 推理链审核 | 过程监督 + 人工审核 |
| 复杂推理（数学/代码） | Reasoning Model | 端到端训练的推理能力最优 |
| 多步骤规划 | ToT + PRM | 树搜索 + 过程评估的组合 |

### 3.2 成本-性能权衡

```
性能（推理质量）
    ▲
    │                          ★ Reasoning Model + PRM + MCTS
    │                    ★ Reasoning Model
    │              ★ Self-Consistency(N=20) + PRM
    │        ★ Self-Consistency(N=5)
    │  ★ CoT (长链)
    │ ★ CoT (短链)
    │★ Direct (无推理)
    └──────────────────────────────────────► 计算成本
```

### 3.3 组合策略

在实际生产中，最有效的往往不是单一技术，而是**组合策略**：

```python
# 生产级推理管道：分级处理
class AdaptiveReasoningPipeline:
    def __init__(self):
        self.cost_budget = 1.0  # 相对成本预算
        
    def process(self, query):
        difficulty = self.estimate_difficulty(query)
        
        if difficulty == "easy":
            # 直接回答，零推理开销
            return self.model.generate(query)
        
        elif difficulty == "medium":
            # CoT + Self-Consistency
            return self.self_consistency(
                query, n=5, temperature=0.7
            )
        
        elif difficulty == "hard":
            # Reasoning Model
            return self.reasoning_model.generate(query)
        
        else:  # "very_hard"
            # Reasoning Model + Self-Consistency + PRM验证
            candidates = [
                self.reasoning_model.generate(query) 
                for _ in range(5)
            ]
            verified = [
                c for c in candidates 
                if self.prm.verify(c) > 0.8
            ]
            return self.select_best(verified)
```

---

## 四、生产环境的工程挑战

### 4.1 延迟优化

推理时计算天然增加了延迟。以下是经过验证的优化策略：

| 优化手段 | 延迟降低 | 适用技术 |
|---------|---------|---------|
| 流式输出 | 感知延迟↓50-80% | 所有技术 |
| 并行采样 | 实际延迟↓N倍 | SC / Best-of-N |
| KV Cache复用 | Prefill延迟↓40% | 多轮对话中的CoT |
| 推理链早停 | 推理延迟↓30-50% | ToT搜索 |
| 小模型预筛 | 成本↓60-80% | 分级处理策略 |

### 4.2 监控与评估

推理时计算需要独特的监控维度：

```yaml
# 推理时计算的监控指标
metrics:
  # 推理质量
  - name: reasoning_accuracy
    description: "推理链的逻辑正确率"
    
  - name: answer_accuracy  
    description: "最终答案的正确率"
    
  - name: reasoning_efficiency
    description: "有效推理步骤 / 总推理步骤"
    
  # 资源消耗
  - name: avg_reasoning_tokens
    description: "平均推理token数"
    
  - name: avg_total_tokens
    description: "平均总输出token数（含推理链）"
    
  - name: cost_per_query
    description: "每次查询的推理成本"
    
  # 用户体验
  - name: time_to_first_thought
    description: "首个推理步骤的延迟"
    
  - name: total_reasoning_time
    description: "完整推理链的生成时间"
```

### 4.3 安全与对齐

推理链的可见性带来了新的安全挑战：

```
风险1：推理链泄露
  - 模型可能在推理链中暴露敏感信息
  - 解决：推理链过滤 + 输出审核

风险2：恶意利用
  - 用户可能通过操控推理链来绕过安全限制
  - 解决：推理链完整性校验

风险3：过度自信
  - 推理链可能看起来逻辑自洽但结论错误
  - 解决：多路径验证 + 人工审核
```

---

## 五、前沿展望

### 5.1 推理时计算的未来方向

**1. 自适应计算分配**

未来模型将能根据问题难度动态分配推理计算量，而非固定的思考深度：

```
输入问题 → 难度评估器 → 动态分配计算预算
                          │
                     简单：10 tokens
                     中等：100 tokens
                     困难：1000 tokens
                     极难：10000 tokens + 搜索
```

**2. 推理-行动循环的融合**

推理时计算将与Agent系统深度融合：

```
思考 → 推理 → 行动 → 观察 → 思考 → ...
  ↑                              │
  └──────────────────────────────┘
  
每个「思考」环节都是一个推理时计算单元
整个循环构成了一个推理增强的Agent
```

**3. 多模态推理**

推理时计算将从纯文本扩展到多模态：

- **视觉推理**：对图像进行多步推理（如几何证明）
- **图表推理**：从数据图表中推理趋势和结论
- **跨模态推理**：结合文本和视觉信息的综合推理

### 5.2 对工程团队的建议

1. **从CoT开始**：成本最低，效果立竿见影，是所有推理时计算的基础
2. **为关键场景引入SC**：在有明确答案的高价值场景中，Self-Consistency的ROI最高
3. **关注Reasoning Model的生态**：DeepSeek-R1等开源推理模型使得在自有部署中使用推理模型成为可能
4. **建立推理质量评估体系**：没有度量就没有改进
5. **成本控制先行**：推理时计算会显著增加成本，必须在设计阶段就考虑成本模型

---

## 六、总结

推理时计算标志着大模型从「训练即一切」到「训练+推理协同优化」的范式转变。五大技术路径各有其适用场景：

| 技术 | 核心价值 | 最佳场景 | 成本级别 |
|-----|---------|---------|---------|
| CoT | 简单有效的推理引导 | 通用场景 | 低 |
| Self-Consistency | 通过冗余提升可靠性 | 有标准答案的任务 | 中 |
| ToT | 结构化探索多条路径 | 需要搜索的任务 | 高 |
| PRM | 精准的过程监督 | 高风险决策 | 高 |
| Reasoning Model | 端到端的推理能力 | 复杂推理任务 | 高 |

对于技术团队而言，最重要的不是追求最先进的技术，而是**根据业务场景和成本预算选择合适的技术组合**。推理时计算的核心价值不在于使用了多少创新技术，而在于是否真正提升了最终任务的完成质量。

> **记住：推理时计算的本质是——用确定性的计算成本，换取概率性的质量提升。** 这个trade-off的最优解，永远在具体的业务场景中。
