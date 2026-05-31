---
title: "大模型模型合并技术深度解析：从线性插值到TIES/DARE的工程实战"
description: "深入解析模型合并(Model Merging)技术原理与工程实践，涵盖线性插值、TIES、DARE、SLERP等主流方法，附带量化对比与生产级部署方案"
date: 2026-06-01
author: "RiceBall"
category: "aiInfra"
tags: ["模型合并", "Model Merging", "TIES", "DARE", "SLERP", "LLM", "推理优化"]
draft: false
---

## 引言：为什么模型合并值得深入研究？

在大模型时代，我们面临一个看似矛盾的困境：**基础模型越来越强大，但特定场景下的表现却往往不尽人意**。传统的解决方案是微调——收集数据、选择基座模型、调整超参数、训练、评估。这套流程虽然有效，但成本高昂、周期漫长，且每次只能产出一个专用模型。

模型合并（Model Merging）提供了一种全新的思路：**不需要训练数据，不需要GPU算力，直接将多个已微调模型的能力融合为一个新模型**。

这项技术的核心价值在于：

| 维度 | 微调 | 模型合并 |
|------|------|----------|
| 训练数据 | 需要 | 不需要 |
| 计算资源 | 需要GPU集群 | 仅需CPU |
| 耗时 | 数小时到数天 | 数秒到数分钟 |
| 试错成本 | 极高 | 极低 |
| 多能力融合 | 需要多任务训练 | 天然支持 |

本文将从原理到实战，全面解析主流模型合并技术的设计哲学、实现细节与生产级应用。

## 一、模型合并的数学本质

### 1.1 参数空间的几何直觉

理解模型合并，首先需要建立一个关键直觉：**神经网络的参数可以看作高维空间中的一个点**。

假设两个微调模型 A 和 B，它们共享相同的架构，但参数值不同。我们可以将每个模型的参数视为 $d$ 维空间中的一个向量（$d$ 通常是数十亿级别）。模型合并的本质，就是在这个高维空间中找到一个"好的"新点，使得合并后的模型同时继承两个源模型的优势。

这个几何直觉解释了为什么简单的线性插值有时有效——如果两个模型的参数在某些子空间中方向一致，线性组合就能保留共同的能力。但当参数方向差异较大时，线性插值可能产生"灾难性遗忘"，导致合并模型在两个源模型的任务上都表现下降。

### 1.2 合并的统一框架

所有模型合并方法都可以用一个统一框架描述：

```
θ_merged = M(θ_A, θ_B, α, ...)
```

其中：
- `θ_A`, `θ_B` 是源模型的参数
- `α` 是合并权重（或调度矩阵）
- `M` 是合并函数（不同的合并策略定义不同的 M）

关键差异在于 `M` 的设计哲学和 `α` 的计算方式。

## 二、主流合并方法深度解析

### 2.1 线性插值（Linear Interpolation / Weight Averaging）

**最基础也最直观的合并方法**，公式极为简洁：

```
θ_merged = α · θ_A + (1 - α) · θ_B
```

**α 的选择**：通常在 0.1 到 0.9 之间搜索，步长 0.05。实践中 α=0.5（等权平均）往往是不错的起点，但最优值通常偏向其中一个模型。

**优势**：
- 实现极其简单，几行代码即可完成
- 计算开销几乎为零
- 对于同基座微调的模型，效果往往出奇地好

**局限**：
- 假设参数空间是线性的，忽略了深层非线性关系
- 对于差异较大的模型（如不同架构或不同基座），效果急剧下降
- 容易产生"参数冲突"——某些参数被反向更新，导致能力互相抵消

**适用场景**：同基座模型的多个LoRA微调权重合并、能力相近的模型融合。

### 2.2 Task Arithmetic

Task Arithmetic 提出了一个关键洞察：**微调模型与基座模型的参数差值（task vector）编码了任务特定的知识**。

```
τ_A = θ_finetuned_A - θ_pretrained
τ_B = θ_finetuned_B - θ_pretrained

θ_merged = θ_pretrained + λ · (α · τ_A + (1 - α) · τ_B)
```

其中 `λ` 是一个缩放因子（通常为 0.1~1.0），控制任务向量的影响强度。

**核心优势**：
- 通过引入基座模型作为"锚点"，有效缓解灾难性遗忘
- `λ` 参数提供了额外的调节自由度
- 支持任务的"加法"和"减法"——可以通过反向任务向量来抑制某些能力

**λ 的调优策略**：
1. 从 λ=0.5 开始
2. 如果合并模型"过于偏向"某个源模型，降低 λ
3. 如果任务能力不足，增大 λ
4. 最佳范围通常在 0.3~0.8 之间

### 2.3 TIES-Merging

**TIES（Trim, Elect Sign, Merge）是目前最流行的合并方法之一**，由 Yale 大学的研究团队提出。它通过三个关键步骤解决了参数冲突问题。

**Step 1: Trim（裁剪）**

移除绝对值较小的参数更新，只保留显著变化的参数：

```python
def trim(task_vector, threshold=0.1):
    """保留 top 10% 的参数更新"""
    k = int(task_vector.numel() * threshold)
    values, indices = torch.topk(task_vector.abs().flatten(), k)
    mask = torch.zeros_like(task_vector).flatten()
    mask[indices] = 1
    return task_vector * mask.reshape(task_vector.shape)
```

直觉：微调过程中，大部分参数的变化是噪声，只有少数参数真正编码了任务知识。裁剪噪声参数能大幅减少合并时的冲突。

**Step 2: Elect Sign（符号选举）**

对于每个参数位置，统计所有任务向量的符号，取多数投票结果：

```python
def elect_sign(task_vectors):
    """多数投票决定每个参数的符号"""
    signs = torch.stack([torch.sign(tv) for tv in task_vectors])
    majority_sign = torch.sign(signs.sum(dim=0))
    return majority_sign
```

这是 TIES 的核心创新——**当两个模型对同一参数的更新方向相反时，选择多数方的方向，而不是简单平均**。这从根本上解决了参数冲突问题。

**Step 3: Merge（合并）**

只保留与多数符号一致的参数更新，然后平均：

```python
def ties_merge(task_vectors, base_model, alpha=0.5):
    trimmed = [trim(tv) for tv in task_vectors]
    signs = elect_sign(trimmed)
    
    merged = base_model.clone()
    for tv in trimmed:
        # 只保留符号一致的更新
        mask = (torch.sign(tv) == signs).float()
        merged += alpha * tv * mask / len(task_vectors)
    
    return merged
```

**TIES 的核心优势**：
- 显式处理参数冲突，合并质量显著优于简单平均
- 对不同的 α 值鲁棒性更强
- 在多模型合并场景下优势尤为明显

### 2.4 DARE（Drop And REscale）

DARE 由 UC Berkeley 提出，核心思想极其大胆：**随机丢弃大部分微调参数，然后对保留的参数进行重新缩放**。

```python
def dare_merge(base_model, finetuned_models, drop_rate=0.9, alpha=0.5):
    merged = base_model.clone()
    
    for model in finetuned_models:
        delta = model - base_model
        
        # 随机丢弃
        mask = torch.bernoulli(torch.ones_like(delta) * (1 - drop_rate))
        
        # 重新缩放（保持期望值不变）
        rescaled_delta = delta * mask / (1 - drop_rate)
        
        merged += alpha * rescaled_delta
    
    return merged
```

**为什么丢弃90%的参数反而有效？**

这看似违反直觉，但背后的原理是：微调只改变了模型参数的一小部分，且这些改变中包含大量噪声。随机丢弃相当于一种正则化——保留最显著的信号，同时消除噪声。重新缩放确保了保留参数的期望贡献不变。

**DARE 的关键超参数**：
- `drop_rate`：通常 0.9~0.95（丢弃率）
- 较高的丢弃率在模型差异较大时更稳健
- 可与 TIES 结合使用（先 TIES 再 DARE）

### 2.5 SLERP（Spherical Linear Interpolation）

SLERP 来自球面几何学，将参数视为超球面上的点，沿大圆弧进行插值：

```python
def slerp(theta_A, theta_B, alpha):
    """球面线性插值"""
    # 归一化
    vec_A = theta_A / torch.norm(theta_A)
    vec_B = theta_B / torch.norm(theta_B)
    
    # 计算夹角
    omega = torch.acos(torch.clamp(torch.dot(vec_A, vec_B), -1, 1))
    
    if omega < 1e-6:
        return (1 - alpha) * theta_A + alpha * theta_B
    
    sin_omega = torch.sin(omega)
    coeff_A = torch.sin((1 - alpha) * omega) / sin_omega
    coeff_B = torch.sin(alpha * omega) / sin_omega
    
    return coeff_A * theta_A + coeff_B * theta_B
```

**SLERP vs 线性插值的核心差异**：

线性插值在参数空间中走直线，而 SLERP 沿球面弧线移动。当两个参数向量的模长差异较大时，SLERP 能保持更稳定的"步长"，避免合并后参数幅值的剧烈变化。

**适用场景**：
- 两个模型的合并（SLERP 天然只支持两个模型）
- 对参数范数敏感的场景
- mergekit 等工具的默认推荐方法

## 三、实战工具链

### 3.1 mergekit：事实上的标准工具

[mergekit](https://github.com/arcee-ai/mergekit) 是目前最成熟的模型合并工具，支持所有主流合并方法。

**安装**：
```bash
pip install mergekit
```

**配置文件示例**（YAML格式）：
```yaml
models:
  - model: meta-llama/Llama-3-8B-Instruct
    parameters:
      weight: 0.5
  - model: meta-llama/Llama-3-8B
    parameters:
      weight: 0.5

merge_method: ties
base_model: meta-llama/Llama-3-8B
parameters:
  density: 0.6
  weight: 0.5
  normalize: true
dtype: bfloat16
out_path: merged-model
```

**执行合并**：
```bash
mergekit-yaml config.yaml output_dir --cuda
```

### 3.2 合并策略选择决策树

根据实际需求选择合适的合并策略：

```
两个模型需要合并？
├── 是否同基座微调？
│   ├── 是 → 线性插值或SLERP
│   │   ├── 参数差异小 → α=0.5 线性插值
│   │   └── 参数差异大 → SLERP
│   └── 否 → Task Arithmetic
│       ├── 两个模型 → SLERP + Task Arithmetic
│       └── 三个以上模型 → TIES-Merging
├── 是否存在明显参数冲突？
│   ├── 是 → TIES-Merging
│   └── 否 → DARE（更鲁棒）
└── 是否需要极致性能？
    └── 搜索最优α/λ参数
```

### 3.3 完整合并流程

```python
from mergekit.merge import MergeConfiguration, run_merge
from mergekit.config import ModelConfig, MergeMethod

# 方式一：通过YAML配置（推荐用于生产）
config = MergeConfiguration.parse_file("merge_config.yaml")
run_merge(config, output_dir="merged_output")

# 方式二：编程接口
config = MergeConfiguration(
    models=[
        ModelConfig(model_path="model_a", parameters={"weight": 0.6}),
        ModelConfig(model_path="model_b", parameters={"weight": 0.4}),
    ],
    merge_method="ties",
    base_model="base_model_path",
    parameters={
        "density": 0.6,
        "weight": 0.5,
        "normalize": True,
    },
    dtype="bfloat16",
)
run_merge(config, output_dir="merged_output")
```

## 四、生产级最佳实践

### 4.1 合并前的模型评估

在合并之前，必须对源模型进行系统评估，确保它们各自在其擅长领域表现良好：

```python
# 使用 lm-evaluation-harness 评估源模型
# 评估维度应覆盖合并后需要的所有能力
evaluation_tasks = {
    "通用能力": ["hellaswag", "arc_challenge", "mmlu"],
    "数学推理": ["gsm8k", "math"],
    "代码生成": ["humaneval", "mbpp"],
    "中文能力": ["ceval", "cmmlu"],
}

for model in [model_a, model_b]:
    print(f"\nEvaluating {model.name}:")
    for domain, tasks in evaluation_tasks.items():
        results = evaluate(model, tasks)
        print(f"  {domain}: {results}")
```

### 4.2 α 参数的系统化搜索

不要依赖直觉选择合并权重，应该进行系统化搜索：

```python
import itertools

def grid_search_alpha(model_a, model_b, eval_func, alpha_range=None):
    """网格搜索最优合并权重"""
    if alpha_range is None:
        alpha_range = [i/20 for i in range(1, 20)]  # 0.05 to 0.95
    
    results = {}
    for alpha in alpha_range:
        merged = linear_merge(model_a, model_b, alpha)
        score = eval_func(merged)
        results[alpha] = score
        print(f"α={alpha:.2f} → Score={score:.4f}")
    
    best_alpha = max(results, key=results.get)
    print(f"\nBest α={best_alpha:.2f} with Score={results[best_alpha]:.4f}")
    return best_alpha, results
```

### 4.3 合并后的深度验证

合并完成后的验证应比单模型评估更严格：

```python
# 合并验证清单
validation_checklist = [
    # 1. 基础能力不退化
    ("基础能力", ["hellaswag", "arc_challenge"]),
    
    # 2. 各源模型的专有能力保持
    ("模型A专有能力", model_a_strengths),
    ("模型B专有能力", model_b_strengths),
    
    # 3. 新增能力验证
    ("融合后新能力", combined_capabilities),
    
    # 4. 边界情况测试
    ("边界情况", edge_cases),
    
    # 5. 对抗性测试
    ("安全性", safety_evals),
]

for name, tasks in validation_checklist:
    results = evaluate(merged_model, tasks)
    assert results > threshold, f"{name} 验证失败！Score={results}"
```

### 4.4 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 合并后模型能力退化 | 参数冲突导致信息丢失 | 使用 TIES 方法，增加 density |
| 某一能力完全丢失 | 合并权重不合理 | 调整 α，偏向被削弱的模型 |
| 输出质量不稳定 | 参数范数不一致 | 使用 normalize=True |
| 推理速度下降 | 参数分布变化影响推理 | 合并后进行量化（GPTQ/AWQ） |
| 中文能力退化 | 英文模型权重过大 | 评估中文benchmark，调整权重 |

## 五、进阶：多模型合并策略

当需要合并三个及以上模型时，策略变得更加复杂：

### 5.1 两两合并 vs 一次性合并

```python
# 策略一：两两合并（级联）
# A+B → AB, AB+C → ABC
# 优点：每次只处理两个模型，调试简单
# 缺点：累积误差，后面的模型可能主导结果

# 策略二：一次性多模型合并
# TIES/DARE 原生支持多模型
# 优点：所有模型平等参与，冲突处理更一致
# 缺点：参数量大，需要更多内存
```

### 5.2 实际案例：构建全能模型

假设我们有三个专精模型：
- Model-Math：数学推理能力突出
- Model-Code：代码生成能力强
- Model-CN：中文理解和生成能力优秀

```yaml
# merge_config.yaml
models:
  - model: ./model-math
    parameters:
      weight: 0.35
  - model: ./model-code
    parameters:
      weight: 0.35
  - model: ./model-cn
    parameters:
      weight: 0.30

merge_method: ties
base_model: meta-llama/Llama-3-8B
parameters:
  density: 0.5
  weight: 0.5
  normalize: true
dtype: bfloat16
out_path: ./model-allround
```

## 六、模型合并的未来方向

### 6.1 自适应合并

当前的合并方法都需要手动选择超参数。未来的方向是**自动化的合并策略搜索**——根据源模型的特性自动选择最优的合并方法和参数。

### 6.2 渐进式合并

不同于一次性合并所有模型，渐进式合并逐步融合模型能力，每步都进行验证，确保能力不退化。这种方法更安全但更耗时。

### 6.3 条件合并与动态路由

将合并从静态权重组合升级为**动态路由**——根据输入内容自动选择使用哪个（或哪些）源模型的能力。这本质上是将模型合并从参数空间扩展到了推理空间。

## 总结

模型合并是大模型时代最被低估的技术之一。它不需要训练数据、不需要GPU算力，却能在几秒钟内将多个专精模型的能力融合为一个全能模型。

**核心要点回顾**：

1. **线性插值**是起点，适合同基座模型的简单合并
2. **Task Arithmetic** 通过引入基座模型锚点缓解遗忘
3. **TIES-Merging** 通过裁剪-符号选举-合并三步法解决参数冲突
4. **DARE** 通过随机丢弃+重新缩放提供鲁棒的合并方案
5. **SLERP** 在球面空间插值，适合参数范数敏感的场景
6. 生产环境中，**mergekit** 是事实标准工具

模型合并不是微调的替代品，而是微调的有力补充。在快速迭代、多能力融合、低成本探索等场景下，它提供了微调无法比拟的效率优势。掌握这项技术，将为你的AI工程实践增添一件强大的武器。
