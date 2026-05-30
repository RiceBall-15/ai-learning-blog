---
title: LLM评测体系实战指南
description: 从评测维度、框架使用、数据集构建到结果分析，全面掌握大语言模型评测的方法论与工程实践
date: 2026-05-30
author: RiceBall-15
category: aiInfra
subCategory: evaluation
tags:
  - LLM
  - evaluation
  - benchmark
  - OpenCompass
  - lm-evaluation-harness
  - model-selection
draft: false
---

# LLM评测体系实战指南

## 引言

随着大语言模型（LLM）的爆发式增长，从 GPT-4、Claude 3.5 到 Llama 3、Qwen 2.5、DeepSeek-V3，模型种类繁多、能力各异。**如何科学、系统地评测一个 LLM，已成为 AI 工程落地的核心问题。**

本文从四个核心维度出发，结合 OpenCompass 和 lm-evaluation-harness 两大主流评测框架的实战用法，提供一套可复用的 LLM 评测体系方案。

---

## 一、LLM 评测维度全景

一个完整的 LLM 评测体系应覆盖以下四个核心维度：

```
┌─────────────────────────────────────────────────────┐
│                LLM 评测维度全景                       │
├──────────┬──────────┬──────────┬────────────────────┤
│  准确性   │  安全性   │  效率    │      成本          │
│ Accuracy │ Safety  │Efficiency│     Cost           │
├──────────┼──────────┼──────────┼────────────────────┤
│ 知识问答  │ 有害内容 │ 推理速度 │  API 调用费用      │
│ 推理能力  │ 隐私保护 │ 并发能力 │  训练/微调成本     │
│ 代码生成  │ 偏见检测 │ 内存占用 │  人力运维成本      │
│ 数学计算  │ 越狱防御 │ 吞吐量   │  性价比比值        │
└──────────┴──────────┴──────────┴────────────────────┘
```

### 1.1 准确性（Accuracy）

准确性是 LLM 评测的核心，通常细分为：

| 能力类别 | 代表任务 | 典型数据集 | 评测指标 |
|---------|---------|-----------|---------|
| 知识问答 | 多选题、开放问答 | MMLU, C-Eval, ARC | Accuracy, F1 |
| 推理能力 | 常识推理、逻辑推理 | HellaSwag, WinoGrande | Accuracy |
| 数学能力 | 算术、代数、几何 | GSM8K, MATH | Exact Match |
| 代码能力 | 函数补全、Bug 修复 | HumanEval, MBPP | Pass@k |
| 长文本 | 长上下文理解 | LongBench, RULER | F1, ROUGE-L |
| 多语言 | 跨语言理解与生成 | MGSM, XNLI | Accuracy |
| 指令遵循 | 格式化输出、约束满足 | IFEval | Strict Accuracy |

### 1.2 安全性（Safety）

安全性评测关注模型在对抗性场景下的鲁棒表现：

| 安全子维度 | 评测方法 | 说明 |
|-----------|---------|------|
| 有害内容生成 | Toxicity 检测 | 模型是否输出暴力、仇恨等有害内容 |
| 隐私泄露 | 成员推理攻击 | 模型是否记忆训练数据中的个人信息 |
| 偏见与公平性 | BBQ, WinoBias | 模型是否对不同群体产生偏见性回答 |
| 越狱防御 | Red-teaming | 面对精心构造的 prompt 是否保持安全边界 |
| 幻觉检测 | Faithfulness 评估 | 模型是否编造不存在的事实 |

### 1.3 效率（Efficiency）

| 指标 | 定义 | 测量方式 |
|-----|------|---------|
| 推理延迟（Latency） | 首 Token 时间（TTFT） | 时间戳差值 |
| 吞吐量（Throughput） | 每秒处理 Token 数 | tokens/s |
| 并发能力 | 同时服务请求数 | 压测工具（如 vLLM benchmark） |
| 内存占用 | 模型加载所需 VRAM | nvidia-smi, torch.cuda |
| 上下文长度 | 支持的最大 Token 数 | 递增输入测试 |

### 1.4 成本（Cost）

| 成本类型 | 评估要素 |
|---------|---------|
| API 调用费 | 输入/输出 Token 单价 × 用量 |
| 自部署成本 | GPU 硬件 + 电力 + 运维 |
| 微调成本 | 数据标注 + 训练算力 |
| 性价比 | `总成本 / 综合评测得分` |

---

## 二、主流评测框架使用指南

### 2.1 OpenCompass

OpenCompass 是上海 AI Lab 开源的 LLM 评测平台，支持 70+ 数据集、30+ 模型，提供开箱即用的评测流水线。

#### 安装与配置

```bash
# 克隆仓库
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .

# 或直接安装
pip install opencompass
```

#### 基础评测：评测一个本地模型

```bash
# 使用默认配置评测 Qwen2.5-7B
python run.py \
    --models qwen2_5_7b \
    --datasets mmlu ceval \
    --work-dir results/qwen25_7b
```

#### 高级配置：自定义评测 Config

OpenCompass 使用 Python Config 系统：

```python
# configs/eval_qwen25_custom.py

from mmengine.config import read_base

# 读取模型和数据集配置
with read_base():
    from .models.qwen.hf_qwen2_5_7b import models as qwen2_5_7b
    from .datasets.mmlu.mmlu_gen import datasets as mmlu_datasets
    from .datasets.ceval.ceval_gen import datasets as ceval_datasets

# 合并配置
models = [*qwen2_5_7b]
datasets = [*mmlu_datasets, *ceval_datasets]

# 评测后端配置
eval = dict(
    partitioner=dict(
        type='SizePartitioner',
        max_task_size=10000,
        min_task_size=100,
    ),
    runner=dict(
        type='SlurmRunner',
        max_num_workers=32,
        task=dict(type='OpenICLInferTask'),
    ),
)
```

#### 批量评测多个模型

```python
# configs/eval_comparison.py
from mmengine.config import read_base

with read_base():
    # 多个模型
    from .models.qwen.hf_qwen2_5_7b import models as qwen25
    from .models.llama.hf_llama3_8b import models as llama3
    from .models.deepseek.hf_deepseek_v3 import models as deepseek
    # 评测集
    from .datasets.mmlu.mmlu_gen import datasets as mmlu
    from .datasets.humaneval.humaneval_gen import datasets as humaneval

models = [*qwen25, *llama3, *deepseek]
datasets = [*mmlu, *humaneval]

# 结果保存
output_dir = 'results/benchmark_comparison'
```

#### 评测 API 模型

```python
# configs/eval_gpt4_api.py
from opencompass.models import OpenAI

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

models = [
    dict(
        abbr='gpt-4o',
        type=OpenAI,
        key='YOUR_API_KEY',
        path='gpt-4o',
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=8,
        rpm_limit=60,
        retry_cfg=dict(max_retry=3, wait_time=1),
    )
]
```

#### 评测结果查看

```bash
# 评测完成后查看结果
python tools/analyze_results.py \
    --results-dir results/qwen25_7b \
    --output-format markdown

# 生成可视化图表
python tools/visualize.py \
    --results results/benchmark_comparison \
    --chart-type radar
```

---

### 2.2 lm-evaluation-harness

lm-evaluation-harness（由 EleutherAI 维护）是社区最广泛使用的 LLM 评测工具，支持 HuggingFace 模型和 API 模型。

#### 安装

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[dev]"
```

#### 基础评测命令

```bash
# 评测 MMLU（5-shot）
lm_eval \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B,dtype=float16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/qwen25_7b

# 评测多个任务
lm_eval \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B,dtype=float16 \
    --tasks mmlu,arc_challenge,hellaswag,truthfulqa_mc \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/qwen25_7b_comprehensive
```

#### 评测 API 模型（OpenAI 兼容）

```bash
# 评测 GPT-4o
lm_eval \
    --model openai-completions \
    --model_args model=gpt-4o,api_base=https://api.openai.com/v1,api_key=YOUR_KEY \
    --tasks mmlu,humaneval \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_path results/gpt4o
```

#### 使用自定义任务

```python
# tasks/custom_eval.yaml
task: custom_mcq
dataset_path: json
dataset_kwargs:
  data_files: ./data/custom_mcq.jsonl
  split: train
output_type: multiple_choice
training_split: train
doc_to_text: "Question: {{question}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
```

```bash
# 运行自定义评测
lm_eval \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B \
    --tasks custom_mcq \
    --custom_tasks ./tasks \
    --batch_size auto
```

#### 对比两个模型

```bash
# 使用 lm_eval 生成对比表
lm_eval \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B \
    --tasks mmlu,arc_challenge \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/qwen25_7b

lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3-8B \
    --tasks mmlu,arc_challenge \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/llama3_8b

# 生成对比报告
python lm_eval/evaluator/results.py \
    --results results/qwen25_7b results/llama3_8b \
    --output comparison.md
```

---

### 2.3 框架对比

| 特性 | OpenCompass | lm-evaluation-harness |
|------|------------|----------------------|
| 数据集数量 | 70+ | 200+ |
| 模型支持 | HuggingFace, API, 自部署 | HuggingFace, vLLM, API |
| 中文评测 | ✅ C-Eval, CMMLU 等丰富 | ⚠️ 需要手动配置 |
| 配置方式 | Python Config | CLI + YAML |
| 分布式评测 | ✅ Slurm, 分片推理 | ⚠️ 依赖 accelerate |
| 可视化 | ✅ 内置报告生成 | ⚠️ 需配合其他工具 |
| 社区生态 | 国内活跃 | 全球社区活跃 |
| 适合场景 | 中文模型评测、企业级部署 | 快速原型评测、社区标准对比 |

**推荐策略**：中文模型优先用 OpenCompass，国际化标准评测用 lm-evaluation-harness，两者结合使用覆盖最全。

---

## 三、自定义评测数据集构建

当标准 Benchmark 无法满足特定业务场景时，需要构建自定义评测集。

### 3.1 构建流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 需求分析  │───→│ 数据收集  │───→│ 格式转换  │───→│ 质量验证  │
│          │    │          │    │          │    │          │
│ 定义能力  │    │ 专家标注  │    │ JSON/JSONL│    │ 抽样人工  │
│ 确定粒度  │    │ 爬虫采集  │    │ 标准化    │    │ 自动校验  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      │
                                                      ▼
                                              ┌──────────┐
                                              │ 迭代优化  │
                                              │ 淘汰坏题  │
                                              └──────────┘
```

### 3.2 数据格式标准

#### OpenCompass 格式（ICL）

```json
{
    "prompt": "以下是一道关于机器学习的题目：\n\n什么是过拟合（Overfitting）？",
    "answer": "过拟合是指模型在训练数据上表现很好，但在未见过的测试数据上表现较差的现象，通常因为模型过于复杂而过度拟合了训练数据中的噪声。",
    "subject": "machine_learning",
    "difficulty": "medium"
}
```

#### lm-evaluation-harness 格式（MCQ）

```json
{
    "question": "以下哪个算法属于无监督学习？",
    "choices": ["线性回归", "K-Means", "逻辑回归", "SVM"],
    "answer": "B",
    "category": "ml_basics",
    "difficulty": "easy"
}
```

#### 生成式评测格式

```json
{
    "input": "请解释什么是 Transformer 架构中的注意力机制。",
    "output": "注意力机制是 Transformer 的核心组件...",
    "reference": "注意力机制允许模型在处理每个位置时...",
    "rubric": {
        "accuracy": 0.4,
        "completeness": 0.3,
        "clarity": 0.3
    }
}
```

### 3.3 自动化构建工具

```python
# build_eval_dataset.py
import json
import hashlib
from pathlib import Path

class EvalDatasetBuilder:
    """LLM 评测数据集构建工具"""

    def __init__(self, name: str):
        self.name = name
        self.items = []

    def add_mcq(self, question: str, choices: list, answer: str,
                category: str = "general", difficulty: str = "medium"):
        """添加多选题"""
        item = {
            "id": hashlib.md5(f"{question}{answer}".encode()).hexdigest()[:8],
            "type": "mcq",
            "question": question,
            "choices": choices,
            "answer": answer,
            "category": category,
            "difficulty": difficulty,
        }
        self.items.append(item)
        return self

    def add_open_ended(self, question: str, reference: str,
                       category: str = "general", difficulty: str = "medium"):
        """添加开放题"""
        item = {
            "id": hashlib.md5(f"{question}".encode()).hexdigest()[:8],
            "type": "open_ended",
            "question": question,
            "reference": reference,
            "category": category,
            "difficulty": difficulty,
        }
        self.items.append(item)
        return self

    def validate(self):
        """验证数据集质量"""
        issues = []
        seen_ids = set()

        for i, item in enumerate(self.items):
            # 检查重复
            if item["id"] in seen_ids:
                issues.append(f"Item {i}: duplicate id '{item['id']}'")
            seen_ids.add(item["id"])

            # MCQ 检查
            if item["type"] == "mcq":
                if item["answer"] not in [c[0] for c in item["choices"]]:
                    issues.append(f"Item {i}: answer '{item['answer']}' not in choices")
                if len(item["choices"]) < 2:
                    issues.append(f"Item {i}: too few choices")

            # 空值检查
            if not item.get("question", "").strip():
                issues.append(f"Item {i}: empty question")

        return issues

    def export(self, path: str, format: str = "jsonl"):
        """导出数据集"""
        issues = self.validate()
        if issues:
            print(f"⚠️  Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ Exported {len(self.items)} items to {path}")
        return self

    def stats(self):
        """打印数据集统计"""
        from collections import Counter
        categories = Counter(item["category"] for item in self.items)
        difficulties = Counter(item["difficulty"] for item in self.items)
        types = Counter(item["type"] for item in self.items)

        print(f"📊 Dataset: {self.name}")
        print(f"   Total items: {len(self.items)}")
        print(f"   Types: {dict(types)}")
        print(f"   Categories: {dict(categories)}")
        print(f"   Difficulties: {dict(difficulties)}")


# 使用示例
builder = EvalDatasetBuilder("ai_inference_eval")

builder \
    .add_mcq(
        question="在 Transformer 中，Multi-Head Attention 的主要优势是什么？",
        choices=[
            "A. 减少计算量",
            "B. 让模型关注不同的表示子空间",
            "C. 增加参数量",
            "D. 减少训练时间"
        ],
        answer="B",
        category="transformer",
        difficulty="medium"
    ) \
    .add_open_ended(
        question="请解释 RAG（检索增强生成）的工作原理及其优缺点。",
        reference="RAG 通过检索外部知识库来增强 LLM 的生成能力...",
        category="rag",
        difficulty="medium"
    )

builder.stats()
builder.export("data/custom_eval.jsonl")
```

### 3.4 质量控制最佳实践

| 环节 | 方法 | 目标 |
|-----|------|-----|
| 人工标注 | 3人交叉标注 + 仲裁 | Kappa > 0.8 |
| 答案验证 | 多源交叉验证 | 错误率 < 2% |
| 难度校准 | GPT-4 预测 + 人工调整 | 分布合理 |
| 题目去重 | 语义相似度 > 0.9 去除 | 无重复 |
| 偏见检查 | 属性敏感度分析 | 无系统性偏见 |
| 定期更新 | 季度审查 + 淘汰坏题 | 保持有效性 |

---

## 四、评测结果分析与模型选型

### 4.1 综合评测对比表

以下为 2026 年主流开源模型的代表性评测数据（数据仅供参考，实际评测请以最新结果为准）：

| 模型 | MMLU | C-Eval | GSM8K | HumanEval | 推理速度(tokens/s) | VRAM(GB) | API价格($/M tokens) |
|------|------|--------|-------|-----------|-------------------|----------|---------------------|
| GPT-4o | 88.7 | 86.2 | 95.8 | 90.2 | 85 | - | $2.5/$10 |
| Claude 3.5 Sonnet | 88.3 | 82.1 | 96.4 | 92.0 | 78 | - | $3/$15 |
| Qwen2.5-72B | 86.1 | 91.0 | 91.6 | 86.4 | 35 | 144 | - |
| Qwen2.5-7B | 71.2 | 78.3 | 79.5 | 68.3 | 65 | 14 | - |
| Llama-3.1-70B | 86.0 | 74.2 | 93.0 | 80.5 | 38 | 140 | - |
| Llama-3.1-8B | 69.4 | 61.2 | 77.0 | 62.1 | 90 | 16 | - |
| DeepSeek-V3 | 88.5 | 89.2 | 94.8 | 88.7 | 42 | 135 | $0.27/$1.1 |
| DeepSeek-R1 | 90.8 | 91.5 | 97.3 | 85.2 | 28 | 150 | $0.55/$2.19 |

### 4.2 维度雷达图（文字描述）

```
                准确性
                  │
            90 ──┤  ★ DeepSeek-R1
            80 ──┤  ★ GPT-4o  ★ Qwen2.5-72B
            70 ──┤  ★ Qwen2.5-7B
                  │
   安全性 ────────┼──────── 效率
                  │
            90 ──┤  ★ DeepSeek-V3 (性价比)
            80 ──┤  ★ Qwen2.5-7B (效率)
            70 ──┤  ★ DeepSeek-R1 (效率低)
                  │
                成本
```

### 4.3 模型选型决策矩阵

不同场景下的推荐选择：

| 应用场景 | 推荐模型 | 关键考量 |
|---------|---------|---------|
| 企业知识问答 | Qwen2.5-72B / GPT-4o | 准确性 + 中文能力 |
| 代码辅助开发 | DeepSeek-V3 / Claude 3.5 | HumanEval 得分 + 上下文长度 |
| 数学推理 | DeepSeek-R1 / Qwen2.5-72B | GSM8K + MATH 得分 |
| 低成本高吞吐 | DeepSeek-V3 (API) / Qwen2.5-7B | 性价比 + 推理速度 |
| 端侧部署 | Qwen2.5-7B / Llama-3.1-8B | VRAM 占用 + 推理速度 |
| 安全敏感场景 | GPT-4o / Claude 3.5 | 安全性得分 + 对齐质量 |

### 4.4 性价比分析模型

```python
# cost_effectiveness.py
import json

def calculate_cost_effectiveness(results: dict) -> dict:
    """
    计算模型性价比

    公式: Score = 综合得分 / (成本系数 × 1000)
    其中成本系数 = API价格(美元/百万tokens) 或 自部署等效成本
    """
    scored_models = {}

    for model_name, data in results.items():
        # 综合得分 (加权平均)
        weights = {
            "mmlu": 0.25,
            "c_eval": 0.20,
            "humaneval": 0.25,
            "gsm8k": 0.20,
            "safety": 0.10,
        }
        composite_score = sum(
            data["benchmarks"].get(task, 0) * weight
            for task, weight in weights.items()
        )

        # 成本系数
        cost = data.get("cost_per_million_tokens", 1.0)
        efficiency = data.get("tokens_per_second", 50)

        # 性价比分数
        cost_effectiveness = composite_score / (cost + 0.01)  # 避免除零

        scored_models[model_name] = {
            "composite_score": round(composite_score, 2),
            "cost_per_m_tokens": cost,
            "tokens_per_second": efficiency,
            "cost_effectiveness": round(cost_effectiveness, 2),
            "recommendation": get_recommendation(composite_score, cost, efficiency),
        }

    return scored_models

def get_recommendation(score: float, cost: float, speed: float) -> str:
    if score > 85 and cost > 5:
        return "高质量但昂贵，适合对质量要求极高的场景"
    elif score > 80 and cost < 2:
        return "⭐ 性价比之王，推荐优先考虑"
    elif score > 70 and speed > 80:
        return "速度快，适合高吞吐量场景"
    elif score < 70:
        return "适合轻量级任务和端侧部署"
    return "综合表现均衡"

# 使用示例
results = {
    "DeepSeek-V3": {
        "benchmarks": {"mmlu": 88.5, "c_eval": 89.2, "humaneval": 88.7, "gsm8k": 94.8, "safety": 82.0},
        "cost_per_million_tokens": 1.1,
        "tokens_per_second": 42,
    },
    "Qwen2.5-72B": {
        "benchmarks": {"mmlu": 86.1, "c_eval": 91.0, "humaneval": 86.4, "gsm8k": 91.6, "safety": 80.0},
        "cost_per_million_tokens": 0.8,
        "tokens_per_second": 35,
    },
    "Qwen2.5-7B": {
        "benchmarks": {"mmlu": 71.2, "c_eval": 78.3, "humaneval": 68.3, "gsm8k": 79.5, "safety": 75.0},
        "cost_per_million_tokens": 0.1,
        "tokens_per_second": 65,
    },
}

analysis = calculate_cost_effectiveness(results)
for name, metrics in sorted(analysis.items(), key=lambda x: -x[1]["cost_effectiveness"]):
    print(f"\n{name}:")
    print(f"  综合得分: {metrics['composite_score']}")
    print(f"  性价比: {metrics['cost_effectiveness']}")
    print(f"  建议: {metrics['recommendation']}")
```

---

## 五、评测最佳实践清单

### ✅ 评测前

- [ ] 明确评测目的（选型 / 监控 / 对标）
- [ ] 选择合适的评测维度和数据集
- [ ] 确保评测环境一致（硬件、框架版本、随机种子）
- [ ] 设置对照组（baseline 模型）

### ✅ 评测中

- [ ] 控制 Few-shot 数量（通常 0/5/10）
- [ ] 记录完整的评测配置和环境信息
- [ ] 监控资源使用（GPU、内存、时间）
- [ ] 保存中间结果以便复现

### ✅ 评测后

- [ ] 多维度交叉分析，避免单一指标决策
- [ ] 进行统计显著性检验
- [ ] 人工抽检模型回答质量
- [ ] 形成可复用的评测报告模板
- [ ] 定期更新评测集（防止数据泄漏和过拟合）

---

## 六、总结

LLM 评测不是一锤子买卖，而是一个**持续迭代的工程体系**。核心要点：

1. **多维度覆盖**：准确性、安全性、效率、成本四维缺一不可
2. **工具组合**：OpenCompass（中文强项）+ lm-evaluation-harness（社区标准）
3. **自定义评测**：标准 Benchmark 是起点，业务场景需要定制化评测集
4. **数据驱动决策**：建立性价比模型，用量化指标指导选型

> 评测体系的终极目标不是选出"最好的"模型，而是找到**最适合你场景**的模型。

---

*本文作者：RiceBall-15 | 发布日期：2026-05-30*
