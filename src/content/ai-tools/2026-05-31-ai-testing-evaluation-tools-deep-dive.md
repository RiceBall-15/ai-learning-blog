---
title: "AI测试与评估工具深度评测：从单元测试到端到端评估，2026年全链路选型指南"
description: "深度评测DeepEval、Ragas、Promptfoo、Braintrust等主流AI测试框架，覆盖单元测试、集成测试、评估基准与生产监控全链路"
date: 2026-05-31
author: "RiceBall"
category: "ai-tools"
subCategory: coding-tools
tags: ["AI测试", "LLM评估", "DeepEval", "Ragas", "Promptfoo", "AI质量保证"]
draft: false
---

# AI测试与评估工具深度评测：从单元测试到端到端评估，2026年全链路选型指南

## 引言

传统软件测试的核心假设是**确定性**：给定相同输入，程序永远返回相同输出。但LLM应用打破了这个假设——同一段Prompt可能产生截然不同的结果，"正确答案"本身也可能是模糊的。

这意味着我们需要全新的测试范式。传统断言（`assertEqual`）不再适用，取而代之的是**评估（Evaluation）**——用概率性、语义级的方式衡量输出质量。

2026年，AI测试工具生态已经从萌芽走向成熟。本文深度评测当前主流的6款工具，覆盖从开发期单元测试到生产环境持续监控的全链路。

## 一、AI测试的层次模型

在深入工具评测之前，我们需要先理解AI测试的层次结构：

```
┌─────────────────────────────────────────────┐
│            AI测试金字塔 (2026)               │
│                                             │
│              ┌─────────┐                    │
│              │ E2E评估  │  ← 端到端质量验证  │
│             ┌┴─────────┴┐                   │
│             │  集成测试   │  ← 组件协作验证   │
│            ┌┴───────────┴┐                  │
│            │  组件测试     │  ← 单模块验证    │
│           ┌┴─────────────┴┐                 │
│           │  Prompt测试    │  ← Prompt质量    │
│          ┌┴───────────────┴┐                │
│          │   单元测试        │  ← 工具/函数   │
│          └─────────────────┘                │
└─────────────────────────────────────────────┘
```

| 测试层次 | 测试对象 | 评估指标 | 典型工具 |
|---------|---------|---------|---------|
| **单元测试** | 工具函数、数据解析 | 准确性、边界情况 | pytest + 自定义断言 |
| **Prompt测试** | Prompt模板、Few-shot | 相似度、一致性 | DeepEval, Promptfoo |
| **组件测试** | RAG管道、Agent节点 | 检索质量、路由准确率 | Ragas, DeepEval |
| **集成测试** | 多组件协作 | 端到端成功率 | Promptfoo, Braintrust |
| **端到端评估** | 完整应用 | 用户满意度、安全性 | Braintrust, 自定义评估 |

## 二、工具深度评测

### 2.1 DeepEval：Python生态的全能选手

**定位：** 开源的LLM评估框架，提供类pytest的测试体验

**GitHub Stars：** 10k+（增长迅猛）

#### 核心特性

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    GEval,
)

# 定义测试用例
test_case = LLMTestCase(
    input="什么是RAG？",
    actual_output="RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术...",
    retrieval_context=["RAG文档片段1", "RAG文档片段2"],
    context=["RAG的定义和原理"],
    expected_output="RAG是一种结合检索和生成的技术..."
)

# 使用预置指标
relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.8)

# 执行测试
assert_test(test_case, [relevancy, faithfulness])
```

#### 评估指标矩阵

DeepEval提供了最丰富的评估指标体系：

| 指标类别 | 指标名称 | 评估内容 | 适用场景 |
|---------|---------|---------|---------|
| **生成质量** | AnswerRelevancy | 回答与问题的相关性 | 所有问答场景 |
| | AnswerRelevancy | 回答的信息密度 | 知识密集型应用 |
| | GEval | 自定义LLM评估标准 | 特定业务需求 |
| **忠实性** | Faithfulness | 输出是否忠实于上下文 | RAG系统 |
| | Hallucination | 是否产生幻觉 | 所有LLM应用 |
| **安全性** | Toxicity | 输出是否包含有害内容 | 面向用户的应用 |
| | Bias | 输出是否存在偏见 | 公平性敏感场景 |
| **连贯性** | Coherence | 多轮对话的逻辑连贯性 | 对话系统 |
| **检索质量** | ContextualPrecision | 检索结果的排序质量 | RAG系统 |
| | ContextualRecall | 检索结果的覆盖度 | RAG系统 |
| | ContextualRelevancy | 检索结果与问题的相关性 | RAG系统 |

#### DeepEval的独特优势

**① 原生Pytest集成**

```bash
# 直接用pytest运行LLM测试
deepeval test run test_llm.py

# 生成评估报告
deepeval test run test_llm.py --report-type html
```

这意味着你可以直接复用现有的测试基础设施：CI/CD集成、测试报告生成、并行执行等。

**② 自定义评估指标**

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# 使用LLM作为评估器
custom_metric = GEval(
    name="专业性",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    evaluation_prompt="""
    评估AI助手的回答是否具有专业性：
    - 是否使用了准确的技术术语？
    - 解释是否清晰且深入？
    - 是否避免了过度简化？
    评分从1到5，5分最高。
    """,
    threshold=4.0,
)
```

**③ 多模态评估**

```python
from deepeval.metrics import MultimodalFaithfulness

# 支持图像+文本的评估
image_test_case = LLMTestCase(
    input="描述这张图片中的内容",
    actual_output="图片展示了一个城市天际线...",
    actual_image_urls=["https://example.com/image.jpg"],
)

metric = MultimodalFaithfulness(threshold=0.8)
```

#### 不足之处

- **依赖外部LLM做评估**：评估本身需要调用LLM，成本较高
- **文档质量参差不齐**：部分新指标的文档不够完善
- **无内置UI**：评估结果需要额外工具可视化

---

### 2.2 Ragas：RAG系统的专属评估框架

**定位：** 专注于RAG系统评估的开源框架

**GitHub Stars：** 7k+

#### 核心理念

Ragas的核心洞察是：**RAG系统的评估应该同时考虑检索质量和生成质量**。它围绕这个理念设计了一套完整的评估指标：

```
┌─────────────────────────────────────────┐
│           Ragas 评估模型                │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         检索质量评估             │   │
│  │  ┌───────────┐ ┌───────────┐   │   │
│  │  │ 精确率     │ │ 召回率     │   │   │
│  │  │ (Precision)│ │ (Recall)  │   │   │
│  │  └───────────┘ └───────────┘   │   │
│  │  ┌───────────┐ ┌───────────┐   │   │
│  │  │ 上下文相关 │ │ 上下文F1   │   │   │
│  │  │ (Relevancy)│ │ (F1 Score)│   │   │
│  │  └───────────┘ └───────────┘   │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         生成质量评估             │   │
│  │  ┌───────────┐ ┌───────────┐   │   │
│  │  │ 忠实度     │ │ 回答相关性 │   │   │
│  │  │(Faithful.) │ │(Answer    │   │   │
│  │  │           │ │ Relevancy)│   │   │
│  │  └───────────┘ └───────────┘   │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │        综合指标                  │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │       RAGAS Score        │   │   │
│  │  │  (所有指标的加权平均)    │   │   │
│  │  └─────────────────────────┘   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 实战代码

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": [
        "如何配置RAG系统的分块策略？",
        "向量数据库选型应该考虑哪些因素？",
    ],
    "answer": [
        "RAG系统的分块策略需要考虑...",
        "向量数据库选型主要考虑...",
    ],
    "contexts": [
        ["分块策略文档片段1", "分块策略文档片段2"],
        ["向量数据库对比文档1", "向量数据库对比文档2"],
    ],
    "ground_truth": [
        "推荐使用递归字符分割，chunk_size=512...",
        "考虑性能、成本、生态、易用性四个维度...",
    ],
}

dataset = Dataset.from_dict(eval_data)

# 运行评估
results = evaluate(
    dataset,
    metrics=[
        faithfulness,         # 忠实度
        answer_relevancy,     # 回答相关性
        context_precision,    # 上下文精确率
        context_recall,       # 上下文召回率
        context_relevancy,    # 上下文相关性
    ],
)

print(results)
# 输出示例:
# {'faithfulness': 0.85, 'answer_relevancy': 0.78,
#  'context_precision': 0.92, 'context_recall': 0.81,
#  'context_relevancy': 0.75}
```

#### Ragas的独特优势

**① 标准化的RAG评估数据集格式**

```python
# Ragas定义了标准的评估数据格式
# 每个样本必须包含：question, answer, contexts
# 可选包含：ground_truth (用于无参考评估)

# 支持从多种来源加载数据
from ragas.dataset_schema import EvaluationDataset

# 从CSV加载
dataset = EvaluationDataset.from_csv("eval_data.csv")

# 从JSON加载
dataset = EvaluationDataset.from_json("eval_data.json")

# 从LangChain文档加载
dataset = EvaluationDataset.from_langchain_documents(docs)
```

**② 自带数据集生成工具**

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper

# 使用LLM自动生成评估数据
generator = TestsetGenerator(
    llm=LangchainLLMWrapper(your_llm),
    # 从文档自动生成问答对
)

testset = generator.generate_with_langchain_docs(
    documents=your_documents,
    testset_size=50,  # 生成50个测试样本
)

# 自动生成的测试集包含：
# - 简单问题
# - 多跳推理问题
# - 条件推理问题
# - 假设性问题
```

**③ 集成度高**

```python
# 与LangChain无缝集成
from ragas.integrations.langchain import evaluate as ragas_evaluate

# 直接评估LangChain链
results = ragas_evaluate(
    chain=your_ragas_chain,
    metrics=[faithfulness, answer_relevancy],
    eval_dataset=eval_dataset,
)

# 与LlamaIndex集成
from ragas.integrations.llamaindex import evaluate as ragas_llamaindex_evaluate
```

#### 不足之处

- **指标计算依赖LLM**：faithfulness等指标需要调用LLM，成本和延迟较高
- **对非RAG场景支持有限**：主要面向RAG系统
- **ground_truth依赖**：部分指标需要参考答案
- **无实时监控能力**：只支持离线评估

---

### 2.3 Promptfoo：Prompt测试的瑞士军刀

**定位：** 开源的Prompt测试和红队工具

**GitHub Stars：** 8k+

#### 核心特性

Promptfoo的独特之处在于它**同时支持Prompt测试和红队攻击测试**：

```yaml
# promptfooconfig.yaml
description: "RAG应用Prompt测试"

prompts:
  - "根据以下上下文回答问题：\n\n上下文：{{context}}\n\n问题：{{query}}"
  - "你是一个专业助手。基于提供的信息回答：\n\n{{context}}\n\n{{query}}"

providers:
  - openai:gpt-4o
  - openai:gpt-4o-mini
  - anthropic:claude-3.5-sonnet

tests:
  - vars:
      context: "RAG是检索增强生成技术"
      query: "什么是RAG？"
    assert:
      - type: contains
        value: "检索增强生成"
      - type: llm-rubric
        value: "回答是否准确描述了RAG技术"
      - type: cost
        threshold: 0.01  # 成本上限

  - vars:
      context: "Python是一种编程语言"
      query: "如何烹饪意大利面？"
    assert:
      - type: contains-any
        value: ["无法回答", "不在我的知识范围内", "不相关"]
```

#### 评估维度

Promptfoo提供了丰富的断言类型：

| 断言类型 | 功能 | 示例 |
|---------|------|------|
| `contains` | 包含指定文本 | 回答中包含关键术语 |
| `llm-rubric` | LLM评分 | 1-5分质量评估 |
| `fact-check` | 事实核查 | 与参考答案对比 |
| `cost` | 成本控制 | 单次调用成本上限 |
| `latency` | 延迟控制 | 响应时间上限 |
| `is-json` | 格式验证 | 输出是否为合法JSON |
| `javascript` | 自定义断言 | 运行任意JS代码 |
| `similar` | 语义相似度 | 与参考答案的相似度 |

#### 红队测试能力

```yaml
# 红队测试配置
redteam:
  purpose: "测试RAG应用的安全性"
  numTests: 20
  
  plugins:
    - harmful-content       # 有害内容生成
    - jailbreak             # 越狱攻击
    - prompt-injection     # Prompt注入
    - hallucination        # 幻觉诱导
  
  strategies:
    - base64               # Base64编码绕过
    - leetspeak            # L33t speak绕过
    - rot13                # ROT13编码绕过
```

```bash
# 运行红队测试
npx promptfoo redteam

# 生成可视化报告
npx promptfoo view
```

#### Promptfoo的独特优势

**① 多模型并行对比**

```yaml
# 同时测试多个模型
providers:
  - openai:gpt-4o
  - openai:gpt-4o-mini
  - anthropic:claude-3.5-sonnet
  - google:gemini-2.0-flash
  - local:ollama:llama3.1

# 自动对比各模型在相同测试用例上的表现
```

**② CI/CD原生支持**

```yaml
# .github/workflows/llm-tests.yml
name: LLM Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Promptfoo Tests
        run: npx promptfoo eval
      - name: Check Thresholds
        run: npx promptfoo eval --grader openai:gpt-4o-mini --threshold 0.8
```

**③ 可视化报告**

Promptfoo提供了一个Web UI，可以直观对比不同Prompt、不同模型的测试结果：

```bash
# 启动Web UI
npx promptfoo view

# 输出评估矩阵视图
# ┌──────────┬──────┬──────┬──────┐
# │ Prompt   │ GPT4o│ Sonn.│ Flash│
# ├──────────┼──────┼──────┼──────┤
# │ 测试1    │ 0.95 │ 0.92 │ 0.88 │
# │ 测试2    │ 0.87 │ 0.91 │ 0.85 │
# │ 测试3    │ 0.93 │ 0.89 │ 0.90 │
# └──────────┴──────┴──────┴──────┘
```

#### 不足之处

- **基于Node.js生态**：Python项目集成需要额外适配
- **评估深度有限**：不如DeepEval的指标体系丰富
- **无内置数据集管理**：需要自行管理评估数据

---

### 2.4 Braintrust：生产级AI评估平台

**定位：** 企业级AI评估和实验管理平台

**GitHub Stars：** 2k+

#### 核心理念

Braintrust的定位与上述工具不同——它不是一个纯测试框架，而是一个**完整的AI评估平台**，包含数据管理、评估执行、实验追踪和生产监控。

```python
import braintrust

# 初始化
braintrust.init(api_key="your-key", project="rag-app")

# 记录评估实验
experiment = braintrust.Experiment(
    name="RAG-v2-evaluation",
    dataset=eval_dataset,
    task=my_rag_function,  # 被测试的函数
)

# 运行评估
results = experiment.run()

# 查看结果
print(f"平均忠实度: {results.avg_score('faithfulness')}")
print(f"最佳Prompt: {results.best_prompt()}")
```

#### Braintrust的独特优势

**① 实验管理**

```python
# 追踪多次实验的对比
experiment1 = braintrust.Experiment(name="baseline")
experiment2 = braintrust.Experiment(name="with-reranker")
experiment3 = braintrust.Experiment(name="with-reranker-v2")

# Web UI中自动对比三个版本的性能
```

**② 生产数据回流**

```python
# 在生产环境中记录数据
@braintrust.traced
def production_rag(query: str) -> str:
    result = rag_pipeline(query)
    return result

# 定期将生产数据采样回评估数据集
# 在Web UI中可以一键将生产失败case加入评估集
```

**③ 人工标注支持**

```python
# 集成人工作为评估的一部分
results = experiment.run(
    human_review=True,  # 需要人工审核
)

# 在Web UI中，标注者可以：
# - 评分（1-5分）
# - 选择最佳输出
# - 添加注释
```

#### 不足之处

- **非完全开源**：核心平台为闭源
- **需要云服务依赖**：本地评估能力有限
- **定价较高**：企业级功能需要付费

---

### 2.5 其他值得关注的工具

#### Evidently AI（开源 ML 监控）

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 主要用于数据漂移检测
# 适合监控生产环境输入数据质量
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=prod_data)
```

#### LangSmith（LangChain官方）

```python
from langsmith import Client

client = Client()
# 与LangChain深度集成
# 支持在线评估、数据集管理
# 主要面向LangChain用户
```

## 三、全链路选型指南

### 3.1 工具对比矩阵

| 维度 | DeepEval | Ragas | Promptfoo | Braintrust |
|------|---------|-------|-----------|------------|
| **核心定位** | LLM评估框架 | RAG评估框架 | Prompt测试+红队 | 企业评估平台 |
| **开源** | ✅ 完全开源 | ✅ 完全开源 | ✅ 完全开源 | ⚠️ 部分开源 |
| **语言** | Python | Python | Node.js/Python | Python |
| **Pytest集成** | ✅ | ❌ | ❌ | ❌ |
| **指标丰富度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **红队能力** | ❌ | ❌ | ✅ 专业级 | ❌ |
| **实验管理** | ⚠️ 基础 | ❌ | ⚠️ 基础 | ✅ 专业级 |
| **生产监控** | ❌ | ❌ | ❌ | ✅ |
| **可视化报告** | ⚠️ 需额外工具 | ⚠️ 需额外工具 | ✅ 内置Web UI | ✅ 内置Web UI |
| **多模型支持** | ✅ | ✅ | ✅ | ✅ |
| **学习曲线** | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐⭐ 中高 |
| **社区活跃度** | 🟢 高 | 🟢 高 | 🟢 高 | 🟡 中 |

### 3.2 场景化推荐

#### 场景1：RAG系统开发团队

```
推荐组合：Ragas + DeepEval

理由：
- Ragas 提供专业的RAG评估指标（检索质量+生成质量）
- DeepEval 补充通用LLM评估能力（安全性、幻觉检测）
- 两者都是Python原生，集成成本低

工作流：
1. 开发阶段：用Ragas评估RAG管道的检索和生成质量
2. 集成测试：用DeepEval的指标做全面质量验证
3. CI/CD：DeepEval的pytest集成直接嵌入流水线
```

#### 场景2：Prompt工程师

```
推荐组合：Promptfoo + DeepEval

理由：
- Promptfoo 提供Prompt A/B测试和可视化对比
- Promptfoo 的红队能力测试Prompt安全性
- DeepEval 的GEval支持自定义评估维度

工作流：
1. Prompt迭代：用Promptfoo对比不同Prompt版本
2. 安全测试：用Promptfoo红队检测Prompt注入风险
3. 质量验证：用DeepEval的GEval做多维度评估
```

#### 场景3：企业级AI平台团队

```
推荐组合：Braintrust + DeepEval + 自定义监控

理由：
- Braintrust 提供实验管理、人工标注、生产监控
- DeepEval 补充深度评估指标
- 自定义监控覆盖特定业务指标

工作流：
1. 实验阶段：Braintrust管理多版本实验
2. 评估阶段：DeepEval提供细粒度评估
3. 上线后：Braintrust生产监控 + 自定义业务指标
```

### 3.3 快速上手路线图

```
第1周：基础建设
├── 安装DeepEval + Ragas（Python项目）
├── 或安装Promptfoo（Node.js项目）
├── 收集20-50个测试样本
└── 运行首次评估，建立baseline

第2周：CI/CD集成
├── 编写评估脚本
├── 集成到GitHub Actions/GitLab CI
├── 设置通过阈值
└── 首次自动评估流水线运行

第3周：持续优化
├── 分析失败case，补充测试样本
├── 调整评估指标权重
├── 引入更多评估维度
└── 建立评估报告机制

第4周：生产监控
├── 采样生产请求加入评估集
├── 设置漂移检测告警
├── 建立定期评估报告
└── 人工标注反馈循环
```

## 四、实战：构建完整的AI测试流水线

### 4.1 项目结构

```
ai-test-pipeline/
├── tests/
│   ├── test_rag_unit.py        # RAG单元测试
│   ├── test_prompt_quality.py  # Prompt质量测试
│   ├── test_safety.py          # 安全性测试
│   └── test_e2e.py             # 端到端测试
├── eval_data/
│   ├── rag_eval_dataset.json   # RAG评估数据集
│   └── safety_test_cases.json  # 安全测试用例
├── src/
│   └── rag_pipeline.py         # 被测试的RAG管道
├── promptfooconfig.yaml         # Promptfoo配置
├── pytest.ini                   # pytest配置
└── .github/
    └── workflows/
        └── llm-tests.yml       # CI/CD流水线
```

### 4.2 核心测试脚本

```python
# tests/test_rag_pipeline.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
)
from ragas import evaluate
from ragas.metrics import faithfulness as ragas_faithfulness
from datasets import Dataset
from src.rag_pipeline import rag_query

# ============ RAG单元测试 ============

class TestRAGPipeline:
    """RAG管道单元测试"""
    
    @pytest.fixture
    def sample_queries(self):
        """测试查询集"""
        return [
            {
                "query": "如何优化RAG系统的检索质量？",
                "expected_keywords": ["分块", "embedding", "重排序"],
            },
            {
                "query": "向量数据库有哪些主流选择？",
                "expected_keywords": ["Pinecone", "Milvus", "Weaviate"],
            },
        ]
    
    def test_answer_relevancy(self):
        """测试回答与问题的相关性"""
        result = rag_query("什么是RAG？")
        
        test_case = LLMTestCase(
            input="什么是RAG？",
            actual_output=result["answer"],
            retrieval_context=result["contexts"],
        )
        
        metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_faithfulness(self):
        """测试回答的忠实度（不产生幻觉）"""
        result = rag_query("RAG的最佳实践有哪些？")
        
        test_case = LLMTestCase(
            input="RAG的最佳实践有哪些？",
            actual_output=result["answer"],
            retrieval_context=result["contexts"],
            context=["RAG最佳实践文档"],
        )
        
        metric = FaithfulnessMetric(threshold=0.8)
        assert_test(test_case, [metric])


# ============ Ragas评估 ============

class TestRAGEvaluation:
    """使用Ragas进行系统级评估"""
    
    @pytest.fixture
    def eval_dataset(self):
        """从文件加载评估数据集"""
        with open("eval_data/rag_eval_dataset.json") as f:
            data = json.load(f)
        
        # 运行RAG获取实际输出
        results = []
        for item in data["samples"]:
            result = rag_query(item["question"])
            results.append({
                "question": item["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item["ground_truth"],
            })
        
        return Dataset.from_dict({
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        })
    
    def test_rag_overall_quality(self, eval_dataset):
        """评估RAG系统整体质量"""
        results = evaluate(
            eval_dataset,
            metrics=[
                ragas_faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        
        # 设置质量门限
        assert results["faithfulness"] >= 0.8, \
            f"忠实度不足: {results['faithfulness']}"
        assert results["answer_relevancy"] >= 0.7, \
            f"回答相关性不足: {results['answer_relevancy']}"
        assert results["context_precision"] >= 0.75, \
            f"上下文精确率不足: {results['context_precision']}"
```

### 4.3 CI/CD集成

```yaml
# .github/workflows/llm-tests.yml
name: AI Quality Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  llm-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install deepeval ragas pytest
      
      - name: Run DeepEval Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          deepeval test run tests/ --report-type json
      
      - name: Run Ragas Evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/test_rag_pipeline.py::TestRAGEvaluation -v
      
      - name: Upload Test Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: llm-test-report
          path: reports/
```

## 五、评估指标的选择策略

### 5.1 指标选择决策树

```
你的应用场景是什么？
    │
    ├── 问答系统 / 知识助手
    │   ├── 必选：AnswerRelevancy, Faithfulness
    │   ├── 推荐：Hallucination, ContextualRecall
    │   └── 可选：Toxicity, Bias
    │
    ├── RAG系统
    │   ├── 必选：Faithfulness, ContextualPrecision, ContextualRecall
    │   ├── 推荐：AnswerRelevancy, ContextualRelevancy
    │   └── 可选：RAGAS Score（综合指标）
    │
    ├── Agent系统
    │   ├── 必选：ToolCallAccuracy, TaskCompletion
    │   ├── 推荐：Coherence, ReasoningQuality
    │   └── 可选：Latency, Cost
    │
    └── 内容生成
        ├── 必选：GEval(原创性), GEval(准确性)
        ├── 推荐：Toxicity, Bias
        └── 可选：Readability, Engagement
```

### 5.2 阈值设定经验

| 指标 | 初期阈值 | 成熟期阈值 | 生产阈值 | 说明 |
|------|---------|-----------|---------|------|
| Faithfulness | 0.7 | 0.85 | 0.9 | 忠实度是底线 |
| AnswerRelevancy | 0.6 | 0.75 | 0.8 | 相关性影响用户体验 |
| Hallucination | 0.3 | 0.15 | 0.1 | 越低越好（反向指标） |
| Toxicity | 0.1 | 0.05 | 0.01 | 安全性零容忍 |
| ContextualPrecision | 0.6 | 0.75 | 0.8 | 检索质量 |

## 结语

AI测试不是可选的——它是将LLM应用从"Demo"推向"产品"的关键一步。

**核心建议：**

1. **尽早建立评估体系**：从第一个Prompt开始就记录评估数据
2. **不要只依赖单一指标**：多维度评估才能全面把握质量
3. **持续迭代评估集**：将生产中的失败case回流到评估集
4. **平衡自动化和人工**：LLM评估做初筛，人工评估做终审

2026年的AI测试工具已经足够成熟，足以支撑企业级应用的质量保证需求。选择合适的工具组合，建立系统化的评估流程，才是AI质量工程的核心竞争力。

---

**参考资源：**
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
- [Ragas Documentation](https://docs.ragas.io/)
- [Promptfoo Documentation](https://www.promptfoo.dev/docs/)
- [Braintrust AI](https://www.braintrust.dev/)
- [Evidently AI](https://www.evidentlyai.com/)
