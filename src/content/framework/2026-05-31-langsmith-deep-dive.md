---
title: "LangSmith深度解析：LLM应用的可观测性与评估平台实战指南"
description: "全面剖析LangSmith作为LLM应用开发平台的核心能力，涵盖Tracing、Evaluation、Prompt Hub、Dataset管理等模块的架构设计与生产实践。"
date: 2026-05-31
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["LangSmith", "LLM可观测性", "Tracing", "评估", "Prompt工程", "MLOps"]
draft: false
---

## 引言：为什么LLM应用需要专门的可观测性平台？

传统软件的可观测性建立在确定性之上——同样的输入必然产生同样的输出，Bug可以稳定复现。但LLM应用打破了这个假设：**同样的Prompt可能产生截然不同的输出，同样的输入在不同温度下结果迥异**。

这意味着传统的APM工具（如Datadog、New Relic）在LLM应用面前几乎无能为力。你需要回答这些问题：

- 这次回答质量下降，是Prompt的问题、模型的问题，还是上下文窗口的问题？
- 用户投诉的"胡说八道"，具体是在哪个环节、因为什么触发的？
- 修改Prompt后，整体效果是变好了还是变差了？哪些维度变好了？

**LangSmith**正是为了解决这些问题而生的。作为LangChain团队推出的企业级LLM应用平台，它已经从一个简单的Tracing工具演变为覆盖**开发-测试-部署-监控**全生命周期的完整平台。

本文将从架构设计、核心模块、生产实践三个维度，深入解析LangSmith的技术能力。

---

## 一、LangSmith架构概览

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    LangSmith Platform                 │
├─────────────┬─────────────┬─────────────┬───────────┤
│   Tracing   │ Evaluation  │  Prompt Hub │  Dataset  │
│   (追踪)     │   (评估)     │  (提示管理)  │  (数据集)   │
├─────────────┴─────────────┴─────────────┴───────────┤
│                   Core Engine                        │
├─────────────┬─────────────┬─────────────┬───────────┤
│   Storage   │   Compute   │   Auth &    │  Webhook  │
│   Layer     │   Layer     │   RBAC      │  & API    │
└─────────────┴─────────────┴─────────────┴───────────┘
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │  LangChain│  │  LangGraph│ │  Custom  │
    │  Framework│ │  (Agent) │  │  Code    │
    └──────────┘  └──────────┘  └─────────┘
```

### 1.2 核心设计理念

LangSmith的设计遵循几个关键原则：

| 原则 | 描述 | 实现方式 |
|------|------|---------|
| **可观测性优先** | 每次LLM调用都应可追踪 | OpenTelemetry兼容的Trace系统 |
| **评估驱动开发** | 用量化指标指导迭代 | 自动化评估流水线 |
| **框架无关** | 不绑定特定LLM框架 | Python/JS SDK + REST API |
| **生产就绪** | 满足企业级安全合规 | SOC2、RBAC、审计日志 |

---

## 二、核心模块深度解析

### 2.1 Tracing：全链路追踪

Tracing是LangSmith最核心的能力。它记录LLM应用中每一次调用的完整链路。

#### 2.1.1 数据模型

LangSmith的Trace模型采用分层结构：

```
Project (项目)
  └── Run (运行)
       ├── Run (子运行 - 如LLM调用)
       │    ├── inputs
       │    ├── outputs
       │    ├── metadata
       │    └── feedback (用户反馈)
       ├── Run (子运行 - 如工具调用)
       └── Run (子运行 - 如Chain调用)
```

每个Run包含：

```python
run_data = {
    "id": "run_abc123",
    "name": "ChatOpenAI",          # 运行名称
    "run_type": "llm",             # 类型：llm/chain/tool/retriever
    "inputs": {                     # 输入
        "messages": [...],
        "temperature": 0.7,
    },
    "outputs": {                    # 输出
        "generations": [[{"text": "..."}]],
        "llm_output": {
            "token_usage": {
                "prompt_tokens": 150,
                "completion_tokens": 200,
                "total_tokens": 350,
            },
            "model_name": "gpt-4o",
        }
    },
    "extra": {                      # 额外元数据
        "user_id": "user_123",
        "session_id": "session_456",
    },
    "tags": ["production", "v2.1"], # 标签
    "parent_run_id": "parent_abc",  # 父运行
    "start_time": "2026-05-31T10:00:00Z",
    "end_time": "2026-05-31T10:00:02Z",
    "execution_time_ms": 2000,
}
```

#### 2.1.2 集成方式

LangSmith提供多种集成方式，适配不同的技术栈：

```python
# 方式1：LangChain自动集成（最简单）
import langchain
langchain.tracing = True  # 开启全局追踪
# 所有LangChain调用自动追踪

# 方式2：装饰器方式（适用于自定义代码）
from langsmith import traceable

@traceable(
    name="my_custom_function",
    run_type="chain",
    tags=["production"],
    metadata={"version": "2.1"},
)
def my_llm_function(query: str) -> str:
    # 这个函数的所有输入输出都会被追踪
    response = call_llm(query)
    return response

# 方式3：手动追踪（最大灵活性）
from langsmith import Client

client = Client()

# 创建父运行
run = client.create_run(
    name="qa_pipeline",
    run_type="chain",
    inputs={"question": "What is RAG?"},
)

# 创建子运行
child_run = client.create_run(
    name="retrieval",
    run_type="retriever",
    inputs={"query": "What is RAG?"},
    parent_run_id=run.id,
)

# 更新子运行输出
client.update_run(
    child_run.id,
    outputs={"documents": retrieved_docs},
)

# 更新父运行
client.update_run(
    run.id,
    outputs={"answer": final_answer},
)
```

#### 2.1.3 性能优化

在生产环境中，Tracing本身不应成为性能瓶颈：

```python
# 异步追踪：不影响主流程性能
from langsmith import traceable
import asyncio

@traceable(run_type="chain")
async def async_pipeline(query: str):
    # 异步追踪，不阻塞主流程
    retrieval_result = await async_retrieve(query)
    llm_result = await async_llm_call(retrieval_result)
    return llm_result

# 批量采样：高流量场景下按比例采样
from langsmith import Client

client = Client(
    sample_rate=0.1,  # 只追踪10%的请求
)

# 条件追踪：只追踪特定条件
@traceable(
    # 只在出错时追踪详情
    on_error=lambda e: log_error_details(e),
)
def pipeline_with_conditional_trace(query: str):
    # 正常情况只记录基本信息
    # 出错时记录完整堆栈
    return process(query)
```

### 2.2 Evaluation：自动化评估

评估是LangSmith的第二大核心能力。它让"Prompt修改后效果是否变好"这个问题有了量化答案。

#### 2.2.1 评估体系架构

```
┌─────────────────────────────────────┐
│         Evaluation Pipeline         │
├─────────────┬──────────────┬────────┤
│   Dataset   │   Evaluator  │ Report │
│  (测试数据)   │   (评估器)    │ (报告)  │
├─────────────┼──────────────┼────────┤
│ - 固定用例   │ - LLM-as-   │ - 指标  │
│ - 对抗用例   │   Judge      │ - 对比  │
│ - 边界用例   │ - 自定义函数  │ - 趋势  │
│ - 用户反馈   │ - 人工标注   │ - 回归  │
└─────────────┴──────────────┴────────┘
```

#### 2.2.2 评估器类型

**类型1：LLM-as-Judge（最常用）**

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 定义LLM评估器
def correctness_evaluator(run, example):
    """使用LLM评判输出的正确性"""
    evaluator_prompt = f"""
    判断以下回答是否正确地回答了问题。
    
    问题: {run.inputs['query']}
    回答: {run.outputs['answer']}
    参考答案: {example.outputs['answer']}
    
    请从以下维度评分(1-5):
    1. 事实正确性
    2. 完整性
    3. 简洁性
    
    返回JSON格式: {{"score": <1-5>, "reason": "<解释>"}}
    """
    
    judgment = call_llm(evaluator_prompt)
    return {"key": "correctness", "score": judgment["score"]}

# 运行评估
results = evaluate(
    target=my_llm_function,           # 被评估的函数
    data="test-dataset",              # 测试数据集
    evaluators=[correctness_evaluator], # 评估器
    experiment_prefix="v2.1-prompts",  # 实验前缀
)
```

**类型2：自定义函数评估器**

```python
import re

def regex_pattern_evaluator(run, example):
    """检查输出是否符合特定格式"""
    output = run.outputs.get("answer", "")
    
    # 检查是否包含结构化数据
    has_json = bool(re.search(r'\{.*\}', output))
    has_list = bool(re.search(r'\[.*\]', output))
    
    return {
        "key": "format_compliance",
        "score": 1.0 if (has_json or has_list) else 0.0,
        "comment": "Output contains structured data" if (has_json or has_list) 
                   else "Output lacks structured format"
    }

def length_evaluator(run, example):
    """检查输出长度是否在合理范围"""
    output = run.outputs.get("answer", "")
    word_count = len(output.split())
    
    # 理想长度：50-500词
    if 50 <= word_count <= 500:
        score = 1.0
    elif 20 <= word_count <= 1000:
        score = 0.7
    else:
        score = 0.3
    
    return {"key": "length", "score": score}
```

**类型3：人工标注评估**

```python
# 在LangSmith UI中设置人工评估
# 支持多种标注模式：

annotation_config = {
    "types": [
        {"type": "label", "name": "quality", 
         "options": ["excellent", "good", "fair", "poor"]},
        {"type": "categorical", "name": "issue_type",
         "options": ["factual_error", "hallucination", "incomplete", "irrelevant"]},
        {"type": "freeform", "name": "improvement_suggestion"},
    ],
    " reviewers": ["team_member_1", "team_member_2"],
    " consensus_required": True,  # 需要多人一致
}
```

#### 2.2.3 A/B测试与对比评估

```python
from langsmith import Client
from langsmith.evaluation import evaluate_comparative

client = Client()

# 对比两个版本的Prompt
results = evaluate_comparative(
    experiments=["v2.0-experiment", "v2.1-experiment"],
    evaluators=[
        # 哪个版本更好？
        lambda runs, example: {
            "key": "preference",
            "choices": ["A", "B"],
            "comment": "Which response is better?"
        }
    ],
)

# 生成对比报告
# 输出类似：
# v2.1 vs v2.0: 68% of cases improved, 12% degraded, 20% unchanged
# Most improved areas: [accuracy, conciseness]
# Regression areas: [creativity in 3% of cases]
```

### 2.3 Prompt Hub：版本化Prompt管理

Prompt管理是LangSmith的第三大能力，解决了"Prompt改了之后不知道改了什么"的痛点。

#### 2.3.1 Prompt版本管理

```python
from langsmith import Client

client = Client()

# 创建Prompt（自动版本化）
prompt = client.push_prompt(
    "qa-assistant",
    object={
        "lc": 1,
        "type": "prompt",
        "kwargs": {
            "template": "你是一个专业的问答助手。\n\n问题：{question}\n\n请基于以下上下文回答：\n{context}",
            "input_variables": ["question", "context"],
        }
    },
    tags=["production", "v2"],
    description="生产环境问答助手Prompt v2",
)

# 获取指定版本
prompt_v1 = client.pull_prompt("qa-assistant:v1")
prompt_v2 = client.pull_prompt("qa-assistant:v2")

# 获取最新版本
latest_prompt = client.pull_prompt("qa-assistant:latest")

# 在代码中使用
@traceable
def qa_pipeline(question: str, context: str):
    prompt = client.pull_prompt("qa-assistant:production")
    formatted = prompt.format(question=question, context=context)
    return call_llm(formatted)
```

#### 2.3.2 Prompt与评估联动

这是LangSmith最强大的工作流之一——修改Prompt后自动触发评估：

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

def run_prompt_experiment(prompt_version: str):
    """运行Prompt实验"""
    prompt = client.pull_prompt(f"qa-assistant:{prompt_version}")
    
    def target(inputs):
        formatted = prompt.format(**inputs)
        return call_llm(formatted)
    
    results = evaluate(
        target=target,
        data="qa-test-set",
        evaluators=[correctness_evaluator, length_evaluator],
        experiment_prefix=f"prompt-{prompt_version}",
    )
    
    return results

# 运行对比实验
v2_results = run_prompt_experiment("v2")
v3_results = run_prompt_experiment("v3")

# 自动对比结果
# v3 correctness: 4.2/5 (↑0.3 vs v2)
# v3 avg_length: 320 words (↓45 vs v2)
# v3 cost: $0.023/query (↓$0.005 vs v2)
```

### 2.4 Dataset管理

Dataset是连接开发和评估的桥梁。

```python
from langsmith import Client

client = Client()

# 创建数据集
dataset = client.create_dataset(
    name="qa-test-set",
    description="问答系统测试数据集",
    data_type="kv",  # key-value格式
)

# 添加测试用例
client.create_examples(
    dataset_id=dataset.id,
    examples=[
        {
            "inputs": {
                "question": "什么是RAG？",
                "context": "RAG是检索增强生成的缩写..."
            },
            "outputs": {
                "answer": "RAG（Retrieval-Augmented Generation）是..."
            },
            "metadata": {"difficulty": "easy", "topic": "definition"},
        },
        # ...更多测试用例
    ],
)

# 从生产日志自动生成数据集
# 筛选高反馈的交互作为测试用例
production_runs = client.list_runs(
    project_name="qa-production",
    filter="feedback.ne.y > 3",  # 用户评价>3分
    limit=100,
)

client.create_examples_from_runs(
    dataset_id=dataset.id,
    runs=production_runs,
)
```

---

## 三、生产环境部署实战

### 3.1 部署架构设计

```
┌──────────────────────────────────────────────────┐
│                 应用层 (Application)               │
├──────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  API     │  │  Worker  │  │  Admin   │      │
│  │  Server  │  │  (异步)   │  │  Panel   │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │              │              │            │
├───────▼──────────────▼──────────────▼────────────┤
│              LangSmith SDK (采样+批量)             │
├──────────────────────────────────────────────────┤
│                   网络层                          │
├──────────────────────────────────────────────────┤
│              LangSmith Cloud / Self-hosted        │
└──────────────────────────────────────────────────┘
```

### 3.2 生产环境配置

```python
# config.py - 生产环境配置
import os
from langsmith import Client

class LangSmithConfig:
    """生产环境LangSmith配置"""
    
    # 基础配置
    API_KEY = os.getenv("LANGSMITH_API_KEY")
    PROJECT = os.getenv("LANGSMITH_PROJECT", "production")
    ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    
    # 采样配置（生产环境必须）
    SAMPLE_RATE = float(os.getenv("LANGSMITH_SAMPLE_RATE", "0.1"))  # 10%采样
    ERROR_SAMPLE_RATE = 1.0  # 错误100%采样
    
    # 批量上传配置（减少API调用）
    BATCH_SIZE = 20
    FLUSH_INTERVAL = 5.0  # 5秒刷新一次
    
    # 评估配置
    EVAL_SAMPLE_RATE = 0.05  # 5%的请求自动评估
    
    @classmethod
    def get_client(cls):
        return Client(
            api_key=cls.API_KEY,
            endpoint=cls.ENDPOINT,
            sample_rate=cls.SAMPLE_RATE,
        )

# middleware.py - 请求追踪中间件
from fastapi import Request
from langsmith import traceable
import time

@traceable(run_type="chain")
async def trace_middleware(request: Request, call_next):
    """FastAPI追踪中间件"""
    start_time = time.time()
    
    # 添加元数据
    metadata = {
        "method": request.method,
        "path": request.url.path,
        "user_agent": request.headers.get("user-agent", "unknown"),
    }
    
    try:
        response = await call_next(request)
        metadata["status_code"] = response.status_code
        return response
    except Exception as e:
        metadata["error"] = str(e)
        metadata["status_code"] = 500
        raise
    finally:
        metadata["response_time_ms"] = (time.time() - start_time) * 1000
```

### 3.3 监控告警配置

```python
# alerting.py - 基于LangSmith数据的告警
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

def check_quality_degradation():
    """检查质量退化"""
    # 获取最近1小时的数据
    recent_runs = client.list_runs(
        project_name="production",
        start_time=datetime.utcnow() - timedelta(hours=1),
        run_type="llm",
    )
    
    # 计算平均反馈分数
    scores = []
    for run in recent_runs:
        if run.feedback:
            scores.extend([f.score for f in run.feedback])
    
    if not scores:
        return None
    
    avg_score = sum(scores) / len(scores)
    
    # 告警阈值
    if avg_score < 3.0:
        send_alert(
            level="WARNING",
            message=f"质量分数下降：最近1小时平均分 {avg_score:.2f}/5.0",
            details={
                "sample_size": len(scores),
                "period": "1h",
                "threshold": 3.0,
            }
        )
    
    return avg_score

def check_cost_anomaly():
    """检查成本异常"""
    recent_runs = client.list_runs(
        project_name="production",
        start_time=datetime.utcnow() - timedelta(hours=1),
        run_type="llm",
    )
    
    total_tokens = sum(
        run.total_tokens or 0 for run in recent_runs
    )
    
    # 正常情况下每小时约100万token
    if total_tokens > 5_000_000:  # 超过500万token
        send_alert(
            level="WARNING",
            message=f"Token使用量异常：最近1小时 {total_tokens:,} tokens",
        )
```

### 3.4 CI/CD集成

```yaml
# .github/workflows/langsmith-eval.yml
name: LangSmith Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install langsmith langchain
      
      - name: Run evaluation
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
        run: |
          python -m evaluation.run_eval \
            --dataset "ci-test-set" \
            --threshold correctness=4.0 \
            --fail-on-regression
      
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            # 将评估结果写入PR评论
            const results = require('./eval_results.json');
            const body = `## LangSmith Evaluation Results
            | Metric | Score | Threshold | Status |
            |--------|-------|-----------|--------|
            | Correctness | ${results.correctness} | 4.0 | ${results.correctness >= 4.0 ? '✅' : '❌'} |
            | Conciseness | ${results.conciseness} | 3.5 | ${results.conciseness >= 3.5 ? '✅' : '❌'} |
            `;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
```

---

## 四、最佳实践总结

### 4.1 Tracing最佳实践

1. **分层追踪**：Chain → Retriever → LLM → Tool，保持清晰的层级关系
2. **添加元数据**：user_id、session_id、version等信息，便于后续筛选分析
3. **合理采样**：生产环境10%采样，错误100%采样
4. **异步上传**：使用批量上传和异步API，避免影响主流程性能

### 4.2 评估最佳实践

1. **建立基线**：首次部署时建立性能基线，后续每次修改都与基线对比
2. **多维度评估**：不要只看准确性，还要看延迟、成本、安全性
3. **持续积累数据集**：将生产中的高质量交互加入测试集
4. **自动化评估流水线**：将评估集成到CI/CD，每次修改自动触发

### 4.3 Prompt管理最佳实践

1. **版本化一切**：每个Prompt变更都应该有版本号和变更说明
2. **与评估联动**：修改Prompt后必须运行评估对比
3. **保持简洁**：Prompt越简单越容易维护和调试
4. **文档化设计意图**：记录每个Prompt的设计思路和权衡

---

## 结语

LangSmith代表了LLM应用工程化的一个重要方向：**用软件工程的方法论来管理AI应用的不确定性**。通过Tracing让每次调用可追溯，通过Evaluation让每次修改可量化，通过Prompt Hub让每次迭代可复现。

对于正在构建LLM应用的团队来说，LangSmith不仅是一个工具，更是一种工程文化的载体——它推动团队从"凭感觉调Prompt"走向"数据驱动的AI工程"。

无论你是否选择LangSmith作为最终方案，它所代表的理念——可观测性、评估驱动、版本管理——都应该成为每个LLM应用团队的基础设施。

---

## 参考资料

1. LangSmith Documentation: https://docs.smith.langchain.com/
2. LangChain GitHub: https://github.com/langchain-ai/langchain
3. LangSmith API Reference: https://api.smith.langchain.com/
4. "Evaluating LLMs is a minefield" - LangChain Blog, 2025
5. "Production-ready LLM applications with LangSmith" - LangChain Webinar, 2026
