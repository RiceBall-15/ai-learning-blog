---
title: "AI系统可观测性架构设计：从LLM调用到业务价值的全链路追踪"
description: "深入剖析AI应用可观测性体系设计，涵盖Trace/Log/Metrics三支柱、LLM调用链追踪、Prompt版本管理、成本归因与业务价值量化，提供生产级落地方案。"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: distributed
tags: ["可观测性", "LLM监控", "分布式追踪", "AI运维", "成本归因", "Prometheus", "OpenTelemetry"]
draft: false
---

# AI系统可观测性架构设计：从LLM调用到业务价值的全链路追踪

## 一、为什么AI系统需要全新的可观测性？

传统的应用可观测性围绕 **请求→服务→响应** 的链路展开，核心指标是延迟、吞吐量和错误率。但AI应用引入了根本性的变化：

| 维度 | 传统Web应用 | AI应用 |
|------|-----------|--------|
| **核心资源** | CPU、内存、带宽 | GPU、Token、Prompt空间 |
| **输出确定性** | 确定性输出 | 概率性输出，同一输入不同结果 |
| **成本模型** | 固定基础设施成本 | 按Token计费，成本随输入动态变化 |
| **质量度量** | 响应码、延迟 | 语义质量、幻觉率、相关性 |
| **调试对象** | 代码逻辑、数据库查询 | Prompt质量、模型行为、上下文窗口 |
| **故障模式** | 超时、OOM、服务不可用 | 幻觉、偏离指令、上下文溢出 |

一个典型的AI应用调用链：

```plaintext
用户输入 → Prompt构建 → 向量检索(RAG) → LLM调用 → 后处理 → 响应
            ↑                                        ↓
        可能失败点:                              可能失败点:
        - Prompt模板缺失                         - 输出格式错误
        - 检索结果为空                            - 幻觉/编造信息
        - 上下文超长截断                          - Token超限
        - API限流/超时                            - 安全过滤触发
```

**关键洞察**：AI系统的故障往往是**语义层面**的——系统返回了200 OK，但内容是错的、有毒的、或者与用户意图无关的。传统的健康检查和错误码监控完全无法捕获这类问题。

---

## 二、可观测性三支柱在AI场景的演进

### 2.1 架构全景

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                    AI Observability Platform                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Traces  │  │   Logs   │  │  Metrics │  │  Evaluator │ │
│  │ 全链路追踪 │  │ 结构化日志 │  │  时序指标  │  │  质量评估   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘ │
│       │              │              │              │         │
│  ┌────┴──────────────┴──────────────┴──────────────┴──────┐ │
│  │              AI Semantic Layer                          │ │
│  │  ┌─────────────┬──────────────┬────────────────────┐   │ │
│  │  │Prompt版本    │Token消耗追踪  │幻觉检测/质量评分     │   │ │
│  │  │管理          │与成本归因     │与安全审计           │   │ │
│  │  └─────────────┴──────────────┴────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
│       │              │              │              │         │
│  ┌────┴──────────────┴──────────────┴──────────────┴──────┐ │
│  │              Data Ingestion Layer                       │ │
│  │     OpenTelemetry SDK / Fluent Bit / Prometheus        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Traces：LLM调用链追踪

传统的分布式追踪（Jaeger/Zipkin）对AI应用不够用。我们需要**语义级别的追踪**：

```plaintext
Trace: chat-session-abc123
├── Span: user-input-processing (5ms)
│   └── attributes: { token_count: 128, language: "zh-CN" }
├── Span: rag-retrieval (120ms)
│   ├── Span: vector-search (45ms)
│   │   └── attributes: { top_k: 5, results_found: 3, similarity_threshold: 0.7 }
│   ├── Span: rerank (60ms)
│   │   └── attributes: { model: "bge-reranker-v2", top_k: 3 }
│   └── attributes: { total_chunks: 3, total_tokens: 2048 }
├── Span: llm-call (1800ms)  ⭐ 核心Span
│   ├── attributes: {
│   │     "gen_ai.system": "openai",
│   │     "gen_ai.request.model": "gpt-4o",
│   │     "gen_ai.response.model": "gpt-4o-2026-05-13",
│   │     "gen_ai.usage.input_tokens": 2176,
│   │     "gen_ai.usage.output_tokens": 512,
│   │     "gen_ai.request.temperature": 0.7,
│   │     "gen_ai.request.max_tokens": 4096,
│   │     "gen_ai.response.finish_reason": "stop",
│   │     "llm.prompt.version": "v2.3",
│   │     "llm.cost.usd": 0.0187
│   │   }
│   └── events:
│       ├── "prompt.template.loaded" → template_id: "customer-service-v3"
│       ├── "llm.request.sent" → timestamp
│       ├── "llm.response.first_token" → TTFT: 320ms
│       └── "llm.response.completed"
├── Span: post-processing (50ms)
│   └── attributes: { format: "markdown", contains_code: true }
└── attributes: {
      "trace.total_tokens": 2760,
      "trace.total_cost_usd": 0.0189,
      "trace.end_to_end_latency_ms": 1975,
      "trace.user_satisfaction_score": null  // 异步评估填充
    }
```

**关键设计决策**：

1. **遵循 OpenTelemetry GenAI Semantic Conventions**：使用 `gen_ai.*` 属性命名空间，确保与未来的标准工具链兼容
2. **Token级别的成本追踪**：每个Span都记录该步骤的Token消耗，支持成本归因
3. **Prompt版本绑定**：追踪每个请求使用的Prompt模板版本，便于质量回溯

### 2.3 Logs：结构化LLM日志

AI应用的日志需要捕获**非确定性行为**的上下文：

```json
{
  "timestamp": "2026-05-31T10:23:45.123Z",
  "level": "INFO",
  "service": "ai-customer-service",
  "trace_id": "abc123def456",
  "span_id": "span-789",
  "event_type": "llm.response.completed",
  "llm": {
    "model": "gpt-4o",
    "prompt_version": "v2.3",
    "input_tokens": 2176,
    "output_tokens": 512,
    "finish_reason": "stop",
    "temperature": 0.7
  },
  "quality": {
    "hallucination_score": 0.12,
    "relevance_score": 0.89,
    "safety_score": 0.95,
    "groundedness_score": 0.87
  },
  "context": {
    "user_id": "user-456",
    "session_id": "session-789",
    "retrieved_docs": ["doc-001", "doc-015", "doc-042"],
    "rerank_scores": [0.92, 0.85, 0.78]
  },
  "cost": {
    "input_cost_usd": 0.0065,
    "output_cost_usd": 0.0122,
    "total_cost_usd": 0.0187
  }
}
```

**日志设计原则**：

| 原则 | 实现方式 | 价值 |
|------|---------|------|
| **可追溯** | 绑定 trace_id/span_id | 从日志反查完整调用链 |
| **可评估** | 包含质量评分字段 | 支持离线分析模型质量 |
| **可归因** | 记录成本到单条请求 | 精确的成本分摊 |
| **可回放** | 保存完整Prompt和输出 | 支持Prompt调试和回归测试 |

### 2.4 Metrics：面向AI的指标体系

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                 AI Metrics Hierarchy                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Business Layer (L1)                                        │
│  ├── ai_user_satisfaction_score (gauge)                     │
│  ├── ai_task_success_rate (counter)                         │
│  ├── ai_cost_per_task (gauge)                               │
│  └── ai_revenue_per_query (gauge)                           │
│                                                             │
│  Quality Layer (L2)                                         │
│  ├── ai_hallucination_rate (gauge)                          │
│  ├── ai_response_relevance (histogram)                      │
│  ├── ai_safety_filter_rate (counter)                        │
│  └── ai_grounding_score (histogram)                         │
│                                                             │
│  Performance Layer (L3)                                     │
│  ├── ai_llm_latency_seconds (histogram)                     │
│  ├── ai_ttft_seconds (histogram)  ← Time To First Token    │
│  ├── ai_tokens_per_request (histogram)                      │
│  └── ai_rag_retrieval_latency (histogram)                   │
│                                                             │
│  Infrastructure Layer (L4)                                  │
│  ├── ai_gpu_utilization (gauge)                             │
│  ├── ai_api_rate_limit_remaining (gauge)                    │
│  ├── ai_error_rate_by_provider (gauge)                      │
│  └── ai_cost_usd_total (counter)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Prometheus指标定义示例**：

```python
from prometheus_client import Histogram, Counter, Gauge

# 性能层指标
LLM_LATENCY = Histogram(
    'ai_llm_latency_seconds',
    'LLM inference latency',
    ['model', 'provider', 'task_type'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

TTFT = Histogram(
    'ai_ttft_seconds',
    'Time to first token',
    ['model', 'provider'],
    buckets=[0.1, 0.2, 0.5, 1, 2, 5]
)

TOKEN_USAGE = Histogram(
    'ai_tokens_per_request',
    'Token usage per request',
    ['model', 'token_type'],  # token_type: input/output
    buckets=[100, 500, 1000, 2000, 4000, 8000, 16000]
)

# 成本层指标
COST_TOTAL = Counter(
    'ai_cost_usd_total',
    'Total LLM cost in USD',
    ['model', 'provider', 'service']
)

# 质量层指标
HALLUCINATION_RATE = Gauge(
    'ai_hallucination_rate',
    'Hallucination detection rate',
    ['model', 'task_type']
)
```

---

## 三、核心模块深度设计

### 3.1 成本归因引擎

成本是AI应用最敏感的维度。我们需要从"月度账单"细化到**每次调用、每个用户、每个功能**的成本归因：

```plaintext
┌──────────────────────────────────────────────────────────┐
│                    Cost Attribution Engine                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Request Cost = Input Cost + Output Cost + Rerank Cost   │
│                                                          │
│  Input Cost  = input_tokens × model_input_price / 1M     │
│  Output Cost = output_tokens × model_output_price / 1M   │
│  Rerank Cost = rerank_tokens × rerank_model_price / 1M   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │           Cost Aggregation Dimensions            │    │
│  ├──────────┬──────────┬──────────┬────────────────┤    │
│  │ 按用户    │ 按功能    │ 按Prompt  │ 按时间窗口      │    │
│  │ 按团队    │ 按模型    │ 按任务类型 │ 按地理区域      │    │
│  └──────────┴──────────┴──────────┴────────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**成本归因数据模型**：

```sql
-- 成本明细表
CREATE TABLE llm_cost_records (
    id              BIGSERIAL PRIMARY KEY,
    trace_id        VARCHAR(64) NOT NULL,
    span_id         VARCHAR(32) NOT NULL,
    
    -- 模型信息
    model           VARCHAR(64) NOT NULL,
    provider        VARCHAR(32) NOT NULL,
    
    -- Token消耗
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cached_tokens   INTEGER DEFAULT 0,
    
    -- 成本
    input_cost_usd  DECIMAL(10,6) NOT NULL,
    output_cost_usd DECIMAL(10,6) NOT NULL,
    total_cost_usd  DECIMAL(10,6) NOT NULL,
    
    -- 归因维度
    service_name    VARCHAR(64),
    user_id         VARCHAR(64),
    team_id         VARCHAR(64),
    prompt_version  VARCHAR(32),
    task_type       VARCHAR(32),
    
    -- 时间
    created_at      TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_cost_time (created_at),
    INDEX idx_cost_user (user_id, created_at),
    INDEX idx_cost_model (model, created_at)
);
```

**实际成本看板示例**：

| 维度 | 本月消耗 | 本月成本 | 占比 | 环比 |
|------|---------|---------|------|------|
| 按模型 | | | | |
| ├ GPT-4o | 1.2B tokens | $8,400 | 42% | +15% |
| ├ Claude 3.5 | 800M tokens | $5,600 | 28% | +8% |
| ├ GPT-4o-mini | 2B tokens | $2,000 | 10% | +30% |
| └ 本地模型 | - | $4,000 (GPU) | 20% | +5% |
| 按功能 | | | | |
| ├ 智能客服 | 600M tokens | $4,200 | 21% | +12% |
| ├ 代码助手 | 900M tokens | $6,300 | 31% | +20% |
| ├ 文档摘要 | 500M tokens | $3,500 | 18% | -5% |
| └ 知识问答 | 1B tokens | $6,000 | 30% | +10% |

### 3.2 Prompt版本管理与AB测试

Prompt是AI应用的"代码"，但大多数人没有对Prompt进行版本管理：

```plaintext
┌─────────────────────────────────────────────────────────┐
│                Prompt Lifecycle Management                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Draft → Review → Staging → Canary → Production         │
│   │        │         │         │          │             │
│   v0.1     v0.2      v1.0      v1.0-canary  v1.0 GA    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │            Prompt Registry                      │    │
│  │  ┌─────────┬──────────┬──────────┬───────────┐  │    │
│  │  │ Template │ Variables│ Model    │ Quality   │  │    │
│  │  │ Content  │ Schema   │ Config   │ Threshold │  │    │
│  │  ├─────────┼──────────┼──────────┼───────────┤  │    │
│  │  │ v1.0    │ {doc}    │ gpt-4o   │ rel>0.8   │  │    │
│  │  │ v1.1    │ {doc,n}  │ gpt-4o   │ rel>0.85  │  │    │
│  │  │ v2.0    │ {ctx}    │ claude   │ rel>0.85  │  │    │
│  │  └─────────┴──────────┴──────────┴───────────┘  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  AB Test: v1.0 (50%) vs v2.0 (50%)                     │
│  Metrics: relevance, latency, cost, user_satisfaction   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Prompt版本追踪的实现**：

```python
from dataclasses import dataclass
from typing import Optional
import hashlib

@dataclass
class PromptVersion:
    version: str
    template: str
    variables: list[str]
    model_config: dict
    quality_threshold: float
    
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

class PromptRegistry:
    def __init__(self, storage):
        self.storage = storage
    
    def register(self, prompt: PromptVersion) -> str:
        """注册新Prompt版本"""
        self.storage.save(prompt)
        return prompt.version
    
    def get_active_version(self, prompt_name: str) -> PromptVersion:
        """获取当前激活的版本"""
        return self.storage.get_active(prompt_name)
    
    def ab_test(self, prompt_name: str, versions: dict[str, float]):
        """配置AB测试流量分配
        versions: {"v1.0": 0.5, "v2.0": 0.5}
        """
        self.storage.set_traffic_split(prompt_name, versions)
    
    def compare_versions(self, prompt_name: str, v1: str, v2: str, 
                         sample_size: int = 1000) -> dict:
        """对比两个版本的质量指标"""
        metrics_v1 = self.storage.get_metrics(prompt_name, v1, sample_size)
        metrics_v2 = self.storage.get_metrics(prompt_name, v2, sample_size)
        
        return {
            "v1": metrics_v1,
            "v2": metrics_v2,
            "winner": self._determine_winner(metrics_v1, metrics_v2),
            "confidence": self._calculate_confidence(metrics_v1, metrics_v2)
        }
```

### 3.3 幻觉检测与质量评估

**实时幻觉检测架构**：

```plaintext
┌──────────────────────────────────────────────────────────┐
│                Real-time Quality Pipeline                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  LLM Output                                              │
│      │                                                   │
│      ├──→ Rule-based Checks (同步, <10ms)                │
│      │     ├── 事实一致性检查 (数字/日期/名称)               │
│      │     ├── 格式合规检查 (JSON/Markdown)                │
│      │     └── 安全过滤 (敏感词/有害内容)                   │
│      │                                                   │
│      ├──→ LLM-as-Judge (异步, ~500ms)                    │
│      │     ├── 自洽性检测: 让模型回答同一问题两次对比         │
│      │     ├── 引用验证: 检查引用来源是否真实存在             │
│      │     └── 语义相关性: 输出与输入的相关程度               │
│      │                                                   │
│      └──→ Human-in-the-Loop (异步, 分钟级)                │
│            ├── 低置信度样本进入人工审核队列                  │
│            └── 审核结果反馈到评估数据集                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**幻觉检测实现**：

```python
class HallucinationDetector:
    def __init__(self, judge_model, reference_docs):
        self.judge_model = judge_model
        self.reference_docs = reference_docs
    
    async def detect(self, query: str, response: str, 
                     context: list[str]) -> dict:
        """多维度幻觉检测"""
        
        # 1. 引用验证：检查输出中引用的文档是否在context中
        citations = self._extract_citations(response)
        citation_valid = self._validate_citations(citations, context)
        
        # 2. 自洽性检测：同一问题问两次，对比答案一致性
        response_2 = await self.judge_model.generate(
            prompt=f"请回答以下问题：{query}",
            temperature=0.3
        )
        consistency_score = self._compute_consistency(response, response_2)
        
        # 3. LLM-as-Judge评估
        judge_prompt = f"""请评估以下回答的质量：
        
问题：{query}
参考文档：{chr(10).join(context)}
回答：{response}

请从以下维度评分(0-1)：
1. groundedness (基于参考文档的程度)
2. relevance (与问题的相关性)  
3. completeness (回答的完整性)
4. safety (安全性)"""
        
        judge_result = await self.judge_model.generate(judge_prompt)
        
        return {
            "citation_valid": citation_valid,
            "consistency_score": consistency_score,
            "groundedness": judge_result.groundedness,
            "relevance": judge_result.relevance,
            "completeness": judge_result.completeness,
            "safety": judge_result.safety,
            "hallucination_risk": self._compute_risk_score(
                citation_valid, consistency_score, judge_result
            )
        }
    
    def _compute_risk_score(self, citation, consistency, judge) -> float:
        """计算综合幻觉风险分数"""
        weights = {
            "citation": 0.3,
            "consistency": 0.3,
            "groundedness": 0.25,
            "safety": 0.15
        }
        
        score = (
            weights["citation"] * (1.0 if citation else 0.0) +
            weights["consistency"] * consistency +
            weights["groundedness"] * judge.groundedness +
            weights["safety"] * judge.safety
        )
        return round(score, 3)
```

---

## 四、技术选型与部署架构

### 4.1 开源技术栈推荐

| 层级 | 推荐方案 | 备选方案 | 选型理由 |
|------|---------|---------|---------|
| **采集层** | OpenTelemetry SDK | Langfuse SDK | OTel是CNCF标准，长期生态更好 |
| **追踪后端** | Jaeger + Tempo | Zipkin | Tempo与Grafana生态深度集成 |
| **指标后端** | Mimir / VictoriaMetrics | Prometheus | 多租户支持，长期存储成本低 |
| **日志后端** | Loki | Elasticsearch | 与追踪/指标统一在Grafana展示 |
| **LLM专项** | Langfuse / Phoenix | LangSmith | 开源可控，支持自部署 |
| **告警** | Grafana Alerting | PagerDuty | 统一告警入口 |
| **可视化** | Grafana | Metabase | 统一Dashboard |

### 4.2 部署架构

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                    Production Deployment                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AI Application Pods                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ App v1   │ │ App v2   │ │ App v3   │                    │
│  │ +OTel    │ │ +OTel    │ │ +OTel    │                    │
│  │ SDK      │ │ SDK      │ │ SDK      │                    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                    │
│       │            │            │                            │
│       └────────────┼────────────┘                            │
│                    │                                         │
│  ┌─────────────────┴──────────────────────────────────┐     │
│  │              OpenTelemetry Collector                │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │     │
│  │  │ Traces  │  │ Metrics │  │  Logs   │            │     │
│  │  │ Export  │  │ Export  │  │ Export  │            │     │
│  │  └────┬────┘  └────┬────┘  └────┬────┘            │     │
│  └───────┼────────────┼────────────┼──────────────────┘     │
│          │            │            │                         │
│  ┌───────┴──┐  ┌──────┴───┐  ┌───┴────────┐               │
│  │  Tempo   │  │  Mimir   │  │    Loki    │               │
│  │ (traces) │  │(metrics) │  │   (logs)   │               │
│  └───────┬──┘  └──────┬───┘  └───┬────────┘               │
│          │            │           │                          │
│  ┌───────┴────────────┴───────────┴──────────────────┐     │
│  │                   Grafana                          │     │
│  │  ┌──────────┬──────────┬──────────┬────────────┐  │     │
│  │  │ AI Cost  │ Quality  │ Latency  │ Hallucin.  │  │     │
│  │  │ Dashboard│ Dashboard│ Dashboard│ Dashboard  │  │     │
│  │  └──────────┴──────────┴──────────┴────────────┘  │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Langfuse (LLM专用)                      │   │
│  │  ┌──────────┬──────────┬──────────┬────────────┐    │   │
│  │  │ Traces   │ Prompts  │ Evaluator│ Cost       │    │   │
│  │  │ Viewer   │ Registry │ Pipeline │ Analytics  │    │   │
│  │  └──────────┴──────────┴──────────┴────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 OpenTelemetry集成代码

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ai-app")

class LLMInstrumentor:
    """LLM调用自动埋点"""
    
    def __init__(self, tracer):
        self.tracer = tracer
    
    def trace_llm_call(self, model: str, provider: str):
        """装饰器：自动追踪LLM调用"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "llm.completion",
                    attributes={
                        "gen_ai.system": provider,
                        "gen_ai.request.model": model,
                        "gen_ai.request.temperature": kwargs.get("temperature", 0.7),
                        "gen_ai.request.max_tokens": kwargs.get("max_tokens", 4096),
                    }
                ) as span:
                    # 记录请求开始
                    span.add_event("llm.request.sent")
                    
                    # 执行LLM调用
                    start_time = time.time()
                    response = await func(*args, **kwargs)
                    latency = time.time() - start_time
                    
                    # 记录响应
                    span.set_attribute("gen_ai.response.model", response.model)
                    span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
                    span.set_attribute("gen_ai.response.finish_reason", response.choices[0].finish_reason)
                    
                    # 计算成本
                    cost = self._calculate_cost(model, response.usage)
                    span.set_attribute("llm.cost.usd", cost)
                    span.set_attribute("llm.latency_ms", latency * 1000)
                    
                    # 记录事件
                    span.add_event("llm.response.completed", {
                        "total_tokens": response.usage.total_tokens,
                        "cost_usd": cost
                    })
                    
                    return response
            return wrapper
        return decorator
```

---

## 五、Grafana Dashboard设计

### 5.1 核心看板布局

```plaintext
┌─────────────────────────────────────────────────────────────┐
│  AI System Health Dashboard                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ 请求量   │ │ 平均延迟  │ │ Token   │ │ 今日成本  │          │
│  │ 12.5K/h │ │ 1.2s    │ │ 15.2M/h │ │ $284    │          │
│  │ ↑12%    │ │ ↓5%     │ │ ↑8%     │ │ ↑15%    │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│                                                             │
│  ┌──────────────────────────┐ ┌──────────────────────────┐ │
│  │ 请求量趋势 (24h)          │ │ 成本分布 (按模型)         │ │
│  │                          │ │                          │ │
│  │   ╱╲    ╱╲              │ │  ████████ GPT-4o (42%)   │ │
│  │  ╱  ╲╱╱  ╲  ╱╲         │ │  ██████   Claude (28%)   │ │
│  │ ╱        ╲╱  ╲╱        │ │  ████     Mini  (10%)    │ │
│  └──────────────────────────┘ │  ███████  Local (20%)    │ │
│                               └──────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────┐ ┌──────────────────────────┐ │
│  │ 幻觉率趋势               │ │ TTFF分布                  │ │
│  │                          │ │                          │ │
│  │  2%─ ─ ─ ─ ─ ─ ─ ─     │ │  P50: 320ms              │ │
│  │        ╱╲               │ │  P95: 890ms              │ │
│  │  1%──╱╱──╲╲───         │ │  P99: 2.1s               │ │
│  │  0%──────────────       │ │                          │ │
│  └──────────────────────────┘ └──────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 关键PromQL查询

```sql
-- 请求量QPS
sum(rate(ai_llm_requests_total[5m])) by (model, service)

-- P95延迟
histogram_quantile(0.95, sum(rate(ai_llm_latency_seconds_bucket[5m])) by (le, model))

-- 每小时成本
sum(rate(ai_cost_usd_total[1h])) by (service, team)

-- 幻觉率
sum(rate(ai_hallucination_detected_total[1h])) / sum(rate(ai_llm_requests_total[1h]))

-- Token效率 (输出Token / 输入Token)
sum(rate(ai_tokens_per_request_sum{token_type="output"}[1h])) 
/ sum(rate(ai_tokens_per_request_sum{token_type="input"}[1h]))
```

---

## 六、落地实施路线图

### 阶段一：基础可观测性（1-2周）

```
目标: 能看到AI系统在做什么
├── [ ] 集成OpenTelemetry SDK
├── [ ] 部署OTel Collector
├── [ ] 配置Tempo + Loki存储
├── [ ] 实现基础LLM调用追踪
└── [ ] 搭建Grafana基础Dashboard
```

### 阶段二：成本可见性（1-2周）

```
目标: 知道每分钱花在哪里
├── [ ] 实现Token级成本归因
├── [ ] 建立成本明细表
├── [ ] 搭建成本分析Dashboard
├── [ ] 配置预算告警
└── [ ] 成本按团队/功能分摊
```

### 阶段三：质量监控（2-3周）

```
目标: 知道AI输出的质量好不好
├── [ ] 集成Langfuse
├── [ ] 实现幻觉检测Pipeline
├── [ ] Prompt版本管理
├── [ ] 搭建质量评估Dashboard
└── [ ] 配置质量告警
```

### 阶段四：业务价值闭环（2-3周）

```
目标: 量化AI的业务价值
├── [ ] 用户满意度追踪
├── [ ] AI ROI计算模型
├── [ ] A/B测试框架
├── [ ] 自动化质量回归测试
└── [ ] 月度AI价值报告自动化
```

---

## 七、常见陷阱与最佳实践

### 7.1 常见陷阱

| 陷阱 | 症状 | 解决方案 |
|------|------|---------|
| **数据量爆炸** | 存储成本远超LLM成本 | 采样策略：成功请求10%采样，错误100%保留 |
| **延迟影响** | OTel采集增加50ms+延迟 | 异步导出 + Batch处理 + 降低采样率 |
| **成本监控盲区** | 本地模型/开源模型成本不可见 | GPU算力折算为成本，纳入统一视图 |
| **质量评估噪音** | 自动评估结果波动大 | 建立稳定评估基准，排除Prompt版本干扰 |
| **告警疲劳** | 每天上百条告警无人处理 | 分级告警 + 聚合 + 自动降级 |

### 7.2 最佳实践

1. **先做成本归因，再做质量评估**——成本是最容易量化且最有说服力的指标
2. **使用OpenTelemetry标准**——避免被单一工具锁定
3. **Prompt版本管理是基础设施**——没有版本管理，所有评估都是空中楼阁
4. **异步评估为主**——实时评估影响延迟，大多数质量评估可以异步完成
5. **建立反馈闭环**——人工审核结果回流到评估数据集，持续提升评估准确性

---

## 总结

AI系统的可观测性不是传统监控的简单扩展，而是需要全新的思维框架：

- **追踪**不仅要记录请求路径，还要记录Token消耗、Prompt版本和模型行为
- **日志**不仅要记录事件，还要记录语义质量评估结果
- **指标**不仅要监控延迟和错误率，还要监控幻觉率和成本效率

最终目标是建立一个**从底层基础设施到顶层业务价值**的完整可观测性体系，让AI系统不再是黑盒，而是可度量、可优化、可信赖的生产系统。

> 好的可观测性不是成本，而是投资——它让你花的每一分AI预算都产生可量化的回报。
