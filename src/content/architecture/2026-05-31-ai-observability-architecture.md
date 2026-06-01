---
title: "AI可观测性架构设计：从日志到全链路追踪的工程实践"
description: "深入解析AI系统可观测性架构设计，涵盖指标采集、分布式追踪、LLM专用可观测性方案与生产级落地实践"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: distributed
tags: ["可观测性", "分布式追踪", "AI运维", "系统架构", "LLM监控"]
draft: false
---

## 引言：为什么AI系统特别需要可观测性？

传统微服务的可观测性（Observability）已经是一套成熟的体系——Metrics、Logging、Tracing三支柱被广泛采用。但当服务核心从"确定性业务逻辑"变成"概率性LLM推理"时，这套体系面临了根本性的挑战。

一个典型的LLM应用请求链路可能是这样的：

```
用户输入 → Prompt模板 → 向量检索 → LLM推理 → 后处理 → 安全过滤 → 输出
```

这条链路中，每个环节都可能引入不确定性：检索结果质量波动、LLM生成内容随机、Token消耗不可预测。**传统监控告诉你"服务是否存活"，但无法告诉你"这次回答质量如何"**。

这就是AI可观测性要解决的核心问题。

## 一、AI可观测性架构全景

### 1.1 与传统可观测性的区别

| 维度 | 传统微服务 | AI系统 |
|------|-----------|--------|
| 核心指标 | QPS、延迟、错误率 | 质量评分、幻觉率、Token效率 |
| 追踪粒度 | HTTP/RPC调用 | Prompt-Response对、推理步骤 |
| 日志重点 | 错误堆栈、业务日志 | 输入输出全文、模型元数据 |
| 告警逻辑 | 阈值/同比/环比 | 质量漂移、成本突增、安全事件 |
| 数据量级 | MB级/天 | GB级/天（含完整Prompt/Response） |

### 1.2 分层架构设计

```
┌─────────────────────────────────────────────────────┐
│                   可视化与告警层                       │
│   Grafana Dashboard │ 自定义告警 │ 质量报表           │
├─────────────────────────────────────────────────────┤
│                   数据处理层                          │
│   Flink/ClickHouse │ 质量评分引擎 │ 异常检测         │
├─────────────────────────────────────────────────────┤
│                   数据采集层                          │
│   OTel SDK │ Langfuse │ 自定义Collector              │
├─────────────────────────────────────────────────────┤
│                   数据源层                            │
│   LLM Gateway │ Vector DB │ Prompt Engine │ 应用服务  │
└─────────────────────────────────────────────────────┘
```

## 二、指标体系设计

### 2.1 四层指标模型

AI系统的指标体系需要覆盖四个层次：

**L1 - 基础设施指标**（复用传统方案）
- GPU利用率、显存占用
- 服务可用性、P99延迟
- 请求QPS、错误率

**L2 - LLM推理指标**（AI特有）
- Token吞吐量（tokens/s）
- Time to First Token (TTFT)
- Token消耗量（input/output/completion）
- 模型版本与配置

**L3 - 质量指标**（核心差异点）
- 幻觉率（Hallucination Rate）
- 相关性评分（Relevance Score）
- 事实一致性（Factual Consistency）
- 安全拒绝率

**L4 - 业务指标**（价值衡量）
- 用户满意度评分
- 任务完成率
- 平均对话轮次
- 成本/收益比

### 2.2 指标采集实践

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# 创建LLM专用指标
meter = metrics.get_meter("llm-gateway")

# Token使用量（直方图，按模型分维度）
token_usage = meter.create_histogram(
    name="llm.tokens.total",
    description="Total tokens consumed per request",
    unit="tokens",
)

# TTFT（直方图，按模型和Prompt类型分维度）
ttft = meter.create_histogram(
    name="llm.ttft.seconds",
    description="Time to first token",
    unit="s",
)

# 质量评分（ Gauge，实时更新）
quality_score = meter.create_gauge(
    name="llm.response.quality",
    description="Quality score of LLM response",
)

def record_request_metrics(model: str, prompt_type: str,
                           input_tokens: int, output_tokens: int,
                           ttft_seconds: float, quality: float):
    attributes = {
        "model": model,
        "prompt_type": prompt_type,
    }
    token_usage.record(input_tokens + output_tokens, attributes)
    ttft.record(ttft_seconds, attributes)
    quality_score.set(quality, attributes)
```

## 三、分布式追踪：LLM链路的完整视图

### 3.1 Span设计

LLM应用的Trace需要表达"嵌套推理"的语义。一次RAG查询的Span结构：

```
[Trace] RAG-Query (2.3s)
  ├── [Span] EmbedQuery (12ms)
  ├── [Span] VectorSearch (450ms)
  │     └── [Span] Reranker (180ms)
  ├── [Span] LLM-Inference (1.6s)
  │     ├── [Span] Prompt-Render (5ms)
  │     ├── [Span] Prefill (800ms)
  │     └── [Span] Decode (800ms, 512 tokens)
  ├── [Span] PostProcess (50ms)
  └── [Span] SafetyCheck (30ms)
```

### 3.2 实现要点

```python
from opentelemetry import trace

tracer = trace.get_tracer("llm-rag-pipeline")

async def rag_query(query: str, config: dict):
    with tracer.start_as_current_span("rag-query") as root:
        root.set_attribute("user.query", query)
        root.set_attribute("rag.top_k", config.get("top_k", 5))

        # 向量检索
        with tracer.start_as_current_span("vector-search") as span:
            results = await vector_db.search(query, top_k=config["top_k"])
            span.set_attribute("db.results_count", len(results))
            span.set_attribute("db.latency_ms", results.latency_ms)

        # LLM推理
        with tracer.start_as_current_span("llm-inference") as span:
            span.set_attribute("llm.model", config["model"])
            span.set_attribute("llm.temperature", config.get("temperature", 0.7))

            response = await llm.generate(prompt)
            span.set_attribute("llm.input_tokens", response.usage.input_tokens)
            span.set_attribute("llm.output_tokens", response.usage.output_tokens)
            span.set_attribute("llm.ttft_ms", response.ttft_ms)

            # 质量评估（异步，不阻塞响应）
            quality = await evaluate_quality(query, response.text)
            span.set_attribute("llm.quality_score", quality.score)
            span.set_attribute("llm.hallucination_flag", quality.has_hallucination)

        root.set_attribute("rag.total_tokens", response.usage.total_tokens)
        return response
```

### 3.3 Trace与日志关联

将Trace ID注入到所有日志中，实现从Trace到日志的跳转：

```python
import logging
from opentelemetry import trace

class LLMTraceFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        ctx = span.get_span_context()
        record.trace_id = format(ctx.trace_id, '032x')
        record.span_id = format(ctx.span_id, '016x')
        return True

# 日志格式包含Trace信息
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] '
    'trace_id=%(trace_id)s span_id=%(span_id)s '
    '%(message)s'
)
```

## 四、LLM专用可观测性方案

### 4.1 Prompt-Response全量存储

AI系统的一个关键需求是**存储完整的Prompt和Response**，用于后续的质量分析、模型迭代和合规审计。但这带来了巨大的存储挑战。

**分层存储策略：**

| 数据类型 | 热存储（7天） | 温存储（30天） | 冷存储（1年） |
|---------|-------------|--------------|-------------|
| Trace元数据 | Elasticsearch | ClickHouse | S3+Parquet |
| Prompt/Response全文 | Redis Cluster | ClickHouse | S3+Parquet |
| 质量评分 | ClickHouse | ClickHouse | S3 |
| GPU指标 | Prometheus | Thanos | S3 |

### 4.2 质量自动评估

在生产环境中集成轻量级质量评估，作为可观测性的一部分：

```python
class QualityEvaluator:
    """轻量级在线质量评估器"""

    def __init__(self):
        # 使用小模型做在线评估，大模型做离线抽样
        self.online_model = load_lightweight_model("qwen-1.5b-eval")
        self.sampling_rate = 0.1  # 10%抽样用大模型精细评估

    async def evaluate(self, query: str, response: str,
                       context: str = None) -> QualityResult:
        # 快速评估：相关性 + 安全性
        relevance = await self.online_model.predict(
            f"判断回答与问题的相关性(1-5分): Q:{query} A:{response}"
        )

        # 抽样精细评估：幻觉检测
        hallucination = None
        if random.random() < self.sampling_rate:
            hallucination = await self.large_model.evaluate(
                query=query, response=response, context=context
            )

        return QualityResult(
            relevance_score=relevance,
            has_hallucination=hallucination,
            evaluation_latency_ms=time.time() - start,
        )
```

### 4.3 成本追踪与预算控制

LLM的Token消耗直接关联成本。可观测性系统需要提供实时成本视图：

```python
# 价格配置（按模型）
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},      # per 1M tokens
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
    "qwen-max": {"input": 2.40, "output": 9.60},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

# 在Trace中记录成本
span.set_attribute("llm.cost_usd", calculate_cost(model, input_tokens, output_tokens))
```

## 五、生产级告警策略

### 5.1 多维告警规则

AI系统的告警不能只看"服务是否存活"，需要多维度覆盖：

| 告警类别 | 触发条件 | 严重级别 | 处理方式 |
|---------|---------|---------|---------|
| 质量退化 | 幻觉率 > 5% 持续10min | P1 | 自动切换模型/降级 |
| 成本异常 | 单用户Token消耗 > 阈值 | P2 | 限流+通知 |
| 延迟恶化 | P99 > SLA * 1.5 持续5min | P1 | 扩容+排查 |
| 安全事件 | 敏感内容命中率突增 | P0 | 立即人工介入 |
| 模型漂移 | 质量评分周环比下降 > 10% | P2 | 触发模型重评估 |

### 5.2 实现示例

```yaml
# Prometheus AlertManager规则
groups:
  - name: llm-quality-alerts
    rules:
      - alert: LLMHallucinationRateHigh
        expr: |
          rate(llm_hallucination_total[5m])
          / rate(llm_response_total[5m]) > 0.05
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "LLM幻觉率超过5%阈值"

      - alert: LLMCostAnomaly
        expr: |
          sum by (user_id) (
            rate(llm_cost_usd_total[1h])
          ) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "用户{{ $labels.user_id }} 小时消费超过$10"

      - alert: LLMTTFTDegraded
        expr: |
          histogram_quantile(0.99,
            rate(llm_ttft_seconds_bucket[5m])
          ) > 3.0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "TTFT P99超过3秒，用户体验严重下降"
```

## 六、技术选型与落地建议

### 6.1 开源方案对比

| 方案 | 定位 | 优势 | 适用场景 |
|------|------|------|---------|
| **Langfuse** | LLM专用可观测性 | 开箱即用、Prompt管理 | 中小团队快速落地 |
| **OpenTelemetry** | 通用可观测性标准 | 生态丰富、厂商无关 | 与现有基础设施集成 |
| **Helicone** | LLM代理+监控 | 零代码接入 | 快速验证阶段 |
| **Phoenix (Arize)** | ML可观测性 | 数据漂移检测 | 模型质量管理 |
| **自建方案** | 完全可控 | 灵活定制 | 大厂/特殊需求 |

### 6.2 落地路径建议

**阶段一（1-2周）：基础监控**
- 部署Langfuse或Helicone
- 接入所有LLM调用的Trace
- 建立基础Dashboard

**阶段二（2-4周）：质量闭环**
- 集成质量自动评估
- 建立Prompt-Response全量存储
- 部署告警规则

**阶段三（1-2月）：深度优化**
- 自建指标体系
- 成本优化引擎
- A/B测试框架集成

## 七、总结

AI可观测性不是传统监控的简单延伸，而是一个需要重新设计的体系。核心要点：

1. **指标体系要覆盖质量层**——不能只看"系统是否活着"，更要看"回答好不好"
2. **Trace要表达LLM语义**——Token消耗、TTFT、质量评分都需要在Trace中体现
3. **Prompt/Response全量存储是刚需**——为模型迭代和合规审计提供数据基础
4. **成本追踪不可忽视**——LLM的Token消耗是实实在在的真金白银
5. **告警要多维度**——质量退化、成本异常、安全事件都需要独立的告警通道

可观测性是AI系统从"能用"到"好用"的关键基础设施。投资可观测性，就是在投资AI系统的长期可维护性。
