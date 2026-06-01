---
title: "AI应用可观测性实战：从LLM调用链追踪到质量监控体系搭建"
description: "系统讲解AI应用可观测性的完整方案，涵盖分布式追踪、Token消耗监控、输出质量评估和告警体系搭建，附完整架构图和部署方案。"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: infra
tags: ["可观测性", "LLM监控", "MLOps", "AI工程化", "生产运维"]
draft: false
---

## 为什么AI应用比传统应用更需要可观测性？

传统Web应用的可观测性已经非常成熟——Metrics、Logging、Tracing三支柱体系经过多年的实践检验。但当应用的核心从确定性逻辑变为大语言模型推理时，可观测性面临全新的挑战：

| 挑战维度 | 传统应用 | AI应用 |
|----------|---------|--------|
| 输出确定性 | 同样输入同样输出 | 同样输入可能不同输出 |
| 质量评估 | HTTP 200即成功 | 返回200但内容可能无意义 |
| 成本追踪 | 固定的计算资源 | Token消耗动态变化，成本难预估 |
| 延迟特性 | 毫秒级可预测 | 秒级且波动大 |
| 失败模式 | 明确的错误码 | 可能"幻觉"或答非所问 |
| 依赖链路 | 数据库/缓存/第三方API | Prompt + Context + Model + Post-processing |

这意味着我们不能简单地把传统可观测性方案搬到AI应用上，而是需要一套专门针对AI特性的可观测性体系。

## AI应用可观测性架构

在实际项目中，我设计了一套分层的可观测性架构：

```
┌─────────────────────────────────────────────────────┐
│                   可视化层                            │
│  Dashboard │ 告警面板 │ 质量报表 │ 成本分析          │
├─────────────────────────────────────────────────────┤
│                   分析层                              │
│  质量评估 │ 漂移检测 │ A/B分析 │ 异常检测            │
├─────────────────────────────────────────────────────┤
│                   采集层                              │
│  Trace采集 │ Metric采集 │ Log采集 │ Feedback采集      │
├─────────────────────────────────────────────────────┤
│                   存储层                              │
│  ClickHouse │ Prometheus │ Elasticsearch │ S3        │
├─────────────────────────────────────────────────────┤
│                   应用层                              │
│  LLM调用 │ RAG管道 │ Agent循环 │ Tool调用            │
└─────────────────────────────────────────────────────┘
```

## 核心观测维度

### 1. 分布式追踪（Tracing）

LLM应用的调用链路通常比传统应用更复杂。以一个典型的RAG Agent为例：

```
用户请求
  └─→ Agent Router (意图识别)
       ├─→ [知识问答路径]
       │    ├─→ Query Rewrite (LLM)
       │    ├─→ Vector Search (Embedding + DB)
       │    ├─→ Context Reranker (LLM)
       │    └─→ Answer Generation (LLM)
       ├─→ [工具调用路径]
       │    ├─→ Tool Selection (LLM)
       │    └─→ Tool Execution (API)
       └─→ [闲聊路径]
            └─→ Chat Generation (LLM)
```

每一跳都应该被追踪，记录输入、输出、耗时和Token消耗。

**实现方案：使用OpenTelemetry + 自定义Span**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化Tracer
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-app")

# LLM调用封装
class ObservableLLM:
    def __init__(self, model_client):
        self.client = model_client
    
    @tracer.start_as_current_span("llm_completion")
    def complete(self, messages, **kwargs):
        span = trace.get_current_span()
        
        # 记录输入
        span.set_attribute("llm.model", self.client.model)
        span.set_attribute("llm.input.messages", str(messages))
        span.set_attribute("llm.input.token_count", count_tokens(messages))
        
        # 调用LLM
        start = time.time()
        response = self.client.chat.completions.create(
            messages=messages, **kwargs
        )
        latency = time.time() - start
        
        # 记录输出
        span.set_attribute("llm.output.content", response.choices[0].message.content)
        span.set_attribute("llm.output.token_count", response.usage.completion_tokens)
        span.set_attribute("llm.latency_ms", latency * 1000)
        span.set_attribute("llm.total_tokens", response.usage.total_tokens)
        span.set_attribute("llm.cost_usd", calculate_cost(response.usage))
        
        return response
```

### 2. 质量监控（Quality Metrics）

这是AI应用可观测性最独特的部分。传统的HTTP状态码在LLM应用中几乎没有意义——返回200但内容是幻觉，这算成功还是失败？

**核心质量指标：**

| 指标 | 定义 | 采集方式 | 告警阈值 |
|------|------|----------|----------|
| 相关性得分 | 回答与问题的相关程度 | LLM-as-Judge | < 0.6 |
| 幻觉率 | 包含虚假信息的比例 | 对比知识库 | > 5% |
| 完整性 | 是否覆盖了问题的所有方面 | 人工抽检 | < 80% |
| 有害内容率 | 包含有害/不当内容的比例 | 分类器 | > 0.1% |
| 用户满意度 | 用户反馈的满意程度 | 星级评分 | < 3.5/5 |

**LLM-as-Judge实现：**

```python
class QualityEvaluator:
    def __init__(self, judge_model):
        self.judge = judge_model
    
    async def evaluate(self, query, response, context=None):
        """对单次LLM输出进行质量评估"""
        
        # 评估相关性
        relevance_prompt = f"""评估以下回答与问题的相关性。
        
问题: {query}
回答: {response}
{f"参考上下文: {context}" if context else ""}

请从1-5分进行评分，其中:
1分 = 完全无关
2分 = 部分相关但偏题
3分 = 基本相关但不完整
4分 = 相关且完整
5分 = 非常相关且深入

请只返回数字评分。"""

        score = await self.judge.complete(relevance_prompt)
        
        # 检测幻觉
        hallucination_prompt = f"""检查以下回答是否存在幻觉（编造事实）。

问题: {query}
回答: {response}
{f"参考上下文: {context}" if context else ""}

请判断回答中的信息是否有据可查，还是编造的。
返回格式: {{"has_hallucination": true/false, "details": "..."}}"""

        hallucination_check = await self.judge.complete(hallucination_prompt)
        
        return {
            "relevance_score": int(score.strip()),
            "hallucination": json.loads(hallucination_check),
            "timestamp": datetime.now().isoformat()
        }
```

**质量指标仪表盘：**

```
┌──────────────────────────────────────────────────┐
│              AI应用质量监控仪表盘                   │
├──────────────────────────────────────────────────┤
│  实时指标 (过去1小时)                              │
│  ┌──────────┬──────────┬──────────┬──────────┐   │
│  │ 请求总量  │ 平均延迟  │ 幻觉率   │ 满意度   │   │
│  │ 12,847   │ 2.3s     │ 2.1%     │ 4.2/5   │   │
│  │ ↑12%     │ ↑0.2s    │ ↓0.3%    │ ↑0.1    │   │
│  └──────────┴──────────┴──────────┴──────────┘   │
│                                                   │
│  质量趋势 (过去7天)                                │
│  相关性: ████████████████░░░░ 82% (↑2%)          │
│  幻觉率: ██░░░░░░░░░░░░░░░░░░  3.2% (↓0.5%)     │
│  完整性: ███████████████░░░░░ 78% (→)            │
│                                                   │
│  模型对比                                          │
│  ┌────────┬────────┬────────┬────────┐            │
│  │  GPT-4o │Claude  │GPT-4o  │ Claude │            │
│  │        │Sonnet  │mini    │ Haiku  │            │
│  ├────────┼────────┼────────┼────────┤            │
│  │4.3/5   │4.1/5   │3.8/5   │3.5/5   │            │
│  │$0.012  │$0.009  │$0.003  │$0.002  │            │
│  └────────┴────────┴────────┴────────┘            │
└──────────────────────────────────────────────────┘
```

### 3. 成本监控（Cost Tracking）

LLM API的成本是动态的，取决于输入输出的Token数。没有成本监控，月底账单可能让你大吃一惊。

**成本追踪数据模型：**

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CostRecord:
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    endpoint: str  # 哪个功能调用的
    user_id: str
    request_id: str
    
# 不同模型的定价 (2026年参考价格)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},      # per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
}

def calculate_cost(model, input_tokens, output_tokens):
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    cost = (input_tokens * pricing["input"] + 
            output_tokens * pricing["output"]) / 1_000_000
    return round(cost, 6)
```

**成本告警配置：**

```yaml
# cost_alerts.yaml
alerts:
  - name: "单日成本超预算"
    condition: "daily_cost > 100"
    action: 
      - notify: "slack:#ai-cost-alerts"
      - throttle: "reduce_to_50%"
    
  - name: "单次请求成本异常"
    condition: "request_cost > 0.5"
    action:
      - notify: "slack:#ai-cost-alerts"
      - log: "full_request"
    
  - name: "某端点成本突增"
    condition: "endpoint_cost_rate > avg * 3 for 10m"
    action:
      - notify: "email:team@company.com"
```

### 4. 漂移检测（Drift Detection）

LLM应用的质量可能随时间漂移——模型API更新、知识库过期、用户需求变化都会导致质量下降。

**漂移检测策略：**

```
┌─────────────────────────────────────────┐
│          漂移检测策略                     │
├─────────────────────────────────────────┤
│                                         │
│  1. 输出分布监控                          │
│     - Token长度分布变化                   │
│     - 响应时间分布变化                     │
│     - 错误率变化趋势                      │
│                                         │
│  2. 质量指标趋势                          │
│     - 相关性得分的滑动平均                  │
│     - 幻觉率的周环比                      │
│     - 用户满意度的移动窗口                  │
│                                         │
│  3. 行为模式监控                          │
│     - 用户Query分布变化                   │
│     - 功能使用频率变化                     │
│     - 回退/重试率变化                     │
│                                         │
│  4. 统计检验                             │
│     - KS检验 (输出长度分布)               │
│     - 卡方检验 (错误类型分布)              │
│     - 时序异常检测 (Isolation Forest)     │
│                                         │
└─────────────────────────────────────────┘
```

**实现代码：**

```python
import numpy as np
from scipy import stats
from collections import deque

class DriftDetector:
    def __init__(self, window_size=1000, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_scores = deque(maxlen=window_size)
        self.current_scores = deque(maxlen=window_size)
    
    def set_baseline(self, scores):
        """设置基线质量分数分布"""
        self.baseline_scores = deque(scores[-self.window_size:], 
                                      maxlen=self.window_size)
    
    def check_drift(self, new_score):
        """检查是否发生漂移"""
        self.current_scores.append(new_score)
        
        if len(self.current_scores) < 100:
            return {"drifted": False, "reason": "样本不足"}
        
        # KS检验：比较当前分布与基线分布
        if len(self.baseline_scores) > 0:
            ks_stat, p_value = stats.ks_2samp(
                list(self.baseline_scores),
                list(self.current_scores)
            )
            
            if p_value < self.threshold:
                return {
                    "drifted": True,
                    "reason": f"分布漂移 (KS={ks_stat:.4f}, p={p_value:.6f})",
                    "severity": "high" if ks_stat > 0.2 else "medium",
                    "suggestion": "建议检查模型API更新或知识库时效性"
                }
        
        # 均值漂移检测
        baseline_mean = np.mean(list(self.baseline_scores))
        current_mean = np.mean(list(self.current_scores))
        mean_change = (current_mean - baseline_mean) / baseline_mean
        
        if abs(mean_change) > 0.1:  # 10%以上的均值变化
            direction = "下降" if mean_change < 0 else "上升"
            return {
                "drifted": True,
                "reason": f"质量均值{direction} {abs(mean_change)*100:.1f}%",
                "severity": "high" if abs(mean_change) > 0.2 else "low",
                "suggestion": f"质量{direction}，建议排查原因"
            }
        
        return {"drifted": False}
```

## 完整部署方案

### 技术栈选择

```
┌─────────────── 生产部署技术栈 ───────────────┐
│                                               │
│  数据采集: OpenTelemetry SDK + Collector       │
│  追踪存储: Jaeger / Tempo                     │
│  指标存储: Prometheus + Thanos                │
│  日志存储: Elasticsearch / Loki               │
│  质量存储: ClickHouse (OLAP)                  │
│  可视化: Grafana                               │
│  告警: AlertManager + PagerDuty              │
│  调度: Airflow (定时评估任务)                  │
│                                               │
└───────────────────────────────────────────────┘
```

### Docker Compose部署

```yaml
# docker-compose.monitoring.yaml
version: '3.8'

services:
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    volumes:
      - ./otel-config.yaml:/etc/otelcol/config.yaml

  # Jaeger for tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "14268:14268"  # Collector HTTP

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # ClickHouse for quality data
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/var/lib/grafana/dashboards
      - ./grafana/provisioning:/etc/grafana/provisioning

volumes:
  clickhouse_data:
  grafana_data:
```

### 质量数据表设计（ClickHouse）

```sql
-- LLM调用追踪表
CREATE TABLE llm_traces (
    timestamp DateTime64(3),
    request_id String,
    trace_id String,
    span_id String,
    parent_span_id String,
    
    -- 模型信息
    model LowCardinality(String),
    provider LowCardinality(String),
    
    -- Token统计
    input_tokens UInt32,
    output_tokens UInt32,
    total_tokens UInt32,
    
    -- 性能指标
    latency_ms UInt32,
    first_token_latency_ms UInt32,
    
    -- 成本
    cost_usd Float64,
    
    -- 质量指标
    quality_score Nullable(Float64),
    relevance_score Nullable(UInt8),
    has_hallucination Nullable(Bool),
    user_rating Nullable(UInt8),
    
    -- 上下文
    endpoint LowCardinality(String),
    user_id String,
    
    -- 分区键
    date Date DEFAULT toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, endpoint, model);

-- 质量趋势聚合表
CREATE MATERIALIZED VIEW quality_trend_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, endpoint, model)
AS SELECT
    toDate(timestamp) as date,
    endpoint,
    model,
    count() as total_requests,
    avg(quality_score) as avg_quality,
    avg(relevance_score) as avg_relevance,
    countIf(has_hallucination) as hallucination_count,
    avg(cost_usd) as avg_cost,
    avg(latency_ms) as avg_latency
FROM llm_traces
GROUP BY date, endpoint, model;
```

## 告警规则设计

好的告警规则应该遵循"少而精"的原则，避免告警疲劳：

```yaml
# alert_rules.yaml
groups:
  - name: llm_quality_alerts
    rules:
      # 质量下降告警
      - alert: LLMQualityDegraded
        expr: avg_over_time(avg_quality[1h]) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "LLM质量下降"
          description: "{{ $labels.endpoint }} 过去1小时平均质量分 {{ $value }}"
      
      # 幻觉率突增
      - alert: HallucinationRateHigh
        expr: |
          rate(hallucination_count[5m]) / rate(total_requests[5m]) > 0.05
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "幻觉率超过阈值"
          description: "端点 {{ $labels.endpoint }} 幻觉率达到 {{ $value | humanizePercentage }}"
      
      # 延迟异常
      - alert: LLMLatencyHigh
        expr: avg_over_time(avg_latency[5m]) > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM响应延迟过高"
          description: "{{ $labels.endpoint }} 平均延迟 {{ $value }}ms"
      
      # 成本异常
      - alert: CostAnomalyDetected
        expr: |
          avg_over_time(total_cost[1h]) > 
          avg_over_time(total_cost[1h] offset 1d) * 2
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "成本异常"
          description: "过去1小时成本是昨日同期的 {{ $value }} 倍"
```

## 实战经验总结

### 经验1：先追踪再优化

不要试图一开始就搭建完美的可观测性体系。建议的演进路径：

```
阶段1 (第1周): 基础追踪
  - 记录每次LLM调用的输入/输出/耗时/Token
  - 简单的日志聚合

阶段2 (第2-4周): 质量评估
  - 引入LLM-as-Judge自动评估
  - 建立基础的质量指标

阶段3 (第2-3月): 成本优化
  - 精细化的成本追踪
  - 基于质量/成本的模型路由

阶段4 (第3-6月): 漂移监控
  - 自动化漂移检测
  - 闭环反馈机制
```

### 经验2：采样而非全量

LLM调用的trace数据量巨大，全量存储成本很高。建议：

- **全量存储**：错误请求、低质量响应、高成本请求
- **采样存储**：正常请求采样10%-20%
- **聚合存储**：每小时/每天的聚合统计数据

### 经验3：反馈闭环

可观测性的最终目的是改进应用。建立反馈闭环：

```
监控发现问题 → 自动生成分析报告 → 触发Prompt优化流程 → 
A/B测试验证 → 发布新版本 → 持续监控
```

这个闭环可以部分自动化——比如用DSPy的优化器根据质量数据自动调整Prompt。

## 总结

AI应用的可观测性不是传统可观测性的简单延伸，而是一个需要专门设计的系统工程。核心要点：

1. **追踪要完整**：从用户请求到最终输出的每一步都要可观测
2. **质量要量化**：不要满足于"能用"，要建立可量化的质量指标体系
3. **成本要透明**：实时追踪Token消耗和API成本，设置合理的告警阈值
4. **漂移要检测**：持续监控模型质量变化，及时发现和响应
5. **演进要渐进**：从基础追踪开始，逐步完善，避免过度设计

可观测性投入的回报是巨大的——它不仅能帮你快速定位线上问题，更能帮你持续优化应用质量、控制成本，最终让用户获得更好的AI体验。
