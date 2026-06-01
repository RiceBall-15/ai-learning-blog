---
title: "大模型应用可观测性工程实践：从日志到全链路追踪"
description: "深入探讨LLM应用的可观测性体系建设，涵盖分布式追踪、成本监控、质量评估三大支柱，附完整工程实现方案"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: infra
tags: ["可观测性", "LLM", "OpenTelemetry", "MLOps", "生产运维"]
draft: false
---

# 大模型应用可观测性工程实践：从日志到全链路追踪

## 引言：为什么LLM应用需要全新的可观测性？

传统Web应用的可观测性体系已经非常成熟——Prometheus指标、ELK日志、Jaeger链路追踪三板斧足以覆盖大部分场景。但当你把LLM引入系统后，会发现这套体系远远不够。

一个典型的LLM应用请求可能涉及：

1. **多模型调用链**：用户问题先经过意图分类模型，再路由到专业模型，最后由评估模型打分
2. **动态token消耗**：每次调用的token数波动极大，直接影响成本
3. **质量评估主观性**：相同输入，不同时间的输出质量可能差异显著
4. **长尾延迟**：单次LLM调用可能耗时数十秒，传统超时策略失效

本文将分享我们在生产环境中构建LLM可观测性体系的完整经验。

---

## 一、可观测性三大支柱在LLM场景下的演进

### 1.1 日志（Logging）：从结构化日志到语义日志

传统日志记录HTTP状态码和响应时间就够了，LLM应用需要记录更多信息：

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class LLMLogEntry:
    # 基础信息
    timestamp: str
    request_id: str
    user_id: str
    
    # 模型调用信息
    model_name: str
    model_version: str
    provider: str  # openai / anthropic / local
    
    # Token统计
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # 延迟分解
    time_to_first_token: float  # 首token延迟(ms)
    total_latency: float        # 总延迟(ms)
    
    # 质量信号
    finish_reason: str  # stop / length / content_filter
    response_quality_score: Optional[float] = None
    
    # 成本
    cost_usd: float = 0.0
    
    # 上下文
    system_prompt_hash: str = ""
    user_message_length: int = 0
    
    def to_log_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class LLMLogger:
    """LLM专用日志记录器"""
    
    # 模型定价表（每1K token，USD）
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-haiku-3.5": {"input": 0.0008, "output": 0.004},
    }
    
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    def calculate_cost(self, model: str, prompt_tokens: int, 
                       completion_tokens: int) -> float:
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        return (prompt_tokens * pricing["input"] + 
                completion_tokens * pricing["output"]) / 1000
    
    def create_entry(self, request_id: str, model: str, 
                     prompt_tokens: int, completion_tokens: int,
                     latency_ms: float, ttft_ms: float,
                     finish_reason: str, **kwargs) -> LLMLogEntry:
        return LLMLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id,
            user_id=kwargs.get("user_id", "anonymous"),
            model_name=model,
            model_version=kwargs.get("version", "latest"),
            provider=kwargs.get("provider", "unknown"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_to_first_token=ttft_ms,
            total_latency=latency_ms,
            finish_reason=finish_reason,
            cost_usd=self.calculate_cost(model, prompt_tokens, completion_tokens),
            system_prompt_hash=kwargs.get("system_prompt_hash", ""),
            user_message_length=kwargs.get("user_msg_length", 0),
        )
```

**关键设计决策**：

| 维度 | 传统日志 | LLM日志 | 设计原因 |
|------|---------|---------|---------|
| Token记录 | 无需 | 必须 | 成本归因 |
| TTFT | 无需 | 必须 | 用户体验监控 |
| 质量评分 | 无需 | 必须 | 模型效果追踪 |
| 成本计算 | 无需 | 必须 | 实时预算告警 |
| 内容脱敏 | 简单 | 复杂 | 含用户隐私信息 |

### 1.2 指标（Metrics）：LLM专用监控指标体系

```python
from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# ===== 核心LLM指标 =====

# 请求计数（按模型、状态、finish_reason分组）
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'provider', 'finish_reason', 'status']
)

# 延迟直方图
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM response latency',
    ['model', 'provider'],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60]
)

# TTFT直方图（首token延迟）
llm_ttft_seconds = Histogram(
    'llm_ttft_seconds',
    'Time to first token',
    ['model', 'provider'],
    buckets=[0.1, 0.3, 0.5, 1, 2, 5]
)

# Token消耗计数器
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens consumed',
    ['model', 'type'],  # type: prompt / completion
    ['model', 'type']
)

# 成本累计
llm_cost_usd_total = Counter(
    'llm_cost_usd_total',
    'Total cost in USD',
    ['model', 'provider']
)

# 并发调用数
llm_concurrent_requests = Gauge(
    'llm_concurrent_requests',
    'Number of concurrent LLM calls',
    ['model']
)

# 质量评分
llm_quality_score = Summary(
    'llm_quality_score',
    'Response quality score',
    ['model']
)


class LLMMetricsCollector:
    """统一的LLM指标收集器"""
    
    def record_request(self, model: str, provider: str, 
                       latency: float, ttft: float,
                       prompt_tokens: int, completion_tokens: int,
                       cost_usd: float, finish_reason: str,
                       quality_score: float = None):
        
        status = "success" if finish_reason == "stop" else "error"
        
        llm_requests_total.labels(
            model=model, provider=provider,
            finish_reason=finish_reason, status=status
        ).inc()
        
        llm_latency_seconds.labels(model=model, provider=provider).observe(latency)
        llm_ttft_seconds.labels(model=model, provider=provider).observe(ttft)
        
        llm_tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
        llm_tokens_total.labels(model=model, type="completion").inc(completion_tokens)
        
        llm_cost_usd_total.labels(model=model, provider=provider).inc(cost_usd)
        
        if quality_score is not None:
            llm_quality_score.labels(model=model).observe(quality_score)
```

### 1.3 追踪（Tracing）：OpenTelemetry集成

LLM应用的链路追踪需要处理几个特殊场景：

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import openai
import hashlib

# 初始化Tracer
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="localhost:4317")
provider.add_span_processor(BatchSpanExporter(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-app")


class LLMTracingWrapper:
    """为LLM调用添加分布式追踪"""
    
    def __init__(self, client):
        self.client = client
    
    def chat_completion(self, messages, model, **kwargs):
        with tracer.start_as_current_span("llm.chat_completion") as span:
            # 记录请求属性
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.provider", self._get_provider(model))
            span.set_attribute("llm.messages.count", len(messages))
            span.set_attribute("llm.temperature", kwargs.get("temperature", 1.0))
            span.set_attribute("llm.max_tokens", kwargs.get("max_tokens", 4096))
            
            # 记录system prompt哈希（不记录原文）
            if messages and messages[0]["role"] == "system":
                prompt_hash = hashlib.sha256(
                    messages[0]["content"][:100].encode()
                ).hexdigest()[:16]
                span.set_attribute("llm.system_prompt.hash", prompt_hash)
            
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                
                # 记录响应
                choice = response.choices[0]
                span.set_attribute("llm.finish_reason", choice.finish_reason)
                span.set_attribute("llm.usage.prompt_tokens", 
                                   response.usage.prompt_tokens)
                span.set_attribute("llm.usage.completion_tokens", 
                                   response.usage.completion_tokens)
                span.set_attribute("llm.usage.total_tokens", 
                                   response.usage.total_tokens)
                
                # 记录响应摘要（截取前200字符）
                span.set_attribute("llm.response.preview", 
                                   choice.message.content[:200])
                
                # 设置状态
                span.set_status(trace.StatusCode.OK)
                
                return response
                
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise
    
    def _get_provider(self, model: str) -> str:
        if model.startswith("gpt"):
            return "openai"
        elif model.startswith("claude"):
            return "anthropic"
        return "unknown"
```

**LLM追踪的关键属性命名规范**：

```
llm.model              → 模型名称
llm.provider           → 服务商
llm.usage.*            → Token使用量
llm.finish_reason      → 结束原因
llm.cost.usd           → 成本
llm.response.preview   → 响应摘要
llm.system_prompt.hash → 系统提示词哈希
```

---

## 二、成本监控与预算告警

### 2.1 多维度成本归因

LLM应用的成本控制是生产环境中最关键的需求之一：

```python
from dataclasses import dataclass
from typing import Dict
import asyncio
from collections import defaultdict

@dataclass
class CostBudget:
    daily_limit_usd: float
    per_user_limit_usd: float
    per_request_limit_usd: float
    alert_threshold: float = 0.8  # 80%时告警

class LLMCostManager:
    """多维度成本管理器"""
    
    def __init__(self, budget: CostBudget):
        self.budget = budget
        self._daily_cost = 0.0
        self._user_costs: Dict[str, float] = defaultdict(float)
        self._model_costs: Dict[str, float] = defaultdict(float)
        self._alerts = []
    
    def check_budget(self, user_id: str, model: str, 
                     estimated_cost: float) -> bool:
        """检查是否超出预算，返回是否允许执行"""
        
        # 检查单次请求限额
        if estimated_cost > self.budget.per_request_limit_usd:
            self._alerts.append({
                "type": "single_request_exceeded",
                "user_id": user_id,
                "model": model,
                "cost": estimated_cost,
                "limit": self.budget.per_request_limit_usd,
            })
            return False
        
        # 检查用户日限额
        user_daily = self._user_costs[user_id] + estimated_cost
        if user_daily > self.budget.per_user_limit_usd:
            self._alerts.append({
                "type": "user_daily_exceeded",
                "user_id": user_id,
                "projected_cost": user_daily,
                "limit": self.budget.per_user_limit_usd,
            })
            return False
        
        # 检查全局日限额
        projected = self._daily_cost + estimated_cost
        if projected > self.budget.daily_limit_usd:
            self._alerts.append({
                "type": "daily_budget_exceeded",
                "projected_cost": projected,
                "limit": self.budget.daily_limit_usd,
            })
            return False
        
        # 告警阈值检查
        if projected > self.budget.daily_limit_usd * self.budget.alert_threshold:
            self._alerts.append({
                "type": "daily_budget_warning",
                "projected_cost": projected,
                "threshold": self.budget.daily_limit_usd * self.budget.alert_threshold,
            })
        
        return True
    
    def record_cost(self, user_id: str, model: str, cost: float):
        """记录实际成本"""
        self._daily_cost += cost
        self._user_costs[user_id] += cost
        self._model_costs[model] += cost
    
    def get_cost_breakdown(self) -> dict:
        """获取成本分布"""
        return {
            "daily_total": round(self._daily_cost, 4),
            "by_model": dict(self._model_costs),
            "top_users": sorted(
                self._user_costs.items(), 
                key=lambda x: x[1], reverse=True
            )[:10],
            "alerts": self._alerts[-10:],  # 最近10条告警
        }
```

### 2.2 成本优化策略表

| 策略 | 实现方式 | 节省幅度 | 适用场景 |
|------|---------|---------|---------|
| Prompt缓存 | LRU Cache + 语义相似度 | 30-50% | 重复查询场景 |
| 模型降级 | 自动fallback到更便宜模型 | 40-70% | 简单任务 |
| 响应缓存 | Redis + TTL | 20-40% | 稳定性要求低的场景 |
| Token压缩 | Prompt压缩/摘要 | 20-35% | 长上下文场景 |
| 批量处理 | API Batch模式 | 50% | 异步任务 |
| 流式输出 | 减少完整响应等待 | 0（体验优化） | 所有场景 |

---

## 三、质量评估与监控

### 3.1 自动化质量评分系统

```python
from enum import Enum
from dataclasses import dataclass
from typing import List

class QualityDimension(Enum):
    RELEVANCE = "relevance"        # 相关性
    ACCURACY = "accuracy"          # 准确性
    COMPLETENESS = "completeness"  # 完整性
    COHERENCE = "coherence"        # 连贯性
    SAFETY = "safety"              # 安全性

@dataclass
class QualityScore:
    dimension: QualityDimension
    score: float  # 0-1
    reason: str

@dataclass 
class QualityReport:
    request_id: str
    scores: List[QualityScore]
    overall_score: float
    flags: List[str]  # 需要人工审查的标记
    
    def to_metrics(self) -> dict:
        return {
            f"quality_{s.dimension.value}": s.score 
            for s in self.scores
        } | {"quality_overall": self.overall_score}


class LLMQualityEvaluator:
    """基于LLM-as-Judge的质量评估器"""
    
    JUDGE_PROMPT = """你是一个LLM输出质量评估专家。请对以下LLM响应进行多维度评分。

## 用户输入
{user_input}

## LLM响应
{llm_response}

## 评估维度
1. 相关性(0-1): 响应是否切题
2. 准确性(0-1): 信息是否正确
3. 完整性(0-1): 是否回答了所有问题
4. 连贯性(0-1): 逻辑是否通顺
5. 安全性(0-1): 是否包含有害内容

请以JSON格式返回每个维度的评分和理由。"""
    
    def __init__(self, judge_model="gpt-4o-mini"):
        self.judge_model = judge_model
    
    async def evaluate(self, user_input: str, 
                       llm_response: str) -> QualityReport:
        """评估LLM响应质量"""
        # 使用轻量模型做评估，控制成本
        judge_prompt = self.JUDGE_PROMPT.format(
            user_input=user_input,
            llm_response=llm_response
        )
        
        # 实际调用judge模型进行评分
        # 这里简化为示例
        scores = [
            QualityScore(QualityDimension.RELEVANCE, 0.85, "基本切题"),
            QualityScore(QualityDimension.ACCURACY, 0.90, "信息准确"),
            QualityScore(QualityDimension.COMPLETENESS, 0.75, "部分遗漏"),
            QualityScore(QualityDimension.COHERENCE, 0.88, "逻辑通顺"),
            QualityScore(QualityDimension.SAFETY, 0.95, "无安全问题"),
        ]
        
        overall = sum(s.score for s in scores) / len(scores)
        flags = [s.dimension.value for s in scores if s.score < 0.6]
        
        return QualityReport(
            request_id="",
            scores=scores,
            overall_score=overall,
            flags=flags
        )
```

### 3.2 质量告警规则

```
# Prometheus告警规则示例（alertmanager.yml片段）
groups:
  - name: llm-quality
    rules:
      - alert: LLMQualityDegraded
        expr: avg_over_time(llm_quality_score{model="gpt-4o"}[5m]) < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "LLM质量下降 - {{ $labels.model }}"
          
      - alert: LLMHighErrorRate
        expr: |
          rate(llm_requests_total{status="error"}[5m]) 
          / rate(llm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          
      - alert: LLMBudgetWarning
        expr: llm_cost_usd_total > 100
        labels:
          severity: warning
```

---

## 四、完整可观测性架构

### 4.1 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                    LLM Application                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  LLM Logger  │  │   Metrics    │  │  Trace Context  │ │
│  │  (结构化日志) │  │  Collector   │  │  Propagator     │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘ │
│         │                │                    │          │
└─────────┼────────────────┼────────────────────┼──────────┘
          │                │                    │
    ┌─────▼─────┐   ┌──────▼──────┐    ┌──────▼──────┐
    │   Loki    │   │ Prometheus  │    │   Jaeger /  │
    │  (日志)   │   │  (指标)     │    │   Tempo     │
    └─────┬─────┘   └──────┬──────┘    └──────┬──────┘
          │                │                    │
    ┌─────▼─────────────────▼────────────────────▼──────┐
    │                  Grafana Dashboard                 │
    │  ┌──────────┐ ┌──────────┐ ┌───────────────────┐  │
    │  │ 成本面板  │ │ 质量面板  │ │ 延迟/可用性面板   │  │
    │  └──────────┘ └──────────┘ └───────────────────┘  │
    └────────────────────────────────────────────────────┘
```

### 4.2 关键看板指标

**成本看板**：

| 指标 | 数据源 | 刷新频率 | 告警阈值 |
|------|--------|---------|---------|
| 每日总成本 | Prometheus Counter | 实时 | >$100/天 |
| 每模型成本占比 | Prometheus Counter | 5min | 单模型>60% |
| 每用户成本Top10 | 自定义Counter | 10min | >$10/天 |
| 缓存命中率 | 自定义Gauge | 1min | <30% |

**延迟看板**：

| 指标 | 数据源 | 刷新频率 | 告警阈值 |
|------|--------|---------|---------|
| P50/P95/P99延迟 | Histogram | 实时 | P95>10s |
| 首token延迟P95 | Histogram | 实时 | >2s |
| 并发请求数 | Gauge | 实时 | >100 |
| 超时请求率 | Counter | 5min | >1% |

**质量看板**：

| 指标 | 数据源 | 刷新频率 | 告警阈值 |
|------|--------|---------|---------|
| 平均质量评分 | Summary | 5min | <0.7 |
| 安全评分分布 | Summary | 5min | <0.8 |
| 内容过滤触发率 | Counter | 1min | >5% |
| 人工审查率 | Counter | 10min | >10% |

---

## 五、实战部署清单

### 部署步骤

```bash
# 1. 安装依赖
pip install opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp prometheus-client

# 2. 配置环境变量
export OTEL_EXPORTER_OTLP_ENDPOINT="localhost:4317"
export OTEL_SERVICE_NAME="llm-app"
export LLM_DAILY_BUDGET_USD="100"

# 3. 启动collector
docker run -d --name otel-collector \
  -p 4317:4317 -p 4318:4318 \
  otel/opentelemetry-collector-contrib

# 4. 部署Prometheus + Grafana
docker-compose up -d  # 包含预配置的看板
```

### 常见陷阱

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 日志量爆炸 | 记录了完整prompt/response | 只记录哈希和摘要 |
| 指标基数过高 | label包含user_id | 使用采样或聚合 |
| 追踪延迟 | OTLP同步export | 使用BatchSpanExporter |
| 成本评估不准 | 未区分定价模型 | 维护完整定价表 |
| 质量评估偏差 | Judge模型不稳定 | 多模型交叉验证 |

---

## 总结

LLM应用的可观测性不是传统监控的简单扩展，而是一套全新的工程体系。核心要点：

1. **日志**：记录token消耗、TTFT、finish_reason等LLM特有维度
2. **指标**：建立成本、延迟、质量三维监控体系
3. **追踪**：使用OpenTelemetry标准化LLM调用链
4. **告警**：设置多级预算告警和质量降级告警
5. **看板**：区分成本、延迟、质量三个独立面板

只有建立起完善的可观测性体系，才能在生产环境中自信地运行LLM应用，并持续优化成本和质量。

---

*更多LLM工程实践内容，欢迎关注本博客的engineering系列。*
