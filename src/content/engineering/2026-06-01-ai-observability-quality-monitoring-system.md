---
title: "AI应用可观测性与质量监控体系建设实战：从日志到智能告警的完整方案"
description: "系统性讲解AI应用可观测性体系的建设方法，涵盖日志采集、链路追踪、质量监控和智能告警等核心环节。"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: "infra"
tags: ["可观测性", "质量监控", "LLM监控", "SRE", "智能告警"]
draft: false
---

# AI应用可观测性与质量监控体系建设实战：从日志到智能告警的完整方案

## 引言

当你的 AI 应用从原型走向生产，一个不可回避的问题浮出水面：**如何知道它在真实世界中表现如何？** 传统应用的监控体系（日志 + 指标 + 链路追踪）依然有效，但 AI 应用引入了全新的挑战——输出的不确定性、模型行为的复杂性、以及用户对质量感知的主观性。

本文将分享我们在生产环境中构建 AI 应用可观测性体系的完整经验，从基础的日志采集到高级的智能告警，帮助你建立一套可落地的监控方案。

---

## 一、AI 应用可观测性的三大支柱

### 1.1 与传统应用的区别

传统应用的可观测性建立在确定性系统之上——同样的输入必然产生同样的输出。AI 应用打破了这个假设：

| 维度 | 传统应用 | AI 应用 |
|------|---------|---------|
| 输出确定性 | 确定性 | 概率性 |
| 质量评估 | 可自动验证 | 需要人工/AI评估 |
| 故障模式 | 逻辑错误/异常 | 幻觉/偏见/质量退化 |
| 输入复杂度 | 结构化数据 | 非结构化文本/图像 |
| 延迟特征 | 可预测 | 受输入长度/复杂度影响 |

### 1.2 三大支柱的 AI 适配

```
┌─────────────────────────────────────────────────────────────┐
│                   AI 应用可观测性体系                         │
├──────────────┬──────────────┬───────────────────────────────┤
│     日志      │    指标      │        链路追踪               │
│   (Logs)     │ (Metrics)    │      (Traces)                │
├──────────────┼──────────────┼───────────────────────────────┤
│ • 请求日志    │ • 延迟指标    │ • 请求链路                    │
│ • 模型调用    │ • 吞吐量     │ • 模型调用链                   │
│ • 输出内容    │ • 错误率     │ • 工具调用链                   │
│ • 质量评估    │ • 质量分数   │ • 上下文传递                   │
│ • 用户反馈    │ • 资源消耗   │ • Token 使用追踪              │
└──────────────┴──────────────┴───────────────────────────────┘
```

---

## 二、日志体系建设

### 2.1 分层日志架构

AI 应用的日志需要分层设计，每一层服务于不同的分析场景：

```python
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class LLMLogEntry:
    """LLM 调用日志条目"""
    # 基础信息
    timestamp: float
    request_id: str
    user_id: str
    session_id: str
    
    # 模型信息
    model_name: str
    model_version: str
    
    # 输入信息
    messages: List[Dict[str, str]]
    system_prompt: str
    
    # 输出信息
    response: str
    finish_reason: str
    
    # Token 统计
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # 延迟信息
    time_to_first_token: float  # TTFT
    total_latency: float
    tokens_per_second: float
    
    # 质量信号
    quality_score: Optional[float] = None
    user_feedback: Optional[str] = None
    
    # 错误信息
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # 上下文
    metadata: Optional[Dict[str, Any]] = None

@dataclass  
class AgentLogEntry:
    """Agent 调用日志条目"""
    timestamp: float
    request_id: str
    session_id: str
    
    # Agent 信息
    agent_name: str
    agent_version: str
    
    # 工具调用
    tool_name: str
    tool_input: str
    tool_output: str
    tool_latency: float
    tool_success: bool
    
    # 推理链
    reasoning_chain: List[Dict[str, str]]
    
    # Token 统计
    total_tokens: int
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
```

### 2.2 日志采集管道

生产环境的日志采集需要考虑高吞吐、低延迟和可靠性：

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  应用实例     │───→│  日志 Agent   │───→│  消息队列     │
│  (多副本)     │    │  (Fluent Bit) │    │  (Kafka)     │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼───────┐
                    │  实时流处理   │←───│  消费者组     │
                    │  (Flink)     │    │  (消费者)     │
                    └──────┬───────┘    └──────────────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 时序数据库│ │ 搜索引擎  │ │ 数据湖    │
        │(Prometheus)│ │(Elasticsearch)│ │(S3/HDFS)│
        └──────────┘ └──────────┘ └──────────┘
```

#### Fluent Bit 配置示例

```yaml
# fluent-bit.conf
[SERVICE]
    Flush         5
    Log_Level     info
    Parsers_File  parsers.conf

[INPUT]
    Name          tail
    Path          /var/log/ai-app/*.log
    Parser        json
    Tag           ai-app.*

[FILTER]
    Name          modify
    Match         ai-app.*
    Add           service_name ai-chatbot
    Add           environment production

[FILTER]
    Name          nest
    Match         ai-app.*
    Operation     lift
    Nested_under  metadata
    Add_prefix    meta_

[OUTPUT]
    Name          kafka
    Match         ai-app.*
    Brokers       kafka-0:9092,kafka-1:9092,kafka-2:9092
    Topics        ai-logs
    Timestamp_Key timestamp
    compression   gzip
```

### 2.3 日志采样策略

在高流量场景下，全量日志采集会带来巨大的存储和处理压力。合理的采样策略至关重要：

```python
import random
from typing import Optional

class AdaptiveSampler:
    """自适应采样器"""
    
    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
        self.error_rate = 1.0  # 错误日志全量采集
        self.slow_rate = 0.5   # 慢请求 50% 采样
        
    def should_sample(
        self, 
        latency: float, 
        has_error: bool,
        quality_score: Optional[float] = None
    ) -> bool:
        """决定是否采样该条日志"""
        
        # 错误日志全量采集
        if has_error:
            return True
            
        # 慢请求提高采样率
        if latency > 5.0:  # 5秒以上
            return random.random() < self.slow_rate
            
        # 质量评分异常的日志提高采样率
        if quality_score is not None and quality_score < 0.6:
            return random.random() < 0.5
            
        # 其他请求使用基础采样率
        return random.random() < self.base_rate

# 使用示例
sampler = AdaptiveSampler(base_rate=0.1)

# 在日志记录时
if sampler.should_sample(
    latency=response.total_latency,
    has_error=response.error is not None,
    quality_score=response.quality_score
):
    log_entry = LLMLogEntry(...)
    logger.info(json.dumps(asdict(log_entry)))
```

---

## 三、指标体系建设

### 3.1 核心指标定义

AI 应用的指标体系需要覆盖性能、质量和资源三个维度：

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# ============ 性能指标 ============

# 请求计数
llm_requests_total = Counter(
    'llm_requests_total',
    'LLM 请求总数',
    ['model', 'status', 'user_type']
)

# 延迟分布
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM 请求延迟（秒）',
    ['model', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# 首 token 延迟
llm_ttft_seconds = Histogram(
    'llm_ttft_seconds',
    '首 token 延迟（秒）',
    ['model'],
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
)

# ============ Token 指标 ============

# Token 使用量
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Token 使用总量',
    ['model', 'type']  # type: prompt/completion
)

# 每秒 token 数
llm_tokens_per_second = Gauge(
    'llm_tokens_per_second',
    '每秒生成 token 数',
    ['model']
)

# ============ 质量指标 ============

# 质量评分
llm_quality_score = Histogram(
    'llm_quality_score',
    '输出质量评分',
    ['model', 'task_type'],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

# 幻觉检测率
llm_hallucination_rate = Gauge(
    'llm_hallucination_rate',
    '幻觉检测率',
    ['model'],
)

# 用户反馈
llm_user_feedback = Counter(
    'llm_user_feedback_total',
    '用户反馈计数',
    ['model', 'feedback_type']  # positive/negative/neutral
)

# ============ 资源指标 ============

# GPU 显存使用
gpu_memory_used = Gauge(
    'gpu_memory_used_bytes',
    'GPU 显存使用量',
    ['gpu_id']
)

# KV Cache 使用率
kv_cache_usage = Gauge(
    'kv_cache_usage_ratio',
    'KV Cache 使用率',
    ['model']
)

# 并发请求数
llm_concurrent_requests = Gauge(
    'llm_concurrent_requests',
    '当前并发请求数',
    ['model']
)
```

### 3.2 关键性能指标（KPI）

基于实际业务经验，以下 KPI 是 AI 应用的核心关注点：

| 指标 | 定义 | 目标值 | 告警阈值 |
|------|------|--------|---------|
| P95 延迟 | 95% 请求的响应时间 | < 2s | > 5s |
| TTFT P95 | 95% 请求的首 token 延迟 | < 500ms | > 1s |
| 吞吐量 | 每秒处理的 token 数 | > 1000 tokens/s | < 500 tokens/s |
| 错误率 | 失败请求占比 | < 0.1% | > 1% |
| 质量分 | 输出质量评分均值 | > 0.8 | < 0.6 |
| 幻觉率 | 检测到的幻觉占比 | < 2% | > 5% |

### 3.3 自定义 Dashboard

```yaml
# Grafana Dashboard 配置示例
dashboard:
  title: "AI 应用监控大盘"
  panels:
    - title: "请求量 & 错误率"
      type: "stat"
      targets:
        - expr: "sum(rate(llm_requests_total[5m]))"
          legend: "RPS"
        - expr: "sum(rate(llm_requests_total{status='error'}[5m])) / sum(rate(llm_requests_total[5m]))"
          legend: "错误率"
    
    - title: "延迟分布"
      type: "heatmap"
      targets:
        - expr: "histogram_quantile(0.50, rate(llm_latency_seconds_bucket[5m]))"
          legend: "P50"
        - expr: "histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))"
          legend: "P95"
        - expr: "histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m]))"
          legend: "P99"
    
    - title: "Token 使用趋势"
      type: "graph"
      targets:
        - expr: "sum(rate(llm_tokens_total{type='prompt'}[5m]))"
          legend: "Input Tokens/s"
        - expr: "sum(rate(llm_tokens_total{type='completion'}[5m]))"
          legend: "Output Tokens/s"
    
    - title: "质量评分分布"
      type: "histogram"
      targets:
        - expr: "histogram_quantile(0.50, rate(llm_quality_score_bucket[5m]))"
          legend: "中位数"
        - expr: "histogram_quantile(0.10, rate(llm_quality_score_bucket[5m]))"
          legend: "P10 (下限)"
```

---

## 四、链路追踪体系

### 4.1 分布式追踪设计

AI 应用的链路追踪需要捕获完整的请求生命周期：

```
┌─────────────────────────────────────────────────────────────────┐
│                         请求链路示例                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Web API] ───────────────────────────────────────────────────→ │
│      │                                                          │
│      ├─→ [Auth Middleware] ─→ [Rate Limiter] ─→ [Router]       │
│      │                                                          │
│      ├─→ [Prompt Builder] ─→ [Context Fetcher]                  │
│      │         │                                    │           │
│      │         └─→ [RAG Retriever] ─→ [Vector DB]  └─→ [Redis] │
│      │                                                          │
│      ├─→ [LLM Gateway] ─→ [Model Router] ─→ [TensorRT-LLM]    │
│      │                          │                               │
│      │                          └─→ [Fallback Model]            │
│      │                                                          │
│      ├─→ [Quality Checker] ─→ [Hallucination Detector]          │
│      │                                                          │
│      └─→ [Response Builder] ─→ [Streaming Handler]              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 追踪上下文传播

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化追踪器
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ai-app")

class LLMGateway:
    """LLM 网关 - 链路追踪示例"""
    
    def __init__(self):
        self.tracer = trace.get_tracer("llm-gateway")
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        with self.tracer.start_as_current_span(
            "llm.chat_completion",
            attributes={
                "model": request.model,
                "user_id": request.user_id,
                "message_count": len(request.messages),
            }
        ) as span:
            try:
                # 1. 路由选择
                with self.tracer.start_as_current_span("model.routing"):
                    model = await self.select_model(request)
                    span.set_attribute("selected_model", model.name)
                
                # 2. 调用模型
                with self.tracer.start_as_current_span(
                    "model.inference",
                    attributes={"model": model.name}
                ) as inference_span:
                    response = await model.generate(request)
                    
                    # 记录 Token 使用
                    inference_span.set_attribute(
                        "token.usage.prompt", response.prompt_tokens
                    )
                    inference_span.set_attribute(
                        "token.usage.completion", response.completion_tokens
                    )
                    inference_span.set_attribute(
                        "latency.ttft", response.ttft
                    )
                
                # 3. 质量检查
                with self.tracer.start_as_current_span("quality.check"):
                    quality_score = await self.check_quality(response)
                    span.set_attribute("quality.score", quality_score)
                
                return response
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise
```

### 4.3 Trace 采样策略

```python
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ParentBasedTraceIdRatio,
    ALWAYS_ON,
    ALWAYS_OFF,
    CompositeSampler,
)

# 生产环境采样策略
def create_sampler(config: dict):
    """根据配置创建采样器"""
    
    # 基础采样率
    base_sampler = ParentBasedTraceIdRatio(config.get("base_rate", 0.1))
    
    # 错误请求全量采样
    error_sampler = ALWAYS_ON
    
    # 慢请求提高采样率
    slow_sampler = ParentBasedTraceIdRatio(0.5)
    
    return CompositeSampler(
        samplers=[
            # 错误请求全量采样（通过 span attributes 判断）
            # 慢请求 50% 采样
            # 其他请求 10% 采样
            base_sampler
        ]
    )
```

---

## 五、质量监控体系

### 5.1 输出质量自动评估

AI 应用的质量监控需要自动化评估输出质量：

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class QualityDimension(Enum):
    RELEVANCY = "relevancy"          # 相关性
    COHERENCE = "coherence"          # 连贯性
    FACTUALITY = "factuality"        # 事实性
    SAFETY = "safety"               # 安全性
    COMPLETENESS = "completeness"   # 完整性

@dataclass
class QualityAssessment:
    """质量评估结果"""
    overall_score: float
    dimensions: dict[QualityDimension, float]
    issues: List[str]
    suggestions: List[str]

class QualityEvaluator:
    """LLM 输出质量评估器"""
    
    def __init__(self, evaluator_model: str = "gpt-4"):
        self.evaluator_model = evaluator_model
    
    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        reference: Optional[str] = None
    ) -> QualityAssessment:
        """评估 LLM 输出质量"""
        
        prompt = f"""请评估以下 LLM 输出的质量。

用户问题: {query}

LLM 输出: {response}

{f"参考上下文: {context}" if context else ""}
{f"参考答案: {reference}" if reference else ""}

请从以下维度进行评估（每个维度 0-1 分）：
1. 相关性：输出是否回答了用户问题
2. 连贯性：输出是否逻辑清晰、语言流畅
3. 事实性：输出中的事实是否准确
4. 安全性：输出是否包含有害内容
5. 完整性：输出是否提供了足够的信息

输出格式（JSON）：
{{
    "overall_score": 0.85,
    "dimensions": {{
        "relevancy": 0.9,
        "coherence": 0.85,
        "factuality": 0.8,
        "safety": 1.0,
        "completeness": 0.7
    }},
    "issues": ["信息不够完整", "某些表述不够准确"],
    "suggestions": ["建议补充更多细节", "可以引用具体来源"]
}}"""
        
        # 调用评估模型
        result = await self.call_llm(prompt)
        return self.parse_result(result)
    
    async def detect_hallucination(
        self,
        response: str,
        context: str
    ) -> tuple[bool, List[str]]:
        """检测幻觉"""
        
        prompt = f"""请检查以下 LLM 输出是否包含幻觉（与上下文不符的内容）。

上下文:
{context}

LLM 输出:
{response}

请列出所有与上下文不符的内容，并说明原因。

输出格式（JSON）：
{{
    "has_hallucination": true,
    "hallucinations": [
        {{
            "claim": "输出中的具体声明",
            "evidence": "上下文中的相关证据",
            "reason": "为什么这是幻觉"
        }}
    ]
}}"""
        
        result = await self.call_llm(prompt)
        parsed = self.parse_result(result)
        return parsed["has_hallucination"], parsed.get("hallucinations", [])
```

### 5.2 质量监控仪表盘

质量监控的核心是实时跟踪输出质量的变化趋势：

```python
# 质量趋势监控
class QualityMonitor:
    """质量趋势监控"""
    
    def __init__(self):
        self.baseline_scores = {}  # 基线分数
        self.alert_thresholds = {
            "score_drop": 0.15,      # 分数下降 15% 触发告警
            "hallucination_rate": 0.05,  # 幻觉率超过 5% 触发告警
            "negative_feedback_rate": 0.1,  # 负面反馈超过 10% 触发告警
        }
    
    async def check_quality_trend(
        self, 
        model: str,
        recent_scores: List[float]
    ) -> Optional[Alert]:
        """检查质量趋势"""
        
        if len(recent_scores) < 10:
            return None
        
        current_avg = sum(recent_scores[-10:]) / 10
        baseline = self.baseline_scores.get(model, 0.8)
        
        # 检查分数下降
        if baseline - current_avg > self.alert_thresholds["score_drop"]:
            return Alert(
                level="warning",
                title=f"模型 {model} 质量分数下降",
                message=f"当前平均分 {current_avg:.2f}，基线 {baseline:.2f}",
                action="建议检查模型版本或数据分布变化"
            )
        
        return None
```

### 5.3 A/B 测试质量对比

```python
class ABTestQualityAnalyzer:
    """A/B 测试质量分析"""
    
    async def compare_variants(
        self,
        variant_a_responses: List[dict],
        variant_b_responses: List[dict]
    ) -> dict:
        """对比两个变体的质量"""
        
        # 评估每个变体的质量
        scores_a = [r["quality_score"] for r in variant_a_responses]
        scores_b = [r["quality_score"] for r in variant_b_responses]
        
        # 统计分析
        mean_a = sum(scores_a) / len(scores_a)
        mean_b = sum(scores_b) / len(scores_b)
        
        # 计算置信区间
        ci_a = self._confidence_interval(scores_a)
        ci_b = self._confidence_interval(scores_b)
        
        # 判断是否有显著差异
        is_significant = not (
            ci_a[0] > ci_b[1] or ci_b[0] > ci_a[1]
        )
        
        return {
            "variant_a": {
                "mean": mean_a,
                "ci_95": ci_a,
                "sample_size": len(scores_a)
            },
            "variant_b": {
                "mean": mean_b,
                "ci_95": ci_b,
                "sample_size": len(scores_b)
            },
            "difference": mean_b - mean_a,
            "is_significant": is_significant,
            "recommendation": "使用变体 B" if mean_b > mean_a and is_significant else "继续测试"
        }
    
    def _confidence_interval(self, scores: list, confidence: float = 0.95):
        """计算 95% 置信区间"""
        import math
        n = len(scores)
        mean = sum(scores) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in scores) / (n - 1))
        se = std / math.sqrt(n)
        
        # 使用 t 分布的近似值
        z = 1.96  # 95% 置信度
        margin = z * se
        
        return (mean - margin, mean + margin)
```

---

## 六、智能告警体系

### 6.1 多级告警策略

```yaml
# 告警规则配置
alerting_rules:
  # P0 - 严重告警（立即响应）
  - name: "llm_high_error_rate"
    condition: "error_rate > 0.05 for 5m"
    severity: "critical"
    channels: ["pagerduty", "slack-critical", "phone"]
    runbook: "https://wiki.internal/runbook/llm-error-rate"
    
  - name: "llm_service_down"
    condition: "up{job='llm-service'} == 0 for 2m"
    severity: "critical"
    channels: ["pagerduty", "slack-critical", "phone"]
    
  # P1 - 高优先级告警（30分钟内响应）
  - name: "llm_high_latency"
    condition: "histogram_quantile(0.95, llm_latency_seconds) > 5 for 10m"
    severity: "warning"
    channels: ["slack-alerts", "email"]
    
  - name: "llm_quality_degradation"
    condition: "avg_over_time(llm_quality_score[1h]) < 0.6"
    severity: "warning"
    channels: ["slack-alerts", "email"]
    
  # P2 - 中优先级告警（24小时内响应）
  - name: "llm_high_token_usage"
    condition: "rate(llm_tokens_total[1h]) > 1000000"
    severity: "info"
    channels: ["slack-info"]
    
  - name: "gpu_memory_high"
    condition: "gpu_memory_used / gpu_memory_total > 0.9"
    severity: "info"
    channels: ["slack-info"]
```

### 6.2 异常检测告警

```python
import numpy as np
from typing import List

class AnomalyDetector:
    """基于统计的异常检测"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: List[float] = []
    
    def update(self, value: float) -> bool:
        """更新观测值并检测异常"""
        self.history.append(value)
        
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        if len(self.history) < 20:
            return False
        
        # 使用 Z-score 检测异常
        mean = np.mean(self.history[:-1])
        std = np.std(self.history[:-1])
        
        if std == 0:
            return False
        
        z_score = (value - mean) / std
        
        # Z-score 超过 3 视为异常
        return abs(z_score) > 3
    
    def detect_trend(self) -> str:
        """检测趋势"""
        if len(self.history) < 50:
            return "insufficient_data"
        
        recent = self.history[-10:]
        older = self.history[-50:-10]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        change_pct = (recent_mean - older_mean) / older_mean
        
        if change_pct > 0.1:
            return "increasing"
        elif change_pct < -0.1:
            return "decreasing"
        else:
            return "stable"

class SmartAlertManager:
    """智能告警管理器"""
    
    def __init__(self):
        self.detectors = {
            "latency": AnomalyDetector(),
            "error_rate": AnomalyDetector(),
            "quality_score": AnomalyDetector(),
            "token_usage": AnomalyDetector(),
        }
        self.alert_history = []
    
    def check_metrics(self, metrics: dict) -> List[dict]:
        """检查指标并生成告警"""
        alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.detectors:
                detector = self.detectors[metric_name]
                
                if detector.update(value):
                    alerts.append({
                        "metric": metric_name,
                        "value": value,
                        "severity": "warning",
                        "message": f"{metric_name} 出现异常值: {value:.4f}",
                        "trend": detector.detect_trend()
                    })
        
        return alerts
```

### 6.3 告警聚合与降噪

```python
class AlertAggregator:
    """告警聚合与降噪"""
    
    def __init__(self, dedup_window: int = 300):
        self.dedup_window = dedup_window  # 5分钟去重窗口
        self.active_alerts = {}
        self.suppressed = set()
    
    def process_alert(self, alert: dict) -> Optional[dict]:
        """处理告警"""
        alert_key = f"{alert['metric']}:{alert.get('source', 'default')}"
        
        # 去重检查
        if alert_key in self.active_alerts:
            last_time = self.active_alerts[alert_key]
            if time.time() - last_time < self.dedup_window:
                return None  # 重复告警，忽略
        
        # 抑制检查
        if alert_key in self.suppressed:
            return None
        
        self.active_alerts[alert_key] = time.time()
        
        # 聚合相关告警
        correlated = self.find_correlated_alerts(alert)
        if correlated:
            alert["correlated"] = correlated
            alert["message"] = f"{alert['message']} (关联 {len(correlated)} 个告警)"
        
        return alert
    
    def find_correlated_alerts(self, alert: dict) -> List[dict]:
        """查找关联告警"""
        correlated = []
        for key, timestamp in self.active_alerts.items():
            if time.time() - timestamp < 60:  # 1分钟内的告警
                if key != f"{alert['metric']}:{alert.get('source', 'default')}":
                    correlated.append({"key": key, "time": timestamp})
        return correlated
```

---

## 七、实战案例：完整的可观测性部署

### 7.1 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI 应用可观测性架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ API 网关 │  │ Worker 1│  │ Worker 2│  │ Worker 3│           │
│  │         │  │         │  │         │  │         │           │
│  │  Tracer │  │  Tracer │  │  Tracer │  │  Tracer │           │
│  │  Logger │  │  Logger │  │  Logger │  │  Logger │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │                  │
│       └────────────┼────────────┼────────────┘                  │
│                    ↓                                             │
│           ┌────────────────┐                                    │
│           │  OTel Collector│                                    │
│           │  (接收/处理)    │                                    │
│           └───────┬────────┘                                    │
│                   │                                             │
│       ┌───────────┼───────────┐                                 │
│       ↓           ↓           ↓                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                          │
│  │ Jaeger  │ │Prometheus│ │   S3    │                          │
│  │ (Traces)│ │(Metrics) │ │(Logs)   │                          │
│  └────┬────┘ └────┬────┘ └────┬────┘                          │
│       │           │           │                                 │
│       └───────────┼───────────┘                                 │
│                   ↓                                             │
│           ┌────────────────┐                                    │
│           │    Grafana     │                                    │
│           │  (Dashboard)   │                                    │
│           └───────┬────────┘                                    │
│                   │                                             │
│       ┌───────────┼───────────┐                                 │
│       ↓           ↓           ↓                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                          │
│  │ 告警管理 │ │ 质量报告 │ │ 成本分析 │                          │
│  └─────────┘ └─────────┘ └─────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 部署配置

```yaml
# docker-compose.yml
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

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards

  # Jaeger
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "14268:14268"  # HTTP collector
```

### 7.3 关键告警规则

```yaml
# prometheus-rules.yml
groups:
  - name: ai-app-alerts
    rules:
      # 错误率告警
      - alert: HighErrorRate
        expr: |
          sum(rate(llm_requests_total{status="error"}[5m])) 
          / sum(rate(llm_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM 服务错误率过高"
          description: "错误率 {{ $value | humanizePercentage }}，超过 5% 阈值"

      # 延迟告警
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m])) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "LLM 服务延迟过高"
          description: "P95 延迟 {{ $value }}s，超过 5s 阈值"

      # 质量告警
      - alert: QualityDegradation
        expr: |
          avg_over_time(llm_quality_score[1h]) < 0.6
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "LLM 输出质量下降"
          description: "当前平均质量分 {{ $value }}，低于 0.6"

      # 资源告警
      - alert: HighGPUUsage
        expr: |
          gpu_memory_used / gpu_memory_total > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU 显存使用率过高"
          description: "使用率 {{ $value | humanizePercentage }}"
```

---

## 八、总结

AI 应用的可观测性不是简单的"加日志、加指标"，而是一个需要系统性设计的工程体系。核心要点：

1. **分层设计**：日志、指标、链路追踪三大支柱缺一不可，但要根据场景合理采样
2. **质量导向**：传统监控关注"系统是否正常"，AI 监控还要关注"输出质量是否达标"
3. **智能告警**：从简单的阈值告警升级到基于统计的异常检测，减少告警疲劳
4. **持续迭代**：可观测性体系需要随着业务发展不断演进

建立完善的可观测性体系，是 AI 应用从"能用"到"好用"的关键一步。它不仅能帮助你快速定位问题，更能为模型优化和产品迭代提供数据支撑。
