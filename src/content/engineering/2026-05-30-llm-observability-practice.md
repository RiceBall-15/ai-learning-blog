---
title: "LLM可观测性实战：从日志采集到智能告警的全链路监控体系"
description: "深入解析LLM应用的可观测性建设，涵盖分布式追踪、指标采集、日志聚合、智能告警与成本分析，构建生产级LLM监控体系的完整技术方案"
date: 2026-05-30
author: "RiceBall-15"
category: "engineering"
subCategory: "infra"
tags: ["LLM可观测性", "分布式追踪", "监控告警", "AI工程化", "MLOps", "成本分析"]
draft: false
---

# LLM可观测性实战：从日志采集到智能告警的全链路监控体系

## 一、引言：为什么LLM应用需要专业可观测性？

### 1.1 传统监控的失效

传统的应用监控基于三个假设：请求是确定性的、延迟是可预测的、错误是明确的。但LLM应用打破了这三个假设：

```
┌─────────────────────────────────────────────────────────────────────┐
│              传统监控 vs LLM监控的差异                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  传统应用监控                    LLM应用监控                         │
│  ┌──────────────────┐          ┌──────────────────┐                │
│  │ 请求成功率: 99.9% │          │ 请求成功率: 95-99%│                │
│  │ 延迟: 50-200ms   │          │ 延迟: 1-30s      │                │
│  │ 错误: 4xx/5xx    │          │ 错误: 多样+隐性  │                │
│  │ 成本: 可预测     │          │ 成本: Token波动  │                │
│  │ 输出: 确定性     │          │ 输出: 概率性     │                │
│  └──────────────────┘          └──────────────────┘                │
│                                                                      │
│  LLM应用特有的监控维度:                                                │
│  ├── Token使用量 (输入/输出/总计)                                     │
│  ├── 延迟分布 (首Token时间 / 完整响应时间)                            │
│  ├── 输出质量 (幻觉率 / 相关性 / 安全性)                              │
│  ├── 成本分析 (按模型/用户/功能)                                      │
│  ├── 上下文利用率 (上下文长度 vs 模型上限)                             │
│  └── 流式输出性能 (Token/s / 首Token延迟)                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 可观测性三大支柱在LLM中的演进

传统的可观测性三大支柱（Metrics、Logging、Tracing）在LLM场景下需要全新的解读：

| 支柱 | 传统应用 | LLM应用 |
|------|----------|---------|
| **Metrics** | QPS、延迟、错误率 | Token/s、首Token延迟、幻觉率、成本/1K tokens |
| **Logging** | 请求日志、错误日志 | Prompt/Completion日志、质量评估日志、安全审计日志 |
| **Tracing** | HTTP请求链路 | LLM调用链路（含RAG检索、多轮对话、工具调用） |

## 二、LLM可观测性架构全景

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM可观测性平台架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      数据采集层                               │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ SDK埋点  │  │ Proxy采集│  │ Sidecar  │  │ Webhook  │  │   │
│  │  │          │  │          │  │          │  │          │  │   │
│  │  │ Python   │  │ LiteLLM  │  │ Envoy   │  │ 回调接口 │  │   │
│  │  │ SDK      │  │ Proxy    │  │ Filter  │  │          │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      数据处理层                               │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ 流处理   │  │ 质量分析 │  │ 成本计算 │  │ 安全检测 │  │   │
│  │  │          │  │          │  │          │  │          │  │   │
│  │  │ Flink /  │  │ 幻觉检测 │  │ Token    │  │ PII检测  │  │   │
│  │  │ Kafka    │  │ 相关性   │  │ 计费     │  │ 注入检测 │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      存储与查询层                              │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ 时序数据 │  │ 日志存储 │  │ 追踪存储 │  │ 向量存储 │  │   │
│  │  │          │  │          │  │          │  │          │  │   │
│  │  │Prometheus│  │ClickHouse│  │Jaeger /  │  │ Milvus  │  │   │
│  │  │/VictoriaM│  │/Elastic  │  │Tempo     │  │ /Chroma │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      可视化与告警层                            │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ Grafana  │  │ 自定义   │  │ 智能告警 │  │ 成本报表 │  │   │
│  │  │ Dashboard│  │ 看板     │  │          │  │          │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据采集方案对比

| 方案 | 实现方式 | 侵入性 | 数据丰富度 | 推荐场景 |
|------|----------|--------|------------|----------|
| **SDK埋点** | 在代码中直接调用SDK | 高 | 高 | 自研应用、需要定制化 |
| **Proxy采集** | 通过代理拦截所有请求 | 低 | 中 | 快速接入、多模型统一 |
| **Sidecar** | 作为Pod边车运行 | 低 | 中 | Kubernetes环境 |
| **Callback** | 利用框架回调接口 | 中 | 中 | 使用LangChain等框架 |

**推荐方案**：对于大多数团队，**Proxy采集**（如LiteLLM Proxy）是最佳起点——零侵入、快速上线、支持多模型。

## 三、核心指标体系

### 3.1 LLM专用指标定义

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM核心指标体系                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  性能指标 Performance                                        │   │
│  │                                                             │   │
│  │  • TTFT (Time To First Token): 首Token延迟                  │   │
│  │    - P50: < 1s    P95: < 3s    P99: < 5s                  │   │
│  │                                                             │   │
│  │  • TPOT (Time Per Output Token): 每Token生成时间            │   │
│  │    - 目标: < 50ms/token                                    │   │
│  │                                                             │   │
│  │  • Throughput: 吞吐量 (tokens/s)                            │   │
│  │    - 目标: > 50 tokens/s                                   │   │
│  │                                                             │   │
│  │  • E2E Latency: 端到端延迟                                  │   │
│  │    - P50: < 5s    P95: < 15s   P99: < 30s                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  成本指标 Cost                                                │   │
│  │                                                             │   │
│  │  • Cost per Request: 每请求成本                             │   │
│  │  • Cost per 1K Tokens: 每千Token成本                        │   │
│  │  • Daily/Monthly Spend: 日/月总支出                          │   │
│  │  • Cost by Model: 按模型分类成本                             │   │
│  │  • Cost by Feature: 按功能分类成本                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  质量指标 Quality                                            │   │
│  │                                                             │   │
│  │  • Hallucination Rate: 幻觉率 (< 5%)                       │   │
│  │  • Relevance Score: 相关性评分 (> 0.8)                      │   │
│  │  • Safety Score: 安全性评分 (> 0.95)                        │   │
│  │  • User Satisfaction: 用户满意度 (> 4.0/5.0)                │   │
│  │  • Refusal Rate: 拒绝率 (监控是否过度拒绝)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  可靠性指标 Reliability                                       │   │
│  │                                                             │   │
│  │  • Success Rate: 成功率 (> 99%)                             │   │
│  │  • Error Rate by Type: 分类错误率                            │   │
│  │  • Retry Rate: 重试率 (< 5%)                                │   │
│  │  • Fallback Rate: 降级率 (< 1%)                             │   │
│  │  • Circuit Breaker Trips: 熔断器触发次数                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 PromQL指标定义

```yaml
# prometheus-rules.yaml
groups:
  - name: llm_metrics
    rules:
      # 性能指标
      - record: llm:ttft:p50
        expr: histogram_quantile(0.5, rate(llm_ttft_seconds_bucket[5m]))
      
      - record: llm:ttft:p99
        expr: histogram_quantile(0.99, rate(llm_ttft_seconds_bucket[5m]))
      
      - record: llm:throughput:avg
        expr: rate(llm_tokens_generated_total[5m])
      
      # 成本指标
      - record: llm:cost:hourly
        expr: sum(rate(llm_cost_dollars_total[1h])) by (model)
      
      - record: llm:cost:daily
        expr: sum(increase(llm_cost_dollars_total[24h])) by (model, feature)
      
      # 质量指标
      - record: llm:hallucination:rate
        expr: sum(rate(llm_hallucination_detected_total[1h])) / sum(rate(llm_requests_total[1h]))
      
      # 可靠性指标
      - record: llm:error:rate
        expr: sum(rate(llm_requests_failed_total[5m])) / sum(rate(llm_requests_total[5m]))
      
      - record: llm:retry:rate
        expr: sum(rate(llm_retries_total[5m])) / sum(rate(llm_requests_total[5m]))
```

## 四、分布式追踪实现

### 4.1 LLM调用链路追踪

LLM应用的调用链路比传统应用更复杂，需要追踪完整的思考过程：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM分布式追踪示例                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Trace ID: abc123def456                                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Span: API Gateway                                          │   │
│  │  Duration: 12.5s                                            │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Span: Authentication                                │   │   │
│  │  │  Duration: 15ms                                      │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Span: Prompt Processing                             │   │   │
│  │  │  Duration: 50ms                                      │   │   │
│  │  │  Attributes:                                         │   │   │
│  │  │    - prompt.tokens: 1250                             │   │   │
│  │  │    - system_prompt.tokens: 800                       │   │   │
│  │  │    - user_input.tokens: 450                          │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Span: RAG Retrieval                                 │   │   │
│  │  │  Duration: 200ms                                     │   │   │
│  │  │  Attributes:                                         │   │   │
│  │  │    - vector_db.query_time: 180ms                     │   │   │
│  │  │    - chunks_retrieved: 5                             │   │   │
│  │  │    - relevance_scores: [0.92, 0.87, 0.85, 0.81, 0.78]│  │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Span: LLM Inference (OpenAI GPT-4o)                 │   │   │
│  │  │  Duration: 11.8s                                     │   │   │
│  │  │  Attributes:                                         │   │   │
│  │  │    - model: gpt-4o                                   │   │   │
│  │  │    - input_tokens: 2500                              │   │   │
│  │  │    - output_tokens: 850                              │   │   │
│  │  │    - ttft: 1.2s                                      │   │   │
│  │  │    - tps: 72 tokens/s                                │   │   │
│  │  │    - cost: $0.025                                    │   │   │
│  │  │    - finish_reason: stop                             │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Span: Output Validation                             │   │   │
│  │  │  Duration: 150ms                                     │   │   │
│  │  │  Attributes:                                         │   │   │
│  │  │    - format_valid: true                              │   │   │
│  │  │    - safety_score: 0.98                              │   │   │
│  │  │    - hallucination_check: pass                       │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 OpenTelemetry集成

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import time
from typing import Dict, Any, Optional

# 初始化Tracer
resource = Resource.create({
    "service.name": "llm-gateway",
    "service.version": "1.0.0",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)


class LLMTracer:
    """LLM调用追踪器"""
    
    def __init__(self):
        self.tracer = trace.get_tracer("llm-tracer")
    
    def trace_llm_call(
        self,
        model: str,
        messages: list,
        kwargs: Dict[str, Any] = None
    ):
        """追踪LLM调用"""
        with self.tracer.start_as_current_span("llm.inference") as span:
            # 记录请求属性
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.input.messages.count", len(messages))
            span.set_attribute("llm.input.tokens.estimated", self._estimate_tokens(messages))
            
            if kwargs:
                span.set_attribute("llm.temperature", kwargs.get("temperature", 0.7))
                span.set_attribute("llm.max_tokens", kwargs.get("max_tokens", 2048))
            
            start_time = time.time()
            
            try:
                # 实际调用LLM (这里用占位符)
                result = self._call_llm(model, messages, kwargs or {})
                
                # 记录响应属性
                latency = time.time() - start_time
                span.set_attribute("llm.latency.total", latency)
                span.set_attribute("llm.output.tokens", result.get("usage", {}).get("completion_tokens", 0))
                span.set_attribute("llm.output.finish_reason", result.get("finish_reason", "unknown"))
                
                # 计算成本
                cost = self._calculate_cost(
                    model,
                    result.get("usage", {}).get("prompt_tokens", 0),
                    result.get("usage", {}).get("completion_tokens", 0)
                )
                span.set_attribute("llm.cost.dollars", cost)
                
                # 添加事件
                span.add_event("llm.response.completed", {
                    "latency": latency,
                    "tokens": result.get("usage", {}).get("completion_tokens", 0),
                })
                
                return result
                
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.add_event("llm.error", {
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                })
                raise
    
    def trace_rag_retrieval(
        self,
        query: str,
        results: list,
        metadata: Dict = None
    ):
        """追踪RAG检索"""
        with self.tracer.start_as_current_span("rag.retrieval") as span:
            span.set_attribute("rag.query.length", len(query))
            span.set_attribute("rag.results.count", len(results))
            
            if results:
                scores = [r.get("score", 0) for r in results]
                span.set_attribute("rag.results.scores.min", min(scores))
                span.set_attribute("rag.results.scores.max", max(scores))
                span.set_attribute("rag.results.scores.avg", sum(scores) / len(scores))
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"rag.metadata.{key}", str(value))
    
    def trace_output_validation(
        self,
        output: str,
        validation_results: Dict[str, bool]
    ):
        """追踪输出验证"""
        with self.tracer.start_as_current_span("llm.output.validation") as span:
            span.set_attribute("llm.output.length", len(output))
            span.set_attribute("llm.output.validation.all_passed", all(validation_results.values()))
            
            for check, passed in validation_results.items():
                span.set_attribute(f"llm.output.validation.{check}", passed)
    
    def _estimate_tokens(self, messages: list) -> int:
        """估算Token数量"""
        # 简单估算: 1个token约4个字符
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算成本"""
        # 价格表 (每1K tokens)
        pricing = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        }
        
        model_pricing = pricing.get(model, {"input": 0.001, "output": 0.003})
        cost = (input_tokens * model_pricing["input"] + output_tokens * model_pricing["output"]) / 1000
        return cost
    
    def _call_llm(self, model: str, messages: list, kwargs: dict) -> dict:
        """实际LLM调用 (占位符)"""
        # 实际实现应调用真正的LLM API
        return {
            "choices": [{"message": {"content": "..."}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            "finish_reason": "stop"
        }
```

## 五、智能告警体系

### 5.1 告警规则设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM智能告警规则矩阵                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  级别: P0 (Critical) - 立即响应                              │   │
│  │                                                             │   │
│  │  • 成功率 < 90% (5分钟内)                                   │   │
│  │  • 全部模型不可用                                            │   │
│  │  • 成本异常飙升 (> 日均3倍)                                 │   │
│  │  • 安全事件 (PII泄露/注入攻击)                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  级别: P1 (High) - 1小时内处理                               │   │
│  │                                                             │   │
│  │  • P99延迟 > 30s (持续5分钟)                                │   │
│  │  • 幻觉率 > 10% (1小时内)                                   │   │
│  │  • 单模型错误率 > 20%                                       │   │
│  │  • 熔断器打开                                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  级别: P2 (Medium) - 4小时内处理                             │   │
│  │                                                             │   │
│  │  • P95延迟 > 15s (持续10分钟)                               │   │
│  │  • 重试率 > 10% (1小时内)                                   │   │
│  │  • 降级触发                                                   │   │
│  │  • Token使用量异常 (> 日均2倍)                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  级别: P3 (Low) - 下个工作日处理                             │   │
│  │                                                             │   │
│  │  • 质量指标轻微下降                                          │   │
│  │  • 成本趋势上升                                              │   │
│  │  • 非核心功能性能下降                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 告警实现

```yaml
# alertmanager-rules.yaml
groups:
  - name: llm_alerts
    rules:
      # P0: 成功率暴跌
      - alert: LLMSuccessRateCritical
        expr: sum(rate(llm_requests_successful_total[5m])) / sum(rate(llm_requests_total[5m])) < 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM成功率严重下降"
          description: "当前成功率: {{ $value | humanizePercentage }}"
          runbook: "检查所有LLM提供商状态，必要时触发全量降级"
      
      # P0: 成本异常
      - alert: LLMCostAnomaly
        expr: sum(rate(llm_cost_dollars_total[1h])) > 3 * sum(rate(llm_cost_dollars_total[1h] offset 1d))
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM成本异常飙升"
          description: "当前小时成本是日均的 {{ $value }} 倍"
          runbook: "检查是否有异常调用循环或恶意请求"
      
      # P1: 延迟过高
      - alert: LLMLatencyHigh
        expr: histogram_quantile(0.99, rate(llm_ttft_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "LLM P99延迟超过30秒"
          description: "当前P99延迟: {{ $value }}s"
          runbook: "检查模型服务状态，考虑切换备用模型"
      
      # P1: 幻觉率过高
      - alert: LLMHallucinationRateHigh
        expr: sum(rate(llm_hallucination_detected_total[1h])) / sum(rate(llm_requests_total[1h])) > 0.1
        for: 30m
        labels:
          severity: high
        annotations:
          summary: "LLM幻觉率超过10%"
          description: "当前幻觉率: {{ $value | humanizePercentage }}"
          runbook: "检查Prompt质量，考虑增加RAG检索数量"
      
      # P2: 重试率过高
      - alert: LLMRetryRateHigh
        expr: sum(rate(llm_retries_total[5m])) / sum(rate(llm_requests_total[5m])) > 0.1
        for: 10m
        labels:
          severity: medium
        annotations:
          summary: "LLM重试率超过10%"
          description: "当前重试率: {{ $value | humanizePercentage }}"
          runbook: "检查网络状况和API限流情况"
```

## 六、成本分析与优化

### 6.1 成本归因模型

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM成本归因分析                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  按维度归因                                                   │   │
│  │                                                             │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  按模型:                                              │   │   │
│  │  │  ├── GPT-4o:        $1,250 (45%)                    │   │   │
│  │  │  ├── Claude-3:      $800 (29%)                      │   │   │
│  │  │  ├── GPT-4o-mini:   $450 (16%)                      │   │   │
│  │  │  └── 本地模型:      $300 (10%)                      │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                                                             │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  按功能:                                              │   │   │
│  │  │  ├── 智能客服:    $900 (32%)                        │   │   │
│  │  │  ├── 内容生成:    $700 (25%)                        │   │   │
│  │  │  ├── 代码助手:    $600 (21%)                        │   │   │
│  │  │  └── 数据分析:    $600 (22%)                        │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                                                             │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  按用户:                                              │   │   │
│  │  │  ├── Top 10%用户: $1,400 (50%)  ← 关注ROI           │   │   │
│  │  │  ├── 中间用户:    $840 (30%)                         │   │   │
│  │  │  └── 长尾用户:    $560 (20%)                         │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  成本优化机会                                                 │   │
│  │                                                             │   │
│  │  1. Prompt压缩: 减少20%输入Token → 节省$200/月             │   │
│  │  2. 模型降级: 简单任务用mini模型 → 节省$300/月              │   │
│  │  3. 缓存复用: 相似请求复用结果 → 节省$150/月                │   │
│  │  4. 批量处理: 非实时请求批量处理 → 节省$100/月              │   │
│  │                                                             │   │
│  │  预估总节省: $750/月 (27%)                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 成本监控代码

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class LLMCostRecord:
    timestamp: datetime
    model: str
    feature: str
    user_id: str
    input_tokens: int
    output_tokens: int
    cost_dollars: float

class LLMCostAnalyzer:
    """LLM成本分析器"""
    
    def __init__(self):
        self.records: List[LLMCostRecord] = []
    
    def record_cost(self, record: LLMCostRecord):
        """记录成本"""
        self.records.append(record)
    
    def get_cost_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        group_by: str = "model"
    ) -> Dict[str, float]:
        """获取成本汇总"""
        filtered = [
            r for r in self.records
            if start_time <= r.timestamp <= end_time
        ]
        
        costs = defaultdict(float)
        for record in filtered:
            key = getattr(record, group_by)
            costs[key] += record.cost_dollars
        
        return dict(costs)
    
    def get_cost_trend(
        self,
        days: int = 7,
        group_by: str = "model"
    ) -> Dict[str, List[float]]:
        """获取成本趋势"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        trend = defaultdict(list)
        current_date = start_date
        
        while current_date <= end_date:
            day_end = current_date + timedelta(days=1)
            daily_costs = self.get_cost_summary(current_date, day_end, group_by)
            
            for key, cost in daily_costs.items():
                trend[key].append(cost)
            
            current_date = day_end
        
        return dict(trend)
    
    def detect_cost_anomaly(
        self,
        current_cost: float,
        historical_avg: float,
        threshold: float = 3.0
    ) -> bool:
        """检测成本异常"""
        if historical_avg == 0:
            return False
        return current_cost > historical_avg * threshold
    
    def get_optimization_suggestions(self) -> List[Dict]:
        """获取成本优化建议"""
        suggestions = []
        
        # 分析模型使用情况
        model_costs = self.get_cost_summary(
            datetime.now() - timedelta(days=30),
            datetime.now(),
            "model"
        )
        
        # 检查是否有过度使用昂贵模型
        if model_costs.get("gpt-4o", 0) > model_costs.get("gpt-4o-mini", 0) * 5:
            suggestions.append({
                "type": "model_downgrade",
                "description": "考虑将简单任务从GPT-4o降级到GPT-4o-mini",
                "estimated_savings": model_costs.get("gpt-4o", 0) * 0.3
            })
        
        # 检查功能成本
        feature_costs = self.get_cost_summary(
            datetime.now() - timedelta(days=30),
            datetime.now(),
            "feature"
        )
        
        for feature, cost in feature_costs.items():
            if cost > 100:  # 阈值
                suggestions.append({
                    "type": "feature_optimization",
                    "feature": feature,
                    "description": f"功能 '{feature}' 成本较高，考虑优化Prompt或添加缓存",
                    "estimated_savings": cost * 0.2
                })
        
        return suggestions
```

## 七、Grafana Dashboard设计

### 7.1 核心面板布局

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM监控Dashboard布局                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Row 1: 核心指标概览                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ 总请求数  │ │ 成功率   │ │ 平均延迟  │ │ 今日成本  │ │ 幻觉率   ││
│  │ 12.5K    │ │ 99.2%   │ │ 4.2s     │ │ $85.20  │ │ 2.1%    ││
│  │ ▲ 12%    │ │ ▼ 0.3%  │ │ ▲ 0.5s   │ │ ▲ $12   │ │ ─ 0%    ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
│                                                                      │
│  Row 2: 延迟与吞吐                                                   │
│  ┌────────────────────────────┐ ┌────────────────────────────┐    │
│  │  延迟分布 (P50/P95/P99)    │ │  吞吐量趋势 (tokens/s)     │    │
│  │  [时间序列图]              │ │  [时间序列图]               │    │
│  └────────────────────────────┘ └────────────────────────────┘    │
│                                                                      │
│  Row 3: 成本分析                                                     │
│  ┌────────────────────────────┐ ┌────────────────────────────┐    │
│  │  成本按模型分布            │ │  成本趋势 (7天)            │    │
│  │  [饼图/柱状图]             │ │  [面积图]                  │    │
│  └────────────────────────────┘ └────────────────────────────┘    │
│                                                                      │
│  Row 4: 质量与可靠性                                                 │
│  ┌────────────────────────────┐ ┌────────────────────────────┐    │
│  │  错误类型分布              │ │  重试/降级事件              │    │
│  │  [柱状图]                  │ │  [事件列表]                 │    │
│  └────────────────────────────┘ └────────────────────────────┘    │
│                                                                      │
│  Row 5: 模型对比                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  模型性能对比 (延迟/成本/质量 矩阵)                         │   │
│  │  [散点图/热力图]                                            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 八、生产部署检查清单

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM可观测性部署检查清单                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ✅ 数据采集                                                         │
│  □ SDK/Proxy埋点就绪                                                │
│  □ 关键属性定义 (model, tokens, latency, cost)                      │
│  □ 采样策略配置 (建议: 生产环境100%, 开发环境10%)                    │
│  □ 敏感数据脱PII处理                                                │
│                                                                      │
│  ✅ 存储与查询                                                       │
│  □ 时序数据库就绪 (Prometheus/VictoriaMetrics)                      │
│  □ 日志存储就绪 (ClickHouse/Elasticsearch)                          │
│  □ 追踪存储就绪 (Jaeger/Tempo)                                      │
│  □ 数据保留策略 (建议: 指标90天, 日志30天, 追踪7天)                  │
│                                                                      │
│  ✅ 告警配置                                                         │
│  □ P0-P3告警规则定义                                                │
│  □ 告警通知渠道 (PagerDuty/Slack/企业微信)                          │
│  □ 告警升级策略                                                     │
│  □ 告警抑制规则 (避免告警风暴)                                       │
│                                                                      │
│  ✅ 可视化                                                           │
│  □ Grafana Dashboard就绪                                           │
│  □ 核心指标概览面板                                                 │
│  □ 成本分析面板                                                     │
│  □ 模型对比面板                                                     │
│                                                                      │
│  ✅ 成本管理                                                         │
│  □ 成本归因模型就绪                                                 │
│  □ 成本异常检测                                                     │
│  □ 成本报表自动生成                                                 │
│                                                                      │
│  ✅ 运维流程                                                         │
│  □ 值班手册更新 (包含LLM特有排障步骤)                                │
│  □ 故障演练 (每月一次)                                              │
│  □ 指标审查会议 (每周一次)                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 九、总结

LLM可观测性不是传统监控的简单扩展，而是需要全新思维的系统工程。核心要点：

1. **指标体系要专门设计**：Token使用量、首Token延迟、幻觉率、成本分析，这些是LLM应用特有的核心指标。

2. **分布式追踪要完整**：从RAG检索到LLM推理，再到输出验证，每一步都需要追踪，才能快速定位问题。

3. **告警要智能分级**：LLM应用的错误类型多样，需要按严重程度分级告警，避免告警疲劳。

4. **成本监控是刚需**：LLM的Token计费模式要求精细的成本归因和异常检测，否则很容易失控。

5. **质量监控不可少**：幻觉率、安全性、相关性，这些质量指标直接影响用户体验和业务价值。

建设LLM可观测性平台是一个渐进过程，建议从Proxy采集开始，逐步完善指标、告警和可视化。记住：**可观测性的目标不是收集更多数据，而是更快地发现问题、定位问题、解决问题。**
