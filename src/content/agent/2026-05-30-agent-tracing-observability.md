---
title: "Agent全链路Tracing与Observability：从日志到智能监控"
description: "深入探讨Agent可观测性体系：Span设计、LangSmith/Phoenix/OpenTelemetry集成、关键监控指标、智能告警与Dashboard设计，以及支持大规模Agent集群的监控系统架构"
date: 2026-05-30
author: RiceBall-15
category: agent
subCategory: agent-dev
tags: ["Tracing", "Observability", "LangSmith", "Phoenix", "OpenTelemetry"]
draft: false
---

## 简介

当一个AI Agent在生产环境中运行时，开发者面临的最大挑战不是让它"能跑"，而是让它"跑得可观测"。Agent的推理过程涉及LLM调用、工具执行、多步决策，传统的日志系统在这种复杂调用链面前显得力不从心。本文系统性地构建Agent全链路可观测性体系——从基础的Span设计到智能监控告警，从单Agent调试到百Agent集群运维，为生产级Agent系统提供完整的可观测性方案。

---

## 1. 为什么Agent需要Tracing：传统日志的不足

### 1.1 Agent执行的复杂性

一个典型Agent的执行链路远比传统微服务复杂：

```
用户请求 → LLM决策 → 工具调用 → 结果评估 → LLM再决策 → ...
```

每次LLM决策都可能触发不同的工具调用，形成**动态的、非确定性的调用树**。传统日志的问题在于：

**信息碎片化**：每个环节的日志是独立的。要追踪一个用户请求的完整链路，需要手动关联多条日志记录，耗时且易出错。

**缺乏因果关系**：日志记录"LLM返回了什么"和"工具执行了什么"，但无法回答"为什么LLM做了这个决策"——因为决策依据（Prompt上下文）未被记录。

**无法回溯推理过程**：当Agent给出错误答案时，传统日志无法完整回放Agent的思考链路，无法定位是LLM幻觉、工具错误还是策略缺陷。

### 1.2 Agent特有挑战

| 挑战 | 传统微服务 | Agent系统 |
|------|-----------|----------|
| 调用链 | 确定性路由 | 动态决策路由 |
| 延迟来源 | 网络+计算 | LLM推理（高延迟）+工具 |
| 状态管理 | 无状态/有状态 | 持续对话上下文 |
| 错误类型 | 异常/超时 | 幻觉/循环/策略失败 |
| 可预测性 | 高 | 低（非确定性） |

Tracing通过**将每次Agent执行组织为一棵完整的Span树**，让每个环节的输入、输出、耗时、关联关系一目了然。

---

## 2. 可观测性三大支柱在Agent中的应用

### 2.1 Metrics（指标）

Metrics回答"系统整体运行得怎么样"：

```python
# Agent关键指标定义
METRICS = {
    "agent_request_total": Counter("agent_requests_total", 
        ["agent_type", "status", "tool_used"]),
    "llm_latency_seconds": Histogram("llm_latency_seconds",
        ["model", "operation"], buckets=[0.5, 1, 2, 5, 10, 30]),
    "token_usage": Counter("token_usage_total",
        ["model", "type"]),  # type: prompt/completion
    "agent_loop_count": Histogram("agent_loop_count",
        ["agent_type"], buckets=[1, 2, 3, 5, 10, 20]),
    "tool_success_rate": Gauge("tool_success_rate",
        ["tool_name"]),
}
```

### 2.2 Logs（日志）

Logs回答"某个具体请求发生了什么"：

```python
import structlog

logger = structlog.get_logger()

async def execute_tool(tool_name: str, args: dict, trace_id: str):
    log = logger.bind(
        trace_id=trace_id,
        tool_name=tool_name,
        args_hash=hash(frozenset(args.items()))
    )
    log.info("tool_execution_start")
    try:
        result = await tools[tool_name].execute(args)
        log.info("tool_execution_success", 
                 result_length=len(str(result)))
        return result
    except Exception as e:
        log.error("tool_execution_failed", error=str(e))
        raise
```

### 2.3 Traces（链路追踪）

Traces回答"这次请求从头到尾经历了什么"：

```
[Trace: user_request_abc123]
├── [Span: agent.think] (2.3s)
│   ├── [Span: llm.chat] (1.8s, tokens=2048)
│   └── [Span: llm.parse_response] (0.02s)
├── [Span: tool.web_search] (0.8s)
│   └── [Span: http.request] (0.7s)
├── [Span: agent.think] (1.9s)
│   ├── [Span: llm.chat] (1.5s, tokens=1536)
│   └── [Span: llm.parse_response] (0.01s)
└── [Span: agent.respond] (0.1s)
```

三者协同：Metrics提供宏观趋势，Traces提供请求级链路，Logs提供细节上下文。通过**Trace ID贯穿三者**，实现从宏观到微观的完整可观测性。

---

## 3. Span设计：Agent调用链的Span划分

Span划分是Tracing的核心设计决策。对Agent而言，需要覆盖三种关键操作：

### 3.1 Span层级架构

```
Agent Trace Span层级设计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Level 0: agent.execution          # 整个Agent执行
├── Level 1: agent.reason         # 推理阶段（可多轮）
│   ├── Level 2: llm.inference    # LLM调用
│   │   ├── input: prompt_messages
│   │   ├── output: response_text
│   │   ├── attributes: model, token_count, temperature
│   │   └── events: [stream_start, stream_end]
│   └── Level 2: agent.decision   # 决策解析
│       ├── output: action_type, tool_name
│       └── attributes: reasoning_chain
├── Level 1: agent.act            # 执行阶段
│   ├── Level 2: tool.invocation  # 工具调用
│   │   ├── Level 3: http.request # 具体HTTP调用
│   │   └── attributes: tool_name, input_schema
│   └── Level 2: tool.validate    # 结果验证
└── Level 1: agent.reflect        # 反思/评估
    └── Level 2: llm.evaluation   # 评估LLM调用
```

### 3.2 核心Span定义与实现

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer_provider = TracerProvider()
tracer = tracer_provider.get_tracer("agent-system")

class AgentTracer:
    """Agent专用Tracing封装"""
    
    def __init__(self, agent_id: str, user_id: str):
        self.agent_id = agent_id
        self.user_id = user_id
    
    def start_agent_trace(self, request: str):
        with tracer.start_as_current_span("agent.execution") as span:
            span.set_attribute("agent.id", self.agent_id)
            span.set_attribute("user.id", self.user_id)
            span.set_attribute("input.request", request[:500])
            return span
    
    def trace_llm_call(self, model: str, messages: list):
        span = tracer.start_span("llm.inference")
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.input_tokens", count_tokens(messages))
        span.set_attribute("llm.temperature", 0.7)
        return span
    
    def trace_tool_call(self, tool_name: str, args: dict):
        span = tracer.start_span("tool.invocation")
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.input", json.dumps(args)[:1000])
        return span
    
    def trace_agent_decision(self, decision: dict):
        span = tracer.start_span("agent.decision")
        span.set_attribute("agent.action", decision["action"])
        span.set_attribute("agent.confidence", decision.get("confidence"))
        return span
```

### 3.3 Span属性设计原则

- **LLM Span**：记录model、prompt_tokens、completion_tokens、temperature、top_p
- **Tool Span**：记录tool_name、input参数摘要、output摘要、status_code
- **Decision Span**：记录决策类型、置信度、推理链摘要
- **错误事件**：使用Span Event记录异常堆栈和上下文

---

## 4. LangSmith：Tracing集成、评估与在线调试

### 4.1 快速集成

LangSmith是LangChain官方的可观测性平台，对LangChain/LangGraph构建的Agent有原生支持：

```python
import os
from langsmith import traceable
from langsmith.run_trees import RunTree

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

@traceable(
    name="agent_reasoning",
    run_type="chain",
    tags=["agent-v2", "production"],
    metadata={"model": "gpt-4o", "max_iterations": 10}
)
async def agent_reason(context: str, tools: list) -> str:
    """推理步骤自动被LangSmith追踪"""
    llm_response = await call_llm_with_tracing(context)
    tool_call = parse_tool_call(llm_response)
    
    if tool_call:
        # 嵌套Span自动关联
        tool_result = await execute_tool_traced(tool_call)
        return await agent_reason(
            f"Previous: {llm_response}\nTool Result: {tool_result}",
            tools
        )
    return llm_response

@traceable(name="llm_call", run_type="llm")
async def call_llm_with_tracing(context: str):
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": context}]
    )
    return response.choices[0].message.content
```

### 4.2 评估与在线调试

LangSmith的核心价值在于**评估闭环**：

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 定义评估函数
def correctness(run, example):
    """检查Agent回答的正确性"""
    expected = example.outputs["answer"]
    actual = run.outputs["response"]
    score = compute_similarity(expected, actual)
    return {"key": "correctness", "score": score}

# 在线上trace上运行评估
results = evaluate(
    agent_reason,
    data="production-eval-dataset",
    evaluators=[correctness],
    experiment_prefix="agent-v3-eval"
)

# 在线调试：对特定trace进行分析
trace = client.read_run("run-id-xxx")
print(f"Latency: {trace.end_time - trace.start_time}")
print(f"Token usage: {trace.extra.get('token_usage')}")
```

### 4.3 LangSmith核心能力

- **Trace Timeline**：可视化Agent执行的完整时间线
- **Dataset管理**：构建测试集，持续评估Agent质量
- **Online Debugging**：生产环境trace的实时查看和过滤
- **A/B Testing**：对比不同Agent策略的效果

---

## 5. Arize Phoenix：开源LLM可观测性

### 5.1 核心特性

Arize Phoenix是开源的LLM可观测性平台，特别擅长**Embedding漂移检测**和**质量退化监控**。

```python
import phoenix as px
from phoenix.otel import register

# 启动Phoenix服务
px.launch_app()

# 注册Tracer Provider
tracer_provider = register(project_name="agent-monitoring")

# 自动Instrument LangChain
from phoenix.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

### 5.2 Embedding漂移检测

Embedding漂移是Agent系统的隐形杀手——当用户查询的语义分布发生变化时，RAG检索质量会静默退化：

```python
from phoenix.embedding_embeddings import EmbeddingDriftDetector

# 定义Embedding漂移检测器
drift_detector = EmbeddingDriftDetector(
    baseline_dataset="production-embeddings-week1",
    current_dataset="production-embeddings-week2",
    dimensionality_reduction="umap"  # 使用UMAP降维可视化
)

# 检测漂移
drift_result = drift_detector.detect()
if drift_result.is_drifting:
    print(f"Embedding漂移检测到！KL散度: {drift_result.kl_divergence}")
    print(f"漂移区域: {drift_result.drift_clusters}")
    # 触发Embedding模型微调或Prompt调整
    trigger_retraining(drift_result)
```

### 5.3 质量评估指标

Phoenix自动计算以下质量指标：

```python
# LLM质量评估
quality_metrics = {
    "relevance_score": px.compute_relevance(
        query_embeddings, retrieval_embeddings
    ),
    "hallucination_rate": px.detect_hallucination(
        context_chunks, llm_response
    ),
    "answer_faithfulness": px.compute_faithfulness(
        llm_response, reference_context
    )
}
```

Phoenix的开源优势在于可深度定制，适合需要私有化部署或有特殊合规要求的企业。

---

## 6. OpenTelemetry集成：自定义Span属性与上下文传播

### 6.1 OpenTelemetry基础配置

OpenTelemetry是可观测性的标准协议，提供最大的灵活性：

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# 配置资源属性
resource = Resource.create({
    "service.name": "agent-service",
    "service.version": "2.1.0",
    "deployment.environment": "production",
    "agent.cluster": "main"
})

# 配置TracerProvider
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="http://collector:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
```

### 6.2 自定义Span属性

```python
from opentelemetry import trace
from opentelemetry.trace import StatusCode

tracer = trace.get_tracer("agent-service")

async def agent_loop(user_query: str, agent_config: dict):
    with tracer.start_as_current_span("agent.loop") as loop_span:
        # 设置Agent维度属性
        loop_span.set_attribute("agent.type", agent_config["type"])
        loop_span.set_attribute("agent.max_iterations", 
                                agent_config["max_iter"])
        loop_span.set_attribute("user.id", agent_config["user_id"])
        loop_span.set_attribute("session.id", agent_config["session_id"])
        
        iteration = 0
        while iteration < agent_config["max_iter"]:
            iteration += 1
            
            # LLM调用Span
            with tracer.start_as_current_span("agent.llm_call") as llm_span:
                llm_span.set_attribute("llm.iteration", iteration)
                llm_span.set_attribute("llm.model", "gpt-4o")
                
                response = await call_llm(user_query)
                
                llm_span.set_attribute("llm.output_tokens", 
                                       response.usage.completion_tokens)
                llm_span.set_attribute("llm.input_tokens", 
                                       response.usage.prompt_tokens)
                llm_span.set_attribute("llm.total_cost_usd",
                    compute_cost(response.usage))
            
            # 工具调用Span（如有）
            if needs_tool(response):
                with tracer.start_as_current_span("agent.tool_call") as tool_span:
                    tool_span.set_attribute("tool.name", extract_tool_name(response))
                    tool_span.set_attribute("tool.iteration", iteration)
                    
                    result = await execute_tool(response)
                    tool_span.set_attribute("tool.success", result is not None)
            
            # 记录Agent决策Event
            loop_span.add_event("agent.decision", {
                "iteration": iteration,
                "action": "continue" if needs_tool(response) else "finish",
                "reasoning": response.reasoning[:200]
            })
            
            if not needs_tool(response):
                break
        
        loop_span.set_attribute("agent.total_iterations", iteration)
        loop_span.set_status(StatusCode.OK)
```

### 6.3 跨服务上下文传播

在多Agent协作场景中，Trace Context需要跨服务传播：

```python
from opentelemetry.context.propagation import TraceContextTextMapPropagator
from opentelemetry import context

async def invoke_remote_agent(target_agent: str, query: str):
    # 提取当前Trace Context
    propagator = TraceContextTextMapPropagator()
    carrier = {}
    propagator.inject(carrier)  # 将trace context注入carrier
    
    # 通过HTTP Header传播到下游Agent
    response = await http_client.post(
        f"http://{target_agent}/invoke",
        json={"query": query},
        headers=carrier  # 包含traceparent等Header
    )
    return response.json()
```

---

## 7. 关键监控指标

### 7.1 Agent核心指标矩阵

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# ━━━━━━ 延迟指标 ━━━━━━
agent_latency = Histogram(
    "agent_end_to_end_latency_seconds",
    "端到端Agent执行延迟",
    ["agent_type", "complexity"],
    buckets=[1, 2, 5, 10, 30, 60, 120]
)

llm_first_token_latency = Histogram(
    "llm_time_to_first_token_seconds",
    "LLM首Token延迟",
    ["model"],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

# ━━━━━━ 吞吐指标 ━━━━━━
requests_total = Counter(
    "agent_requests_total",
    "Agent请求总数",
    ["agent_type", "status"]
)

concurrent_requests = Gauge(
    "agent_concurrent_requests",
    "当前并发Agent执行数"
)

# ━━━━━━ Token消耗 ━━━━━━
token_usage = Counter(
    "agent_token_usage_total",
    "Token消耗总量",
    ["model", "token_type"]  # prompt/completion
)

token_cost_usd = Counter(
    "agent_token_cost_usd_total",
    "Token成本（美元）",
    ["model"]
)

# ━━━━━━ 工具指标 ━━━━━━
tool_success_rate = Gauge(
    "agent_tool_success_rate",
    "工具调用成功率",
    ["tool_name"],
    registry=REGISTRY
)

tool_latency = Histogram(
    "agent_tool_latency_seconds",
    "工具调用延迟",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5]
)

# ━━━━━━ Agent循环指标 ━━━━━━
loop_count = Histogram(
    "agent_loop_count",
    "Agent执行循环次数",
    ["agent_type"],
    buckets=[1, 2, 3, 5, 10, 15, 20]
)

max_loop_reached = Counter(
    "agent_max_loop_reached_total",
    "达到最大循环次数的次数"
)
```

### 7.2 指标计算规则

| 指标 | 计算方式 | 告警阈值 |
|------|---------|---------|
| P50/P95/P99延迟 | 端到端执行时间 | P95 > 30s |
| 首Token延迟 | LLM首次输出时间 | > 3s |
| Token消耗/请求 | prompt + completion | > 10000 |
| 工具成功率 | 成功/总调用 | < 95% |
| Agent循环次数 | 完成前的推理轮数 | > 15 |
| 并发Agent数 | 当前执行中的Agent | > 500 |

---

## 8. 智能告警

### 8.1 多层告警策略

```python
from prometheus_client import ALERT_RULES

# 基于阈值的即时告警
ALERT_RULES = {
    "high_latency": {
        "condition": "agent_p95_latency > 30s for 5m",
        "severity": "warning",
        "action": "notify_oncall"
    },
    "high_error_rate": {
        "condition": "agent_error_rate > 0.1 for 3m",
        "severity": "critical",
        "action": "page_oncall + auto_rollback"
    },
    "cost_spike": {
        "condition": "token_cost_hourly > 2x daily_avg for 1h",
        "severity": "warning",
        "action": "notify_team + throttle_requests"
    },
    "loop_storm": {
        "condition": "avg(agent_loop_count) > 10 for 10m",
        "severity": "critical",
        "action": "notify_oncall + reduce_max_iterations"
    }
}
```

### 8.2 异常检测（基于统计）

```python
import numpy as np
from collections import deque

class AnomalyDetector:
    """基于Z-Score的实时异常检测"""
    
    def __init__(self, window_size=100, threshold=3.0):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold
    
    def update(self, value: float) -> dict:
        self.window.append(value)
        
        if len(self.window) < 20:
            return {"is_anomaly": False, "reason": "insufficient_data"}
        
        arr = np.array(self.window)
        mean, std = arr.mean(), arr.std()
        z_score = (value - mean) / max(std, 1e-6)
        
        return {
            "is_anomaly": abs(z_score) > self.threshold,
            "z_score": z_score,
            "mean": mean,
            "std": std,
            "direction": "high" if z_score > 0 else "low"
        }

# 使用示例
latency_detector = AnomalyDetector(window_size=200, threshold=2.5)

async def on_agent_complete(latency: float, token_count: int):
    # 延迟异常检测
    latency_result = latency_detector.update(latency)
    if latency_result["is_anomaly"]:
        await alert(
            severity="warning",
            message=f"延迟异常: {latency:.2f}s "
                    f"(均值{latency_result['mean']:.2f}s, "
                    f"Z={latency_result['z_score']:.2f})",
            metrics={"latency": latency, **latency_result}
        )
    
    # 成本阈值检测
    cost = compute_cost(token_count)
    if cost > DAILY_COST_THRESHOLD * 2:
        await alert(
            severity="critical",
            message=f"单次请求成本异常: ${cost:.4f}",
            action="throttle"
        )
```

### 8.3 趋势预警

```python
class TrendPredictor:
    """基于线性回归的趋势预警"""
    
    def predict_cost_24h(self, hourly_costs: list) -> dict:
        x = np.arange(len(hourly_costs))
        slope, intercept = np.polyfit(x, hourly_costs, 1)
        
        predicted_24h = intercept + slope * (len(hourly_costs) + 24)
        current_daily = sum(hourly_costs[-24:])
        
        return {
            "current_daily": current_daily,
            "predicted_daily": predicted_24h,
            "trend": "increasing" if slope > 0.01 else "stable",
            "recommendation": (
                "预计24h成本将超标，建议调整策略" 
                if predicted_24h > DAILY_BUDGET 
                else "成本在预算范围内"
            )
        }
```

---

## 9. Dashboard设计

### 9.1 架构概览

```
Agent监控Dashboard架构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────┐
│              Dashboard UI               │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Overview │ │ Traces   │ │ Alerts  │ │
│  │  Panel   │ │  Panel   │ │  Panel  │ │
│  └────┬─────┘ └────┬─────┘ └────┬────┘ │
│       │            │            │       │
│  ┌────┴────────────┴────────────┴────┐  │
│  │         Query Engine              │  │
│  │    (PromQL + TraceQL + LogQL)    │  │
│  └────┬────────────┬────────────┬────┘  │
└───────┼────────────┼────────────┼───────┘
        │            │            │
┌───────┴──┐  ┌──────┴──┐  ┌────┴──────┐
│Prometheus │  │  Jaeger │  │   Loki   │
│ (Metrics) │  │ (Traces)│  │  (Logs)  │
└──────────┘  └─────────┘  └──────────┘
```

### 9.2 Dashboard核心面板

**面板1：Agent运行概览**

```
┌──────────────────────────────────────────────┐
│  🟢 Agent集群健康状态                        │
│                                              │
│  总Agent数: 42/50 (运行中)   活跃请求: 128   │
│  今日请求: 12,450     成功率: 98.2%          │
│  今日Token: 24.5M    今日成本: $82.30        │
│                                              │
│  ┌───延迟分布(P95)──────────────────────┐    │
│  │ ████░░░░ 8.2s (正常 < 15s)           │    │
│  └──────────────────────────────────────┘    │
│  ┌───Token消耗趋势(24h)────────────────┐    │
│  │  /\  /\    /\                        │    │
│  │ /  \/  \  /  \___                    │    │
│  │/         \/                          │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

**面板2：Agent Trace详情**

```
┌──────────────────────────────────────────────┐
│  Trace详情: agent_abc123                     │
│  Duration: 12.3s  Status: ✅ OK              │
│                                              │
│  ── Timeline ──────────────────────────────  │
│  0s     2s     4s     6s     8s    10s  12s  │
│  |======|==|==========|=====|====|====|      │
│  think  tool  think     tool think think     │
│  (2.1s) (0.5s)(3.2s)  (1.1s)(1.8s)(3.6s)   │
│                                              │
│  Token使用: P=4,200 C=1,800 ($0.032)        │
│  工具调用: web_search(2), code_exec(1)       │
│  循环次数: 4/10                              │
└──────────────────────────────────────────────┘
```

**面板3：告警与成本监控**

```
┌──────────────────────────────────────────────┐
│  📊 成本监控                                │
│  今日: $82.30  本周: $423.15  预算: $500/周  │
│  ████████████████░░░░░ 84.6%                 │
│                                              │
│  🔔 活跃告警 (3)                             │
│  ⚠️  [14:32] web_search工具成功率降至92%     │
│  ⚠️  [14:28] Agent-research延迟P95升至25s   │
│  ℹ️  [14:15] 日成本已达预算80%               │
└──────────────────────────────────────────────┘
```

### 9.3 实现技术选型

推荐组合：**Grafana + Prometheus + Jaeger + Loki**

```yaml
# docker-compose.yml
version: "3.8"
services:
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    volumes:
      - ./dashboards:/var/lib/grafana/dashboards
  
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports: ["16686:16686", "4317:4317"]
  
  loki:
    image: grafana/loki:latest
    ports: ["3100:3100"]
```

---

## 10. 面试深度：设计一个支持100个Agent的监控系统

### 10.1 系统架构设计

```
大规模Agent监控系统架构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    ┌─────────────────┐
                    │   Grafana UI    │
                    │  (统一Dashboard)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────┴──┐  ┌───────┴──┐  ┌───────┴──┐
    │ Prometheus │  │  Jaeger   │  │   Loki   │
    │ + Thanos   │  │ (Trace)  │  │  (Logs)  │
    │ (Metrics)  │  │          │  │          │
    └─────┬──────┘  └────┬─────┘  └────┬─────┘
          │              │              │
    ┌─────┴──────────────┴──────────────┴─────┐
    │           OpenTelemetry Collector        │
    │    (接收/处理/路由/采样/聚合)              │
    └─────┬──────────────┬──────────────┬─────┘
          │              │              │
   ┌──────┴──┐    ┌──────┴──┐    ┌──────┴──┐
   │  Agent  │    │  Agent  │    │  Agent  │
   │ Cluster │    │ Cluster │    │ Cluster │
   │  (33)   │    │  (33)   │    │  (34)   │
   └─────────┘    └─────────┘    └─────────┘
```

### 10.2 关键设计决策

**1. 分层采样策略**

100个Agent的全量Tracing数据量巨大，需要智能采样：

```python
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased, ParentBasedTraceIdRatio
)

class AgentSampler:
    """基于Agent特征的智能采样"""
    
    def __init__(self):
        self.base_rate = 0.1      # 基础采样率10%
        self.error_rate = 1.0     # 错误100%采样
        self.slow_rate = 1.0      # 慢请求100%采样
        self.new_agent_rate = 0.5  # 新Agent 50%采样
    
    def should_sample(self, context) -> bool:
        # 错误请求全采样
        if context.get("status") == "error":
            return True
        # 超时请求全采样
        if context.get("latency", 0) > 30:
            return True
        # 新上线Agent提高采样率
        if context.get("agent_age_days", 999) < 7:
            return random.random() < self.new_agent_rate
        # 默认采样
        return random.random() < self.base_rate
```

**2. OTel Collector层聚合**

```
100 Agents → OTel Collector(聚合层) → 后端存储

Collector处理：
├── Span批处理：每5秒或500条Span聚合一次
├── 属性标准化：统一Agent命名/标签
├── 采样决策：在Collector层统一采样，减少Agent端开销
├── 数据压缩：OTLP gRPC压缩传输
└── 路由分发：Metrics→Prometheus, Traces→Jaeger, Logs→Loki
```

**3. Agent元数据管理**

```python
# Agent注册表
agent_registry = {
    "agent-code-reviewer": {
        "team": "platform",
        "model": "gpt-4o",
        "avg_cost_per_request": 0.015,
        "max_concurrency": 20,
        "alert_contacts": ["team-platform@company.com"],
        "sla": {"p95_latency": 15, "success_rate": 0.95}
    },
    "agent-data-analyst": {
        "team": "analytics",
        "model": "claude-3.5-sonnet",
        "avg_cost_per_request": 0.025,
        "max_concurrency": 10,
        "alert_contacts": ["team-analytics@company.com"],
        "sla": {"p95_latency": 30, "success_rate": 0.90}
    }
    # ... 100个Agent配置
}
```

**4. 告警分层与收敛**

```
告警级别设计：
━━━━━━━━━━━━━━━

Level 1 (Agent级): 单个Agent异常
  → 通知Agent负责人
  → 自动降级（限制并发/切换模型）

Level 2 (团队级): 同一团队的Agent集体异常
  → 通知团队Tech Lead
  → 触发团队On-Call

Level 3 (平台级): 超过30%的Agent异常
  → 通知平台负责人
  → 触发全平台降级策略

告警收敛：
├── 相同Agent的连续告警：5分钟内合并
├── 相同团队的告警：聚合为团队级告警
├── 工作时间规则：非工作时间仅Level 3电话通知
└── 静默规则：已知维护窗口自动静默
```

### 10.3 性能与成本考量

| 维度 | 方案 | 预估指标 |
|------|------|---------|
| 存储 | Jaeger + S3归档 | 7天热数据 ~50GB, 30天归档 |
| 查询 | Prometheus分片 + Thanos | 100QPS聚合查询 |
| 采样 | 智能采样(均值10%) | 减少90%存储, 关键链路不丢 |
| 成本 | OTel Collector聚合 | 每Agent月成本 < $20 |
| 可用性 | Collector多副本 | 99.9%采集可用性 |

### 10.4 面试回答要点

> **"设计一个支持100个Agent的监控系统"**
>
> 1. **数据采集层**：每个Agent通过OTel SDK上报，OTel Collector集群接收并聚合，统一采样策略
> 2. **存储层**：Metrics存Prometheus+Thanos，Traces存Jaeger+Cassandra，Logs存Loki+S3
> 3. **分析层**：实时聚合Agent级、团队级、平台级指标，智能异常检测
> 4. **展示层**：Grafana统一Dashboard，分层视图（总览→团队→单Agent→单Trace）
> 5. **告警层**：三层告警（Agent/团队/平台），智能收敛，降级联动
> 6. **关键挑战**：数据量控制（智能采样）、跨Agent关联（Trace Context传播）、成本监控（Token消耗追踪）

---

## 总结

Agent可观测性不是"可选的锦上添花"，而是**生产级Agent系统的必要基础设施**。从Span设计的精心规划，到LangSmith/Phoenix/OpenTelemetry的工具选型，再到智能告警和大规模监控系统的架构设计——每一个环节都直接影响Agent系统的可靠性和可维护性。

核心原则：
- **Trace ID贯穿始终**：让每个请求可追踪
- **三支柱协同**：Metrics看趋势、Traces看链路、Logs看细节
- **智能采样**：在数据完整性和存储成本间找到平衡
- **告警分层收敛**：避免告警风暴，确保关键告警不被淹没
- **成本可观测**：Token消耗是Agent系统的核心成本驱动因素

将可观测性融入Agent系统的设计阶段，而非事后补救，才能真正构建可靠、可维护、可扩展的Agent平台。
