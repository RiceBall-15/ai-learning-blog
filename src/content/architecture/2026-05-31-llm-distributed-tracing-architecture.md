---
title: "大模型应用分布式追踪：全链路可观测性架构设计与实战"
description: "从请求入口到模型推理，构建LLM应用的全链路追踪体系，覆盖Token级监控、延迟归因与成本可视化"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: distributed
tags: ["可观测性", "分布式追踪", "LLM", "架构设计", "SRE"]
draft: false
---

## 引言：为什么LLM应用需要新的可观测性范式

传统微服务的可观测性建立在三大支柱之上：**Metrics（指标）、Logging（日志）、Tracing（追踪）**。这套体系在确定性系统中运行良好——每次HTTP请求的延迟、状态码、错误率都可以通过标准的OpenTelemetry框架捕获。

但LLM应用打破了这一切。

一个典型的RAG增强对话请求可能经历以下路径：

```
用户输入 → Query理解 → 向量检索(×3个库) → Rerank → Prompt组装 
→ LLM推理(流式) → 结构化输出解析 → 引用标注 → 响应组装
```

这里面有三个根本性差异：

| 传统微服务 | LLM应用 |
|-----------|---------|
| 请求延迟可预测（P99 < 500ms） | 推理延迟高度不确定（100ms~30s） |
| 成本与调用次数线性相关 | 成本与Token数强关联，单次调用可达数美元 |
| 失败模式确定（超时、5xx） | 失败模式模糊（内容质量下降、幻觉、偏题） |

本文将从实战角度，介绍一套专为LLM应用设计的全链路可观测性架构。

## 一、追踪模型：从Span到Token级粒度

### 1.1 传统Span模型的局限

OpenTelemetry的Span模型假设每个操作是原子的——一个Span有明确的开始/结束时间，以及确定的输入输出。但LLM推理具有两个独特性质：

**流式输出**：Token逐个生成，一个持续30秒的生成过程，传统模型只能记录为一个大Span，无法看到内部的Token生成节奏。

**非确定性延迟**：相同的Prompt，两次调用可能相差10倍延迟。原因可能是KV Cache命中率不同、Batch调度策略变化，甚至GPU上的其他任务干扰。

### 1.2 Token-Level Span设计

我们在生产环境中采用了**分层Span模型**：

```
[Root Span: user_query]
├── [Span: query_understanding] 12ms
│   └── [Span: intent_classification] 8ms
├── [Span: retrieval] 45ms
│   ├── [Span: vector_search_kb] 12ms  (hits: 15, returned: 5)
│   ├── [Span: vector_search_docs] 11ms (hits: 8, returned: 3)
│   └── [Span: web_search] 18ms  (hits: 120, returned: 10)
├── [Span: rerank] 28ms (input: 18, output: 5)
├── [Span: llm_generation] 3200ms
│   ├── [Metric: first_token_latency] 380ms
│   ├── [Metric: token_throughput] 62 tokens/s
│   ├── [Metric: total_input_tokens] 2847
│   ├── [Metric: total_output_tokens] 512
│   ├── [Event: tool_call] at token=128 → [Span: weather_api] 95ms
│   └── [Event: stop_reason] length
├── [Span: post_processing] 15ms
│   ├── [Span: citation_annotation] 8ms
│   └── [Span: format_validation] 4ms
└── [Metric: end_to_end_latency] 3345ms
```

关键设计点：

**1. 将LLM Span拆分为元数据子项**

```python
class LLMSpanObserver:
    """LLM推理的Span观测器"""
    
    def on_generation_start(self, span: Span, config: dict):
        span.set_attribute("llm.model", config["model"])
        span.set_attribute("llm.temperature", config.get("temperature", 1.0))
        span.set_attribute("llm.max_tokens", config.get("max_tokens"))
        span.set_attribute("llm.system_fingerprint", config.get("system_fingerprint"))
    
    def on_first_token(self, span: Span, latency_ms: float):
        span.add_event("first_token", {"latency_ms": latency_ms})
        # 这个指标对用户体验至关重要
    
    def on_token_batch(self, span: Span, tokens: list[str], 
                       token_ids: list[int], logprobs: list[float]):
        # 累积统计，不逐token记录（太贵）
        self._token_count += len(tokens)
        self._avg_logprob = (
            (self._avg_logprob * (self._token_count - len(tokens)) + 
             sum(logprobs)) / self._token_count
        )
    
    def on_generation_end(self, span: Span):
        span.set_attribute("llm.output_tokens", self._token_count)
        span.set_attribute("llm.avg_logprob", self._avg_logprob)
        span.set_attribute("llm.finish_reason", self._finish_reason)
        # Token级成本计算
        cost = self._calculate_cost(
            self._input_tokens, self._token_count, 
            span.get_attribute("llm.model")
        )
        span.set_attribute("llm.cost_usd", cost)
```

**2. 工具调用作为独立子Span**

当LLM在生成过程中调用外部工具（Function Calling），工具执行必须记录为独立Span，因为它们的延迟特征和LLM推理完全不同：

```
[Span: llm_generation] 5200ms
├── tokens 0-128: 纯推理 (450ms)
├── [Span: tool_call: search_web] (执行耗时: 1800ms)
├── tokens 129-380: 继续推理 (1200ms)  
├── [Span: tool_call: execute_code] (执行耗时: 800ms)
└── tokens 381-512: 最终输出 (950ms)
```

### 1.3 追踪上下文传播

LLM应用的一个特殊挑战是：**LLM推理引擎（如vLLM、SGLang）通常是独立部署的高性能服务**，它们不运行你的应用框架的OpenTelemetry SDK。需要通过HTTP Header传播上下文：

```python
# 应用层发送请求时注入追踪上下文
def call_llm_with_tracing(prompt: str, trace_context: dict) -> str:
    headers = {
        "X-Request-ID": trace_context["request_id"],
        "traceparent": trace_context["traceparent"],  # W3C Trace Context
        "X-User-ID": trace_context.get("user_id", ""),
    }
    
    response = llm_client.post(
        "/v1/completions",
        headers=headers,
        json={"prompt": prompt, "stream": True}
    )
    return response

# LLM服务侧（需要在推理框架中添加Hook）
# vLLM的trace middleware示例
class TraceMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        request_id = scope["headers"].get(b"x-request-id", b"").decode()
        traceparent = scope["headers"].get(b"traceparent", b"").decode()
        
        tracer = trace.get_tracer("llm-inference")
        with tracer.start_as_current_span(
            "vllm.generate",
            context=trace.extract_traceparent(traceparent)
        ) as span:
            span.set_attribute("request_id", request_id)
            await self.app(scope, receive, send)
```

## 二、延迟归因分析：找到真正的瓶颈

### 2.1 延迟瀑布图

当一个LLM请求耗时5秒时，"慢在哪"是首要问题。我们设计了**延迟归因瀑布图**：

```
请求总耗时: 5200ms
├── [12%] Query理解:      620ms  ████████
├── [35%] 检索阶段:      1820ms ███████████████████████
│   ├── 向量检索:         180ms  ███
│   ├── Rerank:           420ms  ██████
│   └── 上下文组装:      1220ms  ██████████████████  ← 瓶颈：数据库查询慢
├── [41%] LLM推理:       2130ms █████████████████████████
│   ├── 首Token延迟:      380ms  █████
│   ├── 生成延迟:        1250ms  ███████████████
│   └── 工具调用:         500ms  ███████
└── [12%] 后处理:         630ms  ████████
```

**关键指标定义**：

```python
@dataclass
class LatencyBreakdown:
    """延迟归因分解"""
    
    total_ms: float
    
    # LLM特有指标
    time_to_first_token_ms: float    # TTFT：用户感知的响应速度
    inter_token_latency_ms: float    # ITL：生成流畅度
    time_between_tools_ms: float     # 工具调用间隔
    queue_time_ms: float             # 排队等待时间（Batch模式下显著）
    
    # 检索相关指标
    retrieval_latency_ms: float
    rerank_latency_ms: float
    
    @property
    def llm_compute_ratio(self) -> float:
        """LLM计算占总耗时比例"""
        return (self.total_ms - self.queue_time_ms) / self.total_ms
    
    @property
    def user_perceived_latency(self) -> float:
        """用户感知延迟 = TTFT + 后续生成时间"""
        return self.time_to_first_token_ms + self.inter_token_latency_ms
```

### 2.2 延迟异常检测

我们使用**动态基线**而非固定阈值来检测异常：

```python
class LatencyAnomalyDetector:
    """基于滑动窗口的延迟异常检测"""
    
    def __init__(self, window_size: int = 100, threshold_sigma: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold_sigma
        self.windows: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
    
    def check(self, operation: str, latency_ms: float) -> AnomalyResult:
        window = self.windows[operation]
        window.append(latency_ms)
        
        if len(window) < 20:  # 至少20个样本才有统计意义
            return AnomalyResult(is_anomaly=False, reason="insufficient_data")
        
        mean = np.mean(window)
        std = np.std(window)
        z_score = (latency_ms - mean) / max(std, 0.1)
        
        if z_score > self.threshold:
            return AnomalyResult(
                is_anomaly=True,
                reason=f"latency_spike: {latency_ms:.0f}ms "
                       f"(baseline: {mean:.0f}±{std:.0f}ms, z={z_score:.1f})",
                severity="high" if z_score > 5 else "medium"
            )
        return AnomalyResult(is_anomaly=False)
```

## 三、成本追踪：Token就是钱

### 3.1 成本模型

LLM应用的成本结构与传统API完全不同。以GPT-4o为例：

| 操作 | 单价 | 一次典型对话成本 |
|------|------|-----------------|
| 输入Token | $2.5/1M tokens | 3000 tokens → $0.0075 |
| 输出Token | $10/1M tokens | 800 tokens → $0.0080 |
| 缓存命中输入 | $1.25/1M tokens | 2000 tokens → $0.0025 |
| 总计 | - | ~$0.018/次对话 |

如果日活1万用户，每人每天10次对话，月成本约 **$5,400**。而如果错误处理不当（重试、幻觉导致的重复调用），成本可能翻倍。

### 3.2 成本追踪实现

```python
class CostTracker:
    """Token级成本追踪"""
    
    # 模型定价表（需定期更新）
    PRICING = {
        "gpt-4o": {"input": 2.5, "output": 10.0, "cached": 1.25},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6, "cached": 0.075},
        "claude-3.5-sonnet": {"input": 3.0, "output": 15.0, "cached": 1.5},
        "deepseek-v3": {"input": 0.27, "output": 1.1, "cached": 0.07},
    }
    
    def track_request(self, model: str, input_tokens: int, 
                      output_tokens: int, cached_tokens: int = 0) -> CostRecord:
        pricing = self.PRICING.get(model, {"input": 0, "output": 0, "cached": 0})
        
        fresh_input = input_tokens - cached_tokens
        cost = (
            fresh_input / 1_000_000 * pricing["input"] +
            cached_tokens / 1_000_000 * pricing["cached"] +
            output_tokens / 1_000_000 * pricing["output"]
        )
        
        return CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            timestamp=datetime.utcnow()
        )
    
    def get_user_daily_cost(self, user_id: str, date: str) -> float:
        """查询用户日成本"""
        records = self.store.query(
            user_id=user_id, date=date
        )
        return sum(r.cost_usd for r in records)
```

### 3.3 成本优化策略的可观测性

不同的优化策略需要不同的监控维度：

```
成本优化效果看板
├── Prompt Caching 效果
│   ├── 缓存命中率: 67.3% (↑ 从 42%)
│   ├── 节省金额: $1,234/天
│   └── 缓存延迟开销: +12ms (可接受)
├── 小模型路由效果  
│   ├── 路由到小模型比例: 34% (简单问题)
│   ├── 准确率下降: <0.5% (可接受)
│   └── 节省金额: $892/天
├── 摘要压缩效果
│   ├── 平均输入Token压缩: 2847 → 1523 (46.5%↓)
│   ├── 信息损失评估: 3.2% (BLEU对比)
│   └── 节省金额: $678/天
└── 总计节省: $2,804/天 (约46%成本降低)
```

## 四、质量追踪：超越延迟和成本

### 4.1 延迟之外的信号

一个LLM响应"快速且便宜"但"毫无用处"，是更糟糕的情况。我们需要追踪**质量信号**：

```python
@dataclass
class QualitySignals:
    """LLM输出质量信号"""
    
    # 自动化信号
    hallucination_score: float     # 幻觉检测分数 (0-1, 越低越好)
    relevance_score: float         # 与问题相关性 (0-1)
    coherence_score: float         # 逻辑连贯性 (0-1)
    citation_accuracy: float       # 引用准确率 (0-1)
    
    # 用户反馈信号
    user_rating: Optional[int]     # 用户显式评分 (1-5)
    user_feedback_type: Optional[str]  # "helpful"/"harmful"/"unclear"
    regenerate_count: int          # 用户重新生成次数（负面信号）
    session_abandonment: bool      # 对话中断（严重负面信号）
```

### 4.2 质量-成本平衡视图

```python
class QualityCostTracker:
    """质量-成本平衡追踪"""
    
    def generate_daily_report(self, date: str) -> QualityCostReport:
        requests = self.store.get_requests(date)
        
        return QualityCostReport(
            total_requests=len(requests),
            avg_cost=mean([r.cost for r in requests]),
            avg_latency=mean([r.latency_ms for r in requests]),
            avg_quality=mean([r.quality_score for r in requests]),
            
            # 帕累托分析：80%成本花在哪些请求上
            high_cost_low_quality=[
                r for r in requests 
                if r.cost > percentile(requests, 80, key=lambda x: x.cost)
                and r.quality_score < 0.6
            ],
            
            # 浪费估算：可以被更小模型处理的请求
            oversized_model_estimates=self._estimate_oversized(requests),
        )
```

## 五、架构实现：基于OpenTelemetry的LLM可观测性平台

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                   用户请求入口                         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              应用网关层 (FastAPI/Express)              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │  Tracing     │ │  Metrics     │ │  Logging     │ │
│  │  Middleware   │ │  Collector   │ │  Enricher    │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ │
└─────────┼────────────────┼────────────────┼─────────┘
          │                │                │
┌─────────▼────────────────▼────────────────▼─────────┐
│              OpenTelemetry Collector                  │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────────┐ │
│  │ Traces  │  │ Metrics  │  │ Logs                │ │
│  │ Pipeline│  │ Pipeline │  │ Pipeline            │ │
│  └────┬────┘  └────┬─────┘  └─────────┬───────────┘ │
└───────┼────────────┼──────────────────┼─────────────┘
        │            │                  │
┌───────▼────┐ ┌─────▼──────┐ ┌────────▼────────────┐
│  Jaeger/   │ │ Prometheus │ │  Loki/Elasticsearch  │
│  Tempo     │ │ + Grafana  │ │                     │
└────────────┘ └────────────┘ └─────────────────────┘
        │            │                  │
┌───────▼────────────▼──────────────────▼─────────────┐
│           LLM可观测性 Dashboard (Grafana)             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ 追踪视图  │ │ 成本看板  │ │ 质量看板  │ │ 告警    │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │
└─────────────────────────────────────────────────────┘
```

### 5.2 核心组件实现

```python
# tracing/middleware.py
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

class LLMTracingMiddleware:
    """LLM应用的OpenTelemetry中间件"""
    
    def __init__(self, app):
        self.app = app
        self.tracer = trace.get_tracer("llm-app")
        self.meter = metrics.get_meter("llm-app")
        
        # 定义指标
        self.request_counter = self.meter.create_counter(
            "llm.requests.total", description="Total LLM requests"
        )
        self.token_histogram = self.meter.create_histogram(
            "llm.tokens", description="Token counts per request"
        )
        self.cost_counter = self.meter.create_counter(
            "llm.cost.usd", description="LLM cost in USD"
        )
        self.latency_histogram = self.meter.create_histogram(
            "llm.latency.ms", description="End-to-end latency"
        )
    
    async def __call__(self, request):
        request_id = str(uuid.uuid4())
        
        with self.tracer.start_as_current_span(
            "llm_request",
            attributes={
                "request.id": request_id,
                "user.id": request.state.user_id,
                "app.version": APP_VERSION,
            }
        ) as span:
            start_time = time.time()
            
            try:
                # 执行请求链路
                result = await self._process_request(request, span)
                
                # 记录成功指标
                self.request_counter.add(1, {"status": "success"})
                self.token_histogram.record(
                    result.total_tokens, {"model": result.model}
                )
                self.cost_counter.add(
                    result.cost_usd, {"model": result.model}
                )
                
                span.set_status(StatusCode.OK)
                return result
                
            except Exception as e:
                self.request_counter.add(1, {"status": "error"})
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, str(e))
                raise
            
            finally:
                latency_ms = (time.time() - start_time) * 1000
                self.latency_histogram.record(latency_ms)
                span.set_attribute("llm.latency_ms", latency_ms)
```

### 5.3 Grafana Dashboard核心面板

```json
{
  "panels": [
    {
      "title": "请求速率 & 错误率",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate(llm_requests_total[5m])",
          "legendFormat": "请求速率 (req/s)"
        },
        {
          "expr": "rate(llm_requests_total{status='error'}[5m]) / rate(llm_requests_total[5m])",
          "legendFormat": "错误率 (%)"
        }
      ]
    },
    {
      "title": "延迟分位数 (P50/P95/P99)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(llm_latency_ms_bucket[5m]))",
          "legendFormat": "P50"
        },
        {
          "expr": "histogram_quantile(0.95, rate(llm_latency_ms_bucket[5m]))",
          "legendFormat": "P95"
        },
        {
          "expr": "histogram_quantile(0.99, rate(llm_latency_ms_bucket[5m]))",
          "legendFormat": "P99"
        }
      ]
    },
    {
      "title": "每小时成本趋势",
      "type": "barchart",
      "targets": [
        {
          "expr": "sum(increase(llm_cost_usd[1h])) by (model)",
          "legendFormat": "{{model}}"
        }
      ]
    }
  ]
}
```

## 六、实战经验与避坑指南

### 6.1 采样策略

LLM应用的追踪数据量极大。一个日均10万请求的应用，每天会产生约500MB的原始追踪数据。**全量采集不可行**。

```python
# 推荐的采样策略
SAMPLING_CONFIG = {
    # 错误请求：100%采样（分析问题必须）
    "error": {"sample_rate": 1.0},
    
    # 高延迟请求：100%采样（性能优化必须）
    "slow": {"threshold_ms": 5000, "sample_rate": 1.0},
    
    # 高成本请求：100%采样（成本优化必须）
    "expensive": {"threshold_usd": 0.1, "sample_rate": 1.0},
    
    # 普通请求：5%采样（趋势分析足够）
    "normal": {"sample_rate": 0.05},
}
```

### 6.2 数据保留策略

| 数据类型 | 保留时间 | 存储位置 |
|---------|---------|---------|
| 追踪数据（完整Span） | 7天 | Jaeger/Tempo |
| 聚合指标 | 90天 | Prometheus |
| 成本聚合数据 | 1年 | ClickHouse |
| 质量评估数据 | 30天 | PostgreSQL |

### 6.3 常见陷阱

**陷阱1：在Span属性中存储完整Prompt**

```python
# ❌ 错误：Prompt可能包含敏感信息，且体积巨大
span.set_attribute("llm.prompt", full_prompt)  # 可能10KB+

# ✅ 正确：只记录摘要信息
span.set_attribute("llm.prompt_tokens", len(tokenize(full_prompt)))
span.set_attribute("llm.prompt_hash", hashlib.sha256(full_prompt.encode()).hexdigest()[:16])
```

**陷阱2：忽略流式输出的追踪**

```python
# ❌ 错误：流式输出时只记录最终结果
response = llm.stream(prompt)
for chunk in response:
    yield chunk
span.set_attribute("llm.output_tokens", final_token_count)

# ✅ 正确：在流式过程中累积统计
token_count = 0
first_token_time = None
async for chunk in llm.astream(prompt):
    if first_token_time is None:
        first_token_time = time.time()
        span.add_event("first_token", {"latency_ms": ...})
    token_count += 1
    yield chunk
span.set_attribute("llm.output_tokens", token_count)
```

**陷阱3：追踪ID与业务ID不一致**

确保所有系统使用统一的 `request_id` 关联追踪、日志和业务数据。这个ID应该从用户请求入口生成，并通过所有中间层传播。

## 总结

LLM应用的可观测性不是传统APM的简单扩展，而是一个需要重新设计的系统。核心要点：

1. **Token级粒度**：传统Span不够，需要记录首Token延迟、Token吞吐量等LLM特有指标
2. **延迟归因**：精确区分检索、推理、工具调用各阶段的耗时
3. **成本可视化**：Token即成本，必须有实时的成本追踪和优化效果评估
4. **质量信号**：延迟和成本只是表面，输出质量才是LLM应用的生命线
5. **实用主义采样**：全量采集不可行，用策略性采样平衡成本和可见性

可观测性是LLM应用从"能跑"到"能用"的关键一步。没有它，你就是在黑暗中飞行。
