---
title: "AI系统可观测性架构设计：从日志到全链路追踪的实战指南"
description: "深入解析AI系统可观测性的三大支柱（日志、指标、追踪），结合LLM应用特点给出完整的可观测性架构方案与落地实践。"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: "distributed"
tags: ["可观测性", "AI系统", "监控", "日志", "分布式追踪", "LLM运维"]
draft: false
---

# AI系统可观测性架构设计：从日志到全链路追踪的实战指南

## 引言：AI系统的"黑盒"困境

传统的Web应用出现bug时，我们可以通过日志、指标和追踪快速定位问题。但在AI系统中，这个过程变得复杂得多：

- **模型输出具有不确定性**：同样的输入可能产生不同的输出，传统的断言式测试不再适用
- **故障模式更多样**：不是简单的"报错"或"不报错"，而是"输出质量下降"、"幻觉增加"、"响应变慢"等渐进式退化
- **因果链路更长**：一次用户请求可能经过提示词构造→模型推理→后处理→安全过滤等多个环节，每个环节都可能引入问题
- **反馈信号更模糊**：用户点击"不满意"到底是因为模型回答错误、格式不对、还是速度太慢？

2025年的一项调查显示，**67%的AI应用团队**表示"难以诊断线上AI质量问题"是他们面临的最大挑战之一。可观测性（Observability）——即通过系统的外部输出推断其内部状态的能力——成为AI系统工程化的关键基础设施。

本文将从AI系统的特殊性出发，设计一套完整的可观测性架构方案。

## 一、AI系统可观测性的特殊挑战

### 1.1 传统可观测性 vs AI可观测性

```
┌───────────────────────────────────────────────────────────┐
│              传统系统 vs AI系统的可观测性差异                 │
│                                                           │
│  维度          传统系统              AI系统                  │
│  ─────────────────────────────────────────────────────    │
│  确定性        确定性输出            概率性输出              │
│  故障模式      报错/超时/崩溃        质量退化/幻觉/偏见       │
│  根因分析      代码逻辑/依赖状态     模型+数据+提示词+配置     │
│  回归测试      单元测试/集成测试      A/B测试/人工评估         │
│  成本监控      计算/存储/网络        Token消耗/GPU时长         │
│  延迟分布      相对稳定              长尾分布（冷启动/长生成）   │
│  用户反馈      明确的错误码          主观的质量评价            │
└───────────────────────────────────────────────────────────┘
```

### 1.2 AI系统的可观测性需求矩阵

```
┌──────────────────────────────────────────────────────┐
│                可观测性需求优先级                       │
│                                                      │
│  高影响                                               │
│  │  ┌─────────────┐  ┌─────────────┐                │
│  │  │ 模型输出质量  │  │  端到端延迟  │                │
│  │  │  监控        │  │  追踪        │                │
│  │  └─────────────┘  └─────────────┘                │
│  │  ┌─────────────┐  ┌─────────────┐                │
│  │  │ Token成本    │  │  幻觉检测    │                │
│  │  │  追踪        │  │  与告警      │                │
│  │  └─────────────┘  └─────────────┘                │
│  │  ┌─────────────┐  ┌─────────────┐                │
│  │  │ 安全过滤     │  │  用户反馈    │                │
│  │  │  日志        │  │  收集        │                │
│  │  └─────────────┘  └─────────────┘                │
│  │  ┌─────────────┐                                 │
│  │  │ 提示词版本   │                                 │
│  │  │  管理        │                                 │
│  │  └─────────────┘                                 │
│  低影响         紧急                    不紧急         │
└──────────────────────────────────────────────────────┘
```

## 二、可观测性三大支柱：AI化改造

### 2.1 支柱一：日志（Logging）

AI系统的日志需要记录的信息远比传统系统丰富。

#### 日志分层设计

```
┌──────────────────────────────────────────────────────┐
│                 AI系统日志分层架构                     │
│                                                      │
│  Layer 4: 业务日志                                    │
│  ├── 用户会话日志（对话历史、满意度评分）               │
│  ├── 业务指标日志（转化率、任务完成率）                 │
│  └── A/B实验日志（实验组、对照组效果对比）              │
│                                                      │
│  Layer 3: AI推理日志                                  │
│  ├── Prompt日志（完整提示词、模板版本、变量值）          │
│  ├── 推理日志（模型输入/输出、Token数、概率分布）        │
│  ├── 后处理日志（格式化、过滤、安全检查结果）            │
│  └── 缓存日志（缓存命中率、相似查询）                  │
│                                                      │
│  Layer 2: 基础设施日志                                │
│  ├── 模型服务日志（加载状态、GPU利用率、批处理队列）      │
│  ├── 网络日志（API调用延迟、重试、降级）                │
│  └── 存储日志（向量数据库查询、缓存读写）               │
│                                                      │
│  Layer 1: 系统日志                                    │
│  ├── 应用日志（启动、配置、异常）                      │
│  └── 审计日志（访问控制、操作记录）                     │
└──────────────────────────────────────────────────────┘
```

#### AI推理日志的结构设计

```json
{
  "trace_id": "req_abc123",
  "span_id": "llm_inference_001",
  "timestamp": "2026-05-31T10:30:00.123Z",
  "level": "INFO",
  "category": "ai_inference",
  
  "prompt": {
    "template_id": "customer_support_v2.1",
    "template_version": "2026-05-28",
    "variables": {
      "user_query": "如何重置密码？",
      "context_docs": ["doc_001", "doc_015"],
      "system_prompt_hash": "sha256:abc..."
    },
    "token_count": {
      "system": 150,
      "user": 45,
      "context": 800,
      "total": 995
    }
  },
  
  "inference": {
    "model": "gpt-4o",
    "model_version": "2026-05-15",
    "temperature": 0.7,
    "max_tokens": 1024,
    "actual_tokens": 256,
    "latency_ms": 1250,
    "finish_reason": "stop",
    "confidence_score": 0.89,
    "logprobs_top5": [
      {"token": "密码", "prob": 0.45},
      {"token": "账户", "prob": 0.22},
      {"token": "重置", "prob": 0.18}
    ]
  },
  
  "post_processing": {
    "safety_filter": {"passed": true, "categories": []},
    "format_transform": {"applied": false},
    "citations_added": 2
  },
  
  "quality_signals": {
    "user_feedback": null,
    "retrieval_relevance": 0.82,
    "answer_completeness": 0.78
  }
}
```

#### 结构化日志的关键字段

| 字段类别 | 关键字段 | 用途 |
|---------|---------|------|
| 追踪标识 | trace_id, span_id | 关联同一请求的所有环节 |
| 模型信息 | model, model_version, temperature | 模型版本管理和参数审计 |
| Prompt信息 | template_id, template_version | 提示词版本追踪和A/B测试 |
| Token消耗 | input_tokens, output_tokens | 成本监控和优化 |
| 延迟信息 | latency_ms, time_to_first_token | 性能监控 |
| 质量信号 | confidence_score, finish_reason | 输出质量初步判断 |
| 安全信息 | safety_filter结果 | 合规审计 |

### 2.2 支柱二：指标（Metrics）

AI系统的指标体系需要覆盖模型质量、系统性能和业务效果三个维度。

#### 指标分类体系

```
┌──────────────────────────────────────────────────────────┐
│                 AI系统指标体系                             │
│                                                          │
│  模型质量指标                                              │
│  ├── accuracy_score     任务准确率（分类/提取等）           │
│  ├── hallucination_rate 幻觉率（检测到的虚构内容比例）       │
│  ├── relevance_score    回答相关性评分                     │
│  ├── completeness_score 回答完整度评分                     │
│  ├── consistency_score  一致性评分（同一问题多次回答的稳定性）│
│  └── safety_violation   安全违规次数                       │
│                                                          │
│  系统性能指标                                              │
│  ├── p50_latency        中位延迟                          │
│  ├── p95_latency        95分位延迟                        │
│  ├── p99_latency        99分位延迟                        │
│  ├── throughput_rps     每秒请求数                        │
│  ├── error_rate         错误率                            │
│  ├── timeout_rate       超时率                            │
│  └── cache_hit_rate     缓存命中率                        │
│                                                          │
│  Token与成本指标                                           │
│  ├── token_per_request  每请求平均Token数                  │
│  ├── cost_per_request   每请求平均成本                     │
│  ├── daily_cost         每日总成本                        │
│  ├── cost_by_endpoint   按接口统计成本                     │
│  └── cost_by_model      按模型统计成本                     │
│                                                          │
│  业务效果指标                                              │
│  ├── user_satisfaction  用户满意度评分                     │
│  ├── task_completion    任务完成率                        │
│  ├── escalation_rate    转人工率                          │
│  └── response_usage_rate 回答被采纳率                      │
└──────────────────────────────────────────────────────────┘
```

#### Prometheus指标配置示例

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# ===== 模型推理指标 =====

# 请求计数
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM inference requests',
    ['model', 'endpoint', 'status']
)

# 推理延迟
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM inference latency',
    ['model', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Token消耗
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens consumed',
    ['model', 'type']  # type: input/output
)

# 输出质量（通过后置评估）
llm_quality_score = Histogram(
    'llm_quality_score',
    'Quality score of LLM output',
    ['model', 'task_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# 幻觉检测
llm_hallucination_detected = Counter(
    'llm_hallucination_detected_total',
    'Number of detected hallucinations',
    ['model', 'severity']
)

# ===== 系统资源指标 =====

# GPU利用率
gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id', 'model']
)

# GPU显存
gpu_memory_used_bytes = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory usage',
    ['gpu_id']
)

# ===== 缓存指标 =====
cache_hits_total = Counter(
    'cache_hits_total',
    'Cache hit count',
    ['cache_type']  # semantic/exact
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Cache miss count',
    ['cache_type']
)
```

#### Grafana仪表板设计

```
┌──────────────────────────────────────────────────────────┐
│              AI系统监控仪表板布局                           │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │   实时请求量      │  │   当前延迟分布    │                │
│  │   (5min窗口)     │  │   (直方图)       │                │
│  │   ████ 120 rps  │  │   P50: 0.8s    │                │
│  └─────────────────┘  │   P95: 2.1s    │                │
│                        └─────────────────┘                │
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Token消耗趋势   │  │   成本趋势       │                │
│  │   (日/周/月)     │  │   (按模型分)     │                │
│  │   📈             │  │   📊             │                │
│  └─────────────────┘  └─────────────────┘                │
│  ┌────────────────────────────────────────┐              │
│  │           模型质量趋势                    │              │
│  │   accuracy: ████████░░ 82%             │              │
│  │   relevance: ███████░░░ 78%            │              │
│  │   hallucination: ██░░░░░░░░ 3.2%       │              │
│  └────────────────────────────────────────┘              │
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │   错误分类       │  │   GPU资源使用     │                │
│  │   Timeout: 12%  │  │   Util: 67%     │                │
│  │   Error: 5%     │  │   Memory: 4.2GB │                │
│  │   Safety: 3%    │  │   Temp: 72°C    │                │
│  └─────────────────┘  └─────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

### 2.3 支柱三：分布式追踪（Tracing）

AI系统的请求链路通常比传统系统更长、更复杂，分布式追踪的重要性更加突出。

#### AI系统追踪架构

```
┌──────────────────────────────────────────────────────────┐
│              AI请求全链路追踪                               │
│                                                          │
│  用户请求                                                 │
│    │                                                     │
│    ▼                                                     │
│  ┌──────────┐   trace_id: req_abc123                     │
│  │ API网关   │──→ span_001: auth_check (15ms)            │
│  └────┬─────┘   span_002: rate_limit (2ms)               │
│       │                                                   │
│       ▼                                                   │
│  ┌──────────┐   span_003: prompt_construction (20ms)     │
│  │ Prompt    │   span_004: retrieval (80ms)               │
│  │ Engine    │   span_005: context_assembly (10ms)        │
│  └────┬─────┘                                            │
│       │                                                   │
│       ▼                                                   │
│  ┌──────────┐   span_006: llm_inference (1200ms)         │
│  │ LLM       │   ├── span_007: prefill (300ms)           │
│  │ Service   │   ├── span_008: decode (850ms)            │
│  └────┬─────┘   └── span_009: sampling (50ms)            │
│       │                                                   │
│       ▼                                                   │
│  ┌──────────┐   span_010: post_processing (30ms)         │
│  │ Post      │   span_011: safety_check (15ms)           │
│  │ Process   │   span_012: format_output (5ms)           │
│  └────┬─────┘                                            │
│       │                                                   │
│       ▼                                                   │
│  ┌──────────┐   span_013: cache_store (5ms)              │
│  │ Cache     │                                            │
│  └────┬─────┘                                            │
│       │                                                   │
│       ▼                                                   │
│  响应返回        总延迟: 1377ms                            │
│                   ├── 网关: 17ms (1.2%)                   │
│                   ├── Prompt: 110ms (8.0%)                │
│                   ├── LLM: 1200ms (87.1%)                 │
│                   ├── 后处理: 50ms (3.6%)                  │
│                   └── 缓存: 5ms (0.4%)                    │
└──────────────────────────────────────────────────────────┘
```

#### OpenTelemetry集成方案

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化tracer
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ai-service")

# ===== AI推理追踪示例 =====

async def handle_ai_request(request):
    with tracer.start_as_current_span("ai_request") as root_span:
        root_span.set_attribute("user.id", request.user_id)
        root_span.set_attribute("request.type", request.task_type)
        
        # 1. Prompt构造
        with tracer.start_as_current_span("prompt_construction") as span:
            prompt_data = await build_prompt(request)
            span.set_attribute("prompt.template_id", prompt_data.template_id)
            span.set_attribute("prompt.token_count", prompt_data.total_tokens)
            span.set_attribute("prompt.retrieval_docs", len(prompt_data.docs))
        
        # 2. 模型推理
        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("llm.model", "gpt-4o")
            span.set_attribute("llm.temperature", 0.7)
            
            response = await llm_client.complete(
                messages=prompt_data.messages,
                max_tokens=1024,
                stream=True
            )
            
            # 流式响应追踪
            first_token_time = None
            tokens_generated = 0
            
            async for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()
                    span.set_attribute("llm.time_to_first_token_ms",
                        (first_token_time - inference_start) * 1000)
                tokens_generated += 1
                yield chunk
            
            span.set_attribute("llm.output_tokens", tokens_generated)
            span.set_attribute("llm.finish_reason", response.finish_reason)
        
        # 3. 后处理
        with tracer.start_as_current_span("post_processing") as span:
            # 安全检查
            with tracer.start_as_current_span("safety_check") as sub_span:
                safety_result = await check_safety(response.text)
                sub_span.set_attribute("safety.passed", safety_result.passed)
                sub_span.set_attribute("safety.categories", 
                    str(safety_result.flagged_categories))
            
            # 格式化
            with tracer.start_as_current_span("format_output") as sub_span:
                formatted = await format_response(response.text)
                sub_span.set_attribute("format.type", formatted.format_type)
        
        # 根span属性
        total_latency = (time.time() - request_start) * 1000
        root_span.set_attribute("response.latency_ms", total_latency)
        root_span.set_attribute("response.total_tokens", 
            prompt_data.total_tokens + tokens_generated)
        
        return formatted
```

## 三、AI特有的可观测性能力

### 3.1 Prompt版本管理与追踪

提示词是AI系统中最频繁变更的"代码"，需要专门的版本管理机制。

```
┌──────────────────────────────────────────────────────────┐
│              Prompt版本管理架构                             │
│                                                          │
│  Prompt Registry (提示词注册中心)                          │
│  │                                                       │
│  ├── templates/                                           │
│  │   ├── customer_support/                                │
│  │   │   ├── v1.0.0  (2026-01-15)  已归档                │
│  │   │   ├── v1.1.0  (2026-03-20)  已归档                │
│  │   │   ├── v2.0.0  (2026-05-01)  灰度中 (20%)         │
│  │   │   └── v2.1.0  (2026-05-28)  生产环境 (80%)       │
│  │   │                                                   │
│  │   ├── code_generation/                                 │
│  │   │   ├── v1.0.0  (2026-02-10)  生产环境              │
│  │   │   └── v1.1.0  (2026-05-15)  测试中               │
│  │   │                                                   │
│  │   └── summarization/                                   │
│  │       └── v1.0.0  (2026-04-01)  生产环境              │
│  │                                                       │
│  ├── evaluations/                                         │
│  │   └── 每个版本的评估结果和回归测试数据                    │
│  │                                                       │
│  └── experiments/                                         │
│      └── A/B实验配置和效果对比                              │
└──────────────────────────────────────────────────────────┘
```

### 3.2 输出质量自动评估

对于AI系统的输出质量，需要建立自动化的评估管道：

```
┌──────────────────────────────────────────────────────────┐
│              输出质量自动评估管道                            │
│                                                          │
│  LLM原始输出                                              │
│    │                                                     │
│    ├──→ [安全性检查] ──→ 是否包含有害内容？                 │
│    │         │              │                             │
│    │         │         Yes──→ 拦截 + 告警                  │
│    │         │         No───→ 继续                        │
│    │                                                     │
│    ├──→ [事实性检查] ──→ 是否与检索文档矛盾？               │
│    │         │              │                             │
│    │         │         矛盾──→ 标记 + 降权                 │
│    │         │         一致──→ 继续                        │
│    │                                                     │
│    ├──→ [相关性评估] ──→ 回答是否与问题相关？               │
│    │         │              │                             │
│    │         │         低分──→ 标记 + 人工审核队列          │
│    │         │         高分──→ 继续                        │
│    │                                                     │
│    ├──→ [完整性检查] ──→ 是否回答了问题的所有部分？          │
│    │         │              │                             │
│    │         │         不完整──→ 补充生成                   │
│    │         │         完整───→ 继续                       │
│    │                                                     │
│    └──→ [格式验证] ──→ 输出格式是否符合要求？               │
│              │              │                             │
│              │         不符合──→ 格式化处理                 │
│              │         符合───→ 最终输出                   │
│                                                          │
│  最终输出 + 质量评分                                       │
└──────────────────────────────────────────────────────────┘
```

### 3.3 幻觉检测机制

幻觉（Hallucination）是AI系统最独特的问题之一，需要专门的检测机制。

```python
class HallucinationDetector:
    """基于检索文档的幻觉检测器"""
    
    def __init__(self, similarity_threshold=0.7, 
                 fact_check_model="gpt-4o-mini"):
        self.similarity_threshold = similarity_threshold
        self.fact_check_model = fact_check_model
    
    async def detect(self, query: str, answer: str, 
                     retrieved_docs: list[str]) -> HallucinationReport:
        
        # 1. 事实声明提取
        claims = await self._extract_claims(answer)
        
        # 2. 逐条验证
        verified_claims = []
        hallucinated_claims = []
        
        for claim in claims:
            # 方法1：语义相似度检查
            is_supported = await self._check_semantic_support(
                claim, retrieved_docs
            )
            
            if not is_supported:
                # 方法2：LLM辅助事实核查
                is_factually_correct = await self._fact_check(
                    claim, retrieved_docs
                )
                
                if not is_factually_correct:
                    hallucinated_claims.append(claim)
                    continue
            
            verified_claims.append(claim)
        
        # 3. 计算幻觉率
        hallucination_rate = (
            len(hallucinated_claims) / len(claims) 
            if claims else 0
        )
        
        return HallucinationReport(
            total_claims=len(claims),
            verified=verified_claims,
            hallucinated=hallucinated_claims,
            hallucination_rate=hallucination_rate,
            severity=self._assess_severity(hallucinated_claims)
        )
    
    async def _extract_claims(self, answer: str) -> list[str]:
        """从回答中提取事实性声明"""
        prompt = f"""从以下回答中提取所有事实性声明，每个声明一行：
        
回答：{answer}

事实性声明："""
        
        response = await self.llm.complete(prompt)
        return [c.strip() for c in response.split('\n') if c.strip()]
    
    async def _check_semantic_support(self, claim: str, 
                                      docs: list[str]) -> bool:
        """检查声明是否被检索文档语义支持"""
        claim_embedding = await self.embedder.embed(claim)
        doc_embeddings = await self.embedder.embed_batch(docs)
        
        similarities = cosine_similarity(claim_embedding, doc_embeddings)
        return max(similarities) >= self.similarity_threshold
```

## 四、告警策略设计

### 4.1 AI系统告警分级

```
┌──────────────────────────────────────────────────────────┐
│              AI系统告警分级策略                             │
│                                                          │
│  P0 - 紧急（5分钟内响应）                                  │
│  ├── 模型服务完全不可用                                    │
│  ├── 幻觉率突增超过20%                                    │
│  ├── 安全过滤失效（有害内容通过）                           │
│  └── 数据泄露事件                                         │
│                                                          │
│  P1 - 高（30分钟内响应）                                   │
│  ├── P95延迟超过阈值2倍                                   │
│  ├── 错误率超过5%                                         │
│  ├── GPU温度超过85°C                                     │
│  └── 模型版本回退                                         │
│                                                          │
│  P2 - 中（2小时内响应）                                    │
│  ├── 用户满意度评分下降超过10%                             │
│  ├── Token消耗异常增长                                    │
│  ├── 缓存命中率低于阈值                                   │
│  └── 部分节点性能下降                                     │
│                                                          │
│  P3 - 低（下个工作日处理）                                 │
│  ├── 模型质量缓慢退化                                     │
│  ├── 新版本A/B测试效果不达预期                             │
│  ├── 日志采集缺失                                         │
│  └── 监控面板异常                                         │
└──────────────────────────────────────────────────────────┘
```

### 4.2 告警规则配置示例

```yaml
# alerting-rules.yaml
groups:
  - name: ai_service_critical
    rules:
      # P0: 模型服务完全不可用
      - alert: LLMServiceDown
        expr: llm_requests_total{status="error"} 
               / llm_requests_total > 0.5
        for: 2m
        labels:
          severity: P0
        annotations:
          summary: "LLM服务错误率超过50%"
          
      # P0: 幻觉率突增
      - alert: HallucinationSpike
        expr: rate(llm_hallucination_detected_total[5m]) 
               > 0.2
        for: 1m
        labels:
          severity: P0
        annotations:
          summary: "幻觉检出率突增"
          
      # P1: 延迟异常
      - alert: HighLatencyP95
        expr: histogram_quantile(0.95, 
               rate(llm_latency_seconds_bucket[5m])) > 5.0
        for: 5m
        labels:
          severity: P1
        annotations:
          summary: "P95延迟超过5秒"
          
      # P2: Token成本异常
      - alert: TokenCostAnomaly
        expr: rate(llm_tokens_total[1h]) 
               > 2 * avg_over_time(
                 rate(llm_tokens_total[1h])[7d:1h]
               )
        for: 30m
        labels:
          severity: P2
        annotations:
          summary: "Token消耗异常增长，可能是死循环或滥用"
```

## 五、落地实践：从0到1搭建AI可观测性

### 5.1 技术选型推荐

```
┌──────────────────────────────────────────────────────────┐
│              AI可观测性技术栈推荐                           │
│                                                          │
│  数据采集层                                               │
│  ├── OpenTelemetry  统一的追踪/指标/日志采集                │
│  ├── Fluent Bit     轻量级日志收集器                       │
│  └── Prometheus     指标采集和存储                         │
│                                                          │
│  数据存储层                                               │
│  ├── ClickHouse     日志存储（高压缩、快查询）              │
│  ├── Prometheus TSDB  指标存储                             │
│  └── Jaeger/Tempo   分布式追踪存储                        │
│                                                          │
│  数据处理层                                               │
│  ├── Grafana        可视化和告警                           │
│  ├── Langfuse       LLM专用可观测性平台                    │
│  └── Arize/Phoenix  AI质量监控平台                        │
│                                                          │
│  分析层                                                   │
│  ├── 自定义评估管道  幻觉检测、质量评分                     │
│  ├── A/B实验平台    效果对比和统计显著性                    │
│  └── 成本分析工具   Token消耗分析和优化建议                 │
└──────────────────────────────────────────────────────────┘
```

### 5.2 渐进式落地路径

```
Phase 1（1-2周）：基础可观测性
├── 接入OpenTelemetry，实现请求追踪
├── 记录基础指标（延迟、错误率、Token消耗）
├── 搭建Grafana基础仪表板
└── 配置P0/P1告警规则

Phase 2（2-4周）：AI专项能力
├── 实现Prompt版本管理和追踪
├── 接入Langfuse等LLM观测平台
├── 搭建输出质量自动评估管道
└── 实现幻觉检测和告警

Phase 3（1-2月）：深度分析
├── 建立A/B实验框架
├── 实现成本分析和优化建议
├── 搭建用户反馈闭环
└── 建立模型质量基线和回归测试

Phase 4（持续）：智能化运维
├── 基于历史数据的异常检测
├── 自动化质量退化根因分析
├── 智能告警降噪和关联分析
└── 成本预测和自动扩缩容
```

### 5.3 成本控制策略

可观测性本身也会产生成本，特别是AI系统的日志量通常很大。

```
成本优化策略：

1. 采样策略
   ├── 基础采样：生产环境100%记录请求级日志
   ├── 智能采样：高质量请求只记录摘要，低质量请求记录详情
   └── 动态采样：在异常期间自动提高采样率

2. 数据分层
   ├── 热数据（7天）：完整日志和追踪，ClickHouse存储
   ├── 温数据（30天）：聚合指标，Prometheus存储
   └── 冷数据（90天+）：摘要统计，对象存储

3. 预估成本（中等规模AI应用，日均10万请求）
   ├── OpenTelemetry + ClickHouse: ~$200/月
   ├── Prometheus + Grafana Cloud: ~$100/月
   ├── Langfuse Cloud: ~$50/月
   └── 总计: ~$350/月（约2500元）
```

## 总结

AI系统的可观测性不是传统可观测性的简单扩展，而是需要针对AI系统特殊性进行专门设计的工程实践。

**核心要点：**

1. **日志要记录"为什么"**：不仅记录发生了什么，还要记录AI为什么这样决策
2. **指标要覆盖质量维度**：除了性能指标，还要关注模型输出质量、幻觉率等AI特有指标
3. **追踪要覆盖全链路**：从Prompt构造到模型推理再到后处理，每个环节都要可观测
4. **告警要智能分级**：区分"服务不可用"和"质量退化"两种不同的故障模式
5. **成本要可控**：通过采样策略和数据分层控制可观测性本身的成本

可观测性不是一次性工程，而是一个持续演进的过程。建议从Phase 1开始，逐步构建完整的可观测性体系。

> 💡 **实践建议**：先从一个核心API接入完整的可观测性管道，验证方案可行性后再推广到所有服务。
