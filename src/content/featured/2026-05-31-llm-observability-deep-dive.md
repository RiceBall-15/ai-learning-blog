---
title: "LLM应用可观测性深度解析：从黑盒到白盒，构建AI系统的全链路监控体系"
description: "深度剖析LLM应用可观测性的三大支柱（Traces、Metrics、Logs），对比LangSmith、Langfuse、Arize Phoenix等主流平台，提供生产级可观测性架构设计与实战指南"
date: "2026-05-31"
author: "RiceBall-15"
category: "featured"
subCategory: "deep-dive"
tags: ["LLM可观测性", "AI监控", "LangSmith", "Langfuse", "Tracing", "MLOps", "LLM运维"]
draft: false
---

# LLM应用可观测性深度解析：从黑盒到白盒

> 当你的RAG系统在凌晨3点开始返回越来越离谱的回答，当你发现Agent在某些用户请求下陷入了无限循环，当你的LLM API账单突然翻了3倍但你不知道为什么——这些都是缺乏可观测性的代价。传统的APM（Application Performance Monitoring）工具在LLM应用面前几乎失灵，因为它们无法理解Token级别的行为、Prompt的语义漂移、以及模型输出的非确定性。本文将系统性地拆解LLM应用可观测性的完整技术栈，从三大支柱到实战架构，帮你把AI系统从"黑盒"变成"白盒"。

---

## 一、为什么LLM应用需要全新的可观测性范式

### 1.1 传统可观测性在LLM场景的失灵

```
┌────────────────────────────────────────────────────────────────────────┐
│           传统APM vs LLM可观测性：为什么"老办法"行不通                   │
│                                                                        │
│  传统Web应用              LLM应用                                      │
│  ─────────────           ──────────                                    │
│  请求→响应，延迟可预测     请求→多次LLM调用→工具调用→最终响应              │
│  HTTP状态码=成功/失败      HTTP 200 ≠ 业务成功（可能返回幻觉）            │
│  输入输出确定性强           输入相同，输出可能完全不同                     │
│  错误有明确的Exception      "错误"是语义级的（答非所问、幻觉、偏见）       │
│  成本与请求量线性相关       成本取决于Token数+模型选择+缓存命中率          │
│  延迟在毫秒级              首Token延迟可能数秒，总延迟数十秒              │
│                                                                        │
│  核心矛盾：传统可观测性关注"系统是否正常运行"，                           │
│           LLM可观测性需要关注"系统输出是否正确且有用"                     │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.2 LLM应用的五大可观测性挑战

| 挑战 | 描述 | 传统方案的不足 |
|------|------|--------------|
| **语义级错误** | 输出看似正常但内容错误/幻觉 | HTTP 200，无异常抛出 |
| **非确定性** | 相同输入可能产生不同输出 | 无法通过重放测试复现 |
| **多步推理追踪** | Agent可能调用5-10次LLM+工具 | 链路追踪缺乏语义理解 |
| **成本归因** | 单次请求可能涉及多个模型 | 无法按请求精确计算成本 |
| **用户反馈闭环** | 用户满意与否难以量化 | 缺乏语义级别的质量指标 |

---

## 二、LLM可观测性三大支柱

### 2.1 支柱一：Traces（链路追踪）

Trace是LLM可观测性的核心，一次用户请求可能涉及多个LLM调用、工具调用、检索操作：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Trace: 用户提问"分析Q1财务报告中的营收增长趋势"                            │
│                                                                         │
│  ├── Span 1: Query Understanding (120ms)                                │
│  │   ├── LLM Call: 意图识别 (gpt-4o-mini, 150 tokens, $0.0001)         │
│  │   └── 输出: {"intent": "financial_analysis", "entities": ["Q1"]}     │
│  │                                                                      │
│  ├── Span 2: Document Retrieval (340ms)                                  │
│  │   ├── Embedding: query → vector (text-embedding-3-small, $0.00001)  │
│  │   ├── Vector Search: top-5 documents (Pinecone, 85ms)               │
│  │   └── Reranker: cross-encoder rerank (180ms)                        │
│  │                                                                      │
│  ├── Span 3: Context Assembly (45ms)                                     │
│  │   └── 构建Prompt: 注入检索文档 + 系统指令 + 对话历史                  │
│  │                                                                      │
│  ├── Span 4: LLM Generation (2,340ms)                                   │
│  │   ├── Model: gpt-4o (input: 3,200 tokens, output: 850 tokens)       │
│  │   ├── Cost: $0.028                                                   │
│  │   └── Usage: prompt_tokens=3200, completion_tokens=850              │
│  │                                                                      │
│  ├── Span 5: Output Validation (80ms)                                    │
│  │   ├── Fact Check: against source documents                           │
│  │   └── Safety Filter: passed                                          │
│  │                                                                      │
│  └── Total: 2,925ms | Total Cost: $0.0281                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键Trace数据点**：

```python
# 生产级Trace数据模型
trace_data = {
    "trace_id": "abc-123-def-456",
    "user_id": "user_789",
    "session_id": "session_012",
    "start_time": "2026-05-31T10:30:00Z",
    "end_time": "2026-05-31T10:30:03Z",
    "total_duration_ms": 2925,
    "total_cost_usd": 0.0281,
    "spans": [
        {
            "span_id": "span_001",
            "type": "llm_call",
            "model": "gpt-4o",
            "input_tokens": 3200,
            "output_tokens": 850,
            "latency_ms": 2340,
            "cost_usd": 0.028,
            "input_messages": [...],  # 完整的输入Prompt
            "output_text": "...",     # 完整的模型输出
            "metadata": {
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9
            }
        }
    ],
    "user_feedback": None,  # 稍后由用户反馈填充
    "quality_score": None   # 稍后由评估Pipeline填充
}
```

### 2.2 支柱二：Metrics（指标体系）

LLM应用需要一套全新的指标体系：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM应用核心指标体系                                    │
│                                                                         │
│  📊 性能指标 (Performance)                                               │
│  ├── TTFT (Time to First Token)    首Token延迟                          │
│  ├── TPS (Tokens Per Second)       生成速度                              │
│  ├── E2E Latency                  端到端延迟                            │
│  └── Throughput                   吞吐量 (req/min)                      │
│                                                                         │
│  💰 成本指标 (Cost)                                                      │
│  ├── Cost per Request             单请求成本                             │
│  ├── Cost per User                单用户成本                             │
│  ├── Token Efficiency             有效Token占比                          │
│  └── Cache Hit Rate              缓存命中率                             │
│                                                                         │
│  🎯 质量指标 (Quality)                                                   │
│  ├── Answer Relevance             回答相关性                             │
│  ├── Faithfulness                 忠实度（是否基于检索内容）              │
│  ├── Hallucination Rate           幻觉率                                 │
│  ├── User Satisfaction (CSAT)     用户满意度                             │
│  └── Task Completion Rate         任务完成率                             │
│                                                                         │
│  🔒 安全指标 (Safety)                                                    │
│  ├── Jailbreak Attempt Rate       越狱尝试率                             │
│  ├── PII Leakage Rate             个人信息泄露率                         │
│  ├── Toxicity Score               有害内容评分                           │
│  └── Policy Violation Rate        违规率                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Grafana Dashboard配置示例**：

```yaml
# Prometheus指标定义
metrics:
  # 请求级别指标
  llm_requests_total:
    type: counter
    labels: [model, endpoint, status]
    description: "LLM请求总数"
  
  llm_request_duration_seconds:
    type: histogram
    labels: [model, endpoint]
    buckets: [0.1, 0.5, 1, 2, 5, 10, 30]
    description: "LLM请求延迟分布"
  
  llm_tokens_total:
    type: counter
    labels: [model, direction]  # direction: input/output
    description: "Token消耗总量"
  
  llm_cost_usd_total:
    type: counter
    labels: [model, user_id]
    description: "总成本(美元)"
  
  # 质量级别指标
  llm_answer_relevance_score:
    type: gauge
    labels: [model, user_id]
    description: "回答相关性评分 (0-1)"
  
  llm_hallucination_detected_total:
    type: counter
    labels: [model, category]
    description: "检测到的幻觉总数"
  
  llm_cache_hit_ratio:
    type: gauge
    labels: [model, cache_type]  # cache_type: semantic/exact
    description: "缓存命中率"
```

### 2.3 支柱三：Logs（日志）

LLM日志需要记录传统日志无法覆盖的语义信息：

```python
import structlog
import json
from datetime import datetime

logger = structlog.get_logger("llm_app")

# 生产级LLM日志记录
def log_llm_interaction(
    trace_id: str,
    model: str,
    messages: list,
    response: str,
    metrics: dict,
    quality: dict = None
):
    """记录完整的LLM交互日志"""
    
    logger.info(
        "llm_interaction",
        trace_id=trace_id,
        timestamp=datetime.utcnow().isoformat(),
        # 模型信息
        model=model,
        temperature=metrics.get("temperature", 0.7),
        
        # Token使用
        input_tokens=metrics["input_tokens"],
        output_tokens=metrics["output_tokens"],
        total_tokens=metrics["input_tokens"] + metrics["output_tokens"],
        
        # 性能指标
        ttft_ms=metrics.get("ttft_ms"),
        total_latency_ms=metrics["total_latency_ms"],
        tps=metrics.get("tps"),
        
        # 成本
        cost_usd=metrics["cost_usd"],
        
        # 内容记录（脱敏后）
        input_summary=_summarize_messages(messages),
        output_summary=response[:200] + "..." if len(response) > 200 else response,
        
        # 质量评估（如果有）
        quality_score=quality.get("score") if quality else None,
        faithfulness=quality.get("faithfulness") if quality else None,
        relevance=quality.get("relevance") if quality else None,
        
        # 用户上下文
        user_id=metrics.get("user_id"),
        session_id=metrics.get("session_id"),
        
        # 标签（用于后续分析）
        tags=_extract_tags(response, quality)
    )


def _summarize_messages(messages: list) -> str:
    """消息摘要（隐私保护）"""
    summaries = []
    for msg in messages[-3:]:  # 只记录最近3条
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # 截断并脱敏
        summary = content[:100] + "..." if len(content) > 100 else content
        summaries.append(f"[{role}] {summary}")
    return " | ".join(summaries)


def _extract_tags(response: str, quality: dict) -> list:
    """从响应和质量评估中提取标签"""
    tags = []
    if quality:
        if quality.get("hallucination_detected"):
            tags.append("hallucination")
        if quality.get("relevance", 1) < 0.5:
            tags.append("low_relevance")
        if quality.get("toxicity_score", 0) > 0.7:
            tags.append("toxic_content")
    # 响应内容标签
    if "抱歉" in response or "无法" in response:
        tags.append("refusal")
    if len(response) < 50:
        tags.append("short_response")
    return tags
```

---

## 三、主流LLM可观测性平台对比

### 3.1 平台全景

```
┌─────────────────────────────────────────────────────────────────────────┐
│                LLM可观测性平台生态                                       │
│                                                                         │
│  ┌─── 商业SaaS ──────────────────────────────────────────────────┐     │
│  │  LangSmith        LangChain官方，生态最完善                      │     │
│  │  Arize Phoenix    开源+商业，ML可观测性标杆                      │     │
│  │  Weights & Biases 全链路ML实验跟踪                               │     │
│  │  Datadog LLM      企业级APM扩展LLM能力                          │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌─── 开源自部署 ────────────────────────────────────────────────┐     │
│  │  Langfuse         最活跃的开源LLM可观测性平台                   │     │
│  │  Helicone         专注LLM代理和成本优化                         │     │
│  │  LiteLLM          统一LLM网关+内置监控                          │     │
│  │  AgentOps         Agent专用可观测性                              │     │
│  └────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心平台深度对比

| 维度 | LangSmith | Langfuse | Arize Phoenix | Helicone |
|------|-----------|----------|--------------|----------|
| **开源** | ❌ 商业 | ✅ 开源(Apache 2.0) | ✅ 开源 | ✅ 开源+商业 |
| **自部署** | ❌ 仅SaaS | ✅ Docker一键部署 | ✅ | ✅ |
| **LangChain集成** | ⭐⭐⭐⭐⭐ 原生 | ⭐⭐⭐⭐ 原生 | ⭐⭐⭐ 需适配 | ⭐⭐⭐ 需适配 |
| **非LangChain框架** | ⭐⭐⭐ OpenAI SDK | ⭐⭐⭐⭐⭐ 框架无关 | ⭐⭐⭐⭐⭐ 框架无关 | ⭐⭐⭐⭐ 框架无关 |
| **Tracing** | ✅ | ✅ | ✅ | ✅ |
| **Evaluation** | ✅ 内置评估器 | ✅ 评分系统 | ✅ ML评估 | ⚠️ 基础 |
| **Prompt管理** | ✅ 版本管理 | ✅ Prompt Playground | ❌ | ❌ |
| **成本追踪** | ✅ | ✅ | ✅ | ✅ 核心功能 |
| **数据隐私** | ⚠️ 数据上传 | ✅ 本地数据 | ✅ 本地数据 | ⚠️ 取决于部署 |
| **用户反馈** | ✅ | ✅ | ✅ | ⚠️ 基础 |
| **生产就绪度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **免费额度** | 有限 | 无限(自部署) | 无限(自部署) | 10万次/月 |
| **社区活跃度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### 3.3 选型决策指南

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM可观测性平台选型决策树                              │
│                                                                         │
│  你在用LangChain吗？                                                     │
│  ├── 是 → 需要自部署？                                                   │
│  │        ├── 是 → Langfuse（LangChain原生集成+自部署）                 │
│  │        └── 否 → LangSmith（最佳LangChain体验）                       │
│  │                                                                      │
│  └── 否 → 框架是什么？                                                   │
│           ├── 纯OpenAI/Anthropic SDK                                    │
│           │    ├── 需要自部署 → Langfuse（框架无关+自部署）              │
│           │    └── 可用SaaS → Helicone（成本优化突出）                   │
│           │                                                              │
│           ├── 多模型混合调用                                              │
│           │    ├── 需要深度ML评估 → Arize Phoenix                       │
│           │    └── 需要统一网关 → LiteLLM + Langfuse                    │
│           │                                                              │
│           └── 企业级生产环境                                               │
│                ├── 已有Datadog → Datadog LLM Monitoring                │
│                └── 纯LLM场景 → LangSmith Enterprise                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 四、实战：构建生产级LLM可观测性架构

### 4.1 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│              生产级LLM可观测性架构                                        │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  LLM App     │───→│  Trace       │───→│  Storage     │              │
│  │  (FastAPI/   │    │  Collector   │    │  (ClickHouse │              │
│  │   Flask)     │    │  (OTel)      │    │   + Redis)   │              │
│  └──────────────┘    └──────────────┘    └──────┬───────┘              │
│         │                                        │                      │
│         │            ┌──────────────┐           │                      │
│         └───────────→│  Metrics     │───────────┘                      │
│                      │  Exporter    │                                  │
│                      │  (Prometheus)│                                  │
│                      └──────┬───────┘                                  │
│                             │                                          │
│                      ┌──────▼───────┐                                  │
│                      │  Grafana     │                                  │
│                      │  Dashboard   │                                  │
│                      └──────────────┘                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Evaluation Pipeline (异步)                                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │  │ Relevance│  │Faithful- │  │Hallucina-│  │ Safety   │        │   │
│  │  │ Judge    │  │ness Check│  │tion Det. │  │ Filter   │        │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 核心代码实现

```python
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
import json


@dataclass
class SpanData:
    """单个Span数据"""
    span_id: str
    span_type: str  # llm_call, tool_call, retrieval, chain
    start_time: float
    end_time: Optional[float] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TraceData:
    """完整的Trace数据"""
    trace_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    spans: list = field(default_factory=list)
    total_cost_usd: float = 0.0
    user_feedback: Optional[int] = None  # 1-5 评分
    quality_scores: dict = field(default_factory=dict)


class LLMTracer:
    """LLM应用链路追踪器"""
    
    def __init__(self, export_fn: Optional[Callable] = None):
        """
        Args:
            export_fn: Trace数据导出函数（发送到Langfuse/LangSmith/自建平台）
        """
        self.export_fn = export_fn
        self._local = {}  # 线程本地存储
    
    def start_trace(
        self,
        user_id: str = None,
        session_id: str = None,
        metadata: dict = None
    ) -> str:
        """开始一个新的Trace"""
        trace_id = str(uuid.uuid4())
        trace = TraceData(
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            start_time=time.time(),
        )
        self._local[trace_id] = trace
        return trace_id
    
    @contextmanager
    def trace_span(
        self,
        trace_id: str,
        span_type: str,
        name: str = None,
        **kwargs
    ):
        """追踪一个Span（上下文管理器）"""
        span_id = str(uuid.uuid4())[:8]
        span = SpanData(
            span_id=span_id,
            span_type=span_type,
            start_time=time.time(),
            **kwargs
        )
        
        try:
            yield span
        except Exception as e:
            span.error = str(e)
            raise
        finally:
            span.end_time = time.time()
            trace = self._local.get(trace_id)
            if trace:
                trace.spans.append(span)
                trace.total_cost_usd += span.cost_usd
    
    def end_trace(self, trace_id: str) -> TraceData:
        """结束Trace并导出"""
        trace = self._local.pop(trace_id, None)
        if trace:
            trace.end_time = time.time()
            if self.export_fn:
                self.export_fn(trace)
        return trace
    
    def log_user_feedback(self, trace_id: str, score: int, comment: str = None):
        """记录用户反馈"""
        trace = self._local.get(trace_id)
        if trace:
            trace.user_feedback = score
            trace.quality_scores["user_comment"] = comment


class CostTracker:
    """LLM成本追踪器"""
    
    # 模型定价表（2026年5月）
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},        # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
        "deepseek-v3": {"input": 0.27, "output": 1.10},
        "text-embedding-3-small": {"input": 0.02, "output": 0},
        "text-embedding-3-large": {"input": 0.13, "output": 0},
    }
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """计算单次调用成本"""
        pricing = self.PRICING.get(model)
        if not pricing:
            return 0.0
        
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        return round(cost, 6)
    
    def generate_cost_report(
        self,
        traces: list,
        period: str = "daily"
    ) -> dict:
        """生成成本报告"""
        total_cost = 0
        model_costs = {}
        user_costs = {}
        
        for trace in traces:
            total_cost += trace.total_cost_usd
            
            user_id = trace.user_id or "anonymous"
            user_costs[user_id] = user_costs.get(user_id, 0) + trace.total_cost_usd
            
            for span in trace.spans:
                if span.model:
                    model = span.model
                    model_costs[model] = model_costs.get(model, 0) + span.cost_usd
        
        return {
            "period": period,
            "total_cost_usd": round(total_cost, 4),
            "by_model": dict(sorted(model_costs.items(), key=lambda x: -x[1])),
            "by_user": dict(sorted(user_costs.items(), key=lambda x: -x[1])[:10]),
            "average_cost_per_request": round(total_cost / max(len(traces), 1), 6),
        }


class QualityEvaluator:
    """LLM输出质量评估器（异步Pipeline）"""
    
    def __init__(self):
        self.evaluators = []
    
    def register(self, evaluator_fn: Callable, name: str):
        """注册评估器"""
        self.evaluators.append({"fn": evaluator_fn, "name": name})
    
    def evaluate(self, trace: TraceData) -> dict:
        """执行所有评估"""
        results = {}
        for eval in self.evaluators:
            try:
                score = eval["fn"](trace)
                results[eval["name"]] = score
            except Exception as e:
                results[eval["name"]] = {"error": str(e)}
        
        trace.quality_scores.update(results)
        return results


# ====== 使用示例 ======

# 初始化
tracer = LLMTracer(export_fn=lambda t: print(f"Exported trace: {t.trace_id}"))
cost_tracker = CostTracker()

# 模拟一次RAG查询的完整Trace
trace_id = tracer.start_trace(user_id="user_123", session_id="session_456")

# Span 1: 查询理解
with tracer.trace_span(trace_id, "llm_call", model="gpt-4o-mini") as span:
    # 模拟LLM调用
    span.input_tokens = 150
    span.output_tokens = 50
    span.cost_usd = cost_tracker.calculate_cost("gpt-4o-mini", 150, 50)
    span.output_data = {"intent": "financial_analysis"}

# Span 2: 文档检索
with tracer.trace_span(trace_id, "retrieval") as span:
    span.metadata = {"retriever": "pinecone", "top_k": 5, "reranker": True}
    span.output_data = {"document_count": 5, "latency_ms": 340}

# Span 3: LLM生成
with tracer.trace_span(trace_id, "llm_call", model="gpt-4o") as span:
    span.input_tokens = 3200
    span.output_tokens = 850
    span.cost_usd = cost_tracker.calculate_cost("gpt-4o", 3200, 850)
    span.output_data = {"response_length": 850}

# 结束Trace
final_trace = tracer.end_trace(trace_id)

# 输出示例
print(f"Trace ID: {final_trace.trace_id}")
print(f"Total Cost: ${final_trace.total_cost_usd:.6f}")
print(f"Total Spans: {len(final_trace.spans)}")
print(f"Total Duration: {(final_trace.end_time - final_trace.start_time)*1000:.0f}ms")
```

---

## 五、高级实践：自动化质量评估Pipeline

### 5.1 RAGAS自动评估集成

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


class RAGQualityPipeline:
    """基于RAGAS的RAG质量自动评估Pipeline"""
    
    def __init__(self, sample_rate: float = 0.1):
        """
        Args:
            sample_rate: 采样率，生产环境不需要评估所有请求
        """
        self.sample_rate = sample_rate
        self.evaluation_buffer = []
    
    def add_to_buffer(self, trace: dict):
        """将Trace加入评估缓冲区"""
        import random
        if random.random() < self.sample_rate:
            self.evaluation_buffer.append(trace)
    
    def run_evaluation(self) -> dict:
        """批量执行评估"""
        if not self.evaluation_buffer:
            return {"status": "no_samples"}
        
        # 转换为RAGAS格式
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        for trace in self.evaluation_buffer:
            eval_data["question"].append(trace["query"])
            eval_data["answer"].append(trace["response"])
            eval_data["contexts"].append(trace["retrieved_docs"])
            eval_data["ground_truth"].append(trace.get("expected_answer", ""))
        
        dataset = Dataset.from_dict(eval_data)
        
        # 执行评估
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,        # 回答是否忠实于上下文
                answer_relevancy,    # 回答是否与问题相关
                context_precision,   # 检索上下文的精确度
                context_recall,      # 检索上下文的召回率
            ],
        )
        
        self.evaluation_buffer.clear()
        
        return {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"],
            "sample_count": len(eval_data["question"]),
        }
```

### 5.2 告警规则设计

```yaml
# LLM应用告警规则
alerts:
  # 质量告警
  - name: "幻觉率过高"
    condition: "hallucination_rate_5m > 0.15"
    severity: critical
    action: "通知团队 + 自动切换到更保守的Prompt模板"
    
  - name: "回答相关性下降"
    condition: "avg_relevance_score_10m < 0.6"
    severity: warning
    action: "通知团队 + 触发RAG参数调优"
  
  - name: "用户满意度暴跌"
    condition: "avg_user_rating_1h < 3.0"
    severity: critical
    action: "通知团队 + 人工审核最近100条对话"

  # 性能告警
  - name: "P99延迟过高"
    condition: "p99_latency_5m > 15000"  # 15秒
    severity: warning
    action: "检查模型服务状态 + 考虑降级到更小模型"
    
  - name: "Token消耗异常"
    condition: "token_usage_rate_5m > baseline * 3"
    severity: critical
    action: "检查是否有Prompt注入攻击或死循环"

  # 成本告警
  - name: "日成本超预算"
    condition: "daily_cost_usd > 500"
    severity: warning
    action: "通知团队 + 开启更激进的缓存策略"
    
  - name: "单用户成本异常"
    condition: "user_cost_1h > 50"
    severity: warning
    action: "检查用户行为是否异常（可能是循环调用）"
```

---

## 六、从0到1的实施路线图

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LLM可观测性实施路线图（4周计划）                              │
│                                                                         │
│  Week 1: 基础Tracing                                                     │
│  ├── 集成Langfuse SDK（或自建Trace收集器）                               │
│  ├── 记录每次LLM调用的输入/输出/Token/延迟                                │
│  ├── 建立基础Trace Dashboard                                             │
│  └── 产出：所有LLM调用可追溯                                             │
│                                                                         │
│  Week 2: 成本监控                                                        │
│  ├── 实现按请求/用户/模型的成本归因                                        │
│  ├── 设置成本告警规则                                                    │
│  ├── 建立成本优化基线                                                    │
│  └── 产出：知道每分钱花在哪里                                             │
│                                                                         │
│  Week 3: 质量评估                                                        │
│  ├── 集成RAGAS自动评估（采样10%请求）                                     │
│  ├── 建立质量趋势Dashboard                                               │
│  ├── 设置质量告警规则                                                    │
│  └── 产出：自动检测幻觉和低质量回答                                       │
│                                                                         │
│  Week 4: 用户反馈闭环                                                    │
│  ├── 前端添加👍/👎反馈按钮                                               │
│  ├── 反馈与Trace关联                                                     │
│  ├── 基于反馈数据优化Prompt和RAG参数                                      │
│  └── 产出：数据驱动的持续优化循环                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 七、总结

LLM应用可观测性不是一个可选项，而是生产环境的**必选项**。没有可观测性的LLM应用就像没有仪表盘的飞机——你可能在飞，但你不知道高度、速度和剩余油量。

**三个核心原则**：

1. **Trace一切**：每个用户请求的完整生命周期都应该可追溯，从输入到输出，从Prompt到每个Token的成本
2. **质量>性能**：对LLM应用来说，"回答是否正确"比"响应是否够快"更重要，质量指标应放在性能指标之前
3. **闭环驱动优化**：可观测性的终极目标不是"看到问题"，而是"自动发现问题并驱动优化"——从数据采集到告警到自动调优的完整闭环

> 在LLM应用的时代，**可观测性就是竞争力**。那些能快速发现并修复幻觉、能精确计算ROI、能基于用户反馈持续迭代的团队，将在AI应用的下半场竞争中占据绝对优势。
