---
title: "AI应用可观测性体系设计：从日志到Trace的全链路监控实战"
description: "系统设计LLM应用的可观测性体系，覆盖LangSmith/LangFuse/Phoenix等主流工具对比、Trace采集架构、评估指标体系与生产告警策略"
date: 2026-05-30
author: "RiceBall-15"
category: "engineering"
subCategory: "infra"
tags: ["可观测性", "Observability", "LLM监控", "LangSmith", "LangFuse", "MLOps", "Trace"]
draft: false
---

# AI应用可观测性体系设计：从日志到Trace的全链路监控实战

## 一、引言：LLM应用的"黑盒"困境

### 1.1 传统监控 vs LLM可观测性

传统Web应用的监控体系相对成熟：请求日志、APM链路追踪、指标采集、告警规则——这些工具已经发展了十几年。但LLM应用带来了一系列全新的挑战：

| 维度 | 传统应用 | LLM应用 |
|------|---------|---------|
| 输出确定性 | 相同输入 → 相同输出 | 相同输入 → 不同输出 |
| 错误类型 | 超时、500错误、空指针 | 幻觉、偏见、不相关回答 |
| 延迟特征 | 毫秒级，相对稳定 | 秒级，随token数波动 |
| 成本模型 | 按请求计费 | 按token计费，波动大 |
| 质量评估 | 可以自动化断言 | 需要人工/LLM评判 |

这意味着传统的"看看日志、看看指标"已经远远不够了。LLM应用需要一套**专门设计的可观测性体系**。

### 1.2 可观测性三大支柱在LLM场景的演进

传统的可观测性三大支柱——**Metrics（指标）、Logs（日志）、Traces（链路追踪）**——在LLM场景下需要重新定义。传统应用的监控关注的是"系统是否正常运行"，而LLM应用的监控还需要回答"模型的回答质量如何"、"回答是否安全"、"推理成本是否合理"这些全新维度的问题。

```
┌─────────────────────────────────────────────────────┐
│                LLM可观测性体系                        │
├──────────────┬──────────────┬───────────────────────┤
│   Metrics    │    Logs      │      Traces           │
├──────────────┼──────────────┼───────────────────────┤
│ • Token/请求  │ • Prompt模板  │ • Agent执行链路        │
│ • 延迟分布    │ • 完整对话    │ • Tool调用序列         │
│ • 错误率     │ • 幻觉标记    │ • 中间推理过程         │
│ • 成本统计    │ • 人工反馈    │ • 分支决策路径         │
│ • 质量评分    │ • 安全事件    │ • 子Agent协作         │
│ • 吞吐量     │ • 模型切换日志 │ • 异常重试链路         │
└──────────────┴──────────────┴───────────────────────┘
```

## 二、Trace架构设计：LLM应用的"全息影像"

### 2.1 Trace层级模型

LLM应用的Trace设计需要捕捉多层信息：

```
Trace (一次用户请求的完整生命周期)
├── Span: HTTP Request (入口)
│   ├── Span: Auth Middleware
│   ├── Span: Intent Classification
│   │   └── LLM Call: GPT-4o-mini (意图识别)
│   ├── Span: RAG Pipeline
│   │   ├── Span: Query Rewriting
│   │   │   └── LLM Call: GPT-4o-mini (查询改写)
│   │   ├── Span: Vector Search
│   │   │   └── DB Query: Qdrant (向量检索)
│   │   ├── Span: Reranking
│   │   │   └── LLM Call: Cohere Rerank
│   │   └── Span: Context Assembly
│   ├── Span: Main Reasoning
│   │   ├── LLM Call: GPT-4o (主推理)
│   │   ├── Tool Call: WebSearch
│   │   └── LLM Call: GPT-4o (整合回答)
│   └── Span: Response Formatting
```

### 2.2 核心数据模型设计

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

@dataclass
class LLMSpan:
    """LLM调用的最小追踪单元"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    
    # LLM特有字段
    model: str = ""                    # 使用的模型
    provider: str = ""                 # 提供商 (openai/anthropic/local)
    prompt_tokens: int = 0             # 输入token数
    completion_tokens: int = 0         # 输出token数
    total_tokens: int = 0
    
    # 内容字段（需要脱敏处理）
    input_messages: List[Dict] = field(default_factory=list)
    output_text: str = ""
    
    # 质量字段
    finish_reason: str = ""            # stop/length/content_filter
    latency_ms: float = 0
    time_to_first_token_ms: float = 0  # TTFT
    tokens_per_second: float = 0       # 吞吐速度
    
    # 成本字段
    cost_usd: float = 0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # 时间戳
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

@dataclass
class AgentSpan:
    """Agent执行的追踪单元"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    
    # Agent特有字段
    agent_type: str = ""               # react/planner/executor
    iteration: int = 0                 # 当前迭代轮次
    max_iterations: int = 10
    
    # 工具调用
    tool_name: str = ""
    tool_input: Any = None
    tool_output: Any = None
    tool_latency_ms: float = 0
    
    # 推理链
    thought: str = ""                  # Agent的思考过程
    action: str = ""                   # Agent的行动
    observation: str = ""              # 行动的结果
    
    # 状态
    status: str = "pending"            # pending/running/completed/failed
    error: Optional[str] = None
    
    # 时间戳
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
```

### 2.3 异步采集架构

在高并发场景下，Trace采集不能阻塞主流程。推荐使用**异步+采样**架构：

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (Python/Node)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ LLM Client  │  │ Tool Executor│  │ Agent Loop  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              ┌───────────────────┐                      │
│              │  Trace Collector   │                      │
│              │  (内存缓冲区)      │                      │
│              └─────────┬─────────┘                      │
│                        │                                │
│              ┌─────────▼─────────┐                      │
│              │  Sampling Strategy │                      │
│              │  • 全量: 错误请求   │                      │
│              │  • 概率: 正常请求   │                      │
│              │  • 保留: 高价值请求  │                      │
│              └─────────┬─────────┘                      │
└────────────────────────┼────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Message Queue     │
              │   (Redis/Kafka)     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Trace Processor   │
              │   (异步消费+存储)    │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ ClickHouse│  │ PostgreSQL│  │ S3/MinIO │
    │ (指标)    │  │ (Trace)  │  │ (原始日志)│
    └──────────┘  └──────────┘  └──────────┘
```

### 2.4 采样策略实现

```python
import random
from collections import defaultdict
from typing import Optional

class AdaptiveSampler:
    """自适应采样器：根据请求特征动态调整采样率"""
    
    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
        self.error_buffer = []  # 错误请求保留最近N条
        self.user_feedback = defaultdict(list)  # 用户反馈缓存
    
    def should_sample(self, trace_context: dict) -> bool:
        """决定是否采样当前请求"""
        
        # 规则1: 错误请求100%采样
        if trace_context.get("has_error", False):
            return True
        
        # 规则2: 包含敏感关键词的请求100%采样
        if trace_context.get("contains_pii", False):
            return True
        
        # 规则3: 高延迟请求100%采样
        if trace_context.get("latency_ms", 0) > 10000:
            return True
        
        # 规则4: 用户标记的低质量回答100%采样
        if trace_context.get("user_feedback") == "negative":
            return True
        
        # 规则5: 按概率采样
        return random.random() < self.base_rate
    
    def adjust_rate(self, error_rate: float, latency_p99: float):
        """根据系统状态动态调整基础采样率"""
        if error_rate > 0.05:  # 错误率>5%
            self.base_rate = min(self.base_rate * 1.5, 1.0)
        elif error_rate < 0.01 and latency_p99 < 3000:
            self.base_rate = max(self.base_rate * 0.8, 0.01)
```

## 三、评估指标体系：从"看到"到"看懂"

### 3.1 LLM特有的质量指标

传统的APM指标（延迟、错误率、吞吐量）只是基础。LLM应用需要一套全新的质量指标体系：

| 指标类别 | 指标名称 | 计算方式 | 健康阈值 |
|---------|---------|---------|---------|
| **质量** | 幻觉率 | 人工抽检/LLM评判 | <5% |
| **质量** | 相关性评分 | 语义相似度/RAGAS | >0.8 |
| **质量** | 拒绝率 | 无法回答的请求比例 | <10% |
| **安全** | 拒答触发率 | 安全策略拦截比例 | 监控趋势 |
| **效率** | TTFT | 首token延迟 | <1s |
| **效率** | Tokens/s | 生成速度 | >30 t/s |
| **效率** | 接受率 | Speculative Decoding | >0.6 |
| **成本** | Cost/1K tokens | 平均每千token成本 | 监控趋势 |
| **成本** | Cache命中率 | Prompt Cache命中比例 | >40% |

### 3.2 实时质量评估流水线

```python
class QualityEvaluator:
    """LLM回答质量的实时评估器"""
    
    def __init__(self):
        self.judge_model = "gpt-4o-mini"  # 用小模型评判
        self.cache = {}  # 评估结果缓存
    
    async def evaluate(self, trace: LLMSpan) -> dict:
        """对单次LLM调用进行多维度评估"""
        
        scores = {}
        
        # 1. 自动化指标
        scores["relevance"] = self._compute_relevance(
            trace.input_messages, 
            trace.output_text
        )
        scores["coherence"] = self._compute_coherence(trace.output_text)
        scores["factuality"] = self._check_factuality(trace.output_text)
        
        # 2. 安全检测
        scores["safety"] = self._check_safety(trace.output_text)
        scores["pii_leak"] = self._detect_pii(trace.output_text)
        
        # 3. 效率指标
        scores["latency_ratio"] = self._compute_latency_ratio(trace)
        scores["cost_efficiency"] = trace.cost_usd / max(trace.completion_tokens, 1)
        
        # 4. 综合评分
        scores["overall"] = self._compute_overall_score(scores)
        
        # 5. 异常标记
        scores["anomalies"] = self._detect_anomalies(scores, trace)
        
        return scores
    
    def _compute_relevance(self, messages: list, response: str) -> float:
        """基于向量相似度计算相关性"""
        # 使用embedding模型计算语义相似度
        # 这里简化为关键词匹配
        query = messages[-1]["content"]
        # 实际实现应使用embedding similarity
        return 0.85  # placeholder
    
    def _check_safety(self, text: str) -> dict:
        """安全检测：有害内容、偏见、暴力等"""
        safety_flags = {
            "harmful": False,
            "biased": False,
            "violent": False,
            "sexual": False,
            "self_harm": False,
        }
        # 实际实现应使用安全分类模型
        return safety_flags
    
    def _detect_anomalies(self, scores: dict, trace: LLMSpan) -> list:
        """异常检测"""
        anomalies = []
        
        if trace.latency_ms > 5000:
            anomalies.append({
                "type": "high_latency",
                "severity": "warning",
                "detail": f"Latency {trace.latency_ms}ms exceeds 5s threshold"
            })
        
        if trace.finish_reason == "length":
            anomalies.append({
                "type": "truncated_output",
                "severity": "warning",
                "detail": "Output was truncated due to max_tokens limit"
            })
        
        if scores.get("safety", {}).get("harmful"):
            anomalies.append({
                "type": "harmful_content",
                "severity": "critical",
                "detail": "Potentially harmful content detected"
            })
        
        return anomalies
```

### 3.3 RAG质量评估框架

对于RAG应用，需要专门的评估指标：

```python
class RAGEvaluator:
    """RAG系统的专用评估器"""
    
    async def evaluate(self, rag_trace: dict) -> dict:
        """
        rag_trace包含:
        - query: 用户查询
        - retrieved_docs: 检索到的文档
        - response: 生成的回答
        - context: 构建的上下文
        """
        
        return {
            # 检索质量
            "context_precision": self._context_precision(
                rag_trace["query"], 
                rag_trace["retrieved_docs"]
            ),
            "context_recall": self._context_recall(
                rag_trace["query"],
                rag_trace["retrieved_docs"],
                rag_trace["ground_truth"]  # 如果有ground truth
            ),
            
            # 生成质量
            "answer_relevancy": self._answer_relevancy(
                rag_trace["query"],
                rag_trace["response"]
            ),
            "faithfulness": self._faithfulness(
                rag_trace["context"],
                rag_trace["response"]
            ),
            
            # 综合质量（RAGAS Score）
            "ragas_score": None,  # 计算后填入
        }
    
    def _faithfulness(self, context: str, response: str) -> float:
        """忠实度：回答是否基于提供的上下文"""
        # 核心问题：回答是否引入了上下文之外的信息
        # 实现方式：将回答分解为claims，检查每个claim是否被上下文支持
        claims = self._extract_claims(response)
        supported = sum(1 for c in claims if self._is_supported(c, context))
        return supported / max(len(claims), 1)
```

## 四、工具选型：LangSmith vs LangFuse vs Phoenix

### 4.1 主流工具对比

| 维度 | LangSmith | LangFuse | Phoenix (Arize) | 自建方案 |
|------|-----------|----------|----------------|---------|
| **部署方式** | SaaS | 开源自部署/SaaS | 开源自部署 | 完全自建 |
| **LLM支持** | LangChain生态 | 框架无关 | 框架无关 | 任意 |
| **评估能力** | 内置LLM评判 | 内置评估 | LLM Evals | 自定义 |
| **成本** | 按trace计费 | 开源免费 | 开源免费 | 运维成本 |
| **数据安全** | 数据上云 | 可私有化 | 可私有化 | 完全可控 |
| **生态** | LangChain深度集成 | 多语言SDK | Python为主 | 无依赖 |
| **学习曲线** | 低 | 中 | 中-高 | 高 |
| **社区活跃度** | 高 | 高 | 中 | - |

### 4.2 LangFuse自部署架构

对于数据安全要求较高的场景，推荐LangFuse自部署：

```yaml
# docker-compose.yml
version: '3.8'

services:
  langfuse-server:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/langfuse
      - REDIS_URL=redis://redis:6379
      - NEXTAUTH_SECRET=your-secret-key
      - NEXTAUTH_URL=http://localhost:3000
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=langfuse
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 4.3 集成代码示例

```python
# LangFuse集成
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-xxx",
    secret_key="sk-xxx",
    host="http://localhost:3000"  # 自部署地址
)

@observe(as_type="generation")
def call_llm(prompt: str, model: str = "gpt-4o"):
    """LLM调用的自动追踪"""
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # LangFuse自动记录：
    # - 输入prompt
    # - 输出response
    # - 延迟
    # - Token使用量
    # - 模型信息
    return response.choices[0].message.content

@observe()  # 自动创建span
def rag_pipeline(query: str):
    """RAG流水线的链路追踪"""
    # 检索阶段
    docs = retrieve_documents(query)
    
    # 生成阶段
    context = "\n".join([d.content for d in docs])
    response = call_llm(f"基于以下信息回答：{context}\n\n问题：{query}")
    
    # 添加自定义评分
    langfuse_context.score_current_trace(
        name="relevance",
        value=0.85,
        comment="相关性评分"
    )
    
    return response
```

## 五、告警与运营：从被动响应到主动预防

### 5.1 告警规则设计

```python
# 告警规则配置
ALERT_RULES = {
    # === 质量告警 ===
    "high_hallucination_rate": {
        "metric": "hallucination_rate_5min",
        "condition": "> 0.1",  # 幻觉率>10%
        "severity": "critical",
        "channels": ["slack", "pagerduty"],
        "cooldown": "10m",
    },
    
    "low_relevance_score": {
        "metric": "avg_relevance_score_10min",
        "condition": "< 0.6",  # 平均相关性<0.6
        "severity": "warning",
        "channels": ["slack"],
        "cooldown": "30m",
    },
    
    # === 性能告警 ===
    "high_ttft": {
        "metric": "p95_ttft_5min",
        "condition": "> 3000",  # P95首token延迟>3s
        "severity": "warning",
        "channels": ["slack"],
        "cooldown": "15m",
    },
    
    "low_throughput": {
        "metric": "tokens_per_second_5min",
        "condition": "< 20",  # 吞吐速度<20 t/s
        "severity": "warning",
        "channels": ["slack"],
        "cooldown": "15m",
    },
    
    # === 成本告警 ===
    "high_cost": {
        "metric": "cost_per_hour",
        "condition": "> $50",  # 每小时成本>$50
        "severity": "info",
        "channels": ["slack"],
        "cooldown": "1h",
    },
    
    "token_budget_exceeded": {
        "metric": "daily_token_usage",
        "condition": "> 10000000",  # 日token使用量>1000万
        "severity": "warning",
        "channels": ["slack", "email"],
        "cooldown": "24h",
    },
    
    # === 安全告警 ===
    "safety_violation": {
        "metric": "safety_violations_5min",
        "condition": "> 0",
        "severity": "critical",
        "channels": ["slack", "pagerduty", "email"],
        "cooldown": "0m",  # 不冷却，每次都告警
    },
}
```

### 5.2 成本监控仪表盘

```
┌─────────────────────────────────────────────────────────┐
│                  LLM成本监控仪表盘                        │
├─────────────────────────────────────────────────────────┤
│  今日成本: $47.83    预算: $100    剩余: $52.17        │
│  ████████████████░░░░░░░░  47.8%                        │
├─────────────────────────────────────────────────────────┤
│  模型分布:                                               │
│  GPT-4o:        $32.15 (67.2%)  ████████████████        │
│  GPT-4o-mini:   $8.92  (18.7%)  █████                   │
│  Claude-3.5:    $4.56  (9.5%)   ███                     │
│  其他:          $2.20  (4.6%)   █                       │
├─────────────────────────────────────────────────────────┤
│  功能分布:                                               │
│  Chat:          $28.40 (59.4%)  ██████████████          │
│  RAG:           $12.30 (25.7%)  ██████                  │
│  Agent:         $4.80  (10.0%)  ███                     │
│  评估:          $2.33  (4.9%)   █                       │
├─────────────────────────────────────────────────────────┤
│  趋势 (近7天):                                          │
│  Mon  $42 | Tue  $38 | Wed  $55 | Thu  $41             │
│  Fri  $48 | Sat  $22 | Sun  --                         │
└─────────────────────────────────────────────────────────┘
```

### 5.3 异常检测与自动降级

```python
class AutoDegradation:
    """LLM应用的自动降级策略"""
    
    def __init__(self):
        self.state = "healthy"  # healthy/degraded/critical
        self.degradation_rules = {
            "degraded": {
                "max_latency_ms": 5000,
                "max_error_rate": 0.05,
                "fallback_model": "gpt-4o-mini",
                "max_tokens": 1024,
            },
            "critical": {
                "max_latency_ms": 10000,
                "max_error_rate": 0.15,
                "fallback_model": "gpt-4o-mini",
                "max_tokens": 512,
                "enable_caching": True,
            }
        }
    
    def check_and_adapt(self, metrics: dict):
        """根据当前指标决定是否降级"""
        
        if metrics["error_rate"] > 0.15 or metrics["p95_latency"] > 10000:
            self._enter_critical()
        elif metrics["error_rate"] > 0.05 or metrics["p95_latency"] > 5000:
            self._enter_degraded()
        elif self.state != "healthy":
            self._recover()
    
    def _enter_degraded(self):
        if self.state == "healthy":
            self.state = "degraded"
            self._switch_model("gpt-4o-mini")
            self._enable_caching(True)
            self._send_alert("降级到gpt-4o-mini")
    
    def _enter_critical(self):
        self.state = "critical"
        self._switch_model("gpt-4o-mini")
        self._set_max_tokens(512)
        self._enable_caching(True)
        self._enable_rate_limiting(True)
        self._send_alert("进入临界状态，已降级并限流")
    
    def _recover(self):
        self.state = "healthy"
        self._switch_model("gpt-4o")
        self._set_max_tokens(4096)
        self._enable_caching(False)
        self._send_alert("恢复正常")
```

## 六、实战案例：构建完整的可观测性体系

### 6.1 技术栈推荐

```
┌─────────────────────────────────────────────────┐
│                  推荐技术栈                       │
├─────────────────────────────────────────────────┤
│  数据采集层:                                     │
│  • LangFuse (Trace采集 + 评估)                   │
│  • OpenTelemetry (标准化协议)                    │
│  • Sentry (错误追踪)                            │
├─────────────────────────────────────────────────┤
│  数据存储层:                                     │
│  • ClickHouse (时序指标，高性能查询)              │
│  • PostgreSQL (Trace详情，关系查询)               │
│  • Redis (实时指标缓存)                          │
├─────────────────────────────────────────────────┤
│  可视化层:                                       │
│  • Grafana (指标仪表盘)                          │
│  • LangFuse UI (Trace详情)                      │
│  • 自定义Web UI (业务指标)                       │
├─────────────────────────────────────────────────┤
│  告警层:                                         │
│  • Alertmanager (Prometheus告警)                │
│  • PagerDuty (On-Call管理)                      │
│  • Slack (通知)                                 │
└─────────────────────────────────────────────────┘
```

### 6.2 实施路线图

| 阶段 | 目标 | 工具 | 时间 |
|------|------|------|------|
| Phase 1 | 基础Trace采集 | LangFuse SDK | 1周 |
| Phase 2 | 质量评估 | LangFuse Evaluation | 2周 |
| Phase 3 | 成本监控 | Grafana + ClickHouse | 1周 |
| Phase 4 | 告警体系 | Alertmanager + Slack | 1周 |
| Phase 5 | 自动降级 | 自定义策略引擎 | 2周 |
| Phase 6 | 持续优化 | A/B测试框架 | 持续 |

### 6.3 常见踩坑经验

在实际搭建LLM可观测性体系的过程中，有几个容易踩的坑值得特别注意：

**坑1：过度记录导致性能下降**

很多团队在初期会倾向于记录所有信息——完整的prompt、完整的response、所有中间步骤。这在低并发时没有问题，但在高并发场景下会严重影响主流程性能。

解决方案：采用**分层记录策略**。核心指标（延迟、token数、错误状态）必须实时记录，完整内容可以异步写入，且必须配合采样策略。

**坑2：评估指标的"虚假繁荣"**

有些团队发现他们的"相关性评分"长期维持在0.9以上，但用户投诉却在增加。原因是评估prompt设计不当，LLM评判器倾向于给出高分。

解决方案：定期用人工标注数据校准自动评估器，建立**评估质量的评估机制**——这是很多人忽略的元评估问题。

**坑3：Prompt版本与Trace脱节**

当Prompt模板更新后，历史Trace中的prompt信息就变成了"过时数据"。如果不能将Trace与Prompt版本关联，就无法准确分析某个Prompt版本的实际效果。

解决方案：在Trace采集时自动注入Prompt版本号，建立Prompt版本 → Trace → 质量评分的完整关联链。

**坑4：多模型场景的成本归因**

当系统使用多个模型（如GPT-4o做主推理、GPT-4o-mini做预处理、Cohere做重排序）时，单个请求的成本需要在多个模型之间分摊。如果不做好归因，就无法准确计算每个功能模块的成本效率。

解决方案：在Trace中记录每个模型调用的成本，并按功能模块聚合，建立**模块级成本归因**体系。

## 七、总结

### 7.1 核心原则

1. **Trace是一等公民**：LLM应用的每一步都应该被追踪，从用户输入到最终输出
2. **质量 > 数量**：监控100个高质量指标比监控1000个低质量指标更有价值
3. **异步优先**：Trace采集不能阻塞主流程，必须异步+采样
4. **成本可控**：监控成本本身也是成本优化的一部分
5. **安全第一**：PII检测和内容安全必须内置在采集流程中

### 7.2 常见陷阱

| 陷阱 | 后果 | 解决方案 |
|------|------|---------|
| 无采样，全量存储 | 存储成本爆炸 | 自适应采样策略 |
| 只看延迟不看质量 | 上线后才发现幻觉 | 质量评估前置 |
| 硬编码评估阈值 | 误报/漏报频繁 | 动态阈值+人工校准 |
| 忽略Prompt版本管理 | 无法复现问题 | Prompt + Trace联动 |
| 告警疲劳 | 真正的问题被忽略 | 分级告警+冷却机制 |

可观测性不是一次性建设，而是持续演进的过程。从最简单的Trace采集开始，逐步建立完整的质量评估和告警体系，最终实现LLM应用的**可观测、可评估、可优化**。
