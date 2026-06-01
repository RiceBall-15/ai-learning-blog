---
title: "AI应用的事件溯源架构：LLM交互链路的完整审计与回放"
description: "深入解析事件溯源模式在AI应用中的实践，覆盖LLM调用审计、对话回放、调试排错、合规追踪等核心场景"
date: 2026-05-31
author: "RiceBall-15"
category: "architecture"
subCategory: distributed
tags: ["事件溯源", "Event Sourcing", "LLM审计", "AI应用架构", "可观测性", "对话回放"]
draft: false
---

# AI应用的事件溯源架构：LLM交互链路的完整审计与回放

## 核心问题：为什么LLM应用需要事件溯源？

传统Web应用的审计日志是"记结果"——用户做了什么操作，数据库变成了什么状态。但LLM应用完全不同：

```
传统Web应用：
  用户请求 → 业务逻辑 → 数据库写入 → 返回结果
  审计点：数据库变更（可直接回溯）

LLM应用：
  用户输入 → 上下文组装 → LLM调用 → 后处理 → 工具调用 → LLM再次调用 → ...
  审计点：？（LLM是黑盒，中间过程在哪里？）
```

一个典型的AI Agent可能在一次用户请求中产生：
- 1次用户输入处理
- 3次LLM调用
- 2次RAG检索
- 5次工具调用
- 2次中间结果处理

如果只记录最终结果，你完全无法回答：
- "为什么模型给出了这个回答？"
- "哪一步出了问题？"
- "能还原当时的完整上下文吗？"
- "这次调用的Token消耗是多少？"

**事件溯源（Event Sourcing）** 为这些问题提供了优雅的答案。

---

## 一、事件溯源的核心理念

### 1.1 传统日志 vs 事件溯源

```
传统日志记录：

  [INFO] 2026-05-31 10:00:01 - 用户A发起查询
  [INFO] 2026-05-31 10:00:02 - 检索到3个相关文档
  [INFO] 2026-05-31 10:00:03 - LLM生成回答
  [INFO] 2026-05-31 10:00:04 - 返回结果给用户

  特点：文本格式，难以查询，信息丢失，无法回放


事件溯源记录：

  Event 1: { type: "user.input", data: {...}, timestamp: 10:00:01 }
  Event 2: { type: "retrieval.query", data: {...}, timestamp: 10:00:01 }
  Event 3: { type: "retrieval.results", data: [...], timestamp: 10:00:02 }
  Event 4: { type: "llm.call", data: {prompt: ..., response: ...}, timestamp: 10:00:02 }
  Event 5: { type: "llm.output", data: {...}, timestamp: 10:00:03 }
  Event 6: { type: "user.response", data: {...}, timestamp: 10:00:04 }

  特点：结构化，可查询，信息完整，可回放
```

### 1.2 事件溯源在LLM应用中的价值

| 价值维度 | 具体场景 | ROI |
|---------|---------|-----|
| **调试排错** | 模型输出异常时，回溯完整调用链 | ⭐⭐⭐⭐⭐ |
| **质量评估** | 分析不同Prompt/模型组合的效果差异 | ⭐⭐⭐⭐⭐ |
| **成本审计** | 精确追踪每次调用的Token消耗 | ⭐⭐⭐⭐ |
| **合规要求** | 满足GDPR、数据安全等合规需求 | ⭐⭐⭐⭐⭐ |
| **持续优化** | 基于历史数据优化Prompt和检索策略 | ⭐⭐⭐⭐ |
| **故障复盘** | 线上问题的完整还原和根因分析 | ⭐⭐⭐⭐⭐ |

---

## 二、LLM交互事件模型设计

### 2.1 事件类型体系

```
LLM交互事件类型层级：

LLMInteractionEvent
├── SessionEvent           # 会话级事件
│   ├── session.created    # 会话创建
│   ├── session.context    # 上下文变更
│   └── session.ended      # 会话结束
├── MessageEvent           # 消息级事件
│   ├── message.user       # 用户输入
│   ├── message.system     # 系统消息
│   └── message.assistant  # 模型输出
├── RetrievalEvent         # 检索级事件
│   ├── retrieval.query    # 检索请求
│   ├── retrieval.results  # 检索结果
│   └── retrieval.rerank   # 重排序结果
├── LLMCallEvent           # LLM调用级事件
│   ├── llm.request        # 调用请求（含完整Prompt）
│   ├── llm.response       # 模型响应
│   ├── llm.tool_call      # 工具调用
│   └── llm.tool_result    # 工具返回
└── PipelineEvent          # 流水线级事件
    ├── pipeline.start     # 流水线开始
    ├── pipeline.step      # 流水线步骤
    └── pipeline.complete  # 流水线完成
```

### 2.2 事件Schema定义

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

@dataclass
class LLMInteractionEvent:
    """LLM交互事件基类"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""                        # 事件类型
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""                        # 会话ID
    trace_id: str = ""                          # 链路追踪ID
    parent_event_id: Optional[str] = None       # 父事件ID
    metadata: dict = field(default_factory=dict) # 元数据
    data: dict = field(default_factory=dict)     # 事件数据

@dataclass
class LLMCallEvent(LLMInteractionEvent):
    """LLM调用事件"""
    event_type: str = "llm.call"
    
    # 调用信息
    model: str = ""                  # 模型名称
    provider: str = ""               # 服务提供商
    temperature: float = 0.0
    max_tokens: int = 0
    
    # 上下文信息
    system_prompt: str = ""          # 系统提示词
    messages: list = field(default_factory=list)  # 对话消息
    
    # 输出信息
    response: str = ""               # 模型输出
    finish_reason: str = ""          # 结束原因
    
    # Token统计
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # 性能数据
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0

@dataclass  
class RetrievalEvent(LLMInteractionEvent):
    """检索事件"""
    event_type: str = "retrieval.query"
    
    query: str = ""                           # 检索查询
    strategy: str = ""                        # 检索策略
    top_k: int = 0
    results: list = field(default_factory=list) # 检索结果
    scores: list = field(default_factory=list)  # 相关性分数
    latency_ms: float = 0.0
```

### 2.3 事件之间的关联关系

```
一次用户请求的事件链：

  ┌─ session.created (session_id: s1)
  │
  ├─ message.user (session_id: s1)
  │   data: { content: "帮我分析Q1的销售数据" }
  │
  ├─ retrieval.query (session_id: s1)
  │   data: { query: "Q1销售数据分析", strategy: "hybrid" }
  │
  ├─ retrieval.results (parent: retrieval.query)
  │   data: { results: [...3个文档], scores: [0.92, 0.87, 0.81] }
  │
  ├─ llm.request (session_id: s1)
  │   data: { model: "gpt-4o", prompt_tokens: 2500, ... }
  │
  ├─ llm.tool_call (parent: llm.request)
  │   data: { tool: "sql_executor", args: { query: "SELECT ..." } }
  │
  ├─ llm.tool_result (parent: llm.tool_call)  
  │   data: { result: [{month: "Jan", revenue: 120000}, ...] }
  │
  ├─ llm.response (parent: llm.request)
  │   data: { output: "Q1总营收380万，同比增长15%...", tokens: 450 }
  │
  └─ message.assistant (session_id: s1)
      data: { content: "根据数据分析..." }
```

---

## 三、事件存储与查询架构

### 3.1 存储架构设计

```
事件存储架构：

┌──────────────────────────────────────────────────┐
│                   API Gateway                     │
│            (事件写入 & 查询入口)                   │
└───────────┬──────────────────────┬────────────────┘
            │ 写入                  │ 查询
            ▼                      ▼
┌───────────────────┐    ┌───────────────────┐
│   事件写入服务     │    │   事件查询服务     │
│                   │    │                   │
│  1. 验证Schema    │    │  1. 时间范围查询   │
│  2. 事件序列化    │    │  2. 事件类型过滤   │
│  3. 发送到Kafka   │    │  3. 链路追踪查询   │
│  4. 写入主存储    │    │  4. 聚合分析      │
└────────┬──────────┘    └────────┬──────────┘
         │                        │
    ┌────┴────┐              ┌────┴────┐
    ▼         ▼              ▼         ▼
┌────────┐ ┌────────┐  ┌────────┐ ┌────────┐
│ Kafka  │ │S3/对象  │  │ClickH.│ │ ES/    │
│(实时流)│ │ 存储    │  │(分析)  │ │ 向量库  │
└────────┘ └────────┘  └────────┘ └────────┘
```

### 3.2 事件写入实现

```python
class EventStore:
    """事件存储"""
    
    def __init__(self, kafka_producer, clickhouse_client, s3_client):
        self.kafka = kafka_producer
        self.clickhouse = clickhouse_client
        self.s3 = s3_client
    
    async def append(self, event: LLMInteractionEvent):
        """追加事件（异步写入）"""
        # 1. Schema验证
        self._validate_event(event)
        
        # 2. 序列化
        serialized = self._serialize(event)
        
        # 3. 实时写入Kafka（供消费者处理）
        await self.kafka.send(
            topic=f"llm-events.{event.event_type}",
            key=event.session_id,
            value=serialized
        )
        
        # 4. 批量写入ClickHouse（用于分析查询）
        self._batch_insert(event)
        
        # 5. 大事件写入S3（完整数据存储）
        if self._is_large_event(event):
            s3_key = f"events/{event.session_id}/{event.event_id}.json"
            await self.s3.put_object(
                bucket="llm-events-archive",
                key=s3_key,
                body=serialized
            )
            # 在ClickHouse中存储引用
            event.metadata["s3_reference"] = s3_key
    
    def _batch_insert(self, event: LLMInteractionEvent):
        """批量写入ClickHouse"""
        batch = [{
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "session_id": event.session_id,
            "trace_id": event.trace_id,
            "model": getattr(event, "model", ""),
            "prompt_tokens": getattr(event, "prompt_tokens", 0),
            "completion_tokens": getattr(event, "completion_tokens", 0),
            "latency_ms": getattr(event, "latency_ms", 0),
            "data_summary": self._summarize(event.data),
        }]
        self.clickhouse.insert("llm_events", batch)
```

### 3.3 事件查询与分析

```python
class EventQueryService:
    """事件查询服务"""
    
    def get_conversation_trace(self, session_id: str) -> list:
        """获取完整对话链路"""
        events = self.clickhouse.query(f"""
            SELECT * FROM llm_events 
            WHERE session_id = '{session_id}'
            ORDER BY timestamp ASC
        """)
        return self._reconstruct_trace(events)
    
    def analyze_llm_calls(self, session_id: str) -> dict:
        """分析一次会话的LLM调用情况"""
        stats = self.clickhouse.query(f"""
            SELECT 
                model,
                COUNT(*) as call_count,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(completion_tokens) as total_completion_tokens,
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency
            FROM llm_events
            WHERE session_id = '{session_id}' 
              AND event_type = 'llm.response'
            GROUP BY model
        """)
        return stats
    
    def find_anomalies(self, time_range: str) -> list:
        """发现异常调用"""
        return self.clickhouse.query(f"""
            SELECT session_id, model, latency_ms, 
                   completion_tokens, finish_reason
            FROM llm_events
            WHERE event_type = 'llm.response'
              AND timestamp BETWEEN {time_range}
              AND (latency_ms > 10000 
                   OR finish_reason = 'length'
                   OR completion_tokens < 10)
            ORDER BY timestamp DESC
        """)
    
    def calculate_costs(self, time_range: str) -> dict:
        """计算Token成本"""
        return self.clickhouse.query(f"""
            SELECT 
                model,
                SUM(prompt_tokens) as input_tokens,
                SUM(completion_tokens) as output_tokens,
                SUM(prompt_tokens) * model_input_price(model) 
                    + SUM(completion_tokens) * model_output_price(model) 
                    as estimated_cost
            FROM llm_events
            WHERE event_type = 'llm.response'
              AND timestamp BETWEEN {time_range}
            GROUP BY model
            ORDER BY estimated_cost DESC
        """)
```

---

## 四、对话回放：AI应用的"时间机器"

对话回放是事件溯源最强大的能力之一——你可以完整还原用户与AI交互的每一刻。

### 4.1 回放架构

```
对话回放流程：

  ┌──────────────────────────────────────────────────────┐
  │                    回放请求                            │
  │        session_id: "s_20260531_001"                   │
  └──────────────────┬───────────────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────────────┐
  │              事件加载                                  │
  │  1. 从ClickHouse加载所有事件                           │
  │  2. 按时间戳排序                                       │
  │  3. 构建事件树（父子关系）                              │
  └──────────────────┬───────────────────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────────────┐
  │              回放渲染                                  │
  │                                                      │
  │  ┌────────────────────────────────────────────────┐  │
  │  │ [10:00:01] 用户输入                             │  │
  │  │ "帮我分析Q1的销售数据"                           │  │
  │  └────────────────────────────────────────────────┘  │
  │           │                                          │
  │  ┌────────▼───────────────────────────────────────┐  │
  │  │ [10:00:01] 检索                                │  │
  │  │ 策略: hybrid | 查询: "Q1销售数据"               │  │
  │  │ 结果: 3个文档 | 相关性: [0.92, 0.87, 0.81]     │  │
  │  └────────────────────────────────────────────────┘  │
  │           │                                          │
  │  ┌────────▼───────────────────────────────────────┐  │
  │  │ [10:00:02] LLM调用 #1                          │  │
  │  │ 模型: gpt-4o | Token: 2500 → 180              │  │
  │  │ 延迟: 1200ms | 输出: "我来分析..."             │  │
  │  │                                                 │  │
  │  │   ┌─────────────────────────────────────────┐  │  │
  │  │   │ 工具调用: sql_executor                    │  │  │
  │  │   │ SQL: SELECT month, revenue FROM sales   │  │  │
  │  │   │ 结果: [{Jan: 120K}, {Feb: 130K}, ...]   │  │  │
  │  │   └─────────────────────────────────────────┘  │  │
  │  └────────────────────────────────────────────────┘  │
  │           │                                          │
  │  ┌────────▼───────────────────────────────────────┐  │
  │  │ [10:00:03] LLM调用 #2                          │  │
  │  │ 模型: gpt-4o | Token: 3200 → 350              │  │
  │  │ 延迟: 1800ms | 输出: "Q1总营收380万..."       │  │
  │  └────────────────────────────────────────────────┘  │
  │                                                      │
  │  ┌────────────────────────────────────────────────┐  │
  │  │ 统计                                          │  │
  │  │ 总Token: 6230 | 总延迟: 3200ms | 估算成本: $0.08│ │
  │  └────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────┘
```

### 4.2 回放实现

```python
class ConversationReplayer:
    """对话回放器"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    def replay(self, session_id: str) -> ReplayResult:
        """完整回放一次对话"""
        # 加载所有事件
        events = self.event_store.get_conversation_trace(session_id)
        
        replay_steps = []
        total_tokens = {"prompt": 0, "completion": 0}
        total_latency = 0
        
        for event in events:
            step = self._render_event(event)
            replay_steps.append(step)
            
            # 累计统计
            if isinstance(event, LLMCallEvent):
                total_tokens["prompt"] += event.prompt_tokens
                total_tokens["completion"] += event.completion_tokens
                total_latency += event.latency_ms
        
        return ReplayResult(
            session_id=session_id,
            steps=replay_steps,
            summary={
                "total_events": len(events),
                "llm_calls": sum(1 for e in events if isinstance(e, LLMCallEvent)),
                "total_tokens": total_tokens,
                "total_latency_ms": total_latency,
                "estimated_cost": self._calculate_cost(total_tokens),
            }
        )
    
    def _render_event(self, event: LLMInteractionEvent) -> ReplayStep:
        """渲染单个事件为可视化步骤"""
        renderer_map = {
            "message.user": self._render_user_message,
            "message.assistant": self._render_assistant_message,
            "retrieval.query": self._render_retrieval_query,
            "retrieval.results": self._render_retrieval_results,
            "llm.request": self._render_llm_request,
            "llm.response": self._render_llm_response,
            "llm.tool_call": self._render_tool_call,
            "llm.tool_result": self._render_tool_result,
        }
        
        renderer = renderer_map.get(event.event_type, self._render_generic)
        return renderer(event)
```

---

## 五、基于事件的调试与诊断

### 5.1 问题诊断流程

```
LLM应用问题诊断流程：

  用户报告："回答质量下降了"
           │
           ▼
  ┌─────────────────────────────────────┐
  │ Step 1: 定位问题会话                │
  │ - 按时间范围 + 用户ID筛选事件       │
  │ - 识别异常会话（延迟高/Token异常）   │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │ Step 2: 回放对话链路                │
  │ - 加载完整事件序列                   │
  │ - 可视化每一步的输入/输出            │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │ Step 3: 对比分析                    │
  │ - 与正常对话对比                     │
  │ - 检查Prompt变化                    │
  │ - 检查检索质量                      │
  │ - 检查模型版本                      │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │ Step 4: 根因定位                    │
  │ ✗ 检索结果相关性下降（0.92→0.45）    │
  │ ✓ 系统提示词无变化                   │
  │ ✗ 模型切换导致输出风格变化           │
  └─────────────────────────────────────┘
```

### 5.2 自动化异常检测

```python
class AnomalyDetector:
    """基于事件流的异常检测"""
    
    def detect_anomalies(self, session_id: str) -> list[Anomaly]:
        events = self.event_store.get_conversation_trace(session_id)
        anomalies = []
        
        for event in events:
            if isinstance(event, LLMCallEvent):
                # 检查延迟异常
                if event.latency_ms > self._get_latency_threshold(event.model):
                    anomalies.append(Anomaly(
                        type="high_latency",
                        event_id=event.event_id,
                        detail=f"延迟 {event.latency_ms}ms 超过阈值",
                        severity="warning"
                    ))
                
                # 检查输出长度异常
                if event.completion_tokens < 10:
                    anomalies.append(Anomaly(
                        type="empty_output",
                        event_id=event.event_id,
                        detail=f"输出仅 {event.completion_tokens} Token",
                        severity="error"
                    ))
                
                # 检查finish_reason
                if event.finish_reason == "length":
                    anomalies.append(Anomaly(
                        type="output_truncated",
                        event_id=event.event_id,
                        detail="输出因达到max_tokens被截断",
                        severity="warning"
                    ))
            
            if isinstance(event, RetrievalEvent):
                # 检查检索质量
                if event.scores and max(event.scores) < 0.3:
                    anomalies.append(Anomaly(
                        type="low_relevance",
                        event_id=event.event_id,
                        detail=f"最高相关性分数仅 {max(event.scores)}",
                        severity="warning"
                    ))
        
        return anomalies
```

### 5.3 A/B测试分析

事件溯源为Prompt A/B测试提供了完美的数据基础：

```python
class PromptABAnalyzer:
    """基于事件的Prompt A/B测试分析"""
    
    def analyze(
        self, 
        experiment_id: str,
        group_a_prompt: str,
        group_b_prompt: str
    ) -> ABTestResult:
        
        # 从事件中筛选两个组的LLM调用
        group_a_events = self._get_events_by_prompt(
            experiment_id, group_a_prompt
        )
        group_b_events = self._get_events_by_prompt(
            experiment_id, group_b_prompt
        )
        
        def calc_metrics(events):
            latencies = [e.latency_ms for e in events]
            tokens = [e.completion_tokens for e in events]
            return {
                "count": len(events),
                "avg_latency": np.mean(latencies),
                "p95_latency": np.percentile(latencies, 95),
                "avg_output_tokens": np.mean(tokens),
                "avg_cost_per_call": np.mean([
                    e.prompt_tokens * 0.0025/1000 + 
                    e.completion_tokens * 0.01/1000 
                    for e in events
                ]),
            }
        
        metrics_a = calc_metrics(group_a_events)
        metrics_b = calc_metrics(group_b_events)
        
        return ABTestResult(
            group_a=metrics_a,
            group_b=metrics_b,
            recommendation=self._make_recommendation(metrics_a, metrics_b)
        )
```

---

## 六、合规与数据安全

### 6.1 GDPR合规实现

```python
class GDPRComplianceManager:
    """GDPR合规管理器"""
    
    def handle_deletion_request(self, user_id: str):
        """处理用户数据删除请求"""
        # 1. 找到用户所有会话
        sessions = self.event_store.query(
            "SELECT DISTINCT session_id FROM llm_events "
            f"WHERE user_id = '{user_id}'"
        )
        
        # 2. 匿名化事件数据
        for session in sessions:
            self._anonymize_session(session["session_id"])
        
        # 3. 从S3删除完整数据
        for session in sessions:
            self._delete_s3_data(session["session_id"])
        
        # 4. 记录删除审计日志
        self._log_deletion_audit(user_id, len(sessions))
    
    def _anonymize_session(self, session_id: str):
        """匿名化会话数据"""
        # 保留事件结构，但清除敏感内容
        self.clickhouse.execute(f"""
            UPDATE llm_events
            SET 
                data_summary = '[ANONYMIZED]',
                metadata = JSONExtractString(metadata, 'non_sensitive_fields')
            WHERE session_id = '{session_id}'
        """)
```

### 6.2 数据保留策略

| 数据类型 | 保留策略 | 存储位置 | 压缩策略 |
|---------|---------|---------|---------|
| 事件元数据 | 永久 | ClickHouse | 3个月后归档 |
| LLM输入/输出 | 90天 | S3 | 30天后转冷存储 |
| 检索结果 | 30天 | ClickHouse | 7天后聚合 |
| Token统计 | 永久 | ClickHouse | 按月聚合 |

---

## 七、性能优化

### 7.1 事件写入优化

```
写入性能优化策略：

  ┌─────────────────────────────────────────────┐
  │           异步写入Pipeline                   │
  │                                             │
  │  事件产生 → 内存队列 → 批量写入 → 持久化     │
  │       │        │         │         │        │
  │    <1ms    ~0ms    ~5ms    ~20ms           │
  │                                             │
  │  关键：事件产生与存储解耦，不影响主流程延迟    │
  └─────────────────────────────────────────────┘
```

```python
class AsyncEventWriter:
    """异步事件写入器"""
    
    def __init__(self, batch_size=100, flush_interval=5.0):
        self.buffer = asyncio.Queue()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._start_flush_task()
    
    async def write(self, event: LLMInteractionEvent):
        """非阻塞写入"""
        # 写入内存队列，立即返回
        await self.buffer.put(event)
    
    async def _flush_loop(self):
        """定期批量写入"""
        while True:
            batch = []
            
            # 收集批量事件
            while len(batch) < self.batch_size:
                try:
                    event = await asyncio.wait_for(
                        self.buffer.get(), 
                        timeout=self.flush_interval
                    )
                    batch.append(event)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                await self._batch_write(batch)
```

### 7.2 查询性能优化

| 查询场景 | 优化方案 | 延迟目标 |
|---------|---------|---------|
| 会话回放 | ClickHouse ORDER BY session_id, timestamp | < 100ms |
| 异常检测 | 物化视图预聚合 | < 50ms |
| 成本统计 | 预计算聚合表 | < 200ms |
| 跨会话分析 | 采样 + 近似计算 | < 500ms |

---

## 八、与现有可观测性体系集成

### 8.1 统一事件模型

```
可观测性三支柱 + LLM事件：

  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │    Logs      │  │   Metrics    │  │   Traces     │
  │  (传统日志)   │  │  (指标监控)   │  │  (链路追踪)   │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  │                 │
                  ▼                 ▼
         ┌────────────────────────────────────┐
         │        LLM事件溯源层               │
         │  • 事件驱动的审计日志               │
         │  • LLM调用链路追踪                 │
         │  • Token消耗指标                   │
         │  • 对话回放能力                    │
         └────────────────────────────────────┘
```

### 8.2 OpenTelemetry集成

```python
from opentelemetry import trace

class TracedEventStore:
    """带链路追踪的事件存储"""
    
    def append(self, event: LLMInteractionEvent):
        tracer = trace.get_tracer("llm-event-store")
        
        with tracer.start_as_current_span("event.append") as span:
            span.set_attribute("event.type", event.event_type)
            span.set_attribute("event.session_id", event.session_id)
            
            if isinstance(event, LLMCallEvent):
                span.set_attribute("llm.model", event.model)
                span.set_attribute("llm.tokens.prompt", event.prompt_tokens)
                span.set_attribute("llm.tokens.completion", event.completion_tokens)
                span.set_attribute("llm.latency_ms", event.latency_ms)
            
            # 写入事件存储
            self._store.write(event)
            
            span.add_event("event.stored", {
                "event_id": event.event_id,
                "storage_latency_ms": storage_latency
            })
```

---

## 九、实施路线图

### Phase 1：基础能力（2-3周）

- [ ] 定义事件Schema
- [ ] 实现异步事件写入
- [ ] ClickHouse建表和基础查询
- [ ] 基本的会话回放API

### Phase 2：分析能力（2-3周）

- [ ] 异常检测规则引擎
- [ ] Token成本分析
- [ ] A/B测试分析框架
- [ ] 可视化Dashboard

### Phase 3：高级能力（3-4周）

- [ ] 对比分析（正常 vs 异常）
- [ ] 自动根因定位
- [ ] 数据保留策略
- [ ] GDPR合规工具

---

## 总结

事件溯源不是银弹，但对于LLM应用来说，它解决了几个核心痛点：

| 痛点 | 事件溯源的解决方式 |
|------|-------------------|
| **不可解释性** | 完整记录每次LLM调用的输入/输出 |
| **调试困难** | 对话回放，精确还原问题现场 |
| **成本不透明** | Token级别精确计量 |
| **合规压力** | 完整审计链路，支持数据删除 |
| **优化无据** | 基于真实数据的Prompt/模型对比 |

```
LLM应用架构演进：

  Level 0: 黑盒调用
  → 调用LLM API，只看最终结果

  Level 1: 日志记录  
  → 记录输入/输出/Token消耗

  Level 2: 链路追踪
  → OpenTelemetry追踪LLM调用链

  Level 3: 事件溯源  ← 你在这里
  → 完整事件模型，支持回放/分析/审计

  Level 4: 智能运维
  → 基于事件流的自动异常检测和根因分析
```

对于正在构建生产级AI应用的团队来说，事件溯源架构值得在Day 1就考虑。前期的投入会在调试、优化、合规等多个维度持续回报。它不性感，但决定了你的AI应用能否经受住生产环境的考验。
