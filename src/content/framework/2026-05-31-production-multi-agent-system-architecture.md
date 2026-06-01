---
title: "生产级多Agent系统架构实战：从状态管理到错误恢复的完整工程方案"
description: "深度解析多Agent系统在生产环境中面临的工程挑战，覆盖状态管理、错误恢复、幂等性设计、人机协作与可观测性等核心问题的实战解决方案"
date: 2026-05-31
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["多Agent系统", "Agent架构", "状态管理", "错误恢复", "人机协作", "生产实践"]
draft: false
---

# 生产级多Agent系统架构实战：从状态管理到错误恢复的完整工程方案

## 从Demo到Production：多Agent系统的工程鸿沟

大多数多Agent系统的Demo看起来很美——几个Agent协作完成一个任务，流程清晰，结果完美。但当你真正把它部署到生产环境时，你会发现一个残酷的事实：**Demo中99%的代码是业务逻辑，Production中99%的代码是工程基础设施**。

一个真实的生产级多Agent系统需要解决的问题：

```
Demo级问题（业务逻辑）：         Production级问题（工程挑战）：
├── Agent如何分工               ├── Agent状态如何持久化
├── 工具如何调用               ├── 失败的Agent如何恢复
├── 结果如何汇总               ├── 并发冲突如何处理
└── Prompt如何优化             ├── 中间状态如何追踪
                               ├── 人机协作如何实现
                               ├── 幂等性如何保证
                               ├── 超时如何处理
                               └── 成本如何控制
```

本文将从**实际踩过的坑**出发，提供一套可直接落地的工程方案。

## 1. Agent状态管理：选择正确的状态机模式

### 状态管理模式对比

| 模式 | 适用场景 | 复杂度 | 可恢复性 | 并发安全 |
|------|----------|--------|----------|----------|
| 简单状态机 | 线性流程 | 低 | 中 | 差 |
| 图状态机（LangGraph） | 复杂分支 | 中 | 高 | 好 |
| 事件溯源 | 需要审计 | 高 | 极高 | 极好 |
| Saga模式 | 长事务 | 高 | 高 | 好 |
| CQRS + Event Sourcing | 读写分离 | 极高 | 极高 | 极好 |

### 生产级状态管理实现

```python
from enum import Enum
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    WAITING_HUMAN = "waiting_human"
    WAITING_PEER = "waiting_peer"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"

@dataclass
class AgentCheckpoint:
    """Agent状态检查点——用于故障恢复"""
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    state: AgentState = AgentState.IDLE
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)
    error_info: Optional[Dict] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 0

class ProductionAgent:
    """生产级Agent：带状态持久化和故障恢复"""
    
    def __init__(self, agent_id: str, state_store, config: dict):
        self.agent_id = agent_id
        self.state_store = state_store  # Redis/DynamoDB/PostgreSQL
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.checkpoint_interval = config.get("checkpoint_interval", 5)
    
    async def run(self, task: dict) -> dict:
        """带检查点和恢复机制的Agent执行"""
        # 1. 尝试恢复上次中断的状态
        checkpoint = await self._restore_checkpoint()
        
        try:
            # 2. 根据恢复点继续执行
            if checkpoint and checkpoint.state == AgentState.FAILED:
                # 从失败点重试
                result = await self._resume_from_failure(checkpoint, task)
            elif checkpoint and checkpoint.state != AgentState.IDLE:
                # 从中断点继续
                result = await self._resume_from_checkpoint(checkpoint, task)
            else:
                # 全新执行
                result = await self._execute_from_start(task)
            
            # 3. 标记完成
            await self._save_checkpoint(AgentState.COMPLETED, result)
            return result
            
        except Exception as e:
            # 4. 错误处理：保存错误状态，等待恢复
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "failed_step": self._get_current_step(),
                "retry_count": self._get_retry_count(),
            }
            await self._save_checkpoint(AgentState.FAILED, error_info=error_info)
            
            # 5. 判断是否可重试
            if self._is_retryable(e) and self._get_retry_count() < self.max_retries:
                return await self._retry_with_backoff(task)
            else:
                # 触发人工介入
                await self._escalate_to_human(error_info)
                raise
    
    async def _save_checkpoint(self, state: AgentState, 
                                context: dict = None, 
                                error_info: dict = None):
        """保存状态检查点"""
        checkpoint = AgentCheckpoint(
            agent_id=self.agent_id,
            state=state,
            context=context or {},
            history=self._get_history(),
            tool_results=self._get_tool_results(),
            error_info=error_info,
        )
        await self.state_store.save(checkpoint)
    
    async def _restore_checkpoint(self) -> Optional[AgentCheckpoint]:
        """从持久化存储恢复检查点"""
        return await self.state_store.get_latest(self.agent_id)
    
    def _is_retryable(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        retryable_errors = [
            "TimeoutError",
            "RateLimitError", 
            "ConnectionError",
            "TemporaryFailureError",
        ]
        return type(error).__name__ in retryable_errors
    
    async def _retry_with_backoff(self, task: dict) -> dict:
        """指数退避重试"""
        import asyncio
        retry_count = self._get_retry_count()
        delay = min(2 ** retry_count * 0.5, 30)  # 最大30秒
        
        await asyncio.sleep(delay)
        return await self.run(task)
```

### 状态持久化选型

```
┌──────────────────────────────────────────────────────────┐
│              状态存储选型决策树                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  需要强一致性？                                           │
│  ├── 是 → PostgreSQL + 序列化JSON                         │
│  │        适用：金融交易、审批流程                          │
│  │        优势：ACID事务、复杂查询                         │
│  │        劣势：延迟较高（~5ms）                           │
│  │                                                       │
│  └── 否 → 需要高吞吐？                                   │
│          ├── 是 → Redis Cluster                          │
│          │        适用：实时对话、推荐系统                  │
│          │        优势：低延迟（~1ms）、高吞吐             │
│          │        劣势：数据可能丢失                       │
│          │                                               │
│          └── 否 → DynamoDB / MongoDB                     │
│                   适用：大多数场景                         │
│                   优势：自动扩展、低成本                   │
│                   劣势：查询能力有限                       │
└──────────────────────────────────────────────────────────┘
```

## 2. 错误恢复与补偿：Saga模式在Agent系统中的应用

### 为什么需要Saga模式？

在多Agent系统中，一个任务可能涉及多个Agent的协作。如果中间某个Agent失败，我们需要**回滚之前Agent的操作**，而不是让系统处于不一致状态。

```
正常流程：
  Agent_A(搜索) → Agent_B(分析) → Agent_C(报告) → 完成

失败场景：
  Agent_A(搜索) → Agent_B(分析) → Agent_C(报告) ❌ 失败
                                              ↓
                                    需要回滚B的分析结果
                                    需要清理A的搜索缓存
```

### Saga实现：编排式 vs 协同式

| 模式 | 实现方式 | 优势 | 劣势 |
|------|----------|------|------|
| 编排式（Orchestration） | 中央协调器管理流程 | 流程清晰、易于调试 | 协调器是单点 |
| 协同式（Choreography） | 事件驱动、各Agent自治 | 去中心化、松耦合 | 流程分散、难以追踪 |

**生产建议**：大多数场景使用**编排式**，只有Agent数量超过10个且流程高度动态时才考虑协同式。

### 编排式Saga实现

```python
from typing import List, Callable, Any
import asyncio

class SagaOrchestrator:
    """Saga编排器：管理多Agent事务的执行与回滚"""
    
    def __init__(self):
        self.steps: List[dict] = []
    
    def add_step(self, name: str, 
                 action: Callable, 
                 compensation: Callable,
                 timeout: float = 60.0):
        """添加一个Saga步骤"""
        self.steps.append({
            "name": name,
            "action": action,
            "compensation": compensation,
            "timeout": timeout,
        })
        return self
    
    async def execute(self, context: dict) -> dict:
        """执行Saga，失败时自动补偿"""
        completed_steps = []
        results = {}
        
        for step in self.steps:
            try:
                # 执行当前步骤
                result = await asyncio.wait_for(
                    step["action"](context, results),
                    timeout=step["timeout"]
                )
                results[step["name"]] = result
                completed_steps.append(step)
                
                # 更新上下文，传递给下一步
                context["previous_results"] = results
                
            except Exception as e:
                # 失败：按逆序执行补偿操作
                print(f"Step '{step['name']}' failed: {e}")
                await self._compensate(completed_steps, results, context)
                raise SagaFailure(
                    failed_step=step["name"],
                    error=e,
                    completed_steps=[s["name"] for s in completed_steps]
                )
        
        return results
    
    async def _compensate(self, completed_steps: List[dict], 
                          results: dict, context: dict):
        """按逆序执行补偿操作"""
        for step in reversed(completed_steps):
            try:
                await step["compensation"](context, results)
                print(f"Compensation for '{step['name']}' succeeded")
            except Exception as comp_error:
                # 补偿失败也需要记录，但不阻塞其他补偿
                print(f"CRITICAL: Compensation for '{step['name']}' "
                      f"failed: {comp_error}")
                # 记录到死信队列，后续人工处理
                await self._send_to_dead_letter(step, comp_error)

class SagaFailure(Exception):
    def __init__(self, failed_step, error, completed_steps):
        self.failed_step = failed_step
        self.original_error = error
        self.completed_steps = completed_steps
        super().__init__(f"Saga failed at step '{failed_step}': {error}")


# ============ 实战案例：多Agent研究任务 ============

async def research_task_saga():
    """多Agent研究任务的Saga编排"""
    
    orchestrator = SagaOrchestrator()
    
    orchestrator.add_step(
        name="search",
        action=search_agent_action,
        compensation=search_agent_compensation,
        timeout=30.0
    ).add_step(
        name="analyze",
        action=analyze_agent_action,
        compensation=analyze_agent_compensation,
        timeout=60.0
    ).add_step(
        name="write_report",
        action=write_report_action,
        compensation=write_report_compensation,
        timeout=120.0
    ).add_step(
        name="review",
        action=review_agent_action,
        compensation=review_compensation,  # 通常为no-op
        timeout=30.0
    )
    
    try:
        results = await orchestrator.execute(context={
            "topic": "2026年AI Agent架构趋势",
            "user_id": "user_123",
            "session_id": "session_456",
        })
        return results
    except SagaFailure as e:
        # 通知用户并提供重试选项
        await notify_user_of_failure(e)
        raise

# 补偿操作实现示例
async def search_agent_compensation(context, results):
    """清理搜索结果缓存"""
    cache_key = f"search:{context['session_id']}"
    await redis.delete(cache_key)

async def analyze_agent_compensation(context, results):
    """删除已生成的分析报告"""
    if "analyze" in results:
        report_id = results["analyze"].get("report_id")
        if report_id:
            await db.delete_analysis_report(report_id)
```

## 3. 幂等性设计：确保Agent操作的安全重试

### 为什么Agent系统特别需要幂等性？

Agent的LLM调用是**非确定性**的——同样的输入可能产生不同的输出。这意味着：
- 网络超时后重试，Agent可能产生重复操作
- 工具调用可能被执行多次
- 消息可能被重复发送

### 幂等性实现方案

```python
import hashlib
import time
from typing import Optional

class IdempotencyManager:
    """幂等性管理器——确保操作只执行一次"""
    
    def __init__(self, store, ttl=3600):
        self.store = store  # Redis/DynamoDB
        self.ttl = ttl
    
    def generate_idempotency_key(self, agent_id: str, 
                                   action: str, 
                                   params: dict) -> str:
        """生成幂等性键"""
        # 基于操作内容生成确定性键
        content = f"{agent_id}:{action}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def execute_once(self, idempotency_key: str, 
                            operation, *args, **kwargs):
        """确保操作只执行一次"""
        # 1. 检查是否已执行过
        existing = await self.store.get(f"idempotent:{idempotency_key}")
        if existing:
            # 已执行过，直接返回之前的结果
            return {
                "result": existing["result"],
                "is_replay": True,
                "original_execution_time": existing["timestamp"],
            }
        
        # 2. 尝试获取执行锁（防并发）
        lock_acquired = await self.store.acquire_lock(
            f"lock:{idempotent_key}", 
            ttl=30
        )
        
        if not lock_acquired:
            # 另一个实例正在执行，等待并获取结果
            return await self._wait_for_result(idempotency_key)
        
        try:
            # 3. 执行操作
            result = await operation(*args, **kwargs)
            
            # 4. 持久化结果
            await self.store.set(
                f"idempotent:{idempotency_key}",
                {
                    "result": result,
                    "timestamp": time.time(),
                },
                ttl=self.ttl
            )
            
            return {
                "result": result,
                "is_replay": False,
            }
            
        finally:
            # 5. 释放锁
            await self.store.release_lock(f"lock:{idempotency_key}")


# ============ 工具调用的幂等性包装 ============

class IdempotentTool:
    """幂等性工具包装器"""
    
    def __init__(self, tool, idempotency_manager: IdempotencyManager):
        self.tool = tool
        self.manager = idempotency_manager
    
    async def execute(self, agent_id: str, params: dict) -> dict:
        """幂等性执行工具"""
        key = self.manager.generate_idempotency_key(
            agent_id=agent_id,
            action=self.tool.name,
            params=params
        )
        
        result = await self.manager.execute_once(
            key,
            self.tool.run,
            **params
        )
        
        return result
```

## 4. 人机协作：Human-in-the-Loop的生产级实现

### 人机协作模式

```
┌────────────────────────────────────────────────────┐
│            人机协作模式矩阵                          │
├──────────┬──────────┬──────────┬──────────┬────────┤
│          │  同步    │  异步    │  批量    │  审批   │
├──────────┼──────────┼──────────┼──────────┼────────┤
│ 模式     │ 实时     │ 消息队列 │ 定时     │ 工作流  │
│ 延迟要求 │ <5s      │ <1h     │ <24h     │ <7d    │
│ 典型场景 │ 实时编辑 │ 反馈    │ 批注    │ 合规审批│
│ 实现     │ WebSocket│ SQS/Kafka│ Cron    │ Temporal│
└──────────┴──────────┴──────────┴──────────┴────────┘
```

### Human-in-the-Loop实现

```python
import asyncio
from enum import Enum
from typing import Any, Optional

class HumanInteractionType(Enum):
    APPROVAL = "approval"           # 审批：通过/拒绝
    INPUT = "input"                 # 输入：需要人类提供信息
    SELECTION = "selection"         # 选择：从多个选项中选择
    FEEDBACK = "feedback"           # 反馈：对Agent输出的评价
    OVERRIDE = "override"           # 覆盖：人类直接修改Agent决策

class HumanInTheLoop:
    """人机协作管理器"""
    
    def __init__(self, notification_service, timeout=300):
        self.notification_service = notification_service
        self.timeout = timeout  # 默认5分钟超时
    
    async def request_human_input(self, 
                                   agent_id: str,
                                   interaction_type: HumanInteractionType,
                                   context: dict,
                                   options: Optional[list] = None) -> dict:
        """请求人类输入"""
        request_id = str(uuid.uuid4())
        
        # 1. 创建请求记录
        request = {
            "request_id": request_id,
            "agent_id": agent_id,
            "interaction_type": interaction_type.value,
            "context": context,
            "options": options,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "timeout_at": (
                datetime.utcnow() + timedelta(seconds=self.timeout)
            ).isoformat(),
        }
        await self.store.save_request(request)
        
        # 2. 发送通知
        await self.notification_service.send(
            user_id=context["user_id"],
            title=f"Agent需要你的{interaction_type.value}",
            body=self._format_request_message(request),
            request_id=request_id,
            actions=options,
        )
        
        # 3. 等待人类响应
        try:
            response = await asyncio.wait_for(
                self._wait_for_response(request_id),
                timeout=self.timeout
            )
            return response
            
        except asyncio.TimeoutError:
            # 超时处理
            await self._handle_timeout(request)
            return await self._get_default_response(interaction_type, context)
    
    async def _wait_for_response(self, request_id: str) -> dict:
        """等待人类响应"""
        while True:
            request = await self.store.get_request(request_id)
            if request["status"] == "completed":
                return request["response"]
            elif request["status"] == "timed_out":
                raise asyncio.TimeoutError()
            await asyncio.sleep(1)
    
    async def _handle_timeout(self, request: dict):
        """处理超时"""
        # 1. 标记为超时
        await self.store.update_request(
            request["request_id"], 
            status="timed_out"
        )
        
        # 2. 根据交互类型决定降级策略
        if request["interaction_type"] == HumanInteractionType.APPROVAL.value:
            # 审批超时：默认拒绝（安全优先）
            await self.notification_service.send(
                user_id=request["context"]["user_id"],
                title="审批超时，已自动拒绝",
                body=f"请求 {request['request_id']} 因超时已自动拒绝"
            )
        elif request["interaction_type"] == HumanInteractionType.FEEDBACK.value:
            # 反馈超时：记录但不阻塞流程
            pass
    
    def _format_request_message(self, request: dict) -> str:
        """格式化请求消息"""
        context = request["context"]
        if request["interaction_type"] == HumanInteractionType.APPROVAL.value:
            return (
                f"Agent请求审批以下操作：\n"
                f"操作：{context.get('action', '未知')}\n"
                f"详情：{context.get('details', '无')}\n"
                f"影响：{context.get('impact', '未知')}"
            )
        elif request["interaction_type"] == HumanInteractionType.SELECTION.value:
            options = request.get("options", [])
            options_text = "\n".join(
                f"  {i+1}. {opt}" for i, opt in enumerate(options)
            )
            return (
                f"请从以下选项中选择：\n{options_text}"
            )
        return f"Agent需要你的输入：{context.get('question', '未知')}"
```

## 5. 可观测性：从黑盒到全链路追踪

### Agent系统的可观测性架构

```
┌──────────────────────────────────────────────────────────┐
│               多Agent系统可观测性架构                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  数据采集层                                               │
│  ├── Agent日志（决策过程、工具调用）                       │
│  ├── LLM调用链（Token使用、延迟、成本）                   │
│  ├── 工具调用链（输入输出、耗时、成功率）                  │
│  └── 系统指标（CPU、内存、网络、队列深度）                 │
│                                                          │
│  数据处理层                                               │
│  ├── 分布式追踪（Trace → Span → Event）                  │
│  ├── 日志聚合（Structured Log → ELK/Loki）               │
│  ├── 指标聚合（Prometheus → Grafana）                     │
│  └── 异常检测（ML-based Anomaly Detection）               │
│                                                          │
│  可视化与告警层                                           │
│  ├── Agent执行流程图（实时）                               │
│  ├── 成本分析看板（Token消耗、API费用）                    │
│  ├── 质量监控看板（成功率、用户满意度）                    │
│  └── 智能告警（异常模式自动识别）                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 生产级追踪实现

```python
import time
import uuid
import json
from contextlib import asynccontextmanager
from typing import Any, Optional

class AgentTracer:
    """Agent分布式追踪器"""
    
    def __init__(self, export_service):
        self.export_service = export_service
    
    @asynccontextmanager
    async def trace_agent(self, agent_id: str, task_id: str):
        """追踪Agent执行"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:8]
        
        trace_context = {
            "trace_id": trace_id,
            "span_id": span_id,
            "agent_id": agent_id,
            "task_id": task_id,
            "start_time": time.time(),
        }
        
        # 记录Agent启动事件
        await self._emit_event("agent.start", trace_context)
        
        try:
            yield trace_context
            
            # 记录Agent完成事件
            trace_context["end_time"] = time.time()
            trace_context["duration_ms"] = (
                trace_context["end_time"] - trace_context["start_time"]
            ) * 1000
            await self._emit_event("agent.end", trace_context)
            
        except Exception as e:
            # 记录Agent失败事件
            trace_context["end_time"] = time.time()
            trace_context["error"] = {
                "type": type(e).__name__,
                "message": str(e),
            }
            await self._emit_event("agent.error", trace_context)
            raise
    
    @asynccontextmanager
    async def trace_llm_call(self, trace_context: dict,
                              model: str, 
                              messages: list):
        """追踪LLM调用"""
        span_id = str(uuid.uuid4())[:8]
        call_context = {
            **trace_context,
            "llm_span_id": span_id,
            "model": model,
            "input_tokens": self._count_tokens(messages),
            "start_time": time.time(),
        }
        
        await self._emit_event("llm.start", call_context)
        
        try:
            yield call_context
            
            call_context["end_time"] = time.time()
            call_context["duration_ms"] = (
                call_context["end_time"] - call_context["start_time"]
            ) * 1000
            await self._emit_event("llm.end", call_context)
            
        except Exception as e:
            call_context["error"] = {
                "type": type(e).__name__,
                "message": str(e),
            }
            await self._emit_event("llm.error", call_context)
            raise
    
    @asynccontextmanager
    async def trace_tool_call(self, trace_context: dict,
                               tool_name: str,
                               tool_input: Any):
        """追踪工具调用"""
        span_id = str(uuid.uuid4())[:8]
        tool_context = {
            **trace_context,
            "tool_span_id": span_id,
            "tool_name": tool_name,
            "tool_input": self._safe_serialize(tool_input),
            "start_time": time.time(),
        }
        
        await self._emit_event("tool.start", tool_context)
        
        try:
            yield tool_context
            
            tool_context["end_time"] = time.time()
            tool_context["duration_ms"] = (
                tool_context["end_time"] - tool_context["start_time"]
            ) * 1000
            await self._emit_event("tool.end", tool_context)
            
        except Exception as e:
            tool_context["error"] = {
                "type": type(e).__name__,
                "message": str(e),
            }
            await self._emit_event("tool.error", tool_context)
            raise
    
    async def _emit_event(self, event_type: str, context: dict):
        """发送追踪事件"""
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **context,
        }
        await self.export_service.export(event)
    
    def _count_tokens(self, messages: list) -> int:
        """估算Token数量"""
        return sum(len(m.get("content", "")) // 4 for m in messages)
    
    def _safe_serialize(self, data: Any) -> str:
        """安全序列化（避免敏感数据泄露）"""
        try:
            serialized = json.dumps(data, default=str)
            # 截断过长的数据
            if len(serialized) > 10000:
                serialized = serialized[:10000] + "...[truncated]"
            return serialized
        except Exception:
            return "[serialization_failed]"
```

## 6. 成本控制：Token经济学在Agent系统中的实践

### 成本失控的典型场景

一个看似简单的多Agent任务，成本可能远超预期：

```
用户请求："帮我分析这份100页的报告"

Agent执行链：
  协调器Agent：~500 tokens（理解任务）        → $0.0015
  搜索Agent：~2000 tokens（搜索相关章节）      → $0.006
  分析Agent：~8000 tokens（分析5个章节）       → $0.024
  报告Agent：~4000 tokens（生成报告）          → $0.012
  审查Agent：~1500 tokens（审查报告质量）       → $0.0045
  
  总计：~16000 tokens → $0.048
  
  如果发生重试（2次）：
  总计：~48000 tokens → $0.144
  
  如果1000个用户每天使用：
  月成本：$0.144 × 1000 × 30 = $4,320
```

### 成本控制策略

```python
class CostController:
    """Agent系统成本控制器"""
    
    def __init__(self, config: dict):
        self.daily_budget = config.get("daily_budget", 100.0)  # 美元
        self.per_task_budget = config.get("per_task_budget", 0.5)
        self.token_budgets = config.get("token_budgets", {
            "gpt-4o": 10000,
            "gpt-4o-mini": 50000,
            "claude-sonnet-4-20250514": 15000,
        })
        self.cost_per_token = {
            "gpt-4o": {"input": 0.0025/1000, "output": 0.01/1000},
            "gpt-4o-mini": {"input": 0.00015/1000, "output": 0.0006/1000},
            "claude-sonnet-4-20250514": {"input": 0.003/1000, "output": 0.015/1000},
        }
    
    def can_execute(self, task_id: str, model: str, 
                    estimated_tokens: int) -> dict:
        """检查是否可以执行"""
        # 1. 检查每日预算
        daily_usage = self._get_daily_usage()
        if daily_usage >= self.daily_budget:
            return {
                "allowed": False,
                "reason": "daily_budget_exceeded",
                "message": f"已达到每日预算上限 ${self.daily_budget}",
            }
        
        # 2. 检查单任务预算
        estimated_cost = self._estimate_cost(model, estimated_tokens)
        if estimated_cost > self.per_task_budget:
            # 尝试降级模型
            fallback_model = self._suggest_fallback_model(
                model, estimated_tokens
            )
            if fallback_model:
                return {
                    "allowed": True,
                    "model": fallback_model,
                    "reason": "model_downgraded",
                    "message": f"降级到 {fallback_model} 以控制成本",
                }
            return {
                "allowed": False,
                "reason": "per_task_budget_exceeded",
                "message": f"预估成本 ${estimated_cost:.4f} 超过单任务预算",
            }
        
        # 3. 检查Token预算
        token_budget = self.token_budgets.get(model, 10000)
        if estimated_tokens > token_budget:
            return {
                "allowed": True,
                "model": model,
                "reason": "token_budget_warning",
                "message": f"预估Token数 {estimated_tokens} 接近预算 {token_budget}",
                "suggestion": "考虑使用更小的上下文窗口",
            }
        
        return {"allowed": True, "model": model}
    
    def record_usage(self, task_id: str, model: str,
                     input_tokens: int, output_tokens: int):
        """记录Token使用量"""
        cost = (
            input_tokens * self.cost_per_token[model]["input"] +
            output_tokens * self.cost_per_token[model]["output"]
        )
        
        usage_record = {
            "task_id": task_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # 持久化到数据库
        self._save_usage(usage_record)
    
    def _suggest_fallback_model(self, current_model: str, 
                                 tokens_needed: int) -> Optional[str]:
        """建议降级模型"""
        fallback_chain = {
            "claude-sonnet-4-20250514": "gpt-4o-mini",
            "gpt-4o": "gpt-4o-mini",
        }
        return fallback_chain.get(current_model)
```

## 7. 综合架构：把一切组合在一起

### 生产级多Agent系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     生产级多Agent系统架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│  │ 用户界面  │────▶│ API网关   │────▶│ 认证鉴权  │                │
│  └──────────┘     └──────────┘     └──────────┘                │
│                                            │                    │
│                                            ▼                    │
│                               ┌──────────────────────┐          │
│                               │   Saga编排器         │          │
│                               │   (状态机 + 事务)    │          │
│                               └──────────┬───────────┘          │
│                                          │                      │
│                    ┌─────────────────────┼─────────────────┐    │
│                    │                     │                 │    │
│                    ▼                     ▼                 ▼    │
│           ┌──────────────┐    ┌──────────────┐   ┌──────────┐  │
│           │  Agent_A     │    │  Agent_B     │   │ Agent_C  │  │
│           │  (搜索)      │    │  (分析)      │   │ (报告)   │  │
│           └──────┬───────┘    └──────┬───────┘   └────┬─────┘  │
│                  │                    │                │        │
│                  ▼                    ▼                ▼        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    基础设施层                              │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐  │   │
│  │  │状态存储 │ │追踪系统 │ │成本控制 │ │人机协作 │ │消息队列│  │   │
│  │  │Redis   │ │Jaeger  │ │监控器  │ │通知服务 │ │Kafka  │  │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └───────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计原则总结

| 原则 | 实现要点 | 常见陷阱 |
|------|----------|----------|
| **状态可恢复** | Checkpoint + 持久化存储 | 内存状态丢失后无法恢复 |
| **操作幂等** | 幂等键 + 结果缓存 | 重复执行导致数据不一致 |
| **失败可补偿** | Saga模式 + 补偿操作 | 补偿操作本身失败 |
| **成本可控** | 预算限制 + 模型降级 | 单次请求成本失控 |
| **可观测** | 分布式追踪 + 结构化日志 | 缺少关键上下文信息 |
| **人机协同** | Human-in-the-Loop + 超时处理 | 人类响应超时阻塞流程 |

## 总结

生产级多Agent系统的核心不是"让Agent更聪明"，而是"让系统更可靠"。本文覆盖的6个工程维度——状态管理、错误恢复、幂等性、人机协作、可观测性、成本控制——是每个多Agent系统都必须面对的挑战。

**行动清单**：
1. 立即：为现有Agent系统添加Checkpoint机制
2. 本周：实现幂等性包装器，保护所有外部工具调用
3. 本月：搭建Agent追踪系统，至少覆盖LLM调用链
4. 下季度：建立成本监控看板，设置预算告警

**记住**：多Agent系统的可靠性 ≈ 各Agent可靠性的乘积。如果每个Agent的可靠性是99%，3个Agent协作的系统可靠性只有97%。工程投入不是可选项，而是必须项。
