---
title: "AI应用异步架构实战：大模型长任务的消息队列设计与工程化"
description: "深度解析AI应用中大模型异步任务的架构设计，涵盖消息队列选型、任务编排、超时重试、状态管理与背压控制的完整工程方案"
date: 2026-05-31
author: "RiceBall"
category: "engineering"
tags: ["异步架构", "消息队列", "LLM工程化", "任务编排", "系统设计"]
draft: false
---

## 引言：为什么AI应用需要特殊的异步架构？

传统Web应用的请求-响应模型通常在毫秒级完成，但大模型应用彻底打破了这一假设。

一个典型的LLM调用可能需要：
- **Chat API**: 2-30秒
- **长文档分析**: 30-120秒
- **视频生成**: 60-600秒
- **批量数据处理**: 数分钟到数小时
- **模型微调训练**: 数小时到数天

当你的用户在等待一个需要30秒才能返回的AI操作时，如果不做任何处理，会面临：
1. **HTTP超时** — Nginx默认60s，很多网关30s就断开
2. **连接池耗尽** — 长连接占用导致新请求无法接入
3. **用户体验崩塌** — 白屏等待，用户反复刷新
4. **系统雪崩** — 突发流量 + 长任务 = 线程/内存资源耗尽

**异步架构不是可选项，而是AI应用的必选项。**

本文将从零构建一套完整的AI应用异步任务处理体系，涵盖架构设计、技术选型、核心实现、生产化运维的全链路实践。

## 一、架构设计：从同步到异步的演进

### 1.1 架构演进路径

```
阶段一: 同步直调（不推荐）
┌────────┐    HTTP    ┌─────────┐    API    ┌─────┐
│ Client │ ────────→  │ Backend │ ────────→  │ LLM │
│        │ ←────────  │         │ ←────────  │     │
└────────┘  等待30s   └─────────┘           └─────┘
问题: 超时、资源浪费、用户体验差

阶段二: 基础异步（推荐起步）
┌────────┐   提交任务   ┌────────┐   异步调用  ┌─────┐
│ Client │ ──────────→ │  API   │ ──────────→ │ LLM │
│        │ ←──────────  │Gateway │            │     │
└────────┘  task_id    └───┬────┘            └─────┘
                           │
         ┌─────────────────┤ 轮询/回调
         ↓                 ↓
    ┌─────────┐      ┌──────────┐
    │ Redis   │      │ Worker   │
    │ (状态)  │      │ (处理)   │
    └─────────┘      └──────────┘

阶段三: 生产级异步（推荐生产环境）
┌────────┐          ┌─────────────┐          ┌──────┐
│ Client │ ←SSE/WS→ │   Gateway   │ ←gRPC──→ │ LLM  │
└────────┘          └──────┬──────┘          └──────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Kafka   │ │  Redis   │ │ PostgreSQL│
        │ (事件流) │ │ (缓存)   │ │ (持久化)  │
        └────┬─────┘ └──────────┘ └──────────┘
             ↓
    ┌────────────────────────┐
    │    Worker Pool         │
    │ ┌─────┐┌─────┐┌─────┐ │
    │ │ W-1 ││ W-2 ││ W-N │ │
    │ └─────┘└─────┘└─────┘ │
    └────────────────────────┘
```

### 1.2 核心设计原则

| 原则 | 说明 | 实践方法 |
|------|------|---------|
| **提交即返回** | 任务提交后立即返回task_id | 入队延迟 < 50ms |
| **状态可查询** | 任何时刻都能获取任务状态 | Redis状态机 + DB审计日志 |
| **结果可追溯** | 历史任务结果持久存储 | 对象存储 + 元数据DB |
| **失败可恢复** | 任务失败后支持重试 | 指数退避 + 死信队列 |
| **背压可控** | 高负载时优雅降级 | 限流器 + 优先级队列 |

## 二、技术选型：消息队列对比与决策

### 2.1 消息队列技术对比

| 特性 | Redis Streams | RabbitMQ | Kafka | NATS JetStream |
|------|--------------|----------|-------|----------------|
| **延迟** | <1ms | <1ms | 2-5ms | <1ms |
| **吞吐量** | 10万/s | 5万/s | 100万/s | 50万/s |
| **持久化** | AOF可配 | 消息级 | 分区级 | 流级别 |
| **消费模式** | 消费者组 | 队列/Topic | 消费者组 | 队列/流 |
| **延迟消息** | 需额外实现 | 原生支持 | 需额外实现 | 原生支持 |
| **运维复杂度** | 低 | 中 | 高 | 低 |
| **适用场景** | 中小规模 | 企业级 | 大数据流 | 云原生 |

### 2.2 我们的选型决策

**推荐方案：Redis Streams + PostgreSQL**

理由：
1. **Redis Streams** 已经满足大多数AI应用的异步需求：低延迟、消费者组、消息持久化
2. **PostgreSQL** 作为任务状态的持久化层，提供事务性保证和查询能力
3. 如果团队已有Redis，几乎零额外基础设施成本
4. 当规模超过10万QPS时，再考虑迁移到Kafka

**升级路径：**
```
Day 1: Redis Streams（单机/哨兵）
  ↓ 规模增长
Month 3: Redis Cluster + 读写分离
  ↓ 消息量爆发
Month 6: 引入Kafka处理事件流，Redis保留热数据
  ↓ 全面微服务化
Year 1: Kafka + 分布式任务调度（Temporal/Argo）
```

## 三、核心实现：完整异步任务系统

### 3.1 数据模型设计

```sql
-- 任务主表
CREATE TABLE llm_tasks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type     VARCHAR(50) NOT NULL,       -- 'chat', 'video_gen', 'batch_process'
    status        VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- pending -> queued -> running -> completed/failed/cancelled
    
    -- 输入
    input_params  JSONB NOT NULL,             -- 任务参数
    priority      INTEGER DEFAULT 5,          -- 1-10, 越高越优先
    
    -- 输出
    output_result JSONB,
    error_message TEXT,
    
    -- 调度信息
    worker_id     VARCHAR(100),
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ,
    
    -- 重试控制
    retry_count   INTEGER DEFAULT 0,
    max_retries   INTEGER DEFAULT 3,
    next_retry_at TIMESTAMPTZ,
    
    -- 超时控制
    timeout_ms    INTEGER DEFAULT 300000,     -- 默认5分钟
    
    -- 审计
    created_by    VARCHAR(100),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 任务状态索引（核心查询路径）
CREATE INDEX idx_tasks_status ON llm_tasks(status) WHERE status IN ('pending', 'queued', 'running');
CREATE INDEX idx_tasks_retry ON llm_tasks(next_retry_at) WHERE status = 'failed' AND retry_count < max_retries;
CREATE INDEX idx_tasks_type ON llm_tasks(task_type, created_at DESC);
```

### 3.2 Redis Stream任务队列

```python
import redis
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Any

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class LLMTask:
    task_type: str
    input_params: dict
    priority: int = 5
    timeout_ms: int = 300_000
    max_retries: int = 3
    task_id: str = None
    status: TaskStatus = TaskStatus.PENDING
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


class TaskQueue:
    """
    基于Redis Streams的AI任务队列
    核心特性：优先级支持、延迟重试、死信队列
    """
    
    STREAM_KEY = "llm:tasks:stream"
    DEAD_LETTER_KEY = "llm:tasks:dead_letter"
    GROUP_NAME = "llm_workers"
    CONSUMER_PREFIX = "worker-"
    
    # 优先级通道
    PRIORITY_CHANNELS = {
        10: "llm:tasks:p10",  # 最高优先级（实时对话）
        8:  "llm:tasks:p8",   # 高优先级（用户可见任务）
        5:  "llm:tasks:p5",   # 普通优先级（默认）
        3:  "llm:tasks:p3",   # 低优先级（批量处理）
        1:  "llm:tasks:p1",   # 最低优先级（后台分析）
    }
    
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        self._ensure_group()
    
    def _ensure_group(self):
        """确保消费者组存在"""
        for stream in [self.STREAM_KEY] + list(self.PRIORITY_CHANNELS.values()):
            try:
                self.r.xgroup_create(stream, self.GROUP_NAME, id="0", mkstream=True)
            except redis.exceptions.ResponseError:
                pass  # 组已存在
    
    def submit_task(self, task: LLMTask) -> str:
        """
        提交任务到队列
        根据优先级路由到不同的Stream
        """
        # 选择优先级通道
        channel = self._select_channel(task.priority)
        
        # 写入Redis Stream
        message = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "input_params": json.dumps(task.input_params),
            "priority": str(task.priority),
            "timeout_ms": str(task.timeout_ms),
            "max_retries": str(task.max_retries),
            "created_at": datetime.utcnow().isoformat(),
        }
        
        # 使用Redis事务保证原子性
        pipe = self.r.pipeline()
        pipe.xadd(channel, message, maxlen=100_000)
        pipe.execute()
        
        return task.task_id
    
    def consume(self, consumer_id: str, count: int = 1, 
                block_ms: int = 5000) -> Optional[LLMTask]:
        """
        从队列消费任务（支持优先级）
        Worker启动时调用此方法获取任务
        """
        # 按优先级从高到低尝试消费
        channels = list(self.PRIORITY_CHANNELS.values())
        
        for channel in channels:
            entries = self.r.xreadgroup(
                self.GROUP_NAME, consumer_id,
                {channel: ">"},
                count=count, block=block_ms
            )
            
            if entries:
                stream, messages = entries[0]
                msg_id, fields = messages[0]
                
                task = LLMTask(
                    task_id=fields[b"task_id"].decode(),
                    task_type=fields[b"task_type"].decode(),
                    input_params=json.loads(fields[b"input_params"]),
                    priority=int(fields[b"priority"]),
                    timeout_ms=int(fields[b"timeout_ms"]),
                    max_retries=int(fields[b"max_retries"]),
                    status=TaskStatus.RUNNING,
                )
                
                # 记录消费信息用于ACK
                task._stream_msg_id = msg_id
                task._stream_key = channel
                
                return task
        
        return None
    
    def ack_task(self, task: LLMTask):
        """确认任务完成"""
        self.r.xack(task._stream_key, self.GROUP_NAME, task._stream_msg_id)
    
    def requeue_task(self, task: LLMTask, error: str):
        """
        任务失败处理：
        - 未达重试上限 → 延迟重试
        - 已达重试上限 → 死信队列
        """
        self.r.xack(task._stream_key, self.GROUP_NAME, task._stream_msg_id)
        
        task.retry_count += 1
        
        if task.retry_count >= task.max_retries:
            # 进入死信队列
            self.r.xadd(self.DEAD_LETTER_KEY, {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "error": error,
                "retry_count": str(task.retry_count),
                "created_at": datetime.utcnow().isoformat(),
            })
        else:
            # 指数退避延迟重试
            delay_seconds = min(300, 2 ** task.retry_count * 5)
            retry_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            
            self.r.zadd("llm:tasks:retry_queue", {
                json.dumps({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "input_params": task.input_params,
                    "priority": task.priority,
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries,
                    "timeout_ms": task.timeout_ms,
                }): retry_at.timestamp()
            })
    
    def _select_channel(self, priority: int) -> str:
        """根据优先级选择对应的Stream通道"""
        for p, channel in sorted(self.PRIORITY_CHANNELS.items(), reverse=True):
            if priority >= p:
                return channel
        return self.PRIORITY_CHANNELS[1]


class RetryScheduler:
    """
    延迟重试调度器
    定期检查重试队列，将到期的任务重新投入主队列
    建议每10秒运行一次
    """
    
    def __init__(self, redis_client: redis.Redis, queue: TaskQueue):
        self.r = redis_client
        self.queue = queue
    
    def tick(self):
        """检查并处理到期的重试任务"""
        now = datetime.utcnow().timestamp()
        
        # 取出到期的重试任务
        ready_tasks = self.r.zrangebyscore(
            "llm:tasks:retry_queue", 0, now, withscores=True
        )
        
        for task_data, score in ready_tasks:
            task_info = json.loads(task_data)
            
            # 重新投入主队列
            task = LLMTask(
                task_id=task_info["task_id"],
                task_type=task_info["task_type"],
                input_params=task_info["input_params"],
                priority=task_info["priority"],
                timeout_ms=task_info["timeout_ms"],
                max_retries=task_info["max_retries"],
            )
            task.retry_count = task_info["retry_count"]
            
            self.queue.submit_task(task)
            
            # 从重试队列中移除
            self.r.zrem("llm:tasks:retry_queue", task_data)
```

### 3.3 Worker处理器

```python
import asyncio
import signal
import time
from typing import Callable, Dict

class LLMWorker:
    """
    LLM任务Worker
    支持：
    - 多任务类型路由
    - 超时监控
    - 优雅关闭
    - 并发控制
    """
    
    def __init__(self, worker_id: str, queue: TaskQueue, 
                 max_concurrent: int = 5):
        self.worker_id = worker_id
        self.queue = queue
        self.max_concurrent = max_concurrent
        self.running = True
        self.handlers: Dict[str, Callable] = {}
        self.active_tasks: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        
        # 注册信号处理
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
    
    def register_handler(self, task_type: str, handler: Callable):
        """注册任务类型处理器"""
        self.handlers[task_type] = handler
    
    async def run(self):
        """主循环：持续消费和处理任务"""
        print(f"Worker {self.worker_id} 启动，最大并发: {self.max_concurrent}")
        
        while self.running:
            task = self.queue.consume(self.worker_id, count=1, block_ms=1000)
            
            if task and task.task_id:
                asyncio.create_task(self._process_with_timeout(task))
    
    async def _process_with_timeout(self, task: LLMTask):
        """带超时控制的任务处理"""
        async with self.active_tasks:
            try:
                handler = self.handlers.get(task.task_type)
                if not handler:
                    raise ValueError(f"未知任务类型: {task.task_type}")
                
                # 创建超时任务
                timeout_seconds = task.timeout_ms / 1000
                result = await asyncio.wait_for(
                    handler(task.input_params),
                    timeout=timeout_seconds
                )
                
                # 标记完成
                self.queue.ack_task(task)
                await self._report_result(task, result)
                
            except asyncio.TimeoutError:
                error = f"任务超时 ({task.timeout_ms}ms)"
                self.queue.requeue_task(task, error)
                await self._report_error(task, error)
                
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                self.queue.requeue_task(task, error)
                await self._report_error(task, error)
    
    async def _report_result(self, task: LLMTask, result: Any):
        """将结果推送给客户端（通过WebSocket/SSE/数据库轮询）"""
        # 实际项目中，这里会：
        # 1. 写入PostgreSQL结果表
        # 2. 通过Redis PubSub通知订阅者
        # 3. 如果有WebSocket连接，直接推送
        print(f"任务完成: {task.task_id} ({task.task_type})")
    
    async def _report_error(self, task: LLMTask, error: str):
        """报告任务错误"""
        print(f"任务失败: {task.task_id} - {error}")
    
    def _shutdown(self, signum, frame):
        """优雅关闭"""
        print(f"收到关闭信号，等待活跃任务完成...")
        self.running = False
```

### 3.4 客户端：实时状态查询与SSE推送

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/api/v1/tasks")
async def submit_task(request: dict):
    """提交任务 - 立即返回task_id"""
    task = LLMTask(
        task_type=request["type"],
        input_params=request["params"],
        priority=request.get("priority", 5),
        timeout_ms=request.get("timeout_ms", 300_000),
    )
    
    task_id = task_queue.submit_task(task)
    
    # 同时在DB中记录任务（略）
    # db.tasks.create(id=task_id, ...)
    
    return {
        "task_id": task_id,
        "status": "pending",
        "poll_url": f"/api/v1/tasks/{task_id}",
        "stream_url": f"/api/v1/tasks/{task_id}/stream",
    }

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """轮询查询任务状态"""
    # 从Redis/DB获取状态（略）
    task_info = get_task_from_db(task_id)
    
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info.get("progress"),
        "result": task_info.get("output_result"),
        "error": task_info.get("error_message"),
        "created_at": task_info["created_at"],
        "elapsed_ms": (datetime.utcnow() - task_info["created_at"]).total_seconds() * 1000,
    }

@app.get("/api/v1/tasks/{task_id}/stream")
async def stream_task_updates(task_id: str):
    """
    SSE实时推送任务状态
    客户端可以实时收到进度更新，无需轮询
    """
    async def event_generator():
        pubsub = redis_client.pubsub()
        channel = f"task:{task_id}:updates"
        await pubsub.subscribe(channel)
        
        try:
            # 首先发送当前状态
            current = get_task_from_db(task_id)
            yield f"data: {json.dumps(current)}\n\n"
            
            # 如果已完成，直接结束
            if current["status"] in ("completed", "failed", "cancelled"):
                return
            
            # 监听后续更新
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    if data.get("status") in ("completed", "failed"):
                        break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        }
    )
```

## 四、生产化：背压控制与优雅降级

### 4.1 背压控制策略

当LLM服务出现故障或延迟飙升时，如果没有背压机制，请求会持续堆积直到系统崩溃。

```python
class BackpressureController:
    """
    背压控制器
    监控队列深度和服务健康度，动态调整流量
    """
    
    def __init__(self, redis_client, 
                 queue_threshold=1000,    # 队列深度阈值
                 latency_threshold=30,     # 延迟阈值(秒)
                 error_rate_threshold=0.1): # 错误率阈值
        self.r = redis_client
        self.queue_threshold = queue_threshold
        self.latency_threshold = latency_threshold
        self.error_rate_threshold = error_rate_threshold
    
    def get_pressure_level(self) -> dict:
        """
        计算当前压力等级
        返回: {level: 0-3, reason: str, action: str}
        """
        queue_depth = self._get_queue_depth()
        avg_latency = self._get_avg_latency()
        error_rate = self._get_error_rate()
        
        if queue_depth > self.queue_threshold * 2:
            return {
                "level": 3,  # 严重
                "reason": f"队列积压严重: {queue_depth}",
                "action": "reject_non_critical",
                "backoff_seconds": 30,
            }
        
        if error_rate > self.error_rate_threshold * 2:
            return {
                "level": 3,
                "reason": f"错误率过高: {error_rate:.1%}",
                "action": "circuit_break",
                "backoff_seconds": 60,
            }
        
        if queue_depth > self.queue_threshold or avg_latency > self.latency_threshold:
            return {
                "level": 2,  # 中等
                "reason": f"队列深度: {queue_depth}, 平均延迟: {avg_latency:.1f}s",
                "action": "slow_down",
                "backoff_seconds": 5,
            }
        
        if error_rate > self.error_rate_threshold:
            return {
                "level": 1,  # 轻微
                "reason": f"错误率偏高: {error_rate:.1%}",
                "action": "monitor",
                "backoff_seconds": 0,
            }
        
        return {
            "level": 0,  # 正常
            "reason": "系统健康",
            "action": "proceed",
            "backoff_seconds": 0,
        }
    
    def should_accept_task(self, priority: int) -> tuple[bool, str]:
        """
        根据压力等级和任务优先级决定是否接受任务
        返回: (是否接受, 拒绝原因)
        """
        pressure = self.get_pressure_level()
        
        if pressure["level"] == 0:
            return True, ""
        
        if pressure["level"] == 1:
            return True, ""
        
        if pressure["level"] == 2:
            if priority < 5:
                return False, f"系统压力中等，低优先级任务暂不接受。{pressure['reason']}"
            return True, ""
        
        # level 3: 只接受最高优先级
        if priority >= 8:
            return True, ""
        return False, f"系统压力严重，仅接受高优先级任务。{pressure['reason']}"
    
    def _get_queue_depth(self) -> int:
        total = 0
        for channel in TaskQueue.PRIORITY_CHANNELS.values():
            total += self.r.xlen(channel)
        return total
    
    def _get_avg_latency(self) -> float:
        """从监控数据中获取近5分钟平均延迟"""
        # 实际中从Prometheus/StatsD获取
        return float(self.r.get("metrics:avg_latency") or 0)
    
    def _get_error_rate(self) -> float:
        return float(self.r.get("metrics:error_rate") or 0)
```

### 4.2 优雅降级策略

```python
class DegradationManager:
    """
    降级策略管理器
    当LLM服务异常时，提供降级方案保障核心链路
    """
    
    DEGRADATION_LEVELS = {
        "full":       {"latency": "normal", "quality": "full",   "features": "all"},
        "lite":       {"latency": "normal", "quality": "reduced","features": "core"},
        "fallback":   {"latency": "fast",   "quality": "basic",  "features": "minimal"},
        "cached":     {"latency": "fast",   "quality": "cached", "features": "read-only"},
        "maintenance":{"latency": "fast",   "quality": "none",   "features": "none"},
    }
    
    def __init__(self):
        self.current_level = "full"
        self.fallback_models = {
            "gpt-4o": "gpt-4o-mini",
            "claude-opus": "claude-haiku",
            "deepseek-v3": "deepseek-v2.5",
        }
        self.response_cache = {}  # 简化示例，生产用Redis
    
    def get_handler(self, original_handler: Callable, task_type: str):
        """
        包装原始处理器，根据当前降级等级自动切换
        """
        async def wrapped_handler(params):
            level = self.current_level
            
            if level == "full":
                return await original_handler(params)
            
            if level == "lite":
                # 使用轻量模型
                lite_params = self._downgrade_params(params)
                return await original_handler(lite_params)
            
            if level == "fallback":
                # 返回缓存的相似结果
                cache_key = self._make_cache_key(params)
                cached = self.response_cache.get(cache_key)
                if cached:
                    return cached
                # 无缓存则用最小模型生成基础回复
                return await self._basic_response(params)
            
            if level == "cached":
                cache_key = self._make_cache_key(params)
                return self.response_cache.get(cache_key, {
                    "message": "系统维护中，暂时无法处理新请求"
                })
            
            if level == "maintenance":
                return {"message": "系统维护中，请稍后再试"}
        
        return wrapped_handler
    
    def _downgrade_params(self, params: dict) -> dict:
        """降低参数质量以换取速度"""
        return {
            **params,
            "max_tokens": min(params.get("max_tokens", 4096), 2048),
            "temperature": 0.3,  # 降低随机性
        }
    
    async def _basic_response(self, params: dict) -> dict:
        """基础兜底回复"""
        return {
            "message": "当前系统负载较高，已为您返回简化回复。如需完整回复请稍后重试。",
            "is_degraded": True,
        }
    
    def _make_cache_key(self, params: dict) -> str:
        return hash(json.dumps(params, sort_keys=True))
```

## 五、监控与可观测性

### 5.1 核心监控指标

一个完善的异步任务系统需要监控以下指标：

| 指标类别 | 指标名 | 告警阈值 | 说明 |
|---------|--------|---------|------|
| **队列健康** | queue_depth | > 1000 | 积压任务数 |
| | queue_growth_rate | > 50/s | 队列增长速率 |
| | consumer_lag | > 100 | 消费者滞后量 |
| **任务指标** | task_success_rate | < 95% | 任务成功率 |
| | task_avg_duration | > 30s | 平均处理时长 |
| | task_p99_duration | > 120s | P99处理时长 |
| | task_timeout_rate | > 5% | 超时率 |
| **重试指标** | retry_rate | > 10% | 重试率 |
| | dead_letter_rate | > 1% | 死信率 |
| **资源指标** | worker_utilization | > 90% | Worker利用率 |
| | active_tasks | > max_concurrent | 活跃任务数 |

### 5.2 Prometheus指标暴露

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义Prometheus指标
task_submitted = Counter(
    'llm_tasks_submitted_total', 
    'Total tasks submitted',
    ['task_type']
)

task_completed = Counter(
    'llm_tasks_completed_total',
    'Total tasks completed',
    ['task_type', 'status']  # status: success/failed/timeout
)

task_duration = Histogram(
    'llm_tasks_duration_seconds',
    'Task processing duration',
    ['task_type'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)

queue_depth = Gauge(
    'llm_queue_depth',
    'Current queue depth',
    ['priority']
)

active_workers = Gauge(
    'llm_active_workers',
    'Number of active workers'
)

# 在Worker中使用
class InstrumentedWorker(LLMWorker):
    async def _process_with_timeout(self, task: LLMTask):
        task.labels(task_type=task.task_type).inc()
        
        start = time.time()
        try:
            await super()._process_with_timeout(task)
            task_completed.labels(task_type=task.task_type, status="success").inc()
        except asyncio.TimeoutError:
            task_completed.labels(task_type=task.task_type, status="timeout").inc()
        except Exception:
            task_completed.labels(task_type=task.task_type, status="failed").inc()
        finally:
            task_duration.labels(task_type=task.task_type).observe(time.time() - start)
```

## 六、实战案例：大规模批量推理任务编排

### 6.1 业务场景

某电商平台需要每天对10万条商品描述进行AI改写和翻译，涉及：
- 调用LLM进行文案优化（每条2-5秒）
- 调用翻译模型进行多语言翻译（每条1-3秒）
- 图片质量检测与标注（每条3-8秒）
- 结果汇总与审核

### 6.2 架构方案

```
┌──────────────────────────────────────────────────┐
│                 批量任务编排器                     │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐ │
│  │ 分片调度 │ →  │ 任务分发  │ →  │ 进度聚合     │ │
│  └─────────┘    └──────────┘    └──────────────┘ │
└──────────────────────────────────────────────────┘
          │
          ├─── 高优先级: AI改写 (实时)     → Channel P10
          ├─── 中优先级: 翻译 (准实时)     → Channel P5
          ├─── 低优先级: 图片标注 (后台)   → Channel P3
          └─── 最低优先级: 汇总统计 (夜间) → Channel P1
```

### 6.3 关键代码

```python
class BatchPipeline:
    """
    批量任务流水线
    支持分片、并行、依赖编排
    """
    
    def __init__(self, queue: TaskQueue):
        self.queue = queue
    
    async def run_daily_rewrite(self, product_ids: list[str]):
        """每日批量改写流水线"""
        batch_size = 100
        total_batches = (len(product_ids) + batch_size - 1) // batch_size
        
        task_ids = []
        
        for i in range(0, len(product_ids), batch_size):
            batch = product_ids[i:i + batch_size]
            
            # 提交改写任务（高优先级）
            for pid in batch:
                task_id = self.queue.submit_task(LLMTask(
                    task_type="product_rewrite",
                    input_params={"product_id": pid},
                    priority=8,
                    timeout_ms=60_000,
                ))
                task_ids.append(task_id)
        
        print(f"已提交 {len(task_ids)} 个改写任务，分 {total_batches} 批")
        
        # 等待所有改写完成后，提交翻译任务
        await self._wait_for_completion(task_ids)
        
        # 第二阶段：翻译（中优先级）
        translate_ids = []
        for pid in product_ids:
            for lang in ["en", "ja", "ko"]:
                task_id = self.queue.submit_task(LLMTask(
                    task_type="product_translate",
                    input_params={"product_id": pid, "target_lang": lang},
                    priority=5,
                    timeout_ms=30_000,
                ))
                translate_ids.append(task_id)
        
        print(f"已提交 {len(translate_ids)} 个翻译任务")
        await self._wait_for_completion(translate_ids)
        
        # 第三阶段：汇总
        print("所有任务完成，开始汇总...")
    
    async def _wait_for_completion(self, task_ids: list[str], 
                                    check_interval: float = 5.0):
        """等待一批任务全部完成"""
        pending = set(task_ids)
        
        while pending:
            for tid in list(pending):
                status = get_task_status(tid)
                if status["status"] in ("completed", "failed"):
                    pending.remove(tid)
            
            if pending:
                print(f"等待中... 剩余 {len(pending)} 个任务")
                await asyncio.sleep(check_interval)
```

## 七、总结与最佳实践

### 7.1 核心原则回顾

1. **提交即返回**：LLM调用必须异步化，客户端不等待
2. **状态可查询**：每个任务都有唯一ID和可查询的状态
3. **优先级路由**：实时任务和批量任务必须分开处理
4. **背压控制**：系统过载时主动拒绝，避免雪崩
5. **优雅降级**：LLM服务故障时有兜底方案

### 7.2 技术选型总结

| 规模 | 消息队列 | 状态存储 | 适用场景 |
|------|---------|---------|---------|
| MVP/原型 | Redis Streams | Redis | 验证想法，快速上线 |
| 中等规模 | Redis Cluster | PostgreSQL | 日均10万任务以下 |
| 大规模 | Kafka + Redis | PostgreSQL + ClickHouse | 日均100万+任务 |
| 超大规模 | Kafka + Temporal | 分布式存储 | 企业级多团队共用 |

### 7.3 避坑指南

| 常见问题 | 原因 | 解决方案 |
|---------|------|---------|
| 任务重复执行 | 消费者ACK丢失 | 幂等设计 + 任务去重表 |
| 死信队列膨胀 | 重试策略不合理 | 指数退避 + 人工干预告警 |
| Worker内存溢出 | 并发无限制 | Semaphore + 内存监控 |
| 超时不准确 | 网络延迟计入 | 超时预算管理（扣除网络RTT） |
| 优先级饿死 | 低优先级永远排不上 | 优先级衰减 + 保底调度 |

> **最后的建议**：异步架构看似增加了系统复杂度，但对于AI应用来说是绝对必要的投入。从最简单的Redis Stream + 轮询开始，逐步演进到生产级方案。记住：**先让它跑起来，再让它跑得好**。
