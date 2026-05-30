---
title: "流式AI应用架构设计：从SSE到WebSocket的实时推理系统"
description: "深入解析AI应用中流式响应的架构设计，对比SSE/WebSocket/gRPC Streaming三种方案，给出生产级流式推理系统的完整技术栈与最佳实践"
date: 2026-05-30
author: RiceBall-15
category: architecture
subCategory: cloud-native
tags: ["流式架构", "SSE", "WebSocket", "gRPC", "LLM推理", "实时系统", "云原生", "系统架构"]
draft: false
---

## 一、引言：为什么AI应用必须"流式化"

大语言模型（LLM）的推理延迟通常在1-30秒之间——生成500个token可能需要5-15秒。如果采用传统的"请求-等待-响应"模式，用户将面对漫长的白屏等待，体验极差。

流式输出（Streaming）通过**逐token推送**的方式，让用户在第一个token生成时就能看到内容，将感知延迟从"总推理时间"降低到"首token延迟（TTFT）"，通常在200-500ms以内。这一看似简单的技术改变，对底层架构设计产生了深远影响：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    传统请求 vs 流式请求                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  传统模式 (Request-Response):                                         │
│  Client ──── Request ────→ Server                                    │
│  Client ←── Response ───── Server  (等待 5-15 秒)                    │
│                                                                       │
│  用户体验: [白屏等待...] → [完整内容一次性出现]                          │
│                                                                       │
│  ──────────────────────────────────────────────────────────────────   │
│                                                                       │
│  流式模式 (Streaming):                                                │
│  Client ──── Request ────→ Server                                    │
│  Client ←── "你" ──────── Server  (200ms)                           │
│  Client ←── "好" ──────── Server  (80ms)                            │
│  Client ←── "，" ──────── Server  (60ms)                            │
│  Client ←── "我" ──────── Server  (90ms)                            │
│  Client ←── "是" ──────── Server  (70ms)                            │
│  ...                                                                 │
│                                                                       │
│  用户体验: [几乎立即看到文字开始出现] → [流畅的打字机效果]                 │
│                                                                       │
│  关键指标:                                                            │
│  TTFT (Time To First Token): 200-500ms                               │
│  TPOT (Time Per Output Token): 30-80ms                               │
│  总延迟: 不变，但感知延迟大幅降低                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 二、三大流式传输协议深度对比

### 2.1 协议全景

```
┌──────────────────────────────────────────────────────────────────────┐
│                    流式传输协议对比                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  SSE (Server-Sent Events)                                            │
│  ┌──────────────────────────────────────────┐                       │
│  │  HTTP/1.1+ 单向流                          │                       │
│  │  Server → Client (仅服务端推送)            │                       │
│  │  基于 HTTP，天然兼容代理/CDN/负载均衡       │                       │
│  │  文本协议，UTF-8 编码                       │                       │
│  │  自动重连机制 (Last-Event-ID)              │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                       │
│  WebSocket                                                           │
│  ┌──────────────────────────────────────────┐                       │
│  │  全双工持久连接                             │                       │
│  │  Client ↔ Server (双向通信)               │                       │
│  │  独立协议，需要升级握手                     │                       │
│  │  支持文本和二进制帧                         │                       │
│  │  适合需要客户端主动发送的场景               │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                       │
│  gRPC Streaming                                                     │
│  ┌──────────────────────────────────────────┐                       │
│  │  HTTP/2 + Protocol Buffers               │                       │
│  │  支持四种模式: Unary/Server/Client/Bi     │                       │
│  │  强类型，二进制序列化                       │                       │
│  │  高性能，适合微服务间通信                   │                       │
│  │  浏览器端需要 gRPC-Web 代理                │                       │
│  └──────────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 详细对比

| 维度 | SSE | WebSocket | gRPC Streaming |
|------|-----|-----------|----------------|
| **传输层** | HTTP/1.1+ | 独立协议(基于TCP) | HTTP/2 |
| **通信方向** | 单向(Server→Client) | 全双工 | Server/Client/Bi |
| **数据格式** | 文本(text/event-stream) | 文本+二进制 | Protobuf(二进制) |
| **浏览器支持** | 原生支持(EventSource) | 原生支持(WebSocket) | 需要gRPC-Web代理 |
| **代理/CDN兼容** | ✅ 天然兼容 | ⚠️ 需要特殊配置 | ⚠️ 需要HTTP/2支持 |
| **自动重连** | ✅ 内置 | ❌ 需手动实现 | ❌ 需手动实现 |
| **连接开销** | 低(HTTP长连接) | 中(握手升级) | 高(HTTP/2+TLS) |
| **协议复杂度** | ⭐ 极简 | ⭐⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| **适用场景** | LLM流式输出 | 实时交互/游戏 | 微服务间流式通信 |
| **OpenAI/Anthropic选择** | ✅ SSE | ❌ | ❌ |

### 2.3 为什么LLM服务商选择SSE

OpenAI、Anthropic、Google等主流LLM服务商无一例外地选择了SSE作为流式API的传输协议，原因如下：

1. **浏览器兼容性**：`EventSource` API原生支持，无需额外库
2. **代理友好**：标准HTTP协议，CDN/LB/Nginx天然支持
3. **实现简单**：服务端只需写入特殊格式的文本流
4. **自动重连**：浏览器内置重连机制，断线恢复零代码
5. **安全性**：基于HTTP，天然支持CORS、Cookie、Token认证

```
SSE 数据格式示例:

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"你"}}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"好"}}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"！"}}]}

data: [DONE]
```

## 三、生产级流式架构设计

### 3.1 完整架构全景

```
┌──────────────────────────────────────────────────────────────────────┐
│                生产级流式AI推理系统架构                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐                                                        │
│  │  Web/App  │  SSE 连接                                              │
│  │  Client   │──────────────┐                                        │
│  └──────────┘               │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────┐                       │
│  │              API Gateway                  │                       │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────┐ │                       │
│  │  │ 认证鉴权 │ │ 限流熔断  │ │ 路由分发  │ │                       │
│  │  └─────────┘ └──────────┘ └───────────┘ │                       │
│  └──────────────────────────────────────────┘                       │
│                              │                                        │
│               ┌──────────────┼──────────────┐                       │
│               ▼              ▼              ▼                        │
│  ┌────────────────┐ ┌──────────────┐ ┌──────────────┐              │
│  │  Stream Router  │ │  Stream Pool │ │  Stream Cache │              │
│  │  (流路由)       │ │  (连接池)     │ │  (流缓存)     │              │
│  └────────────────┘ └──────────────┘ └──────────────┘              │
│               │              │              │                        │
│               ▼              ▼              ▼                        │
│  ┌────────────────────────────────────────────────┐                 │
│  │              Inference Engine                   │                 │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │                 │
│  │  │ vLLM     │  │ SGLang   │  │ TensorRT │    │                 │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │    │                 │
│  │  └──────────┘  └──────────┘  └──────────┘    │                 │
│  └────────────────────────────────────────────────┘                 │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────┐                       │
│  │           Observability Layer             │                       │
│  │  ┌────────┐ ┌──────────┐ ┌────────────┐ │                       │
│  │  │ Metrics │ │ Tracing  │ │ Log Stream │ │                       │
│  │  │Prometheus│ │ OpenTelemetry│ │ Loki  │ │                       │
│  │  └────────┘ └──────────┘ └────────────┘ │                       │
│  └──────────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 流式连接管理

流式连接与传统HTTP请求的核心区别在于**连接生命周期**：一个流式连接可能持续5-30秒，远超传统API的毫秒级响应。这带来了几个架构挑战：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    流式连接生命周期管理                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  传统HTTP:                                                            │
│  [请求] → [处理] → [响应] → [连接关闭]   (总时间: 50ms)              │
│  并发: 10000 QPS / 连接                                              │
│                                                                       │
│  流式HTTP:                                                            │
│  [请求] → [首token] ──── 持续推送 ──── [DONE] → [连接关闭]           │
│            ↑ 200ms         ↑ 5-15秒                  (总时间: 5-15秒)│
│                                                                       │
│  并发瓶颈:                                                            │
│  如果每个连接占用 1 个线程/协程:                                       │
│  1000 并发 × 10秒 = 10000 连接-秒 (传统只需 100 连接-秒)             │
│                                                                       │
│  解决方案:                                                            │
│  1. 异步IO (asyncio/goroutine) - 每连接开销极低                       │
│  2. 连接池复用 - 限制最大并发流式连接数                                │
│  3. 背压控制 - 下游慢时暂停推送                                       │
│  4. 超时机制 - 单连接最大持续时间限制                                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 核心组件实现

#### 3.3.1 流式代理层

流式代理是架构中最关键的组件，负责连接管理、负载均衡和故障转移：

```python
# 流式代理核心逻辑 (Python asyncio)
import asyncio
from typing import AsyncIterator

class StreamProxy:
    def __init__(self, workers: list["InferenceWorker"]):
        self.workers = workers
        self._semaphore = asyncio.Semaphore(max_concurrent_streams)
        self._circuit_breaker = CircuitBreaker(fail_threshold=5, recovery=30)
    
    async def stream_completion(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """流式补全代理"""
        
        # 1. 限流检查
        async with self._semaphore:
            # 2. 熔断检查
            if self._circuit_breaker.is_open:
                yield StreamChunk(error="Service temporarily unavailable")
                return
            
            # 3. 选择 Worker (负载均衡)
            worker = self._select_worker(request)
            
            # 4. 流式转发 + 超时保护
            try:
                async for chunk in asyncio.wait_for(
                    worker.stream(request),
                    timeout=STREAM_TIMEOUT  # 30秒
                ):
                    yield chunk
                self._circuit_breaker.record_success()
                
            except asyncio.TimeoutError:
                yield StreamChunk(error="Stream timeout")
                self._circuit_breaker.record_failure()
                
            except WorkerError:
                # 5. 故障转移到备用 Worker
                fallback = self._select_worker(request, exclude=[worker])
                async for chunk in fallback.stream(request):
                    yield chunk
    
    def _select_worker(
        self, request, exclude=None
    ) -> "InferenceWorker":
        """加权轮询负载均衡"""
        available = [
            w for w in self.workers 
            if w.is_healthy and w not in (exclude or [])
        ]
        # 按GPU显存使用率加权，显存越空闲权重越高
        weights = [1.0 - w.gpu_memory_usage for w in available]
        return random.choices(available, weights=weights, k=1)[0]
```

#### 3.3.2 SSE协议适配

```python
# SSE 协议封装
import json
from dataclasses import dataclass

@dataclass
class SSEEvent:
    event: str = "message"
    data: str = ""
    id: str | None = None
    retry: int | None = None
    
    def to_http_response(self) -> str:
        lines = []
        if self.event:
            lines.append(f"event: {self.event}")
        if self.id:
            lines.append(f"id: {self.id}")
        if self.retry:
            lines.append(f"retry: {self.retry}")
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")
        lines.append("")  # 空行分隔
        return "\n".join(lines) + "\n"

async def stream_to_sse(
    stream: AsyncIterator[StreamChunk]
) -> AsyncIterator[str]:
    """将内部流转为 SSE 格式"""
    
    event_id = 0
    async for chunk in stream:
        event_id += 1
        sse_event = SSEEvent(
            event="message",
            data=json.dumps({
                "id": event_id,
                "content": chunk.text,
                "finish_reason": chunk.finish_reason,
            }),
            id=str(event_id)
        )
        yield sse_event.to_http_response()
    
    # 发送结束事件
    yield SSEEvent(
        event="done",
        data="[DONE]"
    ).to_http_response()
```

#### 3.3.3 背压控制

流式系统中，如果下游消费速度慢于上游生产速度，会导致内存堆积。背压（Backpressure）机制确保系统在高负载下稳定运行：

```python
# 基于 asyncio.Queue 的背压控制
class BoundedStream:
    def __init__(self, max_buffer: int = 128):
        self._queue = asyncio.Queue(maxsize=max_buffer)
        self._closed = False
    
    async def push(self, chunk: StreamChunk):
        """生产者：推送数据块（队列满时阻塞）"""
        if self._closed:
            return
        await self._queue.put(chunk)  # maxsize 满时自动阻塞
    
    async def pop(self) -> StreamChunk | None:
        """消费者：获取下一个数据块"""
        try:
            return await asyncio.wait_for(
                self._queue.get(), timeout=5.0
            )
        except asyncio.TimeoutError:
            return None  # 超时视为流结束
    
    async def close(self):
        self._closed = True
        await self._queue.put(None)  # 哨兵值通知消费者
```

## 四、SSE连接的可靠性工程

### 4.1 断线重连策略

SSE连接可能因为网络抖动、代理超时等原因中断。可靠的重连策略是生产系统的必备能力：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SSE 断线重连状态机                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│                    ┌──────────┐                                      │
│              ┌────→│ CONNECTED │←──────┐                             │
│              │     └────┬─────┘       │                             │
│              │          │             │                              │
│     心跳超时 │          │ 连接断开     │ 重连成功                      │
│              │          ▼             │                              │
│              │    ┌──────────┐       │                              │
│              │    │RECONNECTING│─────┘                              │
│              │    └────┬─────┘                                      │
│              │         │                                            │
│              │    指数退避                                           │
│              │    1s → 2s → 4s → 8s → 16s → 30s (max)              │
│              │         │                                            │
│              │    重试次数 > MAX_RETRIES                             │
│              │         ▼                                            │
│              │    ┌──────────┐                                      │
│              └────│  FAILED   │                                     │
│                   └──────────┘                                      │
│                                                                       │
│  关键机制:                                                            │
│  1. Last-Event-ID: 浏览器自动携带，服务端可从此ID恢复                  │
│  2. 心跳检测: 服务端每30秒发送 :ping 保活                             │
│  3. 断点续传: 根据 event-id 从断点继续推送                            │
│  4. 降级策略: SSE失败后降级为轮询                                     │
└──────────────────────────────────────────────────────────────────────┘
```

**客户端实现（JavaScript）：**

```javascript
class ResilientSSEClient {
    constructor(url, options = {}) {
        this.url = url;
        this.maxRetries = options.maxRetries || 10;
        this.baseDelay = options.baseDelay || 1000;
        this.maxDelay = options.maxDelay || 30000;
        this.retryCount = 0;
        this.lastEventId = null;
    }

    connect() {
        const headers = {};
        if (this.lastEventId) {
            headers['Last-Event-ID'] = this.lastEventId;
        }

        const eventSource = new EventSource(this.url, { headers });

        eventSource.onmessage = (event) => {
            this.retryCount = 0; // 成功收到消息，重置计数
            this.lastEventId = event.lastEventId;
            this.onData(JSON.parse(event.data));
        };

        eventSource.onerror = () => {
            eventSource.close();
            this.scheduleReconnect();
        };
    }

    scheduleReconnect() {
        if (this.retryCount >= this.maxRetries) {
            this.onFailure(new Error('Max retries exceeded'));
            return;
        }

        // 指数退避 + 随机抖动
        const delay = Math.min(
            this.baseDelay * Math.pow(2, this.retryCount) 
                + Math.random() * 1000,
            this.maxDelay
        );
        
        this.retryCount++;
        setTimeout(() => this.connect(), delay);
    }
}
```

### 4.2 超时与资源管理

```python
# 服务端超时管理
import time
from contextlib import asynccontextmanager

STREAM_CONFIGS = {
    "default": {
        "max_duration": 30,      # 最大持续时间: 30秒
        "max_tokens": 4096,      # 最大输出token数
        "heartbeat_interval": 15, # 心跳间隔: 15秒
    },
    "long_context": {
        "max_duration": 120,
        "max_tokens": 16384,
        "heartbeat_interval": 15,
    }
}

@asynccontextmanager
async def managed_stream(config_name: str = "default"):
    """带资源管理的流式上下文"""
    config = STREAM_CONFIGS[config_name]
    start_time = time.monotonic()
    token_count = 0
    
    async def check_limits():
        nonlocal token_count
        elapsed = time.monotonic() - start_time
        
        if elapsed > config["max_duration"]:
            raise StreamLimitExceeded("Duration limit exceeded")
        if token_count > config["max_tokens"]:
            raise StreamLimitExceeded("Token limit exceeded")
    
    try:
        yield check_limits
    finally:
        # 清理：确保连接和GPU资源被正确释放
        elapsed = time.monotonic() - start_time
        log_stream_metrics(
            duration=elapsed,
            tokens=token_count,
            config=config_name
        )
```

### 4.3 心跳保活机制

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SSE 心跳保活机制                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  服务端                                                                │
│  ┌────────────────────────────────────────────────────┐              │
│  │  Stream Worker                                      │              │
│  │                                                     │              │
│  │  每 15 秒发送心跳:                                   │              │
│  │  : ping                                              │              │
│  │  (注释行，以冒号开头，浏览器会忽略但保持连接)            │              │
│  │                                                     │              │
│  │  推送真实数据:                                       │              │
│  │  data: {"content": "你"}                             │              │
│  │  data: {"content": "好"}                             │              │
│  │  : ping  ← 插入心跳                                  │              │
│  │  data: {"content": "！"}                             │              │
│  └────────────────────────────────────────────────────┘              │
│                                                                       │
│  为什么需要心跳:                                                       │
│  1. Nginx 默认 60s proxy_read_timeout，无数据会断开                   │
│  2. CDN/AWS ALB 等中间件有空闲连接超时                                 │
│  3. 移动网络切换时连接可能静默丢失                                     │
│  4. 客户端检测连接是否存活                                             │
└──────────────────────────────────────────────────────────────────────┘
```

## 五、高性能流式推理引擎集成

### 5.1 vLLM 流式输出架构

vLLM 是目前最流行的LLM推理引擎之一，其流式输出基于连续批处理（Continuous Batching）实现：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    vLLM 流式推理内部流程                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Request Queue                                                        │
│  ┌─────────────────────────────────────────┐                        │
│  │ [Req1] [Req2] [Req3] [Req4] ...        │                        │
│  └───────────────┬─────────────────────────┘                        │
│                  ▼                                                    │
│  ┌─────────────────────────────────────────┐                        │
│  │         Scheduler (调度器)               │                        │
│  │  - 检查 KV Cache 可用空间                │                        │
│  │  - 决定 prefill / decode / swap          │                        │
│  │  - Continuous Batching: 动态加入新请求    │                        │
│  └───────────────┬─────────────────────────┘                        │
│                  ▼                                                    │
│  ┌─────────────────────────────────────────┐                        │
│  │         Model Executor                   │                        │
│  │  ┌─────────────────────────────────┐   │                        │
│  │  │  GPU Worker                     │   │                        │
│  │  │  - Prefill: 处理输入token        │   │                        │
│  │  │  - Decode: 逐token生成           │   │                        │
│  │  │  - Speculative: 推测解码加速      │   │                        │
│  │  └─────────────────────────────────┘   │                        │
│  └───────────────┬─────────────────────────┘                        │
│                  ▼                                                    │
│  ┌─────────────────────────────────────────┐                        │
│  │         Stream Output Handler           │                        │
│  │  - 每个 decode step 产生一个 token       │                        │
│  │  - 立即回调到对应的 HTTP stream           │                        │
│  │  - 不等整个序列生成完毕                   │                        │
│  └─────────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 流式输出与连续批处理的协同

连续批处理（Continuous Batching）是vLLM/SGLang的核心优化，它与流式输出天然协同：

```
传统静态批处理:
时间 →
Batch 1: [====Prefill====][==Decode==][==Decode==] → Done
Batch 2:                     [====Prefill====][==Decode==][==Decode==] → Done
         ↑ 短请求等长请求完成，GPU空闲浪费

连续批处理 (Continuous Batching):
时间 →
Req A:   [Prefill][Decode][Decode][Decode][Decode] → Done
Req B:          [Prefill][Decode][Decode][Decode]  → Done  
Req C:                    [Prefill][Decode][Decode] → Done
Req D:                              [Prefill][Decode] → Done
         ↑ 短请求完成立即释放资源，新请求立即加入

流式输出 + 连续批处理的协同:
1. Req A 完成第一个 decode step → 立即通过 SSE 推送 token
2. Req B 的 prefill 完成 → 立即开始 decode → 流式推送
3. GPU 利用率始终维持在 90%+
4. 用户感知: 每个请求几乎立即开始输出
```

### 5.3 多租户流式隔离

在多租户场景下，需要确保一个租户的流式请求不会影响其他租户：

```python
# 多租户流式资源隔离
class TenantStreamIsolator:
    """基于令牌桶的租户级流式限流"""
    
    def __init__(self):
        self.tenant_quotas = {
            "free": {"max_concurrent": 2, "max_tokens_per_sec": 50},
            "pro": {"max_concurrent": 10, "max_tokens_per_sec": 200},
            "enterprise": {"max_concurrent": 50, "max_tokens_per_sec": 1000},
        }
        self.tenant_streams: dict[str, int] = {}  # 当前并发数
    
    async def acquire_slot(self, tenant: str) -> bool:
        quota = self.tenant_quotas.get(tenant, self.tenant_quotas["free"])
        current = self.tenant_streams.get(tenant, 0)
        
        if current >= quota["max_concurrent"]:
            return False  # 并发限制
        
        self.tenant_streams[tenant] = current + 1
        return True
    
    def release_slot(self, tenant: str):
        self.tenant_streams[tenant] = max(
            0, self.tenant_streams.get(tenant, 1) - 1
        )
```

## 六、可观测性：流式系统的监控挑战

流式系统的监控比传统API复杂得多，因为一个请求可能持续数十秒，期间会产生大量中间状态。

### 6.1 关键监控指标

```
┌──────────────────────────────────────────────────────────────────────┐
│                    流式系统监控指标体系                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. 延迟指标                                                          │
│  ├── TTFT (Time To First Token): 首token延迟                         │
│  │   目标: P50 < 200ms, P99 < 500ms                                  │
│  ├── TPOT (Time Per Output Token): 每token生成时间                    │
│  │   目标: P50 < 50ms                                                 │
│  └── E2E Latency: 端到端总延迟                                        │
│      目标: P50 < 5s, P99 < 15s                                       │
│                                                                       │
│  2. 吞吐指标                                                          │
│  ├── Tokens Per Second (TPS): 每秒输出token总数                       │
│  │   目标: 单GPU > 50 TPS (7B模型)                                   │
│  ├── Concurrent Streams: 当前活跃流式连接数                            │
│  │   目标: < 80% 最大容量                                             │
│  └── Queue Depth: 等待队列深度                                        │
│      目标: P99 < 10                                                   │
│                                                                       │
│  3. 质量指标                                                          │
│  ├── Completion Rate: 流式完成率（非异常中断）                          │
│  │   目标: > 99.5%                                                    │
│  ├── Timeout Rate: 超时率                                             │
│  │   目标: < 0.5%                                                     │
│  └── Error Rate: 错误率                                               │
│      目标: < 0.1%                                                     │
│                                                                       │
│  4. 资源指标                                                          │
│  ├── GPU Utilization: GPU利用率                                       │
│  │   目标: > 85%                                                      │
│  ├── GPU Memory: 显存使用率                                           │
│  │   KV Cache 命中率 > 90%                                            │
│  └── Connection Count: 活跃连接数                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 分布式追踪实现

```python
# 流式请求的 OpenTelemetry 追踪
from opentelemetry import trace
from opentelemetry.trace import StatusCode

tracer = trace.get_tracer("stream-ai")

async def traced_stream_completion(request, tenant):
    with tracer.start_as_current_span("stream_completion") as span:
        span.set_attribute("tenant", tenant)
        span.set_attribute("model", request.model)
        
        # 记录 TTFT
        start_time = time.monotonic()
        first_token_recorded = False
        
        async for chunk in stream_proxy.stream_completion(request):
            if not first_token_recorded:
                ttft = (time.monotonic() - start_time) * 1000
                span.set_attribute("ttft_ms", ttft)
                span.add_event("first_token", {"ttft_ms": ttft})
                first_token_recorded = True
            
            # 每100个token记录一次进度
            span.add_event("token_progress", {
                "token_count": chunk.token_count,
            })
            
            yield chunk
        
        # 记录最终指标
        total_time = (time.monotonic() - start_time) * 1000
        span.set_attribute("total_latency_ms", total_time)
        span.set_attribute("total_tokens", chunk.token_count)
        span.set_status(StatusCode.OK)
```

## 七、实战部署方案

### 7.1 Nginx 配置要点

Nginx是流式AI应用中最常见的反向代理，但默认配置不适合流式场景：

```nginx
# nginx.conf - 流式AI应用专用配置
upstream llm_backend {
    server 10.0.1.1:8000 weight=3;  # vLLM Worker 1
    server 10.0.1.2:8000 weight=3;  # vLLM Worker 2
    server 10.0.1.3:8000 weight=2;  # SGLang Worker
    keepalive 64;  # 保持后端长连接
}

server {
    listen 443 ssl;
    server_name api.example.com;

    # 关键：禁用缓冲，启用流式代理
    location /v1/chat/completions {
        proxy_pass http://llm_backend;
        
        # 流式代理核心配置
        proxy_buffering off;           # 禁用响应缓冲！
        proxy_cache off;               # 禁用缓存
        proxy_http_version 1.1;
        proxy_set_header Connection ""; # 启用 keepalive
        
        # SSE 超时设置
        proxy_read_timeout 120s;       # 读超时设为120秒
        proxy_send_timeout 120s;
        
        # 压缩（SSE文本压缩效果好）
        gzip on;
        gzip_types text/event-stream;
        gzip_min_length 1024;
        
        # CORS
        add_header Access-Control-Allow-Origin *;
        add_header Cache-Control "no-cache";
        add_header Content-Type "text/event-stream";
    }
}
```

### 7.2 Kubernetes 部署架构

```yaml
# 流式推理服务 K8s 部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-streaming
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-streaming
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "Qwen/Qwen2.5-7B-Instruct"
        - "--max-model-len"
        - "32768"
        - "--gpu-memory-utilization"
        - "0.9"
        - "--enable-chunked-prefill"  # 启用分块预填充
        - "--max-num-seqs"
        - "64"                         # 最大并发序列数
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            memory: 24Gi
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-streaming-svc
spec:
  selector:
    app: vllm-streaming
  ports:
  - port: 8000
    targetPort: 8000
  # 使用 ClusterIP + Ingress，不用 LoadBalancer
  # 避免每个连接都占用一个公网LB连接
```

## 八、总结与最佳实践

### 8.1 技术选型速查

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| LLM流式输出（Web端） | SSE | 浏览器原生支持，代理友好 |
| LLM流式输出（移动端） | SSE或gRPC | SSE简单，gRPC性能更好 |
| 实时协作AI应用 | WebSocket | 需要双向通信 |
| 微服务间流式通信 | gRPC Streaming | 高性能，强类型 |
| 跨平台SDK | gRPC + SSE适配层 | 统一底层，多协议输出 |

### 8.2 核心最佳实践

1. **永远开启流式**：对于LLM应用，流式输出应该是默认行为
2. **TTFT是第一优先级**：首token延迟比总延迟更重要
3. **心跳保活不可省**：Nginx/CDN的默认超时是60秒
4. **背压控制必须有**：防止慢消费者拖垮整个系统
5. **监控TTFT和TPOT**：这两个指标比QPS更能反映用户体验
6. **渐进式降级**：SSE失败→轮询→缓存响应，保证可用性
7. **连接数有上限**：流式连接占用资源远超普通API，必须限制并发

流式架构看似是传输层的小改动，实际上涉及协议设计、连接管理、资源调度、可观测性等多个维度的系统性工程。理解这些底层机制，才能构建出真正可靠、高性能的AI应用。
