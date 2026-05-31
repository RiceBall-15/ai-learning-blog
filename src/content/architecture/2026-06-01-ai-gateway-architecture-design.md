---
title: "AI网关架构设计：从统一入口到智能流量治理"
description: "系统剖析AI应用网关的架构设计，涵盖LLM请求路由、智能限流、Token计量、故障降级等核心能力，结合生产经验给出可落地的架构方案。"
date: 2026-06-01
author: "RiceBall"
category: "architecture"
subCategory: "distributed"
tags: ["AI网关", "架构设计", "流量治理", "LLM运维", "分布式系统"]
draft: false
---

# AI网关架构设计：从统一入口到智能流量治理

## 前言

随着企业内部 AI 应用的爆发式增长，一个不可回避的问题浮出水面：**如何统一管理数十个甚至上百个 AI 应用对 LLM 服务的访问？**

每个团队各自对接 OpenAI、Claude、通义千问等模型，独立处理重试、限流、计费——这种「各自为政」的模式正在制造大量的运维成本和资源浪费。正如微服务架构催生了 API Gateway，AI 应用的爆发同样催生了 **AI Gateway** 这一新的架构角色。

本文将系统性地拆解 AI 网关的核心能力，分享从 0 到 1 构建企业级 AI Gateway 的架构经验和踩坑教训。

## 1. 为什么需要 AI 网关

### 1.1 现状痛点

在没有 AI 网关的企业环境中，典型场景如下：

```
┌──────────┐     ┌──────────┐
│ 团队 A    │────→│ OpenAI   │  重复处理：重试、超时、计费
│ AI 应用  │     └──────────┘
├──────────┤     ┌──────────┐
│ 团队 B    │────→│ Claude   │  重复处理：重试、超时、计费  
│ AI 应用  │     └──────────┘
├──────────┤     ┌──────────┐
│ 团队 C    │────→│ 通义千问  │  重复处理：重试、超时、计费
│ AI 应用  │     └──────────┘
├──────────┤     ┌──────────┐
│ 团队 D    │────→│ 本地模型  │  重复处理：重试、超时、计费
│ AI 应用  │     └──────────┘
```

每个 AI 应用都要自己处理：
- API Key 管理与轮转
- 多模型接口适配
- 限流与重试策略
- Token 用量统计与计费
- 故障降级与模型切换
- 审计日志与合规

**这些横切关注点不应该由每个业务团队重复实现。**

### 1.2 AI 网关的核心价值

```
┌─────────────────────────────────────────────┐
│              AI Gateway                      │
│                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ 路由策略 │  │ 限流熔断 │  │ 用量计量 │    │
│  └─────────┘  └─────────┘  └─────────┘    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ 协议适配 │  │ 故障降级 │  │ 安全审计 │    │
│  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────┘
         ↑                        ↓
    业务应用层               模型服务层
    (统一接入)            (统一管理)
```

核心价值可以概括为三个「统一」：
- **统一入口**：所有 AI 应用通过网关访问 LLM，无需各自对接
- **统一治理**：限流、熔断、审计等策略在网关层集中实施
- **统一观测**：全量请求的延迟、成功率、Token 用量一目了然

## 2. AI 网关核心架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      AI Gateway 架构                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Ingress Layer                       │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │ REST API │ │ SSE 流式  │ │ gRPC     │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Processing Pipeline                  │   │
│  │                                                       │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ │   │
│  │  │ 鉴权  │→│ 限流  │→│ 路由  │→│ 转换  │→│ 代理  │ │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘ │   │
│  │                                       ↓             │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────────┐   │   │
│  │  │ 审计  │←│ 计量  │←│ 缓存  │←│ 响应处理      │   │   │
│  │  └──────┘  └──────┘  └──────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Model Adapter Layer                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │ OpenAI   │ │ Claude   │ │ 自建模型  │ ...        │   │
│  │  │ Adapter  │ │ Adapter  │ │ Adapter  │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                Observability Layer                    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │ Metrics  │ │ Tracing  │ │ Logging  │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块详解

#### 路由策略引擎

路由是 AI 网关最关键的能力之一。不同于传统 API Gateway 的简单 URL 路由，AI 网关需要基于**请求内容**做智能决策：

```python
from dataclasses import dataclass
from enum import Enum

class RouteStrategy(Enum):
    ROUND_ROBIN = "round_robin"          # 轮询
    LEAST_LATENCY = "least_latency"      # 最低延迟
    COST_OPTIMIZED = "cost_optimized"    # 成本最优
    QUALITY_FIRST = "quality_first"      # 质量优先
    FAILOVER = "failover"               # 故障转移

@dataclass
class RouteRule:
    """路由规则定义"""
    name: str
    match: dict          # 匹配条件
    targets: list        # 目标模型组
    strategy: RouteStrategy
    priority: int = 0

class AIRouter:
    def __init__(self):
        self.rules: list[RouteRule] = []
        self.model_registry: dict = {}
    
    def add_rule(self, rule: RouteRule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def route(self, request: dict) -> list[str]:
        """根据请求内容和规则返回模型优先级列表"""
        for rule in self.rules:
            if self._match(rule.match, request):
                return self._apply_strategy(rule.strategy, rule.targets)
        
        # 默认路由
        return self._default_route()
    
    def _match(self, conditions: dict, request: dict) -> bool:
        """匹配规则条件"""
        for key, value in conditions.items():
            if key == "task_type":
                if request.get("task_type") != value:
                    return False
            elif key == "content_length_min":
                if len(request.get("messages", [{}])[-1].get("content", "")) < value:
                    return False
            elif key == "user_group":
                if request.get("user_group") != value:
                    return False
        return True
    
    def _apply_strategy(self, strategy: RouteStrategy, targets: list) -> list:
        """应用路由策略"""
        if strategy == RouteStrategy.LEAST_LATENCY:
            return sorted(targets, key=lambda t: self.model_registry[t]["p50_latency"])
        elif strategy == RouteStrategy.COST_OPTIMIZED:
            return sorted(targets, key=lambda t: self.model_registry[t]["cost_per_1k_tokens"])
        elif strategy == RouteStrategy.QUALITY_FIRST:
            return sorted(targets, key=lambda t: -self.model_registry[t]["quality_score"])
        else:
            return targets
```

**路由规则配置示例**：

```yaml
routes:
  # 简单任务走低成本模型
  - name: "simple-tasks"
    match:
      task_type: "classification"
    targets: ["gpt-4o-mini", "qwen-turbo"]
    strategy: cost_optimized
    priority: 100

  # 长文本任务走大上下文模型
  - name: "long-context"
    match:
      content_length_min: 8000
    targets: ["gpt-4o", "claude-3-opus"]
    strategy: quality_first
    priority: 90

  # 生产环境优先使用本地部署模型
  - name: "production-safe"
    match:
      user_group: "production"
    targets: ["local-qwen-72b", "gpt-4o"]
    strategy: failover
    priority: 80
```

#### 智能限流与配额管理

AI 网关的限流不能简单使用传统 API 的 QPS 限流，需要引入**多维度限流**：

```python
import time
from collections import defaultdict

class MultiDimensionRateLimiter:
    """多维度限流器：支持按用户、组织、模型、Token 等维度限流"""
    
    def __init__(self):
        # 滑动窗口计数器
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._quotas: dict[str, dict] = {}
    
    def set_quota(self, key: str, limits: dict):
        """
        设置配额
        limits 示例: {"qps": 10, "rpm": 200, "tpm": 100000, "daily_tokens": 10000000}
        """
        self._quotas[key] = limits
    
    def check(self, key: str, token_count: int = 0) -> tuple[bool, str]:
        """检查是否超限，返回 (allowed, reason)"""
        if key not in self._quotas:
            return True, ""
        
        quota = self._quotas[key]
        now = time.time()
        
        # QPS 检查
        if "qps" in quota:
            window = self._windows[f"{key}:qps"]
            window.append(now)
            window[:] = [t for t in window if now - t < 1.0]
            if len(window) > quota["qps"]:
                return False, f"QPS limit exceeded: {len(window)}/{quota['qps']}"
        
        # RPM 检查
        if "rpm" in quota:
            window = self._windows[f"{key}:rpm"]
            window.append(now)
            window[:] = [t for t in window if now - t < 60.0]
            if len(window) > quota["rpm"]:
                return False, f"RPM limit exceeded: {len(window)}/{quota['rpm']}"
        
        # TPM 检查（Token per minute）
        if "tpm" in quota:
            window = self._windows[f"{key}:tpm"]
            window.append((now, token_count))
            window[:] = [(t, tc) for t, tc in window if now - t < 60.0]
            total_tokens = sum(tc for _, tc in window)
            if total_tokens + token_count > quota["tpm"]:
                return False, f"TPM limit exceeded: {total_tokens + token_count}/{quota['tpm']}"
        
        return True, ""
```

#### Token 计量与计费

Token 用量是 AI 网关的核心数据之一。需要区分处理输入 Token 和输出 Token，因为它们的计费单价通常不同：

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: float
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

class TokenMeter:
    """Token 计量器"""
    
    # 各模型定价 (USD per 1M tokens)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "qwen-turbo": {"input": 0.30, "output": 0.60},
    }
    
    def __init__(self, storage):
        self.storage = storage  # 时序数据库或 ClickHouse
    
    async def record(self, usage: TokenUsage):
        """记录一次请求的 Token 用量"""
        cost = self._calculate_cost(usage)
        
        await self.storage.insert({
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost,
            "timestamp": usage.timestamp,
        })
        
        # 更新实时统计
        await self._update_realtime_stats(usage, cost)
    
    def _calculate_cost(self, usage: TokenUsage) -> float:
        pricing = self.PRICING.get(usage.model, {"input": 0, "output": 0})
        return (
            usage.input_tokens * pricing["input"] / 1_000_000 +
            usage.output_tokens * pricing["output"] / 1_000_000
        )
    
    async def get_usage_report(
        self, 
        org_id: str, 
        start_date: str, 
        end_date: str
    ) -> dict:
        """生成用量报告"""
        return await self.storage.query(f"""
            SELECT 
                model,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                SUM(cost_usd) as total_cost,
                COUNT(*) as request_count,
                AVG(input_tokens + output_tokens) as avg_tokens_per_req
            FROM token_usage
            WHERE org_id = '{org_id}'
              AND timestamp BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY model
            ORDER BY total_cost DESC
        """)
```

## 3. 故障降级与容错设计

### 3.1 降级策略

AI 服务的可用性天然不如传统 Web 服务稳定，降级策略是 AI 网关的生命线：

```python
class FallbackChain:
    """故障降级链"""
    
    def __init__(self):
        self._health_status: dict[str, bool] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
    
    async def execute_with_fallback(
        self, 
        request: dict, 
        primary_model: str,
        fallback_models: list[str]
    ) -> dict:
        """带降级的请求执行"""
        
        # 尝试主模型
        if self._is_healthy(primary_model):
            try:
                return await self._call_model(primary_model, request)
            except ModelException as e:
                self._record_failure(primary_model, e)
        
        # 主模型失败，遍历降级链
        for fallback in fallback_models:
            if self._is_healthy(fallback):
                try:
                    logger.warning(f"Fallback from {primary_model} to {fallback}")
                    return await self._call_model(fallback, request)
                except ModelException as e:
                    self._record_failure(fallback, e)
        
        # 全部失败，返回缓存或默认响应
        return await self._emergency_fallback(request)
    
    async def _emergency_fallback(self, request: dict) -> dict:
        """紧急降级：返回缓存的相似回答或预设回复"""
        # 尝试从语义缓存中查找相似问题的历史回答
        cached = await self.semantic_cache.find_similar(request)
        if cached:
            return {
                "choices": [{"message": {
                    "content": f"[缓存回答] {cached['answer']}"
                }}],
                "model": "cache",
                "_fallback": True
            }
        
        # 最终兜底
        return {
            "choices": [{"message": {
                "content": "服务暂时繁忙，请稍后重试。如需紧急支持请联系管理员。"
            }}],
            "model": "fallback_default",
            "_fallback": True
        }
```

### 3.2 断路器实现

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 熔断
    HALF_OPEN = "half_open" # 探测恢复

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0
    
    def can_execute(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                return True
            return False
        else:  # HALF_OPEN
            return self._half_open_calls < self.half_open_max_calls
    
    def record_success(self):
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
    
    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
```

## 4. 流式响应处理

LLM 的流式响应（SSE）是 AI 网关区别于传统 API Gateway 的重要特征。网关需要正确透传流式数据，同时完成 Token 统计：

```python
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def proxy_streaming_response(
    upstream_url: str,
    request_body: dict,
    usage_callback
):
    """流式代理：透传 SSE 数据，同时统计 Token 用量"""
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", upstream_url,
            json=request_body,
            headers={"Accept": "text/event-stream"}
        ) as response:
            
            input_tokens = request_body.get("_estimated_input_tokens", 0)
            output_tokens = 0
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    
                    if data_str == "[DONE]":
                        # 请求完成，回调用量
                        await usage_callback(input_tokens, output_tokens)
                        yield f"data: [DONE]\n\n"
                        continue
                    
                    try:
                        data = json.loads(data_str)
                        # 累计输出 Token
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            # 粗略估算：1 个中文字 ≈ 1.5 tokens
                            output_tokens += max(1, len(content) * 1.5)
                        
                        # 检查 usage 字段（部分模型在最后一个 chunk 返回）
                        if "usage" in data:
                            output_tokens = data["usage"].get("completion_tokens", output_tokens)
                        
                        yield f"data: {data_str}\n\n"
                    except json.JSONDecodeError:
                        yield f"data: {data_str}\n\n"
                elif line.strip():
                    yield f"{line}\n\n"
```

## 5. 可观测性体系

### 5.1 核心监控面板

一个完善的 AI 网关监控面板应包含以下维度：

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Gateway Dashboard                      │
│                                                              │
│  📊 实时指标                                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ QPS: 142 │ │ P50: 832ms│ │ P99: 3.2s│ │ 错误: 0.3%│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                              │
│  💰 今日成本                                                │
│  ┌──────────────────────────────────────────────────┐      │
│  │ GPT-4o:    $234.50  ████████████████░░░░  78%    │      │
│  │ Claude:    $45.20   ███░░░░░░░░░░░░░░░░  15%    │      │
│  │ Qwen:      $21.30   ██░░░░░░░░░░░░░░░░░   7%    │      │
│  │ 合计:      $301.00                              │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  📈 请求趋势 (24h)                                          │
│  ┌──────────────────────────────────────────────────┐      │
│  │     ╭─╮                                           │      │
│  │   ╭─╯ ╰╮  ╭──╮                                   │      │
│  │ ╭─╯     ╰──╯  ╰───╮  ╭─╮                         │      │
│  │─╯                  ╰──╯ ╰──                       │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  🔴 告警                                                    │
│  [WARN] gpt-4o P99 延迟超过 5s 阈值 (当前: 5.8s)          │
│  [INFO] claude-3-opus 配额已用 85%                          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 分布式追踪

```python
import uuid
from contextvars import ContextVar

# 请求级别的追踪上下文
trace_context: ContextVar[dict] = ContextVar('trace_context', default={})

class AIGatewayTracer:
    """AI 网关分布式追踪"""
    
    def start_trace(self, request_id: str = None) -> str:
        req_id = request_id or str(uuid.uuid4())[:8]
        ctx = {
            "request_id": req_id,
            "spans": [],
            "start_time": time.time()
        }
        trace_context.set(ctx)
        return req_id
    
    def add_span(self, name: str, metadata: dict = None):
        ctx = trace_context.get()
        span = {
            "name": name,
            "start": time.time(),
            "metadata": metadata or {}
        }
        ctx["spans"].append(span)
        return span
    
    def finish_span(self, span: dict, result: str = "ok"):
        span["end"] = time.time()
        span["duration_ms"] = (span["end"] - span["start"]) * 1000
        span["result"] = result
    
    def finish_trace(self) -> dict:
        ctx = trace_context.get()
        ctx["total_duration_ms"] = (time.time() - ctx["start_time"]) * 1000
        return ctx

# 使用示例
tracer = AIGatewayTracer()

async def handle_request(request):
    req_id = tracer.start_trace()
    
    # 鉴权
    auth_span = tracer.add_span("auth", {"method": request.auth_method})
    auth_result = await authenticate(request)
    tracer.finish_span(auth_span, auth_result.status)
    
    # 路由
    route_span = tracer.add_span("route", {"model": request.model})
    target = router.route(request)
    tracer.finish_span(route_span, target)
    
    # 调用模型
    model_span = tracer.add_span("model_call", {
        "model": target,
        "input_tokens": estimate_tokens(request.messages)
    })
    response = await call_model(target, request)
    tracer.finish_span(model_span, f"output_tokens={response.usage.output_tokens}")
    
    # 完成追踪
    trace = tracer.finish_trace()
    await metrics.record_trace(trace)
    
    return response
```

## 6. 部署与演进

### 6.1 部署架构建议

```
┌─────────────────────────────────────────────┐
│              负载均衡层 (Nginx/ALB)           │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ GW Pod  │ │ GW Pod  │ │ GW Pod  │  无状态，水平扩展
   │  (主)   │ │  (从)   │ │  (从)   │
   └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │
   ┌────┴───────────┴───────────┴────┐
   │         共享状态层               │
   │  ┌──────┐ ┌──────┐ ┌────────┐  │
   │  │Redis │ │Click-│ │etcd    │  │
   │  │(限流)│ │House │ │(配置)  │  │
   │  └──────┘ │(日志)│ └────────┘  │
   └───────────┴──────┴─────────────┘
```

### 6.2 演进路线图

| 阶段 | 核心能力 | 技术重点 |
|------|---------|---------|
| V1.0 | 基础代理 + API Key 管理 | 多模型适配、流式透传 |
| V2.0 | 限流 + 计量 + 监控 | 多维度限流、Token 计费 |
| V3.0 | 智能路由 + 故障降级 | 内容感知路由、断路器 |
| V4.0 | Prompt 缓存 + 语义缓存 | 相似请求去重、成本优化 |
| V5.0 | AI 安全审计 + 合规 | 内容审核、数据脱敏 |

## 总结

AI 网关是企业 AI 基础设施的关键组件，它的核心价值在于：

1. **降低接入成本**：AI 应用无需关心底层模型差异
2. **提升资源效率**：通过智能路由和缓存减少不必要的模型调用
3. **保障服务质量**：限流、熔断、降级形成完整的可用性保障体系
4. **实现精细管控**：Token 计量、审计日志满足企业合规要求

构建 AI 网关不是一蹴而就的事情，建议从最小可用版本（代理 + 鉴权）开始，根据业务需求逐步增加能力模块。最重要的是，**始终以业务价值为导向**，避免过度工程化。

---

*本文基于多个企业级 AI 网关项目经验总结，部分实现细节可能因具体技术栈而有所差异。*
