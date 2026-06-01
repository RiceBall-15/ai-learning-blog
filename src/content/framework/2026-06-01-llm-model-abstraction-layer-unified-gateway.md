---
title: "LLM应用的统一模型抽象层：从硬编码调用到智能模型路由的架构演进实战"
description: "深度解析LLM应用中模型抽象层的设计与实现，覆盖多模型适配、智能路由、故障转移、成本优化与流式响应的全链路生产级方案"
date: 2026-06-01
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["模型路由", "LLM网关", "多模型架构", "成本优化", "故障转移", "抽象层"]
draft: false
---

## 引言：为什么需要模型抽象层？

在LLM应用的早期阶段，大多数团队的做法是直接在代码中硬编码调用某个模型的API：

```python
# 典型的"能跑就行"阶段
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)
```

这种模式在原型阶段无可厚非，但当应用进入生产环境，问题会迅速暴露：

- **模型供应商锁定**：切换模型意味着大量代码修改
- **缺乏故障转移**：单一模型API不可用时整个应用宕机
- **成本失控**：所有请求都用最贵的模型
- **无法A/B测试**：没有基础设施支撑模型对比实验
- **流式响应碎片化**：不同供应商的SSE协议差异导致重复开发

本文将从实战经验出发，详细介绍如何构建一个生产级的LLM模型抽象层，实现**统一接入、智能路由、弹性降级和成本优化**。

## 架构演进：从v1到v3

### v1：直接调用（原始阶段）

```
┌─────────┐     ┌──────────────┐     ┌──────────┐
│  Application │────▶│ OpenAI Client │────▶│ GPT-4o   │
└─────────┘     └──────────────┘     └──────────┘
```

**问题**：模型切换需要改代码，无故障转移，无法对比模型效果。

### v2：简单适配器（过渡阶段）

```
┌─────────┐     ┌──────────────┐     ┌──────────┐
│  Application │────▶│ ModelAdapter  │────▶│ Any Model│
└─────────┘     └──────────────┘     └──────────┘
```

引入了统一的接口，但路由逻辑、故障转移、成本追踪仍然缺失。

### v3：智能模型网关（生产阶段）

```
┌──────────────────────────────────────────────┐
│              Model Gateway                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Router   │  │  Retry   │  │  Cost    │  │
│  │  Engine   │  │  Handler │  │  Tracker │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Stream   │  │  Fallback│  │  Metrics │  │
│  │  Adapter  │  │  Chain   │  │  Collector│ │
│  └──────────┘  └──────────┘  └──────────┘  │
└──────────────────────────────────────────────┘
         │              │              │
    ┌────┴────┐   ┌─────┴─────┐  ┌────┴────┐
    │  GPT-4o  │   │ Claude 3.5 │  │ Qwen-Max│
    │  Gemini  │   │ Llama 3.1  │  │ DeepSeek│
    └─────────┘   └───────────┘  └─────────┘
```

## 核心设计：统一模型接口

### 1. 抽象接口定义

模型抽象层的核心是定义一个**统一的请求/响应协议**，屏蔽不同供应商的API差异：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import time

@dataclass
class ModelRequest:
    """统一请求模型"""
    messages: list[dict]
    model_hint: str = ""  # 模型提示（非强制）
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    response_format: Optional[dict] = None
    tools: Optional[list[dict]] = None
    metadata: dict = field(default_factory=dict)  # 追踪用

@dataclass
class ModelResponse:
    """统一响应模型"""
    content: str
    model: str  # 实际使用的模型
    provider: str
    usage: dict  # {"prompt_tokens": x, "completion_tokens": y}
    latency_ms: float
    finish_reason: str
    tool_calls: Optional[list[dict]] = None

@dataclass
class StreamChunk:
    """流式响应块"""
    delta: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None

class ModelProvider(ABC):
    """模型供应商抽象接口"""
    
    @abstractmethod
    async def chat(self, request: ModelRequest) -> ModelResponse:
        """同步调用"""
        pass
    
    @abstractmethod  
    async def chat_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """流式调用"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        pass
```

### 2. 供应商适配器实现

每个供应商实现统一接口，处理各自的协议差异：

```python
import httpx
import json
import time

class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60)
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def supported_models(self) -> list[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-preview"]
    
    async def chat(self, request: ModelRequest) -> ModelResponse:
        model = request.model_hint or "gpt-4o"
        start = time.monotonic()
        
        resp = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "response_format": request.response_format,
                "tools": request.tools,
            }
        )
        resp.raise_for_status()
        data = resp.json()
        
        return ModelResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            provider=self.provider_name,
            usage=data["usage"],
            latency_ms=(time.monotonic() - start) * 1000,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            tool_calls=data["choices"][0]["message"].get("tool_calls"),
        )
    
    async def chat_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        model = request.model_hint or "gpt-4o"
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
            }
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    yield StreamChunk(
                        delta=delta,
                        model=chunk["model"],
                        finish_reason=chunk["choices"][0].get("finish_reason"),
                    )
    
    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return resp.status_code == 200
        except Exception:
            return False


class AnthropicProvider(ModelProvider):
    """Anthropic Claude 适配器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            timeout=60,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        )
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def supported_models(self) -> list[str]:
        return ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-opus-4-20250514"]
    
    async def chat(self, request: ModelRequest) -> ModelResponse:
        # Claude 的 API 格式不同：system 消息需要单独提取
        system_msg = ""
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                messages.append(msg)
        
        start = time.monotonic()
        resp = await self.client.post(
            "/v1/messages",
            json={
                "model": request.model_hint or "claude-sonnet-4-20250514",
                "system": system_msg,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
        )
        resp.raise_for_status()
        data = resp.json()
        
        return ModelResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            provider=self.provider_name,
            usage={
                "prompt_tokens": data["usage"]["input_tokens"],
                "completion_tokens": data["usage"]["output_tokens"],
            },
            latency_ms=(time.monotonic() - start) * 1000,
            finish_reason=data["stop_reason"],
        )
    
    async def chat_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        system_msg = ""
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                messages.append(msg)
        
        async with self.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": request.model_hint or "claude-sonnet-4-20250514",
                "system": system_msg,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "stream": True,
            }
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                if chunk["type"] == "content_block_delta":
                    yield StreamChunk(
                        delta=chunk["delta"]["text"],
                        model=chunk.get("model", ""),
                    )
                elif chunk["type"] == "message_stop":
                    break
    
    async def health_check(self) -> bool:
        try:
            resp = await self.client.post(
                "/v1/messages",
                json={"model": "claude-3-5-haiku-20241022", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
            )
            return resp.status_code == 200
        except Exception:
            return False
```

## 智能路由引擎

路由引擎是模型抽象层的"大脑"，负责决定每次请求应该使用哪个模型：

### 1. 路由策略

```python
from enum import Enum
import random

class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"          # 轮询
    COST_OPTIMIZED = "cost_optimized"     # 成本优先
    LATENCY_OPTIMIZED = "latency_optimized"  # 延迟优先
    QUALITY_FIRST = "quality_first"       # 质量优先
    CAPABILITY_BASED = "capability_based" # 能力匹配
    WEIGHTED_RANDOM = "weighted_random"   # 加权随机

@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    provider: str
    cost_per_1k_input: float   # 美元/千token
    cost_per_1k_output: float
    avg_latency_ms: float
    quality_score: float       # 0-100 质量评分
    capabilities: list[str]    # ["code", "reasoning", "multilingual", "vision"]
    max_context: int
    rate_limit: int            # RPM
    priority: int = 1          # 优先级（故障转移用）

class ModelRouter:
    """智能模型路由器"""
    
    def __init__(self):
        self.models: dict[str, ModelConfig] = {}
        self.providers: dict[str, ModelProvider] = {}
        self._health_status: dict[str, bool] = {}
        self._round_robin_index = 0
    
    def register_model(self, config: ModelConfig, provider: ModelProvider):
        self.models[config.model_id] = config
        self.providers[config.provider] = provider
        self._health_status[config.model_id] = True
    
    async def route(
        self,
        request: ModelRequest,
        strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMIZED,
    ) -> tuple[str, ModelProvider]:
        """路由请求到最佳模型"""
        
        # 第一步：过滤可用模型（健康检查 + 上下文长度）
        available = [
            m for m in self.models.values()
            if self._health_status.get(m.model_id, False)
            and m.max_context >= self._estimate_token_count(request.messages)
        ]
        
        if not available:
            raise RuntimeError("No healthy models available")
        
        # 第二步：根据策略选择模型
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            # 按成本排序（输入+输出综合考虑）
            available.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
            selected = available[0]
        
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            available.sort(key=lambda m: m.avg_latency_ms)
            selected = available[0]
        
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            available.sort(key=lambda m: m.quality_score, reverse=True)
            selected = available[0]
        
        elif strategy == RoutingStrategy.CAPABILITY_BASED:
            # 根据请求内容匹配能力
            required_caps = self._detect_capabilities(request)
            scored = []
            for m in available:
                match_count = len(set(required_caps) & set(m.capabilities))
                score = match_count * m.quality_score / max(m.cost_per_1k_input, 0.01)
                scored.append((score, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = scored[0][1]
        
        elif strategy == RoutingStrategy.WEIGHTED_RANDOM:
            weights = [m.quality_score / max(m.cost_per_1k_input, 0.01) for m in available]
            total = sum(weights)
            r = random.uniform(0, total)
            cumulative = 0
            selected = available[0]
            for m, w in zip(available, weights):
                cumulative += w
                if cumulative >= r:
                    selected = m
                    break
        
        else:  # ROUND_ROBIN
            selected = available[self._round_robin_index % len(available)]
            self._round_robin_index += 1
        
        return selected.model_id, self.providers[selected.provider]
    
    def _estimate_token_count(self, messages: list[dict]) -> int:
        """粗略估算token数"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return int(total_chars / 3)  # 中文约1.5token/字，英文约0.75token/word
    
    def _detect_capabilities(self, request: ModelRequest) -> list[str]:
        """检测请求需要的能力"""
        caps = []
        text = " ".join(m.get("content", "") for m in request.messages)
        
        code_keywords = ["代码", "code", "函数", "function", "bug", "debug", "编程"]
        if any(kw in text.lower() for kw in code_keywords):
            caps.append("code")
        
        reasoning_keywords = ["分析", "推理", "为什么", "explain", "reason", "prove"]
        if any(kw in text.lower() for kw in reasoning_keywords):
            caps.append("reasoning")
        
        if request.tools:
            caps.append("function_calling")
        
        if request.response_format:
            caps.append("structured_output")
        
        return caps or ["general"]
    
    def update_health(self, model_id: str, healthy: bool):
        self._health_status[model_id] = healthy
```

### 2. 路由策略选择矩阵

| 场景 | 推荐策略 | 说明 |
|------|---------|------|
| 日常对话/FAQ | COST_OPTIMIZED | 用小模型即可，控制成本 |
| 代码生成/审查 | CAPABILITY_BASED | 需要代码能力，匹配专用模型 |
| 复杂推理/分析 | QUALITY_FIRST | 质量优先，用最强模型 |
| 高并发/低延迟 | LATENCY_OPTIMIZED | 首token延迟敏感场景 |
| A/B测试 | WEIGHTED_RANDOM | 按权重分配流量对比效果 |
| 多模型容灾 | 轮询+故障转移 | 保证高可用 |

## 故障转移链

生产环境中，模型API不可用是常态。故障转移链确保单个模型故障不影响整体服务：

```python
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class CircuitBreaker:
    """熔断器"""
    failure_threshold: int = 3
    recovery_timeout: timedelta = timedelta(minutes=5)
    _failure_count: int = 0
    _last_failure_time: Optional[datetime] = None
    _state: str = "closed"  # closed, open, half-open
    
    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
    
    def record_success(self):
        self._failure_count = 0
        self._state = "closed"
    
    def can_execute(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open" and self._last_failure_time:
            if datetime.now() - self._last_failure_time > self.recovery_timeout:
                self._state = "half-open"
                return True
            return False
        return True  # half-open allows one attempt


class FaultTolerantGateway:
    """容错模型网关"""
    
    def __init__(self, router: ModelRouter):
        self.router = router
        self.breakers: dict[str, CircuitBreaker] = {}
    
    async def execute_with_fallback(
        self,
        request: ModelRequest,
        strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMIZED,
        max_retries: int = 3,
    ) -> ModelResponse:
        """带故障转移的模型调用"""
        
        # 获取路由建议的模型列表（按优先级）
        model_id, provider = await self.router.route(request, strategy)
        
        # 构建故障转移链：当前模型 + 备选模型
        fallback_chain = self._build_fallback_chain(model_id, request)
        
        last_error = None
        for model_name in fallback_chain:
            breaker = self.breakers.setdefault(model_name, CircuitBreaker())
            
            if not breaker.can_execute():
                continue
            
            try:
                # 获取对应的 provider
                config = self.router.models.get(model_name)
                if not config:
                    continue
                p = self.router.providers.get(config.provider)
                if not p:
                    continue
                
                # 创建新请求使用指定模型
                fallback_request = ModelRequest(
                    messages=request.messages,
                    model_hint=model_name,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=False,
                )
                
                response = await p.chat(fallback_request)
                response.model = model_name  # 确保记录实际使用的模型
                breaker.record_success()
                return response
                
            except Exception as e:
                last_error = e
                breaker.record_failure()
                self.router.update_health(model_name, False)
                continue
        
        raise RuntimeError(f"All models failed. Last error: {last_error}")
    
    def _build_fallback_chain(self, primary: str, request: ModelRequest) -> list[str]:
        """构建故障转移链"""
        chain = [primary]
        
        # 按质量评分降序添加备选
        candidates = sorted(
            self.router.models.values(),
            key=lambda m: m.quality_score,
            reverse=True
        )
        for c in candidates:
            if c.model_id != primary:
                chain.append(c.model_id)
        
        return chain


async def execute_with_retry(
    gateway: FaultTolerantGateway,
    request: ModelRequest,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> ModelResponse:
    """指数退避重试 + 故障转移"""
    
    for attempt in range(max_retries):
        try:
            return await gateway.execute_with_fallback(request)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_base * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
```

## 成本追踪与优化

### 1. 成本追踪器

```python
from collections import defaultdict
from dataclasses import dataclass
import json

@dataclass
class CostRecord:
    timestamp: float
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    route_reason: str  # 为什么选择了这个模型

class CostTracker:
    """成本追踪器"""
    
    def __init__(self):
        self.records: list[CostRecord] = []
        self.daily_costs: dict[str, float] = defaultdict(float)
        self.budget_limit: float = 100.0  # 每日预算上限（美元）
    
    def record(self, response: ModelResponse, config: ModelConfig, reason: str = ""):
        cost = (
            response.usage.get("prompt_tokens", 0) / 1000 * config.cost_per_1k_input +
            response.usage.get("completion_tokens", 0) / 1000 * config.cost_per_1k_output
        )
        
        record = CostRecord(
            timestamp=time.time(),
            model=response.model,
            provider=response.provider,
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            cost_usd=cost,
            latency_ms=response.latency_ms,
            route_reason=reason,
        )
        self.records.append(record)
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_costs[today] += cost
    
    def get_daily_report(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = [r for r in self.records 
                        if datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d") == today]
        
        model_costs = defaultdict(float)
        model_tokens = defaultdict(lambda: {"input": 0, "output": 0})
        
        for r in today_records:
            model_costs[r.model] += r.cost_usd
            model_tokens[r.model]["input"] += r.prompt_tokens
            model_tokens[r.model]["output"] += r.completion_tokens
        
        return {
            "date": today,
            "total_cost_usd": self.daily_costs[today],
            "budget_remaining": self.budget_limit - self.daily_costs[today],
            "request_count": len(today_records),
            "by_model": dict(model_costs),
            "tokens_by_model": dict(model_tokens),
            "avg_latency_ms": (
                sum(r.latency_ms for r in today_records) / len(today_records)
                if today_records else 0
            ),
        }
    
    def check_budget(self) -> bool:
        """检查是否超预算"""
        return self.daily_costs[datetime.now().strftime("%Y-%m-%d")] < self.budget_limit
```

### 2. 自动降级策略（预算触发）

```python
class BudgetAwareRouter:
    """预算感知路由器"""
    
    def __init__(self, router: ModelRouter, cost_tracker: CostTracker):
        self.router = router
        self.tracker = cost_tracker
        
        # 预算阈值配置
        self.thresholds = {
            0.8: RoutingStrategy.COST_OPTIMIZED,   # 用掉80%预算切成本优先
            0.95: RoutingStrategy.COST_OPTIMIZED,  # 用掉95%只允许最便宜模型
        }
    
    async def route_with_budget(self, request: ModelRequest) -> tuple[str, ModelProvider]:
        budget_usage = self.tracker.daily_costs.get(
            datetime.now().strftime("%Y-%m-%d"), 0
        ) / self.tracker.budget_limit
        
        # 根据预算使用率自动调整策略
        strategy = RoutingStrategy.QUALITY_FIRST
        for threshold in sorted(self.thresholds.keys()):
            if budget_usage >= threshold:
                strategy = self.thresholds[threshold]
        
        return await self.router.route(request, strategy)
```

## 流式响应统一处理

不同供应商的SSE格式存在差异，抽象层需要统一处理：

```python
async def unified_stream(
    gateway: FaultTolerantGateway,
    request: ModelRequest,
) -> AsyncIterator[StreamChunk]:
    """统一的流式响应处理"""
    
    model_id, provider = await gateway.router.route(request)
    config = gateway.router.models.get(model_id)
    
    if not config:
        raise RuntimeError(f"Model {model_id} not found")
    
    p = gateway.router.providers.get(config.provider)
    if not p:
        raise RuntimeError(f"Provider {config.provider} not found")
    
    stream_request = ModelRequest(
        messages=request.messages,
        model_hint=model_id,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=True,
    )
    
    try:
        async for chunk in p.chat_stream(stream_request):
            yield chunk
    except Exception:
        # 流式失败时降级到非流式
        response = await gateway.execute_with_fallback(request)
        yield StreamChunk(delta=response.content, model=response.model)
```

前端对接示例（Next.js API Route）：

```typescript
// app/api/chat/route.ts
export async function POST(req: Request) {
  const { messages, model } = await req.json();
  
  const response = await fetch(`${GATEWAY_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      model_hint: model,
      stream: true,
    }),
  });
  
  // 统一的 SSE 流转发
  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
```

## OpenAI兼容层

为了无缝对接现有生态（LangChain、LlamaIndex等），提供OpenAI兼容的API接口：

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="LLM Gateway")

# 统一的模型网关实例
gateway = FaultTolerantGateway(ModelRouter())
cost_tracker = CostTracker()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 兼容的 Chat Completions 接口"""
    body = await request.json()
    
    # 转换为内部请求格式
    internal_request = ModelRequest(
        messages=body["messages"],
        model_hint=body.get("model", ""),
        temperature=body.get("temperature", 0.7),
        max_tokens=body.get("max_tokens", 4096),
        stream=body.get("stream", False),
        response_format=body.get("response_format"),
        tools=body.get("tools"),
    )
    
    if body.get("stream"):
        # 流式响应
        async def event_generator():
            async for chunk in unified_stream(gateway, internal_request):
                data = {
                    "id": f"chatcmpl-{random.randint(1000, 9999)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": chunk.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.delta},
                        "finish_reason": chunk.finish_reason,
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    # 非流式响应
    response = await gateway.execute_with_fallback(internal_request)
    
    return {
        "id": f"chatcmpl-{random.randint(1000, 9999)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.content},
            "finish_reason": response.finish_reason,
        }],
        "usage": response.usage,
    }
```

## 生产部署建议

### 配置管理

```yaml
# gateway-config.yaml
models:
  - id: gpt-4o
    provider: openai
    cost_input: 0.0025    # $/1k tokens
    cost_output: 0.01     # $/1k tokens
    max_context: 128000
    capabilities: [general, code, reasoning, vision, function_calling]
    priority: 1
    
  - id: claude-sonnet-4-20250514
    provider: anthropic
    cost_input: 0.003
    cost_output: 0.015
    max_context: 200000
    capabilities: [general, code, reasoning, vision, function_calling]
    priority: 1
    
  - id: deepseek-chat
    provider: deepseek
    cost_input: 0.00014
    cost_output: 0.00028
    max_context: 64000
    capabilities: [general, code, reasoning]
    priority: 2
    
  - id: qwen-max
    provider: dashscope
    cost_input: 0.0016
    cost_output: 0.0064
    max_context: 32000
    capabilities: [general, code, multilingual]
    priority: 2

routing:
  default_strategy: cost_optimized
  fallback_enabled: true
  health_check_interval: 60  # seconds
  
budget:
  daily_limit_usd: 100
  alert_threshold: 0.8

circuit_breaker:
  failure_threshold: 3
  recovery_timeout: 300  # seconds
```

### 监控指标

部署后需要关注的核心指标：

| 指标 | 说明 | 告警阈值 |
|------|------|---------|
| `model_request_total` | 总请求数 | — |
| `model_request_duration_seconds` | 请求延迟 | P99 > 10s |
| `model_request_errors_total` | 错误数 | 5分钟内 > 10 |
| `model_cost_usd_daily` | 每日成本 | > 80% 预算 |
| `model_fallback_total` | 故障转移次数 | 5分钟内 > 5 |
| `model_circuit_open_total` | 熔断器打开次数 | > 0 |

## 总结

模型抽象层不是"过度设计"，而是LLM应用进入生产环境的**必要基础设施**。它的核心价值在于：

1. **解耦业务逻辑与模型选择**：业务代码不关心底层用的是GPT还是Claude
2. **弹性与容错**：单模型故障不影响整体服务
3. **成本可控**：自动根据预算和场景选择最优模型
4. **实验能力**：为A/B测试、模型对比提供基础设施
5. **生态兼容**：OpenAI兼容层让现有工具链无缝接入

从硬编码调用到智能模型网关，这个演进过程本质上是LLM应用从"能用"到"好用"再到"可靠"的必经之路。在实际落地时，建议**从简开始，按需迭代**——先实现统一接口和基本路由，再逐步添加故障转移、成本追踪等高级能力。
