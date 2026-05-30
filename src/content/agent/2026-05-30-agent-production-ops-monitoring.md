---
title: "Agent生产化：监控/日志/限流/降级全链路实践"
description: "深入探讨AI Agent系统在生产环境中的全链路运维实践，涵盖监控指标体系、链路追踪、结构化日志、限流策略、降级方案、成本控制与告警设计，附完整代码示例。"
date: 2026-05-30
author: RiceBall-15
category: agent
subCategory: agent-ops
tags: [生产化, 监控, 限流, 降级, 成本控制]
draft: false
---

# Agent生产化：监控/日志/限流/降级全链路实践

## 引言

当一个AI Agent系统从实验室走向生产环境，你面对的不再只是"能不能跑"的问题，而是"能不能稳定、可控、可追踪地跑"。LLM调用的不确定性、多轮对话的状态管理、Agent工具调用的级联失败——这些在Demo阶段被忽略的问题，在生产环境中会被无限放大。

本文将从监控、日志、限流、降级、成本控制和告警六个维度，系统梳理Agent系统生产化落地的全链路实践方案，并附上可直接复用的代码实现。

---

## 一、Agent监控指标体系：延迟/吞吐/成功率/成本

### 1.1 核心四象限指标

Agent系统的监控需要围绕四个核心维度构建：

| 指标类别 | 具体指标 | 采集频率 | 告警阈值 |
|---------|---------|---------|---------|
| **延迟** | 首Token延迟(TTFT)、端到端延迟、P95/P99延迟 | 实时 | TTFT > 3s, P99 > 30s |
| **吞吐** | QPS、并发Agent数、工具调用吞吐 | 10s聚合 | QPS骤降50% |
| **成功率** | LLM调用成功率、Agent任务完成率、工具调用成功率 | 1min聚合 | < 95% |
| **成本** | Token消耗、每任务成本、模型调用费用 | 5min聚合 | 超预算200% |

### 1.2 监控指标采集实现

```python
import time
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
from functools import wraps
from dataclasses import dataclass, field
from typing import Optional
import json

# ---- Prometheus 指标定义 ----

# Agent延迟指标
agent_latency = Histogram(
    'agent_request_duration_seconds',
    'Agent request end-to-end latency',
    ['agent_type', 'task_type'],
    buckets=[0.5, 1, 2, 5, 10, 15, 20, 30, 60]
)

# 首Token延迟
ttft_latency = Histogram(
    'agent_ttft_seconds',
    'Time to first token latency',
    ['model', 'agent_type'],
    buckets=[0.1, 0.5, 1, 2, 3, 5, 10]
)

# 吞吐量
agent_requests_total = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['agent_type', 'status']
)

# 并发数
agent_concurrent = Gauge(
    'agent_concurrent_requests',
    'Currently processing agent requests',
    ['agent_type']
)

# Token消耗
token_usage = Counter(
    'agent_token_usage_total',
    'Total token consumption',
    ['model', 'token_type', 'agent_type']  # token_type: input/output
)

# 工具调用指标
tool_call_latency = Histogram(
    'agent_tool_call_duration_seconds',
    'Tool call latency',
    ['tool_name', 'status'],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

tool_call_total = Counter(
    'agent_tool_call_total',
    'Total tool calls',
    ['tool_name', 'status']
)

# 成本指标
agent_cost_dollars = Counter(
    'agent_cost_dollars_total',
    'Total cost in USD',
    ['model', 'agent_type']
)


# ---- 装饰器：自动埋点 ----

def monitor_agent(agent_type: str, task_type: str = "default"):
    """Agent监控装饰器，自动采集延迟、吞吐、成功率"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            agent_concurrent.labels(agent_type=agent_type).inc()
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                agent_requests_total.labels(
                    agent_type=agent_type, status="success"
                ).inc()
                return result
            except Exception as e:
                agent_requests_total.labels(
                    agent_type=agent_type, status="error"
                ).inc()
                raise
            finally:
                duration = time.time() - start
                agent_latency.labels(
                    agent_type=agent_type, task_type=task_type
                ).observe(duration)
                agent_concurrent.labels(agent_type=agent_type).dec()
        return wrapper
    return decorator


@dataclass
class TokenUsageTracker:
    """Token使用量追踪器"""
    model: str
    agent_type: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    # 价格表（USD per 1K tokens）
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-sonnet-4": {"input": 0.003, "output": 0.015},
        "deepseek-v3": {"input": 0.00027, "output": 0.0011},
    }

    def record(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        pricing = self.PRICING.get(self.model, {"input": 0.003, "output": 0.015})
        cost = (input_tokens * pricing["input"] + 
                output_tokens * pricing["output"]) / 1000
        self.cost += cost

        # 推送到Prometheus
        token_usage.labels(
            model=self.model, token_type="input", agent_type=self.agent_type
        ).inc(input_tokens)
        token_usage.labels(
            model=self.model, token_type="output", agent_type=self.agent_type
        ).inc(output_tokens)
        agent_cost_dollars.labels(
            model=self.model, agent_type=self.agent_type
        ).inc(cost)

        return cost
```

---

## 二、链路追踪：LangSmith / Phoenix / OpenTelemetry

### 2.1 追踪方案对比

| 工具 | 定位 | 优势 | 适用场景 |
|-----|------|------|---------|
| **LangSmith** | LangChain官方追踪 | 与LangChain深度集成，调试友好 | LangChain/LangGraph项目 |
| **Phoenix (Arize)** | 开源LLM可观测性 | 本地部署，隐私友好，支持评估 | 需要私有化部署的团队 |
| **OpenTelemetry** | 通用分布式追踪 | 标准化，生态丰富，厂商无关 | 微服务架构、多系统集成 |

### 2.2 OpenTelemetry全链路追踪实现

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
import uuid

# ---- 初始化Tracer ----

resource = Resource.create({
    "service.name": "agent-service",
    "service.version": "1.0.0",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="otel-collector:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agent-service", "1.0.0")


class AgentTracer:
    """Agent链路追踪封装"""

    def __init__(self):
        self.tracer = trace.get_tracer("agent-service")

    def trace_agent_run(self, agent_name: str, task: str):
        """追踪Agent完整运行过程"""
        span = self.tracer.start_span(
            "agent.run",
            attributes={
                "agent.name": agent_name,
                "agent.task": task[:500],  # 截断过长的任务
                "agent.request_id": str(uuid.uuid4()),
            }
        )
        return span

    def trace_llm_call(self, model: str, prompt_tokens: int):
        """追踪单次LLM调用"""
        span = self.tracer.start_span(
            "llm.completion",
            attributes={
                "llm.model": model,
                "llm.system": "openai",
                "llm.token_count.prompt": prompt_tokens,
            }
        )
        return span

    def trace_tool_call(self, tool_name: str, tool_input: str):
        """追踪工具调用"""
        span = self.tracer.start_span(
            "tool.call",
            attributes={
                "tool.name": tool_name,
                "tool.input": str(tool_input)[:1000],
            }
        )
        return span

    def trace_react_step(self, step: int, thought: str, action: str):
        """追踪ReAct循环的每一步"""
        span = self.tracer.start_span(
            "agent.react_step",
            attributes={
                "react.step": step,
                "react.thought": thought[:500],
                "react.action": action,
            }
        )
        return span


# ---- 使用示例 ----

async def run_agent_with_tracing(agent, task: str):
    agent_tracer = AgentTracer()

    with agent_tracer.trace_agent_run(agent.name, task) as root_span:
        try:
            # ReAct循环追踪
            for step in range(agent.max_steps):
                with agent_tracer.trace_react_step(step, "", "") as step_span:
                    # LLM调用追踪
                    with agent_tracer.trace_llm_call(agent.model, 0) as llm_span:
                        response = await agent.llm.complete(
                            messages=agent.build_messages(task)
                        )
                        llm_span.set_attribute(
                            "llm.token_count.completion", response.usage.completion_tokens
                        )

                    # 工具调用追踪
                    if response.tool_calls:
                        for tc in response.tool_calls:
                            with agent_tracer.trace_tool_call(tc.name, str(tc.args)) as tool_span:
                                result = await agent.execute_tool(tc.name, tc.args)
                                tool_span.set_attribute("tool.output", str(result)[:500])

            root_span.set_status(Status(StatusCode.OK))
        except Exception as e:
            root_span.set_status(Status(StatusCode.ERROR), str(e))
            root_span.record_exception(e)
            raise
```

### 2.3 LangSmith集成

```python
# LangSmith追踪（与LangChain/LangGraph配合）
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-production"

from langsmith import traceable

@traceable(run_type="chain", name="agent-main")
async def agent_main(query: str, config: dict):
    """带LangSmith追踪的Agent主函数"""
    # LLM调用自动追踪
    response = await agent.llm.ainvoke(
        build_messages(query),
        config={"callbacks": []}  # LangSmith自动注入
    )

    # 工具调用自动追踪
    if response.tool_calls:
        for tc in response.tool_calls:
            result = await agent.tools[tc["name"]].ainvoke(tc["args"])

    return response
```

---

## 三、日志设计：结构化日志 + LLM调用记录

### 3.1 结构化日志框架

```python
import logging
import json
import uuid
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any, Optional
from dataclasses import dataclass, asdict

# ---- 请求上下文 ----

request_context: ContextVar[dict] = ContextVar('request_context', default={})

class AgentContextFilter(logging.Filter):
    """自动注入请求上下文到日志"""
    def filter(self, record):
        ctx = request_context.get({})
        record.request_id = ctx.get("request_id", "")
        record.user_id = ctx.get("user_id", "")
        record.agent_name = ctx.get("agent_name", "")
        return True


# ---- JSON结构化日志格式 ----

class JSONFormatter(logging.Formatter):
    """生产级JSON日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", ""),
            "user_id": getattr(record, "user_id", ""),
            "agent_name": getattr(record, "agent_name", ""),
        }

        # 附加LLM调用信息
        if hasattr(record, "llm_data"):
            log_entry["llm"] = record.llm_data

        # 附加异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logging():
    """初始化生产环境日志"""
    logger = logging.getLogger("agent-service")
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    handler.addFilter(AgentContextFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()
```

### 3.2 LLM调用日志记录

```python
@dataclass
class LLMCallLog:
    """LLM调用日志数据结构"""
    request_id: str
    user_id: str
    model: str
    prompt_messages: list[dict]
    response_content: str
    tool_calls: list[dict]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str  # success / error / timeout
    error_message: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    # 质量指标
    finish_reason: str = ""
    refusal: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # 脱敏处理：移除完整prompt中的敏感信息
        d["prompt_messages"] = [
            {"role": m["role"], "content_length": len(m.get("content", ""))}
            for m in self.prompt_messages
        ]
        return d


class LLMLogger:
    """LLM调用专用日志器"""

    def __init__(self):
        self.logger = logging.getLogger("agent-service.llm")

    def log_call(self, log: LLMCallLog):
        """记录LLM调用日志"""
        extra = {
            "llm_data": {
                "model": log.model,
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "total_tokens": log.input_tokens + log.output_tokens,
                "latency_ms": log.latency_ms,
                "status": log.status,
                "finish_reason": log.finish_reason,
                "tool_calls_count": len(log.tool_calls),
                "estimated_cost_usd": self._estimate_cost(
                    log.model, log.input_tokens, log.output_tokens
                ),
            }
        }

        if log.status == "error":
            self.logger.error(
                f"LLM调用失败: model={log.model} error={log.error_message}",
                extra=extra
            )
        elif log.latency_ms > 10000:
            self.logger.warning(
                f"LLM调用慢: model={log.model} latency={log.latency_ms}ms",
                extra=extra
            )
        else:
            self.logger.info(
                f"LLM调用完成: model={log.model} tokens={log.input_tokens+log.output_tokens}",
                extra=extra
            )

        # 异步写入到日志存储（如ClickHouse/ElasticSearch）
        self._async_persist(log)

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "claude-sonnet-4": (3.0, 15.0),
        }
        p = pricing.get(model, (3.0, 15.0))
        return (input_tokens * p[0] + output_tokens * p[1]) / 1_000_000

    def _async_persist(self, log: LLMCallLog):
        """异步持久化到分析存储"""
        # 实际生产中使用Celery/Dramatiq异步写入ClickHouse
        pass
```

---

## 四、限流策略：令牌桶 + 滑动窗口 + 用户级限流

### 4.1 令牌桶限流器

```python
import asyncio
import time
from collections import defaultdict
from typing import Optional
import redis.asyncio as redis


class TokenBucketRateLimiter:
    """基于Redis的分布式令牌桶限流器"""

    def __init__(
        self,
        redis_client: redis.Redis,
        rate: float,          # 每秒产生的令牌数
        capacity: int,         # 桶容量
        burst_rate: float = 0, # 突发流量额外速率
    ):
        self.redis = redis_client
        self.rate = rate
        self.capacity = capacity
        self.burst_rate = burst_rate

        # Lua脚本：原子性令牌桶操作
        self._lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local requested = tonumber(ARGV[4])
        local ttl = tonumber(ARGV[5])

        local bucket = redis.call('hmget', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now

        -- 补充令牌
        local elapsed = math.max(0, now - last_refill)
        local new_tokens = math.min(capacity, tokens + elapsed * rate)

        local allowed = 0
        local remaining = new_tokens
        if new_tokens >= requested then
            new_tokens = new_tokens - requested
            allowed = 1
            remaining = new_tokens
        end

        redis.call('hmset', key, 'tokens', new_tokens, 'last_refill', now)
        redis.call('expire', key, ttl)

        return {allowed, remaining, math.ceil((requested - new_tokens) / rate * 1000)}
        """

    async def allow(
        self,
        key: str,
        tokens: int = 1,
        ttl: int = 3600
    ) -> tuple[bool, int, int]:
        """
        尝试消费令牌
        返回: (是否允许, 剩余令牌数, 需等待毫秒数)
        """
        now = time.time()
        result = await self.redis.eval(
            self._lua_script, 1, key,
            self.rate, self.capacity, now, tokens, ttl
        )
        return bool(result[0]), int(result[1]), int(result[2])


class SlidingWindowRateLimiter:
    """滑动窗口限流器（精确计数）"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

        self._lua_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])  -- 窗口大小（秒）
        local limit = tonumber(ARGV[2])    -- 窗口内允许的最大请求数
        local now = tonumber(ARGV[3])
        local ttl = tonumber(ARGV[4])

        -- 移除窗口外的请求
        redis.call('zremrangebyscore', key, 0, now - window)

        -- 当前窗口内请求数
        local count = redis.call('zcard', key)

        local allowed = 0
        if count < limit then
            allowed = 1
            redis.call('zadd', key, now, now .. '-' .. math.random(1000000))
            redis.call('expire', key, ttl)
        end

        return {allowed, count, limit - count}
        """

    async def allow(
        self,
        key: str,
        window_seconds: int = 60,
        limit: int = 100,
    ) -> tuple[bool, int, int]:
        """滑动窗口检查"""
        now = time.time()
        result = await self.redis.eval(
            self._lua_script, 1, key,
            window_seconds, limit, now, window_seconds * 2
        )
        return bool(result[0]), int(result[1]), int(result[2])
```

### 4.2 多维度限流策略

```python
@dataclass
class RateLimitConfig:
    """限流配置"""
    # 全局维度
    global_qps: float = 100.0         # 全局QPS限制
    global_burst: int = 200           # 全局突发容量

    # 用户维度
    user_rpm: int = 60                # 每用户每分钟请求
    user_token_per_min: int = 100000  # 每用户每分钟Token
    user_daily_cost_usd: float = 10.0 # 每用户每日成本上限

    # Agent维度
    agent_concurrent: int = 10        # 每Agent最大并发
    agent_tool_rpm: int = 30          # 工具调用RPM

    # 模型维度
    model_rpm: int = 500              # 每模型RPM
    model_tpm: int = 200000           # 每模型TPM


class MultiDimensionRateLimiter:
    """多维度限流器：全局 + 用户 + Agent + 模型"""

    def __init__(self, redis_client: redis.Redis, config: RateLimitConfig):
        self.redis = redis_client
        self.config = config
        self.bucket_limiter = TokenBucketRateLimiter(
            redis_client,
            rate=config.global_qps,
            capacity=config.global_burst,
        )
        self.window_limiter = SlidingWindowRateLimiter(redis_client)

    async def check_all(
        self,
        user_id: str,
        agent_type: str,
        model: str,
        estimated_tokens: int = 0,
    ) -> tuple[bool, Optional[str]]:
        """
        全维度限流检查
        返回: (是否允许, 拒绝原因)
        """

        # 1. 全局限流
        allowed, remaining, wait_ms = await self.bucket_limiter.allow(
            "rate:global", tokens=1
        )
        if not allowed:
            return False, f"全局流量超限，请等待{wait_ms}ms"

        # 2. 用户请求频率
        allowed, count, _ = await self.window_limiter.allow(
            key=f"rate:user:{user_id}:rpm",
            window_seconds=60,
            limit=self.config.user_rpm,
        )
        if not allowed:
            return False, f"用户请求频率超限 ({count}/{self.config.user_rpm} per min)"

        # 3. 用户Token消耗
        if estimated_tokens > 0:
            user_token_key = f"rate:user:{user_id}:tpm"
            allowed, count, _ = await self.window_limiter.allow(
                key=user_token_key,
                window_seconds=60,
                limit=self.config.user_token_per_min,
            )
            if not allowed:
                return False, f"用户Token消耗超限"

        # 4. 模型RPM
        allowed, count, _ = await self.window_limiter.allow(
            key=f"rate:model:{model}:rpm",
            window_seconds=60,
            limit=self.config.model_rpm,
        )
        if not allowed:
            return False, f"模型{model}调用频率超限"

        return True, None
```

---

## 五、降级方案：缓存兜底 + 小模型替代

### 5.1 语义缓存兜底

```python
import hashlib
import json
from typing import Optional
import numpy as np


class SemanticCacheFallback:
    """语义缓存降级层：当主LLM不可用时，从缓存中返回相似结果"""

    def __init__(self, redis_client, embedding_client, similarity_threshold=0.92):
        self.redis = redis_client
        self.embedding = embedding_client
        self.threshold = similarity_threshold

    async def get_or_none(self, query: str, context: str = "") -> Optional[dict]:
        """查找语义相似的缓存结果"""
        cache_key = self._make_key(query, context)

        # 1. 精确匹配
        exact = await self.redis.get(f"cache:exact:{cache_key}")
        if exact:
            return json.loads(exact)

        # 2. 语义匹配
        query_embedding = await self.embedding.embed(query)
        cached_keys = await self.redis.smembers("cache:embeddings:keys")

        best_score = 0.0
        best_match = None

        for key in cached_keys[:1000]:  # 限制扫描范围
            cached_data = await self.redis.hgetall(f"cache:entry:{key}")
            if not cached_data:
                continue

            cached_embedding = json.loads(cached_data.get("embedding", "[]"))
            score = self._cosine_similarity(query_embedding, cached_embedding)

            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = json.loads(cached_data["response"])

        if best_match:
            best_match["_cache"] = {"hit": True, "score": best_score}
            return best_match

        return None

    async def store(self, query: str, context: str, response: dict):
        """存储查询结果到缓存"""
        cache_key = self._make_key(query, context)
        embedding = await self.embedding.embed(query)

        await self.redis.setex(
            f"cache:exact:{cache_key}",
            86400,  # 24小时过期
            json.dumps(response, ensure_ascii=False)
        )

        await self.redis.hset(
            f"cache:entry:{cache_key}",
            mapping={
                "response": json.dumps(response, ensure_ascii=False),
                "embedding": json.dumps(embedding),
                "query": query,
                "timestamp": str(time.time()),
            }
        )
        await self.redis.sadd("cache:embeddings:keys", cache_key)

    def _make_key(self, query: str, context: str) -> str:
        return hashlib.sha256(f"{context}:{query}".encode()).hexdigest()[:16]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### 5.2 模型降级策略

```python
from enum import Enum
from dataclasses import dataclass


class ModelTier(Enum):
    PREMIUM = "premium"    # GPT-4o, Claude Opus
    STANDARD = "standard"  # GPT-4o-mini, Claude Sonnet
    ECONOMY = "economy"    # DeepSeek-V3, Qwen-Turbo


@dataclass
class ModelFallbackChain:
    """模型降级链：按优先级依次尝试"""
    primary: str
    fallbacks: list[str]
    cache_fallback: bool = True


# 预定义降级链
FALLBACK_CHAINS = {
    "complex_reasoning": ModelFallbackChain(
        primary="gpt-4o",
        fallbacks=["claude-sonnet-4", "deepseek-v3", "qwen-max"],
        cache_fallback=True,
    ),
    "code_generation": ModelFallbackChain(
        primary="claude-sonnet-4",
        fallbacks=["gpt-4o", "deepseek-coder-v3", "qwen-coder-turbo"],
        cache_fallback=True,
    ),
    "simple_qa": ModelFallbackChain(
        primary="gpt-4o-mini",
        fallbacks=["deepseek-v3", "qwen-turbo"],
        cache_fallback=True,
    ),
}


class ResilientLLMClient:
    """带降级能力的LLM客户端"""

    def __init__(self, llm_clients: dict, cache: SemanticCacheFallback):
        self.clients = llm_clients   # {model_name: client}
        self.cache = cache
        self.logger = logging.getLogger("agent-service.llm.resilient")

    async def complete_with_fallback(
        self,
        messages: list[dict],
        task_category: str = "complex_reasoning",
        **kwargs,
    ) -> dict:
        """带完整降级链的LLM调用"""

        chain = FALLBACK_CHAINS.get(task_category, FALLBACK_CHAINS["simple_qa"])
        all_models = [chain.primary] + chain.fallbacks
        last_error = None

        for model in all_models:
            try:
                client = self.clients.get(model)
                if not client:
                    continue

                response = await client.complete(messages=messages, model=model, **kwargs)
                response["_meta"] = {"model_used": model, "tier": "primary" if model == chain.primary else "fallback"}
                return response

            except (TimeoutError, RateLimitError) as e:
                self.logger.warning(
                    f"模型{model}调用失败: {e}, 尝试下一个降级模型"
                )
                last_error = e
                continue

            except Exception as e:
                self.logger.error(f"模型{model}异常: {e}")
                last_error = e
                continue

        # 所有模型都失败，尝试缓存降级
        if chain.cache_fallback:
            query = messages[-1].get("content", "")
            cached = await self.cache.get_or_none(query)
            if cached:
                self.logger.info("使用语义缓存兜底")
                cached["_meta"] = {"model_used": "cache_fallback", "tier": "cache"}
                return cached

        raise Exception(f"所有降级路径失败: {last_error}")
```

---

## 六、成本控制：Token预算 + 模型路由

### 6.1 Token预算管理器

```python
from datetime import datetime, timedelta
from typing import Optional


class TokenBudgetManager:
    """Token预算管理器：按用户/团队/任务分配预算"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def set_budget(
        self,
        scope: str,           # "user:123" / "team:456" / "project:789"
        daily_limit: int,     # 每日Token上限
        monthly_limit: int,   # 每月Token上限
        cost_limit_usd: float = 50.0,  # 每日成本上限
    ):
        """设置预算"""
        key = f"budget:{scope}"
        await self.redis.hset(key, mapping={
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "cost_limit_usd": cost_limit_usd,
            "updated_at": str(time.time()),
        })

    async def check_and_consume(
        self,
        scope: str,
        estimated_tokens: int,
        estimated_cost_usd: float,
    ) -> tuple[bool, dict]:
        """检查预算并消费"""
        key = f"budget:{scope}"
        budget = await self.redis.hgetall(key)

        if not budget:
            return True, {"remaining": "unlimited"}

        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        # 检查每日Token预算
        daily_key = f"budget:usage:{scope}:daily:{today}"
        daily_used = int(await self.redis.get(daily_key) or 0)
        daily_limit = int(budget.get("daily_limit", 999_999_999))

        if daily_used + estimated_tokens > daily_limit:
            return False, {
                "reason": "daily_token_exceeded",
                "used": daily_used,
                "limit": daily_limit,
                "remaining": max(0, daily_limit - daily_used),
            }

        # 检查每日成本预算
        cost_key = f"budget:cost:{scope}:daily:{today}"
        cost_used = float(await self.redis.get(cost_key) or 0)
        cost_limit = float(budget.get("cost_limit_usd", 999_999))

        if cost_used + estimated_cost_usd > cost_limit:
            return False, {
                "reason": "daily_cost_exceeded",
                "used_usd": cost_used,
                "limit_usd": cost_limit,
            }

        # 扣减预算
        pipe = self.redis.pipeline()
        pipe.incrby(daily_key, estimated_tokens)
        pipe.expire(daily_key, 86400 * 2)  # 2天过期
        pipe.incrbyfloat(cost_key, estimated_cost_usd)
        pipe.expire(cost_key, 86400 * 2)

        # 月度统计
        monthly_key = f"budget:usage:{scope}:monthly:{month}"
        pipe.incrby(monthly_key, estimated_tokens)
        pipe.expire(monthly_key, 86400 * 35)  # 35天过期
        await pipe.execute()

        return True, {
            "daily_remaining": daily_limit - daily_used - estimated_tokens,
            "cost_remaining": cost_limit - cost_used - estimated_cost_usd,
        }
```

### 6.2 智能模型路由器

```python
class ModelRouter:
    """智能模型路由：根据任务复杂度、预算、延迟要求选择最优模型"""

    # 任务复杂度评估关键词
    COMPLEXITY_SIGNALS = {
        "high": ["分析", "推理", "规划", "compare", "analyze", "reason", "debug"],
        "medium": ["总结", "解释", "翻译", "summarize", "explain", "translate"],
        "low": ["查询", "格式化", "format", "query", "lookup"],
    }

    # 模型能力矩阵
    MODEL_CAPABILITIES = {
        "gpt-4o": {"quality": 0.95, "speed": 0.7, "cost_per_1k": 0.0125, "max_tokens": 128000},
        "gpt-4o-mini": {"quality": 0.80, "speed": 0.9, "cost_per_1k": 0.000375, "max_tokens": 128000},
        "claude-sonnet-4": {"quality": 0.92, "speed": 0.75, "cost_per_1k": 0.009, "max_tokens": 200000},
        "deepseek-v3": {"quality": 0.85, "speed": 0.85, "cost_per_1k": 0.000685, "max_tokens": 64000},
        "qwen-turbo": {"quality": 0.70, "speed": 0.95, "cost_per_1k": 0.0002, "max_tokens": 32000},
    }

    def select_model(
        self,
        task: str,
        budget_per_task: float = 0.05,
        latency_sla_ms: int = 5000,
        quality_threshold: float = 0.8,
    ) -> str:
        """根据任务特征选择最优模型"""

        complexity = self._assess_complexity(task)

        candidates = []
        for model, caps in self.MODEL_CAPABILITIES.items():
            score = self._routing_score(
                caps, complexity, budget_per_task, latency_sla_ms, quality_threshold
            )
            if score > 0:
                candidates.append((model, score))

        if not candidates:
            return "qwen-turbo"  # 最终兜底

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _assess_complexity(self, task: str) -> str:
        task_lower = task.lower()
        scores = {"high": 0, "medium": 0, "low": 0}
        for level, signals in self.COMPLEXITY_SIGNALS.items():
            for signal in signals:
                if signal in task_lower:
                    scores[level] += 1

        if scores["high"] >= 2:
            return "high"
        elif scores["medium"] >= 2:
            return "medium"
        return "low"

    def _routing_score(
        self, caps: dict, complexity: str,
        budget: float, latency_sla: int, quality_threshold: float
    ) -> float:
        # 质量过滤
        min_quality = {"high": 0.9, "medium": 0.75, "low": 0.6}
        if caps["quality"] < min_quality.get(complexity, 0.6):
            return -1

        # 成本过滤
        if caps["cost_per_1k"] > budget * 1000:
            return -1

        # 综合评分
        quality_score = caps["quality"] * 0.4
        speed_score = caps["speed"] * 0.3
        cost_score = (1 - caps["cost_per_1k"] / 0.02) * 0.3  # 归一化

        return quality_score + speed_score + max(0, cost_score)
```

---

## 七、告警规则设计

### 7.1 告警分级与规则

```python
from enum import Enum


class AlertSeverity(Enum):
    CRITICAL = "critical"    # P0: 系统不可用，立即处理
    WARNING = "warning"      # P1: 性能劣化，30分钟内处理
    INFO = "info"            # P2: 趋势异常，24小时内关注


# ---- Prometheus告警规则 (YAML) ----

ALERT_RULES = """
# === P0 Critical Alerts ===
groups:
  - name: agent-critical
    rules:
      # Agent成功率骤降
      - alert: AgentSuccessRateCritical
        expr: |
          (
            rate(agent_requests_total{status="success"}[5m])
            / rate(agent_requests_total[5m])
          ) < 0.85
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Agent成功率低于85%"
          description: "当前成功率: {{ $value | humanizePercentage }}"

      # LLM调用全部失败
      - alert: LLMAllDown
        expr: |
          sum(rate(agent_requests_total{status="error"}[5m])) > 10
          and
          sum(rate(agent_requests_total{status="success"}[5m])) < 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "所有LLM提供商不可用"

      # 端到端延迟超限
      - alert: AgentLatencyCritical
        expr: |
          histogram_quantile(0.99, rate(agent_request_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent P99延迟超过30秒"

  - name: agent-warning
    rules:
      # Token消耗异常
      - alert: TokenUsageAnomaly
        expr: |
          rate(agent_token_usage_total[10m])
          > 2 * avg_over_time(rate(agent_token_usage_total[10m])[1h:10m] offset 1d)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Token消耗异常升高（超日均2倍）"

      # 成本超预算
      - alert: CostBudgetExceeded
        expr: |
          sum(increase(agent_cost_dollars_total[24h])) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "24小时成本超过$100预算"

      # 降级比例升高
      - alert: FallbackRateHigh
        expr: |
          rate(agent_requests_total{tier="fallback"}[10m])
          / rate(agent_requests_total[10m]) > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "降级调用比例超过30%"
"""


# ---- 告警通知器 ----

class AgentAlertNotifier:
    """多渠道告警通知"""

    def __init__(self):
        self.channels = []

    async def notify(self, severity: AlertSeverity, title: str, message: str, context: dict = None):
        """发送告警通知"""
        payload = {
            "severity": severity.value,
            "title": title,
            "message": message,
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runbook_url": f"https://wiki.internal/runbook/{title.replace(' ', '-').lower()}",
        }

        # 企业微信/钉钉通知（P0）
        if severity == AlertSeverity.CRITICAL:
            await self._send_wechat_webhook(payload)

        # Slack/飞书通知（P0+P1）
        if severity in (AlertSeverity.CRITICAL, AlertSeverity.WARNING):
            await self._send_slack(payload)

        # 邮件通知（P2）
        if severity == AlertSeverity.INFO:
            await self._send_email(payload)

    async def _send_wechat_webhook(self, payload: dict):
        """企业微信机器人通知"""
        import httpx
        webhook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json={
                "msgtype": "markdown",
                "markdown": {
                    "content": f"## 🚨 {payload['title']}\n"
                               f"> 级别: **{payload['severity']}**\n"
                               f"> {payload['message']}\n"
                               f"> [Runbook]({payload['runbook_url']})"
                }
            })
```

---

## 八、生产部署检查清单

### 8.1 上线前检查清单

```yaml
# agent-production-checklist.yaml

pre_deployment:
  # === 监控与可观测性 ===
  monitoring:
    - [ ] Prometheus指标端点暴露（/metrics）
    - [ ] 核心指标Dashboard已配置（延迟/吞吐/成功率/成本）
    - [ ] 链路追踪系统已接入（OpenTelemetry/LangSmith）
    - [ ] 日志系统已配置（结构化JSON + ELK/ClickHouse）
    - [ ] 告警规则已配置并通过测试

  # === 限流与熔断 ===
  rate_limiting:
    - [ ] 全局QPS限制已设置
    - [ ] 用户级限流已配置（RPM + TPM + 成本）
    - [ ] 模型级限流已配置
    - [ ] 熔断器已配置（LLM调用失败率触发）
    - [ ] 限流后的友好错误提示已实现

  # === 降级与容灾 ===
  fallback:
    - [ ] 模型降级链已配置并测试
    - [ ] 语义缓存已预热
    - [ ] 工具调用降级方案已实现
    - [ ] 优雅降级提示已配置

  # === 成本控制 ===
  cost_control:
    - [ ] Token预算管理已配置（用户/团队/项目维度）
    - [ ] 智能模型路由已上线
    - [ ] 日/月成本上限已设置
    - [ ] 成本异常告警已配置

  # === 安全与合规 ===
  security:
    - [ ] API密钥已通过Secret Manager管理
    - [ ] PII脱敏已实现（日志/追踪中）
    - [ ] 输入注入防护已部署（Prompt Injection）
    - [ ] 输出内容审核已配置
    - [ ] 审计日志已开启

  # === 性能与容量 ===
  performance:
    - [ ] 压测通过（QPS达到预期1.5倍）
    - [ ] 自动扩缩容已配置
    - [ ] 连接池已优化
    - [ ] 超时策略已配置（LLM/工具/API）
    - [ ] 内存/CPU资源限制已设置

  # === 部署与回滚 ===
  deployment:
    - [ ] 灰度发布策略已配置
    - [ ] 回滚方案已测试
    - [ ] 数据库迁移已备份
    - [ ] 配置热更新机制已验证
    - [ ] SLO/SLA已定义

post_deployment:
  # === 上线后 ===
  verification:
    - [ ] 冒烟测试通过
    - [ ] 监控指标正常（无异常波动）
    - [ ] 告警通知渠道验证（发送测试告警）
    - [ ] 性能基线已记录
    - [ ] 值班人员已通知
```

### 8.2 SLO定义模板

```python
SLO_CONFIG = {
    "agent_service": {
        "availability": {
            "target": 99.9,           # 月可用性
            "measurement": "成功请求 / 总请求",
            "error_budget_minutes": 43.2,  # 月度允许停机时间
        },
        "latency": {
            "p50_target_ms": 3000,    # 50%请求在3秒内完成
            "p95_target_ms": 10000,   # 95%请求在10秒内完成
            "p99_target_ms": 20000,   # 99%请求在20秒内完成
        },
        "quality": {
            "task_completion_rate": 95.0,  # 任务完成率
            "user_satisfaction_score": 4.0, # 用户满意度（1-5）
        },
        "cost": {
            "daily_budget_usd": 100,
            "per_task_budget_usd": 0.10,
            "monthly_budget_usd": 2500,
        }
    }
}
```

---

## 总结

Agent系统生产化的核心挑战在于**不确定性管理**——LLM调用的延迟不确定、质量不确定、成本不确定。本文从六个维度构建了一套完整的生产化实践框架：

1. **监控**：以Prometheus+Grafana构建四象限指标体系，覆盖延迟/吞吐/成功率/成本
2. **追踪**：OpenTelemetry标准化 + LangSmith调试友好的双轨策略
3. **日志**：结构化JSON日志 + LLM调用专项记录 + 敏感信息脱敏
4. **限流**：Redis分布式令牌桶 + 滑动窗口的多维度限流矩阵
5. **降级**：语义缓存兜底 + 模型降级链的弹性架构
6. **成本**：Token预算管理 + 智能模型路由的成本优化

关键原则：**先有可观测性，再有稳定性，最后才是优化**。没有监控的优化是盲目的，没有限流的降级是脆弱的。在Agent系统的生产化道路上，基础设施的成熟度决定了业务创新的天花板。

---

*本文代码示例基于Python 3.11+，依赖：redis, opentelemetry, prometheus_client, numpy, httpx。生产部署前请根据实际技术栈调整。*
