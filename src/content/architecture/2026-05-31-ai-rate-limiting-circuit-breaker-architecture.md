---
title: "AI系统限流熔断架构设计：LLM服务的自适应流量治理实战"
description: "深入解析LLM服务限流熔断架构设计，涵盖令牌桶算法、自适应并发控制、模型级熔断器与多级限流策略的生产级实战方案"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: distributed
tags: ["系统架构", "限流熔断", "LLM网关", "流量治理", "AI系统韧性", "自适应控制"]
draft: false
---

## 引言：LLM服务的流量治理为什么不一样？

在传统微服务中，限流和熔断是"标准动作"——QPS超过阈值就限流，错误率超过50%就熔断。但当你的服务核心变成了LLM API，这套经典方案会遇到三个根本性的挑战：

**挑战一：Token级资源消耗不可预测**。同一个Prompt模板，用户输入10个字和1000个字，消耗的Token可能差100倍。按请求限流毫无意义——100个短请求可能还没1个长请求消耗的Token多。

**挑战二：限流响应的"灰色地带"**。传统服务限流返回429就完事了。但LLM API的429可能意味着你的配额快用完了，也可能是临时过载，两种情况的应对策略完全不同。更麻烦的是，LLM API经常返回200但内容是无意义的垃圾——传统限流对此无能为力。

**挑战三：模型级别的故障隔离**。你的系统可能同时接入GPT-4、Claude、Gemini多个模型。GPT-4限流了，但Claude还能用——传统熔断器是服务级别的，无法感知"同一个服务、不同模型"的故障差异。

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM 限流熔断架构全景                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 请求接入  │───▶│ Token限流 │───▶│ 并发控制 │───▶│ 模型熔断 │  │
│  │ (接入层)  │    │ (资源层)  │    │ (队列层)  │    │ (模型层)  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ IP限流    │    │ 预算控制  │    │ 等待队列  │    │ 故障转移  │  │
│  │ 用户限流  │    │ 配额管理  │    │ 优先级    │    │ 降级模型  │  │
│  │ 全局限流  │    │ 动态定价  │    │ 超时策略  │    │ 恢复检测  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  自适应反馈控制环                          │   │
│  │  实时指标采集 ──▶ 策略动态调整 ──▶ 效果评估 ──▶ 参数更新    │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

本文将从生产实践出发，系统性地讲解LLM服务的限流熔断架构设计。

---

## 一、Token级限流：告别"按请求计数"的原始时代

### 1.1 为什么传统限流在LLM场景失效？

传统限流的核心假设是：**每个请求的资源消耗是均匀的**。这个假设在LLM场景完全不成立。

| 场景 | 传统限流（按请求） | Token级限流 | 差异 |
|------|-------------------|------------|------|
| 简单问答（50 tokens） | 计数1次 | 消耗50 tokens | — |
| 长文分析（8000 tokens） | 计数1次 | 消耗8000 tokens | 160倍 |
| 批量摘要（10篇×2000 tokens） | 计数10次 | 消耗20000 tokens | 2倍于请求数 |

一个"按请求限流100次/分钟"的策略，可能允许用户消耗100万Token，也可能只允许消耗5000 Token——完全取决于用户写多长的Prompt。**这不是限流，这是赌博。**

### 1.2 Token桶算法：LLM限流的基础

Token桶（Token Bucket）是LLM限流的核心算法，但我们需要做三个关键改造：

```python
import time
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class TokenBucket:
    """LLM专用令牌桶：支持Token级限流"""
    capacity: int          # 桶容量（最大Token数）
    refill_rate: float     # 补充速率（tokens/秒）
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    
    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
    
    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    async def acquire(self, tokens: int, timeout: float = 30.0) -> bool:
        """尝试获取指定数量的Token"""
        deadline = time.monotonic() + timeout
        
        while time.monotonic() < deadline:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            # 计算需要等待的时间
            wait_time = (tokens - self.tokens) / self.refill_rate
            await asyncio.sleep(min(wait_time, 0.1))
        
        return False


class LLMLimitter:
    """LLM多维限流器"""
    
    def __init__(self):
        # 三个维度的限流桶
        self.user_buckets: dict[str, TokenBucket] = {}
        self.model_buckets: dict[str, TokenBucket] = {}
        self.global_bucket = TokenBucket(
            capacity=1_000_000,   # 全局100万Token
            refill_rate=16_666.67  # ~100万/分钟
        )
        
        # 模型配置
        self.model_configs = {
            "gpt-4o": {"capacity": 200_000, "rate": 3_333},
            "claude-sonnet": {"capacity": 150_000, "rate": 2_500},
            "gpt-4o-mini": {"capacity": 500_000, "rate": 8_333},
        }
    
    def _get_user_bucket(self, user_id: str) -> TokenBucket:
        if user_id not in self.user_buckets:
            # 每用户每分钟10万Token
            self.user_buckets[user_id] = TokenBucket(
                capacity=100_000,
                refill_rate=1_666.67
            )
        return self.user_buckets[user_id]
    
    def _get_model_bucket(self, model: str) -> TokenBucket:
        if model not in self.model_buckets:
            config = self.model_configs.get(model, 
                {"capacity": 100_000, "rate": 1_666})
            self.model_buckets[model] = TokenBucket(**config)
        return self.model_buckets[model]
    
    async def check_rate_limit(
        self, 
        user_id: str, 
        model: str, 
        estimated_tokens: int
    ) -> dict:
        """三级限流检查：用户 → 模型 → 全局"""
        results = {}
        
        # 1. 用户级限流
        user_bucket = self._get_user_bucket(user_id)
        results["user"] = await user_bucket.acquire(estimated_tokens)
        
        if not results["user"]:
            return {"allowed": False, "reason": "user_quota_exceeded"}
        
        # 2. 模型级限流
        model_bucket = self._get_model_bucket(model)
        results["model"] = await model_bucket.acquire(estimated_tokens)
        
        if not results["model"]:
            # 回退：用户级Token已经消耗了，需要补偿
            user_bucket.tokens += estimated_tokens
            return {"allowed": False, "reason": "model_quota_exceeded"}
        
        # 3. 全局限流
        results["global"] = await self.global_bucket.acquire(estimated_tokens)
        
        if not results["global"]:
            model_bucket.tokens += estimated_tokens
            user_bucket.tokens += estimated_tokens
            return {"allowed": False, "reason": "global_quota_exceeded"}
        
        return {"allowed": True, "tokens_consumed": estimated_tokens}
```

### 1.3 关键改造：预估与补偿

Token桶限流的一个核心难题是：**请求发出前你不知道实际消耗多少Token**。这里有两个实用策略：

**策略一：Prompt Token预估**

```python
def estimate_tokens(messages: list[dict], model: str) -> int:
    """预估Token消耗量"""
    # 粗略估算：中文1.5 token/字，英文0.75 token/word
    total_chars = sum(len(m.get("content", "")) for m in messages)
    
    # 粗粒度估算（1.5倍安全系数）
    estimated = int(total_chars * 1.5)
    
    # 模型特定调整
    model_multipliers = {
        "gpt-4o": 1.0,
        "claude-sonnet": 1.1,    # Claude tokenization略多
        "gpt-4o-mini": 0.95,
    }
    
    return int(estimated * model_multipliers.get(model, 1.0))


def calculate_compensation(actual_tokens: int, estimated_tokens: int) -> int:
    """计算Token补偿量"""
    # 如果实际消耗远超预估，补偿差额
    if actual_tokens > estimated_tokens * 1.5:
        return actual_tokens - estimated_tokens
    return 0
```

**策略二：请求后Token回补**

LLM API返回的usage字段是你的金矿——用它来修正预估模型：

```python
async def handle_completion_response(
    response: dict, 
    user_id: str, 
    model: str, 
    estimated_tokens: int
):
    """处理LLM响应，执行Token回补"""
    actual_tokens = response.get("usage", {}).get("total_tokens", 0)
    compensation = calculate_compensation(actual_tokens, estimated_tokens)
    
    if compensation > 0:
        # 预估少了，需要从用户桶中追加扣除
        user_bucket = limitter._get_user_bucket(user_id)
        user_bucket.tokens -= compensation
        
        # 记录预估误差，用于优化预估模型
        error_ratio = actual_tokens / estimated_tokens
        await metrics.record_estimation_error(model, error_ratio)
```

---

## 二、自适应并发控制：让排队变得聪明

### 2.1 为什么LLM需要特殊的并发控制？

LLM API的并发控制面临一个独特矛盾：

- **限制太严**：用户体验差，明明API能处理，你的网关却在排队
- **限制太松**：API限流429，你的重试风暴反而加剧了上游压力

解决方案是**自适应并发控制**——根据上游API的实际响应动态调整并发上限。

### 2.2 AIMD算法：TCP拥塞控制的AI版本

TCP的拥塞控制经过几十年验证，其核心思想——**加法增、乘法减（AIMD）**——非常适合LLM的并发控制：

```python
import asyncio
from dataclasses import dataclass
from enum import Enum

class ConcurrencyState(Enum):
    STARTUP = "startup"        # 启动期：快速探测
    DRAIN = "drain"            # 排空期：快速降级
    STEADY = "steady"          # 稳定期：缓慢增加


@dataclass
class AdaptiveConcurrency:
    """自适应并发控制器"""
    min_concurrency: int = 5       # 最小并发
    max_concurrency: int = 100     # 最大并发
    initial_concurrency: int = 20  # 初始并发
    
    # AIMD参数
    increase_rate: float = 1.0     # 加法增：每次+1
    decrease_factor: float = 0.5   # 乘法减：减半
    
    # 状态
    current_concurrency: int = 0
    state: ConcurrencyState = ConcurrencyState.STARTUP
    success_window: list = field(default_factory=list)  # 最近N次结果
    window_size: int = 20
    
    def __post_init__(self):
        self.current_concurrency = self.initial_concurrency
    
    def on_request_complete(self, success: bool, latency_ms: float):
        """请求完成回调"""
        self.success_window.append((success, latency_ms))
        if len(self.success_window) > self.window_size:
            self.success_window.pop(0)
        
        self._adjust_concurrency()
    
    def _adjust_concurrency(self):
        if len(self.success_window) < 5:
            return
        
        # 计算成功率
        success_rate = sum(1 for s, _ in self.success_window if s) / len(self.success_window)
        
        # 计算平均延迟
        avg_latency = sum(l for _, l in self.success_window) / len(self.success_window)
        
        if success_rate >= 0.95 and avg_latency < 5000:
            # 状态良好：加法增
            if self.state != ConcurrencyState.STEADY:
                self.state = ConcurrencyState.STEADY
            self.current_concurrency = min(
                self.current_concurrency + self.increase_rate,
                self.max_concurrency
            )
        elif success_rate < 0.8 or avg_latency > 10000:
            # 状态恶化：乘法减
            self.state = ConcurrencyState.DRAIN
            self.current_concurrency = max(
                int(self.current_concurrency * self.decrease_factor),
                self.min_concurrency
            )
    
    @property
    def effective_concurrency(self) -> int:
        return int(self.current_concurrency)
```

### 2.3 优先级队列：不是所有请求都平等

在生产环境中，不同用户的请求应该有不同的优先级：

```python
import heapq
from enum import IntEnum

class RequestPriority(IntEnum):
    CRITICAL = 0    # 系统内部调用、关键业务
    HIGH = 10       # 付费用户、SLA保障
    NORMAL = 20     # 普通用户
    LOW = 30        # 批量任务、后台作业
    BACKGROUND = 40 # 数据分析、报表生成


@dataclass
class PriorityRequest:
    priority: int
    user_id: str
    model: str
    estimated_tokens: int
    created_at: float
    
    def __lt__(self, other):
        # 优先级越高（数字越小）越先处理
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class PriorityQueue:
    """LLM请求优先级队列"""
    
    def __init__(self, max_size: int = 1000):
        self.heap: list[PriorityRequest] = []
        self.max_size = max_size
        self.user_counts: dict[str, int] = {}
        self.user_limits = {
            "enterprise": 200,    # 企业用户并发上限
            "pro": 50,           # 专业版
            "free": 10,          # 免费版
        }
    
    def enqueue(self, request: PriorityRequest) -> bool:
        """入队，带用户并发控制"""
        user_tier = self._get_user_tier(request.user_id)
        limit = self.user_limits.get(user_tier, 10)
        
        current = self.user_counts.get(request.user_id, 0)
        if current >= limit:
            return False  # 用户并发超限
        
        if len(self.heap) >= self.max_size:
            # 队列满，拒绝最低优先级的请求
            if request.priority > self.heap[0].priority:
                return False
        
        heapq.heappush(self.heap, request)
        self.user_counts[request.user_id] = current + 1
        return True
    
    def dequeue(self) -> PriorityRequest | None:
        """出队"""
        if not self.heap:
            return None
        request = heapq.heappop(self.heap)
        self.user_counts[request.user_id] -= 1
        return request
    
    def _get_user_tier(self, user_id: str) -> str:
        """获取用户等级（示例）"""
        # 实际中应从缓存或数据库获取
        return "pro"
```

---

## 三、模型级熔断器：精准隔离故障模型

### 3.1 为什么需要模型级熔断？

传统熔断器的粒度是"服务"——OpenAI的API挂了，就熔断所有OpenAI的调用。但在LLM场景中，经常出现更精细的故障：

- GPT-4o的function calling返回格式异常，但普通对话正常
- Claude的长文本生成超时，但短文本很快
- Gemini的多模态API不可用，但文本API正常

**你需要的不是一个"大熔断器"，而是一组按模型×能力维度的精细熔断器。**

### 3.2 状态机：熔断器的三态模型

```
                    ┌──────────────────────────┐
                    │                          │
        失败率达阈值 │                          │ 探测成功
              ┌─────▼─────┐            ┌───────┴───────┐
              │           │            │               │
    ┌─────────┤   OPEN    │───────────▶│ HALF-OPEN     │
    │         │  (熔断)   │  超时后     │  (半开)       │
    │         │           │            │               │
    │         └───────────┘            └───────┬───────┘
    │                                          │
    │          ┌─────────────┐                 │ 探测失败
    │          │             │                 │
    └──────────┤  CLOSED     │◀────────────────┘
   成功率恢复   │  (关闭)     │  探测失败
               │             │
               └─────────────┘
```

### 3.3 多维度熔断器实现

```python
import time
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

class CircuitState(Enum):
    CLOSED = "closed"         # 正常通过
    OPEN = "open"             # 熔断拒绝
    HALF_OPEN = "half_open"   # 半开探测


@dataclass
class ModelCircuitBreaker:
    """模型级熔断器：支持能力维度的精细故障隔离"""
    model: str
    capability: str          # "chat", "function_calling", "embedding", "vision"
    
    # 熔断参数
    failure_threshold: int = 5         # 失败次数阈值
    failure_rate_threshold: float = 0.5  # 失败率阈值
    recovery_timeout: float = 60.0     # 恢复超时（秒）
    half_open_max_calls: int = 3       # 半开状态最大探测数
    slow_call_threshold: float = 10.0  # 慢调用阈值（秒）
    slow_call_rate_threshold: float = 0.8  # 慢调用率阈值
    
    # 状态
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    
    # 滑动窗口（最近100次调用）
    call_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def can_execute(self) -> bool:
        """判断是否允许执行"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # 检查是否到了恢复超时
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self, latency: float):
        """记录成功调用"""
        self.total_calls += 1
        self.success_count += 1
        self.call_history.append(("success", latency, time.time()))
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            # 半开状态连续成功，关闭熔断器
            if self.half_open_calls >= self.half_open_max_calls:
                self._close()
    
    def record_failure(self, error_type: str, latency: float):
        """记录失败调用"""
        self.total_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.call_history.append(("failure", latency, time.time()))
        
        if self.state == CircuitState.HALF_OPEN:
            # 半开状态失败，立即打开
            self._open()
            return
        
        if self.state == CircuitState.CLOSED:
            # 检查是否达到熔断条件
            if self._should_trip():
                self._open()
    
    def _should_trip(self) -> bool:
        """判断是否应该触发熔断"""
        # 条件1：连续失败次数
        recent_failures = sum(
            1 for status, _, time_ in self.call_history
            if status == "failure" 
            and time.time() - time_ < 60  # 最近60秒
        )
        if recent_failures >= self.failure_threshold:
            return True
        
        # 条件2：失败率
        recent_calls = [h for h in self.call_history if time.time() - h[2] < 60]
        if len(recent_calls) >= 10:
            failure_rate = sum(1 for h in recent_calls if h[0] == "failure") / len(recent_calls)
            if failure_rate >= self.failure_rate_threshold:
                return True
        
        # 条件3：慢调用率
        if len(recent_calls) >= 10:
            slow_calls = sum(1 for h in recent_calls if h[1] > self.slow_call_threshold)
            slow_rate = slow_calls / len(recent_calls)
            if slow_rate >= self.slow_call_rate_threshold:
                return True
        
        return False
    
    def _open(self):
        """打开熔断器"""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
    
    def _close(self):
        """关闭熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0


class MultiModelCircuitManager:
    """多模型熔断管理器"""
    
    def __init__(self):
        # model -> capability -> CircuitBreaker
        self.breakers: dict[str, dict[str, ModelCircuitBreaker]] = {}
    
    def get_breaker(self, model: str, capability: str) -> ModelCircuitBreaker:
        if model not in self.breakers:
            self.breakers[model] = {}
        if capability not in self.breakers[model]:
            self.breakers[model][capability] = ModelCircuitBreaker(
                model=model, capability=capability
            )
        return self.breakers[model][capability]
    
    def can_call(self, model: str, capability: str) -> bool:
        """检查模型的特定能力是否可用"""
        breaker = self.get_breaker(model, capability)
        return breaker.can_execute()
    
    def get_available_models(self, capability: str) -> list[str]:
        """获取某能力下所有可用模型"""
        available = []
        for model, capabilities in self.breakers.items():
            if capability in capabilities:
                if capabilities[capability].can_execute():
                    available.append(model)
        return available
    
    def get_circuit_status(self) -> dict:
        """获取所有熔断器状态（供监控使用）"""
        status = {}
        for model, capabilities in self.breakers.items():
            status[model] = {}
            for cap, breaker in capabilities.items():
                status[model][cap] = {
                    "state": breaker.state.value,
                    "failure_rate": (
                        breaker.failure_count / breaker.total_calls 
                        if breaker.total_calls > 0 else 0
                    ),
                    "total_calls": breaker.total_calls,
                }
        return status
```

---

## 四、多级限流策略：从接入层到模型层的全链路管控

### 4.1 限流层次架构

LLM服务需要在多个层次设置限流，每一层解决不同的问题：

| 限流层级 | 管控维度 | 解决的问题 | 典型策略 |
|---------|---------|-----------|---------|
| **接入层** | IP、用户ID | 恶意请求、DDoS | 滑动窗口限流 |
| **业务层** | 用户等级、功能权限 | 商业化控制 | 配额管理 |
| **Token层** | 消耗的Token数量 | 资源公平分配 | Token桶 |
| **并发层** | 同时处理的请求数 | 上游API保护 | 自适应并发 |
| **模型层** | 模型+能力维度 | 故障隔离 | 熔断器 |

### 4.2 限流响应的标准化

LLM限流的响应应该包含足够的信息，让客户端做出智能决策：

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RateLimitResponse:
    """标准化限流响应"""
    allowed: bool
    reason: str
    
    # 资源信息
    tokens_remaining: Optional[int] = None
    tokens_limit: Optional[int] = None
    reset_at: Optional[float] = None          # 重置时间戳
    
    # 替代建议
    suggested_model: Optional[str] = None     # 推荐的替代模型
    retry_after: Optional[float] = None       # 建议重试等待时间（秒）
    estimated_cost: Optional[float] = None    # 预估Token成本
    
    def to_headers(self) -> dict[str, str]:
        """转换为HTTP响应头"""
        headers = {
            "X-RateLimit-Allowed": str(self.allowed).lower(),
            "X-RateLimit-Reason": self.reason,
        }
        if self.tokens_remaining is not None:
            headers["X-RateLimit-Tokens-Remaining"] = str(self.tokens_remaining)
        if self.tokens_limit is not None:
            headers["X-RateLimit-Tokens-Limit"] = str(self.tokens_limit)
        if self.reset_at is not None:
            headers["X-RateLimit-Reset"] = str(int(self.reset_at))
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        if self.suggested_model:
            headers["X-RateLimit-Suggested-Model"] = self.suggested_model
        return headers


class SmartRateLimiter:
    """智能限流器：根据故障类型给出差异化响应"""
    
    # 模型降级链
    FALLBACK_CHAINS = {
        "gpt-4o": ["claude-sonnet", "gpt-4o-mini", "gemini-pro"],
        "claude-sonnet": ["gpt-4o", "gpt-4o-mini"],
        "gpt-4o-mini": ["gemini-flash", "claude-haiku"],
    }
    
    async def check_and_respond(
        self, 
        user_id: str, 
        model: str, 
        estimated_tokens: int
    ) -> RateLimitResponse:
        """检查限流并生成响应"""
        
        # 1. 检查Token配额
        user_bucket = limitter._get_user_bucket(user_id)
        if user_bucket.tokens < estimated_tokens:
            wait_time = (estimated_tokens - user_bucket.tokens) / user_bucket.refill_rate
            suggested = self._suggest_fallback_model(model)
            
            return RateLimitResponse(
                allowed=False,
                reason="token_quota_exceeded",
                tokens_remaining=int(user_bucket.tokens),
                reset_at=time.time() + wait_time,
                retry_after=wait_time,
                suggested_model=suggested,
            )
        
        # 2. 检查熔断状态
        if not circuit_manager.can_call(model, "chat"):
            suggested = self._suggest_fallback_model(model)
            return RateLimitResponse(
                allowed=False,
                reason="circuit_breaker_open",
                suggested_model=suggested,
                retry_after=60.0,
            )
        
        # 3. 通过
        return RateLimitResponse(
            allowed=True,
            reason="ok",
            tokens_remaining=int(user_bucket.tokens),
            estimated_cost=self._estimate_cost(model, estimated_tokens),
        )
    
    def _suggest_fallback_model(self, model: str) -> Optional[str]:
        """推荐降级模型"""
        chain = self.FALLBACK_CHAINS.get(model, [])
        for fallback in chain:
            if circuit_manager.can_call(fallback, "chat"):
                return fallback
        return None
    
    def _estimate_cost(self, model: str, tokens: int) -> float:
        """预估成本（美元）"""
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "claude-sonnet": {"input": 3.00, "output": 15.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gemini-pro": {"input": 1.25, "output": 5.00},
        }
        rates = pricing.get(model, {"input": 1.0, "output": 4.0})
        # 假设输入:输出 = 3:1
        input_tokens = tokens * 0.75
        output_tokens = tokens * 0.25
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
```

---

## 五、生产实践：LLM网关的限流熔断集成

### 5.1 限流熔断中间件架构

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Gateway 请求处理流程                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Request ──▶ [1] 接入限流 ──▶ [2] Token预估                 │
│                              │                              │
│                    ┌─────────▼──────────┐                  │
│                    │   Token桶限流检查   │                  │
│                    │   (用户+模型+全局)   │                  │
│                    └─────────┬──────────┘                  │
│                              │                              │
│                    ┌─────────▼──────────┐                  │
│                    │   优先级队列排队    │                  │
│                    │   (等待或立即执行)   │                  │
│                    └─────────┬──────────┘                  │
│                              │                              │
│                    ┌─────────▼──────────┐                  │
│                    │   并发控制检查      │                  │
│                    │   (自适应信号量)    │                  │
│                    └─────────┬──────────┘                  │
│                              │                              │
│                    ┌─────────▼──────────┐                  │
│                    │   熔断器状态检查    │                  │
│                    │   (模型+能力维度)   │                  │
│                    └─────────┬──────────┘                  │
│                              │                              │
│                    ┌─────────▼──────────┐                  │
│                    │   模型选择与路由    │                  │
│                    │   (降级链执行)      │                  │
│                    └─────────┬──────────┘                  │
│                              │                              │
│                         LLM API                            │
│                              │                              │
│                    ┌─────────▼──────────┐                  │
│                    │   响应后处理        │                  │
│                    │   - Token回补       │                  │
│                    │   - 熔断器记录      │                  │
│                    │   - 并发控制释放    │                  │
│                    │   - 指标上报        │                  │
│                    └────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 关键指标监控

限流熔断系统的健康度需要通过以下指标来监控：

```python
# 限流熔断系统核心监控指标
METRICS = {
    # 限流指标
    "rate_limit_requests_total": "限流拒绝的请求总数",
    "rate_limit_by_reason": "按原因分类的限流次数",
    "token_bucket_utilization": "Token桶使用率（0-1）",
    "token_estimation_error_ratio": "Token预估误差率",
    
    # 熔断指标
    "circuit_breaker_state": "熔断器状态（0=closed, 1=half_open, 2=open）",
    "circuit_breaker_trip_total": "熔断器触发次数",
    "circuit_breaker_recovery_time": "熔断器恢复耗时",
    
    # 并发指标
    "concurrent_requests_active": "当前活跃请求数",
    "queue_wait_time_p99": "排队等待时间P99",
    "queue_depth": "当前队列深度",
    
    # 降级指标
    "fallback_requests_total": "降级到备用模型的请求数",
    "fallback_success_rate": "降级请求的成功率",
    "model_switch_total": "模型切换次数",
}
```

### 5.3 告警规则设计

| 告警级别 | 触发条件 | 响应动作 |
|---------|---------|---------|
| **P0 紧急** | 全局限流触发率 > 20% 持续5分钟 | 立即扩容、通知On-Call |
| **P1 严重** | 主力模型熔断持续 > 10分钟 | 检查上游API状态、手动切换 |
| **P2 警告** | 降级请求占比 > 30% | 排查原因、优化限流参数 |
| **P3 提醒** | Token预估误差率 > 50% | 优化预估算法 |

---

## 六、架构选型与演进路径

### 6.1 不同规模的架构选型

| 系统规模 | 日请求量 | 推荐架构 | 核心组件 |
|---------|---------|---------|---------|
| **MVP** | < 1万 | 单机限流 | 内存Token桶 + 简单熔断 |
| **初创** | 1-10万 | 单机+Redis | Redis分布式限流 + 本地熔断 |
| **成长** | 10-100万 | 微服务网关 | 多级限流 + 模型级熔断 + 优先级队列 |
| **规模** | > 100万 | 分布式网关集群 | 一致性限流 + 自适应控制 + 全局调度 |

### 6.2 演进路线图

```
阶段1：基础限流
├── 按IP/用户ID的滑动窗口限流
├── 简单的错误计数熔断
└── 手动配置限流参数

    ↓  （3-6个月）

阶段2：Token级管控
├── Token桶算法实现
├── 模型级熔断器
├── 限流响应标准化
└── 基础监控告警

    ↓  （6-12个月）

阶段3：自适应治理
├── AIMD自适应并发控制
├── 智能降级与模型路由
├── 优先级队列
└── 实时指标看板

    ↓  （12个月+）

阶段4：智能化运营
├── 基于历史数据的预测性限流
├── 自动化参数调优
├── 成本感知的请求调度
└── 多维度AB测试框架
```

---

## 总结

LLM服务的限流熔断不是传统微服务限流的简单升级，而是一套需要重新设计的治理体系。核心要点：

1. **Token是LLM的第一资源单位**——所有限流策略都应该以Token为核心维度，而非请求数
2. **并发控制要自适应**——TCP的AIMD思想经过验证，非常适合LLM的并发调控
3. **熔断要足够精细**——模型×能力维度的熔断器，才能避免"一个模型出问题、全部服务停摆"的窘境
4. **限流响应要有智能**——告诉客户端"为什么被限流"以及"该怎么降级"，远比一个429有用得多

好的限流熔断架构，不是让请求变得更慢，而是让系统在压力下依然保持可预测的行为——这对LLM应用的用户体验至关重要。
