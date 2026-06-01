---
title: "AI应用韧性架构设计：从熔断降级到自愈恢复的生产级实践"
description: "深入解析AI应用的韧性架构设计模式，涵盖熔断器、降级策略、重试机制、自愈恢复与混沌工程，构建高可用AI系统的完整技术方案"
date: 2026-05-30
author: "RiceBall-15"
category: "architecture"
subCategory: distributed
tags: ["韧性架构", "熔断降级", "高可用", "混沌工程", "AI系统架构", "分布式系统"]
draft: false
---

# AI应用韧性架构设计：从熔断降级到自愈恢复的生产级实践

## 一、引言：AI系统的脆弱性本质

### 1.1 为什么AI系统比传统系统更脆弱？

传统的Web应用遵循确定性逻辑——输入A总是产生输出B。但AI系统，尤其是基于LLM的应用，天然是**概率性、延迟性、不可预测的**。这种本质差异带来了全新的脆弱性维度：

```
┌─────────────────────────────────────────────────────────────────────┐
│              AI应用 vs 传统应用 的脆弱性对比                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  传统Web应用                    AI/LLM应用                           │
│  ┌──────────────────┐          ┌──────────────────┐                │
│  │ 响应时间: 50-200ms│          │ 响应时间: 1-30s   │                │
│  │ 成功率: 99.9%+    │          │ 成功率: 95-99%    │                │
│  │ 输出确定性        │          │ 输出不确定性      │                │
│  │ 无状态            │          │ 长上下文状态      │                │
│  │ 资源消耗可预测    │          │ Token消耗波动大   │                │
│  │ 错误类型明确      │          │ 错误类型多样      │                │
│  └──────────────────┘          └──────────────────┘                │
│                                                                      │
│  AI应用特有的失败模式:                                                 │
│  ├── API限流 (429 Too Many Requests)                                 │
│  ├── 上下文超长 (Context Length Exceeded)                             │
│  ├── 内容过滤拒绝 (Content Policy Violation)                         │
│  ├── 幻觉输出 (Hallucination)                                        │
│  ├── 延迟飙升 (Latency Spike)                                        │
│  ├── Token配额耗尽 (Quota Exhausted)                                 │
│  └── 模型版本漂移 (Model Drift)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 真实事故案例

**案例一：级联故障。** 某电商平台的AI客服系统在大促期间，因LLM API限流导致请求堆积，下游订单系统被反压击垮，最终造成全站宕机。损失超过500万元。

**案例二：幻觉扩散。** 某金融公司的AI分析报告系统，因模型输出幻觉数据，被下游风控系统误采信，导致错误的风险评估被提交给监管机构。

**案例三：长尾延迟。** 某SaaS产品的AI写作功能，P99延迟达到45秒，用户大量流失。根本原因是未对LLM调用设置超时，导致慢请求阻塞了整个连接池。

这些案例揭示了一个核心命题：**AI应用的韧性不是可选项，而是生存必需品。**

## 二、AI应用韧性架构全景图

### 2.1 韧性设计的核心原则

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI应用韧性设计四原则                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │   防御       │    │   检测       │    │   恢复       │            │
│  │  Defense     │    │  Detection  │    │  Recovery   │            │
│  │             │    │             │    │             │            │
│  │ • 超时控制  │    │ • 健康检查  │    │ • 自动重试  │            │
│  │ • 速率限制  │    │ • 异常检测  │    │ • 降级方案  │            │
│  │ • 输入校验  │    │ • 延迟监控  │    │ • 熔断恢复  │            │
│  │ • 资源隔离  │    │ • 质量评估  │    │ • 状态回滚  │            │
│  └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     适应 Adaptation                          │   │
│  │                                                             │   │
│  │  • 负载自适应        • 路由自愈        • 配置热更新          │   │
│  │  • 模型自切换        • 流量自调度      • 策略自进化          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 韧性层次模型

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI应用韧性层次模型                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 5: 业务韧性                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • 业务降级策略    • 用户体验兜底    • 数据一致性保证          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Layer 4: AI模型韧性                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • 模型版本回退    • 多模型容灾      • 输出质量验证            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Layer 3: 服务韧性                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • 熔断器模式      • 重试与退避      • 超时控制                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Layer 2: 网络韧性                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • 连接池管理      • DNS故障转移     • 多区域部署              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Layer 1: 基础设施韧性                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • 容器自愈        • 自动扩缩容      • 数据持久化              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 三、核心韧性模式详解

### 3.1 熔断器模式（Circuit Breaker）

熔断器是防止故障级联的第一道防线。在AI应用中，熔断器需要针对LLM的特殊性进行适配：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI专用熔断器状态机                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                          ┌──────────────┐                          │
│                          │              │                          │
│              失败率超标   │    关闭       │   成功率恢复            │
│            ┌────────────│   (正常)     │◄─────────────┐          │
│            │            │              │               │          │
│            │            └──────────────┘               │          │
│            │                   │                       │          │
│            ▼                   │ 超时/失败率超标        │          │
│  ┌──────────────┐             │                       │          │
│  │              │             │                       │          │
│  │    打开       │─────────────┘                       │          │
│  │  (拒绝请求)  │                                     │          │
│  │              │──── 冷却期到期 ────►┌──────────────┐│          │
│  └──────────────┘                    │              ││          │
│                                      │   半开       ││          │
│                                      │  (试探请求)  ││          │
│                                      │              ││          │
│                                      └──────────────┘│          │
│                                           │          │          │
│                                           │ 试探失败  │          │
│                                           └──────────┘          │
│                                                                      │
│  AI应用特殊状态:                                                       │
│  ├── 模型降级状态: 主模型不可用，自动切换备用模型                       │
│  ├── 限流降级状态: 超出配额，使用本地小模型替代                         │
│  └── 质量降级状态: 输出质量不达标，触发重试或人工审核                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional

class CircuitState(Enum):
    CLOSED = "closed"       # 正常状态
    OPEN = "open"           # 熔断状态
    HALF_OPEN = "half_open" # 试探状态

class AICircuitBreaker:
    """AI应用专用熔断器"""
    
    def __init__(
        self,
        failure_threshold: int = 5,        # 失败次数阈值
        recovery_timeout: float = 30.0,     # 恢复超时(秒)
        half_open_max_calls: int = 3,       # 半开状态最大试探次数
        success_threshold: int = 2,         # 半开恢复所需成功次数
        # AI应用特有的配置
        latency_threshold: float = 30.0,    # 延迟阈值(秒)
        error_types: list = None,           # 需要熔断的错误类型
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.latency_threshold = latency_threshold
        self.error_types = error_types or [
            "rate_limit", "timeout", "quota_exhausted", "service_unavailable"
        ]
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
            return self._state
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行LLM调用，带熔断保护"""
        current_state = self.state
        
        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"熔断器已打开，将在 {self.recovery_timeout}s 后重试"
            )
        
        if current_state == CircuitState.HALF_OPEN:
            if self._success_count >= self.half_open_max_calls:
                raise CircuitBreakerOpenError("半开状态试探次数已达上限")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            
            # 检查延迟是否超标
            if latency > self.latency_threshold:
                self._record_failure("latency_exceeded")
                return result  # 仍然返回结果，但记录失败
            
            self._record_success()
            return result
            
        except Exception as e:
            error_type = self._classify_error(e)
            if error_type in self.error_types:
                self._record_failure(error_type)
            raise
    
    def _record_failure(self, error_type: str):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                print(f"[CircuitBreaker] 熔断器打开 - 失败类型: {error_type}")
    
    def _record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    print("[CircuitBreaker] 熔断器恢复正常")
            else:
                self._failure_count = 0
    
    def _classify_error(self, error: Exception) -> str:
        error_msg = str(error).lower()
        if "rate" in error_msg and "limit" in error_msg:
            return "rate_limit"
        elif "timeout" in error_msg:
            return "timeout"
        elif "quota" in error_msg:
            return "quota_exhausted"
        else:
            return "unknown"


class CircuitBreakerOpenError(Exception):
    pass
```

### 3.2 智能重试与退避策略

LLM API的重试比传统HTTP重试更复杂，需要考虑Token预算、请求幂等性和上下文一致性：

```
┌─────────────────────────────────────────────────────────────────────┐
│                AI应用智能重试策略对比                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  策略              适用场景           风险              推荐度│   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  立即重试          临时网络抖动       可能加重限流       ★☆☆ │   │
│  │  固定间隔重试      稳定错误恢复       不适应负载变化     ★★☆ │   │
│  │  指数退避          渐进式恢复         延迟累积           ★★★ │   │
│  │  指数退避+抖动     分布式限流规避     实现复杂           ★★★★│   │
│  │  自适应退避        动态调整策略       需要监控数据       ★★★★★│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  推荐: 指数退避 + 随机抖动 + Token预算感知                            │
│                                                                      │
│  退避公式:                                                            │
│  delay = min(base_delay * 2^attempt + random(0, jitter), max_delay) │
│                                                                      │
│  Token预算感知:                                                       │
│  if remaining_tokens < estimated_retry_tokens:                       │
│      trigger_fallback()  # 触发降级而非重试                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
import asyncio
import random
import time
from typing import Callable, Any, Optional

class AIRetryPolicy:
    """AI应用智能重试策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.5,
        # Token预算感知
        token_budget_remaining: Optional[int] = None,
        estimated_retry_tokens: int = 500,
        # 延迟预算
        total_timeout: float = 60.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.token_budget_remaining = token_budget_remaining
        self.estimated_retry_tokens = estimated_retry_tokens
        self.total_timeout = total_timeout
    
    def should_retry(self, attempt: int, error: Exception, tokens_used: int) -> bool:
        """判断是否应该重试"""
        # 检查重试次数
        if attempt >= self.max_retries:
            return False
        
        # 检查Token预算
        if self.token_budget_remaining is not None:
            remaining = self.token_budget_remaining - tokens_used
            if remaining < self.estimated_retry_tokens:
                print(f"[RetryPolicy] Token预算不足: 剩余 {remaining}, 需要 {self.estimated_retry_tokens}")
                return False
        
        # 检查延迟预算
        # (由调用方跟踪总耗时)
        
        # 检查错误类型是否可重试
        return self._is_retryable_error(error)
    
    def get_delay(self, attempt: int) -> float:
        """计算退避延迟"""
        # 指数退避 + 随机抖动
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        
        # 添加随机抖动 (避免惊群效应)
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        error_msg = str(error).lower()
        
        # 可重试的错误
        retryable_patterns = [
            "rate_limit",           # 限流 (等待后重试)
            "timeout",              # 超时 (可能是暂时的)
            "connection",           # 连接错误
            "500", "502", "503",   # 服务器错误
        ]
        
        # 不可重试的错误
        non_retryable_patterns = [
            "invalid_api_key",      # API密钥错误
            "quota_exhausted",      # 配额耗尽 (需要升级)
            "content_policy",       # 内容违规 (需要修改输入)
            "invalid_request",      # 请求格式错误
        ]
        
        for pattern in non_retryable_patterns:
            if pattern in error_msg:
                return False
        
        for pattern in retryable_patterns:
            if pattern in error_msg:
                return True
        
        return False  # 未知错误不重试


async def retry_with_policy(
    func: Callable,
    policy: AIRetryPolicy,
    *args,
    **kwargs
) -> Any:
    """带重试策略的异步执行"""
    last_error = None
    start_time = time.time()
    tokens_used = 0
    
    for attempt in range(policy.max_retries + 1):
        try:
            # 检查总超时
            elapsed = time.time() - start_time
            if elapsed > policy.total_timeout:
                raise TimeoutError(f"总超时 {policy.total_timeout}s 已耗尽")
            
            result = await func(*args, **kwargs)
            return result
            
        except Exception as e:
            last_error = e
            
            # 提取Token使用量 (如果可用)
            if hasattr(e, 'tokens_used'):
                tokens_used += e.tokens_used
            
            if not policy.should_retry(attempt, e, tokens_used):
                raise
            
            delay = policy.get_delay(attempt)
            print(f"[Retry] 第{attempt+1}次重试, 等待 {delay:.2f}s, 错误: {e}")
            await asyncio.sleep(delay)
    
    raise last_error
```

### 3.3 降级策略体系

降级是韧性架构的最后一道防线。AI应用的降级策略需要在功能完整性和用户体验之间找到平衡：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI应用降级策略金字塔                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         ┌─────────┐                                 │
│                         │ Level 4 │  完全降级                        │
│                         │ 返回默认 │  (使用硬编码的静态响应)          │
│                        ┌┴─────────┴┐                                │
│                        │  Level 3  │  模型降级                       │
│                        │ 切换备用   │  (从GPT-4切换到本地小模型)      │
│                       ┌┴───────────┴┐                               │
│                       │   Level 2   │  功能降级                      │
│                       │  简化功能    │  (关闭流式输出,减少上下文)      │
│                      ┌┴─────────────┴┐                              │
│                      │    Level 1    │  质量降级                      │
│                      │  接受较长延迟  │  (增加超时,降低并发)           │
│                     ┌┴───────────────┴┐                             │
│                     │     Level 0     │  正常状态                     │
│                     │    全功能运行    │  (所有功能正常)               │
│                     └─────────────────┘                             │
│                                                                      │
│  降级触发条件:                                                        │
│  ├── 延迟 > 阈值 → Level 1                                           │
│  ├── 错误率 > 5% → Level 2                                           │
│  ├── 主模型不可用 → Level 3                                           │
│  └── 所有模型不可用 → Level 4                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from enum import IntEnum
from typing import Dict, Callable, Any, Optional
import asyncio

class DegradationLevel(IntEnum):
    NORMAL = 0        # 正常
    LATENCY = 1       # 延迟降级
    FEATURE = 2       # 功能降级
    MODEL = 3         # 模型降级
    STATIC = 4        # 静态降级

class AIDegradationManager:
    """AI应用降级管理器"""
    
    def __init__(self):
        self.current_level = DegradationLevel.NORMAL
        self.strategies: Dict[DegradationLevel, Callable] = {}
        self.metrics = {
            "latency_threshold": 10.0,  # 秒
            "error_rate_threshold": 0.05,
        }
    
    def register_strategy(
        self, 
        level: DegradationLevel, 
        strategy: Callable
    ):
        """注册降级策略"""
        self.strategies[level] = strategy
    
    def evaluate_and_degrade(
        self, 
        latency: float, 
        error_rate: float,
        model_available: bool = True
    ) -> DegradationLevel:
        """评估当前状态并决定降级级别"""
        new_level = DegradationLevel.NORMAL
        
        if not model_available:
            new_level = DegradationLevel.MODEL
        elif error_rate > self.metrics["error_rate_threshold"]:
            new_level = DegradationLevel.FEATURE
        elif latency > self.metrics["latency_threshold"]:
            new_level = DegradationLevel.LATENCY
        
        if new_level > self.current_level:
            self.current_level = new_level
            print(f"[Degradation] 降级至 Level {new_level.value}: {new_level.name}")
        
        return self.current_level
    
    async def execute_with_degradation(
        self, 
        primary_func: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """带降级的执行"""
        try:
            result = await primary_func(*args, **kwargs)
            # 成功，尝试恢复
            if self.current_level > DegradationLevel.NORMAL:
                self.current_level = DegradationLevel(self.current_level - 1)
            return result
            
        except Exception as e:
            # 执行降级策略
            level = self.evaluate_and_degrade(
                latency=0,  # 实际应从监控获取
                error_rate=1.0,  # 本次失败
                model_available="quota" not in str(e).lower()
            )
            
            if level in self.strategies:
                return await self.strategies[level](*args, **kwargs)
            raise


# 使用示例
manager = AIDegradationManager()

async def fallback_to_smaller_model(*args, **kwargs):
    """降级到本地小模型"""
    print("[Fallback] 切换到本地 Qwen-7B")
    # 实际实现: 调用本地vLLM服务
    return "这是由本地小模型生成的降级响应"

async def static_fallback(*args, **kwargs):
    """静态降级响应"""
    return {
        "response": "抱歉，AI服务暂时不可用。您的请求已记录，我们将尽快回复。",
        "status": "degraded",
        "ticket_id": f"TK-{int(time.time())}"
    }

manager.register_strategy(DegradationLevel.MODEL, fallback_to_smaller_model)
manager.register_strategy(DegradationLevel.STATIC, static_fallback)
```

## 四、AI输出质量保障

### 4.1 输出验证流水线

LLM的输出不总是可靠的，需要多层验证：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI输出验证流水线                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LLM原始输出                                                         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Layer 1: 格式验证                                           │   │
│  │  • JSON格式检查          • 字段完整性                       │   │
│  │  • Schema验证            • 编码正确性                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼ (通过)                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Layer 2: 安全验证                                           │   │
│  │  • 内容安全过滤          • 敏感信息检测                      │   │
│  │  • PII脱敏              • 注入攻击检测                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼ (通过)                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Layer 3: 质量验证                                           │   │
│  │  • 幻觉检测 (RAG一致性)  • 相关性评分                       │   │
│  │  • 事实核查             • 逻辑一致性                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼ (通过)                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Layer 4: 业务验证                                           │   │
│  │  • 业务规则校验          • 一致性检查                        │   │
│  │  • 格式规范符合          • 历史一致性                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼ (全部通过)                                                    │
│  ✅ 验证通过 → 返回给用户                                             │
│                                                                      │
│  ❌ 任一层失败 → 触发重试或降级                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from typing import Dict, Any, List, Optional
import re

class AIOutputValidator:
    """AI输出多层验证器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.validators = [
            self._validate_format,
            self._validate_safety,
            self._validate_quality,
            self._validate_business,
        ]
    
    def validate(self, output: str, context: Dict = None) -> Dict[str, Any]:
        """执行多层验证"""
        results = {
            "valid": True,
            "layers": [],
            "errors": [],
        }
        
        for validator in self.validators:
            layer_result = validator(output, context or {})
            results["layers"].append(layer_result)
            
            if not layer_result["passed"]:
                results["valid"] = False
                results["errors"].append({
                    "layer": layer_result["layer"],
                    "error": layer_result["error"],
                })
                
                # 非致命错误可以继续验证
                if layer_result.get("fatal", True):
                    break
        
        return results
    
    def _validate_format(self, output: str, context: Dict) -> Dict:
        """Layer 1: 格式验证"""
        expected_format = context.get("expected_format", "text")
        
        if expected_format == "json":
            try:
                import json
                json.loads(output)
                return {"layer": "format", "passed": True}
            except json.JSONDecodeError as e:
                return {
                    "layer": "format",
                    "passed": False,
                    "error": f"JSON格式错误: {e}",
                    "fatal": True,
                }
        
        # 文本长度检查
        max_length = context.get("max_length", 10000)
        if len(output) > max_length:
            return {
                "layer": "format",
                "passed": False,
                "error": f"输出超长: {len(output)} > {max_length}",
                "fatal": False,  # 可以截断
            }
        
        return {"layer": "format", "passed": True}
    
    def _validate_safety(self, output: str, context: Dict) -> Dict:
        """Layer 2: 安全验证"""
        # PII检测 (简单示例)
        pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b1[3-9]\d{9}\b',
            "id_card": r'\b\d{17}[\dXx]\b',
        }
        
        detected_pii = []
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, output):
                detected_pii.append(pii_type)
        
        if detected_pii and not context.get("allow_pii", False):
            return {
                "layer": "safety",
                "passed": False,
                "error": f"检测到敏感信息: {detected_pii}",
                "fatal": True,
            }
        
        # 注入攻击检测
        injection_patterns = [
            r"ignore.*previous.*instructions",
            r"you are now.*",
            r"system:\s*",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return {
                    "layer": "safety",
                    "passed": False,
                    "error": "检测到潜在的注入攻击",
                    "fatal": True,
                }
        
        return {"layer": "safety", "passed": True}
    
    def _validate_quality(self, output: str, context: Dict) -> Dict:
        """Layer 3: 质量验证"""
        # 幻觉检测 (基于引用一致性)
        references = context.get("references", [])
        if references:
            # 简单检查: 输出中的数字是否在引用范围内
            numbers_in_output = re.findall(r'\d+\.?\d*', output)
            # 实际应用中应更复杂
        
        # 相关性检查 (简单关键词匹配)
        required_keywords = context.get("required_keywords", [])
        if required_keywords:
            missing = [kw for kw in required_keywords if kw not in output]
            if missing:
                return {
                    "layer": "quality",
                    "passed": False,
                    "error": f"缺少关键词: {missing}",
                    "fatal": False,
                }
        
        return {"layer": "quality", "passed": True}
    
    def _validate_business(self, output: str, context: Dict) -> Dict:
        """Layer 4: 业务验证"""
        # 业务规则由具体应用定义
        rules = context.get("business_rules", [])
        
        for rule in rules:
            if not rule(output):
                return {
                    "layer": "business",
                    "passed": False,
                    "error": f"业务规则未满足: {rule.__name__}",
                    "fatal": False,
                }
        
        return {"layer": "business", "passed": True}
```

## 五、混沌工程与韧性验证

### 5.1 AI应用混沌实验设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI应用混沌实验矩阵                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────┬───────────────────┬───────────────────┐     │
│  │  故障类型         │  实验方法          │  预期行为          │     │
│  ├───────────────────┼───────────────────┼───────────────────┤     │
│  │  LLM API延迟注入  │  tc netem delay   │  触发超时降级      │     │
│  │  LLM API不可用    │  iptables DROP    │  熔断器打开        │     │
│  │  Token配额耗尽    │  Mock返回429      │  切换备用模型      │     │
│  │  网络分区         │  iptables隔离     │  本地缓存响应      │     │
│  │  内存压力         │  stress-ng        │  优雅降级          │     │
│  │  模型输出异常     │  Mock异常输出     │  输出验证拦截      │     │
│  └───────────────────┴───────────────────┴───────────────────┘     │
│                                                                      │
│  实验频率: 每周一次核心路径实验                                        │
│  实验环境: 预生产环境 (与生产环境一致)                                 │
│  爆炸半径: 从10%流量开始，逐步扩大                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 韧性度量指标

| 指标 | 定义 | 目标值 | 计算方式 |
|------|------|--------|----------|
| **可用性** | 系统正常运行时间占比 | ≥ 99.9% | (总时间 - 故障时间) / 总时间 |
| **恢复时间 (MTTR)** | 从故障到恢复的平均时间 | < 5分钟 | 故障恢复总时间 / 故障次数 |
| **降级成功率** | 降级时仍能返回有效结果的比例 | ≥ 95% | 降级成功次数 / 降级触发次数 |
| **熔断准确性** | 熔断器正确判断故障的比例 | ≥ 90% | 正确熔断次数 / 总熔断次数 |
| **重试成功率** | 重试后成功的比例 | ≥ 80% | 重试成功次数 / 总重试次数 |
| **用户体验保持** | 降级时用户体验评分 | ≥ 4.0/5.0 | 用户满意度调查 |

## 六、生产部署检查清单

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI应用韧性架构部署检查清单                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ✅ 基础设施层                                                       │
│  □ 多区域部署 (至少2个可用区)                                        │
│  □ 容器健康检查配置 (liveness + readiness)                           │
│  □ 自动扩缩容策略 (基于GPU利用率和队列长度)                          │
│  □ 数据持久化和备份                                                  │
│                                                                      │
│  ✅ 服务层                                                           │
│  □ 熔断器配置 (阈值、超时、恢复策略)                                  │
│  □ 重试策略 (最大次数、退避算法、Token预算)                           │
│  □ 超时控制 (连接超时、读取超时、总超时)                              │
│  □ 降级策略 (至少3级降级方案)                                         │
│                                                                      │
│  ✅ AI模型层                                                         │
│  □ 多模型容灾 (主备模型配置)                                         │
│  □ 输出验证流水线 (格式、安全、质量、业务)                            │
│  □ 模型版本管理 (支持快速回退)                                       │
│  □ 幻觉检测机制                                                      │
│                                                                      │
│  ✅ 监控层                                                           │
│  □ 延迟监控 (P50, P95, P99)                                         │
│  □ 错误率监控 (按错误类型分类)                                       │
│  □ Token使用量监控                                                   │
│  □ 降级事件告警                                                      │
│  □ 熔断器状态监控                                                    │
│                                                                      │
│  ✅ 混沌工程                                                          │
│  □ 定期故障注入实验 (至少每月一次)                                    │
│  □ 故障注入工具就绪 (Chaos Mesh / Litmus)                            │
│  □ 实验结果记录和改进                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 七、总结

AI应用的韧性架构设计是一个系统工程，需要从多个层次进行防护。核心要点：

1. **熔断器是第一道防线**：防止故障级联，但需要针对LLM的特殊性（延迟高、错误类型多样）进行适配。

2. **智能重试比简单重试更重要**：Token预算感知、指数退避、错误分类，这些细节能显著提升重试成功率。

3. **降级策略要有层次**：从延迟优化到功能简化，从模型切换到静态兜底，每层都有明确的触发条件和恢复策略。

4. **输出验证是质量保障**：LLM的不确定性要求我们对输出进行多层验证，不能盲目信任模型输出。

5. **混沌工程是验证手段**：通过定期故障注入，验证韧性架构的有效性，发现潜在的薄弱环节。

韧性架构不是一次性工作，而是持续演进的过程。随着AI应用的复杂度增加，韧性设计也需要不断迭代优化。
