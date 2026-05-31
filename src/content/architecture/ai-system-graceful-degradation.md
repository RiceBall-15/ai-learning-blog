---
title: "AI系统的故障恢复与优雅降级架构：从单点失败到全链路韧性的设计模式"
description: "深入剖析AI应用中LLM服务不稳定、API限流、模型幻觉等典型故障场景，系统性讲解重试策略、降级方案、熔断机制与多模型故障转移的生产级架构设计。"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: "distributed"
tags: ["系统架构", "故障恢复", "优雅降级", "AI系统韧性", "熔断器", "多模型路由"]
draft: false
---

## 引言：为什么AI系统需要"优雅降级"？

在传统微服务架构中，一个接口超时了，最多返回一个兜底页面。但在AI应用中，情况远比这复杂——

> 你的LLM API突然返回500，但用户正在等着AI生成方案；你的Embedding模型限流了，但RAG检索是核心链路；你的多模态模型幻觉了，但业务决策已经依赖了它的输出……

AI系统的**故障域**比传统服务多了几个数量级：网络抖动、Token超限、上下文溢出、模型幻觉、API配额耗尽、推理超时……任何一个环节出问题，都可能导致用户体验断崖式下降。

本文将系统性地拆解AI系统的故障模式，并给出一套**从单点恢复到全链路韧性**的架构设计模式。

---

## 一、AI系统故障全景图

在设计恢复策略之前，我们先来画一张完整的故障地图：

| 故障层级 | 故障类型 | 影响范围 | 恢复难度 |
|---------|---------|---------|---------|
| **网络层** | API超时、DNS解析失败、连接池耗尽 | 单个请求 | ⭐ |
| **服务层** | LLM API限流(429)、服务不可用(5xx) | 单个模型 | ⭐⭐ |
| **协议层** | SSE流中断、响应格式异常、Token截断 | 流式输出 | ⭐⭐ |
| **语义层** | 模型幻觉、输出偏离意图、格式不合规 | 业务正确性 | ⭐⭐⭐ |
| **资源层** | Token配额耗尽、内存溢出、GPU OOM | 全局 | ⭐⭐⭐⭐ |
| **链路层** | RAG检索失败、向量库连接异常、知识库过期 | 整条链路 | ⭐⭐⭐⭐ |

**关键洞察**：传统系统的故障大多是二元的——要么成功，要么失败。而AI系统存在大量的**灰度故障**：请求成功了，但结果质量很差。这要求我们的降级策略不仅关注可用性，还要关注**输出质量**。

---

## 二、分级重试策略：不是所有失败都值得重试

### 2.1 错误分类与重试决策矩阵

```python
from enum import Enum
from dataclasses import dataclass

class ErrorCategory(Enum):
    TRANSIENT = "transient"       # 网络抖动、临时过载
    RATE_LIMITED = "rate_limited" # 429 限流
    QUOTA = "quota"               # 配额耗尽（不可重试）
    SEMANTIC = "semantic"         # 输出质量问题
    FATAL = "fatal"               # 认证失败、参数错误

# 不同错误类型的重试策略
RETRY_POLICY = {
    ErrorCategory.TRANSIENT: {
        "max_retries": 3,
        "backoff": "exponential",    # 指数退避
        "base_delay": 0.5,           # 0.5s
        "max_delay": 10.0,           # 最大10s
        "jitter": True,              # 随机抖动
    },
    ErrorCategory.RATE_LIMITED: {
        "max_retries": 5,
        "backoff": "respect_header", # 尊重 Retry-After 头
        "base_delay": 2.0,
        "max_delay": 60.0,
        "jitter": False,
    },
    ErrorCategory.QUOTA: {
        "max_retries": 0,            # 不重试，直接降级
    },
    ErrorCategory.SEMANTIC: {
        "max_retries": 2,            # 允许少量重试（配合重新prompt）
        "backoff": "fixed",
        "base_delay": 0,
    },
    ErrorCategory.FATAL: {
        "max_retries": 0,
    },
}
```

### 2.2 重试的关键反模式

**❌ 反模式1：无差别的盲目重试**

```python
# 经典错误：对所有失败统一重试
for i in range(3):
    try:
        response = llm.call(prompt)
        break
    except Exception:
        continue  # 如果是配额耗尽，重试多少次都没用
```

**✅ 正确做法：错误感知的条件重试**

```python
async def resilient_llm_call(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.acall(prompt)
            
            # 语义级重试：检查输出质量
            if is_low_quality(response):
                prompt = enrich_prompt_with_feedback(prompt, response)
                logger.warning(f"质量不达标，重试 {attempt + 1}")
                continue
            
            return response
            
        except RateLimitError as e:
            wait_time = e.retry_after or calculate_backoff(attempt)
            await asyncio.sleep(wait_time)
            
        except QuotaExhaustedError:
            # 配额耗尽，立即降级，不浪费时间重试
            return await fallback_to_alternative(prompt)
            
        except ContextLengthExceededError:
            # 上下文超长，截断后重试
            prompt = truncate_to_fit(prompt)
            
    # 所有重试用尽，进入降级流程
    return await degrade_gracefully(prompt)
```

---

## 三、熔断器模式：保护下游，也保护自己

### 3.1 传统熔断器 vs AI场景熔断器

传统熔断器（如Hystrix）基于错误率做决策。但在AI系统中，我们需要更细腻的信号：

```python
@dataclass
class AICircuitBreakerConfig:
    # 传统指标
    error_threshold: float = 0.5        # 错误率阈值
    slow_call_threshold: float = 30.0   # 慢调用阈值(秒)
    wait_duration: int = 30             # 熔断恢复等待(秒)
    
    # AI特有指标
    quality_threshold: float = 0.6      # 输出质量阈值
    hallucination_rate: float = 0.1     # 幻觉率阈值
    token_budget_exceeded: bool = False # Token预算是否耗尽
    
    # 动态调整
    half_open_max_calls: int = 3        # 半开状态最大探测次数
```

### 3.2 状态机设计

```
         调用成功 & 质量达标
    ┌──────────────────────────────┐
    │                              ▼
 ┌──────┐    错误率/幻觉率超阈值   ┌────────┐
 │ CLOSED│ ──────────────────────► │  OPEN  │
 └──────┘                          └────────┘
    ▲                                  │
    │  探测调用成功                      │ 等待时间到达
    │                                  ▼
    │                            ┌──────────┐
    └──────────────────────────── │ HALF-OPEN│
         探测调用失败，回滚到OPEN   └──────────┘
```

```python
class AICircuitBreaker:
    """AI场景专用熔断器"""
    
    def __init__(self, name: str, config: AICircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        
        # AI特有：质量追踪
        self.quality_scores: deque = deque(maxlen=100)
        self.hallucination_count = 0
    
    async def execute(self, call_fn, fallback_fn):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF-OPEN"
                self.half_open_calls = 0
            else:
                return await fallback_fn()
        
        try:
            result = await call_fn()
            self._on_success(result)
            return result
        except Exception as e:
            self._on_failure(e)
            return await fallback_fn()
    
    def _on_success(self, result):
        self.success_count += 1
        self.total_calls += 1
        
        # 追踪输出质量
        if hasattr(result, 'quality_score'):
            self.quality_scores.append(result.quality_score)
        
        if self.state == "HALF-OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = "CLOSED"
                self._reset_counters()
    
    def _on_failure(self, error):
        self.failure_count += 1
        self.total_calls += 1
        self.last_failure_time = time.time()
        
        # 检查是否触发熔断
        if self._should_trip():
            self.state = "OPEN"
    
    def _should_trip(self) -> bool:
        # 1. 错误率触发
        if self.total_calls >= 10:
            error_rate = self.failure_count / self.total_calls
            if error_rate > self.config.error_threshold:
                return True
        
        # 2. 幻觉率触发（AI特有）
        if len(self.quality_scores) >= 10:
            avg_quality = sum(self.quality_scores) / len(self.quality_scores)
            if avg_quality < self.config.quality_threshold:
                return True
        
        return False
```

---

## 四、多模型故障转移：让模型成为可互换的组件

### 4.1 设计理念：模型即服务

最优雅的降级策略是：**让用户完全感知不到故障的发生**。这需要我们将多个模型组织成一个层级化的服务架构：

```
┌─────────────────────────────────────────────┐
│                 用户请求                       │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│           模型路由层 (Model Router)           │
│  ┌───────────┬───────────┬───────────────┐  │
│  │  策略引擎  │  健康检查  │  质量评估模块   │  │
│  └───────────┴───────────┴───────────────┘  │
└─────────┬───────────┬───────────┬───────────┘
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Tier 1  │ │  Tier 2  │ │  Tier 3  │
    │ GPT-4o   │ │ Claude   │ │ Gemini   │
    │ (首选)   │ │ (备选)   │ │ (兜底)   │
    └──────────┘ └──────────┘ └──────────┘
```

### 4.2 实现：智能模型路由

```python
from dataclasses import dataclass, field
from typing import Optional
import asyncio

@dataclass
class ModelEndpoint:
    name: str
    tier: int                              # 优先级：1=首选，2=备选，3=兜底
    client: LLMClient
    cost_per_1k_tokens: float              # 成本（用于成本感知路由）
    max_context_length: int                # 最大上下文
    capabilities: list[str]                # 能力标签: ["reasoning", "vision", "code"]
    
    # 运行时状态
    circuit_breaker: AICircuitBreaker = field(default_factory=None)
    avg_latency_ms: float = 0.0
    avg_quality_score: float = 1.0

class ModelRouter:
    """多模型智能路由器"""
    
    def __init__(self, endpoints: list[ModelEndpoint]):
        # 按tier排序，同tier按成本升序
        self.endpoints = sorted(endpoints, key=lambda e: (e.tier, e.cost_per_1k_tokens))
    
    async def route(
        self,
        prompt: str,
        required_capabilities: list[str] = None,
        max_cost_per_request: float = None,
        timeout: float = 30.0,
    ) -> str:
        """智能路由：自动选择最优可用模型"""
        
        # 筛选满足要求的端点
        candidates = self._filter_endpoints(
            required_capabilities, max_cost_per_request
        )
        
        last_error = None
        for endpoint in candidates:
            if endpoint.circuit_breaker and endpoint.circuit_breaker.state == "OPEN":
                continue  # 跳过已熔断的模型
            
            try:
                result = await asyncio.wait_for(
                    endpoint.client.acall(prompt),
                    timeout=timeout
                )
                
                # 后置质量检查
                quality = self._assess_quality(result)
                if quality < 0.3:
                    # 质量太差，尝试下一个模型
                    continue
                
                # 更新质量分数（滑动平均）
                endpoint.avg_quality_score = (
                    0.7 * endpoint.avg_quality_score + 0.3 * quality
                )
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"{endpoint.name}: timeout"
                continue
            except RateLimitError as e:
                # 触发熔断
                if endpoint.circuit_breaker:
                    endpoint.circuit_breaker._on_failure(e)
                last_error = f"{endpoint.name}: rate limited"
                continue
            except Exception as e:
                last_error = f"{endpoint.name}: {str(e)}"
                continue
        
        # 所有模型都失败了
        raise AllModelsExhaustedError(
            f"所有模型均不可用。最后错误: {last_error}"
        )
    
    def _filter_endpoints(self, capabilities, max_cost):
        candidates = []
        for ep in self.endpoints:
            if capabilities and not set(capabilities).issubset(set(ep.capabilities)):
                continue
            if max_cost and ep.cost_per_1k_tokens > max_cost:
                continue
            candidates.append(ep)
        return candidates
```

### 4.3 模型选择策略对比

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **固定层级** | 稳定性优先 | 简单可靠 | 可能浪费高质量模型资源 |
| **成本优先** | 批量处理 | 成本最优 | 可能牺牲质量 |
| **质量感知** | 核心业务 | 输出质量有保障 | 需要质量评估模型 |
| **延迟感知** | 实时交互 | 响应速度最优 | 可能选到质量一般的模型 |
| **A/B轮转** | 模型评估 | 持续积累数据 | 短期体验不稳定 |

---

## 五、RAG链路的优雅降级

RAG系统涉及多个环节，每个环节都可能失败。下面是每个环节的降级策略：

### 5.1 链路全景与降级策略

```
用户Query
    │
    ▼
┌──────────┐     降级策略：直接跳过检索，使用LLM内部知识
│ Query改写 │─────────────────────────────────────────────┐
└──────────┘                                              │
    │                                                     │
    ▼                                                     │
┌──────────┐     降级策略：使用缓存的旧结果               │
│ 向量检索  │─────────────────────────────────────────┐   │
└──────────┘                                         │   │
    │                                               │   │
    ▼                                               │   │
┌──────────┐     降级策略：降级到BM25关键词检索      │   │
│ 重排序   │────────────────────────────────────┐   │   │
└──────────┘                                   │   │   │
    │                                         │   │   │
    ▼                                         │   │   │
┌──────────┐     降级策略：减少Context长度      │   │   │
│ Context  │────────────────────────────────┐  │   │   │
│ 组装     │                                │  │   │   │
└──────────┘                                │  │   │   │
    │                                       │  │   │   │
    ▼                                       ▼  ▼   ▼   ▼
┌──────────────────────────────────────────────────────┐
│                   LLM 生成                           │
│  ┌──────────────────────────────────────────────┐   │
│  │ 多模型路由 + 后置质量检查 + 重试 + 降级        │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### 5.2 降级实现示例

```python
class ResilientRAGPipeline:
    """带完整降级能力的RAG管道"""
    
    async def query(self, user_query: str) -> str:
        # 阶段1：检索（带降级）
        context = await self._retrieve_with_fallback(user_query)
        
        # 阶段2：生成（带降级）
        answer = await self._generate_with_fallback(user_query, context)
        
        return answer
    
    async def _retrieve_with_fallback(self, query: str) -> str:
        """检索降级链：向量检索 → 缓存 → BM25 → 空上下文"""
        
        # Level 1: 完整向量检索
        try:
            results = await asyncio.wait_for(
                self.vector_store.search(query, top_k=5),
                timeout=5.0
            )
            if results and results[0].score > 0.7:
                return self._format_context(results)
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")
        
        # Level 2: 使用缓存的检索结果
        cached = await self.cache.get(f"search:{hash(query)}")
        if cached:
            logger.info("使用缓存的检索结果")
            return cached
        
        # Level 3: 降级到BM25关键词检索
        try:
            bm25_results = await self.bm25_index.search(query, top_k=5)
            if bm25_results:
                return self._format_context(bm25_results)
        except Exception as e:
            logger.warning(f"BM25检索也失败: {e}")
        
        # Level 4: 无上下文生成（纯LLM知识）
        logger.warning("所有检索方式均失败，降级为无上下文生成")
        return ""
    
    async def _generate_with_fallback(self, query: str, context: str) -> str:
        """生成降级链：完整RAG → 截断Context → 无Context → 兜底模板"""
        
        # Level 1: 完整RAG生成
        try:
            prompt = self._build_prompt(query, context)
            result = await self.model_router.route(
                prompt,
                required_capabilities=["reasoning"],
                timeout=15.0
            )
            return result
        except AllModelsExhaustedError:
            logger.warning("所有模型均不可用")
        
        # Level 2: 截断Context后重试
        try:
            truncated_context = self._truncate_context(context, max_tokens=2000)
            prompt = self._build_prompt(query, truncated_context)
            result = await self.model_router.route(
                prompt,
                timeout=10.0
            )
            return result
        except Exception:
            pass
        
        # Level 3: 纯LLM生成（无RAG）
        try:
            prompt = f"请回答以下问题：\n{query}"
            result = await self.model_router.route(prompt, timeout=10.0)
            return f"⚠️ 以下回答未参考知识库：\n\n{result}"
        except Exception:
            pass
        
        # Level 4: 兜底模板
        return (
            "抱歉，AI服务当前不可用。请稍后重试，或联系人工客服获取帮助。"
        )
```

---

## 六、Token预算管理与成本保护

AI系统的另一个关键维度是**成本保护**。Token配额耗尽是一个常见的"不可恢复"故障，必须提前预防。

### 6.1 多层Token预算架构

```
┌──────────────────────────────────────┐
│         全局Token预算 (月度)           │
│         $10,000 / 月                  │
├──────────────────────────────────────┤
│  ┌────────────┐  ┌────────────────┐  │
│  │ 业务线预算  │  │  业务线预算     │  │
│  │ $4,000/月  │  │  $6,000/月     │  │
│  ├────────────┤  ├────────────────┤  │
│  │ 用户级配额  │  │  用户级配额    │  │
│  │ 100K tok/d │  │  200K tok/d    │  │
│  ├────────────┤  ├────────────────┤  │
│  │ 请求级预算  │  │  请求级预算    │  │
│  │ 8K tok/req │  │  16K tok/req   │  │
│  └────────────┘  └────────────────┘  │
└──────────────────────────────────────┘
```

### 6.2 Token预算检查器

```python
class TokenBudgetGuard:
    """Token预算守卫：在请求发出前检查预算"""
    
    def __init__(self, storage: BudgetStorage):
        self.storage = storage
    
    async def check_and_consume(
        self,
        user_id: str,
        business_line: str,
        estimated_tokens: int,
    ) -> tuple[bool, str]:
        """检查是否允许消费，返回 (是否允许, 原因)"""
        
        # 1. 检查请求级预算
        if estimated_tokens > REQUEST_TOKEN_LIMIT:
            return False, f"单次请求Token数({estimated_tokens})超过上限({REQUEST_TOKEN_LIMIT})"
        
        # 2. 检查用户日配额
        user_daily = await self.storage.get_user_daily_usage(user_id)
        if user_daily + estimated_tokens > USER_DAILY_LIMIT:
            remaining = USER_DAILY_LIMIT - user_daily
            return False, f"今日剩余额度不足，剩余 {remaining} tokens。请明日再试。"
        
        # 3. 检查业务线月配额
        line_monthly = await self.storage.get_line_monthly_usage(business_line)
        if line_monthly + estimated_tokens > LINE_MONTHLY_LIMIT:
            return False, f"业务线月度配额即将耗尽，已触发保护机制。"
        
        # 4. 检查全局预算预警线
        global_usage = await self.storage.get_global_monthly_usage()
        if global_usage > GLOBAL_WARNING_THRESHOLD:
            # 预警阈值达到后，自动降级到更便宜的模型
            return True, "BUDGET_WARNING_USE_CHEAPER_MODEL"
        
        return True, "OK"
    
    async def record_usage(self, user_id: str, business_line: str, actual_tokens: int):
        """记录实际消费"""
        await self.storage.record(user_id, business_line, actual_tokens)
```

---

## 七、端到端韧性架构模式

### 7.1 完整的韧性中间件栈

```python
class ResilientAIMiddleware:
    """
    端到端的AI韧性中间件，组合所有防护机制
    请求经过的顺序：预算检查 → 限流 → 熔断检查 → 调用 → 质量检查 → 监控上报
    """
    
    def __init__(self):
        self.budget_guard = TokenBudgetGuard(...)
        self.rate_limiter = AdaptiveRateLimiter(...)
        self.model_router = ModelRouter(...)
        self.quality_checker = OutputQualityChecker(...)
        self.metrics = AIMetricsCollector(...)
    
    async def handle(self, request: AIRequest) -> AIResponse:
        start_time = time.time()
        metadata = {"request_id": request.id, "model": None}
        
        try:
            # ① 预算检查
            allowed, reason = await self.budget_guard.check_and_consume(
                request.user_id,
                request.business_line,
                request.estimated_tokens,
            )
            if not allowed:
                return AIResponse.declined(reason)
            
            use_cheaper = reason == "BUDGET_WARNING_USE_CHEAPER_MODEL"
            
            # ② 自适应限流
            if not await self.rate_limiter.acquire(request.user_id):
                return AIResponse.rate_limited(retry_after=30)
            
            # ③ 模型选择与调用（内含熔断检查和降级）
            model_name = "economy" if use_cheaper else "standard"
            result = await self.model_router.route(
                request.prompt,
                preferred_model=model_name,
                timeout=request.timeout,
            )
            metadata["model"] = result.model_name
            
            # ④ 后置质量检查
            quality = await self.quality_checker.evaluate(result.text)
            if quality.score < 0.2:
                # 质量极低，尝试用备选模型重新生成
                result = await self.model_router.route(
                    request.prompt,
                    exclude_models=[result.model_name],
                    timeout=request.timeout,
                )
                metadata["retried"] = True
            
            # ⑤ 记录消费
            await self.budget_guard.record_usage(
                request.user_id, request.business_line, result.token_usage
            )
            
            return AIResponse.success(
                text=result.text,
                metadata=metadata,
                quality_score=quality.score,
            )
            
        except Exception as e:
            metadata["error"] = str(e)
            return AIResponse.error(str(e), metadata=metadata)
            
        finally:
            # ⑥ 监控上报（始终执行）
            elapsed = time.time() - start_time
            self.metrics.record(
                latency=elapsed,
                model=metadata.get("model"),
                success=metadata.get("error") is None,
            )
```

### 7.2 架构模式总结

| 模式 | 解决的问题 | 核心机制 | 实施复杂度 |
|------|-----------|---------|-----------|
| **错误感知重试** | 瞬时故障 | 按错误类型差异化重试 | ⭐⭐ |
| **AI熔断器** | 下游不可用 | 错误率+质量双维度判断 | ⭐⭐⭐ |
| **多模型路由** | 单模型故障 | 分层降级+智能选择 | ⭐⭐⭐ |
| **RAG降级链** | 检索链路故障 | 多级降级+兜底生成 | ⭐⭐⭐⭐ |
| **Token预算保护** | 成本失控 | 多层预算+自动降级 | ⭐⭐⭐ |
| **韧性中间件栈** | 全链路韧性 | 组合模式+统一编排 | ⭐⭐⭐⭐⭐ |

---

## 八、生产环境中的最佳实践

### 8.1 监控与告警

AI系统的监控需要覆盖多个维度：

```
┌─────────────────────── 可观测性仪表盘 ───────────────────────┐
│                                                                │
│  [可用性]    请求成功率: 99.7%  ▓▓▓▓▓▓▓▓▓░ 99.7%              │
│  [质量]      平均质量分: 0.87   ▓▓▓▓▓▓▓▓░░ 87%                │
│  [延迟]      P99延迟: 3.2s     ▓▓▓▓▓▓▓░░░ 3.2s               │
│  [成本]      今日Token消费: 1.2M  预算剩余: 67%                │
│  [降级]      降级率: 2.3%      ▓▓▓░░░░░░░ 2.3%               │
│  [熔断]      活跃熔断: 0个     全部正常                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 关键告警规则

| 告警条件 | 严重级别 | 处理方式 |
|---------|---------|---------|
| 请求成功率 < 95% | 🔴 P1 | 立即排查，可能需要切换备用模型 |
| 质量分 < 0.6 持续10分钟 | 🟡 P2 | 检查是否有模型版本更新或Prompt漂移 |
| 降级率 > 10% | 🟡 P2 | 检查上游模型健康状态 |
| Token消费突增200% | 🟡 P2 | 排查是否有异常请求或爬虫 |
| P99延迟 > 10s | 🟠 P3 | 评估是否需要优化Prompt或切换更快模型 |
| 熔断器打开 | 🟠 P3 | 检查对应模型服务状态 |

### 8.3 演练与验证

韧性架构不是设计出来就完事的，需要定期**混沌工程演练**：

```python
class ChaosExperiment:
    """AI系统的混沌工程实验"""
    
    async def experiment_model_outage(self):
        """模拟主模型完全不可用"""
        # 随机选择一个Tier-1模型，注入故障
        target = self.get_random_tier1_model()
        await self.fault_injector.inject(target, "service_unavailable")
        
        # 观察：降级是否生效、用户是否感知
        metrics = await self.run_traffic(duration_seconds=60)
        
        assert metrics.success_rate > 0.99, "降级后成功率应>99%"
        assert metrics.user_visible_errors == 0, "用户不应看到任何错误"
        assert metrics.fallback_model_used > 0, "降级模型应该被使用"
        
        # 恢复
        await self.fault_injector.recover(target)
    
    async def experiment_rate_limit(self):
        """模拟API限流"""
        await self.fault_injector.inject(
            self.primary_model, 
            "rate_limit", 
            retry_after=5
        )
        
        metrics = await self.run_traffic(duration_seconds=120)
        
        # 应该看到429被正确处理，请求被路由到其他模型
        assert metrics.total_429_count > 0, "应该有429触发"
        assert metrics.fallback_used > 0, "应该有降级"
```

---

## 九、总结

AI系统的韧性设计与传统微服务有本质区别。我们总结核心要点：

**1. 故障维度更多**：除了传统的网络和服务故障，还需要处理模型幻觉、质量退化、Token耗尽等AI特有问题。

**2. 降级要分层**：每一层（检索、生成、后处理）都应该有独立的降级策略，形成完整的降级链。

**3. 质量也是故障**：输出质量低于阈值应该和请求失败一样被严肃对待，触发重试或降级。

**4. 成本是硬约束**：Token预算耗尽是不可恢复的故障，必须提前预防和控制。

**5. 模型应该可互换**：通过统一接口和智能路由，让多个模型成为可互换的组件，任何一个出问题都能快速切换。

**6. 演练是必须的**：通过混沌工程定期验证韧性机制的有效性，而不是等到线上出问题才发现降级逻辑有bug。

在AI应用成为核心业务基础设施的今天，**系统韧性不是锦上添花，而是生存必需**。
