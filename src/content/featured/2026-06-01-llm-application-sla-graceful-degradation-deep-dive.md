---
title: "LLM应用的SLA保障与优雅降级：从熔断器到多级缓存的生产实战"
description: "深入剖析LLM应用在生产环境中如何构建完整的SLA保障体系，涵盖延迟监控、熔断降级、多级缓存和多模型回退等核心策略，附完整架构代码。"
date: 2026-06-01
author: "RiceBall"
category: "deep-dive"
tags: ["LLM", "SLA", "优雅降级", "熔断器", "生产实战", "可靠性工程", "AI架构"]
draft: false
---

## 引言：LLM应用的可靠性困境

在传统微服务架构中，SLA（Service Level Agreement）保障体系已经非常成熟——超时有熔断器，过载有限流器，异常有重试策略。但当我们将LLM引入生产系统后，这些传统手段遇到了前所未有的挑战：

| 挑战维度 | 传统服务 | LLM服务 |
|---------|---------|---------|
| 响应延迟 | P99 < 500ms | P99 可达 30-120s |
| 延迟方差 | 低（毫秒级波动） | 极高（秒级到分钟级波动） |
| 资源消耗 | 线性可预测 | GPU显存波动大，batch效应明显 |
| 失败模式 | 超时/5xx | 超时/幻觉/截断/内容违规 |
| 成本结构 | 按请求计费 | 按Token计费，一次调用成本可能数十元 |
| 降级选择 | 有限（缓存/静态响应） | 丰富（小模型/规则/缓存/摘要） |

一个典型的LLM应用线上故障场景：

```
用户提问 → 大模型调用超时（30s）→ 无降级策略 → 用户等待 → 体验崩溃
                                    ↓
                           重试风暴 → GPU集群过载 → 级联故障
```

本文将分享我们在生产环境中构建LLM应用SLA保障体系的完整经验，从监控体系建设到多层降级策略，提供一套可直接落地的解决方案。

---

## 一、LLM应用SLA指标体系设计

### 1.1 核心SLA指标

LLM应用的SLA指标需要超越传统的延迟/吞吐/可用性三维框架：

```
┌─────────────────────────────────────────────────────────────┐
│                LLM应用 SLA 指标体系                          │
├─────────────┬───────────────────────────────────────────────┤
│  可用性指标   │  延迟指标           │  质量指标                │
│  · 成功率    │  · TTFT（首Token）  │  · 幻觉率               │
│  · 错误率    │  · TPOT（Token/秒） │  · 相关性得分            │
│  · 降级率    │  · 总延迟            │  · 安全合规率            │
│  · 超时率    │  · P50/P95/P99     │  · 任务完成率            │
├─────────────┼───────────────────────────────────────────────┤
│  资源指标     │  成本指标            │  业务指标                │
│  · GPU利用率  │  · 每次请求成本      │  · 用户满意度            │
│  · 并发槽位  │  · 每Token成本       │  · 任务成功率            │
│  · 队列深度  │  · 成本/营收比       │  · 用户留存率            │
└─────────────┴───────────────────────────────────────────────┘
```

### 1.2 关键指标定义

```python
from dataclasses import dataclass
from enum import Enum

class LLMSLAMetrics:
    """LLM应用核心SLA指标"""

    # === 延迟指标 ===
    TTFT_P95 = "2.0s"        # 首Token延迟 P95
    TTFT_P99 = "5.0s"        # 首Token延迟 P99
    TPOT_P50 = "30 tokens/s" # 每Token生成速度 P50
    TPOT_P99 = "15 tokens/s" # 每Token生成速度 P99
    TOTAL_LATENCY_P95 = "15s" # 总延迟 P95

    # === 可用性指标 ===
    SUCCESS_RATE = "99.5%"    # 成功率
    TIMEOUT_RATE_MAX = "1%"   # 最大超时率
    DEGRADATION_RATE_MAX = "5%" # 最大降级率

    # === 质量指标 ===
    HALLUCINATION_RATE_MAX = "3%"    # 最大幻觉率
    RELEVANCE_SCORE_MIN = "0.75"     # 最低相关性分数
    SAFETY_COMPLIANCE_RATE = "99.9%" # 安全合规率

    # === 成本指标 ===
    AVG_COST_PER_REQUEST = "¥0.05"   # 单次请求平均成本
    MONTHLY_BUDGET = "¥50,000"       # 月度预算上限
```

### 1.3 监控告警分级

| 级别 | 触发条件 | 响应时间 | 处理方式 |
|------|---------|---------|---------|
| P0-致命 | 成功率 < 95% 或 P99 > 60s | 5分钟 | 自动降级 + 人工介入 |
| P1-严重 | TTFT_P95 > 5s 或错误率 > 3% | 15分钟 | 自动降级 + 通知值班 |
| P2-警告 | TPOT_P50 < 20 tokens/s | 30分钟 | 通知值班 + 趋势分析 |
| P3-信息 | 成本偏离预算 > 20% | 1工作日 | 周报分析 + 优化建议 |

---

## 二、LLM专用熔断器设计

### 2.1 为什么传统熔断器不够用

传统熔断器（如Hystrix/Resilience4j）基于简单的错误率阈值触发，但LLM场景需要更精细的策略：

| 判据 | 传统熔断器 | LLM熔断器 |
|------|-----------|-----------|
| 触发条件 | 错误率 > 50% | 多维综合判定 |
| 冷却时间 | 固定（如30s） | 自适应（基于负载） |
| 降级策略 | 静态降级 | 动态多级降级 |
| 恢复策略 | 半开探测 | 渐进式恢复 |
| 队列处理 | 丢弃/排队 | 智能分流 |

### 2.2 多维熔断器实现

```python
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

class CircuitState(Enum):
    CLOSED = "closed"       # 正常放行
    OPEN = "open"           # 拒绝请求
    HALF_OPEN = "half_open" # 试探性放行

@dataclass
class LLMCallRecord:
    timestamp: float
    latency: float
    success: bool
    token_count: int
    cost: float
    model: str

class LLMMultiDimensionalCircuitBreaker:
    """
    LLM多维熔断器 — 基于延迟、错误率、成本三维度联合判定

    与传统熔断器的区别：
    1. 不仅看错误率，还看延迟恶化和成本飙升
    2. 冷却时间基于实时负载动态调整
    3. 支持多级降级策略（模型降级 → 缓存降级 → 规则降级）
    """

    def __init__(
        self,
        failure_rate_threshold: float = 0.3,       # 错误率阈值
        slow_call_rate_threshold: float = 0.5,      # 慢调用率阈值
        cost_spike_threshold: float = 3.0,          # 成本飙升倍数
        window_size: int = 100,                     # 滑动窗口大小
        base_cooldown: float = 30.0,                # 基础冷却时间(秒)
        max_cooldown: float = 300.0,                # 最大冷却时间
    ):
        self.failure_rate_threshold = failure_rate_threshold
        self.slow_call_rate_threshold = slow_call_rate_threshold
        self.cost_spike_threshold = cost_spike_threshold
        self.window_size = window_size
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown

        self.state = CircuitState.CLOSED
        self.records: deque[LLMCallRecord] = deque(maxlen=window_size)
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.current_cooldown = base_cooldown
        self.baseline_cost = 0.0

    def record_call(self, record: LLMCallRecord):
        """记录一次LLM调用结果"""
        self.records.append(record)

        # 更新基线成本（滑动平均）
        if self.baseline_cost == 0:
            self.baseline_cost = record.cost
        else:
            self.baseline_cost = 0.9 * self.baseline_cost + 0.1 * record.cost

        if record.success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        # 在CLOSED状态下持续评估是否需要熔断
        if self.state == CircuitState.CLOSED:
            self._evaluate_transition()

    def should_allow(self) -> bool:
        """判断当前是否应该放行请求"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # 检查冷却时间是否已过
            if time.time() - self.opened_at >= self.current_cooldown:
                self.state = CircuitState.HALF_OPEN
                return True  # 半开状态放行一个试探请求
            return False

        if self.state == CircuitState.HALF_OPEN:
            # 半开状态只放行少量请求
            return True

        return False

    def _evaluate_transition(self):
        """多维度评估是否需要熔断"""
        if len(self.records) < 10:  # 样本不足，不触发
            return

        recent = list(self.records)[-min(50, len(self.records)):]

        # 维度1：错误率
        error_rate = sum(1 for r in recent if not r.success) / len(recent)

        # 维度2：慢调用率（延迟 > 20s 视为慢调用）
        slow_rate = sum(1 for r in recent if r.latency > 20.0) / len(recent)

        # 维度3：成本飙升
        avg_cost = sum(r.cost for r in recent) / len(recent)
        cost_spike = avg_cost / self.baseline_cost if self.baseline_cost > 0 else 1.0

        # 综合判定：任一维度超过阈值即触发熔断
        should_trip = (
            error_rate >= self.failure_rate_threshold or
            slow_rate >= self.slow_call_rate_threshold or
            cost_spike >= self.cost_spike_threshold
        )

        if should_trip:
            self._trip(f"error={error_rate:.1%}, slow={slow_rate:.1%}, cost_spike={cost_spike:.1f}x")

    def _trip(self, reason: str):
        """触发熔断"""
        self.state = CircuitState.OPEN
        self.opened_at = time.time()

        # 自适应冷却时间：连续失败越多，冷却越长
        cooldown_multiplier = min(2 ** (self.consecutive_failures // 3), 10)
        self.current_cooldown = min(
            self.base_cooldown * cooldown_multiplier,
            self.max_cooldown
        )

        print(f"[CIRCUIT BREAKER] TRIPPED: {reason}, cooldown={self.current_cooldown}s")

    def get_metrics(self) -> dict:
        """获取当前熔断器指标"""
        recent = list(self.records)
        return {
            "state": self.state.value,
            "total_calls": len(recent),
            "error_rate": sum(1 for r in recent if not r.success) / len(recent) if recent else 0,
            "avg_latency": sum(r.latency for r in recent) / len(recent) if recent else 0,
            "consecutive_failures": self.consecutive_failures,
        }
```

### 2.3 熔断状态机与降级链路

```
                    ┌──────────────────────────────────────────────┐
                    │              LLM 熔断降级链路                   │
                    └──────────────────────────────────────────────┘

    用户请求 ──→ [熔断器判定]
                    │
          ┌─────────┼─────────────┐
          ↓         ↓             ↓
     ┌────────┐ ┌────────┐  ┌──────────────┐
     │ 通过   │ │ 降级   │  │ 拒绝         │
     └───┬────┘ └───┬────┘  └──────┬───────┘
         │          │              │
         ↓          ↓              ↓
   ┌──────────┐  ┌──────────────────────────┐
   │ 正常模型  │  │  降级决策引擎              │
   │ GPT-4o   │  │                          │
   │ Claude   │  │  Level 1: 快模型替代       │
   └──────────┘  │  Level 2: 语义缓存        │
                 │  Level 3: 本地小模型       │
                 │  Level 4: 规则引擎+模板    │
                 │  Level 5: 静态知识库        │
                 └──────────────────────────┘
```

---

## 三、多级缓存架构

### 3.1 缓存层次设计

LLM应用的缓存与传统Web缓存有本质区别——语义缓存需要理解"相似问题"而非"完全匹配"。

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM 多级缓存架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  L1: 精确匹配缓存 (Redis Hash)                                  │
│  ├── Key: SHA256(prompt + model + params)                      │
│  ├── 命中率: 15-25%                                             │
│  ├── 延迟: < 5ms                                                │
│  └── TTL: 1-24h (取决于内容时效性)                                │
│                                                                 │
│  L2: 语义缓存 (Vector DB + Threshold)                          │
│  ├── 相似度阈值: cosine > 0.95                                  │
│  ├── 命中率: 20-35%                                             │
│  ├── 延迟: 10-50ms                                              │
│  └── 适用: FAQ、知识问答、文档查询                                 │
│                                                                 │
│  L3: 前缀缓存 (Model Provider KV Cache)                        │
│  ├── 命中率: 30-50% (固定System Prompt场景)                     │
│  ├── 延迟: 节省 60-80% TTFT                                     │
│  └── 适用: 长System Prompt、RAG上下文                           │
│                                                                 │
│  L4: 摘要缓存 (Summarized Context Cache)                       │
│  ├── 策略: 对长对话历史做摘要后缓存                               │
│  ├── 命中率: 10-20%                                             │
│  └── 适用: 多轮对话、会话场景                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 语义缓存核心实现

```python
import hashlib
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    model: str
    created_at: float
    hit_count: int
    cost_saved: float

class SemanticCacheManager:
    """
    语义缓存管理器

    核心策略：
    1. 精确匹配优先（毫秒级返回）
    2. 语义匹配兜底（需要embedding计算）
    3. TTL根据内容类型动态调整
    4. 成本感知：贵的调用更值得缓存
    """

    def __init__(self, embedding_model, vector_store, exact_cache, similarity_threshold=0.95):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.exact_cache = exact_cache
        self.similarity_threshold = similarity_threshold

    async def get(
        self,
        query: str,
        model: str,
        params: dict,
    ) -> Optional[str]:
        """多级缓存查询"""
        cache_key = self._make_key(query, model, params)

        # L1: 精确匹配
        exact_result = await self.exact_cache.get(cache_key)
        if exact_result:
            return exact_result

        # L2: 语义匹配
        query_embedding = await self.embedding_model.embed(query)
        similar = await self.vector_store.search(
            query_embedding,
            top_k=1,
            filters={"model": model},
        )

        if similar and similar[0].score >= self.similarity_threshold:
            entry = similar[0]
            # 写回精确缓存，加速下次命中
            await self.exact_cache.set(cache_key, entry.response, ttl=3600)
            return entry.response

        return None

    async def set(
        self,
        query: str,
        response: str,
        model: str,
        params: dict,
        cost: float,
    ):
        """写入多级缓存"""
        cache_key = self._make_key(query, model, params)

        # 写入精确缓存
        ttl = self._compute_ttl(response, cost)
        await self.exact_cache.set(cache_key, response, ttl=ttl)

        # 写入语义缓存
        embedding = await self.embedding_model.embed(query)
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            model=model,
            created_at=time.time(),
            hit_count=0,
            cost_saved=cost,
        )
        await self.vector_store.upsert(entry)

    def _make_key(self, query: str, model: str, params: dict) -> str:
        """生成缓存Key"""
        content = f"{query}|{model}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _compute_ttl(self, response: str, cost: float) -> int:
        """
        动态TTL策略：
        - 高成本响应 → 长缓存（投资回报高）
        - 短响应 → 短缓存（可能是简单查询，变化快）
        - 包含时效信息 → 短缓存
        """
        base_ttl = 3600  # 1小时

        # 成本加成：每¥0.01成本增加10分钟
        cost_boost = int(cost * 600)

        # 内容衰减：短响应可能变化更快
        length_factor = min(len(response) / 500, 2.0)

        return int(base_ttl * length_factor + cost_boost)
```

### 3.3 缓存策略对比

| 缓存层级 | 存储介质 | 命中延迟 | 典型命中率 | 适用场景 | 成本节约 |
|---------|---------|---------|-----------|---------|---------|
| L1-精确 | Redis | < 5ms | 15-25% | 完全相同输入 | 100%调用成本 |
| L2-语义 | Vector DB | 10-50ms | 20-35% | 相似问题 | 100%调用成本 |
| L3-前缀 | Provider KV | 节省TTFT | 30-50% | 固定System Prompt | 60-80%首Token延迟 |
| L4-摘要 | Redis | 5-20ms | 10-20% | 多轮对话续写 | 部分Token成本 |

---

## 四、多模型回退策略

### 4.1 模型降级矩阵

不同场景下的模型降级路径需要精心设计：

```
┌────────────────────────────────────────────────────────────────┐
│                  模型降级决策矩阵                                │
├──────────┬──────────────┬──────────────┬───────────────────────┤
│ 场景      │ 首选模型      │ 降级模型      │ 终极降级               │
├──────────┼──────────────┼──────────────┼───────────────────────┤
│ 复杂推理  │ GPT-4o       │ Claude 3.5   │ GPT-4o-mini + CoT     │
│ 代码生成  │ Claude Opus  │ GPT-4o       │ 本地7B + 规则补全       │
│ 知识问答  │ GPT-4o       │ Gemini 1.5   │ 向量检索 + 模板填充     │
│ 对话聊天  │ Claude Sonnet│ GPT-4o-mini  │ 意图识别 + 模板响应     │
│ 文本摘要  │ GPT-4o-mini  │ 本地14B      │ 抽取式摘要算法          │
│ 分类任务  │ 本地7B       │ 规则引擎      │ 默认类别               │
│ 翻译     │ GPT-4o       │ DeepL API    │ 本地翻译模型            │
└──────────┴──────────────┴──────────────┴───────────────────────┘
```

### 4.2 智能回退路由器实现

```python
from dataclasses import dataclass
from typing import List, Optional
import asyncio

@dataclass
class ModelConfig:
    name: str
    provider: str
    max_latency: float      # 最大可接受延迟(秒)
    quality_score: float    # 质量评分 0-1
    cost_per_1k_tokens: float
    tier: int               # 降级层级 1=最优 5=最低

class LLMFallbackRouter:
    """
    智能模型回退路由器

    核心策略：
    1. 基于熔断器状态自动跳过不可用模型
    2. 基于延迟/成本/质量的多目标选择
    3. 渐进式降级，优先保证可用性
    4. 降级时自动调整Prompt适配小模型
    """

    def __init__(self, model_configs: List[ModelConfig], circuit_breakers: dict):
        # 按tier排序：tier越高越优先使用
        self.models = sorted(model_configs, key=lambda m: -m.tier)
        self.breakers = circuit_breakers

    async def route(
        self,
        request,
        strategy: str = "balanced",
    ) -> tuple[str, str, dict]:
        """
        智能路由：选择最佳可用模型

        Returns:
            (model_name, response, metadata)
        """
        candidates = self._filter_available()

        if not candidates:
            # 所有模型不可用，启用终极降级
            return await self._ultimate_fallback(request)

        # 根据策略选择
        selected = self._select_by_strategy(candidates, strategy)

        for model in selected:
            try:
                response = await self._call_with_timeout(
                    model, request, timeout=model.max_latency
                )
                return model.name, response, {
                    "strategy": strategy,
                    "fallback_level": self.models.index(model),
                }
            except (TimeoutError, Exception) as e:
                # 记录失败，继续尝试下一个模型
                self.breakers[model.name].record_failure()
                continue

        # 所有模型都失败
        return await self._ultimate_fallback(request)

    def _filter_available(self) -> List[ModelConfig]:
        """过滤掉熔断中的模型"""
        return [
            m for m in self.models
            if self.breakers.get(m.name, None) is None
            or self.breakers[m.name].should_allow()
        ]

    def _select_by_strategy(
        self, candidates: List[ModelConfig], strategy: str
    ) -> List[ModelConfig]:
        """根据策略排序候选模型"""
        if strategy == "quality":
            return sorted(candidates, key=lambda m: -m.quality_score)
        elif strategy == "cost":
            return sorted(candidates, key=lambda m: m.cost_per_1k_tokens)
        elif strategy == "latency":
            return sorted(candidates, key=lambda m: m.max_latency)
        else:  # balanced
            return sorted(
                candidates,
                key=lambda m: -(m.quality_score * 0.5 - m.cost_per_1k_tokens * 0.3 - m.max_latency * 0.2)
            )

    async def _ultimate_fallback(self, request) -> tuple[str, str, dict]:
        """终极降级：规则引擎 + 模板"""
        # 根据意图选择预生成的模板响应
        intent = self._detect_intent(request)
        template_response = self._get_template_response(intent)

        return "template_engine", template_response, {
            "strategy": "ultimate_fallback",
            "intent": intent,
        }
```

### 4.3 Prompt自适应降级

大模型降级时，Prompt也需要相应调整——小模型需要更明确的指令：

```python
ADAPTIVE_PROMPTS = {
    "high_tier": """你是一个专业的AI助手。请根据以下信息回答问题：
{context}
问题：{question}""",

    "mid_tier": """请严格根据提供的参考资料回答问题。如果资料中没有相关信息，请明确说"我不确定"。

参考资料：
{context}

问题：{question}

请直接回答，不要编造信息。""",

    "low_tier": """任务：基于参考资料回答问题。

规则：
1. 只使用参考资料中的信息
2. 如果资料不足，回答"根据现有资料无法确定"
3. 回答控制在100字以内

资料：{context}
问题：{question}
答案：""",
}
```

---

## 五、请求限流与优先级调度

### 5.1 基于Token的限流

LLM应用的限流必须考虑Token消耗，而不仅仅是请求数量：

```python
import asyncio
from collections import defaultdict
import time

class TokenAwareRateLimiter:
    """
    Token感知限流器

    传统限流：限制 QPS（每秒请求数）
    LLM限流：限制 TPM（每分钟Token数）+ QPS + 并发

    三层限制：
    1. 全局限流：总Token消耗不超过预算
    2. 用户限流：单用户Token消耗上限
    3. 模型限流：不超出Provider配额
    """

    def __init__(
        self,
        global_tpm_limit: int = 1_000_000,     # 全局每分钟Token上限
        user_tpm_limit: int = 50_000,           # 单用户每分钟Token上限
        max_concurrent: int = 50,               # 最大并发数
        qps_limit: int = 100,                   # QPS上限
    ):
        self.global_tpm_limit = global_tpm_limit
        self.user_tpm_limit = user_tpm_limit
        self.max_concurrent = max_concurrent
        self.qps_limit = qps_limit

        self.global_tpm_usage = 0
        self.user_tpm_usage = defaultdict(int)
        self.concurrent_count = 0
        self.qps_count = 0
        self.qps_window_start = time.time()

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.priority_queue = asyncio.PriorityQueue()

    async def acquire(self, user_id: str, estimated_tokens: int) -> bool:
        """获取请求许可"""
        # 检查QPS
        if self.qps_count >= self.qps_limit:
            return False

        # 检查全局TPM
        if self.global_tpm_usage + estimated_tokens > self.global_tpm_limit:
            return False

        # 检查用户TPM
        if self.user_tpm_usage[user_id] + estimated_tokens > self.user_tpm_limit:
            return False

        # 检查并发
        await self.semaphore.acquire()
        self.concurrent_count += 1
        self.qps_count += 1
        self.global_tpm_usage += estimated_tokens
        self.user_tpm_usage[user_id] += estimated_tokens

        return True

    async def release(self, user_id: str, actual_tokens: int):
        """释放资源并修正Token计数"""
        self.semaphore.release()
        self.concurrent_count -= 1

        # 修正Token计数（实际消耗可能与预估不同）
        # 这里简化处理，实际应使用差值修正
```

### 5.2 优先级调度策略

```
┌──────────────────────────────────────────────────────────┐
│                LLM请求优先级调度                           │
├──────┬───────────────┬─────────────┬─────────────────────┤
│ 优先级│ 场景           │ 超时设置     │ 降级策略             │
├──────┼───────────────┼─────────────┼─────────────────────┤
│ P0   │ 核心业务链路   │ 60s         │ 多级降级             │
│ P1   │ 重要用户交互   │ 30s         │ 模型降级 + 缓存      │
│ P2   │ 辅助功能       │ 15s         │ 快速降级 + 缓存      │
│ P3   │ 批处理/异步    │ 无超时       │ 排队 + 延后执行      │
│ P4   │ 实验/测试      │ 10s         │ 直接拒绝             │
└──────┴───────────────┴─────────────┴─────────────────────┘
```

---

## 六、端到端SLA保障流程

### 6.1 完整请求生命周期

```
┌─────────────────────────────────────────────────────────────────────┐
│                   LLM请求完整SLA保障流程                              │
└─────────────────────────────────────────────────────────────────────┘

用户请求
   │
   ▼
┌──────────────┐
│ 1. 预检查     │ ← 参数校验 + 安全过滤 + 预算检查
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. 限流判定   │ ← Token感知限流 + 优先级标记
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. 缓存查询   │ ← L1精确 → L2语义
└──────┬───────┘
       │ 缓存未命中
       ▼
┌──────────────┐
│ 4. 路由选择   │ ← 熔断器状态 + 多目标优化 → 选择模型
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 5. 执行调用   │ ← 超时控制 + 流式响应 + Token计数
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
 成功     失败
   │       │
   ▼       ▼
┌──────┐ ┌──────────────┐
│ 6.后  │ │ 6.降级执行    │ ← 下一模型/缓存/模板
│ 处理  │ └──────┬───────┘
└──┬───┘        │
   │            ▼
   │      ┌──────────────┐
   │      │ 6b. 降级后处理│
   │      └──────┬───────┘
   │             │
   ▼             ▼
┌──────────────────────┐
│ 7. 写入缓存 + 指标上报 │
└──────────────────────┘
   │
   ▼
响应返回用户
```

### 6.2 整合代码示例

```python
class LLMServiceWithSLA:
    """带完整SLA保障的LLM服务"""

    def __init__(self):
        self.rate_limiter = TokenAwareRateLimiter()
        self.cache_manager = SemanticCacheManager(...)
        self.fallback_router = LLMFallbackRouter(...)
        self.circuit_breakers = {...}
        self.metrics = LLMMetricsCollector()

    async def handle_request(self, request, user_id: str, priority: int = 1):
        """完整SLA保障的请求处理"""
        start_time = time.time()

        try:
            # 1. 预检查
            self._validate_request(request)

            # 2. 限流判定
            estimated_tokens = self._estimate_tokens(request)
            if not await self.rate_limiter.acquire(user_id, estimated_tokens):
                return self._rate_limit_response()

            # 3. 缓存查询
            cached = await self.cache_manager.get(
                request.prompt, request.model, request.params
            )
            if cached:
                self.metrics.record_cache_hit()
                return cached

            # 4-6. 路由 + 调用 + 降级
            model, response, metadata = await self.fallback_router.route(
                request, strategy="balanced"
            )

            # 7. 写入缓存
            cost = self._calculate_cost(model, response)
            await self.cache_manager.set(
                request.prompt, response, model, request.params, cost
            )

            # 上报指标
            latency = time.time() - start_time
            self.metrics.record_success(latency, model, cost)

            return response

        except RateLimitError:
            return self._rate_limit_response()
        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_failure(latency, str(e))
            return self._error_response(e)
        finally:
            await self.rate_limiter.release(user_id, estimated_tokens)
```

---

## 七、生产环境实战数据

### 7.1 优化前后的SLA对比

经过以上策略的系统化实施，我们观察到以下改善：

| 指标 | 优化前 | 优化后 | 改善幅度 |
|------|-------|-------|---------|
| P50延迟 | 8.2s | 2.1s | ↓ 74% |
| P99延迟 | 45s | 8.5s | ↓ 81% |
| 超时率 | 8.3% | 0.6% | ↓ 93% |
| 成功率 | 91.2% | 99.3% | ↑ 8.9% |
| 缓存命中率 | 0% | 32% | ↑ 32% |
| 单次平均成本 | ¥0.08 | ¥0.035 | ↓ 56% |
| 月度总成本 | ¥80,000 | ¥35,000 | ↓ 56% |
| 用户满意度 | 3.2/5 | 4.5/5 | ↑ 41% |

### 7.2 关键经验总结

| 经验 | 说明 |
|------|------|
| **先监控，再优化** | 没有完善的监控体系，任何优化都是盲人摸象 |
| **熔断要多维度** | 仅靠错误率触发熔断会遗漏延迟恶化和成本飙升 |
| **缓存是最大的ROI** | 语义缓存的投入产出比远超其他优化手段 |
| **降级要分级** | 从模型降级到规则降级，每一级都有明确的触发条件 |
| **Prompt要适配** | 小模型降级时必须简化Prompt，否则效果更差 |
| **成本也是SLA** | 没有成本控制的SLA是不可持续的 |

---

## 八、总结

LLM应用的SLA保障是一个系统工程，核心要点：

1. **指标体系**：超越传统可用性指标，纳入TTFT、TPOT、幻觉率、成本等LLM特有指标
2. **熔断器**：多维度判定（延迟+错误率+成本），自适应冷却时间
3. **缓存**：多级缓存架构（精确→语义→前缀→摘要），语义缓存是关键
4. **回退**：多模型降级矩阵 + Prompt自适应 + 终极规则引擎
5. **限流**：Token感知限流 + 优先级调度，而非简单的QPS限制

最终目标是构建一个**成本可控、体验可预期、故障可降级**的LLM应用系统。在AI应用从Demo走向Production的过程中，SLA保障体系是区分"玩具"和"产品"的关键分水岭。

> 💡 **实践建议**：不要试图一次性实施所有策略。建议按以下优先级推进：监控 → 缓存 → 熔断 → 回退 → 限流。每一步都能带来可量化的改善。
