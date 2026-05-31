---
title: "AI应用限流、降级与弹性伸缩工程实践：构建高可用LLM服务的核心策略"
description: "从限流熔断到弹性伸缩，系统性解决LLM应用在高并发、资源受限场景下的可用性与稳定性问题"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: "infra"
tags: ["限流", "降级", "弹性伸缩", "LLM", "高可用", "SRE", "AI工程化"]
draft: false
---

## 引言

LLM应用的可用性挑战与传统Web服务截然不同：GPU资源昂贵且有限、推理延迟是传统API的10-100倍、token消耗带来不可预测的成本波动。一个简单的"用户量翻倍"就可能导致服务雪崩。

根据我们的生产经验，LLM应用面临的可用性挑战可以归纳为：

```
┌─────────────────────────────────────────────────────────────────┐
│              LLM应用可用性核心挑战                                │
├──────────────────┬──────────────────────────────────────────────┤
│  资源受限         │  GPU稀缺、显存固定、无法像CPU服务那样水平扩展   │
│  延迟不可控       │  长文本推理延迟波动大，P99可达P50的5-10倍       │
│  成本不可预测     │  Token消耗随对话轮次指数增长                   │
│  外部依赖         │  依赖LLM API提供商，SLA不在自己手中            │
│  流量不均匀       │  对话场景流量突发性强，传统限流策略失效          │
│  级联故障         │  上游超时导致下游重试风暴，放大故障影响          │
└──────────────────┴──────────────────────────────────────────────┘
```

本文将系统性地介绍LLM应用的限流、降级与弹性伸缩工程实践，这些是从Demo走向生产级LLM服务的必经之路。

## 限流策略：从粗粒度到精细化

### 传统限流的局限

传统Web服务的限流通常基于QPS或并发连接数，但LLM应用需要考虑更多维度：

```
┌─────────────────────────────────────────────────────────────────┐
│           LLM应用限流维度对比                                     │
├─────────────────┬───────────────────────────────────────────────┤
│  传统限流维度    │  LLM增强限流维度                               │
├─────────────────┼───────────────────────────────────────────────┤
│  QPS            │  QPS + Token消耗速率 + GPU占用率               │
│  并发连接       │  并发推理请求 + 活跃会话数                      │
│  带宽           │  输入Token数 + 输出Token数                     │
│  请求大小       │  上下文长度 × 并发数 (实际GPU内存占用)          │
│  用户级限流     │  用户级 + 应用级 + 租户级 三层限流               │
└─────────────────┴───────────────────────────────────────────────┘
```

### 多维度限流架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    多维度限流架构                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    入口层限流                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ 全局QPS  │  │ 用户级   │  │ IP级      │              │  │
│  │  │ 限制器   │  │ 配额管理 │  │ 反爬策略  │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    资源层限流                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ Token    │  │ GPU显存  │  │ 上下文    │              │  │
│  │  │ 预算控制 │  │ 限额管理 │  │ 长度限制  │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    成本层限流                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ 月度预算 │  │ 实时成本 │  │ 异常消费 │              │  │
│  │  │ 硬限制   │  │ 监控告警 │  │ 自动熔断 │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Token感知限流实现

```python
import time
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class TokenBucket:
    """Token感知的限流桶"""
    capacity: int              # 桶容量 (最大Token数)
    refill_rate: int           # 每秒补充的Token数
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    
    def try_acquire(self, estimated_tokens: int) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= estimated_tokens:
            self.tokens -= estimated_tokens
            return True
        return False

class LLMLimiter:
    """LLM应用多维度限流器"""
    
    def __init__(self):
        # 用户级Token配额: 每用户每分钟20000 Token
        self.user_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(capacity=20000, refill_rate=333)
        )
        # 应用级Token配额: 每应用每分钟500000 Token
        self.app_bucket = TokenBucket(capacity=500000, refill_rate=8333)
        # 全局并发限制
        self.max_concurrent = 100
        self.current_concurrent = 0
    
    async def check_rate_limit(
        self, user_id: str, app_id: str, estimated_tokens: int
    ) -> dict:
        """多维度限流检查"""
        result = {"allowed": True, "reason": None, "retry_after": None}
        
        # 1. 全局并发检查
        if self.current_concurrent >= self.max_concurrent:
            result["allowed"] = False
            result["reason"] = "global_concurrent_exceeded"
            result["retry_after"] = 1.0
            return result
        
        # 2. 用户级Token配额
        if not self.user_buckets[user_id].try_acquire(estimated_tokens):
            result["allowed"] = False
            result["reason"] = "user_token_quota_exceeded"
            result["retry_after"] = 5.0
            return result
        
        # 3. 应用级Token配额
        if not self.app_bucket.try_acquire(estimated_tokens):
            result["allowed"] = False
            result["reason"] = "app_token_quota_exceeded"
            result["retry_after"] = 10.0
            # 回滚用户配额
            self.user_buckets[user_id].tokens += estimated_tokens
            return result
        
        return result
```

### 自适应限流算法

传统固定阈值限流无法应对LLM应用的流量波动。自适应限流根据实时系统状态动态调整阈值：

```
┌─────────────────────────────────────────────────────────────────┐
│                 自适应限流决策逻辑                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入信号:                                                        │
│  ├─ GPU利用率 (目标: 70-85%)                                     │
│  ├─ 推理延迟 (P99 vs 基线)                                       │
│  ├─ 队列深度 (待处理请求数)                                      │
│  ├─ 错误率 (5xx比例)                                             │
│  └─ Token消耗速率                                                 │
│                                                                  │
│  决策逻辑:                                                        │
│  if GPU利用率 > 90% OR P99延迟 > 2倍基线:                        │
│      降低限流阈值 30% (更严格)                                    │
│  elif GPU利用率 < 50% AND 队列为空 AND 错误率 < 0.1%:             │
│      提升限流阈值 10% (更宽松)                                    │
│  elif 错误率 > 5%:                                               │
│      立即降低限流阈值 50% (紧急收缩)                              │
│                                                                  │
│  调整频率: 每10秒评估一次                                         │
│  平滑策略: 指数移动平均，避免阈值剧烈波动                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 降级策略：从优雅退化到服务保全

### LLM应用降级分级

LLM应用的降级需要比传统服务更加精细，因为完全不可用和部分可用之间的体验差距巨大：

```
┌─────────────────────────────────────────────────────────────────┐
│                    降级分级策略                                   │
├──────┬──────────────────────────────────────────────────────────┤
│ 级别 │  策略描述                                                  │
├──────┼──────────────────────────────────────────────────────────┤
│  L0  │  正常服务: 所有功能可用，完整LLM推理                       │
├──────┼──────────────────────────────────────────────────────────┤
│  L1  │  轻度降级: 降低模型精度(大→小模型)，缩短最大token数        │
├──────┼──────────────────────────────────────────────────────────┤
│  L2  │  中度降级: 关闭非核心功能(如流式输出)，限制上下文长度       │
├──────┼──────────────────────────────────────────────────────────┤
│  L3  │  重度降级: 使用缓存/模板响应替代LLM推理                    │
├──────┼──────────────────────────────────────────────────────────┤
│  L4  │  紧急降级: 返回静态内容+告知用户稍后重试                    │
└──────┴──────────────────────────────────────────────────────────┘
```

### 模型降级链

```
┌─────────────────────────────────────────────────────────────────┐
│                  模型降级链架构                                   │
│                                                                  │
│  ┌─────────────┐    超时/失败    ┌─────────────┐               │
│  │  GPT-4o     │ ─────────────→ │  GPT-4o-mini │               │
│  │  (主模型)    │                │  (降级模型)   │               │
│  └─────────────┘                └──────┬──────┘               │
│                                        │ 超时/失败              │
│                                        ▼                        │
│                               ┌─────────────┐                  │
│                               │  本地7B模型  │                  │
│                               │  (兜底模型)  │                  │
│                               └──────┬──────┘                  │
│                                      │ 失败                     │
│                                      ▼                          │
│                             ┌─────────────┐                    │
│                             │  模板+缓存   │                    │
│                             │  (最后防线)  │                    │
│                             └─────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

实现示例：

```python
from enum import IntEnum
import asyncio

class DegradationLevel(IntEnum):
    NORMAL = 0
    LIGHT = 1      # 小模型
    MODERATE = 2   # 缩短上下文
    HEAVY = 3      # 模板响应
    CRITICAL = 4   # 静态兜底

class LLMFallbackChain:
    """LLM降级链实现"""
    
    def __init__(self):
        self.models = [
            {"name": "gpt-4o", "timeout": 30, "max_tokens": 4096},
            {"name": "gpt-4o-mini", "timeout": 15, "max_tokens": 2048},
            {"name": "local-qwen-7b", "timeout": 10, "max_tokens": 1024},
        ]
        self.current_level = DegradationLevel.NORMAL
    
    async def invoke(self, messages: list[dict], **kwargs) -> dict:
        """带降级的模型调用"""
        
        # 根据当前降级级别调整参数
        if self.current_level >= DegradationLevel.MODERATE:
            messages = self._truncate_context(messages, max_turns=3)
        
        # 尝试模型链
        for model in self.models:
            try:
                result = await asyncio.wait_for(
                    self._call_model(model["name"], messages, **kwargs),
                    timeout=model["timeout"]
                )
                return result
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue
        
        # 所有模型失败，返回模板响应
        return self._template_response(messages)
    
    def _truncate_context(self, messages: list[dict], max_turns: int) -> list[dict]:
        """截断上下文，只保留system + 最近N轮对话"""
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]
        return system_msgs + other_msgs[-(max_turns * 2):]
    
    def _template_response(self, messages: list[dict]) -> dict:
        """模板兜底响应"""
        return {
            "content": "抱歉，当前服务繁忙。您的问题已记录，我们将尽快回复。",
            "model": "template-fallback",
            "usage": {"total_tokens": 0}
        }
```

### 语义缓存降级

```
┌─────────────────────────────────────────────────────────────────┐
│                 语义缓存降级策略                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  适用场景: FAQ类问题、常见查询、高频重复问题                       │
│                                                                  │
│  实现原理:                                                        │
│  1. 用户查询 → Embedding → 向量检索                               │
│  2. 相似度 > 阈值(0.92) → 返回缓存结果                           │
│  3. 相似度 < 阈值 → 正常LLM推理 → 结果存入缓存                   │
│                                                                  │
│  降级场景:                                                        │
│  ├─ L2降级: 阈值从0.95降到0.85，更多查询命中缓存                  │
│  ├─ L3降级: 阈值降到0.75，几乎全部使用缓存                       │
│  └─ L4降级: 直接返回预设热门问答的缓存结果                        │
│                                                                  │
│  注意事项:                                                        │
│  ├─ 缓存结果需要标记来源，避免误导用户                             │
│  ├─ 实时性要求高的场景不适合缓存                                   │
│  └─ 定期清理过期缓存，避免陈旧信息                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 弹性伸缩：GPU资源的动态管理

### LLM应用伸缩的独特挑战

```
┌─────────────────────────────────────────────────────────────────┐
│            LLM应用 vs 传统Web应用伸缩对比                         │
├─────────────────┬──────────────────┬────────────────────────────┤
│  维度            │  传统Web应用     │  LLM应用                   │
├─────────────────┼──────────────────┼────────────────────────────┤
│  扩展单位        │  容器/Pod        │  GPU实例 (无法拆分)         │
│  扩展速度        │  秒级            │  分钟级 (模型加载)          │
│  资源粒度        │  CPU/Memory可灵活配│ GPU显存固定               │
│  状态管理        │  无状态为主       │  KV Cache需要持久化        │
│  成本模型        │  线性增长        │  阶梯式 (GPU单价高)        │
│  冷启动          │  毫秒级          │  10-60秒 (模型加载)        │
│  预热策略        │  简单有效        │  需要预填充warmup请求       │
└─────────────────┴──────────────────┴────────────────────────────┘
```

### 弹性伸缩架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│              LLM应用弹性伸缩架构                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    监控层                                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ GPU利用率 │  │ 请求队列 │  │ Token消耗 │              │  │
│  │  │ 监控     │  │ 深度监控  │  │ 速率监控  │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    决策层                                  │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ 伸缩策略引擎                                      │   │  │
│  │  ├─ 基于队列深度的水平扩展                            │   │  │
│  │  ├─ 基于时间的预测性扩展 (cron)                       │   │  │
│  │  ├─ 基于GPU利用率的垂直调整                          │   │  │
│  │  └─ 预热池管理 (Pre-warmed instances)                │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    执行层                                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ 实例创建 │  │ 模型加载 │  │ 流量切换  │              │  │
│  │  │ & 预热   │  │ & 就绪   │  │ & 旧实例  │              │  │
│  │  │          │  │          │  │ 优雅下线  │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 预热池策略

LLM实例的冷启动代价极高（模型加载需要10-60秒），预热池是解决这一问题的关键：

```python
import asyncio
from dataclasses import dataclass

@dataclass
class WarmInstance:
    instance_id: str
    model_name: str
    status: str  # "warming", "ready", "busy"
    gpu_memory_used: float
    ready_at: float | None = None

class WarmPoolManager:
    """LLM预热池管理器"""
    
    def __init__(self, target_warm: int = 3):
        self.target_warm = target_warm  # 目标预热实例数
        self.pool: list[WarmInstance] = []
        self.lock = asyncio.Lock()
    
    async def get_instance(self, model_name: str) -> WarmInstance | None:
        """获取一个可用的预热实例"""
        async with self.lock:
            for inst in self.pool:
                if inst.model_name == model_name and inst.status == "ready":
                    inst.status = "busy"
                    return inst
            return None
    
    async def replenish_pool(self):
        """补充预热池到目标数量"""
        async with self.lock:
            ready_count = sum(
                1 for i in self.pool 
                if i.status == "ready" and i.model_name == "qwen-7b"
            )
            
            needed = self.target_warm - ready_count
            for _ in range(needed):
                instance = await self._create_warm_instance("qwen-7b")
                self.pool.append(instance)
    
    async def _create_warm_instance(self, model_name: str) -> WarmInstance:
        """创建并预热一个新实例"""
        instance = WarmInstance(
            instance_id=f"gpu-{model_name}-{id(self)}",
            model_name=model_name,
            status="warming",
            gpu_memory_used=0
        )
        
        # 模拟模型加载过程
        await asyncio.sleep(15)  # 实际中是模型加载时间
        
        # Warmup: 发送几个虚拟请求填充KV Cache
        await self._warmup_requests(instance)
        
        instance.status = "ready"
        instance.ready_at = asyncio.get_event_loop().time()
        return instance
```

### 多级扩缩容策略

```
┌─────────────────────────────────────────────────────────────────┐
│              多级扩缩容触发条件                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  扩容触发:                                                        │
│  ├─ L1 (快速扩容): 队列深度 > 10 持续30秒                        │
│  │  └─ 动作: 从预热池取实例，秒级生效                             │
│  ├─ L2 (常规扩容): GPU利用率 > 80% 持续5分钟                     │
│  │  └─ 动作: 创建新实例，2-5分钟生效                             │
│  └─ L3 (预测性扩容): 基于历史流量模式提前扩容                     │
│     └─ 动作: 定时任务，提前15分钟预热                             │
│                                                                  │
│  缩容触发:                                                        │
│  ├─ L1 (快速缩容): 队列为空 且 GPU利用率 < 30% 持续10分钟       │
│  │  └─ 动作: 优雅下线多余实例                                    │
│  ├─ L2 (常规缩容): GPU利用率 < 40% 持续30分钟                   │
│  │  └─ 动作: 逐步回收实例                                        │
│  └─ L3 (定时缩容): 夜间/低峰期自动缩容                           │
│     └─ 动作: 保留最小实例数                                      │
│                                                                  │
│  保护机制:                                                        │
│  ├─ 冷却时间: 扩缩容操作间隔至少5分钟                            │
│  ├─ 最小实例数: 至少保留1个实例                                   │
│  ├─ 最大实例数: 根据预算设置上限                                  │
│  └─ 状态检查: 确保缩容前请求已全部处理完成                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 成本控制工程

### Token成本实时监控

```
┌─────────────────────────────────────────────────────────────────┐
│              Token成本监控面板                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  实时指标:                                                        │
│  ├─ 今日Token消耗: 12,345,678 tokens                            │
│  ├─ 今日预估成本: $37.04                                         │
│  ├─ 本月累计成本: $892.15 / 预算 $2,000 (44.6%)                 │
│  └─ 成本趋势: ▲ 12% vs 昨日同期                                  │
│                                                                  │
│  按维度拆分:                                                      │
│  ├─ 按模型: GPT-4o (62%) | GPT-4o-mini (28%) | 本地 (10%)      │
│  ├─ 按应用: 客服助手 (45%) | 代码助手 (35%) | 文档助手 (20%)    │
│  └─ 按用户: 高消耗Top10用户占总消耗38%                           │
│                                                                  │
│  告警规则:                                                        │
│  ├─ ⚠️  单日成本超过日预算的80%                                   │
│  ├─ 🚨 单用户1小时Token消耗超过阈值                               │
│  ├─ ⚠️  月度成本超过预算的70%                                    │
│  └─ 🚨 出现异常消费模式 (突增300%+)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 成本优化策略矩阵

```
┌─────────────────────────────────────────────────────────────────┐
│              成本优化策略矩阵                                     │
├──────────────┬──────────────┬───────────────────────────────────┤
│  策略          │  节省比例     │  实施难度                         │
├──────────────┼──────────────┼───────────────────────────────────┤
│  模型路由     │  30-60%      │  低 (按场景选模型)                │
│  Prompt压缩   │  15-40%      │  中 (需要prompt工程)              │
│  语义缓存     │  20-50%      │  中 (需要缓存基础设施)            │
│  批量推理     │  10-25%      │  低 (调整API调用方式)             │
│  本地部署     │  40-70%      │  高 (需要GPU运维能力)             │
│  Token预算    │  10-30%      │  低 (限制最大token数)             │
│  异步批处理   │  15-35%      │  中 (需要异步架构)                │
└──────────────┴──────────────┴───────────────────────────────────┘
```

### 模型路由策略

```python
class ModelRouter:
    """基于任务复杂度的智能模型路由"""
    
    def __init__(self):
        self.routing_rules = [
            {
                "condition": lambda q: len(q) < 50 and q.endswith("?"),
                "model": "gpt-4o-mini",
                "reason": "简单问答"
            },
            {
                "condition": lambda q: "代码" in q or "function" in q.lower(),
                "model": "gpt-4o",
                "reason": "代码任务需要强推理"
            },
            {
                "condition": lambda q: len(q) > 500,
                "model": "gpt-4o",
                "reason": "长文本需要强理解"
            },
        ]
        self.default_model = "gpt-4o-mini"
    
    def route(self, query: str) -> dict:
        """路由决策"""
        for rule in self.routing_rules:
            if rule["condition"](query):
                return {
                    "model": rule["model"],
                    "reason": rule["reason"],
                    "estimated_cost": self._estimate_cost(query, rule["model"])
                }
        
        return {
            "model": self.default_model,
            "reason": "默认路由",
            "estimated_cost": self._estimate_cost(query, self.default_model)
        }
```

## 全链路可观测性

### 监控指标体系

```
┌─────────────────────────────────────────────────────────────────┐
│              LLM应用全链路监控指标                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📊 业务指标                                                      │
│  ├─ 请求成功率 (目标: >99.5%)                                    │
│  ├─ 首Token延迟 TTFD (目标: <2s)                                │
│  ├─ 端到端延迟 (目标: <10s)                                      │
│  ├─ 用户满意度评分                                                │
│  └─ 功能使用分布                                                  │
│                                                                  │
│  🔧 基础设施指标                                                   │
│  ├─ GPU利用率 (目标: 70-85%)                                     │
│  ├─ GPU显存使用率                                                 │
│  ├─ 请求队列深度                                                  │
│  ├─ 活跃连接数                                                    │
│  └─ 模型加载时间                                                  │
│                                                                  │
│  💰 成本指标                                                      │
│  ├─ 每请求平均Token数                                             │
│  ├─ 每请求平均成本                                                │
│  ├─ 缓存命中率                                                    │
│  ├─ 模型路由分布                                                  │
│  └─ 月度/日度成本趋势                                             │
│                                                                  │
│  🛡️ 安全指标                                                      │
│  ├─ 限流触发次数                                                  │
│  ├─ 降级触发次数                                                  │
│  ├─ 异常消费告警                                                  │
│  └─ 内容安全拦截率                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 告警与响应自动化

```
┌─────────────────────────────────────────────────────────────────┐
│              告警响应自动化流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  一级告警 (自动处理):                                              │
│  ├─ GPU利用率>90% → 自动扩容                                     │
│  ├─ 队列深度>50 → 自动扩容 + 触发降级                            │
│  └─ 错误率>5% → 自动切换备用模型                                 │
│                                                                  │
│  二级告警 (通知+自动处理):                                         │
│  ├─ P99延迟>20s → 通知SRE + 自动降级到L2                        │
│  ├─ 日成本超预算80% → 通知财务 + 自动限流收紧                    │
│  └─ 模型API异常 → 通知SRE + 自动切换本地模型                     │
│                                                                  │
│  三级告警 (需人工介入):                                            │
│  ├─ 月成本超预算 → 需人工审批追加预算                              │
│  ├─ 多个模型同时故障 → 需人工决策降级策略                         │
│  └─ 数据安全事件 → 立即人工响应                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 实战案例：从0到1构建高可用LLM网关

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                高可用LLM网关完整架构                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    客户端                                  │  │
│  │  Web App │ Mobile │ API │ 企业集成                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Gateway                            │  │
│  │  认证 │ 限流 │ 路由 │ 负载均衡 │ 监控                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Orchestration Layer                     │  │
│  │                                                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │  │
│  │  │ Token      │  │ 模型路由   │  │ 降级决策   │         │  │
│  │  │ 计数器     │  │ 引擎       │  │ 引擎       │         │  │
│  │  └────────────┘  └────────────┘  └────────────┘         │  │
│  │                                                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │  │
│  │  │ 语义缓存   │  │ 重试管理   │  │ 流量调度   │         │  │
│  │  │ (Redis)    │  │            │  │            │         │  │
│  │  └────────────┘  └────────────┘  └────────────┘         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│              ┌───────────┼───────────┐                          │
│              ▼           ▼           ▼                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │  GPT-4o     │ │  GPT-4o-mini│ │  本地 Qwen  │              │
│  │  Pool       │ │  Pool       │ │  7B Pool    │              │
│  │  (3实例)    │ │  (5实例)    │ │  (2实例)    │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Observability                           │  │
│  │  Metrics (Prometheus) │ Logs (Loki) │ Traces (Jaeger)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键配置参数

```yaml
# LLM网关配置示例
gateway:
  rate_limiting:
    global:
      qps: 1000
      concurrent: 200
    per_user:
      qps: 50
      daily_tokens: 500000
      monthly_budget: 100.0  # 美元
    per_app:
      qps: 500
      daily_tokens: 5000000

  fallback:
    enabled: true
    timeout_seconds: 30
    chain:
      - model: "gpt-4o"
        timeout: 30
        max_retries: 1
      - model: "gpt-4o-mini"
        timeout: 15
        max_retries: 1
      - model: "local-qwen-7b"
        timeout: 10
        max_retries: 0
      - type: "template"
        max_retries: 0

  degradation:
    auto_enabled: true
    levels:
      light:
        gpu_threshold: 80
        actions: ["reduce_max_tokens", "disable_stream"]
      moderate:
        gpu_threshold: 90
        actions: ["switch_small_model", "truncate_context"]
      heavy:
        error_rate_threshold: 0.1
        actions: ["enable_cache_only", "template_response"]

  scaling:
    warm_pool:
      target_count: 3
      model: "qwen-7b"
      max_wait_seconds: 60
    auto_scaling:
      scale_up:
        - metric: "queue_depth"
          threshold: 10
          duration: "30s"
          action: "add_instance"
      scale_down:
        - metric: "gpu_utilization"
          threshold: 30
          duration: "10m"
          action: "remove_instance"
    min_instances: 1
    max_instances: 20
    cooldown_seconds: 300

  cost_control:
    daily_budget: 200.0
    monthly_budget: 5000.0
    alert_thresholds: [0.5, 0.7, 0.85, 0.95]
    hard_limit: true
    routing:
      simple_query: "gpt-4o-mini"
      complex_query: "gpt-4o"
      code_task: "gpt-4o"
      default: "gpt-4o-mini"
```

## 总结

构建高可用的LLM应用需要超越传统Web服务的思维定式。核心要点：

1. **限流要多维度**: 不只是QPS，还要考虑Token消耗、GPU占用、用户配额
2. **降级要分层次**: 从模型切换到模板兜底，每一层都有明确的触发条件和恢复策略
3. **伸缩要预热**: GPU实例的冷启动代价极高，预热池是必备组件
4. **成本要可控**: 智型模型路由 + 语义缓存 + Token预算，三管齐下
5. **可观测要全链路**: 业务指标、基础设施指标、成本指标缺一不可

LLM应用的高可用不是一蹴而就的工程，而是一个持续迭代优化的过程。从最核心的限流和降级开始，逐步完善弹性伸缩和成本控制，最终建立起完整的LLM运维体系。
