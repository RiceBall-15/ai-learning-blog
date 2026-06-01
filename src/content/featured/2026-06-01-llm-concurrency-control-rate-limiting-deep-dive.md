---
title: "LLM应用中的并发控制与限流策略：构建高可用AI服务的核心机制"
description: "从令牌桶到自适应限流，深入解析LLM应用中的并发控制、限流策略与弹性伸缩的生产级实践"
date: 2026-06-01
author: "RiceBall"
category: "featured"
subCategory: deep-dive
tags: ["并发控制", "限流", "弹性伸缩", "LLM应用", "高可用", "流控策略"]
draft: false
---

# LLM应用中的并发控制与限流策略：构建高可用AI服务的核心机制

## 引言：LLM应用的并发困境

构建LLM应用时，工程师们很快会发现一个与传统Web应用截然不同的挑战：**LLM推理是计算密集型的长时操作**。

传统Web请求的处理时间通常在毫秒级，而LLM推理请求可能需要数秒甚至数十秒。这意味着：

- 一个占用GPU的LLM请求，在相同时间内只能处理传统请求的1/100甚至更少
- 突发流量会迅速耗尽GPU资源，导致级联故障
- 不同模型、不同Prompt长度的推理成本差异巨大，简单的QPS限流无法准确控制资源消耗

本文将深入解析LLM应用中的并发控制与限流策略，从基础的令牌桶算法到生产级的自适应限流方案，帮助你构建真正高可用的AI服务。

---

## 一、LLM应用的流量特征分析

### 1.1 与传统Web应用的对比

| 维度 | 传统Web应用 | LLM应用 |
|------|------------|---------|
| **请求延迟** | 10-100ms | 500ms-60s |
| **资源占用** | CPU/内存 | GPU显存/算力 |
| **成本模型** | 按请求计费 | 按Token计费 |
| **并发瓶颈** | 连接数/CPU | GPU显存/显存带宽 |
| **流量波动影响** | 中等 | 极高（GPU资源有限） |
| **降级难度** | 低 | 高（模型推理难以快速降级） |

### 1.2 LLM应用的典型流量模式

```
流量模式示例（客服Agent系统）：

正常时段 (9:00-18:00):
  ├── 平均QPS: 50
  ├── 平均Token数/请求: 2000
  └── GPU利用率: 60%

高峰时段 (14:00-16:00):
  ├── 平均QPS: 200 (4倍增长)
  ├── 平均Token数/请求: 3000 (长对话增多)
  └── GPU利用率: 95% (接近饱和)

异常流量 (系统故障后):
  ├── 瞬时QPS: 2000 (重试风暴)
  ├── 平均Token数/请求: 1500 (短请求增多)
  └── GPU利用率: 100% → OOM
```

### 1.3 为什么传统限流方案不适用

```python
# ❌ 传统令牌桶：无法区分请求成本
class TraditionalTokenBucket:
    def __init__(self, rate=100):  # 100 QPS
        self.rate = rate
        self.tokens = rate
    
    def allow(self):
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False

# 问题：
# 1. 一个20000 token的长请求和一个200 token的短请求消耗相同配额
# 2. 无法感知GPU实际负载
# 3. 无法区分不同模型的资源消耗差异
```

---

## 二、基础限流算法在LLM场景的适配

### 2.1 令牌桶算法（Token Bucket）

令牌桶是最经典的限流算法，适合控制平均速率：

```python
import time
import threading

class LLMBucketRateLimiter:
    """LLM场景的令牌桶限流器"""
    
    def __init__(self, rate: float, capacity: int, cost_calculator):
        """
        Args:
            rate: 每秒生成的令牌数
            capacity: 桶容量
            cost_calculator: 计算请求成本的函数
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.cost_calculator = cost_calculator
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def allow(self, estimated_tokens: int = 2000) -> bool:
        """判断请求是否允许通过"""
        with self.lock:
            self._refill()
            
            # 根据预估Token数计算请求成本
            cost = self.cost_calculator(estimated_tokens)
            
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False
    
    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

# 使用示例
def token_cost(estimated_tokens: int) -> float:
    """根据预估Token数计算成本"""
    # 每1000 tokens消耗1个令牌
    return estimated_tokens / 1000.0

limiter = LLMBucketRateLimiter(
    rate=10,  # 每秒10个令牌
    capacity=100,
    cost_calculator=token_cost
)

# 2000 tokens的请求消耗2个令牌
if limiter.allow(estimated_tokens=2000):
    # 执行LLM推理
    pass
```

### 2.2 漏桶算法（Leaky Bucket）

漏桶算法强制请求以固定速率处理，适合需要严格控制输出速率的场景：

```python
import asyncio
import collections

class LLMLeakyBucket:
    """LLM场景的漏桶限流器 - 强制固定处理速率"""
    
    def __init__(self, leak_rate: float, max_queue: int = 1000):
        self.leak_rate = leak_rate  # 每秒处理请求数
        self.queue = collections.deque()
        self.max_queue = max_queue
        self.last_leak = time.time()
        self._running = False
    
    async def submit(self, request) -> bool:
        """提交请求到队列"""
        if len(self.queue) >= self.max_queue:
            return False  # 队列已满，拒绝请求
        
        future = asyncio.Future()
        self.queue.append((request, future))
        
        if not self._running:
            asyncio.create_task(self._leak())
        
        return await future
    
    async def _leak(self):
        """以固定速率处理队列"""
        self._running = True
        while self.queue:
            now = time.time()
            elapsed = now - self.last_leak
            
            # 计算可以处理的请求数
            can_process = int(elapsed * self.leak_rate)
            
            for _ in range(min(can_process, len(self.queue))):
                request, future = self.queue.popleft()
                result = await self._process_request(request)
                future.set_result(result)
            
            self.last_leak = now
            await asyncio.sleep(1.0 / self.leak_rate)
        
        self._running = False
    
    async def _process_request(self, request):
        """实际处理LLM请求"""
        # 这里调用LLM推理
        return await llm_client.complete(request)
```

### 2.3 滑动窗口计数器（Sliding Window Counter）

滑动窗口计数器比固定窗口更精确，适合需要准确统计的场景：

```python
import time
import threading
from collections import defaultdict

class SlidingWindowRateLimiter:
    """滑动窗口限流器 - 用于精确控制时间段内的请求量"""
    
    def __init__(self, window_seconds: int = 60, max_requests: int = 100):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.windows = defaultdict(list)  # agent_id -> [timestamp, ...]
        self.lock = threading.Lock()
    
    def allow(self, agent_id: str) -> bool:
        """判断指定Agent是否允许请求"""
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # 清理过期记录
            self.windows[agent_id] = [
                ts for ts in self.windows[agent_id] if ts > cutoff
            ]
            
            # 检查是否超过限制
            if len(self.windows[agent_id]) >= self.max_requests:
                return False
            
            # 记录当前请求
            self.windows[agent_id].append(now)
            return True
    
    def get_usage(self, agent_id: str) -> dict:
        """获取当前使用情况"""
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            recent = [ts for ts in self.windows.get(agent_id, []) if ts > cutoff]
            return {
                "current": len(recent),
                "limit": self.max_requests,
                "utilization": len(recent) / self.max_requests,
            }
```

---

## 三、GPU感知的自适应限流

### 3.1 为什么需要GPU感知

LLM应用的核心资源是GPU。传统的限流器只关注请求层面的指标，无法感知GPU的真实状态：

```
场景：GPU显存接近满载

传统限流器视角：
  - QPS: 50（正常）
  - 错误率: 0%
  - 判断: 系统正常

GPU实际状态：
  - 显存使用: 95%
  - 显存碎片: 严重
  - 推理队列: 满载
  - 即将OOM

结果：新请求导致OOM，系统崩溃
```

### 3.2 GPU资源监控

```python
import subprocess
import json

class GPUMonitor:
    """GPU资源监控器"""
    
    def get_gpu_stats(self) -> dict:
        """获取GPU使用情况"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [x.strip() for x in line.split(',')]
                gpus.append({
                    "gpu_util": float(parts[0]),
                    "memory_used": float(parts[1]),
                    "memory_total": float(parts[2]),
                    "temperature": float(parts[3]),
                    "memory_util": float(parts[1]) / float(parts[2]) * 100,
                })
            return {"gpus": gpus}
        except Exception as e:
            return {"error": str(e)}
    
    def get_memory_pressure(self) -> float:
        """计算显存压力指数 (0-1)"""
        stats = self.get_gpu_stats()
        if "error" in stats:
            return 1.0  # 无法获取时假设最大压力
        
        # 取所有GPU中最高的显存使用率
        max_mem_util = max(gpu["memory_util"] for gpu in stats["gpus"])
        
        # 非线性映射：显存使用率超过80%时压力急剧上升
        if max_mem_util < 80:
            return max_mem_util / 100 * 0.5
        else:
            return 0.5 + (max_mem_util - 80) / 20 * 0.5
```

### 3.3 自适应限流器

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class RateLimitConfig:
    """限流配置"""
    base_rate: float = 50.0          # 基础QPS
    min_rate: float = 5.0            # 最低QPS（保护最小服务）
    max_rate: float = 200.0          # 最高QPS
    gpu_pressure_threshold: float = 0.8  # GPU压力阈值
    adjustment_interval: float = 5.0  # 调整间隔（秒）

class AdaptiveRateLimiter:
    """GPU感知的自适应限流器"""
    
    def __init__(self, config: RateLimitConfig, gpu_monitor: GPUMonitor):
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.current_rate = config.base_rate
        self.gpu_pressure_history = []
        self.request_count = 0
        self.window_start = time.time()
        self.lock = asyncio.Lock()
    
    async def allow(self, estimated_tokens: int = 2000) -> dict:
        """判断请求是否允许，并返回限流信息"""
        async with self.lock:
            now = time.time()
            
            # 定期调整速率
            if now - self.window_start >= self.config.adjustment_interval:
                await self._adjust_rate()
                self.window_start = now
            
            # 计算当前时间窗口内的请求数
            time_elapsed = now - self.window_start
            if time_elapsed > 0:
                current_qps = self.request_count / time_elapsed
            else:
                current_qps = 0
            
            # 判断是否允许
            if current_qps < self.current_rate:
                self.request_count += 1
                return {
                    "allowed": True,
                    "current_rate": self.current_rate,
                    "gpu_pressure": self.gpu_monitor.get_memory_pressure(),
                    "estimated_wait": 0,
                }
            else:
                # 计算等待时间
                wait_time = (1 / self.current_rate) - (time_elapsed / self.request_count)
                return {
                    "allowed": False,
                    "current_rate": self.current_rate,
                    "gpu_pressure": self.gpu_monitor.get_memory_pressure(),
                    "estimated_wait": max(0, wait_time),
                    "retry_after": max(0, wait_time),
                }
    
    async def _adjust_rate(self):
        """根据GPU压力动态调整速率"""
        pressure = self.gpu_monitor.get_memory_pressure()
        self.gpu_pressure_history.append(pressure)
        
        # 保持最近10个采样点
        if len(self.gpu_pressure_history) > 10:
            self.gpu_pressure_history.pop(0)
        
        # 计算平均压力
        avg_pressure = sum(self.gpu_pressure_history) / len(self.gpu_pressure_history)
        
        # 自适应调整
        if avg_pressure > self.config.gpu_pressure_threshold:
            # 压力过高，降低速率（最多降低50%）
            reduction = min(0.5, (avg_pressure - self.config.gpu_pressure_threshold) * 2)
            self.current_rate = max(
                self.config.min_rate,
                self.current_rate * (1 - reduction)
            )
        elif avg_pressure < self.config.gpu_pressure_threshold * 0.6:
            # 压力很低，提升速率（最多提升20%）
            increase = min(0.2, (self.config.gpu_pressure_threshold * 0.6 - avg_pressure))
            self.current_rate = min(
                self.config.max_rate,
                self.current_rate * (1 + increase)
            )
        
        print(f"[AdaptiveLimiter] GPU压力: {avg_pressure:.2f}, 调整后速率: {self.current_rate:.1f} QPS")
```

---

## 四、多维度限流策略

### 4.1 分层限流架构

生产级LLM应用需要多层限流协同工作：

```
请求入口
    │
    ▼
┌─────────────────────────────┐
│  第1层：全局QPS限流          │  ← 保护整个系统不被打垮
│  (令牌桶，固定速率)          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  第2层：用户/租户级限流       │  ← 防止单个用户占用过多资源
│  (滑动窗口，按用户ID分桶)    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  第3层：GPU感知限流           │  ← 根据GPU实际负载动态调整
│  (自适应，监控显存)          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  第4层：模型级限流            │  ← 不同模型独立控制
│  (按模型分组，独立配额)      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  第5层：Token预算限流         │  ← 控制总Token消耗成本
│  (滑动窗口，按Token计费)     │
└─────────────────────────────┘
```

### 4.2 完整实现

```python
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

@dataclass
class MultiLevelRateLimiter:
    """多层级限流器"""
    
    # 全局限流
    global_limiter: Optional[LLMBucketRateLimiter] = None
    
    # 用户级限流（每个用户独立）
    user_limiters: Dict[str, SlidingWindowRateLimiter] = field(default_factory=dict)
    
    # GPU感知限流
    adaptive_limiter: Optional[AdaptiveRateLimiter] = None
    
    # Token预算限制
    token_budget: Dict[str, float] = field(default_factory=dict)  # user_id -> 剩余token预算
    
    async def check_all(self, user_id: str, estimated_tokens: int, model: str) -> dict:
        """执行所有层级的限流检查"""
        results = {}
        
        # 1. 全局限流
        if self.global_limiter:
            results["global"] = self.global_limiter.allow(estimated_tokens)
        
        # 2. 用户级限流
        if user_id not in self.user_limiters:
            self.user_limiters[user_id] = SlidingWindowRateLimiter(
                window_seconds=60,
                max_requests=20  # 每用户每分钟最多20次
            )
        results["user"] = self.user_limiters[user_id].allow(user_id)
        
        # 3. GPU感知限流
        if self.adaptive_limiter:
            results["adaptive"] = await self.adaptive_limiter.allow(estimated_tokens)
        
        # 4. Token预算检查
        budget = self.token_budget.get(user_id, 100000)  # 默认10万token预算
        if estimated_tokens > budget:
            results["budget"] = False
        else:
            results["budget"] = True
            self.token_budget[user_id] = budget - estimated_tokens
        
        # 综合判断：所有层级都通过才允许
        allowed = all(results.values())
        
        return {
            "allowed": allowed,
            "details": results,
            "token_budget_remaining": self.token_budget.get(user_id, 0),
        }
```

### 4.3 按模型分组的限流

```python
class ModelAwareRateLimiter:
    """按模型分组的限流器"""
    
    def __init__(self):
        # 不同模型的处理能力差异很大
        self.model_configs = {
            "gpt-4o": {
                "qps_limit": 20,
                "tokens_per_second": 50000,
                "max_concurrent": 10,
            },
            "gpt-4o-mini": {
                "qps_limit": 100,
                "tokens_per_second": 100000,
                "max_concurrent": 50,
            },
            "claude-3.5-sonnet": {
                "qps_limit": 30,
                "tokens_per_second": 40000,
                "max_concurrent": 15,
            },
            "deepseek-v3": {
                "qps_limit": 50,
                "tokens_per_second": 80000,
                "max_concurrent": 25,
            },
        }
        
        # 为每个模型创建独立的限流器
        self.limiters = {}
        for model, config in self.model_configs.items():
            self.limiters[model] = {
                "qps": LLMBucketRateLimiter(
                    rate=config["qps_limit"],
                    capacity=config["qps_limit"] * 2,
                    cost_calculator=lambda x: x / 1000.0
                ),
                "concurrent": asyncio.Semaphore(config["max_concurrent"]),
            }
    
    async def acquire(self, model: str, estimated_tokens: int) -> bool:
        """获取模型配额"""
        if model not in self.limiters:
            return False
        
        limiter = self.limiters[model]
        
        # 检查QPS限制
        if not limiter["qps"].allow(estimated_tokens):
            return False
        
        # 等待并发槽位
        try:
            await asyncio.wait_for(
                limiter["concurrent"].acquire(),
                timeout=5.0  # 最多等待5秒
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def release(self, model: str):
        """释放并发槽位"""
        if model in self.limiters:
            self.limiters[model]["concurrent"].release()
```

---

## 五、优雅降级策略

### 5.1 降级级别定义

```python
from enum import Enum

class DegradationLevel(Enum):
    """降级级别"""
    NORMAL = 0          # 正常服务
    LIGHT = 1           # 轻度降级：限制长请求
    MODERATE = 2        # 中度降级：使用小模型
    SEVERE = 3          # 重度降级：只处理关键请求
    EMERGENCY = 4       # 紧急降级：返回缓存/模板响应
```

### 5.2 多级降级实现

```python
class GracefulDegradationManager:
    """优雅降级管理器"""
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.current_level = DegradationLevel.NORMAL
        
        # 降级策略配置
        self.strategies = {
            DegradationLevel.NORMAL: {
                "max_tokens": 4096,
                "model_fallback": None,
                "cache_ttl": 0,  # 不缓存
            },
            DegradationLevel.LIGHT: {
                "max_tokens": 2048,  # 限制输出长度
                "model_fallback": None,
                "cache_ttl": 60,  # 缓存1分钟
            },
            DegradationLevel.MODERATE: {
                "max_tokens": 1024,
                "model_fallback": "gpt-4o-mini",  # 降级到小模型
                "cache_ttl": 300,  # 缓存5分钟
            },
            DegradationLevel.SEVERE: {
                "max_tokens": 512,
                "model_fallback": "gpt-4o-mini",
                "cache_ttl": 600,  # 缓存10分钟
            },
            DegradationLevel.EMERGENCY: {
                "max_tokens": 0,  # 不执行推理
                "model_fallback": None,
                "cache_ttl": 3600,  # 缓存1小时
                "use_template": True,  # 使用模板响应
            },
        }
    
    def evaluate_and_set_level(self) -> DegradationLevel:
        """根据GPU压力评估并设置降级级别"""
        pressure = self.gpu_monitor.get_memory_pressure()
        
        if pressure < 0.5:
            self.current_level = DegradationLevel.NORMAL
        elif pressure < 0.7:
            self.current_level = DegradationLevel.LIGHT
        elif pressure < 0.85:
            self.current_level = DegradationLevel.MODERATE
        elif pressure < 0.95:
            self.current_level = DegradationLevel.SEVERE
        else:
            self.current_level = DegradationLevel.EMERGENCY
        
        return self.current_level
    
    def get_request_config(self) -> dict:
        """获取当前降级级别的请求配置"""
        return self.strategies[self.current_level]
    
    def should_reject_request(self, priority: str = "normal") -> bool:
        """判断是否应该拒绝请求"""
        if self.current_level == DegradationLevel.EMERGENCY:
            return priority != "critical"
        elif self.current_level == DegradationLevel.SEVERE:
            return priority not in ("high", "critical")
        return False
```

### 5.3 请求优先级队列

```python
import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PriorityRequest:
    """带优先级的请求"""
    priority: int  # 越小越优先
    timestamp: float = field(compare=False)
    request: Any = field(compare=False)
    user_id: str = field(compare=False)
    estimated_tokens: int = field(compare=False)

class PriorityQueueManager:
    """优先级队列管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = []
        self.priority_map = {
            "critical": 0,  # 关键请求（如付费用户、紧急任务）
            "high": 1,      # 高优先级
            "normal": 2,    # 普通请求
            "low": 3,       # 低优先级（如后台任务）
        }
    
    def enqueue(self, request, priority: str = "normal", user_id: str = "", estimated_tokens: int = 2000):
        """入队"""
        if len(self.queue) >= self.max_size:
            # 队列满时，尝试移除最低优先级的请求
            self._evict_lowest_priority()
        
        prio = self.priority_map.get(priority, 2)
        item = PriorityRequest(
            priority=prio,
            timestamp=time.time(),
            request=request,
            user_id=user_id,
            estimated_tokens=estimated_tokens,
        )
        heapq.heappush(self.queue, item)
    
    def dequeue(self) -> Optional[PriorityRequest]:
        """出队 - 返回最高优先级的请求"""
        if self.queue:
            return heapq.heappop(self.queue)
        return None
    
    def _evict_lowest_priority(self):
        """移除最低优先级的请求"""
        if not self.queue:
            return
        
        # 找到最低优先级的请求
        max_idx = max(range(len(self.queue)), key=lambda i: self.queue[i].priority)
        
        # 只有当最低优先级确实低于新请求时才移除
        if self.queue[max_idx].priority > 2:
            heapq.heappop(self.queue)
```

---

## 六、生产环境最佳实践

### 6.1 限流配置参考

```yaml
# 生产环境限流配置
rate_limiting:
  # 全局限流
  global:
    algorithm: token_bucket
    rate: 100  # QPS
    burst: 200  # 突发容量
  
  # 用户级限流
  per_user:
    algorithm: sliding_window
    window: 60s
    max_requests: 20
    max_tokens_per_minute: 100000
  
  # GPU感知限流
  gpu_aware:
    enabled: true
    check_interval: 5s
    pressure_threshold: 0.8
    adjustment_factor: 0.5
  
  # 模型级限流
  per_model:
    gpt-4o:
      qps: 20
      max_concurrent: 10
      max_tokens_per_request: 8192
    gpt-4o-mini:
      qps: 100
      max_concurrent: 50
      max_tokens_per_request: 4096
  
  # 降级策略
  degradation:
    enabled: true
    levels:
      light:
        gpu_pressure: 0.7
        actions: ["limit_output_length"]
      moderate:
        gpu_pressure: 0.85
        actions: ["switch_to_smaller_model", "enable_caching"]
      severe:
        gpu_pressure: 0.95
        actions: ["reject_low_priority", "max_concurrent_reduced"]
      emergency:
        gpu_pressure: 0.98
        actions: ["reject_all_non_critical", "return_cached_response"]
```

### 6.2 监控与告警

```python
# Prometheus指标定义
METRICS = {
    # 限流指标
    "llm_requests_total": "Counter - 总请求数",
    "llm_requests_limited": "Counter - 被限流的请求数",
    "llm_request_duration_seconds": "Histogram - 请求处理时间",
    
    # GPU指标
    "llm_gpu_memory_utilization": "Gauge - GPU显存使用率",
    "llm_gpu_utilization": "Gauge - GPU计算使用率",
    
    # 降级指标
    "llm_degradation_level": "Gauge - 当前降级级别",
    "llm_degradation_events_total": "Counter - 降级事件数",
    
    # 队列指标
    "llm_queue_size": "Gauge - 等待队列大小",
    "llm_queue_wait_seconds": "Histogram - 排队等待时间",
}

# 告警规则
ALERT_RULES = {
    "HighGPUPressure": {
        "condition": "llm_gpu_memory_utilization > 0.9 for 5m",
        "severity": "warning",
        "message": "GPU显存使用率超过90%，持续5分钟",
    },
    "HighRejectionRate": {
        "condition": "rate(llm_requests_limited[5m]) / rate(llm_requests_total[5m]) > 0.3",
        "severity": "critical",
        "message": "请求拒绝率超过30%，可能存在流量异常",
    },
    "LongQueueWait": {
        "condition": "histogram_quantile(0.99, llm_queue_wait_seconds) > 10",
        "severity": "warning",
        "message": "P99排队等待时间超过10秒",
    },
}
```

### 6.3 限流效果评估

```python
class RateLimitMetrics:
    """限流效果评估"""
    
    def __init__(self):
        self.total_requests = 0
        self.limited_requests = 0
        self.degradation_events = 0
        self.start_time = time.time()
    
    def record_request(self, allowed: bool, degradation_level: DegradationLevel):
        """记录请求"""
        self.total_requests += 1
        if not allowed:
            self.limited_requests += 1
        
        if degradation_level != DegradationLevel.NORMAL:
            self.degradation_events += 1
    
    def get_metrics(self) -> dict:
        """获取评估指标"""
        uptime = time.time() - self.start_time
        
        return {
            "total_requests": self.total_requests,
            "limited_requests": self.limited_requests,
            "limit_rate": self.limited_requests / max(1, self.total_requests),
            "degradation_events": self.degradation_events,
            "uptime_seconds": uptime,
            "avg_qps": self.total_requests / max(1, uptime),
        }
```

---

## 七、实战案例：高并发客服系统的限流方案

### 7.1 系统架构

```
┌─────────────────────────────────────────────────┐
│                   API Gateway                    │
│              (Nginx / Kong / Envoy)              │
│         全局限流 + IP级限流 + TLS终止            │
├─────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │ Rate Limit   │    │  Request Queue       │   │
│  │ Service      │    │  (Redis-backed)      │   │
│  │ (多层级限流)  │    │  (优先级队列)        │   │
│  └──────┬───────┘    └──────────┬───────────┘   │
│         │                       │               │
│  ┌──────▼───────────────────────▼───────────┐   │
│  │           LLM Router Service              │   │
│  │  - 模型选择    - 负载均衡    - 降级策略    │   │
│  └──────┬───────────────────────┬───────────┘   │
│         │                       │               │
│  ┌──────▼──────┐          ┌────▼──────────┐    │
│  │ GPU Cluster │          │ GPU Cluster   │    │
│  │ (GPT-4o)   │          │ (GPT-4o-mini) │    │
│  └─────────────┘          └───────────────┘    │
└─────────────────────────────────────────────────┘
```

### 7.2 核心代码

```python
class CustomerServiceRateLimiter:
    """客服系统专用限流器"""
    
    def __init__(self):
        self.multi_level = MultiLevelRateLimiter(
            global_limiter=LLMBucketRateLimiter(rate=100, capacity=200, cost_calculator=lambda x: x/1000),
            adaptive_limiter=AdaptiveRateLimiter(
                config=RateLimitConfig(base_rate=80, min_rate=10, max_rate=150),
                gpu_monitor=GPUMonitor()
            ),
        )
        self.degradation = GracefulDegradationManager(GPUMonitor())
        self.priority_queue = PriorityQueueManager(max_size=5000)
    
    async def handle_request(self, user_id: str, message: str, priority: str = "normal"):
        """处理客户请求"""
        estimated_tokens = len(message) * 2  # 粗略估算
        
        # 1. 检查降级级别
        self.degradation.evaluate_and_set_level()
        
        if self.degradation.should_reject_request(priority):
            return {"error": "系统繁忙，请稍后重试", "retry_after": 30}
        
        # 2. 执行多层级限流检查
        limit_result = await self.multi_level.check_all(user_id, estimated_tokens, "gpt-4o")
        
        if not limit_result["allowed"]:
            # 加入优先级队列等待
            self.priority_queue.enqueue(
                request={"user_id": user_id, "message": message},
                priority=priority,
                user_id=user_id,
                estimated_tokens=estimated_tokens,
            )
            
            # 计算预计等待时间
            queue_size = len(self.priority_queue.queue)
            estimated_wait = queue_size * 0.5  # 假设每个请求平均处理0.5秒
            
            return {
                "error": "请求排队中",
                "queue_position": queue_size,
                "estimated_wait_seconds": estimated_wait,
            }
        
        # 3. 获取降级配置
        config = self.degradation.get_request_config()
        
        # 4. 选择模型
        model = config.get("model_fallback") or "gpt-4o"
        
        # 5. 执行推理
        try:
            result = await llm_client.complete(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=config.get("max_tokens", 4096),
            )
            return {"response": result, "model": model, "degradation": self.degradation.current_level.name}
        except Exception as e:
            return {"error": f"推理失败: {str(e)}", "retry_after": 5}
```

---

## 八、总结

LLM应用的并发控制与限流是一个系统工程，需要从多个维度综合考虑：

### 核心要点回顾

1. **传统限流不够用**：LLM请求的高延迟、高成本特性需要更精细的限流策略
2. **多层限流协同**：全局 → 用户 → GPU → 模型 → Token，层层防护
3. **GPU感知是关键**：限流策略必须与GPU实际负载联动
4. **优雅降级保可用**：当资源不足时，通过降级保证核心服务可用
5. **监控告警不可少**：实时监控限流效果，及时发现和处理异常

### 实施路线图

```
阶段1（1-2周）：基础限流
  └─ 实现令牌桶 + 滑动窗口
  └─ 接入基础监控

阶段2（2-4周）：GPU感知
  └─ 接入GPU监控
  └─ 实现自适应限流
  └─ 添加降级策略

阶段3（1-2月）：完善优化
  └─ 多层限流协同
  └─ 优先级队列
  └─ 模型级限流
  └─ 完整监控告警

阶段4（持续）：迭代优化
  └─ 基于实际数据调优参数
  └─ 引入机器学习预测流量
  └─ 混沌工程验证韧性
```

构建高可用的LLM应用，限流与并发控制是基础设施中的基础设施。希望本文的方案能为你的LLM应用架构提供参考。

---

## 参考资料

1. Google SRE Book - Rate Limiting
2. AWS Well-Architected Framework - Throttling
3. NVIDIA TensorRT-LLM Performance Guide
4. OpenAI API Rate Limits Documentation
5. Anthropic Claude Rate Limits Guide
