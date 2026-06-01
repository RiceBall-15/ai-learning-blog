---
title: "LLM推理请求调度与QoS保障：从连续批处理到多级优先级队列"
description: "深入解析LLM推理系统中的请求调度策略，覆盖连续批处理、优先级队列、抢占式调度、多租户QoS保障等核心技术"
date: 2026-06-01
author: "RiceBall"
category: "aiInfra"
subCategory: "inference"
tags: ["推理优化", "请求调度", "QoS", "Continuous Batching", "GPU调度", "多租户"]
draft: false
---

## 引言：为什么请求调度是LLM推理的关键瓶颈？

传统的Web服务请求调度已经相当成熟——负载均衡、连接池、限流熔断，这些技术栈经过了数十年的工业验证。但LLM推理的请求调度面临完全不同的挑战：

| 维度 | 传统Web服务 | LLM推理服务 |
|------|------------|------------|
| **请求处理时间** | 毫秒级 | 秒级到分钟级 |
| **资源占用** | CPU为主，内存可共享 | GPU独占，显存关键 |
| **输出长度** | 固定（通常小） | 动态（数十到数万token） |
| **批处理特性** | 独立请求，天然可批处理 | 序列级并行，需考虑序列间干扰 |
| **排队成本** | 低（快速释放） | 高（GPU空闲=资金浪费） |
| **抢占可行性** | 高（中断代价小） | 低（KV Cache丢失代价大） |

> **核心挑战：LLM推理的GPU利用率在传统调度下通常只有20-40%。优秀的请求调度策略可以将利用率提升到70-90%，直接等于推理成本降低50%以上。**

本文将系统性地解析LLM推理中的请求调度策略，从基础的连续批处理到生产级的多级优先级调度。

---

## 一、连续批处理（Continuous Batching）：基石技术

### 1.1 传统批处理的局限

传统的静态批处理（Static Batching）要求所有请求在批处理开始时同步，结束时同步：

```
时间 →
请求A: [====生成====]
请求B: [========生成========]
请求C: [==生成==]
              ↑ 静态批处理等待点
              ↓ GPU在A、C完成后空转等待B
```

这导致严重的GPU资源浪费：短请求完成后必须等待最长请求结束。

### 1.2 连续批处理原理

连续批处理（Continuous Batching），也叫Iteration-Level Scheduling，在每个推理迭代（生成一个token）后重新调度：

```
时间 →
请求A: [====生成====] 完成，立即释放槽位
请求B: [========生成========]
请求C: [==生成==] 完成
请求D:     [====新加入====] 填充A释放的槽位
请求E:          [==新加入==] 填充C释放的槽位
```

#### 核心数据结构

```python
from dataclasses import dataclass, field
from typing import Optional
import time
import heapq

@dataclass
class InferenceRequest:
    """LLM推理请求"""
    request_id: str
    prompt_tokens: list[int]
    max_output_tokens: int
    priority: int = 0  # 0=普通, 1=高优先级, 2=关键
    arrival_time: float = field(default_factory=time.time)
    
    # 运行时状态
    generated_tokens: list[int] = field(default_factory=list)
    is_finished: bool = False
    finish_reason: Optional[str] = None  # "stop", "length", "preempted"
    
    # 调度信息
    scheduled_at: Optional[float] = None
    slot_id: Optional[int] = None
    
    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)
    
    @property
    def wait_time(self) -> float:
        if self.scheduled_at is None:
            return time.time() - self.arrival_time
        return self.scheduled_at - self.arrival_time
    
    def __lt__(self, other):
        """用于优先级队列排序"""
        if self.priority != other.priority:
            return self.priority > other.priority  # 高优先级优先
        return self.arrival_time < other.arrival_time  # FIFO fallback


class ContinuousBatchScheduler:
    """连续批处理器"""
    
    def __init__(self, max_batch_size: int, max_total_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        
        # 等待队列（按优先级+到达时间排序）
        self.waiting_queue: list[InferenceRequest] = []
        
        # 当前批次
        self.active_batch: list[InferenceRequest] = []
        
        # 统计
        self.stats = {
            "total_scheduled": 0,
            "total_preempted": 0,
            "avg_wait_time": 0,
            "gpu_utilization": 0
        }
    
    def submit(self, request: InferenceRequest):
        """提交新请求"""
        heapq.heappush(self.waiting_queue, request)
    
    def schedule_next_batch(self) -> list[InferenceRequest]:
        """在每个iteration后调度下一批"""
        # 1. 移除已完成的请求
        self.active_batch = [
            req for req in self.active_batch 
            if not req.is_finished
        ]
        
        # 2. 计算剩余容量
        current_tokens = sum(req.total_tokens for req in self.active_batch)
        remaining_capacity = self.max_total_tokens - current_tokens
        remaining_slots = self.max_batch_size - len(self.active_batch)
        
        # 3. 从等待队列填充
        newly_scheduled = []
        while (self.waiting_queue and 
               remaining_slots > 0 and
               remaining_capacity > 0):
            
            candidate = heapq.heappop(self.waiting_queue)
            
            # 检查是否能放入当前批次
            candidate_tokens = len(candidate.prompt_tokens)
            if candidate_tokens <= remaining_capacity:
                candidate.scheduled_at = time.time()
                candidate.slot_id = len(self.active_batch)
                self.active_batch.append(candidate)
                newly_scheduled.append(candidate)
                
                remaining_slots -= 1
                remaining_capacity -= candidate_tokens
                self.stats["total_scheduled"] += 1
            else:
                # 放回队列
                heapq.heappush(self.waiting_queue, candidate)
                break
        
        return self.active_batch
```

### 1.3 连续批处理的调度策略

#### 策略对比

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **FIFO** | 先来先服务 | 公平，简单 | 无优先级，短请求被阻塞 |
| **SJF（Shortest Job First）** | 预估输出长度，短的优先 | 平均延迟低 | 需要预估，长请求饥饿 |
| **Preemptive SJF** | 允许抢占长请求 | 更好的平均延迟 | 抢占开销 |
| **Priority** | 基于优先级 | 灵活，SLA友好 | 需要定义优先级 |
| **Fair Share** | 公平分配GPU时间 | 多租户公平 | 可能降低整体吞吐 |

#### SJF调度实现

```python
class SJFScheduler(ContinuousBatchScheduler):
    """最短作业优先调度器"""
    
    def __init__(self, *args, estimation_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimation_model = estimation_model  # 输出长度预估模型
    
    def estimate_output_length(self, request: InferenceRequest) -> int:
        """预估请求的输出长度"""
        if self.estimation_model:
            return self.estimation_model.predict(request.prompt_tokens)
        
        # 启发式预估：基于prompt长度和类型
        prompt_len = len(request.prompt_tokens)
        # 简单启发式：输出长度约为prompt长度的0.5-2倍
        estimated = min(
            prompt_len * 1.5,
            request.max_output_tokens
        )
        return int(estimated)
    
    def schedule_next_batch(self) -> list[InferenceRequest]:
        # 移除完成请求
        self.active_batch = [
            req for req in self.active_batch 
            if not req.is_finished
        ]
        
        current_tokens = sum(req.total_tokens for req in self.active_batch)
        remaining_capacity = self.max_total_tokens - current_tokens
        remaining_slots = self.max_batch_size - len(self.active_batch)
        
        # 按预估输出长度排序等待队列
        candidates = []
        temp_queue = []
        while self.waiting_queue:
            req = heapq.heappop(self.waiting_queue)
            estimated_output = self.estimate_output_length(req)
            candidates.append((estimated_output, req))
            temp_queue.append(req)
        
        # 恢复队列
        for req in temp_queue:
            heapq.heappush(self.waiting_queue, req)
        
        # 按预估长度排序（短的优先）
        candidates.sort(key=lambda x: x[0])
        
        newly_scheduled = []
        for estimated_output, candidate in candidates:
            if remaining_slots <= 0 or remaining_capacity <= 0:
                break
            
            candidate_tokens = len(candidate.prompt_tokens)
            if candidate_tokens <= remaining_capacity:
                candidate.scheduled_at = time.time()
                self.active_batch.append(candidate)
                newly_scheduled.append(candidate)
                remaining_slots -= 1
                remaining_capacity -= candidate_tokens
                self.stats["total_scheduled"] += 1
                # 从等待队列中移除
                self.waiting_queue = [
                    r for r in self.waiting_queue 
                    if r.request_id != candidate.request_id
                ]
                heapq.heapify(self.waiting_queue)
        
        return self.active_batch
```

---

## 二、多级优先级队列：生产级调度

### 2.1 设计理念

生产环境中，不同来源的请求有不同的SLA要求：

```
优先级层次设计：

P0 - 实时交互（<2s首token延迟）
     适用：Chat API、实时助手
     策略：独占预留槽位，抢占低优先级
     
P1 - 准实时（<10s首token延迟）  
     适用：流式输出、代码生成
     策略：高优先级队列，不抢占
     
P2 - 批量处理（<60s完成）
     适用：文档分析、批量翻译
     策略：普通队列，FIFO
     
P3 - 离线处理（Best Effort）
     适用：数据标注、模型评估
     策略：空闲时调度，可被抢占
```

### 2.2 多级优先级调度器实现

```python
import asyncio
from enum import IntEnum
from collections import defaultdict

class Priority(IntEnum):
    CRITICAL = 0    # P0: 实时交互
    HIGH = 1        # P1: 准实时
    NORMAL = 2      # P2: 批量处理
    LOW = 3         # P3: 离线处理

class MultiPriorityScheduler:
    """多级优先级调度器"""
    
    def __init__(
        self,
        max_batch_size: int,
        max_total_tokens: int,
        priority_reserved_slots: dict = None
    ):
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        
        # 每个优先级的预留槽位
        self.reserved_slots = priority_reserved_slots or {
            Priority.CRITICAL: 2,  # 预留2个槽位给关键请求
            Priority.HIGH: 1,
            Priority.NORMAL: 0,
            Priority.LOW: 0,
        }
        
        # 按优先级分组的等待队列
        self.waiting_queues: dict[Priority, list] = {
            p: [] for p in Priority
        }
        
        # 当前活跃批次
        self.active_batch: list[InferenceRequest] = []
        
        # 统计
        self.priority_stats = {
            p: {"submitted": 0, "scheduled": 0, "avg_wait": 0}
            for p in Priority
        }
    
    def submit(self, request: InferenceRequest, priority: Priority):
        """提交请求到对应优先级队列"""
        request.priority = priority
        self.waiting_queues[priority].append(request)
        self.priority_stats[priority]["submitted"] += 1
    
    def schedule_next_batch(self) -> list[InferenceRequest]:
        """多级优先级调度"""
        # 1. 移除完成请求
        self.active_batch = [
            req for req in self.active_batch 
            if not req.is_finished
        ]
        
        current_tokens = sum(req.total_tokens for req in self.active_batch)
        remaining_capacity = self.max_total_tokens - current_tokens
        remaining_slots = self.max_batch_size - len(self.active_batch)
        
        # 2. 按优先级从高到低填充
        for priority in Priority:
            if remaining_slots <= 0:
                break
            
            queue = self.waiting_queues[priority]
            slots_for_this_priority = remaining_slots
            
            # 关键优先级：检查预留槽位
            if priority == Priority.CRITICAL:
                reserved = self.reserved_slots[priority]
                used_reserved = sum(
                    1 for req in self.active_batch 
                    if req.priority == Priority.CRITICAL
                )
                # 确保预留槽位可用
                available_reserved = max(0, reserved - used_reserved)
                slots_for_this_priority = max(
                    slots_for_this_priority,
                    available_reserved
                )
            
            # 调度该优先级的请求
            i = 0
            while i < len(queue) and slots_for_this_priority > 0:
                candidate = queue[i]
                
                if len(candidate.prompt_tokens) <= remaining_capacity:
                    candidate.scheduled_at = time.time()
                    self.active_batch.append(candidate)
                    queue.pop(i)
                    slots_for_this_priority -= 1
                    remaining_slots -= 1
                    remaining_capacity -= len(candidate.prompt_tokens)
                    self.priority_stats[priority]["scheduled"] += 1
                else:
                    i += 1
        
        # 3. 检查抢占
        self._check_preemption(remaining_capacity)
        
        return self.active_batch
    
    def _check_preemption(self, remaining_capacity: int):
        """检查是否需要抢占低优先级请求"""
        # 如果有高优先级请求在等待，且当前批次中有低优先级请求
        for high_priority in [Priority.CRITICAL, Priority.HIGH]:
            if self.waiting_queues[high_priority]:
                # 找到可抢占的低优先级请求
                for i, req in enumerate(self.active_batch):
                    if req.priority > high_priority:
                        # 抢占这个请求
                        req.is_finished = True
                        req.finish_reason = "preempted"
                        self.active_batch.pop(i)
                        
                        # 将被抢占的请求放回等待队列（保持优先级）
                        self.waiting_queues[Priority(req.priority)].append(req)
                        self.priority_stats[Priority(req.priority)]["avg_wait"] += 1
                        break
    
    def get_metrics(self) -> dict:
        """获取调度指标"""
        active_by_priority = defaultdict(int)
        for req in self.active_batch:
            active_by_priority[req.priority] += 1
        
        waiting_by_priority = {
            p: len(q) for p, q in self.waiting_queues.items()
        }
        
        return {
            "active_batch_size": len(self.active_batch),
            "active_by_priority": dict(active_by_priority),
            "waiting_by_priority": waiting_by_priority,
            "gpu_utilization": self._estimate_gpu_utilization(),
            "priority_stats": self.priority_stats
        }
    
    def _estimate_gpu_utilization(self) -> float:
        """估算GPU利用率"""
        if not self.active_batch:
            return 0.0
        current_tokens = sum(req.total_tokens for req in self.active_batch)
        return current_tokens / self.max_total_tokens
```

---

## 三、抢占式调度与KV Cache管理

### 3.1 抢占的代价

LLM推理中的抢占与传统CPU调度有本质区别：**抢占一个正在生成的请求意味着丢失其KV Cache，恢复时需要重新计算**。

```
抢占代价分析：

请求状态: 已生成 500 tokens，KV Cache占 2GB显存
抢占后恢复: 需要重新处理500个prompt token
恢复时间: ~500ms（取决于GPU计算速度）

结论: 
- 抢占频繁 → 大量重复计算 → 吞吐下降
- 不抢占 → 高优先级请求延迟增加
- 需要平衡：只在必要时抢占，最小化抢占次数
```

### 3.2 KV Cache管理策略

```python
class KVCacheManager:
    """KV Cache管理器"""
    
    def __init__(self, total_gpu_memory: int, page_size: int = 16):
        """
        Args:
            total_gpu_memory: 总GPU显存（MB）
            page_size: KV Cache页大小（token数）
        """
        self.total_memory = total_gpu_memory
        self.page_size = page_size
        
        # 物理页管理
        self.total_pages = total_gpu_memory // (page_size * 2)  # 假设每token 2MB
        self.free_pages = self.total_pages
        self.page_table: dict[str, list[int]] = {}  # request_id -> page_ids
        
        # 预留页（用于紧急抢占恢复）
        self.reserved_pages = max(10, self.total_pages // 10)
    
    def can_allocate(self, request: InferenceRequest) -> bool:
        """检查是否有足够显存"""
        needed_pages = self._estimate_pages(request)
        available = self.free_pages - self.reserved_pages
        return needed_pages <= available
    
    def allocate(self, request: InferenceRequest) -> bool:
        """为请求分配KV Cache页"""
        needed_pages = self._estimate_pages(request)
        available = self.free_pages - self.reserved_pages
        
        if needed_pages > available:
            return False
        
        # 分配物理页
        allocated = []
        for _ in range(needed_pages):
            allocated.append(self._allocate_page())
        
        self.page_table[request.request_id] = allocated
        return True
    
    def deallocate(self, request_id: str):
        """释放请求的KV Cache页"""
        if request_id in self.page_table:
            for page_id in self.page_table[request_id]:
                self._free_page(page_id)
            del self.page_table[request_id]
    
    def preempt_with_kv_cache_preservation(
        self, 
        request: InferenceRequest
    ) -> dict:
        """
        抢占请求并尝试保留KV Cache
        如果显存不足，只能放弃KV Cache
        """
        if request.request_id not in self.page_table:
            return {"preserved": False, "reason": "no_cache"}
        
        pages = self.page_table[request.request_id]
        pages_needed = len(pages)
        
        # 检查是否有足够显存保留KV Cache
        if self.free_pages >= pages_needed:
            # 保留KV Cache（只是从活跃表移到保留表）
            return {
                "preserved": True,
                "pages": pages,
                "tokens_cached": request.total_tokens
            }
        else:
            # 必须释放KV Cache
            self.deallocate(request.request_id)
            return {
                "preserved": False,
                "reason": "insufficient_memory",
                "tokens_lost": request.total_tokens
            }
    
    def _estimate_pages(self, request: InferenceRequest) -> int:
        """估算请求需要的页数"""
        # 考虑prompt + 预估输出
        estimated_total = len(request.prompt_tokens) + request.max_output_tokens
        return (estimated_total + self.page_size - 1) // self.page_size
    
    def _allocate_page(self) -> int:
        """分配一个物理页"""
        # 简化实现：使用第一个空闲页
        for i in range(self.total_pages):
            if i not in self._used_pages():
                self.free_pages -= 1
                return i
        raise RuntimeError("No free pages available")
    
    def _free_page(self, page_id: int):
        """释放一个物理页"""
        self.free_pages += 1
    
    def _used_pages(self) -> set:
        """获取已使用的页集合"""
        used = set()
        for pages in self.page_table.values():
            used.update(pages)
        return used
```

---

## 四、多租户QoS保障

### 4.1 多租户架构设计

```
┌──────────────────────────────────────────────────────┐
│                   请求路由层                          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐            │
│  │租户A  │  │租户B  │  │租户C  │  │租户D  │            │
│  │SLA:高│  │SLA:中│  │SLA:低│  │SLA:高│            │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘            │
└─────┼─────────┼─────────┼─────────┼────────────────┘
      │         │         │         │
      ▼         ▼         ▼         ▼
┌──────────────────────────────────────────────────────┐
│              租户级配额管理                           │
│  - Token配额（每分钟/每天）                           │
│  - 并发槽位配额                                       │
│  - 优先级映射                                         │
│  - 突发容量控制                                       │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│              GPU资源池化                              │
│  ┌────────────────────────────────────────────┐      │
│  │         统一GPU调度器                        │      │
│  │  - 公平调度（Fair Share）                    │      │
│  - 容量预留（Capacity Reservation）             │      │
│  - 突发借用（Burst Borrowing）                  │      │
│  └────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

### 4.2 租户级QoS实现

```python
import time
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class TenantSLA:
    """租户SLA定义"""
    tenant_id: str
    priority: Priority
    max_concurrent_requests: int
    token_rate_limit: int  # tokens per minute
    max_queue_size: int
    max_latency_ms: int  # P99延迟要求
    burst_capacity: int  # 允许的突发容量

@dataclass 
class TenantQuota:
    """租户实时配额状态"""
    tenant_id: str
    current_concurrent: int = 0
    tokens_used_this_minute: int = 0
    tokens_used_this_hour: int = 0
    last_minute_reset: float = field(default_factory=time.time)
    last_hour_reset: float = field(default_factory=time.time)

class MultiTenantQoSManager:
    """多租户QoS管理器"""
    
    def __init__(self, scheduler: MultiPriorityScheduler):
        self.scheduler = scheduler
        self.tenant_slas: dict[str, TenantSLA] = {}
        self.tenant_quotas: dict[str, TenantQuota] = {}
        
        # 租户级统计
        self.tenant_metrics = defaultdict(lambda: {
            "total_requests": 0,
            "total_tokens": 0,
            "rejected_requests": 0,
            "avg_latency": 0,
            "p99_latency": 0
        })
    
    def register_tenant(self, sla: TenantSLA):
        """注册租户SLA"""
        self.tenant_slas[sla.tenant_id] = sla
        self.tenant_quotas[sla.tenant_id] = TenantQuota(
            tenant_id=sla.tenant_id
        )
    
    def submit_request(
        self, 
        tenant_id: str, 
        request: InferenceRequest
    ) -> dict:
        """
        提交请求，进行租户级准入控制
        
        Returns:
            {"accepted": bool, "reason": str, "estimated_wait": float}
        """
        sla = self.tenant_slas.get(tenant_id)
        quota = self.tenant_quotas.get(tenant_id)
        
        if not sla or not quota:
            return {"accepted": False, "reason": "unknown_tenant"}
        
        # 1. 检查并发配额
        if quota.current_concurrent >= sla.max_concurrent_requests:
            if sla.burst_capacity > 0:
                # 检查是否可以借用突发容量
                burst_used = quota.current_concurrent - sla.max_concurrent_requests
                if burst_used >= sla.burst_capacity:
                    return {
                        "accepted": False, 
                        "reason": "concurrent_limit_exceeded"
                    }
            else:
                return {
                    "accepted": False, 
                    "reason": "concurrent_limit_exceeded"
                }
        
        # 2. 检查Token速率限制
        self._reset_counters_if_needed(quota)
        if quota.tokens_used_this_minute >= sla.token_rate_limit:
            return {
                "accepted": False, 
                "reason": "rate_limit_exceeded",
                "retry_after": 60 - (time.time() % 60)
            }
        
        # 3. 检查队列深度
        queue_depth = len(self.scheduler.waiting_queues[sla.priority])
        if queue_depth >= sla.max_queue_size:
            return {
                "accepted": False, 
                "reason": "queue_full"
            }
        
        # 4. 通过准入控制，提交到调度器
        self.scheduler.submit(request, sla.priority)
        quota.current_concurrent += 1
        self.tenant_metrics[tenant_id]["total_requests"] += 1
        
        # 估算等待时间
        estimated_wait = self._estimate_wait_time(sla.priority, request)
        
        return {
            "accepted": True,
            "reason": "accepted",
            "estimated_wait": estimated_wait,
            "priority": sla.priority.name
        }
    
    def complete_request(
        self, 
        tenant_id: str, 
        request: InferenceRequest,
        tokens_generated: int
    ):
        """请求完成时更新配额"""
        quota = self.tenant_quotas.get(tenant_id)
        sla = self.tenant_slas.get(tenant_id)
        
        if quota and sla:
            quota.current_concurrent = max(
                0, quota.current_concurrent - 1
            )
            quota.tokens_used_this_minute += tokens_generated
            self.tenant_metrics[tenant_id]["total_tokens"] += tokens_generated
    
    def _estimate_wait_time(
        self, 
        priority: Priority, 
        request: InferenceRequest
    ) -> float:
        """估算请求等待时间"""
        queue = self.scheduler.waiting_queues[priority]
        queue_position = len(queue)
        
        # 简单估算：每个请求约需要2秒处理
        avg_time_per_request = 2.0
        estimated_wait = queue_position * avg_time_per_request
        
        # 考虑并发槽位
        batch_size = self.scheduler.max_batch_size
        estimated_wait = estimated_wait / max(1, batch_size // 2)
        
        return estimated_wait
    
    def _reset_counters_if_needed(self, quota: TenantQuota):
        """重置速率限制计数器"""
        now = time.time()
        if now - quota.last_minute_reset >= 60:
            quota.tokens_used_this_minute = 0
            quota.last_minute_reset = now
        if now - quota.last_hour_reset >= 3600:
            quota.tokens_used_this_hour = 0
            quota.last_hour_reset = now
    
    def get_tenant_dashboard(self, tenant_id: str) -> dict:
        """获取租户仪表板数据"""
        sla = self.tenant_slas.get(tenant_id)
        quota = self.tenant_quotas.get(tenant_id)
        metrics = self.tenant_metrics.get(tenant_id, {})
        
        if not sla or not quota:
            return {"error": "tenant not found"}
        
        return {
            "tenant_id": tenant_id,
            "sla": {
                "priority": sla.priority.name,
                "max_concurrent": sla.max_concurrent_requests,
                "rate_limit": sla.token_rate_limit,
                "max_latency": sla.max_latency_ms
            },
            "current_usage": {
                "concurrent_requests": quota.current_concurrent,
                "tokens_this_minute": quota.tokens_used_this_minute,
                "concurrent_utilization": (
                    quota.current_concurrent / sla.max_concurrent_requests
                )
            },
            "metrics": metrics
        }
```

---

## 五、调度策略对比与选型指南

### 5.1 策略对比矩阵

| 策略 | 吞吐量 | 延迟 | 公平性 | 实现复杂度 | 适用场景 |
|------|--------|------|--------|-----------|---------|
| **Static Batching** | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐ | 简单批处理 |
| **Continuous Batching** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 通用推理服务 |
| **SJF** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 有预估能力的场景 |
| **Preemptive SJF** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 高吞吐需求 |
| **Priority Queue** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 多SLA场景 |
| **Fair Share** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 多租户场景 |
| **Multi-Priority + Preemption** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 生产级多租户 |

### 5.2 选型决策树

```
需要LLM推理调度
│
├─ 单租户/单场景？
│   ├─ 是 → 低延迟优先？
│   │       ├─ 是 → SJF + Continuous Batching
│   │       └─ 否 → Continuous Batching（最简单）
│   │
│   └─ 否 → 多租户？
│           ├─ 是 → 需要SLA保障？
│           │       ├─ 是 → Multi-Priority + QoS
│           │       └─ 否 → Fair Share
│           │
│           └─ 否 → Priority Queue
│
└─ 需要抢占？
    ├─ 是 → Preemptive调度 + KV Cache管理
    └─ 否 → Non-preemptive调度
```

---

## 六、生产实践：关键经验

### 6.1 性能调优要点

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **max_batch_size** | 8-32 | 取决于GPU显存，太大会导致OOM |
| **max_total_tokens** | GPU显存的80% | 留20%给KV Cache增长 |
| **page_size** | 16-64 tokens | 权衡内存碎片和管理开销 |
| **preemption_threshold** | P99 > 3倍中位数 | 只在延迟严重偏离时抢占 |
| **rate_limit_window** | 60秒 | 令牌桶窗口大小 |

### 6.2 常见陷阱

1. **过度抢占**：抢占频率过高导致大量重复计算，反而降低吞吐
2. **KV Cache碎片化**：频繁分配/释放导致内存碎片，可用显存低于预期
3. **优先级反转**：低优先级请求持有关键资源，阻塞高优先级请求
4. **速率限制过于激进**：导致正常请求被拒绝，用户体验下降
5. **监控盲区**：只监控平均延迟，忽略P99和尾部延迟

### 6.3 监控指标清单

```python
# 必须监控的核心指标
CORE_METRICS = {
    # 吞吐指标
    "tokens_per_second": "系统整体token生成速率",
    "requests_per_minute": "每分钟处理的请求数",
    "batch_utilization": "平均batch大小 / 最大batch大小",
    
    # 延迟指标
    "ttft_p50": "首token延迟P50",
    "ttft_p99": "首token延迟P99",
    "tpot_p50": "每token生成延迟P50",
    "tpot_p99": "每token生成延迟P99",
    
    # 资源指标
    "gpu_utilization": "GPU计算利用率",
    "gpu_memory_utilization": "GPU显存利用率",
    "kv_cache_hit_rate": "KV Cache命中率",
    
    # 调度指标
    "queue_depth": "等待队列深度",
    "preemption_rate": "抢占率",
    "rejection_rate": "请求拒绝率",
    
    # 租户指标
    "per_tenant_latency": "每个租户的延迟分布",
    "per_tenant_throughput": "每个租户的吞吐",
    "per_tenant_rejection_rate": "每个租户的拒绝率"
}
```

---

## 结语

LLM推理的请求调度是一个兼具理论深度和工程挑战的领域。从基础的连续批处理到生产级的多级优先级调度，每一步优化都可能带来显著的性能提升和成本节约。

关键要点：

1. **连续批处理是基石**：所有生产级调度系统都基于连续批处理
2. **优先级是SLA的保障**：多优先级队列确保关键场景的延迟要求
3. **抢占需要权衡**：KV Cache的保存/丢失直接影响系统吞吐
4. **多租户需要配额管理**：防止一个租户影响其他租户的SLA
5. **监控是持续优化的基础**：没有监控的调度就是盲人摸象

随着LLM应用场景的多样化，请求调度将变得越来越复杂。从单模型到混合模型路由，从单GPU到多节点集群，调度策略需要不断演进以适应新的需求。希望本文提供的技术框架和实践经验，能帮助你在构建LLM推理服务时，建立起高效、可靠的调度体系。
