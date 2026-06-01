---
title: "AI Service Mesh：LLM微服务的智能流量管理与负载均衡"
description: "深入解析AI Service Mesh架构，覆盖智能路由、负载均衡、熔断降级、可观测性等核心能力，结合Envoy/Istio/SGLang实战"
date: 2026-06-01
author: "RiceBall-15"
category: "architecture"
subCategory: microservices
tags: ["Service Mesh", "LLM微服务", "智能路由", "负载均衡", "Envoy", "Istio"]
draft: false
---

# AI Service Mesh：LLM微服务的智能流量管理与负载均衡

## 引言：为什么LLM需要Service Mesh？

传统的Service Mesh（如Istio/Envoy）主要解决HTTP/gRPC流量管理问题。但LLM服务有独特的挑战：

| 挑战 | 传统服务 | LLM服务 |
|------|---------|---------|
| **请求大小** | KB级别 | MB级别（长上下文） |
| **处理时间** | 毫秒级 | 秒级甚至分钟级 |
| **资源消耗** | CPU/内存 | GPU显存/算力 |
| **状态管理** | 无状态为主 | KV Cache有状态 |
| **流量模式** | 均匀分布 | 突发性+长尾延迟 |

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Service Mesh 核心能力                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  智能路由    │  │  负载均衡    │  │  熔断降级   │            │
│  │  Intelligent│  │   Load      │  │   Circuit   │            │
│  │   Routing   │  │  Balancing  │  │  Breaking   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│                 ┌─────────────┐                                 │
│                 │  可观测性    │                                 │
│                 │Observability│                                 │
│                 └─────────────┘                                 │
│                                                                 │
│  特殊能力：                                                     │
│  • GPU感知路由（根据GPU利用率调度）                              │
│  • 显存感知负载均衡（考虑KV Cache）                              │
│  • 模型版本路由（A/B测试、金丝雀发布）                           │
│  • 请求队列管理（优先级队列、公平调度）                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 一、AI Service Mesh架构设计

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Service Mesh 整体架构                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      控制平面                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │  配置管理    │  │  服务发现    │  │  策略引擎   │     │   │
│  │  │  Config     │  │  Service    │  │  Policy     │     │   │
│  │  │  Manager    │  │  Discovery  │  │  Engine     │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      数据平面                            │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │   │
│  │  │  Sidecar │    │  Sidecar │    │  Sidecar │             │   │
│  │  │  Proxy   │    │  Proxy   │    │  Proxy   │             │   │
│  │  └────┬────┘    └────┬────┘    └────┬────┘             │   │
│  │       │              │              │                    │   │
│  │  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐             │   │
│  │  │  LLM    │    │  LLM    │    │  LLM    │             │   │
│  │  │Service 1│    │Service 2│    │Service 3│             │   │
│  │  │(Llama3) │    │(GPT-4)  │    │(Claude) │             │   │
│  │  └─────────┘    └─────────┘    └─────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      可观测性层                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Metrics   │  │    Logs     │  │   Traces    │     │   │
│  │  │   (Prometheus)│ │  (Loki)     │  │  (Jaeger)   │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件设计

```python
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import time
import asyncio

class LBStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    GPU_AWARE = "gpu_aware"
    TOKEN_AWARE = "token_aware"

@dataclass
class LLMEndpoint:
    """LLM服务端点"""
    endpoint_id: str
    host: str
    port: int
    model_name: str
    model_version: str
    gpu_count: int
    gpu_memory: int  # GB
    max_context_length: int
    current_connections: int = 0
    gpu_utilization: float = 0.0
    kv_cache_usage: float = 0.0
    last_health_check: float = 0.0
    healthy: bool = True

@dataclass
class RequestMetadata:
    """请求元数据"""
    request_id: str
    model_name: str
    input_tokens: int
    max_output_tokens: int
    priority: int = 0
    timeout: float = 30.0
    requires_streaming: bool = False

class AIServiceMesh:
    """AI Service Mesh 核心类"""
    
    def __init__(self):
        self.endpoints: dict[str, LLMEndpoint] = {}
        self.lb_strategy: LBStrategy = LBStrategy.GPU_AWARE
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.rate_limiter: RateLimiter = RateLimiter()
        self.metrics: MeshMetrics = MeshMetrics()
    
    def register_endpoint(self, endpoint: LLMEndpoint):
        """注册LLM服务端点"""
        self.endpoints[endpoint.endpoint_id] = endpoint
        self.circuit_breakers[endpoint.endpoint_id] = CircuitBreaker(
            endpoint.endpoint_id
        )
        self.metrics.register_endpoint(endpoint.endpoint_id)
    
    async def route_request(self, metadata: RequestMetadata) -> LLMEndpoint:
        """智能路由：选择最佳端点"""
        # 1. 过滤健康端点
        healthy_endpoints = [
            ep for ep in self.endpoints.values() 
            if ep.healthy and ep.model_name == metadata.model_name
        ]
        
        if not healthy_endpoints:
            raise NoHealthyEndpointError(f"No healthy endpoint for {metadata.model_name}")
        
        # 2. 检查熔断器
        available_endpoints = [
            ep for ep in healthy_endpoints
            if not self.circuit_breakers[ep.endpoint_id].is_open
        ]
        
        if not available_endpoints:
            # 所有熔断器打开，尝试半开状态
            available_endpoints = [
                ep for ep in healthy_endpoints
                if self.circuit_breakers[ep.endpoint_id].is_half_open
            ]
        
        if not available_endpoints:
            raise AllCircuitBreakersOpenError("All circuit breakers are open")
        
        # 3. 应用负载均衡策略
        selected = self._apply_load_balancing(available_endpoints, metadata)
        
        # 4. 更新指标
        self.metrics.record_routing(metadata, selected)
        
        return selected
    
    def _apply_load_balancing(self, endpoints: list[LLMEndpoint], 
                             metadata: RequestMetadata) -> LLMEndpoint:
        """应用负载均衡策略"""
        if self.lb_strategy == LBStrategy.ROUND_ROBIN:
            return self._round_robin(endpoints)
        elif self.lb_strategy == LBStrategy.LEAST_CONNECTIONS:
            return self._least_connections(endpoints)
        elif self.lb_strategy == LBStrategy.GPU_AWARE:
            return self._gpu_aware(endpoints, metadata)
        elif self.lb_strategy == LBStrategy.TOKEN_AWARE:
            return self._token_aware(endpoints, metadata)
        else:
            return endpoints[0]
    
    def _round_robin(self, endpoints: list[LLMEndpoint]) -> LLMEndpoint:
        """轮询策略"""
        # 简化实现，实际应维护索引
        return endpoints[0]
    
    def _least_connections(self, endpoints: list[LLMEndpoint]) -> LLMEndpoint:
        """最少连接数策略"""
        return min(endpoints, key=lambda ep: ep.current_connections)
    
    def _gpu_aware(self, endpoints: list[LLMEndpoint], 
                   metadata: RequestMetadata) -> LLMEndpoint:
        """GPU感知策略：综合考虑GPU利用率和显存"""
        def score(ep: LLMEndpoint) -> float:
            # GPU利用率得分（越低越好）
            gpu_score = 1.0 - ep.gpu_utilization
            
            # 显存使用率得分（越低越好）
            memory_score = 1.0 - ep.kv_cache_usage
            
            # 连接数得分（越少越好）
            conn_score = 1.0 / (ep.current_connections + 1)
            
            # 加权综合得分
            return (gpu_score * 0.5 + 
                    memory_score * 0.3 + 
                    conn_score * 0.2)
        
        return max(endpoints, key=score)
    
    def _token_aware(self, endpoints: list[LLMEndpoint],
                     metadata: RequestMetadata) -> LLMEndpoint:
        """Token感知策略：考虑上下文长度和KV Cache"""
        def score(ep: LLMEndpoint) -> float:
            # 剩余上下文空间
            remaining_context = ep.max_context_length - metadata.input_tokens
            
            if remaining_context <= 0:
                return -1  # 无法处理此请求
            
            # KV Cache可用空间
            available_cache = 1.0 - ep.kv_cache_usage
            
            # 综合得分
            return remaining_context * available_cache
        
        return max(endpoints, key=score)

class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, endpoint_id: str, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0):
        self.endpoint_id = endpoint_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count: int = 0
        self.last_failure_time: float = 0
        self.state: str = "closed"  # closed, open, half_open
    
    @property
    def is_open(self) -> bool:
        return self.state == "open"
    
    @property
    def is_half_open(self) -> bool:
        if self.state == "open":
            # 检查是否可以进入半开状态
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                return True
        return self.state == "half_open"
    
    def record_success(self):
        """记录成功请求"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """记录失败请求"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, requests_per_second: int = 100):
        self.requests_per_second = requests_per_second
        self.tokens: float = requests_per_second
        self.last_refill: float = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """获取请求令牌"""
        async with self.lock:
            # 补充令牌
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.requests_per_second,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_refill = now
            
            # 尝试获取令牌
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
```

---

## 二、智能路由策略详解

### 2.1 多维度路由决策

```
┌─────────────────────────────────────────────────────────────────┐
│                    多维度路由决策流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   收到请求       │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │  模型名称匹配    │───▶│  过滤端点列表    │                   │
│  └─────────────────┘    └────────┬────────┘                   │
│                                  │                              │
│           ┌──────────────────────┴──────────────────────┐      │
│           ▼                                             ▼      │
│  ┌─────────────────┐                          ┌─────────────────┐│
│  │  上下文长度检查   │                          │  显存容量检查    ││
│  │  (input_tokens  │                          │  (kv_cache     ││
│  │   <= max_ctx)   │                          │   available)   ││
│  └────────┬────────┘                          └────────┬────────┘│
│           │                                            │        │
│           └──────────────────────┬─────────────────────┘        │
│                                  ▼                              │
│                       ┌─────────────────┐                      │
│                       │  GPU负载评估     │                      │
│                       │  (utilization   │                      │
│                       │   < threshold)  │                      │
│                       └────────┬────────┘                      │
│                                │                                │
│                                ▼                                │
│                       ┌─────────────────┐                      │
│                       │  优先级排序      │                      │
│                       │  (综合得分)      │                      │
│                       └────────┬────────┘                      │
│                                │                                │
│                                ▼                                │
│                       ┌─────────────────┐                      │
│                       │  选择最佳端点    │                      │
│                       └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 实现代码

```python
from typing import List, Tuple
import heapq

class SmartRouter:
    """智能路由器：多维度决策"""
    
    def __init__(self, mesh: AIServiceMesh):
        self.mesh = mesh
        self.config = {
            "max_gpu_utilization": 0.85,
            "max_kv_cache_usage": 0.90,
            "min_remaining_context": 1000,
            "priority_weights": {
                "gpu_score": 0.4,
                "memory_score": 0.3,
                "context_score": 0.2,
                "connection_score": 0.1
            }
        }
    
    def rank_endpoints(self, endpoints: List[LLMEndpoint],
                      metadata: RequestMetadata) -> List[Tuple[float, LLMEndpoint]]:
        """对端点进行多维度评分和排序"""
        scored_endpoints = []
        
        for ep in endpoints:
            score = self._calculate_score(ep, metadata)
            if score > 0:  # 只保留有效端点
                scored_endpoints.append((score, ep))
        
        # 按得分降序排序（最大堆）
        scored_endpoints.sort(reverse=True)
        
        return scored_endpoints
    
    def _calculate_score(self, ep: LLMEndpoint,
                        metadata: RequestMetadata) -> float:
        """计算端点综合得分"""
        weights = self.config["priority_weights"]
        
        # 1. GPU利用率得分（越低越好）
        if ep.gpu_utilization > self.config["max_gpu_utilization"]:
            return -1  # GPU过载，直接排除
        gpu_score = 1.0 - ep.gpu_utilization
        
        # 2. KV Cache使用率得分（越低越好）
        if ep.kv_cache_usage > self.config["max_kv_cache_usage"]:
            return -1  # 显存不足
        memory_score = 1.0 - ep.kv_cache_usage
        
        # 3. 上下文空间得分
        remaining_context = ep.max_context_length - metadata.input_tokens
        if remaining_context < self.config["min_remaining_context"]:
            return -1  # 上下文空间不足
        context_score = remaining_context / ep.max_context_length
        
        # 4. 连接数得分（越少越好）
        conn_score = 1.0 / (ep.current_connections + 1)
        
        # 加权综合得分
        total_score = (
            gpu_score * weights["gpu_score"] +
            memory_score * weights["memory_score"] +
            context_score * weights["context_score"] +
            conn_score * weights["connection_score"]
        )
        
        return total_score
    
    async def route_with_retry(self, metadata: RequestMetadata,
                              max_retries: int = 3) -> LLMEndpoint:
        """带重试的路由选择"""
        endpoints = [
            ep for ep in self.mesh.endpoints.values()
            if ep.healthy and ep.model_name == metadata.model_name
        ]
        
        if not endpoints:
            raise NoHealthyEndpointError("No healthy endpoint found")
        
        ranked = self.rank_endpoints(endpoints, metadata)
        
        for attempt in range(min(max_retries, len(ranked))):
            score, endpoint = ranked[attempt]
            
            # 检查熔断器
            breaker = self.mesh.circuit_breakers[endpoint.endpoint_id]
            if not breaker.is_open:
                return endpoint
        
        raise NoHealthyEndpointError("All endpoints exhausted")

class WeightedRoundRobin:
    """加权轮询负载均衡"""
    
    def __init__(self):
        self.current_weights: dict[str, int] = {}
    
    def select(self, endpoints: List[LLMEndpoint]) -> LLMEndpoint:
        """选择端点"""
        if not endpoints:
            raise ValueError("No endpoints available")
        
        # 计算权重
        total_weight = sum(ep.gpu_count for ep in endpoints)
        
        # 更新当前权重
        for ep in endpoints:
            current = self.current_weights.get(ep.endpoint_id, 0)
            self.current_weights[ep.endpoint_id] = current + ep.gpu_count
        
        # 选择权重最大的端点
        selected = max(endpoints, key=lambda ep: self.current_weights[ep.endpoint_id])
        
        # 减去总权重
        self.current_weights[selected.endpoint_id] -= total_weight
        
        return selected
```

---

## 三、负载均衡策略深入

### 3.1 策略对比

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **轮询** | 均匀负载 | 简单公平 | 不考虑实际负载 |
| **加权轮询** | 异构集群 | 考虑节点能力 | 权重需手动配置 |
| **最少连接** | 长连接场景 | 动态感知负载 | 需要实时统计 |
| **GPU感知** | LLM推理 | 考虑GPU特性 | 实现复杂 |
| **Token感知** | 长上下文 | 考虑显存限制 | 需要预测输入长度 |

### 3.2 实现代码

```python
from typing import Dict, List
from collections import defaultdict
import random

class AdaptiveLoadBalancer:
    """自适应负载均衡器：根据实时指标动态调整策略"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.strategy: str = "gpu_aware"
        self.adaptation_interval: float = 60.0  # 秒
        self.last_adaptation: float = 0
    
    def adapt_strategy(self, endpoints: List[LLMEndpoint]) -> str:
        """根据集群状态自适应调整策略"""
        current_time = time.time()
        
        # 检查是否需要调整策略
        if current_time - self.last_adaptation < self.adaptation_interval:
            return self.strategy
        
        # 分析集群状态
        avg_gpu_util = sum(ep.gpu_utilization for ep in endpoints) / len(endpoints)
        gpu_variance = sum(
            (ep.gpu_utilization - avg_gpu_util) ** 2 
            for ep in endpoints
        ) / len(endpoints)
        
        # 根据状态选择策略
        if gpu_variance > 0.1:  # GPU负载不均匀
            new_strategy = "gpu_aware"
        elif avg_gpu_util < 0.3:  # GPU负载较轻
            new_strategy = "round_robin"
        else:  # 正常负载
            new_strategy = "least_connections"
        
        self.last_adaptation = current_time
        self.strategy = new_strategy
        
        return new_strategy
    
    def select_endpoint(self, endpoints: List[LLMEndpoint],
                       metadata: RequestMetadata) -> LLMEndpoint:
        """选择端点"""
        # 自适应调整策略
        strategy = self.adapt_strategy(endpoints)
        
        if strategy == "gpu_aware":
            return self._gpu_aware_select(endpoints, metadata)
        elif strategy == "round_robin":
            return self._round_robin_select(endpoints)
        elif strategy == "least_connections":
            return self._least_connections_select(endpoints)
        else:
            return endpoints[0]
    
    def _gpu_aware_select(self, endpoints: List[LLMEndpoint],
                         metadata: RequestMetadata) -> LLMEndpoint:
        """GPU感知选择"""
        def score(ep: LLMEndpoint) -> float:
            # 考虑GPU利用率、显存使用、连接数
            gpu_score = 1.0 - ep.gpu_utilization
            memory_score = 1.0 - ep.kv_cache_usage
            conn_score = 1.0 / (ep.current_connections + 1)
            
            return gpu_score * 0.5 + memory_score * 0.3 + conn_score * 0.2
        
        return max(endpoints, key=score)
    
    def _round_robin_select(self, endpoints: List[LLMEndpoint]) -> LLMEndpoint:
        """轮询选择"""
        return endpoints[random.randint(0, len(endpoints) - 1)]
    
    def _least_connections_select(self, endpoints: List[LLMEndpoint]) -> LLMEndpoint:
        """最少连接选择"""
        return min(endpoints, key=lambda ep: ep.current_connections)

class TokenAwareBalancer:
    """Token感知负载均衡器：考虑上下文长度和KV Cache"""
    
    def __init__(self):
        self.token_usage_history: Dict[str, List[int]] = defaultdict(list)
    
    def estimate_required_tokens(self, metadata: RequestMetadata) -> int:
        """估算所需Token数"""
        # 简单估算：输入Token + 预期输出Token
        estimated_output = min(
            metadata.max_output_tokens,
            metadata.input_tokens * 2  # 假设输出约为输入的2倍
        )
        
        return metadata.input_tokens + estimated_output
    
    def select_endpoint(self, endpoints: List[LLMEndpoint],
                       metadata: RequestMetadata) -> LLMEndpoint:
        """选择能容纳请求的端点"""
        required_tokens = self.estimate_required_tokens(metadata)
        
        # 过滤能容纳请求的端点
        capable_endpoints = []
        
        for ep in endpoints:
            # 检查剩余上下文空间
            remaining_context = ep.max_context_length - metadata.input_tokens
            if remaining_context < metadata.input_tokens:
                continue
            
            # 检查KV Cache空间
            available_cache = ep.gpu_memory * (1 - ep.kv_cache_usage)
            # 粗略估算：1B参数模型约需要1GB显存存储1K token的KV Cache
            estimated_cache_needed = required_tokens * 0.001  # GB
            
            if available_cache < estimated_cache_needed:
                continue
            
            capable_endpoints.append(ep)
        
        if not capable_endpoints:
            raise InsufficientResourcesError("No endpoint can handle this request")
        
        # 在能处理的端点中选择最佳的
        return min(capable_endpoints, key=lambda ep: ep.kv_cache_usage)
```

---

## 四、熔断与降级机制

### 4.1 熔断器状态机

```
┌─────────────────────────────────────────────────────────────────┐
│                    熔断器状态机                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         ┌─────────────┐                         │
│          ┌─────────────▶│   CLOSED    │◀─────────────┐         │
│          │              │  (正常状态)  │              │         │
│          │              └──────┬──────┘              │         │
│          │                     │                     │         │
│          │                     │ 失败次数达到阈值     │         │
│          │                     ▼                     │         │
│          │              ┌─────────────┐              │         │
│          │              │    OPEN     │              │         │
│          │              │  (熔断状态)  │              │         │
│          │              └──────┬──────┘              │         │
│          │                     │                     │         │
│          │                     │ 恢复超时             │         │
│          │                     ▼                     │         │
│          │              ┌─────────────┐              │         │
│          │              │ HALF_OPEN   │──────────────┘         │
│          │              │  (半开状态)  │    测试请求成功         │
│          │              └─────────────┘                         │
│          │                     │                                 │
│          │                     │ 测试请求失败                     │
│          └─────────────────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 降级策略

```python
from typing import Callable, Any
from enum import Enum

class DegradationStrategy(Enum):
    """降级策略枚举"""
    FALLBACK_MODEL = "fallback_model"      # 切换到备用模型
    CACHED_RESPONSE = "cached_response"    # 返回缓存响应
    SIMPLIFIED_OUTPUT = "simplified_output" # 简化输出
    QUEUE_WITH_PRIORITY = "queue_with_priority"  # 队列等待

class DegradationManager:
    """降级管理器"""
    
    def __init__(self):
        self.strategies: Dict[str, DegradationStrategy] = {}
        self.fallback_models: Dict[str, str] = {}
        self.cache: Dict[str, Any] = {}
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    
    def configure_fallback(self, primary_model: str, fallback_model: str):
        """配置备用模型"""
        self.fallback_models[primary_model] = fallback_model
    
    async def handle_degradation(self, original_request: RequestMetadata,
                                error: Exception) -> Any:
        """处理降级"""
        strategy = self.strategies.get(
            original_request.model_name,
            DegradationStrategy.FALLBACK_MODEL
        )
        
        if strategy == DegradationStrategy.FALLBACK_MODEL:
            return await self._fallback_to_model(original_request)
        elif strategy == DegradationStrategy.CACHED_RESPONSE:
            return await self._return_cached(original_request)
        elif strategy == DegradationStrategy.SIMPLIFIED_OUTPUT:
            return await self._simplify_output(original_request)
        elif strategy == DegradationStrategy.QUEUE_WITH_PRIORITY:
            return await self._queue_request(original_request)
    
    async def _fallback_to_model(self, request: RequestMetadata) -> Any:
        """降级到备用模型"""
        fallback_model = self.fallback_models.get(request.model_name)
        
        if not fallback_model:
            raise NoFallbackModelError(f"No fallback model for {request.model_name}")
        
        # 修改请求使用备用模型
        fallback_request = RequestMetadata(
            request_id=f"{request.request_id}_fallback",
            model_name=fallback_model,
            input_tokens=request.input_tokens,
            max_output_tokens=request.max_output_tokens,
            priority=request.priority,
            timeout=request.timeout * 0.5  # 缩短超时
        )
        
        # 使用备用模型执行（简化示例）
        return {"model": fallback_model, "degraded": True}
    
    async def _return_cached(self, request: RequestMetadata) -> Any:
        """返回缓存响应"""
        cache_key = f"{request.model_name}:{hash(request)}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return {"cached": True, "data": cached}
        
        raise CacheMissError("No cached response available")
    
    async def _simplify_output(self, request: RequestMetadata) -> Any:
        """简化输出"""
        # 减少输出长度，降低复杂度
        simplified_request = RequestMetadata(
            request_id=f"{request.request_id}_simplified",
            model_name=request.model_name,
            input_tokens=request.input_tokens,
            max_output_tokens=min(request.max_output_tokens, 256),
            priority=request.priority,
            timeout=request.timeout * 0.3
        )
        
        return {"simplified": True, "max_tokens": 256}
    
    async def _queue_request(self, request: RequestMetadata) -> Any:
        """队列等待"""
        # 放入优先级队列
        await self.priority_queue.put((-request.priority, request))
        
        return {"queued": True, "position": self.priority_queue.qsize()}

class CircuitBreakerWithDegradation:
    """带降级的熔断器"""
    
    def __init__(self, endpoint_id: str, degradation_manager: DegradationManager):
        self.breaker = CircuitBreaker(endpoint_id)
        self.degradation_manager = degradation_manager
        self.failure_callbacks: List[Callable] = []
    
    async def execute_with_protection(self, 
                                     func: Callable,
                                     request: RequestMetadata) -> Any:
        """带保护的执行"""
        # 检查熔断器状态
        if self.breaker.is_open:
            # 尝试降级
            return await self.degradation_manager.handle_degradation(
                request, 
                CircuitBreakerOpenError("Circuit breaker is open")
            )
        
        try:
            # 执行请求
            result = await func()
            self.breaker.record_success()
            return result
            
        except Exception as e:
            self.breaker.record_failure()
            
            # 触发回调
            for callback in self.failure_callbacks:
                await callback(request, e)
            
            # 尝试降级
            return await self.degradation_manager.handle_degradation(request, e)
```

---

## 五、可观测性与监控

### 5.1 核心指标

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Service Mesh 核心指标                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    请求指标       │  │    延迟指标      │  │    资源指标      │ │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤ │
│  │ • 请求总数       │  │ • P50延迟       │  │ • GPU利用率      │ │
│  │ • 成功/失败率   │  │ • P95延迟       │  │ • GPU显存使用    │ │
│  │ • QPS          │  │ • P99延迟       │  │ • KV Cache使用   │ │
│  │ • 并发连接数    │  │ • 超时率        │  │ • CPU/内存使用   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    业务指标      │  │    模型指标      │  │    系统指标      │ │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤ │
│  │ • Token使用量   │  │ • 模型加载时间   │  │ • 熔断器状态     │ │
│  │ • 输出质量评分  │  │ • 推理吞吐量    │  │ • 降级触发次数   │ │
│  │ • 用户满意度    │  │ • 首Token延迟   │  │ • 重试次数       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 监控实现

```python
from dataclasses import dataclass, field
from typing import Dict, List
import time
from prometheus_client import Counter, Histogram, Gauge

@dataclass
class MeshMetrics:
    """Service Mesh指标收集器"""
    
    # Prometheus指标
    request_count = Counter(
        'llm_requests_total',
        'Total LLM requests',
        ['model', 'endpoint', 'status']
    )
    
    request_duration = Histogram(
        'llm_request_duration_seconds',
        'LLM request duration',
        ['model', 'endpoint'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )
    
    gpu_utilization = Gauge(
        'llm_gpu_utilization',
        'GPU utilization ratio',
        ['endpoint', 'gpu_id']
    )
    
    kv_cache_usage = Gauge(
        'llm_kv_cache_usage',
        'KV Cache usage ratio',
        ['endpoint']
    )
    
    active_connections = Gauge(
        'llm_active_connections',
        'Number of active connections',
        ['endpoint']
    )
    
    circuit_breaker_state = Gauge(
        'llm_circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=open, 2=half_open)',
        ['endpoint']
    )
    
    def __init__(self):
        self.metrics_history: Dict[str, List[dict]] = {}
    
    def record_request(self, model: str, endpoint: str, 
                      status: str, duration: float):
        """记录请求指标"""
        self.request_count.labels(
            model=model, endpoint=endpoint, status=status
        ).inc()
        
        self.request_duration.labels(
            model=model, endpoint=endpoint
        ).observe(duration)
    
    def update_gpu_metrics(self, endpoint: str, gpu_id: str,
                          utilization: float):
        """更新GPU指标"""
        self.gpu_utilization.labels(
            endpoint=endpoint, gpu_id=gpu_id
        ).set(utilization)
    
    def update_kv_cache(self, endpoint: str, usage: float):
        """更新KV Cache指标"""
        self.kv_cache_usage.labels(endpoint=endpoint).set(usage)
    
    def update_connections(self, endpoint: str, count: int):
        """更新连接数"""
        self.active_connections.labels(endpoint=endpoint).set(count)
    
    def update_circuit_breaker(self, endpoint: str, state: str):
        """更新熔断器状态"""
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        self.circuit_breaker_state.labels(endpoint=endpoint).set(
            state_map.get(state, 0)
        )
    
    def get_endpoint_stats(self, endpoint: str) -> dict:
        """获取端点统计"""
        # 从Prometheus查询实际指标（简化示例）
        return {
            "endpoint": endpoint,
            "request_count": 0,  # 实际从Prometheus查询
            "avg_duration": 0,
            "gpu_utilization": 0,
            "kv_cache_usage": 0,
            "active_connections": 0
        }

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_rules: List[dict] = []
        self.active_alerts: Dict[str, dict] = {}
    
    def add_rule(self, name: str, condition: Callable, 
                severity: str = "warning"):
        """添加告警规则"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "created_at": time.time()
        })
    
    def evaluate_rules(self, metrics: MeshMetrics):
        """评估告警规则"""
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    self._trigger_alert(rule["name"], rule["severity"])
                else:
                    self._resolve_alert(rule["name"])
            except Exception as e:
                print(f"Error evaluating rule {rule['name']}: {e}")
    
    def _trigger_alert(self, name: str, severity: str):
        """触发告警"""
        if name not in self.active_alerts:
            self.active_alerts[name] = {
                "severity": severity,
                "triggered_at": time.time(),
                "status": "firing"
            }
            print(f"🚨 Alert triggered: {name} (severity: {severity})")
    
    def _resolve_alert(self, name: str):
        """解决告警"""
        if name in self.active_alerts:
            del self.active_alerts[name]
            print(f"✅ Alert resolved: {name}")

# 使用示例
def setup_alerts(alert_manager: AlertManager):
    """设置告警规则"""
    
    # GPU利用率过高告警
    alert_manager.add_rule(
        "high_gpu_utilization",
        lambda m: m.gpu_utilization._value.get() > 0.9,
        severity="critical"
    )
    
    # P99延迟过高告警
    alert_manager.add_rule(
        "high_p99_latency",
        lambda m: True,  # 实际从Prometheus查询
        severity="warning"
    )
    
    # 熔断器打开告警
    alert_manager.add_rule(
        "circuit_breaker_open",
        lambda m: True,  # 实际从Prometheus查询
        severity="warning"
    )
```

---

## 六、Envoy + Istio集成实战

### 6.1 Envoy配置

```yaml
# envoy-llm-service.yaml
static_resources:
  listeners:
  - name: llm_listener
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 8080
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: llm_ingress
          route_config:
            name: llm_routes
            virtual_hosts:
            - name: llm_service
              domains: ["*"]
              routes:
              - match:
                  prefix: /v1/chat/completions
                route:
                  cluster: llm_cluster
                  timeout: 60s
                  retry_policy:
                    retry_on: "5xx,reset,connect-failure"
                    num_retries: 3
                    retry_back_off:
                      base_interval: 1s
                      max_interval: 10s
              - match:
                  prefix: /v1/embeddings
                route:
                  cluster: embedding_cluster
                  timeout: 30s
          http_filters:
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
          
          # 自定义Lua过滤器：LLM感知路由
          - name: envoy.filters.http.lua
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua
              inline_code: |
                function envoy_on_request(handle)
                  -- 获取模型名称
                  local model = handle:headers():get("x-model-name")
                  if model then
                    -- 根据模型设置路由
                    handle:headers():add("x-route-strategy", "model_aware")
                  end
                  
                  -- 获取GPU指标
                  local gpu_util = handle:headers():get("x-gpu-utilization")
                  if gpu_util and tonumber(gpu_util) > 0.9 then
                    -- GPU过载，触发降级
                    handle:headers():add("x-degradation-required", "true")
                  end
                end
  
  clusters:
  - name: llm_cluster
    type: STRICT_DNS
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: llm_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: llm-service-1
                port_value: 8000
            health_check_config:
              port_value: 8000
        - endpoint:
            address:
              socket_address:
                address: llm-service-2
                port_value: 8000
            health_check_config:
              port_value: 8000
    health_checks:
    - timeout: 5s
      interval: 10s
      unhealthy_threshold: 3
      healthy_threshold: 2
      http_health_check:
        path: /health
        host: llm-service
    circuit_breakers:
    thresholds:
    - priority: DEFAULT
      max_connections: 100
      max_pending_requests: 50
      max_requests: 200
      max_retries: 3
```

### 6.2 Istio配置

```yaml
# llm-virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-service
  namespace: ai-platform
spec:
  hosts:
  - llm-service
  http:
  - match:
    - headers:
        x-model-name:
          exact: "llama-3-70b"
    route:
    - destination:
        host: llm-service
        subset: llama-3-70b
        port:
          number: 8000
    timeout: 120s
    retries:
      attempts: 3
      perTryTimeout: 40s
      retryOn: "5xx,reset,connect-failure"
  
  - route:
    - destination:
        host: llm-service
        subset: default
        port:
          number: 8000
    timeout: 60s

---
# llm-destination-rule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: llm-service
  namespace: ai-platform
spec:
  host: llm-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: DEFAULT
        http1MaxPendingRequests: 50
        http2MaxRequests: 200
        maxRequestsPerConnection: 10
        maxRetries: 3
    
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
    
    loadBalancer:
      simple: LEAST_REQUEST
      localityLbSetting:
        enabled: true
  
  subsets:
  - name: llama-3-70b
    labels:
      model: llama-3-70b
    trafficPolicy:
      connectionPool:
        http:
          http2MaxRequests: 50  # 降低并发，避免GPU过载
  
  - name: gpt-4
    labels:
      model: gpt-4
    trafficPolicy:
      connectionPool:
        http:
          http2MaxRequests: 100
  
  - name: default
    trafficPolicy:
      loadBalancer:
        simple: ROUND_ROBIN

---
# llm-peer-authentication.yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: llm-service
  namespace: ai-platform
spec:
  selector:
    matchLabels:
      app: llm-service
  mtls:
    mode: STRICT
```

### 6.3 SGLang集成

```python
from sglang import Engine
from dataclasses import dataclass
from typing import List, Optional
import asyncio

@dataclass
class SGLangConfig:
    """SGLang引擎配置"""
    model_path: str
    tp_size: int = 1
    mem_fraction_static: float = 0.8
    max_num_reqs: int = 128
    max_total_tokens: int = 32768
    chunked_prefill_size: int = 8192
    dp_size: int = 1
    nnodes: int = 1
    nccl_port: int = 2333

class SGLangService:
    """SGLang服务封装"""
    
    def __init__(self, config: SGLangConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self.metrics: ServiceMetrics = ServiceMetrics()
    
    async def start(self):
        """启动SGLang引擎"""
        self.engine = Engine(
            model_path=self.config.model_path,
            tp_size=self.config.tp_size,
            mem_fraction_static=self.config.mem_fraction_static,
            max_num_reqs=self.config.max_num_reqs,
            max_total_tokens=self.config.max_total_tokens,
            chunked_prefill_size=self.config.chunked_prefill_size,
            dp_size=self.config.dp_size,
            nnodes=self.config.nnodes,
            nccl_port=self.config.nccl_port
        )
        
        print(f"SGLang engine started with model: {self.config.model_path}")
    
    async def generate(self, prompt: str, 
                      max_tokens: int = 1024,
                      temperature: float = 0.7,
                      stop: Optional[List[str]] = None) -> dict:
        """生成文本"""
        import time
        start_time = time.time()
        
        try:
            response = await self.engine.async_generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
            
            duration = time.time() - start_time
            
            # 记录指标
            self.metrics.record_generation(
                tokens=len(response),
                duration=duration
            )
            
            return {
                "text": response,
                "tokens": len(response),
                "duration": duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_error(duration)
            raise
    
    async def health_check(self) -> dict:
        """健康检查"""
        return {
            "status": "healthy",
            "model": self.config.model_path,
            "gpu_count": self.config.tp_size,
            "max_tokens": self.config.max_total_tokens
        }

class ServiceMetrics:
    """服务指标"""
    
    def __init__(self):
        self.total_generations = 0
        self.total_tokens = 0
        self.total_duration = 0.0
        self.error_count = 0
    
    def record_generation(self, tokens: int, duration: float):
        """记录生成"""
        self.total_generations += 1
        self.total_tokens += tokens
        self.total_duration += duration
    
    def record_error(self, duration: float):
        """记录错误"""
        self.error_count += 1
    
    def get_stats(self) -> dict:
        """获取统计"""
        return {
            "total_generations": self.total_generations,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_second": self.total_tokens / max(self.total_duration, 0.001),
            "error_rate": self.error_count / max(self.total_generations, 1)
        }
```

---

## 七、总结与最佳实践

### 7.1 架构选择建议

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Service Mesh 最佳实践                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 渐进式采用：                                               │
│     • 从简单的负载均衡开始                                       │
│     • 逐步添加智能路由和熔断                                     │
│     • 最后实现完整的可观测性                                      │
│                                                                 │
│  2. 关键配置：                                                 │
│     • 熔断器阈值：失败率 > 50% 时触发                           │
│     • 重试策略：最多3次，指数退避                                 │
│     • 超时设置：根据模型响应时间调整                              │
│     • 并发限制：根据GPU显存设置                                  │
│                                                                 │
│  3. 监控重点：                                                 │
│     • GPU利用率和显存使用                                       │
│     • 请求延迟分布（P50/P95/P99）                               │
│     • 熔断器状态变化                                            │
│     • 降级触发频率                                              │
│                                                                 │
│  4. 常见陷阱：                                                 │
│     • 避免过度配置：从简单开始                                   │
│     • 注意长连接：LLM请求可能持续数秒                            │
│     • 考虑显存限制：不同模型需要不同配置                          │
│     • 测试降级路径：确保降级逻辑正确                              │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 未来展望

1. **AI原生Service Mesh**：针对LLM特性的原生支持
2. **联邦学习集成**：多集群LLM服务的联邦调度
3. **边缘计算部署**：将LLM推理下沉到边缘节点
4. **自动扩缩容**：基于GPU利用率和请求队列的智能扩缩容

---

## 参考资料

1. Envoy Proxy: https://www.envoyproxy.io/
2. Istio: https://istio.io/
3. SGLang: https://github.com/sgl-project/sglang
4. vLLM: https://github.com/vllm-project/vllm
5. Service Mesh Patterns: https://istiobyexample.dev/
