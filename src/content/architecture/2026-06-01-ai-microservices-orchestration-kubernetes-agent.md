---
title: "AI微服务编排：基于Kubernetes的Agent部署、弹性伸缩与生产级治理"
description: "深度解析如何在Kubernetes上编排LLM Agent微服务，覆盖资源调度、弹性伸缩、灰度发布、故障自愈等生产级治理能力"
date: 2026-06-01
author: "RiceBall-15"
category: "architecture"
subCategory: microservices
tags: ["Kubernetes", "Agent编排", "LLM微服务", "弹性伸缩", "Helm", "KEDA", "生产部署"]
draft: false
---

# AI微服务编排：基于Kubernetes的Agent部署、弹性伸缩与生产级治理

## 引言：为什么Agent需要微服务架构？

当我们将一个AI Agent从单体应用拆分为微服务时，本质上是在回答一个核心问题：**如何让多个协作的LLM组件在分布式环境中可靠、高效地运行？**

一个典型的多Agent系统包含以下组件：

```
┌─────────────────────────────────────────────────────────────────────┐
│                       多Agent微服务架构全景                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐                   │
│  │  Planner  │    │  Coder    │    │  Reviewer │                   │
│  │  Agent    │    │  Agent    │    │  Agent    │                   │
│  │ (规划)    │    │ (编码)    │    │ (审查)    │                   │
│  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                   │
│        │                │                │                           │
│        └────────────────┼────────────────┘                           │
│                         ▼                                            │
│              ┌───────────────────┐                                   │
│              │  Orchestrator     │                                   │
│              │  (编排调度层)      │                                   │
│              └─────────┬─────────┘                                   │
│                        │                                             │
│        ┌───────────────┼───────────────┐                             │
│        ▼               ▼               ▼                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                       │
│  │ LLM Pool  │  │ Tool Pool │  │ Memory    │                       │
│  │ (模型服务) │  │ (工具服务) │  │ (记忆存储) │                       │
│  └───────────┘  └───────────┘  └───────────┘                       │
│                                                                     │
│  基础设施层:                                                        │
│  ┌──────────────────────────────────────────────────┐              │
│  │  Kubernetes + Istio + KEDA + Prometheus/Grafana  │              │
│  └──────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

与传统Web微服务相比，Agent微服务面临独特的挑战：

| 维度 | 传统Web微服务 | Agent微服务 | 影响 |
|------|-------------|-----------|------|
| **请求生命周期** | 毫秒~秒 | 秒~分钟（甚至小时） | 需要长连接管理、异步编排 |
| **资源模型** | CPU/内存 | GPU显存/算力/IO | 调度策略完全不同 |
| **状态管理** | 无状态为主 | KV Cache、对话状态 | 需要有状态调度 |
| **错误模式** | 超时/5xx | 幻觉/逻辑错误/超时 | 需要语义级别的重试策略 |
| **流量模式** | 均匀/可预测 | 突发+长尾分布 | 弹性伸缩策略差异大 |
| **依赖关系** | 数据库/缓存 | 其他Agent/LLM服务 | 需要DAG编排能力 |

## 一、Agent微服务的Kubernetes部署架构

### 1.1 Namespace与资源隔离

```yaml
# namespace.yaml - Agent微服务命名空间规划
apiVersion: v1
kind: Namespace
metadata:
  name: ai-agents
  labels:
    istio-injection: enabled  # 启用Istio Sidecar
    gpu-pool: "shared"        # GPU池标记
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: agent-quota
  namespace: ai-agents
spec:
  hard:
    requests.cpu: "32"
    requests.memory: "64Gi"
    requests.nvidia.com/gpu: "8"
    limits.cpu: "64"
    limits.memory: "128Gi"
    limits.nvidia.com/gpu: "16"
    pods: "100"
```

### 1.2 GPU资源调度策略

Agent微服务的GPU调度需要考虑两个关键因素：**显存占用**和**计算密度**。

```
┌──────────────────────────────────────────────────────────┐
│              GPU调度策略对比                               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  策略1: 静态分区 (Static Partitioning)                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Agent A │ │ Agent B │ │ Agent C │ │ Shared  │       │
│  │ 24GB    │ │ 24GB    │ │ 16GB    │ │ 24GB    │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│  优点: 隔离性好  缺点: 资源浪费大                          │
│                                                          │
│  策略2: 时间片共享 (Time-Sliced Sharing)                   │
│  ┌──────────────────────────────────────────┐            │
│  │  GPU Memory Partition (MIG/Time-slicing) │            │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │            │
│  │  │ A    │ │ B    │ │ C    │ │ A    │   │            │
│  │  └──────┘ └──────┘ └──────┘ └──────┘   │            │
│  └──────────────────────────────────────────┘            │
│  优点: 资源利用率高  缺点: 隔离性差                        │
│                                                          │
│  策略3: 动态弹性 (Dynamic Elastic - KEDA)                │
│  ┌──────────────────────────────────────────┐            │
│  │  按队列深度自动扩缩GPU Pod数量            │            │
│  │  queue_depth: 0 → min=1, queue_depth: 100│            │
│  │                   → min=8                 │            │
│  └──────────────────────────────────────────┘            │
│  优点: 平衡效率和隔离  推荐方案 ✅                        │
└──────────────────────────────────────────────────────────┘
```

```yaml
# coder-agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coder-agent
  namespace: ai-agents
  labels:
    app: coder-agent
    tier: agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: coder-agent
  template:
    metadata:
      labels:
        app: coder-agent
        tier: agent
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      containers:
        - name: coder-agent
          image: registry.internal/ai-agents/coder-agent:v2.3.1
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 50051
              name: grpc
          resources:
            requests:
              cpu: "2"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
          env:
            - name: LLM_ENDPOINT
              value: "http://llm-pool.ai-agents:8080"
            - name: MAX_CONCURRENT_REQUESTS
              value: "8"
            - name: GPU_MEMORY_FRACTION
              value: "0.85"
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 15
          volumeMounts:
            - name: model-cache
              mountPath: /models
            - name: kv-cache-storage
              mountPath: /kv-cache
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
        - name: kv-cache-storage
          emptyDir:
            medium: "Memory"  # 使用内存作为KV Cache存储
            sizeLimit: "8Gi"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values: ["coder-agent"]
                topologyKey: kubernetes.io/hostname
```

## 二、弹性伸缩策略：从HPA到KEDA

### 2.1 传统HPA的局限性

标准Kubernetes HPA基于CPU/内存指标进行扩缩容，但对Agent微服务存在明显不足：

| 扩缩容方案 | 指标来源 | 优势 | 劣势 | 适用场景 |
|-----------|---------|------|------|---------|
| **HPA (CPU/Memory)** | 系统指标 | 简单内置 | 无法感知GPU利用率 | 非GPU密集型Agent |
| **HPA (Custom)** | Prometheus | 灵活 | 需要自定义指标服务器 | GPU利用率伸缩 |
| **KEDA** | 多种事件源 | 事件驱动+缩至零 | 复杂度高 | 队列驱动的Agent |
| **Karpenter** | 节点级 | 智能节点调度 | 基础设施变更 | 大规模GPU集群 |

### 2.2 KEDA实现Agent弹性伸缩

KEDA（Kubernetes Event-driven Autoscaling）是Agent微服务弹性伸缩的最优选择，因为它支持基于**队列深度**、**自定义指标**等多维度触发。

```yaml
# keda-scaledobject-coder-agent.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: coder-agent-scaler
  namespace: ai-agents
spec:
  scaleTargetRef:
    name: coder-agent
  pollingInterval: 15
  cooldownPeriod: 300
  minReplicaCount: 1
  maxReplicaCount: 20
  advanced:
    restoreToOriginalReplicaCount: false
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 60
          policies:
            - type: Percent
              value: 100
              periodSeconds: 60
            - type: Pods
              value: 4
              periodSeconds: 60
          selectPolicy: Max
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
            - type: Percent
              value: 10
              periodSeconds: 120
          selectPolicy: Min
  triggers:
    # 触发器1: Redis队列深度（Agent任务队列）
    - type: redis
      metadata:
        address: redis-cluster.ai-agents:6379
        listName: agent-task-queue
        listLength: "10"
        databaseIndex: "0"
        activationListLength: "1"
    # 触发器2: Prometheus GPU利用率
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: gpu_utilization_percent
        query: |
          avg by (pod) (
            nvidia_gpu_utilization{
              namespace="ai-agents",
              pod=~"coder-agent-.*"
            }
          )
        threshold: "80"
        activationThreshold: "30"
    # 触发器3: 自定义Agent活跃度指标
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: agent_active_sessions
        query: |
          sum(agent_active_sessions{
            namespace="ai-agents",
            service="coder-agent"
          })
        threshold: "16"
```

### 2.3 多维度伸缩策略对比

```
┌───────────────────────────────────────────────────────────────┐
│                  Agent弹性伸缩策略决策树                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Agent任务进入?                                                │
│  │                                                            │
│  ├─ 队列深度 > 0 ──── 是 ──── 当前副本数 < maxReplicas?       │
│  │                       │         │                          │
│  │                       │         ├─ 是 → Scale Up            │
│  │                       │         │   (急迫: +2, 稳定: +1)    │
│  │                       │         └─ 否 → Queue Full Alert    │
│  │                       │                                    │
│  │                       └─ 否 ──── 当前副本数 > minReplicas?  │
│  │                                 │                          │
│  │                                 ├─ 是 → Scale Down          │
│  │                                 │   (cooldown: 5min)        │
│  │                                 └─ 否 → Idle Alert          │
│  │                                                            │
│  └─ GPU利用率 > 90% ──── 是 ──── 强制扩容（优先级最高）        │
│                                                            │
│  特殊策略:                                                    │
│  • 营业时间预热: 9:00前预扩容至预计负载的60%                    │
│  • 周末策略: 缩容至minReplicas，保留GPU节点预热               │
│  • 突发流量: 允许瞬时扩容至maxReplicas的120%                   │
└───────────────────────────────────────────────────────────────┘
```

## 三、Agent间通信与DAG编排

### 3.1 gRPC + 消息队列混合通信模式

Agent微服务之间的通信需要同时支持两种模式：

| 通信模式 | 技术方案 | 适用场景 | 延迟 | 可靠性 |
|---------|---------|---------|------|--------|
| **同步调用** | gRPC | 实时查询、健康检查 | 低(ms) | 中（需重试） |
| **异步编排** | Redis Streams / NATS | 任务分发、结果收集 | 中 | 高（持久化） |
| **事件广播** | Kafka / NATS JetStream | 状态变更通知 | 中 | 高 |

```python
# agent_orchestrator.py - DAG编排核心逻辑
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import asyncio
import redis.asyncio as redis

@dataclass
class AgentNode:
    """DAG中的Agent节点"""
    name: str
    service_name: str
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 2
    fallback_model: Optional[str] = None

class AgentDAGOrchestrator:
    """
    基于DAG的Agent编排器
    支持并行执行无依赖的Agent节点，自动处理依赖关系
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.agents: Dict[str, AgentNode] = {}
        self._grpc_stubs: Dict[str, any] = {}

    def register_agent(self, agent: AgentNode):
        """注册Agent节点到DAG"""
        # 环形依赖检测
        self.agents[agent.name] = agent
        if self._has_cycle():
            del self.agents[agent.name]
            raise ValueError(f"Circular dependency detected for {agent.name}")

    def _has_cycle(self) -> bool:
        """拓扑排序检测环形依赖"""
        visited = set()
        rec_stack = set()

        def dfs(node_name):
            visited.add(node_name)
            rec_stack.add(node_name)
            for dep in self.agents[node_name].dependencies:
                if dep in self.agents:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            rec_stack.discard(node_name)
            return False

        for name in self.agents:
            if name not in visited:
                if dfs(name):
                    return True
        return False

    def _get_execution_layers(self) -> List[List[str]]:
        """将DAG分解为可并行执行的层"""
        in_degree = {name: 0 for name in self.agents}
        for name, agent in self.agents.items():
            for dep in agent.dependencies:
                if dep in in_degree:
                    in_degree[name] += 1

        layers = []
        while in_degree:
            # 当前层: 入度为0的节点
            layer = [n for n, d in in_degree.items() if d == 0]
            if not layer:
                raise ValueError("Unresolvable dependencies in DAG")
            layers.append(layer)
            # 移除当前层，更新入度
            for n in layer:
                del in_degree[n]
            for remaining in in_degree:
                agent = self.agents[remaining]
                in_degree[remaining] -= sum(
                    1 for l in layer if l in agent.dependencies
                )

        return layers

    async def execute(self, task: dict) -> dict:
        """执行整个DAG"""
        layers = self._get_execution_layers()
        context = {"task": task, "results": {}}

        for layer_idx, layer in enumerate(layers):
            print(f"Executing layer {layer_idx}: {layer}")
            # 同一层的Agent并行执行
            tasks = [
                self._execute_agent(agent_name, context)
                for agent_name in layer
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent_name, result in zip(layer, results):
                if isinstance(result, Exception):
                    # 失败处理: 检查是否有降级策略
                    agent = self.agents[agent_name]
                    if agent.fallback_model:
                        context["results"][agent_name] = {
                            "status": "fallback",
                            "model": agent.fallback_model,
                            "error": str(result),
                        }
                    else:
                        raise
                else:
                    context["results"][agent_name] = result

        return context["results"]

    async def _execute_agent(self, agent_name: str, context: dict) -> dict:
        """执行单个Agent，包含重试和超时逻辑"""
        agent = self.agents[agent_name]

        # 从Redis发布任务到Agent服务队列
        task_payload = {
            "agent": agent_name,
            "dependencies": {
                dep: context["results"].get(dep)
                for dep in agent.dependencies
            },
            "task": context["task"],
        }

        task_id = await self.redis.xadd(
            f"agent:{agent.service_name}:tasks",
            {"payload": str(task_payload)},
        )

        # 等待结果（带超时）
        result = await self._wait_for_result(
            agent.service_name, task_id, agent.timeout_seconds
        )
        return result

    async def _wait_for_result(
        self, service: str, task_id: str, timeout: int
    ) -> dict:
        """轮询Redis等待Agent执行结果"""
        import time
        start = time.time()
        while time.time() - start < timeout:
            result = await self.redis.xrevrange(
                f"agent:{service}:results", count=1
            )
            if result and result[0][1].get(b"task_id") == task_id:
                return eval(result[0][1][b"payload"].decode())
            await asyncio.sleep(1)
        raise TimeoutError(f"Agent {service} timed out after {timeout}s")
```

### 3.2 Istio服务网格配置

```yaml
# istio-virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: coder-agent
  namespace: ai-agents
spec:
  hosts:
    - coder-agent
  http:
    - match:
        - headers:
            x-agent-priority:
              exact: "high"
      route:
        - destination:
            host: coder-agent
            subset: high-priority
      timeout: 300s
      retries:
        attempts: 2
        perTryTimeout: 60s
        retryOn: "5xx,reset,connect-failure"
    - route:
        - destination:
            host: coder-agent
            subset: default
          weight: 90
        - destination:
            host: coder-agent
            subset: canary
          weight: 10
      timeout: 600s
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: coder-agent
  namespace: ai-agents
spec:
  host: coder-agent
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: DEFAULT
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 30s
      baseEjectionTime: 120s
      maxEjectionPercent: 50
  subsets:
    - name: default
      labels:
        version: stable
    - name: canary
      labels:
        version: canary
    - name: high-priority
      labels:
        priority: high
```

## 四、生产级健康检查与故障自愈

### 4.1 多层健康检查体系

Agent微服务的健康检查比传统服务更复杂——需要区分**基础设施层**和**AI能力层**。

```python
# health_check.py - 多层健康检查实现
from fastapi import FastAPI, Response
from enum import Enum
import torch

app = FastAPI()

class HealthLevel(str, Enum):
    LIVE = "live"         # 进程存活
    READY = "ready"       # 可接收请求
    HEALTHY = "healthy"   # AI能力正常

@app.get("/health/live")
async def liveness():
    """Liveness: 进程是否存活"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """Readiness: 是否可以接收流量"""
    checks = {
        "llm_service": await _check_llm_service(),
        "redis_connection": await _check_redis(),
        "model_loaded": await _check_model_loaded(),
    }
    all_ok = all(checks.values())
    return Response(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )

@app.get("/health/healthy")
async def health():
    """Health: AI能力是否正常工作（含语义检测）"""
    checks = {
        "basic_inference": await _check_basic_inference(),
        "gpu_memory": await _check_gpu_memory(),
        "queue_depth": await _check_queue_depth(),
        "error_rate_5m": await _check_error_rate(),
    }
    degraded = sum(1 for v in checks.values() if v == "degraded")
    failed = sum(1 for v in checks.values() if v == "failed")

    if failed > 0:
        status_code = 503
        status = "unhealthy"
    elif degraded > 0:
        status_code = 200
        status = "degraded"
    else:
        status_code = 200
        status = "healthy"

    return Response(
        status_code=status_code,
        content={"status": status, "checks": checks},
    )

async def _check_basic_inference() -> bool:
    """发送一个简单的推理请求检测模型是否正常"""
    try:
        response = await llm_client.complete(
            model="current",
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5,
        )
        return "ok" in response.choices[0].message.content.lower()
    except Exception:
        return False

async def _check_gpu_memory() -> str:
    """检查GPU显存使用情况"""
    if not torch.cuda.is_available():
        return "degraded"
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    usage_ratio = allocated / total
    if usage_ratio > 0.95:
        return "failed"
    elif usage_ratio > 0.85:
        return "degraded"
    return "ok"
```

### 4.2 故障自愈流程

```
┌────────────────────────────────────────────────────────────────────┐
│                    Agent微服务故障自愈流程                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  检测阶段                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │ Liveness Probe│────▶│ Pod Restart  │────▶│  恢复流量     │       │
│  │ 失败 (3次)    │     │ (K8s自动)    │     │              │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
│                                                                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │ Readiness    │────▶│ 从Service    │────▶│ 排查后恢复    │       │
│  │ Probe 失败   │     │ 摘除流量     │     │ (自动回添)    │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
│                                                                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │ Health Probe │────▶│ 降级处理     │────▶│ 告警通知      │       │
│  │ degraded     │     │ (限流/切换)   │     │ (PagerDuty)  │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
│                                                                    │
│  GPU故障特殊路径:                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │ GPU ECC Error│────▶│ Pod Eviction │────▶│ 节点隔离      │       │
│  │ / Xid Error  │     │ + NoSchedule │     │ + 运维介入    │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
│                                                                    │
│  级联故障防护:                                                      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ Circuit Breaker: 5xx > 50% → Open (30s) → Half-Open │         │
│  │ Rate Limiter: 按Agent+User维度限流                     │         │
│  │ Fallback: 主Agent不可用 → 切换备用模型/降级响应        │         │
│  └──────────────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────────────┘
```

## 五、灰度发布与版本管理

### 5.1 Agent灰度发布策略

Agent的灰度发布需要考虑**模型版本**和**代码版本**的双重影响：

```yaml
# canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coder-agent-canary
  namespace: ai-agents
  labels:
    app: coder-agent
    version: canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: coder-agent
      version: canary
  template:
    metadata:
      labels:
        app: coder-agent
        version: canary
    spec:
      containers:
        - name: coder-agent
          image: registry.internal/ai-agents/coder-agent:v2.4.0-rc1
          env:
            - name: MODEL_VERSION
              value: "gpt-4o-2026-03"
            - name: CANARY
              value: "true"
            - name: EVALUATION_ENABLED
              value: "true"  # 启用A/B评测
```

### 5.2 自动化灰度评估

```
┌──────────────────────────────────────────────────────────────┐
│                 Agent灰度发布评估流程                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Shadow Mode (影子模式)                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │  100% 流量 → Stable                             │        │
│  │  100% 流量 → Canary (只记录，不返回)             │        │
│  │  持续时间: 24小时                                │        │
│  │  评估指标: 延迟P99, Token消耗, 错误率            │        │
│  └─────────────────────────────────────────────────┘        │
│                          │                                    │
│                    指标达标?                                   │
│                    ├─ 否 → 回滚                               │
│                    └─ 是 ↓                                    │
│                                                              │
│  Phase 2: Canary (5% 流量)                                   │
│  ┌─────────────────────────────────────────────────┐        │
│  │  95% 流量 → Stable                              │        │
│  │   5% 流量 → Canary (真实用户)                    │        │
│  │  持续时间: 48小时                                │        │
│  │  评估指标: 代码质量得分, 用户满意度, 任务完成率   │        │
│  └─────────────────────────────────────────────────┘        │
│                          │                                    │
│                    指标达标?                                   │
│                    ├─ 否 → 回滚                               │
│                    └─ 是 ↓                                    │
│                                                              │
│  Phase 3: Progressive (渐进式)                                │
│  ┌─────────────────────────────────────────────────┐        │
│  │  5% → 25% → 50% → 100%                         │        │
│  │  每个阶段持续 24小时                             │        │
│  │  每个阶段自动评估 + 人工审批门禁                  │        │
│  └─────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

## 六、可观测性与监控告警

### 6.1 Agent微服务监控指标体系

| 指标维度 | 具体指标 | 采集方式 | 告警阈值 |
|---------|---------|---------|---------|
| **延迟** | inference_latency_p99 | Prometheus histogram | > 30s |
| **吞吐** | requests_per_second | Prometheus counter | < 最小吞吐 |
| **GPU** | gpu_utilization | nvidia_exporter | > 95% 持续5min |
| **GPU** | gpu_memory_used | nvidia_exporter | > 90% |
| **Agent** | task_completion_rate | 自定义指标 | < 80% |
| **Agent** | agent_retry_count | 自定义指标 | > 5次/分钟 |
| **质量** | hallucination_rate | 语义检测 | > 2% |
| **成本** | token_cost_per_request | 自定义指标 | > 预算150% |
| **系统** | oom_killed_count | kube_events | > 0 |

### 6.2 Grafana Dashboard配置（Prometheus规则）

```yaml
# prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: agent-microservice-alerts
  namespace: monitoring
spec:
  groups:
    - name: agent-microservice
      rules:
        - alert: AgentHighLatency
          expr: |
            histogram_quantile(0.99,
              sum(rate(agent_inference_duration_seconds_bucket[5m])) by (service, le)
            ) > 30
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Agent {{ $labels.service }} P99延迟超过30秒"

        - alert: AgentGPUMemoryCritical
          expr: |
            nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
          for: 2m
          labels:
            severity: critical
          annotations:
            summary: "GPU显存使用率超过95%，可能导致OOM"

        - alert: AgentTaskQueueBacklog
          expr: |
            redis_llen("agent:coder-agent:tasks") > 100
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "Agent任务队列积压超过100，考虑扩容"
```

## 七、完整部署清单

### 7.1 Helm Chart目录结构

```
agent-microservice/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── namespace.yaml
│   ├── deployment-agent.yaml
│   ├── service-agent.yaml
│   ├── hpa.yaml
│   ├── keda-scaledobject.yaml
│   ├── istio-virtualservice.yaml
│   ├── istio-destinationrule.yaml
│   ├── servicemonitor.yaml
│   ├── prometheus-rules.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── pvc-model-cache.yaml
│   └── networkpolicy.yaml
├── values/
│   ├── production.yaml
│   ├── staging.yaml
│   └── development.yaml
└── README.md
```

### 7.2 values.yaml核心配置

```yaml
# values.yaml
replicaCount:
  min: 2
  max: 20

image:
  repository: registry.internal/ai-agents/agent-base
  tag: "v2.3.1"
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: "2"
    memory: "8Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "4"
    memory: "16Gi"
    nvidia.com/gpu: "1"

autoscaling:
  enabled: true
  type: keda  # keda | hpa
  keda:
    pollingInterval: 15
    cooldownPeriod: 300
    triggers:
      - type: redis
        listLength: "10"
      - type: prometheus
        metricName: gpu_utilization_percent
        threshold: "80"

istio:
  enabled: true
  retries:
    attempts: 2
    perTryTimeout: 60s
  timeout: 300s

monitoring:
  enabled: true
  serviceMonitor:
    interval: 15s
  alertRules: true
```

## 八、常见问题与最佳实践

### 8.1 Agent微服务部署检查清单

| 检查项 | 状态 | 说明 |
|--------|------|------|
| GPU资源配额设置 | ☐ | 避免单个namespace耗尽GPU |
| Pod反亲和性配置 | ☐ | Agent分散到不同节点 |
| Liveness/Readiness探针 | ☐ | 包含AI能力健康检查 |
| KEDA弹性伸缩配置 | ☐ | 队列+GPU利用率双维度 |
| Istio流量管理 | ☐ | 超时、重试、熔断配置 |
| NetworkPolicy | ☐ | Agent间网络隔离 |
| HPA最小副本数≥2 | ☐ | 保证高可用 |
| 模型缓存PVC | ☐ | 避免每次启动重新下载 |
| Prometheus监控 | ☐ | 覆盖延迟/GPU/Agent指标 |
| 灰度发布流程 | ☐ | 影子模式→金丝雀→渐进式 |

### 8.2 性能调优参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_concurrent_requests` | GPU数 × 2 | 避免GPU空闲或过载 |
| `kv_cache_memory_limit` | GPU显存 × 0.15 | 预留给KV Cache |
| `grpc_max_message_size` | 64MB | 支持长上下文传输 |
| `connection_pool_size` | max_concurrent × 1.5 | 避免连接耗尽 |
| `keda_cooldown_period` | 300s | 避免频繁扩缩 |
| `istio_retries.attempts` | 2 | Agent间重试不超过2次 |

## 总结

AI微服务编排的核心挑战在于平衡**性能**、**可靠性和** **成本**。本文的关键实践：

1. **资源调度**: 使用KEDA基于队列深度+GPU利用率双维度弹性伸缩
2. **通信模式**: gRPC同步 + Redis Streams异步混合，DAG编排多Agent协作
3. **健康检查**: 三层健康检查（存活→就绪→AI能力），含语义级检测
4. **灰度发布**: 影子模式→金丝雀→渐进式，每阶段含自动评估
5. **故障自愈**: Circuit Breaker + 降级策略 + GPU故障特殊路径

> 💡 **下一步**: 考虑引入Kueue进行Job级别的GPU资源管理，结合Volcano实现更精细的Batch调度，适用于大规模Agent训练/推理场景。
