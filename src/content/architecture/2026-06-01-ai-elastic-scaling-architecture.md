---
title: "AI应用弹性伸缩架构设计：从Kubernetes到Serverless的AI工作负载管理"
description: "深度解析AI应用弹性伸缩架构，涵盖GPU集群调度、Serverless推理、混合伸缩策略，附生产级K8s HPA配置与成本优化实战"
date: 2026-06-01
author: "RiceBall-15"
category: "architecture"
subCategory: cloud-native
tags: ["弹性伸缩", "Kubernetes", "Serverless", "GPU调度", "HPA", "AI基础设施", "云原生"]
draft: false
---

# AI应用弹性伸缩架构设计：从Kubernetes到Serverless的AI工作负载管理

## 一、引言：AI工作负载的弹性难题

### 1.1 传统Web应用 vs AI应用的伸缩差异

传统Web应用的伸缩相对直观——CPU使用率超过阈值就加实例，低于阈值就缩容。但AI应用的工作负载特征完全不同：

```
┌──────────────────────────────────────────────────────────────────────┐
│              传统Web应用 vs AI应用的伸缩差异                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  传统Web应用                                                         │
│  ┌────────────────────────────────────────────────────┐              │
│  │ 资源类型: CPU + 内存                                │              │
│  │ 启动时间: 毫秒级                                    │              │
│  │ 状态: 无状态                                       │              │
│  │ 异常模式: 请求量波动                                │              │
│  │ 伸缩触发: CPU/内存利用率                            │              │
│  │ 成本模型: 按实例计费，线性扩展                       │              │
│  └────────────────────────────────────────────────────┘              │
│                                                                      │
│  AI应用                                                              │
│  ┌────────────────────────────────────────────────────┐              │
│  │ 资源类型: GPU + 高带宽内存                           │              │
│  │ 启动时间: 分钟级（模型加载）                         │              │
│  │ 状态: 有状态（KV Cache、会话上下文）                  │              │
│  │ 异常模式: 突发峰值 + 长尾延迟                        │              │
│  │ 伸缩触发: GPU利用率/推理延迟/队列深度                 │              │
│  │ 成本模型: GPU资源昂贵，空转即亏损                     │              │
│  └────────────────────────────────────────────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**核心矛盾：GPU的高成本要求极致的利用率，但AI负载的突发性要求快速弹性。**

### 1.2 AI工作负载的三大典型场景

**场景一：实时推理服务（Online Inference）**

用户请求实时到达，要求低延迟响应（<500ms）。典型如聊天机器人、实时图像识别。特点是：
- 请求量波动大，白天是晚上的5-10倍
- 需要快速扩容，用户不能等待
- GPU利用率在高峰期可能只有40-60%（受序列化影响）

**场景二：批处理推理（Batch Inference）**

离线批量处理大量数据，对延迟不敏感但吞吐量要求高。典型如文档批量向量化、离线评估。特点是：
- 可以排队等待，不需要即时响应
- 可以利用低峰期资源，显著降低成本
- 需要高吞吐的调度策略

**场景三：训练工作负载（Training Workload）**

模型训练是长时间运行的GPU密集型任务。特点是：
- 运行时间从小时到天不等
- 需要多GPU分布式训练
- 可以被抢占（Preemptible）以降低成本

### 1.3 为什么通用伸缩方案不适用

Kubernetes原生的HPA（Horizontal Pod Autoscaler）基于CPU/内存指标做伸缩决策，这对AI应用有几个致命问题：

1. **CPU不是瓶颈**：AI推理的瓶颈在GPU，CPU利用率可能只有10%，但GPU已经100%
2. **冷启动太慢**：Pod启动后还需要加载模型（可能几百MB到几十GB），这段时间无法服务
3. **有状态问题**：KV Cache、会话状态无法随意迁移
4. **资源粒度太粗**：Pod request/limit设置的GPU数量是离散的（0.1、0.5、1、2），无法精细控制

---

## 二、架构全景：AI应用弹性伸缩的三层模型

```
┌──────────────────────────────────────────────────────────────────────┐
│                  AI弹性伸缩架构三层模型                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Layer 3: 流量调度层 (Traffic Scheduling)                    │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │    │
│  │  │ 请求路由   │  │ 队列管理   │  │ 降级策略   │             │    │
│  │  └────────────┘  └────────────┘  └────────────┘             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Layer 2: 实例伸缩层 (Instance Scaling)                      │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │    │
│  │  │ HPA/VPA    │  │ KEDA       │  │ Karpenter  │             │    │
│  │  │ 传统伸缩   │  │ 事件驱动   │  │ 节点伸缩   │             │    │
│  │  └────────────┘  └────────────┘  └────────────┘             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Layer 1: 资源管理层 (Resource Management)                    │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │    │
│  │  │ GPU共享    │  │ MIG/时间   │  │ 弹性GPU池  │             │    │
│  │  │ MPS/MIG    │  │ 分片复用   │  │ 多云调度   │             │    │
│  │  └────────────┘  └────────────┘  └────────────┘             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 三、Layer 1：GPU资源精细管理

### 3.1 GPU共享技术选型

在AI推理场景中，单个模型往往不需要整张GPU。NVIDIA提供了多种GPU共享方案：

| 方案 | 原理 | 隔离级别 | 适用场景 | 配置复杂度 |
|------|------|----------|----------|------------|
| **MPS** (Multi-Process Service) | 共享CUDA上下文 | 进程级 | 同一模型多实例 | 低 |
| **MIG** (Multi-Instance GPU) | 硬件级分区 | 硬件级 | A100/H100多租户 | 中 |
| **vGPU** (Time-slicing) | 时间片轮转 | 虚拟级 | 轻量推理 | 低 |
| **GPU Operator + DRA** | 动态资源分配 | K8s原生 | 大规模集群 | 高 |

**生产建议：**

- **小规模（<10个模型）**：使用MIG，硬件隔离最安全
- **中规模（10-100个模型）**：使用MPS + Kubernetes device plugin
- **大规模（>100个模型）**：使用GPU Operator + DRA动态分配

### 3.2 GPU资源分配的Kubernetes配置

```yaml
# 使用NVIDIA GPU Operator的Device Plugin配置
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference-worker
  labels:
    app: llm-inference
spec:
  containers:
  - name: inference
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu: "1"           # 请求1张GPU
        nvidia.com/gpumem: "16000"    # 限制显存16GB（Time-slicing模式）
      requests:
        nvidia.com/gpu: "1"
        nvidia.com/gpumem: "16000"
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "all"
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
    volumeMounts:
    - name: model-cache
      mountPath: /root/.cache/huggingface
  volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-cache-pvc
```

### 3.3 预热池（Warm Pool）策略

模型加载是AI伸缩的最大瓶颈。解决方案是维护一个预热池：

```
┌──────────────────────────────────────────────────────────────────────┐
│                      预热池策略示意                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  冷池       │    │  温池       │    │  热池       │              │
│  │  (Cold)     │    │  (Warm)     │    │  (Hot)      │              │
│  │             │    │             │    │             │              │
│  │  Pod不存在   │───▶│  Pod存在     │───▶│  Pod就绪     │              │
│  │  需要创建    │    │  模型已加载   │    │  可接受请求   │              │
│  │  +加载模型   │    │  等待预热完成 │    │             │              │
│  │             │    │             │    │             │              │
│  │  延迟: 3-5min│    │  延迟: 30s  │    │  延迟: 0ms   │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                                      │
│  伸缩决策:                                                            │
│  ├── 热池 > 阈值 → 直接缩容热池                                      │
│  ├── 热池 < 阈值 → 从温池提升到热池                                   │
│  ├── 温池 < 阈值 → 从冷池创建Pod并预加载                              │
│  └── 温池 > 阈值 → 从温池缩容到冷池                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**KEDA预热池配置示例：**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-inference-scaler
  namespace: ai-services
spec:
  scaleTargetRef:
    name: llm-inference-deployment
  minReplicaCount: 2          # 最小保持2个热实例
  maxReplicaCount: 20
  cooldownPeriod: 600         # 冷却10分钟（避免频繁伸缩）
  pollingInterval: 15
  advanced:
    restoreToOriginalReplicaCount: false
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 60    # 扩容稳定窗口
          policies:
          - type: Percent
            value: 100                      # 每次最多翻倍
            periodSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: 300   # 缩容稳定窗口（5分钟）
          policies:
          - type: Percent
            value: 25                       # 每次最多缩25%
            periodSeconds: 120
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: llm_queue_depth
      query: |
        sum(llm_request_queue_depth{service="llm-inference"})
      threshold: "10"
      activationThreshold: "5"
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: gpu_utilization
      query: |
        avg(DCGM_FI_DEV_GPU_UTIL{service="llm-inference"})
      threshold: "75"          # GPU利用率>75%时扩容
```

---

## 四、Layer 2：实例伸缩策略深度设计

### 4.1 基于KEDA的事件驱动伸缩

KEDA（Kubernetes Event-Driven Autoscaling）是AI工作负载伸缩的最佳选择，因为它支持自定义指标和事件驱动：

```
┌──────────────────────────────────────────────────────────────────────┐
│                  KEDA伸缩决策流程                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐               │
│  │ Prometheus │────▶│   KEDA     │────▶│ HPA        │               │
│  │ 指标采集   │     │ 伸缩决策   │     │ Pod伸缩    │               │
│  └────────────┘     └────────────┘     └────────────┘               │
│       │                   │                   │                      │
│       ▼                   ▼                   ▼                      │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐               │
│  │ GPU利用率   │     │ 多指标融合  │     │ Pod创建/销毁│               │
│  │ 队列深度   │     │ 权重计算   │     │ 模型预加载  │               │
│  │ 推理延迟   │     │ 冷却判断   │     │ 流量切换    │               │
│  │ 错误率     │     │            │     │             │               │
│  └────────────┘     └────────────┘     └────────────┘               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**多指标融合伸缩策略：**

```python
# 伸缩决策权重配置
SCALING_STRATEGY = {
    "metrics": [
        {
            "name": "gpu_utilization",
            "source": "prometheus",
            "query": "avg(DCGM_FI_DEV_GPU_UTIL{service='llm'})",
            "threshold_up": 70,      # GPU > 70% 扩容
            "threshold_down": 30,    # GPU < 30% 缩容
            "weight": 0.4,           # 权重40%
            "cooldown": 120,         # 冷却2分钟
        },
        {
            "name": "queue_depth",
            "source": "prometheus",
            "query": "sum(llm_queue_depth{service='llm'})",
            "threshold_up": 20,      # 队列 > 20 扩容
            "threshold_down": 5,     # 队列 < 5 缩容
            "weight": 0.3,           # 权重30%
            "cooldown": 60,
        },
        {
            "name": "p99_latency",
            "source": "prometheus",
            "query": "histogram_quantile(0.99, llm_request_duration_seconds_bucket)",
            "threshold_up": 2.0,     # P99 > 2s 扩容
            "threshold_down": 0.5,   # P99 < 0.5s 缩容
            "weight": 0.3,           # 权重30%
            "cooldown": 180,
        }
    ],
    # 综合得分计算
    # score = sum(weight * metric_score) / sum(weight)
    # metric_score: 在threshold_up和threshold_down之间归一化到0-1
    # score > 0.6 → 扩容
    # score < 0.3 → 缩容
    # 0.3 <= score <= 0.6 → 维持当前
}
```

### 4.2 Serverless推理：Cold Start的终极解决方案

对于请求量波动大的场景，Serverless推理可以实现真正的"按需付费"：

```
┌──────────────────────────────────────────────────────────────────────┐
│              Serverless推理架构对比                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  方案A: 传统K8s Deployment                                           │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                     │
│  │ GPU  │ │ GPU  │ │ GPU  │ │ GPU  │ │ GPU  │  始终运行              │
│  │ Pod1 │ │ Pod2 │ │ Pod3 │ │ Pod4 │ │ Pod5 │  即使空闲也收费        │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                     │
│  成本: $$$$$ (5×GPU小时)                                              │
│                                                                      │
│  方案B: Serverless (如AWS SageMaker Serverless Inference)            │
│  ┌──────┐              ┌──────┐              ┌──────┐               │
│  │ GPU  │  ──空闲──▶   │ GPU  │  ──空闲──▶   │ GPU  │               │
│  │ Pod1 │  缩到0      │ Pod1 │  缩到0      │ Pod1 │               │
│  └──────┘              └──────┘              └──────┘               │
│  成本: $ (仅按实际推理时间收费)                                         │
│                                                                      │
│  方案C: 混合模式（推荐）                                              │
│  ┌──────┐ ┌──────┐              ┌──────┐ ┌──────┐                  │
│  │ GPU  │ │ GPU  │              │ GPU  │ │ GPU  │                   │
│  │ 保底 │ │ 保底 │  + 弹性     │ 保底 │ │ 保底 │                   │
│  │ Pod1 │ │ Pod2 │  Serverless │ Pod1 │ │ Pod2 │                   │
│  └──────┘ └──────┘              └──────┘ └──────┘                  │
│  成本: $$ (保底 + 按需弹性)                                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**混合模式的Kubernetes实现：**

```yaml
# 保底层：Deployment（始终运行）
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-baseline
spec:
  replicas: 2                    # 始终保持2个实例
  template:
    spec:
      containers:
      - name: inference
        resources:
          limits:
            nvidia.com/gpu: "1"
---
# 弹性层：KEDA + Serverless
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-inference-elastic
spec:
  scaleTargetRef:
    name: llm-inference-elastic-deployment
  minReplicaCount: 0             # 可以缩到0
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      metricName: elastic_queue_depth
      query: |
        sum(llm_queue_depth{service="llm"}) 
        - (2 * avg(llm_inference_throughput{service="llm"}))
      threshold: "5"             # 队列超出保底处理能力5个请求时扩容
```

### 4.3 缩容安全：优雅下线策略

AI模型的缩容比扩容更危险——正在处理的请求不能被中断。优雅下线的关键是**Pod终止生命周期管理**：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Pod优雅终止时序                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  T=0s   收到SIGTERM信号                                               │
│  │                                                                    │
│  ▼                                                                    │
│  T=0s   preStop钩子执行: 调用 /drain 端点                             │
│  │      - 从Service endpoints中移除                                   │
│  │      - 停止接收新请求                                               │
│  │      - 等待当前请求处理完成                                          │
│  │                                                                    │
│  ▼                                                                    │
│  T=30s  preStop钩子结束（或超时）                                      │
│  │                                                                    │
│  ▼                                                                    │
│  T=30s  kubelet发送SIGTERM到容器                                      │
│  │      - 应用开始清理资源                                             │
│  │      - 保存KV Cache（如果需要）                                     │
│  │                                                                    │
│  ▼                                                                    │
│  T=45s  kubelet发送SIGKILL（如果还未退出）                             │
│  │                                                                    │
│  └───── 终止                                                          │
│                                                                      │
│  关键配置:                                                            │
│  - terminationGracePeriodSeconds: 60                                 │
│  - preStop exec: sleep 30 + curl /drain                              │
│  - readinessProbe: 在drain后返回unready                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: inference
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                # 1. 通知服务端停止接收新请求
                curl -X POST http://localhost:8080/drain
                # 2. 等待当前请求处理完成（最多30秒）
                sleep 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
```

---

## 五、Layer 3：流量调度与降级策略

### 5.1 智能请求路由

当多个模型版本或不同规格的实例同时运行时，需要智能路由：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    智能请求路由决策树                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        收到请求                                       │
│                          │                                            │
│                          ▼                                            │
│                    ┌──────────┐                                       │
│                    │ 请求分类 │                                       │
│                    └──────────┘                                       │
│                    │         │                                        │
│              ┌─────┘         └─────┐                                 │
│              ▼                      ▼                                 │
│        ┌──────────┐          ┌──────────┐                            │
│        │ 优先级高 │          │ 优先级低 │                            │
│        │ (VIP用户)│          │ (普通用户)│                            │
│        └──────────┘          └──────────┘                            │
│              │                      │                                 │
│              ▼                      ▼                                 │
│     ┌──────────────┐       ┌──────────────┐                         │
│     │ 路由到保底层  │       │ 检查弹性层   │                         │
│     │ (始终可用)   │       │ 是否有余量   │                         │
│     └──────────────┘       └──────────────┘                         │
│                                  │                                   │
│                          ┌───────┴───────┐                           │
│                          ▼               ▼                           │
│                    ┌──────────┐    ┌──────────┐                      │
│                    │ 有余量   │    │ 无余量   │                      │
│                    │ 路由到   │    │ 入队等待 │                      │
│                    │ 弹性层   │    │ 或降级   │                      │
│                    └──────────┘    └──────────┘                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 降级策略矩阵

当系统过载时，需要有序地降级以保护核心功能：

| 降级级别 | 触发条件 | 降级动作 | 用户影响 |
|----------|----------|----------|----------|
| **L1 轻度** | GPU > 80% 或 P99 > 2s | 关闭非核心功能（如流式输出） | 响应变慢，无流式 |
| **L2 中度** | GPU > 90% 或 队列 > 50 | 限制请求速率，排队等待 | 需要等待10-30s |
| **L3 重度** | GPU > 95% 或 队列 > 100 | 切换到小模型/量化模型 | 输出质量下降 |
| **L4 极限** | GPU故障或服务不可用 | 返回缓存结果/预设回复 | 仅能获取历史结果 |

**Nginx限流配置：**

```nginx
# AI推理服务的限流配置
upstream llm_backend {
    server llm-inference-baseline:8080;    # 保底层
    server llm-inference-elastic:8080;     # 弹性层（可选）
}

# 定义限流区域
limit_req_zone $binary_remote_addr zone=llm_rate:10m rate=10r/s;
limit_req_zone $server_name zone=llm_total:10m rate=100r/s;

server {
    listen 80;

    # 全局限流
    limit_req zone=llm_total burst=20 nodelay;

    # 单用户限流
    location /v1/chat/completions {
        limit_req zone=llm_rate burst=5 nodelay;
        
        # 超限返回429 + 重试建议
        limit_req_status 429;
        
        # 超时配置
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
        
        proxy_pass http://llm_backend;
        
        # 返回重试时间建议
        add_header Retry-After 30 always;
    }
}
```

---

## 六、生产实战：完整的伸缩架构配置

### 6.1 完整的Prometheus监控指标

```yaml
# AI推理服务关键监控指标
groups:
- name: llm_inference
  rules:
  # GPU利用率（5分钟平均）
  - record: llm:gpu_utilization:avg5m
    expr: avg_over_time(DCGM_FI_DEV_GPU_UTIL[5m])
  
  # 推理吞吐量（每秒请求数）
  - record: llm:throughput:rps
    expr: rate(llm_requests_total[5m])
  
  # P99延迟
  - record: llm:latency:p99
    expr: histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))
  
  # 队列深度
  - record: llm:queue:depth
    expr: sum(llm_request_queue_depth)
  
  # GPU显存使用率
  - record: llm:gpu_memory:utilization
    expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE
  
  # 告警规则
  - alert: LLMHighGPUUtilization
    expr: llm:gpu_utilization:avg5m > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "LLM推理GPU利用率过高"
      description: "GPU利用率已超过85%持续5分钟，考虑扩容"
  
  - alert: LLMHighLatency
    expr: llm:latency:p99 > 5
    for: 3m
    labels:
      severity: critical
    annotations:
      summary: "LLM推理P99延迟过高"
      description: "P99延迟超过5秒，用户可能已受影响"
```

### 6.2 成本优化：Spot实例 + 保底混合

```
┌──────────────────────────────────────────────────────────────────────┐
│                  成本优化混合策略                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  资源类型          │  占比   │  成本/小时  │  可用性   │  用途        │
│  ─────────────────┼────────┼───────────┼──────────┼──────────────  │
│  On-Demand GPU    │  30%   │  $3.00    │  100%    │  保底层       │
│  Spot GPU         │  50%   │  $0.90    │  90-95%  │  弹性层       │
│  Reserved GPU     │  20%   │  $1.80    │  100%    │  长期基线     │
│                                                                      │
│  月度成本对比（100 GPU小时/天）：                                      │
│  ├── 纯On-Demand: $3.00 × 100 × 30 = $9,000/月                     │
│  ├── 混合策略:     $3,870/月 (节省57%)                               │
│  └── 加上自动缩容: $2,900/月 (节省68%)                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Spot实例的容错配置：**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-spot
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1        # Spot实例回收时，最多1个不可用
      maxSurge: 2              # 允许多创建2个Pod以应对回收
  template:
    spec:
      # 优先调度到Spot实例
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 90
            preference:
              matchExpressions:
              - key: node.kubernetes.io/capacity-type
                operator: In
                values: ["spot"]
        # 分散到不同节点，降低同时回收风险
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values: ["llm-inference"]
              topologyKey: kubernetes.io/hostname
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: inference
        resources:
          limits:
            nvidia.com/gpu: "1"
          requests:
            nvidia.com/gpu: "1"
        # Spot实例回收时的优雅处理
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - "curl -X POST http://localhost:8080/drain && sleep 15"
```

---

## 七、性能基准与调优

### 7.1 伸缩响应时间基准

| 伸缩方式 | 冷启动时间 | 温启动时间 | 扩容速度 | 缩容安全窗口 |
|----------|-----------|-----------|----------|-------------|
| K8s HPA (新建Pod) | 3-5 min | 30-60s | 1-2 min | 5 min |
| KEDA (事件驱动) | 2-4 min | 20-40s | 30-60s | 3 min |
| Serverless (冷启动) | 5-10 min | 1-2 min | 1-3 min | N/A |
| Serverless (预热) | N/A | 10-30s | 10-30s | N/A |
| Karpenter (节点级) | 5-8 min | 1-2 min | 2-4 min | 5 min |

### 7.2 关键调优参数

```yaml
# 推荐的伸缩参数配置
scaling_config:
  # 扩容
  scale_up:
    stabilization_window: 60s       # 扩容稳定窗口（越小越激进）
    max_step_percent: 100           # 单次最大扩容比例
    cooldown: 120s                  # 扩容后冷却时间
  
  # 缩容
  scale_down:
    stabilization_window: 300s      # 缩容稳定窗口（越大越保守）
    max_step_percent: 25            # 单次最大缩容比例
    cooldown: 600s                  # 缩容后冷却时间
  
  # 预测性伸缩（如果支持）
  predictive:
    enabled: true
    time_window: "7d"               # 基于7天历史数据预测
    forecast_horizon: "1h"          # 预测未来1小时
    min_replicas_buffer: 2          # 预测值基础上多保留2个实例
```

---

## 八、总结：AI弹性伸缩的核心原则

```
┌──────────────────────────────────────────────────────────────────────┐
│                AI弹性伸缩五大核心原则                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 预测优于响应                                                      │
│     ├── 基于历史数据做预测性伸缩                                       │
│     ├── 维护预热池，消除冷启动                                         │
│     └── 利用定时任务预加载模型                                         │
│                                                                      │
│  2. 分层混合                                                          │
│     ├── 保底层保证基本可用性                                          │
│     ├── 弹性层处理峰值负载                                            │
│     └── Spot实例降低成本                                              │
│                                                                      │
│  3. 多指标融合                                                        │
│     ├── 不要只看GPU利用率                                             │
│     ├── 结合队列深度、延迟、错误率                                     │
│     └── 设置合理的权重和阈值                                          │
│                                                                      │
│  4. 优雅缩容                                                          │
│     ├── Pod终止前排空请求                                             │
│     ├── 预留足够的终止宽限期                                          │
│     └── 缩容比扩容更需要谨慎                                          │
│                                                                      │
│  5. 可观测性优先                                                      │
│     ├── 监控是伸缩决策的基础                                          │
│     ├── 建立完整的指标体系                                            │
│     └── 告警要及时，但不要过度                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**最后的忠告：** AI应用的弹性伸缩没有银弹。每个团队的业务场景、流量模型、成本预算都不同。关键是理解每种方案的适用边界，然后根据自己的实际情况做组合。从简单开始，逐步迭代，比一上来就搭建复杂的全自动伸缩系统更实际。

---

## 参考资源

- [Kubernetes HPA 文档](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [KEDA 官方文档](https://keda.sh/docs/)
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator)
- [Karpenter 文档](https://karpenter.sh/docs/)
- [vLLM 生产部署指南](https://docs.vllm.ai/en/latest/getting_started/production.html)
