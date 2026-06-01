---
title: "AI模型服务编排架构：从Kubernetes到Serverless的演进实践"
description: "深入解析AI模型服务的编排架构设计，对比K8s原生部署、KServe、BentoML与Serverless方案，结合生产实战总结最佳实践"
date: 2026-06-01
author: "RiceBall"
category: "architecture"
subCategory: cloud-native
tags: ["Kubernetes", "KServe", "Serverless", "模型服务", "AI架构", "MLOps"]
draft: false
---

## 引言

在AI系统从实验走向生产的旅程中，模型训练只是起点，**如何高效、稳定、弹性地将模型部署为服务**才是真正的挑战。随着企业内部模型数量从几个增长到几十甚至上百个，简单的Docker + Load Balancer方案已经不堪重负。

本文将系统性地剖析AI模型服务编排架构的演进路径，从最基础的Kubernetes原生部署到成熟的KServe方案，再到极致弹性的Serverless架构，并分享我们在生产环境中的实战经验与踩坑教训。

## 架构演进全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI模型服务编排架构演进                              │
├─────────────┬─────────────┬─────────────┬─────────────────────────┤
│  阶段一      │  阶段二      │  阶段三      │  阶段四                   │
│  手动部署    │  K8s原生     │  KServe     │  Serverless              │
│  ─────────  │  ─────────  │  ─────────  │  ─────────              │
│  Docker +   │  Deployment │  Inference  │  Knative +               │
│  Docker     │  + Service  │  Service    │  自动扩缩容               │
│  Compose    │  + HPA      │  + Gateway  │  按需计费                 │
├─────────────┼─────────────┼─────────────┼─────────────────────────┤
│  适合: POC  │  适合: 5-20 │  适合: 20+  │  适合: 流量波动大          │
│  1-3个模型  │  模型        │  模型       │  峰谷比 > 10:1            │
└─────────────┴─────────────┴─────────────┴─────────────────────────┘
```

## 阶段一：Kubernetes原生部署方案

### 基础架构

这是最直接的方式，将每个模型包装为独立的Deployment + Service：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis-v2
  labels:
    app: sentiment-analysis
    version: v2
    framework: pytorch
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sentiment-analysis
      version: v2
  template:
    metadata:
      labels:
        app: sentiment-analysis
        version: v2
    spec:
      containers:
      - name: model-server
        image: registry.internal/sentiment-analysis:v2.1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "4"
            memory: "8Gi"
            nvidia.com/gpu: "1"
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
          periodSeconds: 30
        env:
        - name: MODEL_PATH
          value: "/models/sentiment-analysis/v2"
        - name: MAX_BATCH_SIZE
          value: "32"
        - name: TENSORRT_ENABLED
          value: "true"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      nodeSelector:
        accelerator: nvidia-a100
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-analysis-v2-svc
spec:
  selector:
    app: sentiment-analysis
    version: v2
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

### 自动扩缩容配置

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-analysis-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-analysis-v2
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: inference_queue_depth
      target:
        type: AverageValue
        averageValue: "5"
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

### 方案对比表

```
┌──────────────────┬─────────────────────────────────────────┐
│      维度         │            K8s原生方案                   │
├──────────────────┼─────────────────────────────────────────┤
│  部署复杂度       │  中等（需要手写大量YAML）                  │
│  GPU调度         │  需要手动配置nodeSelector/tolerations      │
│  模型版本管理     │  需要自行实现（Annotation + Label）       │
│  流量分配        │  需要额外配置Ingress/Istio                 │
│  自动扩缩容      │  支持（HPA基于自定义指标）                  │
│  模型预热        │  需要自行实现Init Container                │
│  多框架支持      │  无统一抽象，每个框架单独处理               │
│  适用模型数      │  < 20个                                   │
└──────────────────┴─────────────────────────────────────────┘
```

**关键痛点**：当模型数量增长到20+时，维护大量的Deployment YAML变得极为繁琐，且模型版本管理、A/B测试、流量切换等能力需要大量定制开发。

## 阶段二：KServe——专为AI模型设计的推理平台

### 架构概览

KServe（前KFServing）是Kubernetes上标准化的模型推理平台，解决了K8s原生方案的核心痛点：

```
┌─────────────────────────────────────────────────────────────┐
│                      KServe 架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 InferenceService CRD                  │   │
│  │                                                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │  Predictor │  │  Transformer│  │  Explainer │    │   │
│  │  │  (模型推理) │  │  (前后处理)  │  │  (可解释性) │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │  KServe     │  │  Knative    │  │  Istio           │   │
│  │  Controller │  │  Serving    │  │  (流量管理)       │   │
│  └─────────────┘  └─────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              底层 Kubernetes 集群                      │   │
│  │    GPU Nodes  │  CPU Nodes  │  存储（PVC/NFS/CSI）    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 核心配置示例

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sentiment-analysis
  annotations:
    serving.kserve.io/enable-prometheus-scraping: "true"
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      storageUri: "s3://models/sentiment-analysis/v2"
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
      env:
      - name: MAX_BATCH_SIZE
        value: "32"
      # 自定义推理服务配置
      runtimeVersion: "0.12.0"
      # 多版本流量分配（金丝雀发布）
      canaryTrafficPercent: 20
  transformer:
    containers:
    - image: registry.internal/sentiment-preprocessor:v1
      name: preprocessor
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
    - image: registry.internal/sentiment-postprocessor:v1
      name: postprocessor
```

### 金丝雀发布实战

KServe原生支持金丝雀发布，这在模型迭代中极为重要：

```yaml
# 第一步：部署新版本
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sentiment-analysis
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      storageUri: "s3://models/sentiment-analysis/v3"
      canaryTrafficPercent: 10  # 10%流量切到v3

---
# 第二步：监控指标正常后，逐步增加流量
# kubectl patch inferenceservice sentiment-analysis \
#   -p '{"spec":{"predictor":{"model":{"canaryTrafficPercent":50}}}}'

---
# 第三步：全量切换
# kubectl patch inferenceservice sentiment-analysis \
#   -p '{"spec":{"predictor":{"model":{"canaryTrafficPercent":100}}}}'
```

### 方案对比表

```
┌──────────────────┬────────────────────┬─────────────────────────┐
│      维度         │    K8s原生方案      │       KServe            │
├──────────────────┼────────────────────┼─────────────────────────┤
│  部署复杂度       │  中等（大量YAML）    │  低（一个CRD搞定）       │
│  GPU调度         │  手动配置           │  自动调度                │
│  模型版本管理     │  需自行实现         │  内置版本管理            │
│  金丝雀发布       │  需Istio手动配置    │  原生支持                │
│  自动扩缩容      │  HPA（冷启动）      │  Knative（缩至0）        │
│  多框架支持      │  无                │  支持10+框架              │
│  模型解释性       │  无                │  内置Explainer            │
│  适用模型数      │  < 20个            │  20-200个                 │
└──────────────────┴────────────────────┴─────────────────────────┘
```

## 阶段三：Serverless架构——极致弹性

### 为什么需要Serverless？

在许多业务场景中，模型推理的流量具有显著的**潮汐效应**：

```
流量模式示例：
         ▲ 流量
    5000 │              ╱╲
         │             ╱  ╲        ╱╲
    3000 │            ╱    ╲      ╱  ╲
         │           ╱      ╲    ╱    ╲
    1000 │──────────╱        ╲──╱      ╲──────
         │     工作日白天      周末     工作日
         └──────────────────────────────────→ 时间

峰谷比 = 5000:1000 = 5:1
```

对于峰谷比 > 5:1 的场景，常驻GPU实例的利用率极低，造成严重浪费。Serverless架构允许模型在空闲时**缩容到0**，请求到来时**实时冷启动**。

### Knative + KServe 的Serverless方案

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: document-ocr
  annotations:
    # 关键：启用缩容到0
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "20"
    # 冷启动优化配置
    serving.kserve.io/enable-modelflow: "true"
spec:
  predictor:
    model:
      modelFormat:
        name: onnx
      storageUri: "s3://models/document-ocr/v1"
      # 使用轻量化运行时加速冷启动
      runtimeVersion: "0.12.0"
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
```

### 冷启动优化策略

Serverless的最大挑战是冷启动延迟。以下是我们的优化方案：

```
┌─────────────────────────────────────────────────────────────┐
│                   冷启动优化策略矩阵                          │
├─────────────────┬───────────────┬──────────────────────────┤
│     策略         │    优化效果    │      实现方式             │
├─────────────────┼───────────────┼──────────────────────────┤
│  模型预热        │  -30~50s      │  DaemonSet定时预加载到GPU │
│  镜像预拉取      │  -10~20s      │  节点级镜像缓存           │
│  模型量化        │  -5~15s       │  INT8/FP16量化减小体积     │
│  ONNX Runtime   │  -3~8s        │  替代原生框架推理          │
│  保持最小实例    │  消除冷启动    │  minScale=1（牺牲成本）    │
│  预测性扩缩容    │  提前启动      │  基于历史流量预测          │
└─────────────────┴───────────────┴──────────────────────────┘
```

### 预测性扩缩容实现

```python
"""
基于历史流量模式的预测性扩缩容控制器
"""
import schedule
import numpy as np
from datetime import datetime, timedelta
from kubernetes import client, config

config.load_incluster_config()
apps_api = client.AppsV1Api()

class PredictiveScaler:
    def __init__(self, service_name: str, namespace: str):
        self.service_name = service_name
        self.namespace = namespace
        # 历史流量数据（按小时统计的QPS均值）
        self.hourly_pattern = self._load_historical_data()
    
    def _load_historical_data(self) -> np.ndarray:
        """加载过去30天的每小时平均QPS"""
        # 实际应从Prometheus/时序数据库查询
        # 这里用模拟数据
        return np.array([
            5, 3, 2, 1, 1, 5,     # 00:00-05:00 低谷
            20, 80, 150, 200, 180, 160,  # 06:00-11:00 上升
            140, 120, 130, 170, 190, 210, # 12:00-17:00 高峰
            180, 140, 100, 60, 30, 15,    # 18:00-23:00 下降
        ])
    
    def calculate_desired_replicas(self) -> int:
        """根据历史模式计算30分钟后需要的实例数"""
        now = datetime.now()
        future = now + timedelta(minutes=30)
        hour = future.hour
        
        predicted_qps = self.hourly_pattern[hour]
        # 假设每个实例处理50 QPS
        qps_per_instance = 50
        desired = max(1, int(np.ceil(predicted_qps / qps_per_instance)))
        
        return min(desired, 20)  # 上限20
    
    def scale(self):
        """执行扩缩容"""
        desired = self.calculate_desired_replicas()
        print(f"[{datetime.now()}] 预测30分钟后需要 {desired} 个实例")
        
        # 实际的HPA/Dynamic scaling逻辑
        # ...

scaler = PredictiveScaler("document-ocr", "ai-serving")
schedule.every(5).minutes.do(scaler.scale)
```

### 三种架构方案全景对比

```
┌────────────────┬──────────────┬──────────────┬──────────────────┐
│     维度        │  K8s原生     │   KServe     │  Serverless      │
├────────────────┼──────────────┼──────────────┼──────────────────┤
│  冷启动        │  无（常驻）   │  可选缩至0   │  核心特性         │
│  GPU利用率     │  30-50%      │  50-70%      │  80-95%          │
│  成本（闲置时） │  全额成本     │  可缩至0     │  接近0           │
│  延迟（热请求） │  ~5ms        │  ~5ms        │  ~5ms            │
│  延迟（冷启动） │  N/A         │  30-120s     │  30-120s         │
│  运维复杂度    │  低           │  中           │  高              │
│  最小规模      │  1台GPU服务器 │  1个节点池   │  无（缩至0）      │
│  适合场景      │  稳定流量     │  大部分场景   │  潮汐流量         │
└────────────────┴──────────────┴──────────────┴──────────────────┘
```

## 生产实战：我们的模型服务架构

### 整体架构

```
┌───────────────────────────────────────────────────────────────────┐
│                     生产环境模型服务架构                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  流量入口层                                │    │
│  │  Nginx Ingress → Istio Gateway → VirtualService          │    │
│  └──────────────────────┬───────────────────────────────────┘    │
│                          │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐    │
│  │              模型路由层 (Model Router)                     │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  基于模型类型、延迟要求、成本的智能路由             │    │    │
│  │  │  - GPT-4o → 贵但快，适合复杂推理                   │    │    │
│  │  │  - Qwen-72B → 便宜，适合简单任务                   │    │    │
│  │  │  - 自部署模型 → 无API费用，适合批处理              │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └──────────────────────┬───────────────────────────────────┘    │
│                          │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐    │
│  │              服务编排层                                    │    │
│  │                                                          │    │
│  │  常驻服务（KServe）          弹性服务（Knative）            │    │
│  │  ┌──────────────┐          ┌──────────────┐             │    │
│  │  │ Embedding    │          │ 文档OCR      │             │    │
│  │  │ Reranker     │          │ 图片理解      │             │    │
│  │  │ 向量检索     │          │ 语音识别      │             │    │
│  │  │ minScale=2   │          │ minScale=0   │             │    │
│  │  └──────────────┘          └──────────────┘             │    │
│  │                                                          │    │
│  │  批处理服务                                             │    │
│  │  ┌──────────────┐                                       │    │
│  │  │ 数据标注     │  Job型，完成后自动释放                   │    │
│  │  │ 模型评估     │  Job型，完成后自动释放                   │    │
│  │  └──────────────┘                                       │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              可观测性层                                    │    │
│  │  Prometheus + Grafana + 自定义Dashboard                    │    │
│  │  核心指标: P99延迟 | GPU利用率 | 吞吐量 | 错误率 | 成本     │    │
│  └──────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

### 智能模型路由实现

```python
"""
基于多维度评分的智能模型路由器
"""
from dataclasses import dataclass
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"      # 分类、情感分析
    MODERATE = "moderate"  # 摘要、翻译
    COMPLEX = "complex"    # 推理、代码生成

@dataclass
class ModelProfile:
    name: str
    cost_per_1k_tokens: float  # 成本（美元/千token）
    avg_latency_ms: float      # 平均延迟
    p99_latency_ms: float      # P99延迟
    max_throughput_qps: float  # 最大吞吐
    current_load: float        # 当前负载（0-1）
    capabilities: list         # 能力标签

class ModelRouter:
    def __init__(self):
        self.models = [
            ModelProfile(
                name="gpt-4o",
                cost_per_1k_tokens=0.005,
                avg_latency_ms=200,
                p99_latency_ms=800,
                max_throughput_qps=100,
                current_load=0.3,
                capabilities=["reasoning", "code", "multimodal"]
            ),
            ModelProfile(
                name="qwen-72b-local",
                cost_per_1k_tokens=0.0005,
                avg_latency_ms=150,
                p99_latency_ms=600,
                max_throughput_qps=50,
                current_load=0.6,
                capabilities=["reasoning", "code", "chinese"]
            ),
            ModelProfile(
                name="qwen-7b-local",
                cost_per_1k_tokens=0.0001,
                avg_latency_ms=80,
                p99_latency_ms=300,
                max_throughput_qps=200,
                current_load=0.2,
                capabilities=["classification", "simple-qa", "chinese"]
            ),
        ]
    
    def route(self, task_type: TaskComplexity, 
              latency_sla_ms: float = None) -> ModelProfile:
        """根据任务复杂度和SLA选择最优模型"""
        
        # 简单任务优先用便宜模型
        if task_type == TaskComplexity.SIMPLE:
            candidates = [m for m in self.models 
                         if "classification" in m.capabilities]
        elif task_type == TaskComplexity.COMPLEX:
            candidates = [m for m in self.models 
                         if "reasoning" in m.capabilities]
        else:
            candidates = self.models
        
        # 过滤延迟不达标的
        if latency_sla_ms:
            candidates = [m for m in candidates 
                         if m.p99_latency_ms <= latency_sla_ms]
        
        # 综合评分：成本30% + 延迟30% + 负载均衡40%
        best = min(candidates, key=lambda m: (
            0.3 * m.cost_per_1k_tokens / 0.005 +   # 归一化成本
            0.3 * m.avg_latency_ms / 200 +          # 归一化延迟
            0.4 * m.current_load                     # 负载均衡
        ))
        
        return best

# 使用示例
router = ModelRouter()
model = router.route(
    task_type=TaskComplexity.MODERATE,
    latency_sla_ms=500
)
print(f"路由到: {model.name}, 预估延迟: {model.avg_latency_ms}ms")
```

### 监控指标体系

```
┌─────────────────────────────────────────────────────────────────┐
│                  模型服务监控仪表板                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ 业务指标 ─────────────────────────────────────────────┐    │
│  │  QPS: 1,234/s  │  错误率: 0.02%  │  P99: 180ms        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─ GPU指标 ──────────────────────────────────────────────┐    │
│  │  利用率: 73%   │  显存: 68/80GB  │  温度: 65°C         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─ 成本指标 ─────────────────────────────────────────────┐    │
│  │  今日GPU成本: $12.50  │  月累计: $380  │  预算剩余: 62%  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─ 扩缩容事件 ───────────────────────────────────────────┐    │
│  │  10:00  扩容 sentiment +2 (QPS上升)                    │    │
│  │  10:15  扩容 document-ocr +5 (流量高峰)                 │    │
│  │  14:00  缩容 document-ocr -3 (流量回落)                 │    │
│  │  18:30  缩容 document-ocr → 0 (空闲)                   │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 踩坑经验与最佳实践

### 1. GPU资源碎片化问题

**问题**：当多个小模型各自占用1个GPU时，GPU显存利用率往往只有30-50%。

**解决方案**：使用NVIDIA MPS（Multi-Process Service）实现GPU共享：

```yaml
# 通过MPS将1个A100 (80GB) 分配给3个小模型
resources:
  limits:
    nvidia.com/gpu: "1"
env:
- name: NVIDIA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: "33"  # 每个进程分配33%的SM
```

### 2. 模型预热策略

**问题**：首次请求延迟极高（模型加载 + JIT编译）。

**解决方案**：实现主动预热机制：

```python
"""
模型预热服务 - 在Pod就绪后主动执行推理预热
"""
import requests
import time

def warmup_model(service_url: str, iterations: int = 5):
    """发送dummy请求触发模型预热"""
    dummy_payload = {
        "instances": [
            {"text": "This is a warmup request for model initialization."}
        ] * 8  # 批量预热
    }
    
    for i in range(iterations):
        start = time.time()
        resp = requests.post(
            f"{service_url}/v1/models/default:predict",
            json=dummy_payload
        )
        latency = time.time() - start
        print(f"Warmup iteration {i+1}: {latency:.2f}s")
    
    print("Model warmup complete")

# 在K8s PostStart Hook中调用
```

### 3. 模型存储优化

**问题**：大模型（70B+）存储体积超过200GB，拉取时间过长。

**解决方案**：

```
┌─────────────────────────────────────────────────────────────┐
│                  模型存储优化方案                              │
├─────────────────┬───────────────┬──────────────────────────┤
│     方案         │    效果        │      适用场景             │
├─────────────────┼───────────────┼──────────────────────────┤
│  模型量化        │  -60~75%      │  精度允许的推理场景        │
│  分层存储        │  -40~50%      │  热/温/冷模型分层          │
│  NFS/CSI共享    │  0（无拉取）   │  多节点共享同一模型        │
│  镜像分层缓存    │  -30~40%      │  模型层独立缓存           │
│  ONNX格式       │  -20~30%      │  跨框架兼容               │
└─────────────────┴───────────────┴──────────────────────────┘
```

## 选型决策树

```
                    ┌─────────────────┐
                    │  你的模型有多少？ │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
           ≤ 5个模型                  > 5个模型
                │                         │
        ┌───────┴───────┐         ┌───────┴───────┐
        │  流量波动大吗？ │         │  流量波动大吗？ │
        └───────┬───────┘         └───────┬───────┘
                │                         │
         ┌──────┴──────┐          ┌───────┴──────┐
         │             │          │              │
      波动小         波动大     波动小          波动大
         │             │          │              │
    ┌────┴────┐  ┌─────┴────┐ ┌──┴───┐    ┌─────┴─────┐
    │K8s原生   │  │KServe    │ │KServe│    │KServe +   │
    │Docker +  │  │+ Knative │ │      │    │Knative    │
    │Compose   │  │          │ │      │    │(Serverless)│
    └─────────┘  └──────────┘ └──────┘    └───────────┘
```

## 总结

AI模型服务编排架构的选择没有银弹，需要根据**模型数量、流量模式、团队能力和成本预算**综合决策。核心原则是：

1. **从简单开始**：不要一开始就上Serverless，先用K8s原生方案跑通流程
2. **逐步演进**：当模型数量超过20个时，引入KServe统一管理
3. **按需弹性**：对流量波动大的服务，启用Knative缩容到0能力
4. **可观测性先行**：在架构演进之前，先建好监控体系
5. **成本驱动**：GPU是昂贵资源，架构选择要以成本效益为导向

模型服务架构不是一成不变的，它应该随着业务规模和技术栈的成熟而持续演进。
