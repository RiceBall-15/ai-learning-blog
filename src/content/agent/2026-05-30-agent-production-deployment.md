---
title: "Agent生产化部署：从原型到大规模服务的工程实践"
description: "深入解析AI Agent在生产环境中的容器编排、自动伸缩、成本优化、A/B测试、蓝绿发布、回滚策略、SLA定义及多区域部署等核心工程实践，面向Agent开发者面试及生产化落地。"
date: 2026-05-30
author: "技术学习笔记"
category: "agent"
subCategory: "运维"
tags: ["Agent", "Deployment", "Production", "面试"]
---

# Agent生产化部署：从原型到大规模服务的工程实践

## 引言

当一个AI Agent从Jupyter Notebook走向生产环境时，开发者面对的不再仅仅是prompt工程和模型调用的精度问题——而是如何在高并发、低延迟、高可用的约束下，将Agent系统稳定地服务化。本文将从容器编排、自动伸缩、成本优化、灰度发布、SLA定义等多个维度，系统地梳理Agent生产化部署的工程实践。

---

## 1. Agent服务的容器化与Kubernetes编排

### 1.1 为什么Agent服务需要容器化

Agent系统的典型特征是**有状态、资源异构、依赖复杂**：它可能需要加载向量数据库连接、维护对话历史、持有Tool调用的外部API密钥。容器化提供了环境隔离和可复现性，而Kubernetes提供了编排和自愈能力。

### 1.2 Agent服务的K8s部署模式

生产环境中，Agent服务通常采用以下几种K8s部署模式：

**模式一：单体Agent Pod**

将Agent的所有组件（LLM推理、Tool执行、状态管理）打包在一个Pod中：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-core
  template:
    metadata:
      labels:
        app: agent-core
    spec:
      containers:
      - name: agent
        image: registry.example.com/agent-core:v2.3.1
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
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
```

适用场景：中小规模Agent、快速迭代期、团队K8s经验有限。

**模式二：微服务化Agent**

将Agent拆分为多个独立服务：

- **Agent Orchestrator**：负责对话管理、任务调度
- **LLM Gateway**：统一管理模型调用、路由、限流
- **Tool Executor**：隔离外部API调用（数据库查询、代码执行等）
- **Memory Service**：管理对话历史和长期记忆

```yaml
# LLM Gateway 服务配置
apiVersion: v1
kind: Service
metadata:
  name: llm-gateway
spec:
  selector:
    app: llm-gateway
  ports:
  - port: 8081
    targetPort: 8081
  type: ClusterIP
---
# Tool Executor 服务配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tool-executor
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: tool-exec
        image: registry.example.com/tool-executor:v1.8.0
        env:
        - name: SANDBOX_MODE
          value: "strict"
        resources:
          requests:
            cpu: "1"
            memory: "4Gi"
```

适用场景：大规模Agent、需要独立伸缩、团队有微服务经验。

**模式三：Sidecar模式**

在Agent Pod中注入sidecar容器处理横切关注点：

```yaml
spec:
  containers:
  - name: agent
    image: registry.example.com/agent-core:v2.3.1
  - name: envoy-proxy
    image: envoyproxy/envoy:v1.28.0
    # 负责：mTLS、流量管理、可观测性
  - name: log-collector
    image: registry.example.com/log-collector:v1.2.0
    # 负责：Agent轨迹日志收集和结构化
```

### 1.3 GPU调度策略

如果Agent中涉及本地模型推理（如embedding模型、小模型路由），需要配置GPU调度：

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
nodeSelector:
  accelerator: nvidia-t4
tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

生产建议：将LLM调用（API方式）和本地模型推理（GPU）分开部署，避免GPU Pod的弹性伸缩影响Agent核心逻辑的可用性。

---

## 2. 自动伸缩策略

Agent工作负载的核心挑战在于**流量的不可预测性**——一个爆款Agent可能在数小时内迎来百倍流量，而一个企业内部Agent在工作日和周末的负载差异可达10倍以上。

### 2.1 HPA（Horizontal Pod Autoscaler）

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-core
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: active_requests
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 120
```

关键点：**scaleDown的stabilizationWindow要远大于scaleUp**。Agent服务重启时需要重新加载prompt模板、建立Tool连接，过快的缩容会导致可用性波动。

### 2.2 KEDA（Kubernetes Event-Driven Autoscaling）

Agent服务天然适合事件驱动伸缩——每个用户请求就是一个事件。KEDA提供了更精细的伸缩能力：

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: agent-scaledobject
spec:
  scaleTargetRef:
    name: agent-core
  pollingInterval: 15
  cooldownPeriod: 300
  minReplicaCount: 3
  maxReplicaCount: 100
  triggers:
  - type: kafka
    metadata:
      bootstrapServers: kafka-cluster:9092
      consumerGroup: agent-consumer
      topic: agent-requests
      lagThreshold: "50"
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: agent_queue_depth
      query: sum(agent_requests_waiting)
      threshold: "20"
```

### 2.3 预测性伸缩（Predictive Autoscaling）

对于有明显时间规律的Agent负载（如企业客服Agent在工作日9-18点的高峰），可以使用Kubernetes VPA或自定义的预测性伸缩器：

```python
# 伪代码：基于历史数据的预测性伸缩
def predict_scale(current_time, history_metrics):
    hour = current_time.hour
    day_of_week = current_time.weekday()
    
    # 基于历史同时段的平均负载决定期望副本数
    expected_load = history_metrics.get_average(
        hour=hour, day_of_week=day_of_week
    )
    
    # 提前15分钟扩容，避免冷启动延迟
    target_replicas = calculate_replicas(expected_load)
    return target_replicas
```

### 2.4 混合伸缩策略

生产中推荐组合使用多种伸缩策略：

- **基础层**：HPA基于CPU/内存维持最低保障
- **事件层**：KEDA基于消息队列深度处理突发流量
- **预测层**：CronJob定时任务在已知高峰前提前扩容
- **兜底层**：VPA监控资源使用趋势，避免OOM

---

## 3. 成本优化

Agent服务的成本大头通常来自LLM API调用。以GPT-4级别模型为例，每次Agent对话可能触发3-10次LLM调用（推理、规划、工具选择、最终生成），成本优化是生产化的关键课题。

### 3.1 模型路由（Model Routing）

不是所有请求都需要最强的模型。建立分层模型路由：

```python
class ModelRouter:
    """根据请求复杂度动态路由到不同模型"""
    
    ROUTING_TABLE = {
        "simple_qa": {"model": "gpt-4o-mini", "max_tokens": 512},
        "complex_reasoning": {"model": "gpt-4o", "max_tokens": 2048},
        "code_generation": {"model": "claude-sonnet-4-20250514", "max_tokens": 4096},
        "multi_step_planning": {"model": "o3", "max_tokens": 8192},
    }
    
    def route(self, request: AgentRequest) -> dict:
        # 方式1：基于规则的路由
        if request.tool_calls and len(request.tool_calls) > 3:
            return self.ROUTING_TABLE["multi_step_planning"]
        
        # 方式2：基于分类器的路由（用小模型判断复杂度）
        complexity = self.classify_complexity(request.query)
        if complexity < 0.3:
            return self.ROUTING_TABLE["simple_qa"]
        elif complexity < 0.7:
            return self.ROUTING_TABLE["complex_reasoning"]
        else:
            return self.ROUTING_TABLE["multi_step_planning"]
```

**成本对比**：在生产环境中，模型路由可以将平均API成本降低40-60%，同时保持95%以上场景的输出质量。

### 3.2 语义缓存（Semantic Caching）

Agent系统中大量请求是语义相似的。建立语义缓存层：

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, similarity_threshold=0.92, ttl=3600):
        self.encoder = SentenceTransformer('bge-large-zh-v1.5')
        self.cache_store = {}  # 实际生产用Redis + vector index
        self.threshold = similarity_threshold
        self.ttl = ttl
    
    async def get(self, query: str, context: dict) -> Optional[str]:
        query_embedding = self.encoder.encode(query)
        
        # 在缓存中查找语义相似的已缓存响应
        for cached_key, cached_value in self.cache_store.items():
            similarity = np.dot(query_embedding, cached_value['embedding'])
            if similarity > self.threshold:
                # 检查上下文是否也匹配（用户、权限等）
                if self._context_match(context, cached_value['context']):
                    return cached_value['response']
        return None
    
    async def set(self, query: str, response: str, context: dict):
        embedding = self.encoder.encode(query)
        self.cache_store[query] = {
            'embedding': embedding,
            'response': response,
            'context': context,
            'timestamp': time.time()
        }
```

注意事项：缓存命中率通常在15-30%，但在FAQ类Agent中可达50%以上。必须对缓存的响应进行时效性检查（如股票价格、天气等实时数据不缓存）。

### 3.3 批处理与请求合并（Batching）

对于非实时性场景（如批量内容生成、数据分析报告），使用请求批处理：

```python
class RequestBatcher:
    """将多个相似请求合并为一次batch调用"""
    
    def __init__(self, batch_size=32, max_wait_ms=100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_queue = asyncio.Queue()
    
    async def process_batch(self, batch: List[Request]) -> List[Response]:
        # 利用LLM的batch API或vLLM的continuous batching
        prompts = [r.to_prompt() for r in batch]
        
        # vLLM continuous batching - 吞吐量提升3-5倍
        responses = await self.llm_client.batch_complete(
            prompts, 
            max_tokens=512,
            use_vllm=True
        )
        return responses
```

### 3.4 成本监控与告警

```python
class CostMonitor:
    def __init__(self, daily_budget_usd=1000):
        self.daily_budget = daily_budget_usd
        self.current_cost = 0
        self.cost_by_model = defaultdict(float)
        self.cost_by_user = defaultdict(float)
    
    def record_call(self, model: str, tokens: int, user_id: str):
        cost = self._calculate_cost(model, tokens)
        self.current_cost += cost
        self.cost_by_model[model] += cost
        self.cost_by_user[user_id] += cost
        
        if self.current_cost > self.daily_budget * 0.8:
            self._trigger_alert("Daily budget 80% consumed")
        if self.current_cost > self.daily_budget:
            self._trigger_emergency("Daily budget exceeded")
```

---

## 4. A/B测试与灰度发布

### 4.1 Agent行为的A/B测试

Agent的A/B测试与传统Web应用不同——你需要测试的不只是UI变体，而是**行为策略的差异**：

```yaml
# 使用Istio VirtualService实现Agent策略A/B测试
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: agent-ab-test
spec:
  hosts:
  - agent-service
  http:
  - match:
    - headers:
        x-user-segment:
          exact: "beta-testers"
    route:
    - destination:
        host: agent-service
        subset: canary-v2
      weight: 100
  - route:
    - destination:
        host: agent-service
        subset: stable-v1
      weight: 90
    - destination:
        host: agent-service
        subset: canary-v2
      weight: 10
```

### 4.2 关键指标定义

Agent A/B测试需要关注的独特指标：

- **任务完成率**：Agent是否成功完成了用户请求的任务
- **平均交互轮次**：完成同一任务需要多少轮对话（越少越好）
- **Tool调用成功率**：Agent调用外部工具的成功率
- **幻觉率**：Agent生成了不基于上下文的错误信息的比例
- **用户干预率**：用户需要手动纠正Agent行为的频率
- **端到端延迟P99**：从用户发起到Agent完成的总延迟

```python
class AgentABTestEvaluator:
    def evaluate(self, control_group, treatment_group):
        return {
            "task_completion": {
                "control": control_group.task_completion_rate,
                "treatment": treatment_group.task_completion_rate,
                "p_value": self._calculate_p_value(
                    control_group.task_completion_rate,
                    treatment_group.task_completion_rate
                )
            },
            "avg_turns": {
                "control": control_group.avg_turns,
                "treatment": treatment_group.avg_turns,
            },
            "cost_per_task": {
                "control": control_group.avg_cost_per_task,
                "treatment": treatment_group.avg_cost_per_task,
            },
            "hallucination_rate": {
                "control": control_group.hallucination_rate,
                "treatment": treatment_group.hallucination_rate,
            }
        }
```

---

## 5. 蓝绿部署与回滚策略

### 5.1 蓝绿部署

Agent服务的更新不同于无状态Web服务——Agent的prompt模板、Tool配置、记忆策略的变更可能带来行为的非线性变化。蓝绿部署提供了安全的切换能力：

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: agent-rollout
spec:
  replicas: 10
  strategy:
    blueGreen:
      activeService: agent-active
      previewService: agent-preview
      prePromotionAnalysis:
        templates:
        - templateName: agent-quality-check
        args:
        - name: rollout-type
          value: pre-promotion
      postPromotionAnalysis:
        templates:
        - templateName: agent-quality-check
        args:
        - name: rollout-type
          value: post-promotion
      autoPromotionEnabled: false  # 手动确认后切换
      scaleDownDelaySeconds: 600
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: agent-quality-check
spec:
  args:
  - name: rollout-type
  metrics:
  - name: task-completion-rate
    interval: 5m
    count: 6
    successCondition: result[0] >= 0.95
    failureLimit: 2
    provider:
      prometheus:
        address: http://prometheus:9090
        query: |
          sum(rate(agent_task_success_total{rollout="{{rollout-type}}"}[5m])) /
          sum(rate(agent_task_total{rollout="{{rollout-type}}"}[5m]))
  - name: p99-latency
    interval: 5m
    count: 6
    successCondition: result[0] <= 5000
    failureLimit: 2
    provider:
      prometheus:
        address: http://prometheus:9090
        query: |
          histogram_quantile(0.99, 
            sum(rate(agent_request_duration_seconds_bucket{rollout="{{rollout-type}}"}[5m])) by (le)
          ) * 1000
```

### 5.2 Agent特有的回滚策略

Agent回滚不仅仅是回退代码版本，还需要考虑：

**状态回滚**：用户对话历史可能依赖新版Agent的格式，回滚时需要状态兼容层。

```python
class AgentRollbackManager:
    def rollback(self, target_version: str):
        # 1. 回滚Agent服务代码和配置
        self.deploy_service(target_version)
        
        # 2. 检查prompt模板的向后兼容性
        prompt_diff = self.diff_prompts(current_version, target_version)
        if prompt_diff.has_breaking_changes:
            self._migrate_conversation_history(prompt_diff)
        
        # 3. 切换Tool配置
        self.rollback_tool_config(target_version)
        
        # 4. 通知所有活跃会话
        self.notify_active_sessions(
            "系统已回退至稳定版本，部分功能可能受限"
        )
```

**快速回滚机制**：

```yaml
# 回滚命令模板
apiVersion: batch/v1
kind: Job
metadata:
  name: agent-rollback
spec:
  template:
    spec:
      containers:
      - name: rollback
        image: registry.example.com/rollback-tool:v1.0
        command: ["/bin/sh", "-c"]
        args:
        - |
          # 检测当前版本健康状态
          HEALTH=$(curl -s http://agent-active/health)
          if echo $HEALTH | jq -e '.status == "unhealthy"' > /dev/null; then
            # 执行回滚
            kubectl rollout undo deployment/agent-core --to-revision=$PREVIOUS_REVISION
            # 等待回滚完成
            kubectl rollout status deployment/agent-core --timeout=300s
            # 发送告警
            curl -X POST $SLACK_WEBHOOK -d '{"text":"Agent已自动回滚至版本'$PREVIOUS_REVISION'"}'
          fi
```

### 5.3 渐进式发布

推荐使用Argo Rollouts的渐进式发布，逐步增加新版本流量比例：

```
v1: 100% → v2: 10% → 观察30分钟 → v2: 30% → 观察30分钟 → v2: 60% → v2: 100%
```

每个阶段都通过AnalysisTemplate自动验证关键指标，任何阶段失败都自动回滚。

---

## 6. SLA定义与可用性保障

### 6.1 Agent系统的SLA框架

Agent系统比传统Web服务的SLA定义更复杂，需要涵盖多个维度：

```yaml
# Agent SLA定义
service_level_objectives:
  availability:
    target: 99.9%
    measurement: "成功返回响应的请求数 / 总请求数"
    exclusion: "计划维护窗口、上游LLM服务商故障"
  
  latency:
    target:
      p50: 1500ms
      p95: 4000ms
      p99: 8000ms
    measurement: "从接收请求到返回首个token的时间（TTFB）"
  
  task_completion:
    target: 92%
    measurement: "Agent成功完成用户任务的比例"
    validation: "通过标注数据集定期评估"
  
  hallucination_rate:
    upper_bound: 3%
    measurement: "Agent生成事实性错误的比例"
  
  tool_reliability:
    target: 99.5%
    measurement: "Tool调用成功次数 / 总调用次数"
```

### 6.2 高可用架构

```yaml
# 多层健康检查配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-core
spec:
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0  # Agent服务不允许任何不可用Pod
  template:
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: agent-core
      containers:
      - name: agent
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8080
          failureThreshold: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          periodSeconds: 5
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          periodSeconds: 10
          failureThreshold: 3
```

### 6.3 降级策略

当Agent系统出现异常时，需要优雅降级而非完全不可用：

```python
class AgentDegradationManager:
    GRACEFUL_DEGRADATION_LEVELS = {
        0: "full",          # 全功能模式
        1: "tool_disabled", # 禁用外部Tool调用，仅用模型能力回答
        2: "cached_only",   # 仅返回缓存的响应
        3: "fallback",      # 返回预设的兜底回复
        4: "maintenance",   # 维护模式，返回维护通知
    }
    
    async def handle_request(self, request, system_health):
        degradation_level = self._assess_degradation_level(system_health)
        
        if degradation_level == 0:
            return await self.full_agent_process(request)
        elif degradation_level == 1:
            # 工具不可用时，仅用模型能力
            return await self.model_only_process(request)
        elif degradation_level == 2:
            cached = await self.semantic_cache.get(request.query)
            if cached:
                return cached
            return await self.fallback_response(request)
        elif degradation_level >= 3:
            return self.maintenance_message
```

---

## 7. 多区域部署

### 7.1 多区域部署策略

Agent服务的多区域部署面临独特挑战——LLM API通常由单一提供商提供，而用户数据可能有地域合规要求。

```
                        ┌──────────────┐
                        │ Global LB    │
                        │ (Cloudflare) │
                        └──────┬───────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐
         │ CN Region   │ │ US Region  │ │ EU Region  │
         │             │ │            │ │            │
         │ Agent Core  │ │ Agent Core │ │ Agent Core │
         │ (GPT-4o     │ │ (GPT-4o    │ │ (Claude    │
         │  via Azure) │ │  via API)  │ │  via API)  │
         │             │ │            │ │            │
         │ Vector DB   │ │ Vector DB  │ │ Vector DB  │
         │ (本地化数据) │ │ (共享索引) │ │ (GDPR合规) │
         └─────────────┘ └────────────┘ └────────────┘
```

### 7.2 数据一致性

```python
class MultiRegionAgentCoordinator:
    """处理跨区域的状态同步"""
    
    def __init__(self):
        self.local_store = LocalConversationStore()
        self.global_store = GlobalConversationStore()  # 最终一致性
        self.conflict_resolver = ConversationConflictResolver()
    
    async def save_conversation(self, user_id: str, conversation: dict):
        # 本地写入（强一致性，低延迟）
        await self.local_store.save(user_id, conversation)
        
        # 异步同步到全局存储（最终一致性）
        asyncio.create_task(
            self.global_store.sync(user_id, conversation)
        )
    
    async def load_conversation(self, user_id: str, region: str):
        # 优先从本地读取
        local = await self.local_store.get(user_id)
        if local:
            return local
        
        # 本地不存在，从全局读取
        global_data = await self.global_store.get(user_id)
        if global_data:
            # 缓存到本地
            await self.local_store.save(user_id, global_data)
        return global_data
```

### 7.3 延迟优化

- **模型调用就近路由**：选择与Agent服务同区域的LLM API endpoint
- **嵌入向量本地化**：在每个区域部署embedding模型，避免跨区域的向量检索延迟
- **对话历史CDN缓存**：热门用户的对话历史在边缘节点缓存

---

## 8. 生产架构案例

### 8.1 案例一：企业客服Agent（高并发低延迟）

```
用户 → Cloudflare CDN → API Gateway (限流、认证)
    → Agent Orchestrator (Pod: 10-100, HPA)
    → 并行调用:
       ├─ LLM Gateway (路由到最优模型)
       ├─ Knowledge Base (RAG, 向量检索)
       ├─ Tool Pool (订单查询、工单系统)
       └─ Memory Store (Redis, 对话历史)
```

关键指标：P99 < 3s, 可用性 99.95%, 日均处理 50万+对话

### 8.2 案例二：多模态创作Agent（资源密集型）

```
用户 → WebSocket Gateway (流式输出)
    → Task Queue (Kafka, 异步任务管理)
    → Agent Worker Pool (GPU节点, 3-20 Pod)
       ├─ 推理引擎 (本地模型, GPU调度)
       ├─ 生成服务 (图像/视频/音频)
       └─ 存储服务 (S3, 生成产物)
```

关键指标：任务完成率 > 95%, 平均任务耗时 < 5min, GPU利用率 > 70%

### 8.3 成本数据参考

以中等规模Agent服务（日均10万次对话）为例：

| 成本项 | 月费用（美元） | 占比 |
|--------|--------------|------|
| LLM API调用 | $8,000-15,000 | 50-60% |
| K8s集群（计算+存储） | $3,000-5,000 | 20-25% |
| 向量数据库 | $1,000-2,000 | 8-10% |
| 网络与CDN | $500-1,000 | 3-5% |
| 监控与日志 | $500-800 | 3-4% |
| **总计** | **$13,000-23,800** | 100% |

通过模型路由和语义缓存优化后，LLM API成本可降低40%，整体月费用降至 **$9,000-16,000**。

---

## 9. 面试要点总结

面试中关于Agent生产化部署的高频问题：

1. **如何设计Agent的自动伸缩策略？** 考察HPA/KEDA/预测性伸缩的组合使用
2. **如何降低Agent的LLM调用成本？** 考察模型路由、语义缓存、批处理
3. **Agent服务如何实现蓝绿部署？** 考察状态管理、prompt兼容性、渐进式发布
4. **如何定义Agent系统的SLA？** 考察多维度SLA（可用性、延迟、质量）
5. **多区域部署如何保证数据一致性？** 考察最终一致性模型和本地优先策略
6. **Agent服务出现异常如何降级？** 考察分级降级策略的优雅实现

---

## 总结

Agent的生产化部署是一个系统工程，涉及基础设施（K8s编排、GPU调度）、业务策略（模型路由、成本优化）、工程实践（A/B测试、蓝绿部署）和服务治理（SLA、降级策略）多个层面。核心原则是：**渐进式交付、可观测优先、成本与质量平衡、优雅降级**。

从原型到生产化，最关键的心态转变是：Agent不再是一个"模型调用"，而是一个需要完整SRE实践支撑的**服务**。
