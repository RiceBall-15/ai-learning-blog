---
title: "多 Agent 协作中的 Skill 编排：从单兵作战到军团协同"
description: "探讨多个 Agent 如何通过 Skill 编排实现复杂任务分解、并行执行、结果聚合，以及容错和负载均衡策略"
date: 2026-05-09
author: RiceBall-15
category: agentSkill
subCategory: agent-architecture
tags: ["Agent Skill", "多Agent协作", "任务编排", "分布式", "并行计算"]
series: agent-skill-dev
seriesOrder: 9
---


## 简介

当单个 Agent 无法独立完成复杂任务时，我们需要多个 Agent 协作。如何让它们高效配合？如何分解任务？如何处理失败？本文探讨多 Agent 环境下的 Skill 编排策略，从架构设计到容错机制，帮你构建可靠的多 Agent 系统。

## 问题背景

现实中的复杂任务往往需要多方协作：

1. **任务太复杂**：单个 Agent 的上下文窗口不够
2. **需要专业分工**：代码审查、测试、部署各需不同专长
3. **并行加速**：多个子任务可以同时执行
4. **容错需求**：某个 Agent 挂了，其他能接手

参考 Kubernetes 的 Pod 编排【1】和 Apache Airflow 的 DAG 调度【2】，我们可以借鉴成熟的编排思想。

## 编排模式

### 模式一：顺序流水线（Pipeline）

```
任务输入 → [Agent A] → [Agent B] → [Agent C] → 结果输出
              ↓           ↓           ↓
           Skill 1     Skill 2     Skill 3
```

适用场景：依赖关系明确，必须按顺序执行

```yaml
# pipeline.yaml
name: code-review-pipeline
stages:
  - name: analyze
    agent: code-analyzer
    skill: static-analysis
    input: ${codebase}
    
  - name: review
    agent: senior-reviewer
    skill: code-review
    input: ${analyze.result}
    depends_on: [analyze]
    
  - name: approve
    agent: lead-developer
    skill: approval
    input: ${review.report}
    depends_on: [review]
```

### 模式二：并行扇出-扇入（Fan-out/Fan-in）

```
                 ┌→ [Agent B1] ─┐
任务输入 → [Agent A] → [Agent B2] → [Agent C] → 结果输出
                 └→ [Agent B3] ─┘
```

适用场景：子任务独立，可并行加速

```yaml
# fan-out-fan-in.yaml
name: parallel-analysis
fan_out:
  source: file-list
  agents:
    - id: analyzer-1
      skill: analyze-python
      filter: "*.py"
    - id: analyzer-2
      skill: analyze-javascript
      filter: "*.js"
    - id: analyzer-3
      skill: analyze-config
      filter: "*.yaml"

fan_in:
  agent: aggregator
  skill: merge-reports
  inputs: [analyzer-1, analyzer-2, analyzer-3]
```

### 模式三：Map-Reduce

```
大数据集 → Map 阶段（并行） → Shuffle → Reduce 阶段 → 结果
           [A1][A2][A3]...      ↓        [Reducer]
```

适用场景：大数据处理，需要分片并行

```python
# map_reduce.py
class MapReduceOrchestrator:
    def __init__(self, mapper_skill, reducer_skill):
        self.mapper = mapper_skill
        self.reducer = reducer_skill
    
    async def execute(self, large_dataset):
        # Map 阶段：分片并行
        chunks = self._split(large_dataset, chunk_size=100)
        mapper_tasks = [
            self.mapper.execute(chunk) 
            for chunk in chunks
        ]
        map_results = await asyncio.gather(*mapper_tasks)
        
        # Shuffle 阶段：按 key 分组
        shuffled = self._shuffle(map_results)
        
        # Reduce 阶段：聚合
        final_result = await self.reducer.execute(shuffled)
        return final_result
```

### 模式四：事件驱动（Event-driven）

```
事件源 → [Event Bus] → [Agent A] 订阅事件 1
                    → [Agent B] 订阅事件 2
                    → [Agent C] 订阅事件 1, 3
```

适用场景：松耦合，动态响应

```yaml
# event-driven.yaml
name: reactive-system
event_bus:
  type: redis
  url: redis://localhost:6379

agents:
  - id: monitor
    subscribes: [system.alert]
    skill: alert-handler
    
  - id: logger
    subscribes: [system.*]
    skill: log-event
    
  - id: responder
    subscribes: [incident.created]
    skill: auto-respond
```

### 模式五：层级委派（Hierarchical Delegation）

```
         [Orchestrator Agent]
              /    |    \
             /     |     \
      [Worker A] [Worker B] [Worker C]
          |         |         |
      [Sub A1]  [Sub B1]  [Sub C1]
```

适用场景：复杂任务需要多级分解

```python
# hierarchical.py
class OrchestratorAgent:
    async def execute_complex_task(self, task):
        # 分解任务
        subtasks = self.decompose(task)
        
        # 委派给 Worker
        results = []
        for subtask in subtasks:
            worker = self.select_worker(subtask)
            result = await worker.execute(subtask)
            results.append(result)
        
        # 聚合结果
        return self.aggregate(results)
    
    def select_worker(self, subtask):
        """根据子任务类型选择 Worker"""
        if subtask.type == 'code':
            return self.code_worker
        elif subtask.type == 'data':
            return self.data_worker
        else:
            return self.general_worker
```

## 协调机制

### 共享状态管理

参考 Redis 的分布式锁【3】：

```python
# shared_state.py
import redis
import json
from contextlib import contextmanager

class SharedStateManager:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)
    
    @contextmanager
    def distributed_lock(self, resource, timeout=30):
        """分布式锁"""
        lock_key = f"lock:{resource}"
        lock = self.redis.lock(lock_key, timeout=timeout)
        
        try:
            if lock.acquire(blocking=True, blocking_timeout=10):
                yield lock
            else:
                raise TimeoutError(f"获取锁超时: {resource}")
        finally:
            lock.release()
    
    def set_task_status(self, task_id, status, result=None):
        """更新任务状态"""
        data = {
            'status': status,
            'result': result,
            'timestamp': time.time()
        }
        self.redis.set(f"task:{task_id}", json.dumps(data))
    
    def get_task_status(self, task_id):
        """获取任务状态"""
        data = self.redis.get(f"task:{task_id}")
        return json.loads(data) if data else None
```

### 消息队列

参考 RabbitMQ 的消息模式【4】：

```python
# message_queue.py
import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List
from enum import Enum

class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Message:
    id: str
    type: str
    payload: dict
    priority: MessagePriority
    sender: str
    timestamp: float

class AgentMessageQueue:
    def __init__(self):
        self.queues: Dict[str, asyncio.PriorityQueue] = {}
        self.handlers: Dict[str, List[Callable]] = {}
    
    async def publish(self, agent_id: str, message: Message):
        """发布消息到 Agent 队列"""
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.PriorityQueue()
        
        # 优先级排序（数字小的优先级高）
        priority = -message.priority.value
        await self.queues[agent_id].put((priority, message))
    
    async def subscribe(self, agent_id: str, handler: Callable):
        """订阅消息"""
        if agent_id not in self.handlers:
            self.handlers[agent_id] = []
        self.handlers[agent_id].append(handler)
    
    async def start_consuming(self, agent_id: str):
        """开始消费消息"""
        while True:
            if agent_id in self.queues:
                _, message = await self.queues[agent_id].get()
                
                # 调用所有处理器
                for handler in self.handlers.get(agent_id, []):
                    await handler(message)
```

## 容错机制

### 重试策略

参考 Google SRE 的重试最佳实践【5】：

```python
# retry.py
import asyncio
import random
from functools import wraps
from typing import Optional, Type

class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟（指数退避 + 抖动）"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay

def retry_with_policy(
    policy: RetryPolicy,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(policy.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < policy.max_retries:
                        delay = policy.calculate_delay(attempt)
                        
                        if on_retry:
                            await on_retry(attempt, e, delay)
                        
                        await asyncio.sleep(delay)
                    else:
                        raise
            
            raise last_exception
        return wrapper
    return decorator

# 使用示例
@retry_with_policy(
    policy=RetryPolicy(max_retries=3, base_delay=1.0),
    exceptions=(ConnectionError, TimeoutError)
)
async def call_external_api():
    """可能失败的外部调用"""
    pass
```

### 熔断器

参考 Netflix Hystrix【6】：

```python
# circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
    
    async def call(self, func, *args, **kwargs):
        """通过熔断器调用"""
        if self.state == CircuitState.OPEN:
            if self._should_try_recovery():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitOpenError("熔断器开启，拒绝调用")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """成功回调"""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_try_recovery(self) -> bool:
        """是否应该尝试恢复"""
        if self.last_failure_time is None:
            return True
        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.recovery_timeout)
```

### 降级策略

```python
# fallback.py
from typing import Any, Callable, Optional

class FallbackChain:
    """降级链：按优先级尝试多个实现"""
    
    def __init__(self):
        self.fallbacks: List[Callable] = []
    
    def add(self, priority: int, handler: Callable):
        """添加降级处理器"""
        self.fallbacks.append((priority, handler))
        self.fallbacks.sort(key=lambda x: x[0])
    
    async def execute(self, *args, **kwargs) -> Any:
        """执行，失败时降级"""
        last_error = None
        
        for priority, handler in self.fallbacks:
            try:
                return await handler(*args, **kwargs)
            except Exception as e:
                last_error = e
                print(f"优先级 {priority} 失败，尝试下一个: {e}")
        
        raise AllFallbacksFailedError(
            f"所有降级方案失败: {last_error}"
        )

# 使用示例
chain = FallbackChain()
chain.add(1, use_primary_model)      # 优先使用主模型
chain.add(2, use_backup_model)       # 备用模型
chain.add(3, use_cached_response)    # 缓存响应
chain.add(4, return_default)         # 默认值
```

## 负载均衡

### 策略类型

```python
# load_balancer.py
from typing import List
import random
import time
from dataclasses import dataclass

@dataclass
class AgentInstance:
    id: str
    capacity: int
    current_load: int
    last_health_check: float

class LoadBalancer:
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances: List[AgentInstance] = []
        self._round_robin_index = 0
    
    def select_instance(self) -> AgentInstance:
        """选择实例"""
        available = [
            i for i in self.instances 
            if i.current_load < i.capacity
        ]
        
        if not available:
            raise NoAvailableInstanceError()
        
        if self.strategy == "round_robin":
            return self._round_robin(available)
        elif self.strategy == "least_connections":
            return self._least_connections(available)
        elif self.strategy == "random":
            return self._random(available)
        elif self.strategy == "weighted":
            return self._weighted(available)
        else:
            raise ValueError(f"未知策略: {self.strategy}")
    
    def _round_robin(self, instances: List[AgentInstance]):
        """轮询"""
        instance = instances[self._round_robin_index % len(instances)]
        self._round_robin_index += 1
        return instance
    
    def _least_connections(self, instances: List[AgentInstance]):
        """最少连接"""
        return min(instances, key=lambda i: i.current_load)
    
    def _random(self, instances: List[AgentInstance]):
        """随机"""
        return random.choice(instances)
    
    def _weighted(self, instances: List[AgentInstance]):
        """加权"""
        weights = [i.capacity - i.current_load for i in instances]
        return random.choices(instances, weights=weights)[0]
```

## 实战案例：构建代码审查系统

### 架构设计

```
Git Push Event
      ↓
[Orchestrator]
      ↓
┌─────────┼─────────┐
↓         ↓         ↓
[Static   [Security [Performance
 Analyzer] Scanner]  Tester]
      ↓         ↓         ↓
└─────────┼─────────┘
      ↓
[Report Aggregator]
      ↓
[Review Dashboard]
```

### 实现代码

```python
# code_review_system.py
import asyncio
from typing import List, Dict

class CodeReviewOrchestrator:
    def __init__(self):
        self.static_analyzer = StaticAnalysisAgent()
        self.security_scanner = SecurityScannerAgent()
        self.perf_tester = PerformanceTesterAgent()
        self.aggregator = ReportAggregatorAgent()
    
    async def review_pull_request(self, pr_url: str):
        """审查 Pull Request"""
        
        # 1. 获取代码变更
        changes = await self._fetch_changes(pr_url)
        
        # 2. 并行分析
        analysis_tasks = [
            self.static_analyzer.analyze(changes),
            self.security_scanner.scan(changes),
            self.perf_tester.test(changes)
        ]
        
        results = await asyncio.gather(
            *analysis_tasks,
            return_exceptions=True
        )
        
        # 3. 聚合结果
        report = await self.aggregator.aggregate(results)
        
        # 4. 发布结果
        await self._publish_report(pr_url, report)
        
        return report
    
    async def _fetch_changes(self, pr_url: str):
        """获取 PR 变更"""
        # 调用 GitHub API
        pass
    
    async def _publish_report(self, pr_url: str, report):
        """发布审查报告"""
        # 更新 PR 评论
        pass

# 各 Agent 实现
class StaticAnalysisAgent:
    async def analyze(self, changes) -> Dict:
        """静态分析"""
        issues = []
        for file in changes.files:
            if file.language == 'python':
                issues.extend(
                    await self._analyze_python(file)
                )
        return {'type': 'static', 'issues': issues}

class SecurityScannerAgent:
    async def scan(self, changes) -> Dict:
        """安全扫描"""
        vulnerabilities = []
        # 扫描依赖漏洞
        # 扫描代码注入风险
        return {'type': 'security', 'vulnerabilities': vulnerabilities}
```

## 监控和可观测性

### 关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| 任务完成率 | 成功完成的任务比例 | > 99% |
| 平均执行时间 | 从开始到结束的耗时 | < 30s |
| Agent 利用率 | Agent 繁忙时间占比 | 60-80% |
| 队列深度 | 等待处理的任务数 | < 100 |
| 错误率 | 失败任务占比 | < 1% |

### 分布式追踪

参考 OpenTelemetry【7】：

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# 配置追踪
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

class TracedAgent:
    async def execute(self, task):
        with tracer.start_as_current_span("agent.execute") as span:
            span.set_attribute("task.type", task.type)
            span.set_attribute("task.id", task.id)
            
            try:
                result = await self._do_execute(task)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(e))
                )
                raise
```

## 最佳实践总结

### 编排模式选择
- 简单依赖：顺序流水线
- 独立子任务：扇出-扇入
- 大数据处理：Map-Reduce
- 动态响应：事件驱动
- 复杂分解：层级委派

### 容错设计
- 重试：指数退避 + 抖动
- 熔断：快速失败 + 自动恢复
- 降级：多级 fallback

### 协调机制
- 共享状态：Redis + 分布式锁
- 消息传递：优先级队列
- 负载均衡：按需选择策略

## 参考来源

1. Kubernetes Documentation: "Pod Overview" - https://kubernetes.io/docs/concepts/workloads/pods/
2. Apache Airflow: "DAGs" - https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html
3. Redis Documentation: "Distributed Locks" - https://redis.io/docs/manual/patterns/distributed-locks/
4. RabbitMQ Tutorials: "Topics" - https://www.rabbitmq.com/tutorials/tutorial-five-python
5. Google SRE Book: "Handling Overload" - https://sre.google/sre-book/handling-overload/
6. Netflix Hystrix: "How It Works" - https://github.com/Netflix/Hystrix/wiki/How-it-Works
7. OpenTelemetry Documentation: "Traces" - https://opentelemetry.io/docs/concepts/signals/traces/

---

*本文首发于 RiceBall-15 的技术博客，转载请注明出处。*
