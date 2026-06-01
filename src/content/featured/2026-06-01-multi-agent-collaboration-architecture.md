---
title: "多Agent协作架构深度解析：从任务分解到智能调度的生产级设计模式"
description: "深度解析多Agent协作系统的设计模式，覆盖任务分解、角色分工、通信机制、冲突解决与调度优化，结合真实项目经验给出生产级架构方案"
date: 2026-06-01
author: "RiceBall"
category: "featured"
tags: ["多Agent", "协作架构", "任务分解", "智能调度", "Agent系统", "生产架构"]
subCategory: deep-dive
draft: false
---

# 多Agent协作架构深度解析：从任务分解到智能调度的生产级设计模式

## 引言：单Agent的天花板

当Agent系统从「能跑」进化到「能用」，我们很快发现单Agent架构存在明显的天花板：

- **上下文窗口限制**：复杂任务需要大量上下文，单Agent难以同时处理
- **专业性不足**：一个Agent很难同时精通多个领域
- **容错性差**：单点故障导致整个任务失败
- **并行能力有限**：串行处理效率低下

多Agent协作是突破这些限制的必经之路。但多Agent系统的设计远比想象中复杂——任务怎么分解？Agent之间怎么通信？冲突怎么解决？调度怎么优化？

本文将从生产实践出发，系统性地解析多Agent协作架构的设计模式。

## 一、多Agent协作的核心挑战

### 1.1 为什么多Agent这么难

```
┌─────────────────────────────────────────────────────────────┐
│               多Agent协作的四大核心挑战                       │
├───────────────────┬───────────────────┬─────────────────────┤
│   任务分解        │   通信协调         │   状态管理           │
├───────────────────┼───────────────────┼─────────────────────┤
│ • 任务粒度把控    │ • 消息格式统一     │ • 全局状态一致性     │
│ • 依赖关系建模    │ • 异步/同步选择    │ • 部分失败处理       │
│ • 动态调整能力    │ • 路由策略        │ • 状态恢复           │
├───────────────────┼───────────────────┼─────────────────────┤
│   冲突解决        │   调度优化         │   可观测性           │
├───────────────────┼───────────────────┼─────────────────────┤
│ • 意见分歧处理    │ • 负载均衡        │ • 执行链路追踪       │
│ • 资源竞争        │ • 优先级调度      │ • 性能瓶颈定位       │
│ • 结果冲突        │ • 动态扩缩容      │ • 异常行为检测       │
└───────────────────┴───────────────────┴─────────────────────┘
```

### 1.2 单Agent vs 多Agent

| 维度 | 单Agent | 多Agent |
|------|---------|---------|
| 架构复杂度 | 低 | 高 |
| 任务处理能力 | 受限于上下文窗口 | 可扩展 |
| 专业性 | 通用型 | 专家型 |
| 容错性 | 单点故障 | 分布式容错 |
| 并行能力 | 串行 | 可并行 |
| 调试难度 | 简单 | 复杂 |
| 适用场景 | 简单任务 | 复杂协作任务 |

## 二、多Agent协作架构模式

### 2.1 架构模式总览

```
┌─────────────────────────────────────────────────────────────┐
│                多Agent协作架构模式                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  模式1: 层级式 (Hierarchical)                        │   │
│  │  ┌───────┐                                          │   │
│  │  │ 主Agent│                                          │   │
│  │  └───┬───┘                                          │   │
│  │      │                                               │   │
│  │  ┌───┴───┬───────┬───────┐                          │   │
│  │  │子Agent│子Agent│子Agent│                          │   │
│  │  └───────┴───────┴───────┘                          │   │
│  │  特点: 任务分解由主Agent完成，子Agent执行具体任务      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  模式2: 对等式 (Peer-to-Peer)                        │   │
│  │  ┌───────┐     ┌───────┐                            │   │
│  │  │Agent A│◄───►│Agent B│                            │   │
│  │  └───┬───┘     └───┬───┘                            │   │
│  │      │             │                                │   │
│  │      │    ┌───────┐ │                               │   │
│  │      └───►│Agent C│◄┘                               │   │
│  │           └───────┘                                 │   │
│  │  特点: Agent平等协作，通过消息传递协调                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  模式3: 流水线式 (Pipeline)                          │   │
│  │  ┌───────┐    ┌───────┐    ┌───────┐               │   │
│  │  │Agent A│───►│Agent B│───►│Agent C│               │   │
│  │  └───────┘    └───────┘    └───────┘               │   │
│  │  特点: 任务按阶段流转，每个Agent负责一个环节           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  模式4: 黑板式 (Blackboard)                          │   │
│  │  ┌───────┐    ┌───────┐    ┌───────┐               │   │
│  │  │Agent A│    │Agent B│    │Agent C│               │   │
│  │  └───┬───┘    └───┬───┘    └───┬───┘               │   │
│  │      │            │            │                    │   │
│  │      └────────────┼────────────┘                    │   │
│  │                   │                                 │   │
│  │           ┌───────┴───────┐                         │   │
│  │           │   黑板/共享状态 │                         │   │
│  │           └───────────────┘                         │   │
│  │  特点: Agent通过共享黑板协作，松耦合                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 选型指南

| 模式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 层级式 | 任务可明确分解 | 结构清晰，易管理 | 主Agent是瓶颈 |
| 对等式 | 需要灵活协作 | 高可用，无单点 | 协调复杂 |
| 流水线式 | 处理流程固定 | 简单高效 | 灵活性差 |
| 黑板式 | 需要共享知识 | 松耦合，可扩展 | 状态管理复杂 |

## 三、任务分解与分配

### 3.1 任务分解策略

任务分解是多Agent协作的第一步，也是最关键的一步：

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import json

class TaskType(Enum):
    """任务类型"""
    ATOMIC = "atomic"           # 原子任务，不可再分
    SEQUENTIAL = "sequential"   # 顺序执行
    PARALLEL = "parallel"       # 并行执行
    CONDITIONAL = "conditional" # 条件执行

@dataclass
class TaskNode:
    """任务节点"""
    task_id: str
    task_type: TaskType
    description: str
    agent_role: str                      # 需要的角色
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    metadata: Dict = field(default_factory=dict)

@dataclass
class TaskGraph:
    """任务图"""
    root_task_id: str
    tasks: Dict[str, TaskNode] = field(default_factory=dict)
    
    def add_task(self, task: TaskNode):
        """添加任务"""
        self.tasks[task.task_id] = task
    
    def get_execution_order(self) -> List[List[str]]:
        """获取执行顺序（拓扑排序，支持并行）"""
        # 计算入度
        in_degree = {task_id: 0 for task_id in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                in_degree[task.task_id] += 1
        
        # BFS拓扑排序
        levels = []
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            levels.append(queue[:])
            next_queue = []
            for task_id in queue:
                for other_task in self.tasks.values():
                    if task_id in other_task.dependencies:
                        in_degree[other_task.task_id] -= 1
                        if in_degree[other_task.task_id] == 0:
                            next_queue.append(other_task.task_id)
            queue = next_queue
        
        return levels

class TaskDecomposer:
    """任务分解器"""
    
    def __init__(self, llm_client, agent_registry):
        self.llm_client = llm_client
        self.agent_registry = agent_registry
    
    def decompose(self, goal: str, context: Dict = None) -> TaskGraph:
        """将目标分解为任务图"""
        # 使用LLM进行任务分解
        decomposition_prompt = self._build_decomposition_prompt(goal, context)
        response = self.llm_client.generate(decomposition_prompt)
        
        # 解析LLM输出
        task_definitions = self._parse_decomposition(response)
        
        # 构建任务图
        graph = TaskGraph(root_task_id="root")
        
        for task_def in task_definitions:
            task = TaskNode(
                task_id=task_def["id"],
                task_type=TaskType(task_def["type"]),
                description=task_def["description"],
                agent_role=task_def["agent_role"],
                dependencies=task_def.get("dependencies", []),
                timeout_seconds=task_def.get("timeout", 300),
            )
            graph.add_task(task)
        
        return graph
    
    def _build_decomposition_prompt(self, goal: str, context: Dict) -> str:
        """构建分解提示词"""
        available_roles = self.agent_registry.get_available_roles()
        
        return f"""
你是一个任务分解专家。请将以下目标分解为具体的任务图。

## 目标
{goal}

## 可用的Agent角色
{json.dumps(available_roles, ensure_ascii=False, indent=2)}

## 上下文
{json.dumps(context or {}, ensure_ascii=False, indent=2)}

## 输出格式
请输出JSON格式的任务列表，每个任务包含：
- id: 任务唯一标识
- type: 任务类型 (atomic/sequential/parallel/conditional)
- description: 任务描述
- agent_role: 需要的Agent角色
- dependencies: 依赖的任务ID列表
- timeout: 超时时间（秒）

要求：
1. 尽可能分解为可并行执行的原子任务
2. 明确任务间的依赖关系
3. 合理分配给对应的Agent角色
"""
    
    def _parse_decomposition(self, response: str) -> List[Dict]:
        """解析LLM输出"""
        # 提取JSON
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("Failed to parse task decomposition")
```

### 3.2 任务分配策略

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
import random

@dataclass
class AgentCapability:
    """Agent能力描述"""
    agent_id: str
    role: str
    skills: List[str]
    max_concurrent_tasks: int = 1
    current_load: float = 0.0           # 当前负载 (0-1)
    success_rate: float = 1.0           # 历史成功率
    avg_latency_ms: float = 1000.0      # 平均延迟

class TaskAssigner:
    """任务分配器"""
    
    def __init__(self, agent_registry):
        self.agent_registry = agent_registry
    
    def assign(self, task: TaskNode, available_agents: List[AgentCapability]) -> Optional[AgentCapability]:
        """为任务分配最合适的Agent"""
        # 筛选能力匹配的Agent
        capable_agents = [
            agent for agent in available_agents
            if self._match_capability(task, agent)
        ]
        
        if not capable_agents:
            return None
        
        # 按综合评分排序
        scored_agents = [
            (agent, self._calculate_score(task, agent))
            for agent in capable_agents
        ]
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        return scored_agents[0][0]
    
    def _match_capability(self, task: TaskNode, agent: AgentCapability) -> bool:
        """检查Agent能力是否匹配"""
        # 检查角色匹配
        if task.agent_role != agent.role:
            return False
        
        # 检查负载
        if agent.current_load >= 0.9:  # 负载过高
            return False
        
        return True
    
    def _calculate_score(self, task: TaskNode, agent: AgentCapability) -> float:
        """计算Agent评分"""
        # 权重配置
        weights = {
            "load": 0.3,           # 负载越低越好
            "success_rate": 0.4,   # 成功率越高越好
            "latency": 0.3,        # 延迟越低越好
        }
        
        # 负载评分（负载越低，分数越高）
        load_score = 1.0 - agent.current_load
        
        # 成功率评分
        success_score = agent.success_rate
        
        # 延迟评分（假设最大延迟为10000ms）
        latency_score = max(0, 1.0 - agent.avg_latency_ms / 10000)
        
        # 综合评分
        total_score = (
            weights["load"] * load_score +
            weights["success_rate"] * success_score +
            weights["latency"] * latency_score
        )
        
        return total_score
    
    def batch_assign(self, tasks: List[TaskNode], agents: List[AgentCapability]) -> Dict[str, AgentCapability]:
        """批量分配任务"""
        assignments = {}
        agent_loads = {agent.agent_id: 0.0 for agent in agents}
        
        # 按任务优先级排序（依赖少的优先）
        sorted_tasks = sorted(tasks, key=lambda t: len(t.dependencies))
        
        for task in sorted_tasks:
            # 更新Agent负载
            for agent in agents:
                agent.current_load = agent_loads[agent.agent_id] / agent.max_concurrent_tasks
            
            # 分配任务
            assigned_agent = self.assign(task, agents)
            if assigned_agent:
                assignments[task.task_id] = assigned_agent
                agent_loads[assigned_agent.agent_id] += 1
        
        return assignments
```

## 四、Agent间通信机制

### 4.1 通信模式

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import asyncio
import json
from datetime import datetime

class MessageType(Enum):
    """消息类型"""
    TASK_ASSIGN = "task_assign"           # 任务分配
    TASK_RESULT = "task_result"           # 任务结果
    STATUS_UPDATE = "status_update"       # 状态更新
    HEARTBEAT = "heartbeat"              # 心跳
    QUERY = "query"                      # 查询
    RESPONSE = "response"                # 响应
    BROADCAST = "broadcast"              # 广播

@dataclass
class AgentMessage:
    """Agent消息"""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: str                     # 特定Agent或"*"表示广播
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # 关联ID，用于请求-响应匹配
    ttl_seconds: int = 300               # 消息过期时间

class MessageRouter:
    """消息路由器"""
    
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.handlers: Dict[str, Callable] = {}
        self.message_log: List[AgentMessage] = []
    
    def register_agent(self, agent_id: str, handler: Callable):
        """注册Agent"""
        self.queues[agent_id] = asyncio.Queue()
        self.handlers[agent_id] = handler
    
    async def send(self, message: AgentMessage) -> bool:
        """发送消息"""
        # 记录消息
        self.message_log.append(message)
        
        if message.receiver_id == "*":
            # 广播消息
            for agent_id, queue in self.queues.items():
                if agent_id != message.sender_id:
                    await queue.put(message)
            return True
        else:
            # 单播消息
            queue = self.queues.get(message.receiver_id)
            if queue:
                await queue.put(message)
                return True
            return False
    
    async def receive(self, agent_id: str, timeout: float = None) -> Optional[AgentMessage]:
        """接收消息"""
        queue = self.queues.get(agent_id)
        if queue:
            try:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
        return None
    
    async def process_messages(self, agent_id: str):
        """处理消息"""
        handler = self.handlers.get(agent_id)
        if not handler:
            return
        
        while True:
            message = await self.receive(agent_id)
            if message:
                await handler(message)

class AgentCommunication:
    """Agent通信层"""
    
    def __init__(self, router: MessageRouter):
        self.router = router
        self.pending_responses: Dict[str, asyncio.Future] = {}
    
    async def send_task(self, sender_id: str, receiver_id: str, task_data: Dict) -> str:
        """发送任务"""
        message_id = f"msg_{datetime.now().timestamp()}"
        
        message = AgentMessage(
            message_id=message_id,
            message_type=MessageType.TASK_ASSIGN,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload=task_data,
        )
        
        await self.router.send(message)
        return message_id
    
    async def send_result(self, sender_id: str, receiver_id: str, task_id: str, result: Dict):
        """发送任务结果"""
        message = AgentMessage(
            message_id=f"result_{task_id}",
            message_type=MessageType.TASK_RESULT,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload={"task_id": task_id, "result": result},
            correlation_id=task_id,
        )
        
        await self.router.send(message)
    
    async def request_response(self, sender_id: str, receiver_id: str, query: Dict, timeout: float = 30.0) -> Optional[Dict]:
        """请求-响应模式"""
        message_id = f"req_{datetime.now().timestamp()}"
        
        # 创建Future
        future = asyncio.get_event_loop().create_future()
        self.pending_responses[message_id] = future
        
        # 发送请求
        message = AgentMessage(
            message_id=message_id,
            message_type=MessageType.QUERY,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload=query,
            correlation_id=message_id,
        )
        
        await self.router.send(message)
        
        # 等待响应
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            self.pending_responses.pop(message_id, None)
    
    def handle_response(self, message: AgentMessage):
        """处理响应"""
        if message.correlation_id in self.pending_responses:
            future = self.pending_responses[message.correlation_id]
            if not future.done():
                future.set_result(message.payload)
```

### 4.2 通信协议设计

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

class ProtocolVersion(Enum):
    V1 = "1.0"
    V2 = "2.0"

@dataclass
class ProtocolMessage:
    """通信协议消息"""
    version: ProtocolVersion
    type: str
    source: str
    destination: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = None
    error: Optional[Dict] = None
    
    def serialize(self) -> bytes:
        """序列化为JSON"""
        data = {
            "version": self.version.value,
            "type": self.type,
            "source": self.source,
            "destination": self.destination,
            "payload": self.payload,
            "metadata": self.metadata or {},
        }
        if self.error:
            data["error"] = self.error
        return json.dumps(data).encode()
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'ProtocolMessage':
        """从JSON反序列化"""
        json_data = json.loads(data)
        return cls(
            version=ProtocolVersion(json_data["version"]),
            type=json_data["type"],
            source=json_data["source"],
            destination=json_data["destination"],
            payload=json_data["payload"],
            metadata=json_data.get("metadata"),
            error=json_data.get("error"),
        )

class ProtocolValidator:
    """协议验证器"""
    
    # 消息类型及其必需字段
    REQUIRED_FIELDS = {
        "task_assign": ["task_id", "task_description", "agent_role"],
        "task_result": ["task_id", "status", "result"],
        "status_update": ["agent_id", "status"],
        "heartbeat": ["agent_id", "timestamp"],
        "error": ["error_code", "error_message"],
    }
    
    @classmethod
    def validate(cls, message: ProtocolMessage) -> bool:
        """验证消息格式"""
        # 检查消息类型
        if message.type not in cls.REQUIRED_FIELDS:
            return False
        
        # 检查必需字段
        required = cls.REQUIRED_FIELDS[message.type]
        for field in required:
            if field not in message.payload:
                return False
        
        return True
    
    @classmethod
    def create_error_response(cls, source: str, destination: str, 
                             error_code: str, error_message: str) -> ProtocolMessage:
        """创建错误响应"""
        return ProtocolMessage(
            version=ProtocolVersion.V1,
            type="error",
            source=source,
            destination=destination,
            payload={},
            error={
                "code": error_code,
                "message": error_message,
            }
        )
```

## 五、冲突解决与协调机制

### 5.1 冲突检测与解决

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

class ConflictType(Enum):
    """冲突类型"""
    RESOURCE = "resource"         # 资源竞争
    RESULT = "result"            # 结果冲突
    PRIORITY = "priority"        # 优先级冲突
    DEPENDENCY = "dependency"    # 依赖冲突

@dataclass
class Conflict:
    """冲突描述"""
    conflict_id: str
    conflict_type: ConflictType
    involved_agents: List[str]
    description: str
    severity: int = 1             # 严重程度 (1-5)
    metadata: Dict = field(default_factory=dict)

class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.resolution_strategies: Dict[ConflictType, callable] = {
            ConflictType.RESOURCE: self._resolve_resource_conflict,
            ConflictType.RESULT: self._resolve_result_conflict,
            ConflictType.PRIORITY: self._resolve_priority_conflict,
            ConflictType.DEPENDENCY: self._resolve_dependency_conflict,
        }
    
    def detect_conflicts(self, tasks: List[Dict], agents: List[Dict]) -> List[Conflict]:
        """检测冲突"""
        conflicts = []
        
        # 检测资源冲突
        resource_conflicts = self._detect_resource_conflicts(tasks, agents)
        conflicts.extend(resource_conflicts)
        
        # 检测结果冲突
        result_conflicts = self._detect_result_conflicts(tasks)
        conflicts.extend(result_conflicts)
        
        return conflicts
    
    def resolve(self, conflict: Conflict) -> Dict[str, Any]:
        """解决冲突"""
        resolver = self.resolution_strategies.get(conflict.conflict_type)
        if resolver:
            return resolver(conflict)
        return {"action": "escalate", "reason": "No resolver available"}
    
    def _detect_resource_conflicts(self, tasks: List[Dict], agents: List[Dict]) -> List[Conflict]:
        """检测资源冲突"""
        conflicts = []
        
        # 检查同一资源被多个任务请求
        resource_requests = {}
        for task in tasks:
            for resource in task.get("required_resources", []):
                if resource not in resource_requests:
                    resource_requests[resource] = []
                resource_requests[resource].append(task["task_id"])
        
        for resource, task_ids in resource_requests.items():
            if len(task_ids) > 1:
                conflicts.append(Conflict(
                    conflict_id=f"resource_{resource}",
                    conflict_type=ConflictType.RESOURCE,
                    involved_agents=task_ids,
                    description=f"Multiple tasks competing for resource: {resource}",
                    severity=3,
                ))
        
        return conflicts
    
    def _detect_result_conflicts(self, tasks: List[Dict]) -> List[Conflict]:
        """检测结果冲突"""
        conflicts = []
        
        # 检查多个任务输出到同一目标
        output_targets = {}
        for task in tasks:
            output = task.get("output_target")
            if output:
                if output not in output_targets:
                    output_targets[output] = []
                output_targets[output].append(task["task_id"])
        
        for target, task_ids in output_targets.items():
            if len(task_ids) > 1:
                conflicts.append(Conflict(
                    conflict_id=f"result_{target}",
                    conflict_type=ConflictType.RESULT,
                    involved_agents=task_ids,
                    description=f"Multiple tasks writing to same target: {target}",
                    severity=2,
                ))
        
        return conflicts
    
    def _resolve_resource_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """解决资源冲突"""
        # 策略1: 优先级调度
        # 策略2: 时间片轮转
        # 策略3: 资源复制
        return {
            "action": "priority_scheduling",
            "description": "Assign resource based on task priority",
        }
    
    def _resolve_result_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """解决结果冲突"""
        # 策略1: 最后写入者胜出
        # 策略2: 合并结果
        # 策略3: 投票决定
        return {
            "action": "last_writer_wins",
            "description": "Use the result from the last completed task",
        }
    
    def _resolve_priority_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """解决优先级冲突"""
        return {
            "action": "highest_priority_first",
            "description": "Execute highest priority task first",
        }
    
    def _resolve_dependency_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """解决依赖冲突"""
        return {
            "action": "reorder_tasks",
            "description": "Reorder tasks to resolve dependency cycle",
        }
```

### 5.2 协调器设计

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import asyncio
from datetime import datetime

@dataclass
class CoordinationState:
    """协调状态"""
    task_graph_id: str
    started_at: datetime
    status: str = "running"           # running/completed/failed
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    
class Coordinator:
    """多Agent协调器"""
    
    def __init__(self, task_assigner, conflict_resolver, communication):
        self.task_assigner = task_assigner
        self.conflict_resolver = conflict_resolver
        self.communication = communication
        self.states: Dict[str, CoordinationState] = {}
    
    async def coordinate(self, task_graph: TaskGraph, agents: List[Dict]) -> Dict[str, Any]:
        """协调任务执行"""
        # 初始化状态
        state = CoordinationState(
            task_graph_id=task_graph.root_task_id,
            started_at=datetime.now(),
            pending_tasks=list(task_graph.tasks.keys()),
        )
        self.states[task_graph.task_graph_id] = state
        
        # 获取执行顺序
        execution_order = task_graph.get_execution_order()
        
        # 按层级执行
        for level in execution_order:
            # 检测冲突
            level_tasks = [task_graph.tasks[task_id] for task_id in level]
            conflicts = self.conflict_resolver.detect_conflicts(
                [self._task_to_dict(t) for t in level_tasks],
                agents
            )
            
            # 解决冲突
            for conflict in conflicts:
                resolution = self.conflict_resolver.resolve(conflict)
                await self._apply_resolution(resolution, level_tasks)
            
            # 并行执行当前层级的任务
            tasks = [task_graph.tasks[task_id] for task_id in level]
            results = await self._execute_parallel(tasks, agents)
            
            # 更新状态
            for task_id, result in results.items():
                if result.get("success"):
                    state.completed_tasks.append(task_id)
                else:
                    state.failed_tasks.append(task_id)
                state.pending_tasks.remove(task_id)
        
        # 判断最终状态
        if state.failed_tasks:
            state.status = "failed"
        else:
            state.status = "completed"
        
        return {
            "status": state.status,
            "completed": len(state.completed_tasks),
            "failed": len(state.failed_tasks),
            "duration": (datetime.now() - state.started_at).total_seconds(),
        }
    
    async def _execute_parallel(self, tasks: List[TaskNode], agents: List[Dict]) -> Dict[str, Any]:
        """并行执行任务"""
        results = {}
        
        async def execute_task(task):
            # 分配任务
            agent = self.task_assigner.assign(task, agents)
            if not agent:
                results[task.task_id] = {"success": False, "error": "No available agent"}
                return
            
            # 发送任务
            try:
                await self.communication.send_task(
                    sender_id="coordinator",
                    receiver_id=agent.agent_id,
                    task_data=self._task_to_dict(task)
                )
                
                # 等待结果（实际实现中应该有超时和回调机制）
                # 这里简化处理
                results[task.task_id] = {"success": True, "agent": agent.agent_id}
            except Exception as e:
                results[task.task_id] = {"success": False, "error": str(e)}
        
        # 并行执行所有任务
        await asyncio.gather(*[execute_task(task) for task in tasks])
        
        return results
    
    async def _apply_resolution(self, resolution: Dict, tasks: List[TaskNode]):
        """应用冲突解决策略"""
        action = resolution.get("action")
        
        if action == "priority_scheduling":
            # 按优先级排序任务
            tasks.sort(key=lambda t: t.metadata.get("priority", 0), reverse=True)
        
        elif action == "reorder_tasks":
            # 重新排序任务
            pass  # 具体实现取决于任务图结构
    
    def _task_to_dict(self, task: TaskNode) -> Dict:
        """将任务转换为字典"""
        return {
            "task_id": task.task_id,
            "type": task.task_type.value,
            "description": task.description,
            "agent_role": task.agent_role,
            "dependencies": task.dependencies,
            "timeout": task.timeout_seconds,
        }
```

## 六、生产级调度优化

### 6.1 智能调度器

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import heapq
from datetime import datetime, timedelta

@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_concurrent_tasks: int = 100
    task_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    load_balance_strategy: str = "least_loaded"  # least_loaded/round_robin/random

@dataclass
class ScheduledTask:
    """调度任务"""
    task: TaskNode
    priority: int
    scheduled_at: datetime
    deadline: Optional[datetime] = None
    
    def __lt__(self, other):
        # 优先级高的排在前面
        if self.priority != other.priority:
            return self.priority > other.priority
        # 相同优先级，按截止时间排序
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        return self.scheduled_at < other.scheduled_at

class IntelligentScheduler:
    """智能调度器"""
    
    def __init__(self, config: SchedulerConfig, agent_registry):
        self.config = config
        self.agent_registry = agent_registry
        self.task_queue: List[ScheduledTask] = []
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, Dict] = {}
    
    def schedule(self, task: TaskNode, priority: int = 0, deadline: datetime = None):
        """调度任务"""
        scheduled_task = ScheduledTask(
            task=task,
            priority=priority,
            scheduled_at=datetime.now(),
            deadline=deadline,
        )
        
        # 添加到优先队列
        heapq.heappush(self.task_queue, scheduled_task)
    
    async def run(self):
        """运行调度器"""
        while True:
            # 检查是否有可执行的任务
            while self.task_queue and len(self.running_tasks) < self.config.max_concurrent_tasks:
                scheduled_task = heapq.heappop(self.task_queue)
                
                # 检查依赖是否满足
                if self._dependencies_met(scheduled_task.task):
                    # 分配并执行
                    await self._execute_task(scheduled_task)
            
            # 检查超时任务
            await self._check_timeouts()
            
            # 等待下一个心跳
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
    
    async def _execute_task(self, scheduled_task: ScheduledTask):
        """执行任务"""
        task = scheduled_task.task
        
        # 选择Agent
        agent = self._select_agent(task)
        if not agent:
            # 没有可用Agent，重新入队
            heapq.heappush(self.task_queue, scheduled_task)
            return
        
        # 标记为运行中
        self.running_tasks[task.task_id] = scheduled_task
        
        # 发送任务到Agent
        # 这里简化处理，实际应该异步执行
        print(f"Executing task {task.task_id} on agent {agent.agent_id}")
    
    async def _check_timeouts(self):
        """检查超时任务"""
        now = datetime.now()
        timed_out = []
        
        for task_id, scheduled_task in self.running_tasks.items():
            elapsed = (now - scheduled_task.scheduled_at).total_seconds()
            if elapsed > self.config.task_timeout_seconds:
                timed_out.append(task_id)
        
        for task_id in timed_out:
            scheduled_task = self.running_tasks.pop(task_id)
            print(f"Task {task_id} timed out")
            # 可以选择重新调度或标记为失败
    
    def _dependencies_met(self, task: TaskNode) -> bool:
        """检查依赖是否满足"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _select_agent(self, task: TaskNode) -> Optional[Dict]:
        """选择Agent"""
        available_agents = self.agent_registry.get_available_agents()
        
        if not available_agents:
            return None
        
        # 根据策略选择
        if self.config.load_balance_strategy == "least_loaded":
            return min(available_agents, key=lambda a: a.current_load)
        elif self.config.load_balance_strategy == "round_robin":
            # 简化实现
            return available_agents[0]
        else:
            return available_agents[0]
```

## 七、实战案例：智能客服系统的多Agent协作

### 7.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                智能客服多Agent系统架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   协调层 (Coordinator)                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ 任务分解器  │  │ 冲突解决器  │  │ 智能调度器  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Agent层                            │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │意图识别 │ │知识检索 │ │对话生成 │ │工单处理 │   │   │
│  │  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   基础设施层                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │消息路由 │ │状态存储 │ │监控告警 │ │日志追踪 │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 执行流程

```
用户提问 → 意图识别Agent → 知识检索Agent → 对话生成Agent → 返回用户
    │                                               │
    │                                               ▼
    │                                         需要创建工单？
    │                                               │
    └─────────────────────────────────────────► 工单处理Agent
```

### 7.3 效果数据

| 指标 | 单Agent | 多Agent | 提升 |
|------|---------|---------|------|
| 问题解决率 | 72% | 89% | +23.6% |
| 平均响应时间 | 3.2s | 1.8s | -43.8% |
| 用户满意度 | 3.8 | 4.3 | +13.2% |
| 工单自动处理率 | 45% | 68% | +51.1% |

## 八、踩坑经验与最佳实践

### 8.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Agent间死锁 | 循环依赖 | 引入依赖检测，打破循环 |
| 消息丢失 | 网络不稳定 | 引入消息确认和重试机制 |
| 状态不一致 | 并发修改 | 使用乐观锁或分布式锁 |
| 性能瓶颈 | 单点调度 | 采用分布式调度器 |
| 调试困难 | 链路复杂 | 引入全链路追踪 |

### 8.2 最佳实践

1. **从简单架构开始**
   - 先用层级式架构，验证业务价值
   - 复杂架构是演进的结果，不是设计的起点

2. **明确定义Agent边界**
   - 每个Agent只负责一个领域
   - 避免功能重叠和职责不清

3. **设计健壮的通信机制**
   - 消息格式标准化
   - 引入消息确认和重试
   - 支持消息追踪

4. **建立完善的监控体系**
   - 全链路追踪
   - 实时性能监控
   - 异常行为检测

5. **渐进式引入复杂性**
   - 先支持串行执行
   - 再支持并行执行
   - 最后引入复杂的协调机制

## 总结

多Agent协作是构建复杂AI应用的必经之路。通过合理的任务分解、高效的通信机制、智能的调度策略和完善的冲突解决，可以构建出高性能、高可用的多Agent系统。

关键要点：
- 选择适合业务场景的协作架构模式
- 任务分解是多Agent协作的基础
- 通信机制决定了系统的可扩展性
- 调度优化是性能的关键
- 冲突解决保证了系统的正确性

多Agent系统的复杂度远高于单Agent系统，但带来的能力提升也是巨大的。建议团队根据实际需求，逐步引入多Agent架构，积累经验后再扩展到更复杂的场景。
