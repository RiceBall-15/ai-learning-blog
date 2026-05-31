---
title: '高效任务调度与异步执行：Agent系统的并发与资源管理'
description: '从任务队列到资源隔离，全面解析Agent系统的高效任务调度与异步执行架构'
date: 2026-05-30
author: 'RiceBall-15'
category: 'agent'
subCategory: 'interview'
tags: ['任务调度', '异步执行', '并发控制', '资源管理']
draft: false
---

# 高效任务调度与异步执行：Agent系统的并发与资源管理

## 引言

Agent系统的核心挑战之一：**如何高效地调度和执行大量并发任务？**

一个Agent可能同时需要：调用LLM、执行工具、读写记忆、与其他Agent通信。如果串行执行，延迟会线性累加；如果无限制并行，资源会被耗尽。

本文从任务调度、异步执行、资源管理三个维度，解析Agent系统的高效执行架构。

---

## §1 Agent任务调度的挑战

### 1.1 任务类型多样性

| 任务类型 | 特点 | 资源需求 | 超时要求 |
|----------|------|----------|----------|
| LLM推理 | CPU/GPU密集 | 高 | 30s |
| 工具调用 | IO密集 | 低 | 10s |
| 向量检索 | CPU密集 | 中 | 5s |
| 记忆读写 | IO密集 | 低 | 2s |
| Agent通信 | 网络IO | 低 | 5s |

### 1.2 调度架构

```
┌─────────────────────────────────────────────────────────┐
│                   Task Scheduler                        │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Priority │    │ Resource │    │  Timeout │         │
│  │  Queue   │───▶│ Manager  │───▶│  Guard   │         │
│  └──────────┘    └──────────┘    └──────────┘         │
│       │              │               │                  │
│       ▼              ▼               ▼                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ Worker   │    │ Worker   │    │ Worker   │         │
│  │ Pool     │    │ Pool     │    │ Pool     │         │
│  │ (LLM)   │    │ (Tools)  │    │ (Memory) │         │
│  └──────────┘    └──────────┘    └──────────┘         │
└─────────────────────────────────────────────────────────┘
```

---

## §2 异步执行模式

### 2.1 asyncio基础模式

```python
import asyncio
from typing import Any, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum


class TaskPriority(Enum):
    CRITICAL = 0   # 关键路径
    HIGH = 1       # 高优先级
    NORMAL = 2     # 普通
    LOW = 3        # 低优先级


@dataclass
class AgentTask:
    """Agent任务定义"""
    task_id: str
    func: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 30.0
    retries: int = 2
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class AsyncTaskScheduler:
    """异步任务调度器"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results: dict[str, Any] = {}
        self.errors: dict[str, Exception] = {}
    
    async def submit(self, task: AgentTask) -> Any:
        """提交任务到调度器"""
        
        async with self.semaphore:
            for attempt in range(task.retries + 1):
                try:
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                    self.results[task.task_id] = result
                    return result
                    
                except asyncio.TimeoutError:
                    if attempt == task.retries:
                        self.errors[task.task_id] = TimeoutError(
                            f"Task {task.task_id} timed out after {task.timeout}s"
                        )
                        raise
                        
                except Exception as e:
                    if attempt == task.retries:
                        self.errors[task.task_id] = e
                        raise
                    # 指数退避重试
                    await asyncio.sleep(2 ** attempt)
    
    async def submit_batch(self, tasks: list[AgentTask]) -> list[Any]:
        """批量提交任务"""
        
        # 按优先级排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value)
        
        # 并发执行
        coros = [self.submit(task) for task in sorted_tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)
        
        return results
    
    async def submit_parallel(self, tasks: list[AgentTask]) -> dict:
        """并行执行多个独立任务"""
        
        results = {}
        tasks_map = {}
        
        for task in tasks:
            coro = self.submit(task)
            tasks_map[task.task_id] = coro
        
        # 使用asyncio.gather并发等待所有结果
        done, pending = await asyncio.wait(
            [self.submit(t) for t in tasks],
            timeout=max(t.timeout for t in tasks),
            return_when=asyncio.ALL_COMPLETED
        )
        
        return {
            'completed': len(done),
            'pending': len(pending),
            'results': self.results,
        }
```

### 2.2 Agent执行器

```python
class AgentExecutor:
    """Agent执行器 - 管理Agent的完整执行流程"""
    
    def __init__(self, scheduler: AsyncTaskScheduler):
        self.scheduler = scheduler
        self.checkpoints: dict[str, dict] = {}
    
    async def execute_react_loop(self, agent_state: dict,
                                  tools: list) -> dict:
        """执行ReAct循环"""
        
        max_iterations = 10
        
        for i in range(max_iterations):
            # 1. 思考阶段（LLM调用）
            think_task = AgentTask(
                task_id=f"think_{i}",
                func=self._think,
                args=(agent_state,),
                priority=TaskPriority.HIGH,
                timeout=30.0
            )
            
            thought = await self.scheduler.submit(think_task)
            agent_state['thought'] = thought
            
            # 2. 检查是否需要行动
            if thought.get('action') == 'respond':
                return thought
            
            # 3. 执行工具调用
            tool_task = AgentTask(
                task_id=f"tool_{i}",
                func=self._call_tool,
                args=(thought['tool_name'], thought['tool_args']),
                priority=TaskPriority.NORMAL,
                timeout=10.0
            )
            
            observation = await self.scheduler.submit(tool_task)
            agent_state['observation'] = observation
            
            # 4. 保存检查点
            self.checkpoints[f"step_{i}"] = agent_state.copy()
        
        raise RuntimeError("Agent loop exceeded max iterations")
    
    async def _think(self, state: dict) -> dict:
        """思考阶段"""
        # 调用LLM进行推理
        prompt = self._build_think_prompt(state)
        response = await self.llm.generate(prompt)
        return self._parse_think_response(response)
    
    async def _call_tool(self, tool_name: str, args: dict) -> Any:
        """工具调用"""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await tool.execute(**args)
```

---

## §3 资源管理与隔离

### 3.1 资源池

```python
import asyncio
from typing import Dict, Any


class ResourcePool:
    """资源池 - 管理有限资源的分配和回收"""
    
    def __init__(self, resource_type: str, max_size: int):
        self.resource_type = resource_type
        self.max_size = max_size
        self.available = asyncio.Queue(maxsize=max_size)
        self.in_use = set()
        
        # 初始化资源
        for i in range(max_size):
            resource = self._create_resource(i)
            self.available.put_nowait(resource)
    
    def _create_resource(self, index: int) -> Dict[str, Any]:
        """创建资源实例"""
        return {
            'id': f"{self.resource_type}_{index}",
            'type': self.resource_type,
            'created_at': asyncio.get_event_loop().time(),
        }
    
    async def acquire(self, timeout: float = 10.0) -> Dict[str, Any]:
        """获取资源"""
        try:
            resource = await asyncio.wait_for(
                self.available.get(),
                timeout=timeout
            )
            self.in_use.add(resource['id'])
            return resource
        except asyncio.TimeoutError:
            raise ResourceExhaustedError(
                f"No {self.resource_type} available within {timeout}s"
            )
    
    async def release(self, resource: Dict[str, Any]):
        """释放资源"""
        if resource['id'] in self.in_use:
            self.in_use.remove(resource['id'])
            await self.available.put(resource)
    
    def get_stats(self) -> dict:
        """获取资源统计"""
        return {
            'type': self.resource_type,
            'total': self.max_size,
            'available': self.available.qsize(),
            'in_use': len(self.in_use),
            'utilization': len(self.in_use) / self.max_size,
        }


class ResourceManager:
    """资源管理器 - 统一管理所有资源池"""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
    
    def register_pool(self, resource_type: str, max_size: int):
        """注册资源池"""
        self.pools[resource_type] = ResourcePool(resource_type, max_size)
    
    async def acquire(self, resource_type: str) -> Any:
        """获取资源"""
        pool = self.pools.get(resource_type)
        if not pool:
            raise ValueError(f"Unknown resource type: {resource_type}")
        return await pool.acquire()
    
    async def release(self, resource_type: str, resource: Any):
        """释放资源"""
        pool = self.pools.get(resource_type)
        if pool:
            await pool.release(resource)
    
    def get_all_stats(self) -> dict:
        """获取所有资源统计"""
        return {
            name: pool.get_stats()
            for name, pool in self.pools.items()
        }
```

### 3.2 并发控制

```python
class ConcurrencyController:
    """并发控制器 - 防止资源耗尽"""
    
    def __init__(self, config: dict):
        # 每种资源的并发限制
        self.limits = {
            'llm_calls': config.get('llm_max_concurrent', 5),
            'tool_calls': config.get('tool_max_concurrent', 20),
            'memory_ops': config.get('memory_max_concurrent', 10),
        }
        
        # 信号量
        self.semaphores = {
            name: asyncio.Semaphore(limit)
            for name, limit in self.limits.items()
        }
        
        # 当前使用计数
        self.current_usage = {name: 0 for name in self.limits}
    
    async def execute_with_limit(self, resource_type: str,
                                  func: Callable,
                                  *args, **kwargs) -> Any:
        """在并发限制内执行任务"""
        
        semaphore = self.semaphores.get(resource_type)
        if not semaphore:
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        async with semaphore:
            self.current_usage[resource_type] += 1
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                self.current_usage[resource_type] -= 1
    
    def get_usage(self) -> dict:
        """获取当前使用情况"""
        return {
            name: {
                'current': self.current_usage[name],
                'limit': self.limits[name],
                'utilization': self.current_usage[name] / self.limits[name],
            }
            for name in self.limits
        }
```

---

## §4 任务依赖与DAG执行

```python
from typing import Set, Dict, List
import asyncio


class DAGExecutor:
    """DAG任务执行器 - 处理任务间的依赖关系"""
    
    def __init__(self):
        self.tasks: Dict[str, AgentTask] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.results: Dict[str, Any] = {}
    
    def add_task(self, task_id: str, func: Callable,
                 deps: Set[str] = None):
        """添加任务"""
        self.tasks[task_id] = AgentTask(
            task_id=task_id,
            func=func,
        )
        self.dependencies[task_id] = deps or set()
    
    def get_ready_tasks(self) -> List[str]:
        """获取可执行的任务（依赖已满足）"""
        
        ready = []
        for task_id, deps in self.dependencies.items():
            if task_id not in self.results:
                # 检查所有依赖是否已完成
                if all(dep in self.results for dep in deps):
                    ready.append(task_id)
        
        return ready
    
    async def execute(self) -> Dict[str, Any]:
        """执行整个DAG"""
        
        scheduler = AsyncTaskScheduler(max_concurrent=10)
        
        while True:
            # 获取可执行的任务
            ready_tasks = self.get_ready_tasks()
            
            if not ready_tasks:
                if len(self.results) == len(self.tasks):
                    # 所有任务完成
                    break
                else:
                    # 有循环依赖
                    raise RuntimeError("Circular dependency detected")
            
            # 并行执行就绪任务
            coros = []
            for task_id in ready_tasks:
                task = self.tasks[task_id]
                # 注入依赖结果
                dep_results = {
                    dep: self.results[dep] 
                    for dep in self.dependencies[task_id]
                }
                task.kwargs['dep_results'] = dep_results
                coros.append(self._execute_task(task))
            
            # 等待所有任务完成
            results = await asyncio.gather(*coros, return_exceptions=True)
            
            # 记录结果
            for task_id, result in zip(ready_tasks, results):
                if isinstance(result, Exception):
                    self.results[task_id] = {'error': str(result)}
                else:
                    self.results[task_id] = result
        
        return self.results
    
    async def _execute_task(self, task: AgentTask) -> Any:
        """执行单个任务"""
        return await task.func(**task.kwargs)
```

---

## §5 实战案例：并行工具调用

```python
class ParallelToolExecutor:
    """并行工具执行器 - 同时执行多个工具调用"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_tools(self, tool_calls: list) -> list:
        """并行执行多个工具调用"""
        
        async def _exec_one(tool_call: dict):
            async with self.semaphore:
                tool_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                tool = self.tools.get(tool_name)
                if not tool:
                    return {
                        'tool_call_id': tool_call['id'],
                        'error': f'Tool {tool_name} not found'
                    }
                
                try:
                    result = await asyncio.wait_for(
                        tool.execute(**args),
                        timeout=10.0
                    )
                    return {
                        'tool_call_id': tool_call['id'],
                        'result': result
                    }
                except asyncio.TimeoutError:
                    return {
                        'tool_call_id': tool_call['id'],
                        'error': f'Tool {tool_name} timed out'
                    }
        
        # 并发执行所有工具调用
        tasks = [_exec_one(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
```

---

## §6 性能指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| 并发任务数 | 同时执行的任务数量 | 按资源调整 |
| 任务完成率 | 成功完成的任务比例 | >99% |
| 平均等待时间 | 任务从提交到开始执行的时间 | <100ms |
| 资源利用率 | 资源使用比例 | 60-80% |
| 超时率 | 超时任务的比例 | <5% |

---

## §7 总结

高效任务调度的三个关键：

1. **异步执行**：使用asyncio并发处理IO密集任务
2. **资源管理**：资源池+信号量控制并发度
3. **DAG执行**：处理任务依赖，最大化并行度

**面试要点：**
- 能设计异步任务调度器
- 能处理任务间的依赖关系
- 能实现资源隔离和并发控制

## 参考资料

- Python asyncio官方文档
- Celery任务队列设计
- Ray分布式计算框架
