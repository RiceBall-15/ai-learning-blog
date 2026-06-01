---
title: "Agent规划架构：从任务分解到动态重规划的系统设计"
description: "深入解析Agent规划系统的核心架构，覆盖任务分解策略、规划算法、动态重规划机制、与记忆系统的集成，以及生产环境中的实现方案"
date: 2026-05-31
author: "RiceBall-15"
category: "agent"
subCategory: agent-architecture
tags: ["Agent架构", "任务分解", "动态规划", "Planning", "多步推理"]
draft: false
---

# Agent规划架构：从任务分解到动态重规划的系统设计

## 1. 概念原理：为什么Agent需要规划系统

### 1.1 规划的本质

人类在执行复杂任务时，大脑会自然地进行"预演"——在动手之前先想清楚要做什么、按什么顺序做、可能遇到什么问题。Agent的规划系统正是对这一能力的工程化实现。

**规划（Planning）** 是Agent将高层目标分解为可执行步骤序列，并在执行过程中根据反馈动态调整计划的能力。没有规划能力的Agent只能做单步反应（ReAct模式），面对复杂任务时会陷入"走一步看一步"的低效状态。

### 1.2 规划的核心挑战

| 挑战 | 说明 | 影响 |
|------|------|------|
| **组合爆炸** | 步骤数增加时，可能的路径指数增长 | 搜索空间过大，规划超时 |
| **环境不确定性** | 执行过程中可能出现意外情况 | 计划失效，需要重新规划 |
| **资源约束** | Token预算、时间预算、工具调用次数限制 | 必须在有限资源内完成规划 |
| **子目标冲突** | 不同子任务之间可能存在依赖或矛盾 | 需要协调和优先级管理 |
| **长程依赖** | 前期决策影响后期执行 | 规划需要考虑全局最优 |

### 1.3 规划范式的演进

```
第一代：硬编码流程（If-Then规则）
  ↓ 缺乏灵活性
第二代：ReAct（Reason + Act 交替）
  ↓ 单步推理，缺乏全局视野
第三代：规划-执行分离（Plan-then-Execute）
  ↓ 计划僵化，难以应对变化
第四代：动态规划（Adaptive Planning）
  → 支持重规划、多路径探索、记忆驱动
```

**ReAct vs 规划型Agent的核心区别**：

- **ReAct**：思考→行动→观察→思考→行动...（线性链式）
- **规划型**：生成完整计划→按计划执行→监控偏差→动态调整（分层结构）

ReAct适合简单任务（3-5步），但面对10步以上的复杂任务时，缺乏全局视野会导致频繁的局部最优陷阱。

## 2. 架构设计：规划系统的分层架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────┐
│                  用户目标输入                      │
│              "帮我分析竞品并生成报告"               │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              🧠 规划层 (Planning Layer)            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ 任务分解器   │  │ 依赖分析器   │  │ 资源评估 │ │
│  │ (Decomposer)│  │ (Dependency) │  │ (Budget)│ │
│  └──────┬──────┘  └──────┬───────┘  └────┬────┘ │
│         └────────────────┼───────────────┘       │
│                          ▼                       │
│              ┌───────────────────┐               │
│              │  执行计划 (Plan)   │               │
│              │  Task DAG + 策略   │               │
│              └─────────┬─────────┘               │
└────────────────────────┼────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  执行引擎     │ │  执行引擎     │ │  执行引擎     │
│  Task A      │ │  Task B      │ │  Task C      │
│  (顺序执行)   │ │  (并行执行)   │ │  (条件执行)   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────┐
│              🔄 监控与重规划层                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ 偏差检测器   │  │ 重规划器     │  │ 回滚管理 │ │
│  │ (Monitor)   │  │ (Replanner) │  │ (Rollback)│ │
│  └─────────────┘  └──────────────┘  └─────────┘ │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              💾 记忆层 (Memory Layer)             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ 执行历史     │  │ 经验库       │  │ 上下文   │ │
│  │ (History)   │  │ (Experience)│  │ (Context)│ │
│  └─────────────┘  └──────────────┘  └─────────┘ │
└─────────────────────────────────────────────────┘
```

### 2.2 任务分解策略

#### 策略一：层级分解（Hierarchical Decomposition）

将大目标递归分解为子目标，直到每个子目标可直接执行。

```
目标：写一篇技术博客
├── 研究阶段
│   ├── 搜索相关资料（可执行）
│   ├── 分析竞品文章（可执行）
│   └── 提炼核心观点（可执行）
├── 写作阶段
│   ├── 撰写大纲（可执行）
│   ├── 撰写正文（可执行）
│   └── 添加代码示例（可执行）
└── 发布阶段
    ├── 本地预览（可执行）
    ├── 推送到GitHub（可执行）
    └── 验证部署（可执行）
```

**优点**：结构清晰，易于理解和调试  
**缺点**：分解粒度需要人工预判，过细浪费资源，过粗难以执行

#### 策略二：图分解（Graph Decomposition / DAG）

将任务建模为有向无环图（DAG），显式表达依赖关系。

```
    ┌──────────┐
    │ 搜索资料  │
    └────┬─────┘
         │
    ┌────▼─────┐    ┌──────────┐
    │ 分析竞品  │    │ 提炼观点  │ ← 可与分析竞品并行
    └────┬─────┘    └────┬─────┘
         │               │
    ┌────▼───────────────▼─┐
    │      撰写大纲         │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
  ┌──────────┐  ┌──────────┐
  │ 撰写正文  │  │ 添加代码  │ ← 可并行
  └────┬─────┘  └────┬─────┘
       └──────┬──────┘
              ▼
       ┌──────────┐
       │ 本地预览  │
       └────┬─────┘
            ▼
       ┌──────────┐
       │ 推送发布  │
       └──────────┘
```

**优点**：支持并行执行，精确表达依赖关系  
**缺点**：DAG构建本身需要额外计算，复杂度更高

#### 策略三：渐进式分解（Progressive Decomposition）

不一次性生成完整计划，而是"走一步看一步"——执行当前步骤后，根据结果决定下一步。

```
第1步：搜索资料 → 得到3篇参考文章
第2步：分析第1篇文章 → 发现有价值的架构图
  → 动态决策：将架构图分析加入计划
第3步：分析第2篇文章 → 发现内容重复
  → 动态决策：跳过，直接进入观点提炼
第4步：提炼观点 → 基于已有信息生成大纲
```

**优点**：灵活应对不确定性，避免过度规划  
**缺点**：缺乏全局视野，可能导致局部最优

### 2.3 规划算法对比

| 算法 | 原理 | 适用场景 | Token消耗 | 并行度 |
|------|------|----------|-----------|--------|
| **Chain of Thought** | 线性推理链 | 简单单步任务 | 低 | 无 |
| **Tree of Thought (ToT)** | 树形探索多条推理路径 | 需要回溯的复杂推理 | 高 | 可并行探索分支 |
| **Graph of Thought (GoT)** | 图结构，支持合并和循环 | 多步推理+知识融合 | 中-高 | 部分并行 |
| **LATS** | Monte Carlo树搜索 + Agent | 开放式探索任务 | 高 | 可并行 |
| **Plan-and-Solve** | 先生成完整计划再执行 | 步骤明确的结构化任务 | 中 | 按计划顺序 |
| **Dynamic Replanning** | 执行中监控+重规划 | 高不确定性环境 | 中 | 视重规划范围 |

## 3. 实战实现：构建一个规划型Agent

### 3.1 核心数据结构

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    output: str
    artifacts: list[str] = field(default_factory=list)  # 产出物路径
    metadata: dict = field(default_factory=dict)


@dataclass
class Task:
    """单个可执行任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # 依赖关系
    dependencies: list[str] = field(default_factory=list)  # 依赖的task id列表
    
    # 执行配置
    tool_name: Optional[str] = None  # 要调用的工具
    tool_args: dict = field(default_factory=dict)
    max_retries: int = 2
    
    # 执行结果
    result: Optional[TaskResult] = None
    retry_count: int = 0
    
    # 时间追踪
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_ready(self) -> bool:
        """检查所有依赖是否已完成"""
        return self.status == TaskStatus.PENDING and len(self.dependencies) == 0
    
    def can_retry(self) -> bool:
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries


@dataclass
class ExecutionPlan:
    """执行计划 - 任务的DAG表示"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    tasks: list[Task] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None
    
    def get_ready_tasks(self) -> list[Task]:
        """获取所有依赖已满足的待执行任务"""
        ready = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self.get_task(dep_id) is not None and 
                self.get_task(dep_id).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_met:
                ready.append(task)
        return ready
    
    def get_blocked_tasks(self) -> list[Task]:
        """获取因依赖失败而被阻塞的任务"""
        blocked = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                deps_failed = any(
                    self.get_task(dep_id) is not None and
                    self.get_task(dep_id).status == TaskStatus.FAILED
                    for dep_id in task.dependencies
                )
                if deps_failed:
                    blocked.append(task)
        return blocked
    
    @property
    def progress(self) -> float:
        total = len(self.tasks)
        if total == 0:
            return 1.0
        done = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return done / total
    
    @property
    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in self.tasks
        )
    
    def summary(self) -> str:
        status_counts = {}
        for t in self.tasks:
            status_counts[t.status.value] = status_counts.get(t.status.value, 0) + 1
        parts = [f"{k}: {v}" for k, v in status_counts.items()]
        return f"Plan {self.id} [{self.progress:.0%}] ({', '.join(parts)})"
```

### 3.2 任务分解器（Decomposer）

```python
import json
from typing import Protocol


class LLMClient(Protocol):
    """LLM客户端协议"""
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str: ...


class TaskDecomposer:
    """将高层目标分解为可执行的任务DAG"""
    
    DECOMPOSE_PROMPT = """你是一个任务规划专家。给定一个高层目标，将其分解为具体的、可执行的任务列表。

要求：
1. 每个任务应该是独立可执行的
2. 明确任务之间的依赖关系（前置任务ID）
3. 为每个任务指定最适合的工具类型
4. 控制任务总数在3-8个之间
5. 考虑可能的失败情况，设置合理的重试策略

可用工具类型：
- search: 搜索工具（web search, API查询等）
- analyze: 分析工具（数据处理、文本分析等）
- generate: 生成工具（文本生成、代码生成等）
- execute: 执行工具（文件操作、API调用等）
- verify: 验证工具（测试、校验等）

输出JSON格式：
```json
{{
  "tasks": [
    {{
      "id": "t1",
      "title": "任务标题",
      "description": "详细描述",
      "dependencies": [],
      "tool_type": "search",
      "priority": "high",
      "max_retries": 2
    }}
  ]
}}
```

目标：{goal}
"""
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def decompose(self, goal: str) -> ExecutionPlan:
        """将目标分解为执行计划"""
        prompt = self.DECOMPOSE_PROMPT.format(goal=goal)
        
        response = self.llm.chat([
            {"role": "system", "content": "你是任务规划专家，输出严格的JSON格式。"},
            {"role": "user", "content": prompt}
        ], temperature=0.3)
        
        # 解析LLM输出
        plan_data = self._parse_response(response)
        
        # 构建ExecutionPlan
        plan = ExecutionPlan(goal=goal)
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
        }
        
        for task_data in plan_data.get("tasks", []):
            task = Task(
                id=task_data["id"],
                title=task_data.get("title", ""),
                description=task_data.get("description", ""),
                dependencies=task_data.get("dependencies", []),
                tool_name=task_data.get("tool_type"),
                priority=priority_map.get(
                    task_data.get("priority", "medium"), 
                    TaskPriority.MEDIUM
                ),
                max_retries=task_data.get("max_retries", 2),
            )
            plan.tasks.append(task)
        
        # 验证依赖引用的有效性
        self._validate_plan(plan)
        
        return plan
    
    def _parse_response(self, response: str) -> dict:
        """从LLM响应中提取JSON"""
        # 尝试从markdown代码块中提取
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            return json.loads(response[start:end])
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            return json.loads(response[start:end])
        return json.loads(response)
    
    def _validate_plan(self, plan: ExecutionPlan):
        """验证计划的依赖关系是否有效"""
        task_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise ValueError(
                        f"Task {task.id} 依赖不存在的任务 {dep_id}"
                    )
            # 检查循环依赖
            if self._has_cycle(plan, task.id):
                raise ValueError(
                    f"检测到循环依赖，涉及任务 {task.id}"
                )
    
    def _has_cycle(self, plan: ExecutionPlan, start_id: str) -> bool:
        """DFS检测循环依赖"""
        visited = set()
        stack = [start_id]
        while stack:
            current = stack.pop()
            if current in visited:
                return True
            visited.add(current)
            task = plan.get_task(current)
            if task:
                stack.extend(task.dependencies)
        return False
```

### 3.3 执行引擎（Execution Engine）

```python
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """任务执行引擎 - 按DAG顺序执行任务"""
    
    def __init__(self, llm: LLMClient, tool_registry: dict):
        """
        Args:
            llm: LLM客户端
            tool_registry: 工具注册表 {tool_name: tool_callable}
        """
        self.llm = llm
        self.tools = tool_registry
        self.execution_log: list[dict] = []
    
    async def execute_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """执行整个计划"""
        logger.info(f"开始执行计划: {plan.summary()}")
        
        while not plan.is_complete:
            # 1. 检查被阻塞的任务（依赖失败）
            blocked = plan.get_blocked_tasks()
            for task in blocked:
                task.status = TaskStatus.SKIPPED
                self._log(task, "skipped", "依赖任务失败")
            
            # 2. 获取可执行的任务
            ready_tasks = plan.get_ready_tasks()
            if not ready_tasks:
                if plan.is_complete:
                    break
                # 可能存在循环依赖或所有任务都被阻塞
                logger.error("没有可执行的任务，计划可能陷入死锁")
                break
            
            # 3. 按优先级排序，选择最高优先级的任务执行
            ready_tasks.sort(key=lambda t: t.priority.value)
            
            # 简单策略：每次执行最高优先级的一个任务
            # （可以扩展为并行执行多个无依赖的任务）
            task = ready_tasks[0]
            
            try:
                result = await self._execute_task(task, plan)
                task.result = result
                task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                task.completed_at = time.time()
                
                self._log(task, "completed" if result.success else "failed", 
                         result.output[:200])
                
            except Exception as e:
                logger.error(f"任务 {task.id} 执行异常: {e}")
                task.status = TaskStatus.FAILED
                task.result = TaskResult(success=False, output=str(e))
                task.completed_at = time.time()
                
                # 尝试重试
                if task.can_retry():
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    task.completed_at = None
                    logger.info(f"任务 {task.id} 将重试 ({task.retry_count}/{task.max_retries})")
        
        logger.info(f"计划执行完成: {plan.summary()}")
        return plan
    
    async def _execute_task(self, task: Task, plan: ExecutionPlan) -> TaskResult:
        """执行单个任务"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        
        logger.info(f"执行任务: [{task.id}] {task.title}")
        
        # 根据工具类型选择执行方式
        if task.tool_name and task.tool_name in self.tools:
            # 使用注册的工具
            tool = self.tools[task.tool_name]
            output = await self._call_tool(tool, task, plan)
        else:
            # 使用LLM直接执行
            output = await self._llm_execute(task, plan)
        
        return TaskResult(success=True, output=output)
    
    async def _llm_execute(self, task: Task, plan: ExecutionPlan) -> str:
        """使用LLM执行任务"""
        # 收集上下文：已完成任务的输出
        context_parts = []
        for dep_id in task.dependencies:
            dep_task = plan.get_task(dep_id)
            if dep_task and dep_task.result:
                context_parts.append(
                    f"[{dep_task.title}] 输出: {dep_task.result.output[:500]}"
                )
        
        context = "\n".join(context_parts) if context_parts else "无前置任务上下文"
        
        prompt = f"""请执行以下任务：

任务标题: {task.title}
任务描述: {task.description}

前置任务结果:
{context}

请直接输出执行结果，不需要额外解释。"""
        
        response = self.llm.chat([
            {"role": "system", "content": "你是一个高效的执行助手，直接完成任务并输出结果。"},
            {"role": "user", "content": prompt}
        ], temperature=0.5)
        
        return response
    
    async def _call_tool(self, tool, task: Task, plan: ExecutionPlan) -> str:
        """调用注册的工具"""
        # 准备工具输入
        tool_input = {
            "task_title": task.title,
            "task_description": task.description,
            **task.tool_args,
        }
        
        # 添加依赖任务的输出作为上下文
        for dep_id in task.dependencies:
            dep_task = plan.get_task(dep_id)
            if dep_task and dep_task.result:
                tool_input[f"dep_{dep_id}_output"] = dep_task.result.output
        
        # 调用工具
        if asyncio.iscoroutinefunction(tool):
            result = await tool(**tool_input)
        else:
            result = tool(**tool_input)
        
        return str(result)
    
    def _log(self, task: Task, event: str, detail: str = ""):
        """记录执行日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.id,
            "task_title": task.title,
            "event": event,
            "detail": detail,
        }
        self.execution_log.append(entry)
```

### 3.4 动态重规划器（Replanner）

```python
@dataclass
class ReplanningTrigger:
    """重规划触发条件"""
    task_failure: bool = True        # 任务失败
    output_anomaly: bool = True      # 输出异常
    resource_exhaustion: bool = True  # 资源耗尽
    user_feedback: bool = True       # 用户反馈


class DynamicReplanner:
    """动态重规划器 - 根据执行偏差调整计划"""
    
    ANALYZE_PROMPT = """分析当前执行计划的状态，判断是否需要重规划。

当前计划: {plan_summary}

已完成任务:
{completed_tasks}

失败任务:
{failed_tasks}

待执行任务:
{pending_tasks}

请分析：
1. 失败原因是什么？是否可以绕过？
2. 已完成任务的输出是否符合预期？
3. 是否需要调整剩余任务的策略？
4. 是否需要新增或删除任务？

输出JSON:
```json
{{
  "needs_replan": true/false,
  "reason": "分析原因",
  "actions": [
    {{
      "type": "retry/modify/add/remove/skip",
      "task_id": "受影响的任务ID",
      "description": "具体操作描述"
    }}
  ]
}}
```
"""
    
    def __init__(self, llm: LLMClient, trigger_config: ReplanningTrigger = None):
        self.llm = llm
        self.triggers = trigger_config or ReplanningTrigger()
    
    def should_replan(self, plan: ExecutionPlan) -> bool:
        """判断是否需要重规划"""
        if self.triggers.task_failure:
            failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
            if failed and not all(t.can_retry() for t in failed):
                return True  # 有任务失败且无法重试
        
        if self.triggers.resource_exhaustion:
            # 检查是否所有任务都卡住
            ready = plan.get_ready_tasks()
            blocked = plan.get_blocked_tasks()
            if not ready and blocked:
                return True
        
        return False
    
    def replan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """执行重规划"""
        logger.info(f"触发重规划: {plan.summary()}")
        
        # 准备分析上下文
        completed = []
        failed = []
        pending = []
        
        for task in plan.tasks:
            info = f"- [{task.id}] {task.title}: {task.status.value}"
            if task.result:
                info += f" → {task.result.output[:100]}"
            
            if task.status == TaskStatus.COMPLETED:
                completed.append(info)
            elif task.status == TaskStatus.FAILED:
                failed.append(info)
            else:
                pending.append(info)
        
        prompt = self.ANALYZE_PROMPT.format(
            plan_summary=plan.summary(),
            completed_tasks="\n".join(completed) or "无",
            failed_tasks="\n".join(failed) or "无",
            pending_tasks="\n".join(pending) or "无",
        )
        
        response = self.llm.chat([
            {"role": "system", "content": "你是任务规划专家，分析执行偏差并调整计划。"},
            {"role": "user", "content": prompt}
        ], temperature=0.3)
        
        # 解析重规划决策
        actions = self._parse_actions(response)
        
        # 应用操作
        for action in actions:
            self._apply_action(plan, action)
        
        return plan
    
    def _parse_actions(self, response: str) -> list[dict]:
        """解析重规划操作"""
        try:
            data = json.loads(response) if response.strip().startswith('{') else \
                   json.loads(response[response.index('{'):response.rindex('}') + 1])
            return data.get("actions", [])
        except (json.JSONDecodeError, ValueError):
            logger.warning("无法解析重规划响应，跳过")
            return []
    
    def _apply_action(self, plan: ExecutionPlan, action: dict):
        """应用单个重规划操作"""
        action_type = action.get("type")
        task_id = action.get("task_id")
        
        if action_type == "skip" and task_id:
            task = plan.get_task(task_id)
            if task:
                task.status = TaskStatus.SKIPPED
                logger.info(f"跳过任务 {task_id}: {action.get('description', '')}")
        
        elif action_type == "modify" and task_id:
            task = plan.get_task(task_id)
            if task:
                task.description += f"\n[重规划调整] {action.get('description', '')}"
                logger.info(f"修改任务 {task_id}")
        
        elif action_type == "retry" and task_id:
            task = plan.get_task(task_id)
            if task:
                task.status = TaskStatus.PENDING
                task.retry_count = 0
                task.result = None
                logger.info(f"重试任务 {task_id}")
        
        elif action_type == "add":
            new_task = Task(
                title=action.get("description", "新增任务"),
                description=action.get("description", ""),
                dependencies=[task_id] if task_id else [],
            )
            plan.tasks.append(new_task)
            logger.info(f"新增任务 {new_task.id}")
```

### 3.5 完整的规划Agent

```python
class PlanningAgent:
    """完整的规划型Agent"""
    
    def __init__(self, llm: LLMClient, tools: dict = None):
        self.llm = llm
        self.decomposer = TaskDecomposer(llm)
        self.engine = ExecutionEngine(llm, tools or {})
        self.replanner = DynamicReplanner(llm)
        self.plans: list[ExecutionPlan] = []
    
    async def run(self, goal: str, max_replan_cycles: int = 3) -> str:
        """执行目标，支持动态重规划
        
        Args:
            goal: 高层目标
            max_replan_cycles: 最大重规划次数
            
        Returns:
            最终执行结果摘要
        """
        # 第1步：任务分解
        logger.info(f"目标: {goal}")
        plan = self.decomposer.decompose(goal)
        self.plans.append(plan)
        
        logger.info(f"初始计划: {len(plan.tasks)} 个任务")
        for t in plan.tasks:
            logger.info(f"  [{t.id}] {t.title} (依赖: {t.dependencies})")
        
        # 第2步：执行 + 重规划循环
        replan_count = 0
        while not plan.is_complete and replan_count < max_replan_cycles:
            # 执行
            plan = await self.engine.execute_plan(plan)
            
            # 检查是否需要重规划
            if not plan.is_complete and self.replanner.should_replan(plan):
                replan_count += 1
                logger.info(f"第 {replan_count} 次重规划")
                plan = self.replanner.replan(plan)
        
        # 第3步：生成结果摘要
        return self._generate_summary(plan)
    
    def _generate_summary(self, plan: ExecutionPlan) -> str:
        """生成执行结果摘要"""
        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        skipped = [t for t in plan.tasks if t.status == TaskStatus.SKIPPED]
        
        parts = [f"## 执行结果: {plan.goal}\n"]
        parts.append(f"**完成率**: {plan.progress:.0%} ({len(completed)}/{len(plan.tasks)})\n")
        
        if completed:
            parts.append("### ✅ 已完成")
            for t in completed:
                duration = f" ({t.duration:.1f}s)" if t.duration else ""
                parts.append(f"- **{t.title}**{duration}")
                if t.result:
                    parts.append(f"  结果: {t.result.output[:200]}")
        
        if failed:
            parts.append("\n### ❌ 失败")
            for t in failed:
                parts.append(f"- **{t.title}**: {t.result.output[:100] if t.result else '未知错误'}")
        
        if skipped:
            parts.append("\n### ⏭️ 跳过")
            for t in skipped:
                parts.append(f"- **{t.title}**")
        
        return "\n".join(parts)
```

### 3.6 使用示例

```python
# 简单的LLM客户端实现
class SimpleLLM:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
    
    def chat(self, messages, temperature=0.7):
        # 实际项目中调用OpenAI/Claude API
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content


# 使用示例
async def main():
    llm = SimpleLLM(api_key="your-api-key")
    
    # 注册自定义工具
    tools = {
        "search": search_web,
        "analyze": analyze_text,
        "generate": generate_content,
    }
    
    agent = PlanningAgent(llm, tools)
    
    result = await agent.run(
        "分析Python异步编程的最佳实践，写一篇技术博客，包含代码示例"
    )
    print(result)


# 运行
# asyncio.run(main())
```

## 4. 生产优化：规划系统的工程实践

### 4.1 Token预算管理

规划型Agent最大的成本是Token消耗——规划本身需要LLM调用，执行每个任务也需要LLM调用。一个10步的计划可能消耗数十次LLM调用。

**预算控制策略**：

```python
class TokenBudget:
    """Token预算管理器"""
    
    def __init__(self, total_budget: int = 100_000):
        self.total = total_budget
        self.planning_used = 0
        self.execution_used = 0
        self.replanning_used = 0
    
    @property
    def remaining(self) -> int:
        return self.total - self.planning_used - self.execution_used - self.replanning_used
    
    def can_afford(self, estimated_tokens: int) -> bool:
        return self.remaining >= estimated_tokens
    
    def record_planning(self, tokens: int):
        self.planning_used += tokens
    
    def record_execution(self, task_id: str, tokens: int):
        self.execution_used += tokens
    
    def record_replanning(self, tokens: int):
        self.replanning_used += tokens
    
    def summary(self) -> str:
        return (
            f"预算: {self.total} tokens | "
            f"规划: {self.planning_used} | "
            f"执行: {self.execution_used} | "
            f"重规划: {self.replanning_used} | "
            f"剩余: {self.remaining}"
        )
```

**关键优化技巧**：

| 技巧 | 方法 | 节省比例 |
|------|------|----------|
| **小模型做规划** | 用GPT-4o-mini做分解，GPT-4o做复杂执行 | 40-60% |
| **缓存分解结果** | 相似目标复用历史计划模板 | 20-30% |
| **截断上下文** | 任务输出超过500字时截断 | 15-25% |
| **减少重规划** | 提高首次规划质量，降低重规划频率 | 10-20% |
| **并行执行** | 无依赖的任务同时执行，减少轮次 | 时间-30% |

### 4.2 规划质量保障

**规划质量检查清单**：

```
✅ 所有任务ID唯一且依赖引用有效
✅ 无循环依赖（DAG验证）
✅ 每个任务有明确的完成标准
✅ 任务粒度适中（不过粗不过细）
✅ 关键路径上的任务有重试策略
✅ 总任务数在合理范围（3-8个）
```

**自动质量评估**：

```python
def assess_plan_quality(plan: ExecutionPlan) -> dict:
    """评估计划质量"""
    issues = []
    score = 100
    
    # 检查任务数量
    if len(plan.tasks) < 2:
        issues.append("任务数过少，可能分解不够")
        score -= 10
    elif len(plan.tasks) > 10:
        issues.append("任务数过多，考虑合并或分阶段")
        score -= 15
    
    # 检查关键路径
    critical_path = find_critical_path(plan)
    if len(critical_path) > 5:
        issues.append(f"关键路径过长({len(critical_path)}步)，考虑并行化")
        score -= 10
    
    # 检查重试覆盖
    critical_tasks = [t for t in critical_path if isinstance(t, Task)]
    no_retry = [t for t in critical_tasks if t.max_retries == 0]
    if no_retry:
        issues.append(f"关键路径上有{len(no_retry)}个任务无重试策略")
        score -= 5 * len(no_retry)
    
    # 检查优先级一致性
    priorities = [t.priority for t in plan.tasks]
    if len(set(priorities)) == 1:
        issues.append("所有任务优先级相同，建议区分关键/次要")
        score -= 5
    
    return {"score": max(0, score), "issues": issues}


def find_critical_path(plan: ExecutionPlan) -> list:
    """找到关键路径（最长路径）"""
    # 拓扑排序 + 动态规划
    in_degree = {t.id: 0 for t in plan.tasks}
    for t in plan.tasks:
        for dep in t.dependencies:
            if dep in in_degree:
                in_degree[t.id] += 1
    
    # BFS找最长路径
    dist = {t.id: 0 for t in plan.tasks}
    predecessor = {t.id: None for t in plan.tasks}
    
    queue = [t.id for t in plan.tasks if in_degree[t.id] == 0]
    while queue:
        current_id = queue.pop(0)
        current_task = plan.get_task(current_id)
        
        for task in plan.tasks:
            if current_id in task.dependencies:
                in_degree[task.id] -= 1
                new_dist = dist[current_id] + 1
                if new_dist > dist[task.id]:
                    dist[task.id] = new_dist
                    predecessor[task.id] = current_id
                if in_degree[task.id] == 0:
                    queue.append(task.id)
    
    # 回溯最长路径
    if not dist:
        return []
    end_id = max(dist, key=dist.get)
    path = []
    current = end_id
    while current is not None:
        path.append(plan.get_task(current))
        current = predecessor[current]
    return list(reversed(path))
```

### 4.3 失败恢复策略

生产环境中，规划系统需要处理各种失败场景：

| 失败类型 | 恢复策略 | 实现方式 |
|----------|----------|----------|
| **单任务失败** | 重试 + 降级 | 自动重试N次，失败后用简化版本替代 |
| **依赖链断裂** | 跳过 + 绕过 | 标记后续任务为可选，用默认值填充 |
| **资源耗尽** | 截断 + 合并 | 将多个小任务合并，减少Token消耗 |
| **LLM超时** | 切换模型 | 从GPT-4o降级到GPT-4o-mini |
| **工具不可用** | 替代方案 | 用其他工具或LLM直接完成 |
| **计划完全失败** | 重规划或放弃 | 触发Replanner重新分析，或返回部分结果 |

### 4.4 可观测性与调试

```python
class PlanTracer:
    """计划执行追踪器 - 支持可视化和调试"""
    
    def __init__(self, plan: ExecutionPlan):
        self.plan = plan
        self.timeline: list[dict] = []
    
    def record(self, event: str, task_id: str, detail: str = ""):
        self.timeline.append({
            "time": datetime.now().isoformat(),
            "event": event,
            "task_id": task_id,
            "detail": detail,
        })
    
    def to_gantt_chart(self) -> str:
        """生成文本甘特图"""
        lines = [f"Plan: {self.plan.goal}", "=" * 60]
        
        for task in self.plan.tasks:
            status_icon = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.PENDING: "⏳",
                TaskStatus.SKIPPED: "⏭️",
            }.get(task.status, "❓")
            
            # 进度条
            bar_len = 30
            if task.status == TaskStatus.COMPLETED:
                bar = "█" * bar_len
            elif task.status == TaskStatus.IN_PROGRESS:
                filled = int(bar_len * 0.5)
                bar = "█" * filled + "░" * (bar_len - filled)
            else:
                bar = "░" * bar_len
            
            duration = f"{task.duration:.1f}s" if task.duration else "pending"
            lines.append(
                f"{status_icon} [{task.id}] {task.title[:20]:<20} "
                f"|{bar}| {duration}"
            )
        
        lines.append("=" * 60)
        lines.append(f"进度: {self.plan.progress:.0%}")
        
        return "\n".join(lines)
    
    def to_mermaid(self) -> str:
        """生成Mermaid流程图"""
        lines = ["graph TD"]
        
        for task in self.plan.tasks:
            shape = "[]" if task.status == TaskStatus.COMPLETED else "{}"
            label = f"{task.title[:15]}|{task.status.value}|"
            
            if task.status == TaskStatus.COMPLETED:
                lines.append(f"  {task.id}[\"{label}\"]")
            elif task.status == TaskStatus.FAILED:
                lines.append(f"  {task.id}{{\"{label}\"}}")
            else:
                lines.append(f"  {task.id}(\"{label}\")")
            
            for dep_id in task.dependencies:
                lines.append(f"  {dep_id} --> {task.id}")
        
        return "\n".join(lines)
```

## 5. 面试深度：高频考点与架构决策

### 5.1 核心面试题

**Q1: ReAct和Plan-and-Execute的核心区别是什么？各自的适用场景？**

**答**：核心区别在于**推理粒度和全局视野**。

- **ReAct**：每一步都重新思考下一步该做什么。优势是灵活性高，每步都能根据最新观察调整方向。劣势是缺乏全局视野，容易陷入局部最优，且步骤之间缺乏协调。
- **Plan-and-Execute**：先生成完整计划，然后按计划执行。优势是全局视野好，任务间协调性强，可预测性高。劣势是计划可能僵化，执行过程中遇到意外时调整成本高。

**适用场景**：
- ReAct适合：探索性任务（搜索、调试）、步骤少于5步的简单任务
- Plan-and-Execute适合：结构化任务（报告生成、数据处理流水线）、步骤多于5步的复杂任务

**最佳实践**：生产系统通常采用**混合模式**——高层用Plan-and-Execute保证方向，底层子任务用ReAct保证灵活性。

**Q2: 如何处理规划型Agent中的循环依赖？**

**答**：循环依赖检测是规划系统的基础能力。

1. **构建阶段检测**：用DFS/BFS在任务分解完成后立即检测。如果发现循环，向LLM报告并要求重新分解。
2. **运行时检测**：执行引擎中设置超时——如果一个任务等待超过N轮仍未完成（可能是循环），触发告警。
3. **架构预防**：在Prompt中明确要求"任务之间只能有单向依赖"，并在输出格式中强制要求DAG结构。

```python
# 拓扑排序检测循环
def detect_cycle(tasks: list[Task]) -> bool:
    in_degree = {t.id: len(t.dependencies) for t in tasks}
    queue = [t.id for t in tasks if in_degree[t.id] == 0]
    count = 0
    while queue:
        node = queue.pop(0)
        count += 1
        for t in tasks:
            if node in t.dependencies:
                in_degree[t.id] -= 1
                if in_degree[t.id] == 0:
                    queue.append(t.id)
    return count != len(tasks)  # 有节点未被访问 = 存在循环
```

**Q3: 动态重规划的触发时机和策略？**

**答**：重规划是规划型Agent的核心能力，但过度重规划会浪费资源。关键是在"僵化执行"和"过度调整"之间找到平衡。

**触发时机**：
1. **任务失败且无法重试**：必须重新规划绕过失败任务
2. **输出偏差超过阈值**：例如预期得到数据分析结果，实际得到错误信息
3. **资源剩余不足**：Token预算消耗过快，需要简化剩余任务
4. **外部信号**：用户主动要求调整方向

**策略选择**：
- **局部重规划**：只调整失败任务及其下游任务（推荐，影响最小）
- **全局重规划**：基于所有已完成结果重新生成计划（仅在严重偏差时使用）
- **渐进式调整**：修改任务描述但不改变任务结构（最轻量）

**Q4: 如何评估规划质量？有哪些量化指标？**

**答**：

| 指标 | 计算方式 | 优秀标准 |
|------|----------|----------|
| **完成率** | 成功任务数 / 总任务数 | ≥ 90% |
| **首次通过率** | 无需重规划就完成的比例 | ≥ 70% |
| **关键路径效率** | 理论最短时间 / 实际执行时间 | ≥ 80% |
| **Token效率** | 有用输出Token / 总消耗Token | ≥ 50% |
| **平均重试次数** | 总重试次数 / 任务总数 | ≤ 0.3 |
| **端到端延迟** | 目标输入到结果输出的时间 | 视任务而定 |

### 5.2 架构选型决策

**场景：设计一个自动化代码审查Agent，如何选择规划策略？**

**分析**：
- 代码审查是**结构化任务**——有明确的步骤（读代码→分析→生成报告）
- 步骤之间有**强依赖**——必须先读代码才能分析
- 审查结果需要**一致性**——不同文件的审查标准要统一
- 可能遇到**意外情况**——代码格式错误、缺少测试文件等

**推荐方案**：Plan-and-Execute + 局部ReAct

```
顶层规划：
1. 读取PR信息和变更文件列表
2. 按文件类型分组（Python/JS/配置文件等）
3. 对每组文件执行审查（可并行）
4. 汇总审查结果，生成报告
5. 发送通知

每个子任务内部用ReAct处理细节：
- 遇到语法错误 → 跳过该文件
- 遇到不熟悉的框架 → 查文档后继续
- 审查发现严重问题 → 立即标记并继续
```

### 5.3 开放性设计问题

**如何让Agent学会"不规划"？**

有些任务不需要规划——直接执行更高效。让Agent学会判断"这个任务需要规划吗？"是一个重要的元能力。

**判断依据**：
- 任务复杂度（步骤数、依赖关系）
- 不确定性程度（环境是否可预测）
- 代价（规划本身消耗的资源 vs 直接执行的失败风险）

**实现方式**：在规划层之前加一个**元规划器**（Meta-Planner），用轻量级分类器判断是否需要启动完整规划流程。简单任务直接交给ReAct执行，复杂任务才启动Plan-and-Execute流程。这样可以在整体上获得"简单任务快、复杂任务稳"的效果。

## 参考资源

1. **Plan-and-Solve Prompting** - Wang et al., 2023 - 任务分解与规划的Prompt工程方法
2. **Tree of Thought** - Yao et al., 2023 - 树形推理路径探索
3. **Language Agent Tree Search (LATS)** - Zhou et al., 2023 - 结合MCTS的Agent规划
4. **LLMCompiler** - Kim et al., 2023 - LLM任务的并行执行框架
5. **AutoGPT / BabyAGI** - 开源Agent规划实现参考
6. **DSPy** - Khattab et al., 2023 - 声明式Agent编程框架
