---
title: "Agent技能编排引擎：从ReAct循环到复杂工作流的自动调度"
description: "深入解析Agent技能编排的核心模式，覆盖ReAct执行循环、Plan-and-Execute规划、动态技能路由、组合模式与生产级调度引擎实现"
date: 2026-05-31
author: "RiceBall-15"
category: "agent"
subCategory: "agent-skill"
tags: ["Agent技能", "技能编排", "ReAct", "工作流调度", "Plan-and-Execute"]
draft: false
---

# Agent技能编排引擎：从ReAct循环到复杂工作流的自动调度

## 核心问题：Agent如何决定调用哪些技能、以什么顺序执行？

当Agent拥有数十个技能（工具）时，真正的挑战不在于"能不能调用"，而在于**"什么时候调用、调用哪个、调用几个、按什么顺序"**。这就是技能编排（Skill Orchestration）的核心命题。

一个简单的单步工具调用（如"搜索天气"）不需要编排。但当任务变成"帮我规划一次日本旅行，比较三个城市的机票价格，生成行程表并发邮件"，Agent需要：

1. **任务分解**：将复杂目标拆解为可执行的子任务序列
2. **技能路由**：为每个子任务选择最合适的技能
3. **依赖管理**：识别子任务间的前后依赖关系
4. **并行调度**：无依赖的子任务并行执行以提升效率
5. **错误恢复**：某个技能失败时的降级和重试策略
6. **结果聚合**：将多个技能的输出合并为最终答案

本文将系统讲解Agent技能编排的完整技术栈，从最基础的ReAct循环到生产级的编排引擎。

---

## 一、基础模式：ReAct执行循环

### 1.1 ReAct的核心思想

ReAct（Reasoning + Acting）是Agent技能编排的基石模式。其核心是一个**观察-思考-行动**的循环：

```
while 任务未完成:
    观察(Observation) → 当前状态和上下文
    思考(Thought) → 分析下一步该做什么
    行动(Action) → 调用一个技能
    获取结果 → 技能返回的输出
    判断 → 任务是否完成？
```

### 1.2 ReAct的实现

```python
class ReActAgent:
    """基础ReAct Agent实现"""
    
    def __init__(self, llm, tools: list[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = 10
    
    def run(self, task: str) -> str:
        """执行任务"""
        history = [{"role": "user", "content": task}]
        
        for step in range(self.max_steps):
            # 1. LLM决定下一步行动
            response = self.llm.chat(history)
            
            # 2. 检查是否有工具调用
            if not response.tool_calls:
                return response.content  # 最终回答
            
            # 3. 执行工具调用
            for tool_call in response.tool_calls:
                tool = self.tools[tool_call.name]
                result = tool.execute(**tool_call.arguments)
                
                # 4. 将结果加入历史
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        return "达到最大步数限制，任务未完成"
```

### 1.3 ReAct的局限性

| 问题 | 表现 | 影响 |
|------|------|------|
| **线性执行** | 每次只能调用一个工具 | 无法并行，效率低 |
| **无规划** | 每步决策独立 | 可能偏离目标 |
| **上下文膨胀** | 历史越来越长 | Token消耗剧增 |
| **单点失败** | 一个工具失败全链路中断 | 鲁棒性差 |

当任务复杂度超过3-5步时，纯ReAct模式的效率和可靠性会显著下降。这就是为什么需要更高级的编排模式。

---

## 二、规划模式：Plan-and-Execute

### 2.1 核心思想

Plan-and-Execute将编排分为两个阶段：

```
阶段一：规划（Plan）
    输入：复杂任务
    输出：结构化的执行计划（步骤列表 + 依赖关系）

阶段二：执行（Execute）
    按计划逐步执行，每步可动态调整计划
```

### 2.2 规划器的实现

```python
from dataclasses import dataclass, field
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PlanStep:
    """执行计划中的一个步骤"""
    id: str
    description: str
    tool_name: str | None = None
    dependencies: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    retry_count: int = 0
    max_retries: int = 2

@dataclass
class ExecutionPlan:
    """完整的执行计划"""
    task: str
    steps: list[PlanStep]
    context: dict = field(default_factory=dict)
    
    def get_ready_steps(self) -> list[PlanStep]:
        """获取所有依赖已满足、可立即执行的步骤"""
        completed_ids = {
            s.id for s in self.steps 
            if s.status == StepStatus.COMPLETED
        }
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed_ids for dep in s.dependencies)
        ]
    
    def is_complete(self) -> bool:
        """检查计划是否执行完毕"""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )
    
    def has_failures(self) -> bool:
        """检查是否有失败步骤"""
        return any(s.status == StepStatus.FAILED for s in self.steps)


class Planner:
    """LLM驱动的任务规划器"""
    
    def __init__(self, llm, tools: list[Tool]):
        self.llm = llm
        self.tools = tools
    
    def create_plan(self, task: str) -> ExecutionPlan:
        """将复杂任务分解为执行计划"""
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" 
            for t in self.tools
        )
        
        prompt = f"""你是一个任务规划专家。请将以下任务分解为具体的执行步骤。

可用工具：
{tool_descriptions}

任务：{task}

请以JSON格式输出执行计划，包含：
1. steps: 步骤列表，每个步骤包含 id, description, tool_name, dependencies
2. 确保依赖关系正确（被依赖的步骤必须先完成）
3. 无依赖关系的步骤可以并行执行

输出格式：
{{
    "steps": [
        {{
            "id": "step_1",
            "description": "步骤描述",
            "tool_name": "工具名称（如果不需要工具则为null）",
            "dependencies": []
        }}
    ]
}}"""
        
        response = self.llm.chat([{"role": "user", "content": prompt}])
        plan_data = json.loads(response.content)
        
        steps = [
            PlanStep(
                id=s["id"],
                description=s["description"],
                tool_name=s.get("tool_name"),
                dependencies=s.get("dependencies", [])
            )
            for s in plan_data["steps"]
        ]
        
        return ExecutionPlan(task=task, steps=steps)
```

### 2.3 执行引擎

```python
class PlanAndExecuteEngine:
    """Plan-and-Execute执行引擎"""
    
    def __init__(self, llm, tools: list[Tool]):
        self.llm = llm
        self.planner = Planner(llm, tools)
        self.tools = {t.name: t for t in tools}
    
    async def run(self, task: str) -> str:
        # 阶段一：生成计划
        plan = self.planner.create_plan(task)
        logger.info(f"生成执行计划：{len(plan.steps)}个步骤")
        
        # 阶段二：执行计划
        while not plan.is_complete():
            ready = plan.get_ready_steps()
            
            if not ready:
                if plan.has_failures():
                    return self._handle_failure(plan)
                break
            
            # 并行执行无依赖的步骤
            if len(ready) > 1:
                results = await asyncio.gather(*[
                    self._execute_step(step, plan) 
                    for step in ready
                ])
            else:
                results = [await self._execute_step(ready[0], plan)]
            
            # 更新计划上下文
            for step, result in zip(ready, results):
                plan.context[step.id] = step.result
        
        # 汇总结果
        return self._summarize(plan)
    
    async def _execute_step(self, step: PlanStep, plan: ExecutionPlan) -> str:
        """执行单个步骤"""
        step.status = StepStatus.RUNNING
        
        try:
            if step.tool_name and step.tool_name in self.tools:
                # 从上下文中提取该步骤需要的输入
                input_data = self._prepare_input(step, plan)
                result = self.tools[step.tool_name].execute(**input_data)
                step.result = result
            else:
                # 无工具步骤：用LLM直接推理
                result = self._llm_reason(step, plan)
                step.result = result
            
            step.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            step.retry_count += 1
            if step.retry_count < step.max_retries:
                step.status = StepStatus.PENDING  # 重试
                logger.warning(f"步骤{step.id}失败，重试({step.retry_count}/{step.max_retries})")
            else:
                step.status = StepStatus.FAILED
                logger.error(f"步骤{step.id}永久失败: {e}")
            return str(e)
    
    def _prepare_input(self, step: PlanStep, plan: ExecutionPlan) -> dict:
        """根据依赖步骤的结果准备输入"""
        input_data = {}
        for dep_id in step.dependencies:
            dep_step = next(s for s in plan.steps if s.id == dep_id)
            if dep_step.result:
                input_data[dep_id] = dep_step.result
        return input_data
```

---

## 三、动态技能路由

### 3.1 技能路由的核心问题

当Agent拥有大量技能时，LLM需要一种高效的方式来选择最合适的技能。直接将所有技能描述放入prompt会导致：

- **Token浪费**：50个工具描述可能占用数千Token
- **选择困难**：LLM难以从大量选项中精准选择
- **延迟增加**：更长的prompt意味着更慢的响应

### 3.2 两级路由架构

```
用户请求
    │
    ▼
┌─────────────────────────┐
│   技能分类路由器         │  ← 第一级：选择技能类别
│   (Category Router)     │
└────────┬────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ 搜索类 │ │ 生成类 │  ← 第二级：在类别内选择具体技能
│ 技能池 │ │ 技能池 │
└────────┘ └────────┘
```

### 3.3 实现：向量相似度路由

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SkillRouter:
    """基于向量相似度的技能路由"""
    
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 预计算所有技能描述的embedding
        descriptions = [f"{t.name}: {t.description}" for t in tools]
        self.tool_embeddings = self.encoder.encode(descriptions)
        self.tool_names = [t.name for t in tools]
    
    def route(self, query: str, top_k: int = 3) -> list[str]:
        """根据用户请求选择最相关的技能"""
        query_embedding = self.encoder.encode([query])[0]
        
        # 计算余弦相似度
        similarities = np.dot(self.tool_embeddings, query_embedding) / (
            np.linalg.norm(self.tool_embeddings, axis=1) 
            * np.linalg.norm(query_embedding)
        )
        
        # 返回top-k最相关的技能
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.tool_names[i] for i in top_indices]


class HierarchicalRouter:
    """两级层次化技能路由"""
    
    def __init__(self, categories: dict[str, list[Tool]]):
        """
        categories: {
            "search": [SearchTool, WebSearchTool, ...],
            "generation": [CodeGenTool, ImageGenTool, ...],
            "analysis": [DataAnalysisTool, ChartTool, ...],
        }
        """
        self.categories = categories
        
        # 第一级：类别路由
        self.category_router = SkillRouter([
            Tool(name=cat, description=f"{cat}类别工具")
            for cat in categories
        ])
        
        # 第二级：每个类别的技能路由
        self.tool_routers = {
            cat: SkillRouter(tools)
            for cat, tools in categories.items()
        }
    
    def route(self, query: str) -> dict:
        """两级路由"""
        # 第一级：选择类别
        categories = self.category_router.route(query, top_k=2)
        
        # 第二级：在每个类别内选择技能
        candidates = []
        for cat in categories:
            tools = self.tool_routers[cat].route(query, top_k=3)
            candidates.extend(tools)
        
        return {
            "selected_categories": categories,
            "selected_tools": candidates
        }
```

### 3.4 路由策略对比

| 策略 | 实现复杂度 | 选择精度 | 适用场景 |
|------|-----------|---------|---------|
| **全量描述** | 低 | 高 | 技能<15个 |
| **向量相似度** | 中 | 中高 | 技能15-100个 |
| **两级层次路由** | 高 | 高 | 技能>100个 |
| **LLM分类+路由** | 中 | 高 | 需要语义理解的场景 |
| **规则+关键词** | 低 | 中 | 技能名称语义明确 |

---

## 四、技能组合模式

### 4.1 五种基本组合模式

#### 模式一：顺序执行（Sequential）

```
Task → Skill_A → Skill_B → Skill_C → Result
```

最简单的模式，每个技能的输出作为下一个技能的输入。

```python
class SequentialComposer:
    def execute(self, skills: list[Skill], initial_input: str) -> str:
        current_input = initial_input
        for skill in skills:
            current_input = skill.execute(current_input)
        return current_input
```

**适用场景**：数据处理管道、多步转换

#### 模式二：并行扇出（Parallel Fan-out）

```
         ┌→ Skill_A ─┐
Task ──→ ├→ Skill_B ─┤──→ Aggregator → Result
         └→ Skill_C ─┘
```

多个无依赖的技能并行执行，最后聚合结果。

```python
class ParallelComposer:
    async def execute(self, skills: list[Skill], input_data: str) -> str:
        # 并行执行所有技能
        results = await asyncio.gather(*[
            skill.execute(input_data) 
            for skill in skills
        ])
        
        # 聚合结果
        return self.aggregate(results)
    
    def aggregate(self, results: list[str]) -> str:
        """合并多个技能的输出"""
        return "\n\n---\n\n".join(results)
```

**适用场景**：多源信息收集、对比分析

#### 模式三：条件分支（Conditional Branching）

```
           ┌─ if A ─→ Skill_A ─┐
Task ──→ Router ─┤              ├──→ Result
           └─ if B ─→ Skill_B ─┘
```

根据条件选择不同的技能路径。

```python
class ConditionalComposer:
    def __init__(self, router: SkillRouter):
        self.router = router
    
    def execute(self, task: str, context: dict) -> str:
        # 路由器决定走哪条路径
        selected = self.router.route(task)
        
        if selected["category"] == "search":
            return self._search_path(task)
        elif selected["category"] == "analysis":
            return self._analysis_path(task)
        else:
            return self._default_path(task)
```

**适用场景**：意图识别后的差异化处理

#### 模式四：循环执行（Loop/Iterative）

```
┌──────────────────────────┐
│ while 条件不满足:         │
│   Skill → 检查结果       │
│   不满足 → 调整参数重试   │
└──────────────────────────┘
```

反复执行技能直到满足条件或达到最大次数。

```python
class LoopComposer:
    def execute(self, skill: Skill, task: str, 
                max_iterations: int = 5) -> str:
        result = None
        for i in range(max_iterations):
            result = skill.execute(task, previous=result)
            
            if self._is_satisfactory(result):
                return result
            
            # 根据结果调整输入
            task = self._refine_task(task, result)
        
        return result  # 返回最后一次结果
```

**适用场景**：代码生成+测试修复、迭代优化

#### 模式五：嵌套组合（Nested Composition）

```
Task ──→ Sequential [
    Parallel [ Skill_A, Skill_B ]
    → Conditional {
        if success: Skill_C
        if failure: Skill_D
    }
    → Loop [ Skill_E ]
]
```

将多种模式组合成复杂的执行流程。

### 4.2 组合模式选择指南

```
                    ┌──────────────────────┐
                    │ 任务是否有多个独立子任务？ │
                    └──────────┬───────────┘
                         Yes ↙     ↘ No
                    ┌──────────┐  ┌──────────────┐
                    │ 并行扇出  │  │ 是否需要迭代？  │
                    └──────────┘  └──────┬───────┘
                                    Yes ↙     ↘ No
                               ┌─────────┐  ┌──────────┐
                               │ 循环执行  │  │ 是否有条件？│
                               └─────────┘  └────┬─────┘
                                            Yes ↙     ↘ No
                                        ┌──────────┐ ┌────────┐
                                        │ 条件分支  │ │ 顺序执行 │
                                        └──────────┘ └────────┘
```

---

## 五、生产级编排引擎实现

### 5.1 架构设计

```
┌─────────────────────────────────────────────────────┐
│                   Skill Orchestrator                 │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Plan Engine │  │ Route Engine │  │ Exec Engine│ │
│  │ (规划引擎)   │  │ (路由引擎)   │  │ (执行引擎) │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                │                 │        │
│  ┌──────▼────────────────▼─────────────────▼──────┐ │
│  │              Execution Context                 │ │
│  │  (执行上下文：状态、历史、中间结果、配置)          │ │
│  └────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Skill Pool  │  │ Retry Logic  │  │ Monitoring │ │
│  │ (技能池)     │  │ (重试策略)   │  │ (监控告警)  │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 5.2 完整实现

```python
import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

@dataclass
class SkillExecutionMetrics:
    """技能执行指标"""
    skill_name: str
    start_time: float = 0
    end_time: float = 0
    success: bool = False
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class SkillOrchestrator:
    """生产级技能编排引擎"""
    
    def __init__(
        self,
        llm,
        skills: list[Skill],
        max_concurrent: int = 5,
        max_retries: int = 2,
        timeout_seconds: int = 30,
        enable_metrics: bool = True,
    ):
        self.llm = llm
        self.skills = {s.name: s for s in skills}
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_metrics = enable_metrics
        
        # 路由器
        self.router = SkillRouter(skills)
        
        # 并发控制信号量
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # 指标收集
        self.metrics: list[SkillExecutionMetrics] = []
    
    async def run(self, task: str) -> dict:
        """执行任务"""
        start_time = time.time()
        context = ExecutionContext(task=task)
        
        try:
            # 1. 规划
            plan = await self._plan(task)
            logger.info(f"计划生成完成：{len(plan.steps)}个步骤")
            
            # 2. 执行
            while not plan.is_complete():
                ready = plan.get_ready_steps()
                if not ready:
                    break
                
                # 并行执行就绪步骤
                await asyncio.gather(*[
                    self._execute_with_retry(step, context)
                    for step in ready
                ])
            
            # 3. 汇总结果
            result = await self._synthesize(plan, context)
            
            elapsed = time.time() - start_time
            return {
                "success": True,
                "result": result,
                "steps_executed": len(plan.steps),
                "elapsed_seconds": round(elapsed, 2),
                "metrics": self._get_metrics_summary() if self.enable_metrics else None,
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"任务执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
                "partial_results": context.results,
            }
    
    async def _execute_with_retry(
        self, step: PlanStep, context: ExecutionContext
    ):
        """带重试的步骤执行"""
        async with self._semaphore:  # 并发控制
            metrics = SkillExecutionMetrics(
                skill_name=step.tool_name or "llm_reason"
            )
            
            for attempt in range(self.max_retries + 1):
                metrics.start_time = time.time()
                metrics.retry_count = attempt
                
                try:
                    # 设置超时
                    result = await asyncio.wait_for(
                        self._execute_step(step, context),
                        timeout=self.timeout_seconds
                    )
                    
                    step.status = StepStatus.COMPLETED
                    step.result = result
                    context.results[step.id] = result
                    
                    metrics.end_time = time.time()
                    metrics.success = True
                    
                    if self.enable_metrics:
                        self.metrics.append(metrics)
                    
                    logger.info(
                        f"步骤{step.id}完成 "
                        f"({metrics.duration_ms:.0f}ms, "
                        f"重试{attempt}次)"
                    )
                    return
                    
                except asyncio.TimeoutError:
                    logger.warning(
                        f"步骤{step.id}超时 "
                        f"({self.timeout_seconds}s), "
                        f"尝试{attempt + 1}/{self.max_retries + 1}"
                    )
                except Exception as e:
                    logger.warning(
                        f"步骤{step.id}失败: {e}, "
                        f"尝试{attempt + 1}/{self.max_retries + 1}"
                    )
            
            # 所有重试用完
            step.status = StepStatus.FAILED
            metrics.end_time = time.time()
            if self.enable_metrics:
                self.metrics.append(metrics)
    
    async def _execute_step(
        self, step: PlanStep, context: ExecutionContext
    ) -> str:
        """执行单个步骤"""
        if step.tool_name and step.tool_name in self.skills:
            skill = self.skills[step.tool_name]
            input_data = self._prepare_step_input(step, context)
            
            # 技能执行
            if asyncio.iscoroutinefunction(skill.execute):
                return await skill.execute(**input_data)
            else:
                return skill.execute(**input_data)
        else:
            # 用LLM推理
            return await self._llm_step(step, context)
    
    def _prepare_step_input(
        self, step: PlanStep, context: ExecutionContext
    ) -> dict:
        """从上下文中准备步骤输入"""
        input_data = {}
        for dep_id in step.dependencies:
            if dep_id in context.results:
                input_data[f"from_{dep_id}"] = context.results[dep_id]
        input_data["task"] = step.description
        return input_data
    
    async def _llm_step(
        self, step: PlanStep, context: ExecutionContext
    ) -> str:
        """用LLM执行推理步骤"""
        context_str = json.dumps(context.results, ensure_ascii=False)
        prompt = f"""基于以下上下文，完成这个任务：

任务：{step.description}

已有结果：
{context_str}

请直接给出结果，不要解释过程。"""
        
        response = self.llm.chat([{"role": "user", "content": prompt}])
        return response.content
    
    def _get_metrics_summary(self) -> dict:
        """汇总执行指标"""
        if not self.metrics:
            return {}
        
        total_duration = sum(m.duration_ms for m in self.metrics)
        successful = sum(1 for m in self.metrics if m.success)
        total_retries = sum(m.retry_count for m in self.metrics)
        
        return {
            "total_steps": len(self.metrics),
            "successful": successful,
            "failed": len(self.metrics) - successful,
            "total_duration_ms": round(total_duration, 1),
            "avg_duration_ms": round(total_duration / len(self.metrics), 1),
            "total_retries": total_retries,
        }


@dataclass
class ExecutionContext:
    """执行上下文"""
    task: str
    results: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.3 技能定义规范

```python
from abc import ABC, abstractmethod

class Skill(ABC):
    """技能基类"""
    
    name: str
    description: str
    parameters: dict  # JSON Schema格式的参数定义
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行技能"""
        pass
    
    def to_schema(self) -> dict:
        """导出为OpenAI Function Calling格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class WebSearchSkill(Skill):
    """网页搜索技能"""
    
    name = "web_search"
    description = "搜索互联网获取最新信息"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 5
            }
        },
        "required": ["query"]
    }
    
    def execute(self, query: str, max_results: int = 5) -> str:
        # 实际搜索逻辑
        results = search_api(query, max_results)
        return json.dumps(results, ensure_ascii=False)
```

---

## 六、高级模式：自适应编排

### 6.1 动态计划调整

生产环境中，计划可能需要根据执行结果动态调整：

```python
class AdaptivePlanner:
    """自适应规划器 - 根据执行结果动态调整计划"""
    
    def __init__(self, llm, max_replan_count: int = 3):
        self.llm = llm
        self.max_replan_count = max_replan_count
        self.replan_count = 0
    
    def should_replan(self, plan: ExecutionPlan, step: PlanStep) -> bool:
        """判断是否需要重新规划"""
        if step.status != StepStatus.FAILED:
            return False
        
        if self.replan_count >= self.max_replan_count:
            return False
        
        # 失败步骤超过20%时重新规划
        failed_count = sum(
            1 for s in plan.steps 
            if s.status == StepStatus.FAILED
        )
        return failed_count / len(plan.steps) > 0.2
    
    def replan(self, plan: ExecutionPlan, failed_step: PlanStep) -> ExecutionPlan:
        """基于失败信息重新规划"""
        self.replan_count += 1
        
        completed_results = {
            s.id: s.result 
            for s in plan.steps 
            if s.status == StepStatus.COMPLETED
        }
        
        prompt = f"""原始任务：{plan.task}

已完成步骤的结果：
{json.dumps(completed_results, ensure_ascii=False)}

失败的步骤：{failed_step.description}
失败原因：{failed_step.result}

请重新规划剩余任务。已成功的结果保持不变，只需为失败的步骤提供替代方案。"""
        
        response = self.llm.chat([{"role": "user", "content": prompt}])
        new_plan_data = json.loads(response.content)
        
        # 合并新计划
        return self._merge_plans(plan, new_plan_data)
```

### 6.2 技能缓存与复用

```python
class SkillCache:
    """技能执行结果缓存"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: dict[str, tuple[float, str]] = {}
        self.ttl = ttl_seconds
    
    def _make_key(self, skill_name: str, args: dict) -> str:
        """生成缓存键"""
        args_str = json.dumps(args, sort_keys=True)
        return f"{skill_name}:{hashlib.md5(args_str.encode()).hexdigest()}"
    
    def get(self, skill_name: str, args: dict) -> str | None:
        """获取缓存结果"""
        key = self._make_key(skill_name, args)
        if key in self.cache:
            timestamp, result = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            del self.cache[key]
        return None
    
    def set(self, skill_name: str, args: dict, result: str):
        """设置缓存"""
        key = self._make_key(skill_name, args)
        self.cache[key] = (time.time(), result)
    
    def clear_expired(self):
        """清理过期缓存"""
        now = time.time()
        expired = [
            k for k, (ts, _) in self.cache.items()
            if now - ts > self.ttl
        ]
        for k in expired:
            del self.cache[k]
```

---

## 七、面试深度

### 7.1 高频面试题

**Q1：ReAct和Plan-and-Execute的核心区别是什么？各自的适用场景？**

**A**：ReAct是**反应式**的——每一步只看当前状态决定下一步，适合简单、步骤少（<5步）的任务。Plan-and-Execute是**前瞻式**的——先规划完整计划再执行，适合复杂任务（>5步），能识别并行机会和依赖关系。关键区别在于**信息利用效率**：ReAct只利用当前信息，Plan-and-Execute利用全局信息做决策。

**Q2：如何处理技能执行失败？有哪些容错策略？**

**A**：分三层容错：
1. **重试层**：指数退避重试（适用于临时故障，如网络超时）
2. **降级层**：用备选技能替代（如主搜索引擎不可用时切换备用）
3. **计划调整层**：重新规划剩余任务（如关键技能永久不可用时跳过或重组流程）

实际生产中通常组合使用：先重试2次 → 失败则降级 → 降级也失败则重新规划。

**Q3：大规模技能路由（>100个工具）如何保证选择效率和准确性？**

**A**：使用**两级层次路由**。第一级将技能分为5-10个类别，用向量相似度快速筛选出2-3个相关类别。第二级在类别内用更精细的模型（或LLM）从10-20个技能中选择1-3个。这样将O(N)的搜索降低为O(C + N/C)，C是类别数。同时维护技能描述的embedding索引，支持增量更新。

**Q4：并行执行技能时如何处理资源竞争？**

**A**：使用**并发控制**机制：
- **信号量**：限制同时执行的技能数，防止API限流
- **资源池**：为共享资源（如数据库连接）设置连接池
- **优先级队列**：关键路径上的技能优先执行
- **背压机制**：当队列满时暂停新任务提交

**Q5：如何评估技能编排的效果？关键指标有哪些？**

**A**：
- **成功率**：任务完成率（目标>95%）
- **效率**：平均执行时间、Token消耗量
- **质量**：结果准确率（可通过人工评估或自动验证）
- **鲁棒性**：失败恢复率、平均重试次数
- **成本**：每次任务的API调用费用

### 7.2 开放性问题

- **"如果你的Agent要同时处理1000个并发任务，编排引擎需要做什么改造？"**
  - 关键点：异步架构、任务队列（Redis/RabbitMQ）、Worker池、状态持久化、幂等性设计

- **"技能编排和传统工作流引擎（如Airflow）有什么异同？"**
  - 关键点：动态 vs 静态、LLM驱动 vs 规则驱动、容错策略差异、适用场景

---

## 八、生产优化

### 8.1 性能优化清单

| 优化项 | 效果 | 实现难度 |
|--------|------|---------|
| 技能结果缓存 | 减少重复调用30-50% | 低 |
| 并行执行 | 提升吞吐量2-5倍 | 中 |
| 流式输出 | 降低首字延迟 | 低 |
| 增量上下文 | 减少Token消耗 | 中 |
| 技能预热 | 减少冷启动延迟 | 低 |

### 8.2 监控与告警

```python
class OrchestratorMonitor:
    """编排引擎监控"""
    
    def __init__(self):
        self.alert_thresholds = {
            "failure_rate": 0.1,      # 失败率>10%告警
            "avg_duration_ms": 10000,  # 平均耗时>10s告警
            "max_retries": 3,          # 单步重试>3次告警
        }
    
    def check_health(self, metrics: list[SkillExecutionMetrics]) -> dict:
        """健康检查"""
        if not metrics:
            return {"status": "no_data"}
        
        total = len(metrics)
        failed = sum(1 for m in metrics if not m.success)
        failure_rate = failed / total
        avg_duration = sum(m.duration_ms for m in metrics) / total
        
        alerts = []
        if failure_rate > self.alert_thresholds["failure_rate"]:
            alerts.append(f"高失败率: {failure_rate:.1%}")
        if avg_duration > self.alert_thresholds["avg_duration_ms"]:
            alerts.append(f"高延迟: {avg_duration:.0f}ms")
        
        return {
            "status": "warning" if alerts else "healthy",
            "failure_rate": f"{failure_rate:.1%}",
            "avg_duration_ms": round(avg_duration, 1),
            "alerts": alerts,
        }
```

### 8.3 常见踩坑与解决方案

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| LLM幻觉导致调用不存在的技能 | 模型生成虚构的工具名 | 严格校验工具名白名单 |
| 并行执行时Token超限 | 多个技能同时返回大量数据 | 结果截断 + 懒加载 |
| 循环执行陷入死循环 | 重试条件判断不当 | 设置最大迭代次数 + 监控循环检测 |
| 上下文过长导致LLM推理变慢 | 历史结果不断累积 | 滑动窗口 + 结果摘要 |

---

## 九、总结

Agent技能编排是连接"拥有能力"和"有效使用能力"的桥梁。核心要点：

1. **ReAct是基础**：简单任务用ReAct足够，不要过度设计
2. **Plan-and-Execute是进阶**：复杂任务必须先规划，识别并行和依赖
3. **路由决定效率**：大规模技能池必须用层次化路由，避免全量prompt
4. **组合模式要灵活**：顺序、并行、条件、循环、嵌套——按任务特征选择
5. **生产级要关注**：并发控制、重试策略、缓存、监控、成本控制

从一个简单的ReAct循环出发，逐步引入规划、路由、并行、容错机制，最终构建出一个能在生产环境稳定运行的技能编排引擎。关键不在于用最复杂的模式，而在于**根据任务复杂度选择合适的编排策略**。
