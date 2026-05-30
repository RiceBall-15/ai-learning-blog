---
title: "Agentic AI系统设计模式：从ReAct到Plan-and-Execute的架构演进与实战"
description: "深度解析AI Agent系统的核心设计模式，覆盖ReAct、Plan-and-Execute、Multi-Agent等架构，附生产级实现方案与性能对比"
date: 2026-05-30
author: "RiceBall-15"
category: "featured"
subCategory: "deep-dive"
tags: ["AI Agent", "ReAct", "Plan-and-Execute", "Multi-Agent", "系统设计", "架构模式", "LLM应用"]
draft: false
---

## 一、引言：Agent系统的「第二次革命」

2024年，ReAct（Reasoning + Acting）范式让AI Agent从学术概念走向工程实践。2025年，工具调用和记忆系统让Agent具备了"手脚"和"记忆"。到了2026年，Agent系统正在经历**第二次革命**——从"单体Agent"走向"Agent系统"，从"简单工具调用"走向"自主规划与执行"。

但现实中的Agent系统远没有论文里那么美好。一个典型的生产级Agent系统面临的核心挑战是：

> **"Agent看起来很聪明，但在复杂任务上经常'迷路'——要么陷入循环，要么产出看似合理但实际错误的结果。"**

这不是模型的问题，而是**架构设计的问题**。本文将从架构师的视角，深度剖析Agent系统的核心设计模式，分析每种模式的适用场景和工程陷阱，并给出生产级实现方案。

## 二、Agent系统架构全景

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    2026年 Agent系统架构全景                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │                    编排层 (Orchestration)                    │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │        │
│  │  │ ReAct    │  │Plan&Exec │  │  LangGraph │  │  State   │  │        │
│  │  │ Agent    │  │ Agent    │  │  Workflow  │  │  Machine │  │        │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │        │
│  └───────┼──────────────┼──────────────┼──────────────┼────────┘        │
│          │              │              │              │                   │
│  ┌───────┼──────────────┼──────────────┼──────────────┼────────┐        │
│  │       ▼              ▼              ▼              ▼        │        │
│  │              认知层 (Cognition)                              │        │
│  │  ┌──────────────────────────────────────────────────────┐  │        │
│  │  │  推理引擎 │ 记忆系统 │ 规划器 │ 反思器 │ 自我纠正   │  │        │
│  │  └──────────────────────────────────────────────────────┘  │        │
│  │                     工具层 (Tools)                          │        │
│  │  ┌──────────────────────────────────────────────────────┐  │        │
│  │  │  API调用 │ 文件操作 │ 代码执行 │ 搜索 │ 数据库查询  │  │        │
│  │  └──────────────────────────────────────────────────────┘  │        │
│  │                     基础设施层 (Infra)                      │        │
│  │  ┌──────────────────────────────────────────────────────┐  │        │
│  │  │  LLM服务 │ 向量数据库 │ 消息队列 │ 可观测性 │ 安全  │  │        │
│  │  └──────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 三、核心设计模式深度解析

### 3.1 ReAct模式 — 最基础也最容易踩坑

**模式定义**：Reasoning（推理）+ Acting（行动）的交替执行。

```
ReAct 执行循环：

用户输入
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│  思考   │────▶│  行动   │────▶│  观察   │
│ (Think) │     │ (Act)   │     │(Observe)│
└─────────┘     └─────────┘     └─────────┘
    ▲                               │
    └───────────────────────────────┘
              (循环直到完成)

示例：
Thought: 用户想知道今天的天气，我需要调用天气API
Action: call_weather_api(city="北京")
Observation: 北京今天晴，25°C，东北风3级
Thought: 已经获取到天气信息，可以回答用户了
Answer: 北京今天天气晴朗，温度25°C，东北风3级
```

**生产级实现**：

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "对话历史"]
    current_step: int
    max_steps: int
    tool_calls: list[dict]
    error_count: int

def react_agent(state: AgentState) -> AgentState:
    """ReAct Agent核心循环"""
    messages = state["messages"]
    current_step = state["current_step"]
    max_steps = state["max_steps"]

    # 安全检查：防止无限循环
    if current_step >= max_steps:
        return {
            **state,
            "messages": messages + [
                AIMessage(content="抱歉，任务过于复杂，无法在限定步骤内完成。")
            ],
            "current_step": current_step + 1
        }

    if state.get("error_count", 0) >= 3:
        return {
            **state,
            "messages": messages + [
                AIMessage(content="连续出错过多，请简化您的请求。")
            ],
            "current_step": current_step + 1
        }

    # 调用LLM进行推理
    response = llm.invoke(messages)

    # 检查是否需要工具调用
    if response.tool_calls:
        # 执行工具调用
        tool_results = execute_tools(response.tool_calls)

        # 更新状态
        new_messages = messages + [response] + [
            ToolMessage(content=result, tool_call_id=tc["id"])
            for tc, result in zip(response.tool_calls, tool_results)
        ]

        return {
            **state,
            "messages": new_messages,
            "current_step": current_step + 1,
            "tool_calls": state["tool_calls"] + response.tool_calls
        }
    else:
        # 最终回答
        return {
            **state,
            "messages": messages + [response],
            "current_step": current_step + 1,
        }

def should_continue(state: AgentState) -> str:
    """判断是否继续循环"""
    last_message = state["messages"][-1]

    # 如果最后一条消息有工具调用，继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # 如果达到最大步骤，停止
    if state["current_step"] >= state["max_steps"]:
        return "end"

    # 如果没有工具调用，停止
    return "end"

# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", react_agent)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
    "continue": "agent",
    "end": END
})
react_app = graph.compile()
```

**ReAct模式的工程陷阱**：

| 陷阱 | 表现 | 解决方案 |
|------|------|----------|
| 无限循环 | Agent反复调用同一工具 | 设置max_steps + 去重检测 |
| 推理漂移 | Agent忘记初始目标 | 定期注入系统提示 + 目标检查 |
| 工具选择错误 | 调用不相关工具 | 工具描述优化 + Few-shot示例 |
| 上下文溢出 | 消息历史过长 | 滑动窗口 + 摘要压缩 |
| 幻觉行动 | 声称执行了但实际没有 | 工具结果强制验证 |

### 3.2 Plan-and-Execute模式 — 分而治之

**模式定义**：先规划完整方案，再逐步执行。适合复杂、多步骤任务。

```
Plan-and-Execute 架构：

用户输入
    │
    ▼
┌─────────────────────┐
│    Planner (规划器)   │
│  ┌─────────────────┐ │
│  │ 输入：用户目标   │ │
│  │ 输出：执行计划   │ │
│  └─────────────────┘ │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │  执行计划     │
    │  Step 1: ...  │
    │  Step 2: ...  │
    │  Step 3: ...  │
    └──────┬───────┘
           │
           ▼
┌─────────────────────┐
│   Executor (执行器)   │
│  ┌─────────────────┐ │
│  │ 逐步执行每个步骤 │ │
│  │ 收集执行结果     │ │
│  └─────────────────┘ │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Re-Planner (重规划) │
│  ┌─────────────────┐ │
│  │ 根据执行结果     │ │
│  │ 动态调整计划     │ │
│  └─────────────────┘ │
└──────────┬──────────┘
           │
           ▼
      最终结果
```

**实现方案**：

```python
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    step_id: int
    description: str
    tools_needed: list[str]
    dependencies: list[int] = Field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed

class ExecutionPlan(BaseModel):
    goal: str
    steps: list[PlanStep]
    context: dict = Field(default_factory=dict)

class PlanAndExecuteAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.planner_prompt = self._build_planner_prompt()
        self.executor_prompt = self._build_executor_prompt()

    async def run(self, goal: str) -> str:
        """主执行循环"""
        # Phase 1: 初始规划
        plan = await self._create_plan(goal)

        max_iterations = 5
        for iteration in range(max_iterations):
            # Phase 2: 执行
            results = await self._execute_plan(plan)

            # Phase 3: 检查是否完成
            if self._is_complete(plan, results):
                return self._synthesize_answer(plan, results)

            # Phase 4: 重规划
            plan = await self._replan(goal, plan, results)

        return self._synthesize_answer(plan, results)

    async def _create_plan(self, goal: str) -> ExecutionPlan:
        """创建初始执行计划"""
        response = await self.llm.ainvoke([
            {"role": "system", "content": self.planner_prompt},
            {"role": "user", "content": f"目标：{goal}\n\n请制定详细的执行计划。"}
        ])

        # 解析计划
        plan_data = parse_plan_response(response.content)
        return ExecutionPlan(goal=goal, steps=plan_data["steps"])

    async def _execute_plan(self, plan: ExecutionPlan) -> dict:
        """按依赖顺序执行计划"""
        results = {}

        # 拓扑排序，确定执行顺序
        execution_order = self._topological_sort(plan.steps)

        for step in execution_order:
            # 检查依赖是否满足
            if not self._dependencies_met(step, results):
                step.status = "skipped"
                continue

            # 执行步骤
            try:
                step.status = "running"
                result = await self._execute_step(step, plan.context)
                results[step.step_id] = result
                step.status = "completed"
            except Exception as e:
                results[step.step_id] = {"error": str(e)}
                step.status = "failed"

        return results

    async def _execute_step(self, step: PlanStep, context: dict) -> dict:
        """执行单个步骤"""
        # 构建步骤执行提示
        prompt = f"""
        当前任务：{step.description}
        可用工具：{step.tools_needed}
        已有结果：{context}

        请执行这个任务并返回结果。
        """

        response = await self.llm.ainvoke([
            {"role": "system", "content": self.executor_prompt},
            {"role": "user", "content": prompt}
        ])

        # 如果需要工具调用，执行工具
        if "tool_calls" in response:
            tool_results = await self._execute_tools(response.tool_calls)
            return {"result": response.content, "tools_used": tool_results}

        return {"result": response.content}

    async def _replan(self, goal: str, plan: ExecutionPlan, results: dict) -> ExecutionPlan:
        """根据执行结果重新规划"""
        prompt = f"""
        原始目标：{goal}
        原始计划：{plan.steps}
        执行结果：{results}

        根据已有的执行结果，请重新规划后续步骤。
        保留已完成的步骤，调整未完成或失败的步骤。
        """

        response = await self.llm.ainvoke([
            {"role": "system", "content": self.planner_prompt},
            {"role": "user", "content": prompt}
        ])

        return parse_plan_response(response.content)
```

**Plan-and-Execute vs ReAct 对比**：

| 维度 | ReAct | Plan-and-Execute |
|------|-------|------------------|
| 适用任务 | 简单、单步或少步任务 | 复杂、多步骤任务 |
| 推理深度 | 浅（每步独立决策） | 深（全局规划） |
| 可控性 | 低（LLM自主决定） | 高（计划可审查） |
| 执行效率 | 中（可能走弯路） | 高（按计划执行） |
| 错误恢复 | 难（可能偏离） | 易（重规划机制） |
| 实现复杂度 | 低 | 中高 |
| Token消耗 | 不可预测 | 相对可控 |

### 3.3 Multi-Agent协作模式 — 分工与协作

**模式定义**：多个专职Agent协作完成复杂任务。

```
Multi-Agent 协作架构：

┌─────────────────────────────────────────────────────────────┐
│                    协作拓扑结构                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  模式1：中心化（Hub-and-Spoke）                              │
│  ┌──────────┐                                               │
│  │  协调器   │                                               │
│  │ (Router) │                                               │
│  └────┬─────┘                                               │
│       │                                                      │
│  ┌────┼────────────────────────┐                            │
│  │    │    │                   │                            │
│  ▼    ▼    ▼                   ▼                            │
│ ┌──┐ ┌──┐ ┌──┐              ┌──┐                           │
│ │A1│ │A2│ │A3│    ...       │An│                           │
│ └──┘ └──┘ └──┘              └──┘                           │
│                                                              │
│  模式2：链式（Pipeline）                                     │
│  ┌──┐    ┌──┐    ┌──┐    ┌──┐                              │
│  │A1│───▶│A2│───▶│A3│───▶│A4│                              │
│  └──┘    └──┘    └──┘    └──┘                              │
│                                                              │
│  模式3：层级式（Hierarchical）                               │
│  ┌──────────┐                                               │
│  │  主Agent  │                                               │
│  └────┬─────┘                                               │
│       │                                                      │
│  ┌────┴──────────┐                                          │
│  │               │                                          │
│  ▼               ▼                                          │
│ ┌──────┐    ┌──────┐                                       │
│ │子Agent1│   │子Agent2│                                      │
│ └──┬───┘    └──┬───┘                                       │
│    │           │                                            │
│    ▼           ▼                                            │
│ ┌──────┐    ┌──────┐                                       │
│ │工具A  │    │工具B  │                                       │
│ └──────┘    └──────┘                                       │
│                                                              │
│  模式4：去中心化（Peer-to-Peer）                             │
│  ┌──┐    ┌──┐                                               │
│  │A1│◀──▶│A2│                                               │
│  └──┘    └──┘                                               │
│   ▲  ╲  ╱  ▲                                               │
│   │   ╲╱   │                                                │
│   │   ╱╲   │                                                │
│   ▼  ╱  ╲  ▼                                               │
│  ┌──┐    ┌──┐                                               │
│  │A3│◀──▶│A4│                                               │
│  └──┘    └──┘                                               │
└─────────────────────────────────────────────────────────────┘
```

**实现示例：中心化Multi-Agent系统**：

```python
from dataclasses import dataclass, field
from typing import Callable
import asyncio

@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    tools: list[Callable] = field(default_factory=list)
    llm: str = "claude-sonnet-4-20250514"

class MultiAgentOrchestrator:
    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.message_bus: list[dict] = []

    def register_agent(self, agent: Agent):
        """注册Agent到协调器"""
        self.agents[agent.name] = agent

    async def route_task(self, task: str) -> str:
        """智能路由：将任务分配给最合适的Agent"""
        # Step 1: 分析任务，确定需要哪些Agent
        required_agents = await self._analyze_task(task)

        if len(required_agents) == 1:
            # 单Agent直接执行
            agent = self.agents[required_agents[0]]
            return await self._execute_agent(agent, task)

        # Step 2: 多Agent协作
        if self._is_pipeline_task(required_agents):
            return await self._execute_pipeline(required_agents, task)
        else:
            return await self._execute_parallel(required_agents, task)

    async def _execute_pipeline(self, agent_names: list[str], task: str) -> str:
        """链式执行：Agent按顺序处理"""
        current_input = task
        results = []

        for name in agent_names:
            agent = self.agents[name]
            result = await self._execute_agent(agent, current_input)
            results.append({"agent": name, "result": result})
            current_input = result  # 传递给下一个Agent

        return results[-1]["result"]

    async def _execute_parallel(self, agent_names: list[str], task: str) -> str:
        """并行执行：多个Agent同时处理，最后合并"""
        tasks = [
            self._execute_agent(self.agents[name], task)
            for name in agent_names
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        successful_results = [
            r for r in results if not isinstance(r, Exception)
        ]

        # 使用合并Agent整合结果
        merge_agent = self._select_merge_agent(agent_names)
        merged = await self._execute_agent(
            merge_agent,
            f"合并以下结果：{successful_results}"
        )

        return merged

    async def _execute_agent(self, agent: Agent, task: str) -> str:
        """执行单个Agent"""
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": task}
        ]

        response = await llm_client.ainvoke(
            model=agent.llm,
            messages=messages,
            tools=agent.tools
        )

        return response.content

# 使用示例：代码审查Multi-Agent系统
orchestrator = MultiAgentOrchestrator()

orchestrator.register_agent(Agent(
    name="security_reviewer",
    role="安全审查员",
    system_prompt="你是安全专家，专注于发现代码中的安全漏洞...",
    tools=[scan_sql_injection, check_xss, audit_auth]
))

orchestrator.register_agent(Agent(
    name="performance_reviewer",
    role="性能审查员",
    system_prompt="你是性能专家，专注于发现代码中的性能问题...",
    tools=[analyze_complexity, check_n_plus_1, profile_memory]
))

orchestrator.register_agent(Agent(
    name="code_quality_reviewer",
    role="代码质量审查员",
    system_prompt="你是代码质量专家，专注于代码规范和可维护性...",
    tools=[check_naming, verify_types, assess_coverage]
))

orchestrator.register_agent(Agent(
    name="merge_reviewer",
    role="审查结果合并者",
    system_prompt="你是审查结果合并专家，负责整合多个审查意见..."
))

# 执行代码审查
result = await orchestrator.route_task(
    "审查以下代码的安全性、性能和质量：\n```python\n...\n```"
)
```

### 3.4 反思与自我纠正模式

**模式定义**：Agent在执行过程中进行自我评估和纠正。

```
自我纠正循环：

执行 → 评估 → 纠正 → 重新执行
  │       │       │       │
  ▼       ▼       ▼       ▼
┌────┐  ┌────┐  ┌────┐  ┌────┐
│生成│→ │评估│→ │修正│→ │验证│
│代码│  │代码│  │代码│  │代码│
└────┘  └────┘  └────┘  └────┘
                    │
                    ▼ (如果仍然不通过)
              ┌─────────┐
              │ 回退到   │
              │ 上一步   │
              └─────────┘

评估维度：
1. 正确性：代码是否符合预期
2. 安全性：是否有安全漏洞
3. 性能：是否有性能问题
4. 可维护性：代码是否清晰
```

**实现方案**：

```python
class SelfCorrectingAgent:
    def __init__(self, llm, max_corrections: int = 3):
        self.llm = llm
        self.max_corrections = max_corrections

    async def generate_and_correct(self, task: str) -> str:
        """生成代码并自动纠正"""
        # 初始生成
        code = await self._generate(task)

        for correction_round in range(self.max_corrections):
            # 评估
            evaluation = await self._evaluate(code, task)

            if evaluation["passed"]:
                return code

            # 纠正
            code = await self._correct(code, evaluation["issues"], task)

            # 验证纠正后的代码
            if await self._verify(code):
                return code

        return code  # 返回最后一次尝试的结果

    async def _evaluate(self, code: str, task: str) -> dict:
        """评估代码质量"""
        evaluation_prompt = f"""
        评估以下代码是否满足任务要求：

        任务：{task}
        代码：
        ```python
        {code}
        ```

        评估维度：
        1. 正确性：代码是否正确实现了功能
        2. 边界条件：是否处理了边界情况
        3. 错误处理：是否有适当的错误处理
        4. 类型安全：类型注解是否完整
        5. 性能：是否有明显的性能问题

        输出格式：
        {{
            "passed": true/false,
            "issues": [
                {{"type": "...", "description": "...", "severity": "..."}}
            ],
            "score": 0-100
        }}
        """

        response = await self.llm.ainvoke([
            {"role": "user", "content": evaluation_prompt}
        ])

        return parse_json_response(response.content)

    async def _correct(self, code: str, issues: list, task: str) -> str:
        """根据评估结果纠正代码"""
        correction_prompt = f"""
        以下代码存在以下问题，请修正：

        原始代码：
        ```python
        {code}
        ```

        问题列表：
        {format_issues(issues)}

        任务要求：{task}

        请输出修正后的完整代码。
        """

        response = await self.llm.ainvoke([
            {"role": "user", "content": correction_prompt}
        ])

        return extract_code_from_response(response.content)

    async def _verify(self, code: str) -> bool:
        """验证代码是否可以执行"""
        try:
            # 尝试编译
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False
```

## 四、生产级Agent系统的关键设计

### 4.1 状态管理：检查点与恢复

```python
from datetime import datetime
import json

class AgentCheckpoint:
    """Agent执行状态的检查点管理"""

    def __init__(self, storage_path: str = "./checkpoints"):
        self.storage_path = storage_path

    async def save(self, agent_id: str, state: dict):
        """保存检查点"""
        checkpoint = {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "state": state,
            "messages_count": len(state.get("messages", [])),
            "current_step": state.get("current_step", 0)
        }

        path = f"{self.storage_path}/{agent_id}/checkpoint_{datetime.utcnow().timestamp()}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    async def load_latest(self, agent_id: str) -> dict | None:
        """加载最新的检查点"""
        import glob

        pattern = f"{self.storage_path}/{agent_id}/checkpoint_*.json"
        files = sorted(glob.glob(pattern), reverse=True)

        if not files:
            return None

        with open(files[0], "r") as f:
            checkpoint = json.load(f)

        return checkpoint["state"]

    async def list_checkpoints(self, agent_id: str) -> list[dict]:
        """列出所有检查点"""
        import glob

        pattern = f"{self.storage_path}/{agent_id}/checkpoint_*.json"
        files = sorted(glob.glob(pattern), reverse=True)

        checkpoints = []
        for file in files:
            with open(file, "r") as f:
                checkpoint = json.load(f)
                checkpoints.append({
                    "timestamp": checkpoint["timestamp"],
                    "step": checkpoint["current_step"],
                    "file": file
                })

        return checkpoints
```

### 4.2 错误处理与容错

```python
class AgentError(Enum):
    TOOL_FAILURE = "tool_failure"
    LLM_TIMEOUT = "llm_timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    VALIDATION_FAILED = "validation_failed"

class AgentErrorHandler:
    """Agent错误处理与恢复策略"""

    def __init__(self, llm):
        self.llm = llm
        self.retry_strategies = {
            AgentError.TOOL_FAILURE: self._handle_tool_failure,
            AgentError.LLM_TIMEOUT: self._handle_llm_timeout,
            AgentError.CONTEXT_OVERFLOW: self._handle_context_overflow,
            AgentError.MAX_STEPS_EXCEEDED: self._handle_max_steps,
            AgentError.VALIDATION_FAILED: self._handle_validation_failure,
        }

    async def handle_error(self, error: AgentError, context: dict) -> dict:
        """处理错误并返回恢复策略"""
        handler = self.retry_strategies.get(error)
        if handler:
            return await handler(context)

        return {"action": "abort", "reason": f"Unhandled error: {error}"}

    async def _handle_tool_failure(self, context: dict) -> dict:
        """处理工具调用失败"""
        tool_name = context.get("tool_name")
        error_msg = context.get("error_message")

        # 尝试替代工具
        alternative = await self._find_alternative_tool(tool_name)
        if alternative:
            return {
                "action": "retry_with_alternative",
                "tool": alternative,
                "reason": f"原工具 {tool_name} 失败，使用替代工具"
            }

        # 尝试简化请求
        return {
            "action": "simplify_and_retry",
            "reason": f"工具 {tool_name} 不可用，尝试简化任务"
        }

    async def _handle_context_overflow(self, context: dict) -> dict:
        """处理上下文溢出"""
        messages = context.get("messages", [])

        # 策略1：压缩旧消息
        if len(messages) > 10:
            summary = await self._summarize_messages(messages[:5])
            return {
                "action": "compress_context",
                "summary": summary,
                "reason": "上下文过长，压缩历史消息"
            }

        # 策略2：切换到更长上下文的模型
        return {
            "action": "switch_model",
            "model": "claude-opus-4-20250514",
            "reason": "需要更长上下文窗口"
        }

    async def _handle_max_steps(self, context: dict) -> dict:
        """处理超过最大步数"""
        # 尝试使用更高效的策略
        return {
            "action": "switch_strategy",
            "strategy": "plan_and_execute",
            "reason": "当前策略效率低下，切换到规划执行模式"
        }
```

### 4.3 可观测性：追踪与调试

```python
import logging
from dataclasses import dataclass
from typing import Any
import time

@dataclass
class AgentTrace:
    """Agent执行追踪"""
    agent_id: str
    step: int
    action: str
    input_data: Any
    output_data: Any
    duration_ms: float
    token_usage: dict
    timestamp: float

class AgentTracer:
    """Agent执行追踪器"""

    def __init__(self):
        self.traces: list[AgentTrace] = []
        self.logger = logging.getLogger("agent_tracer")

    def trace_step(self, agent_id: str, step: int):
        """追踪步骤的装饰器"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000

                    trace = AgentTrace(
                        agent_id=agent_id,
                        step=step,
                        action=func.__name__,
                        input_data=str(args)[:200],
                        output_data=str(result)[:200],
                        duration_ms=duration,
                        token_usage=self._extract_token_usage(result),
                        timestamp=time.time()
                    )
                    self.traces.append(trace)

                    self.logger.info(
                        f"[{agent_id}] Step {step}: {func.__name__} "
                        f"completed in {duration:.1f}ms"
                    )

                    return result

                except Exception as e:
                    duration = (time.time() - start_time) * 1000
                    self.logger.error(
                        f"[{agent_id}] Step {step}: {func.__name__} "
                        f"failed after {duration:.1f}ms: {e}"
                    )
                    raise

            return wrapper
        return decorator

    def get_summary(self, agent_id: str) -> dict:
        """获取执行摘要"""
        agent_traces = [t for t in self.traces if t.agent_id == agent_id]

        if not agent_traces:
            return {"error": "No traces found"}

        total_duration = sum(t.duration_ms for t in agent_traces)
        total_tokens = sum(
            t.token_usage.get("total_tokens", 0)
            for t in agent_traces
        )

        return {
            "agent_id": agent_id,
            "total_steps": len(agent_traces),
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
            "avg_step_duration_ms": total_duration / len(agent_traces),
            "steps": [
                {
                    "step": t.step,
                    "action": t.action,
                    "duration_ms": t.duration_ms,
                    "tokens": t.token_usage.get("total_tokens", 0)
                }
                for t in agent_traces
            ]
        }
```

## 五、性能优化与成本控制

### 5.1 Token消耗优化

```
┌─────────────────────────────────────────────────────────────────┐
│                    Token消耗优化策略                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  策略1：模型路由（Model Routing）                                │
│  ─────────────────────────────────                               │
│  简单任务 → 小模型（GPT-4o-mini / Claude Haiku）               │
│  复杂推理 → 大模型（Claude Opus / o3）                         │
│  代码生成 → 专用模型（DeepSeek Coder / Codestral）             │
│                                                                  │
│  Token节省：60-70%                                               │
│                                                                  │
│  策略2：上下文压缩（Context Compression）                        │
│  ─────────────────────────────────                               │
│  定期总结历史消息 → 保留关键信息                                 │
│  工具结果缓存 → 相同请求直接返回                                 │
│  去除冗余信息 → 只保留与当前任务相关的上下文                     │
│                                                                  │
│  Token节省：40-50%                                               │
│                                                                  │
│  策略3：预测执行（Speculative Execution）                        │
│  ─────────────────────────────────                               │
│  预测可能的下一步 → 并行执行                                     │
│  缓存常见模式 → 避免重复推理                                     │
│                                                                  │
│  延迟降低：30-40%                                                │
│                                                                  │
│  策略4：批处理（Batching）                                       │
│  ─────────────────────────────────                               │
│  多个独立任务 → 批量调用                                         │
│  工具调用合并 → 减少API调用次数                                  │
│                                                                  │
│  成本降低：20-30%                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 延迟优化

```python
class LatencyOptimizer:
    """Agent延迟优化器"""

    def __init__(self):
        self.tool_cache = {}
        self.response_cache = {}

    async def optimized_execute(self, agent, task: str) -> str:
        """优化执行流程"""
        # 1. 检查缓存
        cache_key = self._compute_cache_key(task)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        # 2. 预测性工具调用
        predicted_tools = await self._predict_tools(task)

        # 3. 并行预加载
        preloaded = await self._parallel_preload(predicted_tools)

        # 4. 执行Agent
        result = await agent.run(task, preloaded_context=preloaded)

        # 5. 缓存结果
        self.response_cache[cache_key] = result

        return result

    async def _predict_tools(self, task: str) -> list[str]:
        """预测任务可能需要的工具"""
        prediction_prompt = f"""
        分析以下任务，预测可能需要的工具：

        任务：{task}

        可用工具列表：
        - search_web: 网络搜索
        - query_database: 数据库查询
        - read_file: 文件读取
        - execute_code: 代码执行
        - send_email: 发送邮件

        输出需要的工具名称列表（JSON数组）。
        """

        response = await self.llm.ainvoke([
            {"role": "user", "content": prediction_prompt}
        ])

        return parse_json_response(response.content)
```

## 六、架构选型决策指南

### 6.1 选型决策树

```
开始选型
  │
  ├─ 任务复杂度？
  │   ├─ 简单（1-3步）→ ReAct模式
  │   │   └─ 需要自我纠正？→ 添加反思循环
  │   │
  │   ├─ 中等（3-10步）→ Plan-and-Execute
  │   │   └─ 任务可并行？→ 并行执行分支
  │   │
  │   └─ 复杂（>10步）→ Multi-Agent + Plan-and-Execute
  │       └─ 需要专业知识分工？→ 角色化Agent
  │
  ├─ 可靠性要求？
  │   ├─ 高（金融/医疗）→ Plan-and-Execute + 检查点 + 人工审核
  │   ├─ 中（内部工具）→ ReAct + 错误恢复
  │   └─ 低（探索性）→ 简单ReAct
  │
  ├─ 延迟要求？
  │   ├─ 实时（<1s）→ 简单ReAct + 小模型
  │   ├─ 准实时（1-10s）→ ReAct + 模型路由
  │   └─ 可异步（>10s）→ Plan-and-Execute + 异步执行
  │
  └─ 成本预算？
      ├─ 低 → 小模型 + 缓存 + 批处理
      ├─ 中 → 模型路由 + 上下文压缩
      └─ 高 → 全功能 + 大模型 + 实时追踪
```

### 6.2 技术栈推荐

| 场景 | 推荐框架 | 模型 | 部署方式 |
|------|----------|------|----------|
| 快速原型 | LangChain | Claude Haiku | 本地 |
| 生产ReAct | LangGraph | Claude Sonnet | K8s |
| 复杂规划 | AutoGen | Claude Opus | K8s |
| Multi-Agent | CrewAI | 混合 | K8s |
| 企业级 | Semantic Kernel | Azure OpenAI | 私有云 |

## 七、实战案例：构建一个代码审查Agent系统

### 7.1 系统架构

```
代码审查Agent系统架构：

┌─────────────────────────────────────────────────────────────┐
│                    代码审查Agent系统                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GitHub Webhook ──▶ API Gateway ──▶ 任务队列               │
│                                          │                   │
│                                          ▼                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              协调器 (Orchestrator)                    │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  1. 获取PR差异                                 │  │    │
│  │  │  2. 分析变更类型                               │  │    │
│  │  │  3. 分配审查Agent                              │  │    │
│  │  │  4. 合并审查结果                               │  │    │
│  │  │  5. 发布审查评论                               │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│  ┌──────────┬───────────┼───────────┬──────────┐           │
│  │          │           │           │          │           │
│  ▼          ▼           ▼           ▼          ▼           │
│ ┌────┐   ┌────┐     ┌────┐     ┌────┐    ┌────┐         │
│ │安全 │   │性能 │     │逻辑 │     │风格 │    │测试 │         │
│ │审查 │   │审查 │     │审查 │     │审查 │    │审查 │         │
│ └────┘   └────┘     └────┘     └────┘    └────┘         │
│                                                              │
│  结果合并 ──▶ 评分计算 ──▶ 评论发布 ──▶ 通知开发者         │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 核心代码

```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()

class CodeReviewSystem:
    def __init__(self):
        self.orchestrator = MultiAgentOrchestrator()
        self._setup_agents()

    def _setup_agents(self):
        """设置审查Agent"""
        self.orchestrator.register_agent(Agent(
            name="security_reviewer",
            role="安全审查",
            system_prompt="""你是安全专家，专注于发现：
            - SQL注入、XSS等注入漏洞
            - 认证和授权问题
            - 敏感数据泄露
            - 依赖安全漏洞
            输出格式：[严重程度] [文件:行号] 问题描述""",
            tools=[scan_dependencies, check_secrets]
        ))

        self.orchestrator.register_agent(Agent(
            name="performance_reviewer",
            role="性能审查",
            system_prompt="""你是性能专家，专注于发现：
            - O(n²)或更高复杂度的算法
            - N+1查询问题
            - 内存泄漏风险
            - 不必要的同步操作
            输出格式：[影响程度] [文件:行号] 问题描述 + 优化建议""",
            tools=[analyze_complexity]
        ))

        self.orchestrator.register_agent(Agent(
            name="logic_reviewer",
            role="逻辑审查",
            system_prompt="""你是逻辑审查专家，专注于发现：
            - 边界条件处理不当
            - 错误处理缺失
            - 业务逻辑错误
            - 并发安全问题
            输出格式：[风险等级] [文件:行号] 问题描述""",
            tools=[]
        ))

    async def review_pull_request(self, pr_data: dict) -> dict:
        """审查PR"""
        # 获取变更文件
        files = await self._get_pr_files(pr_data["pr_number"])

        # 分类变更
        changes_by_type = self._classify_changes(files)

        # 并行审查
        review_tasks = []
        for change_type, change_files in changes_by_type.items():
            task = self._create_review_task(change_type, change_files)
            review_tasks.append(task)

        results = await asyncio.gather(*review_tasks)

        # 合并结果
        merged = self._merge_results(results)

        # 计算评分
        score = self._calculate_score(merged)

        # 发布评论
        await self._post_review_comment(pr_data, merged, score)

        return {"score": score, "issues": merged}
```

## 八、总结与最佳实践

### 8.1 架构设计原则

```
Agent系统设计的五项原则：

1. 明确边界（Clear Boundaries）
   ─────────────────────────
   每个Agent只负责一件事
   Agent之间通过消息通信，不共享状态
   工具调用有明确的输入输出契约

2. 渐进增强（Progressive Enhancement）
   ─────────────────────────────────────
   从简单ReAct开始，根据需要增加复杂度
   不要过度设计，80%的场景用ReAct就够了

3. 失败安全（Fail-Safe）
   ─────────────────────
   每个步骤都有超时和重试机制
   关键操作有检查点，可以恢复
   始终有降级方案

4. 可观测性（Observability）
   ─────────────────────────
   记录每一步的输入输出
   追踪Token消耗和延迟
   支持事后分析和调试

5. 成本意识（Cost-Aware）
   ─────────────────────
   模型路由：小任务用小模型
   缓存策略：避免重复调用
   批处理：合并独立请求
```

### 8.2 避坑清单

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| Agent陷入循环 | 缺乏终止条件 | 设置max_steps + 循环检测 |
| 输出质量不稳定 | Prompt不够精确 | Few-shot示例 + 输出格式约束 |
| 工具调用失败率高 | 工具描述不清晰 | 优化工具描述 + 参数校验 |
| 成本失控 | 无限制的上下文 | 上下文压缩 + Token预算 |
| 调试困难 | 缺乏追踪 | 结构化日志 + 执行追踪 |

### 8.3 2026年下半年趋势

> **Agent系统正在从"能用"走向"好用"，核心变化是：**
>
> 1. **更智能的规划**：从固定流程到LLM驱动的动态规划
> 2. **更强的协作**：Multi-Agent成为标配，角色分工更精细
> 3. **更好的可观测性**：全链路追踪、成本分析、质量评估一体化
> 4. **更完善的容错**：检查点恢复、自动降级、人工干预机制成熟
> 5. **更低的成本**：模型路由、缓存、批处理让Agent系统可以大规模部署

> *"设计Agent系统的关键不是让它变得更聪明，而是让它变得更可靠。"*

---

**参考资源**：
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
- [AutoGen文档](https://microsoft.github.io/autogen/)
- [CrewAI文档](https://docs.crewai.com/)
- [ReAct论文](https://arxiv.org/abs/2210.03629)
- [Plan-and-Solve论文](https://arxiv.org/abs/2305.04091)
