---
title: "Agent Loop执行机制：ReAct/Plan-Execute/自反思循环的工程实现"
description: "深度剖析三大Agent Loop执行模式的状态机本质，含完整Python工程实现、并发安全、循环检测、中断恢复等生产级设计"
date: 2026-05-30
author: "RiceBall-15"
category: agent
subCategory: agent-dev
tags: ["Agent Loop", "ReAct", "Plan-Execute", "Reflexion", "状态机"]
draft: false
---

## 核心问题

Agent 的本质是什么？不是一次 LLM 调用，而是一个**循环执行的状态机**——它不断地感知环境、做出决策、执行动作、观察结果，直到任务完成或资源耗尽。这个循环，就是 Agent Loop。

所有 Agent 框架（LangChain、AutoGPT、CrewAI、OpenAI Agents SDK）的底层骨架，都是某种形式的 Agent Loop。理解它的工程实现，是区分"会调 API"和"能造系统"的分水岭。

---

## 一、Agent Loop 本质：状态机视角的执行循环

从形式化角度看，任何 Agent Loop 都可以建模为一个有限状态机（FSM）：

```
┌─────────────────────────────────────────────────┐
│                Agent Loop FSM                   │
│                                                 │
│  ┌──────┐   LLM推理   ┌──────┐   工具执行  ┌──────┐
│  │ INIT │───────────→│THINK │───────────→│ACT   │
│  └──────┘            └──────┘            └──────┘
│                              ↑                │
│                              │   环境反馈       │
│                              └────────────────│
│                                                 │
│  ACT ──→ OBSERVE ──→ 判断终止条件              │
│           │  │                                  │
│           │  └──→ 未完成 → 回到 THINK           │
│           └────→ 已完成 → DONE                  │
└─────────────────────────────────────────────────┘
```

**状态定义**：
- `INIT`：初始化上下文，加载工具、提示词
- `THINK`：LLM 推理，产生思考过程和行动意图
- `ACT`：执行具体工具调用或生成最终输出
- `OBSERVE`：收集执行结果，更新上下文
- `DONE`：任务完成或触发终止条件

关键洞察：**所有 Agent Loop 的区别，只在于"思考"和"行动"的组织方式不同。**

---

## 二、ReAct Loop：思考→行动→观察的最简实现

ReAct（Reasoning + Acting）是最经典的 Agent Loop 模式。核心思想：让 LLM 在每一步先"思考"（Thought），再"行动"（Action），最后"观察"（Observation）结果。

### 完整 Python 实现

```python
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum


class LoopStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    DUPLICATE_STATE = "duplicate_state"
    ERROR = "error"


@dataclass
class AgentState:
    """Agent Loop 的完整状态，支持序列化和检查点"""
    iteration: int = 0
    history: list = field(default_factory=list)  # (thought, action, observation) 三元组
    state_hashes: set = field(default_factory=set)  # 用于循环检测
    start_time: float = field(default_factory=time.time)
    final_answer: Optional[str] = None

    def compute_hash(self, thought: str, action: str) -> str:
        """计算当前状态的哈希值，用于检测重复状态"""
        content = f"{thought}||{action}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_step(self, thought: str, action: str, observation: str):
        state_hash = self.compute_hash(thought, action)
        self.state_hashes.add(state_hash)
        self.history.append({
            "iteration": self.iteration,
            "thought": thought,
            "action": action,
            "observation": observation,
        })
        self.iteration += 1

    def to_checkpoint(self) -> dict:
        """序列化为检查点，支持持久化"""
        return {
            "iteration": self.iteration,
            "history": self.history,
            "state_hashes": list(self.state_hashes),
            "start_time": self.start_time,
            "final_answer": self.final_answer,
        }

    @classmethod
    def from_checkpoint(cls, data: dict) -> "AgentState":
        """从检查点恢复"""
        state = cls()
        state.iteration = data["iteration"]
        state.history = data["history"]
        state.state_hashes = set(data["state_hashes"])
        state.start_time = data["start_time"]
        state.final_answer = data.get("final_answer")
        return state


def parse_llm_output(llm_response: str) -> tuple[str, str, Optional[str]]:
    """
    解析 LLM 输出，提取 Thought、Action 和 Final Answer。
    这里简化处理，实际项目中用结构化输出更可靠。
    """
    thought = ""
    action = ""
    final_answer = None

    for line in llm_response.strip().split("\n"):
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            action = line[len("Action:"):].strip()
        elif line.startswith("Final Answer:"):
            final_answer = line[len("Final Answer:"):].strip()

    return thought, action, final_answer


def react_loop(
    initial_prompt: str,
    llm_call: Callable[[str], str],
    tool_executor: Callable[[str], str],
    max_iterations: int = 10,
    timeout_seconds: float = 120.0,
    state_hash_dedup: bool = True,
    checkpoint_callback: Optional[Callable[[dict], None]] = None,
) -> tuple[LoopStatus, AgentState]:
    """
    ReAct Agent Loop 完整实现

    参数:
        initial_prompt: 初始用户查询
        llm_call: LLM 调用函数，接收 prompt 返回 str
        tool_executor: 工具执行函数，接收 action 返回 observation
        max_iterations: 最大迭代次数
        timeout_seconds: 超时时间（秒）
        state_hash_dedup: 是否启用状态哈希去重
        checkpoint_callback: 检查点回调函数，每步调用

    返回:
        (LoopStatus, AgentState) 终止状态和完整执行状态
    """
    state = AgentState()
    system_prompt = (
        "You are a helpful assistant. Think step by step.\n"
        "Format your response as:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name(args)>\n"
        "or\n"
        "Thought: <your reasoning>\n"
        "Final Answer: <answer>\n"
    )

    current_context = f"{system_prompt}\n\nUser: {initial_prompt}"

    while True:
        # 检查最大迭代
        if state.iteration >= max_iterations:
            return LoopStatus.MAX_ITERATIONS, state

        # 检查超时
        elapsed = time.time() - state.start_time
        if elapsed > timeout_seconds:
            return LoopStatus.TIMEOUT, state

        # Step 1: THINK — 调用 LLM
        try:
            llm_response = llm_call(current_context)
        except Exception as e:
            state.history.append({
                "iteration": state.iteration,
                "error": str(e),
            })
            return LoopStatus.ERROR, state

        thought, action, final_answer = parse_llm_output(llm_response)

        # 检查是否给出最终答案
        if final_answer:
            state.final_answer = final_answer
            return LoopStatus.COMPLETED, state

        # 检查状态哈希去重
        if state_hash_dedup:
            current_hash = state.compute_hash(thought, action)
            if current_hash in state.state_hashes:
                return LoopStatus.DUPLICATE_STATE, state

        # Step 2: ACT — 执行工具
        try:
            observation = tool_executor(action)
        except Exception as e:
            observation = f"Error executing tool: {str(e)}"

        # Step 3: OBSERVE — 更新状态
        state.add_step(thought, action, observation)

        # 更新上下文
        current_context += (
            f"\nThought: {thought}\n"
            f"Action: {action}\n"
            f"Observation: {observation}\n"
        )

        # 检查点持久化
        if checkpoint_callback:
            checkpoint_callback(state.to_checkpoint())

    # 不会到达这里，但为了类型检查
    return LoopStatus.ERROR, state


# 使用示例
def demo_react():
    """演示 ReAct Loop 的使用"""

    def mock_llm(prompt: str) -> str:
        """模拟 LLM 调用"""
        if "Search" not in prompt and "Observation" not in prompt:
            return "Thought: I need to search for the weather.\nAction: Search(Beijing weather)"
        elif prompt.count("Observation") < 2:
            return "Thought: Got weather data, need to convert units.\nAction: ConvertUnit(Celsius to Fahrenheit)"
        else:
            return "Thought: I have enough information now.\nFinal Answer: 北京今天晴，25°C (77°F)"

    def mock_tool(action: str) -> str:
        """模拟工具执行"""
        if "Search" in action:
            return "北京今天：晴，25°C，湿度40%"
        elif "ConvertUnit" in action:
            return "25°C = 77°F"
        return "Unknown action"

    status, state = react_loop(
        initial_prompt="北京今天天气怎么样？请用华氏度告诉我。",
        llm_call=mock_llm,
        tool_executor=mock_tool,
        max_iterations=5,
        timeout_seconds=30.0,
    )

    print(f"Status: {status.value}")
    print(f"Iterations: {state.iteration}")
    print(f"Answer: {state.final_answer}")
    print(f"History length: {len(state.history)}")
    for step in state.history:
        print(f"  [{step.get('iteration')}] Thought: {step.get('thought', 'N/A')}")


if __name__ == "__main__":
    demo_react()
```

**关键设计点**：

1. **三重终止条件**：最大迭代次数、超时、状态重复——三者缺一不可
2. **状态哈希去重**：防止 Agent 陷入相同思考-行动对的死循环
3. **检查点回调**：每步都可通过 callback 持久化状态，支持中断恢复
4. **异常隔离**：LLM 调用和工具执行的异常分开处理，不会导致整个 Loop 崩溃

---

## 三、Plan-Execute Loop：先规划后执行

与 ReAct 的"走一步看一步"不同，Plan-Execute 模式先生成完整计划，再按计划执行。适用于任务结构清晰、可提前分解的场景。

```
┌────────────────────────────────────────────────────────┐
│                Plan-Execute Loop                       │
│                                                        │
│  用户查询 → Plan Generation → 任务分解 → 并行执行      │
│              (LLM规划)       (子任务列表)  (Worker池)  │
│                                    ↓                   │
│                            结果聚合 → 输出              │
│                                                        │
│  如果某个子任务失败 → 重新规划（Re-plan）              │
└────────────────────────────────────────────────────────┘
```

### 完整 Python 实现

```python
import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class SubTask:
    """子任务定义"""
    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)  # 依赖的子任务 ID
    result: Optional[str] = None
    status: str = "pending"  # pending / running / completed / failed


@dataclass
class ExecutionPlan:
    """执行计划"""
    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def get_subtask(self, task_id: str) -> Optional[SubTask]:
        for st in self.subtasks:
            if st.id == task_id:
                return st
        return None

    def get_ready_tasks(self) -> list[SubTask]:
        """获取所有依赖已满足的待执行任务"""
        ready = []
        for st in self.subtasks:
            if st.status != "pending":
                continue
            deps_met = all(
                self.get_subtask(dep_id) is not None
                and self.get_subtask(dep_id).status == "completed"
                for dep_id in st.dependencies
            )
            if deps_met:
                ready.append(st)
        return ready

    def all_completed(self) -> bool:
        return all(st.status in ("completed", "failed") for st in self.subtasks)

    def any_failed(self) -> bool:
        return any(st.status == "failed" for st in self.subtasks)

    def to_checkpoint(self) -> dict:
        return {
            "goal": self.goal,
            "subtasks": [
                {
                    "id": st.id,
                    "description": st.description,
                    "dependencies": st.dependencies,
                    "result": st.result,
                    "status": st.status,
                }
                for st in self.subtasks
            ],
            "created_at": self.created_at,
        }

    @classmethod
    def from_checkpoint(cls, data: dict) -> "ExecutionPlan":
        plan = cls(goal=data["goal"], created_at=data["created_at"])
        for st_data in data["subtasks"]:
            plan.subtasks.append(SubTask(**st_data))
        return plan


def parse_plan_from_llm(llm_output: str) -> list[dict]:
    """
    从 LLM 输出解析子任务计划。
    实际项目中应使用结构化输出（JSON mode / function calling）。
    """
    try:
        tasks = json.loads(llm_output)
        return tasks if isinstance(tasks, list) else []
    except json.JSONDecodeError:
        # 降级：简单文本解析
        tasks = []
        for i, line in enumerate(llm_output.strip().split("\n")):
            line = line.strip()
            if line and not line.startswith("#"):
                tasks.append({
                    "id": f"task_{i}",
                    "description": line,
                    "dependencies": [],
                })
        return tasks


class PlanExecuteLoop:
    """Plan-Execute Agent Loop"""

    def __init__(
        self,
        llm_call: Callable[[str], str],
        task_executor: Callable[[SubTask], str],
        max_replans: int = 3,
        max_workers: int = 4,
        timeout_seconds: float = 300.0,
    ):
        self.llm_call = llm_call
        self.task_executor = task_executor
        self.max_replans = max_replans
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds

    def _generate_plan(self, goal: str) -> ExecutionPlan:
        """Step 1: 让 LLM 生成执行计划"""
        planning_prompt = f"""
        Given the following goal, create a structured execution plan as a JSON array.
        Each element should have: id, description, dependencies (list of task IDs it depends on).

        Goal: {goal}

        Return only the JSON array, no other text.
        """
        llm_output = self.llm_call(planning_prompt)
        tasks_data = parse_plan_from_llm(llm_output)

        plan = ExecutionPlan(goal=goal)
        for t in tasks_data:
            plan.subtasks.append(SubTask(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
            ))
        return plan

    def _execute_parallel(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Step 2: 按依赖关系并行执行子任务"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not plan.all_completed():
                ready_tasks = plan.get_ready_tasks()
                if not ready_tasks and not plan.all_completed():
                    # 可能存在循环依赖或全部失败
                    break

                future_to_task = {}
                for task in ready_tasks:
                    task.status = "running"
                    future = executor.submit(self.task_executor, task)
                    future_to_task[future] = task

                for future in as_completed(future_to_task, timeout=self.timeout_seconds):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=10)
                        task.result = result
                        task.status = "completed"
                    except Exception as e:
                        task.result = f"Error: {str(e)}"
                        task.status = "failed"

        return plan

    def _aggregate_results(self, plan: ExecutionPlan) -> str:
        """Step 3: 聚合所有子任务结果"""
        aggregation_prompt = f"""
        Goal: {plan.goal}

        Results from subtasks:
        """
        for st in plan.subtasks:
            aggregation_prompt += f"\n- [{st.id}] {st.description}: {st.result}"

        aggregation_prompt += "\n\nSynthesize a comprehensive answer based on these results:"
        return self.llm_call(aggregation_prompt)

    def _replan(self, goal: str, failed_tasks: list[SubTask], completed_tasks: list[SubTask]) -> ExecutionPlan:
        """失败时重新规划"""
        replan_prompt = f"""
        Original goal: {goal}

        These tasks completed successfully:
        {[{"id": t.id, "result": t.result} for t in completed_tasks]}

        These tasks failed:
        {[{"id": t.id, "description": t.description, "error": t.result} for t in failed_tasks]}

        Create a new plan that:
        1. Preserves completed work
        2. Replaces failed tasks with alternative approaches
        3. Adds any new tasks needed

        Return as JSON array.
        """
        llm_output = self.llm_call(replan_prompt)
        tasks_data = parse_plan_from_llm(llm_output)

        plan = ExecutionPlan(goal=goal)
        for t in tasks_data:
            plan.subtasks.append(SubTask(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
            ))
        return plan

    def run(self, goal: str, checkpoint_callback: Optional[Callable] = None) -> tuple[str, ExecutionPlan]:
        """
        执行 Plan-Execute Loop

        返回:
            (最终答案, 执行计划)
        """
        start_time = time.time()
        replan_count = 0
        plan = self._generate_plan(goal)

        while replan_count <= self.max_replans:
            # 检查超时
            if time.time() - start_time > self.timeout_seconds:
                break

            # 并行执行
            plan = self._execute_parallel(plan)

            # 检查点
            if checkpoint_callback:
                checkpoint_callback(plan.to_checkpoint())

            # 检查是否有失败任务
            if not plan.any_failed():
                break

            # 重新规划
            replan_count += 1
            completed = [st for st in plan.subtasks if st.status == "completed"]
            failed = [st for st in plan.subtasks if st.status == "failed"]
            plan = self._replan(goal, failed, completed)

        # 聚合结果
        final_answer = self._aggregate_results(plan)
        return final_answer, plan
```

**Plan-Execute 的核心优势**：

1. **并行执行**：无依赖关系的子任务可同时执行，总延迟 = 关键路径长度而非所有任务之和
2. **失败恢复**：某个子任务失败不影响其他任务，只需 Re-plan 失败部分
3. **可预测性**：计划提前生成，用户可预览和修改

---

## 四、Reflexion Loop：自反思循环

Reflexion 模式的核心洞察：**人类不是一次就做对的，而是通过反思和改进来提升质量。** 每次执行后，Agent 会评估自己的表现，生成反思结论，然后带着新的理解重新尝试。

```
┌──────────────────────────────────────────────────┐
│                Reflexion Loop                     │
│                                                   │
│  执行任务 → 评估结果 → 生成反思 → 改进策略        │
│     ↑                                        │    │
│     └────────────────────────────────────────┘    │
│                                                   │
│  终止条件：达到质量阈值 或 最大反思次数           │
└──────────────────────────────────────────────────┘
```

### 完整 Python 实现

```python
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ReflexionStep:
    """单次反思循环的记录"""
    attempt: int
    execution_result: str
    evaluation_score: float  # 0.0 ~ 1.0
    reflection: str  # 反思内容
    improved_strategy: str  # 改进策略
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflexionState:
    """Reflexion Loop 的完整状态"""
    goal: str
    steps: list[ReflexionStep] = field(default_factory=list)
    current_strategy: str = ""
    best_result: Optional[str] = None
    best_score: float = 0.0

    def to_checkpoint(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "attempt": s.attempt,
                    "execution_result": s.execution_result,
                    "evaluation_score": s.evaluation_score,
                    "reflection": s.reflection,
                    "improved_strategy": s.improved_strategy,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "current_strategy": self.current_strategy,
            "best_result": self.best_result,
            "best_score": self.best_score,
        }


class ReflexionLoop:
    """
    Reflexion 自反思循环实现

    基于论文: "Reflexion: Language Agents with Verbal Reinforcement Learning"
    (Shinn et al., 2023)
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        executor: Callable[[str, str], str],  # (goal, strategy) -> result
        evaluator: Callable[[str, str], float],  # (goal, result) -> score [0,1]
        quality_threshold: float = 0.9,
        max_attempts: int = 5,
        timeout_seconds: float = 300.0,
    ):
        self.llm_call = llm_call
        self.executor = executor
        self.evaluator = evaluator
        self.quality_threshold = quality_threshold
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds

    def _reflect(
        self, goal: str, execution_result: str,
        score: float, history: list[ReflexionStep]
    ) -> tuple[str, str]:
        """
        Step 2: 生成反思和改进策略

        Returns:
            (reflection_text, improved_strategy)
        """
        history_text = "\n".join(
            f"Attempt {s.attempt}: score={s.evaluation_score:.2f}\n"
            f"  Reflection: {s.reflection}\n"
            f"  Strategy used: {s.improved_strategy}"
            for s in history
        ) if history else "No previous attempts."

        reflect_prompt = f"""
You are an AI agent reflecting on your performance.

Goal: {goal}

Your latest execution result:
{execution_result}

Evaluation score: {score:.2f} (0=failure, 1=perfect)

Previous history:
{history_text}

Please provide:
1. A detailed reflection on what went wrong or could be improved.
2. A concrete improved strategy for the next attempt.

Format your response as:
REFLECTION: <your reflection>
STRATEGY: <your improved strategy>
"""
        response = self.llm_call(reflect_prompt)

        reflection = ""
        strategy = ""
        for line in response.split("\n"):
            if line.startswith("REFLECTION:"):
                reflection = line[len("REFLECTION:"):].strip()
            elif line.startswith("STRATEGY:"):
                strategy = line[len("STRATEGY:"):].strip()

        return reflection, strategy

    def run(
        self, goal: str,
        initial_strategy: str = "",
        checkpoint_callback: Optional[Callable] = None,
    ) -> tuple[str, ReflexionState]:
        """
        执行 Reflexion Loop

        返回:
            (最佳结果, 完整反思状态)
        """
        state = ReflexionState(goal=goal)
        state.current_strategy = initial_strategy or "No specific strategy. Try your best."
        start_time = time.time()

        for attempt in range(1, self.max_attempts + 1):
            # 检查超时
            if time.time() - start_time > self.timeout_seconds:
                break

            # Step 1: 执行任务
            try:
                execution_result = self.executor(goal, state.current_strategy)
            except Exception as e:
                execution_result = f"Execution failed: {str(e)}"

            # Step 2: 评估结果
            score = self.evaluator(goal, execution_result)

            # 更新最佳结果
            if score > state.best_score:
                state.best_score = score
                state.best_result = execution_result

            # 达到质量阈值，提前退出
            if score >= self.quality_threshold:
                state.steps.append(ReflexionStep(
                    attempt=attempt,
                    execution_result=execution_result,
                    evaluation_score=score,
                    reflection="Goal achieved - quality threshold met.",
                    improved_strategy="N/A",
                ))
                break

            # Step 3: 反思
            reflection, improved_strategy = self._reflect(
                goal, execution_result, score, state.steps
            )

            # 记录这一步
            state.steps.append(ReflexionStep(
                attempt=attempt,
                execution_result=execution_result,
                evaluation_score=score,
                reflection=reflection,
                improved_strategy=improved_strategy,
            ))

            # Step 4: 更新策略
            state.current_strategy = improved_strategy

            # 检查点
            if checkpoint_callback:
                checkpoint_callback(state.to_checkpoint())

        return state.best_result or "", state


# 演示
def demo_reflexion():
    """演示 Reflexion Loop"""

    def mock_executor(goal: str, strategy: str) -> str:
        """模拟执行：随着尝试次数增加，结果越来越好"""
        attempt = strategy.count("improve") + 1
        if attempt <= 1:
            return f"Attempt 1 result: basic solution for '{goal}'"
        elif attempt <= 2:
            return f"Attempt 2 result: improved solution considering edge cases for '{goal}'"
        else:
            return f"Attempt 3 result: high-quality solution with comprehensive coverage for '{goal}'"

    def mock_evaluator(goal: str, result: str) -> float:
        """模拟评估"""
        if "Attempt 1" in result:
            return 0.4
        elif "Attempt 2" in result:
            return 0.7
        else:
            return 0.95

    def mock_llm(prompt: str) -> str:
        if "REFLECTION" in prompt:
            return (
                "REFLECTION: The first attempt missed edge cases and lacked depth.\n"
                "STRATEGY: improve comprehensiveness, improve edge case handling"
            )
        return "General response"

    rl = ReflexionLoop(
        llm_call=mock_llm,
        executor=mock_executor,
        evaluator=mock_evaluator,
        quality_threshold=0.9,
        max_attempts=5,
    )

    result, state = rl.run(goal="写一篇关于Python异步编程的技术文章")

    print(f"Best Score: {state.best_score:.2f}")
    print(f"Attempts: {len(state.steps)}")
    print(f"Final Result: {result[:100]}...")
    for step in state.steps:
        print(f"  Attempt {step.attempt}: score={step.evaluation_score:.2f}")


if __name__ == "__main__":
    demo_reflexion()
```

**Reflexion 的核心价值**：

1. **渐进改进**：每次反思都积累经验，策略逐步优化
2. **质量保证**：通过评估分数控制输出质量，低于阈值自动重试
3. **可解释性**：每步反思过程都是可审计的，便于调试和优化

---

## 五、三种 Loop 对比

| 维度 | ReAct | Plan-Execute | Reflexion |
|------|-------|-------------|-----------|
| **核心思想** | 边想边做，逐步推理 | 先规划，后执行 | 执行→反思→改进→再执行 |
| **适用场景** | 开放式问答、多步工具调用 | 结构化任务、可分解问题 | 质量敏感型任务、创意写作 |
| **延迟** | 中等（串行推理+执行） | 低（并行执行抵消延迟） | 高（多次执行+反思） |
| **Token 成本** | 中等 | 低（规划一次，执行高效） | 高（每次反思消耗大量推理） |
| **复杂度** | 低 | 中 | 高 |
| **可控性** | 低（LLM自主决策） | 高（计划可审查） | 中（反思不可预测） |
| **失败恢复** | 弱（需重新执行） | 强（Re-plan 失败部分） | 强（反思后改进策略） |
| **适合模型** | 任意 LLM | 需要强推理能力的 LLM | 需要强反思能力的 LLM |
| **典型框架** | LangChain, AutoGPT | LangGraph, DSPy | Reflexion, Self-Refine |
| **最大迭代** | 通常 5~15 步 | 通常 1~3 轮 + 并行子任务 | 通常 3~5 次反思 |

**选型决策树**：
- 任务步骤明确、可分解？→ Plan-Execute
- 需要高质量输出、有评估手段？→ Reflexion
- 通用场景、需要灵活应对？→ ReAct
- 实际生产？→ 通常是三者的混合体

---

## 六、工程实现要点

### 6.1 检查点持久化

```python
import json
import os
from pathlib import Path


class CheckpointStore:
    """检查点持久化存储"""

    def __init__(self, storage_dir: str = "/tmp/agent_checkpoints"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, agent_id: str, state: dict):
        """保存检查点"""
        path = self.storage_dir / f"{agent_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self, agent_id: str) -> Optional[dict]:
        """加载检查点"""
        path = self.storage_dir / f"{agent_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_checkpoints(self) -> list[str]:
        """列出所有检查点"""
        return [p.stem for p in self.storage_dir.glob("*.json")]
```

### 6.2 错误恢复策略

```python
from functools import wraps
import random
import time


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retriable_exceptions: tuple = (Exception,),
):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retriable_exceptions as e:
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    time.sleep(delay)
                    print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
            return None
        return wrapper
    return decorator
```

### 6.3 循环检测（State Hash 去重）

```python
from collections import deque
import hashlib


class LoopDetector:
    """
    Agent 无限循环检测器

    使用滑动窗口 + 状态哈希双重机制检测循环
    """

    def __init__(self, window_size: int = 20, hash_threshold: int = 3):
        self.window_size = window_size
        self.hash_threshold = hash_threshold
        self.recent_hashes: deque = deque(maxlen=window_size)
        self.hash_counts: dict[str, int] = {}

    def check(self, state_hash: str) -> tuple[bool, str]:
        """
        检查是否检测到循环

        Returns:
            (is_loop_detected, reason)
        """
        self.recent_hashes.append(state_hash)

        # 统计哈希出现次数
        self.hash_counts[state_hash] = self.hash_counts.get(state_hash, 0) + 1

        # 检查1: 单一状态重复过多
        if self.hash_counts[state_hash] >= self.hash_threshold:
            return True, f"State {state_hash[:8]}... repeated {self.hash_counts[state_hash]} times"

        # 检查2: 滑动窗口内的循环模式检测
        if len(self.recent_hashes) >= 6:
            # 尝试检测周期性循环（长度 2~5）
            hashes_list = list(self.recent_hashes)
            for period in range(2, min(6, len(hashes_list) // 2 + 1)):
                is_periodic = True
                for i in range(period, min(period * 3, len(hashes_list))):
                    if hashes_list[i] != hashes_list[i - period]:
                        is_periodic = False
                        break
                if is_periodic and len(hashes_list) >= period * 2:
                    return True, f"Periodic loop detected with period={period}"

        return False, ""

    def reset(self):
        self.recent_hashes.clear()
        self.hash_counts.clear()
```

### 6.4 并发安全

```python
import threading
import asyncio
from contextlib import asynccontextmanager


class ThreadSafeAgentState:
    """线程安全的 Agent 状态管理"""

    def __init__(self):
        self._lock = threading.RLock()
        self._iteration = 0
        self._history: list = []

    @property
    def iteration(self) -> int:
        with self._lock:
            return self._iteration

    def add_step(self, step: dict):
        with self._lock:
            self._iteration += 1
            self._history.append(step)

    def get_history_copy(self) -> list:
        """获取历史的深拷贝，避免并发修改"""
        with self._lock:
            return list(self._history)


class AsyncAgentLock:
    """异步 Agent 锁，防止并发执行冲突"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_agents: set = set()

    async def acquire(self, agent_id: str) -> bool:
        if agent_id in self._active_agents:
            return False  # 同一 Agent 不允许并发执行
        await self._lock.acquire()
        self._active_agents.add(agent_id)
        return True

    async def release(self, agent_id: str):
        self._active_agents.discard(agent_id)
        self._lock.release()
```

---

## 七、面试深度：如何设计一个支持中断恢复的 Agent Loop？

**考察点**：状态序列化、检查点机制、优雅降级

### 设计方案

核心思想：**将 Agent Loop 的每一帧状态都视为可序列化的快照**，支持随时保存和恢复。

```python
import pickle
import sqlite3
from dataclasses import asdict
from typing import Any


class ResumableAgentLoop:
    """
    支持中断恢复的 Agent Loop 框架

    关键设计：
    1. 每步都生成可序列化的检查点
    2. 检查点包含完整执行上下文（不仅是状态，还有LLM调用历史）
    3. 恢复时从检查点重建完整上下文，LLM感知到"之前发生了什么"
    """

    def __init__(
        self,
        agent_id: str,
        db_path: str = "agent_checkpoints.db",
    ):
        self.agent_id = agent_id
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 检查点存储"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                agent_id TEXT,
                step INTEGER,
                state BLOB,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (agent_id, step)
            )
        """)
        conn.commit()
        conn.close()

    def save_checkpoint(self, step: int, state: Any, context: str = ""):
        """保存检查点"""
        serialized = pickle.dumps(state)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (agent_id, step, state, context) VALUES (?, ?, ?, ?)",
            (self.agent_id, step, serialized, context),
        )
        conn.commit()
        conn.close()

    def load_latest_checkpoint(self) -> tuple[int, Any]:
        """加载最新检查点，返回 (step, state)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT step, state FROM checkpoints WHERE agent_id = ? ORDER BY step DESC LIMIT 1",
            (self.agent_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return row[0], pickle.loads(row[1])
        return 0, None

    def load_checkpoint_at_step(self, step: int) -> Any:
        """加载指定步骤的检查点"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT state FROM checkpoints WHERE agent_id = ? AND step = ?",
            (self.agent_id, step),
        )
        row = cursor.fetchone()
        conn.close()
        return pickle.loads(row[1]) if row else None

    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """清理旧检查点，只保留最近 N 个"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            DELETE FROM checkpoints
            WHERE agent_id = ? AND step NOT IN (
                SELECT step FROM checkpoints
                WHERE agent_id = ?
                ORDER BY step DESC
                LIMIT ?
            )
        """, (self.agent_id, self.agent_id, keep_last))
        conn.commit()
        conn.close()

    def run_with_resume(
        self,
        initial_prompt: str,
        loop_fn: Any,
        **kwargs,
    ) -> Any:
        """
        从检查点恢复执行

        用法:
            agent = ResumableAgentLoop("my_agent")
            result = agent.run_with_resume("user query", my_loop_function)
        """
        # 尝试恢复
        start_step, saved_state = self.load_latest_checkpoint()

        if saved_state is not None:
            print(f"Resuming from step {start_step}")
            # 重建上下文：让 LLM 知道之前发生了什么
            resume_context = (
                f"[System: This is a resumed execution from step {start_step}. "
                f"Previous context has been restored.]\n"
                f"Original query: {initial_prompt}\n"
                f"Previous state: {saved_state}"
            )
            return loop_fn(resume_context, start_step=start_step, **kwargs)
        else:
            print("Starting fresh execution")
            return loop_fn(initial_prompt, start_step=0, **kwargs)
```

**面试回答要点**：

1. **状态序列化**：Agent 的完整状态（LLM 调用历史、工具结果、中间变量）必须可序列化
2. **增量保存**：不必每次调用 LLM 都保存，可在关键节点（工具调用前后）保存
3. **上下文重建**：恢复时不是简单续接，要让 LLM "知道"之前发生了什么（通过 system prompt 注入）
4. **存储选型**：轻量级用 SQLite，生产用 Redis/S3，支持分布式恢复
5. **清理策略**：旧检查点要定期清理，避免存储爆炸

---

## 八、面试深度：如何检测和避免 Agent 的无限循环？

**考察点**：哈希去重、模式检测、熔断机制、成本控制

### 多层防御策略

```python
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


class CircuItState(Enum):
    CLOSED = "closed"       # 正常执行
    OPEN = "open"           # 熔断，拒绝执行
    HALF_OPEN = "half_open" # 试探性恢复


@dataclass
class LoopGuard:
    """
    Agent Loop 守卫器：多层防御无限循环

    层次1: 状态哈希去重（相同状态不重复执行）
    层次2: 循环模式检测（周期性循环检测）
    层次3: 熔断器（连续异常自动熔断）
    层次4: 资源预算（token/时间/成本上限）
    """

    max_iterations: int = 20
    max_time_seconds: float = 300.0
    max_tokens: int = 100_000
    max_cost_usd: float = 10.0
    hash_dedup_window: int = 50
    hash_repeat_threshold: int = 3
    period_detect_window: int = 30
    circuit_breaker_threshold: int = 5  # 连续失败次数
    circuit_breaker_timeout: float = 60.0  # 熔断恢复等待时间

    # 运行时状态
    _iteration: int = 0
    _start_time: float = 0
    _total_tokens: int = 0
    _total_cost: float = 0.0
    _hash_history: deque = None
    _hash_counts: dict = None
    _circuit_state: CircuItState = CircuItState.CLOSED
    _consecutive_failures: int = 0
    _last_failure_time: float = 0

    def __post_init__(self):
        self._start_time = time.time()
        self._hash_history = deque(maxlen=self.hash_dedup_window)
        self._hash_counts = {}

    def record_step(self, state_hash: str, tokens_used: int = 0, cost: float = 0.0) -> tuple[bool, str]:
        """
        记录一步执行，检查是否触发终止条件

        Returns:
            (should_stop, reason)
        """
        self._iteration += 1
        self._total_tokens += tokens_used
        self._total_cost += cost
        self._hash_history.append(state_hash)
        self._hash_counts[state_hash] = self._hash_counts.get(state_hash, 0) + 1

        # 层次1: 最大迭代
        if self._iteration >= self.max_iterations:
            return True, f"Max iterations ({self.max_iterations}) reached"

        # 层次2: 超时
        if time.time() - self._start_time > self.max_time_seconds:
            return True, f"Timeout ({self.max_time_seconds}s) reached"

        # 层次3: Token 预算
        if self._total_tokens >= self.max_tokens:
            return True, f"Token budget ({self.max_tokens}) exceeded"

        # 层次4: 成本预算
        if self._total_cost >= self.max_cost_usd:
            return True, f"Cost budget (${self.max_cost_usd}) exceeded"

        # 层次5: 状态哈希重复
        if self._hash_counts[state_hash] >= self.hash_repeat_threshold:
            return True, f"State hash repeated {self._hash_counts[state_hash]} times"

        # 层次6: 周期性循环检测
        if self._detect_periodic_loop():
            return True, "Periodic loop pattern detected"

        # 层次7: 熔断器
        if self._circuit_state == CircuItState.OPEN:
            if time.time() - self._last_failure_time > self.circuit_breaker_timeout:
                self._circuit_state = CircuItState.HALF_OPEN
            else:
                return True, "Circuit breaker is OPEN"

        return False, ""

    def record_failure(self):
        """记录失败，更新熔断器"""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        if self._consecutive_failures >= self.circuit_breaker_threshold:
            self._circuit_state = CircuItState.OPEN

    def record_success(self):
        """记录成功，重置熔断器"""
        self._consecutive_failures = 0
        self._circuit_state = CircuItState.CLOSED

    def _detect_periodic_loop(self) -> bool:
        """检测周期性循环"""
        hashes = list(self._hash_history)
        if len(hashes) < 4:
            return False

        for period in range(2, min(6, len(hashes) // 2 + 1)):
            match_count = 0
            for i in range(period, len(hashes)):
                if hashes[i] == hashes[i - period]:
                    match_count += 1
                else:
                    break
            # 如果连续匹配达到周期的2倍，判定为循环
            if match_count >= period * 2:
                return True

        return False

    def get_metrics(self) -> dict:
        """获取当前运行指标"""
        return {
            "iterations": self._iteration,
            "elapsed_seconds": time.time() - self._start_time,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "unique_states": len(self._hash_counts),
            "circuit_state": self._circuit_state.value,
        }
```

**面试回答要点**：

1. **不要只靠单一手段**：单一的最大迭代次数不够，需要多层防御
2. **状态哈希是基础**：但要注意——LLM 的非确定性输出可能导致语义相同但哈希不同的状态
3. **模式检测更智能**：周期性检测（如 A→B→A→B）比简单计数更早发现问题
4. **熔断器保护成本**：连续失败时快速熔断，避免浪费 token 和 API 调用
5. **资源预算兜底**：时间、token、成本三重预算，任何一项超限都立即停止
6. **监控可观察性**：每个终止原因都应记录日志和指标，便于事后分析
7. **优雅降级**：循环检测触发时不应直接报错，而是返回当前最佳结果

---

## 总结

Agent Loop 是 Agent 系统的心跳。三种核心模式各有适用场景：

- **ReAct**：最简单、最通用，适合大多数 Agent 场景
- **Plan-Execute**：结构化、可并行，适合复杂但可分解的任务
- **Reflexion**：高质量输出的保证，适合质量敏感型应用

在工程实现上，**循环检测**、**检查点持久化**、**错误恢复**和**并发安全**是四个绕不开的基础设施。面试中能够从状态机视角分析 Agent Loop，给出多层防御无限循环的方案，以及设计中断恢复机制，是区分"会用框架"和"理解本质"的关键。
