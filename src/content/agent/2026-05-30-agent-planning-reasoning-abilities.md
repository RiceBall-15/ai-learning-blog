---
title: "Agent自主规划与推理能力：CoT/ToT/GoT/任务分解全解析"
description: "深度解析Agent规划能力的推理范式演进（CoT→ToT→GoT）、三大任务分解策略、Plan-and-Execute架构设计，含完整Python实现和面试深度问答"
date: 2026-05-30
author: "RiceBall-15"
category: agent
subCategory: agent-dev
tags: ["CoT", "ToT", "GoT", "任务分解", "Agent规划"]
draft: false
---

## 核心问题

Agent 能不能自主完成复杂任务，**不取决于模型有多强，而取决于它的规划能力有多好**。

一个没有规划能力的 Agent，就像一个拿到复杂项目却直接动手的程序员——要么遗漏关键步骤，要么在错误方向上浪费大量计算资源。而一个有规划能力的 Agent，能像经验丰富的技术负责人一样：先理解目标，拆解任务，评估方案，再逐步执行。

本文系统性地拆解 Agent 规划能力的技术栈：从最基础的推理链（CoT），到树形探索（ToT）、图结构推理（GoT），再到工程实践中最关键的任务分解策略和动态调整机制。

---

## 一、Agent 规划能力的本质：将复杂任务分解为可执行步骤

Agent 的规划能力，本质上是一种**层次化的决策能力**。它需要完成三件事：

1. **理解意图**：把模糊的用户需求转化为明确的目标
2. **路径设计**：找到从当前状态到目标状态的可行路径
3. **资源调度**：决定每一步需要什么工具、什么数据、多少计算资源

```
┌─────────────────────────────────────────────────┐
│              Agent 规划能力架构                    │
├─────────────────────────────────────────────────┤
│  用户意图  ──→  目标解析  ──→  子目标分解         │
│                                    │              │
│                    ┌───────────────┼────────────┐ │
│                    ▼               ▼            ▼ │
│              [子任务1]        [子任务2]     [子任务3]│
│                    │               │            │ │
│                    ▼               ▼            ▼ │
│              [工具调用1]     [工具调用2]   [工具调用3]│
│                    │               │            │ │
│                    └───────┬───────┘────────────┘ │
│                            ▼                      │
│                      结果聚合 & 验证               │
└─────────────────────────────────────────────────┘
```

关键洞察：**规划的粒度决定了执行的质量**。粒度太粗，执行时缺乏指导；粒度太细，规划本身的计算开销会超过直接执行的成本。好的规划是在"可控性"和"开销"之间找到平衡。

---

## 二、Chain-of-Thought (CoT)：线性推理链，基础但重要

CoT 是最基础的推理范式，核心思想是**让模型展示中间推理步骤**，而不是直接给出答案。

### 基础 CoT 实现

```python
class ChainOfThought:
    """链式推理：线性推理链"""
    
    def __init__(self, llm):
        self.llm = llm
        self.chain: list[str] = []
    
    def reason(self, problem: str) -> dict:
        """线性推理：每一步依赖上一步的结果"""
        prompt = f"""请一步步分析以下问题，每一步都要基于前一步的结论。

问题：{problem}

请按以下格式输出：
步骤1: [推理内容]
步骤2: [基于步骤1的推理]
...
结论: [最终答案]"""
        
        response = self.llm.generate(prompt)
        steps = self._parse_steps(response)
        
        return {
            "steps": steps,
            "conclusion": steps[-1] if steps else "",
            "depth": len(steps)
        }
    
    def _parse_steps(self, response: str) -> list[str]:
        """解析推理步骤"""
        lines = response.strip().split("\n")
        steps = []
        for line in lines:
            if line.startswith("步骤") or line.startswith("Step"):
                steps.append(line)
            elif line.startswith("结论") or line.startswith("Conclusion"):
                steps.append(line)
        return steps
```

### CoT 的局限性

CoT 是线性的——每一步只能基于前一步的结论。这意味着：
- **无法回溯**：如果第3步发现第2步的假设错误，无法回到第2步重新推理
- **无法分支**：对于有多种解法的问题，CoT 只能选择一条路径走到底
- **单点故障**：任何一步推理出错，后续所有步骤都会被误导

---

## 三、Tree-of-Thought (ToT)：树形探索，多路径评估选择最优

ToT 解决了 CoT 的核心问题：**在每一步探索多个候选方案，评估后选择最优路径继续**。

### ToT 核心实现

```python
import heapq
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ThoughtNode:
    """思维节点"""
    content: str
    depth: int
    score: float = 0.0
    parent: Optional['ThoughtNode'] = field(default=None)
    children: list['ThoughtNode'] = field(default_factory=list)
    
    def path(self) -> list[str]:
        """获取从根到当前节点的完整路径"""
        node = self
        path = []
        while node:
            path.append(node.content)
            node = node.parent
        return list(reversed(path))
    
    def __lt__(self, other):
        return self.score > other.score  # 最大堆

class TreeOfThought:
    """树形思维：多路径探索与评估"""
    
    def __init__(self, llm, max_depth: int = 5, branching: int = 3,
                 beam_width: int = 2):
        self.llm = llm
        self.max_depth = max_depth
        self.branching = branching  # 每步生成的候选数
        self.beam_width = beam_width  # 保留的最优路径数
    
    def solve(self, problem: str) -> dict:
        """BFS方式的ToT搜索"""
        # 初始化：根节点
        root = ThoughtNode(content=problem, depth=0)
        current_nodes = [root]
        
        for depth in range(1, self.max_depth + 1):
            all_candidates = []
            
            # 对每个当前节点，生成多个候选
            for node in current_nodes:
                candidates = self._generate_thoughts(node, depth)
                all_candidates.extend(candidates)
            
            if not all_candidates:
                break
            
            # 评估所有候选
            for candidate in all_candidates:
                candidate.score = self._evaluate(candidate)
            
            # 保留 beam_width 个最优节点
            current_nodes = sorted(all_candidates, 
                                   key=lambda x: x.score, 
                                   reverse=True)[:self.beam_width]
            
            # 检查是否找到满意解
            for node in current_nodes:
                if self._is_solution(node):
                    return {
                        "solution": node.path(),
                        "score": node.score,
                        "depth": depth,
                        "explored": len(all_candidates)
                    }
        
        # 返回最佳搜索结果
        best = max(current_nodes, key=lambda x: x.score) if current_nodes else root
        return {
            "solution": best.path(),
            "score": best.score,
            "depth": self.max_depth,
            "explored": sum(1 for _ in current_nodes)
        }
    
    def _generate_thoughts(self, parent: ThoughtNode, 
                           depth: int) -> list[ThoughtNode]:
        """为父节点生成多个候选思维"""
        path = parent.path()
        prompt = f"""基于以下推理路径，生成{self.branching}个不同的下一步思路：

推理路径：
{chr(10).join(path)}

请生成{self.branching}个不同方向的思考，用 [1] [2] [3] 标记。"""
        
        response = self.llm.generate(prompt)
        candidates = []
        for i in range(self.branching):
            content = self._extract_candidate(response, i + 1)
            if content:
                candidates.append(
                    ThoughtNode(content=content, depth=depth, parent=parent)
                )
        return candidates
    
    def _evaluate(self, node: ThoughtNode) -> float:
        """评估思维节点的质量（LLM-as-Judge）"""
        path = node.path()
        prompt = f"""评估以下推理路径的质量（0-1分）：

路径：
{chr(10).join(path)}

评估标准：
1. 逻辑连贯性
2. 信息完整性
3. 结论合理性

请只输出一个数字（0-1之间）。"""
        
        response = self.llm.generate(prompt)
        try:
            return float(response.strip())
        except ValueError:
            return 0.5
    
    def _is_solution(self, node: ThoughtNode) -> bool:
        """判断是否为满意解"""
        return node.score > 0.85 and node.depth >= 2
    
    def _extract_candidate(self, response: str, index: int) -> Optional[str]:
        """提取第index个候选"""
        marker = f"[{index}]"
        if marker in response:
            start = response.index(marker) + len(marker)
            end = response.find(f"[{index+1}]", start)
            if end == -1:
                end = len(response)
            return response[start:end].strip()
        return None
```

ToT 的关键优势是**可回溯、可并行评估**，代价是计算开销增加 O(branching) 倍。

---

## 四、Graph-of-Thought (GoT)：图结构推理，支持收敛和循环

GoT 是 ToT 的泛化：节点之间的连接不再限制为树形，而是**任意的有向图**。这意味着：
- **合并（Merge）**：多条推理路径可以汇聚到同一节点
- **循环（Cycle）**：可以进行迭代精化
- **反馈（Feedback）**：后续节点可以修改前序节点的内容

### GoT 核心实现

```python
from collections import defaultdict
from enum import Enum

class OpType(Enum):
    GENERATE = "generate"    # 生成新思维
    AGGREGATE = "aggregate"  # 合并多个思维
    REFINE = "refine"        # 精化已有思维
    SCORE = "score"          # 评分筛选

@dataclass
class GraphNode:
    id: str
    content: str
    score: float = 0.0
    op: OpType = OpType.GENERATE

class GraphOfThought:
    """图结构推理：支持合并、精化、循环"""
    
    def __init__(self, llm, max_iterations: int = 10):
        self.llm = llm
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, list[str]] = defaultdict(list)
        self.max_iterations = max_iterations
        self._node_counter = 0
    
    def _next_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"
    
    def add_node(self, content: str, op: OpType = OpType.GENERATE) -> str:
        node_id = self._next_id()
        self.nodes[node_id] = GraphNode(id=node_id, content=content, op=op)
        return node_id
    
    def add_edge(self, from_id: str, to_id: str):
        self.edges[from_id].append(to_id)
    
    def solve(self, problem: str) -> dict:
        """GoT求解流程"""
        # 1. 初始节点
        root_id = self.add_node(problem)
        
        # 2. 多路径生成
        init_nodes = []
        for _ in range(3):
            new_id = self.add_node(problem, OpType.GENERATE)
            self.add_edge(root_id, new_id)
            self.nodes[new_id].content = self._generate_thought(problem)
            init_nodes.append(new_id)
        
        # 3. 迭代：精化 + 合并 + 评分
        current_frontier = init_nodes
        for iteration in range(self.max_iterations):
            next_frontier = []
            
            for node_id in current_frontier:
                node = self.nodes[node_id]
                
                # 精化操作
                refined_id = self._refine(node)
                next_frontier.append(refined_id)
            
            # 合并操作：选择两个相关节点合并
            if len(next_frontier) >= 2:
                merged_id = self._aggregate(next_frontier[:2])
                next_frontier.append(merged_id)
            
            # 评分筛选
            scored_nodes = []
            for nid in next_frontier:
                score = self._score_node(self.nodes[nid])
                self.nodes[nid].score = score
                if score > 0.7:
                    scored_nodes.append(nid)
            
            if scored_nodes:
                current_frontier = scored_nodes
                # 检查收敛
                if self._check_convergence(scored_nodes):
                    break
        
        # 4. 返回最佳结果
        best_id = max(current_frontier, 
                      key=lambda x: self.nodes[x].score)
        return {
            "solution": self.nodes[best_id].content,
            "score": self.nodes[best_id].score,
            "graph_size": len(self.nodes),
            "iterations": iteration + 1
        }
    
    def _generate_thought(self, context: str) -> str:
        prompt = f"针对以下问题，提出一个创新的解决思路：\n{context}"
        return self.llm.generate(prompt)
    
    def _refine(self, node: GraphNode) -> str:
        new_id = self.add_node("", OpType.REFINE)
        prompt = f"""请精化以下推理，使其更准确、更完整：

原文：{node.content}

请输出精化后的版本。"""
        self.nodes[new_id].content = self.llm.generate(prompt)
        self.add_edge(node.id, new_id)
        return new_id
    
    def _aggregate(self, node_ids: list[str]) -> str:
        new_id = self.add_node("", OpType.AGGREGATE)
        contents = [self.nodes[nid].content for nid in node_ids]
        prompt = f"""请合并以下多个推理视角，形成更全面的结论：

视角1：{contents[0]}
视角2：{contents[1]}

请输出合并后的统一结论。"""
        self.nodes[new_id].content = self.llm.generate(prompt)
        for nid in node_ids:
            self.add_edge(nid, new_id)
        return new_id
    
    def _score_node(self, node: GraphNode) -> float:
        prompt = f"请评估以下内容的质量（0-1分，只输出数字）：\n{node.content}"
        try:
            return float(self.llm.generate(prompt).strip())
        except ValueError:
            return 0.5
    
    def _check_convergence(self, node_ids: list[str]) -> bool:
        """检查评分是否收敛"""
        scores = [self.nodes[nid].score for nid in node_ids]
        if len(scores) < 2:
            return False
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        return variance < 0.01  # 方差小于阈值则收敛
```

GoT 的图结构允许**信息复用和迭代精化**，特别适合需要多视角综合的复杂问题。

---

## 五、任务分解策略

任务分解是 Agent 规划的核心操作。不同的分解策略适用于不同的任务结构。

### 1. 递归分解（分治法）

```python
def recursive_decompose(task: str, llm, max_depth: int = 5, 
                        depth: int = 0) -> dict:
    """递归分解：像编程的分治法"""
    # 基线条件：任务足够简单，直接执行
    prompt = f"""判断以下任务是否可以直接执行（无需再分解）：
任务：{task}

回答 YES 或 NO，然后说明原因。"""
    
    response = llm.generate(prompt)
    is_atomic = response.strip().upper().startswith("YES")
    
    if is_atomic or depth >= max_depth:
        return {"task": task, "type": "atomic", "subtasks": []}
    
    # 分解为子任务
    decompose_prompt = f"""将以下任务分解为2-4个子任务：
任务：{task}

要求：
1. 子任务之间尽量独立
2. 合起来覆盖原任务的所有要求
3. 每个子任务都有明确的输入和输出

输出格式（JSON数组）：["子任务1", "子任务2", ...]"""
    
    response = llm.generate(decompose_prompt)
    subtasks = parse_json_list(response)
    
    # 递归分解每个子任务
    children = []
    for subtask in subtasks:
        child = recursive_decompose(subtask, llm, max_depth, depth + 1)
        children.append(child)
    
    return {
        "task": task,
        "type": "composite",
        "subtasks": children
    }
```

**适用场景**：问题可以自然地分为独立子问题，子问题之间无依赖。

### 2. MapReduce 并行分解

```python
import asyncio
from typing import Callable

class MapReduceDecomposer:
    """MapReduce并行分解：适合可并行处理的大任务"""
    
    def __init__(self, llm, map_fn: Callable = None, 
                 reduce_fn: Callable = None):
        self.llm = llm
        self.map_fn = map_fn or self._default_map
        self.reduce_fn = reduce_fn or self._default_reduce
    
    async def execute(self, task: str, data_chunks: list[str]) -> dict:
        """
        Map阶段：对每个数据块并行执行任务
        Reduce阶段：汇总所有结果
        """
        # Map阶段：并行执行
        map_results = await asyncio.gather(*[
            self.map_fn(task, chunk) for chunk in data_chunks
        ])
        
        # Reduce阶段：汇总
        final_result = await self.reduce_fn(task, map_results)
        
        return {
            "map_results": map_results,
            "final_result": final_result,
            "parallelism": len(data_chunks)
        }
    
    async def _default_map(self, task: str, chunk: str) -> str:
        prompt = f"任务：{task}\n\n数据：{chunk}\n\n请基于数据完成任务。"
        return await self.llm.agenerate(prompt)
    
    async def _default_reduce(self, task: str, results: list[str]) -> str:
        prompt = f"""任务：{task}

以下是各部分的处理结果：
{chr(10).join(f'部分{i+1}: {r}' for i, r in enumerate(results))}

请汇总以上结果，给出统一的最终答案。"""
        return await self.llm.agenerate(prompt)
```

**适用场景**：数据量大但处理逻辑相同，如批量文档分析、多文件代码审查。

### 3. DAG 依赖图分解

```python
from collections import defaultdict, deque

class DAGDecomposer:
    """DAG分解：基于依赖关系的任务调度"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tasks: dict[str, dict] = {}
        self.dependencies: dict[str, set[str]] = defaultdict(set)
        self.dependents: dict[str, set[str]] = defaultdict(set)
    
    def analyze_and_build(self, complex_task: str) -> dict:
        """分析复杂任务，构建DAG"""
        prompt = f"""分析以下复杂任务，识别所有子任务及其依赖关系：

任务：{complex_task}

输出JSON格式：
{{
  "tasks": {{
    "task_id": {{
      "description": "任务描述",
      "dependencies": ["依赖的task_id"],
      "type": "search|analyze|generate|validate"
    }}
  }}
}}"""
        
        response = self.llm.generate(prompt)
        task_graph = parse_json(response)
        
        # 构建依赖图
        for tid, info in task_graph["tasks"].items():
            self.tasks[tid] = info
            for dep in info.get("dependencies", []):
                self.dependencies[tid].add(dep)
                self.dependents[dep].add(tid)
        
        return self._topological_sort()
    
    def _topological_sort(self) -> dict:
        """拓扑排序：确定执行顺序"""
        in_degree = defaultdict(int)
        for tid in self.tasks:
            in_degree[tid] = len(self.dependencies[tid])
        
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        levels = []
        
        while queue:
            level = list(queue)
            levels.append(level)
            next_queue = deque()
            for tid in level:
                for dep in self.dependents[tid]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        next_queue.append(dep)
            queue = next_queue
        
        return {
            "levels": levels,           # 每层可并行执行
            "total_tasks": len(self.tasks),
            "max_parallel": max(len(l) for l in levels) if levels else 0,
            "critical_path": self._find_critical_path()
        }
    
    def _find_critical_path(self) -> list[str]:
        """找出关键路径（最长路径）"""
        # 简化实现：找最长的依赖链
        def dfs(tid):
            if not self.dependents[tid]:
                return [tid]
            longest = []
            for dep in self.dependents[tid]:
                path = dfs(dep)
                if len(path) > len(longest):
                    longest = path
            return [tid] + longest
        
        # 从入度为0的节点开始
        roots = [tid for tid in self.tasks 
                 if len(self.dependencies[tid]) == 0]
        if not roots:
            return []
        
        best_path = []
        for root in roots:
            path = dfs(root)
            if len(path) > len(best_path):
                best_path = path
        return best_path
    
    def get_execution_plan(self) -> list[dict]:
        """生成可执行的调度计划"""
        result = self.analyze_and_build("")  # 假设已分析
        plan = []
        
        for level_idx, level in enumerate(result["levels"]):
            parallel_tasks = []
            for tid in level:
                parallel_tasks.append({
                    "id": tid,
                    "description": self.tasks[tid]["description"],
                    "type": self.tasks[tid]["type"]
                })
            plan.append({
                "step": level_idx + 1,
                "parallel": parallel_tasks,
                "can_parallelize": len(parallel_tasks) > 1
            })
        
        return plan
```

**适用场景**：任务之间有复杂的前后依赖，如软件构建流水线、多步数据分析。

三种分解策略对比：

| 策略 | 结构 | 适用场景 | 复杂度 |
|------|------|----------|--------|
| 递归分解 | 树 | 独立子问题 | O(n) |
| MapReduce | 星形 | 同构数据处理 | O(n/p) |
| DAG | 有向无环图 | 有依赖关系的复合任务 | O(V+E) |

---

## 六、目标规划：从高层目标到可执行子目标的映射

目标规划的关键是**建立层次化的目标树**，将抽象的意图逐层细化为可执行的操作。

```python
class GoalPlanner:
    """层次化目标规划器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def plan(self, user_goal: str) -> dict:
        """从用户目标生成分层规划"""
        
        # Level 1: 目标解析
        objectives = self._decompose_goal(user_goal)
        
        # Level 2: 为每个目标生成子目标
        full_plan = {}
        for obj in objectives:
            sub_goals = self._decompose_objective(obj)
            full_plan[obj["name"]] = sub_goals
        
        # Level 3: 为每个子目标匹配动作
        executable_plan = {}
        for obj_name, sub_goals in full_plan.items():
            executable_plan[obj_name] = []
            for sg in sub_goals:
                actions = self._plan_actions(sg)
                executable_plan[obj_name].append({
                    "sub_goal": sg["name"],
                    "actions": actions
                })
        
        return {
            "goal": user_goal,
            "objective_count": len(objectives),
            "total_sub_goals": sum(len(v) for v in full_plan.values()),
            "plan": executable_plan
        }
    
    def _decompose_goal(self, goal: str) -> list[dict]:
        prompt = f"""将以下高层目标分解为2-5个主要目标：

用户目标：{goal}

输出JSON数组：
[{{"name": "目标名称", "description": "目标描述", "priority": "high|medium|low"}}]"""
        
        response = self.llm.generate(prompt)
        return parse_json(response)
    
    def _decompose_objective(self, objective: dict) -> list[dict]:
        prompt = f"""将以下目标分解为可执行的子目标：

目标：{objective["name"]}
描述：{objective["description"]}

要求：每个子目标应该是一个具体的、可验证的动作。

输出JSON数组：
[{{"name": "子目标名称", "success_criteria": "成功标准", "estimated_steps": 步数}}]"""
        
        response = self.llm.generate(prompt)
        return parse_json(response)
    
    def _plan_actions(self, sub_goal: dict) -> list[dict]:
        prompt = f"""为以下子目标规划具体动作：

子目标：{sub_goal["name"]}
成功标准：{sub_goal["success_criteria"]}

输出JSON数组：
[{{"action": "动作描述", "tool": "需要的工具", "input": "输入要求"}}]"""
        
        response = self.llm.generate(prompt)
        return parse_json(response)
```

---

## 七、规划与执行的解耦：Plan-and-Execute 架构

Plan-and-Execute 的核心思想：**先生成完整计划，再逐步执行**。规划器和执行器可以是不同的模型，甚至不同的系统。

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
    id: int
    description: str
    tool: str = ""
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    depends_on: list[int] = field(default_factory=list)

class PlanAndExecute:
    """规划与执行分离架构"""
    
    def __init__(self, planner_llm, executor_llm, tools: dict):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools
        self.plan: list[PlanStep] = []
        self.max_retries = 3
    
    def run(self, goal: str) -> dict:
        # Phase 1: 规划（可以使用更强大的模型）
        self.plan = self._create_plan(goal)
        
        # Phase 2: 执行（可以使用更便宜的模型）
        execution_log = []
        
        for step in self.plan:
            # 检查依赖是否满足
            if not self._dependencies_met(step):
                step.status = StepStatus.SKIPPED
                execution_log.append(f"跳过步骤{step.id}: 依赖未满足")
                continue
            
            # 执行步骤
            success = self._execute_step(step)
            execution_log.append(
                f"步骤{step.id}: {'✓' if success else '✗'} - {step.description}"
            )
            
            # 失败处理
            if not success:
                recovery = self._handle_failure(step)
                if not recovery:
                    break  # 无法恢复，终止
        
        return {
            "plan": [(s.id, s.description, s.status.value) for s in self.plan],
            "execution_log": execution_log,
            "completed": all(s.status == StepStatus.COMPLETED 
                           for s in self.plan)
        }
    
    def _create_plan(self, goal: str) -> list[PlanStep]:
        prompt = f"""为以下目标创建详细的执行计划：

目标：{goal}

请规划每个步骤，包括：
1. 步骤描述
2. 需要使用的工具
3. 步骤之间的依赖关系

输出JSON数组：
[{{"id": 1, "description": "步骤描述", "tool": "工具名", "depends_on": []}}]"""
        
        response = self.planner.generate(prompt)
        steps_data = parse_json(response)
        
        return [PlanStep(**s) for s in steps_data]
    
    def _dependencies_met(self, step: PlanStep) -> bool:
        for dep_id in step.depends_on:
            dep = next((s for s in self.plan if s.id == dep_id), None)
            if not dep or dep.status != StepStatus.COMPLETED:
                return False
        return True
    
    def _execute_step(self, step: PlanStep) -> bool:
        step.status = StepStatus.RUNNING
        
        for attempt in range(self.max_retries):
            try:
                prompt = f"""执行以下步骤：
步骤：{step.description}
工具：{step.tool}

请调用合适的工具完成此步骤。"""
                
                result = self.executor.generate(prompt)
                step.result = result
                step.status = StepStatus.COMPLETED
                return True
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    step.status = StepStatus.FAILED
                    step.result = f"错误: {str(e)}"
                    return False
    
    def _handle_failure(self, failed_step: PlanStep) -> bool:
        """失败时尝试调整计划"""
        prompt = f"""执行计划中的步骤{failed_step.id}失败了：

失败步骤：{failed_step.description}
错误信息：{failed_step.result}

当前剩余计划：
{chr(10).join(f'{s.id}: {s.description} [{s.status.value}]' for s in self.plan if s.status == StepStatus.PENDING)}

请建议如何调整计划以继续完成目标。"""
        
        response = self.planner.generate(prompt)
        # 解析调整建议，修改剩余计划
        return self._adjust_plan(response)
    
    def _adjust_plan(self, adjustment: str) -> bool:
        """根据LLM建议调整计划"""
        # 解析调整方案并修改self.plan
        return True  # 简化实现
```

**Plan-and-Execute 的优势**：
- **规划可以用大模型**（贵但强），**执行可以用小模型**（便宜但快）
- **可重规划**：执行失败时，只修改受影响的后续步骤
- **可缓存**：相似目标的规划可以复用

---

## 八、规划质量评估：如何判断一个规划方案的好坏

好的规划需要满足以下标准：

```python
class PlanEvaluator:
    """规划质量评估器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate(self, goal: str, plan: list[dict]) -> dict:
        """多维度评估规划质量"""
        scores = {}
        
        # 1. 完整性：计划是否覆盖了目标的所有要求
        scores["completeness"] = self._eval_completeness(goal, plan)
        
        # 2. 可行性：每一步是否真的可以执行
        scores["feasibility"] = self._eval_feasibility(plan)
        
        # 3. 效率：是否存在冗余步骤
        scores["efficiency"] = self._eval_efficiency(plan)
        
        # 4. 鲁棒性：对失败的容错能力
        scores["robustness"] = self._eval_robustness(plan)
        
        # 综合评分
        weights = {"completeness": 0.35, "feasibility": 0.3,
                   "efficiency": 0.2, "robustness": 0.15}
        overall = sum(scores[k] * weights[k] for k in weights)
        
        return {
            "scores": scores,
            "overall": overall,
            "grade": self._grade(overall),
            "recommendations": self._get_recommendations(scores, plan)
        }
    
    def _eval_completeness(self, goal: str, plan: list[dict]) -> float:
        prompt = f"""评估以下计划是否完整覆盖了目标需求：

目标：{goal}
计划步骤：
{chr(10).join(f'步骤{i+1}: {s.get("description", "")}' for i, s in enumerate(plan))}

是否遗漏了什么？评分（0-1）："""
        return float(self.llm.generate(prompt).strip())
    
    def _eval_feasibility(self, plan: list[dict]) -> float:
        prompt = f"""评估以下计划的每一步是否实际可行：

计划步骤：
{chr(10).join(f'步骤{i+1}: {s.get("description", "")}' for i, s in enumerate(plan))}

评估：步骤是否依赖了不存在的工具？是否超出了模型能力？
评分（0-1）："""
        return float(self.llm.generate(prompt).strip())
    
    def _eval_efficiency(self, plan: list[dict]) -> float:
        prompt = f"""评估以下计划的执行效率：

计划步骤：
{chr(10).join(f'步骤{i+1}: {s.get("description", "")}' for i, s in enumerate(plan))}

是否存在可以合并的步骤？是否存在不必要的步骤？
评分（0-1）："""
        return float(self.llm.generate(prompt).strip())
    
    def _eval_robustness(self, plan: list[dict]) -> float:
        prompt = f"""评估以下计划的鲁棒性：

计划步骤：
{chr(10).join(f'步骤{i+1}: {s.get("description", "")}' for i, s in enumerate(plan))}

如果某一步失败了，计划能否继续？是否有重试/回退机制？
评分（0-1）："""
        return float(self.llm.generate(prompt).strip())
    
    def _grade(self, score: float) -> str:
        if score >= 0.85: return "A (优秀)"
        if score >= 0.70: return "B (良好)"
        if score >= 0.55: return "C (一般)"
        return "D (需要改进)"
    
    def _get_recommendations(self, scores: dict, plan: list[dict]) -> list[str]:
        recs = []
        if scores["completeness"] < 0.7:
            recs.append("计划可能遗漏了部分目标要求，建议补充")
        if scores["feasibility"] < 0.7:
            recs.append("部分步骤可行性存疑，建议简化或拆分")
        if scores["efficiency"] < 0.7:
            recs.append("存在可优化的冗余步骤")
        if scores["robustness"] < 0.7:
            recs.append("建议为关键步骤添加失败处理逻辑")
        return recs
```

---

## 九、面试深度：如何让 Agent 在规划失败时自动调整策略？

这是面试中高频考察的系统设计问题。

### 核心答案框架

**三个层次的调整机制：**

**第一层：步骤级重试（最简单）**
```python
class StepRetryStrategy:
    def __init__(self, llm, max_retries=3):
        self.llm = llm
        self.max_retries = max_retries
    
    def execute_with_retry(self, step: str, context: str) -> dict:
        for attempt in range(self.max_retries):
            try:
                result = self._try_execute(step, context)
                return {"success": True, "result": result, "attempts": attempt + 1}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # 第2次：简化任务
                    # 第3次：换工具/换思路
                    step = self._simplify_or_redirect(step, e)
        return {"success": False, "error": "所有重试均失败"}
    
    def _simplify_or_redirect(self, step: str, error) -> str:
        prompt = f"""执行以下步骤失败了：
步骤：{step}
错误：{error}

请生成一个更简单的替代方案来达到相同目的。"""
        return self.llm.generate(prompt)
```

**第二层：规划级重规划（Plan Repair）**

当多个步骤失败或整体计划不再可行时，触发重规划：
```python
class PlanRepair:
    def __init__(self, planner_llm, original_goal: str):
        self.planner = planner_llm
        self.goal = original_goal
    
    def repair(self, current_plan: list, completed: list, 
               failed_step: str, error: str) -> list:
        """基于当前执行状态重新规划"""
        prompt = f"""原始目标：{self.goal}

已完成的步骤：{completed}
失败的步骤：{failed_step}
失败原因：{error}

当前计划中尚未执行的步骤：
{[s for s in current_plan if s not in completed]}

请基于以上信息，重新规划剩余步骤，确保能完成原始目标。
只输出需要修改的部分。"""
        
        new_plan = self.planner.generate(prompt)
        return self._merge_plans(completed, parse_json(new_plan))
```

**第三层：策略级切换（Meta-Strategy）**

```python
class MetaStrategy:
    """策略级调整：当一种推理范式失效时，切换到另一种"""
    
    STRATEGIES = ["cot", "tot", "got", "react", "reflexion"]
    
    def __init__(self, llm):
        self.llm = llm
        self.current_strategy = "cot"
        self.performance_history = []
    
    def select_strategy(self, task: str, context: dict) -> str:
        """根据任务特征选择最佳策略"""
        prompt = f"""任务：{task}
任务特征：复杂度={context.get('complexity', 'medium')}, 
         类型={context.get('type', 'general')}

历史上各策略的表现：
{self._format_history()}

请选择最适合此任务的推理策略。"""
        
        selected = self.llm.generate(prompt)
        self.current_strategy = selected.strip().lower()
        return self.current_strategy
    
    def on_strategy_failure(self, strategy: str, error: str):
        """策略失败时的调整逻辑"""
        self.performance_history.append({
            "strategy": strategy,
            "success": False,
            "error": error
        })
        # 切换到下一个策略
        idx = (self.STRATEGIES.index(strategy) + 1) % len(self.STRATEGIES)
        self.current_strategy = self.STRATEGIES[idx]
    
    def _format_history(self) -> str:
        return "\n".join(
            f"  {h['strategy']}: {'成功' if h['success'] else '失败'}"
            for h in self.performance_history[-5:]
        )
```

---

## 十、面试深度：如何设计一个支持动态调整的规划系统？

这是系统设计级别的问题，需要展示架构思维。

### 核心架构：三环反馈模型

```
┌─────────────────────────────────────────────┐
│           动态调整规划系统架构                 │
├─────────────────────────────────────────────┤
│                                             │
│   ┌─────────┐    ┌──────────┐    ┌────────┐│
│   │  感知层  │───→│  规划层   │───→│ 执行层 ││
│   │Perception│    │ Planning │    │Execution││
│   └────┬────┘    └────┬─────┘    └───┬────┘│
│        │              │              │      │
│        └──────────────┼──────────────┘      │
│                       ▼                     │
│              ┌────────────────┐             │
│              │   监控与评估层   │             │
│              │   Monitor      │             │
│              └────────────────┘             │
│                     │                       │
│         ┌───────────┼───────────┐          │
│         ▼           ▼           ▼          │
│   [异常检测]   [进度跟踪]   [质量评估]     │
│         │           │           │          │
│         └───────────┼───────────┘          │
│                     ▼                      │
│              ┌────────────────┐            │
│              │   调整决策器    │            │
│              │  RePlanner     │            │
│              └────────────────┘            │
│                     │                      │
│         ┌───────────┼───────────┐         │
│         ▼           ▼           ▼         │
│   [步骤级修复]  [规划重排]  [策略切换]    │
│                                             │
└─────────────────────────────────────────────┘
```

### 完整实现

```python
import time
from typing import Optional

@dataclass
class ExecutionState:
    """执行状态快照"""
    completed_steps: list[int]
    current_step: int
    results: dict[int, str]
    errors: list[str]
    start_time: float
    elapsed: float

class DynamicPlanner:
    """支持动态调整的规划系统"""
    
    def __init__(self, llm, tools: dict, 
                 checkpoint_interval: int = 3):
        self.llm = llm
        self.tools = tools
        self.checkpoint_interval = checkpoint_interval
        self.plan: list[dict] = []
        self.state = ExecutionState(
            completed_steps=[], current_step=0,
            results={}, errors=[], start_time=time.time(),
            elapsed=0
        )
        self.history: list[dict] = []  # 执行历史，用于反思
    
    def run(self, goal: str, max_iterations: int = 20) -> dict:
        """主循环：执行-监控-调整"""
        # 初始规划
        self.plan = self._create_plan(goal)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 1. 执行当前步骤
            step = self._get_current_step()
            if not step:
                break  # 计划已完成
            
            result = self._execute_step(step)
            
            # 2. 监控：检查执行质量
            monitor_result = self._monitor(step, result)
            
            # 3. 根据监控结果决定是否调整
            if monitor_result["needs_adjustment"]:
                adjustment = self._decide_adjustment(
                    monitor_result, self.state
                )
                self._apply_adjustment(adjustment)
            
            # 4. 定期检查点：验证整体方向
            if iteration % self.checkpoint_interval == 0:
                self._checkpoint(goal)
        
        return self._finalize(goal)
    
    def _create_plan(self, goal: str) -> list[dict]:
        prompt = f"""为以下目标创建执行计划：

目标：{goal}

要求：
1. 每步有明确的输入/输出
2. 标注每步使用的工具
3. 标注步骤间的依赖

输出JSON数组：[{{"id": 0, "desc": "...", "tool": "...", "deps": []}}]"""
        return parse_json(self.llm.generate(prompt))
    
    def _execute_step(self, step: dict) -> str:
        """执行单个步骤"""
        # 收集依赖步骤的结果
        dep_results = {
            dep: self.state.results.get(dep, "")
            for dep in step.get("deps", [])
        }
        
        prompt = f"""执行以下步骤：

步骤描述：{step['desc']}
可用工具：{list(self.tools.keys())}
前置步骤结果：{dep_results}

请执行此步骤并返回结果。"""
        
        result = self.llm.generate(prompt)
        self.state.results[step["id"]] = result
        self.state.completed_steps.append(step["id"])
        return result
    
    def _monitor(self, step: dict, result: str) -> dict:
        """监控执行质量"""
        prompt = f"""监控以下步骤的执行结果：

步骤：{step['desc']}
结果：{result}

评估：
1. 结果是否符合预期？(yes/no)
2. 是否存在明显错误？(yes/no)
3. 是否需要调整后续计划？(yes/no)

JSON格式输出：{{"quality": 0-1, "needs_adjustment": bool, "issues": [...]}}"""
        
        response = self.llm.generate(prompt)
        return parse_json(response)
    
    def _decide_adjustment(self, monitor_result: dict, 
                          state: ExecutionState) -> dict:
        """决定调整策略"""
        issues = monitor_result.get("issues", [])
        quality = monitor_result.get("quality", 0.5)
        
        if quality < 0.3:
            # 严重问题：重规划
            return {"type": "replan", "reason": "质量过低"}
        elif quality < 0.6:
            # 中等问题：重排后续步骤
            return {"type": "reorder", "reason": "需要优化"}
        elif issues:
            # 轻微问题：调整具体步骤
            return {"type": "adjust_step", "issues": issues}
        else:
            return {"type": "none"}
    
    def _apply_adjustment(self, adjustment: dict):
        """应用调整"""
        adj_type = adjustment["type"]
        
        if adj_type == "replan":
            self.plan = self._replan(adjustment)
        elif adj_type == "reorder":
            self.plan = self._reorder(adjustment)
        elif adj_type == "adjust_step":
            self._adjust_current_step(adjustment)
    
    def _replan(self, adjustment: dict) -> list[dict]:
        """重规划：保留已完成部分，重新规划剩余"""
        completed_ids = set(self.state.completed_steps)
        
        prompt = f"""基于以下执行历史，重新规划剩余步骤：

已完成的步骤及结果：
{chr(10).join(f'步骤{k}: {v[:100]}' for k, v in self.state.results.items())}

失败/问题：{adjustment.get('reason', '质量不足')}

请重新规划剩余需要执行的步骤，尽量复用已有结果。"""
        
        new_plan = parse_json(self.llm.generate(prompt))
        return new_plan
    
    def _reorder(self, adjustment: dict) -> list[dict]:
        """重排：优化后续步骤顺序"""
        # 简化实现：跳过已完成的，返回剩余步骤
        completed = set(self.state.completed_steps)
        return [s for s in self.plan if s["id"] not in completed]
    
    def _adjust_current_step(self, adjustment: dict):
        """调整当前步骤"""
        issues = adjustment.get("issues", [])
        prompt = f"""当前步骤存在问题：{issues}

请生成一个改进版本的步骤描述。"""
        improved = self.llm.generate(prompt)
        # 找到并更新当前步骤
        current = self._get_current_step()
        if current:
            current["desc"] = improved
    
    def _get_current_step(self) -> Optional[dict]:
        """获取下一个待执行的步骤"""
        completed = set(self.state.completed_steps)
        for step in self.plan:
            if step["id"] not in completed:
                # 检查依赖是否满足
                deps = set(step.get("deps", []))
                if deps.issubset(completed):
                    return step
        return None
    
    def _checkpoint(self, goal: str):
        """定期检查点：验证整体方向"""
        prompt = f"""检查当前执行状态：

原始目标：{goal}
已完成步骤：{self.state.completed_steps}
已耗时：{time.time() - self.state.start_time:.0f}秒

当前进度是否合理？是否需要调整整体策略？"""
        
        self.history.append({
            "time": time.time() - self.state.start_time,
            "completed": len(self.state.completed_steps),
            "total": len(self.plan),
            "assessment": self.llm.generate(prompt)
        })
    
    def _finalize(self, goal: str) -> dict:
        """生成最终报告"""
        return {
            "goal": goal,
            "total_steps": len(self.plan),
            "completed": len(self.state.completed_steps),
            "results": self.state.results,
            "checkpoints": self.history,
            "elapsed": time.time() - self.state.start_time
        }
```

### 面试答题要点

1. **展示架构思维**：三环模型（感知→规划→执行）+ 反馈环路
2. **体现层次化设计**：步骤级修复 → 规划级重排 → 策略级切换
3. **强调工程实践**：检查点机制、执行历史、成本控制
4. **给出具体数字**：如"最多重试3次"、"每3步做一次检查点"

---

## 总结

Agent 的规划能力不是一个"有或没有"的问题，而是一个**能力谱系**：

| 推理范式 | 信息流向 | 适用场景 | 计算开销 |
|----------|----------|----------|----------|
| CoT | 线性链 | 简单推理 | O(n) |
| ToT | 树形展开 | 有多种解法的决策 | O(n × b) |
| GoT | 任意图 | 多视角综合问题 | O(V + E) |

任务分解策略则是规划的**工程实现**：递归分解适合独立子问题，MapReduce 适合同构数据处理，DAG 适合有依赖关系的复杂流程。

而 Plan-and-Execute 架构和动态调整机制，则是将规划从"一次性"升级为"持续性"的关键。真正强大的 Agent，不是规划一次就完美执行的，而是**能在执行过程中不断修正自己的规划**。

这就是 Agent 规划能力的终极形态：**不是做出最好的计划，而是拥有在计划失败时找到新路径的能力**。
