---
title: "Agent架构实战指南：六种主流架构的代码实现与选型决策树"
description: "从ReAct到LATS，六种Agent架构的核心实现代码、选型决策树、生产演进路径，附面试回答框架，一篇搞定Agent架构从理论到实战"
date: 2026-05-31
author: 'RiceBall-15'
category: interview
subCategory: orchestration
tags: ['Agent架构', '代码实现', '架构选型', '面试']
draft: false
---

# Agent架构实战指南：六种主流架构的代码实现与选型决策树

> 这是面试方向「各类Agent架构」的深度补充文章。上一篇讲清楚了每种架构的原理和对比，这一篇聚焦**怎么用、怎么选、怎么落地**。

## §1 架构选型决策树：什么场景用什么架构

选架构不是拍脑袋，而是一个系统化的决策过程。以下是我在实际项目中总结的决策流程：

```
你的Agent要解决什么问题？
│
├── 需要多步推理 + 工具调用？
│   ├── 任务复杂度中等、步骤顺序依赖
│   │   └── ✅ ReAct（最简单，上手最快）
│   │
│   ├── 任务对准确率要求高、允许重试
│   │   └── ✅ Reflexion（加反思循环）
│   │
│   ├── 任务可以分解为独立子任务
│   │   ├── 子任务之间有依赖（DAG）
│   │   │   └── ✅ LLMCompiler（并行调度）
│   │   │
│   │   └── 需要全局规划 + 分步执行
│   │       └── ✅ Plan-and-Execute（规划与执行分离）
│   │
│   └── 搜索空间大、需要探索多条路径
│       └── ✅ LATS（树搜索 + 反思）
│
├── 需要让LLM自己学会什么时候用工具？
│   └── ✅ ToolFormer（训练/微调阶段）
│
└── 不确定？从ReAct开始，按需升级
```

**一句话原则**：80%的场景用ReAct就够了。复杂度不够就不要上复杂架构。

### 六种架构速览对比表

| 架构 | 核心机制 | LLM调用模式 | 并行能力 | 反思能力 | 实现复杂度 | 适用场景 |
|------|---------|------------|---------|---------|-----------|---------|
| ReAct | 思考-行动循环 | 串行 | 无 | 无 | ⭐ | 通用问答、工具调用 |
| Reflexion | 反思+重试 | 串行+回溯 | 无 | ✅ 强 | ⭐⭐ | 代码生成、推理纠错 |
| Plan-and-Execute | 规划器+执行器 | 分阶段 | 子任务级 | 弱 | ⭐⭐ | 复杂多步任务 |
| LLMCompiler | DAG并行调度 | 并行+串行 | ✅ 强 | 无 | ⭐⭐⭐ | 多工具并行调用 |
| ToolFormer | 自主工具学习 | 内嵌调用 | N/A | 无 | ⭐⭐⭐⭐ | 训练阶段优化 |
| LATS | 树搜索+反思 | 多分支探索 | 分支级 | ✅ 强 | ⭐⭐⭐⭐ | 复杂规划、探索性任务 |

### 选型决策辅助表

| 场景关键词 | 推荐架构 | 理由 |
|-----------|---------|------|
| 「帮我查一下...然后...」| ReAct | 简单多步，串行足够 |
| 「写一个程序，确保能通过测试」| Reflexion | 需要反复验证和修正 |
| 「分析这个项目并制定计划」| Plan-and-Execute | 先规划再执行的典型场景 |
| 「同时调用这5个API获取数据」| LLMCompiler | 并行调用是核心需求 |
| 「在代码库中找到最佳修改方案」| LATS | 需要探索多个候选方案 |
| 「让模型自己决定什么时候查天气」| ToolFormer | 工具使用的自主决策 |

---

## §2 ReAct架构实战：核心循环实现

ReAct是最基础也最常用的Agent架构。它的核心是一个无限循环：思考 → 行动 → 观察 → 再思考。

### 核心循环的Python实现

这段代码展示了ReAct循环的本质——没有框架依赖，纯逻辑实现：

```python
class ReActAgent:
    def __init__(self, llm, tools: dict, max_steps: int = 10):
        self.llm = llm
        self.tools = tools  # {"tool_name": callable}
        self.max_steps = max_steps

    def run(self, query: str) -> str:
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        for step in range(self.max_steps):
            # 1. LLM 生成 Thought + Action
            response = self.llm.chat(messages)
            action = self._parse_action(response)

            # 2. 终止条件：LLM 给出最终答案
            if action["type"] == "finish":
                return action["answer"]

            # 3. 执行工具，获取观察结果
            tool_fn = self.tools.get(action["tool"])
            if tool_fn is None:
                observation = f"Error: tool '{action['tool']}' not found"
            else:
                observation = tool_fn(**action["args"])

            # 4. 将 Thought + Action + Observation 追加到上下文
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        return "达到最大步骤数，任务未完成"

    def _parse_action(self, text: str) -> dict:
        """解析 LLM 输出中的 Thought 和 Action"""
        import re
        thought = re.search(r'Thought:\s*(.+?)(?:\n|$)', text)
        action_match = re.search(r'Action:\s*(\w+)\((.+?)\)', text)

        if not action_match:
            # 没有找到 Action，说明 LLM 直接给答案
            return {"type": "finish", "answer": text}

        return {
            "type": "tool",
            "thought": thought.group(1) if thought else "",
            "tool": action_match.group(1),
            "args": self._parse_args(action_match.group(2))
        }
```

### ReAct循环的关键设计点

| 设计点 | 选择 | 说明 |
|-------|------|------|
| 终止机制 | LLM显式输出 `finish` | 比固定轮次更灵活 |
| 错误处理 | 工具报错作为 Observation 返回 | Agent 可以自我修正 |
| 上下文管理 | 全量历史追加 | 简单但要注意 token 上限 |
| 最大步数 | 必须设上限 | 防止无限循环，通常 5-15 步 |

### ReAct的致命弱点

ReAct最大的问题是**没有全局视角**——每一步决策只基于历史，不考虑未来。举个例子：

```
用户问：帮我订明天北京到上海的机票，最便宜的

ReAct的实际执行：
Step 1: Thought → 查航班 → 查到5个航班
Step 2: Thought → 查价格 → 找到最便宜的
Step 3: Thought → 订票 → 订完了
看起来没问题？但如果 Step 1 查的是"北京到广州"的航班呢？
ReAct不会回头修正，它只能基于当前观察继续走。
```

这就是为什么我们需要 Reflexion。

---

## §3 Reflexion架构实战：加入自我反思

Reflexion在ReAct的基础上增加了一个关键环节：**在任务失败后，让Agent反思原因，然后带着反思结果重新尝试**。

### 核心思想

```
┌──────────────────────────────────────────────────┐
│                  Reflexion 循环                    │
│                                                   │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │  执行     │───▶│  评估     │───▶│  反思     │  │
│   │ (ReAct)  │    │(通过？)   │    │ (总结教训) │  │
│   └──────────┘    └──────────┘    └──────────┘  │
│        ▲              │               │          │
│        │           通过│           不通过│          │
│        │              ▼               │          │
│        │          返回结果             │          │
│        └───────────────┴───────────────┘          │
│              (带着反思重新执行)                     │
└──────────────────────────────────────────────────┘
```

### 实现代码

```python
class ReflexionAgent:
    def __init__(self, llm, tools: dict, max_retries: int = 3):
        self.llm = llm
        self.tools = tools
        self.max_retries = max_retries
        self.memories: list[str] = []  # 历次反思的记忆

    def run(self, query: str, validator) -> str:
        for attempt in range(self.max_retries):
            # 1. 带着反思记忆执行 ReAct
            result = self._execute_react(query)

            # 2. 验证结果
            is_valid, feedback = validator(result)

            if is_valid:
                return result

            # 3. 反思：分析失败原因
            reflection = self._reflect(query, result, feedback)
            self.memories.append(reflection)

            print(f"Attempt {attempt + 1} failed. Reflection: {reflection}")

        return f"经过 {self.max_retries} 次尝试仍未成功。最后结果: {result}"

    def _execute_react(self, query: str) -> str:
        """将反思记忆注入到 ReAct 的系统提示中"""
        memory_text = "\n".join(
            f"经验 {i+1}: {m}" for i, m in enumerate(self.memories)
        )
        system_prompt = f"""你是一个能够从错误中学习的AI助手。
之前尝试中总结的经验：
{memory_text if memory_text else "无"}

请利用这些经验避免重复犯错。"""

        agent = ReActAgent(self.llm, self.tools)
        messages = [{"role": "system", "content": system_prompt}]
        return agent.run(query)

    def _reflect(self, query, result, feedback) -> str:
        prompt = f"""分析以下任务执行的失败原因：
任务：{query}
执行结果：{result}
反馈：{feedback}

请用一句话总结失败的核心原因和改进建议："""

        return self.llm.chat([
            {"role": "user", "content": prompt}
        ])
```

### Reflexion的杀手锏：为什么它对代码生成特别有效

```
场景：写一个函数，让测试用例通过

ReAct:   写代码 → 测试失败 → 写新代码 → 测试失败 → ...
         （每次都是从零开始猜，没有记忆）

Reflexion: 写代码 → 测试失败 → 反思："我忘了处理空列表的情况"
           → 写新代码（记得处理空列表） → 测试通过 ✅
```

Reflexion的反思记忆是**跨尝试持久化**的，这让Agent能真正"学到东西"。

---

## §4 Plan-and-Execute实战：规划器+执行器分离

Plan-and-Execute的核心洞察：**人类做事也是先想清楚再动手**。把规划和执行分开，各司其职。

### 架构图

```
┌─────────────────────────────────────────────┐
│              用户请求                         │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│          Planner（规划器）                     │
│  输入：用户请求 + 可用工具列表                   │
│  输出：有序步骤列表 [Step1, Step2, Step3]      │
│  模型：可以用更强的模型（如 GPT-4）             │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│          Executor（执行器）                     │
│  逐步执行计划中的每一步                         │
│  模型：可以用更便宜的模型（如 GPT-3.5）         │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│       Re-planner（可选：重新规划）              │
│  当执行中发现计划需要调整时                      │
│  生成新的剩余步骤                              │
└─────────────────────────────────────────────┘
```

### 实现代码

```python
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor_llm, tools: dict):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools

    def run(self, query: str) -> str:
        # 阶段一：规划
        plan = self._create_plan(query)
        context = []  # 执行结果的累积上下文

        for step in plan:
            # 阶段二：逐步执行
            result = self._execute_step(step, context)
            context.append({"step": step, "result": result})

        # 阶段三：汇总结果
        return self._synthesize(query, context)

    def _create_plan(self, query: str) -> list[dict]:
        tool_descriptions = "\n".join(
            f"- {name}: {fn.__doc__}"
            for name, fn in self.tools.items()
        )
        prompt = f"""你是一个任务规划器。将用户的请求分解为具体步骤。

可用工具：
{tool_descriptions}

用户请求：{query}

请输出JSON格式的步骤列表：
[{{"step_id": 1, "description": "...", "tool": "tool_name", "args": "描述"}}]

注意：
- 步骤之间如果有依赖关系，顺序不能错
- 步骤之间如果没有依赖，可以在同一层级
- 每个步骤应该是原子操作"""

        response = self.planner.chat([{"role": "user", "content": prompt}])
        return self._parse_plan(response)

    def _execute_step(self, step: dict, context: list) -> str:
        """执行单个步骤，可以使用更便宜的模型"""
        # 构建执行上下文
        context_text = "\n".join(
            f"Step {c['step']['step_id']}: {c['step']['description']}\n结果: {c['result']}"
            for c in context
        ) if context else "这是第一个步骤"

        prompt = f"""根据以下计划执行当前步骤。
已有执行结果：
{context_text}

当前步骤：{step['description']}
使用工具：{step['tool']}

请直接执行，返回执行结果。"""

        response = self.executor.chat([{"role": "user", "content": prompt}])
        return response
```

### Plan-and-Execute vs ReAct：什么时候选谁？

| 维度 | ReAct | Plan-and-Execute |
|------|-------|------------------|
| 决策粒度 | 每一步都让LLM决策 | 只在规划阶段决策 |
| LLM调用次数 | N步 × 2（思考+行动）| 1 + N步 |
| 成本控制 | 每步都用同一个模型 | 规划用贵模型，执行用便宜模型 |
| 计划可审查性 | 黑盒（推理过程隐含）| 显式计划，可人工审核 |
| 灵活性 | 高（随时调整）| 中（需要re-planner）|
| 适合任务 | 探索性任务 | 结构化任务 |

**实际建议**：如果你的Agent要执行**不可逆操作**（比如删数据库、发邮件），一定要用Plan-and-Execute——先出计划，人工确认后再执行。

---

## §5 LLMCompiler实战：DAG并行工具调用

LLMCompiler解决的核心问题：**当多个工具调用之间没有依赖时，为什么不并行执行？**

### 问题场景

```
用户问：北京和上海今天分别几度？空气质量如何？

ReAct的做法（串行）：
  Step 1: 查北京天气 → 等待返回
  Step 2: 查上海天气 → 等待返回
  Step 3: 查北京空气质量 → 等待返回
  Step 4: 查上海空气质量 → 等待返回
  总耗时 = 4 × 单次API延迟

LLMCompiler的做法（并行）：
  无依赖 → 并行执行
  ┌─ 查北京天气 ─┐
  ├─ 查上海天气 ─┤  → 同时执行，总耗时 ≈ 1次API延迟
  ├─ 查北京空气 ─┤
  └─ 查上海空气 ─┘
```

### DAG构建与调度核心

```python
import asyncio
from dataclasses import dataclass

@dataclass
class ToolCall:
    id: str
    tool_name: str
    args: dict
    dependencies: list[str]  # 依赖的其他 ToolCall id

class LLMCompilerAgent:
    def __init__(self, llm, tools: dict):
        self.llm = llm
        self.tools = tools

    async def run(self, query: str) -> str:
        # 1. LLM 生成调用计划（DAG）
        plan = self._plan_tool_calls(query)

        # 2. 按拓扑序分层，同层并行执行
        layers = self._topological_layers(plan)

        results = {}
        for layer in layers:
            # 同一层的所有调用并行执行
            tasks = [
                self._execute_tool(tc, results)
                for tc in layer
            ]
            layer_results = await asyncio.gather(*tasks)
            for tc, result in zip(layer, layer_results):
                results[tc.id] = result

        # 3. 将所有结果汇总，生成最终答案
        return self._synthesize_answer(query, results)

    def _topological_layers(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        """将DAG分解为可并行执行的层"""
        layers = []
        remaining = list(calls)
        executed_ids = set()

        while remaining:
            # 找出所有依赖都已执行的调用
            ready = [
                tc for tc in remaining
                if all(dep in executed_ids for dep in tc.dependencies)
            ]
            if not ready:
                raise ValueError("检测到循环依赖")
            layers.append(ready)
            for tc in ready:
                remaining.remove(tc)
                executed_ids.add(tc.id)

        return layers

    async def _execute_tool(self, tc: ToolCall, context: dict) -> str:
        """执行单个工具调用，用上下文替换参数中的引用"""
        resolved_args = self._resolve_refs(tc.args, context)
        tool_fn = self.tools[tc.tool_name]
        return await asyncio.to_thread(tool_fn, **resolved_args)
```

### DAG调度的可视化示例

```
用户查询："比较北京和上海的天气、交通和房价"

LLM生成的调用计划：
┌─────────────┐
│ 查北京天气    │──────┐
├─────────────┤      │
│ 查上海天气    │──────┤
├─────────────┤      ├──▶ 汇总比较
│ 查北京交通    │──────┤
├─────────────┤      │
│ 查上海交通    │──────┤
├─────────────┤      │
│ 查北京房价    │──────┤
├─────────────┤      │
│ 查上海房价    │──────┘
└─────────────┘

执行层级：
Layer 0: [北京天气, 上海天气, 北京交通, 上海交通, 北京房价, 上海房价] ← 全部并行
Layer 1: [汇总比较] ← 依赖 Layer 0 全部完成
```

### LLMCompiler的关键设计决策

| 决策点 | 选项 | 推荐 |
|-------|------|------|
| 依赖解析方式 | LLM直接输出 / 代码解析 | 代码解析更可靠 |
| 参数引用机制 | 变量替换 / 结果注入 | 结果注入更灵活 |
| 错误处理 | 全部失败则中止 / 部分失败继续 | 取决于业务容忍度 |
| 超时策略 | 每层设超时 / 整体设超时 | 每层设超时更精细 |

---

## §6 ToolFormer思路：让Agent自主学习工具使用

ToolFormer与前面五种架构有本质区别——它不是运行时架构，而是**让模型学会在推理过程中自主插入工具调用**。

### 核心思想

```
传统方式：用户定义工具 → Agent被动调用
ToolFormer：模型自主判断「此处应该调用工具」

原始文本："巴黎是法国的首都，约有210万居民"
ToolFormer版本："巴黎是法国的首都，约有{{calculator('2.1M=2100000')}}210万居民"
```

### 训练流程（面试常考）

```
┌────────────────────────────────────────────────────┐
│                ToolFormer 训练三阶段                  │
│                                                     │
│  阶段1：标注候选插入点                                │
│  ┌──────────────────────────────────────────┐      │
│  │  原始文本 → LLM标记「可以插入工具调用的位置」│      │
│  │  "The capital of France is Paris"         │      │
│  │       ↑ 可以查百科                         │      │
│  │  "It has 2.1 million inhabitants"         │      │
│  │       ↑ 可以做单位转换                      │      │
│  └──────────────────────────────────────────┘      │
│                                                     │
│  阶段2：生成工具调用并执行                            │
│  ┌──────────────────────────────────────────┐      │
│  │  在候选位置生成工具调用                     │      │
│  │  执行调用，获取结果                         │      │
│  │  保留「降低困惑度」的调用（过滤噪声）        │      │
│  └──────────────────────────────────────────┘      │
│                                                     │
│  阶段3：微调模型                                     │
│  ┌──────────────────────────────────────────┐      │
│  │  用筛选后的数据微调原始模型                  │      │
│  │  模型学会在推理中自主插入工具调用            │      │
│  └──────────────────────────────────────────┘      │
└────────────────────────────────────────────────────┘
```

### ToolFormer的生产应用启示

在实际生产中，我们通常不需要完整复现ToolFormer的训练流程，但可以借鉴其思想：

1. **工具使用日志收集**：记录Agent的工具调用序列和结果
2. **质量评估**：标注哪些调用是有用的、哪些是无效的
3. **Prompt优化**：将高频有用调用模式写入system prompt
4. **渐进式学习**：通过few-shot示例引导模型学会工具使用

**面试要点**：面试官问ToolFormer时，要能说清楚它和ReAct的本质区别——ReAct是prompt工程层面的工具调用，ToolFormer是**模型能力层面**的工具使用。

---

## §7 LATS实战：树搜索+反思的Agent实现

LATS（Language Agent Tree Search）是目前最复杂的Agent架构之一，核心思想来自蒙特卡洛树搜索（MCTS）。

### 核心思想

```
ReAct:  一条路走到底
Reflexion: 一条路走到底 → 失败 → 反思 → 换条路走到底
LATS:  构建搜索树 → 评估每个节点 → 选择最有希望的分支 → 反思并剪枝

        根节点（用户查询）
       /    |    \
     A      B      C
    / \     |     / \
  A1  A2    B1   C1  C2
  ✓        ✗    ✗   ✓ ← LATS 会比较 A1 和 C2 的结果
```

### 核心搜索算法

```python
import math

class LATSAgent:
    def __init__(self, llm, tools: dict, num_simulations: int = 10):
        self.llm = llm
        self.tools = tools
        self.num_simulations = num_simulations

    class Node:
        def __init__(self, state, parent=None):
            self.state = state        # 当前状态/上下文
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0          # 累计价值
            self.reflection = None    # 反思内容

    def run(self, query: str) -> str:
        root = self.Node(state={"query": query, "history": []})

        for _ in range(self.num_simulations):
            # 1. 选择：从根节点向下选择最有潜力的节点
            node = self._select(root)

            # 2. 扩展：生成子节点（LLM的多个候选行动）
            if not node.children:
                children = self._expand(node)
                if children:
                    node.children = children

            # 3. 模拟：从选中的节点执行到终点
            score = self._simulate(node)

            # 4. 反弹：更新路径上所有节点的统计信息
            self._backpropagate(node, score)

        # 返回最佳路径的最终结果
        best_child = max(root.children, key=lambda n: n.value / max(n.visits, 1))
        return self._get_answer(best_child)

    def _select(self, node) -> 'Node':
        """UCB1 选择策略（借鉴MCTS）"""
        while node.children:
            node = max(
                node.children,
                key=lambda n: self._ucb1(n)
            )
        return node

    def _ucb1(self, node) -> float:
        """上置信界公式"""
        if node.visits == 0:
            return float('inf')
        exploitation = node.value / node.visits
        exploration = math.sqrt(
            2 * math.log(node.parent.visits) / node.visits
        )
        return exploitation + exploration

    def _expand(self, node) -> list['Node']:
        """让LLM生成多个候选行动"""
        prompt = f"""基于以下状态，生成3个不同的候选行动。
当前状态：{node.state}
反思经验：{node.reflection or '无'}

输出JSON数组，每个元素包含 action 和 description。"""

        candidates = self._parse_candidates(
            self.llm.chat([{"role": "user", "content": prompt}])
        )
        return [
            self.Node(
                state={**node.state, "action": c["action"]},
                parent=node
            )
            for c in candidates
        ]

    def _simulate(self, node) -> float:
        """从当前节点执行到终点，返回质量评分"""
        # 执行行动，获取结果
        result = self._execute_action(node)
        # 让LLM评分
        score = self._evaluate(node.state, result)
        return score

    def _backpropagate(self, node, score):
        """将评分向上传播"""
        while node:
            node.visits += 1
            node.value += score
            # 如果是叶节点且效果差，生成反思
            if score < 0.3 and node.parent:
                node.reflection = self._reflect(node)
            node = node.parent
```

### LATS vs 其他架构：什么时候值得上？

| 条件 | 是否值得用LATS |
|------|--------------|
| 任务有明确的成功/失败标准 | ✅ 值得 |
| 每次执行成本低（如纯API调用）| ✅ 值得 |
| 每次执行成本高（如修改代码仓库）| ❌ 慎用 |
| 候选方案少于5个 | ❌ 没必要，用Reflexion |
| 候选方案超过10个 | ✅ 树搜索有优势 |
| 需要实时响应 | ❌ 多次模拟太慢 |

---

## §8 六种架构的生产演进路径：从MVP到大规模服务

### 阶段一：MVP验证（1-2周）

```
目标：验证Agent能完成核心任务
架构：ReAct
技术栈：
  - LLM: GPT-4o-mini（成本低、速度适中）
  - 框架: 手写循环 或 LangChain（快速原型）
  - 工具: 3-5个核心API
  - 部署: 单进程、无队列、直接暴露HTTP接口

关键指标：任务完成率 > 70%
```

### 阶段二：体验优化（2-4周）

```
目标：提升准确率和用户体验
架构：ReAct + Reflexion
技术栈：
  - LLM: GPT-4o（准确性更高）
  - 新增：验证器（代码/单元测试/规则引擎）
  - 新增：反思记忆存储（Redis/SQLite）
  - 新增：超时和重试机制

关键指标：任务完成率 > 90%，平均步骤 < 8步
```

### 阶段三：性能提升（1-2月）

```
目标：降低延迟和成本
架构：Plan-and-Execute + LLMCompiler
技术栈：
  - 规划器: GPT-4o（强模型，调用少）
  - 执行器: GPT-4o-mini（便宜模型，调用多）
  - 新增：任务队列（Celery/BullMQ）
  - 新增：并行执行引擎（asyncio/线程池）
  - 新增：结果缓存（相同查询复用）

关键指标：P95延迟 < 10s，单次成本降低50%
```

### 阶段四：大规模服务（3-6月）

```
目标：支撑高并发、多租户
架构：微服务 + 事件驱动 + LATS
技术栈：
  - LLM网关: 统一的模型调用层（支持多模型切换）
  - Agent编排: Kubernetes + 自定义CRD
  - 消息队列: Kafka（事件驱动）
  - 向量数据库: Pinecone/Weaviate（记忆系统）
  - 监控: OpenTelemetry + Grafana
  - 安全: Guardrails（输入输出过滤）

关键指标：QPS > 100，SLA 99.9%
```

### 演进路径总结

```
MVP ──────▶ 体验优化 ──────▶ 性能提升 ──────▶ 大规模服务
ReAct       + Reflexion      + Plan&Execute   + 微服务架构
             + 验证器         + LLMCompiler    + LATS（复杂场景）
             + 反思记忆       + 缓存           + 多模型策略

  ⚡ 快      🎯 准           🚀 快             📈 大
```

### 生产环境的五个必做事项

| 优先级 | 事项 | 说明 |
|-------|------|------|
| P0 | **最大步骤数限制** | 防止无限循环，通常5-15步 |
| P0 | **工具调用超时** | 单次调用不超过30秒 |
| P0 | **成本监控** | 每次请求的token消耗和费用 |
| P1 | **输入过滤** | Prompt注入防护 |
| P1 | **结果缓存** | 相同查询的幂等结果缓存 |
| P1 | **降级策略** | 主模型不可用时切换备用模型 |
| P2 | **可观测性** | 每步的trace记录 |
| P2 | **人工审核** | 高风险操作的审核流程 |

---

## §9 面试题：面试官问"设计一个Agent系统"时的回答框架

### 标准回答框架（STAR-A法）

```
S - Situation（场景定义）：明确Agent要解决什么问题
T - Tool（工具设计）：需要哪些工具，如何接入
A - Architecture（架构选型）：为什么选这个架构
R - Reliability（可靠性）：如何保证稳定性和准确性
A - Assessment（评估度量）：如何衡量Agent的效果
```

### 回答模板（以"设计一个客服Agent"为例）

**面试官**：请设计一个智能客服Agent系统。

**回答**：

**S - 场景**：
> 客服Agent需要处理三类问题：常见问题问答（占比60%）、订单状态查询（30%）、投诉处理（10%）。需要支持多轮对话，准确率要求>95%。

**T - 工具**：
> 需要三组工具：
> 1. 知识库检索（RAG，对接产品文档）
> 2. 订单系统API（查询订单状态、物流信息）
> 3. 工单系统API（创建工单、转人工）
>
> 工具接入层统一使用Function Calling格式，支持超时和重试。

**A - 架构选型**：
> 采用 **Plan-and-Execute + Reflexion** 的组合架构：
> - 规划器：判断用户意图，分解为可执行步骤
> - 执行器：按步骤调用工具
> - 反思模块：当答案置信度低时，自动请求人工协助
>
> 选这个架构的原因：
> 1. 客服场景对准确性要求高，Plan-and-Execute可以先确认意图再行动
> 2. 投诉处理需要人工审核，Plan-and-Execute天然支持人工介入
> 3. 不需要树搜索（LATS），因为客服对话是线性的

**R - 可靠性**：
> 1. **Prompt注入防护**：输入层加Guardrails过滤恶意指令
> 2. **降级策略**：主模型故障时切换到备用模型
> 3. **兜底机制**：3次尝试失败后自动转人工
> 4. **幻觉控制**：所有回答必须引用知识库来源

**A - 评估**：
> 1. 离线评估：用1000条真实对话测试任务完成率
> 2. 在线指标：用户满意度（CSAT）、平均解决时长、转人工率
> 3. A/B测试：新旧版本并行，对比核心指标

### 面试官常见追问及应对

| 追问 | 应对要点 |
|------|---------|
| 「为什么不直接用ReAct？」| 客服场景需要准确性保证，ReAct没有全局规划能力 |
| 「工具调用失败了怎么办？」| 重试3次 → 降级到知识库回答 → 转人工 |
| 「如何处理多轮对话？」| 维护对话历史 + 意图状态机（上下文理解）|
| 「成本怎么控制？」| 规划用强模型（调用少），执行用弱模型（调用多）|
| 「如何保证安全性？」| 输入过滤 + 输出过滤 + 敏感操作审核 + 权限控制 |
| 「大规模部署怎么设计？」| 微服务架构 + 消息队列 + 水平扩展 + 多租户隔离 |

### 加分项：画架构图

面试中如果能现场画出系统架构图，会是巨大的加分项。以下是简化的架构图框架：

```
                    ┌──────────┐
                    │  用户请求  │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ 输入过滤  │ (Guardrails)
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ 意图识别  │ (Planner LLM)
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────┐   ┌──────┐   ┌──────┐
        │知识库 │   │订单API│   │工单API│
        └──────┘   └──────┘   └──────┘
              │          │          │
              └──────────┼──────────┘
                         │
                    ┌────▼─────┐
                    │ 结果验证  │ (Reflexion)
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ 输出过滤  │ (安全检查)
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ 返回用户  │
                    └──────────┘
```

---

## 总结：一张图记住所有架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 架构全景图                            │
│                                                              │
│  简单 ◀────────────────────────────────────────▶ 复杂        │
│                                                              │
│  ReAct ──▶ Reflexion ──▶ Plan&Execute ──▶ LATS             │
│   │                      │                    │              │
│   │                      │                    │              │
│   └──── 并行维度 ────────┘                    │              │
│                │                              │              │
│           LLMCompiler                    ToolFormer          │
│        (并行工具调用)                    (自主学习)            │
│                                                              │
│  选型原则：能用简单的就不要用复杂的                              │
│  演进路径：ReAct → +Reflexion → +Plan&Execute → +LLMCompiler │
└─────────────────────────────────────────────────────────────┘
```

**记住三个核心原则**：

1. **从ReAct开始**：80%的场景够用，不要过度设计
2. **按需加复杂度**：准确率不够加Reflexion，延迟太高加LLMCompiler
3. **生产化优先考虑可靠性**：降级、重试、监控比架构本身更重要

面试时，能说出「根据场景选择合适架构」比「我知道所有架构」更重要。架构是手段，解决问题才是目的。
