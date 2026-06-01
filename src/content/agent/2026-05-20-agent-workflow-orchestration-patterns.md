---
title: "Agent 工作流编排模式深度对比：Chain、DAG 与 State Machine 架构解析"
description: "深入对比三种主流Agent工作流编排模式——Chain、DAG与State Machine，从架构设计、性能表现到生产选型，给出完整决策框架"
date: 2026-05-20
author: "RiceBall-15"
category: agent
subCategory: agent-architecture
tags: ["Agent工作流", "编排模式", "Chain", "DAG", "State Machine", "LangGraph", "DSPy", "架构设计"]
draft: false
---

## 核心问题

当 Agent 需要执行多步骤任务时，如何组织步骤之间的依赖关系、错误恢复和数据传递？这是 Agent 工作流编排要解决的根本问题。

当前业界有三种主流编排模式：Chain（链式）、DAG（有向无环图）和 State Machine（状态机）。三种模式并非互斥——成熟的生产系统往往是它们的混合体。

## 一、三种模式的本质差异

```
Chain:      A → B → C → D
             线性，依赖隐式

DAG:          A
             / \
            B   C
             \ /
              D
           显式依赖，无环

State Machine:
            ┌─ IDLE ─┐
            │   ↓     │
            │ FETCH   │
            │   ↓     │
          ┌─→PARSE── │
          │   ↓      │
          │ RETRY───→┘
          │   ↓
          └── DONE
          有状态，允许回环
```

| 特性 | Chain | DAG | State Machine |
|------|-------|-----|---------------|
| **依赖表达** | 隐式顺序 | 显式依赖边 | 状态转换规则 |
| **错误恢复** | 从头重试 | 重试失败子图 | 定义回退状态 |
| **并行执行** | 不支持 | 天然支持 | 需额外设计 |
| **调试难度** | 低 | 中 | 高 |
| **动态分支** | 有限 | 条件节点 | 完整支持 |
| **核心库** | LangChain Chain | LangGraph、DSPy | Temporal、AWS Step Functions |
| **适用场景** | 简单流水线 | 数据预处理、评估 | 复杂业务工作流 |

## 二、Chain 模式：最简单的编排

Chain 模式将任务组织为顺序执行的线性步骤。每个步骤的输出是下一步的输入。

### 2.1 实现机制

链式编排的核心是组合模式（Composite Pattern）——每个任务单元实现同一接口，然后串联执行：

```
Input → StepA(ctx) → StepB(ctx) → StepC(ctx) → Output
               共享上下文对象传递状态
```

LangChain 的 `RunnableSequence` 是最典型的实现。每个 Runnable 实现 `invoke(input)` 方法，框架自动将上一节点的输出传给下一节点。

### 2.2 核心限制

**错误恢复困难**：如果 StepC 失败，整个链必须从头重试。中间步骤的 LLM 调用费用全部浪费。

**无法并行**：每个步骤必须等待前一步完成，GPU/CPU 利用率受限于单步延迟。

**分支表达能力弱**：虽然可以通过 `RunnableBranch` 做条件路由，但嵌套分支很快会变得不可维护。

### 2.3 适用场景

- 简单的 RAG 流水线（检索→增强→生成）
- 单路数据转换管道
- 快速原型验证

## 三、DAG 模式：显式依赖的图编排

DAG 模式将任务建模为有向无环图，节点是任务单元，边表示依赖关系。无环约束保证执行可以终止。

### 3.1 架构设计

```
           ┌─────────┐
           │ Validate │ (并行检查所有输入)
           └────┬────┘
           ┌────┴────┐
      ┌────▼──┐ ┌───▼────┐
      │Search │ │ Retrieve│ (并行执行)
      └───┬───┘ └───┬────┘
           ┌────┴────┐
           │  Merge   │ (同步点)
           └────┬────┘
           ┌────▼────┐
           │Generate │
           └────┬────┘
           ┌────▼────┐
           │Validate │
           └─────────┘
```

DAG 的核心优势在于：
1. **显式依赖**：每个节点声明它依赖哪些节点，框架自动解析执行顺序
2. **自动并行**：无依赖的节点可以并行执行
3. **部分重试**：失败时只重试受影响的子图，而非整个流程

### 3.2 DSPy 的 Program 抽象

DSPy 将 LLM 程序建模为可组合的模块（Module），通过声明式方式定义数据流图：

每个 Module 定义输入/输出签名，DSPy 自动优化提示词并管理数据流。这是 DAG 模式在 AI 领域的典型应用。

### 3.3 LangGraph 的状态机式 DAG

LangGraph 在 DAG 基础上增加了状态管理——节点可以读写共享状态，边可以是条件边：

```
State Schema:
  { messages: Message[]
    agent_out: str | None
    next: str }

节点执行函数:
  node(state) → 修改后的state

条件边:
  state.next == "tool" → tool节点
  state.next == "respond" → 输出节点
```

LangGraph 的特殊之处在于它支持**循环边**——Agent 可以在"思考→工具调用→思考"之间循环，直到满足终止条件。严格来说这超出了 DAG 的定义（引入了环），LangGraph 通过最大步数限制保证终止。

### 3.4 性能对比

| 指标 | Chain | DAG (LangGraph) | DAG (DSPy) |
|------|-------|-----------------|------------|
| 10步串行延迟 | 10×单步 | 依赖图关键路径 | 依赖图关键路径 |
| 并行加速比 | 1× | 2-5×（取决于图结构） | 2-5× |
| 重试粒度 | 全流程 | 子图 | 子图 |
| 状态管理 | 隐式上下文 | 显式State Schema | 模块间参数传递 |
| 学习曲线 | 低 | 高 | 中 |

## 四、State Machine 模式：复杂业务工作流

状态机模式将工作流建模为有限状态（Finite State），状态之间的转换由事件触发。

### 4.1 核心概念

```
States: { IDLE, FETCHING, PARSING, VALIDATING, RETRYING, DONE, FAILED }

Transitions:
  IDLE      ──(start)──→ FETCHING
  FETCHING  ──(data)───→ PARSING
  FETCHING  ──(error)──→ RETRYING
  RETRYING  ──(retry)──→ FETCHING
  RETRYING  ──(max)────→ FAILED
  PARSING   ──(valid)──→ VALIDATING
  PARSING   ──(invalid)→ FAILED
  VALIDATING─(pass)────→ DONE
```

与 DAG 的关键区别：
- DAG 描述了"做什么"（任务依赖），状态机描述了"处于什么状态"（系统状态）
- DAG 的节点是任务，状态机的节点是状态
- 状态机天然支持重试、补偿、超时等复杂模式

### 4.2 生产案例：AWS Step Functions

Step Functions 是云原生的状态机服务，支持：

1. **Task 状态**：执行一个任务单元（Lambda、Batch 等）
2. **Choice 状态**：条件分支
3. **Parallel 状态**：并行执行
4. **Map 状态**：对集合动态迭代
5. **Wait 状态**：延迟执行
6. **Catch/Retry**：错误处理

```json
{
  "StartAt": "Extract",
  "States": {
    "Extract": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:extract",
      "Next": "ParallelTransform",
      "Catch": [{ "ErrorEquals": ["States.ALL"], "Next": "HandleError" }]
    },
    "ParallelTransform": {
      "Type": "Parallel",
      "Branches": [
        { "StartAt": "TransformA", ... },
        { "StartAt": "TransformB", ... }
      ],
      "Next": "Load"
    }
  }
}
```

### 4.3 Temporal 工作流

Temporal 提供了更灵活的状态机——工作流代码就是状态机定义：

```python
@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, input: Input) -> Output:
        try:
            data = await execute_activity(fetch_data, input.url,
                retry_options=RetryOptions(max_attempts=3))
            result = await execute_activity(process_data, data,
                heartbeat_timeout=30)
            return result
        except Exception as e:
            await execute_activity(notify_failure, e)
            raise
```

Temporal 自动持久化中间状态——即使工作流运行了 30 天、进程重启 10 次，状态也不会丢失。

## 五、混合模式：生产级的答案

在实际生产系统中，三种模式不是非此即彼的关系，而是分层使用：

```
外层：状态机（Temporal / Step Functions）
  │  管理工作流生命周期、重试、补偿、超时
  │
  ├─ 中层：DAG（LangGraph / DSPy）
  │    管理任务依赖、并行执行、数据流
  │    │
  │    └─ 内层：Chain（简单的串行步骤）
  │          简单的 A→B→C 管道
  │
  └─ 错误处理层：状态机的 Retry/Catch
```

### 5.1 选型决策树

```
任务包含 LLM 调用？
├── 是，且步骤数 ≤ 5，线性依赖 → Chain（快速原型）
├── 是，且步骤有并行分支 → DAG（LangGraph / DSPy）
├── 是，且需要复杂的错误恢复 → State Machine + DAG
├── 是，且需要人机交互 → State Machine（Temporal）
└── 否，纯数据管道 → DAG（Airflow / Prefect）
```

### 5.2 实际案例：内容审核 Agent

一个生产级内容审核 Agent 的工作流：

| 层次 | 模式 | 用途 |
|------|------|------|
| 全局 | State Machine (Temporal) | 管理审核生命周期、超时、人工升级 |
| 子流程 | DAG (LangGraph) | 并行执行文本审核、图片审核、音频审核 |
| 单路径 | Chain | 单个审核模型的多轮推理（初判→详细分析→打分） |
| 重试 | State Machine Retry | 审核服务超时重试（指数退避） |

## 六、性能与可观测性

| 指标 | Chain | DAG | State Machine |
|------|-------|-----|---------------|
| 端到端延迟 | 累加 | 关键路径 | 状态转换开销 |
| 并行度 | 1 | 节点数 | 有限（P/S模式） |
| 状态持久化 | 无 | 可选 | 强制 |
| 监控粒度 | 步骤级别 | 节点级别 | 状态转换级别 |
| 调试方式 | print/log | 图可视化 | 状态回放 |

**关键发现**：当工作流步骤数超过 5 步或包含分支逻辑时，Chain 模式的调试成本呈指数增长。此时切换到 DAG 或 State Machine，虽然初始复杂度更高，但长期维护成本反而更低。

## 七、小结

三种编排模式对应不同的复杂度阶段：

1. **Chain** — 启动最快，适合 ≤5 步的简单流水线
2. **DAG** — 适合 5-20 步的并行任务，依赖明确，LangGraph 是当前 Agent 领域的主流选择
3. **State Machine** — 适合复杂业务工作流（20+ 步、人机交互、长周期任务），生产级可靠性

**推荐选型**：80% 的 Agent 场景用 LangGraph（DAG + 有限循环）就够了。当需要持久化运行超过 1 小时的任务、跨服务补偿、或多步人工审批时，套一层 Temporal/Step Functions 状态机。