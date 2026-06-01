---
title: "Agent状态机设计模式：从有限状态到自主决策的演进实践"
description: "深度解析Agent系统中的状态机设计模式，涵盖FSM/状态图/行为树/效用Agent四大范式，结合LangGraph/CrewAI实战案例，构建可靠可控的Agent系统"
date: 2026-06-01
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["Agent", "状态机", "FSM", "LangGraph", "行为树", "效用Agent", "Agent架构", "设计模式"]
draft: false
---

# Agent状态机设计模式：从有限状态到自主决策的演进实践

## 前言

在构建Agent系统时，一个核心矛盾始终存在：**我们既希望Agent足够自主，能自主规划和决策；又希望Agent足够可控，行为可预测、可调试、可回滚。**

状态机（State Machine）是解决这个矛盾的关键工具。通过显式定义Agent的状态空间和转移规则，我们可以在保持灵活性的同时获得对Agent行为的完全掌控。

本文将从最简单的有限状态机（FSM）出发，逐步演进到行为树（Behavior Tree）和效用Agent（Utility Agent），并结合LangGraph、CrewAI等框架的实战经验，展示如何设计既智能又可靠的Agent系统。

---

## 一、为什么Agent需要状态机？

### 1.1 没有状态管理的Agent会怎样？

```
┌─────────────────────────────────────────────────────────────────┐
│              没有状态管理的Agent：混沌之源                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  用户："帮我订一张明天去上海的机票"                                  │
│                                                                  │
│  Agent内部流程（无状态管理）：                                      │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐                    │
│  │ ？？ │ ──→ │ 搜索 │ ──→ │ ？？ │ ──→ │ 预订 │  ← 跳过了什么？   │
│  └─────┘     └─────┘     └─────┘     └─────┘                    │
│                                                                  │
│  问题：                                                           │
│  • 为什么跳过了"确认航班"步骤？                                     │
│  • 为什么没有"选择座位"环节？                                       │
│  • 出错了如何回退到上一步？                                         │
│  • 如何确保每次执行相同请求时行为一致？                               │
│                                                                  │
│  根本原因：Agent不知道自己"在哪里"，也不知道下一步"该做什么"         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 状态机带来的核心价值

| 价值 | 说明 | 示例 |
|------|------|------|
| **可预测性** | 相同状态+相同输入=相同输出 | 每次订票流程一致 |
| **可调试性** | 可以打印当前状态和转移历史 | 快速定位Agent卡在哪一步 |
| **可恢复性** | 出错可以从特定状态重试 | 搜索失败只重试搜索，不重头开始 |
| **可观测性** | 监控各状态的停留时间和转移频率 | 发现"确认"步骤用户放弃率高 |
| **可干预性** | 人工可以在任意状态接管 | 客服可以手动跳到"支付"状态 |

---

## 二、四大状态管理范式

### 2.1 范式全景对比

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Agent状态管理范式演进                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   简单                                                              │   复杂
│   ◄────────────────────────────────────────────────────────────────► │
│                                                                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │   FSM    │ →  │  状态图   │ →  │ 行为树   │ →  │ 效用Agent │     │
│   │ 有限状态机│    │Statechart│    │Behavior  │    │ Utility  │     │
│   │          │    │          │    │  Tree    │    │  Agent   │     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                                                                       │
│   适用：              适用：              适用：             适用：     │
│   简单线性流程        复杂并行流程        层次化任务         动态决策   │
│   固定工作流         有并发需求          行为组合           多目标优化 │
│   快速原型           需要状态嵌套        可视化编辑         自适应行为 │
│                                                                       │
│   代表实现：          代表实现：          代表实现：          代表实现： │
│   LangGraph(简单)    XState             自定义实现          AutoGen   │
│   自定义FSM          LangGraph(高级)     GOAP              效用系统   │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 范式选择决策树

```
            你的Agent需要什么级别的控制？
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
      简单线性      有条件分支    多目标动态
      固定流程      有限并行      自主决策
         │            │            │
         ▼            ▼            ▼
       FSM/          状态图/      效用Agent/
      LangGraph     LangGraph    多目标优化
         │            │            │
    ┌────┴────┐  ┌────┴────┐  ┌───┴────┐
    ▼         ▼  ▼         ▼  ▼        ▼
  快速开发  严格控制  需要可视化  复杂条件  需要学习  多约束
  原型验证  可预测   编辑状态图  逻辑嵌套  适应环境  平衡
```

---

## 三、有限状态机（FSM）：最简单也最实用

### 3.1 核心概念

```
┌─────────────────────────────────────────────────────────────────┐
│                    FSM核心要素                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  状态（State）：Agent在某一时刻的"位置"                             │
│  ├── IDLE: 空闲，等待用户输入                                      │
│  ├── THINKING: 正在思考/规划                                       │
│  ├── EXECUTING: 正在执行工具调用                                   │
│  ├── WAITING: 等待外部输入（如用户确认）                            │
│  └── DONE: 任务完成                                               │
│                                                                  │
│  转移（Transition）：从一个状态到另一个状态                         │
│  ├── IDLE → THINKING: 收到用户输入                                 │
│  ├── THINKING → EXECUTING: 决定需要调用工具                        │
│  ├── THINKING → DONE: 决定直接回复用户                              │
│  ├── EXECUTING → THINKING: 工具返回结果，需要继续思考               │
│  ├── EXECUTING → WAITING: 需要用户确认                             │
│  └── WAITING → EXECUTING: 用户确认，继续执行                        │
│                                                                  │
│  动作（Action）：在转移时执行的操作                                 │
│  └── 可以是：调用LLM、执行工具、发送消息、记录日志                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Python实现：最小FSM

```python
from enum import Enum
from typing import Callable, Dict, Tuple
import asyncio

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"

class AgentFSM:
    """最小化Agent有限状态机实现"""
    
    def __init__(self):
        self.state = AgentState.IDLE
        self.transitions: Dict[Tuple[AgentState, str], Tuple[AgentState, Callable]] = {}
        self.state_handlers: Dict[AgentState, Callable] = {}
        self.history: list = []
    
    def add_transition(self, from_state: AgentState, event: str, to_state: AgentState, action: Callable = None):
        """添加状态转移规则"""
        self.transitions[(from_state, event)] = (to_state, action)
    
    def add_state_handler(self, state: AgentState, handler: Callable):
        """添加状态处理函数"""
        self.state_handlers[state] = handler
    
    async def trigger(self, event: str, context: dict = None):
        """触发事件，驱动状态转移"""
        key = (self.state, event)
        if key not in self.transitions:
            raise ValueError(f"No transition from {self.state} on event '{event}'")
        
        to_state, action = self.transitions[key]
        
        # 记录转移历史
        self.history.append({
            'from': self.state.value,
            'event': event,
            'to': to_state.value,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        # 执行动作
        if action:
            await action(context)
        
        # 转移状态
        self.state = to_state
        
        # 执行新状态的处理函数
        if to_state in self.state_handlers:
            await self.state_handlers[to_state](context)
    
    def get_current_state(self) -> str:
        return self.state.value
    
    def get_history(self) -> list:
        return self.history.copy()


# === 使用示例：订票Agent ===

async def search_flights(ctx: dict):
    """搜索航班"""
    print("🔍 搜索航班中...")
    ctx['flights'] = [{"id": "CA1234", "price": 800}, {"id": "MU5678", "price": 650}]

async def confirm_flight(ctx: dict):
    """确认航班"""
    print(f"✅ 已确认航班: {ctx.get('selected_flight', 'CA1234')}")

async def process_payment(ctx: dict):
    """处理支付"""
    print("💳 处理支付中...")

# 创建FSM
fsm = AgentFSM()

# 定义转移规则
fsm.add_transition(AgentState.IDLE, "user_input", AgentState.THINKING)
fsm.add_transition(AgentState.THINKING, "need_search", AgentState.EXECUTING, search_flights)
fsm.add_transition(AgentState.THINKING, "direct_reply", AgentState.DONE)
fsm.add_transition(AgentState.EXECUTING, "search_complete", AgentState.WAITING)
fsm.add_transition(AgentState.WAITING, "user_confirm", AgentState.EXECUTING, confirm_flight)
fsm.add_transition(AgentState.EXECUTING, "payment_needed", AgentState.EXECUTING, process_payment)
fsm.add_transition(AgentState.EXECUTING, "task_complete", AgentState.DONE)
```

### 3.3 LangGraph实现：生产级FSM

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: list
    current_step: str
    search_results: dict
    user_confirmation: bool
    error: str | None

# 创建状态图
workflow = StateGraph(AgentState)

# 定义节点（状态处理函数）
def think(state: AgentState) -> AgentState:
    """思考：决定下一步行动"""
    llm = ChatOpenAI(model="gpt-4o")
    
    system_prompt = """你是一个订票助手。根据用户消息决定下一步：
    - 如果需要搜索航班，回复 SEARCH_FLIGHTS
    - 如果需要用户确认，回复 WAIT_CONFIRM
    - 如果可以直接回复，回复 REPLY"""
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ])
    
    decision = response.content.strip()
    return {"current_step": decision}

def search_flights(state: AgentState) -> AgentState:
    """搜索航班"""
    # 实际调用航班API
    results = {"flights": [{"id": "CA1234", "price": 800}]}
    return {"search_results": results, "current_step": "WAIT_CONFIRM"}

def wait_confirm(state: AgentState) -> AgentState:
    """等待用户确认"""
    # 在实际应用中，这里会暂停等待用户输入
    return {"current_step": "CONFIRMED"}

def process_payment(state: AgentState) -> AgentState:
    """处理支付"""
    # 实际支付逻辑
    return {"current_step": "DONE"}

# 定义节点
workflow.add_node("think", think)
workflow.add_node("search_flights", search_flights)
workflow.add_node("wait_confirm", wait_confirm)
workflow.add_node("process_payment", process_payment)

# 定义条件路由
def route_after_think(state: AgentState) -> str:
    step = state["current_step"]
    if step == "SEARCH_FLIGHTS":
        return "search_flights"
    elif step == "WAIT_CONFIRM":
        return "wait_confirm"
    else:
        return "process_payment"

# 添加边
workflow.set_entry_point("think")
workflow.add_conditional_edges("think", route_after_think)
workflow.add_edge("search_flights", "wait_confirm")
workflow.add_edge("wait_confirm", "process_payment")
workflow.add_edge("process_payment", END)

# 编译
app = workflow.compile()
```

---

## 四、状态图（Statechart）：处理复杂并行状态

### 4.1 从FSM到状态图的演进

```
┌──────────────────────────────────────────────────────────────────────┐
│               FSM vs Statechart：关键差异                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  FSM（有限状态机）：                                                   │
│  ┌─────┐    ┌─────┐    ┌─────┐                                      │
│  │ IDLE│───→│THINK│───→│EXEC │    任何时刻只能在一个状态              │
│  └─────┘    └─────┘    └─────┘                                      │
│                                                                       │
│  Statechart（状态图）：                                                │
│  ┌─────────────────────────────────────┐                             │
│  │           ACTIVE (并行区域)          │                             │
│  │  ┌─────────┐     ┌─────────┐        │                             │
│  │  │ 任务状态 │     │ 通信状态 │        │    可以同时处于多个状态       │
│  │  │ THINK   │     │ LISTEN  │        │                             │
│  │  │ EXEC    │     │ TALK    │        │                             │
│  │  └─────────┘     └─────────┘        │                             │
│  └─────────────────────────────────────┘                             │
│                                                                       │
│  核心区别：                                                           │
│  • FSM：状态互斥，一个时刻只有一个状态                                  │
│  • Statechart：支持状态层次、并行、历史状态                             │
│  • Statechart是FSM的超集，FSM是Statechart的特例                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 并行状态实战：多任务Agent

```python
from dataclasses import dataclass, field
from typing import Set
import asyncio

@dataclass
class ParallelState:
    """支持并行状态的状态机"""
    
    # 当前激活的状态集合（支持并行）
    active_states: Set[str] = field(default_factory=set)
    
    # 状态层次结构
    hierarchy: dict = field(default_factory=lambda: {
        "root": {"children": ["task", "communication"]},
        "task": {"children": ["idle", "thinking", "executing"]},
        "communication": {"children": ["listening", "speaking", "silent"]},
    })
    
    # 转移规则
    transitions: dict = field(default_factory=dict)
    
    def enter_state(self, state: str):
        """进入状态"""
        self.active_states.add(state)
        # 自动进入子状态
        if state in self.hierarchy:
            for child in self.hierarchy[state]["children"]:
                self.active_states.add(child)
    
    def exit_state(self, state: str):
        """退出状态"""
        self.active_states.discard(state)
        # 自动退出子状态
        if state in self.hierarchy:
            for child in self.hierarchy[state]["children"]:
                self.active_states.discard(child)
    
    def is_active(self, state: str) -> bool:
        """检查状态是否激活"""
        return state in self.active_states
    
    def on_event(self, event: str, context: dict):
        """处理事件，并行检查所有激活状态"""
        for state in list(self.active_states):
            key = (state, event)
            if key in self.transitions:
                action = self.transitions[key]
                action(self, context)


# 使用示例：多任务Agent
agent = ParallelState()
agent.enter_state("root")
# 现在 active_states = {"root", "task", "communication", "idle", "listening"}

# Agent可以同时"思考"和"监听"
agent.enter_state("thinking")
agent.exit_state("idle")
# 现在 active_states = {"root", "task", "communication", "thinking", "listening"}
```

---

## 五、行为树（Behavior Tree）：层次化任务分解

### 5.1 行为树核心概念

```
┌─────────────────────────────────────────────────────────────────┐
│                    行为树核心节点类型                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  组合节点（Composite）：                                          │
│  ├── Sequence (顺序)：所有子节点成功才成功                         │
│  ├── Selector (选择)：任一子节点成功即成功                         │
│  └── Parallel (并行)：所有子节点并行执行                           │
│                                                                  │
│  装饰节点（Decorator）：                                          │
│  ├── Inverter：反转子节点结果                                     │
│  ├── Repeater：重复执行子节点N次                                  │
│  ├── UntilFail：一直执行直到失败                                  │
│  └── Cooldown：冷却时间内不重复执行                               │
│                                                                  │
│  叶节点（Leaf）：                                                 │
│  ├── Action：执行具体动作                                         │
│  └── Condition：检查条件                                         │
│                                                                  │
│  执行结果：                                                       │
│  ├── SUCCESS：成功                                                │
│  ├── FAILURE：失败                                                │
│  └── RUNNING：正在执行中                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 订票Agent行为树

```
                         ┌─────────┐
                         │  ROOT   │
                         │ Sequence│
                         └────┬────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
      ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
      │   理解    │    │   规划    │    │   执行    │
      │  用户意图  │    │  任务步骤  │    │  并返回   │
      └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
            │                 │                 │
            │           ┌─────┴─────┐     ┌─────┴─────┐
            │           │  Selector │     │  并行执行  │
            │           └─────┬─────┘     │  子任务    │
            │        ┌────────┼────────┐  └─────┬─────┘
            │        │        │        │        │
      ┌─────┴───┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐
      │ LLM理解 │ │搜索 │ │预订 │ │支付 │ │通知 │
      │ 意图    │ │航班 │ │航班 │ │     │ │用户 │
      └─────────┘ └─────┘ └─────┘ └─────┘ └─────┘
```

### 5.3 Python行为树实现

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
import asyncio

class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"

class BTNode(ABC):
    """行为树节点基类"""
    
    @abstractmethod
    async def tick(self, context: dict) -> NodeStatus:
        pass

class ActionNode(BTNode):
    """动作节点：执行具体操作"""
    
    def __init__(self, name: str, action: callable):
        self.name = name
        self.action = action
    
    async def tick(self, context: dict) -> NodeStatus:
        try:
            result = await self.action(context)
            return NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            print(f"Action '{self.name}' failed: {e}")
            return NodeStatus.FAILURE

class ConditionNode(BTNode):
    """条件节点：检查条件"""
    
    def __init__(self, name: str, condition: callable):
        self.name = name
        self.condition = condition
    
    async def tick(self, context: dict) -> NodeStatus:
        return NodeStatus.SUCCESS if self.condition(context) else NodeStatus.FAILURE

class SequenceNode(BTNode):
    """顺序节点：所有子节点必须成功"""
    
    def __init__(self, children: List[BTNode]):
        self.children = children
    
    async def tick(self, context: dict) -> NodeStatus:
        for child in self.children:
            status = await child.tick(context)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

class SelectorNode(BTNode):
    """选择节点：任一子节点成功即成功"""
    
    def __init__(self, children: List[BTNode]):
        self.children = children
    
    async def tick(self, context: dict) -> NodeStatus:
        for child in self.children:
            status = await child.tick(context)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE


# === 构建订票Agent行为树 ===

async def understand_intent(ctx):
    """理解用户意图"""
    intent = await llm_understand(ctx['user_input'])
    ctx['intent'] = intent
    return True

async def search_flights(ctx):
    """搜索航班"""
    results = await flight_api.search(ctx['intent'])
    ctx['flights'] = results
    return len(results) > 0

async def book_flight(ctx):
    """预订航班"""
    success = await flight_api.book(ctx['selected_flight'])
    return success

# 构建行为树
booking_tree = SequenceNode([
    ActionNode("理解意图", understand_intent),
    SelectorNode([
        SequenceNode([
            ConditionNode("有航班", lambda ctx: len(ctx.get('flights', [])) > 0),
            ActionNode("预订航班", book_flight),
        ]),
        ActionNode("无航班通知", lambda ctx: notify_no_flights(ctx)),
    ]),
])

# 执行
context = {"user_input": "订一张明天去上海的机票"}
result = await booking_tree.tick(context)
```

---

## 六、效用Agent（Utility Agent）：动态决策

### 6.1 核心思想

```
┌─────────────────────────────────────────────────────────────────┐
│                    效用Agent决策机制                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传统状态机：                                                      │
│  if condition_A:                                                 │
│      do_action_X                                                 │
│  elif condition_B:                                               │
│      do_action_Y                                                 │
│  问题：规则是硬编码的，无法处理复杂的多目标权衡                       │
│                                                                  │
│  效用Agent：                                                      │
│  for action in possible_actions:                                 │
│      action.utility = compute_utility(action, world_state)       │
│  best_action = max(possible_actions, key=lambda a: a.utility)    │
│                                                                  │
│  优点：                                                           │
│  • 可以量化每个动作的"价值"                                        │
│  • 自动选择当前最优动作                                           │
│  • 可以处理多个冲突目标的权衡                                      │
│  • 行为可以通过调整效用函数来"训练"                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 多目标效用计算

```python
from dataclasses import dataclass
from typing import Dict, Callable
import math

@dataclass
class Action:
    name: str
    cost: float  # 执行成本
    effects: Dict[str, float]  # 对世界状态的影响

class UtilityAgent:
    """效用Agent：基于效用函数做决策"""
    
    def __init__(self):
        self.world_state: Dict[str, float] = {}
        self.goals: Dict[str, float] = {}  # 目标名称 -> 权重
        self.actions: list[Action] = []
    
    def compute_utility(self, action: Action) -> float:
        """计算动作的效用值"""
        utility = 0
        
        # 计算目标达成的效用
        for effect_key, effect_value in action.effects.items():
            if effect_key in self.goals:
                current = self.world_state.get(effect_key, 0)
                new_value = current + effect_value
                # 边际效用递减
                utility += self.goals[effect_key] * math.tanh(new_value)
        
        # 扣除执行成本
        utility -= action.cost
        
        return utility
    
    def decide(self) -> Action | None:
        """选择最优动作"""
        if not self.actions:
            return None
        
        scored_actions = [(action, self.compute_utility(action)) for action in self.actions]
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        best_action, best_utility = scored_actions[0]
        
        # 效用为负说明没有值得执行的动作
        if best_utility < 0:
            return None
        
        return best_action
    
    def execute(self, action: Action):
        """执行动作并更新世界状态"""
        for key, value in action.effects.items():
            self.world_state[key] = self.world_state.get(key, 0) + value


# === 多目标订票Agent ===

agent = UtilityAgent()

# 定义目标（权重越高越重要）
agent.goals = {
    "flight_booked": 10.0,      # 订到航班
    "price_low": 5.0,           # 价格低
    "time_convenient": 3.0,     # 时间方便
    "user_satisfied": 8.0,      # 用户满意
}

# 定义可选动作
agent.actions = [
    Action(
        name="订最便宜航班",
        cost=0.1,
        effects={"flight_booked": 1, "price_low": 0.8, "time_convenient": 0.2}
    ),
    Action(
        name="订最方便航班",
        cost=0.1,
        effects={"flight_booked": 1, "price_low": 0.2, "time_convenient": 0.9}
    ),
    Action(
        name="继续搜索更多选项",
        cost=0.3,
        effects={"flight_booked": 0, "price_low": 0.5, "time_convenient": 0.5}
    ),
]

# Agent决策
best = agent.decide()
print(f"选择: {best.name}")  # 输出：订最便宜航班
```

---

## 七、混合架构：LangGraph + 行为树 + 效用决策

### 7.1 架构设计

```
┌──────────────────────────────────────────────────────────────────────┐
│                    混合架构：三层决策模型                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    策略层（LangGraph）                        │    │
│  │    定义全局工作流、状态转移、条件路由                           │    │
│  │    负责：任务分解、流程控制、错误恢复                           │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│  ┌──────────────────────────┴──────────────────────────────────┐    │
│  │                    执行层（行为树）                            │    │
│  │    定义每个任务的具体执行步骤                                  │    │
│  │    负责：工具调用、重试逻辑、子任务编排                         │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│  ┌──────────────────────────┴──────────────────────────────────┐    │
│  │                    决策层（效用Agent）                         │    │
│  │    在多个可选动作中选择最优                                    │    │
│  │    负责：多目标优化、资源分配、动态调整                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  数据流：                                                             │
│  用户输入 → LangGraph(规划) → 行为树(执行) → 效用Agent(微调) → 输出    │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 LangGraph实现混合架构

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class HybridAgentState(TypedDict):
    messages: list
    task_plan: list  # LangGraph规划的任务列表
    current_task: str
    task_results: dict
    utility_scores: dict  # 效用评分

# === 策略层：LangGraph定义全局流程 ===

def plan_tasks(state: HybridAgentState) -> HybridAgentState:
    """策略层：规划任务"""
    # 使用LLM分解任务
    tasks = llm_plan(state["messages"][-1].content)
    return {"task_plan": tasks, "current_task": tasks[0] if tasks else None}

def execute_with_behavior_tree(state: HybridAgentState) -> HybridAgentState:
    """执行层：使用行为树执行当前任务"""
    task = state["current_task"]
    bt = build_behavior_tree_for_task(task)
    result = bt.tick(state)
    return {"task_results": {**state["task_results"], task: result}}

def utility_optimize(state: HybridAgentState) -> HybridAgentState:
    """决策层：效用优化"""
    # 评估当前结果，决定是否需要调整
    scores = compute_utility_scores(state["task_results"])
    return {"utility_scores": scores}

def route_next(state: HybridAgentState) -> str:
    """路由：决定下一步"""
    plan = state["task_plan"]
    current = state["current_task"]
    
    # 检查是否有未完成的任务
    current_idx = plan.index(current) if current in plan else -1
    if current_idx < len(plan) - 1:
        return "continue"
    return "finish"

# 构建LangGraph
workflow = StateGraph(HybridAgentState)
workflow.add_node("plan", plan_tasks)
workflow.add_node("execute", execute_with_behavior_tree)
workflow.add_node("optimize", utility_optimize)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "optimize")
workflow.add_conditional_edges("optimize", route_next, {
    "continue": "execute",
    "finish": END
})

app = workflow.compile()
```

---

## 八、实战案例：多Agent协作的状态管理

### 8.1 多Agent状态同步

```python
from dataclasses import dataclass, field
from typing import Dict, List
import asyncio
import uuid

@dataclass
class AgentStatus:
    agent_id: str
    state: str
    current_task: str | None = None
    progress: float = 0.0

@dataclass
class MultiAgentOrchestrator:
    """多Agent编排器：管理多个Agent的状态"""
    
    agents: Dict[str, AgentStatus] = field(default_factory=dict)
    shared_memory: dict = field(default_factory=dict)
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    
    def register_agent(self, agent_id: str, initial_state: str = "idle"):
        """注册Agent"""
        self.agents[agent_id] = AgentStatus(
            agent_id=agent_id,
            state=initial_state
        )
    
    async def broadcast(self, message: dict):
        """广播消息给所有Agent"""
        await self.message_queue.put(message)
    
    def get_agent_states(self) -> Dict[str, str]:
        """获取所有Agent状态"""
        return {aid: a.state for aid, a in self.agents.items()}
    
    def update_agent_state(self, agent_id: str, new_state: str, task: str = None):
        """更新Agent状态"""
        if agent_id in self.agents:
            self.agents[agent_id].state = new_state
            self.agents[agent_id].current_task = task
            self.agents[agent_id].progress = 0.0
    
    async def coordinate(self):
        """协调多个Agent的执行"""
        while True:
            # 检查是否有Agent需要协调
            idle_agents = [
                aid for aid, a in self.agents.items() 
                if a.state == "idle"
            ]
            
            if not idle_agents:
                await asyncio.sleep(0.1)
                continue
            
            # 分配任务给空闲Agent
            pending_tasks = self.shared_memory.get("pending_tasks", [])
            if pending_tasks:
                for agent_id in idle_agents[:len(pending_tasks)]:
                    task = pending_tasks.pop(0)
                    self.update_agent_state(agent_id, "working", task)
                    # 发送任务给Agent
                    await self.broadcast({
                        "type": "task_assignment",
                        "agent_id": agent_id,
                        "task": task
                    })
```

---

## 九、生产环境最佳实践

### 9.1 状态持久化

```python
import json
from datetime import datetime

class StatePersistence:
    """状态持久化：支持状态恢复"""
    
    def __init__(self, storage_path: str = "./agent_states"):
        self.storage_path = storage_path
    
    async def save_state(self, agent_id: str, state: dict, version: int = 0):
        """保存状态快照"""
        snapshot = {
            "agent_id": agent_id,
            "state": state,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "checksum": self._compute_checksum(state)
        }
        
        filename = f"{self.storage_path}/{agent_id}_v{version}.json"
        with open(filename, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
    
    async def load_state(self, agent_id: str, version: int = None) -> dict | None:
        """加载状态快照"""
        if version is None:
            # 加载最新版本
            version = self._get_latest_version(agent_id)
        
        filename = f"{self.storage_path}/{agent_id}_v{version}.json"
        try:
            with open(filename, "r") as f:
                snapshot = json.load(f)
            
            # 验证checksum
            if self._compute_checksum(snapshot["state"]) != snapshot["checksum"]:
                raise ValueError("State checksum mismatch, state may be corrupted")
            
            return snapshot["state"]
        except FileNotFoundError:
            return None
    
    async def rollback(self, agent_id: str, target_version: int) -> dict | None:
        """回滚到指定版本"""
        return await self.load_state(agent_id, target_version)
```

### 9.2 状态监控仪表板

```python
from dataclasses import dataclass
from typing import List
import time

@dataclass
class StateMetrics:
    """状态监控指标"""
    
    state_name: str
    entry_count: int = 0
    total_duration: float = 0.0
    error_count: int = 0
    avg_duration: float = 0.0
    
    def record_entry(self, duration: float, error: bool = False):
        self.entry_count += 1
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.entry_count
        if error:
            self.error_count += 1

class StateMonitor:
    """状态监控：追踪Agent状态转移"""
    
    def __init__(self):
        self.metrics: dict[str, StateMetrics] = {}
        self.transition_log: List[dict] = []
    
    def record_transition(self, from_state: str, to_state: str, duration: float):
        """记录状态转移"""
        self.transition_log.append({
            "from": from_state,
            "to": to_state,
            "duration": duration,
            "timestamp": time.time()
        })
        
        # 更新指标
        if to_state not in self.metrics:
            self.metrics[to_state] = StateMetrics(state_name=to_state)
        self.metrics[to_state].record_entry(duration)
    
    def get_bottleneck_states(self) -> List[dict]:
        """找出瓶颈状态（平均停留时间最长）"""
        return sorted(
            self.metrics.values(),
            key=lambda m: m.avg_duration,
            reverse=True
        )[:5]
    
    def get_error_prone_states(self) -> List[dict]:
        """找出容易出错的状态"""
        return sorted(
            [m for m in self.metrics.values() if m.error_count > 0],
            key=lambda m: m.error_count / m.entry_count,
            reverse=True
        )
```

### 9.3 十条实战建议

```
┌─────────────────────────────────────────────────────────────────┐
│               Agent状态机设计十条实战建议                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 从FSM开始，需要时再升级                                       │
│     → 不要一开始就用最复杂的方案                                   │
│                                                                  │
│  2. 状态转移必须有日志                                             │
│     → 没有日志的状态机是黑盒                                       │
│                                                                  │
│  3. 实现状态持久化                                                 │
│     → Agent重启后能从断点继续                                      │
│                                                                  │
│  4. 设置超时机制                                                   │
│     → 防止Agent卡在某个状态永远不返回                               │
│                                                                  │
│  5. 设计回退路径                                                   │
│     → 每个状态都应该能回退到上一个状态                              │
│                                                                  │
│  6. 使用条件路由而非硬编码                                         │
│     → 让LLM决定下一步，而不是写死if-else                          │
│                                                                  │
│  7. 监控状态停留时间                                               │
│     → 发现异常状态（如卡在思考状态太久）                            │
│                                                                  │
│  8. 实现优雅降级                                                   │
│     → 主流程失败时，切换到简化流程                                 │
│                                                                  │
│  9. 测试所有状态转移路径                                           │
│     → 特别是错误路径和边界情况                                     │
│                                                                  │
│  10. 保持状态简洁                                                  │
│      → 超过10个状态就该考虑是否设计合理了                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 十、总结：选择适合你的方案

### 10.1 方案选择指南

| 你的情况 | 推荐方案 | 理由 |
|---------|---------|------|
| 快速原型、简单流程 | FSM + LangGraph | 简单、快速、LangGraph生态好 |
| 复杂工作流、多条件分支 | LangGraph + 状态图 | LangGraph天然支持条件路由 |
| 需要可视化编辑流程 | LangGraph Studio | 官方可视化工具 |
| 层次化任务分解 | 行为树 | 自然表达任务层次 |
| 多目标动态决策 | 效用Agent | 可以量化和优化决策 |
| 复杂Agent系统 | 混合架构 | 三层模型各司其职 |

### 10.2 核心要点回顾

1. **状态机不是万能的**：简单任务用FSM，复杂任务用行为树或混合架构
2. **LangGraph是当前最佳实践**：它在FSM基础上加入了LLM驱动的条件路由
3. **持久化和监控是生产必需**：没有持久化的状态机在生产环境不可靠
4. **测试要覆盖所有路径**：特别是错误路径和边界情况
5. **保持简洁**：状态越多越难维护，10个状态以内是理想状态

状态机是构建可靠Agent系统的基石。选择正确的方案，从简单开始，逐步演进，你就能构建出既智能又可控的Agent系统。
