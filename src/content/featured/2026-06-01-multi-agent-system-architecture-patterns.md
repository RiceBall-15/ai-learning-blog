---
title: "多Agent系统架构演进：从集中式到去中心化的协作范式"
description: "深入解析多Agent系统的架构模式演进，覆盖集中式协调、层级式委托、去中心化自组织等协作范式，结合AutoGen/CrewAI/LangGraph实战对比"
date: 2026-06-01
author: "RiceBall-15"
category: "featured"
subCategory: ai-architecture
tags: ["多Agent系统", "架构模式", "协作范式", "AutoGen", "CrewAI", "LangGraph"]
draft: false
---

# 多Agent系统架构演进：从集中式到去中心化的协作范式

## 引言：为什么需要多Agent？

单个Agent的能力边界正在快速扩展，但在面对复杂任务时，仍然面临三大核心挑战：

| 挑战 | 单Agent局限 | 多Agent解决方案 |
|------|------------|----------------|
| **任务复杂度** | 单一上下文窗口无法承载所有信息 | 任务分解与专业化Agent |
| **能力多样性** | 单一模型难以同时精通代码、分析、创意 | 专家Agent协作 |
| **容错性** | 单点失败导致整个任务失败 | 冗余与故障转移机制 |

```
                    ┌─────────────────────────────────────────┐
                    │         多Agent系统架构演进路线          │
                    └─────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │   集中式架构   │   ──▶   │   层级式架构   │   ──▶   │   去中心化架构  │
    │ (Orchestrator)│          │ (Hierarchical) │          │ (Decentralized)│
    └───────────────┘          └───────────────┘          └───────────────┘
    • 主控Agent协调            • 任务逐层分解              • 自组织协作
    • 中心化决策               • 层级权限管理              • 分布式共识
    • 简单可靠                 • 可扩展性强                • 高容错性
```

---

## 一、集中式协调架构（Orchestrator Pattern）

### 1.1 架构原理

集中式架构是最简单直接的多Agent模式：一个中央编排器（Orchestrator）负责任务分解、分配和结果聚合。

```
┌─────────────────────────────────────────────────────────────────┐
│                    集中式协调架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                        ┌─────────────┐                          │
│                        │ Orchestrator │                          │
│                        │   (主控Agent) │                          │
│                        └──────┬──────┘                          │
│                               │                                 │
│            ┌──────────────────┼──────────────────┐              │
│            ▼                  ▼                  ▼              │
│     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│     │  Research   │    │   Coding    │    │  Writing    │      │
│     │    Agent    │    │    Agent    │    │    Agent    │      │
│     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │
│            │                  │                  │              │
│            └──────────────────┴──────────────────┘              │
│                               │                                 │
│                        ┌──────▼──────┐                          │
│                        │  结果聚合    │                          │
│                        └─────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心实现

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class OrchestratorState(TypedDict):
    task: str
    subtasks: list[dict]
    results: dict[str, str]
    final_output: str

def orchestrator_node(state: OrchestratorState) -> dict:
    """主控Agent：分析任务并分解为子任务"""
    task = state["task"]
    
    # 任务分解策略
    subtasks = [
        {"id": "research", "agent": "research_agent", 
         "prompt": f"研究以下主题：{task}"},
        {"id": "analysis", "agent": "analysis_agent",
         "prompt": f"分析以下内容：{task}"},
        {"id": "writing", "agent": "writing_agent",
         "prompt": f"撰写以下内容：{task}"}
    ]
    
    return {"subtasks": subtasks}

def worker_node(state: OrchestratorState, agent_type: str) -> dict:
    """工作Agent：执行具体子任务"""
    # 根据agent_type选择对应的LLM配置
    agent_configs = {
        "research": {"model": "gpt-4", "temperature": 0.3},
        "analysis": {"model": "claude-3", "temperature": 0.1},
        "writing": {"model": "gpt-4", "temperature": 0.7}
    }
    
    # 执行任务（简化示例）
    result = execute_agent_task(agent_type, state["task"])
    
    return {"results": {agent_type: result}}

def aggregator_node(state: OrchestratorState) -> dict:
    """结果聚合：合并所有子任务结果"""
    results = state["results"]
    
    # 聚合策略：按顺序组合
    final = "\n\n".join([
        f"## {k.upper()}\n{v}" for k, v in results.items()
    ])
    
    return {"final_output": final}

# 构建工作流图
workflow = StateGraph(OrchestratorState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("research", lambda s: worker_node(s, "research"))
workflow.add_node("analysis", lambda s: worker_node(s, "analysis"))
workflow.add_node("writing", lambda s: worker_node(s, "writing"))
workflow.add_node("aggregator", aggregator_node)

# 定义边
workflow.add_edge("orchestrator", "research")
workflow.add_edge("orchestrator", "analysis")
workflow.add_edge("orchestrator", "writing")
workflow.add_edge("research", "aggregator")
workflow.add_edge("analysis", "aggregator")
workflow.add_edge("writing", "aggregator")
workflow.add_edge("aggregator", END)

workflow.set_entry_point("orchestrator")
graph = workflow.compile()
```

### 1.3 优缺点分析

| 维度 | 优点 | 缺点 |
|------|------|------|
| **复杂度** | 实现简单，易于调试 | 编排器成为瓶颈 |
| **可扩展性** | 添加新Agent容易 | 编排器负载线性增长 |
| **容错性** | 单个Agent失败不影响其他 | 编排器单点故障 |
| **适用场景** | 任务可并行、依赖关系清晰 | 需要复杂交互的任务 |

---

## 二、层级式委托架构（Hierarchical Pattern）

### 2.1 架构原理

层级式架构将任务分解为多层，每层有专门的管理者Agent，负责将任务进一步分解并委托给下层Agent。

```
┌─────────────────────────────────────────────────────────────────┐
│                    层级式委托架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     ┌─────────────────┐                         │
│                     │   CEO Agent     │                         │
│                     │  (战略决策层)    │                         │
│                     └────────┬────────┘                         │
│                              │                                  │
│            ┌─────────────────┼─────────────────┐                │
│            ▼                 ▼                 ▼                │
│     ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│     │   VP-Research│   │  VP-Engineering│   │  VP-Marketing│       │
│     │  (研究副总裁) │   │  (工程副总裁)  │   │ (市场副总裁) │       │
│     └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│            │                 │                 │                │
│     ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐       │
│     │             │   │             │   │             │        │
│  ┌──▼──┐  ┌──▼──┐ │  ┌──▼──┐  ┌──▼──┐ │  ┌──▼──┐  ┌──▼──┐   │
│  │Analyst│  │Writer│ │  │ Coder│  │Tester│ │  │Designer│  │Copywriter│   │
│  └─────┘  └─────┘ │  └─────┘  └─────┘ │  └─────┘  └─────┘   │
│                    │                    │                      │
│                    └────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心实现

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentNode:
    name: str
    role: str
    level: int
    parent: Optional['AgentNode'] = None
    children: list['AgentNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child: 'AgentNode'):
        child.parent = self
        self.children.append(child)

class HierarchicalOrchestrator:
    def __init__(self):
        self.root = AgentNode("CEO", "strategic", level=0)
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """构建层级结构"""
        # VP层
        vp_research = AgentNode("VP-Research", "research", level=1)
        vp_engineering = AgentNode("VP-Engineering", "engineering", level=1)
        vp_marketing = AgentNode("VP-Marketing", "marketing", level=1)
        
        # Worker层
        analyst = AgentNode("Analyst", "data_analysis", level=2)
        writer = AgentNode("Writer", "content_writing", level=2)
        coder = AgentNode("Coder", "code_generation", level=2)
        tester = AgentNode("Tester", "quality_assurance", level=2)
        
        # 建立层级关系
        self.root.add_child(vp_research)
        self.root.add_child(vp_engineering)
        self.root.add_child(vp_marketing)
        
        vp_research.add_child(analyst)
        vp_research.add_child(writer)
        vp_engineering.add_child(coder)
        vp_engineering.add_child(tester)
    
    def delegate_task(self, task: str, node: AgentNode = None) -> dict:
        """递归委托任务"""
        if node is None:
            node = self.root
        
        # 如果是叶子节点，执行任务
        if not node.children:
            return self._execute_task(node, task)
        
        # 否则，分解任务并委托给子节点
        subtasks = self._decompose_task(task, node)
        results = {}
        
        for child, subtask in zip(node.children, subtasks):
            results[child.name] = self.delegate_task(subtask, child)
        
        # 聚合子节点结果
        return self._aggregate_results(results, node)
    
    def _decompose_task(self, task: str, node: AgentNode) -> list[str]:
        """根据Agent角色分解任务"""
        decomposition_rules = {
            "strategic": [
                f"分析{task}的研究需求",
                f"设计{task}的技术方案",
                f"制定{task}的市场策略"
            ],
            "research": [
                f"收集{task}的相关数据",
                f"整理{task}的研究报告"
            ],
            "engineering": [
                f"实现{task}的核心功能",
                f"测试{task}的功能完整性"
            ]
        }
        
        return decomposition_rules.get(node.level, [task])
    
    def _execute_task(self, node: AgentNode, task: str) -> dict:
        """执行具体任务（简化示例）"""
        # 这里调用实际的LLM
        return {
            "agent": node.name,
            "role": node.role,
            "result": f"完成任务：{task}",
            "status": "success"
        }
    
    def _aggregate_results(self, results: dict, node: AgentNode) -> dict:
        """聚合子节点结果"""
        return {
            "node": node.name,
            "level": node.level,
            "sub_results": results,
            "summary": f"聚合{len(results)}个子任务结果"
        }

# 使用示例
orchestrator = HierarchicalOrchestrator()
result = orchestrator.delegate_task("开发一个智能客服系统")
```

### 2.3 优缺点分析

| 维度 | 优点 | 缺点 |
|------|------|------|
| **复杂度** | 职责分离清晰 | 层级过深时管理复杂 |
| **可扩展性** | 各层独立扩展 | 跨层级协作困难 |
| **容错性** | 局部故障不影响全局 | 上层故障影响所有下层 |
| **适用场景** | 大型复杂项目 | 需要频繁跨层级交互 |

---

## 三、去中心化自组织架构（Decentralized Pattern）

### 3.1 架构原理

去中心化架构中没有中央控制点，Agent之间通过协商、投票、共识等机制自主协作。

```
┌─────────────────────────────────────────────────────────────────┐
│                    去中心化自组织架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│        ┌─────────┐     协作     ┌─────────┐                     │
│        │ Agent A │◄────────────▶│ Agent B │                     │
│        │(研究专家)│             │(编码专家)│                     │
│        └────┬────┘             └────┬────┘                     │
│             │                       │                          │
│        协作 │                       │ 协作                     │
│             │    ┌─────────┐        │                          │
│             └───▶│ Agent C │◄───────┘                          │
│                  │(分析专家)│                                   │
│                  └────┬────┘                                   │
│                       │                                        │
│                  协作 │                                        │
│                       ▼                                        │
│                  ┌─────────┐                                   │
│                  │ Agent D │                                   │
│                  │(执行专家)│                                   │
│                  └─────────┘                                   │
│                                                                 │
│  特点：                                                         │
│  • 无中央控制点                                                 │
│  • Agent间直接通信                                              │
│  • 基于协议/规则自主协作                                         │
│  • 高度容错，无单点故障                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心实现

```python
import asyncio
from dataclasses import dataclass, field
from typing import Callable
import json

@dataclass
class Message:
    sender: str
    receiver: str
    content: dict
    msg_type: str  # "request", "response", "proposal", "vote"
    timestamp: float = 0

class DecentralizedAgent:
    def __init__(self, agent_id: str, capabilities: list[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.neighbors: list['DecentralizedAgent'] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.consensus_cache: dict = {}
    
    def connect(self, other: 'DecentralizedAgent'):
        """建立Agent间连接"""
        if other not in self.neighbors:
            self.neighbors.append(other)
            other.neighbors.append(self)
    
    async def broadcast(self, message: dict, msg_type: str = "proposal"):
        """广播消息给所有邻居"""
        for neighbor in self.neighbors:
            await neighbor.message_queue.put(Message(
                sender=self.agent_id,
                receiver=neighbor.agent_id,
                content=message,
                msg_type=msg_type
            ))
    
    async def handle_message(self, message: Message):
        """处理接收到的消息"""
        if message.msg_type == "proposal":
            # 收到提案，评估并投票
            vote = await self.evaluate_proposal(message.content)
            await self.send_vote(message.sender, vote)
        
        elif message.msg_type == "vote":
            # 收到投票，更新共识
            self.update_consensus(message.content)
        
        elif message.msg_type == "request":
            # 收到任务请求，评估是否接受
            if self.can_handle(message.content):
                result = await self.execute_task(message.content)
                await self.send_response(message.sender, result)
    
    async def evaluate_proposal(self, proposal: dict) -> dict:
        """评估提案并返回投票"""
        # 基于自身能力评估
        relevance = sum(1 for cap in self.capabilities 
                       if cap in proposal.get("required", []))
        
        return {
            "agent_id": self.agent_id,
            "vote": relevance > 0,
            "confidence": relevance / max(len(proposal.get("required", [])), 1)
        }
    
    def update_consensus(self, vote_data: dict):
        """更新共识状态"""
        agent_id = vote_data["agent_id"]
        self.consensus_cache[agent_id] = vote_data
    
    async def run_consensus(self, proposal: dict, timeout: float = 5.0):
        """运行共识协议"""
        # 广播提案
        await self.broadcast(proposal, "proposal")
        
        # 等待投票
        votes = []
        try:
            while len(votes) < len(self.neighbors):
                msg = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=timeout
                )
                if msg.msg_type == "vote":
                    votes.append(msg.content)
        except asyncio.TimeoutError:
            pass
        
        # 统计投票结果
        total = len(votes)
        approvals = sum(1 for v in votes if v["vote"])
        
        return {
            "proposal": proposal,
            "total_votes": total,
            "approvals": approvals,
            "passed": approvals > total / 2
        }
    
    def can_handle(self, task: dict) -> bool:
        """判断是否能处理任务"""
        required = set(task.get("required", []))
        return required.issubset(set(self.capabilities))
    
    async def execute_task(self, task: dict) -> dict:
        """执行任务（简化示例）"""
        return {
            "agent_id": self.agent_id,
            "task": task,
            "result": f"Agent {self.agent_id} 完成任务",
            "status": "success"
        }

# 创建去中心化Agent网络
async def create_decentralized_network():
    agents = [
        DecentralizedAgent("researcher", ["data_analysis", "web_search"]),
        DecentralizedAgent("coder", ["code_generation", "testing"]),
        DecentralizedAgent("reviewer", ["code_review", "quality_assurance"]),
        DecentralizedAgent("deployer", ["deployment", "monitoring"])
    ]
    
    # 建立全连接网络
    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            agent1.connect(agent2)
    
    return agents

# 运行共识示例
async def consensus_example():
    agents = await create_decentralized_network()
    
    proposal = {
        "task": "开发智能推荐系统",
        "required": ["data_analysis", "code_generation", "testing"],
        "deadline": "2026-06-15"
    }
    
    # 任意Agent发起共识
    result = await agents[0].run_consensus(proposal)
    print(f"共识结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
```

### 3.3 优缺点分析

| 维度 | 优点 | 缺点 |
|------|------|------|
| **复杂度** | 高度灵活，适应性强 | 实现复杂，调试困难 |
| **可扩展性** | 理论上无限扩展 | 通信开销指数增长 |
| **容错性** | 无单点故障，高度容错 | 一致性难以保证 |
| **适用场景** | 动态环境、高可靠性要求 | 需要严格一致性的场景 |

---

## 四、框架实战对比

### 4.1 主流框架架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    主流多Agent框架架构对比                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    AutoGen      │  │     CrewAI      │  │   LangGraph     │ │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤ │
│  │ 架构: 集中式    │  │ 架构: 层级式    │  │ 架构: 混合式    │ │
│  │ 通信: 直接对话  │  │ 通信: 任务委托  │  │ 通信: 状态传递  │ │
│  │ 状态: 对话历史  │  │ 状态: 角色记忆  │  │ 状态: 全局状态  │ │
│  │ 适用: 对话协作  │  │ 适用: 任务执行  │  │ 适用: 复杂流程  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  复杂度: ★★☆☆☆      复杂度: ★★★☆☆      复杂度: ★★★★☆          │
│  灵活性: ★★★☆☆      灵活性: ★★☆☆☆      灵活性: ★★★★★          │
│  可靠性: ★★★☆☆      可靠性: ★★★★☆      可靠性: ★★★★☆          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 框架选型决策表

| 维度 | AutoGen | CrewAI | LangGraph |
|------|---------|--------|-----------|
| **架构模式** | 集中式对话 | 层级式任务 | 状态机工作流 |
| **学习曲线** | 低 | 中 | 高 |
| **适用场景** | 多Agent对话、辩论 | 角色扮演、任务执行 | 复杂工作流、条件分支 |
| **状态管理** | 对话历史 | 角色状态 | 全局状态图 |
| **错误处理** | 基础重试 | 任务重试 | 图级别重试 |
| **可视化** | 对话日志 | 任务流程 | 图可视化 |
| **社区活跃度** | 高 | 高 | 高 |
| **企业采用** | 中 | 中 | 高 |

### 4.3 CrewAI实战示例

```python
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool

# 定义工具
class WebSearchTool(BaseTool):
    name: str = "WebSearch"
    description: str = "搜索互联网获取最新信息"
    
    def _run(self, query: str) -> str:
        # 实际实现调用搜索API
        return f"搜索结果：{query} 的相关信息"

class CodeAnalysisTool(BaseTool):
    name: str = "CodeAnalysis"
    description: str = "分析代码质量和潜在问题"
    
    def _run(self, code: str) -> str:
        # 实际实现调用代码分析工具
        return f"代码分析结果：{code}"

# 定义Agent
researcher = Agent(
    role="高级研究员",
    goal="深入研究技术趋势和最佳实践",
    backstory="你是一位资深技术研究员，擅长发现新兴技术和行业趋势",
    tools=[WebSearchTool()],
    verbose=True
)

developer = Agent(
    role="全栈开发工程师",
    goal="编写高质量、可维护的代码",
    backstory="你是一位经验丰富的全栈工程师，精通Python和现代Web技术",
    tools=[CodeAnalysisTool()],
    verbose=True
)

reviewer = Agent(
    role="技术评审专家",
    goal="确保技术方案的合理性和代码质量",
    backstory="你是一位严格的技术评审专家，对代码质量和架构设计有很高标准",
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究最新的AI Agent架构模式和最佳实践",
    expected_output="详细的技术调研报告，包含架构图和对比分析",
    agent=researcher
)

development_task = Task(
    description="基于调研结果，实现一个多Agent协作系统原型",
    expected_output="可运行的Python代码，包含完整的实现和测试",
    agent=developer
)

review_task = Task(
    description="评审调研报告和代码实现，提供改进建议",
    expected_output="详细的评审报告，包含问题列表和改进建议",
    agent=reviewer
)

# 组建团队
crew = Crew(
    agents=[researcher, developer, reviewer],
    tasks=[research_task, development_task, review_task],
    verbose=2
)

# 执行任务
result = crew.kickoff()
print(f"最终结果：{result}")
```

---

## 五、架构选型指南

### 5.1 选型决策树

```
                        ┌─────────────────┐
                        │ 任务复杂度评估   │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌─────────────┐           ┌─────────────┐
            │  简单任务    │           │  复杂任务    │
            │ (单步可完成) │           │ (多步协作)   │
            └─────────────┘           └──────┬──────┘
                    │                         │
                    ▼                         │
            ┌─────────────┐                   │
            │   单Agent   │                   │
            │   即可完成   │                   │
            └─────────────┘                   │
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
            ┌─────────────┐                                   ┌─────────────┐
            │  依赖关系清晰 │                                   │  依赖关系复杂 │
            │  可并行执行  │                                   │  需要动态协调 │
            └──────┬──────┘                                   └──────┬──────┘
                   │                                                  │
                   ▼                                                  ▼
            ┌─────────────┐                                   ┌─────────────┐
            │  集中式架构  │                                   │  去中心化架构 │
            │ (Orchestrator)│                                 │ (Decentralized)│
            └─────────────┘                                   └─────────────┘
```

### 5.2 场景匹配矩阵

| 场景类型 | 推荐架构 | 理由 |
|---------|---------|------|
| **客服系统** | 集中式 | 任务明确，可并行处理 |
| **代码审查** | 层级式 | 需要多级审核流程 |
| **研究协作** | 去中心化 | 需要动态分工和知识共享 |
| **内容创作** | 集中式 | 流程清晰，角色固定 |
| **系统运维** | 去中心化 | 需要快速响应和故障转移 |
| **产品开发** | 层级式 | 需要跨团队协调 |

---

## 六、生产化最佳实践

### 6.1 监控与可观测性

```python
import time
from dataclasses import dataclass
from typing import Optional
from contextlib import asynccontextmanager

@dataclass
class AgentMetrics:
    agent_id: str
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    total_tokens: int = 0

class MultiAgentMonitor:
    def __init__(self):
        self.metrics: dict[str, AgentMetrics] = {}
        self.task_history: list[dict] = []
    
    def record_task(self, agent_id: str, task_id: str, 
                    success: bool, duration: float, tokens: int):
        """记录任务执行指标"""
        if agent_id not in self.metrics:
            self.metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        
        metrics = self.metrics[agent_id]
        metrics.task_count += 1
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        # 更新平均响应时间
        metrics.avg_response_time = (
            (metrics.avg_response_time * (metrics.task_count - 1) + duration) 
            / metrics.task_count
        )
        metrics.total_tokens += tokens
        
        # 记录任务历史
        self.task_history.append({
            "agent_id": agent_id,
            "task_id": task_id,
            "success": success,
            "duration": duration,
            "tokens": tokens,
            "timestamp": time.time()
        })
    
    def get_health_status(self) -> dict:
        """获取系统健康状态"""
        total_tasks = sum(m.task_count for m in self.metrics.values())
        success_tasks = sum(m.success_count for m in self.metrics.values())
        
        return {
            "total_tasks": total_tasks,
            "success_rate": success_tasks / max(total_tasks, 1),
            "agent_count": len(self.metrics),
            "avg_response_time": sum(m.avg_response_time for m in self.metrics.values()) / max(len(self.metrics), 1),
            "total_tokens": sum(m.total_tokens for m in self.metrics.values())
        }

# 使用示例
monitor = MultiAgentMonitor()

@asynccontextmanager
async def track_agent_task(agent_id: str, task_id: str):
    """跟踪Agent任务执行"""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        monitor.record_task(agent_id, task_id, True, duration, 0)
    except Exception as e:
        duration = time.time() - start_time
        monitor.record_task(agent_id, task_id, False, duration, 0)
        raise

# 在Agent执行任务时使用
async with track_agent_task("researcher", "task_001"):
    result = await researcher.execute(task)
```

### 6.2 错误处理与重试

```python
import asyncio
from typing import Callable, Any
from functools import wraps

def retry_with_fallback(max_retries: int = 3, 
                       fallback_agents: list[str] = None):
    """带故障转移的重试装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    
                    # 如果有备用Agent，尝试切换
                    if fallback_agents and attempt < len(fallback_agents):
                        fallback_id = fallback_agents[attempt]
                        print(f"Switching to fallback agent: {fallback_id}")
                        # 这里可以实际切换Agent逻辑
            
            # 所有重试都失败，使用最终备用方案
            if fallback_agents:
                return await execute_fallback(
                    fallback_agents[-1], args, kwargs
                )
            
            raise last_error
        return wrapper
    return decorator

async def execute_fallback(agent_id: str, args: tuple, kwargs: dict) -> Any:
    """执行备用Agent"""
    # 实现备用Agent逻辑
    return {"status": "fallback", "agent": agent_id}

# 使用示例
@retry_with_fallback(max_retries=3, fallback_agents=["agent_b", "agent_c"])
async def critical_task(data: dict) -> dict:
    """关键任务，带故障转移"""
    # 可能失败的任务逻辑
    return {"result": "success"}
```

---

## 七、总结与展望

### 7.1 架构选择建议

```
┌─────────────────────────────────────────────────────────────────┐
│                    多Agent架构选择速查表                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  场景：简单任务 + 明确流程           → 集中式架构               │
│  场景：复杂项目 + 多级审核           → 层级式架构               │
│  场景：动态环境 + 高可靠性要求       → 去中心化架构             │
│                                                                 │
│  框架选择：                                                   │
│  • 快速原型 + 对话场景           → AutoGen                    │
│  • 角色扮演 + 任务执行           → CrewAI                     │
│  • 复杂工作流 + 生产环境         → LangGraph                  │
│                                                                 │
│  关键原则：                                                   │
│  1. 从简单架构开始，按需演进                                    │
│  2. 优先考虑可观测性和错误处理                                  │
│  3. 渐进式增加Agent数量和复杂度                                │
│  4. 建立完善的监控和告警机制                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 未来趋势

1. **自适应架构**：根据任务负载和复杂度动态切换架构模式
2. **跨框架互操作**：不同框架的Agent能够无缝协作
3. **联邦学习集成**：多Agent协作与联邦学习结合，保护数据隐私
4. **边缘计算部署**：将Agent部署到边缘设备，实现分布式智能

---

## 参考资料

1. Microsoft AutoGen: https://github.com/microsoft/autogen
2. CrewAI: https://github.com/joaomdmoura/crewai
3. LangGraph: https://github.com/langchain-ai/langgraph
4. Multi-Agent Systems: A Modern Approach to AI
5. Distributed Artificial Intelligence: Theory and Applications
