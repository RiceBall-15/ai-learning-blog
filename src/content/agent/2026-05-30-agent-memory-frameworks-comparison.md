---
title: "AI Agent记忆框架深度对比：Mem0 vs Letta vs Zep vs LangGraph"
description: "深度解析四大主流AI Agent记忆框架的架构设计、API风格、存储方案与适用场景，附完整代码示例与对比分析。"
date: 2026-05-30
author: RiceBall-15
category: agent
subCategory: agent-memory
tags: [Mem0, Letta, MemGPT, Zep, LangGraph, 记忆框架, 对比分析]
draft: false
---

# AI Agent记忆框架深度对比：Mem0 vs Letta vs Zep vs LangGraph

## 引言

LLM的上下文窗口限制是构建持久化AI Agent的核心瓶颈。即使GPT-4o的128K上下文窗口，在多轮对话、长期任务记忆、跨会话偏好等场景下仍捉襟见肘。2025-2026年，四大主流记忆框架逐渐成型：**Mem0** 专注通用记忆层、**Letta**（原MemGPT）实现自管理的有状态Agent、**Zep** 构建时序知识图谱、**LangGraph** 提供基于检查点的工作流记忆。本文从架构设计、API设计、存储方案、检索策略等维度进行深度对比，并给出选型建议。

## 一、Mem0：通用Agent记忆层

### 1.1 架构概览

Mem0定位为「AI Agent的长期记忆基础设施」，提供即插即用的记忆提取、存储、检索能力。其V2版本引入了**图记忆（Graph Memory）**和**自动去重（Dedup）**两大核心特性。

```
┌─────────────────────────────────────────────┐
│              Mem0 Architecture               │
├─────────────────────────────────────────────┤
│                                             │
│  User Message ──► Extractor (LLM) ──┐      │
│                                      │      │
│  ┌──────────────┐    ┌──────────────┐│      │
│  │  Vector Store │    │ Graph Memory ││      │
│  │  (Qdrant/     │◄───│ (Neo4j/      ││      │
│  │   PG/Redis)   │    │  Memgraph)   ││      │
│  └──────┬───────┘    └──────┬───────┘│      │
│         │                   │         │      │
│         └───────┬───────────┘         │      │
│                 ▼                     │      │
│          Dedup & Merge                │      │
│                 │                     │      │
│                 ▼                     │      │
│          Memory Search               │      │
│          (Semantic + Graph)           │      │
└─────────────────────────────────────────────┘
```

V2关键改进：
- **图记忆**：自动从对话中提取实体关系，构建知识图谱，支持多跳推理
- **去重机制**：相同语义的记忆自动合并，避免冗余存储
- **多租户支持**：user_id + agent_id + run_id 三级命名空间隔离

### 1.2 核心API

```python
from mem0 import Memory

# 初始化，V2默认启用图记忆
memory = Memory(
    version="v2",
    config={
        "llm": {
            "provider": "openai",
            "config": {"model": "gpt-4o-mini"}
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"}
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {"collection_name": "agent_memory"}
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password"
            }
        }
    }
)

# 添加记忆 - LLM自动提取关键事实
memory.add(
    "我更喜欢在晚上10点后收到项目更新",
    user_id="user_001",
    agent_id="assistant_main"
)

# 搜索记忆 - 同时查询向量和图
results = memory.search(
    "什么时候给用户发项目更新？",
    user_id="user_001",
    agent_id="assistant_main"
)

# 返回示例:
# [{"id": "mem_abc123",
#   "memory": "用户偏好在晚上10点后收到项目更新",
#   "score": 0.92,
#   "metadata": {"created_at": "2026-05-30T10:00:00Z"}}]

# 获取所有记忆
all_memories = memory.get_all(user_id="user_001")

# 更新记忆
memory.update(memory_id="mem_abc123", data="用户改为喜欢早上9点收到更新")

# 删除记忆
memory.delete(memory_id="mem_abc123")
```

### 1.3 去重机制

Mem0 V2的核心优势之一是自动去重。当添加相似记忆时，系统会：

1. **语义相似度检测**：与已有记忆计算embedding距离
2. **冲突判断**：如果新记忆与旧记忆矛盾，更新旧记忆而非新增
3. **增量合并**：如果新记忆是旧记忆的补充，合并为一条完整记忆

```python
# 添加第一条记忆
memory.add("我的工作邮箱是 alice@company.com", user_id="alice")
# 存储: "我的工作邮箱是 alice@company.com"

# 添加相关但不矛盾的记忆
memory.add("个人邮箱是 alice@gmail.com", user_id="alice")
# 存储: 两条独立记忆

# 添加矛盾记忆 - 触发更新
memory.add("我的工作邮箱改成了 alice@newcompany.com", user_id="alice")
# 自动更新旧记忆为新邮箱地址（去重生效）
```

## 二、Letta（原MemGPT）：自管理有状态Agent

### 2.1 架构概览

Letta的核心思想来自操作系统虚拟内存——**让LLM自己管理记忆**。Agent拥有类似OS的分层内存结构，并通过「心跳」机制自主决定何时压缩、归档、检索记忆。

```
┌──────────────────────────────────────────────────┐
│              Letta Memory Hierarchy               │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  In-Context Memory (系统提示注入)         │   │
│  │  ┌────────────┐  ┌────────────────────┐  │   │
│  │  │ Core Memory │  │ Recall Memory      │  │   │
│  │  │ (持久画像)  │  │ (对话历史检索)     │  │   │
│  │  └────────────┘  └────────────────────┘  │   │
│  │  ┌────────────┐  ┌────────────────────┐  │   │
│  │  │ Archival   │  │ Working Context    │  │   │
│  │  │ Memory     │  │ (临时变量/草稿)    │  │   │
│  │  │ (长期存储) │  └────────────────────┘  │   │
│  │  └────────────┘                          │   │
│  └──────────────────────────────────────────┘   │
│                    ▲                             │
│           Agent自主管理 (心跳机制)                │
│                    ▼                             │
│  ┌──────────────────────────────────────────┐   │
│  │  Persistence Layer (SQLite/Postgres)      │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

Letta将记忆分为四层：
- **Core Memory**：始终在上下文中的核心信息（人物画像、关键约束）
- **Recall Memory**：可检索的对话历史，按语义或时间回溯
- **Archival Memory**：长期存储，容量无限，通过搜索访问
- **Working Context**：当前任务的临时工作变量

### 2.2 核心API

```python
from letta import create_client

# 创建Agent，定义Core Memory初始状态
client = create_client()

agent = client.create_agent(
    name="personal_assistant",
    # Core Memory初始值 - 这些始终在上下文中
    memory={
        "persona": "你是一个专业的个人助手",
        "human": "用户是软件工程师，偏好Python，住在东京"
    },
    # 可用工具
    tools=["web_search", "read_file", "write_file"],
    model="openai/gpt-4o"
)

# Agent会自动管理记忆 - 你只需正常对话
response = client.send_message(
    agent_id=agent.id,
    role="user",
    message="帮我规划明天的东京Tech Meetup行程"
)

# 查看Agent当前Core Memory（由Agent自己维护）
core_memory = client.get_agent_memory(agent.id)
print(core_memory["persona"])
# → "你是一个专业的个人助手"
print(core_memory["human"])
# → "用户是软件工程师，偏好Python，住在东京
#    原计划明天参加东京Tech Meetup，需要交通规划"

# Agent自动调用core_memory_append/replace更新画像
# 无需手动管理

# 手动注入Archival Memory（如外部数据源）
client.agent_archival_memory_insert(
    agent_id=agent.id,
    content="Tokyo Tech Meetup 2026: 5月31日14:00-18:00, 秋叶原UDX"
)
```

### 2.3 心跳机制与自主记忆管理

Letta独特之处在于Agent通过**心跳（Heartbeat）**自主管理记忆：

```python
# Agent在每次推理后会决定是否需要:
# 1. 更新Core Memory（core_memory_replace）
# 2. 搜索Recall Memory（conversation_search）
# 3. 存入Archival Memory（archival_memory_insert）
# 4. 搜索Archival Memory（archival_memory_search）

# 心跳函数 - Agent定时执行
def heartbeat(agent_id):
    """Agent自主决定是否需要管理记忆"""
    # Letta在后台触发Agent推理
    # Agent可以调用记忆管理函数
    response = client.send_message(
        agent_id=agent_id,
        role="user",
        message="[system] heartbeat - 请检查并整理你的记忆"
    )
    return response

# 在长时间运行的Agent中定期调用
import schedule
schedule.every(5).minutes.do(heartbeat, agent_id=agent.id)
```

这种「Agent自管理记忆」的设计让Letta在长期交互场景下表现出色——Agent自己知道什么时候需要记住什么、遗忘什么、压缩什么。

## 三、Zep：时序知识图谱记忆

### 3.1 架构概览

Zep的核心创新是**时序知识图谱（Temporal Knowledge Graph）**——不仅记录实体关系，还记录关系随时间的变化。

```
┌───────────────────────────────────────────────┐
│              Zep Architecture                  │
├───────────────────────────────────────────────┤
│                                               │
│  Conversation Stream ──► Entity Extractor     │
│         │                      │              │
│         ▼                      ▼              │
│  ┌───────────┐      ┌──────────────────┐     │
│  │  Chat      │      │  Knowledge Graph │     │
│  │  History   │      │  (Neo4j)         │     │
│  │  (Postgres)│      │  ┌─────────┐    │     │
│  └───────────┘      │  │ Entity  │    │     │
│                      │  │  A      │    │     │
│  ┌───────────┐      │  │ / \     │    │     │
│  │  Semantic  │      │  │B   C    │    │     │
│  │  Search    │      │  └─────────┘    │     │
│  │  Index     │      │  + Timestamps   │     │
│  └───────────┘      │  + Validity     │     │
│                      └──────────────────┘     │
│                             │                 │
│                             ▼                 │
│              Graph-enhanced RAG              │
│                                               │
└───────────────────────────────────────────────┘
```

### 3.2 核心API

```python
import httpx

# 初始化Zep客户端
from zep_cloud import ZepClient

zep = ZepClient(api_key="your-api-key", project="my-agent")

# 创建会话
session = zep.conversation.create(
    session_id="chat_001",
    user_id="user_bob"
)

# 添加消息 - 自动构建知识图谱
zep.conversation.add_messages(
    session_id="chat_001",
    messages=[
        {"role": "user", "content": "我是Bob，在Google工作，负责Gemini项目的后端"},
        {"role": "assistant", "content": "你好Bob！Gemini后端是很有挑战的项目。"},
        {"role": "user", "content": "下个月我要转到DeepMind团队了"},
    ]
)
# Zep自动提取: Bob→works_at→Google, Bob→will_transfer→DeepMind
# 并标记时间: 2026年6月生效

# 语义搜索 - 结合图谱上下文
results = zep.conversation.search(
    session_id="chat_001",
    query="Bob在哪里工作？",
    search_scope="messages",
    limit=5
)

# 图谱搜索 - 直接查询实体关系
graph = zep.conversation.get_entity_graph(session_id="chat_001")

# 获取摘要（自动维护）
summary = zep.conversation.get_summary(session_id="chat_001")

# 获取用户画像（跨会话）
profile = zep.user.get("user_bob")
print(profile.facts)
# → ["Bob在Google工作", "Bob即将转到DeepMind", ...]
```

### 3.3 时序感知检索

Zep的独特价值在于**时序感知**——当实体关系随时间变化时，能正确回答「现在」的问题：

```python
# 传统方案: Bob在Google工作 → 永远返回"Google"
# Zep方案: 自动追踪关系的时间有效性

# 当Bob真的转到DeepMind后，添加新对话
zep.conversation.add_messages(
    session_id="chat_001",
    messages=[
        {"role": "user", "content": "我今天正式加入DeepMind了！"}
    ]
)

# 查询"Bob在哪工作" - Zep返回最新的DeepMind
# 而非过时的Google
results = zep.conversation.search(
    session_id="chat_001",
    query="Bob在哪里工作？"
)
# → "Bob目前在DeepMind工作"（时序感知）

# 也可以查询历史状态
results = zep.conversation.search(
    session_id="chat_001",
    query="Bob之前在哪里工作？"
)
# → "Bob之前在Google工作"
```

## 四、LangGraph：工作流级别的记忆检查点

### 4.1 架构概览

LangGraph的记忆方案不同于上述三者——它不是专门的记忆系统，而是**图执行引擎的检查点机制**，核心是状态的持久化和可恢复。

```
┌──────────────────────────────────────────────┐
│           LangGraph Checkpointing             │
├──────────────────────────────────────────────┤
│                                              │
│  State Graph                                 │
│  ┌──────┐   ┌──────┐   ┌──────┐            │
│  │Node A├──►│Node B├──►│Node C│            │
│  └──┬───┘   └──┬───┘   └──┬───┘            │
│     │          │          │                  │
│     ▼          ▼          ▼                  │
│  ┌──────────────────────────────────────┐   │
│  │         Checkpointer                 │   │
│  │  ┌─────────┐  ┌──────────────────┐   │   │
│  │  │ SQLite  │  │ Thread-level     │   │   │
│  │  │ /Postgres│ │ State Snapshots  │   │   │
│  │  └─────────┘  └──────────────────┘   │   │
│  │  ┌────────────────────────────────┐   │   │
│  │  │  Human-in-the-loop Breakpoints│   │   │
│  │  │  Time Travel / Replay         │   │   │
│  │  └────────────────────────────────┘   │   │
│  └──────────────────────────────────────┘   │
│                                              │
└──────────────────────────────────────────────┘
```

### 4.2 核心API

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# 或使用持久化存储
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from operator import add

# 定义状态Schema
class AgentState(TypedDict):
    messages: Annotated[list, add]
    memory: Annotated[list, add]  # 跨步骤累积的记忆
    step_count: int

# SQLite检查点（轻量开发）
with SqliteSaver.from_conn_string("checkpoints.db") as memory:
    graph = StateGraph(AgentState)
    
    # 定义节点
    def think_node(state: AgentState):
        return {
            "messages": ["[思考] 正在分析用户需求..."],
            "step_count": state["step_count"] + 1
        }
    
    def act_node(state: AgentState):
        # 可以读取之前的状态
        history = state["messages"]
        return {
            "messages": ["[执行] 已完成操作"],
            "memory": [f"Step {state['step_count']}: 执行了某操作"]
        }
    
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_edge(START, "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", END)
    
    app = graph.compile(checkpointer=memory)
    
    # 使用线程ID隔离会话状态
    config = {"configurable": {"thread_id": "session_001"}}
    
    # 第一次调用
    result = app.invoke(
        {"messages": ["帮我分析市场数据"], "memory": [], "step_count": 0},
        config=config
    )
    
    # 第二次调用 - 自动恢复上次状态
    result = app.invoke(
        {"messages": ["继续刚才的分析"], "memory": [], "step_count": 0},
        config=config
    )
    # step_count 会从上次的值继续，messages自动累积

    # Postgres生产级方案
    # with PostgresSaver.from_conn_string("postgresql://...") as memory:
    #     app = graph.compile(checkpointer=memory)
```

### 4.3 时间旅行（Time Travel）

LangGraph检查点的独特能力是**状态回溯**：

```python
# 获取某个检查点的状态快照
config = {"configurable": {"thread_id": "session_001"}}

# 列出所有检查点
states = list(app.get_state_history(config))

# 回溯到某个中间状态并重新执行
target_config = states[2].config  # 第3个检查点
app.invoke(
    {"messages": ["从这里开始换一个方向"], "memory": [], "step_count": 0},
    config=target_config
)
# 后续执行基于该检查点，而非最新状态
```

## 五、综合对比

| 维度 | Mem0 V2 | Letta | Zep | LangGraph |
|------|---------|-------|-----|-----------|
| **核心定位** | 通用记忆层 | 有状态Agent引擎 | 时序知识图谱 | 工作流检查点 |
| **存储方案** | Qdrant/PG/Redis + Neo4j | SQLite/Postgres | Neo4j + Postgres | SQLite/Postgres/Redis |
| **记忆类型** | 向量 + 图 | 分层(核心/回忆/归档) | 图 + 时序关系 | 状态快照 |
| **检索方式** | 语义搜索 + 图遍历 | Agent自主决定 | 语义 + 图 + 时间过滤 | 检查点恢复 |
| **API风格** | 企业级SDK | 自管理Agent | RESTful | 图执行引擎 |
| **去重能力** | ✅ V2自动去重 | ❌ 需手动管理 | ⚠️ 部分（实体合并） | ❌ 不适用 |
| **时序感知** | ⚠️ 基础 | ⚠️ 基础 | ✅ 核心特性 | ⚠️ 仅检查点顺序 |
| **多Agent** | ✅ 命名空间隔离 | ✅ 多Agent实例 | ✅ 用户级隔离 | ✅ 线程级隔离 |
| **LangChain集成** | ✅ 原生支持 | ⚠️ 社区集成 | ✅ 原生支持 | ✅ 原生支持 |
| **自部署** | ✅ Docker/K8s | ✅ Docker/K8s | ✅ Docker/K8s | ✅ 纯Python |
| **云托管** | ✅ mem0.ai | ✅ letta.com | ✅ getzep.com | ❌ 无（自部署） |
| **开源协议** | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT |
| **价格(云)** | 免费层 + 付费 | 免费层 + 付费 | 免费层 + 付费 | 免费（开源） |

## 六、选型建议

### 选 Mem0：当你需要「即插即用」的通用记忆

```python
# 适用场景:
# - 已有Agent框架，只需添加记忆能力
# - 需要跨用户/跨Agent的记忆隔离
# - 对去重和记忆合并有明确需求
# - 快速原型开发

from mem0 import Memory
memory = Memory()
# 三行代码接入记忆能力
memory.add("用户偏好深色主题", user_id="user_1")
```

**推荐理由**：API最简洁，去重能力最强，图记忆V2让多跳推理成为可能。

### 选 Letta：当你要构建「有自主记忆管理」的长期Agent

```python
# 适用场景:
# - 长期陪伴型Agent（个人助手、教育Agent）
# - 需要Agent自主决定记住/遗忘
# - 多轮复杂对话，上下文容易溢出
# - 需要Core Memory始终在上下文中

# 推荐理由: 唯一让LLM自己管理记忆的方案，
# 最接近人类记忆的工作方式
```

### 选 Zep：当时间线是关键

```python
# 适用场景:
# - 客户关系管理（CRM Agent）
# - 项目管理Agent（需要追踪状态变化）
# - 医疗/法律等需要精确时间线的场景
# - 多轮对话中实体关系频繁变化

# 推荐理由: 时序知识图谱是独有能力，
# "Bob现在在哪工作"vs"Bob之前在哪工作"可以区分
```

### 选 LangGraph：当你已经在用LangGraph构建工作流

```python
# 适用场景:
# - 已有LangGraph工作流，需要状态持久化
# - Human-in-the-loop需要状态可恢复
# - 多步骤任务需要时间旅行/重试
# - 不需要语义记忆，只需要状态快照

# 推荐理由: 不是记忆系统，而是工作流状态管理，
# 与LangGraph生态深度绑定
```

## 七、组合使用策略

实际项目中，这些框架并不互斥。一个成熟的Agent架构可能同时使用：

```python
# 架构示例: LangGraph + Mem0
from langgraph.graph import StateGraph
from mem0 import Memory

class AdvancedAgentState(TypedDict):
    messages: Annotated[list, add]
    user_id: str

# Mem0负责长期记忆（跨会话）
long_term_memory = Memory(version="v2")

# LangGraph负责工作流状态（会话内）
def query_with_memory(state: AdvancedAgentState):
    # 从Mem0检索相关长期记忆
    memories = long_term_memory.search(
        state["messages"][-1],
        user_id=state["user_id"]
    )
    
    # 注入上下文
    context = "\n".join([m["memory"] for m in memories])
    
    # LLM推理
    response = llm.invoke(f"相关记忆:\n{context}\n\n用户: {state['messages'][-1]}")
    
    # 异步存储新记忆
    long_term_memory.add(
        state["messages"][-1],
        user_id=state["user_id"]
    )
    
    return {"messages": [response]}

# LangGraph检查点管理会话内状态
graph = StateGraph(AdvancedAgentState)
graph.add_node("query", query_with_memory)
# ... 配置检查点 ...
```

## 总结

| 需求 | 推荐方案 |
|------|---------|
| 快速接入记忆，API简洁 | **Mem0** |
| Agent自主管理记忆 | **Letta** |
| 时序关系追踪 | **Zep** |
| 工作流状态持久化 | **LangGraph** |
| 完整长期+短期记忆 | **Mem0 + LangGraph** |
| 自主记忆+知识图谱 | **Letta + Zep** |

记忆框架的选择本质上取决于你的Agent需要「记住什么」以及「如何使用记忆」。Mem0是最佳通用选择，Letta在自主性上领先，Zep的时序能力独一无二，LangGraph则是工作流场景的自然延伸。理解每个框架的设计哲学，才能做出正确的技术决策。
