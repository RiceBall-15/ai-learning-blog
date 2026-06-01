---
title: "Letta（原MemGPT）深度解析：为AI Agent构建无限记忆系统"
description: "深入解析Letta框架的自编辑记忆架构、分层记忆管理与无限上下文窗口的实现原理，附生产级实战指南"
date: 2026-06-01
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["Letta", "MemGPT", "AI Agent", "记忆系统", "无限上下文", "长期记忆"]
draft: false
---

# Letta（原MemGPT）深度解析：为AI Agent构建无限记忆系统

## 引言：为什么LLM需要"无限记忆"？

大语言模型（LLM）有一个根本性的限制——**上下文窗口**。即使是最新的模型支持128K甚至1M tokens的上下文，但在实际Agent应用中，这个窗口会迅速被以下内容填满：

- 系统提示词（System Prompt）
- 工具调用的输入输出
- 多轮对话历史
- 检索到的外部知识

更关键的是，LLM的"记忆"是**无状态的**——每次推理都是独立的，模型本身不具备跨会话的持久化记忆能力。当对话结束，所有上下文都会丢失。

**Letta**（原名MemGPT）正是为了解决这个问题而诞生的。它的核心理念是：**让AI Agent像人类一样管理自己的记忆**——有工作记忆（短期）、有长期存储、还能主动决定什么该记住、什么该遗忘。

本文将深入解析Letta的架构设计、记忆管理机制、以及如何在生产环境中构建具有无限记忆的AI Agent系统。

---

## 一、Letta的核心设计理念

### 1.1 从MemGPT到Letta的演进

MemGPT最初是加州大学伯克利分校的研究项目，论文《MemGPT: Towards LLMs as Operating Systems》提出了一个开创性的想法：**将LLM视为操作系统中的进程，将上下文窗口视为内存，将外部存储视为磁盘**。

2024年，MemGPT团队将其产品化为Letta，并开源了完整的框架。其核心设计哲学可以总结为三点：

| 设计原则 | 说明 | 与传统方案的区别 |
|---------|------|-----------------|
| **自编辑记忆** | LLM主动决定读写哪些记忆 | 传统RAG被动检索 |
| **分层记忆架构** | 工作记忆 + 长期存储 + 归档 | 传统方案只有上下文窗口 |
| **记忆自主管理** | Agent自己决定何时压缩/归档 | 传统方案由外部逻辑控制 |

### 1.2 架构总览

```
┌─────────────────────────────────────────────┐
│                  Letta Agent                 │
├─────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────────┐   │
│  │  System Prompt │    │  Working Memory  │   │
│  │  (固定指令)    │    │  (可编辑上下文)   │   │
│  └──────────────┘    └──────────────────┘   │
│         │                    │               │
│         ▼                    ▼               │
│  ┌──────────────────────────────────────┐   │
│  │         Memory Management Layer      │   │
│  │  ┌─────────┐ ┌──────────┐ ┌───────┐ │   │
│  │  │ Core    │ │ archival │ │ recall│ │   │
│  │  │ memory  │ │ memory   │ │ memory│ │   │
│  │  └─────────┘ └──────────┘ └───────┘ │   │
│  └──────────────────────────────────────┘   │
│         │                    │               │
│         ▼                    ▼               │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │  Vector DB   │    │  Relational DB   │   │
│  │  (语义检索)   │    │  (结构化存储)     │   │
│  └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 二、三层记忆架构深度解析

Letta的记忆系统是其最核心的创新。它将记忆分为三个层次，每一层都有明确的职责和访问模式。

### 2.1 Core Memory（核心记忆）

核心记忆是**始终存在于上下文窗口中**的信息，类似于人类的"工作记忆"。它包含两个核心部分：

- **Persona**：Agent的人格定义（"你是谁"）
- **Human**：关于用户的已知信息（"用户是谁"）

```python
from letta import LettaAgent, Block

# 创建Agent并设置核心记忆
agent = LettaAgent(
    name="assistant",
    system_prompt="你是一个专业的AI助手",
)

# 更新核心记忆
agent.update_memory(
    block_label="human",
    value="用户叫张三，是一名后端工程师，主要使用Python和Go"
)

# 核心记忆会自动注入到每次推理的上下文中
response = agent.send_message("帮我设计一个微服务架构")
```

**关键设计**：Core Memory的大小是**固定且受限**的。这意味着Agent必须主动管理这部分记忆——决定哪些信息足够重要，值得永远保留在工作记忆中。

### 2.2 Archival Memory（归档记忆）

归档记忆是**超出上下文窗口的长期存储**，类似于人类的"长期记忆"。它存储对话历史、学习到的知识、用户偏好等。

归档记忆的写入通过**函数调用**实现——Agent可以主动调用 `archival_memory_insert` 将重要信息存入归档：

```python
# Agent在对话过程中主动存储重要信息
# 这个过程是自动的，由LLM自己决定
agent.send_message("记住：用户的项目使用Kubernetes部署，集群在AWS EKS上")

# 后续对话中，Agent可以检索归档记忆
agent.send_message("上次说的部署方案，帮我优化一下")
# Agent会自动搜索归档记忆，找到相关信息
```

### 2.3 Recall Memory（召回记忆）

召回记忆是**对话历史的检索索引**，类似于人类的"情景记忆"。它允许Agent回忆过去的对话片段：

```python
# 检索过去的对话
messages = agent.get_conversation_history(
    query="用户之前提过的数据库选型",
    limit=5
)

# 按时间范围检索
messages = agent.get_messages_between(
    start_time="2026-05-01",
    end_time="2026-06-01"
)
```

### 2.4 三层记忆的协作机制

```
用户输入 → Agent推理 → 决定是否需要：
  ├─ 更新 Core Memory（修改persona/human信息）
  ├─ 写入 Archival Memory（存储新知识）
  ├─ 搜索 Archival Memory（获取长期知识）
  ├─ 搜索 Recall Memory（获取对话历史）
  └─ 生成回复
```

这个过程的关键在于：**记忆的读写完全由LLM自己决定**，而不是由外部代码控制。这是Letta与传统RAG方案的本质区别。

---

## 三、自编辑记忆：Letta的核心创新

### 3.1 什么是自编辑记忆？

传统方案中，记忆管理是由**外部逻辑**控制的：
- 代码决定何时检索向量数据库
- 代码决定何时压缩对话历史
- 代码决定哪些信息存入长期存储

Letta的自编辑记忆则将这些决策**交给了LLM本身**：

```
传统方案：代码 → 决定检索 → LLM生成回复
Letta：   LLM → 决定是否需要检索 → 检索 → LLM生成回复
```

### 3.2 自编辑记忆的实现机制

Letta通过**特殊函数调用**实现自编辑记忆。Agent在推理过程中可以调用以下内置函数：

| 函数 | 作用 | 类比 |
|-----|------|------|
| `core_memory_append` | 向核心记忆追加信息 | 更新便签纸 |
| `core_memory_replace` | 替换核心记忆中的内容 | 修改便签纸 |
| `archival_memory_insert` | 向归档记忆写入信息 | 存入笔记本 |
| `archival_memory_search` | 搜索归档记忆 | 查阅笔记本 |
| `conversation_search` | 搜索对话历史 | 回忆过去对话 |
| `conversation_search_date` | 按时间搜索对话 | 按日期回忆 |

```python
# Letta Agent的内部推理过程示例
"""
用户：我的项目用的是PostgreSQL，帮我优化慢查询

Agent内部思考：
1. 用户提到了数据库选型，我应该更新核心记忆
2. 调用 core_memory_replace("human", "用户使用PostgreSQL作为主数据库")
3. 生成回复：关于PostgreSQL慢查询优化的建议...
"""
```

### 3.3 自编辑记忆 vs 传统RAG

| 维度 | 传统RAG | Letta自编辑记忆 |
|------|--------|----------------|
| **检索触发** | 代码硬编码 | LLM自主决定 |
| **存储时机** | 固定规则 | LLM判断重要性 |
| **记忆组织** | 扁平向量存储 | 分层结构化管理 |
| **上下文利用** | 检索结果拼接到prompt | 精细控制哪些信息进入上下文 |
| **记忆一致性** | 需要外部逻辑维护 | Agent自主维护和更新 |

---

## 四、生产级实战：构建带无限记忆的客服Agent

### 4.1 场景描述

构建一个企业级客服Agent，需要：
- 记住每个客户的偏好和历史
- 跨会话保持上下文
- 在对话过程中主动学习新信息
- 处理数千个客户的并发对话

### 4.2 架构设计

```
┌─────────────────────────────────────────────────┐
│                   Load Balancer                  │
├─────────────────────────────────────────────────┤
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ Agent Pod │  │ Agent Pod │  │ Agent Pod │   │
│  │  (Letta)  │  │  (Letta)  │  │  (Letta)  │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │              │              │           │
│  ┌─────▼──────────────▼──────────────▼─────┐    │
│  │           Letta Server (共享)             │    │
│  │  ┌──────────┐  ┌──────────┐             │    │
│  │  │ Agent    │  │  State   │             │    │
│  │  │ Manager  │  │  Store   │             │    │
│  │  └──────────┘  └──────────┘             │    │
│  └─────────────────────────────────────────┘    │
│        │              │              │           │
│  ┌─────▼──────┐ ┌────▼──────┐ ┌────▼──────┐   │
│  │ PostgreSQL │ │  Vector   │ │   Redis   │   │
│  │ (状态存储)  │ │    DB     │ │ (缓存)    │   │
│  └────────────┘ └───────────┘ └───────────┘   │
└─────────────────────────────────────────────────┘
```

### 4.3 完整实现

```python
from letta import LettaServer, LettaAgent, Block, create_client
from letta.schemas.agent import AgentState
from letta.schemas.message import Message

# 1. 连接Letta Server
client = create_client(base_url="http://localhost:8083")

# 2. 创建客服Agent模板
def create_customer_agent(customer_id: str, customer_name: str):
    """为每个客户创建专属Agent"""
    agent = client.create_agent(
        name=f"support-{customer_id}",
        system_prompt="""你是专业的客户服务Agent。
        
你的核心记忆管理规则：
1. 当用户提到偏好时，立即更新human块
2. 当用户报告问题时，存入archival memory
3. 当用户询问之前的问题时，搜索archival memory
4. 定期整理核心记忆，确保信息不过时""",
        model="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
        # 核心记忆配置
        blocks=[
            Block(
                label="persona",
                value="你是一个专业的客户服务Agent，擅长解决技术问题。",
                limit=2000,
            ),
            Block(
                label="human",
                value=f"客户ID: {customer_id}\n客户姓名: {customer_name}",
                limit=2000,
            ),
        ],
    )
    return agent

# 3. 处理客户对话
def handle_conversation(agent: LettaAgent, customer_message: str):
    """处理单轮对话，Agent会自动管理记忆"""
    response = client.send_message(
        agent_id=agent.id,
        role="user",
        message=customer_message,
    )
    return response

# 4. 批量创建客户Agent
customers = [
    ("C001", "张三"),
    ("C002", "李四"),
    ("C003", "王五"),
]

agents = {}
for cid, name in customers:
    agents[cid] = create_customer_agent(cid, name)

# 5. 模拟对话流程
# 客户C001第一次咨询
response = handle_conversation(agents["C001"], "我的订单#12345一直显示处理中，已经3天了")
print(response.messages[-1].content)

# 客户C001第二次咨询（跨会话记忆）
response = handle_conversation(agents["C001"], "上次那个订单问题解决了吗？")
# Agent会自动搜索archival memory，找到之前的对话记录
print(response.messages[-1].content)
```

### 4.4 记忆管理最佳实践

#### 策略一：核心记忆的精简原则

核心记忆始终占据上下文窗口，因此必须**精简且高价值**：

```python
# ❌ 错误：核心记忆塞满无关信息
human_block.value = """
用户叫张三，30岁，住在北京，喜欢打篮球，
上周三买了个手机壳，昨天又问了充电器的事情...
"""

# ✅ 正确：核心记忆只保留关键信息
human_block.value = """
客户ID: C001
姓名: 张三
偏好: 偏好在线客服，不喜欢电话沟通
技术栈: Python/Django，使用AWS
当前问题: 订单#12345延迟（待跟进）
"""
```

#### 策略二：归档记忆的结构化存储

```python
# Agent会自动将信息结构化存入归档记忆
# 但你可以通过提示词引导存储格式
system_prompt = """归档记忆管理规则：
1. 问题记录格式：[日期] 问题类型 - 描述 - 状态
2. 偏好记录格式：[类别] 偏好内容 - 来源
3. 知识记录格式：[主题] 关键信息 - 置信度
"""
```

#### 策略三：记忆过期与清理

```python
# Letta支持基于时间的记忆清理
# 配置记忆过期策略
agent_config = {
    "memory_verification": {
        "enabled": True,
        "frequency": 10,  # 每10轮对话验证一次
        "max_core_memory_age": 100,  # 核心记忆最大年龄
    }
}
```

---

## 五、Letta Server架构与部署

### 5.1 Letta Server组件

Letta Server是生产部署的核心组件，提供以下能力：

```
┌─────────────────────────────────────┐
│          Letta Server               │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │      Agent Runtime          │   │
│  │  - LLM推理引擎              │   │
│  │  - 函数调用处理              │   │
│  │  - 记忆管理逻辑              │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │      State Management       │   │
│  │  - Agent状态持久化           │   │
│  │  - 对话历史管理              │   │
│  │  - 记忆块管理                │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │      Storage Layer          │   │
│  │  - PostgreSQL (元数据)       │   │
│  │  - Vector DB (语义检索)      │   │
│  │  - File System (文件存储)    │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 5.2 Docker部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  letta-server:
    image: letta/letta-server:latest
    ports:
      - "8083:8083"
    environment:
      - LETTA_PG_URI=postgresql://user:pass@postgres:5432/letta
      - LETTA_REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: letta
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

### 5.3 性能优化配置

```python
# 生产环境配置优化
server_config = {
    # LLM推理配置
    "llm_config": {
        "model": "openai/gpt-4o-mini",  # 使用小模型做记忆管理
        "context_window": 16384,
        "temperature": 0.2,  # 低温度保证记忆操作的确定性
    },
    
    # 记忆检索配置
    "memory_config": {
        "archival_memory_enabled": True,
        "recall_memory_enabled": True,
        "max_archival_memory_results": 10,
        "embedding_batch_size": 100,
    },
    
    # 并发配置
    "concurrency": {
        "max_concurrent_agents": 50,
        "request_queue_size": 1000,
        "timeout_seconds": 30,
    },
    
    # 缓存配置
    "cache": {
        "enabled": True,
        "ttl_seconds": 300,
        "max_size": 10000,
    }
}
```

---

## 六、Letta与其他记忆框架对比

### 6.1 框架对比矩阵

| 特性 | Letta | Mem0 | Zep | LangGraph |
|------|-------|------|-----|-----------|
| **记忆类型** | 三层架构 | 向量+图 | 时序+向量 | 图状态机 |
| **自编辑** | ✅ LLM主动管理 | ❌ 外部控制 | ❌ 外部控制 | ❌ 外部控制 |
| **跨会话** | ✅ 原生支持 | ✅ 支持 | ✅ 支持 | ⚠️ 需额外实现 |
| **记忆验证** | ✅ 内置机制 | ❌ 无 | ❌ 无 | ❌ 无 |
| **生产就绪** | ✅ Letta Server | ✅ Mem0 Cloud | ✅ Zep Cloud | ✅ LangGraph Cloud |
| **学习曲线** | 中等 | 低 | 低 | 高 |
| **社区活跃度** | 高 | 高 | 中 | 高 |

### 6.2 选型建议

```
需要自编辑记忆 + 复杂Agent逻辑？
  └─ 是 → Letta
  
只需要简单的对话历史持久化？
  └─ 是 → Mem0 或 Zep

已经在用LangChain生态？
  └─ 是 → LangGraph + 自定义记忆模块

需要企业级SLA和托管服务？
  └─ 是 → Letta Cloud 或 Mem0 Cloud
```

---

## 七、进阶：记忆系统的工程挑战

### 7.1 记忆一致性问题

当多个Agent实例并发访问同一个客户的记忆时，可能出现一致性问题：

```python
# 问题场景：两个请求同时更新同一个客户的记忆
# 请求1：用户说"我改用Go了"
# 请求2：用户说"我还在用Python"
# 结果：核心记忆可能包含矛盾信息

# 解决方案：乐观锁 + 冲突合并
class MemoryConsistencyManager:
    def __init__(self, client):
        self.client = client
    
    def update_with_conflict_resolution(self, agent_id, block_label, new_value):
        """带冲突解决的记忆更新"""
        # 1. 获取当前版本
        current = self.client.get_agent(agent_id)
        current_block = current.get_block(block_label)
        
        # 2. 检查是否有并发修改
        if current_block.updated_at > self.last_read_time:
            # 3. 合并冲突
            merged = self.merge_memory_values(
                current_block.value, 
                new_value
            )
            new_value = merged
        
        # 4. 更新记忆
        self.client.update_block(
            block_id=current_block.id,
            value=new_value
        )
    
    def merge_memory_values(self, old, new):
        """智能合并记忆值"""
        # 使用LLM进行智能合并
        return self.client.llm.complete(
            f"请合并以下两段记忆信息，保留最新且不矛盾的内容：\n"
            f"现有记忆：{old}\n新信息：{new}"
        )
```

### 7.2 记忆检索效率

随着归档记忆的增长，检索效率会下降。优化策略：

```python
# 策略1：分层索引
# 将归档记忆按主题/时间分区
memory_index = {
    "technical": VectorIndex(agent_id, topic="technical"),
    "business": VectorIndex(agent_id, topic="business"),
    "temporal": {
        "2026-Q1": VectorIndex(agent_id, period="2026-Q1"),
        "2026-Q2": VectorIndex(agent_id, period="2026-Q2"),
    }
}

# 策略2：记忆压缩
# 定期将旧记忆压缩为摘要
async def compress_old_memories(agent_id, days_threshold=30):
    """压缩超过30天的归档记忆"""
    old_memories = await get_memories_older_than(agent_id, days_threshold)
    
    # 用LLM生成摘要
    summary = await llm.complete(
        f"请将以下记忆片段压缩为简洁的摘要：\n{old_memories}"
    )
    
    # 替换原记忆
    await replace_memories(agent_id, old_memories, summary)
```

### 7.3 多Agent记忆共享

在多Agent协作场景中，Agent之间需要共享记忆：

```python
# 场景：多个Agent协作处理一个复杂任务
# Agent A：需求分析师
# Agent B：技术架构师
# Agent C：开发者

# 解决方案：共享记忆池
class SharedMemoryPool:
    def __init__(self, pool_id):
        self.pool_id = pool_id
        self.shared_blocks = {}  # 共享记忆块
    
    def add_agent(self, agent_id, read_only=False):
        """将Agent加入共享池"""
        # 为Agent添加共享记忆块
        client.create_block(
            agent_id=agent_id,
            label=f"shared_{self.pool_id}",
            value="",
            limit=4000,
            read_only=read_only,  # 只读Agent不能修改共享记忆
        )
    
    def broadcast(self, message):
        """向所有Agent广播信息"""
        for agent_id in self.agents:
            client.update_block(
                block_id=f"shared_{self.pool_id}",
                value=message
            )
```

---

## 八、性能基准与最佳实践

### 8.1 性能基准

在标准配置下（GPT-4o-mini + pgvector）：

| 操作 | 延迟 | 吞吐量 |
|------|------|--------|
| 单轮对话 | 800-1500ms | 15-25 req/s |
| 记忆搜索 | 50-200ms | 100+ req/s |
| 记忆写入 | 100-300ms | 80+ req/s |
| Agent创建 | 200-500ms | 50+ req/s |

### 8.2 最佳实践清单

1. **核心记忆保持精简**：控制在1000 tokens以内，只放最关键的信息
2. **归档记忆结构化**：使用一致的格式存储，便于检索
3. **定期记忆清理**：压缩过时信息，保持记忆质量
4. **监控记忆使用率**：跟踪核心记忆填充率，避免溢出
5. **使用小模型做记忆管理**：记忆操作不需要强推理能力，用小模型降低成本
6. **实施记忆验证**：定期检查记忆的一致性和准确性
7. **备份记忆数据**：定期备份PostgreSQL和向量数据库

---

## 九、总结与展望

Letta（MemGPT）代表了AI Agent记忆系统的一个重要方向：**让Agent自己管理自己的记忆**。这种自编辑记忆的范式，相比传统的外部控制方案，具有以下优势：

1. **更灵活**：Agent可以根据对话内容动态决定记忆策略
2. **更高效**：LLM直接判断什么信息重要，减少无效检索
3. **更自然**：模拟人类的记忆管理方式，用户体验更好

未来，随着LLM推理能力的提升和上下文窗口的扩大，Letta的记忆管理策略也会不断演进。我们可以期待：

- **更智能的记忆压缩**：LLM自动判断何时压缩、如何压缩
- **跨Agent记忆共享**：多Agent协作场景下的记忆同步
- **记忆推理**：基于记忆进行复杂的推理和规划

对于想要构建具有长期记忆的AI Agent系统的团队，Letta提供了一个成熟且经过验证的解决方案。它的自编辑记忆架构，为Agent的智能化程度带来了质的飞跃。

---

## 参考资料

1. Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems" - UC Berkeley, 2023
2. Letta官方文档: https://docs.letta.com/
3. Letta GitHub: https://github.com/letta-ai/letta
4. Mem0: https://github.com/mem0ai/mem0
5. Zep: https://www.getzep.com/
