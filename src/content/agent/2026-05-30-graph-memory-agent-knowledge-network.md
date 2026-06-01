---
title: '基于图的记忆系统：Agent如何构建知识网络'
description: '用知识图谱增强Agent记忆，实现关系推理与多跳查询的图记忆架构实战'
date: 2026-05-30
author: 'RiceBall-15'
category: 'agent'
subCategory: agent-memory
tags: ['Graph Memory', '知识图谱', 'Neo4j', '向量图混合', '关系推理']
draft: false
---

# 基于图的记忆系统：Agent如何构建知识网络

## 引言：从向量到图谱的范式迁移

在构建能长期记忆、跨会话推理的 AI Agent 时，大多数开发者的第一选择是向量数据库——把对话文本 embedding 后存入 Pinecone、Milvus 或 ChromaDB，需要时做语义检索。这在「找相似内容」的场景下足够了，但一旦涉及**实体关系推理**、**多跳查询**和**因果链追溯**，纯向量方案就开始力不从心。

图记忆（Graph Memory）将记忆组织为**实体-关系-实体**的有向图，每个节点携带属性，每条边标注关系类型。这种结构天然支持路径搜索、子图查询和结构化推理，是构建真正"理解知识网络"的 Agent 的关键基础设施。

本文将从架构设计到生产实战，完整拆解图记忆系统的核心组件。

---

## 1. 向量记忆的天花板：为什么 Agent 需要图？

### 1.1 向量检索的本质局限

向量记忆的工作方式是：文本 → embedding → 存入向量库 → 相似度检索。它的核心假设是**语义相近的内容在向量空间中距离近**。这个假设在以下场景中失效：

| 场景 | 向量记忆表现 | 问题 |
|------|-------------|------|
| "张三的老板是谁？" | 检索到张三相关的文本片段 | 无法精确命中"汇报关系"这条边 |
| "李四管理的项目中，哪个涉及AI？" | 可能返回李四或AI相关的碎片 | 缺乏"管理→项目→涉及技术"的多跳路径 |
| "上次王五反馈的bug和哪个模块有关？" | 返回王五相关的所有bug | 无法区分不同bug的不同模块归属 |
| "从用户投诉到根因的完整链路" | 无能为力 | 因果链是图结构，不是相似度问题 |

### 1.2 图记忆的核心优势

图记忆（知识图谱）将信息组织为三元组（Triple）：

```
(实体A) --[关系]--> (实体B)
```

例如：

```
(张三) --[汇报给]--> (李经理)
(李四) --[管理]--> (智能客服项目)
(智能客服项目) --[使用技术]--> (Neo4j)
(王五) --[反馈]--> (Bug#1024)
(Bug#1024) --[影响模块]--> (支付模块)
```

这种结构允许：
- **精确关系查询**：直接遍历特定关系类型的边
- **多跳推理**：A→B→C→D 的路径发现
- **子图展开**：从某个实体出发，展开其完整的关系网络
- **因果链追溯**：从结果沿边反向查找原因

---

## 2. 图记忆架构：实体-关系图的设计

### 2.1 核心数据模型

```
┌─────────────────────────────────────────────────────┐
│                   Agent Memory Graph                 │
│                                                      │
│  ┌──────────┐    [works_at]    ┌──────────────┐     │
│  │  Person   │───────────────▶│  Organization │     │
│  │ -name     │                │ -name         │     │
│  │ -role     │                │ -industry     │     │
│  │ -embedding│                │ -embedding    │     │
│  └─────┬────┘                └───────────────┘     │
│        │                                             │
│   [discussed]                                        │
│        │                                             │
│        ▼                                             │
│  ┌──────────┐    [related_to]  ┌──────────────┐     │
│  │  Topic    │───────────────▶│  Topic        │     │
│  │ -name     │                │ -name         │     │
│  │ -category │                │ -category     │     │
│  └─────┬────┘                └───────────────┘     │
│        │                                             │
│   [mentioned_in]                                    │
│        │                                             │
│        ▼                                             │
│  ┌──────────┐    [has_entity]  ┌──────────────┐     │
│  │ Conversation│─────────────▶│  Entity       │     │
│  │ -timestamp │               │ -type         │     │
│  │ -content   │               │ -properties   │     │
│  └──────────┘                └──────────────┘     │
└─────────────────────────────────────────────────────┘
```

### 2.2 节点和边的类型定义

一个典型的 Agent 图记忆系统包含以下核心类型：

```python
from enum import Enum

class NodeType(Enum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    TOPIC = "Topic"
    CONVERSATION = "Conversation"
    ENTITY = "Entity"
    EVENT = "Event"
    DOCUMENT = "Document"
    USER_PREFERENCE = "UserPreference"

class EdgeType(Enum):
    # 组织关系
    WORKS_AT = "WORKS_AT"
    MANAGES = "MANAGES"
    REPORTS_TO = "REPORTS_TO"
    # 对话关系
    DISCUSSED = "DISCUSSED"
    MENTIONED_IN = "MENTIONED_IN"
    ASKED_ABOUT = "ASKED_ABOUT"
    # 知识关系
    RELATED_TO = "RELATED_TO"
    CAUSES = "CAUSES"
    PART_OF = "PART_OF"
    # 记忆关系
    REMEMBERS = "REMEMBERS"
    FORGOT = "FORGOT"
    UPDATED = "UPDATED"
```

---

## 3. Neo4j 作为 Agent 记忆后端

### 3.1 为什么选择 Neo4j

Neo4j 是最成熟的图数据库，提供：
- **Cypher 查询语言**：声明式图模式匹配
- **ACID 事务**：保证记忆写入的一致性
- **APOC 插件**：内置 BFS/DFS、最短路径等图算法
- **向量索引（Neo4j 5.x+）**：原生支持向量检索，实现图+向量混合

### 3.2 连接与初始化

```python
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

class AgentGraphMemory:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._ensure_constraints()

    def _ensure_constraints(self):
        """确保唯一性约束"""
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:Person) REQUIRE n.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:Organization) REQUIRE n.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:Topic) REQUIRE n.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:Conversation) REQUIRE n.id IS UNIQUE
            """)
            # 为所有节点创建向量索引（Neo4j 5.x+）
            session.run("""
                CREATE VECTOR INDEX IF NOT EXISTS
                entity_embedding FOR (n:Entity) ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
```

### 3.3 图记忆的核心 CRUD 操作

```python
    def add_entity(self, node_type, entity_id, properties, text=None):
        """添加或更新实体节点"""
        embedding = self.embedder.encode(text or str(properties)).tolist() \
            if text else None
        properties["id"] = entity_id
        if embedding:
            properties["embedding"] = embedding

        with self.driver.session() as session:
            session.run(f"""
                MERGE (n:{node_type} {{id: $id}})
                SET n += $props
            """, id=entity_id, props=properties)

    def add_relation(self, source_id, target_id, rel_type, properties=None):
        """添加关系边"""
        with self.driver.session() as session:
            session.run(f"""
                MATCH (a {{id: $source}})
                MATCH (b {{id: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $props
            """, source=source_id, target=target_id,
                props=properties or {})

    def get_entity_network(self, entity_id, depth=2):
        """获取实体的关系网络（子图展开）"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (start {id: $id})-[*1..$depth]-(neighbor)
                RETURN path, length(path) AS depth
                ORDER BY depth
            """, id=entity_id, depth=depth)
            return [record["path"] for record in result]
```

---

## 4. 从对话中构建知识图谱

### 4.1 实体与关系提取

核心思路：用 LLM 从对话文本中提取结构化的三元组，然后写入图数据库。

```python
import json
from openai import OpenAI

EXTRACTION_PROMPT = """你是一个知识图谱构建专家。从以下对话中提取实体和关系。

对话内容：
{conversation}

请以JSON格式返回提取的三元组列表：
{{
  "entities": [
    {{"id": "unique_id", "type": "Person|Organization|Topic|...", 
      "name": "实体名称", "properties": {{"key": "value"}}}},
    ...
  ],
  "relations": [
    {{"source": "entity_id", "target": "entity_id", 
      "type": "RELATION_TYPE", "properties": {{"key": "value"}}}},
    ...
  ]
}}

只返回JSON，不要其他内容。"""

class ConversationToGraph:
    def __init__(self, memory: AgentGraphMemory, llm_client: OpenAI):
        self.memory = memory
        self.llm = llm_client

    def extract_and_store(self, conversation_id, conversation_text):
        """从对话中提取知识并写入图"""
        # Step 1: LLM 提取三元组
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    conversation=conversation_text
                )
            }],
            response_format={"type": "json_object"}
        )
        triples = json.loads(response.choices[0].message.content)

        # Step 2: 创建对话节点
        self.memory.add_entity("Conversation", conversation_id, {
            "content_preview": conversation_text[:200],
            "timestamp": "2026-05-30T10:00:00Z"
        })

        # Step 3: 写入实体和关系
        for entity in triples["entities"]:
            self.memory.add_entity(
                node_type=entity["type"],
                entity_id=entity["id"],
                properties=entity["properties"],
                text=entity["name"]
            )
            # 建立实体与对话的关联
            self.memory.add_relation(
                entity["id"], conversation_id, "MENTIONED_IN"
            )

        for rel in triples["relations"]:
            self.memory.add_relation(
                rel["source"], rel["target"],
                rel["type"], rel.get("properties")
            )

        return triples
```

### 4.2 实际提取示例

输入对话：

> 用户：我们公司的智能客服项目是由张三负责的，他向李经理汇报。这个项目用的是Neo4j做知识图谱存储，上周上线后用户满意度提升了15%。

LLM 提取结果：

```json
{
  "entities": [
    {"id": "zhang_san", "type": "Person", "name": "张三", 
     "properties": {"role": "项目负责人"}},
    {"id": "li_manager", "type": "Person", "name": "李经理",
     "properties": {"role": "技术总监"}},
    {"id": "smart_cs_project", "type": "Project", "name": "智能客服项目",
     "properties": {"status": "已上线"}},
    {"id": "neo4j", "type": "Technology", "name": "Neo4j",
     "properties": {"category": "图数据库"}}
  ],
  "relations": [
    {"source": "zhang_san", "target": "smart_cs_project", 
     "type": "MANAGES", "properties": {}},
    {"source": "zhang_san", "target": "li_manager", 
     "type": "REPORTS_TO", "properties": {}},
    {"source": "smart_cs_project", "target": "neo4j", 
     "type": "USES_TECHNOLOGY", "properties": {}},
    {"source": "smart_cs_project", "target": "user_satisfaction", 
     "type": "HAS_METRIC", "properties": {"value": "+15%"}}
  ]
}
```

---

## 5. 图记忆的检索策略

### 5.1 检索算法对比

| 算法 | 适用场景 | 时间复杂度 | 返回结果 |
|------|---------|-----------|---------|
| BFS（广度优先） | 展开实体的直接关联 | O(V+E) | 按距离分层的邻居 |
| DFS（深度优先） | 探索特定关系链路 | O(V+E) | 完整关系路径 |
| 最短路径 | 两个实体间的最少跳数连接 | O(V+E) | 路径及中间节点 |
| PageRank | 识别图中最重要的实体 | O(V+E) | 带权重的实体排名 |
| 社区检测 | 发现实体聚类 | O(V+E) | 社群划分 |
| 向量相似度 | 语义相近的实体 | O(log V) | 最相似的 N 个节点 |

### 5.2 多跳路径查询

```python
    def find_path(self, source_id, target_id, max_depth=5):
        """查找两个实体间的最短路径"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (a {id: $source})-[*..$depth]-(b {id: $target})
                )
                RETURN path, length(path) AS hops
            """, source=source_id, target=target_id, depth=max_depth)
            record = result.single()
            return record["path"] if record else None

    def expand_context(self, entity_id, relationship_types=None, depth=1):
        """展开实体的上下文网络（类似 GraphRAG 的局部检索）"""
        rel_filter = ""
        if relationship_types:
            types = "|".join(relationship_types)
            rel_filter = f":{types}"

        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (center {{id: $id}})-[r{rel_filter}*1..{depth}]-(neighbor)
                WITH DISTINCT neighbor, 
                     length(shortestPath((center)-[*]-(neighbor))) AS dist
                RETURN neighbor.id AS id, 
                       labels(neighbor) AS labels,
                       properties(neighbor) AS props,
                       dist
                ORDER BY dist
                LIMIT 20
            """, id=entity_id)
            return [dict(record) for record in result]

    def multi_hop_query(self, query_description, start_entity=None):
        """多跳查询：结合 LLM 生成 Cypher"""
        cypher_prompt = f"""根据以下查询需求，生成 Cypher 查询语句。
        图中有以下节点标签：Person, Organization, Topic, Conversation, Project, Technology
        有以下关系类型：WORKS_AT, MANAGES, REPORTS_TO, DISCUSSED, RELATED_TO, 
        USES_TECHNOLOGY, CAUSES, PART_OF, HAS_METRIC

        查询需求：{query_description}

        只返回 Cypher 查询语句，不要其他内容。"""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": cypher_prompt}]
        )
        cypher = response.choices[0].message.content.strip()

        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]
```

### 5.3 检索流程可视化

```
用户查询: "张三管理的项目用了哪些技术？"
                    │
                    ▼
        ┌──────────────────────┐
        │   1. 实体识别/链接    │
        │  "张三" → zhang_san  │
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  2. 图结构化查询      │
        │  MATCH (zhang_san)   │
        │  -[:MANAGES]->(proj) │
        │  -[:USES_TECH]->(t)  │
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  3. 结果组装与排序     │
        │  Neo4j, Python, RAG  │
        └──────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  4. LLM 生成回答      │
        │  "张三管理的智能客服   │
        │   项目使用了 Neo4j"   │
        └──────────────────────┘
```

---

## 6. 向量 + 图的混合记忆架构

纯图记忆在**模糊语义查询**上不如向量检索（"用户之前提到过一个关于性能的问题"），而纯向量记忆在**结构化关系查询**上力不从心。最佳实践是将两者结合。

### 6.1 混合架构设计

```
┌─────────────────────────────────────────────────────────┐
│                   Hybrid Memory Layer                    │
│                                                          │
│  ┌─────────────┐                    ┌─────────────────┐ │
│  │ Vector Store │                    │   Graph Store    │ │
│  │ (ChromaDB)   │                    │   (Neo4j)       │ │
│  │              │                    │                  │ │
│  │ • 全文语义    │                    │ • 实体关系       │ │
│  │ • 情感倾向    │                    │ • 结构化属性     │ │
│  │ • 隐含意图    │                    │ • 因果链路       │ │
│  │ • 模糊记忆    │                    │ • 组织架构       │ │
│  └──────┬───────┘                    └────────┬────────┘ │
│         │                                      │          │
│         └──────────┬───────────────────────────┘          │
│                    │                                      │
│                    ▼                                      │
│         ┌──────────────────┐                             │
│         │   Fusion Engine   │                             │
│         │                  │                             │
│         │ 1. 路由判断查询类型│                             │
│         │ 2. 并行检索        │                             │
│         │ 3. 结果融合与排序  │                             │
│         │ 4. 上下文组装      │                             │
│         └──────────────────┘                             │
└─────────────────────────────────────────────────────────┘
```

### 6.2 混合检索实现

```python
class HybridMemory:
    def __init__(self, graph_memory: AgentGraphMemory, vector_store):
        self.graph = graph_memory
        self.vector = vector_store
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, strategy="auto"):
        """混合检索：自动选择或指定策略"""
        if strategy == "auto":
            strategy = self._classify_query(query)

        if strategy == "graph":
            return self._graph_retrieve(query)
        elif strategy == "vector":
            return self._vector_retrieve(query)
        else:  # hybrid
            graph_results = self._graph_retrieve(query)
            vector_results = self._vector_retrieve(query)
            return self._fuse_results(graph_results, vector_results, query)

    def _classify_query(self, query):
        """用 LLM 判断查询类型"""
        prompt = f"""判断以下查询最适合用图查询、向量查询还是混合查询：
        
查询："{query}"

- graph: 查询涉及实体关系、多跳推理、组织结构
- vector: 查询涉及模糊语义、情感、相似内容
- hybrid: 查询同时涉及关系和语义

只返回一个词：graph, vector, 或 hybrid"""
        
        # 简化实现：基于关键词的规则判断
        graph_keywords = ["谁", "关系", "管理", "属于", "连接", "路径"]
        vector_keywords = ["类似的", "感觉", "大概", "印象"]
        
        if any(kw in query for kw in graph_keywords):
            return "graph"
        elif any(kw in query for kw in vector_keywords):
            return "vector"
        return "hybrid"

    def _fuse_results(self, graph_results, vector_results, query):
        """融合图检索和向量检索的结果"""
        fused = []

        # 图结果：精确关系匹配，置信度高
        for r in graph_results:
            fused.append({
                "content": r.get("content", ""),
                "source": "graph",
                "score": 0.9,  # 图查询结果的基准分
                "metadata": r
            })

        # 向量结果：语义相关，按相似度打分
        for r in vector_results:
            fused.append({
                "content": r.get("content", ""),
                "source": "vector",
                "score": r.get("similarity", 0.5),
                "metadata": r
            })

        # 去重并按综合得分排序
        seen = set()
        unique = []
        for item in sorted(fused, key=lambda x: x["score"], reverse=True):
            content_hash = hash(item["content"][:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(item)

        return unique[:10]  # 返回 top 10
```

### 6.3 图中嵌入向量：Neo4j 5.x 的原生混合方案

Neo4j 5.x 引入了原生向量索引，允许在图节点上直接存储和查询向量：

```python
def hybrid_query_with_neo4j(self, query_text, entity_type=None):
    """利用 Neo4j 原生向量索引做混合查询"""
    query_embedding = self.embedder.encode(query_text).tolist()

    label_filter = f":{entity_type}" if entity_type else ""

    with self.driver.session() as session:
        result = session.run(f"""
            // 向量相似度检索
            CALL db.index.vector.queryNodes(
                'entity_embedding', 10, $embedding
            )
            YIELD node AS similar_node, score AS vector_score
            
            // 在相似节点上展开图关系
            MATCH (similar_node)-[r]-(connected)
            WHERE similar_node:Person OR similar_node:Topic 
               OR similar_node:Organization
            
            RETURN similar_node.id AS node_id,
                   labels(similar_node) AS labels,
                   type(r) AS relationship,
                   connected.id AS connected_id,
                   labels(connected) AS connected_labels,
                   vector_score
            ORDER BY vector_score DESC
            LIMIT 20
        """, embedding=query_embedding)
        return [dict(record) for record in result]
```

这个查询的精妙之处在于：先用向量找到语义最相关的节点，再沿图边展开它的关系网络，**一次查询同时获得语义相似性和结构化关系**。

---

## 7. 多 Agent 系统中的图记忆

### 7.1 共享知识图谱架构

在多 Agent 协作场景中，图记忆可以作为**共享知识层**：

```
┌──────────────────────────────────────────────┐
│           Multi-Agent Shared Memory           │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Agent A   │  │ Agent B   │  │ Agent C   │   │
│  │ (调研)    │  │ (分析)    │  │ (执行)    │   │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘   │
│        │              │              │         │
│        ▼              ▼              ▼         │
│  ┌──────────────────────────────────────────┐ │
│  │        Shared Knowledge Graph            │ │
│  │                                          │ │
│  │   Agent A writes: [发现]→(问题X)        │ │
│  │   Agent B writes: (问题X)-[原因是]→(Y)  │ │
│  │   Agent C reads:  (Y)-[解决方案]→(Z)    │ │
│  │                                          │ │
│  │   所有Agent共享同一张图，各自读写不同区域 │ │
│  └──────────────────────────────────────────┘ │
│                                               │
│  ┌──────────────────────────────────────────┐ │
│  │       Access Control (Agent-level)        │ │
│  │   Agent A: READ + WRITE (Research)       │ │
│  │   Agent B: READ + WRITE (Analysis)       │ │
│  │   Agent C: READ (Execution, needs audit) │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### 7.2 多 Agent 图记忆同步

```python
class MultiAgentGraphMemory:
    def __init__(self, driver, agent_id):
        self.driver = driver
        self.agent_id = agent_id

    def write_with_attribution(self, entity_data, rel_data=None):
        """带来源标注的写入"""
        with self.driver.session() as session:
            # 写入实体，标注来源 Agent
            session.run("""
                MERGE (n {id: $id})
                SET n += $props
                SET n.last_writer = $agent
                SET n.last_updated = datetime()
                // 添加写入历史
                MERGE (n)-[:WRITTEN_BY {timestamp: datetime()}]->(agent:Agent {id: $agent})
            """, id=entity_data["id"], props=entity_data["properties"],
                agent=self.agent_id)

            if rel_data:
                session.run(f"""
                    MATCH (a {{id: $source}})
                    MATCH (b {{id: $target}})
                    MERGE (a)-[r:{rel_data['type']} {{agent: $agent}}]->(b)
                    SET r += $props
                """, source=rel_data["source"], target=rel_data["target"],
                    agent=self.agent_id, props=rel_data.get("properties", {}))

    def get_subgraph_view(self, agent_role):
        """获取当前 Agent 的视图子图"""
        # 不同角色的 Agent 看到不同的图子集
        role_filters = {
            "researcher": ["DISCUSSED", "MENTIONED_IN", "RELATED_TO"],
            "analyst": ["CAUSES", "RELATED_TO", "HAS_METRIC"],
            "executor": ["MANAGES", "PART_OF", "USES_TECHNOLOGY"],
        }
        allowed_rels = role_filters.get(agent_role, [])

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]-(m)
                WHERE type(r) IN $allowed_rels
                RETURN n, r, m
                LIMIT 200
            """, allowed_rels=allowed_rels)
            return [dict(record) for record in result]

    def detect_conflicts(self):
        """检测不同 Agent 写入的冲突"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]-(m)
                WITH n, m, collect(DISTINCT r.agent) AS agents
                WHERE size(agents) > 1
                RETURN n.id AS entity1, m.id AS entity2, 
                       agents AS conflicting_agents
            """)
            return [dict(record) for record in result]
```

---

## 8. 实战案例：基于图记忆的智能客服 Agent

### 8.1 场景描述

一个 B2B SaaS 公司的智能客服 Agent，需要：
- 记住每个客户的历史交互、偏好和投诉
- 跨会话追踪问题的解决状态
- 理解客户组织结构，识别决策链路
- 从历史案例中找到相似问题的解决方案

### 8.2 客服图记忆模型

```
┌─────────────────────────────────────────────────────────┐
│                Customer Support Knowledge Graph          │
│                                                          │
│  ┌──────────┐  customer_of  ┌──────────────┐            │
│  │ Customer  │─────────────▶│  Company      │            │
│  │ -name     │              │ -plan: Pro    │            │
│  │ -role     │              │ -industry: 医疗│            │
│  │ -priority │              │ -revenue: 高  │            │
│  └─────┬────┘              └──────────────┘            │
│        │                                                  │
│   submitted                                            │
│        │                                                  │
│        ▼                                                  │
│  ┌──────────┐  affects      ┌──────────────┐            │
│  │ Ticket    │─────────────▶│  Module       │            │
│  │ -status   │              │ -name         │            │
│  │ -severity │              │ -team         │            │
│  │ -created  │              │ -on_call      │            │
│  └─────┬────┘              └──────────────┘            │
│        │                                                  │
│   resolved_by                                           │
│        │                                                  │
│        ▼                                                  │
│  ┌──────────┐  similar_to   ┌──────────────┐            │
│  │ Solution  │◀────────────│  Solution     │            │
│  │ -content  │              │ -content      │            │
│  │ -success  │              │ -success_rate │            │
│  └──────────┘              └──────────────┘            │
│                                                          │
│  ┌──────────┐  escalated_to ┌──────────────┐            │
│  │ Agent     │─────────────▶│  Team         │            │
│  │ -name     │              │ -expertise    │            │
│  │ -rating   │              │ -availability │            │
│  └──────────┘              └──────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 8.3 完整实现

```python
class CustomerSupportAgent:
    def __init__(self, graph_memory, llm):
        self.memory = graph_memory
        self.llm = llm

    def handle_ticket(self, customer_id, ticket_content):
        """处理客户工单的完整流程"""
        
        # 1. 查询客户的历史信息
        customer_context = self._get_customer_context(customer_id)
        
        # 2. 从当前对话提取新知识
        self._extract_and_store_knowledge(ticket_content, customer_id)
        
        # 3. 查找相似历史案例
        similar_cases = self._find_similar_resolved_cases(ticket_content)
        
        # 4. 检查是否需要升级（复杂度/优先级判断）
        escalation_info = self._check_escalation_needed(
            customer_id, ticket_content
        )
        
        # 5. 组装上下文生成回复
        context = {
            "customer_info": customer_context,
            "similar_cases": similar_cases,
            "escalation": escalation_info,
            "current_ticket": ticket_content
        }
        
        return self._generate_response(context)

    def _get_customer_context(self, customer_id):
        """从图中获取客户的完整上下文"""
        with self.memory.driver.session() as session:
            result = session.run("""
                // 客户基本信息
                MATCH (c:Customer {id: $cid})
                
                // 公司信息
                OPTIONAL MATCH (c)-[:customer_of]->(company:Company)
                
                // 最近的工单
                OPTIONAL MATCH (c)-[:submitted]->(t:Ticket)
                WHERE t.created > datetime() - duration('P30D')
                
                // 未解决的问题
                OPTIONAL MATCH (c)-[:submitted]->(open:Ticket)
                WHERE open.status IN ['open', 'pending']
                
                // 客户偏好
                OPTIONAL MATCH (c)-[:has_preference]->(pref:Preference)
                
                RETURN c.name AS name,
                       c.priority AS priority,
                       company.name AS company,
                       company.plan AS plan,
                       collect(DISTINCT {
                           ticket_id: t.id, 
                           status: t.status,
                           severity: t.severity
                       }) AS recent_tickets,
                       count(DISTINCT open) AS open_tickets,
                       collect(DISTINCT pref.content) AS preferences
            """, cid=customer_id)
            return dict(result.single()) if result.peek() else {}

    def _find_similar_resolved_cases(self, ticket_content):
        """结合图和向量查找相似案例"""
        # 先用向量找语义相似的历史工单
        embedding = self.memory.embedder.encode(ticket_content).tolist()
        
        with self.memory.driver.session() as session:
            result = session.run("""
                // 向量相似度匹配
                CALL db.index.vector.queryNodes(
                    'ticket_embedding', 5, $embedding
                )
                YIELD node AS ticket, score
                
                // 只看已解决的工单
                WHERE ticket.status = 'resolved'
                
                // 展开解决方案
                MATCH (ticket)-[:resolved_by]->(solution:Solution)
                
                RETURN ticket.id AS ticket_id,
                       ticket.content AS content,
                       score AS similarity,
                       solution.content AS solution_content,
                       solution.success_rate AS success_rate
                ORDER BY score * solution.success_rate DESC
                LIMIT 3
            """, embedding=embedding)
            return [dict(record) for record in result]

    def _check_escalation_needed(self, customer_id, ticket_content):
        """基于图结构判断是否需要升级"""
        with self.memory.driver.session() as session:
            result = session.run("""
                MATCH (c:Customer {id: $cid})
                
                // 检查重复问题次数
                OPTIONAL MATCH (c)-[:submitted]->(t:Ticket)
                WHERE t.status IN ['open', 'pending', 'reopened']
                WITH c, count(t) AS open_count
                
                // 检查客户优先级
                WITH c, open_count,
                     CASE c.priority
                         WHEN 'critical' THEN 3
                         WHEN 'high' THEN 2
                         WHEN 'normal' THEN 1
                         ELSE 0
                     END AS priority_score
                
                // 检查是否有 SLA 风险
                OPTIONAL MATCH (c)-[:submitted]->(old:Ticket)
                WHERE old.status = 'open' 
                  AND old.created < datetime() - duration('P2D')
                WITH c, open_count, priority_score,
                     count(old) AS overdue_count
                
                RETURN open_count, priority_score, overdue_count,
                       CASE 
                           WHEN overdue_count > 0 THEN true
                           WHEN open_count >= 3 THEN true
                           WHEN priority_score >= 3 THEN true
                           ELSE false
                       END AS needs_escalation
            """, cid=customer_id)
            record = result.single()
            return dict(record) if record else {"needs_escalation": False}

    def _extract_and_store_knowledge(self, conversation, customer_id):
        """从对话中提取知识并写入图"""
        extractor = ConversationToGraph(self.memory, self.llm)
        triples = extractor.extract_and_store(
            f"ticket_{customer_id}_{int(time.time())}",
            conversation
        )
        # 额外建立与客户的关联
        for entity in triples.get("entities", []):
            self.memory.add_relation(
                customer_id, entity["id"], "SUBMITTED_ABOUT"
            )
```

### 8.4 查询效果对比

**查询**："王医生的医院之前遇到过类似的数据导出问题吗？"

**纯向量方案**：
```
检索结果：返回包含"数据导出"关键词的 5 条历史文本片段
问题：无法关联到王医生所在的医院，可能返回不相关公司的案例
```

**图记忆方案**：
```cypher
MATCH (c:Customer {name: "王医生"})-[:customer_of]->(comp:Company)
MATCH (comp)<-[:customer_of]-(other_c:Customer)
MATCH (other_c)-[:submitted]->(t:Ticket)
WHERE t.content CONTAINS "数据导出" AND t.status = "resolved"
MATCH (t)-[:resolved_by]->(s:Solution)
RETURN other_c.name AS similar_customer,
       t.content AS similar_ticket,
       s.content AS proven_solution
ORDER BY s.success_rate DESC
```
```
结果：精准返回同行业公司的已解决数据导出问题及解决方案
推理路径：王医生 → 其所在医院 → 同行业其他客户 → 数据导出工单 → 解决方案
```

---

## 9. 性能优化与最佳实践

### 9.1 图记忆的工程考量

| 维度 | 建议 |
|------|------|
| **节点数量** | 单个 Agent 知识图控制在 10 万节点以内，超过需分区 |
| **索引策略** | 所有高频查询的属性创建索引，向量字段创建向量索引 |
| **过期策略** | 低优先级关系设置 TTL，定期清理过期的弱连接 |
| **写入批量** | 对话结束后批量写入，避免频繁单条写入 |
| **缓存层** | 热点实体的邻居缓存在 Redis，减少图数据库查询 |
| **安全隔离** | 多 Agent 共享图时，按 Agent 角色做子图权限控制 |

### 9.2 常见陷阱

1. **过度提取**：不是每个对话都需要提取三元组，设定信息密度阈值
2. **关系爆炸**：一个实体的关系超过 100 条时查询变慢，需要关系裁剪
3. **循环引用**：A 关联 B、B 关联 C、C 又关联 A，查询时需设最大深度
4. **Embedding 漂移**：实体描述更新后向量也要同步更新，否则向量检索失效

---

## 总结

图记忆不是要替代向量记忆，而是为 Agent 提供了一层**结构化的关系推理能力**。在实际系统中，最佳实践是：

1. **向量负责语义理解**：模糊查询、情感分析、相似内容匹配
2. **图负责关系推理**：多跳查询、因果链追溯、组织结构导航
3. **融合层负责编排**：自动判断查询类型，协调两种检索策略

知识图谱让 Agent 从"记住了什么文本"进化到"理解了什么关系"，这是构建真正有记忆、有推理能力的 Agent 的关键一步。

---

**参考架构**：本文代码基于 Neo4j 5.x + sentence-transformers 构建，完整示例可参考 [Graph Memory SDK](https://github.com/nicepkg/graph-memory) 和 [LangChain Neo4j Graph](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)。
