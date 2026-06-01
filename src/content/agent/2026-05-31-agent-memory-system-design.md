---
title: "Agent记忆系统设计：从短期上下文到长期知识的完整架构"
description: "系统解析Agent记忆系统的设计原理，覆盖工作记忆、短期记忆、长期记忆、情景记忆的分层架构与实现"
date: 2026-05-31
author: "RiceBall-15"
category: "agent"
subCategory: agent-memory
tags: ["Agent记忆", "记忆系统", "向量数据库", "RAG"]
draft: false
---

# Agent记忆系统设计：从短期上下文到长期知识的完整架构

## 核心问题：为什么Agent需要记忆？

没有记忆的Agent就像一个"金鱼"——每次对话都从零开始。用户体验极差：
- 用户："我之前说过我喜欢Python"
- Agent："请问您使用什么编程语言？"
- 用户：（再次说明）→ Agent：（下次又忘了）

更严重的是：**Agent无法从经验中学习**。它不会记住哪些方法有效、哪些工具好用、哪些错误要避免。

记忆系统的目标是让Agent具备：
1. **上下文连续性**：记住当前对话的内容
2. **个性化**：记住用户的偏好和历史
3. **知识积累**：从经验中学习和改进
4. **决策参考**：基于历史做出更好的决策

---

## 一、记忆分层架构

### 1.1 四层记忆模型

```
┌─────────────────────────────────────────────┐
│              工作记忆 (Working Memory)        │
│         LLM上下文窗口，当前任务状态            │
│              容量：4K-128K tokens             │
├─────────────────────────────────────────────┤
│              短期记忆 (Short-term Memory)     │
│         当前会话的对话历史                     │
│              存储：Redis/内存                 │
├─────────────────────────────────────────────┤
│              长期记忆 (Long-term Memory)      │
│         用户偏好/知识/经验                    │
│              存储：向量数据库                  │
├─────────────────────────────────────────────┤
│              情景记忆 (Episodic Memory)       │
│         具体事件和经历的记录                  │
│              存储：关系数据库                  │
└─────────────────────────────────────────────┘
```

### 1.2 记忆层级对比

| 记忆类型 | 存储位置 | 生命周期 | 容量 | 读写速度 | 用途 |
|---------|---------|---------|------|---------|------|
| **工作记忆** | LLM上下文 | 当前轮次 | 极小 | 极快 | 当前推理 |
| **短期记忆** | Redis | 会话级 | 中等 | 快 | 对话历史 |
| **长期记忆** | 向量数据库 | 持久化 | 大 | 中 | 用户知识 |
| **情景记忆** | PostgreSQL | 持久化 | 大 | 中 | 事件记录 |

---

## 二、工作记忆：当前任务状态

### 2.1 上下文管理

```python
class WorkingMemory:
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.messages = []
        self.system_prompt = ""
    
    def add_message(self, role: str, content: str):
        """添加消息，自动管理上下文长度"""
        self.messages.append({"role": role, "content": content})
        
        # 检查token数量，超出时压缩
        if self.count_tokens() > self.max_tokens:
            self.compress()
    
    def compress(self):
        """压缩历史消息，保留关键信息"""
        # 1. 保留system prompt
        # 2. 保留最近N轮对话
        # 3. 中间消息生成摘要
        recent = self.messages[-6:]  # 最近3轮
        older = self.messages[:-6]
        
        if older:
            summary = self.generate_summary(older)
            self.messages = [
                {"role": "system", "content": f"历史摘要：{summary}"}
            ] + recent
    
    def count_tokens(self) -> int:
        """估算当前token数量"""
        return sum(len(m["content"]) // 2 for m in self.messages)
```

### 2.2 上下文压缩策略

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **滑动窗口** | 保留最近N条消息 | 简单 | 可能丢失重要信息 |
| **摘要压缩** | 用LLM生成历史摘要 | 保留关键信息 | 增加延迟 |
| **重要性筛选** | 只保留高重要性消息 | 精准 | 需要评估重要性 |
| **混合策略** | 结合多种策略 | 平衡效果 | 实现复杂 |

---

## 三、短期记忆：会话级存储

### 3.1 对话历史管理

```python
class ShortTermMemory:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1小时过期
    
    async def save_turn(self, session_id: str, turn: dict):
        """保存一轮对话"""
        key = f"session:{session_id}:history"
        await self.redis.rpush(key, json.dumps(turn))
        await self.redis.expire(key, self.ttl)
    
    async def get_history(self, session_id: str, limit: int = 10) -> list:
        """获取对话历史"""
        key = f"session:{session_id}:history"
        history = await self.redis.lrange(key, -limit, -1)
        return [json.loads(h) for h in history]
    
    async def get_summary(self, session_id: str) -> str:
        """获取对话摘要"""
        history = await self.get_history(session_id, limit=50)
        return await self.generate_summary(history)
```

### 3.2 会话状态管理

```python
class SessionState:
    """管理会话级状态"""
    def __init__(self):
        self.current_task = None
        self.tool_results = {}
        self.user_preferences = {}
        self.conversation_context = []
    
    def update_task(self, task: dict):
        """更新当前任务"""
        self.current_task = task
    
    def record_tool_result(self, tool_name: str, result: any):
        """记录工具调用结果"""
        self.tool_results[tool_name] = result
    
    def get_context_for_llm(self) -> list:
        """生成LLM可用的上下文"""
        context = []
        if self.current_task:
            context.append(f"当前任务：{self.current_task['description']}")
        if self.tool_results:
            context.append(f"已获取信息：{json.dumps(self.tool_results, ensure_ascii=False)}")
        return context
```

---

## 四、长期记忆：持久化存储

### 4.1 向量数据库设计

```python
class LongTermMemory:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store  # Chroma/Milvus/Pinecone
        self.embedder = embedder
    
    async def save_memory(self, user_id: str, memory: MemoryEntry):
        """保存记忆到向量数据库"""
        # 1. 生成embedding
        embedding = await self.embedder.embed(memory.content)
        
        # 2. 存储
        await self.vector_store.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[{
                "user_id": user_id,
                "type": memory.type,
                "importance": memory.importance,
                "created_at": memory.created_at
            }]
        )
    
    async def recall(self, user_id: str, query: str, top_k: int = 5) -> list:
        """检索相关记忆"""
        # 1. 生成查询embedding
        query_embedding = await self.embedder.embed(query)
        
        # 2. 向量搜索
        results = await self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id}
        )
        
        return results
    
    async def consolidate(self, user_id: str):
        """记忆整合：合并相似记忆，删除过期记忆"""
        # 获取用户所有记忆
        memories = await self.get_all_memories(user_id)
        
        # 1. 去重：合并相似度>0.9的记忆
        clusters = self.cluster_similar(memories, threshold=0.9)
        for cluster in clusters:
            merged = self.merge_memories(cluster)
            await self.save_memory(user_id, merged)
        
        # 2. 遗忘：删除importance<0.3且超过30天的记忆
        await self.delete_old_memories(user_id, days=30, min_importance=0.3)
```

### 4.2 记忆类型设计

| 记忆类型 | 内容 | 更新频率 | 示例 |
|---------|------|---------|------|
| **事实记忆** | 用户的客观信息 | 低 | "用户是Python开发者" |
| **偏好记忆** | 用户的喜好 | 中 | "用户喜欢简洁的回答" |
| **经验记忆** | 任务执行的经验 | 高 | "用pandas处理CSV最快" |
| **关系记忆** | 人际关系信息 | 低 | "用户的同事叫张三" |

### 4.3 记忆重要性评估

```python
class ImportanceScorer:
    """评估记忆的重要性"""
    
    def score(self, memory: MemoryEntry) -> float:
        score = 0.0
        
        # 1. 信息类型
        if memory.type == "fact":
            score += 0.3
        elif memory.type == "preference":
            score += 0.4
        elif memory.type == "experience":
            score += 0.5
        
        # 2. 提及频率
        if memory.mention_count > 3:
            score += 0.2
        
        # 3. 用户明确强调
        if memory.explicitly_stated:
            score += 0.3
        
        # 4. 时效性
        age_days = (datetime.now() - memory.created_at).days
        if age_days < 7:
            score += 0.2
        elif age_days > 90:
            score -= 0.2
        
        return min(max(score, 0.0), 1.0)
```

---

## 五、情景记忆：事件记录

### 5.1 数据库设计

```sql
CREATE TABLE episodic_memories (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- task, conversation, tool_call
    event_data JSONB NOT NULL,
    outcome VARCHAR(50),  -- success, failure, partial
    importance FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

-- 索引
CREATE INDEX idx_user_events ON episodic_memories(user_id, created_at);
CREATE INDEX idx_event_type ON episodic_memories(event_type);
```

### 5.2 情景记忆查询

```python
class EpisodicMemory:
    def __init__(self, db):
        self.db = db
    
    async def record_event(self, user_id: str, event: dict):
        """记录事件"""
        await self.db.execute("""
            INSERT INTO episodic_memories (id, user_id, session_id, event_type, event_data)
            VALUES ($1, $2, $3, $4, $5)
        """, uuid4(), user_id, event["session_id"], event["type"], json.dumps(event))
    
    async def get_similar_experiences(self, user_id: str, context: str, limit: int = 5):
        """查找相似经历"""
        # 1. 基于event_data的文本搜索
        results = await self.db.fetch("""
            SELECT * FROM episodic_memories
            WHERE user_id = $1
            AND event_data::text ILIKE $2
            ORDER BY importance DESC, created_at DESC
            LIMIT $3
        """, user_id, f"%{context}%", limit)
        
        return results
    
    async def get_successful_patterns(self, user_id: str, task_type: str):
        """获取成功的经验模式"""
        return await self.db.fetch("""
            SELECT event_data, COUNT(*) as frequency
            FROM episodic_memories
            WHERE user_id = $1
            AND event_type = $2
            AND outcome = 'success'
            GROUP BY event_data
            ORDER BY frequency DESC
            LIMIT 10
        """, user_id, task_type)
```

---

## 六、记忆系统集成

### 6.1 完整架构

```
┌──────────────────────────────────────────────────────┐
│                    Agent Core                         │
│                                                       │
│  ┌───────────────────────────────────────────────┐  │
│  │              Memory Manager                    │  │
│  │                                                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │  │
│  │  │ 工作记忆  │ │ 短期记忆  │ │ 长期记忆 │     │  │
│  │  │          │ │          │ │          │     │  │
│  │  │ 上下文   │ │ 对话历史 │ │ 用户知识 │     │  │
│  │  │ 管理     │ │          │ │          │     │  │
│  │  └──────────┘ └──────────┘ └──────────┘     │  │
│  │                                                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │  │
│  │  │ 情景记忆  │ │ 记忆检索  │ │ 记忆压缩 │     │  │
│  │  │          │ │          │ │          │     │  │
│  │  │ 事件记录 │ │ 相关性   │ │ 去重合并 │     │  │
│  │  │          │ │ 排序     │ │          │     │  │
│  │  └──────────┘ └──────────┘ └──────────┘     │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 6.2 记忆检索流程

```
用户输入 → 提取关键词 → 多路检索 → 相关性排序 → 上下文注入 → LLM推理
              │            │            │            │
           实体识别     语义搜索     重要性权重   摘要压缩
                       时间衰减     去重
```

### 6.3 记忆写入流程

```
对话结束 → 提取记忆点 → 评估重要性 → 分类存储 → 去重检查 → 写入数据库
              │            │            │            │
           NER/意图      打分模型    事实/偏好/经验  相似度检查
```

---

## 七、最佳实践与常见问题

### 7.1 记忆管理策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **主动遗忘** | 定期清理过期/低重要性记忆 | 长期运行的Agent |
| **记忆巩固** | 定期整合相似记忆 | 知识积累型Agent |
| **隐私保护** | 敏感信息加密存储 | 涉及用户隐私 |
| **记忆共享** | 多Agent共享公共记忆 | 多Agent协作 |

### 7.2 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **记忆检索慢** | 向量库查询慢 | 优化索引+缓存热点 |
| **记忆不准** | 语义理解偏差 | 增加元数据过滤 |
| **记忆膨胀** | 只增不减 | 定期清理+压缩 |
| **隐私泄露** | 敏感信息存储 | 加密+访问控制 |

### 7.3 性能优化

| 优化方向 | 具体措施 |
|---------|---------|
| **检索速度** | 向量索引优化+缓存 |
| **存储效率** | 压缩+增量更新 |
| **写入性能** | 批量写入+异步 |
| **查询准确** | 混合检索+重排序 |

---

## 总结

Agent记忆系统的设计要点：

1. **分层设计**：工作/短期/长期/情景记忆各司其职
2. **重要性评估**：不是所有信息都值得记忆
3. **定期整合**：合并相似记忆，清理过期记忆
4. **隐私保护**：敏感信息加密存储
5. **性能优化**：检索速度和存储效率并重

> 记忆系统的本质是**让Agent从"无状态工具"变成"有经验的助手"**。
