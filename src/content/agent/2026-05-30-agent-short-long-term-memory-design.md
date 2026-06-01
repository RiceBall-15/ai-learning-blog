---
title: 'Agent短期记忆与长期记忆：架构设计全解析'
description: '深入解析AI Agent的多层记忆架构，从工作记忆到语义记忆的设计模式与工程实践'
date: 2026-05-30
author: 'RiceBall-15'
category: 'agent'
subCategory: agent-memory
tags: ['短期记忆', '长期记忆', '工作记忆', '情景记忆', '记忆架构']
draft: false
---

# Agent短期记忆与长期记忆：架构设计全解析

## 引言：为什么Agent需要分层记忆？

人类的记忆并非单一系统，而是由多个子系统协同运作：工作记忆负责当前思考，情景记忆记录经历，语义记忆存储知识，程序记忆保存技能。现代AI Agent正在借鉴这一认知科学框架，构建类似的多层记忆架构。

本文将系统性地解析Agent记忆系统的八大核心设计维度，结合代码示例与架构图，为开发者提供一份完整的记忆架构工程指南。

---

## 一、工作记忆（上下文窗口）：限制与设计

工作记忆是Agent在单次推理过程中能够"看到"的信息范围，对应LLM的上下文窗口（Context Window）。

### 核心挑战

```
┌──────────────────────────────────────────────────────┐
│                    上下文窗口 (Context Window)        │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │ System   │ │ Tool     │ │ History  │ │ User   │  │
│  │ Prompt   │ │ Results  │ │ Messages │ │ Input  │  │
│  │ ~2K tok  │ │ ~8K tok  │ │ ~20K tok │ │ ~2K tok│  │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────────┐│
│  │ 剩余容量: ~38K tokens（128K窗口）                ││
│  │ 可用于记忆检索注入                                ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

**关键限制：**
- **硬性天花板**：模型支持的最大token数（4K / 32K / 128K / 200K）
- **推理成本**：注意力计算复杂度为O(n²)，长窗口意味着高延迟
- **信息稀释**：随着上下文增长，模型对早期信息的注意力衰减（Lost in the Middle问题）

### 设计模式：滑动窗口 + 摘要压缩

```python
class WorkingMemory:
    """工作记忆管理器：维护上下文窗口内的信息"""
    
    def __init__(self, max_tokens: int = 128000, reserve_ratio: float = 0.3):
        self.max_tokens = max_tokens
        self.reserve_for_memory = int(max_tokens * reserve_ratio)
        self.system_prompt_tokens = 0
        self.recent_messages = []
        self.compressed_summary = ""
    
    def build_context(self, messages: list, memory_retrievals: list) -> list:
        """构建发送给LLM的上下文"""
        context = []
        
        # 1. System Prompt（固定）
        context.append(self._format_system_prompt())
        
        # 2. 历史摘要（压缩后的早期对话）
        if self.compressed_summary:
            context.append({
                "role": "system",
                "content": f"[历史摘要]\n{self.compressed_summary}"
            })
        
        # 3. 检索到的记忆片段
        if memory_retrievals:
            context.append({
                "role": "system",
                "content": self._format_memory_context(memory_retrievals)
            })
        
        # 4. 最近的N条消息（保留完整细节）
        remaining_tokens = self.max_tokens - self._count_tokens(context)
        context.extend(self._select_recent(messages, remaining_tokens))
        
        return context
    
    def _format_memory_context(self, memories: list) -> str:
        """格式化检索到的记忆为上下文注入"""
        lines = ["[相关记忆]"]
        for i, mem in enumerate(memories, 1):
            lines.append(f"{i}. [{mem['type']}] {mem['content']} (来源: {mem['source']})")
        return "\n".join(lines)
    
    def compress_and_evict(self, messages: list) -> str:
        """当上下文超限时，压缩旧消息为摘要"""
        old_messages = messages[:-self.keep_recent]  # 保留最近N条
        return self.llm.summarize(old_messages)
```

### 关键设计决策

| 策略 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| 完整保留 | 短对话（<20轮） | 零信息损失 | 上下文溢出风险 |
| 滑动窗口 | 通用场景 | 简单可控 | 早期信息完全丢失 |
| 摘要压缩 | 长对话 | 兼顾容量与信息量 | 摘要质量依赖模型 |
| 检索注入 | 超长交互 | 按需获取相关信息 | 检索延迟开销 |

---

## 二、情景记忆（Episodic Memory）：对话历史管理

情景记忆记录的是"发生了什么"——具体的对话事件、用户行为序列和交互经历。

### 架构设计

```
┌───────────────────────────────────────────────────────────┐
│                   情景记忆系统 (Episodic Memory)           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ 会话分段器   │───▶│ 事件提取器   │───▶│ 时序存储引擎  │  │
│  │ Segmenter   │    │ Extractor   │    │ Temporal Store│  │
│  └─────────────┘    └─────────────┘    └──────────────┘  │
│        │                  │                    │          │
│        ▼                  ▼                    ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ 会话边界检测 │    │ 意图/行动/   │    │ 时间戳索引    │  │
│  │ 话题切换识别 │    │ 结果三元组   │    │ 用户ID索引   │  │
│  └─────────────┘    └─────────────┘    └──────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 事件数据模型

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class EpisodicEvent:
    """情景记忆的基本单元"""
    event_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    
    # 事件三元组
    intent: str          # 用户意图：如"查询天气"
    action: str          # Agent执行的动作：如"调用weather_api"
    result: str          # 执行结果：如"北京26°C晴"
    
    # 元数据
    importance_score: float = 0.5   # 重要性评分 [0,1]
    emotion_tag: Optional[str] = None  # 情感标签
    topics: list = field(default_factory=list)
    
    # 向量化存储
    embedding: Optional[list] = None

class EpisodicMemoryStore:
    """情景记忆存储与检索"""
    
    def __init__(self, vector_db, metadata_db):
        self.vector_db = vector_db      # 向量数据库（语义检索）
        self.metadata_db = metadata_db  # 结构化数据库（精确查询）
    
    async def record_event(self, event: EpisodicEvent):
        """记录新的情景事件"""
        # 生成事件的文本表示用于embedding
        text = f"[{event.intent}] 用户执行了 {event.action}，结果: {event.result}"
        event.embedding = await self.embed(text)
        
        # 双写：向量库 + 元数据
        await self.vector_db.upsert(event.event_id, event.embedding, {
            "user_id": event.user_id,
            "session_id": event.session_id,
            "intent": event.intent,
            "importance": event.importance_score,
            "timestamp": event.timestamp.isoformat()
        })
        
        await self.metadata_db.insert(event)
    
    async def recall_similar(self, query: str, user_id: str, top_k: int = 5) -> list:
        """基于语义相似性回忆类似经历"""
        query_embedding = await self.embed(query)
        results = await self.vector_db.query(
            query_embedding, 
            filter={"user_id": user_id},
            top_k=top_k
        )
        return [self._hydrate_event(r) for r in results]
    
    async def recall_temporal(self, user_id: str, 
                              since: datetime, until: datetime) -> list:
        """基于时间范围回忆经历"""
        return await self.metadata_db.query(
            user_id=user_id,
            time_range=(since, until),
            order_by="timestamp DESC"
        )
```

### 情景记忆的遗忘机制

并非所有对话都同等重要。情景记忆需要模拟人类的"艾宾浩斯遗忘曲线"：

```python
import math

def importance_decay(event: EpisodicEvent, now: datetime) -> float:
    """结合重要性与时间衰减计算记忆权重"""
    hours_elapsed = (now - event.timestamp).total_seconds() / 3600
    
    # 基础衰减：指数衰减，半衰期72小时
    time_decay = math.exp(-0.693 * hours_elapsed / 72)
    
    # 重要性增强：高重要性事件衰减更慢
    importance_boost = 1.0 + event.importance_score * 2.0
    
    # 重复激活效应：被多次回忆的事件衰减更慢
    recall_boost = 1.0 + math.log(1 + event.recall_count)
    
    return time_decay * importance_boost * recall_boost
```

---

## 三、语义记忆（Semantic Memory）：知识提取与存储

语义记忆存储的是"是什么"——从对话中提取的事实、用户偏好、领域知识等结构化信息。

### 知识提取流水线

```
对话文本 ──▶ 信息提取 ──▶ 实体消歧 ──▶ 知识融合 ──▶ 语义记忆库
                │              │              │
           LLM结构化抽取   实体链接/      与已有知识
           (NER+关系)     指代消解       去重/合并
```

```python
class SemanticMemoryExtractor:
    """从对话中提取语义记忆"""
    
    EXTRACTION_PROMPT = """从以下对话中提取结构化的事实信息。
    
    对话内容:
    {conversation}
    
    请提取以下类型的语义记忆，以JSON格式返回:
    - user_preferences: 用户偏好 (如: 喜欢/不喜欢什么)
    - factual_knowledge: 事实知识 (如: 用户的职业、住址)
    - domain_knowledge: 领域知识 (如: 专业术语、行业规则)
    - relationships: 人际关系 (如: 用户提到的家人、同事)
    
    每条记忆包含:
    - subject: 主体
    - predicate: 谓语/关系
    - object: 客体
    - confidence: 置信度 [0,1]
    - source_conversation_id: 来源对话ID
    """
    
    async def extract(self, conversation: str, session_id: str) -> list:
        response = await self.llm.generate(
            self.EXTRACTION_PROMPT.format(conversation=conversation)
        )
        memories = self._parse_json(response)
        
        # 去重与融合
        for mem in memories:
            existing = await self.semantic_store.find_similar(mem)
            if existing and existing.confidence > mem.confidence:
                # 已有更强证据，跳过
                continue
            elif existing:
                # 合并为更强的证据
                mem = self._merge_knowledge(existing, mem)
            
            await self.semantic_store.upsert(mem)
        
        return memories
```

### 知识图谱存储

```python
class SemanticKnowledgeGraph:
    """基于图结构的语义记忆存储"""
    
    async def store_entity(self, entity: dict):
        """存储实体节点"""
        await self.graph.upsert_node(
            label=entity["type"],      # e.g., "User", "Preference", "Fact"
            properties={
                "name": entity["subject"],
                "value": entity["object"],
                "confidence": entity["confidence"],
                "last_verified": datetime.now(),
                "source": entity["source"]
            }
        )
    
    async def store_relation(self, subject: str, relation: str, obj: str):
        """存储关系边"""
        await self.graph.upsert_edge(
            from_node=subject,
            to_node=obj,
            edge_type=relation,
            properties={"weight": 1.0, "created_at": datetime.now()}
        )
    
    async def query_subgraph(self, entity: str, depth: int = 2) -> dict:
        """查询实体周围的子图，获取相关知识"""
        subgraph = await self.graph.bfs(entity, max_depth=depth)
        return self._format_for_context(subgraph)
```

---

## 四、程序记忆（Procedural Memory）：技能学习

程序记忆存储的是"怎么做"——Agent学到的操作流程、工具使用模式、成功策略。

### 技能模式库

```python
@dataclass
class ProceduralSkill:
    """程序记忆：可复用的操作技能"""
    skill_id: str
    name: str
    description: str
    
    # 技能模板
    trigger_pattern: str      # 触发条件的自然语言描述
    action_sequence: list     # 操作步骤序列
    expected_outcome: str     # 预期结果
    
    # 学习统计
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

class ProceduralMemoryManager:
    """管理Agent的程序记忆（技能库）"""
    
    async def learn_from_demonstration(self, task: str, steps: list):
        """从成功执行中学习技能"""
        skill = ProceduralSkill(
            skill_id=generate_id(),
            name=task,
            description=self._summarize_steps(steps),
            trigger_pattern=task,
            action_sequence=steps,
            expected_outcome=steps[-1].get("result", ""),
            success_count=1
        )
        await self.skill_store.upsert(skill)
    
    async def refine_skill(self, skill_id: str, feedback: str, success: bool):
        """根据执行反馈优化技能"""
        skill = await self.skill_store.get(skill_id)
        if success:
            skill.success_count += 1
        else:
            skill.failure_count += 1
            # 失败时让LLM分析原因并修正步骤
            revised = await self._llm_analyze_and_fix(
                skill.action_sequence, feedback
            )
            skill.action_sequence = revised
        
        skill.last_used = datetime.now()
        await self.skill_store.upsert(skill)
    
    async def retrieve_skill(self, task_description: str) -> Optional[ProceduralSkill]:
        """根据任务描述检索最匹配的技能"""
        # 基于语义相似度匹配
        candidates = await self.skill_store.semantic_search(task_description, top_k=3)
        
        if not candidates:
            return None
        
        # 结合相似度和成功率排序
        ranked = sorted(candidates, 
                       key=lambda s: self._composite_score(s), 
                       reverse=True)
        
        # 置信度门槛
        best = ranked[0]
        if best.success_rate < 0.5 and best.success_count + best.failure_count >= 5:
            return None  # 技能不可靠，不使用
        
        return best
```

---

## 五、记忆固化策略（Memory Consolidation）

记忆固化是将短期信息转化为长期记忆的过程，类似于人类睡眠时的记忆整理。

### 固化流水线架构

```
┌─────────────────────────────────────────────────────────────┐
│                   记忆固化流水线 (Consolidation Pipeline)    │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 触发条件  │─▶│ 内容筛选  │─▶│ 信息压缩  │─▶│ 持久存储  │   │
│  │ 检测     │  │ 与过滤   │  │ 与整合   │  │ 与索引   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       │             │              │              │         │
│  ·会话结束     ·重要性>阈值   ·多条合并为   ·写入向量库   │
│  ·定时触发     ·非重复信息    一条精炼知识  ·更新知识图谱  │
│  ·缓冲区满     ·有新价值     ·生成摘要     ·更新技能库    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class MemoryConsolidationEngine:
    """记忆固化引擎：将短期记忆转化为长期记忆"""
    
    def __init__(self, episodic_store, semantic_store, procedural_store):
        self.episodic = episodic_store
        self.semantic = semantic_store
        self.procedural = procedural_store
    
    async def consolidate_session(self, session_id: str):
        """固化一次会话的记忆"""
        events = await self.episodic.get_session_events(session_id)
        
        # 1. 提取语义记忆
        knowledge_items = await self._extract_knowledge(events)
        for item in knowledge_items:
            await self.semantic.upsert(item)
        
        # 2. 检测是否有新的操作模式可学习
        action_patterns = self._find_action_patterns(events)
        for pattern in action_patterns:
            if pattern.frequency >= 3:  # 重复出现3次以上
                await self.procedural.learn_from_demonstration(
                    pattern.name, pattern.steps
                )
        
        # 3. 更新会话摘要（用于情景记忆检索）
        summary = await self.llm.summarize_session(events)
        await self.episodic.store_summary(session_id, summary)
    
    def _find_action_patterns(self, events: list) -> list:
        """检测重复出现的操作模式"""
        action_seqs = []
        window_size = 3
        
        for i in range(len(events) - window_size + 1):
            seq = [e.action for e in events[i:i+window_size]]
            action_seqs.append(tuple(seq))
        
        # 统计频率
        from collections import Counter
        freq = Counter(action_seqs)
        
        return [Pattern(name=f"pattern_{hash(seq)}", 
                       steps=list(seq), 
                       frequency=count)
                for seq, count in freq.items() 
                if count >= 3]
```

### 固化触发策略

| 触发条件 | 时机 | 适用场景 |
|---------|------|---------|
| 会话结束触发 | 用户关闭对话/超时 | 最常见的固化时机 |
| 定时批量固化 | 每N小时/每天 | 大量用户场景 |
| 实时增量固化 | 每条重要消息 | 高价值交互场景 |
| 缓冲区溢出 | 短期缓存满 | 内存受限环境 |

---

## 六、基于注意力的记忆检索（Attention-Based Retrieval）

Agent在需要记忆时，不是全量加载，而是通过注意力机制精准检索相关信息。

### 多路召回 + 精排架构

```
用户查询 (Query)
     │
     ├──▶ [语义检索] 向量相似度匹配 (Embedding)
     │         │
     ├──▶ [关键词检索] BM25精确匹配
     │         │
     ├──▶ [时间检索] 最近/特定时间段
     │         │
     ├──▶ [图检索] 实体关系遍历
     │         │
     └──▶ [类型过滤] 按记忆类型筛选
                │
                ▼
        ┌─────────────────┐
        │   候选合并池     │
        │  (Recall Pool)  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   交叉注意力精排  │
        │ Cross-Attention  │
        │   Re-ranking     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Top-K 记忆注入   │
        │ (进入上下文窗口)  │
        └─────────────────┘
```

```python
class AttentionBasedRetriever:
    """基于注意力机制的记忆检索器"""
    
    def __init__(self, vector_index, keyword_index, graph_store):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.graph_store = graph_store
    
    async def retrieve(self, query: str, context: dict, top_k: int = 5) -> list:
        """多路召回 + 精排"""
        
        # Phase 1: 多路召回（每路召回更多候选）
        candidates = {}
        
        # 路径1: 语义向量检索
        semantic_hits = await self.vector_index.search(
            query, filter=context.get("user_filter"), top_k=top_k * 3
        )
        for hit in semantic_hits:
            candidates[hit.id] = {"mem": hit, "semantic_score": hit.score}
        
        # 路径2: 关键词检索
        keyword_hits = await self.keyword_index.search(query, top_k=top_k * 2)
        for hit in keyword_hits:
            if hit.id in candidates:
                candidates[hit.id]["keyword_score"] = hit.score
            else:
                candidates[hit.id] = {"mem": hit, "keyword_score": hit.score}
        
        # 路径3: 实体关系图检索
        entities = self._extract_entities(query)
        for entity in entities:
            graph_hits = await self.graph_store.get_related(entity, depth=2)
            for hit in graph_hits:
                if hit.id in candidates:
                    candidates[hit.id]["graph_score"] = hit.relevance
        
        # Phase 2: 交叉注意力精排
        if not candidates:
            return []
        
        candidate_list = list(candidates.values())
        reranked = await self._cross_attention_rerank(query, candidate_list)
        
        return reranked[:top_k]
    
    async def _cross_attention_rerank(self, query: str, candidates: list) -> list:
        """使用交叉注意力对候选记忆进行精排"""
        
        # 构造精排输入
        pairs = []
        for c in candidates:
            mem_text = c["mem"].get("content", "")
            pairs.append(f"查询: {query}\n记忆: {mem_text}")
        
        # 批量打分
        scores = await self.llm.score_relevance(pairs)
        
        # 综合多路信号
        for c, attn_score in zip(candidates, scores):
            c["final_score"] = (
                0.4 * c.get("semantic_score", 0) +
                0.3 * attn_score +
                0.2 * c.get("keyword_score", 0) +
                0.1 * c.get("graph_score", 0)
            )
        
        return sorted(candidates, key=lambda c: c["final_score"], reverse=True)
```

---

## 七、记忆淘汰策略（Memory Eviction Policies）

有限的存储资源需要智能的淘汰策略，确保最有价值的记忆被保留。

### 策略对比

```
┌──────────────────────────────────────────────────────────┐
│                 记忆淘汰策略矩阵                          │
├──────────────┬──────────┬──────────┬──────────┬─────────┤
│     策略     │  算法    │  复杂度  │  适用层  │  原则   │
├──────────────┼──────────┼──────────┼──────────┼─────────┤
│ LRU          │ O(1)     │ 低       │ 工作记忆 │ 最近使用 │
│ LFU          │ O(1)     │ 低       │ 情景记忆 │ 最常使用 │
│ 重要性淘汰    │ O(n)     │ 中       │ 所有层   │ 重要性  │
│ 语义去重      │ O(n²)    │ 高       │ 语义记忆 │ 唯一性  │
│ 时间衰减     │ O(1)     │ 低       │ 情景记忆 │ 新鲜度  │
│ 价值密度      │ O(n)     │ 中       │ 所有层   │ 信息量  │
└──────────────┴──────────┴──────────┴──────────┴─────────┘
```

### 混合淘汰策略实现

```python
from collections import OrderedDict
import heapq

class HybridEvictionPolicy:
    """混合淘汰策略：结合多维度信号"""
    
    def __init__(self, max_capacity: int):
        self.max_capacity = max_capacity
        self.memories = {}  # id -> MemoryItem
        self.access_order = OrderedDict()  # LRU追踪
        self.frequency = {}  # LFU计数
    
    def should_evict(self) -> bool:
        return len(self.memories) >= self.max_capacity
    
    def evict(self) -> str:
        """选择要淘汰的记忆"""
        if not self.should_evict():
            return None
        
        # 计算每个记忆的淘汰优先级（分数越低越该淘汰）
        candidates = []
        for mid, mem in self.memories.items():
            eviction_score = self._compute_eviction_score(mid, mem)
            heapq.heappush(candidates, (eviction_score, mid))
        
        # 淘汰分数最低的
        _, evict_id = heapq.heappop(candidates)
        del self.memories[evict_id]
        self.access_order.pop(evict_id, None)
        self.frequency.pop(evict_id, None)
        
        return evict_id
    
    def _compute_eviction_score(self, mem_id: str, mem) -> float:
        """计算淘汰分数（越高越该保留）"""
        
        # 因子1: 最近使用时间（LRU）
        lru_score = 1.0 / (1.0 + mem.hours_since_access)
        
        # 因子2: 使用频率（LFU）
        freq_score = min(self.frequency.get(mem_id, 1) / 10.0, 1.0)
        
        # 因子3: 内在重要性
        importance = mem.importance_score
        
        # 因子4: 独特性（与其它记忆的语义重叠度越低越好）
        uniqueness = mem.uniqueness_score  # 预计算
        
        # 因子5: 关联性（与其他记忆的连接数）
        connectivity = min(mem.link_count / 5.0, 1.0)
        
        # 加权综合
        return (
            0.25 * lru_score +
            0.20 * freq_score +
            0.25 * importance +
            0.15 * uniqueness +
            0.15 * connectivity
        )
```

---

## 八、多层记忆层级设计模式

### 完整架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                   Agent 多层记忆架构全景图                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Layer 0: 工作记忆 (Working Memory)                        │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  System Prompt │ Recent Messages │ Retrieved Memory  │  │  │
│  │  │  ◀───────────── 上下文窗口 (128K tokens) ──────────▶ │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  生命周期: 单次推理        延迟: 0ms        容量: 窗口限制  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ▲                                   │
│                     检索/注入 │                                   │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │  Layer 1: 情景记忆 (Episodic Memory)                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ 当前会话  │  │ 近期会话  │  │ 历史会话  │  │ 关键事件  │  │  │
│  │  │ (实时)   │  │ (小时级)  │  │ (天级)   │  │ (永久)   │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  │  生命周期: 分钟~月       延迟: 10-100ms    容量: GB级     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ▲                                   │
│                     提取/固化 │                                   │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │  Layer 2: 语义记忆 (Semantic Memory)                       │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │  │
│  │  │ 用户画像   │  │ 领域知识   │  │ 实体关系知识图谱    │  │  │
│  │  │ (Preferences)│ │(Domain KB) │  │ (Knowledge Graph)  │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │  │
│  │  生命周期: 永久        延迟: 50-200ms     容量: 无限      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ▲                                   │
│                     反馈/修正 │                                   │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │  Layer 3: 程序记忆 (Procedural Memory)                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ 技能模板  │  │ 工具策略  │  │ 错误模式  │  │ 优化路径  │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  │  生命周期: 永久        延迟: 5-50ms       容量: 有限      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 数据流与层级交互

```python
class AgentMemoryArchitecture:
    """完整的Agent多层记忆架构"""
    
    def __init__(self):
        self.working_memory = WorkingMemory(max_tokens=128000)
        self.episodic_store = EpisodicMemoryStore(vector_db, metadata_db)
        self.semantic_store = SemanticKnowledgeGraph(graph_db)
        self.procedural_store = ProceduralMemoryStore(skill_db)
        self.consolidation = MemoryConsolidationEngine(
            self.episodic_store, self.semantic_store, self.procedural_store
        )
        self.retriever = AttentionBasedRetriever(vector_db, keyword_db, graph_db)
    
    async def process_turn(self, user_input: str, session_context: dict):
        """处理一轮用户交互"""
        
        # ── Step 1: 检索相关记忆（从所有长期层） ──
        episodic_memories = await self.episodic_store.recall_similar(
            user_input, session_context["user_id"]
        )
        semantic_memories = await self.semantic_store.query_subgraph(
            self._extract_key_entities(user_input)
        )
        procedural_memories = await self.procedural_store.retrieve_skill(user_input)
        
        all_memories = episodic_memories + semantic_memories
        if procedural_memories:
            all_memories.append(procedural_memories)
        
        # ── Step 2: 多路召回 + 精排 ──
        retrieved = await self.retriever.retrieve(
            user_input, 
            context=session_context,
            top_k=7
        )
        
        # ── Step 3: 注入工作记忆（上下文窗口） ──
        context = self.working_memory.build_context(
            session_context["messages"],
            retrieved
        )
        
        # ── Step 4: LLM推理 ──
        response = await self.llm.generate(context)
        
        # ── Step 5: 更新工作记忆 ──
        self.working_memory.add_turn(user_input, response)
        
        # ── Step 6: 记录情景事件 ──
        event = EpisodicEvent(
            intent=self._classify_intent(user_input),
            action=response.tool_calls or "text_response",
            result=response.text[:200],
            importance_score=self._estimate_importance(user_input, response)
        )
        await self.episodic_store.record_event(event)
        
        # ── Step 7: 检查是否需要固化 ──
        if self._should_consolidate(session_context):
            await self.consolidation.consolidate_session(session_context["session_id"])
        
        # ── Step 8: 检查工作记忆是否需要压缩 ──
        if self.working_memory.estimated_tokens > self.working_memory.max_tokens * 0.8:
            self.working_memory.compress_and_evict(
                session_context["messages"]
            )
        
        return response
```

### 各层关键设计参数

| 维度 | 工作记忆 | 情景记忆 | 语义记忆 | 程序记忆 |
|------|---------|---------|---------|---------|
| **存储介质** | 模型上下文 | 向量DB + 关系DB | 知识图谱 | 模式库 |
| **访问延迟** | 0ms（内存） | 10-100ms | 50-200ms | 5-50ms |
| **容量上限** | 128K tokens | 100万条/用户 | 理论无限 | ~1万条 |
| **淘汰策略** | 滑动窗口/摘要 | 时间衰减+重要性 | 语义去重 | 成功率过滤 |
| **固化频率** | N/A | 会话结束 | 增量实时 | 多次成功后 |
| **读写比** | 极高（每次推理） | 高（检索为主） | 中（查询+更新） | 低（检索为主） |

---

## 总结与最佳实践

### 设计原则

1. **分层解耦**：每层记忆独立存储，通过统一接口交互，避免耦合
2. **按需检索**：不要全量加载，使用多路召回 + 精排按需获取
3. **渐进固化**：短期信息逐步整合为长期知识，避免一次性处理
4. **智能遗忘**：淘汰不是简单删除，而是基于多维度信号的智能决策
5. **反馈闭环**：程序记忆需要持续学习，根据成功/失败不断优化

### 技术选型建议

- **向量数据库**：Qdrant / Milvus / Weaviate（情景与语义记忆）
- **图数据库**：Neo4j / NebulaGraph（语义关系网络）
- **嵌入模型**：text-embedding-3-small（成本优先）/ BGE-M3（多语言）
- **LLM**：GPT-4o / Claude Sonnet（记忆提取与摘要）

Agent记忆架构是一个持续演进的系统。随着模型能力的提升和存储技术的发展，未来的记忆系统将更加接近人类的认知结构——不仅记住发生了什么，更理解为什么重要，以及如何在未来的决策中运用这些记忆。

---

*本文是Agent记忆架构系列的第一篇，后续将深入探讨向量检索优化、记忆压缩算法、以及多Agent共享记忆等高级主题。*
