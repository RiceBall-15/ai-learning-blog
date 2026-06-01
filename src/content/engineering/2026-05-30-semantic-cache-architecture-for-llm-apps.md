---
title: "AI应用的语义缓存架构：从Token级到请求级的智能缓存体系"
description: "深度解析语义缓存的架构设计、向量化匹配策略、一致性保障机制与生产级部署方案，降低LLM API调用成本40-70%"
date: 2026-05-30
author: "RiceBall-15"
category: "engineering"
subCategory: infra
tags: ["语义缓存", "Semantic Cache", "LLM成本优化", "向量相似度", "Redis", "AI工程化"]
draft: false
---

# AI应用的语义缓存架构：从Token级到请求级的智能缓存体系

## 一、引言：LLM应用的成本黑洞

### 1.1 为什么传统缓存在LLM场景失效？

传统应用的缓存策略基于一个核心假设：**相同输入 → 相同输出**。但LLM应用打破了这个假设：

```
┌─────────────────────────────────────────────────────────────────────┐
│              传统缓存 vs LLM语义缓存的差异                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  传统缓存（精确匹配）:                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Key: "什么是机器学习？"                                       │   │
│  │  Hash: 0xa3f2b1...                                          │   │
│  │  命中: 仅完全相同的字符串                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  LLM实际查询模式:                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  "什么是机器学习？"                                           │   │
│  │  "机器学习是什么？"          ← 语义相同，字符串不同            │   │
│  │  "请解释一下机器学习"         ← 语义相同，表达不同             │   │
│  │  "ML的定义是什么？"          ← 语义相同，使用缩写             │   │
│  │  "机器学习的基本概念"         ← 语义高度相似                   │   │
│  │                                                              │   │
│  │  传统缓存命中率: ~5%                                          │   │
│  │  语义缓存命中率: ~40-60%                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 语义缓存的价值量化

以一个典型的客服AI应用为例（日均10万次请求）：

| 指标 | 无缓存 | 精确缓存 | 语义缓存 |
|------|--------|----------|----------|
| **日请求量** | 100,000 | 100,000 | 100,000 |
| **缓存命中率** | 0% | 5% | 45% |
| **实际API调用** | 100,000 | 95,000 | 55,000 |
| **日均成本（GPT-4o）** | $1,500 | $1,425 | $825 |
| **月成本** | $45,000 | $42,750 | $24,750 |
| **月节省** | - | $2,250 (5%) | **$20,250 (45%)** |

## 二、语义缓存的核心架构

### 2.1 三层缓存架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    三层语义缓存架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Layer 1: 精确缓存层 (Exact Match Cache)                     │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  Redis Hash / Memcached                               │  │    │
│  │  │  Key: query_hash(query + system_prompt + model)       │  │    │
│  │  │  延迟: <1ms | 命中率: 5-15%                            │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │ 未命中                                    │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Layer 2: 前缀缓存层 (Prefix Match Cache)                    │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  Trie树 / 前缀索引                                      │  │    │
│  │  │  匹配: "什么是机器学习" → "什么是机器学习的..."         │  │    │
│  │  │  延迟: 2-5ms | 命中率: 10-20%                           │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │ 未命中                                    │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Layer 3: 语义缓存层 (Semantic Match Cache)                  │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  向量数据库 (Redis Vector / Milvus / Qdrant)          │  │    │
│  │  │  匹配: embedding相似度 > threshold                      │  │    │
│  │  │  延迟: 5-20ms | 命中率: 30-50%                          │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          │ 未命中                                    │
│                          ▼                                           │
│                  ┌──────────────────┐                               │
│                  │  调用LLM API      │                               │
│                  │  写入三层缓存      │                               │
│                  └──────────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 语义匹配的数学原理

语义缓存的核心是将查询文本转换为向量，通过向量相似度匹配语义相近的缓存条目：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    语义匹配流程                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Query: "什么是机器学习？"                                            │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────┐                                               │
│  │  Embedding模型    │  text-embedding-3-small / BGE-M3             │
│  │  (768维向量)      │  延迟: 10-30ms                                │
│  └────────┬─────────┘                                               │
│           ▼                                                          │
│  [0.12, -0.34, 0.56, ..., 0.78]  (768维浮点向量)                    │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │  向量数据库检索    │  ANN近似最近邻搜索                             │
│  │  Top-K + 阈值过滤  │  cosine similarity > 0.92                    │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  命中缓存条目:                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  原始查询: "机器学习是什么？"                                   │   │
│  │  相似度: 0.96                                                 │   │
│  │  缓存响应: "机器学习是人工智能的一个分支..."                     │   │
│  │  缓存时间: 2026-05-29 14:30:00                                │   │
│  │  剩余TTL: 23h 30m                                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 三、核心实现

### 3.1 基础语义缓存实现

```python
import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from openai import OpenAI
import hashlib
import json
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    response: str
    embedding: list[float]
    model: str
    created_at: datetime
    ttl_hours: int = 24
    hit_count: int = 0
    metadata: dict = None

class SemanticCache:
    """
    三层语义缓存实现
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        similarity_threshold: float = 0.92,
        exact_match_ttl: int = 3600,      # 精确缓存1小时
        semantic_match_ttl: int = 86400,   # 语义缓存24小时
    ):
        self.redis = Redis.from_url(redis_url)
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.exact_match_ttl = exact_match_ttl
        self.semantic_match_ttl = semantic_match_ttl
        
        # 初始化向量索引
        self._init_vector_index()
    
    def _init_vector_index(self):
        """初始化Redis向量搜索索引"""
        try:
            # 检查索引是否已存在
            self.redis.ft("semantic_cache").info()
        except Exception:
            # 创建新的向量索引
            schema = [
                TextField("query"),
                TextField("response"),
                TextField("model"),
                VectorField(
                    "embedding",
                    "VECTOR",
                    "FLAT", {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dim,
                        "DISTANCE_METRIC": "COSINE",
                    }
                ),
                NumericField("created_at"),
                NumericField("hit_count"),
            ]
            
            self.redis.ft("semantic_cache").create_index(
                schema,
                definition=IndexDefinition(
                    prefix=["cache:"],
                    index_type=IndexType.HASH,
                )
            )
    
    def _get_embedding(self, text: str) -> list[float]:
        """获取文本的embedding向量"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _exact_match_key(self, query: str, model: str) -> str:
        """生成精确匹配的缓存Key"""
        content = f"{query}:{model}"
        return f"cache:exact:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _get_context_hash(self, system_prompt: str, model: str) -> str:
        """生成上下文指纹（system_prompt + model）"""
        content = f"{system_prompt}:{model}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get(
        self,
        query: str,
        system_prompt: str = "",
        model: str = "gpt-4o",
        use_semantic: bool = True,
    ) -> Optional[str]:
        """
        从缓存中获取响应
        
        Args:
            query: 用户查询
            system_prompt: 系统提示词
            model: 模型名称
            use_semantic: 是否启用语义缓存
            
        Returns:
            缓存的响应或None
        """
        context_hash = self._get_context_hash(system_prompt, model)
        
        # Layer 1: 精确匹配
        exact_key = self._exact_match_key(query, context_hash)
        cached = self.redis.get(exact_key)
        if cached:
            entry = json.loads(cached)
            self._update_hit_count(exact_key, entry)
            return entry["response"]
        
        # Layer 2: 语义匹配
        if use_semantic:
            embedding = self._get_embedding(query)
            
            # 向量搜索
            results = self.redis.ft("semantic_cache").search(
                f"(@model:{context_hash})=>[KNN 5 @embedding $vec AS score]",
                {
                    "vec": np.array(embedding, dtype=np.float32).tobytes()
                },
                dialect=2,
            )
            
            # 检查最高相似度的结果
            for doc in results.docs:
                score = float(doc.score)
                if score >= self.similarity_threshold:
                    # 检查TTL是否过期
                    created_at = datetime.fromtimestamp(int(doc.created_at))
                    if datetime.now() - created_at < timedelta(hours=24):
                        # 命中语义缓存
                        self._update_hit_count(
                            f"cache:{doc.id}", 
                            json.loads(doc.json())
                        )
                        return doc.response
        
        return None
    
    def set(
        self,
        query: str,
        response: str,
        system_prompt: str = "",
        model: str = "gpt-4o",
        ttl_hours: int = 24,
    ):
        """
        写入缓存
        
        Args:
            query: 用户查询
            response: LLM响应
            system_prompt: 系统提示词
            model: 模型名称
            ttl_hours: 过期时间（小时）
        """
        context_hash = self._get_context_hash(system_prompt, model)
        embedding = self._get_embedding(query)
        
        # 写入精确缓存
        exact_key = self._exact_match_key(query, context_hash)
        entry = {
            "query": query,
            "response": response,
            "model": context_hash,
            "created_at": int(datetime.now().timestamp()),
            "hit_count": 0,
        }
        self.redis.setex(exact_key, self.exact_match_ttl, json.dumps(entry))
        
        # 写入语义缓存
        cache_id = hashlib.md5(f"{query}:{context_hash}".encode()).hexdigest()
        semantic_key = f"cache:{cache_id}"
        
        self.redis.hset(semantic_key, mapping={
            "query": query,
            "response": response,
            "model": context_hash,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
            "created_at": str(int(datetime.now().timestamp())),
            "hit_count": "0",
        })
        self.redis.expire(semantic_key, self.semantic_match_ttl)
    
    def _update_hit_count(self, key: str, entry: dict):
        """更新命中次数"""
        entry["hit_count"] = entry.get("hit_count", 0) + 1
        self.redis.set(key, json.dumps(entry))
```

### 3.2 高级特性：上下文感知缓存

```python
class ContextAwareSemanticCache(SemanticCache):
    """
    上下文感知的语义缓存
    考虑system_prompt、conversation_history等因素
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 多轮对话缓存
        self.conversation_cache = {}
    
    def get_with_context(
        self,
        query: str,
        conversation_history: list[dict],
        system_prompt: str = "",
        model: str = "gpt-4o",
    ) -> Optional[str]:
        """
        基于完整上下文的缓存匹配
        """
        # 1. 生成上下文指纹
        context_fingerprint = self._compute_context_fingerprint(
            conversation_history, system_prompt
        )
        
        # 2. 检查对话上下文缓存
        context_key = f"ctx:{context_fingerprint}:{hashlib.md5(query.encode()).hexdigest()}"
        cached = self.redis.get(context_key)
        if cached:
            return json.loads(cached)["response"]
        
        # 3. 语义匹配（考虑上下文）
        # 将query + 最近N轮对话拼接后进行embedding
        context_query = self._build_context_query(query, conversation_history)
        embedding = self._get_embedding(context_query)
        
        # 搜索向量数据库
        results = self.redis.ft("semantic_cache").search(
            f"(@model:{self._get_context_hash(system_prompt, model)})=>[KNN 5 @embedding $vec AS score]",
            {"vec": np.array(embedding, dtype=np.float32).tobytes()},
            dialect=2,
        )
        
        # 检查上下文匹配
        for doc in results.docs:
            if float(doc.score) >= self.similarity_threshold:
                # 额外验证上下文一致性
                if self._validate_context_consistency(
                    doc, conversation_history
                ):
                    return doc.response
        
        return None
    
    def _compute_context_fingerprint(
        self, 
        history: list[dict],
        system_prompt: str
    ) -> str:
        """
        计算上下文指纹
        只取最近3轮对话，避免指纹过于稀疏
        """
        recent_history = history[-3:] if len(history) > 3 else history
        content = json.dumps(recent_history + [{"role": "system", "content": system_prompt}])
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _build_context_query(
        self, 
        query: str, 
        history: list[dict]
    ) -> str:
        """
        构建包含上下文的查询字符串
        """
        # 取最近2轮对话作为上下文
        recent = history[-2:] if len(history) > 2 else history
        context_parts = [f"{msg['role']}: {msg['content']}" for msg in recent]
        context_parts.append(f"user: {query}")
        return " ".join(context_parts)
    
    def _validate_context_consistency(
        self, 
        cached_doc, 
        current_history: list[dict]
    ) -> bool:
        """
        验证缓存条目的上下文与当前上下文是否一致
        防止不同对话上下文的误命中
        """
        # 简化实现：检查对话主题是否一致
        cached_context = cached_doc.get("context_hash", "")
        current_context = self._compute_context_fingerprint(current_history, "")
        
        # 允许一定程度的上下文差异
        return cached_context == current_context
```

### 3.3 缓存一致性保障

```python
class CacheConsistencyManager:
    """
    缓存一致性管理器
    处理缓存失效、更新和并发问题
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def invalidate_by_pattern(self, pattern: str):
        """
        按模式失效缓存
        用于知识库更新后清除相关缓存
        """
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(
                cursor, match=f"cache:*{pattern}*", count=100
            )
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
    
    def invalidate_stale_entries(
        self, 
        max_age_hours: int = 24,
        min_hit_count: int = 2,
    ):
        """
        清理过期和低频缓存
        """
        cursor = 0
        deleted_count = 0
        
        while True:
            cursor, keys = self.redis.scan(cursor, match="cache:*", count=100)
            
            for key in keys:
                entry = self.redis.hgetall(key)
                if not entry:
                    continue
                    
                created_at = int(entry.get(b"created_at", 0))
                hit_count = int(entry.get(b"hit_count", 0))
                
                age_hours = (datetime.now().timestamp() - created_at) / 3600
                
                # 删除过期或低频条目
                if age_hours > max_age_hours or hit_count < min_hit_count:
                    self.redis.delete(key)
                    deleted_count += 1
            
            if cursor == 0:
                break
        
        return deleted_count
    
    def handle_model_update(
        self, 
        old_model: str, 
        new_model: str,
    ):
        """
        模型更新时的缓存迁移
        """
        # 标记旧模型缓存为即将过期
        pattern = f"cache:*{old_model}*"
        cursor = 0
        
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                # 设置较短的TTL让旧缓存自然过期
                self.redis.expire(key, 300)  # 5分钟后过期
            if cursor == 0:
                break
```

## 四、生产级部署方案

### 4.1 部署架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    语义缓存生产部署架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │  应用服务     │                                                   │
│  │  (FastAPI)   │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  语义缓存服务 (独立微服务)                                      │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────────┐   │   │
│  │  │ Cache Layer│ │ Embedding  │ │ Consistency Manager    │   │   │
│  │  │ (3层缓存)  │ │ Service    │ │ (一致性保障)            │   │   │
│  │  └─────┬──────┘ └─────┬──────┘ └────────────────────────┘   │   │
│  └────────┼──────────────┼─────────────────────────────────────┘   │
│           │              │                                          │
│           ▼              ▼                                          │
│  ┌────────────────┐ ┌────────────────┐                             │
│  │  Redis Cluster │ │  Embedding API │                             │
│  │  (向量存储)    │ │  (OpenAI/BGE)  │                             │
│  │  3主3从        │ │  多模型支持     │                             │
│  └────────────────┘ └────────────────┘                             │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  监控告警                                                      │   │
│  │  • 缓存命中率 (目标: >40%)                                     │   │
│  │  • P99延迟 (目标: <50ms)                                      │   │
│  │  • 内存使用率 (目标: <70%)                                     │   │
│  │  • 命中率下降告警 (阈值: <30%)                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 性能优化

```python
# 优化1: Embedding批处理
class BatchEmbeddingService:
    """批量embedding服务，减少API调用次数"""
    
    def __init__(self, batch_size: int = 32, max_wait_ms: int = 50):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_queue = asyncio.Queue()
    
    async def get_embedding(self, text: str) -> list[float]:
        """获取单个文本的embedding"""
        future = asyncio.Future()
        await self.pending_queue.put((text, future))
        return await future
    
    async def _batch_processor(self):
        """批量处理embedding请求"""
        while True:
            batch = []
            try:
                # 收集批量请求
                item = await self.pending_queue.get()
                batch.append(item)
                
                # 在超时前尽量收集更多请求
                try:
                    while len(batch) < self.batch_size:
                        item = await asyncio.wait_for(
                            self.pending_queue.get(),
                            timeout=self.max_wait_ms / 1000
                        )
                        batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # 批量调用embedding API
                texts = [item[0] for item in batch]
                embeddings = await self._batch_embed(texts)
                
                # 返回结果
                for (text, future), embedding in zip(batch, embeddings):
                    future.set_result(embedding)
                    
            except Exception as e:
                for _, future in batch:
                    future.set_exception(e)

# 优化2: 本地向量索引（热数据）
class LocalVectorCache:
    """
    本地向量缓存层
    将高频访问的数据缓存在内存中
    """
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}  # 实际使用faiss或annoy
        self.access_count = {}
        self.max_size = max_size
    
    def get_local(self, query: str) -> Optional[str]:
        """从本地缓存获取"""
        if query in self.cache:
            self.access_count[query] = self.access_count.get(query, 0) + 1
            return self.cache[query]["response"]
        return None
    
    def set_local(self, query: str, response: str, embedding: list[float]):
        """写入本地缓存"""
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            least_used = min(self.access_count, key=self.access_count.get)
            del self.cache[least_used]
            del self.access_count[least_used]
        
        self.cache[query] = {
            "response": response,
            "embedding": embedding,
        }
        self.access_count[query] = 1

# 优化3: 预热策略
class CacheWarmer:
    """缓存预热服务"""
    
    def warm_by_query_log(
        self, 
        query_log: list[str],
        semantic_cache: SemanticCache,
    ):
        """
        根据历史查询日志预热缓存
        在服务启动时或低峰期执行
        """
        # 按频率排序
        from collections import Counter
        query_freq = Counter(query_log)
        top_queries = query_freq.most_common(1000)
        
        print(f"预热 {len(top_queries)} 个高频查询...")
        
        for query, freq in top_queries:
            # 检查是否已在缓存中
            if not semantic_cache.get(query):
                # 调用LLM生成响应并缓存
                response = call_llm(query)
                semantic_cache.set(query, response, ttl_hours=48)
```

## 五、缓存策略对比

### 5.1 不同缓存粒度的对比

| 策略 | 命中率 | 延迟 | 一致性 | 适用场景 |
|------|--------|------|--------|----------|
| **精确缓存** | 5-15% | <1ms | 强 | 精确重复查询 |
| **前缀缓存** | 10-20% | 2-5ms | 强 | 补全类查询 |
| **语义缓存** | 30-50% | 5-20ms | 最终一致 | 开放式问答 |
| **混合缓存** | 40-60% | 5-20ms | 最终一致 | 生产环境推荐 |

### 5.2 相似度阈值选择

```
┌─────────────────────────────────────────────────────────────────────┐
│              相似度阈值 vs 命中率 vs 准确率关系                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  阈值    命中率    准确率    误命中风险                                │
│  ──────────────────────────────────────                             │
│  0.80    65%      85%     ⚠️ 高（可能返回不相关答案）                 │
│  0.85    55%      90%     ⚠️ 中高                                    │
│  0.90    45%      95%     ✅ 中（生产推荐下限）                       │
│  0.92    40%      97%     ✅ 低（推荐默认值）                         │
│  0.95    30%      99%     ✅ 极低（保守策略）                         │
│  0.98    15%      99.9%   ✅ 几乎无误命中                            │
│                                                                      │
│  建议:                                                               │
│  • 客服系统: 0.92-0.95（准确性优先）                                 │
│  • 知识问答: 0.90-0.92（平衡命中率和准确性）                          │
│  • 创意生成: 不建议使用语义缓存                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 六、监控与运维

### 6.1 关键监控指标

```python
# Prometheus指标定义
from prometheus_client import Counter, Histogram, Gauge

# 缓存命中率
cache_hits_total = Counter(
    'semantic_cache_hits_total',
    'Total cache hits',
    ['cache_layer', 'match_type']  # layer: exact/semantic, type: hit/miss
)

# 缓存延迟
cache_latency_seconds = Histogram(
    'semantic_cache_latency_seconds',
    'Cache lookup latency',
    ['operation'],  # get/set/invalidate
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
)

# 缓存大小
cache_size_bytes = Gauge(
    'semantic_cache_size_bytes',
    'Current cache size in bytes',
    ['cache_layer']
)

# Embedding延迟
embedding_latency_seconds = Histogram(
    'embedding_latency_seconds',
    'Embedding generation latency',
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
)

# 告警规则（Grafana/Prometheus）
ALERT_RULES = {
    "LowCacheHitRate": {
        "expr": "rate(semantic_cache_hits_total{match_type='hit'}[5m]) / rate(semantic_cache_hits_total[5m]) < 0.3",
        "for": "5m",
        "severity": "warning",
        "message": "缓存命中率低于30%，请检查相似度阈值或缓存数据质量",
    },
    "HighCacheLatency": {
        "expr": "histogram_quantile(0.99, rate(semantic_cache_latency_seconds_bucket[5m])) > 0.05",
        "for": "3m",
        "severity": "critical",
        "message": "缓存P99延迟超过50ms，请检查Redis性能",
    },
    "CacheMemoryHigh": {
        "expr": "redis_memory_used_bytes / redis_memory_max_bytes > 0.8",
        "for": "5m",
        "severity": "warning",
        "message": "Redis内存使用率超过80%，请清理过期缓存或扩容",
    },
}
```

### 6.2 A/B测试框架

```python
class CacheABTest:
    """
    缓存策略A/B测试框架
    用于验证不同缓存配置的效果
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.experiments = {}
    
    def create_experiment(
        self,
        name: str,
        control: dict,      # 对照组配置
        treatment: dict,     # 实验组配置
        traffic_split: float = 0.5,  # 流量分配比例
    ):
        """创建A/B测试实验"""
        self.experiments[name] = {
            "control": control,
            "treatment": treatment,
            "traffic_split": traffic_split,
            "start_time": datetime.now(),
            "metrics": {
                "control": {"hits": 0, "misses": 0, "latency_sum": 0},
                "treatment": {"hits": 0, "misses": 0, "latency_sum": 0},
            }
        }
    
    def get_config(self, experiment_name: str, user_id: str) -> dict:
        """根据用户ID决定使用哪组配置"""
        experiment = self.experiments[experiment_name]
        
        # 基于用户ID的确定性分配
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        bucket = int(user_hash[:8], 16) / 0xFFFFFFFF
        
        if bucket < experiment["traffic_split"]:
            return experiment["treatment"]
        else:
            return experiment["control"]
    
    def record_metric(
        self, 
        experiment_name: str, 
        group: str,
        hit: bool,
        latency: float,
    ):
        """记录实验指标"""
        metrics = self.experiments[experiment_name]["metrics"][group]
        if hit:
            metrics["hits"] += 1
        else:
            metrics["misses"] += 1
        metrics["latency_sum"] += latency
    
    def get_results(self, experiment_name: str) -> dict:
        """获取实验结果"""
        experiment = self.experiments[experiment_name]
        results = {}
        
        for group in ["control", "treatment"]:
            metrics = experiment["metrics"][group]
            total = metrics["hits"] + metrics["misses"]
            
            results[group] = {
                "hit_rate": metrics["hits"] / total if total > 0 else 0,
                "avg_latency": metrics["latency_sum"] / total if total > 0 else 0,
                "total_requests": total,
            }
        
        # 计算提升
        control_hr = results["control"]["hit_rate"]
        treatment_hr = results["treatment"]["hit_rate"]
        results["improvement"] = {
            "hit_rate_lift": (treatment_hr - control_hr) / control_hr if control_hr > 0 else 0,
            "statistical_significance": self._compute_significance(
                experiment["metrics"]["control"],
                experiment["metrics"]["treatment"],
            )
        }
        
        return results
```

## 七、最佳实践总结

### 7.1 选型决策矩阵

```
┌─────────────────────────────────────────────────────────────────────┐
│                  语义缓存选型决策矩阵                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  你的LLM应用场景？                                                   │
│       │                                                              │
│       ├── 客服/FAQ ──→ 高相似度阈值(0.95) + 长TTL(48h)              │
│       │                                                              │
│       ├── 知识问答 ──→ 中等阈值(0.92) + 中TTL(24h)                  │
│       │                                                              │
│       ├── 代码助手 ──→ 不建议语义缓存（代码精确性要求高）              │
│       │                                                              │
│       ├── 内容生成 ──→ 低阈值(0.85) + 短TTL(6h) + 版本标记           │
│       │                                                              │
│       └── 搜索增强 ──→ 前缀缓存为主 + 语义缓存辅助                   │
│                                                                      │
│  数据规模？                                                          │
│       │                                                              │
│       ├── <10万条 ──→ Redis单机 + 本地向量索引                       │
│       │                                                              │
│       ├── 10-100万条 ──→ Redis Cluster + FAISS                      │
│       │                                                              │
│       └── >100万条 ──→ Milvus/Qdrant + 分布式Embedding服务           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 常见陷阱

| 陷阱 | 表现 | 解决方案 |
|------|------|----------|
| **阈值过低** | 返回不相关答案，用户投诉 | 从0.95开始，逐步降低 |
| **忽略上下文** | 不同对话返回相同答案 | 实现上下文感知缓存 |
| **缓存雪崩** | 大量缓存同时过期 | TTL添加随机抖动 |
| **数据漂移** | 知识更新后缓存未失效 | 实现主动失效机制 |
| **Embedding漂移** | 模型更新后相似度计算不一致 | 版本化Embedding模型 |
| **过度缓存** | 内存溢出 | LRU淘汰 + 定期清理 |

---

**参考资源：**
- [Redis Vector Search文档](https://redis.io/docs/interact/search-and-query/vector-search/)
- [GPT Cache开源项目](https://github.com/zilliztech/GPTCache)
- [LangChain缓存模块](https://python.langchain.com/docs/modules/model_io/llms/llm_caching)
- [Semantic Cache论文](https://arxiv.org/abs/2311.09396)
