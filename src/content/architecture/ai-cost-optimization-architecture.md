---
title: "AI应用成本优化架构：从模型路由到缓存策略的全链路成本控制方案"
description: "深入解析AI应用的六大成本优化架构模式，涵盖模型路由、语义缓存、Prompt压缩、批处理调度等关键技术，附真实场景的ROI分析与架构选型指南"
date: 2026-05-30
author: RiceBall-15
category: architecture
tags: ["AI架构", "成本优化", "模型路由", "语义缓存", "Prompt压缩", "系统架构"]
draft: false
---

## 一、引言：AI应用的成本陷阱

当你的AI应用从MVP走向生产环境，最让CTO夜不能寐的不是技术故障，而是月底的云账单。

一个典型的AI应用成本结构：

```
┌─────────────────────────────────────────────────────────────────────┐
│                  AI 应用月度成本构成（示例）                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GPU推理成本     ████████████████████████████████  62%              │
│  API调用成本     ██████████████                   28%              │
│  向量数据库       ████                              5%              │
│  存储与网络       ██                                3%              │
│  监控与运维       █                                 2%              │
│                                                                      │
│  月总成本: ¥180,000+                                                 │
│  年化成本: ¥2,160,000+                                               │
└─────────────────────────────────────────────────────────────────────┘
```

**核心矛盾**：用户期望的是"聪明且快速"的AI，而"聪明"往往意味着更大的模型和更高的成本。本文将从架构层面系统性地解决这个矛盾。

### 成本优化的三大原则

| 原则 | 说明 | 典型收益 |
|------|------|---------|
| **按需匹配** | 不同任务使用不同模型，避免大材小用 | 降低40-60%推理成本 |
| **消除重复** | 语义级别的缓存与复用 | 降低30-50%调用量 |
| **压缩输入** | 减少token消耗而不损失质量 | 降低20-40%token成本 |

## 二、架构全景：六大成本优化模式

```
┌─────────────────────────────────────────────────────────────────────┐
│                AI 成本优化架构全景                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        ┌──────────────┐                              │
│                        │  用户请求     │                              │
│                        └──────┬───────┘                              │
│                               │                                      │
│                        ┌──────▼───────┐                              │
│                        │  请求分类器   │                              │
│                        │  (意图识别)   │                              │
│                        └──────┬───────┘                              │
│                               │                                      │
│              ┌────────────────┼────────────────┐                     │
│              │                │                │                     │
│        ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐              │
│        │ 语义缓存   │   │ Prompt    │   │ 模型路由   │              │
│        │ (命中→直接 │   │ 压缩器    │   │ (按任务选  │              │
│        │  返回)     │   │           │   │  择模型)   │              │
│        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘              │
│              │               │                │                     │
│              │          ┌────▼────┐    ┌──────▼──────┐              │
│              │          │小模型   │    │  大模型      │              │
│              │          │(GPT-4o  │    │ (Claude     │              │
│              │          │ mini)   │    │  Opus)      │              │
│              │          └────┬────┘    └──────┬──────┘              │
│              │               │                │                     │
│              └───────────────┼────────────────┘                     │
│                              │                                       │
│                       ┌──────▼───────┐                              │
│                       │  响应后处理   │                              │
│                       │  (流式/批处理)│                              │
│                       └──────┬───────┘                              │
│                              │                                       │
│                       ┌──────▼───────┐                              │
│                       │  成本追踪    │                              │
│                       │  (按用户/场景)│                              │
│                       └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

## 三、模式一：智能模型路由

### 3.1 问题定义

不同任务对模型能力的需求差异巨大。让一个简单的FAQ问答调用GPT-4级别模型，就像用航空发动机驱动割草机。

### 3.2 路由架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    智能模型路由架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   请求分析层                                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │    │
│  │  │复杂度评估 │  │任务分类   │  │质量要求   │                  │    │
│  │  │(token数,  │  │(FAQ/创作/ │  │(高/中/低) │                  │    │
│  │  │ 多轮等)   │  │ 代码/分析)│  │          │                  │    │
│  │  └──────────┘  └──────────┘  └──────────┘                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   路由决策层                                  │    │
│  │                                                              │    │
│  │  任务类型        推荐模型            预估成本/1K tokens       │    │
│  │  ────────────────────────────────────────────────────────── │    │
│  │  简单FAQ         GPT-4o-mini         ¥0.001                 │    │
│  │  文本分类        Qwen-7B             ¥0.0005                │    │
│  │  代码生成        GPT-4o              ¥0.01                  │    │
│  │  复杂推理        Claude Opus          ¥0.075                 │    │
│  │  多模态理解      GPT-4o Vision        ¥0.01                  │    │
│  │  长文档摘要      Gemini 1.5 Pro       ¥0.007                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 实现方案

路由策略的核心是**任务复杂度评估器**。我们采用两层评估：

**第一层：规则路由（快速，零成本）**

```python
class RuleBasedRouter:
    """基于规则的快速路由，处理明确的低成本场景"""
    
    def route(self, request: ChatRequest) -> str:
        # 1. 长度规则：短文本用小模型
        if request.token_count < 200:
            return "gpt-4o-mini"
        
        # 2. 关键词规则：代码任务用代码模型
        if self._is_code_task(request.messages):
            return "gpt-4o"
        
        # 3. 任务类型规则
        task_type = self._classify_task(request.messages)
        TASK_MODEL_MAP = {
            "faq": "gpt-4o-mini",
            "classification": "qwen-7b",
            "translation": "gpt-4o-mini",
            "creative_writing": "gpt-4o",
            "complex_reasoning": "claude-opus",
        }
        return TASK_MODEL_MAP.get(task_type, "gpt-4o")
```

**第二层：分类器路由（精准，微小成本）**

```python
class ClassifierRouter:
    """基于分类器的精准路由，处理边界情况"""
    
    def __init__(self):
        # 一个轻量级分类器，判断请求复杂度
        self.classifier = load_model("complexity-classifier-v2")
    
    def route(self, request: ChatRequest) -> str:
        # 提取特征
        features = {
            "token_count": request.token_count,
            "has_context": len(request.messages) > 2,
            "requires_tool_use": request.tools is not None,
            "language_complexity": self._analyze_complexity(request),
        }
        
        # 分类器预测
        complexity = self.classifier.predict(features)  # low/medium/high
        
        ROUTE_MAP = {
            "low": "gpt-4o-mini",
            "medium": "gpt-4o",
            "high": "claude-opus",
        }
        return ROUTE_MAP[complexity]
```

### 3.4 路由效果对比

| 指标 | 无路由（全用GPT-4o） | 智能路由 | 改善幅度 |
|------|---------------------|---------|---------|
| 平均成本/请求 | ¥0.032 | ¥0.011 | -65.6% |
| 响应延迟(P50) | 1.2s | 0.8s | -33.3% |
| 用户满意度 | 4.2/5 | 4.3/5 | +2.4% |
| 月总成本 | ¥180,000 | ¥62,000 | -65.6% |

> **关键洞察**：智能路由不仅降低了成本，还提升了响应速度。因为简单任务用小模型处理更快，大模型只在真正需要时才被调用。

### 3.5 高级策略：级联路由（Cascade Routing）

对于边界情况，采用**级联策略**：先用小模型尝试，如果质量不达标再升级到大模型。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    级联路由策略                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  请求 → 小模型(4o-mini) → 质量评估                                    │
│                              │                                       │
│                     ┌────────┴────────┐                              │
│                     │                 │                              │
│                 质量达标           质量不达标                          │
│                     │                 │                              │
│                     ▼                 ▼                              │
│                 直接返回         中等模型(4o) → 质量评估                │
│                                             │                       │
│                                    ┌────────┴────────┐              │
│                                    │                 │              │
│                                质量达标           质量不达标          │
│                                    │                 │              │
│                                    ▼                 ▼              │
│                                直接返回      大模型(Claude Opus)      │
│                                                                      │
│  优势：90%的请求在第一层就满足，只有10%需要升级                         │
│  劣势：边界case延迟增加（多一次调用）                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 四、模式二：语义缓存

### 4.1 传统缓存的局限

传统的精确匹配缓存（exact match cache）对AI应用几乎无效。用户的问题表述千变万化：

- "Python怎么读取Excel文件？"
- "用Python读取Excel的方法是什么？"
- "如何用Python打开xlsx文件？"

这三个问题语义相同，但字符串完全不同。

### 4.2 语义缓存架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    语义缓存架构                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│  │ 用户查询  │ → │ Embedding │ → │ 向量相似度 │                      │
│  │          │    │ 生成      │    │ 搜索      │                      │
│  └──────────┘    └──────────┘    └────┬─────┘                      │
│                                       │                              │
│                              ┌────────┴────────┐                    │
│                              │                 │                    │
│                          相似度>0.95        相似度<0.95              │
│                              │                 │                    │
│                              ▼                 ▼                    │
│                    ┌─────────────┐   ┌──────────────┐              │
│                    │ 命中缓存     │   │ 未命中        │              │
│                    │ 直接返回     │   │ → 调用LLM    │              │
│                    │ (延迟<50ms) │   │ → 写入缓存    │              │
│                    └─────────────┘   └──────────────┘              │
│                                                                      │
│  缓存存储层：                                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Redis (热数据)     │  Milvus/Pinecone (全量)               │    │
│  │  - LRU淘汰策略      │  - 向量索引                            │    │
│  │  - TTL过期          │  - 持久化存储                          │    │
│  │  - QPS > 10,000     │  - 支持十亿级数据                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 核心实现

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
from pymilvus import connections, Collection

class SemanticCache:
    """语义缓存：基于向量相似度的AI响应缓存"""
    
    def __init__(self, similarity_threshold=0.95, ttl=3600):
        self.encoder = SentenceTransformer('bge-large-zh-v1.5')
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.milvus = Collection('ai_cache')
        self.threshold = similarity_threshold
        self.ttl = ttl
    
    def get(self, query: str) -> dict | None:
        """查询缓存，返回相似度最高的结果"""
        # 1. 生成查询向量
        query_embedding = self.encoder.encode(query).tolist()
        
        # 2. 先查Redis热缓存（快速路径）
        cached = self._search_redis(query_embedding)
        if cached and cached['similarity'] >= self.threshold:
            return cached['response']
        
        # 3. 再查Milvus全量缓存（精确路径）
        results = self.milvus.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=1,
            output_fields=["query", "response", "timestamp"]
        )
        
        if results and results[0].distance >= self.threshold:
            response = results[0].entity.get('response')
            # 回填Redis热缓存
            self._set_redis(query_embedding, response)
            return response
        
        return None
    
    def set(self, query: str, response: dict):
        """写入缓存"""
        embedding = self.encoder.encode(query).tolist()
        self.milvus.insert([[embedding], [query], [response], [time.time()]])
        self._set_redis(embedding, response)
```

### 4.4 缓存一致性挑战

语义缓存面临的独特挑战：

| 挑战 | 说明 | 解决方案 |
|------|------|---------|
| **时间敏感性** | "今天的天气"每天不同 | 基于内容的TTL策略 |
| **个性化差异** | 同一问题不同用户需要不同答案 | 用户ID作为缓存键的一部分 |
| **知识更新** | 模型更新后旧缓存可能不准确 | 缓存版本化 + 主动失效 |
| **幻觉传播** | 错误回答被缓存后反复返回 | 置信度阈值过滤 |

```python
class CacheInvalidationStrategy:
    """智能缓存失效策略"""
    
    def should_cache(self, query: str, response: dict, context: dict) -> bool:
        """判断是否应该缓存"""
        
        # 1. 时间敏感查询不缓存
        TIME_SENSITIVE_PATTERNS = ['今天', '现在', '最新', '当前']
        if any(p in query for p in TIME_SENSITIVE_PATTERNS):
            return False
        
        # 2. 低置信度响应不缓存
        if response.get('confidence', 0) < 0.8:
            return False
        
        # 3. 包含个人数据的响应需要标记
        if self._contains_personal_data(response):
            return False
        
        # 4. 有工具调用的响应需要验证工具状态
        if response.get('tool_calls'):
            return False
        
        return True
```

## 五、模式三：Prompt压缩

### 5.1 为什么Prompt压缩重要

在一个RAG应用中，Prompt的典型构成：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Prompt Token 构成分析                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  System Prompt      ████████████                 15% (500 tokens)  │
│  检索文档(RAG)      ██████████████████████████   35% (1,500 tokens)│
│  对话历史           ████████████████████         25% (1,000 tokens) │
│  用户问题           ███                          5%  (200 tokens)  │
│  输出示例           ████████                     10% (400 tokens)  │
│  工具描述           ███████                      10% (400 tokens)  │
│                                                                      │
│  总计: ~4,000 tokens/请求                                            │
│  月总消耗: 4,000 × 100万请求 = 40亿 tokens                           │
│  月成本(按$3/1M tokens): ¥870,000                                    │
└─────────────────────────────────────────────────────────────────────┘
```

如果能将Prompt压缩30%，直接节省**¥260,000/月**。

### 5.2 三层压缩策略

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Prompt 三层压缩策略                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  第一层：结构化压缩（静态，开发时完成）                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Before: "你是一个专业的AI助手，你需要根据用户的问题..."      │    │
│  │  After:  "Role: AI助手 | Task: 回答用户问题 | Tone: 专业"    │    │
│  │  收益: System Prompt 减少 40-60%                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  第二层：动态压缩（运行时，根据上下文）                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  - 对话历史摘要：超过N轮后自动摘要                            │    │
│  │  - 检索文档截断：只保留最相关的段落                            │    │
│  │  - 工具描述过滤：只加载本次可能用到的工具                      │    │
│  │  收益: 每次请求减少 20-40% tokens                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  第三层：语义压缩（智能，LLM辅助）                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  - 小模型摘要长文档                                          │    │
│  │  - 关键信息提取                                              │    │
│  │  - 冗余消除                                                  │    │
│  │  收益: RAG文档消耗减少 50-70%                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 对话历史压缩实现

```python
class ConversationCompressor:
    """对话历史智能压缩器"""
    
    def __init__(self, max_history_tokens=1000, summary_model="gpt-4o-mini"):
        self.max_tokens = max_history_tokens
        self.summary_model = summary_model
    
    def compress(self, messages: list[dict]) -> list[dict]:
        """压缩对话历史"""
        
        # 1. 计算当前token数
        total_tokens = sum(count_tokens(m['content']) for m in messages)
        
        if total_tokens <= self.max_tokens:
            return messages  # 无需压缩
        
        # 2. 保留system prompt和最近3轮对话
        system_msg = messages[0]  # 假设第一条是system
        recent_msgs = messages[-6:]  # 最近3轮（user+assistant）
        old_msgs = messages[1:-6]   # 需要压缩的历史
        
        # 3. 用小模型摘要历史
        if old_msgs:
            summary = self._summarize(old_msgs)
            compressed = [
                system_msg,
                {"role": "system", "content": f"[对话历史摘要] {summary}"},
                *recent_msgs
            ]
        else:
            compressed = messages
        
        return compressed
    
    def _summarize(self, messages: list[dict]) -> str:
        """使用小模型生成对话摘要"""
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        
        response = call_llm(
            model=self.summary_model,
            messages=[{
                "role": "system",
                "content": "请用3-5句话总结以下对话的关键信息，包括用户的主要需求和已达成的结论。"
            }, {
                "role": "user",
                "content": conversation
            }],
            max_tokens=200
        )
        return response['content']
```

## 六、模式四：批量请求优化

### 6.1 从实时到批处理

并非所有AI请求都需要实时响应。对于可以延迟处理的场景，批处理能显著降低成本。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    请求时间敏感度分类                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  实时要求（<1s）    用户对话、搜索建议、实时翻译                      │
│       ↓                                                             │
│  交互式（1-5s）     内容生成、代码补全、文档分析                      │
│       ↓                                                             │
│  近实时（5-30s）    长文档摘要、批量分类、数据分析                    │
│       ↓                                                             │
│  异步批处理（分钟级）数据标注、报告生成、模型评估                     │
│       ↓                                                             │
│  离线批处理（小时级）训练数据准备、全量文档处理、指标计算              │
│                                                                      │
│  成本对比：                                                          │
│  实时API: $3.00 / 1M tokens                                         │
│  批处理API: $1.50 / 1M tokens (50%折扣)                             │
│  离线自部署: $0.80 / 1M tokens (73%折扣)                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 智能批处理调度器

```python
class BatchScheduler:
    """智能批处理调度器"""
    
    def __init__(self):
        self.realtime_queue = asyncio.Queue()
        self.batch_queue = asyncio.Queue()
        self.batch_size = 100
        self.batch_interval = 60  # 秒
    
    async def submit(self, request: ChatRequest) -> str:
        """提交请求，自动判断是否走批处理"""
        
        # 1. 判断时间敏感度
        urgency = self._assess_urgency(request)
        
        if urgency == "realtime":
            return await self._process_realtime(request)
        
        elif urgency == "interactive":
            # 短等待后批处理
            return await self._process_batch_with_timeout(request, timeout=5)
        
        else:
            # 纯批处理
            future = asyncio.Future()
            await self.batch_queue.put((request, future))
            return await future
    
    def _assess_urgency(self, request: ChatRequest) -> str:
        """评估请求的时间敏感度"""
        
        # 1. 用户明确标记
        if request.metadata.get("priority") == "high":
            return "realtime"
        
        # 2. 任务类型
        BATCH_FRIENDLY = ["summarization", "classification", "translation"]
        if request.task_type in BATCH_FRIENDLY:
            return "batch"
        
        # 3. 是否有用户在等待（WebSocket连接）
        if request.source == "api_sync":
            return "realtime"
        
        return "interactive"
    
    async def _batch_processor(self):
        """批处理执行器"""
        while True:
            batch = []
            deadline = time.time() + self.batch_interval
            
            # 收集批次
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    item = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=max(0.1, deadline - time.time())
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                # 批量调用API
                results = await self._call_batch_api(
                    [item[0] for item in batch]
                )
                # 设置结果
                for (req, future), result in zip(batch, results):
                    future.set_result(result)
```

## 七、模式五：模型蒸馏与边缘部署

### 7.1 蒸馏架构

将大模型的知识蒸馏到小模型，在边缘设备上运行。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    模型蒸馏成本对比                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  场景: 智能客服（日均100万次对话）                                    │
│                                                                      │
│  方案A: 全云端 (GPT-4o)                                              │
│  ├─ 单次成本: ¥0.03                                                 │
│  ├─ 日成本: ¥30,000                                                 │
│  └─ 年成本: ¥10,950,000                                             │
│                                                                      │
│  方案B: 蒸馏+边缘 (GPT-4o蒸馏到Qwen-7B)                              │
│  ├─ 蒸馏训练成本: ¥50,000 (一次性)                                   │
│  ├─ 边缘GPU成本: ¥200/天 (8卡A10)                                   │
│  ├─ 云端回退(10%复杂case): ¥3,000/天                                 │
│  ├─ 日成本: ¥3,200                                                  │
│  └─ 年成本: ¥1,208,000                                              │
│                                                                      │
│  节省: ¥9,742,000/年 (89%)                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 混合部署策略

```python
class HybridDeployment:
    """混合部署：边缘+云端协同"""
    
    def __init__(self, edge_model, cloud_model):
        self.edge_model = edge_model    # Qwen-7B on GPU
        self.cloud_model = cloud_model  # GPT-4o API
    
    async def predict(self, request) -> dict:
        # 1. 边缘模型快速推理
        edge_result = await self.edge_model.predict(request)
        
        # 2. 置信度评估
        if edge_result.confidence >= 0.85:
            return edge_result  # 高置信度，直接返回
        
        # 3. 低置信度回退到云端
        cloud_result = await self.cloud_model.predict(request)
        
        # 4. 用云端结果更新边缘模型（在线学习）
        self._update_edge_model(request, cloud_result)
        
        return cloud_result
```

## 八、模式六：成本监控与告警

### 8.1 成本可观测性架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    成本监控与告警体系                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    数据采集层                                 │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │API调用日志│  │Token使用  │  │模型路由   │  │延迟与质量  │  │    │
│  │  │          │  │量        │  │决策日志   │  │指标       │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    分析引擎                                   │    │
│  │                                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │    │
│  │  │实时聚合   │  │异常检测   │  │成本预测   │                 │    │
│  │  │(每分钟)   │  │(Z-Score) │  │(时序模型) │                 │    │
│  │  └──────────┘  └──────────┘  └──────────┘                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    告警与动作                                 │    │
│  │                                                              │    │
│  │  预算告警: 月度预算超过80% → 通知                             │    │
│  │  异常告警: 单用户成本异常 → 限流                              │    │
│  │  预测告警: 按当前趋势月底超支 → 自动降级                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 成本追踪实现

```python
from dataclasses import dataclass
from datetime import datetime
import prometheus_client

@dataclass
class CostTracker:
    """AI应用成本追踪器"""
    
    # Prometheus指标
    request_cost = prometheus_client.Counter(
        'ai_request_cost_dollars',
        'Cost per AI request in dollars',
        ['model', 'task_type', 'user_id']
    )
    
    token_usage = prometheus_client.Counter(
        'ai_token_usage_total',
        'Total tokens used',
        ['model', 'direction']  # direction: input/output
    )
    
    def track(self, request, response, model: str):
        """追踪单次请求的成本"""
        
        # 1. 计算token成本
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # 2. 按模型定价计算
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        # 3. 记录指标
        self.request_cost.labels(
            model=model,
            task_type=request.task_type,
            user_id=request.user_id
        ).inc(cost)
        
        self.token_usage.labels(model=model, direction="input").inc(input_tokens)
        self.token_usage.labels(model=model, direction="output").inc(output_tokens)
        
        return cost
    
    def _calculate_cost(self, model, input_tokens, output_tokens):
        """按模型定价计算成本"""
        
        PRICING = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-opus": {"input": 15.00, "output": 75.00},
            "qwen-7b-edge": {"input": 0.0, "output": 0.0},  # 自部署
        }
        
        pricing = PRICING.get(model, PRICING["gpt-4o"])
        cost = (input_tokens * pricing["input"] + 
                output_tokens * pricing["output"]) / 1_000_000
        
        return cost
```

## 九、综合实战：成本优化效果

### 9.1 某电商平台AI客服系统优化案例

```
┌─────────────────────────────────────────────────────────────────────┐
│                    优化前后对比                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                    优化前          优化后           改善               │
│  ──────────────────────────────────────────────────────────────     │
│  日均请求量          100万          100万           -                 │
│  平均模型            GPT-4o         混合路由         -                 │
│  月Token消耗        420亿          180亿           -57%              │
│  月API成本          ¥126万         ¥42万           -67%              │
│  月GPU成本          ¥0             ¥8万            +¥8万             │
│  月总成本           ¥126万         ¥50万           -60%              │
│  平均延迟(P50)      1.8s           0.9s            -50%              │
│  用户满意度          4.1/5          4.4/5           +7.3%             │
│  缓存命中率         0%             35%             +35%              │
│  年化节省           -              -               ¥912万            │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 各模式贡献度

| 优化模式 | 成本节省 | 实施难度 | ROI |
|---------|---------|---------|-----|
| 智能模型路由 | 35% | ⭐⭐ | 极高 |
| 语义缓存 | 20% | ⭐⭐⭐ | 高 |
| Prompt压缩 | 15% | ⭐ | 高 |
| 批处理优化 | 10% | ⭐⭐ | 中 |
| 模型蒸馏 | 25% | ⭐⭐⭐⭐ | 长期高 |
| **综合** | **60%** | - | **极高** |

## 十、架构选型指南

### 10.1 按应用阶段选择

```
┌─────────────────────────────────────────────────────────────────────┐
│                    不同阶段的优化策略                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MVP阶段（日请求 < 10万）                                             │
│  ├─ 必做: Prompt优化（零成本）                                        │
│  ├─ 推荐: 简单规则路由                                                │
│  └─ 暂缓: 语义缓存、模型蒸馏                                          │
│                                                                      │
│  增长期（日请求 10-100万）                                            │
│  ├─ 必做: 智能模型路由                                                │
│  ├─ 推荐: 语义缓存 + Prompt压缩                                      │
│  └─ 考虑: 批处理优化                                                  │
│                                                                      │
│  成熟期（日请求 > 100万）                                             │
│  ├─ 必做: 全部优化模式                                                │
│  ├─ 推荐: 模型蒸馏 + 边缘部署                                        │
│  └─ 高级: 成本监控与自动调优                                          │
│                                                                      │
│  规模化（日请求 > 1000万）                                            │
│  ├─ 必做: 自建推理集群                                                │
│  ├─ 推荐: 定制化模型 + 全链路优化                                      │
│  └─ 战略: 模型训练自主化                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 快速评估清单

在实施成本优化前，先回答以下问题：

- [ ] 你是否清楚当前的成本结构？（哪些请求最贵）
- [ ] 你是否有请求分类的数据？（哪些是简单任务，哪些是复杂任务）
- [ ] 你的请求中是否有重复模式？（适合语义缓存）
- [ ] 你的Prompt是否有压缩空间？（通常有30-50%）
- [ ] 你是否有可以批处理的任务？（非实时场景）
- [ ] 你是否有自部署GPU的条件？（模型蒸馏的前提）

## 十一、总结

AI应用的成本优化不是单一技术问题，而是一个**系统性架构工程**。

**核心要点**：

1. **按需匹配**：不是所有请求都需要最强模型，智能路由是最高ROI的优化
2. **消除重复**：语义缓存能将重复查询的成本降到接近零
3. **压缩输入**：Prompt和文档的压缩是投入产出比最高的优化
4. **分级处理**：实时、交互、批处理、离线四级策略覆盖所有场景
5. **持续监控**：没有度量就没有优化，成本可观测性是基础

**行动建议**：

```
优先级排序：
1. 先做Prompt优化（0成本，立竿见影）
2. 再做智能模型路由（最大收益）
3. 然后建设语义缓存（长期收益）
4. 最后考虑模型蒸馏（需要投入但ROI高）
```

记住：**成本优化的目标不是省钱，而是用同样的预算做更多的事情**。将节省的成本投入到更好的模型、更多的场景、更好的用户体验上，才是AI应用成本优化的真正价值。
