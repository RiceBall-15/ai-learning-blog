---
title: "AI模型推理成本优化工程实践：批处理、缓存与路由的组合拳"
description: "从Token计费到架构层面，系统性拆解LLM推理成本优化的六大工程策略，附真实场景的ROI分析与落地代码"
date: 2026-06-01
author: "RiceBall-15"
category: "engineering"
subCategory: "infra"
tags: ["LLM推理", "成本优化", "推理架构", "KV Cache", "Prompt Caching", "模型路由", "工程实践"]
draft: false
---

# AI模型推理成本优化工程实践：批处理、缓存与路由的组合拳

> 大模型API的Token计费让很多团队的月账单直逼服务器成本。但你是否真正分析过，你的Token花在了哪里？哪些可以避免？本文从实际生产经验出发，系统性拆解LLM推理成本优化的六大工程策略，帮助团队在不牺牲质量的前提下，将推理成本降低40%-70%。

---

## 一、成本账本：你的Token到底花在哪了？

### 1.1 成本构成拆解

在优化之前，先搞清楚钱花在了哪里。以一个典型的RAG应用为例，单次请求的Token消耗可以拆分为：

```
┌─────────────────────────────────────────────────┐
│              单次请求Token消耗分布                 │
├──────────────────┬──────────┬───────────────────┤
│     组成部分      │ 比例     │     可优化空间     │
├──────────────────┼──────────┼───────────────────┤
│ System Prompt    │ 15-25%   │ ★★★★★ (重复)     │
│ 检索上下文        │ 30-45%   │ ★★★★ (可精简)    │
│ 对话历史          │ 10-20%   │ ★★★ (可压缩)     │
│ 用户输入          │ 5-10%    │ ★ (不可控)        │
│ 模型输出          │ 20-35%   │ ★★★ (可控制长度)  │
└──────────────────┴──────────┴───────────────────┘
```

### 1.2 真实案例：一个月烧掉3万的RAG应用

某知识问答SaaS应用，日均10万次请求，使用GPT-4o模型：

| 指标 | 优化前 | 问题分析 |
|------|--------|----------|
| 平均输入Token | 4,200 | System Prompt 800+ 历史1500+ 检索1800+ |
| 平均输出Token | 680 | 大量冗余的格式化输出 |
| 日均Token消耗 | 4.88亿 | — |
| 月度API费用 | ¥32,400 | 输入¥19,440 + 输出¥12,960 |

**核心发现：** 800Token的System Prompt每天重复发送10万次，占总输入Token的19%。这是一笔完全可以大幅削减的开销。

---

## 二、策略一：Prompt Cache — 被忽视的降本利器

### 2.1 原理与适用场景

Prompt Cache（也叫Prefix Cache）是目前各大API厂商都支持但使用率极低的优化手段。其核心思想是：**对于相同的前缀Token，模型服务端可以复用KV Cache，避免重复计算。**

```
┌──────────────────────────────────────────┐
│           Prompt Cache 工作原理           │
├──────────────────────────────────────────┤
│                                          │
│  请求1: [System Prompt + Query A]        │
│  请求2: [System Prompt + Query B]        │
│  请求3: [System Prompt + Query C]        │
│           │                              │
│           ▼                              │
│  ┌─────────────────┐                     │
│  │  System Prompt   │ ← KV Cache复用     │
│  │  (固定前缀)      │   缓存命中时       │
│  ├─────────────────┤   计算成本降90%+    │
│  │  Query A/B/C    │ ← 仅增量计算        │
│  └─────────────────┘                     │
│                                          │
└──────────────────────────────────────────┘
```

### 2.2 各厂商支持情况

| 厂商 | 缓存机制 | 缓存命中折扣 | 最小前缀长度 | 生效方式 |
|------|----------|-------------|-------------|----------|
| OpenAI | Automatic Caching | 50% off | 1024 tokens | 自动（无需配置） |
| Anthropic | Prompt Caching | 90% off | 1024 tokens | 需标记缓存断点 |
| Google | Context Caching | 75% off | 32K tokens | 需显式创建缓存 |
| DeepSeek | Context Caching | 90% off | — | 自动 |
| 阿里云百炼 | 智能缓存 | 90% off | — | 自动 |

### 2.3 实战：Anthropic Prompt Cache的最佳实践

```python
import anthropic

client = anthropic.Anthropic()

# 关键：将固定内容标记为缓存断点
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """你是一个专业的法律顾问助手。以下是公司的规章制度：
                
                [2000字的规章制度内容...]
                
                请根据以上规章制度回答用户的问题。""",
                "cache_control": {"type": "ephemeral"}  # 标记为可缓存
            },
            {
                "type": "text", 
                "text": "员工出差期间的加班如何计算？"
            }
        ]
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=messages
)

# 缓存命中时的计费:
# - 缓存写入: 标准价格
# - 缓存读取: 仅10%价格 (节省90%)
```

### 2.4 收益测算

沿用前面的RAG案例，System Prompt 800 Token，假设缓存命中率80%：

| 指标 | 优化前 | 使用Cache后 |
|------|--------|------------|
| System Prompt日成本 | ¥5,760 | ¥1,152 (减少80%) |
| 月度节省 | — | ¥36,864 |
| **投资回报** | — | **仅需配置cache_control** |

---

## 三、策略二：语义缓存 — 同类问题零成本响应

### 3.1 核心思想

Prompt Cache解决的是"相同前缀"的复用，而**语义缓存**解决的是"相似问题"的复用。用户问"Python如何读取JSON"和"Python读取JSON文件的方法"本质上是同一个问题，没必要重复调用大模型。

```
┌──────────────────────────────────────────────┐
│              语义缓存架构                      │
├──────────────────────────────────────────────┤
│                                              │
│  用户Query → Embedding → 向量检索            │
│                              │               │
│                    ┌─────────┴─────────┐     │
│                    │  相似度 > 阈值?     │     │
│                    └─────────┬─────────┘     │
│                         是/否                │
│                    ┌────┴────┐               │
│                    ▼         ▼               │
│               返回缓存    调用LLM            │
│               (0成本)    (正常计费)           │
│                          │                   │
│                          ▼                   │
│                    存入缓存库                 │
│                                              │
└──────────────────────────────────────────────┘
```

### 3.2 实现方案

```python
import numpy as np
from openai import OpenAI
import redis
import json
import hashlib

client = OpenAI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 缓存配置
CACHE_THRESHOLD = 0.92  # 余弦相似度阈值
CACHE_TTL = 86400 * 7   # 缓存7天过期

def get_embedding(text: str) -> list[float]:
    """获取文本的embedding向量"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度"""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_semantic_cache(query: str) -> dict | None:
    """检查语义缓存"""
    query_embedding = get_embedding(query)
    
    # 从Redis获取所有缓存的embedding
    cached_keys = redis_client.keys("semantic_cache:*")
    
    best_match = None
    best_score = 0
    
    for key in cached_keys:
        cached = json.loads(redis_client.get(key))
        similarity = cosine_similarity(query_embedding, cached['embedding'])
        
        if similarity > best_score and similarity >= CACHE_THRESHOLD:
            best_score = similarity
            best_match = cached
    
    if best_match:
        # 更新访问计数（用于LRU淘汰）
        redis_client.incr(f"cache_hits:{best_match['query_hash']}")
        return best_match['response']
    
    return None

def set_semantic_cache(query: str, response: str):
    """写入语义缓存"""
    embedding = get_embedding(query)
    query_hash = hashlib.md5(query.encode()).hexdigest()
    
    cache_entry = {
        'query': query,
        'query_hash': query_hash,
        'embedding': embedding,
        'response': response,
        'created_at': '2026-06-01'
    }
    
    redis_client.setex(
        f"semantic_cache:{query_hash}",
        CACHE_TTL,
        json.dumps(cache_entry, ensure_ascii=False)
    )
```

### 3.3 缓存命中率的影响因素

```
┌─────────────────────────────────────────────────┐
│        语义缓存命中率 vs 关键参数关系              │
├──────────────────┬──────────────────────────────┤
│     参数          │     影响                      │
├──────────────────┼──────────────────────────────┤
│ 相似度阈值        │ 阈值越高 → 命中率↓ 但准确率↑  │
│                  │ 推荐: 0.90-0.95              │
├──────────────────┼──────────────────────────────┤
│ Embedding模型    │ 模型越强 → 相似度计算越准      │
│                  │ 推荐: text-embedding-3-small  │
├──────────────────┼──────────────────────────────┤
│ 缓存TTL          │ TTL越长 → 命中率↑ 但过期风险↑ │
│                  │ 推荐: 3-7天                   │
├──────────────────┼──────────────────────────────┤
│ Query标准化      │ 去除停用词/大小写 → 命中率↑   │
│                  │ 推荐: 做预处理                │
├──────────────────┼──────────────────────────────┤
│ 热点集中度        │ 问题越集中 → 命中率越高       │
│                  │ 客服场景通常>40%              │
└──────────────────┴──────────────────────────────┘
```

### 3.4 适用场景评估

| 场景 | 适用度 | 典型命中率 | 注意事项 |
|------|--------|-----------|----------|
| 客服问答 | ★★★★★ | 40-60% | 问题高度重复，效果最佳 |
| 知识库检索 | ★★★★ | 25-40% | 需配合检索结果缓存 |
| 代码生成 | ★★★ | 15-25% | 需求差异较大 |
| 创意写作 | ★ | <5% | 几乎没有重复问题 |
| 数据分析 | ★★ | 5-15% | 取决于数据是否重复 |

---

## 四、策略三：请求批处理 — 把零散订单变成批发采购

### 4.1 为什么批处理能省钱？

模型厂商的GPU资源是按峰值配置的。当大量请求并发涌入时，厂商需要更多的GPU来应对。而**批处理（Batching）**让请求排队合并处理，提高了GPU利用率，厂商自然愿意给折扣。

```
┌──────────────────────────────────────────┐
│           两种处理模式对比                  │
├──────────────────────────────────────────┤
│                                          │
│  实时模式 (Real-time):                    │
│  请求A → [GPU 1] → 结果A  (30ms)        │
│  请求B → [GPU 2] → 结果B  (30ms)        │
│  请求C → [GPU 3] → 结果C  (30ms)        │
│  GPU利用率: ~30%  |  成本: 标准价         │
│                                          │
│  批处理模式 (Batching):                   │
│  请求A ─┐                               │
│  请求B ─┼→ [GPU 1+2] → A,B,C  (200ms)  │
│  请求C ─┘                               │
│  GPU利用率: ~85%  |  成本: 50%折扣        │
│                                          │
└──────────────────────────────────────────┘
```

### 4.2 各厂商的批处理方案

| 厂商 | 批处理产品 | 折扣力度 | 最大延迟 | 适用场景 |
|------|-----------|---------|---------|----------|
| OpenAI | Batch API | 50% off | 24小时 | 离线批量处理 |
| Anthropic | Message Batches | 50% off | 24小时 | 离线批量处理 |
| Google | Batch Prediction | 50% off | 数小时 | 离线批量处理 |
| DeepSeek | — | — | — | 无专门批处理 |
| Azure | Managed Batch | 30-50% off | 可配置 | 企业级离线任务 |

### 4.3 实战：OpenAI Batch API的使用

```python
import json
import time
import openai

client = openai.OpenAI()

def create_batch_job(input_file_path: str) -> str:
    """创建批处理任务"""
    # 1. 上传输入文件
    with open(input_file_path, 'rb') as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    # 2. 创建批处理任务
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "夜间批量分析用户反馈"
        }
    )
    
    return batch.id

def wait_for_completion(batch_id: str, poll_interval: int = 60):
    """轮询等待批处理完成"""
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"状态: {batch.status} | 完成: {batch.request_counts.completed}/{batch.request_counts.total}")
        
        if batch.status == "completed":
            return batch.output_file_id
        elif batch.status == "failed":
            raise Exception(f"批处理失败: {batch.errors}")
        
        time.sleep(poll_interval)

# 输入文件格式 (JSONL):
# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", 
#  "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "..."}]}}
# {"custom_id": "request-2", ...}
```

### 4.4 适用场景矩阵

```
┌─────────────────────────────────────────────────┐
│          批处理 vs 实时 决策矩阵                   │
├─────────────┬──────────┬────────────────────────┤
│    场景      │ 推荐模式  │        原因             │
├─────────────┼──────────┼────────────────────────┤
│ 用户即时问答  │ 实时     │ 需要毫秒级响应          │
│ 数据标注     │ 批处理   │ 可等待数小时             │
│ 报告生成     │ 批处理   │ 可以过夜出结果           │
│ 内容审核     │ 批处理   │ 准实时即可(分钟级)       │
│ 摘要生成     │ 批处理   │ 离线处理历史数据         │
│ 翻译任务     │ 批处理   │ 文档翻译不需要即时响应    │
│ 客服对话     │ 实时     │ 必须即时回复             │
│ 代码审查     │ 混合     │ 关键路径实时，其余批处理   │
└─────────────┴──────────┴────────────────────────┘
```

---

## 五、策略四：模型路由 — 用对的模型处理对的问题

### 5.1 核心思想

不是所有问题都需要最强的模型。简单问题用小模型，复杂问题用大模型——这就是**模型路由**的核心思想。

```
┌──────────────────────────────────────────────┐
│              模型路由架构                      │
├──────────────────────────────────────────────┤
│                                              │
│  用户请求                                     │
│     │                                        │
│     ▼                                        │
│  ┌─────────────┐                             │
│  │ 分类器/规则   │ ← 判断问题复杂度             │
│  └──────┬──────┘                             │
│         │                                    │
│    ┌────┴────┬──────────┐                    │
│    ▼         ▼          ▼                    │
│  简单问题   中等问题    复杂问题               │
│    │         │          │                    │
│    ▼         ▼          ▼                    │
│  GPT-4o-mini GPT-4o    GPT-4o               │
│  ¥0.15/M    ¥2.5/M    ¥2.5/M              │
│  (input)    (input)    (input)              │
│                                              │
│  效果: 70%请求走小模型 → 成本降低60%+         │
│                                              │
└──────────────────────────────────────────────┘
```

### 5.2 路由策略设计

```python
from enum import Enum
from dataclasses import dataclass

class Complexity(Enum):
    SIMPLE = "simple"      # 简单查询、格式化、翻译
    MEDIUM = "medium"      # 一般问答、摘要、分类
    COMPLEX = "complex"    # 推理、代码生成、分析

@dataclass
class ModelConfig:
    model: str
    input_price: float   # $/M tokens
    output_price: float
    max_tokens: int

# 模型配置表
MODEL_REGISTRY = {
    Complexity.SIMPLE: ModelConfig(
        model="gpt-4o-mini",
        input_price=0.15,
        output_price=0.60,
        max_tokens=4096
    ),
    Complexity.MEDIUM: ModelConfig(
        model="gpt-4o",
        input_price=2.50,
        output_price=10.00,
        max_tokens=8192
    ),
    Complexity.COMPLEX: ModelConfig(
        model="o3",
        input_price=10.00,
        output_price=40.00,
        max_tokens=32768
    ),
}

def classify_complexity(query: str, context: dict) -> Complexity:
    """基于规则+启发式的复杂度分类"""
    
    # 规则1: 长度启发
    if len(query) < 50 and not any(kw in query for kw in ['分析', '对比', '设计', '解释', '推理']):
        return Complexity.SIMPLE
    
    # 规则2: 关键词匹配
    complex_keywords = ['推理', '证明', '设计架构', '优化算法', '代码审查', 'debug']
    if any(kw in query for kw in complex_keywords):
        return Complexity.COMPLEX
    
    # 规则3: 上下文长度
    if context.get('history_length', 0) > 10 or context.get('retrieved_docs', 0) > 5:
        return Complexity.COMPLEX
    
    # 规则4: 嵌入分类器（可选，更准确）
    # embedding_score = classifier.predict(query)
    # if embedding_score > 0.8: return Complexity.COMPLEX
    
    return Complexity.MEDIUM

def route_request(query: str, context: dict = None) -> ModelConfig:
    """路由请求到合适的模型"""
    context = context or {}
    complexity = classify_complexity(query, context)
    return MODEL_REGISTRY[complexity]
```

### 5.3 进阶：基于置信度的动态路由

```python
class ConfidenceBasedRouter:
    """基于置信度的两级路由：先用小模型，不确定时升级"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.fallback_model = MODEL_REGISTRY[Complexity.COMPLEX]
    
    def route_with_fallback(self, query: str, context: dict) -> tuple[ModelConfig, str]:
        """带降级的路由"""
        initial_model = route_request(query, context)
        
        # 如果初始路由就是高复杂度，直接返回
        if initial_model == self.fallback_model:
            return initial_model, "direct"
        
        # 否则先用初始模型，检查输出置信度
        response = call_llm(initial_model, query)
        
        # 置信度检查（基于logprobs或自定义评分）
        confidence = self._check_confidence(response)
        
        if confidence >= self.confidence_threshold:
            return initial_model, "initial_success"
        else:
            # 置信度不够，升级到更强大的模型
            return self.fallback_model, "upgraded"
    
    def _check_confidence(self, response: dict) -> float:
        """检查模型输出的置信度"""
        # 方案1: 使用logprobs
        if 'logprobs' in response:
            avg_logprob = sum(response['logprobs']) / len(response['logprobs'])
            return sigmoid(avg_logprob)
        
        # 方案2: 使用模型自我评估
        eval_prompt = f"请评估以下回答的置信度(0-1): {response['content']}"
        eval_score = call_llm(MODEL_REGISTRY[Complexity.SIMPLE], eval_prompt)
        return float(eval_score)
        
        # 方案3: 基于规则的启发式
        # 例如：回答中包含"不确定"、"可能"等词汇时降低置信度
```

### 5.4 路由效果评估

```
┌─────────────────────────────────────────────────────┐
│           模型路由优化效果对比                         │
├───────────────┬──────────┬──────────┬───────────────┤
│     指标       │ 无路由    │ 简单路由  │ 置信度路由     │
├───────────────┼──────────┼──────────┼───────────────┤
│ 平均成本/请求  │ $0.012   │ $0.005   │ $0.006        │
│ 质量达标率     │ 100%     │ 95%      │ 98%           │
│ 成本降低       │ —        │ 58%      │ 50%           │
│ 延迟(P50)     │ 800ms    │ 400ms    │ 450ms         │
│ 延迟(P99)     │ 2500ms   │ 1200ms   │ 1800ms        │
│ 实现复杂度     │ —        │ ★★       │ ★★★★         │
└───────────────┴──────────┴──────────┴───────────────┘
```

---

## 六、策略五：输出控制 — 从源头减少Token浪费

### 6.1 常见的输出Token浪费模式

```
┌─────────────────────────────────────────────────┐
│           输出Token浪费的五大来源                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. 冗余格式化 (占15-25%)                        │
│     "根据您的要求，以下是分析结果：\n\n"           │
│     → 直接输出结果即可                           │
│                                                 │
│  2. 过度解释 (占10-20%)                          │
│     模型主动添加的"顺便说一下..."                 │
│     → 用 system prompt 约束                     │
│                                                 │
│  3. 重复总结 (占5-15%)                           │
│     "综上所述，..." + "总结一下..."               │
│     → max_tokens 硬限制                          │
│                                                 │
│  4. 不必要的JSON格式化 (占10-20%)                │
│     大段的 JSON schema 定义                      │
│     → 使用 Structured Output                    │
│                                                 │
│  5. 多轮对话历史累积 (占20-40%)                  │
│     5轮对话后历史占大量Token                      │
│     → 滑动窗口 + 摘要压缩                       │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 6.2 输出控制实战

```python
# 策略1: 通过System Prompt控制输出风格
SYSTEM_PROMPT_COST_OPTIMIZED = """你是一个简洁的助手。规则：
1. 直接回答问题，不要寒暄
2. 不要说"根据您的要求"、"以下是分析"等冗余开头
3. 使用列表而非长段落
4. 不要在末尾重复总结
5. 如果用户没要求解释，就只给答案"""

# 策略2: 控制max_tokens
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": query}],
    max_tokens=500,  # 硬限制输出长度
    temperature=0.3  # 降低随机性，减少冗余
)

# 策略3: 对话历史滑动窗口
def truncate_history(messages: list[dict], max_history_tokens: int = 2000) -> list[dict]:
    """保留system prompt + 最近的对话，丢弃中间部分"""
    system_msg = messages[0]  # system prompt
    conversation = messages[1:]
    
    truncated = [system_msg]
    current_tokens = 0
    
    # 从最新的消息往前保留
    for msg in reversed(conversation):
        msg_tokens = estimate_tokens(msg['content'])
        if current_tokens + msg_tokens > max_history_tokens:
            # 插入摘要标记
            truncated.insert(1, {
                "role": "system",
                "content": "[之前的对话已被压缩]"
            })
            break
        truncated.insert(1, msg)
        current_tokens += msg_tokens
    
    return truncated

# 策略4: 对话摘要压缩
def summarize_history(messages: list[dict], client) -> list[dict]:
    """用小模型压缩长对话历史"""
    history_text = "\n".join([
        f"{m['role']}: {m['content']}" for m in messages[1:]
    ])
    
    summary = client.chat.completions.create(
        model="gpt-4o-mini",  # 用小模型做摘要
        messages=[{
            "role": "system",
            "content": "用50字以内概括以下对话的关键信息："
        }, {
            "role": "user",
            "content": history_text
        }],
        max_tokens=100
    )
    
    return [
        messages[0],  # 保留system prompt
        {"role": "system", "content": f"之前的对话摘要：{summary.choices[0].message.content}"}
    ]
```

---

## 七、策略六：混合部署 — 自建推理的成本拐点

### 7.1 何时该考虑自建？

```
┌─────────────────────────────────────────────────┐
│        API调用 vs 自建推理 决策框架                │
├─────────────────────────────────────────────────┤
│                                                 │
│  月度Token消耗                                  │
│  ┌─────────┬──────────────────────────┐        │
│  │ < 1亿    │ 纯API，成本优化为主       │        │
│  │ 1-10亿   │ 混合方案，部分自建        │        │
│  │ > 10亿   │ 自建为主，API为辅         │        │
│  └─────────┴──────────────────────────┘        │
│                                                 │
│  关键考量因素:                                   │
│  ├── 延迟要求: P99 > 500ms → 考虑自建           │
│  ├── 数据合规: 敏感数据 → 必须自建               │
│  ├── 模型定制: 需要微调 → 必须自建               │
│  ├── 团队能力: 有ML运维团队 → 可以自建           │
│  └── 成本预测: 12个月TCO对比                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 7.2 自建推理的成本模型

```
┌─────────────────────────────────────────────────┐
│         自建推理月度成本估算 (基于H100)            │
├─────────────────────────────────────────────────┤
│                                                 │
│  基础设施成本:                                   │
│  ├── GPU租用: 1×H100 = ¥15,000/月              │
│  ├── 服务器: ¥2,000/月                          │
│  ├── 网络: ¥500/月                              │
│  └── 运维人力: ¥8,000/月 (分摊)                 │
│  合计: ¥25,500/月                               │
│                                                 │
│  推理能力:                                       │
│  ├── Qwen-72B (INT4量化): ~100 tokens/s        │
│  ├── 月产能: ~260亿 tokens                      │
│  └── 单位成本: ¥0.098/百万tokens               │
│                                                 │
│  vs API成本:                                     │
│  ├── GPT-4o-mini: ¥1.08/百万tokens (input)     │
│  ├── 节省比例: ~91%                             │
│  └── 回本周期: 月消耗 > 3亿tokens               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 7.3 混合部署架构

```
┌──────────────────────────────────────────────────┐
│              混合推理架构                          │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌───────────────┐                               │
│  │  请求分类器    │                               │
│  └───────┬───────┘                               │
│          │                                       │
│    ┌─────┴──────┐                                │
│    ▼            ▼                                │
│  本地推理      API调用                             │
│  (自建集群)    (云端)                              │
│    │            │                                │
│    ▼            ▼                                │
│  ┌─────────────────────┐                         │
│  │  路由策略:           │                         │
│  │  - 普通问答→本地     │                         │
│  │  - 敏感数据→本地     │                         │
│  │  - 复杂推理→API      │                         │
│  │  - 本地GPU满载→API   │                         │
│  │  - 模型切换→API      │                         │
│  └─────────────────────┘                         │
│                                                  │
│  监控指标:                                        │
│  ├── 本地/远程请求比例                            │
│  ├── 本地推理延迟 vs API延迟                      │
│  ├── GPU利用率                                    │
│  └── 月度成本对比                                 │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 八、组合拳：全链路优化实战

### 8.1 真实案例：将月成本从3.2万降到8千

回到开头的RAG应用案例，综合运用以上策略：

```
┌─────────────────────────────────────────────────────┐
│          全链路优化效果对比                            │
├──────────────┬──────────┬──────────┬────────────────┤
│    策略       │ 优化前    │ 优化后    │   节省         │
├──────────────┼──────────┼──────────┼────────────────┤
│ Prompt Cache │ ¥19,440  │ ¥3,888   │ ¥15,552 (80%) │
│ 语义缓存      │ ¥0       │ -¥8,000  │ ¥8,000 (免费) │
│ 输出控制      │ ¥12,960  │ ¥6,480   │ ¥6,480 (50%)  │
│ 模型路由      │ ¥0       │ -¥4,000  │ ¥4,000 (可选) │
├──────────────┼──────────┼──────────┼────────────────┤
│ 总计         │ ¥32,400  │ ¥8,368   │ ¥24,032 (74%) │
└──────────────┴──────────┴──────────┴────────────────┘
```

### 8.2 实施优先级建议

```
┌──────────────────────────────────────────────────┐
│           优化策略实施优先级                        │
├──────┬──────────┬───────────────────────────────┤
│ 优先级│   策略    │        实施难度/收益           │
├──────┼──────────┼───────────────────────────────┤
│  P0  │ Prompt   │ 难度: ★   收益: ★★★★★        │
│      │ Cache    │ 1天完成，立竿见影               │
├──────┼──────────┼───────────────────────────────┤
│  P1  │ 输出控制  │ 难度: ★★  收益: ★★★★         │
│      │          │ 1-2天，修改Prompt即可          │
├──────┼──────────┼───────────────────────────────┤
│  P2  │ 批处理   │ 难度: ★★★ 收益: ★★★          │
│      │          │ 2-3天，需要改造调用链           │
├──────┼──────────┼───────────────────────────────┤
│  P3  │ 语义缓存  │ 难度: ★★★★ 收益: ★★★★       │
│      │          │ 1周，需要Redis+向量检索         │
├──────┼──────────┼───────────────────────────────┤
│  P4  │ 模型路由  │ 难度: ★★★★ 收益: ★★★★★     │
│      │          │ 1-2周，需要分类器+评估体系      │
├──────┼──────────┼───────────────────────────────┤
│  P5  │ 混合部署  │ 难度: ★★★★★ 收益: ★★★★★   │
│      │          │ 1月+，需要ML运维能力            │
└──────┴──────────┴───────────────────────────────┘
```

---

## 九、监控与持续优化

### 9.1 成本监控仪表盘核心指标

```python
# 核心监控指标定义
COST_METRICS = {
    "per_request": {
        "avg_input_tokens": "平均输入Token数",
        "avg_output_tokens": "平均输出Token数",
        "avg_total_tokens": "平均总Token数",
        "cost_per_request": "单次请求成本",
    },
    "cache": {
        "prompt_cache_hit_rate": "Prompt Cache命中率",
        "semantic_cache_hit_rate": "语义缓存命中率",
        "cache_savings": "缓存节省金额",
    },
    "routing": {
        "model_distribution": "各模型调用占比",
        "routing_accuracy": "路由准确率",
        "fallback_rate": "降级率",
    },
    "quality": {
        "user_satisfaction": "用户满意度",
        "response_accuracy": "回答准确率",
        "quality_degradation": "质量下降比例",
    }
}
```

### 9.2 成本-质量平衡的红线

```
┌──────────────────────────────────────────────────┐
│          成本优化的安全边界                         │
├──────────────────────────────────────────────────┤
│                                                  │
│  ✅ 安全区:                                       │
│  ├── 语义缓存命中率 < 60% (避免过度缓存)          │
│  ├── 模型路由降级率 < 10%                         │
│  ├── 用户满意度下降 < 5%                          │
│  └── 响应延迟增加 < 20%                           │
│                                                  │
│  ⚠️ 警告区:                                      │
│  ├── 语义缓存命中率 60-80%                        │
│  ├── 模型路由降级率 10-20%                        │
│  ├── 用户满意度下降 5-10%                         │
│  └── 响应延迟增加 20-50%                          │
│                                                  │
│  🚨 危险区:                                      │
│  ├── 语义缓存命中率 > 80% (可能返回错误答案)       │
│  ├── 模型路由降级率 > 20% (路由分类器失效)         │
│  ├── 用户满意度下降 > 10%                         │
│  └── 响应延迟增加 > 50%                           │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 十、总结与行动清单

### 10.1 一张图总结六大策略

```
┌──────────────────────────────────────────────────┐
│                                                  │
│          LLM推理成本优化全景图                     │
│                                                  │
│    输入侧          处理侧          输出侧         │
│  ┌────────┐    ┌──────────┐    ┌────────┐       │
│  │Prompt  │    │ 模型路由  │    │输出控制 │       │
│  │Cache   │    │ (选对模型)│    │(精简输出)│       │
│  │(复用前缀)│   └──────────┘    └────────┘       │
│  └────────┘                                     │
│  ┌────────┐    ┌──────────┐    ┌────────┐       │
│  │语义缓存 │    │ 批处理    │    │混合部署 │       │
│  │(复用问答)│   │(合并请求) │    │(自建推理)│       │
│  └────────┘    └──────────┘    └────────┘       │
│                                                  │
│  综合效果: 成本降低 40%-74%                       │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 10.2 快速行动清单

1. **今天就做**: 检查你的API调用是否开启了Prompt Cache，优化System Prompt的复用结构
2. **本周完成**: 实施输出控制，修改System Prompt减少冗余输出，设置合理的max_tokens
3. **本月规划**: 评估语义缓存和模型路由的投入产出比，选择1-2个策略试点
4. **季度目标**: 建立完整的成本监控体系，持续优化Token使用效率

> 成本优化不是一次性的工作，而是持续的过程。建立监控、设定红线、定期复盘——这才是工程化优化的正确打开方式。
