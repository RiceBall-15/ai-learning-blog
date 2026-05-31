---
title: "生产级 RAG 系统架构实战：从 Naive RAG 到 Advanced RAG 的演进之路"
description: "深入解析 RAG 系统从基础架构到生产级部署的完整演进，涵盖检索优化、重排序、混合检索、分块策略、评估体系等关键环节，附带可落地的架构方案。"
date: 2026-06-01
author: "RiceBall"
category: "framework"
subCategory: "rag"
tags: ["RAG", "检索增强生成", "向量数据库", "混合检索", "重排序", "分块策略"]
draft: false
---

# 生产级 RAG 系统架构实战：从 Naive RAG 到 Advanced RAG 的演进之路

## 引言：RAG 为什么这么难做好？

几乎所有接触 LLM 应用开发的团队都试过 RAG——把文档切块、向量化、检索、拼 prompt、生成回答。Demo 阶段看起来效果不错，一旦放到生产环境，各种问题接踵而来：

- **检索不到相关内容**：用户问的问题文档里明明有答案，但向量搜索就是检索不出来
- **检索到的内容不相关**：召回了大量相似但无用的文档块
- **回答质量不稳定**：同样的问题，有时回答很好，有时胡说八道
- **无法处理复杂查询**：需要多跳推理或跨文档关联的问题几乎无法处理

问题的根源在于：**Naive RAG 的架构假设过于理想化**。它假设分块是合理的、向量检索是充分的、LLM 能从有限上下文中推理出正确答案。在生产环境中，这些假设几乎都不成立。

本文将完整梳理 RAG 系统从 Naive 到 Advanced 的演进路径，每个阶段解决什么问题、引入什么技术、带来什么新的挑战，并给出生产级架构的落地方案。

---

## 一、Naive RAG：基础架构与核心问题

### 1.1 架构概览

Naive RAG 是最经典的 RAG 架构，流程非常直接：

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  文档加载  │ →  │  文本分块  │ →  │ 向量化    │ →  │ 向量数据库 │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                    ↓
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  生成回答  │ ←  │  Prompt   │ ←  │ 检索 TopK │ ←  │  用户查询  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 1.2 分块策略的选择

分块（Chunking）是 RAG 系统中最被低估的环节。分块策略直接影响检索质量和生成效果。

#### 常见分块策略对比

| 策略 | 实现方式 | 优点 | 缺点 | 适用场景 |
|------|----------|------|------|----------|
| **固定长度** | 按字符/token 数切分 | 实现简单，可预测 | 破坏语义边界 | 通用文本 |
| **递归分割** | 按段落→句子→字符递归切分 | 尊重语义结构 | 需要调参 | 结构化文档 |
| **语义分块** | 用 Embedding 检测语义边界 | 最大化语义完整性 | 计算开销大 | 高质量要求 |
| **文档结构** | 按 Markdown 标题/HTML 标签切分 | 保留文档结构 | 依赖文档格式 | 技术文档 |
| **滑动窗口** | 重叠窗口切分 | 减少边界信息丢失 | 存储冗余 | 对话/叙事文本 |

#### 生产环境的分块配置建议

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 推荐配置：递归分割 + 重叠
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,          # 每块约 512 tokens
    chunk_overlap=100,       # 块间重叠 100 tokens
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
    # 中文文档优先按段落和句号切分
)

# 分块时添加元数据
chunks = splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = i
    chunk.metadata["source_file"] = chunk.metadata.get("source", "unknown")
    chunk.metadata["chunk_id"] = f"{chunk.metadata['source_file']}_{i}"
```

### 1.3 Naive RAG 的三大核心问题

**问题 1：检索精度不足**

纯向量检索（Dense Retrieval）依赖 Embedding 模型的语义理解能力，但它有几个致命弱点：

- **关键词匹配弱**：用户搜索 "2024年Q3营收" 但向量检索可能匹配到 "2023年Q3营收"
- **精确查询失败**：搜索 "错误代码 E-5021" 但向量检索返回所有包含 "错误代码" 的文档
- **语义漂移**：概念相近但实际无关的文档被高分召回

**问题 2：上下文窗口利用低效**

把 Top-K 个文档块一股脑塞进 prompt，但：

- K 太小 → 可能遗漏关键信息
- K 太大 → 引入噪声，干扰 LLM 判断
- 相关文档之间可能矛盾，LLM 无法分辨

**问题 3：无法处理复杂查询**

简单事实查询（"公司成立于哪一年？"）效果尚可，但：

- 多跳推理（"A 公司收购 B 公司后，B 公司的 CEO 是谁？"）
- 聚合查询（"去年各部门的离职率分别是多少？"）
- 对比查询（"产品 A 和产品 B 的定价策略有什么区别？"）

---

## 二、Advanced RAG：针对性优化

### 2.1 架构演进概览

Advanced RAG 在 Naive RAG 的基础上增加了多个优化模块：

```
                    ┌─────────────────────┐
                    │    Query 理解层       │
                    │  (意图识别/查询改写)   │
                    └─────────┬───────────┘
                              ↓
┌──────────┐    ┌─────────────────────┐    ┌──────────┐
│  文档加载  │ →  │    混合检索引擎       │ ←  │  向量数据库 │
└──────────┘    │  (Dense + Sparse +   │    └──────────┘
                │   Hybrid + Rerank)   │
                └─────────┬───────────┘
                          ↓
                ┌─────────────────────┐
                │    上下文构建层       │
                │  (去重/压缩/排序)     │
                └─────────┬───────────┘
                          ↓
                ┌─────────────────────┐
                │    生成 + 验证层      │
                │  (Self-RAG/验证链)   │
                └─────────────────────┘
```

### 2.2 Query 理解与改写

生产环境中，用户的查询往往模糊、口语化或过于简短。Query 理解层的目的是**将用户查询转化为更适合检索的形式**。

#### 技术方案 1：查询扩展（Query Expansion）

用 LLM 生成多个查询变体，提高召回覆盖率：

```python
EXPANSION_PROMPT = """
你是一个查询优化专家。给定用户问题，请生成3个不同角度的搜索查询，用于从知识库中检索相关信息。

用户问题：{query}

请输出3个查询，每行一个：
"""

def expand_query(query: str, llm) -> list[str]:
    response = llm.generate(EXPANSION_PROMPT.format(query=query))
    expanded = response.strip().split("\n")
    return [query] + expanded  # 原始查询 + 扩展查询
```

#### 技术方案 2：查询分解（Multi-Query / Step-back）

对于复杂查询，将其分解为多个子问题：

```python
DECOMPOSE_PROMPT = """
将以下复杂问题分解为2-4个独立的子问题，每个子问题可以从知识库中直接检索到答案。

原始问题：{query}

子问题：
"""

def decompose_query(query: str, llm) -> list[str]:
    response = llm.generate(DECOMPOSE_PROMPT.format(query=query))
    sub_queries = response.strip().split("\n")
    return sub_queries
```

#### 技术方案 3：HyDE（Hypothetical Document Embeddings）

让 LLM 先生成一个"假想文档"，再用这个文档的 embedding 去检索真实文档：

```python
HYPOTHESIS_PROMPT = """
请写一段可能包含以下问题答案的文档内容（不需要准确，只需要合理）：

问题：{query}

假想文档：
"""

def hyde_search(query: str, llm, vector_store):
    # 1. 生成假想文档
    hypothesis = llm.generate(HYPOTHESIS_PROMPT.format(query=query))
    # 2. 用假想文档的 embedding 检索
    results = vector_store.similarity_search(hypothesis, k=10)
    return results
```

### 2.3 混合检索（Hybrid Retrieval）

这是 Advanced RAG 中**投入产出比最高**的优化手段。

#### 为什么需要混合检索？

| 检索方式 | 优势 | 励势 |
|----------|------|------|
| Dense (向量) | 语义理解强 | 精确匹配弱 |
| Sparse (BM25) | 关键词匹配强 | 语义理解弱 |
| Hybrid (混合) | 两者优势互补 | 需要权重调优 |

#### 实现方案

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

# Dense Retriever
vector_store = Chroma.from_documents(documents, embedding_model)
dense_retriever = vector_store.as_retriever(
    search_type="mmr",  # MMR 多样性检索
    search_kwargs={"k": 20, "fetch_k": 40}
)

# Sparse Retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(
    documents,
    preprocess_func=preprocess_for_chinese,  # 中文预处理
)
bm25_retriever.k = 20

# Hybrid Retriever (RRF 融合)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4],  # Dense 权重 0.6, Sparse 权重 0.4
)
```

#### Reciprocal Rank Fusion (RRF)

简单的分数融合（如加权求和）在不同检索器分数尺度不同时效果不好。RRF 是更鲁棒的融合策略：

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list], 
    k: int = 60
) -> list[tuple]:
    """
    Reciprocal Rank Fusion
    k: 常数，通常取 60
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item.metadata.get("chunk_id", str(item))
            if item_id not in scores:
                scores[item_id] = {"score": 0, "item": item}
            scores[item_id]["score"] += 1.0 / (k + rank)
    
    # 按融合分数排序
    sorted_results = sorted(
        scores.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    return [r["item"] for r in sorted_results]
```

### 2.4 重排序（Reranking）

检索阶段为了效率通常使用近似最近邻（ANN），精度有限。重排序阶段使用计算量更大但更精确的模型对候选文档重新排序。

#### Cross-Encoder Reranker

```python
from sentence_transformers import CrossEncoder

# 加载 Cross-Encoder 模型
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query: str, documents: list, top_k: int = 5):
    # 构造 query-document 对
    pairs = [(query, doc.page_content) for doc in documents]
    
    # 计算相关性分数
    scores = reranker.predict(pairs)
    
    # 按分数排序
    ranked = sorted(
        zip(documents, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return [doc for doc, score in ranked[:top_k]]
```

#### 重排序的位置选择

重排序应该在**最终送入 LLM 之前**执行，而不是在检索阶段：

```
检索阶段（高召回，低精度）:
  向量检索 Top-50 + BM25 Top-50 → RRF 融合 → Top-30
                                                                    ↓
重排序阶段（低召回，高精度）:
  Cross-Encoder 对 Top-30 重新打分 → Top-5
                                                                    ↓
生成阶段:
  Top-5 文档 + 查询 → LLM 生成回答
```

### 2.5 上下文压缩与优化

重排序后的 Top-K 文档直接塞进 prompt 仍然可能有问题：文档块之间可能有重复信息，或包含大量无关的上下文。

#### 文档压缩技术

```python
COMPRESS_PROMPT = """
请根据用户问题，从以下文档中提取最相关的信息，去除无关内容，生成简洁的上下文：

用户问题：{query}

文档内容：
{context}

请输出提取的关键信息：
"""

def compress_context(query: str, documents: list, llm):
    context = "\n\n".join([doc.page_content for doc in documents])
    compressed = llm.generate(
        COMPRESS_PROMPT.format(query=query, context=context)
    )
    return compressed
```

#### Lost-in-the-Middle 问题

研究表明，LLM 对上下文中间位置的信息关注度最低（U 型曲线）。优化策略：

```python
def position_aware_insertion(
    query: str, 
    documents: list
) -> str:
    """
    将最相关的文档放在上下文的开头和结尾，
    次相关的放在中间位置
    """
    if len(documents) <= 2:
        return "\n\n".join([doc.page_content for doc in documents])
    
    # 最相关的放首位，第二相关的放末位
    ordered = [documents[0], documents[-1]] + documents[1:-1]
    return "\n\n".join([doc.page_content for doc in ordered])
```

---

## 三、生产级 RAG 架构设计

### 3.1 完整架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     生产级 RAG 架构                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Query 理解层                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │ 意图识别  │  │ 查询改写  │  │ 查询分解/扩展     │   │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   检索层                               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │ Dense    │  │ Sparse   │  │ Knowledge Graph  │   │  │
│  │  │ Vector   │  │ BM25     │  │ 知识图谱检索      │   │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │  │
│  │              ↓           ↓            ↓               │  │
│  │         ┌──────────────────────────────┐              │  │
│  │         │     RRF / 分数融合            │              │  │
│  │         └──────────────┬───────────────┘              │  │
│  │                        ↓                              │  │
│  │         ┌──────────────────────────────┐              │  │
│  │         │     Cross-Encoder 重排序      │              │  │
│  │         └──────────────┬───────────────┘              │  │
│  └────────────────────────┼──────────────────────────────┘  │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 上下文构建层                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │ 去重/去噪 │  │ 上下文压缩 │  │ 位置优化          │   │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   生成层                               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │ Prompt   │  │ LLM 生成  │  │ 回答验证/引用     │   │  │
│  │  │ 模板引擎  │  │          │  │ 来源追溯          │   │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   监控层                               │  │
│  │  检索质量监控 | 生成质量监控 | 延迟监控 | 成本监控       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 向量数据库选型

| 特性 | Milvus | Qdrant | Weaviate | Chroma |
|------|--------|--------|----------|--------|
| 部署模式 | 分布式/单机 | 分布式/单机 | 分布式/单机 | 嵌入式 |
| 标量过滤 | ✅ 强 | ✅ 强 | ✅ 中 | ✅ 基础 |
| 混合检索 | ✅ (Sparse-BM25) | ✅ (Sparse) | ✅ (BM25) | ❌ |
| 多租户 | ✅ | ✅ | ✅ | ❌ |
| 持久化 | ✅ | ✅ | ✅ | ✅ |
| 性能 | 极高 | 高 | 中 | 低 |
| 适合规模 | 亿级向量 | 千万级向量 | 千万级向量 | 百万级以下 |

**生产环境推荐**：

- 大规模（>1000万向量）：**Milvus** 分布式部署
- 中等规模：**Qdrant**（性能好，API 友好）
- 快速原型：**Chroma**（嵌入式，零运维）

### 3.3 Embedding 模型选择

| 模型 | 维度 | 中文支持 | 检索性能 | 推理速度 | 推荐场景 |
|------|------|----------|----------|----------|----------|
| BGE-M3 | 1024 | ✅ 强 | 极高 | 中 | 生产环境首选 |
| text-embedding-3-large | 3072 | ✅ 强 | 极高 | 低 (API) | 有预算的团队 |
| GTE-Qwen2 | 1536 | ✅ 强 | 高 | 中 | 中文场景 |
| E5-Mistral-7B | 4096 | ✅ 中 | 极高 | 低 | 离线场景 |
| m3e-base | 768 | ✅ 强 | 中 | 高 | 资源受限 |

---

## 四、RAG 评估体系

### 4.1 评估维度

生产级 RAG 系统需要系统化的评估体系：

```
RAG 评估
├── 检索质量评估
│   ├── Recall@K：Top-K 中包含正确文档的比例
│   ├── MRR：正确文档的平均倒数排名
│   ├── nDCG：考虑排名位置的检索质量
│   └── Hit Rate：至少召回一个正确文档的查询比例
│
├── 生成质量评估
│   ├── Faithfulness：回答是否忠实于检索到的文档
│   ├── Relevance：回答是否与问题相关
│   ├── Completeness：回答是否完整覆盖了问题
│   └── Hallucination Rate：幻觉率（生成了文档中不存在的信息）
│
└── 端到端评估
    ├── Answer Correctness：最终答案的正确性
    ├── User Satisfaction：用户满意度（A/B 测试）
    └── Latency & Cost：延迟和成本
```

### 4.2 使用 RAGAS 进行评估

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 准备评估数据
eval_dataset = {
    "question": ["公司成立于哪一年？", "CEO 是谁？"],
    "answer": ["公司成立于 2018 年", "CEO 是张三"],
    "contexts": [
        ["公司于2018年3月在北京成立..."],
        ["张三自2020年起担任公司CEO..."],
    ],
    "ground_truth": ["2018年", "张三"],
}

# 运行评估
result = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,        # 回答是否忠实于上下文
        answer_relevancy,    # 回答与问题的相关性
        context_precision,   # 检索到的上下文的精确度
        context_recall,      # 检索到的上下文的召回率
    ],
)

print(result)
```

### 4.3 在线监控指标

```python
# 生产环境的监控指标
monitoring_config = {
    # 检索质量监控
    "retrieval_metrics": {
        "avg_retrieval_score": {
            "alert_threshold": 0.5,  # 平均检索分数低于 0.5 告警
            "description": "平均检索相关性分数"
        },
        "retrieval_latency_p99": {
            "alert_threshold_ms": 200,
            "description": "检索延迟 P99"
        },
        "empty_retrieval_rate": {
            "alert_threshold": 0.1,  # 空检索率超过 10% 告警
            "description": "检索返回空结果的比例"
        },
    },
    
    # 生成质量监控
    "generation_metrics": {
        "hallucination_rate": {
            "alert_threshold": 0.05,  # 幻觉率超过 5% 告警
            "description": "LLM 幻觉率"
        },
        "generation_latency_p99": {
            "alert_threshold_ms": 5000,
            "description": "生成延迟 P99"
        },
        "citation_accuracy": {
            "alert_threshold": 0.8,  # 引用准确率低于 80% 告警
            "description": "引用来源的准确率"
        },
    },
    
    # 成本监控
    "cost_metrics": {
        "avg_tokens_per_query": {
            "alert_threshold": 4000,
            "description": "每次查询的平均 token 消耗"
        },
        "embedding_cost_per_1k_queries": {
            "description": "每 1000 次查询的 embedding 成本"
        },
    },
}
```

---

## 五、实战案例：企业知识库 RAG 系统

### 5.1 需求分析

为某企业构建内部知识库问答系统，需求：

- **数据量**：10 万份文档（PDF、Word、Markdown），约 5 亿 tokens
- **查询量**：日均 5000 次查询
- **延迟要求**：P95 < 3 秒
- **准确率要求**：Answer Correctness > 85%

### 5.2 架构方案

```
┌─────────────────────────────────────────────────────────┐
│                    离线处理管道                           │
│                                                         │
│  文档解析 → 语义分块 → BGE-M3 向量化 → Milvus 入库       │
│      ↓                                                   │
│  BM25 索引构建（Elasticsearch）                          │
│      ↓                                                   │
│  元数据提取（作者/日期/部门/分类）                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    在线服务架构                           │
│                                                         │
│  Nginx → API Gateway → Query Understanding Service     │
│                              ↓                          │
│                    Hybrid Retrieval Service              │
│                    (Milvus + Elasticsearch)              │
│                              ↓                          │
│                    Reranking Service (GPU)               │
│                              ↓                          │
│                    Context Builder                       │
│                              ↓                          │
│                    LLM Service (vLLM)                    │
│                              ↓                          │
│                    Response + Citation Service           │
└─────────────────────────────────────────────────────────┘
```

### 5.3 关键配置

```yaml
# 分块配置
chunking:
  strategy: "recursive"
  chunk_size: 512
  chunk_overlap: 100
  separators: ["\n\n", "\n", "。", "；", " "]
  min_chunk_size: 100  # 最小块大小，避免碎片

# 检索配置
retrieval:
  dense:
    top_k: 50
    similarity_threshold: 0.3
    mmr_lambda: 0.7  # MMR 多样性参数
  sparse:
    top_k: 50
    bm25_k1: 1.5
    bm25_b: 0.75
  fusion:
    method: "rrf"
    rrf_k: 60
    weights: [0.6, 0.4]  # [dense, sparse]
  rerank:
    model: "BAAI/bge-reranker-v2-m3"
    top_k: 5  # 最终送入 LLM 的文档数

# 生成配置
generation:
  model: "Qwen2.5-72B-Instruct"
  max_tokens: 2048
  temperature: 0.1  # 低温度，提高确定性
  top_p: 0.9
  system_prompt: |
    你是一个企业知识库问答助手。请基于提供的文档内容回答问题。
    如果文档中没有相关信息，请明确说明"根据现有知识库，暂未找到相关信息"。
    回答时请标注信息来源（文档名称）。
```

### 5.4 性能优化经验

**优化 1：向量检索预过滤**

```python
# 利用标量字段预过滤，缩小向量检索范围
results = vector_store.similarity_search(
    query_embedding,
    k=50,
    filter={
        "$and": [
            {"department": {"$in": ["技术部", "产品部"]}},
            {"date": {"$gte": "2024-01-01"}},
        ]
    }
)
```

**优化 2：异步检索 + 并发**

```python
import asyncio

async def hybrid_search(query: str):
    # Dense 和 Sparse 检索并行执行
    dense_task = asyncio.create_task(dense_search(query))
    sparse_task = asyncio.create_task(sparse_search(query))
    
    dense_results, sparse_results = await asyncio.gather(
        dense_task, sparse_task
    )
    
    # RRF 融合
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    
    # 重排序
    reranked = await rerank(query, fused[:30])
    
    return reranked[:5]
```

**优化 3：缓存策略**

```python
import hashlib
from functools import lru_cache

# 查询级缓存（相同查询直接返回缓存结果）
@lru_cache(maxsize=10000)
def cached_search(query_hash: str, query: str):
    return hybrid_search(query)

# Embedding 缓存（避免重复计算 embedding）
embedding_cache = {}

def get_embedding(text: str, model):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash not in embedding_cache:
        embedding_cache[text_hash] = model.encode(text)
    return embedding_cache[text_hash]
```

---

## 六、常见问题与解决方案

### 6.1 检索质量差

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 召回率低 | Embedding 模型不适合当前领域 | 微调 Embedding 模型 |
| 精确匹配差 | 纯向量检索无法处理关键词 | 添加 BM25 混合检索 |
| 结果不相关 | 分块粒度不合理 | 调整 chunk_size，尝试语义分块 |
| 重复内容多 | 块间重叠太大 | 减小 overlap，添加去重逻辑 |

### 6.2 生成质量差

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 幻觉严重 | 上下文不足或 LLM 能力不足 | 增加检索数量，换更强的 LLM |
| 回答不完整 | Top-K 太小 | 增加 K 值，使用上下文压缩 |
| 回答矛盾 | 多个文档信息冲突 | 添加矛盾检测，优先信任高分文档 |
| 引用错误 | LLM 无法准确定位来源 | 使用结构化 prompt，要求逐条引用 |

---

## 七、总结：RAG 系统的演进路线

```
阶段 1: Naive RAG
  ✅ 基础向量检索
  ✅ 固定分块
  ✅ 简单 prompt
  问题：精度低，无法处理复杂查询

        ↓ 优化

阶段 2: Advanced RAG
  ✅ 混合检索 (Dense + Sparse)
  ✅ 重排序 (Cross-Encoder)
  ✅ Query 理解 (改写/分解)
  ✅ 上下文压缩
  问题：架构复杂，维护成本高

        ↓ 优化

阶段 3: Modular RAG
  ✅ 可插拔的检索模块
  ✅ 自适应检索策略
  ✅ 多轮对话支持
  ✅ 知识图谱融合
  问题：需要更智能的路由

        ↓ 优化

阶段 4: Agentic RAG
  ✅ Agent 驱动的自主检索
  ✅ 多步推理
  ✅ 工具调用
  ✅ 反思与验证
```

**核心建议**：

1. **先做好基础**：分块策略和混合检索是性价比最高的优化
2. **评估驱动**：建立完整的评估体系，用数据指导优化方向
3. **渐进式优化**：不要一次性引入所有技术，逐步迭代
4. **监控先行**：生产环境必须有完善的监控告警

RAG 不是一个"一劳永逸"的方案，而是一个需要持续优化的系统工程。理解每个环节的原理和权衡，才能在实际项目中构建出真正可靠的 RAG 系统。

---

> **参考资料**：
> - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
> - [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
> - [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
> - [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity](https://arxiv.org/abs/2402.03216)
> - [Building Production-Grade RAG Systems](https://www.rungalileo.io/blog/building-production-grade-rag-systems)
