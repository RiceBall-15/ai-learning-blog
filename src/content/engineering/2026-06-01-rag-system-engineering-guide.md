---
title: "RAG系统工程化实战：从架构设计到生产部署的完整指南"
description: "深度剖析RAG系统的核心组件设计与工程化实践，涵盖检索优化、重排序、混合检索策略、评估体系构建等关键环节"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: "ai-coding"
tags: ["RAG", "检索增强生成", "向量数据库", "Embedding", "LLM", "AI工程化"]
draft: false
---

## 引言

RAG（Retrieval-Augmented Generation）已经成为企业级AI应用的标配架构。但从一个简单的"检索+生成"Demo到一个稳定可靠的生产级RAG系统，中间横亘着巨大的工程鸿沟。

根据我们的实践经验，一个成熟的RAG系统需要解决以下核心挑战：

```
┌─────────────────────────────────────────────────────────────────┐
│                  RAG系统工程化核心挑战                             │
├──────────────────┬──────────────────────────────────────────────┤
│  检索质量         │  噪声文档、语义漂移、多语言混合检索             │
│  上下文管理       │  信息过载、上下文窗口限制、幻觉抑制             │
│  数据处理         │  增量更新、多格式解析、分块策略优化              │
│  评估体系         │  端到端评估、组件级评估、在线A/B测试             │
│  性能优化         │  延迟优化、吞吐提升、成本控制                   │
│  可观测性         │  检索链路追踪、质量监控、异常告警               │
└──────────────────┴──────────────────────────────────────────────┘
```

本文将从零搭建一个生产级RAG系统的完整架构，并分享我们在每个环节的实战经验。

## 系统架构全景

```
┌───────────────────────────────────────────────────────────────────┐
│                     生产级 RAG 系统架构                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    离线处理层 (Offline)                      │  │
│  │                                                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │ 文档解析  │→│ 智能分块  │→│ Embedding │→│ 向量入库  │ │  │
│  │  │ Parser   │  │ Chunker  │  │ Encoder  │  │ Indexer  │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  │                         │                                   │  │
│  │                    元数据提取                                 │  │
│  │               (标题/表格/图片/链接)                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    在线服务层 (Online)                       │  │
│  │                                                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │ Query    │→│ 混合检索  │→│ 重排序    │→│ 上下文    │ │  │
│  │  │ 理解     │  │ Search   │  │ Reranker │  │ 组装      │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  │       │              │             │              │         │  │
│  │  意图识别      向量+关键词    交叉编码器    Prompt模板        │  │
│  │  查询改写      BM25+Dense    精排打分     引用标注           │  │
│  │                                                            │  │
│  │  ┌──────────┐  ┌──────────┐                               │  │
│  │  │ 生成回答  │→│ 质量校验  │                               │  │
│  │  │ LLM Gen  │  │ QA Check │                               │  │
│  │  └──────────┘  └──────────┘                               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    评估与监控层                               │  │
│  │  检索质量 │ 生成质量 │ 延迟监控 │ 用户反馈 │ A/B测试          │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

## 核心组件一：智能分块策略

分块（Chunking）是RAG系统中最容易被低估的环节。分块策略直接决定了检索质量和最终回答的准确性。

### 分块策略对比

```
┌──────────────────┬──────────┬──────────┬──────────────────────────┐
│     策略          │ 适用场景  │ 优点      │ 缺点                      │
├──────────────────┼──────────┼──────────┼──────────────────────────┤
│  固定长度分块     │ 简单场景  │ 实现简单  │ 语义断裂、上下文丢失        │
│  句子级分块       │ 通用文本  │ 保持语义  │ 长文档块大小不均匀          │
│  语义分块         │ 复杂文档  │ 语义完整  │ 计算成本高                 │
│  文档结构分块     │ 技术文档  │ 保留结构  │ 依赖文档格式               │
│  父子文档分块     │ 知识库    │ 检索精度高│ 存储开销大                 │
│  递归分块         │ 通用      │ 平衡性好  │ 需要调参                   │
└──────────────────┴──────────┴──────────┴──────────────────────────┘
```

### 生产级分块实现

```python
"""
智能分块引擎 - 支持多策略混合分块
"""
from dataclasses import dataclass, field
from typing import List, Optional
import re

@dataclass
class ChunkConfig:
    chunk_size: int = 512           # 目标块大小（字符数）
    chunk_overlap: int = 50         # 块间重叠
    min_chunk_size: int = 100       # 最小块大小
    max_chunk_size: int = 1024      # 最大块大小
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",      # 段落分隔
        "\n",        # 换行
        "。",        # 中文句号
        ". ",        # 英文句号
        "；",        # 中文分号
        "; ",        # 英文分号
        "，",        # 中文逗号
        ", ",        # 英文逗号
        " ",         # 空格
        "",          # 字符级（最后手段）
    ])

class SemanticChunker:
    """基于文档结构和语义的智能分块器"""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
    
    def chunk_document(self, text: str, metadata: dict = None) -> List[dict]:
        """主入口：智能选择分块策略"""
        metadata = metadata or {}
        
        # 策略1：Markdown/HTML文档 → 结构化分块
        if self._is_structured_doc(text):
            return self._structural_chunking(text, metadata)
        
        # 策略2：技术文档 → 标题感知分块
        if self._is_tech_doc(text):
            return self._heading_aware_chunking(text, metadata)
        
        # 策略3：通用文本 → 递归语义分块
        return self._recursive_semantic_chunking(text, metadata)
    
    def _recursive_semantic_chunking(self, text: str, 
                                      metadata: dict) -> List[dict]:
        """递归语义分块：优先按语义边界分割"""
        chunks = []
        self._split_recursive(text, chunks, metadata, depth=0)
        return chunks
    
    def _split_recursive(self, text: str, chunks: list, 
                         metadata: dict, depth: int):
        """递归分割实现"""
        if len(text) <= self.config.chunk_size:
            if len(text) >= self.config.min_chunk_size:
                chunks.append({
                    "content": text.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks),
                        "char_count": len(text),
                    }
                })
            return
        
        # 按当前层级的分隔符分割
        separator = self.config.separators[min(depth, len(self.config.separators)-1)]
        
        if separator == "":
            # 字符级分割
            parts = [text[i:i+self.config.chunk_size] 
                    for i in range(0, len(text), 
                                  self.config.chunk_size - self.config.chunk_overlap)]
        else:
            parts = text.split(separator)
        
        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.config.chunk_size:
                current_chunk += part + separator
            else:
                if current_chunk.strip():
                    self._split_recursive(
                        current_chunk.strip(), chunks, metadata, depth + 1
                    )
                current_chunk = part + separator
        
        if current_chunk.strip():
            self._split_recursive(
                current_chunk.strip(), chunks, metadata, depth + 1
            )
    
    def _structural_chunking(self, text: str, metadata: dict) -> List[dict]:
        """按文档结构分块（Markdown标题层级）"""
        chunks = []
        sections = re.split(r'^(#{1,6}\s+.+)$', text, flags=re.MULTILINE)
        
        current_heading = ""
        current_content = ""
        
        for section in sections:
            if re.match(r'^#{1,6}\s+', section):
                # 保存上一个section
                if current_content.strip():
                    chunks.append({
                        "content": current_content.strip(),
                        "metadata": {
                            **metadata,
                            "heading": current_heading,
                            "chunk_index": len(chunks),
                        }
                    })
                current_heading = section.strip()
                current_content = section + "\n"
            else:
                current_content += section
        
        # 最后一个section
        if current_content.strip():
            chunks.append({
                "content": current_content.strip(),
                "metadata": {
                    **metadata,
                    "heading": current_heading,
                    "chunk_index": len(chunks),
                }
            })
        
        return chunks
    
    def _is_structured_doc(self, text: str) -> bool:
        return bool(re.search(r'^#{1,3}\s+', text, re.MULTILINE))
    
    def _is_tech_doc(self, text: str) -> bool:
        patterns = [r'```', r'def\s+\w+', r'class\s+\w+', r'import\s+']
        return sum(1 for p in patterns if re.search(p, text)) >= 2

# 使用示例
chunker = SemanticChunker(ChunkConfig(
    chunk_size=512,
    chunk_overlap=50,
    min_chunk_size=100,
))

doc_text = open("technical_guide.md").read()
chunks = chunker.chunk_document(doc_text, metadata={
    "source": "technical_guide.md",
    "version": "v2.1",
})
```

## 核心组件二：混合检索策略

单一的向量检索或关键词检索都有明显的局限性。生产级RAG系统需要**混合检索**来兼顾语义理解和精确匹配。

### 检索策略对比

```
┌─────────────────┬──────────────┬──────────────┬──────────────────┐
│     策略         │  语义理解     │  精确匹配     │  适用场景          │
├─────────────────┼──────────────┼──────────────┼──────────────────┤
│  向量检索        │  ★★★★★      │  ★★☆☆☆      │  语义相似查询      │
│  BM25关键词      │  ★★☆☆☆      │  ★★★★★      │  精确术语查询      │
│  混合检索        │  ★★★★☆      │  ★★★★☆      │  通用场景          │
│  多路召回+融合   │  ★★★★★      │  ★★★★☆      │  高质量要求场景    │
└─────────────────┴──────────────┴──────────────┴──────────────────┘
```

### 混合检索引擎实现

```python
"""
混合检索引擎 - 融合向量检索与BM25关键词检索
"""
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    doc_id: str
    content: str
    score: float
    retrieval_method: str
    metadata: dict

class HybridRetriever:
    """混合检索引擎：向量检索 + BM25 + 重排序"""
    
    def __init__(self, 
                 vector_store,           # 向量数据库客户端
                 bm25_index,             # BM25索引
                 reranker,               # 重排序模型
                 vector_weight: float = 0.6,
                 bm25_weight: float = 0.4):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def retrieve(self, query: str, top_k: int = 20, 
                 rerank_top_k: int = 5) -> List[RetrievalResult]:
        """混合检索主流程"""
        
        # 并行执行双路检索
        vector_results = self._vector_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)
        
        # RRF (Reciprocal Rank Fusion) 融合
        fused_results = self._rrf_fusion(vector_results, bm25_results, top_k)
        
        # 重排序
        reranked = self._rerank(query, fused_results, rerank_top_k)
        
        return reranked
    
    def _vector_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """向量语义检索"""
        # 使用查询扩展提升召回率
        expanded_queries = self._expand_query(query)
        
        all_results = []
        for q in [query] + expanded_queries:
            results = self.vector_store.search(
                query_embedding=self._encode(q),
                top_k=top_k,
                score_threshold=0.5
            )
            all_results.extend(results)
        
        # 去重并保留最高分
        seen = {}
        for r in all_results:
            if r.doc_id not in seen or r.score > seen[r.doc_id].score:
                seen[r.doc_id] = r
        
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]
    
    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """BM25关键词检索"""
        # 查询预处理：分词、去停用词、同义词扩展
        processed_query = self._preprocess_query(query)
        
        results = self.bm25_index.search(
            query=processed_query,
            top_k=top_k
        )
        
        # BM25分数归一化到[0, 1]
        if results:
            max_score = max(r.score for r in results)
            for r in results:
                r.score = r.score / max_score if max_score > 0 else 0
        
        return results
    
    def _rrf_fusion(self, vec_results: List[RetrievalResult],
                     bm25_results: List[RetrievalResult],
                     top_k: int) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion 融合算法"""
        k = 60  # RRF常数
        doc_scores = {}
        
        # 向量检索贡献
        for rank, result in enumerate(vec_results):
            rrf_score = self.vector_weight * (1.0 / (k + rank + 1))
            if result.doc_id not in doc_scores:
                doc_scores[result.doc_id] = {
                    "result": result,
                    "score": 0
                }
            doc_scores[result.doc_id]["score"] += rrf_score
        
        # BM25检索贡献
        for rank, result in enumerate(bm25_results):
            rrf_score = self.bm25_weight * (1.0 / (k + rank + 1))
            if result.doc_id not in doc_scores:
                doc_scores[result.doc_id] = {
                    "result": result,
                    "score": 0
                }
            doc_scores[result.doc_id]["score"] += rrf_score
        
        # 排序返回
        sorted_docs = sorted(doc_scores.values(), 
                           key=lambda x: x["score"], reverse=True)
        
        results = []
        for item in sorted_docs[:top_k]:
            r = item["result"]
            r.score = item["score"]
            r.retrieval_method = "hybrid_rrf"
            results.append(r)
        
        return results
    
    def _rerank(self, query: str, results: List[RetrievalResult],
                top_k: int) -> List[RetrievalResult]:
        """使用交叉编码器进行重排序"""
        if not results:
            return []
        
        pairs = [(query, r.content) for r in results]
        scores = self.reranker.predict(pairs)
        
        for i, score in enumerate(scores):
            results[i].score = float(score)
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    
    def _expand_query(self, query: str) -> List[str]:
        """查询扩展：生成多个语义相关的查询"""
        # 实际应使用LLM生成扩展查询
        # 这里展示核心逻辑
        expansions = []
        
        # 策略1：同义词替换
        # 策略2：LLM生成变体
        # 策略3：基于历史查询的相似查询
        
        return expansions
    
    def _preprocess_query(self, query: str) -> str:
        """查询预处理"""
        import jieba
        # 中文分词
        tokens = jieba.lcut(query)
        # 过滤停用词
        stop_words = {"的", "了", "是", "在", "和", "有"}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return " ".join(tokens)
    
    def _encode(self, text: str) -> np.ndarray:
        """文本编码为向量"""
        # 调用Embedding模型
        pass
```

## 核心组件三：上下文组装与幻觉抑制

检索到相关文档后，如何组装上下文并抑制LLM幻觉是另一个关键挑战。

### 上下文组装策略

```python
"""
智能上下文组装器 - 动态调整上下文窗口
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ContextConfig:
    max_tokens: int = 4096          # 最大上下文token数
    reserved_tokens: int = 1024     # 预留给回答的token
    citation_enabled: bool = True   # 启用引用标注
    dedup_enabled: bool = True      # 启用去重
    compression_ratio: float = 0.8  # 上下文压缩比

class ContextAssembler:
    """智能上下文组装器"""
    
    def __init__(self, config: ContextConfig):
        self.config = config
        self.max_context_tokens = config.max_tokens - config.reserved_tokens
    
    def assemble(self, query: str, 
                 retrieved_docs: List[dict],
                 template_name: str = "default") -> dict:
        """组装最终的上下文"""
        
        # Step 1: 去重
        docs = self._deduplicate(retrieved_docs) if self.config.dedup_enabled \
               else retrieved_docs
        
        # Step 2: 按相关性排序并截断
        docs = self._fit_to_budget(docs)
        
        # Step 3: 添加引用标注
        if self.config.citation_enabled:
            docs = self._add_citations(docs)
        
        # Step 4: 构建最终prompt
        context = self._build_context(query, docs, template_name)
        
        return {
            "prompt": context,
            "sources": [d["metadata"]["source"] for d in docs],
            "chunk_count": len(docs),
            "token_usage": self._estimate_tokens(context),
        }
    
    def _deduplicate(self, docs: List[dict]) -> List[dict]:
        """基于内容相似度的去重"""
        seen_contents = set()
        unique_docs = []
        
        for doc in docs:
            # 使用内容指纹（简化版，实际应用SimHash）
            content_key = self._content_fingerprint(doc["content"])
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _fit_to_budget(self, docs: List[dict]) -> List[dict]:
        """在token预算内选择最优文档组合"""
        selected = []
        current_tokens = 0
        
        for doc in docs:
            doc_tokens = self._estimate_tokens(doc["content"])
            if current_tokens + doc_tokens <= self.max_context_tokens:
                selected.append(doc)
                current_tokens += doc_tokens
            else:
                # 尝试截断最后一个文档
                remaining = self.max_context_tokens - current_tokens
                if remaining > 100:  # 至少保留100 token
                    truncated = doc.copy()
                    truncated["content"] = self._truncate_to_tokens(
                        doc["content"], remaining
                    )
                    truncated["metadata"]["truncated"] = True
                    selected.append(truncated)
                break
        
        return selected
    
    def _add_citations(self, docs: List[dict]) -> List[dict]:
        """为每个文档块添加引用编号"""
        cited_docs = []
        for i, doc in enumerate(docs):
            cited = doc.copy()
            cited["citation_id"] = i + 1
            cited["content"] = f"[{i+1}] {doc['content']}"
            cited_docs.append(cited)
        return cited_docs
    
    def _build_context(self, query: str, docs: List[dict],
                       template_name: str) -> str:
        """构建最终的Prompt"""
        
        # 引用块
        context_parts = []
        for doc in docs:
            source = doc["metadata"].get("source", "未知来源")
            context_parts.append(
                f"【文档{doc['citation_id']}】来源: {source}\n{doc['content']}"
            )
        
        context_block = "\n\n".join(context_parts)
        
        # Prompt模板
        prompt = f"""你是一个专业的问答助手。请根据以下参考资料回答用户的问题。

## 参考资料
{context_block}

## 回答要求
1. 仅基于提供的参考资料回答，不要编造信息
2. 如果资料中没有相关信息，请明确说明
3. 回答中引用信息时，请标注来源编号，如 [1]
4. 保持回答简洁准确

## 用户问题
{query}

## 回答"""
        
        return prompt
    
    def _content_fingerprint(self, content: str) -> str:
        """内容指纹（简化版）"""
        import hashlib
        normalized = content.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数（中英文混合）"""
        # 简化估算：中文1字≈2token，英文1词≈1token
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return chinese_chars * 2 + english_words
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """截断文本到指定token数"""
        # 简化实现
        estimated_ratio = max_tokens / self._estimate_tokens(text)
        char_limit = int(len(text) * estimated_ratio * 0.8)
        return text[:char_limit] + "... [已截断]"
```

### 幻觉检测与抑制

```python
"""
基于引用验证的幻觉检测器
"""
import re
from typing import List, Tuple

class HallucinationDetector:
    """检测LLM回答中的幻觉内容"""
    
    def __init__(self, llm_client, embedding_model):
        self.llm = llm_client
        self.embedder = embedding_model
    
    def check(self, answer: str, context_docs: List[dict]) -> dict:
        """检查回答是否存在幻觉"""
        
        # Step 1: 提取回答中的事实性声明
        claims = self._extract_claims(answer)
        
        # Step 2: 检查每个声明是否有上下文支持
        supported = []
        unsupported = []
        
        for claim in claims:
            is_supported, evidence = self._verify_claim(claim, context_docs)
            if is_supported:
                supported.append({"claim": claim, "evidence": evidence})
            else:
                unsupported.append({"claim": claim})
        
        # Step 3: 计算幻觉率
        total = len(claims)
        hallucination_rate = len(unsupported) / total if total > 0 else 0
        
        return {
            "hallucination_rate": hallucination_rate,
            "total_claims": total,
            "supported_claims": len(supported),
            "unsupported_claims": len(unsupported),
            "details": {
                "supported": supported,
                "unsupported": unsupported
            },
            "is_reliable": hallucination_rate < 0.2  # 低于20%视为可靠
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """提取事实性声明"""
        # 使用LLM提取声明
        prompt = f"""请从以下文本中提取所有事实性声明，每行一个：

{text}

事实性声明："""
        
        response = self.llm.generate(prompt)
        claims = [c.strip() for c in response.strip().split('\n') if c.strip()]
        return claims
    
    def _verify_claim(self, claim: str, 
                      docs: List[dict]) -> Tuple[bool, str]:
        """验证声明是否有上下文支持"""
        claim_embedding = self.embedder.encode(claim)
        
        best_score = 0
        best_evidence = ""
        
        for doc in docs:
            doc_embedding = self.embedder.encode(doc["content"])
            similarity = np.dot(claim_embedding, doc_embedding) / (
                np.linalg.norm(claim_embedding) * np.linalg.norm(doc_embedding)
            )
            
            if similarity > best_score:
                best_score = similarity
                best_evidence = doc["content"]
        
        # 相似度阈值
        return best_score > 0.75, best_evidence
```

## 核心组件四：评估体系

没有评估就没有优化。一个完善的RAG评估体系需要覆盖检索、生成和端到端三个层面。

### 评估指标体系

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG评估指标体系                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ 检索质量 ─────────────────────────────────────────────┐    │
│  │  Recall@K     │ 前K个结果中包含相关文档的比例              │    │
│  │  Precision@K  │ 前K个结果中相关文档的比例                  │    │
│  │  MRR          │ 第一个相关结果的倒数排名                   │    │
│  │  NDCG@K       │ 考虑排序位置的归一化折扣累积增益            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─ 生成质量 ─────────────────────────────────────────────┐    │
│  │  Faithfulness  │ 回答与上下文的一致性（无幻觉）             │    │
│  │  Relevance     │ 回答与问题的相关性                        │    │
│  │  Completeness  │ 回答的完整度                              │    │
│  │  Citation Accuracy │ 引用标注的准确性                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─ 端到端质量 ───────────────────────────────────────────┐    │
│  │  Answer Correctness │ 最终回答的正确性                     │    │
│  │  User Satisfaction  │ 用户满意度（SUS评分）                 │    │
│  │  Latency P99        │ 端到端延迟                          │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 自动化评估框架

```python
"""
RAG自动化评估框架
"""
import json
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class EvalSample:
    query: str
    expected_answer: str
    expected_doc_ids: List[str]  # 期望检索到的文档ID

class RAGEvaluator:
    """RAG系统端到端评估器"""
    
    def __init__(self, rag_system, eval_dataset: List[EvalSample]):
        self.rag = rag_system
        self.dataset = eval_dataset
    
    def evaluate(self) -> Dict:
        """运行完整评估"""
        results = {
            "retrieval_metrics": {},
            "generation_metrics": {},
            "end_to_end_metrics": {},
        }
        
        # 收集所有样本的评估结果
        retrieval_scores = []
        generation_scores = []
        
        for sample in self.dataset:
            # 运行RAG系统
            rag_result = self.rag.query(sample.query)
            
            # 评估检索质量
            ret_score = self._evaluate_retrieval(
                retrieved_ids=[d["id"] for d in rag_result["documents"]],
                expected_ids=sample.expected_doc_ids
            )
            retrieval_scores.append(ret_score)
            
            # 评估生成质量
            gen_score = self._evaluate_generation(
                generated=rag_result["answer"],
                expected=sample.expected_answer,
                context=rag_result["context"]
            )
            generation_scores.append(gen_score)
        
        # 汇总指标
        results["retrieval_metrics"] = self._aggregate_retrieval_scores(
            retrieval_scores
        )
        results["generation_metrics"] = self._aggregate_generation_scores(
            generation_scores
        )
        results["end_to_end_metrics"] = {
            "overall_score": np.mean([
                r["recall"] for r in retrieval_scores
            ]) * 0.4 + np.mean([
                g["faithfulness"] for g in generation_scores
            ]) * 0.3 + np.mean([
                g["relevance"] for g in generation_scores
            ]) * 0.3
        }
        
        return results
    
    def _evaluate_retrieval(self, retrieved_ids: List[str],
                            expected_ids: List[str]) -> Dict:
        """评估单个样本的检索质量"""
        retrieved_set = set(retrieved_ids)
        expected_set = set(expected_ids)
        
        # Recall@K
        recall = len(retrieved_set & expected_set) / len(expected_set) \
                 if expected_set else 0
        
        # Precision@K
        precision = len(retrieved_set & expected_set) / len(retrieved_set) \
                    if retrieved_set else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_set:
                mrr = 1.0 / (i + 1)
                break
        
        return {
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        }
    
    def _evaluate_generation(self, generated: str, expected: str,
                             context: str) -> Dict:
        """评估单个样本的生成质量"""
        # 使用LLM-as-Judge进行质量评估
        faithfulness = self._judge_faithfulness(generated, context)
        relevance = self._judge_relevance(generated, expected)
        completeness = self._judge_completeness(generated, expected)
        
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
        }
    
    def _judge_faithfulness(self, answer: str, context: str) -> float:
        """评估回答的忠实度（是否基于上下文）"""
        prompt = f"""请评估以下回答是否完全基于提供的上下文，没有编造信息。

上下文：{context}

回答：{answer}

评分标准：
1.0 - 完全基于上下文
0.8 - 基本基于上下文，有少量推断
0.6 - 部分基于上下文，有明显推断
0.4 - 大量编造内容
0.2 - 几乎完全编造

评分（仅返回数字）："""
        
        score = float(self.rag.llm.generate(prompt).strip())
        return score
    
    def _judge_relevance(self, generated: str, expected: str) -> float:
        """评估回答的相关性"""
        prompt = f"""请评估生成回答与标准答案的相关性。

标准答案：{expected}

生成回答：{generated}

评分（0-1，仅返回数字）："""
        
        score = float(self.rag.llm.generate(prompt).strip())
        return score
    
    def _judge_completeness(self, generated: str, expected: str) -> float:
        """评估回答的完整度"""
        prompt = f"""请评估生成回答是否涵盖了标准答案的所有关键信息点。

标准答案：{expected}

生成回答：{generated}

评分（0-1，仅返回数字）："""
        
        score = float(self.rag.llm.generate(prompt).strip())
        return score
    
    def _aggregate_retrieval_scores(self, scores: List[Dict]) -> Dict:
        """汇总检索评估指标"""
        return {
            "recall@k": np.mean([s["recall"] for s in scores]),
            "precision@k": np.mean([s["precision"] for s in scores]),
            "mrr": np.mean([s["mrr"] for s in scores]),
        }
    
    def _aggregate_generation_scores(self, scores: List[Dict]) -> Dict:
        """汇总生成评估指标"""
        return {
            "faithfulness": np.mean([s["faithfulness"] for s in scores]),
            "relevance": np.mean([s["relevance"] for s in scores]),
            "completeness": np.mean([s["completeness"] for s in scores]),
        }

# 使用示例
eval_dataset = [
    EvalSample(
        query="如何配置Kubernetes的HPA自动扩缩容？",
        expected_answer="通过创建HorizontalPodAutoscaler资源...",
        expected_doc_ids=["doc_k8s_hpa_001", "doc_k8s_hpa_002"]
    ),
    # ... 更多评估样本
]

evaluator = RAGEvaluator(rag_system, eval_dataset)
results = evaluator.evaluate()
print(json.dumps(results, indent=2, ensure_ascii=False))
```

## 性能优化实战

### 延迟优化

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG延迟优化策略                                │
├──────────────────┬───────────────┬──────────────────────────────┤
│     优化点        │    优化效果    │      实现方式                 │
├──────────────────┼───────────────┼──────────────────────────────┤
│  Embedding缓存   │  -50~200ms   │  Redis缓存热门查询的向量      │
│  检索结果缓存    │  -100~500ms   │  LRU缓存完全相同的查询        │
│  向量检索加速    │  -20~50ms     │  HNSW/IVF索引 + GPU加速      │
│  Reranker量化   │  -30~80ms     │  INT8量化交叉编码器           │
│  流式输出       │  首字节-200ms  │  检索完成后立即开始流式生成    │
│  预计算索引     │  -100~300ms   │  离线构建BM25+向量索引        │
└──────────────────┴───────────────┴──────────────────────────────┘
```

### 吞吐优化

```python
"""
异步并行检索引擎 - 提升吞吐量
"""
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncHybridRetriever:
    """异步并行混合检索"""
    
    def __init__(self, vector_client, bm25_client, reranker):
        self.vector_client = vector_client
        self.bm25_client = bm25_client
        self.reranker = reranker
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def retrieve_async(self, queries: List[str], 
                              top_k: int = 10) -> List[List[dict]]:
        """批量异步检索"""
        tasks = [
            self._single_query_retrieve(q, top_k) 
            for q in queries
        ]
        results = await asyncio.gather(*tasks)
        return results
    
    async def _single_query_retrieve(self, query: str, 
                                      top_k: int) -> List[dict]:
        """单查询的异步检索"""
        # 并行执行向量检索和BM25检索
        vector_task = asyncio.create_task(
            self._async_vector_search(query, top_k)
        )
        bm25_task = asyncio.create_task(
            self._async_bm25_search(query, top_k)
        )
        
        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task
        )
        
        # 融合和重排序
        fused = self._rrf_fusion(vector_results, bm25_results, top_k * 2)
        
        # 异步重排序
        reranked = await self._async_rerank(query, fused, top_k)
        
        return reranked
```

## 总结与最佳实践

```
┌─────────────────────────────────────────────────────────────────┐
│                  RAG工程化最佳实践清单                            │
├────┬────────────────────────────────────────────────────────────┤
│ 1  │ 分块策略决定检索质量，不要用简单的固定长度分块                   │
│ 2  │ 混合检索（向量+BM25）是生产环境的标配，不是可选项               │
│ 3  │ 重排序器是性价比最高的优化手段，务必引入                        │
│ 4  │ 上下文组装要考虑token预算和引用标注                            │
│ 5  │ 评估体系要覆盖检索、生成和端到端三个层面                       │
│ 6  │ 缓存是提升延迟最直接的手段，热门查询缓存命中率可达60%+          │
│ 7  │ 可观测性要贯穿全链路，每个环节的输入输出都要可追踪               │
│ 8  │ 增量更新机制必不可少，避免全量重建索引                          │
│ 9  │ 幻觉检测要成为生产流程的一环，不能只靠人工抽检                  │
│ 10 │ A/B测试驱动优化，不要凭感觉调参                                │
└────┴────────────────────────────────────────────────────────────┘
```

RAG系统的工程化是一个持续迭代的过程。从最初的简单原型，到生产级系统，每个环节都有大量的工程细节需要打磨。希望本文的实战经验能为你的RAG系统建设提供有价值的参考。
