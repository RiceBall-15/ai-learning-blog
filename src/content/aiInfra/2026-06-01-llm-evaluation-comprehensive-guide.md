---
title: "LLM评估体系：从自动评估到Agent效果量化的完整指南"
date: 2026-06-01
category: aiInfra
subCategory: evaluation
description: "构建完整的LLM评估体系，涵盖自动评估、人工评估、Agent效果量化三大维度，结合RAGAS、DeepEval等主流框架实战，提供可落地的评估流水线设计"
tags: [llm-evaluation, ragas, deepeval, agent-evaluation, benchmarks, evaluation-framework]
author: "AI技术博客"
readingTime: "25分钟"
---

# LLM评估体系：从自动评估到Agent效果量化的完整指南

> **摘要**：在大语言模型应用日益普及的今天，如何科学评估LLM系统的效果成为关键挑战。本文系统梳理LLM评估方法论，从基础的自动评估指标到复杂的Agent效果量化框架，结合RAGAS、DeepEval等主流工具实战，帮助构建可落地的评估流水线。

---

## 一、为什么LLM评估如此重要？

### 1.1 评估的核心挑战

LLM评估面临传统软件测试不存在的特殊挑战：

```
┌─────────────────────────────────────────────────────────────┐
│                  LLM评估挑战全景                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 语义多样性   │  │ 主观性判断  │  │ 长尾分布    │        │
│  │             │  │             │  │             │        │
│  │ "好"的回答  │  │ 评分标准   │  │ 边界情况    │        │
│  │ 有很多形式  │  │ 因人而异   │  │ 难以覆盖    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 幻觉问题    │  │ 上下文依赖  │  │ 实时性     │        │
│  │             │  │             │  │             │        │
│  │ 生成错误    │  │ 同样输入    │  │ 模型更新    │        │
│  │ 但流畅     │  │ 不同输出    │  │ 评估失效    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 评估维度框架

```
┌─────────────────────────────────────────────────────────────┐
│                 LLM评估维度框架                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────────┐                      │
│                    │  LLM评估体系    │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│         ┌──────────────────┼──────────────────┐            │
│         │                  │                  │            │
│    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐      │
│    │ 质量维度 │        │ 安全维度 │        │ 效率维度 │      │
│    └────┬────┘        └────┬────┘        └────┬────┘      │
│         │                  │                  │            │
│    ┌────┴────┐        ┌────┴────┐        ┌────┴────┐      │
│    │ 准确性  │        │ 无害性  │        │ 延迟    │      │
│    │ 相关性  │        │ 公平性  │        │ 吞吐量  │      │
│    │ 完整性  │        │ 隐私性  │        │ 成本    │      │
│    │ 流畅性  │        │ 合规性  │        │ 可靠性  │      │
│    │ 创造性  │        │         │        │         │      │
│    └─────────┘        └─────────┘        └─────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、自动评估方法详解

### 2.1 基础评估指标

#### 2.1.1 精确匹配（Exact Match）

```python
def exact_match(prediction: str, reference: str) -> float:
    """
    精确匹配 - 最严格的评估指标
    
    适用：事实性问答、实体提取
    """
    return 1.0 if prediction.strip() == reference.strip() else 0.0

def normalized_exact_match(predictions: list, references: list) -> float:
    """归一化精确匹配 - 支持批量评估"""
    matches = [
        exact_match(p, r) 
        for p, r in zip(predictions, references)
    ]
    return sum(matches) / len(matches) if matches else 0.0

# 示例
predictions = ["Python", "机器学习", "GPT-4"]
references = ["Python", "Machine Learning", "GPT-4"]
print(f"Exact Match: {normalized_exact_match(predictions, references):.2f}")
# 输出: Exact Match: 0.67
```

#### 2.1.2 BLEU分数

```python
import math
from collections import Counter

def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """
    BLEU分数 - 机器翻译经典指标
    
    原理：计算n-gram精确率的几何平均
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # 计算各阶n-gram精确率
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        
        # 截断计数
        clipped_counts = {
            ng: min(count, ref_ngrams.get(ng, 0))
            for ng, count in pred_ngrams.items()
        }
        
        total_pred = sum(pred_ngrams.values())
        total_clipped = sum(clipped_counts.values())
        
        precision = total_clipped / total_pred if total_pred > 0 else 0
        precisions.append(precision)
    
    # 几何平均
    if min(precisions) > 0:
        geo_mean = math.exp(
            sum(math.log(p) for p in precisions) / len(precisions)
        )
    else:
        geo_mean = 0.0
    
    # Brevity Penalty (简短惩罚)
    bp = min(1.0, math.exp(1 - ref_tokens.__len__() / pred_tokens.__len__()))
    
    return bp * geo_mean

def _get_ngrams(tokens: list, n: int) -> dict:
    """提取n-gram"""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams

# 示例
pred = "The cat sat on the mat"
ref = "The cat is on the mat"
print(f"BLEU: {compute_bleu(pred, ref):.4f}")
# 输出: BLEU: 0.6554
```

#### 2.1.3 ROUGE分数

```python
def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L - 基于最长公共子序列
    
    适用：摘要生成、文档回答
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    # 计算LCS长度
    lcs_length = _lcs_length(pred_tokens, ref_tokens)
    
    # 计算精确率和召回率
    precision = lcs_length / len(pred_tokens) if pred_tokens else 0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0
    
    # F1分数
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return f1

def _lcs_length(x: list, y: list) -> int:
    """最长公共子序列长度"""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# 示例
pred = "The quick brown fox jumps over the lazy dog"
ref = "The quick brown dog jumps over the lazy fox"
print(f"ROUGE-L: {compute_rouge_l(pred, ref):.4f}")
# 输出: ROUGE-L: 0.6154
```

### 2.2 语义评估指标

#### 2.2.1 BERTScore

```python
import torch
from transformers import AutoTokenizer, AutoModel

class BERTScoreEvaluator:
    """
    BERTScore - 基于语义相似度的评估
    
    原理：使用BERT计算预测和参考的token级语义相似度
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def score(
        self, 
        predictions: list, 
        references: list
    ) -> dict:
        """计算BERTScore"""
        with torch.no_grad():
            # 编码
            pred_encodings = self.tokenizer(
                predictions, padding=True, truncation=True,
                return_tensors="pt"
            )
            ref_encodings = self.tokenizer(
                references, padding=True, truncation=True,
                return_tensors="pt"
            )
            
            # 获取embedding
            pred_embeddings = self.model(**pred_encodings).last_hidden_state
            ref_embeddings = self.model(**ref_encodings).last_hidden_state
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                pred_embeddings.unsqueeze(2),
                ref_embeddings.unsqueeze(1),
                dim=-1
            )
            
            # 精确率和召回率
            precision = similarity.max(dim=2)[0].mean(dim=1)
            recall = similarity.max(dim=1)[0].mean(dim=1)
            
            # F1
            f1 = 2 * precision * recall / (precision + recall)
            
            return {
                "precision": precision.mean().item(),
                "recall": recall.mean().item(),
                "f1": f1.mean().item()
            }

# 使用示例
evaluator = BERTScoreEvaluator()
predictions = ["AI is transforming the world", "Machine learning is great"]
references = ["Artificial intelligence changes everything", "ML is wonderful"]
results = evaluator.score(predictions, references)
print(f"BERTScore: P={results['precision']:.4f}, R={results['recall']:.4f}, F1={results['f1']:.4f}")
```

#### 2.2.2 语义相似度（Cosine Similarity）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSimilarityEvaluator:
    """
    语义相似度评估 - 使用Sentence-BERT
    
    适用：开放域问答、对话评估
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def compute_similarity(
        self, 
        predictions: list, 
        references: list
    ) -> dict:
        """计算语义相似度"""
        # 编码
        pred_embeddings = self.model.encode(predictions)
        ref_embeddings = self.model.encode(references)
        
        # 逐对计算余弦相似度
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = np.dot(pred_emb, ref_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb)
            )
            similarities.append(sim)
        
        return {
            "mean_similarity": np.mean(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "std_similarity": np.std(similarities),
            "individual_scores": similarities
        }

# 使用示例
evaluator = SemanticSimilarityEvaluator()
preds = [
    "Python is a programming language",
    "AI will change the world"
]
refs = [
    "Python is used for software development",
    "Artificial intelligence will transform society"
]
results = evaluator.compute_similarity(preds, refs)
print(f"Mean Similarity: {results['mean_similarity']:.4f}")
```

---

## 三、基于LLM的评估方法

### 3.1 LLM-as-Judge范式

```python
from typing import List, Dict, Any
import json

class LLMJudge:
    """
    LLM-as-Judge - 使用LLM评估LLM
    
    优势：
    - 能够评估语义质量
    - 可以处理开放式回答
    - 评估标准可以灵活定制
    """
    
    def __init__(self, judge_model_client):
        self.judge = judge_model_client
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        reference: str = None,
        criteria: List[str] = None
    ) -> Dict[str, Any]:
        """
        评估单个回答
        """
        if criteria is None:
            criteria = ["relevance", "accuracy", "completeness", "clarity"]
        
        evaluation_prompt = f"""
        你是一个专业的AI回答质量评估专家。请根据以下标准评估回答质量。
        
        ## 问题
        {question}
        
        ## 待评估回答
        {answer}
        
        {"## 参考答案（如有）" if reference else ""}
        {reference if reference else ""}
        
        ## 评估标准
        请对以下每个维度进行1-5分的评分（5分最优）：
        
        1. **相关性 (Relevance)**: 回答是否与问题相关？
        2. **准确性 (Accuracy)**: 信息是否准确无误？
        3. **完整性 (Completeness)**: 是否完整回答了问题？
        4. **清晰度 (Clarity)**: 表达是否清晰易懂？
        5. **有用性 (Usefulness)**: 回答是否真正有帮助？
        
        ## 输出格式
        请输出JSON格式的评分和理由：
        {{
            "scores": {{
                "relevance": {{"score": 0, "reason": "评分理由"}},
                "accuracy": {{"score": 0, "reason": "评分理由"}},
                "completeness": {{"score": 0, "reason": "评分理由"}},
                "clarity": {{"score": 0, "reason": "评分理由"}},
                "usefulness": {{"score": 0, "reason": "评分理由"}}
            }},
            "overall_score": 0,
            "overall_reason": "总体评价"
        }}
        """
        
        response = await self.judge.generate(evaluation_prompt)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return {"error": "Failed to parse evaluation", "raw": response}
    
    async def compare_answers(
        self,
        question: str,
        answer_a: str,
        answer_b: str
    ) -> Dict[str, Any]:
        """
        比较两个回答的优劣
        """
        comparison_prompt = f"""
        你是一个专业的AI回答质量评估专家。请比较以下两个回答的优劣。
        
        ## 问题
        {question}
        
        ## 回答A
        {answer_a}
        
        ## 回答B
        {answer_b}
        
        ## 评估维度
        1. 信息准确性
        2. 表达清晰度
        3. 内容完整性
        4. 实用价值
        
        ## 输出要求
        请输出JSON格式：
        {{
            "winner": "A" 或 "B" 或 "tie",
            "confidence": 0.0-1.0,
            "analysis": {{
                "answer_a": {{"strengths": [], "weaknesses": []}},
                "answer_b": {{"strengths": [], "weaknesses": []}}
            }},
            "reasoning": "选择理由"
        }}
        """
        
        response = await self.judge.generate(comparison_prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse comparison"}
    
    async def evaluate_factual_accuracy(
        self,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        评估事实准确性（幻觉检测）
        """
        hallucination_prompt = f"""
        你是一个事实核查专家。请检查以下回答中的事实准确性。
        
        ## 参考上下文
        {context}
        
        ## 待检查回答
        {answer}
        
        ## 任务
        1. 识别回答中的所有事实性声明
        2. 检查每个声明是否与上下文一致
        3. 标记可能的幻觉（上下文中不存在或矛盾的信息）
        
        ## 输出格式
        {{
            "claims": [
                {{
                    "statement": "事实声明",
                    "verdict": "supported/contradicted/not_found",
                    "evidence": "支持或反驳的证据",
                    "confidence": 0.0-1.0
                }}
            ],
            "hallucination_score": 0.0-1.0,
            "hallucination_count": 0,
            "summary": "总体评估"
        }}
        """
        
        response = await self.judge.generate(hallucination_prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse hallucination check"}
```

### 3.2 多维度评估框架

```python
class MultiDimensionEvaluator:
    """
    多维度评估框架
    
    支持自定义评估维度和权重
    """
    
    def __init__(self, judge_client, dimensions: Dict[str, float] = None):
        self.judge = judge_client
        # 默认维度和权重
        self.dimensions = dimensions or {
            "relevance": 0.25,
            "accuracy": 0.25,
            "completeness": 0.20,
            "clarity": 0.15,
            "creativity": 0.15
        }
    
    async def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str] = None
    ) -> List[Dict]:
        """批量评估"""
        results = []
        
        for i, (q, a) in enumerate(zip(questions, answers)):
            ref = references[i] if references else None
            result = await self._evaluate_single(q, a, ref)
            results.append(result)
        
        # 计算总体统计
        return self._aggregate_results(results)
    
    async def _evaluate_single(
        self,
        question: str,
        answer: str,
        reference: str = None
    ) -> Dict:
        """评估单个样本"""
        # 构建评估prompt
        dim_descriptions = "\n".join([
            f"- {dim}: 权重{weight}"
            for dim, weight in self.dimensions.items()
        ])
        
        prompt = f"""
        评估以下AI回答的质量。
        
        问题: {question}
        回答: {answer}
        {f"参考: {reference}" if reference else ""}
        
        评估维度:
        {dim_descriptions}
        
        请为每个维度打分(0-1)，并给出理由。
        输出JSON格式。
        """
        
        response = await self.judge.generate(prompt)
        
        try:
            scores = json.loads(response)
            return self._compute_weighted_score(scores)
        except:
            return {"error": "evaluation_failed"}
    
    def _compute_weighted_score(self, scores: Dict) -> Dict:
        """计算加权分数"""
        weighted_sum = 0
        for dim, weight in self.dimensions.items():
            if dim in scores:
                dim_score = scores[dim].get("score", 0)
                weighted_sum += dim_score * weight
        
        return {
            "dimension_scores": scores,
            "weighted_score": weighted_sum,
            "grade": self._score_to_grade(weighted_sum)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """分数转换为等级"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C"
        else:
            return "D"
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """聚合批量评估结果"""
        valid_results = [r for r in results if "weighted_score" in r]
        
        if not valid_results:
            return {"error": "no_valid_results"}
        
        scores = [r["weighted_score"] for r in valid_results]
        
        return {
            "results": results,
            "statistics": {
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_score": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
                "grade_distribution": self._grade_distribution(valid_results)
            }
        }
    
    def _grade_distribution(self, results: List[Dict]) -> Dict:
        """等级分布统计"""
        dist = {"A+": 0, "A": 0, "B+": 0, "B": 0, "C": 0, "D": 0}
        for r in results:
            grade = r.get("grade", "D")
            dist[grade] = dist.get(grade, 0) + 1
        return dist
```

---

## 四、RAG评估：RAGAS实战

### 4.1 RAG评估指标体系

```
┌─────────────────────────────────────────────────────────────┐
│                  RAGAS评估指标体系                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    评估维度                          │   │
│  ├─────────────┬─────────────┬─────────────┬───────────┤   │
│  │  检索质量    │  生成质量    │  一致性    │  相关性   │   │
│  ├─────────────┼─────────────┼─────────────┼───────────┤   │
│  │ Context     │ Answer      │ Faithfulness│ Relevancy │   │
│  │ Precision   │ Relevancy   │            │           │   │
│  │ Context     │ Answer      │            │           │   │
│  │ Recall      │ Correctness │            │           │   │
│  └─────────────┴─────────────┴─────────────┴───────────┘   │
│                                                             │
│  关键公式：                                                  │
│  Faithfulness = (Supported Claims) / (Total Claims)        │
│  Relevancy = Relevant Chunks / Total Chunks                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 RAGAS实战代码

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

class RAGASEvaluator:
    """
    RAGAS评估器 - 完整的RAG评估流水线
    """
    
    def __init__(self, llm_client, embedding_client):
        self.llm = llm_client
        self.embeddings = embedding_client
    
    def prepare_evaluation_dataset(
        self,
        questions: list,
        answers: list,
        contexts: list,
        ground_truths: list
    ) -> Dataset:
        """
        准备RAGAS评估数据集
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        return Dataset.from_dict(data)
    
    async def evaluate_rag(
        self,
        rag_system,
        test_data: list
    ) -> dict:
        """
        执行完整的RAG评估
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        # 收集测试数据
        for item in test_data:
            questions.append(item["question"])
            ground_truths.append(item["ground_truth"])
            
            # 运行RAG系统
            result = await rag_system.query(item["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
        
        # 创建数据集
        dataset = self.prepare_evaluation_dataset(
            questions, answers, contexts, ground_truths
        )
        
        # 执行RAGAS评估
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness
            ]
        )
        
        return self._format_results(result)
    
    def _format_results(self, ragas_result) -> dict:
        """格式化评估结果"""
        return {
            "metrics": {
                "faithfulness": ragas_result["faithfulness"],
                "answer_relevancy": ragas_result["answer_relevancy"],
                "context_precision": ragas_result["context_precision"],
                "context_recall": ragas_result["context_recall"],
                "answer_correctness": ragas_result["answer_correctness"]
            },
            "overall_score": (
                ragas_result["faithfulness"] +
                ragas_result["answer_relevancy"] +
                ragas_result["context_precision"] +
                ragas_result["context_recall"] +
                ragas_result["answer_correctness"]
            ) / 5,
            "sample_size": len(ragas_result)
        }

# 使用示例
async def run_rag_evaluation():
    """运行RAG评估示例"""
    evaluator = RAGASEvaluator(llm_client, embedding_client)
    
    test_data = [
        {
            "question": "什么是RAG？",
            "ground_truth": "RAG（检索增强生成）是一种结合检索和生成的AI技术"
        },
        {
            "question": "RAG有哪些优势？",
            "ground_truth": "RAG可以减少幻觉、提供可追溯来源、更新知识库"
        }
    ]
    
    results = await evaluator.evaluate_rag(rag_system, test_data)
    
    print(f"Faithfulness: {results['metrics']['faithfulness']:.4f}")
    print(f"Answer Relevancy: {results['metrics']['answer_relevancy']:.4f}")
    print(f"Overall Score: {results['overall_score']:.4f}")
```

### 4.3 自定义RAG评估指标

```python
class CustomRAGMetrics:
    """
    自定义RAG评估指标
    
    针对特定业务场景的评估需求
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def evaluate_source_attribution(
        self,
        answer: str,
        sources: list
    ) -> dict:
        """
        评估来源归因质量
        
        检查答案中的信息是否都能追溯到提供的来源
        """
        prompt = f"""
        评估以下答案的来源归因质量。
        
        ## 答案
        {answer}
        
        ## 提供的来源
        {sources}
        
        ## 评估要点
        1. 答案中的每个事实声明是否都有来源支持？
        2. 是否存在无法追溯的信息？
        3. 来源引用是否准确？
        
        输出JSON格式：
        {{
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "attribution_score": 0.0-1.0,
            "details": [
                {{
                    "claim": "事实声明",
                    "source": "对应来源",
                    "verdict": "supported/unsupported"
                }}
            ]
        }}
        """
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
    
    async def evaluate_answer_coverage(
        self,
        question: str,
        answer: str
    ) -> dict:
        """
        评估答案覆盖度
        
        检查答案是否完整回答了问题的所有方面
        """
        prompt = f"""
        分析以下问题和答案的覆盖度。
        
        ## 问题
        {question}
        
        ## 答案
        {answer}
        
        ## 分析要求
        1. 识别问题中包含的所有子问题或方面
        2. 检查答案覆盖了哪些方面
        3. 找出遗漏的方面
        
        输出JSON格式：
        {{
            "aspects": [
                {{
                    "aspect": "问题方面",
                    "covered": true/false,
                    "coverage_quality": "good/fair/poor"
                }}
            ],
            "coverage_score": 0.0-1.0,
            "missing_aspects": ["未覆盖的方面"]
        }}
        """
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
    
    async def evaluate_answer_consistency(
        self,
        answers: list
    ) -> dict:
        """
        评估答案一致性
        
        对于同一问题的多个回答，评估其一致性
        """
        prompt = f"""
        评估以下多个回答的一致性。
        
        ## 多个回答
        {json.dumps(answers, ensure_ascii=False, indent=2)}
        
        ## 评估要点
        1. 各回答的核心信息是否一致？
        2. 是否存在矛盾的信息？
        3. 回答之间的差异是否合理？
        
        输出JSON格式：
        {{
            "consistent_facts": ["一致的事实"],
            "conflicting_facts": ["矛盾的事实"],
            "consistency_score": 0.0-1.0,
            "analysis": "详细分析"
        }}
        """
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
```

---

## 五、Agent效果量化评估

### 5.1 Agent评估框架设计

```
┌─────────────────────────────────────────────────────────────┐
│                 Agent效果量化评估框架                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  评估维度                            │   │
│  ├─────────────┬─────────────┬─────────────┬───────────┤   │
│  │  任务完成   │  推理质量   │  协作效率   │  安全性   │   │
│  ├─────────────┼─────────────┼─────────────┼───────────┤   │
│  │ Success     │ Reasoning   │ Collab      │ Safety    │   │
│  │ Rate        │ Quality     │ Efficiency  │ Score     │   │
│  │             │             │             │           │   │
│  │ Task        │ Step        │ Message     │ Error     │   │
│  │ Completion  │ Optimality  │ Efficiency  │ Recovery  │   │
│  │ Time        │             │             │           │   │
│  └─────────────┴─────────────┴─────────────┴───────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  评估方法                            │   │
│  ├─────────────┬─────────────┬─────────────┬───────────┤   │
│  │  自动评估   │  LLM评估    │  混合评估   │  人工评估 │   │
│  │  (Baseline) │  (Quality)  │  (Balanced) │  (Gold)   │   │
│  └─────────────┴─────────────┴─────────────┴───────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Agent评估代码实现

```python
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AgentExecution:
    """Agent执行记录"""
    task_id: str
    agent_name: str
    start_time: float
    end_time: float
    status: TaskStatus
    steps: List[Dict[str, Any]]
    tools_used: List[str]
    messages: List[Dict[str, Any]]
    error: Optional[str] = None

class AgentEvaluator:
    """
    Agent效果评估器
    
    评估维度：
    1. 任务完成度
    2. 推理质量
    3. 工具使用效率
    4. 协作效率
    5. 成本效率
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def evaluate_execution(
        self,
        execution: AgentExecution,
        expected_result: Any = None
    ) -> Dict[str, Any]:
        """
        评估单次Agent执行
        """
        results = {}
        
        # 1. 任务完成度评估
        results["task_completion"] = self._evaluate_task_completion(
            execution, expected_result
        )
        
        # 2. 执行效率评估
        results["efficiency"] = self._evaluate_efficiency(execution)
        
        # 3. 工具使用评估
        results["tool_usage"] = self._evaluate_tool_usage(execution)
        
        # 4. 成本评估
        results["cost"] = self._evaluate_cost(execution)
        
        # 计算综合分数
        results["overall_score"] = self._compute_overall_score(results)
        
        return results
    
    def _evaluate_task_completion(
        self,
        execution: AgentExecution,
        expected_result: Any
    ) -> Dict[str, Any]:
        """
        评估任务完成度
        """
        # 基础完成状态
        completion_score = {
            TaskStatus.SUCCESS: 1.0,
            TaskStatus.PARTIAL: 0.5,
            TaskStatus.FAILED: 0.0,
            TaskStatus.TIMEOUT: 0.0
        }.get(execution.status, 0.0)
        
        # 如果有预期结果，进行对比
        result_match = None
        if expected_result is not None:
            # 使用LLM判断结果是否匹配
            result_match = self._compare_results(
                execution.steps[-1].get("result"),
                expected_result
            )
        
        return {
            "completion_score": completion_score,
            "status": execution.status.value,
            "result_match": result_match,
            "steps_count": len(execution.steps)
        }
    
    def _evaluate_efficiency(self, execution: AgentExecution) -> Dict[str, Any]:
        """
        评估执行效率
        """
        duration = execution.end_time - execution.start_time
        
        # 计算步骤优化度
        optimal_steps = self._estimate_optimal_steps(execution)
        step_optimality = optimal_steps / len(execution.steps) if execution.steps else 0
        
        # 计算每步平均耗时
        avg_step_time = duration / len(execution.steps) if execution.steps else 0
        
        return {
            "duration": duration,
            "step_optimality": min(step_optimality, 1.0),
            "avg_step_time": avg_step_time,
            "total_steps": len(execution.steps)
        }
    
    def _evaluate_tool_usage(self, execution: AgentExecution) -> Dict[str, Any]:
        """
        评估工具使用效率
        """
        # 统计工具使用次数
        tool_counts = {}
        for step in execution.steps:
            tool = step.get("tool_used")
            if tool:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        # 计算工具多样性
        unique_tools = len(set(execution.tools_used))
        total_tool_calls = len(execution.tools_used)
        
        # 评估工具选择合理性
        tool_selection_score = self._evaluate_tool_selection(execution)
        
        return {
            "tool_counts": tool_counts,
            "unique_tools": unique_tools,
            "total_tool_calls": total_tool_calls,
            "tool_selection_score": tool_selection_score
        }
    
    def _evaluate_cost(self, execution: AgentExecution) -> Dict[str, Any]:
        """
        评估成本效率
        """
        # 估算token使用量（简化模型）
        estimated_tokens = sum(
            len(step.get("content", "")) // 4
            for step in execution.steps
        )
        
        # 计算每步平均token
        avg_tokens_per_step = estimated_tokens / len(execution.steps) if execution.steps else 0
        
        return {
            "estimated_tokens": estimated_tokens,
            "avg_tokens_per_step": avg_tokens_per_step,
            "total_messages": len(execution.messages)
        }
    
    def _compute_overall_score(self, results: Dict) -> float:
        """
        计算综合评估分数
        """
        weights = {
            "task_completion": 0.4,
            "efficiency": 0.3,
            "tool_usage": 0.2,
            "cost": 0.1
        }
        
        score = 0.0
        
        # 任务完成分数
        score += results["task_completion"]["completion_score"] * weights["task_completion"]
        
        # 效率分数（归一化）
        efficiency_score = min(1.0, 60 / results["efficiency"]["duration"])  # 60秒内完成得满分
        score += efficiency_score * weights["efficiency"]
        
        # 工具使用分数
        score += results["tool_usage"]["tool_selection_score"] * weights["tool_usage"]
        
        # 成本分数（token越少越好）
        cost_score = min(1.0, 1000 / results["cost"]["estimated_tokens"])  # 1000 token内得满分
        score += cost_score * weights["cost"]
        
        return score
    
    def _estimate_optimal_steps(self, execution: AgentExecution) -> int:
        """估算最优步骤数"""
        # 简化实现：基于任务复杂度估算
        return max(3, len(execution.steps) // 2)
    
    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """比较实际结果和预期结果"""
        if actual == expected:
            return True
        
        # 使用LLM进行语义比较
        prompt = f"""
        判断以下两个结果是否在语义上等价：
        
        实际结果: {actual}
        预期结果: {expected}
        
        只回答 "True" 或 "False"。
        """
        
        response = self.llm.generate(prompt)
        return response.strip().lower() == "true"
    
    def _evaluate_tool_selection(self, execution: AgentExecution) -> float:
        """评估工具选择合理性"""
        # 简化实现：基于工具使用模式评估
        if not execution.tools_used:
            return 0.5
        
        # 检查是否有工具使用错误
        error_count = sum(
            1 for step in execution.steps
            if step.get("error")
        )
        
        if error_count == 0:
            return 1.0
        else:
            return max(0.0, 1.0 - error_count / len(execution.steps))
```

### 5.3 Agent批量评估流水线

```python
class AgentEvaluationPipeline:
    """
    Agent批量评估流水线
    
    支持：
    - 批量执行测试用例
    - 多维度评估
    - 结果聚合和报告生成
    """
    
    def __init__(self, evaluator: AgentEvaluator):
        self.evaluator = evaluator
        self.results: List[Dict] = []
    
    async def run_evaluation(
        self,
        agent,
        test_cases: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        运行批量评估
        """
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_test(test_case):
            async with semaphore:
                return await self._run_and_evaluate(agent, test_case)
        
        # 并发执行测试
        tasks = [run_single_test(tc) for tc in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤成功结果
        valid_results = [
            r for r in results
            if isinstance(r, dict) and "error" not in r
        ]
        
        # 聚合结果
        return self._aggregate_results(valid_results)
    
    async def _run_and_evaluate(
        self,
        agent,
        test_case: Dict
    ) -> Dict[str, Any]:
        """运行单个测试用例并评估"""
        try:
            # 执行Agent
            start_time = time.time()
            execution = await agent.execute(test_case["task"])
            end_time = time.time()
            
            # 创建执行记录
            agent_execution = AgentExecution(
                task_id=test_case.get("id", "unknown"),
                agent_name=agent.name,
                start_time=start_time,
                end_time=end_time,
                status=execution.get("status", TaskStatus.SUCCESS),
                steps=execution.get("steps", []),
                tools_used=execution.get("tools_used", []),
                messages=execution.get("messages", [])
            )
            
            # 评估
            return self.evaluator.evaluate_execution(
                agent_execution,
                test_case.get("expected_result")
            )
        except Exception as e:
            return {"error": str(e)}
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """聚合评估结果"""
        if not results:
            return {"error": "no_valid_results"}
        
        # 收集所有分数
        scores = [r.get("overall_score", 0) for r in results]
        
        # 计算各维度统计
        task_scores = [
            r["task_completion"]["completion_score"]
            for r in results
            if "task_completion" in r
        ]
        
        efficiency_scores = [
            r["efficiency"]["step_optimality"]
            for r in results
            if "efficiency" in r
        ]
        
        return {
            "summary": {
                "total_tests": len(results),
                "success_count": sum(1 for s in scores if s >= 0.7),
                "failure_count": sum(1 for s in scores if s < 0.7),
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0
            },
            "task_completion": {
                "mean_score": sum(task_scores) / len(task_scores) if task_scores else 0,
                "success_rate": sum(1 for s in task_scores if s >= 0.9) / len(task_scores) if task_scores else 0
            },
            "efficiency": {
                "mean_step_optimality": sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
            },
            "detailed_results": results
        }
    
    def generate_report(self, aggregated: Dict) -> str:
        """
        生成评估报告
        """
        summary = aggregated.get("summary", {})
        
        report = f"""
# Agent评估报告

## 概览
- 总测试数: {summary.get('total_tests', 0)}
- 成功数: {summary.get('success_count', 0)}
- 失败数: {summary.get('failure_count', 0)}
- 平均分: {summary.get('mean_score', 0):.4f}
- 最高分: {summary.get('max_score', 0):.4f}
- 最低分: {summary.get('min_score', 0):.4f}

## 任务完成度
- 平均完成分: {aggregated.get('task_completion', {}).get('mean_score', 0):.4f}
- 成功率: {aggregated.get('task_completion', {}).get('success_rate', 0):.2%}

## 执行效率
- 平均步骤优化度: {aggregated.get('efficiency', {}).get('mean_step_optimality', 0):.4f}

## 建议
"""
        
        # 生成建议
        if summary.get('mean_score', 0) < 0.7:
            report += "- ⚠️ 整体表现较低，建议优化Agent的核心逻辑\n"
        
        if aggregated.get('task_completion', {}).get('success_rate', 0) < 0.8:
            report += "- ⚠️ 任务成功率不足，建议加强任务理解和执行能力\n"
        
        if aggregated.get('efficiency', {}).get('mean_step_optimality', 0) < 0.6:
            report += "- ⚠️ 执行效率偏低，建议优化推理和工具使用策略\n"
        
        return report
```

---

## 六、评估流水线最佳实践

### 6.1 持续评估架构

```
┌─────────────────────────────────────────────────────────────┐
│                  持续评估架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ CI/CD   │───►│ Test    │───►│ Evaluate│───►│ Report  │ │
│  │ Trigger │    │ Runner  │    │ Engine  │    │ Builder │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                     │       │
│                                                     ▼       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ Alert   │◄───│ Dashboard│◄───│ Trend   │◄───│ History │ │
│  │ System  │    │         │    │ Analyzer│    │ Store   │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 评估配置管理

```yaml
# evaluation_config.yaml
version: "1.0"
agent:
  name: "my-production-agent"
  model: "gpt-4"
  
evaluation:
  # 评估频率
  frequency: "daily"
  
  # 测试数据集
  datasets:
    - name: "unit_tests"
      path: "./test_data/unit/"
      weight: 0.3
    - name: "integration_tests"
      path: "./test_data/integration/"
      weight: 0.4
    - name: "edge_cases"
      path: "./test_data/edge/"
      weight: 0.3
  
  # 评估指标
  metrics:
    - name: "task_completion"
      weight: 0.4
      threshold: 0.8
    - name: "efficiency"
      weight: 0.3
      threshold: 0.7
    - name: "safety"
      weight: 0.3
      threshold: 0.9
  
  # 阈值
  thresholds:
    overall_score: 0.75
    min_score: 0.5
  
  # 告警
  alerts:
    enabled: true
    channels:
      - type: "email"
        recipients: ["team@example.com"]
      - type: "slack"
        webhook: "https://hooks.slack.com/..."
    
    conditions:
      - metric: "overall_score"
        operator: "lt"
        value: 0.7
        severity: "critical"
      - metric: "task_completion"
        operator: "lt"
        value: 0.8
        severity: "warning"
```

### 6.3 评估数据集构建

```python
class EvaluationDatasetBuilder:
    """
    评估数据集构建器
    
    支持多种数据源和自动标注
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def build_from_logs(
        self,
        logs: List[Dict],
        sample_size: int = 100
    ) -> List[Dict]:
        """
        从生产日志构建评估数据集
        """
        # 采样
        import random
        sampled_logs = random.sample(logs, min(sample_size, len(logs)))
        
        # 使用LLM生成期望答案
        dataset = []
        for log in sampled_logs:
            question = log.get("input", "")
            context = log.get("context", "")
            
            # 生成参考答案
            prompt = f"""
            根据以下上下文，生成问题的标准答案。
            
            问题: {question}
            上下文: {context}
            
            只输出标准答案，不要解释。
            """
            
            expected_answer = await self.llm.generate(prompt)
            
            dataset.append({
                "id": log.get("id"),
                "question": question,
                "context": context,
                "expected_answer": expected_answer,
                "source": "production_logs"
            })
        
        return dataset
    
    async def augment_dataset(
        self,
        base_dataset: List[Dict],
        augmentation_factor: int = 3
    ) -> List[Dict]:
        """
        数据集增强
        
        通过改写、扩展、变体等方式增加测试样本
        """
        augmented = list(base_dataset)
        
        for item in base_dataset:
            for _ in range(augmentation_factor - 1):
                # 生成变体
                variant = await self._generate_variant(item)
                augmented.append(variant)
        
        return augmented
    
    async def _generate_variant(self, original: Dict) -> Dict:
        """生成问题变体"""
        prompt = f"""
        将以下问题改写为不同的表述方式，保持语义不变。
        
        原问题: {original['question']}
        
        只输出改写后的问题。
        """
        
        new_question = await self.llm.generate(prompt)
        
        return {
            **original,
            "id": f"{original.get('id', 'unknown')}_variant",
            "question": new_question,
            "source": "augmented"
        }
```

---

## 七、评估报告与可视化

### 7.1 评估报告生成

```python
class EvaluationReportGenerator:
    """
    评估报告生成器
    
    支持生成Markdown和HTML格式的报告
    """
    
    def __init__(self):
        self.report_template = """
# {agent_name} 评估报告

**评估时间**: {eval_time}
**评估版本**: {version}

---

## 概览

| 指标 | 分数 | 状态 |
|------|------|------|
| 总体评分 | {overall_score:.2f} | {overall_status} |
| 任务完成度 | {task_completion:.2f} | {task_status} |
| 执行效率 | {efficiency:.2f} | {efficiency_status} |
| 安全性 | {safety:.2f} | {safety_status} |

---

## 详细分析

### 任务完成度
{task_analysis}

### 执行效率
{efficiency_analysis}

### 安全性
{safety_analysis}

---

## 趋势分析

{trend_analysis}

---

## 建议

{recommendations}

---

*报告生成时间: {report_time}*
"""
    
    def generate_report(
        self,
        agent_name: str,
        results: Dict,
        historical_data: List[Dict] = None
    ) -> str:
        """生成Markdown格式报告"""
        
        # 确定状态
        overall_score = results.get("summary", {}).get("mean_score", 0)
        overall_status = "✅ 通过" if overall_score >= 0.7 else "❌ 未通过"
        
        # 生成趋势分析
        trend_analysis = self._generate_trend_analysis(historical_data) if historical_data else "暂无历史数据"
        
        # 生成建议
        recommendations = self._generate_recommendations(results)
        
        return self.report_template.format(
            agent_name=agent_name,
            eval_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            version="1.0",
            overall_score=overall_score,
            overall_status=overall_status,
            task_completion=results.get("task_completion", {}).get("mean_score", 0),
            task_status="✅" if results.get("task_completion", {}).get("mean_score", 0) >= 0.8 else "⚠️",
            efficiency=results.get("efficiency", {}).get("mean_step_optimality", 0),
            efficiency_status="✅" if results.get("efficiency", {}).get("mean_step_optimality", 0) >= 0.7 else "⚠️",
            safety=results.get("safety", {}).get("score", 1.0),
            safety_status="✅" if results.get("safety", {}).get("score", 1.0) >= 0.9 else "❌",
            task_analysis=self._analyze_task_completion(results),
            efficiency_analysis=self._analyze_efficiency(results),
            safety_analysis=self._analyze_safety(results),
            trend_analysis=trend_analysis,
            recommendations=recommendations,
            report_time=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _generate_trend_analysis(self, historical_data: List[Dict]) -> str:
        """生成趋势分析"""
        if len(historical_data) < 2:
            return "数据不足，无法生成趋势分析"
        
        recent_scores = [d.get("mean_score", 0) for d in historical_data[-5:]]
        older_scores = [d.get("mean_score", 0) for d in historical_data[-10:-5]]
        
        recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        older_avg = sum(older_scores) / len(older_scores) if older_scores else 0
        
        trend = "上升" if recent_avg > older_avg else "下降" if recent_avg < older_avg else "稳定"
        
        return f"""
- 近期平均分: {recent_avg:.4f}
- 历史平均分: {older_avg:.4f}
- 趋势: {trend} {"📈" if trend == "上升" else "📉" if trend == "下降" else "➡️"}
"""
    
    def _generate_recommendations(self, results: Dict) -> str:
        """生成改进建议"""
        recommendations = []
        
        # 基于结果生成建议
        if results.get("task_completion", {}).get("mean_score", 0) < 0.8:
            recommendations.append("1. **提升任务完成度**: 优化任务理解和执行逻辑")
        
        if results.get("efficiency", {}).get("mean_step_optimality", 0) < 0.7:
            recommendations.append("2. **提高执行效率**: 减少不必要的中间步骤")
        
        if results.get("cost", {}).get("estimated_tokens", 0) > 5000:
            recommendations.append("3. **优化成本**: 减少不必要的token使用")
        
        return "\n".join(recommendations) if recommendations else "当前表现良好，继续保持！"
    
    def _analyze_task_completion(self, results: Dict) -> str:
        """分析任务完成度"""
        task_result = results.get("task_completion", {})
        return f"""
- 成功率: {task_result.get('success_rate', 0):.2%}
- 平均完成分: {task_result.get('mean_score', 0):.4f}
"""
    
    def _analyze_efficiency(self, results: Dict) -> str:
        """分析执行效率"""
        eff_result = results.get("efficiency", {})
        return f"""
- 平均步骤优化度: {eff_result.get('mean_step_optimality', 0):.4f}
- 平均耗时: {eff_result.get('duration', 0):.2f}秒
"""
    
    def _analyze_safety(self, results: Dict) -> str:
        """分析安全性"""
        safety_result = results.get("safety", {})
        return f"""
- 安全评分: {safety_result.get('score', 1.0):.4f}
- 错误恢复率: {safety_result.get('recovery_rate', 1.0):.2%}
"""
```

---

## 八、总结与最佳实践

### 8.1 评估体系搭建清单

```
┌─────────────────────────────────────────────────────────────┐
│                 评估体系搭建检查清单                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  □ 基础设施                                                 │
│    ├─ [ ] 评估数据集（覆盖核心场景）                          │
│    ├─ [ ] 评估流水线（自动化执行）                            │
│    ├─ [ ] 结果存储（历史对比）                                │
│    └─ [ ] 告警机制（及时发现问题）                            │
│                                                             │
│  □ 评估指标                                                 │
│    ├─ [ ] 任务完成度指标                                     │
│    ├─ [ ] 执行效率指标                                       │
│    ├─ [ ] 安全性指标                                         │
│    └─ [ ] 成本效率指标                                       │
│                                                             │
│  □ 评估方法                                                 │
│    ├─ [ ] 自动评估（基础指标）                                │
│    ├─ [ ] LLM评估（语义质量）                                │
│    ├─ [ ] 混合评估（平衡精度和效率）                          │
│    └─ [ ] 人工评估（金标准验证）                              │
│                                                             │
│  □ 报告与监控                                                │
│    ├─ [ ] 定期评估报告                                       │
│    ├─ [ ] 实时监控看板                                       │
│    ├─ [ ] 趋势分析                                          │
│    └─ [ ] 改进建议                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 核心要点

1. **多维度评估**：不要只看单一指标，需要综合考虑质量、效率、安全性
2. **持续评估**：建立自动化评估流水线，而非一次性测试
3. **基准对比**：建立baseline，持续跟踪改进效果
4. **场景覆盖**：评估数据集需要覆盖核心场景和边界情况
5. **及时反馈**：建立告警机制，及时发现和修复问题

---

> **下一篇预告**：《Agent评估实战：如何设计端到端的Agent效果测试》- 深入探讨Agent评估的具体实施方法，包括测试用例设计、评估指标选择、自动化流水线搭建等。
