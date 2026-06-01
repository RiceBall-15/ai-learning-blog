---
title: "LLM级联推理架构：智能路由与成本优化的生产级方案"
description: "深度解析LLM级联推理架构设计，涵盖模型路由策略、质量评估机制、成本优化方案与生产级降级策略，附完整架构图与代码示例"
date: 2026-06-01
author: "RiceBall"
category: "architecture"
subCategory: "distributed"
tags: ["LLM", "级联推理", "模型路由", "成本优化", "推理架构", "Mixture of Agents", "智能调度"]
draft: false
---

# LLM级联推理架构：智能路由与成本优化的生产级方案

## 一个真实的成本困境

假设你运营着一个日均处理 500 万次 LLM 请求的客服系统。经过分析，你发现一个令人震惊的事实：

```
┌────────────────────────────────────────────────────────────┐
│              请求复杂度分布（真实数据）                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  简单查询（"你好"、"退款政策"、FAQ类）          45%          │
│  中等任务（多轮对话、简单推理、摘要）           35%          │
│  复杂推理（长文分析、数学证明、代码生成）        20%          │
│                                                            │
│  但所有请求都在用 GPT-4 级别模型处理！                       │
│                                                            │
│  月度成本：~$150,000                                        │
│  理论最低成本：~$35,000（如果按需路由）                       │
│  浪费比例：77%                                              │
└────────────────────────────────────────────────────────────┘
```

这就是 **LLM级联推理架构（Cascade Inference）** 要解决的核心问题：**不是所有请求都需要最强的模型，也不是所有请求都需要最便宜的模型——而是需要"刚好够用"的模型。**

## 什么是级联推理？

级联推理的核心思想很简单：**用小模型先尝试，搞不定再升级到大模型。**

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM 级联推理 vs 传统模式                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传统模式（一刀切）                                               │
│  ┌─────────┐                                                     │
│  │  请求    │ ──→ GPT-4 处理一切 ──→ 响应                        │
│  └─────────┘    成本高，延迟高，但质量稳定                          │
│                                                                  │
│  级联模式（智能路由）                                              │
│  ┌─────────┐                                                     │
│  │  请求    │ ──→ 路由器 ──┬──→ 小模型（45%请求，0.02x成本）       │
│  └─────────┘              ├──→ 中模型（35%请求，0.1x成本）         │
│                           └──→ 大模型（20%请求，1.0x成本）         │
│                                                                  │
│  关键区别：不是简单的 if-else，而是基于请求语义的动态路由            │
└──────────────────────────────────────────────────────────────────┘
```

但实际实现远比这复杂。真正的级联推理需要解决三个核心问题：

1. **路由器怎么选**：用什么模型做分类器？延迟开销多少？
2. **阈值怎么定**：什么程度的"搞不定"才需要升级？
3. **质量怎么保**：如何确保级联后的整体质量不低于单一模型？

## 架构设计：三层级联推理系统

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM 级联推理架构（生产级）                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐              │
│  │  API网关  │───→│  智能路由器   │───→│  模型池管理器  │              │
│  └──────────┘    └──────┬───────┘    └───────┬───────┘              │
│                         │                    │                      │
│                  ┌──────┴───────┐     ┌──────┴───────┐              │
│                  │ 请求分析器    │     │  质量评估器   │              │
│                  │  - 复杂度预估 │     │  - 输出校验   │              │
│                  │  - 领域识别   │     │  - 置信度检测 │              │
│                  │  - Token预估  │     │  - 一致性比对 │              │
│                  └──────────────┘     └──────────────┘              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      模型池（Model Pool）                    │    │
│  │                                                             │    │
│  │   Tier 1: 轻量模型          Tier 2: 中等模型       Tier 3:  │    │
│  │   ┌──────────────┐         ┌──────────────┐      ┌───────┐ │    │
│  │   │ GPT-4o-mini  │         │ GPT-4o       │      │GPT-4  │ │    │
│  │   │ Qwen2.5-7B   │         │ Claude Sonnet │      │Claude │ │    │
│  │   │ Llama3.1-8B  │         │ Qwen2.5-72B  │      │Opus   │ │    │
│  │   └──────────────┘         └──────────────┘      └───────┘ │    │
│  │   成本: 0.02x              成本: 0.1x           成本: 1.0x  │    │
│  │   延迟: 200-500ms          延迟: 500-1500ms     延迟: 2-8s  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    监控与反馈系统                             │    │
│  │  - 路由准确率  |  - 质量评分  |  - 成本追踪  |  - 延迟分析  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 智能路由器：级联推理的大脑

路由器是整个系统的核心。它需要在 **极低延迟**（<50ms）内做出准确的路由决策。

```python
from dataclasses import dataclass
from enum import Enum
import tiktoken

class ComplexityLevel(Enum):
    """请求复杂度等级"""
    SIMPLE = "simple"       # FAQ、简单问答、格式转换
    MODERATE = "moderate"   # 多轮推理、摘要、简单代码
    COMPLEX = "complex"     # 长文分析、数学推理、复杂代码

@dataclass
class RoutingDecision:
    """路由决策结果"""
    tier: int                    # 目标模型层级 (1/2/3)
    model: str                   # 具体模型名
    confidence: float            # 路由置信度
    estimated_tokens: int        # 预估输出Token数
    fallback_tier: int           # 降级目标层级
    reasoning: str               # 路由理由（用于调试）

class CascadeRouter:
    """级联推理路由器"""
    
    # 基于规则 + 语义的混合路由策略
    COMPLEXITY_KEYWORDS = {
        ComplexityLevel.COMPLEX: [
            "分析这段代码", "证明", "推导", "对比分析",
            "写一篇", "详细解释", "架构设计", "方案评估"
        ],
        ComplexityLevel.MODERATE: [
            "总结", "翻译", "改写", "解释", "帮我写",
            "列举", "比较", "建议"
        ],
    }
    
    def __init__(self, max_simple_tokens: int = 512):
        self.max_simple_tokens = max_simple_tokens
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def route(self, messages: list[dict], metadata: dict = None) -> RoutingDecision:
        """
        核心路由逻辑：分析请求特征，选择最佳模型层级
        
        策略：先用规则快速过滤，再用特征打分精细分类
        """
        # Step 1: 快速特征提取（<5ms）
        features = self._extract_features(messages, metadata)
        
        # Step 2: 复杂度评分（<10ms）
        score = self._compute_complexity_score(features)
        
        # Step 3: 映射到模型层级
        tier, model = self._score_to_tier(score)
        
        return RoutingDecision(
            tier=tier,
            model=model,
            confidence=min(score / 100, 1.0),
            estimated_tokens=features["estimated_output_tokens"],
            fallback_tier=min(tier + 1, 3),
            reasoning=f"score={score:.1f}, tokens_est={features['estimated_output_tokens']}"
        )
    
    def _extract_features(self, messages: list[dict], metadata: dict) -> dict:
        """从请求中提取路由特征"""
        user_text = " ".join(
            m["content"] for m in messages if m["role"] == "user"
        )
        token_count = len(self.encoder.encode(user_text))
        
        return {
            "token_count": token_count,
            "message_count": len(messages),
            "has_system_prompt": any(m["role"] == "system" for m in messages),
            "keyword_complexity": self._keyword_score(user_text),
            "estimated_output_tokens": min(token_count * 2, 4096),
            "is_multi_turn": len(messages) > 2,
        }
    
    def _keyword_score(self, text: str) -> float:
        """基于关键词的复杂度评分"""
        score = 0.0
        for level, keywords in self.COMPLEXITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    if level == ComplexityLevel.COMPLEX:
                        score += 30
                    elif level == ComplexityLevel.MODERATE:
                        score += 10
        return min(score, 100)
    
    def _compute_complexity_score(self, features: dict) -> float:
        """综合复杂度评分（0-100）"""
        score = 0.0
        
        # Token数量权重 (40%)
        if features["token_count"] > 2000:
            score += 35
        elif features["token_count"] > 500:
            score += 20
        else:
            score += 5
        
        # 关键词复杂度 (30%)
        score += features["keyword_complexity"] * 0.3
        
        # 多轮对话 (15%)
        if features["is_multi_turn"]:
            score += min(features["message_count"] * 3, 15)
        
        # 系统提示复杂度 (15%)
        if features["has_system_prompt"]:
            score += 10
        
        return min(score, 100)
    
    def _score_to_tier(self, score: float) -> tuple[int, str]:
        """将评分映射到模型层级"""
        if score < 25:
            return 1, "gpt-4o-mini"
        elif score < 60:
            return 2, "gpt-4o"
        else:
            return 3, "gpt-4"
```

### 质量评估与自动升级

路由不是单向的——如果低级模型的输出质量不够，系统需要自动升级到更高级的模型：

```python
@dataclass
class QualityCheckResult:
    """质量评估结果"""
    passed: bool
    score: float           # 0-100
    should_upgrade: bool
    issues: list[str]

class QualityGate:
    """质量门控：决定是否需要升级到更高级模型"""
    
    # 各层级的质量阈值
    QUALITY_THRESHOLDS = {
        1: {"min_score": 70, "max_retries": 0},   # 小模型：不重试，直接升级
        2: {"min_score": 80, "max_retries": 1},    # 中模型：允许1次重试
        3: {"min_score": 85, "max_retries": 2},    # 大模型：允许2次重试
    }
    
    def evaluate(
        self, 
        input_messages: list[dict], 
        output: str, 
        tier: int,
        routing_decision: RoutingDecision
    ) -> QualityCheckResult:
        """
        评估输出质量，决定是否需要升级
        
        评估维度：
        1. 完整性：输出是否完整（没有截断）
        2. 相关性：输出是否与输入相关
        3. 一致性：输出是否自洽
        4. 格式：输出是否符合预期格式
        """
        issues = []
        score = 100.0
        
        # 1. 完整性检查
        if output.endswith("...") or output.endswith("…"):
            issues.append("output_truncated")
            score -= 30
        
        # 2. 长度检查（太短可能意味着理解不足）
        if len(output) < 50 and len(input_messages[-1]["content"]) > 200:
            issues.append("output_too_short")
            score -= 20
        
        # 3. 关键词覆盖检查
        input_text = input_messages[-1]["content"]
        if self._has_question(input_text):
            # 如果输入是问题，检查输出是否包含相关回答
            if not self._output_addresses_question(input_text, output):
                issues.append("output_not_addressing_question")
                score -= 25
        
        # 4. 一致性检查（重复内容检测）
        if self._has_repetition(output):
            issues.append("output_repetitive")
            score -= 15
        
        threshold = self.QUALITY_THRESHOLDS[tier]
        passed = score >= threshold["min_score"]
        
        # 决定是否升级
        should_upgrade = (
            not passed and 
            tier < 3  # 已经是最高等级，无法再升级
        )
        
        return QualityCheckResult(
            passed=passed,
            score=score,
            should_upgrade=should_upgrade,
            issues=issues
        )
    
    def _has_question(self, text: str) -> bool:
        return "?" in text or "？" in text or "什么" in text or "如何" in text
    
    def _output_addresses_question(self, question: str, answer: str) -> bool:
        """简单的相关性检查：关键词重叠率"""
        q_words = set(question.split())
        a_words = set(answer.split())
        overlap = len(q_words & a_words) / max(len(q_words), 1)
        return overlap > 0.15
    
    def _has_repetition(self, text: str) -> bool:
        """检测重复内容"""
        sentences = text.split("。")
        if len(sentences) < 3:
            return False
        # 检查连续句子是否高度相似
        for i in range(len(sentences) - 1):
            if len(sentences[i]) > 10 and sentences[i] == sentences[i+1]:
                return True
        return False
```

### 级联推理执行器

将路由器、模型池和质量门控组合成完整的级联推理系统：

```python
import asyncio
import time
from typing import AsyncGenerator

class CascadeInferenceEngine:
    """LLM级联推理引擎"""
    
    # 模型配置
    MODEL_CONFIG = {
        1: {"model": "gpt-4o-mini", "max_tokens": 4096, "cost_per_1k": 0.00015},
        2: {"model": "gpt-4o", "max_tokens": 8192, "cost_per_1k": 0.005},
        3: {"model": "gpt-4", "max_tokens": 16384, "cost_per_1k": 0.03},
    }
    
    def __init__(self, llm_client):
        self.router = CascadeRouter()
        self.quality_gate = QualityGate()
        self.llm = llm_client
        self.metrics = MetricsCollector()
    
    async def infer(
        self, 
        messages: list[dict],
        temperature: float = 0.7,
        max_retries: int = 2
    ) -> dict:
        """
        执行级联推理
        
        流程：路由 → 调用模型 → 质量评估 → 必要时升级 → 返回结果
        """
        start_time = time.time()
        
        # Step 1: 路由决策
        decision = self.router.route(messages)
        self.metrics.record_routing(decision)
        
        current_tier = decision.tier
        output = None
        quality_result = None
        
        for attempt in range(max_retries + 1):
            # Step 2: 调用当前层级模型
            config = self.MODEL_CONFIG[current_tier]
            output = await self._call_model(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=temperature,
            )
            
            # Step 3: 质量评估
            quality_result = self.quality_gate.evaluate(
                messages, output, current_tier, decision
            )
            
            self.metrics.record_quality(
                tier=current_tier,
                score=quality_result.score,
                passed=quality_result.passed
            )
            
            if quality_result.passed:
                break
            
            # Step 4: 升级到更高级模型
            if quality_result.should_upgrade:
                current_tier += 1
                self.metrics.record_upgrade(
                    from_tier=decision.tier,
                    to_tier=current_tier,
                    reason=quality_result.issues
                )
                continue
            
            # 当前层级的重试
            if attempt < max_retries:
                continue
        
        elapsed = time.time() - start_time
        cost = self._compute_cost(
            config["cost_per_1k"], 
            output
        )
        
        return {
            "output": output,
            "meta": {
                "initial_tier": decision.tier,
                "final_tier": current_tier,
                "model": self.MODEL_CONFIG[current_tier]["model"],
                "upgraded": current_tier != decision.tier,
                "quality_score": quality_result.score if quality_result else 0,
                "latency_ms": int(elapsed * 1000),
                "estimated_cost": cost,
                "routing_confidence": decision.confidence,
            }
        }
    
    async def infer_stream(
        self, 
        messages: list[dict],
        temperature: float = 0.7
    ) -> AsyncGenerator[dict, None]:
        """
        流式级联推理：先快速路由，流式输出，实时质量监控
        
        这种模式适合交互式场景（如聊天机器人）
        """
        decision = self.router.route(messages)
        current_tier = decision.tier
        
        yield {"type": "routing", "tier": current_tier, "model": self.MODEL_CONFIG[current_tier]["model"]}
        
        full_output = ""
        buffer = ""
        
        async for chunk in self.llm.stream(
            model=self.MODEL_CONFIG[current_tier]["model"],
            messages=messages,
            max_tokens=self.MODEL_CONFIG[current_tier]["max_tokens"],
        ):
            full_output += chunk
            buffer += chunk
            
            # 每100个token检查一次质量
            if len(full_output) % 100 == 0:
                if self._detect_early_quality_issue(buffer):
                    # 发现质量问题，准备升级
                    yield {"type": "quality_warning", "message": "检测到可能的质量问题，准备升级模型"}
                    # 在流式场景中，我们记录这个事件但不中断当前流
                    # 等当前流结束后再决定是否重试
                    break
            
            yield {"type": "chunk", "content": chunk}
        
        # 流式结束后进行最终质量评估
        quality_result = self.quality_gate.evaluate(messages, full_output, current_tier, decision)
        
        if quality_result.should_upgrade:
            # 升级并重新生成（非流式，因为已经中断了）
            upgraded_result = await self.infer(messages, temperature)
            yield {"type": "upgraded", "output": upgraded_result["output"], "meta": upgraded_result["meta"]}
        
        yield {"type": "done", "final_tier": current_tier}
    
    async def _call_model(self, model: str, messages: list[dict], **kwargs) -> str:
        """调用LLM模型"""
        response = await self.llm.chat(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content
    
    def _detect_early_quality_issue(self, text: str) -> bool:
        """早期质量问题检测"""
        # 检测明显的幻觉特征
        hallucination_signals = [
            "根据我的训练数据",
            "作为AI",
            "我没有最新的",
            "截至我的知识截止",
        ]
        return any(signal in text for signal in hallucination_signals)
    
    def _compute_cost(self, cost_per_1k: float, output: str) -> float:
        """计算单次推理成本"""
        import tiktoken
        encoder = tiktoken.encoding_for_model("gpt-4")
        output_tokens = len(encoder.encode(output))
        return (output_tokens / 1000) * cost_per_1k


class MetricsCollector:
    """级联推理指标收集器"""
    
    def __init__(self):
        self.routing_stats = []
        self.quality_stats = []
        self.upgrade_stats = []
    
    def record_routing(self, decision: RoutingDecision):
        self.routing_stats.append({
            "tier": decision.tier,
            "confidence": decision.confidence,
            "timestamp": time.time(),
        })
    
    def record_quality(self, tier: int, score: float, passed: bool):
        self.quality_stats.append({
            "tier": tier,
            "score": score,
            "passed": passed,
            "timestamp": time.time(),
        })
    
    def record_upgrade(self, from_tier: int, to_tier: int, reason: list[str]):
        self.upgrade_stats.append({
            "from": from_tier,
            "to": to_tier,
            "reason": reason,
            "timestamp": time.time(),
        })
    
    def get_summary(self) -> dict:
        """生成汇总报告"""
        total = len(self.routing_stats)
        if total == 0:
            return {}
        
        tier_distribution = {}
        for stat in self.routing_stats:
            tier = stat["tier"]
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        
        upgrade_rate = len(self.upgrade_stats) / total * 100
        avg_quality = (
            sum(s["score"] for s in self.quality_stats) / len(self.quality_stats)
            if self.quality_stats else 0
        )
        
        return {
            "total_requests": total,
            "tier_distribution": {f"tier_{k}": f"{v/total*100:.1f}%" for k, v in tier_distribution.items()},
            "upgrade_rate": f"{upgrade_rate:.1f}%",
            "average_quality_score": f"{avg_quality:.1f}",
        }
```

## 生产环境中的关键挑战

### 挑战一：路由延迟开销

路由本身也需要时间。如果路由器太慢，级联推理的延迟优势就会被抵消。

```
┌────────────────────────────────────────────────────────────────┐
│                  路由延迟预算分析                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  总预算: 500ms（用户可接受的最大延迟）                            │
│                                                                │
│  路由阶段:                                                      │
│  ┌─────────────────────────────────────┐                       │
│  │ 规则匹配:      1-3ms     ✓ 极快      │                       │
│  │ Token计数:     2-5ms     ✓ 极快      │                       │
│  │ 关键词匹配:    3-8ms     ✓ 可接受     │                       │
│  │ 语义嵌入:      20-80ms   ⚠ 有风险     │                       │
│  │ LLM分类器:     100-500ms ✗ 太慢       │                       │
│  └─────────────────────────────────────┘                       │
│                                                                │
│  推荐方案：规则 + 轻量特征（<10ms路由开销）                       │
│  仅在高价值场景使用LLM分类器（离线或缓存）                        │
└────────────────────────────────────────────────────────────────┘
```

**最佳实践：混合路由策略**

```python
class HybridRouter:
    """混合路由策略：规则优先，模型兜底"""
    
    def __init__(self):
        self.rule_router = RuleBasedRouter()     # 规则路由：<5ms
        self.model_router = ModelBasedRouter()   # 模型路由：50-200ms
        self.cache = LRUCache(maxsize=10000)     # 缓存热点路由决策
    
    def route(self, messages: list[dict]) -> RoutingDecision:
        # 1. 检查缓存
        cache_key = self._compute_cache_key(messages)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 2. 规则路由（快速路径）
        rule_decision = self.rule_router.route(messages)
        
        # 3. 如果规则置信度高，直接使用
        if rule_decision.confidence > 0.85:
            self.cache[cache_key] = rule_decision
            return rule_decision
        
        # 4. 置信度不够，使用模型路由（但只在非实时场景）
        # 对于实时场景，我们接受规则路由的结果
        model_decision = self.model_router.route(messages)
        
        # 5. 取置信度更高的结果
        final = model_decision if model_decision.confidence > rule_decision.confidence else rule_decision
        self.cache[cache_key] = final
        
        return final
```

### 挑战二：降级策略

当高级模型不可用或超时时，如何优雅降级：

```
┌──────────────────────────────────────────────────────────────┐
│                    降级策略矩阵                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  场景                    降级策略              用户感知        │
│  ─────────────────────────────────────────────────────────   │
│  Tier 3 超时             → Tier 2 + 重试       轻微延迟增加    │
│  Tier 3 超时 + Tier 2    → Tier 2 + 精简prompt 响应可能略短   │
│    也超时                                         但完整        │
│  全部超时                → 缓存最近结果         返回缓存内容    │
│                           + 提示"稍后重试"      + 重试建议      │
│  质量不达标              → 同层级重试 → 升级    可能稍慢        │
│  API 限流               → 路由到其他Provider    几乎无感        │
│                                                              │
│  关键原则：                                                    │
│  1. 永远不要返回空白或错误给用户                                │
│  2. 降级时优先保证完整性，其次保证质量                           │
│  3. 所有降级路径必须有监控和告警                                │
└──────────────────────────────────────────────────────────────┘
```

```python
class GracefulDegradation:
    """优雅降级策略"""
    
    async def infer_with_fallback(
        self, 
        messages: list[dict],
        decision: RoutingDecision
    ) -> dict:
        """带降级的推理"""
        
        strategies = [
            # 策略1: 按路由结果调用
            (decision.tier, messages),
            # 策略2: 降级到下一层级
            (min(decision.tier + 1, 3), messages),
            # 策略3: 使用精简prompt调用中等模型
            (2, self._simplify_messages(messages)),
            # 策略4: 返回缓存的类似响应
            (None, None),  # 特殊标记
        ]
        
        for tier, msgs in strategies:
            if tier is None:
                # 缓存降级
                cached = await self._get_cached_response(messages)
                if cached:
                    return {
                        "output": cached["response"],
                        "meta": {
                            "degradation_level": "cache_fallback",
                            "warning": "返回缓存响应，请稍后重试获取最新结果",
                        }
                    }
                continue
            
            try:
                config = self.MODEL_CONFIG[tier]
                result = await asyncio.wait_for(
                    self._call_model(config["model"], msgs),
                    timeout=config.get("timeout", 30)
                )
                if result:
                    return {
                        "output": result,
                        "meta": {
                            "degradation_level": f"tier_{tier}",
                            "original_tier": decision.tier,
                        }
                    }
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics.record_error(tier, str(e))
                continue
        
        # 所有策略都失败
        return {
            "output": "系统暂时繁忙，请稍后重试。您的问题已记录，我们会尽快处理。",
            "meta": {"degradation_level": "full_fallback"},
        }
    
    def _simplify_messages(self, messages: list[dict]) -> list[dict]:
        """精简prompt以适配更低级模型"""
        simplified = []
        for msg in messages:
            if msg["role"] == "system":
                # 精简系统提示
                simplified.append({
                    "role": "system",
                    "content": "You are a helpful assistant. Answer concisely."
                })
            else:
                # 对用户消息进行摘要（如果太长）
                if len(msg["content"]) > 1000:
                    simplified.append({
                        "role": msg["role"],
                        "content": msg["content"][:1000] + "\n\nPlease address the key points above."
                    })
                else:
                    simplified.append(msg)
        return simplified
```

### 挑战三：成本-质量权衡

级联推理不是万能的——升级过程本身会增加成本和延迟。关键是找到 **盈亏平衡点**：

```
┌────────────────────────────────────────────────────────────────────┐
│               成本-质量权衡分析（真实场景数据）                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  场景A: 客服FAQ系统                                                │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  单一GPT-4:     成本$150K/月  |  质量95分  |  延迟2.1s │         │
│  │  级联推理:       成本$42K/月   |  质量91分  |  延迟1.3s  │         │
│  │  节省:           72%          |  质量损失4分 | 延迟-38%  │         │
│  │  结论: ✅ 非常适合（高ROI）                                │
│  └──────────────────────────────────────────────────────┘         │
│                                                                    │
│  场景B: 代码审查助手                                               │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  单一GPT-4:     成本$80K/月   |  质量88分  |  延迟3.5s  │         │
│  │  级联推理:       成本$55K/月   |  质量82分  |  延迟2.8s  │         │
│  │  节省:           31%          |  质量损失6分 | 延迟-20%  │         │
│  │  结论: ⚠ 谨慎使用（质量敏感场景）                         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                    │
│  场景C: 医疗问答系统                                               │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  单一GPT-4:     成本$200K/月  |  质量99分  |  延迟4.2s  │         │
│  │  级联推理:       成本$185K/月  |  质量94分  |  延迟3.1s  │         │
│  │  节省:           7.5%         |  质量损失5分 | 延迟-26%  │         │
│  │  结论: ❌ 不适合（安全关键，质量零容忍）                     │
│  └──────────────────────────────────────────────────────┘         │
│                                                                    │
│  经验法则:                                                          │
│  • 质量损失 < 3分 且 节省 > 40% → 强烈推荐                         │
│  • 质量损失 < 5分 且 节省 > 25% → 可以使用                         │
│  • 质量损失 > 5分 或 安全关键场景 → 不推荐                          │
└────────────────────────────────────────────────────────────────────┘
```

## 高级模式：Mixture of Agents 级联

除了简单的"小→大"级联，还有一种更强大的模式：**多模型并行投票**。

```
┌──────────────────────────────────────────────────────────────────┐
│              Mixture of Agents 级联模式                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入: "请分析这段代码的时间复杂度并优化"                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  并行调用3个模型（在中等复杂度场景）                        │     │
│  │                                                          │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │     │
│  │  │ GPT-4o   │  │ Claude   │  │ Gemini   │              │     │
│  │  │ mini     │  │ Sonnet   │  │ 2.0 Flash│              │     │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │     │
│  │       │              │              │                   │     │
│  │       └──────────┬───┘──────────────┘                   │     │
│  │                  ▼                                      │     │
│  │         ┌────────────────┐                              │     │
│  │         │  一致性聚合器   │                              │     │
│  │         │  - 投票机制     │                              │     │
│  │         │  - 差异检测     │                              │     │
│  │         │  - 置信度评估   │                              │     │
│  │         └────────┬───────┘                              │     │
│  │                  │                                      │     │
│  │                  ▼                                      │     │
│  │         ┌────────────────┐                              │     │
│  │         │ 最终输出/升级判断│                              │     │
│  │         └────────────────┘                              │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  优势: 3个廉价模型并行 = 接近大模型质量，成本约0.3x               │
│  适用: 中等复杂度、需要高可靠性的场景                              │
└──────────────────────────────────────────────────────────────────┘
```

```python
class MoACascadeEngine:
    """Mixture of Agents 级联引擎"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.agents = {
            "fast": "gpt-4o-mini",
            "balanced": "claude-3.5-sonnet",
            "creative": "gemini-2.0-flash",
        }
    
    async def infer(self, messages: list[dict]) -> dict:
        """多智能体并行推理 + 聚合"""
        
        # 并行调用多个模型
        tasks = [
            self._call_agent(name, model, messages)
            for name, model in self.agents.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤成功结果
        valid_results = [
            (name, result) 
            for name, result in zip(self.agents.keys(), results)
            if not isinstance(result, Exception) and result
        ]
        
        if not valid_results:
            # 全部失败，升级到大模型
            return await self._fallback_to_large_model(messages)
        
        # 聚合策略
        if len(valid_results) == 1:
            return {
                "output": valid_results[0][1],
                "meta": {"mode": "single_agent", "agent": valid_results[0][0]}
            }
        
        # 多个结果可用时，进行聚合
        return await self._aggregate(valid_results, messages)
    
    async def _aggregate(self, results: list[tuple], messages: list[dict]) -> dict:
        """
        聚合多个模型的输出
        
        策略：
        1. 如果所有模型一致 → 任选一个
        2. 如果部分一致 → 选择多数意见
        3. 如果全部不同 → 升级到大模型
        """
        outputs = [r[1] for r in results]
        
        # 检查一致性（使用简单的语义相似度）
        consistency = self._check_consistency(outputs)
        
        if consistency["agreement_level"] > 0.8:
            # 高度一致，选择最详细的一个
            best = max(outputs, key=len)
            return {
                "output": best,
                "meta": {
                    "mode": "moa_consensus",
                    "agents": [r[0] for r in results],
                    "consistency": consistency["agreement_level"],
                }
            }
        elif consistency["agreement_level"] > 0.5:
            # 部分一致，使用聚合器
            aggregated = await self._llm_aggregate(outputs, messages)
            return {
                "output": aggregated,
                "meta": {
                    "mode": "moa_aggregated",
                    "consistency": consistency["agreement_level"],
                }
            }
        else:
            # 分歧太大，升级到大模型
            return await self._fallback_to_large_model(messages)
    
    def _check_consistency(self, outputs: list[str]) -> dict:
        """检查多个输出的一致性"""
        # 简单的关键词重叠一致性
        word_sets = [set(o.split()) for o in outputs]
        intersections = []
        
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                if word_sets[i] and word_sets[j]:
                    overlap = len(word_sets[i] & word_sets[j]) / len(word_sets[i] | word_sets[j])
                    intersections.append(overlap)
        
        avg_agreement = sum(intersections) / len(intersections) if intersections else 0
        
        return {
            "agreement_level": avg_agreement,
            "pairwise_scores": intersections,
        }
    
    async def _llm_aggregate(self, outputs: list[str], messages: list[dict]) -> str:
        """使用LLM聚合多个输出"""
        prompt = f"""以下是多个AI模型对同一问题的回答，请综合它们的优点，给出最佳回答：

问题：{messages[-1]["content"]}

回答1：{outputs[0]}
回答2：{outputs[1] if len(outputs) > 1 else "N/A"}
{f"回答3：{outputs[2]}" if len(outputs) > 2 else ""}

请综合以上回答，给出最准确、最完整的回答。"""
        
        result = await self.llm.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return result.choices[0].message.content
```

## 监控与可观测性

级联推理系统的监控需要关注几个独特指标：

```
┌──────────────────────────────────────────────────────────────────┐
│                 级联推理核心监控指标                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📊 路由效率指标                                                  │
│  ├── routing_latency_p99    < 10ms                              │
│  ├── routing_accuracy       > 90%（与人工标注对比）               │
│  ├── tier_distribution      目标: T1=50% T2=35% T3=15%         │
│  └── cache_hit_rate         > 60%                               │
│                                                                  │
│  📈 质量指标                                                      │
│  ├── quality_score_avg      > 85分                               │
│  ├── upgrade_rate           < 15%（太高说明路由不准）             │
│  ├── quality_by_tier        T1>70 T2>80 T3>85                   │
│  └── user_satisfaction      > 4.5/5.0                           │
│                                                                  │
│  💰 成本指标                                                      │
│  ├── cost_per_request_avg   对比基线下降 > 40%                   │
│  ├── total_monthly_cost     预算内                               │
│  ├── cost_by_tier           各层级成本占比                       │
│  └── cost_per_quality_point 性价比指标                           │
│                                                                  │
│  ⚡ 延迟指标                                                      │
│  ├── latency_p50            < 1s                                │
│  ├── latency_p99            < 5s                                │
│  ├── latency_by_tier        各层级延迟分布                       │
│  └── degradation_impact     降级增加的延迟                       │
│                                                                  │
│  🔄 可靠性指标                                                    │
│  ├── fallback_rate          < 10%                               │
│  ├── full_fallback_rate     < 1%                                │
│  ├── model_availability     > 99.5%                             │
│  └── error_recovery_rate    > 99%                               │
└──────────────────────────────────────────────────────────────────┘
```

```python
# Prometheus 指标定义示例
from prometheus_client import Counter, Histogram, Gauge

# 路由分布
cascade_routing_total = Counter(
    'cascade_routing_total',
    'Total routing decisions by tier',
    ['tier']
)

# 质量评分
cascade_quality_score = Histogram(
    'cascade_quality_score',
    'Quality scores of cascade outputs',
    ['tier'],
    buckets=[50, 60, 70, 80, 85, 90, 95, 100]
)

# 升级率
cascade_upgrade_total = Counter(
    'cascade_upgrade_total',
    'Total model upgrades',
    ['from_tier', 'to_tier', 'reason']
)

# 成本节省
cascade_cost_saving = Gauge(
    'cascade_cost_saving_ratio',
    'Cost saving ratio compared to always using top-tier model'
)

# 延迟分布
cascade_latency = Histogram(
    'cascade_latency_seconds',
    'Cascade inference latency',
    ['tier'],
    buckets=[0.1, 0.5, 1, 2, 3, 5, 10]
)
```

## 上线检查清单

```
┌──────────────────────────────────────────────────────────────────┐
│                 级联推理系统上线检查清单                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  □ 路由器                                                         │
│    □ 规则路由覆盖 > 80% 的请求类型                                │
│    □ 路由延迟 p99 < 10ms                                         │
│    □ 路由准确率 > 85%（标注1000+样本验证）                         │
│    □ 缓存命中率 > 50%                                            │
│                                                                  │
│  □ 质量门控                                                       │
│    □ 各层级质量阈值已通过离线评测校准                               │
│    □ 升级率在预期范围内（5%-15%）                                  │
│    □ 假阴性率 < 2%（不该升级时升级了）                              │
│    □ 假阳性率 < 5%（该升级时没升级）                                │
│                                                                  │
│  □ 降级策略                                                       │
│    □ 所有降级路径已测试                                            │
│    □ 超时设置合理（各层级单独配置）                                 │
│    □ 缓存降级可用且内容新鲜                                        │
│    □ 用户侧错误信息友好                                            │
│                                                                  │
│  □ 监控告警                                                       │
│    □ 路由延迟异常告警（>20ms）                                     │
│    □ 质量评分下降告警（<阈值的80%）                                 │
│    □ 升级率异常告警（>25%或<2%）                                   │
│    □ 成本超预算告警                                                │
│    □ 模型不可用告警                                                │
│                                                                  │
│  □ A/B测试                                                        │
│    □ 对照组：单一模型 vs 级联推理                                   │
│    □ 观察指标：质量、延迟、成本、用户满意度                          │
│    □ 最低运行时间：7天                                              │
│    □ 显著性检验：p < 0.05                                          │
└──────────────────────────────────────────────────────────────────┘
```

## 总结

LLM级联推理架构的本质是 **用智能路由替代一刀切**。它不是简单的"小模型便宜"，而是一套包含路由、质量评估、降级和监控的完整工程体系。

核心收益：
- **成本降低 40-70%**：大部分简单请求由轻量模型处理
- **延迟降低 20-40%**：小模型响应更快
- **质量损失 < 5%**：质量门控确保关键场景不受影响

核心挑战：
- **路由准确性**：错误的路由会导致质量下降或成本浪费
- **质量评估**：如何快速、准确地判断输出质量
- **运维复杂度**：多模型管理、监控、告警的复杂度显著增加

级联推理适合 **请求复杂度分布不均** 的场景。如果你的请求中超过 40% 可以由轻量模型处理，那级联推理就是你的最佳选择。

---

*参考资料：*
- *Google Research: "Mixture of Agents" (2024)*
- *Anthropic: "Model Cascades for Efficient Inference"*
- *OpenAI: "GPT-4 Architecture and Scaling"*
- *Together AI: "Mixture of Agents Technical Report"*
