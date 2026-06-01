---
title: "ModelRouter：AI应用的智能模型路由框架——从硬编码调用到自适应路由的架构演进"
description: "深度解析AI应用模型路由框架的设计哲学，涵盖路由策略、成本优化、故障转移与可观测性的完整工程方案。"
date: 2026-06-01
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["模型路由", "Model Router", "AI框架", "成本优化", "故障转移", "模型编排", "LLM网关"]
draft: false
---

# ModelRouter：AI应用的智能模型路由框架——从硬编码调用到自适应路由的架构演进

## 引言：当"调用一个模型"变成"管理一队模型"

在AI应用的早期阶段，调用大模型是一件简单的事——`openai.chat.completions.create(model="gpt-4")`，一行代码搞定。但随着业务规模扩大，你会发现：

- **成本失控**：所有请求都用GPT-4，每月账单让你怀疑人生
- **延迟不可控**：高峰期GPT-4响应变慢，用户体验直线下降
- **单点故障**：OpenAI API挂了，你的服务也跟着挂
- **能力不匹配**：简单问答用GPT-4是杀鸡用牛刀，复杂推理用GPT-3.5又力不从心

这就是**模型路由（Model Routing）**要解决的问题。它不是一个简单的负载均衡器，而是一个**智能调度系统**，能根据请求特征、模型能力、成本约束和系统状态，动态选择最优的模型来处理每个请求。

```
┌──────────────────────────────────────────────────────────────────┐
│                     模型路由架构演进                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: 硬编码 (Hardcoded)                                     │
│  ├── 所有请求 → 单一模型                                          │
│  ├── 简单，但无弹性                                               │
│  └── 适用：MVP、个人项目                                           │
│                                                                   │
│  Phase 2: 配置化 (Configurable)                                   │
│  ├── 请求类型 → 模型映射表                                        │
│  ├── 灵活，但需手动维护                                            │
│  └── 适用：中小规模、稳定场景                                       │
│                                                                   │
│  Phase 3: 智能路由 (Intelligent Routing)                          │
│  ├── 请求特征 + 模型能力 → 最优匹配                               │
│  ├── 自动学习，持续优化                                            │
│  └── 适用：大规模、多模型场景                                       │
│                                                                   │
│  Phase 4: 自适应路由 (Adaptive Routing)                           │
│  ├── 实时感知模型状态 + 成本 + 延迟 → 动态决策                     │
│  ├── 故障自动转移、流量自动调节                                     │
│  └── 适用：生产级AI平台                                           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## 一、模型路由的核心设计

### 1.1 路由策略体系

模型路由的核心是一个**策略引擎**，它决定每个请求应该被路由到哪个模型。不同的策略适用于不同的场景。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import random

class RoutingStrategy(Enum):
    """路由策略类型"""
    ROUND_ROBIN = "round_robin"              # 轮询
    COST_OPTIMIZED = "cost_optimized"        # 成本优先
    LATENCY_OPTIMIZED = "latency_optimized"  # 延迟优先
    QUALITY_OPTIMIZED = "quality_optimized"  # 质量优先
    LOAD_BALANCED = "load_balanced"          # 负载均衡
    CAPABILITY_MATCH = "capability_match"    # 能力匹配
    ADAPTIVE = "adaptive"                    # 自适应

@dataclass
class ModelProfile:
    """模型档案：描述模型的能力、成本和性能特征"""
    model_id: str
    provider: str                           # 提供商: openai, anthropic, local
    
    # 能力指标 (0-1)
    reasoning_capability: float = 0.5       # 推理能力
    coding_capability: float = 0.5          # 编码能力
    creative_capability: float = 0.5        # 创意能力
    factual_accuracy: float = 0.5           # 事实准确性
    multilingual: float = 0.5              # 多语言能力
    max_context_length: int = 4096          # 最大上下文长度
    
    # 成本指标 (美元/1M tokens)
    input_cost: float = 0.0                 # 输入成本
    output_cost: float = 0.0               # 输出成本
    
    # 性能指标
    avg_latency_ms: float = 1000.0          # 平均延迟
    p99_latency_ms: float = 3000.0         # P99延迟
    throughput_rps: float = 10.0            # 每秒请求数
    
    # 状态
    is_available: bool = True               # 是否可用
    error_rate: float = 0.0                 # 错误率
    last_health_check: float = 0.0          # 最后健康检查时间

@dataclass
class RoutingRequest:
    """路由请求：描述一个需要路由的请求"""
    request_id: str
    user_query: str                         # 用户输入
    task_type: str                          # 任务类型: qa/coding/creative/translation/...
    quality_requirements: str = "standard"  # 质量要求: low/standard/high/critical
    latency_budget_ms: float = 5000.0      # 延迟预算
    cost_budget: float = 0.1               # 成本预算 (美元)
    preferred_providers: List[str] = field(default_factory=list)
    exclude_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingDecision:
    """路由决策：路由器返回的结果"""
    model_id: str
    provider: str
    strategy_used: RoutingStrategy
    confidence: float                       # 决策置信度
    estimated_cost: float                   # 预估成本
    estimated_latency_ms: float            # 预估延迟
    fallback_models: List[str] = field(default_factory=list)  # 备选模型
    reasoning: str = ""                     # 决策理由（可解释性）
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 1.2 能力匹配路由器

最核心的路由策略是**能力匹配**：根据请求的特征，找到最擅长处理这类任务的模型。

```python
import numpy as np
from typing import Tuple

class CapabilityMatcher:
    """能力匹配路由器
    
    通过计算请求特征与模型能力的匹配度来选择最优模型。
    使用加权余弦相似度进行匹配。
    """
    
    # 任务类型与能力维度的权重映射
    TASK_CAPABILITY_WEIGHTS = {
        "qa": {
            "reasoning_capability": 0.3,
            "factual_accuracy": 0.4,
            "creative_capability": 0.1,
            "coding_capability": 0.0,
            "multilingual": 0.2
        },
        "coding": {
            "reasoning_capability": 0.3,
            "factual_accuracy": 0.1,
            "creative_capability": 0.0,
            "coding_capability": 0.6,
            "multilingual": 0.0
        },
        "creative": {
            "reasoning_capability": 0.1,
            "factual_accuracy": 0.1,
            "creative_capability": 0.7,
            "coding_capability": 0.0,
            "multilingual": 0.1
        },
        "translation": {
            "reasoning_capability": 0.1,
            "factual_accuracy": 0.1,
            "creative_capability": 0.2,
            "coding_capability": 0.0,
            "multilingual": 0.6
        },
        "analysis": {
            "reasoning_capability": 0.5,
            "factual_accuracy": 0.3,
            "creative_capability": 0.0,
            "coding_capability": 0.1,
            "multilingual": 0.1
        }
    }
    
    def __init__(self, model_profiles: List[ModelProfile]):
        self.profiles = {p.model_id: p for p in model_profiles}
    
    def compute_match_score(self, request: RoutingRequest,
                            model: ModelProfile) -> float:
        """计算请求与模型的匹配分数"""
        weights = self.TASK_CAPABILITY_WEIGHTS.get(
            request.task_type,
            self.TASK_CAPABILITY_WEIGHTS["qa"]
        )
        
        # 加权能力匹配
        score = 0.0
        total_weight = 0.0
        
        capabilities = {
            "reasoning_capability": model.reasoning_capability,
            "factual_accuracy": model.factual_accuracy,
            "creative_capability": model.creative_capability,
            "coding_capability": model.coding_capability,
            "multilingual": model.multilingual
        }
        
        for cap_name, weight in weights.items():
            if weight > 0:
                score += capabilities.get(cap_name, 0) * weight
                total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        # 上下文长度惩罚
        if request.metadata.get("estimated_tokens", 0) > model.max_context_length * 0.8:
            score *= 0.3  # 接近上下文上限时大幅降分
        
        # 成本惩罚
        estimated_tokens = request.metadata.get("estimated_tokens", 1000)
        estimated_cost = (model.input_cost * estimated_tokens / 1_000_000 +
                         model.output_cost * estimated_tokens * 0.3 / 1_000_000)
        
        if estimated_cost > request.cost_budget:
            score *= 0.1  # 超出预算时几乎排除
        
        # 延迟惩罚
        if model.avg_latency_ms > request.latency_budget_ms:
            score *= 0.3  # 超出延迟预算时降分
        
        # 可用性检查
        if not model.is_available or model.error_rate > 0.1:
            score *= 0.05
        
        # 提供商偏好加分
        if (request.preferred_providers and 
            model.provider in request.preferred_providers):
            score *= 1.2
        
        # 排除检查
        if model.model_id in request.exclude_models:
            score = 0.0
        
        return score
    
    def route(self, request: RoutingRequest) -> Tuple[str, float, List[str]]:
        """执行能力匹配路由
        
        Returns:
            (best_model_id, match_score, ranked_candidates)
        """
        scores = {}
        for model_id, model in self.profiles.items():
            scores[model_id] = self.compute_match_score(request, model)
        
        # 按分数排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not ranked or ranked[0][1] <= 0:
            # 没有可用模型，返回默认
            return "gpt-3.5-turbo", 0.0, []
        
        best_model_id, best_score = ranked[0]
        fallbacks = [m for m, s in ranked[1:4] if s > 0.3]
        
        return best_model_id, best_score, fallbacks
```

### 1.3 成本优化路由器

对于成本敏感的场景，路由器需要在**质量**和**成本**之间找到最佳平衡点。

```python
class CostOptimizedRouter:
    """成本优化路由器
    
    核心思想：用最小的成本满足质量要求。
    使用"级联策略"：先尝试便宜模型，质量不够再升级。
    """
    
    # 模型等级与成本
    MODEL_TIERS = {
        "tier1_ultra": {  # GPT-4, Claude Opus 等
            "models": ["gpt-4-turbo", "claude-3-opus"],
            "cost_multiplier": 1.0,
            "quality_threshold": 0.9
        },
        "tier2_standard": {  # GPT-4o, Claude Sonnet 等
            "models": ["gpt-4o", "claude-3-sonnet", "gemini-pro"],
            "cost_multiplier": 0.3,
            "quality_threshold": 0.75
        },
        "tier3_economy": {  # GPT-3.5, Claude Haiku 等
            "models": ["gpt-3.5-turbo", "claude-3-haiku", "gemini-flash"],
            "cost_multiplier": 0.05,
            "quality_threshold": 0.5
        },
        "tier4_free": {  # 开源模型
            "models": ["llama-3-70b", "mixtral-8x7b", "qwen-72b"],
            "cost_multiplier": 0.01,
            "quality_threshold": 0.4
        }
    }
    
    def __init__(self, model_profiles: List[ModelProfile],
                 cost_per_dollar: float = 1.0):
        self.profiles = {p.model_id: p for p in model_profiles}
        self.cost_per_dollar = cost_per_dollar
        
        # 历史成本统计
        self.cost_history: Dict[str, List[float]] = {}
    
    def select_tier(self, request: RoutingRequest) -> str:
        """根据请求特征选择模型等级"""
        task_complexity = self._estimate_complexity(request)
        
        if task_complexity > 0.8 or request.quality_requirements == "critical":
            return "tier1_ultra"
        elif task_complexity > 0.5 or request.quality_requirements == "high":
            return "tier2_standard"
        elif task_complexity > 0.2 or request.quality_requirements == "standard":
            return "tier3_economy"
        else:
            return "tier4_free"
    
    def _estimate_complexity(self, request: RoutingRequest) -> float:
        """估算请求复杂度 (0-1)"""
        complexity = 0.0
        
        # 基于任务类型
        task_complexity = {
            "qa": 0.3, "coding": 0.7, "creative": 0.5,
            "translation": 0.4, "analysis": 0.6, "summarization": 0.3
        }
        complexity += task_complexity.get(request.task_type, 0.5) * 0.4
        
        # 基于输入长度
        query_length = len(request.user_query)
        if query_length > 2000:
            complexity += 0.3
        elif query_length > 500:
            complexity += 0.15
        
        # 基于质量要求
        quality_map = {"low": 0, "standard": 0.3, "high": 0.6, "critical": 0.9}
        complexity += quality_map.get(request.quality_requirements, 0.3) * 0.3
        
        return min(1.0, complexity)
    
    def cascade_route(self, request: RoutingRequest) -> List[RoutingDecision]:
        """级联路由：从低到高尝试，直到质量满足要求
        
        返回一个有序的模型列表，按优先级排列。
        调用者可以依次尝试，直到获得满意结果。
        """
        tier = self.select_tier(request)
        tier_config = self.MODEL_TIERS[tier]
        
        decisions = []
        for model_id in tier_config["models"]:
            if model_id not in self.profiles:
                continue
            
            model = self.profiles[model_id]
            estimated_cost = self._estimate_cost(model, request)
            
            decisions.append(RoutingDecision(
                model_id=model_id,
                provider=model.provider,
                strategy_used=RoutingStrategy.COST_OPTIMIZED,
                confidence=tier_config["quality_threshold"],
                estimated_cost=estimated_cost,
                estimated_latency_ms=model.avg_latency_ms,
                reasoning=f"选择{tier}等级模型，预估复杂度{self._estimate_complexity(request):.2f}"
            ))
        
        # 如果当前等级模型不够，添加更高等级作为fallback
        tier_order = ["tier4_free", "tier3_economy", "tier2_standard", "tier1_ultra"]
        current_idx = tier_order.index(tier)
        
        for higher_tier in tier_order[current_idx+1:]:
            for model_id in self.MODEL_TIERS[higher_tier]["models"]:
                if model_id in self.profiles:
                    model = self.profiles[model_id]
                    decisions.append(RoutingDecision(
                        model_id=model_id,
                        provider=model.provider,
                        strategy_used=RoutingStrategy.COST_OPTIMIZED,
                        confidence=self.MODEL_TIERS[higher_tier]["quality_threshold"],
                        estimated_cost=self._estimate_cost(model, request),
                        estimated_latency_ms=model.avg_latency_ms,
                        reasoning=f"Fallback: {higher_tier}等级模型"
                    ))
        
        return decisions
    
    def _estimate_cost(self, model: ModelProfile,
                       request: RoutingRequest) -> float:
        """预估请求成本"""
        estimated_input_tokens = len(request.user_query) * 1.5  # 粗略估算
        estimated_output_tokens = estimated_input_tokens * 0.3
        
        cost = (model.input_cost * estimated_input_tokens / 1_000_000 +
                model.output_cost * estimated_output_tokens / 1_000_000)
        
        return round(cost, 6)
```

## 二、自适应路由引擎

### 2.1 基于强化学习的路由决策

自适应路由的核心是**学习**：根据历史调用结果，不断优化路由策略。

```python
import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple

class AdaptiveRouter:
    """自适应路由器
    
    使用多臂老虎机(Multi-Armed Bandit)算法进行路由决策。
    核心思想：在"探索"（尝试新模型）和"利用"（使用已知最优模型）之间平衡。
    
    使用Thompson Sampling算法：
    - 每个模型维护一个Beta分布参数(α, β)
    - α: 成功次数 + 1
    - β: 失败次数 + 1
    - 决策时从每个模型的Beta分布采样，选择采样值最高的模型
    """
    
    def __init__(self, model_profiles: List[ModelProfile],
                 exploration_rate: float = 0.1):
        self.profiles = {p.model_id: p for p in model_profiles}
        self.exploration_rate = exploration_rate
        
        # Thompson Sampling参数
        self.alpha: Dict[str, float] = {m: 1.0 for m in self.profiles}  # 成功次数
        self.beta: Dict[str, float] = {m: 1.0 for m in self.profiles}   # 失败次数
        
        # 性能统计
        self.total_calls: Dict[str, int] = defaultdict(int)
        self.total_cost: Dict[str, float] = defaultdict(float)
        self.total_latency: Dict[str, float] = defaultdict(float)
        self.total_quality_scores: Dict[str, List[float]] = defaultdict(list)
        
        # 衰减因子（近期数据权重更高）
        self.decay_factor = 0.95
    
    def route(self, request: RoutingRequest) -> RoutingDecision:
        """使用Thompson Sampling进行路由决策"""
        
        # ε-greedy探索
        if random.random() < self.exploration_rate:
            # 探索：随机选择一个可用模型
            available = [m for m, p in self.profiles.items() if p.is_available]
            if not available:
                return self._fallback_decision()
            
            model_id = random.choice(available)
            model = self.profiles[model_id]
            
            return RoutingDecision(
                model_id=model_id,
                provider=model.provider,
                strategy_used=RoutingStrategy.ADAPTIVE,
                confidence=0.5,
                estimated_cost=self._estimate_cost(model, request),
                estimated_latency_ms=model.avg_latency_ms,
                reasoning="探索性选择：随机尝试模型以收集更多数据"
            )
        
        # 利用：从Thompson Sampling分布中采样
        samples = {}
        for model_id in self.profiles:
            if not self.profiles[model_id].is_available:
                continue
            
            # 从Beta分布采样
            sample = random.betavariate(self.alpha[model_id], self.beta[model_id])
            
            # 考虑成本因素
            model = self.profiles[model_id]
            cost_factor = 1.0 / (1.0 + self._estimate_cost(model, request) * 100)
            
            # 综合得分
            samples[model_id] = sample * 0.7 + cost_factor * 0.3
        
        if not samples:
            return self._fallback_decision()
        
        # 选择得分最高的模型
        best_model_id = max(samples, key=samples.get)
        best_model = self.profiles[best_model_id]
        
        return RoutingDecision(
            model_id=best_model_id,
            provider=best_model.provider,
            strategy_used=RoutingStrategy.ADAPTIVE,
            confidence=samples[best_model_id],
            estimated_cost=self._estimate_cost(best_model, request),
            estimated_latency_ms=best_model.avg_latency_ms,
            reasoning=f"Thompson Sampling决策 (α={self.alpha[best_model_id]:.1f}, "
                     f"β={self.beta[best_model_id]:.1f})"
        )
    
    def update(self, model_id: str, success: bool,
               latency_ms: float, cost: float,
               quality_score: float = None):
        """更新模型的性能统计
        
        Args:
            model_id: 模型ID
            success: 是否成功
            latency_ms: 实际延迟
            cost: 实际成本
            quality_score: 质量评分 (0-1)，可选
        """
        self.total_calls[model_id] += 1
        
        if success:
            self.alpha[model_id] += 1.0
        else:
            self.beta[model_id] += 1.0
        
        # 应用衰减（防止历史数据权重过高）
        self.alpha[model_id] *= self.decay_factor
        self.beta[model_id] *= self.decay_factor
        
        # 确保最小值
        self.alpha[model_id] = max(1.0, self.alpha[model_id])
        self.beta[model_id] = max(1.0, self.beta[model_id])
        
        # 更新统计
        self.total_cost[model_id] += cost
        self.total_latency[model_id] += latency_ms
        
        if quality_score is not None:
            self.total_quality_scores[model_id].append(quality_score)
            # 只保留最近100个质量评分
            if len(self.total_quality_scores[model_id]) > 100:
                self.total_quality_scores[model_id] = \
                    self.total_quality_scores[model_id][-100:]
    
    def get_model_statistics(self) -> Dict[str, Dict]:
        """获取所有模型的统计信息"""
        stats = {}
        
        for model_id in self.profiles:
            calls = self.total_calls[model_id]
            if calls == 0:
                stats[model_id] = {"status": "no_data"}
                continue
            
            avg_latency = self.total_latency[model_id] / calls
            avg_cost = self.total_cost[model_id] / calls
            success_rate = self.alpha[model_id] / (
                self.alpha[model_id] + self.beta[model_id]
            )
            
            quality_scores = self.total_quality_scores[model_id]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            stats[model_id] = {
                "total_calls": calls,
                "success_rate": round(success_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "avg_cost": round(avg_cost, 6),
                "avg_quality_score": round(avg_quality, 4),
                "thompson_alpha": round(self.alpha[model_id], 2),
                "thompson_beta": round(self.beta[model_id], 2),
                "exploration_priority": round(
                    self.beta[model_id] / (self.alpha[model_id] + self.beta[model_id]),
                    4
                )
            }
        
        return stats
    
    def _estimate_cost(self, model: ModelProfile,
                       request: RoutingRequest) -> float:
        """预估成本"""
        estimated_tokens = len(request.user_query) * 2
        return (model.input_cost * estimated_tokens / 1_000_000 +
                model.output_cost * estimated_tokens * 0.3 / 1_000_000)
    
    def _fallback_decision(self) -> RoutingDecision:
        """兜底决策"""
        return RoutingDecision(
            model_id="gpt-3.5-turbo",
            provider="openai",
            strategy_used=RoutingStrategy.ADAPTIVE,
            confidence=0.3,
            estimated_cost=0.001,
            estimated_latency_ms=1000,
            reasoning="所有模型不可用，使用默认模型"
        )
```

### 2.2 感知成本的智能路由

```python
class CostAwareRouter:
    """感知成本的智能路由器
    
    在满足质量要求的前提下，最小化成本。
    使用动态规划思想：将请求预算分配到最优化的成本-质量曲线上。
    """
    
    def __init__(self, model_profiles: List[ModelProfile]):
        self.profiles = {p.model_id: p for p in model_profiles}
        self.daily_budget: float = 100.0  # 每日预算（美元）
        self.daily_spent: float = 0.0
        self.budget_reset_time: float = 0.0
    
    def route_with_budget(self, request: RoutingRequest,
                          quality_threshold: float = 0.7) -> RoutingDecision:
        """带预算约束的路由"""
        
        # 检查并重置每日预算
        current_time = time.time()
        if current_time - self.budget_reset_time > 86400:  # 24小时
            self.daily_spent = 0.0
            self.budget_reset_time = current_time
        
        remaining_budget = self.daily_budget - self.daily_spent
        
        # 如果预算不足，降级到免费模型
        if remaining_budget <= 0.01:
            return self._route_to_free_tier(request)
        
        # 收集候选模型
        candidates = []
        for model_id, model in self.profiles.items():
            if not model.is_available:
                continue
            
            estimated_cost = self._estimate_cost(model, request)
            
            # 成本检查
            if estimated_cost > min(remaining_budget, request.cost_budget):
                continue
            
            # 质量评估
            quality_score = self._estimate_quality(model, request)
            
            if quality_score >= quality_threshold:
                # 计算性价比（质量/成本）
                cost_efficiency = quality_score / max(estimated_cost, 0.0001)
                
                candidates.append({
                    "model_id": model_id,
                    "model": model,
                    "cost": estimated_cost,
                    "quality": quality_score,
                    "efficiency": cost_efficiency
                })
        
        if not candidates:
            # 没有满足条件的模型，尝试降低质量要求
            return self._route_with_relaxed_constraints(
                request, quality_threshold * 0.8
            )
        
        # 按性价比排序，选择最优
        candidates.sort(key=lambda x: x["efficiency"], reverse=True)
        best = candidates[0]
        
        return RoutingDecision(
            model_id=best["model_id"],
            provider=best["model"].provider,
            strategy_used=RoutingStrategy.COST_OPTIMIZED,
            confidence=best["quality"],
            estimated_cost=best["cost"],
            estimated_latency_ms=best["model"].avg_latency_ms,
            reasoning=f"性价比最优: 质量{best['quality']:.2f}/成本${best['cost']:.4f}="
                     f"效率{best['efficiency']:.0f}"
        )
    
    def _estimate_quality(self, model: ModelProfile,
                          request: RoutingRequest) -> float:
        """估算模型对特定请求的质量得分"""
        # 基于任务类型的权重
        weights = {
            "reasoning": model.reasoning_capability,
            "factual": model.factual_accuracy,
            "creative": model.creative_capability,
            "coding": model.coding_capability,
            "multilingual": model.multilingual
        }
        
        task_weights = {
            "qa": {"reasoning": 0.3, "factual": 0.5, "creative": 0.1},
            "coding": {"coding": 0.7, "reasoning": 0.2},
            "creative": {"creative": 0.7, "reasoning": 0.2},
            "translation": {"multilingual": 0.6, "creative": 0.2},
            "analysis": {"reasoning": 0.5, "factual": 0.3}
        }
        
        tw = task_weights.get(request.task_type, {"reasoning": 0.5})
        
        score = 0.0
        for cap, weight in tw.items():
            score += weights.get(cap, 0.5) * weight
        
        return score
    
    def _estimate_cost(self, model: ModelProfile,
                       request: RoutingRequest) -> float:
        """估算成本"""
        tokens = len(request.user_query) * 2
        return (model.input_cost * tokens / 1_000_000 +
                model.output_cost * tokens * 0.3 / 1_000_1000_000)
    
    def _route_to_free_tier(self, request: RoutingRequest) -> RoutingDecision:
        """路由到免费/低成本模型"""
        free_models = [
            m for m, p in self.profiles.items()
            if p.is_available and p.input_cost < 0.001
        ]
        
        if free_models:
            model_id = free_models[0]
            model = self.profiles[model_id]
            return RoutingDecision(
                model_id=model_id,
                provider=model.provider,
                strategy_used=RoutingStrategy.COST_OPTIMIZED,
                confidence=0.5,
                estimated_cost=0.0,
                estimated_latency_ms=model.avg_latency_ms,
                reasoning="每日预算已用完，降级到免费模型"
            )
        
        return RoutingDecision(
            model_id="llama-3-70b",
            provider="local",
            strategy_used=RoutingStrategy.COST_OPTIMIZED,
            confidence=0.4,
            estimated_cost=0.0,
            estimated_latency_ms=2000,
            reasoning="每日预算已用完，使用本地模型"
        )
    
    def _route_with_relaxed_constraints(self, request: RoutingRequest,
                                         quality_threshold: float) -> RoutingDecision:
        """放宽约束重新路由"""
        # 递归调用，降低质量阈值
        return self.route_with_budget(request, quality_threshold)
```

## 三、故障转移与高可用设计

### 3.1 多层故障转移机制

```python
import time
from typing import Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_BROKEN = "circuit_broken"

@dataclass
class CircuitBreaker:
    """熔断器：防止对故障模型的持续调用"""
    failure_threshold: int = 5          # 失败阈值
    recovery_timeout: float = 60.0      # 恢复超时（秒）
    half_open_max_calls: int = 3        # 半开状态最大调用数
    
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    status: HealthStatus = HealthStatus.HEALTHY
    half_open_calls: int = 0
    
    def record_success(self):
        """记录成功调用"""
        if self.status == HealthStatus.CIRCUIT_BROKEN:
            return
        
        self.success_count += 1
        self.failure_count = max(0, self.failure_count - 1)
        
        if self.status == HealthStatus.DEGRADED and self.failure_count == 0:
            self.status = HealthStatus.HEALTHY
    
    def record_failure(self):
        """记录失败调用"""
        self.failure_count += 1
        self.success_count = max(0, self.success_count - 1)
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.status = HealthStatus.CIRCUIT_BROKEN
    
    def can_execute(self) -> bool:
        """检查是否可以执行调用"""
        if self.status == HealthStatus.HEALTHY:
            return True
        
        if self.status == HealthStatus.DEGRADED:
            return True
        
        if self.status == HealthStatus.CIRCUIT_BROKEN:
            # 检查是否超过恢复超时
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.status = HealthStatus.DEGRADED
                self.half_open_calls = 0
                return True
            return False
        
        return False
    
    def record_half_open_result(self, success: bool):
        """记录半开状态的调用结果"""
        self.half_open_calls += 1
        
        if success:
            self.success_count += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.status = HealthStatus.HEALTHY
                self.failure_count = 0
        else:
            self.status = HealthStatus.CIRCUIT_BROKEN
            self.last_failure_time = time.time()


class HighAvailabilityRouter:
    """高可用路由器
    
    故障转移策略：
    1. 主模型失败 → 尝试备用模型
    2. 所有预设模型失败 → 启动探索模式
    3. 探索模式也失败 → 降级到缓存响应
    4. 缓存也没有 → 返回友好错误
    """
    
    def __init__(self, model_profiles: List[ModelProfile]):
        self.profiles = {p.model_id: p for p in model_profiles}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            m: CircuitBreaker() for m in self.profiles
        }
        
        # 响应缓存
        self.response_cache: Dict[str, str] = {}
        self.cache_ttl: float = 3600.0  # 1小时
    
    def route_with_fallback(self, request: RoutingRequest,
                            primary_model: str,
                            fallback_chain: List[str]) -> RoutingDecision:
        """带故障转移的路由"""
        
        # 尝试主模型
        if self._can_use_model(primary_model):
            model = self.profiles[primary_model]
            return RoutingDecision(
                model_id=primary_model,
                provider=model.provider,
                strategy_used=RoutingStrategy.ADAPTIVE,
                confidence=0.9,
                estimated_cost=self._estimate_cost(model, request),
                estimated_latency_ms=model.avg_latency_ms,
                fallback_models=fallback_chain,
                reasoning="主模型可用"
            )
        
        # 主模型不可用，遍历备用链
        for fallback_model in fallback_chain:
            if self._can_use_model(fallback_model):
                model = self.profiles[fallback_model]
                return RoutingDecision(
                    model_id=fallback_model,
                    provider=model.provider,
                    strategy_used=RoutingStrategy.ADAPTIVE,
                    confidence=0.7,
                    estimated_cost=self._estimate_cost(model, request),
                    estimated_latency_ms=model.avg_latency_ms,
                    fallback_models=[
                        m for m in fallback_chain if m != fallback_model
                    ],
                    reasoning=f"主模型{primary_model}不可用，切换到{fallback_model}"
                )
        
        # 所有预设模型都不可用，尝试探索
        return self._explore_alternatives(request)
    
    def _can_use_model(self, model_id: str) -> bool:
        """检查模型是否可用"""
        if model_id not in self.profiles:
            return False
        
        if not self.profiles[model_id].is_available:
            return False
        
        return self.circuit_breakers[model_id].can_execute()
    
    def _explore_alternatives(self, request: RoutingRequest) -> RoutingDecision:
        """探索替代模型"""
        for model_id, model in self.profiles.items():
            if (model.is_available and 
                self.circuit_breakers[model_id].can_execute()):
                return RoutingDecision(
                    model_id=model_id,
                    provider=model.provider,
                    strategy_used=RoutingStrategy.ADAPTIVE,
                    confidence=0.3,
                    estimated_cost=self._estimate_cost(model, request),
                    estimated_latency_ms=model.avg_latency_ms,
                    reasoning=f"探索模式：所有预设模型不可用，尝试{model_id}"
                )
        
        # 真的没有可用模型了
        return RoutingDecision(
            model_id="none",
            provider="none",
            strategy_used=RoutingStrategy.ADAPTIVE,
            confidence=0.0,
            estimated_cost=0.0,
            estimated_latency_ms=0.0,
            reasoning="所有模型不可用"
        )
    
    def record_result(self, model_id: str, success: bool,
                      latency_ms: float, error: Optional[str] = None):
        """记录调用结果，更新熔断器状态"""
        if model_id not in self.circuit_breakers:
            return
        
        cb = self.circuit_breakers[model_id]
        
        if success:
            cb.record_success()
        else:
            cb.record_failure()
            
            # 更新模型状态
            if model_id in self.profiles:
                model = self.profiles[model_id]
                model.error_rate = min(1.0, model.error_rate + 0.1)
    
    def _estimate_cost(self, model: ModelProfile,
                       request: RoutingRequest) -> float:
        tokens = len(request.user_query) * 2
        return (model.input_cost * tokens / 1_000_000 +
                model.output_cost * tokens * 0.3 / 1_000_000)
    
    def get_health_report(self) -> Dict[str, Dict]:
        """获取健康报告"""
        report = {}
        
        for model_id, cb in self.circuit_breakers.items():
            model = self.profiles.get(model_id)
            report[model_id] = {
                "status": cb.status.value,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
                "error_rate": model.error_rate if model else 0,
                "is_available": model.is_available if model else False
            }
        
        return report
```

## 四、完整路由器实现

### 4.1 统一路由器

将所有策略整合到一个统一的路由器中：

```python
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)

class ModelRouter:
    """统一模型路由器
    
    整合所有路由策略，提供统一的路由接口。
    
    使用方式:
        router = ModelRouter(model_profiles)
        decision = router.route(request)
        # 使用 decision.model_id 调用模型
    """
    
    def __init__(self, model_profiles: List[ModelProfile],
                 default_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
                 daily_budget: float = 100.0):
        
        self.model_profiles = model_profiles
        self.default_strategy = default_strategy
        
        # 初始化各子路由器
        self.capability_matcher = CapabilityMatcher(model_profiles)
        self.cost_router = CostOptimizedRouter(model_profiles)
        self.adaptive_router = AdaptiveRouter(model_profiles)
        self.cost_aware_router = CostAwareRouter(model_profiles)
        self.ha_router = HighAvailabilityRouter(model_profiles)
        
        self.cost_aware_router.daily_budget = daily_budget
        
        # 路由日志
        self.routing_log: List[Dict] = []
    
    def route(self, request: RoutingRequest,
              strategy: Optional[RoutingStrategy] = None) -> RoutingDecision:
        """执行路由
        
        Args:
            request: 路由请求
            strategy: 路由策略，None则使用默认策略
        """
        strategy = strategy or self.default_strategy
        
        # 根据策略选择路由器
        if strategy == RoutingStrategy.CAPABILITY_MATCH:
            model_id, score, fallbacks = self.capability_matcher.route(request)
            model = self._get_model(model_id)
            
            decision = RoutingDecision(
                model_id=model_id,
                provider=model.provider if model else "unknown",
                strategy_used=strategy,
                confidence=score,
                estimated_cost=self.cost_router._estimate_cost(model, request) if model else 0,
                estimated_latency_ms=model.avg_latency_ms if model else 0,
                fallback_models=fallbacks,
                reasoning=f"能力匹配: 得分{score:.3f}"
            )
        
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            cascade = self.cost_router.cascade_route(request)
            decision = cascade[0] if cascade else self._fallback_decision()
        
        elif strategy == RoutingStrategy.ADAPTIVE:
            decision = self.adaptive_router.route(request)
        
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            decision = self.cost_aware_router.route_with_budget(request)
        
        else:
            # 默认使用能力匹配
            model_id, score, fallbacks = self.capability_matcher.route(request)
            model = self._get_model(model_id)
            
            decision = RoutingDecision(
                model_id=model_id,
                provider=model.provider if model else "unknown",
                strategy_used=RoutingStrategy.CAPABILITY_MATCH,
                confidence=score,
                estimated_cost=self.cost_router._estimate_cost(model, request) if model else 0,
                estimated_latency_ms=model.avg_latency_ms if model else 0,
                fallback_models=fallbacks,
                reasoning=f"默认能力匹配: 得分{score:.3f}"
            )
        
        # 应用高可用故障转移
        if not self.ha_router._can_use_model(decision.model_id):
            decision = self.ha_router.route_with_fallback(
                request,
                decision.model_id,
                decision.fallback_models
            )
        
        # 记录路由日志
        self._log_routing(request, decision)
        
        return decision
    
    def record_outcome(self, request_id: str, model_id: str,
                       success: bool, latency_ms: float,
                       cost: float, quality_score: Optional[float] = None,
                       error: Optional[str] = None):
        """记录调用结果，用于自适应学习"""
        self.adaptive_router.update(
            model_id=model_id,
            success=success,
            latency_ms=latency_ms,
            cost=cost,
            quality_score=quality_score
        )
        
        self.ha_router.record_result(
            model_id=model_id,
            success=success,
            latency_ms=latency_ms,
            error=error
        )
        
        if success:
            self.cost_aware_router.daily_spent += cost
    
    def get_statistics(self) -> Dict:
        """获取路由统计"""
        return {
            "adaptive_stats": self.adaptive_router.get_model_statistics(),
            "health_report": self.ha_router.get_health_report(),
            "daily_budget_remaining": (
                self.cost_aware_router.daily_budget - 
                self.cost_aware_router.daily_spent
            ),
            "total_routes": len(self.routing_log)
        }
    
    def _get_model(self, model_id: str) -> Optional[ModelProfile]:
        for p in self.model_profiles:
            if p.model_id == model_id:
                return p
        return None
    
    def _fallback_decision(self) -> RoutingDecision:
        return RoutingDecision(
            model_id="gpt-3.5-turbo",
            provider="openai",
            strategy_used=self.default_strategy,
            confidence=0.3,
            estimated_cost=0.001,
            estimated_latency_ms=1000,
            reasoning="路由失败，使用默认模型"
        )
    
    def _log_routing(self, request: RoutingRequest, decision: RoutingDecision):
        """记录路由日志"""
        self.routing_log.append({
            "request_id": request.request_id,
            "timestamp": time.time(),
            "task_type": request.task_type,
            "selected_model": decision.model_id,
            "strategy": decision.strategy_used.value,
            "confidence": decision.confidence,
            "estimated_cost": decision.estimated_cost,
            "reasoning": decision.reasoning
        })
        
        # 只保留最近1000条日志
        if len(self.routing_log) > 1000:
            self.routing_log = self.routing_log[-1000:]
```

## 五、模型路由的最佳实践

### 5.1 路由策略选择矩阵

| 场景 | 推荐策略 | 关键指标 | 注意事项 |
|------|---------|---------|---------|
| 成本敏感型SaaS | COST_OPTIMIZED + 级联 | 月度成本/请求 | 设置成本预算上限 |
| 实时交互应用 | LATENCY_OPTIMIZED | P99延迟 | 考虑本地模型 |
| 企业级AI平台 | ADAPTIVE + 熔断 | 可用性/质量 | 多区域部署 |
| 研究/实验 | CAPABILITY_MATCH | 任务准确率 | 允许较高成本 |
| 混合工作负载 | ADAPTIVE（默认） | 综合指标 | 动态调整权重 |

### 5.2 路由器部署架构

```
┌──────────────────────────────────────────────────────────────────┐
│                  模型路由器生产部署架构                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  应用服务A   │    │  应用服务B   │    │  应用服务C   │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │
│         │                  │                  │                   │
│         └──────────────────┼──────────────────┘                   │
│                            │                                      │
│                    ┌───────▼───────┐                              │
│                    │   模型路由器    │                              │
│                    │  (ModelRouter) │                              │
│                    └───────┬───────┘                              │
│                            │                                      │
│         ┌──────────────────┼──────────────────┐                   │
│         │                  │                  │                    │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐          │
│  │  路由决策    │    │  状态监控    │    │  成本追踪    │          │
│  │  (决策引擎)  │    │  (健康检查)  │    │  (费用统计)  │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │
│         │                  │                  │                    │
│         └──────────────────┼──────────────────┘                   │
│                            │                                      │
│  ┌─────────────────────────▼───────────────────────────────┐     │
│  │                    模型提供商层                            │     │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │     │
│  │  │OpenAI  │  │Anthropic│  │Google  │  │本地模型 │        │     │
│  │  │GPT-4o  │  │Sonnet  │  │Gemini  │  │Llama   │        │     │
│  │  └────────┘  └────────┘  └────────┘  └────────┘        │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 关键指标监控

```python
class RouterMonitor:
    """路由器监控"""
    
    def __init__(self, router: ModelRouter):
        self.router = router
    
    def get_dashboard_metrics(self) -> Dict:
        """获取监控面板指标"""
        stats = self.router.get_statistics()
        
        # 计算关键指标
        total_routes = stats["total_routes"]
        model_usage = {}
        
        for log in self.router.routing_log:
            model = log["selected_model"]
            model_usage[model] = model_usage.get(model, 0) + 1
        
        return {
            "overview": {
                "total_routes_24h": total_routes,
                "active_models": len([
                    m for m, s in stats["health_report"].items()
                    if s["status"] in ["healthy", "degraded"]
                ]),
                "circuit_breakers_open": len([
                    m for m, s in stats["health_report"].items()
                    if s["status"] == "circuit_broken"
                ]),
                "daily_budget_remaining": stats["daily_budget_remaining"]
            },
            "model_distribution": {
                model: {
                    "count": count,
                    "percentage": round(count / max(total_routes, 1) * 100, 1)
                }
                for model, count in model_usage.items()
            },
            "health_status": stats["health_report"]
        }
```

## 总结

模型路由不是简单的负载均衡，而是一个**智能决策系统**。它需要同时考虑：

```
┌─────────────────────────────────────────────────────────┐
│            模型路由的核心决策维度                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 能力匹配：哪个模型最擅长处理这个任务？                 │
│     └── 任务类型 × 模型能力 = 匹配分数                    │
│                                                          │
│  2. 成本优化：在预算内找到最优解                          │
│     └── 级联策略 + 性价比排序                             │
│                                                          │
│  3. 延迟感知：满足用户的响应时间要求                       │
│     └── 延迟预算 + 实时延迟监控                           │
│                                                          │
│  4. 高可用：故障时自动转移，服务不中断                      │
│     └── 熔断器 + 备用链 + 探索模式                        │
│                                                          │
│  5. 自适应学习：从历史数据中持续优化                       │
│     └── Thompson Sampling + 衰减因子                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**核心原则**：好的模型路由应该是**透明的**——用户不需要知道背后调用了哪个模型，但开发者可以随时查看路由决策和理由。

---

*相关文章推荐*:
- 《LLM应用网关架构设计：从API代理到智能路由的演进之路》
- 《AI应用成本优化架构：从模型路由到缓存策略的全链路成本控制方案》
- 《AI系统多模型路由架构：智能调度与故障转移的工程实践》
