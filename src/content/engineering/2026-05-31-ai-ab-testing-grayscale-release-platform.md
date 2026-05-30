---
title: "AI应用的灰度发布与实验平台：Prompt A/B测试与效果评估的工程化体系"
description: "系统讲解AI应用的灰度发布体系设计，涵盖Prompt版本管理、A/B测试框架、流量染色策略与在线效果评估，附完整工程化实现方案。"
date: 2026-05-31
author: "RiceBall"
category: "engineering"
subCategory: "infra"
tags: ["灰度发布", "A/B测试", "Prompt管理", "效果评估", "LLM工程化", "实验平台"]
draft: false
---

# AI应用的灰度发布与实验平台：Prompt A/B测试与效果评估的工程化体系

## 一、为什么AI应用需要专属的灰度发布体系

### 1.1 传统灰度发布在AI场景的"水土不服"

传统的灰度发布（Canary Release）建立在一个核心假设上：**新版本和旧版本的行为差异是确定性的**。用户请求到同一个endpoint，相同输入必然产生相同输出。

但AI应用打破了这个假设：

```
传统软件灰度 vs AI应用灰度
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  传统软件:                                                        │
│  用户A → v1.0 → 结果X  ✅ (确定性，可预测)                        │
│  用户A → v2.0 → 结果X' ✅ (确定性，可预测)                        │
│  差异: 结构化可比较(X vs X')，可以做精确回归测试                   │
│                                                                   │
│  AI应用 (Prompt变更):                                              │
│  用户A → Prompt v1 → "请帮我总结这篇文章..." → 输出Y               │
│  用户A → Prompt v2 → "请帮我概括这篇文章..." → 输出Y'              │
│  差异: 非结构化，无法精确比较                                      │
│  可能: v2在大多数场景更好，但特定场景反而更差                        │
│  可能: 模型版本更新后，v2的优势消失                                 │
│                                                                   │
│  AI应用 (模型切换):                                                │
│  用户A → GPT-4o → 输出Y1                                          │
│  用户A → Claude-3.5 → 输出Y2                                      │
│  差异: 风格、长度、推理能力完全不同                                 │
│  可能: 同一个模型升级后，之前好的Prompt反而变差                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 AI灰度发布的五大特殊挑战

| 挑战 | 传统软件 | AI应用 | 影响 |
|------|---------|--------|------|
| **评估维度** | 二元(通过/失败) | 多维(质量/安全/延迟/成本) | 需要多指标综合决策 |
| **确定性** | 相同输入=相同输出 | 相同输入≠相同输出 | 需要统计显著性检验 |
| **延迟评估** | 即时可测 | 需要长期观察用户行为 | 灰度周期更长 |
| **交互影响** | 版本间隔离 | 用户对话上下文跨版本 | 需要会话级灰度 |
| **成本评估** | 极低 | Token成本可变 | 需要成本效益分析 |

---

## 二、AI灰度发布全景架构

### 2.1 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI应用灰度发布平台 (AI Canary Release Platform)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                       流量路由层 (Traffic Router)                 │        │
│  │                                                                  │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │        │
│  │  │ 用户分组  │  │ 流量染色  │  │ 会话保持  │  │ 降级回退  │       │        │
│  │  │ (Grouping)│  │ (Coloring)│  │(Sticky)  │  │(Fallback)│       │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    实验管理层 (Experiment Manager)                │        │
│  │                                                                  │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │        │
│  │  │ 实验配置  │  │ 版本管理  │  │ 渐进放量  │  │ 自动停止  │       │        │
│  │  │ (Config) │  │(Version) │  │(Ramp-up) │  │(AutoStop)│       │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    效果评估层 (Evaluation Engine)                 │        │
│  │                                                                  │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │        │
│  │  │ 指标采集  │  │ 统计分析  │  │ 用户反馈  │  │ 报告生成  │       │        │
│  │  │(Metrics) │  │(Stats)   │  │(Feedback)│  │(Report)  │       │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 流量染色策略

流量染色是灰度发布的基础设施。AI应用需要更精细的染色策略：

```python
from dataclasses import dataclass
from enum import Enum
import hashlib

class TrafficColor(Enum):
    """流量颜色定义"""
    CONTROL = "control"       # 对照组（当前线上版本）
    CANARY = "canary"         # 金丝雀（新版本，小流量）
    EXPERIMENT = "experiment" # 实验组（新版本，中等流量）
    ROLLOUT = "rollout"       # 全量组（准备全量发布）

@dataclass
class TrafficRule:
    """流量染色规则"""
    experiment_id: str
    variant_id: str
    color: TrafficColor
    percentage: float  # 0.0 - 1.0
    
class TrafficRouter:
    """AI应用流量路由器"""
    
    def __init__(self):
        self.rules: dict[str, list[TrafficRule]] = {}  # experiment_id -> rules
    
    def add_rule(self, experiment_id: str, rule: TrafficRule):
        """添加流量规则"""
        if experiment_id not in self.rules:
            self.rules[experiment_id] = []
        self.rules[experiment_id].append(rule)
    
    def route(self, user_id: str, experiment_id: str) -> TrafficColor:
        """根据用户ID确定流量颜色（确定性路由）"""
        rules = self.rules.get(experiment_id, [])
        if not rules:
            return TrafficColor.CONTROL
        
        # 使用一致性哈希确保同一用户始终路由到同一组
        hash_value = self._consistent_hash(user_id, experiment_id)
        
        cumulative = 0.0
        for rule in sorted(rules, key=lambda r: r.color.value):
            cumulative += rule.percentage
            if hash_value < cumulative:
                return rule.color
        
        return TrafficColor.CONTROL
    
    def route_with_session(self, user_id: str, session_id: str, 
                          experiment_id: str) -> TrafficColor:
        """会话级染色：同一会话内保持颜色一致"""
        # 会话ID的优先级高于用户ID
        return self.route(f"{user_id}:{session_id}", experiment_id)
    
    def _consistent_hash(self, key: str, experiment_id: str) -> float:
        """一致性哈希，返回0-1之间的值"""
        combined = f"{experiment_id}:{key}"
        hash_hex = hashlib.sha256(combined.encode()).hexdigest()
        return int(hash_hex[:8], 16) / 0xFFFFFFFF
    
    def get_traffic_split(self, experiment_id: str) -> dict:
        """获取当前流量分配情况"""
        rules = self.rules.get(experiment_id, [])
        return {
            rule.color.value: rule.percentage 
            for rule in rules
        }
```

---

## 三、Prompt版本管理与A/B测试

### 3.1 Prompt版本管理核心设计

Prompt是AI应用的核心资产之一，需要像代码一样管理版本：

```python
# Prompt版本管理
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class PromptVersion:
    """Prompt版本定义"""
    version_id: str               # 版本ID (e.g., "v2.3.1")
    template: str                 # Prompt模板
    variables: dict = field(default_factory=dict)  # 变量定义
    model: str = "gpt-4o"        # 目标模型
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # 元信息
    author: str = ""
    created_at: str = ""
    description: str = ""
    tags: list = field(default_factory=list)
    
    # 评估信息
    eval_score: Optional[float] = None  # 自动评估分数
    human_score: Optional[float] = None  # 人工评估分数

class PromptManager:
    """Prompt版本管理器"""
    
    def __init__(self):
        self.prompts: dict[str, list[PromptVersion]] = {}  # prompt_name -> versions
        self.active_versions: dict[str, str] = {}  # prompt_name -> active_version_id
    
    def register(self, name: str, version: PromptVersion):
        """注册新的Prompt版本"""
        if name not in self.prompts:
            self.prompts[name] = []
        self.prompts[name].append(version)
    
    def get_active(self, name: str) -> Optional[PromptVersion]:
        """获取当前活跃版本"""
        version_id = self.active_versions.get(name)
        if not version_id:
            return None
        return self._find_version(name, version_id)
    
    def set_active(self, name: str, version_id: str):
        """设置活跃版本"""
        version = self._find_version(name, version_id)
        if version:
            self.active_versions[name] = version_id
    
    def create_variant(self, name: str, base_version_id: str, 
                       template: str, description: str) -> PromptVersion:
        """基于现有版本创建变体（用于A/B测试）"""
        base = self._find_version(name, base_version_id)
        if not base:
            raise ValueError(f"Base version {base_version_id} not found")
        
        variant = PromptVersion(
            version_id=f"{base_version_id}-variant-{datetime.now().strftime('%H%M%S')}",
            template=template,
            variables=base.variables.copy(),
            model=base.model,
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            description=description,
            tags=base.tags + ["ab-test"]
        )
        
        self.register(name, variant)
        return variant
    
    def render(self, name: str, version_id: str, **kwargs) -> str:
        """渲染Prompt模板"""
        version = self._find_version(name, version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        rendered = version.template
        for key, value in kwargs.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered
    
    def _find_version(self, name: str, version_id: str) -> Optional[PromptVersion]:
        for v in self.prompts.get(name, []):
            if v.version_id == version_id:
                return v
        return None
```

### 3.2 A/B测试框架

这是AI灰度发布的核心——**如何科学地评估Prompt/模型变更的效果**。

```python
# AI应用A/B测试框架
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

class MetricType(Enum):
    """评估指标类型"""
    BINARY = "binary"           # 二元指标（通过/失败）
    RATING = "rating"           # 评分指标（1-5分）
    CONTINUOUS = "continuous"   # 连续指标（延迟、成本）
    PROPORTION = "proportion"   # 比例指标（点击率、满意度）

@dataclass
class ABTestConfig:
    """A/B测试配置"""
    test_id: str
    name: str
    
    # 变体定义
    control_id: str              # 对照组版本ID
    treatment_id: str            # 实验组版本ID
    
    # 流量配置
    traffic_percentage: float = 0.1  # 实验组流量比例
    
    # 评估指标
    primary_metric: str = ""      # 主要评估指标
    secondary_metrics: list = field(default_factory=list)
    
    # 统计配置
    min_sample_size: int = 100    # 最小样本量
    significance_level: float = 0.05  # 显著性水平
    power: float = 0.8           # 统计功效
    
    # 安全配置
    guardrail_metrics: dict = field(default_factory=dict)  # 护栏指标
    # 例: {"latency_p99": {"max": 5000}, "error_rate": {"max": 0.01}}

@dataclass
class ABTestResult:
    """A/B测试结果"""
    test_id: str
    status: str  # "running", "completed", "stopped"
    
    control_stats: dict = field(default_factory=dict)
    treatment_stats: dict = field(default_factory=dict)
    
    primary_result: dict = field(default_factory=dict)
    # {
    #     "metric": "satisfaction_score",
    #     "control_mean": 3.8,
    #     "treatment_mean": 4.2,
    #     "improvement": 0.105,  # 10.5%
    #     "p_value": 0.023,
    #     "confidence_interval": [0.012, 0.198],
    #     "is_significant": True
    # }
    
    guardrail_results: dict = field(default_factory=dict)
    recommendation: str = ""  # "ship", "extend", "rollback"

class ABTestEngine:
    """A/B测试引擎"""
    
    def __init__(self, metrics_store, prompt_manager):
        self.metrics = metrics_store
        self.prompt_mgr = prompt_manager
        self.active_tests: dict[str, ABTestConfig] = {}
    
    def create_test(self, config: ABTestConfig) -> str:
        """创建新的A/B测试"""
        self.active_tests[config.test_id] = config
        return config.test_id
    
    def record_observation(self, test_id: str, user_id: str, 
                          variant: str, metrics: dict):
        """记录一次观察数据"""
        self.metrics.store(
            test_id=test_id,
            user_id=user_id,
            variant=variant,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def analyze(self, test_id: str) -> ABTestResult:
        """分析A/B测试结果"""
        config = self.active_tests[test_id]
        
        # 获取两组数据
        control_data = self.metrics.get_variant_data(test_id, "control")
        treatment_data = self.metrics.get_variant_data(test_id, "treatment")
        
        # 样本量检查
        if len(control_data) < config.min_sample_size or \
           len(treatment_data) < config.min_sample_size:
            return ABTestResult(
                test_id=test_id,
                status="insufficient_data",
                recommendation=f"Need more samples: control={len(control_data)}, treatment={len(treatment_data)}"
            )
        
        # 主要指标分析
        primary_result = self._analyze_metric(
            control_data, treatment_data, config.primary_metric
        )
        
        # 护栏指标检查
        guardrail_results = self._check_guardrails(
            control_data, treatment_data, config.guardrail_metrics
        )
        
        # 综合推荐
        recommendation = self._make_recommendation(
            primary_result, guardrail_results
        )
        
        return ABTestResult(
            test_id=test_id,
            status="completed",
            primary_result=primary_result,
            guardrail_results=guardrail_results,
            recommendation=recommendation
        )
    
    def _analyze_metric(self, control: list, treatment: list, 
                       metric_name: str) -> dict:
        """分析单个指标"""
        control_values = [d[metric_name] for d in control if metric_name in d]
        treatment_values = [d[metric_name] for d in treatment if metric_name in d]
        
        control_mean = sum(control_values) / len(control_values)
        treatment_mean = sum(treatment_values) / len(treatment_values)
        
        # t检验
        t_stat, p_value = self._welch_t_test(control_values, treatment_values)
        
        # 效应量 (Cohen's d)
        pooled_std = math.sqrt(
            (self._variance(control_values) + self._variance(treatment_values)) / 2
        )
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # 置信区间
        improvement = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0
        ci = self._bootstrap_ci(control_values, treatment_values)
        
        return {
            "metric": metric_name,
            "control_mean": control_mean,
            "control_std": self._std(control_values),
            "treatment_mean": treatment_mean,
            "treatment_std": self._std(treatment_values),
            "improvement": improvement,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "confidence_interval": ci,
            "is_significant": p_value < 0.05 and abs(cohens_d) > 0.2
        }
    
    def _welch_t_test(self, x: list, y: list) -> tuple:
        """Welch t检验（不假设方差齐性）"""
        n1, n2 = len(x), len(y)
        m1, m2 = sum(x)/n1, sum(y)/n2
        v1, v2 = self._variance(x), self._variance(y)
        
        t_stat = (m2 - m1) / math.sqrt(v1/n1 + v2/n2) if (v1/n1 + v2/n2) > 0 else 0
        
        # 近似自由度 (Welch-Satterthwaite)
        num = (v1/n1 + v2/n2) ** 2
        den = (v1/n1)**2 / (n1-1) + (v2/n2)**2 / (n2-1)
        df = num / den if den > 0 else 1
        
        # 简化的p值近似（生产环境应使用scipy）
        p_value = self._t_distribution_p_value(t_stat, df)
        
        return t_stat, p_value
    
    def _check_guardrails(self, control: list, treatment: list, 
                         guardrails: dict) -> dict:
        """检查护栏指标"""
        results = {}
        for metric, limits in guardrails.items():
            control_values = [d[metric] for d in control if metric in d]
            treatment_values = [d[metric] for d in treatment if metric in d]
            
            control_mean = sum(control_values) / max(len(control_values), 1)
            treatment_mean = sum(treatment_values) / max(len(treatment_values), 1)
            
            breached = False
            if "max" in limits and treatment_mean > limits["max"]:
                breached = True
            if "min" in limits and treatment_mean < limits["min"]:
                breached = True
            
            results[metric] = {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "limit": limits,
                "breached": breached
            }
        
        return results
    
    def _make_recommendation(self, primary: dict, guardrails: dict) -> str:
        """综合推荐"""
        # 1. 护栏被突破 → 回滚
        any_guardrail_breached = any(g.get("breached", False) for g in guardrails.values())
        if any_guardrail_breached:
            return "rollback"
        
        # 2. 主要指标显著提升 → 上线
        if primary.get("is_significant", False) and primary.get("improvement", 0) > 0:
            return "ship"
        
        # 3. 主要指标显著下降 → 回滚
        if primary.get("is_significant", False) and primary.get("improvement", 0) < 0:
            return "rollback"
        
        # 4. 不显著但趋势正向 → 继续观察
        if primary.get("improvement", 0) > 0:
            return "extend"
        
        # 5. 其他 → 放弃
        return "rollback"
    
    # 数学工具函数
    def _variance(self, data: list) -> float:
        n = len(data)
        mean = sum(data) / n
        return sum((x - mean) ** 2 for x in data) / (n - 1) if n > 1 else 0
    
    def _std(self, data: list) -> float:
        return math.sqrt(self._variance(data))
    
    def _t_distribution_p_value(self, t: float, df: float) -> float:
        """t分布p值的近似计算"""
        x = df / (df + t ** 2)
        # Beta函数的近似
        if t > 0:
            return 0.5 * x ** (df / 2)
        else:
            return 1 - 0.5 * x ** (df / 2)
    
    def _bootstrap_ci(self, x: list, y: list, n_bootstrap: int = 1000, 
                     confidence: float = 0.95) -> list:
        """Bootstrap置信区间"""
        import random
        diffs = []
        for _ in range(n_bootstrap):
            x_sample = [random.choice(x) for _ in range(len(x))]
            y_sample = [random.choice(y) for _ in range(len(y))]
            diffs.append(sum(y_sample)/len(y_sample) - sum(x_sample)/len(x_sample))
        
        diffs.sort()
        lower = int(n_bootstrap * (1 - confidence) / 2)
        upper = int(n_bootstrap * (1 + confidence) / 2)
        return [diffs[lower], diffs[upper]]
```

### 3.3 评估指标体系

AI应用的评估需要多维度指标：

```
AI应用A/B测试指标体系
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. 质量指标 (Quality Metrics)                                   │
│  ├── 用户满意度评分 (1-5分)                                       │
│  ├── 任务完成率 (用户是否达到了目的)                               │
│  ├── 回答准确率 (可自动评估的部分)                                 │
│  └── 相关性评分 (回答与问题的匹配度)                               │
│                                                                  │
│  2. 效率指标 (Efficiency Metrics)                                 │
│  ├── 首Token延迟 (TTFT)                                         │
│  ├── 端到端延迟                                                  │
│  ├── 吞吐量 (RPS)                                               │
│  └── Token消耗量                                                 │
│                                                                  │
│  3. 成本指标 (Cost Metrics)                                      │
│  ├── 每次请求成本 ($)                                            │
│  ├── 每用户每天成本                                              │
│  └── 成本效益比 (质量提升/成本增加)                               │
│                                                                  │
│  4. 安全指标 (Safety Metrics)                                    │
│  ├── 有害内容生成率                                              │
│  ├── 拒绝率 (不该拒绝的被拒绝)                                    │
│  └── 幻觉率 (事实性错误比例)                                     │
│                                                                  │
│  5. 用户行为指标 (Behavior Metrics)                               │
│  ├── 续用率 (是否继续使用)                                        │
│  ├── 追问率 (是否需要更多澄清)                                    │
│  ├── 重新生成率 (是否不满意重试)                                   │
│  └── 会话长度                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、渐进式放量策略

### 4.1 放量阶梯设计

```
渐进式放量策略 (Progressive Rollout)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  Phase 1: 内部测试 (Internal Testing)                                │
│  ├── 流量: 0% → 2% (仅内部员工)                                      │
│  ├── 持续: 1-3天                                                      │
│  ├── 关注: 基本功能正确性、严重Bug                                     │
│  └── 退出条件: 无P0级Bug                                              │
│                                                                       │
│  Phase 2: 种子用户 (Seed Users)                                      │
│  ├── 流量: 2% → 10% (种子用户 + 1%随机用户)                          │
│  ├── 持续: 3-5天                                                      │
│  ├── 关注: 质量指标、延迟、成本                                        │
│  └── 退出条件: 主要指标不劣化 + 护栏指标不突破                         │
│                                                                       │
│  Phase 3: 小流量 (Small Traffic)                                     │
│  ├── 流量: 10% → 30%                                                 │
│  ├── 持续: 5-7天                                                      │
│  ├── 关注: 统计显著性、长尾Case                                       │
│  └── 退出条件: 主要指标显著提升                                        │
│                                                                       │
│  Phase 4: 大流量 (Large Traffic)                                     │
│  ├── 流量: 30% → 50%                                                 │
│  ├── 持续: 3-5天                                                      │
│  ├── 关注: 稳定性、极端场景                                           │
│  └── 退出条件: 全面验证通过                                            │
│                                                                       │
│  Phase 5: 全量 (Full Rollout)                                        │
│  ├── 流量: 50% → 100%                                                │
│  ├── 持续: 1-2天                                                      │
│  ├── 关注: 全量后的系统表现                                            │
│  └── 完成条件: 稳定运行24小时                                         │
│                                                                       │
│  ⚠️ 任意阶段触发护栏指标 → 自动回滚                                    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 自动化渐进放量

```python
class ProgressiveRollout:
    """自动化渐进放量控制器"""
    
    PHASES = [
        {"name": "internal",  "traffic": 0.02, "duration_days": 2, "auto_promote": True},
        {"name": "seed",      "traffic": 0.10, "duration_days": 4, "auto_promote": True},
        {"name": "small",     "traffic": 0.30, "duration_days": 5, "auto_promote": True},
        {"name": "large",     "traffic": 0.50, "duration_days": 3, "auto_promote": False},
        {"name": "full",      "traffic": 1.00, "duration_days": 1, "auto_promote": False},
    ]
    
    def __init__(self, traffic_router, ab_test_engine, alert_service):
        self.router = traffic_router
        self.ab_engine = ab_test_engine
        self.alerts = alert_service
        self.rollout_states: dict[str, dict] = {}
    
    def start_rollout(self, test_id: str) -> str:
        """启动渐进放量"""
        self.rollout_states[test_id] = {
            "current_phase": 0,
            "phase_start": datetime.now(),
            "status": "running"
        }
        
        # 应用第一阶段流量
        phase = self.PHASES[0]
        self._apply_traffic(test_id, phase["traffic"])
        
        return f"Rollout started: Phase 0 ({phase['name']}) at {phase['traffic']*100}% traffic"
    
    def check_and_advance(self, test_id: str):
        """检查是否应该推进到下一阶段"""
        state = self.rollout_states[test_id]
        phase = self.PHASES[state["current_phase"]]
        
        # 1. 检查护栏指标
        if self._check_guardrails_failed(test_id):
            self._trigger_rollback(test_id, "Guardrail metrics breached")
            return
        
        # 2. 检查当前阶段的评估结果
        result = self.ab_engine.analyze(test_id)
        
        if result.recommendation == "rollback":
            self._trigger_rollback(test_id, f"Negative results: {result.primary_result}")
            return
        
        # 3. 检查时间是否到达
        days_elapsed = (datetime.now() - state["phase_start"]).days
        if days_elapsed < phase["duration_days"]:
            return  # 还没到时间
        
        # 4. 检查是否应该推进
        should_advance = False
        
        if phase["auto_promote"]:
            # 自动阶段：指标不劣化就推进
            should_advance = result.recommendation in ("ship", "extend")
        else:
            # 手动阶段：需要显著提升才推进
            should_advance = result.recommendation == "ship"
        
        if should_advance:
            self._advance_phase(test_id)
        else:
            # 给更多时间
            self.alerts.send(
                f"Phase {state['current_phase']} extended: "
                f"Recommendation={result.recommendation}"
            )
    
    def _advance_phase(self, test_id: str):
        """推进到下一阶段"""
        state = self.rollout_states[test_id]
        state["current_phase"] += 1
        state["phase_start"] = datetime.now()
        
        if state["current_phase"] >= len(self.PHASES):
            state["status"] = "completed"
            self._apply_traffic(test_id, 1.0)
            self.alerts.send(f"Rollout {test_id}: Full rollout completed!")
            return
        
        phase = self.PHASES[state["current_phase"]]
        self._apply_traffic(test_id, phase["traffic"])
        
        self.alerts.send(
            f"Rollout {test_id}: Advanced to Phase {state['current_phase']} "
            f"({phase['name']}) at {phase['traffic']*100}% traffic"
        )
    
    def _trigger_rollback(self, test_id: str, reason: str):
        """触发回滚"""
        state = self.rollout_states[test_id]
        state["status"] = "rolled_back"
        self._apply_traffic(test_id, 0.0)  # 流量全部回到对照组
        self.alerts.send_alert(
            f"🚨 Rollback triggered for {test_id}: {reason}",
            severity="critical"
        )
    
    def _apply_traffic(self, test_id: str, traffic_pct: float):
        """应用流量比例"""
        config = self.ab_engine.active_tests.get(test_id)
        if config:
            rule = TrafficRule(
                experiment_id=test_id,
                variant_id=config.treatment_id,
                color=TrafficColor.CANARY,
                percentage=traffic_pct
            )
            self.router.add_rule(test_id, rule)
    
    def _check_guardrails_failed(self, test_id: str) -> bool:
        """检查护栏指标"""
        result = self.ab_engine.analyze(test_id)
        return any(
            g.get("breached", False) 
            for g in result.guardrail_results.values()
        )
```

---

## 五、在线效果评估体系

### 5.1 自动化评估Pipeline

```python
# 在线效果自动化评估
class OnlineEvaluator:
    """在线效果自动化评估"""
    
    def __init__(self, llm_as_judge, feedback_store):
        self.judge = llm_as_judge
        self.feedback = feedback_store
    
    def evaluate_response(self, query: str, response: str, 
                         context: list[dict] = None) -> dict:
        """评估单次响应质量"""
        
        # 1. LLM-as-Judge 评估
        judge_result = self._llm_judge(query, response, context)
        
        # 2. 自动化指标计算
        auto_metrics = self._compute_auto_metrics(query, response)
        
        # 3. 安全检查
        safety_result = self._safety_check(response)
        
        return {
            "judge_scores": judge_result,
            "auto_metrics": auto_metrics,
            "safety": safety_result,
            "overall_quality": self._compute_overall(
                judge_result, auto_metrics, safety_result
            )
        }
    
    def _llm_judge(self, query: str, response: str, 
                   context: list[dict] = None) -> dict:
        """LLM-as-Judge评估"""
        prompt = f"""请作为评审专家，评估以下AI回答的质量。

用户问题: {query}

AI回答: {response}

{f"参考上下文: {context}" if context else ""}

请从以下维度评分（1-5分）：

1. **准确性**: 回答中的事实是否正确？
2. **完整性**: 是否全面回答了问题的所有方面？
3. **相关性**: 回答是否与问题高度相关？
4. **可读性**: 回答的表达是否清晰、易于理解？
5. **有用性**: 回答对用户是否有实际帮助？

请以JSON格式输出，包含每个维度的分数和总体评语：
{{"accuracy": 4, "completeness": 3, "relevance": 5, "readability": 4, "usefulness": 4, "overall": "回答质量良好，但可以更详细..."}}
"""
        
        raw = self.judge.generate(prompt)
        return self._parse_judge_response(raw)
    
    def _compute_auto_metrics(self, query: str, response: str) -> dict:
        """计算自动化指标"""
        return {
            "response_length": len(response),
            "query_response_ratio": len(response) / max(len(query), 1),
            "contains_code": "```" in response,
            "contains_list": any(line.strip().startswith(("-", "*", "1.", "2.")) 
                                for line in response.split("\n")),
            "language_match": self._detect_language(query) == self._detect_language(response),
        }
    
    def _safety_check(self, response: str) -> dict:
        """安全检查"""
        return {
            "has_harmful_content": self._detect_harmful(response),
            "has_pii": self._detect_pii(response),
            "has_bias": self._detect_bias(response),
        }
    
    def _compute_overall(self, judge, auto, safety) -> float:
        """计算综合质量分"""
        if safety.get("has_harmful_content"):
            return 0.0  # 有有害内容，直接0分
        
        judge_avg = sum(judge.values()) / max(len(judge), 1)
        return judge_avg
    
    def batch_evaluate(self, interactions: list[dict]) -> dict:
        """批量评估并生成统计报告"""
        results = [self.evaluate_response(**i) for i in interactions]
        
        scores = [r["overall_quality"] for r in results]
        
        return {
            "count": len(results),
            "mean_score": sum(scores) / len(scores),
            "score_distribution": self._histogram(scores),
            "safety_issues": sum(1 for r in results if r["safety"].get("has_harmful_content")),
            "high_quality_rate": sum(1 for s in scores if s >= 4) / len(scores),
            "low_quality_rate": sum(1 for s in scores if s < 3) / len(scores),
        }
```

### 5.2 用户反馈闭环

```
用户反馈收集与闭环
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  反馈收集渠道                                                  │
│  ├── 显式反馈: 👍/👎 点击 + 评分 + 文字反馈                     │
│  ├── 隐式反馈: 重新生成、复制、追问、放弃                        │
│  └── 行为信号: 停留时间、阅读完成率                             │
│                                                               │
│  反馈处理流程                                                   │
│  ├── 1. 实时聚合: 按版本/实验组聚合                             │
│  ├── 2. 趋势监控: 监控反馈率变化趋势                             │
│  ├── 3. 异常检测: 自动检测反馈率突降/突升                       │
│  └── 4. 归因分析: 定位导致反馈变化的具体版本/变更                │
│                                                               │
│  闭环行动                                                      │
│  ├── 反馈率下降 → 触发根因分析 → 针对性修复                     │
│  ├── 负面反馈集中 → 人工审核 → 问题定位 → Prompt调整            │
│  └── 正面反馈集中 → 提取最佳实践 → 推广到其他场景               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

```python
class FeedbackCollector:
    """用户反馈收集器"""
    
    def __init__(self, store, alert_service):
        self.store = store
        self.alerts = alert_service
    
    def record_feedback(self, interaction_id: str, user_id: str,
                       experiment_id: str, variant: str,
                       feedback_type: str, value: float = None,
                       comment: str = None):
        """记录用户反馈"""
        self.store.insert({
            "interaction_id": interaction_id,
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant,
            "feedback_type": feedback_type,  # "like", "dislike", "rating"
            "value": value,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_implicit_feedback(self, interaction_id: str, 
                                 signals: dict):
        """记录隐式反馈信号"""
        self.store.insert({
            "interaction_id": interaction_id,
            "type": "implicit",
            "signals": signals,
            # signals可能包含:
            # {"regenerated": True, "copied": True, "session_length": 5}
            "timestamp": datetime.now().isoformat()
        })
    
    def compute_feedback_rate(self, experiment_id: str, 
                              variant: str,
                              time_window_hours: int = 24) -> dict:
        """计算反馈率"""
        interactions = self.store.get_interactions(
            experiment_id, variant, time_window_hours
        )
        
        total = len(interactions)
        positive = sum(1 for i in interactions if i.get("feedback_type") == "like")
        negative = sum(1 for i in interactions if i.get("feedback_type") == "dislike")
        
        return {
            "total_interactions": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "positive_rate": positive / max(total, 1),
            "negative_rate": negative / max(total, 1),
            "net_positive_rate": (positive - negative) / max(total, 1),
        }
```

---

## 六、实战案例：完整的Prompt A/B测试流程

### 6.1 场景描述

假设我们有一个AI客服助手，当前Prompt (v1) 的用户满意度为3.6/5分。我们设计了新的Prompt (v2)，目标是提升满意度到4.0/5分。

### 6.2 完整流程

```
完整A/B测试流程时间线
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  Day 0: 准备阶段                                                          │
│  ├── 注册Prompt v1和v2到版本管理系统                                        │
│  ├── 定义评估指标: 主要指标=满意度评分, 护栏指标=延迟<3s, 成本<$0.01/次    │
│  ├── 创建A/B测试配置: 10%流量给v2                                          │
│  └── 启动渐进放量: Phase 0 (内部测试)                                      │
│                                                                           │
│  Day 1-2: 内部测试                                                        │
│  ├── 内部员工测试100+次交互                                                │
│  ├── 检查: 基本功能正确、无P0级Bug                                         │
│  ├── 发现: v2在多轮对话中偶尔丢失上下文                                     │
│  ├── 修复: 调整v2的context注入策略                                         │
│  └── 推进到Phase 1                                                        │
│                                                                           │
│  Day 3-6: 种子用户测试                                                     │
│  ├── 10%流量分配给v2 (约1000+次交互/天)                                    │
│  ├── Day 4: v2满意度评分4.1, v1为3.6                                      │
│  ├── Day 5: 发现v2延迟比v1高15%, 但仍在护栏范围内                           │
│  ├── Day 6: 统计检验p=0.08, 接近显著                                       │
│  └── 决策: 继续观察, 推进到Phase 2                                         │
│                                                                           │
│  Day 7-11: 小流量测试                                                      │
│  ├── 30%流量分配给v2 (约3000+次交互/天)                                    │
│  ├── Day 9: p值达到0.02, 统计显著                                          │
│  ├── Day 10: 满意度提升11%, 延迟增加8%, 成本增加3%                          │
│  ├── Day 11: 护栏指标全部正常                                              │
│  └── 决策: 准备全量发布                                                     │
│                                                                           │
│  Day 12-14: 全量发布                                                       │
│  ├── 逐步提升到50% → 100%                                                 │
│  ├── Day 14: 稳定运行24小时                                                │
│  └── 完成: v2正式成为新的v1                                                 │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.3 关键代码：端到端流程编排

```python
# 端到端A/B测试流程编排
class ABTestOrchestrator:
    """A/B测试流程编排器"""
    
    def __init__(self, prompt_mgr, traffic_router, ab_engine, 
                 rollout_controller, evaluator, feedback_collector):
        self.prompts = prompt_mgr
        self.router = traffic_router
        self.ab_engine = ab_engine
        self.rollout = rollout_controller
        self.evaluator = evaluator
        self.feedback = feedback_collector
    
    def run_full_test(self, 
                     prompt_name: str,
                     control_version: str,
                     treatment_version: str,
                     primary_metric: str = "satisfaction_score",
                     target_improvement: float = 0.10) -> dict:
        """运行完整的A/B测试流程"""
        
        # 1. 创建测试
        test_id = f"ab-{prompt_name}-{datetime.now().strftime('%Y%m%d')}"
        config = ABTestConfig(
            test_id=test_id,
            name=f"Prompt A/B Test: {prompt_name}",
            control_id=control_version,
            treatment_id=treatment_version,
            traffic_percentage=0.1,
            primary_metric=primary_metric,
            guardrail_metrics={
                "latency_ms": {"max": 3000},
                "cost_per_request": {"max": 0.01},
                "error_rate": {"max": 0.01}
            },
            min_sample_size=200
        )
        self.ab_engine.create_test(config)
        
        # 2. 启动渐进放量
        self.rollout.start_rollout(test_id)
        
        # 3. 返回监控信息
        return {
            "test_id": test_id,
            "status": "started",
            "next_check": "Run check_and_advance() periodically",
            "monitoring": {
                "primary_metric": primary_metric,
                "target_improvement": f"{target_improvement*100}%",
                "guardrails": config.guardrail_metrics
            }
        }
    
    def handle_request(self, user_id: str, session_id: str,
                      experiment_id: str, query: str) -> dict:
        """处理用户请求（运行时路由）"""
        
        # 1. 路由到对应变体
        color = self.router.route_with_session(user_id, session_id, experiment_id)
        
        # 2. 获取对应版本的Prompt
        config = self.ab_engine.active_tests.get(experiment_id)
        if color == TrafficColor.CONTROL:
            version_id = config.control_id
        else:
            version_id = config.treatment_id
        
        # 3. 渲染Prompt并调用模型
        prompt = self.prompts.render(
            "customer_service", version_id, 
            query=query
        )
        
        # 4. 调用模型（省略具体调用）
        # response = call_model(prompt, ...)
        
        # 5. 评估响应质量
        # eval_result = self.evaluator.evaluate_response(query, response)
        
        # 6. 记录观察数据
        # self.ab_engine.record_observation(
        #     experiment_id, user_id, color.value,
        #     metrics={
        #         "satisfaction_score": eval_result["overall_quality"],
        #         "latency_ms": latency,
        #         "cost_per_request": cost
        #     }
        # )
        
        return {
            "variant": version_id,
            "color": color.value,
            # "response": response
        }
```

---

## 七、最佳实践与常见陷阱

### 7.1 最佳实践清单

```
AI灰度发布最佳实践
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  1. 永远保留回滚能力                                               │
│  ├── 每个版本都要可一键回滚                                        │
│  ├── 回滚决策阈值要提前定义                                        │
│  └── 回滚后的流量恢复要平滑                                        │
│                                                                   │
│  2. 护栏指标是底线                                                  │
│  ├── 延迟、成本、错误率是必须监控的护栏                             │
│  ├── 护栏指标突破 → 自动回滚，不需要人工判断                       │
│  └── 护栏指标的阈值要基于历史数据设定                               │
│                                                                   │
│  3. 样本量要够                                                      │
│  ├── 小样本的统计检验不可靠                                        │
│  ├── AI输出的方差通常比传统A/B测试更大                              │
│  └── 宁可多测几天，不要过早做决策                                   │
│                                                                   │
│  4. 关注长期效应                                                     │
│  ├── 用户行为可能需要时间才能体现                                   │
│  ├── 短期满意度提升 ≠ 长期留存提升                                  │
│  └── 关键测试至少观察7天                                            │
│                                                                   │
│  5. 记录一切                                                        │
│  ├── 每次变更的Prompt版本、模型版本、参数                            │
│  ├── 每个实验的完整评估报告                                        │
│  └── 失败的实验同样有价值，记录教训                                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 常见陷阱

| 陷阱 | 表现 | 解决方案 |
|------|------|---------|
| **样本量不足** | p=0.08就宣布"趋势向好" | 等待p<0.05，至少200样本 |
| **忽略多重检验** | 同时测10个指标，有一个显著就宣布成功 | 使用Bonferroni校正或FDR控制 |
| **观察者效应** | 人工审核影响了样本选择 | 严格随机化，审核与抽样解耦 |
| **短期偏差** | 新版本初期新鲜感导致好评 | 至少观察7天，关注趋势而非绝对值 |
| **护城河思维** | "我们已经投入了3周，不能放弃" | 设定明确的止损条件并严格执行 |

---

## 八、总结

AI应用的灰度发布不是传统灰度的简单复制，而是需要一套**专属的工程化体系**：

1. **流量染色**：基于一致性哈希的确定性路由，支持会话级保持
2. **版本管理**：Prompt像代码一样管理版本，支持分支和变体
3. **A/B测试**：多维度评估 + 统计显著性检验 + 护栏指标
4. **渐进放量**：自动化阶梯式放量，异常自动回滚
5. **效果评估**：LLM-as-Judge + 用户反馈闭环

**核心原则**：在AI的不确定性中建立确定性的工程化流程。每一次Prompt变更都应该是可控、可观测、可回滚的。

---

> **参考文献**
> 1. Kohavi, R. et al. "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing" (2020)
> 2. Zheng, L. et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
> 3. Anthropic. "Model Card and Evaluations for Claude Models" (2024)
> 4. OpenAI. "GPT-4 System Card" (2023)
> 5. Google. "PaLM 2 Technical Report" (2023)
