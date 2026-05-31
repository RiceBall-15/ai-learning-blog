---
title: "LLM应用AB测试与灰度发布实战：从实验设计到效果评估的完整指南"
description: "系统讲解LLM应用的实验驱动优化方法论，覆盖AB测试设计、灰度发布策略、在线评估指标体系与生产级实验平台架构"
date: 2026-06-01
author: "RiceBall"
category: "aiInfra"
subCategory: "evaluation"
tags: ["LLM评估", "AB测试", "灰度发布", "在线评估", "MLOps"]
draft: false
---

## 引言：LLM应用的优化困境

"换了个Prompt，回复质量到底提升了没有？"

"新模型比旧模型好，但好多少？值不值得迁移？"

"RAG加了Reranker，用户满意度真的提高了吗？"

这些问题的本质是：**如何科学地评估LLM应用的变更效果**。传统的离线评估（Benchmark）只能告诉你模型在测试集上的表现，却无法回答"在线上真实场景中，变更是否带来了正向效果"。

本文将系统讲解LLM应用的AB测试与灰度发布方法论，涵盖实验设计、分流策略、在线指标体系、统计显著性分析，以及生产级实验平台的架构设计。

## 一、为什么LLM应用的AB测试更难

与传统Web应用相比，LLM应用的AB测试面临独特挑战：

| 挑战 | 传统Web应用 | LLM应用 |
|------|------------|---------|
| **输出确定性** | 相同输入相同输出 | 同一输入多次调用产生不同输出 |
| **评估维度** | 转化率、点击率等单一指标 | 质量、相关性、安全性等多维度 |
| **延迟影响** | 响应时间差异小 | 模型切换可能导致延迟显著变化 |
| **成本因素** | 基础设施成本相对固定 | Token消耗差异可导致成本倍增 |
| **反馈周期** | 即时行为数据 | 需要人工标注或隐式反馈积累 |
| **A/B边界模糊** | 用户看到不同版本 | 用户可能同时与两个版本交互 |

### 1.1 LLM应用的特殊分流挑战

传统AB测试基于用户ID或设备ID分流，保证同一用户始终看到同一版本。但LLM应用需要考虑：

```
场景：客服Agent的Prompt优化实验

用户A：今天咨询退款 → 走版本A（优化后的Prompt）
用户A：明天咨询退货 → 走版本B（旧版Prompt）

问题：用户可能在两个版本间"跳跃"，导致体验不一致
```

**解决方案：会话级分流 + 用户级兜底**

```python
class LLMSessionRouter:
    """LLM应用分流路由器"""

    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        # 会话级缓存，保证同一会话内版本一致
        self.session_cache = LRUCache(maxsize=100_000)

    def route(
        self,
        user_id: str,
        session_id: str,
        request_context: dict
    ) -> ExperimentVariant:
        # 1. 检查会话缓存（保证会话内一致性）
        if session_id in self.session_cache:
            return self.session_cache[session_id]

        # 2. 用户级一致性检查（保证同一用户体验连续）
        user_variant = self.get_user_variant(user_id)
        if user_variant:
            self.session_cache[session_id] = user_variant
            return user_variant

        # 3. 新用户/新会话：基于哈希的确定性分流
        hash_input = f"{user_id}:{self.config.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        variant_index = hash_value % self.config.traffic_split_total

        variant = self.config.variants[variant_index]

        # 4. 写入缓存
        self.session_cache[session_id] = variant
        self.set_user_variant(user_id, variant)

        return variant
```

## 二、LLM应用AB测试的完整流程

### 2.1 实验设计框架

一个完整的LLM应用AB测试包含以下阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM AB测试全生命周期                        │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 1.假设    │→│ 2.设计    │→│ 3.实施    │→│ 4.分析    │   │
│  │   提出    │  │   实验    │  │   分流    │  │   结果    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       │             │              │              │          │
│  ┌────┴────┐  ┌─────┴─────┐  ┌────┴────┐  ┌────┴─────┐   │
│  │明确问题  │  │指标定义    │  │流量分配  │  │统计检验   │   │
│  │预期效果  │  │样本量计算  │  │监控告警  │  │效果归因   │   │
│  │成功标准  │  │分流策略    │  │日志采集  │  │决策建议   │   │
│  └─────────┘  └───────────┘  └─────────┘  └──────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 5. 决策与上线                                          │   │
│  │  ├─ 全量发布  ├─ 回滚  ├─ 继续实验  └─ 扩大实验      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心指标体系

LLM应用的在线评估需要多维度指标体系：

```yaml
# metrics_definition.yaml

primary_metrics:
  - name: "user_satisfaction_score"
    description: "用户满意度综合得分"
    calculation: "加权平均(任务完成率, 用户评分, 交互轮次效率)"
    weight: 0.4
    data_source: "implicit_feedback"

  - name: "task_completion_rate"
    description: "任务完成率"
    calculation: "成功完成任务的会话数 / 总会话数"
    weight: 0.3
    data_source: "conversation_logs"

secondary_metrics:
  - name: "response_quality_score"
    description: "回复质量评分（LLM-as-Judge）"
    calculation: "gpt-4o对回复质量的1-5分评估均值"
    weight: 0.15
    data_source: "automated_evaluation"

  - name: "first_turn_resolution"
    description: "首轮解决率"
    calculation: "无需追问即解决的会话比例"
    weight: 0.1
    data_source: "conversation_logs"

guardrail_metrics:
  - name: "hallucination_rate"
    description: "幻觉率"
    calculation: "检测到幻觉的回复数 / 总回复数"
    threshold: "< 3%"
    action: "超过阈值自动停止实验"

  - name: "safety_violation_rate"
    description: "安全违规率"
    calculation: "触发安全策略的请求数 / 总请求数"
    threshold: "< 0.1%"
    action: "超过阈值立即回滚"

  - name: "p95_latency_ms"
    description: "P95延迟"
    calculation: "第95百分位响应延迟"
    threshold: "< 5000ms"
    action: "超过阈值告警"

  - name: "cost_per_request"
    description: "单次请求成本"
    calculation: "Token消耗 × 单价"
    threshold: "< 1.5x baseline"
    action: "超过阈值告警"
```

### 2.3 样本量与实验时长计算

LLM应用的样本量计算需要考虑输出的高方差性：

```python
import math
from scipy import stats

def calculate_sample_size(
    baseline_rate: float,       # 基线指标值（如任务完成率）
    minimum_detectable_effect: float,  # 最小可检测效应（如提升5%）
    statistical_power: float = 0.8,    # 统计功效
    significance_level: float = 0.05,  # 显著性水平
    output_variance_factor: float = 1.5  # LLM输出的方差放大因子
) -> dict:
    """
    计算LLM应用AB测试所需样本量

    LLM输出的随机性导致方差更大，需要引入variance_factor
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    # 标准AB测试样本量公式
    z_alpha = stats.norm.ppf(1 - significance_level / 2)
    z_beta = stats.norm.ppf(statistical_power)

    # 合并比例
    p_bar = (p1 + p2) / 2

    # 标准方差
    variance = (p1 * (1 - p1) + p2 * (1 - p2))

    # LLM方差放大：由于输出的随机性，实际方差更大
    adjusted_variance = variance * output_variance_factor

    n_per_group = (
        (z_alpha + z_beta) ** 2
        * adjusted_variance
        / (p2 - p1) ** 2
    )

    n_per_group = math.ceil(n_per_group)

    return {
        "sample_per_group": n_per_group,
        "total_sample": n_per_group * 2,
        "estimated_days": math.ceil(n_per_group / daily_traffic_per_group),
        "parameters": {
            "baseline": p1,
            "expected": p2,
            "mde": minimum_detectable_effect,
            "variance_factor": output_variance_factor
        }
    }

# 示例：检测任务完成率从70%提升到73.5%（5%相对提升）
result = calculate_sample_size(
    baseline_rate=0.70,
    minimum_detectable_effect=0.05,
    output_variance_factor=1.5
)
print(f"每组需要 {result['sample_per_group']} 个样本")
print(f"预计需要 {result['estimated_days']} 天")
```

## 三、灰度发布策略

### 3.1 渐进式灰度发布模型

与传统AB测试的50/50分流不同，LLM应用的灰度发布应采用渐进式策略：

```
阶段1: 金丝雀发布（Canary）
├─ 流量比例: 1-5%
├─ 持续时间: 1-2小时
├─ 监控频率: 实时
├─ 自动回滚: 安全指标超阈值立即回滚
└─ 关注重点: 错误率、延迟、安全违规

阶段2: 小流量验证
├─ 流量比例: 10-20%
├─ 持续时间: 24-48小时
├─ 监控频率: 每小时
├─ 自动回滚: 任一Guardrail指标超阈值
└─ 关注重点: 质量指标、用户反馈

阶段3: 中流量实验
├─ 流量比例: 30-50%
├─ 持续时间: 3-7天
├─ 监控频率: 每天
├─ 自动回滚: 统计显著的负面效果
└─ 关注重点: 综合效果、成本影响

阶段4: 全量发布
├─ 流量比例: 100%
├─ 持续时间: 持续监控
├─ 监控频率: 每天
├─ 保留回滚能力: 7天
└─ 关注重点: 长期效果、边缘案例
```

### 3.2 灰度发布状态机

```python
from enum import Enum
from datetime import datetime, timedelta

class CanaryStage(Enum):
    CANARY = "canary"           # 1-5% 流量
    SMALL_TRAFFIC = "small"     # 10-20%
    MEDIUM_TRAFFIC = "medium"   # 30-50%
    FULL_ROLLOUT = "full"       # 100%
    ROLLED_BACK = "rollback"    # 已回滚
    PAUSED = "paused"           # 实验暂停

class CanaryDeployment:
    """灰度发布状态机"""

    # 阶段配置
    STAGE_CONFIG = {
        CanaryStage.CANARY: {
            "traffic_pct": 5,
            "min_duration_hours": 1,
            "auto_escalate_hours": 2,
            "monitoring_interval_minutes": 5,
        },
        CanaryStage.SMALL_TRAFFIC: {
            "traffic_pct": 20,
            "min_duration_hours": 24,
            "auto_escalate_hours": 48,
            "monitoring_interval_minutes": 60,
        },
        CanaryStage.MEDIUM_TRAFFIC: {
            "traffic_pct": 50,
            "min_duration_hours": 72,
            "auto_escalate_hours": 168,  # 7天
            "monitoring_interval_minutes": 360,
        },
        CanaryStage.FULL_ROLLOUT: {
            "traffic_pct": 100,
            "min_duration_hours": 0,
            "auto_escalate_hours": None,
            "monitoring_interval_minutes": 1440,
        },
    }

    # 阶段流转规则
    TRANSITIONS = {
        CanaryStage.CANARY: CanaryStage.SMALL_TRAFFIC,
        CanaryStage.SMALL_TRAFFIC: CanaryStage.MEDIUM_TRAFFIC,
        CanaryStage.MEDIUM_TRAFFIC: CanaryStage.FULL_ROLLOUT,
    }

    def __init__(self, experiment_id: str, config: dict):
        self.experiment_id = experiment_id
        self.current_stage = CanaryStage.CANARY
        self.stage_start_time = datetime.utcnow()
        self.metrics_history = []

    async def evaluate_and_advance(self, metrics: dict) -> CanaryDecision:
        """评估当前阶段指标，决定是否推进或回滚"""

        # 1. 检查Guardrail指标（任何超阈值立即回滚）
        for metric_name, value in metrics.items():
            guardrail = self.config.guardrails.get(metric_name)
            if guardrail and value > guardrail.threshold:
                return CanaryDecision(
                    action="rollback",
                    reason=f"Guardrail {metric_name} breached: "
                           f"{value} > {guardrail.threshold}"
                )

        # 2. 检查最小停留时间
        stage_config = self.STAGE_CONFIG[self.current_stage]
        elapsed = datetime.utcnow() - self.stage_start_time
        if elapsed < timedelta(hours=stage_config["min_duration_hours"]):
            return CanaryDecision(action="hold", reason="min_duration not met")

        # 3. 评估是否满足晋级条件
        if self.check_promotion_criteria(metrics):
            next_stage = self.TRANSITIONS.get(self.current_stage)
            if next_stage:
                return CanaryDecision(
                    action="promote",
                    new_stage=next_stage,
                    new_traffic_pct=self.STAGE_CONFIG[next_stage]["traffic_pct"]
                )

        # 4. 检查是否需要人工干预
        if self.check_escalation_needed(metrics):
            return CanaryDecision(
                action="escalate_to_human",
                reason="Metrics show concerning trends"
            )

        return CanaryDecision(action="continue")

    def check_promotion_criteria(self, metrics: dict) -> bool:
        """检查晋级条件"""
        criteria = {
            CanaryStage.CANARY: {
                "error_rate": lambda x: x < 0.01,
                "p95_latency_ms": lambda x: x < 5000,
                "safety_violation_rate": lambda x: x < 0.001,
            },
            CanaryStage.SMALL_TRAFFIC: {
                "task_completion_rate": lambda x: x >= 0.70,
                "hallucination_rate": lambda x: x < 0.03,
                "user_satisfaction_score": lambda x: x >= 3.5,
            },
            CanaryStage.MEDIUM_TRAFFIC: {
                "task_completion_rate": lambda x: x >= 0.72,
                "user_satisfaction_score": lambda x: x >= 3.8,
                "cost_per_request": lambda x: x < 1.5 * self.baseline_cost,
            },
        }

        stage_criteria = criteria.get(self.current_stage, {})
        for metric_name, check_fn in stage_criteria.items():
            value = metrics.get(metric_name)
            if value is None or not check_fn(value):
                return False
        return True
```

## 四、LLM专用在线评估方法

### 4.1 LLM-as-Judge在线评估

使用强模型对弱模型的输出进行在线质量评估：

```python
class OnlineLLMJudge:
    """在线LLM质量评估器"""

    JUDGE_PROMPT = """请评估以下AI助手回复的质量。

用户问题：{user_query}
AI回复：{ai_response}
参考上下文：{context}

请从以下维度评分（1-5分）：
1. 相关性：回复是否与问题相关？
2. 准确性：回复中的信息是否准确？
3. 完整性：是否完整回答了问题？
4. 清晰度：表述是否清晰易懂？
5. 安全性：是否包含有害或不当内容？

输出JSON格式：
{{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "clarity": <1-5>,
    "safety": <1-5>,
    "overall": <1-5>,
    "issues": ["<问题描述>"],
    "reasoning": "<评分理由>"
}}"""

    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
        self.client = AsyncOpenAI()

    async def evaluate(
        self,
        user_query: str,
        ai_response: str,
        context: str = ""
    ) -> QualityScore:
        """评估单个回复质量"""
        prompt = self.JUDGE_PROMPT.format(
            user_query=user_query,
            ai_response=ai_response,
            context=context
        )

        response = await self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,  # 低温度保证评估一致性
        )

        result = json.loads(response.choices[0].message.content)
        return QualityScore(**result)

    async def batch_evaluate(
        self,
        samples: list[dict],
        sample_rate: float = 0.1  # 抽样10%进行评估
    ) -> list[QualityScore]:
        """批量抽样评估"""
        import random
        sampled = random.sample(samples, int(len(samples) * sample_rate))

        # 并发评估，限制并发数避免API限流
        semaphore = asyncio.Semaphore(10)

        async def eval_with_semaphore(sample):
            async with semaphore:
                return await self.evaluate(
                    sample['query'],
                    sample['response'],
                    sample.get('context', '')
                )

        results = await asyncio.gather(
            *[eval_with_semaphore(s) for s in sampled]
        )
        return results
```

### 4.2 隐式反馈信号采集

在不需要用户主动评分的情况下，通过行为信号推断质量：

```python
class ImplicitFeedbackCollector:
    """隐式反馈信号采集器"""

    def collect_signals(
        self,
        conversation: Conversation
    ) -> ImplicitFeedback:
        signals = {}

        # 信号1：对话轮次效率
        # 高质量回复通常在更少轮次内解决问题
        signals['turns_to_resolution'] = conversation.turn_count
        signals['is_single_turn'] = conversation.turn_count == 1

        # 信号2：用户编辑行为
        # 如果用户复制了AI回复后进行了编辑，说明回复不够好
        signals['user_edited_copy'] = any(
            action.type == 'copy_then_edit'
            for action in conversation.user_actions
        )

        # 信号3：追问模式
        # 用户追问"你确定吗？"、"再说一次"等表示不信任
        clarification_patterns = [
            r'你确定', r'really', r'are you sure',
            r'再说一次', r'can you rephrase',
            r'不对', r'that\'s wrong', r'incorrect'
        ]
        signals['clarification_count'] = sum(
            1 for msg in conversation.messages
            if msg.role == 'user' and
            any(re.search(p, msg.content, re.I) for p in clarification_patterns)
        )

        # 信号4：会话放弃率
        # 用户在得到回复后立即离开，可能表示不满意
        signals['immediate_abandon'] = (
            conversation.last_user_action == 'leave' and
            conversation.time_since_last_ai_response < 5  # 秒
        )

        # 信号5：任务完成信号
        # 用户发送了感谢、确认等结束信号
        completion_patterns = [
            r'谢谢', r'thanks', r'解决了', r'搞定了',
            r'perfect', r'got it', r'ok'
        ]
        signals['completion_signal'] = any(
            re.search(p, msg.content, re.I)
            for msg in conversation.messages
            if msg.role == 'user'
            for p in completion_patterns
        )

        return ImplicitFeedback(
            conversation_id=conversation.id,
            signals=signals,
            composite_score=self.compute_composite_score(signals)
        )

    def compute_composite_score(self, signals: dict) -> float:
        """计算隐式反馈综合得分"""
        weights = {
            'is_single_turn': 0.25,
            'completion_signal': 0.25,
            'immediate_abandon': -0.30,
            'user_edited_copy': -0.20,
            'clarification_count': -0.15,
        }

        score = 0.5  # 基线分
        for signal, weight in weights.items():
            value = signals.get(signal, 0)
            if isinstance(value, bool):
                score += weight * (1 if value else 0)
            elif isinstance(value, (int, float)):
                score += weight * min(value, 1)  # 归一化

        return max(0, min(1, score))  # 限制在[0, 1]
```

## 五、统计显著性分析

### 5.1 适用LLM场景的统计检验方法

```python
import numpy as np
from scipy import stats

class LLMExperimentAnalyzer:
    """LLM实验结果统计分析器"""

    def analyze_binary_metric(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int
    ) -> ExperimentResult:
        """分析二值指标（如任务完成率）"""

        p_control = control_conversions / control_total
        p_treatment = treatment_conversions / treatment_total

        # 双比例Z检验
        p_pooled = (
            (control_conversions + treatment_conversions) /
            (control_total + treatment_total)
        )
        se = math.sqrt(
            p_pooled * (1 - p_pooled) *
            (1/control_total + 1/treatment_total)
        )

        z_stat = (p_treatment - p_control) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # 效应量（Cohen's h）
        effect_size = 2 * (
            math.asin(math.sqrt(p_treatment)) -
            math.asin(math.sqrt(p_control))
        )

        # 置信区间
        ci_95 = (
            (p_treatment - p_control) - 1.96 * se,
            (p_treatment - p_control) + 1.96 * se
        )

        return ExperimentResult(
            metric="binary",
            control_rate=p_control,
            treatment_rate=p_treatment,
            relative_lift=(p_treatment - p_control) / p_control,
            z_statistic=z_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval_95=ci_95,
            is_significant=p_value < 0.05,
            sample_size_control=control_total,
            sample_size_treatment=treatment_total,
        )

    def analyze_continuous_metric(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray
    ) -> ExperimentResult:
        """分析连续指标（如评分、延迟）"""

        # Welch's t检验（不假设方差齐性）
        t_stat, p_value = stats.ttest_ind(
            treatment_values, control_values,
            equal_var=False
        )

        # 效应量（Cohen's d）
        pooled_std = math.sqrt(
            (control_values.std()**2 + treatment_values.std()**2) / 2
        )
        effect_size = (
            treatment_values.mean() - control_values.mean()
        ) / pooled_std

        # Bootstrap置信区间
        ci_95 = self.bootstrap_ci(
            control_values, treatment_values, n_bootstrap=10000
        )

        return ExperimentResult(
            metric="continuous",
            control_rate=control_values.mean(),
            treatment_rate=treatment_values.mean(),
            relative_lift=(
                (treatment_values.mean() - control_values.mean()) /
                control_values.mean()
            ),
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval_95=ci_95,
            is_significant=p_value < 0.05,
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values),
        )

    def analyze_with_multiple_testing_correction(
        self,
        results: list[ExperimentResult]
    ) -> list[ExperimentResult]:
        """多重检验校正（Bonferroni）"""
        alpha_corrected = 0.05 / len(results)

        for result in results:
            result.adjusted_p_value = min(
                result.p_value * len(results),
                1.0
            )
            result.is_significant_corrected = (
                result.adjusted_p_value < 0.05
            )

        return results

    def bootstrap_ci(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        n_bootstrap: int = 10000,
        ci_level: float = 0.95
    ) -> tuple[float, float]:
        """Bootstrap置信区间估计"""
        diffs = []
        for _ in range(n_bootstrap):
            ctrl_sample = np.random.choice(control, size=len(control), replace=True)
            treat_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            diffs.append(treat_sample.mean() - ctrl_sample.mean())

        alpha = (1 - ci_level) / 2
        lower = np.percentile(diffs, alpha * 100)
        upper = np.percentile(diffs, (1 - alpha) * 100)
        return (lower, upper)
```

### 5.2 贝叶斯方法：更适合LLM的评估范式

频率派方法需要固定样本量，而贝叶斯方法可以随时做出决策：

```python
class BayesianABTest:
    """贝叶斯AB测试"""

    def beta_binomial_test(
        self,
        control_successes: int,
        control_trials: int,
        treatment_successes: int,
        treatment_trials: int,
        prior_alpha: float = 1,
        prior_beta: float = 1
    ) -> BayesianResult:
        """
        Beta-Binomial模型

        优势：
        - 可以随时查看结果，不需要等到预设样本量
        - 直接给出"Treatment更好的概率"
        - 可以计算"期望损失"（选错方案的代价）
        """

        # 后验分布
        alpha_ctrl = prior_alpha + control_successes
        beta_ctrl = prior_beta + (control_trials - control_successes)
        alpha_treat = prior_alpha + treatment_successes
        beta_treat = prior_beta + (treatment_trials - treatment_successes)

        # 蒙特卡洛采样
        n_samples = 100_000
        ctrl_samples = np.random.beta(alpha_ctrl, beta_ctrl, n_samples)
        treat_samples = np.random.beta(alpha_treat, beta_treat, n_samples)

        # P(Treatment > Control)
        prob_treatment_better = (treat_samples > ctrl_samples).mean()

        # 期望损失
        loss_if_choose_control = np.maximum(treat_samples - ctrl_samples, 0).mean()
        loss_if_choose_treatment = np.maximum(ctrl_samples - treat_samples, 0).mean()

        # 最优方案
        if prob_treatment_better > 0.95:
            recommendation = "adopt_treatment"
        elif prob_treatment_better < 0.05:
            recommendation = "keep_control"
        else:
            recommendation = "need_more_data"

        return BayesianResult(
            prob_treatment_better=prob_treatment_better,
            expected_loss_control=loss_if_choose_control,
            expected_loss_treatment=loss_if_choose_treatment,
            recommendation=recommendation,
            posterior_control={"alpha": alpha_ctrl, "beta": beta_ctrl},
            posterior_treatment={"alpha": alpha_treat, "beta": beta_treat},
        )
```

**贝叶斯方法的核心优势：**

| 特性 | 频率派方法 | 贝叶斯方法 |
|------|-----------|-----------|
| 何时能出结果 | 必须等样本量达标 | 随时可以 |
| 结果解读 | p值（反直觉） | P(T更好)（直觉） |
| 样本量不均等处理 | 复杂 | 天然支持 |
| 多变量联合分析 | 需要多重校正 | 自然融合 |
| 先验知识利用 | 不支持 | 可以引入先验 |

## 六、生产级实验平台架构

### 6.1 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                    Experiment Dashboard                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ 实验配置  │  │ 实时监控  │  │ 结果分析  │              │
│  │   UI     │  │   仪表盘  │  │   报告    │              │
│  └──────────┘  └──────────┘  └──────────┘              │
├──────────────────────────────────────────────────────────┤
│                    Experiment Service                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 实验管理器    │  │ 分流路由      │  │ 指标计算      │  │
│  │ (CRUD/状态机) │  │ (确定性哈希)  │  │ (实时聚合)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├──────────────────────────────────────────────────────────┤
│                    Data Pipeline                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 事件采集      │→│ 流式处理      │→│ 指标存储      │  │
│  │ (Kafka)       │  │ (Flink)      │  │ (ClickHouse)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├──────────────────────────────────────────────────────────┤
│                    Agent Execution Layer                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │  LLM Gateway (携带 experiment_id, variant_id)     │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 6.2 实验配置与管理

```python
from pydantic import BaseModel
from enum import Enum

class ExperimentStatus(Enum):
    DRAFT = "draft"
    CANARY = "canary"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"

class ExperimentConfig(BaseModel):
    """实验配置"""
    experiment_id: str
    name: str
    description: str
    owner: str

    # 变体定义
    variants: list[VariantConfig]
    traffic_split: dict[str, float]  # {"control": 50, "treatment_a": 30, "treatment_b": 20}

    # 指标定义
    primary_metric: str
    secondary_metrics: list[str]
    guardrail_metrics: dict[str, GuardrailConfig]

    # 分流策略
    traffic_allocation_key: str  # "user_id" | "session_id" | "request_id"
    holdout_group: str | None  # 永远不参与实验的对照组

    # 时间配置
    start_time: datetime | None
    end_time: datetime | None
    min_duration_hours: int = 24
    max_duration_hours: int = 720  # 30天

    # 安全配置
    auto_rollback_on_guardrail_breach: bool = True
    max_rollback_count: int = 3

class VariantConfig(BaseModel):
    """变体配置"""
    variant_id: str
    name: str
    description: str

    # 变体参数（按实验类型不同）
    # Prompt实验
    system_prompt: str | None = None
    # 模型实验
    model_id: str | None = None
    # RAG实验
    rag_config: dict | None = None
    # 参数实验
    temperature: float | None = None
    max_tokens: int | None = None
```

## 七、实战案例：RAG Reranker实验

### 7.1 实验背景

我们有一个RAG客服系统，当前使用BM25检索Top-10文档后直接拼接给LLM。现在要测试添加Cross-Encoder Reranker（取Top-3）是否能提升回答质量。

### 7.2 实验配置

```yaml
experiment:
  id: "rag_reranker_v1"
  name: "RAG Reranker效果评估"
  description: "测试Cross-Encoder Reranker对回答质量的影响"

  variants:
    - id: "control"
      name: "BM25 Top-10"
      description: "现有方案：BM25检索10篇文档直接拼接"
      rag_config:
        retrieval: "bm25"
        top_k: 10
        reranker: null

    - id: "treatment"
      name: "BM25 + CrossEncoder Top-3"
      description: "新方案：BM25检索后CrossEncoder重排取Top-3"
      rag_config:
        retrieval: "bm25"
        top_k: 10
        reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        final_top_k: 3

  traffic_split:
    control: 50
    treatment: 50

  primary_metric: "task_completion_rate"
  secondary_metrics:
    - "user_satisfaction_score"
    - "first_turn_resolution"
  guardrail_metrics:
    hallucination_rate:
      threshold: 0.03
      action: "auto_rollback"
    p95_latency_ms:
      threshold: 8000
      action: "alert"
    cost_per_request:
      threshold: 1.5  # 倍数，相对于control
      action: "alert"

  traffic_allocation_key: "session_id"
```

### 7.3 结果分析报告

经过7天的实验，收集到以下数据：

```
═══════════════════════════════════════════════════════
  实验报告: RAG Reranker效果评估
  实验ID: rag_reranker_v1
  实验时间: 2026-05-20 ~ 2026-05-27
  总样本数: 12,847 (Control: 6,423 | Treatment: 6,424)
═══════════════════════════════════════════════════════

  主要指标:
  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ 指标                │ Control  │ Treatment│ Lift     │ p-value  │
  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ 任务完成率          │ 71.2%    │ 76.8%    │ +7.9%    │ 0.0012   │
  │                     │          │          │          │ ✅显著    │
  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘

  次要指标:
  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ 指标                │ Control  │ Treatment│ Lift     │ p-value  │
  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ 用户满意度          │ 3.62     │ 3.91     │ +8.0%    │ 0.0089   │
  │                     │          │          │          │ ✅显著    │
  │ 首轮解决率          │ 45.3%    │ 52.1%    │ +15.0%   │ 0.0003   │
  │                     │          │          │          │ ✅显著    │
  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘

  Guardrail指标:
  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ 指标                │ Control  │ Treatment│ 阈值     │ 状态     │
  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ 幻觉率              │ 2.8%     │ 1.9%     │ <3%      │ ✅通过    │
  │ P95延迟             │ 3,241ms  │ 3,892ms  │ <8,000ms │ ✅通过    │
  │ 单次成本            │ $0.0042  │ $0.0028  │ <1.5x    │ ✅通过    │
  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘

  结论: 建议全量发布 Treatment 方案
  理由: 主要指标和次要指标均达到统计显著性，所有Guardrail指标通过。
        由于Top-3减少了LLM输入Token，成本反而降低了33%。
═══════════════════════════════════════════════════════
```

## 八、总结与最佳实践

### 8.1 核心原则

1. **先小后大**：永远从金丝雀发布开始，逐步扩大流量
2. **Guardrail优先**：安全和质量底线不可妥协，任何违规立即停止
3. **多维度评估**：单一指标容易误导，必须看指标组合
4. **贝叶斯优先**：在LLM场景下，贝叶斯方法比频率派更实用
5. **隐式反馈**：不依赖用户主动评分，通过行为信号推断质量
6. **成本纳入**：效果提升但成本翻倍的方案，需要权衡ROI

### 8.2 常见陷阱

| 陷阱 | 描述 | 应对 |
|------|------|------|
| **过早停止** | 看到早期数据就下结论 | 必须达到最小样本量 |
| **偷看问题** | 频繁检查p值增加假阳性 | 使用序贯检验或贝叶斯方法 |
| **辛普森悖论** | 整体指标上升但子群下降 | 分层分析关键维度 |
| **新鲜感偏差** | 用户因新奇而暂时满意 | 实验至少持续1周 |
| **选择偏差** | 分流不均匀导致基线不同 | 检查SRM（Sample Ratio Mismatch） |

### 8.3 推荐工具栈

| 层级 | 推荐方案 |
|------|---------|
| 实验管理 | GrowthBook（开源）/ Statsig（SaaS） |
| 分流路由 | 自研（基于一致性哈希） |
| 实时指标 | ClickHouse + Grafana |
| 统计分析 | scipy / PyMC（贝叶斯） |
| LLM评估 | LLM-as-Judge + 隐式反馈 |
| 灰度发布 | 自研状态机 / Argo Rollouts |

---

> **延伸阅读**
> - Trustworthy Online Controlled Experiments (Ron Kohavi et al.)
> - Bayesian Methods for Hackers (Cam Davidson-Pilon)
> - Google: Overlapping Experiment Infrastructure
> - Microsoft: ExP Platform — Online Experimentation at Scale
