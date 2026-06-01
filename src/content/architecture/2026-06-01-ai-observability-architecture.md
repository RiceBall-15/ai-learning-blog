---
title: "AI系统可观测性架构：从日志收集到智能诊断的生产级实践"
description: "深入剖析AI系统可观测性架构设计，涵盖LLM请求追踪、模型行为监控、异常检测与自动根因分析，附完整架构图与Prometheus+Grafana实战配置"
date: 2026-06-01
author: "RiceBall"
category: "architecture"
subCategory: "cloud-native"
tags: ["AI架构", "可观测性", "监控", "LLM", "MLOps", "Prometheus", "Grafana", "分布式追踪"]
draft: false
---

# AI系统可观测性架构：从日志收集到智能诊断的生产级实践

## 一个凌晨3点的故障故事

凌晨3点，你被电话吵醒。客服系统反馈：AI助手回答质量严重下降，用户投诉激增。

你打开监控面板：
- API响应时间：正常（200ms）
- GPU利用率：正常（65%）
- 内存使用：正常（70%）
- 错误率：0%

**一切指标都正常，但用户体验明显恶化。**

这就是AI系统可观测性的核心挑战：**传统监控指标无法反映模型行为的质量变化**。

```
传统监控 vs AI系统可观测性:
─────────────────────────────────────────────

传统系统监控:
  ✅ 响应时间、吞吐量、错误率、资源利用率
  ✅ 这些指标告诉你"系统在运行"
  ❌ 无法告诉你"系统在正确地运行"

AI系统需要的额外维度:
  📊 模型输出质量（相关性、准确性、安全性）
  📊 推理过程（token使用、推理时间、注意力分布）
  📊 数据漂移（输入分布变化、概念漂移）
  📊 用户反馈（满意度、修正率、重新提问率）

凌晨3点的故事:
  根因: 模型微调后，输出长度从平均200token增加到800token
       → KV Cache内存翻倍 → 推理引擎自动降级到更小的batch size
       → 吞吐量下降 → 用户排队 → 部分请求超时被截断
       → 截断的输出质量下降 → 用户投诉
  
  传统监控看不到什么:
    • 每个请求的输出质量
    • 模型输出分布的变化
    • 推理引擎的动态调整
```

---

## 一、AI系统可观测性的三大支柱

### 1.1 架构全景图

```
AI系统可观测性架构:
─────────────────────────────────────────────

┌─────────────────────────────────────────────────────────┐
│                    数据采集层 (Collection)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ 应用日志 │  │ 模型指标 │  │ 追踪数据 │  │ 用户   │ │
│  │ (Logs)   │  │(Metrics) │  │(Traces)  │  │反馈    │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │             │             │             │       │
└───────┼─────────────┼─────────────┼─────────────┼───────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                    数据处理层 (Processing)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ 日志聚合 │  │ 指标存储 │  │ 追踪关联 │  │ 反馈   │ │
│  │ (Loki)   │  │(Prometheus│  │(Jaeger)  │  │聚合    │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │             │             │             │       │
└───────┼─────────────┼─────────────┼─────────────┼───────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                    智能分析层 (Analytics)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ 可视化   │  │ 异常检测 │  │ 根因分析 │  │ 预警   │ │
│  │(Grafana) │  │(ML模型)  │  │(图推理)  │  │系统    │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 四大数据类型

```
AI系统可观测性数据:
─────────────────────────────────────────────

1. 应用日志 (Logs)
   • 推理请求日志: 请求ID、模型版本、输入输出、延迟
   • 系统日志: 错误、警告、配置变更
   • 审计日志: 安全事件、合规记录
   
2. 模型指标 (Metrics)
   • 推理性能: 延迟、吞吐量、GPU利用率
   • 模型质量: 准确率、相关性、安全性评分
   • 资源指标: 内存、CPU、网络、存储
   
3. 追踪数据 (Traces)
   • 请求链路: 从用户请求到响应的完整路径
   • 模型调用: 每次模型推理的详细信息
   • 依赖关系: 服务间调用、缓存命中、数据库查询
   
4. 用户反馈 (Feedback)
   • 显式反馈: 点赞/点踩、评分、评论
   • 隐式反馈: 重新提问、编辑、放弃
   • 行为数据: 停留时间、复制粘贴、分享
```

---

## 二、LLM请求追踪：从入口到输出的全链路可见

### 2.1 追踪架构设计

```
LLM请求追踪架构:
─────────────────────────────────────────────

用户请求
    │
    ▼
┌─────────────┐
│ API Gateway │ ← Trace ID 生成
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────────────────────────────┐
│   Router    │────→│         追踪数据存储                 │
└──────┬──────┘     │  ┌───────────┐  ┌───────────────┐  │
       │            │  │ Trace DB  │  │ Span Store    │  │
       ▼            │  │ (ClickHouse) │ (Elasticsearch)│  │
┌─────────────┐     │  └───────────┘  └───────────────┘  │
│ Load        │     └─────────────────────────────────────┘
│ Balancer    │              ▲
└──────┬──────┘              │
       │                     │
       ▼                     │
┌─────────────┐              │
│  LLM Worker │──────────────┘
│             │ ← 记录每个Span:
│  ┌────────┐ │   • Prefill时间
│  │ Decode │ │   • Decode时间
│  │ Engine │ │   • Token使用量
│  └────────┘ │   • KV Cache状态
└──────┬──────┘   • 采样概率
       │
       ▼
┌─────────────┐
│  Response   │
│  Aggregator │ ← 汇总所有Span，计算端到端指标
└─────────────┘
```

### 2.2 追踪数据模型

```python
# LLM追踪数据模型
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

@dataclass
class LLMSpan:
    """LLM推理追踪的单个Span"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    
    # 基本信息
    operation: str = "llm.inference"  # 操作类型
    model_name: str = ""              # 模型名称
    model_version: str = ""           # 模型版本
    
    # 时间信息
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Token信息
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # 性能信息
    prefill_time_ms: float = 0.0    # Prefill阶段时间
    decode_time_ms: float = 0.0     # Decode阶段时间
    tokens_per_second: float = 0.0  # 生成速度
    
    # 资源信息
    gpu_memory_used_mb: float = 0.0
    gpu_utilization: float = 0.0
    
    # 质量信息
    quality_score: Optional[float] = None  # 模型输出质量评分
    safety_score: Optional[float] = None   # 安全性评分
    
    # 上下文信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "prefill_time_ms": self.prefill_time_ms,
            "decode_time_ms": self.decode_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_utilization": self.gpu_utilization,
            "quality_score": self.quality_score,
            "safety_score": self.safety_score,
            "metadata": self.metadata,
            "tags": self.tags,
        }


@dataclass
class LLMTrace:
    """LLM请求的完整追踪"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: List[LLMSpan] = field(default_factory=list)
    
    # 请求信息
    request_id: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # 端到端指标
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    
    # 状态
    status: str = "success"  # success, error, timeout
    error_message: Optional[str] = None
    
    def add_span(self, span: LLMSpan):
        """添加Span并更新端到端指标"""
        self.spans.append(span)
        self.total_duration_ms = sum(s.duration_ms for s in self.spans)
        self.total_tokens = sum(s.total_tokens for s in self.spans)
    
    def get_critical_path(self) -> List[LLMSpan]:
        """获取关键路径（耗时最长的Span链）"""
        if not self.spans:
            return []
        
        # 按开始时间排序，构建调用树
        sorted_spans = sorted(self.spans, key=lambda s: s.start_time)
        
        # 找出关键路径（简化实现）
        critical_path = [sorted_spans[0]]
        current_end = sorted_spans[0].end_time
        
        for span in sorted_spans[1:]:
            if span.start_time <= current_end:
                # 重叠的Span，选择耗时更长的
                if span.duration_ms > critical_path[-1].duration_ms:
                    critical_path[-1] = span
            else:
                critical_path.append(span)
                current_end = span.end_time
        
        return critical_path
```

### 2.3 追踪实现示例

```python
# LLM追踪中间件
import time
from contextlib import contextmanager
from typing import Generator

class LLMTracer:
    """LLM请求追踪器"""
    
    def __init__(self, collector_endpoint: str):
        self.collector_endpoint = collector_endpoint
        self.spans: List[LLMSpan] = []
        
    @contextmanager
    def trace_inference(
        self,
        model_name: str,
        model_version: str,
        input_text: str,
        **kwargs
    ) -> Generator[LLMSpan, None, None]:
        """追踪一次LLM推理"""
        span = LLMSpan(
            model_name=model_name,
            model_version=model_version,
            operation="llm.inference",
        )
        
        start_time = time.time()
        
        try:
            yield span
            
        except Exception as e:
            span.tags.append("error")
            span.metadata["error"] = str(e)
            raise
            
        finally:
            span.end_time = datetime.now()
            span.duration_ms = (time.time() - start_time) * 1000
            
            # 计算tokens_per_second
            if span.decode_time_ms > 0:
                span.tokens_per_second = (
                    span.output_tokens / (span.decode_time_ms / 1000)
                )
            
            # 发送到收集器
            self._send_span(span)
    
    def _send_span(self, span: LLMSpan):
        """发送Span到收集器（异步）"""
        # 实际实现中应该异步发送，避免影响推理性能
        import requests
        try:
            requests.post(
                f"{self.collector_endpoint}/spans",
                json=span.to_dict(),
                timeout=0.1  # 非阻塞
            )
        except:
            pass  # 收集失败不应影响主流程


# 使用示例
tracer = LLMTracer("http://collector:8080")

with tracer.trace_inference(
    model_name="llama-3-70b",
    model_version="v1.2",
    input_text="请解释量子计算"
) as span:
    
    # 执行推理
    output = model.generate(input_text)
    
    # 记录指标
    span.input_tokens = tokenizer.encode(input_text).length
    span.output_tokens = tokenizer.encode(output).length
    span.total_tokens = span.input_tokens + span.output_tokens
    
    # 记录GPU信息
    span.gpu_memory_used_mb = get_gpu_memory_used()
    span.gpu_utilization = get_gpu_utilization()
    
    # 计算质量分数
    span.quality_score = evaluate_output_quality(input_text, output)
```

---

## 三、模型行为监控：超越传统指标

### 3.1 模型质量指标体系

```
模型质量监控指标:
─────────────────────────────────────────────

一级指标 (核心质量):
├── 输出相关性 (Relevance)
│   ├── 语义相似度 (BERTScore)
│   ├── 关键词覆盖率
│   └── 用户满意度评分
│
├── 输出准确性 (Accuracy)
│   ├── 事实性检查 (FactScore)
│   ├── 幻觉检测率
│   └── 逻辑一致性
│
├── 输出安全性 (Safety)
│   ├── 有害内容检测率
│   ├── 偏见评分
│   └── 隐私泄露检测
│
└── 输出效率 (Efficiency)
    ├── 响应时间
    ├── Token使用效率
    └── 重复率

二级指标 (运维质量):
├── 可用性
│   ├── 成功率
│   ├── 超时率
│   └── 降级率
│
├── 性能
│   ├── P50/P95/P99延迟
│   ├── 吞吐量
│   └── 并发处理能力
│
└── 资源
    ├── GPU利用率
    ├── 内存使用率
    └── 成本效率 ($/1K tokens)
```

### 3.2 实时质量评估管道

```python
# 模型输出质量评估管道
from dataclasses import dataclass
from typing import List, Callable
import numpy as np

@dataclass
class QualityAssessment:
    """质量评估结果"""
    request_id: str
    timestamp: datetime
    
    # 质量分数 (0-1)
    relevance_score: float = 0.0
    accuracy_score: float = 0.0
    safety_score: float = 0.0
    efficiency_score: float = 0.0
    
    # 综合分数
    overall_score: float = 0.0
    
    # 详细信息
    details: dict = None
    
    # 是否触发告警
    alerts: List[str] = None


class QualityAssessmentPipeline:
    """模型输出质量评估管道"""
    
    def __init__(self):
        self.evaluators: List[Callable] = []
        self.thresholds = {
            "relevance": 0.6,
            "accuracy": 0.7,
            "safety": 0.9,
            "overall": 0.65,
        }
    
    def add_evaluator(self, evaluator: Callable):
        """添加评估器"""
        self.evaluators.append(evaluator)
    
    def assess(
        self,
        request_id: str,
        input_text: str,
        output_text: str,
        context: dict = None
    ) -> QualityAssessment:
        """执行质量评估"""
        assessment = QualityAssessment(
            request_id=request_id,
            timestamp=datetime.now()
        )
        
        # 运行所有评估器
        for evaluator in self.evaluators:
            result = evaluator(input_text, output_text, context)
            assessment.details = {**assessment.details, **result}
        
        # 计算综合分数
        assessment.relevance_score = assessment.details.get("relevance", 0)
        assessment.accuracy_score = assessment.details.get("accuracy", 0)
        assessment.safety_score = assessment.details.get("safety", 1)
        assessment.efficiency_score = assessment.details.get("efficiency", 0)
        
        # 加权综合分数
        assessment.overall_score = (
            0.3 * assessment.relevance_score +
            0.3 * assessment.accuracy_score +
            0.3 * assessment.safety_score +
            0.1 * assessment.efficiency_score
        )
        
        # 检查是否触发告警
        assessment.alerts = self._check_alerts(assessment)
        
        return assessment
    
    def _check_alerts(self, assessment: QualityAssessment) -> List[str]:
        """检查是否需要触发告警"""
        alerts = []
        
        if assessment.safety_score < self.thresholds["safety"]:
            alerts.append("SAFETY_VIOLATION")
        
        if assessment.overall_score < self.thresholds["overall"]:
            alerts.append("QUALITY_DEGRADATION")
        
        if assessment.relevance_score < self.thresholds["relevance"]:
            alerts.append("RELEVANCE_LOW")
        
        return alerts


# 评估器实现示例
def relevance_evaluator(input_text: str, output_text: str, context: dict) -> dict:
    """相关性评估器"""
    # 使用BERTScore计算语义相似度
    from bert_score import score as bert_score
    
    P, R, F1 = bert_score([output_text], [input_text], lang="zh")
    
    return {
        "relevance": F1.item(),
        "relevance_precision": P.item(),
        "relevance_recall": R.item(),
    }


def safety_evaluator(input_text: str, output_text: str, context: dict) -> dict:
    """安全性评估器"""
    from safety_checker import SafetyChecker
    
    checker = SafetyChecker()
    result = checker.check(output_text)
    
    return {
        "safety": 1.0 if result.is_safe else 0.0,
        "safety_categories": result.flagged_categories,
    }


def accuracy_evaluator(input_text: str, output_text: str, context: dict) -> dict:
    """准确性评估器（简化版）"""
    # 实际实现中应该使用更复杂的方法
    # 如FactScore、RAGAS等
    
    # 简单检查：是否包含明显的矛盾或幻觉
    has_contradiction = check_contradictions(output_text)
    has_hallucination = check_hallucinations(output_text, context)
    
    score = 1.0
    if has_contradiction:
        score -= 0.3
    if has_hallucination:
        score -= 0.4
    
    return {
        "accuracy": max(0, score),
        "has_contradiction": has_contradiction,
        "has_hallucination": has_hallucination,
    }
```

### 3.3 数据漂移检测

```
数据漂移检测架构:
─────────────────────────────────────────────

实时数据流
    │
    ▼
┌─────────────────┐
│ 数据预处理      │
│ • Token分布     │
│ • 序列长度      │
│ • 语言特征      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────────┐
│ 漂移检测器      │────→│ 基线数据库          │
│                 │     │ • 历史分布统计       │
│ • KL散度        │     │ • 基线特征向量       │
│ • PSI指标       │     │ • 概率分布参数       │
│ • KS检验        │     └─────────────────────┘
│ • 特征重要度    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 漂移分类        │
│                 │
│ • 无漂移        │→ 正常监控
│ • 轻微漂移      │→ 增加监控频率
│ • 显著漂移      │→ 触发告警 + 模型重训练评估
│ • 严重漂移      │→ 紧急切换模型 + 人工介入
└─────────────────┘
```

```python
# 数据漂移检测器
import numpy as np
from scipy import stats
from typing import Dict, Tuple

class DataDriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, baseline_window_days: int = 30):
        self.baseline_window_days = baseline_window_days
        self.baseline_stats: Dict[str, dict] = {}
        
    def update_baseline(self, feature_name: str, values: np.ndarray):
        """更新基线统计"""
        self.baseline_stats[feature_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "histogram": np.histogram(values, bins=50),
            "quantiles": np.percentile(values, [25, 50, 75]),
        }
    
    def detect_drift(
        self,
        feature_name: str,
        current_values: np.ndarray,
        method: str = "psi"
    ) -> Tuple[bool, float, str]:
        """
        检测数据漂移
        
        Returns:
            (is_drifted, drift_score, drift_level)
        """
        if feature_name not in self.baseline_stats:
            raise ValueError(f"Baseline not found for {feature_name}")
        
        baseline = self.baseline_stats[feature_name]
        
        if method == "psi":
            # Population Stability Index
            drift_score = self._calculate_psi(
                baseline["histogram"],
                np.histogram(current_values, bins=50)
            )
            
        elif method == "kl":
            # KL散度
            drift_score = self._calculate_kl_divergence(
                baseline["histogram"],
                np.histogram(current_values, bins=50)
            )
            
        elif method == "ks":
            # Kolmogorov-Smirnov检验
            drift_score = self._calculate_ks_test(
                baseline["values"],
                current_values
            )
            
        elif method == "wasserstein":
            # Wasserstein距离
            drift_score = stats.wasserstein_distance(
                baseline["values"],
                current_values
            )
        
        # 判断漂移程度
        is_drifted, drift_level = self._classify_drift(drift_score, method)
        
        return is_drifted, drift_score, drift_level
    
    def _calculate_psi(
        self,
        baseline_hist: Tuple[np.ndarray, np.ndarray],
        current_hist: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """计算PSI指标"""
        baseline_counts, _ = baseline_hist
        current_counts, _ = current_hist
        
        # 归一化为概率分布
        baseline_probs = baseline_counts / baseline_counts.sum() + 1e-10
        current_probs = current_counts / current_counts.sum() + 1e-10
        
        # PSI = Σ (P_current - P_baseline) * ln(P_current / P_baseline)
        psi = np.sum((current_probs - baseline_probs) * np.log(current_probs / baseline_probs))
        
        return psi
    
    def _classify_drift(self, drift_score: float, method: str) -> Tuple[bool, str]:
        """根据漂移分数分类"""
        thresholds = {
            "psi": {"warning": 0.1, "critical": 0.25},
            "kl": {"warning": 0.1, "critical": 0.5},
            "ks": {"warning": 0.05, "critical": 0.1},
            "wasserstein": {"warning": 0.1, "critical": 0.3},
        }
        
        threshold = thresholds.get(method, {"warning": 0.1, "critical": 0.3})
        
        if drift_score < threshold["warning"]:
            return False, "normal"
        elif drift_score < threshold["critical"]:
            return True, "warning"
        else:
            return True, "critical"
```

---

## 四、异常检测与智能预警

### 4.1 多维度异常检测

```
AI系统异常检测维度:
─────────────────────────────────────────────

1. 指标异常
   • 单指标异常: 某个指标超出正常范围
   • 多指标异常: 多个指标同时异常（可能是系统性问题）
   • 趋势异常: 指标变化趋势异常（缓慢退化）

2. 模式异常
   • 请求模式异常: 请求量、请求类型分布异常
   • 错误模式异常: 错误类型、错误率异常
   • 延迟模式异常: 延迟分布、长尾延迟异常

3. 行为异常
   • 模型输出异常: 输出质量突然下降
   • 用户行为异常: 用户反馈模式突然改变
   • 系统行为异常: 资源使用模式突然改变

4. 关联异常
   • 服务间关联: 一个服务异常导致其他服务异常
   • 指标间关联: 某些指标应该相关但出现背离
   • 时间关联: 某个时间点后多个指标同时异常
```

### 4.2 智能预警系统

```python
# 智能预警系统
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """预警信息"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    
    # 关联信息
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    
    # 上下文
    affected_services: List[str] = None
    suggested_actions: List[str] = None
    
    # 生命周期
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class IntelligentAlertSystem:
    """智能预警系统"""
    
    def __init__(self):
        self.alert_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        # 预警规则
        self.rules = [
            self._rule_quality_degradation,
            self._rule_latency_spike,
            self._rule_error_rate_increase,
            self._rule_resource_exhaustion,
            self._rule_data_drift,
        ]
        
        # 预警聚合
        self.aggregation_window = timedelta(minutes=5)
        self.correlation_threshold = 0.8
    
    async def evaluate(
        self,
        metrics: Dict[str, float],
        context: dict
    ) -> List[Alert]:
        """评估所有预警规则"""
        new_alerts = []
        
        for rule in self.rules:
            alerts = await rule(metrics, context)
            new_alerts.extend(alerts)
        
        # 聚合相关预警
        aggregated_alerts = self._aggregate_alerts(new_alerts)
        
        # 更新活跃预警
        for alert in aggregated_alerts:
            self.active_alerts[alert.alert_id] = alert
        
        return aggregated_alerts
    
    async def _rule_quality_degradation(
        self,
        metrics: Dict[str, float],
        context: dict
    ) -> List[Alert]:
        """质量退化预警"""
        alerts = []
        
        quality_score = metrics.get("quality_score", 1.0)
        baseline_quality = context.get("baseline_quality", 0.85)
        
        if quality_score < baseline_quality * 0.8:
            alerts.append(Alert(
                alert_id=f"quality-{datetime.now().timestamp()}",
                severity=AlertSeverity.CRITICAL,
                title="模型质量严重退化",
                description=f"当前质量分数: {quality_score:.2f}, "
                           f"基线: {baseline_quality:.2f}",
                timestamp=datetime.now(),
                metric_name="quality_score",
                current_value=quality_score,
                threshold_value=baseline_quality * 0.8,
                suggested_actions=[
                    "检查最近的模型更新",
                    "分析输入数据分布变化",
                    "准备回滚到上一个稳定版本",
                ]
            ))
        
        return alerts
    
    async def _rule_latency_spike(
        self,
        metrics: Dict[str, float],
        context: dict
    ) -> List[Alert]:
        """延迟突增预警"""
        alerts = []
        
        p95_latency = metrics.get("p95_latency_ms", 0)
        baseline_latency = context.get("baseline_p95_latency_ms", 1000)
        
        # 延迟增加超过2倍
        if p95_latency > baseline_latency * 2:
            alerts.append(Alert(
                alert_id=f"latency-{datetime.now().timestamp()}",
                severity=AlertSeverity.WARNING,
                title="P95延迟异常增高",
                description=f"当前P95延迟: {p95_latency:.0f}ms, "
                           f"基线: {baseline_latency:.0f}ms",
                timestamp=datetime.now(),
                metric_name="p95_latency_ms",
                current_value=p95_latency,
                threshold_value=baseline_latency * 2,
                suggested_actions=[
                    "检查GPU利用率",
                    "检查是否有大批量请求",
                    "检查模型版本是否变更",
                ]
            ))
        
        return alerts
    
    async def _rule_error_rate_increase(
        self,
        metrics: Dict[str, float],
        context: dict
    ) -> List[Alert]:
        """错误率增加预警"""
        alerts = []
        
        error_rate = metrics.get("error_rate", 0.0)
        
        if error_rate > 0.05:  # 5%错误率
            severity = AlertSeverity.CRITICAL if error_rate > 0.1 else AlertSeverity.WARNING
            
            alerts.append(Alert(
                alert_id=f"error-{datetime.now().timestamp()}",
                severity=severity,
                title="错误率超出阈值",
                description=f"当前错误率: {error_rate:.2%}",
                timestamp=datetime.now(),
                metric_name="error_rate",
                current_value=error_rate,
                threshold_value=0.05,
            ))
        
        return alerts
    
    def _aggregate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """聚合相关预警"""
        if not alerts:
            return []
        
        # 按时间窗口分组
        time_groups: Dict[datetime, List[Alert]] = {}
        
        for alert in alerts:
            # 找到最近的时间分组
            found_group = False
            for group_time in time_groups:
                if abs((alert.timestamp - group_time).total_seconds()) < self.aggregation_window.total_seconds():
                    time_groups[group_time].append(alert)
                    found_group = True
                    break
            
            if not found_group:
                time_groups[alert.timestamp] = [alert]
        
        # 对每个时间分组，合并相关预警
        aggregated = []
        for group_time, group_alerts in time_groups.items():
            if len(group_alerts) == 1:
                aggregated.append(group_alerts[0])
            else:
                # 合并为一个综合预警
                combined = Alert(
                    alert_id=f"combined-{group_time.timestamp()}",
                    severity=max(a.severity for a in group_alerts),
                    title=f"检测到 {len(group_alerts)} 个相关异常",
                    description="\n".join(a.description for a in group_alerts),
                    timestamp=group_time,
                    affected_services=list(set(
                        s for a in group_alerts for s in (a.affected_services or [])
                    )),
                    suggested_actions=list(set(
                        action for a in group_alerts for action in (a.suggested_actions or [])
                    )),
                )
                aggregated.append(combined)
        
        return aggregated
```

---

## 五、Prometheus + Grafana 实战配置

### 5.1 指标定义

```yaml
# prometheus-rules.yml
groups:
  - name: llm_inference_rules
    rules:
      # 推理延迟
      - record: llm:inference_latency_seconds:p95
        expr: histogram_quantile(0.95, 
          rate(llm_inference_duration_seconds_bucket[5m]))
      
      # 吞吐量
      - record: llm:tokens_per_second:avg
        expr: rate(llm_tokens_generated_total[5m]) / 
              rate(llm_inference_duration_seconds_count[5m])
      
      # 质量分数
      - record: llm:quality_score:avg
        expr: avg(llm_output_quality_score) by (model)
      
      # 安全性分数
      - record: llm:safety_score:avg
        expr: avg(llm_output_safety_score) by (model)

  - name: llm_alerts
    rules:
      # 质量退化告警
      - alert: LLMQualityDegradation
        expr: llm:quality_score:avg < 0.7
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "模型质量退化"
          description: "模型 {{ $labels.model }} 质量分数降至 {{ $value }}"
      
      # 延迟异常告警
      - alert: LLMLatencySpike
        expr: llm:inference_latency_seconds:p95 > 2
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "推理延迟异常"
          description: "P95延迟达到 {{ $value }}秒"
      
      # 错误率告警
      - alert: LLMHighErrorRate
        expr: rate(llm_inference_errors_total[5m]) / 
              rate(llm_inference_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM推理错误率过高"
          description: "错误率达到 {{ $value | humanizePercentage }}"
      
      # 数据漂移告警
      - alert: LLMDataDrift
        expr: llm_input_feature_drift_psi > 0.25
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "输入数据漂移"
          description: "特征 {{ $labels.feature }} PSI值达到 {{ $value }}"
```

### 5.2 Grafana仪表盘配置

```json
{
  "dashboard": {
    "title": "LLM推理监控仪表盘",
    "panels": [
      {
        "title": "推理延迟分布",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(llm_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(llm_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(llm_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "模型质量分数",
        "type": "gauge",
        "targets": [
          {
            "expr": "llm:quality_score:avg",
            "legendFormat": "{{model}}"
          }
        ],
        "thresholds": [
          {"value": 0, "color": "red"},
          {"value": 0.6, "color": "yellow"},
          {"value": 0.8, "color": "green"}
        ]
      },
      {
        "title": "GPU利用率",
        "type": "timeseries",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "Token使用量",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(llm_tokens_generated_total[1h]))",
            "legendFormat": "Tokens/小时"
          }
        ]
      },
      {
        "title": "错误率趋势",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(llm_inference_errors_total[5m]) / rate(llm_inference_requests_total[5m])",
            "legendFormat": "错误率"
          }
        ],
        "thresholds": [
          {"value": 0.01, "color": "yellow"},
          {"value": 0.05, "color": "red"}
        ]
      }
    ]
  }
}
```

### 5.3 部署配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus-rules.yml:/etc/prometheus/rules.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - loki_data:/loki

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "6831:6831"    # Agent

  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"

volumes:
  prometheus_data:
  grafana_data:
  loki_data:
```

---

## 六、生产环境最佳实践

### 6.1 可观测性成熟度模型

```
可观测性成熟度等级:
─────────────────────────────────────────────

Level 1: 基础监控
  ✅ 关键指标收集（延迟、吞吐、错误率）
  ✅ 基础告警（阈值告警）
  ✅ 简单仪表盘
  ❌ 缺少模型质量监控
  ❌ 缺少分布式追踪

Level 2: 全面监控
  ✅ 模型质量指标
  ✅ 分布式追踪
  ✅ 日志聚合
  ✅ 基础数据漂移检测
  ❌ 缺少智能分析
  ❌ 缺少自动根因分析

Level 3: 智能监控
  ✅ 异常检测（ML模型）
  ✅ 自动根因分析
  ✅ 预测性告警
  ✅ 自适应采样
  ❌ 缺少闭环自动化

Level 4: 自主运维
  ✅ 自动故障恢复
  ✅ 自动扩缩容
  ✅ 自动模型回滚
  ✅ 自动质量优化
```

### 6.2 关键实践建议

```
生产环境最佳实践:
─────────────────────────────────────────────

1. 采样策略
   • 正常请求: 10%采样（节省存储）
   • 错误请求: 100%采样（完整追踪）
   • 高质量请求: 50%采样（用于质量分析）
   
2. 告警疲劳管理
   • 分级告警: info → warning → critical → emergency
   • 告警聚合: 相关告警合并，避免告警风暴
   • 静默规则: 维护窗口自动静默
   • 告警升级: 未处理的告警自动升级
   
3. 数据保留策略
   • 原始日志: 7天
   • 聚合指标: 90天
   • 追踪数据: 30天
   • 质量报告: 永久
   
4. 性能影响最小化
   • 异步收集: 所有遥测数据异步发送
   • 本地缓存: 网络异常时本地缓存
   • 采样降级: 高负载时自动降低采样率
   • 资源隔离: 监控组件使用独立资源
```

### 6.3 成本优化

```
可观测性成本优化:
─────────────────────────────────────────────

成本构成:
┌──────────────────┬────────────┬────────────┐
│ 组件             │ 占比       │ 优化空间   │
├──────────────────┼────────────┼────────────┤
│ 存储             │ 45%        │ 高         │
│ 计算             │ 30%        │ 中         │
│ 网络             │ 15%        │ 低         │
│ 许可证           │ 10%        │ 中         │
└──────────────────┴────────────┴────────────┘

优化策略:
1. 智能采样
   • 降低正常请求采样率到5%
   • 保留错误请求100%采样
   • 预期成本降低: 40-60%

2. 数据分层
   • 热数据: SSD存储（7天）
   • 温数据: HDD存储（90天）
   • 冷数据: 对象存储（归档）
   • 预期成本降低: 30-50%

3. 聚合优化
   • 减少原始数据保留时间
   • 增加聚合数据粒度
   • 预期成本降低: 20-40%

4. 开源替代
   • Prometheus替代Datadog
   • Loki替代ELK
   • Jaeger替代商业APM
   • 预期成本降低: 60-80%
```

---

## 七、总结

### 7.1 核心要点

```
AI系统可观测性关键要点:
─────────────────────────────────────────────

1. 传统监控不足
   • 需要模型质量、数据漂移、用户反馈等额外维度
   • 响应时间正常不代表用户体验正常

2. 四大数据支柱
   • 日志、指标、追踪、反馈
   • 缺一不可，相互关联

3. 智能化是趋势
   • 从阈值告警到智能异常检测
   • 从被动响应到预测性运维

4. 成本需要平衡
   • 采样策略是关键
   • 分层存储是基础

5. 持续演进
   • 从Level 1逐步提升到Level 4
   • 根据业务需求选择合适的成熟度
```

### 7.2 实施路线图

```
实施路线图:
─────────────────────────────────────────────

第1阶段 (1-2周): 基础建设
  • 部署Prometheus + Grafana
  • 定义核心指标
  • 配置基础告警

第2阶段 (2-4周): 全面覆盖
  • 集成分布式追踪
  • 部署日志聚合
  • 添加模型质量评估

第3阶段 (1-2月): 智能化
  • 实现异常检测
  • 添加数据漂移监控
  • 优化告警规则

第4阶段 (持续): 优化演进
  • 成本优化
  • 性能优化
  • 新功能迭代
```

---

## 参考资源

- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Documentation: https://grafana.com/docs/
- OpenTelemetry Specification: https://opentelemetry.io/docs/
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
- Evidently AI - ML Monitoring: https://www.evidentlyai.com/
- Arize AI - ML Observability: https://arize.com/
