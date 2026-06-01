---
title: "LLM应用可观测性最佳实践：从日志采集到智能告警的全链路监控体系"
description: "深度解析LLM应用可观测性的完整实践方案，涵盖Trace、Metrics、Logging三支柱，结合实际生产经验给出可落地的监控告警体系设计"
date: 2026-06-01
author: "RiceBall"
category: "featured"
subCategory: "deep-dive"
tags: ["可观测性", "LLM监控", "分布式追踪", "Prometheus", "Grafana", "智能告警", "AIOps"]
draft: false
---

# LLM应用可观测性最佳实践：从日志采集到智能告警的全链路监控体系

## 引言：为什么LLM应用的可观测性如此特殊？

传统的Web应用监控，我们关注的是：请求延迟、错误率、QPS、CPU/内存使用率。这套体系已经非常成熟，Prometheus + Grafana几乎成了行业标准。

但当你把LLM应用推向生产环境时，会发现**传统的监控体系完全不够用**：

- **Token级别的成本追踪**：一个请求消耗了多少Token、花了多少钱？传统监控根本不记录这些
- **输出质量监控**：回答是否准确？是否产生了幻觉？是否安全合规？这些"软指标"怎么量化？
- **流式响应监控**：ChatGPT式的SSE流式输出，如何监控首Token延迟和总生成时间？
- **多模型切换监控**：同一个接口背后可能调用了3-5个不同的模型，如何区分监控？
- **Prompt版本关联**：出了问题，是模型的锅还是Prompt的锅？如何快速定位？

我在一个日均处理200万次LLM调用的平台上工作时，曾经遇到过一个经典case：线上用户投诉"回答质量突然变差了"。我们翻了两天日志，发现：
1. 模型没有变更
2. Prompt没有变更
3. 代码没有变更
4. 但是用户满意度确实下降了15%

最终定位到原因：**上游知识库的一个数据源停止更新了，导致RAG检索到的内容过时**。这个case让我深刻认识到：**LLM应用的可观测性，必须覆盖到数据层**。

## LLM应用可观测性的独特挑战

先看一张全景对比表：

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│    观测维度      │   传统Web应用        │   LLM应用            │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 核心指标        │ QPS/延迟/错误率      │ Token/质量/延迟/成本  │
│ 数据粒度        │ 请求级               │ Token级              │
│ 质量度量        │ HTTP状态码           │ 语义相关性/安全性     │
│ 成本追踪        │ 服务器成本           │ 每次调用的Token成本   │
│ 变更影响        │ 代码部署             │ Prompt/模型/数据变更  │
│ 失败模式        │ 超时/500错误         │ 幻觉/偏见/注入攻击   │
│ 依赖关系        │ 数据库/缓存/消息队列  │ 模型/向量库/知识库    │
│ 用户体验        │ 页面加载速度         │ 首Token延迟/生成质量  │
└─────────────────┴──────────────────────┴──────────────────────┘
```

这个对比告诉我们：**LLM应用需要一套全新的可观测性体系**。

## 三支柱体系：Trace、Metrics、Logging

### 支柱一：分布式追踪（Trace）——看见请求的完整生命周期

LLM应用的调用链通常比传统应用更长、更复杂：

```
┌───────────────────────────────────────────────────────────┐
│                 LLM应用完整调用链                           │
│                                                           │
│  用户请求                                                 │
│     │                                                     │
│     ▼                                                     │
│  ┌─────────────┐                                          │
│  │ API Gateway │ ← TraceID生成                            │
│  └──────┬──────┘                                          │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────┐                                          │
│  │ Prompt引擎  │ ← 获取Prompt模板、变量注入                │
│  └──────┬──────┘                                          │
│         │                                                 │
│    ┌────┴────┐                                            │
│    │         │                                            │
│    ▼         ▼                                            │
│  ┌──────┐ ┌──────────┐                                    │
│  │RAG   │ │知识检索   │ ← 向量检索 + 关键词检索            │
│  │检索  │ │增强       │                                    │
│  └──┬───┘ └─────┬────┘                                    │
│     │           │                                         │
│     └─────┬─────┘                                         │
│           ▼                                               │
│     ┌──────────┐                                          │
│     │ 内容安全  │ ← 敏感内容过滤                           │
│     │ 检查     │                                          │
│     └─────┬────┘                                          │
│           │                                               │
│           ▼                                               │
│     ┌──────────┐                                          │
│     │ LLM推理  │ ← 实际模型调用（SSE流式）                 │
│     │ 调用     │                                          │
│     └─────┬────┘                                          │
│           │                                               │
│           ▼                                               │
│     ┌──────────┐                                          │
│     │ 后处理   │ ← 格式化、安全检查、引用标注              │
│     └─────┬────┘                                          │
│           │                                               │
│           ▼                                               │
│     ┌──────────┐                                          │
│     │ 结果返回  │ ← 流式结束、Token统计                    │
│     └──────────┘                                          │
│                                                           │
│  总耗时: 2.3s | Token: 输入1,245 输出892 | 成本: ¥0.015  │
└───────────────────────────────────────────────────────────┘
```

**Trace数据模型设计**：

```json
{
  "traceId": "abc-123-def-456",
  "spanId": "span-001",
  "parentSpanId": "span-000",
  "operationName": "llm.inference",
  "startTime": "2026-06-01T14:30:00.123Z",
  "duration": 1247,
  "tags": {
    "model": "gpt-4o-2026-05",
    "model_provider": "openai",
    "input_tokens": 1245,
    "output_tokens": 892,
    "total_tokens": 2137,
    "cost_usd": 0.015,
    "temperature": 0.3,
    "max_tokens": 2048,
    "stream": true,
    "first_token_latency_ms": 234,
    "tokens_per_second": 42.5,
    "prompt_version": "v3.2.0",
    "experiment_id": "prompt-opt-v12",
    "user_id": "user-789",
    "app_id": "customer-service"
  },
  "logs": [
    {
      "timestamp": "2026-06-01T14:30:00.200Z",
      "level": "INFO",
      "message": "RAG检索完成，找到3条相关文档",
      "fields": {
        "retrieval_count": 3,
        "retrieval_latency_ms": 67,
        "top_score": 0.92
      }
    }
  ]
}
```

**关键设计决策**：

1. **Token级别的Span**：每个LLM调用的Span必须记录input/output tokens和成本
2. **首Token延迟**：对于流式响应，单独记录Time-to-First-Token (TTFT)
3. **Prompt版本关联**：每次调用都要记录使用的Prompt版本，便于质量归因
4. **实验标记**：A/B实验的分组信息要写入Trace，便于对比分析

### 支柱二：指标体系（Metrics）——量化AI应用的健康度

LLM应用的指标体系需要比传统应用更丰富。我总结了一个**四层指标模型**：

```
┌───────────────────────────────────────────────────────┐
│                 LLM应用四层指标模型                     │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 第四层：业务指标                                  │ │
│  │  - 用户满意度 (CSAT)                              │ │
│  │  - 任务完成率                                     │ │
│  │  - 对话轮次                                       │ │
│  │  - 转化率/留存率                                  │ │
│  └─────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 第三层：质量指标                                  │ │
│  │  - 回答相关性评分                                  │ │
│  │  - 幻觉率                                         │ │
│  │  - 安全合规率                                      │ │
│  │  - 引用准确率                                      │ │
│  └─────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 第二层：性能指标                                  │ │
│  │  - 请求延迟 (P50/P95/P99)                        │ │
│  │  - 首Token延迟 (TTFT)                            │ │
│  │  - Token生成速率 (TPS)                           │ │
│  │  - 并发数/队列深度                                │ │
│  └─────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 第一层：资源指标                                  │ │
│  │  - Token消耗 (输入/输出/总)                       │ │
│  │  - API调用成本                                    │ │
│  │  - GPU利用率 (自部署场景)                          │ │
│  │  - 网络带宽                                       │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

**Prometheus指标定义**：

```python
# LLM应用核心Prometheus指标
from prometheus_client import Counter, Histogram, Gauge

# ========== 第一层：资源指标 ==========

# Token消耗统计
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens consumed',
    ['app_id', 'model', 'provider', 'direction']  # direction: input/output
)

# API调用成本
llm_cost_total = Counter(
    'llm_cost_total',
    'Total API cost in USD',
    ['app_id', 'model', 'provider']
)

# ========== 第二层：性能指标 ==========

# 请求延迟分布
llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['app_id', 'model', 'stream'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# 首Token延迟
llm_first_token_latency_seconds = Histogram(
    'llm_first_token_latency_seconds',
    'Time to first token',
    ['app_id', 'model'],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# Token生成速率
llm_tokens_per_second = Histogram(
    'llm_tokens_per_second',
    'Tokens generated per second',
    ['app_id', 'model'],
    buckets=[10, 20, 30, 50, 80, 100, 150]
)

# ========== 第三层：质量指标 ==========

# 回答质量评分（需要人工或自动评估）
llm_quality_score = Gauge(
    'llm_quality_score',
    'Response quality score (0-1)',
    ['app_id', 'model', 'evaluation_method']
)

# 幻觉检测率
llm_hallucination_rate = Gauge(
    'llm_hallucination_rate',
    'Hallucination detection rate',
    ['app_id', 'model']
)

# 安全拦截率
llm_safety_block_rate = Counter(
    'llm_safety_block_total',
    'Safety check blocks',
    ['app_id', 'block_reason']
)

# ========== 第四层：业务指标 ==========

# 用户满意度（通过反馈收集）
llm_user_satisfaction = Gauge(
    'llm_user_satisfaction',
    'User satisfaction score',
    ['app_id']
)

# 任务完成率
llm_task_completion_rate = Gauge(
    'llm_task_completion_rate',
    'Task completion rate',
    ['app_id', 'task_type']
)
```

### 支柱三：结构化日志（Logging）——事后分析的素材

LLM应用的日志需要比传统应用更结构化，因为**出了问题你需要快速定位是哪个环节**：

```python
import structlog

logger = structlog.get_logger()

class LLMRequestLogger:
    """LLM请求结构化日志记录器"""
    
    def log_request(self, request, response, trace_id):
        # 记录完整的请求上下文
        logger.info(
            "llm_request_completed",
            trace_id=trace_id,
            
            # 请求信息
            app_id=request.app_id,
            user_id=request.user_id,
            
            # 模型信息
            model=response.model,
            provider=response.provider,
            prompt_version=request.prompt_version,
            
            # Token统计
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            
            # 性能数据
            total_latency_ms=response.latency_ms,
            first_token_latency_ms=response.first_token_latency,
            tokens_per_second=response.tps,
            
            # 成本
            cost_usd=response.cost,
            
            # 质量信号
            finish_reason=response.finish_reason,
            safety_check_passed=response.safety_passed,
            
            # 实验信息
            experiment_id=request.experiment_id,
            variant_id=request.variant_id,
            
            # 输入输出摘要（截断到安全长度）
            input_preview=request.messages[-1]['content'][:200],
            output_preview=response.content[:200],
        )
    
    def log_error(self, request, error, trace_id):
        logger.error(
            "llm_request_failed",
            trace_id=trace_id,
            app_id=request.app_id,
            model=request.model,
            error_type=type(error).__name__,
            error_message=str(error),
            
            # 错误分类
            error_category=self.classify_error(error),
            # 重试信息
            retry_count=request.retry_count,
            # 降级信息
            fallback_used=request.fallback_model is not None,
        )
    
    def classify_error(self, error) -> str:
        """错误分类：便于告警规则配置"""
        error_str = str(error).lower()
        if 'rate_limit' in error_str:
            return 'rate_limit'
        elif 'timeout' in error_str:
            return 'timeout'
        elif 'context_length' in error_str:
            return 'context_overflow'
        elif 'safety' in error_str or 'content_policy' in error_str:
            return 'safety_block'
        else:
            return 'unknown'
```

## 智能告警：从"阈值告警"到"智能检测"

### 传统告警的局限

传统的阈值告警在LLM应用中有明显的局限：

```
传统告警规则：
  IF latency_p99 > 5000ms THEN alert

问题：
1. 周一早上9点延迟本来就高（用户量大），频繁误报
2. Prompt质量下降但延迟没变，完全漏报
3. 成本异常但没超过阈值，无法及时发现
```

### LLM应用智能告警体系

```
┌───────────────────────────────────────────────────────┐
│              LLM智能告警体系                           │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 基础告警层（阈值规则）                            │ │
│  │  - 错误率 > 5%                                   │ │
│  │  - 延迟P99 > 10s                                 │ │
│  │  - Token消耗突增 > 200%                           │ │
│  │  - GPU显存 > 90%                                 │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         │                             │
│  ┌──────────────────────▼──────────────────────────┐ │
│  │ 趋势告警层（基线对比）                            │ │
│  │  - 质量评分偏离历史均值 > 2σ                       │ │
│  │  - 成本趋势异常（日环比）                          │ │
│  │  - TTFT持续劣化（连续5分钟上升）                   │ │
│  │  - 用户满意度下降趋势                              │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         │                             │
│  ┌──────────────────────▼──────────────────────────┐ │
│  │ 关联告警层（多维度关联）                           │ │
│  │  - Prompt变更 + 质量下降 → Prompt回滚建议          │ │
│  │  - 模型切换 + 延迟上升 → 模型切换建议              │ │
│  │  - 知识库更新 + 相关性下降 → 数据层排查             │ │
│  │  - 多实验同时异常 → 实验隔离检查                   │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         │                             │
│  ┌──────────────────────▼──────────────────────────┐ │
│  │ 根因分析层（AI辅助）                              │ │
│  │  - 自动关联最近的配置变更                          │ │
│  │  - 智能推断可能的故障原因                          │ │
│  │  - 生成处置建议和回滚方案                          │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

**Prometheus告警规则示例**：

```yaml
# LLM应用告警规则
groups:
  - name: llm_basic_alerts
    rules:
      # 基础：错误率告警
      - alert: LLMHighErrorRate
        expr: |
          sum(rate(llm_requests_total{status="error"}[5m])) 
          / sum(rate(llm_requests_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM错误率超过5%"
          
      # 基础：延迟告警
      - alert: LLMHighLatency
        expr: |
          histogram_quantile(0.99, 
            rate(llm_request_duration_seconds_bucket[5m])
          ) > 10
        for: 3m
        labels:
          severity: warning
          
      # 趋势：成本异常
      - alert: LLMCostAnomaly
        expr: |
          llm_cost_total - llm_cost_total offset 1d 
          > 2 * (llm_cost_total offset 1d - llm_cost_total offset 2d)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "LLM日成本同比异常增长超过200%"
          
      # 趋势：质量下降
      - alert: LLMQualityDegradation
        expr: |
          llm_quality_score < 0.7
          and llm_quality_score offset 1h > 0.8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM回答质量评分下降超过10个百分点"
```

## 实战：构建LLM监控看板

### Grafana Dashboard设计

一个好的LLM监控看板应该包含以下几个核心面板：

```
┌───────────────────────────────────────────────────────────┐
│                    LLM应用监控看板                         │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 📊 核心指标概览                                      │ │
│  │                                                      │ │
│  │  今日调用: 2.3M    成本: $1,234    质量: 87.2%     │ │
│  │  平均延迟: 1.2s    TTFT: 280ms    TPS: 45.3       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────┬──────────────────────────────┐ │
│  │ 📈 请求趋势          │ 💰 成本分布                   │ │
│  │                      │                              │ │
│  │  QPS曲线             │  按模型: GPT-4o 60%           │ │
│  │  错误率曲线          │          Claude 25%           │ │
│  │  延迟分布            │          其他 15%             │ │
│  │                      │                              │ │
│  │  [实时折线图]         │  [饼图/堆叠柱状图]           │ │
│  └──────────────────────┴──────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────┬──────────────────────────────┐ │
│  │ 🧠 质量监控          │ ⚡ 性能监控                   │ │
│  │                      │                              │ │
│  │  质量评分趋势        │  TTFT趋势                    │ │
│  │  幻觉率趋势          │  TPS分布                     │ │
│  │  安全拦截率          │  P50/P95/P99延迟             │ │
│  │                      │                              │ │
│  │  [折线图+阈值线]     │  [分位数曲线]                │ │
│  └──────────────────────┴──────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 🔍 Trace分析                                         │ │
│  │                                                      │ │
│  │  慢查询Top10 | 错误Trace | 成本最高请求              │ │
│  │                                                      │ │
│  │  [表格 + 点击跳转详情]                                │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────┬──────────────────────────────┐ │
│  │ 🧪 A/B实验对比       │ 📋 Prompt版本对比             │ │
│  │                      │                              │ │
│  │  Control vs A vs B  │  v3.1.0 vs v3.2.0            │ │
│  │  质量/延迟/成本对比   │  质量/成本/满意度对比         │ │
│  │                      │                              │ │
│  │  [并排柱状图]         │  [对比折线图]                │ │
│  └──────────────────────┴──────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### 核心Grafana面板配置

```json
{
  "dashboard": {
    "title": "LLM应用监控看板",
    "panels": [
      {
        "title": "今日Token消耗趋势",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(llm_tokens_total{direction=\"input\"}[5m]))",
            "legendFormat": "输入Token/s"
          },
          {
            "expr": "sum(rate(llm_tokens_total{direction=\"output\"}[5m]))",
            "legendFormat": "输出Token/s"
          }
        ]
      },
      {
        "title": "各模型调用成本占比",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum(increase(llm_cost_total[1h])) by (model)",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "首Token延迟分布",
        "type": "histogram",
        "targets": [
          {
            "expr": "llm_first_token_latency_seconds_bucket",
            "legendFormat": "{{le}}s"
          }
        ]
      }
    ]
  }
}
```

## 自动化质量评估：让"软指标"变硬

LLM应用最难监控的是"质量"——回答是否准确、是否有幻觉、是否安全。我总结了一个**三级质量评估体系**：

```
┌───────────────────────────────────────────────────────┐
│              三级质量评估体系                           │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 第一级：自动规则检查（实时，100%覆盖）             │ │
│  │  - 回答长度是否在合理范围                          │ │
│  │  - 是否包含拒绝话术（不该拒绝时拒绝了）            │ │
│  │  - 是否包含敏感信息泄露                            │ │
│  │  - 引用格式是否正确                                │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 第二级：LLM-as-Judge（近实时，采样评估）           │ │
│  │  - 用小模型评估大模型的回答质量                    │ │
│  │  - 相关性评分（0-10）                              │ │
│  │  - 幻觉检测（是否捏造事实）                        │ │
│  │  - 安全性检查                                      │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 第三级：人工评估（离线，抽样评估）                  │ │
│  │  - 专家标注回答质量                                │ │
│  │  - 用户满意度反馈                                  │ │
│  │  - 定期基准测试（benchmark）                       │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

**LLM-as-Judge实现**：

```python
class LLMQualityEvaluator:
    """使用小模型评估大模型的回答质量"""
    
    JUDGE_PROMPT = """你是一个AI回答质量评估专家。请评估以下AI回答的质量。

用户问题：{question}
AI回答：{answer}
参考知识：{context}

请从以下维度评分（0-10分）：
1. 相关性：回答是否切题
2. 准确性：信息是否正确
3. 完整性：是否覆盖了问题的关键方面
4. 幻觉：是否包含无中生有的信息（0=无幻觉，10=严重幻觉）
5. 安全性：是否存在有害内容

请返回JSON格式：
{{"relevance": 8, "accuracy": 9, "completeness": 7, "hallucination": 1, "safety": 10}}"""

    def evaluate(self, question, answer, context, sample_rate=0.05):
        """采样评估（5%的请求进行质量评估）"""
        if random.random() > sample_rate:
            return None
        
        # 使用小模型进行评估（成本低）
        judge_response = call_model(
            model="gpt-4o-mini",
            prompt=self.JUDGE_PROMPT.format(
                question=question,
                answer=answer,
                context=context[:2000]  # 截断避免过长
            ),
            params={"temperature": 0, "response_format": "json"}
        )
        
        scores = json.loads(judge_response)
        
        # 记录到Prometheus
        llm_quality_score.labels(
            app_id=self.app_id,
            model=self.model,
            evaluation_method="llm_judge"
        ).set(self.weighted_average(scores))
        
        return scores
```

## 实战踩坑与经验总结

### 踩坑1：流式响应的监控陷阱

**问题**：SSE流式响应中，如果用户中途断开连接，传统监控会记录为"成功"（HTTP 200），但实际上回答是不完整的。

**解决方案**：

```python
class StreamingMonitor:
    def monitor_stream(self, stream, trace_id):
        chunks = []
        start_time = time.time()
        
        try:
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except ClientDisconnect:
            # 用户断开连接
            record_metric(
                "stream_incomplete",
                trace_id=trace_id,
                chunks_received=len(chunks),
                duration=time.time() - start_time
            )
            raise
        finally:
            # 无论是否完成，都记录流式统计
            complete = len(chunks) > 0 and self.is_final_chunk(chunks[-1])
            record_metric(
                "stream_complete" if complete else "stream_interrupted",
                trace_id=trace_id,
                total_chunks=len(chunks),
                total_tokens=sum(c.get('tokens', 0) for c in chunks),
                duration=time.time() - start_time
            )
```

### 踩坑2：成本监控的"温水煮青蛙"

**问题**：成本是缓慢增长的，不会突然翻倍，但日积月累的浪费非常可观。

**解决方案**：多维度成本分析

```sql
-- 按维度分析成本分布
SELECT 
    app_id,
    model,
    DATE_TRUNC('hour', timestamp) as hour,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(cost_usd) as total_cost,
    -- 成本效率指标
    SUM(cost_usd) / COUNT(*) as cost_per_request,
    SUM(cost_usd) / NULLIF(SUM(output_tokens), 0) as cost_per_output_token
FROM llm_cost_logs
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY app_id, model, DATE_TRUNC('hour', timestamp)
ORDER BY total_cost DESC;
```

### 踩坑3：告警疲劳

**问题**：LLM应用的告警特别多（模型偶尔超时、内容偶尔被拦截等），如果都发告警，运维同学会被淹没。

**解决方案**：告警分级 + 聚合

```
告警分级策略：
├── P0（立即处理）：错误率>10%持续5分钟、全量质量下降
├── P1（30分钟内处理）：错误率>5%持续10分钟、成本异常
├── P2（2小时内处理）：延迟持续上升、单模型质量下降
└── P3（日报汇总）：轻微波动、趋势变化

告警聚合规则：
├── 同一告警5分钟内不重复发送
├── 同类型告警聚合为一条（如"3个模型同时超时"）
└── 非工作时间只发P0告警
```

## 技术栈推荐

| 组件 | 推荐方案 | 说明 |
|------|---------|------|
| 指标采集 | Prometheus + OpenTelemetry | 标准化采集，支持自动埋点 |
| 日志收集 | Fluent Bit + ClickHouse | 轻量采集，列式存储高效查询 |
| 分布式追踪 | Jaeger / Tempo | 与OpenTelemetry集成 |
| 可视化 | Grafana | 丰富的LLM监控Dashboard |
| 告警管理 | Alertmanager + PagerDuty | 分级告警与升级机制 |
| 质量评估 | LLM-as-Judge + 人工标注 | 自动+人工双重保障 |
| 成本分析 | 自研成本分析模块 | 按模型/应用/用户多维度分析 |

## 总结：可观测性是LLM应用的生命线

LLM应用的可观测性不是一个"可选的锦上添花"，而是**生产环境的必备能力**。没有完善的可观测性，你会：

1. **无法定位问题**：用户说"回答变差了"，你不知道是模型、Prompt还是数据的问题
2. **无法控制成本**：Token成本在悄悄增长，你到月底看账单才发现
3. **无法保证质量**：幻觉、偏见、安全问题，你只能等用户投诉才知道
4. **无法持续优化**：没有数据支撑，你不知道哪些改进建议是有效的

构建LLM可观测性的核心原则：

- **全链路覆盖**：从用户请求到模型返回，每一个环节都要可观测
- **多维度指标**：性能、质量、成本、业务四个维度缺一不可
- **智能告警**：不是"有问题才告警"，而是"有趋势就预警"
- **闭环反馈**：监控→发现问题→分析原因→优化→验证效果→持续监控

**可观测性做得好，LLM应用才能真正上线。** 在这个AI应用快速迭代的时代，可观测性不是成本，而是投资——投资于你对系统的**理解和控制能力**。
