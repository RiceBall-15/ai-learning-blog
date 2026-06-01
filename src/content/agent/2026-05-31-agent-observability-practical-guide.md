---
title: "Agent可观测性实战：从零搭建生产级Tracing与监控系统"
description: "从零搭建Agent可观测性系统，覆盖OTel部署、Span设计、成本追踪、告警策略到生产踩坑，面试深度指南。"
date: 2026-05-31
author: 'RiceBall-15'
category: 'agent'
subCategory: interview
tags: ['Agent可观测性', 'Tracing', '监控', '面试']
draft: false
---

# Agent可观测性实战：从零搭建生产级Tracing与监控系统

> **导读**：本文是面试方向7（全链路Tracing）的深度补充。当你把一个Agent从Demo推向生产环境时，最先崩溃的往往不是推理质量，而是"你根本不知道发生了什么"。本文从Agent可观测性的独特挑战出发，覆盖Tracing系统搭建、Span设计、成本追踪、告警设计到生产踩坑的全流程，帮助你在面试中展现真正的系统设计深度。

---

## §1 Agent可观测性的独特挑战

传统的微服务可观测性已经是一门成熟的学科。但当你把Agent系统扔进去的时候，你会发现——几乎所有的假设都失效了。

### 1.1 非确定性：同一个输入，不同的路径

传统服务是确定性的：同样的请求走同样的代码路径，返回同样的结果。Agent不是。

```
用户输入: "帮我分析上周的销售数据"
├─ 路径A: 直接调用SQL工具查询 → 1次LLM调用 → 完成
├─ 路径B: 先推理需要拆分问题 → 3次LLM调用 → 2次工具调用 → 完成
└─ 路径C: 第一次SQL失败 → 重试 → 换查询策略 → 5次LLM调用 → 完成
```

这意味着：**你不能预定义指标的基线**。一个任务可能1秒完成，同一个任务可能花30秒。你的告警阈值该怎么设？

### 1.2 长链路：一个Trace可能包含50+个Span

一个典型的ReAct Agent任务的执行链路：

| 环节 | 平均Span数 | 说明 |
|------|-----------|------|
| 任务规划 | 1-3 | 理解意图、拆分子任务 |
| 推理循环 | 5-20 | 每轮ReAct循环1个推理Span + N个工具Span |
| 工具调用 | 3-15 | SQL查询、API调用、文件读取等 |
| 结果整合 | 2-5 | 多轮总结、格式化输出 |

一次用户交互可能产生30-50个Span，而传统微服务API调用通常只有3-5个。

### 1.3 多模型调用：成本和延迟的黑洞

现代Agent架构中，一个任务可能涉及多个模型的调用：

```
用户请求
  → Embedding模型（检索增强）     // 便宜，快
  → 小模型（意图分类）            // 便宜，快  
  → 大模型（核心推理）            // 贵，慢
  → 小模型（格式化输出）          // 便宜，快
  → Embedding模型（记忆写入）     // 便宜，快
```

每个模型的成本、延迟、错误率都不同，但它们在同一个Trace里。**如果不做细粒度的Span设计，你根本分不清钱花在哪了。**

### 1.4 成本敏感：每秒都在烧钱

与传统API调用不同，LLM调用按Token计费。一个Agent的推理循环一旦进入"死循环"（不断重试、反复推理），每秒都在烧钱。传统的错误率和延迟监控无法捕捉到这种"成本失控"问题。

### 1.5 为什么传统APM在Agent面前失效

传统的APM（Application Performance Monitoring）工具假设了一个稳定的请求-响应模型：请求进入，经过若干确定的处理步骤，返回响应。但Agent系统打破了这个假设。Agent可能会自己决定调用什么工具、走什么路径、何时终止。更关键的是，Agent的行为还受到LLM模型本身的影响——模型的输出是概率性的，同样的输入可能产生不同的行为序列。这意味着，传统的"基于固定阈值"的告警策略在Agent场景下会频繁误报，而真正的异常又可能被淹没在噪声中。

> **核心观点**：Agent可观测性不是传统APM的简单延伸，它需要一套专门为非确定性、多模型、成本敏感场景设计的监控体系。这不仅是技术问题，更是监控哲学的根本转变——从"监控确定性行为"到"理解概率性行为"。

---

## §2 三大支柱在Agent中的映射

可观测性的三大支柱——Tracing、Metrics、Logs——在Agent系统中各有新的语义。

### 2.1 Trace → Span：Agent执行链路的完整记录

传统Tracing关注的是请求在服务间的流转。Agent Tracing关注的是**思考过程的完整记录**。

一个Agent Trace的核心结构：

```
Trace: agent_task_001
│
├─ Span: agent_react_loop (root, 总耗时)
│   │
│   ├─ Span: reasoning_cycle_1 (第1轮推理)
│   │   ├─ Span: llm_call_gpt4 (思考过程, 2048 tokens)
│   │   ├─ Span: tool_call_sql (查询数据库, 120ms)
│   │   └─ Span: llm_call_gpt4 (评估结果, 512 tokens)
│   │
│   ├─ Span: reasoning_cycle_2 (第2轮推理)
│   │   ├─ Span: llm_call_gpt4 (继续思考, 1024 tokens)
│   │   └─ Span: tool_call_api (调用外部API, 340ms)
│   │
│   ├─ Span: reasoning_cycle_3 (第3轮推理)
│   │   └─ Span: llm_call_gpt4 (生成最终答案, 3072 tokens)
│   │
│   └─ Span: memory_write (写入长期记忆)
│
├─ Metrics: agent_total_tokens, agent_total_cost, agent_total_latency
└─ Logs: [结构化] 用户输入、每步决策、最终输出
```

### 2.2 Metrics → 自定义指标：Agent专用的度量体系

传统Metrics主要关注QPS、延迟P99、错误率。Agent需要额外的维度：

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **Token消耗** | `agent.tokens.total` / `agent.tokens.input` / `agent.tokens.output` | 按模型、用户、任务类型分组 |
| **成本** | `agent.cost.usd` / `agent.cost.per_task` | 实时成本追踪 |
| **循环效率** | `agent.reasoning_cycles` / `agent.tool_calls_per_task` | 任务复杂度的代理指标 |
| **质量** | `agent.task_success_rate` / `agent.user_satisfaction` | 事后反馈 |
| **模型切换** | `agent.model_routing_decisions` | 不同模型的使用频率 |

### 2.3 Logs → 结构化日志：决策过程的完整审计

Agent日志必须记录**每一步决策的原因**，而不仅仅是结果：

```json
{
  "timestamp": "2026-05-31T10:23:45.123Z",
  "trace_id": "abc123",
  "level": "info",
  "component": "reasoning_engine",
  "event": "tool_selection",
  "context": {
    "step": 3,
    "available_tools": ["sql_query", "api_call", "file_read"],
    "selected_tool": "sql_query",
    "reason": "用户要求分析销售数据，SQL查询是最直接的方式",
    "confidence": 0.92
  }
}
```

这种日志不仅是调试工具，更是**合规审计**和**质量回溯**的基础。

### 2.4 三者的协同关系

```
          ┌─────────────────────────────────┐
          │         Grafana Dashboard        │
          │  ┌───────────┐  ┌─────────────┐  │
          │  │  Metrics   │  │   Traces    │  │
          │  │  (实时)    │  │  (详情)     │  │
          │  └─────┬─────┘  └──────┬──────┘  │
          │        │               │          │
          │  ┌─────▼───────────────▼──────┐   │
          │  │        Logs (上下文)        │   │
          │  └────────────────────────────┘   │
          └─────────────────────────────────┘
                     ▲           ▲
                     │           │
              Metrics关联    Trace关联
```

**工作流**：发现Metrics异常 → 通过trace_id定位到具体的Trace → 查看关联Logs理解上下文。三者通过`trace_id`和`span_id`串联。

### 2.5 三大支柱的数据流设计

在Agent场景中，三大支柱的数据流需要精心设计。常见的错误是将三者割裂开来独立建设，导致后续关联查询变得极其困难。正确的做法是在数据采集阶段就将`trace_id`注入到所有数据源中。具体来说，当OTel SDK创建一个Trace时，会生成一个全局唯一的`trace_id`，这个ID需要贯穿到Metrics的标签（label）和Logs的字段中。这样在Grafana中，你可以从一个异常的Metrics面板点击进入，直接跳转到对应的Trace详情，再从Trace中查看关联的Logs。这种"从宏观到微观"的逐层下钻能力，是Agent问题排查的生命线。

---

## §3 从零搭建Tracing系统

### 3.1 架构选型：OTel + Grafana 全家桶

推荐的技术栈：

| 组件 | 选型 | 理由 |
|------|------|------|
| **SDK** | OpenTelemetry SDK | 行业标准，厂商无关，生态最丰富 |
| **采集** | OTel Collector | 统一采集层，支持多种导出器 |
| **存储（Traces）** | Grafana Tempo | 无索引设计，存储成本低，与Grafana原生集成 |
| **存储（Metrics）** | Prometheus / Mimir | Prometheus成熟，Mimir支持多租户 |
| **存储（Logs）** | Loki | 轻量，标签索引，与Tempo原生关联 |
| **可视化** | Grafana | 统一面板，Trace-Metrics-Logs联动 |
| **告警** | Grafana Alerting / PagerDuty | 统一告警管理 |

### 3.2 部署架构

```
                        ┌──────────────────────┐
                        │    Agent 服务集群      │
                        │  (Python/TypeScript)  │
                        │   OTel SDK 内嵌       │
                        └──────────┬───────────┘
                                   │ OTLP (gRPC/HTTP)
                                   ▼
                        ┌──────────────────────┐
                        │   OTel Collector      │
                        │  ┌────────────────┐   │
                        │  │ Receivers      │   │  ← 接收OTLP数据
                        │  │ Processors     │   │  ← 采样、批处理、属性添加
                        │  │ Exporters      │   │  ← 分发到各存储后端
                        │  └────────────────┘   │
                        └──┬────┬────┬──────────┘
                           │    │    │
                ┌──────────┘    │    └──────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌───────────┐ ┌──────────────┐
        │ Grafana Tempo│ │ Prometheus│ │    Loki      │
        │ (Traces)     │ │ (Metrics) │ │   (Logs)     │
        └──────┬───────┘ └─────┬─────┘ └──────┬───────┘
               │               │               │
               └───────┬───────┴───────┬───────┘
                       ▼               ▼
                ┌──────────────────────────────┐
                │      Grafana Dashboard       │
                │  统一可视化 + 告警 + 查询      │
                └──────────────────────────────┘
```

### 3.3 OTel Collector 关键配置

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  # 批处理：减少网络请求
  batch:
    timeout: 5s
    send_batch_size: 1024

  # 自定义采样：对Agent场景优化
  probabilistic_sampler:
    sampling_percentage: 10  # 生产环境采样10%

  # 添加Agent相关属性
  attributes:
    actions:
      - key: service.env
        value: "production"
        action: upsert
      - key: agent.version
        value: "v2.3.1"
        action: upsert

exporters:
  # Traces → Tempo
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

  # Metrics → Prometheus
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: "agent"

  # Logs → Loki
  loki:
    endpoint: "http://loki:3100/loki/api/v1/push"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, probabilistic_sampler, attributes]
      exporters: [otlp/tempo]
    metrics:
      receivers: [otlp]
      processors: [batch, attributes]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [loki]
```

### 3.4 Docker Compose 快速部署

使用Docker Compose可以在几分钟内搭建完整的可观测性栈，非常适合本地开发验证和概念验证。以下是核心服务的编排配置：

```yaml
# docker-compose.yaml (精简版)
services:
  tempo:
    image: grafana/tempo:2.4.0
    volumes:
      - ./tempo-config.yaml:/etc/tempo/config.yaml
    ports:
      - "3200:3200"

  prometheus:
    image: prom/prometheus:v2.51.0
    volumes:
      - ./prometheus.yaml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"

  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.97.0
    volumes:
      - ./otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml
    ports:
      - "4317:4317"
      - "4318:4318"
```

> **实际建议**：在生产环境中，Tempo推荐使用对象存储（S3/MinIO）作为后端，避免本地磁盘瓶颈。Prometheus在大规模场景下用Mimir替换，以获得更好的水平扩展能力和多租户支持。Grafana需要预先配置好数据源，建议使用Provisioning机制自动关联Tempo、Prometheus和Loki。

---

## §4 Agent专用Span设计模式

Span设计是Agent可观测性的核心。一个好的Span设计能让你在排查问题时"看到"Agent的思考过程。

### 4.1 四种核心Span类型

| Span类型 | Name模式 | 关键Attributes | 用途 |
|---------|---------|---------------|------|
| **LLM调用** | `llm.{provider}.{model}` | `input_tokens`, `output_tokens`, `model_id`, `temperature`, `cost_usd` | 记录每次模型调用的成本和性能 |
| **工具调用** | `tool.{tool_name}` | `input_params`, `output_size`, `tool_status`, `latency_ms` | 追踪工具执行情况 |
| **推理循环** | `reasoning.cycle.{n}` | `cycle_number`, `thought`, `action_selected`, `confidence` | 记录ReAct循环的每一步决策 |
| **记忆读写** | `memory.{read/write}` | `memory_type`, `memory_size`, `relevance_score` | 追踪上下文管理 |

### 4.2 LLM调用Span设计详解

LLM调用是Agent中最昂贵的操作，必须精细追踪：

```
Span: llm.openai.gpt-4-turbo
├── Attributes:
│   ├── llm.provider: "openai"
│   ├── llm.model: "gpt-4-turbo"
│   ├── llm.input_tokens: 2048
│   ├── llm.output_tokens: 512
│   ├── llm.temperature: 0.7
│   ├── llm.cost_usd: 0.03584
│   ├── llm.latency_ms: 2340
│   ├── llm.status: "success"
│   └── agent.task_id: "task_abc123"
│
├── Events:
│   ├── "llm.request_sent" (timestamp, prompt_hash)
│   ├── "llm.first_token" (timestamp, ttft_ms)
│   └── "llm.response_complete" (timestamp)
│
└── Links:
    └── Span: reasoning.cycle.2 (关联到推理循环)
```

**关键设计决策**：

- **不记录完整的Prompt和Response**（体积太大，可能包含敏感信息），而是记录`prompt_hash`和`response_preview`
- **单独的`first_token`事件**：用于追踪首Token延迟（TTFT），这是用户体验的关键指标
- **成本直接计算为USD**，而非事后归因
- **错误状态的详细记录**：包括错误类型（速率限制、模型过载、网络超时）、重试次数和最终结果。这些信息对于识别模型提供商的稳定性问题至关重要

### 4.3 推理循环Span的嵌套设计

ReAct模式的Span嵌套是Agent Tracing中最具挑战性的部分：

```
Span: agent.react_loop (root)
│
├─ Span: reasoning.cycle.1
│   ├─ Span: llm.openai.gpt-4-turbo (thought: "需要查询销售数据")
│   ├─ Span: tool.sql_query (SELECT * FROM sales WHERE ...)
│   └─ Span: llm.openai.gpt-4-turbo (evaluation: "数据不够详细")
│
├─ Span: reasoning.cycle.2
│   ├─ Span: llm.openai.gpt-4-turbo (thought: "需要按地区分组")
│   ├─ Span: tool.sql_query (SELECT region, SUM(amount) ...)
│   └─ Span: llm.openai.gpt-4-turbo (evaluation: "数据充足")
│
└─ Span: reasoning.cycle.3
    └─ Span: llm.openai.gpt-4-turbo (final_answer: "综合分析报告")
```

### 4.4 Span属性命名规范

建立统一的属性命名规范至关重要：

```
命名规则: {domain}.{entity}.{attribute}

领域(domain):
  llm      → 模型相关
  tool     → 工具相关
  agent    → Agent框架相关
  memory   → 记忆系统相关
  cost     → 成本相关

示例:
  llm.input_tokens        → 模型输入token数
  llm.output_tokens       → 模型输出token数
  llm.cost_usd            → 模型调用成本(美元)
  tool.name               → 工具名称
  tool.execution_time_ms  → 工具执行时间
  agent.task_type         → 任务类型
  agent.user_id           → 用户ID
  memory.type             → 记忆类型(short_term/long_term)
  cost.total              → 任务总成本
```

---

## §5 成本追踪：Token消耗归因

成本追踪是Agent可观测性中最被低估、也最有价值的部分。

### 5.1 成本归因模型

```
总成本 = Σ (每次LLM调用的成本)
单次成本 = input_tokens × input_price + output_tokens × output_price

归因维度:
├─ 按任务归因: 哪些任务最贵？
├─ 按用户归因: 哪些用户消耗最多？
├─ 按Agent类型归因: 哪类Agent最耗资源？
├─ 按模型归因: 不同模型的成本分布？
└─ 按步骤归因: 成本花在推理还是工具调用？
```

### 5.2 实时成本计算核心逻辑

```python
from opentelemetry import trace
from dataclasses import dataclass

@dataclass
class ModelPricing:
    """模型定价配置"""
    input_price_per_1k: float   # 每1000 input tokens的价格(USD)
    output_price_per_1k: float  # 每1000 output tokens的价格(USD)

MODEL_PRICING = {
    "gpt-4-turbo": ModelPricing(0.01, 0.03),
    "gpt-3.5-turbo": ModelPricing(0.0005, 0.0015),
    "claude-3-opus": ModelPricing(0.015, 0.075),
    "claude-3-sonnet": ModelPricing(0.003, 0.015),
}

tracer = trace.get_tracer("agent.cost-tracker")

def track_llm_call(model: str, input_tokens: int, output_tokens: int,
                   task_id: str, user_id: str):
    """记录LLM调用并计算成本"""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return

    cost = (input_tokens / 1000 * pricing.input_price_per_1k +
            output_tokens / 1000 * pricing.output_price_per_1k)

    with tracer.start_as_current_span(f"llm.{model}") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        span.set_attribute("cost.usd", round(cost, 6))
        span.set_attribute("agent.task_id", task_id)
        span.set_attribute("agent.user_id", user_id)

        # 同时更新Metrics
        cost_counter.add(cost, {
            "model": model,
            "task_id": task_id,
            "user_id": user_id,
        })
```

### 5.3 多维度成本分析表

| 分析维度 | 典型发现 | 优化方向 |
|---------|---------|---------|
| **按模型** | 90%成本来自GPT-4调用 | 对简单任务使用GPT-3.5-Turbo |
| **按任务步骤** | 60%成本在推理循环中 | 优化Prompt减少推理轮次 |
| **按用户** | 5%用户消耗40%成本 | 引入用户级别的Token配额 |
| **按时段** | 凌晨批量任务成本飙升 | 合并批量任务，使用更便宜的模型 |
| **按Agent类型** | 数据分析Agent成本是问答Agent的10倍 | 调整不同Agent的成本预算 |

### 5.4 成本预算与熔断

在生产环境中，必须实现成本熔断机制。这类似于微服务中的熔断器模式，但触发条件不是错误率，而是成本消耗速度。一个典型的场景是：某个Agent任务因为推理循环无法收敛，持续调用大模型，单个任务的成本从预期的几美分飙升到几美元甚至几十美元。如果没有成本熔断机制，这种"烧钱死循环"可能在管理员发现之前已经造成了显著的财务损失。

```python
class CostCircuitBreaker:
    """基于成本的熔断器"""

    def __init__(self, max_cost_per_task: float = 0.50,
                 max_cost_per_user_daily: float = 5.00):
        self.max_cost_per_task = max_cost_per_task
        self.max_cost_per_user_daily = max_cost_per_user_daily

    def check_budget(self, task_id: str, user_id: str,
                     current_cost: float, daily_cost: float) -> bool:
        """检查是否超出预算，返回True表示允许继续"""
        if current_cost > self.max_cost_per_task:
            logger.warning(f"Task {task_id} exceeded per-task budget: "
                         f"${current_cost:.4f} > ${self.max_cost_per_task}")
            return False

        if daily_cost > self.max_cost_per_user_daily:
            logger.warning(f"User {user_id} exceeded daily budget: "
                         f"${daily_cost:.4f} > ${self.max_cost_per_user_daily}")
            return False

        return True
```

---

## §6 告警设计

### 6.1 四类核心告警

| 告警类型 | 触发条件 | 严重等级 | 响应动作 |
|---------|---------|---------|---------|
| **延迟告警** | P95延迟 > 30s（可配置） | Warning/Critical | 检查模型API响应、排查慢查询 |
| **错误率告警** | 5分钟窗口错误率 > 5% | Critical | 自动切换备用模型、通知OnCall |
| **成本异常告警** | 单任务成本 > $0.5 或 日成本异常增长200% | Warning | 检查是否有Agent循环、暂停高消耗任务 |
| **质量退化告警** | 任务成功率 < 85% 或 用户投诉率上升 | Warning/Critical | 回滚最近的Prompt变更、检查模型版本 |

### 6.2 告警规则设计

基于Grafana Alerting的告警规则示例：

```yaml
# 延迟告警
groups:
  - name: agent-latency
    rules:
      - alert: AgentHighLatency
        expr: |
          histogram_quantile(0.95,
            rate(agent_task_duration_seconds_bucket[5m])
          ) > 30
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Agent任务P95延迟超过30秒"
          description: "当前P95延迟: {{ $value }}s"

  # 成本异常告警
  - name: agent-cost
    rules:
      - alert: AgentCostSpike
        expr: |
          sum(rate(agent_cost_usd_total[1h])) by (user_id)
          > 2 * avg_over_time(
            sum(rate(agent_cost_usd_total[1h])) by (user_id)[7d:1h]
          )
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "用户 {{ $labels.user_id }} 成本异常飙升"
          description: "当前成本是7天均值的{{ $value }}倍"

  # 推理循环次数告警（检测Agent死循环）
  - name: agent-loop
    rules:
      - alert: AgentReasoningLoopDetected
        expr: |
          agent_reasoning_cycles > 15
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "检测到Agent推理循环异常"
          description: "任务 {{ $labels.task_id }} 已执行 {{ $value }} 轮推理循环"
```

### 6.3 告警降噪策略

Agent系统天然会产生大量告警噪声。关键降噪策略：

1. **窗口聚合**：不基于单次请求告警，基于5分钟/1小时窗口聚合
2. **基线自适应**：用7天滑动平均值作为动态基线，而非固定阈值
3. **分层告警**：Warning（人工关注）→ Critical（立即响应）→ Emergency（自动熔断）
4. **关联抑制**：如果模型API整体超时，抑制所有该模型的下游告警

---

## §7 实战：Grafana Dashboard搭建

### 7.1 Dashboard架构

一个完整的Agent监控Dashboard应包含四个区域：

```
┌──────────────────────────────────────────────────────────┐
│  Row 1: 全局概览                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ 当前QPS  │ │ 平均延迟  │ │ 错误率   │ │ 日成本   │     │
│  │   12.3   │ │   8.2s   │ │  2.1%    │ │ $127.50  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
├──────────────────────────────────────────────────────────┤
│  Row 2: 成本分析                                          │
│  ┌──────────────────────┐ ┌──────────────────────┐       │
│  │ 成本趋势图(7天)       │ │ 按模型成本饼图        │       │
│  │ [Time Series]        │ │ [Pie Chart]          │       │
│  └──────────────────────┘ └──────────────────────┘       │
├──────────────────────────────────────────────────────────┤
│  Row 3: 性能分析                                          │
│  ┌──────────────────────┐ ┌──────────────────────┐       │
│  │ 延迟分布(直方图)       │ │ 推理循环次数分布      │       │
│  │ [Heatmap]            │ │ [Histogram]          │       │
│  └──────────────────────┘ └──────────────────────┘       │
├──────────────────────────────────────────────────────────┤
│  Row 4: Trace详情                                        │
│  ┌──────────────────────────────────────────────┐        │
│  │ 最近Trace列表 + Trace对比视图                  │        │
│  │ [Tempo Trace Panel]                          │        │
│  └──────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### 7.2 关键Grafana查询（PromQL）

**全局概览面板**：

```promql
# 当前QPS
sum(rate(agent_task_total[5m]))

# 平均延迟
histogram_quantile(0.50, sum(rate(agent_task_duration_seconds_bucket[5m])) by (le))

# P95延迟
histogram_quantile(0.95, sum(rate(agent_task_duration_seconds_bucket[5m])) by (le))

# 错误率
sum(rate(agent_task_total{status="error"}[5m])) / sum(rate(agent_task_total[5m]))
```

**成本分析面板**：

```promql
# 按模型的成本分布
sum(rate(agent_llm_cost_usd_total[1h])) by (llm_model)

# 每用户日成本
sum(increase(agent_llm_cost_usd_total[24h])) by (agent_user_id)
  topk(10)

# 成本趋势（与昨天对比）
sum(rate(agent_llm_cost_usd_total[1h]))
  /
  sum(rate(agent_llm_cost_usd_total[1h] offset 24h))
```

**推理效率面板**：

```promql
# 平均推理循环次数
histogram_quantile(0.50,
  sum(rate(agent_reasoning_cycles_bucket[1h])) by (le))

# 工具调用成功率
sum(rate(agent_tool_call_total{status="success"}[1h])) by (tool_name)
  /
  sum(rate(agent_tool_call_total[1h])) by (tool_name)
```

### 7.3 Trace查询示例

在Grafana Tempo中，使用TraceQL查询Agent特定的Trace：

```
# 查找所有耗时超过30秒的Agent任务
{ span.agent.task_type = "data_analysis" } | duration > 30s

# 查找成本超过$0.1的单次任务
{ span.cost.usd > 0.1 }

# 查找有LLM调用错误的Trace
{ span.llm.status = "error" }

# 查找推理循环超过10轮的复杂任务
{ span.agent.reasoning_cycles > 10 }
```

### 7.4 Log-Trace关联查询

在Loki中通过`trace_id`关联日志和Trace：

```
# 查找特定任务的所有日志
{job="agent"} | json | trace_id = "abc123def456"

# 查找所有包含"error"的日志并跳转到对应Trace
{job="agent"} |~ "error" | json | line_format "{{.trace_id}}"
```

---

## §8 生产踩坑

### 8.1 采样策略

这是生产环境中最关键的决策之一：

| 采样策略 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| **全量采集** | 开发/测试环境 | 无遗漏 | 存储成本极高 |
| **概率采样** | 低流量服务 | 简单均匀 | 可能遗漏关键Trace |
| **自适应采样** | 生产环境推荐 | 智能平衡 | 实现复杂 |
| **尾部采样** | 高价值场景 | 保留异常/慢请求 | 增加延迟 |

**推荐策略：自适应采样 + 尾部采样组合**

```
OTel Collector Pipeline:
1. Probabilistic Sampler: 基础10%采样
2. Tail Sampler:
   - 错误请求: 100%保留
   - P99延迟请求: 100%保留
   - 高成本请求(>$0.1): 100%保留
   - 其余: 按概率采样
```

**关键配置**：

```yaml
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 100000
    policies:
      # 保留所有错误
      - name: error-policy
        type: status_code
        status_code: {status_codes: [ERROR]}
      # 保留高延迟
      - name: latency-policy
        type: latency
        latency: {threshold_ms: 15000}
      # 保留高成本
      - name: cost-policy
        type: string_attribute
        string_attribute: {key: cost.usd, values: ["0.1"]}
      # 其余10%概率采样
      - name: probabilistic-policy
        type: probabilistic
        probabilistic: {sampling_percentage: 10}
```

### 8.2 存储成本控制

Tracing数据是海量的。一个典型的成本估算：

| 场景 | 每日Trace数 | 平均Span数/Trace | 每Span大小 | 日存储量 |
|------|-----------|-------------------|-----------|---------|
| 小规模(100用户) | 10,000 | 20 | 2KB | ~400MB |
| 中规模(1000用户) | 100,000 | 25 | 2KB | ~5GB |
| 大规模(10000用户) | 1,000,000 | 30 | 2KB | ~60GB |

**成本控制手段**：

1. **采样**：前面讨论的自适应采样，可将存储量降低到1/10
2. **压缩**：Tempo使用压缩存储，实际磁盘用量远小于原始数据
3. **保留策略**：热数据7天（SSD），温数据30天（HDD），冷数据90天（S3）
4. **Span裁剪**：对超大Span（如包含完整Prompt的），截断到合理大小
5. **属性过滤**：只采集有价值的Attributes，过滤掉debug级别的属性

### 8.3 性能开销

在Agent中嵌入Tracing需要注意性能影响：

| 开销来源 | 影响程度 | 优化方案 |
|---------|---------|---------|
| Span创建 | 低（~1μs） | 几乎无影响 |
| 数据序列化 | 中（~50μs/Span） | 批量序列化，异步执行 |
| 网络传输 | 中（~100μs） | 本地缓冲+批量发送 |
| SDK初始化 | 低（~10ms） | 启动时一次性初始化 |

**最佳实践**：

- 使用**异步导出**（OTel SDK默认支持），避免阻塞主业务逻辑
- 设置合理的**批量大小和超时**（batch size: 1024, timeout: 5s）
- 对于高频Span（如每秒>1000次），考虑**客户端预聚合**
- 设置**内存队列上限**，避免OOM（推荐最大队列大小: 2048）

### 8.4 多租户隔离

在SaaS场景下，Agent平台通常需要支持多租户：

```
租户隔离架构:
├─ 数据隔离
│   ├─ OTel Collector: 按tenant_id路由到不同的后端
│   ├─ Tempo: 使用租户ID作为namespace
│   └─ Loki: 使用租户ID作为标签
│
├─ 查询隔离
│   ├─ Grafana: 按租户限制Dashboard可见性
│   └─ API: 按租户限制Trace/Log查询范围
│
└─ 成本隔离
    ├─ 按租户聚合Metrics
    └─ 按租户设置成本配额和告警阈值
```

**OTel Collector多租户路由**：

```yaml
processors:
  # 根据资源属性中的tenant_id进行路由
  routing:
    table:
      - target: otlp/tempo-tenant-a
        expression: attributes["tenant.id"] == "tenant_a"
      - target: otlp/tempo-default
        expression: "true"  # 默认路由
```

### 8.5 网络与安全考量

在生产环境中，Tracing数据可能包含敏感信息（如用户查询内容、工具调用参数）。必须在数据流的早期阶段进行脱敏处理。推荐的做法是在OTel Collector的Processor层添加`transform`处理器，对特定属性进行正则替换或哈希处理。例如，将用户查询中的手机号、邮箱等敏感信息替换为占位符。同时，Collector与后端存储之间的通信必须使用TLS加密，特别是在跨可用区部署的场景下。此外，Trace数据的访问需要严格的权限控制，确保不同团队只能查看自己负责的Agent数据。

---

## §9 面试深度：如何设计一个支持千级Agent的可观测性平台

这是面试中可能会遇到的系统设计题。以下是完整的思路和要点。

### 9.1 需求分析

**规模**：支持1000+个Agent实例同时运行，每个Agent每秒产生5-20个Span。

**数据量估算**：

| 指标 | 数值 |
|------|------|
| Agent实例数 | 1,000 |
| 平均Span产生速率 | 10/s/Agent |
| 总Span速率 | 10,000/s |
| 每Span平均大小 | 2KB |
| 原始数据吞吐 | 20MB/s ≈ 1.7TB/天 |

**核心挑战**：

1. **高吞吐写入**：10K spans/s，峰值可能达到50K spans/s
2. **低延迟查询**：P99查询延迟 < 3s
3. **成本可控**：月存储成本控制在合理范围内
4. **多租户隔离**：不同团队的Agent数据完全隔离

### 9.2 架构设计

```
┌───────────────────────────────────────────────────────────────┐
│                     千级Agent可观测性平台                       │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    数据采集层                              │  │
│  │                                                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ Agent #1 │  │ Agent #2 │  │ Agent #N │  ...          │  │
│  │  │ OTel SDK │  │ OTel SDK │  │ OTel SDK │              │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │  │
│  │       │              │              │                    │  │
│  └───────┼──────────────┼──────────────┼────────────────────┘  │
│          │              │              │                       │
│          ▼              ▼              ▼                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   OTel Collector 集群                    │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐           │  │
│  │  │ C1     │ │ C2     │ │ C3     │ │ C4     │           │  │
│  │  │(接收+  │ │(接收+  │ │(接收+  │ │(接收+  │           │  │
│  │  │ 采样)  │ │ 采样)  │ │ 采样)  │ │ 采样)  │           │  │
│  │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘           │  │
│  └──────┼──────────┼──────────┼──────────┼─────────────────┘  │
│         │          │          │          │                     │
│         ▼          ▼          ▼          ▼                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   数据处理层                               │  │
│  │                                                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ 采样处理  │  │ 路由分发  │  │ 数据压缩  │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └───────────┬───────────┬───────────┬──────────────────────┘  │
│              │           │           │                         │
│         ┌────┘     ┌─────┘     ┌─────┘                        │
│         ▼          ▼           ▼                               │
│  ┌────────────┐ ┌──────────┐ ┌────────────┐                   │
│  │   Tempo    │ │Prometheus│ │    Loki     │                   │
│  │ (Traces)   │ │(Metrics) │ │   (Logs)    │                   │
│  │ + S3后端   │ │ + Mimir  │ │  + S3后端   │                   │
│  └──────┬─────┘ └────┬─────┘ └──────┬─────┘                   │
│         │            │              │                          │
│         └──────┬─────┴──────┬───────┘                          │
│                ▼            ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    查询与展示层                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ Grafana  │  │ 自定义API │  │ 告警引擎  │              │  │
│  │  │Dashboard │  │ (成本查询)│  │ (Grafana  │              │  │
│  │  │          │  │          │  │  Alerting)│              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### 9.3 关键设计决策

#### 决策1：OTel Collector集群化

单个Collector在高吞吐下会成为瓶颈。需要：

- **水平扩展**：部署4-8个Collector实例，前面挂负载均衡
- **角色分离**：接收层（receivers）和处理层（processors）分开部署
- **内存队列**：设置合理的队列大小，避免OOM

#### 决策2：采样策略分层

```
第1层 - 客户端采样（SDK层）:
  - 基础10%概率采样
  - 100%保留错误请求

第2层 - Collector采样（服务端）:
  - 自适应采样：根据当前吞吐动态调整采样率
  - 尾部采样：保留高延迟、高成本的Trace

第3层 - 存储采样（后端）:
  - 按时间保留：热数据完整存储，冷数据降采样
```

#### 决策3：存储后端选型

| 组件 | 推荐 | 理由 |
|------|------|------|
| Traces | Grafana Tempo + S3 | 无索引设计，S3存储成本最低 |
| Metrics | Mimir + S3 | 支持多租户，高可用 |
| Logs | Loki + S3 | 轻量级，与Tempo原生关联 |

#### 决策4：成本优化

```
存储成本优化路径:
1. 采样（降10x）→ 2. 压缩（降3x）→ 3. 冷热分离（降5x）
综合效果: 原始1.7TB/天 → 实际存储 ~11GB/天（90天保留）
月存储成本: ~$300-500（S3标准存储）
```

### 9.4 面试回答模板

在面试中回答这类系统设计题时，建议按以下结构组织：

1. **明确需求和规模**：先确认数据量、查询模式、SLA要求
2. **画出架构图**：展示从采集到存储到展示的完整链路
3. **逐层讲解设计决策**：每一层为什么这样选，权衡了什么
4. **讨论Trade-off**：采样vs完整性、成本vs延迟、简单vs灵活
5. **提出扩展方案**：如果规模增长10倍，架构如何演进

**关键加分项**：

- 提到Agent场景的特殊性（非确定性、成本敏感、推理循环）
- 展示对OpenTelemetry生态的熟悉程度
- 能讨论具体的成本估算和优化手段
- 提到生产踩坑经验（采样策略、性能开销、多租户隔离）

---

## 总结

Agent可观测性不是一个"有了就行"的附加功能，而是Agent系统能否在生产环境中稳定运行的基石。许多团队在Agent开发阶段投入大量精力优化模型效果和Prompt质量，却忽视了可观测性的建设，等到系统上线后才发现"出了问题完全无法定位根因"。这种事后补救的成本远高于事前设计。

**回顾核心要点**：

| 主题 | 关键Takeaway |
|------|-------------|
| **独特挑战** | 非确定性、长链路、多模型调用、成本敏感 |
| **三大支柱** | Trace记录思考过程，Metrics追踪实时状态，Logs保存决策上下文 |
| **技术选型** | OTel + Tempo + Prometheus + Loki + Grafana |
| **Span设计** | LLM/工具/推理/记忆四种核心Span类型，统一属性命名 |
| **成本追踪** | 实时Token归因 + 多维度成本分析 + 预算熔断 |
| **告警设计** | 延迟/错误率/成本/质量四类告警 + 降噪策略 |
| **生产踩坑** | 自适应采样 + 冷热存储 + 异步导出 + 多租户隔离 |
| **面试深度** | 千级Agent平台的完整设计，展示Trade-off思维 |

> **给面试者的建议**：可观测性不是一个独立的面试方向，它与Agent架构、Prompt工程、模型选型都密切相关。在面试中主动提及相关性，展示你的系统性思维。

---

*本文是面试方向7（全链路Tracing）的深度补充文章，适合在掌握基础Tracing概念后深入学习。建议配合实际动手部署一个OTel + Grafana的最小环境来加深理解。*
