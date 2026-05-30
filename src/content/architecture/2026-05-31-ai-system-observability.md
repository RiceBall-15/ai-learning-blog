---
title: "AI系统可观测性架构：从日志收集到智能告警的全链路监控实战"
description: "深入解析AI应用可观测性架构设计，涵盖LLM调用追踪、Token成本监控、质量评估与智能告警，构建生产级AI系统的监控体系"
date: 2026-05-31
author: "RiceBall-15"
category: "architecture"
subCategory: "distributed"
tags: ["可观测性", "AI监控", "LLM追踪", "智能告警", "系统架构", "MLOps"]
draft: false
---

# AI系统可观测性架构：从日志收集到智能告警的全链路监控实战

## 一、引言：AI系统为什么需要全新的可观测性？

### 1.1 传统监控的失效

传统的可观测性三支柱——Metrics（指标）、Logs（日志）、Traces（追踪）——在面对AI系统时出现了严重的适用性缺口。原因在于AI系统引入了几个传统系统中不存在的维度：

```
┌─────────────────────────────────────────────────────────────────────┐
│              传统系统 vs AI系统的可观测性需求差异                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  传统Web系统                    AI/LLM系统                           │
│  ┌──────────────────┐          ┌──────────────────┐                │
│  │ HTTP状态码        │          │ 语义质量评估      │                │
│  │ 响应延迟 (ms)     │          │ 输出质量评分      │                │
│  │ CPU/内存/磁盘     │          │ Token消耗/成本    │                │
│  │ 错误率/成功率     │          │ 幻觉率/准确率     │                │
│  │ QPS/吞吐量        │          │ 并发会话/上下文   │                │
│  └──────────────────┘          └──────────────────┘                │
│                                                                      │
│  AI系统特有的观测维度:                                                 │
│  ├── Token级成本追踪 (每次调用的Token用量与费用)                       │
│  ├── 输出质量评估 (相关性、准确性、幻觉检测)                           │
│  ├── 模型版本与Prompt版本追踪                                         │
│  ├── 用户满意度信号 (点赞/踩、编辑、重试)                              │
│  ├── 上下文窗口利用率                                                 │
│  └── 多模型路由决策追踪                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 一个真实场景

假设你的AI客服系统在某个下午突然出现用户投诉激增。传统监控可能告诉你：API响应正常、错误率为零、服务器负载正常。但实际上，LLM的输出质量正在下降——模型开始生成不相关或错误的回答，而这一切在传统的HTTP 200状态码下完全不可见。

这就是AI系统可观测性的核心挑战：**传统监控关注的是"系统是否正常运行"，而AI系统需要回答的是"系统的输出质量是否达标"。**

## 二、AI系统可观测性架构全景

### 2.1 架构总览

一个完整的AI系统可观测性架构应该包含以下层次：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AI系统可观测性架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   数据采集层  │  │   存储聚合层  │  │   分析引擎层  │  │   展示告警层  │  │
│  │             │  │             │  │             │  │             │  │
│  │ • SDK埋点   │  │ • 时序数据库 │  │ • 实时流处理 │  │ • 可视化面板 │  │
│  │ • 日志收集  │  │ • 日志聚合  │  │ • 批量分析  │  │ • 智能告警  │  │
│  │ • 链路追踪  │  │ • 文档索引  │  │ • ML异常检测 │  │ • 根因分析  │  │
│  │ • 事件采集  │  │ • 向量存储  │  │ • 质量评估  │  │ • 自动报告  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        监控对象层                                  │   │
│  │                                                                 │   │
│  │  LLM调用链    Prompt管理    模型版本    用户交互    成本追踪      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件选型对比

| 组件类型 | 推荐方案 | 适用场景 | 注意事项 |
|---------|---------|---------|---------|
| **日志收集** | Loki + Promtail | 轻量级日志聚合 | 不适合全文检索场景 |
| **链路追踪** | Jaeger / Tempo | LLM调用链追踪 | 注意采样率对成本的影响 |
| **指标存储** | Prometheus + Mimir | 时序指标监控 | 长期存储需要Mimir或Thanos |
| **异常检测** | 自建ML模型 / Prophet | 输出质量异常检测 | 需要足够的历史数据 |
| **告警引擎** | AlertManager / PagerDuty | 多渠道告警 | 注意告警疲劳问题 |
| **成本分析** | 自建仪表板 | Token成本追踪与优化 | 需要与计费系统集成 |

## 三、LLM调用链追踪：最核心的监控维度

### 3.1 设计LLM追踪的数据模型

LLM调用追踪需要捕获的信息远比传统API调用丰富：

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

@dataclass
class LLMTrace:
    """LLM调用追踪数据模型"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: Optional[str] = None
    
    # 调用信息
    model: str = ""                    # 模型名称，如 "gpt-4o"
    model_version: str = ""            # 模型版本
    prompt_version: str = ""           # Prompt版本标识
    
    # 输入信息
    messages: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # 输出信息
    response: str = ""
    finish_reason: str = ""            # stop / length / content_filter
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Token统计
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # 成本信息
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    
    # 质量评估（异步填充）
    quality_score: Optional[float] = None      # 0-1
    relevance_score: Optional[float] = None    # 0-1
    hallucination_detected: Optional[bool] = None
    
    # 用户反馈
    user_feedback: Optional[str] = None        # positive / negative / null
    user_edited_response: bool = False
    user_retried: bool = False
    
    # 元数据
    session_id: str = ""
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMTraceAggregation:
    """LLM调用聚合统计（按时间段/模型/Prompt版本聚合）"""
    period: str = ""              # 时间段标识
    model: str = ""
    prompt_version: str = ""
    
    total_calls: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    total_tokens: int = 0
    avg_tokens_per_call: float = 0.0
    total_cost_usd: float = 0.0
    avg_cost_per_call: float = 0.0
    
    avg_quality_score: float = 0.0
    hallucination_rate: float = 0.0
    user_satisfaction_rate: float = 0.0
    retry_rate: float = 0.0
```

### 3.2 轻量级追踪中间件实现

```python
import time
import json
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger("llm_tracer")

class LLMTracer:
    """轻量级LLM追踪器"""
    
    def __init__(self, export_fn: Callable[[LLMTrace], None] = None):
        self.export_fn = export_fn or self._default_export
        self._local = __import__('threading').local()
    
    def trace(self, func: Callable) -> Callable:
        """装饰器：自动追踪LLM调用"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace = LLMTrace()
            start_time = time.perf_counter()
            
            try:
                # 提取输入信息
                trace.messages = kwargs.get('messages', [])
                trace.model = kwargs.get('model', 'unknown')
                trace.temperature = kwargs.get('temperature', 0.7)
                
                # 调用原始函数
                result = func(*args, **kwargs)
                
                # 提取输出信息
                trace.response = result.get('content', '')
                trace.finish_reason = result.get('finish_reason', 'unknown')
                trace.tool_calls = result.get('tool_calls', [])
                
                # Token统计
                usage = result.get('usage', {})
                trace.prompt_tokens = usage.get('prompt_tokens', 0)
                trace.completion_tokens = usage.get('completion_tokens', 0)
                trace.total_tokens = usage.get('total_tokens', 0)
                
                # 计算延迟
                trace.latency_ms = (time.perf_counter() - start_time) * 1000
                
                # 异步导出
                self.export_fn(trace)
                
                return result
                
            except Exception as e:
                trace.tags['error'] = str(e)
                trace.finish_reason = 'error'
                trace.latency_ms = (time.perf_counter() - start_time) * 1000
                self.export_fn(trace)
                raise
        
        return wrapper
    
    def _default_export(self, trace: LLMTrace):
        """默认导出：写入日志（生产环境应替换为向量数据库或消息队列）"""
        logger.info(json.dumps({
            'trace_id': trace.trace_id,
            'model': trace.model,
            'tokens': trace.total_tokens,
            'latency_ms': round(trace.latency_ms, 2),
            'cost_usd': trace.cost_usd,
            'quality_score': trace.quality_score,
        }, ensure_ascii=False))
```

### 3.3 实际使用示例

```python
tracer = LLMTracer()

@tracer.trace
def call_llm(messages, model="gpt-4o", temperature=0.7):
    """封装LLM调用"""
    # 这里是实际的LLM API调用
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return {
        'content': response.choices[0].message.content,
        'finish_reason': response.choices[0].finish_reason,
        'tool_calls': response.choices[0].message.tool_calls or [],
        'usage': {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }
    }

# 使用示例
result = call_llm(
    messages=[{"role": "user", "content": "什么是RAG？"}],
    model="gpt-4o",
    temperature=0.7
)
```

## 四、成本监控与优化

### 4.1 Token成本追踪架构

Token成本是AI系统运营中最容易被忽视但增长最快的费用项。一个完善的成本监控系统需要：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Token成本追踪架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐ │
│  │   实时成本流   │────▶│  成本聚合引擎  │────▶│  成本仪表板   │ │
│  │               │     │               │     │               │ │
│  │ • 每次调用的  │     │ • 按模型聚合  │     │ • 日/周/月   │ │
│  │   Token单价  │     │ • 按功能聚合  │     │   成本趋势   │ │
│  │ • 按需定价   │     │ • 按用户聚合  │     │ • 成本预警   │ │
│  │ • 批量折扣   │     │ • 按Prompt版本│     │ • 优化建议   │ │
│  └───────────────┘     └───────────────┘     └───────────────┘ │
│                                                                  │
│  模型定价参考 (截至2026年5月):                                    │
│  ┌──────────────┬────────────┬────────────┬──────────────────┐ │
│  │ 模型          │ 输入/1M     │ 输出/1M     │ 备注              │ │
│  ├──────────────┼────────────┼────────────┼──────────────────┤ │
│  │ GPT-4o       │ $2.50      │ $10.00     │ 主力模型           │ │
│  │ GPT-4o-mini  │ $0.15      │ $0.60      │ 简单任务           │ │
│  │ Claude 3.5   │ $3.00      │ $15.00     │ 长文本场景         │ │
│  │ DeepSeek-V3  │ $0.27      │ $1.10      │ 性价比之选         │ │
│  │ Qwen-Max     │ $1.60      │ $6.40      │ 中文场景           │ │
│  └──────────────┴────────────┴────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 成本优化策略

**策略一：模型路由（Model Routing）**

根据任务复杂度自动选择合适的模型，而非所有任务都用最强模型：

```python
class ModelRouter:
    """基于任务复杂度的智能模型路由"""
    
    MODEL_TIERS = {
        'simple': 'gpt-4o-mini',      # 简单查询、格式转换
        'medium': 'gpt-4o',           # 一般对话、信息提取
        'complex': 'claude-3.5-sonnet', # 复杂推理、代码生成
        'critical': 'gpt-4o',         # 关键业务决策
    }
    
    def route(self, task_type: str, context: dict) -> str:
        """根据任务类型和上下文选择模型"""
        
        # 简单任务：短输入、不需要复杂推理
        if task_type == 'classification' and context.get('input_length', 0) < 200:
            return self.MODEL_TIERS['simple']
        
        # 中等任务：一般对话、信息提取
        if task_type in ['summarize', 'extract', 'chat']:
            return self.MODEL_TIERS['medium']
        
        # 复杂任务：需要深度推理
        if task_type in ['code_generation', 'analysis', 'planning']:
            return self.MODEL_TIERS['complex']
        
        # 关键决策：不允许出错
        if task_type == 'decision':
            return self.MODEL_TIERS['critical']
        
        # 默认：中等模型
        return self.MODEL_TIERS['medium']
    
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """估算调用成本"""
        PRICING = {
            'gpt-4o-mini': (0.15 / 1_000_000, 0.60 / 1_000_000),
            'gpt-4o': (2.50 / 1_000_000, 10.00 / 1_000_000),
            'claude-3.5-sonnet': (3.00 / 1_000_000, 15.00 / 1_000_000),
        }
        input_price, output_price = PRICING.get(model, (2.50 / 1_000_000, 10.00 / 1_000_000))
        return prompt_tokens * input_price + completion_tokens * output_price
```

**策略二：Prompt缓存（Prompt Caching）**

对于重复性的系统提示和少量示例，利用缓存减少重复输入Token：

```python
import hashlib
from typing import Optional
from datetime import datetime, timedelta

class PromptCache:
    """Prompt缓存管理器"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _hash_prompt(self, messages: list, model: str) -> str:
        """生成Prompt哈希"""
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, messages: list, model: str) -> Optional[str]:
        """获取缓存的响应"""
        key = self._hash_prompt(messages, model)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.utcnow() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[key]
        return None
    
    def set(self, messages: list, model: str, response: str):
        """设置缓存"""
        key = self._hash_prompt(messages, model)
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.utcnow(),
            'hits': 0
        }
```

**策略三：输出压缩与截断**

对于不需要完整输出的场景，设置合理的 `max_tokens` 避免浪费：

```python
# 不同场景的Token配置
TOKEN_CONFIG = {
    'classification': {'max_tokens': 50, 'temperature': 0},
    'extraction': {'max_tokens': 500, 'temperature': 0},
    'summarization': {'max_tokens': 1000, 'temperature': 0.3},
    'chat': {'max_tokens': 2000, 'temperature': 0.7},
    'code_generation': {'max_tokens': 4000, 'temperature': 0.2},
    'creative_writing': {'max_tokens': 4000, 'temperature': 0.9},
}
```

## 五、输出质量评估

### 5.1 自动化质量评估架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      输出质量评估流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ LLM输出  │───▶│ 规则检查  │───▶│ LLM评估  │───▶│ 人工抽检  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │              │               │               │         │
│       ▼              ▼               ▼               ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ 原始响应  │    │ 格式验证  │    │ 相关性   │    │ 标注质量  │ │
│  │          │    │ 长度检查  │    │ 准确性   │    │ 反馈收集  │ │
│  │          │    │ 敏感词   │    │ 幻觉检测  │    │ 模型改进  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                  │
│  评估策略:                                                        │
│  • 规则检查: 实时（<10ms）                                         │
│  • LLM评估: 异步采样（10-20%的调用）                               │
│  • 人工抽检: 每日/每周批量进行                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 LLM-as-Judge 实现

使用LLM评估LLM的输出，是一种成本可控且效果不错的质量评估方式：

```python
class OutputQualityEvaluator:
    """LLM输出质量评估器"""
    
    EVALUATION_PROMPT = """你是一个严格的AI输出质量评估专家。请评估以下AI系统的输出质量。

## 用户输入
{user_input}

## AI输出
{ai_output}

## 评估维度
请从以下4个维度进行评分（1-5分）：

1. **相关性** (Relevance): 输出是否直接回答了用户的问题？
2. **准确性** (Accuracy): 输出中的信息是否正确？
3. **完整性** (Completeness): 输出是否涵盖了必要的信息？
4. **安全性** (Safety): 输出是否包含有害、偏见或不当内容？

## 输出格式
请严格按照JSON格式输出：
{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "safety": <1-5>,
    "overall_score": <1-5>,
    "issues": ["问题描述1", "问题描述2"],
    "suggestion": "改进建议"
}
"""
    
    async def evaluate(
        self, 
        user_input: str, 
        ai_output: str,
        sample_rate: float = 0.1  # 10%采样评估
    ) -> Optional[Dict]:
        """评估AI输出质量"""
        
        # 采样控制：不是每次调用都评估
        import random
        if random.random() > sample_rate:
            return None
        
        prompt = self.EVALUATION_PROMPT.format(
            user_input=user_input[:1000],  # 截断过长输入
            ai_output=ai_output[:2000]     # 截断过长输出
        )
        
        # 使用轻量级模型进行评估
        result = await call_llm(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",  # 用便宜模型做评估
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        try:
            evaluation = json.loads(result['content'])
            return evaluation
        except json.JSONDecodeError:
            return None
```

## 六、智能告警设计

### 6.1 告警策略分层

AI系统的告警需要分层设计，避免告警疲劳：

```
┌─────────────────────────────────────────────────────────────────┐
│                        告警策略分层                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  P0 - 紧急 (立即响应)                                            │
│  ├── LLM API完全不可用                                           │
│  ├── 幻觉率突增超过20%                                            │
│  └── 成本异常飙升（日成本超过预算300%）                             │
│                                                                  │
│  P1 - 严重 (15分钟内响应)                                         │
│  ├── 响应延迟P99超过30秒                                          │
│  ├── 成功率降至90%以下                                            │
│  └── 质量评分低于3.0（5分制）                                      │
│                                                                  │
│  P2 - 警告 (1小时内响应)                                          │
│  ├── Token消耗异常增长（偏离基线50%）                               │
│  ├── 用户重试率超过10%                                            │
│  └── 新模型版本质量回归                                            │
│                                                                  │
│  P3 - 通知 (工作时间响应)                                         │
│  ├── 成本趋势异常                                                 │
│  ├── 质量评分缓慢下降                                              │
│  └── 缓存命中率低于预期                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 告警规则配置示例

```python
ALERT_RULES = {
    # P0: 紧急告警
    "llm_api_down": {
        "condition": "success_rate < 0.5 AND duration > 5min",
        "severity": "P0",
        "channels": ["pagerduty", "slack_urgent", "phone_call"],
        "cooldown": "5min",
        "description": "LLM API可用性严重下降"
    },
    
    "hallucination_spike": {
        "condition": "hallucination_rate > 0.2 AND sample_size > 50",
        "severity": "P0",
        "channels": ["pagerduty", "slack_urgent"],
        "cooldown": "10min",
        "description": "幻觉率异常飙升"
    },
    
    # P1: 严重告警
    "high_latency": {
        "condition": "p99_latency > 30000 AND duration > 3min",
        "severity": "P1",
        "channels": ["slack_alert", "email"],
        "cooldown": "15min",
        "description": "P99延迟超过30秒"
    },
    
    "low_success_rate": {
        "condition": "success_rate < 0.9 AND duration > 5min",
        "severity": "P1",
        "channels": ["slack_alert", "email"],
        "cooldown": "10min",
        "description": "成功率降至90%以下"
    },
    
    # P2: 警告
    "token_cost_spike": {
        "condition": "daily_cost > 3 * avg_daily_cost",
        "severity": "P2",
        "channels": ["slack_info"],
        "cooldown": "1h",
        "description": "日Token成本异常增长"
    },
    
    "quality_degradation": {
        "condition": "avg_quality_score < 3.0 AND sample_size > 100",
        "severity": "P2",
        "channels": ["slack_info", "email"],
        "cooldown": "2h",
        "description": "输出质量评分持续偏低"
    },
}
```

### 6.3 基于异常检测的智能告警

传统阈值告警在AI系统中容易产生大量误报。基于统计的异常检测能更准确地发现问题：

```python
import numpy as np
from collections import deque

class AnomalyDetector:
    """基于统计的AI系统异常检测"""
    
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold  # 标准差倍数
        self.history = deque(maxlen=window_size)
    
    def update(self, value: float) -> bool:
        """更新数据点并检测是否异常"""
        self.history.append(value)
        
        if len(self.history) < 20:  # 数据不足，不报警
            return False
        
        values = np.array(self.history)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:  # 没有波动，任何偏离都是异常
            return value != mean
        
        z_score = abs(value - mean) / std
        return z_score > self.threshold
    
    def get_baseline(self) -> dict:
        """获取当前基线统计"""
        if len(self.history) < 10:
            return {"status": "insufficient_data"}
        
        values = np.array(self.history)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
```

## 七、部署与运维实践

### 7.1 生产环境部署架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   生产环境可观测性部署架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AI应用服务                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Tracer   │  │ Metrics  │  │ Logger   │            │   │
│  │  │ SDK      │  │ SDK      │  │ SDK      │            │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │   │
│  └───────┼──────────────┼──────────────┼──────────────────┘   │
│          │              │              │                       │
│          ▼              ▼              ▼                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │ OTel     │    │Prometheus│    │ Promtail │               │
│  │ Collector│    │  Agent   │    │  Agent   │               │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘               │
│       │               │               │                       │
│       ▼               ▼               ▼                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  Tempo   │    │ Mimir    │    │  Loki    │               │
│  │ (Traces) │    │(Metrics) │    │  (Logs)  │               │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘               │
│       │               │               │                       │
│       └───────────────┼───────────────┘                       │
│                       ▼                                       │
│              ┌──────────────────┐                             │
│              │     Grafana      │                             │
│              │   (统一面板)      │                             │
│              └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 关键运维指标

| 指标类别 | 具体指标 | 告警阈值 | 监控频率 |
|---------|---------|---------|---------|
| **可用性** | LLM调用成功率 | < 95% | 实时 |
| **性能** | P95延迟 | > 15s | 实时 |
| **性能** | P99延迟 | > 30s | 实时 |
| **成本** | 日均Token消耗 | > 基线200% | 每小时 |
| **质量** | 输出质量评分 | < 3.0/5.0 | 每小时 |
| **质量** | 幻觉检测率 | > 10% | 每日 |
| **用户** | 重试率 | > 15% | 实时 |
| **用户** | 负面反馈率 | > 5% | 每小时 |

## 八、总结与最佳实践

### 8.1 渐进式建设路径

对于资源有限的团队，建议按以下优先级逐步建设可观测性：

1. **第一阶段：基础监控（1-2周）**
   - 部署LLM调用追踪（记录每次调用的延迟、Token、成本）
   - 配置基本告警（可用性、延迟）
   - 搭建简单的Grafana仪表板

2. **第二阶段：质量监控（2-4周）**
   - 实现用户反馈收集（点赞/踩）
   - 接入LLM-as-Judge质量评估
   - 建立质量基线和趋势分析

3. **第三阶段：智能优化（4-8周）**
   - 实现模型路由和成本优化
   - 部署异常检测模型
   - 建立A/B测试和灰度发布机制

### 8.2 核心原则

> **可观测性不是成本中心，而是AI系统的质量保障和成本优化的基础设施。** 一个完善的监控体系能帮你发现问题、优化成本、提升质量，最终让AI系统在生产环境中稳定运行。

记住：**你无法优化你看不到的东西。** 在AI系统中，可观测性不是锦上添花，而是雪中送炭。
