---
title: "LLM网关架构设计：从单点调用到企业级AI服务平台的演进之路"
description: "深入剖析LLM网关的架构设计，涵盖路由策略、流量管理、可观测性等核心模块，结合生产实战经验总结企业级AI服务平台的最佳实践"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: distributed
tags: ["LLM", "网关架构", "微服务", "负载均衡", "AI基础设施"]
draft: false
---

## 引言

随着大语言模型（LLM）在企业中的广泛落地，一个容易被忽视但至关重要的基础设施组件正在快速演进——**LLM 网关**。与传统 API 网关不同，LLM 网关需要处理流式输出、Token 计费、模型路由、上下文管理等独特挑战。本文将从零搭建一个企业级 LLM 网关的完整架构，分享我们在生产环境中的实战经验。

## 为什么需要 LLM 网关？

在没有统一网关的情况下，团队通常会遇到以下问题：

```
┌─────────────────────────────────────────────────────────┐
│                    痛点分析                               │
├──────────────┬──────────────────────────────────────────┤
│ 模型碎片化    │ 5+ 个 LLM 供应商，各自 SDK 不同           │
│ 成本失控      │ 无法统一监控 Token 消耗与费用              │
│ 安全风险      │ API Key 散落在各业务代码中                 │
│ 可用性差      │ 单个模型故障直接影响业务                  │
│ 无法切换      │ 迁移模型需要改动所有调用方                 │
└──────────────┴──────────────────────────────────────────┘
```

LLM 网关的核心价值在于**将 LLM 调用从"点对点直连"升级为"统一服务平面"**。

## 整体架构设计

### 分层架构图

```
┌──────────────────────────────────────────────────────────────┐
│                      客户端层 (Client)                        │
│         Web App  │  Mobile  │  内部系统  │  第三方接入          │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   API 网关层 (Gateway)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ 认证鉴权  │ │ 限流熔断  │ │ 请求路由  │ │  协议转换       │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   业务逻辑层 (Core)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ 模型路由  │ │ Prompt   │ │ 流式处理  │ │  Token 计费     │  │
│  │ 引擎     │ │ 管理器   │ │ 管道     │ │  引擎          │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   适配层 (Adapter)                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ OpenAI   │ │ Claude   │ │ 本地模型  │ │  自定义模型     │  │
│  │ Adapter  │ │ Adapter  │ │ Adapter  │ │  Adapter       │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   基础设施层 (Infra)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ 缓存层   │ │ 消息队列  │ │ 监控告警  │ │  模型注册中心   │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 核心设计原则

1. **Provider 无关性**：业务代码不感知底层使用的是哪个模型
2. **流式优先**：所有路径都优先支持 SSE 流式输出
3. **故障隔离**：单个 Provider 故障不影响其他服务
4. **成本可控**：实时 Token 计费与预算告警

## 核心模块详解

### 1. 模型路由引擎

模型路由是网关的"大脑"，决定了每个请求应该发送到哪个模型。我们设计了多维度路由策略：

```python
class ModelRouter:
    """模型路由器 - 支持多维度路由策略"""
    
    def __init__(self):
        self.strategies = []
        self.fallback_chain = []
    
    def add_strategy(self, strategy: RoutingStrategy):
        self.strategies.append(strategy)
    
    async def route(self, request: LLMRequest) -> ModelEndpoint:
        # 按优先级依次尝试路由策略
        for strategy in self.strategies:
            result = await strategy.resolve(request)
            if result is not None:
                return result
        
        # 所有策略都未命中，使用降级链
        return await self._fallback(request)
    
    async def _fallback(self, request: LLMRequest) -> ModelEndpoint:
        """降级策略：从最稳定到最便宜"""
        for endpoint in self.fallback_chain:
            if await endpoint.health_check():
                return endpoint
        raise NoAvailableModelException("所有模型均不可用")


# 路由策略示例：基于任务类型的路由
class TaskTypeStrategy(RoutingStrategy):
    """根据请求内容自动选择模型"""
    
    TASK_MODEL_MAP = {
        "code_generation": ["claude-3.5-sonnet", "gpt-4o"],
        "creative_writing": ["gpt-4o", "claude-3-opus"],
        "classification": ["gpt-4o-mini", "claude-3-haiku"],
        "translation": ["gpt-4o-mini", "deepseek-v3"],
    }
    
    async def resolve(self, request: LLMRequest) -> Optional[ModelEndpoint]:
        task_type = await self._classify_task(request.messages)
        candidates = self.TASK_MODEL_MAP.get(task_type, ["gpt-4o"])
        
        for model_id in candidates:
            endpoint = self.registry.get_healthy(model_id)
            if endpoint:
                return endpoint
        return None
```

### 2. 流式输出管道

LLM 网关最复杂的技术挑战之一是流式输出（Streaming）的透传。我们需要确保：

- 上游的 SSE 事件能无损透传到下游
- 支持中途断开连接时的资源清理
- 流式 Token 能被实时计费

```python
class StreamingPipeline:
    """流式输出处理管道"""
    
    async def handle_stream(
        self, request: LLMRequest, response: StreamingResponse
    ) -> AsyncGenerator[StreamChunk, None]:
        
        token_counter = TokenCounter()
        buffer = StreamBuffer()
        
        try:
            async for chunk in response.stream():
                # 1. 实时统计 Token
                token_counter.update(chunk)
                
                # 2. 检查预算是否超限
                if token_counter.exceeds_budget(request.budget):
                    yield StreamChunk.create_stop("budget_exceeded")
                    break
                
                # 3. 缓冲与转发
                buffer.append(chunk)
                if buffer.should_flush():
                    yield buffer.flush()
            
            # 记录本次调用的 Token 消耗
            await self.metrics.record(
                model=response.model_id,
                input_tokens=token_counter.input_tokens,
                output_tokens=token_counter.output_tokens,
                latency_ms=token_counter.elapsed_ms,
            )
            
        except ConnectionClosed:
            await self._cleanup_partial(response)
        finally:
            await response.close()
```

### 3. 成本管控与计费

企业级场景下，LLM 的成本管控是刚需。我们设计了三级预算控制：

```
┌─────────────────────────────────────────────────────┐
│              三级预算控制体系                          │
├───────────┬────────────────┬────────────────────────┤
│   级别    │   控制粒度      │   触发动作              │
├───────────┼────────────────┼────────────────────────┤
│ L1 请求级  │ 单次请求 Token  │ 超限直接拒绝            │
│ L2 用户级  │ 日/月累计 Token │ 超限降级到小模型         │
│ L3 租户级  │ 月度费用总额    │ 超限暂停服务+告警        │
└───────────┴────────────────┴────────────────────────┘
```

```python
class CostController:
    """成本控制器"""
    
    async def check_budget(self, request: LLMRequest) -> BudgetCheck:
        user_id = request.user_id
        tenant_id = request.tenant_id
        
        # L2: 检查用户日限额
        user_daily = await self.store.get_daily_usage(user_id)
        if user_daily >= self.get_user_daily_limit(user_id):
            return BudgetCheck(
                allowed=False,
                reason="daily_limit_exceeded",
                suggestion="请升级套餐或等待明日额度重置"
            )
        
        # L2.5: 建议降级模型
        if user_daily >= self.get_user_daily_limit(user_id) * 0.8:
            return BudgetCheck(
                allowed=True,
                suggested_model="gpt-4o-mini",
                warning="接近今日额度上限，已自动推荐经济模型"
            )
        
        return BudgetCheck(allowed=True)
```

## 可观测性设计

生产环境的 LLM 网关必须具备完善的可观测性。我们建立了三大支柱：

### 核心监控指标

| 指标类别 | 指标名称 | 说明 | 告警阈值 |
|---------|---------|------|---------|
| 延迟 | TTFT (首Token延迟) | 用户等待感知的关键指标 | > 2s |
| 延迟 | TPOT (每Token延迟) | 影响阅读体验 | > 100ms |
| 延迟 | E2E 延迟 | 端到端总延迟 | > 30s |
| 吞吐 | QPS | 每秒处理请求数 | > 80% 容量 |
| 成本 | Token/请求 | 单次请求平均消耗 | 超预算 20% |
| 质量 | 拒绝率 | 模型拒绝回答的比例 | > 5% |
| 稳定性 | 错误率 | 5xx 错误占比 | > 1% |
| 稳定性 | 超时率 | 超时请求占比 | > 2% |

### 分布式追踪

对于 LLM 调用链路，我们使用 OpenTelemetry 进行端到端追踪：

```
[用户请求]
    │
    ├─── [Gateway 接收] trace_id=abc123
    │       ├─── [认证鉴权] 15ms
    │       ├─── [路由决策] 3ms → model=gpt-4o
    │       ├─── [Prompt模板渲染] 2ms
    │       ├─── [Provider 调用] 
    │       │       ├─── [连接建立] 45ms
    │       │       ├─── [首Token] 820ms  ← TTFT
    │       │       └─── [流式传输] 3200ms ← 总传输时间
    │       ├─── [Token计费] 1ms
    │       └─── [日志写入] 5ms
    └─── [总耗时] 4091ms
```

## 高可用设计

### 多级容灾策略

```
正常流量 ──→ Provider A (主力)
                │
                ├─ 健康检查失败 ──→ Provider B (备选)
                │                      │
                │                      ├─ 也不可用 ──→ Provider C (兜底)
                │                      │                 │
                │                      │                 └─ 全部不可用 ──→ 本地模型
                │                      │
                │                      └─ 响应异常 ──→ 重试策略(最多2次)
                │
                └─ 超时(>30s) ──→ 自动切换到备选
```

### 关键实践

1. **熔断器**：连续 5 次失败触发熔断，30s 后半开探测
2. **重试退避**：指数退避 + 抖动，最多重试 2 次
3. **连接池管理**：每个 Provider 独立连接池，避免相互影响
4. **优雅降级**：大模型不可用时自动切换到小模型，保证基本可用

## 部署架构

### 推荐部署拓扑

```
                    ┌─────────────┐
                    │   CDN/WAF   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Load Balancer│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐┌────▼─────┐┌────▼─────┐
        │ Gateway-1 ││ Gateway-2││ Gateway-3│
        └─────┬─────┘└────┬─────┘└────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
  ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
  │  Redis    │     │ PostgreSQL│     │ Prometheus│
  │ (缓存+限流)│     │ (日志+配置)│     │ (监控)    │
  └───────────┘     └───────────┘     └───────────┘
```

### 资源规划参考

| 规模 | QPS | 内存 | CPU | 实例数 |
|-----|-----|-----|-----|-------|
| 小型团队 | < 100 | 2G | 1核 | 2 |
| 中型企业 | 100-1000 | 4G | 2核 | 3-5 |
| 大型平台 | > 1000 | 8G+ | 4核+ | 5-10 |

## 总结

LLM 网关不仅仅是"请求转发器"，它是企业 AI 服务能力的核心枢纽。关键设计要点：

1. **路由智能化**：基于任务类型、成本、延迟等多维度自动选模
2. **成本可感知**：三级预算控制，避免 Token 费用失控
3. **流式优先**：从架构层面原生支持 SSE 流式输出
4. **高可用**：多 Provider 冗余 + 熔断降级 + 优雅容错
5. **可观测**：端到端追踪 + 实时指标 + 智能告警

从单点直连到统一网关，这不仅是架构的升级，更是 AI 工程化成熟度的体现。
