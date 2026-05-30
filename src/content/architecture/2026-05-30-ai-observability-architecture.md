---
title: "AI系统可观测性架构：从黑盒到全链路追踪的工程实践"
description: "深入剖析AI系统可观测性架构设计，覆盖LLM调用追踪、Token成本监控、幻觉检测、Agent行为审计，附完整架构方案与实现代码"
date: 2026-05-30
author: "RiceBall-15"
category: "architecture"
tags: ["可观测性", "AI系统", "监控", "Agent", "LLM", "架构设计"]
draft: false
---

# AI系统可观测性架构：从黑盒到全链路追踪的工程实践

## 一、引言：AI系统的"黑盒"困境

传统微服务系统的可观测性已经非常成熟——日志（Logs）、指标（Metrics）、链路追踪（Traces）构成了三大支柱。但当系统中引入LLM和Agent后，这套体系面临根本性挑战：

1. **不确定性**：同样的输入，每次输出都不同，无法用传统断言验证
2. **成本不透明**：Token消耗难以预估，一个Agent循环可能烧掉数十美元
3. **质量难量化**：输出是否"正确"？是否有幻觉？是否符合安全策略？
4. **行为不可审计**：Agent的多步推理链路如何追溯？决策依据是什么？

本文将从架构设计的角度，系统性地解决这些问题，构建一套**AI系统全链路可观测性架构**。

## 二、AI可观测性的四大支柱

传统可观测性有三大支柱，AI系统需要扩展为四大支柱：

| 支柱 | 传统系统 | AI系统扩展 |
|------|---------|-----------|
| Logs | 应用日志、错误日志 | LLM交互日志、Prompt/Response记录 |
| Metrics | QPS、延迟、错误率 | Token消耗、模型延迟、质量评分 |
| Traces | 调用链路 | Prompt→推理→工具调用→输出 全链路 |
| **Evaluations** | 无 | 幻觉检测、质量评估、安全审计 |

### 2.1 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                    AI可观测性平台                          │
├──────────────┬──────────────┬──────────────┬─────────────┤
│  Log Layer   │ Metric Layer │ Trace Layer  │ Eval Layer  │
│  交互日志     │  聚合指标     │  分布式追踪   │  质量评估    │
├──────────────┴──────────────┴──────────────┴─────────────┤
│                    统一数据总线                            │
├──────────────┬──────────────┬──────────────┬─────────────┤
│  ClickHouse  │  Prometheus  │   Jaeger     │  PG + pgvector│
│  (日志存储)   │  (指标采集)   │  (链路追踪)   │  (评估数据)   │
└──────────────┴──────────────┴──────────────┴─────────────┘
         ▲              ▲              ▲              ▲
         │              │              │              │
    ┌────┴────┐   ┌─────┴────┐  ┌─────┴────┐  ┌─────┴────┐
    │ AI应用  │   │ AI应用    │  │ AI应用   │  │ 评估服务  │
    │ (SDK)   │   │ (SDK)    │  │ (SDK)    │  │ (异步)   │
    └─────────┘   └──────────┘  └──────────┘  └──────────┘
```

## 三、LLM交互日志设计

### 3.1 日志Schema设计

LLM交互日志是AI可观测性的基础。与传统API日志不同，它需要记录完整的Prompt和Response：

```typescript
interface LLMInteractionLog {
  // 基础信息
  traceId: string;
  spanId: string;
  timestamp: number;
  
  // 模型信息
  model: string;           // "gpt-4o", "claude-sonnet-4-20250514"
  provider: string;        // "openai", "anthropic", "local"
  
  // 输入
  messages: Message[];
  systemPrompt?: string;
  tools?: ToolDefinition[];
  
  // 输出
  response: {
    content?: string;
    toolCalls?: ToolCall[];
    finishReason: string;
  };
  
  // Token统计
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    estimatedCost: number;  // 美元
  };
  
  // 性能
  latency: {
    firstTokenMs: number;   // TTFT (Time To First Token)
    totalMs: number;
    tokensPerSecond: number;
  };
  
  // 质量（异步填充）
  quality?: {
    relevanceScore: number;
    hallucinationDetected: boolean;
    safetyFlags: string[];
  };
  
  // 上下文
  metadata: {
    userId?: string;
    sessionId?: string;
    feature?: string;
    environment: 'dev' | 'staging' | 'prod';
  };
}
```

### 3.2 非侵入式日志采集

关键设计原则：**日志采集不应侵入业务代码**。推荐使用装饰器模式或中间件：

```python
# Python SDK 采集示例
from observability import LLMObservable

class ChatService:
    @LLMObservable(
        capture_prompt=True,      # 记录完整Prompt
        capture_response=True,    # 记录完整Response
        mask_pii=True,            # 脱敏个人信息
        sample_rate=0.1,          # 生产环境10%采样
    )
    async def chat(self, user_message: str) -> str:
        response = await self.llm_client.chat(
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content
```

```typescript
// TypeScript SDK 采集示例
import { Traceable } from '@ai-observe/core';

class AgentService {
  @Traceable({ name: 'agent-reasoning', captureIO: true })
  async reason(task: string): Promise<AgentResult> {
    // 所有LLM调用自动被追踪
    const plan = await this.llm.plan(task);
    const result = await this.executePlan(plan);
    return result;
  }
}
```

### 3.3 高吞吐场景的采样策略

AI系统的日志量可能非常大（尤其是Agent系统，一次请求可能触发数十次LLM调用）。需要分层采样：

| 环境 | 采样策略 | 存储保留 |
|------|---------|---------|
| 开发环境 | 100%全量记录 | 7天 |
| 测试环境 | 100%全量记录 | 30天 |
| 生产环境 | 10%基础采样 + 100%异常采样 | 90天 |
| 高价值请求 | 100%全量记录 | 180天 |

"高价值请求"的判定标准：
- 用户明确反馈的请求
- 触发安全策略的请求
- Token消耗超过阈值的请求
- 延迟异常的请求

## 四、全链路追踪架构

### 4.1 AI调用链的特殊性

传统微服务的调用链是同步的、确定性的：

```
用户请求 → Service A → Service B → Service C → 响应
```

AI系统的调用链是异步的、不确定的、可能包含循环：

```
用户请求 → Agent推理 → [工具调用1 → 外部API]
                      → [LLM判断 → 工具调用2 → 数据库查询]
                      → [LLM判断 → 工具调用3 → 代码执行]
                      → [LLM最终输出] → 响应
```

### 4.2 Span设计

为AI系统设计专用的Span类型：

```python
# Span类型定义
class SpanType(Enum):
    LLM_CALL = "llm.call"           # LLM推理调用
    TOOL_CALL = "tool.call"         # 工具调用
    AGENT_STEP = "agent.step"       # Agent推理步骤
    RETRIEVAL = "retrieval"         # RAG检索
    GUARDRAIL = "guardrail"         # 安全检查
    TRANSFORM = "transform"         # 数据转换

# 每个Span包含AI特有属性
class AISpanAttributes:
    llm_model: str
    prompt_tokens: int
    completion_tokens: int
    tool_name: str
    tool_input: dict
    tool_output: Any
    agent_reasoning: str  # Agent的推理过程
```

### 4.3 分布式追踪实现

使用OpenTelemetry作为追踪基础设施，扩展AI专用语义：

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# 初始化追踪
provider = TracerProvider()
tracer = trace.get_tracer("ai-system")

async def agent_loop(task: str):
    with tracer.start_as_current_span("agent-loop") as root_span:
        root_span.set_attribute("agent.task", task)
        
        for step in range(max_steps):
            with tracer.start_as_current_span(f"agent-step-{step}") as step_span:
                # LLM推理
                with tracer.start_as_current_span("llm-reasoning") as llm_span:
                    response = await llm.chat(messages=messages)
                    llm_span.set_attribute("llm.model", model)
                    llm_span.set_attribute("llm.tokens", response.usage.total)
                
                # 工具调用
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        with tracer.start_as_current_span(
                            f"tool-{tool_call.name}"
                        ) as tool_span:
                            tool_span.set_attribute("tool.name", tool_call.name)
                            result = await execute_tool(tool_call)
                            tool_span.set_attribute("tool.result_size", len(str(result)))
                
                # 终止判断
                if is_final(response):
                    break
```

## 五、Token成本监控与告警

### 5.1 成本模型设计

Token成本是AI系统最直接的运营成本。需要建立精细的成本模型：

```typescript
interface CostModel {
  // 模型定价（每1M tokens，美元）
  models: Record<string, {
    inputPrice: number;
    outputPrice: number;
  }>;
  
  // 缓存折扣
  cacheDiscount: number;  // 通常 0.5x
  
  // 批量折扣
  batchDiscount: number;  // 通常 0.5x
}

const COST_MODEL: CostModel = {
  models: {
    'gpt-4o':        { inputPrice: 2.50,  outputPrice: 10.00 },
    'gpt-4o-mini':   { inputPrice: 0.15,  outputPrice: 0.60 },
    'claude-sonnet-4-20250514':  { inputPrice: 3.00,  outputPrice: 15.00 },
    'claude-haiku':  { inputPrice: 0.25,  outputPrice: 1.25 },
  },
  cacheDiscount: 0.5,
  batchDiscount: 0.5,
};
```

### 5.2 多维度成本分析

成本监控需要支持多个维度的聚合分析：

| 分析维度 | 典型查询 | 业务价值 |
|---------|---------|---------|
| 按模型 | 各模型的日/周/月Token消耗 | 模型选型优化 |
| 按功能 | 各业务功能的Token消耗 | 功能ROI分析 |
| 按用户 | 各用户的Token消耗 | 用户分层计费 |
| 按Agent步骤 | 各步骤的Token消耗 | Agent效率优化 |
| 按工具 | 各工具调用的Token消耗 | 工具链优化 |

### 5.3 告警规则设计

```yaml
# 成本告警规则
alerts:
  - name: "单次请求Token异常"
    condition: "request.total_tokens > 100000"
    severity: warning
    action: "记录并标记，可能的Agent循环"
    
  - name: "日成本超阈值"
    condition: "daily_cost > daily_budget * 1.2"
    severity: critical
    action: "通知团队 + 触发限流"
    
  - name: "Agent循环检测"
    condition: "agent.steps > 10 && agent.final_answer == null"
    severity: critical
    action: "终止Agent + 告警"
    
  - name: "成本趋势异常"
    condition: "weekly_cost > avg_weekly_cost * 2"
    severity: warning
    action: "分析成本增长原因"
```

## 六、Agent行为审计架构

### 6.1 为什么需要Agent审计

Agent系统的核心风险在于**自主性**——它能够自己决定调用哪些工具、执行什么操作。没有审计，你无法回答：

- Agent为什么做出了这个决策？
- 它调用了哪些外部服务？
- 它是否执行了危险操作？
- 用户的敏感数据是否被正确处理？

### 6.2 审计日志设计

```typescript
interface AgentAuditLog {
  // 会话信息
  sessionId: string;
  userId: string;
  startTime: number;
  endTime: number;
  
  // 任务描述
  task: string;
  taskClassification: 'safe' | 'sensitive' | 'critical';
  
  // 推理链
  reasoningChain: {
    step: number;
    thought: string;          // Agent的推理过程
    action: string;           // 执行的动作
    actionInput: any;         // 动作输入
    actionOutput: any;        // 动作输出
    observation: string;      // 执行结果观察
    timestamp: number;
  }[];
  
  // 工具调用记录
  toolCalls: {
    toolName: string;
    input: any;
    output: any;
    duration: number;
    success: boolean;
    error?: string;
  }[];
  
  // 安全审计
  securityAudit: {
    sensitiveDataAccessed: boolean;
    externalApiCalls: string[];
    fileSystemOperations: string[];
    networkRequests: string[];
    riskScore: number;  // 0-100
  };
  
  // 最终结果
  result: {
    status: 'success' | 'partial' | 'failed' | 'safety_blocked';
    output: any;
    userFeedback?: 'positive' | 'negative';
  };
}
```

### 6.3 实时审计拦截

在Agent执行关键操作前，插入审计拦截层：

```python
class AuditInterceptor:
    """Agent操作审计拦截器"""
    
    # 高风险操作列表
    HIGH_RISK_TOOLS = {
        'file_delete', 'database_write', 'api_call_external',
        'code_execute', 'payment_process', 'user_data_access'
    }
    
    async def intercept(self, tool_call: ToolCall) -> bool:
        """拦截并审计工具调用，返回是否允许执行"""
        
        # 记录审计日志
        await self.audit_logger.log_tool_call(tool_call)
        
        # 高风险操作需要额外验证
        if tool_call.name in self.HIGH_RISK_TOOLS:
            risk_score = await self.risk_assessor.assess(tool_call)
            
            if risk_score > 80:
                # 高风险：阻断并告警
                await self.alert_service.critical(
                    f"Agent高风险操作被阻断: {tool_call.name}"
                )
                return False
            
            if risk_score > 50:
                # 中风险：记录并放行，但标记
                await self.audit_logger.flag_as_sensitive(tool_call)
        
        return True
```

## 七、质量评估体系

### 7.1 自动化质量评估

AI输出的质量评估需要结合规则检查和模型评估：

```python
class QualityEvaluator:
    """AI输出质量评估器"""
    
    async def evaluate(self, interaction: LLMInteractionLog) -> QualityResult:
        scores = {}
        
        # 1. 幻觉检测
        scores['hallucination'] = await self.detect_hallucination(
            response=interaction.response.content,
            context=interaction.messages,
        )
        
        # 2. 相关性评分
        scores['relevance'] = await self.score_relevance(
            query=interaction.messages[-1].content,
            response=interaction.response.content,
        )
        
        # 3. 安全检查
        scores['safety'] = await self.safety_check(
            response=interaction.response.content,
        )
        
        # 4. 格式合规
        scores['format'] = self.check_format(
            response=interaction.response.content,
            expected_format=interaction.metadata.get('expected_format'),
        )
        
        # 5. 一致性检查（与历史输出对比）
        scores['consistency'] = await self.check_consistency(
            response=interaction.response.content,
            similar_queries=await self.find_similar(interaction),
        )
        
        return QualityResult(
            overall_score=self._compute_weighted_score(scores),
            scores=scores,
            flags=self._generate_flags(scores),
        )
```

### 7.2 评估驱动的持续优化

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  用户反馈   │───→│  质量评估    │───→│  数据分析   │
│  (正/负)    │    │  (自动+人工) │    │  (趋势洞察) │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Prompt优化 │←───│  策略生成    │←───│  根因分析   │
│  (自动调优) │    │  (规则引擎)  │    │  (问题定位) │
└─────────────┘    └──────────────┘    └─────────────┘
```

关键指标看板：

| 指标 | 计算方式 | 目标值 | 告警阈值 |
|------|---------|--------|---------|
| 幻觉率 | 幻觉检测命中数/总请求数 | < 2% | > 5% |
| 用户满意度 | 正面反馈/总反馈 | > 85% | < 70% |
| 首Token延迟 | TTFT P95 | < 500ms | > 2000ms |
| 端到端延迟 | 总延迟 P95 | < 10s | > 30s |
| Token效率 | 有效输出/总Token | > 0.3 | < 0.1 |
| 工具成功率 | 成功调用/总调用 | > 95% | < 80% |

## 八、技术选型与落地建议

### 8.1 开源方案对比

| 方案 | 定位 | 优势 | 劣势 |
|------|------|------|------|
| LangSmith | LLM应用平台 | 功能全面，LangChain生态 | 商业产品，成本较高 |
| Langfuse | LLM可观测性 | 开源，自部署，功能完善 | 社区相对较小 |
| Arize Phoenix | ML可观测性 | 评估能力强，UI美观 | 侧重传统ML |
| Helicone | LLM代理层 | 简单易用，按需计费 | 自定义能力有限 |
| 自建方案 | 完全自定义 | 灵活，无外部依赖 | 开发成本高 |

### 8.2 推荐架构（中小团队）

对于中小团队，推荐基于Langfuse自建：

```
┌─────────────────────────────────────────┐
│              AI应用层                     │
├─────────────────────────────────────────┤
│    Langfuse SDK（非侵入式采集）           │
├─────────────────────────────────────────┤
│              Langfuse Server             │
│  ┌───────┐  ┌───────┐  ┌──────────┐    │
│  │Traces │  │Scores │  │  Prompts │    │
│  └───────┘  └───────┘  └──────────┘    │
├─────────────────────────────────────────┤
│         PostgreSQL + ClickHouse          │
└─────────────────────────────────────────┘
```

部署命令：

```bash
# Docker Compose 一键部署
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d

# 访问 http://localhost:3000
```

### 8.3 落地路线图

| 阶段 | 时间 | 目标 | 交付物 |
|------|------|------|--------|
| 第一阶段 | 1-2周 | 基础日志采集 | LLM调用日志全量记录 |
| 第二阶段 | 2-3周 | 链路追踪 | Agent全链路可视化 |
| 第三阶段 | 1-2周 | 成本监控 | Token消耗看板 + 告警 |
| 第四阶段 | 2-3周 | 质量评估 | 自动化质量评分系统 |
| 第五阶段 | 持续 | 审计与合规 | Agent行为审计 + 安全拦截 |

## 九、结语

AI系统的可观测性不是"锦上添花"，而是**生产必备**。没有可观测性的AI系统，就像没有仪表盘的飞机——你不知道它在飞多高、燃料还剩多少、是否偏离航线。

核心原则总结：

1. **日志先行**：先能记录一切，再谈分析和优化
2. **成本可见**：每一个Token都要有归属，每一次调用都要有计费
3. **行为可审计**：Agent的每一步决策都要可追溯、可解释
4. **质量可度量**：用数据驱动Prompt优化和模型选型
5. **渐进式落地**：不要追求一步到位，按阶段逐步完善

当你的AI系统拥有了完善的可观测性，你就从"凭感觉调优"进化到了"用数据决策"。这才是AI工程化的真正起点。
