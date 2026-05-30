---
title: "AI Gateway架构设计：构建企业级AI请求网关"
description: "从零设计企业级AI Gateway，涵盖路由、限流、缓存、可观测性与多模型管理的完整方案"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: "microservices"
tags: ["AI Gateway", "架构设计", "微服务", "可观测性", "限流"]
draft: false
---

## 引言

随着AI能力渗透到业务的每个角落，一个核心问题浮出水面：**如何统一管理组织内数十个AI模型、数百个AI应用的请求？** 传统的API Gateway无法处理流式输出、Token计费、模型路由等AI特有需求。AI Gateway应运而生——它是AI时代的"入口守门员"。

本文将从架构演进、核心模块设计、实战部署三个层面，分享构建企业级AI Gateway的完整经验。

## 一、为什么需要AI Gateway

### 1.1 没有Gateway的痛点

```
现状（没有统一Gateway）:

App A ──→ OpenAI API
App B ──→ 本地Ollama
App C ──→ Azure OpenAI
App D ──→ 内部vLLM集群
App E ──→ Anthropic API
```

| 痛点 | 具体表现 | 风险 |
|------|----------|------|
| **密钥管理** | API Key散落在各应用配置中 | 泄露风险、审计困难 |
| **成本失控** | 无法统一计量Token消耗 | 月度账单不可预测 |
| **模型切换** | 更换模型需要改代码、重新部署 | 迁移成本高 |
| **质量监控** | 无法统一收集延迟、错误率指标 | 问题排查困难 |
| **安全合规** | 敏感数据直接发往外部API | 数据泄露风险 |

### 1.2 AI Gateway的定位

```
Client Apps
    ↓
┌─────────────────────────────────────────┐
│            AI Gateway                   │
│  ┌─────┬──────┬──────┬──────┬────────┐ │
│  │ Auth │ Rate │ Cache│Filter│Router  │ │
│  │ 认证  │ 限流  │ 缓存  │过滤  │路由    │ │
│  └─────┴──────┴──────┴──────┴────────┘ │
│  ┌─────────────────────────────────┐   │
│  │     Observability Layer          │   │
│  │     可观测性层（日志/指标/追踪）    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓
┌────┬────┬────┬────┬────┐
│OpenAI│Claude│vLLM│Ollama│自研│
└────┴────┴────┴────┴────┘
```

## 二、核心架构设计

### 2.1 整体架构

采用**分层架构**，每层职责清晰：

```
┌────────────────────────────────────────────┐
│                  接入层                      │
│   HTTP/gRPC/WebSocket → 协议转换 → 请求标准化  │
├────────────────────────────────────────────┤
│                  策略层                      │
│   认证鉴权 → 限流熔断 → 内容过滤 → 付费计量   │
├────────────────────────────────────────────┤
│                  路由层                      │
│   模型选择 → 负载均衡 → 故障转移 → A/B测试    │
├────────────────────────────────────────────┤
│                  适配层                      │
│   Provider适配 → 协议适配 → 响应格式标准化     │
├────────────────────────────────────────────┤
│                  可观测层                    │
│   结构化日志 → Prometheus指标 → OpenTelemetry│
└────────────────────────────────────────────┘
```

### 2.2 请求生命周期

一次AI请求在Gateway中的完整路径：

```
Client → ① 请求校验 → ② 认证鉴权 → ③ 限流检查
  → ④ 内容安全过滤 → ⑤ 缓存查询 → ⑥ 模型路由
  → ⑦ Provider适配 → ⑧ 上游调用 → ⑨ 响应处理
  → ⑩ 计量计费 → ⑪ 指标上报 → Client
```

每个环节都是可插拔的中间件，支持按需组合。

## 三、核心模块设计

### 3.1 统一请求模型

不同Provider的API格式各异，Gateway需要统一内部请求模型：

```typescript
// 统一请求模型
interface UnifiedChatRequest {
  // 统一字段
  messages: Message[];
  model: string;          // 逻辑模型名，如 "gpt-4o-auto"
  stream: boolean;
  temperature?: number;
  maxTokens?: number;
  
  // Gateway扩展字段
  metadata: {
    appId: string;        // 调用方应用标识
    userId: string;       // 最终用户标识
    priority: 'low' | 'medium' | 'high';
    cacheKey?: string;    // 自定义缓存键
    timeout?: number;     // 超时时间(ms)
  };
}

// Provider适配接口
interface ProviderAdapter {
  name: string;
  translateRequest(req: UnifiedChatRequest): any;
  translateResponse(resp: any): UnifiedChatResponse;
  streamAdapter(resp: ReadableStream): ReadableStream;
}
```

### 3.2 智能路由引擎

路由不只是简单的负载均衡，需要考虑多维度因素：

```typescript
class SmartRouter {
  // 路由决策矩阵
  route(request: UnifiedChatRequest): ProviderTarget {
    const candidates = this.filterByCapability(request);
    
    // 评分模型：综合多维度打分
    const scored = candidates.map(provider => ({
      provider,
      score: this.score(provider, request)
    }));
    
    scored.sort((a, b) => b.score - a.score);
    return scored[0].provider;
  }

  private score(provider: Provider, request: UnifiedChatRequest): number {
    const weights = {
      latency: 0.3,      // 历史延迟表现
      availability: 0.25, // 可用性
      cost: 0.2,          // 成本
      quality: 0.15,      // 输出质量
      load: 0.1,          // 当前负载
    };
    
    return (
      weights.latency * this.latencyScore(provider) +
      weights.availability * this.availabilityScore(provider) +
      weights.cost * this.costScore(provider, request) +
      weights.quality * this.qualityScore(provider, request) +
      weights.load * (1 - provider.currentLoad)
    );
  }
}
```

**路由规则示例**：

| 规则 | 条件 | 动作 |
|------|------|------|
| 成本优先 | `metadata.priority === 'low'` | 路由到最便宜的模型 |
| 质量优先 | `metadata.priority === 'high'` | 路由到最强模型 |
| 地域合规 | 用户在中国大陆 | 仅路由到境内Provider |
| 故障转移 | 主Provider错误率>5% | 自动切换备用Provider |
| A/B测试 | 流量比例10% | 路由到新模型评估 |

### 3.3 Token级限流

AI场景的限流与传统API不同，需要基于Token消耗而非请求数：

```typescript
class TokenRateLimiter {
  // 滑动窗口 + 令牌桶混合方案
  private buckets: Map<string, TokenBucket> = new Map();
  
  async checkLimit(request: UnifiedChatRequest): Promise<RateLimitResult> {
    const key = request.metadata.appId;
    const bucket = this.getOrCreateBucket(key);
    
    // 预估Token消耗（基于消息长度启发式估算）
    const estimatedTokens = this.estimateTokens(request);
    
    if (!bucket.tryConsume(estimatedTokens)) {
      return {
        allowed: false,
        retryAfter: bucket.getRefillTime(),
        currentUsage: bucket.currentUsage,
        limit: bucket.capacity,
      };
    }
    
    return { allowed: true };
  }

  // 基于历史数据动态调整限制
  async adjustLimits(appId: string): Promise<void> {
    const usage = await this.getUsageStats(appId);
    const tier = this.calculateTier(usage);
    
    // 不同等级不同配额
    const limits = {
      free: { rpm: 10, tpm: 10_000 },
      basic: { rpm: 60, tpm: 100_000 },
      pro: { rpm: 500, tpm: 1_000_000 },
      enterprise: { rpm: 5000, tpm: 10_000_000 },
    };
    
    this.buckets.get(appId)!.updateCapacity(limits[tier]);
  }
}
```

### 3.4 智能缓存层

AI缓存的关键挑战是**语义相似性**——相同含义的不同表达应该命中缓存：

```typescript
class SemanticCache {
  // 两级缓存：精确匹配 + 语义匹配
  private exactCache: LRUCache<string, CachedResponse>;
  private semanticCache: VectorIndex;  // 向量索引

  async get(request: UnifiedChatRequest): Promise<CachedResponse | null> {
    // Level 1: 精确匹配（毫秒级）
    const cacheKey = this.computeExactKey(request);
    const exact = this.exactCache.get(cacheKey);
    if (exact) return exact;
    
    // Level 2: 语义匹配（需要向量检索）
    const queryEmbed = await this.embed(request.messages);
    const similar = await this.semanticCache.search(queryEmbed, {
      threshold: 0.92,  // 相似度阈值
      maxResults: 3,
    });
    
    if (similar.length > 0) {
      // 验证语义一致性（用小模型快速校验）
      const verified = await this.verifySemantic(
        request.messages, 
        similar[0].messages
      );
      if (verified) return similar[0].response;
    }
    
    return null;
  }
}
```

**缓存策略分层**：

| 缓存层 | 命中条件 | 延迟 | 适用场景 |
|--------|----------|------|----------|
| L1 精确匹配 | 完全相同的请求 | <1ms | 重复查询、FAQ |
| L2 语义匹配 | 含义相似的请求 | 5-20ms | 知识问答 |
| L3 Prefix共享 | 共享system prompt | — | 多轮对话(结合RadixAttention) |

### 3.5 可观测性体系

AI应用的可观测性有其特殊性——需要追踪Token级的细粒度指标：

```typescript
// 结构化日志格式
interface AIGatewayLog {
  // 请求维度
  requestId: string;
  appId: string;
  userId: string;
  
  // 模型维度
  model: string;
  provider: string;
  
  // Token维度（AI特有）
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  
  // 性能维度
  ttft: number;           // 首Token延迟(ms)
  e2eLatency: number;     // 端到端延迟(ms)
  tokensPerSecond: number; // 生成速度
  
  // 质量维度（可选）
  finishReason: string;
  contentFilterHit: boolean;
  
  // 成本维度
  estimatedCost: number;  // 美元
}
```

**关键监控指标**：

```
# Prometheus指标示例
ai_gateway_requests_total{app, model, provider, status}
ai_gateway_tokens_total{app, model, type="prompt|completion"}
ai_gateway_latency_seconds{app, model, quantile}
ai_gateway_ttft_seconds{app, model, quantile}
ai_gateway_cost_dollars_total{app, model}
ai_gateway_cache_hit_ratio{app}
ai_gateway_error_rate{app, provider}
```

## 四、流式输出处理

流式（SSE）是AI应用的标准交互方式，Gateway需要透明代理流式响应：

```typescript
class StreamingProxy {
  async proxySSE(
    upstream: ReadableStream,
    client: ServerResponse
  ): Promise<void> {
    const reader = upstream.getReader();
    const encoder = new TextEncoder();
    
    // 设置SSE头
    client.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });
    
    let buffer = '';
    let tokenCount = 0;
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += new TextDecoder().decode(value);
      
      // 逐chunk处理
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            // 流结束，记录统计
            this.recordStreamStats(tokenCount);
            client.write('data: [DONE]\n\n');
            break;
          }
          
          // 实时转发，同时统计Token
          const parsed = JSON.parse(data);
          tokenCount += parsed.choices?.[0]?.delta?.content?.length || 0;
          client.write(line + '\n\n');
        }
      }
    }
    
    client.end();
  }
}
```

## 五、部署架构

### 5.1 推荐部署模式

```
                    ┌─────────────┐
                    │  Load       │
                    │  Balancer   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ Gateway │  │ Gateway │  │ Gateway │
        │ Node 1  │  │ Node 2  │  │ Node 3  │
        └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │
             └────────────┼────────────┘
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Redis    │ │ PG       │ │Prometheus│
        │ (缓存/   │ │ (日志/   │ │(指标)    │
        │  限流)   │ │  审计)   │ │          │
        └──────────┘ └──────────┘ └──────────┘
```

### 5.2 关键配置

```yaml
# gateway-config.yaml
providers:
  openai:
    type: openai
    apiKey: ${OPENAI_API_KEY}
    baseURL: https://api.openai.com/v1
    models: [gpt-4o, gpt-4o-mini, o3]
    rateLimit: { rpm: 500, tpm: 2_000_000 }
    
  anthropic:
    type: anthropic
    apiKey: ${ANTHROPIC_API_KEY}
    models: [claude-sonnet-4-20250514, claude-3-5-haiku-20241022]
    rateLimit: { rpm: 300, tpm: 1_000_000 }
    
  internal-vllm:
    type: openai  # vLLM兼容OpenAI格式
    baseURL: http://vllm-cluster.internal:8000
    models: [llama-3.1-8b, qwen2.5-72b]
    rateLimit: { rpm: 1000, tpm: 5_000_000 }

routing:
  default: openai
  rules:
    - match: { model: "fast-*" }
      route: [internal-vllm, openai]  # 优先内部，降级外部
    - match: { app: "customer-service" }
      route: [anthropic]              # 客服用Claude
    - match: { region: "cn" }
      route: [internal-vllm]          # 境内数据不出境

cache:
  enabled: true
  semanticThreshold: 0.92
  ttlSeconds: 3600
  maxEntries: 100000

observability:
  logging:
    level: info
    format: json
    output: stdout
  metrics:
    enabled: true
    port: 9090
  tracing:
    enabled: true
    exporter: otlp
    endpoint: http://otel-collector:4317
```

## 六、安全设计

### 6.1 多层安全防护

```
Layer 1: 认证鉴权
  ├── API Key / JWT Token
  ├── OAuth 2.0 (企业SSO集成)
  └── mTLS (服务间通信)

Layer 2: 内容安全
  ├── Prompt注入检测
  ├── PII/敏感信息过滤
  ├── 毒性内容检测
  └── 越狱攻击防御

Layer 3: 数据安全
  ├── 请求/响应加密传输
  ├── 日志脱敏
  ├── 数据驻留合规
  └── 审计日志不可篡改

Layer 4: 运行安全
  ├── 请求大小限制
  ├── 超时控制
  ├── 异常流量检测
  └── 自动熔断
```

### 6.2 Prompt注入防御

```typescript
class InjectionDetector {
  // 多策略组合检测
  detect(input: string): InjectionResult {
    const checks = [
      this.checkRoleConfusion(input),
      this.checkInstructionOverride(input),
      this.checkExfiltrationPattern(input),
      this.checkJailbreakPatterns(input),
    ];
    
    const flags = checks.filter(c => c.detected);
    
    if (flags.length >= 2) {
      return { blocked: true, reason: 'Suspicious input pattern' };
    }
    
    return { blocked: false };
  }

  private checkRoleConfusion(input: string): CheckResult {
    const patterns = [
      /ignore\s+(all\s+)?previous\s+instructions/i,
      /forget\s+everything/i,
      /you\s+are\s+now\s+(?:DAN|jailbroken)/i,
      /system\s*:\s*/i,  // 伪造系统消息
    ];
    
    return {
      detected: patterns.some(p => p.test(input)),
      type: 'role_confusion'
    };
  }
}
```

## 七、落地效果

在实际部署中，AI Gateway带来了显著的改善：

| 指标 | 部署前 | 部署后 | 改善 |
|------|--------|--------|------|
| API Key泄露事件 | 季度2-3次 | 0次 | 100%↓ |
| 月度AI成本 | 不可控 | 精确计量，预算内 | 可预测 |
| 模型切换时间 | 数天 | 分钟级 | 100x↑ |
| 故障恢复时间 | 30min+ | 自动转移<1min | 30x↑ |
| 端到端延迟监控 | 无 | 全链路可观测 | 从0到1 |

## 总结

AI Gateway不是传统API Gateway的简单扩展，而是AI原生的基础设施。它的核心价值在于：

1. **统一入口**：屏蔽底层Provider差异，业务无感切换模型
2. **精细管控**：Token级限流、计费、安全防护
3. **智能路由**：基于成本、延迟、质量的多维度调度
4. **全面可观测**：从Token到延迟的全链路监控

构建AI Gateway是一次"先难后易"的投资——前期的架构设计投入，换来的是业务长期的敏捷性和可控性。
