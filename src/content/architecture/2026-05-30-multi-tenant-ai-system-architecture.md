---
title: "多租户AI系统架构设计：从单体到弹性隔离的生产实践"
description: "深入剖析多租户AI系统的架构演进路径、隔离策略、资源调度与成本分摊，结合真实场景给出可落地的架构方案"
date: 2026-05-30
author: "RiceBall-15"
category: "architecture"
subCategory: "microservices"
tags: ["多租户", "AI架构", "系统设计", "弹性伸缩", "资源隔离", "微服务"]
draft: false
---

# 多租户AI系统架构设计：从单体到弹性隔离的生产实践

## 一、引言：当AI应用从服务一个客户到服务一千个客户

### 1.1 多租户问题的特殊性

传统SaaS系统的多租户架构已经非常成熟——数据库Schema隔离、行级权限、资源配额管理，这些都是经典方案。但AI系统的多租户面临一系列**传统SaaS从未遇到过的挑战**：

```
┌──────────────────────────────────────────────────────────────┐
│              AI多租户 vs 传统SaaS多租户                        │
│                                                              │
│  传统SaaS:                                                   │
│  ├── 资源消耗: CPU/内存 (可预测、线性增长)                      │
│  ├── 扩展方式: 水平扩展数据库+应用服务器                        │
│  ├── 性能基准: 响应时间 < 200ms                               │
│  ├── 成本模型: 固定服务器成本 + 按用户计费                      │
│  └── 隔离需求: 数据隔离 + API隔离                              │
│                                                              │
│  AI系统:                                                     │
│  ├── 资源消耗: GPU算力 (突发性、不可预测)                       │
│  ├── 扩展方式: GPU集群调度 (受限于硬件供应)                     │
│  ├── 性能基准: 推理延迟 100ms-30s (差异数百倍)                  │
│  ├── 成本模型: Token消耗 (与使用量强相关)                       │
│  └── 隔离需求: 数据+算力+模型+知识库 全方位隔离                  │
│                                                              │
│  核心矛盾: GPU资源昂贵且有限，但租户的需求弹性极大               │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 多租户AI系统的典型场景

| 场景 | 租户数量 | 单租户请求量 | 隔离要求 | 典型案例 |
|------|---------|------------|---------|---------|
| AI SaaS平台 | 100-10000 | 低-中 | 数据隔离 | ChatGPT、Claude |
| 企业AI中台 | 5-50 | 中-高 | 数据+算力隔离 | 内部AI服务平台 |
| AI模型托管 | 10-200 | 高 | 强算力隔离 | 模型推理平台 |
| 垂直行业AI | 3-20 | 极高 | 全方位隔离 | 金融/医疗AI系统 |

### 1.3 架构演进全景

本文将沿着多租户AI系统的**四个演进阶段**，逐步深入每个阶段的架构设计、核心挑战和解决方案：

```
阶段1          阶段2          阶段3          阶段4
单体共享   →   逻辑隔离   →   资源池化   →   弹性隔离
(1-5租户)      (5-50租户)     (50-500租户)    (500+租户)
```

## 二、阶段1：单体共享架构（MVP验证期）

### 2.1 架构设计

对于AI应用的MVP（最小可行产品）阶段，多租户最简单的实现方式就是**应用层路由**——所有租户共享同一套基础设施，通过代码逻辑区分租户。

```
┌──────────────────────────────────────────────────────────────┐
│                 阶段1: 单体共享架构                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              统一API入口                           │        │
│  │  Header: X-Tenant-ID: tenant_abc                 │        │
│  └──────────────────────┬───────────────────────────┘        │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │              AI应用服务 (单体)                     │        │
│  │                                                  │        │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │        │
│  │  │ 租户识别  │→ │ 业务逻辑  │→ │ Prompt模板管理 │ │        │
│  │  └──────────┘  └──────────┘  └───────────────┘ │        │
│  └──────────────────────┬───────────────────────────┘        │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │              共享数据库 (多租户表)                  │        │
│  │                                                  │        │
│  │  conversations表:                                 │        │
│  │  | id | tenant_id | user_id | message | ...   |  │        │
│  │                                                  │        │
│  │  每个查询都带 WHERE tenant_id = ?                │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              共享LLM调用                          │        │
│  │  所有租户 → 同一个OpenAI Key → 同一个速率限制      │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 核心实现

```python
# 租户上下文管理
from contextvars import ContextVar
from dataclasses import dataclass

tenant_context: ContextVar['TenantContext'] = ContextVar('tenant_context')

@dataclass
class TenantContext:
    tenant_id: str
    plan: str  # free / pro / enterprise
    rate_limit: int  # 每分钟最大请求数
    model_access: list[str]  # 可使用的模型列表
    budget_limit: float  # 月度预算上限

# 中间件: 从请求中提取租户信息
@app.middleware("http")
async def tenant_middleware(request: Request):
    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        return JSONResponse({"error": "Missing tenant ID"}, status_code=400)
    
    # 从数据库加载租户配置
    tenant = await db.get_tenant(tenant_id)
    tenant_context.set(TenantContext(
        tenant_id=tenant.id,
        plan=tenant.plan,
        rate_limit=TENANT_PLANS[tenant.plan]["rate_limit"],
        model_access=TENANT_PLANS[tenant.plan]["models"],
        budget_limit=TENANT_PLANS[tenant.plan]["budget"],
    ))
    
    response = await call_next(request)
    return response

# 路由层: 根据租户Plan选择模型
async def chat_completion(messages, tenant: TenantContext):
    # 检查预算
    usage = await db.get_tenant_usage(tenant.tenant_id)
    if usage.current_month_cost >= tenant.budget_limit:
        raise BudgetExceededError(tenant.tenant_id)
    
    # 根据Plan选择模型
    if tenant.plan == "enterprise":
        model = "gpt-4o"
    elif tenant.plan == "pro":
        model = "gpt-4o-mini"
    else:
        model = "gpt-4o-mini"  # 免费用户也用mini
    
    # 检查模型权限
    if model not in tenant.model_access:
        raise ModelAccessDeniedError(model, tenant.plan)
    
    # 速率限制
    if not rate_limiter.allow(tenant.tenant_id, tenant.rate_limit):
        raise RateLimitExceededError(tenant.tenant_id)
    
    # 调用LLM
    response = await llm_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    # 记录用量
    await db.record_usage(tenant.tenant_id, response.usage)
    
    return response
```

### 2.3 阶段1的局限性

当租户数量从5增长到50时，这个架构的瓶颈开始暴露：

| 问题 | 表现 | 根因 |
|------|------|------|
| **数据泄露风险** | 某个租户的查询偶尔返回其他租户的数据 | WHERE条件遗漏 |
| **性能互相影响** | 租户A的大批量请求导致租户B延迟飙升 | 共享资源池无隔离 |
| **成本不透明** | 无法精确计算每个租户的真实成本 | 缺少细粒度计量 |
| **无法弹性计费** | 无法按Token消耗向租户收费 | 缺少Token级计量 |
| **升级维护困难** | 改一个租户的配置可能影响其他租户 | 单体耦合 |

## 三、阶段2：逻辑隔离架构（规模化初期）

### 3.1 架构设计

阶段2的核心改进是引入**租户感知的资源调度**——虽然底层资源仍然共享，但在逻辑层面实现了清晰的隔离边界。

```
┌──────────────────────────────────────────────────────────────┐
│                 阶段2: 逻辑隔离架构                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              API Gateway + 租户路由                │        │
│  │                                                  │        │
│  │  /v1/tenants/{tenant_id}/chat                     │        │
│  │  /v1/tenants/{tenant_id}/models                   │        │
│  │  /v1/tenants/{tenant_id}/usage                    │        │
│  └──────────────────────┬───────────────────────────┘        │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │              租户调度器 (Tenant Scheduler)         │        │
│  │                                                  │        │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │        │
│  │  │ 配额管理    │  │ 优先级队列  │  │ 降级策略    │ │        │
│  │  │ (Quota)    │  │(Priority)  │  │(Fallback)  │ │        │
│  │  └────────────┘  └────────────┘  └────────────┘ │        │
│  └──────────────────────┬───────────────────────────┘        │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │              模型服务层 (按租户分组)                │        │
│  │                                                  │        │
│  │  Enterprise Pool:  GPT-4o / Claude-4-sonnet      │        │
│  │  Pro Pool:         GPT-4o-mini / DeepSeek-V3     │        │
│  │  Free Pool:        Llama-3 (自部署) / DeepSeek    │        │
│  │                                                  │        │
│  │  每个Pool独立配置:                                 │        │
│  │  - 速率限制                                       │        │
│  │  - 并发上限                                       │        │
│  │  - 重试策略                                       │        │
│  │  - 成本预算                                       │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              数据层 (Schema隔离)                   │        │
│  │                                                  │        │
│  │  PostgreSQL:                                      │        │
│  │  ├── schema_tenant_abc: conversations, files     │        │
│  │  ├── schema_tenant_def: conversations, files     │        │
│  │  └── schema_tenant_ghi: conversations, files     │        │
│  │                                                  │        │
│  │  向量数据库:                                       │        │
│  │  ├── Collection: kb_tenant_abc                    │        │
│  │  ├── Collection: kb_tenant_def                    │        │
│  │  └── Collection: kb_tenant_ghi                    │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 租户配额管理引擎

这是阶段2最核心的组件——它决定了每个租户能用多少资源、以什么优先级使用：

```python
from enum import Enum
from dataclasses import dataclass, field

class TenantTier(Enum):
    ENTERPRISE = "enterprise"
    PRO = "pro"
    FREE = "free"

@dataclass
class QuotaConfig:
    """租户配额配置"""
    tier: TenantTier
    # 并发控制
    max_concurrent_requests: int
    max_tokens_per_minute: int  # TPM
    max_requests_per_minute: int  # RPM
    # 成本控制
    monthly_budget_usd: float
    # 模型权限
    allowed_models: list[str]
    # 优先级 (0-100, 越高越优先)
    priority: int
    # 降级配置
    fallback_models: list[str]  # 主模型不可用时的降级选项
    
    # 预定义配置
    PRESETS = {
        TenantTier.ENTERPRISE: {
            "max_concurrent_requests": 100,
            "max_tokens_per_minute": 1_000_000,
            "max_requests_per_minute": 1000,
            "monthly_budget_usd": 50000,
            "allowed_models": ["gpt-4o", "claude-4-sonnet", "deepseek-v3"],
            "priority": 90,
            "fallback_models": ["claude-4-sonnet", "deepseek-v3"],
        },
        TenantTier.PRO: {
            "max_concurrent_requests": 20,
            "max_tokens_per_minute": 200_000,
            "max_requests_per_minute": 200,
            "monthly_budget_usd": 5000,
            "allowed_models": ["gpt-4o-mini", "deepseek-v3"],
            "priority": 60,
            "fallback_models": ["deepseek-v3", "llama-3-70b"],
        },
        TenantTier.FREE: {
            "max_concurrent_requests": 5,
            "max_tokens_per_minute": 50_000,
            "max_requests_per_minute": 50,
            "monthly_budget_usd": 100,
            "allowed_models": ["llama-3-8b", "deepseek-chat"],
            "priority": 30,
            "fallback_models": ["llama-3-8b"],
        },
    }


class TenantScheduler:
    """租户调度器 - 管理请求优先级和资源分配"""
    
    def __init__(self):
        self.quotas: dict[str, QuotaConfig] = {}
        self.usage_tracker: dict[str, UsageTracker] = {}
        self.priority_queue: PriorityQueue = PriorityQueue()
    
    async def schedule_request(self, tenant_id: str, request: LLMRequest):
        """调度一个LLM请求"""
        quota = self.quotas[tenant_id]
        usage = self.usage_tracker[tenant_id]
        
        # 1. 检查月度预算
        if usage.monthly_cost >= quota.monthly_budget_usd:
            raise BudgetExceeded(
                tenant_id=tenant_id,
                current=usage.monthly_cost,
                limit=quota.monthly_budget_usd
            )
        
        # 2. 检查速率限制 (Token Bucket算法)
        if not usage.token_bucket.consume(
            tokens=request.estimated_tokens,
            rate=quota.max_tokens_per_minute
        ):
            # 放入优先级队列等待
            await self.priority_queue.enqueue(
                tenant_id=tenant_id,
                priority=quota.priority,
                request=request,
                max_wait=30  # 最多等待30秒
            )
            return await self.priority_queue.dequeue(tenant_id)
        
        # 3. 选择模型 (支持降级)
        model = self._select_model(quota, request.preferred_model)
        
        # 4. 执行请求
        response = await self._execute_with_fallback(
            model=model,
            fallbacks=quota.fallback_models,
            request=request
        )
        
        # 5. 记录用量
        await usage.record(
            tokens=response.usage.total_tokens,
            model=model,
            cost=self._calculate_cost(model, response.usage)
        )
        
        return response
    
    def _select_model(self, quota: QuotaConfig, preferred: str) -> str:
        """根据配额选择模型"""
        if preferred in quota.allowed_models:
            return preferred
        
        # 不在允许列表中，选择最高权限的可用模型
        for model in quota.allowed_models:
            if self._is_model_available(model):
                return model
        
        raise NoModelAvailable(tenant_id=quota.tenant_id)
    
    async def _execute_with_fallback(self, model, fallbacks, request):
        """带降级的请求执行"""
        try:
            return await self._call_model(model, request)
        except (ModelOverloadedError, RateLimitError) as e:
            # 主模型失败，尝试降级
            for fallback in fallbacks:
                try:
                    response = await self._call_model(fallback, request)
                    # 记录降级事件用于监控
                    await self._log_fallback(model, fallback, e)
                    return response
                except Exception:
                    continue
            raise AllModelsFailed(request_id=request.id)
```

### 3.3 数据库Schema隔离策略

数据隔离是多租户AI系统安全性的基石。常见的三种策略对比：

| 策略 | 隔离级别 | 运维复杂度 | 成本 | 适用场景 |
|------|---------|-----------|------|---------|
| **行级隔离** | 低 | 低 | 低 | 租户少、信任度高 |
| **Schema隔离** | 中 | 中 | 中 | 中等规模SaaS |
| **数据库隔离** | 高 | 高 | 高 | 金融/医疗/合规要求 |

**推荐：Schema隔离作为默认方案**

```sql
-- 租户A的Schema
CREATE SCHEMA tenant_abc;

CREATE TABLE tenant_abc.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    title VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    model_used VARCHAR(100),
    total_tokens INTEGER,
    cost_usd DECIMAL(10, 6)
);

CREATE TABLE tenant_abc.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES tenant_abc.conversations(id),
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 向量数据库按Collection隔离
-- 使用Qdrant示例
# POST /collections/kb_tenant_abc/points
# 每个租户独立的Collection，物理隔离
```

### 3.4 租户间知识库隔离

AI系统的多租户隔离不仅仅是数据层面，还涉及**知识库**和**模型微调**的隔离：

```
┌──────────────────────────────────────────────────────────────┐
│              租户级知识库隔离架构                               │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │           统一RAG服务层                            │        │
│  │                                                  │        │
│  │  POST /rag/query                                 │        │
│  │  {                                               │        │
│  │    "tenant_id": "abc",                            │        │
│  │    "query": "我们的退款政策是什么？",                │        │
│  │    "knowledge_base_id": "kb_return_policy"        │        │
│  │  }                                               │        │
│  └──────────────────────┬───────────────────────────┘        │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │          租户级向量存储隔离                         │        │
│  │                                                  │        │
│  │  ┌──────────────────┐  ┌──────────────────┐      │        │
│  │  │ 租户A: Qdrant    │  │ 租户B: Qdrant    │      │        │
│  │  │ Collection:      │  │ Collection:      │      │        │
│  │  │  kb_return       │  │  kb_product      │      │        │
│  │  │  kb_faq          │  │  kb_tech_docs    │      │        │
│  │  │  kb_manual       │  │  kb_onboarding   │      │        │
│  │  │ (独立实例/命名空间)│  │ (独立实例/命名空间)│      │        │
│  │  └──────────────────┘  └──────────────────┘      │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │          租户级Prompt模板隔离                      │        │
│  │                                                  │        │
│  │  租户A: "你是{company_name}的客服助手..."          │        │
│  │  租户B: "你是{company_name}的技术支持专家..."      │        │
│  │                                                  │        │
│  │  模板存储在租户Schema中，支持版本管理               │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

## 四、阶段3：资源池化架构（规模化中期）

### 4.1 架构设计

当租户数量达到50-500时，简单的逻辑隔离已经不够了。需要引入**GPU资源池化**和**智能调度**，实现真正的资源共享和动态分配。

```
┌──────────────────────────────────────────────────────────────┐
│                 阶段3: 资源池化架构                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              智能调度层 (Intelligent Scheduler)    │        │
│  │                                                  │        │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │        │
│  │  │ 工作负载预测 │  │ GPU资源分配 │  │ 成本优化    │ │        │
│  │  │ (ML-based) │  │ (Bin Pack) │  │ (Spot实例) │ │        │
│  │  └────────────┘  └────────────┘  └────────────┘ │        │
│  └──────────────────────┬───────────────────────────┘        │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │              GPU资源池 (GPU Resource Pool)         │        │
│  │                                                  │        │
│  │  ┌────────────────────────────────────────────┐  │        │
│  │  │  GPU Cluster A (On-Demand)                 │  │        │
│  │  │  8x A100 80GB                             │  │        │
│  │  │  用途: 高优先级租户 + 复杂推理任务            │  │        │
│  │  └────────────────────────────────────────────┘  │        │
│  │                                                  │        │
│  │  ┌────────────────────────────────────────────┐  │        │
│  │  │  GPU Cluster B (Spot/抢占式)               │  │        │
│  │  │  16x A100 40GB                            │  │        │
│  │  │  用途: 批量推理 + 低优先级租户               │  │        │
│  │  └────────────────────────────────────────────┘  │        │
│  │                                                  │        │
│  │  ┌────────────────────────────────────────────┐  │        │
│  │  │  GPU Cluster C (自建/混合)                  │  │        │
│  │  │  4x H100 80GB                             │  │        │
│  │  │  用途: 微调训练 + 企业租户专属               │  │        │
│  │  └────────────────────────────────────────────┘  │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              推理服务框架层                         │        │
│  │                                                  │        │
│  │  vLLM Instance 1: Llama-3-70B (租户A,C,E共享)    │        │
│  │  vLLM Instance 2: GPT-4o-mini (租户B,D,F共享)    │        │
│  │  vLLM Instance 3: DeepSeek-V3 (租户G,H,I共享)    │        │
│  │                                                  │        │
│  │  通过LoRA热插拔实现租户级模型定制:                   │        │
│  │  vLLM Instance 4: Llama-3-70B base              │        │
│  │    ├── LoRA: tenant_A_adapter (客服微调)          │        │
│  │    ├── LoRA: tenant_B_adapter (法律微调)          │        │
│  │    └── LoRA: tenant_C_adapter (医疗微调)          │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 GPU资源调度器

资源池化的核心是调度器——它决定哪个租户的请求在哪个GPU上执行：

```python
from dataclasses import dataclass
from enum import Enum
import asyncio

class ComputePriority(Enum):
    CRITICAL = 100   # 企业客户实时请求
    HIGH = 75        # Pro客户实时请求
    MEDIUM = 50      # 普通请求
    LOW = 25         # 批量推理
    BACKGROUND = 0   # 异步任务

@dataclass
class GPUResource:
    gpu_id: str
    cluster: str  # on-demand / spot / dedicated
    model_loaded: str  # 当前加载的模型
    vram_used: float  # 已用VRAM (GB)
    vram_total: float  # 总VRAM (GB)
    queue_depth: int  # 排队请求数
    
    @property
    def available_vram(self):
        return self.vram_total - self.vram_used
    
    @property
    def load_score(self):
        """综合负载评分 (0-100, 越低越好)"""
        vram_score = (self.vram_used / self.vram_total) * 60
        queue_score = min(self.queue_depth / 10, 1) * 40
        return vram_score + queue_score


class GPUScheduler:
    """GPU资源调度器"""
    
    def __init__(self):
        self.gpu_pool: list[GPUResource] = []
        self.pending_requests: dict[ComputePriority, asyncio.Queue] = {
            p: asyncio.Queue() for p in ComputePriority
        }
    
    async def schedule(self, request: InferenceRequest, tenant: TenantConfig):
        """调度推理请求到最优GPU"""
        
        priority = self._compute_priority(tenant)
        
        # 1. 查找可用GPU (模型兼容 + 有空余VRAM)
        candidates = self._find_compatible_gpus(
            model=request.model,
            vram_needed=self._estimate_vram(request),
            tenant_id=tenant.id
        )
        
        if not candidates:
            # 无可用GPU，加入等待队列
            await self.pending_requests[priority].put(request)
            return await self._wait_for_gpu(request, timeout=30)
        
        # 2. 选择最优GPU (最低负载 + 最近邻)
        best_gpu = min(candidates, key=lambda g: g.load_score)
        
        # 3. 如果需要LoRA切换，评估切换开销
        if best_gpu.model_loaded != request.model:
            switch_cost = self._estimate_lora_switch_cost(
                best_gpu, request.model, tenant.lora_adapter
            )
            # 如果切换开销太高，找其他候选
            if switch_cost > 5:  # 5秒阈值
                alternatives = [g for g in candidates 
                               if g.load_score < 80 and g.gpu_id != best_gpu.gpu_id]
                if alternatives:
                    best_gpu = min(alternatives, key=lambda g: g.load_score)
        
        # 4. 执行推理
        response = await self._execute_on_gpu(best_gpu, request, tenant)
        
        # 5. 更新GPU状态
        best_gpu.queue_depth -= 1
        
        return response
    
    def _compute_priority(self, tenant: TenantConfig) -> ComputePriority:
        """根据租户等级和请求类型计算优先级"""
        base_priority = {
            "enterprise": ComputePriority.HIGH,
            "pro": ComputePriority.MEDIUM,
            "free": ComputePriority.LOW,
        }[tenant.plan]
        
        # 交互式请求提升优先级
        if tenant.request_type == "interactive":
            return ComputePriority(min(base_priority.value + 25, 100))
        
        return base_priority
    
    def _find_compatible_gpus(self, model, vram_needed, tenant_id):
        """查找兼容的GPU"""
        return [
            gpu for gpu in self.gpu_pool
            if gpu.available_vram >= vram_needed
            and (gpu.model_loaded == model or gpu.available_vram >= vram_needed * 1.5)
            and gpu.load_score < 90
        ]
```

### 4.3 LoRA热插拔：租户级模型定制

资源池化架构中一个关键技术是**LoRA热插拔**——在同一个基础模型上，通过动态加载不同的LoRA适配器来满足租户的定制化需求：

```
┌──────────────────────────────────────────────────────────────┐
│              LoRA热插拔架构                                   │
│                                                              │
│  基础模型: Llama-3-70B (固定加载在GPU上, ~140GB)              │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │           LoRA适配器注册表                         │        │
│  │                                                  │        │
│  │  tenant_A: /models/tenant_a_customer_service.lora │        │
│  │  tenant_B: /models/tenant_b_legal_expert.lora    │        │
│  │  tenant_C: /models/tenant_c_medical.lora         │        │
│  │                                                  │        │
│  │  每个LoRA适配器大小: 100MB - 2GB                   │        │
│  │  加载时间: 0.5s - 3s                              │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  请求流程:                                                    │
│  1. 收到 tenant_A 的请求                                      │
│  2. 检查当前GPU上加载的LoRA                                    │
│  3. 如果不是 tenant_A 的 → 卸载当前LoRA                       │
│  4. 加载 tenant_A 的LoRA (0.5-3s)                            │
│  5. 执行推理                                                  │
│                                                              │
│  优化: 多GPU场景下，可以在不同GPU上预加载不同的LoRA             │
│  GPU-1: 预加载 tenant_A (高频用户)                             │
│  GPU-2: 预加载 tenant_B (高频用户)                             │
│  GPU-3: 通用池 (动态切换)                                     │
└──────────────────────────────────────────────────────────────┘
```

```python
# vLLM LoRA热插拔配置
# vllm serve 起始配置
"""
vllm serve meta-llama/Llama-3-70B \
    --enable-lora \
    --lora-modules \
        tenant_a=/models/tenant_a_customer_service.lora,\
        tenant_b=/models/tenant_b_legal_expert.lora,\
        tenant_c=/models/tenant_c_medical.lora \
    --max-lora-rank 64 \
    --max-loras 4 \
    --gpu-memory-utilization 0.9
"""

# 请求时指定使用哪个LoRA
import openai

client = openai.OpenAI(base_url="http://gpu-cluster:8000/v1")

# 租户A的请求 — 自动使用customer_service LoRA
response = client.chat.completions.create(
    model="meta-llama/Llama-3-70B",  # 基础模型
    messages=[{"role": "user", "content": "如何申请退款？"}],
    extra_body={
        "model": "tenant_a",  # 路由到租户A的LoRA
    }
)
```

## 五、阶段4：弹性隔离架构（成熟期）

### 5.1 架构设计

当租户数量超过500，或者服务金融、医疗等高合规行业时，需要进入**弹性隔离**阶段——每个核心租户拥有独立的计算资源，同时在低峰期共享资源以降低成本。

```
┌──────────────────────────────────────────────────────────────┐
│                 阶段4: 弹性隔离架构                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              弹性资源管理器 (Elastic Resource Mgr) │        │
│  │                                                  │        │
│  │  ┌────────────────┐  ┌────────────────────────┐  │        │
│  │  │ 工作负载预测    │  │ 弹性伸缩策略             │  │        │
│  │  │ (时间序列ML)   │  │ (HPA + VPA + Predictive)│  │        │
│  │  └────────────────┘  └────────────────────────┘  │        │
│  └──────────────────────┬───────────────────────────┘        │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │              租户级资源组 (Tenant Resource Groups) │        │
│  │                                                  │        │
│  │  ┌────────────────────────────────────────────┐  │        │
│  │  │  企业租户A: 专属资源组                       │  │        │
│  │  │  ├── 4x H100 (常驻)                        │  │        │
│  │  │  ├── 独立向量数据库实例                      │  │        │
│  │  │  ├── 独立Redis缓存                          │  │        │
│  │  │  ├── 网络隔离 (VPC Peering)                 │  │        │
│  │  │  └── 独立监控和告警                          │  │        │
│  │  └────────────────────────────────────────────┘  │        │
│  │                                                  │        │
│  │  ┌────────────────────────────────────────────┐  │        │
│  │  │  企业租户B: 专属资源组                       │  │        │
│  │  │  ├── 2x A100 (常驻)                        │  │        │
│  │  │  ├── 共享向量数据库 (独立Namespace)          │  │        │
│  │  │  └── 独立日志存储                            │  │        │
│  │  └────────────────────────────────────────────┘  │        │
│  │                                                  │        │
│  │  ┌────────────────────────────────────────────┐  │        │
│  │  │  共享资源池 (中小租户共享)                    │  │        │
│  │  │  ├── 8x A100 (弹性伸缩)                    │  │        │
│  │  │  ├── 资源隔离: Kubernetes Namespace          │  │        │
│  │  │  ├── 网络隔离: NetworkPolicy                 │  │        │
│  │  │  └── 存储隔离: PVC per tenant               │  │        │
│  │  └────────────────────────────────────────────┘  │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              Kubernetes编排层                      │        │
│  │                                                  │        │
│  │  Namespace: tenant-a-dedicated                   │        │
│  │  ├── Deployment: inference-service (replicas: 4)  │        │
│  │  ├── HPA: min=4, max=8, target=70% GPU util     │        │
│  │  ├── NetworkPolicy: 只允许 tenant-a 网段访问      │        │
│  │  └── PVC: 10TB encrypted storage                 │        │
│  │                                                  │        │
│  │  Namespace: shared-pool                          │        │
│  │  ├── Deployment: inference-service (replicas: 4-16)│        │
│  │  ├── HPA: min=4, max=16, target=80% GPU util    │        │
│  │  ├── NetworkPolicy: 允许所有中小租户访问           │        │
│  │  └── ResourceQuota: per-namespace limits         │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 弹性伸缩策略

弹性伸缩是阶段4的核心能力——在保证服务质量的前提下，动态调整GPU资源以优化成本：

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ScalingPolicy:
    """弹性伸缩策略"""
    tenant_id: str
    
    # 预测性伸缩 (基于历史数据)
    predictive_enabled: bool
    prediction_lookback_days: int  # 历史数据回溯天数
    
    # 反应性伸缩 (基于实时指标)
    reactive_enabled: bool
    scale_up_threshold: float   # GPU利用率阈值 (触发扩容)
    scale_down_threshold: float # GPU利用率阈值 (触发缩容)
    cooldown_seconds: int       # 伸缩冷却时间
    
    # 成本约束
    max_instances: int          # 最大实例数
    min_instances: int          # 最小实例数 (保底)
    monthly_budget_usd: float   # 月度预算


class ElasticResourceManager:
    """弹性资源管理器"""
    
    def __init__(self, k8s_client, metrics_client):
        self.k8s = k8s_client
        self.metrics = metrics_client
    
    async def evaluate_scaling(self, tenant_id: str):
        """评估并执行伸缩决策"""
        policy = await self.get_policy(tenant_id)
        
        # 1. 预测性伸缩: 基于历史模式预测未来负载
        predicted_load = await self._predict_load(
            tenant_id, 
            lookback=policy.prediction_lookback_days,
            horizon=timedelta(hours=4)  # 预测未来4小时
        )
        
        # 2. 反应性伸缩: 基于当前指标
        current_metrics = await self.metrics.get_tenant_metrics(tenant_id)
        
        # 3. 综合决策
        desired_replicas = self._compute_desired_replicas(
            predicted_load=predicted_load,
            current_metrics=current_metrics,
            policy=policy
        )
        
        # 4. 应用预算约束
        estimated_cost = self._estimate_cost(desired_replicas, tenant_id)
        if estimated_cost > policy.monthly_budget_usd:
            desired_replicas = self._reduce_within_budget(
                desired_replicas, policy.monthly_budget_usd, tenant_id
            )
        
        # 5. 执行伸缩
        current_replicas = await self.k8s.get_replicas(
            namespace=f"tenant-{tenant_id}",
            deployment="inference-service"
        )
        
        if desired_replicas != current_replicas:
            await self.k8s.scale(
                namespace=f"tenant-{tenant_id}",
                deployment="inference-service",
                replicas=min(desired_replicas, policy.max_instances)
            )
            
            # 记录伸缩事件
            await self._log_scaling_event(
                tenant_id, current_replicas, desired_replicas, 
                reason=self._explain_decision(predicted_load, current_metrics)
            )
    
    def _compute_desired_replicas(self, predicted_load, current_metrics, policy):
        """计算期望的副本数"""
        if policy.predictive_enabled and predicted_load > 0.7:
            # 预测到未来负载上升，提前扩容
            return min(
                int(predicted_load * policy.max_instances * 1.2),  # 多预留20%
                policy.max_instances
            )
        
        if policy.reactive_enabled:
            if current_metrics.gpu_utilization > policy.scale_up_threshold:
                # 当前负载高，扩容
                return min(
                    current_metrics.current_replicas + 2,
                    policy.max_instances
                )
            elif current_metrics.gpu_utilization < policy.scale_down_threshold:
                # 当前负载低，缩容
                return max(
                    current_metrics.current_replicas - 1,
                    policy.min_instances
                )
        
        return current_metrics.current_replicas  # 保持不变
```

### 5.3 网络隔离与安全

在弹性隔离架构中，租户间的网络隔离至关重要：

```yaml
# Kubernetes NetworkPolicy — 租户A的网络隔离
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tenant-a-isolation
  namespace: tenant-a-dedicated
spec:
  podSelector:
    matchLabels:
      app: inference-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # 只允许来自API Gateway的流量
    - from:
        - namespaceSelector:
            matchLabels:
              name: api-gateway
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # 允许访问LLM Provider API
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
    # 允许访问租户A的数据库
    - to:
        - namespaceSelector:
            matchLabels:
              tenant: tenant-a
---
# 资源配额 — 限制每个租户的资源使用
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-a-quota
  namespace: tenant-a-dedicated
spec:
  hard:
    requests.nvidia.com/gpu: "8"       # 最多8个GPU
    limits.nvidia.com/gpu: "16"        # 峰值可突增到16个
    requests.cpu: "32"
    requests.memory: "128Gi"
    persistentvolumeclaims: "10"
```

## 六、成本分摊与计费架构

多租户AI系统的成本分摊是一个复杂但关键的问题。Token消耗、GPU时长、存储占用、网络流量，每一项都需要精确计量和公平分摊。

```
┌──────────────────────────────────────────────────────────────┐
│              多维度成本计费架构                                 │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              用量采集层 (Usage Collection)         │        │
│  │                                                  │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │        │
│  │  │Token计量  │ │GPU时长计量│ │ 存储/网络流量计量 │ │        │
│  │  │(每请求)   │ │(每任务)  │ │ (实时统计)        │ │        │
│  │  └─────┬────┘ └─────┬────┘ └────────┬─────────┘ │        │
│  └────────┼────────────┼───────────────┼────────────┘        │
│           │            │               │                     │
│  ┌────────▼────────────▼───────────────▼────────────┐        │
│  │              成本计算引擎 (Cost Engine)            │        │
│  │                                                  │        │
│  │  输入:                                           │        │
│  │  ├── input_tokens × model_input_price            │        │
│  │  ├── output_tokens × model_output_price          │        │
│  │  ├── gpu_hours × gpu_type_price                  │        │
│  │  ├── storage_gb × storage_price                  │        │
│  │  └── network_gb × egress_price                   │        │
│  │                                                  │        │
│  │  输出:                                           │        │
│  │  ├── 实时费用仪表盘                               │        │
│  │  ├── 租户级账单                                   │        │
│  │  ├── 异常消费告警                                 │        │
│  │  └── 成本优化建议                                 │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              定价模型 (Pricing Models)              │        │
│  │                                                  │        │
│  │  Model 1: 按量付费 (Pay-per-Use)                  │        │
│  │  └── input: $1/M tokens, output: $3/M tokens     │        │
│  │                                                  │        │
│  │  Model 2: 包月套餐 (Subscription)                  │        │
│  │  └── Pro: $99/月, 含1M tokens, 超出按量计费        │        │
│  │                                                  │        │
│  │  Model 3: 资源预留 (Reserved)                     │        │
│  │  └── Enterprise: $5000/月, 专属GPU + 无限tokens   │        │
│  │                                                  │        │
│  │  Model 4: 混合计费 (Hybrid)                       │        │
│  │  └── 基础费 + 按量费 + 性能溢价                    │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

```python
# 成本计量与计费核心逻辑
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class UsageRecord:
    tenant_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    gpu_type: str | None  # 自部署模型才有
    gpu_hours: float | None
    storage_bytes: int
    network_bytes: int

class CostEngine:
    """成本计算引擎"""
    
    # 模型定价表 (每百万Token)
    MODEL_PRICING = {
        "gpt-4o": {"input": Decimal("2.50"), "output": Decimal("10.00")},
        "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
        "claude-4-sonnet": {"input": Decimal("3.00"), "output": Decimal("15.00")},
        "deepseek-v3": {"input": Decimal("0.27"), "output": Decimal("1.10")},
        "llama-3-70b": {"input": Decimal("0.00"), "output": Decimal("0.00")},  # 自部署
    }
    
    # GPU定价 (每小时)
    GPU_PRICING = {
        "H100": Decimal("3.50"),
        "A100-80GB": Decimal("2.10"),
        "A100-40GB": Decimal("1.50"),
        "L4": Decimal("0.50"),
    }
    
    def calculate_cost(self, record: UsageRecord) -> Decimal:
        """计算单次请求的成本"""
        total = Decimal("0")
        
        # 1. API模型费用
        if record.model in self.MODEL_PRICING:
            pricing = self.MODEL_PRICING[record.model]
            total += (Decimal(record.input_tokens) / 1_000_000) * pricing["input"]
            total += (Decimal(record.output_tokens) / 1_000_000) * pricing["output"]
        
        # 2. GPU费用 (自部署模型)
        if record.gpu_type and record.gpu_hours:
            total += Decimal(record.gpu_hours) * self.GPU_PRICING.get(record.gpu_type, Decimal("0"))
        
        # 3. 存储费用 (每月$0.023/GB)
        storage_gb = Decimal(record.storage_bytes) / (1024 ** 3)
        total += storage_gb * Decimal("0.023") / 30  # 按天分摊
        
        # 4. 网络费用 (出站$0.09/GB)
        network_gb = Decimal(record.network_bytes) / (1024 ** 3)
        total += network_gb * Decimal("0.09")
        
        return total.quantize(Decimal("0.000001"))
    
    async def generate_tenant_bill(self, tenant_id: str, period: str):
        """生成租户账单"""
        records = await self.get_usage_records(tenant_id, period)
        
        bill = {
            "tenant_id": tenant_id,
            "period": period,
            "items": [],
            "total": Decimal("0"),
        }
        
        # 按模型分组统计
        model_usage = defaultdict(lambda: {"input": 0, "output": 0, "cost": Decimal("0")})
        
        for record in records:
            cost = self.calculate_cost(record)
            model_usage[record.model]["input"] += record.input_tokens
            model_usage[record.model]["output"] += record.output_tokens
            model_usage[record.model]["cost"] += cost
            bill["total"] += cost
        
        for model, usage in model_usage.items():
            bill["items"].append({
                "model": model,
                "input_tokens": usage["input"],
                "output_tokens": usage["output"],
                "cost": str(usage["cost"]),
            })
        
        bill["total"] = str(bill["total"])
        return bill
```

## 七、架构选型指南

### 7.1 阶段匹配矩阵

| 阶段 | 租户数 | 团队规模 | 投入预算 | 适用场景 |
|------|--------|---------|---------|---------|
| **阶段1: 单体共享** | 1-5 | 1-3人 | <$10K | MVP验证、内部工具 |
| **阶段2: 逻辑隔离** | 5-50 | 3-8人 | $10K-50K | 早期SaaS、B端产品 |
| **阶段3: 资源池化** | 50-500 | 8-20人 | $50K-200K | 规模化SaaS、AI平台 |
| **阶段4: 弹性隔离** | 500+ | 20+人 | $200K+ | 企业级平台、合规行业 |

### 7.2 关键决策点

```
                        你的AI应用需要多租户吗？
                               │
                    ┌──────────┴──────────┐
                    │                     │
               仅内部使用              对外提供服务
                    │                     │
              阶段1足够             需要数据隔离吗？
              (单体共享)                │
                              ┌────────┴────────┐
                              │                 │
                         基础隔离            强隔离
                         (数据脱敏)         (合规要求)
                              │                 │
                         阶段2              阶段4
                        (逻辑隔离)        (弹性隔离)
                              │
                     需要GPU资源共享吗？
                              │
                     ┌────────┴────────┐
                     │                 │
                    是                 否
                     │                 │
                  阶段3              阶段2
                (资源池化)         (逻辑隔离)
```

## 八、总结

### 8.1 核心原则

1. **隔离是渐进的**：不要一开始就设计最复杂的架构，从单体共享开始，根据实际需求逐步演进
2. **成本感知优先**：AI系统的成本结构与传统SaaS完全不同，必须从第一天就建立精确的成本计量
3. **弹性是必需品**：GPU资源昂贵，必须通过弹性伸缩来优化成本，不能像传统应用那样预留固定资源
4. **安全贯穿始终**：数据隔离不是可选项，特别是面对企业客户时，租户间的数据安全是底线

### 8.2 技术栈推荐

| 阶段 | 计算 | 存储 | 调度 | 监控 |
|------|------|------|------|------|
| 阶段1 | FastAPI | PostgreSQL | 手动 | 日志 |
| 阶段2 | FastAPI + Redis | PG Schema隔离 | 简单限流 | Prometheus |
| 阶段3 | vLLM + Kubernetes | PG + Qdrant | K8s HPA + 自定义 | Prometheus + Grafana |
| 阶段4 | K8s + GPU Operator | 多实例 + 加密 | Predictive HPA | 全链路可观测性 |

多租户AI系统的架构设计没有银弹，关键是理解你的业务阶段和客户特征，选择匹配的架构方案，然后在演进中持续优化。

---

*本文基于2026年AI基础设施最佳实践撰写，涉及的技术方案已在多个生产环境中验证。*
