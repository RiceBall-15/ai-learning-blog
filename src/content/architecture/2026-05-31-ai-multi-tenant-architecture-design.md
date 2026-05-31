---
title: "AI应用多租户架构设计：从数据隔离到模型共享的完整方案"
description: "系统性剖析AI应用多租户架构的核心挑战与解决方案，覆盖数据隔离、模型路由、成本分摊与安全合规的生产级设计"
date: 2026-05-31
author: "RiceBall"
category: "architecture"
subCategory: "distributed"
tags: ["多租户", "AI架构", "SaaS", "数据隔离", "模型路由", "成本优化"]
draft: false
---

# AI应用多租户架构设计：从数据隔离到模型共享的完整方案

> "多租户不是把数据库加个tenant_id字段那么简单——尤其当你的租户在用大模型的时候。"

在传统SaaS架构中，多租户设计已经有成熟的模式：共享数据库+行级隔离、独立Schema、独立数据库。但当AI能力成为产品的核心竞争力时，多租户架构面临全新的挑战：

- **模型资源是有限的**：GPU算力昂贵，不可能为每个租户部署独立模型
- **Prompt和上下文是敏感的**：租户A的提示词不能被租户B看到
- **模型输出是不确定的**：同一个Prompt在不同时间可能产生不同结果
- **成本需要精确分摊**：Token消耗、API调用次数需要按租户统计

本文将从真实生产经验出发，系统性地剖析AI应用多租户架构的设计原则、核心挑战与完整解决方案。

---

## 目录

1. [AI多租户的三大核心挑战](#一ai多租户的三大核心挑战)
2. [数据隔离：从行级隔离到向量隔离](#二数据隔离从行级隔离到向量隔离)
3. [模型资源共享：路由、调度与隔离](#三模型资源共享路由调度与隔离)
4. [Prompt与上下文的安全隔离](#四prompt与上下文的安全隔离)
5. [成本分摊与计量](#五成本分摊与计量)
6. [生产级架构设计](#六生产级架构设计)
7. [实战案例：三种典型的多租户AI架构](#七实战案例三种典型的多租户ai架构)
8. [性能优化与扩展策略](#八性能优化与扩展策略)

---

## 一、AI多租户的三大核心挑战

### 1.1 挑战一：模型资源的共享与隔离

```
传统SaaS多租户：
├── 数据库：可以轻松为每个租户创建独立Schema
├── 应用服务器：无状态，水平扩展简单
└── 存储：按租户分目录，隔离清晰

AI应用多租户：
├── 模型服务：GPU昂贵，共享是必然选择
├── 向量数据库：embedding模型需要共享，但数据必须隔离
├── 缓存层：KV Cache、Prompt缓存需要隔离
└── 推理队列：需要按租户优先级调度
```

### 1.2 挑战二：Prompt与知识的安全边界

传统应用的数据隔离相对简单——数据库层面的行级安全策略就能解决大部分问题。但AI应用的"数据"不仅存储在数据库中，还流动在：

| 数据类型 | 存储位置 | 隔离难度 |
|---------|---------|---------|
| 用户数据 | 数据库 | 低（行级隔离） |
| Prompt模板 | 配置/数据库 | 中（需要版本控制） |
| 系统提示词 | 内存/配置 | 高（运行时注入） |
| RAG知识库 | 向量数据库 | 高（向量空间隔离） |
| 对话历史 | 缓存/数据库 | 中（需要加密） |
| 模型微调数据 | 训练存储 | 极高（物理隔离） |

### 1.3 挑战三：成本的精确分摊

```
AI应用成本构成：
├── 推理成本 (60-80%)
│   ├── Token消耗（按租户计量）
│   ├── GPU时间（共享资源池）
│   └── API调用（第三方模型）
├── 存储成本 (10-20%)
│   ├── 向量数据库
│   ├── 对话历史
│   └── 知识库
└── 网络成本 (5-10%)
    ├── 内部调用
    └── 外部API
```

---

## 二、数据隔离：从行级隔离到向量隔离

### 2.1 关系型数据的行级隔离

对于传统的关系型数据（用户信息、对话记录、配置等），行级隔离是最常用的方案：

```sql
-- 方案一：行级安全策略（RLS）- PostgreSQL
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON conversations
    USING (tenant_id = current_setting('app.current_tenant'));

-- 方案二：应用层过滤
-- 所有查询必须包含 tenant_id 条件
SELECT * FROM conversations 
WHERE tenant_id = 'tenant_123' 
AND user_id = 'user_456';
```

**RLS方案的优缺点**：

| 维度 | 优点 | 缺点 |
|-----|------|------|
| 安全性 | 数据库层面强制隔离 | 需要正确配置策略 |
| 性能 | 索引可优化 | 大表可能有性能影响 |
| 维护 | 应用层无需关心 | 策略管理复杂 |
| 审计 | 可追踪违规访问 | 需要额外的审计日志 |

### 2.2 向量数据库的租户隔离

向量数据库（如Qdrant、Milvus、Weaviate）的租户隔离需要特别设计：

```python
class VectorDBMultiTenant:
    """向量数据库多租户管理"""
    
    def __init__(self, vector_db_client):
        self.client = vector_db_client
    
    # ========== 方案一：Collection级别隔离 ==========
    
    async def setup_tenant_collection(self, tenant_id: str):
        """为租户创建独立的Collection"""
        collection_name = f"knowledge_{tenant_id}"
        
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 1536,  # embedding维度
                "distance": "Cosine"
            }
        )
        
        # 创建租户元数据索引
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata",
            field_schema={"type": "keyword"}
        )
    
    async def search_tenant(
        self, 
        tenant_id: str, 
        query_vector: list,
        top_k: int = 10
    ):
        """在租户的Collection中搜索"""
        collection_name = f"knowledge_{tenant_id}"
        
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return results
    
    # ========== 方案二：Payload过滤隔离 ==========
    
    async def setup_shared_collection(self):
        """共享Collection + Payload过滤"""
        # 创建一个大的共享Collection
        collection_name = "knowledge_shared"
        
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 1536,
                "distance": "Cosine"
            }
        )
        
        # 创建tenant_id索引
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="tenant_id",
            field_schema={"type": "keyword"}
        )
    
    async def search_shared(
        self,
        tenant_id: str,
        query_vector: list,
        top_k: int = 10
    ):
        """在共享Collection中按租户过滤搜索"""
        results = await self.client.search(
            collection_name="knowledge_shared",
            query_vector=query_vector,
            limit=top_k,
            query_filter={
                "must": [
                    {"key": "tenant_id", "match": {"value": tenant_id}}
                ]
            }
        )
        
        return results
```

**两种方案的对比**：

| 方案 | 隔离级别 | 性能 | 维护成本 | 适用场景 |
|-----|---------|------|---------|---------|
| Collection隔离 | 完全隔离 | 高（小Collection更快） | 高（Collection数量多） | 租户数量少（<100） |
| Payload过滤 | 逻辑隔离 | 中（大Collection） | 低（统一管理） | 租户数量多（>100） |

### 2.3 混合隔离策略

生产环境中，通常需要混合使用多种隔离策略：

```python
class HybridIsolationManager:
    """混合隔离策略管理器"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 大租户：独立Collection
        self.large_tenants = set(config.get("large_tenants", []))
        
        # 小租户：共享Collection
        self.small_tenant_collection = "knowledge_shared"
    
    async def get_search_strategy(self, tenant_id: str) -> dict:
        """根据租户规模选择隔离策略"""
        
        if tenant_id in self.large_tenants:
            return {
                "strategy": "dedicated",
                "collection": f"knowledge_{tenant_id}",
                "max_results": 100
            }
        else:
            return {
                "strategy": "shared",
                "collection": self.small_tenant_collection,
                "filter": {"tenant_id": tenant_id},
                "max_results": 50
            }
    
    async def search(
        self,
        tenant_id: str,
        query_vector: list,
        top_k: int = 10
    ):
        """统一的搜索接口"""
        strategy = await self.get_search_strategy(tenant_id)
        
        if strategy["strategy"] == "dedicated":
            return await self._search_dedicated(
                strategy["collection"],
                query_vector,
                top_k
            )
        else:
            return await self._search_shared(
                strategy["collection"],
                strategy["filter"],
                query_vector,
                top_k
            )
```

---

## 三、模型资源共享：路由、调度与隔离

### 3.1 模型资源的三层共享模型

```
┌─────────────────────────────────────────────────────────────────┐
│                    模型资源共享架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: 基础模型共享                                           │
│  ├── 所有租户共享同一个基础模型（如GPT-4、Claude）               │
│  ├── 通过API Key隔离（不同租户不同Key）                         │
│  └── 成本：按Token计费                                          │
│                                                                 │
│  Layer 2: 微调模型共享                                          │
│  ├── 为特定行业/场景微调的模型                                  │
│  ├── 多个租户共享微调模型                                       │
│  └── 成本：GPU租赁 + 微调成本                                   │
│                                                                 │
│  Layer 3: 租户专属模型                                          │
│  ├── 大客户专属的微调模型                                       │
│  ├── 独立部署，完全隔离                                        │
│  └── 成本：专属GPU + 维护成本                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 智能模型路由器

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio

class ModelTier(Enum):
    BASIC = "basic"           # 基础模型（如GPT-3.5）
    ADVANCED = "advanced"     # 高级模型（如GPT-4）
    PREMIUM = "premium"       # 高级模型（如Claude Opus）
    CUSTOM = "custom"         # 租户专属模型

@dataclass
class TenantModelConfig:
    """租户模型配置"""
    tenant_id: str
    tier: ModelTier
    allowed_models: List[str]
    rate_limit: int  # 每分钟请求数
    daily_token_limit: int
    priority: int  # 1-10, 数字越大优先级越高
    
    # 成本控制
    max_cost_per_day: float  # 美元
    alert_threshold: float   # 成本告警阈值（百分比）

class ModelRouter:
    """模型路由器"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 模型服务端点
        self.model_endpoints = {
            "gpt-4": {"endpoint": "https://api.openai.com/v1", "tier": ModelTier.ADVANCED},
            "gpt-3.5-turbo": {"endpoint": "https://api.openai.com/v1", "tier": ModelTier.BASIC},
            "claude-opus": {"endpoint": "https://api.anthropic.com", "tier": ModelTier.PREMIUM},
            "claude-sonnet": {"endpoint": "https://api.anthropic.com", "tier": ModelTier.ADVANCED},
        }
        
        # 租户配置缓存
        self.tenant_configs: Dict[str, TenantModelConfig] = {}
        
        # 限流器
        self.rate_limiters: Dict[str, RateLimiter] = {}
    
    async def route_request(
        self,
        tenant_id: str,
        request: dict,
        preferred_model: str = None
    ) -> dict:
        """路由模型请求"""
        
        # 1. 获取租户配置
        tenant_config = await self.get_tenant_config(tenant_id)
        
        # 2. 检查限流
        if not await self.check_rate_limit(tenant_id):
            raise RateLimitExceeded(f"租户 {tenant_id} 已达到速率限制")
        
        # 3. 检查成本预算
        if not await self.check_cost_budget(tenant_id):
            raise BudgetExceeded(f"租户 {tenant_id} 已达到成本预算上限")
        
        # 4. 选择模型
        model = self.select_model(tenant_config, preferred_model)
        
        # 5. 路由到相应的服务
        endpoint = self.model_endpoints[model]
        
        # 6. 记录使用量
        await self.record_usage(tenant_id, model, request)
        
        return {
            "model": model,
            "endpoint": endpoint["endpoint"],
            "headers": await self.get_auth_headers(tenant_id, model),
            "request": request
        }
    
    def select_model(
        self,
        config: TenantModelConfig,
        preferred: Optional[str] = None
    ) -> str:
        """选择合适的模型"""
        
        # 如果有偏好模型且在允许列表中
        if preferred and preferred in config.allowed_models:
            return preferred
        
        # 根据租户等级选择默认模型
        tier_model_map = {
            ModelTier.BASIC: ["gpt-3.5-turbo"],
            ModelTier.ADVANCED: ["gpt-4", "claude-sonnet"],
            ModelTier.PREMIUM: ["claude-opus", "gpt-4"],
            ModelTier.CUSTOM: config.allowed_models,
        }
        
        available_models = tier_model_map.get(config.tier, ["gpt-3.5-turbo"])
        
        # 选择第一个可用的模型
        for model in available_models:
            if model in config.allowed_models:
                return model
        
        # 降级到基础模型
        return "gpt-3.5-turbo"
    
    async def check_rate_limit(self, tenant_id: str) -> bool:
        """检查速率限制"""
        if tenant_id not in self.rate_limiters:
            config = await self.get_tenant_config(tenant_id)
            self.rate_limiters[tenant_id] = RateLimiter(
                max_requests=config.rate_limit,
                window_seconds=60
            )
        
        return await self.rate_limiters[tenant_id].acquire()
    
    async def check_cost_budget(self, tenant_id: str) -> bool:
        """检查成本预算"""
        config = await self.get_tenant_config(tenant_id)
        current_cost = await self.get_daily_cost(tenant_id)
        
        if current_cost >= config.max_cost_per_day:
            return False
        
        # 检查是否需要告警
        if current_cost >= config.max_cost_per_day * config.alert_threshold:
            await self.send_cost_alert(tenant_id, current_cost)
        
        return True

class RateLimiter:
    """滑动窗口限流器"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    async def acquire(self) -> bool:
        """尝试获取请求许可"""
        now = asyncio.get_event_loop().time()
        
        # 清理过期请求
        self.requests = [
            req_time for req_time in self.requests
            if now - req_time < self.window_seconds
        ]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
```

### 3.3 GPU资源的租户调度

对于自托管的模型服务，需要更精细的GPU资源调度：

```python
class GPUScheduler:
    """GPU资源调度器"""
    
    def __init__(self, gpu_pool: dict):
        """
        gpu_pool: {
            "gpu_0": {"memory": 80, "utilization": 0.3, "tenant": None},
            "gpu_1": {"memory": 80, "utilization": 0.7, "tenant": "tenant_a"},
        }
        """
        self.gpu_pool = gpu_pool
        self.tenant_quotas = {}
    
    async def allocate(
        self,
        tenant_id: str,
        memory_required: int,
        priority: int
    ) -> dict:
        """为租户分配GPU资源"""
        
        # 1. 检查租户配额
        if not self.check_quota(tenant_id, memory_required):
            raise QuotaExceeded(f"租户 {tenant_id} GPU配额不足")
        
        # 2. 查找可用GPU
        available_gpus = [
            gpu_id for gpu_id, info in self.gpu_pool.items()
            if info["memory"] - self.get_used_memory(gpu_id) >= memory_required
        ]
        
        if not available_gpus:
            # 尝试迁移低优先级租户
            await self.evict_low_priority(tenant_id, memory_required)
            available_gpus = self.find_available_gpus(memory_required)
        
        # 3. 选择最优GPU（考虑亲和性、负载均衡）
        best_gpu = self.select_best_gpu(available_gpus, tenant_id, priority)
        
        # 4. 分配资源
        self.gpu_pool[best_gpu]["tenant"] = tenant_id
        self.gpu_pool[best_gpu]["utilization"] += memory_required / 80
        
        return {
            "gpu_id": best_gpu,
            "memory_allocated": memory_required,
            "estimated_ready_time": self.estimate_start_time(best_gpu)
        }
    
    def select_best_gpu(
        self,
        available: List[str],
        tenant_id: str,
        priority: int
    ) -> str:
        """选择最优GPU"""
        
        scores = []
        for gpu_id in available:
            info = self.gpu_pool[gpu_id]
            
            # 计算评分
            score = 0
            
            # 负载均衡：选择利用率较低的
            score += (1 - info["utilization"]) * 0.4
            
            # 亲和性：如果该租户已在该GPU上，加分
            if info.get("tenant") == tenant_id:
                score += 0.3
            
            # 优先级：高优先级租户分配到更好的GPU
            score += priority / 10 * 0.3
            
            scores.append((gpu_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
```

---

## 四、Prompt与上下文的安全隔离

### 4.1 Prompt模板的租户隔离

```python
class PromptIsolationManager:
    """Prompt安全隔离管理器"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Prompt模板存储
        self.global_templates = {}  # 全局模板
        self.tenant_templates = {}  # 租户专属模板
    
    async def get_prompt_template(
        self,
        tenant_id: str,
        template_name: str
    ) -> dict:
        """获取Prompt模板（租户隔离）"""
        
        # 1. 优先查找租户专属模板
        tenant_key = f"{tenant_id}:{template_name}"
        if tenant_key in self.tenant_templates:
            return self.tenant_templates[tenant_key]
        
        # 2. 查找全局模板
        if template_name in self.global_templates:
            template = self.global_templates[template_name]
            
            # 检查租户是否有权限访问
            if self.check_template_access(tenant_id, template):
                return template
        
        # 3. 使用默认模板
        return self.get_default_template(template_name)
    
    async def save_tenant_template(
        self,
        tenant_id: str,
        template_name: str,
        template: dict
    ):
        """保存租户专属模板"""
        
        # 验证模板内容（防止注入）
        validated = self.validate_template(template)
        
        # 加密存储敏感内容
        if self.contains_sensitive_content(template):
            validated = await self.encrypt_template(validated)
        
        # 存储
        tenant_key = f"{tenant_id}:{template_name}"
        self.tenant_templates[tenant_key] = validated
    
    async def compose_prompt(
        self,
        tenant_id: str,
        template_name: str,
        variables: dict
    ) -> str:
        """组装最终的Prompt"""
        
        # 1. 获取模板
        template = await self.get_prompt_template(tenant_id, template_name)
        
        # 2. 注入租户上下文
        tenant_context = await self.get_tenant_context(tenant_id)
        
        # 3. 组装
        prompt = self.render_template(template, variables)
        
        # 4. 添加租户标识（用于审计）
        prompt = self.add_tenant_marker(prompt, tenant_id)
        
        # 5. 安全检查（确保没有泄露其他租户信息）
        if not self.safety_check(prompt, tenant_id):
            raise SecurityViolation("Prompt安全检查失败")
        
        return prompt
    
    def safety_check(self, prompt: str, tenant_id: str) -> bool:
        """Prompt安全检查"""
        
        # 检查是否包含其他租户的标识
        other_tenant_patterns = [
            r"tenant_id\s*[:=]\s*['\"]?(?!['\"]?" + tenant_id + r")\w+",
            # 更多安全检查规则...
        ]
        
        import re
        for pattern in other_tenant_patterns:
            if re.search(pattern, prompt):
                return False
        
        return True
```

### 4.2 RAG知识库的隔离策略

```python
class RAGIsolationManager:
    """RAG知识库隔离管理"""
    
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
    
    async def query_knowledge(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5
    ) -> List[dict]:
        """查询租户知识库"""
        
        # 1. 生成查询向量
        query_embedding = await self.llm.embed(query)
        
        # 2. 构建过滤条件（租户隔离）
        filter_condition = {
            "must": [
                {"key": "tenant_id", "match": {"value": tenant_id}},
                {"key": "status", "match": {"value": "active"}}
            ]
        }
        
        # 3. 执行向量搜索
        results = await self.vector_db.search(
            collection="knowledge_base",
            query_vector=query_embedding,
            query_filter=filter_condition,
            limit=top_k
        )
        
        # 4. 结果后处理（去除敏感信息）
        filtered_results = [
            self.sanitize_result(r, tenant_id)
            for r in results
        ]
        
        return filtered_results
    
    def sanitize_result(self, result: dict, tenant_id: str) -> dict:
        """清理搜索结果中的敏感信息"""
        
        sanitized = result.copy()
        
        # 移除内部元数据
        if "metadata" in sanitized:
            metadata = sanitized["metadata"]
            
            # 保留租户相关的元数据
            safe_keys = ["source", "category", "created_at"]
            sanitized["metadata"] = {
                k: v for k, v in metadata.items()
                if k in safe_keys
            }
        
        # 移除可能的内部ID
        for key in ["_id", "internal_id", "system_id"]:
            sanitized.pop(key, None)
        
        return sanitized
```

---

## 五、成本分摊与计量

### 5.1 成本计量数据模型

```python
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
from enum import Enum

class CostType(Enum):
    INFERENCE = "inference"       # 模型推理
    STORAGE = "storage"          # 存储
    NETWORK = "network"          # 网络
    FINE_TUNING = "fine_tuning"  # 微调
    SUPPORT = "support"          # 技术支持

@dataclass
class UsageRecord:
    """使用量记录"""
    record_id: str
    tenant_id: str
    timestamp: datetime
    
    # 资源使用
    model: str
    input_tokens: int
    output_tokens: int
    
    # 成本信息
    cost_type: CostType
    unit_price: float  # 单价（每1K token）
    total_cost: float  # 总成本
    
    # 元数据
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class TenantCostSummary:
    """租户成本汇总"""
    tenant_id: str
    period: date
    
    # 分项成本
    inference_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    fine_tuning_cost: float = 0.0
    
    # 总成本
    total_cost: float = 0.0
    
    # 使用量统计
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # 预算
    budget_limit: float = 0.0
    budget_used_percent: float = 0.0
```

### 5.2 实时成本追踪

```python
class CostTracker:
    """成本追踪器"""
    
    def __init__(self, redis_client, db_client):
        self.redis = redis_client  # 实时计数
        self.db = db_client        # 持久化存储
    
    async def record_usage(self, record: UsageRecord):
        """记录使用量"""
        
        # 1. 更新实时计数器（Redis）
        today = date.today().isoformat()
        pipe = self.redis.pipeline()
        
        # 按租户统计
        pipe.hincrby(f"cost:{today}", f"{record.tenant_id}:requests", 1)
        pipe.hincrby(f"cost:{today}", f"{record.tenant_id}:tokens", 
                     record.input_tokens + record.output_tokens)
        pipe.hincrbyfloat(f"cost:{today}", f"{record.tenant_id}:cost", 
                         record.total_cost)
        
        # 设置过期时间（保留30天）
        pipe.expire(f"cost:{today}", 30 * 24 * 3600)
        
        await pipe.execute()
        
        # 2. 异步写入数据库（用于报表和审计）
        await self.db.insert_usage_record(record)
        
        # 3. 检查是否需要告警
        await self.check_cost_alerts(record.tenant_id)
    
    async def get_realtime_cost(self, tenant_id: str) -> dict:
        """获取实时成本"""
        today = date.today().isoformat()
        
        data = await self.redis.hgetall(f"cost:{today}")
        
        return {
            "tenant_id": tenant_id,
            "date": today,
            "requests": int(data.get(f"{tenant_id}:requests", 0)),
            "tokens": int(data.get(f"{tenant_id}:tokens", 0)),
            "cost": float(data.get(f"{tenant_id}:cost", 0.0))
        }
    
    async def check_cost_alerts(self, tenant_id: str):
        """检查成本告警"""
        
        # 获取租户配置
        config = await self.get_tenant_config(tenant_id)
        
        # 获取当前成本
        current_cost = await self.get_realtime_cost(tenant_id)
        
        # 检查是否超过阈值
        if current_cost["cost"] > config.daily_budget * 0.8:
            await self.send_alert(
                tenant_id=tenant_id,
                alert_type="cost_warning",
                message=f"今日成本已达预算的{current_cost['cost']/config.daily_budget*100:.1f}%"
            )
        
        if current_cost["cost"] > config.daily_budget:
            await self.send_alert(
                tenant_id=tenant_id,
                alert_type="cost_exceeded",
                message=f"今日成本已超过预算上限"
            )
            
            # 可选：自动降级模型
            if config.auto_downgrade_on_budget:
                await self.downgrade_model_tier(tenant_id)
```

### 5.3 成本分摊报告

```python
class CostReporting:
    """成本报告生成器"""
    
    async def generate_tenant_report(
        self,
        tenant_id: str,
        start_date: date,
        end_date: date
    ) -> dict:
        """生成租户成本报告"""
        
        # 从数据库聚合数据
        query = """
            SELECT 
                DATE(timestamp) as day,
                cost_type,
                SUM(total_cost) as daily_cost,
                SUM(input_tokens + output_tokens) as total_tokens,
                COUNT(*) as request_count
            FROM usage_records
            WHERE tenant_id = %s
            AND timestamp >= %s
            AND timestamp < %s
            GROUP BY DATE(timestamp), cost_type
            ORDER BY day
        """
        
        rows = await self.db.fetch(query, tenant_id, start_date, end_date)
        
        # 按天和类型汇总
        daily_summary = {}
        type_summary = {}
        
        for row in rows:
            day = row["day"]
            cost_type = row["cost_type"]
            
            if day not in daily_summary:
                daily_summary[day] = {"cost": 0, "tokens": 0, "requests": 0}
            daily_summary[day]["cost"] += row["daily_cost"]
            daily_summary[day]["tokens"] += row["total_tokens"]
            daily_summary[day]["requests"] += row["request_count"]
            
            if cost_type not in type_summary:
                type_summary[cost_type] = 0
            type_summary[cost_type] += row["daily_cost"]
        
        # 生成报告
        report = {
            "tenant_id": tenant_id,
            "period": {"start": start_date, "end": end_date},
            "summary": {
                "total_cost": sum(d["cost"] for d in daily_summary.values()),
                "total_tokens": sum(d["tokens"] for d in daily_summary.values()),
                "total_requests": sum(d["requests"] for d in daily_summary.values()),
                "avg_daily_cost": sum(d["cost"] for d in daily_summary.values()) / max(len(daily_summary), 1)
            },
            "by_type": type_summary,
            "daily_breakdown": daily_summary
        }
        
        return report
```

---

## 六、生产级架构设计

### 6.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AI应用多租户架构完整方案                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         接入层                                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │   │
│  │  │ API网关 │  │ 认证服务 │  │ 限流器  │  │ 负载均衡 │           │   │
│  │  │(Kong/   │  │(Auth0/  │  │(Redis)  │  │(Nginx/  │           │   │
│  │  │ APISIX) │  │ Keycloak)│  │         │  │ HAProxy)│           │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         业务层                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │                   租户上下文注入                          │   │   │
│  │  │  • 解析租户ID  • 加载租户配置  • 注入安全上下文         │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                          │                                      │   │
│  │        ┌─────────────────┼─────────────────┐                   │   │
│  │        ▼                 ▼                 ▼                   │   │
│  │  ┌─────────┐      ┌─────────┐      ┌─────────┐                │   │
│  │  │ Agent   │      │  RAG    │      │  Tools  │                │   │
│  │  │ Service │      │ Service │      │ Service │                │   │
│  │  └─────────┘      └─────────┘      └─────────┘                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       模型层                                    │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │                   模型路由器                              │   │   │
│  │  │  • 租户等级映射  • 负载均衡  • 故障转移  • 成本控制     │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                          │                                      │   │
│  │        ┌─────────────────┼─────────────────┐                   │   │
│  │        ▼                 ▼                 ▼                   │   │
│  │  ┌─────────┐      ┌─────────┐      ┌─────────┐                │   │
│  │  │ OpenAI  │      │ Claude  │      │  本地   │                │   │
│  │  │  API    │      │  API    │      │  模型   │                │   │
│  │  └─────────┘      └─────────┘      └─────────┘                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       数据层                                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │   │
│  │  │PostgreSQL│  │  Redis  │  │ Qdrant  │  │   S3    │           │   │
│  │  │(RLS隔离) │  │(缓存隔离)│  │(向量隔离)│  │(文件隔离)│           │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      可观测性层                                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │   │
│  │  │Prometheus│  │  Grafana │  │  Jaeger  │  │  ELK   │           │   │
│  │  │(指标)    │  │(可视化)  │  │(追踪)    │  │(日志)   │           │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 核心服务实现

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Optional
import jwt

app = FastAPI()

class TenantContext:
    """租户上下文"""
    def __init__(self, tenant_id: str, config: dict):
        self.tenant_id = tenant_id
        self.config = config

async def get_tenant_context(
    authorization: str = Depends(...)
) -> TenantContext:
    """从请求中提取租户上下文"""
    
    # 1. 解析JWT Token
    try:
        payload = jwt.decode(authorization, SECRET_KEY, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的认证凭证")
    
    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=403, detail="缺少租户信息")
    
    # 2. 加载租户配置
    config = await load_tenant_config(tenant_id)
    
    # 3. 检查租户状态
    if config.get("status") != "active":
        raise HTTPException(status_code=403, detail="租户已禁用")
    
    return TenantContext(tenant_id=tenant_id, config=config)

@app.post("/api/v1/chat")
async def chat(
    request: ChatRequest,
    tenant: TenantContext = Depends(get_tenant_context)
):
    """多租户聊天接口"""
    
    # 1. 检查使用配额
    await check_usage_quota(tenant.tenant_id, request)
    
    # 2. 获取租户的Prompt模板
    prompt = await get_tenant_prompt(
        tenant.tenant_id, 
        request.template_name
    )
    
    # 3. 注入租户上下文
    context = await inject_tenant_context(
        tenant.tenant_id,
        request.query
    )
    
    # 4. 路由到合适的模型
    model_config = await route_to_model(
        tenant.tenant_id,
        tenant.config["model_tier"]
    )
    
    # 5. 调用模型
    response = await call_model(
        model=model_config["model"],
        prompt=prompt,
        context=context,
        user_message=request.query
    )
    
    # 6. 记录使用量
    await record_usage(
        tenant_id=tenant.tenant_id,
        model=model_config["model"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens
    )
    
    # 7. 返回响应（移除内部信息）
    return sanitize_response(response, tenant.tenant_id)
```

---

## 七、实战案例：三种典型的多租户AI架构

### 7.1 案例一：SaaS AI平台（共享模型）

**场景**：提供AI写作、翻译、总结等功能的SaaS平台

**架构特点**：
- 所有租户共享同一套模型（通过API调用）
- 租户隔离通过API Key和速率限制实现
- 成本按Token使用量计费

```
┌─────────────────────────────────────────────────────────────┐
│                    SaaS AI平台架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  租户A ──┐                                                  │
│          │                                                  │
│  租户B ──┼──► API网关 ──► 租户认证 ──► 限流 ──► 模型路由    │
│          │                                        │         │
│  租户C ──┘                                        ▼         │
│                                            ┌──────────┐     │
│                                            │ OpenAI   │     │
│                                            │  API     │     │
│                                            └──────────┘     │
│                                                             │
│  隔离机制：                                                 │
│  • API Key：每个租户独立的Key                               │
│  • 速率限制：按租户等级设置QPS                              │
│  • 成本配额：每日/每月Token上限                             │
│  • 数据隔离：行级安全（对话记录）                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**实现要点**：

```python
# 简化的SaaS平台租户配置
SAAS_TENANT_CONFIGS = {
    "free": {
        "model": "gpt-3.5-turbo",
        "rate_limit": 10,  # 每分钟10次
        "daily_token_limit": 50000,
        "features": ["chat", "summarize"]
    },
    "pro": {
        "model": "gpt-4",
        "rate_limit": 60,
        "daily_token_limit": 500000,
        "features": ["chat", "summarize", "translate", "write"]
    },
    "enterprise": {
        "model": "gpt-4-turbo",
        "rate_limit": 300,
        "daily_token_limit": 5000000,
        "features": ["all"],
        "custom_templates": True,
        "priority_support": True
    }
}
```

### 7.2 案例二：企业AI中台（混合模型）

**场景**：大型企业的内部AI平台，多个业务部门共享

**架构特点**：
- 共享基础模型 + 部门专属微调模型
- 数据严格隔离（合规要求）
- 成本按部门分摊

```
┌─────────────────────────────────────────────────────────────┐
│                    企业AI中台架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  统一接入层                          │   │
│  │  SSO认证 → 部门识别 → 权限校验 → 请求路由          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐           │
│  │ 研发部    │     │ 市场部    │     │ 财务部    │           │
│  │ 专属模型  │     │ 共享模型  │     │ 共享模型  │           │
│  │ 代码生成  │     │ 文案生成  │     │ 报表分析  │           │
│  └──────────┘     └──────────┘     └──────────┘           │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  模型服务层                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│  │  │ 基础模型 │  │ 微调模型 │  │ 专属模型 │             │   │
│  │  │ (共享)  │  │ (部门)  │  │ (独占)  │             │   │
│  │  └─────────┘  └─────────┘  └─────────┘             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  隔离机制：                                                 │
│  • SSO认证：基于企业目录的用户身份                          │
│  • 部门隔离：数据库Schema级别隔离                           │
│  • 模型隔离：微调模型按部门部署                             │
│  • 审计日志：完整的操作追踪                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 案例三：AI原生产品（租户专属模型）

**场景**：为客户提供定制AI助手的平台，每个客户有专属模型

**架构特点**：
- 每个大客户有专属的微调模型
- 完全的数据和模型隔离
- 成本较高但安全性最强

```
┌─────────────────────────────────────────────────────────────┐
│                   AI原生产品架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  模型编排层                          │   │
│  │                                                     │   │
│  │  客户A ──► 专属模型A (微调+RAG)                     │   │
│  │  客户B ──► 专属模型B (微调+RAG)                     │   │
│  │  客户C ──► 共享模型 (基础版)                        │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  基础设施层                          │   │
│  │                                                     │   │
│  │  客户A：                                          │   │
│  │  ├── 专属GPU集群 (2x A100)                        │   │
│  │  ├── 专属向量数据库                                │   │
│  │  ├── 专属S3存储桶                                  │   │
│  │  └── 独立的监控和日志系统                          │   │
│  │                                                     │   │
│  │  客户B：                                          │   │
│  │  ├── 专属GPU (1x A100)                            │   │
│  │  ├── 共享向量数据库 (隔离Collection)               │   │
│  │  └── 共享监控 (租户维度视图)                       │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  隔离机制：                                                 │
│  • 物理隔离：专属GPU和存储                                 │
│  • 网络隔离：VPC级别的网络分段                             │
│  • 数据隔离：独立的数据库和存储桶                          │
│  • 合规审计：满足金融/医疗级合规要求                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 八、性能优化与扩展策略

### 8.1 缓存策略

```python
class MultiTenantCacheManager:
    """多租户缓存管理"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_cached_response(
        self,
        tenant_id: str,
        request_hash: str
    ) -> Optional[dict]:
        """获取缓存的响应"""
        
        cache_key = f"response:{tenant_id}:{request_hash}"
        cached = await self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        return None
    
    async def cache_response(
        self,
        tenant_id: str,
        request_hash: str,
        response: dict,
        ttl: int = 300
    ):
        """缓存响应"""
        
        cache_key = f"response:{tenant_id}:{request_hash}"
        await self.redis.setex(
            cache_key,
            ttl,
            json.dumps(response)
        )
    
    async def invalidate_tenant_cache(self, tenant_id: str):
        """清除租户的所有缓存"""
        
        pattern = f"response:{tenant_id}:*"
        keys = await self.redis.keys(pattern)
        
        if keys:
            await self.redis.delete(*keys)
```

### 8.2 异步处理

```python
class AsyncTenantProcessor:
    """异步租户处理器"""
    
    def __init__(self, task_queue):
        self.queue = task_queue
    
    async def process_batch(
        self,
        requests: List[dict]
    ) -> List[dict]:
        """批量处理请求（提高吞吐量）"""
        
        # 按租户分组
        tenant_groups = {}
        for req in requests:
            tenant_id = req["tenant_id"]
            if tenant_id not in tenant_groups:
                tenant_groups[tenant_id] = []
            tenant_groups[tenant_id].append(req)
        
        # 并行处理不同租户的请求
        tasks = []
        for tenant_id, tenant_requests in tenant_groups.items():
            task = self.process_tenant_batch(tenant_id, tenant_requests)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return [item for sublist in results for item in sublist]
    
    async def process_tenant_batch(
        self,
        tenant_id: str,
        requests: List[dict]
    ) -> List[dict]:
        """处理单个租户的批量请求"""
        
        results = []
        
        for req in requests:
            # 单个租户内的请求可以复用上下文
            result = await self.process_single(req)
            results.append(result)
        
        return results
```

### 8.3 扩展策略

```
┌─────────────────────────────────────────────────────────────┐
│                    扩展策略矩阵                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  水平扩展：                                                 │
│  ├── 无状态服务：增加实例数量                               │
│  ├── 有状态服务：分片（按租户ID哈希）                      │
│  └── 模型服务：多GPU并行推理                               │
│                                                             │
│  垂直扩展：                                                 │
│  ├── 升级GPU型号（A100 → H100）                           │
│  ├── 增加单机内存和CPU                                     │
│  └── 使用更快的存储（NVMe SSD）                            │
│                                                             │
│  按租户扩展：                                               │
│  ├── 小租户：共享资源池                                    │
│  ├── 中租户：资源预留                                      │
│  └── 大租户：专属资源                                      │
│                                                             │
│  弹性伸缩：                                                 │
│  ├── 基于请求量的自动扩缩容                                │
│  ├── 基于成本的预算控制                                    │
│  └── 基于时间的定时扩缩容（高峰期扩容）                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 结语

AI应用的多租户架构设计，本质上是在**资源共享**和**租户隔离**之间寻找平衡点。没有一种方案适合所有场景，需要根据业务规模、安全要求、成本预算和技术能力来选择合适的架构模式。

**核心原则**：

1. **分层隔离**：不同层级采用不同的隔离策略
2. **最小权限**：每个租户只能访问授权的资源
3. **精确计量**：所有资源使用都需要追踪和计费
4. **弹性扩展**：架构要能随业务增长平滑扩展
5. **安全合规**：满足行业和地区的合规要求

随着AI应用的普及，多租户架构将成为越来越多AI产品的基础设施。希望本文的实践经验能够帮助你在构建AI产品时，做出更好的架构决策。

---

*本文是AI系统架构系列的第二篇。下一篇将探讨AI应用的可观测性架构设计，敬请期待。*
