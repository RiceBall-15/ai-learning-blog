---
title: "LLM应用的Token预算控制与成本治理工程实践"
description: "系统解析LLM应用中Token预算控制的完整工程体系，涵盖预算分配策略、实时消耗监控、成本告警机制、多租户计费与ROI分析，附生产级代码实现"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: "infra"
tags: ["Token预算", "成本治理", "LLM应用", "AI工程化", "成本优化", "多租户计费"]
draft: false
---

# LLM应用的Token预算控制与成本治理工程实践

## 引言：Token就是AI时代的"算力货币"

在LLM应用中，每一个Token都有成本。一个看似简单的对话，背后可能是：

- GPT-4o: ~$0.005/1K input tokens, ~$0.015/1K output tokens
- Claude 3.5 Sonnet: ~$0.003/1K input, ~$0.015/1K output
- 自部署开源模型: GPU租赁 + 运维人力

当你的应用从100个用户增长到10万个用户时，**Token成本可能从每月$100暴涨到$100,000**。更糟糕的是，如果不加控制，成本增长往往是指数级的——因为用户会越来越依赖AI，每次交互的Token消耗会持续增加。

**Token预算控制不是一个可选的优化项，而是LLM应用生存的必要条件。**

本文将系统性地解析Token预算控制的工程实践，从预算分配到实时监控，从成本告警到多租户计费，构建一套完整的成本治理体系。

---

## 一、Token成本模型分析

### 1.1 成本构成

```
┌──────────────────────────────────────────────────────────┐
│                LLM应用Token成本构成                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  直接成本 (60-80%)                                       │
│  ├── API调用费用 (按Token计费)                            │
│  ├── 自部署GPU成本 (按时间计费)                           │
│  └── 向量数据库查询费用                                   │
│                                                          │
│  间接成本 (20-40%)                                       │
│  ├── Prompt存储与版本管理                                 │
│  ├── 上下文缓存维护                                      │
│  ├── 日志与审计存储                                      │
│  └── 监控与告警系统                                      │
│                                                          │
│  隐性成本 (难以量化)                                     │
│  ├── 过度Token消耗导致的响应延迟                          │
│  ├── 缓存失效导致的重复计算                               │
│  └── 低质量Prompt导致的Token浪费                          │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Token消耗分析

```python
@dataclass
class TokenConsumption:
    """Token消耗分析"""
    input_tokens: int
    output_tokens: int
    cached_tokens: int  # 命中缓存的Token数
    model: str
    timestamp: datetime
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def cache_hit_rate(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.input_tokens
    
    @property
    def effective_cost(self) -> float:
        """实际成本（考虑缓存折扣）"""
        pricing = MODEL_PRICING[self.model]
        # 缓存Token通常有折扣
        effective_input = self.input_tokens - self.cached_tokens
        cached_cost = self.cached_tokens * pricing["input_cached"]
        regular_cost = effective_input * pricing["input"]
        output_cost = self.output_tokens * pricing["output"]
        return (cached_cost + regular_cost + output_cost) / 1000

# 模型定价表
MODEL_PRICING = {
    "gpt-4o": {
        "input": 0.0025,
        "input_cached": 0.00125,  # 缓存折扣50%
        "output": 0.01,
    },
    "claude-3.5-sonnet": {
        "input": 0.003,
        "input_cached": 0.0015,
        "output": 0.015,
    },
    "deepseek-v3": {
        "input": 0.00014,
        "input_cached": 0.000014,
        "output": 0.00028,
    },
}
```

---

## 二、预算分配架构

### 2.1 多层级预算体系

```
┌──────────────────────────────────────────────────────────┐
│                  Token预算层级结构                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  组织级预算 (Organization)                                │
│  ├── 月度总预算: $10,000                                 │
│  ├── 告警阈值: 80% ($8,000)                             │
│  └── 硬限制: 100% ($10,000)                             │
│       │                                                  │
│       ├── 项目级预算 (Project)                            │
│       │   ├── 月度预算: $3,000                           │
│       │   ├── 告警阈值: 80%                              │
│       │   └── 硬限制: 100%                               │
│       │       │                                          │
│       │       ├── 用户级预算 (User)                       │
│       │       │   ├── 月度预算: $50                      │
│       │       │   ├── 告警阈值: 80%                      │
│       │       │   └── 硬限制: 100%                       │
│       │       │                                          │
│       │       └── 会话级预算 (Session)                    │
│       │           ├── 单次预算: $0.50                    │
│       │           └── 硬限制: $1.00                      │
│       │                                                  │
│       └── 项目级预算 (Project)                            │
│           └── ...                                        │
└──────────────────────────────────────────────────────────┘
```

### 2.2 预算配置系统

```python
class TokenBudgetConfig:
    """Token预算配置管理"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.budgets: Dict[str, Budget] = {}
        self._initialize_budgets()
    
    def _initialize_budgets(self):
        """初始化各层级预算"""
        # 组织级预算
        org_config = self.config["organization"]
        self.budgets["org"] = Budget(
            level="organization",
            monthly_limit=org_config["monthly_limit_usd"],
            alert_threshold=org_config.get("alert_threshold", 0.8),
            hard_limit=org_config.get("hard_limit", 1.0),
        )
        
        # 项目级预算
        for project in self.config.get("projects", []):
            self.budgets[f"project:{project['name']}"] = Budget(
                level="project",
                monthly_limit=project["monthly_limit_usd"],
                alert_threshold=project.get("alert_threshold", 0.8),
                hard_limit=project.get("hard_limit", 1.0),
                parent_budget="org",
            )
        
        # 用户级预算
        for user in self.config.get("users", []):
            self.budgets[f"user:{user['id']}"] = Budget(
                level="user",
                monthly_limit=user["monthly_limit_usd"],
                alert_threshold=user.get("alert_threshold", 0.8),
                hard_limit=user.get("hard_limit", 1.0),
                parent_budget=f"project:{user['project']}",
            )

@dataclass
class Budget:
    """预算定义"""
    level: str  # organization, project, user, session
    monthly_limit: float
    alert_threshold: float  # 0.0 - 1.0
    hard_limit: float  # 0.0 - 1.0
    parent_budget: Optional[str] = None
    
    def check(self, current_usage: float) -> BudgetStatus:
        """检查预算状态"""
        usage_ratio = current_usage / self.monthly_limit if self.monthly_limit > 0 else 0
        
        if usage_ratio >= self.hard_limit:
            return BudgetStatus.EXCEEDED
        elif usage_ratio >= self.alert_threshold:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.OK
```

### 2.3 预算配置YAML示例

```yaml
# budget-config.yaml
organization:
  monthly_limit_usd: 10000
  alert_threshold: 0.8
  hard_limit: 1.0
  alert_channels:
    - type: slack
      webhook: "https://hooks.slack.com/services/xxx"
    - type: email
      recipients: ["admin@company.com"]

projects:
  - name: "customer-support-bot"
    monthly_limit_usd: 3000
    alert_threshold: 0.75
    hard_limit: 0.95
    model_preference: ["deepseek-v3", "gpt-4o-mini"]
    
  - name: "code-review-agent"
    monthly_limit_usd: 2000
    alert_threshold: 0.8
    hard_limit: 1.0
    model_preference: ["gpt-4o", "claude-3.5-sonnet"]

users:
  - id: "user-001"
    project: "customer-support-bot"
    monthly_limit_usd: 50
    tier: "premium"  # premium用户有更高预算
    
  - id: "user-002"
    project: "customer-support-bot"
    monthly_limit_usd: 10
    tier: "free"

sessions:
  default:
    per_session_limit_usd: 0.50
    per_request_limit_usd: 0.10
```

---

## 三、实时消耗监控

### 3.1 Token消耗追踪器

```python
class TokenTracker:
    """实时Token消耗追踪"""
    
    def __init__(self, redis_client, db_client):
        self.redis = redis_client
        self.db = db_client
        self.metrics_collector = MetricsCollector()
    
    async def track_consumption(
        self,
        user_id: str,
        project: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        metadata: Optional[Dict] = None
    ) -> ConsumptionRecord:
        """记录一次Token消耗"""
        
        # 1. 计算成本
        cost = self._calculate_cost(model, input_tokens, output_tokens, cached_tokens)
        
        # 2. 生成消耗记录
        record = ConsumptionRecord(
            id=generate_uuid(),
            user_id=user_id,
            project=project,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # 3. 更新Redis实时计数器
        await self._update_redis_counters(record)
        
        # 4. 持久化到数据库
        await self.db.insert_consumption(record)
        
        # 5. 检查预算
        await self._check_budgets(record)
        
        # 6. 发送指标
        self.metrics_collector.record(record)
        
        return record
    
    async def _update_redis_counters(self, record: ConsumptionRecord):
        """更新Redis实时计数器"""
        pipe = self.redis.pipeline()
        
        # 用户级计数器
        user_key = f"token:user:{record.user_id}:monthly:{record.timestamp.strftime('%Y-%m')}"
        pipe.incrbyfloat(f"{user_key}:cost", record.cost_usd)
        pipe.incrby(f"{user_key}:input_tokens", record.input_tokens)
        pipe.incrby(f"{user_key}:output_tokens", record.output_tokens)
        pipe.expire(user_key, 60 * 60 * 24 * 35)  # 35天过期
        
        # 项目级计数器
        project_key = f"token:project:{record.project}:monthly:{record.timestamp.strftime('%Y-%m')}"
        pipe.incrbyfloat(f"{project_key}:cost", record.cost_usd)
        pipe.incrby(f"{project_key}:input_tokens", record.input_tokens)
        pipe.incrby(f"{project_key}:output_tokens", record.output_tokens)
        pipe.expire(project_key, 60 * 60 * 24 * 35)
        
        # 组织级计数器
        org_key = f"token:org:monthly:{record.timestamp.strftime('%Y-%m')}"
        pipe.incrbyfloat(f"{org_key}:cost", record.cost_usd)
        pipe.incrby(f"{org_key}:input_tokens", record.input_tokens)
        pipe.incrby(f"{org_key}:output_tokens", record.output_tokens)
        pipe.expire(org_key, 60 * 60 * 24 * 35)
        
        # 会话级计数器
        session_key = f"token:session:{record.metadata.get('session_id', 'unknown')}"
        pipe.incrbyfloat(f"{session_key}:cost", record.cost_usd)
        pipe.incrby(f"{session_key}:tokens", record.input_tokens + record.output_tokens)
        pipe.expire(session_key, 60 * 60 * 24)  # 24小时过期
        
        await pipe.execute()
    
    def _calculate_cost(
        self, 
        model: str, 
        input_tokens: int, 
        output_tokens: int,
        cached_tokens: int
    ) -> float:
        """计算Token成本"""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
        
        # 缓存Token有折扣
        regular_input = input_tokens - cached_tokens
        cached_cost = cached_tokens * pricing["input_cached"] / 1000
        regular_cost = regular_input * pricing["input"] / 1000
        output_cost = output_tokens * pricing["output"] / 1000
        
        return cached_cost + regular_cost + output_cost
```

### 3.2 预算检查与拦截

```python
class BudgetEnforcer:
    """预算执行器：实时检查并拦截超预算请求"""
    
    def __init__(self, redis_client, config: TokenBudgetConfig):
        self.redis = redis_client
        self.config = config
        self.alert_manager = AlertManager()
    
    async def check_budget(
        self, 
        user_id: str, 
        project: str,
        estimated_tokens: int,
        model: str
    ) -> BudgetCheckResult:
        """检查请求是否在预算范围内"""
        
        # 1. 计算预估成本
        estimated_cost = self._estimate_cost(model, estimated_tokens)
        
        # 2. 检查用户级预算
        user_budget = self.config.budgets.get(f"user:{user_id}")
        if user_budget:
            user_usage = await self._get_current_usage("user", user_id)
            if user_usage + estimated_cost > user_budget.monthly_limit * user_budget.hard_limit:
                return BudgetCheckResult(
                    allowed=False,
                    reason="user_budget_exceeded",
                    current_usage=user_usage,
                    limit=user_budget.monthly_limit,
                    remaining=user_budget.monthly_limit - user_usage
                )
        
        # 3. 检查项目级预算
        project_budget = self.config.budgets.get(f"project:{project}")
        if project_budget:
            project_usage = await self._get_current_usage("project", project)
            if project_usage + estimated_cost > project_budget.monthly_limit * project_budget.hard_limit:
                return BudgetCheckResult(
                    allowed=False,
                    reason="project_budget_exceeded",
                    current_usage=project_usage,
                    limit=project_budget.monthly_limit,
                    remaining=project_budget.monthly_limit - project_usage
                )
        
        # 4. 检查组织级预算
        org_budget = self.config.budgets.get("org")
        if org_budget:
            org_usage = await self._get_current_usage("organization", "default")
            if org_usage + estimated_cost > org_budget.monthly_limit * org_budget.hard_limit:
                return BudgetCheckResult(
                    allowed=False,
                    reason="organization_budget_exceeded",
                    current_usage=org_usage,
                    limit=org_budget.monthly_limit,
                    remaining=org_budget.monthly_limit - org_usage
                )
        
        # 5. 检查是否触发告警
        await self._check_alert_thresholds(user_id, project)
        
        return BudgetCheckResult(
            allowed=True,
            estimated_cost=estimated_cost,
            remaining_budget=org_budget.monthly_limit - org_usage if org_budget else None
        )
    
    async def _check_alert_thresholds(self, user_id: str, project: str):
        """检查是否触发告警阈值"""
        # 用户级告警
        user_budget = self.config.budgets.get(f"user:{user_id}")
        if user_budget:
            user_usage = await self._get_current_usage("user", user_id)
            usage_ratio = user_usage / user_budget.monthly_limit
            if usage_ratio >= user_budget.alert_threshold:
                await self.alert_manager.send_alert(
                    level="warning",
                    target=f"user:{user_id}",
                    message=f"用户 {user_id} Token使用量已达 {usage_ratio:.1%}",
                    current_usage=user_usage,
                    limit=user_budget.monthly_limit
                )
        
        # 项目级告警
        project_budget = self.config.budgets.get(f"project:{project}")
        if project_budget:
            project_usage = await self._get_current_usage("project", project)
            usage_ratio = project_usage / project_budget.monthly_limit
            if usage_ratio >= project_budget.alert_threshold:
                await self.alert_manager.send_alert(
                    level="warning",
                    target=f"project:{project}",
                    message=f"项目 {project} Token使用量已达 {usage_ratio:.1%}",
                    current_usage=project_usage,
                    limit=project_budget.monthly_limit
                )
    
    async def _get_current_usage(self, level: str, identifier: str) -> float:
        """获取当前使用量"""
        month = datetime.now().strftime("%Y-%m")
        
        if level == "user":
            key = f"token:user:{identifier}:monthly:{month}:cost"
        elif level == "project":
            key = f"token:project:{identifier}:monthly:{month}:cost"
        elif level == "organization":
            key = f"token:org:monthly:{month}:cost"
        else:
            return 0.0
        
        usage = await self.redis.get(key)
        return float(usage) if usage else 0.0
    
    def _estimate_cost(self, model: str, estimated_tokens: int) -> float:
        """预估成本"""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
        # 假设输入输出比为3:1
        input_tokens = estimated_tokens * 0.75
        output_tokens = estimated_tokens * 0.25
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
```

---

## 四、智能成本优化策略

### 4.1 模型路由降级

当预算紧张时，自动降级到更便宜的模型：

```python
class ModelRouterWithBudget:
    """基于预算的智能模型路由"""
    
    async def select_model(
        self,
        user_id: str,
        project: str,
        task_type: str,
        estimated_tokens: int
    ) -> ModelSelection:
        """根据预算和任务类型选择模型"""
        
        # 1. 获取预算状态
        budget_status = await self.get_budget_status(user_id, project)
        
        # 2. 根据预算状态和任务类型选择模型
        if budget_status.usage_ratio >= 0.95:
            # 预算即将耗尽：强制使用最便宜的模型
            return ModelSelection(
                model="deepseek-v3",
                reason="budget_critical",
                quality_tradeoff="high"
            )
        
        elif budget_status.usage_ratio >= 0.80:
            # 预算告警：使用性价比最高的模型
            if task_type in ["simple_qa", "summarization", "translation"]:
                return ModelSelection(
                    model="gpt-4o-mini",
                    reason="budget_warning",
                    quality_tradeoff="low"
                )
            else:
                return ModelSelection(
                    model="deepseek-v3",
                    reason="budget_warning",
                    quality_tradeoff="medium"
                )
        
        elif budget_status.usage_ratio >= 0.50:
            # 预算充足但需注意：根据任务类型选择
            model_map = {
                "simple_qa": "gpt-4o-mini",
                "summarization": "gpt-4o-mini",
                "translation": "deepseek-v3",
                "code_review": "gpt-4o",
                "complex_reasoning": "gpt-4o",
                "creative_writing": "claude-3.5-sonnet",
            }
            return ModelSelection(
                model=model_map.get(task_type, "gpt-4o-mini"),
                reason="budget_optimized",
                quality_tradeoff="minimal"
            )
        
        else:
            # 预算充裕：使用最佳模型
            model_map = {
                "simple_qa": "gpt-4o-mini",
                "summarization": "gpt-4o",
                "translation": "deepseek-v3",
                "code_review": "gpt-4o",
                "complex_reasoning": "gpt-4o",
                "creative_writing": "claude-3.5-sonnet",
            }
            return ModelSelection(
                model=model_map.get(task_type, "gpt-4o"),
                reason="budget_plentiful",
                quality_tradeoff="none"
            )
```

### 4.2 Prompt优化与Token压缩

```python
class TokenOptimizer:
    """Token使用优化器"""
    
    async def optimize_prompt(
        self,
        prompt: str,
        budget_remaining: float,
        max_output_tokens: int
    ) -> OptimizedPrompt:
        """优化Prompt以减少Token消耗"""
        
        # 1. 分析当前Prompt的Token消耗
        current_tokens = self.count_tokens(prompt)
        
        # 2. 根据预算决定优化策略
        if budget_remaining < 0.1:  # 预算非常紧张
            optimized = await self.aggressive_compress(prompt)
            strategy = "aggressive"
        elif budget_remaining < 0.3:  # 预算较紧张
            optimized = await self.moderate_compress(prompt)
            strategy = "moderate"
        else:  # 预算充足
            optimized = await self.light_compress(prompt)
            strategy = "light"
        
        return OptimizedPrompt(
            original=prompt,
            optimized=optimized,
            original_tokens=current_tokens,
            optimized_tokens=self.count_tokens(optimized),
            strategy=strategy,
            savings_ratio=1 - (self.count_tokens(optimized) / current_tokens)
        )
    
    async def aggressive_compress(self, prompt: str) -> str:
        """激进压缩：大幅减少Token消耗"""
        # 1. 移除所有非必要的格式和说明
        compressed = self.remove_filler_words(prompt)
        compressed = self.simplify_formatting(compressed)
        
        # 2. 使用更简洁的指令
        compressed = self.simplify_instructions(compressed)
        
        # 3. 截断过长的上下文
        compressed = self.truncate_context(compressed, max_tokens=2000)
        
        # 4. 使用缩写和简写
        compressed = self.apply_abbreviations(compressed)
        
        return compressed
    
    async def moderate_compress(self, prompt: str) -> str:
        """适度压缩：保持质量的同时减少Token"""
        compressed = self.remove_filler_words(prompt)
        compressed = self.simplify_formatting(compressed)
        compressed = self.truncate_context(compressed, max_tokens=4000)
        return compressed
    
    async def light_compress(self, prompt: str) -> str:
        """轻度压缩：仅移除明显的冗余"""
        compressed = self.remove_filler_words(prompt)
        compressed = self.simplify_formatting(compressed)
        return compressed
    
    def count_tokens(self, text: str) -> int:
        """计算Token数量"""
        # 使用tiktoken或模型特定的tokenizer
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(text))
```

### 4.3 缓存策略

```python
class SemanticCache:
    """语义缓存：减少重复的Token消耗"""
    
    def __init__(self, vector_db, ttl_seconds: int = 3600):
        self.vector_db = vector_db
        self.ttl = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
    
    async def get_or_compute(
        self,
        query: str,
        compute_fn: Callable,
        similarity_threshold: float = 0.95
    ) -> CacheResult:
        """获取缓存或计算新结果"""
        
        # 1. 查找语义相似的缓存
        similar = await self.vector_db.search(
            query_embedding=await self.embed(query),
            top_k=3,
            threshold=similarity_threshold
        )
        
        if similar:
            # 2. 缓存命中
            self.hit_count += 1
            cached = similar[0]
            
            # 检查是否过期
            if datetime.now() - cached.timestamp < timedelta(seconds=self.ttl):
                return CacheResult(
                    result=cached.response,
                    from_cache=True,
                    tokens_saved=cached.tokens_used,
                    cost_saved=cached.cost
                )
        
        # 3. 缓存未命中，执行计算
        self.miss_count += 1
        result = await compute_fn(query)
        
        # 4. 存入缓存
        await self.vector_db.store(
            query=query,
            query_embedding=await self.embed(query),
            response=result.response,
            tokens_used=result.tokens_used,
            cost=result.cost,
            timestamp=datetime.now()
        )
        
        return CacheResult(
            result=result.response,
            from_cache=False,
            tokens_saved=0,
            cost_saved=0
        )
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
```

---

## 五、多租户计费系统

### 5.1 计费数据模型

```python
class BillingRecord:
    """计费记录"""
    
    def __init__(self):
        self.id = generate_uuid()
        self.tenant_id: str = ""  # 租户ID
        self.user_id: str = ""  # 用户ID
        self.project: str = ""  # 项目名
        self.service: str = ""  # 服务名
        self.model: str = ""  # 使用的模型
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_tokens: int = 0
        self.cost_usd: float = 0.0
        self.timestamp: datetime = datetime.now()
        self.metadata: Dict = {}

class TenantBilling:
    """租户计费管理"""
    
    async def generate_invoice(self, tenant_id: str, month: str) -> Invoice:
        """生成月度账单"""
        
        # 1. 查询所有消费记录
        records = await self.db.query_consumptions(
            tenant_id=tenant_id,
            month=month
        )
        
        # 2. 按项目分组统计
        project_stats = defaultdict(lambda: {
            "total_cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "requests": 0,
            "models": defaultdict(lambda: {"cost": 0.0, "tokens": 0})
        })
        
        for record in records:
            stats = project_stats[record.project]
            stats["total_cost"] += record.cost_usd
            stats["input_tokens"] += record.input_tokens
            stats["output_tokens"] += record.output_tokens
            stats["cached_tokens"] += record.cached_tokens
            stats["requests"] += 1
            stats["models"][record.model]["cost"] += record.cost_usd
            stats["models"][record.model]["tokens"] += (
                record.input_tokens + record.output_tokens
            )
        
        # 3. 生成账单
        invoice = Invoice(
            id=generate_uuid(),
            tenant_id=tenant_id,
            month=month,
            total_cost=sum(s["total_cost"] for s in project_stats.values()),
            projects=[
                ProjectBilling(
                    name=project,
                    cost=stats["total_cost"],
                    tokens=stats["input_tokens"] + stats["output_tokens"],
                    requests=stats["requests"],
                    models=dict(stats["models"])
                )
                for project, stats in project_stats.items()
            ],
            generated_at=datetime.now()
        )
        
        return invoice
```

### 5.2 计费仪表盘

```python
class BillingDashboard:
    """计费仪表盘数据"""
    
    async def get_dashboard_data(
        self, 
        tenant_id: str, 
        time_range: str = "30d"
    ) -> DashboardData:
        """获取仪表盘数据"""
        
        # 1. 总览数据
        overview = await self.get_overview(tenant_id, time_range)
        
        # 2. 趋势数据
        trend = await self.get_cost_trend(tenant_id, time_range)
        
        # 3. 项目分布
        project_distribution = await self.get_project_distribution(tenant_id, time_range)
        
        # 4. 模型使用分布
        model_distribution = await self.get_model_distribution(tenant_id, time_range)
        
        # 5. 用户使用排行
        user_ranking = await self.get_user_ranking(tenant_id, time_range)
        
        # 6. 预算使用情况
        budget_status = await self.get_budget_status(tenant_id)
        
        return DashboardData(
            overview=overview,
            trend=trend,
            project_distribution=project_distribution,
            model_distribution=model_distribution,
            user_ranking=user_ranking,
            budget_status=budget_status
        )
    
    async def get_overview(self, tenant_id: str, time_range: str) -> Overview:
        """获取总览数据"""
        stats = await self.db.aggregate_consumptions(
            tenant_id=tenant_id,
            time_range=time_range,
            group_by=None
        )
        
        return Overview(
            total_cost=stats["total_cost"],
            total_tokens=stats["total_tokens"],
            total_requests=stats["total_requests"],
            avg_cost_per_request=stats["total_cost"] / max(stats["total_requests"], 1),
            avg_tokens_per_request=stats["total_tokens"] / max(stats["total_requests"], 1),
            cache_hit_rate=stats["cached_tokens"] / max(stats["total_input_tokens"], 1),
            cost_change_pct=stats.get("cost_change_pct", 0),
        )
```

---

## 六、告警与自动化响应

### 6.1 多级告警策略

```python
class CostAlertManager:
    """成本告警管理器"""
    
    ALERT_LEVELS = {
        "info": {"threshold": 0.5, "channels": ["slack"]},
        "warning": {"threshold": 0.7, "channels": ["slack", "email"]},
        "critical": {"threshold": 0.9, "channels": ["slack", "email", "sms"]},
        "emergency": {"threshold": 1.0, "channels": ["slack", "email", "sms", "phone"]},
    }
    
    async def check_and_alert(self, tenant_id: str, project: str):
        """检查并发送告警"""
        
        # 1. 获取预算使用情况
        budget_status = await self.get_budget_status(tenant_id, project)
        
        # 2. 确定告警级别
        alert_level = self.determine_alert_level(budget_status.usage_ratio)
        
        if alert_level:
            # 3. 构建告警消息
            alert = Alert(
                level=alert_level,
                tenant_id=tenant_id,
                project=project,
                usage_ratio=budget_status.usage_ratio,
                current_cost=budget_status.current_cost,
                budget_limit=budget_status.budget_limit,
                message=self.format_alert_message(budget_status),
                timestamp=datetime.now()
            )
            
            # 4. 发送到指定渠道
            channels = self.ALERT_LEVELS[alert_level]["channels"]
            await self.send_to_channels(alert, channels)
            
            # 5. 触发自动化响应
            await self.trigger_auto_response(alert)
    
    async def trigger_auto_response(self, alert: Alert):
        """触发自动化响应"""
        
        if alert.level == "critical":
            # 1. 自动降级模型
            await self.auto_downgrade_models(alert.project)
            
            # 2. 限制高消耗用户
            await self.limit_high_consumers(alert.project)
            
            # 3. 增加缓存TTL
            await self.increase_cache_ttl(alert.project)
        
        elif alert.level == "emergency":
            # 1. 暂停非关键服务
            await self.pause_non_critical_services(alert.project)
            
            # 2. 通知管理层
            await self.notify_management(alert)
            
            # 3. 启动成本优化脚本
            await self.run_cost_optimization(alert.project)
```

### 6.2 自动化成本优化

```python
class AutoCostOptimizer:
    """自动化成本优化"""
    
    async def optimize(self, project: str, budget_status: BudgetStatus):
        """根据预算状态自动优化"""
        
        optimizations = []
        
        # 1. 模型降级
        if budget_status.usage_ratio >= 0.8:
            await self.downgrade_models(project)
            optimizations.append("model_downgrade")
        
        # 2. Prompt压缩
        if budget_status.usage_ratio >= 0.7:
            await self.enable_prompt_compression(project)
            optimizations.append("prompt_compression")
        
        # 3. 增强缓存
        if budget_status.usage_ratio >= 0.6:
            await self.enhance_caching(project)
            optimizations.append("enhanced_caching")
        
        # 4. 限制并发
        if budget_status.usage_ratio >= 0.9:
            await self.limit_concurrency(project)
            optimizations.append("concurrency_limit")
        
        # 5. 记录优化操作
        await self.log_optimizations(project, optimizations)
        
        return optimizations
    
    async def downgrade_models(self, project: str):
        """自动降级模型配置"""
        # 更新项目配置，使用更便宜的模型
        config = await self.get_project_config(project)
        
        config.model_preferences = {
            "simple_qa": "deepseek-v3",
            "summarization": "deepseek-v3",
            "translation": "deepseek-v3",
            "code_review": "gpt-4o-mini",
            "complex_reasoning": "gpt-4o-mini",
        }
        
        await self.update_project_config(project, config)
    
    async def enhance_caching(self, project: str):
        """增强缓存策略"""
        config = await self.get_project_config(project)
        
        # 增加缓存TTL
        config.cache_ttl = 7200  # 2小时
        
        # 降低缓存相似度阈值（更多命中）
        config.cache_similarity_threshold = 0.90
        
        # 启用语义缓存
        config.semantic_cache_enabled = True
        
        await self.update_project_config(project, config)
```

---

## 七、ROI分析与报告

### 7.1 ROI计算模型

```python
class ROIAnalyzer:
    """AI应用ROI分析"""
    
    async def calculate_roi(
        self,
        project: str,
        time_range: str = "30d"
    ) -> ROIReport:
        """计算项目ROI"""
        
        # 1. 获取成本数据
        costs = await self.get_costs(project, time_range)
        
        # 2. 获取价值数据（需要业务系统提供）
        value = await self.get_business_value(project, time_range)
        
        # 3. 计算ROI
        total_cost = costs.total_api_cost + costs.infrastructure_cost + costs.engineering_cost
        total_value = value.revenue_generated + value.cost_saved + value.productivity_gain
        
        roi = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
        
        return ROIReport(
            project=project,
            time_range=time_range,
            total_cost=total_cost,
            total_value=total_value,
            roi_percentage=roi,
            cost_breakdown=CostBreakdown(
                api_costs=costs.api_costs_by_model,
                infrastructure=costs.infrastructure_cost,
                engineering=costs.engineering_cost,
            ),
            value_breakdown=ValueBreakdown(
                revenue=value.revenue_generated,
                cost_savings=value.cost_saved,
                productivity=value.productivity_gain,
            ),
            recommendations=self.generate_recommendations(costs, value)
        )
    
    def generate_recommendations(self, costs, value) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 1. 模型优化建议
        if costs.api_costs_by_model.get("gpt-4o", 0) > costs.total_api_cost * 0.5:
            recommendations.append(
                "GPT-4o使用占比超过50%，建议评估是否可以将部分任务迁移到更便宜的模型"
            )
        
        # 2. 缓存优化建议
        if costs.cache_hit_rate < 0.3:
            recommendations.append(
                f"缓存命中率仅为{costs.cache_hit_rate:.1%}，建议优化缓存策略"
            )
        
        # 3. Prompt优化建议
        if costs.avg_tokens_per_request > 2000:
            recommendations.append(
                f"平均每次请求消耗{costs.avg_tokens_per_request} tokens，建议优化Prompt"
            )
        
        return recommendations
```

### 7.2 成本报告模板

```
┌──────────────────────────────────────────────────────────┐
│           LLM应用月度成本报告                              │
│           项目: customer-support-bot                      │
│           期间: 2026-05-01 ~ 2026-05-31                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  📊 总览                                                 │
│  ├── 总成本: $2,847.32                                  │
│  ├── 总Token: 12,450,000                                │
│  ├── 总请求: 45,230                                     │
│  ├── 平均成本/请求: $0.063                               │
│  └── 环比变化: +12.3%                                   │
│                                                          │
│  💰 成本分布                                             │
│  ├── GPT-4o: $1,234.56 (43.4%)                         │
│  ├── GPT-4o-mini: $892.11 (31.3%)                      │
│  ├── DeepSeek-V3: $456.78 (16.0%)                      │
│  └── 其他: $263.87 (9.3%)                               │
│                                                          │
│  📈 趋势                                                 │
│  ├── 第1周: $589.22                                     │
│  ├── 第2周: $678.45                                     │
│  ├── 第3周: $756.12                                     │
│  └── 第4周: $823.53                                     │
│                                                          │
│  🎯 预算使用                                             │
│  ├── 预算: $3,000.00                                    │
│  ├── 已用: $2,847.32 (94.9%)                           │
│  └── 剩余: $152.68                                      │
│                                                          │
│  ⚡ 优化建议                                             │
│  ├── 1. 启用语义缓存，预计节省15%                        │
│  ├── 2. 将简单QA迁移到DeepSeek-V3，预计节省20%          │
│  └── 3. 优化Prompt模板，预计节省10%                      │
│                                                          │
│  📊 ROI分析                                              │
│  ├── 成本: $2,847.32                                    │
│  ├── 价值: $15,230.00                                   │
│  └── ROI: 435%                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 八、生产部署清单

### 8.1 部署检查表

```
□ 基础设施
  ├── □ Redis集群已部署（用于实时计数器）
  ├── □ 时序数据库已部署（用于历史分析）
  ├── □ 告警系统已配置（Slack/Email/SMS）
  └── □ 监控仪表盘已搭建

□ 预算配置
  ├── □ 组织级预算已设置
  ├── □ 项目级预算已设置
  ├── □ 用户级预算已设置
  ├── □ 会话级预算已设置
  └── □ 告警阈值已配置

□ 模型路由
  ├── □ 模型定价表已更新
  ├── □ 降级策略已配置
  ├── □ 缓存策略已优化
  └── □ Prompt压缩已启用

□ 计费系统
  ├── □ 计费数据模型已设计
  ├── □ 账单生成逻辑已实现
  ├── □ 支付集成已完成
  └── □ 发票系统已配置

□ 自动化
  ├── □ 自动降级逻辑已测试
  ├── □ 自动告警已验证
  ├── □ 自动优化已部署
  └── □ 回滚机制已准备
```

### 8.2 监控指标

```python
# 关键监控指标
METRICS = {
    # 成本指标
    "cost_total_daily": "每日总成本",
    "cost_total_monthly": "每月总成本",
    "cost_per_request": "每次请求平均成本",
    "cost_per_token": "每个Token平均成本",
    
    # 使用量指标
    "tokens_input_daily": "每日输入Token数",
    "tokens_output_daily": "每日输出Token数",
    "tokens_cached_daily": "每日缓存Token数",
    "requests_daily": "每日请求数",
    
    # 效率指标
    "cache_hit_rate": "缓存命中率",
    "avg_tokens_per_request": "每次请求平均Token数",
    "model_distribution": "模型使用分布",
    
    # 预算指标
    "budget_usage_org": "组织预算使用率",
    "budget_usage_project": "项目预算使用率",
    "budget_usage_user": "用户预算使用率",
}
```

---

## 九、总结

Token预算控制是LLM应用可持续发展的基石。核心设计原则：

1. **分层预算**：组织→项目→用户→会话，逐级细化控制
2. **实时监控**：基于Redis的实时计数器，毫秒级响应
3. **智能路由**：根据预算状态自动选择最优模型
4. **多级告警**：从信息通知到紧急响应，覆盖全场景
5. **自动优化**：模型降级、缓存增强、Prompt压缩组合拳
6. **透明计费**：多租户计费，让用户清楚自己的消耗

最终目标是构建一个**成本可控、价值可见、优化自动**的Token治理体系，让AI应用在为用户创造价值的同时，也能为组织带来合理的ROI。

---

> **关键数字**：通过实施本文的Token预算控制体系，典型LLM应用可以实现：
> - Token成本降低30-50%
> - 预算超支风险降低90%
> - 成本可视化程度提升100%
> - ROI分析准确度提升80%
