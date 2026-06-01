---
title: "PydanticAI实战：类型安全的Agent开发新范式"
description: "深入解析PydanticAI框架的设计哲学、核心机制与实战应用，探索如何用类型系统构建可靠、可维护的AI Agent系统"
date: 2026-06-01
author: "RiceBall"
category: "framework"
tags: ["PydanticAI", "Agent框架", "类型安全", "Python", "AI工程"]
draft: false
---

## 为什么需要一个新的Agent框架？

2025年以来，AI Agent框架领域涌现了大量选择：LangGraph、CrewAI、AutoGen、Dify……每个框架都有自己的设计哲学和适用场景。但在实际生产中，我们经常遇到一个核心痛点：**Agent系统的类型安全和可维护性**。

大多数Agent框架在处理LLM交互时，倾向于使用动态类型和字典传递数据。这在原型阶段没有问题，但当系统复杂度上升到生产级别时，类型缺失带来的问题会急剧放大：

- 输入输出结构难以验证，运行时错误频繁
- Agent之间的数据传递缺乏约束，难以追踪数据流
- Prompt模板与数据结构脱节，修改一处容易引发连锁问题
- 测试和调试困难，缺少编译期的类型检查

想象一个典型的场景：你的Agent系统有5个Agent互相协作，每个Agent的输入输出都是字典。当某个Agent的输出格式悄悄变了（比如字段名从`user_name`改成了`username`），整个系统可能在运行了几小时后才在某个边缘情况中崩溃。这类问题在静态类型系统中根本不会发生。

PydanticAI正是为了解决这些问题而生。它由Pydantic团队开发，将Pydantic的类型系统与LLM交互深度融合，提供了一种**以类型为驱动的Agent开发范式**。

## PydanticAI的核心设计哲学

### 类型即接口

PydanticAI的核心思想是：**用Python类型定义来描述Agent的输入输出契约**。这不仅仅是类型标注，而是贯穿整个Agent生命周期的结构化约束。

```python
from pydantic_ai import Agent
from pydantic import BaseModel

# 定义输出结构
class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    budget_level: str  # budget, moderate, luxury
    highlights: list[str]
    warnings: list[str]

# 创建Agent，输出结构在类型层面被约束
travel_agent = Agent(
    'openai:gpt-4o',
    system_prompt='你是一个专业的旅行规划师。',
    result_type=TravelPlan,
)

# result一定是TravelPlan类型，不是字典
result = travel_agent.run_sync('帮我规划一个东京5日游，预算适中')
print(result.output.destination)  # 类型安全的属性访问
print(result.output.budget_level)  # IDE自动补全，不会拼错字段名
```

当LLM的输出不符合`TravelPlan`的结构时，PydanticAI会自动重试，直到输出通过类型验证。这意味着你在代码中使用的每一个字段都保证存在且类型正确——这在传统字典方案中是不可能的。

### 依赖注入机制

这是PydanticAI区别于其他框架的最重要特性。它借鉴了FastAPI的依赖注入模式，让你可以声明Agent运行时需要的外部依赖：

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class TravelDeps:
    weather_api: WeatherAPI
    flight_api: FlightAPI
    hotel_api: HotelAPI

travel_agent = Agent(
    'openai:gpt-4o',
    system_prompt='根据实时天气和航班信息为用户规划旅行。',
    deps_type=TravelDeps,
    result_type=TravelPlan,
)

@travel_agent.tool
async def get_weather(ctx: RunContext[TravelDeps], city: str) -> str:
    """获取指定城市的天气预报。"""
    weather = await ctx.deps.weather_api.get_forecast(city)
    return f"{city}未来3天天气: {weather.forecast}"

@travel_agent.tool
async def search_flights(ctx: RunContext[TravelDeps], from_city: str, to_city: str) -> str:
    """搜索航班信息。"""
    flights = await ctx.deps.flight_api.search(from_city, to_city)
    return "\n".join(f"{f.airline} {f.departure} - {f.price}" for f in flights[:5])
```

这种设计的好处是显而易见的：

1. **依赖可测试**：测试时注入mock依赖，无需真正调用外部API
2. **依赖可追踪**：每个工具声明了它需要什么依赖，数据流清晰
3. **依赖可替换**：切换Weather API提供商时，只改依赖注入配置，不需要修改Agent逻辑
4. **依赖可组合**：不同Agent可以共享同一套依赖，也可以使用不同的依赖配置

### 工具系统的深度设计

PydanticAI的工具系统不仅仅是函数调用那么简单。它支持多种工具类型，每种都有明确的类型约束：

```python
from pydantic import BaseModel, Field
from typing import Annotated

# 简单工具：LLM自动决定参数
@agent.tool
async def get_weather(ctx: RunContext[deps], city: str) -> str:
    """获取城市天气信息。"""
    return await ctx.deps.weather.get(city)

# 带复杂参数的工具
class SearchParams(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, ge=1, le=20, description="最大结果数")
    filters: dict[str, str] = Field(default_factory=dict, description="过滤条件")

@agent.tool
async def search_products(ctx: RunContext[deps], params: SearchParams) -> str:
    """搜索商品，支持关键词和过滤条件。"""
    results = await ctx.deps.product_db.search(
        query=params.query,
        limit=params.max_results,
        filters=params.filters,
    )
    return format_results(results)

# 静态工具：不需要LLM参数，直接执行
@agent.tool_plain
def get_current_time() -> str:
    """获取当前时间。"""
    return datetime.now().isoformat()
```

工具的docstring会被提取并发送给LLM，作为LLM决定是否调用该工具的依据。因此，工具的docstring质量直接影响Agent的决策质量。一个常见的最佳实践是：**docstring要描述工具的用途和参数语义，而不仅仅是函数的功能**。

### 多步对话与状态管理

PydanticAI对多轮对话提供了原生支持，通过`message_history`参数实现上下文传递：

```python
from pydantic_ai.messages import ModelMessage

# 第一轮对话
result1 = travel_agent.run_sync('我想去日本旅行')

# 第二轮对话，传入第一轮的消息历史
result2 = travel_agent.run_sync(
    '预算控制在1万以内，行程缩短到3天',
    message_history=result1.all_messages(),
)

# result2的输出会考虑第一轮的上下文
print(result2.output.budget_level)  # 会反映预算限制
```

更复杂的场景是多Agent协作。你可以让一个Agent的输出作为另一个Agent的输入，形成Agent链：

```python
# 研究Agent：分析需求
research_agent = Agent(
    'openai:gpt-4o',
    result_type=ResearchReport,
    system_prompt='你是一个需求分析师，负责深入分析用户需求。'
)

# 规划Agent：基于研究结果制定方案
planning_agent = Agent(
    'openai:gpt-4o', 
    result_type=ProjectPlan,
    system_prompt='你是一个项目规划师，基于研究结果制定项目计划。'
)

# 链式调用
research = research_agent.run_sync('我们需要一个实时数据分析平台')
plan = planning_agent.run_sync(
    f'基于以下研究结果制定项目计划：\n{research.output.model_dump_json()}',
)

print(plan.output.milestones)  # 类型安全的项目里程碑
```

## 实战：构建一个生产级的客服Agent

让我们通过一个完整的例子来展示PydanticAI在实际场景中的应用。假设我们要构建一个电商客服Agent，它需要：

1. 理解用户问题并分类意图
2. 查询订单系统获取相关信息
3. 生成结构化的回复
4. 在需要时升级到人工客服

### 第一步：定义数据模型

```python
from pydantic import BaseModel, Field
from enum import Enum

class IntentType(str, Enum):
    ORDER_STATUS = "order_status"
    RETURN_REQUEST = "return_request"
    PRODUCT_INQUIRY = "product_inquiry"
    COMPLAINT = "complaint"
    GENERAL = "general"

class CustomerIntent(BaseModel):
    """Agent对用户意图的理解。"""
    intent: IntentType
    confidence: float = Field(ge=0, le=1)
    order_id: str | None = None
    product_name: str | None = None
    requires_human: bool = False
    reasoning: str

class AgentResponse(BaseModel):
    """Agent生成的结构化回复。"""
    message: str = Field(description="给用户的回复内容")
    actions: list[str] = Field(description="需要执行的后续动作")
    escalate: bool = Field(description="是否需要转人工")
    sentiment: str = Field(description="用户情绪判断: positive/neutral/negative")
```

注意每个字段的`description`——这不仅是文档，还会被PydanticAI传递给LLM，帮助LLM理解每个字段应该填什么。

### 第二步：定义依赖和工具

```python
@dataclass
class ServiceDeps:
    order_db: OrderDatabase
    product_db: ProductDatabase
    knowledge_base: KnowledgeBase
    notification_service: NotificationService

customer_agent = Agent(
    'openai:gpt-4o',
    system_prompt="""你是一个专业的电商客服。请根据用户的问题：
    1. 准确理解用户意图
    2. 查询相关信息
    3. 给出专业、友好的回复
    4. 如果问题无法自动解决，建议转人工
    
    重要规则：
    - 涉及退款金额超过500元必须转人工
    - 用户情绪激动时优先安抚并建议转人工
    - 始终保持礼貌和专业""",
    deps_type=ServiceDeps,
    result_type=AgentResponse,
)

@customer_agent.tool
async def query_order(ctx: RunContext[ServiceDeps], order_id: str) -> str:
    """查询订单详情。order_id为订单编号。"""
    order = await ctx.deps.order_db.get(order_id)
    if not order:
        return f"未找到订单 {order_id}"
    return f"订单{order_id}: 状态={order.status}, 金额={order.total}, 下单时间={order.created_at}"

@customer_agent.tool
async def search_product(ctx: RunContext[ServiceDeps], keyword: str) -> str:
    """搜索商品信息。keyword为搜索关键词。"""
    products = await ctx.deps.product_db.search(keyword, limit=3)
    return "\n".join(f"{p.name} - ¥{p.price} - {p.stock_status}" for p in products)

@customer_agent.tool
async def check_knowledge_base(ctx: RunContext[ServiceDeps], question: str) -> str:
    """查询知识库获取常见问题解答。"""
    answer = await ctx.deps.knowledge_base.search(question)
    return answer if answer else "未找到相关知识库内容"

@customer_agent.tool
async def send_notification(ctx: RunContext[ServiceDeps], user_id: str, message: str) -> str:
    """向用户发送通知消息。"""
    await ctx.deps.notification_service.send(user_id, message)
    return f"通知已发送给用户 {user_id}"
```

### 第三步：测试与验证

PydanticAI的一大优势是测试友好。由于依赖注入的存在，我们可以轻松地用mock替换外部系统：

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_order_inquiry():
    # 创建mock依赖
    mock_order_db = AsyncMock()
    mock_order_db.get.return_value = Order(
        id="ORD-12345",
        status="已发货",
        total=299.00,
        created_at="2026-05-28"
    )
    
    deps = ServiceDeps(
        order_db=mock_order_db,
        product_db=AsyncMock(),
        knowledge_base=AsyncMock(),
        notification_service=AsyncMock(),
    )
    
    result = await customer_agent.run(
        '我的订单ORD-12345到哪了？',
        deps=deps,
    )
    
    # 验证输出结构
    assert isinstance(result.output, AgentResponse)
    assert result.output.sentiment in ["positive", "neutral", "negative"]
    assert isinstance(result.output.actions, list)
    
    # 验证调用了正确的依赖
    mock_order_db.get.assert_called_once_with("ORD-12345")
```

这种测试方式的优势在于：你不需要真正调用LLM API就能验证Agent的业务逻辑。通过mock依赖，你可以精确控制每个工具的返回值，从而测试各种边界情况。

### 第四步：监控与可观测性

在生产环境中，PydanticAI提供了内置的结构化日志和追踪能力：

```python
from pydantic_ai.settings import settings

# 配置日志级别
settings.log_level = 'INFO'

# 每次调用都会记录结构化的调用日志
# 包括：输入、输出、token消耗、工具调用链路
result = await customer_agent.run(
    '我的订单到了吗？',
    deps=deps,
)

# 获取调用统计
print(f"Token消耗: {result.usage().total_tokens}")
print(f"工具调用次数: {len(result.all_messages())}")
```

## PydanticAI vs 其他框架的对比

| 特性 | PydanticAI | LangGraph | CrewAI | AutoGen |
|------|-----------|-----------|--------|---------|
| 类型安全 | ⭐⭐⭐⭐⭐ 原生Pydantic集成 | ⭐⭐⭐ 有类型支持 | ⭐⭐ 字典为主 | ⭐⭐ 动态类型 |
| 依赖注入 | ✅ 原生支持 | ⚠️ 需手动实现 | ❌ 不支持 | ❌ 不支持 |
| 学习曲线 | 中等（需熟悉Pydantic） | 较高 | 低 | 低 |
| 多Agent支持 | ✅ 通过Agent组合 | ✅ 图结构编排 | ✅ 角色定义 | ✅ 对话模式 |
| 结构化输出 | ✅ 类型约束 | ⚠️ 需配置 | ⚠️ 可选 | ⚠️ 需配置 |
| 测试友好度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 生产就绪度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 社区生态 | 成长中 | 成熟 | 成熟 | 成熟 |
| 适用场景 | 需要类型安全的生产系统 | 复杂工作流编排 | 多角色协作 | 研究/实验 |

值得注意的是，PydanticAI并不是要取代这些框架，而是填补了一个特定的生态位：**当你需要Agent系统具有强类型约束和高可测试性时**。

## 最佳实践与踩坑经验

### 1. 善用嵌套模型描述复杂输出

```python
class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    details: list[AnalysisDetail]
    recommendations: list[Recommendation]
    metadata: AnalysisMetadata

# 让LLM直接生成完整的嵌套结构
agent = Agent('openai:gpt-4o', result_type=AnalysisResult)
```

嵌套越深，LLM出错的概率越大。对于超过2层的嵌套结构，建议拆分成多个Agent，每个Agent负责一层。

### 2. 合理设计Prompt与类型的关系

类型定义不只是数据结构，它也是给LLM的隐式指令。字段名和描述会影响LLM的输出质量：

```python
class ProductReview(BaseModel):
    # 好的字段定义：明确语义和约束
    sentiment_score: float = Field(
        ge=0, le=1,
        description="情感评分，0为极度负面，1为极度正面"
    )
    key_issues: list[str] = Field(
        description="用户反馈的核心问题，最多列出3个"
    )
    
    # 差的字段定义：语义模糊
    # score: float  # 什么score？范围？含义？
    # issues: list  # 什么issues？什么类型？
```

### 3. 异常处理策略

```python
from pydantic_ai import Agent
from pydantic import ValidationError
from pydantic_ai.exceptions import ModelHTTPError

agent = Agent('openai:gpt-4o', result_type=AgentResponse)

try:
    result = await agent.run(user_input, deps=deps)
except ValidationError as e:
    # 类型验证失败 - LLM输出不符合预期结构
    # 这在retry用尽后才会抛出
    logger.warning(f"Output validation failed after retries: {e}")
    # 可以降级到更简单的输出格式
except ModelHTTPError as e:
    # LLM API调用失败
    logger.error(f"LLM API error: {e}")
    # 可以切换到备用模型
except Exception as e:
    # 其他未预期错误
    logger.error(f"Unexpected error: {e}")
```

### 4. 利用retries机制处理LLM输出不稳定

PydanticAI内置了重试机制，当输出不符合类型约束时会自动重试：

```python
agent = Agent(
    'openai:gpt-4o',
    result_type=AgentResponse,
    retries=3,  # 类型验证失败时最多重试3次
)
```

重试的默认行为是将验证失败的错误信息反馈给LLM，让它修正输出。这比手动实现重试循环要优雅得多。

### 5. 模型选择策略

PydanticAI支持多种LLM提供商，选择合适的模型对成本和效果有重大影响：

```python
# 简单任务用小模型
simple_agent = Agent('openai:gpt-4o-mini', result_type=SimpleResponse)

# 复杂推理用大模型
complex_agent = Agent('openai:gpt-4o', result_type=ComplexAnalysis)

# 也可以用本地模型
local_agent = Agent('ollama:llama3.1', result_type=LocalResponse)

# 甚至用Anthropic的模型
claude_agent = Agent('anthropic:claude-3.5-sonnet', result_type=ClaudeResponse)
```

## 适用场景总结

PydanticAI最适合以下场景：

1. **生产级Agent系统**：需要严格类型检查和错误处理
2. **多Agent协作**：Agent之间的数据传递需要明确契约
3. **团队协作开发**：类型定义就是最好的文档，新成员看类型就知道数据结构
4. **需要高测试覆盖率的项目**：依赖注入让mock测试变得自然
5. **API服务封装**：Agent作为服务暴露给前端或其他系统

如果你的Agent系统还在原型探索阶段，或者主要使用Python字典传递数据，LangGraph或CrewAI可能更适合快速验证想法。但一旦决定走向生产，PydanticAI的类型安全特性会为你节省大量的调试和维护成本。

## 总结

PydanticAI的核心贡献是将类型系统的严谨性引入了LLM应用开发。它不是要替代LangGraph或CrewAI，而是提供了一种不同的思考方式：**通过类型约束来保证Agent行为的可预测性**。在AI应用从实验走向生产的过程中，这种工程化的思维方式会越来越重要。

对于已经在使用Pydantic的团队来说，PydanticAI几乎是零学习成本的升级。即使不使用Pydantic，它也值得作为一个参考设计，启发我们思考如何在AI应用中引入更强的类型约束和更清晰的依赖管理。

技术选型没有绝对的对错，关键是理解每个框架的设计取舍。PydanticAI选择了"类型安全优先"这条路，这条路上的风景，值得每个认真对待AI工程化的团队去探索。
