---
title: "AI系统架构模式全景：从单体到事件驱动，LLM应用的架构选型指南"
description: "系统性梳理LLM应用的六大架构模式——单体、分层、管道、事件驱动、Actor模型、混合架构，覆盖选型决策、性能对比与实战代码"
date: 2026-06-01
author: "RiceBall-15"
category: "featured"
subCategory: ai-architecture
tags: ["AI架构", "LLM应用", "事件驱动", "Actor模型", "架构选型", "系统设计"]
draft: false
---

# AI系统架构模式全景：从单体到事件驱动，LLM应用的架构选型指南

## 引言：为什么LLM应用需要新的架构模式？

传统的软件架构模式（MVC、微服务、Serverless）是为**确定性计算**设计的。但LLM应用本质上是**概率性计算**——输出不可预测、延迟波动大、资源消耗非线性。这意味着我们需要重新审视架构选型。

```
┌─────────────────────────────────────────────────────────────────────┐
│              传统软件 vs LLM应用 的架构需求差异                         │
├───────────────────────────────┬─────────────────────────────────────┤
│         传统软件               │          LLM应用                    │
├───────────────────────────────┼─────────────────────────────────────┤
│  输入→确定性处理→输出          │  输入→概率性推理→输出(可能不一致)    │
│  延迟: 稳定、可预测            │  延迟: 波动大、长尾分布              │
│  资源: CPU/内存线性可预测       │  资源: GPU显存非线性、KV Cache状态   │
│  错误: 确定性错误码            │  错误: 幻觉、逻辑错误、拒绝回答      │
│  扩展: 水平扩展简单            │  扩展: GPU限制、显存瓶颈             │
│  调试: 日志+断点               │  调试: Prompt追溯+语义分析           │
│  一致性: ACID/最终一致性       │  一致性: 语义一致性(更难)            │
└───────────────────────────────┴─────────────────────────────────────┘
```

本文将系统性梳理六大架构模式，帮你做出正确的架构选型决策。

## 一、六大架构模式总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                  LLM应用架构模式演进路线                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  复杂度递增 ─────────────────────────────────────────────▶           │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  单体     │  │  分层     │  │  管道     │  │ 事件驱动  │          │
│  │ Monolith │→ │ Layered  │→ │ Pipeline │→ │ Event-   │          │
│  │          │  │          │  │          │  │ Driven   │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
│       │              │              │              │                 │
│       ▼              ▼              ▼              ▼                 │
│  ┌──────────┐  ┌──────────┐                                          │
│  │  Actor   │  │  混合     │                                          │
│  │  Model   │←─│  Hybrid  │                                          │
│  │          │  │          │                                          │
│  └──────────┘  └──────────┘                                          │
│                                                                     │
│  适用规模:                                                           │
│  单体: 1个Agent, <100 RPS                                            │
│  分层: 2-5个Agent, 100-1000 RPS                                      │
│  管道: 线性工作流, 100-5000 RPS                                       │
│  事件驱动: 异步场景, 1000+ RPS                                        │
│  Actor: 高并发Agent协作, 5000+ RPS                                   │
│  混合: 复杂生产环境, 企业级规模                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 二、模式1: 单体架构 (Monolith)

### 2.1 架构设计

单体架构是LLM应用最简单的起步方式，所有组件（Prompt管理、LLM调用、工具调用、输出解析）在单一进程中运行。

```
┌─────────────────────────────────────────┐
│          单体LLM应用架构                  │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐  │
│  │          API Gateway              │  │
│  └───────────────┬───────────────────┘  │
│                  ▼                       │
│  ┌───────────────────────────────────┐  │
│  │         LLM Application           │  │
│  │  ┌─────────┐ ┌─────────┐         │  │
│  │  │ Prompt  │ │  Tool   │         │  │
│  │  │ Manager │ │ Registry│         │  │
│  │  └─────────┘ └─────────┘         │  │
│  │  ┌─────────────────────────┐     │  │
│  │  │    LLM Client           │     │  │
│  │  │  (OpenAI/Local/Custom)  │     │  │
│  │  └─────────────────────────┘     │  │
│  │  ┌─────────┐ ┌─────────┐         │  │
│  │  │ Output  │ │ Memory  │         │  │
│  │  │ Parser  │ │ Store   │         │  │
│  │  └─────────┘ └─────────┘         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  SQLite / In-Memory Storage       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 2.2 核心实现

```python
# monolith_agent.py - 单体Agent完整实现
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import json
import hashlib

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable

class MonolithAgent:
    """
    单体LLM Agent - 所有逻辑在一个类中
    适用: 快速原型、小规模应用 (<100 RPS)
    """
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.tools: List[Tool] = []
        self.memory: List[Dict] = []
        self.max_memory = 100
        self._cache = {}  # 简单缓存

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools.append(tool)

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools
        )
        return f"""You are a helpful AI assistant.
Available tools:
{tool_descriptions}

When you need to use a tool, respond with a JSON block:
```json
{{"tool": "tool_name", "args": {{"param": "value"}}}}
```

After receiving tool results, continue the conversation naturally."""

    def _check_cache(self, user_input: str) -> Optional[str]:
        """简单缓存 - 相同输入返回缓存结果"""
        key = hashlib.md5(
            f"{user_input}:{self.model}:{self.temperature}".encode()
        ).hexdigest()
        return self._cache.get(key)

    def _set_cache(self, user_input: str, response: str, ttl: int = 300):
        """设置缓存"""
        key = hashlib.md5(
            f"{user_input}:{self.model}:{self.temperature}".encode()
        ).hexdigest()
        self._cache[key] = response

    async def chat(self, user_input: str) -> str:
        """主对话循环"""
        # 1. 检查缓存
        cached = self._check_cache(user_input)
        if cached:
            return cached

        # 2. 构建消息
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *self.memory,
            {"role": "user", "content": user_input},
        ]

        # 3. LLM调用（含工具调用循环）
        max_iterations = 5
        for _ in range(max_iterations):
            response = await self._call_llm(messages)

            # 检查是否需要工具调用
            tool_call = self._parse_tool_call(response)
            if not tool_call:
                # 最终回复
                self._update_memory(user_input, response)
                self._set_cache(user_input, response)
                return response

            # 执行工具
            tool_result = await self._execute_tool(tool_call)
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result: {tool_result}"
            })

        return "I've reached the maximum number of tool calls."

    async def _call_llm(self, messages: List[Dict]) -> str:
        """LLM调用 - 可替换为任何LLM提供商"""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                },
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                timeout=60,
            )
            return resp.json()["choices"][0]["message"]["content"]

    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """从LLM输出中解析工具调用"""
        if "```json" in response:
            try:
                start = response.index("```json") + 7
                end = response.index("```", start)
                return json.loads(response[start:end])
            except (ValueError, json.JSONDecodeError):
                return None
        return None

    async def _execute_tool(self, tool_call: Dict) -> str:
        """执行工具"""
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = await tool.func(**args)
                    return json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    return f"Error: {str(e)}"
        return f"Tool '{tool_name}' not found"

    def _update_memory(self, user_input: str, response: str):
        """更新对话记忆"""
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": response})
        # 滑动窗口
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]


# 使用示例
agent = MonolithAgent(model="gpt-4o")

async def search_web(query: str) -> dict:
    """搜索工具"""
    return {"results": [f"Result for: {query}"]}

agent.register_tool(Tool(
    name="search_web",
    description="Search the web for information",
    parameters={"query": {"type": "string"}},
    func=search_web,
))
```

### 2.3 单体架构的适用边界

| 维度 | 评估 | 说明 |
|------|------|------|
| 开发速度 | ⭐⭐⭐⭐⭐ | 最快上手，一个文件搞定 |
| 运维成本 | ⭐⭐⭐⭐⭐ | 无需K8s，单进程部署 |
| 可扩展性 | ⭐⭐ | 垂直扩展为主，水平扩展需重复部署 |
| 可观测性 | ⭐⭐ | 日志为主，缺乏结构化监控 |
| 容错性 | ⭐ | 单点故障，进程崩溃全部丢失 |
| **适用场景** | MVP验证、个人工具、内部小工具 |

## 三、模式2: 分层架构 (Layered)

### 3.1 架构设计

分层架构将LLM应用分为清晰的层次，每层职责单一，便于测试和维护。

```
┌───────────────────────────────────────────────────────────────┐
│                    分层LLM应用架构                              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 4: Presentation Layer (展示层)                          │
│  ┌───────────────────────────────────────────────────┐       │
│  │  REST API / WebSocket / CLI / Web UI              │       │
│  └───────────────────────┬───────────────────────────┘       │
│                          ▼                                    │
│  Layer 3: Orchestration Layer (编排层)                         │
│  ┌───────────────────────────────────────────────────┐       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │       │
│  │  │ Workflow │  │  Agent   │  │  Chain   │       │       │
│  │  │ Engine   │  │ Router   │  │ Builder  │       │       │
│  │  └──────────┘  └──────────┘  └──────────┘       │       │
│  └───────────────────────┬───────────────────────────┘       │
│                          ▼                                    │
│  Layer 2: Intelligence Layer (智能层)                          │
│  ┌───────────────────────────────────────────────────┐       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │       │
│  │  │  Prompt  │  │  LLM     │  │  Output  │       │       │
│  │  │ Template │  │  Client  │  │  Parser  │       │       │
│  │  └──────────┘  └──────────┘  └──────────┘       │       │
│  └───────────────────────┬───────────────────────────┘       │
│                          ▼                                    │
│  Layer 1: Infrastructure Layer (基础设施层)                     │
│  ┌───────────────────────────────────────────────────┐       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │       │
│  │  │  Cache   │  │  Queue   │  │  Storage │       │       │
│  │  │ (Redis)  │  │(Redis Q) │  │  (DB)    │       │       │
│  │  └──────────┘  └──────────┘  └──────────┘       │       │
│  └───────────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 核心实现

```python
# layered_agent.py - 分层架构实现
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

# ====== Layer 1: Infrastructure ======
class CacheProvider(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[str]: ...
    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = 300): ...

class RedisCache(CacheProvider):
    def __init__(self, redis_client):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(key)

    async def set(self, key: str, value: str, ttl: int = 300):
        await self.redis.setex(key, ttl, value)

# ====== Layer 2: Intelligence ======
class PromptTemplate:
    """Prompt模板管理"""
    def __init__(self, template: str, variables: list):
        self.template = template
        self.variables = variables

    def render(self, **kwargs) -> str:
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

class LLMClient(ABC):
    @abstractmethod
    async def complete(self, messages: list, **kwargs) -> str: ...

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    async def complete(self, messages: list, **kwargs) -> str:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": self.model, "messages": messages, **kwargs},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60,
            )
            return resp.json()["choices"][0]["message"]["content"]

class OutputParser(ABC):
    @abstractmethod
    def parse(self, raw_output: str) -> Dict[str, Any]: ...

class JSONOutputParser(OutputParser):
    def parse(self, raw_output: str) -> Dict[str, Any]:
        try:
            if "```json" in raw_output:
                start = raw_output.index("```json") + 7
                end = raw_output.index("```", start)
                return json.loads(raw_output[start:end])
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {"raw": raw_output, "error": "parse_failed"}

# ====== Layer 3: Orchestration ======
class AgentRouter:
    """基于意图的Agent路由"""
    def __init__(self):
        self.routes: Dict[str, callable] = {}

    def register(self, intent: str, handler: callable):
        self.routes[intent] = handler

    async def route(self, user_input: str, intent: str) -> Any:
        handler = self.routes.get(intent, self.routes.get("default"))
        return await handler(user_input)

# ====== Layer 4: Presentation ======
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    latency_ms: float

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # 编排层逻辑
    import time
    start = time.time()

    # 1. 意图识别
    intent = await intent_classifier.classify(req.message)

    # 2. 路由到对应Agent
    result = await agent_router.route(req.message, intent)

    latency = (time.time() - start) * 1000
    return ChatResponse(
        response=result,
        intent=intent,
        latency_ms=latency,
    )
```

### 3.3 分层架构 vs 单体架构

| 维度 | 单体架构 | 分层架构 |
|------|---------|---------|
| **代码组织** | 一个文件 | 4层分离 |
| **可测试性** | 端到端测试 | 每层独立单元测试 |
| **可替换性** | 改一处动全身 | 替换某一层不影响其他层 |
| **开发效率** | 快（原型） | 中等（需设计） |
| **运行时开销** | 低 | 层间调用有少量开销 |
| **学习曲线** | 低 | 中等 |
| **适用规模** | <100 RPS | 100-1000 RPS |

## 四、模式3: 管道架构 (Pipeline)

### 4.1 架构设计

管道架构将LLM处理流程拆分为独立的**阶段**，每个阶段可独立扩展和替换。这是RAG系统和多步推理的天然选择。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM管道架构 (Pipeline)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  用户输入                                                           │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ Stage 1  │───▶│ Stage 2  │───▶│ Stage 3  │───▶│ Stage 4  │     │
│  │ 意图识别  │    │ 检索增强  │    │ 推理生成  │    │ 输出过滤  │     │
│  │ Intent   │    │  RAG     │    │ Generate │    │  Filter  │     │
│  │ Classify │    │ Retrieve │    │          │    │          │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│      │               │               │               │              │
│      ▼               ▼               ▼               ▼              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ Metrics  │    │ Metrics  │    │ Metrics  │    │ Metrics  │     │
│  │ & Trace  │    │ & Trace  │    │ & Trace  │    │ & Trace  │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                                                                     │
│  管道特点:                                                          │
│  • 每个Stage可独立扩展副本数                                        │
│  • Stage间通过消息队列解耦                                          │
│  • 支持条件分支 (if/else) 和循环 (loop)                             │
│  • 失败重试在Stage级别独立处理                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 核心实现

```python
# pipeline_agent.py - 管道架构实现
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio
import time
import json

@dataclass
class PipelineContext:
    """管道上下文 - 在Stage间传递数据"""
    user_input: str = ""
    intent: str = ""
    retrieved_docs: List[Dict] = field(default_factory=list)
    generated_text: str = ""
    filtered_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)

class PipelineStage(ABC):
    """管道阶段抽象基类"""
    name: str

    @abstractmethod
    async def process(self, ctx: PipelineContext) -> PipelineContext:
        ...

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        """包装process方法，自动记录耗时"""
        start = time.time()
        ctx = await self.process(ctx)
        elapsed = (time.time() - start) * 1000
        ctx.stage_timings[self.name] = elapsed
        return ctx

class LLMPipeline:
    """LLM管道引擎"""
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.error_handlers: Dict[str, callable] = {}

    def add_stage(self, stage: PipelineStage) -> "LLMPipeline":
        self.stages.append(stage)
        return self  # 支持链式调用

    def on_error(self, stage_name: str, handler: callable):
        self.error_handlers[stage_name] = handler

    async def execute(self, user_input: str) -> PipelineContext:
        ctx = PipelineContext(user_input=user_input)

        for stage in self.stages:
            try:
                ctx = await stage.execute(ctx)
            except Exception as e:
                if stage.name in self.error_handlers:
                    ctx = await self.error_handlers[stage.name](ctx, e)
                else:
                    raise PipelineError(
                        f"Stage '{stage.name}' failed: {e}", stage.name
                    )

        return ctx

# ====== 具体Stage实现 ======

class IntentClassificationStage(PipelineStage):
    name = "intent_classification"

    def __init__(self, llm_client, intent_prompt: str):
        self.llm = llm_client
        self.prompt = intent_prompt

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        response = await self.llm.complete([{
            "role": "system",
            "content": self.prompt
        }, {
            "role": "user",
            "content": ctx.user_input
        }])
        ctx.intent = response.strip()
        ctx.metadata["intent_confidence"] = 0.85  # 简化
        return ctx

class RAGRetrievalStage(PipelineStage):
    name = "rag_retrieval"

    def __init__(self, vector_store, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        # 根据意图调整检索策略
        search_params = self._get_search_params(ctx.intent)
        docs = await self.vector_store.search(
            query=ctx.user_input,
            top_k=self.top_k,
            **search_params,
        )
        ctx.retrieved_docs = docs
        ctx.metadata["docs_count"] = len(docs)
        return ctx

    def _get_search_params(self, intent: str) -> dict:
        strategies = {
            "factual": {"similarity_threshold": 0.8, "rerank": True},
            "creative": {"similarity_threshold": 0.5, "rerank": False},
            "code": {"filter": {"type": "code"}, "rerank": True},
        }
        return strategies.get(intent, {"rerank": False})

class GenerationStage(PipelineStage):
    name = "generation"

    def __init__(self, llm_client, max_tokens: int = 2000):
        self.llm = llm_client
        self.max_tokens = max_tokens

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        # 构建RAG上下文
        context = "\n\n".join([
            f"[Doc {i+1}]: {d.get('content', '')}"
            for i, d in enumerate(ctx.retrieved_docs[:5])
        ])

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {ctx.user_input}"},
        ]

        ctx.generated_text = await self.llm.complete(
            messages, max_tokens=self.max_tokens
        )
        return ctx

class OutputFilterStage(PipelineStage):
    name = "output_filter"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        # 过滤敏感内容、幻觉检测
        text = ctx.generated_text

        # 1. 简单幻觉检测: 检查是否引用了不存在的文档
        if ctx.retrieved_docs:
            # 实际应用中使用NLI模型
            pass

        # 2. 敏感词过滤
        sensitive_patterns = ["password", "secret", "api_key"]
        for pattern in sensitive_patterns:
            if pattern.lower() in text.lower():
                text = text.replace(pattern, "[REDACTED]")

        # 3. 格式标准化
        ctx.filtered_text = text.strip()
        return ctx

# ====== 组装管道 ======
pipeline = LLMPipeline()
pipeline.add_stage(IntentClassificationStage(llm, INTENT_PROMPT))
pipeline.add_stage(RAGRetrievalStage(vector_store, top_k=5))
pipeline.add_stage(GenerationStage(llm, max_tokens=2000))
pipeline.add_stage(OutputFilterStage())
```

### 4.3 管道架构的高级特性

```
┌──────────────────────────────────────────────────────────────┐
│              管道架构高级特性                                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 条件分支 (Conditional Branching)                          │
│  ┌─────────┐                                                │
│  │ Intent  │─── "code" ───▶ Code Generation Stage           │
│  │ Router  │─── "chat" ───▶ Chat Generation Stage           │
│  └─────────┘─── "rag"  ───▶ RAG Generation Stage            │
│                                                              │
│  2. 并行执行 (Parallel Execution)                             │
│  ┌──────────────────────────────┐                           │
│  │      Parallel Stage Group    │                           │
│  │  ┌──────┐ ┌──────┐ ┌──────┐│                           │
│  │  │Web   │ │Vector│ │SQL   ││ ──▶ Merge Stage            │
│  │  │Search│ │Search│ │Query ││                           │
│  │  └──────┘ └──────┘ └──────┘│                           │
│  └──────────────────────────────┘                           │
│                                                              │
│  3. 循环重试 (Loop with Retry)                                │
│  ┌──────────────────────────────────┐                       │
│  │  Generate → Validate → Pass?     │                       │
│  │      ▲         │ No    │ Yes     │                       │
│  │      └─────────┘       ▼         │                       │
│  │                        Output    │                       │
│  │  Max retries: 3                  │                       │
│  └──────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

## 五、模式4: 事件驱动架构 (Event-Driven)

### 5.1 架构设计

事件驱动架构（EDA）将Agent间通信完全解耦，通过事件总线实现异步协作。这是高并发、大规模Agent系统的首选。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    事件驱动LLM架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐      │
│  │ Planner   │  │ Coder     │  │ Reviewer  │  │ Deployer  │      │
│  │ Agent     │  │ Agent     │  │ Agent     │  │ Agent     │      │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘      │
│        │publish        │publish        │publish        │publish     │
│        ▼               ▼               ▼               ▼           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Event Bus (Kafka/NATS)                    │  │
│  │                                                             │  │
│  │  Topics:                                                    │  │
│  │  • agent.task.created     - 新任务创建                       │  │
│  │  • agent.task.completed   - 任务完成                         │  │
│  │  • agent.code.generated   - 代码生成                         │  │
│  │  • agent.review.passed    - 审查通过                         │  │
│  │  • agent.review.failed    - 审查失败                         │  │
│  │  • agent.deployment.done  - 部署完成                         │  │
│  │  • agent.error.occurred   - 错误发生                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│        │subscribe       │subscribe       │subscribe                │
│        ▼               ▼               ▼                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                    │
│  │ State     │  │ Metrics   │  │ Alert     │                    │
│  │ Store     │  │ Collector │  │ Handler   │                    │
│  └───────────┘  └───────────┘  └───────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 核心实现

```python
# event_driven_agent.py - 事件驱动架构实现
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
from enum import Enum
import json
import time
from collections import defaultdict

class EventType(str, Enum):
    TASK_CREATED = "agent.task.created"
    TASK_COMPLETED = "agent.task.completed"
    CODE_GENERATED = "agent.code.generated"
    REVIEW_PASSED = "agent.review.passed"
    REVIEW_FAILED = "agent.review.failed"
    DEPLOYMENT_DONE = "agent.deployment.done"
    ERROR_OCCURRED = "agent.error.occurred"

@dataclass
class Event:
    event_type: EventType
    payload: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.source}:{self.event_type}:{int(self.timestamp*1000)}"

class EventBus:
    """
    事件总线 - 内存实现 (生产环境用Kafka/NATS)
    支持: 发布/订阅、事件过滤、死信队列
    """
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._dead_letter_queue: List[Event] = []
        self._event_log: List[Event] = []
        self._max_log_size = 10000

    def subscribe(self, event_type: EventType, handler: Callable):
        """订阅事件"""
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: Callable):
        """订阅所有事件"""
        for et in EventType:
            self._subscribers[et].append(handler)

    async def publish(self, event: Event):
        """发布事件"""
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Handler error for {event.event_type}: {e}")
                self._dead_letter_queue.append(event)

    def get_event_history(self, event_type: EventType = None, limit: int = 50) -> List[Event]:
        """获取事件历史"""
        if event_type:
            events = [e for e in self._event_log if e.event_type == event_type]
        else:
            events = self._event_log
        return events[-limit:]

# ====== Agent实现 ======

class BaseAgent:
    """Agent基类"""
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self._setup_subscriptions()

    def _setup_subscriptions(self):
        """子类重写，设置事件订阅"""
        pass

    async def emit(self, event_type: EventType, payload: dict):
        """发布事件"""
        event = Event(
            event_type=event_type,
            payload=payload,
            source=self.name,
        )
        await self.event_bus.publish(event)

class PlannerAgent(BaseAgent):
    def _setup_subscriptions(self):
        self.event_bus.subscribe(EventType.TASK_CREATED, self.handle_task)

    async def handle_task(self, event: Event):
        """处理新任务 - 拆分为子任务"""
        task = event.payload
        subtasks = await self._decompose_task(task)

        for subtask in subtasks:
            await self.emit(EventType.TASK_CREATED, {
                "parent_id": task["id"],
                "subtask": subtask,
                "assigned_to": subtask.get("assignee", "coder"),
            })

    async def _decompose_task(self, task: dict) -> list:
        """任务拆分逻辑"""
        return [
            {"id": "sub-1", "type": "code", "assignee": "coder-agent"},
            {"id": "sub-2", "type": "test", "assignee": "tester-agent"},
        ]

class CoderAgent(BaseAgent):
    def _setup_subscriptions(self):
        self.event_bus.subscribe(EventType.TASK_CREATED, self.handle_task)
        self.event_bus.subscribe(EventType.REVIEW_FAILED, self.handle_review_failure)

    async def handle_task(self, event: Event):
        """处理编码任务"""
        if event.payload.get("assigned_to") != self.name:
            return

        code = await self._generate_code(event.payload)
        await self.emit(EventType.CODE_GENERATED, {
            "task_id": event.payload["id"],
            "code": code,
            "language": "python",
        })

    async def handle_review_failure(self, event: Event):
        """审查失败后重新生成"""
        task_id = event.payload["task_id"]
        feedback = event.payload["feedback"]
        new_code = await self._regenerate_with_feedback(task_id, feedback)
        await self.emit(EventType.CODE_GENERATED, {
            "task_id": task_id,
            "code": new_code,
            "revision": event.payload.get("revision", 1) + 1,
        })

    async def _generate_code(self, task: dict) -> str:
        return f"# Generated code for {task['id']}"

    async def _regenerate_with_feedback(self, task_id: str, feedback: str) -> str:
        return f"# Revised code for {task_id} based on: {feedback}"

class ReviewerAgent(BaseAgent):
    def _setup_subscriptions(self):
        self.event_bus.subscribe(EventType.CODE_GENERATED, self.handle_code)

    async def handle_code(self, event: Event):
        """审查代码"""
        passed = await self._review(event.payload["code"])
        if passed:
            await self.emit(EventType.REVIEW_PASSED, {
                "task_id": event.payload["task_id"],
            })
        else:
            await self.emit(EventType.REVIEW_FAILED, {
                "task_id": event.payload["task_id"],
                "feedback": "Code quality issues found",
            })

    async def _review(self, code: str) -> bool:
        return len(code) > 10  # 简化判断

# ====== 组装系统 ======
async def run_event_driven_system():
    bus = EventBus()
    planner = PlannerAgent("planner-agent", bus)
    coder = CoderAgent("coder-agent", bus)
    reviewer = ReviewerAgent("reviewer-agent", bus)

    # 启动: 创建一个新任务
    await bus.publish(Event(
        event_type=EventType.TASK_CREATED,
        payload={"id": "task-001", "description": "Build a REST API"},
        source="user",
    ))
```

### 5.3 事件驱动 vs 管道架构

| 维度 | 管道架构 | 事件驱动架构 |
|------|---------|------------|
| **数据流向** | 单向线性 | 多向网状 |
| **耦合度** | Stage间半耦合 | 完全解耦 |
| **扩展方式** | 热插拔Stage | 独立部署Agent |
| **错误处理** | Stage级重试 | 事件级重试+死信队列 |
| **调试难度** | 中等（链路清晰） | 高（需分布式追踪） |
| **适用场景** | 线性工作流(RAG) | 异步协作(多Agent) |
| **吞吐量** | 中等 | 高（天然异步） |

## 六、模式5: Actor模型

### 6.1 架构设计

Actor模型将每个Agent视为独立的Actor，拥有自己的状态和邮箱，通过消息传递通信。这是最接近人类团队协作模式的架构。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Actor模型LLM架构                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │                      Actor System                         │     │
│  │                                                           │     │
│  │  ┌──────────────────┐    ┌──────────────────┐            │     │
│  │  │   Actor A        │    │   Actor B        │            │     │
│  │  │ ┌──────────────┐ │    │ ┌──────────────┐ │            │     │
│  │  │ │   State      │ │    │ │   State      │ │            │     │
│  │  │ │ - memory     │ │    │ │ - memory     │ │            │     │
│  │  │ │ - context    │ │    │ │ - context    │ │            │     │
│  │  │ └──────────────┘ │    │ └──────────────┘ │            │     │
│  │  │ ┌──────────────┐ │    │ ┌──────────────┐ │            │     │
│  │  │ │   Mailbox    │ │    │ │   Mailbox    │ │            │     │
│  │  │ │ [msg1,msg2]  │◀┼────┼▶│ [msg3]       │ │            │     │
│  │  │ └──────────────┘ │    │ └──────────────┘ │            │     │
│  │  │   behavior:      │    │   behavior:      │            │     │
│  │  │   receive(msg):  │    │   receive(msg):  │            │     │
│  │  │     process()    │    │     process()    │            │     │
│  │  │     reply()      │    │     reply()      │            │     │
│  │  │     forward()    │    │     forward()    │            │     │
│  │  └──────────────────┘    └──────────────────┘            │     │
│  │           │                        │                      │     │
│  │           └──────────┬─────────────┘                      │     │
│  │                      ▼                                    │     │
│  │              ┌──────────────┐                             │     │
│  │              │  Actor C     │                             │     │
│  │              │ (Supervisor) │                             │     │
│  │              │  监督子Actor  │                             │     │
│  │              └──────────────┘                             │     │
│  └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 核心实现

```python
# actor_agent.py - Actor模型实现
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import time
import uuid

class ActorState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class Message:
    sender: str
    receiver: str
    content: Dict[str, Any]
    msg_type: str = "request"
    reply_to: Optional[str] = None
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

class Actor:
    """
    Actor基类 - 拥有独立状态、邮箱和行为
    """
    def __init__(self, actor_id: str, supervisor: Optional["Actor"] = None):
        self.actor_id = actor_id
        self.state = ActorState.IDLE
        self.mailbox: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.supervisor = supervisor
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._behaviors: Dict[str, Callable] = {}
        self._state_data: Dict[str, Any] = {}  # Actor私有状态

    def behavior(self, msg_type: str):
        """装饰器: 注册消息处理行为"""
        def decorator(func):
            self._behaviors[msg_type] = func
            return func
        return decorator

    async def start(self):
        """启动Actor消息循环"""
        self._running = True
        self._task = asyncio.create_task(self._message_loop())

    async def stop(self):
        """停止Actor"""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _message_loop(self):
        """消息处理循环"""
        while self._running:
            try:
                msg: Message = await asyncio.wait_for(
                    self.mailbox.get(), timeout=1.0
                )
                self.state = ActorState.PROCESSING

                handler = self._behaviors.get(msg.msg_type)
                if handler:
                    try:
                        await handler(msg)
                    except Exception as e:
                        self.state = ActorState.FAILED
                        if self.supervisor:
                            await self.supervisor.send(Message(
                                sender=self.actor_id,
                                receiver=self.supervisor.actor_id,
                                content={
                                    "type": "actor_failed",
                                    "actor_id": self.actor_id,
                                    "error": str(e),
                                },
                                msg_type="failure_report",
                            ))
                        else:
                            print(f"Actor {self.actor_id} failed: {e}")
                else:
                    print(f"No handler for msg_type: {msg.msg_type}")

                self.state = ActorState.IDLE

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def send(self, msg: Message):
        """发送消息到Actor邮箱"""
        await self.mailbox.put(msg)

    def tell(self, target: "Actor", msg_type: str, content: dict):
        """Fire-and-forget消息发送"""
        asyncio.create_task(target.send(Message(
            sender=self.actor_id,
            receiver=target.actor_id,
            content=content,
            msg_type=msg_type,
        )))

    async def ask(self, target: "Actor", msg_type: str, content: dict, timeout: float = 30) -> Any:
        """Request-Reply模式"""
        reply_queue = asyncio.Queue()
        msg = Message(
            sender=self.actor_id,
            receiver=target.actor_id,
            content=content,
            msg_type=msg_type,
            reply_to=self.actor_id,
        )
        # 注册临时回复处理器
        reply_handler = lambda m: reply_queue.put_nowait(m)
        target._behaviors[f"reply_{msg.msg_id}"] = reply_handler

        await target.send(msg)
        try:
            reply = await asyncio.wait_for(reply_queue.get(), timeout=timeout)
            return reply.content
        except asyncio.TimeoutError:
            return {"error": "timeout"}


# ====== 具体Agent Actor实现 ======

class PlannerActor(Actor):
    def __init__(self):
        super().__init__("planner")
        self._state_data["task_count"] = 0

    @Actor.behavior("plan_task")
    async def handle_plan(self, msg: Message):
        """规划任务"""
        self._state_data["task_count"] += 1
        task = msg.content["task"]

        # 拆分任务
        subtasks = [
            {"id": f"sub-{self._state_data['task_count']}-{i}", "desc": f"Part {i+1}: {task}"}
            for i in range(3)
        ]

        # 回复发送者
        if msg.reply_to:
            print(f"Planner: sending reply to {msg.reply_to}")

class CoderActor(Actor):
    def __init__(self):
        super().__init__("coder")
        self._state_data["code_count"] = 0

    @Actor.behavior("generate_code")
    async def handle_generate(self, msg: Message):
        """生成代码"""
        self._state_data["code_count"] += 1
        task = msg.content
        # 模拟代码生成
        code = f"# Code for: {task.get('desc', 'unknown')}\nprint('Hello')"
        print(f"Coder: generated code ({len(code)} chars)")

class SupervisorActor(Actor):
    """监督者Actor - 管理子Actor生命周期"""
    def __init__(self):
        super().__init__("supervisor")
        self._state_data["children"] = {}

    def supervise(self, child: Actor):
        child.supervisor = self
        self._state_data["children"][child.actor_id] = child

    @Actor.behavior("failure_report")
    async def handle_failure(self, msg: Message):
        """处理子Actor失败"""
        actor_id = msg.content["actor_id"]
        error = msg.content["error"]
        print(f"Supervisor: actor {actor_id} failed: {error}")

        # 重启策略
        child = self._state_data["children"].get(actor_id)
        if child:
            child.state = ActorState.IDLE
            print(f"Supervisor: restarted {actor_id}")

# ====== Actor系统示例 ======
async def run_actor_system():
    supervisor = SupervisorActor()
    planner = PlannerActor()
    coder = CoderActor()

    supervisor.supervise(planner)
    supervisor.supervise(coder)

    await supervisor.start()
    await planner.start()
    await coder.start()

    # 发送任务
    planner.tell(coder, "generate_code", {"desc": "Build REST API"})
    await asyncio.sleep(2)

    await planner.stop()
    await coder.stop()
    await supervisor.stop()
```

## 七、模式6: 混合架构 (Hybrid)

### 7.1 架构设计

生产级LLM应用几乎都是混合架构——在不同层次使用不同的架构模式。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    混合架构 - 生产级LLM应用                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 1: API Gateway (单体模式)                         │      │
│  │  认证、限流、路由 → 单一入口                              │      │
│  └──────────────────────┬───────────────────────────────────┘      │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 2: Orchestration (事件驱动模式)                    │      │
│  │  任务分发、Agent协调 → Kafka/NATS事件总线                  │      │
│  └──────────────────────┬───────────────────────────────────┘      │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 3: Agent Processing (Actor模式)                   │      │
│  │  每个Agent是独立Actor，拥有私有状态和行为                   │      │
│  └──────────────────────┬───────────────────────────────────┘      │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 4: Data Processing (管道模式)                      │      │
│  │  RAG检索、文档处理 → Stage管道                             │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  为什么是混合?                                                      │
│  • API层需要简单高效 → 单体                                         │
│  • Agent协调需要解耦 → 事件驱动                                     │
│  • Agent内部需要状态 → Actor模型                                    │
│  • 数据处理需要流水线 → 管道模式                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 架构选型决策矩阵

| 场景 | 推荐架构 | 理由 |
|------|---------|------|
| **快速原型验证** | 单体 | 最小开发成本 |
| **内部知识库问答** | 分层+管道 | RAG管道 + 分层清晰 |
| **客服聊天机器人** | 事件驱动 | 多渠道异步接入 |
| **代码助手(IDE)** | Actor模型 | 本地状态+实时交互 |
| **企业级AI平台** | 混合架构 | 不同层次不同需求 |
| **多Agent协作** | 事件驱动+Actor | Agent解耦+内部状态 |

### 7.3 选型决策流程图

```
开始
  │
  ├─ Q1: 需要多个独立Agent协作吗?
  │   ├─ 否 → Q2: 是否需要RAG/多步处理?
  │   │         ├─ 否 → 单体架构 ✅
  │   │         └─ 是 → 管道架构 ✅
  │   │
  │   └─ 是 → Q3: Agent间通信模式?
  │           ├─ 同步为主 → 分层架构 ✅
  │           ├─ 异步为主 → 事件驱动 ✅
  │           └─ 混合 → Q4: Agent需要复杂内部状态?
  │                     ├─ 否 → 事件驱动 ✅
  │                     └─ 是 → Actor模型 ✅
  │
  └─ Q5: 是否是生产级企业应用?
      ├─ 否 → 选择最适合的单一架构
      └─ 是 → 混合架构 ✅ (不同层次不同模式)
```

## 八、架构性能对比

### 8.1 基准测试结果（模拟数据）

| 架构模式 | 延迟P50 | 延迟P99 | 吞吐量(RPS) | GPU利用率 | 开发效率 | 运维复杂度 |
|---------|--------|--------|------------|----------|---------|-----------|
| 单体 | 120ms | 450ms | 50 | 45% | ⭐⭐⭐⭐⭐ | ⭐ |
| 分层 | 130ms | 500ms | 80 | 55% | ⭐⭐⭐⭐ | ⭐⭐ |
| 管道 | 150ms | 600ms | 120 | 65% | ⭐⭐⭐ | ⭐⭐⭐ |
| 事件驱动 | 200ms | 800ms | 500 | 80% | ⭐⭐ | ⭐⭐⭐⭐ |
| Actor | 180ms | 700ms | 400 | 75% | ⭐⭐ | ⭐⭐⭐⭐ |
| 混合 | 160ms | 650ms | 350 | 72% | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 8.2 资源消耗对比

| 架构模式 | 内存(MB) | CPU核数 | GPU | 外部依赖 |
|---------|---------|--------|-----|---------|
| 单体 | 256 | 1 | 可选 | 无 |
| 分层 | 512 | 2 | 可选 | Redis |
| 管道 | 1024 | 4 | 1 | Redis+Queue |
| 事件驱动 | 2048 | 8 | 2-4 | Kafka/NATS+Redis |
| Actor | 1536 | 4 | 1-2 | Actor框架 |
| 混合 | 4096 | 16 | 4-8 | 全套基础设施 |

## 总结

选择LLM应用架构没有银弹。核心决策原则：

1. **从简单开始**: 先用单体/分层验证产品价值
2. **按需演进**: 当遇到瓶颈时再引入更复杂的架构
3. **混合使用**: 生产系统往往需要在不同层次使用不同模式
4. **AI特殊性**: 始终考虑GPU调度、KV Cache、语义级错误等LLM特有挑战

> 💡 **下一步建议**: 根据你的团队规模和业务场景，选择1-2种架构模式进行PoC验证，再决定最终架构方案。
