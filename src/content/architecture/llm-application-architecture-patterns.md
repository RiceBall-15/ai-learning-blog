---
title: "LLM应用架构设计模式：从单体推理到智能体集群"
description: "深入剖析LLM应用的五种核心架构模式，结合真实场景分析如何构建高可用、可扩展的AI系统"
date: 2025-06-01
author: "RiceBall"
category: "architecture"
subCategory: distributed
tags: ["系统架构", "LLM应用", "Agent架构", "微服务", "高可用"]
draft: false
---

## 引言：架构设计决定了LLM应用的上限

很多团队在构建LLM应用时，第一步就写了一个API调用，接了一个Prompt，然后就宣称"AI应用上线了"。当用户量增长、需求变复杂时，才发现整个系统像沙子一样脆弱——改一个Prompt可能导致全局异常，增加一个功能需要重构整个调用链。

LLM应用不是简单的API调用。它是一个**分布式系统**，需要像对待传统后端系统一样认真设计架构。本文将系统性地梳理LLM应用的五种核心架构模式，每种模式都有其适用场景和工程代价。

---

## 一、架构演进路线

```
Level 1: 单体调用         Level 2: 流水线          Level 3: Agent架构
┌─────────┐             ┌──→ Filter ──┐          ┌─── Planner ───┐
│  Prompt  │             │  ┌─────────┐ │          │               │
│  + LLM   │  ──────→   │  │   LLM   │ │  ─────→  │  ┌─ Tool ─┐  │
│  + 简单逻辑│             │  └─────────┘ │          │  │ Router │  │
└─────────┘             └──→ PostProc ──┘          │  └────────┘  │
                                                    └───────────────┘

Level 4: 微服务化Agent      Level 5: 智能体集群
┌──────────────┐          ┌─────────────────────────┐
│ Agent Service │          │    Orchestrator         │
│ ┌────┐┌────┐│          │  ┌────┐  ┌────┐  ┌────┐│
│ │Pln ││Tool││          │  │Agt1│  │Agt2│  │Agt3││
│ └────┘└────┘│          │  └──┬─┘  └──┬─┘  └──┬─┘│
│ ┌────┐┌────┐│          │     └───┬───┘───┬───┘  │
│ │Mem ││Eval││          │    ┌────┴───┐  ┌─┴────┐│
│ └────┘└────┘│          │    │Shared  │  │Shared││
└──────────────┘          │    │Memory  │  │Tools ││
                          │    └────────┘  └──────┘│
                          └─────────────────────────┘
```

---

## 二、模式一：Prompt-as-Config（提示词即配置）

### 2.1 核心思想

将Prompt从代码中解耦，作为配置管理。这是最简单但最容易被忽视的架构模式。

### 2.2 工程实现

```yaml
# prompts/summarizer.yaml
name: "summarizer"
version: "2.3"
model: "qwen2.5-72b-instruct"
temperature: 0.3
max_tokens: 2048

system_prompt: |
  你是一个专业的文本摘要助手。请按照以下规则进行摘要：
  1. 保留原文的核心观点和关键数据
  2. 使用与原文相同的语言
  3. 摘要长度控制在原文的20%-30%
  4. 不要添加原文中没有的信息

# 按任务类型配置不同参数
variants:
  news:
    temperature: 0.3
    system_prompt_override: |
      作为新闻摘要，请特别注意保留时间、地点、人物等要素
  
  technical:
    temperature: 0.1
    max_tokens: 4096
    system_prompt_override: |
      作为技术文档摘要，请保留所有技术术语和关键参数
```

```python
# 模板引擎
from jinja2 import Environment, FileSystemLoader
import yaml

class PromptManager:
    def __init__(self, prompts_dir="prompts"):
        self.env = Environment(loader=FileSystemLoader(prompts_dir))
        self.prompts = {}
    
    def load(self, prompt_name: str, variant: str = None):
        """加载并渲染Prompt模板"""
        key = f"{prompt_name}:{variant or 'default'}"
        if key not in self.prompts:
            with open(f"prompts/{prompt_name}.yaml") as f:
                config = yaml.safe_load(f)
            
            if variant and variant in config.get("variants", {}):
                config.update(config["variants"][variant])
            
            self.prompts[key] = config
        return self.prompts[key]
    
    def render(self, prompt_name: str, **kwargs):
        config = self.load(prompt_name, kwargs.get("variant"))
        template = self.env.get_string(f"{prompt_name}/template.j2")
        return {
            "system": config["system_prompt"],
            "user": template.render(**kwargs),
            "model": config["model"],
            "params": {
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
            }
        }
```

### 2.3 适用场景与局限

| 适用 | 不适用 |
|------|--------|
| 单一功能的LLM应用 | 复杂的多步骤推理 |
| 快速原型验证 | 需要动态决策的场景 |
| 内容生成类任务 | 需要工具调用的Agent |

---

## 三、模式二：LLM Pipeline（流水线模式）

### 3.1 核心思想

将复杂的LLM任务拆分为多个顺序执行的步骤，每个步骤有独立的Prompt和处理逻辑。类似传统的ETL流水线。

### 3.2 架构设计

```
输入 → [路由] → [预处理] → [LLM步骤1] → [后处理] → [LLM步骤2] → [输出]
         │                                                      │
         └── [回退路径] ←────────────── [质检] ←────────────────┘
```

### 3.3 工程实现

```python
from dataclasses import dataclass
from typing import Callable, Any
import asyncio

@dataclass
class PipelineStep:
    name: str
    handler: Callable
    retry: int = 3
    fallback: Callable = None

class LLMPipeline:
    def __init__(self):
        self.steps: list[PipelineStep] = []
        self.hooks = {"before": [], "after": [], "error": []}
    
    def add_step(self, step: PipelineStep):
        self.steps.append(step)
        return self
    
    def before(self, hook: Callable):
        self.hooks["before"].append(hook)
        return self
    
    def after(self, hook: Callable):
        self.hooks["after"].append(hook)
        return self
    
    async def execute(self, input_data: dict) -> dict:
        context = {"input": input_data, "metadata": {}}
        
        for hook in self.hooks["before"]:
            context = await hook(context)
        
        for step in self.steps:
            for attempt in range(step.retry):
                try:
                    result = await step.handler(context)
                    context["output"] = result
                    context["metadata"][step.name] = {"status": "success"}
                    break
                except Exception as e:
                    if attempt == step.retry - 1 and step.fallback:
                        context["output"] = await step.fallback(context)
                    elif attempt == step.retry - 1:
                        raise
                    context["metadata"][step.name] = {
                        "status": "retry", "attempt": attempt + 1
                    }
        
        for hook in self.hooks["after"]:
            context = await hook(context)
        
        return context

# 实战示例：智能文档分析流水线
pipeline = LLMPipeline()

pipeline.add_step(PipelineStep(
    name="classify",
    handler=classify_document,    # 文档分类
    retry=2,
))

pipeline.add_step(PipelineStep(
    name="extract",
    handler=extract_entities,     # 实体抽取
    retry=3,
    fallback=lambda ctx: {"entities": []},  # 兜底方案
))

pipeline.add_step(PipelineStep(
    name="summarize",
    handler=generate_summary,     # 摘要生成
    retry=2,
))

pipeline.add_step(PipelineStep(
    name="quality_check",
    handler=validate_output,      # 输出质检
    retry=1,
))
```

### 3.4 关键设计决策

**错误处理策略**是Pipeline模式的核心难点。LLM调用与传统API不同——它可能返回"看似正确但实际错误"的结果，而不是抛出异常。

```python
# 质检Hook示例
async def quality_check_hook(context):
    output = context.get("output", "")
    metadata = context.get("metadata", {})
    
    # 检查1：输出是否为空
    if not output or len(output.strip()) < 10:
        raise ValueError("输出过短，可能生成失败")
    
    # 检查2：是否包含拒绝回答的模式
    refusal_patterns = [
        "我无法", "我不能", "抱歉，我无法", "As an AI"
    ]
    if any(p in output for p in refusal_patterns):
        context["metadata"]["quality"] = "refused"
        # 可以触发回退或重试
    
    # 检查3：输出长度是否合理
    if len(output) > context["input"].get("max_expected", 5000):
        context["metadata"]["quality"] = "too_long"
    
    return context
```

---

## 四、模式三：Agent架构（自主决策模式）

### 4.1 核心思想

Agent不再按预设流水线执行，而是根据目标**自主决定**下一步做什么。这是当前AI应用中最复杂的架构模式。

### 4.2 核心组件

```
┌─────────────────────────────────────────────────┐
│                  Agent Runtime                   │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Planning  │◄──►│Execution │◄──►│  Memory   │   │
│  │  Module   │    │  Module  │    │  Module   │   │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘   │
│       │               │               │          │
│       ▼               ▼               ▼          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │   LLM    │    │  Tool    │    │  State   │   │
│  │  Router  │    │ Registry │    │   Store   │   │
│  └──────────┘    └──────────┘    └──────────┘   │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 4.3 工程实现：ReAct Agent

```python
from typing import Any
import json

class ReActAgent:
    """ReAct (Reasoning + Acting) Agent"""
    
    def __init__(self, llm_client, tools: dict, max_steps: int = 10):
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps
        self.state = {
            "thoughts": [],
            "actions": [],
            "observations": [],
        }
    
    def build_system_prompt(self) -> str:
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""你是一个智能助手，可以通过以下工具完成任务：

{tool_descriptions}

请使用以下格式进行推理：

Thought: [你的思考过程]
Action: [工具名称]
Action Input: [输入参数，JSON格式]

当你准备好最终答案时：
Thought: [最终总结]
Final Answer: [最终答案]"""
    
    async def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": task},
        ]
        
        for step in range(self.max_steps):
            # 1. LLM推理
            response = await self.llm.chat(messages)
            content = response["choices"][0]["message"]["content"]
            
            # 2. 解析Action
            action = self._parse_action(content)
            
            if action is None:
                # Final Answer
                final_answer = self._parse_final_answer(content)
                return final_answer
            
            # 3. 执行工具
            tool_name, tool_input = action
            observation = await self._execute_tool(tool_name, tool_input)
            
            # 4. 更新状态
            self.state["thoughts"].append(content)
            self.state["actions"].append(f"{tool_name}({tool_input})")
            self.state["observations"].append(observation)
            
            # 5. 将观察结果加入对话
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\n请继续推理。"
            })
        
        return "达到最大步骤数限制，未能完成任务"
    
    def _parse_action(self, content: str):
        """解析LLM输出中的Action"""
        if "Final Answer:" in content:
            return None
        
        if "Action:" in content:
            lines = content.split("\n")
            action = None
            action_input = None
            
            for line in lines:
                if line.strip().startswith("Action:"):
                    action = line.split(":", 1)[1].strip()
                elif line.strip().startswith("Action Input:"):
                    raw = line.split(":", 1)[1].strip()
                    action_input = json.loads(raw)
            
            return (action, action_input)
        
        return None
    
    async def _execute_tool(self, name: str, input_data: Any) -> str:
        """执行工具并返回观察结果"""
        if name not in self.tools:
            return f"错误：工具 {name} 不存在"
        
        try:
            tool = self.tools[name]
            result = await tool.execute(input_data)
            return str(result)
        except Exception as e:
            return f"工具执行错误：{str(e)}"
```

### 4.4 Agent架构的关键挑战

#### 挑战1：无限循环

Agent可能陷入"思考-行动-观察-再思考"的死循环：

```python
# 安全措施：循环检测
class LoopDetector:
    def __init__(self, window_size: int = 5):
        self.history = []
        self.window_size = window_size
    
    def check(self, action: str) -> bool:
        """返回True表示检测到循环"""
        self.history.append(action)
        
        if len(self.history) > self.window_size:
            recent = self.history[-self.window_size:]
            # 检查是否有重复的Action序列
            unique_actions = len(set(recent))
            if unique_actions <= 2:
                return True
        
        return False
```

#### 挑战2：成本控制

一个失控的Agent可能在几秒钟内消耗大量Token：

```python
# 成本控制中间件
class CostController:
    def __init__(self, max_cost_per_run: float = 1.0):  # 最大$1
        self.max_cost = max_cost_per_run
        self.current_cost = 0.0
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int,
                      model: str) -> float:
        # 简化的成本估算
        pricing = {
            "gpt-4": (0.03/1000, 0.06/1000),
            "gpt-4o": (0.005/1000, 0.015/1000),
            "qwen-72b": (0.002/1000, 0.006/1000),
        }
        input_price, output_price = pricing.get(model, (0.002/1000, 0.006/1000))
        return prompt_tokens * input_price + completion_tokens * output_price
    
    def check_budget(self) -> bool:
        return self.current_cost < self.max_cost
```

#### 挑战3：可观测性

Agent的行为是非确定性的，需要完整的Trace：

```python
import time
import uuid
from contextlib import asynccontextmanager

class AgentTracer:
    def __init__(self):
        self.traces = []
    
    @asynccontextmanager
    async def trace(self, operation: str, metadata: dict = None):
        trace_id = str(uuid.uuid4())[:8]
        start = time.time()
        
        span = {
            "trace_id": trace_id,
            "operation": operation,
            "start": start,
            "metadata": metadata or {},
            "status": "running",
        }
        
        try:
            yield span
            span["status"] = "success"
        except Exception as e:
            span["status"] = "error"
            span["error"] = str(e)
            raise
        finally:
            span["duration_ms"] = (time.time() - start) * 1000
            self.traces.append(span)
            self._log(span)
    
    def _log(self, span):
        print(f"[{span['trace_id']}] {span['operation']} "
              f"{'✓' if span['status'] == 'success' else '✗'} "
              f"{span['duration_ms']:.0f}ms")
```

---

## 五、模式四：微服务化Agent（服务拆分模式）

### 5.1 核心思想

将Agent拆分为独立的微服务，每个服务负责一个特定能力。这是大规模Agent系统的必然选择。

### 5.2 服务拆分策略

```
┌─────────────────────────────────────────────────┐
│                 API Gateway                      │
│              (路由 + 限流 + 鉴权)                 │
└──────────┬──────────┬──────────┬────────────────┘
           │          │          │
    ┌──────▼───┐ ┌───▼──────┐ ┌▼──────────┐
    │ Planner  │ │ Executor │ │  Memory   │
    │ Service  │ │ Service  │ │  Service  │
    │          │ │          │ │           │
    │ ·任务规划 │ │ ·工具执行 │ │ ·向量存储  │
    │ ·目标分解 │ │ ·结果解析 │ │ ·对话历史  │
    │ ·优先排序 │ │ ·重试逻辑 │ │ ·知识检索  │
    └──────────┘ └──────────┘ └───────────┘
           │          │          │
    ┌──────▼──────────▼──────────▼──────────┐
    │           Message Queue               │
    │         (Kafka / Redis Stream)        │
    └───────────────────────────────────────┘
```

### 5.3 服务间通信设计

```python
# Agent Service 间的消息协议
from pydantic import BaseModel
from enum import Enum

class MessageType(str, Enum):
    TASK_ASSIGN = "task.assign"
    TASK_RESULT = "task.result"
    TOOL_REQUEST = "tool.request"
    TOOL_RESPONSE = "tool.response"
    MEMORY_QUERY = "memory.query"
    MEMORY_RESPONSE = "memory.response"

class AgentMessage(BaseModel):
    message_id: str
    message_type: MessageType
    source_service: str
    target_service: str
    payload: dict
    metadata: dict = {}
    timestamp: float
    ttl: int = 30  # 消息存活时间(秒)

# 路由服务
class AgentRouter:
    def __init__(self):
        self.service_registry = {}  # 服务注册表
        self.message_queue = None   # 消息队列
    
    async def route(self, message: AgentMessage):
        """根据消息类型路由到目标服务"""
        routing_table = {
            MessageType.TASK_ASSIGN: "planner-service",
            MessageType.TOOL_REQUEST: "executor-service",
            MessageType.MEMORY_QUERY: "memory-service",
        }
        
        target = routing_table.get(message.message_type)
        if not target:
            raise ValueError(f"Unknown message type: {message.message_type}")
        
        # 检查目标服务健康状态
        if not self._is_healthy(target):
            # 降级到备用服务
            target = self._get_fallback(target)
        
        await self.message_queue.publish(target, message)
```

### 5.4 数据一致性保证

微服务架构下的数据一致性是最大挑战。Agent的"记忆"可能分布在多个服务中：

```python
# 最终一致性方案：Event Sourcing
class MemoryEventStore:
    """基于事件溯源的记忆管理"""
    
    def __init__(self, redis_client, vector_db):
        self.redis = redis_client
        self.vector_db = vector_db
    
    async def append_event(self, agent_id: str, event: dict):
        """追加记忆事件"""
        event["timestamp"] = time.time()
        event["agent_id"] = agent_id
        
        # 1. 写入事件流（持久化）
        await self.redis.xadd(
            f"agent:{agent_id}:events",
            event
        )
        
        # 2. 异步更新向量索引（最终一致）
        await self._update_vector_index(agent_id, event)
    
    async def query_memory(self, agent_id: str, query: str, top_k: int = 5):
        """查询相关记忆"""
        # 向量检索 + 时间衰减
        results = await self.vector_db.search(
            collection=f"agent_{agent_id}_memory",
            query=query,
            top_k=top_k,
            filter={"agent_id": agent_id}
        )
        
        # 时间衰减：近期记忆权重更高
        decay_factor = 0.95  # 每天衰减5%
        for result in results:
            age_days = (time.time() - result["timestamp"]) / 86400
            result["adjusted_score"] = result["score"] * (decay_factor ** age_days)
        
        return sorted(results, key=lambda x: x["adjusted_score"], reverse=True)
```

---

## 六、模式五：智能体集群（多Agent协作）

### 6.1 核心思想

多个专业化Agent协作完成复杂任务，类似一个团队的分工协作。这是最复杂但也最强大的架构模式。

### 6.2 协作架构

```
┌─────────────────────────────────────────────────────┐
│                  Orchestrator                        │
│              (任务分配 + 结果聚合)                    │
├──────┬──────┬──────┬──────┬──────┬──────────────────┤
│      │      │      │      │      │                  │
▼      ▼      ▼      ▼      ▼      ▼                  │
┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐                 │
│研究 ││编码 ││测试 ││文档 ││审查 ││部署 │ ← 专业化Agent │
│Agent││Agent││Agent││Agent││Agent││Agent│                 │
└──┬─┘└──┬─┘└──┬─┘└──┬─┘└──┬─┘└──┬─┘                 │
   │     │     │     │     │     │                     │
   └──┬──┘──┬──┘──┬──┘──┬──┘──┬──┘                     │
      │     │     │     │     │                         │
   ┌──▼─────▼─────▼─────▼─────▼──┐                    │
   │    Shared Knowledge Base     │ ← 共享知识库        │
   │    (向量库 + 图数据库)        │                    │
   └──────────────────────────────┘                    │
```

### 6.3 多Agent协作模式

#### 模式A：管道式协作（Sequential）

```python
class SequentialPipeline:
    """研究 → 编码 → 测试 → 文档"""
    
    async def execute(self, task: str) -> dict:
        # 1. 研究Agent：分析需求
        research = await self.agents["researcher"].run(
            f"分析以下需求并给出技术方案：{task}"
        )
        
        # 2. 编码Agent：实现方案
        code = await self.agents["coder"].run(
            f"根据以下技术方案实现代码：\n{research}"
        )
        
        # 3. 测试Agent：验证代码
        test_result = await self.agents["tester"].run(
            f"测试以下代码并报告问题：\n{code}"
        )
        
        # 4. 文档Agent：生成文档
        docs = await self.agents["docwriter"].run(
            f"为以下代码生成文档：\n{code}\n\n测试结果：{test_result}"
        )
        
        return {"code": code, "tests": test_result, "docs": docs}
```

#### 模式B：辩论式协作（Debate）

```python
class DebateOrchestrator:
    """多个Agent对同一问题进行辩论，达成共识"""
    
    async def debate(self, topic: str, rounds: int = 3) -> str:
        agents = list(self.agents.values())
        positions = {}
        
        for round_num in range(rounds):
            round_arguments = []
            
            for agent in agents:
                # 每个Agent给出论点
                context = f"话题：{topic}\n\n之前的讨论：\n"
                if round_arguments:
                    context += "\n".join(round_arguments[-3:])
                
                argument = await agent.run(
                    f"请就以下话题发表你的观点（第{round_num+1}轮）：\n{context}"
                )
                round_arguments.append(f"{agent.name}: {argument}")
            
            # 最后一轮：综合所有观点
            if round_num == rounds - 1:
                consensus = await agents[0].run(
                    f"基于以下讨论，给出最终结论：\n" +
                    "\n".join(round_arguments)
                )
                return consensus
        
        return round_arguments[-1] if round_arguments else ""
```

#### 模式C：投票式协作（Voting）

```python
class VotingOrchestrator:
    """多Agent独立执行，投票决定最终结果"""
    
    async def execute_with_voting(self, task: str, 
                                   num_agents: int = 3) -> str:
        # 多个Agent独立执行
        results = await asyncio.gather(*[
            self.agents[f"agent_{i}"].run(task)
            for i in range(num_agents)
        ])
        
        # 投票机制
        votes = {}
        for result in results:
            # 简化：基于结果相似度投票
            normalized = self._normalize(result)
            votes[normalized] = votes.get(normalized, 0) + 1
        
        # 选择票数最多的结果
        winner = max(votes.items(), key=lambda x: x[1])
        
        # 如果票数不够集中，触发仲裁Agent
        if winner[1] < num_agents * 0.6:
            return await self.agents["arbiter"].run(
                f"多个Agent给出了不同结果，请仲裁：\n" +
                "\n".join([f"Agent {i}: {r}" for i, r in enumerate(results)])
            )
        
        return winner[0]
```

### 6.4 多Agent系统的关键工程问题

#### 问题1：通信协议

```python
# 基于JSON-RPC的Agent间通信
class AgentProtocol:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.handlers = {}
    
    async def call(self, target_agent: str, method: str, 
                   params: dict) -> dict:
        """同步调用其他Agent"""
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
            "from": self.agent_id,
        }
        
        response = await self.transport.send(target_agent, request)
        return response.get("result")
    
    def expose(self, method_name: str, handler: Callable):
        """暴露Agent能力给其他Agent调用"""
        self.handlers[method_name] = handler
```

#### 问题2：冲突解决

```python
class ConflictResolver:
    """Agent间冲突解决"""
    
    strategies = {
        "priority": "_resolve_by_priority",
        "vote": "_resolve_by_vote",
        "merge": "_resolve_by_merge",
        "arbiter": "_resolve_by_arbiter",
    }
    
    async def resolve(self, conflicts: list, strategy: str = "priority"):
        resolver = getattr(self, self.strategies[strategy])
        return await resolver(conflicts)
    
    async def _resolve_by_priority(self, conflicts):
        """按Agent优先级决定"""
        # 系统Agent > 用户Agent > 辅助Agent
        priority_order = ["system", "planner", "executor", "assistant"]
        sorted_conflicts = sorted(
            conflicts,
            key=lambda c: priority_order.index(c.agent_type)
        )
        return sorted_conflicts[0]
    
    async def _resolve_by_merge(self, conflicts):
        """合并多个Agent的建议"""
        merged = {}
        for conflict in conflicts:
            for key, value in conflict.proposal.items():
                if key not in merged:
                    merged[key] = []
                merged[key].append(value)
        
        # 对每个字段，选择置信度最高的建议
        for key in merged:
            merged[key] = max(merged[key], key=lambda x: x.get("confidence", 0))
        
        return merged
```

---

## 七、架构选型指南

### 7.1 决策矩阵

| 场景 | 推荐模式 | 复杂度 | 延迟 | 成本 | 可维护性 |
|------|----------|--------|------|------|----------|
| 简单内容生成 | Prompt-as-Config | ⭐ | 低 | 低 | ⭐⭐⭐⭐⭐ |
| 文档处理流水线 | LLM Pipeline | ⭐⭐ | 中 | 中 | ⭐⭐⭐⭐ |
| 客服/问答系统 | Agent架构 | ⭐⭐⭐ | 中 | 中 | ⭐⭐⭐ |
| 复杂任务自动化 | 微服务化Agent | ⭐⭐⭐⭐ | 高 | 高 | ⭐⭐ |
| 研究/分析系统 | 智能体集群 | ⭐⭐⭐⭐⭐ | 高 | 很高 | ⭐ |

### 7.2 常见陷阱

```
❌ 陷阱1：过度设计
   一个简单的摘要功能，非要搞成多Agent协作
   → 先用最简单的方案，复杂度是逐步引入的

❌ 陷阱2：忽视可观测性
   Agent做了10步才完成任务，你却不知道中间发生了什么
   → 从第一天就建立完整的Trace系统

❌ 陷阱3：没有Fallback
   Agent调用工具失败，整个任务就挂了
   → 每个环节都要有降级方案

❌ 陷阱4：忽视成本控制
   Agent陷入循环，一个请求烧掉$50
   → 设置Token上限、成本上限、步骤上限

❌ 陷阱5：Prompt和代码强耦合
   改一个Prompt要重新部署整个服务
   → Prompt必须外部化管理
```

---

## 八、实战案例：一个RAG+Agent系统的架构演进

### 8.1 阶段一：最简方案

```
用户提问 → 相关文档检索(RAG) → LLM生成回答
```

问题：当文档更新时，缓存失效不及时；无法处理"需要多步推理"的问题。

### 8.2 阶段二：引入Pipeline

```
用户提问 → 意图识别 → 路由 → [RAG路径 | 推理路径] → 后处理 → 回答
```

解决了路由问题，但每增加一个新能力就要改Pipeline。

### 8.3 阶段三：Agent架构

```
用户提问 → Agent(自主决定用什么工具) → 工具执行 → 回答
```

灵活性大幅提升，但调试和成本控制变难。

### 8.4 最终方案

```
┌────────────────────────────────────────────────┐
│                 Orchestrator                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Intent  │  │ Router  │  │  Cost   │        │
│  │Detector │  │         │  │Controller│        │
│  └────┬────┘  └────┬────┘  └─────────┘        │
│       └──────┬─────┘                           │
│              │                                  │
│  ┌───────────▼────────────┐                    │
│  │    Agent Runtime       │                    │
│  │  ┌──────┐  ┌────────┐ │                    │
│  │  │RAG   │  │Tool    │ │                    │
│  │  │Tool  │  │Registry│ │                    │
│  │  └──────┘  └────────┘ │                    │
│  └────────────────────────┘                    │
│                                                 │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Semantic│  │ Conversation│ │ Prompt  │      │
│  │ Cache   │  │ Store     │  │ Config  │      │
│  └─────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────┘
```

---

## 总结

LLM应用架构设计没有银弹，但有明确的演进路径：

1. **从简单开始**：不要过度设计，先用最简方案验证价值
2. **逐步引入复杂度**：遇到瓶颈时再升级架构模式
3. **可观测性第一**：无论哪种模式，完整的Trace和监控是必须的
4. **成本控制是红线**：每种架构模式都需要考虑成本边界
5. **Prompt与代码分离**：这是所有模式的共同前提

> **好的架构不是一开始就设计出来的，而是在持续迭代中演进出来的。**

选择适合你当前阶段的架构模式，在需要时平滑升级。记住：架构是手段，不是目的。
