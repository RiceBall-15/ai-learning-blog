---
title: "A2A（Agent-to-Agent）协议深度解析：跨平台Agent互操作的工业级实践"
description: "深入剖析Google A2A协议的设计哲学、核心架构、生产级实现方案，对比MCP协议，附完整代码示例与架构图"
date: 2026-06-01
author: "RiceBall-15"
category: "framework"
subCategory: "protocols"
tags: ["A2A协议", "Agent互操作", "MCP", "跨平台Agent", "协议设计"]
draft: false
---

## 说在前面

2025年4月，Google联合Salesforce、SAP等50余家企业发布了A2A（Agent-to-Agent）协议，与Anthropic主导的MCP协议形成了AI Agent生态的两大支柱。如果说MCP定义了Agent与工具的交互标准，那么A2A则定义了Agent与Agent之间的对话标准。

然而，很多团队在落地A2A时踩了大量坑：Agent Card发现机制不工作、跨组织认证链路断裂、长时间任务的状态同步丢失……本文将从协议设计到生产实践，给出一份完整的深度解析。

---

## 一、A2A协议的核心定位

### 1.1 为什么需要A2A？

```
┌──────────────────────────────────────────────────────────────────┐
│                    AI Agent生态的通信需求                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MCP协议解决的：Agent ↔ 工具/数据源                               │
│  ┌─────────┐    MCP     ┌──────────┐    MCP     ┌──────────┐   │
│  │ Agent A  │◄─────────►│ 数据库/   │◄─────────►│ 代码仓库  │   │
│  │ (主Agent)│           │ API/文件  │           │ /本地工具 │   │
│  └─────────┘            └──────────┘           └──────────┘   │
│                                                                  │
│  A2A协议解决的：Agent ↔ Agent                                     │
│  ┌─────────┐   A2A    ┌──────────┐   A2A    ┌──────────┐     │
│  │ Agent A  │◄───────►│ Agent B  │◄───────►│ Agent C  │     │
│  │ (调度者) │         │ (专业Agent)│         │ (领域专家)│     │
│  └─────────┘          └──────────┘         └──────────┘     │
│                                                                  │
│  核心区别：                                                       │
│  • MCP：单向工具调用，无状态，同步为主                             │
│  • A2A：双向对话协商，有状态，异步为主                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 A2A与MCP的关系

很多团队纠结A2A和MCP该选哪个，这是一个伪命题——它们是互补关系：

| 维度 | MCP | A2A |
|------|-----|-----|
| **交互对象** | Agent → 工具/数据源 | Agent → Agent |
| **通信模型** | 请求-响应（单向调用） | 对话协商（双向） |
| **状态管理** | 无状态 | 有状态（Session） |
| **任务模型** | 即时完成 | 支持长时间运行 |
| **能力发现** | Server注册 | Agent Card声明 |
| **典型场景** | 查数据库、读文件、调API | 多Agent协作、跨组织Agent互联 |

> **关键洞察**：一个完整的Agent系统通常同时使用MCP（连接工具）和A2A（连接其他Agent）。主Agent通过MCP操作本地资源，通过A2A调度远程专业Agent。

---

## 二、A2A协议核心架构

### 2.1 协议分层

```
┌─────────────────────────────────────────────────┐
│              A2A协议分层架构                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  Layer 5: 应用层                         │    │
│  │  • Agent业务逻辑                         │    │
│  │  • 多Agent编排策略                       │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                           │
│  ┌──────────────────▼──────────────────────┐    │
│  │  Layer 4: 任务管理层                     │    │
│  │  • Task生命周期（create→progress→done）  │    │
│  │  • 状态轮询与通知                        │    │
│  │  • Artifact（产出物）管理                │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                           │
│  ┌──────────────────▼──────────────────────┐    │
│  │  Layer 3: 传输层                         │    │
│  │  • JSON-RPC 2.0 over HTTP               │    │
│  │  • Server-Sent Events（SSE）             │    │
│  │  • 支持长连接流式传输                     │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                           │
│  ┌──────────────────▼──────────────────────┐    │
│  │  Layer 2: 认证与授权层                    │    │
│  │  • OAuth 2.0 / API Key                  │    │
│  │  • Agent Card上的安全策略声明            │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                           │
│  ┌──────────────────▼──────────────────────┐    │
│  │  Layer 1: 发现层                         │    │
│  │  • Agent Card（JSON元数据）              │    │
│  │  • Well-Known URI (/.well-known/agent.json)│
│  └─────────────────────────────────────────┘    │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 2.2 Agent Card — Agent的"身份证"

Agent Card是A2A协议的入口点，它声明了Agent的能力、技能和接入方式：

```json
{
  "name": "FinancialAdvisorAgent",
  "description": "专业金融分析Agent，提供投资组合建议、风险评估和市场分析",
  "url": "https://finance-agent.example.com",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "authentication": {
    "schemes": ["oauth2", "apiKey"]
  },
  "skills": [
    {
      "id": "portfolio-analysis",
      "name": "投资组合分析",
      "description": "分析投资组合的风险收益特征，提供优化建议",
      "tags": ["finance", "investment", "risk-analysis"],
      "examples": [
        "分析我的投资组合风险",
        "帮我优化资产配置"
      ]
    },
    {
      "id": "market-research",
      "name": "市场研究",
      "description": "提供个股/行业的深度研究报告",
      "tags": ["finance", "market", "research"],
      "examples": [
        "研究一下半导体行业",
        "分析特斯拉的估值"
      ]
    }
  ],
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text", "file"]
}
```

### 2.3 核心数据模型

```
Agent能力发现与任务执行流程：

  ┌─────────┐                                    ┌─────────────┐
  │ Agent A  │                                    │  Agent B    │
  │ (调度者) │                                    │ (金融分析)  │
  └────┬────┘                                    └──────┬──────┘
       │                                                │
       │  1. GET /.well-known/agent.json                 │
       │───────────────────────────────────────────────►│
       │                                                │
       │  2. 返回Agent Card                              │
       │◄───────────────────────────────────────────────│
       │                                                │
       │  3. POST /a2a (method: tasks/send)              │
       │  {                                              │
       │    "task": {                                    │
       │      "id": "task-123",                         │
       │      "message": {                              │
       │        "role": "user",                         │
       │        "parts": [{"type":"text",               │
       │          "text":"分析苹果公司股票"}]             │
       │      }                                         │
       │    }                                           │
       │  }                                              │
       │───────────────────────────────────────────────►│
       │                                                │
       │  4. 返回Task状态(submitted)                     │
       │◄───────────────────────────────────────────────│
       │                                                │
       │  5. GET /a2a (method: tasks/get) 轮询          │
       │───────────────────────────────────────────────►│
       │                                                │
       │  6. 返回Task状态(working)                       │
       │◄───────────────────────────────────────────────│
       │                                                │
       │  7. SSE推送/轮询获取最终结果                     │
       │◄───────────────────────────────────────────────│
       │  {                                              │
       │    "task": {                                    │
       │      "status": "completed",                    │
       │      "artifacts": [{                           │
       │        "parts": [{"type":"text",               │
       │          "text":"苹果公司分析报告..."}]          │
       │      }]                                        │
       │    }                                           │
       │  }                                              │
```

---

## 三、生产级实现方案

### 3.1 Python实现A2A Server

```python
from a2a.server.agent_execution import AgentExecutor, RequestContextHolder
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCard, Task, TaskState, TaskStatus,
    Message, Part, TextPart, Artifact
)
import json
import uuid
from datetime import datetime


class FinancialAgentExecutor(AgentExecutor):
    """金融分析Agent执行器"""

    async def execute(
        self,
        task: Task,
        context: RequestContextHolder,
    ) -> None:
        """核心执行逻辑"""
        user_message = task.messages[-1]
        query = self._extract_text(user_message)

        # 更新任务状态为working
        await context.update_task(
            Task(
                id=task.id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        role="agent",
                        parts=[TextPart(text="正在分析中...")]
                    )
                )
            )
        )

        # 调用LLM或专业分析工具
        analysis_result = await self._call_analysis_llm(query)

        # 生成最终Artifact
        artifact = Artifact(
            name="analysis_report",
            parts=[TextPart(text=analysis_result)],
            metadata={"format": "markdown"}
        )

        # 更新任务为完成状态
        await context.update_task(
            Task(
                id=task.id,
                status=TaskStatus(state=TaskState.completed),
                artifacts=[artifact]
            )
        )

    async def _call_analysis_llm(self, query: str) -> str:
        """调用专业分析LLM"""
        prompt = f"""你是一位专业的金融分析师。
请对以下问题进行深度分析：

{query}

要求：
1. 数据驱动，引用具体指标
2. 风险提示充分
3. 结论明确，给出投资建议"""
        # 实际生产中接入OpenAI/Anthropic等
        return f"## {query} 深度分析报告\n\n[分析内容...]"


# 构建A2A应用
def create_a2a_app():
    agent_card = AgentCard(
        name="FinancialAdvisorAgent",
        description="专业金融分析Agent",
        url="https://finance-agent.example.com",
        version="1.0.0",
        capabilities={"streaming": True, "pushNotifications": False},
        authentication={"schemes": ["apiKey"]},
        skills=[{
            "id": "portfolio-analysis",
            "name": "投资组合分析",
            "description": "分析投资组合的风险收益特征",
            "tags": ["finance"],
        }],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    executor = FinancialAgentExecutor()
    handler = DefaultRequestHandler(agent_executor=executor)

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )


if __name__ == "__main__":
    app = create_a2a_app()
    app.run(host="0.0.0.0", port=8000)
```

### 3.2 A2A Client — 调度远程Agent

```python
import httpx
import asyncio
from a2a.client import A2AClient
from a2a.types import (
    SendMessageRequest, TaskQueryParams,
    TaskSendParams, Message, Part, TextPart
)


async def discover_and_call_agent(agent_url: str, query: str):
    """发现并调用远程Agent"""
    # 1. 获取Agent Card
    async with httpx.AsyncClient() as client:
        card_response = await client.get(
            f"{agent_url}/.well-known/agent.json"
        )
        agent_card = card_response.json()
        print(f"发现Agent: {agent_card['name']}")
        print(f"可用技能: {[s['name'] for s in agent_card['skills']]}")

    # 2. 创建A2A客户端
    a2a_client = await A2AClient.get_client_from_agent_card_url(
        httpx_client=httpx.AsyncClient(),
        agent_card_url=f"{agent_url}/.well-known/agent.json",
    )

    # 3. 发送任务
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    send_params = TaskSendParams(
        id=task_id,
        message=Message(
            role="user",
            parts=[TextPart(text=query)]
        ),
        metadata={"priority": "high"}
    )

    result = await a2a_client.send_task(send_params)
    print(f"任务状态: {result.task.status.state}")

    # 4. 轮询等待结果（生产环境建议用SSE）
    while result.task.status.state in ("submitted", "working"):
        await asyncio.sleep(2)
        result = await a2a_client.get_task(
            TaskQueryParams(id=task_id)
        )
        print(f"当前状态: {result.task.status.state}")

    # 5. 提取结果
    if result.task.artifacts:
        for artifact in result.task.artifacts:
            for part in artifact.parts:
                if part.type == "text":
                    print(f"\n分析结果:\n{part.text}")

    return result


# 使用示例
asyncio.run(discover_and_call_agent(
    agent_url="https://finance-agent.example.com",
    query="分析2026年半导体行业投资机会"
))
```

### 3.3 多Agent编排模式

```
┌─────────────────────────────────────────────────────────────────┐
│                  多Agent编排架构模式                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  模式1：集中调度（Orchestrator Pattern）                          │
│                                                                  │
│  ┌──────────────┐                                                │
│  │ Orchestrator │───A2A───►┌──────────┐                        │
│  │   Agent      │          │ Agent B  │                        │
│  │              │───A2A───►│ (翻译)   │                        │
│  │  (调度者)    │          └──────────┘                        │
│  │              │───A2A───►┌──────────┐                        │
│  └──────────────┘          │ Agent C  │                        │
│                            │ (财务)   │                        │
│                            └──────────┘                        │
│  优点：简单可控  缺点：调度者是瓶颈                                │
│                                                                  │
│  模式2：链式传递（Pipeline Pattern）                              │
│                                                                  │
│  ┌──────┐  A2A  ┌──────┐  A2A  ┌──────┐  A2A  ┌──────┐     │
│  │输入   │─────►│清洗   │─────►│分析   │─────►│报告   │     │
│  │Agent │      │Agent │      │Agent │      │Agent │     │
│  └──────┘      └──────┘      └──────┘      └──────┘     │
│  优点：各司其职  缺点：延迟累积                                     │
│                                                                  │
│  模式3：对等协商（Peer-to-Peer Pattern）                          │
│                                                                  │
│  ┌──────┐ A2A ┌──────┐                                            │
│  │Agent │◄───►│Agent │                                            │
│  │  A   │     │  B   │                                            │
│  └──┬───┘     └──┬───┘                                            │
│     │  A2A       │  A2A                                           │
│     │            │                                                │
│  ┌──▼───┐     ┌──▼───┐                                            │
│  │Agent │◄───►│Agent │                                            │
│  │  C   │     │  D   │                                            │
│  └──────┘     └──────┘                                            │
│  优点：灵活可扩展  缺点：协调复杂                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、实战踩坑与最佳实践

### 4.1 常见问题与解决方案

| 问题 | 现象 | 根因 | 解决方案 |
|------|------|------|----------|
| Agent Card发现失败 | 404/超时 | Well-Known URI配置错误 | 检查`/.well-known/agent.json`路径；确保HTTPS证书有效 |
| 任务状态丢失 | 轮询返回404 | Agent重启后内存状态丢失 | 持久化Task状态到Redis/DB |
| 跨域请求被拒 | CORS错误 | 浏览器安全策略 | 服务端配置`Access-Control-Allow-Origin` |
| 长任务超时 | HTTP 504 | 任务执行时间超过网关超时 | 使用SSE流式返回 + 任务轮询解耦 |
| 认证链路断裂 | 401/403 | OAuth Token过期未刷新 | 实现Token自动刷新中间件 |

### 4.2 状态管理最佳实践

```python
# Task状态持久化（使用Redis）
import redis.asyncio as redis

class TaskStateManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def save_task(self, task_id: str, task: dict):
        """保存Task状态"""
        key = f"a2a:task:{task_id}"
        await self.redis.setex(
            key,
            3600 * 24,  # 24小时过期
            json.dumps(task, ensure_ascii=False)
        )

    async def get_task(self, task_id: str) -> dict | None:
        """获取Task状态"""
        key = f"a2a:task:{task_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def transition_task(
        self, task_id: str, new_state: str, message: str = ""
    ):
        """Task状态流转（带乐观锁）"""
        key = f"a2a:task:{task_id}"
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # 校验状态流转合法性
        valid_transitions = {
            "submitted": ["working", "failed"],
            "working": ["completed", "failed", "canceled"],
        }
        current = task["status"]["state"]
        if new_state not in valid_transitions.get(current, []):
            raise ValueError(
                f"Invalid transition: {current} → {new_state}"
            )

        task["status"]["state"] = new_state
        task["status"]["message"] = message
        task["updatedAt"] = datetime.utcnow().isoformat()
        await self.save_task(task_id, task)
```

### 4.3 安全加固策略

```yaml
# A2A安全配置清单

authentication:
  # 1. 强制HTTPS
  transport: "https"
  # 2. Agent间认证
  schemes:
    - oauth2        # 推荐：企业级场景
    - apiKey        # 可选：简单场景

authorization:
  # 3. 技能级权限控制
  skills:
    - id: "portfolio-analysis"
      required_scopes: ["finance:read"]
    - id: "trade-execution"
      required_scopes: ["finance:trade"]  # 高危操作需额外授权

rate_limiting:
  # 4. 请求限流
  global: "100/min"
  per_skill:
    portfolio-analysis: "20/min"
    trade-execution: "5/min"

audit:
  # 5. 审计日志
  log_all_requests: true
  retention_days: 90
  alert_on_anomaly: true
```

---

## 五、A2A与MCP的选型决策树

```
你的Agent需要什么？
│
├── 需要操作本地工具/数据库/文件？
│   └── 用 MCP ✅
│
├── 需要调用另一个Agent的能力？
│   └── 用 A2A ✅
│
├── 需要长时间运行的任务（分钟/小时级）？
│   └── 用 A2A ✅（原生支持异步任务）
│
├── 需要跨组织/跨平台的Agent互联？
│   └── 用 A2A ✅（Agent Card发现机制）
│
└── 需要同时连接工具和其他Agent？
    └── MCP + A2A 组合使用 ✅
        主Agent ←A2A→ 专业Agent（各自通过MCP连接本地工具）
```

---

## 六、未来展望

### 6.1 A2A协议的演进方向

1. **Agent经济网络**：基于A2A协议构建Agent市场，Agent可以"雇佣"其他Agent完成子任务，形成Agent间的微服务经济
2. **联邦Agent**：跨组织的Agent协作，数据不出域，通过A2A协议交换推理结果而非原始数据
3. **Agent治理框架**：Agent的行为审计、责任追溯、异常熔断等治理能力标准化

### 6.2 A2A与MCP的融合趋势

两大协议正在走向融合：
- MCP正在增加对异步任务的支持
- A2A在Agent Card中引入了MCP Server描述
- 未来可能出现统一的Agent通信协议层

---

## 七、总结

| 问题 | 答案 |
|------|------|
| A2A是什么？ | Agent与Agent之间的通信协议标准 |
| 和MCP的关系？ | 互补——MCP管工具调用，A2A管Agent协作 |
| 什么场景用A2A？ | 多Agent编排、跨组织Agent互联、长时任务 |
| 生产落地的核心挑战？ | 状态管理、安全认证、服务发现可靠性 |
| 下一步？ | 关注A2A协议的版本演进和MCP的融合趋势 |

> **最后的话**：A2A协议为Agent生态提供了互操作的基础，但它不是银弹。在实际落地中，协议本身只解决了"怎么通信"的问题，"怎么协作"仍需要架构师在Agent编排、状态管理、安全策略等方面进行深度设计。建议从一个具体的多Agent场景开始试点，逐步扩展到完整的Agent网络。

---

**参考文献**：
1. Google A2A Protocol Specification - https://github.com/google/A2A
2. Anthropic MCP Specification - https://modelcontextprotocol.io
3. Agent Interoperability Patterns - Agent Protocol Working Group
