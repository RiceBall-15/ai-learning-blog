---
title: "A2A协议深度解析：Google提出的Agent间通信标准——从单体Agent到多Agent协作"
description: "深入剖析A2A（Agent-to-Agent）协议的核心架构、通信机制、Agent Card设计与生产级实现，构建真正的多Agent协作系统"
date: 2026-05-30
author: "RiceBall-15"
category: "framework"
tags: ["A2A", "Agent", "多Agent协作", "协议", "框架应用"]
draft: false
---

# A2A协议深度解析：Google提出的Agent间通信标准——从单体Agent到多Agent协作

## 一、引言：多Agent协作的通信困境

### 1.1 从单Agent到多Agent的必然演进

2025年，AI Agent领域出现了一个显著趋势：从"超级单体Agent"向"专业多Agent协作"的范式转移。原因很直接：

- **单体Agent的局限**：一个Agent试图掌握所有技能，导致Prompt膨胀、上下文窗口耗尽、专业性不足
- **专业分工的优势**：每个Agent专注一个领域，质量更高、维护更简单、可独立迭代

但多Agent协作面临一个核心问题：**Agent之间如何通信？**

### 1.2 现有方案的碎片化

目前多Agent通信的实现方式五花八门：

| 方案 | 实现方式 | 问题 |
|------|---------|------|
| 共享消息总线 | 所有Agent连同一个MQ | 耦合度高，难以独立部署 |
| 直接HTTP调用 | Agent A调Agent B的API | 缺乏标准化，发现困难 |
| 框架内建通信 | LangGraph/AutoGen的Channel | 框架锁定，不可互操作 |
| 文件/数据库中转 | 读写共享存储 | 延迟高，无实时性 |

这种碎片化与MCP出现前的工具集成困境如出一辙。我们需要一个**标准化的Agent间通信协议**。

### 1.3 A2A的诞生

2025年4月，Google联合15+合作伙伴发布了A2A（Agent-to-Agent）协议，旨在解决这一问题。A2A的核心愿景：

> **让不同框架、不同厂商构建的Agent能够互相发现、互相通信、互相协作。**

A2A不是要取代现有Agent框架，而是在框架之上提供一个**通信标准层**：

```
┌─────────────────────────────────────────────┐
│              A2A协议层（通信标准）           │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ LangGraph│  │ CrewAI   │  │ AutoGen  │ │
│  │  Agent   │  │  Agent   │  │  Agent   │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│       ↑              ↑              ↑       │
│       └──────────────┼──────────────┘       │
│                      │                      │
│              A2A 协议通信                   │
│                                             │
└─────────────────────────────────────────────┘
```

## 二、A2A核心概念：六大支柱

### 2.1 概念总览

A2A协议围绕六个核心概念构建：

```
┌─────────────────────────────────────────────────────┐
│                   A2A核心概念                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Agent Card（Agent名片）                         │
│     → Agent的自我描述，能力声明                      │
│                                                     │
│  2. Agent-to-Agent Communication（通信）            │
│     → 基于HTTP的请求-响应模式                       │
│                                                     │
│  3. Task（任务）                                    │
│     → Agent间协作的基本单元                         │
│                                                     │
│  4. Streaming（流式传输）                           │
│     → 实时进度更新                                  │
│                                                     │
│  5. Push Notifications（推送通知）                  │
│     → 异步任务完成通知                              │
│                                                     │
│  6. Agent-to-Agent Authentication（认证）           │
│     → 基于OpenID Connect的身份验证                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 2.2 Agent Card：Agent的"数字名片"

Agent Card是A2A协议的入口点，定义了Agent的身份、能力和连接方式：

```json
{
  "name": "数据分析Agent",
  "description": "专业的数据分析Agent，擅长SQL查询、数据可视化和统计分析",
  "url": "https://data-agent.company.com",
  "version": "1.0.0",
  "documentationUrl": "https://docs.company.com/data-agent",
  "provider": {
    "organization": "Company AI",
    "url": "https://company.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "authentication": {
    "schemes": ["Bearer"],
    "credentials": "https://auth.company.com/.well-known/oauth-authorization-server"
  },
  "defaultInputModes": ["text", "text/plain", "application/json"],
  "defaultOutputModes": ["text", "text/plain", "application/json", "image/png"],
  "skills": [
    {
      "id": "sql-query",
      "name": "SQL查询",
      "description": "执行SQL查询并返回结构化结果",
      "tags": ["database", "sql", "query"],
      "examples": [
        "查询最近7天的用户活跃数据",
        "统计每个地区的销售额"
      ]
    },
    {
      "id": "data-visualization",
      "name": "数据可视化",
      "description": "将数据转换为图表和可视化报告",
      "tags": ["chart", "visualization", "report"],
      "examples": [
        "生成月度销售趋势图",
        "创建用户分群分布图"
      ]
    }
  ]
}
```

**Agent Card的设计哲学**：

1. **自描述性**：其他Agent无需事先了解，通过读取Card即可理解能力
2. **可发现性**：通过Well-Known URL（`/.well-known/agent.json`）标准化发现
3. **可互操作**：支持多模态输入输出，不限于文本

### 2.3 Agent发现机制

A2A定义了标准化的Agent发现流程：

```
Agent A（需求方）                    Agent B（提供方）
      │                                   │
      │  1. GET /.well-known/agent.json   │
      │──────────────────────────────────▶│
      │                                   │
      │  2. 返回Agent Card                │
      │◀──────────────────────────────────│
      │                                   │
      │  3. 解析能力声明                  │
      │     分析是否匹配需求              │
      │                                   │
      │  4. 发送Task请求                  │
      │──────────────────────────────────▶│
      │                                   │
```

对于企业级场景，还可以通过**Agent注册中心**实现大规模发现：

```
┌─────────────────────────────────────────┐
│           Agent注册中心                  │
├─────────────────────────────────────────┤
│                                         │
│  按能力索引：                           │
│  ┌─────────────┐                       │
│  │ SQL查询     │ → [数据分析Agent,      │
│  │             │    数据库Agent]        │
│  ├─────────────┤                       │
│  │ 代码生成   │ → [编程Agent,          │
│  │             │    代码审查Agent]      │
│  ├─────────────┤                       │
│  │ 翻译       │ → [多语言Agent]        │
│  └─────────────┘                       │
│                                         │
└─────────────────────────────────────────┘
```

## 三、通信机制：Task生命周期

### 3.1 Task状态机

A2A的核心协作单元是Task（任务），它有明确的状态生命周期：

```
┌─────────┐     submit      ┌──────────┐
│ (无任务) │────────────────▶│ submitted │
└─────────┘                 └────┬─────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
              ┌──────────┐ ┌─────────┐ ┌──────────┐
              │ working  │ │ failed  │ │ canceled │
              └────┬─────┘ └─────────┘ └──────────┘
                   │
          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼
    ┌─────────┐ ┌──────┐ ┌────────┐
    │completed│ │input │ │rejected│
    └─────────┘ │reques│ └────────┘
                │-ted  │
                └──────┘
```

状态说明：

| 状态 | 含义 | 触发条件 |
|------|------|---------|
| `submitted` | 任务已提交 | Client发送Task请求 |
| `working` | Agent正在处理 | Agent开始执行 |
| `input-required` | 需要额外输入 | Agent需要更多信息 |
| `completed` | 任务完成 | Agent返回最终结果 |
| `failed` | 任务失败 | 执行出错 |
| `canceled` | 任务取消 | Client主动取消 |
| `rejected` | 任务被拒绝 | Agent无法处理 |

### 3.2 Task请求与响应

**创建Task请求**：

```json
{
  "jsonrpc": "2.0",
  "id": "task-001",
  "method": "tasks/send",
  "params": {
    "id": "task-001",
    "sessionId": "session-abc",
    "messages": [
      {
        "role": "user",
        "parts": [
          {
            "type": "text",
            "text": "查询最近30天的用户留存率，并生成趋势图"
          }
        ]
      }
    ],
    "metadata": {
      "priority": "high",
      "deadline": "2026-05-31T18:00:00Z"
    }
  }
}
```

**Task响应（working状态）**：

```json
{
  "jsonrpc": "2.0",
  "id": "task-001",
  "result": {
    "id": "task-001",
    "sessionId": "session-abc",
    "status": {
      "state": "working",
      "message": {
        "role": "agent",
        "parts": [
          {
            "type": "text",
            "text": "正在查询数据库，预计需要30秒..."
          }
        ]
      }
    },
    "artifacts": []
  }
}
```

**Task响应（completed状态）**：

```json
{
  "jsonrpc": "2.0",
  "id": "task-001",
  "result": {
    "id": "task-001",
    "sessionId": "session-abc",
    "status": {
      "state": "completed"
    },
    "artifacts": [
      {
        "name": "留存率数据",
        "parts": [
          {
            "type": "data",
            "data": {
              "retention_rates": [
                {"day": 1, "rate": 0.85},
                {"day": 7, "rate": 0.42},
                {"day": 30, "rate": 0.18}
              ]
            }
          }
        ]
      },
      {
        "name": "趋势图",
        "parts": [
          {
            "type": "file",
            "file": {
              "name": "retention_trend.png",
              "mimeType": "image/png",
              "uri": "https://data-agent.company.com/artifacts/task-001/retention_trend.png"
            }
          }
        ]
      }
    ]
  }
}
```

### 3.3 多轮对话

A2A支持Task内的多轮对话，用于处理需要澄清的场景：

```
用户 → Agent: "分析销售数据"
Agent → 用户 (input-required): "请指定分析的时间范围？"
用户 → Agent: "最近一个季度"
Agent → 用户 (completed): [分析结果]
```

```json
// Agent请求更多信息
{
  "status": {
    "state": "input-required",
    "message": {
      "role": "agent",
      "parts": [
        {
          "type": "text",
          "text": "请指定分析的时间范围：最近一周/一月/一季/一年？"
        }
      ]
    }
  }
}

// 用户补充信息后继续
{
  "method": "tasks/send",
  "params": {
    "id": "task-001",
    "messages": [
      {
        "role": "user",
        "parts": [
          {
            "type": "text",
            "text": "最近一个季度"
          }
        ]
      }
    ]
  }
}
```

## 四、流式传输与推送通知

### 4.1 Server-Sent Events（SSE）流式传输

对于长时间运行的任务，A2A支持SSE实时推送进度：

```
Client                                    Server
  │                                         │
  │── POST /tasks/send (stream: true) ────▶│
  │                                         │
  │◀── SSE: {"status": "working"} ─────────│
  │◀── SSE: {"status": "working",          │
  │         "progress": "查询中..."} ───────│
  │◀── SSE: {"status": "working",          │
  │         "progress": "生成图表..."} ─────│
  │◀── SSE: {"status": "completed"} ───────│
  │                                         │
```

SSE事件格式：

```
event: TaskStatusUpdateEvent
data: {"id":"task-001","status":{"state":"working"},"final":false}

event: TaskStatusUpdateEvent
data: {"id":"task-001","status":{"state":"working"},"final":false}

event: TaskStatusUpdateEvent
data: {"id":"task-001","status":{"state":"completed"},"final":true}
```

### 4.2 Push Notifications（推送通知）

当Client无法保持长连接时（如移动端、Webhook），可使用推送通知：

```
Client                                    Server
  │                                         │
  │── POST /tasks/send ──────────────────▶│
  │   (pushNotification: {                 │
  │     url: "https://client.com/webhook"  │
  │   })                                   │
  │                                         │
  │◀── 202 Accepted ───────────────────────│
  │                                         │
  │     ... (Server异步处理) ...            │
  │                                         │
  │◀── POST https://client.com/webhook ────│
  │    {                                    │
  │      "taskId": "task-001",             │
  │      "status": "completed",            │
  │      "artifacts": [...]                │
  │    }                                    │
  │                                         │
```

推送通知配置：

```json
{
  "method": "tasks/send",
  "params": {
    "id": "task-002",
    "messages": [...],
    "pushNotification": {
      "url": "https://my-app.com/webhook/a2a",
      "token": "webhook-secret-token"
    }
  }
}
```

## 五、多Agent协作架构模式

### 5.1 三种协作模式

A2A支持多种Agent协作模式，适用于不同场景：

**① 链式协作（Chain）**

```
用户 → Agent A → Agent B → Agent C → 最终结果
        (翻译)    (摘要)    (排版)
```

适用于有明确前后依赖的流水线任务。

**② 路由协作（Router）**

```
              ┌→ Agent A (SQL专家)
用户 → Router │
              ├→ Agent B (可视化专家)
              │
              └→ Agent C (报告专家)
```

适用于需要根据任务类型分配给不同专家的场景。

**③ 辩论协作（Debate）**

```
         ┌→ Agent A (方案一) ──┐
用户 →   │                     ├→ 仲裁Agent → 最终方案
         └→ Agent B (方案二) ──┘
```

适用于需要多角度分析的复杂决策场景。

### 5.2 实战案例：多Agent数据分析系统

让我们构建一个完整的多Agent数据分析系统：

```
┌─────────────────────────────────────────────────────┐
│                多Agent数据分析系统                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐                                   │
│  │  Orchestrator│  ← 任务分解与协调                  │
│  │  (编排Agent) │                                   │
│  └──────┬──────┘                                   │
│         │                                           │
│    ┌────┼────┬────────┐                            │
│    ▼    ▼    ▼        ▼                            │
│  ┌───┐┌───┐┌───┐  ┌───┐                          │
│  │SQL││VIS││NLP│  │RPT│                          │
│  │Agent│Agent│Agent│  │Agent│                       │
│  └───┘└───┘└───┘  └───┘                          │
│   ↑    ↑    ↑        ↑                            │
│   │    │    │        │                            │
│  数据  图表  分析    报告                          │
│  查询  生成  洞察    输出                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Orchestrator Agent实现**：

```python
import httpx
from typing import List

class OrchestratorAgent:
    def __init__(self):
        self.agent_registry = {
            "sql": "https://sql-agent.company.com",
            "visualization": "https://vis-agent.company.com",
            "nlp": "https://nlp-agent.company.com",
            "report": "https://report-agent.company.com",
        }
        self.http_client = httpx.AsyncClient()
    
    async def discover_agent(self, agent_type: str) -> dict:
        """通过Agent Card发现Agent能力"""
        url = self.agent_registry[agent_type]
        card_url = f"{url}/.well-known/agent.json"
        resp = await self.http_client.get(card_url)
        return resp.json()
    
    async def delegate_task(self, agent_type: str, task: dict) -> dict:
        """向专业Agent委派任务"""
        agent_url = self.agent_registry[agent_type]
        
        # 1. 发现Agent
        card = await self.discover_agent(agent_type)
        
        # 2. 检查能力匹配
        required_skill = task.get("required_skill")
        available_skills = [s["id"] for s in card.get("skills", [])]
        if required_skill not in available_skills:
            raise ValueError(f"Agent {agent_type} 不支持技能: {required_skill}")
        
        # 3. 发送Task
        resp = await self.http_client.post(
            f"{agent_url}/tasks/send",
            json={
                "jsonrpc": "2.0",
                "id": task["id"],
                "method": "tasks/send",
                "params": {
                    "id": task["id"],
                    "messages": task["messages"],
                },
            },
        )
        return resp.json()
    
    async def orchestrate(self, user_request: str) -> dict:
        """编排多Agent协作"""
        # Step 1: SQL Agent查询数据
        sql_result = await self.delegate_task("sql", {
            "id": "step-1",
            "required_skill": "sql-query",
            "messages": [{
                "role": "user",
                "parts": [{"type": "text", "text": f"执行查询: {user_request}"}]
            }],
        })
        
        # Step 2: NLP Agent分析洞察
        analysis_result = await self.delegate_task("nlp", {
            "id": "step-2",
            "required_skill": "text-analysis",
            "messages": [{
                "role": "user",
                "parts": [{"type": "text", "text": f"分析以下数据: {sql_result}"}]
            }],
        })
        
        # Step 3: Visualization Agent生成图表
        viz_result = await self.delegate_task("visualization", {
            "id": "step-3",
            "required_skill": "chart-generation",
            "messages": [{
                "role": "user",
                "parts": [{"type": "text", "text": f"为以下数据生成图表: {sql_result}"}]
            }],
        })
        
        # Step 4: Report Agent整合报告
        report_result = await self.delegate_task("report", {
            "id": "step-4",
            "required_skill": "report-generation",
            "messages": [{
                "role": "user",
                "parts": [
                    {"type": "text", "text": f"数据分析: {analysis_result}"},
                    {"type": "text", "text": f"图表: {viz_result}"},
                ]
            }],
        })
        
        return report_result
```

## 六、A2A vs MCP：互补而非竞争

### 6.1 核心区别

A2A和MCP经常被混淆，但它们解决的是不同层面的问题：

| 维度 | A2A | MCP |
|------|-----|-----|
| 通信对象 | Agent ↔ Agent | Agent ↔ Tool |
| 通信模式 | 双向对等协作 | 单向工具调用 |
| 能力发现 | Agent Card | tools/list |
| 任务模型 | 有状态Task | 无状态Request |
| 身份认证 | Agent身份 | 工具权限 |
| 典型场景 | 多Agent分工协作 | Agent使用外部工具 |

### 6.2 协同使用

A2A和MCP可以（也应该）协同使用：

```
┌─────────────────────────────────────────────────────┐
│              Agent = A2A + MCP                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │           Agent Core                │           │
│  │      (LLM + 推理 + 记忆)           │           │
│  └──────────┬──────────┬───────────────┘           │
│             │          │                            │
│    ┌────────▼───┐  ┌───▼────────┐                  │
│    │  A2A层     │  │  MCP层      │                  │
│    │            │  │             │                  │
│    │ 与其他Agent │  │ 使用工具    │                  │
│    │ 通信协作   │  │ (文件/DB等) │                  │
│    └────────────┘  └─────────────┘                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**实际工作流**：

1. **Agent A**通过A2A协议向**Agent B**发起协作请求
2. **Agent B**接收任务后，通过MCP协议调用数据库工具查询数据
3. **Agent B**处理数据后，通过A2A协议将结果返回给**Agent A**
4. **Agent A**收到结果后，通过MCP协议调用文件工具保存报告

## 七、安全模型：信任与认证

### 7.1 Agent身份认证

A2A基于OpenID Connect实现Agent身份认证：

```
┌──────────┐                    ┌──────────┐
│ Agent A  │                    │ Agent B  │
│          │                    │          │
│ 持有ID   │──── 1.请求 ──────▶│          │
│ Token    │                    │          │
│          │◀─── 2.验证Token ──│          │
│          │     (向Auth Server)│          │
│          │                    │          │
│          │◀─── 3.认证成功 ───│          │
│          │                    │          │
```

### 7.2 权限控制

A2A支持基于技能的权限控制：

```json
{
  "authentication": {
    "schemes": ["Bearer"],
    "credentials": "https://auth.company.com/.well-known/oauth-authorization-server"
  },
  "skills": [
    {
      "id": "sql-query",
      "permissions": {
        "required_scopes": ["data:read"],
        "allowed_callers": ["orchestrator-agent", "analyst-agent"],
        "rate_limit": "100/hour"
      }
    }
  ]
}
```

## 八、生产环境部署

### 8.1 部署架构

```
┌─────────────────────────────────────────────────────┐
│                  生产部署架构                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐                                   │
│  │ API Gateway │  ← 认证、限流、路由               │
│  └──────┬──────┘                                   │
│         │                                           │
│    ┌────┼────────────┐                            │
│    ▼    ▼            ▼                            │
│  ┌───┐┌───┐  ┌──────────┐                        │
│  │A  ││B  │  │ Agent    │                        │
│  │   ││   │  │ Registry │                        │
│  └───┘└───┘  └──────────┘                        │
│                                                     │
│  每个Agent独立部署：                               │
│  • Docker容器                                      │
│  • 独立端口                                        │
│  • 独立扩缩容                                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 8.2 可观测性

```python
# A2A可观测性中间件
class A2AObservability:
    def __init__(self):
        self.tracer = TraceProvider.get_tracer("a2a")
        self.metrics = MetricsProvider.get_metrics("a2a")
    
    async def trace_task(self, task_id: str, operation: str):
        with self.tracer.start_as_current_span(
            f"a2a.task.{operation}",
            attributes={"task.id": task_id}
        ) as span:
            yield span
    
    def record_task_metrics(self, task: Task):
        self.metrics.record_histogram(
            "a2a.task.duration",
            value=task.duration,
            labels={
                "agent": task.agent_id,
                "status": task.status,
                "skill": task.skill_id,
            }
        )
```

## 九、未来展望

### 9.1 A2A的演进方向

1. **更丰富的交互模式**：支持音频、视频等多模态Agent通信
2. **更完善的信任机制**：去中心化的Agent身份验证
3. **更智能的路由**：基于Agent能力的动态任务分配
4. **更广泛的生态**：更多框架、更多厂商的接入

### 9.2 A2A + MCP的未来

A2A和MCP共同构成了AI Agent的"网络协议栈"：

```
┌─────────────────────────────────────────┐
│        应用层（Agent逻辑）              │
├─────────────────────────────────────────┤
│        A2A层（Agent间通信）             │
├─────────────────────────────────────────┤
│        MCP层（工具集成）                │
├─────────────────────────────────────────┤
│        传输层（HTTP/stdio）             │
├─────────────────────────────────────────┤
│        基础设施层（网络/计算）           │
└─────────────────────────────────────────┘
```

## 十、总结

A2A协议的出现标志着多Agent协作从"框架内建"走向"标准互通"。它的核心价值在于：

1. **互操作性**：不同框架的Agent可以无缝协作
2. **可发现性**：Agent Card让能力声明标准化
3. **有状态协作**：Task模型支持复杂的多轮交互
4. **安全内建**：基于OpenID Connect的身份认证

对于开发者而言，A2A不是要你重写现有的Agent系统，而是在现有系统之上添加一个标准化的通信层。这就像HTTP不关心你用什么语言编写服务器一样，A2A也不关心你用什么框架构建Agent。

展望未来，A2A + MCP将共同构建AI Agent的"互联网"——一个Agent可以自由发现、自由协作、自由使用工具的开放生态。这不仅是技术的演进，更是AI从"单体智能"走向"群体智能"的关键一步。

---

*本文基于A2A协议规范（2025-04版本）和Google官方文档撰写。协议仍在快速演进中，建议关注官方仓库获取最新信息。*
