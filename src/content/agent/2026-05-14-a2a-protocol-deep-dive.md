---
title: A2A协议：Google的多Agent通信标准深度解读
description: 深度剖析Google推出的Agent-to-Agent (A2A) 协议，对比MCP定位差异，详解Agent Card、Task生命周期、消息机制与多Agent互操作实践
date: 2026-05-14
author: RiceBall-15
category: agent
tags: [A2A, Agent, 多智能体, Google, 通信协议, 互操作性]
draft: false
---

# A2A协议：Google的多Agent通信标准深度解读

## 简介

2025年4月，Google联合50+技术伙伴（包括Salesforce、SAP、MongoDB等）推出了A2A（Agent-to-Agent）协议，旨在解决多Agent系统间的互操作性问题。如果说MCP是Agent与工具之间的"USB-C"，那么A2A就是Agent与Agent之间的"HTTP"。本文深入解读A2A协议的设计哲学、核心机制，并与MCP做系统性对比。

## 问题背景

### 多Agent互操作的碎片化困境

当前多Agent生态面临的根本问题：

```
Agent A (LangChain)  ─── 无法直接调用 ───  Agent B (CrewAI)
Agent A (自研框架)    ─── 无法直接调用 ───  Agent C (Google ADK)
Agent A (Python)     ─── 无法直接调用 ───  Agent D (Java/Spring)
```

**每个Agent框架都有自己的通信约定**：
- LangChain用内部的AgentExecutor
- CrewAI用Crew/Task模型
- AutoGen用GroupChat
- Google ADK用Session

**结果**：想让不同框架开发的Agent协作，只能靠胶水代码一对一适配。

### A2A的定位

A2A要解决的不是"Agent内部怎么运行"，而是"Agent之间怎么通信"：

```
┌────────────────────────────────────────────────────┐
│                  Agent A 内部实现                    │
│  (LangChain / CrewAI / 自研框架 / ...)              │
│                    │                                │
│              A2A协议层 (统一接口)                    │
└────────────────────┼────────────────────────────────┘
                     │  ← A2A通信 →
┌────────────────────┼────────────────────────────────┐
│              A2A协议层 (统一接口)                    │
│                    │                                │
│                  Agent B 内部实现                    │
│  (Google ADK / AutoGen / 自研框架 / ...)            │
└────────────────────────────────────────────────────┘
```

## 协议架构

### 核心设计原则

1. **框架无关**：协议层与实现层分离，任何语言/框架都能接入
2. **异步优先**：Agent协作天然需要等待，异步Task是第一类概念
3. **能力发现**：Agent通过标准方式声明自己的能力
4. **渐进增强**：支持从简单消息到复杂流式交互的渐进式能力

### 技术基础

A2A基于成熟的Web标准构建：

| 技术层 | 选择 | 理由 |
|-------|------|------|
| 传输 | HTTP(S) | 最广泛的网络基础设施 |
| 消息格式 | JSON-RPC 2.0 | 轻量、标准化、双向 |
| 流式 | Server-Sent Events | 标准HTTP流式方案 |
| 认证 | OAuth 2.0 / API Key | 复用企业现有体系 |
| Schema | JSON Schema | 工具/能力描述标准 |

## 核心概念

### 1. Agent Card（Agent名片）

每个A2A Agent必须暴露一个Agent Card，描述自己的身份和能力。

```json
{
  "name": "ResearchAgent",
  "description": "深度技术调研Agent，擅长学术论文检索和分析",
  "url": "https://research-agent.example.com/a2a",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": false
  },
  "authentication": {
    "schemes": ["bearer"]
  },
  "skills": [
    {
      "id": "paper_search",
      "name": "论文检索",
      "description": "在arXiv、Semantic Scholar等平台搜索学术论文",
      "tags": ["research", "academic", "search"],
      "examples": [
        "帮我找最近关于Transformer架构改进的论文"
      ],
      "inputModes": ["text"],
      "outputModes": ["text", "file"]
    },
    {
      "id": "literature_review",
      "name": "文献综述",
      "description": "对多篇论文进行综合分析和综述",
      "tags": ["research", "analysis"]
    }
  ],
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"]
}
```

**关键字段解读**：

| 字段 | 作用 | 与MCP的区别 |
|------|------|------------|
| `skills` | 声明可执行的任务类型 | 类似MCP的Tools，但粒度更粗（任务级 vs 函数级） |
| `capabilities` | 声明协议级能力 | MCP的capabilities更聚焦工具/资源/提示原语 |
| `inputModes/outputModes` | 支持的媒体类型 | MCP通过MIME type在Resource级别定义 |

### 2. Task（任务）

A2A的核心交互单元。一次Agent间协作被建模为Task。

**Task生命周期**：

```
                ┌──────────┐
                │ submitted │ ← 任务已提交
                └─────┬────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
        ┌──────────┐   ┌──────────┐
        │  working  │   │rejected  │ ← Agent拒绝执行
        └─────┬────┘   └──────────┘
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
┌──────────┐ ┌────┐ ┌──────────┐
│completed │ │auth│ │  failed  │
└──────────┘ └────┘ └──────────┘
                │
                ▼
          ┌──────────┐
          │canceled  │
          └──────────┘
```

**状态说明**：
| 状态 | 含义 | 后续可能 |
|------|------|---------|
| `submitted` | 任务已提交，等待处理 | working / rejected |
| `working` | Agent正在处理 | completed / failed / auth-required |
| `completed` | 任务完成 | (终态) |
| `failed` | 任务失败 | (终态) |
| `canceled` | 任务被取消 | (终态) |
| `rejected` | Agent拒绝执行 | (终态) |
| `auth-required` | 需要额外认证 | working / failed |
| `input-required` | 需要用户提供更多信息 | working / failed |

### 3. Message与Part

Task内的通信通过Message完成，Message由多个Part组成。

**Part类型**：

```
Part
  ├── TextPart: 纯文本内容
  ├── FilePart: 文件（内联base64或URL引用）
  └── DataPart: 结构化数据（JSON）
```

**消息示例**：
```json
{
  "role": "agent",
  "messageId": "msg-123",
  "taskId": "task-456",
  "parts": [
    {"kind": "text", "text": "我找到了3篇相关论文："},
    {"kind": "data", "data": {
      "papers": [
        {"title": "Attention Is All You Need", "year": 2017},
        {"title": "GPT-4 Technical Report", "year": 2023}
      ]
    }},
    {"kind": "file", "file": {
      "name": "survey_report.pdf",
      "mimeType": "application/pdf",
      "uri": "https://storage.example.com/reports/survey.pdf"
    }}
  ]
}
```

## 通信模式

### 1. 同步请求-响应（Synchronous）

最简单的模式。发送Task，等待完成，获取结果。

```
Client Agent                Server Agent
    │                            │
    │─── tasks/send ────────────►│
    │                            │ (处理中...)
    │◄── TaskStatus: completed ──│
    │                            │
```

**适用场景**：快速任务（<30s），如简单查询、格式转换。

### 2. 异步轮询（Asynchronous Polling）

提交任务后定期检查状态。

```
Client Agent                Server Agent
    │                            │
    │─── tasks/send ────────────►│
    │◄── TaskStatus: submitted ──│
    │                            │ (处理中...)
    │─── tasks/get ─────────────►│
    │◄── TaskStatus: working ────│
    │                            │ (继续处理...)
    │─── tasks/get ─────────────►│
    │◄── TaskStatus: completed ──│
    │                            │
```

**适用场景**：长时间任务（分钟/小时级），如数据分析、报告生成。

### 3. 流式更新（Streaming）

通过SSE实时推送任务进展。

```
Client Agent                Server Agent
    │                            │
    │─── tasks/send ────────────►│
    │◄── SSE: status.working ────│
    │◄── SSE: artifact (part 1) ─│  ← 中间结果
    │◄── SSE: artifact (part 2) ─│  ← 中间结果
    │◄── SSE: status.completed ──│
    │                            │
```

**适用场景**：需要实时反馈的任务，如实时搜索、流式生成。

### 4. 推送通知（Push Notification）

Server主动推送任务状态变更到Client指定的Webhook。

```
Client Agent                Server Agent              Webhook
    │                            │                        │
    │─── tasks/send ────────────►│                        │
    │ (含 pushNotification config)│                       │
    │                            │── POST /callback ─────►│
    │                            │   (status: completed)  │
    │                            │                        │
```

**适用场景**：客户端可能断开连接的场景，如移动端Agent。

## A2A vs MCP：互补而非竞争

这是理解A2A最重要的认知。两者解决不同维度的问题：

```
                    MCP (垂直)
                       │
          ┌────────────┼────────────┐
          │            │            │
       工具/数据源   工具/数据源   工具/数据源
          │            │            │
  ────────┼────────────┼────────────┼──────── A2A (水平)
          │            │            │
      Agent A ←────→ Agent B ←────→ Agent C
```

| 维度 | MCP | A2A |
|------|-----|-----|
| 通信方向 | Agent ↔ 工具/数据 (垂直) | Agent ↔ Agent (水平) |
| 交互粒度 | 函数级（Tool调用） | 任务级（Task协作） |
| 核心原语 | Tools, Resources, Prompts | Skills, Tasks, Messages |
| 状态管理 | 通常无状态 | 有状态（Task生命周期） |
| 发起方 | 总是Agent发起 | Agent间对等发起 |
| 典型场景 | 查数据库、调API、读文件 | 委托任务、协作完成复杂工作 |

### 组合使用场景

一个Agent同时使用MCP和A2A：

```
┌─ Research Agent ────────────────────────────┐
│                                             │
│  MCP Server: arXiv API  ──→ 搜索论文        │
│  MCP Server: PDF Parser ──→ 解析PDF         │
│  MCP Server: DB Access  ──→ 存储结果        │
│                                             │
│  A2A Peer: Writing Agent ──→ 委托撰写报告   │
│  A2A Peer: Review Agent  ──→ 委托同行评审   │
│                                             │
└─────────────────────────────────────────────┘
```

## 与其他多Agent方案对比

| 维度 | A2A | AutoGen GroupChat | CrewAI Crew | LangGraph |
|------|-----|-------------------|-------------|-----------|
| 定位 | 通信协议 | 运行时框架 | 运行时框架 | 编排框架 |
| 跨框架 | ✅ 协议统一 | ❌ 框架内 | ❌ 框架内 | ❌ 框架内 |
| 跨语言 | ✅ HTTP | ❌ Python | ❌ Python | ❌ Python |
| 网络部署 | ✅ 远程Agent | ⚠️ 需额外适配 | ❌ 本地为主 | ⚠️ 需额外适配 |
| 生态 | Google + 50伙伴 | Microsoft | 独立 | LangChain |
| 成熟度 | 早期（2025.04） | 较成熟 | 较成熟 | 较成熟 |

**A2A的独特价值**：不试图替代框架，只标准化通信层。让不同框架开发的Agent可以互操作。

## 实践指南

### 接入A2A的方式

**方式1：原生实现**
直接基于A2A规范实现HTTP端点。适合已有Agent想要暴露A2A接口。

```
需要实现的端点:
  POST /a2a              → 处理JSON-RPC请求
  GET  /a2a/.well-known/agent-card.json → 暴露Agent Card
  POST /a2a/webhook      → 接收推送通知（可选）
```

**方式2：使用SDK**
Google和社区提供了多语言SDK：
- Python: `a2a-sdk` 
- TypeScript: `@a2a-js/sdk`
- Go: 社区贡献

### Agent Card发布

Agent Card应该通过两个渠道发布：
1. **Well-Known URI**：`https://example.com/.well-known/agent-card.json`
2. **注册中心**：向Agent目录服务注册（类似API Gateway）

### 安全考量

| 风险 | 对策 |
|------|------|
| Agent Card被篡改 | 签名验证 + HTTPS |
| 跨Agent数据泄露 | 按Task隔离上下文 |
| 恶意Agent | Agent白名单 + 能力审核 |
| 任务无限执行 | Task超时 + 资源配额 |

## 当前局限与展望

### 现有局限

1. **生态早期**：2025年4月发布，实际生产案例还很少
2. **编排层缺失**：A2A只定义通信协议，不提供Agent编排逻辑
3. **复杂交互模式**：对多方协商、竞标等复杂交互模式支持有限
4. **调试工具不足**：缺乏类似MCP Inspector的调试工具

### 未来方向

- **与MCP深度融合**：Agent可以同时暴露A2A接口和MCP Server
- **Agent市场**：基于Agent Card的Agent发现和组合平台
- **标准化评估**：基于A2A的Agent能力基准测试

## 总结

A2A是多Agent互操作的通信标准，核心价值在于：

1. **打破框架壁垒**：不同框架开发的Agent可以互操作
2. **复用Web标准**：HTTP + JSON-RPC + SSE，学习成本低
3. **Task为中心**：有状态的任务模型，适配长时间协作
4. **与MCP互补**：MCP管垂直（工具），A2A管水平（Agent间）

对于正在构建多Agent系统的团队，A2A值得作为跨Agent通信的首选方案。

---

**参考资料**：
- Google A2A Protocol Specification (2025.04)
- A2A GitHub: github.com/google/A2A
- MCP Specification: spec.modelcontextprotocol.io
- "Why the World Needs an Agent Protocol" - Google Blog
