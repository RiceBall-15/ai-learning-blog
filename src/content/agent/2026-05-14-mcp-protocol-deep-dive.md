---
title: MCP协议深度解析：从架构设计到自定义Server开发
description: 全面剖析Anthropic推出的Model Context Protocol，对比OpenAPI Plugin和LangChain Tools，详解核心原语、传输层机制，并给出自定义MCP Server的开发实践
date: 2026-05-14
author: RiceBall-15
category: agent
tags: [MCP, Agent, 协议设计, 工具调用, Anthropic, 标准化]
draft: false
---

# MCP协议深度解析：从架构设计到自定义Server开发

## 简介

MCP（Model Context Protocol）是Anthropic于2024年底推出的开放协议，旨在标准化AI Agent与外部工具/数据源之间的通信。它被称为"AI领域的USB-C"——一个统一接口替代碎片化的集成方式。本文深入剖析MCP的协议架构、核心原语、传输层机制，并对比竞品方案，最后给出自定义Server的开发实践。

## 问题背景

在MCP出现之前，Agent与工具的集成面临严重的碎片化问题：

| 集成方式 | 典型场景 | 核心痛点 |
|---------|---------|---------|
| OpenAI Function Calling | 单厂商绑定 | 只支持OpenAI生态，无法跨模型复用 |
| LangChain Tools | 框架绑定 | 工具定义耦合框架，迁移成本高 |
| 自定义HTTP API | 完全自由 | 每个工具都要重写Schema和调用逻辑 |
| OpenAPI Plugin | Web API | 聚焦HTTP REST，不适合本地资源和实时流 |

**核心矛盾**：工具开发者需要为每个模型/框架重复适配，Agent开发者需要为每个工具重写集成代码。

MCP的解法：**协议层统一，让工具和Agent各自只对接一个标准**。

## 协议架构

### Client-Server模型

```
┌──────────────────────────────────────────────┐
│                  AI Application              │
│  ┌─────────────┐  ┌─────────────┐           │
│  │ MCP Client 1│  │ MCP Client 2│  ...      │
│  └──────┬──────┘  └──────┬──────┘           │
│         │                │                   │
│    ┌────┴────┐     ┌─────┴────┐             │
│    │Transport│     │Transport │             │
│    └────┬────┘     └─────┬────┘             │
└─────────┼────────────────┼──────────────────┘
          │                │
    ┌─────┴─────┐   ┌──────┴─────┐
    │MCP Server │   │ MCP Server │
    │  (File)   │   │  (DB/API)  │
    └───────────┘   └────────────┘
```

**MCP Host**：运行AI模型的应用（如Claude Desktop、IDE插件）
**MCP Client**：在Host内维护与Server的1:1连接
**MCP Server**：暴露工具/资源/提示的轻量服务

### 核心设计原则

1. **协议与传输分离**：协议逻辑不绑定具体传输方式
2. **渐进式能力协商**：Client和Server在初始化时声明各自能力
3. **双向通信**：Server也可以向Client发起请求（如采样请求）
4. **安全边界**：每个Server运行在独立进程，天然隔离

## 核心原语详解

MCP定义了三大原语，覆盖Agent对外交互的主要场景：

### 1. Tools（工具）

Agent可以调用的动作。类似Function Calling，但更丰富。

```json
{
  "name": "query_database",
  "description": "执行SQL查询并返回结果",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sql": {"type": "string", "description": "SQL查询语句"},
      "database": {"type": "string", "enum": ["prod", "staging"]}
    },
    "required": ["sql"]
  },
  "annotations": {
    "readOnlyHint": true,
    "openWorldHint": false
  }
}
```

**与Function Calling的区别**：
- 支持`annotations`元数据（是否只读、是否影响外部世界）
- 支持`_meta`字段传递自定义元数据
- 工具列表可动态变化（Server发`tools/list_changed`通知）

### 2. Resources（资源）

Agent可以读取的数据源。类似文件系统，但不限于文件。

```
URI格式: scheme://path
示例:
  file:///project/src/main.py
  postgres://localhost/mydb/users
  github://owner/repo/README.md
```

**核心特性**：
- 支持资源模板（动态URI，如`user://{user_id}/profile`）
- 支持订阅（资源变化时通知Client）
- 返回MIME类型，Client可据此决定渲染方式

### 3. Prompts（提示模板）

Server提供的可复用提示模板。

```json
{
  "name": "code_review",
  "description": "对代码变更进行审查",
  "arguments": [
    {"name": "diff", "description": "Git diff内容", "required": true},
    {"name": "focus", "description": "审查重点", "required": false}
  ]
}
```

**使用场景**：Server开发者可以提供最佳实践的Prompt，用户无需自己设计。

## 传输层对比

MCP支持多种传输方式，适应不同部署场景：

| 传输方式 | 适用场景 | 通信模式 | 延迟 | 部署复杂度 |
|---------|---------|---------|------|-----------|
| stdio | 本地工具（CLI集成） | 同机进程间 | <1ms | 低 |
| HTTP + SSE | 远程服务（Web） | 请求-响应 + 服务端推送 | ~10-50ms | 中 |
| Streamable HTTP | 远程服务（新版） | 双向流 | ~10-50ms | 中 |

### stdio传输

最简单的传输方式。Host启动Server进程，通过stdin/stdout通信。

```
Host启动: child_process.spawn("my-mcp-server", ["--config", "x.json"])
通信: JSON-RPC 2.0 over stdin/stdout
关闭: 发送SIGTERM或关闭stdin
```

**优势**：零网络开销，适合本地工具
**限制**：只能单机部署，不支持多Host共享

### HTTP + SSE传输

传统Web架构。Client发起HTTP请求，Server通过SSE推送通知。

```
Client → POST /mcp (JSON-RPC请求)
Server → SSE: event: message, data: {JSON-RPC响应}
Server → SSE: event: message, data: {通知/请求}  (可选的后续推送)
```

**优势**：标准HTTP基础设施，防火墙友好
**限制**：SSE是单向的，Server-to-Client请求需要额外轮询

### Streamable HTTP（2025新标准）

改进的HTTP传输，支持真正的双向流。

```
Client → POST /mcp (可包含多个请求)
Server → Response (Content-Type: text/event-stream)
  event: message
  data: {"jsonrpc":"2.0","id":1,"result":{...}}
  event: message
  data: {"jsonrpc":"2.0","method":"notifications/progress",...}
```

**关键改进**：
- Server可以在同一响应中返回多个消息
- 支持服务端主动发起请求
- 兼容无状态模式（每个请求独立session）

## 自定义MCP Server开发

### 技术选型

| 语言 | SDK | 适用场景 |
|------|-----|---------|
| TypeScript | @modelcontextprotocol/sdk | Web服务、IDE插件集成 |
| Python | mcp (PyPI) | 数据处理、ML模型包装 |
| Go | mark3labs/mcp-go | 高性能服务、系统工具 |
| Java/Spring | spring-ai-mcp | 企业级集成、Spring生态 |

### 开发流程（以Python为例）

```
第1步: 定义Server和工具
  @server.tool()
  def query_database(sql: str, database: str = "prod") -> dict:
      """执行SQL查询"""
      ...

第2步: 定义资源（可选）
  @server.resource("schema://{table}")
  def get_table_schema(table: str) -> str:
      ...

第3步: 运行Server
  mcp.run()  # 默认stdio传输
```

### 生产化要点

**1. 认证与授权**
- stdio传输：由Host进程管理，通常不需要额外认证
- HTTP传输：在传输层加OAuth2 / API Key

**2. 速率限制**
- 在Server内部实现per-tool的rate limit
- MCP协议本身不定义限流，由Server自行控制

**3. 错误处理**
- 标准错误码：-32600(无效请求)、-32601(方法不存在)、-32602(无效参数)
- 业务错误通过result返回，不要用error码

## 与竞品协议对比

| 维度 | MCP | OpenAI Plugin | LangChain Tools | A2A |
|------|-----|--------------|-----------------|-----|
| 定位 | Agent↔工具 | ChatGPT↔API | Agent↔工具(框架内) | Agent↔Agent |
| 跨模型 | ✅ 协议层统一 | ❌ ChatGPT绑定 | ❌ 框架绑定 | ✅ 协议层统一 |
| 本地工具 | ✅ stdio | ❌ 仅HTTP | ✅ | ❌ 设计为远程 |
| 资源订阅 | ✅ 原生支持 | ❌ | ❌ | ❌ |
| 双向通信 | ✅ Server可请求 | ❌ 单向 | ❌ 单向 | ✅ |
| 成熟度 | 快速增长中 | 稳定但停滞 | 高 | 早期阶段 |

**MCP vs A2A的关系**：
- MCP解决Agent↔工具的通信（垂直方向）
- A2A解决Agent↔Agent的通信（水平方向）
- 两者互补而非替代

## 生产部署注意事项

### 安全边界

```
每个MCP Server应视为不可信来源：
  - 工具返回内容可能包含Prompt Injection
  - 资源内容可能包含恶意指令
  - 必须在Host侧做输入/输出过滤
```

### 性能优化

- **连接池**：对HTTP传输复用连接
- **工具缓存**：对幂等工具的结果做TTL缓存
- **延迟加载**：工具列表首次调用时获取，后续用`tools/list_changed`通知刷新

### 版本兼容

MCP协议仍在快速演进（2025年3月发布了Streamable HTTP）。生产部署应：
- 锁定SDK版本
- 在能力协商阶段检查协议版本
- 对不支持的能力优雅降级

## 总结

MCP通过标准化协议层解决了Agent工具集成的碎片化问题。核心价值在于：

1. **工具开发者只写一次**，所有MCP Host都能使用
2. **Agent开发者只对接一个协议**，所有MCP Server都能集成
3. **三大原语**（Tools/Resources/Prompts）覆盖主流交互场景
4. **传输层灵活**，从本地stdio到远程HTTP均可

对于正在构建Agent系统的团队，MCP值得作为工具集成层的首选方案。

---

**参考资料**：
- Anthropic MCP Specification (2025.03)
- MCP SDK: github.com/modelcontextprotocol
- Spring AI MCP Integration
- Google A2A Protocol Whitepaper
