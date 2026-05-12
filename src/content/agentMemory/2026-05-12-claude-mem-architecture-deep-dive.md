---
title: "Claude-Mem：AI Agent持久化记忆系统架构深度解析"
description: "深度解析claude-mem项目的架构设计、Hook生命周期机制、SQLite/ChromaDB双存储方案、渐进式搜索策略，以及如何解决AI Agent跨会话上下文丢失问题"
date: 2026-05-12
author: RiceBall-15
category: agentMemory
tags:
  - AI Agent
  - 持久化记忆
  - Claude-Mem
  - 系统架构
  - Hook机制
  - ChromaDB
  - 向量搜索
draft: false
---

# Claude-Mem：AI Agent持久化记忆系统架构深度解析

> **项目来源**：[thedotmack/claude-mem](https://github.com/thedotmack/claude-mem) | **Stars**: 74,911 | **License**: Apache 2.0 | **版本**: 6.5.0

## 一、Claude-Mem是什么？

Claude-Mem是一个**持久化记忆压缩系统**，专为Claude Code等AI Agent设计。它解决了AI Agent最核心的痛点：**会话结束后上下文丢失**。

### 核心价值主张

- **跨会话持久化**：Agent在会话中执行的所有操作都会被捕获、压缩，并在未来的会话中注入相关上下文
- **多平台支持**：不仅支持Claude Code，还兼容OpenClaw、Codex、Gemini、Hermes、Copilot、OpenCode等
- **零干预运行**：全自动操作，无需手动干预
- **Token效率**：通过渐进式披露（Progressive Disclosure）机制，节省约10倍Token消耗

## 二、解决什么问题？

### 传统AI Agent的困境

```
会话1: 用户详细描述了项目结构、技术栈、开发规范...
       Agent理解并开始工作...
       会话结束 → 所有上下文丢失 ❌

会话2: 用户："继续上次的工作"
       Agent："请问项目是什么技术栈？" 😤
```

### Claude-Mem的解决方案

```
会话1: 所有工具使用、决策、发现都被捕获 → 压缩存储
       ↓
会话2: 自动注入相关上下文
       Agent："基于上次的进展，我们继续优化认证模块" ✅
```

## 三、核心架构

### 3.1 系统分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Claude Code (宿主层)                                        │
│  ├── Hook System (5个生命周期事件)                            │
│  └── MCP Client (搜索工具)                                   │
├─────────────────────────────────────────────────────────────┤
│  CLI层 (Bun运行时)                                           │
│  ├── bun-runner.js (Node→Bun桥接)                           │
│  ├── hook-command.ts (编排器)                                │
│  └── handlers/ (context, session-init, observation,         │
│                 summarize, session-complete)                 │
├─────────────────────────────────────────────────────────────┤
│  Worker守护进程 (Express, 端口37700+(uid%100))               │
│  ├── SessionManager (会话生命周期)                           │
│  ├── SDKAgent (Claude Agent SDK)                            │
│  ├── SearchManager (搜索编排)                                │
│  ├── ProcessRegistry (子进程管理)                            │
│  └── ChromaSync (嵌入同步)                                   │
├─────────────────────────────────────────────────────────────┤
│  存储层                                                       │
│  ├── SQLite (claude-mem.db) ─ 结构化数据                     │
│  ├── ChromaDB (chroma.sqlite3) ─ 向量嵌入                    │
│  └── MCP Server (Claude Code接口)                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Hook生命周期机制

Hook是Claude-Mem的核心触发机制，包含5个生命周期事件：

| 事件 | Handler | 功能 | 超时 |
|------|---------|------|------|
| **Setup** | version-check.js | 版本检查，提示修复 | 60s |
| **SessionStart** | worker start + context | 启动Worker服务，注入上下文 | 60s |
| **UserPromptSubmit** | session-init | 注册会话，启动SDK Agent，语义注入 | 60s |
| **PostToolUse** | observation | 捕获工具使用 → 入队Worker | 120s |
| **Stop** | summarize | 请求会话摘要 | 120s |
| **SessionEnd** | session-complete | 结束会话，排空待处理消息 | 30s |

### 3.3 顺序执行流程（同步链路）

```
用户输入提示
    ↓
UserPromptSubmit Hook
    ↓
POST /api/sessions/init ──────→ 注册会话到SQLite
    ↓
POST /api/context/semantic ────→ 从ChromaDB检索相关记忆
    ↓
注入上下文到当前会话
    ↓
Agent开始工作，每次工具调用触发PostToolUse
    ↓
POST /api/sessions/observations → 入队PendingMessageStore
    ↓
Session结束 → POST /api/sessions/summarize
    ↓
POST /api/sessions/complete + drain队列
```

### 3.4 后台异步触发机制

#### 异步队列处理（PendingMessageStore）

```
工具调用 → enqueue() → INSERT pending状态行
                          ↓
              SDKAgent.startSession()
                          ↓
              Claude Agent SDK → ResponseProcessor
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
    storeObservations()  chromaSync  broadcastObservation()
         ↓                ↓              ↓
       SQLite           ChromaDB      SSE/UI实时推送
```

#### Generator重启循环（容错机制）

```
Generator崩溃 → 重试1 (1s) → 重试2 (2s) → 重试3 (4s)
                ↓
    连续重启 > 3次 → 停止，让迭代器结束
```

**关键设计**：计数器在Generator自然完成时重置为0，待处理消息在队列中保持，下次有效响应时由解析器清除。

#### 优雅降级策略

```
传输错误 (ECONNREFUSED, timeout, 5xx) → exit 0 (永不阻塞Claude Code)
客户端bug (4xx, TypeError, ReferenceError) → exit 2 (阻塞，需要修复)
```

**核心原则**：Worker不可用时**永远不会阻塞用户的Claude Code会话**。

## 四、存储与查询机制

### 4.1 SQLite存储结构

| 表名 | 关键字段 | 用途 |
|------|----------|------|
| **projects** | id, name, root_path | 项目管理 |
| **server_sessions** | content_session_id, memory_session_id, status | 会话生命周期 |
| **agent_events** | source_type, event_type, payload | 事件追踪 |
| **memory_items** | kind, type, title, narrative, facts | 记忆条目（核心） |
| **memory_sources** | memory_item_id, source_type | 记忆来源追踪 |
| **api_keys** | key_hash, scopes | API密钥管理 |
| **audit_log** | actor_type, action, target_type | 审计日志 |

### 4.2 FTS5全文搜索

```sql
CREATE VIRTUAL TABLE memory_items_fts USING fts5(
    memory_item_id UNINDEXED,
    project_id UNINDEXED,
    title,
    subtitle,
    text,
    narrative,
    facts,
    concepts,
    tokenize='porter unicode61'
);
```

支持Porter词干提取和Unicode处理，实现高效的全文检索。

### 4.3 ChromaDB向量搜索

每个Observation生成多个文档用于语义搜索：

```
obs_{id}_narrative  → 主文本
obs_{id}_fact_0     → 第一个事实
obs_{id}_fact_1     → 第二个事实
...
```

通过chroma-mcp进程进行通信（stdio协议）。

### 4.4 混合搜索架构（3层工作流）

Claude-Mem通过**4个MCP工具**实现高效的Token节省搜索：

```
Layer 1: search (获取紧凑索引)
    ↓ ~50-100 tokens/result
Layer 2: timeline (获取时间线上下文)
    ↓
Layer 3: get_observations (仅获取过滤后ID的完整详情)
    ↓ ~500-1,000 tokens/result
```

**Token节省效果**：通过先过滤再获取详情，节省约10倍Token消耗。

### 4.5 去重机制

```typescript
SHA256(memory_session_id + title + narrative).substring(0, 16) → content_hash
// 30秒窗口内相同hash → 返回现有ID（不插入）
```

## 五、两种Session ID的设计

Claude-Mem巧妙地设计了两种会话ID：

- **contentSessionId**：来自Claude Code，会话期间不变
- **memorySessionId**：来自SDK Agent，每次Worker重启时变化

这种设计解决了Worker重启时的会话连续性问题，通过SessionStore处理转换，确保外键约束的完整性。

## 六、应用场景

### 6.1 长期项目开发

```bash
# 第1天：初始化项目
Claude: 理解项目结构，建立技术栈认知
# 存储：项目架构、技术选型、代码规范

# 第30天：继续开发
Claude: 自动回忆"这个项目使用React 18 + TypeScript，
        状态管理用Zustand，API层用React Query..."
```

### 6.2 Bug追踪与修复

```bash
# 发现Bug
存储：bugfix - "认证token过期处理逻辑错误"

# 3周后类似问题
搜索：authentication bug
返回：#123 - 认证token过期处理逻辑错误 (详情)
```

### 6.3 跨会话知识积累

- 技术决策的上下文和原因
- 踩过的坑和解决方案
- 代码审查发现的模式
- 性能优化的经验

## 七、与其他方案对比

| 特性 | Claude-Mem | CLAUDE.md文件 | 手动笔记 |
|------|------------|---------------|----------|
| 自动化程度 | ✅ 全自动 | ❌ 手动维护 | ❌ 手动维护 |
| 语义搜索 | ✅ ChromaDB | ❌ 无 | ❌ 无 |
| Token效率 | ✅ 10x节省 | ❌ 全量加载 | ❌ 全量加载 |
| 跨会话持久化 | ✅ | ⚠️ 需手动更新 | ⚠️ 需手动更新 |
| 多Agent支持 | ✅ | ❌ 仅Claude | ❌ 无 |

## 八、关键设计哲学

### 8.1 渐进式披露（Progressive Disclosure）

不要一次性加载所有记忆，而是分层加载：
1. 先看索引（50 tokens）
2. 再看时间线（100 tokens）
3. 最后获取详情（500 tokens）

### 8.2 永不阻塞原则

无论Worker发生什么问题，都不能影响用户的Claude Code使用体验。这是核心设计约束。

### 8.3 二进制解析器

解析器只有两种状态：`{ valid: true, observations, summary }` 或 `{ valid: false }`。不可解析的响应保持队列不变，会话迭代器继续处理。

## 九、总结

Claude-Mem通过**Hook生命周期 + Worker守护进程 + SQLite/ChromaDB双存储 + 渐进式搜索**的架构，优雅地解决了AI Agent的上下文持久化问题。其核心设计亮点包括：

1. **全自动Hook机制**：无需手动干预
2. **异步队列处理**：保证数据一致性
3. **优雅降级**：永不阻塞用户
4. **混合搜索**：兼顾效率与准确性
5. **Token优化**：10倍节省成本

对于需要长期维护的项目，Claude-Mem提供了一种**让AI Agent真正"记住"过去**的解决方案。

---

**参考资源**：
- GitHub仓库：https://github.com/thedotmack/claude-mem
- 官方文档：https://docs.claude-mem.ai
- 架构概览：https://docs.claude-mem.ai/architecture/overview
