---
title: "Claude-Mem系列(2)：Hook生命周期机制详解 - 自动捕获Agent行为的核心"
description: "深入解析Claude-Mem的5个Hook生命周期事件、触发时机、处理逻辑、超时机制，以及如何实现零干预的自动化记忆捕获"
date: 2026-05-12
author: RiceBall-15
category: agentMemory
subCategory: agent-memory
tags:
  - AI Agent
  - Hook机制
  - 生命周期
  - Claude-Mem
  - 自动化
draft: false
series: claude-mem-deep
seriesOrder: 2
---

# Claude-Mem系列(2)：Hook生命周期机制详解

> **前置阅读**：[Claude-Mem系列(1)：项目概述与核心价值](./2026-05-12-claude-mem-overview-and-value.md)

## 一、什么是Hook机制？

Hook是Claude-Mem实现**全自动记忆捕获**的核心机制。它通过Claude Code的Plugin API，在Agent生命周期的关键节点自动触发，无需用户任何干预。

### 1.1 Hook的本质

```
传统方式：
用户操作 → 手动记录 → 手动存储 → 手动检索

Hook方式：
Agent生命周期事件 → 自动触发Hook → 自动处理 → 自动存储
```

### 1.2 Claude Code的Hook事件

Claude Code提供了以下Hook事件：

| 事件 | 触发时机 | 典型用途 |
|------|----------|----------|
| **Setup** | Claude Code启动时 | 版本检查、依赖安装 |
| **SessionStart** | 新会话开始 | 初始化、上下文注入 |
| **UserPromptSubmit** | 用户提交提示 | 会话注册、语义注入 |
| **PostToolUse** | 工具调用完成后 | 捕获操作、入队处理 |
| **Stop** | Agent停止响应时 | 生成摘要 |
| **SessionEnd** | 会话结束时 | 清理、完成处理 |

## 二、Hook实现架构

### 2.1 目录结构

```
~/.claude/plugins/marketplaces/thedotmack/
├── plugin/
│   ├── hooks/
│   │   ├── setup/
│   │   │   └── version-check.js      # Setup Hook
│   │   ├── session-start/
│   │   │   └── context-inject.js     # SessionStart Hook
│   │   ├── user-prompt-submit/
│   │   │   └── session-init.js       # UserPromptSubmit Hook
│   │   ├── post-tool-use/
│   │   │   └── observation.js        # PostToolUse Hook
│   │   ├── stop/
│   │   │   └── summarize.js          # Stop Hook
│   │   └── session-end/
│   │       └── session-complete.js   # SessionEnd Hook
│   └── modes/
│       ├── code.js                   # 默认模式
│       └── code--zh.js               # 中文模式
├── src/
│   └── hooks/
│       └── hook-response.ts          # Hook响应处理
└── package.json
```

### 2.2 Hook执行模型

```typescript
// Hook执行的基本模式
export async function hookHandler(context: HookContext): Promise<HookResponse> {
  try {
    // 1. 解析输入
    const input = parseHookInput(context.stdin);
    
    // 2. 调用Worker API
    const response = await callWorkerAPI(input);
    
    // 3. 返回响应
    return {
      continue: true,        // 是否继续执行
      suppressOutput: false,  // 是否抑制输出
      hookSpecificOutput: response
    };
  } catch (error) {
    // 优雅降级：永不阻塞
    return { continue: true };
  }
}
```

## 三、逐个Hook详解

### 3.1 Setup Hook（版本检查）

**触发时机**：Claude Code启动时

**职责**：
- 检查插件版本是否匹配
- 提示用户执行修复命令（如需要）
- 安装依赖（首次运行）

**实现逻辑**：

```javascript
// plugin/hooks/setup/version-check.js
export default async function setup(context) {
  // 读取安装版本标记
  const installVersion = await readFile('.install-version');
  const currentVersion = getPackageVersion();
  
  if (installVersion !== currentVersion) {
    // 版本不匹配，提示修复
    console.error('Claude-Mem版本不匹配，请运行: npx claude-mem repair');
  }
  
  // 检查Bun是否安装
  if (!await hasCommand('bun')) {
    await installBun();
  }
  
  // 检查uv是否安装（用于ChromaDB）
  if (!await hasCommand('uv')) {
    await installUv();
  }
  
  // 非阻塞：始终返回成功
  return { continue: true };
}
```

**超时**：60秒

**关键设计**：
- 使用`.install-version`文件记录安装版本
- 检查通过后不重复安装依赖
- 即使失败也不阻塞Claude Code

### 3.2 SessionStart Hook（上下文注入）

**触发时机**：新会话开始

**职责**：
- 启动Worker服务（如未运行）
- 注入历史上下文到当前会话
- 初始化会话状态

**执行流程**：

```
SessionStart触发
    ↓
检查Worker是否运行
    ↓ (未运行)
启动Worker (端口37700 + uid%100)
    ↓
POST /api/context/semantic
    ↓
检索相关历史记忆
    ↓
格式化为上下文注入
    ↓
返回给Claude Code
```

**代码实现**：

```javascript
// plugin/hooks/session-start/context-inject.js
export default async function sessionStart(context) {
  // 1. 确保Worker运行
  await ensureWorkerRunning();
  
  // 2. 获取项目路径
  const projectPath = context.cwd;
  
  // 3. 语义检索相关记忆
  const memories = await fetch('http://localhost:37777/api/context/semantic', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      projectPath,
      query: 'project context and recent work',
      limit: 5
    })
  }).then(r => r.json());
  
  // 4. 格式化上下文
  const contextBlock = formatMemories(memories);
  
  // 5. 返回注入指令
  return {
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'SessionStart',
      additionalContext: contextBlock
    }
  };
}
```

**超时**：60秒

**注入格式**：

```markdown
## 📚 项目历史记忆

### 最近的工作
- [2026-05-10] 修复了认证token刷新逻辑 (commit: abc123)
- [2026-05-08] 优化了API响应缓存，性能提升40%

### 重要决策
- 选择Zustand而非Redux：因为项目规模小，不需要Redux的复杂性
- API层使用React Query：提供开箱即用的缓存和重试

### 踩坑记录
- ⚠️ 认证API的token有效期实际为15分钟（文档说是1小时）
```

### 3.3 UserPromptSubmit Hook（会话注册）

**触发时机**：用户提交提示时

**职责**：
- 注册会话到SQLite
- 启动SDK Agent
- 执行语义注入（更精确的上下文）

**执行流程**：

```
用户提交提示
    ↓
UserPromptSubmit Hook触发
    ↓
POST /api/sessions/init ─────────→ 注册会话
    ↓
POST /api/context/semantic ──────→ 针对当前提示检索
    ↓
启动SDK Agent（异步）
    ↓
返回注入的上下文
```

**关键代码**：

```javascript
// plugin/hooks/user-prompt-submit/session-init.js
export default async function userPromptSubmit(context) {
  const { prompt, sessionId, cwd } = context;
  
  // 1. 注册会话
  const session = await fetch('http://localhost:37777/api/sessions/init', {
    method: 'POST',
    body: JSON.stringify({
      contentSessionId: sessionId,
      projectPath: cwd,
      platformSource: 'claude'
    })
  }).then(r => r.json());
  
  // 2. 针对用户提示的语义检索
  const relevantMemories = await fetch('http://localhost:37777/api/context/semantic', {
    method: 'POST',
    body: JSON.stringify({
      query: prompt,  // 使用用户提示作为查询
      projectPath: cwd,
      limit: 3
    })
  }).then(r => r.json());
  
  // 3. 启动SDK Agent（异步，不阻塞）
  startSDKAgent(session.memorySessionId).catch(console.error);
  
  // 4. 返回上下文
  return {
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'UserPromptSubmit',
      additionalContext: formatMemories(relevantMemories)
    }
  };
}
```

**超时**：60秒

**与SessionStart的区别**：

| 维度 | SessionStart | UserPromptSubmit |
|------|--------------|------------------|
| **触发时机** | 会话开始 | 每次用户输入 |
| **检索范围** | 项目整体上下文 | 针对当前提示 |
| **检索结果** | 宽泛 | 精确 |
| **Token消耗** | 较多 | 较少 |

### 3.4 PostToolUse Hook（操作捕获）

**触发时机**：每次工具调用完成后

**职责**：
- 捕获工具使用详情
- 入队待处理消息
- 实时推送到Web UI

**这是最核心的Hook**，因为它负责捕获所有Agent行为。

**执行流程**：

```
工具调用完成（如文件编辑、命令执行）
    ↓
PostToolUse Hook触发
    ↓
POST /api/sessions/observations
    ↓
PendingMessageStore.enqueue()
    ↓ (异步)
SDKAgent处理 → 存储 → ChromaDB同步
    ↓
SSE推送到Web UI
```

**捕获的数据结构**：

```typescript
interface Observation {
  // 基本信息
  memorySessionId: string;
  type: 'file_read' | 'file_edit' | 'command' | 'search' | 'api_call';
  
  // 内容
  title: string;           // 简短标题
  narrative: string;       // 详细描述
  
  // 关联信息
  filesRead: string[];     // 读取的文件
  filesModified: string[]; // 修改的文件
  
  // 结构化数据
  facts: string[];         // 提取的事实
  concepts: string[];      // 相关概念
  
  // 元数据
  metadata: {
    toolName: string;
    toolInput: any;
    toolOutput: any;
    duration: number;
  };
}
```

**示例**：

```javascript
// 当Agent执行文件编辑时
PostToolUse({
  tool: 'Edit',
  input: {
    file_path: '/src/auth/token.ts',
    old_string: 'const TOKEN_EXPIRY = 3600;',
    new_string: 'const TOKEN_EXPIRY = 900; // 实际为15分钟'
  },
  output: 'File edited successfully'
})
    ↓
捕获为Observation:
{
  type: 'file_edit',
  title: '修改token过期时间',
  narrative: '将TOKEN_EXPIRY从3600秒(1小时)改为900秒(15分钟)，' +
             '因为API实际有效期为15分钟',
  filesModified: ['/src/auth/token.ts'],
  facts: ['API token有效期实际为15分钟'],
  concepts: ['认证', 'token', '过期时间']
}
```

**超时**：120秒

### 3.5 Stop Hook（会话摘要）

**触发时机**：Agent停止响应时

**职责**：
- 请求生成会话摘要
- 提取关键决策和发现
- 准备长期存储

**执行流程**：

```
Agent停止响应
    ↓
Stop Hook触发
    ↓
POST /api/sessions/summarize
    ↓
SDKAgent.startSession()  // 处理待处理消息
    ↓
生成会话摘要
    ↓
存储到session_summaries表
```

**摘要结构**：

```typescript
interface SessionSummary {
  memorySessionId: string;
  
  // 摘要内容
  request: string;        // 用户的主要请求
  learned: string;        // 学到的东西
  completed: string;      // 完成的工作
  
  // 关联的Observations
  observationIds: number[];
}
```

**超时**：120秒

### 3.6 SessionEnd Hook（会话完成）

**触发时机**：会话结束时

**职责**：
- 结束会话状态
- 排空待处理消息
- 清理资源

**执行流程**：

```
会话结束
    ↓
SessionEnd Hook触发
    ↓
POST /api/sessions/complete
    ↓
drain pending messages
    ↓
更新会话状态为'completed'
    ↓
清理临时资源
```

**超时**：30秒

## 四、Hook间协作机制

### 4.1 完整的会话生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Code启动                               │
│                         ↓                                       │
│                    Setup Hook                                   │
│                    (版本检查)                                    │
├─────────────────────────────────────────────────────────────────┤
│                    新会话开始                                    │
│                         ↓                                       │
│                    SessionStart Hook                            │
│                    (启动Worker + 注入项目上下文)                  │
├─────────────────────────────────────────────────────────────────┤
│                    用户输入提示                                  │
│                         ↓                                       │
│                    UserPromptSubmit Hook                        │
│                    (注册会话 + 针对性语义注入)                    │
│                         ↓                                       │
│                    Agent开始工作                                 │
│                         ↓                                       │
│                    ┌───────────────────────┐                    │
│                    │  工具调用1             │                    │
│                    │      ↓                │                    │
│                    │  PostToolUse Hook     │                    │
│                    │  (捕获操作)           │                    │
│                    └───────────────────────┘                    │
│                         ↓                                       │
│                    ┌───────────────────────┐                    │
│                    │  工具调用2             │                    │
│                    │      ↓                │                    │
│                    │  PostToolUse Hook     │                    │
│                    │  (捕获操作)           │                    │
│                    └───────────────────────┘                    │
│                         ↓                                       │
│                    ... (循环)                                    │
├─────────────────────────────────────────────────────────────────┤
│                    Agent停止响应                                 │
│                         ↓                                       │
│                    Stop Hook                                    │
│                    (生成会话摘要)                                │
├─────────────────────────────────────────────────────────────────┤
│                    会话结束                                      │
│                         ↓                                       │
│                    SessionEnd Hook                              │
│                    (完成处理 + 清理)                             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 数据流转

```
UserPromptSubmit → session_init (SQLite)
                → semantic_search (ChromaDB)
                → context_injection (返回给Claude)

PostToolUse → observation (PendingQueue)
           → SDKAgent (异步处理)
           → store_observation (SQLite)
           → chroma_sync (ChromaDB)
           → broadcast (SSE → Web UI)

Stop → summarize (SDKAgent)
    → store_summary (SQLite)

SessionEnd → complete (SQLite)
          → drain_queue (确保所有消息处理完)
```

## 五、Hook的错误处理

### 5.1 优雅降级原则

```typescript
// hook-command.ts 中的错误处理
try {
  const result = await executeHook(hookName, input);
  return result;
} catch (error) {
  if (isTransportError(error)) {
    // 传输错误：Worker不可用
    // 永不阻塞Claude Code
    log.warn('Worker unavailable, continuing without memory');
    return { continue: true };
  }
  
  if (isClientError(error)) {
    // 客户端错误：可能是bug
    // 阻塞以提示修复
    log.error('Hook client error:', error);
    return { continue: false, error: error.message };
  }
  
  // 未知错误：保守处理
  return { continue: true };
}
```

### 5.2 错误分类

| 错误类型 | 示例 | 处理策略 |
|----------|------|----------|
| **传输错误** | ECONNREFUSED, timeout, 5xx | exit 0，不阻塞 |
| **客户端错误** | 4xx, TypeError, ReferenceError | exit 2，阻塞修复 |
| **超时** | Hook执行超过限制 | exit 0，不阻塞 |

### 5.3 为什么"永不阻塞"？

```
场景：Worker服务崩溃

如果阻塞：
  用户：输入提示
  Claude Code：等待Hook响应...
  用户：😭 卡住了，什么都做不了

如果不阻塞：
  用户：输入提示
  Claude Code：Hook失败，继续执行（无记忆注入）
  用户：😊 正常工作，只是这次没有历史上下文
```

**核心原则**：记忆是增强，不是必需。没有记忆，Agent仍然可以工作。

## 六、Hook性能优化

### 6.1 超时设置

| Hook | 超时 | 原因 |
|------|------|------|
| Setup | 60s | 可能需要安装依赖 |
| SessionStart | 60s | 可能启动Worker |
| UserPromptSubmit | 60s | 语义检索可能较慢 |
| PostToolUse | 120s | 捕获复杂操作 |
| Stop | 120s | 生成摘要需要时间 |
| SessionEnd | 30s | 主要是清理操作 |

### 6.2 异步处理

```javascript
// PostToolUse中的异步处理
export default async function postToolUse(context) {
  // 同步：快速入队
  await enqueueObservation(context);
  
  // 异步：不阻塞Hook返回
  processObservationAsync().catch(console.error);
  
  // 立即返回
  return { continue: true };
}
```

### 6.3 缓存策略

```javascript
// SessionStart中的Worker启动缓存
const workerCache = {
  pid: null,
  lastCheck: 0,
  checkInterval: 5000  // 5秒内不重复检查
};

async function ensureWorkerRunning() {
  const now = Date.now();
  
  if (workerCache.pid && (now - workerCache.lastCheck) < workerCache.checkInterval) {
    return;  // 缓存有效，跳过检查
  }
  
  // 实际检查
  workerCache.pid = await startWorker();
  workerCache.lastCheck = now;
}
```

## 七、总结

Hook机制是Claude-Mem实现**全自动记忆捕获**的核心：

1. **5个Hook事件**覆盖Agent完整生命周期
2. **非侵入式设计**：不修改Agent核心逻辑
3. **优雅降级**：Hook失败不影响Agent使用
4. **异步处理**：快速响应，后台处理
5. **数据流转**：Hook → Queue → Agent → Storage

**关键洞察**：Hook机制的本质是**事件驱动架构**在AI Agent领域的应用。通过监听生命周期事件，实现关注点分离和自动化处理。

---

**下一篇**：[Claude-Mem系列(3)：存储架构 - SQLite + ChromaDB双存储详解](./2026-05-12-claude-mem-storage-architecture.md)
