---
title: "Claude-Mem系列(5)：异步队列与容错机制 - 永不阻塞的后台处理"
description: "深入解析Claude-Mem的异步队列处理机制、PendingMessageStore、Generator重启策略、优雅降级设计，以及如何保证Agent永不被阻塞"
date: 2026-05-12
author: RiceBall-15
category: agentMemory
subCategory: agent-memory
tags:
  - AI Agent
  - 异步队列
  - 容错机制
  - Claude-Mem
  - 优雅降级
draft: false
---

# Claude-Mem系列(5)：异步队列与容错机制

> **前置阅读**：[Claude-Mem系列(4)：搜索架构 - 渐进式披露与Token优化](./2026-05-12-claude-mem-search-architecture.md)

## 一、为什么需要异步处理？

### 1.1 同步处理的问题

```
场景：Agent执行文件编辑

同步方式：
  工具调用 → 等待处理完成 → 返回结果
                ↓
            可能耗时500ms-2s
                ↓
            Agent被阻塞 😤

如果每个工具调用都这样：
  10个工具调用 × 1秒 = 10秒额外等待
```

### 1.2 异步处理的优势

```
异步方式：
  工具调用 → 入队 → 立即返回
                ↓
            后台异步处理
                ↓
            Agent继续工作 😊

总等待时间：接近0
```

### 1.3 Claude-Mem的设计原则

> **核心原则**：Worker不可用时**永远不会阻塞用户的Claude Code会话**。

这意味着：
1. Hook执行必须快速返回
2. 失败不应影响Agent
3. 数据最终一致性（非实时）

## 二、PendingMessageStore架构

### 2.1 数据库表结构

```sql
CREATE TABLE pending_messages (
  id TEXT PRIMARY KEY,
  
  -- 消息类型
  message_type TEXT NOT NULL 
    CHECK(message_type IN ('observation', 'summary', 'session_init', 'session_complete')),
  
  -- 会话关联
  memory_session_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  
  -- 消息内容
  payload TEXT NOT NULL,  -- JSON格式
  
  -- 处理状态
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
  
  -- 重试机制
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 3,
  
  -- 时间戳
  created_at_epoch INTEGER NOT NULL,
  updated_at_epoch INTEGER NOT NULL,
  processed_at_epoch INTEGER,
  
  -- 错误信息
  error_message TEXT,
  
  FOREIGN KEY(memory_session_id) REFERENCES server_sessions(memory_session_id)
);

-- 索引
CREATE INDEX idx_pending_messages_status ON pending_messages(status);
CREATE INDEX idx_pending_messages_session ON pending_messages(memory_session_id);
CREATE INDEX idx_pending_messages_created ON pending_messages(created_at_epoch);
```

### 2.2 入队操作

```typescript
// PostToolUse Hook调用
async function enqueueObservation(context: HookContext): Promise<void> {
  const observation = {
    memorySessionId: context.memorySessionId,
    projectId: context.projectId,
    type: context.toolName,
    title: generateTitle(context),
    narrative: generateNarrative(context),
    facts: extractFacts(context),
    filesModified: context.filesModified
  };
  
  // 插入pending_messages表
  await db.prepare(`
    INSERT INTO pending_messages 
    (id, message_type, memory_session_id, project_id, payload, status, created_at_epoch, updated_at_epoch)
    VALUES (?, 'observation', ?, ?, ?, 'pending', ?, ?)
  `).run(
    generateId(),
    observation.memorySessionId,
    observation.projectId,
    JSON.stringify(observation),
    Date.now(),
    Date.now()
  );
}
```

**关键点**：
- 插入即返回，不等待处理
- 使用事务保证原子性
- payload为完整的消息内容

### 2.3 出队操作

```typescript
// Worker后台处理
class MessageProcessor {
  async processNextBatch(): Promise<void> {
    // 获取待处理消息
    const messages = await db.prepare(`
      SELECT * FROM pending_messages 
      WHERE status = 'pending' 
      ORDER BY created_at_epoch ASC 
      LIMIT 10
    `).all();
    
    if (messages.length === 0) return;
    
    // 标记为处理中
    const ids = messages.map(m => m.id);
    await db.prepare(`
      UPDATE pending_messages 
      SET status = 'processing', updated_at_epoch = ?
      WHERE id IN (${ids.map(() => '?').join(',')})
    `).run(Date.now(), ...ids);
    
    // 处理每个消息
    for (const message of messages) {
      try {
        await this.processMessage(message);
        
        // 标记为完成
        await db.prepare(`
          UPDATE pending_messages 
          SET status = 'completed', processed_at_epoch = ?, updated_at_epoch = ?
          WHERE id = ?
        `).run(Date.now(), Date.now(), message.id);
        
      } catch (error) {
        await this.handleFailure(message, error);
      }
    }
  }
  
  async processMessage(message: PendingMessage): Promise<void> {
    const payload = JSON.parse(message.payload);
    
    switch (message.message_type) {
      case 'observation':
        await this.processObservation(payload);
        break;
      case 'summary':
        await this.processSummary(payload);
        break;
      case 'session_init':
        await this.processSessionInit(payload);
        break;
      case 'session_complete':
        await this.processSessionComplete(payload);
        break;
    }
  }
}
```

### 2.4 失败处理

```typescript
async handleFailure(message: PendingMessage, error: Error): Promise<void> {
  const newRetryCount = message.retry_count + 1;
  
  if (newRetryCount >= message.max_retries) {
    // 超过重试次数，标记为失败
    await db.prepare(`
      UPDATE pending_messages 
      SET status = 'failed', 
          retry_count = ?,
          error_message = ?,
          updated_at_epoch = ?
      WHERE id = ?
    `).run(newRetryCount, error.message, Date.now(), message.id);
    
    // 记录失败事件
    await logEvent({
      type: 'message_processing_failed',
      messageId: message.id,
      error: error.message
    });
    
  } else {
    // 重新入队，等待重试
    await db.prepare(`
      UPDATE pending_messages 
      SET status = 'pending',
          retry_count = ?,
          error_message = ?,
          updated_at_epoch = ?
      WHERE id = ?
    `).run(newRetryCount, error.message, Date.now(), message.id);
  }
}
```

## 三、Generator重启机制

### 3.1 什么是Generator？

Claude-Mem使用Generator模式处理流式响应：

```typescript
// SDK Agent的Generator
async function* processSession(memorySessionId: string) {
  // 持续监听新消息
  while (true) {
    const messages = await getPendingMessages(memorySessionId);
    
    if (messages.length === 0) {
      // 没有新消息，等待
      await sleep(1000);
      continue;
    }
    
    // 处理消息并yield结果
    for (const message of messages) {
      const result = await processMessage(message);
      yield result;
    }
  }
}
```

### 3.2 Generator崩溃场景

```
场景1：网络超时
  Generator正在处理消息
  → 调用OpenAI API超时
  → Generator抛出异常
  → 需要重启

场景2：内存不足
  Generator处理大量消息
  → 内存占用过高
  → 进程被OOM Killer终止
  → 需要重启

场景3：数据库锁定
  Generator写入SQLite
  → 数据库被锁定
  → 抛出SQLITE_BUSY错误
  → 需要重启
```

### 3.3 重启策略：指数退避

```typescript
class GeneratorManager {
  private restartCount = 0;
  private maxConsecutiveRestarts = 3;
  private generator: AsyncGenerator | null = null;
  
  async start(memorySessionId: string): Promise<void> {
    while (true) {
      try {
        this.generator = processSession(memorySessionId);
        
        // 消费Generator
        for await (const result of this.generator) {
          // 处理结果
          await this.handleResult(result);
          
          // 成功处理，重置计数器
          this.restartCount = 0;
        }
        
        // Generator正常结束
        break;
        
      } catch (error) {
        this.restartCount++;
        
        if (this.restartCount > this.maxConsecutiveRestarts) {
          // 连续重启过多，停止
          console.error('Too many consecutive restarts, stopping');
          break;
        }
        
        // 指数退避：1s, 2s, 4s, 8s...
        const delay = Math.pow(2, this.restartCount - 1) * 1000;
        console.log(`Generator crashed, restarting in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }
}
```

**关键设计**：

```
成功处理消息 → restartCount = 0
Generator崩溃 → restartCount++
               → 等待 2^(n-1) 秒
               → 重启Generator

连续重启 > 3次 → 停止，让迭代器结束
```

### 3.4 为什么这样设计？

```
场景：临时网络问题

传统方式：
  Generator崩溃 → 立即重启 → 又崩溃 → 立即重启...
  → 快速消耗资源，问题未解决

指数退避：
  Generator崩溃 → 等1秒 → 重启
  又崩溃 → 等2秒 → 重启
  又崩溃 → 等4秒 → 重启
  网络恢复 → 成功 → 重置计数器
  
  → 给系统恢复的时间
  → 避免资源浪费
```

### 3.5 消息不丢失保证

```
Generator崩溃时：
  1. 正在处理的消息：
     - 状态是'processing'
     - 重启后会重新获取（因为超时机制）
  
  2. 队列中的消息：
     - 状态是'pending'
     - 重启后正常处理
  
  3. 已处理的消息：
     - 状态是'completed'
     - 不会被重复处理
```

## 四、优雅降级设计

### 4.1 错误分类

```typescript
// 错误类型定义
enum ErrorType {
  TRANSPORT = 'transport',     // 传输错误
  CLIENT = 'client',           // 客户端错误
  SERVER = 'server',           // 服务端错误
  TIMEOUT = 'timeout',         // 超时
  UNKNOWN = 'unknown'          // 未知错误
}

// 错误分类逻辑
function classifyError(error: Error): ErrorType {
  if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
    return ErrorType.TRANSPORT;
  }
  
  if (error.statusCode >= 400 && error.statusCode < 500) {
    return ErrorType.CLIENT;
  }
  
  if (error.statusCode >= 500) {
    return ErrorType.SERVER;
  }
  
  if (error.name === 'TimeoutError') {
    return ErrorType.TIMEOUT;
  }
  
  return ErrorType.UNKNOWN;
}
```

### 4.2 降级策略

```typescript
// Hook中的错误处理
async function hookWithErrorHandling(
  hookName: string,
  handler: () => Promise<HookResponse>
): Promise<HookResponse> {
  try {
    return await handler();
    
  } catch (error) {
    const errorType = classifyError(error);
    
    switch (errorType) {
      case ErrorType.TRANSPORT:
        // Worker不可用
        log.warn(`${hookName}: Worker unavailable, continuing without memory`);
        return { continue: true };  // 不阻塞
        
      case ErrorType.TIMEOUT:
        // 超时
        log.warn(`${hookName}: Timeout, continuing without memory`);
        return { continue: true };  // 不阻塞
        
      case ErrorType.SERVER:
        // 服务端错误
        log.error(`${hookName}: Server error`, error);
        return { continue: true };  // 不阻塞
        
      case ErrorType.CLIENT:
        // 客户端错误：可能是bug
        log.error(`${hookName}: Client error`, error);
        return { 
          continue: false, 
          error: `Hook error: ${error.message}` 
        };  // 阻塞，提示修复
        
      default:
        // 未知错误：保守处理
        log.error(`${hookName}: Unknown error`, error);
        return { continue: true };
    }
  }
}
```

### 4.3 降级矩阵

| 错误类型 | 是否阻塞Agent | 原因 |
|----------|---------------|------|
| 传输错误 | ❌ 不阻塞 | Worker不可用，但Agent仍可工作 |
| 超时 | ❌ 不阻塞 | 网络问题，不应影响用户体验 |
| 服务端错误 | ❌ 不阻塞 | 服务端问题，用户无法解决 |
| 客户端错误 | ✅ 阻塞 | 可能是配置或代码bug，需要修复 |
| 未知错误 | ❌ 不阻塞 | 保守处理，避免意外阻塞 |

### 4.4 为什么"永不阻塞"？

```
场景：Worker服务崩溃

如果阻塞：
  用户：输入提示
  Claude Code：等待Hook响应... (卡住)
  用户：😭 什么都做不了，只能重启Claude Code
  
如果优雅降级：
  用户：输入提示
  Claude Code：Hook失败，继续执行（无记忆注入）
  用户：😊 正常工作，只是这次没有历史上下文
  用户：可以稍后手动重启Worker
```

**核心洞察**：记忆是增强，不是必需。没有记忆，Agent仍然可以工作。

## 五、Session生命周期管理

### 5.1 状态机

```
┌─────────────────────────────────────────────────────────────────┐
│                    Session状态机                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   [created] ──→ [active] ──→ [completed]                        │
│                  │                                               │
│                  └──→ [failed]                                   │
│                                                                   │
│   状态转换：                                                      │
│   - created → active: UserPromptSubmit Hook                     │
│   - active → completed: SessionEnd Hook (成功)                  │
│   - active → failed: SessionEnd Hook (失败) 或 超时              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 超时处理

```typescript
// 会话超时检测
class SessionTimeoutChecker {
  private checkInterval = 60000;  // 1分钟
  private sessionTimeout = 3600000;  // 1小时
  
  async check(): Promise<void> {
    const cutoffEpoch = Date.now() - this.sessionTimeout;
    
    // 查找超时的活跃会话
    const timeoutSessions = await db.prepare(`
      SELECT memory_session_id FROM server_sessions
      WHERE status = 'active'
      AND updated_at_epoch < ?
    `).all(cutoffEpoch);
    
    // 标记为失败
    for (const session of timeoutSessions) {
      await db.prepare(`
        UPDATE server_sessions
        SET status = 'failed', completed_at_epoch = ?, updated_at_epoch = ?
        WHERE memory_session_id = ?
      `).run(Date.now(), Date.now(), session.memory_session_id);
      
      log.warn(`Session ${session.memory_session_id} timed out`);
    }
  }
}
```

### 5.3 队列排空

```typescript
// Session结束时排空队列
async function drainQueue(memorySessionId: string): Promise<void> {
  const maxWaitTime = 30000;  // 最多等待30秒
  const startTime = Date.now();
  
  while (true) {
    // 检查是否还有待处理消息
    const pendingCount = await db.prepare(`
      SELECT COUNT(*) as count FROM pending_messages
      WHERE memory_session_id = ?
      AND status IN ('pending', 'processing')
    `).get(memorySessionId);
    
    if (pendingCount.count === 0) {
      // 队列已空
      break;
    }
    
    if (Date.now() - startTime > maxWaitTime) {
      // 超时，强制完成
      log.warn(`Queue drain timeout for session ${memorySessionId}`);
      break;
    }
    
    // 等待一下再检查
    await sleep(1000);
  }
}
```

## 六、监控与告警

### 6.1 队列监控

```typescript
// 队列健康检查
interface QueueHealth {
  pending: number;
  processing: number;
  failed: number;
  oldestPendingAge: number;  // 最老的pending消息的年龄（秒）
}

async function getQueueHealth(): Promise<QueueHealth> {
  const stats = await db.prepare(`
    SELECT 
      status,
      COUNT(*) as count,
      MIN(created_at_epoch) as oldest
    FROM pending_messages
    WHERE status IN ('pending', 'processing', 'failed')
    GROUP BY status
  `).all();
  
  const result: QueueHealth = {
    pending: 0,
    processing: 0,
    failed: 0,
    oldestPendingAge: 0
  };
  
  for (const stat of stats) {
    result[stat.status] = stat.count;
    if (stat.status === 'pending' && stat.oldest) {
      result.oldestPendingAge = (Date.now() - stat.oldest) / 1000;
    }
  }
  
  return result;
}
```

### 6.2 告警规则

```typescript
// 告警检查
async function checkAlerts(): Promise<Alert[]> {
  const alerts: Alert[] = [];
  const health = await getQueueHealth();
  
  // 规则1：pending消息过多
  if (health.pending > 100) {
    alerts.push({
      level: 'warning',
      message: `Pending queue has ${health.pending} messages`
    });
  }
  
  // 规则2：failed消息过多
  if (health.failed > 10) {
    alerts.push({
      level: 'error',
      message: `${health.failed} messages failed to process`
    });
  }
  
  // 规则3：最老消息过旧
  if (health.oldestPendingAge > 300) {  // 5分钟
    alerts.push({
      level: 'warning',
      message: `Oldest pending message is ${health.oldestPendingAge}s old`
    });
  }
  
  return alerts;
}
```

## 七、性能优化

### 7.1 批量处理

```typescript
// 批量处理消息
async function processBatch(messages: PendingMessage[]): Promise<void> {
  // 按类型分组
  const grouped = groupBy(messages, 'message_type');
  
  // 批量处理每种类型
  for (const [type, typeMessages] of Object.entries(grouped)) {
    switch (type) {
      case 'observation':
        await processObservationsBatch(typeMessages);
        break;
      case 'summary':
        await processSummariesBatch(typeMessages);
        break;
    }
  }
}

// 批量插入memory_items
async function processObservationsBatch(messages: PendingMessage[]): Promise<void> {
  const observations = messages.map(m => JSON.parse(m.payload));
  
  const insertMany = db.transaction(() => {
    const stmt = db.prepare(`
      INSERT INTO memory_items 
      (id, project_id, server_session_id, kind, type, title, narrative, facts, concepts, created_at_epoch)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    
    for (const obs of observations) {
      stmt.run(
        generateId(), obs.projectId, obs.sessionId,
        'observation', obs.type, obs.title, obs.narrative,
        JSON.stringify(obs.facts), JSON.stringify(obs.concepts),
        Date.now()
      );
    }
  });
  
  insertMany();
}
```

### 7.2 连接池

```typescript
// SQLite连接池
class SQLitePool {
  private pool: Database[] = [];
  private maxConnections = 5;
  
  async acquire(): Promise<Database> {
    if (this.pool.length > 0) {
      return this.pool.pop()!;
    }
    
    if (this.pool.length < this.maxConnections) {
      return createConnection();
    }
    
    // 等待可用连接
    return new Promise((resolve) => {
      const check = () => {
        if (this.pool.length > 0) {
          resolve(this.pool.pop()!);
        } else {
          setTimeout(check, 10);
        }
      };
      check();
    });
  }
  
  release(conn: Database): void {
    this.pool.push(conn);
  }
}
```

## 八、总结

Claude-Mem的异步队列与容错机制确保了：

1. **永不阻塞Agent**：Hook快速返回，后台异步处理
2. **消息不丢失**：pending_messages表保证持久化
3. **优雅降级**：错误分类处理，传输错误不阻塞
4. **自动恢复**：Generator重启机制，指数退避
5. **最终一致性**：队列排空保证数据完整

**关键设计原则**：
- 记忆是增强，不是必需
- 失败不应传播到用户
- 最终一致性优于实时一致性

---

**下一篇**：[Claude-Mem系列(6)：应用场景与最佳实践](./2026-05-12-claude-mem-best-practices.md)
