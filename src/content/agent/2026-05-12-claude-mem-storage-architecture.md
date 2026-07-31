---
title: "Claude-Mem系列(3)：存储架构 - SQLite + ChromaDB双存储详解"
description: "深入解析Claude-Mem的双存储架构设计，包括SQLite关系型存储Schema、ChromaDB向量存储、FTS5全文搜索，以及双存储同步机制"
date: 2026-05-12
author: RiceBall-15
category: agentMemory
subCategory: agent-memory
tags:
  - AI Agent
  - SQLite
  - ChromaDB
  - 向量搜索
  - 存储架构
draft: false
series: claude-mem-deep
seriesOrder: 3
---

# Claude-Mem系列(3)：存储架构详解

> **前置阅读**：[Claude-Mem系列(2)：Hook生命周期机制详解](./2026-05-12-claude-mem-hook-lifecycle.md)

## 一、为什么需要双存储？

Claude-Mem面临两种完全不同的查询需求：

| 需求 | 示例 | 适合的存储 |
|------|------|------------|
| **结构化查询** | "获取最近10条observation" | SQLite |
| **全文搜索** | "搜索包含'authentication'的记录" | SQLite FTS5 |
| **语义搜索** | "查找与token过期相关的问题" | ChromaDB向量搜索 |

单一存储无法同时满足这三种需求，因此采用**双存储架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude-Mem存储层                         │
├─────────────────────────┬───────────────────────────────────┤
│       SQLite            │           ChromaDB                │
├─────────────────────────┼───────────────────────────────────┤
│ • 结构化数据             │ • 向量嵌入                        │
│ • 关系型查询             │ • 语义相似度搜索                  │
│ • FTS5全文搜索           │ • 多语言支持                      │
│ • 事务支持               │ • 近似最近邻(ANN)                 │
└─────────────────────────┴───────────────────────────────────┘
```

## 二、SQLite存储架构

### 2.1 核心表结构

#### projects（项目表）

```sql
CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT UNIQUE,
  root_path TEXT UNIQUE,              -- 项目根路径，用于关联
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at_epoch INTEGER NOT NULL,
  updated_at_epoch INTEGER NOT NULL
);

-- 索引
CREATE INDEX idx_projects_root_path ON projects(root_path);
```

**用途**：管理多个项目，每个项目有独立的记忆空间。

#### server_sessions（会话表）

```sql
CREATE TABLE server_sessions (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  content_session_id TEXT,            -- Claude Code的会话ID
  memory_session_id TEXT,             -- SDK Agent的会话ID
  platform_source TEXT NOT NULL DEFAULT 'claude',
  title TEXT,
  status TEXT NOT NULL DEFAULT 'active' 
    CHECK(status IN ('active', 'completed', 'failed')),
  metadata TEXT NOT NULL DEFAULT '{}',
  started_at_epoch INTEGER NOT NULL,
  completed_at_epoch INTEGER,
  updated_at_epoch INTEGER NOT NULL,
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- 索引
CREATE INDEX idx_server_sessions_project ON server_sessions(project_id);
CREATE INDEX idx_server_sessions_content ON server_sessions(content_session_id);
CREATE INDEX idx_server_sessions_memory ON server_sessions(memory_session_id);
CREATE INDEX idx_server_sessions_status ON server_sessions(status);
```

**关键设计**：两种Session ID

```
contentSessionId: 来自Claude Code，会话期间不变
memorySessionId:  来自SDK Agent，每次Worker重启变化

为什么要两种ID？
- Claude Code的ID用于关联用户视角
- SDK Agent的ID用于处理Worker重启场景
```

#### memory_items（记忆条目表）- 核心表

```sql
CREATE TABLE memory_items (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  server_session_id TEXT,
  legacy_observation_id INTEGER,      -- 兼容旧版本
  
  -- 记忆类型
  kind TEXT NOT NULL 
    CHECK(kind IN ('observation', 'summary', 'prompt', 'manual')),
  type TEXT NOT NULL,                  -- 细分类型
  
  -- 内容字段
  title TEXT,
  subtitle TEXT,
  text TEXT,
  narrative TEXT,                      -- 详细叙述
  
  -- 结构化数据
  facts TEXT NOT NULL DEFAULT '[]',    -- JSON数组：提取的事实
  concepts TEXT NOT NULL DEFAULT '[]', -- JSON数组：相关概念
  
  -- 文件关联
  files_read TEXT NOT NULL DEFAULT '[]',
  files_modified TEXT NOT NULL DEFAULT '[]',
  
  -- 元数据
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at_epoch INTEGER NOT NULL,
  updated_at_epoch INTEGER NOT NULL,
  
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
  FOREIGN KEY(server_session_id) REFERENCES server_sessions(id) ON DELETE SET NULL
);

-- 唯一索引（兼容旧版本）
CREATE UNIQUE INDEX ux_memory_items_legacy_observation
  ON memory_items(legacy_observation_id)
  WHERE legacy_observation_id IS NOT NULL;

-- 查询优化索引
CREATE INDEX idx_memory_items_project_time ON memory_items(project_id, created_at_epoch DESC);
CREATE INDEX idx_memory_items_session_time ON memory_items(server_session_id, created_at_epoch DESC);
CREATE INDEX idx_memory_items_kind_type ON memory_items(kind, type);
```

**kind vs type的区别**：

```typescript
// kind: 大类
type Kind = 'observation' | 'summary' | 'prompt' | 'manual';

// type: 细分类型
const typeMapping = {
  observation: ['file_read', 'file_edit', 'command', 'search', 'api_call'],
  summary: ['session_summary', 'daily_summary', 'weekly_summary'],
  prompt: ['user_prompt', 'system_prompt'],
  manual: ['note', 'decision', 'lesson']
};
```

#### memory_sources（记忆来源表）

```sql
CREATE TABLE memory_sources (
  id TEXT PRIMARY KEY,
  memory_item_id TEXT NOT NULL,
  source_type TEXT NOT NULL 
    CHECK(source_type IN ('observation', 'session_summary', 'user_prompt', 'manual', 'import')),
  legacy_table TEXT,
  legacy_id INTEGER,
  source_uri TEXT,
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at_epoch INTEGER NOT NULL,
  FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE
);

CREATE INDEX idx_memory_sources_item ON memory_sources(memory_item_id);
CREATE INDEX idx_memory_sources_legacy ON memory_sources(legacy_table, legacy_id);
```

**用途**：追踪记忆的来源，支持数据溯源。

#### agent_events（事件表）

```sql
CREATE TABLE agent_events (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  server_session_id TEXT,
  source_type TEXT NOT NULL 
    CHECK(source_type IN ('hook', 'worker', 'provider', 'server', 'api')),
  event_type TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}',
  content_session_id TEXT,
  memory_session_id TEXT,
  occurred_at_epoch INTEGER NOT NULL,
  created_at_epoch INTEGER NOT NULL,
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
  FOREIGN KEY(server_session_id) REFERENCES server_sessions(id) ON DELETE SET NULL
);

CREATE INDEX idx_agent_events_project_time ON agent_events(project_id, occurred_at_epoch DESC);
CREATE INDEX idx_agent_events_session_time ON agent_events(server_session_id, occurred_at_epoch DESC);
CREATE INDEX idx_agent_events_type ON agent_events(event_type);
```

**用途**：审计日志、调试追踪、行为分析。

### 2.2 FTS5全文搜索

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

**FTS5配置解析**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `tokenize` | `porter unicode61` | Porter词干提取 + Unicode支持 |
| `porter` | - | 词干提取：running → run |
| `unicode61` | - | Unicode 6.1标准分词 |

**搜索示例**：

```sql
-- 基础搜索
SELECT * FROM memory_items_fts 
WHERE memory_items_fts MATCH 'authentication token';

-- 带权重的搜索
SELECT * FROM memory_items_fts 
WHERE memory_items_fts MATCH 'title:authentication OR narrative:token';

-- 短语搜索
SELECT * FROM memory_items_fts 
WHERE memory_items_fts MATCH '"token expiry"';
```

**自动同步机制**：

```typescript
// 启动时检查FTS索引是否与memory_items同步
const memoryItemCount = db.prepare('SELECT COUNT(*) AS count FROM memory_items').get();
const ftsItemCount = db.prepare('SELECT COUNT(*) AS count FROM memory_items_fts').get();

if (memoryItemCount.count !== ftsItemCount.count) {
  // 重建FTS索引
  const rebuildFts = db.transaction(() => {
    db.run('DELETE FROM memory_items_fts');
    db.run(`
      INSERT INTO memory_items_fts (
        memory_item_id, project_id, title, subtitle, text, narrative, facts, concepts
      )
      SELECT id, project_id, title, subtitle, text, narrative, facts, concepts
      FROM memory_items
    `);
  });
  rebuildFts();
}
```

### 2.3 去重机制

```typescript
// 基于内容的去重
function generateContentHash(
  memorySessionId: string,
  title: string,
  narrative: string
): string {
  const content = `${memorySessionId}${title}${narrative}`;
  return SHA256(content).substring(0, 16);  // 取前16位
}

// 插入前检查
const existingId = db.prepare(`
  SELECT id FROM memory_items 
  WHERE content_hash = ? 
  AND created_at_epoch > ?
`).get(hash, Date.now() - 30000);  // 30秒窗口

if (existingId) {
  return existingId;  // 返回已存在的ID，不重复插入
}
```

**30秒窗口的原因**：
- 防止同一操作被多次捕获（如网络重试）
- 允许短时间内相同内容的不同实例

## 三、ChromaDB向量存储

### 3.1 为什么需要向量搜索？

**传统关键词搜索的局限**：

```
查询："解决认证问题"
关键词匹配：authentication, auth, login...

漏掉的相关内容：
- "修复了用户登录失败的bug" (没有"auth"关键词)
- "token刷新逻辑优化" (没有"认证"关键词)
```

**语义搜索的优势**：

```
查询："解决认证问题"
向量相似度匹配：
- "修复了用户登录失败的bug" ✓ (语义相关)
- "token刷新逻辑优化" ✓ (语义相关)
- "用户权限验证" ✓ (语义相关)
```

### 3.2 ChromaDB存储结构

```typescript
// 每个Observation生成多个文档
function generateDocuments(observation: Observation): Document[] {
  const documents: Document[] = [];
  
  // 主文档：narrative
  documents.push({
    id: `obs_${observation.id}_narrative`,
    text: observation.narrative,
    metadata: {
      observation_id: observation.id,
      type: observation.type,
      project_id: observation.projectId
    }
  });
  
  // 事实文档：每个fact单独存储
  observation.facts.forEach((fact, index) => {
    documents.push({
      id: `obs_${observation.id}_fact_${index}`,
      text: fact,
      metadata: {
        observation_id: observation.id,
        type: 'fact',
        project_id: observation.projectId
      }
    });
  });
  
  return documents;
}
```

**为什么要拆分存储？**

```
原始Observation：
  narrative: "修改了认证模块，发现token有效期实际为15分钟而非1小时，
              更新了过期检查逻辑，现在在请求前会验证token是否即将过期"
  facts: ["API token有效期实际为15分钟", "需要在请求前检查过期时间"]

拆分后：
  doc_1: narrative全文 (用于上下文检索)
  doc_2: "API token有效期实际为15分钟" (用于精确事实检索)
  doc_3: "需要在请求前检查过期时间" (用于精确事实检索)
```

### 3.3 嵌入生成

```typescript
// 使用OpenAI嵌入模型
const embeddingModel = 'text-embedding-3-small';
const embeddingDimension = 1536;

async function generateEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: embeddingModel,
    input: text
  });
  return response.data[0].embedding;
}

// 批量生成
async function generateEmbeddings(texts: string[]): Promise<number[][]> {
  const response = await openai.embeddings.create({
    model: embeddingModel,
    input: texts  // 支持批量
  });
  return response.data.map(item => item.embedding);
}
```

### 3.4 向量搜索实现

```python
# chroma-mcp服务
import chromadb

client = chromadb.PersistentClient(path="./chroma.sqlite3")
collection = client.get_or_create_collection(
    name="observations",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

def search(query: str, project_id: str, limit: int = 10):
    # 生成查询向量
    query_embedding = generate_embedding(query)
    
    # 向量搜索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where={"project_id": project_id}  # 过滤条件
    )
    
    return results
```

### 3.5 ChromaDB与SQLite同步

```typescript
// ChromaSync服务
class ChromaSync {
  private pendingSync: Observation[] = [];
  private syncInterval: NodeJS.Timeout;
  
  constructor() {
    // 定时同步
    this.syncInterval = setInterval(() => this.flush(), 5000);
  }
  
  // 添加待同步项
  enqueue(observation: Observation) {
    this.pendingSync.push(observation);
    
    // 达到阈值立即同步
    if (this.pendingSync.length >= 10) {
      this.flush();
    }
  }
  
  // 执行同步
  async flush() {
    if (this.pendingSync.length === 0) return;
    
    const batch = this.pendingSync.splice(0);
    
    // 生成文档
    const documents = batch.flatMap(obs => generateDocuments(obs));
    
    // 批量嵌入
    const texts = documents.map(doc => doc.text);
    const embeddings = await generateEmbeddings(texts);
    
    // 批量插入ChromaDB
    await collection.add({
      ids: documents.map(doc => doc.id),
      documents: texts,
      embeddings: embeddings,
      metadatas: documents.map(doc => doc.metadata)
    });
  }
}
```

## 四、双存储查询策略

### 4.1 查询路由

```typescript
// SearchManager中的查询路由
class SearchManager {
  async search(query: SearchQuery): Promise<SearchResult> {
    const { type, text, filters } = query;
    
    switch (type) {
      case 'structural':
        // 结构化查询：直接SQLite
        return this.sqliteSearch(filters);
        
      case 'fulltext':
        // 全文搜索：SQLite FTS5
        return this.ftsSearch(text, filters);
        
      case 'semantic':
        // 语义搜索：ChromaDB
        return this.vectorSearch(text, filters);
        
      case 'hybrid':
        // 混合搜索：两者结合
        return this.hybridSearch(text, filters);
    }
  }
  
  // 混合搜索：先向量，再FTS补充
  async hybridSearch(text: string, filters: Filters): Promise<SearchResult> {
    // 1. 向量搜索（召回）
    const vectorResults = await this.vectorSearch(text, filters);
    
    // 2. FTS搜索（补充）
    const ftsResults = await this.ftsSearch(text, filters);
    
    // 3. 合并去重
    return this.mergeResults(vectorResults, ftsResults);
  }
}
```

### 4.2 查询性能对比

| 查询类型 | 存储 | 平均耗时 | 结果质量 |
|----------|------|----------|----------|
| 结构化查询 | SQLite | <10ms | 精确 |
| 全文搜索 | SQLite FTS5 | 20-50ms | 关键词匹配 |
| 语义搜索 | ChromaDB | 50-100ms | 语义相关 |
| 混合搜索 | 两者 | 100-150ms | 最佳 |

## 五、存储优化策略

### 5.1 SQLite优化

```sql
-- WAL模式：提升并发性能
PRAGMA journal_mode=WAL;

-- 调整缓存大小
PRAGMA cache_size=-2000;  -- 2MB

-- 同步模式
PRAGMA synchronous=NORMAL;  -- 平衡性能和安全
```

### 5.2 ChromaDB优化

```python
# HNSW索引参数优化
collection = client.get_or_create_collection(
    name="observations",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,  # 构建时搜索深度
        "hnsw:M": 16,                  # 每个节点的邻居数
        "hnsw:search_ef": 128          # 搜索时搜索深度
    }
)
```

### 5.3 批量操作

```typescript
// 批量插入SQLite
const insertMany = db.transaction((items: MemoryItem[]) => {
  const stmt = db.prepare(`
    INSERT INTO memory_items (id, project_id, kind, type, title, narrative, facts, concepts, created_at_epoch)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  
  for (const item of items) {
    stmt.run(item.id, item.projectId, item.kind, item.type, 
             item.title, item.narrative, 
             JSON.stringify(item.facts), JSON.stringify(item.concepts),
             Date.now());
  }
});
```

## 六、数据生命周期

### 6.1 数据保留策略

```typescript
// 定期清理旧数据
async function cleanupOldData(retentionDays: number = 90) {
  const cutoffEpoch = Date.now() - (retentionDays * 24 * 60 * 60 * 1000);
  
  // SQLite清理
  db.prepare(`
    DELETE FROM memory_items 
    WHERE created_at_epoch < ? 
    AND kind != 'manual'  // 手动创建的不删除
  `).run(cutoffEpoch);
  
  // ChromaDB清理（通过重新同步）
  await chromaSync.fullResync();
}
```

### 6.2 数据归档

```typescript
// 归档旧会话
async function archiveOldSessions(projectId: string, days: number = 30) {
  const cutoffEpoch = Date.now() - (days * 24 * 60 * 60 * 1000);
  
  // 导出为JSON
  const sessions = db.prepare(`
    SELECT * FROM server_sessions 
    WHERE project_id = ? 
    AND started_at_epoch < ?
  `).all(projectId, cutoffEpoch);
  
  // 写入归档文件
  await writeFile(`archive_${projectId}_${Date.now()}.json`, 
                  JSON.stringify(sessions, null, 2));
  
  // 删除原始数据
  db.prepare(`
    DELETE FROM server_sessions 
    WHERE project_id = ? 
    AND started_at_epoch < ?
  `).run(projectId, cutoffEpoch);
}
```

## 七、总结

Claude-Mem的双存储架构设计体现了**用合适的工具做合适的事**的原则：

1. **SQLite**：结构化数据、关系查询、事务支持、FTS5全文搜索
2. **ChromaDB**：向量嵌入、语义搜索、近似最近邻
3. **同步机制**：确保双存储数据一致性
4. **查询路由**：根据查询类型选择最优存储

**关键洞察**：存储架构的选择应该基于**实际查询模式**，而非技术偏好。Claude-Mem通过双存储，同时满足了结构化查询和语义搜索的需求。

---

**下一篇**：[Claude-Mem系列(4)：搜索架构 - 渐进式披露与Token优化](./2026-05-12-claude-mem-search-architecture.md)
