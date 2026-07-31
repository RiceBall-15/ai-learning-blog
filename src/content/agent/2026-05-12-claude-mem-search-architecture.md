---
title: "Claude-Mem系列(4)：搜索架构 - 渐进式披露与Token优化策略"
description: "深入解析Claude-Mem的搜索架构设计，包括3层渐进式披露机制、MCP工具实现、Token优化策略，以及如何实现10倍成本节省"
date: 2026-05-12
author: RiceBall-15
category: agentMemory
subCategory: agent-memory
tags:
  - AI Agent
  - 搜索架构
  - Token优化
  - MCP
  - 渐进式披露
draft: false
series: claude-mem-deep
seriesOrder: 4
---

# Claude-Mem系列(4)：搜索架构详解

> **前置阅读**：[Claude-Mem系列(3)：存储架构 - SQLite + ChromaDB双存储详解](./2026-05-12-claude-mem-storage-architecture.md)

## 一、搜索的核心挑战

### 1.1 Token成本问题

在AI Agent场景中，每次检索记忆都会消耗Token：

```
传统方式：全量加载记忆
  100条记忆 × 500 tokens/条 = 50,000 tokens
  成本：50,000 × $0.015/1K = $0.75/次查询

如果每个会话查询5次：$3.75/会话
```

这个成本太高了，需要优化。

### 1.2 相关性问题

不是所有记忆都与当前任务相关：

```
当前任务："修复登录bug"

相关记忆：
  ✓ "token过期处理逻辑"
  ✓ "用户认证流程"
  ✓ "session管理优化"

不相关记忆：
  ✗ "数据库索引优化"
  ✗ "前端样式调整"
  ✗ "API文档更新"
```

如何只加载相关记忆？

### 1.3 Claude-Mem的解决方案：渐进式披露

```
传统方式：
  查询 → 返回所有相关记忆(500 tokens/条) → Agent处理

渐进式披露：
  查询 → 返回索引(50 tokens/条) → Agent筛选 → 只获取详情(500 tokens/条)
```

**核心思想**：先看目录，再看内容。不要一开始就给全文。

## 二、3层搜索架构

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 1: Search (索引层)                      │
│   输入：查询文本                                                  │
│   输出：紧凑索引列表（ID + 标题 + 类型 + 时间）                   │
│   Token消耗：~50-100 tokens/结果                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 2: Timeline (时间线层)                  │
│   输入：选定的ID                                                 │
│   输出：时间线上下文（前后事件）                                  │
│   Token消耗：~100-200 tokens/结果                                │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 3: Get Observations (详情层)            │
│   输入：过滤后的ID列表                                           │
│   输出：完整详情（narrative + facts + concepts）                 │
│   Token消耗：~500-1000 tokens/结果                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Token节省效果

```
场景：搜索"认证相关问题"

传统方式：
  返回50条完整记录 × 500 tokens = 25,000 tokens

渐进式披露：
  Layer 1: 返回50条索引 × 75 tokens = 3,750 tokens
  Agent筛选出10条相关
  Layer 2: 10条时间线 × 150 tokens = 1,500 tokens
  Agent进一步筛选出5条
  Layer 3: 5条详情 × 750 tokens = 3,750 tokens
  总计：9,000 tokens

节省：25,000 - 9,000 = 16,000 tokens (64%)
```

实际上，通过智能筛选，节省可达**90%以上**。

## 三、MCP工具实现

### 3.1 MCP协议简介

MCP（Model Context Protocol）是Claude Code的工具扩展协议：

```typescript
// MCP工具定义
interface MCPTool {
  name: string;
  description: string;
  inputSchema: JSONSchema;
  handler: (input: any) => Promise<any>;
}
```

### 3.2 Search工具

```typescript
// Layer 1: 索引搜索
const searchTool: MCPTool = {
  name: 'search',
  description: '搜索记忆索引，返回紧凑的ID列表',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: '搜索查询' },
      type: { 
        type: 'string', 
        enum: ['observation', 'summary', 'prompt', 'manual'],
        description: '过滤类型'
      },
      project_id: { type: 'string', description: '项目ID' },
      date_from: { type: 'string', description: '开始日期' },
      date_to: { type: 'string', description: '结束日期' },
      limit: { type: 'number', default: 10, description: '返回数量' }
    },
    required: ['query']
  },
  handler: async (input) => {
    // 混合搜索：FTS5 + ChromaDB
    const results = await searchManager.hybridSearch({
      text: input.query,
      filters: {
        type: input.type,
        projectId: input.project_id,
        dateRange: { from: input.date_from, to: input.date_to }
      },
      limit: input.limit
    });
    
    // 返回紧凑索引
    return {
      results: results.map(r => ({
        id: r.id,
        title: r.title,
        type: r.type,
        kind: r.kind,
        created_at: r.created_at,
        relevance_score: r.score
      })),
      total: results.length
    };
  }
};
```

**返回格式**：

```json
{
  "results": [
    {
      "id": 123,
      "title": "修复token过期处理",
      "type": "bugfix",
      "kind": "observation",
      "created_at": "2026-05-10T14:30:00Z",
      "relevance_score": 0.92
    },
    {
      "id": 456,
      "title": "优化认证流程",
      "type": "optimization",
      "kind": "observation",
      "created_at": "2026-05-08T09:15:00Z",
      "relevance_score": 0.87
    }
  ],
  "total": 10
}
```

**Token消耗**：每条约75 tokens（ID + 标题 + 元数据）

### 3.3 Timeline工具

```typescript
// Layer 2: 时间线上下文
const timelineTool: MCPTool = {
  name: 'timeline',
  description: '获取指定ID的时间线上下文',
  inputSchema: {
    type: 'object',
    properties: {
      observation_id: { type: 'number', description: '观察ID' },
      context_window: { 
        type: 'number', 
        default: 3, 
        description: '前后各取几条'
      }
    },
    required: ['observation_id']
  },
  handler: async (input) => {
    // 获取目标记录
    const target = await getObservation(input.observation_id);
    
    // 获取时间线上的前后记录
    const timeline = await getTimeline({
      sessionId: target.session_id,
      timestamp: target.created_at,
      window: input.context_window
    });
    
    return {
      target: {
        id: target.id,
        title: target.title,
        created_at: target.created_at
      },
      before: timeline.before.map(t => ({
        id: t.id,
        title: t.title,
        type: t.type,
        created_at: t.created_at
      })),
      after: timeline.after.map(t => ({
        id: t.id,
        title: t.title,
        type: t.type,
        created_at: t.created_at
      }))
    };
  }
};
```

**返回格式**：

```json
{
  "target": {
    "id": 123,
    "title": "修复token过期处理",
    "created_at": "2026-05-10T14:30:00Z"
  },
  "before": [
    {
      "id": 122,
      "title": "发现token过期bug",
      "type": "bug_report",
      "created_at": "2026-05-10T14:25:00Z"
    },
    {
      "id": 121,
      "title": "用户反馈登录失败",
      "type": "user_feedback",
      "created_at": "2026-05-10T14:20:00Z"
    }
  ],
  "after": [
    {
      "id": 124,
      "title": "测试token刷新",
      "type": "test",
      "created_at": "2026-05-10T14:35:00Z"
    },
    {
      "id": 125,
      "title": "部署修复到生产",
      "type": "deployment",
      "created_at": "2026-05-10T14:40:00Z"
    }
  ]
}
```

**用途**：
- 理解事件的上下文
- 判断相关性
- 追踪问题的发展脉络

**Token消耗**：每条约150 tokens（标题 + 时间 + 类型 + 上下文）

### 3.4 Get Observations工具

```typescript
// Layer 3: 完整详情
const getObservationsTool: MCPTool = {
  name: 'get_observations',
  description: '获取指定ID的完整详情',
  inputSchema: {
    type: 'object',
    properties: {
      ids: { 
        type: 'array', 
        items: { type: 'number' },
        description: '观察ID列表'
      }
    },
    required: ['ids']
  },
  handler: async (input) => {
    // 批量获取
    const observations = await getObservationsByIds(input.ids);
    
    return {
      observations: observations.map(obs => ({
        id: obs.id,
        kind: obs.kind,
        type: obs.type,
        title: obs.title,
        narrative: obs.narrative,
        facts: JSON.parse(obs.facts),
        concepts: JSON.parse(obs.concepts),
        files_read: JSON.parse(obs.files_read),
        files_modified: JSON.parse(obs.files_modified),
        metadata: JSON.parse(obs.metadata),
        created_at: obs.created_at
      }))
    };
  }
};
```

**返回格式**：

```json
{
  "observations": [
    {
      "id": 123,
      "kind": "observation",
      "type": "bugfix",
      "title": "修复token过期处理",
      "narrative": "发现API的token有效期实际为15分钟，而非文档声称的1小时。修改了token刷新逻辑，在每次请求前检查过期时间，如果即将过期则自动刷新。",
      "facts": [
        "API token有效期实际为15分钟",
        "文档中的1小时是错误的",
        "需要在请求前检查过期时间"
      ],
      "concepts": ["认证", "token", "过期", "API", "bug修复"],
      "files_read": ["src/auth/token.ts", "docs/api.md"],
      "files_modified": ["src/auth/token.ts", "src/api/client.ts"],
      "metadata": {
        "tool": "Edit",
        "commit_hash": "abc123",
        "pr_number": 456
      },
      "created_at": "2026-05-10T14:30:00Z"
    }
  ]
}
```

**Token消耗**：每条约500-1000 tokens（完整内容）

## 四、3层工作流示例

### 4.1 完整查询流程

```
用户："帮我修复认证相关的bug"

Claude使用MCP工具：
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: search(query="authentication bug", limit=10)            │
│ 返回：10条紧凑索引 (750 tokens)                                  │
├─────────────────────────────────────────────────────────────────┤
│ Claude分析索引，发现#123和#456最相关                             │
├─────────────────────────────────────────────────────────────────┤
│ Step 2: timeline(observation_id=123, context_window=2)          │
│ 返回：#123的时间线上下文 (450 tokens)                            │
├─────────────────────────────────────────────────────────────────┤
│ Claude确认#123确实是相关问题                                     │
├─────────────────────────────────────────────────────────────────┤
│ Step 3: get_observations(ids=[123])                             │
│ 返回：#123的完整详情 (750 tokens)                                │
└─────────────────────────────────────────────────────────────────┘

总Token消耗：750 + 450 + 750 = 1,950 tokens
传统方式：10 × 500 = 5,000 tokens
节省：61%
```

### 4.2 智能筛选示例

```typescript
// Claude的决策逻辑（伪代码）
async function searchMemories(query: string) {
  // Layer 1: 获取索引
  const index = await search({ query, limit: 20 });
  
  // Claude分析：哪些真正相关？
  const relevantIds = index.results
    .filter(r => r.relevance_score > 0.8)  // 高相关度
    .filter(r => isRecentEnough(r.created_at))  // 时间相关
    .map(r => r.id)
    .slice(0, 5);  // 只取前5个
  
  if (relevantIds.length > 3) {
    // 如果候选较多，先用timeline筛选
    const timelines = await Promise.all(
      relevantIds.map(id => timeline({ observation_id: id }))
    );
    
    // 基于时间线进一步筛选
    const finalIds = timelines
      .filter(t => hasRelevantContext(t))
      .map(t => t.target.id);
    
    // Layer 3: 获取最终详情
    return get_observations({ ids: finalIds });
  }
  
  // 候选较少，直接获取详情
  return get_observations({ ids: relevantIds });
}
```

## 五、搜索性能优化

### 5.1 缓存策略

```typescript
// 搜索结果缓存
class SearchCache {
  private cache = new Map<string, { result: any; timestamp: number }>();
  private ttl = 60000;  // 1分钟
  
  get(query: string): any | null {
    const entry = this.cache.get(query);
    if (!entry) return null;
    
    if (Date.now() - entry.timestamp > this.ttl) {
      this.cache.delete(query);
      return null;
    }
    
    return entry.result;
  }
  
  set(query: string, result: any): void {
    this.cache.set(query, { result, timestamp: Date.now() });
  }
}
```

### 5.2 预取策略

```typescript
// 当用户开始输入时预取
async function prefetchOnTyping(partialQuery: string) {
  if (partialQuery.length < 3) return;  // 太短不预取
  
  // 异步预取，不阻塞
  searchManager.prefetch({
    query: partialQuery,
    limit: 5
  }).catch(console.error);
}
```

### 5.3 批量操作

```typescript
// 批量获取observations
const getObservationsTool = {
  handler: async (input: { ids: number[] }) => {
    // 批量查询，避免N+1问题
    const placeholders = input.ids.map(() => '?').join(',');
    const observations = db.prepare(`
      SELECT * FROM memory_items 
      WHERE id IN (${placeholders})
    `).all(...input.ids);
    
    return { observations };
  }
};
```

## 六、搜索质量优化

### 6.1 混合搜索权重

```typescript
// 混合搜索：结合FTS5和ChromaDB
async function hybridSearch(query: string, limit: number) {
  // FTS5搜索（关键词匹配）
  const ftsResults = await ftsSearch(query, limit * 2);
  
  // ChromaDB搜索（语义匹配）
  const vectorResults = await vectorSearch(query, limit * 2);
  
  // 合并并去重
  const merged = mergeResults(ftsResults, vectorResults);
  
  // 重新排序（加权）
  return merged
    .map(r => ({
      ...r,
      // FTS5得分 + 向量得分的加权和
      combined_score: (r.fts_score * 0.4) + (r.vector_score * 0.6)
    }))
    .sort((a, b) => b.combined_score - a.combined_score)
    .slice(0, limit);
}
```

### 6.2 时间衰减

```typescript
// 近期记忆权重更高
function applyTimeDecay(results: SearchResult[]): SearchResult[] {
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  
  return results.map(r => {
    const ageDays = (now - r.created_at) / dayMs;
    // 指数衰减：每30天权重减半
    const decayFactor = Math.pow(0.5, ageDays / 30);
    
    return {
      ...r,
      adjusted_score: r.score * decayFactor
    };
  });
}
```

### 6.3 项目隔离

```typescript
// 搜索时限定项目范围
async function searchWithProjectScope(
  query: string, 
  projectId: string
) {
  return search({
    query,
    where: { project_id: projectId },  // SQLite
    filter: { project_id: projectId }  // ChromaDB
  });
}
```

## 七、Token优化最佳实践

### 7.1 索引设计原则

```typescript
// 索引应该包含什么？
interface SearchIndex {
  id: number;              // 必须：用于后续查询
  title: string;           // 必须：快速判断相关性
  type: string;            // 必须：过滤维度
  created_at: string;      // 必须：时间过滤
  relevance_score: number; // 可选：排序依据
  preview: string;         // 可选：简短预览（<50字）
}
```

**原则**：索引只包含**判断相关性所需的最少信息**。

### 7.2 详情获取原则

```typescript
// 什么时候获取详情？
function shouldFetchDetails(
  index: SearchIndex, 
  context: QueryContext
): boolean {
  // 1. 相关度高
  if (index.relevance_score < 0.7) return false;
  
  // 2. 时间相关（最近7天）
  const ageDays = (Date.now() - index.created_at) / (24*60*60*1000);
  if (ageDays > 7 && index.relevance_score < 0.9) return false;
  
  // 3. 类型匹配
  if (context.preferredTypes && !context.preferredTypes.includes(index.type)) {
    return false;
  }
  
  return true;
}
```

### 7.3 上下文注入优化

```typescript
// 注入时的格式优化
function formatForInjection(observation: Observation): string {
  // 只注入关键信息
  return `
## ${observation.title}
- **类型**: ${observation.type}
- **时间**: ${formatDate(observation.created_at)}
- **关键发现**: ${observation.facts.join('; ')}
- **相关文件**: ${observation.files_modified.join(', ')}
  `.trim();
  
  // 不注入完整的narrative（除非特别需要）
}
```

## 八、监控与调优

### 8.1 搜索性能监控

```typescript
// 搜索指标收集
interface SearchMetrics {
  query: string;
  layer1_tokens: number;
  layer2_tokens: number;
  layer3_tokens: number;
  total_tokens: number;
  results_count: number;
  cache_hit: boolean;
  duration_ms: number;
}

// 记录每次搜索
function recordSearchMetrics(metrics: SearchMetrics) {
  db.prepare(`
    INSERT INTO search_metrics 
    (query, layer1_tokens, layer2_tokens, layer3_tokens, total_tokens, 
     results_count, cache_hit, duration_ms, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    metrics.query, metrics.layer1_tokens, metrics.layer2_tokens,
    metrics.layer3_tokens, metrics.total_tokens, metrics.results_count,
    metrics.cache_hit ? 1 : 0, metrics.duration_ms, Date.now()
  );
}
```

### 8.2 Token使用报告

```typescript
// 生成Token使用报告
function generateTokenReport(projectId: string, days: number = 7) {
  const report = db.prepare(`
    SELECT 
      DATE(created_at/1000, 'unixepoch') as date,
      COUNT(*) as search_count,
      SUM(total_tokens) as total_tokens,
      AVG(total_tokens) as avg_tokens_per_search,
      SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits
    FROM search_metrics
    WHERE project_id = ?
    AND created_at > ?
    GROUP BY date
    ORDER BY date DESC
  `).all(projectId, Date.now() - days * 24 * 60 * 60 * 1000);
  
  return report;
}
```

## 九、总结

Claude-Mem的搜索架构通过**渐进式披露**实现了显著的Token优化：

1. **3层架构**：索引 → 时间线 → 详情
2. **Token节省**：60-90%的Token消耗减少
3. **智能筛选**：Agent参与决策，只获取真正需要的
4. **混合搜索**：FTS5 + ChromaDB，兼顾关键词和语义
5. **持续优化**：缓存、预取、批量操作

**关键洞察**：搜索架构的核心不是"搜索更快"，而是"**让Agent用最少的Token做出最好的决策**"。

---

**下一篇**：[Claude-Mem系列(5)：异步队列与容错机制](./2026-05-12-claude-mem-async-queue.md)
