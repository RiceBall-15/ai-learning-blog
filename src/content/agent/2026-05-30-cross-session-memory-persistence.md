---
title: '跨会话记忆持久化与增量同步策略'
description: '解决Agent的会话边界问题，实现记忆跨设备、跨会话的持久化存储与增量同步架构'
date: 2026-05-30
author: 'RiceBall-15'
category: 'agent'
subCategory: 'agent-memory'
tags: ['跨会话记忆', '持久化', '增量同步', 'CRDT', '分布式一致性']
draft: false
---

# 跨会话记忆持久化与增量同步策略

## 引言

每个AI Agent都面临一个根本性的挑战：**会话边界**。

用户今天和Agent聊了1小时，建立了丰富的上下文——用户是后端工程师、偏好Java生态、正在做一个微服务重构项目。明天用户再来，这些信息全部丢失。Agent又变成了一个"失忆的陌生人"。

跨会话记忆持久化解决的核心问题是：**如何让Agent"记住"用户，跨越会话的鸿沟。**

本文深入分析跨会话记忆的技术架构，从持久化策略到分布式同步，提供完整的工程实现方案。

---

## §1 会话边界问题分析

### 1.1 为什么需要跨会话记忆

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  会话 1      │    │  会话 2      │    │  会话 3      │
│  2026-05-28  │───▶│  2026-05-29  │───▶│  2026-05-30  │
│  Java重构    │    │  (空白)      │    │  继续重构?   │
│  微服务拆分  │    │  重新介绍    │    │  又忘了...   │
└─────────────┘    └─────────────┘    └─────────────┘
         ↓                ↓                 ↓
    有价值的记忆      记忆断层           反复劳动
```

### 1.2 会话状态的生命周期

```python
class SessionLifecycle:
    """会话状态的完整生命周期"""
    
    STATES = {
        'active':    '当前活跃会话 - 内存中，实时读写',
        'dormant':   '休眠会话 - 已结束，内存释放，数据待持久化',
        'archived':  '归档会话 - 已持久化，可恢复',
        'evicted':   '淘汰会话 - 超过保留策略，已清理',
    }
    
    # 典型时间线
    timeline = """
    用户发起会话 → active（内存中，延迟<10ms）
         ↓ 用户关闭/超时
    自动持久化 → dormant（写入存储，延迟<100ms）
         ↓ 超过7天未访问
    冷存储归档 → archived（压缩存储，恢复延迟<1s）
         ↓ 超过90天
    选择性淘汰 → evicted（根据策略保留摘要或删除）
    """
```

### 1.3 跨会话记忆的三个维度

| 维度 | 问题 | 技术方案 |
|------|------|----------|
| **时间维度** | 上次会话的信息如何延续 | 持久化 + 摘要压缩 |
| **空间维度** | 多设备间如何保持一致 | 增量同步 + 冲突解决 |
| **语义维度** | 如何关联不同会话的上下文 | 知识图谱 + 会话关联 |

---

## §2 记忆检查点设计

### 2.1 检查点架构

```
┌──────────────────────────────────────────────────┐
│                Memory Checkpoint System           │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ 自动检查点│  │ 手动检查点│  │ 事件检查点│      │
│  │ (定时)   │  │ (用户)   │  │ (关键动作)│      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └──────────────┼──────────────┘            │
│                      ▼                            │
│            ┌─────────────────┐                   │
│            │  Checkpoint      │                   │
│            │  Aggregator      │                   │
│            └────────┬────────┘                   │
│                     ▼                            │
│    ┌────────────────────────────────┐            │
│    │  Checkpoint Store              │            │
│    │  ┌─────────┐  ┌─────────┐    │            │
│    │  │ SQLite  │  │ S3/OSS  │    │            │
│    │  │ (热数据) │  │ (冷数据) │    │            │
│    │  └─────────┘  └─────────┘    │            │
│    └────────────────────────────────┘            │
└──────────────────────────────────────────────────┘
```

### 2.2 检查点数据结构

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json


@dataclass
class MemoryCheckpoint:
    """记忆检查点 - 跨会话持久化的核心数据结构"""
    
    # 唯一标识
    checkpoint_id: str
    user_id: str
    session_id: str
    
    # 时间信息
    created_at: datetime
    session_start: datetime
    session_end: datetime
    
    # 记忆内容
    conversation_summary: str          # 对话摘要
    key_entities: List[str]            # 关键实体
    user_preferences: Dict[str, Any]   # 用户偏好
    learned_facts: List[str]           # 学习到的事实
    pending_tasks: List[str]           # 待处理任务
    
    # 元数据
    memory_size_bytes: int = 0
    importance_score: float = 0.0      # 0-1 重要性评分
    access_count: int = 0              # 访问次数
    
    # 同步信息
    version: int = 1                   # 版本号
    parent_checkpoint_id: Optional[str] = None  # 父检查点
    checksum: str = ""                 # 数据完整性校验
    
    def compute_checksum(self):
        """计算校验和，确保数据完整性"""
        data = json.dumps({
            'conversation_summary': self.conversation_summary,
            'key_entities': self.key_entities,
            'user_preferences': self.user_preferences,
            'learned_facts': self.learned_facts,
        }, ensure_ascii=False, sort_keys=True)
        self.checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
        return self.checksum
    
    def decay_importance(self, days_since_access: int) -> float:
        """重要性衰减 - 越久没访问越不重要"""
        import math
        decay_rate = 0.05  # 每天衰减5%
        self.importance_score *= math.exp(-decay_rate * days_since_access)
        return self.importance_score
```

### 2.3 自动检查点触发策略

```python
class CheckpointTrigger:
    """检查点触发策略 - 何时保存记忆"""
    
    # 触发条件
    TRIGGERS = {
        'time_based':     '每30分钟自动保存',
        'turn_based':     '每10轮对话保存',
        'importance':     '检测到重要信息时立即保存',
        'session_end':    '会话结束时保存',
        'threshold':      '上下文使用率超过80%时保存',
    }
    
    def should_checkpoint(self, session_state: dict) -> bool:
        """判断是否需要创建检查点"""
        
        reasons = []
        
        # 时间触发
        if session_state.get('minutes_since_last_checkpoint', 0) >= 30:
            reasons.append('time_based')
        
        # 轮次触发
        if session_state.get('turns_since_last_checkpoint', 0) >= 10:
            reasons.append('turn_based')
        
        # 重要性触发
        if session_state.get('last_turn_importance', 0) > 0.7:
            reasons.append('importance')
        
        # 上下文触发
        if session_state.get('context_usage_ratio', 0) > 0.8:
            reasons.append('threshold')
        
        # 会话结束触发
        if session_state.get('session_ending', False):
            reasons.append('session_end')
        
        return len(reasons) > 0, reasons
    
    def compute_importance(self, message: str, context: dict) -> float:
        """评估单条消息的重要性"""
        score = 0.0
        
        # 关键词匹配
        high_importance_keywords = [
            '必须', '重要', '记住', '不要忘记', '以后都',
            '偏好', '喜欢', '讨厌', '密码', '账号',
        ]
        for kw in high_importance_keywords:
            if kw in message:
                score += 0.2
        
        # 包含实体信息
        if any(entity in message for entity in context.get('entities', [])):
            score += 0.15
        
        # 包含决策/结论
        decision_words = ['决定', '选择', '确认', '同意', '方案']
        for dw in decision_words:
            if dw in message:
                score += 0.1
        
        return min(score, 1.0)
```

---

## §3 增量同步架构

### 3.1 增量 vs 全量同步

```
全量同步（简单但低效）:
┌──────────┐                    ┌──────────┐
│  Device A │ ──全部记忆数据──▶ │  Device B │
│  100MB    │                   │  100MB    │
└──────────┘                    └──────────┘
  每次同步 100MB，网络开销大

增量同步（高效但复杂）:
┌──────────┐                    ┌──────────┐
│  Device A │ ──delta差异──▶    │  Device B │
│  v1 → v2  │    (仅2KB)       │  v1 → v2  │
└──────────┘                    └──────────┘
  只传输变化部分，带宽节省99%
```

### 3.2 基于版本向量的增量同步

```python
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MemoryOperation:
    """单次记忆操作 - 同步的最小单位"""
    op_id: str
    op_type: str          # 'add', 'update', 'delete'
    key: str              # 记忆键
    value: any            # 记忆值
    timestamp: float      # 操作时间戳
    device_id: str        # 来源设备
    version_vector: Dict[str, int]  # 版本向量


class IncrementalSyncEngine:
    """增量同步引擎 - 基于版本向量的冲突检测"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.version_vector: Dict[str, int] = {}
        self.operation_log: List[MemoryOperation] = []
        self.pending_ops: List[MemoryOperation] = []
    
    def add_operation(self, op: MemoryOperation):
        """添加新操作到本地日志"""
        self.operation_log.append(op)
        self.pending_ops.append(op)
        
        # 更新版本向量
        if op.device_id in self.version_vector:
            self.version_vector[op.device_id] += 1
        else:
            self.version_vector[op.device_id] = 1
    
    def compute_delta(self, remote_version: Dict[str, int]) -> List[MemoryOperation]:
        """计算需要发送的增量操作"""
        delta = []
        for op in self.operation_log:
            # 只发送远程还没有的操作
            remote_v = remote_version.get(op.device_id, 0)
            op_v = self.version_vector.get(op.device_id, 0)
            if op.timestamp > remote_v:
                delta.append(op)
        return delta
    
    def merge_remote(self, remote_ops: List[MemoryOperation]) -> List[str]:
        """合并远程操作，返回冲突列表"""
        conflicts = []
        
        for remote_op in remote_ops:
            # 检查本地是否有冲突操作
            local_conflicts = [
                op for op in self.operation_log
                if op.key == remote_op.key
                and op.timestamp > remote_op.timestamp
                and op.device_id != remote_op.device_id
            ]
            
            if local_conflicts:
                # 冲突解决：使用最后写入胜出 (LWW)
                for local_op in local_conflicts:
                    if remote_op.timestamp > local_op.timestamp:
                        # 远程更新，覆盖本地
                        conflicts.append(f"conflict: {remote_op.key} "
                                       f"(remote wins: {remote_op.device_id})")
                    else:
                        conflicts.append(f"conflict: {remote_op.key} "
                                       f"(local wins: {self.device_id})")
            else:
                # 无冲突，直接应用
                self.operation_log.append(remote_op)
                self._apply_operation(remote_op)
        
        return conflicts
    
    def _apply_operation(self, op: MemoryOperation):
        """应用单个操作到本地存储"""
        if op.op_type == 'add':
            self._store[op.key] = op.value
        elif op.op_type == 'update':
            self._store[op.key] = op.value
        elif op.op_type == 'delete':
            self._store.pop(op.key, None)
```

### 3.3 冲突解决策略

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **Last Write Wins (LWW)** | 简单偏好设置 | 实现简单 | 可能丢失并发修改 |
| **CRDT** | 计数器、集合 | 无冲突合并 | 实现复杂 |
| **操作合并** | 结构化记忆 | 保留双方修改 | 需要合并规则 |
| **用户仲裁** | 关键决策 | 最准确 | 需要用户参与 |

### 3.4 CRDT 在记忆同步中的应用

```python
class CRDTSet:
    """基于 CRDT 的去重集合 - 用于同步记忆条目"""
    
    def __init__(self):
        # 每个元素: {element: {device_id: (timestamp, is_active)}}
        self.elements: Dict[str, Dict[str, Tuple[float, bool]]] = {}
    
    def add(self, element: str, device_id: str, timestamp: float):
        """添加元素（永不删除，只标记）"""
        if element not in self.elements:
            self.elements[element] = {}
        
        existing = self.elements[element].get(device_id)
        if existing is None or timestamp > existing[0]:
            self.elements[element][device_id] = (timestamp, True)
    
    def remove(self, element: str, device_id: str, timestamp: float):
        """删除元素（标记为删除，保留墓碑）"""
        if element not in self.elements:
            self.elements[element] = {}
        
        existing = self.elements[element].get(device_id)
        if existing is None or timestamp > existing[0]:
            self.elements[element][device_id] = (timestamp, False)
    
    def lookup(self, element: str) -> bool:
        """查询元素是否存活 - 所有副本都删除才删除"""
        if element not in self.elements:
            return False
        # 只要任一设备还标记为 active，元素就存活
        return any(is_active for _, is_active in self.elements[element].values())
    
    def merge(self, remote: 'CRDTSet'):
        """合并远程 CRDT 集合"""
        for element, remote_versions in remote.elements.items():
            if element not in self.elements:
                self.elements[element] = remote_versions.copy()
            else:
                for device_id, (timestamp, is_active) in remote_versions.items():
                    existing = self.elements[element].get(device_id)
                    if existing is None or timestamp > existing[0]:
                        self.elements[element][device_id] = (timestamp, is_active)
```

---

## §4 生产架构设计

### 4.1 完整同步架构

```
┌─────────────────────────────────────────────────────────────┐
│                  跨会话记忆持久化架构                         │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Device A │  │ Device B │  │ Device C │    客户端层      │
│  │  (手机)  │  │  (电脑)  │  │  (平板)  │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│       │              │              │                       │
│       ▼              ▼              ▼                       │
│  ┌──────────────────────────────────────────┐              │
│  │           Sync Gateway (WebSocket)        │  同步层     │
│  │  ┌──────────┐  ┌──────────┐             │              │
│  │  │ Delta    │  │ Conflict │             │              │
│  │  │ Compress │  │ Resolver │             │              │
│  │  └──────────┘  └──────────┘             │              │
│  └──────────────────┬───────────────────────┘              │
│                     ▼                                      │
│  ┌──────────────────────────────────────────┐              │
│  │           Memory Store                   │  存储层     │
│  │  ┌────────┐ ┌────────┐ ┌────────┐     │              │
│  │  │ Redis  │ │ Postgre│ │ Milvus │     │              │
│  │  │ (热)   │ │ (持久) │ │ (向量) │     │              │
│  │  └────────┘ └────────┘ └────────┘     │              │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 三层存储架构

```python
class MemoryStore:
    """三层记忆存储 - 热温冷分级"""
    
    def __init__(self):
        self.hot_store = RedisMemory()    # 热数据：活跃会话记忆
        self.warm_store = PostgresMemory() # 温数据：近期检查点
        self.cold_store = S3Memory()      # 冷数据：历史归档
    
    async def save_checkpoint(self, checkpoint: MemoryCheckpoint):
        """保存检查点到对应层级"""
        
        # 判断存储层级
        if checkpoint.importance_score > 0.7:
            # 高重要性 → 热存储
            await self.hot_store.save(checkpoint)
        elif checkpoint.importance_score > 0.3:
            # 中重要性 → 温存储
            await self.warm_store.save(checkpoint)
        else:
            # 低重要性 → 冷存储（压缩后）
            await self.cold_store.save_compressed(checkpoint)
    
    async def restore_session(self, user_id: str, 
                              session_id: str) -> MemoryCheckpoint:
        """恢复会话记忆 - 从热到冷逐层查找"""
        
        # 1. 先查热存储
        checkpoint = await self.hot_store.get(user_id, session_id)
        if checkpoint:
            return checkpoint
        
        # 2. 查温存储
        checkpoint = await self.warm_store.get(user_id, session_id)
        if checkpoint:
            # 提升到热存储（预热）
            await self.hot_store.save(checkpoint)
            return checkpoint
        
        # 3. 查冷存储
        checkpoint = await self.cold_store.get_decompressed(user_id, session_id)
        if checkpoint:
            # 提升到温存储
            await self.warm_store.save(checkpoint)
            return checkpoint
        
        return None
    
    async def cleanup_old_checkpoints(self, user_id: str, 
                                      keep_days: int = 30):
        """清理旧检查点 - 保留摘要，删除详情"""
        old_checkpoints = await self.warm_store.get_old(user_id, keep_days)
        
        for cp in old_checkpoints:
            # 压缩摘要
            summary = await self._compress_to_summary(cp)
            await self.cold_store.save_summary(user_id, cp.checkpoint_id, summary)
            # 删除详细数据
            await self.warm_store.delete(cp.checkpoint_id)
```

### 4.3 Redis 热存储实现

```python
import redis.asyncio as redis
import json
from datetime import timedelta


class RedisMemory:
    """Redis 热存储 - 活跃会话记忆"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url)
    
    async def save(self, checkpoint: MemoryCheckpoint):
        """保存检查点到 Redis"""
        key = f"memory:{checkpoint.user_id}:{checkpoint.session_id}"
        
        data = {
            'conversation_summary': checkpoint.conversation_summary,
            'key_entities': checkpoint.key_entities,
            'user_preferences': checkpoint.user_preferences,
            'learned_facts': checkpoint.learned_facts,
            'pending_tasks': checkpoint.pending_tasks,
            'importance_score': checkpoint.importance_score,
            'version': checkpoint.version,
            'created_at': checkpoint.created_at.isoformat(),
        }
        
        # 序列化存储
        await self.client.setex(
            name=key,
            time=timedelta(days=7),  # 7天TTL
            value=json.dumps(data, ensure_ascii=False)
        )
        
        # 维护用户索引
        await self.client.sadd(
            f"user:{checkpoint.user_id}:sessions",
            checkpoint.session_id
        )
    
    async def get(self, user_id: str, session_id: str) -> dict:
        """获取检查点"""
        key = f"memory:{user_id}:{session_id}"
        data = await self.client.get(key)
        if data:
            # 延长 TTL（被访问的数据更可能再被访问）
            await self.client.expire(key, timedelta(days=7))
            return json.loads(data)
        return None
    
    async def search_similar(self, user_id: str, 
                             query_entities: list) -> list:
        """基于实体搜索相关会话记忆"""
        # 获取用户所有会话
        sessions = await self.client.smembers(
            f"user:{user_id}:sessions"
        )
        
        results = []
        for session_id in sessions:
            data = await self.get(user_id, session_id.decode())
            if not data:
                continue
            
            # 计算实体重叠度
            session_entities = set(data.get('key_entities', []))
            query_set = set(query_entities)
            overlap = len(session_entities & query_set) / max(len(query_set), 1)
            
            if overlap > 0:
                results.append({
                    'session_id': session_id.decode(),
                    'relevance': overlap,
                    'data': data
                })
        
        # 按相关性排序
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:5]
```

---

## §5 会话上下文恢复

### 5.1 上下文恢复策略

```python
class ContextRestorer:
    """会话上下文恢复器 - 新会话开始时自动加载历史"""
    
    def __init__(self, memory_store: MemoryStore, llm_client):
        self.store = memory_store
        self.llm = llm_client
    
    async def restore_context(self, user_id: str) -> dict:
        """恢复用户上下文"""
        
        # 1. 获取最近的检查点
        recent_checkpoints = await self.store.get_recent(user_id, limit=5)
        
        # 2. 获取用户画像
        user_profile = await self.store.get_user_profile(user_id)
        
        # 3. 获取待处理任务
        pending_tasks = await self.store.get_pending_tasks(user_id)
        
        # 4. 生成上下文摘要
        context = {
            'user_profile': user_profile,
            'recent_activities': [
                {
                    'session_id': cp.session_id,
                    'summary': cp.conversation_summary,
                    'date': cp.created_at,
                }
                for cp in recent_checkpoints
            ],
            'pending_tasks': pending_tasks,
            'restored_at': datetime.now(),
        }
        
        # 5. LLM 压缩生成简洁的上下文提示
        context_prompt = await self._compress_context(context)
        context['system_prompt_addition'] = context_prompt
        
        return context
    
    async def _compress_context(self, context: dict) -> str:
        """用 LLM 压缩历史上下文为系统提示"""
        
        prompt = f"""
        以下是用户的历史记忆摘要，请压缩为一段简洁的上下文提示（200字以内），
        用于帮助AI助手快速了解用户背景。

        用户画像: {context['user_profile']}
        
        最近活动:
        {json.dumps(context['recent_activities'], ensure_ascii=False, indent=2)}
        
        待处理任务:
        {context['pending_tasks']}
        
        请输出简洁的上下文提示:
        """
        
        response = await self.llm.generate(prompt)
        return response.text
```

### 5.2 智能上下文注入

```python
class SmartContextInjector:
    """智能上下文注入 - 根据当前对话选择性加载历史"""
    
    def __init__(self, memory_store, vector_store):
        self.memory = memory_store
        self.vectors = vector_store
    
    async def inject_relevant_context(self, current_message: str, 
                                       user_id: str) -> str:
        """根据当前消息，注入相关的历史上下文"""
        
        # 1. 向量检索相关历史片段
        relevant_memories = await self.vectors.search(
            query=current_message,
            user_id=user_id,
            top_k=3
        )
        
        # 2. 获取用户长期偏好
        preferences = await self.memory.get_user_preferences(user_id)
        
        # 3. 智能组装上下文
        context_parts = []
        
        if relevant_memories:
            context_parts.append("## 相关历史记录")
            for mem in relevant_memories:
                context_parts.append(f"- [{mem['date']}] {mem['summary']}")
        
        if preferences:
            context_parts.append("\n## 用户偏好")
            for key, value in preferences.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
```

---

## §6 性能优化

### 6.1 关键性能指标

| 指标 | 目标值 | 优化策略 |
|------|--------|----------|
| 检查点保存延迟 | < 50ms | 异步写入 + 批量提交 |
| 上下文恢复延迟 | < 200ms | 预热缓存 + 并行加载 |
| 增量同步延迟 | < 500ms | WebSocket + Delta压缩 |
| 冲突解决延迟 | < 100ms | 本地优先 + 异步仲裁 |
| 存储压缩比 | > 5:1 | 摘要压缩 + 去重 |

### 6.2 批量写入优化

```python
class BatchWriter:
    """批量写入器 - 减少IO开销"""
    
    def __init__(self, store, batch_size=50, flush_interval=5.0):
        self.store = store
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer: List[MemoryCheckpoint] = []
        self._last_flush = time.time()
    
    async def write(self, checkpoint: MemoryCheckpoint):
        """加入缓冲区"""
        self.buffer.append(checkpoint)
        
        # 达到批量大小或超时，触发写入
        if (len(self.buffer) >= self.batch_size or 
            time.time() - self._last_flush > self.flush_interval):
            await self.flush()
    
    async def flush(self):
        """批量写入到存储"""
        if not self.buffer:
            return
        
        # 批量写入
        await self.store.batch_save(self.buffer)
        
        # 清空缓冲区
        count = len(self.buffer)
        self.buffer.clear()
        self._last_flush = time.time()
        
        return count
```

---

## §7 实战：完整的跨会话Agent

```python
class PersistentAgent:
    """支持跨会话记忆的AI Agent"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_store = MemoryStore()
        self.context_restorer = ContextRestorer(self.memory_store, llm)
        self.injector = SmartContextInjector(self.memory_store, vector_store)
        self.checkpoint_trigger = CheckpointTrigger()
        self.current_session = None
    
    async def start_session(self):
        """开始新会话 - 自动恢复上下文"""
        
        # 1. 恢复历史上下文
        context = await self.context_restorer.restore_context(self.user_id)
        
        # 2. 构建系统提示
        system_prompt = f"""
你是一个有记忆的AI助手。

{context.get('system_prompt_addition', '')}

如果用户提到之前讨论过的话题，请参考历史记录给出连贯的回答。
"""
        
        self.current_session = {
            'session_id': str(uuid4()),
            'system_prompt': system_prompt,
            'messages': [],
            'context': context,
            'start_time': datetime.now(),
        }
        
        return f"你好！我记得我们之前的交流。{context.get('system_prompt_addition', '')[:100]}..."
    
    async def chat(self, user_message: str) -> str:
        """处理用户消息"""
        
        # 1. 注入相关历史上下文
        relevant_context = await self.injector.inject_relevant_context(
            user_message, self.user_id
        )
        
        # 2. 调用LLM
        full_prompt = f"{relevant_context}\n\n用户: {user_message}"
        response = await llm.generate(full_prompt)
        
        # 3. 记录对话
        self.current_session['messages'].append({
            'role': 'user', 'content': user_message
        })
        self.current_session['messages'].append({
            'role': 'assistant', 'content': response.text
        })
        
        # 4. 检查是否需要创建检查点
        should_save, reasons = self.checkpoint_trigger.should_checkpoint({
            'minutes_since_last_checkpoint': self._minutes_since_checkpoint(),
            'turns_since_last_checkpoint': len(self.current_session['messages']) // 2,
            'session_ending': False,
        })
        
        if should_save:
            await self._create_checkpoint(reasons)
        
        return response.text
    
    async def end_session(self):
        """结束会话 - 保存最终状态"""
        await self._create_checkpoint(['session_end'])
    
    async def _create_checkpoint(self, reasons: list):
        """创建检查点"""
        checkpoint = MemoryCheckpoint(
            checkpoint_id=str(uuid4()),
            user_id=self.user_id,
            session_id=self.current_session['session_id'],
            created_at=datetime.now(),
            session_start=self.current_session['start_time'],
            session_end=datetime.now(),
            conversation_summary=await self._summarize_session(),
            key_entities=self._extract_entities(),
            user_preferences={},
            learned_facts=self._extract_facts(),
            pending_tasks=[],
            importance_score=0.5,
        )
        checkpoint.compute_checksum()
        
        await self.memory_store.save_checkpoint(checkpoint)
```

---

## §8 总结

### 关键技术要点

1. **检查点设计**：多触发策略（时间、轮次、重要性）确保关键记忆不丢失
2. **增量同步**：版本向量 + CRDT 实现高效的多设备同步
3. **三层存储**：热温冷分级，平衡访问速度和存储成本
4. **智能恢复**：根据当前对话动态加载相关历史，避免信息过载
5. **批量写入**：缓冲区 + 批量提交优化写入性能

### 生产部署清单

- [ ] 选择同步协议（WebSocket vs gRPC）
- [ ] 配置存储后端（Redis + PostgreSQL + Milvus）
- [ ] 设置检查点触发策略
- [ ] 实现冲突解决规则
- [ ] 监控同步延迟和存储使用量
- [ ] 配置数据保留策略
- [ ] 测试多设备同步场景

跨会话记忆是Agent从"工具"进化为"助手"的关键一步。当Agent真正记住用户时，每一次对话都是在前一次基础上的深化，而不是从零开始。
