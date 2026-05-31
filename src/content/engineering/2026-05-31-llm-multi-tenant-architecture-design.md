---
title: "LLM应用多租户架构设计：从租户隔离到资源调度的工程实践"
description: "深入解析LLM应用中多租户架构的核心挑战，涵盖认证隔离、资源隔离、数据隔离和成本分摊四大维度，附完整的架构设计方案与代码实现"
date: 2026-05-31
author: "RiceBall-15"
category: "engineering"
subCategory: "infra"
tags: ["多租户", "LLM应用", "架构设计", "租户隔离", "资源管理", "成本优化"]
draft: false
---

# LLM应用多租户架构设计：从租户隔离到资源调度的工程实践

## 一、引言：LLM应用的多租户之痛

如果你正在把一个LLM应用从"给内部团队用"推向"给外部客户提供SaaS服务"，你一定会遇到多租户架构的挑战。

传统的Web应用多租户已经有了成熟的方案——数据库Schema隔离、行级权限、连接池管理。但LLM应用引入了全新的复杂性：

```
┌─────────────────────────────────────────────────────────────┐
│              LLM多租户 vs 传统多租户 差异                      │
├──────────────┬──────────────────┬───────────────────────────┤
│   维度        │   传统Web应用     │   LLM应用                 │
├──────────────┼──────────────────┼───────────────────────────┤
│ 计算资源     │ CPU/内存相对     │ GPU极其昂贵               │
│              │ 便宜且均匀       │ 不同模型成本差100倍        │
├──────────────┼──────────────────┼───────────────────────────┤
│ 请求延迟     │ 毫秒级，可预测   │ 秒级，波动大              │
│              │                  │ 受模型负载影响            │
├──────────────┼──────────────────┼───────────────────────────┤
│ 输入大小     │ 结构化数据       │ 文本/Prompt长度差异巨大   │
│              │ KB级             │ 几十到几十万Token         │
├──────────────┼──────────────────┼───────────────────────────┤
│ 输出大小     │ 结构化数据       │ 长文本生成，不确定性强    │
│              │ 可预估           │ Token消耗难预测           │
├──────────────┼──────────────────┼───────────────────────────┤
│ 成本模型     │ 固定成本为主     │ 按Token计费              │
│              │                  │ 单请求成本可达数美元      │
├──────────────┼──────────────────┼─────────────────────────┤
│ 安全要求     │ 数据库级隔离     │ Prompt注入风险           │
│              │                  │ 模型记忆泄露风险         │
└──────────────┴──────────────────┴───────────────────────────┘
```

这些差异意味着：**传统多租户方案在LLM场景下几乎全部失效**。你需要一套全新的架构来应对。

---

## 二、多租户架构设计全景

### 2.1 四大核心挑战

LLM应用的多租户架构需要同时解决四个维度的问题：

```
┌─────────────────────────────────────────────────────────────┐
│               LLM多租户架构四大支柱                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌───────────────┐                        │
│                    │   认证隔离     │                        │
│                    │  Auth Isolation│                        │
│                    └───────┬───────┘                        │
│                            │                                │
│  ┌───────────────┐         │         ┌───────────────┐     │
│  │   数据隔离     │─────────┼─────────│   资源隔离     │     │
│  │ Data Isolation │         │         │Resource Isolation│   │
│  └───────┬───────┘         │         └───────┬───────┘     │
│          │                 │                 │              │
│          └─────────────────┼─────────────────┘              │
│                            │                                │
│                    ┌───────┴───────┐                        │
│                    │   成本分摊     │                        │
│                    │ Cost Allocation│                        │
│                    └───────────────┘                        │
│                                                             │
│  认证隔离：确保租户A无法调用租户B的API Key/模型               │
│  数据隔离：确保租户A的上下文/知识库不泄露给租户B               │
│  资源隔离：防止租户A的高并发拖垮租户B的服务                   │
│  成本分摊：精确追踪每个租户的Token消耗和费用                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM多租户架构全景图                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                            │
│  │ 租户A   │  │ 租户B   │  │ 租户C   │  ← 客户端                   │
│  │ App     │  │ App     │  │ App     │                             │
│  └────┬────┘  └────┬────┘  └────┬────┘                            │
│       │              │              │                                │
│       ▼              ▼              ▼                                │
│  ┌─────────────────────────────────────────────────┐                │
│  │              API Gateway / LLM Gateway           │                │
│  │  ┌──────────┬──────────┬──────────┬──────────┐ │                │
│  │  │ 租户路由  │ 认证鉴权  │ 限流控制  │ 成本计量 │ │                │
│  │  └──────────┴──────────┴──────────┴──────────┘ │                │
│  └────────────────────────┬────────────────────────┘                │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────┐                │
│  │              LLM Application Layer               │                │
│  │  ┌──────────┬──────────┬──────────┬──────────┐ │                │
│  │  │Prompt管理 │RAG引擎   │Agent引擎  │工具管理  │ │                │
│  │  └──────────┴──────────┴──────────┴──────────┘ │                │
│  └────────────────────────┬────────────────────────┘                │
│                           │                                         │
│          ┌────────────────┼────────────────┐                        │
│          ▼                ▼                ▼                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │  租户A数据    │ │  租户B数据    │ │  租户C数据    │                │
│  │  ├─ 知识库    │ │  ├─ 知识库    │ │  ├─ 知识库    │                │
│  │  ├─ 向量库    │ │  ├─ 向量库    │ │  ├─ 向量库    │                │
│  │  └─ 对话历史  │ │  └─ 对话历史  │ │  └─ 对话历史  │                │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────┐                │
│  │              LLM Provider Layer                   │                │
│  │  ┌──────────┬──────────┬──────────┬──────────┐ │                │
│  │  │OpenAI    │ Claude   │ 本地模型  │ 混合路由  │ │                │
│  │  │租户A Key │租户B Key │ 共享GPU  │ 智能调度  │ │                │
│  │  └──────────┴──────────┴──────────┴──────────┘ │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、维度一：认证与鉴权隔离

### 3.1 API Key管理模式

LLM应用的认证隔离面临一个独特挑战：**租户可能需要使用自己的LLM API Key**。

```
┌─────────────────────────────────────────────────────────────┐
│              三种API Key管理模式                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式1: 平台统一Key（适合中小租户）                           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ 租户A   │  │  平台     │  │  OpenAI  │                  │
│  │         │─▶│ 统一Key   │─▶│  API     │                  │
│  │         │  │ 管理中心  │  │          │                  │
│  └─────────┘  └──────────┘  └──────────┘                  │
│  优点: 简单, 成本可控                                       │
│  缺点: 无法为租户定制模型/参数                               │
│                                                             │
│  模式2: 租户自带Key（适合企业客户）                           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ 租户A   │  │  平台     │  │  OpenAI  │                  │
│  │自有Key  │─▶│ Key存储   │─▶│  API     │                  │
│  │         │  │ (加密)    │  │(租户A的) │                  │
│  └─────────┘  └──────────┘  └──────────┘                  │
│  优点: 租户完全控制, 企业合规                                │
│  缺点: Key管理复杂, 安全风险高                               │
│                                                             │
│  模式3: 混合模式（推荐）                                     │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ 租户A   │  │          │  │  OpenAI  │                  │
│  │自有Key  │─▶│  智能路由 │─▶│  API     │                  │
│  ├─────────┤  │          │  ├──────────┤                  │
│  │ 租户B   │  │ Key路由表 │  │  Claude  │                  │
│  │平台Key  │─▶│ + 优先级  │─▶│  API     │                  │
│  └─────────┘  └──────────┘  └──────────┘                  │
│  优点: 灵活, 兼顾体验和合规                                 │
│  缺点: 实现复杂度最高                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 实现代码：安全的Key管理

```python
from cryptography.fernet import Fernet
from dataclasses import dataclass
from typing import Optional
import os
import hashlib

@dataclass
class TenantLLMConfig:
    """租户LLM配置"""
    tenant_id: str
    provider: str           # openai, anthropic, azure, local
    encrypted_key: bytes    # 加密存储的API Key
    model_preferences: dict # 模型偏好 {"default": "gpt-4o", "fast": "gpt-4o-mini"}
    rate_limit: dict        # 限流配置 {"rpm": 60, "tpm": 100000}
    cost_limit: float       # 月度成本上限（美元）

class TenantKeyManager:
    """租户API Key安全管理器"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or os.environ["MASTER_ENCRYPTION_KEY"].encode()
        self.fernet = Fernet(self.master_key)
    
    def store_key(self, tenant_id: str, provider: str, api_key: str) -> None:
        """加密存储租户的API Key"""
        encrypted = self.fernet.encrypt(api_key.encode())
        # 存储到数据库（此处简化为内存）
        self._save_to_db(tenant_id, provider, encrypted)
    
    def get_key(self, tenant_id: str, provider: str) -> str:
        """安全获取租户的API Key"""
        encrypted = self._load_from_db(tenant_id, provider)
        if encrypted is None:
            raise ValueError(f"No key found for tenant {tenant_id}, provider {provider}")
        return self.fernet.decrypt(encrypted).decode()
    
    def rotate_key(self, tenant_id: str, provider: str, new_key: str) -> None:
        """密钥轮换"""
        # 先验证新Key有效
        self._validate_key(provider, new_key)
        # 存储新Key
        self.store_key(tenant_id, provider, new_key)
        # 记录审计日志
        self._audit_log(tenant_id, "key_rotation", provider)
```

### 3.3 租户级路由中间件

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib

class TenantRoutingMiddleware(BaseHTTPMiddleware):
    """根据租户ID路由到对应的LLM配置"""
    
    def __init__(self, app, key_manager: TenantKeyManager):
        super().__init__(app)
        self.key_manager = key_manager
    
    async def dispatch(self, request: Request, call_next):
        # 1. 从请求中提取租户ID
        tenant_id = self._extract_tenant_id(request)
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Missing tenant identifier")
        
        # 2. 获取租户的LLM配置
        tenant_config = self._get_tenant_config(tenant_id)
        
        # 3. 检查租户状态（是否过期、是否被暂停）
        if tenant_config.status != "active":
            raise HTTPException(status_code=403, detail="Tenant account suspended")
        
        # 4. 注入租户上下文到请求中
        request.state.tenant_id = tenant_id
        request.state.llm_config = tenant_config
        
        # 5. 继续处理
        response = await call_next(request)
        
        # 6. 记录本次请求的成本
        await self._log_usage(tenant_id, request, response)
        
        return response
    
    def _extract_tenant_id(self, request: Request) -> str:
        """从多种来源提取租户ID"""
        # 优先级: Header > Query > JWT Token
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            return tenant_id
        
        tenant_id = request.query_params.get("tenant_id")
        if tenant_id:
            return tenant_id
        
        # 从JWT Token中提取
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self._decode_jwt_tenant(token)
        
        return None
```

---

## 四、维度二：数据隔离

### 4.1 隔离策略选择

```
┌─────────────────────────────────────────────────────────────┐
│              数据隔离三种策略对比                              │
├──────────────┬──────────────┬───────────────────────────────┤
│   策略        │  实现方式     │  适用场景                      │
├──────────────┼──────────────┼───────────────────────────────┤
│              │              │                               │
│  独立数据库   │ 每租户一个    │ 大型企业客户                   │
│  (Silo)      │ 数据库实例    │ 合规要求高的行业               │
│              │              │ 数据主权要求                   │
│              │              │                               │
│  优势: 完全隔离, 性能可预测                                  │
│  劣势: 成本高, 运维复杂, 扩展性差                            │
│              │              │                               │
├──────────────┼──────────────┼───────────────────────────────┤
│              │              │                               │
│  共享数据库   │ 同库不同Schema│ 中大型SaaS                     │
│  (Bridge)    │ 或独立Schema  │ 需要一定隔离级别               │
│              │              │                               │
│  优势: 平衡成本和隔离                                        │
│  劣势: Schema管理复杂, 跨租户查询困难                        │
│              │              │                               │
├──────────────┼──────────────┼───────────────────────────────┤
│              │              │                               │
│  共享表       │ tenant_id    │ 中小型SaaS                     │
│  (Pool)      │ 字段隔离      │ 快速迭代阶段                   │
│              │              │                               │
│  优势: 简单, 成本最低, 易扩展                                │
│  劣势: 隔离性最弱, 需严格行级权限                            │
│              │              │                               │
└──────────────┴──────────────┴───────────────────────────────┘
```

**LLM应用推荐**：混合策略——基础数据用Pool模式（对话记录、配置），敏感数据用Silo模式（知识库、向量库）。

### 4.2 向量数据库的租户隔离

向量数据库是RAG应用的核心组件，其多租户隔离尤为关键：

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import uuid

class TenantVectorStore:
    """多租户向量数据库管理器"""
    
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "shared_knowledge_base"
    
    def upsert(self, tenant_id: str, documents: list[dict]):
        """为特定租户插入文档"""
        points = []
        for doc in documents:
            points.append({
                "id": str(uuid.uuid4()),
                "vector": self._embed(doc["content"]),
                "payload": {
                    "tenant_id": tenant_id,  # 关键：每个向量都标记租户ID
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "created_at": doc.get("created_at"),
                }
            })
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, tenant_id: str, query_vector: list[float], top_k: int = 5):
        """仅搜索特定租户的文档"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            ),
            limit=top_k
        )
        return results
    
    def delete_tenant_data(self, tenant_id: str):
        """删除租户所有数据（GDPR合规）"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            )
        )
```

### 4.3 对话历史隔离

```python
from sqlalchemy import Column, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from datetime import datetime

Base = declarative_base()

class ConversationMessage(Base):
    """共享表模式的对话消息表"""
    __tablename__ = "conversation_messages"
    
    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), index=True, nullable=False)  # 租户隔离字段
    user_id = Column(String(36), index=True, nullable=False)    # 用户隔离字段
    session_id = Column(String(36), index=True, nullable=False) # 会话隔离字段
    role = Column(String(20), nullable=False)  # system/user/assistant
    content = Column(Text, nullable=False)
    token_count = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class TenantConversationStore:
    """多租户对话存储"""
    
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
    
    def save_message(self, tenant_id: str, user_id: str, 
                     session_id: str, role: str, content: str):
        with Session(self.engine) as session:
            msg = ConversationMessage(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                token_count=str(self._count_tokens(content))
            )
            session.add(msg)
            session.commit()
    
    def get_history(self, tenant_id: str, session_id: str, limit: int = 20):
        """获取对话历史——自动过滤租户"""
        with Session(self.engine) as session:
            return session.query(ConversationMessage)\
                .filter(
                    ConversationMessage.tenant_id == tenant_id,  # 强制租户过滤
                    ConversationMessage.session_id == session_id
                )\
                .order_by(ConversationMessage.created_at.desc())\
                .limit(limit)\
                .all()
```

---

## 五、维度三：资源隔离与限流

### 5.1 多层限流架构

LLM应用的限流比传统Web应用复杂得多，因为**单个请求的成本可能相差1000倍**（一个GPT-3.5请求 vs 一个GPT-4o长文本请求）。

```
┌─────────────────────────────────────────────────────────────────┐
│                  LLM多租户限流架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: 令牌桶限流（请求级别）                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  租户A: 60 RPM, 120 TPM                              │       │
│  │  租户B: 30 RPM, 60 TPM                               │       │
│  │  租户C: 10 RPM, 20 TPM (免费版)                      │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│  Layer 2: 成本限额（Token级别）                                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  租户A: 月度 $500 上限                                │       │
│  │  租户B: 月度 $200 上限                                │       │
│  │  租户C: 月度 $10 上限 (免费版)                        │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│  Layer 3: 并发控制（GPU级别）                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  租户A: 最大 20 并发请求                              │       │
│  │  租户B: 最大 10 并发请求                              │       │
│  │  租户C: 最大 2 并发请求 (免费版)                      │       │
│  │  共享GPU: 总并发不超过 100                            │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│  Layer 4: 模型路由（智能降级）                                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  租户A请求GPT-4o → 使用GPT-4o                         │       │
│  │  租户C请求GPT-4o → 降级为GPT-4o-mini                   │       │
│  │  高负载时段 → 所有非VIP降级为本地模型                   │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Token级别的精确限流

```python
import time
from dataclasses import dataclass, field
from typing import Optional
import redis

@dataclass
class TenantRateLimit:
    """租户限流配置"""
    requests_per_minute: int
    tokens_per_minute: int
    max_concurrent: int
    monthly_cost_limit: float  # 美元
    priority: int  # 1=VIP, 2=Standard, 3=Free

class MultiTenantRateLimiter:
    """多租户Token级限流器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def check_rate_limit(self, tenant_id: str, estimated_tokens: int) -> dict:
        """检查租户是否可以发起本次请求"""
        config = self._get_tenant_config(tenant_id)
        now = int(time.time())
        window = 60  # 1分钟窗口
        
        # 1. 检查RPM（每分钟请求数）
        rpm_key = f"rate:{tenant_id}:rpm:{now // window}"
        current_rpm = self.redis.incr(rpm_key)
        self.redis.expire(rpm_key, window * 2)
        
        if current_rpm > config.requests_per_minute:
            return {
                "allowed": False,
                "reason": "requests_per_minute_exceeded",
                "retry_after": window - (now % window),
                "current": current_rpm,
                "limit": config.requests_per_minute
            }
        
        # 2. 检查TPM（每分钟Token数）
        tpm_key = f"rate:{tenant_id}:tpm:{now // window}"
        current_tpm = self.redis.incrby(tpm_key, estimated_tokens)
        self.redis.expire(tpm_key, window * 2)
        
        if current_tpm > config.tokens_per_minute:
            # 回滚RPM计数
            self.redis.decr(rpm_key)
            return {
                "allowed": False,
                "reason": "tokens_per_minute_exceeded",
                "retry_after": window - (now % window),
                "current": current_tpm,
                "limit": config.tokens_per_minute
            }
        
        # 3. 检查月度成本上限
        month_key = f"cost:{tenant_id}:{now // (30 * 86400)}"
        current_cost = float(self.redis.get(month_key) or 0)
        
        if current_cost >= config.monthly_cost_limit:
            return {
                "allowed": False,
                "reason": "monthly_cost_exceeded",
                "current_cost": current_cost,
                "limit": config.monthly_cost_limit
            }
        
        # 4. 检查并发数
        concurrent_key = f"concurrent:{tenant_id}"
        current_concurrent = int(self.redis.get(concurrent_key) or 0)
        
        if current_concurrent >= config.max_concurrent:
            return {
                "allowed": False,
                "reason": "concurrent_limit_exceeded",
                "current": current_concurrent,
                "limit": config.max_concurrent
            }
        
        return {"allowed": True}
    
    def record_cost(self, tenant_id: str, input_tokens: int, 
                    output_tokens: int, model: str):
        """记录本次请求的成本"""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        # 更新月度成本计数器
        month_key = f"cost:{tenant_id}:{int(time.time()) // (30 * 86400)}"
        self.redis.incrbyfloat(month_key, cost)
        self.redis.expire(month_key, 31 * 86400)
        
        # 发送成本告警（接近限额时）
        config = self._get_tenant_config(tenant_id)
        current_cost = float(self.redis.get(month_key) or 0)
        usage_ratio = current_cost / config.monthly_cost_limit
        
        if usage_ratio > 0.8:
            self._send_cost_alert(tenant_id, current_cost, config.monthly_cost_limit)
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算请求成本（美元）"""
        # 简化的成本表
        pricing = {
            "gpt-4o": {"input": 2.5 / 1_000_000, "output": 10 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.6 / 1_000_000},
            "claude-3.5-sonnet": {"input": 3 / 1_000_000, "output": 15 / 1_000_000},
            "claude-3.5-haiku": {"input": 0.8 / 1_000_000, "output": 4 / 1_000_000},
        }
        
        rates = pricing.get(model, {"input": 0, "output": 0})
        return input_tokens * rates["input"] + output_tokens * rates["output"]
```

---

## 六、维度四：成本分摊与计量

### 6.1 成本追踪模型

```python
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class CostCategory(Enum):
    LLM_INFERENCE = "llm_inference"     # 模型推理
    VECTOR_STORAGE = "vector_storage"   # 向量存储
    EMBEDDING = "embedding"             # 文本向量化
    RAG_RETRIEVAL = "rag_retrieval"     # RAG检索
    TOOL_EXECUTION = "tool_execution"   # 工具调用
    STORAGE = "storage"                 # 文件存储

@dataclass
class UsageRecord:
    """单条使用记录"""
    tenant_id: str
    user_id: str
    session_id: str
    category: CostCategory
    model: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    metadata: dict
    timestamp: datetime

class TenantCostTracker:
    """租户成本追踪器"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def record_usage(self, record: UsageRecord):
        """记录使用量"""
        self.db.insert("usage_records", {
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
            "session_id": record.session_id,
            "category": record.category.value,
            "model": record.model,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost_usd": record.cost_usd,
            "latency_ms": record.latency_ms,
            "metadata_json": json.dumps(record.metadata),
            "created_at": record.timestamp.isoformat()
        })
        
        # 更新实时聚合表（用于快速查询）
        self._update_daily_aggregate(record)
        self._update_monthly_aggregate(record)
    
    def get_tenant_usage_report(self, tenant_id: str, 
                                 start_date: str, end_date: str) -> dict:
        """生成租户使用报告"""
        records = self.db.query("""
            SELECT 
                category,
                model,
                COUNT(*) as request_count,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cost_usd) as total_cost,
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency
            FROM usage_records
            WHERE tenant_id = %s 
              AND created_at BETWEEN %s AND %s
            GROUP BY category, model
            ORDER BY total_cost DESC
        """, (tenant_id, start_date, end_date))
        
        return {
            "tenant_id": tenant_id,
            "period": {"start": start_date, "end": end_date},
            "breakdown": records,
            "total_cost": sum(r["total_cost"] for r in records),
            "total_requests": sum(r["request_count"] for r in records),
        }
```

### 6.2 成本优化：智能模型路由

一个有效的成本优化策略是**根据请求复杂度自动选择最经济的模型**：

```python
class ModelRouter:
    """智能模型路由器——根据请求特征选择最优模型"""
    
    def __init__(self, tenant_config: dict):
        self.config = tenant_config
    
    def route(self, request: dict) -> str:
        """根据请求特征决定使用哪个模型"""
        
        # 1. 检查租户指定的模型
        if request.get("preferred_model"):
            return request["preferred_model"]
        
        # 2. 根据输入长度路由
        input_tokens = self._estimate_tokens(request["messages"])
        
        if input_tokens < 500:
            # 简单短文本 → 小模型
            return self.config.get("fast_model", "gpt-4o-mini")
        
        if input_tokens < 4000:
            # 中等长度 → 标准模型
            return self.config.get("default_model", "gpt-4o")
        
        # 长文本 → 检查是否需要长上下文模型
        if request.get("needs_reasoning", False):
            return self.config.get("reasoning_model", "o3-mini")
        
        return self.config.get("default_model", "gpt-4o")
    
    def route_with_fallback(self, request: dict, 
                            current_cost_ratio: float) -> str:
        """带成本感知的路由——当成本接近限额时自动降级"""
        
        primary_model = self.route(request)
        
        # 如果月度成本使用超过80%，降级到更便宜的模型
        if current_cost_ratio > 0.8:
            downgrade_map = {
                "o3-mini": "gpt-4o",
                "gpt-4o": "gpt-4o-mini",
                "claude-3.5-sonnet": "claude-3.5-haiku",
            }
            return downgrade_map.get(primary_model, primary_model)
        
        return primary_model
```

---

## 七、生产部署实践

### 7.1 完整的请求处理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM请求完整处理流程                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 请求到达                                                        │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ POST /v1/chat/completions                            │           │
│  │ Headers: X-Tenant-ID: tenant_abc                     │           │
│  │          Authorization: Bearer xxx                   │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │                                           │
│  2. 租户识别 + 认证      ▼                                           │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ • 提取tenant_id from Header/JWT                     │           │
│  │ • 验证租户身份和权限                                  │           │
│  │ • 加载租户配置（模型偏好、限流、成本限额）              │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │                                           │
│  3. 限流检查             ▼                                           │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ • RPM/TPM检查                                       │           │
│  │ • 并发数检查                                         │           │
│  │ • 月度成本限额检查                                    │           │
│  │ • 如超限 → 返回429 + Retry-After                     │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │                                           │
│  4. Prompt构建          ▼                                           │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ • 加载租户自定义System Prompt                         │           │
│  │ • 注入租户专属知识（RAG检索）                         │           │
│  │ • 应用租户级Guardrails                               │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │                                           │
│  5. 模型路由            ▼                                            │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ • 根据租户配置选择模型                                 │           │
│  │ • 智能降级（成本/负载感知）                            │           │
│  │ • 选择API Key（平台Key/租户Key）                      │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │                                           │
│  6. LLM调用             ▼                                            │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ • 发起模型推理请求                                    │           │
│  │ • 记录延迟和Token使用                                 │           │
│  │ • 错误重试（指数退避）                                │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │                                           │
│  7. 后处理 + 计费       ▼                                            │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ • 输出内容过滤（Guardrails）                          │           │
│  │ • 记录使用量和成本                                    │           │
│  │ • 更新租户实时统计                                    │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │                                           │
│  8. 返回响应             ▼                                           │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ • 标准化响应格式                                      │           │
│  │ • 注入成本信息头（X-Cost-USD, X-Tokens-Used）         │           │
│  │ • 返回给客户端                                       │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 健康检查与监控

```python
from prometheus_client import Counter, Histogram, Gauge

# 租户级Prometheus指标
TENANT_REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["tenant_id", "model", "status"]
)

TENANT_TOKEN_USAGE = Counter(
    "llm_tokens_total",
    "Total tokens consumed",
    ["tenant_id", "model", "direction"]  # direction: input/output
)

TENANT_COST_USD = Counter(
    "llm_cost_usd_total",
    "Total cost in USD",
    ["tenant_id", "model"]
)

TENANT_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["tenant_id", "model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

TENANT_ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Currently active LLM requests",
    ["tenant_id"]
)

# 告警规则示例（Prometheus AlertManager）
ALERT_RULES = """
# 租户成本异常告警
- alert: TenantCostSpike
  expr: rate(llm_cost_usd_total[1h]) > 50
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Tenant {{ $labels.tenant_id }} cost spike detected"

# 租户错误率告警
- alert: TenantHighErrorRate
  expr: |
    rate(llm_requests_total{status=~"5.."}[5m]) 
    / rate(llm_requests_total[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Tenant {{ $labels.tenant_id }} error rate > 10%"

# 租户延迟异常告警
- alert: TenantHighLatency
  expr: histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 10
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Tenant {{ $labels.tenant_id }} P95 latency > 10s"
"""
```

---

## 八、常见陷阱与应对

### 8.1 陷阱清单

```
┌─────────────────────────────────────────────────────────────┐
│              LLM多租户常见陷阱 Top 10                         │
├────┬────────────────────────┬───────────────────────────────┤
│ #  │ 陷阱                   │ 应对策略                       │
├────┼────────────────────────┼───────────────────────────────┤
│ 1  │ Prompt注入跨租户泄露    │ 每次请求注入租户专属System     │
│    │                        │ Prompt + 内容过滤              │
├────┼────────────────────────┼───────────────────────────────┤
│ 2  │ 向量库查询穿透         │ 强制tenant_id过滤条件          │
│    │ (忘记加租户过滤)        │ + ORM层面的行级安全策略        │
├────┼────────────────────────┼───────────────────────────────┤
│ 3  │ 成本失控               │ 分层限流 + 实时成本告警        │
│    │ (一个租户耗尽所有预算)  │ + 自动降级机制                 │
├────┼────────────────────────┼───────────────────────────────┤
│ 4  │ 模型记忆泄露           │ 不共享对话上下文               │
│    │ (A的上下文出现在B的回答)│ + 每次请求独立创建会话         │
├────┼────────────────────────┼───────────────────────────────┤
│ 5  │ 限流绕过               │ 多层限流（网关+应用+模型）     │
│    │                        │ + Redis原子操作                 │
├────┼────────────────────────┼───────────────────────────────┤
│ 6  │ 租户配置缓存过期       │ 配置变更实时推送到所有节点      │
│    │                        │ + 定期刷新兜底                 │
├────┼────────────────────────┼───────────────────────────────┤
│ 7  │ 日志泄露               │ 日志脱敏 + 租户级日志隔离       │
│    │ (其他租户数据出现在日志)│ + 结构化日志                   │
├────┼────────────────────────┼───────────────────────────────┤
│ 8  │ 并发资源争抢           │ 租户级连接池 + 队列隔离         │
│    │ (一个租户占满所有连接)  │ + 优先级队列                   │
├────┼────────────────────────┼───────────────────────────────┤
│ 9  │ GDPR数据删除不彻底     │ 统一删除API + 多存储联动删除   │
│    │ (只删了数据库，忘了向量)│ + 删除确认机制                 │
├────┼────────────────────────┼───────────────────────────────┤
│ 10 │ API Key轮换不及时      │ 自动检测过期 + 提前轮换        │
│    │ (Key过期导致服务中断)   │ + 健康检查                     │
└────┴────────────────────────┴───────────────────────────────┘
```

---

## 九、总结

LLM应用的多租户架构是一个**系统工程**，需要同时在认证、数据、资源、成本四个维度建立隔离机制。与传统Web应用不同，LLM的高成本、高延迟、不确定性特征使得多租户设计更加复杂。

**核心原则**：

1. **认证隔离是底线**：租户永远无法访问其他租户的API Key和凭证
2. **数据隔离是合规**：向量库、对话历史、知识库都必须强制租户过滤
3. **资源隔离是稳定性**：分层限流确保一个租户的行为不影响其他租户
4. **成本分摊是商业模式**：精确的Token级计费是SaaS盈利的基础

**架构演进建议**：

- **MVP阶段**：Pool模式 + 基础限流，快速验证商业模式
- **增长阶段**：引入Token级计费 + 智能模型路由，优化成本结构
- **规模阶段**：Bridge/Silo混合模式 + 多区域部署，满足企业合规需求

多租户不是一次性设计，而是一个随着业务规模**持续演进**的过程。关键是尽早建立成本计量体系——**你无法优化你无法度量的东西**。

---

## 参考资料

- [Multi-Tenant Architecture for LLM Applications - AWS](https://aws.amazon.com/blogs/machine-learning/multi-tenant-architecture-for-llm-applications/)
- [Building Multi-Tenant AI Applications - Azure](https://learn.microsoft.com/en-us/azure/architecture/guide/multitenant/overview)
- [LangSmith Multi-Tenancy Best Practices](https://docs.smith.langchain.com/monitoring/guides/multi-tenant)
- [Qdrant Multi-Tenancy Documentation](https://qdrant.tech/documentation/guides/multi-tenancy/)
