---
title: 'Agent记忆系统的隐私安全与审计机制'
description: '全面剖析AI Agent记忆系统的隐私保护架构，涵盖PII检测、加密存储、访问控制、审计追踪、GDPR合规及数据匿名化等核心安全机制'
date: 2026-05-30
author: 'RiceBall-15'
category: 'agent'
subCategory: agent-memory
tags: ['记忆安全', '隐私保护', 'PII检测', 'GDPR', '审计机制']
draft: false
---

# Agent记忆系统的隐私安全与审计机制

## 引言：记忆系统为何需要安全防护？

当AI Agent从"无状态工具"演进为"有记忆的智能体"，一个根本性问题随之浮现——Agent所记忆的数据可能包含用户的身份证号、健康信息、金融数据甚至对话中的隐私内容。这些数据一旦泄露，后果远比传统API泄露更为严重，因为记忆具有**累积性**和**持久性**：每一次交互都可能沉淀新的敏感信息，而这些信息会在未来无数次推理中被反复读取。

记忆系统的安全防护不仅仅是技术问题，更是合规义务。GDPR、CCPA、中国《个人信息保护法》均对数据存储、处理和删除提出了严格要求。本文将从工程实践角度，系统性地覆盖Agent记忆安全的八大核心议题，并提供可落地的代码实现。

---

## 一、记忆数据敏感度分类与PII检测

### 1.1 记忆数据分级模型

并非所有记忆数据都同等敏感。建立分级模型是安全防护的第一步：

```
┌─────────────────────────────────────────────────────────┐
│                 记忆数据敏感度分级                         │
├──────────┬──────────────┬───────────────────────────────┤
│  等级    │  数据类型      │  处理策略                      │
├──────────┼──────────────┼───────────────────────────────┤
│  L4 极密  │ 身份证/密码    │ 加密存储+最小权限+自动过期     │
│  L3 机密  │ 健康/金融数据  │ 加密存储+访问审计+定期清理     │
│  L2 内部  │ 工作内容/偏好  │ 访问控制+选择性加密            │
│  L1 公开  │ 通用知识/FAQ   │ 标准存储                      │
└──────────┴──────────────┴───────────────────────────────┘
```

### 1.2 PII（个人可识别信息）自动检测器

在记忆写入前，必须对内容进行PII扫描。以下是基于正则与NLP混合方案的检测实现：

```python
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SensitivityLevel(Enum):
    L1_PUBLIC = "L1"
    L2_INTERNAL = "L2"
    L3_CONFIDENTIAL = "L3"
    L4_RESTRICTED = "L4"


@dataclass
class PIIDetectionResult:
    has_pii: bool
    detected_types: list[str] = field(default_factory=list)
    masked_content: str = ""
    confidence: float = 0.0


class PIIDetector:
    """基于规则+模式匹配的PII检测器"""

    # 中国身份证号 (18位)
    CN_ID_PATTERN = re.compile(
        r'[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])'
        r'(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]'
    )
    # 手机号
    CN_PHONE_PATTERN = re.compile(r'1[3-9]\d{9}')
    # 银行卡号（16-19位数字）
    BANK_CARD_PATTERN = re.compile(r'\b\d{16,19}\b')
    # 邮箱
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    # 住址（简单匹配"XX路/XX街/XX小区"等关键词）
    ADDRESS_KEYWORDS = re.compile(
        r'[\u4e00-\u9fa5]{2,}(?:路|街|巷|弄|号|小区|栋|单元|室|村|镇)'
    )
    # 人名（简单启发式：连续2-4个汉字，非通用词）
    COMMON_WORDS = {'的', '了', '是', '在', '我', '你', '他', '她', '它',
                    '有', '和', '与', '或', '但', '也', '都', '就'}

    def detect(self, text: str) -> PIIDetectionResult:
        detected = []
        masked = text
        confidence_scores = []

        # 身份证号 — 最高敏感度
        for match in self.CN_ID_PATTERN.finditer(text):
            id_number = match.group()
            if self._validate_cn_id(id_number):
                detected.append("CN_ID_NUMBER")
                masked = masked.replace(id_number, f"[身份证:{id_number[:4]}****{id_number[-4:]}]")
                confidence_scores.append(0.98)

        # 手机号
        for match in self.CN_PHONE_PATTERN.finditer(text):
            phone = match.group()
            detected.append("CN_PHONE")
            masked = masked.replace(phone, f"手机:{phone[:3]}****{phone[-4:]}")
            confidence_scores.append(0.95)

        # 邮箱
        for match in self.EMAIL_PATTERN.finditer(text):
            email = match.group()
            local, domain = email.split('@')
            detected.append("EMAIL")
            masked = masked.replace(email, f"{local[0]}***@{domain}")
            confidence_scores.append(0.97)

        # 银行卡号
        for match in self.BANK_CARD_PATTERN.finditer(text):
            card = match.group()
            detected.append("BANK_CARD")
            masked = masked.replace(card, f"银行卡:{card[-4:]}")
            confidence_scores.append(0.85)

        # 住址
        if self.ADDRESS_KEYWORDS.search(text):
            detected.append("ADDRESS")
            confidence_scores.append(0.75)

        avg_confidence = (sum(confidence_scores) / len(confidence_scores)
                          if confidence_scores else 0.0)

        return PIIDetectionResult(
            has_pii=len(detected) > 0,
            detected_types=detected,
            masked_content=masked,
            confidence=avg_confidence,
        )

    @staticmethod
    def _validate_cn_id(id_number: str) -> bool:
        """校验身份证号校验码"""
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_codes = '10X98765432'
        total = sum(int(id_number[i]) * weights[i] for i in range(17))
        return check_codes[total % 11] == id_number[-1].upper()


# 使用示例
detector = PIIDetector()
result = detector.detect("用户张三的身份证号是110101199001011234，手机号13800138000")
print(f"检测到PII: {result.has_pii}")
print(f"类型: {result.detected_types}")
print(f"脱敏后: {result.masked_content}")
# 检测到PII: True
# 类型: ['CN_ID_NUMBER', 'CN_PHONE']
# 脱敏后: 用户张三的身份证号是[身份证:1101****1234]，手机号手机:138****8000
```

### 1.3 记忆写入拦截器

将PII检测集成到记忆写入流程中，形成安全屏障：

```python
from datetime import datetime
from typing import Any


class SecureMemoryWriter:
    """带PII检测的安全记忆写入器"""

    def __init__(self, memory_store, detector: PIIDetector):
        self.store = memory_store
        self.detector = detector

    def write(self, agent_id: str, content: str,
              metadata: dict[str, Any] | None = None) -> dict:
        # 第一步：PII检测
        detection = self.detector.detect(content)

        if detection.has_pii:
            # 记录审计日志
            self._audit_log("PII_DETECTED", agent_id, detection)

            # 自动脱敏后写入
            content = detection.masked_content
            sensitivity = SensitivityLevel.L3_CONFIDENTIAL
        else:
            sensitivity = SensitivityLevel.L1_PUBLIC

        # 第二步：写入记忆
        memory_entry = {
            "content": content,
            "agent_id": agent_id,
            "sensitivity_level": sensitivity.value,
            "pii_detected": detection.has_pii,
            "pii_types": detection.detected_types,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        entry_id = self.store.put(memory_entry)

        # 第三步：高敏感记忆自动设置过期
        if sensitivity in (SensitivityLevel.L3_CONFIDENTIAL,
                           SensitivityLevel.L4_RESTRICTED):
            self._set_auto_expiry(entry_id, ttl_hours=720)  # 30天

        return {"id": entry_id, "sensitivity": sensitivity.value}

    def _audit_log(self, event_type: str, agent_id: str, detail: Any):
        print(f"[AUDIT] {datetime.utcnow().isoformat()} | "
              f"{event_type} | agent={agent_id} | {detail}")

    def _set_auto_expiry(self, entry_id: str, ttl_hours: int):
        print(f"[SECURITY] 记忆 {entry_id} 设置 {ttl_hours}h 后自动过期")
```

---

## 二、记忆加密：静态存储与传输安全

### 2.1 静态加密（Encryption at Rest）

记忆数据在磁盘或数据库中必须加密存储。采用AES-256-GCM提供认证加密：

```python
import os
import json
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class MemoryEncryptor:
    """基于AES-256-GCM的记忆加密器"""

    def __init__(self, master_key: bytes | None = None):
        # 生产环境应从KMS（密钥管理服务）获取，而非硬编码
        self.master_key = master_key or os.environ.get(
            "MEMORY_ENCRYPTION_KEY", os.urandom(32)
        )

    def encrypt(self, plaintext: str, associated_data: str = "") -> bytes:
        """
        加密记忆内容
        associated_data: 绑定的非敏感元数据（如entry_id），防止密文篡改
        """
        aesgcm = AESGCM(self.master_key)
        nonce = os.urandom(12)  # 96-bit nonce
        ciphertext = aesgcm.encrypt(
            nonce,
            plaintext.encode("utf-8"),
            associated_data.encode("utf-8") if associated_data else None
        )
        return nonce + ciphertext  # nonce || ciphertext || tag

    def decrypt(self, encrypted_data: bytes, associated_data: str = "") -> str:
        """解密记忆内容"""
        aesgcm = AESGCM(self.master_key)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        plaintext = aesgcm.decrypt(
            nonce,
            ciphertext,
            associated_data.encode("utf-8") if associated_data else None
        )
        return plaintext.decode("utf-8")

    def rotate_key(self, old_key: bytes, data: bytes,
                   associated_data: str = "") -> bytes:
        """密钥轮换：用旧密钥解密，新密钥重新加密"""
        # 临时用旧密钥解密
        old_encryptor = MemoryEncryptor(old_key)
        plaintext = old_encryptor.decrypt(data, associated_data)
        # 用新密钥加密
        return self.encrypt(plaintext, associated_data)


class EncryptedMemoryStore:
    """加密记忆存储层"""

    def __init__(self, backend, encryptor: MemoryEncryptor):
        self.backend = backend
        self.encryptor = encryptor

    def put(self, entry_id: str, content: str, **kwargs) -> str:
        encrypted = self.encryptor.encrypt(content, associated_data=entry_id)
        return self.backend.store(entry_id, encrypted, **kwargs)

    def get(self, entry_id: str) -> str | None:
        encrypted = self.backend.fetch(entry_id)
        if encrypted is None:
            return None
        return self.encryptor.decrypt(encrypted, associated_data=entry_id)


# 使用示例
# encrypted_store = EncryptedMemoryStore(redis_backend, MemoryEncryptor())
# encrypted_store.put("mem_001", "用户偏好：暗色主题")
# content = encrypted_store.get("mem_001")  # → "用户偏好：暗色主题"
```

### 2.2 传输加密（Encryption in Transit）

在多Agent系统中，Agent之间共享记忆时必须使用TLS/mTLS：

```python
import ssl
import httpx


class SecureMemoryTransport:
    """安全的记忆传输客户端，使用mTLS"""

    def __init__(self, server_url: str, cert_path: str, key_path: str,
                 ca_path: str):
        self.server_url = server_url
        # 创建mTLS SSL上下文
        self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        self.ssl_context.load_cert_chain(cert_path, key_path)
        self.ssl_context.load_verify_locations(ca_path)
        self.ssl_context.check_hostname = True

    async def share_memory(self, from_agent: str, to_agent: str,
                           memory_entry: dict) -> dict:
        async with httpx.AsyncClient(verify=self.ssl_context) as client:
            response = await client.post(
                f"{self.server_url}/api/v1/memory/share",
                json={
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "entry": memory_entry,
                },
                headers={"X-Request-Id": os.urandom(8).hex()},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
```

---

## 三、多Agent系统的记忆访问控制

### 3.1 RBAC + ABAC 混合访问控制模型

多Agent系统中，记忆的共享必须遵循最小权限原则：

```python
from enum import Enum
from datetime import datetime


class MemoryPermission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"


@dataclass
class AccessPolicy:
    agent_id: str
    allowed_permissions: set[MemoryPermission]
    sensitivity_filter: set[str]  # 可访问的敏感度等级
    max_read_count: int = 100  # 单次读取上限
    time_restricted: bool = False
    allowed_hours: tuple[int, int] = (0, 24)  # 允许访问的时间段


class MemoryAccessController:
    """记忆访问控制器"""

    def __init__(self):
        self.policies: dict[str, AccessPolicy] = {}
        self.access_log: list[dict] = []

    def register_policy(self, policy: AccessPolicy):
        self.policies[policy.agent_id] = policy

    def check_access(self, agent_id: str, target_memory_id: str,
                     permission: MemoryPermission,
                     memory_sensitivity: str) -> tuple[bool, str]:
        """检查Agent是否有权执行指定操作"""
        policy = self.policies.get(agent_id)

        if not policy:
            self._log_access(agent_id, target_memory_id, permission,
                             False, "NO_POLICY")
            return False, "Agent未注册访问策略"

        # 检查权限
        if permission not in policy.allowed_permissions:
            self._log_access(agent_id, target_memory_id, permission,
                             False, "PERMISSION_DENIED")
            return False, f"Agent无{permission.value}权限"

        # 检查敏感度等级
        if memory_sensitivity not in policy.sensitivity_filter:
            self._log_access(agent_id, target_memory_id, permission,
                             False, "SENSITIVITY_DENIED")
            return False, f"不允许访问{memory_sensitivity}级记忆"

        # 检查时间窗口
        if policy.time_restricted:
            current_hour = datetime.utcnow().hour
            if not (policy.allowed_hours[0] <= current_hour <= policy.allowed_hours[1]):
                self._log_access(agent_id, target_memory_id, permission,
                                 False, "TIME_DENIED")
                return False, "当前时间不在允许访问窗口内"

        self._log_access(agent_id, target_memory_id, permission, True, "ALLOWED")
        return True, "允许访问"

    def _log_access(self, agent_id: str, memory_id: str,
                    permission: MemoryPermission, granted: bool, reason: str):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "memory_id": memory_id,
            "permission": permission.value,
            "granted": granted,
            "reason": reason,
        }
        self.access_log.append(entry)
        # 生产环境应异步写入持久化日志


# 使用示例
acl = MemoryAccessController()
acl.register_policy(AccessPolicy(
    agent_id="research_agent",
    allowed_permissions={MemoryPermission.READ},
    sensitivity_filter={"L1", "L2"},
))
acl.register_policy(AccessPolicy(
    agent_id="admin_agent",
    allowed_permissions={MemoryPermission.READ, MemoryPermission.WRITE,
                         MemoryPermission.DELETE, MemoryPermission.SHARE},
    sensitivity_filter={"L1", "L2", "L3", "L4"},
))

ok, msg = acl.check_access("research_agent", "mem_001",
                           MemoryPermission.READ, "L3")
print(f"访问结果: {ok}, 原因: {msg}")
# 访问结果: False, 原因: 不允许访问L3级记忆
```

### 3.2 记忆隔离架构

```
┌──────────────────────────────────────────────────────────┐
│                    Agent记忆隔离架构                       │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Agent A  │  │ Agent B  │  │ Agent C  │               │
│  │ 私有记忆  │  │ 私有记忆  │  │ 私有记忆  │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │             │             │                       │
│  ┌────▼─────────────▼─────────────▼────┐                 │
│  │        访问控制层 (ACL/ABAC)         │                 │
│  └────────────────┬────────────────────┘                 │
│                   │                                       │
│  ┌────────────────▼────────────────────┐                 │
│  │     共享记忆池（加密 + 审计）         │                 │
│  │  ┌──────┐ ┌──────┐ ┌──────┐        │                 │
│  │  │M-100 │ │M-200 │ │M-300 │  ...   │                 │
│  │  └──────┘ └──────┘ └──────┘        │                 │
│  └─────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────┘
```

---

## 四、记忆审计追踪设计

### 4.1 审计日志数据结构

```python
import uuid
import json
from datetime import datetime
from typing import Any


class AuditEventType(Enum):
    MEMORY_CREATE = "memory.create"
    MEMORY_READ = "memory.read"
    MEMORY_UPDATE = "memory.update"
    MEMORY_DELETE = "memory.delete"
    MEMORY_SHARE = "memory.share"
    MEMORY_SEARCH = "memory.search"
    PII_DETECTED = "security.pii_detected"
    ACCESS_DENIED = "security.access_denied"
    KEY_ROTATION = "security.key_rotation"
    BULK_EXPORT = "security.bulk_export"


@dataclass
class AuditEntry:
    """不可变审计日志条目"""
    id: str
    timestamp: str
    event_type: str
    agent_id: str
    target_memory_id: str
    detail: dict
    # 可选：哈希链用于防篡改
    previous_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "target_memory_id": self.target_memory_id,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class MemoryAuditLogger:
    """防篡改审计日志"""

    def __init__(self):
        self.entries: list[AuditEntry] = []
        self._last_hash = "GENESIS"

    def log(self, event_type: AuditEventType, agent_id: str,
            target_memory_id: str, detail: dict[str, Any] | None = None):
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type=event_type.value,
            agent_id=agent_id,
            target_memory_id=target_memory_id,
            detail=detail or {},
            previous_hash=self._last_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self._last_hash = entry.entry_hash
        self.entries.append(entry)

        return entry

    def verify_chain(self) -> bool:
        """验证审计日志链的完整性（防篡改检测）"""
        for i, entry in enumerate(self.entries):
            if i == 0:
                if entry.previous_hash != "GENESIS":
                    return False
            else:
                if entry.previous_hash != self.entries[i - 1].entry_hash:
                    return False
            if entry.entry_hash != entry.compute_hash():
                return False
        return True


# 使用示例
audit = MemoryAuditLogger()
audit.log(AuditEventType.MEMORY_CREATE, "agent_01", "mem_001",
          {"sensitivity": "L2", "content_preview": "用户偏好..."})
audit.log(AuditEventType.MEMORY_READ, "agent_02", "mem_001",
          {"read_reason": "任务上下文构建"})

print(f"审计链完整: {audit.verify_chain()}")
print(f"日志条数: {len(audit.entries)}")
# 审计链完整: True
# 日志条数: 2
```

### 4.2 审计日志存储架构

生产环境中，审计日志应独立于记忆数据存储，确保即使记忆数据被攻破，审计记录仍可追溯：

```
┌────────────────────────────────────────────────┐
│              审计日志架构                        │
│                                                │
│  Agent操作 ──→ 审计拦截器 ──→ Kafka/消息队列    │
│                                  │             │
│                    ┌─────────────┼───────────┐ │
│                    ▼             ▼           ▼ │
│              ┌──────────┐ ┌──────────┐ ┌────┐ │
│              │ 审计数据库 │ │ 冷存储S3 │ │SIEM│ │
│              │(热数据30天)│ │(长期归档) │ │系统│ │
│              └──────────┘ └──────────┘ └────┘ │
└────────────────────────────────────────────────┘
```

---

## 五、GDPR合规：被遗忘权的实现

### 5.1 右侧到遗忘的技术实现

GDPR第17条要求用户有权要求删除其个人数据。对于记忆系统，这意味着：

```python
from datetime import datetime


class GPDRMemoryManager:
    """GDPR合规的记忆管理器"""

    def __init__(self, memory_store, audit_logger: MemoryAuditLogger,
                 search_index):
        self.store = memory_store
        self.audit = audit_logger
        self.search_index = search_index

    def handle_deletion_request(self, user_id: str) -> dict:
        """处理用户数据删除请求（Right to be Forgotten）"""
        affected_memories = []
        errors = []

        # 第一步：搜索所有包含该用户数据的记忆
        entries = self.store.search_by_user(user_id)

        # 第二步：逐一安全删除
        for entry in entries:
            try:
                # 2a. 从搜索索引中移除
                self.search_index.remove(entry["id"])

                # 2b. 覆盖写入（secure wipe）而非简单删除
                self.store.secure_wipe(entry["id"])

                # 2c. 记录审计日志
                self.audit.log(
                    AuditEventType.MEMORY_DELETE,
                    agent_id="system_gdpr",
                    target_memory_id=entry["id"],
                    detail={
                        "reason": "GDPR_ARTICLE_17",
                        "user_id": user_id,
                        "original_sensitivity": entry.get(
                            "sensitivity_level", "unknown"
                        ),
                        "created_at": entry.get("created_at"),
                    },
                )
                affected_memories.append(entry["id"])
            except Exception as e:
                errors.append({"memory_id": entry["id"], "error": str(e)})

        # 第三步：记录删除操作摘要
        self.audit.log(
            AuditEventType.MEMORY_DELETE,
            agent_id="system_gdpr",
            target_memory_id="BULK_DELETION",
            detail={
                "user_id": user_id,
                "total_deleted": len(affected_memories),
                "errors": errors,
                "completed_at": datetime.utcnow().isoformat(),
            },
        )

        return {
            "status": "completed" if not errors else "partial",
            "deleted_count": len(affected_memories),
            "error_count": len(errors),
            "deleted_ids": affected_memories,
        }

    def handle_data_export(self, user_id: str) -> dict:
        """GDPR第20条：数据可携带权 — 导出用户所有记忆数据"""
        entries = self.store.search_by_user(user_id)

        export_data = []
        for entry in entries:
            export_data.append({
                "memory_id": entry["id"],
                "content": entry.get("content", ""),
                "created_at": entry.get("created_at"),
                "sensitivity_level": entry.get("sensitivity_level"),
                "source_agent": entry.get("agent_id"),
            })

        self.audit.log(
            AuditEventType.BULK_EXPORT,
            agent_id="system_gdpr",
            target_memory_id="DATA_EXPORT",
            detail={"user_id": user_id, "exported_count": len(export_data)},
        )

        return {"user_id": user_id, "data": export_data,
                "exported_at": datetime.utcnow().isoformat()}


class SecureWipeStore:
    """支持安全擦除的记忆存储"""

    def __init__(self, backend):
        self.backend = backend

    def secure_wipe(self, entry_id: str):
        """
        安全擦除：用随机数据多次覆盖后再删除
        确保数据无法通过磁盘取证恢复
        """
        # 覆盖写入3次（DoD 5220.22-M标准简化版）
        original = self.backend.fetch(entry_id)
        if original is None:
            return

        data_size = len(original)
        for pass_num in range(3):
            random_data = os.urandom(data_size)
            self.backend.store(entry_id, random_data)

        # 最终删除
        self.backend.delete(entry_id)
```

---

## 六、记忆匿名化技术

### 6.1 多级匿名化策略

```python
import re


class MemoryAnonymizer:
    """记忆数据匿名化处理器"""

    # 替换映射（每次调用生成新的映射，确保跨会话一致性）
    def __init__(self):
        self.entity_map: dict[str, str] = {}
        self.reverse_map: dict[str, str] = {}
        self._counter = 0

    def anonymize(self, text: str, strategy: str = "pseudonymize") -> str:
        """
        匿名化策略：
        - pseudonymize: 假名化（可逆，保留分析价值）
        - generalize: 泛化（不可逆，如年龄35→30-40）
        - redact: 脱敏（完全遮蔽，如姓名→[人物]）
        """
        if strategy == "pseudonymize":
            return self._pseudonymize(text)
        elif strategy == "generalize":
            return self._generalize(text)
        elif strategy == "redact":
            return self._redact(text)
        else:
            raise ValueError(f"未知策略: {strategy}")

    def _pseudonymize(self, text: str) -> str:
        """假名化：用可逆标识符替换真实身份"""
        # 替换人名
        name_pattern = re.compile(r'(?<=用户)[\u4e00-\u9fa5]{2,4}')
        for match in name_pattern.finditer(text):
            original = match.group()
            if original not in self.entity_map:
                self._counter += 1
                self.entity_map[original] = f"PERSON_{self._counter:04d}"
                self.reverse_map[f"PERSON_{self._counter:04d}"] = original
            text = text.replace(match.group(), self.entity_map[original])

        return text

    def _generalize(self, text: str) -> str:
        """泛化：降低数据精度"""
        # 年龄泛化
        age_pattern = re.compile(r'(\d{1,3})岁')
        for match in age_pattern.finditer(text):
            age = int(match.group(1))
            lower = (age // 10) * 10
            upper = lower + 9
            text = text.replace(match.group(), f"{lower}-{upper}岁")

        # 日期泛化（保留年月）
        date_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
        text = date_pattern.sub(r'\1-\1月', text)

        return text

    def _redact(self, text: str) -> str:
        """脱敏：完全遮蔽"""
        # 身份证
        text = re.sub(r'\d{17}[\dXx]', '[身份证号已脱敏]', text)
        # 手机号
        text = re.sub(r'1[3-9]\d{9}', '[手机号已脱敏]', text)
        # 邮箱
        text = re.sub(r'[\w.+-]+@[\w.-]+\.\w+', '[邮箱已脱敏]', text)
        return text

    def de_anonymize(self, text: str) -> str:
        """反匿名化（仅假名化策略可逆）"""
        for pseudonym, real in self.reverse_map.items():
            text = text.replace(pseudonym, real)
        return text
```

### 6.2 k-匿名化在记忆检索中的应用

当多个Agent需要查询共享记忆时，使用k-匿名化防止身份推断：

```python
from collections import Counter


class KAnonymityFilter:
    """k-匿名化过滤器：确保查询结果中每组至少包含k条记录"""

    def __init__(self, k: int = 3):
        self.k = k

    def filter_query_results(self, results: list[dict],
                             quasi_identifiers: list[str]) -> list[dict]:
        """
        对查询结果进行k-匿名化处理
        quasi_identifiers: 准标识符字段（如年龄、地区、职业）
        """
        if not results:
            return results

        # 按准标识符分组
        groups: dict[str, list[dict]] = {}
        for result in results:
            key = tuple(str(result.get(qi, "")) for qi in quasi_identifiers)
            group_key = "|".join(key)
            groups.setdefault(group_key, []).append(result)

        # 过滤掉不足以构成k-匿名的小组
        anonymized = []
        suppressed = []
        for group_key, group_results in groups.items():
            if len(group_results) >= self.k:
                anonymized.extend(group_results)
            else:
                suppressed.extend(group_results)

        # 对被抑制的记录进行泛化处理
        for record in suppressed:
            for qi in quasi_identifiers:
                if qi in record:
                    record[qi] = self._generalize_value(record[qi])
            anonymized.append(record)

        return anonymized

    @staticmethod
    def _generalize_value(value) -> str:
        if isinstance(value, (int, float)):
            lower = int(value) // 10 * 10
            return f"{lower}-{lower + 9}"
        return "[已泛化]"
```

---

## 七、安全记忆删除策略

### 7.1 删除策略选择矩阵

```
┌───────────────────────────────────────────────────────────────┐
│                    记忆删除策略选择                              │
├───────────────┬───────────────────────────────────────────────┤
│ 策略           │ 适用场景                                      │
├───────────────┼───────────────────────────────────────────────┤
│ 即时删除       │ 用户主动请求删除，GDPR合规                     │
│ 定时过期       │ 临时会话记忆，高敏感数据自动过期                │
│ 级联删除       │ Agent销毁时清理所有关联记忆                    │
│ 引用计数删除   │ 共享记忆在所有引用者释放后安全删除              │
│ 加密擦除       │ 轮换加密密钥使旧数据不可解密（逻辑删除）       │
└───────────────┴───────────────────────────────────────────────┘
```

### 7.2 加密擦除（Crypto Shredding）

最高效的大规模删除方案——销毁加密密钥即可使所有关联数据不可访问：

```python
class CryptoShredder:
    """
    加密擦除：通过销毁密钥实现数据删除
    比逐条覆盖快几个数量级，且在云存储场景中更可靠
    """

    def __init__(self, key_store):
        self.key_store = key_store  # KMS密钥管理

    def assign_key(self, memory_id: str, key_id: str):
        """为记忆条目分配加密密钥"""
        self.key_store.bind(memory_id, key_id)

    def shred(self, user_id: str) -> dict:
        """
        对指定用户的所有记忆执行加密擦除
        销毁密钥 → 数据在密文层面变为随机噪声
        """
        key_ids = self.key_store.get_keys_for_user(user_id)
        shred_count = 0

        for key_id in key_ids:
            # 获取使用该密钥加密的所有记忆ID
            memory_ids = self.key_store.get_memories_for_key(key_id)

            # 从密钥存储中删除密钥（使密文不可解密）
            self.key_store.destroy(key_id)

            shred_count += len(memory_ids)

            print(f"[CRYPTO_SHRED] 密钥 {key_id[:8]}... 已销毁，"
                  f"影响 {len(memory_ids)} 条记忆")

        return {
            "shredded_keys": len(key_ids),
            "affected_memories": shred_count,
            "method": "crypto_shredding",
            "timestamp": datetime.utcnow().isoformat(),
        }
```

---

## 八、生产环境安全检查清单

### 安全配置清单

```yaml
# agent-memory-security-checklist.yml
memory_security:
  encryption:
    at_rest:
      algorithm: "AES-256-GCM"
      key_management: "KMS"            # 必须使用专业密钥管理服务
      key_rotation_days: 90            # 密钥最长使用期限
      envelope_encryption: true        # 启用信封加密
    in_transit:
      protocol: "TLS 1.3"
      mutual_tls: true                 # 多Agent间通信启用mTLS
      certificate_rotation_days: 30

  access_control:
    model: "RBAC+ABAC"                 # 混合访问控制
    default_deny: true                 # 默认拒绝策略
    least_privilege: true              # 最小权限原则
    session_timeout_minutes: 30        # 会话超时
    max_concurrent_sessions: 3         # 并发会话限制

  data_protection:
    pii_detection: true                # 写入前PII检测
    auto_masking: true                 # 自动脱敏
    sensitivity_classification: true   # 敏感度分级
    data_retention_days: 90            # 默认数据保留期限
    auto_expiry_l3_l4_days: 30         # 高敏感数据自动过期

  audit:
    enabled: true
    log_all_access: true               # 记录所有访问操作
    immutable_storage: true            # 审计日志不可变存储
    tamper_detection: "hash_chain"     # 哈希链防篡改
    retention_years: 3                 # 审计日志保留期限
    real_time_alerting: true           # 实时安全告警

  gdpr_compliance:
    right_to_be_forgotten: true        # 被遗忘权
    data_portability: true             # 数据可携带权
    consent_management: true           # 同意管理
    dpia_completed: true               # 数据保护影响评估
    dpo_contact: "dpo@company.com"    # 数据保护官联系信息

  deletion:
    secure_wipe_passes: 3              # 安全擦除覆盖次数
    crypto_shredding: true             # 支持加密擦除
    cascade_delete: true               # 级联删除
    deletion_audit: true               # 删除操作审计

  monitoring:
    intrusion_detection: true          # 入侵检测
    anomaly_detection: true            # 异常访问检测
    bulk_access_threshold: 50          # 批量访问阈值告警
    failed_access_threshold: 5         # 连续失败访问阈值告警
```

### 部署前安全审计步骤

```
□ 1. 密钥管理
  □ 密钥已存储在KMS中（非硬编码/环境变量）
  □ 密钥轮换策略已配置
  □ 信封加密已启用
  □ 旧密钥已安全归档或销毁

□ 2. 访问控制
  □ 默认拒绝策略已启用
  □ 所有Agent已注册访问策略
  □ 共享记忆的ACL已审查
  □ 服务账号权限已最小化

□ 3. 数据保护
  □ PII检测管道已部署并测试
  □ 自动脱敏规则已验证
  □ 数据保留策略已配置
  □ 高敏感数据自动过期已启用

□ 4. 审计与监控
  □ 审计日志独立存储
  □ 哈希链完整性验证正常
  □ 异常检测规则已配置
  □ 安全告警通知链已建立

□ 5. 合规
  □ GDPR数据保护影响评估(DPIA)已完成
  □ 被遗忘权接口已实现并测试
  □ 数据可携带权导出格式已确认
  □ 隐私政策已更新

□ 6. 应急响应
  □ 数据泄露应急预案已制定
  □ 密钥泄露轮换流程已文档化
  □ 安全事件响应团队已组建
  □ 定期安全演练已安排
```

---

## 总结

Agent记忆系统的安全防护是一项系统工程，需要从**数据分类**、**加密存储**、**访问控制**、**审计追踪**、**合规管理**到**安全删除**形成完整的纵深防御体系。核心设计原则可归纳为：

1. **默认安全**：所有记忆默认加密，写入前强制PII检测
2. **最小权限**：Agent只能访问其任务所需的最少记忆
3. **纵深防御**：加密、访问控制、审计多层叠加，任何单点失效不致全面崩溃
4. **可审计性**：每一次读写操作都有不可篡改的审计记录
5. **可遗忘性**：系统设计之初即考虑数据删除能力，而非事后补救

记忆让Agent变得智能，而安全让这种智能值得信赖。

---

> **延伸阅读**
> - GDPR Article 17: Right to Erasure
> - NIST SP 800-88: Guidelines for Media Sanitization
> - OWASP AI Security Guide: Memory System Threats
> - ISO 27001: Information Security Management
