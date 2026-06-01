---
title: "AI应用中的Schema版本管理：让Pydantic模型演进不再破坏生产系统"
description: "深入解析LLM应用中结构化输出的Schema版本管理策略，涵盖向后兼容、灰度迁移、缓存兼容等核心挑战，助你构建可持续演进的AI应用系统。"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: "infra"
tags: ["Schema版本管理", "Pydantic", "LLM应用", "向后兼容", "AI工程化", "版本控制"]
draft: false
---

# AI应用中的Schema版本管理：让Pydantic模型演进不再破坏生产系统

> "你的Pydantic模型改了一个字段名，线上三个服务同时炸了。"

如果你正在用LLM构建结构化输出的生产系统，你一定遇到过这个问题：**随着业务演进，Pydantic模型不可避免地需要修改——加字段、改类型、重命名、删除废弃字段——但每次修改都可能导致上下游服务崩溃。**

这不只是简单的版本管理问题。在AI应用中，Schema变更的影响面远比传统API更大：LLM的输出质量可能因为Schema变化而波动，缓存中的历史数据可能无法反序列化，灰度发布期间新旧模型可能同时运行。这篇文章将深入拆解这些问题的解决方案。

---

## 一、为什么AI应用的Schema管理比传统API更难？

### 1.1 传统API vs LLM应用的Schema管理

```
传统API的Schema管理：
┌──────────┐    ┌──────────────┐    ┌──────────┐
│ 服务A v1  │───▶│ API Gateway  │───▶│ 服务B v1  │
└──────────┘    │  (版本路由)   │    └──────────┘
               └──────────────┘
                │
┌──────────┐    │              │    ┌──────────┐
│ 服务A v2  │───▶│              │───▶│ 服务B v2  │
└──────────┘    └──────────────┘    └──────────┘

LLM应用的Schema管理：
┌──────────┐    ┌──────────────┐    ┌──────────┐
│ Pydantic │───▶│   LLM API    │───▶│ 缓存层   │
│ Model v1 │    │ (OpenAI等)   │    │ (历史数据)│
└──────────┘    └──────────────┘    └──────────┘
      │              │                    │
      ▼              ▼                    ▼
  改了字段名    模型输出质量变化      旧缓存失效
  上游解析失败  新旧Schema不兼容     读取报错
```

### 1.2 AI应用特有的Schema管理挑战

| 挑战 | 传统API | LLM应用 |
|------|---------|---------|
| Schema定义 | Protobuf/JSON Schema | Pydantic模型 |
| 版本管理 | API版本号（v1/v2） | 无标准方案 |
| 向后兼容 | 强制保证 | 难以保证 |
| 缓存兼容 | 结构化存储 | JSON反序列化 |
| 灰度发布 | 流量切换 | 新旧模型并行 |
| 输出质量 | 确定性 | 概率性 |
| 变更频率 | 低（季度级） | 高（周级甚至日级） |

---

## 二、Schema变更的四种类型与应对策略

### 2.1 变更类型分类

```python
# 类型1：新增字段（最安全）
class ExtractResultV1(BaseModel):
    name: str
    age: int

class ExtractResultV2(BaseModel):
    name: str
    age: int
    email: Optional[str] = None  # 新增可选字段

# 类型2：修改字段类型（中等风险）
class ExtractResultV1(BaseModel):
    amount: str  # "1200元"

class ExtractResultV2(BaseModel):
    amount: float  # 1200.0 — 类型变了

# 类型3：重命名字段（高风险）
class ExtractResultV1(BaseModel):
    company_name: str

class ExtractResultV2(BaseModel):
    organization: str  # 改名了！

# 类型4：删除字段（最高风险）
class ExtractResultV1(BaseModel):
    name: str
    deprecated_field: str  # 要删了

class ExtractResultV2(BaseModel):
    name: str  # deprecated_field 没了
```

### 2.2 每种类型的应对策略

```
Schema变更风险矩阵
┌─────────────────────────────────────────────────────────────┐
│                  影响范围                                     │
│         低                    中                    高        │
├─────────────────────────────────────────────────────────────┤
│  概   新增Optional字段    修改字段类型       删除字段        │
│  率   ────────────       重命名字段          重组结构        │
│  高   策略：直接发布      策略：灰度迁移       策略：版本分支  │
│                                                   并行运行   │
│  概                                                      │
│  率   新增Required字段    修改枚举值范围                      │
│  低   策略：加默认值      策略：兼容层                       │
│         灰度发布          逐步切换                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心方案：Pydantic模型的版本化设计

### 3.1 方案一：内置向后兼容层

最实用的方案是在模型内部维护兼容逻辑：

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from datetime import datetime
from enum import Enum

# ====== V1版本 ======
class SentimentV1(BaseModel):
    text: str
    score: float  # -1.0 到 1.0
    label: str    # "positive"/"negative"/"neutral"

# ====== V2版本（改进版）======
class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"  # 新增

class SentimentV2(BaseModel):
    text: str
    score: float
    confidence: float = Field(ge=0, le=1)  # 新增置信度
    label: SentimentLabel  # 从str改为枚举
    keywords: list[str] = Field(default_factory=list)  # 新增
    
    @field_validator('label', mode='before')
    @classmethod
    def normalize_label(cls, v):
        """兼容V1的字符串输入"""
        if isinstance(v, str):
            try:
                return SentimentLabel(v)
            except ValueError:
                # 兼容旧版本中可能出现的非标准值
                mapping = {
                    "pos": SentimentLabel.POSITIVE,
                    "neg": SentimentLabel.NEGATIVE,
                    "neu": SentimentLabel.NEUTRAL,
                    "mixed": SentimentLabel.MIXED,
                }
                return mapping.get(v.lower(), SentimentLabel.NEUTRAL)
        return v

    @field_validator('score', mode='before')
    @classmethod
    def normalize_score(cls, v):
        """兼容V1中可能返回的百分比格式"""
        if isinstance(v, str):
            # "80%" -> 0.8
            v = v.replace('%', '')
            try:
                val = float(v)
                return val / 100 if val > 1 else val
            except ValueError:
                return 0.0
        return v

    class Config:
        # 标记版本信息
        schema_extra = {"version": "2.0", "backward_compatible": True}
```

### 3.2 方案二：版本化适配器模式

更优雅的做法是将兼容逻辑抽离为独立的适配层：

```python
from pydantic import BaseModel
from typing import Any, TypeVar, Generic
from abc import ABC, abstractmethod
import json

T_old = TypeVar('T_old')
T_new = TypeVar('T_new')

class SchemaAdapter(ABC, Generic[T_old, T_new]):
    """Schema版本适配器基类"""
    
    @abstractmethod
    def to_new(self, old_data: dict) -> dict:
        """将旧版本数据转换为新版本"""
        pass
    
    @abstractmethod
    def to_old(self, new_data: dict) -> dict:
        """将新版本数据转换为旧版本（用于回滚）"""
        pass
    
    def migrate(self, old_model: type[T_old], new_model: type[T_new]) -> type[T_new]:
        """创建一个自动迁移的包装器"""
        class MigratedModel(new_model):
            @classmethod
            def from_legacy(cls, data: dict | str | bytes) -> 'MigratedModel':
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                migrated = self.to_new(data)
                return cls(**migrated)
        
        return MigratedModel


# ====== 具体适配器实现 ======

class PersonV1(BaseModel):
    name: str
    age: int
    company: str  # 要重命名为org

class PersonV2(BaseModel):
    name: str
    age: int
    organization: str  # 从company重命名
    department: str = "unknown"  # 新增

class PersonAdapter(SchemaAdapter[PersonV1, PersonV2]):
    
    def to_new(self, old_data: dict) -> dict:
        """V1 → V2"""
        return {
            "name": old_data["name"],
            "age": old_data["age"],
            "organization": old_data.get("company", "unknown"),
            "department": old_data.get("department", "unknown"),
        }
    
    def to_old(self, new_data: dict) -> dict:
        """V2 → V1（回滚用）"""
        return {
            "name": new_data["name"],
            "age": new_data["age"],
            "company": new_data.get("organization", "unknown"),
        }

# 使用适配器
adapter = PersonAdapter()
MigratedPerson = adapter.migrate(PersonV1, PersonV2)

# 从旧版本数据创建新版本对象
old_data = {"name": "张三", "age": 30, "company": "科技公司"}
new_person = MigratedPerson.from_legacy(old_data)
print(new_person.organization)  # "科技公司"
print(new_person.department)    # "unknown"
```

### 3.3 方案三：Schema Registry模式

对于大型系统，可以引入集中式的Schema注册中心：

```python
import hashlib
import json
from typing import Any, Type
from pydantic import BaseModel
from datetime import datetime

class SchemaRegistry:
    """集中式Schema注册中心"""
    
    _schemas: dict[str, dict] = {}
    _adapters: dict[str, callable] = {}
    
    @classmethod
    def register(cls, name: str, schema_class: Type[BaseModel], 
                 description: str = ""):
        """注册一个Schema版本"""
        version_key = f"{name}@{schema_class.__version__ if hasattr(schema_class, '__version__') else 'latest'}"
        
        cls._schemas[version_key] = {
            "class": schema_class,
            "schema_json": schema_class.model_json_schema(),
            "registered_at": datetime.now(),
            "description": description,
        }
        
        # 自动生成schema指纹，用于检测变更
        schema_hash = hashlib.md5(
            json.dumps(schema_class.model_json_schema(), sort_keys=True).encode()
        ).hexdigest()[:8]
        
        cls._schemas[version_key]["hash"] = schema_hash
        print(f"注册Schema: {version_key} (hash: {schema_hash})")
    
    @classmethod
    def register_adapter(cls, from_version: str, to_version: str, 
                         adapter_fn: callable):
        """注册版本适配器"""
        key = f"{from_version}→{to_version}"
        cls._adapters[key] = adapter_fn
    
    @classmethod
    def get_schema(cls, name: str, version: str = "latest") -> Type[BaseModel]:
        """获取指定版本的Schema"""
        key = f"{name}@{version}"
        if key not in cls._schemas:
            raise KeyError(f"Schema {key} 未注册")
        return cls._schemas[key]["class"]
    
    @classmethod
    def migrate(cls, data: dict, from_version: str, to_version: str,
                name: str) -> dict:
        """通过适配器链进行版本迁移"""
        current = data
        current_version = from_version
        
        while current_version != to_version:
            adapter_key = f"{name}@{current_version}→{name}@{to_version}"
            # 也尝试直接跳转
            direct_key = f"{name}@{current_version}→{name}@{to_version}"
            
            if direct_key in cls._adapters:
                current = cls._adapters[direct_key](current)
                current_version = to_version
            else:
                raise ValueError(f"未找到从 {current_version} 到 {to_version} 的适配器")
        
        return current
    
    @classmethod
    def detect_breaking_changes(cls, name: str, 
                                 old_version: str, new_version: str) -> list[str]:
        """检测两个版本之间的破坏性变更"""
        old_key = f"{name}@{old_version}"
        new_key = f"{name}@{new_version}"
        
        if old_key not in cls._schemas or new_key not in cls._schemas:
            return ["版本不存在"]
        
        old_schema = cls._schemas[old_key]["schema_json"]
        new_schema = cls._schemas[new_key]["schema_json"]
        
        breaking_changes = []
        
        old_props = old_schema.get("properties", {})
        new_props = new_schema.get("properties", {})
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))
        
        # 检测字段删除
        deleted = set(old_props.keys()) - set(new_props.keys())
        if deleted:
            breaking_changes.append(f"删除字段: {deleted}")
        
        # 检测新增必填字段
        added_required = new_required - old_required
        if added_required:
            breaking_changes.append(f"新增必填字段: {added_required}")
        
        # 检测类型变更
        for field_name in set(old_props.keys()) & set(new_props.keys()):
            old_type = old_props[field_name].get("type")
            new_type = new_props[field_name].get("type")
            if old_type != new_type:
                breaking_changes.append(
                    f"字段 '{field_name}' 类型变更: {old_type} → {new_type}"
                )
        
        return breaking_changes


# ====== 使用示例 ======

class ReportV1(BaseModel):
    __version__ = "1.0"
    title: str
    content: str
    author: str

class ReportV2(BaseModel):
    __version__ = "2.0"
    title: str
    summary: str  # content重命名为summary
    author: str
    tags: list[str] = []  # 新增

# 注册Schema
SchemaRegistry.register("Report", ReportV1, "V1版本")
SchemaRegistry.register("Report", ReportV2, "V2版本")

# 检测破坏性变更
changes = SchemaRegistry.detect_breaking_changes("Report", "1.0", "2.0")
print(f"破坏性变更: {changes}")
# ['删除字段: {...}', "字段 'content' 类型变更..."]

# 注册适配器
def report_v1_to_v2(data: dict) -> dict:
    return {
        "title": data["title"],
        "summary": data.get("content", ""),
        "author": data["author"],
        "tags": data.get("tags", []),
    }

SchemaRegistry.register_adapter("Report@1.0", "Report@2.0", report_v1_to_v2)

# 自动迁移
old_report = {"title": "月报", "content": "本月完成...", "author": "张三"}
migrated = SchemaRegistry.migrate(old_report, "1.0", "2.0", "Report")
print(migrated)  # {"title": "月报", "summary": "本月完成...", ...}
```

---

## 四、LLM输出Schema的灰度发布

### 4.1 问题：新旧模型并行运行时的Schema不兼容

在灰度发布期间，你可能同时运行两个版本的LLM调用链：

```
灰度期间的复杂状态：
┌─────────────────────────────────────────────────┐
│              Production Traffic                   │
│                    │                              │
│          ┌────────┴────────┐                     │
│          │                 │                     │
│          ▼                 ▼                     │
│    ┌──────────┐     ┌──────────┐                │
│    │ 灰度 10% │     │ 正常 90% │                │
│    │ Model V2 │     │ Model V1 │                │
│    └──────────┘     └──────────┘                │
│          │                 │                     │
│          ▼                 ▼                     │
│    Schema V2         Schema V1                   │
│          │                 │                     │
│          ▼                 ▼                     │
│    ┌──────────────────────────┐                  │
│    │     下游服务（统一消费）   │                  │
│    │  如何同时处理两种Schema？  │                  │
│    └──────────────────────────┘                  │
└─────────────────────────────────────────────────┘
```

### 4.2 解决方案：双Schema兼容层

```python
from pydantic import BaseModel
from typing import Union
import json

class AnalysisV1(BaseModel):
    """V1: 简单分析结果"""
    result: str
    confidence: float

class AnalysisV2(BaseModel):
    """V2: 增强分析结果"""
    result: str
    confidence: float
    reasoning: str          # 新增：推理过程
    alternatives: list[str] # 新增：备选答案
    metadata: dict = {}     # 新增：元数据

class AnalysisUnified(BaseModel):
    """统一Schema：同时兼容V1和V2"""
    result: str
    confidence: float
    reasoning: str = ""          # V1没有时用默认值
    alternatives: list[str] = [] # V1没有时用默认值
    metadata: dict = {}
    schema_version: str = "unknown"
    
    @classmethod
    def from_v1(cls, data: dict) -> 'AnalysisUnified':
        return cls(
            result=data["result"],
            confidence=data["confidence"],
            schema_version="1.0",
        )
    
    @classmethod
    def from_v2(cls, data: dict) -> 'AnalysisUnified':
        return cls(
            result=data["result"],
            confidence=data["confidence"],
            reasoning=data.get("reasoning", ""),
            alternatives=data.get("alternatives", []),
            metadata=data.get("metadata", {}),
            schema_version="2.0",
        )
    
    @classmethod
    def from_raw(cls, raw: dict) -> 'AnalysisUnified':
        """自动检测版本并转换"""
        if "reasoning" in raw or "alternatives" in raw:
            return cls.from_v2(raw)
        return cls.from_v1(raw)


# ====== 使用示例 ======

# 模拟灰度期间的混合数据流
requests = [
    {"result": "正确", "confidence": 0.95},  # V1格式
    {"result": "正确", "confidence": 0.92,   # V2格式
     "reasoning": "基于...", "alternatives": ["错误"]},
    {"result": "不确定", "confidence": 0.4},  # V1格式
]

# 统一处理，无需关心版本
for req in requests:
    unified = AnalysisUnified.from_raw(req)
    print(f"版本: {unified.schema_version}, 结果: {unified.result}")
```

---

## 五、缓存兼容性：历史数据的安全读取

### 5.1 问题：旧缓存数据无法反序列化

```python
import json
from pydantic import BaseModel
from typing import Optional

# V1的缓存数据
cached_data_v1 = json.dumps({
    "name": "张三",
    "tags": "Python,AI",        # V1用逗号分隔字符串
    "score": "85%"              # V1用百分比字符串
})

# V2的模型定义
class UserProfileV2(BaseModel):
    name: str
    tags: list[str]              # V2改为列表
    score: float                 # V2改为浮点数
    
    # 直接反序列化会报错！
    # user = UserProfileV2.model_validate_json(cached_data_v1)  # 💥
```

### 5.2 解决方案：带版本标记的缓存适配层

```python
import json
from pydantic import BaseModel, field_validator
from typing import Any, Optional
import hashlib

class CacheManager:
    """带版本兼容的缓存管理器"""
    
    @staticmethod
    def wrap_with_version(data: dict, schema_name: str, 
                          version: str) -> dict:
        """在缓存数据中嵌入版本信息"""
        return {
            "__schema__": schema_name,
            "__version__": version,
            "__hash__": hashlib.md5(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()[:8],
            "data": data,
        }
    
    @staticmethod
    def unwrap(wrapped: dict, target_model: type[BaseModel],
               adapters: dict[str, callable] = None) -> Any:
        """从缓存中安全提取数据"""
        if "__schema__" not in wrapped:
            # 旧格式（没有版本标记），直接尝试解析
            try:
                return target_model.model_validate(wrapped)
            except Exception:
                return None
        
        cached_version = wrapped["__version__"]
        data = wrapped["data"]
        
        # 检查是否需要迁移
        target_version = getattr(target_model, '__version__', 'latest')
        
        if cached_version != target_version and adapters:
            adapter_key = f"{cached_version}→{target_version}"
            if adapter_key in adapters:
                data = adapters[adapter_key](data)
        
        try:
            return target_model.model_validate(data)
        except Exception as e:
            print(f"缓存反序列化失败: {e}")
            return None


# ====== 带版本兼容的Pydantic模型 ======

class UserProfileV2(BaseModel):
    __version__ = "2.0"
    name: str
    tags: list[str] = []
    score: float = 0.0
    
    @field_validator('tags', mode='before')
    @classmethod
    def normalize_tags(cls, v):
        """兼容V1的逗号分隔字符串"""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v
    
    @field_validator('score', mode='before')
    @classmethod
    def normalize_score(cls, v):
        """兼容V1的百分比格式"""
        if isinstance(v, str):
            v = v.replace('%', '').strip()
            try:
                val = float(v)
                return val / 100 if val > 1 else val
            except ValueError:
                return 0.0
        return v


# ====== 完整的缓存读写流程 ======

cache_mgr = CacheManager()

# 写入缓存（带版本标记）
user_data = {"name": "张三", "tags": ["Python", "AI"], "score": 0.85}
wrapped = cache_mgr.wrap_with_version(user_data, "UserProfile", "2.0")
# wrapped = {"__schema__": "UserProfile", "__version__": "2.0", "data": {...}}

# 读取缓存（自动兼容）
result = cache_mgr.unwrap(wrapped, UserProfileV2)
print(result.name)   # "张三"
print(result.tags)   # ["Python", "AI"]
print(result.score)  # 0.85

# 模拟读取V1缓存（自动转换）
v1_cache = cache_mgr.wrap_with_version(
    {"name": "李四", "tags": "Java,Spring", "score": "92%"},
    "UserProfile", "1.0"
)
result = cache_mgr.unwrap(v1_cache, UserProfileV2)
print(result.tags)   # ["Java", "Spring"] — 自动从字符串转为列表
print(result.score)  # 0.92 — 自动从百分比转为浮点数
```

---

## 六、生产实践：完整的版本管理流程

### 6.1 Schema变更的完整流程

```
Schema变更生命周期
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  1. 设计阶段                                             │
│  ┌─────────────────────────────────────────┐            │
│  │ • 定义新版本Schema                       │            │
│  │ • 编写兼容性验证测试                      │            │
│  │ • 评估破坏性变更影响面                    │            │
│  │ • 编写版本适配器                          │            │
│  └─────────────────────┬───────────────────┘            │
│                        ▼                                 │
│  2. 注册阶段                                             │
│  ┌─────────────────────────────────────────┐            │
│  │ • 注册到Schema Registry                  │            │
│  │ • 生成版本间的适配器                      │            │
│  │ • 更新下游服务的依赖声明                  │            │
│  └─────────────────────┬───────────────────┘            │
│                        ▼                                 │
│  3. 灰度阶段                                             │
│  ┌─────────────────────────────────────────┐            │
│  │ • 5%流量使用新Schema                     │            │
│  │ • 监控反序列化错误率                      │            │
│  │ • 监控LLM输出质量变化                    │            │
│  │ • 验证缓存兼容性                          │            │
│  └─────────────────────┬───────────────────┘            │
│                        ▼                                 │
│  4. 全量阶段                                             │
│  ┌─────────────────────────────────────────┐            │
│  │ • 100%切换到新Schema                     │            │
│  │ • 保留旧版本适配器（回滚用）              │            │
│  │ • 清理过期缓存数据                        │            │
│  │ • 更新Schema Registry文档                │            │
│  └─────────────────────────────────────────┘            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 自动化Schema兼容性测试

```python
import pytest
from pydantic import BaseModel
from typing import get_type_hints
import json

class SchemaCompatibilityTest:
    """Schema兼容性自动化测试框架"""
    
    @staticmethod
    def test_adding_optional_field_is_safe(old_model: type, 
                                            new_model: type,
                                            field_name: str,
                                            field_type: type,
                                            default_value: Any):
        """测试：新增Optional字段是安全的"""
        # 构造旧版本数据
        old_instance = old_model.model_validate(
            SchemaCompatibilityTest._generate_sample(old_model)
        )
        old_dict = old_instance.model_dump()
        
        # 新版本应该能直接接受旧数据
        new_instance = new_model.model_validate(old_dict)
        
        # 新字段应该有默认值
        assert getattr(new_instance, field_name) == default_value
    
    @staticmethod
    def test_field_rename_backward_compat(old_model: type,
                                           new_model: type,
                                           old_field: str,
                                           new_field: str,
                                           adapter_fn: callable):
        """测试：字段重命名的向后兼容"""
        old_sample = SchemaCompatibilityTest._generate_sample(old_model)
        old_instance = old_model.model_validate(old_sample)
        old_dict = old_instance.model_dump()
        
        # 通过适配器转换
        migrated = adapter_fn(old_dict)
        new_instance = new_model.model_validate(migrated)
        
        # 验证数据完整迁移
        old_value = getattr(old_instance, old_field)
        new_value = getattr(new_instance, new_field)
        assert old_value == new_value
    
    @staticmethod
    def test_type_change_coercion(old_model: type,
                                    new_model: type,
                                    field_name: str,
                                    old_value: Any,
                                    expected_new: Any):
        """测试：类型变更的自动转换"""
        # 模拟旧版本缓存数据
        old_sample = SchemaCompatibilityTest._generate_sample(old_model)
        old_sample[field_name] = old_value
        
        # 新版本应该能自动转换
        new_instance = new_model.model_validate(old_sample)
        actual = getattr(new_instance, field_name)
        
        assert actual == expected_new
    
    @staticmethod
    def _generate_sample(model: type) -> dict:
        """根据模型定义生成示例数据"""
        sample = {}
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            if annotation == str:
                sample[field_name] = f"sample_{field_name}"
            elif annotation == int:
                sample[field_name] = 42
            elif annotation == float:
                sample[field_name] = 3.14
            elif annotation == bool:
                sample[field_name] = True
            elif hasattr(annotation, '__origin__'):
                sample[field_name] = []
            else:
                sample[field_name] = None
        return sample


# ====== 运行兼容性测试 ======

class UserV1(BaseModel):
    name: str
    email: str

class UserV2(BaseModel):
    name: str
    email: str
    phone: str = ""  # 新增可选字段

# 测试1：新增字段是安全的
SchemaCompatibilityTest.test_adding_optional_field_is_safe(
    old_model=UserV1,
    new_model=UserV2,
    field_name="phone",
    field_type=str,
    default_value=""
)
print("✅ 新增Optional字段测试通过")

# 测试2：缓存数据兼容性
cached_v1 = {"name": "张三", "email": "z@test.com"}
user_v2 = UserV2.model_validate(cached_v1)
assert user_v2.phone == ""  # 自动使用默认值
print("✅ 缓存兼容性测试通过")
```

---

## 七、最佳实践总结

### 7.1 Schema设计原则

```
AI应用Schema设计的五条铁律
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  1. 新增字段一律用Optional + 默认值                       │
│     ❌ department: str                                   │
│     ✅ department: str = "unknown"                       │
│                                                          │
│  2. 永远不要删除字段，只标记废弃                           │
│     ❌ 直接从Model中移除字段                              │
│     ✅ deprecated_field: str = Field(deprecated=True)    │
│                                                          │
│  3. 字段重命名必须保留旧字段至少一个版本周期               │
│     ❌ company → organization（直接改）                   │
│     ✅ company: str = Field(deprecated=True) +           │
│        organization: str                                │
│                                                          │
│  4. 类型变更必须向后兼容                                  │
│     ❌ amount: str → amount: float（直接改）              │
│     ✅ amount: float = Field(json_schema_extra=          │
│        {"coerce": ["str→float"]})                        │
│                                                          │
│  5. 每次Schema变更都要有对应的测试用例                     │
│     ❌ 改了就发，不写测试                                 │
│     ✅ test_v1_to_v2_migration()                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Schema版本管理检查清单

在发布Schema变更前，逐项检查：

```markdown
## Schema变更发布检查清单

### 设计阶段
- [ ] 新版本Schema已定义
- [ ] 破坏性变更已评估
- [ ] 兼容性测试已编写
- [ ] 适配器已实现（如需要）

### 注册阶段
- [ ] 新版本已注册到Registry
- [ ] 旧版本仍保留（未删除）
- [ ] 下游服务已通知
- [ ] 缓存策略已更新

### 灰度阶段
- [ ] 5%流量已切换
- [ ] 反序列化错误率 < 0.1%
- [ ] LLM输出质量无显著波动
- [ ] 缓存命中率正常

### 全量阶段
- [ ] 100%流量已切换
- [ ] 旧版本适配器保留（至少2个发布周期）
- [ ] 过期缓存已清理
- [ ] 文档已更新
- [ ] 监控告警已配置
```

---

## 八、总结

AI应用的Schema版本管理是一个被严重低估的工程问题。与传统API不同，LLM应用中的Schema变更还涉及到：

1. **LLM输出质量的变化**：Schema变更可能影响模型的生成行为
2. **缓存兼容性**：历史数据的安全反序列化
3. **灰度发布**：新旧模型并行运行时的Schema共存
4. **概率性输出**：同一个Schema，模型可能返回不同的结构

核心策略总结：

| 策略 | 适用场景 | 复杂度 | 推荐度 |
|------|---------|--------|--------|
| 内置兼容层 | 小型项目，变更少 | 低 | ⭐⭐⭐⭐ |
| 适配器模式 | 中型项目，多版本并存 | 中 | ⭐⭐⭐⭐⭐ |
| Schema Registry | 大型项目，多团队协作 | 高 | ⭐⭐⭐⭐ |
| 统一Schema | 灰度发布期间 | 中 | ⭐⭐⭐⭐⭐ |

**最终建议：** 在项目初期就引入Schema版本管理机制。等到线上出了问题再补救，成本是事前预防的10倍以上。

---

**相关资源：**
- Pydantic V2文档：https://docs.pydantic.dev
- Instructor结构化输出：https://python.useinstructor.com
- OpenAI Structured Outputs：https://platform.openai.com/docs/guides/structured-outputs
- JSON Schema规范：https://json-schema.org
