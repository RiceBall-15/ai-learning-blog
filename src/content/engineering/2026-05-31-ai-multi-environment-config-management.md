---
title: "AI应用多环境一致性保障：从开发到生产的配置管理工程实践"
description: "系统讲解AI应用多环境配置管理的工程实践，涵盖Prompt版本化、模型参数管理、环境隔离策略与配置热更新的完整方案"
date: 2026-05-31
author: "RiceBall"
category: "engineering"
subCategory: infra
tags: ["AI工程化", "配置管理", "环境一致性", "Prompt版本化", "DevOps", "LLM运维"]
draft: false
---

## 引言：AI应用的配置管理，远比你想的复杂

在传统Web应用中，配置管理的核心挑战是"数据库连接串在不同环境是否正确"。但在AI应用中，配置管理的复杂度发生了质变——你不仅要管理传统的环境变量，还要管理：

- **Prompt模板**：一个词的改动可能让输出质量从90分掉到60分
- **模型参数**：temperature、top_p、max_tokens这些参数的组合有无穷可能
- **RAG配置**：向量库连接、检索策略、重排序参数、上下文窗口大小
- **安全规则**：内容过滤的敏感词列表、输出格式约束、角色设定
- **降级策略**：主力模型不可用时的备选方案和回退逻辑

更致命的是，AI应用的很多配置**没有"对错"之分，只有"好坏"之别**。一个Prompt在开发环境测试效果很好，到了生产环境面对真实用户的多样化输入就可能翻车。传统的"配置文件拷贝"策略完全无法应对这种场景。

```
┌─────────────────────────────────────────────────────────────────┐
│               AI 应用配置管理的复杂度矩阵                        │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│  配置类型      │  变更频率      │  影响范围      │  回滚难度        │
├───────────────┼───────────────┼───────────────┼─────────────────┤
│  环境变量      │  低（部署时）   │  全局         │  ⭐              │
│  模型参数      │  中（调优时）   │  单模型/链路   │  ⭐⭐             │
│  Prompt模板   │  高（持续迭代） │  业务功能级    │  ⭐⭐⭐            │
│  安全规则      │  中（合规要求） │  全链路       │  ⭐⭐⭐⭐           │
│  RAG配置      │  低-中         │  检索链路      │  ⭐⭐⭐            │
│  降级策略      │  低（架构调整） │  整体可用性    │  ⭐⭐⭐⭐⭐          │
└───────────────┴───────────────┴───────────────┴─────────────────┘
```

本文将从工程实践出发，系统性地讲解AI应用的多环境配置管理方案。

---

## 一、配置分层：AI应用的配置架构设计

### 1.1 配置分层模型

AI应用的配置应该分为四个层次，每层有不同的管理策略：

```
┌─────────────────────────────────────────────────────────┐
│                    配置分层架构                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Layer 4: 业务配置（高变更频率）                          │
│  ├── Prompt模板（版本化管理）                             │
│  ├── 输出格式约束                                        │
│  └── 业务规则参数                                        │
│                                                         │
│  Layer 3: AI模型配置（中变更频率）                         │
│  ├── 模型选择与路由策略                                   │
│  ├── 推理参数（temperature, top_p等）                     │
│  ├── Token预算与限流参数                                  │
│  └── 降级与熔断策略                                      │
│                                                         │
│  Layer 2: RAG配置（低-中变更频率）                         │
│  ├── 向量库连接与检索策略                                  │
│  ├── 重排序模型配置                                       │
│  ├── 上下文窗口管理                                      │
│  └── 文档分块策略                                        │
│                                                         │
│  Layer 1: 基础设施配置（低变更频率）                       │
│  ├── 数据库/缓存连接串                                   │
│  ├── API密钥与认证                                       │
│  ├── 网络与部署配置                                      │
│  └── 日志与监控配置                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 配置数据模型

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import hashlib
import json
import time


class ConfigLayer(Enum):
    """配置层级"""
    INFRASTRUCTURE = "infrastructure"   # L1: 基础设施
    RAG = "rag"                         # L2: RAG配置
    MODEL = "model"                     # L3: AI模型
    BUSINESS = "business"               # L4: 业务配置


class ConfigScope(Enum):
    """配置作用域"""
    GLOBAL = "global"       # 全局生效
    ENVIRONMENT = "env"     # 环境级别
    SERVICE = "service"     # 服务级别
    FEATURE = "feature"     # 功能级别
    USER = "user"           # 用户级别


@dataclass
class AIConfigItem:
    """AI应用配置项"""
    key: str
    value: Any
    layer: ConfigLayer
    scope: ConfigScope
    
    # 版本与审计
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    author: str = ""
    description: str = ""
    
    # 安全与合规
    encrypted: bool = False
    sensitive: bool = False         # 是否为敏感配置（如API密钥）
    requires_approval: bool = False # 生产环境是否需要审批
    
    # 标签
    tags: list[str] = field(default_factory=list)
    
    @property
    def content_hash(self) -> str:
        """配置内容的哈希值，用于变更检测"""
        content = json.dumps({
            "key": self.key,
            "value": self.value,
            "version": self.version
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "layer": self.layer.value,
            "scope": self.scope.value,
            "version": self.version,
            "content_hash": self.content_hash,
            "updated_at": self.updated_at,
        }


@dataclass
class PromptConfig(AIConfigItem):
    """Prompt专用配置"""
    template: str = ""
    variables: dict[str, str] = field(default_factory=dict)
    model_constraints: list[str] = field(default_factory=list)  # 适用的模型列表
    quality_score: float = 0.0      # Prompt质量评分
    ab_test_group: str = ""         # A/B测试分组
    max_tokens_output: int = 2048   # 最大输出Token数
    
    # Prompt安全约束
    system_message: str = ""
    output_format: str = "text"     # text, json, markdown
    content_filter_level: str = "standard"  # none, standard, strict
```

### 1.3 配置合并策略

多环境配置管理的核心挑战是：**如何在不同环境之间共享基础配置，同时允许环境特定的覆盖**。

```python
from typing import Optional
import copy


class ConfigMerger:
    """AI应用配置合并引擎"""
    
    # 配置优先级：用户级 > 功能级 > 服务级 > 环境级 > 全局级
    SCOPE_PRIORITY = {
        ConfigScope.USER: 100,
        ConfigScope.FEATURE: 80,
        ConfigScope.SERVICE: 60,
        ConfigScope.ENVIRONMENT: 40,
        ConfigScope.GLOBAL: 20,
    }
    
    def __init__(self):
        # 配置存储：scope -> key -> ConfigItem
        self.configs: dict[str, dict[str, AIConfigItem]] = {}
        # 层级默认值
        self.layer_defaults: dict[ConfigLayer, dict[str, Any]] = {}
    
    def register_default(self, layer: ConfigLayer, defaults: dict[str, Any]):
        """注册层级默认值"""
        self.layer_defaults[layer] = defaults
    
    def set_config(self, item: AIConfigItem, scope: ConfigScope):
        """设置配置项"""
        scope_key = scope.value
        if scope_key not in self.configs:
            self.configs[scope_key] = {}
        
        # 检查是否需要审批
        if item.requires_approval and scope == ConfigScope.ENVIRONMENT:
            item.tags.append("pending_approval")
        
        self.configs[scope_key][item.key] = item
    
    def get_config(
        self, 
        key: str, 
        environment: str = "production",
        service: str = "",
        feature: str = "",
        user_id: str = ""
    ) -> Optional[AIConfigItem]:
        """按优先级获取配置"""
        
        # 构建作用域搜索列表（优先级从高到低）
        search_scopes = [ConfigScope.GLOBAL]
        
        if environment:
            search_scopes.append(ConfigScope.ENVIRONMENT)
        if service:
            search_scopes.append(ConfigScope.SERVICE)
        if feature:
            search_scopes.append(ConfigScope.FEATURE)
        if user_id:
            search_scopes.append(ConfigScope.USER)
        
        # 按优先级排序
        search_scopes.sort(
            key=lambda s: self.SCOPE_PRIORITY[s], 
            reverse=True
        )
        
        # 查找配置
        for scope in search_scopes:
            scope_configs = self.configs.get(scope.value, {})
            if key in scope_configs:
                item = scope_configs[key]
                # 跳过待审批的配置
                if "pending_approval" in item.tags:
                    continue
                return item
        
        return None
    
    def get_effective_config(
        self,
        layer: ConfigLayer,
        environment: str = "production",
        **kwargs
    ) -> dict[str, Any]:
        """获取某个层级在特定环境下的所有有效配置"""
        
        defaults = self.layer_defaults.get(layer, {})
        result = copy.deepcopy(defaults)
        
        # 从全局到具体，逐层覆盖
        for scope in [ConfigScope.GLOBAL, ConfigScope.ENVIRONMENT, 
                       ConfigScope.SERVICE, ConfigScope.FEATURE]:
            scope_configs = self.configs.get(scope.value, {})
            for key, item in scope_configs.items():
                if item.layer == layer and "pending_approval" not in item.tags:
                    result[key] = item.value
        
        return result
```

---

## 二、Prompt版本化管理：像管理代码一样管理Prompt

### 2.1 为什么Prompt需要版本管理？

Prompt是AI应用最核心的资产之一，但大多数团队的Prompt管理还停留在"改了就存、存了就忘"的原始阶段。一个典型的灾难场景：

> 工程师A优化了客服机器人的Prompt，上线效果很好。三天后工程师B觉得某个回复风格不好，改了一版。一周后工程师C发现输出格式不对，又改了一版。两周后产品经理说"还是第一版好"——但没有人知道第一版的Prompt长什么样了。

Prompt版本管理要解决三个核心问题：**谁改了什么**、**改了效果如何**、**能不能回滚**。

### 2.2 Prompt版本管理模型

```python
import hashlib
from dataclasses import dataclass, field
from enum import Enum


class PromptStatus(Enum):
    DRAFT = "draft"           # 草稿
    TESTING = "testing"       # 测试中
    ACTIVE = "active"         # 生产使用
    DEPRECATED = "deprecated" # 已废弃
    ARCHIVED = "archived"     # 已归档


@dataclass
class PromptVersion:
    """Prompt版本记录"""
    prompt_id: str             # Prompt标识符
    version: int               # 版本号
    template: str              # Prompt模板内容
    variables: dict[str, str]  # 变量定义
    
    # 元信息
    status: PromptStatus = PromptStatus.DRAFT
    author: str = ""
    description: str = ""
    changelog: str = ""
    
    # 模型约束
    compatible_models: list[str] = field(default_factory=list)
    min_model_capability: str = "standard"  # standard, advanced, reasoning
    
    # 质量评估
    quality_metrics: dict[str, float] = field(default_factory=dict)
    # {
    #     "relevance": 0.85,
    #     "coherence": 0.92,
    #     "safety": 0.98,
    #     "format_compliance": 0.95,
    # }
    
    # 时间戳
    created_at: float = 0.0
    activated_at: float = 0.0
    
    # A/B测试
    ab_test_config: Optional[dict] = None
    # {
    #     "experiment_id": "exp_20260531_001",
    #     "traffic_ratio": 0.1,
    #     "target_metric": "user_satisfaction",
    #     "control_version": 3,
    # }
    
    @property
    def content_hash(self) -> str:
        content = f"{self.template}:{json.dumps(self.variables, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class PromptManager:
    """Prompt版本管理器"""
    
    def __init__(self):
        # prompt_id -> list[PromptVersion]
        self.versions: dict[str, list[PromptVersion]] = {}
        # prompt_id -> active_version
        self.active_versions: dict[str, int] = {}
    
    def create_prompt(
        self, 
        prompt_id: str, 
        template: str, 
        variables: dict[str, str],
        author: str,
        description: str = ""
    ) -> PromptVersion:
        """创建新Prompt"""
        if prompt_id in self.versions:
            raise ValueError(f"Prompt '{prompt_id}' already exists")
        
        version = PromptVersion(
            prompt_id=prompt_id,
            version=1,
            template=template,
            variables=variables,
            author=author,
            description=description,
        )
        self.versions[prompt_id] = [version]
        return version
    
    def update_prompt(
        self,
        prompt_id: str,
        template: str,
        variables: dict[str, str],
        author: str,
        changelog: str = ""
    ) -> PromptVersion:
        """更新Prompt（创建新版本）"""
        if prompt_id not in self.versions:
            raise ValueError(f"Prompt '{prompt_id}' not found")
        
        existing = self.versions[prompt_id]
        new_version_num = len(existing) + 1
        
        # 检查内容是否有变化
        latest = existing[-1]
        if (template == latest.template and 
            variables == latest.variables):
            return latest  # 没有变化，返回当前版本
        
        new_version = PromptVersion(
            prompt_id=prompt_id,
            version=new_version_num,
            template=template,
            variables=variables,
            author=author,
            changelog=changelog,
            compatible_models=latest.compatible_models.copy(),
        )
        
        self.versions[prompt_id].append(new_version)
        return new_version
    
    def activate_version(self, prompt_id: str, version: int) -> bool:
        """激活指定版本"""
        if prompt_id not in self.versions:
            return False
        
        for v in self.versions[prompt_id]:
            if v.version == version:
                if v.status not in (PromptStatus.TESTING, PromptStatus.DRAFT):
                    v.status = PromptStatus.ACTIVE
                    self.active_versions[prompt_id] = version
                    
                    # 将之前活跃的版本降级
                    for old_v in self.versions[prompt_id]:
                        if old_v.version != version and old_v.status == PromptStatus.ACTIVE:
                            old_v.status = PromptStatus.DEPRECATED
                    
                    return True
        return False
    
    def rollback(self, prompt_id: str) -> Optional[PromptVersion]:
        """回滚到上一个版本"""
        if prompt_id not in self.versions:
            return None
        
        current_version = self.active_versions.get(prompt_id)
        if not current_version:
            return None
        
        # 找到上一个活跃过的版本
        versions = self.versions[prompt_id]
        for v in reversed(versions):
            if v.version < current_version and v.status == PromptStatus.DEPRECATED:
                self.activate_version(prompt_id, v.version)
                return v
        
        return None
    
    def get_active(self, prompt_id: str) -> Optional[PromptVersion]:
        """获取当前活跃版本"""
        version_num = self.active_versions.get(prompt_id)
        if not version_num:
            return None
        
        for v in self.versions.get(prompt_id, []):
            if v.version == version_num:
                return v
        return None
    
    def render(
        self, 
        prompt_id: str, 
        variables: dict[str, str],
        model: str = ""
    ) -> Optional[str]:
        """渲染Prompt模板"""
        version = self.get_active(prompt_id)
        if not version:
            return None
        
        # 检查模型兼容性
        if model and version.compatible_models:
            if model not in version.compatible_models:
                raise ValueError(
                    f"Model '{model}' not compatible with prompt '{prompt_id}' "
                    f"v{version.version}. Compatible: {version.compatible_models}"
                )
        
        # 渲染模板
        template = version.template
        for var_name, var_value in {**version.variables, **variables}.items():
            template = template.replace(f"{{{var_name}}}", var_value)
        
        return template
    
    def get_version_history(self, prompt_id: str) -> list[dict]:
        """获取版本历史"""
        versions = self.versions.get(prompt_id, [])
        return [
            {
                "version": v.version,
                "status": v.status.value,
                "author": v.author,
                "changelog": v.changelog,
                "quality_metrics": v.quality_metrics,
                "created_at": v.created_at,
            }
            for v in versions
        ]
```

### 2.3 Prompt A/B测试框架

```python
import random
import time
from dataclasses import dataclass


@dataclass
class ABTestExperiment:
    """A/B测试实验"""
    experiment_id: str
    prompt_id: str
    control_version: int       # 对照组版本
    treatment_version: int     # 实验组版本
    traffic_ratio: float       # 实验组流量比例（0-1）
    target_metric: str         # 目标指标
    min_sample_size: int = 100
    start_time: float = 0.0
    end_time: float = 0.0
    
    # 实时统计
    control_impressions: int = 0
    control_conversions: int = 0
    treatment_impressions: int = 0
    treatment_conversions: int = 0


class PromptABTestManager:
    """Prompt A/B测试管理器"""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.experiments: dict[str, ABTestExperiment] = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        prompt_id: str,
        control_version: int,
        treatment_version: int,
        traffic_ratio: float,
        target_metric: str,
        min_sample_size: int = 100,
    ) -> ABTestExperiment:
        """创建A/B测试实验"""
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            prompt_id=prompt_id,
            control_version=control_version,
            treatment_version=treatment_version,
            traffic_ratio=traffic_ratio,
            target_metric=target_metric,
            min_sample_size=min_sample_size,
            start_time=time.time(),
        )
        self.experiments[experiment_id] = experiment
        return experiment
    
    def assign_version(
        self, 
        experiment_id: str, 
        user_id: str
    ) -> int:
        """为用户分配实验版本（一致性哈希确保同一用户始终看到同一版本）"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return 0
        
        # 使用用户ID的哈希值确定分组
        hash_val = int(hashlib.md5(
            f"{experiment_id}:{user_id}".encode()
        ).hexdigest(), 16) % 100
        
        if hash_val < experiment.traffic_ratio * 100:
            return experiment.treatment_version
        return experiment.control_version
    
    def record_outcome(
        self,
        experiment_id: str,
        user_id: str,
        version: int,
        conversion: bool
    ):
        """记录实验结果"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return
        
        if version == experiment.control_version:
            experiment.control_impressions += 1
            if conversion:
                experiment.control_conversions += 1
        else:
            experiment.treatment_impressions += 1
            if conversion:
                experiment.treatment_conversions += 1
    
    def analyze(self, experiment_id: str) -> dict:
        """分析实验结果"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        # 计算转化率
        control_rate = (
            experiment.control_conversions / experiment.control_impressions
            if experiment.control_impressions > 0 else 0
        )
        treatment_rate = (
            experiment.treatment_conversions / experiment.treatment_impressions
            if experiment.treatment_impressions > 0 else 0
        )
        
        # 提升幅度
        lift = (
            (treatment_rate - control_rate) / control_rate * 100
            if control_rate > 0 else 0
        )
        
        # 样本量检查
        total_impressions = (
            experiment.control_impressions + experiment.treatment_impressions
        )
        sufficient_sample = total_impressions >= experiment.min_sample_size * 2
        
        # 简化的显著性判断（实际应使用统计检验）
        significant = (
            sufficient_sample and 
            abs(lift) > 5  # 5%以上的提升视为显著
        )
        
        return {
            "experiment_id": experiment_id,
            "control": {
                "version": experiment.control_version,
                "impressions": experiment.control_impressions,
                "conversions": experiment.control_conversions,
                "rate": f"{control_rate:.2%}",
            },
            "treatment": {
                "version": experiment.treatment_version,
                "impressions": experiment.treatment_impressions,
                "conversions": experiment.treatment_conversions,
                "rate": f"{treatment_rate:.2%}",
            },
            "lift": f"{lift:+.1f}%",
            "significant": significant,
            "sufficient_sample": sufficient_sample,
            "recommendation": (
                "建议采用实验组" if significant and lift > 0
                else "建议保留对照组" if significant and lift < 0
                else "需要更多数据"
            ),
        }
```

---

## 三、环境隔离策略：开发、测试、生产的三层防线

### 3.1 环境配置隔离矩阵

| 配置项 | 开发环境 | 测试环境 | 预发环境 | 生产环境 |
|-------|---------|---------|---------|---------|
| **LLM模型** | gpt-4o-mini | gpt-4o | gpt-4o | gpt-4o + 备用 |
| **Prompt版本** | 最新开发版 | 测试版 | 预发布版 | 稳定版 |
| **Temperature** | 0.7（高探索） | 0.5（平衡） | 0.3（保守） | 0.3（保守） |
| **日志级别** | DEBUG | INFO | INFO | WARN |
| **限流策略** | 无 | 模拟限流 | 与生产一致 | 完整限流 |
| **降级策略** | 直接报错 | 记录但继续 | 与生产一致 | 完整降级链 |
| **安全过滤** | 关闭 | 基础过滤 | 与生产一致 | 完整过滤 |
| **缓存** | 关闭 | Redis | Redis | Redis集群 |

### 3.2 环境感知的配置加载器

```python
import os
import json
from pathlib import Path
from typing import Optional


class EnvironmentConfigLoader:
    """环境感知的AI应用配置加载器"""
    
    # 环境检测顺序：环境变量 > 部署标记 > 默认值
    ENVIRONMENT_SOURCES = [
        lambda: os.getenv("AI_APP_ENV"),           # 环境变量
        lambda: os.getenv("DEPLOY_ENV"),           # 部署平台变量
        lambda: self._detect_from_k8s(),           # Kubernetes namespace
        lambda: "development",                      # 默认值
    ]
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.environment = self._detect_environment()
        self._config_cache: dict[str, dict] = {}
    
    def _detect_environment(self) -> str:
        """检测当前环境"""
        for source in self.ENVIRONMENT_SOURCES:
            try:
                env = source()
                if env:
                    return env.lower()
            except Exception:
                continue
        return "development"
    
    def _detect_from_k8s(self) -> Optional[str]:
        """从Kubernetes namespace推断环境"""
        try:
            namespace_file = Path("/var/run/secrets/kubernetes.io/namespace")
            if namespace_file.exists():
                namespace = namespace_file.read_text().strip()
                if "prod" in namespace:
                    return "production"
                elif "staging" in namespace:
                    return "staging"
                elif "dev" in namespace:
                    return "development"
        except Exception:
            pass
        return None
    
    def load_config(self, config_name: str) -> dict:
        """加载配置文件（带环境覆盖）"""
        cache_key = f"{self.environment}:{config_name}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # 1. 加载基础配置
        base_config = self._load_file(f"{config_name}.json")
        
        # 2. 加载环境特定覆盖
        env_config = self._load_file(
            f"{config_name}.{self.environment}.json"
        )
        
        # 3. 合并（环境配置覆盖基础配置）
        merged = self._deep_merge(base_config, env_config)
        
        # 4. 处理环境变量替换
        merged = self._resolve_env_vars(merged)
        
        self._config_cache[cache_key] = merged
        return merged
    
    def _load_file(self, filename: str) -> dict:
        """加载JSON配置文件"""
        filepath = self.config_dir / self.environment / filename
        if not filepath.exists():
            filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        
        try:
            return json.loads(filepath.read_text())
        except json.JSONDecodeError:
            return {}
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """深度合并配置"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _resolve_env_vars(self, config: dict) -> dict:
        """解析配置中的环境变量引用"""
        result = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                result[key] = os.getenv(env_var, value)
            elif isinstance(value, dict):
                result[key] = self._resolve_env_vars(value)
            else:
                result[key] = value
        return result
    
    def get_ai_config(self) -> dict:
        """获取AI相关配置"""
        return self.load_config("ai-config")
    
    def get_prompt_config(self, prompt_id: str) -> dict:
        """获取Prompt配置"""
        prompts = self.load_config("prompts")
        return prompts.get(prompt_id, {})
    
    def get_model_config(self, model_name: str) -> dict:
        """获取模型配置"""
        models = self.load_config("models")
        return models.get(model_name, {})
```

### 3.3 配置热更新机制

AI应用的配置变更不应该依赖重启服务。尤其是Prompt模板和安全规则，可能需要在几分钟内完成热更新：

```python
import asyncio
import json
import time
from typing import Callable, Any


class ConfigHotReloader:
    """配置热更新管理器"""
    
    def __init__(self, poll_interval: float = 30.0):
        self.poll_interval = poll_interval
        self.watchers: dict[str, dict] = {}
        self._running = False
        
        # 配置变更回调
        self.callbacks: dict[str, list[Callable]] = {}
    
    def watch(
        self, 
        config_key: str, 
        callback: Callable[[str, Any, Any], None]
    ):
        """注册配置变更监听"""
        if config_key not in self.callbacks:
            self.callbacks[config_key] = []
        self.callbacks[config_key].append(callback)
    
    async def start_watching(self):
        """启动配置轮询"""
        self._running = True
        while self._running:
            try:
                await self._check_updates()
            except Exception as e:
                print(f"Config watch error: {e}")
            await asyncio.sleep(self.poll_interval)
    
    async def _check_updates(self):
        """检查配置更新"""
        for key, watcher in self.watchers.items():
            try:
                current_hash = self._get_config_hash(key)
                if current_hash != watcher.get("last_hash"):
                    old_value = watcher.get("value")
                    new_value = self._load_config_value(key)
                    
                    # 更新缓存
                    watcher["last_hash"] = current_hash
                    watcher["value"] = new_value
                    
                    # 触发回调
                    if key in self.callbacks:
                        for callback in self.callbacks[key]:
                            try:
                                callback(key, old_value, new_value)
                            except Exception as e:
                                print(f"Callback error for {key}: {e}")
            except Exception as e:
                print(f"Error checking {key}: {e}")
    
    def _get_config_hash(self, key: str) -> str:
        """获取配置的当前哈希值"""
        # 实际实现中，应该从配置存储获取
        # 这里简化为从文件读取
        try:
            with open(f"/config/{key}.json", "r") as f:
                content = f.read()
                return hashlib.md5(content.encode()).hexdigest()
        except FileNotFoundError:
            return ""
    
    def _load_config_value(self, key: str) -> Any:
        """加载配置值"""
        try:
            with open(f"/config/{key}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None


class AIConfigHotReloadManager:
    """AI应用配置热更新管理器"""
    
    def __init__(self):
        self.reloader = ConfigHotReloader(poll_interval=15.0)
        self.prompt_manager: Optional[PromptManager] = None
        
        # 注册配置变更回调
        self.reloader.watch("prompts", self._on_prompt_change)
        self.reloader.watch("models", self._on_model_config_change)
        self.reloader.watch("safety_rules", self._on_safety_rules_change)
    
    def _on_prompt_change(self, key: str, old_val: Any, new_val: Any):
        """Prompt配置变更回调"""
        if not self.prompt_manager:
            return
        
        if old_val and new_val:
            # 检查哪些Prompt发生了变化
            old_prompts = old_val if isinstance(old_val, dict) else {}
            new_prompts = new_val if isinstance(new_val, dict) else {}
            
            for prompt_id, new_content in new_prompts.items():
                old_content = old_prompts.get(prompt_id)
                if old_content != new_content:
                    print(f"Hot-reloading prompt: {prompt_id}")
                    # 自动创建新版本
                    self.prompt_manager.update_prompt(
                        prompt_id=prompt_id,
                        template=new_content.get("template", ""),
                        variables=new_content.get("variables", {}),
                        author="hot-reload",
                        changelog="Auto-reloaded from config"
                    )
    
    def _on_model_config_change(self, key: str, old_val: Any, new_val: Any):
        """模型配置变更回调"""
        print(f"Model config changed, updating routing rules...")
        # 更新模型路由规则
    
    def _on_safety_rules_change(self, key: str, old_val: Any, new_val: Any):
        """安全规则变更回调"""
        print(f"Safety rules updated, reloading filters...")
        # 重新加载安全过滤规则
    
    async def start(self):
        """启动热更新"""
        await self.reloader.start_watching()
```

---

## 四、配置审计与回滚：AI应用的安全网

### 4.1 配置变更审计

每一次配置变更都应该被记录，尤其是涉及Prompt和安全规则的变更：

```python
from dataclasses import dataclass, field
from enum import Enum
import uuid


class ConfigChangeType(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ROLLBACK = "rollback"
    ACTIVATE = "activate"


@dataclass
class ConfigAuditRecord:
    """配置变更审计记录"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    change_type: ConfigChangeType = ConfigChangeType.UPDATE
    config_key: str = ""
    environment: str = ""
    
    # 变更内容
    old_value: Any = None
    new_value: Any = None
    old_hash: str = ""
    new_hash: str = ""
    
    # 变更人
    operator: str = ""
    reason: str = ""
    approved_by: str = ""
    
    # 时间
    timestamp: float = field(default_factory=time.time)
    
    # 关联
    related_ticket: str = ""
    related_pr: str = ""


class ConfigAuditLogger:
    """配置审计日志"""
    
    def __init__(self, audit_log_path: str = "./config_audit.log"):
        self.audit_log_path = audit_log_path
        self.records: list[ConfigAuditRecord] = []
    
    def log_change(self, record: ConfigAuditRecord):
        """记录配置变更"""
        self.records.append(record)
        
        # 写入审计日志
        log_entry = json.dumps({
            "record_id": record.record_id,
            "change_type": record.change_type.value,
            "config_key": record.config_key,
            "environment": record.environment,
            "old_hash": record.old_hash,
            "new_hash": record.new_hash,
            "operator": record.operator,
            "reason": record.reason,
            "timestamp": record.timestamp,
        }, ensure_ascii=False)
        
        with open(self.audit_log_path, "a") as f:
            f.write(log_entry + "\n")
    
    def get_history(
        self,
        config_key: Optional[str] = None,
        environment: Optional[str] = None,
        limit: int = 50
    ) -> list[ConfigAuditRecord]:
        """查询配置变更历史"""
        filtered = self.records
        
        if config_key:
            filtered = [r for r in filtered if r.config_key == config_key]
        if environment:
            filtered = [r for r in filtered if r.environment == environment]
        
        return sorted(filtered, key=lambda r: r.timestamp, reverse=True)[:limit]
    
    def get_rollback_point(
        self,
        config_key: str,
        target_timestamp: float
    ) -> Optional[ConfigAuditRecord]:
        """找到指定时间点之前的最近一次变更（用于回滚）"""
        relevant = [
            r for r in self.records
            if r.config_key == config_key and r.timestamp <= target_timestamp
        ]
        
        if relevant:
            return max(relevant, key=lambda r: r.timestamp)
        return None
```

### 4.2 配置回滚机制

```python
class ConfigRollbackManager:
    """配置回滚管理器"""
    
    def __init__(
        self, 
        config_loader: EnvironmentConfigLoader,
        audit_logger: ConfigAuditLogger
    ):
        self.config_loader = config_loader
        self.audit_logger = audit_logger
    
    def rollback_config(
        self,
        config_key: str,
        environment: str,
        target_version: int,
        operator: str,
        reason: str = ""
    ) -> bool:
        """回滚配置到指定版本"""
        # 1. 找到目标版本的配置
        history = self.audit_logger.get_history(
            config_key=config_key,
            environment=environment
        )
        
        target_record = None
        for record in history:
            if record.new_hash and self._get_version_from_hash(
                record.new_hash
            ) == target_version:
                target_record = record
                break
        
        if not target_record:
            return False
        
        # 2. 获取当前配置（用于审计）
        current_config = self.config_loader.load_config(config_key)
        
        # 3. 应用回滚
        self._apply_config(config_key, environment, target_record.old_value)
        
        # 4. 记录审计
        audit_record = ConfigAuditRecord(
            change_type=ConfigChangeType.ROLLBACK,
            config_key=config_key,
            environment=environment,
            old_value=current_config,
            new_value=target_record.old_value,
            operator=operator,
            reason=f"Rollback to v{target_version}: {reason}",
            old_hash=target_record.new_hash,
            new_hash=target_record.old_hash,
        )
        self.audit_logger.log_change(audit_record)
        
        return True
    
    def _apply_config(self, key: str, environment: str, value: Any):
        """应用配置到文件系统"""
        config_dir = Path(f"./config/{environment}")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / f"{key}.json"
        with open(config_file, "w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
    
    def _get_version_from_hash(self, content_hash: str) -> int:
        """从哈希值获取版本号（简化实现）"""
        return int(content_hash[:4], 16) % 1000
```

---

## 五、配置管理平台架构

### 5.1 端到端配置管理流程

```
┌─────────────────────────────────────────────────────────────┐
│                AI 应用配置管理全生命周期                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 配置编辑  │───▶│ 配置审核  │───▶│ 配置测试  │              │
│  │ (IDE/API) │    │ (变更审批) │    │ (A/B测试) │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │                              │                      │
│       │         ┌──────────────┐     │                      │
│       │         │  配置版本库   │     │                      │
│       └────────▶│  (Git/DB)    │◀────┘                      │
│                 └──────┬───────┘                            │
│                        │                                    │
│                 ┌──────▼───────┐                            │
│                 │  配置分发     │                            │
│                 │  (推送/拉取)  │                            │
│                 └──────┬───────┘                            │
│                        │                                    │
│           ┌────────────┼────────────┐                       │
│           ▼            ▼            ▼                       │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐                 │
│    │ 开发环境  │ │ 测试环境  │ │ 生产环境  │                 │
│    └──────────┘ └──────────┘ └──────────┘                 │
│           │            │            │                       │
│           └────────────┼────────────┘                       │
│                        ▼                                    │
│                 ┌──────────────┐                            │
│                 │  变更监控     │                            │
│                 │  审计日志     │                            │
│                 │  回滚能力     │                            │
│                 └──────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 配置管理的核心原则

| 原则 | 说明 | 实践要点 |
|-----|------|---------|
| **单一来源** | 所有配置有一个权威来源 | 使用配置中心或Git仓库作为Single Source of Truth |
| **不可变配置** | 配置变更产生新版本，不修改旧版本 | 每次变更创建审计记录，支持任意版本回滚 |
| **环境隔离** | 不同环境的配置严格隔离 | 配置文件按环境目录组织，禁止跨环境引用 |
| **最小权限** | 生产配置变更需要审批 | 关键配置变更设置审批流程 |
| **可观测** | 所有配置变更有完整的审计链路 | 记录谁在什么时间改了什么，为什么改 |
| **可回滚** | 任何配置变更都可以快速回滚 | 保持历史版本，支持一键回滚 |

### 5.3 配置管理检查清单

在AI应用上线前，检查以下配置管理能力是否到位：

- [ ] **Prompt版本化**：所有Prompt模板是否都在版本控制下？
- [ ] **模型参数管理**：temperature、top_p等参数是否按环境配置？
- [ ] **环境隔离**：开发和生产是否使用独立的配置？
- [ ] **配置审计**：是否有完整的配置变更记录？
- [ ] **回滚能力**：配置出问题时能否在5分钟内回滚？
- [ ] **热更新**：关键配置变更是否不需要重启服务？
- [ ] **降级策略**：模型不可用时的备选方案是否配置好？
- [ ] **安全规则**：内容过滤和安全策略是否版本化管理？
- [ ] **监控告警**：配置变更是否触发通知？

---

## 总结

AI应用的配置管理不是传统DevOps配置管理的简单扩展——它需要处理**没有对错之分、只有好坏之别的概率性配置**。核心要点：

1. **配置分层是基础**——将基础设施、RAG、模型、业务配置分层管理，每层有不同的变更频率和管理策略
2. **Prompt需要版本管理**——像管理代码一样管理Prompt，支持版本对比、A/B测试和快速回滚
3. **环境隔离是底线**——开发环境可以随便改，生产环境必须审批，但所有环境使用同一套配置管理流程
4. **热更新是效率保障**——Prompt和安全规则的变更不应该依赖服务重启
5. **审计和回滚是安全网**——每一次配置变更都要记录，每一个错误配置都要能回滚

好的配置管理让你的AI应用在快速迭代的同时保持稳定——这才是工程化的真正价值。
