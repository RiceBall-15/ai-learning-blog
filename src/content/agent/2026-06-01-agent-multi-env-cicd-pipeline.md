---
title: "Agent多环境管理与CI/CD流水线：从开发到生产的完整实践"
description: "深入解析Agent系统的多环境管理策略与CI/CD流水线设计，涵盖环境隔离、自动化测试、基础设施即代码、蓝绿部署等核心实践"
date: 2026-06-01
author: "RiceBall-15"
category: "agent"
subCategory: "agent-ops"
tags: ["Agent", "CI/CD", "多环境管理", "DevOps", "生产部署", "基础设施即代码"]
draft: false
---

# Agent多环境管理与CI/CD流水线：从开发到生产的完整实践

## 一、概念原理

### 1.1 为什么Agent系统需要专门的多环境管理

传统Web应用的多环境管理相对成熟——开发、测试、预发、生产环境之间的差异主要集中在数据和配置上。但Agent系统引入了一个全新的复杂维度：**LLM行为的不确定性**。

一个典型的Agent系统涉及以下组件：

```
┌─────────────────────────────────────────────────┐
│                  Agent系统                       │
├─────────────────────────────────────────────────┤
│  Prompt模板层    │  LLM推理层    │  工具执行层   │
│  (版本控制)      │  (模型切换)    │  (沙箱隔离)   │
├─────────────────────────────────────────────────┤
│  记忆存储层      │  安全过滤层    │  监控采集层   │
│  (数据隔离)      │  (策略分级)    │  (环境标签)   │
└─────────────────────────────────────────────────┘
```

这导致了三个核心挑战：

**挑战一：Prompt漂移**。同一个Prompt在GPT-4和Claude-3.5上的表现可能截然不同，甚至同一个模型在不同温度参数下的输出风格都会有显著差异。开发环境用小模型调试通过的功能，切换到生产环境的大模型后可能出现边界case。

**挑战二：工具链差异**。开发环境可能使用Mock API、本地数据库、沙箱工具；生产环境连接真实API、生产数据库、有权限限制的工具。工具的返回格式、错误处理、超时行为在不同环境间存在差异。

**挑战三：状态不可复现**。Agent的记忆系统、对话历史、上下文窗口在每次运行时都是动态的。一个在测试环境"通过"的用例，可能因为上下文长度变化、记忆检索结果不同而在生产中失败。

### 1.2 环境分级模型

针对Agent系统的特殊性，推荐采用**五级环境模型**：

| 环境层级 | 名称 | 用途 | LLM配置 | 工具配置 | 数据策略 |
|---------|------|------|---------|---------|---------|
| L0 | 本地开发 | 开发者单机调试 | Mock/小模型 | Mock工具 | 合成数据 |
| L1 | 集成测试 | 自动化CI验证 | 小模型 | 沙箱工具 | 测试数据集 |
| L2 | 预发环境 | 上线前验证 | 生产模型(限额) | 生产工具(只读) | 脱敏数据 |
| L3 | 灰度环境 | 渐进式发布 | 生产模型 | 生产工具 | 生产数据子集 |
| L4 | 生产环境 | 全量服务 | 生产模型 | 生产工具 | 生产数据 |

每一层级都承载不同的验证目标：

- **L0**：验证Prompt逻辑、工具接口、基本流程
- **L1**：验证端到端集成、多Agent协作、错误恢复
- **L2**：验证真实LLM行为、性能基线、成本预估
- **L3**：验证生产级流量下的稳定性、监控告警
- **L4**：全量服务，持续监控

### 1.3 环境隔离原则

Agent系统的环境隔离需要覆盖四个维度：

```
环境隔离四维度：
┌──────────────┬─────────────────────────────────────┐
│ 数据隔离     │ 每层环境使用独立数据集，生产数据   │
│              │ 不可逆向流向低层环境               │
├──────────────┼─────────────────────────────────────┤
│ 模型隔离     │ 不同环境可切换模型供应商/版本，    │
│              │ 配置统一管理但运行时隔离           │
├──────────────┼─────────────────────────────────────┤
│ 工具隔离     │ 工具沙箱化，开发环境Mock、测试环境 │
│              │ 限速、生产环境全权限               │
├──────────────┼─────────────────────────────────────┤
│ 网络隔离     │ 低层环境无法访问生产网络资源，     │
│              │ API密钥按环境分级管理              │
└──────────────┴─────────────────────────────────────┘
```

## 二、架构设计

### 2.1 多环境管理架构

整体架构采用**配置驱动 + 环境注入**模式：

```
┌─────────────────────────────────────────────────────────┐
│                    配置管理层                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ env-config/  │  │ prompt-      │  │ tool-registry │  │
│  │ dev.yaml     │  │ templates/   │  │ /dev.yaml     │  │
│  │ test.yaml    │  │ dev/         │  │ test.yaml     │  │
│  │ staging.yaml │  │ test/        │  │ staging.yaml  │  │
│  │ prod.yaml    │  │ prod/        │  │ prod.yaml     │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ 环境变量注入
┌────────────────────────▼────────────────────────────────┐
│                    运行时引擎                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Agent Runtime                        │   │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────────┐ │   │
│  │  │ Prompt     │→ │ LLM      │→ │ Tool         │ │   │
│  │  │ Loader     │  │ Router   │  │ Executor     │ │   │
│  │  └────────────┘  └──────────┘  └──────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    基础设施层                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ LLM API  │  │ 数据库   │  │ 监控系统 │              │
│  │ (按环境) │  │ (按环境) │  │ (统一)   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 CI/CD流水线架构

Agent系统的CI/CD流水线与传统应用有显著区别，核心差异在于**LLM验证环节**需要额外的测试策略：

```
代码提交
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ 静态分析    │────→│ Lint + 格式化    │
│ (Phase 1)   │     │ 安全扫描         │
└─────────────┘     └──────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ 单元测试    │────→│ Prompt模板测试   │
│ (Phase 2)   │     │ 工具Mock测试     │
│             │     │ 记忆系统测试     │
└─────────────┘     └──────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ 集成测试    │────→│ 多Agent协作验证  │
│ (Phase 3)   │     │ 端到端流程测试   │
│             │     │ 异常恢复测试     │
└─────────────┘     └──────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ LLM验证    │────→│ 输出质量评估     │
│ (Phase 4)   │     │ 行为一致性检测   │
│             │     │ 成本预算验证     │
└─────────────┘     └──────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ 部署验证    │────→│ 蓝绿/金丝雀     │
│ (Phase 5)   │     │ 健康检查         │
│             │     │ 监控告警验证     │
└─────────────┘     └──────────────────┘
```

### 2.3 Prompt版本管理架构

Prompt是Agent系统最核心的资产，需要独立的版本管理策略：

```
prompt-registry/
├── main/                          # 生产版本
│   ├── system-prompt.yaml         # 系统级Prompt
│   ├── task-prompts/              # 任务级Prompt
│   │   ├── code-review.yaml
│   │   ├── data-analysis.yaml
│   │   └── customer-service.yaml
│   └── tool-prompts/              # 工具调用Prompt
│       ├── search-tool.yaml
│       └── db-query.yaml
├── staging/                       # 预发版本（可能领先于main）
│   └── ... (继承main，允许修改)
├── templates/                     # Prompt模板库
│   └── ... (通用模板，被各版本引用)
└── versions/                      # 版本历史
    ├── v1.0.0/
    ├── v1.1.0/
    └── v1.2.0-rc1/
```

**版本管理规则**：
- 每次Prompt变更必须创建新版本，不允许直接修改已部署版本
- 版本号遵循语义化版本：主版本号（行为变更）.次版本号（功能新增）.修订号（修复）
- 预发环境可以使用RC版本进行验证
- 生产环境只允许使用正式版本

## 三、实战实现

### 3.1 环境配置文件设计

创建统一的环境配置体系：

```yaml
# env-config/dev.yaml
agent:
  env: development
  debug: true
  log_level: DEBUG

llm:
  provider: openai
  model: gpt-4o-mini          # 开发用小模型
  temperature: 0.7
  max_tokens: 2048
  api_key: ${DEV_OPENAI_KEY}   # 从环境变量读取
  timeout: 60
  # Mock模式：返回预定义响应，不调用真实API
  mock: false

tools:
  search:
    provider: mock              # Mock搜索引擎
    mock_responses: true
  database:
    host: localhost
    port: 5432
    database: agent_dev
    # 使用Docker Compose启动的本地数据库
  code_execution:
    sandbox: docker             # Docker沙箱隔离
    image: agent-sandbox:latest
    timeout: 30

memory:
  backend: sqlite               # 本地SQLite
  path: ./data/agent-dev.db
  
monitoring:
  enabled: true
  provider: local               # 本地Jaeger

prompts:
  registry: ./prompt-registry/dev/
```

```yaml
# env-config/prod.yaml
agent:
  env: production
  debug: false
  log_level: INFO

llm:
  provider: openai
  model: gpt-4o                # 生产用大模型
  temperature: 0.3             # 生产环境降低随机性
  max_tokens: 4096
  api_key: ${PROD_OPENAI_KEY}
  timeout: 120
  retry:
    max_attempts: 3
    backoff_factor: 2
  rate_limit:
    requests_per_minute: 60
    tokens_per_minute: 150000
  # 降级策略：主模型不可用时自动切换
  fallback:
    provider: anthropic
    model: claude-3.5-sonnet

tools:
  search:
    provider: tavily
    api_key: ${TAVILY_API_KEY}
    rate_limit: 100             # 每分钟100次
  database:
    host: ${PROD_DB_HOST}
    port: 5432
    database: agent_production
    pool_size: 20               # 连接池
    ssl: true
  code_execution:
    sandbox: kubernetes         # K8s沙箱
    namespace: agent-sandbox-prod
    timeout: 60
    resource_limits:
      cpu: "1"
      memory: "512Mi"

memory:
  backend: redis-cluster        # Redis集群
  cluster_nodes:
    - ${REDIS_NODE_1}
    - ${REDIS_NODE_2}
    - ${REDIS_NODE_3}
  ttl: 86400                   # 24小时过期

monitoring:
  enabled: true
  provider: datadog
  metrics_prefix: agent.prod

prompts:
  registry: ./prompt-registry/main/
```

### 3.2 环境切换中间件

实现一个轻量级的环境上下文管理器：

```python
# agent/runtime/env_context.py
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class EnvironmentConfig:
    """环境配置数据类"""
    env: str
    debug: bool
    log_level: str
    llm: Dict[str, Any]
    tools: Dict[str, Any]
    memory: Dict[str, Any]
    monitoring: Dict[str, Any]
    prompts: Dict[str, Any]

class EnvContext:
    """环境上下文管理器 - 负责加载和注入环境配置"""
    
    _instance = None
    _current_env: Optional[EnvironmentConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, env_name: str = None) -> 'EnvContext':
        """
        加载环境配置
        
        优先级：环境变量 AGENT_ENV > 参数指定 > 默认 dev
        """
        if env_name is None:
            env_name = os.environ.get('AGENT_ENV', 'dev')
        
        config_path = Path(f'env-config/{env_name}.yaml')
        if not config_path.exists():
            raise FileNotFoundError(
                f"环境配置文件不存在: {config_path}\n"
                f"可用环境: {[f.stem for f in Path('env-config/').glob('*.yaml')]}"
            )
        
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        
        # 环境变量插值
        config = cls._resolve_env_vars(raw_config)
        
        cls._current_env = EnvironmentConfig(**config)
        cls._instance = cls._current_env
        return cls
    
    @staticmethod
    def _resolve_env_vars(config: Any) -> Any:
        """递归解析配置中的 ${VAR} 引用"""
        if isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            var_name = config[2:-1]
            value = os.environ.get(var_name)
            if value is None:
                raise EnvironmentError(
                    f"环境变量 {var_name} 未设置\n"
                    f"请在 .env 文件或系统环境中设置: export {var_name}=..."
                )
            return value
        elif isinstance(config, dict):
            return {k: EnvContext._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [EnvContext._resolve_env_vars(item) for item in config]
        return config
    
    @property
    def current(self) -> EnvironmentConfig:
        if self._current_env is None:
            self.load()
        return self._current_env
    
    @contextmanager
    def override(self, **kwargs):
        """临时覆盖配置项（用于测试）"""
        original = self._current_env
        overrides = {}
        
        def deep_update(base, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in base:
                    deep_update(base[key], value)
                else:
                    overrides[key] = base.get(key)
                    base[key] = value
        
        deep_update(self._current_env.__dict__, kwargs)
        try:
            yield
        finally:
            # 恢复原始配置
            for key, value in overrides.items():
                setattr(self._current_env, key, value)


# 使用示例
def get_agent_config():
    """获取当前环境的Agent配置"""
    ctx = EnvContext.load()
    cfg = ctx.current
    
    return {
        "llm": {
            "provider": cfg.llm["provider"],
            "model": cfg.llm["model"],
            "temperature": cfg.llm["temperature"],
            "api_key": cfg.llm["api_key"],
        },
        "tools": cfg.tools,
        "memory": cfg.memory,
    }
```

### 3.3 CI/CD流水线实现

使用GitHub Actions实现Agent系统的完整CI/CD流水线：

```yaml
# .github/workflows/agent-cicd.yaml
name: Agent CI/CD Pipeline

on:
  push:
    branches: [main, develop, 'release/**']
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'
  AGENT_ENV: test

jobs:
  # ============================================
  # Phase 1: 静态分析
  # ============================================
  lint:
    name: "Phase 1: Lint & Security Scan"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install ruff bandit safety
          pip install -r requirements.txt
      
      - name: Ruff lint
        run: ruff check src/ tests/ --output-format=github
      
      - name: Security scan
        run: |
          bandit -r src/ -f json -o bandit-report.json || true
          safety check --json > safety-report.json || true
      
      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: lint-reports
          path: |
            bandit-report.json
            safety-report.json

  # ============================================
  # Phase 2: 单元测试
  # ============================================
  unit-test:
    name: "Phase 2: Unit Tests"
    runs-on: ubuntu-latest
    needs: lint
    services:
      redis:
        image: redis:7-alpine
        ports: [6379:6379]
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run unit tests
        env:
          AGENT_ENV: test
          REDIS_URL: redis://localhost:6379
        run: |
          pytest tests/unit/ \
            --cov=src/ \
            --cov-report=xml \
            --junitxml=unit-test-results.xml \
            -x -q
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # ============================================
  # Phase 3: 集成测试
  # ============================================
  integration-test:
    name: "Phase 3: Integration Tests"
    runs-on: ubuntu-latest
    needs: unit-test
    services:
      redis:
        image: redis:7-alpine
        ports: [6379:6379]
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: agent_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: [5432:5432]
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Run integration tests
        env:
          AGENT_ENV: test
          REDIS_URL: redis://localhost:6379
          DATABASE_URL: postgresql://test:test@localhost:5432/agent_test
          # 使用Mock LLM API，不调用真实API
          OPENAI_API_KEY: "sk-test-mock-key"
          LLM_MOCK_MODE: "true"
        run: |
          pytest tests/integration/ \
            --junitxml=integration-test-results.xml \
            -x -v

  # ============================================
  # Phase 4: LLM行为验证（关键环节）
  # ============================================
  llm-validation:
    name: "Phase 4: LLM Behavior Validation"
    runs-on: ubuntu-latest
    needs: integration-test
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Run LLM validation suite
        env:
          AGENT_ENV: test
          OPENAI_API_KEY: ${{ secrets.TEST_OPENAI_KEY }}
          # 限制测试环境的API调用预算
          LLM_BUDGET_LIMIT: "5.00"  # 最多$5
        run: |
          python -m pytest tests/llm_validation/ \
            --junitxml=llm-validation-results.xml \
            -v \
            --timeout=300
      
      - name: Compare with baseline
        run: |
          python scripts/compare_llm_outputs.py \
            --baseline baselines/main/ \
            --current llm-validation-results.xml \
            --threshold 0.85 \
            --output llm-comparison-report.md
      
      - name: Upload validation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: llm-validation-report
          path: llm-comparison-report.md

  # ============================================
  # Phase 5: 部署
  # ============================================
  deploy-staging:
    name: "Phase 5a: Deploy to Staging"
    runs-on: ubuntu-latest
    needs: [lint, unit-test, integration-test, llm-validation]
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          # 使用环境特定的配置
          cp env-config/staging.yaml agent-env.yaml
          # 部署逻辑...
      
      - name: Smoke test staging
        run: |
          # 运行冒烟测试验证部署成功
          python scripts/smoke_test.py --env staging
      
      - name: Notify staging deployment
        run: |
          echo "Staging deployment successful"
          # 发送Slack通知...

  deploy-production:
    name: "Phase 5b: Deploy to Production"
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    environment: production  # 需要手动审批
    steps:
      - uses: actions/checkout@v4
      
      - name: Blue-Green deployment
        run: |
          echo "Starting blue-green deployment..."
          # 1. 部署到绿色环境
          # 2. 健康检查
          # 3. 流量切换
          # 4. 旧版本保留30分钟
      
      - name: Health check
        run: |
          python scripts/health_check.py \
            --env production \
            --timeout 120 \
            --threshold 0.95
      
      - name: Notify production deployment
        run: |
          echo "Production deployment successful"
```

### 3.4 LLM行为验证框架

LLM行为验证是Agent系统CI/CD中最独特的环节。传统的断言式测试不适用于LLM输出，需要引入**评估框架**：

```python
# tests/llm_validation/base_validator.py
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field

@dataclass
class ValidationCase:
    """LLM验证用例"""
    name: str
    input_prompt: str
    expected_behavior: str           # 行为描述（非精确匹配）
    max_tokens: int = 512
    temperature: float = 0.0
    tags: List[str] = field(default_factory=list)
    # 断言函数列表
    assertions: List[Callable] = field(default_factory=list)
    # 成本预算（单个用例）
    max_cost_usd: float = 0.01

@dataclass
class ValidationResult:
    """验证结果"""
    case_name: str
    passed: bool
    output: str
    score: float                     # 0-1，与基线的相似度
    cost_usd: float
    latency_ms: float
    errors: List[str] = field(default_factory=list)

class LLMValidator:
    """LLM行为验证器"""
    
    def __init__(self, client, baseline_dir: str = "baselines/main/"):
        self.client = client
        self.baseline_dir = Path(baseline_dir)
        self.results: List[ValidationResult] = []
        self.total_cost = 0.0
    
    def validate(self, cases: List[ValidationCase]) -> List[ValidationResult]:
        """运行所有验证用例"""
        budget_limit = float(os.environ.get('LLM_BUDGET_LIMIT', '5.00'))
        
        for case in cases:
            if self.total_cost + case.max_cost_usd > budget_limit:
                print(f"⚠️ 预算耗尽，跳过: {case.name}")
                continue
            
            result = self._run_case(case)
            self.results.append(result)
            self.total_cost += result.cost_usd
            
            # 基线对比
            baseline = self._load_baseline(case.name)
            if baseline:
                result.score = self._compare_output(result.output, baseline)
                if result.score < 0.85:
                    result.passed = False
                    result.errors.append(
                        f"输出与基线偏差过大: {result.score:.2f} < 0.85"
                    )
        
        return self.results
    
    def _run_case(self, case: ValidationCase) -> ValidationResult:
        """执行单个验证用例"""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=os.environ.get('LLM_TEST_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": case.input_prompt}],
            max_tokens=case.max_tokens,
            temperature=case.temperature,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        output = response.choices[0].message.content
        cost = self._estimate_cost(response)
        
        # 运行自定义断言
        errors = []
        for assertion in case.assertions:
            try:
                if not assertion(output):
                    errors.append(f"断言失败: {assertion.__name__}")
            except Exception as e:
                errors.append(f"断言异常: {str(e)}")
        
        return ValidationResult(
            case_name=case.name,
            passed=len(errors) == 0,
            output=output,
            score=1.0,
            cost_usd=cost,
            latency_ms=latency_ms,
            errors=errors,
        )
    
    def _load_baseline(self, case_name: str) -> str:
        """加载基线输出"""
        baseline_file = self.baseline_dir / f"{case_name}.json"
        if baseline_file.exists():
            with open(baseline_file) as f:
                data = json.load(f)
            return data.get('output', '')
        return None
    
    def _compare_output(self, current: str, baseline: str) -> float:
        """
        对比输出与基线的相似度
        使用简单的文本相似度 + 语义关键点匹配
        """
        # 简单实现：基于关键点的匹配
        baseline_points = set(baseline.lower().split())
        current_points = set(current.lower().split())
        
        if not baseline_points:
            return 1.0
        
        overlap = len(baseline_points & current_points)
        return overlap / len(baseline_points)
    
    def _estimate_cost(self, response) -> float:
        """估算API调用成本"""
        usage = response.usage
        # GPT-4o-mini定价
        input_cost = usage.prompt_tokens * 0.00015 / 1000
        output_cost = usage.completion_tokens * 0.0006 / 1000
        return input_cost + output_cost
    
    def generate_report(self) -> str:
        """生成验证报告"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        report = f"""# LLM验证报告

## 概览
- 总用例数: {total}
- 通过: {passed}
- 失败: {total - passed}
- 通过率: {passed/total*100:.1f}%
- 总成本: ${self.total_cost:.4f}

## 用例详情
"""
        for r in self.results:
            status = "✅" if r.passed else "❌"
            report += f"""
### {status} {r.case_name}
- 得分: {r.score:.2f}
- 延迟: {r.latency_ms:.0f}ms
- 成本: ${r.cost_usd:.6f}
- 错误: {r.errors if r.errors else '无'}
"""
        return report
```

### 3.5 基础设施即代码（IaC）

使用Terraform管理Agent系统的基础设施：

```hcl
# infra/main.tf
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ============================================
# 环境变量模块
# ============================================
variable "environment" {
  type        = string
  description = "部署环境 (dev/staging/prod)"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "环境必须是 dev, staging 或 prod"
  }
}

# 按环境差异化配置
locals {
  env_config = {
    dev = {
      instance_type    = "t3.medium"
      min_count        = 1
      max_count        = 2
      db_instance_class = "db.t3.micro"
      cache_node_type   = "cache.t3.micro"
    }
    staging = {
      instance_type    = "t3.large"
      min_count        = 2
      max_count        = 4
      db_instance_class = "db.t3.small"
      cache_node_type   = "cache.t3.small"
    }
    prod = {
      instance_type    = "c5.2xlarge"
      min_count        = 3
      max_count        = 10
      db_instance_class = "db.r5.large"
      cache_node_type   = "cache.r5.large"
    }
  }
  
  config = local.env_config[var.environment]
}

# ============================================
# EKS集群 - Agent运行环境
# ============================================
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = "agent-${var.environment}"
  cluster_version = "1.29"
  
  node_groups = {
    agent = {
      instance_types = [local.config.instance_type]
      min_size       = local.config.min_count
      max_size       = local.config.max_count
      
      labels = {
        environment = var.environment
        workload    = "agent"
      }
    }
    
    # 独立的工具执行节点组（沙箱隔离）
    tools = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 5
      
      labels = {
        environment = var.environment
        workload    = "tools"
      }
    }
  }
}

# ============================================
# Agent部署（Kubernetes）
# ============================================
resource "kubernetes_namespace" "agent" {
  metadata {
    name = "agent-${var.environment}"
    labels = {
      environment = var.environment
      managed_by  = "terraform"
    }
  }
}

resource "kubernetes_deployment" "agent" {
  metadata {
    name      = "agent-core"
    namespace = kubernetes_namespace.agent.metadata[0].name
  }
  
  spec {
    replicas = local.config.min_count
    
    selector {
      match_labels = {
        app = "agent-core"
      }
    }
    
    template {
      metadata {
        labels = {
          app         = "agent-core"
          environment = var.environment
        }
      }
      
      spec {
        container {
          name  = "agent"
          image = "agent-core:${var.environment}-latest"
          
          port {
            container_port = 8080
          }
          
          env_from {
            config_map_ref {
              name = "agent-config-${var.environment}"
            }
          }
          
          env_from {
            secret_ref {
              name = "agent-secrets-${var.environment}"
            }
          }
          
          # 资源限制 - Agent系统需要较多内存
          resources {
            requests = {
              cpu    = "500m"
              memory = "1Gi"
            }
            limits = {
              cpu    = "2"
              memory = "4Gi"
            }
          }
          
          # 健康检查
          liveness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }
          
          readiness_probe {
            http_get {
              path = "/ready"
              port = 8080
            }
            initial_delay_seconds = 10
            period_seconds        = 5
          }
        }
        
        # 工具执行沙箱容器（sidecar）
        container {
          name  = "tool-sandbox"
          image = "tool-sandbox:${var.environment}-latest"
          
          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1"
              memory = "2Gi"
            }
          }
        }
      }
    }
  }
}

# ============================================
# 数据库
# ============================================
resource "aws_db_instance" "agent" {
  identifier     = "agent-${var.environment}"
  instance_class = local.config.db_instance_class
  
  engine         = "postgres"
  engine_version = "16"
  
  allocated_storage     = var.environment == "prod" ? 100 : 20
  max_allocated_storage = var.environment == "prod" ? 500 : 50
  
  multi_az               = var.environment == "prod"
  db_subnet_group_name   = aws_db_subnet_group.agent.name
  vpc_security_group_ids = [aws_security_group.db.id]
  
  backup_retention_period = var.environment == "prod" ? 30 : 7
  
  tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ============================================
# Redis缓存（记忆系统）
# ============================================
resource "aws_elasticache_replication_group" "agent" {
  replication_group_id = "agent-${var.environment}"
  
  node_type            = local.config.cache_node_type
  num_cache_clusters   = var.environment == "prod" ? 3 : 1
  
  automatic_failover_enabled = var.environment == "prod"
  
  parameter_group_name = "default.redis7"
  
  security_group_ids = [aws_security_group.redis.id]
}
```

## 四、生产优化

### 4.1 环境配置管理最佳实践

**配置漂移检测**：定期对比各环境的配置差异，防止手动修改导致的环境不一致。

```python
# scripts/config_drift_detector.py
import yaml
import hashlib
from pathlib import Path
from typing import Dict, List

class ConfigDriftDetector:
    """检测各环境配置漂移"""
    
    def __init__(self, config_dir: str = "env-config/"):
        self.config_dir = Path(config_dir)
    
    def detect_drift(self) -> Dict[str, List[str]]:
        """检测配置漂移"""
        configs = {}
        for env_file in self.config_dir.glob("*.yaml"):
            with open(env_file) as f:
                config = yaml.safe_load(f)
            configs[env_file.stem] = self._hash_config(config)
        
        # 对比相邻环境的差异
        drifts = {}
        env_names = sorted(configs.keys())
        
        for i in range(len(env_names) - 1):
            env_a, env_b = env_names[i], env_names[i + 1]
            diff = self._compare_configs(
                configs[env_a], configs[env_b]
            )
            if diff:
                drifts[f"{env_a} <-> {env_b}"] = diff
        
        return drifts
    
    def _hash_config(self, config: dict) -> str:
        return hashlib.md5(
            yaml.dump(config, sort_keys=True).encode()
        ).hexdigest()
    
    def _compare_configs(self, config_a: dict, config_b: dict) -> List[str]:
        """对比两个配置的差异"""
        differences = []
        all_keys = set(config_a.keys()) | set(config_b.keys())
        
        for key in all_keys:
            if key not in config_a:
                differences.append(f"  {key}: 仅在 {config_b} 中存在")
            elif key not in config_b:
                differences.append(f"  {key}: 仅在 {config_a} 中存在")
            elif config_a[key] != config_b[key]:
                # 对于非敏感配置，检查是否合理
                if key in ('temperature', 'timeout', 'max_tokens'):
                    differences.append(
                        f"  {key}: {config_a[key]} -> {config_b[key]}"
                    )
        
        return differences


if __name__ == "__main__":
    detector = ConfigDriftDetector()
    drifts = detector.detect_drift()
    
    if drifts:
        print("⚠️ 发现配置漂移:")
        for env_pair, diffs in drifts.items():
            print(f"\n{env_pair}:")
            for diff in diffs:
                print(diff)
    else:
        print("✅ 无配置漂移")
```

### 4.2 蓝绿部署策略

Agent系统的蓝绿部署需要特别注意**有状态组件**（记忆系统、对话历史）的迁移：

```
                    流量切换流程
                    
  ┌──────────┐     ┌──────────┐
  │  蓝环境   │     │  绿环境   │
  │  (v1.2)  │     │  (v1.3)  │
  └────┬─────┘     └────┬─────┘
       │                │
       ▼                ▼
  ┌──────────┐     ┌──────────┐
  │ Agent    │     │ Agent    │
  │ Runtime  │     │ Runtime  │
  └────┬─────┘     └────┬─────┘
       │                │
       ▼                ▼
  ┌──────────┐     ┌──────────┐
  │ Redis    │     │ Redis    │
  │ (共享)   │     │ (共享)   │
  └──────────┘     └──────────┘
  
  步骤1: 部署绿环境，运行健康检查
  步骤2: 流量从蓝切换到绿（10% → 50% → 100%）
  步骤3: 蓝环境保留30分钟用于回滚
  步骤4: 确认无问题后关闭蓝环境
```

**记忆系统迁移策略**：

```python
# scripts/memory_migration.py
class MemoryMigration:
    """蓝绿部署时的记忆系统迁移"""
    
    def __init__(self, old_redis, new_redis):
        self.old = old_redis
        self.new = new_redis
    
    async def migrate_active_sessions(self):
        """迁移活跃会话（最近1小时有活动的）"""
        # 扫描所有会话key
        cursor = 0
        migrated = 0
        
        while True:
            cursor, keys = await self.old.scan(
                cursor=cursor, match="session:*", count=100
            )
            
            for key in keys:
                ttl = await self.old.ttl(key)
                if ttl > 0 and ttl < 3600:  # 1小时内过期的
                    data = await self.old.get(key)
                    await self.new.setex(key, ttl, data)
                    migrated += 1
            
            if cursor == 0:
                break
        
        return migrated
```

### 4.3 监控与告警

为多环境Agent系统建立统一监控：

```python
# monitoring/agent_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 按环境标签区分指标
ENV_LABELS = ['environment', 'model', 'task_type']

# LLM调用指标
llm_requests_total = Counter(
    'agent_llm_requests_total',
    'LLM API调用总数',
    ENV_LABELS
)

llm_latency_seconds = Histogram(
    'agent_llm_latency_seconds',
    'LLM API调用延迟',
    ENV_LABELS,
    buckets=[0.5, 1, 2, 5, 10, 30, 60]
)

llm_tokens_used = Counter(
    'agent_llm_tokens_used_total',
    'LLM Token使用量',
    ENV_LABELS + ['direction']  # direction: input/output
)

# 工具调用指标
tool_calls_total = Counter(
    'agent_tool_calls_total',
    '工具调用总数',
    ENV_LABELS + ['tool_name', 'success']
)

tool_latency_seconds = Histogram(
    'agent_tool_latency_seconds',
    '工具调用延迟',
    ENV_LABELS + ['tool_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

# Agent任务指标
agent_tasks_total = Counter(
    'agent_tasks_total',
    'Agent任务总数',
    ENV_LABELS + ['status']  # status: success/failure/timeout
)

agent_active_sessions = Gauge(
    'agent_active_sessions',
    '活跃会话数',
    ['environment']
)

# 成本追踪
agent_cost_usd = Counter(
    'agent_cost_usd_total',
    'Agent运行成本（美元）',
    ENV_LABELS
)

# 环境特定告警规则
ALERT_RULES = """
# 生产环境高错误率告警
- alert: AgentHighErrorRate
  expr: |
    rate(agent_tasks_total{status="failure", environment="prod"}[5m])
    / rate(agent_tasks_total{environment="prod"}[5m]) > 0.05
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "生产环境Agent错误率超过5%"

# 生产环境成本超预算告警
- alert: AgentCostOverBudget
  expr: |
    increase(agent_cost_usd_total{environment="prod"}[1h]) > 50
  labels:
    severity: warning
  annotations:
    summary: "生产环境Agent小时成本超过$50"

# 测试环境LLM延迟异常
- alert: TestEnvHighLatency
  expr: |
    histogram_quantile(0.95, 
      rate(agent_llm_latency_seconds_bucket{environment="test"}[5m])
    ) > 30
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "测试环境P95延迟超过30秒"
"""
```

## 五、面试深度

### 5.1 高频面试题

**Q1: Agent系统的多环境管理和传统Web应用有什么区别？**

**核心差异点**：

| 维度 | 传统Web应用 | Agent系统 |
|------|------------|-----------|
| 配置差异 | 数据库、API地址、密钥 | 额外包含LLM模型、温度、Prompt版本 |
| 行为不确定性 | 确定性（相同输入→相同输出） | 非确定性（LLM输出有随机性） |
| 测试策略 | 断言式测试 | 基于评估框架的语义对比 |
| 数据流 | 请求→响应 | 请求→推理→工具调用→推理→响应 |
| 部署风险 | 配置错误、依赖缺失 | Prompt漂移、模型切换、成本失控 |

**关键回答**：Agent系统的环境管理核心挑战在于LLM行为的非确定性。同一个Prompt在不同环境（不同模型版本、不同温度参数）下的表现可能截然不同，因此需要引入LLM行为验证环节，通过评估框架对比输出质量，而非简单的断言测试。

**Q2: 如何设计Agent系统的CI/CD流水线？**

**五阶段模型**：

1. **静态分析**：代码质量、安全扫描
2. **单元测试**：Prompt模板测试、工具Mock测试、记忆系统测试
3. **集成测试**：多Agent协作验证、端到端流程测试
4. **LLM验证**：输出质量评估、行为一致性检测、成本预算验证
5. **部署验证**：蓝绿/金丝雀部署、健康检查、监控告警验证

**关键点**：Phase 4（LLM验证）是Agent系统CI/CD最独特的环节。需要建立评估框架，定义行为基线，对比每次变更的输出质量，确保Prompt修改不会引入回归问题。同时需要设置成本预算上限，防止LLM API调用失控。

**Q3: Agent系统灰度发布时，如何处理有状态组件（记忆系统）？**

**策略**：

1. **共享存储层**：Redis/数据库在蓝绿环境间共享，新版本直接读取旧数据
2. **主动迁移**：对于环境特定的数据（如本地文件缓存），在切换流量前主动迁移
3. **兼容性设计**：数据格式变更时，新版本需同时兼容新旧格式
4. **回滚准备**：保留旧环境30分钟，记忆系统支持版本回退

**Q4: Prompt版本管理的最佳实践是什么？**

**核心原则**：
- 每次Prompt变更必须创建新版本，不允许直接修改已部署版本
- 版本号遵循语义化版本：主版本号（行为变更）.次版本号（功能新增）.修订号（修复）
- 生产环境只允许使用正式版本，预发环境可以使用RC版本
- Prompt变更需要经过LLM验证流程才能合并到main分支

**Q5: 如何在Agent系统中实现成本控制？**

**多层控制策略**：

1. **预算层**：每个环境设置月度预算上限，超过自动告警
2. **速率层**：API调用速率限制，防止单用户/单任务消耗过多资源
3. **模型层**：根据任务复杂度自动选择模型（简单任务用小模型）
4. **缓存层**：语义缓存，相似请求复用历史结果
5. **降级层**：超预算时自动降级到低成本模型或Mock响应

### 5.2 架构设计面试题

**Q6: 如果让你从零设计一个Agent系统的CI/CD流水线，你会如何决策？**

**决策框架**：

```
1. 团队规模和成熟度
   ├── 小团队（<5人）→ 简化流水线，重点放在Phase 1-3
   ├── 中团队（5-20人）→ 完整五阶段，LLM验证用自动化评估
   └── 大团队（>20人）→ 分层流水线，不同模块独立验证

2. LLM API成本预算
   ├── 低预算（<$100/月）→ LLM验证用小模型，限制测试用例数
   ├── 中预算（$100-1000/月）→ 完整验证，使用生产模型
   └── 高预算（>$1000/月）→ 可以用A/B测试验证，多模型对比

3. 合规要求
   ├── 无特殊要求 → 标准流水线
   ├── 数据合规（GDPR等）→ 增加数据脱敏测试、隐私检查
   └── 行业合规（金融/医疗）→ 增加安全审计、输出审查环节
```

### 5.3 开放性问题

**Q7: 如何处理Agent系统中Prompt修改的回归测试问题？**

**思路**：
1. 建立Prompt行为基线：记录每个Prompt在关键场景下的输出
2. 变更影响分析：评估Prompt修改可能影响的下游任务
3. 渐进式验证：小范围灰度验证，对比新旧Prompt的输出质量
4. 自动回滚机制：如果新Prompt导致质量下降，自动回滚到旧版本

**Q8: 在资源受限的环境下（如2核2G服务器），如何优化Agent系统的CI/CD？**

**优化策略**：
1. **跳过本地构建**：直接推送到GitHub，依赖云端CI
2. **分层测试**：PR只跑Phase 1-2，main分支跑完整流水线
3. **Mock优先**：开发和测试环境使用Mock LLM，只在验证阶段调用真实API
4. **增量测试**：根据代码变更范围，只运行受影响的测试用例
5. **缓存依赖**：使用GitHub Actions缓存Python依赖和模型文件

---

## 总结

Agent系统的多环境管理与CI/CD流水线设计是一个系统工程，核心挑战在于：

1. **行为非确定性**：LLM输出的随机性要求引入评估框架而非断言测试
2. **多维度隔离**：数据、模型、工具、网络四个维度都需要环境隔离
3. **成本可控**：LLM API调用需要预算管理、速率限制、模型降级等多层控制
4. **版本管理**：Prompt作为核心资产，需要独立的版本管理和灰度策略

成功的Agent CI/CD流水线应该在**自动化验证**和**成本控制**之间找到平衡，既要保证每次变更的质量，又要避免验证过程本身消耗过多资源。
