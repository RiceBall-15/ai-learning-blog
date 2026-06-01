---
title: "LLM应用的CI/CD与持续交付：从Prompt版本管理到模型灰度发布的工程化实践"
description: "系统讲解大模型应用的持续集成与交付体系，覆盖Prompt版本管理、模型评估自动化、A/B测试框架与灰度发布策略"
date: 2026-05-30
author: "RiceBall-15"
category: "engineering"
subCategory: infra
tags: ["CI/CD", "LLM工程化", "Prompt管理", "灰度发布", "模型评估", "持续交付", "DevOps"]
draft: false
---

## 一、引言：LLM应用的DevOps困境

传统的CI/CD体系在LLM应用面前几乎是"失灵"的。

原因很简单——传统软件的构建是**确定性**的：同样的代码、同样的依赖、同样的配置，构建结果一定相同。但LLM应用的核心组件——Prompt、模型、向量数据库、RAG管线——每一环都带有**不确定性**。

```
传统软件 CI/CD:                    LLM应用 CI/CD:
┌──────────────┐                  ┌──────────────┐
│   代码变更    │                  │  Prompt变更   │
│      ↓       │                  │      ↓       │
│   编译构建    │                  │  模型推理     │ ← 不确定性引入
│      ↓       │                  │      ↓       │
│   单元测试    │                  │  评估判断     │ ← 需要语义级测试
│      ↓       │                  │      ↓       │
│   集成测试    │                  │  人工/自动评审 │ ← 需要领域专家
│      ↓       │                  │      ↓       │
│   部署上线    │                  │  灰度发布     │ ← 需要指标监控
└──────────────┘                  └──────────────┘
  确定性、快速、自动化              不确定性、需要人工介入
```

本文将从工程实践出发，构建一套**适配LLM应用特性**的CI/CD体系。

---

## 二、LLM应用CI/CD全景架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM应用 CI/CD 全景架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                        输入源 (Inputs)                          │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │  Prompt变更  │  模型更新  │  Schema变更  │  依赖更新  │  配置变更  │        │
│  └──────────────────┬──────────────────────────────────────────────┘        │
│                     │                                                        │
│  ┌──────────────────▼──────────────────────────────────────────────┐        │
│  │                   构建流水线 (Build Pipeline)                    │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │                                                                  │        │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │        │
│  │  │ 版本快照 │→│ 语法检查  │→│ 依赖解析  │→│ 构件生成      │   │        │
│  │  │ (Git)   │  │ (Lint)   │  │ (Lock)   │  │ (Artifact)   │   │        │
│  │  └─────────┘  └──────────┘  └──────────┘  └──────────────┘   │        │
│  │                                                                  │        │
│  └──────────────────┬──────────────────────────────────────────────┘        │
│                     │                                                        │
│  ┌──────────────────▼──────────────────────────────────────────────┐        │
│  │                   评估流水线 (Evaluation Pipeline)               │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │                                                                  │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │        │
│  │  │ 自动评估 │→│ 回归测试  │→│ 对比评估  │→│ 人工审核     │  │        │
│  │  │ (Metrics)│  │ (Golden) │  │ (A/B)    │  │ (Review)     │  │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │        │
│  │                                                                  │        │
│  └──────────────────┬──────────────────────────────────────────────┘        │
│                     │                                                        │
│  ┌──────────────────▼──────────────────────────────────────────────┐        │
│  │                   发布流水线 (Release Pipeline)                  │        │
│  ├─────────────────────────────────────────────────────────────────┤        │
│  │                                                                  │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │        │
│  │  │ 构建镜像 │→│ 灰度发布  │→│ 流量切换  │→│ 全量/回滚    │  │        │
│  │  │ (Docker) │  │ (Canary) │  │ (Traffic)│  │ (Promote)    │  │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │        │
│  │                                                                  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 与传统CI/CD的关键差异

| 维度 | 传统CI/CD | LLM应用CI/CD |
|------|----------|-------------|
| **构建产物** | 二进制文件/Docker镜像 | Prompt模板 + 模型配置 + 评估基准 |
| **测试方式** | 断言式(Assertions) | 评估式(Evaluations) |
| **质量门禁** | 通过/失败(Pass/Fail) | 阈值比较(Threshold) |
| **发布策略** | 蓝绿/滚动 | 灰度+流量比例+在线评估 |
| **回滚依据** | 错误率/延迟 | 语义质量+用户反馈+业务指标 |
| **监控重点** | 系统指标 | 语义指标+系统指标 |

---

## 三、Prompt版本管理

Prompt是LLM应用的核心"代码"，但它不像传统代码那样有明确的语法结构。有效的Prompt版本管理需要解决三个问题：**存储、对比、回滚**。

### 3.1 Prompt存储结构

```
prompts/
├── customer_service/
│   ├── v1/
│   │   ├── system.md          # System Prompt
│   │   ├── examples.yaml      # Few-shot示例
│   │   ├── schema.json        # 输出Schema
│   │   ├── metadata.yaml      # 元信息
│   │   └── evaluation/        # 关联评估结果
│   │       ├── golden_dataset.json
│   │       ├── eval_results.json
│   │       └── score_history.json
│   ├── v2/
│   │   ├── system.md
│   │   ├── examples.yaml
│   │   ├── schema.json
│   │   ├── metadata.yaml
│   │   ├── changelog.md       # 变更说明
│   │   └── evaluation/
│   └── latest -> v2           # 符号链接指向当前版本
│
├── data_extractor/
│   ├── v1/
│   └── v2/
│
└── prompt_registry.yaml       # 全局注册表
```

### 3.2 元信息管理

每个Prompt版本都需要记录详细的元信息：

```yaml
# v2/metadata.yaml
prompt_id: customer_service
version: "2.2.0"
status: production          # draft | staging | production | archived
created_by: "RiceBall"
created_at: "2026-05-28"
last_modified: "2026-05-30"

# 模型依赖
model_requirements:
  min_model: "gpt-4o-mini"
  recommended_model: "gpt-4o"
  context_window: 128000
  supports_json_mode: true

# 评估基准
evaluation:
  dataset: "./evaluation/golden_dataset.json"
  min_score:
    accuracy: 0.92
    format_compliance: 0.99
    latency_p95_ms: 2000
  baseline_comparison:
    version: "2.1.0"
    metrics: ["accuracy", "latency"]

# 变更记录
changelog:
  - version: "2.2.0"
    date: "2026-05-30"
    changes:
      - "优化退货场景的意图识别准确率"
      - "增加urgency字段的默认值推断"
    related_ticket: "FEAT-1234"
    risk_level: medium        # low | medium | high
```

### 3.3 Prompt Diff工具

Prompt的变更是自然语言级别的，传统的文本diff往往噪音太大。需要一个**语义感知的diff工具**：

```python
from dataclasses import dataclass
from difflib import unified_diff
import yaml
import json

@dataclass
class PromptDiff:
    """Prompt版本差异分析"""
    version_a: str
    version_b: str
    sections: list[dict]
    risk_assessment: str

def diff_prompts(v_a_path: str, v_b_path: str) -> PromptDiff:
    """对比两个Prompt版本的差异"""
    
    with open(v_a_path) as f:
        v_a = yaml.safe_load(f)
    with open(v_b_path) as f:
        v_b = yaml.safe_load(f)
    
    sections = []
    
    # 1. System Prompt差异
    if v_a.get("system") != v_b.get("system"):
        sys_diff = list(unified_diff(
            v_a["system"].splitlines(),
            v_b["system"].splitlines(),
            fromfile=f"v{v_a['version']}",
            tofile=f"v{v_b['version']}",
            lineterm=""
        ))
        sections.append({
            "type": "system_prompt",
            "changes": len(sys_diff),
            "diff": "\n".join(sys_diff),
        })
    
    # 2. Few-shot示例差异
    if v_a.get("examples") != v_b.get("examples"):
        sections.append({
            "type": "examples",
            "changes": _count_example_changes(
                v_a.get("examples", []),
                v_b.get("examples", [])
            ),
        })
    
    # 3. Schema差异
    if v_a.get("schema") != v_b.get("schema"):
        schema_changes = _diff_json_schema(
            v_a.get("schema", {}),
            v_b.get("schema", {})
        )
        sections.append({
            "type": "schema",
            "breaking_changes": schema_changes["breaking"],
            "additions": schema_changes["additions"],
        })
    
    # 4. 风险评估
    risk = _assess_risk(sections)
    
    return PromptDiff(
        version_a=v_a["version"],
        version_b=v_b["version"],
        sections=sections,
        risk_assessment=risk,
    )

def _assess_risk(sections: list[dict]) -> str:
    """自动评估变更风险"""
    has_breaking = any(
        s.get("breaking_changes") for s in sections
    )
    has_schema_change = any(
        s["type"] == "schema" for s in sections
    )
    total_changes = sum(s.get("changes", 0) for s in sections)
    
    if has_breaking:
        return "high"
    elif has_schema_change or total_changes > 50:
        return "medium"
    else:
        return "low"
```

---

## 四、自动化评估体系

### 4.1 评估流水线设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    自动化评估流水线                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐     ┌──────────────────────────────────────┐         │
│  │ Golden    │     │           评估引擎                    │         │
│  │ Dataset  │────→│                                       │         │
│  │ (测试集)  │     │  ┌─────────┐ ┌─────────┐ ┌────────┐ │         │
│  └──────────┘     │  │ 精确匹配 │ │ 语义相似 │ │ LLM裁判│ │         │
│                    │  │ (Fast)  │ │(Medium) │ │(Slow)  │ │         │
│                    │  └────┬────┘ └────┬────┘ └───┬────┘ │         │
│                    │       │           │          │       │         │
│                    │       └─────┬─────┘──────────┘       │         │
│                    │             │                        │         │
│                    │      ┌──────▼──────┐                │         │
│                    │      │  聚合评分    │                │         │
│                    │      └──────┬──────┘                │         │
│                    └─────────────┼────────────────────────┘         │
│                                  │                                   │
│                    ┌─────────────▼─────────────┐                   │
│                    │      质量门禁 (Quality Gate)│                   │
│                    │                             │                   │
│                    │  accuracy >= 0.92 ?         │                   │
│                    │  format   >= 0.99 ?         │                   │
│                    │  latency  <= 2000ms ?       │                   │
│                    │  regression_count == 0 ?    │                   │
│                    └─────────────┬─────────────┘                   │
│                                  │                                   │
│                    ┌─────────────▼─────────────┐                   │
│                    │     PASS → 可以发布         │                   │
│                    │     FAIL → 阻止发布+告警    │                   │
│                    └───────────────────────────┘                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 三级评估策略

```python
from enum import Enum
from dataclasses import dataclass
import asyncio

class EvalLevel(Enum):
    FAST = "fast"         # 精确匹配，毫秒级
    MEDIUM = "medium"     # 语义相似，秒级
    SLOW = "slow"         # LLM裁判，十秒级

@dataclass
class EvalResult:
    level: EvalLevel
    score: float
    details: dict
    duration_ms: float

class LLMJudge:
    """LLM裁判评估器：用更强的模型评估当前模型输出"""
    
    EVALUATION_PROMPT = """你是一个严格的质量评估专家。
    
请评估以下AI输出的质量，按照给定的评分标准打分。

## 评估维度
1. 准确性 (0-1): 信息是否正确完整
2. 相关性 (0-1): 是否回答了问题
3. 格式合规 (0-1): 是否符合要求的输出格式
4. 语言质量 (0-1): 表达是否清晰自然

## 评估标准
- 0.9-1.0: 优秀，可直接使用
- 0.7-0.9: 良好，小问题可接受
- 0.5-0.7: 一般，需要改进
- 0.0-0.5: 较差，不可接受

请以JSON格式输出评估结果。"""

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
        judge_model: str = "gpt-4o",
    ) -> EvalResult:
        """使用LLM进行语义级评估"""
        import json
        from openai import OpenAI
        
        client = OpenAI()
        
        eval_input = f"""
## 用户输入
{input_text}

## AI输出
{output_text}

{"## 参考答案" + chr(10) + expected if expected else ""}

请按照评分标准进行评估。"""
        
        start = asyncio.get_event_loop().time()
        
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": self.EVALUATION_PROMPT},
                {"role": "user", "content": eval_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        duration = (asyncio.get_event_loop().time() - start) * 1000
        result = json.loads(response.choices[0].message.content)
        
        overall_score = (
            result["accuracy"] * 0.35 +
            result["relevance"] * 0.25 +
            result["format_compliance"] * 0.25 +
            result["language_quality"] * 0.15
        )
        
        return EvalResult(
            level=EvalLevel.SLOW,
            score=overall_score,
            details=result,
            duration_ms=duration,
        )


class EvaluationPipeline:
    """三级评估流水线"""
    
    def __init__(self, golden_dataset: list[dict]):
        self.dataset = golden_dataset
        self.judge = LLMJudge()
    
    async def run(
        self,
        llm_func,
        levels: list[EvalLevel] = None,
        concurrency: int = 5,
    ) -> dict:
        """
        运行评估流水线
        
        Args:
            llm_func: 被评估的LLM函数
            levels: 评估级别
            concurrency: 并发数
        """
        if levels is None:
            levels = [EvalLevel.FAST, EvalLevel.MEDIUM, EvalLevel.SLOW]
        
        results = []
        
        # Level 1: Fast - 精确匹配 (毫秒级)
        if EvalLevel.FAST in levels:
            fast_results = await self._run_fast_eval(llm_func)
            results.extend(fast_results)
            
            # 快速检查：如果精确匹配失败率超过20%，直接停止
            fast_fail_rate = sum(
                1 for r in fast_results if r.score < 0.8
            ) / len(fast_results)
            if fast_fail_rate > 0.2:
                return self._compile_results(results, "FAST_FAILED")
        
        # Level 2: Medium - 语义相似 (秒级)
        if EvalLevel.MEDIUM in levels:
            medium_results = await self._run_medium_eval(
                llm_func, concurrency
            )
            results.extend(medium_results)
        
        # Level 3: Slow - LLM裁判 (十秒级，抽样)
        if EvalLevel.SLOW in levels:
            # 抽样10%进行LLM裁判评估
            sample_size = max(1, len(self.dataset) // 10)
            sample = self.dataset[:sample_size]
            slow_results = await self._run_slow_eval(
                llm_func, sample, concurrency
            )
            results.extend(slow_results)
        
        return self._compile_results(results, "COMPLETED")
    
    async def _run_fast_eval(self, llm_func) -> list[EvalResult]:
        """精确匹配评估"""
        results = []
        for case in self.dataset:
            output = llm_func(case["input"])
            score = self._exact_match(output, case["expected"])
            results.append(EvalResult(
                level=EvalLevel.FAST,
                score=score,
                details={"match": score == 1.0},
                duration_ms=0,
            ))
        return results
    
    def _compile_results(
        self, results: list[EvalResult], status: str
    ) -> dict:
        """汇总评估结果"""
        import statistics
        
        scores = [r.score for r in results]
        
        return {
            "status": status,
            "total_cases": len(results),
            "metrics": {
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
            },
            "by_level": {
                level.value: [
                    r.score for r in results if r.level == level
                ] for level in EvalLevel
            },
            "quality_gate": {
                "accuracy_pass": statistics.mean(scores) >= 0.92,
                "no_critical_failures": min(scores) >= 0.5,
            },
        }
```

### 4.3 评估基准数据集管理

Golden Dataset是评估质量的基石：

```json
{
  "version": "3.2.0",
  "last_updated": "2026-05-30",
  "cases": [
    {
      "id": "CS-001",
      "category": "intent_recognition",
      "input": "我上周买的手机壳想退货",
      "expected": {
        "intent": "return_item",
        "confidence_gte": 0.8
      },
      "evaluation_criteria": {
        "accuracy": "意图分类必须为return_item",
        "format": "必须输出有效JSON",
        "latency_max_ms": 3000
      },
      "tags": ["退货", "意图识别", "简单场景"]
    },
    {
      "id": "CS-002",
      "category": "intent_recognition",
      "input": "你好呀",
      "expected": {
        "intent": "general_qa",
        "needs_clarification": true
      },
      "evaluation_criteria": {
        "accuracy": "闲聊应识别为general_qa",
        "should_clarify": "无明确需求时应追问"
      },
      "tags": ["闲聊", "意图识别", "边界场景"]
    }
  ],
  "metadata": {
    "total_cases": 200,
    "categories": {
      "intent_recognition": 80,
      "parameter_extraction": 60,
      "edge_cases": 40,
      "multi_turn": 20
    }
  }
}
```

---

## 五、灰度发布策略

### 5.1 灰度发布架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM应用灰度发布架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         用户请求                                      │
│                            │                                         │
│                            ▼                                         │
│                    ┌──────────────┐                                 │
│                    │  流量路由层   │                                 │
│                    │  (Router)    │                                 │
│                    └──────┬───────┘                                 │
│                           │                                          │
│              ┌────────────┼────────────┐                            │
│              │            │            │                            │
│              ▼            ▼            ▼                            │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│     │ Stable   │  │ Canary   │  │ Shadow   │                     │
│     │ (90%)    │  │ (10%)    │  │ (日志)    │                     │
│     │ v2.2.0   │  │ v2.3.0   │  │ v2.4.0   │                     │
│     └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
│          │              │              │                            │
│          │              │              │                            │
│     ┌────▼─────────────▼──────────────▼────┐                      │
│     │           在线评估引擎                 │                      │
│     │                                        │                      │
│     │  • 延迟监控                            │                      │
│     │  • 输出质量抽检                         │                      │
│     │  • 用户反馈收集                         │                      │
│     │  • 错误率统计                           │                      │
│     └────────────────┬─────────────────────┘                      │
│                      │                                              │
│     ┌────────────────▼─────────────────────┐                      │
│     │          决策引擎                      │                      │
│     │                                        │                      │
│     │  IF canary.score >= stable.score       │                      │
│     │     AND canary.latency <= stable.latency*1.2                 │
│     │     AND canary.error_rate < 0.01       │                      │
│     │  THEN → 扩大流量/全量发布               │                      │
│     │  ELSE → 回滚到stable                   │                      │
│     └──────────────────────────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 流量路由实现

```python
import random
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class ModelVersion:
    """模型版本配置"""
    version: str
    model_id: str
    prompt_version: str
    traffic_weight: float = 0.0
    is_active: bool = True
    
    # 运行时指标
    total_requests: int = 0
    error_count: int = 0
    latency_sum_ms: float = 0.0
    quality_scores: list[float] = field(default_factory=list)

@dataclass
class LLMRouter:
    """LLM流量路由器"""
    versions: dict[str, ModelVersion]
    
    def route(self, request_id: str) -> ModelVersion:
        """根据流量权重路由请求"""
        active_versions = [
            v for v in self.versions.values() if v.is_active
        ]
        
        if not active_versions:
            raise RuntimeError("无可用版本")
        
        # 加权随机选择
        weights = [v.traffic_weight for v in active_versions]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        
        chosen = random.choices(
            active_versions, weights=probabilities, k=1
        )[0]
        
        return chosen
    
    def record_result(
        self,
        version: str,
        latency_ms: float,
        is_error: bool,
        quality_score: Optional[float] = None,
    ):
        """记录请求结果"""
        v = self.versions[version]
        v.total_requests += 1
        if is_error:
            v.error_count += 1
        v.latency_sum_ms += latency_ms
        if quality_score is not None:
            v.quality_scores.append(quality_score)
    
    def get_metrics(self, version: str) -> dict:
        """获取版本指标"""
        v = self.versions[version]
        return {
            "version": v.version,
            "total_requests": v.total_requests,
            "error_rate": (
                v.error_count / v.total_requests
                if v.total_requests > 0 else 0
            ),
            "avg_latency_ms": (
                v.latency_sum_ms / v.total_requests
                if v.total_requests > 0 else 0
            ),
            "avg_quality_score": (
                sum(v.quality_scores) / len(v.quality_scores)
                if v.quality_scores else 0
            ),
        }


class CanaryDeploymentManager:
    """灰度发布管理器"""
    
    CANARY_STAGES = [
        {"traffic": 5,  "duration_min": 10, "min_requests": 50},
        {"traffic": 20, "duration_min": 30, "min_requests": 200},
        {"traffic": 50, "duration_min": 60, "min_requests": 500},
        {"traffic": 100, "duration_min": 0,  "min_requests": 0},
    ]
    
    def __init__(self, router: LLMRouter):
        self.router = router
        self.current_stage = 0
        self.stage_start_time = time.time()
    
    def should_promote(self) -> dict:
        """判断是否应该推进灰度"""
        if self.current_stage >= len(self.CANARY_STAGES) - 1:
            return {"action": "already_full", "should_promote": False}
        
        stage = self.CANARY_STAGES[self.current_stage]
        canary_version = self._get_canary_version()
        stable_version = self._get_stable_version()
        
        canary_metrics = self.router.get_metrics(canary_version)
        stable_metrics = self.router.get_metrics(stable_version)
        
        # 检查时间条件
        elapsed_min = (time.time() - self.stage_start_time) / 60
        time_ready = elapsed_min >= stage["duration_min"]
        
        # 检查样本量条件
        sample_ready = (
            canary_metrics["total_requests"] >= stage["min_requests"]
        )
        
        # 检查质量条件
        quality_ok = self._check_quality(canary_metrics, stable_metrics)
        
        if time_ready and sample_ready:
            if quality_ok:
                return {
                    "action": "promote",
                    "should_promote": True,
                    "next_traffic": self.CANARY_STAGES[
                        self.current_stage + 1
                    ]["traffic"],
                    "metrics": canary_metrics,
                }
            else:
                return {
                    "action": "rollback",
                    "should_promote": False,
                    "reason": "质量指标不达标",
                    "metrics": canary_metrics,
                }
        
        return {
            "action": "wait",
            "should_promote": False,
            "elapsed_min": elapsed_min,
            "target_min": stage["duration_min"],
        }
    
    def _check_quality(
        self, canary: dict, stable: dict
    ) -> bool:
        """质量检查"""
        # 错误率检查
        if canary["error_rate"] > 0.01:
            return False
        
        # 延迟检查：不超过stable的1.2倍
        if stable["avg_latency_ms"] > 0:
            latency_ratio = (
                canary["avg_latency_ms"] / stable["avg_latency_ms"]
            )
            if latency_ratio > 1.2:
                return False
        
        # 质量分数检查
        if canary["avg_quality_score"] > 0:
            if stable["avg_quality_score"] > 0:
                if canary["avg_quality_score"] < stable["avg_quality_score"]:
                    return False
        
        return True
```

---

## 六、GitHub Actions实战配置

### 6.1 完整的CI/CD Pipeline

```yaml
# .github/workflows/llm-app-cicd.yml
name: LLM Application CI/CD

on:
  push:
    paths:
      - 'prompts/**'
      - 'schemas/**'
      - 'evaluations/**'
  pull_request:
    paths:
      - 'prompts/**'
      - 'schemas/**'

env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  EVALUATION_THRESHOLD: "0.92"

jobs:
  # Stage 1: 静态检查
  lint-and-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Validate Prompt Structure
        run: |
          python scripts/validate_prompts.py
          
      - name: Validate Schema Compatibility
        run: |
          python scripts/validate_schemas.py
      
      - name: Check Prompt Size Limits
        run: |
          python scripts/check_prompt_limits.py --max-tokens 8000

  # Stage 2: 自动评估
  evaluation:
    needs: lint-and-validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Fast Evaluation
        run: |
          python -m evaluation.run \
            --level fast \
            --dataset evaluations/golden_dataset.json \
            --output eval_results.json
      
      - name: Check Quality Gate
        run: |
          python scripts/quality_gate.py \
            --results eval_results.json \
            --threshold ${{ env.EVALUATION_THRESHOLD }}
      
      - name: Run Regression Test
        run: |
          python -m evaluation.regression \
            --baseline eval_results_baseline.json \
            --current eval_results.json
      
      - name: Upload Evaluation Report
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-report
          path: eval_results.json

  # Stage 3: 对比评估 (仅PR)
  comparison:
    if: github.event_name == 'pull_request'
    needs: evaluation
    runs-on: ubuntu-latest
    steps:
      - name: Compare with Baseline
        run: |
          python -m evaluation.compare \
            --baseline eval_results_baseline.json \
            --current eval_results.json \
            --output comparison.md
      
      - name: Comment PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('comparison.md', 'utf8');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: report
            });

  # Stage 4: 灰度发布 (仅main分支)
  canary-deploy:
    if: github.ref == 'refs/heads/main'
    needs: evaluation
    runs-on: ubuntu-latest
    steps:
      - name: Deploy Canary (5% traffic)
        run: |
          python scripts/canary_deploy.py \
            --version ${{ github.sha }} \
            --traffic 5 \
            --stage 0
      
      - name: Wait for Metrics
        run: sleep 600  # 等待10分钟收集指标
      
      - name: Check Canary Health
        id: canary_check
        run: |
          python scripts/canary_health.py \
            --version ${{ github.sha }} \
            --min-score ${{ env.EVALUATION_THRESHOLD }} \
            --max-error-rate 0.01
      
      - name: Promote Canary
        if: steps.canary_check.outputs.status == 'healthy'
        run: |
          python scripts/canary_promote.py \
            --version ${{ github.sha }} \
            --traffic 100
```

---

## 七、监控与告警

### 7.1 监控看板指标

```python
class LLMMonitoringMetrics:
    """LLM应用监控指标定义"""
    
    # ---- 系统指标 ----
    SYSTEM_METRICS = {
        "llm_request_total": {
            "type": "counter",
            "labels": ["model", "prompt_version", "status"],
            "description": "LLM请求总量",
        },
        "llm_latency_seconds": {
            "type": "histogram",
            "buckets": [0.5, 1, 2, 5, 10, 30],
            "labels": ["model", "prompt_version"],
            "description": "LLM请求延迟",
        },
        "llm_token_usage": {
            "type": "counter",
            "labels": ["model", "prompt_version", "type"],
            "description": "Token使用量 (input/output)",
        },
        "llm_error_total": {
            "type": "counter",
            "labels": ["model", "error_type"],
            "description": "LLM错误总数",
        },
    }
    
    # ---- 质量指标 (LLM特有) ----
    QUALITY_METRICS = {
        "llm_output_quality_score": {
            "type": "gauge",
            "labels": ["prompt_version", "category"],
            "description": "输出质量分数 (0-1)",
        },
        "llm_format_compliance_rate": {
            "type": "gauge",
            "labels": ["prompt_version"],
            "description": "格式合规率",
        },
        "llm_intent_accuracy": {
            "type": "gauge",
            "labels": ["prompt_version", "intent"],
            "description": "意图识别准确率",
        },
        "llm_tool_call_success_rate": {
            "type": "gauge",
            "labels": ["tool_name"],
            "description": "工具调用成功率",
        },
    }
    
    # ---- 成本指标 ----
    COST_METRICS = {
        "llm_cost_usd_total": {
            "type": "counter",
            "labels": ["model", "prompt_version"],
            "description": "LLM调用总成本 (USD)",
        },
        "llm_cost_per_request": {
            "type": "histogram",
            "buckets": [0.001, 0.01, 0.05, 0.1, 0.5],
            "description": "单次请求成本分布",
        },
    }
    
    # ---- 告警规则 ----
    ALERT_RULES = {
        "quality_degradation": {
            "condition": "llm_format_compliance_rate < 0.95 for 5m",
            "severity": "critical",
            "action": "自动回滚到上一个稳定版本",
        },
        "latency_spike": {
            "condition": "llm_latency_seconds_p99 > 10 for 3m",
            "severity": "warning",
            "action": "通知值班工程师",
        },
        "error_rate_high": {
            "condition": "llm_error_total rate > 0.05 for 2m",
            "severity": "critical",
            "action": "自动回滚 + 紧急通知",
        },
        "cost_anomaly": {
            "condition": "llm_cost_usd_total rate > 2x baseline for 1h",
            "severity": "warning",
            "action": "检查是否有异常流量",
        },
    }
```

---

## 八、总结与Checklist

### 8.1 LLM应用CI/CD建设路线图

```
Phase 1: 基础能力 (1-2周)
├── Prompt版本管理 (Git + metadata)
├── 基础自动化测试 (精确匹配)
└── 简单的部署脚本

Phase 2: 评估体系 (2-4周)
├── Golden Dataset管理
├── 三级评估流水线
├── 质量门禁自动化
└── PR级别的自动对比报告

Phase 3: 灰度发布 (4-6周)
├── 流量路由层
├── 灰度发布管理器
├── 自动化回滚机制
└── 在线指标监控

Phase 4: 智能运维 (持续)
├── 自适应评估 (根据数据自动调整阈值)
├── Prompt自动优化 (A/B测试驱动)
├── 成本自动优化 (模型路由)
└── 根因分析 (异常自动诊断)
```

### 8.2 上线前Checklist

| 检查项 | 是否就绪 | 备注 |
|--------|---------|------|
| Prompt版本已纳入Git管理 | ☐ | 有完整的元信息 |
| 输出Schema使用Pydantic定义 | ☐ | 支持自动验证 |
| Golden Dataset覆盖核心场景 | ☐ | 至少100条测试用例 |
| 评估流水线可自动运行 | ☐ | 集成到CI/CD |
| 质量门禁阈值已确定 | ☐ | 有历史基线参考 |
| 灰度发布流程已验证 | ☐ | 有完整的回滚路径 |
| 监控告警已配置 | ☐ | 覆盖质量+系统+成本 |
| 回滚方案已测试 | ☐ | 3分钟内可完成回滚 |
| 成本预算已设定 | ☐ | 有日/周/月预算上限 |

LLM应用的CI/CD不是传统DevOps的简单迁移，而是需要**重新思考软件质量的定义**。当输出从"对/错"变成"好/更好"时，我们的工程体系也必须随之进化。建立这套体系需要时间，但每一步投入都会在生产稳定性中得到回报。
