---
title: "AI Agent评估框架：从能力测试到生产验证的完整方法论"
description: "深度解析AI Agent评估体系设计，覆盖任务完成率、推理质量、工具使用、多轮交互等维度，附自动化评估Pipeline与Benchmark构建实战"
date: 2026-06-01
author: "RiceBall-15"
category: "aiInfra"
subCategory: evaluation
tags: ["AI Agent", "评估框架", "Benchmark", "自动化测试", "LLM评估", "Agent评测", "质量保证"]
draft: false
---

## 一、引言：Agent评估的「测不准原理」

2026年，AI Agent已经广泛应用于代码生成、数据分析、客户服务等场景。但一个核心问题始终困扰着从业者：

> **"Agent在Demo中表现出色，但在生产环境中频繁失败——我们如何系统地评估Agent的真实能力？"**

Agent评估的困难在于其**非确定性**和**任务复杂性**。传统的软件测试基于确定性输入输出，但Agent的输出具有随机性，且同一个任务可能有多种正确的解决路径。这导致：

- 传统的Pass/Fail测试无法衡量Agent能力
- 人工评估成本高昂且不可扩展
- 评估标准难以量化和统一

本文将构建一套完整的Agent评估方法论，覆盖：

- Agent评估的多维度框架设计
- 自动化评估Pipeline架构
- Benchmark数据集构建方法
- 生产环境持续评估策略
- 评估结果可视化与决策支持

---

## 二、Agent评估维度体系

### 2.1 评估维度全景

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI Agent评估维度体系                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         功能性维度                                    │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │ 任务完成率   │  │ 推理质量     │  │ 工具使用能力 │              │    │
│  │  │ (30%)       │  │ (25%)       │  │ (20%)       │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         非功能性维度                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │ 响应延迟     │  │ 资源消耗     │  │ 安全性       │              │    │
│  │  │ (10%)       │  │ (10%)       │  │ (5%)        │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         交互性维度                                    │    │
│  │  ┌──────────────┐  ┌──────────────┐                                │    │
│  │  │ 多轮对话能力 │  │ 用户体验     │                                │    │
│  │  │ (未计权重)   │  │ (参考指标)   │                                │    │
│  │  └──────────────┘  └──────────────┘                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 各维度详细定义

| 维度 | 子指标 | 权重 | 评估方法 | 及格线 |
|------|--------|------|---------|--------|
| **任务完成率** | 完全完成率 | 15% | 自动化判定 | ≥ 80% |
| | 部分完成率 | 10% | 人工评分 | ≥ 60% |
| | 错误恢复率 | 5% | 自动化测试 | ≥ 70% |
| **推理质量** | 逻辑正确性 | 10% | LLM-as-Judge | ≥ 85% |
| | 推理步骤合理性 | 10% | 人工评分 | ≥ 80% |
| | 创新性解决方案 | 5% | 人工评分 | ≥ 70% |
| **工具使用** | 工具选择准确性 | 10% | 自动化判定 | ≥ 90% |
| | 工具调用效率 | 5% | 自动化测试 | ≤ 1.5x最优 |
| | 工具结果整合 | 5% | LLM-as-Judge | ≥ 80% |
| **响应延迟** | 首Token延迟 | 5% | 基准测试 | ≤ 2s |
| | 总响应时间 | 5% | 基准测试 | ≤ 30s |
| **资源消耗** | Token效率 | 5% | 自动化测试 | ≥ 0.8 |
| | 成本效益比 | 5% | 成本分析 | ≤ 目标成本 |
| **安全性** | Prompt注入防护 | 3% | 红队测试 | 100% |
| | 敏感信息泄露 | 2% | 自动化扫描 | 0泄露 |

---

## 三、自动化评估Pipeline架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    AI Agent自动化评估Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         数据层                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Test Cases   │  │ Golden Set   │  │ Real User    │  │ Synthetic    │  │  │
│  │  │ (测试用例)   │  │ (标准答案)   │  │ Queries      │  │ Data         │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                   │
│  ┌───────────────────────────▼───────────────────────────────────────────────┐  │
│  │                         执行层                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │  │
│  │  │ Agent Runner │  │ Parallel     │  │ Timeout      │                    │  │
│  │  │ (执行Agent)  │  │ Executor     │  │ Manager      │                    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                   │
│  ┌───────────────────────────▼───────────────────────────────────────────────┐  │
│  │                         评估层                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │  │
│  │  │ Auto Judge   │  │ LLM Judge    │  │ Human Judge  │                    │  │
│  │  │ (规则判定)   │  │ (模型评判)   │  │ (人工评判)   │                    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                   │
│  ┌───────────────────────────▼───────────────────────────────────────────────┐  │
│  │                         分析层                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │  │
│  │  │ Score Computation │ │ Trend Analysis │ │ Report Generation│            │  │
│  │  │ (评分计算)   │  │ (趋势分析)   │  │ (报告生成)   │                    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心实现

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import asyncio
import time
import json

class Verdict(Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    ERROR = "error"

@dataclass
class TestCase:
    """测试用例"""
    case_id: str
    task_description: str
    expected_output: Optional[str] = None
    expected_tools: List[str] = field(default_factory=list)
    max_latency_ms: float = 30000
    max_tokens: int = 10000
    tags: List[str] = field(default_factory=list)
    
@dataclass
class EvalResult:
    """评估结果"""
    case_id: str
    verdict: Verdict
    score: float  # 0.0 - 1.0
    latency_ms: float
    tokens_used: int
    tools_called: List[str]
    output_preview: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class AgentEvaluator:
    """Agent评估器"""
    
    def __init__(self, agent_fn: Callable, config: Dict = None):
        self.agent_fn = agent_fn
        self.config = config or {}
        self.judges = {
            "auto": self._auto_judge,
            "llm": self._llm_judge,
            "hybrid": self._hybrid_judge,
        }
        
    async def evaluate_single(self, test_case: TestCase, 
                              judge_type: str = "hybrid") -> EvalResult:
        """评估单个测试用例"""
        start_time = time.time()
        
        try:
            # 执行Agent
            result = await asyncio.wait_for(
                self.agent_fn(test_case.task_description),
                timeout=test_case.max_latency_ms / 1000
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # 判断结果
            judge_fn = self.judges.get(judge_type, self._auto_judge)
            verdict, score, details = await judge_fn(test_case, result)
            
            return EvalResult(
                case_id=test_case.case_id,
                verdict=verdict,
                score=score,
                latency_ms=latency_ms,
                tokens_used=result.get("tokens", 0),
                tools_called=result.get("tools", []),
                output_preview=str(result.get("output", ""))[:200],
                details=details
            )
            
        except asyncio.TimeoutError:
            return EvalResult(
                case_id=test_case.case_id,
                verdict=Verdict.ERROR,
                score=0.0,
                latency_ms=test_case.max_latency_ms,
                tokens_used=0,
                tools_called=[],
                output_preview="",
                error="Timeout"
            )
        except Exception as e:
            return EvalResult(
                case_id=test_case.case_id,
                verdict=Verdict.ERROR,
                score=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                tools_called=[],
                output_preview="",
                error=str(e)
            )
    
    async def evaluate_batch(self, test_cases: List[TestCase],
                             parallel: int = 5,
                             judge_type: str = "hybrid") -> List[EvalResult]:
        """批量评估"""
        semaphore = asyncio.Semaphore(parallel)
        
        async def eval_with_limit(tc):
            async with semaphore:
                return await self.evaluate_single(tc, judge_type)
        
        tasks = [eval_with_limit(tc) for tc in test_cases]
        return await asyncio.gather(*tasks)
    
    async def _auto_judge(self, test_case: TestCase, 
                          result: dict) -> tuple:
        """自动规则判定"""
        output = result.get("output", "")
        tools = result.get("tools", [])
        
        # 检查是否完成
        completed = self._check_completion(test_case, output)
        
        # 检查工具使用
        tools_correct = self._check_tools(test_case.expected_tools, tools)
        
        # 计算分数
        score = 0.0
        if completed:
            score += 0.7
        if tools_correct:
            score += 0.3
            
        verdict = Verdict.PASS if score >= 0.8 else (
            Verdict.PARTIAL if score >= 0.5 else Verdict.FAIL
        )
        
        return verdict, score, {
            "completed": completed,
            "tools_correct": tools_correct
        }
    
    async def _llm_judge(self, test_case: TestCase, 
                         result: dict) -> tuple:
        """LLM评判"""
        # 构建评判Prompt
        judge_prompt = f"""请评估以下Agent任务的完成质量。

任务描述: {test_case.task_description}

Agent输出: {result.get('output', '')}

预期输出: {test_case.expected_output or '无具体预期'}

请从以下维度评分（0-1）:
1. 任务完成度: 是否完成了任务的核心目标？
2. 输出质量: 输出是否准确、完整、有条理？
3. 工具使用: 是否合理使用了工具？

请返回JSON格式:
{{"completion": 0.0-1.0, "quality": 0.0-1.0, "tool_use": 0.0-1.0, "overall": 0.0-1.0, "feedback": "简短评语"}}
"""
        
        # 调用LLM评判（实际实现中调用GPT-4o等）
        # 这里简化为模拟
        judge_result = {
            "completion": 0.85,
            "quality": 0.80,
            "tool_use": 0.90,
            "overall": 0.85,
            "feedback": "任务基本完成，输出质量良好"
        }
        
        score = judge_result["overall"]
        verdict = Verdict.PASS if score >= 0.8 else (
            Verdict.PARTIAL if score >= 0.5 else Verdict.FAIL
        )
        
        return verdict, score, judge_result
    
    async def _hybrid_judge(self, test_case: TestCase, 
                            result: dict) -> tuple:
        """混合评判（自动 + LLM）"""
        # 先进行自动判定
        auto_verdict, auto_score, auto_details = await self._auto_judge(
            test_case, result
        )
        
        # 如果自动判定为FAIL，进行LLM评判以获得更详细的反馈
        if auto_verdict == Verdict.FAIL:
            llm_verdict, llm_score, llm_details = await self._llm_judge(
                test_case, result
            )
            return llm_verdict, llm_score, {
                "auto": auto_details,
                "llm": llm_details
            }
        
        return auto_verdict, auto_score, auto_details
    
    def _check_completion(self, test_case: TestCase, output: str) -> bool:
        """检查任务是否完成"""
        if test_case.expected_output:
            # 简单的相似度检查（实际应用中使用更复杂的匹配）
            return self._similarity(output, test_case.expected_output) > 0.7
        return bool(output)  # 有输出即视为完成
    
    def _check_tools(self, expected: List[str], actual: List[str]) -> bool:
        """检查工具使用是否正确"""
        if not expected:
            return True
        return all(t in actual for t in expected)
    
    def _similarity(self, a: str, b: str) -> float:
        """计算相似度（简化版）"""
        # 实际应用中使用embedding相似度
        return len(set(a.lower().split()) & set(b.lower().split())) / max(
            len(set(a.lower().split())), len(set(b.lower().split())), 1
        )
```

---

## 四、Benchmark数据集构建

### 4.1 数据集结构设计

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json

class TaskCategory(Enum):
    """任务类别"""
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_WRITING = "document_writing"
    RESEARCH = "research"
    PLANNING = "planning"
    TROUBLESHOOTING = "troubleshooting"

class DifficultyLevel(Enum):
    """难度等级"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

@dataclass
class BenchmarkCase:
    """Benchmark测试用例"""
    case_id: str
    category: TaskCategory
    difficulty: DifficultyLevel
    task: str
    context: str  # 上下文信息
    expected_output: Optional[str] = None
    expected_tools: List[str] = field(default_factory=list)
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    source: str = ""  # 数据来源
    
@dataclass
class BenchmarkSuite:
    """Benchmark测试集"""
    name: str
    version: str
    description: str
    cases: List[BenchmarkCase]
    metadata: Dict[str, Any] = field(default_factory=dict)

class BenchmarkBuilder:
    """Benchmark构建器"""
    
    def __init__(self):
        self.cases: List[BenchmarkCase] = []
        
    def add_case(self, case: BenchmarkCase):
        """添加测试用例"""
        self.cases.append(case)
        
    def add_code_generation_cases(self):
        """添加代码生成测试用例"""
        cases = [
            BenchmarkCase(
                case_id="code_001",
                category=TaskCategory.CODE_GENERATION,
                difficulty=DifficultyLevel.EASY,
                task="编写一个Python函数，计算斐波那契数列的第n项",
                context="需要处理边界情况，n>=0",
                expected_output="def fibonacci(n): ...",
                expected_tools=["code_execution"],
                evaluation_criteria={
                    "correctness": "能正确计算斐波那契数",
                    "efficiency": "时间复杂度O(n)或更好",
                    "edge_cases": "处理n=0, n=1等边界情况"
                }
            ),
            BenchmarkCase(
                case_id="code_002",
                category=TaskCategory.CODE_GENERATION,
                difficulty=DifficultyLevel.MEDIUM,
                task="实现一个LRU缓存，支持get和put操作，时间复杂度O(1)",
                context="使用Python，需要线程安全",
                expected_tools=["code_execution", "testing"],
                evaluation_criteria={
                    "correctness": "get/put操作正确",
                    "complexity": "O(1)时间复杂度",
                    "thread_safety": "支持并发访问"
                }
            ),
            BenchmarkCase(
                case_id="code_003",
                category=TaskCategory.CODE_GENERATION,
                difficulty=DifficultyLevel.HARD,
                task="设计并实现一个分布式任务调度器",
                context="支持任务依赖、优先级、失败重试，使用消息队列",
                expected_tools=["code_execution", "architecture_design", "testing"],
                evaluation_criteria={
                    "architecture": "架构设计合理",
                    "functionality": "支持所有需求功能",
                    "scalability": "可扩展性良好"
                }
            ),
        ]
        self.cases.extend(cases)
        return self
    
    def add_data_analysis_cases(self):
        """添加数据分析测试用例"""
        cases = [
            BenchmarkCase(
                case_id="data_001",
                category=TaskCategory.DATA_ANALYSIS,
                difficulty=DifficultyLevel.EASY,
                task="分析销售数据，找出Top 10产品和月度趋势",
                context="CSV文件，包含product_name, sales_amount, date字段",
                expected_tools=["pandas", "visualization"],
                evaluation_criteria={
                    "accuracy": "计算结果正确",
                    "insight": "提供有价值的洞察",
                    "visualization": "图表清晰易懂"
                }
            ),
            BenchmarkCase(
                case_id="data_002",
                category=TaskCategory.DATA_ANALYSIS,
                difficulty=DifficultyLevel.MEDIUM,
                task="用户行为分析：构建用户分群模型",
                context="包含user_id, action, timestamp的事件日志",
                expected_tools=["pandas", "sklearn", "visualization"],
                evaluation_criteria={
                    "methodology": "分群方法合理",
                    "interpretation": "结果可解释",
                    "actionable": "提供业务建议"
                }
            ),
        ]
        self.cases.extend(cases)
        return self
    
    def build(self) -> BenchmarkSuite:
        """构建Benchmark测试集"""
        return BenchmarkSuite(
            name="AgentBench-2026",
            version="1.0",
            description="AI Agent综合能力评测集",
            cases=self.cases,
            metadata={
                "total_cases": len(self.cases),
                "categories": list(set(c.category.value for c in self.cases)),
                "difficulties": list(set(c.difficulty.value for c in self.cases))
            }
        )
    
    def export(self, filepath: str):
        """导出为JSON"""
        suite = self.build()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "name": suite.name,
                "version": suite.version,
                "description": suite.description,
                "metadata": suite.metadata,
                "cases": [
                    {
                        "case_id": c.case_id,
                        "category": c.category.value,
                        "difficulty": c.difficulty.value,
                        "task": c.task,
                        "context": c.context,
                        "expected_output": c.expected_output,
                        "expected_tools": c.expected_tools,
                        "evaluation_criteria": c.evaluation_criteria,
                        "tags": c.tags
                    }
                    for c in suite.cases
                ]
            }, f, ensure_ascii=False, indent=2)
```

### 4.2 多维度评估矩阵

| 任务类别 | 简单 | 中等 | 困难 | 专家 | 评估重点 |
|---------|------|------|------|------|---------|
| **代码生成** | 函数实现 | 数据结构 | 系统设计 | 架构优化 | 正确性、效率、可维护性 |
| **数据分析** | 基础统计 | 分群建模 | 预测分析 | 因果推断 | 准确性、洞察、可视化 |
| **文档写作** | 简单报告 | 技术文档 | 方案设计 | 战略规划 | 完整性、准确性、逻辑性 |
| **研究** | 信息检索 | 综合分析 | 创新发现 | 前沿探索 | 全面性、深度、原创性 |
| **规划** | 简单任务 | 多步骤 | 复杂项目 | 战略规划 | 合理性、可行性、风险控制 |
| **故障排查** | 简单错误 | 复杂错误 | 系统故障 | 架构问题 | 定位准确性、修复方案 |

---

## 五、LLM-as-Judge评估系统

### 5.1 评估Prompt设计

```python
class LLMJudge:
    """LLM评判系统"""
    
    JUDGE_PROMPT_TEMPLATE = """你是一个专业的AI Agent评估专家。请根据以下标准评估Agent的任务完成质量。

## 任务描述
{task_description}

## Agent输出
{agent_output}

## 评估维度
请从以下维度进行评分（每项0-10分）：

### 1. 任务完成度 (Task Completion)
- 是否完成了任务的核心目标？
- 是否有遗漏的关键部分？
- 评分标准：0-完全未完成，5-部分完成，10-完全完成

### 2. 输出质量 (Output Quality)
- 输出是否准确无误？
- 内容是否完整全面？
- 结构是否清晰有条理？
- 评分标准：0-质量很差，5-质量一般，10-质量优秀

### 3. 推理过程 (Reasoning Process)
- 推理逻辑是否正确？
- 步骤是否合理有序？
- 是否有创新性思考？
- 评分标准：0-逻辑混乱，5-逻辑基本正确，10-逻辑严谨创新

### 4. 工具使用 (Tool Usage)
- 是否选择了合适的工具？
- 工具调用是否高效？
- 工具结果是否有效整合？
- 评分标准：0-完全错误，5-基本正确，10-最优使用

## 输出格式
请严格按照以下JSON格式输出评估结果：
```json
{{
    "task_completion": {{
        "score": 0-10,
        "reasoning": "评分理由"
    }},
    "output_quality": {{
        "score": 0-10,
        "reasoning": "评分理由"
    }},
    "reasoning_process": {{
        "score": 0-10,
        "reasoning": "评分理由"
    }},
    "tool_usage": {{
        "score": 0-10,
        "reasoning": "评分理由"
    }},
    "overall_score": 0-10,
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["不足1", "不足2"],
    "improvement_suggestions": ["建议1", "建议2"]
}}
```
"""
    
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
        
    async def evaluate(self, task: str, output: str) -> dict:
        """使用LLM进行评估"""
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            task_description=task,
            agent_output=output
        )
        
        # 调用LLM（实际实现中调用API）
        # 这里返回模拟结果
        return {
            "task_completion": {"score": 8, "reasoning": "基本完成任务"},
            "output_quality": {"score": 7, "reasoning": "质量良好"},
            "reasoning_process": {"score": 8, "reasoning": "逻辑清晰"},
            "tool_usage": {"score": 9, "reasoning": "工具使用恰当"},
            "overall_score": 8.0,
            "strengths": ["推理清晰", "工具使用得当"],
            "weaknesses": ["可以更详细"],
            "improvement_suggestions": ["增加更多细节说明"]
        }
```

### 5.2 多评判者一致性保障

```python
import asyncio
from typing import List
import statistics

class MultiJudgeEvaluator:
    """多评判者评估系统"""
    
    def __init__(self, judges: List[LLMJudge]):
        self.judges = judges
        
    async def evaluate_with_consensus(self, task: str, output: str,
                                       min_judges: int = 3) -> dict:
        """使用多个评判者进行评估，确保一致性"""
        if len(self.judges) < min_judges:
            raise ValueError(f"需要至少{min_judges}个评判者")
            
        # 并行调用所有评判者
        tasks = [judge.evaluate(task, output) for judge in self.judges]
        results = await asyncio.gather(*tasks)
        
        # 计算一致性
        scores = [r["overall_score"] for r in results]
        consistency = self._calculate_consistency(scores)
        
        # 计算加权平均（基于一致性）
        if consistency >= 0.8:  # 高一致性
            final_score = statistics.mean(scores)
        else:  # 低一致性，使用中位数
            final_score = statistics.median(scores)
            
        return {
            "final_score": final_score,
            "consistency": consistency,
            "individual_scores": scores,
            "detailed_results": results,
            "agreement_level": "high" if consistency >= 0.8 else "low"
        }
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """计算评判者一致性"""
        if len(scores) < 2:
            return 1.0
            
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        
        # 使用变异系数的倒数作为一致性指标
        if mean == 0:
            return 0.0
            
        cv = stdev / mean
        consistency = max(0, 1 - cv)
        
        return consistency
```

---

## 六、生产环境持续评估

### 6.1 评估Pipeline架构

```python
from datetime import datetime, timedelta
from typing import List, Dict
import asyncio

class ContinuousEvaluator:
    """持续评估系统"""
    
    def __init__(self, agent_fn, evaluator: AgentEvaluator):
        self.agent_fn = agent_fn
        self.evaluator = evaluator
        self.history: List[Dict] = []
        
    async def run_continuous_evaluation(self, 
                                        test_suite: BenchmarkSuite,
                                        interval_hours: int = 24,
                                        sample_ratio: float = 0.1):
        """运行持续评估"""
        while True:
            # 采样测试用例
            sampled_cases = self._sample_cases(test_suite, sample_ratio)
            
            # 执行评估
            results = await self.evaluator.evaluate_batch(
                sampled_cases, parallel=5
            )
            
            # 记录结果
            eval_record = {
                "timestamp": datetime.now().isoformat(),
                "suite_version": test_suite.version,
                "cases_evaluated": len(results),
                "results": [
                    {
                        "case_id": r.case_id,
                        "verdict": r.verdict.value,
                        "score": r.score,
                        "latency_ms": r.latency_ms,
                        "tokens": r.tokens_used
                    }
                    for r in results
                ],
                "summary": self._compute_summary(results)
            }
            
            self.history.append(eval_record)
            
            # 检查是否需要告警
            self._check_alerts(eval_record)
            
            # 等待下一轮
            await asyncio.sleep(interval_hours * 3600)
    
    def _sample_cases(self, suite: BenchmarkSuite, 
                      ratio: float) -> List[BenchmarkCase]:
        """采样测试用例"""
        import random
        n = max(1, int(len(suite.cases) * ratio))
        return random.sample(suite.cases, n)
    
    def _compute_summary(self, results: List[EvalResult]) -> Dict:
        """计算评估摘要"""
        total = len(results)
        passed = sum(1 for r in results if r.verdict == Verdict.PASS)
        partial = sum(1 for r in results if r.verdict == Verdict.PARTIAL)
        failed = sum(1 for r in results if r.verdict == Verdict.FAIL)
        errors = sum(1 for r in results if r.verdict == Verdict.ERROR)
        
        avg_score = sum(r.score for r in results) / total if total > 0 else 0
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
        total_tokens = sum(r.tokens_used for r in results)
        
        return {
            "pass_rate": passed / total if total > 0 else 0,
            "partial_rate": partial / total if total > 0 else 0,
            "fail_rate": failed / total if total > 0 else 0,
            "error_rate": errors / total if total > 0 else 0,
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
            "total_tokens": total_tokens
        }
    
    def _check_alerts(self, eval_record: Dict):
        """检查是否需要告警"""
        summary = eval_record["summary"]
        
        # 失败率告警
        if summary["fail_rate"] > 0.2:
            self._send_alert(
                "HIGH_FAILURE_RATE",
                f"失败率: {summary['fail_rate']:.1%}"
            )
            
        # 错误率告警
        if summary["error_rate"] > 0.1:
            self._send_alert(
                "HIGH_ERROR_RATE",
                f"错误率: {summary['error_rate']:.1%}"
            )
            
        # 延迟告警
        if summary["avg_latency_ms"] > 30000:
            self._send_alert(
                "HIGH_LATENCY",
                f"平均延迟: {summary['avg_latency_ms']:.0f}ms"
            )
    
    def _send_alert(self, alert_type: str, message: str):
        """发送告警"""
        # 实际实现中发送到Slack/PagerDuty等
        print(f"ALERT [{alert_type}]: {message}")
    
    def get_trend(self, days: int = 7) -> Dict:
        """获取趋势数据"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [h for h in self.history if h["timestamp"] >= cutoff]
        
        if not recent:
            return {}
            
        return {
            "period_days": days,
            "evaluations_count": len(recent),
            "avg_pass_rate": sum(
                h["summary"]["pass_rate"] for h in recent
            ) / len(recent),
            "avg_score": sum(
                h["summary"]["avg_score"] for h in recent
            ) / len(recent),
            "trend": self._calculate_trend(recent)
        }
    
    def _calculate_trend(self, records: List[Dict]) -> str:
        """计算趋势方向"""
        if len(records) < 2:
            return "insufficient_data"
            
        scores = [r["summary"]["avg_score"] for r in records]
        first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        
        if second_half > first_half * 1.05:
            return "improving"
        elif second_half < first_half * 0.95:
            return "declining"
        else:
            return "stable"
```

---

## 七、评估结果可视化

### 7.1 Grafana Dashboard配置

```json
{
  "dashboard": {
    "title": "AI Agent 评估监控",
    "panels": [
      {
        "title": "任务完成率趋势",
        "type": "timeseries",
        "targets": [{
          "expr": "agent_eval_pass_rate",
          "legendFormat": "完成率"
        }]
      },
      {
        "title": "各维度评分",
        "type": "radar",
        "targets": [
          {"expr": "agent_eval_task_completion", "legendFormat": "任务完成"},
          {"expr": "agent_eval_output_quality", "legendFormat": "输出质量"},
          {"expr": "agent_eval_reasoning", "legendFormat": "推理过程"},
          {"expr": "agent_eval_tool_usage", "legendFormat": "工具使用"}
        ]
      },
      {
        "title": "按类别统计",
        "type": "barchart",
        "targets": [{
          "expr": "sum by (category) (agent_eval_score)",
          "legendFormat": "{{category}}"
        }]
      },
      {
        "title": "按难度统计",
        "type": "barchart",
        "targets": [{
          "expr": "sum by (difficulty) (agent_eval_score)",
          "legendFormat": "{{difficulty}}"
        }]
      }
    ]
  }
}
```

### 7.2 评估报告生成

```python
class EvaluationReporter:
    """评估报告生成器"""
    
    def generate_report(self, results: List[EvalResult], 
                        suite: BenchmarkSuite) -> str:
        """生成评估报告"""
        summary = self._compute_summary(results)
        
        report = f"""# AI Agent评估报告

## 评估概览
- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 测试集: {suite.name} v{suite.version}
- 测试用例数: {len(results)}

## 总体评分
- **总分: {summary['avg_score']:.2f}/1.0**
- 通过率: {summary['pass_rate']:.1%}
- 部分完成率: {summary['partial_rate']:.1%}
- 失败率: {summary['fail_rate']:.1%}

## 详细统计
| 指标 | 值 |
|------|-----|
| 平均延迟 | {summary['avg_latency_ms']:.0f}ms |
| 总Token消耗 | {summary['total_tokens']:,} |
| 错误数 | {summary['error_count']} |

## 各类别表现
{self._category_breakdown(results)}

## 改进建议
{self._generate_recommendations(results)}
"""
        return report
    
    def _compute_summary(self, results: List[EvalResult]) -> Dict:
        """计算摘要"""
        total = len(results)
        passed = sum(1 for r in results if r.verdict == Verdict.PASS)
        partial = sum(1 for r in results if r.verdict == Verdict.PARTIAL)
        failed = sum(1 for r in results if r.verdict == Verdict.FAIL)
        errors = sum(1 for r in results if r.verdict == Verdict.ERROR)
        
        return {
            "avg_score": sum(r.score for r in results) / total if total else 0,
            "pass_rate": passed / total if total else 0,
            "partial_rate": partial / total if total else 0,
            "fail_rate": failed / total if total else 0,
            "error_count": errors,
            "avg_latency_ms": sum(r.latency_ms for r in results) / total if total else 0,
            "total_tokens": sum(r.tokens_used for r in results)
        }
    
    def _category_breakdown(self, results: List[EvalResult]) -> str:
        """按类别分析"""
        # 按case_id前缀分组
        categories = {}
        for r in results:
            cat = r.case_id.split("_")[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        lines = ["| 类别 | 数量 | 平均分 | 通过率 |", "|------|------|--------|--------|"]
        for cat, cat_results in categories.items():
            avg_score = sum(r.score for r in cat_results) / len(cat_results)
            pass_rate = sum(1 for r in cat_results if r.verdict == Verdict.PASS) / len(cat_results)
            lines.append(f"| {cat} | {len(cat_results)} | {avg_score:.2f} | {pass_rate:.1%} |")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, results: List[EvalResult]) -> str:
        """生成改进建议"""
        low_score_cases = [r for r in results if r.score < 0.6]
        
        if not low_score_cases:
            return "所有测试用例表现良好，继续保持！"
        
        recommendations = []
        
        # 分析失败原因
        tool_failures = sum(1 for r in low_score_cases 
                          if "tool" in str(r.details).lower())
        if tool_failures > len(low_score_cases) * 0.3:
            recommendations.append("- 工具使用能力需要加强，建议增加工具调用训练")
        
        latency_issues = sum(1 for r in low_score_cases 
                           if r.latency_ms > 30000)
        if latency_issues > len(low_score_cases) * 0.2:
            recommendations.append("- 响应延迟过高，建议优化Prompt或使用更快的模型")
        
        return "\n".join(recommendations) if recommendations else "无特别建议"
```

---

## 八、最佳实践与总结

### 8.1 评估体系设计原则

| 原则 | 说明 | 实践建议 |
|------|------|---------|
| **全面性** | 覆盖功能性和非功能性维度 | 建立多维度评估矩阵 |
| **可重复性** | 评估结果可复现 | 使用固定种子、版本化测试集 |
| **可扩展性** | 支持新增评估维度 | 模块化评估组件设计 |
| **成本效益** | 评估成本可控 | 分层评估策略（快速/完整） |
| **持续性** | 持续监控Agent质量 | 集成到CI/CD流水线 |

### 8.2 评估策略分层

```
┌─────────────────────────────────────────────────────────────────┐
│                    评估策略分层                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  第一层：快速冒烟测试（每次提交）                                   │
│  - 10-20个核心用例                                               │
│  - 自动判定，< 5分钟                                             │
│  - 通过才允许合并                                                │
│                                                                  │
│  第二层：完整测试（每日/每周）                                     │
│  - 100-200个用例                                                │
│  - 混合判定（自动+LLM）                                         │
│  - 生成详细报告                                                  │
│                                                                  │
│  第三层：深度评估（每月/版本发布前）                                │
│  - 500+用例                                                     │
│  - 多评判者评估                                                  │
│  - 人工抽样审核                                                  │
│  - 竞品对比分析                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 核心要点总结

1. **评估不是一次性的**：建立持续评估机制，监控Agent质量趋势
2. **多维度评估**：不要只看任务完成率，还要评估推理质量、工具使用、安全性等
3. **自动化优先**：尽可能使用自动化判定，LLM-as-Judge作为补充
4. **数据驱动**：用评估数据指导Agent优化，而非凭感觉
5. **成本意识**：评估本身也有成本，设计分层策略平衡覆盖度和成本

### 8.4 未来趋势

- **Agent-as-Judge**：用Agent来评估Agent，形成评估闭环
- **自适应评估**：根据Agent能力动态调整评估难度
- **多模态评估**：覆盖文本、图像、代码等多模态任务
- **实时评估**：在生产环境中进行在线评估和A/B测试
- **评估标准化**：行业将形成统一的Agent评估标准和Benchmark

---

## 参考资料

1. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)
2. [GAIA: A Benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)
3. [LLM-as-a-Judge: Using LLMs for Evaluation](https://arxiv.org/abs/2306.05685)
4. [LangSmith Evaluation Guide](https://docs.smith.langchain.com/evaluation)
5. [Agent Evaluation Best Practices](https://www.langchain.com/blog/agent-evaluation)
