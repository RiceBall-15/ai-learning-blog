---
title: "Agent技能系统设计：从工具调用到技能进化的完整架构"
description: "系统解析Agent技能系统的设计原理，覆盖工具注册、技能发现、技能组合、技能进化等核心机制"
date: 2026-05-31
author: "RiceBall-15"
category: "agent"
subCategory: agent-skill
tags: ["Agent技能", "工具系统", "技能进化", "能力扩展"]
draft: false
---

# Agent技能系统设计：从工具调用到技能进化的完整架构

## 核心问题：Agent如何获得和管理能力？

Agent的核心能力来自它能调用的工具。但简单的工具列表不够——Agent需要：
1. **动态发现**：运行时知道有哪些工具可用
2. **智能选择**：根据任务选择最合适的工具
3. **技能组合**：多个工具组合完成复杂任务
4. **技能进化**：从经验中学习新的技能组合

技能系统的目标是将"工具调用"升级为"技能管理"。

---

## 一、技能系统架构

### 1.1 技能层次模型

```
┌─────────────────────────────────────────┐
│           复合技能 (Composite)            │
│      多个原子技能的组合                   │
├─────────────────────────────────────────┤
│           原子技能 (Atomic)              │
│      单一功能的完整实现                   │
├─────────────────────────────────────────┤
│           基础工具 (Tool)                │
│      最小功能单元                        │
└─────────────────────────────────────────┘
```

### 1.2 技能 vs 工具

| 维度 | 工具 | 技能 |
|------|------|------|
| **粒度** | 单一操作 | 多步骤流程 |
| **智能** | 被动调用 | 主动选择 |
| **组合** | 独立使用 | 可组合 |
| **学习** | 静态 | 可进化 |
| **示例** | `search(query)` | `信息检索技能` |

---

## 二、工具注册与管理

### 2.1 工具定义

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Callable
import json

@dataclass
class ToolDefinition:
    """工具定义"""
    name: str                    # 工具名称
    description: str             # 工具描述
    parameters: Dict[str, Any]   # 参数Schema
    function: Callable           # 实现函数
    category: str = "general"    # 分类
    tags: List[str] = None       # 标签
    required_capabilities: List[str] = None  # 所需能力
    
    def to_prompt(self) -> str:
        """生成LLM可用的工具描述"""
        return f"""
工具名称：{self.name}
描述：{self.description}
参数：{json.dumps(self.parameters, ensure_ascii=False, indent=2)}
"""
```

### 2.2 工具注册中心

```python
class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.categories: Dict[str, List[str]] = {}
        self.capability_map: Dict[str, List[str]] = {}
    
    def register(self, tool: ToolDefinition):
        """注册工具"""
        self.tools[tool.name] = tool
        
        # 分类索引
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        self.categories[tool.category].append(tool.name)
        
        # 能力索引
        if tool.required_capabilities:
            for cap in tool.required_capabilities:
                if cap not in self.capability_map:
                    self.capability_map[cap] = []
                self.capability_map[cap].append(tool.name)
    
    def get_tools_for_task(self, task_description: str, available_capabilities: List[str] = None) -> List[ToolDefinition]:
        """根据任务描述获取可用工具"""
        if available_capabilities is None:
            # 返回所有工具
            return list(self.tools.values())
        
        # 过滤：只返回有能力调用的工具
        available_tools = []
        for tool in self.tools.values():
            if tool.required_capabilities is None:
                available_tools.append(tool)
            elif all(cap in available_capabilities for cap in tool.required_capabilities):
                available_tools.append(tool)
        
        return available_tools
    
    def get_tool_prompt(self, task_type: str = None) -> str:
        """生成工具描述Prompt"""
        tools = self.tools.values()
        if task_type:
            tools = [t for t in tools if t.category == task_type]
        
        return "可用工具：\n" + "\n".join(t.to_prompt() for t in tools)
```

### 2.3 工具分类体系

| 分类 | 说明 | 示例工具 |
|------|------|---------|
| **信息检索** | 获取外部信息 | search, web_fetch, database_query |
| **数据处理** | 处理和转换数据 | parse_json, filter_data, aggregate |
| **内容生成** | 生成文本/代码/图片 | generate_text, write_code, create_image |
| **系统操作** | 执行系统命令 | run_command, file_operation |
| **通信交互** | 与外部系统交互 | send_email, api_call, notification |

---

## 三、技能抽象层

### 3.1 技能定义

```python
@dataclass
class SkillDefinition:
    """技能定义"""
    name: str
    description: str
    required_tools: List[str]         # 所需工具
    input_schema: Dict[str, Any]      # 输入Schema
    output_schema: Dict[str, Any]     # 输出Schema
    steps: List[SkillStep]            # 执行步骤
    examples: List[SkillExample]      # 使用示例
    
    def can_execute(self, available_tools: List[str]) -> bool:
        """检查是否可以执行"""
        return all(tool in available_tools for tool in self.required_tools)

@dataclass
class SkillStep:
    """技能执行步骤"""
    step_id: str
    description: str
    tool_name: str
    input_mapping: Dict[str, str]     # 输入映射
    output_mapping: Dict[str, str]    # 输出映射
    condition: str = None             # 条件表达式

@dataclass
class SkillExample:
    """技能使用示例"""
    input: Dict[str, Any]
    output: Dict[str, Any]
    explanation: str
```

### 3.2 技能执行引擎

```python
class SkillExecutor:
    """技能执行引擎"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
        self.execution_history = []
    
    async def execute(self, skill: SkillDefinition, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行技能"""
        # 1. 检查工具可用性
        available_tools = [t.name for t in self.registry.tools.values()]
        if not skill.can_execute(available_tools):
            missing = set(skill.required_tools) - set(available_tools)
            raise SkillExecutionError(f"缺少工具：{missing}")
        
        # 2. 按步骤执行
        context = inputs.copy()
        for step in skill.steps:
            # 检查条件
            if step.condition and not self.evaluate_condition(step.condition, context):
                continue
            
            # 映射输入
            tool_input = self.map_input(step.input_mapping, context)
            
            # 执行工具
            tool = self.registry.tools[step.tool_name]
            result = await tool.function(**tool_input)
            
            # 映射输出
            self.map_output(step.output_mapping, result, context)
        
        # 3. 返回结果
        return context
    
    def map_input(self, mapping: Dict[str, str], context: Dict) -> Dict:
        """映射输入参数"""
        return {k: context[v] for k, v in mapping.items()}
    
    def map_output(self, mapping: Dict[str, str], result: Any, context: Dict):
        """映射输出结果"""
        if isinstance(result, dict):
            for k, v in mapping.items():
                context[k] = result.get(v)
        else:
            for k, v in mapping.items():
                context[k] = result
```

---

## 四、技能发现与选择

### 4.1 技能检索

```python
class SkillSelector:
    """技能选择器"""
    
    def __init__(self, skill_store, embedder):
        self.skill_store = skill_store
        self.embedder = embedder
    
    async def find_skills(self, task: str, top_k: int = 5) -> List[SkillDefinition]:
        """根据任务描述查找相关技能"""
        # 1. 语义搜索
        task_embedding = await self.embedder.embed(task)
        similar_skills = await self.skill_store.search(
            embedding=task_embedding,
            top_k=top_k * 2  # 多检索一些，后续筛选
        )
        
        # 2. 关键词匹配
        keywords = self.extract_keywords(task)
        keyword_matched = await self.skill_store.keyword_search(keywords)
        
        # 3. 合并去重
        all_skills = self.merge_results(similar_skills, keyword_matched)
        
        # 4. 评分排序
        scored_skills = self.score_skills(all_skills, task)
        
        return scored_skills[:top_k]
    
    def score_skills(self, skills: List[SkillDefinition], task: str) -> List[SkillDefinition]:
        """对技能评分"""
        scored = []
        for skill in skills:
            score = 0.0
            
            # 语义相似度
            score += self.semantic_similarity(skill, task) * 0.5
            
            # 历史成功率
            score += self.get_success_rate(skill.name) * 0.3
            
            # 描述质量
            score += len(skill.description) / 1000 * 0.1
            
            # 示例数量
            score += len(skill.examples) / 10 * 0.1
            
            scored.append((skill, score))
        
        return [s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
```

### 4.2 技能选择策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **直接匹配** | 精确匹配已知技能 | 重复任务 |
| **语义搜索** | 基于语义相似度 | 新任务 |
| **组合推荐** | 推荐技能组合 | 复杂任务 |
| **经验推荐** | 基于历史成功经验 | 有历史数据 |

---

## 五、技能组合与编排

### 5.1 技能组合模式

| 模式 | 说明 | 示例 |
|------|------|------|
| **顺序组合** | 技能依次执行 | 搜索→分析→生成 |
| **并行组合** | 技能同时执行 | 同时搜索多个来源 |
| **条件组合** | 根据条件选择 | 成功→A，失败→B |
| **循环组合** | 重复执行直到满足条件 | 重试直到成功 |

### 5.2 技能编排器

```python
class SkillComposer:
    """技能组合编排器"""
    
    def __init__(self, skill_executor: SkillExecutor):
        self.executor = skill_executor
    
    async def compose_sequential(self, skills: List[SkillDefinition], inputs: Dict) -> Dict:
        """顺序组合"""
        context = inputs
        for skill in skills:
            context = await self.executor.execute(skill, context)
        return context
    
    async def compose_parallel(self, skills: List[SkillDefinition], inputs: Dict) -> Dict:
        """并行组合"""
        tasks = [self.executor.execute(skill, inputs) for skill in skills]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        merged = inputs.copy()
        for result in results:
            if isinstance(result, dict):
                merged.update(result)
        return merged
    
    async def compose_conditional(self, branches: List[Tuple[str, SkillDefinition]], inputs: Dict) -> Dict:
        """条件组合"""
        for condition, skill in branches:
            if self.evaluate_condition(condition, inputs):
                return await self.executor.execute(skill, inputs)
        return inputs
```

---

## 六、技能进化系统

### 6.1 技能学习

| 学习方式 | 说明 | 数据来源 |
|---------|------|---------|
| **成功经验** | 记录成功的技能调用 | 执行历史 |
| **失败经验** | 记录失败和错误 | 错误日志 |
| **用户反馈** | 用户对结果的评价 | 反馈系统 |
| **自动发现** | 自动发现新的工具组合 | 探索机制 |

### 6.2 技能评估

```python
class SkillEvaluator:
    """技能评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_execution(self, skill_name: str, success: bool, latency: float, cost: float):
        """记录执行结果"""
        if skill_name not in self.metrics:
            self.metrics[skill_name] = {
                "total": 0,
                "success": 0,
                "total_latency": 0,
                "total_cost": 0
            }
        
        m = self.metrics[skill_name]
        m["total"] += 1
        if success:
            m["success"] += 1
        m["total_latency"] += latency
        m["total_cost"] += cost
    
    def get_skill_score(self, skill_name: str) -> Dict:
        """获取技能评分"""
        if skill_name not in self.metrics:
            return {"score": 0.5, "reason": "无历史数据"}
        
        m = self.metrics[skill_name]
        if m["total"] == 0:
            return {"score": 0.5, "reason": "无执行记录"}
        
        success_rate = m["success"] / m["total"]
        avg_latency = m["total_latency"] / m["total"]
        avg_cost = m["total_cost"] / m["total"]
        
        # 综合评分
        score = (
            success_rate * 0.5 +
            (1 - min(avg_latency / 10, 1)) * 0.3 +  # 延迟越低越好
            (1 - min(avg_cost / 1, 1)) * 0.2  # 成本越低越好
        )
        
        return {
            "score": score,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "avg_cost": avg_cost
        }
```

### 6.3 技能进化机制

```
执行技能 → 记录结果 → 评估效果 → 发现改进点 → 优化技能 → 验证效果
    │          │          │          │          │          │
  调用工具   日志存储   评分计算   分析模式   参数调整   A/B测试
```

---

## 七、最佳实践

### 7.1 工具设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 一个工具只做一件事 | 搜索工具不做数据处理 |
| **幂等性** | 重复调用结果相同 | 查询操作天然幂等 |
| **无状态** | 不依赖外部状态 | 每次调用独立 |
| **容错性** | 失败时返回清晰错误 | 包含错误码和描述 |

### 7.2 技能管理策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **版本控制** | 技能有版本号 | 生产环境 |
| **灰度发布** | 新技能逐步放量 | 技能更新 |
| **A/B测试** | 对比新旧技能效果 | 效果验证 |
| **回滚机制** | 问题时快速回滚 | 故障恢复 |

### 7.3 性能优化

| 优化方向 | 具体措施 |
|---------|---------|
| **工具缓存** | 缓存工具描述和结果 |
| **异步执行** | 并行调用独立工具 |
| **批量处理** | 合并多个工具调用 |
| **预热** | 提前加载常用工具 |

---

## 总结

Agent技能系统的核心要点：

1. **分层设计**：工具→技能→复合技能的层次结构
2. **动态发现**：运行时检索和选择技能
3. **智能组合**：多个技能编排完成复杂任务
4. **持续进化**：从经验中学习和优化技能
5. **可观测性**：完整记录技能执行过程

> 技能系统的本质是**将Agent的"工具调用能力"升级为"解决问题的能力"**。
