---
title: "Agent技能组合模式：从原子能力到复杂工作流的编排艺术"
description: "深入解析Agent技能组合的核心模式，覆盖串行链式、并行扇出、条件路由、动态嵌套、递归组合等编排策略，包含完整实现与生产级优化"
date: 2026-05-31
author: "RiceBall-15"
category: "AI智能体"
subCategory: agent-skill
tags: ["Agent", "Skill", "工作流编排", "组合模式", "架构设计"]
draft: false
---

# Agent技能组合模式：从原子能力到复杂工作流的编排艺术

## 一、概念原理：为什么技能组合是Agent智能的核心

### 1.1 从工具调用到技能组合的范式跃迁

传统AI Agent的工具调用是**单步原子操作**——给定一个工具名和参数，执行一次，返回结果。但在真实业务场景中，绝大多数任务需要**多个工具协同完成**。一个简单的"帮我订机票"任务，背后可能涉及：查询航班 → 比较价格 → 检查座位 → 锁定舱位 → 填写乘客信息 → 支付。每个步骤对应一个原子技能，而把这些技能串联成完整流程的能力，就是**技能组合**。

技能组合的本质是**将Agent的推理能力从单步决策扩展到多步规划**。它解决了三个核心问题：

| 问题 | 单步工具调用 | 技能组合 |
|------|-------------|---------|
| **任务复杂度** | 只能处理一步完成的任务 | 可处理多步骤、多条件的复杂任务 |
| **错误恢复** | 失败即终止 | 支持重试、回退、替代路径 |
| **资源利用** | 串行执行 | 并行执行、流水线优化 |

### 1.2 技能组合的数学抽象

从形式化角度看，技能组合可以抽象为一个**有向无环图（DAG）**：

```
技能组合 = G(V, E)
其中：
  V = {skill₁, skill₂, ..., skillₙ}  // 技能节点集合
  E = {(skillᵢ, skillⱼ, condition)}  // 有向边，可带条件
  
每个技能: skillᵢ = (input_schema, execute_fn, output_schema, metadata)
```

这个抽象揭示了技能组合的三个关键维度：

1. **拓扑结构**：节点之间的连接方式（串行、并行、条件分支）
2. **数据流**：技能之间传递的数据格式和转换规则
3. **控制流**：执行顺序、条件判断、循环和错误处理

### 1.3 与其他编排范式的对比

技能组合不是孤立的概念，它与多种编排范式有交集和区别：

```
┌─────────────────────────────────────────────────────────────┐
│                    编排范式对比                               │
├──────────────┬──────────────┬──────────────┬───────────────┤
│              │ 技能组合      │ 工作流引擎    │ 函数编排       │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ 执行主体      │ LLM推理驱动   │ 预定义流程    │ 代码调用       │
│ 灵活性       │ 高（动态决策） │ 中（模板化）   │ 低（硬编码）    │
│ 可预测性     │ 中           │ 高           │ 高            │
│ 适用场景     │ 非结构化任务   │ 结构化流程    │ 程序化逻辑     │
│ 错误处理     │ 自适应        │ 预定义策略    │ 异常捕获       │
│ 典型实现     │ LangGraph     │ Airflow      │ Prefect       │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

## 二、架构设计：技能组合系统的整体架构

### 2.1 分层架构

一个完整的技能组合系统采用四层架构：

```
┌────────────────────────────────────────────────────────────────┐
│                     用户请求层 (Request Layer)                   │
│         自然语言输入 → 意图识别 → 任务分解                       │
├────────────────────────────────────────────────────────────────┤
│                   组合编排层 (Composition Layer)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ 串行链式  │ │ 并行扇出  │ │ 条件路由  │ │ 动态嵌套/递归    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
│          ↓ DAG 构建 → 拓扑排序 → 执行计划生成                    │
├────────────────────────────────────────────────────────────────┤
│                   技能执行层 (Execution Layer)                   │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐  │
│  │ Skill  │ │ Skill  │ │ Skill  │ │ Skill  │ │ Skill Pool │  │
│  │   A    │ │   B    │ │   C    │ │   D    │ │ (Registry) │  │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────────┘  │
├────────────────────────────────────────────────────────────────┤
│                   基础设施层 (Infrastructure Layer)              │
│    状态管理 │ 并发控制 │ 超时熔断 │ 日志追踪 │ 结果缓存        │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据结构

技能组合系统的核心数据结构定义了整个系统的表达能力：

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import uuid

class SkillType(Enum):
    """技能类型"""
    ATOMIC = "atomic"          # 原子技能：单步执行
    COMPOSITE = "composite"    # 组合技能：包含子技能
    CONDITIONAL = "conditional"  # 条件技能：根据条件选择子技能
    LOOP = "loop"              # 循环技能：重复执行直到满足条件

class CompositionPattern(Enum):
    """组合模式"""
    SEQUENTIAL = "sequential"    # 串行链式
    PARALLEL = "parallel"        # 并行扇出
    PIPELINE = "pipeline"        # 流水线
    ROUTER = "router"            # 条件路由
    MAP_REDUCE = "map_reduce"    # 映射归约
    DYNAMIC = "dynamic"          # 动态嵌套

@dataclass
class SkillDefinition:
    """技能定义"""
    id: str
    name: str
    description: str
    skill_type: SkillType
    input_schema: dict           # JSON Schema
    output_schema: dict          # JSON Schema
    execute: Callable            # 执行函数
    timeout: float = 30.0        # 超时时间（秒）
    retry_policy: dict = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2.0,
        "retry_on": ["timeout", "temporary_error"]
    })
    metadata: dict = field(default_factory=dict)

@dataclass
class CompositionNode:
    """组合节点：DAG中的一个节点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    skill: Optional[SkillDefinition] = None
    pattern: Optional[CompositionPattern] = None
    children: list = field(default_factory=list)
    condition: Optional[Callable] = None  # 条件函数
    input_mapping: dict = field(default_factory=dict)   # 输入映射
    output_mapping: dict = field(default_factory=dict)  # 输出映射
    metadata: dict = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    """执行计划：DAG拓扑排序后的线性序列"""
    steps: list                  # 执行步骤列表
    parallel_groups: list        # 可并行执行的步骤组
    total_estimated_time: float  # 预估总执行时间
    resource_requirements: dict  # 资源需求
```

### 2.3 组合模式详解

#### 模式一：串行链式（Sequential Chain）

最基础的组合模式，前一个技能的输出是后一个技能的输入：

```
skill_A → skill_B → skill_C → result

数据流: A.output = B.input, B.output = C.input
```

适用场景：任务有明确的先后依赖关系，如 ETL 流程、审批链。

```python
class SequentialComposer:
    """串行链式组合器"""
    
    async def compose(self, skills: list[SkillDefinition], 
                      initial_input: dict) -> dict:
        """按顺序执行技能链"""
        current_input = initial_input
        execution_trace = []
        
        for i, skill in enumerate(skills):
            step_start = time.time()
            
            # 输入映射：将上一步输出映射为当前输入
            mapped_input = self._map_input(
                current_input, 
                skill.input_schema
            )
            
            # 执行技能
            try:
                result = await asyncio.wait_for(
                    skill.execute(mapped_input),
                    timeout=skill.timeout
                )
            except asyncio.TimeoutError:
                result = await self._handle_timeout(skill, mapped_input)
            except Exception as e:
                result = await self._handle_error(skill, e, mapped_input)
            
            # 记录执行轨迹
            execution_trace.append({
                "step": i,
                "skill_id": skill.id,
                "input": mapped_input,
                "output": result,
                "duration": time.time() - step_start,
                "status": "success" if result.get("success") else "error"
            })
            
            # 输出映射：将当前输出传递给下一步
            current_input = self._map_output(result, skill.output_schema)
        
        return {
            "result": current_input,
            "trace": execution_trace,
            "total_steps": len(skills)
        }
    
    def _map_input(self, context: dict, schema: dict) -> dict:
        """根据 schema 从上下文中提取所需输入"""
        mapped = {}
        for key, spec in schema.get("properties", {}).items():
            source = spec.get("source", key)
            if source in context:
                mapped[key] = context[source]
            elif spec.get("required", False):
                raise ValueError(f"Missing required input: {key}")
        return mapped
    
    def _map_output(self, result: dict, schema: dict) -> dict:
        """将技能输出映射到统一上下文"""
        output = {}
        for key in schema.get("properties", {}):
            if key in result:
                output[key] = result[key]
        return output
```

#### 模式二：并行扇出（Parallel Fan-out）

多个技能同时执行，最终汇聚结果：

```
        ┌→ skill_B₁ →┐
skill_A → skill_B₂ → ──→ skill_C
        └→ skill_B₃ →┘

执行: B₁, B₂, B₃ 并行，C 等待所有完成
```

适用场景：多源数据聚合、并行搜索、批量处理。

```python
class ParallelFanoutComposer:
    """并行扇出组合器"""
    
    async def compose(self, fan_out_skills: list[SkillDefinition],
                      reducer_skill: SkillDefinition,
                      input_data: dict,
                      max_concurrency: int = 5) -> dict:
        """扇出-归约模式"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_execute(skill):
            async with semaphore:
                return await self._execute_with_retry(skill, input_data)
        
        # 并行执行所有扇出技能
        tasks = [bounded_execute(s) for s in fan_out_skills]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分离成功和失败的结果
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        if failed:
            logging.warning(f"Parallel fan-out: {len(failed)}/{len(results)} failed")
        
        # 归约：将多个结果合并
        reduced_input = {
            "results": successful,
            "failures": [{"error": str(f)} for f in failed],
            "original_input": input_data
        }
        
        return await reducer_skill.execute(reduced_input)
```

#### 模式三：条件路由（Conditional Router）

根据运行时条件动态选择执行路径：

```
                    ┌→ skill_B (高优先级路径)
skill_A → condition┤
                    └→ skill_C (普通路径)

条件: 由 LLM 或规则引擎决定
```

适用场景：智能客服（根据问题类型路由）、异常处理（降级策略）。

```python
class ConditionalRouterComposer:
    """条件路由组合器"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def compose(self, router_skill: SkillDefinition,
                      routes: dict[str, SkillDefinition],
                      fallback: Optional[SkillDefinition],
                      input_data: dict) -> dict:
        """条件路由执行"""
        
        # 方式1: 规则路由（优先）
        route_key = self._rule_based_route(input_data, routes.keys())
        
        if route_key is None:
            # 方式2: LLM路由（兜底）
            route_key = await self._llm_route(router_skill, input_data, routes)
        
        # 执行选定路径
        if route_key and route_key in routes:
            selected_skill = routes[route_key]
        elif fallback:
            selected_skill = fallback
            route_key = "fallback"
        else:
            raise ValueError(f"No route found and no fallback defined")
        
        result = await selected_skill.execute(input_data)
        result["_routed_to"] = route_key
        return result
    
    def _rule_based_route(self, input_data: dict, available_routes: set) -> Optional[str]:
        """基于规则的路由决策"""
        # 优先级规则：精确匹配 > 模糊匹配 > 默认
        text = input_data.get("text", "").lower()
        
        # 关键词路由表
        route_keywords = {
            "billing": ["付款", "账单", "发票", "退款", "billing"],
            "technical": ["报错", "异常", "bug", "crash", "technical"],
            "general": ["你好", "请问", "怎么", "help"],
        }
        
        for route, keywords in route_keywords.items():
            if route in available_routes:
                if any(kw in text for kw in keywords):
                    return route
        
        return None
    
    async def _llm_route(self, router_skill, input_data, routes):
        """基于LLM的智能路由"""
        route_descriptions = "\n".join([
            f"- {key}: {skill.description}"
            for key, skill in routes.items()
        ])
        
        prompt = f"""根据用户输入，选择最合适的处理路径。

可用路径:
{route_descriptions}

用户输入: {input_data.get('text', '')}

请只返回路径名称（如 {list(routes.keys())[0]}），不要其他内容。"""
        
        response = await self.llm.generate(prompt)
        route = response.strip().lower()
        
        return route if route in routes else None
```

#### 模式四：动态嵌套（Dynamic Nesting）

技能可以动态包含其他技能，形成树状结构：

```
skill_Composite
├── skill_A (原子)
├── skill_Composite₂ (嵌套组合)
│   ├── skill_B (原子)
│   └── skill_C (原子)
└── skill_D (原子)
```

适用场景：复杂业务流程中，某些子流程本身也是组合技能。

```python
class DynamicNestingComposer:
    """动态嵌套组合器：支持技能递归组合"""
    
    async def compose(self, composite_skill: SkillDefinition,
                      context: dict,
                      max_depth: int = 5) -> dict:
        """递归执行组合技能"""
        return await self._execute_node(composite_skill, context, depth=0, max_depth=max_depth)
    
    async def _execute_node(self, skill: SkillDefinition,
                            context: dict, depth: int, max_depth: int) -> dict:
        """递归执行节点"""
        if depth > max_depth:
            raise RecursionError(f"Maximum nesting depth {max_depth} exceeded")
        
        if skill.skill_type == SkillType.ATOMIC:
            # 原子技能：直接执行
            return await skill.execute(context)
        
        elif skill.skill_type == SkillType.COMPOSITE:
            # 组合技能：递归执行子技能
            sub_results = []
            for sub_skill in skill.metadata.get("children", []):
                sub_result = await self._execute_node(
                    sub_skill, context, depth + 1, max_depth
                )
                sub_results.append(sub_result)
                # 将子技能结果合并到上下文
                context.update(sub_result)
            
            return {"sub_results": sub_results, "depth": depth}
        
        elif skill.skill_type == SkillType.CONDITIONAL:
            # 条件技能：评估条件后选择子技能
            condition_fn = skill.metadata.get("condition")
            if condition_fn and condition_fn(context):
                true_branch = skill.metadata.get("true_branch")
                return await self._execute_node(true_branch, context, depth + 1, max_depth)
            else:
                false_branch = skill.metadata.get("false_branch")
                return await self._execute_node(false_branch, context, depth + 1, max_depth)
        
        elif skill.skill_type == SkillType.LOOP:
            # 循环技能：重复执行直到条件满足
            loop_body = skill.metadata.get("body")
            exit_condition = skill.metadata.get("exit_condition", lambda ctx: True)
            max_iterations = skill.metadata.get("max_iterations", 10)
            
            iterations = 0
            while not exit_condition(context) and iterations < max_iterations:
                result = await self._execute_node(loop_body, context, depth + 1, max_depth)
                context.update(result)
                iterations += 1
            
            return {"iterations": iterations, "final_context": context}
```

## 三、实战实现：构建完整的技能组合引擎

### 3.1 技能注册与发现

```python
import json
from pathlib import Path

class SkillRegistry:
    """技能注册中心：管理所有可用技能"""
    
    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}
        self._categories: dict[str, list[str]] = {}
        self._load_builtin_skills()
    
    def register(self, skill: SkillDefinition):
        """注册技能"""
        self._skills[skill.id] = skill
        category = skill.metadata.get("category", "general")
        self._categories.setdefault(category, []).append(skill.id)
        logging.info(f"Registered skill: {skill.id} ({skill.name})")
    
    def discover(self, query: str = None, 
                 category: str = None,
                 input_schema: dict = None) -> list[SkillDefinition]:
        """发现匹配的技能"""
        candidates = list(self._skills.values())
        
        # 按类别过滤
        if category:
            skill_ids = self._categories.get(category, [])
            candidates = [s for s in candidates if s.id in skill_ids]
        
        # 按查询关键词过滤
        if query:
            query_lower = query.lower()
            candidates = [
                s for s in candidates
                if query_lower in s.name.lower() 
                or query_lower in s.description.lower()
            ]
        
        # 按输入schema兼容性过滤
        if input_schema:
            compatible = []
            for skill in candidates:
                if self._is_input_compatible(skill.input_schema, input_schema):
                    compatible.append(skill)
            candidates = compatible
        
        return candidates
    
    def _is_input_compatible(self, skill_schema: dict, 
                              available_input: dict) -> bool:
        """检查技能输入与可用数据是否兼容"""
        required = skill_schema.get("required", [])
        properties = skill_schema.get("properties", {})
        
        for req_field in required:
            if req_field not in available_input:
                # 检查是否有默认值
                if req_field in properties and "default" in properties[req_field]:
                    continue
                return False
        return True
    
    def _load_builtin_skills(self):
        """加载内置技能"""
        # 文件操作技能
        self.register(SkillDefinition(
            id="file_read",
            name="文件读取",
            description="读取本地文件内容",
            skill_type=SkillType.ATOMIC,
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "size": {"type": "integer"}
                }
            },
            execute=self._execute_file_read,
            timeout=10.0
        ))
        
        # HTTP请求技能
        self.register(SkillDefinition(
            id="http_request",
            name="HTTP请求",
            description="发送HTTP请求并返回响应",
            skill_type=SkillType.ATOMIC,
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "default": "GET"},
                    "headers": {"type": "object"},
                    "body": {"type": "object"}
                },
                "required": ["url"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status_code": {"type": "integer"},
                    "body": {"type": "object"}
                }
            },
            execute=self._execute_http_request,
            timeout=30.0
        ))
        
        # LLM推理技能
        self.register(SkillDefinition(
            id="llm_reason",
            name="LLM推理",
            description="使用LLM进行推理和文本生成",
            skill_type=SkillType.ATOMIC,
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "context": {"type": "object"},
                    "max_tokens": {"type": "integer", "default": 2048}
                },
                "required": ["prompt"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tokens_used": {"type": "integer"}
                }
            },
            execute=self._execute_llm_reason,
            timeout=60.0
        ))
```

### 3.2 DAG编排器

```python
import networkx as nx
from collections import defaultdict

class DAGComposer:
    """基于DAG的技能组合编排器"""
    
    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.graph = nx.DiGraph()
        self._execution_results = {}
    
    def build_from_plan(self, plan: dict) -> None:
        """从JSON计划构建DAG"""
        self.graph.clear()
        
        for node in plan["nodes"]:
            self.graph.add_node(
                node["id"],
                skill_id=node.get("skill_id"),
                pattern=node.get("pattern"),
                condition=node.get("condition"),
                input_mapping=node.get("input_mapping", {}),
                output_mapping=node.get("output_mapping", {})
            )
        
        for edge in plan["edges"]:
            self.graph.add_edge(
                edge["from"],
                edge["to"],
                data_key=edge.get("data_key"),
                condition=edge.get("condition")
            )
        
        # 验证DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph contains cycles - not a valid DAG")
        
        logging.info(f"DAG built: {self.graph.number_of_nodes()} nodes, "
                     f"{self.graph.number_of_edges()} edges")
    
    def execute(self, initial_input: dict) -> dict:
        """执行DAG"""
        # 拓扑排序
        topo_order = list(nx.topological_sort(self.graph))
        
        # 识别可并行执行的层级
        parallel_levels = self._get_parallel_levels(topo_order)
        
        execution_log = []
        context = dict(initial_input)
        
        for level in parallel_levels:
            level_tasks = []
            for node_id in level:
                node_data = self.graph.nodes[node_id]
                task = self._execute_node(node_id, node_data, context)
                level_tasks.append((node_id, task))
            
            # 同一层级的节点并行执行
            if len(level_tasks) == 1:
                node_id, task = level_tasks[0]
                result = asyncio.run(task) if asyncio.iscoroutine(task) else task
                self._update_context(context, node_id, result)
                execution_log.append({"node": node_id, "result": result})
            else:
                # 并行执行
                results = asyncio.gather(*[t for _, t in level_tasks])
                for (node_id, _), result in zip(level_tasks, results):
                    self._update_context(context, node_id, result)
                    execution_log.append({"node": node_id, "result": result})
        
        return {
            "output": context.get("final_output", context),
            "execution_log": execution_log,
            "dag_stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "parallel_levels": len(parallel_levels),
                "max_parallelism": max(len(level) for level in parallel_levels)
            }
        }
    
    def _get_parallel_levels(self, topo_order: list) -> list[list[str]]:
        """将拓扑排序转换为并行层级"""
        levels = []
        assigned = set()
        
        for node in topo_order:
            # 找到该节点所有前驱
            predecessors = set(self.graph.predecessors(node))
            
            # 如果所有前驱都已分配，该节点可以执行
            if predecessors.issubset(assigned):
                # 查找或创建新层级
                placed = False
                for level in levels:
                    # 检查该层级中是否有该节点的前驱
                    level_nodes = set(level)
                    if not level_nodes.intersection(predecessors):
                        level.append(node)
                        placed = True
                        break
                
                if not placed:
                    levels.append([node])
                
                assigned.add(node)
        
        return levels
    
    def _update_context(self, context: dict, node_id: str, result: dict):
        """将节点执行结果更新到上下文"""
        node_data = self.graph.nodes[node_id]
        output_mapping = node_data.get("output_mapping", {})
        
        if output_mapping:
            for source_key, target_key in output_mapping.items():
                if source_key in result:
                    context[target_key] = result[source_key]
        else:
            context.update(result)
        
        self._execution_results[node_id] = result
```

### 3.3 LLM驱动的动态组合

```python
class LLMCompositionPlanner:
    """LLM驱动的动态组合规划器"""
    
    def __init__(self, llm_client, registry: SkillRegistry):
        self.llm = llm_client
        self.registry = registry
    
    async def plan(self, user_request: str, context: dict = None) -> dict:
        """根据用户请求动态生成执行计划"""
        
        # 1. 发现可用技能
        available_skills = self.registry.discover()
        skill_descriptions = self._format_skill_descriptions(available_skills)
        
        # 2. 构建规划prompt
        prompt = f"""你是一个AI Agent的技能编排专家。根据用户请求，设计一个技能执行计划。

## 可用技能
{skill_descriptions}

## 用户请求
{user_request}

## 当前上下文
{json.dumps(context or {}, ensure_ascii=False, indent=2)}

## 输出要求
请输出一个JSON格式的执行计划，包含以下字段：
- "thought": 你的分析过程
- "nodes": 技能节点列表，每个节点包含 id, skill_id, input_mapping, output_mapping
- "edges": 节点间的依赖关系，每个边包含 from, to, data_key
- "initial_input": 初始输入数据

要求：
1. 优先使用串行模式，仅在无依赖关系时使用并行
2. 每个技能的输入必须能从上下文或其他技能输出中获取
3. 如果请求超出已有技能能力，明确说明无法完成的部分
"""
        
        # 3. 生成计划
        response = await self.llm.generate(prompt, max_tokens=4096)
        
        # 4. 解析和验证计划
        try:
            plan = json.loads(response)
            self._validate_plan(plan, available_skills)
            return plan
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试提取JSON
            plan = self._extract_json(response)
            self._validate_plan(plan, available_skills)
            return plan
    
    def _format_skill_descriptions(self, skills: list) -> str:
        """格式化技能描述"""
        lines = []
        for skill in skills:
            lines.append(f"### {skill.id}: {skill.name}")
            lines.append(f"描述: {skill.description}")
            lines.append(f"输入: {json.dumps(skill.input_schema, ensure_ascii=False)}")
            lines.append(f"输出: {json.dumps(skill.output_schema, ensure_ascii=False)}")
            lines.append("")
        return "\n".join(lines)
    
    def _validate_plan(self, plan: dict, available_skills: list):
        """验证计划的合法性"""
        skill_ids = {s.id for s in available_skills}
        
        for node in plan.get("nodes", []):
            if node.get("skill_id") not in skill_ids:
                raise ValueError(f"Unknown skill: {node.get('skill_id')}")
        
        # 检查循环依赖
        edges = plan.get("edges", [])
        graph = nx.DiGraph()
        for edge in edges:
            graph.add_edge(edge["from"], edge["to"])
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Plan contains circular dependencies")
    
    def _extract_json(self, text: str) -> dict:
        """从文本中提取JSON"""
        # 尝试提取 ```json ... ``` 块
        import re
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # 尝试找到最外层的 { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        raise ValueError("Could not extract JSON from response")
```

## 四、生产优化：从实验室到生产环境

### 4.1 执行性能优化

#### 缓存策略

技能组合系统中，缓存是提升性能的关键手段。不同层级需要不同的缓存策略：

```python
class SkillCache:
    """多层级技能缓存"""
    
    def __init__(self, redis_client=None, local_ttl=300):
        self.redis = redis_client
        self.local_cache = {}
        self.local_ttl = local_ttl
    
    async def get_or_execute(self, skill: SkillDefinition, 
                              input_data: dict) -> dict:
        """缓存优先执行"""
        cache_key = self._make_key(skill, input_data)
        
        # L1: 本地内存缓存（最快，容量小）
        if cache_key in self.local_cache:
            entry = self.local_cache[cache_key]
            if time.time() - entry["time"] < self.local_ttl:
                return entry["data"]
        
        # L2: Redis缓存（较快，容量大）
        if self.redis:
            cached = await self.redis.get(f"skill_cache:{cache_key}")
            if cached:
                data = json.loads(cached)
                self.local_cache[cache_key] = {"data": data, "time": time.time()}
                return data
        
        # L3: 执行技能
        result = await skill.execute(input_data)
        
        # 写入缓存
        self.local_cache[cache_key] = {"data": result, "time": time.time()}
        if self.redis:
            await self.redis.setex(
                f"skill_cache:{cache_key}",
                self.local_ttl * 2,
                json.dumps(result)
            )
        
        return result
    
    def _make_key(self, skill: SkillDefinition, input_data: dict) -> str:
        """生成缓存键"""
        content = f"{skill.id}:{json.dumps(input_data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
```

#### 并发控制与资源管理

```python
class ResourceAwareExecutor:
    """资源感知执行器"""
    
    def __init__(self, max_concurrent=10, max_memory_mb=512):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_memory = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self._lock = asyncio.Lock()
    
    async def execute_with_resources(self, skill: SkillDefinition,
                                      input_data: dict) -> dict:
        """带资源控制的执行"""
        estimated_memory = skill.metadata.get("estimated_memory", 10 * 1024 * 1024)
        
        async with self.semaphore:
            # 检查内存
            async with self._lock:
                if self.current_memory + estimated_memory > self.max_memory:
                    await self._wait_for_memory(estimated_memory)
                self.current_memory += estimated_memory
            
            try:
                result = await skill.execute(input_data)
                return result
            finally:
                async with self._lock:
                    self.current_memory -= estimated_memory
    
    async def _wait_for_memory(self, needed: int):
        """等待足够内存"""
        while self.current_memory + needed > self.max_memory:
            await asyncio.sleep(0.1)
```

### 4.2 容错与恢复

```python
class FaultTolerantComposer:
    """容错组合器：支持多种恢复策略"""
    
    RECOVERY_STRATEGIES = {
        "retry": "重试",
        "fallback": "降级",
        "skip": "跳过",
        "compensate": "补偿",
        "circuit_break": "熔断"
    }
    
    async def execute_with_recovery(self, skill: SkillDefinition,
                                     input_data: dict,
                                     strategy: str = "retry",
                                     fallback_skill: SkillDefinition = None) -> dict:
        """带恢复策略的执行"""
        
        if strategy == "retry":
            return await self._execute_with_retry(skill, input_data)
        
        elif strategy == "fallback":
            try:
                return await skill.execute(input_data)
            except Exception as e:
                logging.warning(f"Primary skill failed: {e}, using fallback")
                if fallback_skill:
                    return await fallback_skill.execute(input_data)
                raise
        
        elif strategy == "skip":
            try:
                return await skill.execute(input_data)
            except Exception as e:
                logging.warning(f"Skill skipped due to error: {e}")
                return {"skipped": True, "error": str(e)}
        
        elif strategy == "compensate":
            result = await skill.execute(input_data)
            # 记录操作用于补偿
            self._record_for_compensation(skill, input_data, result)
            return result
        
        elif strategy == "circuit_break":
            return await self._execute_with_circuit_breaker(skill, input_data)
    
    async def _execute_with_retry(self, skill: SkillDefinition,
                                   input_data: dict) -> dict:
        """带指数退避的重试"""
        max_retries = skill.retry_policy["max_retries"]
        backoff = skill.retry_policy["backoff_factor"]
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await skill.execute(input_data)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = backoff ** attempt
                    logging.warning(
                        f"Skill {skill.id} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
        
        raise last_error
    
    async def _execute_with_circuit_breaker(self, skill: SkillDefinition,
                                             input_data: dict) -> dict:
        """熔断器模式"""
        # 简化的熔断器实现
        state = self._get_circuit_state(skill.id)
        
        if state == "open":
            # 熔断中，快速失败
            raise CircuitBreakerOpenError(f"Circuit breaker open for {skill.id}")
        
        try:
            result = await skill.execute(input_data)
            self._record_success(skill.id)
            return result
        except Exception as e:
            self._record_failure(skill.id)
            raise
    
    def _record_for_compensation(self, skill, input_data, result):
        """记录操作用于补偿"""
        compensation_log.append({
            "skill_id": skill.id,
            "input": input_data,
            "output": result,
            "timestamp": time.time(),
            "compensation_fn": skill.metadata.get("compensation")
        })
```

### 4.3 可观测性

```python
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SpanContext:
    """分布式追踪Span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    skill_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

class CompositionTracer:
    """技能组合追踪器"""
    
    def __init__(self):
        self.spans: list[SpanContext] = []
        self.metrics = defaultdict(int)
    
    def start_span(self, skill_id: str, parent_id: str = None) -> SpanContext:
        """开始一个新的追踪Span"""
        span = SpanContext(
            trace_id=self._get_trace_id(),
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=parent_id,
            skill_id=skill_id,
            start_time=time.time()
        )
        self.spans.append(span)
        self.metrics[f"skill.{skill_id}.invocations"] += 1
        return span
    
    def end_span(self, span: SpanContext, status: str = "ok", 
                 error: str = None):
        """结束追踪Span"""
        span.end_time = time.time()
        span.status = status
        if error:
            span.events.append({"name": "error", "message": error})
            self.metrics[f"skill.{span.skill_id}.errors"] += 1
        
        duration = span.end_time - span.start_time
        self.metrics[f"skill.{span.skill_id}.duration"] = duration
    
    def get_trace_tree(self) -> dict:
        """生成追踪树"""
        root_spans = [s for s in self.spans if s.parent_span_id is None]
        
        def build_tree(span):
            children = [s for s in self.spans if s.parent_span_id == span.span_id]
            return {
                "skill": span.skill_id,
                "status": span.status,
                "duration_ms": (span.end_time - span.start_time) * 1000 if span.end_time else None,
                "children": [build_tree(c) for c in children]
            }
        
        return {
            "trace_id": root_spans[0].trace_id if root_spans else None,
            "spans": [build_tree(r) for r in root_spans],
            "total_spans": len(self.spans),
            "metrics": dict(self.metrics)
        }
    
    def _get_trace_id(self) -> str:
        if self.spans:
            return self.spans[0].trace_id
        return str(uuid.uuid4())[:16]
```

## 五、面试深度：核心考点与设计决策

### 5.1 高频面试题

#### Q1: 如何设计一个支持动态组合的技能系统？

**考察点**：系统设计能力、对组合模式的理解

**回答要点**：

1. **核心抽象**：每个技能定义输入/输出Schema，组合系统通过Schema兼容性自动连接
2. **组合模式**：至少支持串行、并行、条件路由三种基础模式
3. **动态性**：LLM驱动的组合规划，运行时根据任务动态构建DAG
4. **容错**：每个组合节点支持独立的重试/降级策略
5. **可观测性**：完整的执行轨迹和性能指标

**加分项**：提到DAG拓扑排序、并行层级识别、资源感知调度。

#### Q2: 技能组合中的循环依赖如何检测和处理？

**考察点**：图论基础、工程实践

**回答要点**：

1. **检测**：构建有向图后用拓扑排序检测环（DFS或Kahn算法）
2. **预防**：
   - 组合计划生成时强制DAG约束
   - 注册时检查技能间的依赖关系
3. **处理**：
   - 运行时发现循环 → 抛出异常并记录
   - 限制最大递归深度
   - 对于合法的循环（如重试循环）用显式循环节点而非隐式依赖

#### Q3: 如何在技能组合中实现事务一致性？

**考察点**：分布式系统、补偿事务

**回答要点**：

1. **Saga模式**：每个技能执行后记录补偿操作
2. **执行流程**：
   ```
   正向执行: skill_A → skill_B → skill_C (失败)
   补偿执行: compensate_B → compensate_A
   ```
3. **关键设计**：
   - 补偿函数必须是幂等的
   - 补偿操作本身也可能失败，需要重试机制
   - 补偿日志持久化，支持崩溃恢复

#### Q4: 如何评估和优化技能组合的性能？

**考察点**：性能工程、系统优化

**回答要点**：

1. **度量指标**：
   - 端到端延迟（P50/P99）
   - 技能执行时间占比（找出瓶颈）
   - 并行度利用率
   - 缓存命中率

2. **优化策略**：
   - **拓扑优化**：调整执行顺序，最大化并行度
   - **缓存**：对确定性技能结果缓存
   - **批处理**：将多个小调用合并为批量调用
   - **预热**：预测性预加载常用技能

### 5.2 架构选型决策

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| ETL数据管道 | 串行链式 | 步骤间有严格顺序依赖 |
| 多源搜索聚合 | 并行扇出+归约 | 各源独立查询，结果合并 |
| 智能客服 | 条件路由 | 不同问题需要不同处理流程 |
| 复杂业务流程 | 动态嵌套 | 子流程本身也是组合 |
| 递归数据处理 | 递归组合 | 数据结构本身是树形 |
| 实时流处理 | 流水线 | 数据持续到达，需要低延迟 |

### 5.3 开放性设计问题

**问题**：设计一个支持百万级技能的技能市场，用户可以搜索、安装、组合技能完成复杂任务。

**思考方向**：

1. **技能发现**：向量化索引 + 语义搜索，支持自然语言描述查找技能
2. **技能版本**：语义化版本控制，兼容性矩阵，灰度发布
3. **组合安全**：技能沙箱隔离，权限声明与审核，资源配额
4. **组合验证**：Schema兼容性自动检查，组合计划静态分析
5. **运行时**：分布式执行引擎，支持跨节点技能调用

---

## 总结

技能组合是Agent从"能调用工具"进化到"能解决复杂问题"的关键能力。本文从概念原理出发，系统介绍了四种核心组合模式（串行链式、并行扇出、条件路由、动态嵌套），并给出了完整的实现方案。

核心要点：
- **DAG是基础**：所有组合模式都可以抽象为有向无环图
- **Schema是粘合剂**：技能间的连接通过输入/输出Schema的兼容性自动完成
- **容错是刚需**：生产环境中必须考虑重试、降级、补偿等恢复策略
- **可观测性是保障**：完整的执行追踪和性能指标是调试和优化的基础

掌握技能组合模式，不仅能提升Agent解决复杂任务的能力，也是理解LangGraph、CrewAI等主流Agent框架设计思想的关键。
