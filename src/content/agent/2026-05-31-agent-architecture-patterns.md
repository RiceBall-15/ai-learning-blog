---
title: "Agent架构设计模式：从ReAct到多Agent协作的系统化设计"
description: "系统梳理Agent架构的核心设计模式，覆盖单Agent推理、工具调用、记忆系统、多Agent协作等关键架构"
date: 2026-05-31
author: "RiceBall-15"
category: "agent"
subCategory: "agent-architecture"
tags: ["Agent架构", "设计模式", "ReAct", "多Agent"]
draft: false
---

# Agent架构设计模式：从ReAct到多Agent协作的系统化设计

## 核心问题：Agent架构怎么设计才对？

很多人开发Agent的流程是：调通API → 堆功能 → 出问题 → 修修补补。结果Agent能跑，但代码混乱、难以扩展、生产环境各种问题。

正确的做法是：**先理解架构模式，再根据场景选择合适的模式组合**。

---

## 一、Agent架构演进

### 1.1 从简单到复杂

```
Prompt链 → ReAct → 工具增强 → 记忆系统 → 多Agent → 自主Agent
(单步)    (推理)   (能力扩展)  (状态管理)  (协作)    (进化)
```

### 1.2 架构选型矩阵

| 架构模式 | 复杂度 | 能力上限 | 开发成本 | 适用场景 |
|---------|--------|---------|---------|---------|
| **Prompt链** | 低 | 低 | 低 | 简单流程 |
| **ReAct** | 中 | 中 | 中 | 单轮推理 |
| **工具增强Agent** | 中高 | 高 | 中高 | 复杂任务 |
| **记忆增强Agent** | 高 | 很高 | 高 | 多轮对话 |
| **多Agent系统** | 很高 | 极高 | 很高 | 复杂协作 |
| **自主Agent** | 极高 | 极高 | 极高 | 开放世界 |

---

## 二、核心架构模式

### 2.1 ReAct模式

**原理**：推理（Reason）+ 行动（Act）交替进行

```
用户输入 → Thought → Action → Observation → Thought → Action → ... → 最终答案
              │         │         │
              │         │         └── 工具返回结果
              │         └── 调用工具
              └── LLM分析下一步
```

**实现**：

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
    
    async def run(self, query: str, max_steps: int = 10) -> str:
        history = [{"role": "user", "content": query}]
        
        for step in range(max_steps):
            # 1. Thought + Action
            response = await self.llm.chat(history)
            
            # 2. 解析是否需要工具调用
            if "Action:" in response:
                action, input = self.parse_action(response)
                
                # 3. 执行工具
                if action in self.tools:
                    observation = await self.tools[action].run(input)
                    history.append({"role": "assistant", "content": response})
                    history.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    return f"Final Answer: 工具{action}不存在"
            else:
                # 4. 最终回答
                return self.parse_final_answer(response)
        
        return "达到最大步数，未得出答案"
```

**优点**：推理过程可解释，适合需要逻辑推理的任务。
**缺点**：每步都需要LLM调用，延迟和成本较高。

### 2.2 Plan-and-Execute模式

**原理**：先制定计划，再逐步执行

```
用户输入 → Planner（制定计划）→ Executor（执行计划）→ Re-planner（调整计划）
                │                    │                    │
           生成步骤列表          逐步执行            根据结果调整
```

**实现**：

```python
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools
    
    async def run(self, query: str) -> str:
        # 1. 制定计划
        plan = await self.create_plan(query)
        
        results = []
        for i, step in enumerate(plan.steps):
            # 2. 执行每一步
            result = await self.execute_step(step, results)
            results.append(result)
            
            # 3. 检查是否需要调整计划
            if self.needs_replan(result, plan):
                plan = await self.replan(query, results)
        
        # 4. 生成最终答案
        return await self.synthesize_answer(query, results)
    
    async def create_plan(self, query: str) -> Plan:
        prompt = f"为以下任务制定执行计划：\n{query}\n\n请输出JSON格式的步骤列表"
        response = await self.planner.chat([{"role": "user", "content": prompt}])
        return Plan.parse(response)
```

**优点**：整体规划，减少无效步骤。
**缺点**：计划可能不够灵活，需要replan机制。

### 2.3 工具增强架构

**架构设计**：

```
┌──────────────────────────────────────────────┐
│                Agent Core                     │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  推理引擎 │  │ 工具路由器 │  │ 记忆管理 │  │
│  │  (LLM)   │  │          │  │          │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │             │          │
│  ┌────▼─────────────▼─────────────▼─────┐  │
│  │           工具注册中心                 │  │
│  └────┬────────────┬────────────┬────────┘  │
│       │            │            │            │
│  ┌────▼───┐  ┌────▼───┐  ┌────▼───┐      │
│  │ 工具A  │  │ 工具B  │  │ 工具C  │      │
│  │(搜索)  │  │(数据库)│  │(API)   │      │
│  └────────┘  └────────┘  └────────┘      │
└──────────────────────────────────────────────┘
```

**工具注册与路由**：

```python
class ToolRouter:
    def __init__(self):
        self.tools = {}
        self.tool_descriptions = []
    
    def register(self, tool):
        self.tools[tool.name] = tool
        self.tool_descriptions.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        })
    
    def get_tool_prompt(self) -> str:
        """生成工具描述，注入到system prompt"""
        return "可用工具：\n" + json.dumps(self.tool_descriptions, ensure_ascii=False)
    
    async def route(self, tool_name: str, params: dict) -> str:
        """路由到具体工具执行"""
        if tool_name not in self.tools:
            return f"工具{tool_name}不存在"
        return await self.tools[tool_name].run(**params)
```

---

## 三、记忆系统架构

### 3.1 记忆层级

| 记忆类型 | 存储位置 | 生命周期 | 容量 | 用途 |
|---------|---------|---------|------|------|
| **工作记忆** | LLM上下文 | 当前对话 | 4K-128K | 当前任务状态 |
| **短期记忆** | 向量数据库 | 会话级 | 中等 | 会话历史 |
| **长期记忆** | 向量数据库 | 持久化 | 大 | 用户偏好/知识 |
| **情景记忆** | 关系数据库 | 持久化 | 大 | 具体事件记录 |

### 3.2 记忆管理架构

```
┌─────────────────────────────────────────────┐
│              记忆管理系统                     │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  写入器   │  │  检索器   │  │  压缩器  │  │
│  │          │  │          │  │          │  │
│  │•提取关键  │  │•语义搜索  │  │•摘要压缩  │  │
│  │•分类存储  │  │•时间排序  │  │•重要性筛选│  │
│  │•去重合并  │  │•相关性排序│  │•合并相似  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

### 3.3 记忆检索策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **语义搜索** | 基于向量相似度 | 查找相关内容 |
| **时间衰减** | 近期记忆权重更高 | 会话上下文 |
| **重要性排序** | 高重要性记忆优先 | 关键决策参考 |
| **混合检索** | 语义+时间+重要性综合 | 通用场景 |

---

## 四、多Agent协作架构

### 4.1 协作模式对比

| 模式 | 通信方式 | 复杂度 | 适用场景 |
|------|---------|--------|---------|
| **层级模式** | 上下级指令 | 低 | 明确分工 |
| **对等模式** | 点对点通信 | 中 | 灵活协作 |
| **投票模式** | 多数决 | 中 | 决策场景 |
| **管道模式** | 顺序传递 | 低 | 流水线处理 |
| **黑板模式** | 共享状态 | 高 | 知识共享 |

### 4.2 层级式多Agent架构

```
┌─────────────────────────────────────────┐
│           Orchestrator (编排者)           │
│         负责任务分解和结果汇总            │
└────┬─────────────┬─────────────┬────────┘
     │             │             │
┌────▼───┐   ┌────▼───┐   ┌────▼───┐
│Worker 1│   │Worker 2│   │Worker 3│
│(研究员) │   │(编码者) │   │(审查者) │
└────────┘   └────────┘   └────────┘
```

**实现**：

```python
class OrchestratorAgent:
    def __init__(self, planner, workers):
        self.planner = planner
        self.workers = workers  # {role: agent}
    
    async def run(self, task: str) -> str:
        # 1. 任务分解
        subtasks = await self.decompose(task)
        
        # 2. 分配给worker执行
        results = {}
        for subtask in subtasks:
            worker = self.select_worker(subtask)
            result = await worker.execute(subtask)
            results[subtask.id] = result
        
        # 3. 汇总结果
        return await self.synthesize(results)
    
    async def decompose(self, task: str) -> List[Subtask]:
        prompt = f"将以下任务分解为子任务：\n{task}"
        response = await self.planner.chat([{"role": "user", "content": prompt}])
        return Subtask.parse_list(response)
    
    def select_worker(self, subtask: Subtask) -> Agent:
        # 根据子任务类型选择合适的worker
        role_map = {
            "research": "researcher",
            "coding": "coder",
            "review": "reviewer"
        }
        role = role_map.get(subtask.type, "general")
        return self.workers[role]
```

### 4.3 Agent通信协议

```python
class AgentMessage:
    """Agent间通信的消息格式"""
    def __init__(self, sender: str, receiver: str, content: str, msg_type: str):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.msg_type = msg_type  # request/response/notification
        self.timestamp = time.time()
        self.metadata = {}

class MessageBus:
    """消息总线，负责Agent间通信"""
    def __init__(self):
        self.subscribers = {}  # agent_id -> callback
    
    def subscribe(self, agent_id: str, callback):
        self.subscribers[agent_id] = callback
    
    async def publish(self, message: AgentMessage):
        if message.receiver in self.subscribers:
            await self.subscribers[message.receiver](message)
```

---

## 五、架构设计最佳实践

### 5.1 关键设计原则

| 原则 | 说明 | 实践 |
|------|------|------|
| **单一职责** | 每个Agent只做一件事 | 避免"万能Agent" |
| **松耦合** | Agent间通过接口通信 | 不共享状态 |
| **可观测性** | 每个决策都有日志 | 记录推理过程 |
| **可恢复性** | 故障能自动恢复 | 检查点+重试 |
| **可观测性** | 运行状态透明 | 监控+告警 |

### 5.2 常见反模式

| 反模式 | 表现 | 解决方案 |
|--------|------|---------|
| **God Agent** | 一个Agent做所有事 | 拆分为多个专职Agent |
| **无限循环** | Agent反复调用同一工具 | 设置最大步数+循环检测 |
| **状态丢失** | Agent忘记之前做的事 | 引入记忆系统 |
| **工具爆炸** | 注册太多工具导致选择困难 | 分类+动态加载 |
| **上下文溢出** | 输入超出模型限制 | 压缩+摘要+分块 |

### 5.3 性能优化

| 优化方向 | 具体措施 |
|---------|---------|
| **延迟优化** | 流式输出+并行工具调用 |
| **成本优化** | 语义缓存+模型级联 |
| **并发优化** | 异步处理+连接池 |
| **质量优化** | Prompt优化+few-shot |

---

## 六、实战：设计一个代码审查Agent

### 6.1 需求分析

**目标**：自动审查代码变更，发现潜在问题

**功能拆分**：
- 代码分析：理解代码逻辑
- 问题检测：发现Bug/安全漏洞/性能问题
- 建议生成：给出改进建议
- 报告生成：生成结构化审查报告

### 6.2 架构设计

```
┌─────────────────────────────────────────────┐
│           Code Review Agent                  │
│                                              │
│  ┌──────────┐                               │
│  │  Orchestrator                            │
│  │  协调各阶段                                │
│  └────┬─────┘                               │
│       │                                      │
│  ┌────▼─────┐  ┌──────────┐  ┌──────────┐  │
│  │ 代码分析  │  │ 问题检测  │  │ 报告生成 │  │
│  │ Agent    │→│ Agent    │→│ Agent    │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

### 6.3 工具配置

| 工具 | 功能 | 输出 |
|------|------|------|
| **AST解析** | 代码结构分析 | 抽象语法树 |
| **静态分析** | 发现代码异味 | 问题列表 |
| **安全扫描** | 检测安全漏洞 | 漏洞报告 |
| **性能分析** | 识别性能瓶颈 | 优化建议 |

---

## 总结

Agent架构设计的核心要点：

1. **选对模式**：根据任务复杂度选择合适的架构模式
2. **分层设计**：推理层、工具层、记忆层分离
3. **可扩展**：工具、记忆、Agent都能灵活添加
4. **可观测**：每个决策都有日志和追踪
5. **可恢复**：故障能自动检测和恢复

> 好的Agent架构不是一步到位的，而是**从简单开始，根据需求逐步演进**。
