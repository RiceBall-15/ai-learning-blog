---
title: "多Agent协同模式：从简单编排到智能团队的进化之路"
description: "深入剖析多Agent系统的核心编排模式、通信拓扑、冲突解决机制与常见反模式，结合AutoGen、CrewAI、LangGraph等框架的生产实践，探讨如何构建高效的智能Agent团队。"
date: 2026-05-30
author: "技术学习笔记"
category: "agent"
subCategory: interview
tags: ["Agent", "MultiAgent", "Orchestration", "面试"]
---

# 多Agent协同模式：从简单编排到智能团队的进化之路

> 当单个Agent的能力触及天花板时，让多个Agent像人类团队一样协作，就成了突破能力边界的关键。但多Agent系统远非"多写几个Agent就能搞定"——它涉及编排拓扑、通信协议、冲突消解、容错恢复等一系列复杂工程问题。

## 一、为什么需要多Agent系统？

单Agent系统在处理**跨领域推理**、**长链路任务分解**、**高并发子任务**时，往往面临三个核心瓶颈：

| 瓶颈 | 表现 | 单Agent的局限 |
|------|------|--------------|
| **能力边界** | 一个Agent难以同时精通代码生成、数据分析、自然语言理解 | 角色固化，切换上下文损失大 |
| **上下文溢出** | 复杂任务需要超长上下文窗口 | Token限制导致信息丢失 |
| **错误累积** | 单点故障导致全链路崩溃 | 缺乏冗余和自我纠错机制 |

多Agent系统通过**角色分工、并行执行、交叉验证**来解决这些问题。但正如分布式系统一样，引入并发和通信就引入了复杂性。

## 二、核心编排模式（Orchestration Patterns）

### 2.1 顺序编排（Sequential Pipeline）

**最简单也最常见的模式**：Agent A → Agent B → Agent C，数据单向流动。

```
[需求分析Agent] → [架构设计Agent] → [代码生成Agent] → [测试Agent] → [部署Agent]
```

**特点：**
- 每个Agent的输入是前一个Agent的输出，形成DAG（有向无环图）
- 调试简单，链路清晰
- 容错性差：任何环节失败，后续全部停摆

**适用场景：** 文档处理流水线、ETL任务、内容审核链

**LangGraph实现示例：**

```python
from langgraph.graph import StateGraph, END

# 定义状态
class ArticleState(TypedDict):
    topic: str
    outline: str
    draft: str
    review_feedback: str
    final: str

# 构建顺序图
workflow = StateGraph(ArticleState)
workflow.add_node("outline", create_outline)
workflow.add_node("draft", write_draft)
workflow.add_node("review", review_draft)
workflow.add_node("revise", revise_draft)

workflow.add_edge("outline", "draft")
workflow.add_edge("draft", "review")
workflow.add_conditional_edges("review", decide_revision, {
    "needs_revision": "revise",
    "approved": END
})
workflow.add_edge("revise", "review")  # 回环修正

graph = workflow.compile()
```

### 2.2 并行编排（Parallel Fan-out/Fan-in）

**任务拆分后并行执行，最后汇聚结果。** 类似MapReduce模式。

```
            ┌→ [Agent A: 数据清洗] ──┐
[调度Agent] → [Agent B: 特征提取] ──┼→ [汇总Agent] → 最终结果
            └→ [Agent C: 模型推理] ──┘
```

**关键挑战：**
- **负载均衡**：不同子任务的耗时差异可能很大
- **结果一致性**：如何合并多个Agent的输出？
- **部分失败处理**：某个Agent超时或返回异常时怎么办？

**实际经验：** 在生产环境中，我建议为并行任务设置**超时阈值**和**降级策略**。例如，当3个并行Agent中有1个超时时，用该Agent的缓存/上次结果作为降级输出，而不是阻塞整个流程。

### 2.3 层级编排（Hierarchical Orchestration）

**自上而下的指挥链**，类似军队组织结构。一个Orchestrator Agent负责任务分配、进度监控和结果整合。

```
              [Orchestrator Agent]
             /         |          \
    [子任务A Agent] [子任务B Agent] [子任务C Agent]
         |              |              |
    [执行层Agent]  [执行层Agent]  [执行层Agent]
```

**优势：**
- 关注点分离：Orchestrator只管调度，执行Agent专注业务
- 可扩展性强：新增Agent只需注册到Orchestrator
- 层级容错：单个执行Agent失败不影响全局

**风险：**
- Orchestrator成为单点瓶颈和单点故障
- 多层通信带来显著的延迟累积
- Orchestrator的LLM调用成本最高（需理解所有子任务）

## 三、高级协作模式

### 3.1 Leader-Follower（领导者-追随者）模式

这是层级编排的典型实现。Leader负责任务拆解和分配，Follower执行具体工作并向Leader汇报。

**核心设计要点：**

```python
class LeaderAgent:
    def __init__(self):
        self.followers = {}  # name -> agent能力描述
        self.task_queue = []  # 待分配任务队列
        self.state_store = {}  # 全局状态存储

    def decompose_task(self, task: str) -> list[SubTask]:
        """将复杂任务拆解为子任务"""
        plan = self.llm.plan(
            task=task,
            available_agents=self.followers,
            constraints=self.constraints
        )
        return plan.sub_tasks

    def assign_and_monitor(self, sub_tasks):
        """分配任务并监控执行"""
        for task in sub_tasks:
            best_agent = self.select_agent(task)
            self.followers[best_agent].execute(task)
            # 异步监控 + 超时检测
            self.monitor_with_timeout(best_agent, task)
```

**生产实践注意事项：**
1. **Leader的LLM必须是能力最强的模型**（如GPT-4o），否则拆解质量堪忧
2. **Follower不需要知道彼此的存在**，降低耦合
3. **必须实现心跳机制**，检测Follower是否存活

### 3.2 Debate-Consensus（辩论-共识）模式

**多个Agent独立推理，然后交叉验证、辩论，最终达成共识。** 这是提升输出质量最有效的模式之一。

```
[Agent A: 独立分析] ──┐
[Agent B: 独立分析] ──┼→ [辩论轮] → [共识裁决] → 最终输出
[Agent C: 独立分析] ──┘
```

**典型实现流程：**

1. **独立推理阶段**：3个Agent各自完成任务，输出结果和置信度
2. **交叉审阅阶段**：每个Agent审阅其他Agent的结果，提出异议
3. **辩论阶段**：针对分歧点进行多轮辩论
4. **共识阶段**：由裁决Agent综合各方观点，输出最终结果

**AutoGen中的实现：**

```python
import autogen

# 定义三个不同视角的Agent
analyst_a = autogen.AssistantAgent(
    name="安全分析师",
    system_message="你专注于从安全角度分析代码，找出潜在漏洞。"
)

analyst_b = autogen.AssistantAgent(
    name="性能分析师",
    system_message="你专注于从性能角度分析代码，找出性能瓶颈。"
)

analyst_c = autogen.AssistantAgent(
    name="可维护性分析师",
    system_message="你专注于从代码可维护性角度分析，找出设计问题。"
)

# 裁决Agent
judge = autogen.AssistantAgent(
    name="裁决者",
    system_message="""你需要综合三位分析师的意见，
    对分歧点进行权衡，给出最终结论。如果存在无法调和的分歧，
    说明理由并给出你的推荐方案。"""
)

# GroupChat实现辩论
groupchat = autogen.GroupChat(
    agents=[analyst_a, analyst_b, analyst_c, judge],
    messages=[],
    max_round=8,  # 最多辩论8轮
    speaker_selection_method="round_robin",  # 轮流发言
)
```

**适用场景：**
- 代码Review（多视角审查）
- 风险评估（乐观/悲观/中立视角）
- 事实核查（多源信息交叉验证）
- 创意生成（头脑风暴 + 可行性评估）

**注意事项：** 辩论轮数需要精心设计。太少（1-2轮）可能无法收敛；太多（>8轮）会导致LLM调用成本激增且边际收益递减。**实测3-5轮是最佳平衡点。**

### 3.3 Swarm Intelligence（群体智能）

受蚂蚁群落、鸟群行为启发的**无中心协调模式**。每个Agent只与局部邻居交互，通过简单规则涌现出复杂的集体行为。

**核心特征：**
- **无全局调度者**：每个Agent自主决策
- **局部通信**：只与"邻居"Agent交换信息
- **涌现行为**：简单规则产生复杂的全局优化

**实现思路：**

```python
class SwarmAgent:
    def __init__(self, agent_id: str, neighbors: list[str]):
        self.id = agent_id
        self.neighbors = neighbors  # 通信拓扑
        self.local_state = {}  # 本地状态
        self.shared_pheromone = {}  # 共享的"信息素"

    def step(self):
        # 1. 收集邻居的信息素
        neighbor_signals = self.collect_signals()

        # 2. 更新本地状态（受邻居影响）
        self.update_state(neighbor_signals)

        # 3. 执行本地任务
        result = self.execute_local_task()

        # 4. 释放信息素（影响邻居）
        self.deposit_pheromone(result)

    def deposit_pheromone(self, result):
        """将自己的发现'广播'给邻居"""
        for neighbor in self.neighbors:
            self.shared_pheromone[neighbor] = result
```

**典型应用：**
- **分布式搜索**：多个Agent从不同方向搜索解决方案，通过信息素共享找到的线索
- **负载均衡**：任务Agent自动迁移到负载低的节点
- **故障自愈**：某个Agent故障后，邻居自动接管其任务

**局限性：** 群体智能在需要**全局一致性决策**的场景下表现不佳，更适合探索性、分布式任务。

## 四、通信拓扑（Communication Topology）

Agent之间的通信拓扑直接影响系统的性能、容错性和扩展性。

### 4.1 星型拓扑（Star Topology）

```
        [Agent B]
           ↑
[Agent A] ← [Center] → [Agent D]
           ↓
        [Agent C]
```

| 维度 | 评价 |
|------|------|
| **延迟** | 低（2跳可达任意Agent） |
| **容错性** | 差（中心节点故障=全系统故障） |
| **扩展性** | 中等（中心节点带宽有限） |
| **协调复杂度** | 低（中心统一调度） |

**适合场景：** Leader-Follower模式、Orchestrator编排

### 4.2 网状拓扑（Mesh Topology）

```
[Agent A] ←→ [Agent B]
  ↕    ╲   ╱    ↕
[Agent C] ←→ [Agent D]
```

| 维度 | 评价 |
|------|------|
| **延迟** | 低（1跳直达） |
| **容错性** | 优秀（无单点故障） |
| **扩展性** | 差（N个Agent需要N*(N-1)/2条链路） |
| **协调复杂度** | 高（需共识算法） |

**适合场景：** Debate-Consensus模式、Swarm Intelligence、小规模（<10个Agent）高可靠系统

### 4.3 链式拓扑（Chain Topology）

```
[Agent A] → [Agent B] → [Agent C] → [Agent D]
```

| 维度 | 评价 |
|------|------|
| **延迟** | 高（线性增长） |
| **容错性** | 极差（任一节点断裂=全链断裂） |
| **扩展性** | 优秀（线性扩展） |
| **协调复杂度** | 极低 |

**适合场景：** Sequential Pipeline、数据处理流水线

### 4.4 拓扑选择决策矩阵

```
                    容错性要求
                低 ────────── 高
           ┌─────────┬─────────┐
    扩  低 │  链式    │  星型    │
    展     │ (Pipeline)│(Orchestrate)│
    性  ───┼─────────┼─────────┤
    需  高 │  链式+   │  混合    │
    求     │  重试    │ (Mesh+Star)│
           └─────────┴─────────┘
```

**实战建议：** 大多数生产系统采用**混合拓扑**。例如，核心编排层用星型（Orchestrator ↔ 专业Agent），专业Agent之间用网状（跨域协作），执行层用链式（数据流水线）。

## 五、冲突解决机制

当多个Agent产生矛盾或冲突的输出时，需要系统化的解决机制。

### 5.1 优先级仲裁（Priority-based Arbitration）

为每个Agent分配优先级，冲突时高优先级Agent的输出胜出。

```python
def resolve_by_priority(results: dict[str, AgentResult]) -> AgentResult:
    """按优先级解决冲突"""
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].priority,
        reverse=True
    )
    return sorted_results[0][1]
```

**优点：** 简单、确定性强
**缺点：** 高优先级Agent可能被过度依赖，低优先级Agent的意见被忽略

### 5.2 投票机制（Voting）

多数投票、加权投票或一致性投票。

```python
def weighted_voting(results: dict[str, AgentResult]) -> AgentResult:
    """加权投票解决冲突"""
    scores = defaultdict(float)
    for agent_id, result in results.items():
        confidence = result.confidence  # 置信度作为权重
        for option in result.options:
            scores[option] += confidence * result.votes[option]
    return max(scores, key=scores.get)
```

### 5.3 裁决者模式（Judge/Referee）

引入一个独立的裁决Agent，综合各方观点做出最终决策。这在Debate-Consensus模式中已体现。

### 5.4 基于规则的消歧（Rule-based Disambiguation）

预定义冲突解决规则，适用于可预测的冲突场景。

```python
CONFLICT_RULES = [
    # 安全性 > 性能 > 代码简洁性
    ("safety", "performance", "safety"),
    ("safety", "readability", "safety"),
    ("performance", "readability", "readability"),
]

def rule_based_resolution(conflict_type: str) -> str:
    for rule in CONFLICT_RULES:
        if conflict_type == f"{rule[0]}_vs_{rule[1]}":
            return rule[2]
    return "escalate"  # 无法解决时升级
```

### 5.5 选择建议

| 场景 | 推荐机制 | 理由 |
|------|---------|------|
| 安全关键系统 | 优先级 + 规则 | 确定性优先 |
| 创意/探索任务 | 投票 | 多样性保护 |
| 复杂决策 | 裁决者 | 全局视角 |
| 实时系统 | 优先级 | 低延迟 |

## 六、常见反模式（Anti-patterns）与陷阱

### 6.1 ❌ 过度编排（Over-orchestration）

**症状：** 5个Agent能完成的任务，用了20个Agent和复杂的编排图。

**后果：**
- LLM调用成本飙升（每个Agent至少一次LLM调用）
- 端到端延迟增加
- 调试难度指数级增长

**解法：** 遵循 **"必要复杂度原则"** —— 先用最少的Agent跑通MVP，只在性能或质量瓶颈处引入更多Agent。

### 6.2 ❌ 无限循环（Infinite Loop）

**症状：** Agent A调用Agent B，B调用A，形成循环。

**解法：**
```python
class SafeOrchestrator:
    def __init__(self, max_rounds: int = 10):
        self.max_rounds = max_rounds
        self.execution_trace = []

    def execute(self, task):
        round_count = 0
        while not self.is_complete(task):
            round_count += 1
            if round_count > self.max_rounds:
                raise MaxRoundsExceeded(
                    f"超过最大轮次 {self.max_rounds}，"
                    f"最后状态: {task.state}"
                )
            # 记录执行轨迹
            self.execution_trace.append(task.current_state)
            task = self.next_step(task)
        return task.result
```

### 6.3 ❌ 沉默失败（Silent Failure）

**症状：** 某个Agent执行失败，但错误被吞掉，下游Agent拿到空结果继续执行。

**解法：** 实现**全链路追踪 + 结果校验**：
- 每个Agent的输出必须经过校验（非空、格式正确、质量达标）
- 失败时必须显式上报，不能静默跳过
- 使用OpenTelemetry等工具做分布式追踪

### 6.4 ❌ 上下文泄露（Context Leakage）

**症状：** Agent之间共享过多状态，导致意外耦合。

```
# 错误示范：Agent B意外依赖了Agent A的内部状态
agent_b.system_message = f"你已知 {agent_a.internal_state}"

# 正确做法：只传递必要的接口数据
agent_b.system_message = f"输入数据：{agent_a.get_public_output()}"
```

### 6.5 ❌ 单点依赖（Single Point of Dependency）

**症状：** 所有Agent的执行都依赖同一个外部服务（如某个API）。

**解法：** 实现**熔断器模式**和**降级策略**：
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_count = 0
        self.state = "closed"  # closed / open / half-open
        self.last_failure_time = None

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                return self.fallback(*args, **kwargs)

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            return self.fallback(*args, **kwargs)
```

### 6.6 ❌ Prompt Injection传播

**症状：** 用户输入被一个Agent处理后，恶意内容传播到下游Agent，引发越狱攻击。

**解法：**
- 每个Agent的输入都经过**清洗和验证**
- Agent之间的通信使用**结构化格式**（JSON Schema校验）
- 关键Agent设置**输出过滤器**

### 6.7 反模式速查表

| 反模式 | 危害等级 | 检测难度 | 根因 |
|--------|---------|---------|------|
| 过度编排 | ⭐⭐⭐ | 低 | 架构设计阶段缺乏简化意识 |
| 无限循环 | ⭐⭐⭐⭐⭐ | 中 | 缺少终止条件和轮次限制 |
| 沉默失败 | ⭐⭐⭐⭐ | 高 | 错误处理不完善 |
| 上下文泄露 | ⭐⭐⭐ | 中 | Agent边界定义模糊 |
| 单点依赖 | ⭐⭐⭐⭐ | 低 | 缺少冗余设计 |
| Prompt注入传播 | ⭐⭐⭐⭐⭐ | 高 | 缺少输入验证层 |

## 七、生产实践案例分析

### 7.1 AutoGen：微软的多Agent对话框架

**架构特点：**
- **GroupChat机制**：多个Agent在一个共享对话中交互
- **Speaker Selection**：支持Round Robin、Random、自定义选择策略
- **Human-in-the-loop**：可以插入人类Agent参与讨论

**生产实践中的经验：**

```python
# AutoGen v0.4 的多Agent软件开发团队
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

# 定义专业Agent
coder = AssistantAgent("coder", model_client=llm,
    system_message="你是高级Python开发者，负责编写代码。")

reviewer = AssistantAgent("reviewer", model_client=llm,
    system_message="你是代码审查专家，负责Review代码质量。")

tester = AssistantAgent("tester", model_client=llm,
    system_message="你是测试工程师，负责编写和运行测试用例。")

# 组建团队
team = RoundRobinGroupChat(
    participants=[coder, reviewer, tester],
    termination_condition=TextMentionTermination("APPROVED"),
    max_turns=15,
)

# 执行任务
result = await team.run(task="实现一个带缓存的HTTP客户端")
```

**踩坑经验：**
1. **GroupChat的max_turns必须设置**，否则Agent会陷入无意义的对话循环
2. **Agent的system_message需要明确职责边界**，否则会出现角色漂移
3. **Round Robin不如自定义speaker selection高效**，因为不考虑Agent的实际需求

### 7.2 CrewAI：面向角色的多Agent框架

**核心理念：** 将Agent建模为具有**角色（Role）、目标（Goal）、背景故事（Backstory）**的"虚拟员工"。

```python
from crewai import Agent, Task, Crew, Process

# 定义Agent（强调角色设计）
researcher = Agent(
    role="高级市场研究员",
    goal="深入分析目标市场，发现未被满足的需求",
    backstory="""你是一位拥有10年经验的市场研究专家，
    曾在多家顶级咨询公司工作，擅长从海量数据中发现趋势。""",
    tools=[search_tool, data_analysis_tool],
    llm="gpt-4o",
    verbose=True,
)

writer = Agent(
    role="内容策略师",
    goal="将研究洞察转化为引人入胜的商业报告",
    backstory="""你是一位资深商业写作者，擅长将复杂的数据
    转化为易于理解的商业洞察。""",
    llm="gpt-4o",
)

# 定义任务
research_task = Task(
    description="分析AI Agent市场的发展趋势和竞争格局",
    expected_output="包含市场规模、增长预测、主要玩家分析的报告",
    agent=researcher,
)

writing_task = Task(
    description="基于研究结果撰写执行摘要",
    expected_output="2000字以内的执行摘要，包含关键发现和建议",
    agent=writer,
    context=[research_task],  # 依赖研究任务的输出
)

# 组建Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
)

result = crew.kickoff()
```

**CrewAI的独特价值：**
1. **Backstory设计**是杀手锏——给Agent赋予"人格"显著提升输出质量
2. **Task的context依赖**天然支持数据流传递
3. **Process支持sequential和hierarchical**两种模式
4. 内置**memory系统**（短期记忆、长期记忆、实体记忆）

### 7.3 LangGraph：图驱动的Agent编排

**核心理念：** 将Agent工作流建模为**状态图（State Graph）**，节点是Agent或函数，边是状态转换。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class CodeReviewState(TypedDict):
    code: str
    issues: list[str]
    review_round: int
    approved: bool

def static_analysis(state: CodeReviewState) -> CodeReviewState:
    """静态分析Agent"""
    issues = run_linter(state["code"])
    return {"issues": issues}

def ai_review(state: CodeReviewState) -> CodeReviewState:
    """AI代码审查Agent"""
    new_issues = llm_review(state["code"])
    return {
        "issues": state["issues"] + new_issues,
        "review_round": state["review_round"] + 1
    }

def decide_next(state: CodeReviewState):
    """决策节点"""
    if not state["issues"]:
        return "approved"
    if state["review_round"] >= 3:
        return "force_approve"  # 防止无限循环
    return "needs_fix"

# 构建图
graph = StateGraph(CodeReviewState)
graph.add_node("static_analysis", static_analysis)
graph.add_node("ai_review", ai_review)

graph.set_entry_point("static_analysis")
graph.add_edge("static_analysis", "ai_review")
graph.add_conditional_edges("ai_review", decide_next, {
    "needs_fix": "ai_review",  # 回环修正
    "approved": END,
    "force_approve": END,
})

app = graph.compile()
```

**LangGraph的差异化优势：**
1. **状态显式管理**——每个节点的输入输出都是显式的TypedDict
2. **可视化调试**——图结构天然支持可视化
3. **Human-in-the-loop**——可以中断图执行，等待人工输入后继续
4. **Checkpointer**——支持断点续传和时间旅行调试

### 7.4 三大框架对比

| 维度 | AutoGen | CrewAI | LangGraph |
|------|---------|--------|-----------|
| **核心抽象** | 对话 | 角色+任务 | 状态图 |
| **学习曲线** | 中 | 低 | 高 |
| **灵活性** | 高 | 中 | 极高 |
| **调试体验** | 中 | 好 | 优秀 |
| **生产稳定性** | 中 | 中 | 高 |
| **适合场景** | 研究/探索 | 业务工作流 | 复杂有状态流程 |
| **社区生态** | 大（微软） | 活跃 | 快速增长 |

## 八、多Agent系统性能优化

### 8.1 减少LLM调用

LLM调用是多Agent系统的**最大成本和延迟来源**。优化策略：

1. **缓存相同输入的响应**：对于确定性任务，用语义缓存避免重复调用
2. **批量处理**：合并多个小请求为一次批量调用
3. **模型降级**：非关键Agent使用更小/更快的模型
4. **提前终止**：设置置信度阈值，达标时提前退出

```python
class SmartOrchestrator:
    def __init__(self):
        self.cache = SemanticCache()  # 语义缓存
        self.model_tier = {
            "orchestrator": "gpt-4o",       # 核心决策用强模型
            "analyst": "gpt-4o-mini",       # 分析用中等模型
            "formatter": "gpt-4o-mini",     # 格式化用小模型
            "validator": "rules_engine",     # 校验用规则引擎，零LLM
        }

    async def execute(self, task: str):
        # 1. 检查缓存
        cached = self.cache.get(task)
        if cached:
            return cached

        # 2. 按需分配模型
        agent_model = self.model_tier[agent.role]
        if agent_model == "rules_engine":
            result = await self.execute_rules(agent, task)
        else:
            result = await self.execute_llm(agent, task, model=agent_model)

        # 3. 缓存结果
        self.cache.set(task, result)
        return result
```

### 8.2 并行化优化

```python
import asyncio

async def parallel_execution(agents: list[Agent], task: Task):
    """并行执行多个Agent，带超时和降级"""
    tasks = [agent.execute(task) for agent in agents]

    # 带超时的并行执行
    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
        timeout=30  # 全局超时30秒
    )

    # 处理部分失败
    successful = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Agent {agents[i].name} 失败: {result}")
            # 降级策略
            fallback = await get_fallback_result(agents[i], task)
            successful.append(fallback)
        else:
            successful.append(result)

    return successful
```

### 8.3 流水线与批处理

对于多轮交互的场景，**流式处理**可以显著降低端到端延迟：

```python
async def streaming_pipeline(agents, initial_input):
    """流水线式处理：Agent A完成第一步后立即开始，不用等所有步骤"""
    current_output = initial_input

    for agent in agents:
        # Agent A的输出立即流入Agent B，不等待Agent C
        current_output = await agent.process_streaming(current_output)

    return current_output
```

### 8.4 资源预算控制

```python
class BudgetController:
    def __init__(self, max_tokens: int = 100_000, max_cost_usd: float = 5.0):
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.used_tokens = 0
        self.used_cost = 0.0

    def can_afford(self, estimated_tokens: int) -> bool:
        return (
            self.used_tokens + estimated_tokens <= self.max_tokens
            and self.used_cost + self.estimate_cost(estimated_tokens) <= self.max_cost_usd
        )

    def record_usage(self, tokens: int, cost: float):
        self.used_tokens += tokens
        self.used_cost += cost
```

### 8.5 性能监控指标

| 指标 | 目标值 | 监控方式 |
|------|--------|---------|
| 端到端延迟 | P95 < 60s | 分布式追踪 |
| Agent成功率 | > 95% | 执行日志 |
| LLM调用次数/任务 | < 10 | 计数器 |
| Token消耗/任务 | < 50K | 用量统计 |
| 冲突解决延迟 | < 5s | 计时器 |
| 任务完成率 | > 90% | 业务指标 |

## 九、面试高频问题

### Q1: 如何选择多Agent编排模式？

**回答框架：**
1. **任务是否可并行？** → 是：并行/Fan-out模式；否：顺序/Pipeline模式
2. **是否需要多视角验证？** → 是：Debate-Consensus模式
3. **任务复杂度是否需要分层？** → 是：Hierarchical/Leader-Follower模式
4. **是否需要高容错？** → 是：Mesh拓扑 + 重试 + 降级
5. **规模多大？** → >10个Agent考虑Swarm，<5个考虑GroupChat

### Q2: 多Agent系统的调试比单Agent难在哪里？

**核心难点：**
- **可观测性差**：多个Agent的对话交织，难以追踪哪个决策导致了最终结果
- **非确定性**：同样的输入可能因为Agent交互顺序不同产生不同结果
- **级联故障**：一个Agent的微小错误可能在下游被放大
- **状态爆炸**：N个Agent各有K种状态，总状态空间是K^N

**解法：** 分布式追踪（LangSmith/Langfuse）+ 结构化日志 + 断点续传

### Q3: 如何控制多Agent系统的成本？

**策略：**
1. 模型分级：关键Agent用强模型，辅助Agent用弱模型
2. 语义缓存：避免重复调用
3. 早停机制：达到质量要求后提前终止
4. 预算控制器：硬性限制Token和成本上限
5. 定期审计：分析Agent调用链，消除不必要的LLM调用

## 十、总结

多Agent系统的核心不是"多"，而是"协"。一个好的多Agent架构应该：

1. **简单优先**：能用1个Agent解决的问题不要用3个
2. **边界清晰**：每个Agent有明确的职责和输出契约
3. **容错优先**：默认假设任何Agent都可能失败
4. **可观测**：全链路追踪，每个决策可追溯
5. **成本可控**：预算机制、缓存策略、模型分级

从AutoGen的对话驱动，到CrewAI的角色驱动，再到LangGraph的图驱动，多Agent框架正在快速进化。但无论框架如何变化，**系统设计的核心原则不变**——清晰的边界、可靠的通信、优雅的降级。

---

*本文基于AutoGen 0.4+、CrewAI 0.100+、LangGraph 0.2+版本撰写。技术细节可能随框架版本更新而变化，请参考官方文档获取最新信息。*
