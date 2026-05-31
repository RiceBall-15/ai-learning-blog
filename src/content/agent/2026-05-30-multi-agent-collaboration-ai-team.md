---
title: '多Agent协同与AI Team自主协作：架构设计与实战'
description: '深入解析多Agent协作拓扑、AutoGen/CrewAI/LangGraph三大框架实战、任务分解与冲突解决策略，以及AI Team自主协作的生产化设计'
date: 2026-05-30
author: 'RiceBall-15'
category: 'agent'
subCategory: 'interview'
tags: ['多Agent', 'AutoGen', 'CrewAI', 'AI Team', '协作架构']
draft: false
---

# 多Agent协同与AI Team自主协作：架构设计与实战

## 引言

单个Agent在面对复杂任务时暴露出明显的天花板。当任务涉及多领域知识融合、长链推理、并行处理或多步决策时，单Agent架构捉襟见肘。多Agent协同系统应运而生——让多个专业化Agent像团队一样分工协作，彼此通信、互相校验、共同产出高质量结果。本文将从架构设计、主流框架实战、任务分解与冲突解决机制，到AI Team自主协作的生产化考量，进行系统性的深度剖析。

---

## 一、为什么需要多Agent：单Agent的局限性

单Agent系统在以下场景中遭遇根本性瓶颈：

**上下文窗口溢出。** LLM的上下文窗口是有限资源。当一个Agent需要同时持有任务规划、工具调用结果、历史对话、领域知识时，token很快被耗尽。多Agent通过角色拆分，将不同上下文分配给不同Agent，有效缓解这一压力。

**能力专业化不足。** 一个Agent的prompt很难同时精通代码生成、安全审计、架构设计和文档撰写。就像现实中的团队不可能一个人承担所有角色，Agent也需要专业化分工。

**推理深度与广度的矛盾。** 深度推理需要大量token展开思维链，广度探索需要并行多路径尝试。单Agent在同一时刻只能选择一条路径，多Agent则可以并行探索后聚合结果。

**缺乏校验与纠错机制。** 单Agent生成的结果没有第二视角来审核，错误容易积累。多Agent可以实现"生成-审核-修正"的闭环，显著提升输出质量。

**错误隔离需求。** 单Agent一旦在某一步出错，后续所有步骤都建立在错误基础上。多Agent系统中，一个Agent的失败不会直接污染其他Agent的工作。

---

## 二、多Agent协作拓扑：星型/网状/层级/环形

Agent之间的通信拓扑决定了系统的协作效率和容错能力。

### 2.1 星型拓扑（Hub-and-Spoke）

所有Agent通过中心协调者（Orchestrator）通信。协调者负责任务分配、结果收集和最终聚合。

```
        [Orchestrator]
       /    |    |    \
  [Agent A] [Agent B] [Agent C] [Agent D]
```

**优势：** 架构简单，协调者可做全局决策，易于监控。
**劣势：** 协调者是单点瓶颈，扩展性受限。
**适用：** 任务可明确拆分为独立子任务的场景。

### 2.2 网状拓扑（Mesh）

Agent之间点对点直接通信，无需中心节点。每个Agent可以向任意其他Agent发送消息。

```
  [Agent A] ←→ [Agent B]
      ↕    ╲  ╱    ↕
  [Agent C] ←→ [Agent D]
```

**优势：** 高容错，无单点故障；信息传播快。
**劣势：** 通信复杂度O(n²)，需要明确的路由策略避免消息风暴。
**适用：** 需要深度交互的讨论式协作（如多Agent辩论）。

### 2.3 层级拓扑（Hierarchical）

Agent按层级组织，上层Agent管理下层Agent，形成树状结构。

```
         [Manager]
        /         \
  [Tech Lead]   [PM]
   /    \         |
[Dev] [QA]   [Docs]
```

**优势：** 符合组织管理直觉，职责清晰。
**劣势：** 层级过深导致延迟增大。
**适用：** 工程团队模拟、复杂项目的分层管理。

### 2.4 环形拓扑（Ring）

Agent按环形排列，消息依次传递，每个Agent处理后转发给下一个。

```
  [Agent A] → [Agent B]
      ↑              ↓
  [Agent D] ← [Agent C]
```

**优势：** 适合流水线式处理，每个环节专注自己的工序。
**劣势：** 任一环节阻塞影响全链路。
**适用：** 文本润色流水线（翻译→校对→格式化→审核）。

---

## 三、AutoGen框架：ConversableAgent, GroupChat, Manager模式

AutoGen由微软开发，核心思想是**Agent之间通过对话协作完成任务**。它的基本抽象是`ConversableAgent`——任何可以发送和接收消息的Agent。

### 3.1 核心概念

- **ConversableAgent：** 基础Agent类，支持对话、代码执行、人类反馈
- **AssistantAgent：** 带预设系统提示的助手Agent
- **UserProxyAgent：** 代理人类输入或自动触发回复
- **GroupChat：** 多Agent群聊管理器，控制发言顺序
- **GroupChatManager：** 驱动群聊的运行时

### 3.2 完整代码示例：3-Agent代码审查系统

```python
"""
AutoGen 多Agent代码审查系统
场景：Code Author生成代码 → Security Reviewer审查安全性 → 
      Performance Reviewer审查性能 → 最终汇总
"""
import autogen

# 全局配置
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": "your-api-key",
            "temperature": 0.3,
        }
    ],
    "timeout": 120,
}

# ========== 定义角色 ==========

# 代码作者：接收需求，编写代码
code_author = autogen.AssistantAgent(
    name="CodeAuthor",
    system_message="""你是一个资深Python开发者。
当收到编码需求时，编写清晰、高效的Python代码。
代码必须包含类型注解、docstring和错误处理。
将代码块发送给SecurityReviewer进行安全审查。""",
    llm_config=llm_config,
)

# 安全审查员：专注安全漏洞检测
security_reviewer = autogen.AssistantAgent(
    name="SecurityReviewer",
    system_message="""你是一个应用安全专家，专注于代码安全审查。
检查以下安全问题：
1. SQL注入、XSS、命令注入等注入攻击
2. 硬编码密钥和敏感信息泄露
3. 不安全的反序列化
4. 认证和授权缺陷
5. 不安全的文件操作

输出格式：
- 🔴 严重：必须修复
- 🟡 警告：建议修复  
- 🟢 通过：无安全问题

审查完成后，将结果和原始代码一起发送给PerformanceReviewer。""",
    llm_config=llm_config,
)

# 性能审查员：专注性能和代码质量
performance_reviewer = autogen.AssistantAgent(
    name="PerformanceReviewer",
    system_message="""你是一个性能工程专家，专注于代码性能和质量审查。
检查以下问题：
1. 时间复杂度和空间复杂度
2. 数据库查询效率（N+1问题等）
3. 内存泄漏风险
4. 并发安全性
5. 代码可维护性和设计模式

输出最终审查报告，包含安全审查结果的整合，
以及你自己的性能分析。格式化为完整的审查报告。""",
    llm_config=llm_config,
)

# 人类代理：自动回复以推进流程
human_proxy = autogen.UserProxyAgent(
    name="HumanProxy",
    human_input_mode="NEVER",  # 全自动模式
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(
        "审查报告结束"
    ),
    code_execution_config=False,
)

# ========== 三人对话模式 ==========

def run_direct_review():
    """方式1：手动编排的两人审查链"""
    # Code Author 写代码
    human_proxy.initiate_chat(
        code_author,
        message="""请实现一个用户认证函数，要求：
1. 支持邮箱/密码登录
2. 密码使用bcrypt哈希
3. 生成JWT token
4. 包含输入验证和错误处理""",
    )

# ========== 群聊模式（更灵活）==========

def run_group_review():
    """方式2：GroupChat多Agent群聊审查"""
    group_chat = autogen.GroupChat(
        agents=[human_proxy, code_author, 
                security_reviewer, performance_reviewer],
        messages=[],
        max_round=15,
        speaker_selection_method="auto",  # LLM自动决定谁说话
        # 也可用 "round_robin" 轮流发言
        # 或自定义函数决定
    )

    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    human_proxy.initiate_chat(
        manager,
        message="""请团队协作完成以下任务：
CodeAuthor实现一个简单的REST API端点（FastAPI），
SecurityReviewer进行安全审查，
PerformanceReviewer进行性能审查。
最终输出完整的代码和审查报告。

具体需求：实现 /api/users/{id} 的GET接口，
从数据库查询用户信息并返回。""",
    )

# ========== Manager模式：委托式协作 ==========

def run_manager_delegation():
    """方式3：Manager模式——Manager分配任务"""
    manager = autogen.AssistantAgent(
        name="Manager",
        system_message="""你是项目Manager，负责：
1. 分析任务需求
2. 将任务拆解并分配给合适的团队成员
3. 汇总所有成员的输出
4. 确保最终交付物完整

你的团队成员：
- CodeAuthor：负责编码实现
- SecurityReviewer：负责安全审查
- PerformanceReviewer：负责性能审查

请按照 分配→审查→汇总 的流程推进。""",
        llm_config=llm_config,
    )

    # 给Manager创建一个UserProxy来接收人类任务
    proxy = autogen.UserProxyAgent(
        name="TaskIssuer",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )

    proxy.initiate_chat(
        manager,
        message="""新任务：实现一个JWT认证中间件，
要求支持token刷新、黑名单机制和速率限制。
请Manager分配任务并协调团队完成。""",
    )

if __name__ == "__main__":
    run_group_review()
```

### 3.3 AutoGen关键设计要点

- **`speaker_selection_method="auto"`：** 由LLM根据对话上下文决定下一个发言者，适合开放讨论
- **`is_termination_msg`：** 定义终止条件，防止Agent无限循环
- **`max_round`：** 硬性限制对话轮次，避免token烧穿
- **Human-in-the-loop：** 通过`human_input_mode="TERMINATE"`可以在关键节点暂停等待人类确认

---

## 四、CrewAI框架：Crew, Agent, Task, Process

CrewAI的设计哲学是**模拟人类团队协作**。它用Crew（团队）、Agent（成员）、Task（任务）、Process（流程）四个核心抽象来构建多Agent系统。

### 4.1 核心概念映射

| CrewAI概念 | 类比 | 作用 |
|------------|------|------|
| Crew | 团队 | 包含Agent和Task的协作单元 |
| Agent | 团队成员 | 有角色、目标、背景故事的个体 |
| Task | 具体任务 | 分配给特定Agent的工作项 |
| Process | 工作流程 | 顺序执行、层级管理或共识决策 |

### 4.2 完整代码示例：CrewAI内容生产团队

```python
"""
CrewAI 多Agent内容生产团队
场景：编辑团队协作完成一篇技术博客
Researcher调研 → Writer撰写 → Editor审核 → 最终发布
"""
from crewai import Agent, Task, Crew, Process
from crewai.tools import SerperDevTool, WebsiteSearchTool
from pydantic import BaseModel

# ========== 定义工具 ==========
# search_tool = SerperDevTool()  # 需要 SERPER_API_KEY
# scrape_tool = WebsiteSearchTool()

# ========== 定义Agent（团队成员）==========

researcher = Agent(
    role="高级技术研究员",
    goal="深入调研给定技术主题，收集最新资料和数据",
    backstory="""你是一位拥有10年技术写作经验的研究员。
你擅长从多个来源收集信息，验证数据准确性，
并整理成结构化的调研报告。你对AI、分布式系统、
云原生等技术领域有深入了解。""",
    verbose=True,
    allow_delegation=True,  # 允许委托其他Agent
    max_iter=5,
    # tools=[search_tool, scrape_tool],
)

writer = Agent(
    role="资深技术作者",
    goal="将调研结果转化为高质量的技术博客文章",
    backstory="""你是一位技术博客作者，擅长将复杂概念
用通俗易懂的语言解释清楚。你的文章结构清晰、
举例恰当、代码示例丰富。你坚持"先讲Why再讲How"的写作原则。
文章字数控制在3000-5000字，面向中高级开发者。""",
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

editor = Agent(
    role="技术编辑",
    goal="审核文章质量，确保技术准确性、可读性和完整性",
    backstory="""你是一位严格的技术编辑，拥有丰富的
技术出版经验。你关注：技术准确性、逻辑连贯性、
代码可运行性、表述的专业性。你会给出具体的
修改建议，而不是笼统的评价。""",
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

# ========== 定义Task（具体任务）==========

research_task = Task(
    description="""调研"多Agent协作架构"主题，产出一份调研报告。
要求：
1. 梳理主流多Agent框架（AutoGen、CrewAI、LangGraph）的特点
2. 收集至少3个生产级案例
3. 总结各架构模式的适用场景
4. 整理关键性能数据和基准测试结果
输出格式：结构化Markdown报告""",
    expected_output="结构化的Markdown调研报告，包含框架对比表格和案例分析",
    agent=researcher,
)

writing_task = Task(
    description="""基于调研报告撰写一篇技术博客文章。
标题：《多Agent协同：从架构设计到生产实践》
要求：
1. 包含架构设计的图示说明
2. 至少一个完整的代码示例
3. 包含生产化部署的注意事项
4. 文风专业但易懂，面向中高级开发者
5. 字数4000-5000字""",
    expected_output="完整的Markdown格式技术博客文章",
    agent=writer,
    context=[research_task],  # 依赖调研任务的输出
)

editing_task = Task(
    description="""审核技术博客文章，输出审核意见。
审核维度：
1. 【技术准确性】代码示例是否可运行？概念是否正确？
2. 【逻辑连贯性】文章结构是否清晰？过渡是否自然？
3. 【可读性】是否面向目标读者？术语使用是否恰当？
4. 【完整性】是否覆盖了核心要点？是否有遗漏？
输出格式：逐项审核意见 + 修改建议 + 最终评分（1-10）""",
    expected_output="结构化的审核报告，包含评分和修改建议",
    agent=editor,
    context=[writing_task],  # 依赖写作任务的输出
)

# ========== 组建Crew（团队）==========

content_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # 顺序执行：调研→写作→审核
    verbose=True,
    # memory=True,  # 开启记忆，Agent可以引用之前的对话
    # max_rpm=10,   # 限制每分钟请求次数
)

# ========== 执行 ==========

def run_sequential():
    """顺序流程：按任务依赖依次执行"""
    result = content_crew.kickoff()
    print("\n" + "=" * 60)
    print("最终输出：")
    print(result.raw)

def run_hierarchical():
    """层级流程：Manager Agent自动分配任务"""
    hierarchical_crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",  # Manager使用更强的模型
        verbose=True,
    )
    result = hierarchical_crew.kickoff()

# ========== 自定义Process：共识决策 ==========

from crewai import Crew
from typing import List

class ConsensusProcess:
    """共识流程：多个Agent独立产出，投票/聚合决定最终结果"""
    
    def __init__(self, agents: List[Agent], threshold: float = 0.7):
        self.agents = agents
        self.threshold = threshold
    
    def execute(self, task: Task):
        """并行执行所有Agent的任务，聚合结果"""
        results = []
        for agent in self.agents:
            result = agent.execute_task(task)
            results.append({
                "agent": agent.role,
                "output": result,
            })
        
        # 聚合：可以用另一个Agent来综合所有结果
        synthesizer = Agent(
            role="结果聚合器",
            goal="综合多个Agent的输出，产出最优结果",
            backstory="你擅长整合多方观点，提炼共识。",
        )
        
        synthesis_task = Task(
            description=f"综合以下{len(results)}个Agent的输出：\n" +
                "\n".join([f"## {r['agent']}\n{r['output']}" 
                          for r in results]),
            expected_output="综合后的最终输出",
            agent=synthesizer,
        )
        return synthesizer.execute_task(synthesis_task)

if __name__ == "__main__":
    run_sequential()
```

### 4.3 CrewAI三种Process对比

| Process | 执行方式 | 适用场景 | 优势 | 劣势 |
|---------|----------|----------|------|------|
| `sequential` | 按任务顺序依次执行 | 有明确依赖关系的流水线 | 简单可控 | 无法并行 |
| `hierarchical` | Manager自动分配任务 | 复杂项目需要动态调度 | 灵活，自动决策 | Manager成为瓶颈 |
| 自定义consensus | 并行执行+聚合 | 多视角分析、A/B方案对比 | 并行高效 | 聚合逻辑需自定义 |

---

## 五、LangGraph多Agent：子图+消息传递

LangGraph的核心优势在于将多Agent协作建模为**状态图（StateGraph）**。每个Agent是图中的一个节点，Agent之间的消息传递通过图的边来实现。

### 5.1 架构思想

```python
"""
LangGraph多Agent架构示意
核心：每个Agent是一个子图，Agent之间通过共享状态通信
"""
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)

# 定义Agent（每个Agent是一个子图节点）
code_agent = create_react_agent(
    model, 
    tools=[], 
    prompt="你是代码开发者，负责编写代码。"
)
review_agent = create_react_agent(
    model,
    tools=[],
    prompt="你是代码审查员，负责审查代码质量和安全性。"
)

# 构建多Agent工作流图
workflow = StateGraph(MessagesState)

# 添加Agent节点
workflow.add_node("coder", code_agent)
workflow.add_node("reviewer", review_agent)

# 定义路由逻辑
def should_continue(state: MessagesState):
    """根据审查结果决定是否需要修改"""
    last_message = state["messages"][-1]
    if "需要修改" in last_message.content:
        return "coder"
    return END

# 构建边
workflow.add_edge(START, "coder")
workflow.add_conditional_edges("coder", lambda _: "reviewer")
workflow.add_conditional_edges("reviewer", should_continue)

# 编译运行
app = workflow.compile()
result = app.invoke({
    "messages": [("user", "写一个快速排序算法")]
})
```

### 5.2 LangGraph的独特优势

- **状态持久化：** 每个节点的中间状态可保存，支持断点续跑
- **人机协作节点：** `interrupt`功能可在关键节点暂停等待人类输入
- **子图复用：** 一个Agent的子图可以在多个工作流中复用
- **可视化调试：** 图结构天然支持可视化，方便调试和理解

---

## 六、任务分解策略

### 6.1 递归分解（Divide and Conquer）

将复杂任务递归分解为更小的子任务，直到每个子任务可以由单个Agent独立完成。

```
[实现电商系统]
├── [用户模块]
│   ├── [注册/登录]
│   ├── [用户画像]
│   └── [权限管理]
├── [商品模块]
│   ├── [商品CRUD]
│   ├── [库存管理]
│   └── [搜索推荐]
└── [订单模块]
    ├── [下单流程]
    ├── [支付集成]
    └── [物流追踪]
```

### 6.2 MapReduce模式

**Map阶段：** 将任务拆分为独立子任务，并行分配给多个Agent执行。
**Reduce阶段：** 收集所有Agent的输出，由聚合Agent合并为最终结果。

```python
# 伪代码示意
def map_reduce(task, agents):
    # Map: 并行执行
    sub_tasks = decompose(task)
    results = parallel_execute(agents, sub_tasks)
    
    # Reduce: 聚合结果
    return aggregator_agent.synthesize(results)
```

### 6.3 投票决策

多个Agent对同一问题独立给出答案，通过投票机制选出最优解。

```python
def voting_decision(task, agents, strategy="majority"):
    """投票决策策略"""
    answers = [agent.solve(task) for agent in agents]
    
    if strategy == "majority":
        return Counter(answers).most_common(1)[0][0]
    elif strategy == "weighted":
        # 根据Agent的历史准确率加权
        weights = [agent.confidence for agent in agents]
        return weighted_majority(answers, weights)
    elif strategy == "llm_judge":
        # 用另一个LLM来评判哪个答案最好
        return judge_agent.evaluate(answers)
```

---

## 七、冲突解决机制

多Agent系统中，Agent之间可能产生矛盾——对同一问题给出不同结论，或在资源分配上产生竞争。

### 7.1 优先级仲裁

为每个Agent分配优先级权重，当产生冲突时，高优先级Agent的意见被采纳。

```python
PRIORITY = {
    "SecurityAgent": 10,  # 安全最高优先
    "PerformanceAgent": 7,
    "FeatureAgent": 5,
    "UXAgent": 3,
}

def resolve_by_priority(conflicts):
    """按优先级解决冲突"""
    return max(conflicts, key=lambda c: PRIORITY[c.agent_name])
```

### 7.2 投票机制

适用于Agent地位平等的场景。少数服从多数，或加权投票。

### 7.3 Leader裁决

指定一个Leader Agent，当其他Agent产生分歧时，由Leader做最终裁决。Leader通常使用更强的模型（如GPT-4o而非GPT-4o-mini）。

```python
class LeaderArbiter:
    def __init__(self, leader_model):
        self.leader = Agent(role="仲裁者", model=leader_model)
    
    def arbitrate(self, agent_a_view, agent_b_view, context):
        return self.leader.decide(
            f"Agent A认为：{agent_a_view}\n"
            f"Agent B认为：{agent_b_view}\n"
            f"上下文：{context}\n"
            f"请做出最终裁决并说明理由。"
        )
```

---

## 八、AI Team自主协作：Agent自组织、角色分工、结果聚合

AI Team自主协作是多Agent系统的高级形态——Agent不依赖预设的固定流程，而是根据任务动态自组织。

### 8.1 自组织机制

```python
class AITeam:
    """AI Team自主协作框架"""
    
    def __init__(self, available_agents):
        self.pool = available_agents  # Agent人才库
        self.active_team = []
    
    def form_team(self, task):
        """根据任务动态组建团队"""
        # 分析任务需求
        task_analysis = self.analyze_task(task)
        
        # 从Agent池中选择合适的成员
        self.active_team = [
            agent for agent in self.pool
            if agent.can_handle(task_analysis)
        ]
        
        # 自动分配角色
        self.assign_roles(task_analysis)
        
        return self.active_team
    
    def assign_roles(self, analysis):
        """基于能力匹配自动分配角色"""
        role_mapping = {
            "coding": "developer",
            "reviewing": "reviewer",
            "testing": "tester",
            "documenting": "technical_writer",
        }
        
        for agent in self.active_team:
            best_role = max(
                role_mapping.values(),
                key=lambda role: agent.skill_score(role)
            )
            agent.assigned_role = best_role
    
    def execute(self, task):
        """自主协作执行"""
        team = self.form_team(task)
        
        # 动态决定执行策略
        if self.is_parallelizable(task):
            results = self.parallel_execute(team, task)
        else:
            results = self.sequential_execute(team, task)
        
        # 结果聚合
        return self.aggregate(results)
```

### 8.2 角色分工的动态调整

在执行过程中，Agent可以根据中间结果动态调整分工：

- **能力发现：** Agent在执行中暴露新的能力，团队重新分配任务
- **负载均衡：** 当某个Agent过载时，自动将部分任务转移给空闲Agent
- **升级/降级：** 简单子任务分配给小模型Agent，复杂子任务升级给大模型

### 8.3 结果聚合策略

- **加权融合：** 根据Agent在该领域的专长权重聚合
- **链式验证：** Agent A的输出经过Agent B验证后才纳入最终结果
- **辩论收敛：** 多个Agent对分歧点进行多轮辩论，逐步收敛到共识

---

## 九、生产化考量：隔离/超时/降级/监控

多Agent系统上线生产环境，必须解决以下关键问题：

### 9.1 故障隔离

```python
import asyncio
from asyncio import TimeoutError

async def isolated_agent_call(agent, task, timeout=30):
    """带隔离的Agent调用"""
    try:
        result = await asyncio.wait_for(
            agent.execute(task),
            timeout=timeout
        )
        return {"status": "success", "result": result}
    except TimeoutError:
        return {"status": "timeout", "result": None}
    except Exception as e:
        return {"status": "error", "result": None, "error": str(e)}
```

### 9.2 超时与降级

每个Agent调用必须有超时控制。当关键Agent超时或失败时，系统需要降级策略：

- **重试降级：** 大模型失败→回退到小模型
- **功能降级：** 复杂分析失败→返回简化版结果
- **人工降级：** 自动化失败→转交人类处理

### 9.3 可观测性

```python
import time
import logging

class AgentTracer:
    """多Agent调用链追踪"""
    
    def __init__(self):
        self.spans = []
    
    def trace(self, agent_name, task_id):
        """记录Agent调用的完整生命周期"""
        span = {
            "agent": agent_name,
            "task_id": task_id,
            "start_time": time.time(),
            "token_usage": 0,
            "status": "running",
        }
        self.spans.append(span)
        
        def finish(status="success"):
            span["end_time"] = time.time()
            span["duration"] = span["end_time"] - span["start_time"]
            span["status"] = status
            logging.info(
                f"[Trace] {agent_name} | {task_id} | "
                f"{span['duration']:.2f}s | {status}"
            )
        
        return finish
```

### 9.4 资源控制

- **Token预算：** 为整个多Agent会话设定总token上限，避免无限消耗
- **并发控制：** 限制同时运行的Agent数量，防止API限流
- **成本追踪：** 按Agent维度统计token消耗，便于成本优化

---

## 十、面试深度：设计一个3-Agent代码审查系统

> **面试题：** 请设计一个3-Agent协作的代码审查系统，说明架构设计、Agent角色定义、通信机制和异常处理。

### 参考答案

**架构选择：** 星型拓扑 + 流水线模式混合。一个Orchestrator负责任务调度，三个Agent按流水线执行（编写→安全审查→性能审查）。

```
                    [Orchestrator]
                   /      |       \
            [Code Author] → [Security Reviewer] → [Perf Reviewer]
                   ↑                                |
                   └──────── 修改请求 ──────────────┘
```

**三个Agent的职责定义：**

| Agent | 输入 | 输出 | 超时 | 降级策略 |
|-------|------|------|------|----------|
| Code Author | PR描述+需求 | 代码diff | 60s | 切换小模型重试 |
| Security Reviewer | 代码diff | 安全问题列表 | 45s | 跳过，标记为"未审查" |
| Perf Reviewer | 代码diff+安全报告 | 最终审查报告 | 45s | 仅输出基础质量检查 |

**通信机制：** Agent之间通过结构化消息传递（JSON Schema），而非自由文本。Orchestrator维护全局上下文，每个Agent的输出经Schema验证后才传递给下游。

**异常处理策略：**

```python
async def code_review_pipeline(pr_info):
    """代码审查流水线，带完整异常处理"""
    context = {"pr": pr_info, "reviews": {}}
    
    try:
        # Step 1: 代码生成
        code = await isolated_agent_call(
            code_author, pr_info, timeout=60
        )
        if code["status"] != "success":
            return degraded_response("代码生成失败")
        context["code"] = code["result"]
        
        # Step 2: 安全审查（可降级）
        security = await isolated_agent_call(
            security_reviewer, code["result"], timeout=45
        )
        if security["status"] != "success":
            logging.warning("安全审查降级，标记为未审查")
            context["security"] = {"status": "skipped"}
        else:
            context["security"] = security["result"]
        
        # Step 3: 性能审查
        perf = await isolated_agent_call(
            performance_reviewer, context, timeout=45
        )
        if perf["status"] != "success":
            return degraded_response("性能审查失败")
        
        # Step 4: 汇总
        return aggregate_reviews(context, perf["result"])
        
    except Exception as e:
        logging.error(f"审查流水线异常: {e}")
        return degraded_response("系统异常，已转交人工审查")
```

**面试加分点：**
- 提到用JSON Schema约束Agent间的通信格式
- 提到为每个Agent设置独立的token预算
- 提到用LangSmith/LangFuse做调用链追踪
- 提到A/B测试不同Agent配置的审查效果
- 提到引入"置信度评分"，低置信度结果自动触发二次审查

---

## 总结

多Agent协同不是简单地"多个LLM并行调用"，而是需要精心设计的架构决策：

1. **拓扑选择决定协作效率**——根据任务特性选择星型、网状、层级或环形
2. **框架选型各有侧重**——AutoGen适合对话式协作，CrewAI适合角色化团队，LangGraph适合复杂状态机
3. **任务分解是核心**——递归分解、MapReduce、投票决策各有适用场景
4. **冲突解决保证质量**——优先级仲裁、投票、Leader裁决构建决策闭环
5. **生产化是终极考验**——隔离、超时、降级、监控缺一不可

当我们将AI Team的自主协作能力与工程化的生产保障结合起来，就能构建出真正可靠的多Agent系统——这不是科幻，而是正在发生的工程实践。
