---
title: 'LangGraph状态机深入：复杂Agent工作流设计模式'
description: '从状态管理到条件路由，全面解析LangGraph构建复杂Agent工作流的核心设计模式'
date: 2026-05-30
author: 'RiceBall-15'
category: 'framework'
subCategory: agent-framework
tags: ['LangGraph', '状态机', 'Agent工作流', '设计模式']
draft: false
---

# LangGraph状态机深入：复杂Agent工作流设计模式

## 引言

LangChain构建单轮问答足够，但当你需要一个**有状态、可中断、可恢复**的复杂Agent工作流时，LangGraph才是正确答案。

LangGraph的核心思想：**将Agent工作流建模为状态图(State Graph)**。节点是处理逻辑，边是状态转移，条件边是动态路由。

本文深入解析LangGraph的核心概念和6种实战设计模式。

---

## §1 LangGraph核心概念

### 1.1 架构全景

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Runtime                     │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  State   │───▶│   Node   │───▶│   Edge   │         │
│  │  (状态)   │    │  (处理)   │    │  (转移)   │         │
│  └──────────┘    └──────────┘    └──────────┘         │
│       │              │               │                  │
│       ▼              ▼               ▼                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ Checkpoint│   │  Worker  │    │ Condition│         │
│  │  (持久化) │   │ (执行器)  │    │  (路由)   │         │
│  └──────────┘    └──────────┘    └──────────┘         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# 1. State：状态定义
class AgentState(TypedDict):
    """Agent状态 - 节点间传递的数据结构"""
    messages: Annotated[List, add_messages]  # 消息历史（自动追加）
    current_step: str                         # 当前步骤
    context: dict                             # 上下文信息
    results: list                             # 中间结果


# 2. Node：处理节点
def research_node(state: AgentState) -> dict:
    """研究节点 - 搜索信息"""
    query = state['messages'][-1].content
    # 执行搜索逻辑
    search_results = search(query)
    return {
        "current_step": "analyze",
        "results": search_results
    }


# 3. Edge：状态转移
def route_after_research(state: AgentState) -> str:
    """条件路由 - 根据研究结果决定下一步"""
    if len(state.get('results', [])) > 5:
        return "analyze"  # 结果足够，进入分析
    else:
        return "research_more"  # 结果不足，继续搜索
```

---

## §2 设计模式

### 2.1 Router模式：动态路由

```python
# 构建路由图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("classify", classify_query)
workflow.add_node("handle_tech", handle_technical)
workflow.add_node("handle_general", handle_general)
workflow.add_node("handle_creative", handle_creative)

# 添加条件边
workflow.add_conditional_edges(
    "classify",
    route_by_type,  # 路由函数
    {
        "technical": "handle_tech",
        "general": "handle_general",
        "creative": "handle_creative",
    }
)

# 所有处理节点汇聚到END
workflow.add_edge("handle_tech", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("handle_creative", END)

# 设置入口
workflow.set_entry_point("classify")

# 编译
app = workflow.compile()
```

### 2.2 Parallel模式：并行处理

```python
import asyncio
from langgraph.graph import StateGraph, END


class ParallelState(TypedDict):
    query: str
    search_results: list
    news_results: list
    db_results: list
    merged_results: list


def search_node(state: ParallelState) -> dict:
    """搜索节点"""
    results = web_search(state['query'])
    return {"search_results": results}


def news_node(state: ParallelState) -> dict:
    """新闻节点"""
    results = news_search(state['query'])
    return {"news_results": results}


def db_node(state: ParallelState) -> dict:
    """数据库节点"""
    results = db_query(state['query'])
    return {"db_results": results}


def merge_node(state: ParallelState) -> dict:
    """合并节点 - 汇总并行结果"""
    all_results = (
        state.get('search_results', []) +
        state.get('news_results', []) +
        state.get('db_results', [])
    )
    # 去重并排序
    merged = deduplicate_and_rank(all_results)
    return {"merged_results": merged}


# 构建并行图
workflow = StateGraph(ParallelState)

workflow.add_node("search", search_node)
workflow.add_node("news", news_node)
workflow.add_node("db", db_node)
workflow.add_node("merge", merge_node)

# 并行执行三个节点
workflow.add_edge("__start__", "search")
workflow.add_edge("__start__", "news")
workflow.add_edge("__start__", "db")

# 三个节点都完成后进入merge
workflow.add_edge("search", "merge")
workflow.add_edge("news", "merge")
workflow.add_edge("db", "merge")

workflow.add_edge("merge", END)

app = workflow.compile()
```

### 2.3 Human-in-the-Loop模式：人工介入

```python
from langgraph.checkpoint.memory import MemorySaver


class ApprovalState(TypedDict):
    proposal: str
    approved: bool
    feedback: str


def generate_proposal(state: ApprovalState) -> dict:
    """生成提案"""
    proposal = llm.generate(
        f"根据以下需求生成方案: {state['messages'][-1].content}"
    )
    return {"proposal": proposal}


def review_node(state: ApprovalState) -> dict:
    """人工审核节点 - 暂停等待人工输入"""
    # 这个节点会暂停执行，等待人工输入
    return {"approved": True, "feedback": ""}


def execute_if_approved(state: ApprovalState) -> str:
    """条件执行 - 根据审核结果决定"""
    if state.get('approved'):
        return "execute"
    else:
        return "revise"


# 构建带人工介入的图
workflow = StateGraph(ApprovalState)

workflow.add_node("generate", generate_proposal)
workflow.add_node("review", review_node)
workflow.add_node("execute", execute_proposal)
workflow.add_node("revise", revise_proposal)

workflow.add_edge("generate", "review")
workflow.add_conditional_edges(
    "review",
    execute_if_approved,
    {"execute": "execute", "revise": "revise"}
)
workflow.add_edge("execute", END)
workflow.add_edge("revise", "generate")  # 修订后重新生成

# 使用MemorySaver支持持久化
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["review"]  # 在review节点前暂停
)

# 运行时
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"proposal": ""}, config)

# 人工审核后恢复
app.update_state(config, {"approved": True})
result = app.invoke(None, config)
```

### 2.4 SubGraph模式：子图复用

```python
def create_search_subgraph() -> StateGraph:
    """创建搜索子图 - 可复用的搜索模块"""
    
    class SearchState(TypedDict):
        query: str
        results: list
        filter_type: str
    
    workflow = StateGraph(SearchState)
    
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("db_search", db_search_node)
    workflow.add_node("filter", filter_results)
    
    workflow.add_conditional_edges(
        "classify",
        lambda s: s.get('filter_type', 'web'),
        {"web": "web_search", "db": "db_search"}
    )
    workflow.add_edge("web_search", "filter")
    workflow.add_edge("db_search", "filter")
    workflow.add_edge("filter", END)
    
    workflow.set_entry_point("classify")
    return workflow.compile()


# 主图中使用子图
main_workflow = StateGraph(MainState)

search_subgraph = create_search_subgraph()

main_workflow.add_node("search", search_subgraph)  # 子图作为节点
main_workflow.add_node("analyze", analyze_node)
main_workflow.add_node("respond", respond_node)

main_workflow.add_edge("search", "analyze")
main_workflow.add_edge("analyze", "respond")
main_workflow.add_edge("respond", END)
```

### 2.5 循环重试模式

```python
class RetryState(TypedDict):
    task: str
    attempts: int
    max_attempts: int
    result: str
    error: str


def attempt_node(state: RetryState) -> dict:
    """执行任务节点"""
    try:
        result = execute_task(state['task'])
        return {"result": result, "error": ""}
    except Exception as e:
        return {"error": str(e)}


def should_retry(state: RetryState) -> str:
    """判断是否需要重试"""
    if state.get('error') and state['attempts'] < state.get('max_attempts', 3):
        return "retry"
    return "done"


workflow = StateGraph(RetryState)

workflow.add_node("attempt", attempt_node)
workflow.add_node("wait", wait_node)  # 等待后重试

workflow.add_edge("attempt", "decide")
workflow.add_conditional_edges(
    "decide",
    should_retry,
    {"retry": "wait", "done": END}
)
workflow.add_edge("wait", "attempt")  # 重试

app = workflow.compile()
```

---

## §3 生产部署

### 3.1 LangServe部署

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="Agent Service")

# 添加LangGraph路由
add_routes(
    app,
    workflow,
    path="/agent",
    enable_feedback_endpoint=True,
    enable_state_endpoint=True,
)

# 启动
# uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3.2 持久化配置

```python
from langgraph.checkpoint.postgres import PostgresSaver


# 生产环境使用PostgreSQL持久化
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agent_db"
)

app = workflow.compile(checkpointer=checkpointer)

# 支持会话恢复
config = {"configurable": {"thread_id": "session-123"}}
# 即使服务重启，也能从上次状态继续
```

---

## §4 总结

| 模式 | 适用场景 | 复杂度 |
|------|----------|--------|
| Router | 动态路由不同处理逻辑 | ⭐⭐ |
| Parallel | 多数据源并行查询 | ⭐⭐⭐ |
| Human-in-the-Loop | 需要人工审核的流程 | ⭐⭐⭐⭐ |
| SubGraph | 复用通用处理模块 | ⭐⭐⭐ |
| 循环重试 | 不稳定任务的容错处理 | ⭐⭐ |

LangGraph的核心价值：**将复杂的Agent逻辑可视化为状态图，便于调试、监控和优化。**

## 参考资料

- LangGraph官方文档：https://langchain-ai.github.io/langgraph/
- LangGraph设计模式：https://langchain-ai.github.io/langgraph/how-tos/
