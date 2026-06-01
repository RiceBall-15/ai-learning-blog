---
title: "Quickstart - LangChain快速入门指南"
date: 2026-04-29
category: "ai-engineering"
subCategory: ai-coding
tags: ["LangChain", "AI Agents", "LangGraph", "Python"]
source: "LangChain官方文档"
original_url: "https://docs.langchain.com/oss/python/langchain/quickstart"
---


本文提供了创建AI Agent的完整实践指南，从基础到进阶，涵盖了Agent开发的核心要素。通过简单示例展示如何在几分钟内创建功能完整的AI Agent，适合AI工程师快速入门和参考。

## 一、核心框架对比

LangChain提供两种Agent框架，各有其适用场景：

### LangChain Agents
适用于需要细粒度控制的场景，开发者可以精确控制Agent的每个行为步骤。当需要自定义复杂的工具调用逻辑、实现特殊的决策流程或集成深度定制的系统时，LangChain Agents是理想选择。

### Deep Agents
内置了丰富的功能，包括规划能力、文件系统工具和子Agent等，适合快速开发和原型验证。Deep Agents开箱即用，减少了大量基础功能的开发时间，特别适合需要快速验证想法的场景。

## 二、快速上手实战

### 基础Agent创建

创建一个可以回答问题和调用工具的简单Agent只需几行代码：

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.4",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```

在这个例子中，Agent理解用户询问旧金山的天气，自动调用天气工具并传入城市名称，返回相应的结果。整个过程展示了Agent如何理解意图、选择工具并执行操作的基本流程。

### 工具创建与集成

工具是Agent与外部系统交互的桥梁。使用`@tool`装饰器可以快速创建工具，工具的名称、描述和参数会成为模型提示词的一部分，帮助Agent理解如何正确使用工具。

```python
import urllib.error
import urllib.request
from langchain.tools import tool

@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text
```

这个工具可以从URL获取文档内容，设置了超时和错误处理机制，确保在网络异常时能够优雅地返回错误信息。

## 三、高级配置与优化

### 模型参数配置

模型初始化时可以配置多个参数以适应不同的使用场景：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai:gpt-5.4",
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
)
```

- **temperature**: 控制输出的随机性，值越低输出越确定性
- **timeout**: 设置请求超时时间，避免长时间等待
- **max_tokens**: 限制最大令牌数，控制输出长度和成本

### 内存管理

使用LangGraph的InMemorySaver实现内存管理，让Agent能够记住对话历史：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

在生产环境中，应该使用持久化检查点保存消息历史，确保在Agent重启后仍然能够恢复对话状态。

### 系统提示词设计

系统提示词定义了Agent的角色和行为，应该保持具体和可操作性：

```python
SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""
```

良好的系统提示词应该明确Agent的能力边界、行为规范和交互方式，避免模糊或过于宽泛的描述。

## 四、生产环境最佳实践

### LangSmith追踪

使用LangSmith追踪Agent调用，设置环境变量即可启用：

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="***"
```

LangSmith提供了完整的调用链追踪，包括工具调用、决策过程和输入输出，对于调试和优化Agent行为至关重要。

### 持久化检查点

在InMemorySaver的基础上，生产环境应该使用持久化检查点：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string(":memory:")
```

持久化检查点确保Agent的状态可以跨会话保存，支持多轮对话的上下文维护。

## 五、实战经验总结

### 选择合适的框架

- **快速原型开发**: 优先选择Deep Agents，利用其内置的规划和子Agent能力快速验证想法
- **复杂业务逻辑**: 使用LangChain Agents实现精细控制，确保每个步骤都符合业务需求
- **性能优化**: 对于高频调用的场景，LangChain Agents提供了更多优化空间

### 工具开发要点

1. **清晰的文档**: 工具的docstring应该详细描述功能、参数和返回值
2. **错误处理**: 所有可能失败的操作都应该有适当的错误处理机制
3. **超时控制**: 避免因外部系统响应慢而阻塞Agent的整个执行流程
4. **类型注解**: 使用Python类型注解提高代码可读性和IDE支持

### 测试策略

Agent的测试应该涵盖以下几个方面：

1. **单元测试**: 测试每个工具的独立功能
2. **集成测试**: 测试Agent与外部系统的集成
3. **端到端测试**: 测试完整的使用场景
4. **边界测试**: 测试异常输入和错误情况

## 六、总结与展望

LangChain为AI Agent开发提供了强大的工具和框架，通过合理选择框架、精心设计系统提示词、正确配置模型参数和工具，可以快速构建功能强大的Agent应用。

核心要点：
1. 根据需求选择LangChain Agents或Deep Agents
2. 使用`@tool`装饰器快速创建工具，注重工具文档质量
3. 合理配置模型参数以适应不同场景
4. 生产环境必须使用持久化检查点
5. 启用LangSmith追踪以便调试和优化
6. 系统提示词设计要保持具体和可操作性

通过掌握这些核心概念和最佳实践，开发者可以在LangChain生态系统中高效地构建复杂且可靠的AI应用。
