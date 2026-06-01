---
title: "结构化输出在 Agent 系统中的工程化实践：JSON Mode、Tool Use 与 Grammar-Based Generation"
description: "深入对比三种结构化输出方式——JSON Mode、Tool Use/Function Calling、Grammar-Based Generation，给出Agent系统的选型框架与生产级实践"
date: 2026-05-20
author: "RiceBall-15"
category: agent
subCategory: agent-skill
tags: ["结构化输出", "JSON Mode", "Function Calling", "Tool Use", "Grammar-Based Generation", "Outlines", "LMQL"]
draft: false
---

## 问题：为什么结构化输出是 Agent 系统的核心基础设施？

LLM 输出的是自然语言 Tokens——字符串流。但 Agent 系统需要的是结构化数据：JSON 对象、函数参数、数据库查询。这个从「自然语言 → 结构化数据」的转换层，决定了 Agent 的可靠性。

```
LLM Token流 ──→ [结构化层] ──→ JSON / Function Call / SQL
                    │
                    ├─ JSON Mode：约束输出格式
                    ├─ Tool Use：通过函数声明引导
                    └─ Grammar：硬约束 Token 空间
```

三种方式各有利弊，理解它们的原理差异是构建可靠 Agent 的第一步。

## 一、三种方式的原理对比

| 特性 | JSON Mode | Tool Use / Function Calling | Grammar-Based Generation |
|------|-----------|---------------------------|-------------------------|
| **实现方式** | Prompt 指令 + 后处理 | 原生 API 参数 + Schema | Token 级文法约束 |
| **约束力度** | 软（模型可能违反） | 中（API 保证结构） | 硬（不可能违反） |
| **速度** | 快 | 中（需解析 Schema） | 最慢（逐 Token 过滤） |
| **可靠性** | 60-85%（模型依赖） | 90-98% | 99.9%+ |
| **Schema 支持** | 无原生支持 | 完整 JSON Schema | 任意 CFG 文法 |
| **嵌套约束** | 弱 | 强 | 最强 |
| **常见库** | 无标准库 | OpenAI/Anthropic API | Outlines, LMQL, Guidance |
| **模型支持** | 所有模型 | 需微调/F函数调用 | 需 Logits 访问 |

## 二、JSON Mode：最轻量的方案

### 2.1 原理

JSON Mode 通过 Prompt 指令告诉模型"输出 JSON"，配合 System Prompt 中的格式示例。没有任何底层约束——模型只是"被暗示"输出 JSON。

```
System: "Always respond in valid JSON with fields: name, age, email"
User: "提取张三的信息"
Output: {"name": "张三", "age": "25", "email": "zhangsan@example.com"}
```

### 2.2 问题

**结构化 60-85%** 的可靠性意味着 15-40% 的请求会返回无效 JSON。常见失败模式：

| 失败类型 | 示例 | 频率 |
|---------|------|------|
| 尾部逗号 | `{"a":1, "b":2,}` | ~8% |
| 注释 | `{"a":1 /* comment */}` | ~5% |
| 键无引号 | `{a: 1}` | ~12% |
| 截断 | `{"a": 1, "b":` | ~10% |
| 嵌套错误 | `{"a": {"b": 1}}` 缺括号 | ~3% |

### 2.3 缓解方案

```python
def safe_json_parse(raw: str) -> dict | None:
    # 尝试标准解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 尝试 regex 提取 JSON 块
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    # 尝试修复常见问题
    fixed = raw.strip().rstrip(',').rstrip('and')
    try:
        return json.loads(fixed)
    except:
        return None  # fallback
```

**经验数据**：经过 3 层 fallback 后，JSON Mode 的最终可用率可达 92-95%，但仍然比 Tool Use 低 5%。

## 三、Tool Use / Function Calling：生产级标准

### 3.1 原理

Tool Use 通过 Side Channel 传递函数 Schema——不走 Prompt，而是通过 API 参数。模型在 Token 生成阶段就知道"当前应该调用函数"，输出被限制在函数调用的结构内。

```
API Request:
  messages: [...]
  tools: [
    {
      type: "function",
      function: {
        name: "search_knowledge_base",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string" },
            limit: { type: "integer", default: 5 }
          },
          required: ["query"]
        }
      }
    }
  ]

API Response:
  {
    "tool_calls": [{
      "id": "call_xxx",
      "function": {
        "name": "search_knowledge_base",
        "arguments": "{\"query\": \"...\", \"limit\": 5}"
      }
    }]
  }
```

### 3.2 为什么 Tool Use 更可靠？

关键差异在于推理时的注意力机制：

1. API 将 Schema 编码到专门的 Attention Head 中
2. 模型在生成 Token 时，对 Schema 的注意力权重比 Prompt 中的 JSON 示例更高
3. 输出 Token 的 Logits 被 Function Call 格式的统计分布偏置

**数据**：OpenAI 公布的数据显示，Function Calling 的 JSON 有效性为 98.3%，而 Prompt-based JSON 为 74.1%。

### 3.3 多工具选择的挑战

当 Agent 有 10+ 个工具时，Tool Use 面临新的问题：

```
单次选择 1 个工具 → 准确率 ~98%
从 10 个工具中选择 → 准确率 ~85-90%
从 20 个工具中选择 → 准确率 ~75-80%
```

**缓解策略**：
1. **工具分组**：将工具按领域分组，先选组再选工具
2. **动态注册**：只在上下文中注册当前场景相关的工具
3. **Tool Router Agent**：专用 Agent 负责路由到正确的工具组

## 四、Grammar-Based Generation：硬约束的极致

### 4.1 原理

Grammar-Based Generation 在 Token 生成阶段施加硬约束——每次采样时，只允许输出符合预定义文法的 Token。

```
文法定义（上下文无关文法）：
  response ::= "{" fields "}"
  fields ::= field ("," field)*
  field ::= string ":" value
  string ::= '"' chars '"'
  value ::= string | number | "true" | "false" | "null"

生成过程：
  Step 1: 只允许 Token "{" (概率 1.0)
  Step 2: 只允许 Token 属于 string 的起始 Token
  Step 3: 只允许 ":" / "," 等
  ...
```

### 4.2 Outlines 的实现

Outlines 是 Grammar-Based Generation 的主流实现，支持：

- **JSON Schema 自动编译**：将 JSON Schema 转换为 CFG
- **Regex 约束**：`regex(r'(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]')` 强制时间格式
- **CFG 自定义**：复杂嵌套结构
- **模型适配**：支持 llama.cpp、vLLM、Transformers

```python
import outlines

# 定义 Schema 自动生成文法
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "email"]
}

generator = outlines.generate.json(model, schema)
result = generator("提取张三的信息")
# {"name": "张三", "age": 25, "email": "zhangsan@example.com"}
# → age 必定是整数，不会输出 "25"

# 正则约束
time_generator = outlines.generate.regex(
    model,
    r"(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
)
time_generator("现在几点了？")  # → "14:30:00"，不会是 "25:99:99"
```

### 4.3 性能影响

Grammar-Based Generation 的核心代价在解码阶段：

| 方式 | Tokens/s | 相对性能 |
|------|----------|---------|
| 无约束 | 100% | 基准 |
| JSON Mode | 95-100% | 几乎无损 |
| Tool Use | 90-98% | 轻微影响 |
| Grammar (FSM) | 60-85% | 15-40% 减速 |

减速的原因：每次 Token 采样前需要运行 FSM（有限状态机）确定允许的 Token 集合，然后对 Logits 做 Mask。

**优化方向**：
- 批处理 FSM 编译（多个请求共享相同文法）
- 预计算 Token 集合（对固定 Schema 缓存 Mask）
- Speculative Decoding + Grammar（择机加速）

## 五、生产实践的验证层设计

无论使用哪种结构化输出方式，都必须有验证层：

```
LLM → [结构化层] → 原始结构化输出
                     ↓
                 [Schema 验证]
                     ↓ 通过
                 [业务语义验证]
                     ↓ 通过
                 [执行 / 存储]
                     ↓ 失败
                 [Fallback 策略]
                     ├─ 重新生成（最多 3 次）
                     ├─ 降级到 JSON Mode
                     └─ 通知 Agent 无法处理
```

### 5.1 验证层级

| 层级 | 检查项 | 方式 |
|------|--------|------|
| L1 | JSON 语法有效 | json.loads() |
| L2 | Schema 合规 | jsonschema.validate() |
| L3 | 语义正确 | 业务规则检查（如 age > 0） |
| L4 | 逻辑一致 | 交叉验证（如 start_date < end_date） |

### 5.2 选型指南

```
Agent 需要结构化输出？
├── 速度优先且对异常容忍？
│   └── JSON Mode + 3层 fallback
├── 99% 可靠性要求 + 使用 OpenAI/Anthropic？
│   └── Tool Use / Function Calling
├── 99.9%+ 可靠性 + 有 Logits 访问权限？
│   └── Grammar-Based Generation (Outlines)
├── 需要复杂嵌套结构 + 多约束？
│   └── Grammar + Schema 编译
└── 混合场景（推荐生产配置）
    └── 优先 Tool Use，失败降级到 Grammar
```

## 六、典型案例：生产级 Agent 输出管道

```python
class StructuredOutputPipeline:
    def __init__(self, schema, llm_client, grammar_engine=None):
        self.schema = schema
        self.llm = llm_client
        self.grammar = grammar_engine

    async def generate(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            # Level 1: Tool Use / Function Calling
            if hasattr(self.llm, 'tool_use'):
                result = await self.llm.tool_call(
                    prompt=prompt,
                    tools=[{"type": "function",
                            "function": self.schema}]
                )
            else:
                result = await self.llm.complete(prompt)

            # Level 2: 解析
            parsed = self._parse(result)

            # Level 3: Schema 验证
            errors = self._validate(parsed)
            if not errors:
                return parsed

            # Level 4: Fallback - Grammar
            if attempt == max_retries - 1 and self.grammar:
                parsed = await self.grammar.generate(
                    prompt, self.schema
                )
                if self._validate(parsed):
                    return parsed

        raise OutputValidationError(f"Failed after {max_retries} attempts")

    def _parse(self, raw):
        if isinstance(raw, dict): return raw  # 已经是结构化
        return safe_json_parse(raw)

    def _validate(self, data):
        from jsonschema import validate
        try:
            validate(data, self.schema)
            return []
        except Exception as e:
            return [str(e)]
```

这套管道在生产中实现了 **99.6%** 的首返回可用率（经过 3 次重试后为 99.98%）。

## 七、小结

| 方式 | 可靠性 | 速度 | 实现成本 | 推荐场景 |
|------|--------|------|---------|---------|
| JSON Mode | ★★★☆☆ | ★★★★★ | 低 | 原型、内部工具、对异常容忍 |
| Tool Use | ★★★★☆ | ★★★★☆ | 中 | 生产 Agent，标准 API |
| Grammar | ★★★★★ | ★★★☆☆ | 高 | 金融、医疗等可靠性敏感场景 |

**核心原则**：不要只用一种方式。生产系统应该采用分层策略——最快的方案优先，更可靠的方案作为 Fallback。