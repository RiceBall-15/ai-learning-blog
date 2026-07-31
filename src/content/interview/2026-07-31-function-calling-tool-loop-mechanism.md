---
title: "Agent 面试深挖（二）：Function Calling 到底怎么工作——模型如何「选」工具"
description: "模型并不执行工具，它只输出一张结构化的点菜单。本文拆开 Function Calling 的完整请求-响应循环、工具循环的三重收敛条件、并行 tool_calls 的 id 对齐陷阱、工具与参数幻觉的三类处理，以及工具数量爆炸后的 Tool RAG 方案。"
date: 2026-07-31
author: "技术学习笔记"
category: interview
subCategory: tools-protocol
series: agent-interview-deep
seriesOrder: 2
tags: ["Agent", "Function Calling", "工具调用", "MCP", "面试"]
---

# Function Calling 到底怎么工作：模型如何「选」工具

> 上一篇讲了 DAG 编排怎么保证跑得对。这一篇往下一层：Agent = 模型 + 工具，那**模型到底怎么决定调哪个工具、参数从哪来**？
>
> 这是很多人的认知盲区——能背出「幂等」「Outbox」，但被问「模型怎么选工具的」就卡住。不讲清这个循环，后面的幂等和治理都是空中楼阁。

## 一、先破除最大的误解

**模型不会执行工具，它只会「点菜」。**

很多人以为模型直接调了 API。实际上模型只输出一段结构化数据说「我要调 X 工具，参数是 Y」，**真正的执行发生在编排层**。

这个区分不是抠字眼，它直接决定了架构：

> 因为执行在编排层，所以**权限校验、幂等、鉴权、审计、限流全部在编排层做，模型碰不到**。

所以「防止模型被诱导调用高危工具」根本不是模型的活，是编排层的活。这一句答出来就已经和「只接过工具」的候选人分开了。

## 二、完整循环（背下来）

```text
用户：「帮我查下我上个月的订单」
  ↓
① 编排把「工具清单」作为 tools schema 传给模型
     每个工具 = { name, description, parameters(JSON Schema) }
  ↓
② 模型不返回自然语言，而返回结构化 tool_calls
     [{ id, name, arguments(JSON) }]     ← 可能多个 = 并行调用
  ↓
③ 编排解析 tool_calls → schema 校验参数 → 执行（治理/幂等都在这里）
  ↓
④ 每个工具结果作为 role=tool 的消息追加回对话，按 id 对齐
  ↓
⑤ 带着工具结果再请求模型：
     要么继续调工具（进入下一轮）
     要么输出最终答案（finish_reason = stop，结束）
```

### 三个必须点清的认知

**1. 模型只决策，编排才执行。** 见上一节。

**2. `description` 和 `parameters schema` 的质量直接决定选得准不准。**

这是工具类 badcase 的**头号来源**，而且经常被误判成「模型笨」。两个工具描述含糊、语义重叠，模型必然选错。举例：

```text
❌ 坏描述
  queryOrder:  "查询订单"
  refundOrder: "订单相关操作"        ← 语义重叠，模型会乱选

✅ 好描述
  queryOrder:  "按用户ID和时间范围查询历史订单列表。只读，不修改任何数据。
                不适用于：查询单个订单详情（用 getOrderDetail）、发起退款（用 refundOrder）"
  refundOrder: "对指定订单发起退款。会真实扣减商家余额，属于不可逆操作。
                调用前必须已确认订单状态为已支付。"
```

写清「**适用 / 不适用**」边界，比换更大的模型有效得多。

**3. 本质是个循环（tool loop）。**

这就是 ReAct「思考 → 行动 → 观察 → 再思考」的工程化落地，只不过「行动」被结构化成了 `tool_calls`。所以它**可能来回好几轮**——而只要是循环，就必须有终止条件。

## 三、工具循环的终止与失控控制

```text
toolLoop(messages):
  for round in 1..maxRounds:                  // ★ 轮数上限，防不收敛
    resp = model.call(messages, tools)

    if resp.finish == "stop":
      return resp.content                     // 模型给最终答案，结束

    for call in resp.tool_calls:              // 可能多个
      result = executeTool(call)              // 校验 / 治理 / 幂等都在这
      messages.append(toolMessage(call.id, result))   // ★ 按 id 对齐回喂

    // 循环继续，带着工具结果再问模型

  return forceSummarize(messages)             // 到上限强制收尾，绝不无限转
```

**三重收敛条件**（缺一个都可能烧穿预算）：

1. 模型主动 `stop`
2. `maxRounds` 轮数上限
3. token / 成本预算耗尽

再加一道**重复调用早停**：同一工具 + 同一参数连续调用多次，判定为空转，直接停。

配套监测指标：**平均工具调用数**、**平均对话轮数**——超阈值告警。这类指标能抓到「答案对但过程差」的隐性失败（详见系列第八篇）。

## 四、并行 tool_calls 与 id 对齐陷阱

模型一次可能返回多个 `tool_calls`：

```text
无依赖（查天气 + 查汇率）
  → 并行执行（DAG fan-out + bulkhead 分舱，防单工具拖垮全局）

有依赖（先查订单号，再用订单号查物流）
  → 模型会分轮返回：第 1 轮查订单，结果回喂后第 2 轮才查物流
  → 编排不需要自己推依赖，模型的分轮就是依赖表达

结果回喂
  → 严格按 tool_call.id 匹配，顺序无关
```

**这里有个高频 bug**：并行结果回喂时如果**按顺序**而不是按 `id` 对齐，会把 A 工具的结果喂到 B 的位置上，模型彻底混乱，产出看起来「像幻觉」但其实是编排的锅。

> 排查口径：并行工具场景下断言 `id → result` 一一对应。

## 五、工具与参数幻觉的三类处理

| 幻觉类型 | 现象 | 处理 |
|----------|------|------|
| **幻觉工具名** | 模型调了不存在的工具 | 校验工具名是否在清单内；不存在 → **反问模型**（附可用工具列表），不要直接崩 |
| **参数幻觉** | 模型编造不符合 schema 的参数 | schema 校验失败 → **reprompt with error**（把校验错误信息回喂让它改），绝不盲目照调 |
| **该调不调 / 乱调** | 简单问题也去调工具，或该调时不调 | 提升 description 质量 + few-shot 示例 + 必要时用 `tool_choice` 强制或禁止 |

关键原则：**校验失败不是终止，是反馈**。把错误信息回喂给模型让它自我修正，比直接报错给用户体验好得多。

```text
executeTool(call):
  if call.name not in toolRegistry:
    return repromptModel(f"工具 {call.name} 不存在，可用工具：{toolRegistry.names()}")

  if not schema.validate(toolRegistry[call.name].argsSchema, call.arguments):
    return repromptModel(f"参数不符合 schema：{validationError}，请修正后重试")

  // 校验通过，进入治理与执行链路（见系列第三篇）
  return governedExecute(call)
```

## 六、工具数量爆炸 → Tool RAG

这是资深加分点。面试官问「你有几百个工具怎么办」，答得出「工具检索」直接区分「只接过几个工具」和「做过工具平台」。

```text
问题：几百个工具，全部 schema 塞进 context
  → ① 爆 token（工具描述本身很占 context）
  → ② 模型选择质量随工具数增加而下降（选项越多越容易选错）

解法：工具检索（把「记忆召回」的思想迁移到工具选择）
  toolCandidates = toolVectorStore.search(userQuery, topN=10)   // 先检索相关工具
  resp = model.call(messages, tools=toolCandidates)             // 只给这 N 个的 schema

分层策略：
  高频常用工具 → 常驻 context（保证基础能力稳定）
  长尾工具     → 按需检索（省 token、提升选择质量）
```

## 七、工具凭证：绝不进模型上下文

```text
工具要调外部 API，凭证（API key / OAuth token）怎么管？

- 凭证【绝不】进 prompt / 进模型上下文（防泄漏）
- 编排层按 (tenant, tool) 从密钥管理服务取凭证 → 注入工具调用的 header
- 多租户隔离：A 租户的调用绝不能用 B 租户的凭证（scope 强绑定）
- OAuth token 过期 → 自动刷新；刷新失败 → 该工具降级，而不是整个对话失败
```

为什么强调「绝不进上下文」：一旦进了上下文，它就可能被写进日志、被 Trace 快照记录、被模型在后续回答里复述出来——泄漏面瞬间放大。

## 八、Function Calling / MCP / A2A 的层次

这三个经常被混着问，一句话各自定位：

- **Function Calling**：**模型侧**接口——模型决定「选哪个工具」
- **MCP**：**工具侧**协议——工具如何被发现、被调用、被鉴权
- **A2A**：**Agent 侧**协议——Agent 如何作为能力被别的 Agent 调用

三者互补非互斥：**模型用 FC 选工具，执行走 MCP，多 Agent 协作走 A2A**。

## 九、生产踩坑

### 坑 1：模型总选错工具

**现象**：某类意图下模型稳定选错（该查订单却调了退款）。
**排查**：看 `tool_calls` 选的工具与意图不符，且两个工具 description 语义重叠。
**根因**：描述写太泛、边界不清——**不是模型笨**。
**修复**：description 写清适用/不适用 + few-shot + 按场景收窄可用工具集。
**验证**：离线工具选择准确率。

### 坑 2：并行工具结果串号

**现象**：并行调多工具时模型回答逻辑错乱。
**排查**：结果回喂按顺序而非 `tool_call.id` 对齐。
**修复**：严格按 id 匹配。
**验证**：断言 id-result 一一对应。

### 坑 3：工具循环不收敛

**现象**：某类 query 下模型反复调工具、自问自答，5-6 轮不停，成本飙升。
**排查**：无轮数上限、无重复调用检测。
**修复**：`maxRounds` + 重复调用早停 + 平均轮数告警。
**验证**：平均轮数下降、成本回落。

### 坑 4：工具太多，选择质量塌方

**现象**：工具从几十个加到几百个后，选择准确率明显下降。
**根因**：全量 schema 塞 context，选项过载。
**修复**：Tool RAG 按 query 检索 top-N 候选。
**验证**：工具选择准确率随工具总数增长保持稳定。

### 坑 5：凭证串租户 / 泄漏进模型

**现象**：A 租户调用用到了 B 租户凭证，或日志出现明文 key。
**修复**：凭证从不进模型上下文；按 (tenant, tool) scope 强绑定；日志脱敏。
**验证**：凭证审计无跨租户、日志无明文密钥。

## 十、追问与反陷阱

| 追问 | 陷阱 | 口径 |
|------|------|------|
| 「模型到底怎么知道调哪个工具、参数从哪来？」 | 说不清、当黑盒 | 「Function Calling：工具 schema 传给模型，模型输出结构化 `tool_calls`（name + arguments），编排执行后把结果按 id 回喂，模型继续推理——一个『模型↔工具』循环。**模型只决策不执行**；description/schema 质量决定选得准不准。」 |
| 「模型一次要调 3 个工具怎么办？」 | 只会串行 | 「并行 `tool_calls`：无依赖并行执行（fan-out + bulkhead），有依赖模型会分轮给；结果严格按 `tool_call.id` 对齐回喂，顺序无关。」 |
| 「你有几百个工具，怎么让模型选？」 | 全塞 context | 「不能全塞——爆 token 且选项过载导致选择退化。Tool RAG：按 query 检索 top-N 候选，高频常驻 + 长尾检索。这是把记忆召回思想迁移到工具选择。」 |
| 「工具循环会不会无限转？」 | 没有收敛机制 | 「三重收敛：模型 stop / `maxRounds` / 预算耗尽；再加重复调用早停 + 平均轮数告警监测空转。」 |
| 「工具凭证会泄漏给模型吗？」 | 凭证进 prompt | 「绝不进模型上下文。编排按 (tenant,tool) 从密钥服务取、注入 header；多租户 scope 强隔离；OAuth 过期自动刷新，刷新失败该工具降级而非整个对话失败。」 |
| 「工具调用不就是 HTTP 请求包一层嘛」 | 抹掉复杂度 | 「发请求是最后一步。难的是**模型怎么选对工具**（FC 循环 + description 质量）、**多轮怎么收敛**（maxRounds + 空转检测）、**失败怎么不双花**（幂等 + Outbox）、**几百个工具怎么选**（Tool RAG）。」 |

## 十一、一句话速记

> Function Calling 是「模型点菜、编排做菜」：模型只输出结构化 `tool_calls`，执行与治理全在编排层。选得准不准取决于 **description 边界写得清不清**，跑得稳不稳取决于**循环有没有收敛条件**，扩得动扩不动取决于**有没有 Tool RAG**。

---

**上一篇**：DAG 编排与条件汇合 ｜ **下一篇**：工具治理——幂等防双花与人机协同
**系列目录**：[Agent 工程师面试深度系列](/ai-learning-blog/series/agent-interview-deep)
