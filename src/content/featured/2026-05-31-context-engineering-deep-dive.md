---
title: "Context Engineering：大模型应用的隐形核心方法论"
description: "从Prompt Engineering到Context Engineering的范式迁移，深入解析上下文工程的核心理念、技术栈与实战框架，构建可靠的LLM应用系统。"
date: 2026-05-31
author: "RiceBall"
category: "featured"
subCategory: "deep-dive"
tags: ["Context Engineering", "LLM应用", "Prompt Engineering", "RAG", "系统设计"]
draft: false
---

# Context Engineering：大模型应用的隐形核心方法论

> "Prompt Engineering是教你怎么说话，Context Engineering是教你怎么思考。"

2025年下半年以来，AI工程界悄然兴起一个新概念——**Context Engineering（上下文工程）**。这不是又一个营销术语，而是对过去两年LLM应用开发实践的深度总结。Andrej Karpathy、Tobi Lütke等大佬纷纷站台，Tobi甚至直言："我更愿意把我的工作称为Context Engineering而非Prompt Engineering。"

这篇文章将深入拆解Context Engineering的核心理念、技术框架与实战方法论，帮你理解为什么它正在取代Prompt Engineering成为AI应用开发的第一性原理。

---

## 一、为什么Prompt Engineering正在被超越？

### 1.1 Prompt Engineering的局限性

Prompt Engineering关注的是**单次调用的输入优化**——如何设计一个好的提示词让模型给出更好的回答。这在简单场景下足够，但在真实生产系统中面临根本性挑战：

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|-------------------|
| 关注范围 | 单次输入优化 | 全生命周期信息管理 |
| 核心问题 | "怎么问" | "给什么信息" |
| 适用场景 | 单轮对话、简单任务 | 复杂多步工作流 |
| 失败模式 | 回答质量差 | 系统性决策失败 |
| 可维护性 | 硬编码prompt | 动态组装上下文 |
| 系统思维 | 无 | 有 |

### 1.2 一个真实案例：为什么你的RAG系统总是不够好？

假设你在构建一个企业知识问答系统。你的RAG pipeline大概是这样的：

```
用户问题 → 向量检索 → Top-K文档 → Prompt拼接 → LLM回答
```

你花了大量时间调Prompt："请根据以下参考资料回答用户问题，如果不确定请说不知道..."但效果始终不稳定。

**问题出在哪里？**

- 有时候检索到的文档和问题根本不相关，但你没做质量过滤
- 有时候上下文窗口被无关信息填满，关键信息被挤出
- 有时候系统提示词里塞了太多规则，模型"选择性失明"
- 有时候多轮对话的历史记录互相矛盾，模型被搞糊涂

这些问题的共同根源是：**你只在优化"怎么问"，而没有系统性地管理"给什么信息"。**

这就是Context Engineering要解决的问题。

---

## 二、Context Engineering的核心定义

### 2.1 什么是Context Engineering？

**Context Engineering是一门系统性地收集、组织、压缩和管理发送给LLM的上下文信息的工程学科。**

它的核心假设是：**LLM的输出质量几乎完全取决于输入上下文的质量和相关性。** 模型本身的能力已经很强了（GPT-4o、Claude 4等），瓶颈在于我们能否给它提供正确、完整、精炼的上下文。

用一个公式来表达：

```
应用效果 = f(模型能力, 上下文质量)
```

在模型能力趋同的今天，**上下文质量成了区分优秀AI应用和普通AI应用的决定性因素。**

### 2.2 Context Engineering的五大支柱

```
┌─────────────────────────────────────────────────┐
│            Context Engineering                  │
├──────────┬──────────┬──────────┬────────┬───────┤
│ 上下文   │ 信息检索 │ 动态压缩 │ 状态管理│质量   │
│ 选择     │ 与路由   │ 与裁剪   │ 与持久化│保障   │
└──────────┴──────────┴──────────┴────────┴───────┘
```

#### 支柱一：上下文选择（Context Selection）

**核心问题：在众多可用信息中，选择哪些放入上下文窗口？**

这不是简单的"越多越好"。研究表明，当上下文中包含不相关信息时，模型的推理能力会显著下降（即"注意力稀释"效应）。

**关键策略：**
- **相关性评分**：对候选信息进行相关性打分，设置动态阈值
- **类型分层**：区分系统指令、用户查询、检索结果、历史对话等不同类型
- **容量规划**：基于任务复杂度动态分配上下文窗口配额

```python
class ContextSelector:
    """上下文选择器 - 决定什么信息应该进入LLM"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.budget = {
            "system": 0.15,      # 15% 给系统提示
            "tools": 0.10,       # 10% 给工具描述
            "history": 0.25,     # 25% 给对话历史
            "retrieval": 0.40,   # 40% 给检索结果
            "scratchpad": 0.10   # 10% 给思考空间
        }
    
    def select(self, candidates: list[ContextItem]) -> list[ContextItem]:
        """根据预算和相关性选择上下文"""
        selected = []
        remaining_budget = self.max_tokens
        
        # 按类型分配预算
        for item in sorted(candidates, key=lambda x: x.relevance_score, reverse=True):
            category = item.category
            budget_for_category = self.max_tokens * self.budget.get(category, 0.1)
            
            if item.tokens <= remaining_budget and item.tokens <= budget_for_category:
                selected.append(item)
                remaining_budget -= item.tokens
        
        return selected
```

#### 支柱二：信息检索与路由（Retrieval & Routing）

**核心问题：如何从数据源中高效获取相关信息？**

现代AI应用的信息源是多样的：向量数据库、关系型数据库、API接口、文件系统、实时事件流...需要一个智能的路由层来决定查什么、去哪里查。

```python
class InformationRouter:
    """信息路由器 - 根据查询意图选择数据源"""
    
    def __init__(self):
        self.routes = {
            "knowledge_base": VectorStoreRetriever(),
            "api_data": APIRetriever(),
            "conversation_history": HistoryRetriever(),
            "real_time": StreamRetriever(),
        }
    
    async def route(self, query: QueryContext) -> list[ContextItem]:
        """根据查询分析结果路由到不同数据源"""
        # 1. 分析查询意图，确定需要哪些数据源
        intent = await self.analyze_intent(query)
        
        # 2. 并行查询多个数据源
        tasks = []
        for source in intent.required_sources:
            tasks.append(self.routes[source].retrieve(query))
        
        results = await asyncio.gather(*tasks)
        
        # 3. 合并、去重、排序
        return self.merge_and_rank(results)
```

**实际案例 - 智能客服系统的信息路由：**

```
用户问题: "我上周买的那件红色卫衣怎么还没发货？"

路由分析:
├── 查询类型: 订单状态查询
├── 需要的信息:
│   ├── [必查] 用户订单数据 → 订单数据库
│   ├── [必查] 物流信息 → 物流API
│   ├── [可选] 相似问题 → 知识库
│   └── [可选] 对话历史 → 会话存储
├── 优先级: 订单 > 物流 > 历史 > 知识库
└── 时间约束: 500ms内完成
```

#### 支柱三：动态压缩与裁剪（Dynamic Compression）

**核心问题：当信息量超出上下文窗口时，如何智能压缩？**

上下文窗口是有限的（即使是200K token的Claude，在处理复杂任务时也可能不够用），而且研究表明过长的上下文会导致模型性能下降（"Lost in the Middle"问题）。

**三种压缩策略：**

| 策略 | 适用场景 | 实现方式 | 信息损失 |
|------|---------|---------|---------|
| 提取式压缩 | 文档检索结果 | 关键句提取、段落摘要 | 低 |
| 抽象式压缩 | 多轮对话历史 | LLM摘要、意图提取 | 中 |
| 结构化压缩 | 工具调用结果 | 字段筛选、格式转换 | 低 |

```python
class ContextCompressor:
    """上下文压缩器 - 三种策略组合使用"""
    
    async def compress(self, context: list[ContextItem], 
                       max_tokens: int) -> list[ContextItem]:
        """分级压缩策略"""
        total = self.count_tokens(context)
        
        if total <= max_tokens:
            return context  # 无需压缩
        
        # Level 1: 结构化压缩（低损失）
        context = self结构性压缩(context)
        total = self.count_tokens(context)
        if total <= max_tokens:
            return context
        
        # Level 2: 提取式压缩（中损失）
        context = await self提取式压缩(context)
        total = self.count_tokens(context)
        if total <= max_tokens:
            return context
        
        # Level 3: 抽象式压缩（高损失，最后手段）
        context = await self抽象式压缩(context, max_tokens)
        return context
    
    def 结构性压缩(self, context: list[ContextItem]) -> list[ContextItem]:
        """去除冗余字段，保留核心信息"""
        compressed = []
        for item in context:
            if item.type == "tool_result":
                # 只保留返回值的关键字段
                item.data = self.extract_key_fields(item.data)
            elif item.type == "document":
                # 只保留相关段落
                item.data = self.extract_relevant_paragraphs(item.data)
            compressed.append(item)
        return compressed
```

#### 支柱四：状态管理与持久化（State Management）

**核心问题：如何在多轮交互中维护一致、连贯的上下文？**

这是Context Engineering中最容易被忽视的部分。一个复杂的AI Agent可能有：

- 对话历史（短期记忆）
- 用户画像（长期记忆）
- 任务状态（工作记忆）
- 工具调用历史（执行记忆）
- 系统配置（永久记忆）

这些状态需要被精心管理，既不能遗漏关键信息，也不能无限膨胀。

```python
class ContextStateManager:
    """上下文状态管理器"""
    
    def __init__(self):
        self.layers = {
            # 永久记忆：系统指令、工具定义
            "permanent": PersistentContext(),
            
            # 长期记忆：用户偏好、历史摘要
            "long_term": LongTermMemory(),
            
            # 工作记忆：当前任务状态、中间结果
            "working": WorkingMemory(),
            
            # 短期记忆：最近几轮对话
            "short_term": SlidingWindowMemory(max_turns=10),
        }
    
    def build_context(self, user_message: str) -> FullContext:
        """分层构建上下文"""
        context = FullContext()
        
        # 从最稳定到最易变的顺序组装
        context.add(self.layers["permanent"].get())
        context.add(self.layers["long_term"].get())
        context.add(self.layers["working"].get())
        context.add(self.layers["short_term"].get())
        context.add(user_message)
        
        return context
    
    def after_llm_call(self, llm_output: LLMOutput):
        """LLM调用后更新状态"""
        # 更新短期记忆
        self.layers["short_term"].add(llm_output)
        
        # 如果有任务进展，更新工作记忆
        if llm_output.has_task_update():
            self.layers["working"].update(llm_output.task_state)
        
        # 定期压缩对话历史到长期记忆
        if self.layers["short_term"].needs_compression():
            summary = self.compress_history()
            self.layers["long_term"].update(summary)
```

#### 支柱五：质量保障（Quality Assurance）

**核心问题：如何验证上下文质量，确保LLM收到的是可靠的信息？**

在生产环境中，你需要一个"上下文质检"层，确保：

- 检索到的信息确实是相关的（避免幻觉传播）
- 信息是最新的（避免过时数据误导）
- 不存在矛盾信息（避免模型混淆）
- 格式符合模型的输入要求

---

## 三、Context Engineering实战框架

### 3.1 上下文组装Pipeline

一个完整的Context Engineering Pipeline包含以下阶段：

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 意图分析 │───→│ 信息检索 │───→│ 上下文   │───→│ 质量检查 │───→│ 最终组装 │
│ & 路由   │    │ & 聚合   │    │ 压缩     │    │ & 清洗   │    │ & 发送   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
  确定需要      从多个源      压缩到窗口    过滤无关/     构建最终
  什么信息      获取数据      预算内        矛盾内容      prompt
```

### 3.2 实战：构建一个智能代码助手的Context Pipeline

```python
class CodeAssistantContext:
    """代码助手的上下文构建器"""
    
    async def build_context(self, request: CodeRequest) -> FullContext:
        """为代码任务构建完整上下文"""
        
        # 阶段1: 意图分析
        intent = await self.analyze_code_intent(request)
        # intent: {type: "bug_fix"|"new_feature"|"refactor", 
        #          language: "python", complexity: "medium"}
        
        # 阶段2: 多源信息检索
        context_items = []
        
        # 检索1: 项目代码上下文
        relevant_files = await self.retrieve_project_context(
            request, max_files=5, max_tokens=3000
        )
        context_items.extend(relevant_files)
        
        # 检索2: 相关文档和规范
        docs = await self.retrieve_documentation(
            intent.language, intent.type, max_tokens=1000
        )
        context_items.extend(docs)
        
        # 检索3: 类似问题的解决方案
        similar = await self.retrieve_similar_issues(
            request.description, max_tokens=1500
        )
        context_items.extend(similar)
        
        # 检索4: 代码规范和最佳实践
        standards = await self.get_coding_standards(
            intent.language, max_tokens=500
        )
        context_items.extend(standards)
        
        # 阶段3: 压缩和优化
        context_items = await self.compress_context(
            context_items, 
            max_tokens=6000  # 留空间给系统提示和输出
        )
        
        # 阶段4: 质量检查
        context_items = self.quality_check(context_items)
        
        # 阶段5: 组装最终上下文
        return FullContext(
            system=self.build_system_prompt(intent),
            tools=self.get_available_tools(intent),
            context=context_items,
            history=self.get_relevant_history(),
            user=request.to_message()
        )
```

### 3.3 上下文预算分配策略

不同任务类型需要不同的预算分配：

```
┌────────────────────────────────────────────────────────────┐
│              上下文预算分配策略（8K token窗口）              │
├────────────┬────────┬─────────┬────────┬─────────┬────────┤
│ 任务类型   │ 系统   │ 工具    │ 历史   │ 检索    │ 思考   │
├────────────┼────────┼─────────┼────────┼─────────┼────────┤
│ 简单问答   │ 500    │ -       │ 2000   │ 4000    │ 1500   │
│ 代码生成   │ 800    │ 1000    │ 1500   │ 3500    │ 1200   │
│ 多步推理   │ 1000   │ 2000    │ 1000   │ 2500    │ 1500   │
│ 文档分析   │ 500    │ 500     │ 500    │ 5500    │ 1000   │
│ 创意写作   │ 800    │ -       │ 3000   │ 3000    │ 1200   │
└────────────┴────────┴─────────┴────────┴─────────┴────────┘
```

---

## 四、Context Engineering的反模式

### 4.1 反模式一：信息堆积症

**症状：** 把所有能找到的信息都塞进上下文，认为"宁多勿少"。

**后果：**
- 模型注意力被稀释，关键信息被忽略
- Token消耗飙升，成本不可控
- 响应延迟增加，用户体验下降

**正确做法：** 每次添加信息前问自己："这条信息对于回答当前问题是否必要？"

### 4.2 反模式二：静态Prompt依赖症

**症状：** 一个Prompt模板用到底，不根据任务类型和信息量动态调整。

**后果：**
- 简单问题用了复杂的Prompt，浪费token
- 复杂任务的Prompt不够详细，输出质量差

**正确做法：** 建立任务分级体系，不同任务使用不同的上下文模板。

### 4.3 反模式三：忽视上下文污染

**症状：** 检索到的信息质量参差不齐，包含过时、矛盾或无关的内容。

**后果：**
- 模型被错误信息误导，输出质量不可预测
- 用户对系统失去信任

**正确做法：** 建立信息质量评分和过滤机制，在送入模型前进行"上下文清洗"。

```
上下文清洗流程:
原始检索结果 (20条)
  ↓ 相关性过滤 (保留 > 0.7 相关性分数的)
候选上下文 (12条)
  ↓ 时效性过滤 (保留 30天内的)
新鲜上下文 (9条)
  ↓ 矛盾检测 (标记并解决矛盾)
干净上下文 (8条)
  ↓ 去重和压缩
最终上下文 (6条，放入模型)
```

### 4.4 反模式四：黑盒上下文构建

**症状：** 上下文组装逻辑是一个不透明的黑盒，出了问题无法调试。

**后果：**
- 模型输出差时，无法定位是哪个环节的问题
- 无法进行A/B测试和迭代优化

**正确做法：** 为上下文构建的每个阶段添加日志和可观测性。

```python
class ObservableContextBuilder:
    """可观测的上下文构建器"""
    
    async def build(self, query: str) -> FullContext:
        trace = ContextTrace()
        
        # 每一步都记录决策过程
        with trace.span("intent_analysis"):
            intent = await self.analyze(query)
            trace.record("intent", intent)
        
        with trace.span("retrieval"):
            results = await self.retrieve(query, intent)
            trace.record("sources", [r.source for r in results])
            trace.record("count", len(results))
        
        with trace.span("compression"):
            before = self.count_tokens(results)
            compressed = await self.compress(results)
            after = self.count_tokens(compressed)
            trace.record("compression_ratio", after / before)
        
        with trace.span("quality_check"):
            issues = self.quality_check(compressed)
            trace.record("issues", issues)
        
        # 保存trace用于后续分析
        await self.save_trace(trace)
        
        return self.assemble(compressed)
```

---

## 五、Context Engineering与现有技术的关系

### 5.1 Context Engineering ≠ Prompt Engineering

Prompt Engineering是Context Engineering的**子集**。Context Engineering关注的是整个信息管理生命周期，而Prompt Engineering只关注最终组装出来的那段文字。

```
Context Engineering (大)
├── 数据源管理
├── 检索策略
├── 信息过滤与排序
├── 动态压缩
├── 状态管理
├── 质量保障
└── Prompt Engineering (小)
    ├── 指令设计
    ├── 格式规范
    └── 示例选择
```

### 5.2 Context Engineering 与 RAG 的关系

RAG（检索增强生成）是Context Engineering的一个**具体实现**。传统的RAG只解决了"从知识库检索相关信息"这一个问题，而Context Engineering把这个问题放在了更宏观的框架下：

- RAG解决的是**外部知识检索**这一个数据源
- Context Engineering需要同时管理**多种数据源**（知识库、API、历史、工具结果等）
- Context Engineering还需要处理RAG解决不了的**信息压缩、质量控制和预算分配**问题

### 5.3 Context Engineering 与 Agent 的关系

AI Agent是Context Engineering最重要的**应用场景**。Agent的每次决策都依赖于上下文：

- **规划阶段**：需要足够的任务描述、约束条件和可用工具信息
- **执行阶段**：需要准确的环境状态、工具结果和历史操作
- **反思阶段**：需要完整的执行轨迹和评价标准

没有好的Context Engineering，Agent就会变成一个"失忆的决策者"——每次做决策时都缺少关键信息。

---

## 六、落地建议与Checklist

### 6.1 Context Engineering成熟度模型

| 级别 | 描述 | 典型做法 |
|------|------|---------|
| L0 | 无上下文管理 | 直接把所有信息拼接到prompt |
| L1 | 基础RAG | 向量检索 + 固定模板 |
| L2 | 动态检索 | 根据查询意图调整检索策略 |
| L3 | 分层管理 | 区分不同类型上下文，独立管理 |
| L4 | 自适应组装 | 动态预算分配 + 质量监控 |
| L5 | 端到端优化 | 上下文构建与模型推理联合优化 |

### 6.2 实施Checklist

```
□ 1. 盘点你的信息源
  - 内部知识库
  - 外部API
  - 对话历史
  - 工具定义和结果
  - 用户画像
  - 系统状态

□ 2. 建立上下文预算模型
  - 定义每个上下文类型的token配额
  - 基于任务类型动态调整配额
  - 设置总预算上限

□ 3. 实现分层检索策略
  - 关键信息（必查）：实时获取，高质量保证
  - 补充信息（可选）：按需获取，允许延迟
  - 背景信息（低优先级）：缓存获取，允许过时

□ 4. 添加压缩和裁剪机制
  - 识别压缩触发条件（token数超阈值）
  - 实现分级压缩策略
  - 记录压缩损失用于调优

□ 5. 建立质量保障体系
  - 相关性评分
  - 时效性检查
  - 矛盾检测
  - 格式验证

□ 6. 建设可观测性
  - 每次上下文构建的trace
  - 各阶段的性能指标
  - A/B测试框架
```

---

## 七、总结

Context Engineering不是又一个空洞的概念，而是AI应用开发从"手工调参"走向"系统工程"的必然产物。它要求我们：

1. **从单次优化转向系统思维**：不是调好一个prompt就完了，而是管理整个信息流
2. **从静态模板转向动态组装**：根据任务类型、信息量和质量实时调整上下文
3. **从经验驱动转向数据驱动**：通过可观测性和A/B测试持续优化上下文策略

当你开始用Context Engineering的视角审视你的LLM应用时，你会发现问题的答案往往不在模型层，而在上下文层。**给模型正确的信息，比教模型正确的技巧，重要得多。**

---

*本文是Context Engineering的入门指南。后续文章将深入探讨具体的Context Engineering实现框架（如Anthropic的Claude Code上下文管理、OpenAI的Agents SDK等），敬请关注。*
