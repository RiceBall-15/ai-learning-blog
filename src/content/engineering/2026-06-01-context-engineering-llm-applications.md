---
title: "Context Engineering实战：系统化构建LLM应用上下文的艺术"
description: "深入剖析LLM应用中的上下文工程方法论，涵盖上下文设计模式、动态上下文组装、Token预算管理与性能优化，用真实案例展示如何让LLM输出质量提升3倍。"
date: 2026-06-01
author: "RiceBall"
category: "engineering"
subCategory: ai-coding
tags: ["Context Engineering", "LLM应用", "Prompt工程", "RAG", "上下文管理", "Token优化"]
draft: false
---

## 引言：为什么Prompt Engineering不够用了？

2024年，Andrej Karpathy在推特上提出了"Context Engineering"这个概念，很快在AI工程圈引发了广泛讨论。他的核心观点是：

> "当你从简单的提示词调试转向构建复杂的AI系统时，你面对的不再是'写好一个Prompt'的问题，而是'如何系统化地构建和管理LLM的完整上下文'。"

这个转变意味着什么？让我们用一个真实场景来说明。

**场景：客服Agent系统**

```
传统Prompt Engineering思路：
"你是一个客服助手，请根据以下信息回答用户问题：{相关信息}"

Context Engineering思路：
┌─────────────────────────────────────────────┐
│              System Context                   │
│  ├─ 角色定义 + 行为约束                       │
│  ├─ 企业知识库（动态检索，Top-K）              │
│  ├─ 用户画像（历史交互摘要）                   │
│  ├─ 当前会话上下文（滑动窗口）                 │
│  ├─ 工具调用结果（异步获取）                   │
│  ├─ 安全规则 + 合规约束                        │
│  └─ 输出格式规范                              │
└─────────────────────────────────────────────┘
```

Prompt Engineering关注的是"这句话怎么写"，Context Engineering关注的是"这些信息从哪来、怎么组装、什么时候用"。

## 一、Context Engineering的核心框架

### 1.1 三层架构

```
Context Engineering 三层架构
│
├── Layer 1: 信息采集层（Information Gathering）
│   ├── 静态上下文：系统提示词、角色定义、格式规范
│   ├── 动态上下文：RAG检索结果、工具调用输出
│   └── 用户上下文：会话历史、用户画像、偏好设置
│
├── Layer 2: 上下文组装层（Context Assembly）
│   ├── 优先级排序：什么信息最重要
│   ├── Token预算分配：各部分占多少Token
│   ├── 冲突消解：信息矛盾时如何取舍
│   └── 时效性管理：旧信息何时淘汰
│
└── Layer 3: 输出控制层（Output Control）
    ├── 格式约束：JSON、Markdown、纯文本
    ├── 长度控制：输出Token上限
    └── 安全过滤：敏感信息脱敏
```

### 1.2 与传统Prompt Engineering的对比

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|---------------------|
| 关注点 | 单条指令的措辞 | 完整上下文的架构设计 |
| 信息来源 | 人工编写 | 多源动态组装 |
| Token管理 | 粗放（大致估计） | 精细（预算分配） |
| 可维护性 | 硬编码在代码中 | 模块化、可配置 |
| 测试方式 | 人工评估 | 自动化回归测试 |
| 适用场景 | 简单的单轮交互 | 复杂的多轮、多工具系统 |

## 二、上下文设计模式

### 2.1 模式一：分层上下文（Layered Context）

这是最基础也最重要的模式。核心思想是将上下文按优先级分层，确保最重要的信息不被挤出上下文窗口。

```python
class LayeredContext:
    """分层上下文管理器"""
    
    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.layers = []
    
    def add_layer(self, name: str, content: str, priority: int):
        """
        添加上下文层
        priority: 1=最高（必须保留）, 5=最低（优先丢弃）
        """
        self.layers.append({
            'name': name,
            'content': content,
            'priority': priority,
            'tokens': self._count_tokens(content)
        })
        # 按优先级排序（数字越小越重要）
        self.layers.sort(key=lambda x: x['priority'])
    
    def build_context(self) -> str:
        """按优先级组装上下文，确保不超过Token上限"""
        total = 0
        selected = []
        
        for layer in self.layers:
            if total + layer['tokens'] <= self.max_tokens:
                selected.append(layer)
                total += layer['tokens']
            else:
                # 尝试截断低优先级层
                remaining = self.max_tokens - total
                if remaining > 100 and layer['priority'] > 3:
                    truncated = self._truncate(layer['content'], remaining)
                    selected.append({**layer, 'content': truncated})
                break
        
        return '\n\n'.join([
            f"[{l['name']}]\n{l['content']}" 
            for l in selected
        ])
    
    def _count_tokens(self, text: str) -> int:
        # 简化的Token计数（实际应使用tiktoken等）
        return len(text) // 2
    
    def _truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens * 2]  # 简化
```

**使用示例**：

```python
ctx = LayeredContext(max_tokens=8192)

# Layer 1: 系统指令（最高优先级，不可丢弃）
ctx.add_layer('system', SYSTEM_PROMPT, priority=1)

# Layer 2: 安全规则（高优先级）
ctx.add_layer('safety', SAFETY_RULES, priority=2)

# Layer 3: 用户画像（中高优先级）
ctx.add_layer('user_profile', get_user_profile(user_id), priority=3)

# Layer 4: RAG检索结果（中优先级）
ctx.add_layer('knowledge', rag.retrieve(query), priority=3)

# Layer 5: 对话历史（中低优先级）
ctx.add_layer('history', get_recent_history(user_id, limit=10), priority=4)

# Layer 6: 工具输出（低优先级，动态）
ctx.add_layer('tools', tool_results, priority=5)

final_context = ctx.build_context()
```

### 2.2 模式二：滑动窗口 + 摘要（Sliding Window + Summary）

对于长对话场景，简单的截断会丢失关键信息。这个模式通过维护一个"摘要缓冲区"来保留历史要点。

```
对话历史管理策略
│
├── 最近N轮：完整保留（精确上下文）
│   └── 例：最近3轮对话原文
│
├── 历史窗口：LLM摘要（压缩上下文）
│   └── 例：之前对话的关键信息摘要
│
└── 长期记忆：向量检索（按需召回）
    └── 例：用户历史偏好、过往决策记录
```

```python
class ConversationContext:
    """带摘要的对话上下文管理"""
    
    def __init__(self, max_recent: int = 3, max_history_tokens: int = 1000):
        self.recent_messages = []  # 最近N轮完整消息
        self.history_summary = ""  # 历史摘要
        self.max_recent = max_recent
        self.max_history_tokens = max_history_tokens
    
    def add_message(self, role: str, content: str):
        """添加新消息，自动管理上下文"""
        self.recent_messages.append({'role': role, 'content': content})
        
        if len(self.recent_messages) > self.max_recent:
            # 将超出的消息压缩为摘要
            overflow = self.recent_messages[:len(self.recent_messages) - self.max_recent]
            self.recent_messages = self.recent_messages[-self.max_recent:]
            self.history_summary = self._summarize(
                self.history_summary, overflow
            )
    
    def get_context_messages(self) -> list:
        """返回用于LLM调用的消息列表"""
        messages = []
        
        # 添加历史摘要
        if self.history_summary:
            messages.append({
                'role': 'system',
                'content': f'之前对话的要点：\n{self.history_summary}'
            })
        
        # 添加最近对话
        messages.extend(self.recent_messages)
        
        return messages
    
    def _summarize(self, old_summary: str, new_messages: list) -> str:
        """使用LLM将旧摘要和新消息合并"""
        new_text = '\n'.join([
            f"{m['role']}: {m['content']}" for m in new_messages
        ])
        
        prompt = f"""请将以下对话历史摘要合并为简洁的要点列表：

旧摘要：
{old_summary}

新消息：
{new_text}

输出要求：保留所有关键决策、用户偏好和待办事项，控制在200字以内。"""
        
        return call_llm(prompt)
```

### 2.3 模式三：条件上下文（Conditional Context）

不同场景需要不同的上下文组合。条件上下文模式根据运行时条件动态决定包含哪些上下文组件。

```python
class ConditionalContext:
    """根据场景动态组装上下文"""
    
    def __init__(self):
        self.context_builders = {}
    
    def register(self, name: str, builder: Callable, condition: Callable):
        """注册上下文构建器及其启用条件"""
        self.context_builders[name] = {
            'builder': builder,
            'condition': condition
        }
    
    def build(self, session: dict) -> str:
        """根据当前会话条件组装上下文"""
        parts = []
        
        for name, config in self.context_builders.items():
            if config['condition'](session):
                content = config['builder'](session)
                if content:
                    parts.append(f"[{name}]\n{content}")
        
        return '\n\n'.join(parts)

# 使用示例
ctx = ConditionalContext()

# 始终包含系统提示词
ctx.register(
    'system',
    builder=lambda s: SYSTEM_PROMPT,
    condition=lambda s: True
)

# 仅在用户是VIP时包含个性化内容
ctx.register(
    'personalization',
    builder=lambda s: get_personalization(s['user_id']),
    condition=lambda s: s.get('user_tier') == 'vip'
)

# 仅在涉及代码问题时包含代码规范
ctx.register(
    'code_standards',
    builder=lambda s: get_code_standards(s['language']),
    condition=lambda s: '代码' in s.get('query', '') or 'code' in s.get('query', '').lower()
)

# 仅在涉及敏感操作时包含安全规则
ctx.register(
    'security_rules',
    builder=lambda s: get_security_rules(s['operation_type']),
    condition=lambda s: s.get('operation_type') in ['delete', 'modify', 'execute']
)
```

## 三、Token预算管理

### 3.1 预算分配策略

Token预算管理是Context Engineering中最被低估但最重要的技术。不同的上下文组件对输出质量的影响差异巨大。

```
Token预算分配指南（以8K上下文窗口为例）
├── 系统提示词 + 安全规则：5-10%（400-800 tokens）
│   └── 这部分必须完整，不可截断
├── 用户画像/个性化：5-10%（400-800 tokens）
│   └── 精炼的关键信息，不要冗余
├── RAG检索结果：30-40%（2400-3200 tokens）
│   └── 核心信息源，按相关性排序
├── 对话历史：20-30%（1600-2400 tokens）
│   └── 最近完整对话 + 历史摘要
├── 工具输出：10-15%（800-1200 tokens）
│   └── 只保留与当前问题相关的部分
└── 预留缓冲：10%（800 tokens）
    └── 应对不确定性
```

### 3.2 动态预算调整

```python
class TokenBudgetManager:
    """动态Token预算管理器"""
    
    def __init__(self, total_budget: int = 8192):
        self.total_budget = total_budget
        self.allocations = {}
    
    def allocate(self, component: str, ratio: float, min_tokens: int = 100, 
                 max_tokens: int = None):
        """分配预算比例"""
        allocated = int(self.total_budget * ratio)
        if max_tokens:
            allocated = min(allocated, max_tokens)
        allocated = max(allocated, min_tokens)
        
        self.allocations[component] = {
            'budget': allocated,
            'used': 0
        }
    
    def consume(self, component: str, tokens: int) -> dict:
        """消耗预算，返回是否允许以及调整建议"""
        if component not in self.allocations:
            return {'allowed': False, 'reason': '未分配预算的组件'}
        
        alloc = self.allocations[component]
        remaining = alloc['budget'] - alloc['used']
        
        if tokens <= remaining:
            alloc['used'] += tokens
            return {'allowed': True, 'remaining': remaining - tokens}
        
        # 超出预算，需要截断
        truncated_tokens = remaining
        alloc['used'] = alloc['budget']
        return {
            'allowed': 'partial',
            'truncated_to': truncated_tokens,
            'original': tokens
        }
    
    def get_summary(self) -> dict:
        """返回预算使用摘要"""
        summary = {
            'total_budget': self.total_budget,
            'total_used': sum(a['used'] for a in self.allocations.values()),
            'components': {}
        }
        
        for name, alloc in self.allocations.items():
            summary['components'][name] = {
                'budget': alloc['budget'],
                'used': alloc['used'],
                'utilization': f"{alloc['used']/alloc['budget']*100:.1f}%"
            }
        
        return summary
```

### 3.3 Token使用可视化

在实际生产中，我们需要监控Token的使用情况：

```
Token预算使用监控面板
┌─────────────────────────────────────────────┐
│ 总预算: 8192 tokens  |  已使用: 6547 (79.9%) │
├─────────────────────────────────────────────┤
│ 系统提示词  ████████░░░░  680/820  (82.9%)  │
│ 用户画像    ██████░░░░░░  520/820  (63.4%)  │
│ RAG检索    ████████████  3100/3200 (96.9%)  │  ← 告警：接近上限
│ 对话历史    ████████░░░░  1800/2400 (75.0%) │
│ 工具输出    ████░░░░░░░░  447/1200 (37.3%)  │
│ 缓冲区     ░░░░░░░░░░░░  0/800   (0.0%)   │
└─────────────────────────────────────────────┘

⚠️ 告警：RAG检索结果占用率96.9%，建议：
   1. 增加检索结果的相关性阈值
   2. 减少返回的文档数量（Top-K从10降至5）
   3. 对检索结果进行二次摘要压缩
```

## 四、生产级上下文管理

### 4.1 上下文缓存

重复请求相同上下文时，缓存可以显著降低延迟和成本。

```python
import hashlib
from functools import lru_cache

class ContextCache:
    """上下文缓存，支持TTL和智能失效"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _make_key(self, context_parts: dict) -> str:
        """基于上下文内容生成缓存键"""
        content = str(sorted(context_parts.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_or_build(self, context_parts: dict, builder: Callable) -> str:
        """获取缓存或构建新上下文"""
        key = self._make_key(context_parts)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['context']
        
        # 构建新上下文
        context = builder(context_parts)
        self.cache[key] = {
            'context': context,
            'timestamp': time.time()
        }
        
        return context
    
    def invalidate(self, pattern: str = None):
        """使缓存失效"""
        if pattern is None:
            self.cache.clear()
        else:
            self.cache = {
                k: v for k, v in self.cache.items() 
                if pattern not in k
            }
```

### 4.2 上下文版本控制

生产环境中，上下文模板的变更需要版本控制和灰度发布。

```yaml
# context-templates/v1.0/customer-service.yaml
version: "1.0"
components:
  system_prompt:
    template: "templates/system_v1.txt"
    priority: 1
    mutable: false  # 不允许运行时修改
  
  knowledge_base:
    source: "rag"
    top_k: 5
    similarity_threshold: 0.7
    priority: 3
    mutable: true
  
  safety_rules:
    template: "templates/safety_v2.txt"
    priority: 2
    mutable: false

# context-templates/v1.1/customer-service.yaml
version: "1.1"
changes:
  - component: knowledge_base
    change: "top_k: 5 → 10"
    reason: "用户反馈信息不够全面"
    rollout: "10%"  # 灰度10%流量
```

### 4.3 上下文测试

```python
class ContextTestSuite:
    """上下文组装的自动化测试"""
    
    def __init__(self, context_builder):
        self.builder = context_builder
        self.test_cases = []
    
    def add_test(self, name: str, session: dict, assertions: list):
        """添加测试用例"""
        self.test_cases.append({
            'name': name,
            'session': session,
            'assertions': assertions
        })
    
    def run_all(self) -> list:
        """运行所有测试"""
        results = []
        for tc in self.test_cases:
            context = self.builder.build(tc['session'])
            passed = all(a(context) for a in tc['assertions'])
            results.append({
                'name': tc['name'],
                'passed': passed,
                'context_length': len(context)
            })
        return results

# 使用示例
suite = ContextTestSuite(my_context_builder)

# 测试1：基本上下文完整性
suite.add_test(
    '基本上下文包含系统指令',
    session={'user_id': '123', 'query': '你好'},
    assertions=[
        lambda ctx: '系统指令' in ctx,
        lambda ctx: len(ctx) > 100
    ]
)

# 测试2：VIP用户个性化
suite.add_test(
    'VIP用户获得个性化内容',
    session={'user_id': '456', 'user_tier': 'vip', 'query': '推荐产品'},
    assertions=[
        lambda ctx: '个性化推荐' in ctx,
        lambda ctx: 'VIP' in ctx
    ]
)

# 测试3：Token预算不超限
suite.add_test(
    'Token预算控制在限制内',
    session={'user_id': '789', 'query': '详细解释', 'rag_results': 'x' * 5000},
    assertions=[
        lambda ctx: len(ctx) < 8192 * 2  # 简化的Token估算
    ]
)

results = suite.run_all()
for r in results:
    status = '✅' if r['passed'] else '❌'
    print(f"{status} {r['name']} ({r['context_length']} chars)")
```

## 五、实战案例：客服Agent的上下文架构

### 5.1 完整架构图

```
客服Agent上下文架构
│
├── 请求接收
│   └── 用户消息 + 会话ID
│
├── 上下文采集（并行）
│   ├── 用户服务：用户画像、VIP等级、历史工单
│   ├── 知识检索：向量搜索 + BM25混合检索
│   ├── 会话服务：滑动窗口 + 历史摘要
│   └── 工具服务：查询订单状态、库存等
│
├── 上下文组装
│   ├── Token预算：8192 tokens
│   ├── 优先级排序：安全 > 用户信息 > 知识 > 历史
│   └── 冲突消解：新信息覆盖旧信息
│
├── LLM推理
│   ├── 模型：GPT-4o / Claude-3.5-Sonnet
│   ├── 温度：0.3（客服场景偏保守）
│   └── 输出格式：结构化JSON
│
└── 输出处理
    ├── 安全过滤：敏感信息脱敏
    ├── 格式化：转换为前端可渲染格式
    └── 日志记录：上下文快照 + 输出
```

### 5.2 关键指标

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 首次回答准确率 | 62% | 89% | +43.5% |
| 平均对话轮次 | 4.2轮 | 2.1轮 | -50% |
| 上下文Token消耗 | 7800/8192 | 5200/8192 | -33.3% |
| 平均响应延迟 | 2.8s | 1.6s | -42.9% |
| 客户满意度 | 3.2/5 | 4.3/5 | +34.4% |

## 六、常见陷阱与避坑指南

### 6.1 Top-10 上下文工程陷阱

| 排名 | 陷阱 | 后果 | 解决方案 |
|------|------|------|---------|
| 1 | 上下文过长导致"大海捞针" | 关键信息被淹没 | 分层上下文 + 优先级排序 |
| 2 | RAG检索噪声 | 不相关信息干扰输出 | 提高相似度阈值 + 二次重排 |
| 3 | 对话历史无摘要 | Token溢出，丢失早期信息 | 滑动窗口 + 摘要机制 |
| 4 | 硬编码Token限制 | 换模型时上下文失效 | 动态Token预算管理 |
| 5 | 缺少上下文测试 | Prompt变更导致质量回退 | 自动化回归测试 |
| 6 | 信息重复 | 浪费Token，降低输出质量 | 去重 + 压缩 |
| 7 | 时效性忽略 | 过时信息误导LLM | TTL机制 + 时效性标记 |
| 8 | 个人隐私泄露 | 上下文中包含敏感信息 | PII检测 + 动态脱敏 |
| 9 | 工具输出未过滤 | 大量原始数据占用Token | 工具层面的结果裁剪 |
| 10 | 缺少监控告警 | Token使用异常无法及时发现 | 预算使用监控面板 |

### 6.2 调试流程

```
上下文调试流程
│
├── Step 1: 打印完整上下文
│   └── 检查信息是否完整、无噪声
│
├── Step 2: Token使用分析
│   └── 各组件占比是否合理
│
├── Step 3: A/B测试
│   └── 对比不同上下文组装策略的输出质量
│
├── Step 4: 错误案例分析
│   └── 对失败case逐条分析上下文问题
│
└── Step 5: 用户反馈闭环
    └── 收集真实用户反馈，持续优化
```

## 七、工具与生态

### 7.1 上下文管理工具

| 工具 | 类型 | 核心功能 | 适用场景 |
|------|------|---------|---------|
| LangSmith | 全链路 | 上下文追踪、评估 | 开发调试阶段 |
| Langfuse | 开源可观测 | 上下文日志、分析 | 生产监控 |
| PromptLayer | 版本管理 | Prompt版本、A/B测试 | 多版本管理 |
| Anthropic Workbench | 官方工具 | 模型特定优化 | Claude开发者 |
| 自建方案 | 完全定制 | 灵活、可控 | 复杂业务场景 |

### 7.2 技术栈推荐

```
上下文管理技术栈
│
├── Token计数：tiktoken / anthropic-tokenizer
├── 向量检索：Pinecone / Weaviate / pgvector
├── 缓存层：Redis / 本地LRU Cache
├── 监控：Langfuse / 自建Prometheus + Grafana
└── 测试：自建框架 + DeepEval
```

## 八、总结与展望

### 8.1 核心要点

1. **Context Engineering ≠ Prompt Engineering**：前者是系统工程，后者是文案技巧
2. **分层是基础**：所有上下文管理都始于合理的分层和优先级排序
3. **Token预算是核心**：精细化的Token管理直接决定输出质量
4. **测试不可少**：上下文变更需要自动化回归测试
5. **监控是保障**：生产环境必须有Token使用和输出质量的实时监控

### 8.2 未来趋势

- **自适应上下文**：根据查询复杂度动态调整上下文深度
- **多模型上下文**：不同模型使用不同的上下文策略
- **上下文学习**：LLM自主学习最优的上下文使用方式
- **跨模态上下文**：统一管理文本、图像、代码等多模态上下文

Context Engineering是LLM应用从"Demo级"走向"生产级"的关键技术。掌握它，你的AI应用将不再是一个"聪明但不靠谱"的Demo，而是一个"稳定且可预测"的生产系统。
