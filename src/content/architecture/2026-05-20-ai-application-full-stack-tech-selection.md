---
title: "AI 应用全链路技术选型：从模型层到应用层的架构设计框架"
description: "从模型推理层、中间件层到应用层，给出 AI 原生应用的完整技术栈选型框架与架构设计模式"
date: 2026-05-20
author: "RiceBall-15"
category: architecture
subCategory: distributed
tags: ["架构设计", "技术选型", "全链路", "AI应用", "中间件", "系统架构", "框架"]
draft: false
---

## 问题：AI 应用为何往往「选型灾难」？

当前 AI 技术栈碎片化程度堪比 2010 年代的 JavaScript 生态：

```
模型层:       FP16? FP8? AWQ? GGUF? LoRA? DoRA?
推理引擎:     vLLM? SGLang? TGI? Ollama? TensorRT-LLM?
编排框架:     LangChain? LangGraph? DSPy? CrewAI?
向量数据库:   Pinecone? Weaviate? Qdrant? Chroma? Milvus?
应用框架:     FastAPI? Next.js? Gradio? Streamlit?
监控:         LangSmith? Weights & Biases? Arize? PHO?
```

选型灾难的根源：没有一个框架能覆盖全链路，每层都有多个竞品，且各层的选择互相耦合。

## 一、AI 应用的分层架构

```
┌─────────────────────────────────────┐
│         应用层 (Application)         │
│  API Gateway、Web UI、业务流程编排    │
│  Next.js / FastAPI / Streamlit       │
├─────────────────────────────────────┤
│         中间件层 (Middleware)         │
│  RAG Pipeline、Agent Orchestration   │
│  LangChain / LangGraph / DSPy        │
├─────────────────────────────────────┤
│         推理层 (Inference)            │
│  LLM Serving、Embedding、Reranker     │
│  vLLM / SGLang / Ollama              │
├─────────────────────────────────────┤
│         数据层 (Data)                 │
│  Vector DB、Cache、Message Queue       │
│  Qdrant / Redis / Kafka              │
├─────────────────────────────────────┤
│         模型层 (Model)                │
│  Base Model、Adapter、Quantization    │
│  Qwen / DeepSeek / LLaMA             │
├─────────────────────────────────────┤
│         基础设施 (Infrastructure)      │
│  GPU / Cluster / Monitoring / CI/CD   │
│  K8s / Prometheus / Ray              │
└─────────────────────────────────────┘
```

| 层次 | 核心关注点 | 跨层耦合因素 |
|------|-----------|-------------|
| 应用层 | 用户体验、可维护性 | 推理延迟要求、框架兼容性 |
| 中间件层 | 工作流编排、工具链 | 模型能力边界、API 兼容性 |
| 推理层 | 吞吐量、延迟、成本 | 量化方案、batch 策略 |
| 数据层 | 检索速度、一致性 | embedding 模型维度、索引策略 |
| 模型层 | 能力边界、部署成本 | 量化级别、精度要求 |

## 二、每层选型的核心权衡

### 2.1 模型层：能力 vs 成本

```
模型选型矩阵（2026年5月）:

        成本↓  中等    高
能力↑
  SOTA    │  DeepSeek V4  │  Claude 4
          │  Qwen-72B     │  GPT-5
          │  Llama 4      │
          │               │
  中等    │  Qwen-32B     │  
          │  DeepSeek V3  │  
          │  MiMo         │  
          │               │
  基础    │  Qwen-7B      │  
          │  GLM-4-9B     │  
          └───────────────┘
```

**选型原则**：
- 核心推理任务 → 部署能力最强的模型（群组分配最多 GPU）
- 简单分类/提取 → 部署 7-32B 量化模型（AWQ 4-bit，单卡即可）
- 多任务混合 → 用 Router 分发到不同模型

### 2.2 推理层：吞吐量 vs 延迟

| 场景 | 首选引擎 | 理由 |
|------|---------|------|
| Chat / 对话 | vLLM | 最高吞吐，改善 Continuous Batching |
| 结构化输出 (JSON/SQL) | SGLang | RadixAttention + 加速 Constrained Decoding |
| 代码补全 | TensorRT-LLM | 确定性延迟，FP8 加速 |
| 本地开发 | Ollama | 最简单部署，支持 GGUF |
| 边缘部署 | llama.cpp + GGUF | 唯一 CPU/弱 GPU 可行方案 |

**跨层耦合**：推理层选择会影响中间件层的 API schema。vLLM 和 SGLang 都支持 OpenAI API 格式，切换成本最低。TensorRT-LLM 的自定义 API 切换成本较高。

### 2.3 中间件层：灵活性 vs 简洁性

| 框架 | 抽象层级 | 学习曲线 | 灵活性 | 生产成熟度 | 典型场景 |
|------|---------|---------|--------|-----------|---------|
| LangChain | 高层 | 中 | 中 | 高 | 标准 RAG、Chain |
| LangGraph | 图层 | 高 | 高 | 中高 | 多 Agent、复杂工作流 |
| DSPy | 编程 | 中高 | 高 | 中 | 自动 Prompt 优化 |
| CrewAI | Agent 层 | 低 | 低 | 中 | 多 Agent 协作 |
| 原生代码 | - | 低 | 最高 | 最高 | 对确定性要求高的系统 |

**关键决策**：LangChain 类框架的优势在于快速原型，但生产系统中，超过 60% 的团队最终会逐步用原生代码替代框架组件（尤其 RAG Pipeline 和 Tool 定义）。

### 2.4 数据层：一致性 vs 速度

| 类型 | 方案 | 延迟 | 一致性 | 适用场景 |
|------|------|------|--------|---------|
| 向量数据库 | Qdrant | 5-10ms | 强 | 生产级 RAG |
| 向量数据库 | Chroma | 2-5ms | 弱 | 原型/小规模 |
| 缓存 | Redis + RedisVL | 1-3ms | 强 | 高频命中场景 |
| 全文检索 | Elasticsearch | 10-50ms | 强 | 混合搜索 |
| 消息队列 | Kafka | 5-100ms | 强 | 异步事件驱动 |

**新模式：统计缓存（Statistical Cache）**

传统缓存命中/未命中是二元的。AI 应用中出现了统计缓存——缓存近似结果而非精确匹配：

```
用户A: "Attention机制的原理"
  → 缓存键: embedding(用户查询) 近邻匹配
  → 命中: 之前用户B问了"Transformer中的Attention"，返回摘要
  → 延迟: 3ms（vs 500ms 推理）
  → 精度: 85% 的相关性阈值
```

### 2.5 应用层：Web vs API vs 本地

| 模式 | 框架 | UX 质量 | 开发速度 | 部署复杂度 |
|------|------|---------|---------|-----------|
| 聊天 UI | Next.js + Vercel AI SDK | 最高 | 高 | 中 |
| API 服务 | FastAPI | 中 | 高 | 低 |
| 数据应用 | Streamlit / Gradio | 中 | 极高 | 低 |
| 桌面应用 | Tauri / Electron | 最高 | 低 | 高 |
| 移动端 | React Native / Expo | 中 | 中 | 高 |

## 三、典型架构模式

### 3.1 模式 A：标准 RAG 应用

```
用户请求
  │
  ▼
[Router Agent] ── 判断：需要检索？直接推理？
  │                    │
  │ 需要检索            │ 直接推理
  ▼                    ▼
[Query Rewrite]    [LLM Response]
  │
  ▼
[向量检索 Qdrant] ← [Embedding Model (BGE/Qwen-Embedding)]
  │
  ▼
[Reranker (BGE-Reranker / Cohere)]
  │
  ▼
[Prompt Assembly + Context Injection]
  │
  ▼
[vLLM / SGLang] → LLM → 最终输出
```

**推荐技术栈**：Qdrant + vLLM + FastAPI + Next.js/Streamlit

### 3.2 模式 B：多 Agent 协作

```
用户请求
  │
  ▼
[Supervisor Agent (Qwen-72B)]
  │  规划任务、分配子任务
  ├──────────────────────┐
  ▼                      ▼
[Researcher Agent]     [Writer Agent]
  │                      │
  ├─ Web Search          ├─ LLM Generate
  ├─ Doc RAG             ├─ Format Check
  └─ Fact Check          └─ Style Align
  │                      │
  └──────────┬───────────┘
             ▼
  [Reviewer Agent]
   质量检查、合并输出
             │
             ▼
        用户反馈
```

**推荐技术栈**：LangGraph + vLLM + SGLang + Redis

### 3.3 模式 C：流式数据管道

```
事件源 (Kafka / Webhook)
  │
  ▼
[Event Stream Processor]
  │ 使用 LLM 实时分析、分类、提取
  ▼
[Vector Store (Qdrant)] ──→ [RAG Query Service]
  │
  ▼
[Result Sink]
  数据库 / Webhook / Notification
```

**推荐技术栈**：Kafka + SGLang (流式推理) + Qdrant + FastAPI

## 四、全链路选型决策矩阵

```
你的启动条件？
│
├── 单模型单场景（最简）
│   ├── 模型: Qwen-7B/DeepSeek R1 (本地) / API (云端)
│   ├── 推理: vLLM / OpenAI API
│   ├── 编排: 原生 Python
│   ├── 数据: 无 DB 或简单 Postgres
│   └── 应用: FastAPI + 简单前端
│
├── RAG 应用
│   ├── 模型: Qwen-72B/DeepSeek V4
│   ├── 推理: vLLM（主）+ SGLang（结构化）
│   ├── 编排: LangChain → 逐步迁移到原生
│   ├── 数据: Qdrant + Redis Cache
│   └── 应用: Next.js + Vercel AI SDK
│
├── 多 Agent 系统
│   ├── 模型: Qwen-72B (主干) + Qwen-7B (叶子Agent)
│   ├── 推理: vLLM (多模型部署)
│   ├── 编排: LangGraph / Temporal
│   ├── 数据: Qdrant + Message Queue
│   └── 监控: LangSmith + Prometheus
│
└── 企业级平台（高并发、多租户）
    ├── 模型: 分级部署（FP16/FP8/AWQ）
    ├── 推理: vLLM + TensorRT-LLM
    ├── 编排: LangGraph + Temporal (状态机)
    ├── 数据: Qdrant Cluster + Redis Cluster + Kafka
    ├── 监控: Arize + W&B + Prometheus
    └── 基础设施: K8s + Ray + GPU Autoscaler
```

## 五、架构演进的常见路径

```
Phase 1: 验证期 (~1个月)
  模型: API (OpenAI/DeepSeek API)
  推理: 无
  应用: Streamlit / FastAPI
  → 目标: 最快验证产品假设

Phase 2: 自部署期 (~3个月)
  模型: 自部署 Qwen-32B/72B
  推理: vLLM (单卡 → 多卡)
  应用: FastAPI + React
  → 目标: 控制推理成本

Phase 3: 规模化期 (~6个月)
  模型: 分层模型部署（大/中/小 三级）
  推理: vLLM Cluster + SGLang (特定路由)
  编排: LangGraph → 原生 + Temporal
  数据: Qdrant + Redis + Kafka
  监控: 全链路可观测
  → 目标: 支撑多场景、高并发

Phase 4: 成熟期 (~12个月)
  模型: FP8 + AWQ 4-bit 多级量化
  推理: vLLM + TensorRT-LLM 混合
  编排: Temporal 状态机 + 原生 Agent
  数据: 向量 + 缓存 + 流式一体化
  → 目标: 成本最优、可靠性最高
```

## 六、核心原则总结

1. **不要追求最佳，追求可替换**——每层的 API 应该标准化，这样选型错误时可以低成本替换
2. **推理层是瓶颈**——推理框架的选择会影响所有上层。vLLM 兼容 OpenAI API 是最安全的选择
3. **中间件层最容易被替换**——不要投入过多精力优化 LangChain/LangGraph 的封装，关注业务抽象
4. **数据层最难替换**——向量 DB 的选择（自托管 vs 托管、维度策略、索引算法）会影响架构多年
5. **先 API 后自部署**——验证期用商业 API 跑通流程，量上来后再自部署推理服务，不要早期就投入基础架构

**最后**：所有架构决策都是 trade-off。这篇文章给的推荐是 2026 年 5 月的 snapshot，LLM 生态以月为单位迭代——保持对每层替代方案的关注，比选择「最佳方案」更重要。