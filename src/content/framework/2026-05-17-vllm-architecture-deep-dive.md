---
title: "vLLM 深度解剖：从 PagedAttention 到分布式推理的完整架构"
description: "深入解析 vLLM 推理引擎的核心架构——PagedAttention、连续批处理、前缀缓存、投机解码、P/D 分离，以及从单 GPU 到多节点的扩展路径"
date: 2026-05-17
author: "RiceBall-15"
category: framework
tags: ["vLLM", "LLM推理", "PagedAttention", "连续批处理", "分布式推理", "高性能推理"]
draft: false
---

# vLLM 深度解剖：从 PagedAttention 到分布式推理的完整架构

## 为什么需要理解 vLLM 的内部机制？

vLLM 是当前最广泛使用的 LLM 推理引擎之一，但大多数开发者只停留在 `pip install vllm && vllm serve` 的层面。当遇到以下问题时，表面知识就不够用了：

- **显存利用率低**：不了解 PagedAttention 的块分配机制，无法调优 `gpu_memory_utilization`
- **长文本推理卡顿**：不理解 Chunked Prefill，无法处理长 prompt 独占引擎步的问题
- **多轮对话重复计算**：不了解前缀缓存的工作原理，每次都在重复计算 system prompt
- **延迟优化无从下手**：不理解 Prefill/Decode 的性能差异，无法针对性优化

本文基于 vLLM V1 引擎的源码分析（commit 42172ad），从最底层的引擎构造讲起，逐层构建到分布式推理系统。

## 整体架构：从单次推理到分布式服务

vLLM 的架构分为五层，每一层解决一个特定问题：

| 层级 | 组件 | 职责 | 对应类 |
|------|------|------|--------|
| L1: 引擎核心 | LLM Engine + EngineCore | 调度、KV 管理、前向传播 | `LLM`, `EngineCore` |
| L2: 高级特性 | Chunked Prefill, Prefix Cache, Speculative Decoding | 优化推理效率 | 各模块组件 |
| L3: 执行器 | UniProcExecutor → MultiProcExecutor | 单 GPU → 多 GPU 执行 | `Executor` 子类 |
| L4: 分布式协调 | DPCoordinator, Connector | 数据并行、KV 传输 | `DPLBAsyncMPClient` |
| L5: 服务层 | FastAPI + Uvicorn | OpenAI 兼容 API | `AsyncLLM` |

### 核心组件关系

```
┌─────────────────────────────────────────────────────┐
│                    LLM Engine                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │Processor │  │ Scheduler│  │  Output Processor  │  │
│  │(tokenize)│  │(调度决策) │  │  (detokenize)      │  │
│  └──────────┘  └────┬─────┘  └───────────────────┘  │
│                     │                                │
│         ┌───────────┼───────────┐                    │
│         │           │           │                    │
│    ┌────▼────┐ ┌────▼────┐ ┌───▼──────┐            │
│    │Waiting  │ │Running  │ │KV Cache  │            │
│    │Queue    │ │Queue    │ │Manager   │            │
│    │(prefill)│ │(decode) │ │(块分配)  │            │
│    └─────────┘ └─────────┘ └──────────┘            │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │              Model Executor                   │   │
│  │  ┌─────────┐  ┌──────────┐  ┌────────────┐  │   │
│  │  │ Worker  │  │Model     │  │InputBatch  │  │   │
│  │  │(GPU操作)│  │Runner    │  │(CPU缓冲区) │  │   │
│  │  └─────────┘  └──────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 引擎构造：三大初始化步骤

当执行 `LLM(model="...")` 时，引擎按顺序完成三步关键初始化：

### 1. 设备初始化

- 分配 CUDA 设备，验证模型 dtype 支持（如 bf16）
- 根据 `gpu_memory_utilization`（默认 0.8）计算可用显存
- 创建 `model_runner`（持有采样器、KV 缓存、前向传播缓冲区）
- 创建 `InputBatch`（CPU 侧的前向传播缓冲区、块表、采样元数据）

### 2. 模型加载

- 实例化模型架构并加载权重
- 调用 `model.eval()` 切换到推理模式
- 可选：调用 `torch.compile()` 编译模型

### 3. KV 缓存初始化（最关键）

这一步决定了推理引擎的显存利用效率：

1. 获取每层 KV 缓存规格（标准 Transformer vs 混合模型如 Jamba）
2. 运行一次 dummy 前向传播，通过 GPU 内存快照计算可容纳的 KV 缓存块数
3. 分配、重塑 KV 缓存张量并绑定到注意力层
4. 准备注意力元数据（设置 FlashAttention 后端）
5. 为每个预热 batch size 捕获 CUDA Graph（除非指定 `--enforce-eager`）

**KV 缓存块大小计算**（标准 Transformer，非 MLA）：

```
block_bytes = 2 (K/V) × block_size(默认16) × num_kv_heads × head_size × dtype_bytes(如 bf16=2)
```

CUDA Graph 的作用是将整个 GPU 工作序列录制为 DAG，后续直接回放，消除内核启动开销，显著降低延迟。

## PagedAttention：突破显存墙的核心

传统推理引擎为每个序列预分配连续显存，导致严重的内存碎片和浪费。PagedAttention 借鉴操作系统的虚拟内存思想，将 KV 缓存切分为固定大小的块（默认 16 token），通过块表（block table）进行间接寻址。

### 块分配机制

KV Cache Manager 维护一个 `free_block_queue`（双向链表），通常包含数十万个可用块。分配流程：

```
allocate_slots(request, num_new_tokens):
    num_blocks = ceil(num_new_tokens / block_size)
    
    # 检查可用块数
    if free_block_queue.size < num_blocks:
        # 尝试抢占低优先级请求的块
        evict_low_priority_requests()
        if still_insufficient:
            skip_scheduling()  # 跳过本步调度
    
    # 从空闲队列分配块
    blocks = free_block_queue.pop_left(num_blocks)
    req_to_blocks[request_id].extend(blocks)
    return blocks
```

### 连续批处理（Continuous Batching）

传统静态批处理中，所有请求必须等最长的那个完成。连续批处理允许在每个引擎步结束后，既有请求继续生成，新请求也能加入。

vLLM V1 的关键改进：**可以在同一个步中混合处理 Prefill 和 Decode 请求**。V0 只能交替处理其中一种。

调度优先级：
1. **优先处理 Decode 请求**（已在 running 队列中）——保障已开始生成的请求延迟
2. **然后处理 Prefill 请求**（从 waiting 队列取出）——新请求开始计算

每步的三阶段循环：

```
while has_requests:
    # 1. 调度：选择本步要处理的请求
    schedule_requests()
    
    # 2. 前向传播：运行模型并采样
    output_token_ids = model_executor.execute_model()
    
    # 3. 后处理：追加 token、检查停止条件
    for request in running_requests:
        request.append_token(output_token_ids)
        if request.should_stop():
            free_kv_blocks(request)
            return_output_early(request)
```

**为什么连续批处理可行？** 因为前向传播将所有序列展平为一个"超级序列"，位置索引和注意力掩码确保每个序列只关注自己的 token，无需右填充。

## 高级特性一：Chunked Prefill

### 问题

一个超长 prompt 的 Prefill 会独占整个引擎步，导致其他请求延迟激增。

### 方案

将长 prompt 的 Prefill 拆分为多个小块（默认 8 token 一块），分散到多个引擎步中执行。

```
原始 prompt: [x-y-z]  （z 为不完整块，如 2 个 token）

Step 1: 处理 [x] → 计算 K/V，存入缓存块
Step 2: 处理 [y] → 计算 K/V，存入缓存块  
Step 3: 处理 [z] → 计算 K/V，存入缓存块，采样第一个输出 token
```

**启用方式**：设置 `long_prefill_token_threshold` 为正整数。当 prompt 长度超过 token budget 时，会自动触发 Chunked Prefill。

### 效果

| 指标 | 无 Chunked Prefill | 有 Chunked Prefill |
|------|-------------------|-------------------|
| 长 prompt 独占时间 | 整个引擎步 | 仅一个 chunk |
| 其他请求延迟 | 显著增加 | 基本不受影响 |
| TFTT（首 token 延迟） | 高且波动大 | 更平稳 |

## 高级特性二：前缀缓存（Prefix Caching）

### 场景

多轮对话中，system prompt + 历史消息作为前缀在每次请求中重复出现。没有前缀缓存时，每次都要重新计算这些 token 的 K/V。

### 工作原理

1. **首次请求**：将 prompt 按 block_size 切分，对每个完整块计算哈希（前一块哈希 + 当前 token + 元数据），存入 `cached_block_hash_to_block`
2. **后续请求**：对新 prompt 计算哈希，调用 `find_longest_cache_hit` 查找匹配的块，直接复用其 K/V

```
请求1: [system_prompt(80 tokens)] + [user_msg_1]
       → 计算 5 个块的 K/V，哈希缓存

请求2: [system_prompt(80 tokens)] + [user_msg_2]
       → 哈希命中 5 个块，跳过 Prefill，只计算 user_msg_2
```

### 关键细节

- **块失效机制**：块只有在从 `free_block_queue` 被重新分配时才失效。此时清除其哈希并从缓存中移除
- **引用计数**：多个请求共享同一块时，引用计数递增；请求完成后递减
- **默认启用**：`enable_prefix_caching = True`（默认）
- **对齐要求**：前缀必须对齐到 block_size 边界，否则不完整部分需要重新计算

### 性能影响

| 场景 | 无前缀缓存 | 有前缀缓存 |
|------|-----------|-----------|
| 多轮对话（80 token system prompt） | 每次计算 80 token | 仅首次计算 |
| RAG 场景（共享上下文） | 重复计算文档 token | 复用已缓存块 |
| 显存占用 | 不变 | 略增（哈希表开销） |

## 高级特性三：引导解码（Guided Decoding）

引导解码通过有限状态机（FSM）约束每步的 token 采样，确保输出符合指定语法。

### 支持的语法层级

- **正则表达式**（Chomsky 3 型）：任意 regex 模式
- **上下文无关文法**（Chomsky 2 型）：覆盖大多数编程语言
- **选择约束**：限定输出为指定选项之一

### 工作流程

1. 请求添加时，`grammar_init` 选择后端编译器（如 xgrammar），异步编译语法
2. 调度阶段，编译完成后更新 `_grammar_bitmask`
3. 前向传播产生 logits 后，位掩码展开到词表大小，将禁止 token 的 logits 设为 `-∞`
4. 采样后，FSM 状态前移到下一状态

```
示例：约束输出为 ["Positive", "Negative"]

Step 1: FSM 只允许 "P" 或 "N"
Step 2: 如果采样 "P"，FSM 转到 "Positive" 分支，只允许 "o"
Step 3: 依次 "s" → "i" → "t" → "i" → "v" → "e"
```

### 位掩码机制

假设词表大小 32，`_grammar_bitmask` 是一个 32 位整数，二进制表示编码哪些 token 允许（1）vs 禁止（0）。展开后与 logits 逐元素相乘。

## 高级特性四：投机解码（Speculative Decoding）

### 核心思想

自回归生成中，每个 token 都需要一次完整的模型前向传播。投机解码用一个小模型（draft model）廉价地猜测 k 个 token，然后用大模型一次性验证。

### 算法流程

```
1. Draft: 小模型在当前上下文上提议 k 个 token
2. Verify: 大模型在 context + k 个 draft token 上运行一次前向传播
           产生 k+1 个位置的概率分布
3. Accept/Reject: 从左到右逐个检查
   - 如果 P_large(token) ≥ P_draft(token) → 接受
   - 否则以 P_large/P_draft 的概率接受
   - 遇到第一个拒绝就停止
   - 如果全部接受，额外采样第 k+1 个 token（"免费"）
```

**关键保证**：投机解码在统计上等价于标准自回归采样——输出分布完全一致，但潜在速度提升 k+1 倍。

### vLLM 支持的 Draft 方案

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **n-gram** | 在序列中查找历史匹配，提议后续 token | 零额外开销，无需训练 | 准确率较低 |
| **EAGLE** | 保留嵌入层和 LM Head，用轻量 MLP 替换 Transformer 栈 | 准确率高 | 需要额外微调 |
| **Medusa** | 在大模型上训练辅助线性头，并行预测 k 个 token | 无需小模型 | 需要额外训练 |

**注意**：vLLM V1 不支持 LLM 小模型作为 draft（V0 曾支持），只支持上述三种方案。

## 高级特性五：P/D 分离（Disaggregated Prefill/Decode）

### 动机

Prefill 和 Decode 有截然不同的性能特征：

| 特性 | Prefill | Decode |
|------|---------|--------|
| 计算模式 | 处理全部 prompt token | 仅处理 1 个新 token |
| 瓶颈 | **计算密集型** | **显存带宽密集型** |
| 优化目标 | TFTT（首 token 延迟） | ITL（token 间延迟） |

将它们分离到不同实例上，可以独立优化各自的延迟。

### 架构

```
                  ┌─────────────────┐
                  │   KV Cache      │
                  │   Service       │
                  └────────┬────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼───┐  ┌────▼─────┐  ┌──▼─────────┐
     │ Prefill    │  │ Prefill  │  │  Decode    │
     │ Instance 1 │  │Instance 2│  │  Instance  │
     │ (GPU 0-3)  │  │(GPU 4-7) │  │  (GPU 8+)  │
     └────────────┘  └──────────┘  └────────────┘
```

- N 个 Prefill 实例处理输入，将 K/V 写入缓存服务
- M 个 Decode 实例从缓存服务读取 K/V，执行延迟敏感的生成
- 根据实时请求负载自动扩缩容

### 实现细节

1. **连接器抽象**：`KVTransferConfig` 配置 KV 缓存的传输方式（SharedStorageConnector 用于调试，LMCache/NIXL 用于生产）
2. **调度阶段**：Decode 实例调用 `get_num_new_matched_tokens` 检查外部缓存中的 token
3. **前向传播前后**：进入上下文管理器，Prefill 上传 KV，Decode 加载 KV
4. **首次加载**：Decode 仅在第一步从外部加载 KV，后续在本地计算/存储

## 从单 GPU 到多节点：执行器的演进

### UniProcExecutor（单 GPU）

单进程执行，所有操作在一个 Worker 中完成。适用于小模型或开发调试。

### MultiProcExecutor（多 GPU，同节点）

当模型需要张量并行（TP）时，使用多进程执行器：

```
MultiProcExecutor
├── Worker Process (rank 0, GPU 0) ← Driver
├── Worker Process (rank 1, GPU 1)
├── Worker Process (rank 2, GPU 2)
└── Worker Process (rank 3, GPU 3)

通信方式：rpc_broadcast_mq（共享内存消息队列）
```

**工作流程**：
1. 主进程为每个 rank 创建守护进程
2. 每个 Worker 完成设备初始化、模型加载、KV 缓存初始化
3. 通过共享内存队列接收工作项，执行前向传播，返回结果
4. 从引擎视角，MultiProcExecutor 和 UniProcExecutor 接口完全一致

### 分布式服务（多节点）

在两个 8×H100 节点上运行 4 个 vLLM 引擎的典型配置：

```
节点 1（Headless）:
  vllm serve <model>
    --tensor-parallel-size 4
    --data-parallel-size 4
    --data-parallel-size-local 2
    --data-parallel-start-rank 0
    --headless

节点 2（API Server）:
  vllm serve <model>
    --tensor-parallel-size 4
    --data-parallel-size 4
    --data-parallel-size-local 2
    --data-parallel-start-rank 2
```

**每个 DP 副本的线程模型**：

```
DPEngineCoreProc
├── Input Thread  ← 从 ZMQ socket 接收请求，放入 input_queue
├── Main Thread   ← 从 input_queue 取请求，驱动引擎执行
└── Output Thread ← 从 output_queue 取结果，通过 socket 返回
```

**DP Coordinator** 协调前端和后端：
- 定期发送负载均衡信息（队列大小、请求数量）
- 处理动态扩缩容命令
- 管理 DP 波次计数器

**Dummy Steps**：当某个 DP 副本有工作而其他空闲时，空闲副本执行 dummy step 参与同步点，避免阻塞活跃副本（主要针对 MoE 模型的 EP 同步）。

## 请求完整生命周期

从 `curl` 命令到返回结果，经过以下路径：

```
curl POST /v1/completions
    │
    ▼
FastAPI + Uvicorn (服务层)
    │
    ▼
AsyncLLM → DPLBAsyncMPClient (异步客户端)
    │
    ▼
DP Coordinator (负载均衡决策)
    │
    ▼
DPEngineCoreProc Input Thread (接收请求)
    │
    ▼
EngineCore → Scheduler (调度决策)
    │  ├─ KV Cache Manager: allocate_slots()
    │  ├─ Prefix Cache: find_longest_cache_hit()
    │  └─ Chunked Prefill: 拆分长 prompt
    │
    ▼
Model Executor → Worker → Model Runner (前向传播)
    │  ├─ 准备输入: CPU→GPU 拷贝, slot_mapping
    │  ├─ 前向传播: PagedAttention 内核
    │  ├─ 采样: temperature, top-p, top-k
    │  └─ 可选: Rejection Sampler (投机解码)
    │
    ▼
Output Processor (detokenize, 检查停止条件)
    │
    ▼
返回 Response
```

## 实战调优建议

### 显存调优

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `gpu_memory_utilization` | 显存使用比例 | 0.85-0.92（留余量给 CUDA） |
| `max_model_len` | 最大序列长度 | 根据显存和 batch size 平衡 |
| `block_size` | KV 缓存块大小 | 默认 16，一般不改 |
| `enable_prefix_caching` | 前缀缓存 | True（多轮对话必开） |

### 延迟优化

| 场景 | 优化策略 |
|------|---------|
| 首 token 延迟高 | 启用 Chunked Prefill，避免长 prompt 独占 |
| token 间延迟高 | 使用投机解码（n-gram 最简单） |
| 多轮对话慢 | 启用前缀缓存，复用 system prompt |
| 结构化输出慢 | 使用 xgrammar 后端，异步编译语法 |

### 吞吐量优化

| 场景 | 优化策略 |
|------|---------|
| 单 GPU 吞吐瓶颈 | 使用 TP=2/4 跨 GPU 分片 |
| 跨节点扩展 | 使用 DP + 负载均衡 |
| Prefill/Decode 混合负载 | P/D 分离，独立扩缩容 |
| GPU 利用率低 | 调高 `gpu_memory_utilization`，增大 batch size |

## 与 SGLang 的关键差异

| 特性 | vLLM | SGLang |
|------|------|--------|
| KV 管理 | PagedAttention（块分配） | RadixAttention（基数树） |
| 前缀缓存 | 哈希匹配 | 前缀树自动共享 |
| 调度策略 | FCFS + Priority | 基于代价的调度 |
| 投机解码 | n-gram, EAGLE, Medusa | 多种 draft 方案 |
| P/D 分离 | Connector 抽象 | Mooncake 集成 |
| 编程接口 | OpenAI API 兼容 | 原生 DSL + API 兼容 |

vLLM 更适合通用推理服务，生态最成熟；SGLang 在复杂推理管线（多轮调用、树搜索）上有独特优势。

## 总结

vLLM 的核心创新不在于单一技术，而在于将操作系统的设计思想（虚拟内存、分页、调度）系统性地应用到 LLM 推理中：

1. **PagedAttention** 解决了显存碎片问题，使 KV 缓存利用率接近 100%
2. **连续批处理** 消除了静态批处理的等待浪费
3. **前缀缓存** 通过哈希匹配避免重复计算
4. **Chunked Prefill** 防止长 prompt 独占引擎步
5. **投机解码** 用小模型加速大模型的采样
6. **P/D 分离** 将不同瓶颈的计算解耦到独立实例

理解这些机制，才能在生产环境中做出正确的架构和调优决策。

---

*参考来源*：
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/blog/2025-09-05-anatomy-of-vllm) — Aleksa Gordic, vLLM Blog
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023
- [vLLM 官方文档](https://docs.vllm.ai/)
- [SGLang 官方文档](https://docs.sglang.ai/)
