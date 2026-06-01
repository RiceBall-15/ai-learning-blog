---
title: "LLM Serving架构深潜：从Continuous Batching到PD分离的技术演进"
description: "深入解析LLM推理服务架构的两次范式转变——Continuous Batching、PagedAttention、KV-Cache量化、Prefix Caching、PD分离部署、Elastic EP等关键技术的设计原理、性能权衡与实战选型"
date: 2026-05-24
author: "RiceBall-15"
category: aiInfra
subCategory: inference
tags: ["LLM推理", "vLLM", "Continuous Batching", "PD分离", "KV-Cache量化", "Elastic EP", "推理优化"]
draft: false
---

# LLM Serving架构深潜：从Continuous Batching到PD分离的技术演进

## 一、问题背景：推理服务的内存墙

大语言模型推理面临一个根本性的矛盾：**计算利用率随batch size提升而增加，但KV-Cache内存占用也随batch size和序列长度线性增长**。以Llama-3.3-70B为例，BF16精度下每token的KV-Cache占用2×80×128×2bytes≈40KB。在128k上下文、batch size=64的场景下，单次请求的KV-Cache就高达2.5GB——GPU显存很快成为瓶颈而非算力。

这催生了LLM Serving领域的两次范式转变：

| 阶段 | 核心创新 | 里程碑 | 解决的问题 |
|------|---------|--------|-----------|
| 1.0 - 静态Batch | 固定batch推理 | 早期框架 | 无法动态调度请求 |
| 2.0 - Continuous Batching | 请求级调度 | vLLM v0 | 消除"排队气泡"，提升吞吐3-5x |
| 2.5 - KV-Cache优化 | 量化+前缀缓存 | vLLM v0.4+ | 突破显存墙，支持超长上下文 |
| 3.0 - PD分离 | Prefill/Decode独立部署 | vLLM v0.15+ | 消除PD干扰，Pareto最优吞吐 |
| 3.5 - 弹性MoE | Elastic EP + 容错 | vLLM v0.20+ | 动态扩缩容，运行时拓扑重配 |

本文逐一剖析每项技术的内核原理、设计权衡和生产实战选型建议。

---

## 二、Continuous Batching与PagedAttention

### 2.1 问题：请求间的气泡浪费

传统批处理（static batching）将所有请求拼接成一个batch，等最慢的请求完成后统一返回。这导致严重的"气泡"——早期完成的请求必须等待其他请求，GPU利用率大幅下降。

### 2.2 Continuous Batching原理

Continuous Batching（又称iteration-level scheduling）在**每个decoding step**结束后重新调度：完成的请求立即离开，新请求立即加入，每一步的batch composition都在变化。

```
时间轴 → 
Step 1: [Req1 | Req2 | Req3]  ← 全部在推理
Step 2: [Req1 | Req2 | Req3 | Req4] ← Req4加入
Step 3: [Req1 (done) | Req2 | Req3 | Req4 | Req5] ← Req1离开，Req5加入
Step 4: [Req2 | Req3 | Req4 | Req5] ← 继续
```

关键实现挑战：如何**在推理过程中动态管理每个请求的KV-Cache**，而不是预先分配固定大小的连续显存块？

### 2.3 PagedAttention：操作系统的虚拟内存思想

vLLM的核心创新——PagedAttention——将KV-Cache管理类比为操作系统虚拟内存：

- **固定大小的Block**：KV-Cache按固定大小的block分配（通常16或128 tokens/block）
- **逻辑到物理的映射**：每个请求使用逻辑block ID，通过block table映射到物理显存
- **按需分配**：请求开始时只需分配少量block，后续推理中动态分配
- **内存共享**：Prefix Caching和Beam Search中可以共享block

```
逻辑视角（每个请求）：
Req1: [Token0-15] [Token16-31] [Token32-47] [Token48-...]
       Block 0      Block 1      Block 2      Block 3

物理视角（全局block表）：
Block Table: [0→PhysA] [1→PhysC] [2→PhysF] [3→PhysH]
             [Req2 Block0→PhysB] [Req2 Block1→PhysC] ← 共享Block1！
```

**性能收益**：
- 内存碎片减少90%+（传统预分配方案碎片高达60%+）
- 支持更大的batch size和更长的上下文
- 同行请求可共享相同前缀的block（Prefix Caching的基础）

### 2.4 实战要点

| 参数 | 推荐值 | 影响 |
|------|--------|------|
| `--block-size` | 16（短上下文）/ 128（长上下文） | 小block减少内存浪费，大block减少管理开销 |
| `--max-num-seqs` | 256-2048 | 越大batch越大，但block争用加剧 |
| `--enable-prefix-caching` | true | 共享前缀，对agent/chat场景效果显著 |

---

## 三、KV-Cache量化：FP8 vs TurboQuant的全面对决

### 3.1 为什么需要KV-Cache量化？

BF16精度下，KV-Cache是最大的显存消耗源。在长上下文场景中，KV-Cache占用的显存超过模型权重本身。KV-Cache量化的目标：**在精度损失可控的前提下，将KV-Cache的存储位数从16bit降到更低**。

### 3.2 FP8量化：硬件原生的最优解

vLLM通过`--kv-cache-dtype fp8`启用FP8 KV-Cache量化。其核心优势：

- **硬件原生支持**：H100/H200的FP8 Tensor Core直接处理FP8精度的矩阵运算
- **计算上也收益**：注意力计算直接在FP8上进行，无需反量化
- **2x容量提升**：与BF16相比显存占用减半
- **精度几乎无损**：在AIME25、GPQA等推理基准上恢复率>99%

### 3.3 TurboQuant：更低bit的激进压缩

TurboQuant将KV-Cache压缩到3-4bit，但**仅压缩存储，注意力计算仍回退到BF16**。这带来了两个根本性问题：

| 指标 | FP8 | TurboQuant k8v4 | TurboQuant 4bit-nc | TQ 3bit-nc |
|------|-----|-----------------|-------------------|------------|
| KV-Cache容量提升 | 2x | 2.4x | 3.4x | 4x+ |
| 吞吐量(BF16基准) | 100%+ | 80% | 75% | 66% |
| 推理精度恢复 | >99% | >98% | 96% | ~80% |
| 延时开销 | ~0% | 10-30% | 20-50% | 高达68% |
| 长上下文(128k+) | 稳定 | 稳定 | 微降 | AUC下降30% |

**为什么TurboQuant性能反而更差？** 原因在于反量化开销：TurboQuant的注意力计算每次都需要从3-4bit打包格式解码回BF16，这个解码过程本身就是耗时的CUDA操作，且随batch size增大而线性增长。

> **实战结论**：FP8是NVidia GPU上的默认最优选择。只有当显存极度受限（如边缘设备）且可接受10-20%精度损失时，才考虑TurboQuant 4bit-nc。TQ 3bit-nc在推理任务上精度损失可达20个点，不建议生产使用。

---

## 四、Prefix Caching：共享计算的艺术

### 4.1 场景分析

Agentic工作负载和多轮对话中，大量请求共享相同的前缀：
- **System Prompt**：所有Agent调用共享相同系统提示词
- **对话历史**：同一会话的每个新请求共享之前的历史
- **RAG上下文**：相同知识库前缀

### 4.2 实现原理

Prefix Caching通过hash技术实现block级别的共享：

```
请求A: [System Prompt | User Query A | ...]
         Block 0-10   Block 11-20
请求B: [System Prompt | User Query B | ...]
         Block 0-10 (共享!)  Block 11-20 (新计算)
```

vLLM在block table中维护一个全局hash表（hash table size = `num_gpu_blocks`）。当新请求进入时，计算其每个block的hash值，如果hash命中且内容一致，则直接引用已有block，跳过计算。

### 4.3 性能提升

结合Mooncake等外部KV-Cache存储，Agentic场景下的吞吐提升可达**3.8x**（vLLM × Mooncake集成数据）。

### 4.4 陷阱：缓存污染与哈希冲突

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| 哈希冲突 | 不同内容算出的hash相同，导致错误输出 | 增加hash位数，使用内容校验 |
| 缓存污染 | 低价值请求占满缓存 | 设置prefix-caching-min-tokens阈值 |
| 缓存过期 | Agent场景下多轮会话的旧前缀被覆写 | 设置LRU淘汰策略 |

---

## 五、PD分离部署：消除Prefill-Decode干扰

### 5.1 核心矛盾

Prefill阶段和Decode阶段对GPU资源的需求截然不同：

| 特征 | Prefill | Decode |
|------|---------|--------|
| 计算模式 | 计算密集（矩阵乘法） | 访存密集（KV-Cache读取） |
| KV-Cache增长 | 快速（每步写入大量新tokens） | 缓慢（每步写入1 token） |
| 延迟敏感度 | 中（影响TTFT） | 高（影响TPOT） |
| 批次灵敏度 | 高吞吐随batch线性增长 | 受显存带宽限制 |

将两者混合在同一个GPU上运行时，Prefill的矩阵计算会抢占Decode的访存带宽，导致decode step被阻塞——这就是**PD干扰**。

### 5.2 PD分离架构

PD分离的核心思想：**将Prefill和Decode分别部署在不同GPU实例上**，通过高速互联（NVLink/RDMA）传输KV-Cache。

```
                Prefill GPUs (TP=4)          Decode GPUs (TP=4)
                       │                          │
输入请求 ──→ [Prefill Engine] ──KV-Cache──→ [Decode Engine] ──→ 输出
                              (NIXL RDMA)
```

在vLLM中，PD分离通过NIXL（NVIDIA Interconnect Library）实现GPU间的零拷贝KV-Cache传输。整个过程在Decode端看来是**一次异步RDMA READ**，无需中间缓冲、无需数据重排。

### 5.3 NIXL传输流程（4阶段）

1. **注册内存区域**：每个worker将KV-Cache张量注册到NIXL，使其可通过RDMA访问
2. **创建Block描述符**：为每个block创建(address, length, device_id)描述符
3. **握手**：Decode实例首次需要从Prefetch实例拉数据时，互换元数据
4. **传输**：Scheduler告诉Decode哪些block需要拉取，Decode发出RDMA READ

### 5.4 混合SSM模型的PD分离挑战

Hybrid SSM-FA模型（如NVIDIA Nemotron-H）进一步增加了PD分离的复杂性。SSM层的状态与Attention层完全不同：

| 维度 | Attention层(KV-Cache) | SSM层(Mamba状态) |
|------|----------------------|-----------------|
| 状态类型 | 每token(K,V)对 | collapsed conv + SSM状态 |
| Block大小 | block_size × num_kv_heads × head_dim | 固定大小，不随序列增长 |
| 数据传输 | uniform descriptor | 需要3-descriptor分解 |

vLLM的解法——**Dual Descriptor Views**：在同一个物理内存上注册两套NIXL block描述符，一套用于Attention，一套用于SSM。SSM的conv状态通过DS layout（dim, state_len）排列，使每个异构TP rank只需读取自己需要的连续字节。

### 5.5 性能收益

在8×H200上对Nemotron Super 120B的测试表明：

| 指标 | 同机部署(TP=8) | PD分离(P=4, D=4) | 提升 |
|------|---------------|-----------------|------|
| 吞吐量(低并发) | 基线 | 相近 | ~0% |
| 吞吐量(高并发256) | 瓶颈 | Pareto主导 | 30-50%+ |
| Burst TTFT | 随batch爆炸 | 稳定 | 5-10x |

---

## 六、Elastic EP：MoE模型的运行时弹性扩缩容

### 6.1 MoE推理的独特性

MoE模型（如DeepSeek-V2/V3、Mixtral）与Dense模型在推理架构上有本质不同：

- **Attention层保持Dense**：采用Data Parallel (DP) 注意力
- **Expert层稀疏路由**：采用Expert Parallelism (EP)，每个GPU只负责部分expert

传统vLLM中EP是静态的：一旦启动，服务能力固定。无法根据流量波峰波谷动态调整。

### 6.2 Elastic EP的设计

Elastic EP的核心理念：**运行时重新配置DP大小**，一个API调用完成扩缩容：

```bash
# 扩容
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 16}'

# 缩容
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 8}'
```

### 6.3 扩容6步状态机

扩容涉及到多个运行时状态需要重新配置：

| 步骤 | 操作 | 关键技术 |
|------|------|---------|
| 1. 请求排空 | 等待进行中的in-flight请求完成 | 可选，`VLLM_ELASTIC_EP_DRAIN_REQUESTS=1` |
| 2. 新引擎初始化 | Ray DP启动新worker | 使用Ray分布式调度 |
| 3. Standby通信组 | 现有rank创建备用group而不销毁当前group | StatelessGroupCoordinator，零中断 |
| 4. Expert权重传输 | 将非expert权重从现有rank广播到新rank | 复用EPLB的GPU-GPU传输路径 |
| 5. 切换 | 旧→新的原子切换 | CUDA graph释放→激活新group→重新warmup |
| 6. EPLB重排 | EPLB在所有rank间重新分布expert | 普通EPLB执行 |

### 6.4 DP Rank间的协调难题

一个工程上的精妙设计：DP引擎核心是异步运行的，不同rank收到重配置通知的时间不同。如果先到的rank直接进入下一步，而其他rank还在执行forward，就会死锁。

Elastic EP使用**two-stage barrier**解决：
1. **Timeout barrier**：若超时未完成，推断某些peer还在engine loop中，回退到下一轮迭代
2. **Non-timeout barrier**：所有rank到达相同边界后，再同步进入下一阶段

### 6.5 容错基石

Elastic EP为vLLM的容错（fault tolerance）提供了基础运行时重配置路径：

```
检测故障 → 缩容(移除故障rank+重分布expert) → 扩容(添加替代capacity)
```

NIXL EP作为通信后端时，还能提供EP侧的故障检测、报告和恢复能力。

---

## 七、实战选型指南

### 7.1 场景化推荐

| 场景 | 推荐配置 | 关键参数 |
|------|---------|---------|
| Chat/对话 | FP8 + Prefix Cache | `--kv-cache-dtype fp8 --enable-prefix-caching` |
| Agent/RAG（长上下文） | FP8 + PD分离 | 加上`--kv-transfer-config`启用NIXL |
| 高并发推理 | PD分离 + Elastic EP | 自建autoscaling策略对接`/scale_elastic_ep` |
| 边缘部署（显存受限） | TurboQuant 4bit-nc | `--kv-cache-dtype turboquant_4bit_nc` |
| MoE模型（DeepSeek系列） | FP8 + EP + Elastic EP | `--enable-expert-parallel --enable-elastic-ep` |

### 7.2 不要踩的坑

1. **TurboQuant不是FP8的替代品**：它们是不同维度的优化。FP8同时优化存储和计算，TurboQuant只压缩存储
2. **PD分离不是银弹**：在低并发（<32 concurrent users）场景，PD分离的收益微乎其微，反而增加网络开销
3. **Elastic EP目前限制**：`tensor_parallel_size`必须为1，不支持DBO和MoE draft model
4. **Prefix Caching在大模型上效果不同**：70B+模型上缓存命中率高，小模型因推理速度快缓存收益有限
5. **Block Size的选择**：16 vs 128没有绝对最优，需要在内存利用率和管理开销之间做trade-off

---

## 八、总结

LLM Serving架构正经历从"跑起来"到"跑得快"再到"跑得好"的演进。核心方向始终围绕一个矛盾：**如何在大batch下管理好KV-Cache的内存与计算资源**。

| 技术 | 解决的问题 | 适用阶段 |
|------|-----------|---------|
| PagedAttention + Continuous Batching | 消除批处理气泡 | 必须 |
| FP8 KV-Cache量化 | 缓解显存墙 | 强烈推荐 |
| Prefix Caching | 共享前缀计算 | Agent/RAG场景推荐 |
| PD分离 | 消除Prefill-Decode干扰 | 高并发场景推荐 |
| Elastic EP | MoE动态扩缩容 | 生产MoE推荐 |

对于大多数生产部署，推荐的最小配置：**FP8 + Prefix Caching + Continuous Batching**耗时不到5行参数，即可获得2x KV-Cache容量和稳定的推理性能。随着并发量上升和上下文变长，逐步引入PD分离和Elastic EP。

---

**参考来源**：
1. vLLM Blog: "The State of FP8 KV-Cache and Attention Quantization in vLLM" (2026-04-22)
2. vLLM Blog: "A First Comprehensive Study of TurboQuant" (2026-05-11)
3. vLLM Blog: "Disaggregated Serving for Hybrid SSM Models in vLLM" (2026-04-21)
4. vLLM Blog: "Elastic Expert Parallelism in vLLM" (2026-05-14)
5. vLLM Blog: "Serving Agentic Workloads at Scale with vLLM x Mooncake" (2026-05-06)
6. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
7. vLLM GitHub: RFC #20323 (Elastic EP), PR #34861, RFC #30112 (Fault Tolerance)