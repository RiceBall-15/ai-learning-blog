---
title: "Prefill-Decode分离架构深度解析：LLM推理规模化部署的新范式"
description: "系统剖析PD分离架构(Prefill-Decode Disaggregation)的技术原理、工程实现与生产部署，覆盖Mooncake、DistServe、Splitwise等主流方案，解析如何通过计算解耦实现LLM推理的极致吞吐与超低延迟"
date: 2026-05-31
author: "RiceBall-15"
category: "aiInfra"
subCategory: inference
tags: ["PD分离", "Prefill-Decode", "LLM推理", "推理架构", "Mooncake", "DistServe", "Splitwise", "推理优化"]
draft: false
---

# Prefill-Decode分离架构深度解析：LLM推理规模化部署的新范式

## 一、引言：LLM推理的"不可能三角"

### 1.1 一个真实的架构困境

某AI平台的推理团队面临着一个典型困境：

- 业务要求：**首token延迟(TTFT) < 500ms，吞吐 > 1000 req/s**
- 资源约束：40张A100-80G GPU
- 现状：使用vLLM统一部署，TTFT约800ms，吞吐约400 req/s

尝试过各种优化——调大batch size能提升吞吐但恶化延迟，调小batch size能降低延迟但浪费算力。**他们发现自己陷入了LLM推理的"不可能三角"：**

```
            低延迟(TTFT)
               △
              / \
             /   \
            /     \
           /  当前  \
          /  约束   \
         /           \
        ▽─────────────▽
    高吞吐          低成本
```

这三个目标在传统的一体化（Monolithic）推理架构下，**最多只能同时满足两个**。

### 1.2 问题的根源：Prefill和Decode的天然矛盾

LLM推理由两个本质不同的计算阶段组成：

```
用户请求 → [Prefill阶段] → [Decode阶段] → 输出

Prefill（预填充）：
  输入: 整个prompt（可能数千token）
  计算: 所有输入token并行计算KV Cache
  特点: 计算密集型(Compute-Bound)
  耗时: 与输入长度近似线性
  GPU利用: 接近理论峰值

Decode（解码生成）：
  输入: 前一步生成的单个token
  计算: 自回归逐token生成
  特点: 访存密集型(Memory-Bound)  
  耗时: 每步常数时间，但需重复多次
  GPU利用: 大部分时间在等显存读取
```

这两种阶段的计算特性差异巨大：

| 特性 | Prefill | Decode |
|-----|---------|--------|
| 算术强度(OP/Byte) | 高（大矩阵乘法） | 低（向量-矩阵乘法） |
| 计算瓶颈 | 计算单元(ALU) | 显存带宽(HBM) |
| 批量处理效率 | 高（天然支持并行） | 低（逐token序列化） |
| GPU利用率 | 60-90% | 10-30% |
| 延迟影响因素 | 输入长度 × batch size | 输出长度（每步恒定） |

当Prefill和Decode共享同一GPU时，它们会互相干扰：

```
传统一体化部署：

GPU 0: [  Prefill  ][ Decode ][  Prefill  ][ Decode ]...
        ↑ 计算密集   ↑ 访存密集  ↑ 又切换到计算  ↑ 又切换

问题：
1. Prefill抢占算力 → Decode延迟升高(TPOT增加)
2. Decode占用显存 → Prefill的batch无法做大(吞吐下降)
3. KV Cache混在同一显存池 → 内存碎片化严重
```

### 1.3 分离的直觉

**如果Prefill和Decode天然不兼容，为什么不把它们放到不同的GPU上？**

这就是Prefill-Decode Disaggregation（PD分离）的核心思想：

```
PD分离架构：

请求 → [Prefill节点集群] → KV Cache传输 → [Decode节点集群] → 输出
         ↑ GPU 0,1,2             网络          ↑ GPU 3,4,5
         纯Prefill              ←────→         纯Decode
         高利用率                               高利用率
```

---

## 二、技术原理：PD分离的核心机制

### 2.1 架构概览

PD分离的典型架构包含三个核心组件：

```
                    ┌──────────────────┐
                    │   请求调度器     │
                    │  (Router/Proxy)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ Prefill    │  │ Prefill    │  │ Prefill    │
     │ Worker 0   │  │ Worker 1   │  │ Worker 2   │
     │ (4×A100)   │  │ (4×A100)   │  │ (4×A100)   │
     └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
            │               │               │
            │    KV Cache Transfer (RDMA)    │
            │               │               │
            ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ Decode     │  │ Decode     │  │ Decode     │
     │ Worker 0   │  │ Worker 1   │  │ Worker 2   │
     │ (4×A100)   │  │ (4×A100)   │  │ (4×A100)   │
     └────────────┘  └────────────┘  └────────────┘
```

### 2.2 三个核心技术挑战

PD分离看似简单，但在工程实现中面临三个核心挑战：

**挑战一：KV Cache传输**

Prefill计算完的KV Cache需要传输给Decode节点。对于一个70B模型、4K上下文的请求：

```
KV Cache大小计算：
  layers × 2(K+V) × seq_len × hidden_dim × dtype
  
  以LLaMA-70B为例（80层，8192 hidden，FP16）：
  80 × 2 × 4096 × 8192 × 2 bytes = 10.7 GB
  
  这是一个请求的KV Cache！
```

10.7GB通过PCIe（~64 GB/s）传输需要约170ms。在200Gbps RDMA网络上，约430ms。这还不算打包和序列化的开销。

**挑战二：调度均衡**

Prefill和Decode的工作负载模式不同，需要独立的负载均衡策略：

```
传统均衡：  按请求数均衡
PD分离均衡：按计算量均衡

Prefill的均衡：
  - 按输入token数分配（而非请求数）
  - 一个10K token的请求 ≈ 10个1K token的请求
  - 需要考虑：输入长度分布、模型的chunk size

Decode的均衡：
  - 按活跃序列数分配（KV Cache占用）
  - 考虑：最大并发数、输出长度预期
```

**挑战三：容错与弹性**

当Prefill或Decode节点故障时：

```
场景1: Prefill节点故障
  - 影响：新请求无法预填充
  - 恢复：重启节点，重新加载模型
  - 对策：Prefill集群保持冗余

场景2: Decode节点故障
  - 影响：正在生成的请求中断
  - 恢复：需要从最近的checkpoint恢复
  - 对策：定期checkpoint + KV Cache备份

场景3: 网络拥塞
  - 影响：KV Cache传输延迟增加
  - 恢复：自动重试 + 备选路径
  - 对策：RDMA网络 + 流量控制
```

---

## 三、主流方案深度对比

### 3.1 方案全景

2024-2025年，PD分离领域涌现了多个重要方案：

| 方案 | 提出者 | 核心特点 | 开源情况 |
|-----|--------|---------|---------|
| Splitwise | Microsoft | 首个系统性PD分离方案 | 部分开源 |
| DistServe | UC Berkeley | 专注KV Cache传输优化 | 开源 |
| Mooncake | Moonshot AI | 生产级KV Cache Pool架构 | 开源 |
| TetriInfer | ByteDance | 异构硬件适配 | 论文 |
| DCoLLM | - | 动态集群分配 | 论文 |

### 3.2 Splitwise：开创者

Splitwise是微软在2024年提出的PD分离方案，其核心贡献是**系统性地论证了PD分离的可行性**。

**核心设计决策：**

```
Splitwise的架构选择：
1. Prefill节点使用高算力GPU (如H100 SXM)
   → 最大化prefill吞吐
   
2. Decode节点使用高带宽GPU (如A100)
   → KV Cache传输更高效
   
3. KV Cache通过NVLink + InfiniBand传输
   → 跨节点低延迟
```

**性能数据（来自论文）：**

| 指标 | 传统架构 | Splitwise | 提升 |
|-----|---------|-----------|------|
| TTFT (4K输入) | 384ms | 156ms | 2.5× |
| TPOT | 23ms | 18ms | 1.3× |
| 吞吐 | 基准 | +1.5× | - |
| 成本效率 | 基准 | +1.3× | - |

### 3.3 DistServe：KV Cache传输优化

DistServe专注于解决PD分离中最关键的性能瓶颈——KV Cache传输。

**核心技术：P2P KV Cache Transfer**

```
传统方式：KV Cache → CPU内存 → 网络 → CPU内存 → GPU显存
           ↑ 多次拷贝，延迟高

DistServe方式：KV Cache → RDMA直接传输 → GPU显存
               ↑ 零拷贝(RDMA)，延迟低
               
关键技术：
1. GPUDirect RDMA：GPU显存直接与网卡通信
2. Pipeline传输：边计算边传输，隐藏延迟
3. 压缩传输：对KV Cache进行量化压缩后再传输
```

**压缩传输的效果：**

```
传输原始KV Cache (FP16):
  4K context × 80层 × 2 × 8192 × 2B = 10.7 GB
  
传输压缩后KV Cache (INT4):
  4K context × 80层 × 2 × 8192 × 0.5B = 2.7 GB
  
传输延迟(200Gbps RDMA):
  原始: ~430ms
  压缩: ~108ms
  ↓ 75%
```

### 3.4 Mooncake：生产级KV Cache Pool

Mooncake是月之暗面(Moonshot AI)开源的PD分离方案，其最大特点是引入了**独立的KV Cache Pool**。

**架构创新：三池分离**

```
Mooncake架构：

┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│ Prefill Pool │ ←→  │ KV Cache Pool   │ ←→  │ Decode Pool  │
│ (计算密集)   │     │ (存储密集)       │     │ (访存密集)   │
│ H100集群     │     │ 大容量NVMe/OBJ  │     │ A100集群     │
└─────────────┘     └─────────────────┘     └─────────────┘

KV Cache Pool的介质层级：
  L1: GPU HBM (最快，最贵)
  L2: GPU DRAM (中等)
  L3: NVMe SSD (较慢，便宜)
  L4: 对象存储 (最慢，最便宜)
```

**关键设计：KV Cache的分层存储**

```python
# Mooncake的KV Cache分层管理（伪代码）
class KVCachePool:
    def __init__(self):
        self.l1_cache = GPUMemoryPool(max_size="200GB")  # HBM
        self.l2_cache = DRAMPool(max_size="1TB")          # CPU内存
        self.l3_cache = NVMePool(max_size="10TB")         # SSD
        
    def store(self, request_id, kv_cache):
        """写入时：先存快速层，异步降级"""
        self.l1_cache.put(request_id, kv_cache)
        self.async_promote_to_l2(request_id)
        
    def fetch(self, request_id):
        """读取时：优先L1，miss则逐级查找"""
        if self.l1_cache.has(request_id):
            return self.l1_cache.get(request_id)
        elif self.l2_cache.has(request_id):
            data = self.l2_cache.get(request_id)
            self.l1_cache.put(request_id, data)  # 回填L1
            return data
        elif self.l3_cache.has(request_id):
            data = self.l3_cache.get(request_id)
            self.l1_cache.put(request_id, data)  # 回填L1
            return data
        else:
            raise KVCacheMiss(request_id)
```

**Mooncake的性能数据：**

| 指标 | 传统vLLM | Mooncake | 提升 |
|-----|---------|----------|------|
| 吞吐 | 基准 | 3× | 3× |
| TTFT P99 | 基准 | 0.4× | 2.5× 降低 |
| GPU利用率 | 35% | 65% | 1.9× |
| 每百万token成本 | 基准 | 0.35× | 65% 降低 |

---

## 四、工程实现深度指南

### 4.1 部署架构设计

**集群规模规划**

```yaml
# 典型PD分离集群配置
prefill_cluster:
  nodes: 4
  gpus_per_node: 8
  gpu_type: "H100-SXM-80G"
  model: "DeepSeek-V3-671B"
  tp_size: 8  # 每个节点内tensor parallelism
  dp_size: 4  # 跨节点data parallelism
  max_concurrent_prefill: 64
  chunk_size: 8192  # Prefill分块大小

decode_cluster:
  nodes: 4
  gpus_per_node: 8
  gpu_type: "A100-80G"
  model: "DeepSeek-V3-671B"
  tp_size: 8
  dp_size: 4
  max_concurrent_decode: 512
  
kv_cache_pool:
  nodes: 2
  storage: "NVMe SSD RAID0"
  capacity: "20TB"
  network: "200Gbps RDMA"

router:
  nodes: 2  # 高可用
  algorithm: "weighted-round-robin"
  health_check_interval: "5s"
```

### 4.2 KV Cache传输优化实战

**1. 分块传输策略**

```
全量传输（不推荐）：
  Prefill完成 → 打包全部KV Cache → 传输 → Decode开始
  延迟 = Prefill时间 + 传输时间 + Decode开始时间
  
分块传输（推荐）：
  Prefill chunk 1 → 传输 chunk 1 → Decode开始处理 chunk 1
  Prefill chunk 2 → 传输 chunk 2 → Decode等待 chunk 2
  ...
  延迟 = Prefill时间 + max(传输延迟, Decode首步延迟)
  
  关键：分块大小要匹配Decode的处理速度
```

**2. KV Cache压缩**

```python
# FP8 KV Cache压缩示例
import torch

def compress_kv_cache_fp8(kv_cache):
    """
    将FP16 KV Cache压缩为FP8
    压缩比: 2:1
    质量损失: < 0.5% (在大多数任务上)
    """
    # 逐层量化
    for layer_idx in range(len(kv_cache)):
        k = kv_cache[layer_idx]['k']  # [batch, heads, seq, dim]
        v = kv_cache[layer_idx]['v']
        
        # 对每个head独立量化
        for head_idx in range(k.shape[1]):
            k_head = k[:, head_idx, :, :]
            v_head = v[:, head_idx, :, :]
            
            # FP8量化
            k_scale = k_head.abs().max() / 448.0  # FP8 max
            v_scale = v_head.abs().max() / 448.0
            
            k_fp8 = (k_head / k_scale).to(torch.float8_e4m3fn)
            v_fp8 = (v_head / v_scale).to(torch.float8_e4m3fn)
            
            kv_cache[layer_idx]['k'][:, head_idx, :, :] = k_fp8
            kv_cache[layer_idx]['v'][:, head_idx, :, :] = v_fp8
            kv_cache[layer_idx]['k_scale'] = k_scale
            kv_cache[layer_idx]['v_scale'] = v_scale
    
    return kv_cache
```

### 4.3 调度策略

**请求路由决策流程：**

```
收到请求
    │
    ▼
估算输入长度
    │
    ├─ 短输入 (< 512 tokens)
    │   └─ 路由到 Prefill节点 A（负载最低）
    │
    ├─ 中输入 (512-4K tokens)  
    │   └─ 路由到 Prefill节点 B（当前利用率最低）
    │
    └─ 长输入 (> 4K tokens)
        └─ 路由到 Prefill节点 C（剩余显存最多）
    
    │
    ▼
Prefill完成
    │
    ▼
根据输出长度预估路由到Decode节点
    │
    ├─ 短输出 (< 512 tokens)
    │   └─ 路由到 Decode节点 X（可复用短序列池）
    │
    └─ 长输出 (> 512 tokens)
        └─ 路由到 Decode节点 Y（剩余KV Cache空间最多）
```

### 4.4 监控与告警

```yaml
# PD分离架构的关键监控指标
monitoring:
  prefill_metrics:
    - name: "prefill_throughput"
      alert: "avg_5m < 100"  # 每秒处理请求 < 100
    - name: "prefill_latency_p99"  
      alert: "avg_5m > 2000"  # P99延迟 > 2s
    - name: "prefill_gpu_utilization"
      alert: "avg_5m < 0.3"  # GPU利用率 < 30%
  
  decode_metrics:
    - name: "decode_throughput"
      alert: "avg_5m < 50"
    - name: "decode_tpot_p99"
      alert: "avg_5m > 50"  # 每token延迟 > 50ms
    - name: "active_sequences"
      alert: "current > max_capacity * 0.9"  # 接近满载
  
  kv_cache_metrics:
    - name: "kv_cache_transfer_latency"
      alert: "avg_5m > 100"  # 传输延迟 > 100ms
    - name: "kv_cache_pool_usage"
      alert: "current > 0.85"  # 缓存池使用率 > 85%
    - name: "kv_cache_hit_rate"
      alert: "avg_5m < 0.7"  # 命中率 < 70%
  
  network_metrics:
    - name: "rdma_packet_loss"
      alert: "avg_5m > 0.001"  # 丢包率 > 0.1%
    - name: "inter_node_bandwidth"
      alert: "avg_5m < 150"  # 带宽 < 150Gbps (200Gbps的75%)
```

---

## 五、PD分离 vs 传统架构：何时选择？

### 5.1 决策框架

```
是否选择PD分离？

Q1: 日均请求量 > 10万？
    ├─ 否 → 传统架构可能更经济
    └─ 是 → 继续

Q2: TTFT要求 < 500ms？
    ├─ 否 → 传统架构 + 优化可能足够
    └─ 是 → 继续

Q3: 输入长度波动大（方差高）？
    ├─ 否 → 传统架构 + 静态优化可能够用
    └─ 是 → PD分离优势明显

Q4: 有GPU集群 + 高速网络基础设施？
    ├─ 否 → 部署成本太高，不建议
    └─ 是 → PD分离是正确选择

Q5: 团队有分布式系统经验？
    ├─ 否 → 建议从Mooncake等成熟方案开始
    └─ 是 → 可以考虑自研或深度定制
```

### 5.2 成本分析

| 维度 | 传统架构 | PD分离 | 差异原因 |
|-----|---------|--------|---------|
| GPU数量 | 基准N | 约1.3N | 需要独立Prefill和Decode集群 |
| GPU利用率 | 30-40% | 55-70% | 计算类型匹配，利用率提升 |
| 单请求成本 | 基准 | 0.6-0.8× | 利用率提升抵消了GPU增加 |
| 吞吐/美元 | 基准 | 1.2-1.5× | 综合成本效率提升 |
| 运维复杂度 | 低 | 中-高 | 多集群管理、网络调试 |
| 人才需求 | 一般 | 较高 | 需要分布式系统+AI Infra经验 |

### 5.3 典型适用场景

| 场景 | 适合PD分离？ | 理由 |
|-----|-------------|------|
| 高并发API服务 | ✅ 是 | 吞吐和延迟都能优化 |
| 长上下文RAG | ✅ 是 | 长Prefill和短Decode天然适合分离 |
| 交互式聊天 | ⚠️ 视情况 | 如果延迟要求高则适合 |
| 批量离线推理 | ❌ 否 | 可以容忍延迟，传统架构更简单 |
| 多模态推理 | ✅ 是 | 多模态Prefill计算量大，更适合分离 |

---

## 六、未来演进

### 6.1 技术趋势

**1. 硬件层面**

- **CXL内存扩展**：通过CXL协议实现GPU显存与CPU内存的统一寻址，KV Cache Pool可以扩展到TB级
- **Chiplet架构**：未来GPU可能原生支持Prefill和Decode的硬件级分离
- **专用KV Cache加速卡**：类似DPU的专用硬件处理KV Cache的存储和传输

**2. 软件层面**

- **自适应分离**：根据实时负载动态调整Prefill和Decode的资源比例
- **多级流水线**：Prefill → KV Cache压缩 → 传输 → KV Cache解压 → Decode，全流水线并行
- **跨集群调度**：在多个数据中心之间调度，实现全球最优的资源利用

**3. 模型层面**

- **长上下文模型的挑战**：128K+上下文的KV Cache传输量巨大，需要更激进的压缩策略
- **推理模型的特殊需求**：推理模型（如R1）的长输出导致Decode时间远大于Prefill，分离比例需要调整

### 6.2 对团队的建议

1. **先做好基础**：确保vLLM/SGLang的单机部署已经优化到位，再考虑分离
2. **从Mooncake开始**：它是目前最成熟的开源方案，文档和社区支持都很好
3. **网络是关键**：PD分离的性能很大程度取决于网络带宽和延迟，投资高速网络是值得的
4. **监控先行**：在部署前就建立完善的监控体系，PD分离的故障排查比传统架构更复杂
5. **渐进式迁移**：可以先将最耗时的长输入请求路由到PD分离集群，逐步扩大范围

---

## 七、总结

PD分离架构代表了LLM推理从「通用计算」走向「专用架构」的重要一步。它的核心洞察是：**当两种计算模式存在本质矛盾时，与其在妥协中寻找平衡，不如将它们彻底解耦**。

| 架构 | 核心思想 | 适用阶段 |
|-----|---------|---------|
| 传统一体化 | 简单统一，一个GPU处理所有 | 早期/小规模 |
| PD分离 | 计算解耦，专GPU做专事 | 规模化/高性能 |
| PD分离 + KV Cache Pool | 三级池化，弹性调度 | 超大规模/全球化 |

对于正在构建LLM推理基础设施的团队而言，PD分离不再是一个「未来可能需要」的技术——它已经是经过大规模生产验证的**当前最佳实践**。关键是选择合适的时机和方案，在正确的基础设施上迈出这一步。

> **记住：PD分离不是银弹，它解决的是LLM推理中的一个特定矛盾。在你的场景中，这个矛盾是否是主要瓶颈，决定了PD分离是否值得你投入的工程成本。**
