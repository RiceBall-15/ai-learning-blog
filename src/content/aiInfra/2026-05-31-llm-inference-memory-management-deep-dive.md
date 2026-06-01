---
title: "LLM推理系统内存管理深度解析：从CUDA显存到PagedAttention的工程实践"
description: "系统性拆解LLM推理中的内存管理难题：CUDA显存碎片化、KV Cache膨胀、Batch并发内存冲突，深入解析PagedAttention、内存池化、显存交换等核心技术，提供生产级内存优化方案"
date: "2026-05-31"
author: "RiceBall-15"
category: "aiInfra"
subCategory: inference
tags: ["LLM推理", "内存管理", "CUDA", "PagedAttention", "vLLM", "KV Cache", "显存优化", "推理优化"]
draft: false
---

# LLM推理系统内存管理深度解析：从CUDA显存到PagedAttention的工程实践

> 当你把一个70B参数的模型部署到A100-80G上，发现只能同时服务3个用户；当你看到GPU利用率只有40%但系统却报OOM；当你发现KV Cache吃掉了比模型本身还多的显存——这些都是LLM推理系统内存管理的真实痛点。LLM推理的内存问题不同于传统Web服务，它涉及模型权重、KV Cache、激活值三类内存的复杂交互，而其中KV Cache的动态增长特性使得内存管理成为推理系统的核心技术挑战。本文将从CUDA显存的底层原理出发，系统性地拆解LLM推理内存管理的完整技术栈。

---

## 一、LLM推理的内存全景：你以为的"放不下"到底卡在哪？

### 1.1 三类内存的此消彼长

LLM推理系统的内存可以分为三个核心区域，它们之间的配比直接决定了系统的吞吐量和并发能力：

```
┌────────────────────────────────────────────────────────────────────┐
│                    LLM推理系统的内存三区模型                         │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │              GPU 总显存 (如 A100-80GB)                     │     │
│  │                                                          │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │     │
│  │  │  模型权重     │  │   KV Cache   │  │   激活值/临时   │ │     │
│  │  │  (静态)       │  │  (动态增长)   │  │   (计算中间态)  │ │     │
│  │  │              │  │              │  │                │ │     │
│  │  │ 7B: ~14GB    │  │  每个token   │  │  取决于batch    │ │     │
│  │  │ 70B: ~140GB  │  │  约1-2MB     │  │  size和seq_len  │ │     │
│  │  │ (FP16)       │  │  (取决于模型) │  │                │ │     │
│  │  │              │  │              │  │                │ │     │
│  │  │  固定不动     │  │  随请求动态   │  │  每步计算后     │ │     │
│  │  │              │  │  分配和释放   │  │  可复用         │ │     │
│  │  └──────────────┘  └──────────────┘  └────────────────┘ │     │
│  │                                                          │     │
│  │  剩余空间 = 可用于KV Cache + 激活值的空间                   │     │
│  │  目标：最大化这部分空间的利用效率                            │     │
│  │                                                          │     │
│  └──────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
```

### 1.2 模型权重的显存占用计算

模型权重是推理系统中最"确定"的内存消耗。理解它的计算方式是做内存规划的基础：

```
模型显存占用计算公式：

显存(字节) = 参数量 × 每参数字节数

┌──────────────┬──────────┬──────────┬──────────┬──────────────┐
│ 模型          │ 参数量    │ FP32     │ FP16/BF16│ INT8/AWQ     │
├──────────────┼──────────┼──────────┼──────────┼──────────────┤
│ LLaMA-2 7B   │ 6.7B     │ 26.8 GB  │ 13.4 GB  │ 6.7 GB       │
│ LLaMA-2 13B  │ 13.0B    │ 52.0 GB  │ 26.0 GB  │ 13.0 GB      │
│ LLaMA-2 70B  │ 70.0B    │ 280 GB   │ 140 GB   │ 70 GB        │
│ DeepSeek-V3  │ 671B*    │ 2684 GB  │ 1342 GB  │ 671 GB       │
│ Qwen-2.5 72B │ 72.0B    │ 288 GB   │ 144 GB   │ 72 GB        │
└──────────────┴──────────┴──────────┴──────────┴──────────────┘

* DeepSeek-V3为MoE模型，激活参数仅37B，实际激活显存远小于671B全量
```

**量化对显存的影响——不是线性关系：**

```
以LLaMA-2 70B为例：

FP16 (140GB) ──→ INT8 (70GB) ──→ INT4 (35GB)
     ↓ 50%           ↓ 50%
   显存减半         再减半

但实际推理中：
┌──────────────────────────────────────────────┐
│ FP16: 140GB权重 + 20GB KV Cache = 160GB       │
│   → A100-80G: ❌ 无法部署                      │
│   → 2×A100: ✅ 但显存利用率仅 50%              │
│                                               │
│ INT8: 70GB权重 + 20GB KV Cache = 90GB          │
│   → 2×A100: ✅ 显存利用率 56%                   │
│                                               │
│ INT4: 35GB权重 + 20GB KV Cache = 55GB          │
│   → 1×A100: ✅ 显存利用率 69%                   │
│   → 剩余25GB可服务更多并发请求                   │
│                                               │
│ 💡 量化不仅降低部署门槛，更关键的是释放了          │
│    更多空间给KV Cache，提升并发能力               │
└──────────────────────────────────────────────┘
```

### 1.3 KV Cache——推理系统的"内存黑洞"

KV Cache是LLM推理中最具挑战性的内存管理对象。它的大小随序列长度线性增长，且在多用户并发场景下会迅速膨胀：

```
KV Cache 显存占用计算（Multi-Head Attention）：

单个token的KV Cache大小 = 2 × num_layers × num_heads × head_dim × dtype_bytes

以LLaMA-2 70B为例 (80层, 64头, 128维, FP16):
= 2 × 80 × 64 × 128 × 2 bytes
= 2,621,440 bytes ≈ 2.5 MB per token

┌─────────────────────────────────────────────────────────────────┐
│            LLaMA-2 70B KV Cache并发占用一览                      │
│                                                                 │
│  序列长度    │ 1用户    │ 10用户   │ 32用户   │ 64用户   │ 128用户  │
│  ───────────┼──────────┼──────────┼──────────┼──────────┼─────────│
│  512 tokens │ 1.25 GB  │ 12.5 GB  │ 40 GB    │ 80 GB    │ 160 GB  │
│  1K tokens  │ 2.5 GB   │ 25 GB    │ 80 GB    │ 160 GB   │ 320 GB  │
│  2K tokens  │ 5 GB     │ 50 GB    │ 160 GB   │ 320 GB   │ 640 GB  │
│  4K tokens  │ 10 GB    │ 100 GB   │ 320 GB   │ 640 GB   │ 1.28 TB │
│  8K tokens  │ 20 GB    │ 200 GB   │ 640 GB   │ 1.28 TB  │ 2.56 TB │
│                                                                 │
│  ⚠️ 注意：2K序列长度下，128并发用户的KV Cache就需要640GB          │
│     这就是为什么"能部署"和"能并发服务"是两回事                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 内存碎片化——被低估的隐形杀手

即使用完了所有技术优化，内存碎片化依然会让你的实际可用空间远小于理论值：

```
内存碎片化的三种类型：

1. 内部碎片 (Internal Fragmentation)
   ────────────────────────────────
   预分配的内存块大于实际需要
   
   分配: [████████████] 1024 tokens
   使用: [██████░░░░░░] 600 tokens
   浪费: 40% 的预分配空间
   
2. 外部碎片 (External Fragmentation)
   ────────────────────────────────
   空闲内存总量足够，但不连续
   
   已分配: [████][████][░░][████][░░][████]
   空闲:        16KB       8KB      16KB
   需求: 20KB连续内存 → ❌ 无法分配（虽然空闲总量40KB）
   
3. KV Cache增长碎片
   ────────────────
   请求结束释放的KV Cache留下不规则空洞
   
   时间T1: [用户A KV][用户B KV][用户C KV][空闲]
   时间T2: [用户A KV][空闲    ][用户C KV][用户D KV]
                                         ↑ 新用户D被迫放在末尾
   时间T3: [用户A KV][用户E KV][空闲    ][用户D KV]
                                         ↑ 用户C结束，但空洞无法
                                           被新请求有效利用
```

---

## 二、PagedAttention：LLM内存管理的范式革命

### 2.1 传统KV Cache管理的困境

在PagedAttention出现之前，KV Cache的管理方式通常有两种，都存在严重问题：

```
方案一：预分配（Pre-allocation）
────────────────────────────────
为每个请求预分配最大序列长度的连续显存

请求1: [████████████████████████████] (预分配4K tokens空间)
实际用: [██████░░░░░░░░░░░░░░░░░░░░░░] (实际只用了1K tokens)
浪费:   75% 的预分配空间

问题：
• 无法预知用户实际会输入多长
• 预分配过大 → 浪费严重
• 预分配过小 → 需要重新分配（代价极高）
• 不同请求的KV Cache无法共享内存页


方案二：动态分配（Malloc-based）
───────────────────────────────
用CUDA malloc按需分配

问题：
• CUDA malloc本身开销大（~10μs，对比GPU计算是"永恒"）
• 碎片化严重：频繁分配释放产生大量内存空洞
• 无法做内存碎片整理（GPU不支持defragment）
• 最终导致：明明有足够空闲显存，却无法分配连续空间
```

### 2.2 PagedAttention的核心思想

PagedAttention借鉴了操作系统虚拟内存的分页机制，将KV Cache的管理从"连续内存分配"转变为"分页管理"：

```
┌─────────────────────────────────────────────────────────────────┐
│                PagedAttention 核心思想                           │
│                                                                 │
│  传统方式：连续内存分配                                            │
│  ─────────────────────                                          │
│  请求1的KV Cache: [████████████████████] 一整块连续显存           │
│  请求2的KV Cache: [████████████] 一整块连续显存                   │
│  问题：碎片化、浪费、不可扩展                                      │
│                                                                 │
│  PagedAttention：分页管理                                        │
│  ──────────────────                                             │
│  物理显存被划分为固定大小的"页"（Page）                             │
│  每个Page存储固定数量的Token的KV Cache                             │
│  请求的KV Cache通过"页表"映射到物理页                              │
│                                                                 │
│  物理显存:                                                       │
│  [Page0][Page1][Page2][Page3][Page4][Page5][Page6][Page7]       │
│    ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓         │
│  请求1   请求1   请求2   请求1   请求3   请求2   空闲   空闲        │
│  (非连续分布，通过页表管理)                                        │
│                                                                 │
│  关键突破：                                                       │
│  • 按需分配：用多少页分配多少页，不预分配                           │
│  • 非连续存储：请求的KV Cache可以分散在不同物理页                   │
│  • 零碎片：页是固定大小的，不存在外部碎片                           │
│  • 即时释放：请求结束立即释放页，可被新请求复用                     │
│  • 共享页：相同前缀的请求可以共享物理页（Prefix Caching）           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 PagedAttention的数据结构设计

```python
"""
PagedAttention核心数据结构的Python模拟
展示页表、块管理和内存池的设计思路
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque

@dataclass
class KVCacheBlock:
    """KV Cache物理块"""
    block_id: int
    capacity: int          # 可存储的token数量
    token_count: int = 0   # 当前已存储的token数量
    ref_count: int = 0     # 引用计数（用于Copy-on-Write）
    
    def is_full(self) -> bool:
        return self.token_count >= self.capacity
    
    def remaining(self) -> int:
        return self.capacity - self.token_count

@dataclass
class SequenceState:
    """单个请求的序列状态"""
    seq_id: int
    block_table: List[int] = field(default_factory=list)  # 逻辑块→物理块映射
    seq_length: int = 0
    max_length: int = 2048
    
    def logical_blocks(self) -> int:
        """需要的逻辑块数"""
        return (self.seq_length + self.capacity - 1) // self.capacity

class PagedAttentionMemoryManager:
    """
    PagedAttention内存管理器
    管理KV Cache的分页分配、释放和共享
    """
    
    def __init__(self, total_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.total_blocks = total_blocks
        
        # 物理块池
        self.blocks: Dict[int, KVCacheBlock] = {
            i: KVCacheBlock(block_id=i, capacity=block_size)
            for i in range(total_blocks)
        }
        
        # 空闲块队列
        self.free_blocks: deque = deque(range(total_blocks))
        
        # 活跃序列
        self.sequences: Dict[int, SequenceState] = {}
        
        # 统计信息
        self.stats = {
            "allocated": 0,
            "freed": 0,
            "shared_copies": 0,
            "oom_events": 0,
        }
    
    def allocate_sequence(self, seq_id: int, max_length: int = 2048) -> bool:
        """分配新序列"""
        seq = SequenceState(
            seq_id=seq_id,
            max_length=max_length,
            capacity=self.block_size,
        )
        self.sequences[seq_id] = seq
        
        # 分配第一个块
        return self._grow_sequence(seq)
    
    def _grow_sequence(self, seq: SequenceState) -> bool:
        """为序列增长分配新的物理块"""
        if not self.free_blocks:
            self.stats["oom_events"] += 1
            return False
        
        block_id = self.free_blocks.popleft()
        block = self.blocks[block_id]
        block.ref_count = 1
        block.token_count = 0
        
        seq.block_table.append(block_id)
        self.stats["allocated"] += 1
        return True
    
    def append_token(self, seq_id: int) -> bool:
        """向序列追加一个token的KV Cache"""
        seq = self.sequences[seq_id]
        
        # 检查最后一个块是否已满
        if seq.block_table:
            last_block = self.blocks[seq.block_table[-1]]
            if not last_block.is_full():
                last_block.token_count += 1
                seq.seq_length += 1
                return True
        
        # 需要分配新块
        if not self._grow_sequence(seq):
            return False
        
        # 在新块中写入token
        new_block = self.blocks[seq.block_table[-1]]
        new_block.token_count = 1
        seq.seq_length += 1
        return True
    
    def free_sequence(self, seq_id: int):
        """释放序列的所有块"""
        seq = self.sequences.pop(seq_id)
        
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            
            if block.ref_count == 0:
                self.free_blocks.append(block_id)
                block.token_count = 0
                self.stats["freed"] += 1
            else:
                # 还有其他序列引用（Copy-on-Write场景）
                pass
    
    def copy_on_write(self, seq_id: int, fork_seq_id: int):
        """
        Copy-on-Write：为序列分叉创建共享页
        用于Beam Search等场景，多个候选共享前缀的KV Cache
        """
        source = self.sequences[seq_id]
        
        forked = SequenceState(
            seq_id=fork_seq_id,
            block_table=source.block_table.copy(),
            seq_length=source.seq_length,
            max_length=source.max_length,
            capacity=self.block_size,
        )
        self.sequences[fork_seq_id] = forked
        
        # 增加引用计数
        for block_id in forked.block_table:
            self.blocks[block_id].ref_count += 1
        
        self.stats["shared_copies"] += 1
    
    def get_utilization(self) -> dict:
        """获取内存利用率统计"""
        used = self.total_blocks - len(self.free_blocks)
        
        # 计算实际使用的token vs 分配的块容量
        total_token_capacity = 0
        total_tokens_used = 0
        for block_id in range(self.total_blocks):
            block = self.blocks[block_id]
            if block.ref_count > 0:
                total_token_capacity += block.capacity
                total_tokens_used += block.token_count
        
        internal_fragmentation = (
            1 - total_tokens_used / total_token_capacity
            if total_token_capacity > 0 else 0
        )
        
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": used,
            "free_blocks": len(self.free_blocks),
            "block_utilization": f"{used / self.total_blocks * 100:.1f}%",
            "internal_fragmentation": f"{internal_fragmentation * 100:.1f}%",
            "active_sequences": len(self.sequences),
            "stats": self.stats,
        }
```

### 2.4 PagedAttention的实际效果

```
传统方式 vs PagedAttention 对比（LLaMA-2 70B, A100-80G×4）:

┌──────────────────────┬────────────────┬────────────────┐
│ 指标                  │ 传统预分配      │ PagedAttention  │
├──────────────────────┼────────────────┼────────────────┤
│ 最大并发数（4K seq）   │ ~8 用户        │ ~32 用户        │
│ 显存利用率            │ 35-50%         │ 85-95%         │
│ 内存碎片率            │ 30-60%         │ < 5%           │
│ 新请求分配延迟        │ 50-200μs       │ < 1μs          │
│ 请求结束释放延迟      │ 10-50μs        │ < 0.1μs        │
│ 前缀共享支持          │ ❌ 不支持       │ ✅ 原生支持     │
│ 长序列处理            │ 需要预分配大块  │ 按需增长       │
└──────────────────────┴────────────────┴────────────────┘

关键数字：PagedAttention将显存利用率从~40%提升到~90%
意味着同样的硬件可以服务2x以上的并发请求
```

---

## 三、KV Cache优化技术全景

### 3.1 KV Cache压缩技术对比

PagedAttention解决了内存管理效率问题，但KV Cache本身的大小依然是瓶颈。多种压缩技术可以从不同维度减小KV Cache：

```
KV Cache压缩技术矩阵

┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ 技术              │ 压缩比    │ 质量损失  │ 速度影响  │ 硬件要求  │ 成熟度    │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ GQA/MQA          │ 2-8x     │ 极小     │ 无       │ 通用     │ ⭐⭐⭐⭐⭐ │
│ (分组/多查询注意力) │          │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ MLA              │ 5-10x    │ 极小     │ 无       │ 通用     │ ⭐⭐⭐⭐   │
│ (Multi-head Latent)│         │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ KV Cache量化      │ 2-4x     │ 小       │ 轻微     │ FP8硬件  │ ⭐⭐⭐⭐   │
│ (FP8/INT4)       │          │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Token淘汰         │ 2-5x     │ 中       │ 无       │ 通用     │ ⭐⭐⭐    │
│ (H2O/SnapKV)     │          │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ KV Cache共享      │ N/A      │ 无       │ 无       │ 通用     │ ⭐⭐⭐⭐   │
│ (Prefix Caching)  │          │          │          │          │          │
└──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### 3.2 GQA/MQA——架构层面的根本解法

分组查询注意力（Grouped-Query Attention）和多查询注意力（Multi-Query Attention）是从模型架构层面减少KV Cache的最优雅方案：

```
标准MHA vs MQA vs GQA 的KV Cache对比

┌─────────────────────────────────────────────────────────────┐
│ 标准 Multi-Head Attention (MHA)                              │
│                                                             │
│  Q: [h1][h2][h3][h4][h5][h6][h7][h8]  8个头                 │
│  K: [k1][k2][k3][k4][k5][k6][k7][k8]  8组KV Cache          │
│  V: [v1][v2][v3][v4][v5][v6][v7][v8]  8组                  │
│                                                             │
│  KV Cache大小: 8 × (dim/8) × 2 = 2 × dim                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Multi-Query Attention (MQA)                                  │
│                                                             │
│  Q: [h1][h2][h3][h4][h5][h6][h7][h8]  8个头                 │
│  K: [k1]                              1组KV Cache           │
│  V: [v1]                              1组                   │
│                                                             │
│  所有Q头共享同一组KV                                          │
│  KV Cache大小: 1 × (dim/8) × 2 = 0.25 × dim                │
│  压缩比: 8x                                                  │
│                                                             │
│  代价: 质量有轻微下降                                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Grouped-Query Attention (GQA)                                │
│                                                             │
│  Q: [h1][h2][h3][h4][h5][h6][h7][h8]  8个头                 │
│  K: [k1][k2]                          2组KV Cache           │
│  V: [v1][v2]                          2组                   │
│                                                             │
│  每4个Q头共享一组KV                                           │
│  KV Cache大小: 2 × (dim/8) × 2 = 0.5 × dim                 │
│  压缩比: 4x                                                  │
│                                                             │
│  代价: 质量损失极小（实际效果几乎无损）                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

实际模型采用情况：
• LLaMA-2 70B: GQA (8 KV heads, 64 Q heads → 8x压缩)
• Mistral 7B: GQA (8 KV heads, 32 Q heads → 4x压缩)
• DeepSeek-V2/V3: MLA (更激进的压缩，5-10x)
• Qwen-2.5 72B: GQA
```

### 3.3 KV Cache量化——精度换空间

```python
"""
KV Cache量化的实现方案对比
从FP16到INT4的量化路径
"""
import numpy as np

class KVCacheQuantizer:
    """
    KV Cache量化器
    支持多种量化精度：FP8, INT8, INT4
    """
    
    @staticmethod
    def quantize_fp8(kv_cache: np.ndarray) -> tuple:
        """
        FP8量化（E4M3格式）
        最常见的KV Cache量化方案
        """
        # 找到per-tensor或per-channel的scale
        abs_max = np.max(np.abs(kv_cache))
        scale = abs_max / 448.0  # FP8 E4M3最大值
        
        quantized = np.clip(
            np.round(kv_cache / scale), 
            -448, 448
        ).astype(np.float8_e4m3fn)  # 实际需CUDA kernel
        
        return quantized, scale
    
    @staticmethod
    def dequantize_fp8(quantized, scale):
        """FP8反量化"""
        return quantized.astype(np.float16) * scale
    
    @staticmethod
    def quantize_int4_groupwise(
        kv_cache: np.ndarray, 
        group_size: int = 128
    ) -> tuple:
        """
        INT4分组量化
        每group_size个元素共享一个scale和zero_point
        更精细的量化，质量更好
        """
        original_shape = kv_cache.shape
        # reshape为group
        n_groups = kv_cache.size // group_size
        grouped = kv_cache.reshape(n_groups, group_size)
        
        # 每组独立量化
        mins = grouped.min(axis=1, keepdims=True)
        maxs = grouped.max(axis=1, keepdims=True)
        
        scales = (maxs - mins) / 15.0  # INT4: 0-15
        zero_points = (-mins / scales).round()
        
        quantized = np.clip(
            np.round(grouped / scales + zero_points),
            0, 15
        ).astype(np.uint8)  # 4bit packed
        
        return quantized, scales, zero_points

# 量化效果对比
"""
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 量化精度       │ 压缩比        │ 质量影响      │ 额外计算开销  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ FP16 (基线)   │ 1x           │ 无           │ 无           │
│ FP8          │ 2x           │ <0.1% perplexity增加 │ 微小  │
│ INT8         │ 2x           │ <0.2% perplexity增加 │ 微小  │
│ INT4 (group) │ 4x           │ 0.5-2% perplexity增加│ 需反量化│
│ INT4 (per-tensor) │ 4x      │ 1-3% perplexity增加 │ 需反量化│
└──────────────┴──────────────┴──────────────┴──────────────┘

生产建议：FP8量化是性价比最高的选择
• 2x压缩，质量损失几乎不可感知
• 大多数GPU硬件原生支持
• vLLM/SGLang已集成支持
"""
```

### 3.4 Token淘汰——智能缩减KV Cache

当序列长度超出KV Cache容量限制时，需要智能地淘汰不重要的token：

```
Token淘汰策略对比

┌──────────────────────────────────────────────────────────────┐
│ 策略              │ 核心思想           │ 适用场景              │
├──────────────────┼────────────────────┼──────────────────────┤
│ H2O              │ 按累积注意力分数    │ 通用，效果均衡        │
│ (Heavy-Hitter    │ 淘汰低分token       │                      │
│  Oracle)         │                    │                      │
├──────────────────┼────────────────────┼──────────────────────┤
│ SnapKV           │ 保留注意力模式      │ 长文档理解            │
│                  │ 的"快照"关键token  │                      │
├──────────────────┼────────────────────┼──────────────────────┤
│ StreamingLLM     │ 保留开头+最近的     │ 流式对话              │
│                  │ "attention sink"  │                      │
├──────────────────┼────────────────────┼──────────────────────┤
│ Dynamic NTK      │ 按距当前token的     │ 超长序列              │
│                  │ 距离加权保留        │                      │
└──────────────────┴────────────────────┴──────────────────────┘

Attention Sink现象：
┌──────────────────────────────────────────────────┐
│ 注意力分数分布 (LLaMA-2 7B, 4K context)           │
│                                                  │
│  Token位置 →  1   50  100 ... 3900 3950 4000     │
│  注意力分数:                                              │
│  ████████   高                                       │
│  █          低                                       │
│  █          低                                       │
│  ...                                                  │
│  █          低                                       │
│  ███        中高  ← 最近的token注意力更高              │
│  ████       高                                       │
│                                                  │
│  关键发现: 第一个token总是获得异常高的注意力分数          │
│  原因: attention sink效应——模型将"无处安放"的            │
│        注意力集中到序列开头                            │
│                                                  │
│  StreamingLLM: 始终保留前4个 + 最近N个token          │
└──────────────────────────────────────────────────┘
```

---

## 四、生产级内存管理系统设计

### 4.1 内存管理架构全景

```
┌──────────────────────────────────────────────────────────────────┐
│           生产级LLM推理内存管理架构                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    调度层 (Scheduler)                        │ │
│  │  • 请求排队与优先级管理                                       │ │
│  │  • 内存预检查：请求进入前评估是否可分配                        │ │
│  │  • 抢占策略：低优先级请求在内存不足时被换出                     │ │
│  └─────────────────────────┬───────────────────────────────────┘ │
│                            │                                     │
│  ┌─────────────────────────┴───────────────────────────────────┐ │
│  │                  内存池管理器 (Memory Pool)                   │ │
│  │                                                             │ │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │ │
│  │  │ Block Table│  │ Free Block │  │  Copy-on-Write       │  │ │
│  │  │ Manager    │  │ Allocator  │  │  Manager             │  │ │
│  │  │            │  │            │  │                      │  │ │
│  │  │ 逻辑→物理  │  │ 空闲块管理  │  │ 共享页管理            │  │ │
│  │  │ 页表映射   │  │ 首次适配    │  │ 引用计数              │  │ │
│  │  │            │  │            │  │ 写时复制              │  │ │
│  │  └────────────┘  └────────────┘  └──────────────────────┘  │ │
│  │                                                             │ │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │ │
│  │  │ Swap       │  │ Quantize   │  │  Eviction            │  │ │
│  │  │ Manager    │  │ Manager    │  │  Policy              │  │ │
│  │  │            │  │            │  │                      │  │ │
│  │  │ GPU↔CPU    │  │ FP8/INT4   │  │ H2O / LRU / 优先级   │  │ │
│  │  │ 显存交换   │  │ KV Cache   │  │ 淘汰                 │  │ │
│  │  │            │  │ 量化       │  │                      │  │ │
│  │  └────────────┘  └────────────┘  └──────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│  ┌─────────────────────────┴───────────────────────────────────┐ │
│  │                    硬件抽象层 (HAL)                           │ │
│  │  • CUDA Memory Pool                                         │ │
│  │  • Pinned Memory for CPU-GPU Transfer                       │ │
│  │  • NVLink/NVSwitch for Multi-GPU                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 内存预检查与准入控制

```python
"""
LLM推理系统的内存准入控制
在请求进入GPU之前就决定是否接纳
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class AdmissionResult(Enum):
    ACCEPT = "accept"          # 接纳，有足够内存
    QUEUE = "queue"            # 排队，当前内存不足但可以等
    EVICT_AND_ACCEPT = "evict" # 驱逐低优先级后接纳
    REJECT = "reject"          # 拒绝，系统已满

@dataclass
class MemoryBudget:
    """内存预算"""
    total_blocks: int
    model_weight_blocks: int  # 模型权重占用的块数
    reserved_blocks: int = 10  # 预留紧急缓冲
    
    @property
    def available_blocks(self) -> int:
        return self.total_blocks - self.model_weight_blocks - self.reserved_blocks

class AdmissionController:
    """
    请求准入控制器
    在调度阶段完成内存预检查，避免运行时OOM
    """
    
    def __init__(self, budget: MemoryBudget, block_size: int = 16):
        self.budget = budget
        self.block_size = block_size
    
    def estimate_kv_blocks(
        self, 
        seq_length: int, 
        num_layers: int, 
        num_kv_heads: int,
        head_dim: int,
        bytes_per_element: int = 2,  # FP16
    ) -> int:
        """
        估算一个请求需要的KV Cache块数
        在请求实际执行前就计算出内存需求
        """
        # 单token的KV Cache大小
        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
        
        # 总KV Cache大小
        total_kv_bytes = seq_length * kv_per_token
        
        # 每个块能存储的KV Cache
        block_kv_capacity = self.block_size * kv_per_token
        
        # 需要的块数（向上取整）
        return (total_kv_bytes + block_kv_capacity - 1) // block_kv_capacity
    
    def check_admission(
        self, 
        seq_length: int,
        priority: int = 0,
        currently_used_blocks: int = 0,
        **model_config,
    ) -> AdmissionResult:
        """
        决定是否接纳新请求
        综合考虑内存状态、请求优先级和驱逐策略
        """
        # 估算需要的块数
        needed_blocks = self.estimate_kv_blocks(seq_length, **model_config)
        
        available = self.budget.available_blocks - currently_used_blocks
        
        # 情况1：内存充足，直接接纳
        if needed_blocks <= available:
            return AdmissionResult.ACCEPT
        
        # 情况2：内存不足，但可以通过驱逐来满足
        # 计算需要驱逐多少
        deficit = needed_blocks - available
        if deficit <= available * 0.5:  # 驱逐量不超过当前使用的50%
            return AdmissionResult.EVICT_AND_ACCEPT
        
        # 情况3：内存严重不足，建议排队
        if priority < 2:  # 非紧急请求排队
            return AdmissionResult.QUEUE
        
        # 情况4：紧急请求但内存实在不够
        return AdmissionResult.REJECT
    
    def get_memory_report(self, currently_used_blocks: int) -> dict:
        """生成内存使用报告"""
        available = self.budget.available_blocks - currently_used_blocks
        return {
            "total_blocks": self.budget.total_blocks,
            "model_blocks": self.budget.model_weight_blocks,
            "reserved_blocks": self.budget.reserved_blocks,
            "used_blocks": currently_used_blocks,
            "available_blocks": max(0, available),
            "utilization": f"{(currently_used_blocks / self.budget.available_blocks) * 100:.1f}%",
            "max_additional_sequences": available // 16,  # 估算可容纳的短序列数
        }
```

### 4.3 GPU-CPU显存交换（Offloading）

当GPU显存不够时，可以将暂时不活跃的KV Cache换出到CPU内存：

```
GPU-CPU显存交换策略

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  GPU显存 (80GB A100)          CPU内存 (512GB DDR5)            │
│  ┌──────────────────┐        ┌──────────────────┐            │
│  │ 模型权重 (140GB)  │        │                  │            │
│  │ ████              │        │  换出的KV Cache    │            │
│  │ ████              │  ←─→   │  ████████████    │            │
│  │ 活跃KV Cache      │ PCIe   │  ████████████    │            │
│  │ ██████            │ 50GB/s │  ████████████    │            │
│  │ 空闲              │        │  等待重新调度      │            │
│  └──────────────────┘        └──────────────────┘            │
│                                                              │
│  交换决策逻辑：                                                │
│  1. GPU显存使用率 > 90% → 触发换出                             │
│  2. 优先换出"长时间未活跃"的请求的KV Cache                       │
│  3. 换出操作与推理计算pipeline重叠，减少等待                     │
│  4. 被换出的请求重新调度时，优先从CPU读回                        │
│                                                              │
│  延迟影响：                                                    │
│  • PCIe 4.0 x16: ~25 GB/s 单向                               │
│  • 换出1GB KV Cache: ~40ms                                    │
│  • 换入1GB KV Cache: ~40ms                                    │
│  • 可通过流水线化和预取减少感知延迟                              │
│                                                              │
│  适用场景：                                                    │
│  • 多租户推理：低活跃租户的KV Cache换出                         │
│  • 长上下文处理：处理完前文后换出前文的KV Cache                   │
│  • 批处理调度：利用CPU内存缓存等待批处理的请求                    │
└──────────────────────────────────────────────────────────────┘
```

### 4.4 多GPU显存管理

```
多GPU推理中的内存管理挑战

┌───────────────────────────────────────────────────────────────┐
│           多GPU内存管理：张量并行 + KV Cache分布                 │
│                                                               │
│  方案一：KV Cache集中管理                                       │
│  ──────────────────────                                       │
│  GPU0: [模型权重层0-19] + [所有KV Cache]                       │
│  GPU1: [模型权重层20-39] + [空闲]                               │
│  GPU2: [模型权重层40-59] + [空闲]                               │
│  GPU3: [模型权重层60-79] + [空闲]                               │
│                                                               │
│  问题：KV Cache集中在GPU0，成为瓶颈                              │
│  GPU0显存占用 >> 其他GPU                                        │
│                                                               │
│  方案二：KV Cache均匀分布（主流方案）                             │
│  ────────────────────────────                                  │
│  GPU0: [权重0-19] + [KV Cache块0-15]                           │
│  GPU1: [权重20-39] + [KV Cache块16-31]                         │
│  GPU2: [权重40-59] + [KV Cache块32-47]                         │
│  GPU3: [权重60-79] + [KV Cache块48-63]                         │
│                                                               │
│  优势：                                                        │
│  • 显存负载均衡                                                │
│  • 注意力计算可以并行化                                         │
│  • 每个GPU的KV Cache独立管理                                   │
│                                                               │
│  挑战：                                                        │
│  • 注意力计算需要跨GPU通信                                     │
│  • AllGather操作的通信开销                                     │
│  • 需要精确协调各GPU的块分配                                    │
└───────────────────────────────────────────────────────────────┘
```

---

## 五、内存优化实战：从OOM到2x吞吐

### 5.1 典型OOM场景与解决方案

```
┌───────────────────────────────────────────────────────────────┐
│               LLM推理OOM的5种典型场景                          │
│                                                               │
│  场景1: 部署时OOM                                              │
│  ─────────────────                                            │
│  症状: 加载模型时CUDA OOM                                      │
│  原因: 模型权重 > 可用显存                                      │
│  解决:                                                        │
│    ① 量化部署 (FP16→INT4, 显存需求减75%)                       │
│    ② 张量并行 (多GPU分担权重)                                   │
│    ③ 模型权重offload到CPU (延迟换吞吐)                         │
│                                                               │
│  场景2: 首次请求OOM                                             │
│  ────────────────                                             │
│  症状: 第一个请求进来时OOM                                     │
│  原因: KV Cache预分配过大                                      │
│  解决:                                                        │
│    ① 使用PagedAttention按需分配                                │
│    ② 降低max_sequence_length                                  │
│    ③ 预留更多空间给KV Cache                                     │
│                                                               │
│  场景3: 并发增加时OOM                                          │
│  ──────────────────                                           │
│  症状: 正常运行一段时间后，用户数增加时OOM                       │
│  原因: KV Cache总量超出显存                                     │
│  解决:                                                        │
│    ① 内存准入控制（限制并发数）                                  │
│    ② GQA/MQA架构减少每token的KV Cache                          │
│    ③ KV Cache量化 (FP8, 2x压缩)                                │
│    ④ Token淘汰 (H2O/SnapKV)                                    │
│                                                               │
│  场景4: 长序列OOM                                               │
│  ──────────────                                               │
│  症状: 正常短请求没问题，长上下文请求OOM                          │
│  原因: 单个请求的KV Cache过大                                   │
│  解决:                                                        │
│    ① 流式KV Cache管理                                          │
│    ② 上下文压缩/摘要                                            │
│    ③ 动态序列长度限制                                           │
│                                                               │
│  场景5: 突发流量OOM                                             │
│  ──────────────                                               │
│  症状: 流量突增时批量OOM                                        │
│  原因: 缺乏流控和限流                                           │
│  解决:                                                        │
│    ① 请求排队 + 流量整形                                        │
│    ② 弹性扩缩容                                                │
│    ③ 优雅降级（降低模型精度/截断长度）                           │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 A100-80G上70B模型的内存规划实战

```
场景：LLaMA-2 70B 部署在4×A100-80G上
目标：最大化并发服务能力

内存规划计算：
┌──────────────────────────────────────────────────────────────┐
│ Step 1: 模型权重                                             │
│ LLaMA-2 70B FP16 = 140GB                                    │
│ 4×A100 = 320GB总显存                                        │
│ 权重占比: 140/320 = 43.75%                                   │
│ 剩余: 180GB                                                  │
├──────────────────────────────────────────────────────────────┤
│ Step 2: 计算激活值预留                                        │
│ 每GPU预留 ~5GB 用于计算中间结果                                │
│ 4×5 = 20GB                                                  │
│ 剩余: 160GB                                                  │
├──────────────────────────────────────────────────────────────┤
│ Step 3: KV Cache预算                                         │
│ 可用于KV Cache: 160GB                                        │
│ LLaMA-2 70B: 每token KV Cache ≈ 1MB (GQA后)                 │
│ 理论可存储: 160GB / 1MB = ~160K tokens                        │
├──────────────────────────────────────────────────────────────┤
│ Step 4: 并发能力计算                                         │
│ 场景A: 4K context, 64并发                                    │
│   KV Cache: 64 × 4K × 1MB = 256GB > 160GB ❌                │
│   → 需要KV Cache量化(FP8): 256/2 = 128GB < 160GB ✅          │
│                                                               │
│ 场景B: 2K context, 80并发                                     │
│   KV Cache: 80 × 2K × 1MB = 160GB = 160GB ✅                 │
│   → 刚好满足，需要严格准入控制                                  │
│                                                               │
│ 场景C: 2K context, 40并发 + FP8量化                            │
│   KV Cache: 40 × 2K × 0.5MB = 40GB << 160GB ✅✅              │
│   → 大量剩余空间用于长序列请求                                  │
├──────────────────────────────────────────────────────────────┤
│ 💡 最终推荐配置:                                               │
│   • 使用GQA架构 (LLaMA-2 70B原生支持)                         │
│   • KV Cache使用FP8量化                                       │
│   • 并发限制: 60个2K序列 或 30个4K序列                          │
│   • 启用PagedAttention管理显存                                 │
│   • 预留10%显存作为安全缓冲                                    │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 内存监控与告警

```python
"""
LLM推理系统内存监控
实时追踪内存使用状态，提前预警OOM风险
"""
import time
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class MemorySnapshot:
    timestamp: float
    gpu_used_mb: float
    gpu_total_mb: float
    kv_cache_used_mb: float
    kv_cache_capacity_mb: float
    active_sequences: int
    allocated_blocks: int
    free_blocks: int

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(
        self, 
        gpu_total_mb: float,
        kv_cache_capacity_mb: float,
        warning_threshold: float = 0.80,
        critical_threshold: float = 0.90,
    ):
        self.gpu_total = gpu_total_mb
        self.kv_capacity = kv_cache_capacity_mb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.history: List[MemorySnapshot] = []
        self.alert_callbacks: List[Callable] = []
    
    def record(self, snapshot: MemorySnapshot):
        """记录内存快照并检查告警"""
        self.history.append(snapshot)
        
        # GPU显存使用率
        gpu_util = snapshot.gpu_used_mb / self.gpu_total
        kv_util = snapshot.kv_cache_used_mb / self.kv_capacity
        
        # 触发告警
        if gpu_util >= self.critical_threshold:
            self._alert("CRITICAL", f"GPU显存使用率 {gpu_util:.1%}，接近OOM", snapshot)
        elif gpu_util >= self.warning_threshold:
            self._alert("WARNING", f"GPU显存使用率 {gpu_util:.1%}", snapshot)
        
        if kv_util >= 0.95:
            self._alert("CRITICAL", f"KV Cache使用率 {kv_util:.1%}，即将耗尽", snapshot)
    
    def _alert(self, level: str, message: str, snapshot: MemorySnapshot):
        """触发告警"""
        for callback in self.alert_callbacks:
            callback(level, message, snapshot)
    
    def predict_oom_time(self) -> float:
        """
        基于历史趋势预测OOM时间
        返回预计剩余秒数，-1表示不会OOM
        """
        if len(self.history) < 10:
            return -1
        
        recent = self.history[-10:]
        growth_rates = []
        
        for i in range(1, len(recent)):
            dt = recent[i].timestamp - recent[i-1].timestamp
            if dt > 0:
                dv = recent[i].kv_cache_used_mb - recent[i-1].kv_cache_used_mb
                growth_rates.append(dv / dt)
        
        if not growth_rates:
            return -1
        
        avg_growth = sum(growth_rates) / len(growth_rates)
        if avg_growth <= 0:
            return -1
        
        remaining = self.kv_capacity - self.history[-1].kv_cache_used_mb
        return remaining / avg_growth if avg_growth > 0 else -1
    
    def get_report(self) -> dict:
        """生成内存使用报告"""
        if not self.history:
            return {"status": "no data"}
        
        latest = self.history[-1]
        oom_eta = self.predict_oom_time()
        
        return {
            "gpu_used": f"{latest.gpu_used_mb:.0f}/{self.gpu_total:.0f} MB",
            "gpu_utilization": f"{latest.gpu_used_mb / self.gpu_total:.1%}",
            "kv_cache_used": f"{latest.kv_cache_used_mb:.0f}/{self.kv_capacity:.0f} MB",
            "kv_cache_utilization": f"{latest.kv_cache_used_mb / self.kv_capacity:.1%}",
            "active_sequences": latest.active_sequences,
            "blocks": f"{latest.allocated_blocks}/{latest.allocated_blocks + latest.free_blocks}",
            "oom_eta_seconds": f"{oom_eta:.0f}" if oom_eta > 0 else "N/A",
            "history_length": len(self.history),
        }
```

---

## 六、总结：LLM内存管理的核心认知

```
┌──────────────────────────────────────────────────────────────┐
│           LLM推理内存管理的6个核心认知                          │
│                                                              │
│  1️⃣  KV Cache是内存管理的主要矛盾                              │
│     模型权重是固定的，KV Cache是动态增长的。                     │
│     优化KV Cache = 优化并发能力。                               │
│                                                              │
│  2️⃣  PagedAttention是分水岭                                   │
│     分页管理彻底解决了碎片化和预分配问题。                        │
│     从"能部署"到"能高效并发服务"的关键技术。                     │
│                                                              │
│  3️⃣  量化是最直接的杠杆                                        │
│     权重量化降低部署门槛，KV Cache量化提升并发能力。              │
│     FP8是当前性价比最优的选择。                                 │
│                                                              │
│  4️⃣  架构设计影响内存效率                                      │
│     GQA/MLA从模型设计层面减少KV Cache，是根本解法。              │
│     选择支持GQA的模型可以大幅降低内存压力。                      │
│                                                              │
│  5️⃣  准入控制优于事后补救                                       │
│     在请求进入前就评估内存需求，避免运行时OOM。                   │
│     宁可让请求排队，也不能让GPU OOM崩溃。                       │
│                                                              │
│  6️⃣  内存管理需要全局视角                                      │
│     权重 + KV Cache + 激活值三者此消彼长。                      │
│     优化任何一个都需要考虑对其他两个的影响。                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

> **最后的忠告**：LLM推理的内存管理看似是底层技术问题，但直接影响上层业务的并发能力和服务质量。一个优秀的内存管理方案可以让你在同样的硬件上服务2-3倍的用户。投入时间理解内存管理的原理和工程实践，是LLM推理系统走向生产化的必经之路。记住：**GPU显存是LLM推理中最稀缺的资源，如何高效管理它，决定了你的推理系统能走多远**。
