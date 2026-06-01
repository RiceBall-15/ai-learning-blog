---
title: "LLM 推理服务生产部署实战：从架构选型到性能调优的完整指南"
description: "系统对比 vLLM、SGLang、TGI、TensorRT-LLM 四大推理服务的架构差异，给出生产环境下从选型到调优的完整决策框架"
date: 2026-05-20
author: "RiceBall-15"
category: aiInfra
subCategory: inference
tags: ["推理服务", "vLLM", "SGLang", "TGI", "TensorRT-LLM", "生产部署", "性能调优"]
draft: false
---

## 核心问题

LLM 推理服务是将模型能力转化为产品价值的关键基础设施。选错推理引擎，可能导致：

| 问题 | 表现 | 后果 |
|------|------|------|
| 吞吐量不足 | QPS 打满后请求排队超时 | 用户等待 10s+ |
| 延迟过高 | TTFT > 2s | 交互体验差 |
| 显存泄漏 | 服务运行 24h+ 后 OOM | 周期性重启 |
| 量化兼容差 | 模型量化后精度下降 >5% | 业务效果退化 |

## 一、四大推理引擎架构对比

### 1.1 整体对比

| 特性 | vLLM | SGLang | TGI | TensorRT-LLM |
|------|------|--------|-----|--------------|
| **开发者** | UC Berkeley | Stanford | HuggingFace | NVIDIA |
| **开源协议** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **核心技术** | PagedAttention | RadixAttention | Prefix Caching | TensorRT 编译 |
| **调度策略** | Async + Continuous Batching | Radix-aware Scheduling | Token-level Scheduling | Planner-based |
| **首 Token 延迟** | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| **吞吐量** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★☆ |
| **长上下文** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| **多模态** | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| **社区活跃度** | 极高 | 高 | 高 | 中 |

### 1.2 vLLM——当前生态选择

vLLM 的核心创新是 **PagedAttention**——将 KV-Cache 分页管理，类似操作系统虚拟内存：

```
传统 KV-Cache: [连续分配，固定大小]
  ┌──────────────┬──────────────┬──────────────┐
  │ Request A    │ Request B    │ Request C    │
  │ Block 0-31   │ Block 0-31   │ Block 0-31   │
  │(21%利用率)   │(35%利用率)   │(45%利用率)   │
  └──────────────┴──────────────┴──────────────┘
              整体 KV-Cache 利用: ~33%

PagedAttention:
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │A0│B0│A1│C0│B1│A2│C1│B2│C2│  │  │  │
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
              整体 KV-Cache 利用: ~94%
```

**架构特点**：
- Async LLM Engine：请求处理与 Token 生成解耦
- Continuous Batching：动态调度请求进出 batch
- Prefix Caching (Automatic Prefix Caching)：自动识别公共前缀并复用 KV-Cache
- Speculative Decoding 支持：EAGLE、Medusa、Prompt Lookup

**适用场景**：通用推理，vLLM 是最安全的选择。社区最大，问题响应最快。

### 1.3 SGLang——结构化推理的革新者

SGLang 的核心理念是 **RadixAttention**——将推理过程的 KV-Cache 组织为 Trie 树：

```
传统 prefix caching:
  "什么是Attention机制？" → 缓存完整前缀KV
  "什么是Attention机制和...？" → 只匹配部分，无效

RadixAttention:
            "什么是"
           /        \
    "Attention机制？"   "Transformer中的..."
        |                      |
    "Attention机制和..."    "位置编码..."
        |                 (共享前缀节点)
    (共享 "Attention" 前缀)
```

**RadixAttention 带来的关键提升**：
- 细粒度前缀复用：不要求完全匹配前缀，最长公共前缀即可
- Constrained Decoding 原生集成：SGLang 的 Grammar 约束比 vLLM 快 5-10x
- Structured Generation：JSON Schema 直接编译为 Radix Tree 约束

SGLang 在读密集型场景（RAG、Chat with Docs）中通常比 vLLM 快 20-50%。

**适用场景**：大量结构化输出、RAG 检索、需要 Grammar Constraints 的场景。

### 1.4 TGI——HuggingFace 的一站式方案

TGI (Text Generation Inference) 的优势在于 HuggingFace 生态集成：
- 模型下载：`huggingface-cli` 内置
- 自动版本管理：兼容 Hub 上的所有 Transformers 模型
- 原生 Tokenizer 集成

**性能劣势**：连续批处理的 Token 级调度不如 vLLM/SGLang 精细，在长上下文场景吞吐量较低。

**适用场景**：快速部署 HuggingFace 模型、团队已深度使用 HuggingFace 生态。

### 1.5 TensorRT-LLM——NVIDIA 原生的极致性能

TensorRT-LLM 是 NVIDIA 的精调引擎：

```
TGI / vLLM:
  模型权重 → PyTorch 执行 → CUDA Kernel
  (每次推理都走 PyTorch 调度)

TensorRT-LLM:
  模型权重 → [TensorRT 编译期优化] → .engine 文件 → CUDA Kernel
  (编译期完成算子融合、内存规划、图优化)
```

**编译期优化**：算子融合（Fused Attention + MLP）、层融合、INT4/INT8/Float8 Kernel 自动选择、内存池预分配。

**性能优势**：
- 推理速度：TensorRT-LLM 比 PyTorch-based 框架快 1.5-3x
- 延迟确定性：编译后不会出现 PyTorch JIT 的 Warm-up 延迟
- 多 GPU 通信优化：NCCL 深度集成，AllReduce 延迟更低

**适用场景**：NVIDIA GPU 专属部署、延迟敏感型服务、需要低功耗高吞吐的边缘部署。

## 二、部署架构选择

### 2.1 单节点部署

```
                    ┌─────────────────────┐
                    │     Gateway/Nginx    │
                    │  (路由 + 限流 + 鉴权) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ vLLM GPU0│    │ vLLM GPU1│    │ vLLM GPU2│
        │  8B Qwen │    │  72B Qwen│    │  Embed   │
        └──────────┘    └──────────┘    └──────────┘
              │                │               │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    Metrics/Prometheus│
                    └─────────────────────┘
```

### 2.2 分布式部署

```
                           ┌──────────┐
                           │  Load     │
                           │  Balancer │
                           └────┬─────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
   ┌──────────┐          ┌──────────┐          ┌──────────┐
   │ vLLM Pod │          │ vLLM Pod │  ...     │ vLLM Pod │
   │ DP=4, TP=2│          │ DP=4, TP=2│          │ DP=4, TP=2│
   └────┬─────┘          └────┬─────┘          └────┬─────┘
        │                     │                     │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │ InfiniBand          │ InfiniBand          │ InfiniBand
   │ Ring Bus            │ Ring Bus            │ Ring Bus
   └─────────┘          └─────────┘          └─────────┘
```

## 三、性能调优实战

### 3.1 关键性能指标

| 缩写 | 全称 | 含义 | 正常值 |
|------|------|------|--------|
| TTFT | Time to First Token | 用户看到第一个 Token 的时间 | < 500ms |
| TPOT | Time per Output Token | 输出 Token 的间隔时间 | < 50ms |
| ITL | Inter-Token Latency | Token 间延迟（同 TPOT） | < 50ms |
| Throughput | Requests/sec | 系统吞吐量 | 场景依赖 |
| GPU Utilization | GPU 利用率 | 计算资源使用率 | > 85% |

### 3.2 关键配置参数

**max_num_batched_tokens** (vLLM)

此参数决定最大 batch 的 Token 总数。增大此值可提升吞吐量，但超过特定值后收益递减：

```
max_num_batched_tokens 调优曲线:
Throughput
  ↑
  │         ____/
  │        /
  │  _____/    ← 收益递减点（通常 4096-8192）
  │ /
  └────────────────→ max_num_batched_tokens
```

**建议值（基于 70B 模型，A100-80G）**：

| 场景 | 推荐值 | 理由 |
|------|--------|------|
| 低延迟优先 | 2048 | 减少排队等待 |
| 吞吐量优先 | 8192 | 充分填充 batch |
| 长上下文 | 16384 | 支持 32K 上下文 |

**enable_prefix_caching**

| 场景 | 值 | 效果 |
|------|-----|------|
| Chat | false | 消息几乎无重复前缀 |
| RAG | true | 系统提示词共享，缓存命中率 60-90% |
| Code Completion | true | 公共上下文前缀极长 |

### 3.3 监控体系

```yaml
# Prometheus 关键指标
metrics:
  - name: vllm:request_slo_achieved
    desc: "SLO 达标率（<2s TTFT）"
    threshold: 0.95

  - name: vllm:gpu_cache_usage
    desc: "KV-Cache 使用率"
    warning: 0.8
    critical: 0.95

  - name: vllm:request_prompt_tokens
    desc: "请求的 Prompt 长度分布"
    p50: < 2048
    p99: < 16384

  - name: vllm:num_requests_running
    desc: "当前运行中的请求数"
    warning: 32
    critical: 64
```

## 四、常见问题与解决方案

| 问题 | 诊断 | 修复 |
|------|------|------|
| OOM 频繁 | 监控 gpu_cache_usage > 95% | 减少 max_num_seqs / 降低 max_model_len |
| TTFT 突增 | KV-Cache 未命中导致大量计算 | 开启 prefix caching / 增加 batch size |
| TPOT 不稳定 | GPU 利用率抖动 | 固定调度策略为 preemption_mode=recompute |
| 推理结果错误 | 量化精度 + 模型不兼容 | 排查 quant_method 和模型格式一致性 |
| 请求超时 | QPS 超过服务容量 | 扩容 / 加限流 / 模型降级到小模型 |
| 显存泄漏 | 长期运行后 Memory Usage 持续增长 | 设置 max_num_seqs 上限 / 定期重启 |

## 五、选型决策树

```
你的场景？
├── 通用推理，需要最大社区支持
│   └── vLLM ← 最安全的选择
├── 大量结构化输出 / Grammar 约束
│   └── SGLang ← 结构化推理最优
├── HuggingFace 生态深度绑定
│   └── TGI ← 无缝集成
├── NVIDIA 专属集群，追求极致性能
│   └── TensorRT-LLM ← 编译优化最强
└── 混合场景（推荐）
    └── vLLM + SGLang
        ├─ vLLM 作为通用推理（主入口）
        └─ SGLang 作为结构化推理（特定路由）
```

## 六、小结

1. **vLLM** 是通用推理的首选——社区最大、最稳定、性能优秀
2. **SGLang** 在结构化输出场景领先——RadixAttention + Constrained Decoding 是真正的架构创新
3. **TGI** 适合 HuggingFace 生态团队——部署最快，但性能不是最优
4. **TensorRT-LLM** 适合极致性能场景——延迟确定性和吞吐量最高，但开发调试周期长

生产环境推荐：vLLM 作为主力推理引擎，SGLang 作为结构化输出场景的补充。两者共享相同的 OpenAI API 接口，切换成本极低。