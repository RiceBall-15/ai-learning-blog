---
title: "LLM推理性能剖析：Profiling工具链与系统级瓶颈诊断方法论"
description: "系统性介绍LLM推理性能剖析方法论，涵盖nsight、PyTorch Profiler、自定义tracing等工具链，从GPU计算到内存带宽的全栈瓶颈诊断"
date: 2026-06-01
author: "RiceBall-15"
category: "aiInfra"
subCategory: inference
tags: ["LLM推理", "性能剖析", "Profiling", "GPU优化", "瓶颈诊断", "NVIDIA Nsight"]
draft: false
---

# LLM推理性能剖析：Profiling工具链与系统级瓶颈诊断方法论

## 一、为什么LLM推理需要专门的性能剖析？

### 1.1 传统Web性能剖析的局限性

大多数工程师熟悉的性能剖析工具——New Relic、Datadog APM、Chrome DevTools——都是围绕**请求-响应**模型设计的。它们假设一个请求对应一次CPU计算、一次数据库查询、一次网络IO。但LLM推理打破了这个假设：

```
┌─────────────────────────────────────────────────────────────────────┐
│              传统应用 vs LLM推理 的性能特征差异                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  传统Web应用                    LLM推理服务                           │
│  ┌──────────────────┐          ┌──────────────────┐                │
│  │ 单次请求延迟: 50ms│          │ 单次请求延迟: 1-30s│               │
│  │ CPU密集型        │          │ GPU密集型         │                │
│  │ 内存: GB级       │          │ 显存: 24-192GB    │                │
│  │ 瓶颈: CPU/网络   │          │ 瓶颈: 多维度      │                │
│  │ 并发模型: 线程   │          │ 并发模型: 流式批处理│               │
│  │ 状态: 无状态     │          │ 状态: KV Cache    │                │
│  └──────────────────┘          └──────────────────┘                │
│                                                                      │
│  LLM推理特有的性能维度:                                               │
│  ├── Prefill阶段 (计算密集) vs Decode阶段 (内存带宽密集)             │
│  ├── KV Cache管理与内存碎片                                          │
│  ├── GPU SM利用率与内存带宽利用率的权衡                               │
│  ├── Tensor Core利用率 vs CUDA Core利用率                            │
│  ├── Batch调度策略对吞吐量的影响                                      │
│  └── 多卡通信开销 (TP/PP/EP)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 性能剖析的核心目标

LLM推理性能剖析需要回答三个层次的问题：

| 层次 | 问题 | 典型工具 |
|------|------|---------|
| **宏观层** | 系统整体吞吐量和延迟分布如何？ | Prometheus + Grafana |
| **中观层** | Prefill和Decode各阶段耗时如何分布？ | 自定义Tracing |
| **微观层** | GPU上每个Kernel的执行效率如何？ | NVIDIA Nsight Systems |

## 二、LLM推理的性能瓶颈全景

### 2.1 两阶段瓶颈模型

LLM推理分为两个本质不同的阶段，它们的瓶颈完全不同：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM推理两阶段瓶颈模型                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Prefill阶段 (首Token延迟的主要贡献)                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  输入: [token_1, token_2, ..., token_N]                      │   │
│  │  计算: Attention矩阵 (N×N) + FFN (N×d)                       │   │
│  │  瓶颈: GPU计算能力 (FLOPS)                                    │   │
│  │  指标: TFLOPS利用率, 计算强度 (Arithmetic Intensity)          │   │
│  │  优化: Flash Attention, Tensor Parallelism, Tensor Core      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Decode阶段 (每Token延迟的主要贡献)                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  输入: [上一时刻输出]                                         │   │
│  │  计算: 单Token前向传播 + KV Cache读写                         │   │
│  │  瓶颈: GPU内存带宽 (GB/s)                                     │   │
│  │  指标: 内存带宽利用率, Batch Size                             │   │
│  │  优化: KV Cache量化, Flash Decoding, Batching                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  计算强度对比:                                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Prefill: AI >> 1.0 (计算密集) → GPU算力是瓶颈               │   │
│  │  Decode:  AI << 1.0 (带宽密集) → 内存带宽是瓶颈              │   │
│  │                                                              │   │
│  │  其中 AI = FLOPs / Bytes (Arithmetic Intensity)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 瓶颈诊断决策树

在开始剖析之前，先用决策树快速定位瓶颈方向：

```
                    LLM推理延迟高
                         │
                    ┌────┴────┐
                    │         │
              首Token延迟高   生成速度慢
                    │         │
              ┌─────┴─────┐  ┌─────┴─────┐
              │           │  │           │
           Prefill慢   调度慢  Decode慢  内存不足
              │           │  │           │
         ┌────┴────┐     │  ┌────┴────┐  │
         │         │     │  │         │  │
      计算不足  通信慢  排队久 带宽不足  Batch小 OOM
         │         │     │  │         │  │
      检查TFLOPS 检查NVLink  检查  检查内存   检查
      利用率     利用率   调度器 带宽利用率 Batch策略
```

## 三、性能剖析工具链详解

### 3.1 工具链全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM推理 Profiling 工具链                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  应用层 Profiling│  │  系统层 Profiling│  │  硬件层 Profiling│     │
│  │                  │  │                  │  │                  │     │
│  │ • PyTorch        │  │ • nvidia-smi     │  │ • Nsight Systems │     │
│  │   Profiler       │  │ • dstat          │  │ • Nsight Compute │     │
│  │ • vLLM Internal  │  │ • perf           │  │ • NCU (Nsight    │     │
│  │   Tracing        │  │ • pidstat        │  │   Compute)      │     │
│  │ • Custom Trace   │  │ • mpstat         │  │ • GPU Metrics    │     │
│  │   Decorator      │  │                  │  │                  │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│         │                      │                      │              │
│         └──────────────────────┼──────────────────────┘              │
│                                │                                     │
│                    ┌───────────┴───────────┐                        │
│                    │   统一分析与可视化       │                        │
│                    │   • TensorBoard        │                        │
│                    │   • Perfetto UI        │                        │
│                    │   • 自定义 Dashboard    │                        │
│                    └───────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 NVIDIA Nsight Systems：GPU时间线分析

Nsight Systems是LLM推理性能剖析的**核心工具**，它提供GPU上所有CUDA操作的精确时间线。

**安装与基本使用：**

```bash
# 安装Nsight Systems (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2024.6.1.90_2024.6.1.90-1_amd64.deb
sudo dpkg -i nsight-systems-2024.6.1.90_2024.6.1.90-1_amd64.deb

# 方式一：命令行采集
nsys profile -t cuda,nvtx,osrt \
  --duration=30 \
  --output=llm_inference_profile \
  python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct

# 方式二：对已运行的服务进行attach
nsys profile -p $(pgrep -f vllm) \
  --duration=30 \
  --output=inference_snapshot
```

**关键指标解读：**

```python
"""
Nsight Systems 输出的关键指标解读
"""
import subprocess
import json

def analyze_nsys_output(report_path: str) -> dict:
    """
    解析nsys export输出的JSON报告
    
    关键关注点:
    1. Kernel执行时间分布
    2. GPU空闲间隙
    3. 内存拷贝开销
    4. CUDA Stream并行度
    """
    # 使用nsys export将报告转为JSON
    # nsys export -t json -o report.json report.nsys-rep
    
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    analysis = {
        "kernel_stats": {},
        "gpu_utilization": 0.0,
        "memory_copy_overhead": 0.0,
        "stream_parallelism": 0,
    }
    
    # 分析Kernel执行
    kernels = [item for item in data['traceEvents'] 
               if item.get('cat') == 'cuda_kernel']
    
    total_kernel_time = sum(k.get('dur', 0) for k in kernels)
    
    # 按Kernel名称分组统计
    kernel_groups = {}
    for k in kernels:
        name = k.get('name', 'unknown')
        if name not in kernel_groups:
            kernel_groups[name] = {'count': 0, 'total_dur': 0}
        kernel_groups[name]['count'] += 1
        kernel_groups[name]['total_dur'] += k.get('dur', 0)
    
    # 排序找出耗时最长的Kernel
    sorted_kernels = sorted(
        kernel_groups.items(),
        key=lambda x: x[1]['total_dur'],
        reverse=True
    )
    
    analysis['kernel_stats'] = {
        'top_kernels': sorted_kernels[:10],
        'total_kernel_time_us': total_kernel_time,
        'kernel_count': len(kernels),
    }
    
    return analysis

# 使用示例
# result = analyze_nsys_output('report.json')
# for name, stats in result['kernel_stats']['top_kernels']:
#     print(f"{name}: {stats['count']}次, 总耗时 {stats['total_dur']/1000:.1f}ms")
```

**Nsight Systems中需要关注的GPU时间线特征：**

| 时间线特征 | 含义 | 优化方向 |
|-----------|------|---------|
| 大片绿色间隙 | GPU空闲，等待数据或指令 | 优化Batch调度 |
| 长黄色条 | CUDA内存拷贝 | 使用Unified Memory或预分配 |
| 密集绿色条 | GPU正在执行Kernel | 正常状态，检查Kernel效率 |
| 交替的绿黄条 | 频繁的Kernel启动+数据传输 | 使用CUDA Graph减少启动开销 |
| 单一Stream | 未利用GPU并行能力 | 启用多Stream |

### 3.3 PyTorch Profiler：Python层性能剖析

对于需要理解Python层逻辑与GPU操作对应关系的场景，PyTorch Profiler是最直接的工具：

```python
import torch
import torch.profiler
from vllm import LLM, SamplingParams

def profile_vllm_inference():
    """
    使用PyTorch Profiler剖析LLM推理
    
    输出: Chrome Trace格式，可在chrome://tracing中查看
    """
    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    prompts = [
        "Explain the concept of transformer architecture in detail.",
        "Write a Python function to implement quicksort.",
        "Compare and contrast SQL and NoSQL databases.",
    ] * 10  # 增加请求量以获得稳定的Profiling结果
    
    # 配置Profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,      # 预热2个step
            warmup=3,    # 预热3个step
            active=5,    # 采集5个step
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        for i in range(10):
            outputs = llm.generate(prompts, sampling_params)
            profiler.step()
    
    # 打印关键统计
    print(profiler.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
```

**PyTorch Profiler输出的关键列：**

```
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                      Name    Self CPU %      Self CPU   CPU total %     CPU total  GPU total %     GPU total    Self GPU     Self GPU %  
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                        aten::linear:         15.2%        2.341s        45.6%        7.034s        38.2%       12.514s        0.000s         0.0%  
              aten::scaled_dot_product...      12.8%        1.972s        32.1%        4.949s        45.8%       15.012s        0.000s         0.0%  
                     aten::addmm:              8.5%        1.311s         8.5%        1.311s         0.0%        0.000s        0.000s         0.0%  
           aten::_scaled_dot_product_...       0.0%        0.001s         0.0%        0.001s        45.8%       15.012s       15.012s       100.0%  
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
```

### 3.4 自定义Tracing：业务层性能剖析

在生产环境中，我们通常需要将性能剖析与业务逻辑关联。一个实用的自定义Tracing装饰器：

```python
import time
import json
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class TraceSpan:
    """单个追踪跨度"""
    name: str
    start_time: float
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['TraceSpan'] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'start': self.start_time,
            'end': self.end_time,
            'duration_ms': round(self.duration_ms, 3),
            'metadata': self.metadata,
            'children': [c.to_dict() for c in self.children],
        }

class LLMTracer:
    """
    LLM推理专用Tracer
    
    自动追踪:
    - 请求级别的端到端延迟
    - Prefill / Decode 各阶段耗时
    - Token生成速度 (tokens/sec)
    - 队列等待时间
    """
    
    def __init__(self, service_name: str = "llm-inference"):
        self.service_name = service_name
        self.spans: List[TraceSpan] = []
        self._current_span: Optional[TraceSpan] = None
    
    @contextmanager
    def trace(self, name: str, **metadata):
        """创建追踪跨度"""
        span = TraceSpan(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata,
        )
        
        if self._current_span:
            self._current_span.children.append(span)
        
        previous_span = self._current_span
        self._current_span = span
        
        try:
            yield span
        finally:
            span.end_time = time.perf_counter()
            self._current_span = previous_span
            
            if previous_span is None:
                self.spans.append(span)
    
    def trace_request(self, request_id: str, prompt_tokens: int):
        """追踪完整的推理请求"""
        return self.trace(
            'inference_request',
            request_id=request_id,
            prompt_tokens=prompt_tokens,
        )
    
    def trace_prefill(self):
        """追踪Prefill阶段"""
        return self.trace('prefill')
    
    def trace_decode(self, target_tokens: int):
        """追踪Decode阶段"""
        return self.trace('decode', target_tokens=target_tokens)
    
    def trace_queue_wait(self):
        """追踪队列等待时间"""
        return self.trace('queue_wait')
    
    def export_summary(self) -> dict:
        """导出追踪摘要"""
        if not self.spans:
            return {}
        
        request_spans = [s for s in self.spans if s.name == 'inference_request']
        
        if not request_spans:
            return {}
        
        durations = [s.duration_ms for s in request_spans]
        
        # 提取Prefill和Decode耗时
        prefill_durations = []
        decode_durations = []
        
        for req in request_spans:
            for child in req.children:
                if child.name == 'prefill':
                    prefill_durations.append(child.duration_ms)
                elif child.name == 'decode':
                    decode_durations.append(child.duration_ms)
        
        # 计算tokens/sec
        token_throughputs = []
        for req in request_spans:
            target_tokens = req.metadata.get('prompt_tokens', 0)
            decode_child = next(
                (c for c in req.children if c.name == 'decode'), None
            )
            if decode_child and decode_child.duration_ms > 0:
                throughput = target_tokens / (decode_child.duration_ms / 1000)
                token_throughputs.append(throughput)
        
        summary = {
            'total_requests': len(request_spans),
            'avg_latency_ms': sum(durations) / len(durations),
            'p50_latency_ms': sorted(durations)[len(durations) // 2],
            'p99_latency_ms': sorted(durations)[int(len(durations) * 0.99)],
        }
        
        if prefill_durations:
            summary['avg_prefill_ms'] = sum(prefill_durations) / len(prefill_durations)
        
        if decode_durations:
            summary['avg_decode_ms'] = sum(decode_durations) / len(decode_durations)
        
        if token_throughputs:
            summary['avg_tokens_per_sec'] = sum(token_throughputs) / len(token_throughputs)
        
        return summary


# ============ 使用示例 ============

tracer = LLMTracer()

def serve_inference_request(request_id: str, prompt: str, max_tokens: int = 256):
    """模拟推理服务处理请求"""
    prompt_tokens = len(prompt.split())  # 简化：用空格分词估算
    
    with tracer.trace_request(request_id, prompt_tokens=prompt_tokens) as req_span:
        
        # 1. 队列等待
        with tracer.trace_queue_wait():
            time.sleep(0.001)  # 模拟队列等待
        
        # 2. Prefill阶段
        with tracer.trace_prefill() as prefill_span:
            time.sleep(0.05)  # 模拟prefill计算
            prefill_span.metadata['tokens_processed'] = prompt_tokens
        
        # 3. Decode阶段
        with tracer.trace_decode(target_tokens=max_tokens) as decode_span:
            time.sleep(0.2)  # 模拟decode
            decode_span.metadata['tokens_generated'] = max_tokens
    
    return {"request_id": request_id, "status": "completed"}

# 批量执行
for i in range(10):
    serve_inference_request(
        request_id=f"req_{i:03d}",
        prompt=f"This is test prompt number {i} for profiling purposes.",
        max_tokens=128,
    )

# 输出摘要
summary = tracer.export_summary()
print(json.dumps(summary, indent=2))
```

## 四、实战：系统级瓶颈诊断

### 4.1 GPU利用率诊断

```bash
#!/bin/bash
# gpu_monitor.sh - 实时GPU监控脚本

# 监控GPU利用率、显存使用、温度
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,\
memory.used,memory.total,temperature.gpu,power.draw \
--format=csv -l 1000 > gpu_metrics.csv &

GPU_PID=$!

# 监控进程级GPU使用
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
--format=csv -l 2000 > gpu_process.csv &

PROCESS_PID=$!

echo "GPU监控已启动 (PID: $GPU_PID, $PROCESS_PID)"
echo "按 Ctrl+C 停止监控"

# 等待用户中断
trap "kill $GPU_PID $PROCESS_PID 2>/dev/null; exit 0" INT TERM
wait
```

**GPU利用率诊断表：**

| GPU利用率 | 显存带宽利用率 | 诊断结论 | 优化方向 |
|-----------|---------------|---------|---------|
| < 30% | < 20% | GPU严重空闲 | 检查Batch Size、队列调度 |
| < 30% | > 80% | 内存带宽瓶颈 | Decode阶段正常，增大Batch |
| > 80% | < 30% | 计算未充分并行 | 检查Tensor Core利用、Flash Attention |
| > 80% | > 80% | 理想状态 | 关注延迟是否满足SLA |
| > 80% | > 95% | 带宽饱和 | 考虑KV Cache量化、模型并行 |

### 4.2 vLLM推理服务瓶颈定位

```python
"""
vLLM推理服务瓶颈定位脚本

使用vLLM的内置metrics进行瓶颈分析
"""
import requests
import time
import statistics
from dataclasses import dataclass
from typing import List

@dataclass
class InferenceBenchmarkResult:
    """推理基准测试结果"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_throughput: float  # requests/sec
    avg_tokens_per_sec: float

def benchmark_inference(
    base_url: str = "http://localhost:8000",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_requests: int = 100,
    concurrency: int = 10,
    prompt: str = "Explain the theory of relativity in simple terms.",
    max_tokens: int = 128,
) -> InferenceBenchmarkResult:
    """
    对vLLM推理服务进行负载测试
    
    通过分析延迟分布来定位瓶颈:
    - 高方差 → 排队调度问题
    - 高P99但P50正常 → 长尾延迟（GC、内存碎片）
    - 所有指标都高 → 计算/内存瓶颈
    """
    import concurrent.futures
    
    latencies: List[float] = []
    token_counts: List[int] = []
    errors = 0
    
    def single_request():
        nonlocal errors
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            latency = (time.perf_counter() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                tokens = data['usage']['completion_tokens']
                return latency, tokens
            else:
                errors += 1
                return latency, 0
        except Exception as e:
            errors += 1
            return 0, 0
    
    # 使用线程池模拟并发
    total_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            latency, tokens = future.result()
            if latency > 0:
                latencies.append(latency)
                token_counts.append(tokens)
    
    total_time = time.perf_counter() - total_start
    
    sorted_latencies = sorted(latencies)
    
    result = InferenceBenchmarkResult(
        total_requests=num_requests,
        successful_requests=len(latencies),
        failed_requests=errors,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=sorted_latencies[len(sorted_latencies) // 2],
        p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)],
        p99_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)],
        avg_throughput=len(latencies) / total_time,
        avg_tokens_per_sec=sum(token_counts) / (sum(latencies) / 1000) if token_counts else 0,
    )
    
    return result

def diagnose_bottleneck(result: InferenceBenchmarkResult) -> str:
    """
    基于基准测试结果自动诊断瓶颈
    
    诊断逻辑:
    1. 失败率 > 5% → 服务不稳定
    2. P99/P50 > 5x → 长尾延迟严重
    3. 吞吐量 < 1 req/s → 计算瓶颈
    4. tokens/sec < 10 → 内存带宽瓶颈
    """
    diagnosis = []
    
    # 检查失败率
    fail_rate = result.failed_requests / result.total_requests
    if fail_rate > 0.05:
        diagnosis.append(f"⚠️ 高失败率: {fail_rate:.1%}，检查OOM、超时或服务健康状态")
    
    # 检查长尾延迟
    if result.p50_latency_ms > 0:
        tail_ratio = result.p99_latency_ms / result.p50_latency_ms
        if tail_ratio > 5:
            diagnosis.append(
                f"⚠️ 严重长尾延迟: P99/P50 = {tail_ratio:.1f}x，"
                f"可能原因: KV Cache碎片化、Batch调度不均、GC停顿"
            )
    
    # 检查吞吐量
    if result.avg_throughput < 1.0:
        diagnosis.append(
            f"⚠️ 吞吐量极低: {result.avg_throughput:.2f} req/s，"
            f"可能原因: Batch Size太小、Prefill计算瓶颈、模型太大"
        )
    
    # 检查Token生成速度
    if result.avg_tokens_per_sec < 10:
        diagnosis.append(
            f"⚠️ Token生成速度慢: {result.avg_tokens_per_sec:.1f} tokens/s，"
            f"可能原因: 内存带宽不足、未启用Flash Decoding、KV Cache效率低"
        )
    
    if not diagnosis:
        diagnosis.append("✅ 未检测到明显瓶颈，各项指标正常")
    
    return "\n".join(diagnosis)


# 运行基准测试
result = benchmark_inference(
    base_url="http://localhost:8000",
    num_requests=50,
    concurrency=5,
)

print(f"=== 推理基准测试结果 ===")
print(f"成功请求: {result.successful_requests}/{result.total_requests}")
print(f"平均延迟: {result.avg_latency_ms:.1f}ms")
print(f"P50延迟:  {result.p50_latency_ms:.1f}ms")
print(f"P95延迟:  {result.p95_latency_ms:.1f}ms")
print(f"P99延迟:  {result.p99_latency_ms:.1f}ms")
print(f"吞吐量:   {result.avg_throughput:.2f} req/s")
print(f"Token速度: {result.avg_tokens_per_sec:.1f} tokens/s")
print(f"\n=== 瓶颈诊断 ===")
print(diagnose_bottleneck(result))
```

### 4.3 CUDA Kernel效率分析

使用Nsight Compute分析单个Kernel的执行效率：

```bash
# 分析特定Kernel的执行效率
ncu --set full \
  --target-processes all \
  --kernel-name "flash_forward_kernel" \
  --launch-skip 100 \
  --launch-count 50 \
  --output kernel_analysis \
  python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-1B

# 关键指标解读
# SM (Streaming Multiprocessor) 利用率
# - 目标: > 80%
# - 低利用率原因: 线程块太小、Bank Conflict、寄存器溢出

# 内存带宽利用率  
# - 目标: > 70% (Decode阶段)
# - 低利用率原因: 访存不连续、Cache Miss

# Tensor Core利用率
# - 目标: > 60% (Prefill阶段)
# - 低利用率原因: 矩阵维度不匹配、未使用最优数据类型
```

## 五、生产环境性能监控体系

### 5.1 推理服务关键指标

```
┌─────────────────────────────────────────────────────────────────────┐
│               LLM推理服务监控指标体系                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────── 延迟指标 ───────────────────┐                 │
│  │  • TTFT (Time to First Token): 首Token延迟     │                 │
│  │  • TPOT (Time Per Output Token): 每Token延迟   │                 │
│  │  • E2E Latency: 端到端延迟                     │                 │
│  │  • Queue Wait Time: 队列等待时间                │                 │
│  └────────────────────────────────────────────────┘                 │
│                                                                      │
│  ┌─────────────────── 吞吐指标 ───────────────────┐                 │
│  │  • Requests/sec: 每秒处理请求数                 │                 │
│  │  • Tokens/sec: 每秒处理Token数                  │                 │
│  │  • Prefill Tokens/sec: Prefill吞吐              │                 │
│  │  • Decode Tokens/sec: Decode吞吐                │                 │
│  └────────────────────────────────────────────────┘                 │
│                                                                      │
│  ┌─────────────────── 资源指标 ───────────────────┐                 │
│  │  • GPU Utilization: GPU利用率                   │                 │
│  │  • GPU Memory Used: 显存使用量                   │                 │
│  │  • KV Cache Usage: KV Cache使用率               │                 │
│  │  • Batch Size: 当前Batch大小                    │                 │
│  │  • Running/Waiting Requests: 运行/等待请求数    │                 │
│  └────────────────────────────────────────────────┘                 │
│                                                                      │
│  ┌─────────────────── 质量指标 ───────────────────┐                 │
│  │  • Error Rate: 错误率 (OOM/Timeout/Rejected)   │                 │
│  │  • SLA Compliance: SLA达标率                    │                 │
│  │  • Prefix Cache Hit Rate: 前缀缓存命中率        │                 │
│  └────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Prometheus + Grafana 监控配置

```yaml
# prometheus.yml - LLM推理服务监控配置
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm-inference'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    
  - job_name: 'gpu-exporter'
    static_configs:
      - targets: ['localhost:9400']  # dcgm-exporter
    
  - job_name: 'system-metrics'
    static_configs:
      - targets: ['localhost:9100']  # node-exporter
```

```python
"""
关键告警规则配置
"""
ALERT_RULES = {
    # 首Token延迟告警
    "high_ttft": {
        "expr": "histogram_quantile(0.99, vllm:e2e_request_latency_seconds_bucket) > 5",
        "for": "5m",
        "severity": "warning",
        "message": "P99首Token延迟超过5秒",
    },
    
    # GPU利用率过低
    "low_gpu_util": {
        "expr": "DCGM_FI_DEV_GPU_UTIL < 30",
        "for": "10m",
        "severity": "info",
        "message": "GPU利用率持续低于30%，可能资源浪费",
    },
    
    # 显存不足
    "high_memory_usage": {
        "expr": "DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE > 0.95",
        "for": "2m",
        "severity": "critical",
        "message": "显存使用率超过95%，存在OOM风险",
    },
    
    # 吞吐量下降
    "low_throughput": {
        "expr": "rate(vllm:num_requests_completed_total[5m]) < 0.5",
        "for": "5m",
        "severity": "warning",
        "message": "吞吐量下降至0.5 req/s以下",
    },
    
    # 错误率过高
    "high_error_rate": {
        "expr": "rate(vllm:num_requests_failed_total[5m]) / rate(vllm:num_requests_total[5m]) > 0.05",
        "for": "3m",
        "severity": "critical",
        "message": "请求失败率超过5%",
    },
}
```

## 六、性能优化案例库

### 6.1 案例一：Decode阶段内存带宽瓶颈

**症状：** GPU利用率40%，但显存带宽利用率95%，Decode速度仅15 tokens/s。

**诊断过程：**
```bash
# 1. 使用nsys确认Decode阶段特征
nsys profile -t cuda --duration=10 --output=decode_analysis python -m vllm...

# 2. 分析Kernel执行模式
# 发现: 大量小Kernel频繁启动，每个Kernel只处理少量数据

# 3. 确认是内存带宽瓶颈
# FLOPS利用率低 + 带宽利用率高 = 典型的Memory-Bound
```

**解决方案：**
```bash
# 启用Flash Decoding减少Decode阶段的Kernel数量
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192

# 或者增大Batch Size以提高带宽利用率
# 通过增加并发请求来利用更大的Batch
```

### 6.2 案例二：Prefill阶段Tensor Core利用率低

**症状：** 首Token延迟3秒，但GPU计算利用率仅35%。

**诊断过程：**
```python
# 使用Nsight Compute分析单个Prefill Kernel
# ncu --set full --kernel-name "volta_s884gemm" ./inference_binary

# 关键发现:
# 1. Tensor Core利用率: 15% (目标: >60%)
# 2. 原因: 输入序列长度波动大，导致矩阵维度不匹配
# 3. 短序列(< 128 tokens)的Prefill效率极低
```

**解决方案：**
```python
# 方案1: 使用Chunked Prefill将短序列合并
# vllm自动将多个短请求的Prefill合并为一个大Batch

# 方案2: 对短序列使用专门优化的Kernel
from sglang import function, image, system, user, assistant

@function()
def short_seq_prefill():
    """短序列专用Prefill路径"""
    # 使用更适合小矩阵的CUDA Core而非Tensor Core
    pass
```

## 七、总结：性能剖析最佳实践

```
┌─────────────────────────────────────────────────────────────────────┐
│                  LLM推理性能剖析最佳实践清单                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ☐ 1. 先宏观后微观                                                   │
│     先用Prometheus监控整体指标，再用Nsight深入GPU层                    │
│                                                                      │
│  ☐ 2. 分阶段分析                                                     │
│     Prefill和Decode分开分析，它们的瓶颈完全不同                       │
│                                                                      │
│  ☐ 3. 对比基线                                                       │
│     每次优化前后都要跑基准测试，量化改进效果                           │
│                                                                      │
│  ☐ 4. 关注P99而非平均值                                              │
│     平均延迟可能掩盖严重的长尾问题                                    │
│                                                                      │
│  ☐ 5. 模拟真实负载                                                   │
│     单请求测试无法暴露并发调度问题，必须用多并发测试                   │
│                                                                      │
│  ☐ 6. 持续监控                                                       │
│     部署后持续监控，及时发现性能退化                                  │
│                                                                      │
│  ☐ 7. 建立性能回归测试                                               │
│     每次代码变更后自动运行基准测试，防止性能回退                       │
│                                                                      │
│  推荐工具组合:                                                        │
│  • 开发阶段: PyTorch Profiler + TensorBoard                          │
│  • 详细分析: NVIDIA Nsight Systems + Nsight Compute                  │
│  • 生产监控: Prometheus + Grafana + 自定义Metrics                    │
│  • 快速诊断: 自定义Tracer + 统计分析脚本                             │
└─────────────────────────────────────────────────────────────────────┘
```

性能剖析不是一次性工作，而是LLM推理服务的**持续工程实践**。随着模型规模增大、用户量增长、业务场景变化，性能瓶颈也在不断演变。建立系统化的剖析方法论和监控体系，才能在问题出现前发现隐患，在问题出现后快速定位根因。
