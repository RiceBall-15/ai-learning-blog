---
title: "LLM应用的SRE实践：从传统监控到AI原生可观测性"
description: "系统讲解LLM应用在生产环境中的可靠性工程实践，涵盖监控体系、告警策略、故障排查与混沌工程"
date: 2026-05-31
author: "RiceBall"
category: "engineering"
subCategory: "infra"
tags: ["SRE", "LLM监控", "可观测性", "混沌工程", "生产运维"]
draft: false
---

## 引言：LLM应用的可靠性新挑战

传统Web应用的SRE已经非常成熟——SLI/SLO定义清晰、监控体系完善、故障恢复机制标准化。但当我们将LLM集成到生产系统中时，会发现**传统SRE的很多假设不再成立**：

1. **输出是非确定性的**：同一个输入，每次可能产生不同的输出，传统的"回归测试"概念需要重新定义
2. **延迟分布是重尾的**：LLM推理延迟从几十毫秒到几十秒不等，P99和P50的差距可能达到100倍
3. **成本是不可预测的**：token级别的计费模型意味着每次请求的成本随输出长度动态变化
4. **质量难以量化**：输出的"好坏"没有简单的二元判定，需要多维度评估
5. **依赖外部模型服务**：核心能力依赖于外部API（OpenAI、Anthropic等），可用性受制于第三方

本文将系统性地介绍如何为LLM应用构建一套完整的SRE体系。

## 核心概念：LLM应用的SLI体系定义

### 延迟维度（Latency SLIs）

LLM应用的延迟需要拆解为多个子阶段：

```
用户请求 → [预处理] → [排队等待] → [Prefill] → [Decode] → [后处理] → [响应返回]
              ↓              ↓           ↓          ↓           ↓
          预处理延迟      排队延迟    Prefill延迟  Decode延迟   后处理延迟
```

```python
# 延迟SLI的定义和采集
class LatencySLIs:
    def __init__(self):
        self.metrics = {
            'total_latency': Histogram('llm_total_latency_seconds', 
                                       'Total request latency'),
            'prefill_latency': Histogram('llm_prefill_latency_seconds',
                                        'Prefill phase latency'),
            'decode_latency': Histogram('llm_decode_latency_seconds',
                                       'Decode phase latency'),
            'ttft': Histogram('llm_time_to_first_token_seconds',
                             'Time to first token'),
            'tpot': Histogram('llm_time_per_output_token_seconds',
                             'Time per output token'),
            'queue_wait': Histogram('llm_queue_wait_seconds',
                                   'Queue wait time'),
        }
    
    def record_request(self, timings):
        """记录单次请求的延迟指标"""
        self.metrics['total_latency'].observe(timings['total'])
        self.metrics['prefill_latency'].observe(timings['prefill'])
        self.metrics['decode_latency'].observe(timings['decode'])
        self.metrics['ttft'].observe(timings['ttft'])
        self.metrics['tpot'].observe(timings['tpot'])
        self.metrics['queue_wait'].observe(timings['queue_wait'])
```

### 质量维度（Quality SLIs）

质量SLI是LLM应用特有的，需要多维度评估：

| 维度 | SLI指标 | 定义 | 目标值 |
|------|---------|------|--------|
| **相关性** | Relevance Score | 输出与问题的相关程度 | > 0.85 |
| **准确性** | Factual Accuracy | 输出的事实准确性 | > 0.90 |
| **完整性** | Completeness Score | 回答覆盖问题所有方面 | > 0.80 |
| **安全性** | Safety Score | 输出是否符合安全策略 | > 0.99 |
| **格式合规** | Format Compliance | 输出是否符合指定格式 | > 0.95 |
| **一致性** | Consistency Score | 多次查询的输出一致性 | > 0.75 |

```python
# 质量SLI评估框架
class QualitySLIEvaluator:
    def __init__(self):
        self.judge_model = load_judge_model()  # 用于质量评估的LLM
    
    def evaluate_response(self, query, response, context=None):
        """评估单个响应的质量"""
        scores = {}
        
        # 相关性评估
        scores['relevance'] = self.judge_model.score(
            query=query, response=response,
            criteria="How relevant is the response to the query?"
        )
        
        # 安全性评估
        scores['safety'] = self.safety_classifier.check(response)
        
        # 格式合规评估
        scores['format'] = self.format_checker.validate(response)
        
        return scores
    
    def compute_sli(self, responses):
        """计算批量响应的SLI"""
        all_scores = [self.evaluate_response(r['query'], r['response']) 
                      for r in responses]
        
        return {
            'relevance_p50': np.percentile([s['relevance'] for s in all_scores], 50),
            'safety_rate': sum(1 for s in all_scores if s['safety'] > 0.9) / len(all_scores),
            'format_compliance': sum(1 for s in all_scores if s['format'] == 1.0) / len(all_scores),
        }
```

### 成本维度（Cost SLIs）

```python
# 成本SLI定义
class CostSLIs:
    def __init__(self, pricing_config):
        self.pricing = pricing_config  # 模型定价配置
        
    def calculate_request_cost(self, request):
        """计算单次请求的成本"""
        input_tokens = request['input_tokens']
        output_tokens = request['output_tokens']
        model = request['model']
        
        input_cost = input_tokens * self.pricing[model]['input_per_token']
        output_cost = output_tokens * self.pricing[model]['output_per_token']
        
        return input_cost + output_cost
    
    def get_cost_metrics(self, requests):
        """获取成本指标"""
        costs = [self.calculate_request_cost(r) for r in requests]
        return {
            'total_cost': sum(costs),
            'avg_cost_per_request': np.mean(costs),
            'p99_cost_per_request': np.percentile(costs, 99),
            'cost_per_1k_tokens': np.mean(costs) / np.mean([r['input_tokens'] + r['output_tokens'] for r in requests]) * 1000,
        }
```

## 监控体系架构

### 三层监控架构

```
┌─────────────────────────────────────────────────────┐
│                    可视化层                           │
│  Grafana Dashboard / 自定义UI / 移动端告警           │
├─────────────────────────────────────────────────────┤
│                    分析层                             │
│  Prometheus + ClickHouse / 自定义LLM分析引擎          │
├─────────────────────────────────────────────────────┤
│                    采集层                             │
│  OpenTelemetry SDK / 自定义Exporter / 日志采集       │
└─────────────────────────────────────────────────────┘
```

### OpenTelemetry集成

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

# 初始化OpenTelemetry
tracer_provider = TracerProvider()
tracer = tracer_provider.get_tracer("llm-application")
meter = metrics.get_meter("llm-application")

# 延迟直方图
latency_histogram = meter.create_histogram(
    name="llm.request.duration",
    description="LLM request duration",
    unit="ms"
)

# Token计数器
token_counter = meter.create_counter(
    name="llm.tokens.usage",
    description="Token usage count",
    unit="tokens"
)

class LLMInstrumentor:
    def __init__(self, tracer, meter):
        self.tracer = tracer
        self.meter = meter
        
    def trace_request(self, func):
        """装饰器：自动追踪LLM请求"""
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("llm_request") as span:
                # 记录输入
                span.set_attribute("llm.model", kwargs.get('model', 'unknown'))
                span.set_attribute("llm.input_tokens", kwargs.get('input_tokens', 0))
                
                # 执行请求
                start = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                
                # 记录指标
                latency_histogram.record(duration, {"model": kwargs.get('model')})
                token_counter.add(result['output_tokens'], {"type": "output"})
                
                # 记录输出
                span.set_attribute("llm.output_tokens", result['output_tokens'])
                span.set_attribute("llm.duration_ms", duration)
                span.set_attribute("llm.total_cost", result['cost'])
                
                return result
        return wrapper
```

### 日志结构化

```python
import json
import uuid
from datetime import datetime

class StructuredLLMLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log_request(self, request_data, response_data, metadata=None):
        """记录结构化的LLM请求日志"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4()),
            "session_id": request_data.get('session_id'),
            "user_id": request_data.get('user_id'),
            
            # 请求信息
            "input": {
                "model": request_data['model'],
                "messages": request_data['messages'],
                "temperature": request_data.get('temperature', 1.0),
                "max_tokens": request_data.get('max_tokens', 4096),
                "input_tokens": request_data.get('input_tokens'),
            },
            
            # 响应信息
            "output": {
                "content": response_data.get('content', ''),
                "finish_reason": response_data.get('finish_reason'),
                "output_tokens": response_data.get('output_tokens'),
                "latency_ms": response_data.get('latency_ms'),
            },
            
            # 质量评估
            "quality": response_data.get('quality_scores', {}),
            
            # 成本
            "cost": {
                "input_cost": response_data.get('input_cost'),
                "output_cost": response_data.get('output_cost'),
                "total_cost": response_data.get('total_cost'),
            },
            
            # 元数据
            "metadata": metadata or {},
        }
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
```

## 告警策略设计

### 告警分级体系

| 级别 | 触发条件 | 响应时间 | 通知方式 | 示例 |
|------|---------|---------|---------|------|
| **P0 - 致命** | 服务完全不可用，或输出严重违反安全策略 | 5分钟 | 电话+短信+IM | 模型服务宕机、安全过滤器失效 |
| **P1 - 严重** | 质量SLI低于SLO，成本异常飙升 | 15分钟 | 短信+IM | 准确率下降20%、成本翻倍 |
| **P2 - 警告** | 延迟P99超标，错误率上升 | 30分钟 | IM+邮件 | P99延迟>10s、错误率>5% |
| **P3 - 信息** | 趋势性变化，容量预警 | 1小时 | 邮件 | 流量持续增长20%/天 |

### 智能告警规则

```python
# 基于多维度的智能告警
class LLMMonitoringAlerts:
    def __init__(self):
        self.alert_rules = [
            # P0: 安全事件
            {
                'name': 'safety_violation',
                'condition': lambda metrics: metrics['safety_violation_rate'] > 0.01,
                'severity': 'P0',
                'message': '安全策略违规率超过1%，可能存在安全漏洞',
            },
            
            # P1: 质量下降
            {
                'name': 'quality_degradation',
                'condition': lambda metrics: metrics['quality_sli'] < 0.80,
                'severity': 'P1',
                'message': f'质量SLI降至{metrics["quality_sli"]:.2%}，低于SLO目标0.85',
            },
            
            # P1: 成本异常
            {
                'name': 'cost_spike',
                'condition': lambda metrics: metrics['cost_per_1k_tokens'] > metrics['cost_baseline'] * 2,
                'severity': 'P1',
                'message': '单token成本超过基线200%',
            },
            
            # P2: 延迟异常
            {
                'name': 'latency_spike',
                'condition': lambda metrics: metrics['p99_latency'] > 15000,  # 15秒
                'severity': 'P2',
                'message': f'P99延迟达到{metrics["p99_latency"]/1000:.1f}秒',
            },
            
            # P2: 错误率
            {
                'name': 'error_rate',
                'condition': lambda metrics: metrics['error_rate'] > 0.05,
                'severity': 'P2',
                'message': f'错误率达到{metrics["error_rate"]:.2%}',
            },
        ]
    
    def check_alerts(self, current_metrics):
        """检查所有告警规则"""
        triggered = []
        for rule in self.alert_rules:
            if rule['condition'](current_metrics):
                triggered.append({
                    'rule': rule['name'],
                    'severity': rule['severity'],
                    'message': rule['message'],
                    'timestamp': datetime.utcnow(),
                    'metrics': current_metrics,
                })
        return triggered
```

### 异常检测：从规则到AI

传统规则告警的局限性在于**阈值难以设定**。LLM应用的流量模式、延迟分布往往是非平稳的，固定阈值容易产生误报或漏报。

```python
# 基于统计的异常检测
class LLMMetricAnomalyDetector:
    def __init__(self, window_size=300, sensitivity=3.0):
        self.window_size = window_size  # 5分钟窗口
        self.sensitivity = sensitivity  # 3-sigma
        self.baseline = {}  # 基线统计
        
    def update_baseline(self, metric_name, values):
        """更新基线统计"""
        self.baseline[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
        }
        
    def detect_anomaly(self, metric_name, current_value):
        """检测异常"""
        if metric_name not in self.baseline:
            return False
            
        baseline = self.baseline[metric_name]
        z_score = (current_value - baseline['mean']) / (baseline['std'] + 1e-8)
        
        return abs(z_score) > self.sensitivity
    
    def detect_drift(self, metric_name, recent_values, lookback=24):
        """检测分布漂移"""
        # 使用KL散度检测分布变化
        historical = self.baseline[metric_name]['distribution']
        recent = np.histogram(recent_values, bins=50, density=True)[0]
        
        # 避免除以零
        historical = historical + 1e-10
        recent = recent + 1e-10
        
        kl_divergence = np.sum(historical * np.log(historical / recent))
        return kl_divergence > 0.5  # KL散度阈值
```

## 故障排查方法论

### LLM特有的故障模式

```
┌────────────────────────────────────────────────────┐
│                LLM故障分类树                        │
├────────────────────────────────────────────────────┤
│                                                    │
│  ├── 基础设施故障                                   │
│  │   ├── GPU OOM                                   │
│  │   ├── 显存泄漏                                  │
│  │   └── CUDA错误                                  │
│  │                                                 │
│  ├── 模型服务故障                                   │
│  │   ├── API限流                                   │
│  │   ├── 模型服务宕机                               │
│  │   └── 推理引擎异常                               │
│  │                                                 │
│  ├── 输入相关故障                                   │
│  │   ├── Prompt注入攻击                            │
│  │   ├── 输入长度超限                               │
│  │   └── 特殊字符导致Tokenization异常               │
│  │                                                 │
│  ├── 输出相关故障                                   │
│  │   ├── 幻觉（Hallucination）                     │
│  │   ├── 重复输出（Repetition）                     │
│  │   ├── 格式不合规                                │
│  │   └── 安全违规输出                               │
│  │                                                 │
│  └── 系统集成故障                                   │
│      ├── Function Calling失败                      │
│      ├── 上下文窗口溢出                             │
│      └── 多轮对话上下文丢失                         │
└────────────────────────────────────────────────────┘
```

### 故障排查流程

```python
class LLMFaultDiagnoser:
    def __init__(self, monitoring_client, log_client):
        self.monitoring = monitoring_client
        self.logs = log_client
        
    def diagnose(self, incident):
        """标准化故障排查流程"""
        report = {
            'incident_id': incident['id'],
            'start_time': incident['start_time'],
            'symptoms': [],
            'root_cause': None,
            'impact': {},
        }
        
        # Step 1: 确认故障范围
        report['symptoms'] = self._identify_symptoms(incident)
        
        # Step 2: 排查基础设施
        infra_issues = self._check_infrastructure(incident)
        if infra_issues:
            report['root_cause'] = infra_issues
            return report
            
        # Step 3: 排查模型服务
        model_issues = self._check_model_service(incident)
        if model_issues:
            report['root_cause'] = model_issues
            return report
            
        # Step 4: 排查应用层
        app_issues = self._check_application(incident)
        report['root_cause'] = app_issues
        
        # Step 5: 评估影响
        report['impact'] = self._assess_impact(incident)
        
        return report
    
    def _check_infrastructure(self, incident):
        """检查基础设施状态"""
        checks = {
            'gpu_utilization': self.monitoring.get_metric('gpu_utilization', incident['time_range']),
            'gpu_memory': self.monitoring.get_metric('gpu_memory_used', incident['time_range']),
            'cpu_usage': self.monitoring.get_metric('cpu_usage', incident['time_range']),
            'disk_io': self.monitoring.get_metric('disk_io_read', incident['time_range']),
        }
        
        issues = []
        if checks['gpu_utilization']['avg'] > 0.98:
            issues.append('GPU利用率持续过高，可能存在死循环或低效计算')
        if checks['gpu_memory']['max'] > 0.95:
            issues.append('GPU显存使用接近上限，可能发生OOM')
            
        return issues
```

### 实战案例：幻觉故障排查

```python
# 案例：某RAG系统突然出现大量幻觉输出
# 排查步骤

# 1. 收集幻觉样本
hallucination_samples = collect_samples(
    quality_filter=lambda r: r['quality_scores']['factual_accuracy'] < 0.5,
    time_range='last_2_hours'
)

# 2. 分析共性
patterns = analyze_patterns(hallucination_samples)
# 发现：所有幻觉都发生在引用外部数据源时

# 3. 根因分析
# 检查向量数据库检索结果
for sample in hallucination_samples[:10]:
    retrieved = vector_db.search(sample['query'], top_k=5)
    print(f"Query: {sample['query'][:50]}...")
    print(f"Retrieved chunks: {[r['content'][:30] for r in retrieved]}")
    print(f"Response claims: {extract_claims(sample['response'])}")
    print("---")

# 发现：向量数据库索引损坏，返回了不相关的文档片段
# 根因：昨日索引更新任务异常中断，导致部分索引不一致
```

## 混沌工程实践

### LLM应用的混沌实验设计

```python
class LLMChaosEngine:
    def __init__(self):
        self.experiments = []
        
    def design_experiment(self, hypothesis, failure_mode):
        """设计混沌实验"""
        experiment = {
            'hypothesis': hypothesis,
            'failure_mode': failure_mode,
            'steady_state': self._define_steady_state(),
            'method': self._design_method(failure_mode),
            'rollback': self._define_rollback(),
        }
        self.experiments.append(experiment)
        return experiment
    
    def run_injection(self, experiment):
        """执行故障注入"""
        # 记录当前稳态
        baseline = self._measure_steady_state()
        
        # 注入故障
        fault = self._inject_fault(experiment['failure_mode'])
        
        # 观察系统行为
        observation_window = 300  # 5分钟
        observations = self._observe_system(observation_window)
        
        # 评估结果
        result = self._evaluate_experiment(baseline, observations)
        
        # 清理
        self._remove_fault(fault)
        
        return result

# 常见混沌实验场景
chaos_scenarios = [
    {
        'name': 'API限流模拟',
        'description': '模拟OpenAI API返回429限流',
        'method': '在代理层注入限流响应',
        'expected_behavior': '系统自动降级到本地模型或缓存',
    },
    {
        'name': '网络延迟注入',
        'description': '向模型服务添加200ms网络延迟',
        'method': '使用tc/netem注入延迟',
        'expected_behavior': '用户感知延迟增加但服务不中断',
    },
    {
        'name': '输出截断模拟',
        'description': '模拟模型输出被截断（finish_reason=length）',
        'method': '在代理层截断响应',
        'expected_behavior': '系统自动重试或提示用户缩短输入',
    },
    {
        'name': '幻觉注入',
        'description': '模拟模型返回事实性错误的输出',
        'method': '使用专门的prompt构造错误输出',
        'expected_behavior': 'RAG验证机制检测并拒绝错误输出',
    },
    {
        'name': '显存泄漏模拟',
        'description': '模拟CUDA显存泄漏',
        'method': '在特定条件下分配不释放的显存',
        'expected_behavior': '监控系统检测显存异常并触发重启',
    },
]
```

### 混沌实验执行框架

```python
class ChaosExperimentRunner:
    def __init__(self, target_system):
        self.target = target_system
        self.results = []
        
    def run_full_experiment(self, scenario):
        """执行完整的混沌实验"""
        print(f"🧪 开始实验: {scenario['name']}")
        print(f"📋 假设: {scenario['hypothesis']}")
        
        # 1. 确认稳态
        print("1️⃣ 确认稳态...")
        steady_state = self._confirm_steady_state()
        if not steady_state:
            print("❌ 系统未处于稳态，实验中止")
            return None
        
        # 2. 注入故障
        print(f"2️⃣ 注入故障: {scenario['method']}")
        fault_handle = self._inject_fault(scenario['failure_mode'])
        
        # 3. 观察
        print("3️⃣ 观察系统行为...")
        observations = self._observe(duration=scenario.get('duration', 300))
        
        # 4. 评估
        print("4️⃣ 评估结果...")
        passed = self._evaluate(scenario, observations)
        
        # 5. 清理
        print("5️⃣ 清理故障...")
        self._cleanup(fault_handle)
        
        result = {
            'scenario': scenario['name'],
            'passed': passed,
            'observations': observations,
            'timestamp': datetime.utcnow(),
        }
        self.results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"实验结果: {status}")
        
        return result
```

## SLO定义与错误预算

### LLM应用的SLO体系

```python
class LLMSLO:
    def __init__(self):
        self.slos = {
            # 可用性SLO
            'availability': {
                'target': 0.999,  # 99.9%可用性
                'window': '30d',
                'measurement': 'successful_requests / total_requests',
            },
            
            # 延迟SLO
            'latency_ttft': {
                'target': 0.95,  # 95%的请求TTFT<2s
                'threshold_ms': 2000,
                'window': '30d',
            },
            'latency_tpot': {
                'target': 0.90,  # 90%的请求TPOT<100ms
                'threshold_ms': 100,
                'window': '30d',
            },
            
            # 质量SLO
            'quality_relevance': {
                'target': 0.85,  # 平均相关性>0.85
                'window': '7d',
                'measurement': 'avg_relevance_score',
            },
            'quality_safety': {
                'target': 0.999,  # 安全率>99.9%
                'window': '30d',
                'measurement': 'safe_responses / total_responses',
            },
            
            # 成本SLO
            'cost_efficiency': {
                'target': 0.002,  # 平均成本<$0.002/请求
                'window': '30d',
                'measurement': 'total_cost / total_requests',
            },
        }
    
    def calculate_error_budget(self, slo_name):
        """计算错误预算"""
        slo = self.slos[slo_name]
        budget = 1 - slo['target']
        
        # 获取当前消耗
        consumed = self._get_budget_consumption(slo_name)
        remaining = budget - consumed
        
        return {
            'slo': slo_name,
            'target': slo['target'],
            'total_budget': budget,
            'consumed': consumed,
            'remaining': remaining,
            'remaining_percentage': remaining / budget * 100,
        }
```

### 错误预算策略

```python
class ErrorBudgetPolicy:
    def __init__(self, slo_calculator):
        self.slo_calc = slo_calculator
        
    def get_deployment_policy(self):
        """根据错误预算剩余量决定部署策略"""
        budget = self.slo_calc.calculate_error_budget('availability')
        remaining_pct = budget['remaining_percentage']
        
        if remaining_pct > 50:
            return {
                'policy': 'normal',
                'description': '正常部署流程',
                'canary_percentage': 10,
                'rollout_speed': 'standard',
            }
        elif remaining_pct > 20:
            return {
                'policy': 'cautious',
                'description': '谨慎部署，增加验证',
                'canary_percentage': 5,
                'rollout_speed': 'slow',
                'additional_checks': ['quality_gate', 'cost_gate'],
            }
        elif remaining_pct > 5:
            return {
                'policy': 'restricted',
                'description': '仅允许紧急修复部署',
                'canary_percentage': 1,
                'rollout_speed': 'very_slow',
                'require_approval': True,
            }
        else:
            return {
                'policy': 'freeze',
                'description': '部署冻结，优先修复可靠性问题',
                'allow_emergency_only': True,
            }
```

## Grafana Dashboard设计

### 核心面板布局

```json
{
  "dashboard": {
    "title": "LLM Application SRE Dashboard",
    "panels": [
      {
        "title": "Overview - Error Budget Remaining",
        "type": "stat",
        "targets": [{"expr": "1 - (llm_errors_total / llm_requests_total)"}],
        "thresholds": {"steps": [{"value": 0.999, "color": "green"}, {"value": 0.99, "color": "yellow"}, {"value": 0, "color": "red"}]}
      },
      {
        "title": "Request Rate & Error Rate",
        "type": "timeseries",
        "targets": [
          {"expr": "rate(llm_requests_total[5m])", "legendFormat": "Requests/s"},
          {"expr": "rate(llm_errors_total[5m]) / rate(llm_requests_total[5m])", "legendFormat": "Error Rate"}
        ]
      },
      {
        "title": "Latency Distribution",
        "type": "heatmap",
        "targets": [
          {"expr": "histogram_quantile(0.50, rate(llm_latency_seconds_bucket[5m]))", "legendFormat": "P50"},
          {"expr": "histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))", "legendFormat": "P95"},
          {"expr": "histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m]))", "legendFormat": "P99"}
        ]
      },
      {
        "title": "Token Usage & Cost",
        "type": "timeseries",
        "targets": [
          {"expr": "rate(llm_tokens_total[5m])", "legendFormat": "Tokens/s"},
          {"expr": "rate(llm_cost_dollars_total[5m])", "legendFormat": "Cost/s"}
        ]
      },
      {
        "title": "Quality Metrics",
        "type": "timeseries",
        "targets": [
          {"expr": "llm_quality_relevance_score", "legendFormat": "Relevance"},
          {"expr": "llm_quality_safety_score", "legendFormat": "Safety"}
        ]
      }
    ]
  }
}
```

## 总结

LLM应用的SRE实践需要在传统SRE的基础上进行多维度扩展。核心要点：

| 维度 | 传统SRE | LLM应用SRE |
|------|---------|-----------|
| **SLI** | 延迟、错误率、吞吐 | +Token延迟、质量评分、成本效率 |
| **监控** | Metrics + Logs | +Traces + 输出质量分析 |
| **告警** | 基于阈值 | +基于分布变化、AI异常检测 |
| **故障排查** | 确定性输出 | 非确定性输出，需要概率分析 |
| **混沌工程** | 网络、CPU、内存 | +幻觉注入、限流、输出截断 |
| **SLO** | 可用性、延迟 | +质量、安全、成本 |

构建一套完善的LLM应用SRE体系不是一蹴而就的，建议从以下优先级逐步推进：

1. **第一步**：建立基础监控（延迟、错误率、Token消耗）—— 1-2周
2. **第二步**：定义SLI/SLO和告警规则 —— 1周
3. **第三步**：构建质量评估管道 —— 2-3周
4. **第四步**：实施混沌工程实验 —— 持续迭代
5. **第五步**：建立错误预算策略和自动化响应 —— 持续迭代

LLM应用的可靠性工程是一个快速发展的领域，随着大模型在生产环境中的深入应用，更多的最佳实践和工具将会涌现。保持对新技术的关注，持续迭代和优化，是构建可靠LLM系统的关键。
