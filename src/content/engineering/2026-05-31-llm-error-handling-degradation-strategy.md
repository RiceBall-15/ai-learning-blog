---
title: "LLM应用的错误处理与优雅降级策略实战"
description: "深度解析LLM应用中的异常处理、模型降级、超时控制、重试机制等容错策略，附完整代码示例与生产级最佳实践"
date: 2026-05-31
author: "RiceBall-15"
category: "engineering"
subCategory: "infra"
tags: ["LLM", "错误处理", "降级策略", "容错", "可靠性", "AI工程化"]
draft: false
---

# LLM应用的错误处理与优雅降级策略实战

## 引言

LLM应用与传统Web应用的最大区别之一，就是其**不确定性**。传统API调用通常只有成功或失败两种状态，而LLM应用面临的异常情况要复杂得多：

- **API限流**：OpenAI的RPM限制可能导致429错误
- **模型过载**：高峰期模型服务可能返回503
- **输出异常**：模型可能返回格式错误、内容不当或空响应
- **成本失控**：Token消耗超出预期，账单暴涨
- **延迟波动**：响应时间从几百毫秒到几十秒不等

本文将从实战角度，系统性地讲解如何构建一个**高可用、可容错**的LLM应用系统。

---

## 一、LLM应用异常全景图

### 1.1 异常分类与影响

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM应用异常全景图                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  网络层异常      │  │  模型层异常      │  │  业务层异常          │ │
│  │                 │  │                 │  │                     │ │
│  │  • 连接超时     │  │  • 速率限制     │  │  • 输出格式错误     │ │
│  │  • 读取超时     │  │  • 模型过载     │  │  • 内容违规         │ │
│  │  • DNS解析失败  │  │  • 模型下线     │  │  • 空响应           │ │
│  │  • SSL证书错误  │  │  • 配额耗尽     │  │  • 幻觉输出         │ │
│  │                 │  │  • 鉴权失败     │  │  • Prompt注入       │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
│           │                    │                      │            │
│           ▼                    ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    影响等级评估                               │   │
│  │  低影响: 偶发超时 → 重试可解决                                │   │
│  │  中影响: 限流/过载 → 需要降级策略                             │   │
│  │  高影响: 服务不可用 → 需要完整容错方案                         │   │
│  │  严重: 成本失控/安全问题 → 需要紧急干预                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 异常影响量化

| 异常类型 | 发生概率 | 用户影响 | 业务影响 | 恢复时间 |
|---------|---------|---------|---------|---------|
| 网络超时 | 高(10-20%) | 延迟增加 | 无 | 自动重试 |
| API限流 | 中(5-10%) | 请求失败 | 部分用户受影响 | 1-5分钟 |
| 模型过载 | 低(1-5%) | 服务不可用 | 严重 | 5-30分钟 |
| 输出异常 | 中(3-8%) | 答案质量下降 | 用户体验差 | 手动修复 |
| 成本失控 | 低(1-2%) | 无直接感知 | 财务损失 | 人工干预 |

---

## 二、多层容错架构

### 2.1 分层防护设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                     多层容错架构                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Layer 1: 预防层 (Prevention)                                  │ │
│  │  • 输入验证与清洗                                              │ │
│  │  • Prompt模板化                                                │ │
│  │  • Token预算控制                                               │ │
│  │  • 速率限制                                                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Layer 2: 检测层 (Detection)                                   │ │
│  │  • 异常类型识别                                                │ │
│  │  • 严重等级评估                                                │ │
│  │  • 影响范围分析                                                │ │
│  │  • 监控告警                                                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Layer 3: 恢复层 (Recovery)                                    │ │
│  │  • 自动重试                                                    │ │
│  │  • 模型降级                                                    │ │
│  │  • 缓存回退                                                    │ │
│  │  • 静态响应                                                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Layer 4: 兜底层 (Fallback)                                    │ │
│  │  • 本地模型                                                    │ │
│  │  • 预生成响应                                                  │ │
│  │  • 人工转接                                                    │ │
│  │  • 服务降级提示                                                │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、重试策略设计

### 3.1 指数退避重试

```java
@Component
public class RetryableLLMClient {
    
    private final LLMClient delegate;
    private final MeterRegistry registry;
    
    // 重试配置
    private static final int MAX_RETRIES = 3;
    private static final long BASE_DELAY_MS = 1000;
    private static final long MAX_DELAY_MS = 30000;
    
    public LLMResponse chat(String prompt) {
        int attempt = 0;
        Exception lastException = null;
        
        while (attempt <= MAX_RETRIES) {
            try {
                LLMResponse response = delegate.chat(prompt);
                
                // 验证响应质量
                if (isValidResponse(response)) {
                    recordSuccess(attempt);
                    return response;
                }
                
                // 响应无效，视为失败
                lastException = new InvalidResponseException(
                    "响应格式无效: " + response);
                    
            } catch (RateLimitException e) {
                // 速率限制：使用更长的退避时间
                long delay = calculateRateLimitDelay(e);
                log.warn("速率限制，等待 {}ms 后重试", delay);
                sleep(delay);
                lastException = e;
                
            } catch (ModelOverloadedException e) {
                // 模型过载：尝试降级到备用模型
                if (attempt < MAX_RETRIES) {
                    return fallbackToAlternative(prompt, e);
                }
                lastException = e;
                
            } catch (TransientException e) {
                // 临时故障：标准重试
                long delay = calculateBackoff(attempt);
                log.warn("临时故障，等待 {}ms 后重试", delay);
                sleep(delay);
                lastException = e;
                
            } catch (PermanentException e) {
                // 永久故障：立即失败，不重试
                recordFailure(e);
                throw new LLMServiceException("永久故障", e);
            }
            
            attempt++;
        }
        
        // 所有重试都失败
        recordFailure(lastException);
        throw new LLMServiceException("重试次数耗尽", lastException);
    }
    
    private long calculateBackoff(int attempt) {
        // 指数退避 + 随机抖动
        long delay = BASE_DELAY_MS * (long) Math.pow(2, attempt);
        long jitter = ThreadLocalRandom.current()
            .nextLong(0, delay / 2);
        return Math.min(delay + jitter, MAX_DELAY_MS);
    }
    
    private long calculateRateLimitDelay(RateLimitException e) {
        // 使用API返回的重试时间
        if (e.getRetryAfter() != null) {
            return e.getRetryAfter().toMillis();
        }
        // 默认等待60秒
        return 60000;
    }
    
    private boolean isValidResponse(LLMResponse response) {
        if (response == null || response.getContent() == null) {
            return false;
        }
        
        String content = response.getContent().trim();
        
        // 检查空响应
        if (content.isEmpty()) {
            return false;
        }
        
        // 检查是否是错误提示
        if (content.contains("I apologize") && 
            content.contains("cannot")) {
            return false;
        }
        
        // 检查Token消耗是否异常
        if (response.getUsage().getTotalTokens() > 4000) {
            log.warn("Token消耗异常: {}", 
                response.getUsage().getTotalTokens());
        }
        
        return true;
    }
}
```

### 3.2 电路断路器

```java
@Component
public class CircuitBreakerLLMClient {
    
    private final LLMClient delegate;
    private final CircuitBreaker circuitBreaker;
    
    public CircuitBreakerLLMClient(LLMClient delegate) {
        this.delegate = delegate;
        this.circuitBreaker = CircuitBreaker.custom()
            .failureRateThreshold(50)           // 失败率阈值50%
            .slowCallRateThreshold(80)          // 慢调用率阈值80%
            .slowCallDurationThreshold(Duration.ofSeconds(10))
            .waitDurationInOpenState(Duration.ofMinutes(1))
            .slidingWindowSize(10)
            .minimumNumberOfCalls(5)
            .recordExceptions(TransientException.class,
                            RateLimitException.class,
                            ModelOverloadedException.class)
            .ignoreExceptions(PermanentException.class)
            .build();
        
        // 监听状态变化
        circuitBreaker.getEventPublisher()
            .onStateTransition(event -> {
                log.warn("断路器状态变化: {} -> {}", 
                    event.getStateTransition().getFromState(),
                    event.getStateTransition().getToState());
                
                if (event.getStateTransition()
                    .getToState() == State.CLOSED) {
                    // 恢复正常，发送通知
                    notifyRecovery();
                }
            });
    }
    
    public LLMResponse chat(String prompt) {
        return circuitBreaker.executeSupplier(() -> 
            delegate.chat(prompt)
        );
    }
    
    // 熔断时的降级逻辑
    public LLMResponse chatWithFallback(String prompt) {
        try {
            return chat(prompt);
        } catch (CallNotPermittedException e) {
            log.warn("断路器开启，执行降级逻辑");
            return fallbackResponse(prompt);
        }
    }
    
    private LLMResponse fallbackResponse(String prompt) {
        // 方案1: 返回缓存的相似响应
        Optional<LLMResponse> cached = cacheService
            .findSimilar(prompt);
        if (cached.isPresent()) {
            return cached.get();
        }
        
        // 方案2: 返回预生成的通用响应
        return LLMResponse.builder()
            .content("系统暂时繁忙，请稍后再试。")
            .source("fallback")
            .build();
    }
}
```

---

## 四、模型降级策略

### 4.1 降级链设计

```java
@Service
public class ModelDegradationService {
    
    // 模型降级链：按优先级排列
    private final List<ModelEndpoint> degradationChain = List.of(
        new ModelEndpoint("gpt-4", "primary", 30),
        new ModelEndpoint("gpt-3.5-turbo", "secondary", 15),
        new ModelEndpoint("claude-3-sonnet", "tertiary", 20),
        new ModelEndpoint("llama-3-70b", "local", 5)
    );
    
    public LLMResponse chatWithDegradation(String prompt, 
                                            String preferredModel) {
        for (ModelEndpoint endpoint : degradationChain) {
            if (endpoint.getName().equals(preferredModel) || 
                endpoint.isAvailable()) {
                try {
                    LLMResponse response = callModel(endpoint, prompt);
                    
                    // 记录成功调用，更新健康状态
                    endpoint.recordSuccess();
                    
                    // 如果不是首选模型，记录降级
                    if (!endpoint.getName().equals(preferredModel)) {
                        recordDegradation(preferredModel, 
                                         endpoint.getName());
                    }
                    
                    return response;
                    
                } catch (Exception e) {
                    log.warn("模型 {} 调用失败: {}", 
                        endpoint.getName(), e.getMessage());
                    endpoint.recordFailure();
                    
                    // 标记模型为不可用
                    if (endpoint.getFailureCount() > 5) {
                        endpoint.setAvailable(false);
                        scheduleRecovery(endpoint);
                    }
                }
            }
        }
        
        // 所有模型都不可用
        throw new AllModelsUnavailableException(
            "所有模型均不可用，请检查服务状态");
    }
    
    @Data
    private static class ModelEndpoint {
        private final String name;
        private final String tier;
        private final int timeoutSeconds;
        private volatile boolean available = true;
        private final AtomicInteger failureCount = new AtomicInteger(0);
        private final AtomicLong lastFailureTime = new AtomicLong(0);
        
        public void recordSuccess() {
            failureCount.set(0);
            available = true;
        }
        
        public void recordFailure() {
            failureCount.incrementAndGet();
            lastFailureTime.set(System.currentTimeMillis());
        }
    }
}
```

### 4.2 智能路由决策

```java
@Component
public class SmartModelRouter {
    
    private final ModelHealthMonitor healthMonitor;
    private final CostTracker costTracker;
    private final LatencyTracker latencyTracker;
    
    public ModelEndpoint selectModel(String query, 
                                      ModelSelectionCriteria criteria) {
        
        // 1. 获取所有可用模型
        List<ModelEndpoint> available = healthMonitor
            .getAvailableModels();
        
        // 2. 根据任务复杂度筛选
        int complexity = analyzeComplexity(query);
        available = filterByComplexity(available, complexity);
        
        // 3. 根据成本约束筛选
        if (criteria.getMaxCost() != null) {
            available = filterByCost(available, criteria.getMaxCost());
        }
        
        // 4. 根据延迟要求筛选
        if (criteria.getMaxLatency() != null) {
            available = filterByLatency(available, 
                criteria.getMaxLatency());
        }
        
        // 5. 综合评分选择最优模型
        return available.stream()
            .max(Comparator.comparingDouble(model -> 
                calculateScore(model, complexity, criteria)))
            .orElseThrow(() -> new NoSuitableModelException(
                "没有合适的模型满足约束条件"));
    }
    
    private double calculateScore(ModelEndpoint model, 
                                   int complexity,
                                   ModelSelectionCriteria criteria) {
        double score = 0;
        
        // 健康度权重 (40%)
        double healthScore = healthMonitor.getHealthScore(model);
        score += healthScore * 0.4;
        
        // 成本权重 (30%)
        double costScore = 1.0 - costTracker
            .getRelativeCost(model);
        score += costScore * 0.3;
        
        // 延迟权重 (20%)
        double latencyScore = 1.0 - latencyTracker
            .getRelativeLatency(model);
        score += latencyScore * 0.2;
        
        // 质量权重 (10%)
        double qualityScore = getQualityScore(model, complexity);
        score += qualityScore * 0.1;
        
        return score;
    }
}
```

---

## 五、超时控制策略

### 5.1 多级超时设计

```java
@Configuration
public class TimeoutConfig {
    
    // 连接超时：与LLM服务建立连接的最大时间
    @Bean
    public Duration connectTimeout() {
        return Duration.ofSeconds(5);
    }
    
    // 读取超时：等待模型响应的最大时间
    @Bean
    public Duration readTimeout() {
        return Duration.ofSeconds(60);
    }
    
    // 总超时：整个请求的最大处理时间
    @Bean
    public Duration totalTimeout() {
        return Duration.ofSeconds(90);
    }
    
    // 流式响应超时：每个chunk的最大等待时间
    @Bean
    public Duration streamChunkTimeout() {
        return Duration.ofSeconds(10);
    }
}

@Service
public class TimeoutAwareLLMClient {
    
    private final LLMClient delegate;
    private final TimeoutConfig timeoutConfig;
    
    public LLMResponse chat(String prompt) {
        return CompletableFuture
            .supplyAsync(() -> delegate.chat(prompt))
            .orTimeout(timeoutConfig.getTotalTimeout().toMillis(), 
                      TimeUnit.MILLISECONDS)
            .exceptionally(ex -> {
                if (ex instanceof TimeoutException) {
                    log.warn("LLM调用超时，执行降级");
                    return fallbackForTimeout(prompt);
                }
                throw new CompletionException(ex);
            })
            .join();
    }
    
    // 流式响应的超时处理
    public Flux<LLMChunk> chatStream(String prompt) {
        return delegate.chatStream(prompt)
            .timeout(Duration.ofSeconds(30))
            .onErrorResume(TimeoutException.class, e -> {
                log.warn("流式响应超时");
                return Flux.error(new LLMServiceException(
                    "流式响应超时", e));
            });
    }
}
```

---

## 六、输出验证与安全过滤

### 6.1 输出质量验证

```java
@Component
public class OutputValidator {
    
    // 验证链：按顺序执行所有验证器
    private final List<OutputValidator> validators = List.of(
        new FormatValidator(),
        new ContentValidator(),
        new SafetyValidator(),
        new CostValidator()
    );
    
    public ValidationResult validate(String output, 
                                      ValidationContext context) {
        List<String> errors = new ArrayList<>();
        
        for (OutputValidator validator : validators) {
            ValidationResult result = validator.validate(output, context);
            if (!result.isValid()) {
                errors.addAll(result.getErrors());
            }
        }
        
        return ValidationResult.builder()
            .isValid(errors.isEmpty())
            .errors(errors)
            .build();
    }
}

// 格式验证
@Component
public class FormatValidator implements OutputValidator {
    
    @Override
    public ValidationResult validate(String output, 
                                      ValidationContext context) {
        List<String> errors = new ArrayList<>();
        
        // 检查空响应
        if (output == null || output.trim().isEmpty()) {
            errors.add("输出为空");
        }
        
        // 检查长度限制
        if (output.length() > context.getMaxOutputLength()) {
            errors.add("输出超过长度限制: " + output.length());
        }
        
        // 检查是否包含错误标记
        if (output.contains("Error:") || 
            output.contains("Exception:")) {
            errors.add("输出包含错误信息");
        }
        
        return ValidationResult.of(errors.isEmpty(), errors);
    }
}

// 安全验证
@Component
public class SafetyValidator implements OutputValidator {
    
    private final List<Pattern> prohibitedPatterns = List.of(
        Pattern.compile("(?i)ignore.*previous.*instructions"),
        Pattern.compile("(?i)you.*are.*now.*a"),
        Pattern.compile("(?i)pretend.*you.*are"),
        Pattern.compile("(?i)system.*prompt.*leak")
    );
    
    private final List<Pattern> sensitivePatterns = List.of(
        Pattern.compile("\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}"),
        Pattern.compile("\\d{18}"),
        Pattern.compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    );
    
    @Override
    public ValidationResult validate(String output, 
                                      ValidationContext context) {
        List<String> errors = new ArrayList<>();
        
        // 检查Prompt注入
        for (Pattern pattern : prohibitedPatterns) {
            if (pattern.matcher(output).find()) {
                errors.add("检测到潜在的Prompt注入攻击");
                break;
            }
        }
        
        // 检查敏感信息泄露
        for (Pattern pattern : sensitivePatterns) {
            if (pattern.matcher(output).find()) {
                errors.add("输出包含敏感信息");
                break;
            }
        }
        
        return ValidationResult.of(errors.isEmpty(), errors);
    }
}
```

### 6.2 自动修复

```java
@Service
public class OutputRepairService {
    
    private final LLMClient repairClient;
    
    public String repair(String output, List<String> errors) {
        String repairPrompt = String.format("""
            以下是一个AI输出，存在以下问题：
            %s
            
            请修复这些问题并返回修正后的输出：
            
            原始输出：
            %s
            
            要求：
            1. 保持原始内容的核心意思
            2. 修复格式问题
            3. 移除敏感信息
            4. 确保输出安全
            """, 
            String.join("\n", errors.stream()
                .map(e -> "- " + e)
                .toList()),
            output);
        
        return repairClient.chat(repairPrompt);
    }
}
```

---

## 七、成本控制策略

### 7.1 Token预算管理

```java
@Component
public class TokenBudgetManager {
    
    private final TokenCounter tokenCounter;
    private final CostCalculator costCalculator;
    
    // 检查是否超出预算
    public boolean isWithinBudget(String prompt, 
                                   String model,
                                   double maxBudget) {
        int estimatedTokens = tokenCounter.count(prompt);
        double estimatedCost = costCalculator.calculate(
            model, estimatedTokens);
        
        return estimatedCost <= maxBudget;
    }
    
    // 自动截断以适应预算
    public String truncateToBudget(String prompt, 
                                    String model,
                                    double maxBudget) {
        int maxTokens = costCalculator.getMaxTokens(
            model, maxBudget);
        
        if (tokenCounter.count(prompt) <= maxTokens) {
            return prompt;
        }
        
        // 智能截断：保留重要部分
        return smartTruncate(prompt, maxTokens);
    }
    
    private String smartTruncate(String prompt, int maxTokens) {
        // 优先保留系统提示和用户问题
        String[] parts = prompt.split("(?<=\\n)\\n");
        
        StringBuilder result = new StringBuilder();
        int currentTokens = 0;
        
        // 保留最后一部分（通常是用户问题）
        String userPart = parts[parts.length - 1];
        result.append(userPart);
        currentTokens += tokenCounter.count(userPart);
        
        // 从前往后添加其他部分，直到超出预算
        for (int i = 0; i < parts.length - 1; i++) {
            int partTokens = tokenCounter.count(parts[i]);
            if (currentTokens + partTokens <= maxTokens) {
                result.insert(0, parts[i] + "\n");
                currentTokens += partTokens;
            } else {
                break;
            }
        }
        
        return result.toString();
    }
}
```

### 7.2 成本监控告警

```java
@Component
public class CostMonitor {
    
    private final MeterRegistry registry;
    private final AlertService alertService;
    
    private final AtomicReference<BigDecimal> dailyCost = 
        new AtomicReference<>(BigDecimal.ZERO);
    
    private static final BigDecimal DAILY_BUDGET = 
        new BigDecimal("100.00");
    
    public void recordCost(String model, int tokens, 
                           double cost) {
        dailyCost.updateAndGet(current -> 
            current.add(BigDecimal.valueOf(cost)));
        
        // 记录指标
        registry.gauge("ai.cost.daily", dailyCost.get());
        registry.counter("ai.cost.total", 
            "model", model)
            .increment(cost);
        
        // 检查告警阈值
        BigDecimal ratio = dailyCost.get()
            .divide(DAILY_BUDGET, 2, RoundingMode.HALF_UP);
        
        if (ratio.compareTo(new BigDecimal("0.8")) >= 0) {
            alertService.sendAlert(AlertSeverity.WARNING,
                "AI成本已达日预算的" + ratio.multiply(
                    BigDecimal.valueOf(100)) + "%");
        }
        
        if (ratio.compareTo(BigDecimal.ONE) >= 0) {
            alertService.sendAlert(AlertSeverity.CRITICAL,
                "AI成本已超出日预算！");
            
            // 触发紧急降级
            triggerEmergencyDegradation();
        }
    }
}
```

---

## 八、监控与可观测性

### 8.1 关键指标定义

```java
@Component
public class LLMMetrics {
    
    private final Counter requestCounter;
    private final Counter errorCounter;
    private final Timer latencyTimer;
    private final DistributionSummary tokenUsage;
    private final Gauge activeRequests;
    
    public LLMMetrics(MeterRegistry registry) {
        // 请求计数
        this.requestCounter = Counter.builder("llm.requests.total")
            .description("Total LLM requests")
            .tag("model", "gpt-4")
            .register(registry);
        
        // 错误计数
        this.errorCounter = Counter.builder("llm.errors.total")
            .description("Total LLM errors")
            .tag("model", "gpt-4")
            .register(registry);
        
        // 延迟分布
        this.latencyTimer = Timer.builder("llm.request.duration")
            .description("LLM request latency")
            .publishPercentiles(0.5, 0.9, 0.99)
            .publishPercentileHistogram()
            .register(registry);
        
        // Token使用分布
        this.tokenUsage = DistributionSummary.builder("llm.tokens")
            .description("Token usage per request")
            .publishPercentiles(0.5, 0.9, 0.99)
            .register(registry);
        
        // 活跃请求数
        AtomicInteger active = new AtomicInteger(0);
        this.activeRequests = Gauge.builder(
            "llm.requests.active", active, AtomicInteger::get)
            .description("Active LLM requests")
            .register(registry);
    }
    
    public void recordRequest(String model, long latencyMs,
                              int inputTokens, int outputTokens,
                              boolean success) {
        requestCounter.increment();
        latencyTimer.record(latencyMs, TimeUnit.MILLISECONDS);
        tokenUsage.record(inputTokens + outputTokens);
        
        if (!success) {
            errorCounter.increment();
        }
    }
}
```

### 8.2 告警规则

```yaml
# prometheus-rules.yml
groups:
  - name: llm-alerts
    rules:
      # 错误率告警
      - alert: HighLLMErrorRate
        expr: rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "LLM错误率过高"
          description: "过去5分钟错误率超过10%"
      
      # 延迟告警
      - alert: HighLLMLatency
        expr: histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m])) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "LLM延迟过高"
          description: "P99延迟超过10秒"
      
      # 成本告警
      - alert: HighLLMCost
        expr: ai_cost_daily > 80
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AI成本接近预算上限"
          description: "当日成本已超过80元"
```

---

## 九、完整容错框架

### 9.1 统一容错入口

```java
@Service
public class ResilientLLMService {
    
    private final ModelDegradationService degradationService;
    private final RetryableLLMClient retryClient;
    private final CircuitBreakerLLMClient circuitBreakerClient;
    private final OutputValidator outputValidator;
    private final CostMonitor costMonitor;
    private final LLMMetrics metrics;
    
    public LLMResponse chat(String prompt, String model) {
        long startTime = System.currentTimeMillis();
        boolean success = false;
        
        try {
            // 1. 成本预检
            if (!costMonitor.isWithinBudget(prompt, model, 1.0)) {
                model = "gpt-3.5-turbo";  // 降级到经济模型
            }
            
            // 2. 执行调用（带重试和熔断）
            LLMResponse response = circuitBreakerClient.chatWithFallback(
                prompt, model, degradationService);
            
            // 3. 输出验证
            ValidationResult validation = outputValidator
                .validate(response.getContent());
            
            if (!validation.isValid()) {
                // 尝试修复
                String repaired = outputRepairService.repair(
                    response.getContent(), validation.getErrors());
                response.setContent(repaired);
            }
            
            // 4. 记录成本
            costMonitor.recordCost(model, 
                response.getUsage().getTotalTokens(),
                calculateCost(model, response.getUsage()));
            
            success = true;
            return response;
            
        } finally {
            long latency = System.currentTimeMillis() - startTime;
            metrics.recordRequest(model, latency, 0, 0, success);
        }
    }
}
```

---

## 十、总结与最佳实践

### 10.1 容错策略选择矩阵

| 场景 | 推荐策略 | 实现复杂度 | 效果 |
|------|---------|-----------|------|
| 偶发网络故障 | 指数退避重试 | 低 | 高 |
| API限流 | 速率限制 + 退避 | 低 | 高 |
| 模型过载 | 模型降级链 | 中 | 高 |
| 服务不可用 | 断路器 + 缓存回退 | 中 | 高 |
| 输出异常 | 验证 + 修复 | 高 | 中 |
| 成本失控 | Token预算 + 监控 | 中 | 高 |

### 10.2 关键原则

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM应用容错最佳实践                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 预防优于恢复                                                     │
│     • 输入验证、Token预算、速率限制                                   │
│                                                                     │
│  2. 快速失败                                                         │
│     • 永久故障立即失败，不浪费重试次数                                 │
│                                                                     │
│  3. 优雅降级                                                         │
│     • 提供降级选项，而非完全失败                                      │
│                                                                     │
│  4. 全面监控                                                         │
│     • 延迟、错误率、成本、Token消耗                                   │
│                                                                     │
│  5. 自动恢复                                                         │
│     • 断路器自动恢复、模型健康检查                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

构建高可用的LLM应用需要系统性地考虑容错设计。通过多层防护、智能降级、全面监控，可以显著提升系统的可靠性和用户体验。

---

## 参考资源

- [OpenAI API错误处理指南](https://platform.openai.com/docs/guides/error-codes)
- [Resilience4j官方文档](https://resilience4j.readme.io/)
- [Spring Retry参考文档](https://docs.spring.io/spring-retry/reference/)
