---
title: "LangChain4j 实战进阶：从核心组件到生产部署的深度指南"
description: "深入解析 LangChain4j 的核心架构、Spring Boot 集成、自定义 Tool/Retriever 开发以及生产环境性能优化策略，附完整代码示例与架构图。"
date: 2026-05-30
author: RiceBall-15
category: framework
subCategory: agent-framework
tags:
  - LangChain4j
  - LLM
  - AI Agent
  - Spring Boot
  - RAG
  - Java
  - 机器学习
draft: false
---

# LangChain4j 实战进阶：从核心组件到生产部署的深度指南

## 引言

LangChain4j 是 Java 生态中最活跃的 LLM 应用开发框架，它将 LangChain 的核心理念带入了 JVM 世界，提供了类型安全、与 Spring 生态深度集成的 AI 应用开发体验。本文将从架构层面深入剖析 LangChain4j 的核心组件，并通过大量实战代码展示如何在生产环境中构建高质量的 AI Agent 应用。

---

## 一、LangChain4j 核心组件深度解析

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      LangChain4j 架构全景图                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   AiService   │  │    Chain     │  │        Agent         │  │
│  │  (声明式接口)  │  │ (顺序执行链)  │  │  (自主决策 + 工具调用) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│  ┌──────┴─────────────────┴──────────────────────┴───────────┐  │
│  │                     Core Layer                             │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │  │
│  │  │ LLM/Chat│  │  Memory   │  │  Tools   │  │ Retriever │  │  │
│  │  │ Model   │  │ (对话记忆) │  │ (外部能力)│  │ (检索增强) │  │  │
│  │  └────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬─────┘  │  │
│  └───────┼─────────────┼──────────────┼─────────────┼─────────┘  │
│          │             │              │             │            │
│  ┌───────┴─────────────┴──────────────┴─────────────┴─────────┐  │
│  │                   Integration Layer                        │  │
│  │  Spring Boot │ Quarkus │ MicroProfile │ Plain Java        │  │
│  └────────────────────────────────────────────────────────────┘  │
│          │             │              │             │            │
│  ┌───────┴─────────────┴──────────────┴─────────────┴─────────┐  │
│  │                   Provider Layer                            │  │
│  │ OpenAI │ Azure │ Ollama │ Anthropic │ Google │ Local Models │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 ChatModel — 对话模型的抽象与封装

LangChain4j 对 LLM 的抽象是整个框架的基石。所有模型交互都通过 `ChatLanguageModel` 或 `StreamingChatLanguageModel` 接口完成：

```java
// 1. 基础对话模型
ChatLanguageModel model = OpenAiChatModel.builder()
    .apiKey(System.getenv("OPENAI_API_KEY"))
    .modelName("gpt-4o")
    .temperature(0.7)
    .timeout(Duration.ofSeconds(30))
    .maxTokens(4096)
    .build();

// 2. 流式输出模型 — 用于实时响应场景
StreamingChatLanguageModel streamingModel = OpenAiStreamingChatModel.builder()
    .apiKey(System.getenv("OPENAI_API_KEY"))
    .modelName("gpt-4o")
    .temperature(0.7)
    .build();

// 3. 支持多模态的视觉模型
String imageUrl = "https://example.com/image.png";
String response = visionModel.chat(
    ChatMessage.user(
        MessageContent.image(imageUrl),
        MessageContent.text("描述这张图片的内容")
    )
);
```

### 1.3 Memory — 对话记忆机制

记忆系统是构建有状态对话应用的关键。LangChain4j 提供了多层次的记忆抽象：

```
┌─────────────────────────────────────────────────┐
│              Memory 类型层次                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  ChatMemory (接口)                               │
│    ├── MessageWindowChatMemory                   │
│    │     └── 滑动窗口，保留最近 N 条消息            │
│    ├── TokenWindowChatMemory                      │
│    │     └── 按 Token 数量控制上下文长度             │
│    ├── MessageStoreChatMemory                     │
│    │     └── 持久化存储，支持跨会话记忆              │
│    └── CompositeChatMemory                        │
│          └── 组合多种记忆策略                       │
│                                                  │
│  ChatMemoryProvider (Provider 接口)               │
│    └── 为不同用户动态提供独立的 Memory 实例          │
│                                                  │
└─────────────────────────────────────────────────┘
```

```java
// 方式一：滑动窗口记忆 — 最简单的记忆策略
ChatMemory memory = MessageWindowChatMemory.builder()
    .maxMessages(20)          // 保留最近 20 条消息
    .id("user-session-123")
    .build();

// 方式二：Token 窗口记忆 — 精确控制上下文长度
ChatMemory tokenMemory = TokenWindowChatMemory.builder()
    .maxTokens(4000)          // 控制在 4000 Token 以内
    .tokenCountEstimator(new OpenAiTokenCountEstimator())
    .id("user-session-123")
    .build();

// 方式三：基于持久化存储的记忆 — 跨会话保持
ChatMemory persistentMemory = MessageStoreChatMemory.builder()
    .chatMemoryStore(new InMemoryChatMemoryStore())  // 可替换为 Redis/DB
    .maxMessages(50)
    .id("user-session-123")
    .build();

// 方式四：组合记忆 — 同时使用短期和长期记忆
ChatMemory compositeMemory = CompositeChatMemory.builder()
    .add("short-term", MessageWindowChatMemory.withMaxMessages(10))
    .add("long-term", persistentMemory)
    .build();
```

### 1.4 AiService — 声明式 AI 接口

AiService 是 LangChain4j 最优雅的抽象之一，它让你用定义 Java 接口的方式来构建 AI 应用：

```java
// 定义 AI 服务接口
@AiService
interface Assistant {

    @SystemMessage("""
        你是一名专业的技术文档翻译专家。
        请将用户输入的技术文档翻译为中文，保持专业术语准确。
        """)
    String translate(@UserMessage String document);

    // 带记忆的聊天 — LangChain4j 会自动注入 @MemoryId 对应的 ChatMemory
    @MemoryId
    @SystemMessage("你是一个友好的技术助手，回答简洁明了。")
    String chatWithMemory(@MemoryId String sessionId, @UserMessage String message);

    // 流式输出
    @SystemMessage("你是一个创意写作助手。")
    @Temperature(0.9)
    StreamingChatLanguageModel.StreamingResponseHandler<String> streamWrite(
        @UserMessage String topic
    );
}

// 构建 AiService
AiServices.builder(Assistant.class)
    .chatLanguageModel(model)
    .chatMemoryProvider(memoryId -> 
        MessageWindowChatMemory.builder()
            .maxMessages(20)
            .id(memoryId)
            .build()
    )
    .build();
```

### 1.5 Chain — 链式调用与编排

Chain 模式允许将多个处理步骤串行组合，形成可复用的处理管道：

```java
// 定义一个摘要链
public class SummaryChain {

    private final ChatLanguageModel model;

    public SummaryChain(ChatLanguageModel model) {
        this.model = model;
    }

    public String execute(String longDocument) {
        // Step 1: 提取关键信息
        String keyPoints = model.chat(
            PromptTemplate.from("请提取以下文档的关键信息：\n\n{{document}}")
                .apply(Map.of("document", longDocument))
                .contents()
        );

        // Step 2: 生成摘要
        String summary = model.chat(
            PromptTemplate.from("基于以下关键信息，生成一段简洁的摘要：\n\n{{keyPoints}}")
                .apply(Map.of("keyPoints", keyPoints))
                .contents()
        );

        // Step 3: 优化语言
        String refined = model.chat(
            PromptTemplate.from("将以下摘要优化为专业技术文档风格：\n\n{{summary}}")
                .apply(Map.of("summary", summary))
                .contents()
        );

        return refined;
    }
}

// 使用 LangChain4j 内置的 AiServices 编排多步流程
@AiService
interface AnalysisPipeline {

    @SystemMessage("你是数据分析师。")
    String analyzeData(@SystemMessage("步骤1：识别数据中的异常值") @UserMessage String data);

    @SystemMessage("你是报告撰写专家。")
    String writeReport(@SystemMessage("步骤2：基于分析结果撰写报告") @UserMessage String analysis);
}
```

### 1.6 Agent — 自主决策与工具调用

Agent 是 LangChain4j 的核心能力，它让 LLM 能够自主决定何时调用外部工具：

```
┌─────────────────────────────────────────────────────────┐
│                   Agent 执行流程                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  用户输入 ──▶ LLM 推理 ──▶ 决策分支                      │
│                                    │                     │
│                    ┌───────────────┼───────────────┐     │
│                    ▼               ▼               ▼     │
│              直接回复        调用工具 1        调用工具 2  │
│                                  │               │      │
│                                  ▼               ▼      │
│                          获取工具结果    获取工具结果     │
│                                  │               │      │
│                                  └───────┬───────┘      │
│                                          ▼              │
│                              反馈给 LLM 继续推理          │
│                                          │              │
│                                          ▼              │
│                              最终回复（或再次调用工具）     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

```java
// 定义工具类
public class WeatherTools {

    @Tool("获取指定城市的当前天气信息")
    public String getWeather(
        @ToolParameter(description = "城市名称，如：北京、上海") String city
    ) {
        // 实际项目中调用天气 API
        return String.format("%s 当前天气：晴，温度 25°C，湿度 45%%", city);
    }

    @Tool("获取指定城市的未来7天天气预报")
    public String getForecast(
        @ToolParameter(description = "城市名称") String city,
        @ToolParameter(description = "天数，1-7") int days
    ) {
        return String.format("%s 未来 %d 天预报：晴转多云，温度 22-30°C", city, days);
    }
}

// 定义带工具的 Agent 服务
@AiService
interface WeatherAgent {

    @SystemMessage("""
        你是一个天气助手。当用户询问天气相关问题时，
        使用提供的工具获取实时数据并给出建议。
        """)
    String answer(@UserMessage String question);
}

// 构建 Agent
WeatherAgent agent = AiServices.builder(WeatherAgent.class)
    .chatLanguageModel(model)
    .tools(new WeatherTools())                    // 注册工具
    .build();

String result = agent.answer("明天去北京出差，天气怎么样？需要带伞吗？");
```

---

## 二、与 Spring Boot 集成的最佳实践

### 2.1 项目配置

```xml
<!-- pom.xml 依赖配置 -->
<properties>
    <langchain4j.version>1.0.0-beta3</langchain4j.version>
</properties>

<dependencies>
    <!-- LangChain4j Spring Boot Starter -->
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-spring-boot-starter</artifactId>
        <version>${langchain4j.version}</version>
    </dependency>
    
    <!-- OpenAI 模型支持 -->
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-open-ai-spring-boot-starter</artifactId>
        <version>${langchain4j.version}</version>
    </dependency>
    
    <!-- RAG 支持 -->
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-embeddings-all-MiniLM-L6-v2-spring-boot-starter</artifactId>
        <version>${langchain4j.version}</version>
    </dependency>
    
    <!-- 向量存储 -->
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-pgvector-spring-boot-starter</artifactId>
        <version>${langchain4j.version}</version>
    </dependency>
</dependencies>
```

```yaml
# application.yml
langchain4j:
  open-ai:
    chat-model:
      api-key: ${OPENAI_API_KEY}
      model-name: gpt-4o
      temperature: 0.7
      log-requests: true
      log-responses: false           # 生产环境建议关闭
      timeout: PT30S
    streaming-chat-model:
      api-key: ${OPENAI_API_KEY}
      model-name: gpt-4o
    embedding-model:
      api-key: ${OPENAI_API_KEY}
      model-name: text-embedding-3-small

  pgvector:
    url: jdbc:postgresql://localhost:5432/ai_vectors
    user: ${DB_USER}
    password: ${DB_PASSWORD}
    table-name: document_embeddings
    dimension: 1536
```

### 2.2 项目结构设计

```
src/main/java/com/example/
├── config/
│   ├── AiServiceConfig.java        # AI 服务配置
│   ├── MemoryConfig.java           # 记忆配置
│   └── RagConfig.java              # RAG 配置
├── service/
│   ├── agent/
│   │   ├── AssistantService.java   # 主助手服务
│   │   ├── ResearchAgent.java      # 研究型 Agent
│   │   └── CodingAgent.java        # 编程辅助 Agent
├── tool/
│   ├── DatabaseTool.java           # 数据库查询工具
│   ├── ApiCallTool.java            # API 调用工具
│   └── FileOperationTool.java      # 文件操作工具
├── memory/
│   ├── RedisChatMemoryStore.java   # Redis 记忆存储
│   └── UserMemoryProvider.java     # 用户级记忆管理
├── rag/
│   ├── DocumentProcessor.java      # 文档处理器
│   ├── CustomRetriever.java        # 自定义检索器
│   └── EmbeddingService.java       # 嵌入服务
└── controller/
    └── AiController.java           # REST 接口
```

### 2.3 完整的 Spring Boot 集成示例

```java
// ========== 配置类 ==========

@Configuration
@EnableConfigurationProperties
public class AiServiceConfig {

    @Bean
    public ChatMemoryStore chatMemoryStore(RedisTemplate<String, String> redis) {
        return new RedisChatMemoryStore(redis);
    }

    @Bean
    public ChatMemoryProvider chatMemoryProvider(ChatMemoryStore store) {
        return memoryId -> MessageStoreChatMemory.builder()
            .chatMemoryStore(store)
            .maxMessages(30)
            .id(memoryId)
            .build();
    }

    @Bean
    public AssistantService assistantService(
        ChatLanguageModel chatModel,
        StreamingChatLanguageModel streamingModel,
        ChatMemoryProvider memoryProvider,
        DatabaseTool dbTool,
        ApiCallTool apiTool
    ) {
        return AiServices.builder(AssistantService.class)
            .chatLanguageModel(chatModel)
            .streamingChatLanguageModel(streamingModel)
            .chatMemoryProvider(memoryProvider)
            .tools(dbTool, apiTool)
            .build();
    }
}

// ========== Redis 记忆存储 ==========

@Component
public class RedisChatMemoryStore implements ChatMemoryStore {

    private final RedisTemplate<String, String> redis;
    private static final Duration TTL = Duration.ofHours(2);

    public RedisChatMemoryStore(RedisTemplate<String, String> redis) {
        this.redis = redis;
    }

    @Override
    public List<ChatMessage> getMessages(String memoryId) {
        String json = redis.opsForValue().get("chat:memory:" + memoryId);
        if (json == null) {
            return new ArrayList<>();
        }
        return deserializeMessages(json);
    }

    @Override
    public void updateMessages(String memoryId, List<ChatMessage> messages) {
        String json = serializeMessages(messages);
        redis.opsForValue().set("chat:memory:" + memoryId, json, TTL);
    }

    @Override
    public void deleteMessages(String memoryId) {
        redis.delete("chat:memory:" + memoryId);
    }
}

// ========== AI 服务接口 ==========

@AiService
public interface AssistantService {

    @SystemMessage("""
        你是一个全能技术助手，能够：
        1. 回答技术问题
        2. 查询数据库信息
        3. 调用外部 API 获取实时数据
        请根据用户需求选择合适的工具来完成任务。
        """)
    String chat(@MemoryId String sessionId, @UserMessage String message);

    // 流式输出版本
    @SystemMessage("你是一个全能技术助手。")
    void streamChat(
        @MemoryId String sessionId,
        @UserMessage String message,
        TokenStreamHandler handler
    );
}

// ========== 控制器 ==========

@RestController
@RequestMapping("/api/ai")
@CrossOrigin
public class AiController {

    private final AssistantService assistant;

    public AiController(AssistantService assistant) {
        this.assistant = assistant;
    }

    @PostMapping("/chat")
    public ChatResponse chat(@RequestBody ChatRequest request) {
        String response = assistant.chat(request.getSessionId(), request.getMessage());
        return new ChatResponse(response, request.getSessionId());
    }

    @PostMapping("/chat/stream")
    public void streamChat(
        @RequestBody ChatRequest request,
        SseEmitter emitter
    ) {
        StringBuilder fullResponse = new StringBuilder();
        
        assistant.streamChat(request.getSessionId(), request.getMessage(), 
            TokenStreamHandler.builder()
                .onPartialResponse(token -> emitter.send(SseEmitter.event().data(token)))
                .onCompleteResponse(response -> emitter.complete())
                .onError(error -> emitter.completeWithError(error))
                .build()
        );
    }
}
```

---

## 三、自定义 Tool 和 Retriever 开发

### 3.1 自定义 Tool 开发实战

```java
/**
 * 自定义数据库查询工具 — 支持安全的只读 SQL 查询
 */
@Component
public class DatabaseQueryTool {

    private final JdbcTemplate jdbcTemplate;
    private final ChatLanguageModel model;

    public DatabaseQueryTool(JdbcTemplate jdbcTemplate, ChatLanguageModel model) {
        this.jdbcTemplate = jdbcTemplate;
        this.model = model;
    }

    @Tool("""
        执行只读数据库查询。当用户需要查询数据、统计信息或报告时使用此工具。
        注意：仅支持 SELECT 查询，不允许修改数据。
        """)
    public String executeQuery(
        @ToolParameter(description = "要执行的 SQL 查询语句，仅允许 SELECT") 
        String sqlQuery
    ) {
        // 安全校验 — 只允许 SELECT
        String normalizedSql = sqlQuery.trim().toUpperCase();
        if (!normalizedSql.startsWith("SELECT")) {
            return "错误：仅支持 SELECT 查询";
        }

        // 检查危险关键词
        if (containsDangerousKeywords(normalizedSql)) {
            return "错误：查询包含不允许的关键词";
        }

        try {
            List<Map<String, Object>> results = jdbcTemplate.queryForList(sqlQuery);

            if (results.isEmpty()) {
                return "查询结果为空";
            }

            // 如果结果过多，让 LLM 做摘要
            if (results.size() > 20) {
                String dataSummary = formatResults(results.subList(0, 20));
                String additionalInfo = String.format(
                    "（共 %d 条结果，仅显示前 20 条）", results.size()
                );

                String summarized = model.chat(
                    PromptTemplate.from("""
                        基于以下查询结果，生成一份简洁的数据摘要。
                        结果数据：
                        {{data}}
                        {{info}}
                        """)
                    .apply(Map.of("data", dataSummary, "info", additionalInfo))
                    .contents()
                );
                return summarized;
            }

            return formatResults(results);
        } catch (Exception e) {
            return "查询执行失败：" + e.getMessage();
        }
    }

    @Tool("获取数据库中的表结构信息")
    public String getTableSchema(
        @ToolParameter(description = "表名称") String tableName
    ) {
        try {
            List<Map<String, Object>> columns = jdbcTemplate.queryForList(
                "SELECT column_name, data_type, is_nullable " +
                "FROM information_schema.columns " +
                "WHERE table_name = ?", tableName
            );
            return formatResults(columns);
        } catch (Exception e) {
            return "获取表结构失败：" + e.getMessage();
        }
    }

    private boolean containsDangerousKeywords(String sql) {
        String[] dangerous = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"};
        for (String keyword : dangerous) {
            if (sql.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    private String formatResults(List<Map<String, Object>> results) {
        StringBuilder sb = new StringBuilder();
        if (!results.isEmpty()) {
            // 表头
            Set<String> columns = results.get(0).keySet();
            sb.append(String.join(" | ", columns)).append("\n");
            sb.append("-".repeat(columns.size() * 15)).append("\n");

            // 数据行
            for (Map<String, Object> row : results) {
                String line = columns.stream()
                    .map(col -> String.valueOf(row.get(col)))
                    .collect(Collectors.joining(" | "));
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }
}

/**
 * 外部 API 调用工具 — 带重试和超时控制
 */
@Component
public class ExternalApiTool {

    private final RestClient restClient;
    private final RetryTemplate retryTemplate;

    public ExternalApiTool(RestClient.Builder builder, RetryTemplate retryTemplate) {
        this.restClient = builder.baseUrl("https://api.example.com")
            .defaultHeader("Accept", "application/json")
            .build();
        this.retryTemplate = retryTemplate;
    }

    @Tool("调用外部 REST API 获取数据")
    public String callApi(
        @ToolParameter(description = "API 端点路径，如 /users/list") String endpoint,
        @ToolParameter(description = "查询参数，JSON 格式") String queryParams
    ) {
        try {
            String response = retryTemplate.execute(context ->
                restClient.get()
                    .uri(uriBuilder -> {
                        uriBuilder.path(endpoint);
                        // 解析查询参数
                        Map<String, String> params = parseJson(queryParams);
                        params.forEach(uriBuilder::queryParam);
                        return uriBuilder.build();
                    })
                    .retrieve()
                    .body(String.class)
            );

            return response;
        } catch (Exception e) {
            return "API 调用失败：" + e.getMessage();
        }
    }
}
```

### 3.2 自定义 Retriever 开发实战

```
┌─────────────────────────────────────────────────────────┐
│                   RAG Pipeline 架构                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  文档导入阶段：                                            │
│  ┌────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐  │
│  │ 文档源  │──▶│ 文档解析器 │──▶│ 文本分割  │──▶│ 嵌入  │  │
│  │PDF/MD  │   │          │   │          │   │ 向量化 │  │
│  └────────┘   └──────────┘   └──────────┘   └───┬───┘  │
│                                                  │      │
│                                                  ▼      │
│  ┌─────────────────────────────────────────────────┐    │
│  │            向量数据库 (PGVector/Milvus)          │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│  查询阶段：               │                                │
│                         ▼                                │
│  ┌────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐  │
│  │ 用户查询 │──▶│ 查询嵌入  │──▶│ 相似度搜索│──▶│重排序 │  │
│  └────────┘   └──────────┘   └──────────┘   └───┬───┘  │
│                                                  │      │
│                                                  ▼      │
│                                       ┌───────────────┐  │
│                                       │  LLM 生成答案  │  │
│                                       │  (带引用来源)   │  │
│                                       └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

```java
/**
 * 自定义混合检索器 — 结合向量搜索 + 关键词搜索 + BM25
 */
@Component
public class HybridRetriever implements Retriever {

    private final EmbeddingStore<TextSegment> embeddingStore;
    private final EmbeddingModel embeddingModel;
    private final Analyzer chineseAnalyzer;
    private final Map<String, DocumentIndex> bm25Index;  // BM25 倒排索引

    public HybridRetriever(
        EmbeddingStore<TextSegment> embeddingStore,
        EmbeddingModel embeddingModel
    ) {
        this.embeddingStore = embeddingStore;
        this.embeddingModel = embeddingModel;
        this.chineseAnalyzer = new SmartChineseAnalyzer();
        this.bm25Index = new ConcurrentHashMap<>();
    }

    @Override
    public List<Content> retrieve(DevUserMessage userMessage) {
        String query = userMessage.singleText();

        // 并行执行多种检索策略
        CompletableFuture<List<EmbeddingMatch<TextSegment>>> vectorFuture =
            CompletableFuture.supplyAsync(() -> vectorSearch(query));

        CompletableFuture<List<ScoredDocument>> bm25Future =
            CompletableFuture.supplyAsync(() -> bm25Search(query));

        CompletableFuture<List<EmbeddingMatch<TextSegment>>> keywordFuture =
            CompletableFuture.supplyAsync(() -> keywordSearch(query));

        // 等待所有结果
        CompletableFuture.allOf(vectorFuture, bm25Future, keywordFuture).join();

        List<EmbeddingMatch<TextSegment>> vectorResults = vectorFuture.join();
        List<ScoredDocument> bm25Results = bm25Future.join();
        List<EmbeddingMatch<TextSegment>> keywordResults = keywordFuture.join();

        // Reciprocal Rank Fusion 合并结果
        List<RankedResult> merged = reciprocalRankFusion(
            vectorResults, bm25Results, keywordResults,
            weights: new double[]{0.5, 0.3, 0.2}  // 向量检索权重最高
        );

        // 重排序 — 使用交叉编码器或 LLM
        List<RankedResult> reranked = rerank(query, merged);

        // 返回 Top-K
        return reranked.stream()
            .limit(10)
            .map(result -> Content.from(result.textSegment, result.score))
            .collect(Collectors.toList());
    }

    private List<EmbeddingMatch<TextSegment>> vectorSearch(String query) {
        TextSegment querySegment = TextSegment.from(query);
        Embedding queryEmbedding = embeddingModel.embed(querySegment).content();

        return embeddingStore.findRelevant(
            queryEmbedding,
            20,           // 召回更多候选
            0.6           // 相似度阈值
        );
    }

    private List<ScoredDocument> bm25Search(String query) {
        List<String> terms = analyzeQuery(query);
        Map<String, ScoredDocument> docScores = new HashMap<>();

        for (String term : terms) {
            List<ScoredDocument> docs = bm25Index.getOrDefault(term, 
                Collections.emptyList());
            for (ScoredDocument doc : docs) {
                docScores.merge(doc.id, doc, 
                    (existing, newDoc) -> new ScoredDocument(
                        existing.id, 
                        existing.text,
                        existing.score + newDoc.score
                    ));
            }
        }

        return docScores.values().stream()
            .sorted(Comparator.comparingDouble(ScoredDocument::score).reversed())
            .limit(20)
            .collect(Collectors.toList());
    }

    /**
     * Reciprocal Rank Fusion 算法
     * RRF(d) = Σ 1/(k + rank_i(d))
     */
    private List<RankedResult> reciprocalRankFusion(
        List<EmbeddingMatch<TextSegment>> vectorResults,
        List<ScoredDocument> bm25Results,
        List<EmbeddingMatch<TextSegment>> keywordResults,
        double[] weights
    ) {
        Map<String, RankedResult> scores = new HashMap<>();
        int k = 60;  // RRF 常量

        // 向量检索结果
        for (int i = 0; i < vectorResults.size(); i++) {
            EmbeddingMatch<TextSegment> match = vectorResults.get(i);
            String id = match.id();
            scores.merge(id, new RankedResult(id, match.embeddingMatch().segment(), 
                weights[0] / (k + i + 1)),
                (a, b) -> new RankedResult(id, a.textSegment, a.score + b.score));
        }

        // BM25 结果
        for (int i = 0; i < bm25Results.size(); i++) {
            ScoredDocument doc = bm25Results.get(i);
            String id = doc.id;
            scores.merge(id, new RankedResult(id, TextSegment.from(doc.text), 
                weights[1] / (k + i + 1)),
                (a, b) -> new RankedResult(id, a.textSegment, a.score + b.score));
        }

        // 关键词检索结果
        for (int i = 0; i < keywordResults.size(); i++) {
            EmbeddingMatch<TextSegment> match = keywordResults.get(i);
            String id = match.id();
            scores.merge(id, new RankedResult(id, match.embeddingMatch().segment(), 
                weights[2] / (k + i + 1)),
                (a, b) -> new RankedResult(id, a.textSegment, a.score + b.score));
        }

        return scores.values().stream()
            .sorted(Comparator.comparingDouble(RankedResult::score).reversed())
            .collect(Collectors.toList());
    }

    private List<String> analyzeQuery(String query) {
        TokenStream tokenStream = chineseAnalyzer.tokenStream("field", query);
        List<String> terms = new ArrayList<>();
        try {
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
                String term = tokenStream.getAttribute(OffsetAttribute.class)
                    .toString();
                if (term.length() > 1) {  // 过滤单字符
                    terms.add(term);
                }
            }
            tokenStream.end();
            tokenStream.close();
        } catch (IOException e) {
            throw new RuntimeException("分词失败", e);
        }
        return terms;
    }
}

/**
 * 文档处理器 — 负责文档的解析、分割和存储
 */
@Component
public class DocumentProcessor {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;

    public DocumentProcessor(EmbeddingModel embeddingModel, 
                              EmbeddingStore<TextSegment> embeddingStore) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
    }

    /**
     * 处理 PDF 文档
     */
    public ProcessingResult processPdfDocument(String filePath) throws IOException {
        // 1. 解析 PDF
        Document document = ApacheTikaDocumentParser.builder()
            .build()
            .parse(new File(filePath));

        return processDocument(document, filePath);
    }

    /**
     * 处理 Markdown 文档 — 保持章节结构
     */
    public ProcessingResult processMarkdownDocument(String content, String source) {
        // 使用 Markdown 分割器，按标题层级分割
        Document document = Document.from(content, 
            Metadata.from("source", source));

        return processDocument(document, source);
    }

    private ProcessingResult processDocument(Document document, String source) {
        // 2. 文本分割 — 使用递归字符分割器
        TextSplitter splitter = new RecursiveCharacterTextSplitter(
            1000,      // 每块最大 1000 字符
            200,       // 重叠 200 字符
            "\\n\\n",  // 优先按段落分割
            "\\n",     // 其次按行分割
            " "        // 最后按空格分割
        );

        List<TextSegment> segments = splitter.split(document);

        // 3. 批量嵌入
        List<Embedding> embeddings = embeddingModel.embedAll(segments)
            .contents();

        // 4. 存入向量数据库
        List<String> ids = new ArrayList<>();
        for (int i = 0; i < segments.size(); i++) {
            String id = UUID.randomUUID().toString();
            Metadata metadata = segments.get(i).metadata()
                .put("source", source)
                .put("chunk_index", i)
                .put("total_chunks", segments.size());

            embeddingStore.add(id, embeddings.get(i), 
                TextSegment.from(segments.get(i).text(), metadata));
            ids.add(id);
        }

        return new ProcessingResult(ids.size(), source);
    }
}
```

---

## 四、性能优化与生产部署

### 4.1 性能优化策略全景

```
┌─────────────────────────────────────────────────────────┐
│                 性能优化策略全景图                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │   LLM 调用优化   │  │   缓存策略       │               │
│  │  · 流式输出      │  │  · 语义缓存      │               │
│  │  · Token 控制    │  │  · 结果缓存      │               │
│  │  · 并发请求      │  │  · 嵌入缓存      │               │
│  │  · 模型路由      │  │                  │               │
│  └────────┬────────┘  └────────┬────────┘               │
│           │                     │                        │
│           └──────────┬──────────┘                        │
│                      │                                   │
│  ┌───────────────────┼───────────────────┐               │
│  │                   │                   │               │
│  ▼                   ▼                   ▼               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐│
│  │ 向量检索优化     │ │ 并发与异步       │ │ 监控与限流   ││
│  │  · HNSW 索引    │ │  · WebFlux       │ │  · Metrics  ││
│  │  · 预过滤       │ │  · 虚拟线程      │ │  · Rate     ││
│  │  · 批量检索     │ │  · CompletableFuture │ │    Limit   ││
│  │  · 索引压缩     │ │                  │ │  · Circuit  ││
│  └─────────────────┘ └─────────────────┘ │    Breaker  ││
│                                          └─────────────┘│
└─────────────────────────────────────────────────────────┘
```

### 4.2 语义缓存实现

```java
/**
 * 语义缓存 — 相似问题命中缓存，避免重复调用 LLM
 */
@Component
public class SemanticCache {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<CacheEntry> cacheStore;
    private final ChatLanguageModel model;
    private final double similarityThreshold = 0.92;
    private final Duration cacheTtl = Duration.ofHours(24);

    public SemanticCache(EmbeddingModel embeddingModel, 
                         EmbeddingStore<CacheEntry> cacheStore,
                         ChatLanguageModel model) {
        this.embeddingModel = embeddingModel;
        this.cacheStore = cacheStore;
        this.model = model;
    }

    /**
     * 带缓存的对话方法
     */
    public String chatWithCache(String sessionId, String userMessage) {
        // 1. 尝试从缓存获取
        String cachedResponse = getFromCache(userMessage);
        if (cachedResponse != null) {
            log.debug("缓存命中: {}", userMessage.substring(0, Math.min(50, userMessage.length())));
            return cachedResponse;
        }

        // 2. 缓存未命中，调用 LLM
        String response = model.chat(userMessage);

        // 3. 存入缓存
        putToCache(userMessage, response);

        return response;
    }

    private String getFromCache(String query) {
        Embedding queryEmbedding = embeddingModel.embed(query).content();

        List<EmbeddingMatch<CacheEntry>> matches = cacheStore.findRelevant(
            queryEmbedding, 1, similarityThreshold
        );

        if (matches.isEmpty()) {
            return null;
        }

        CacheEntry entry = matches.get(0).embeddingMatch().embedded();

        // 检查是否过期
        if (Instant.now().isAfter(entry.createdAt().plus(cacheTtl))) {
            cacheStore.remove(matches.get(0).embeddingMatch().id());
            return null;
        }

        return entry.response();
    }

    private void putToCache(String query, String response) {
        CacheEntry entry = new CacheEntry(query, response, Instant.now());
        TextSegment segment = TextSegment.from(query);
        Embedding embedding = embeddingModel.embed(segment).content();

        cacheStore.add(UUID.randomUUID().toString(), embedding, segment);
    }
}

/**
 * LLM 调用优化 — 合理控制 Token 使用
 */
@Component
public class TokenOptimizer {

    private final ChatLanguageModel model;

    /**
     * 智能截断 — 保留重要信息的同时控制 Token 数
     */
    public String smartTruncate(String document, int maxTokens) {
        TokenCountEstimator estimator = new OpenAiTokenCountEstimator();
        int currentTokens = estimator.estimateTokenCount(document);

        if (currentTokens <= maxTokens) {
            return document;
        }

        // 使用 LLM 智能提取关键信息
        String truncated = model.chat(
            PromptTemplate.from("""
                以下文档过长，请提取关键信息，确保保留：
                1. 核心观点和结论
                2. 关键数据和数字
                3. 重要的专有名词
                文档内容：
                {{document}}
                """)
            .apply(Map.of("document", document))
            .contents()
        );

        return truncated;
    }

    /**
     * 模型路由 — 根据问题复杂度选择模型
     */
    public String routeModel(String query) {
        // 简单问题用小模型
        if (isSimpleQuery(query)) {
            return simpleModel.chat(query);
        }
        // 复杂问题用大模型
        return complexModel.chat(query);
    }

    private boolean isSimpleQuery(String query) {
        // 基于规则的简单判断
        String[] simplePatterns = {"你好", "谢谢", "什么是", "几点了"};
        for (String pattern : simplePatterns) {
            if (query.contains(pattern) && query.length() < 50) {
                return true;
            }
        }
        return false;
    }
}
```

### 4.3 并发与异步处理

```java
/**
 * 基于虚拟线程的并发 Agent 执行器
 * 利用 Java 21+ 虚拟线程实现高并发
 */
@Component
public class ConcurrentAgentExecutor {

    private final ExecutorService virtualExecutor = Executors.newVirtualThreadPerTaskExecutor();
    private final ChatLanguageModel model;

    /**
     * 并行执行多个 Agent 任务
     */
    public List<AgentResult> executeConcurrently(List<AgentTask> tasks) {
        List<CompletableFuture<AgentResult>> futures = tasks.stream()
            .map(task -> CompletableFuture.supplyAsync(() -> {
                long start = System.currentTimeMillis();
                try {
                    String result = executeTask(task);
                    return new AgentResult(
                        task.id(), result, true,
                        System.currentTimeMillis() - start
                    );
                } catch (Exception e) {
                    return new AgentResult(
                        task.id(), e.getMessage(), false,
                        System.currentTimeMillis() - start
                    );
                }
            }, virtualExecutor))
            .toList();

        return futures.stream()
            .map(CompletableFuture::join)
            .collect(Collectors.toList());
    }

    /**
     * 流式输出 — 使用 Project Reactor 实现非阻塞
     */
    public Flux<String> streamResponse(String sessionId, String message) {
        StreamingChatLanguageModel streamingModel = ...;

        Sinks.Many<String> sink = Sinks.many().unicast().onBackpressureBuffer();

        streamingModel.chat(message, new StreamingChatResponseHandler() {
            @Override
            public void onPartialResponse(String partialResponse) {
                sink.tryEmitNext(partialResponse);
            }

            @Override
            public void onCompleteResponse(ChatResponse response) {
                sink.tryEmitComplete();
            }

            @Override
            public void onError(Throwable error) {
                sink.tryEmitError(error);
            }
        });

        return sink.asFlux();
    }

    /**
     * 超时控制 — 防止 LLM 调用阻塞
     */
    public String chatWithTimeout(String message, Duration timeout) {
        return CompletableFuture
            .supplyAsync(() -> model.chat(message), virtualExecutor)
            .orTimeout(timeout.toMillis(), TimeUnit.MILLISECONDS)
            .exceptionally(ex -> {
                if (ex instanceof TimeoutException) {
                    return "请求超时，请稍后重试";
                }
                return "处理失败：" + ex.getMessage();
            })
            .join();
    }
}
```

### 4.4 监控与可观测性

```java
/**
 * LLM 调用监控拦截器
 */
@Component
public class LlmMetricsInterceptor {

    private final MeterRegistry meterRegistry;
    private final Counter requestCounter;
    private final Timer latencyTimer;
    private final AtomicLong tokenUsageGauge;

    public LlmMetricsInterceptor(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;

        this.requestCounter = Counter.builder("llm.requests.total")
            .description("Total LLM requests")
            .register(meterRegistry);

        this.latencyTimer = Timer.builder("llm.request.duration")
            .description("LLM request latency")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(meterRegistry);

        this.tokenUsageGauge = meterRegistry.gauge("llm.token.usage", 
            new AtomicLong(0));
    }

    public void recordRequest(String model, long latencyMs, int tokens) {
        requestCounter.increment(Tags.of("model", model));
        latencyTimer.record(latencyMs, TimeUnit.MILLISECONDS);
        tokenUsageGauge.addAndGet(tokens);
    }

    /**
     * 限流器 — 基于令牌桶算法
     */
    @Bean
    public RateLimiter rateLimiter() {
        return RateLimiter.builder()
            .limit(100)                          // 每秒 100 个请求
            .timeout(Duration.ofSeconds(5))       // 等待超时 5 秒
            .build();
    }

    /**
     * 熔断器 — 防止级联故障
     */
    @Bean
    public CircuitBreaker circuitBreaker() {
        return CircuitBreaker.builder()
            .failureRateThreshold(50)             // 失败率 50% 触发熔断
            .waitDurationInOpenState(Duration.ofSeconds(30))
            .slidingWindowSize(10)
            .minimumNumberOfCalls(5)
            .build();
    }
}

/**
 * 生产环境健康检查
 */
@Component
public class AiHealthIndicator implements HealthIndicator {

    private final ChatLanguageModel model;
    private final EmbeddingStore<?> embeddingStore;

    @Override
    public Health health() {
        try {
            // 检查 LLM 连通性
            String testResponse = model.chat("Say 'OK'");

            // 检查向量数据库
            embeddingStore.findRelevant(
                embeddingModel.embed("test").content(), 1
            );

            return Health.up()
                .withDetail("llm", "connected")
                .withDetail("vectorStore", "connected")
                .build();
        } catch (Exception e) {
            return Health.down()
                .withDetail("error", e.getMessage())
                .build();
        }
    }
}
```

### 4.5 Docker 部署配置

```dockerfile
# Dockerfile
FROM eclipse-temurin:21-jre-alpine AS runtime

WORKDIR /app

# 复制预构建的 JAR
COPY target/langchain4j-app.jar app.jar

# JVM 优化参数
ENV JAVA_OPTS="-XX:+UseZGC \
  -XX:MaxRAMPercentage=75 \
  -XX:+UseStringDeduplication \
  -XX:+ParallelRefProcEnabled \
  -Djava.security.egd=file:/dev/./urandom"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/actuator/health || exit 1

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=prod
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=jdbc:postgresql://postgres:5432/ai_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
      - milvus
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg16
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: ai_db
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

  milvus:
    image: milvusdb/milvus:v2.4
    volumes:
      - milvusdata:/var/lib/milvus
    environment:
      - ETCD_ENDPOINTS=etcd:2379

volumes:
  pgdata:
  redisdata:
  milvusdata:
```

### 4.6 部署架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    生产部署架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Load Balancer (Nginx)               │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ App Instance│ │ App Instance│ │ App Instance│          │
│  │    (Pod 1)  │ │    (Pod 2)  │ │    (Pod 3)  │          │
│  │  Spring Boot│ │  Spring Boot│ │  Spring Boot│          │
│  │ + LangChain4j│+ LangChain4j│+ LangChain4j │          │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │
│         │               │               │                    │
│         └───────────────┼───────────────┘                    │
│                         │                                    │
│  ┌──────────────────────┼───────────────────────────────┐   │
│  │                 Shared Infrastructure                  │   │
│  │  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────┐ │   │
│  │  │ Redis   │  │PostgreSQL│  │  Milvus   │  │MinIO │ │   │
│  │  │(Session │  │(业务数据 +│  │(向量存储) │  │(文档 │ │   │
│  │  │ + Cache)│  │ PGVector)│  │           │  │ 存储)│ │   │
│  │  └─────────┘  └──────────┘  └───────────┘  └──────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LLM Provider (外部服务)                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ OpenAI   │  │ Azure    │  │  自建模型 (vLLM)  │   │   │
│  │  │  API     │  │ OpenAI   │  │  (私有部署)       │   │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Observability Stack                      │   │
│  │  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │   │
│  │  │Prometheus│  │ Grafana  │  │  Jaeger (Tracing) │   │   │
│  │  │(Metrics)│  │(Dashboard│  │                   │   │   │
│  │  └─────────┘  └──────────┘  └───────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、实战案例：构建一个完整的客服 Agent

综合运用以上所有技术，下面我们构建一个完整的智能客服系统：

```java
/**
 * 智能客服系统 — 完整实现
 */
@Component
@Slf4j
public class CustomerServiceAgent {

    private final AssistantService assistant;
    private final DocumentProcessor docProcessor;
    private final CustomRetriever retriever;
    private final SemanticCache cache;
    private final ConcurrentAgentExecutor executor;

    /**
     * 处理用户咨询
     */
    public CustomerServiceResponse handleInquiry(CustomerInquiry inquiry) {
        long startTime = System.currentTimeMillis();

        try {
            // 1. 检查缓存
            String cachedResponse = cache.get(inquiry.getMessage());
            if (cachedResponse != null) {
                return CustomerServiceResponse.builder()
                    .answer(cachedResponse)
                    .fromCache(true)
                    .latencyMs(System.currentTimeMillis() - startTime)
                    .build();
            }

            // 2. 并行检索相关信息
            List<CompletableFuture<List<Content>>> retrievalFutures = List.of(
                CompletableFuture.supplyAsync(() -> 
                    retriever.retrieve(inquiry.getMessage())
                ),
                CompletableFuture.supplyAsync(() -> 
                    searchKnowledgeBase(inquiry.getCategory())
                )
            );

            List<Content> retrievedContent = retrievalFutures.stream()
                .flatMap(f -> f.join().stream())
                .collect(Collectors.toList());

            // 3. 构建增强提示
            String enhancedPrompt = buildRagPrompt(inquiry, retrievedContent);

            // 4. 调用 AI 服务
            String response = assistant.chat(
                inquiry.getSessionId(), 
                enhancedPrompt
            );

            // 5. 缓存结果
            cache.put(inquiry.getMessage(), response);

            // 6. 记录指标
            long latency = System.currentTimeMillis() - startTime;
            log.info("客服响应完成 - 耗时: {}ms, 缓存: false", latency);

            return CustomerServiceResponse.builder()
                .answer(response)
                .fromCache(false)
                .latencyMs(latency)
                .sources(extractSources(retrievedContent))
                .build();

        } catch (Exception e) {
            log.error("客服处理失败", e);
            return CustomerServiceResponse.builder()
                .answer("抱歉，系统暂时出现问题，请稍后重试或联系人工客服。")
                .error(true)
                .latencyMs(System.currentTimeMillis() - startTime)
                .build();
        }
    }

    /**
     * 批量处理 — 并发优化
     */
    public List<CustomerServiceResponse> handleBatchInquiries(
            List<CustomerInquiry> inquiries
    ) {
        return executor.executeConcurrently(
            inquiries.stream()
                .map(inquiry -> new AgentTask(
                    inquiry.getId(),
                    () -> handleInquiry(inquiry)
                ))
                .toList()
        ).stream()
        .map(result -> (CustomerServiceResponse) result.result())
        .toList();
    }
}
```

---

## 六、总结与展望

LangChain4j 为 Java 开发者提供了构建企业级 AI 应用的完整工具链。本文从以下四个维度进行了深入探讨：

| 维度 | 核心要点 | 关键技术 |
|------|---------|---------|
| **核心组件** | ChatModel / Memory / AiService / Agent | 类型安全、声明式编程 |
| **Spring Boot 集成** | Auto-Configuration / Bean 管理 | Starter、YAML 配置 |
| **自定义开发** | Tool / Retriever / 文档处理 | 安全校验、混合检索 |
| **生产部署** | 缓存 / 并发 / 监控 / 容器化 | 语义缓存、虚拟线程、Prometheus |

**未来展望：**

1. **多模态 Agent** — 结合视觉、语音、代码执行的全能 Agent
2. **Agent 编排** — 多 Agent 协作的复杂工作流（如 CrewAI、AutoGen 思路）
3. **本地模型集成** — Ollama + GGUF 本地部署，数据不出企业
4. **可观测性增强** — 完整的 LLM 调用链路追踪和成本分析

LangChain4j 生态正在快速成熟，随着 1.0 正式版的到来，它将成为 Java AI 应用开发的首选框架。

---

*本文首发于 2026-05-30，由 RiceBall-15 撰写。如需转载请联系作者。*
