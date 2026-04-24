---
title: "Agent记忆系统详解"
description: "深入解析Agent的记忆系统，包括各种记忆类型、实现原理、代码示例和最佳实践"
date: 2026-04-24
author: "RiceBall-15"
category: "Agent记忆系统"
tags: ["记忆系统", "ChatMemory", "RAG", "向量数据库", "长期记忆", "短期记忆"]
draft: false
---

# LangChain4j 记忆系统详解

> 文档版本: v1.0
> 创建时间: 2025-04-22
> 最后更新: 2025-04-22

---

## 📋 目录

- [核心概念](#核心概念)
- [记忆类型](#记忆类型)
- [实现原理](#实现原理)
- [代码示例](#代码示例)
- [最佳实践](#最佳实践)
- [进阶用法](#进阶用法)

---

## 核心概念

LangChain4j的记忆系统解决了LLM的上下文限制问题，提供了多种记忆策略来管理对话历史和检索相关信息。

### 为什么需要记忆？

1. **上下文窗口限制** - LLM有固定的token限制
2. **多轮对话** - 需要记住历史对话
3. **长期知识** - 需要从大量文档中检索相关信息
4. **成本优化** - 只发送最相关的信息给LLM

### 记忆系统架构

```
用户输入
    ↓
[RetrievalAugmentor] ← 检索增强（从向量数据库获取相关信息）
    ↓
[ChatMemory] ← 短期对话记忆
    ↓
合并上下文
    ↓
发送给LLM
```

---

## 记忆类型

### 1. ChatMemory - 对话记忆

#### MessageWindowChatMemory

保留最近N条消息：

```java
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.message.MessageWindowChatMemory;

ChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);
```

**原理**：
- 使用队列结构存储消息
- 当消息数量超过maxMessages时，自动删除最早的消息
- 适合短对话场景

#### TokenWindowChatMemory

基于token数量限制：

```java
import dev.langchain4j.memory.TokenWindowChatMemory;

ChatMemory memory = TokenWindowChatMemory.withMaxTokens(2000, tokenizer);
```

**原理**：
- 计算每条消息的token数
- 累计总token数，超过限制时删除最早消息
- 更精确控制上下文大小

#### UserMessageWindowChatMemory

只保留用户消息（用于压缩记忆）：

```java
import dev.langchain4j.memory.UserMessageWindowChatMemory;

ChatMemory memory = UserMessageWindowChatMemory.withMaxMessages(20);
```

### 2. RetrievalAugmentor - 检索增强

#### ContentRetriever + Retriever

```java
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.retriever.Retriever;
import dev.langchain4j.data.segment.TextSegment;

Retriever<TextSegment> retriever = EmbeddingStoreRetriever.builder()
    .embeddingStore(embeddingStore)
    .embeddingModel(embeddingModel)
    .maxResults(5)
    .minScore(0.7)
    .build();

ContentRetriever contentRetriever = Retriever.from(retriever);
```

**原理**：
1. 将用户查询转换为向量
2. 在向量数据库中搜索相似的文档片段
3. 返回最相关的N个片段作为上下文

---

## 实现原理

### ChatMemory接口

```java
public interface ChatMemory {

    /**
     * 添加一条消息到记忆中
     */
    void add(ChatMessage message);

    /**
     * 获取所有消息
     */
    List<ChatMessage> messages();

    /**
     * 清空记忆
     */
    void clear();
}
```

### MessageWindowChatMemory实现

```java
public class MessageWindowChatMemory implements ChatMemory {

    private final LinkedList<ChatMessage> messages;
    private final int maxMessages;

    public MessageWindowChatMemory(int maxMessages) {
        this.messages = new LinkedList<>();
        this.maxMessages = maxMessages;
    }

    @Override
    public void add(ChatMessage message) {
        messages.add(message);
        // 超过限制，删除最早的消息
        while (messages.size() > maxMessages) {
            messages.removeFirst();
        }
    }

    @Override
    public List<ChatMessage> messages() {
        return new ArrayList<>(messages);
    }
}
```

### TokenWindowChatMemory实现

```java
public class TokenWindowChatMemory implements ChatMemory {

    private final LinkedList<ChatMessage> messages = new LinkedList<>();
    private final int maxTokens;
    private final Tokenizer tokenizer;
    private int totalTokens = 0;

    @Override
    public void add(ChatMessage message) {
        int messageTokens = tokenizer.estimateTokenCount(message.text());
        messages.add(message);
        totalTokens += messageTokens;

        // 超过token限制，从最早的消息开始删除
        while (totalTokens > maxTokens && !messages.isEmpty()) {
            ChatMessage oldest = messages.removeFirst();
            totalTokens -= tokenizer.estimateTokenCount(oldest.text());
        }
    }
}
```

### RetrievalAugmentor实现

```java
public class DefaultRetrievalAugmentor implements RetrievalAugmentor {

    private final ContentRetriever contentRetriever;

    @Override
    public List<ChatMessage> augment(List<ChatMessage> messages) {
        // 1. 提取用户查询（最后一条用户消息）
        String userQuery = extractUserQuery(messages);

        // 2. 检索相关内容
        List<Content> relevantContents = contentRetriever.retrieve(userQuery);

        // 3. 将检索结果转换为系统消息
        List<ChatMessage> augmentedMessages = new ArrayList<>();
        augmentedMessages.add(SystemMessage.from(
            "以下是相关的参考信息：\n" +
            formatContents(relevantContents)
        ));

        // 4. 添加原始消息
        augmentedMessages.addAll(messages);

        return augmentedMessages;
    }
}
```

---

## 代码示例

### 示例1: 基础对话记忆

```java
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.message.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

public class BasicMemoryExample {

    public static void main(String[] args) {
        // 1. 创建模型
        ChatLanguageModel model = OpenAiChatModel.builder()
            .apiKey("your-api-key")
            .modelName("gpt-4")
            .build();

        // 2. 创建记忆（保留最近10条消息）
        ChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);

        // 3. 创建AI服务
        ChatAssistant assistant = AiServices.builder(ChatAssistant.class)
            .chatLanguageModel(model)
            .chatMemory(memory)
            .build();

        // 4. 多轮对话
        System.out.println(assistant.chat("我叫张三"));
        System.out.println(assistant.chat("我刚才说了什么名字？"));
        System.out.println(assistant.chat("请记住：我喜欢编程"));
        System.out.println(assistant.chat("我喜欢什么？"));
    }

    interface ChatAssistant {
        String chat(String message);
    }
}
```

### 示例2: Token窗口记忆

```java
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.memory.TokenWindowChatMemory;

public class TokenWindowMemoryExample {

    public static void main(String[] args) {
        // 1. 创建tokenizer
        OpenAiTokenizer tokenizer = new OpenAiTokenizer("gpt-4");

        // 2. 创建Token窗口记忆（最多2000 tokens）
        ChatMemory memory = TokenWindowChatMemory.withMaxTokens(2000, tokenizer);

        // 3. 使用方式与MessageWindow相同
        // ...
    }
}
```

### 示例3: 检索增强记忆（RAG）

```java
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.InMemoryEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;

public class RagMemoryExample {

    public static void main(String[] args) {
        // 1. 创建向量存储
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 2. 创建嵌入模型
        EmbeddingModel embeddingModel = OpenAiEmbeddingModel.builder()
            .apiKey("your-api-key")
            .modelName("text-embedding-3-small")
            .build();

        // 3. 添加文档到向量库
        String document1 = "LangChain4j是一个Java版本的LangChain框架...";
        String document2 = "它提供了记忆系统、工具调用等功能...";

        embeddingStore.add(Embedding.from(
            embeddingModel.embed(document1).content(),
            TextSegment.from(document1)
        ));

        embeddingStore.add(Embedding.from(
            embeddingModel.embed(document2).content(),
            embeddingModel.embed(document2).content(),
            TextSegment.from(document2)
        ));

        // 4. 创建检索器
        ContentRetriever contentRetriever = Retriever.from(
            EmbeddingStoreRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.7)
                .build()
        );

        // 5. 创建AI服务，集成检索增强
        ChatModel model = OpenAiChatModel.builder()
            .apiKey("your-api-key")
            .build();

        RagAssistant assistant = AiServices.builder(RagAssistant.class)
            .chatLanguageModel(model)
            .contentRetriever(contentRetriever)
            .build();

        // 6. 问答
        String answer = assistant.chat("LangChain4j有什么功能？");
        System.out.println(answer);
    }

    interface RagAssistant {
        String chat(String question);
    }
}
```

### 示例4: 混合记忆系统（短期+长期）

```java
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.message.MessageWindowChatMemory;

public class HybridMemoryExample {

    public static void main(String[] args) {
        // 1. 短期记忆（最近10轮对话）
        ChatMemory shortTermMemory = MessageWindowChatMemory.withMaxMessages(10);

        // 2. 长期记忆（从向量库检索）
        ContentRetriever longTermMemory = createRetriever();

        // 3. 创建AI服务
        HybridAssistant assistant = AiServices.builder(HybridAssistant.class)
            .chatLanguageModel(model)
            .chatMemory(shortTermMemory)
            .contentRetriever(longTermMemory)
            .systemMessage(
                "你是一个智能助手。如果有短期记忆，优先使用；" +
                "如果没有，可以从长期记忆中检索相关信息。"
            )
            .build();

        // 4. 使用
        String response = assistant.chat("我上周的工作计划是什么？");
    }

    interface HybridAssistant {
        String chat(String message);
    }
}
```

---

## 最佳实践

### 1. 选择合适的记忆类型

| 场景 | 推荐记忆类型 |
|------|------------|
| 短对话（<10轮） | MessageWindowChatMemory |
| 长对话、需要精确控制 | TokenWindowChatMemory |
| 需要检索企业知识库 | RetrievalAugmentor + EmbeddingStore |
| 综合场景 | 混合记忆（短期+长期） |

### 2. 优化记忆窗口大小

```java
// 根据模型上下文窗口设置
// GPT-4: 8k/32k tokens
// GPT-3.5: 4k tokens
// Claude: 200k tokens

int maxTokens = 2000; // 为系统提示和响应预留空间

ChatMemory memory = TokenWindowChatMemory.withMaxTokens(maxTokens, tokenizer);
```

### 3. 定期持久化记忆

```java
public class PersistentChatMemory implements ChatMemory {

    private ChatMemory inMemory;
    private String sessionId;
    private ChatMemoryRepository repository;

    @Override
    public void add(ChatMessage message) {
        inMemory.add(message);
        // 异步持久化到数据库
        repository.save(sessionId, inMemory.messages());
    }

    @Override
    public List<ChatMessage> messages() {
        // 先从缓存加载
        List<ChatMessage> cached = inMemory.messages();
        if (cached.isEmpty()) {
            // 从数据库加载
            inMemory = repository.load(sessionId);
        }
        return inMemory.messages();
    }
}
```

### 4. 记忆压缩策略

当对话历史很长时，使用LLM压缩：

```java
public class CompressingMemory implements ChatMemory {

    private ChatMemory delegate;
    private ChatLanguageModel model;
    private int maxMessagesBeforeCompress = 50;

    @Override
    public void add(ChatMessage message) {
        delegate.add(message);

        // 当消息超过阈值时压缩
        if (delegate.messages().size() > maxMessagesBeforeCompress) {
            compress();
        }
    }

    private void compress() {
        List<ChatMessage> messages = delegate.messages();

        String summary = model.generate(
            "请总结以下对话的关键信息，保留重要细节：\n" +
            messages + "\n\n总结："
        );

        // 保留最近的10条消息 + 总结
        List<ChatMessage> recent = messages.subList(
            Math.max(0, messages.size() - 10),
            messages.size()
        );

        delegate.clear();
        delegate.add(SystemMessage.from(
            "以下是对话历史的总结：\n" + summary
        ));
        recent.forEach(delegate::add);
    }
}
```

---

## 进阶用法

### 自定义记忆策略

```java
public class SmartMemory implements ChatMemory {

    private final List<ChatMessage> messages = new ArrayList<>();
    private final int maxMessages;

    @Override
    public void add(ChatMessage message) {
        messages.add(message);

        // 智能删除：保留重要消息
        while (messages.size() > maxMessages) {
            int leastImportantIndex = findLeastImportantMessage();
            messages.remove(leastImportantIndex);
        }
    }

    private int findLeastImportantMessage() {
        // 基于消息长度、关键词等计算重要性
        // 返回最不重要消息的索引
        return 0;
    }
}
```

### 跨Session记忆共享

```java
public class SharedMemory implements ChatMemory {

    private static final Map<String, List<ChatMessage>> globalMemory =
        new ConcurrentHashMap<>();

    private String userId;

    @Override
    public void add(ChatMessage message) {
        globalMemory.computeIfAbsent(userId, k -> new ArrayList<>())
                   .add(message);
    }

    @Override
    public List<ChatMessage> messages() {
        return globalMemory.getOrDefault(userId, new ArrayList<>());
    }
}
```

### 时间衰减记忆

```java
public class TimeDecayMemory implements ChatMemory {

    private final Map<ChatMessage, Long> messageTimestamps = new HashMap<>();
    private final long retentionMillis;

    @Override
    public void add(ChatMessage message) {
        messageTimestamps.put(message, System.currentTimeMillis());
    }

    @Override
    public List<ChatMessage> messages() {
        long now = System.currentTimeMillis();
        messageTimestamps.entrySet().removeIf(entry ->
            now - entry.getValue() > retentionMillis
        );
        return new ArrayList<>(messageTimestamps.keySet());
    }
}
```

---

## 总结

LangChain4j的记忆系统提供了灵活的记忆管理方案：

1. **ChatMemory** - 管理短期对话历史
2. **RetrievalAugmentor** - 长期知识检索（RAG）
3. **混合策略** - 结合短期和长期记忆
4. **自定义扩展** - 可以实现任何记忆策略

选择合适的记忆策略，根据应用场景优化窗口大小和检索参数，可以显著提升Agent的性能和用户体验。

---

## 相关资源

- 官方文档: https://docs.langchain4j.dev/
- GitHub仓库: https://github.com/langchain4j/langchain4j
- 记忆模块: https://docs.langchain4j.dev/modules/memory/
