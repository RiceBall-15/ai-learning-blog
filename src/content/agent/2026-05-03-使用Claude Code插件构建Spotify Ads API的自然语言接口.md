---
title: 使用Claude Code插件构建Spotify Ads API的自然语言接口
date: 2026-05-01
category: agentMemory
tags: ['Claude-Code', 'Spotify-API', 'AI代理', 'OpenAPI', '自然语言接口']
source: Spotify Engineering
source_url: https://engineering.atspotify.com/2026/5/spotify-ads-api-claude-plugins/
---

## 简介

Spotify工程团队开发了一套基于Claude Code插件的AI代理系统，允许用户通过自然语言与Spotify Ads API交互。该插件完全由Markdown文件、bash脚本和两个小型Python助手构建，无需编译代码即可实现复杂API调用的自动化。这种方法显著降低了使用复杂广告API的门槛，让非技术人员也能通过简单的对话创建广告活动。

## 技术架构核心

### 四大组件设计

该插件架构包含四个核心组件：

1. **Skills（技能）**：定义可用的命令和操作。每个技能都是一个自包含的Markdown文件，描述了命令的行为、API端点、请求格式和输出处理逻辑。

2. **Agents（代理）**：负责处理自然语言请求。通过系统提示将LLM转变为特定领域的专家，例如`agents/spotify-ads-request-builder.md`将Claude转变为Spotify Ads API专家。

3. **Hooks（钩子）**：拦截和修改工具调用。例如，使用`PreToolUse`钩子透明地刷新OAuth令牌并注入HTTP头，确保API调用的安全性。

4. **Settings（设置）**：存储API密钥、OAuth令牌等配置信息，与技能和代理分离，便于管理和版本控制。

### OpenAPI规范作为知识库

系统利用OpenAPI规范作为API的单一真实来源，支持动态加载和按需查询。这种设计避免了将完整API文档注入上下文窗口的浪费，只在实际需要时检索相关部分。OpenAPI Links功能允许通过运行时表达式（如`$response.body#/id`和`$request.path.ad_account_id`）定义操作间关系，为LLM代理提供了显式的工作流导航图。

### 多步骤编排能力

代理能够将复杂的自然语言请求分解为正确的API调用序列。例如，用户只需说"创建一个名为Back to School Promo的音频广告活动，针对美国25-44岁人群，预算100美元/天"，系统会自动：
- 创建广告活动
- 创建广告集
- 创建广告
- 处理所有依赖关系和ID传递

## 实战经验与代码示例

### Skills定义示例

每个技能都是一个Markdown文件，以下是一个典型的技能结构：

```markdown
# create-campaign

## 描述
创建新的广告活动

## API端点
POST /v3/ads/accounts/{account_id}/campaigns

## 请求参数
- name: 活动名称
- objective: 目标类型（brand_awareness, traffic, engagement等）
- budget_daily_amount: 日预算
- targeting: 目标受众配置

## 输出处理
返回创建的活动ID和状态
```

### Hooks实现示例

使用PreToolUse钩子自动刷新OAuth令牌：

```python
def pre_tool_use_hook(tool_name, tool_input):
    if tool_name.startswith("spotify_ads_"):
        # 检查令牌是否即将过期
        if needs_token_refresh():
            new_token = refresh_oauth_token()
            # 注入新的Authorization头
            tool_input["headers"] = {
                "Authorization": f"Bearer {new_token}"
            }
    return tool_input
```

### Agent系统提示设计

`agents/spotify-ads-request-builder.md`的核心提示：

```markdown
你是一个Spotify Ads API专家。当用户要求创建或管理广告活动时：

1. 分析用户意图，确定需要执行的API调用序列
2. 检查必要的参数，如有缺失则询问用户
3. 按正确顺序调用API，确保依赖关系（如广告ID传递给广告集）
4. 使用OpenAPI Links指导操作流程
5. 向用户清晰报告每一步的结果和错误

始终将用户的自然语言请求转换为具体的API调用，并验证响应的正确性。
```

### OpenAPI Links使用

定义操作间关系，支持多步骤工作流：

```yaml
paths:
  /campaigns:
    post:
      operationId: createCampaign
      responses:
        '201':
          description: Campaign created
          links:
            CreateAdSet:
              operationId: createAdSet
              parameters:
                account_id: '$request.path.account_id'
                campaign_id: '$response.body#/id'

  /campaigns/{campaign_id}/adsets:
    post:
      operationId: createAdSet
      parameters:
        - name: campaign_id
          in: path
          required: true
          schema:
            type: string
```

### 实际使用示例

用户只需用自然语言描述需求：

```
Create an audio campaign called Back to School Promo targeting 25-44 year olds in the US with $100/day budget
```

系统自动执行：
1. 创建广告活动（POST /campaigns）
2. 创建广告集（POST /campaigns/{id}/adsets），继承活动ID
3. 创建广告（POST /campaigns/{id}/ads），继承广告集ID
4. 返回完整的结果摘要

## 关键优势

### 1. "Markdown即代码"的简洁性

整个插件系统主要由Markdown文件构成，无需编译、无需传统SDK、无需复杂的构建过程。这使得系统易于定制、扩展和版本控制，开发者可以快速添加新功能或修改现有行为。

### 2. 透明可调试的执行

所有API调用都显示为curl命令，用户可以完全看到系统在做什么，便于验证和控制。这种透明性对于需要精确控制广告投放的用户尤其重要。

### 3. 单一真实来源的维护性

使用OpenAPI规范作为API的权威描述，当API变更时只需更新OpenAPI文件，无需修改大量代码。这大大简化了长期维护工作。

### 4. 显式的工作流导航

通过OpenAPI Links定义操作间关系，LLM代理获得了清晰的工作流地图，减少了在复杂API调用序列中的错误率。

## 总结与启示

Spotify的这个案例展示了如何用最简单的技术栈（Markdown + 脚本）构建功能强大的AI代理系统。关键成功因素包括：

- 将复杂性隐藏在智能代理之后，提供简单的自然语言界面
- 利用OpenAPI等标准化规范减少重复工作
- 设计透明的执行流程增强用户信任
- 采用"插件化"架构便于功能扩展和定制

这种架构不仅适用于广告API，还可以推广到其他需要多步骤操作的复杂API场景，为AI代理系统设计提供了宝贵的参考经验。

---

**来源**：Spotify Engineering Blog  
**原文链接**：https://engineering.atspotify.com/2026/5/spotify-ads-api-claude-plugins/  
**发布日期**：2026-05-01
