---
title: 告别 AI 乱改代码：OpenSpec 规范驱动开发完全指南
date: 2026-04-27
category: AI编程
source: 掘金
original_url: https://juejin.cn/post/7633248335969501211
tags: ['AI编程', 'OpenSpec', '规范驱动开发', 'AI工具']
description: OpenSpec 通过规范驱动开发模式，让 AI 编程变得可控可追溯。它将人的判断前置到规范制定阶段，通过 "先定规范，再写代码" 的流程，让 AI 在清晰边界内工作，消除 AI 乱改代码的风险。
---

# 告别 AI 乱改代码：OpenSpec 规范驱动开发完全指南

## 简介

OpenSpec 是由 Fission-AI 团队开发的开源规范驱动开发（Spec-Driven Development, SDD）框架，专为 AI 编程助手（如 Cursor、Claude、GitHub Copilot）设计。其核心目标是解决 AI 编码时需求模糊、理解偏差、代码不可控的问题，通过 "先定规范，再写代码" 的流程，让人与 AI 在开发前达成共识。

## 核心定位与价值

### 核心理念：Spec First, Code Later（规范优先，代码后置）

- **Spec (规范)**：人与 AI 的 "契约"，明确定义需求、设计、任务清单
- **Code (代码)**：AI 严格按规范生成，杜绝 "自由发挥" 和 "幻觉"

### 主要价值

1. **AI 开发可控**：AI 必须遵循文档规范，输出可预测、高质量代码
2. **全链路可追溯**：从需求提案到代码归档，每一步变更都有文档记录
3. **团队协作高效**：统一规范消除沟通歧义，新成员可快速通过文档理解项目
4. **兼容主流工具**：原生支持 20+ AI 编辑器
5. **零侵入、轻量级**：一键初始化，不破坏现有项目结构

## 安装与初始化

### 环境要求

Node.js ≥ 20.19.0

### 安装命令

```bash
npm install -g @fission-ai/openspec@latest
```

### 初始化

```bash
openspec init
```

初始化后创建的目录结构包括 openspec 文件夹（含 config.yaml、specs 文件夹、changes 文件夹），以及为 Claude Code 和 Cursor 各生成 8 个文件（4 commands + 4 skills）。

## 核心功能与工作流程

### 四大核心功能

- **propose**：创建变更提案
- **apply**：实施任务
- **explore**：探索思考
- **archive**：归档变更

### 快速执行流程

```
/opsx:new ──► /opsx:ff ──► /opsx:apply ──► /opsx:verify ──► /opsx:archive
```

### 探索模式（需求不明确时）

```
/opsx:explore ──► /opsx:new ──► /opsx:continue ──► ... ──► /opsx:apply
```

### 多变更支持

可以同时处理多个变更，支持上下文切换。

### 完成变更推荐流程

```
/opsx:apply ──► /opsx:verify ──► /opsx:archive
```

## 项目结构

```
openspec/
├── specs/              # 系统行为规范（真实来源）
│   └── <domain>/
│       └── spec.md
├── changes/            # 提议的变更
│   ├── <change-name>/  # 进行中的变更
│   │   ├── proposal.md
│   │   ├── design.md
│   │   ├── tasks.md
│   │   └── specs/      # 增量规范（变更内容）
│   └── archive/        # 已归档的变更
└── config.yaml         # 项目配置
```

## 使用心得与最佳实践

### AI 时代的开发模式转变

- 从 "怎么写" 到 "写什么"：精力更多放在需求的准确表达、方案的合理性判断
- 从 "个人经验" 到 "可传递的规范"：决策通过规范文档沉淀下来
- 从 "结果导向" 到 "过程可控"：关心每一步变更是否有据可查、可回溯

### OpenSpec 的价值

- **规范沉淀**：specs/ 目录是项目的 "活文档"，始终反映系统当前真实行为
- **需求记录**：每个变更都完整保留了 "为什么做、做什么、怎么做"
- **快速上手**：新成员或 AI 可通过规范文档快速建立对系统的理解

### 解决了 AI 编程的核心顾虑

不敢让 AI 批量改代码，改完之后不知道怎么自测，出了问题也不知道从哪里排查。OpenSpec 通过规范驱动，让 AI 执行的每一步都来自事先确认过的 tasks.md，变更范围在 proposal.md 和 specs/ 中明确，归档前做三维度验证。

### 工作流命令选择建议

- **/opsx:ff**：需求明确、准备开始构建时使用
- **/opsx:continue**：探索中、想逐步审查每一步时使用
- **/opsx:explore**：实现方式、技术栈选型不确定时使用
- **/opsx:verify**：归档前必须使用，从完备性、正确性、连贯性三个维度验证

## 自定义配置

通过 `openspec/config.yaml` 可以自定义工作流行为，包括：

- **schema**：定义工作流模式（必填）
- **context**：给 AI 的项目背景（可选）
- **rules**：流程生成规则（可选）

## 总结

OpenSpec 的核心不是工具本身，而是它背后的工作方式：把人的判断前置，让 AI 在清晰的边界内执行。规范文档不是额外的负担，而是需求、设计、实现之间的共识载体——它让 AI 可控、让变更可追溯、让团队协作更顺畅。

---

**来源**：掘金
**原文链接**：https://juejin.cn/post/7633248335969501211
**发布日期**：2026-04-27