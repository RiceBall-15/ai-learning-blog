# AI 学习博客

基于 Astro 构建的技术博客，记录 AI 相关技术的学习笔记。

## 📖 博客介绍

这个博客记录了我学习 **AI 技术** 的过程，包括：

- **LangChain4j** - Java 版本的 LangChain 框架
- **SGLang** - 高性能的 LLM 推理框架
- **LLM训练** - 大语言模型训练技术和最佳实践

## 🛠️ 技术栈

- **Astro** - 现代静态站点生成器
- **GitHub Pages** - 托管服务
- **深色主题** - 护眼的阅读体验

## 🚀 快速开始

### 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### 构建

```bash
# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

## 📝 添加新文章

按照技术分类，在相应的目录下创建 Markdown 文件：

### LangChain4j 文章
```
src/content/langchain4j/文章标题.md
```

### SGLang 文章
```
src/content/sglang/文章标题.md
```

### LLM 训练文章
```
src/content/llm-training/文章标题.md
```

**文件头元数据：**

```yaml
---
title: "文章标题"
description: "文章描述"
date: 2025-04-23
author: "RiceBall-15"
category: "分类名称"
tags: ["标签1", "标签2"]
draft: false
---
```

## 📂 博客结构

```
src/
├── content/
│   ├── langchain4j/     # LangChain4j 学习笔记
│   ├── sglang/          # SGLang 学习笔记
│   └── llm-training/    # LLM 训练学习笔记
├── layouts/             # 布局模板
├── pages/               # 页面组件
└── styles/              # 样式文件
```

## 🌐 在线访问

博客地址：https://riceball-15.github.io/ai-learning-blog

## 📄 许可证

MIT License

## 👤 关于作者

我是 **RiceBall-15**，一名对 AI 技术充满热情的开发者。
我相信通过记录和分享，可以更好地理解和掌握前沿技术。

## 📮 联系方式

- GitHub: [@RiceBall-15](https://github.com/RiceBall-15)
- Email: Li_Yuanzhuo@163.com

---

**持续学习，持续分享！** 🚀