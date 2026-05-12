import { defineCollection, z } from 'astro:content';

// 通用文章schema
const postSchema = z.object({
  title: z.string(),
  description: z.string().optional(),
  date: z.coerce.date(),
  author: z.string().default('RiceBall-15'),
  category: z.string().optional(),
  tags: z.array(z.string()).default([]),
  draft: z.boolean().default(false),
});

// ============ 6个核心分类 ============

// 🤖 AI智能体 (合并: agentMemory, agentSkill, agentOps, llmAgent, aiAgent, agentSystem)
const agent = defineCollection({
  type: 'content',
  schema: postSchema,
});

// 🧠 模型训练 (合并: llmTraining, modelDeploymentTraining, modelDeployment)
const modelTraining = defineCollection({
  type: 'content',
  schema: postSchema,
});

// ⚡ 推理与评估 (合并: sglang, evaluation)
const inference = defineCollection({
  type: 'content',
  schema: postSchema,
});

// 📚 框架应用 (原: langchain4j)
const framework = defineCollection({
  type: 'content',
  schema: postSchema,
});

// 🛠 AI工程化 (合并: aiCoding, ai-engineering, aiInfrastructure, frontend, learning-methodology)
const engineering = defineCollection({
  type: 'content',
  schema: postSchema,
});

// 🏗 系统架构 (原: architecture)
const architecture = defineCollection({
  type: 'content',
  schema: postSchema,
});

export const collections = {
  agent,
  modelTraining,
  inference,
  framework,
  engineering,
  architecture,
};

// ============ 分类元数据（供首页和导航使用）============
export const categoryMeta: Record<string, { name: string; icon: string; description: string }> = {
  agent: {
    name: 'AI智能体',
    icon: '🤖',
    description: 'Agent架构、记忆系统、技能开发与运维实践',
  },
  modelTraining: {
    name: '模型训练',
    icon: '🧠',
    description: 'LLM训练技术、微调方法、分布式训练与模型部署',
  },
  inference: {
    name: '推理与评估',
    icon: '⚡',
    description: '高性能推理框架、模型评估与基准测试',
  },
  framework: {
    name: '框架应用',
    icon: '📚',
    description: '主流AI框架的深度解析与实战指南',
  },
  engineering: {
    name: 'AI工程化',
    icon: '🛠',
    description: 'AI编程工具、基础设施、工程实践与学习方法',
  },
  architecture: {
    name: '系统架构',
    icon: '🏗',
    description: 'AI系统架构设计、分布式系统与工程实践',
  },
};

// ============ 旧分类到新分类的映射（兼容定时任务）============
export const legacyCollectionMap: Record<string, string> = {
  agentMemory: 'agent',
  agentSkill: 'agent',
  agentOps: 'agent',
  llmAgent: 'agent',
  aiAgent: 'agent',
  agentSystem: 'agent',
  llmTraining: 'modelTraining',
  modelDeploymentTraining: 'modelTraining',
  modelDeployment: 'modelTraining',
  sglang: 'inference',
  evaluation: 'inference',
  langchain4j: 'framework',
  aiCoding: 'engineering',
  'ai-engineering': 'engineering',
  aiInfrastructure: 'engineering',
  frontend: 'engineering',
  'learning-methodology': 'engineering',
  architecture: 'architecture',
};
