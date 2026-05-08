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

// 定义不同的集合
const langchain4j = defineCollection({
  type: 'content',
  schema: postSchema,
});

const sglang = defineCollection({
  type: 'content',
  schema: postSchema,
});

const llmTraining = defineCollection({
  type: 'content',
  schema: postSchema,
});

const agentMemory = defineCollection({
  type: 'content',
  schema: postSchema,
});

const evaluation = defineCollection({
  type: 'content',
  schema: postSchema,
});

const aiCoding = defineCollection({
  type: 'content',
  schema: postSchema,
});

const modelDeploymentTraining = defineCollection({
  type: 'content',
  schema: postSchema,
});

const llmAgent = defineCollection({
  type: 'content',
  schema: postSchema,
});

const agentSystem = defineCollection({
  type: 'content',
  schema: postSchema,
});

const agentOps = defineCollection({
  type: 'content',
  schema: postSchema,
});

const learningMethodology = defineCollection({
  type: 'content',
  schema: postSchema,
});

const aiInfrastructure = defineCollection({
  type: 'content',
  schema: postSchema,
});

const aiEngineering = defineCollection({
  type: 'content',
  schema: postSchema,
});

const frontend = defineCollection({
  type: 'content',
  schema: postSchema,
});

const agentSkill = defineCollection({
  type: 'content',
  schema: postSchema,
});

const architecture = defineCollection({
  type: 'content',
  schema: postSchema,
});

export const collections = {
  langchain4j,
  sglang,
  llmTraining,
  agentMemory,
  evaluation,
  aiCoding,
  modelDeploymentTraining,
  llmAgent,
  agentSystem,
  agentOps,
  learningMethodology,
  aiInfrastructure,
  aiEngineering,
  frontend,
  agentSkill,
  architecture
};