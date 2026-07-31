import { defineCollection, z } from 'astro:content';

// 通用文章schema
const postSchema = z.object({
  title: z.string(),
  description: z.string().optional(),
  date: z.coerce.date(),
  author: z.string().default('RiceBall-15'),
  category: z.string().optional(),
  subCategory: z.string().optional(), // 子分类
  tags: z.array(z.string()).default([]),
  draft: z.boolean().default(false),
  // ===== 系列文章支持 =====
  series: z.string().optional(),      // 所属系列 id（见 seriesMeta）
  seriesOrder: z.number().optional(), // 系列内序号，用于排序与上下篇导航
});

// ============ 7个核心分类 ============

// 🤖 AI智能体 (合并: agentMemory, agentSkill, agentOps, llmAgent, aiAgent, agentSystem)
const agent = defineCollection({
  type: 'content',
  schema: postSchema,
});

// 🔧 AI基础设施 (合并: modelTraining, inference, llmTraining, sglang, evaluation, modelDeployment等)
const aiInfra = defineCollection({
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

// 📖 优秀文章精选 (新增: 参考优秀博主风格的文章)
const featured = defineCollection({
  type: 'content',
  schema: postSchema,
});

// 🔧 AI工具评测 (新增: AI工具深度评测与对比)
const aiTools = defineCollection({
  type: 'content',
  schema: postSchema,
});

// 💼 面试精选 (新增: 独立面试分类)
const interview = defineCollection({
  type: 'content',
  schema: postSchema,
});

export const collections = {
  agent,
  aiInfra,
  framework,
  engineering,
  architecture,
  featured,
  'ai-tools': aiTools,
  interview,
};

// ============ 子分类定义 ============
export const subCategories: Record<string, { id: string; name: string; icon: string }[]> = {
  agent: [
    { id: 'agent-architecture', name: '架构设计', icon: 'AC' },
    { id: 'agent-memory', name: '记忆系统', icon: 'MM' },
    { id: 'agent-skill', name: '技能开发', icon: 'SK' },
    { id: 'agent-ops', name: '运维实践', icon: 'OP' },
  ],
  aiInfra: [
    { id: 'model-training', name: '模型训练', icon: 'TR' },
    { id: 'inference', name: '推理优化', icon: 'IF' },
    { id: 'evaluation', name: '量化评估', icon: 'EV' },
  ],
  framework: [
    { id: 'rag', name: 'RAG系统', icon: 'RG' },
    { id: 'agent-framework', name: 'Agent框架', icon: 'AF' },
    { id: 'protocols', name: '工具协议', icon: 'PR' },
  ],
  engineering: [
    { id: 'ai-coding', name: 'AI编程工具', icon: 'CD' },
    { id: 'infra', name: '基础设施', icon: 'IS' },
    { id: 'learning', name: '学习方法', icon: 'LN' },
  ],
  architecture: [
    { id: 'distributed', name: '分布式系统', icon: 'DS' },
    { id: 'microservices', name: '微服务架构', icon: 'MS' },
    { id: 'cloud-native', name: '云原生', icon: 'CN' },
  ],
  featured: [
    { id: 'ai-architecture', name: 'AI架构', icon: 'AA' },
    { id: 'deep-dive', name: '深度解析', icon: 'DD' },
  ],
  // 面试分类按「Agent 工程师真实面试考察面」划分，而非通用的系统设计/算法/行为三分法
  interview: [
    { id: 'orchestration', name: '编排与调度', icon: 'OR' },
    { id: 'memory-retrieval', name: '记忆与检索', icon: 'MR' },
    { id: 'tools-protocol', name: '工具与协议', icon: 'TP' },
    { id: 'gateway-cost', name: '网关与成本', icon: 'GC' },
    { id: 'eval-observability', name: '评测与可观测', icon: 'EO' },
    { id: 'production', name: '生产与运维', icon: 'PD' },
    { id: 'security', name: '安全与对齐', icon: 'SC' },
    { id: 'context-prompt', name: '上下文与提示词', icon: 'CP' },
    { id: 'guide', name: '面试方法论', icon: 'GD' },
  ],
  'ai-tools': [
    { id: 'coding-tools', name: '编程工具', icon: 'CT' },
    { id: 'browser-tools', name: '浏览器工具', icon: 'BT' },
    { id: 'protocol-tools', name: '协议工具', icon: 'PT' },
  ],
};

// ============ 标签 URL slug ============
// 标签是自由文本，里面可能有 '/'（如 CI/CD）或空格。直接拿它当路由参数，
// 斜杠会被当成路径分隔符，Astro 报 "Missing parameter: tag" 并中断整个构建。
// 所以统一走这个函数生成 URL 片段；凡是链接到标签页的地方都必须用它。
export const tagSlug = (t: string) =>
  t.trim().replace(/[\/\\?#%\s]+/g, '-');

// ============ 系列元数据（供系列索引页使用）============
// 系列 = 有阅读顺序的成套文章，与「分类」正交：分类回答「这属于哪个领域」，
// 系列回答「按什么顺序读完能形成完整认知」。
export const seriesMeta: Record<
  string,
  { name: string; icon: string; description: string; collection: string }
> = {
  'agent-interview-deep': {
    name: 'Agent 工程师面试深度系列',
    icon: 'SR',
    description:
      '面向后端/平台方向的 Agent 工程师面试，从 DAG 编排、工具调用、记忆分层到召回准确率与评测闭环，每篇一个考察面，含机制推导、伪码与生产踩坑。',
    collection: 'interview',
  },
  'agent-skill-dev': {
    name: 'Agent Skill 开发全流程',
    icon: 'SK',
    description:
      '从概念与架构入手，依次走过编写规范、生命周期、依赖管理、测试与调试、性能与安全，最后到多 Agent 编排与企业级平台化。按这个顺序读能形成完整的 Skill 工程认知。',
    collection: 'agent',
  },
  'claude-mem-deep': {
    name: 'Claude-Mem 记忆系统深度解析',
    icon: 'MM',
    description:
      '逐层拆解一个开源 Agent 记忆系统：项目定位、Hook 生命周期、SQLite + 向量双存储、渐进式披露的搜索架构、异步队列容错，最后回到落地实践。',
    collection: 'agent',
  },
  'claude-code-arch': {
    name: 'Claude Code 架构解析',
    icon: 'AC',
    description:
      '从设计哲学讲到 Hook 机制、Skills 与 Memory 的协作，最后做同类工具的横向对比。适合想理解 Agent 编程工具内部怎么搭的读者。',
    collection: 'agent',
  },
};

// ============ 分类元数据（供首页和导航使用）============
export const categoryMeta: Record<string, { name: string; icon: string; description: string }> = {
  agent: {
    name: 'AI智能体',
    icon: 'AG',
    description: 'Agent架构、记忆系统、技能开发与运维实践',
  },
  interview: {
    name: '面试精选',
    icon: 'IV',
    description: '系统设计、算法编程、行为面试与技术面试深度解析',
  },
  aiInfra: {
    name: 'AI基础设施',
    icon: 'IN',
    description: '模型训练、推理优化、量化评估与AI Infra工程实践',
  },
  framework: {
    name: '框架应用',
    icon: 'FW',
    description: '主流AI框架的深度解析与实战指南',
  },
  engineering: {
    name: 'AI工程化',
    icon: 'EN',
    description: 'AI编程工具、基础设施、工程实践与学习方法',
  },
  architecture: {
    name: '系统架构',
    icon: 'AR',
    description: 'AI系统架构设计、分布式系统与工程实践',
  },
  featured: {
    name: '优秀文章精选',
    icon: 'FT',
    description: '参考优秀博主风格的深度技术文章与架构深度解析',
  },
  'ai-tools': {
    name: 'AI工具评测',
    icon: 'TL',
    description: 'AI工具深度评测、对比分析与最佳实践',
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
  llmTraining: 'aiInfra',
  modelDeploymentTraining: 'aiInfra',
  modelDeployment: 'aiInfra',
  sglang: 'aiInfra',
  evaluation: 'aiInfra',
  langchain4j: 'framework',
  aiCoding: 'engineering',
  'ai-engineering': 'engineering',
  aiInfrastructure: 'engineering',
  frontend: 'engineering',
  'learning-methodology': 'engineering',
  architecture: 'architecture',
  'ai-tools': 'ai-tools',
};

// ============ 旧子分类到新子分类的映射 ============
export const legacySubCategoryMap: Record<string, string> = {
  agentMemory: 'agent-memory',
  agentSkill: 'agent-skill',
  agentOps: 'agent-ops',
  llmAgent: 'agent-architecture',
  aiAgent: 'agent-architecture',
  agentSystem: 'agent-architecture',
  llmTraining: 'model-training',
  modelDeploymentTraining: 'model-training',
  modelDeployment: 'inference',
  sglang: 'inference',
  evaluation: 'evaluation',
  langchain4j: 'rag',
  aiCoding: 'ai-coding',
  'ai-engineering': 'ai-coding',
  aiInfrastructure: 'infra',
  frontend: 'infra',
  'learning-methodology': 'learning',
  architecture: 'distributed',
};
