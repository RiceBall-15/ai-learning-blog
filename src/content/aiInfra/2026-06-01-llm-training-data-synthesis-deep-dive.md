---
title: "LLM训练数据合成技术深度解析：从Self-Instruct到高质量领域数据生成"
description: "系统剖析LLM训练数据合成的技术演进、主流方法论、质量控制体系与生产实践，助你用合成数据突破数据瓶颈"
date: 2026-06-01
author: "RiceBall"
category: "aiInfra"
subCategory: model-training
tags: ["LLM", "合成数据", "Self-Instruct", "数据工程", "模型训练", "数据质量", "Evol-Instruct"]
draft: false
---

## 引言：为什么合成数据成为必选项

2024-2026年，LLM领域出现了一个显著趋势：**高质量训练数据正在成为比算力更稀缺的资源**。公开可用的高质量文本数据已被主要模型厂商消耗殆尽，而人工标注数据的成本持续攀升（单条高质量数据成本已达$5-50）。

在这样的背景下，**合成数据（Synthetic Data）** 从"学术玩具"变成了"生产必需品"。Llama 3.1的训练中，合成数据占比超过30%；Phi系列模型更是将合成数据作为核心策略，用"教科书级别"的合成数据训练出了远超同参数规模的模型。

本文将从技术原理、主流方法、质量控制和生产实践四个维度，深度解析LLM训练数据合成技术。

---

## 一、合成数据的技术演进

### 1.1 发展脉络

```
时间线：LLM合成数据技术演进
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2022.10  Self-Instruct (Wang et al.)
         │  └─ 开创性工作：用LLM生成指令数据
         │     核心思路：种子任务 → LLM生成新任务 → 过滤 → 迭代
         │
2023.03  Alpaca (Stanford)
         │  └─ 52K条GPT-3.5生成的指令数据
         │     证明了合成数据训练的有效性
         │
2023.05  WizardLM / Evol-Instruct
         │  └─ 指令进化：深度进化 + 广度进化
         │     从简单指令演化出复杂指令
         │
2023.08  Orca (Microsoft)
         │  └─ 推理增强：利用教师模型的思维链
         │     不仅学答案，还学推理过程
         │
2023.10  Magpie (Alignment Lab)
         │  └─ 零种子生成：直接从LLM的系统提示出发
         │     自动生成完整的对话数据
         │
2024.02  Phi-2 / Textbooks Are All You Need
         │  └─ 质量至上：用合成的"教科书"数据训练
         │     小模型 + 高质量合成数据 > 大模型 + 低质量数据
         │
2024.06  Cosmopedia (HuggingFace)
         │  └─ 大规模合成：250亿token的合成教科书
         │     展示了合成数据的规模化可行性
         │
2025.01  STILL-2 / Deita
         │  └─ 数据质量自动化：自动评估 + 自动选择
         │     用LLM自动评估数据质量
         │
2025-26  当前趋势
         └─ 多模态合成 + 领域定制 + 质量自动化
            合成数据成为标准训练流程的必备环节
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 1.2 核心分类体系

| 类别 | 方法 | 适用场景 | 典型产出 |
|-----|------|---------|---------|
| **指令合成** | Self-Instruct, Evol-Instruct | SFT阶段 | 指令-响应对 |
| **推理合成** | Orca, Chain-of-Thought蒸馏 | 推理能力增强 | 带思维链的响应 |
| **领域合成** | Textbooks Are All You Need | 领域知识注入 | 领域教材/文档 |
| **偏好合成** | UltraFeedback, RLHF数据 | 对齐训练 | chosen/rejected对 |
| **代码合成** | OSS-Instruct, CodeContests | 代码能力训练 | 代码-描述对 |
| **多模态合成** | ShareGPT4V, LLaVA系列 | 多模态训练 | 图文对 |

---

## 二、主流合成方法深度解析

### 2.1 Self-Instruct：开创性框架

Self-Instruct是LLM数据合成的基石。核心流程：

```
Self-Instruct 流程
━━━━━━━━━━━━━━━━━━

种子任务池 (175条人工编写)
    │
    ▼
┌─────────────────────────────────────┐
│  1. 从种子池采样若干任务作为示例      │
│  2. 输入给LLM，要求生成新任务        │
│  3. 过滤（去重、格式检查、长度检查）  │
│  4. 对新任务生成响应                  │
│  5. 加入种子池                       │
└─────────────────────────────────────┘
    │
    ▼ 重复迭代
    │
最终数据集 (52K+ 条)
```

```python
class SelfInstructPipeline:
    """Self-Instruct数据合成管道"""
    
    def __init__(self, llm_client, seed_tasks: list[dict]):
        self.llm = llm_client
        self.seed_tasks = seed_tasks
        self.task_pool = list(seed_tasks)
    
    def generate_batch(
        self,
        batch_size: int = 4,
        num_samples_per_task: int = 20,
    ) -> list[dict]:
        """生成一批新任务"""
        
        generated_tasks = []
        
        for _ in range(num_samples_per_task):
            # 1. 从任务池采样示例
            examples = random.sample(self.task_pool, min(batch_size, len(self.task_pool)))
            
            # 2. 构建prompt
            prompt = self._build_generation_prompt(examples)
            
            # 3. 生成新任务
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2048,
            )
            
            # 4. 解析生成的任务
            new_tasks = self._parse_tasks(response)
            
            # 5. 过滤
            filtered_tasks = self._filter_tasks(new_tasks)
            
            generated_tasks.extend(filtered_tasks)
        
        return generated_tasks
    
    def _build_generation_prompt(self, examples: list[dict]) -> str:
        """构建任务生成prompt"""
        
        example_text = "\n".join(
            f"Task {i+1}: {ex['instruction']}"
            for i, ex in enumerate(examples)
        )
        
        return f"""Below are some tasks. Create one new task that is different from them but in the same general category.

{example_text}

New Task:"""
    
    def _filter_tasks(self, tasks: list[dict]) -> list[dict]:
        """过滤低质量任务"""
        
        filtered = []
        for task in tasks:
            instruction = task.get("instruction", "")
            
            # 长度过滤
            if len(instruction) < 15 or len(instruction) > 1000:
                continue
            
            # 重复检测
            if self._is_duplicate(instruction):
                continue
            
            # 格式检查
            if not self._is_well_formatted(instruction):
                continue
            
            filtered.append(task)
        
        return filtered
    
    def _is_duplicate(self, instruction: str) -> bool:
        """简单的重复检测"""
        instruction_lower = instruction.lower().strip()
        for existing in self.task_pool:
            existing_lower = existing["instruction"].lower().strip()
            # 使用编辑距离或Jaccard相似度
            if self._jaccard_similarity(instruction_lower, existing_lower) > 0.7:
                return True
        return False
```

### 2.2 Evol-Instruct：指令进化

Evol-Instruct的核心创新是**指令进化**——从简单指令出发，通过深度进化和广度进化，生成复杂多样的指令。

```python
class EvolInstructPipeline:
    """Evol-Instruct指令进化管道"""
    
    # 深度进化：增加指令的复杂度
    DEEP_PROMPTS = [
        "请增加更多约束条件",
        "请要求更详细的推理过程",
        "请加入更多边界情况的处理",
        "请将任务分解为多个子任务",
        "请增加对错误处理的要求",
    ]
    
    # 广度进化：增加指令的多样性
    BROAD_PROMPTS = [
        "请用不同的表达方式重写这个指令",
        "请将这个任务应用到不同的领域",
        "请改变指令的约束条件",
        "请增加新的输入格式要求",
    ]
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def evolve(
        self,
        instruction: str,
        evolution_type: str = "deep",
        num_rounds: int = 3,
    ) -> str:
        """执行指令进化"""
        
        current_instruction = instruction
        prompts = self.DEEP_PROMPTS if evolution_type == "deep" else self.BROAD_PROMPTS
        
        for round_idx in range(num_rounds):
            # 随机选择进化策略
            evolution_prompt = random.choice(prompts)
            
            # 执行进化
            evolved = self.llm.generate(
                messages=[
                    {"role": "system", "content": "你是一个指令进化专家。请根据要求进化给定的指令，保持其核心意图但增加复杂度或多样性。"},
                    {"role": "user", "content": f"原始指令：{current_instruction}\n\n进化要求：{evolution_prompt}\n\n进化后的指令："}
                ],
                temperature=0.8,
            )
            
            # 检查进化质量
            if self._validate_evolution(current_instruction, evolved):
                current_instruction = evolved
            else:
                # 进化失败，跳过这轮
                continue
        
        return current_instruction
    
    def _validate_evolution(self, original: str, evolved: str) -> bool:
        """验证进化质量"""
        
        # 进化后的指令不能太短
        if len(evolved) < len(original) * 0.5:
            return False
        
        # 进化后的指令不能和原始指令完全相同
        if self._jaccard_similarity(original, evolved) > 0.9:
            return False
        
        # 进化后的指令应该确实更复杂（通过LLM判断）
        judgment = self.llm.generate(
            messages=[
                {"role": "user", "content": f"原始指令：{original}\n进化后指令：{evolved}\n\n进化后的指令是否确实比原始指令更复杂或更多样？回答'是'或'否'。"}
            ],
            temperature=0.1,
        )
        
        return "是" in judgment
```

### 2.3 推理数据合成

Orca风格的推理数据合成是当前最具价值的方向之一。核心思想：**不只教模型"答案是什么"，还教模型"为什么是这个答案"**。

```python
class ReasoningDataSynthesizer:
    """推理数据合成器"""
    
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
    
    def synthesize_reasoning_data(
        self,
        task: str,
        input_data: str,
        temperature: float = 0.7,
    ) -> dict:
        """生成带推理过程的训练数据"""
        
        # Step 1: 教师模型生成详细推理过程
        reasoning_response = self.teacher.generate(
            messages=[
                {"role": "system", "content": "你是一个推理专家。请详细展示你的思考过程，包括：\n1. 理解问题的关键点\n2. 可能的解题路径\n3. 选择某条路径的原因\n4. 逐步推理过程\n5. 验证答案"},
                {"role": "user", "content": f"任务：{task}\n输入：{input_data}"}
            ],
            temperature=temperature,
        )
        
        # Step 2: 提取答案和推理过程
        answer = self._extract_answer(reasoning_response)
        reasoning_chain = self._extract_reasoning_chain(reasoning_response)
        
        # Step 3: 生成多个推理路径（多样性）
        alternative_reasonings = []
        for _ in range(3):
            alt_reasoning = self.teacher.generate(
                messages=[
                    {"role": "system", "content": "请用不同的思路重新分析这个问题，展示不同的推理路径。"},
                    {"role": "user", "content": f"任务：{task}\n输入：{input_data}\n参考答案：{answer}"}
                ],
                temperature=0.9,
            )
            alternative_reasonings.append(alt_reasoning)
        
        return {
            "instruction": f"{task}\n{input_data}",
            "output": answer,
            "reasoning_chain": reasoning_chain,
            "alternative_reasonings": alternative_reasonings,
            "quality_score": self._estimate_quality(reasoning_response),
        }
    
    def _estimate_quality(self, response: str) -> float:
        """估计推理质量"""
        # 简单的质量评估
        quality_indicators = [
            "首先" in response or "第一步" in response,  # 有步骤感
            "因此" in response or "所以" in response,      # 有因果推理
            "检查" in response or "验证" in response,      # 有验证步骤
            len(response) > 200,                            # 足够详细
            response.count("\n") >= 5,                      # 有结构化
        ]
        return sum(quality_indicators) / len(quality_indicators)
```

### 2.4 领域数据合成

```python
class DomainDataSynthesizer:
    """领域数据合成器"""
    
    def __init__(self, llm_client, domain_knowledge: str):
        self.llm = llm_client
        self.domain_knowledge = domain_knowledge
    
    def generate_textbook_chapters(
        self,
        topic: str,
        num_chapters: int = 10,
        difficulty_levels: list[str] = ["beginner", "intermediate", "advanced"],
    ) -> list[dict]:
        """生成教科书级别的领域内容"""
        
        chapters = []
        
        for level in difficulty_levels:
            chapters_per_level = num_chapters // len(difficulty_levels)
            
            for i in range(chapters_per_level):
                # 生成章节大纲
                outline = self._generate_outline(topic, level, i)
                
                # 生成完整章节
                chapter_content = self._generate_chapter(
                    topic=topic,
                    outline=outline,
                    difficulty=level,
                )
                
                # 生成配套练习
                exercises = self._generate_exercises(chapter_content, level)
                
                chapters.append({
                    "topic": topic,
                    "level": level,
                    "chapter_num": i + 1,
                    "outline": outline,
                    "content": chapter_content,
                    "exercises": exercises,
                    "quality_score": self._evaluate_chapter(chapter_content),
                })
        
        return chapters
    
    def _generate_outline(self, topic: str, level: str, chapter_idx: int) -> str:
        """生成章节大纲"""
        
        difficulty_context = {
            "beginner": "适合初学者，从基础概念开始",
            "intermediate": "需要一定基础，深入原理",
            "advanced": "面向专家，讨论前沿技术",
        }
        
        return self.llm.generate(
            messages=[
                {"role": "system", "content": f"你是一个{topic}领域的资深教育专家。"},
                {"role": "user", "content": f"请为以下教科书章节生成大纲：\n\n主题：{topic}\n难度：{difficulty_context[level]}\n章节：第{chapter_idx + 1}章\n\n要求：\n1. 包含3-5个小节\n2. 每个小节有明确的学习目标\n3. 包含实例和练习\n4. 逐步递进，逻辑清晰"}
            ],
            temperature=0.6,
        )
    
    def _generate_chapter(self, topic: str, outline: str, difficulty: str) -> str:
        """生成完整章节内容"""
        
        return self.llm.generate(
            messages=[
                {"role": "system", "content": "你是一个专业的技术教科书作者。请按照大纲撰写详细的教科书章节内容。"},
                {"role": "user", "content": f"主题：{topic}\n难度：{difficulty}\n大纲：\n{outline}\n\n请按照大纲撰写完整的章节内容，要求：\n1. 概念解释清晰准确\n2. 有具体的代码示例或案例分析\n3. 包含图表说明\n4. 每个小节后有小结\n5. 语言专业但易懂"}
            ],
            temperature=0.5,
            max_tokens=4096,
        )
```

---

## 三、合成数据质量控制体系

### 3.1 质量评估维度

```
┌─────────────────────────────────────────────────────────────────┐
│                   合成数据质量评估体系                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   准确性      │  │   多样性      │  │   实用性      │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ • 事实正确性  │  │ • 话题覆盖度  │  │ • 任务相关性  │          │
│  │ • 逻辑一致性  │  │ • 表达多样性  │  │ • 可执行性    │          │
│  │ • 格式规范性  │  │ • 难度分布    │  │ • 难度适中    │          │
│  │ • 无有害内容  │  │ • 风格变化    │  │ • 有实际价值  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │   独立性      │  │   安全性      │                             │
│  ├──────────────┤  ├──────────────┤                             │
│  │ • 去重率      │  │ • 无偏见      │                             │
│  │ • 新颖性      │  │ • 无有害内容  │                             │
│  │ • 非模板化    │  │ • 合规性      │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 自动化质量评估

```python
from dataclasses import dataclass
from enum import Enum

class QualityDimension(Enum):
    ACCURACY = "accuracy"
    DIVERSITY = "diversity"
    USEFULNESS = "usefulness"
    INDEPENDENCE = "independence"
    SAFETY = "safety"

@dataclass
class QualityReport:
    overall_score: float
    dimension_scores: dict[QualityDimension, float]
    issues: list[str]
    recommendations: list[str]

class SyntheticDataEvaluator:
    """合成数据自动评估器"""
    
    def __init__(self, llm_client, reference_data: list[dict] = None):
        self.llm = llm_client
        self.reference_data = reference_data or []
    
    def evaluate(self, data: list[dict]) -> QualityReport:
        """综合评估合成数据质量"""
        
        scores = {}
        
        # 评估每个维度
        scores[QualityDimension.ACCURACY] = self._evaluate_accuracy(data)
        scores[QualityDimension.DIVERSITY] = self._evaluate_diversity(data)
        scores[QualityDimension.USEFULNESS] = self._evaluate_usefulness(data)
        scores[QualityDimension.INDEPENDENCE] = self._evaluate_independence(data)
        scores[QualityDimension.SAFETY] = self._evaluate_safety(data)
        
        # 计算总分（加权平均）
        weights = {
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.DIVERSITY: 0.20,
            QualityDimension.USEFULNESS: 0.25,
            QualityDimension.INDEPENDENCE: 0.15,
            QualityDimension.SAFETY: 0.15,
        }
        
        overall = sum(
            scores[dim] * weights[dim]
            for dim in scores
        )
        
        # 生成建议
        issues = []
        recommendations = []
        
        for dim, score in scores.items():
            if score < 0.7:
                issues.append(f"{dim.value}得分偏低: {score:.2f}")
                recommendations.append(self._get_recommendation(dim, score))
        
        return QualityReport(
            overall_score=overall,
            dimension_scores=scores,
            issues=issues,
            recommendations=recommendations,
        )
    
    def _evaluate_accuracy(self, data: list[dict]) -> float:
        """评估准确性"""
        
        # 采样评估
        sample_size = min(50, len(data))
        samples = random.sample(data, sample_size)
        
        correct_count = 0
        
        for item in samples:
            # 使用LLM评估事实准确性
            judgment = self.llm.generate(
                messages=[
                    {"role": "system", "content": "你是一个事实核查专家。请判断以下内容的事实准确性。"},
                    {"role": "user", "content": f"内容：{item.get('output', item.get('response', ''))}\n\n请评估：\n1. 内容是否事实正确？\n2. 是否存在明显的事实错误？\n3. 是否有逻辑矛盾？\n\n回答格式：score(0-1), issues(如有)"}
                ],
                temperature=0.1,
            )
            
            score = self._parse_score(judgment)
            correct_count += score
        
        return correct_count / sample_size
    
    def _evaluate_diversity(self, data: list[dict]) -> float:
        """评估多样性"""
        
        # 基于多个指标的多样性评估
        texts = [item.get("instruction", item.get("input", "")) for item in data]
        
        # 1. 长度多样性
        lengths = [len(t) for t in texts]
        length_cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        length_score = min(length_cv / 0.5, 1.0)  # 归一化
        
        # 2. 词汇多样性（Type-Token Ratio）
        all_tokens = " ".join(texts).split()
        unique_tokens = len(set(all_tokens))
        ttr_score = min(unique_tokens / max(len(all_tokens), 1) * 10, 1.0)
        
        # 3. 主题多样性（基于embedding聚类）
        if len(texts) > 10:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 尝试不同的聚类数
            n_clusters = min(10, len(texts) // 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # 计算聚类均衡度
            cluster_sizes = np.bincount(clusters)
            cluster_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
            cluster_score = max(cluster_balance, 0)
        else:
            cluster_score = 0.5
        
        return (length_score + ttr_score + cluster_score) / 3
    
    def _evaluate_independence(self, data: list[dict]) -> float:
        """评估独立性（去重效果）"""
        
        if not self.reference_data:
            return 1.0  # 没有参考数据时默认满分
        
        # 检查合成数据与参考数据的重复率
       合成_texts = set()
        reference_texts = set()
        
        for item in data:
            text = item.get("instruction", item.get("input", ""))
            合成_texts.add(text.lower().strip())
        
        for item in self.reference_data:
            text = item.get("instruction", item.get("input", ""))
            reference_texts.add(text.lower().strip())
        
        # 计算重叠率
        overlap = 合成_texts & reference_texts
        overlap_rate = len(overlap) / max(len(合成_texts), 1)
        
        # 重叠率越低越好
        return 1.0 - min(overlap_rate, 1.0)
```

### 3.3 质量过滤Pipeline

```python
class QualityFilterPipeline:
    """质量过滤管道"""
    
    def __init__(self, evaluator: SyntheticDataEvaluator):
        self.evaluator = evaluator
        self.filters = [
            self._filter_by_length,
            self._filter_by_format,
            self._filter_by_duplicates,
            self._filter_by_harmful_content,
            self._filter_by_quality_score,
        ]
    
    def filter(self, data: list[dict], target_size: int = None) -> list[dict]:
        """执行多级过滤"""
        
        filtered = data
        
        for filter_func in self.filters:
            before_count = len(filtered)
            filtered = filter_func(filtered)
            after_count = len(filtered)
            
            logger.info(
                f"过滤器 {filter_func.__name__}: "
                f"{before_count} → {after_count} "
                f"(移除 {before_count - after_count} 条)"
            )
        
        # 如果指定了目标大小，进行采样
        if target_size and len(filtered) > target_size:
            filtered = self._quality_aware_sampling(filtered, target_size)
        
        return filtered
    
    def _filter_by_length(self, data: list[dict]) -> list[dict]:
        """长度过滤"""
        result = []
        for item in data:
            instruction = item.get("instruction", item.get("input", ""))
            output = item.get("output", item.get("response", ""))
            
            # 指令长度
            if len(instruction) < 10 or len(instruction) > 2000:
                continue
            
            # 输出长度
            if len(output) < 20 or len(output) > 5000:
                continue
            
            result.append(item)
        
        return result
    
    def _filter_by_format(self, data: list[dict]) -> list[dict]:
        """格式过滤"""
        result = []
        for item in data:
            # 检查是否有空值
            if not item.get("instruction") and not item.get("input"):
                continue
            if not item.get("output") and not item.get("response"):
                continue
            
            # 检查是否有明显的格式错误
            output = item.get("output", item.get("response", ""))
            if output.count("```") % 2 != 0:  # 代码块未闭合
                continue
            
            result.append(item)
        
        return result
    
    def _filter_by_duplicates(self, data: list[dict]) -> list[dict]:
        """去重过滤"""
        seen = set()
        result = []
        
        for item in data:
            # 使用指令+输出的组合作为去重key
            key = f"{item.get('instruction', '')}|{item.get('output', '')}"
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            if key_hash not in seen:
                seen.add(key_hash)
                result.append(item)
        
        return result
    
    def _filter_by_harmful_content(self, data: list[dict]) -> list[dict]:
        """有害内容过滤"""
        # 使用关键词 + LLM双重过滤
        harmful_keywords = [
            # 这里应该包含完整的有害关键词列表
            # 实际项目中建议使用专门的内容安全服务
        ]
        
        result = []
        for item in data:
            text = f"{item.get('instruction', '')} {item.get('output', '')}"
            
            # 关键词快速过滤
            if any(kw in text.lower() for kw in harmful_keywords):
                continue
            
            result.append(item)
        
        return result
    
    def _quality_aware_sampling(self, data: list[dict], target_size: int) -> list[dict]:
        """质量感知采样"""
        
        # 评估每条数据的质量
        scored_data = []
        for item in data:
            # 简单的质量评分
            score = self._quick_quality_score(item)
            scored_data.append((score, item))
        
        # 按质量排序
        scored_data.sort(key=lambda x: x[0], reverse=True)
        
        # 选择top-N
        return [item for _, item in scored_data[:target_size]]
    
    def _quick_quality_score(self, item: dict) -> float:
        """快速质量评分"""
        score = 0.0
        
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        
        # 长度适中加分
        if 50 < len(instruction) < 500:
            score += 0.2
        if 100 < len(output) < 2000:
            score += 0.2
        
        # 有结构化内容加分
        if "\n" in output:
            score += 0.1
        if "1." in output or "步骤" in output:
            score += 0.1
        
        # 有示例加分
        if "例如" in output or "比如" in output or "```" in output:
            score += 0.1
        
        # 无明显错误加分
        if "错误" not in output and "抱歉" not in output:
            score += 0.1
        
        return min(score, 1.0)
```

---

## 四、生产实践：构建合成数据工厂

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     合成数据工厂架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  任务定义层   │────▶│  数据生成层   │────▶│  质量控制层   │    │
│  ├──────────────┤     ├──────────────┤     ├──────────────┤    │
│  │ • 领域分析    │     │ • 多模型生成  │     │ • 自动评估    │    │
│  │ • 任务拆解    │     │ • 并行处理    │     │ • 人工抽检    │    │
│  │ • 种子数据    │     │ • 进化迭代    │     │ • 质量过滤    │    │
│  │ • 难度规划    │     │ • 多样性控制  │     │ • 去重处理    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│           │                    │                    │            │
│           ▼                    ▼                    ▼            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    数据存储与管理层                        │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ • 版本控制    • 元数据管理    • 血缘追踪    • 质量报告     │  │
│  └──────────────────────────────────────────────────────────┘  │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    训练集成层                              │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ • 数据格式转换  • 配比策略  • 训练监控  • 效果评估        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 分布式合成管道

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SynthesisJob:
    """合成任务"""
    job_id: str
    topic: str
    target_count: int
    method: str  # "self_instruct", "evol_instruct", "reasoning", etc.
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = None
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)

class DistributedSynthesisPipeline:
    """分布式合成管道"""
    
    def __init__(
        self,
        llm_clients: list,  # 多个LLM客户端，用于并行和冗余
        max_workers: int = 4,
        batch_size: int = 10,
    ):
        self.llm_clients = llm_clients
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_pipeline(
        self,
        job: SynthesisJob,
        quality_threshold: float = 0.7,
    ) -> SynthesisJob:
        """执行完整的合成管道"""
        
        logger.info(f"Starting synthesis job: {job.job_id} - {job.topic}")
        
        try:
            # Phase 1: 生成原始数据
            raw_data = await self._generate_raw_data(job)
            
            # Phase 2: 质量过滤
            filtered_data = await self._quality_filter(raw_data, quality_threshold)
            
            # Phase 3: 去重
            deduplicated = await self._deduplicate(filtered_data)
            
            # Phase 4: 格式化输出
            final_data = await self._format_output(deduplicated, job.method)
            
            job.results = final_data
            job.status = "completed"
            job.completed_at = datetime.now()
            
            logger.info(
                f"Job {job.job_id} completed: "
                f"{len(raw_data)} raw → {len(final_data)} final"
            )
            
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"Job {job.job_id} failed: {e}")
        
        return job
    
    async def _generate_raw_data(self, job: SynthesisJob) -> list[dict]:
        """并行生成原始数据"""
        
        all_data = []
        tasks = []
        
        # 分配工作到不同的LLM客户端
        items_per_client = job.target_count // len(self.llm_clients)
        
        for i, client in enumerate(self.llm_clients):
            task = asyncio.create_task(
                self._generate_batch(
                    client=client,
                    topic=job.topic,
                    method=job.method,
                    target_count=items_per_client,
                )
            )
            tasks.append(task)
        
        # 收集结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Generation task failed: {result}")
            else:
                all_data.extend(result)
        
        return all_data
    
    async def _generate_batch(
        self,
        client,
        topic: str,
        method: str,
        target_count: int,
    ) -> list[dict]:
        """使用单个客户端生成一批数据"""
        
        batch = []
        
        for i in range(0, target_count, self.batch_size):
            batch_size = min(self.batch_size, target_count - i)
            
            try:
                if method == "self_instruct":
                    items = await self._self_instruct_batch(client, topic, batch_size)
                elif method == "evol_instruct":
                    items = await self._evol_instruct_batch(client, topic, batch_size)
                elif method == "reasoning":
                    items = await self._reasoning_batch(client, topic, batch_size)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                batch.extend(items)
                
            except Exception as e:
                logger.warning(f"Batch generation failed: {e}")
                continue
        
        return batch
    
    async def _quality_filter(self, data: list[dict], threshold: float) -> list[dict]:
        """质量过滤"""
        
        evaluator = SyntheticDataEvaluator(self.llm_clients[0])
        
        # 批量评估（避免逐条评估太慢）
        batch_size = 20
        filtered = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # 并行评估
            tasks = [
                asyncio.create_task(self._evaluate_item(evaluator, item))
                for item in batch
            ]
            
            results = await asyncio.gather(*tasks)
            
            for item, score in zip(batch, results):
                if score >= threshold:
                    item["quality_score"] = score
                    filtered.append(item)
        
        return filtered
    
    async def _evaluate_item(self, evaluator, item: dict) -> float:
        """评估单条数据"""
        
        try:
            report = evaluator.evaluate([item])
            return report.overall_score
        except Exception:
            return 0.0
```

### 4.3 合成数据与真实数据的混合策略

```python
class DataMixingStrategy:
    """数据混合策略"""
    
    def __init__(self, real_data: list[dict], synthetic_data: list[dict]):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
    
    def compute_optimal_mix(
        self,
        target_size: int,
        real_ratio: float = 0.3,
        quality_priority: bool = True,
    ) -> list[dict]:
        """计算最优混合比例"""
        
        real_count = int(target_size * real_ratio)
        synthetic_count = target_size - real_count
        
        # 选择最高质量的真实数据
        real_selected = self._select_by_quality(
            self.real_data, real_count
        )
        
        # 选择最高质量的合成数据
        synthetic_selected = self._select_by_quality(
            self.synthetic_data, synthetic_count
        )
        
        # 混合
        mixed = real_selected + synthetic_selected
        
        # 打乱顺序
        random.shuffle(mixed)
        
        return mixed
    
    def adaptive_mixing(
        self,
        target_size: int,
        eval_results: dict,
    ) -> list[dict]:
        """基于评估结果的自适应混合"""
        
        # 根据评估结果调整比例
        if eval_results.get("real_data_quality", 0) > 0.8:
            # 真实数据质量高，增加真实数据比例
            real_ratio = 0.5
        elif eval_results.get("synthetic_data_quality", 0) > 0.7:
            # 合成数据质量不错，可以多用
            real_ratio = 0.2
        else:
            # 都一般，保持默认
            real_ratio = 0.3
        
        return self.compute_optimal_mix(target_size, real_ratio)
```

---

## 五、成本与效率优化

### 5.1 合成数据的成本分析

| 成本项 | Self-Instruct | Evol-Instruct | 推理合成 | 领域合成 |
|-------|--------------|--------------|---------|---------|
| API调用次数 | 10万次 | 30万次 | 20万次 | 50万次 |
| 平均Token/次 | 500 | 800 | 1500 | 2000 |
| 总Token消耗 | 5亿 | 24亿 | 30亿 | 100亿 |
| 估算成本 | $50-100 | $200-500 | $300-600 | $800-2000 |
| 产出数据量 | 5万条 | 3万条 | 2万条 | 1万章 |

### 5.2 成本优化策略

```python
class CostOptimizedSynthesis:
    """成本优化的合成策略"""
    
    def __init__(self, llm_clients: dict):
        # 不同成本级别的LLM
        self.clients = {
            "cheap": llm_clients.get("gpt-4o-mini"),    # $0.15/1M tokens
            "medium": llm_clients.get("gpt-4o"),         # $5/1M tokens
            "expensive": llm_clients.get("o3"),           # $10/1M tokens
        }
    
    def select_model_for_task(self, task_type: str, quality_requirement: str) -> str:
        """根据任务类型和质量要求选择模型"""
        
        model_selection = {
            # 简单任务用便宜模型
            ("generation", "standard"): "cheap",
            ("formatting", "standard"): "cheap",
            ("deduplication", "standard"): "cheap",
            
            # 中等任务用中等模型
            ("generation", "high"): "medium",
            ("evolution", "standard"): "medium",
            ("quality_check", "standard"): "medium",
            
            # 关键任务用最好模型
            ("reasoning", "high"): "expensive",
            ("evaluation", "high"): "expensive",
            ("safety_check", "high"): "expensive",
        }
        
        return model_selection.get((task_type, quality_requirement), "medium")
    
    def batch_process_with_cost_control(
        self,
        data: list[dict],
        budget_limit: float = 100.0,
    ) -> list[dict]:
        """带成本控制的批量处理"""
        
        total_cost = 0.0
        results = []
        
        for item in data:
            # 估算本次调用成本
            estimated_tokens = self._estimate_tokens(item)
            model = self.select_model_for_task(item["type"], item["quality"])
            estimated_cost = self._estimate_cost(model, estimated_tokens)
            
            # 检查预算
            if total_cost + estimated_cost > budget_limit:
                logger.warning(f"Budget limit reached: ${total_cost:.2f}")
                break
            
            # 执行调用
            result = self.clients[model].generate(item["prompt"])
            results.append(result)
            
            total_cost += estimated_cost
        
        logger.info(f"Total cost: ${total_cost:.2f}, processed: {len(results)} items")
        return results
```

---

## 六、总结与最佳实践

### 合成数据的关键原则

```
┌─────────────────────────────────────────────────────────────────┐
│                    合成数据最佳实践                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 质量 > 数量                                                  │
│     └─ 1000条高质量合成数据 > 10万条低质量数据                     │
│                                                                  │
│  2. 多样性是关键                                                  │
│     └─ 避免合成数据的"模式坍缩"，确保覆盖面广                     │
│                                                                  │
│  3. 持续迭代                                                      │
│     └─ 合成数据不是一次性工作，需要根据模型反馈持续优化             │
│                                                                  │
│  4. 混合使用                                                      │
│     └─ 合成数据 + 真实数据，而非替代                              │
│                                                                  │
│  5. 质量监控                                                      │
│     └─ 建立自动化的质量评估体系，持续监控数据质量                  │
│                                                                  │
│  6. 成本控制                                                      │
│     └─ 不同任务使用不同级别的模型，平衡质量和成本                   │
│                                                                  │
│  7. 安全第一                                                      │
│     └─ 严格过滤有害内容，确保合成数据的安全性                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 不同场景的推荐方案

| 场景 | 推荐方法 | 目标产出 | 预算范围 |
|-----|---------|---------|---------|
| 通用能力提升 | Self-Instruct + Evol-Instruct | 5-10万条指令数据 | $100-300 |
| 推理能力增强 | Orca风格推理合成 | 2-5万条推理链 | $300-600 |
| 领域知识注入 | 教科书风格合成 | 1-3万条领域数据 | $500-1500 |
| 代码能力训练 | OSS-Instruct + 代码生成 | 5-10万条代码数据 | $200-500 |
| 对齐训练 | UltraFeedback + 偏好合成 | 1-3万条偏好对 | $400-800 |

合成数据技术已经从"锦上添花"变成了"必选项"。掌握这项技术，将帮助你在模型训练中突破数据瓶颈，用更少的成本获得更好的效果。

---

*如果你在合成数据方面有实践经验，欢迎交流讨论。*
