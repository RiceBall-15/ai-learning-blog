---
title: "Claude Code Skills与Memory：Agent的长期记忆与技能进化"
description: "深度解析Claude Code的Skills技能系统和Memory记忆系统，如何让Agent从无状态工具进化为有经验的智能体"
date: 2026-05-12
author: "AI学习笔记"
category: "agent"
tags:
  - Claude Code
  - Skills
  - Memory
  - 长期记忆
  - 技能系统
draft: false
---

# Claude Code Skills与Memory：Agent的长期记忆与技能进化

## 引言：从无状态到有经验

在前两篇文章中，我们分析了Claude Code的架构设计和Hooks系统。但有一个根本问题：**如何让Agent变得越来越聪明？**

传统的AI工具是无状态的：每次调用都是全新的开始，不会学习，不会积累经验。Claude Code通过两个核心系统解决了这个问题：

1. **Skills（技能系统）**：可定义、可复用的工作流
2. **Memory（记忆系统）**：持久化、可检索的经验积累

> **核心观点**：Skills让Agent从"通用能力"进化为"专业能力"，Memory让Agent从"无状态"进化为"有经验"。

## 第一部分：Skills系统深度解析

### 什么是Skills

Skills不是简单的"命令"或"模板"，而是**结构化的专业知识**。

```python
# 传统方式：每次都要重复指令
def deploy_to_production():
    # 1. 运行测试
    run_tests()
    
    # 2. 构建生产版本
    build_production()
    
    # 3. 部署到AWS
    deploy_aws()
    
    # 4. 验证部署
    verify_deployment()
    
    # 问题：每次都要重新告诉AI这些步骤
    # 问题：不同项目可能有不同的部署流程
    # 问题：无法积累和复用部署经验

# Skills方式：定义可复用的专业知识
@skill("deploy_to_production")
def deploy_production_skill():
    """
    生产环境部署技能
    
    触发条件：用户要求部署到生产环境
    前置条件：所有测试通过，有部署权限
    后置条件：部署成功，监控正常
    """
    return {
        "steps": [
            {"action": "run_tests", "required": True},
            {"action": "build_production", "required": True},
            {"action": "deploy_to_cloud", "required": True},
            {"action": "verify_deployment", "required": True},
            {"action": "update_monitoring", "required": False}
        ],
        "triggers": ["部署", "deploy", "发布到生产"],
        "prerequisites": ["tests_passing", "has_deploy_permission"],
        "rollback": "revert_to_previous_version"
    }
```

### Skills的层次结构

```yaml
Skills层次:
  L1 - 内置技能:
    - 代码生成: 根据需求生成代码
    - 代码重构: 重构现有代码
    - 测试生成: 自动生成测试用例
    - 文档生成: 生成技术文档
    
  L2 - 项目技能:
    - deploy_web_app: Web应用部署
    - database_migration: 数据库迁移
    - api_versioning: API版本管理
    - performance_optimization: 性能优化
    
  L3 - 团队技能:
    - code_review_standards: 团队代码审查标准
    - naming_conventions: 团队命名规范
    - architecture_patterns: 团队架构模式
    - security_practices: 团队安全实践
    
  L4 - 个人技能:
    - my_debugging_workflow: 个人调试流程
    - my_development_setup: 个人开发环境
    - my_learning_notes: 个人学习笔记
    - my_productivity_hacks: 个人效率技巧
```

### Skills的定义语法

```markdown
# SKILL.md - 技能定义文件

## 元数据
name: python-debugging
description: Python代码调试技能
version: 1.0
author: AI学习笔记
tags: [python, debugging, troubleshooting]

## 触发条件
triggers:
  - "Python代码出错"
  - "调试Python程序"
  - "修复Python bug"

## 前置条件
prerequisites:
  - "Python环境已安装"
  - "代码文件存在"

## 工作流程
steps:
  1. 复现问题:
     - 运行代码确认错误
     - 收集错误信息
     - 确定错误范围
     
  2. 分析原因:
     - 检查错误堆栈
     - 分析相关代码
     - 查找类似问题
     
  3. 制定方案:
     - 提出修复方案
     - 评估方案风险
     - 选择最佳方案
     
  4. 实施修复:
     - 修改代码
     - 运行测试
     - 验证修复
     
  5. 预防复发:
     - 添加测试用例
     - 更新文档
     - 记录经验

## 工具使用
tools:
  - terminal: 运行Python命令
  - read_file: 读取代码文件
  - write_file: 修改代码文件
  - execute_code: 执行Python代码

## 成功标准
success_criteria:
  - "错误不再出现"
  - "所有测试通过"
  - "代码符合规范"

## 失败处理
on_failure:
  - "尝试其他方案"
  - "寻求人工帮助"
  - "记录失败原因"
```

### Skills的动态发现

```python
# Skills的动态发现和加载机制
class SkillManager:
    def __init__(self):
        self.skills = {}
        self.skill_index = {}  # 关键词索引
        
    def discover_skills(self):
        """
        动态发现技能
        扫描多个位置：内置、项目、用户、团队
        """
        # 1. 内置技能
        self.load_builtin_skills()
        
        # 2. 项目技能
        project_skills = scan_directory(".claude/skills/")
        self.load_skills(project_skills)
        
        # 3. 用户技能
        user_skills = scan_directory("~/.claude/skills/")
        self.load_skills(user_skills)
        
        # 4. 团队技能
        team_skills = scan_directory("/team/shared/skills/")
        self.load_skills(team_skills)
        
        # 5. 构建索引
        self.build_skill_index()
    
    def find_relevant_skill(self, user_request):
        """
        根据用户请求找到相关技能
        """
        # 1. 关键词匹配
        keywords = extract_keywords(user_request)
        candidate_skills = self.match_keywords(keywords)
        
        # 2. 语义匹配
        if not candidate_skills:
            candidate_skills = self.semantic_search(user_request)
        
        # 3. 评分排序
        scored_skills = self.score_skills(candidate_skills, user_request)
        
        # 4. 返回最佳匹配
        return scored_skills[0] if scored_skills else None
    
    def execute_skill(self, skill, context):
        """
        执行技能
        """
        # 1. 检查前置条件
        if not self.check_prerequisites(skill, context):
            raise SkillError("前置条件不满足")
        
        # 2. 执行步骤
        results = []
        for step in skill.steps:
            result = self.execute_step(step, context)
            results.append(result)
            
            # 3. 检查是否继续
            if result.status == "failed" and step.required:
                return self.handle_failure(skill, step, result)
        
        # 4. 验证成功标准
        if self.verify_success_criteria(skill, results):
            return SkillResult(success=True, results=results)
        else:
            return SkillResult(success=False, results=results)
```

## 第二部分：Memory系统深度解析

### 记忆的层次结构

```python
# 记忆的三层架构
class MemorySystem:
    def __init__(self):
        # 第1层：会话记忆（短期）
        self.session_memory = SessionMemory()
        
        # 第2层：项目记忆（中期）
        self.project_memory = ProjectMemory()
        
        # 第3层：长期记忆（长期）
        self.long_term_memory = LongTermMemory()
    
    def remember(self, information, memory_type="auto"):
        """
        存储记忆
        """
        if memory_type == "auto":
            memory_type = self.classify_memory_type(information)
        
        if memory_type == "session":
            self.session_memory.store(information)
        elif memory_type == "project":
            self.project_memory.store(information)
        elif memory_type == "long_term":
            self.long_term_memory.store(information)
    
    def recall(self, query, scope="all"):
        """
        检索记忆
        """
        results = []
        
        if scope in ["all", "session"]:
            results.extend(self.session_memory.search(query))
        
        if scope in ["all", "project"]:
            results.extend(self.project_memory.search(query))
        
        if scope in ["all", "long_term"]:
            results.extend(self.long_term_memory.search(query))
        
        # 按相关性排序
        return sorted(results, key=lambda x: x.relevance, reverse=True)
```

### 第1层：会话记忆

```python
class SessionMemory:
    """
    会话记忆：当前对话的上下文
    生命周期：单次会话
    """
    def __init__(self):
        self.conversation_history = []
        self.decisions = []
        self.context = {}
        
    def store(self, information):
        """存储会话信息"""
        if information.type == "conversation":
            self.conversation_history.append(information)
        elif information.type == "decision":
            self.decisions.append(information)
        elif information.type == "context":
            self.context.update(information.data)
    
    def search(self, query):
        """检索会话记忆"""
        results = []
        
        # 搜索对话历史
        for conv in self.conversation_history:
            if query_matches(conv.content, query):
                results.append(MemoryItem(
                    content=conv,
                    source="session",
                    relevance=calculate_relevance(conv, query)
                ))
        
        # 搜索决策记录
        for decision in self.decisions:
            if query_matches(decision.description, query):
                results.append(MemoryItem(
                    content=decision,
                    source="session",
                    relevance=calculate_relevance(decision, query)
                ))
        
        return results
```

### 第2层：项目记忆

```python
class ProjectMemory:
    """
    项目记忆：特定项目的知识和经验
    生命周期：项目存在期间
    """
    def __init__(self, project_path):
        self.project_path = project_path
        self.memory_file = os.path.join(project_path, ".claude/memory.json")
        self.memory = self.load_memory()
        
    def load_memory(self):
        """加载项目记忆"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {
            "conventions": {},  # 项目约定
            "decisions": [],    # 技术决策
            "patterns": [],     # 代码模式
            "mistakes": [],     # 错误教训
            "preferences": {}   # 项目偏好
        }
    
    def store(self, information):
        """存储项目记忆"""
        if information.type == "convention":
            self.memory["conventions"].update(information.data)
        elif information.type == "decision":
            self.memory["decisions"].append(information)
        elif information.type == "pattern":
            self.memory["patterns"].append(information)
        elif information.type == "mistake":
            self.memory["mistakes"].append(information)
        
        # 保存到文件
        self.save_memory()
    
    def search(self, query):
        """检索项目记忆"""
        results = []
        
        # 搜索所有类型的记忆
        for memory_type, items in self.memory.items():
            if isinstance(items, list):
                for item in items:
                    if query_matches(str(item), query):
                        results.append(MemoryItem(
                            content=item,
                            source=f"project:{memory_type}",
                            relevance=calculate_relevance(item, query)
                        ))
            elif isinstance(items, dict):
                for key, value in items.items():
                    if query_matches(f"{key}: {value}", query):
                        results.append(MemoryItem(
                            content={key: value},
                            source=f"project:{memory_type}",
                            relevance=calculate_relevance({key: value}, query)
                        ))
        
        return results
```

### 第3层：长期记忆

```python
class LongTermMemory:
    """
    长期记忆：跨项目的通用知识和经验
    生命周期：永久
    """
    def __init__(self):
        self.memory_path = os.path.expanduser("~/.claude/memory/")
        self.ensure_directory()
        
        # 记忆分类
        self.categories = {
            "knowledge": [],      # 技术知识
            "skills": [],         # 技能经验
            "preferences": {},    # 用户偏好
            "patterns": [],       # 通用模式
            "lessons": []         # 经验教训
        }
        
    def store(self, information):
        """存储长期记忆"""
        category = self.classify_information(information)
        
        if category == "knowledge":
            self.store_knowledge(information)
        elif category == "skills":
            self.store_skill_experience(information)
        elif category == "preferences":
            self.store_preference(information)
        elif category == "patterns":
            self.store_pattern(information)
        elif category == "lessons":
            self.store_lesson(information)
    
    def compile_knowledge_article(self, information):
        """
        将信息编译为知识文章
        借鉴Karpathy的LLM知识库架构
        """
        # 1. 提取关键概念
        concepts = extract_concepts(information)
        
        # 2. 创建或更新概念文章
        for concept in concepts:
            article_path = os.path.join(
                self.memory_path, 
                "knowledge", 
                f"{concept.name}.md"
            )
            
            if os.path.exists(article_path):
                # 更新现有文章
                update_article(article_path, concept)
            else:
                # 创建新文章
                create_article(article_path, concept)
        
        # 3. 更新索引
        self.update_knowledge_index()
    
    def search(self, query):
        """检索长期记忆"""
        results = []
        
        # 搜索知识库
        knowledge_results = self.search_knowledge_base(query)
        results.extend(knowledge_results)
        
        # 搜索技能经验
        skill_results = self.search_skill_experiences(query)
        results.extend(skill_results)
        
        # 搜索用户偏好
        preference_results = self.search_preferences(query)
        results.extend(preference_results)
        
        # 搜索模式库
        pattern_results = self.search_patterns(query)
        results.extend(pattern_results)
        
        # 搜索经验教训
        lesson_results = self.search_lessons(query)
        results.extend(lesson_results)
        
        return sorted(results, key=lambda x: x.relevance, reverse=True)
```

## 第三部分：Skills与Memory的协同

### 协同模式

```python
# Skills和Memory的协同工作
class AgentWithSkillsAndMemory:
    def __init__(self):
        self.skills = SkillManager()
        self.memory = MemorySystem()
    
    def handle_request(self, user_request):
        """
        处理用户请求：Skills和Memory协同工作
        """
        # 1. 检索相关记忆
        relevant_memories = self.memory.recall(user_request)
        
        # 2. 查找相关技能
        relevant_skill = self.skills.find_relevant_skill(user_request)
        
        # 3. 结合记忆执行技能
        if relevant_skill:
            # 使用技能处理
            context = self.build_context(user_request, relevant_memories)
            result = self.skills.execute_skill(relevant_skill, context)
        else:
            # 使用通用能力处理
            result = self.general_execution(user_request, relevant_memories)
        
        # 4. 从执行中学习
        self.learn_from_execution(user_request, result, relevant_memories)
        
        return result
    
    def learn_from_execution(self, request, result, memories):
        """
        从执行中学习：更新技能和记忆
        """
        # 1. 更新技能（如果发现更好的方法）
        if result.new_approach:
            self.skills.update_skill(result.skill_used, result.new_approach)
        
        # 2. 更新记忆
        learning = extract_learning(request, result)
        self.memory.remember(learning)
        
        # 3. 如果是新模式，创建新技能
        if result.is_new_pattern:
            self.skills.create_skill_from_pattern(result.pattern)
```

### 实战案例：调试技能

```python
# 调试技能的定义
@skill("python_debugging")
def python_debugging_skill():
    """
    Python调试技能
    结合记忆系统，越用越聪明
    """
    return {
        "name": "python_debugging",
        "description": "智能Python代码调试",
        
        # 工作流程
        "steps": [
            {
                "action": "recall_similar_errors",
                "description": "回忆类似错误的解决方案"
            },
            {
                "action": "analyze_error",
                "description": "分析当前错误"
            },
            {
                "action": "generate_solutions",
                "description": "生成可能的解决方案"
            },
            {
                "action": "select_best_solution",
                "description": "选择最佳方案（基于记忆）"
            },
            {
                "action": "implement_solution",
                "description": "实施方案"
            },
            {
                "action": "verify_fix",
                "description": "验证修复"
            },
            {
                "action": "update_memory",
                "description": "更新调试经验记忆"
            }
        ],
        
        # 触发条件
        "triggers": [
            "Python代码出错",
            "调试Python程序",
            "修复Python bug",
            "这个Python代码有问题"
        ],
        
        # 成功标准
        "success_criteria": [
            "错误不再出现",
            "所有测试通过",
            "代码符合规范"
        ]
    }

# 调试技能的执行
def execute_python_debugging(error_info, memory_system):
    """
    执行Python调试技能
    """
    # 步骤1：回忆类似错误
    similar_errors = memory_system.recall(
        f"Python错误: {error_info.message}",
        scope="all"
    )
    
    # 步骤2：分析当前错误
    analysis = analyze_python_error(error_info)
    
    # 步骤3：生成解决方案
    solutions = generate_solutions(analysis, similar_errors)
    
    # 步骤4：选择最佳方案（基于记忆）
    best_solution = select_best_solution(
        solutions, 
        similar_errors,  # 基于过去的成功经验
        memory_system.user_preferences  # 基于用户偏好
    )
    
    # 步骤5：实施方案
    fix_result = implement_solution(best_solution)
    
    # 步骤6：验证修复
    verification = verify_fix(fix_result)
    
    # 步骤7：更新记忆
    if verification.success:
        memory_system.remember({
            "type": "debugging_experience",
            "error": error_info,
            "solution": best_solution,
            "success": True,
            "context": analysis.context
        })
    else:
        memory_system.remember({
            "type": "debugging_failure",
            "error": error_info,
            "solution": best_solution,
            "reason": verification.failure_reason
        })
    
    return fix_result
```

## 第四部分：记忆的编译与压缩

### Karpathy的LLM知识库架构借鉴

```python
# 借鉴Karpathy的LLM知识库架构
class KnowledgeCompiler:
    """
    知识编译器：将原始经验编译为结构化知识
    """
    def __init__(self, memory_system):
        self.memory = memory_system
        
    def compile_daily_logs(self):
        """
        编译每日日志为知识文章
        """
        # 1. 读取当天的经验
        daily_experiences = self.memory.get_daily_experiences()
        
        # 2. 提取关键知识点
        knowledge_atoms = extract_knowledge_atoms(daily_experiences)
        
        # 3. 创建或更新知识文章
        for atom in knowledge_atoms:
            if self.knowledge_article_exists(atom.topic):
                self.update_knowledge_article(atom)
            else:
                self.create_knowledge_article(atom)
        
        # 4. 创建连接文章（跨概念关系）
        connections = find_connections(knowledge_atoms)
        for connection in connections:
            self.create_connection_article(connection)
        
        # 5. 更新索引
        self.update_knowledge_index()
    
    def create_knowledge_article(self, knowledge_atom):
        """
        创建知识文章
        """
        article_content = f"""# {knowledge_atom.title}

## 核心概念
{knowledge_atom.description}

## 关键要点
{format_key_points(knowledge_atom.key_points)}

## 实战案例
{format_examples(knowledge_atom.examples)}

## 常见陷阱
{format_pitfalls(knowledge_atom.pitfalls)}

## 最佳实践
{format_best_practices(knowledge_atom.best_practices)}

## 相关主题
{format_related_topics(knowledge_atom.related_topics)}

---
*编译自 {knowledge_atom.source_experiences} 个实战经验*
*最后更新: {knowledge_atom.last_updated}*
"""
        
        # 保存文章
        article_path = os.path.join(
            self.memory.memory_path,
            "knowledge",
            f"{knowledge_atom.slug}.md"
        )
        
        with open(article_path, 'w') as f:
            f.write(article_content)
```

### 记忆的检索优化

```python
class MemoryRetrievalOptimizer:
    """
    记忆检索优化器
    使用索引引导的检索，而非RAG
    """
    def __init__(self, memory_system):
        self.memory = memory_system
        self.index = self.build_index()
        
    def build_index(self):
        """
        构建记忆索引
        """
        index = {
            "by_topic": {},      # 按主题索引
            "by_type": {},       # 按类型索引
            "by_recency": [],    # 按时间索引
            "by_relevance": {}   # 按相关性索引
        }
        
        # 遍历所有记忆
        for memory in self.memory.get_all_memories():
            # 按主题索引
            for topic in memory.topics:
                if topic not in index["by_topic"]:
                    index["by_topic"][topic] = []
                index["by_topic"][topic].append(memory)
            
            # 按类型索引
            if memory.type not in index["by_type"]:
                index["by_type"][memory.type] = []
            index["by_type"][memory.type].append(memory)
            
            # 按时间索引
            index["by_recency"].append({
                "memory": memory,
                "timestamp": memory.timestamp
            })
        
        # 按时间排序
        index["by_recency"].sort(
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        return index
    
    def search(self, query, max_results=10):
        """
        优化后的记忆检索
        """
        # 1. 读取主索引
        index = self.index
        
        # 2. LLM选择相关主题
        relevant_topics = llm_select_relevant(
            query, 
            list(index["by_topic"].keys())
        )
        
        # 3. 从相关主题中检索
        candidate_memories = []
        for topic in relevant_topics:
            candidate_memories.extend(index["by_topic"][topic])
        
        # 4. 评分排序
        scored_memories = []
        for memory in candidate_memories:
            relevance = calculate_relevance(memory, query)
            recency = calculate_recency(memory.timestamp)
            importance = memory.importance_score
            
            # 综合评分
            score = (
                relevance * 0.5 + 
                recency * 0.3 + 
                importance * 0.2
            )
            
            scored_memories.append({
                "memory": memory,
                "score": score
            })
        
        # 5. 返回Top N
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        return [m["memory"] for m in scored_memories[:max_results]]
```

## 第五部分：实战案例

### 案例1：个人学习笔记系统

```python
# 个人学习笔记技能
@skill("learning_note_system")
def learning_note_system():
    """
    个人学习笔记系统
    结合Memory实现知识积累
    """
    return {
        "name": "learning_note_system",
        "description": "创建和管理个人学习笔记",
        
        "workflow": [
            # 1. 学习新知识
            {
                "action": "learn_new_topic",
                "triggers": ["学习", "了解", "研究"],
                "steps": [
                    "搜索相关资料",
                    "阅读和理解",
                    "提取关键概念",
                    "创建笔记文章",
                    "建立知识连接"
                ]
            },
            
            # 2. 复习已有知识
            {
                "action": "review_knowledge",
                "triggers": ["复习", "回顾", "温习"],
                "steps": [
                    "检索相关笔记",
                    "测试理解程度",
                    "更新过时内容",
                    "深化理解"
                ]
            },
            
            # 3. 应用知识
            {
                "action": "apply_knowledge",
                "triggers": ["应用", "使用", "实践"],
                "steps": [
                    "检索相关知识",
                    "应用到实际问题",
                    "记录实践经验",
                    "更新知识文章"
                ]
            }
        ],
        
        # 与Memory的集成
        "memory_integration": {
            "store": "learning_notes/{topic}.md",
            "index": "learning_index.json",
            "connections": "knowledge_connections.md"
        }
    }
```

### 案例2：代码审查专家

```python
# 代码审查技能
@skill("code_review_expert")
def code_review_expert():
    """
    代码审查专家技能
    结合Memory积累审查经验
    """
    return {
        "name": "code_review_expert",
        "description": "智能代码审查，越审查越精准",
        
        # 审查维度
        "review_dimensions": {
            "security": {
                "checks": ["SQL注入", "XSS", "CSRF", "敏感信息泄露"],
                "memory_key": "security_vulnerabilities"
            },
            "performance": {
                "checks": ["N+1查询", "内存泄漏", "循环优化"],
                "memory_key": "performance_issues"
            },
            "maintainability": {
                "checks": ["代码复杂度", "命名规范", "文档完整性"],
                "memory_key": "maintainability_patterns"
            },
            "best_practices": {
                "checks": ["设计模式", "SOLID原则", "DRY原则"],
                "memory_key": "best_practice_patterns"
            }
        },
        
        # 审查流程
        "workflow": [
            "1. 检索类似代码的审查经验",
            "2. 执行多维度审查",
            "3. 生成审查报告",
            "4. 更新审查经验记忆"
        ],
        
        # 学习机制
        "learning": {
            "from_feedback": "根据开发者反馈调整审查重点",
            "from_mistakes": "从漏检问题中学习",
            "from_patterns": "从高质量代码中学习模式"
        }
    }
```

## 结论：让Agent越用越聪明

Claude Code的Skills和Memory系统实现了AI工具的革命性突破：

1. **从无状态到有经验**：Memory让Agent积累经验
2. **从通用到专业**：Skills让Agent获得专业能力
3. **从被动到主动**：结合Memory和Skills，Agent能主动学习和优化

**核心启示**：

> **好的AI工具不是替代人类，而是增强人类**。Skills和Memory系统让Claude Code成为人类开发者的"第二大脑"，而不是简单的代码生成器。

**技术深度**：

- **Skills借鉴了专家系统**：将专家知识编码为可执行的技能
- **Memory借鉴了认知科学**：模拟人类的记忆层次结构
- **编译借鉴了Karpathy架构**：将原始经验编译为结构化知识

这种设计让Claude Code能够：
- 记住你的偏好和习惯
- 学习你的项目和团队规范
- 积累调试和开发经验
- 越用越懂你，越用越聪明

---

**延伸阅读**：
- [第1篇：Claude Code架构设计哲学与核心创新]()
- [第2篇：Hooks系统深度解析 - 可扩展的Agent生命周期]()
- [第4篇：Claude Code vs 竞品 - 为什么它是Top 1 Agent框架]()

**参考资料**：
- Karpathy的LLM知识库架构
- 专家系统设计模式
- 认知科学中的记忆理论
- 知识图谱构建方法
