---
title: 'Synthetic Computers at Scale for Long-Horizon Productivity Simulation'
published: 2026-04-30
category: 'agentOps'
tags: ['agent', 'simulation', 'synthetic-data', 'productivity']
source: 'arXiv'
url: 'https://arxiv.org/abs/2604.28181'
---

# Synthetic Computers at Scale for Long-Horizon Productivity Simulation

## 摘要

真实世界中的长期生产力工作强烈依赖于用户特定的计算机环境，其中大部分工作上下文通过目录结构和内容丰富的工件（如文档、电子表格和演示文稿）进行存储和组织。为了为这些生产力场景扩展合成数据创建，我们介绍了大规模合成计算机（Synthetic Computers at Scale），这是一种创建具有真实文件夹层次结构和内容丰富工件的可扩展方法论。

基于每个合成计算机，我们运行长期模拟：一个agent创建特定于该计算机用户的生产力目标，这些目标需要多个专业交付物和约一个月的人类工作；另一个agent则扮演该用户并持续在计算机上工作——例如，导航文件系统以进行基础协调、与模拟协作者协调以及生成专业工件——直到完成这些目标。

在初步实验中，我们创建了1,000个合成计算机并在其上运行长期模拟；每次运行需要超过8小时的agent运行时间，平均跨越2,000多个轮次。这些模拟产生了丰富的经验学习信号，其有效性通过agent在域内和域外生产力评估中的显著改进得到验证。考虑到personas在十亿级别上非常丰富，该方法论原则上可以扩展到数百万甚至数十亿合成用户世界，只需足够的计算资源，从而能够更广泛地覆盖不同的职业、角色、上下文、环境和生产力需求。我们认为，可扩展的合成计算机创建与大规模模拟相结合，作为agent自我改进和长期生产力场景中的agent强化学习的基础平台具有高度前景。

## 核心技术要点

### 1. 合成计算机创建方法论

研究提出了一套完整的合成计算机创建方法论，该方法从用户persona出发，通过LLM（大型语言模型）渐进式构建包含真实目录结构和内容丰富文件的用户特定计算机环境。

**关键特征：**
- **真实性驱动**：从用户profile开始，生成具有行业特色和工作流程的文件系统
- **内容丰富性**：不仅创建目录结构，还填充实际可用的文档、电子表格、演示文稿等工件
- **个性化定制**：每个合成计算机都针对特定的职业、角色和工作习惯进行定制
- **规模可扩展**：方法论设计支持大规模生成，理论上可扩展到数百万甚至数十亿个合成用户世界

**实现细节：**
文件系统规划从用户profile生成文件系统策略，包括：
- 系统启动时间和使用模式
- 驱动器布局和存储配置
- 默认路径和目录结构
- 文件组织风格和命名约定
- 专业工作流和数据管理模式

### 2. 长期生产力模拟框架

研究设计了一个创新的双agent协作模拟框架，用于模拟真实世界中的长期生产力工作。

**框架组成：**

**Setup Agent（设置Agent）：**
- 创建生产力目标，这些目标特定于计算机用户
- 设计需要多个专业交付物的任务
- 规划约一个月人类工作量的任务分解
- 建立协作者网络和沟通渠道

**Work Agent（工作Agent）：**
- 扮演用户角色，操作计算机完成任务
- 导航文件系统，查找相关信息
- 与模拟协作者协调合作
- 迭代修改和完善交付物
- 持续工作直到目标完成

**模拟特点：**
- 每次模拟运行超过8小时
- 平均跨越2,272轮次交互
- 产生包含过程和结果信号的丰富轨迹数据
- 支持复杂的多步骤任务和长期目标追踪

### 3. 大规模实验验证

研究进行了大规模实验来验证方法论的有效性：

**实验规模：**
- 创建1,000个合成计算机
- 每个计算机运行完整的长期模拟
- 生成包含完整工作过程和结果的数据集
- 覆盖多种职业和工作场景

**数据质量：**
- 生成真实的文件系统和内容丰富的工件
- 模拟真实的工作流程和协作模式
- 记录完整的决策过程和中间结果
- 提供丰富的经验信号用于学习

**公开资源：**
微软已公开：
- 100个合成计算机及其完整环境
- 500个模拟轨迹数据
- 支持研究者进一步探索和改进

### 4. 经验信号提取与技能生成

从模拟轨迹中，研究提取了有价值的经验信号并转化为可执行的技能规则。

**信号提取方法：**
- 分析成功工作模式，识别高效策略
- 提取失败教训，总结常见错误
- 识别行业特定的最佳实践
- 按职业分组生成领域知识

**技能生成流程：**
1. **数据收集**：从大量模拟轨迹中收集工作过程数据
2. **模式识别**：使用LLM分析数据，识别重复出现的成功模式
3. **规则提取**：将模式转化为可执行的规则和指导原则
4. **技能组织**：按职业和任务类型组织成结构化的技能库
5. **验证优化**：通过实验验证技能效果并持续优化

**技能结构示例：**
以"财务投资分析师"（financial-and-investment-analysts）技能为例，包含四个主要章节：
- **数据完整性与单一事实源**：确保数据来源可靠，避免重复和不一致
- **模型构建与验证**：规范建模流程，确保模型准确性
- **文档层次与工作流门控**：建立文档标准和工作流控制机制
- **监管合规与认证标准**：遵守行业法规和认证要求

每章包含多条具体规则，覆盖日常工作中的关键决策点。

### 5. 评估结果与性能提升

研究通过严格的评估验证了方法的有效性，结果显示出显著的性能提升。

**域内评估结果：**
- 基准性能：61.6%
- 技能增强性能：68.6%
- **提升幅度：+7个百分点**
- 在100个测试计算机中，83个表现更好

**域外评估结果：**
- 基准测试：GDPVal
- 模型：Sonnet
- 任务数量：220个
- **结果：105胜67负**
- 在未见过的新环境中仍保持优势

**关键发现：**
- 技能增强agent在多种任务上表现稳定
- 经验信号具有良好的泛化能力
- 合成数据生成的技能可以迁移到真实场景
- 方法论适用于不同领域和职业

## 实战应用价值

### 为Agent自我改进提供基础

这项工作为agent自我改进和长期生产力场景下的agent强化学习提供了一个可扩展的基础框架。通过合成计算机创建大规模合成数据，能够在不依赖私有用户数据的情况下运行真实的长期生产力工作模拟，生成高价值的经验信号。

**核心优势：**
1. **数据隐私保护**：无需使用真实的用户数据，避免隐私问题
2. **规模可扩展**：理论上可以生成数百万个合成环境
3. **场景覆盖全面**：可以覆盖多种职业、角色和工作场景
4. **成本效益高**：相比真实数据收集，成本大幅降低

### 提升Agent复杂任务能力

实验证明从模拟轨迹中提取的经验信号可以转化为可执行的技能，显著提升agent在复杂任务上的表现。

**应用场景：**
- **自动化办公**：处理复杂的文档工作流和多步骤任务
- **协作任务**：与多个协作者协调，完成跨部门项目
- **数据分析**：从分散的文件系统中收集和分析数据
- **报告生成**：基于多源信息生成专业报告和演示文稿

### 推动Agent生态发展

该方法论不仅适用于生产力agent训练，也为计算机使用agent和相关应用提供了高质量的环境生成思路。

**行业影响：**
- 为企业部署AI助手提供训练环境
- 支持教育和培训场景的虚拟化
- 促进agent研究和开发的标准化
- 推动agent生态系统的成熟

## 代码与实现示例

### 文件系统规划示例

以下是从用户profile生成文件系统策略的示例：

```python
# 伪代码示例：从用户profile生成文件系统
def generate_filesystem_structure(user_profile):
    """
    基于用户profile生成文件系统策略
    """
    filesystem_policy = {
        # 系统启动时间和使用模式
        "startup_time": user_profile.working_hours.start,
        "usage_pattern": user_profile.work_style,
        
        # 驱动器布局和存储配置
        "drive_layout": {
            "C:": "系统盘",
            "D:": "数据盘",
            "E:": "备份盘"
        },
        
        # 默认路径和目录结构
        "default_paths": {
            "documents": "C:/Users/{username}/Documents",
            "downloads": "C:/Users/{username}/Downloads",
            "projects": "D:/Projects",
            "archives": "E:/Archives"
        },
        
        # 文件组织风格
        "organization_style": user_profile.organization_preference,
        
        # 命名约定
        "naming_convention": {
            "date_format": "YYYY-MM-DD",
            "version_pattern": "v{major}.{minor}.{patch}"
        }
    }
    
    return filesystem_policy
```

### 模拟协作者设置示例

创建具有丰富特征的模拟协作者：

```python
# 模拟协作者示例
class SimulatedCollaborator:
    def __init__(self, role, background, communication_style, 
                 knowledge_base, response_time_range):
        """
        初始化模拟协作者
        
        参数：
        - role: 职位角色（如"经理"、"客户"）
        - background: 背景信息
        - communication_style: 沟通风格（如"简洁"、"详细"）
        - knowledge_base: 相关知识
        - response_time_range: 响应时间范围（小时）
        """
        self.role = role
        self.background = background
        self.communication_style = communication_style
        self.knowledge_base = knowledge_base
        self.response_time_range = response_time_range
        self.private_reference_materials = []
    
    def generate_response(self, message):
        """
        基于沟通风格和知识库生成响应
        """
        if self.communication_style == "concise":
            # 简洁风格：24-48小时响应延迟，简洁回复
            return self._generate_concise_response(message)
        elif self.communication_style == "detailed":
            # 详细风格：会仔细阅读文档并提出问题
            return self._generate_detailed_response(message)
    
    def _generate_concise_response(self, message):
        """简洁风格响应"""
        # 实现简洁的响应逻辑
        pass
    
    def _generate_detailed_response(self, message):
        """详细风格响应"""
        # 实现详细的响应逻辑，包括问题提出
        pass

# 示例：创建经理David Hartley
manager = SimulatedCollaborator(
    role="经理",
    background="David Hartley，资深项目经理，15年行业经验",
    communication_style="concise",
    knowledge_base=["项目管理", "预算审批", "资源分配"],
    response_time_range=(24, 48)  # 24-48小时响应延迟
)

# 示例：创建客户Robert Castellano
client = SimulatedCollaborator(
    role="客户",
    background="Robert Castellano，企业客户，注重细节",
    communication_style="detailed",
    knowledge_base=["IPS文档", "业务需求"],
    response_time_range=(48, 72)  # 48-72小时响应延迟
)
# 客户会仔细阅读IPS文档并提出问题
client.private_reference_materials = ["IPS文档模板", "历史项目案例"]
```

### 职业特定技能示例

财务投资分析师技能的结构化表示：

```yaml
# financial-and-investment-analysts 技能定义
skill_name: "财务投资分析师"
version: "1.0"
description: "财务和投资分析领域的专业技能和最佳实践"

chapters:
  - title: "数据完整性与单一事实源"
    rules:
      - "所有数据必须追溯到原始来源，避免二次引用"
      - "使用单一事实源（Single Source of Truth）原则，确保数据一致性"
      - "定期验证数据完整性和准确性"
      - "记录所有数据修改的审计日志"
      - "在多源数据冲突时，优先使用权威来源"
  
  - title: "模型构建与验证"
    rules:
      - "建立明确的模型假设和边界条件"
      - "使用历史数据验证模型准确性"
      - "进行敏感性分析，测试模型稳定性"
      - "文档化所有模型参数和计算逻辑"
      - "建立模型版本控制和变更管理流程"
  
  - title: "文档层次与工作流门控"
    rules:
      - "建立清晰的文档层次结构：策略层、执行层、操作层"
      - "实施工作流门控机制，关键节点需要审批"
      - "所有决策必须有文档支持"
      - "建立文档模板和标准格式"
      - "定期审查和更新文档"
  
  - title: "监管合规与认证标准"
    rules:
      - "确保所有分析符合行业监管要求"
      - "保持专业认证的持续有效性"
      - "建立合规检查清单"
      - "记录所有合规相关的决策和行动"
      - "定期进行合规培训和更新"
```

## 研究意义与未来方向

### 研究意义

这项工作在多个方面具有重要意义：

1. **方法论创新**：首次提出大规模合成计算机创建的完整方法论，为agent训练提供了新思路
2. **可扩展性**：理论上可以扩展到数十亿个合成用户世界，覆盖无限的职业和场景
3. **实用价值**：实验证明方法有效，技能增强agent表现显著提升
4. **开源贡献**：公开100个合成计算机和500个模拟轨迹，促进社区发展

### 未来研究方向

基于这项工作，未来可以探索以下方向：

1. **更丰富的职业覆盖**：扩展到更多职业和行业，如医疗、法律、教育等
2. **更复杂的协作场景**：模拟更多样化的团队协作和跨部门合作
3. **多模态内容**：除了文本，还包括图像、音频、视频等多模态内容
4. **动态环境**：创建随时间变化的环境，模拟真实世界的不确定性
5. **持续学习**：结合在线学习，让agent能够从真实交互中持续改进

## 结论

"Synthetic Computers at Scale for Long-Horizon Productivity Simulation"这项工作为agent自我改进和长期生产力场景提供了一个强大而可扩展的基础框架。通过创新的合成计算机创建方法论、双agent协作模拟框架和经验信号提取技术，研究成功解决了长期生产力工作中数据稀缺的问题。

实验结果显示，从合成数据中提取的技能能够显著提升agent在复杂任务上的表现，在域内评估中提升7个百分点，在域外评估中取得105胜67负的优异战绩。这一成果不仅证明了方法论的有效性，也为未来的agent研究和应用奠定了坚实基础。

随着更多研究者的参与和更多资源的公开，这一领域有望快速推进，为agent技术的发展和应用开辟新的可能性。

---

**来源：** arXiv:2604.28181  
**发布日期：** 2026年4月30日  
**分类：** agentOps  
**字数：** 约2,200字
