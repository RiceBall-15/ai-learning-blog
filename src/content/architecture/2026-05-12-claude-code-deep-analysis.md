---
title: "Claude Code 深度架构调研：从Hook系统到多Agent协作"
description: "深入分析Claude Code的核心架构，包括Hook系统、MCP协议、工具系统、子Agent协作、上下文管理和安全机制，以及可借鉴的设计模式"
date: 2026-05-12
author: RiceBall-15
category: architecture
tags: [claude-code, agent, hook, mcp, architecture, multi-agent]
draft: false
---

# Claude Code 深度架构调研

> **调研时间**: 2026-05-12  
> **调研目的**: 深入理解Claude Code的核心架构，参考其实用流程放到自己的Agent中  
> **资料来源**: GitHub官方仓库 (122K Stars)、官方文档、插件系统源码

---

## 一、Claude Code 概述

### 1.1 是什么
Claude Code是Anthropic开发的**自主编程Agent**，运行在终端中，通过自然语言指令理解代码库并执行编程任务。

### 1.2 核心定位
- **不是IDE插件**：而是独立的TUI应用（终端UI）
- **不是Chatbot**：而是有状态的Agent，可以自主执行多步骤任务
- **不是单一工具**：而是**编排系统**，可以协调多个子Agent

### 1.3 关键特性
- 自主编码：读写文件、执行命令、管理Git工作流
- 多Agent协作：可同时运行多个专业子Agent
- Hook系统：事件驱动的自动化机制
- MCP集成：外部工具服务器的统一接口
- 上下文感知：项目级记忆（CLAUDE.md）

---

## 二、核心架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ TUI终端界面   │  │ VSCode集成   │  │ CLI Print模式 │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      会话管理层                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Session Manager                                       │   │
│  │  - 会话持久化（5小时TTL）                                │   │
│  │  - 上下文压缩（70%阈值触发）                             │   │
│  │  - 检查点机制（/rewind回滚）                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      编排引擎层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Agent Loop   │  │ Hook Engine  │  │ Tool Dispatch│       │
│  │ 主循环编排    │  │ 事件触发器   │  │ 工具调度器    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      工具集成层                               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ 内置工具    │ │ MCP服务器  │ │ 子Agent    │ │ Shell    │ │
│  │ Read/Write │ │ 外部服务   │ │ 专业Agent  │ │ 命令执行  │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      模型调用层                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Anthropic API (Claude Sonnet/Opus/Haiku)              │   │
│  │  - Function Calling                                    │   │
│  │  - Prompt Caching                                      │   │
│  │  - Reasoning Tokens (Extended Thinking)                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、Hook系统架构（最值得借鉴的设计）

### 3.1 Hook类型全览

Claude Code定义了**8种Hook事件**，覆盖Agent生命周期的每个关键节点：

| Hook类型 | 触发时机 | 典型用途 |
|---------|---------|---------|
| `UserPromptSubmit` | 用户提交提示前 | 输入验证、日志记录 |
| `PreToolUse` | 工具执行前 | 安全检查、危险命令拦截 |
| `PostToolUse` | 工具执行后 | 自动格式化、代码检查 |
| `Notification` | 权限请求/等待输入时 | 桌面通知、告警 |
| `Stop` | Claude完成响应时 | 完成日志、状态更新 |
| `SubagentStop` | 子Agent完成时 | Agent编排、结果汇总 |
| `PreCompact` | 上下文压缩前 | 备份会话转录 |
| `SessionStart` | 会话开始时 | 加载开发上下文 |

### 3.2 Hook配置结构

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",           // 匹配工具类型
      "hooks": [{
        "type": "command",          // Hook类型
        "command": "安全检查脚本"    // 执行的命令
      }]
    }],
    "PostToolUse": [{
      "matcher": "Write(*.py)",     // 支持通配符匹配
      "hooks": [{
        "type": "command",
        "command": "ruff check --fix $CLAUDE_FILE_PATHS"
      }]
    }]
  }
}
```

### 3.3 Hook环境变量

Hook执行时可以访问以下环境变量：

| 变量 | 内容 |
|-----|------|
| `CLAUDE_PROJECT_DIR` | 当前项目路径 |
| `CLAUDE_FILE_PATHS` | 正在修改的文件 |
| `CLAUDE_TOOL_INPUT` | 工具参数（JSON格式） |

### 3.4 安全拦截机制

```bash
# PreToolUse Hook 示例：拦截危险命令
if echo "$CLAUDE_TOOL_INPUT" | grep -qE 'rm -rf|git push.*--force'; then
  echo '危险命令被拦截!'
  exit 2  # exit 2 = 阻止执行
fi
```

**关键设计**：
- `exit 0` = 允许继续
- `exit 2` = 阻止执行（但不终止会话）
- `exit 1` = 错误（会话继续但工具不执行）

### 3.5 借鉴价值

**为什么Hook系统设计优秀？**
1. **事件驱动**：不侵入主循环，通过外部脚本响应事件
2. **声明式配置**：JSON配置，无需修改代码
3. **细粒度控制**：matcher支持通配符，精确控制触发条件
4. **安全优先**：PreToolUse可以拦截危险操作
5. **可扩展**：任何命令行工具都可以作为Hook

---

## 四、MCP协议集成

### 4.1 MCP是什么

Model Context Protocol（模型上下文协议）是Anthropic提出的**开放标准**，用于将AI模型连接到外部工具和服务。

### 4.2 MCP架构

```
┌─────────────────────────────────────────────┐
│              Claude Code (Client)            │
│  ┌─────────────────────────────────────┐    │
│  │  MCP Client                          │    │
│  │  - 发现工具                            │    │
│  │  - 调用工具                            │    │
│  │  - 处理结果                            │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                    │
                    │ 协议通信
                    ▼
┌─────────────────────────────────────────────┐
│              MCP Server                      │
│  ┌─────────────┐  ┌─────────────┐           │
│  │ 工具暴露     │  │ 资源暴露     │           │
│  │ query_db    │  │ db://tables │           │
│  │ insert_row  │  │ db://schema │           │
│  └─────────────┘  └─────────────┘           │
│  ┌─────────────────────────────────────┐    │
│  │  实际服务连接                          │    │
│  │  - 数据库                              │    │
│  │  - API服务                             │    │
│  │  - 外部系统                            │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 4.3 MCP传输协议

| 传输方式 | 用途 | 配置示例 |
|---------|------|---------|
| `stdio` | 本地进程 | `npx @modelcontextprotocol/server-github` |
| `http` | 远程服务 | `https://api.example.com/mcp` |
| `sse` | Server-Sent Events | 实时数据流 |

### 4.4 MCP作用域

```bash
# 用户级（全局）
claude mcp add -s user github -- npx @modelcontextprotocol/server-github

# 项目级（团队共享，git追踪）
claude mcp add -s project postgres -- npx @anthropic-ai/server-postgres

# 本地级（个人，gitignore）
claude mcp add -s local puppeteer -- npx @anthropic-ai/server-puppeteer
```

### 4.5 MCP限制与调优

- **工具描述**：每服务器2KB上限
- **结果大小**：默认有上限，可用`maxResultSizeChars`注解放宽到500K字符
- **输出Token**：`MAX_MCP_OUTPUT_TOKENS`环境变量控制

### 4.6 借鉴价值

1. **标准化接口**：任何服务都可以通过MCP暴露为工具
2. **传输灵活性**：支持本地进程和远程服务
3. **作用域隔离**：用户/项目/本地三级配置
4. **即插即用**：npm包形式分发，零配置使用

---

## 五、工具系统设计

### 5.1 工具发现与注册

Claude Code的工具系统采用**自动发现**机制：

```python
# 工具注册（参考Hermes实现）
registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(...),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

### 5.2 权限控制层级

```
CLI Flags (最高优先级)
    │
    ▼
Local Settings (.claude/settings.local.json)  # 个人，gitignore
    │
    ▼
Project Settings (.claude/settings.json)      # 团队，git追踪
    │
    ▼
User Settings (~/.claude/settings.json)       # 全局
```

### 5.3 权限配置示例

```json
{
  "permissions": {
    "allow": [
      "Bash(npm run lint:*)",    // 允许lint命令
      "WebSearch",                // 允许搜索
      "Read"                      // 允许读取
    ],
    "ask": [
      "Write(*.ts)",              // 写入TS文件需确认
      "Bash(git push*)"           // git push需确认
    ],
    "deny": [
      "Read(.env)",               // 禁止读取.env
      "Bash(rm -rf *)"            // 禁止删除
    ]
  }
}
```

### 5.4 工具名称语法

```
Read                    # 所有文件读取
Edit                    # 文件编辑（已有文件）
Write                   # 文件创建（新文件）
Bash                    # 所有shell命令
Bash(git *)             # 仅git命令
Bash(git commit *)      # 仅git commit
Bash(npm run lint:*)    # 通配符匹配
WebSearch               # 网页搜索
WebFetch                # 网页获取
mcp__<server>__<tool>   # 特定MCP工具
```

### 5.5 借鉴价值

1. **白名单/黑名单机制**：`allow/ask/deny`三级控制
2. **模式匹配**：支持通配符的细粒度权限
3. **层级覆盖**：CLI > 本地 > 项目 > 用户
4. **动态权限**：Shift+Tab可切换权限模式

---

## 六、子Agent系统

### 6.1 Agent定义格式

```markdown
---
name: security-reviewer
description: Security-focused code review
model: opus                    # 可指定模型
tools: [Read, Bash]            # 限制工具集
color: yellow                  # 显示颜色
---

You are a senior security engineer. Review code for:
- Injection vulnerabilities
- Authentication/authorization flaws
- Secrets in code
```

### 6.2 Agent位置优先级

1. `.claude/agents/` — 项目级，团队共享
2. `--agents` CLI标志 — 会话级，动态定义
3. `~/.claude/agents/` — 用户级，个人

### 6.3 多Agent协作模式

**Feature Development Plugin的7阶段工作流**：

```
Phase 1: Discovery（需求理解）
    │
    ▼
Phase 2: Codebase Exploration（代码探索）
    │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ │ code-explorer│ │ code-explorer│ │ code-explorer│
    │ │ 探索相似特性  │ │ 探索架构模式  │ │ 探索UI模式   │
    │ └─────────────┘ └─────────────┘ └─────────────┘
    │           ↓ 并行执行，汇总结果
    ▼
Phase 3: Clarifying Questions（澄清问题）
    │
    ▼
Phase 4: Architecture Design（架构设计）
    │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ │code-architect│ │code-architect│ │code-architect│
    │ │ 最小改动方案  │ │ 清洁架构方案  │ │ 务实平衡方案  │
    │ └─────────────┘ └─────────────┘ └─────────────┘
    │           ↓ 用户选择方案
    ▼
Phase 5: Implementation（实现）
    │
    ▼
Phase 6: Quality Review（质量审查）
    │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ │code-reviewer │ │code-reviewer │ │code-reviewer │
    │ │ 简洁性/DRY   │ │ Bug/正确性   │ │ 规范/抽象    │
    │ └─────────────┘ └─────────────┘ └─────────────┘
    │           ↓ 置信度评分，过滤误报
    ▼
Phase 7: Summary（总结）
```

### 6.4 Code Review Plugin的置信度评分

```markdown
置信度评分系统：
- 0: 不自信，误报
- 25: 有些自信，可能是真的
- 50: 中等自信，真实但次要
- 75: 高度自信，真实且重要
- 100: 绝对确定

阈值：≥80 才报告（过滤误报）
```

### 6.5 借鉴价值

1. **专业分工**：每个Agent专注特定领域
2. **并行执行**：多个Agent同时工作，提高效率
3. **置信度过滤**：避免误报干扰
4. **用户决策点**：关键步骤等待用户确认
5. **模板化定义**：Markdown + YAML frontmatter

---

## 七、上下文管理

### 7.1 系统提示层级

```
全局：~/.claude/CLAUDE.md              # 所有项目生效
    │
    ▼
项目：./CLAUDE.md                      # 项目特定（git追踪）
    │
    ▼
本地：.claude/CLAUDE.local.md          # 个人覆盖（gitignore）
    │
    ▼
规则目录：.claude/rules/*.md           # 模块化规则
```

### 7.2 上下文压缩策略

- **触发阈值**：上下文使用率达到50%时考虑压缩
- **目标比例**：压缩到20%
- **关键保留**：CLAUDE.md内容在压缩后保留
- **质量影响**：>70%使用率时精度开始下降，>85%时幻觉风险显著增加

### 7.3 自动记忆

Claude自动在`~/.claude/projects/<project>/memory/`存储项目级记忆：
- 限制：25KB或200行
- 跨会话累积
- 与CLAUDE.md分离

### 7.4 借鉴价值

1. **分层配置**：全局/项目/本地三级
2. **模块化规则**：rules目录替代单一文件
3. **主动压缩**：阈值触发的自动压缩
4. **持久记忆**：跨会话的知识累积

---

## 八、安全机制

### 8.1 权限模式

| 模式 | 行为 |
|-----|------|
| `default` | 默认，需确认危险操作 |
| `acceptEdits` | 自动接受文件编辑 |
| `plan` | 仅规划，不执行 |
| `auto` | 自动接受大部分操作 |
| `dontAsk` | 不询问，直接执行 |
| `bypassPermissions` | 完全跳过权限（危险） |

### 8.2 安全Hook示例

```json
{
  "PreToolUse": [{
    "matcher": "Bash",
    "hooks": [{
      "type": "command",
      "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qE 'rm -rf|git push.*--force|:(){ :|:& };:'; then echo 'Dangerous command blocked!' && exit 2; fi"
    }]
  }]
}
```

### 8.3 9种安全模式检测（security-guidance插件）

1. 命令注入
2. XSS攻击
3. eval使用
4. 危险HTML
5. pickle反序列化
6. os.system调用
7. 硬编码密钥
8. 不安全的正则
9. 路径遍历

### 8.4 借鉴价值

1. **多层防护**：权限模式 + Hook + 插件检测
2. **最小权限**：默认仅允许必要操作
3. **可组合**：不同模式可切换
4. **用户可控**：Shift+Tab快速切换

---

## 九、与Hermes Agent对比分析

| 维度 | Claude Code | Hermes Agent |
|------|-------------|--------------|
| **定位** | 专业编码Agent | 通用Agent框架 |
| **运行环境** | 终端TUI | 终端 + 多平台网关 |
| **Hook系统** | 8种事件，声明式配置 | 无内置Hook，通过脚本实现 |
| **MCP支持** | 原生支持 | 通过MCP工具集支持 |
| **子Agent** | 专业Agent定义 | delegate_task委托 |
| **上下文管理** | CLAUDE.md + 自动压缩 | Skills + Memory |
| **安全机制** | 权限模式 + Hook | TIRITH安全检查 |
| **会话管理** | 5小时TTL + 检查点 | SQLite持久化 |
| **工具发现** | 自动注册 | 手动定义 |
| **并行执行** | 多Agent并行 | delegate_task并行 |

### 9.1 Claude Code的优势

1. **Hook系统成熟**：事件驱动的自动化机制更完善
2. **MCP原生支持**：标准化的外部工具集成
3. **专业Agent设计**：针对编码任务优化
4. **权限控制精细**：白名单/黑名单/模式匹配
5. **上下文压缩智能**：阈值触发，质量感知

### 9.2 Hermes Agent的优势

1. **多平台支持**：10+消息平台集成
2. **Provider灵活**：20+模型提供商
3. **Memory持久化**：跨会话记忆
4. **Skills系统**：可复用的程序知识
5. **Cron调度**：内置定时任务

---

## 十、可借鉴的设计模式

### 10.1 Hook系统设计（强烈推荐）

**借鉴方式**：
1. 定义标准事件类型（PreToolUse、PostToolUse等）
2. JSON配置声明式Hook
3. 环境变量传递上下文
4. exit code控制流程

**实现建议**：
```python
# 在Hermes中实现类似Hook系统
HOOK_EVENTS = [
    "UserPromptSubmit",
    "PreToolUse", 
    "PostToolUse",
    "Notification",
    "Stop",
    "SubagentStop",
    "PreCompact",
    "SessionStart"
]

def execute_hook(event_type: str, context: dict) -> bool:
    """执行Hook，返回是否允许继续"""
    hooks = load_hooks_config().get(event_type, [])
    for hook in hooks:
        if matcher_matches(hook["matcher"], context):
            result = run_command(hook["command"], env={
                "CLAUDE_PROJECT_DIR": context["project_dir"],
                "CLAUDE_FILE_PATHS": ",".join(context.get("files", [])),
                "CLAUDE_TOOL_INPUT": json.dumps(context.get("tool_input", {}))
            })
            if result.exit_code == 2:  # 阻止执行
                return False
    return True
```

### 10.2 多Agent协作模式（推荐）

**借鉴方式**：
1. 专业Agent定义（Markdown + YAML frontmatter）
2. 并行执行 + 结果汇总
3. 置信度评分过滤
4. 用户决策点设计

**实现建议**：
```python
# Agent定义格式
AGENT_TEMPLATE = """
---
name: {name}
description: {description}
model: {model}
tools: {tools}
---
{system_prompt}
"""

# 并行执行多个Agent
async def run_parallel_agents(agents: List[Agent], task: str) -> List[AgentResult]:
    tasks = [agent.execute(task) for agent in agents]
    results = await asyncio.gather(*tasks)
    return results

# 置信度过滤
def filter_by_confidence(issues: List[Issue], threshold: int = 80) -> List[Issue]:
    return [i for i in issues if i.confidence >= threshold]
```

### 10.3 上下文管理策略（推荐）

**借鉴方式**：
1. 分层配置（全局/项目/本地）
2. 模块化规则（rules目录）
3. 阈值触发压缩
4. 关键内容保留

**实现建议**：
```python
# 上下文管理器
class ContextManager:
    def __init__(self):
        self.global_config = load_config("~/.claude/CLAUDE.md")
        self.project_config = load_config("./CLAUDE.md")
        self.local_config = load_config(".claude/CLAUDE.local.md")
        self.rules = load_rules(".claude/rules/*.md")
    
    def get_context(self) -> str:
        """按优先级合并配置"""
        context = []
        context.append(self.global_config)
        context.append(self.project_config)
        context.append(self.local_config)
        context.extend(self.rules)
        return "\n".join(context)
    
    def should_compress(self, usage_ratio: float) -> bool:
        """判断是否需要压缩"""
        return usage_ratio > 0.5
    
    def compress(self, content: str, target_ratio: float = 0.2) -> str:
        """压缩上下文，保留CLAUDE.md"""
        # 保留关键配置，压缩对话历史
        pass
```

### 10.4 工具权限控制（推荐）

**借鉴方式**：
1. 三级权限（allow/ask/deny）
2. 模式匹配（通配符）
3. 层级覆盖（CLI > 本地 > 项目 > 用户）

**实现建议**：
```python
# 权限检查器
class PermissionChecker:
    def __init__(self, config: dict):
        self.allow = config.get("allow", [])
        self.ask = config.get("ask", [])
        self.deny = config.get("deny", [])
    
    def check(self, tool_name: str, tool_input: str) -> str:
        """返回 'allow', 'ask', 或 'deny'"""
        # 1. 检查deny列表
        for pattern in self.deny:
            if self._match(pattern, tool_name, tool_input):
                return "deny"
        
        # 2. 检查ask列表
        for pattern in self.ask:
            if self._match(pattern, tool_name, tool_input):
                return "ask"
        
        # 3. 检查allow列表
        for pattern in self.allow:
            if self._match(pattern, tool_name, tool_input):
                return "allow"
        
        # 4. 默认ask
        return "ask"
    
    def _match(self, pattern: str, tool_name: str, tool_input: str) -> bool:
        """通配符匹配"""
        # Bash(git *) 格式处理
        # Write(*.py) 格式处理
        pass
```

---

## 十一、总结

### 11.1 Claude Code的核心设计理念

1. **事件驱动**：Hook系统实现松耦合的自动化
2. **标准协议**：MCP实现工具的标准化集成
3. **专业分工**：子Agent系统实现任务分解
4. **安全优先**：多层权限控制
5. **上下文感知**：智能的上下文管理

### 11.2 最值得借鉴的3个设计

1. **Hook系统**：事件驱动的自动化机制，可显著提升Agent的灵活性
2. **多Agent协作**：专业Agent并行工作，提升复杂任务处理能力
3. **上下文分层**：全局/项目/本地三级配置，适应不同场景

### 11.3 实施建议

**短期（1-2周）**：
- 实现基础Hook系统（PreToolUse、PostToolUse）
- 添加工具权限控制（allow/ask/deny）
- 实现CLAUDE.md式的项目配置

**中期（1-2月）**：
- 完善8种Hook事件
- 实现MCP客户端
- 添加子Agent系统

**长期（3-6月）**：
- 构建Agent市场
- 实现智能上下文压缩
- 添加自动记忆系统

---

**调研完成** ✓
