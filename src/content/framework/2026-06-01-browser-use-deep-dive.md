---
title: "Browser-Use深度解析：开源浏览器自动化Agent框架的架构与实战"
description: "深入解析Browser-Use框架的核心架构——Playwright驱动、视觉理解、DOM抽象三大支柱，结合生产级部署经验，打造能真正操作网页的AI Agent"
date: 2026-06-01
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["Browser-Use", "浏览器自动化", "AI Agent", "Playwright", "Web Agent", "网页操作"]
draft: false
---

## 引言：为什么AI Agent需要"看见"网页

传统的LLM应用通过API与外部世界交互，但互联网上80%的信息只能通过网页获取。Browser-Use的核心理念是：**让AI Agent像人类一样操作浏览器**——打开网页、阅读内容、点击按钮、填写表单、提交数据。

与传统的Selenium/Playwright脚本不同，Browser-Use不需要预定义DOM选择器，而是通过**视觉理解**和**语义分析**来理解页面，实现了真正的"零脚本"浏览器自动化。

```
传统方案：                          Browser-Use：
                                    
用户 → 编写选择器脚本 → 执行        用户 → 自然语言指令 → AI理解 → 操作网页
      ↓                                ↓
   维护成本高                       自适应，无需维护
   页面变化就挂                     理解页面语义
   无法处理动态内容                 动态适应变化
```

本文将深入解析Browser-Use的架构设计、核心组件、生产部署经验和性能优化策略。

---

## 一、Browser-Use架构全景

### 1.1 整体架构

Browser-Use采用分层架构设计，将浏览器操作抽象为可组合的技能层：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户指令层 (User Intent)                   │
│        "帮我在这个电商网站搜索iPhone并加入购物车"              │
├─────────────────────────────────────────────────────────────┤
│                    规划层 (Planning Layer)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 任务分解  │→│ 步骤规划  │→│ 错误恢复  │→│ 进度追踪  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    理解层 (Perception Layer)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 视觉理解  │  │ DOM分析   │  │ 文本提取  │  │ 页面状态  │  │
│  │ (VLM)    │  │ (HTML)   │  │ (OCR)    │  │ 检测     │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    执行层 (Action Layer)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 点击      │  │ 输入      │  │ 滚动      │  │ 导航      │  │
│  │ Click    │  │ Type     │  │ Scroll   │  │ Navigate │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    浏览器驱动层 (Browser Driver)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Playwright / Chromium                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件关系

```python
# Browser-Use核心类关系（简化）
from dataclasses import dataclass
from typing import Optional, List, Callable
from enum import Enum

class ActionType(Enum):
    """浏览器操作类型"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    SELECT = "select"
    HOVER = "hover"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    TAB_SWITCH = "tab_switch"

@dataclass
class BrowserAction:
    """浏览器操作定义"""
    action_type: ActionType
    target: str          # 目标描述（自然语言或CSS选择器）
    value: Optional[str] = None  # 输入值
    coordinates: Optional[tuple[int, int]] = None  # 坐标（视觉操作时使用）

@dataclass
class PageState:
    """页面状态快照"""
    url: str
    title: str
    content: str           # 页面文本内容
    screenshot: bytes      # 页面截图
    interactive_elements: list  # 可交互元素列表
    dom_summary: str       # DOM结构摘要

class BrowserAgent:
    """Browser-Use核心Agent"""
    
    def __init__(self, llm, browser_config=None):
        self.llm = llm
        self.browser = None  # Playwright browser instance
        self.page = None     # 当前页面
        self.history = []    # 操作历史
        self.config = browser_config or BrowserConfig()
    
    async def run(self, task: str) -> dict:
        """执行浏览器任务"""
        # 1. 初始化浏览器
        await self._init_browser()
        
        # 2. 任务规划
        plan = await self._plan_task(task)
        
        # 3. 逐步执行
        for step in plan.steps:
            # 获取当前页面状态
            state = await self._get_page_state()
            
            # 决定下一步操作
            action = await self._decide_action(task, state, step)
            
            # 执行操作
            result = await self._execute_action(action)
            
            # 记录历史
            self.history.append({
                "step": step,
                "action": action,
                "result": result,
                "state": state
            })
            
            # 错误处理
            if result.error:
                recovery = await self._handle_error(result.error, state)
                if not recovery.success:
                    return {"success": False, "error": result.error}
        
        return {"success": True, "history": self.history}
```

---

## 二、视觉理解：让AI"看见"网页

### 2.1 截图+VLM的视觉理解管线

Browser-Use最强大的能力是通过视觉理解来操作网页，而不仅仅依赖DOM：

```python
class VisualPerceiver:
    """视觉感知器：通过截图理解页面"""
    
    def __init__(self, vlm_client):
        self.vlm = vlm_client  # 视觉语言模型（如GPT-4V, Qwen-VL）
    
    async def analyze_screenshot(
        self, 
        screenshot: bytes, 
        task: str,
        action_history: list = None
    ) -> dict:
        """
        分析页面截图，理解页面内容和可用操作
        
        返回结构:
        {
            "page_description": "页面的整体描述",
            "relevant_elements": [
                {
                    "description": "搜索框",
                    "position": [x, y, width, height],
                    "actionable": True,
                    "suggested_action": "click_and_type"
                },
                ...
            ],
            "recommended_next_action": "在搜索框中输入关键词",
            "confidence": 0.85
        }
        """
        prompt = f"""你是一个网页分析专家。请仔细分析这个网页截图。

当前任务：{task}

{f"操作历史：{json.dumps(action_history[-3:], ensure_ascii=False)}" if action_history else ""}

请分析：
1. 这个页面是什么？（类型、主要功能）
2. 页面上有哪些可交互的元素？（按钮、输入框、链接等）
3. 这些元素的位置（用像素坐标表示，左上角为原点）
4. 为了完成任务，下一步应该做什么？

请用JSON格式回答：
{{
    "page_description": "页面描述",
    "page_type": "搜索页|列表页|详情页|表单页|...",
    "relevant_elements": [
        {{
            "id": "element_1",
            "description": "元素描述",
            "position": [x, y, width, height],
            "element_type": "button|input|link|image|...",
            "actionable": true,
            "suggested_action": "推荐的操作"
        }}
    ],
    "recommended_next_action": "下一步操作建议",
    "confidence": 0.0-1.0,
    "potential_issues": ["可能的问题列表"]
}}
"""
        
        response = await self.vlm.analyze(
            image=screenshot,
            prompt=prompt,
            temperature=0.1  # 低温度，确保稳定性
        )
        
        return json.loads(response)
    
    async def locate_element(
        self,
        screenshot: bytes,
        element_description: str
    ) -> dict:
        """
        精确定位页面元素
        
        用于需要精确点击的场景
        """
        prompt = f"""请在截图中找到以下元素的精确位置。

目标元素：{element_description}

返回该元素的中心点坐标（像素值）：
{{"center_x": 像素x坐标, "center_y": 像素y坐标, "confidence": 置信度}}
"""
        response = await self.vlm.analyze(
            image=screenshot,
            prompt=prompt,
            temperature=0.0
        )
        return json.loads(response)
```

### 2.2 视觉理解的优化策略

视觉理解是Browser-Use最耗时的环节，以下是优化策略：

```python
class VisualOptimizer:
    """视觉理解优化器"""
    
    def __init__(self):
        self.screenshot_cache = {}
        self.element_registry = {}  # 已识别元素缓存
    
    async def optimized_analyze(
        self,
        page_state: PageState,
        task: str
    ) -> dict:
        """
        优化的视觉分析策略
        
        策略1：增量分析 - 只分析变化的部分
        策略2：分辨率优化 - 降低截图质量加速处理
        策略3：元素缓存 - 重复出现的元素复用识别结果
        """
        
        # 策略1：检测页面变化
        page_hash = hashlib.md5(page_state.screenshot).hexdigest()
        if page_hash in self.screenshot_cache:
            # 页面未变化，复用上次分析
            return self.screenshot_cache[page_hash]
        
        # 策略2：优化截图质量
        optimized_screenshot = self._optimize_screenshot(
            page_state.screenshot,
            quality=75,  # 降低质量换取速度
            max_width=1280  # 限制最大宽度
        )
        
        # 策略3：分区域分析
        # 如果页面很长，只分析可视区域
        visible_area = self._crop_visible_area(
            optimized_screenshot,
            viewport_height=720
        )
        
        # 执行分析
        analysis = await self._analyze_region(visible_area, task)
        
        # 缓存结果
        self.screenshot_cache[page_hash] = analysis
        
        return analysis
    
    def _optimize_screenshot(
        self,
        screenshot: bytes,
        quality: int = 75,
        max_width: int = 1280
    ) -> bytes:
        """优化截图质量"""
        from PIL import Image
        import io
        
        img = Image.open(io.BytesIO(screenshot))
        
        # 限制宽度
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize(
                (max_width, int(img.height * ratio)),
                Image.Resampling.LANCZOS
            )
        
        # 压缩质量
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()
    
    def _crop_visible_area(
        self,
        screenshot: bytes,
        viewport_height: int = 720
    ) -> bytes:
        """裁剪可视区域"""
        from PIL import Image
        import io
        
        img = Image.open(io.BytesIO(screenshot))
        cropped = img.crop((0, 0, img.width, viewport_height))
        
        buffer = io.BytesIO()
        cropped.save(buffer, format='JPEG')
        return buffer.getvalue()
```

---

## 三、DOM抽象：语义化的页面理解

### 3.1 DOM树的智能压缩

直接处理完整的DOM树既低效又容易出错，Browser-Use通过智能压缩来提取关键信息：

```python
class DOMProcessor:
    """DOM处理器：将复杂DOM树压缩为语义化摘要"""
    
    # 可交互元素的标签和属性
    INTERACTIVE_TAGS = {
        "a", "button", "input", "select", "textarea",
        "details", "summary", "label"
    }
    
    INTERACTIVE_ROLES = {
        "button", "link", "searchbox", "textbox",
        "combobox", "checkbox", "radio", "tab"
    }
    
    async def process(self, page) -> str:
        """
        处理页面DOM，生成语义化摘要
        
        返回的摘要包含：
        - 页面结构概览
        - 所有可交互元素及其描述
        - 元素之间的层级关系
        """
        # 注入JavaScript提取关键信息
        js_code = """
        () => {
            const elements = [];
            
            // 遍历所有可交互元素
            const interactiveElements = document.querySelectorAll(
                'a, button, input, select, textarea, [role="button"], [role="link"], [onclick]'
            );
            
            interactiveElements.forEach((el, index) => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                
                // 过滤不可见元素
                if (style.display === 'none' || 
                    style.visibility === 'hidden' ||
                    rect.width === 0 || 
                    rect.height === 0) {
                    return;
                }
                
                // 提取元素信息
                elements.push({
                    index: index,
                    tag: el.tagName.toLowerCase(),
                    type: el.type || '',
                    text: (el.textContent || '').trim().substring(0, 100),
                    placeholder: el.placeholder || '',
                    ariaLabel: el.getAttribute('aria-label') || '',
                    href: el.href || '',
                    role: el.getAttribute('role') || '',
                    position: {
                        x: Math.round(rect.x),
                        y: Math.round(rect.y),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    },
                    visible: rect.top < window.innerHeight && rect.bottom > 0
                });
            });
            
            return JSON.stringify(elements);
        }
        """
        
        raw_elements = await page.evaluate(js_code)
        elements = json.loads(raw_elements)
        
        # 生成语义化摘要
        return self._generate_summary(elements)
    
    def _generate_summary(self, elements: list) -> str:
        """生成DOM语义摘要"""
        lines = ["=== 页面可交互元素 ===\n"]
        
        # 按类型分组
        groups = {
            "links": [],
            "buttons": [],
            "inputs": [],
            "others": []
        }
        
        for el in elements:
            tag = el["tag"]
            if tag == "a":
                groups["links"].append(el)
            elif tag == "button" or el["role"] == "button":
                groups["buttons"].append(el)
            elif tag in ("input", "select", "textarea"):
                groups["inputs"].append(el)
            else:
                groups["others"].append(el)
        
        # 链接
        if groups["links"]:
            lines.append("【链接】")
            for el in groups["links"][:10]:  # 限制数量
                text = el["text"] or el["href"][:50]
                lines.append(f"  [{el['index']}] {text}")
        
        # 按钮
        if groups["buttons"]:
            lines.append("\n【按钮】")
            for el in groups["buttons"][:10]:
                text = el["text"] or el["ariaLabel"] or "未命名按钮"
                lines.append(f"  [{el['index']}] {text}")
        
        # 输入框
        if groups["inputs"]:
            lines.append("\n【输入框】")
            for el in groups["inputs"]:
                label = el["placeholder"] or el["ariaLabel"] or el["text"] or "输入框"
                input_type = el["type"] if el["type"] != "text" else ""
                type_info = f" ({input_type})" if input_type else ""
                lines.append(f"  [{el['index']}] {label}{type_info}")
        
        return "\n".join(lines)
```

### 3.2 智能元素匹配

```python
class ElementMatcher:
    """智能元素匹配器：将自然语言描述映射到DOM元素"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def match(
        self,
        element_description: str,
        dom_summary: str,
        page_state: PageState
    ) -> dict:
        """
        匹配自然语言描述到具体的DOM元素
        
        参数:
            element_description: 如"搜索框"、"提交按钮"
            dom_summary: DOM语义摘要
            page_state: 当前页面状态
        
        返回:
            {"element_index": int, "confidence": float, "action_type": str}
        """
        prompt = f"""你是一个网页元素匹配专家。

用户想要操作的元素：{element_description}

当前页面的可交互元素：
{dom_summary}

请找到最匹配的元素，并确定操作类型。

返回JSON格式：
{{
    "element_index": 匹配元素的索引号,
    "confidence": 匹配置信度 0.0-1.0,
    "action_type": "click|type|select|hover",
    "reason": "匹配理由"
}}
"""
        response = await self.llm.generate(prompt, temperature=0.0)
        return json.loads(response)
    
    async def fuzzy_match(
        self,
        description: str,
        elements: list,
        top_k: int = 3
    ) -> list[dict]:
        """
        模糊匹配：当精确匹配失败时的备选方案
        
        使用语义相似度进行模糊匹配
        """
        # 构建元素描述列表
        element_descriptions = []
        for el in elements:
            desc = f"{el['tag']}"
            if el["text"]:
                desc += f" 文本:'{el['text']}'"
            if el["placeholder"]:
                desc += f" 占位符:'{el['placeholder']}'"
            if el["ariaLabel"]:
                desc += f" 标签:'{el['ariaLabel']}'"
            element_descriptions.append(desc)
        
        # 用LLM进行语义匹配
        prompt = f"""用户要找的元素：{description}

以下是页面上的元素列表：
{json.dumps(element_descriptions, ensure_ascii=False, indent=2)}

请返回与用户描述最相关的{top_k}个元素，按匹配度排序。

返回JSON格式：
{{
    "matches": [
        {{"index": 元素索引, "score": 匹配分数 0-1, "reason": "匹配理由"}},
        ...
    ]
}}
"""
        response = await self.llm.generate(prompt, temperature=0.0)
        result = json.loads(response)
        return result["matches"]
```

---

## 四、任务规划：从自然语言到操作序列

### 4.1 任务分解与规划

```python
class TaskPlanner:
    """任务规划器：将复杂任务分解为可执行的操作序列"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def plan(self, task: str, initial_state: PageState) -> list[dict]:
        """
        将复杂任务分解为操作步骤
        
        例如：
        输入："在京东搜索iPhone 16，找到最便宜的，加入购物车"
        输出：
        [
            {"action": "navigate", "target": "jd.com", "reason": "打开京东"},
            {"action": "locate", "target": "搜索框", "reason": "找到搜索入口"},
            {"action": "type", "target": "搜索框", "value": "iPhone 16", "reason": "输入搜索关键词"},
            {"action": "click", "target": "搜索按钮", "reason": "执行搜索"},
            {"action": "locate", "target": "排序选项", "reason": "准备按价格排序"},
            {"action": "click", "target": "价格从低到高", "reason": "排序找到最便宜的"},
            {"action": "locate", "target": "第一个商品", "reason": "选择最便宜的商品"},
            {"action": "click", "target": "第一个商品", "reason": "进入商品详情"},
            {"action": "locate", "target": "加入购物车按钮", "reason": "准备加入购物车"},
            {"action": "click", "target": "加入购物车", "reason": "完成加入购物车"}
        ]
        """
        prompt = f"""你是一个浏览器任务规划专家。请将用户的任务分解为具体的操作步骤。

用户任务：{task}

当前页面状态：
- URL: {initial_state.url}
- 标题: {initial_state.title}

要求：
1. 每个步骤必须是原子操作（一个步骤只做一件事）
2. 步骤之间的依赖关系要清晰
3. 考虑可能的错误情况和恢复策略
4. 如果任务可以并行处理，标注出来

返回JSON格式的步骤列表：
{{
    "steps": [
        {{
            "id": 1,
            "action": "操作类型",
            "target": "操作目标",
            "value": "输入值（如有）",
            "reason": "执行此步骤的原因",
            "precondition": "前置条件（如有）",
            "fallback": "失败时的备选方案（如有）"
        }}
    ],
    "estimated_steps": 预估总步骤数,
    "complexity": "low|medium|high",
    "risks": ["潜在风险列表"]
}}
"""
        response = await self.llm.generate(prompt, temperature=0.2)
        result = json.loads(response)
        return result["steps"]
    
    async def replan(
        self,
        original_task: str,
        history: list,
        current_state: PageState,
        error: str = None
    ) -> list[dict]:
        """
        重新规划：当执行遇到问题时动态调整计划
        """
        prompt = f"""任务执行过程中遇到了问题，需要重新规划。

原始任务：{original_task}

已完成的步骤：
{json.dumps(history, ensure_ascii=False, indent=2)}

{f"错误信息：{error}" if error else ""}

当前页面状态：
- URL: {current_state.url}
- 标题: {current_state.title}

请根据当前情况，生成剩余的执行步骤。如果原始计划需要调整，请说明原因。

返回JSON格式：
{{
    "analysis": "对当前情况的分析",
    "adjusted_plan": true/false,
    "steps": [剩余步骤列表]
}}
"""
        response = await self.llm.generate(prompt, temperature=0.2)
        result = json.loads(response)
        return result["steps"]
```

### 4.2 操作执行引擎

```python
class ActionExecutor:
    """操作执行引擎：将规划的操作转化为实际的浏览器操作"""
    
    def __init__(self, page, visual_perceiver, element_matcher):
        self.page = page
        self.visual = visual_perceiver
        self.matcher = element_matcher
    
    async def execute(self, action: dict, page_state: PageState) -> dict:
        """
        执行单个操作
        
        返回:
        {"success": bool, "error": str|None, "new_state": PageState}
        """
        try:
            action_type = action["action"]
            
            if action_type == "navigate":
                return await self._navigate(action["target"])
            
            elif action_type == "locate":
                # 先用DOM匹配
                match = await self.matcher.match(
                    action["target"],
                    page_state.dom_summary,
                    page_state
                )
                
                if match["confidence"] < 0.5:
                    # DOM匹配失败，尝试视觉匹配
                    visual_result = await self.visual.locate_element(
                        page_state.screenshot,
                        action["target"]
                    )
                    if visual_result["confidence"] > 0.7:
                        # 用坐标点击
                        return await self._click_at(
                            visual_result["center_x"],
                            visual_result["center_y"]
                        )
                
                return match
            
            elif action_type == "click":
                return await self._click(action["target"], page_state)
            
            elif action_type == "type":
                return await self._type(
                    action["target"],
                    action.get("value", ""),
                    page_state
                )
            
            elif action_type == "scroll":
                return await self._scroll(action.get("direction", "down"))
            
            elif action_type == "select":
                return await self._select(
                    action["target"],
                    action.get("value", ""),
                    page_state
                )
            
            else:
                return {"success": False, "error": f"Unknown action: {action_type}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _click(self, target: str, page_state: PageState) -> dict:
        """点击操作"""
        # DOM匹配
        match = await self.matcher.match(target, page_state.dom_summary, page_state)
        
        if match["confidence"] > 0.7:
            # 通过索引找到元素并点击
            js = f"""
            () => {{
                const elements = document.querySelectorAll(
                    'a, button, input, select, [role="button"], [onclick]'
                );
                const visibleElements = Array.from(elements).filter(el => {{
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                }});
                if (visibleElements[{match['element_index']}]) {{
                    visibleElements[{match['element_index']}].click();
                    return true;
                }}
                return false;
            }}
            """
            result = await self.page.evaluate(js)
            return {"success": result, "error": None if result else "Element not found"}
        
        # 视觉匹配作为备选
        visual = await self.visual.locate_element(
            page_state.screenshot, target
        )
        if visual["confidence"] > 0.7:
            return await self._click_at(visual["center_x"], visual["center_y"])
        
        return {"success": False, "error": f"Could not locate element: {target}"}
    
    async def _type(self, target: str, value: str, page_state: PageState) -> dict:
        """输入操作"""
        # 先点击目标元素
        click_result = await self._click(target, page_state)
        if not click_result["success"]:
            return click_result
        
        # 清空现有内容
        await self.page.keyboard.press("Control+a")
        await self.page.keyboard.press("Backspace")
        
        # 逐字输入（模拟人类输入）
        await self.page.keyboard.type(value, delay=50)
        
        return {"success": True, "error": None}
    
    async def _scroll(self, direction: str, amount: int = 500) -> dict:
        """滚动操作"""
        delta = amount if direction == "down" else -amount
        await self.page.mouse.wheel(0, delta)
        await asyncio.sleep(0.5)  # 等待页面响应
        return {"success": True, "error": None}
    
    async def _click_at(self, x: int, y: int) -> dict:
        """坐标点击"""
        await self.page.mouse.click(x, y)
        return {"success": True, "error": None}
```

---

## 五、错误处理与恢复

### 5.1 智能错误恢复

浏览器自动化中的错误是不可避免的，Browser-Use需要智能的错误恢复机制：

```python
class ErrorRecovery:
    """智能错误恢复器"""
    
    # 常见错误模式和恢复策略
    ERROR_PATTERNS = {
        "element_not_found": {
            "description": "找不到目标元素",
            "strategies": [
                "scroll_and_retry",      # 滚动后重试
                "wait_and_retry",        # 等待后重试
                "alternative_locator",   # 使用备选定位器
                "visual_fallback",       # 切换到视觉定位
            ]
        },
        "element_not_interactable": {
            "description": "元素不可交互",
            "strategies": [
                "wait_for_element",      # 等待元素可交互
                "scroll_into_view",      # 滚动到元素可见
                "remove_overlay",        # 移除遮挡层
                "javascript_click",      # 用JS强制点击
            ]
        },
        "page_load_timeout": {
            "description": "页面加载超时",
            "strategies": [
                "refresh_and_retry",     # 刷新后重试
                "wait_longer",           # 增加等待时间
                "check_network",         # 检查网络状况
            ]
        },
        "captcha_detected": {
            "description": "检测到验证码",
            "strategies": [
                "notify_user",           # 通知用户手动处理
                "wait_for_solution",     # 等待验证码解决
            ]
        },
        "session_expired": {
            "description": "会话过期",
            "strategies": [
                "re_login",              # 重新登录
                "restore_session",       # 恢复会话
            ]
        }
    }
    
    def __init__(self, llm):
        self.llm = llm
        self.max_retries = 3
        self.retry_history = []
    
    async def handle_error(
        self,
        error: Exception,
        page_state: PageState,
        action_history: list
    ) -> dict:
        """
        智能错误处理
        
        返回:
        {
            "recovered": bool,
            "strategy_used": str,
            "recovery_action": dict|None,
            "message": str
        }
        """
        error_type = self._classify_error(error)
        
        # 检查重试次数
        if len(self.retry_history) >= self.max_retries:
            return {
                "recovered": False,
                "strategy_used": "max_retries_exceeded",
                "recovery_action": None,
                "message": f"已达到最大重试次数 ({self.max_retries})"
            }
        
        # 获取恢复策略
        pattern = self.ERROR_PATTERNS.get(error_type, {})
        strategies = pattern.get("strategies", [])
        
        # 让LLM选择最佳恢复策略
        strategy = await self._select_strategy(
            error_type, error, page_state, action_history, strategies
        )
        
        # 执行恢复策略
        recovery_action = await self._execute_strategy(strategy, page_state)
        
        self.retry_history.append({
            "error_type": error_type,
            "strategy": strategy,
            "success": recovery_action is not None
        })
        
        return {
            "recovered": recovery_action is not None,
            "strategy_used": strategy,
            "recovery_action": recovery_action,
            "message": f"使用策略 '{strategy}' 尝试恢复"
        }
    
    def _classify_error(self, error: Exception) -> str:
        """错误分类"""
        error_msg = str(error).lower()
        
        if "element" in error_msg and "not found" in error_msg:
            return "element_not_found"
        elif "element" in error_msg and ("interact" in error_msg or "click" in error_msg):
            return "element_not_interactable"
        elif "timeout" in error_msg:
            return "page_load_timeout"
        elif "captcha" in error_msg or "verify" in error_msg:
            return "captcha_detected"
        elif "session" in error_msg or "login" in error_msg:
            return "session_expired"
        else:
            return "unknown"
    
    async def _select_strategy(
        self,
        error_type: str,
        error: Exception,
        page_state: PageState,
        history: list,
        available_strategies: list
    ) -> str:
        """用LLM选择最佳恢复策略"""
        prompt = f"""浏览器操作遇到了错误，请选择最佳恢复策略。

错误类型：{error_type}
错误信息：{str(error)}

可用策略：{json.dumps(available_strategies)}

当前页面URL：{page_state.url}

已尝试的策略（避免重复使用）：
{json.dumps([h['strategy'] for h in self.retry_history])}

请返回推荐的策略名称（必须是可用策略之一）：
"""
        response = await self.llm.generate(prompt, temperature=0.0)
        strategy = response.strip().strip('"')
        
        # 确保返回的策略在可用列表中
        if strategy not in available_strategies:
            strategy = available_strategies[0] if available_strategies else "wait_and_retry"
        
        return strategy
    
    async def _execute_strategy(
        self,
        strategy: str,
        page_state: PageState
    ) -> dict | None:
        """执行恢复策略"""
        if strategy == "scroll_and_retry":
            await self.page.mouse.wheel(0, 300)
            await asyncio.sleep(1)
            return {"action": "scroll_and_retry"}
        
        elif strategy == "wait_and_retry":
            await asyncio.sleep(2)
            return {"action": "wait_and_retry"}
        
        elif strategy == "refresh_and_retry":
            await self.page.reload()
            await asyncio.sleep(3)
            return {"action": "refresh_and_retry"}
        
        elif strategy == "visual_fallback":
            return {"action": "switch_to_visual_mode"}
        
        elif strategy == "notify_user":
            return {"action": "pause_for_user_input", "reason": "captcha_detected"}
        
        return None
```

---

## 六、生产部署实战

### 6.1 配置管理

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class BrowserConfig:
    """浏览器配置"""
    # 浏览器设置
    headless: bool = True  # 无头模式
    slow_mo: int = 100     # 操作间隔（毫秒），模拟人类操作速度
    
    # 代理设置
    proxy: Optional[str] = None
    user_agent: Optional[str] = None
    
    # 超时设置
    navigation_timeout: int = 30000      # 页面导航超时（毫秒）
    action_timeout: int = 10000          # 操作超时
    page_load_timeout: int = 60000       # 页面加载超时
    
    # 视觉设置
    viewport_width: int = 1280
    viewport_height: int = 720
    screenshot_quality: int = 75         # JPEG质量
    
    # 安全设置
    max_actions_per_task: int = 50       # 单任务最大操作数
    max_retries: int = 3                 # 最大重试次数
    blocked_domains: List[str] = field(default_factory=list)  # 禁止访问的域名
    
    # 存储设置
    user_data_dir: Optional[str] = None  # 用户数据目录（保持登录状态）
    downloads_dir: str = "./downloads"
    
    # 代理认证
    proxy_auth: Optional[dict] = None  # {"username": "", "password": ""}

@dataclass  
class AgentConfig:
    """Agent配置"""
    # LLM设置
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    
    # 规划设置
    planning_enabled: bool = True
    replan_on_error: bool = True
    max_planning_retries: int = 3
    
    # 视觉设置
    vision_enabled: bool = True
    vision_model: str = "gpt-4o"  # 视觉理解模型
    
    # 日志设置
    verbose: bool = False
    save_screenshots: bool = False  # 保存每步截图（调试用）
    log_file: Optional[str] = None
```

### 6.2 并发任务管理

```python
class BrowserPool:
    """浏览器实例池：支持并发任务执行"""
    
    def __init__(self, config: BrowserConfig, pool_size: int = 3):
        self.config = config
        self.pool_size = pool_size
        self.available = asyncio.Queue()
        self.in_use = {}
    
    async def initialize(self):
        """初始化浏览器池"""
        from playwright.async_api import async_playwright
        
        self.playwright = await async_playwright().start()
        
        for i in range(self.pool_size):
            browser = await self.playwright.chromium.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo
            )
            await self.available.put({"id": i, "browser": browser})
    
    async def acquire(self) -> dict:
        """获取一个浏览器实例"""
        return await self.available.get()
    
    async def release(self, instance: dict):
        """释放浏览器实例"""
        # 清理状态
        pages = instance["browser"].pages
        for page in pages:
            if not page.url.startswith("about:"):
                await page.close()
        
        await self.available.put(instance)
    
    async def shutdown(self):
        """关闭所有浏览器实例"""
        while not self.available.empty():
            instance = await self.available.get()
            await instance["browser"].close()
        await self.playwright.stop()

class TaskRunner:
    """并发任务执行器"""
    
    def __init__(self, browser_pool: BrowserPool, llm):
        self.pool = browser_pool
        self.llm = llm
    
    async def run_task(self, task: str, task_id: str) -> dict:
        """执行单个任务"""
        instance = await self.pool.acquire()
        
        try:
            agent = BrowserAgent(
                llm=self.llm,
                browser=instance["browser"],
                config=AgentConfig()
            )
            
            result = await agent.run(task)
            result["task_id"] = task_id
            
            return result
        
        finally:
            await self.pool.release(instance)
    
    async def run_tasks(self, tasks: list[dict]) -> list[dict]:
        """
        并发执行多个任务
        
        tasks: [{"id": "task_1", "task": "..."}, ...]
        """
        semaphore = asyncio.Semaphore(self.pool.pool_size)
        
        async def limited_run(task_info):
            async with semaphore:
                return await self.run_task(task_info["task"], task_info["id"])
        
        results = await asyncio.gather(
            *[limited_run(t) for t in tasks],
            return_exceptions=True
        )
        
        return results
```

### 6.3 监控与可观测性

```python
import time
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class StepMetrics:
    """单步操作指标"""
    step_id: int
    action_type: str
    target: str
    start_time: float
    end_time: float = 0
    success: bool = False
    error: str = None
    screenshot_path: str = None
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

@dataclass
class TaskMetrics:
    """任务级指标"""
    task_id: str
    task_description: str
    start_time: float
    end_time: float = 0
    steps: List[StepMetrics] = field(default_factory=list)
    total_screenshots: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    
    @property
    def total_duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0
        return sum(1 for s in self.steps if s.success) / len(self.steps)
    
    def to_report(self) -> dict:
        return {
            "task_id": self.task_id,
            "duration_ms": self.total_duration_ms,
            "total_steps": len(self.steps),
            "success_rate": self.success_rate,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "slowest_step": max(
                self.steps, 
                key=lambda s: s.duration_ms
            ) if self.steps else None,
            "errors": [s for s in self.steps if s.error]
        }

class BrowserMonitor:
    """浏览器操作监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, TaskMetrics] = {}
    
    def start_task(self, task_id: str, description: str) -> TaskMetrics:
        """开始监控一个任务"""
        metrics = TaskMetrics(
            task_id=task_id,
            task_description=description,
            start_time=time.time()
        )
        self.metrics[task_id] = metrics
        return metrics
    
    def record_step(
        self,
        task_id: str,
        step_id: int,
        action_type: str,
        target: str,
        success: bool,
        error: str = None
    ):
        """记录一步操作"""
        step = StepMetrics(
            step_id=step_id,
            action_type=action_type,
            target=target,
            start_time=time.time(),
            end_time=time.time(),
            success=success,
            error=error
        )
        self.metrics[task_id].steps.append(step)
    
    def end_task(self, task_id: str):
        """结束任务监控"""
        self.metrics[task_id].end_time = time.time()
    
    def get_summary(self) -> dict:
        """获取所有任务的汇总"""
        total_tasks = len(self.metrics)
        successful = sum(
            1 for m in self.metrics.values() 
            if m.steps and all(s.success for s in m.steps)
        )
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful,
            "success_rate": successful / max(total_tasks, 1),
            "avg_duration_ms": sum(
                m.total_duration_ms for m in self.metrics.values()
            ) / max(total_tasks, 1),
            "total_llm_calls": sum(
                m.llm_calls for m in self.metrics.values()
            ),
            "total_tokens": sum(
                m.llm_tokens for m in self.metrics.values()
            )
        }
```

---

## 七、最佳实践与性能优化

### 7.1 性能优化策略

| 策略 | 效果 | 实现复杂度 |
|------|------|------------|
| 截图缓存 | 减少30%视觉调用 | 低 |
| DOM增量更新 | 减少50%DOM处理时间 | 中 |
| 并发浏览器实例 | 3-5倍吞吐量提升 | 中 |
| 操作合并 | 减少20%操作次数 | 高 |
| 预测性等待 | 减少40%等待时间 | 高 |

### 7.2 稳定性保障

```python
class StabilityGuard:
    """稳定性保障：防止无限循环和资源耗尽"""
    
    def __init__(self, config: BrowserConfig):
        self.config = config
        self.action_count = 0
        self.page_url_history = []
        self.stuck_counter = 0
    
    def check_action_limit(self) -> bool:
        """检查是否超过操作次数限制"""
        self.action_count += 1
        return self.action_count <= self.config.max_actions_per_task
    
    def check_page_stuck(self, current_url: str) -> bool:
        """检测页面是否卡住（URL长时间不变）"""
        self.page_url_history.append(current_url)
        
        # 检查最近5次URL是否相同
        if len(self.page_url_history) >= 5:
            recent = self.page_url_history[-5:]
            if len(set(recent)) == 1:
                self.stuck_counter += 1
                if self.stuck_counter >= 3:
                    return True  # 页面卡住了
            else:
                self.stuck_counter = 0
        
        return False
    
    def check_domain_block(self, url: str) -> bool:
        """检查是否访问了禁止的域名"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain in self.config.blocked_domains
```

---

## 八、与主流方案对比

| 特性 | Browser-Use | Playwright直接使用 | Selenium | WebVoyager |
|------|-------------|-------------------|----------|------------|
| 自然语言驱动 | ✅ | ❌ | ❌ | ✅ |
| 视觉理解 | ✅ | ❌ | ❌ | ✅ |
| DOM智能分析 | ✅ | 手动 | 手动 | 有限 |
| 错误自动恢复 | ✅ | 手动 | 手动 | 有限 |
| 多浏览器支持 | Chromium | 多 | 多 | Chromium |
| 学习曲线 | 低 | 中 | 中 | 低 |
| 生产稳定性 | 中 | 高 | 高 | 低 |
| 社区活跃度 | 高 | 高 | 中 | 低 |

---

## 总结

Browser-Use代表了AI Agent操作网页的新范式——从"编写脚本"到"理解意图"。它的核心价值在于：

1. **降低门槛**：用自然语言替代DOM选择器，大幅降低浏览器自动化的开发成本
2. **视觉理解**：通过VLM理解页面，适应动态变化的网页
3. **智能恢复**：自动处理常见错误，提高任务完成率

但同时也需要注意：

1. **性能开销**：视觉理解会增加延迟和成本，需要合理使用缓存
2. **稳定性**：相比传统脚本，AI驱动的方案在稳定性上仍有差距
3. **适用场景**：更适合探索性、低频的操作，高频场景仍建议用传统方案

未来，随着VLM能力的提升和推理成本的下降，Browser-Use类的框架将在RPA、数据采集、自动化测试等领域发挥越来越重要的作用。
