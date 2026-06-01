---
title: "LLM工具调用错误处理与重试机制深度解析：构建鲁棒的Agent Tool Use系统"
description: "从实战角度深度剖析LLM工具调用中的常见错误类型、智能重试策略、降级方案与容错架构设计，帮你构建真正可靠的Agent系统"
date: 2026-06-01
author: "RiceBall"
category: "framework"
subCategory: agent-framework
tags: ["LLM", "Tool Use", "Agent", "错误处理", "重试机制", "容错设计", "Function Calling"]
draft: false
---

## 写在前面

在构建基于LLM的Agent系统时，**工具调用（Tool Use / Function Calling）** 是连接大模型与外部世界的核心桥梁。但在生产环境中，工具调用失败几乎不可避免——API超时、参数格式错误、权限不足、服务不可用……这些问题在开发环境很难遇到，一旦上线就会频繁出现。

根据我在多个Agent项目中的实战经验，**工具调用失败率在生产环境中通常在5%-15%之间**，如果缺乏有效的错误处理机制，Agent的端到端成功率会断崖式下降。

本文将从**错误分类、重试策略、降级方案、架构设计**四个维度，深度解析如何构建一个鲁棒的LLM工具调用系统。

---

## 一、工具调用错误的全景分类

### 1.1 错误分类体系

```
┌─────────────────────────────────────────────────────────────────┐
│                   LLM工具调用错误分类体系                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  LLM侧错误   │  │  传输层错误   │  │  工具侧错误   │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ • 参数格式错误 │  │ • 网络超时    │  │ • 服务不可用   │          │
│  │ • 参数类型不匹配│  │ • 连接中断    │  │ • 权限不足     │          │
│  │ • 缺少必填参数 │  │ • DNS解析失败  │  │ • 参数校验失败 │          │
│  │ • 幻觉参数    │  │ • TLS握手失败  │  │ • 业务逻辑错误 │          │
│  │ • 工具名幻觉  │  │ • 负载均衡失败 │  │ • 资源不足     │          │
│  │ • 参数值幻觉  │  │              │  │ • 配额耗尽     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  协议层错误   │  │  编排层错误   │                             │
│  ├──────────────┤  ├──────────────┤                             │
│  │ • JSON解析失败│  │ • 工具未注册   │                             │
│  │ • Schema不匹配│  │ • 参数映射错误 │                             │
│  │ • Content-Type│  │ • 循环调用     │                             │
│  │ • 超出Token限制│ │ • 并发冲突     │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 各类错误的出现频率与影响

| 错误类型 | 出现频率 | 影响程度 | 可恢复性 | 典型场景 |
|---------|---------|---------|---------|---------|
| LLM参数幻觉 | ⭐⭐⭐⭐⭐ | 中 | 高 | 模型生成不存在的参数名或参数值 |
| 网络超时 | ⭐⭐⭐⭐ | 中 | 高 | 调用第三方API超时 |
| 参数格式错误 | ⭐⭐⭐ | 低 | 高 | 模型输出的JSON格式不符合Schema |
| 服务不可用 | ⭐⭐ | 高 | 中 | 目标服务宕机或限流 |
| 权限不足 | ⭐⭐ | 高 | 低 | API Key过期或权限变更 |
| 业务逻辑错误 | ⭐⭐⭐ | 中 | 高 | 查询不存在的数据或操作违反业务规则 |
| 并发冲突 | ⭐⭐ | 中 | 中 | 多Agent同时操作同一资源 |

### 1.3 一个真实的错误日志分析

```python
# 真实场景：用户问"帮我查一下张三的订单，然后申请退款"
# Agent执行工具调用链：

# 调用1: search_user(name="张三") → 成功，返回user_id
# 调用2: query_orders(user_id="U12345") → 成功，返回订单列表
# 调用3: refund_order(order_id="ORD789", amount=299.00) → 失败

# 错误日志：
# {
#   "error_type": "tool_call_error",
#   "tool": "refund_order",
#   "error": "Order status is 'shipped', cannot refund directly",
#   "suggestion": "Please call cancel_order first",
#   "retry_count": 0,
#   "latency_ms": 234
# }
```

这个场景揭示了一个关键问题：**工具调用错误不仅仅是技术问题，更是业务逻辑问题**。LLM需要理解错误的语义，才能做出正确的决策。

---

## 二、LLM侧的参数错误处理

### 2.1 参数幻觉（Parameter Hallucination）

参数幻觉是最常见也最棘手的问题。LLM可能会：
- 生成不存在的参数名
- 生成参数值超出合法范围
- 给必填参数传None
- 类型不匹配（该传int传了string）

```python
import json
from typing import Any, Callable

class ToolParameterValidator:
    """工具参数校验器"""
    
    def __init__(self, tool_schema: dict):
        self.schema = tool_schema
        self.properties = tool_schema.get("properties", {})
        self.required = set(tool_schema.get("required", []))
    
    def validate_and_fix(self, params: dict) -> dict:
        """校验并自动修复参数"""
        fixed = {}
        errors = []
        
        for param_name, param_info in self.properties.items():
            value = params.get(param_name)
            
            if value is None:
                if param_name in self.required:
                    # 尝试使用默认值
                    if "default" in param_info:
                        fixed[param_name] = param_info["default"]
                    else:
                        errors.append(f"必填参数 {param_name} 缺失")
                continue
            
            # 类型修复
            expected_type = param_info.get("type")
            if expected_type == "integer" and isinstance(value, str):
                try:
                    fixed[param_name] = int(value)
                except ValueError:
                    errors.append(f"参数 {param_name} 无法转换为整数: {value}")
            elif expected_type == "number" and isinstance(value, str):
                try:
                    fixed[param_name] = float(value)
                except ValueError:
                    errors.append(f"参数 {param_name} 无法转换为数字: {value}")
            elif expected_type == "string" and not isinstance(value, str):
                fixed[param_name] = str(value)
            else:
                fixed[param_name] = value
            
            # 枚举值校验
            if "enum" in param_info and value not in param_info["enum"]:
                errors.append(f"参数 {param_name} 值 {value} 不在合法范围 {param_info['enum']}")
        
        # 检查是否有未定义的参数（可能是幻觉参数）
        extra_params = set(params.keys()) - set(self.properties.keys())
        if extra_params:
            # 记录但不阻断，可能是LLM理解了新的语义
            pass
        
        return {"params": fixed, "errors": errors, "fixed": True}
```

### 2.2 Schema校验与自动修复策略

```python
from dataclasses import dataclass
from enum import Enum

class FixStrategy(Enum):
    REJECT = "reject"           # 直接拒绝，重新生成
    AUTO_FIX = "auto_fix"       # 自动修复
    FALLBACK = "fallback"       # 使用默认值
    RETRY_WITH_FEEDBACK = "retry_with_feedback"  # 带错误信息重试

@dataclass
class ValidationResult:
    is_valid: bool
    fixed_params: dict
    errors: list[str]
    strategy_used: FixStrategy

class ToolCallErrorHandler:
    """工具调用错误处理器"""
    
    def __init__(self, tools: dict[str, dict]):
        self.tools = tools
        self.validators = {
            name: ToolParameterValidator(schema)
            for name, schema in tools.items()
        }
    
    def handle_error(self, tool_name: str, params: dict, error: Exception) -> ValidationResult:
        """处理工具调用错误，选择最佳修复策略"""
        
        if tool_name not in self.tools:
            return ValidationResult(
                is_valid=False,
                fixed_params={},
                errors=[f"未知工具: {tool_name}"],
                strategy_used=FixStrategy.REJECT
            )
        
        validator = self.validators[tool_name]
        result = validator.validate_and_fix(params)
        
        if not result["errors"]:
            return ValidationResult(
                is_valid=True,
                fixed_params=result["params"],
                errors=[],
                strategy_used=FixStrategy.AUTO_FIX
            )
        
        # 根据错误类型选择策略
        critical_errors = [e for e in result["errors"] if "必填" in e]
        type_errors = [e for e in result["errors"] if "转换" in e]
        
        if critical_errors and not type_errors:
            # 缺少必填参数，让LLM重新生成
            return ValidationResult(
                is_valid=False,
                fixed_params=result["params"],
                errors=result["errors"],
                strategy_used=FixStrategy.RETRY_WITH_FEEDBACK
            )
        
        if type_errors and not critical_errors:
            # 类型错误，自动修复
            return ValidationResult(
                is_valid=True,
                fixed_params=result["params"],
                errors=[],
                strategy_used=FixStrategy.AUTO_FIX
            )
        
        # 复合错误，直接拒绝
        return ValidationResult(
            is_valid=False,
            fixed_params=result["params"],
            errors=result["errors"],
            strategy_used=FixStrategy.REJECT
        )
```

---

## 三、智能重试策略设计

### 3.1 重试策略对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    重试策略对比                                    │
├──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│ 策略      │ 适用场景  │ 实现复杂度│ 用户体验 │ 核心思想              │
├──────────┼──────────┼──────────┼──────────┼──────────────────────┤
│ 简单重试   │ 临时故障  │ ⭐       │ 差      │ 相同参数重复调用       │
│ 指数退避   │ 网络抖动  │ ⭐⭐     │ 中      │ 间隔递增的重试        │
│ 参数修正重试│ 参数错误  │ ⭐⭐⭐   │ 好      │ 分析错误后修正参数     │
│ 语义重试   │ 业务错误  │ ⭐⭐⭐⭐  │ 优秀    │ LLM理解错误后重新规划  │
│ 降级重试   │ 服务不可用│ ⭐⭐⭐   │ 中      │ 切换到备选方案        │
└──────────┴──────────┴──────────┴──────────┴──────────────────────┘
```

### 3.2 指数退避重试实现

```python
import asyncio
import random
import time
from typing import Optional

class ExponentialBackoffRetrier:
    """指数退避重试器"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算退避延迟"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # 加入随机抖动，避免惊群效应
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    async def retry(self, func, *args, **kwargs):
        """执行重试"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)
        
        raise last_error
```

### 3.3 语义感知重试——让LLM理解错误

这是最强大也最复杂的重试策略。核心思想是：**将错误信息反馈给LLM，让LLM根据错误上下文重新生成工具调用**。

```python
class SemanticRetryHandler:
    """语义感知重试处理器"""
    
    def __init__(self, llm_client, max_retries: int = 3):
        self.llm = llm_client
        self.max_retries = max_retries
    
    async def execute_with_retry(
        self,
        tool_call: dict,
        tool_executor: callable,
        context: list[dict],
    ) -> dict:
        """带语义重试的工具调用执行"""
        
        retry_context = list(context)  # 保留原始上下文
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await tool_executor(
                    tool_call["name"],
                    tool_call["arguments"]
                )
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                }
                
            except ToolCallError as e:
                if attempt >= self.max_retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1,
                    }
                
                # 将错误信息加入上下文，让LLM重新决策
                error_feedback = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps({
                        "error": True,
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "suggestion": e.suggestion if hasattr(e, 'suggestion') else None,
                        "available_actions": self._get_suggested_actions(e),
                    }),
                }
                
                retry_context.append(error_feedback)
                
                # 让LLM基于错误信息重新生成工具调用
                new_tool_call = await self.llm.generate_tool_call(
                    messages=retry_context,
                    tools=self._get_available_tools(tool_call["name"]),
                )
                
                if new_tool_call and new_tool_call["name"] != "error":
                    tool_call = new_tool_call
                else:
                    # LLM无法修复，返回错误
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1,
                        "llm_suggestion": "LLM无法生成有效的替代方案",
                    }
        
        return {"success": False, "error": "Max retries exceeded", "attempts": self.max_retries}
    
    def _get_suggested_actions(self, error: Exception) -> list[str]:
        """根据错误类型提供建议操作"""
        suggestions = []
        
        if "timeout" in str(error).lower():
            suggestions.extend([
                "尝试使用更小的数据范围",
                "检查网络连接",
                "稍后重试",
            ])
        elif "permission" in str(error).lower() or "403" in str(error):
            suggestions.extend([
                "检查API Key是否有效",
                "确认权限范围",
                "联系管理员",
            ])
        elif "not found" in str(error).lower() or "404" in str(error):
            suggestions.extend([
                "检查资源ID是否正确",
                "确认资源是否已创建",
                "使用搜索接口查找资源",
            ])
        
        return suggestions
```

### 3.4 重试策略的决策树

```
工具调用失败
    │
    ├─ 是临时性错误？（网络超时、503、429）
    │   ├─ 是 → 指数退避重试（最多3次）
    │   │       └─ 仍然失败 → 降级到备选方案
    │   │
    │   └─ 否 ↓
    │
    ├─ 是参数错误？（400、参数校验失败）
    │   ├─ 是 → 分析错误详情
    │   │       ├─ 可自动修复 → 修复后重试
    │   │       └─ 不可修复 → 语义重试（让LLM重新生成）
    │   │
    │   └─ 否 ↓
    │
    ├─ 是业务错误？（数据不存在、状态冲突）
    │   ├─ 是 → 语义重试（让LLM理解错误并重新规划）
    │   │       └─ LLM无法修复 → 返回错误给用户
    │   │
    │   └─ 否 ↓
    │
    └─ 是权限/配置错误？（401、403、配置缺失）
        └─ 是 → 立即失败，不重试
              └─ 返回明确的错误说明
```

---

## 四、降级与兜底方案

### 4.1 工具级降级

当主要工具不可用时，自动切换到备选方案：

```python
@dataclass
class ToolFallbackChain:
    """工具降级链"""
    primary: str              # 主要工具
    fallbacks: list[str]      # 降级工具列表
    fallback_threshold: int = 3  # 触发降级的失败次数

class ToolOrchestrator:
    """工具编排器，支持降级"""
    
    def __init__(self, tools: dict, fallback_chains: dict[str, ToolFallbackChain]):
        self.tools = tools
        self.fallback_chains = fallback_chains
        self.failure_counts: dict[str, int] = {}
    
    async def execute(self, tool_name: str, params: dict) -> dict:
        """执行工具调用，支持自动降级"""
        
        chain = self.fallback_chains.get(tool_name)
        if not chain:
            # 没有降级链，直接执行
            return await self.tools[tool_name].execute(params)
        
        # 从主要工具开始尝试
        tools_to_try = [chain.primary] + chain.fallbacks
        
        for current_tool in tools_to_try:
            # 检查是否需要降级
            if current_tool == chain.primary:
                if self.failure_counts.get(current_tool, 0) >= chain.fallback_threshold:
                    continue  # 跳过已知不稳定的工具
            
            try:
                result = await self.tools[current_tool].execute(params)
                
                # 成功，重置失败计数
                self.failure_counts[current_tool] = 0
                return result
                
            except Exception as e:
                self.failure_counts[current_tool] = \
                    self.failure_counts.get(current_tool, 0) + 1
                
                if current_tool == chain.primary:
                    # 主要工具失败，记录日志并尝试降级
                    logger.warning(
                        f"Primary tool {current_tool} failed, "
                        f"attempting fallback. Error: {e}"
                    )
                    continue
                else:
                    # 降级工具也失败，继续尝试下一个
                    logger.warning(
                        f"Fallback tool {current_tool} also failed: {e}"
                    )
                    continue
        
        # 所有工具都失败了
        raise AllToolsFailedError(
            f"All tools in chain failed: {tools_to_try}"
        )
```

### 4.2 响应级降级

当工具返回的结果不符合预期时，使用缓存或默认值：

```python
class ResponseDegradationHandler:
    """响应降级处理器"""
    
    def __init__(self, cache_client, default_responses: dict):
        self.cache = cache_client
        self.defaults = default_responses
    
    async def handle_degraded_response(
        self,
        tool_name: str,
        params: dict,
        error: Exception,
    ) -> dict:
        """处理降级响应"""
        
        # 策略1: 尝试从缓存获取
        cache_key = self._build_cache_key(tool_name, params)
        cached = await self.cache.get(cache_key)
        if cached:
            return {
                "result": cached,
                "degradation_level": "cache_hit",
                "warning": "使用缓存数据，可能不是最新",
            }
        
        # 策略2: 使用默认响应
        if tool_name in self.defaults:
            default = self.defaults[tool_name]
            if callable(default):
                default = default(params)
            return {
                "result": default,
                "degradation_level": "default_value",
                "warning": "使用默认值，请核实后使用",
            }
        
        # 策略3: 返回部分结果
        if hasattr(error, 'partial_result') and error.partial_result:
            return {
                "result": error.partial_result,
                "degradation_level": "partial_result",
                "warning": f"仅获取到部分数据: {error.message}",
            }
        
        # 所有降级策略都失败
        raise DegradationFailedError(
            f"No degradation available for {tool_name}: {error}"
        )
```

### 4.3 降级策略矩阵

| 错误场景 | 降级策略 | 用户感知 | 数据准确性 |
|---------|---------|---------|-----------|
| API超时 | 切换到备选API | 无感知 | 高 |
| 服务宕机 | 返回缓存数据 | 提示"数据可能不是最新" | 中 |
| 参数校验失败 | LLM自动修正 | 无感知 | 高 |
| 配额耗尽 | 切换到免费接口 | 提示"功能受限" | 中 |
| 数据不存在 | 返回相关推荐 | 提示"未找到精确匹配" | 低 |
| 网络完全中断 | 离线模式 | 提示"当前离线" | 低 |

---

## 五、循环调用检测与防护

### 5.1 循环调用的识别

Agent系统中一个隐蔽但严重的问题是**工具调用循环**：LLM调用工具A，工具A的输出触发LLM再次调用工具A，形成无限循环。

```python
import hashlib
from collections import deque

class CircularCallDetector:
    """循环调用检测器"""
    
    def __init__(self, window_size: int = 10, max_same_tool: int = 3):
        self.window_size = window_size
        self.max_same_tool = max_same_tool
        self.call_history: deque = deque(maxlen=window_size)
    
    def _compute_call_signature(self, tool_name: str, params: dict) -> str:
        """计算调用签名"""
        content = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def check(self, tool_name: str, params: dict) -> tuple[bool, str]:
        """检查是否存在循环调用
        
        Returns:
            (is_circular, reason)
        """
        signature = self._compute_call_signature(tool_name, params)
        
        # 检查完全相同的连续调用
        if len(self.call_history) >= 2:
            recent = list(self.call_history)[-2:]
            if all(s == signature for s in recent):
                return True, f"检测到连续相同调用: {tool_name}"
        
        # 检查同一工具的频繁调用
        same_tool_count = sum(
            1 for s in self.call_history
            if self._tool_from_signature(s) == tool_name
        )
        if same_tool_count >= self.max_same_tool:
            return True, f"工具 {tool_name} 在最近{self.window_size}次调用中被调用了{same_tool_count}次"
        
        # 检查调用模式循环（A→B→A→B 模式）
        if len(self.call_history) >= 4:
            recent_4 = list(self.call_history)[-4:]
            if recent_4[0] == recent_4[2] and recent_4[1] == recent_4[3]:
                return True, "检测到交替循环调用模式"
        
        self.call_history.append(signature)
        return False, ""
    
    def _tool_from_signature(self, signature: str) -> str:
        """从签名还原工具名（简化实现）"""
        # 实际应用中需要维护签名到工具名的映射
        return signature  # placeholder
```

### 5.2 循环调用的处理策略

```python
class CircularCallHandler:
    """循环调用处理器"""
    
    def __init__(self, llm_client, max_circular_retries: int = 2):
        self.llm = llm_client
        self.max_circular_retries = max_circular_retries
        self.detector = CircularCallDetector()
    
    async def handle(self, tool_call: dict) -> dict:
        """处理潜在的循环调用"""
        
        is_circular, reason = self.detector.check(
            tool_call["name"],
            tool_call["arguments"]
        )
        
        if not is_circular:
            return tool_call  # 正常执行
        
        logger.warning(f"循环调用检测: {reason}")
        
        # 尝试让LLM换一种方式
        for attempt in range(self.max_circular_retries):
            alternative = await self.llm.generate_alternative_tool_call(
                original=tool_call,
                reason=reason,
                message=f"请换一种方式完成这个任务，避免重复调用{tool_call['name']}。",
            )
            
            if alternative and alternative["name"] != tool_call["name"]:
                return alternative
        
        # LLM无法跳出循环，返回错误
        raise CircularCallError(
            f"无法跳出工具调用循环: {reason}"
        )
```

---

## 六、错误恢复的完整流程

### 6.1 错误恢复状态机

```
                    ┌──────────────┐
                    │   初始状态    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  执行工具调用  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         成功│       参数错误│       服务错误│
              │            │            │
       ┌──────▼──┐  ┌──────▼──┐  ┌──────▼──┐
       │ 返回结果 │  │ 参数修复 │  │ 指数退避 │
       └─────────┘  └────┬────┘  └────┬────┘
                         │            │
                  ┌──────▼──────┐     │
                  │  校验通过？  │     │
                  └──┬───────┬──┘     │
                是│       否│         │
                  │         │         │
           ┌──────▼──┐  ┌──▼─────┐   │
           │ 重试执行 │  │语义重试│   │
           └────┬────┘  └──┬─────┘   │
                │          │         │
                └────┬─────┘    ┌────▼────┐
                     │          │ 重试成功? │
                ┌────▼────┐  ┌──┴───────┐ │
                │ 返回结果 │  │是│     否│ │
                └─────────┘  │  │    ┌──▼──┐
                              │  │    │降级 │
                              │  │    │处理 │
                              │  │    └──┬──┘
                              │  │       │
                              │  │  ┌────▼────┐
                              │  │  │降级成功? │
                              │  │  └──┬───┬──┘
                              │  │  是│   否│
                              │  │  │   │  │
                              │  │ ┌▼┐ ┌▼──▼┐
                              │  │ │✅│ │ ❌  │
                              │  │ └─┘ └────┘
                              │  │
                              └──┘
```

### 6.2 完整的工具调用执行器

```python
class RobustToolExecutor:
    """鲁棒的工具调用执行器"""
    
    def __init__(
        self,
        tools: dict,
        llm_client,
        cache_client=None,
        fallback_chains: dict = None,
    ):
        self.tools = tools
        self.llm = llm_client
        self.cache = cache_client
        self.param_handler = ToolCallErrorHandler(tools)
        self.semantic_retry = SemanticRetryHandler(llm_client)
        self.circular_detector = CircularCallDetector()
        self.fallback_handler = ToolOrchestrator(tools, fallback_chains or {})
    
    async def execute(
        self,
        tool_call: dict,
        context: list[dict],
        max_retries: int = 3,
    ) -> dict:
        """执行工具调用的完整流程"""
        
        # Step 1: 循环调用检测
        is_circular, reason = self.circular_detector.check(
            tool_call["name"], tool_call["arguments"]
        )
        if is_circular:
            tool_call = await self._handle_circular(tool_call, reason)
        
        # Step 2: 参数校验与修复
        validation = self.param_handler.handle_error(
            tool_call["name"],
            tool_call["arguments"],
            None,
        )
        
        if validation.strategy_used == FixStrategy.REJECT:
            return await self._handle_rejection(tool_call, validation.errors)
        
        # Step 3: 执行调用（带降级）
        try:
            result = await self.fallback_handler.execute(
                tool_call["name"],
                validation.fixed_params,
            )
            return {
                "success": True,
                "result": result,
                "fixes_applied": validation.errors,
                "degradation": None,
            }
            
        except Exception as e:
            # Step 4: 语义重试
            retry_result = await self.semantic_retry.execute_with_retry(
                tool_call, 
                self._execute_single,
                context,
            )
            
            if retry_result["success"]:
                return {
                    "success": True,
                    "result": retry_result["result"],
                    "attempts": retry_result["attempts"],
                    "degradation": None,
                }
            
            # Step 5: 最终降级
            return await self._final_fallback(tool_call, e)
    
    async def _final_fallback(self, tool_call: dict, error: Exception) -> dict:
        """最终降级处理"""
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "tool": tool_call["name"],
            "message": f"工具 {tool_call['name']} 调用失败，已尝试所有恢复策略。"
                       f"请检查参数或联系管理员。",
        }
```

---

## 七、生产环境的最佳实践

### 7.1 错误监控与告警

```python
class ToolCallMetrics:
    """工具调用指标监控"""
    
    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "success_count": 0,
            "failure_count": 0,
            "retry_count": 0,
            "degradation_count": 0,
            "circular_detection_count": 0,
            "by_tool": {},  # 按工具分类的指标
            "error_types": {},  # 按错误类型分类
        }
    
    def record_call(self, tool_name: str, result: dict):
        """记录一次调用"""
        self.metrics["total_calls"] += 1
        
        if result.get("success"):
            self.metrics["success_count"] += 1
        else:
            self.metrics["failure_count"] += 1
        
        self.metrics["retry_count"] += result.get("attempts", 1) - 1
        
        if result.get("degradation"):
            self.metrics["degradation_count"] += 1
        
        # 按工具分类
        if tool_name not in self.metrics["by_tool"]:
            self.metrics["by_tool"][tool_name] = {
                "calls": 0, "successes": 0, "failures": 0
            }
        self.metrics["by_tool"][tool_name]["calls"] += 1
        if result.get("success"):
            self.metrics["by_tool"][tool_name]["successes"] += 1
        else:
            self.metrics["by_tool"][tool_name]["failures"] += 1
    
    def get_health_report(self) -> dict:
        """生成健康报告"""
        total = self.metrics["total_calls"]
        if total == 0:
            return {"status": "no_data"}
        
        success_rate = self.metrics["success_count"] / total
        avg_retries = self.metrics["retry_count"] / total
        
        status = "healthy"
        if success_rate < 0.9:
            status = "warning"
        if success_rate < 0.7:
            status = "critical"
        
        return {
            "status": status,
            "success_rate": f"{success_rate:.1%}",
            "total_calls": total,
            "avg_retries": f"{avg_retries:.2f}",
            "degradation_rate": f"{self.metrics['degradation_count'] / total:.1%}",
            "top_failing_tools": self._get_top_failing(5),
        }
```

### 7.2 配置化管理

```yaml
# tool_call_config.yaml
tool_call:
  retry:
    max_retries: 3
    base_delay: 1.0
    max_delay: 30.0
    exponential_base: 2.0
    jitter: true
  
  timeout:
    default: 10.0
    per_tool:
      search_web: 5.0
      execute_code: 30.0
      send_email: 15.0
  
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
    half_open_max_calls: 3
  
  degradation:
    enable_cache: true
    cache_ttl: 3600
    enable_default_responses: true
  
  circular_detection:
    window_size: 10
    max_same_tool: 3
    max_total_calls_per_turn: 20
```

### 7.3 错误日志规范

```python
import structlog

logger = structlog.get_logger("tool_call")

def log_tool_call(
    tool_name: str,
    params: dict,
    result: dict,
    context: dict,
):
    """结构化工具调用日志"""
    
    log_data = {
        "tool": tool_name,
        "params_hash": hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8],
        "success": result.get("success", False),
        "duration_ms": result.get("duration_ms"),
        "attempts": result.get("attempts", 1),
        "degradation": result.get("degradation"),
        "session_id": context.get("session_id"),
        "turn_id": context.get("turn_id"),
    }
    
    if result.get("success"):
        logger.info("tool_call_success", **log_data)
    else:
        log_data["error_type"] = result.get("error_type")
        log_data["error_message"] = result.get("error")
        logger.error("tool_call_failed", **log_data)
```

---

## 八、总结与建议

### 核心原则

1. **不要假设工具调用永远成功**：在生产环境中，5%-15%的失败率是常态
2. **分层处理**：参数错误→自动修复，网络错误→重试，业务错误→语义重试，服务错误→降级
3. **让LLM参与错误恢复**：语义重试是最强大的武器，让LLM理解错误并重新规划
4. **监控一切**：没有监控的重试策略是盲目的，必须建立完善的指标体系
5. **优雅降级**：永远给用户一个可用的响应，即使数据不完整

### 选择重试策略的决策框架

| 你的情况 | 推荐策略 | 优先级 |
|---------|---------|-------|
| 刚开始构建Agent系统 | 简单重试 + 指数退避 | P0 |
| 面临参数格式问题 | 参数校验 + 自动修复 | P0 |
| 需要高可靠性 | 语义重试 + 降级链 | P1 |
| 生产环境大规模部署 | 完整的错误恢复状态机 + 监控 | P1 |
| 对延迟敏感 | 超时控制 + 快速降级 | P1 |

构建鲁棒的工具调用系统不是一次性工作，而是需要持续迭代的过程。从简单的重试开始，逐步引入参数修复、语义重试和降级方案，最终构建出一个能自我恢复的Agent系统。

---

*如果你正在构建Agent系统，欢迎分享你在工具调用错误处理方面的实践经验。*
