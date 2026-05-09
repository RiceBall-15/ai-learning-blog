---
title: "Agent Skill 调试与错误处理：从崩溃到优雅降级"
description: "深入探讨 Agent Skill 的调试技巧、错误处理机制、日志系统设计，以及生产环境的问题排查方法"
date: 2026-05-09
author: RiceBall-15
category: agentSkill
tags: ["Agent Skill", "调试", "错误处理", "日志", "问题排查"]
---

# Agent Skill 调试与错误处理：从崩溃到优雅降级

## 简介

"代码能跑就行"是危险的想法。当 Skill 在生产环境崩溃，你需要快速定位问题；当输入异常，你需要优雅降级而非直接报错。本文探讨如何构建健壮的 Skill，从调试技巧到错误处理，让你的代码在各种情况下都能从容应对。

## 问题背景

调试和错误处理的典型痛点：

1. **调试困难**：复现问题难，定位更难
2. **错误信息模糊**：只知道"出错了"，不知道为什么
3. **级联故障**：一个错误导致整个系统崩溃
4. **日志混乱**：关键信息被淹没在海量日志中
5. **恢复困难**：出错后无法自动恢复

参考 Google SRE 的错误预算理论【1】和 Python 的 EAFP 原则【2】，我们需要建立系统化的调试和错误处理机制。

## 调试技巧

### 结构化日志

```python
# structured_logger.py
import logging
import json
from datetime import datetime
from typing import Any, Dict
from contextlib import contextmanager
import traceback
import uuid

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # JSON 格式化器
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """记录错误日志"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
            kwargs['traceback'] = traceback.format_exc()
        self._log(logging.ERROR, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """内部日志方法"""
        extra = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            **kwargs
        }
        self.logger.log(level, message, extra=extra)

class JsonFormatter(logging.Formatter):
    """JSON 格式化器"""
    
    def format(self, record):
        log_data = {
            'timestamp': record.extra.get('timestamp'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # 添加额外字段
        for key, value in record.extra.items():
            if key not in ['timestamp']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)
```

### 上下文追踪

```python
# context_tracker.py
import contextvars
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from functools import wraps
import uuid

@dataclass
class ExecutionContext:
    """执行上下文"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skill_name: str = ""
    user_id: str = ""
    parent_request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0

# 上下文变量（线程安全）
current_context: contextvars.ContextVar[ExecutionContext] = \
    contextvars.ContextVar('current_context')

class ContextTracker:
    """上下文追踪器"""
    
    @staticmethod
    def get_current() -> Optional[ExecutionContext]:
        """获取当前上下文"""
        try:
            return current_context.get()
        except LookupError:
            return None
    
    @staticmethod
    @contextmanager
    def track(skill_name: str, **metadata):
        """追踪执行上下文"""
        parent = ContextTracker.get_current()
        
        ctx = ExecutionContext(
            skill_name=skill_name,
            parent_request_id=parent.request_id if parent else None,
            metadata=metadata
        )
        
        token = current_context.set(ctx)
        try:
            yield ctx
        finally:
            current_context.reset(token)
    
    @staticmethod
    def with_context(func):
        """上下文装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = ContextTracker.get_current()
            if ctx:
                # 记录调用信息
                ctx.metadata['last_call'] = func.__name__
            return func(*args, **kwargs)
        return wrapper
```

### 断点调试

```python
# breakpoint_debug.py
import pdb
import sys
from typing import Callable, Any
from functools import wraps

class ConditionalBreakpoint:
    """条件断点"""
    
    def __init__(self, condition: Callable[[Any], bool]):
        self.condition = condition
        self.hit_count = 0
    
    def check(self, **locals_dict) -> bool:
        """检查是否触发断点"""
        try:
            if self.condition(locals_dict):
                self.hit_count += 1
                return True
        except:
            pass
        return False

def breakpoint_on_error(func):
    """错误时触发断点"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n❌ 异常发生: {e}")
            print("进入调试模式...")
            pdb.post_mortem()
            raise
    return wrapper

def trace_calls(func):
    """追踪函数调用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"→ 调用 {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            print(f"← {func.__name__} 返回: {result}")
            return result
        except Exception as e:
            print(f"← {func.__name__} 抛出异常: {e}")
            raise
    return wrapper
```

### 远程调试

```python
# remote_debug.py
import socket
import threading
import code
import sys
from io import StringIO

class RemoteDebugger:
    """远程调试服务器"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5678):
        self.host = host
        self.port = port
        self._server = None
        self._running = False
    
    def start(self):
        """启动调试服务器"""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self._running = True
        
        print(f"🐛 调试服务器启动: {self.host}:{self.port}")
        
        thread = threading.Thread(target=self._accept_connections)
        thread.daemon = True
        thread.start()
    
    def _accept_connections(self):
        """接受连接"""
        while self._running:
            try:
                client, addr = self._server.accept()
                print(f"🔗 调试客户端连接: {addr}")
                
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client,)
                )
                thread.daemon = True
                thread.start()
            except:
                break
    
    def _handle_client(self, client: socket.socket):
        """处理客户端"""
        # 欢迎消息
        client.send(b"Welcome to Skill Remote Debugger\n")
        client.send(b">>> ")
        
        # 交互式 Python 环境
        console = code.InteractiveConsole()
        
        while True:
            try:
                data = client.recv(1024)
                if not data:
                    break
                
                command = data.decode().strip()
                
                if command == 'quit':
                    client.send(b"Goodbye!\n")
                    break
                
                # 执行命令
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                try:
                    console.push(command)
                    output = sys.stdout.getvalue()
                    error = sys.stderr.getvalue()
                    
                    if output:
                        client.send(output.encode())
                    if error:
                        client.send(error.encode())
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                
                client.send(b">>> ")
            
            except Exception as e:
                client.send(f"Error: {e}\n>>> ".encode())
        
        client.close()
```

## 错误处理

### 异常层次设计

```python
# exceptions.py

class SkillError(Exception):
    """Skill 基础异常"""
    
    def __init__(self, message: str, 
                 error_code: str = None,
                 details: dict = None):
        super().__init__(message)
        self.error_code = error_code or 'SKILL_ERROR'
        self.details = details or {}

class ValidationError(SkillError):
    """输入验证错误"""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message, error_code='VALIDATION_ERROR', **kwargs)
        self.field = field

class ExecutionError(SkillError):
    """执行错误"""
    
    def __init__(self, message: str, 
                 skill_name: str = None,
                 original_error: Exception = None,
                 **kwargs):
        super().__init__(message, error_code='EXECUTION_ERROR', **kwargs)
        self.skill_name = skill_name
        self.original_error = original_error

class TimeoutError(SkillError):
    """超时错误"""
    
    def __init__(self, message: str, 
                 timeout_seconds: float = None,
                 **kwargs):
        super().__init__(message, error_code='TIMEOUT_ERROR', **kwargs)
        self.timeout_seconds = timeout_seconds

class ResourceError(SkillError):
    """资源错误"""
    
    def __init__(self, message: str,
                 resource_type: str = None,
                 limit: float = None,
                 actual: float = None,
                 **kwargs):
        super().__init__(message, error_code='RESOURCE_ERROR', **kwargs)
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual

class DependencyError(SkillError):
    """依赖错误"""
    
    def __init__(self, message: str,
                 dependency_name: str = None,
                 **kwargs):
        super().__init__(message, error_code='DEPENDENCY_ERROR', **kwargs)
        self.dependency_name = dependency_name

class ConfigurationError(SkillError):
    """配置错误"""
    
    def __init__(self, message: str,
                 config_key: str = None,
                 **kwargs):
        super().__init__(message, error_code='CONFIG_ERROR', **kwargs)
        self.config_key = config_key
```

### 统一错误处理

```python
# error_handler.py
from typing import Callable, Any, Optional
from functools import wraps
import traceback
import logging

class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self, 
                 logger: logging.Logger = None,
                 notify: Callable = None):
        self.logger = logger or logging.getLogger(__name__)
        self.notify = notify
        self._handlers = {}
    
    def register(self, 
                 exception_type: type,
                 handler: Callable[[Exception], Any]):
        """注册异常处理器"""
        self._handlers[exception_type] = handler
    
    def handle(self, error: Exception) -> Any:
        """处理异常"""
        # 查找精确匹配
        for exc_type, handler in self._handlers.items():
            if isinstance(error, exc_type):
                return handler(error)
        
        # 查找父类匹配
        for exc_type, handler in self._handlers.items():
            if isinstance(error, exc_type):
                return handler(error)
        
        # 默认处理
        return self._default_handler(error)
    
    def _default_handler(self, error: Exception):
        """默认处理器"""
        self.logger.error(
            f"未处理异常: {type(error).__name__}: {error}",
            exc_info=True
        )
        
        # 发送通知
        if self.notify:
            self.notify(error)
        
        # 重新抛出
        raise error

# 全局错误处理器
error_handler = ErrorHandler()

# 装饰器
def handle_errors(fallback: Any = None,
                  reraise: bool = True,
                  notify: bool = False):
    """错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录错误
                logging.error(
                    f"函数 {func.__name__} 执行失败: {e}",
                    exc_info=True
                )
                
                # 发送通知
                if notify and error_handler.notify:
                    error_handler.notify(e)
                
                # 返回 fallback
                if fallback is not None:
                    return fallback
                
                # 重新抛出
                if reraise:
                    raise
                
                return None
        return wrapper
    return decorator
```

### 优雅降级

```python
# graceful_degradation.py
from typing import Callable, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

class DegradationLevel(Enum):
    """降级级别"""
    FULL = "full"           # 完整功能
    DEGRADED = "degraded"   # 降级模式
    MINIMAL = "minimal"     # 最小功能
    OFFLINE = "offline"     # 离线模式

@dataclass
class DegradationStrategy:
    """降级策略"""
    level: DegradationLevel
    handler: Callable
    condition: Callable[[], bool]
    description: str

class GracefulDegradation:
    """优雅降级管理器"""
    
    def __init__(self):
        self.strategies: List[DegradationStrategy] = []
        self.current_level = DegradationLevel.FULL
        self.logger = logging.getLogger(__name__)
    
    def register(self, 
                 level: DegradationLevel,
                 handler: Callable,
                 condition: Callable[[], bool],
                 description: str = ""):
        """注册降级策略"""
        self.strategies.append(DegradationStrategy(
            level=level,
            handler=handler,
            condition=condition,
            description=description
        ))
        
        # 按级别排序
        self.strategies.sort(
            key=lambda s: list(DegradationLevel).index(s.level)
        )
    
    def execute(self, *args, **kwargs) -> Any:
        """执行，自动降级"""
        for strategy in self.strategies:
            try:
                # 检查条件
                if not strategy.condition():
                    continue
                
                self.logger.info(
                    f"使用 {strategy.level.value} 模式: "
                    f"{strategy.description}"
                )
                
                result = strategy.handler(*args, **kwargs)
                self.current_level = strategy.level
                return result
                
            except Exception as e:
                self.logger.warning(
                    f"{strategy.level.value} 模式失败: {e}"
                )
                continue
        
        raise AllStrategiesFailedError("所有降级策略都失败了")

# 使用示例
degradation = GracefulDegradation()

# 完整模式：调用远程 API
degradation.register(
    level=DegradationLevel.FULL,
    handler=lambda x: call_remote_api(x),
    condition=lambda: check_network_available(),
    description="远程 API 调用"
)

# 降级模式：使用本地缓存
degradation.register(
    level=DegradationLevel.DEGRADED,
    handler=lambda x: get_from_cache(x),
    condition=lambda: cache_available(),
    description="本地缓存"
)

# 最小模式：返回默认值
degradation.register(
    level=DegradationLevel.MINIMAL,
    handler=lambda x: get_default_value(x),
    condition=lambda: True,
    description="默认值"
)
```

### 重试机制

```python
# retry_mechanism.py
import asyncio
import time
from typing import Callable, Any, Tuple, Optional
from functools import wraps
from enum import Enum

class RetryStrategy(Enum):
    """重试策略"""
    FIXED = "fixed"              # 固定间隔
    EXPONENTIAL = "exponential"  # 指数退避
    LINEAR = "linear"           # 线性增长
    FIBONACCI = "fibonacci"     # 斐波那契

class RetryConfig:
    """重试配置"""
    
    def __init__(self,
                 max_attempts: int = 3,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True,
                 retry_on: Tuple[type, ...] = (Exception,),
                 stop_on: Tuple[type, ...] = ()):
        self.max_attempts = max_attempts
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = retry_on
        self.stop_on = stop_on

class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self._fib_cache = {0: 0, 1: 1}
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2 ** attempt)
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fib(attempt + 1)
        
        # 添加抖动
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return min(delay, self.config.max_delay)
    
    def _fib(self, n: int) -> int:
        """斐波那契数"""
        if n not in self._fib_cache:
            self._fib_cache[n] = self._fib(n-1) + self._fib(n-2)
        return self._fib_cache[n]

def retry(config: RetryConfig = None):
    """重试装饰器"""
    config = config or RetryConfig()
    manager = RetryManager(config)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.stop_on as e:
                    raise
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = manager.calculate_delay(attempt)
                        logging.warning(
                            f"重试 {attempt + 1}/{config.max_attempts}: "
                            f"{e}, 等待 {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.stop_on as e:
                    raise
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = manager.calculate_delay(attempt)
                        logging.warning(
                            f"重试 {attempt + 1}/{config.max_attempts}: "
                            f"{e}, 等待 {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        raise
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
```

## 日志系统

### 日志级别设计

```python
# log_levels.py
import logging

# 自定义日志级别
TRACE = 5
DEBUG = logging.DEBUG      # 10
INFO = logging.INFO        # 20
NOTICE = 25                # 介于 INFO 和 WARNING 之间
WARNING = logging.WARNING  # 30
ERROR = logging.ERROR      # 40
CRITICAL = logging.CRITICAL # 50
ALERT = 55                 # 需要立即关注
EMERGENCY = 60             # 系统不可用

logging.addLevelName(TRACE, 'TRACE')
logging.addLevelName(NOTICE, 'NOTICE')
logging.addLevelName(ALERT, 'ALERT')
logging.addLevelName(EMERGENCY, 'EMERGENCY')

class SkillLogger(logging.Logger):
    """Skill 专用日志器"""
    
    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kwargs)
    
    def notice(self, message, *args, **kwargs):
        if self.isEnabledFor(NOTICE):
            self._log(NOTICE, message, args, **kwargs)
    
    def alert(self, message, *args, **kwargs):
        if self.isEnabledFor(ALERT):
            self._log(ALERT, message, args, **kwargs)
    
    def emergency(self, message, *args, **kwargs):
        if self.isEnabledFor(EMERGENCY):
            self._log(EMERGENCY, message, args, **kwargs)

# 设置为默认日志器
logging.setLoggerClass(SkillLogger)
```

### 日志轮转

```python
# log_rotation.py
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

class SkillLogManager:
    """日志管理器"""
    
    def __init__(self,
                 log_dir: str = "/var/log/skill",
                 max_size_mb: int = 100,
                 backup_count: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.backup_count = backup_count
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取日志器"""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            # 文件处理器（按大小轮转）
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{name}.log",
                maxBytes=self.max_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
            
            # 错误日志单独记录
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{name}.error.log",
                maxBytes=self.max_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
                'Traceback: %(exc_info)s'
            ))
            logger.addHandler(error_handler)
        
        return logger
```

## 生产环境排查

### 问题定位流程

```
1. 收集信息
   ├── 错误日志
   ├── 监控指标
   └── 用户反馈

2. 复现问题
   ├── 相同输入
   ├── 相同环境
   └── 相同时间

3. 定位根因
   ├── 二分法排查
   ├── 日志分析
   └── 性能分析

4. 修复验证
   ├── 单元测试
   ├── 集成测试
   └── 生产验证

5. 复盘总结
   ├── 根因分析
   ├── 改进措施
   └── 知识沉淀
```

### 常见问题排查

```python
# troubleshooting.py
class Troubleshooter:
    """问题排查器"""
    
    @staticmethod
    def check_memory():
        """检查内存使用"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    @staticmethod
    def check_cpu():
        """检查 CPU 使用"""
        import psutil
        
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'load_avg': psutil.getloadavg()
        }
    
    @staticmethod
    def check_disk():
        """检查磁盘使用"""
        import psutil
        
        disk = psutil.disk_usage('/')
        return {
            'total_gb': disk.total / 1024 / 1024 / 1024,
            'used_gb': disk.used / 1024 / 1024 / 1024,
            'free_gb': disk.free / 1024 / 1024 / 1024,
            'percent': disk.percent
        }
    
    @staticmethod
    def check_connections():
        """检查网络连接"""
        import psutil
        
        connections = psutil.net_connections()
        return {
            'total': len(connections),
            'established': len([c for c in connections if c.status == 'ESTABLISHED']),
            'listening': len([c for c in connections if c.status == 'LISTEN'])
        }
    
    @staticmethod
    def generate_report():
        """生成诊断报告"""
        return {
            'memory': Troubleshooter.check_memory(),
            'cpu': Troubleshooter.check_cpu(),
            'disk': Troubleshooter.check_disk(),
            'connections': Troubleshooter.check_connections()
        }
```

## 最佳实践总结

### 调试技巧
- 结构化日志：JSON 格式，便于分析
- 上下文追踪：串联请求链路
- 远程调试：生产环境也能调试

### 错误处理
- 异常层次：分类明确
- 统一处理：避免遗漏
- 优雅降级：保证可用性

### 日志系统
- 级别分明：TRACE/DEBUG/INFO/WARNING/ERROR
- 轮转管理：避免日志爆炸
- 敏感信息：脱敏处理

### 问题排查
- 标准流程：收集→复现→定位→修复→复盘
- 监控告警：早发现早处理
- 知识沉淀：避免重复踩坑

## 参考来源

1. Google SRE Book: "Managing Incidents" - https://sre.google/sre-book/managing-incidents/
2. Python PEP 20: "EAFP vs LBYL" - https://docs.python.org/3/glossary.html#term-eafp
3. Python Logging Documentation - https://docs.python.org/3/library/logging.html
4. Brendan Gregg: "Systems Performance" - https://www.brendangregg.com/systems-performance-2nd-edition-book.html
5. The Art of Debugging - https://nostarch.com/debugging

---

*本文首发于 RiceBall-15 的技术博客，转载请注明出处。*
