---
title: "Agent Skill 安全性与权限控制：构建可信的 Skill 生态"
description: "深入探讨 Agent Skill 的安全威胁模型、权限控制机制、沙箱隔离、代码审计，以及企业级安全实践"
date: 2026-05-09
author: RiceBall-15
category: agentSkill
subCategory: agent-skill
tags: ["Agent Skill", "安全", "权限控制", "沙箱", "RBAC"]
---


## 简介

Agent Skill 可以执行代码、访问文件、调用 API — 这些能力如果被滥用，后果不堪设想。一个恶意 Skill 可以窃取数据、植入后门、甚至控制整个系统。本文探讨如何构建安全的 Skill 生态，从威胁分析到防护机制，确保 Agent 系统的安全可控。

## 问题背景

Skill 安全的典型威胁：

1. **代码注入**：Skill 执行恶意代码
2. **权限提升**：Skill 获得超出预期的权限
3. **数据泄露**：Skill 窃取敏感信息
4. **供应链攻击**：恶意 Skill 伪装成合法 Skill
5. **资源滥用**：Skill 消耗过多资源导致 DoS

参考 OWASP Top 10【1】和 MITRE ATT&CK【2】，我们需要建立全面的安全防护。

## 威胁模型

### 攻击面分析

```
┌─────────────────────────────────────────────────┐
│                  Skill 系统                      │
├─────────────────────────────────────────────────┤
│  攻击面 1: Skill 加载                            │
│  - 恶意代码注入                                  │
│  - 依赖混淆攻击                                  │
├─────────────────────────────────────────────────┤
│  攻击面 2: Skill 执行                            │
│  - 权限提升                                      │
│  - 数据泄露                                      │
│  - 资源耗尽                                      │
├─────────────────────────────────────────────────┤
│  攻击面 3: Skill 通信                            │
│  - 中间人攻击                                    │
│  - 消息篡改                                      │
├─────────────────────────────────────────────────┤
│  攻击面 4: Skill 存储                            │
│  - 未授权访问                                    │
│  - 数据篡改                                      │
└─────────────────────────────────────────────────┘
```

### 攻击向量

| 攻击类型 | 描述 | 风险等级 |
|---------|------|---------|
| 代码注入 | 通过输入执行任意代码 | 高 |
| 路径遍历 | 访问系统敏感文件 | 高 |
| SSRF | 利用服务器发起内部请求 | 高 |
| 反序列化 | 通过恶意数据执行代码 | 高 |
| 资源耗尽 | CPU/内存/磁盘 DoS | 中 |
| 信息泄露 | 暴露系统敏感信息 | 中 |

## 权限控制模型

### 基于角色的访问控制（RBAC）

参考 NIST RBAC 标准【3】：

```
用户 (User)
   │
   ├── 角色 (Role)
   │      │
   │      ├── 权限 (Permission)
   │      │      │
   │      │      ├── 资源 (Resource)
   │      │      └── 操作 (Operation)
```

```python
# rbac.py
from dataclasses import dataclass
from enum import Enum
from typing import Set, Dict, List

class Operation(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"

class ResourceType(Enum):
    FILE = "file"
    TERMINAL = "terminal"
    NETWORK = "network"
    DATABASE = "database"
    ENVIRONMENT = "environment"

@dataclass
class Permission:
    resource: ResourceType
    operation: Operation
    scope: str = "*"  # 资源范围，如 "/tmp/*"

@dataclass
class Role:
    name: str
    permissions: Set[Permission]
    inherits: List[str] = None  # 继承的角色

class RBACManager:
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}
    
    def create_role(self, name: str, permissions: Set[Permission]):
        """创建角色"""
        self.roles[name] = Role(name=name, permissions=permissions)
    
    def assign_role(self, user_id: str, role_name: str):
        """分配角色"""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role_name)
    
    def check_permission(self, user_id: str, 
                         resource: ResourceType,
                         operation: Operation,
                         scope: str = "*") -> bool:
        """检查权限"""
        user_roles = self.user_roles.get(user_id, set())
        
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if not role:
                continue
            
            # 检查直接权限
            for perm in role.permissions:
                if (perm.resource == resource and 
                    perm.operation == operation and
                    self._match_scope(perm.scope, scope)):
                    return True
            
            # 检查继承角色
            if role.inherits:
                for inherited in role.inherits:
                    if self._check_role_permission(
                        inherited, resource, operation, scope
                    ):
                        return True
        
        return False
    
    def _match_scope(self, pattern: str, scope: str) -> bool:
        """匹配资源范围"""
        if pattern == "*":
            return True
        # 支持通配符匹配
        import fnmatch
        return fnmatch.fnmatch(scope, pattern)
```

### 预定义角色

```python
# predefined_roles.py
def setup_default_roles(rbac: RBACManager):
    """设置默认角色"""
    
    # 只读角色
    rbac.create_role("viewer", {
        Permission(ResourceType.FILE, Operation.READ, "/data/*"),
        Permission(ResourceType.DATABASE, Operation.READ, "*"),
    })
    
    # 开发者角色
    rbac.create_role("developer", {
        Permission(ResourceType.FILE, Operation.READ, "/data/*"),
        Permission(ResourceType.FILE, Operation.WRITE, "/data/*"),
        Permission(ResourceType.TERMINAL, Operation.EXECUTE, "safe-*"),
        Permission(ResourceType.NETWORK, Operation.READ, "*"),
    })
    
    # 运维角色
    rbac.create_role("operator", {
        Permission(ResourceType.FILE, Operation.READ, "*"),
        Permission(ResourceType.FILE, Operation.WRITE, "/var/*"),
        Permission(ResourceType.TERMINAL, Operation.EXECUTE, "*"),
        Permission(ResourceType.NETWORK, Operation.READ, "*"),
        Permission(ResourceType.NETWORK, Operation.WRITE, "*"),
    })
    
    # 管理员角色（继承所有角色）
    rbac.create_role("admin", {
        Permission(ResourceType.FILE, Operation.ADMIN, "*"),
        Permission(ResourceType.TERMINAL, Operation.ADMIN, "*"),
        Permission(ResourceType.DATABASE, Operation.ADMIN, "*"),
    })
```

## 沙箱隔离

### 进程隔离

参考 Docker 的容器隔离【4】：

```python
# sandbox.py
import subprocess
import tempfile
import os
import resource
from pathlib import Path
from typing import Dict, Optional

class Sandbox:
    def __init__(self, 
                 memory_limit_mb: int = 256,
                 cpu_limit_percent: int = 50,
                 network_enabled: bool = False,
                 allowed_paths: list = None):
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.network_enabled = network_enabled
        self.allowed_paths = allowed_paths or ["/tmp"]
        
        # 创建临时工作目录
        self.work_dir = tempfile.mkdtemp(prefix="skill_sandbox_")
    
    def execute(self, command: str, 
                env: Dict[str, str] = None,
                timeout: int = 30) -> Dict:
        """在沙箱中执行命令"""
        
        # 准备环境变量
        sandbox_env = os.environ.copy()
        if env:
            sandbox_env.update(env)
        
        # 设置资源限制
        def set_limits():
            # 内存限制
            memory_bytes = self.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, 
                             (memory_bytes, memory_bytes))
            
            # CPU 时间限制
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            
            # 文件大小限制
            file_size = 100 * 1024 * 1024  # 100MB
            resource.setrlimit(resource.RLIMIT_FSIZE, 
                             (file_size, file_size))
            
            # 进程数限制
            resource.setrlimit(resource.RLIMIT_NPROC, (100, 100))
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,
                env=sandbox_env,
                preexec_fn=set_limits,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': '执行超时',
                'exit_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'exit_code': -1
            }
    
    def cleanup(self):
        """清理沙箱"""
        import shutil
        shutil.rmtree(self.work_dir, ignore_errors=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
```

### 文件系统隔离

```python
# filesystem_sandbox.py
from pathlib import Path
import os

class FileSystemSandbox:
    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self._validate_root()
    
    def _validate_root(self):
        """验证根路径安全性"""
        # 禁止危险路径
        forbidden = ["/", "/etc", "/sys", "/proc", "/dev"]
        if str(self.root) in forbidden:
            raise SecurityError(f"禁止使用系统路径: {self.root}")
    
    def resolve_path(self, user_path: str) -> Path:
        """解析用户路径，确保在沙箱内"""
        # 拼接路径
        full_path = (self.root / user_path).resolve()
        
        # 检查是否逃逸
        if not str(full_path).startswith(str(self.root)):
            raise SecurityError(
                f"路径逃逸检测: {user_path} -> {full_path}"
            )
        
        return full_path
    
    def read_file(self, path: str) -> bytes:
        """安全读取文件"""
        safe_path = self.resolve_path(path)
        
        if not safe_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if not safe_path.is_file():
            raise IsADirectoryError(f"不是文件: {path}")
        
        # 检查文件大小
        if safe_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
            raise SecurityError("文件过大")
        
        return safe_path.read_bytes()
    
    def write_file(self, path: str, content: bytes):
        """安全写入文件"""
        safe_path = self.resolve_path(path)
        
        # 确保父目录存在
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查写入大小
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise SecurityError("内容过大")
        
        safe_path.write_bytes(content)
```

### 网络隔离

```python
# network_sandbox.py
from typing import List, Set
import ipaddress
import re

class NetworkSandbox:
    def __init__(self):
        # 允许的目标
        self.allowed_domains: Set[str] = set()
        self.allowed_ips: Set[ipaddress.IPv4Network] = set()
        
        # 禁止的目标（内网）
        self.blocked_networks = [
            ipaddress.IPv4Network("10.0.0.0/8"),
            ipaddress.IPv4Network("172.16.0.0/12"),
            ipaddress.IPv4Network("192.168.0.0/16"),
            ipaddress.IPv4Network("127.0.0.0/8"),
        ]
    
    def allow_domain(self, domain: str):
        """允许域名"""
        self.allowed_domains.add(domain.lower())
    
    def allow_network(self, network: str):
        """允许网段"""
        self.allowed_ips.add(ipaddress.IPv4Network(network))
    
    def check_request(self, url: str) -> bool:
        """检查请求是否允许"""
        from urllib.parse import urlparse
        import socket
        
        parsed = urlparse(url)
        host = parsed.hostname
        
        if not host:
            return False
        
        # 解析 IP
        try:
            ip = ipaddress.IPv4Address(socket.gethostbyname(host))
        except socket.gaierror:
            return False
        
        # 检查是否在黑名单
        for blocked in self.blocked_networks:
            if ip in blocked:
                raise SecurityError(f"禁止访问内网: {host}")
        
        # 检查白名单
        if self.allowed_domains:
            if host.lower() not in self.allowed_domains:
                raise SecurityError(f"域名不在白名单: {host}")
        
        if self.allowed_ips:
            allowed = any(ip in net for net in self.allowed_ips)
            if not allowed:
                raise SecurityError(f"IP 不在白名单: {ip}")
        
        return True
```

## 代码审计

### 静态分析

参考 SonarQube 的规则【5】：

```python
# code_audit.py
import ast
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class SecurityIssue:
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    rule: str
    message: str
    line: int
    column: int

class SecurityAuditor:
    def __init__(self):
        self.issues: List[SecurityIssue] = []
    
    def audit_python(self, code: str) -> List[SecurityIssue]:
        """审计 Python 代码"""
        self.issues = []
        
        try:
            tree = ast.parse(code)
            self._check_ast(tree)
        except SyntaxError:
            pass
        
        self._check_patterns(code)
        
        return self.issues
    
    def _check_ast(self, tree: ast.AST):
        """检查 AST"""
        for node in ast.walk(tree):
            # 检查危险函数调用
            if isinstance(node, ast.Call):
                self._check_dangerous_call(node)
            
            # 检查 exec/eval
            if isinstance(node, ast.Exec):
                self.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    rule="exec-usage",
                    message="禁止使用 exec",
                    line=node.lineno,
                    column=node.col_offset
                ))
    
    def _check_dangerous_call(self, node: ast.Call):
        """检查危险函数调用"""
        dangerous_functions = {
            'eval': '禁止使用 eval',
            'exec': '禁止使用 exec',
            'compile': '禁止使用 compile',
            '__import__': '禁止使用 __import__',
            'globals': '禁止使用 globals',
            'locals': '禁止使用 locals',
        }
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in dangerous_functions:
                self.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    rule=f"{func_name}-usage",
                    message=dangerous_functions[func_name],
                    line=node.lineno,
                    column=node.col_offset
                ))
    
    def _check_patterns(self, code: str):
        """检查正则模式"""
        patterns = [
            (r'os\.system\s*\(', "禁止使用 os.system"),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', "禁止 shell=True"),
            (r'pickle\.loads?\s*\(', "禁止使用 pickle"),
            (r'yaml\.load\s*\((?!.*Loader)', "使用 yaml.safe_load"),
            (r'open\s*\(.*["\']w["\']', "检查文件写入权限"),
        ]
        
        for pattern, message in patterns:
            for match in re.finditer(pattern, code):
                line = code[:match.start()].count('\n') + 1
                self.issues.append(SecurityIssue(
                    severity="HIGH",
                    rule="pattern-match",
                    message=message,
                    line=line,
                    column=0
                ))
```

### 依赖审计

```python
# dependency_audit.py
import json
from typing import Dict, List
from pathlib import Path

class DependencyAuditor:
    def __init__(self, vuln_db_path: str = None):
        self.vuln_db = self._load_vuln_db(vuln_db_path)
    
    def _load_vuln_db(self, path: str) -> Dict:
        """加载漏洞数据库"""
        if path and Path(path).exists():
            with open(path) as f:
                return json.load(f)
        return {}
    
    def audit_dependencies(self, 
                          requirements: Dict[str, str]) -> List[Dict]:
        """审计依赖"""
        issues = []
        
        for package, version in requirements.items():
            if package in self.vuln_db:
                vulns = self.vuln_db[package]
                for vuln in vulns:
                    if self._is_affected(version, vuln['affected']):
                        issues.append({
                            'package': package,
                            'version': version,
                            'vulnerability': vuln['id'],
                            'severity': vuln['severity'],
                            'description': vuln['description'],
                            'fixed_in': vuln.get('fixed_in'),
                            'recommendation': f"升级到 {vuln.get('fixed_in', '最新版本')}"
                        })
        
        return issues
    
    def _is_affected(self, version: str, affected_range: str) -> bool:
        """检查版本是否受影响"""
        from packaging import version as pkg_version
        from packaging.specifiers import SpecifierSet
        
        try:
            spec = SpecifierSet(affected_range)
            return pkg_version.parse(version) in spec
        except:
            return False
```

## 输入验证

### 白名单验证

```python
# input_validator.py
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ValidationRule:
    field: str
    type: str  # string, integer, float, email, url, path
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None

class InputValidator:
    def __init__(self, rules: List[ValidationRule]):
        self.rules = {r.field: r for r in rules}
    
    def validate(self, data: Dict) -> Dict[str, Any]:
        """验证输入数据"""
        errors = []
        validated = {}
        
        for field, rule in self.rules.items():
            value = data.get(field)
            
            # 检查必填
            if rule.required and value is None:
                errors.append(f"{field} 是必填字段")
                continue
            
            if value is None:
                continue
            
            # 类型检查
            if not self._check_type(value, rule.type):
                errors.append(f"{field} 类型错误，期望 {rule.type}")
                continue
            
            # 长度检查
            if rule.min_length and len(str(value)) < rule.min_length:
                errors.append(f"{field} 长度不能小于 {rule.min_length}")
                continue
            
            if rule.max_length and len(str(value)) > rule.max_length:
                errors.append(f"{field} 长度不能大于 {rule.max_length}")
                continue
            
            # 模式检查
            if rule.pattern and not re.match(rule.pattern, str(value)):
                errors.append(f"{field} 格式不正确")
                continue
            
            # 枚举检查
            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(f"{field} 值不在允许范围内")
                continue
            
            validated[field] = value
        
        if errors:
            raise ValidationError(errors)
        
        return validated
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查类型"""
        type_map = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'list': list,
            'dict': dict,
        }
        
        if expected_type == 'email':
            return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(value)))
        
        if expected_type == 'url':
            return bool(re.match(r'^https?://', str(value)))
        
        if expected_type == 'path':
            # 路径安全检查
            return not any(c in str(value) for c in ['..', '~', '$'])
        
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        
        return True

class ValidationError(Exception):
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"验证失败: {'; '.join(errors)}")
```

## 审计日志

### 日志记录

```python
# audit_logger.py
import json
import time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

class AuditLogger:
    def __init__(self, log_dir: str = "/var/log/skill-audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, 
                  event_type: str,
                  skill_name: str,
                  user_id: str,
                  action: str,
                  resource: str,
                  result: str,
                  details: Dict[str, Any] = None):
        """记录审计事件"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'skill_name': skill_name,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'details': details or {}
        }
        
        # 写入日志文件
        log_file = self.log_dir / f"{datetime.now():%Y-%m-%d}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def log_permission_check(self,
                            skill_name: str,
                            user_id: str,
                            resource: str,
                            operation: str,
                            granted: bool):
        """记录权限检查"""
        self.log_event(
            event_type='permission_check',
            skill_name=skill_name,
            user_id=user_id,
            action=operation,
            resource=resource,
            result='granted' if granted else 'denied'
        )
    
    def log_execution(self,
                     skill_name: str,
                     user_id: str,
                     command: str,
                     success: bool,
                     duration: float):
        """记录执行事件"""
        self.log_event(
            event_type='execution',
            skill_name=skill_name,
            user_id=user_id,
            action='execute',
            resource=command,
            result='success' if success else 'failure',
            details={'duration': duration}
        )
    
    def query_events(self,
                    start_time: datetime = None,
                    end_time: datetime = None,
                    event_type: str = None,
                    skill_name: str = None) -> list:
        """查询审计事件"""
        events = []
        
        for log_file in sorted(self.log_dir.glob("*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    event = json.loads(line)
                    
                    # 过滤条件
                    if start_time and event['timestamp'] < start_time.isoformat():
                        continue
                    if end_time and event['timestamp'] > end_time.isoformat():
                        continue
                    if event_type and event['event_type'] != event_type:
                        continue
                    if skill_name and event['skill_name'] != skill_name:
                        continue
                    
                    events.append(event)
        
        return events
```

## 安全配置

### 最小权限原则

```yaml
# skill-security.yaml
security:
  # 默认拒绝所有
  default_policy: deny
  
  # 文件系统
  filesystem:
    read:
      allowed_paths:
        - "/data/*"
        - "/tmp/*"
      denied_paths:
        - "/etc/*"
        - "/root/*"
    write:
      allowed_paths:
        - "/tmp/*"
      max_size_mb: 100
  
  # 网络访问
  network:
    outbound:
      allowed_domains:
        - "*.github.com"
        - "*.python.org"
      allowed_ports: [80, 443]
    inbound:
      enabled: false
  
  # 终端执行
  terminal:
    allowed_commands:
      - "python"
      - "pip"
      - "git"
    denied_commands:
      - "rm"
      - "sudo"
      - "chmod"
    timeout_seconds: 30
  
  # 资源限制
  resources:
    max_memory_mb: 256
    max_cpu_percent: 50
    max_processes: 10
    max_open_files: 100
```

## 最佳实践总结

### 权限控制
- 最小权限原则
- RBAC 角色管理
- 定期权限审计

### 沙箱隔离
- 进程级隔离
- 文件系统限制
- 网络访问控制

### 代码安全
- 静态分析扫描
- 依赖漏洞检查
- 输入验证白名单

### 监控审计
- 全量日志记录
- 异常行为检测
- 定期安全评估

## 参考来源

1. OWASP Top 10 - https://owasp.org/www-project-top-ten/
2. MITRE ATT&CK - https://attack.mitre.org/
3. NIST RBAC - https://csrc.nist.gov/projects/role-based-access-control
4. Docker Security - https://docs.docker.com/engine/security/
5. SonarQube Rules - https://rules.sonarsource.com/

---

*本文首发于 RiceBall-15 的技术博客，转载请注明出处。*
