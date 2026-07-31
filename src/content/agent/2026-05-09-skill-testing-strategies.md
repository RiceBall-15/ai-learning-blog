---
title: "Agent Skill 测试策略：从单元测试到端到端验证"
description: "深入探讨 Agent Skill 的测试方法论，包括单元测试、集成测试、端到端测试，以及测试自动化和覆盖率管理"
date: 2026-05-09
author: RiceBall-15
category: agentSkill
subCategory: agent-skill
tags: ["Agent Skill", "测试", "单元测试", "集成测试", "CI/CD"]
series: agent-skill-dev
seriesOrder: 5
---


## 简介

"代码能跑"和"代码正确"是两回事。没有测试的 Skill 就像没有安全网的高空作业 — 随时可能摔得很惨。本文探讨如何为 Agent Skill 建立完整的测试体系，从单元测试到端到端验证，确保代码质量。

## 问题背景

测试缺失的典型后果：

1. **Bug 频发**：上线后问题不断
2. **回归风险**：改一处坏三处
3. **重构困难**：不敢改代码
4. **协作障碍**：不知道改动是否正确
5. **维护成本高**：手动测试耗时耗力

参考 Martin Fowler 的测试金字塔【1】和 Google 的测试工程实践【2】，我们需要建立系统化的测试策略。

## 测试金字塔

```
            ┌─────────────────┐
            │    E2E 测试      │
            │   (端到端)       │
            │   数量: 少       │
            │   速度: 慢       │
            │   成本: 高       │
            ├─────────────────┤
            │   集成测试       │
            │  (组件协作)      │
            │   数量: 中       │
            │   速度: 中       │
            │   成本: 中       │
            ├─────────────────┤
            │   单元测试       │
            │  (独立函数)      │
            │   数量: 多       │
            │   速度: 快       │
            │   成本: 低       │
            └─────────────────┘
```

### 测试比例建议

| 测试类型 | 比例 | 执行时间 | 目标覆盖率 |
|---------|------|---------|-----------|
| 单元测试 | 70% | < 1ms/个 | 90%+ |
| 集成测试 | 20% | < 100ms/个 | 80%+ |
| E2E 测试 | 10% | < 5s/个 | 关键路径 |

## 单元测试

### 测试框架选择

```python
# 使用 pytest 作为测试框架
# requirements: pytest, pytest-cov, pytest-asyncio, pytest-mock

# conftest.py - 测试配置
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_llm():
    """模拟 LLM 调用"""
    mock = MagicMock()
    mock.generate.return_value = "模拟响应"
    return mock

@pytest.fixture
def sample_skill():
    """示例 Skill 实例"""
    from my_skill import MySkill
    return MySkill()

@pytest.fixture
def temp_dir(tmp_path):
    """临时目录"""
    return tmp_path
```

### 基本单元测试

```python
# test_my_skill.py
import pytest
from my_skill import MySkill, ValidationError, ConfigError

class TestMySkill:
    """MySkill 单元测试"""
    
    def test_initialization(self, sample_skill):
        """测试初始化"""
        assert sample_skill is not None
        assert sample_skill.name == "my-skill"
    
    def test_process_valid_input(self, sample_skill):
        """测试有效输入处理"""
        result = sample_skill.process("有效输入")
        assert result.status == "success"
        assert result.data is not None
    
    def test_process_empty_input(self, sample_skill):
        """测试空输入"""
        with pytest.raises(ValidationError) as exc_info:
            sample_skill.process("")
        assert "输入不能为空" in str(exc_info.value)
    
    def test_process_long_input(self, sample_skill):
        """测试超长输入"""
        long_input = "x" * 10000
        with pytest.raises(ValidationError) as exc_info:
            sample_skill.process(long_input)
        assert "输入过长" in str(exc_info.value)
    
    @pytest.mark.parametrize("input,expected", [
        ("hello", "HELLO"),
        ("world", "WORLD"),
        ("", ValidationError),
        ("123", "123"),
    ])
    def test_process_various_inputs(self, sample_skill, input, expected):
        """参数化测试多种输入"""
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                sample_skill.process(input)
        else:
            result = sample_skill.process(input)
            assert result.data == expected
```

### Mock 和 Stub

```python
# test_with_mocks.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

class TestSkillWithMocks:
    """使用 Mock 的测试"""
    
    @patch('my_skill.external_api_call')
    def test_with_mocked_api(self, mock_api, sample_skill):
        """测试模拟的 API 调用"""
        # 设置 Mock 行为
        mock_api.return_value = {"status": "ok", "data": "mocked"}
        
        result = sample_skill.fetch_data()
        
        # 验证调用
        mock_api.assert_called_once()
        assert result.data == "mocked"
    
    @patch('my_skill.file_system')
    def test_file_operations(self, mock_fs, sample_skill, temp_dir):
        """测试文件操作"""
        # 模拟文件系统
        mock_fs.read_file.return_value = "文件内容"
        mock_fs.write_file.return_value = True
        
        # 执行操作
        content = sample_skill.read_config("config.yaml")
        assert content == "文件内容"
        
        # 验证写入
        sample_skill.save_result({"key": "value"})
        mock_fs.write_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_operation(self, sample_skill):
        """测试异步操作"""
        with patch('my_skill.async_api_call', 
                   new_callable=AsyncMock) as mock_async:
            mock_async.return_value = {"result": "async_data"}
            
            result = await sample_skill.async_process("input")
            
            assert result.data == "async_data"
            mock_async.assert_awaited_once()
```

### 边界条件测试

```python
# test_edge_cases.py
import pytest

class TestEdgeCases:
    """边界条件测试"""
    
    def test_none_input(self, sample_skill):
        """None 输入"""
        with pytest.raises(ValidationError):
            sample_skill.process(None)
    
    def test_unicode_input(self, sample_skill):
        """Unicode 输入"""
        result = sample_skill.process("你好世界 🌍")
        assert result.status == "success"
    
    def test_special_characters(self, sample_skill):
        """特殊字符"""
        special = "<script>alert('xss')</script>"
        result = sample_skill.sanitize(special)
        assert "<script>" not in result
    
    def test_very_large_data(self, sample_skill):
        """大数据量"""
        large_data = [{"id": i, "value": f"item_{i}"} 
                      for i in range(100000)]
        # 确保不会内存溢出
        result = sample_skill.process_batch(large_data)
        assert len(result) == 100000
    
    def test_concurrent_access(self, sample_skill):
        """并发访问"""
        import threading
        
        results = []
        errors = []
        
        def worker():
            try:
                result = sample_skill.process("并发测试")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) 
                   for _ in range(100)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 100
```

## 集成测试

### 组件协作测试

```python
# test_integration.py
import pytest
from my_skill import MySkill
from data_processor import DataProcessor
from output_formatter import OutputFormatter

class TestSkillIntegration:
    """Skill 集成测试"""
    
    @pytest.fixture
    def integrated_system(self):
        """集成系统"""
        return {
            'skill': MySkill(),
            'processor': DataProcessor(),
            'formatter': OutputFormatter()
        }
    
    def test_full_pipeline(self, integrated_system):
        """测试完整流水线"""
        skill = integrated_system['skill']
        processor = integrated_system['processor']
        formatter = integrated_system['formatter']
        
        # 执行完整流程
        raw_data = skill.fetch_data()
        processed = processor.process(raw_data)
        formatted = formatter.format(processed)
        
        # 验证结果
        assert formatted is not None
        assert 'data' in formatted
        assert formatted['status'] == 'success'
    
    def test_error_propagation(self, integrated_system):
        """测试错误传播"""
        skill = integrated_system['skill']
        processor = integrated_system['processor']
        
        # 模拟 processor 失败
        processor.process = MagicMock(
            side_effect=ValueError("处理失败")
        )
        
        # 验证错误正确传播
        with pytest.raises(ValueError) as exc_info:
            skill.execute_full_workflow()
        
        assert "处理失败" in str(exc_info.value)
```

### 外部依赖测试

```python
# test_external_deps.py
import pytest
import responses

class TestExternalDependencies:
    """外部依赖测试"""
    
    @responses.activate
    def test_http_api(self, sample_skill):
        """测试 HTTP API 调用"""
        # 模拟 HTTP 响应
        responses.add(
            responses.GET,
            'https://api.example.com/data',
            json={'result': 'success'},
            status=200
        )
        
        result = sample_skill.fetch_from_api()
        assert result['result'] == 'success'
    
    @responses.activate
    def test_api_retry_on_failure(self, sample_skill):
        """测试 API 失败重试"""
        # 第一次失败，第二次成功
        responses.add(
            responses.GET,
            'https://api.example.com/data',
            json={'error': 'server error'},
            status=500
        )
        responses.add(
            responses.GET,
            'https://api.example.com/data',
            json={'result': 'success'},
            status=200
        )
        
        result = sample_skill.fetch_from_api(retry=2)
        assert result['result'] == 'success'
        assert len(responses.calls) == 2
    
    def test_database_operations(self, sample_skill, tmp_path):
        """测试数据库操作"""
        import sqlite3
        
        # 创建测试数据库
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        conn.commit()
        
        # 测试写入
        sample_skill.save_to_db(conn, {"id": 1, "value": "test"})
        
        # 验证写入
        cursor = conn.execute("SELECT * FROM test WHERE id = 1")
        row = cursor.fetchone()
        assert row == (1, "test")
```

## 端到端测试

### 完整场景测试

```python
# test_e2e.py
import pytest
import subprocess
import json
from pathlib import Path

class TestEndToEnd:
    """端到端测试"""
    
    @pytest.fixture(scope="class")
    def setup_environment(self):
        """设置测试环境"""
        # 启动依赖服务
        subprocess.run(["docker-compose", "up", "-d"], 
                      check=True)
        
        yield
        
        # 清理
        subprocess.run(["docker-compose", "down"], 
                      check=True)
    
    def test_cli_execution(self, setup_environment):
        """测试命令行执行"""
        result = subprocess.run(
            ["python", "-m", "my_skill", 
             "--input", "测试输入",
             "--output", "json"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output['status'] == 'success'
    
    def test_api_endpoint(self, setup_environment):
        """测试 API 端点"""
        import requests
        
        response = requests.post(
            "http://localhost:8000/api/skill",
            json={"input": "测试"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
    
    def test_full_workflow(self, setup_environment):
        """测试完整工作流"""
        # 1. 准备输入
        input_file = Path("test_input.json")
        input_file.write_text(json.dumps({
            "type": "process",
            "data": "测试数据"
        }))
        
        # 2. 执行 Skill
        result = subprocess.run(
            ["python", "-m", "my_skill", 
             "--input-file", str(input_file)],
            capture_output=True,
            text=True
        )
        
        # 3. 验证输出
        assert result.returncode == 0
        
        # 4. 检查输出文件
        output_file = Path("output.json")
        assert output_file.exists()
        
        output = json.loads(output_file.read_text())
        assert output['processed'] is True
```

### 性能测试

```python
# test_performance.py
import pytest
import time
from statistics import mean, stdev

class TestPerformance:
    """性能测试"""
    
    def test_response_time(self, sample_skill):
        """测试响应时间"""
        times = []
        
        for _ in range(100):
            start = time.perf_counter()
            sample_skill.process("性能测试")
            times.append(time.perf_counter() - start)
        
        avg_time = mean(times)
        std_time = stdev(times)
        
        # 响应时间要求
        assert avg_time < 0.1, f"平均响应时间 {avg_time:.3f}s 过长"
        assert std_time < 0.05, f"响应时间波动 {std_time:.3f}s 过大"
    
    def test_throughput(self, sample_skill):
        """测试吞吐量"""
        start = time.perf_counter()
        count = 0
        
        while time.perf_counter() - start < 1.0:
            sample_skill.process("吞吐量测试")
            count += 1
        
        # 吞吐量要求
        assert count >= 100, f"吞吐量 {count}/s 不足"
    
    def test_memory_usage(self, sample_skill):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行大量操作
        for _ in range(1000):
            sample_skill.process("内存测试")
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024
        
        # 内存增长要求
        assert memory_increase < 100, f"内存增长 {memory_increase:.1f}MB 过多"
```

## 测试覆盖率

### 覆盖率配置

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=my_skill
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
```

### 覆盖率报告

```python
# 生成覆盖率报告
# pytest --cov=my_skill --cov-report=html

# 输出示例：
# Name                    Stmts   Miss  Cover   Missing
# -----------------------------------------------------
# my_skill/__init__.py       10      0   100%
# my_skill/core.py          150     12    92%   45-50, 78-82
# my_skill/utils.py          80      5    94%   23, 67-70
# -----------------------------------------------------
# TOTAL                    240     17    93%
```

## 测试自动化

### CI/CD 集成

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        pytest --cov=my_skill --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 预提交检查

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/ -x -q
        language: system
        types: [python]
        pass_filenames: false
      
      - id: coverage
        name: coverage check
        entry: pytest --cov-fail-under=80
        language: system
        types: [python]
        pass_filenames: false
```

## 测试数据管理

### Fixture 管理

```python
# conftest.py
import pytest
import json
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def sample_inputs(test_data_dir):
    """示例输入数据"""
    with open(test_data_dir / "inputs.json") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def expected_outputs(test_data_dir):
    """预期输出数据"""
    with open(test_data_dir / "outputs.json") as f:
        return json.load(f)

@pytest.fixture
def random_input():
    """随机输入"""
    import random
    import string
    return ''.join(random.choices(string.ascii_letters, k=10))
```

### 快照测试

```python
# test_snapshots.py
import pytest
from syrupy import snapshot

class TestSnapshots:
    """快照测试"""
    
    def test_output_format(self, sample_skill, snapshot):
        """测试输出格式（快照）"""
        result = sample_skill.process("快照测试")
        assert result == snapshot
    
    def test_json_structure(self, sample_skill, snapshot):
        """测试 JSON 结构（快照）"""
        result = sample_skill.to_json()
        assert result == snapshot
```

## 最佳实践总结

### 测试原则
- 测试金字塔：单元 > 集成 > E2E
- FIRST 原则：Fast, Independent, Repeatable, Self-validating, Timely
- AAA 模式：Arrange, Act, Assert

### 测试策略
- 先写测试（TDD）
- 覆盖边界条件
- Mock 外部依赖

### 自动化
- CI/CD 集成
- 预提交检查
- 覆盖率门禁

### 维护
- 定期清理过时测试
- 保持测试代码质量
- 测试文档化

## 参考来源

1. Martin Fowler: "Test Pyramid" - https://martinfowler.com/bliki/TestPyramid.html
2. Google Testing Blog - https://testing.googleblog.com/
3. pytest Documentation - https://docs.pytest.org/
4. Python unittest.mock - https://docs.python.org/3/library/unittest.mock.html
5. Test Driven Development - https://en.wikipedia.org/wiki/Test-driven_development

---

*本文首发于 RiceBall-15 的技术博客，转载请注明出处。*
