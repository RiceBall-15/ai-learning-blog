---
title: "AI应用的隐私保护架构：从联邦学习到可信计算的生产实践"
description: "深度解析AI应用中的隐私保护架构设计，涵盖联邦学习、差分隐私、安全多方计算、可信执行环境与数据脱敏技术，附完整架构图与生产级实现方案"
date: 2026-06-01
author: "RiceBall"
category: "architecture"
subCategory: "distributed"
tags: ["隐私计算", "联邦学习", "差分隐私", "可信执行环境", "数据安全", "安全多方计算", "AI安全"]
draft: false
---

# AI应用的隐私保护架构：从联邦学习到可信计算的生产实践

## 隐私合规正在重塑AI架构

2026年，全球数据隐私法规已从"建议合规"演变为"强制执行"。GDPR、CCPA、中国《个人信息保护法》以及新出台的EU AI Act对AI系统提出了前所未有的隐私要求：

```
┌──────────────────────────────────────────────────────────────────┐
│              AI系统面临的隐私挑战全景                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📋 法规要求                                                     │
│  ├── GDPR: 数据最小化、目的限制、存储限制、被遗忘权               │
│  ├── CCPA: 知情同意、数据可携带权、拒绝出售权                     │
│  ├── PIPL: 最小必要、单独同意、跨境传输限制                       │
│  └── EU AI Act: 高风险AI系统必须进行数据保护影响评估             │
│                                                                  │
│  ⚠️  AI系统特殊性                                               │
│  ├── 训练数据：海量个人数据 → 模型可能"记住"敏感信息             │
│  ├── 推理数据：用户输入可能包含隐私信息 → 需要安全处理            │
│  ├── 模型参数：本身可能泄露训练数据信息 → 需要防提取攻击          │
│  └── 日志数据：交互日志包含敏感对话 → 需要脱敏存储               │
│                                                                  │
│  💰 违规代价                                                     │
│  ├── GDPR: 最高2000万欧元或全球营业额4%                          │
│  ├── CCPA: 每次违规最高$7,500                                    │
│  ├── PIPL: 最高5000万元或上一年营业额5%                          │
│  └── 声誉损失：用户信任崩塌，品牌价值受损                        │
└──────────────────────────────────────────────────────────────────┘
```

传统的"收集数据→集中训练→部署模型"模式已经行不通了。AI架构需要在 **数据可用性** 和 **隐私保护** 之间找到平衡点。

## 隐私保护技术栈总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                   AI隐私保护技术栈（从底层到应用层）                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  第4层: 应用层隐私                                                   │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Prompt脱敏  |  输出过滤  |  用户授权管理  |  审计日志     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  第3层: 算法层隐私                                                   │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  联邦学习  |  差分隐私  |  安全多方计算  |  同态加密       │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  第2层: 系统层隐私                                                   │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  TEE可信执行  |  联合推理  |  数据沙箱  |  访问控制        │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  第1层: 基础设施层隐私                                               │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  数据加密  |  密钥管理  |  网络隔离  |  审计追踪           │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## 模式一：联邦学习架构

### 核心思想

联邦学习让数据 **"可用不可见"**——原始数据留在本地，只有模型梯度在各方之间安全交换。

```
┌──────────────────────────────────────────────────────────────────┐
│                    联邦学习工作流程                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌──────────────────┐                          │
│                    │    中心聚合器     │                          │
│                    │  (Server/Aggregator)                        │
│                    └────────┬─────────┘                          │
│                             │                                    │
│              ┌──────────────┼──────────────┐                     │
│              │              │              │                     │
│              ▼              ▼              ▼                     │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│    │  参与方 A     │ │  参与方 B     │ │  参与方 C     │           │
│    │  (医院)       │ │  (医院)       │ │  (医院)       │           │
│    │              │ │              │ │              │           │
│    │ 本地数据      │ │ 本地数据      │ │ 本地数据      │           │
│    │ ↓            │ │ ↓            │ │ ↓            │           │
│    │ 本地训练      │ │ 本地训练      │ │ 本地训练      │           │
│    │ ↓            │ │ ↓            │ │ ↓            │           │
│    │ 上传梯度(加密) │ │ 上传梯度(加密) │ │ 上传梯度(加密) │           │
│    └──────────────┘ └──────────────┘ └──────────────┘           │
│                                                                  │
│  关键特性:                                                       │
│  ✅ 原始数据永远不离开本地                                        │
│  ✅ 只传输模型梯度/参数                                          │
│  ✅ 中心服务器无法推断单个参与者的数据                             │
│  ⚠️  但仍可能通过梯度反推信息（需要额外防护）                      │
└──────────────────────────────────────────────────────────────────┘
```

### 生产级联邦学习系统架构

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import hashlib
import time

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    num_rounds: int = 100               # 联邦轮次
    min_participants: int = 3            # 最少参与方数
    max_participants: int = 50           # 最多参与方数
    rounds_per_aggregation: int = 1      # 聚合频率
    differential_privacy: bool = True    # 是否启用差分隐私
    dp_epsilon: float = 8.0             # 差分隐私隐私预算
    dp_delta: float = 1e-5             # 差分隐私delta
    secure_aggregation: bool = True     # 是否启用安全聚合
    max_staleness: int = 5              # 最大过时轮数
    min_data_ratio: float = 0.01        # 最小数据占比（剔除异常参与方）

@dataclass
class ClientState:
    """参与方状态"""
    client_id: str
    data_size: int
    model_version: int = 0
    last_contribution_round: int = 0
    total_contributions: int = 0
    avg_loss: float = 0.0
    is_active: bool = True
    trust_score: float = 1.0            # 信任分数（用于异常检测）


class FederatedLearningOrchestrator:
    """联邦学习编排器（中心服务器）"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.clients: dict[str, ClientState] = {}
        self.global_model_version = 0
        self.round_history = []
        
    def register_client(self, client_id: str, data_size: int):
        """注册参与方"""
        self.clients[client_id] = ClientState(
            client_id=client_id,
            data_size=data_size
        )
    
    def select_participants(self, round_num: int) -> list[str]:
        """
        参与方选择策略
        
        不是所有参与方每轮都参与，需要考虑：
        1. 数据量（太少的参与方贡献有限）
        2. 历史贡献（长期不参与的需要激活）
        3. 信任分数（异常参与方需要降权）
        4. 系统资源（部分参与方可能离线）
        """
        eligible = [
            c for c in self.clients.values()
            if c.is_active 
            and c.data_size > 0
            and c.trust_score > 0.3
        ]
        
        # 按数据量加权随机选择
        weights = np.array([c.data_size for c in eligible], dtype=float)
        weights = weights / weights.sum()
        
        # 确保最少数目参与
        num_select = min(
            max(self.config.min_participants, len(eligible) // 2),
            self.config.max_participants,
            len(eligible)
        )
        
        selected_indices = np.random.choice(
            len(eligible),
            size=num_select,
            replace=False,
            p=weights
        )
        
        selected = [eligible[i].client_id for i in selected_indices]
        
        # 记录选择信息
        self.round_history.append({
            "round": round_num,
            "selected": selected,
            "total_eligible": len(eligible),
            "timestamp": time.time(),
        })
        
        return selected
    
    def aggregate_gradients(
        self,
        client_updates: dict[str, np.ndarray],
        round_num: int
    ) -> np.ndarray:
        """
        安全聚合客户端梯度
        
        采用加权平均，权重基于各方数据量
        """
        if len(client_updates) < self.config.min_participants:
            raise ValueError(
                f"参与方数不足: {len(client_updates)} < {self.config.min_participants}"
            )
        
        total_weight = 0
        aggregated = None
        
        for client_id, gradient in client_updates.items():
            client = self.clients.get(client_id)
            if not client:
                continue
            
            weight = client.data_size
            
            # 异常值过滤：梯度范数过大可能是投毒攻击
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 10.0:  # 可配置的阈值
                weight *= 0.1  # 降权而不是直接剔除
                client.trust_score *= 0.9
            
            if aggregated is None:
                aggregated = weight * gradient
            else:
                aggregated += weight * gradient
            total_weight += weight
        
        if total_weight == 0:
            raise ValueError("所有参与方权重为0")
        
        aggregated = aggregated / total_weight
        
        # 应用差分隐私噪声
        if self.config.differential_privacy:
            aggregated = self._add_dp_noise(aggregated)
        
        return aggregated
    
    def _add_dp_noise(self, gradient: np.ndarray) -> np.ndarray:
        """添加差分隐私噪声"""
        sensitivity = 1.0  # 梯度敏感度（需要根据实际情况调整）
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.config.dp_delta)) / self.config.dp_epsilon
        noise = np.random.normal(0, sigma, gradient.shape)
        return gradient + noise
    
    def check_convergence(self, round_num: int) -> bool:
        """检查是否收敛"""
        if len(self.round_history) < 10:
            return False
        
        recent_losses = [
            h.get("avg_loss", float('inf'))
            for h in self.round_history[-10:]
        ]
        
        # 损失变化小于阈值
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        return loss_change < 0.001


class FederatedClient:
    """联邦学习参与方"""
    
    def __init__(self, client_id: str, local_data, local_model):
        self.client_id = client_id
        self.local_data = local_data
        self.local_model = local_model
        self.local_epochs = 5
        self.learning_rate = 0.01
    
    def local_train(self, global_model_params: np.ndarray) -> dict:
        """
        本地训练
        
        关键：只上传梯度，不上传数据
        """
        # 加载全局模型参数
        self.local_model.set_params(global_model_params)
        
        # 本地训练
        for epoch in range(self.local_epochs):
            loss = self._train_one_epoch()
        
        # 计算与全局模型的差异（梯度）
        gradient = global_model_params - self.local_model.get_params()
        
        # 梯度裁剪（防止梯度爆炸泄露信息）
        grad_norm = np.linalg.norm(gradient)
        max_norm = 1.0
        if grad_norm > max_norm:
            gradient = gradient * (max_norm / grad_norm)
        
        return {
            "client_id": self.client_id,
            "gradient": gradient,
            "data_size": len(self.local_data),
            "loss": loss,
        }
    
    def _train_one_epoch(self) -> float:
        """训练一个epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch in self.local_data:
            # 前向传播
            output = self.local_model.forward(batch["input"])
            loss = self._compute_loss(output, batch["label"])
            
            # 反向传播
            gradient = self.local_model.backward(loss)
            
            # 梯度裁剪
            gradient = np.clip(gradient, -1.0, 1.0)
            
            # 参数更新
            self.local_model.update(gradient, self.learning_rate)
            
            total_loss += loss
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _compute_loss(self, output, label) -> float:
        """计算损失（简化实现）"""
        return float(np.mean((output - label) ** 2))
```

### 联邦学习的安全增强

联邦学习的主要威胁是 **梯度反推攻击**——通过分析梯度值推断训练数据。需要多层防护：

```
┌──────────────────────────────────────────────────────────────────┐
│               联邦学习安全防护体系                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  威胁1: 梯度反推攻击                                             │
│  ├─ 攻击方式：分析梯度值推断训练样本特征                          │
│  ├─ 防护措施：                                                   │
│  │   ├── 差分隐私：在梯度中添加校准噪声                          │
│  │   ├── 安全聚合：使用同态加密聚合梯度                          │
│  │   └── 梯度稀疏化：只上传Top-K梯度分量                         │
│  └─ 推荐方案：差分隐私 + 安全聚合（双重防护）                     │
│                                                                  │
│  威胁2: 投毒攻击                                                 │
│  ├─ 攻击方式：恶意参与方上传恶意梯度，破坏全局模型                 │
│  ├─ 防护措施：                                                   │
│  │   ├── 梯度范数异常检测                                        │
│  │   ├── 历史行为信誉评分                                        │
│  │   ├── Krum/Trimmed Mean鲁棒聚合                               │
│  │   └─ 参与方准入门槛（数据量、历史贡献）                        │
│  └─ 推荐方案：多维度异常检测 + 鲁棒聚合                           │
│                                                                  │
│  威胁3: 成员推断攻击                                             │
│  ├─ 攻击方式：判断某条数据是否参与了训练                          │
│  ├─ 防护措施：                                                   │
│  │   ├── 差分隐私（降低过拟合，减少推断可能）                     │
│  │   ├── 正则化和Dropout                                        │
│  │   └── 模型蒸馏（降低模型对训练数据的记忆）                     │
│  └─ 推荐方案：差分隐私 + 模型蒸馏                                 │
│                                                                  │
│  威胁4: 中心服务器不可信                                         │
│  ├─ 攻击方式：中心服务器尝试推断各参与方的数据                     │
│  ├─ 防护措施：                                                   │
│  │   ├── 安全聚合协议（服务器只看到聚合结果）                     │
│  │   ├── 可信执行环境（TEE）                                     │
│  │   └── 去中心化联邦学习（无中心服务器）                         │
│  └─ 推荐方案：安全聚合 + TEE                                     │
└──────────────────────────────────────────────────────────────────┘
```

## 模式二：差分隐私架构

### 差分隐私在AI系统中的应用

差分隐私提供了一种 **数学可证明** 的隐私保证：无论某条数据是否在数据集中，对输出结果的影响都微乎其微。

```
┌──────────────────────────────────────────────────────────────────┐
│                 差分隐私在AI系统中的三个应用点                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  应用点1: 训练阶段（DP-SGD）                                    │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  标准SGD:                                               │     │
│  │  θ ← θ - η · (1/B) Σ ∇L(xᵢ, θ)                       │     │
│  │                                                         │     │
│  │  DP-SGD:                                                │     │
│  │  1. 计算每个样本的梯度 gᵢ = ∇L(xᵢ, θ)                  │     │
│  │  2. 裁剪: g̃ᵢ = gᵢ · min(1, C/‖gᵢ‖)                   │     │
│  │  3. 聚合: G = (1/B) Σ g̃ᵢ + N(0, σ²C²I)                 │     │
│  │                                                         │     │
│  │  C = 裁剪阈值（控制单个样本的影响）                       │     │
│  │  σ = 噪声倍率（控制隐私预算）                             │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  应用点2: 推理阶段（本地差分隐私LDP）                            │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  用户输入 → 本地扰动 → 服务器处理 → 返回结果             │     │
│  │                                                         │     │
│  │  扰动方式（按数据类型）：                                 │     │
│  │  - 数值数据: 拉普拉斯机制                                │     │
│  │  - 分类数据: 随机响应                                    │     │
│  │  - 文本数据: Token级随机替换                              │     │
│  │  - 向量数据: 高斯机制                                    │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  应用点3: 查询阶段（中心差分隐私CDP）                            │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  数据分析查询 → 查询结果 + 噪声 → 返回结果               │     │
│  │                                                         │     │
│  │  适用于: 统计分析、模型评估、数据报表                      │     │
│  │  优势: 不改变模型，只需在查询结果上加噪声                  │     │
│  └────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### DP-SGD 实现

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class DPSGDOptimizer:
    """差分隐私SGD优化器"""
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        noise_multiplier: float = 1.0,    # σ: 噪声倍率
        max_grad_norm: float = 1.0,        # C: 裁剪阈值
        batch_size: int = 64,
        dataset_size: int = 50000,
        epochs: int = 10,
        delta: float = 1e-5,
    ):
        self.model = model
        self.lr = lr
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.epochs = epochs
        self.delta = delta
        
        # 计算隐私预算
        self.epsilon = self._compute_epsilon()
    
    def _compute_epsilon(self) -> float:
        """计算隐私预算ε（使用Rényi Differential Privacy）"""
        # 简化的计算，实际应使用privacy_accountant
        steps = self.epochs * (self.dataset_size // self.batch_size)
        epsilon = self.noise_multiplier * np.sqrt(2 * steps * np.log(1/self.delta))
        return epsilon
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch（DP-SGD）"""
        total_loss = 0
        num_batches = 0
        
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Step 1: 计算每个样本的梯度
            per_sample_grads = []
            
            for i in range(data.size(0)):
                self.model.zero_grad()
                output = self.model(data[i:i+1])
                loss = nn.functional.cross_entropy(output, target[i:i+1])
                loss.backward()
                
                # 收集单个样本的梯度
                sample_grad = torch.cat([
                    p.grad.view(-1) for p in self.model.parameters()
                ])
                per_sample_grads.append(sample_grad)
            
            per_sample_grads = torch.stack(per_sample_grads)
            
            # Step 2: 梯度裁剪（按样本）
            grad_norms = torch.norm(per_sample_grads, dim=1, keepdim=True)
            clip_factors = torch.clamp(
                self.max_grad_norm / (grad_norms + 1e-8),
                max=1.0
            )
            clipped_grads = per_sample_grads * clip_factors
            
            # Step 3: 聚合 + 添加噪声
            mean_grad = clipped_grads.mean(dim=0)
            noise = torch.normal(
                0,
                self.noise_multiplier * self.max_grad_norm,
                size=mean_grad.shape
            )
            noisy_grad = mean_grad + noise
            
            # Step 4: 更新模型参数
            idx = 0
            for param in self.model.parameters():
                param_size = param.numel()
                param.grad = noisy_grad[idx:idx+param_size].view(param.shape)
                idx += param_size
            
            torch.optim.SGD(self.model.parameters(), lr=self.lr).step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)


class PrivateInferenceEngine:
    """隐私保护推理引擎"""
    
    def __init__(self, model, epsilon: float = 4.0):
        self.model = model
        self.epsilon = epsilon
        self.tokenizer = None  # 假设已初始化
    
    def private_inference(self, user_input: str) -> dict:
        """
        本地差分隐私推理
        
        在用户端对输入进行扰动，服务器只看到扰动后的输入
        """
        # Step 1: Token化
        tokens = self.tokenizer.encode(user_input)
        
        # Step 2: 本地扰动（随机响应）
        perturbed_tokens = self._local_perturbation(tokens)
        
        # Step 3: 推理（服务器端，看到的是扰动后的输入）
        output = self.model.generate(perturbed_tokens)
        
        # Step 4: 输出后处理（去噪/过滤敏感信息）
        safe_output = self._post_process(output)
        
        return {
            "output": safe_output,
            "privacy_guarantee": {
                "epsilon": self.epsilon,
                "mechanism": "local_dp_random_response",
                "input_perturbed": True,
            }
        }
    
    def _local_perturbation(self, tokens: list[int]) -> list[int]:
        """本地差分隐私扰动"""
        vocab_size = self.tokenizer.vocab_size
        perturbed = []
        
        for token in tokens:
            if np.random.random() < self.epsilon / (self.epsilon + 2):
                # 以高概率保持原始token
                perturbed.append(token)
            else:
                # 以低概率替换为随机token
                random_token = np.random.randint(0, vocab_size)
                perturbed.append(random_token)
        
        return perturbed
    
    def _post_process(self, output: str) -> str:
        """输出后处理：过滤敏感信息"""
        # 1. 移除可能的PII（个人身份信息）
        output = self._remove_pii(output)
        
        # 2. 检查是否泄露了输入中的敏感信息
        output = self._check_leakage(output)
        
        return output
    
    def _remove_pii(self, text: str) -> str:
        """移除个人身份信息"""
        import re
        
        # 电话号码
        text = re.sub(r'1[3-9]\d{9}', '[电话已脱敏]', text)
        
        # 身份证号
        text = re.sub(r'\d{17}[\dXx]', '[身份证已脱敏]', text)
        
        # 邮箱
        text = re.sub(r'[\w.]+@[\w.]+', '[邮箱已脱敏]', text)
        
        return text
    
    def _check_leakage(self, output: str) -> str:
        """检查是否泄露了输入中的敏感信息"""
        # 实际生产中需要更复杂的检测逻辑
        return output
```

## 模式三：安全多方计算与同态加密

### 技术对比

```
┌──────────────────────────────────────────────────────────────────┐
│              安全计算技术对比                                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  技术              隐私保证    性能      适用场景                  │
│  ─────────────────────────────────────────────────────────────   │
│  差分隐私          数学证明    ⚡⚡⚡⚡   统计查询/模型训练          │
│  安全多方计算      信息论/密码 ⚡⚡      联合统计/联合建模          │
│  同态加密          密码学      ⚡        安全推理/加密数据计算      │
│  可信执行环境      硬件隔离    ⚡⚡⚡     通用安全计算              │
│  秘密分享          密码学      ⚡⚡⚡     分布式安全计算            │
│                                                                  │
│  选择建议:                                                       │
│  ├── 需要数学可证明的隐私 → 差分隐私                              │
│  ├── 多方协作计算 → 安全多方计算 / 秘密分享                       │
│  ├── 需要在加密数据上直接计算 → 同态加密                          │
│  ├── 通用安全计算需求 → TEE                                       │
│  └── 成本敏感 → 差分隐私 + TEE 组合                               │
└──────────────────────────────────────────────────────────────────┘
```

### 同态加密在AI推理中的应用

```python
# 同态加密推理示意（使用TenSEAL库的概念实现）

class HomomorphicInferenceEngine:
    """
    同态加密推理引擎
    
    允许在加密数据上直接进行模型推理，
    服务器全程看不到明文数据
    """
    
    def __init__(self):
        # 同态加密参数
        self.poly_modulus_degree = 8192
        self.coeff_mod_bit_sizes = [60, 40, 40, 60]
        self.global_scale = 2**40
        
    def encrypt_input(self, plaintext_vector: list[float]) -> 'CKKSEncryptedVector':
        """
        加密输入向量
        
        CKKS方案支持浮点数运算，适合AI推理
        """
        # 实际使用TenSEAL/Palisade等库
        # encrypted = tenseal.ckks_vector(self.context, plaintext_vector)
        encrypted = {"type": "ckks_vector", "data": plaintext_vector}
        return encrypted
    
    def secure_inference(self, encrypted_input, model_params: dict) -> dict:
        """
        在加密数据上执行推理
        
        核心操作：
        1. 矩阵乘法（密文 × 密文权重）
        2. 加法（密文 + 密文偏置）
        3. 激活函数（多项式近似）
        """
        # 模型参数也需要加密
        encrypted_weights = self._encrypt_model_weights(model_params)
        
        # 执行加密推理（在密文空间运算）
        encrypted_output = self._encrypted_linear_layer(
            encrypted_input, 
            encrypted_weights
        )
        
        return {
            "encrypted_output": encrypted_output,
            "metadata": {
                "scheme": "CKKS",
                "security_level": 128,
                "supports_float": True,
            }
        }
    
    def decrypt_output(self, encrypted_output) -> list[float]:
        """解密输出（只有数据所有者可以解密）"""
        # 实际使用TenSEAL
        # plaintext = encrypted_output.decrypt()
        plaintext = encrypted_output.get("data", [])
        return plaintext
    
    def _encrypted_linear_layer(self, encrypted_input, encrypted_weights):
        """加密线性层（矩阵乘法 + 偏置加法）"""
        # 在实际TenSEAL中，这会自动在密文上执行
        # result = encrypted_input @ encrypted_weights["W"] + encrypted_weights["b"]
        pass
    
    def _encrypt_model_weights(self, model_params: dict) -> dict:
        """加密模型权重"""
        # 模型权重可以在密钥持有者处加密后发送到服务器
        pass
```

### 安全多方计算（MPC）架构

```
┌──────────────────────────────────────────────────────────────────┐
│              安全多方计算联合推理架构                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  场景: 两家医院联合使用AI模型分析患者数据，互不暴露患者信息          │
│                                                                  │
│  ┌──────────────┐                         ┌──────────────┐      │
│  │  医院A        │                         │  医院B        │      │
│  │  (有数据Xa)   │                         │  (有数据Xb)   │      │
│  │  (有模型Wa)   │                         │  (有模型Wb)   │      │
│  └──────┬───────┘                         └──────┬───────┘      │
│         │                                        │               │
│         │        秘密分享 (Secret Sharing)        │               │
│         ├──←────────────────────────────────────→──│              │
│         │                                        │               │
│         ▼                                        ▼               │
│  ┌──────────────┐                         ┌──────────────┐      │
│  │  本地计算      │                         │  本地计算      │      │
│  │  分片梯度      │                         │  分片梯度      │      │
│  └──────┬───────┘                         └──────┬───────┘      │
│         │                                        │               │
│         └──────────→──┐    ┌──←──────────────────┘               │
│                       ▼    ▼                                     │
│              ┌──────────────────┐                                │
│              │   计算协调器       │                                │
│              │  (不接触明文)     │                                │
│              │                  │                                │
│              │  合并分片 → 推理结果│                                │
│              └──────────────────┘                                │
│                       │                                          │
│                       ▼                                          │
│              ┌──────────────────┐                                │
│              │  联合推理结果      │                                │
│              │  (双方共同解密)   │                                │
│              └──────────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

## 模式四：可信执行环境（TEE）

### TEE在AI推理中的应用

```
┌──────────────────────────────────────────────────────────────────┐
│              TEE隐私保护AI推理架构                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                 普通执行环境 (REE)                     │       │
│  │                                                       │       │
│  │  ┌─────────┐    ┌─────────────┐    ┌──────────┐     │       │
│  │  │  用户    │───→│  API Gateway │───→│  路由器   │     │       │
│  │  └─────────┘    └─────────────┘    └────┬─────┘     │       │
│  │                                         │            │       │
│  └─────────────────────────────────────────┼────────────┘       │
│                                            │                    │
│                            ┌───────────────┤                    │
│                            ▼               │                    │
│  ┌─────────────────────────────────────────┼────────────┐      │
│  │            可信执行环境 (TEE/SGX)         │            │      │
│  │                                         │            │      │
│  │  ┌─────────────┐   ┌──────────────────┐ │            │      │
│  │  │  安全飞地     │   │  加密模型加载     │ │            │      │
│  │  │  (Enclave)  │   │                  │ │            │      │
│  │  │             │◄──│  远程认证证明     │ │            │      │
│  │  │  推理执行    │   │  (Remote Attestation)          │      │
│  │  │             │   └──────────────────┘ │            │      │
│  │  │  数据解密    │                        │            │      │
│  │  │  模型推理    │   ┌──────────────────┐ │            │      │
│  │  │  结果加密    │   │  密钥管理服务     │ │            │      │
│  │  │             │──→│  (KMS)           │ │            │      │
│  │  └─────────────┘   └──────────────────┘ │            │      │
│  │                                          │            │      │
│  └──────────────────────────────────────────┼────────────┘      │
│                                              │                  │
│                                              ▼                  │
│                                     加密结果 → 用户解密           │
│                                                                  │
│  关键特性:                                                       │
│  ✅ 内存加密：TEE内数据始终加密，即使操作系统被攻破               │
│  ✅ 远程认证：客户端可验证TEE环境的真实性                         │
│  ✅ 密钥隔离：TEE外无法访问密钥                                  │
│  ⚠️  性能开销：约10-30%（内存加密/解密）                         │
└──────────────────────────────────────────────────────────────────┘
```

```python
class TEEInferenceService:
    """
    基于TEE的安全推理服务
    
    使用Intel SGX或AMD SEV技术保护推理过程
    """
    
    def __init__(self, tee_config: dict):
        self.tee_config = tee_config
        self.enclave_id = None
        self.attestation_done = False
    
    async def initialize_enclave(self) -> bool:
        """初始化安全飞地"""
        # 1. 创建Enclave
        # self.enclave_id = sgx_create_enclave(self.tee_config["enclave_path"])
        
        # 2. 生成密钥对
        # key_pair = sgx_generate_key_pair(self.enclave_id)
        
        # 3. 远程认证
        # attestation = sgx_remote_attestation(self.enclave_id)
        
        self.attestation_done = True
        return True
    
    async def secure_inference(
        self, 
        encrypted_data: bytes,
        model_id: str
    ) -> dict:
        """
        安全推理流程
        
        1. 数据以加密形式传入TEE
        2. TEE内部解密数据
        3. 使用受保护的模型进行推理
        4. 结果加密后返回
        """
        if not self.attestation_done:
            raise RuntimeError("TEE未完成远程认证")
        
        # 1. 验证数据完整性
        # if not sgx_verify_data_integrity(encrypted_data):
        #     raise SecurityError("数据完整性验证失败")
        
        # 2. 在Enclave内解密并推理
        # plaintext_data = sgx_decrypt(self.enclave_id, encrypted_data)
        # model = sgx_load_model(self.enclave_id, model_id)
        # result = model.inference(plaintext_data)
        # encrypted_result = sgx_encrypt(self.enclave_id, result)
        
        # 简化示意
        result = {"prediction": "positive", "confidence": 0.92}
        encrypted_result = self._encrypt_result(result)
        
        return {
            "encrypted_output": encrypted_result,
            "attestation": {
                "tee_type": self.tee_config.get("tee_type", "SGX"),
                "security_level": "L3",
                "integrity_verified": True,
            }
        }
    
    def _encrypt_result(self, result: dict) -> bytes:
        """加密推理结果"""
        import json
        import base64
        # 实际应使用TEE内的密钥加密
        return base64.b64encode(json.dumps(result).encode())
```

## 生产部署：隐私保护AI系统的架构模式

### 综合架构

```
┌──────────────────────────────────────────────────────────────────────┐
│           隐私保护AI系统 - 生产级综合架构                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  用户层                                                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  用户设备                                                   │     │
│  │  ├── 本地差分隐私扰动                                       │     │
│  │  ├── 端侧推理（敏感数据不出设备）                            │     │
│  │  └── 加密传输到服务端                                       │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  网关层                                                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  API Gateway + 隐私策略引擎                                 │     │
│  │  ├── 用户授权验证                                          │     │
│  │  ├── 数据分类标签提取                                       │     │
│  │  ├── 隐私策略路由（决定保护级别）                            │     │
│  │  └── 审计日志记录                                          │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│  处理层（按隐私级别路由）                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ 低敏感度       │ │ 中敏感度      │ │ 高敏感度      │                │
│  │ 标准推理       │ │ TEE推理      │ │ 联邦学习      │                │
│  │              │ │              │ │              │                │
│  │ 普通LLM      │ │ 加密数据      │ │ 数据不出域    │                │
│  │ 标准结果      │ │ 安全飞地      │ │ 本地训练      │                │
│  │              │ │ 可信计算      │ │ 梯度交换      │                │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
│                              │                                      │
│                              ▼                                      │
│  存储层                                                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  数据存储策略                                               │     │
│  │  ├── 原始数据: 加密存储 + 访问控制 + 自动过期               │     │
│  │  ├── 脱敏数据: 可用于模型训练和分析                         │     │
│  │  ├── 日志数据: 差分隐私保护 + 审计追踪                      │     │
│  │  └── 模型数据: 模型水印 + 访问控制 + 版本管理               │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  监控层                                                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  隐私合规监控                                               │     │
│  │  ├── 隐私预算追踪（差分隐私ε使用情况）                       │     │
│  │  ├── 数据访问审计                                           │     │
│  │  ├── 异常行为检测                                           │     │
│  │  └── 合规报告自动生成                                       │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### 数据分级与隐私策略

```python
from enum import Enum
from dataclasses import dataclass

class DataSensitivity(Enum):
    """数据敏感度分级"""
    PUBLIC = "public"           # 公开数据（无限制）
    INTERNAL = "internal"       # 内部数据（基本访问控制）
    CONFIDENTIAL = "confidential"  # 机密数据（加密 + 审计）
    RESTRICTED = "restricted"   # 受限数据（最高级别保护）

class PrivacyStrategy:
    """隐私保护策略配置"""
    
    STRATEGIES = {
        DataSensitivity.PUBLIC: {
            "encryption": False,
            "differential_privacy": False,
            "tee_required": False,
            "federated": False,
            "retention_days": 365,
            "audit_level": "basic",
        },
        DataSensitivity.INTERNAL: {
            "encryption": True,
            "differential_privacy": False,
            "tee_required": False,
            "federated": False,
            "retention_days": 180,
            "audit_level": "standard",
        },
        DataSensitivity.CONFIDENTIAL: {
            "encryption": True,
            "differential_privacy": True,
            "dp_epsilon": 8.0,
            "tee_required": True,
            "federated": False,
            "retention_days": 90,
            "audit_level": "detailed",
        },
        DataSensitivity.RESTRICTED: {
            "encryption": True,
            "differential_privacy": True,
            "dp_epsilon": 4.0,
            "tee_required": True,
            "federated": True,
            "retention_days": 30,
            "audit_level": "comprehensive",
        },
    }
    
    @classmethod
    def get_strategy(cls, sensitivity: DataSensitivity) -> dict:
        return cls.STRATEGIES[sensitivity]


class DataClassifier:
    """数据分类器：自动判断数据敏感度"""
    
    SENSITIVE_PATTERNS = {
        DataSensitivity.RESTRICTED: [
            r'\d{17}[\dXx]',           # 身份证号
            r'\d{16}',                  # 银行卡号
            r'密码|password|pwd',       # 密码
        ],
        DataSensitivity.CONFIDENTIAL: [
            r'1[3-9]\d{9}',            # 手机号
            r'[\w.]+@[\w.]+',           # 邮箱
            r'地址|addr|address',        # 地址
            r'收入|薪资|salary',         # 收入信息
        ],
        DataSensitivity.INTERNAL: [
            r'员工编号|工号|emp_id',     # 员工信息
            r'内部|internal|confidential',  # 标记信息
        ],
    }
    
    def classify(self, text: str) -> DataSensitivity:
        """对文本进行敏感度分类"""
        import re
        
        for sensitivity, patterns in self.SENSITIVE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return sensitivity
        
        return DataSensitivity.PUBLIC
    
    def get_applicable_strategies(self, text: str) -> dict:
        """获取适用的隐私策略"""
        sensitivity = self.classify(text)
        return PrivacyStrategy.get_strategy(sensitivity)
```

## 实施路线图

```
┌──────────────────────────────────────────────────────────────────┐
│           隐私保护AI系统实施路线图                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: 基础合规（1-2个月）                                    │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  □ 数据分类分级体系建立                                   │     │
│  │  □ 基本的数据加密（传输 + 存储）                          │     │
│  │  □ 访问控制和审计日志                                    │     │
│  │  □ Prompt脱敏和输出过滤                                  │     │
│  │  □ 隐私政策和用户授权流程                                │     │
│  │  目标: 满足基本合规要求                                  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Phase 2: 高级保护（3-4个月）                                    │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  □ 差分隐私集成（DP-SGD + LDP）                          │     │
│  │  □ TEE安全推理部署                                      │     │
│  │  □ 自动化PII检测和脱敏                                   │     │
│  │  □ 隐私预算追踪系统                                      │     │
│  │  □ 隐私保护的模型评估框架                                │     │
│  │  目标: 实现高级隐私保护                                  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Phase 3: 生态协同（5-6个月）                                    │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  □ 联邦学习平台搭建                                     │     │
│  │  □ 跨组织安全计算网络                                    │     │
│  │  □ 隐私计算即服务（PCaaS）                               │     │
│  │  □ 合规自动化审计                                        │     │
│  │  □ 隐私保护的持续集成/部署                               │     │
│  │  目标: 构建隐私计算生态                                  │     │
│  └────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

## 技术选型建议

```
┌──────────────────────────────────────────────────────────────────┐
│           隐私保护技术选型决策树                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  你的场景是什么？                                                 │
│  │                                                               │
│  ├── 需要多方协作训练模型？                                       │
│  │   ├── 参与方互不信任 → 联邦学习 + 安全聚合                     │
│  │   ├── 参与方部分信任 → 联邦学习 + 差分隐私                     │
│  │   └── 需要精确计算 → 安全多方计算                              │
│  │                                                               │
│  ├── 需要保护推理数据？                                           │
│  │   ├── 用户端保护 → 本地差分隐私                               │
│  │   ├── 服务端保护 → TEE安全飞地                                │
│  │   └── 两端都要 → LDP + TEE                                    │
│  │                                                               │
│  ├── 需要在加密数据上计算？                                       │
│  │   ├── 简单运算 → 同态加密（BFV/CKKS）                         │
│  │   ├── 复杂运算 → TEE                                          │
│  │   └── 混合场景 → 同态加密 + TEE                               │
│  │                                                               │
│  └── 预算有限？                                                   │
│      ├── 最低成本 → 差分隐私（纯软件方案）                         │
│      ├── 中等预算 → 差分隐私 + 联邦学习                           │
│      └── 充足预算 → TEE + 安全多方计算                            │
│                                                                  │
│  参考实现:                                                       │
│  ├── 差分隐私: PySyft, OpenDP, Google DP Library                 │
│  ├── 联邦学习: FedML, Flower, PySyft                              │
│  ├── 安全多方计算: MP-SPDZ, CrypTen                               │
│  ├── 同态加密: TenSEAL, Microsoft SEAL                           │
│  └── TEE: Intel SGX, AMD SEV, ARM TrustZone                     │
└──────────────────────────────────────────────────────────────────┘
```

## 总结

隐私保护架构不是AI系统的"附加功能"，而是 **必须内建的核心能力**。随着法规趋严和用户隐私意识提升，没有隐私保护的AI系统将面临巨大的法律和商业风险。

核心原则：
- **数据最小化**：只收集必要的数据
- **用途限制**：数据只用于声明的目的
- **安全保障**：全链路加密和访问控制
- **透明可审计**：所有数据处理可追踪

技术选择：
- **简单场景**：差分隐私 + 输出过滤（低成本，高回报）
- **多方协作**：联邦学习 + 安全聚合（数据不动，模型动）
- **高敏感场景**：TEE + 同态加密（硬件级安全保障）
- **综合方案**：分层分级，按敏感度选择不同保护策略

记住：**隐私保护的最佳时机是架构设计阶段，而不是上线之后。**

---

*参考资料：*
- *Google: "Differential Privacy for Deep Learning"*
- *Intel: "Intel SGX Developer Guide"*
- *OpenMined: "Privacy preserving machine learning"*
- *Flower: "A Friendly Federated Learning Framework"*
- *EU AI Act: "High-Risk AI System Requirements"*
