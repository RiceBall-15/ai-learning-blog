---
title: "Agent 面试深挖（七）：模型网关——Token 维度限流、熔断降级与潮汐弹性"
description: "Agent 的成本与稳定性瓶颈都在模型出口。本文讲清 LLM 限流为什么比 QPS 限流难（调用前不知道输出多少 token）、令牌桶 reconcile 为什么会泄漏、熔断只看错误率会漏掉什么、降级链如何避免级联击穿，以及潮汐扩缩容为什么必须真正调释放 API。"
date: 2026-07-31
author: "技术学习笔记"
category: interview
subCategory: gateway-cost
series: agent-interview-deep
seriesOrder: 7
tags: ["Agent", "模型网关", "限流熔断", "成本优化", "弹性伸缩", "面试"]
---

# 模型网关：Token 维度限流、熔断降级与潮汐弹性

> Agent 的成本与稳定性瓶颈**都在模型出口**。一次用户开口可能打 1–3 次模型、0–2 次工具、1 次召回，而且流式生成还占着连接。
>
> 这一篇讲「模型出口收费站」怎么建，以及成本治理里最容易踩空的一脚。

## 一、为什么需要网关

朴素做法：每个节点直接 `new XxxClient().chat(...)`。问题：

- **绑死供应商**：换厂商要改所有节点
- **无保护**：一个节点打爆配额，重试风暴把供应商打 429，雪崩
- **无降级**：供应商挂了，业务硬失败
- **无观测**：不知道首包多少、token 花在哪、哪个模型贵
- **成本失控**：外采推理资源高峰不够、低谷空转烧钱，没人管

所以需要在业务节点和供应商之间加一层**网关**，统一处理多供应商适配、限流、熔断、降级、观测、成本治理。

调用链从需求反推出来：

```text
业务节点 → 网关
  → 选供应商 Client（流式/非流式适配）
  → Limit 门禁（并发 / QPS / Token）
  → Breaker 熔断检查
  → （可选）缓存
  → 调用（支持取消）
  → 失败走 Degrade 降级链
  → 打点（首包 TTFT / Token / 总耗时）
```

## 二、LLM 限流为什么比普通限流难

普通限流按 QPS / 并发就够了。但 LLM 的真正瓶颈常是 **tokens/sec**（GPU 显存与算力），不是 QPS。

**难点**：**调用前你不知道输出会生成多少 token。**

解法是「**估算准入 + 事后校正**」：

```text
// 准入：保守上界占坑
estimate = inputTokens + req.maxOutput
if not bucket.tryAcquire(estimate): return degrade(LIMIT)
req.reservation = estimate

try:
  resp = client.stream(req, cancelToken)
finally:
  // ★ 无论成功/失败/取消都要回填，否则桶泄漏
  real   = resp?.usage.outputTokens ?? 0
  actual = inputTokens + real
  delta  = req.reservation - actual     // 估多了归还、估少了追扣
  bucket.release(delta)
  metrics.record(estErr = actual - req.reservation)   // 观测估算误差，反哺默认值
```

### 这里有个高频生产事故：令牌桶「假满」

`reconcile` 必须在 `finally` 里——**异常和取消路径是漏回填的高发区**。漏一次，桶就永久少一格；累积下去桶「假满」，明明 QPS 没涨却全线限流，而且越跑越严重、当场极难定位。

再配一个**定时对账**（按真实 usage 重算桶）兜底累积误差。

### 为什么三个维度都要限

```text
只限 QPS  → 一次长输出请求吃满 GPU 算力，QPS 没超但 token 打爆
只限并发  → 并发槽满了但每个请求都很轻，吞吐被低估
只限 Token → 调用前不知道输出，只能估算 + 事后 reconcile（必要但不充分）
→ 三维分开限，互不挤占
```

### Redis Lua 原子判定（防超卖）

超卖的根因是「读-判-写」竞态：两个请求同时读到 `cur=99`，都判 `99+1≤100` 通过，都写 100。解法是一次 Lua 原子判定：

```lua
-- KEYS[1]=桶key  ARGV[1]=limit  ARGV[2]=window_ms  ARGV[3]=cost
local cur = tonumber(redis.call('GET', KEYS[1]) or '0')
if cur + tonumber(ARGV[3]) > tonumber(ARGV[1]) then
  return 0                         -- 拒绝
end
redis.call('INCRBY', KEYS[1], ARGV[3])
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1                           -- 放行
```

固定窗口有「窗口交界瞬时两倍流量」的突刺；要更平滑就上滑动窗口（Lua 里维护时间戳有序集），但内存与操作更重。**易被突刺打爆的高价值模型用滑动，普通的用固定窗口。**

## 三、熔断：只看错误率会漏掉什么

```text
CLOSED --(错误率 > 阈值)--> OPEN --(冷却到期)--> HALF_OPEN
HALF_OPEN --(单飞探测成功)--> CLOSED
HALF_OPEN --(探测失败)--> OPEN

关键：半开只放单飞探测 —— 防恢复期雷鸣群发
     （否则刚恢复就被全量探测流量又打挂）
OPEN 强制走 Degrade 链
```

### 深挖点：慢但不报错

这是只看错误率必然漏掉的场景——**供应商偶发响应极慢但不返回错误**，错误率没超阈值，熔断不触发，但对话首包全线拉长，体验已经崩了。

**修复**：熔断维度不止错误率，还要加：

- **P99 延迟维度**（延迟超阈值也计入「坏」）
- **超时即判失败**（超时算进错误率）

只看错误率是新手写法。

## 四、降级链：防级联击穿

主模型挂了，全部流量瞬间压给降级目标，把备份也打成 429——这是很常见的二次事故。降级链需要**五道护栏**：

```text
degrade(req, reason):
  chain = req.degradeChain      // 例: [主力模型 → 小模型 → 自研 → 静态兜底]
  for (i, fb) in chain[req.currentTier+1 :]:
    if breaker.isOpen(fb):   continue      // ① 跳过已熔断的降级目标
    if fb == req.model:      continue      // ② 防降级回自己
    if i > maxDegradeDepth:  break         // ③ 深度上限，防无限降级
    if not fbLimit.acquire(fb): continue   // ④ 降级目标独立配额（防备份被瞬间压垮）
    tag(req, degradedFrom=req.model, to=fb)// ⑤ 打标：降级请求可被抽样质检
    return gateway.call(req.withModel(fb), tier=i)

  return staticFallback(req)               // ⑥ 全链路挂：兜底话术，绝不 500
```

**降级的哲学是「优雅降质」，不是「换一个也可能挂的」。**

### 降级会不会静默降质？

会。降级到更小的模型，质量可能下降。**对策**：降级请求**打标 + 抽样质检**，不让降级变成「悄悄变差」。这是被问到时的必答点。

## 五、取消传播：断线不空烧 Token

```text
客户端断流 → CancellationToken → 中断底层 HTTP / 工具调用
→ 状态标记 CANCELLED（与 FAILED 分列）
→ 观测 tokens_after_cancel 指标趋近 0
```

**为什么必须分列**：`CANCELLED` 不能计入业务失败分母——否则用户主动划走会污染告警、失真成功率。

**双检查点**：提交前 + 流式过程中都要查（断线可能发生在任意时刻）。

## 六、潮汐弹性：成本治理的关键一脚

推理资源（尤其外采云卡）按时间计费，流量有明显日周期——晚高峰要扩、低谷要缩。

### 为什么 HPA 不够

```text
① HPA 不碰云卡计费 API：它只扩 Pod，不会调「开通/释放推理卡」API，卡还在烧钱
② HPA 不懂业务降级链：扩容失败时它不知道要跨厂商降级
③ HPA 是反应式，无预测提前期：云卡开通 + 预热是分钟级，
   等峰值来了 HPA 才反应，追不上
```

所以需要一个**跨系统协调器**，不是定时器，也不是 HPA。

### 潮汐控制器状态机

```text
forecast(t+Δ) = 季节性基线(日/周) ⊕ 短窗趋势(EWMA)
Δ(预测提前期) ≥ 云卡开通时延 + 预热时延      // 否则扩容追不上峰值

SCALE_OUT（预测将越峰）:
  开通/挂载云卡 → 注册 Endpoint = WARMING
  → 预热请求 → 权重 5% → 100% 渐进放量
  → 同步上调该 endpoint 的 Limit 配额

SCALE_IN（低谷确认）:
  先降 Limit → DRAIN（权重→0，等在途请求完成）
  → ★ 调云厂商释放 API 停计费

防抖：滞回带 [low, high] + 冷却期，避免在阈值附近反复扩缩
```

### 最容易栽的一脚：摘流 ≠ 释放

> **缩容必须真正调用释放 / 停计费 API。**
>
> 只把节点从路由摘掉、卡还在跑 = **空转烧钱**。

很多团队栽在这——以为「摘流」就是「缩容」，结果账单没降。这是本篇最值钱的一句。

### 缩容时粘滞会话怎么优雅迁移

```text
onScaleIn(endpoint):
  endpoint.status = DRAINING
  registry.setWeight(endpoint, 0)      // 新会话不再粘这里
  // 已粘会话不立即踢：等自然结束或 stickyTTL 到期后重选路
  waitInflightDrain(endpoint, timeout)
  if timedOut: forceRebind(endpoint.sessions)   // 超时才强迁
  cloudApi.release(endpoint)            // ★ 真正释放停计费
```

粘滞会话硬切会丢前缀 KV cache、首包飙高。`DRAINING` 软迁移让老会话平滑落地。

### 选路：加权随机 + 会话粘滞

```text
selectClient(model):
  candidates = registry.endpoints(model).filter(status==ACTIVE and weight>0)
  // 会话粘滞：同会话优先选上次副本 → 复用前缀 KV cache
  if sticky = sessionBinding.get(model, sessionId):
    if sticky in candidates: return sticky
  return weightedRandom(candidates)
```

会话粘滞的隐性收益：同一会话路由到同一副本 → **复用前缀 KV cache** → 更低首包、更低成本。

### 预测错了怎么办

**不追求预测准，追求「预测错了也不崩」**：

- 保底 min 副本（任何时候都在线）
- 滞回带防抖
- 实时 protection（超限触发降级而非硬失败）
- 人工 override

预测错的代价是「多开一点卡」（可接受的成本），不是「崩」（不可接受）。

### 自研资源的双队列

自研推理资源既服务在线对话（高优），又跑离线批任务（记忆抽取 / 评测，低优）：

```text
在线队列优先调度，保 min 副本不被批任务抢光
低谷时 GPU 时间片给批任务（可抢占，且不计入对话成功率分母）
```

闲时算力不浪费、高峰在线不被抢。

## 七、生产踩坑

### 坑 1：晚高峰 429 雪崩

**排查**：晚高峰 QPS 超单供应商配额，且重试无限制放大请求量。
**止血**：限制重试次数（**重试风暴比原始故障更致命**）+ 熔断提前介入。
**根治**：潮汐预测提前扩容 + 降级链跨厂商分流。
**验证**：峰值不再触发 429 雪崩。

### 坑 2：低谷云卡空转烧钱

**排查**：低谷节点已 DRAIN 摘流，但**没调释放 API**。
**修复**：补上「DRAIN → 调释放 API 停计费」完整闭环。
**验证**：闲时 card_minutes 与实际在线副本匹配。

### 坑 3：断线后账单仍涨

**排查**：断线未传播取消信号，模型还在生成。
**修复**：CancellationToken 贯穿到底层连接。
**验证**：`tokens_after_cancel` 趋近 0。

### 坑 4：扩容实例尾延迟差

**排查**：新实例冷启动 + 模型未预热就全量放量。
**修复**：WARMING 预热 + 渐进权重 5%→100% + 旧节点粘滞。
**验证**：新实例首包 P95 逐步收敛。

### 坑 5：令牌桶假满，QPS 没涨却全线限流

**排查**：`reservation` 次数远多于 `reconcile` 次数。
**根因**：异常/取消路径没走 finally 回填。
**修复**：`reconcile` 移入 finally + 定时对账重算桶。

### 坑 6：注册表陈旧，选到已释放的死节点

**排查**：本地注册表缓存推送有延迟。
**修复**：`configEpoch` 单调递增 + 选路失败即摘本地缓存换下一个 + 短 TTL 兜底。
**验证**：缩容瞬间错误率无尖刺。

## 八、追问与反陷阱

| 追问 | 陷阱 | 口径 |
|------|------|------|
| **Token 怎么限？输出还没生成就知道？** | 说不清 | 「准入用『输入 + maxOutput』保守上界占坑，调用后用真实 usage 回填校正（reconcile 必须在 finally，否则桶泄漏），并观测估算误差反哺默认值。」 |
| 估少了实际爆了怎么办？ | 以为准入锁死上限 | 「三道兜底：reconcile 追扣令牌、maxOutput 硬截断流式、estErr 反哺默认值。占坑—追扣—截断。」 |
| **熔断按错误率，供应商只是变慢不报错怎么防？** | 只用错误率 | 「熔断维度不止错误率，还有 P99 延迟 + 超时计入错误。慢到超时就是错。只看错误率会漏掉『慢但不报错』这类体验事故。」 |
| 降级链最后一环也挂了呢？ | 无限降级或直接 500 | 「链尾是静态兜底（预置话术/缓存），绝不 500。降级目标要独立配额、跳过已熔断、有深度上限、防自环——否则主模型故障会把备份级联打爆。」 |
| 降级会不会静默降质？ | 只讲好处 | 「会。所以降级请求打标 + 抽样质检，不让它变成悄悄变差。」 |
| **和 K8s HPA 区别？** | 把协调器降级成定时任务 | 「HPA 是单 Deployment 的反应式伸缩；这里要协调**云卡计费 API + 网关注册表 + 降级链**三套异构系统，且必须**预测提前期 ≥ 开通+预热**。是协调器不是定时器。」 |
| 「预测准确率多少」 | 逼你报数 | 「我不报准确率——用**保底副本 + 滞回带 + 实时 protection**兜底，预测错了也不崩，代价是多开一点卡。验证靠峰值未触发 429 + 闲时 card_minutes 下降。」 |
| 「限流会不会误杀正常请求」 | 把限流当纯负面 | 「限流是保护下游也是保护 SLA；被拒请求走降级链而非硬失败，体验上是从『更贵更慢』降到『够用更快』，不是报错。」 |
| 缩容把粘滞会话的副本释放了呢？ | 硬切 | 「DRAINING 软迁移：先摘权重不接新会话，老会话等自然结束或 stickyTTL 到期再重选，超时才强迁。硬切会丢前缀缓存、首包飙高。」 |

## 九、一句话速记

> Agent 的成本与稳定性瓶颈在模型出口。稳定性靠**三维限流（并发/QPS/Token，Token 用估算准入 + finally 里 reconcile）+ 多维熔断（错误率 **和** P99 延迟）+ 五道护栏的降级链 + 取消传播**；成本靠**潮汐预测提前扩容 + 真正调释放 API**——摘流 ≠ 释放，这一脚踩空账单就降不下来。而且潮汐追求「预测错了也不崩」，不追求「预测准」。

---

**上一篇**：召回准确率 ｜ **下一篇**：评测闭环——怎么证明你的改动真的有效
**系列目录**：[Agent 工程师面试深度系列](/ai-learning-blog/series/agent-interview-deep)
