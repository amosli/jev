# 06-4 RLCD 奖励与训练：严格适当评分规则，怎么把"分类"变成"会报概率的分类"

> 本文是 `06-Laya源码解读.md` 的子文档，专注 Laya 论文方法里那个最容易被一句"Reinforcement Learning with Calibrated Decisions"糊弄过去的部分：**RLCD 到底 RL 了什么？奖励长什么样？为什么这个奖励能"校准"？以及官方训练脚本在训练循环里到底写了什么。**
> 全部行号指向本目录 `laya-repo/`（v0.3.20）。带 `实测` 标记的数字来自 `examples/laya_internals_demo.py` 的真实运行（2026-09-26）。
> 读完这一篇你应该能回答：为什么"分数"和"校准"在 Laya 里是同一件事训练出来的，而不是事后两个独立工序。

---

## 0. 一句话结论

RLCD 的全部方法论 = **把"选哪个"改写成"报一个分布"，然后用严格适当评分规则（strictly proper scoring rule）作为奖励。** 适当性定理保证了：这个奖励期望**当且仅当**模型报出的分布等于真实分布时取最大——所以"报出来的概率"不是奖励的副产品，而是奖励的**唯一最优解**。温度（`06-3`）只是最后一步微调，真正把"可校准"写进模型的是这个奖励。

---

## 1. 为什么不用"选对就 +1"的分类奖励

传统分类训练（cross-entropy on one-hot）只区分"argmax 对不对"。它对模型"以 51% 的把握选对了"和"以 99% 的把握选对了"给**完全相同**的梯度信号。结果就是标准的过度自信：argmax 准确率不错，但报出来的概率全在 0.99——按置信度卡 0.9 门控，结果 20% 是错的。

Laya 的选择是：奖励函数的作用对象是**整个分布 q ∈ Δ^(K−1)**，而不是 argmax。这样模型如果以 0.51 报对了，会拿到比"以 0.99 报对了"更少的期望奖励，梯度会推它把"不确定"如实报出来。

### 1.1 严格适当性的定义（本文的数学核心）

设真实答案为随机变量 I，真实分布 p ∈ Δ^(K−1)，模型报出 q。对**任意**评分规则 S，定义**期望奖励**：

```
E[r] = Σ_i  p_i · S(q, onehot(i))       # 用每个可能的真实答案去打分，再按真实概率加权平均
```

**严格适当性定理**（Gneiting & Raftery, 2007）：

> S 是严格适当评分规则 ⟺ 对任意 p ≠ q，都有 E[r(q)] < E[r(p)]。
> 等价地：**E[r] 唯一的 argmax 就是 q = p**。

这个"充分必要条件"正是 Laya 一切概率承诺的根基：

- 模型想最大化期望奖励，**唯一**出路是让 q 逼近 p；
- 任何虚高自信（在低概率答案上多报质量）都会在期望上亏分；
- 与 RLHF 的偏好对不同——**偏好对可以被 reward hacking**（找到打分函数的漏洞），而严格适当评分规则**没有可利用的漏洞**，因为它是从对数损失族推出来的、对 p 与 q 的全局对称凹函数。

> **对比锚点**（`01-JEV与Laya深度对比.md §3.3`）：RLHF 用人类偏好对，单条对错只是"参考"、可被 hack；RLCD 用适当评分规则，奖励函数本身就是"报错分布就扣分"的严格目标，没有 hack 空间。代价是 Laya 只能做固定选项的封闭决策——这正是它的产品边界。

---

## 2. 奖励函数：27 行，三项（common.py:213-239）

```python
def proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)                    # ① 对数得分
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)   # ② 球面得分
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)   # ③ 秩概率分
        r = r - w_rps * rps * is_score                          # 只对 score 生效
    return r
```

逐项拆解：

### 2.1 对数得分（log score）——主力项

```
log_score = Σ_i target_i · log(q_i)
```

- `target` 可以是 one-hot，也可以是**软分布**（见 §3）——这是它比纯 CE 灵活的地方；
- `log_floor = −9.21` = `ln(10⁻⁴)`：`q` 被 clamp 到不低于 1e-4，即**单个选项的概率不至于无限接近 0 也要被罚到 −∞**。这个地板防止模型用"把正确项概率压到 0 来躲 log"这种病态最优；
- 双重 clamp：`clamp_min(1e-12)` 防 `log(0)` = −inf 产生 NaN 梯度；`clamp_min(−9.21)` 是有意的 reward shaping 下限。

对数得分本身就是严格适当的（它就是负的交叉熵家族）。所以 `log_score` 一项已经保证"q 逼近 p"。**球面得分和 RPS 是"额外纠偏"，不是校准的来源。**

### 2.2 球面得分（spherical score）——有界辅助项

```
sph = Σ_i target_i·q_i  /  ‖q‖₂
```

这是 `target`（视为向量）在 `q` 方向上的投影长度的归一化版本。数学上它和 **Brier score 的高次变体**相关（球面 / SRPS）。它给期望奖励加了一个 `∈ [0,1]` 的**有界**项：

- 好处：log score 无界，模型可以通过"把错项压到极低"无限刷分；球面项有界，防止这种病态，让梯度更稠密、更稳；
- 训练时权重从默认 `0.5` 调到 `0.75`（`06-4 §3.2`）——软标签场景下需要更强的质量对齐信号。

### 2.3 秩概率分（ranked probability score, RPS）——只对有序题型

```
rps = Σ_j (CDF_q(j) − CDF_target(j))²  /  (K−1)
```

**只在 `qtype == "score"` 时生效**（`is_score` 门控）。为什么只有 score 需要它：

- `choice`（无序）和 `noul`（无序二值）——把概率排在"哪一档"没有意义，只有"落在正确选项"有意义；
- `score`（有序，`level 0 … level K-1`）——把概率质量排在**正确的档位**上，比"恰好答对具体某一档"更有价值。如果真实答案是"level 2"，那把概率集中在 level 1-2-3 的**邻近档**也是合理的部分信息；RPS 在 CDF（累积分布）空间度量误差，天然惩罚"档位排错"，容忍"档位排近"。

`/(K−1)` 是归一化，让不同 K 的 RPS 可比（否则 K 越大 RPS 天然越大）。

### 2.4 我们的数值验证（实测）

`examples/laya_internals_demo.py §5` 用真实 `proper_reward` 做了两组检查：

**验证 A：期望奖励的 argmax 是否精确落在真分布上**（严格适当性的数值确认）——固定 target，网格扫描 q₁（其余均分）：

```
target=[0.7, 0.3]    choice (log+sph)      → 最优 q1 = 0.700  (真值 0.70)  ✓
target=[0.7, 0.3]    score  (log+sph−rps)  → 最优 q1 = 0.700  (真值 0.70)  ✓
target=[0.6, 0.25, 0.15] choice K=3        → 最优 q1 = 0.600  (真值 0.60)  ✓
```

**扫描的 argmax 精确等于真值**——这就是"严格适当"在数值上的样子：奖励面在 q=p 处有一个尖锐的全局最优点。

**验证 B：RPS 只惩罚 score**（§5c）——同一组输入 `q=[0.7,0.3], target=[1,0]`：

```
choice  r = 0.102898
score   r = 0.012898     (差 0.090000 = RPS 项的代价)
```

`choice` 题型下 RPS 项被 `is_score` 门控关掉；`score` 题型下多扣了 0.09。差值恰好是 RPS 的贡献，验证了门控逻辑。

---

## 3. 官方训练循环：GRPO 式策略梯度（notebook `train_ddp.py`）

> 全部在 `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`（cell 8）。

RLCD 名字里的 "RL" 是真的——它用**策略梯度**训练，不是直接把 `proper_reward` 当 loss 反传（因为 `proper_reward` 里 log 有 `clamp`、norm 有 `clamp`，不是处处可导的干净目标；策略梯度绕开可导性要求）。具体是 **GRPO 风格**（Group Relative Policy Optimization，用"同一状态下采样一组、组内标准化"代替 value network）。

三步：

### 3.1 采 G 组带噪分布（零均值投影噪声）

```python
G = 4
eps = randn(G, ...) * sigma * mask
eps -= eps.mean(-1, keepdim=True)      # 零均值投影：减组内均值，保证和不变
z   = logits.detach() + eps
q   = softmax(z)
```

- **为什么加噪**：策略梯度需要"从当前分布采样"。但 Laya 是确定性前向（只出 logits，不采样 token），所以构造一个**平滑的采样代理**：在 logits 上加高斯噪声得到 G 个"伪样本分布"；
- **零均值投影**（减组内均值）：保证 `Σ_i ε_i = 0`，即噪声不改变总概率质量方向（否则系统性地抬高分母），且与 `mask`（有效选项）相乘，只在有效选项上加噪；
- **`sigma` 退火 0.4 → 0.1**（线性）：训练早期噪声大（强探索、梯度覆盖广），后期噪声小（细调、靠近最优）。这是 RL 里的标准 exploration→exploitation 退火；
- **`logits.detach()`**：噪声分布不反传，梯度只经由 §3.3 的 log-prob 回流到 logits。

### 3.2 打分 + 组内标准化 → advantage

```python
r   = proper_reward(q, target, qtype, mask, w_sph=0.75, w_rps=1.0)
adv = (r - r.mean(0)) / (r.std() + 1e-6)
```

- **`adv` 是"相对"回报**：减去组内均值、除去组内标准差。这与 GRPO 的核心思想一致——**不需要 value network**，用"同一题采 G 个、谁比平均好"作为基线；
- **`w_sph` 从默认 0.5 调到 0.75**（训练用软标签，需要更强的质量对齐信号——软目标下 log 项的区分度略降，补一点有界的球面项）；
- `+1e-6` 防 std=0 除零（当 G 个样本奖励完全相同时）。

### 3.3 高斯 log-prob 的 REINFORCE 梯度 + 软交叉熵兜底

```python
logp = −Σ (z − logits)² · mask / (2σ²)          # 高斯核的 log 密度（REINFORCE 梯度）
loss = −(adv · logp).mean() + 1.0 · (−Σ target · log_softmax(logits))
```

- **第一项是策略梯度**：把 `−adv·logp` 对 logits 求导，adv>0 的分布被拉向、adv<0 的被推离。因为噪声是高斯的，logp 有闭式 `−(z−logits)²/2σ²`，梯度= `−(z−logits)/σ² · adv`——本质是"往奖励高的采样点方向挪 logits"；
- **第二项是软交叉熵**（系数 1.0）：纯 RL 训练不稳（采样噪声大、adv 噪声大），加一项直接监督的 CE 兜底。**软** CE（target 是分布不是 one-hot）与训练标签一致；
- `CE` 用 `log_softmax(logits)`（未加噪的干净 logits），保证有一条稳定的监督梯度通路。

### 3.4 act 头在训练里被摘掉——`0.0 * act.sum()`

```python
loss = (loss_rl + 1.0*loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
```

**`0.0 * act.sum()` 让 act 头的梯度恒为 0。** 这不是笔误——是显式把 act 头从训练图里摘出来（避免它产生无意义的梯度污染/显存占用），但保留在计算图里让代码能跑通。

**这就是 issue #185（"act_probability has no signal"）的直接代码级原因**——不是模型学坏了，是它从来没被训练过。详见 `06-2 决策头与前向` §act 头一节。

> 推论：如果你要启用 act 头（做"该不该自动执行"的判断），**不能只改这个 0.0**，还得自己设计 act 奖励。Laya 给了成本常量的接口（`build_model` 里 `n_act = len(cfg["act_costs"]) + 1`，见 `common.py:210`），README 提到"执行对 +1 / 执行错 −3 / 升级 −0.5 ⇒ 理论边界 p≈62.5%"，但**官方训练脚本没有实现这个损失**——需要自己接。

### 3.5 训练超参与内存

- **LR 分档**：encoder `2.5e-5`，head `1.0e-4`——决策头学得比主干快 4 倍（头是随机初始化的，主干是预训练权重）；
- **有效 batch**：`MICRO_BATCH 8 × 2 GPU × GRAD_ACCUM 4 = 64`；
- **head_checkpointing**（`common.py:147`）：非 reentrant checkpoint，**冻结 encoder 时也能训练 head**（常规 reentrant checkpoint 在输入无梯度时不会传梯度到后面，这里特意用 `use_reentrant=False`）。默认 False，eval 和 `no_grad` 下自动绕过。浏览器 agent 案例实测：关掉梯度检查点是 1.25× 的"免费加速"（`docs/finetune_browser_agent.md`）。

---

## 4. 软标签：RLCD 相比 one-hot CE 的第二个优势

前面提到 `target` 可以是软分布。官方 notebook 的构造方式是**金标分布（gold distribution）**——对每条训练样本，除了正确答案，还给次优答案分配递减的目标质量（不是 one-hot，而是"正确 0.7、次优 0.2、次次 0.1"这种）。这带来两个好处：

1. **编码了标注者的不确定性**：如果两个答案都合理，人标本来也分不清 0.6/0.4，硬压成 one-hot 反而制造了假的确定性；
2. **奖励面更稠密**：软目标下 `proper_reward` 每一项都有信息（one-hot 时只压正确项、不管错项之间的相对质量），梯度信号更密、更好训。

这与 `06-1` 的 `[MASK]` 软标签思想、以及 browser-agent 案例的"Step-2 negatives"（中间态负样本）是一脉相承的：**Laya 全程用软目标，因为决策本身就带模糊性，硬标签是把不确定性提前扔掉。**

---

## 5. 多轮扩展：TD(λ) 自举目标（common.py:242-257，官方 notebook 未调用）

单轮决策的 RLCD 是一次性奖励。多轮 agent 场景（一个 episode 走 T 步、每步一个决策）需要把终态回报回传给每个前缀——`td_lambda_targets` 就是干这个的：

```python
def td_lambda_targets(p_true, batch, lam=1.0):
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(...)          # 同一 episode 的所有步
        idx = idx[torch.argsort(batch["ep_step"][idx])]  # 按步序
        y = batch["target"][idx[-1], 1]           # 终态目标
        G = y
        for j in range(len(idx) - 1, -1, -1):     # 从终态往前回溯
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[next] + lam * G   # TD(λ) 递推
            ...
```

- **按 `ep_group` 分组、`ep_step` 排序**：一个 batch 里的行可能来自不同 episode 的不同步，先聚成轨迹再排序；
- **TD(λ) 递推** `G = (1−λ)·p_true[下一轮] + λ·G`：`λ=1` 退化成蒙特卡洛（只用终态回报，忽略中间），`λ=0` 退化成一步 TD（只用下一步的即时目标）。可调；
- `ep_group < 0` 的行被跳过（不属于任何 episode 的独立样本）；
- **公开 notebook 没有调用它**——多轮 agent 场景需要自己接。你要做的 Laya 多轮训练，这是现成的组件。

---

## 6. 一份可直接改的训练骨架

把上面的机制拼成最小可用的训练循环（基于官方 notebook，去掉 act 头的 0.0 干扰，保留全部校准逻辑）：

```python
import torch, torch.nn.functional as F
from laya.common import build_model, proper_reward

# ---- 1) 模型：加载预训练 checkpoint，冻结 encoder（省显存大头）----
model = build_model(cfg, encoder_dir=..., pretrained=True)
model.encoder.requires_grad_(False)          # 冻结主干，只训 head
model.head_checkpointing = True             # 冻结 encoder 时也能训 head
model.train()

head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
opt = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 2.5e-5},   # 若解冻
    {"params": head_params,                       "lr": 1.0e-4},  # head 学更快
])

# ---- 2) RLCD 训练步（GRPO 式）----
G, sigma_hi, sigma_lo = 4, 0.4, 0.1
for batch in loader:
    logits, qtype, target, mask = forward(batch)     # (N, K)

    # 3a) 采 G 组带零均值投影噪声的分布
    eps = torch.randn(G, *logits.shape, device=logits.device) * sigma * mask
    eps = eps - eps.mean(-1, keepdim=True)           # 零均值投影
    z = logits.detach()[None] + eps                  # (G, N, K)
    q = torch.softmax(z, -1)

    # 3b) 严格适当奖励 → 组内标准化 advantage
    r   = proper_reward(q, target[None], qtype[None], mask[None], w_sph=0.75, w_rps=1.0)
    adv = (r - r.mean(0)) / (r.std() + 1e-6)         # (G, N)

    # 3c) 高斯 log-prob 的 REINFORCE 梯度
    logp = -((z - logits[None])**2 * mask[None]).sum(-1) / (2 * sigma**2)
    loss_rl = -(adv * logp).mean()

    # 3d) 软交叉熵兜底（干净 logits，不用噪声版）
    loss_ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()

    loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM
    loss.backward()

# ---- 3) sigma 退火 0.4 → 0.1（跨 epoch 或按 step）----
# ---- 4) 训练完，用【没进过训练的】留出集拟合温度（见 06-3 §4）----
```

**与官方 notebook 的三处关键差异（都是刻意为之）**：
1. 冻结 encoder 只训 head（省显存，样本少时更稳；官方全量训但 2×T4 也够）；
2. 导出 config 时**不继承** `temperature_by_options`（`06-3 §4.4` 的静默失效坑）；
3. 如果你要用 act 头，把 `0.0 * act.sum()` 换成真正的 act 损失（自己设计，见 §3.4 推论）。

---

## 7. 实操清单：RLCD / 概率可信度的 6 条落地准则

1. **不要用 argmax 准确率评估概率可信度**——用 ECE（`06-3 §5.1`）在 `answer_confidence` 上。argmax 准和概率准是两件事。
2. **软标签优于 one-hot**。标注拿不准两个答案时，给软分布（0.6/0.4），比硬压 one-hot 更贴近决策本质，且 `proper_reward` 原生支持。
3. **高基数场景补负样本**。browser-agent 案例的教训（`docs/finetune_browser_agent.md`）：如果每条正样本都有历史、每条负样本都没历史，模型会学到"有历史⇒正例"这种捷径。Step-2 negatives（中间态负样本）是必需的。
4. **多轮 agent 场景自己接 `td_lambda_targets`**，官方 notebook 不调用它。λ=1 起步（纯回报回传），有信号了再往下调。
5. **act 头当前不可用**（`0.0 * act.sum()`，issue #185）。要做"自动执行 vs 人工"的门控，用 `answer_confidence` 卡阈值（`06-3`），别依赖 `act_probability`。
6. **训练完一定重新拟合温度**，且校准集要用留出集（`06-3 §4.1`）。在训练过的样本上拟合温度量的是"拟合"不是"校准"。

---

## 8. 参考文献与源码锚点

| 结论 | 出处 |
|---|---|
| `proper_reward` 全文（对数+球面+RPS，27 行） | `laya/common.py:213-239` |
| RPS 只对 score 生效 | `laya/common.py:232-238` |
| 严格适当性验证（argmax 落在真分布） | `examples/laya_internals_demo.py §5b` |
| RPS 门控数值验证（choice vs score 差 0.09） | `examples/laya_internals_demo.py §5c` |
| GRPO 训练循环（采样/advantage/logp/软CE/`0.0*act`） | `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` cell 8 |
| `td_lambda_targets`（TD(λ) 多轮回传） | `laya/common.py:242-257`（notebook 未调用） |
| `head_checkpointing`（冻结 encoder 也能训 head） | `laya/common.py:147-150`；README "Fine-Tuning" 一节 |
| act 头无信号 = `0.0 * act.sum()` | notebook cell 8；对应 issue #185 |
| 严格适当性的理论出处 | Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA |
| RLCD 训练数据构造七步法、负样本教训 | `docs/finetune_browser_agent.md` |
| 训练 LR / batch 配置 | notebook cell 8（encoder 2.5e-5 / head 1.0e-4 / 有效 batch 64） |

> 上一层的"温度"怎么用见 `06-3-温度与校准.md`；决策头前向见 `06-2-决策头与前向.md`；序列与 [MASK] 见 `06-1-序列构造与MASK机制.md`。
