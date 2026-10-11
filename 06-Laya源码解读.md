# 06 Laya 源码解读：核心思想与实现（总览）

> 基准：`laya-repo/`（v0.3.20 克隆，本目录内）。全部行号引用自实际源码，可对照打开。目标不是复述 README，而是回答：**这个 2000 行的包为什么能做到"毫秒级、可校准、不解析"的结构化决策？它的设计权衡是什么？哪些地方藏着使用者必须知道的细节？**

**本文是总览**——把 Laya 的设计讲成一个整体。RLCD、`[MASK]`、温度这三个机制需要机制级的逐行拆解，放在四篇子文档里：

| 子文档 | 讲什么 | 核心机制 |
|---|---|---|
| [06-1 序列构造与 MASK 机制](06-1-序列构造与MASK机制.md) | `build_sequence` 的 50 行怎么把选项塞进序列、token 预算怎么会计、`[MASK]` 槽位怎么被读、高基数为什么塌到 0.425 | **`[MASK]` 机制**、预算会计、题型渲染 |
| [06-2 决策头与前向](06-2-决策头与前向.md) | `DecisionModel.forward` 五步流水线：encoder → type_emb → 2 层 head → gather → scorer；act 头为什么没信号 | 决策头、gather、head_checkpointing |
| [06-3 温度与校准](06-3-温度与校准.md) | 温度缩放的数学、clamp 与 RuntimeWarning、三个 checkpoint 的温度实况、ECE 实测、双置信度为什么不能同阈值 | **温度机制**、post-hoc 校准、ECE |
| [06-4 RLCD 奖励与训练](06-4-RLCD奖励与训练.md) | 严格适当评分规则、`proper_reward` 三项、GRPO 式训练循环、`0.0*act` 的后果、TD(λ) 多轮回传 | **RLCD**、GRPO、软标签 |

> 本文 + 四篇子文档 = 完整解读。子文档里的每个数字都来自 `examples/laya_internals_demo.py` 的真实运行（用 `.venv/bin/python examples/laya_internals_demo.py` 可完整复现）。运行时/路由/批量这些"子文档未覆盖"的部分仍在本文 §1、§2、§3.4、§3.5。
> 阅读顺序建议：本文 §1 核心思想 → §2 一次 predict 生命周期 → 按需读子文档 → 本文 §4 设计权衡。

---

## 0. 模块地图

```
laya/
├── common.py     (373 行)  ★ 心脏：输入编码 build_sequence + DecisionModel + RLCD 奖励 + 校准工具
├── agent.py      (931 行)  运行时：加载/校验/批量/解码/OOM 降级 + laya.load()
├── router.py     (817 行)  多 checkpoint 调度：语言路由、LRU、preload
├── lang.py       (422 行)  零依赖语言检测（Unicode 脚本区间 + 功能词启发式）
├── hooks.py      HookRegistry：start/end/route/load/evict/error 六个挂点
├── serve.py      (301 行)  Jev 兼容 HTTP 面（/v1/systemone）
├── fast.py       TileLang 融合核 + bf16 常驻 + CUDA Graph（可选 extra）
├── shortlist.py  高基数选项的 embedding 粗筛（coarse-to-fine）
├── structured.py JSON Schema/pydantic → questions 编译器（纯 Python，torch-free）
├── onnx_agent.py ONNX Runtime 推理
├── presets.py    五套生产问题模板
└── cli.py / mcp/ / integrations/（LangChain）
```

依赖方向干净：`common` 不依赖 `agent`（精调 notebook 只 import `common`），`structured` 完全 torch-free，`serve`/`mcp` 的重依赖全部延迟导入。**整个"决策模型"的学术内容集中在 `common.py` 一个文件里**——这是读码第一原则：先读它。

---

## 1. 七个核心思想

### 思想一：决策即"抽取"，不是"生成"——一切复杂度都进了序列布局

生成式模型把答案变成 token 序列（要采样、要解析、会幻觉）；Laya 把答案变成**位置**。`build_sequence`（common.py:70-120）定义了唯一的输入格式：

```
[CLS] <题型> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
      └────── head 区，预算 head_max_len ──────┘      └── 剩余预算全部给 state ──┘
```

三个布局决策值得注意（机制级拆解见 [06-1](06-1-序列构造与MASK机制.md)）：

1. **选项在 state 之前**。选项是"答案候选"，必须落在 head 区由决策头直接读取；state 是"证据"，排最后可整体截断。作者在浏览器 agent 案例中实测：把候选从 state 挪进选项区（而不是塞在 state 的 JSON 里）是收益最大的单点改动（Mind2Web 点击 top-1 0.44→0.51）——这个布局就是那 lessons 的代码固化；
2. **每个选项 = 一个 [MASK] + ≤48 token 描述**（common.py:95-101）。[MASK] 位置的最终隐状态就是该选项的"表示"，决策头只需对每个 marker 位置跑一次 MLP 打分——**K 个选项 = K 个 marker，一次前向全部出分**；
3. **token 预算是显式会计**（common.py:102-107）：`opt_budget = head_max_len − Σ选项长度`，不够时每选项砍到 `max(4, (head_max_len−16)//K)`。**Banking77 上 77 选项每选项只剩 4 token、准确率塌到 0.425 的架构根源就在这几行**——官方 `BENCHMARKS.md:147-149` 自己定性为"预算天花板而非能力差距"。这也是 `shortlist.py` 存在的理由（先粗筛 top-k 再进决策面）。

题型差异被压到最小：choice 渲染 `选项名: 描述`；score 渲染 `level i: 描述`（位置即序数）；noul 固定两个槽位 `[false, true]`，返回 P(true)（common.py:49-67）。三种题型共用同一个模型、同一次前向、同一套校准。

### 思想二：决策头 = "标记抽取 + 轻量聚合"，主干是冻结的现成 encoder

`DecisionModel.forward`（common.py:139-175）只有五步：

```python
h = encoder(input_ids, attention_mask).last_hidden_state   # ModernBERT/mmBERT，现成权重
h = h + self.type_emb(qtype)[:, None, :]                   # ① 告诉模型"这是哪种题型"
h = head(h)                                                # ② 2 层 TransformerEncoder 聚合
m = torch.gather(h, 1, marker_pos)                         # ③ 抽出每个 [MASK] 位置的向量
logits = self.scorer(m).squeeze(-1)                        # ④ 每选项一个 logit
act_logits = self.act_head(cat[pooled, feats])             # ⑤ 升级/执行判断头
```

- **type_emb（①）** 是一个 3×d 的嵌入，加到*每个* token 上——同一份文本在不同题型下的表示被显式区分。这是"三种题型一个模型"的实现机关，成本几乎为零；
- **决策头（②）** 用 `nn.TransformerEncoder`（norm_first）而非 MLP，因为"选项 i 的合理性"依赖"选项 j 说了什么"（对比性打分），需要选项 marker 之间互相 attend；
- **act 头（⑤）** 的输入是 `[CLS] 池化 + 4 个分布特征`（common.py:172）：top1 概率、top1−top2 差、归一化熵、K/255。**即"该不该自动执行"被建模为分布形状的函数**——训练时配成本敏感奖励（执行对 +1 / 执行错 −3 / 升级 −0.5 ⇒ 理论边界 p≈62.5%）。注意：这个头当前输出无信号（issue #185，几乎恒 1.0），代码里它照常返回但**使用者应忽略它**，见思想三。

### 思想三：校准是一等公民——而且代码里"校准"有两个不同的含义

这是全仓库最容易被使用者误读的地方，源码注释花了三段话讲清楚（common.py:273-299、agent.py:670-675）。**机制级拆解、数值实测见 [06-3 温度与校准](06-3-温度与校准.md)**。摘要：

| 字段 | 计算 | 性质 |
|---|---|---|
| `answer_confidence` | `max(p)` | **温度拟合的目标、ECE 度量的对象**。"置信度 c 的答案约 c 比例正确"这个保证只对它成立 |
| `confidence` | choice/score：`1 − H(p)/log K`（分布集中度）；noul：`max(p, 1−p)` | 启发式信号，**不是**温度拟合的对象，绝对值没有概率语义 |
| `action.act_probability` | act 头输出 | 当前无信号（issue #185，AUROC 0.30 vs 置信度 0.77），忽略 |

`_decode_answers` 的注释（agent.py:670-675）原话："Rather than change one underneath existing callers, report both: `answer_confidence` is the calibrated one, on every question type"。源码 docstring 更直接（common.py:281-282）："the two must not be compared against the same threshold"。**README 的门控示例仍用 `confidence`，但绝对阈值门控的统计学保证在 `answer_confidence` 上**——本知识库 01/02/03 与各 demo 已按此修正。

温度机制三句话（详见 06-3）：
- 解析优先级：**语言专属覆盖 > 分桶（如 `choice:11+`）> 每题型**；softmax 前做 `logits / t`；
- **T < 1 是锐化不是软化**。出厂 `choice:11+` 桶温度 0.1006，等效把 logits 放大 ~10 倍——我们实测它把一个 0.24 的 top 概率报成 **0.85**（源码注释称 0.99，两者指向同一失效模式）；
- **clamp 到 [0.5, 5] 并发 RuntimeWarning**。但实测发现 clamp 到 0.5 后 ECE 仍达 0.366，**比完全不校准（ECE 0.279）更差**——出现该告警时应重新校准或退回 T=1.0，不要盲信 clamp 后的值。

### 思想四：RLCD 的奖励函数在代码里只有 27 行

**机制级拆解见 [06-4 RLCD 奖励与训练](06-4-RLCD奖励与训练.md)**。`proper_reward`（common.py:213-239）就是"严格适当评分规则"的全部：

```python
log_score = (target * log(q).clamp_min(-9.21)).sum(-1)   # 对数得分，地板 ln(1e-4)
sph = (target * q).sum(-1) / q.norm(-1)                   # 球面得分
r = log_score + 0.5 * sph
r -= 1.0 * rps * (qtype == score)                         # RPS 只用于序数题
```

为什么这能逼出校准？**严格适当评分规则的期望在"报告分布 = 真实分布"处唯一最大化**——模型任何虚高自信都会在期望上亏分，与 RLHF 的偏好对（可被 reward hacking）形成对照。我们用数值扫描验证了这一点：固定 target、网格扫描 q₁，期望奖励的 argmax **精确落在真分布上**（target=[0.7,0.3] → 最优 q₁=0.700；[0.6,0.25,0.15] → 0.600）。

多轮扩展 `td_lambda_targets`（common.py:242-257）用 TD(λ=1) 把终态回报回传给轨迹前缀，25 行，**公开 notebook 未调用**——多轮 agent 场景需自己接。

### 思想五：路由必须发生在 forward 之前——因为模型会"自信地错"

`Router._route`（router.py:435 起）的决策优先级链：

```
显式 model > 显式 task > 检测到的 typed-decisions 工作流(需 opt-in) > 显式 lang > lang_guess > 脚本/语言检测 > default
```

为什么路由不能靠模型自己的置信度兜底？README 的数据：英文 checkpoint 读高棉语**准确率 0.000、置信度 0.952**。错误发生在表示层，输出层的置信度看不见它——所以唯一可靠的防线是**在进模型前用零依赖的文本检测分流**。`lang.py`（422 行）就是这条防线：Unicode 脚本区间判定（`_SCRIPT_RANGES`）→ 混合文本按非拉丁字母占比升级 → 拉丁文用功能词表启发式识语言。设计取向明确：**宁可粗糙要零依赖**（"deliberately dependency-free"），并留了 `lang_guess` 接缝让外部 LID 模型接管（router.py 构造器注释称其为 "the seam for a real LID model"）。

LRU 参数 `max_loaded=2` 的理由写在 docstring（router.py:171-176）：自动路由只会在 english/multilingual 之间切换，`=1` 会在每次语言切换时重建刚逐出的 checkpoint（秒级）；纯单语言流量永远不建第二个、零成本。另外**推理故意放在锁外**（router.py:236-238 注释）：生命周期用 RLock 保护，`Agent.system_one` 不串行——并发预测共享同一 checkpoint，这是 GPU 形态的选择。

### 思想六：批量 = "状态×问题"展平成行

`predict_batch`（agent.py:711-842）的核心一步：N 个状态 × M 个问题 = N×M 行，经 `collate_items` 打包成 `(N·M, L)` 张量共享前向（README 实测 GPU 上 ~9–10× 提速的来源）。三个细节：

1. **state 只 tokenize 一次、所有问题复用**（agent.py:569-576 注释："re-serializing … was pure duplicated work"）；
2. **会话列表从左截断**（agent.py:566-568）：时间序 newest-last，右截断会静默丢最新一轮——`truncate_left` 保住最近意图。这类"截断方向"细节是最容易在复刻时踩错的地方；
3. `sort_by_length` 只在 `1 < batch_size < N` 时生效，窗口=8 个 batch 内按最长行排序后恢复原序（agent.py:789-798），注释同时警告"改 batch 形状会有浮点微差，阈值系统需回归"。

`usage.input_tokens` = attention mask 实际求和（非 padding），`output_tokens` 恒为 0——**决策不产生输出 token**，这就是计费/延迟优势的会计体现。

### 思想七：防御式运行时——失败路径被认真对待

- **OOM 降级**（agent.py:623-646）：CUDA OOM → 先 `deaccelerate()` 摘掉 fast path 再迁 CPU + 关 amp（注释解释了顺序错误会怎样：带着 CUDA 图替换的 forward 迁移后仍会撞同一个坑）；MPS/CPU 上 amp 不支持 → 一次性降 fp32 重试；
- **单选项题的 top-2**（common.py:161-171）：`p.topk(2)` 在 K=1 时会崩，代码补零保持 act 头特征语义一致；
- **hooks 不掩盖主错误**（agent.py:822-841）：`on_error`/`on_predict_end` 自己抛异常时，异常被挂到 `__context__` 而不是替换原异常；
- **校验前置**：选项超出 head 预算直接 `ValueError` 指名问题（agent.py:582-583），而不是静默截断；`collate_items` 里 target 长度与 marker 数不符时报错信息直接解释真实错误（common.py:350-358，含 issue #311 引用）；
- **serve.py 单 worker 池**是显式设计（188-195 行注释）：一次一个前向是单卡的物理事实，扩容靠多副本。

这批细节的总和构成一种风格：**每个已知的失败模式都在代码里留下一段解释性注释**——这也是它适合作为教学级工程范本的原因。

---

## 2. 一次 `router.predict(state, questions)` 的完整生命周期

```
Router.predict (router.py:503)
 ├─ Router.route → _route (router.py:435)
 │    ├─ lang_guess / analyse() 脚本+语言检测          # <0.5ms，纯 Python
 │    └─ RouteDecision{model, repo, reason}            # reason 是人话，排障用
 ├─ load(name)（带 LRU touch；不在锁内做推理）
 │    └─ Agent(repo, subfolder=…)                      # 首次：snapshot_download 只拉该子目录
 │         ├─ rl_agent_config.json 校验                 # 没有 = 不是 Laya checkpoint
 │         ├─ tokenizer 加载 + _fix_tokenizer_config
 │         └─ build_model: encoder + DecisionModel 头 + 温度 buffer
 ├─ Agent.system_one(state, questions)  @torch.no_grad (agent.py:845)
 │    ├─ _check_question ×M                            # 题型/选项合法性
 │    ├─ _to_internal ×M                               # 归一化；instructions 非字符串 →
 │    │                                                #   ensure_ascii JSON（德语 bug 修复, 545-549）
 │    ├─ _encode_state：state tokenize 1 次 → build_sequence ×M
 │    │    └─ [CLS] head [MASK]… [SEP] state [SEP]，markers 记录位置
 │    ├─ collate_items → (M, L) 张量                   # M 行 = M 个问题
 │    ├─ _infer：amp 门控 → forward → (OOM→CPU 降级重试)
 │    │    └─ encoder → +type_emb → head×2 → gather markers → scorer → logits
 │    └─ _decode_answers ×M (agent.py:653)
 │         ├─ 温度：lang覆盖 > 分桶 > 每题型 → logits/t → softmax
 │         ├─ choice:  argmax → choice + probabilities + 双置信度
 │         ├─ score:   Σ i·p_i → 期望分 + legend + 分布
 │         └─ noul:    p[true] + confidence=max(p,1−p)
 ├─ 组装 {model:"laya-rl-agent", answers, usage, routing}
 └─ hooks: on_predict_end（失败路径也会走 on_error/on_predict_end）
```

耗时分布（本机 RTX 5060 Ti 实测，见 03 教程 §2.5）：检测 <0.5ms、编码+前向 ~10ms、解码 <1ms——**首问的 50–70s 全在 checkpoint 加载**，这就是"生产必须 preload"的数据支撑。

---

## 3. 关键代码精读（机制部分见子文档）

### 3.1–3.3 已拆入子文档

| 原小节 | 现位置 |
|---|---|
| `build_sequence` 预算会计 | [06-1 §2.3](06-1-序列构造与MASK机制.md)（含 2–77 选项的完整塌陷实测表、Banking77 0.425 的官方定性） |
| act 头特征 | [06-2 §4](06-2-决策头与前向.md)（含 `0.0*act.sum()` 的代码级根因、issue #185 的 AUROC 对比） |
| 温度解析与双置信度 | [06-3 §2、§5、§6](06-3-温度与校准.md)（含三个 checkpoint 温度实况、clamp 数值表、ECE 实测、双置信度分歧表） |

原文里保留的三行核心，够用版在此：

```python
# 预算会计（common.py:102-107）—— 选项区分度 = head_max_len / K
opt_budget = head_max_len − sum(len(o) for o in opt_ids)
if opt_budget < 16: opt_ids = [o[: max(4, (head_max_len−16)//K)] for o in opt_ids]

# act 头特征（common.py:158-172）—— detach：不让 act 梯度污染主分布
feats = [top1, top1−top2, ent, k/255];  act_logits = act_head(cat[h_CLS, feats])

# 温度解析 + 双置信度（agent.py:662-675）
t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])   # 语言覆盖更高
z = logits[r, :k] / t_scale; p = softmax(z)
answers[qid] = {"confidence": 1 − H/logK,           # 启发式（choice/score），不可当概率
                "answer_confidence": max(p), …}      # 门控用这个
```

### 3.4 `Router._route` 优先级链（router.py:435-475）

每一级都"能回答才接管，否则下落"：显式参数最优先；`lang_guess` 返回 None 视为弃权落到内置检测；检测不到字母（纯数字/符号 state）用 `default`。**这个"弃权下落"模式**是外部 LID 模型可以安全接入的原因——不确定就交还给保守路径，不强迫表态。

### 3.5 `serve.py` 的并发模型（188-199, 273-283）

单线程 `ThreadPoolExecutor(1)` + `asyncio.Lock` + `run_in_executor`：推理是同步 torch 且 CPU 上数百毫秒，必须离开事件循环；单 worker 是"一次一个前向"的物理事实而非偷懒；同时健康检查不被长推理阻塞。401 比较用 `hmac.compare_digest` 且两侧先编码（注释记录了一个真实漏洞：latin-1 头解码使 str 比较在非 ASCII 时抛 TypeError → 500）。

---

## 4. 设计权衡点评（如果你要复刻）

| 决策 | 替代方案 | 为什么选它 |
|---|---|---|
| 选项放 head 区、[MASK] 抽取 | 生成式（LLM 出 JSON）/ 双塔（state、选项各自编码） | 单次前向、无解析、选项间对比可用注意力表达；双塔省不了什么因为 state 只编码一次 |
| 2 层 TransformerEncoder 决策头 | MLP / 更深 | 选项间需要 attend（对比打分）；更深在 421M 底座上收益递减且拖延迟 |
| type_emb 加在每个 token | 每题型独立头 | 3 个嵌入向量 vs 3 套头的参数/维护成本；共享底座还能跨题型迁移 |
| 温度分桶（题型×选项数） | 全局单温度 | 不同 K 的分布锐度天然不同，单温度会顾此失彼（ECE 数据：分桶拟合 0.466→0.081） |
| 检测式路由（脚本+功能词） | 模型置信度路由 / 始终用 multilingual | "自信地错"不可检测；multilingual 英文性能掉 12pt（MASSIVE en 0.783 vs 0.657）；零依赖检测 <0.5ms |
| `max_loaded=2` LRU | 全 preload / =1 | 见 §思想五；默认值让单语言部署零成本、双语部署零重建 |
| hooks 六挂点 + 失败不掩盖 | 继承扩展 | 不 fork 就能脱敏/审计/缓存；错误路径的 hook 异常挂 `__context__` 保住原始栈 |
| 结构化 decide 拒绝自由字符串 | LLM 式"尽力填" | 只答封闭选项是产品边界也是诚实边界——`structured.py` 头注明确"Anything that cannot be answered from a fixed option set … is rejected" |

---

## 5. 对使用者的三个代码级启示（反哺 03/05 教程）

1. **`head_max_len` 是你需要真正理解的唯一超参**：选项区分度 = head_max_len/K。改它的正确姿势是 `agent.cfg["head_max_len"] = 512`（03 教程 §8.3），但要意识到这同时压缩 state 预算（`max_len − head_max_len`），长文场景要一起调 `max_len`；
2. **精调数据直接对齐 `build_sequence`**：05 文档的 typed-decisions JSONL 在训练时经同一个 `build_sequence` 变成序列与 marker——所以你的选项描述写多长（≤48 token 才不被砍）与选项顺序，训练与推理是同一套规则。选项顺序扰动既是评测项（`research/eval/metamorphic.py`）也可以是增广手段；
3. **两个置信度字段从今天起分开用**：`answer_confidence` 做绝对阈值门控（校准保证），`confidence` 只做同类内的相对排序；`action.act_probability` 忽略直到 issue #185 被修复。

---

## 6. 与本目录其他文档的印证关系

| 源码事实 | 印证文档 |
|---|---|
| 预算会计（→ [06-1 §2.3](06-1-序列构造与MASK机制.md)）→ Banking77 0.425 | 01 对比 §4.2 高基数短板；05 调优三板斧 |
| act 头 `0.0*act.sum()`（→ [06-2 §4](06-2-决策头与前向.md)） | issue #185；03 教程门控改用 `answer_confidence` |
| 双置信度（→ [06-3 §6](06-3-温度与校准.md)） | 03 教程 §2.3/§8.1（已修正）、各 demo 门控字段 |
| 温度 clamp 与 choice:11+ 实测 ECE（→ [06-3 §5](06-3-温度与校准.md)） | 03 教程排错表"invalid temperatures"行；01 对比 §6.2 校准不足 |
| multilingual 无事后校准（→ [06-3 §3.2](06-3-温度与校准.md)） | 02 企业指南多语言门控阈值须自标定 |
| RLCD 奖励 27 行（→ [06-4 §2](06-4-RLCD奖励与训练.md)） | 01 对比 §3.3 训练方法（魔搭长文口径的代码证据） |
| GRPO 训练循环与软标签（→ [06-4 §3](06-4-RLCD奖励与训练.md)） | 05 精调流程；03 教程训练章节 |
| 路由优先级链与 lang_guess 接缝 | 03 教程 §8.4；02 企业指南多语言方案 |
| serve 单 worker/限流常数 | 02 企业指南 §5.2 服务化清单 |
| fast.py CUDA Graph 全程持锁（28-30 行注释） | 03 教程 §5.2 调优要点；同卡混部 OOM 降级路径（agent.py:623-646）即 03 排错表实测行 |

> 结论：Laya 的核心思想可以压缩成一句话——**"把'判断'从语言生成里拆出来，变成一个带预算会计、带校准责任、带前置路由的分类器，然后用工程手段（而不是概率本身）去兜住它所有的已知失败模式。"** 概率只是信号，防线全在结构里。
