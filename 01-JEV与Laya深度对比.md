# JEV 与 Laya 深度对比：System 1 决策模型的学术原理与工程实践

> 定位：从学术与工程两个视角讲清 JEV（TypeSafe AI 的托管决策模型）与 Laya（Convai Innovations 的开源决策模型）是什么、怎么做的、差在哪、各自的不足。
> 所有数据均标注来源；【实测】= 本文作者 2026-09-24 在本机（RTX 5060 Ti 8GB / CUDA 13.2 / Python 3.12 / laya 0.3.20）复现的数据；【官方】= 厂商自报；【三方】= 独立第三方实测。存疑口径显式标注。

---

## 0. TL;DR

| 维度 | JEV（TypeSafe AI） | Laya（Convai Innovations） |
|---|---|---|
| 形态 | 托管 API 双入口（jevai.org 社区端点免费限流 + api.typesafe.ai 商业端点；闭源权重） | 开源模型 + Python SDK（Apache 2.0） |
| 定位 | "System One Model"：毫秒级结构化决策，永不生成文本 | 同一定位的开源对标实现，协议兼容 Jev |
| 单次决策延迟 | p50 ≈ 236–276 ms（三方实测，法国调用，商业端点）【三方】；社区 API（国内网络）0.76–1.1s【实测】 | T4 GPU 32.8–39.5 ms【官方基准】；本机 RTX 5060 Ti p50 10.9 ms【实测】 |
| 准确率（typed-decisions 基准） | 0.727【官方发布，Laya 仓库转引】 | 0.766（需用 `laya-typed-decisions` 精调版；base 版仅 0.362）【官方基准】 |
| 高基数选项（>20 个选项） | 强（Banking77 上 0.870，上限 255 选项）【三方】 | 弱（Banking77 默认 0.425，受 token 预算限制）【官方基准】 |
| 校准（ECE，越低越好） | 开箱 0.144 优于 Laya 开箱；温度拟合后 Laya 0.081 vs Jev 0.246【官方基准】 | 需做温度拟合，拟合后领先 |
| 多语言 | 无公开基准 | 51 语言 MASSIVE：多语言版 45/51 语言可用（>3×随机）【官方基准】 |
| 成本 | $0.042/百万 input tokens（≈$42/十亿）【官方】 | 自托管 $0（硬件+运维成本） |
| 私有化 | 不支持（云端 only，数据出境） | 完全支持（本地/内网/边缘） |
| 与 LLM 关系 | 替代 LLM 做"封闭选项决策" | 同左 + 可作 Jev 的自托管替身（`/v1/systemone` 协议兼容） |

一句话：**Jev 是"开箱即用、闭源托管"的选择；Laya 是"可私有化、可微调、更快更便宜，但 zero-shot 弱、必须自己做工程与校准"的选择。两者不是替代关系，而是一条渐进迁移路径**（详见 [02-企业落地实战指南](02-企业落地实战指南.md)）。

---

## 1. 背景：为什么需要"System 1 决策模型"

### 1.1 LLM 做决策的四个工程痛点

生产系统中大量环节本质上不是"生成文本"，而是**从有限选项中做判断**：工单该派给哪个部门？这段话是否涉及欺诈？这个请求该走小模型还是大模型？现在普遍做法是拿 LLM + prompt + JSON mode 来做，存在四个问题：

1. **延迟**：自回归逐 token 生成，一次判断动辄数秒；
2. **成本**：为输出 5 个 token 而支付整个生成过程的算力；
3. **解析脆弱**：输出是文本，需要 parse + schema 校验 + 重试，失败率不为零；
4. **过度自信**：RLHF 后的 LLM 概率校准差，置信度不能直接用作自动化门槛。

TypeSafe AI 官方给出的对照数字（【官方】，见 [typesafe.ai](https://typesafe.ai)）：典型 System 1 工作流中，LLM 路径 $0.013880/决策、8.566s；Jev 路径 $0.000081/决策、0.114s。厂商自报口径，仅作量级参考，但方向与第三方实测一致（见 §4）。

### 1.2 学术脉络：从"生成"到"判别"

- **System 1 / System 2**：借用 Kahneman 的双系统隐喻。System 1 = 快速、直觉、模式化的判断；System 2 = 慢速、链式推理。决策模型专攻前者，把 System 2 留给 LLM。
- **非自回归判别架构**：用双向 encoder（非 decoder）在**一次前向传播**中输出全部决策——没有 token 序列，就没有解析问题，也没有幻觉性输出。Laya 用的 ModernBERT-large / mmBERT-base 即属此类。
- **作者谱系**：Laya 作者 Nandakishor M 的前置工作是 **SalesRLAgent**（arXiv 2025-03-30，"A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization"，宣称销售转化预测 96.7% 准确率），Laya 是该路线的通用化；作者在 dev.to 长文中自述"一年前做了非自回归决策模型，然后一家前沿实验室把它称为突破"（暗指 Jev 验证了同一路线）。来源：[dev.to 工程长文](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)。

> 学术诚实的说明：截至 2026-09，**JEV 没有公开论文、没有开放权重、没有开放评测集**，其技术细节只有官方博客口径；Laya 也无正式论文，但其架构、训练、评测完全开源可复现，并有第三方（arXiv 2026 其他论文）引用"Laya: Typed decisions from an encoder in one forward pass"。本文学术部分以 Laya 为主展开，因为它是可验证的一方。

---

## 2. JEV 是什么

### 2.1 官方口径

- **产品**：TypeSafe AI 的 Jev，自称第一个公开的 "System One Model"，2026-09 中旬发布（官网博客《Introducing System One Models & Jev》，[typesafe.ai](https://typesafe.ai)）。
- **官方宣称**：新架构 + 新采样器（sampler）+ **RLCD**（强化学习 + 严格适当评分规则，strictly proper scoring rules）训练；相比 LLM "快 193.6×、便宜 444.6×"（官方口径，工作流级对比）；定价 **$42/十亿 input tokens**（即 $0.042/百万，见 [typesafe.ai](https://typesafe.ai)）。
- **注意**：`jevai.org` 是社区枢纽站——既展示插件与案例，也托管**免费社区 API**（见 §2.2 入口 A）；TypeSafe 官方的**商业 API** 在 api.typesafe.ai（入口 B）。两者同源（都调 Jev 模型）但 Key、限流、SLA 完全不同。

### 2.2 API 形态：一套模型，两个入口（均经实测/源码核实）

Jev 实际有**两个 API 入口**，Key 不通用——这是接入前必须分清的第一件事：

**入口 A：jevai.org 社区 API**（免费 Key，[jevai.org/agent/keys](https://www.jevai.org/agent/keys) 申请；文档 [jevai.org/docs](https://www.jevai.org/docs)；【实测】本仓库 `.env` 中的 `jev_` 前缀 Key 即此类型，调用成功）

- Base：`https://www.jevai.org`，仅 POST，Bearer 认证，请求体上限 32KiB，响应包络 `{"code": 0, "message": "ok", "data": {...}}`；
- **6 个端点**（5 个预设工作流 + 1 个原生决策）：

| 端点 | 用途 | 返回选项 |
|---|---|---|
| `/api/v1/decisions/tool-guard` | agent 工具调用守卫（给 tool/action/副作用/安全措施/政策） | allow / confirm / review / deny + needs_confirmation(noul) + risk(score 0–3) |
| `/api/v1/decisions/model-route` | 给任务选模型（候选 ≥2，附描述与优先级） | 选中的候选 + 概率 |
| `/api/v1/decisions/route` | 任务路由 | proceed_fast / deep_review / split_task / block |
| `/api/v1/decisions/research` | 证据是否支持结论 | accept / verify_more / reject |
| `/api/v1/decisions/completion` | agent 目标是否完成 | complete / verify_more / incomplete + 完成概率 |
| `/api/v1/decisions` | **原生决策：`state` + `questions`（choice/score/noul）——与 Laya `predict` 同构** | 同 Jev 商业端点 schema |

- 每个 preset 另配 MCP 工具名（`jev_guard_tool_call`、`jev_route_model` 等），可直接挂进 Claude Code/Cursor；
- 【实测】行为特征：限流**严格**（突发必 429，间隔 25–55s 仍可能 429，仅适合 PoC/调试）；站点 WAF 拦截 `Python-urllib` 默认 UA（需自定义 UA）；成功调用延迟 0.76–1.1s（国内网络，含往返）；
- 官方使用守则（文档原文要点）：概率是信号不是授权；不可逆动作保留人审；不要发送密码/密钥/无关隐私数据。

**入口 B：TypeSafe 托管商业 API**（生产路径；Key 在 typesafe.ai 申请）

- 端点：`POST https://api.typesafe.ai/v1/systemone`，Bearer 认证；另有 `GET /v1/models`；
- 官方 Python SDK：`pip install typesafe-sdk`（[github.com/typesafe-ai/typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python)），默认模型名 **`jev-latest`**（`constants.py: DEFAULT_MODEL`；第三方实测解析出具体版本 `jev-1.13.0`）；环境变量 `TYPESAFE_API_KEY / TYPESAFE_BASE_URL / TYPESAFE_DEFAULT_MODEL`；
- 请求/响应（源自 SDK `_schemas/models.py`）：`state`（str/dict/list）+ `questions`（choice/score/noul 三题型）→ `{model, answers, usage}`，SDK 提供 `.answers/.choices/.nouls` 视图；score 返回期望等级+legend+分布，choice 返回选项+confidence+全分布，noul 返回 P(true)；
- 选项数上限：**255**，256 个选项起返回 400 "Too many choices"【三方实测，[nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)】；
- 【实测】社区 Key 对该端点**不可用**（`/v1/systemone`、`/v1/models` 均 401）——两套 Key 体系互不通用。

### 2.3 已知短板（汇总，详细分析见 §6）

- 闭源黑盒：无权重、无论文、无开放评测集，审计困难；
- 数据必须出内网（托管 API），金融/医疗等强合规场景需走法务评估；
- 延迟由网络往返主导：商业端点三方实测 p50 236–276 ms（法国调用），与官方"最快 150 ms"口径有差距；社区端点【实测】国内网络 0.76–1.1s；
- 供应商锁定 + 按 token 计费（虽然单价极低）；
- **双 Key 体系易混淆**：社区 Key（jevai.org，免费、限流严）与商业 Key（typesafe.ai，付费）不互通【实测 401】；接入前先确认手里是哪种 Key；
- Key 生命周期管理：401 错误信息不含吊销/过期语义，排障成本高。

---

## 3. Laya 是什么（架构完全开源）

> 源码：[github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)（Apache 2.0，v0.3.20）；权重：HuggingFace [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)；中文用户也可从魔搭镜像获取：[modelscope.cn/models/convaiinnovations/laya](https://modelscope.cn/models/convaiinnovations/laya)。下文凡引源码路径均相对仓库根。

### 3.1 三个 checkpoint + Router

| checkpoint | encoder | 参数量 | 上下文 | 适用 |
|---|---|---|---|---|
| `laya`（english） | ModernBERT-large | 421M | 512 | 英文 |
| `laya-multilingual` | mmBERT-base | 322M | 1024（可扩至 8192） | 100+ 语言，速度约 2× |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decisions 工作流（精调版） |

`Router`（`laya/router.py`）在**前向传播之前**做脚本/语言检测（纯 Python，<0.5ms），把非英文请求路由给 multilingual。为什么必须前置路由而不是靠置信度兜底？README 的实测：英文 checkpoint 处理高棉语时**准确率 0.000 而置信度 0.952**——模型"自信地错"，置信度门控救不了，必须在进模型前分流。

### 3.2 决策头架构（魔搭社区技术长文 + 仓库源码口径）

Laya 的核心不是 encoder 本身（那是现成的 ModernBERT），而是其上的**决策头栈**：

1. **[MASK] 选项抽取**：把每个选项以 `[MASK]` 位拼进输入，encoder 对每个选项位输出一个表示向量——本质是把"生成答案"变成"抽取/打分"；
2. **两层决策头 Transformer**（约 25M 参数，d=1024）聚合上下文与选项表示；
3. **选项打分器 MLP**（约 1M 参数）输出每个选项的 logit，按题型做 softmax：choice 在选项维，score 在等级维（序数），noul 固定 `[false, true]` 两槽；
4. **每题型独立温度**（temperature scaling）做校准，数值 clamp 到 [0.5, 5]；
5. **置信度 = 1 − 归一化熵**（H(p)/log K），因此"统计上有意义"是训练目标保证的，不是启发式；
6. **act/escalate 头**：把 CLS 向量 + 4 个分布特征（最大概率、top1−top2 差、归一化熵、K/255）喂给 2 层 MLP，输出 P(自动执行) 与 P(升级人工)。训练用**成本敏感矩阵**：执行正确 +1.0、执行错误 −3.0、升级人工 −0.5 ⇒ 理论决策边界 p ≈ 62.5% 时才该自动执行。
   - **但注意**：`act_probability` 当前版本几乎没有信号（issue [#185](https://github.com/NandhaKishorM/laya/issues/185)：几乎所有输入返回 1.0，对正确性的 AUROC 仅 0.30）；应改用置信度字段做门控（同 issue 实测 AUROC 0.77）。字段区分（源码 agent.py:670-675）：`answer_confidence`（=max(p)）才是温度拟合与 ECE 度量的校准量，`confidence`（熵式）只是启发式——绝对阈值门控用 `answer_confidence`，详见 [06 源码解读](06-Laya源码解读.md) §思想三。

### 3.3 训练方法：RLCD（对严格适当评分规则做强化学习）

魔搭长文与仓库口径一致：

- **奖励 = 严格适当评分规则的组合**：log score（对数概率，clamp 在 −9.21）+ 球面分数（spherical score）− 有序概率分数（RPS，ranked probability score，用于序数 score 题）。适当评分规则的性质：模型只有输出**真实校准的概率**才能最大化期望奖励——这是"置信度可用"的数学根源，也是与 RLHF（偏好对、易被 reward hacking）最大的学术差异；
- **策略梯度**：REINFORCE + 组基线（GRPO 风格），每个样本采 G=8 个带噪候选 logit（高斯探索，σ 从 1.0 退火到 0.3），组内优势归一化；
- **多轮决策**：用 TD(λ=1.0) 的前缀回报处理轨迹型任务（源头即作者的 SalesRLAgent 销售转化轨迹）；
- **数据**：官方口径 100% 人工标注的公开数据集（MASSIVE、XNLI、AG News、BoolQ 等）。

### 3.4 工程周边（Jev 不具备或需自建的部分）

- `laya.serve`：HTTP 服务，暴露 **`POST /v1/systemone`——与 Jev 托管 API 同构的线协议**（`laya/serve.py` 明确写了 schema 兼容，hs-jev 等客户端改 baseUrl 即可切换）；
- 批量：`predict_batch`（同题多状态共享前向），`sort_by_length` 减 padding；
- 高基数短名单：`predict_shortlist`（embedding 粗筛 top-k 再进决策头）；
- 结构化输出：`agent.decide(state, schema=...)` 支持 JSON Schema / pydantic；
- 预设：`triage/guard/moderation/router/email` 五套调优好的问题组（`laya/presets.py`）；
- Hooks：`on_predict_start/end、on_route、on_load、on_evict、on_error`，可做 PII 脱敏、审计、缓存；
- MCP Server（`laya-mcp-server`）、LangChain/LangGraph 集成（`LayaRouter`、`LayaGuardrail`）、ONNX 导出、TileLang GPU fast path、Docker Compose、NixOS 模块；
- 社区生态：华为昇腾适配（laya-Ascend）、Apple MLX/ANE（laya-apple）、Google ADK 工具包、浏览器 agent 决策头案例（[docs/finetune_browser_agent.md](laya-repo/docs/finetune_browser_agent.md)：零样本元素选择 0.10 → 精调 0.66，真实任务成功率 0→62%，17–23ms/步）。

---

## 4. 性能对比：速度、准确率、校准、成本

### 4.1 延迟

| 口径 | JEV | Laya |
|---|---|---|
| 三方实测 p50（托管 API，法国调用） | 236 ms（4 选项）/ 256 ms（6 选项）/ 246 ms（72 选项）【三方，[AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks)】 | — |
| 三方实测 p50（nibzard） | 264–276 ms，2→255 选项延迟平稳【三方】 | — |
| 社区 API（jevai.org，国内网络）【实测】 | 成功调用 0.76–1.1s/次（含网络往返与限流队列；突发即 429） | — |
| 官方基准（T4 GPU，一次前向） | — | 1 题 39.5ms（en）/ 32.8ms（ml）；10 题批 158.6/72.3ms；批吞吐 103–332 题/s【官方，可复现 notebook】 |
| 知乎/魔搭口径（作者侧早期基准） | 平均 ~400ms（70–500ms，最好 150ms）【官方转述】 | p50 38.4ms / p95 42.1ms；10 题批 156ms【官方转述】 |
| 本机实测（RTX 5060 Ti，2026-09-24）【实测】 | 商业端点无 Key 未复测；社区端点见上 | 单题 n=30：p50 **10.9ms** / p95 13.9ms；3 题一次前向 ~17ms；批 100 状态（bs=100）11.5ms/条；`laya-serve` 环回 HTTP 往返 ~13.5ms |

结论：**同任务的推理延迟 Laya 领先约一个数量级（~7–8×，官方口径；本机 GPU 上差距更大）**，但注意两口径的测量位置不同——Jev 是"客户端到托管 API 的完整网络往返"，Laya 是"进程内推理"。对云端调用方而言，Laya 自托管后网络距离可控，这恰是自托管的价值之一。CPU 上 Laya 无优势：193–464ms/次【官方】，生产建议 GPU。

### 4.2 准确率（同为 Laya 官方基准跑法：字节级相同问题、固定种子；Jev 数字为官方发布值转引，样本与 prompt 不完全同，作方向性参考）

> **同题对照样本**（2026-09-24【实测】，单例不具统计性，仅作观察）：同一句中文"我三月份被扣了两次款……否则我们就取消订阅"，Jev 社区 API 给 churn_risk=**0.97**（正确识别取消威胁），Laya multilingual 本地给 **0.163**（漏检）。Jev 在该样本上明显更准，与"中文场景 Jev 开箱质量更稳"的方向一致；Laya 需精调补齐。

| 基准 | Jev 1.13.0 | Laya 路由模式 | 说明 |
|---|---|---|---|
| typed-decisions（2,000 决策，4 工作流） | 0.727 | **0.766**（typed-decisions 版） | base 版仅 0.362/0.352（低于多数类基线 0.461）——**Laya 的能力来自精调，不是 zero-shot** |
| AG News（4 类） | 0.910 | **0.950** | |
| DAIR Emotion（6 类） | 0.480 | **0.595** | Jev 在 16% 样本上给真标签概率为 0（置信度分支场景的硬伤）【官方基准】 |
| Banking77（72/77 类） | **0.870** | 0.425 | **Jev 大幅领先**：选项多时 Laya 的 head_max_len token 预算（英文 192/多语言 256）摊到每选项仅 3–4 token |
| 软分布匹配（soft accuracy） | **0.580** | 0.471 | Jev 的分布更像教师分布 |
| ECE（开箱） | **0.144** | 0.213 | Jev 开箱校准更好 |
| ECE（温度拟合后） | 0.246 | **0.081** | Laya 拟合后反超 3×；multilingual 版出厂未拟合（0.314→需自行拟合至 ~0.106） |
| 多语言（MASSIVE 51 语言，20 选项） | 无公开基准 | 多语言版 45/51 语言可用；macro acc 0.366 | 中文 zh-CN：0.63（ml）/0.62（en）——**中文可用但不顶尖，生产前必须自测+精调** |
| 序数打分（SST-5） | 无公开对照 | 0.372 | **score 是最弱题型** |

### 4.3 成本（口径需谨慎，全部标注）

- JEV：商业端点 $0.042/百万 input tokens（官方）【官方】。假设单次决策 input ≈200 tokens，则 ≈ **$0.0000084/次**；官方工作流示例口径 $0.000081/次（含 prompt 开销更大）【官方】。社区端点免费但限流严格（【实测】约几分钟放行 1 次，仅够 PoC）。
- Laya：自托管 token 成本为 0。真实成本 = 硬件/云租用 + 运维。参考：T4 级云 GPU 市价约 $0.3–0.7/小时（因云厂商而异，【未验证】仅作规划口径）；本机实测 RTX 5060 Ti 常驻两个 checkpoint 仅占 1.7GB 显存【实测】，8GB 卡即可承载可观吞吐（官方批吞吐 103–332 题/s@T4）。
- 经验结论：**日均百万级以下决策，两者成本都可忽略；真正的成本分野在（a）数据合规成本（Jev 需出境评估）、（b）大规模下 Jev 计费线性增长 vs Laya 固定基础设施成本、（c）精调的人力成本（Laya 一次精调 2×T4 约 4–5 小时，官方 notebook 口径）**。

### 4.4 部署与集成能力

| 能力 | JEV | Laya |
|---|---|---|
| 接入方式 | 商业端点 `/v1/systemone` + 官方 SDK（Python；社区有 Haskell 等）；社区端点 6 条路径（5 预设 + 原生 decisions，另有 MCP 工具 jev_decide / jev_guard_tool_call 等） | 进程内 SDK / HTTP（`laya-serve`，商业协议同构）/ MCP / LangChain |
| 私有化/离线 | ✗ | ✓（含 Docker、NixOS、ONNX、昇腾、Apple silicon） |
| 微调 | ✗（闭源） | ✓（官方 notebook，Kaggle 免费 2×T4 可跑完） |
| 权重/审计 | ✗ | ✓ Apache 2.0，评测脚本全开源 |
| Jev 客户端迁移 | — | `laya-serve` 面向**商业协议**（`/v1/systemone`）schema 兼容，改 baseUrl 即切换【实测通过，响应 `model` 字段为 `laya-rl-agent`】；社区端点是另一套线协议（`{code,message,data}` 包络、路径不同），面向社区 API 写的客户端**不能**直接指向 laya-serve |

---

## 5. 两者关系：竞品，还是上下游？

工程视角下 **Jev 与 Laya 构成一条"渐进迁移路径"**，而非二选一：

```
阶段0  规则/正则 + 人工           （有边界样例、量小）
阶段1  Jev 社区 API 免费验证       （jevai.org，限流下足够 PoC；或商业 Key 直连当天上线）
阶段2  Laya 自托管 + laya-serve    （数据不出内网；面向商业协议的客户端零改动）
阶段3  Laya + 领域精调            （typed-decisions 版证明：0.362 → 0.766）
阶段4  决策模型集群 + 数据飞轮      （置信度门控 → 人工标注 → 周期性精调）
```

Laya 仓库的 `laya-serve` 有意做成了 Jev **商业协议**（`/v1/systemone`）的替身（源码 `laya/serve.py` 开头注释即写明），这是理解两者关系的最直接证据：**同一抽象（state + questions → typed answers），两种交付形态**。注意社区端点是另一套 HTTP 面（路径与 `{code,message,data}` 包络不同），从社区 API 迁到 laya-serve 需要改客户端的路径与解包逻辑（答案字段本身同构）。

---

## 6. 不足清单（两边都不留情面）

### 6.1 JEV 的不足

1. **完全不透明**：无权重、无论文、无公开评测集与复现脚本；"永不幻觉"的宣传只能按字面理解为"不生成文本"，概率本身的质量无法独立审计（第三方基准目前只有少量独立复测）。
2. **数据主权**：state 原文必须发送到云端（商业端点或社区端点均如此）。互联网金融、医疗、政务场景大概率直接被合规否决；跨境传输需法律评估。
3. **延迟由网络主导**：商业端点三方实测 p50 ≈ 240–280ms（比官方宣传口径高 60%+）；社区端点【实测】国内网络 0.8–1.1s；对内网高频调用（如每笔交易实时风控）不友好。
4. **社区端点限流严格**【实测】：突发请求几乎必 429，仅适合 PoC 与调试，无法承载生产流量；生产必须走商业 Key。
5. **供应商锁定与经营风险**：早期访问产品，定价、SLA、模型版本（jev-1.13.0）都由厂商单方控制；无自托管退路（这正是 Laya 存在的生态位）。
6. **双 Key 体系易混淆**：社区 Key 与商业 Key 不互通【实测 401】，401/429 错误信息语义有限，排障成本高。
7. **准确率长板有限**：在 Laya 官方基准转引口径下，Jev 仅在高基数选项（>20 选项）与软分布匹配上领先；低基数常见任务上反而落后。

### 6.2 Laya 的不足

1. **zero-shot 基本不可用**：typed-decisions 上 base 版 0.362/0.352，低于多数类基线 0.461【官方自报】。开箱即用的只有通用任务（AG News 0.947、BoolQ 0.830），领域任务**必须精调或使用 typed-decisions 版**。
2. **高基数选项是架构性短板**：选项共享 `head_max_len` token 预算，77 选项每选项摊 3–4 token；Banking77 默认 0.425 vs Jev 0.870。缓解手段（`head_max_len` 调到 512、`predict_shortlist` 粗筛、两级分类）都需要额外工程。
3. **score（序数打分）是最弱题型**：SST-5 仅 0.372；multilingual 版在 score 题上有位置偏置（很少选第一个等级，issue [#131](https://github.com/NandhaKishorM/laya/issues/131)）。
4. **noul 可能跟随标签文字而非语义**：英文版在明显正例上返回自信的"no"（issue [#156](https://github.com/NandhaKishorM/laya/issues/156)），官方建议用 `labels` 覆写或改用两选项 choice 并自测。
5. **自信地错**：英文 checkpoint 处理非拉丁文字准确率 0.000 而置信度 0.952（高棉语）——必须启用 Router 且不能省。
6. **校准不是开箱即得**：multilingual 出厂**没有**拟合温度（README 明确警告）；本会话实测还复现了温度 clamp 警告（`choice:11+ 0.1005→0.5`，置信度应视为未校准）【实测】。要拿置信度做自动化门槛，**必须在自己数据上做温度拟合**。
7. **`act_probability` 无信号**（issue #185）：自动化升级判断请用 `answer_confidence`（=max(p)，校准量；AUROC 0.77 的排序结论对两个置信度字段均适用，但绝对阈值语义只在 answer_confidence 上成立，见 [06 源码解读](06-Laya源码解读.md)）。
8. **中文不顶尖**：MASSIVE zh-CN 0.62–0.63（20 选项意图），中文生产前必须自建评测集验证；本会话中文实测 3 例中 2 例存在争议（详见 [03 教程 §10](03-Laya本地使用教程-入门到精通.md)）【实测】。
9. **长文本能力有限**：multilingual 版 `max_len=8192` 下，前文约 4000 token 内表现稳定，之后波动（8–17/20 正确）【官方自报】。
10. **CPU 推理慢**：193–464ms/次，边缘无 GPU 场景需评估。
11. **项目治理风险**：个人主导的开源项目（Buy-Me-a-Coffee 赞助），bus factor 低；版本迭代快（0.2.1→0.3.20），接口有变动记录。
12. **生态与中文资料少**：文档以英文为主，中文社区资料基本只有魔搭长文一篇量级。

### 6.3 共同的边界（不要再问"它能不能写周报"）

- 只做**封闭选项决策**：选项集必须预先声明；不能生成文本、不能开放问答、不能推理。
- "永不幻觉"的正确理解：**不产生虚构文本**；但概率本身仍可能"自信地错"（两边的校准短板见上）——自动化门槛必须配置信度门控与人审回路。
- 上下文是"扁平 state"：没有多轮对话记忆、工具调用等 agentic 语义；多轮需要外层编排。

---

## 7. 与大模型配合的总模式（细节与代码见 02/03 章）

| 模式 | 数据流 | 用途 | 成本/延迟收益 |
|---|---|---|---|
| ① 级联路由（最常用） | 用户请求 → Laya 判难度/领域 → 简单走小模型或规则，复杂转 LLM | LLM 网关降本 | 官方预设 `router_questions` 即为此设计；大模型调用量通常可降 30–70%（取决于流量分布，需自测） |
| ② 护栏 | 输入 → Laya guard（越狱/注入/敏感信息）→ LLM → 输出 → Laya 复核 | 安全合规 | 拦截一个恶意请求 ≈ 10ms + 0 token，而 LLM 做同样事 ≈ 秒级 |
| ③ 结构化抽取验证 | LLM 生成候选结论 → Laya 对候选打分/选择 | 提升 LLM 输出可靠性 | 把"LLM 自由生成 + 解析"换成"LLM 列选项 + 决策模型选择" |
| ④ Agent 工具 | MCP：Claude/Cursor 等把决策当工具调。Laya 用本地 `laya-mcp-server`；Jev 社区 API 直接提供云端 MCP 工具（`jev_decide`、`jev_guard_tool_call`、`jev_route_model`、`jev_route_task`、`jev_check_research`、`jev_review_completion`） | agent 内嵌 System 1 | agent 不再为分类烧 token |
| ⑤ 数据飞轮 | 线上低置信样本 → 人工标注 → 周期性精调 Laya | 准确率持续提升 | typed-decisions 版证明精调收益 +40pt |
| ⑥ 上下文压缩决策 | agent 上下文接近上限时用决策模型判定哪些片段可丢 | 长会话 agent（社区已有 Claude Code compaction 插件同类做法，jevai.org 社区案例） | 每次压缩决策从秒级降到毫秒 |

---

## 8. 本教程的验证方法（可复现性声明）

- Laya 侧：本机装 `laya==0.3.20`（venv，Python 3.12），HF 走 hf-mirror.com；实测脚本与输出见 [03 教程](03-Laya本地使用教程-入门到精通.md) §2.5 与 [examples/](examples/)；
- Jev 侧：**社区 API 已实测打通**——`/api/v1/decisions`（原生英文+中文）、`/api/v1/decisions/tool-guard`、`/api/v1/decisions/completion` 三个端点均 200（成功调用 0.76–1.1s；`model-route` 因限流未完成成功调用）；商业端点 `api.typesafe.ai` 无商业 Key 未测（社区 Key 实测 401，两套 Key 不互通），其协议字段核实自官方 SDK 源码；
- 全部引用 URL 见 [04-参考资料.md](04-参考资料.md)，并区分"已验证（本次实抓）"与"未验证（仅搜索可见）"。
