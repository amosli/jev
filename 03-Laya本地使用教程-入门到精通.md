# Laya 本地使用教程：从入门到精通

> 基准版本：`laya==0.3.20`（2026-09，PyPI 最新）。所有示例在 Python 3.12 + RTX 5060 Ti 8GB (CUDA 13.2) + Ubuntu 上**实跑通过**（2026-09-24），输出为真实记录；CPU/MPS 差异处会注明。源码仓库已克隆在本目录 `laya-repo/`，可对照阅读。
> 配套脚本：[examples/laya_quickstart.py](examples/laya_quickstart.py)（本文 §2–§5 的可运行版）、[examples/bench_local.py](examples/bench_local.py)（§6 性能复现）。

---

## 1. 安装（含国内网络加速）

### 1.1 标准安装

```bash
# Python ≥ 3.10（依赖 torch 2.14 / transformers 5.x 的硬性要求）
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install laya
python -I -c "import laya; print(laya.__version__)"   # -I 防止本地目录遮蔽安装
```

可选增强（按需）：

```bash
pip install "laya[serve]"       # HTTP 服务（fastapi+uvicorn）
pip install "laya[mcp]"         # MCP Server
pip install "laya[langchain]"   # LangChain/LangGraph 集成
pip install "laya[onnx]"        # ONNX Runtime 推理
pip install "laya[fast]"        # TileLang GPU fast path（CUDA）
pip install "laya[structured]"  # pydantic schema 决策
```

### 1.2 国内网络加速（实测有效）

```bash
# PyPI 走清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple laya

# HuggingFace 权重走 hf-mirror（模型首次使用时自动下载，英文包约 1 分钟）
export HF_ENDPOINT=https://hf-mirror.com

# 或从魔搭社区取权重后指定本地目录加载（模型页：modelscope.cn/models/convaiinnovations/laya）
pip install modelscope
modelscope download --model convaiinnovations/laya --local_dir ./laya-weights
agent = laya.load("./laya-weights")   # 本地目录需含 model.safetensors + rl_agent_config.json
```

### 1.3 GPU 环境

CUDA 版 torch 单独装（先 venv、再按 [pytorch.org](https://pytorch.org/get-started/locally/) 装、最后装 laya）。没有 GPU 也能跑：CPU 193–464ms/次【官方】——开发调试够用，生产吞吐不够。Apple silicon 走 MPS 自动适配，或用社区 laya-apple（MLX/ANE）。

---

## 2. 五分钟上手

### 2.1 CLI（最快验证）

```bash
laya "I was charged twice, please refund"   # 纯路由判断，毫秒级、不下载权重
laya "Refactor this service" --predict      # 完整决策（首次下载权重）
laya "Mein Konto wurde zweimal belastet" --lang de
laya "My payment failed twice" --preset triage   # 内置预设：triage/email/guard/moderation/router
laya                                         # 交互模式
```

### 2.2 Router 模式（推荐入口）

```python
from laya import Router

router = Router()            # 懒加载；生产用 Router(preload=True)
state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}
result = router.predict(state, questions)
```

### 2.3 真实输出（本机实跑记录【实测】）

```python
result["answers"]["department"]
# {'type': 'choice', 'choice': 'billing',
#  'probabilities': {'billing': 0.9867, 'technical': 0.0078, 'other': 0.0055},
#  'confidence': 0.9275, 'answer_confidence': 0.9867, 'action': {'act_probability': 1.0}}
result["answers"]["urgency"]
# {'type': 'score', 'score': 1.7764, 'legend': {'0': 'not urgent', '1': 'soon', '2': 'blocking'},
#  'probabilities': {'0': 0.0386, '1': 0.1464, '2': 0.815}, 'confidence': 0.4778, ...}
result["answers"]["churn_risk"]
# {'type': 'noul', 'noul': 0.878, 'confidence': 0.878, ...}
result["routing"]   # {'model': 'english', 'repo': 'convaiinnovations/laya/english', 'reason': '...'}
result["usage"]     # {'input_tokens': 164, 'output_tokens': 0}   ← 决策不产生输出 token
```

读法：

- `choice/score/noul` 是答案本体；`probabilities` 是全分布（做门控比只看 argmax 好）；
- **两个置信度字段要分清**（源码 agent.py:670-675）：`confidence` 在 choice/score 上是分布集中度（`1−H/log K`），只是启发式信号；`answer_confidence` = `max(p)` 才是**温度拟合与 ECE 度量的对象**——"置信度 c 的答案约 c 比例正确"的保证只对它成立。**绝对阈值门控请用 `answer_confidence`**（noul 上两者恰好相等）；
- `routing.reason` 是人话解释（如 "non-Latin script (han, 100% of letters); the English checkpoint cannot read it"），排障好用；
- `action.act_probability` 当前无信号（issue #185），**门控一律不要用它**。

### 2.4 中文与多语言

```python
r = router.predict({"body": "我三月份被扣了两次款，请今天退回重复扣的钱，否则我们就取消订阅。"}, questions)
r["routing"]["model"]   # 'multilingual'（Router 检测汉字自动切换）
```

中文实测观察【实测】：路由与部门判断（billing/technical）正常；但 3 条样本中 2 条有争议——明确"否则取消订阅"的 churn_risk 只有 0.163；"理财产品收益怎么计算"被判成 billing（0.97）而非 other。**结论：中文能跑，但选项描述要写得更细，且上生产前必须自建中文评测集验证**（多语言 checkpoint 在 MASSIVE zh-CN 20 选项意图任务上 0.63【官方基准】，不属于顶尖水平）。

### 2.5 冷启动与延迟（真实数字，便于容量规划【实测】）

| 阶段 | 实测 |
|---|---|
| 权重下载（hf-mirror） | english ≈58s；multilingual ≈46s |
| 首次 predict（含加载建图） | english ≈66s；multilingual ≈54s |
| 热身后单次 predict（3 题一次前向） | 16.7–17.6ms |
| 英文单题 n=30 | p50 **10.9ms**，p95 13.9ms |
| 双 checkpoint 常驻显存 | 1.72GB |

生产含义：**进程启动后必须预热**（`Router(preload=True)` + 一次 dummy predict），否则第一个线上请求要吃 1 分钟加载。

---

## 3. 三种题型精讲

### 3.1 choice（最常用）

```python
{"type": "choice",
 "instructions": "Which team should handle this?",
 "criteria": {"billing": "invoices, payments, refunds",   # 键=返回的标签，值=给模型看的描述
              "technical": "bugs, outages, system errors"}}
```

- **选项描述比选项名重要**：模型主要读描述。`{"yes": None, "no": None}` 这类布尔词标签会出事（README 明确警告：模型跟随标签词）；
- 选项数 2–10 是舒适区；20+ 开始需要关注（见 §8.3 高基数）。

### 3.2 score（序数打分，最弱题型，慎用）

```python
{"type": "score", "instructions": "How urgent is this?",
 "criteria": ["not urgent", "soon", "blocking"]}   # 有序等级，从 0 开始
```

- 返回期望分（如 1.78 = 介于 soon 与 blocking 之间偏 blocking），`probabilities` 是各等级分布；
- **已知短板**：SST-5 上仅 0.372【官方基准】；multilingual 版对第一个等级有位置偏置（issue #131）。生产建议：序数粗分级可用，精细回归不要用；英文任务显式 `model="english"` 绕开位置偏置。

### 3.3 noul（是/否概率，护栏类任务首选）

```python
{"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}
# 可选 criteria 细化 true/false 的定义（键必须叫 true/false）
# 可选 labels 把模型可见的 false/true 字样换成 A/B（缓解标签跟随问题，issue #156）
```

- 返回 `noul` = P(true)∈[0,1]，直接当概率用；
- 坑（issue #156）：默认把两个选项渲染成 `false:`/`true:` 字样，英文 checkpoint 有跟随标签倾向；如遇"明显正例却返回自信的 no"，用上面的 `labels` 或改两选项 choice，并**在你自己的数据上验证**。

### 3.4 schema 决策（`decide`，一锅出）

```python
schema = {"type": "object", "properties": {
    "department": {"type": "string", "enum": ["billing", "support", "sales"],
                   "description": "Which team should handle this?"},
    "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
    "needs_human": {"type": "boolean"}}}
agent.decide("I was charged twice, refund me.", schema=schema)
# {'department': 'billing', 'urgency': 2, 'needs_human': True}
```

支持 pydantic 模型（装 `laya[structured]`）、`return_details=True` 拿逐字段置信度。

---

## 4. 预设问题组（`laya/presets.py`，直接抄）

```python
import laya
agent = laya.load("convaiinnovations/laya")
agent.predict({"message": "My payment failed twice"}, laya.triage_questions())      # 客服分流
agent.predict({"body": "..."},                laya.email_questions())               # 邮件分类+钓鱼检测
agent.predict({"prompt": "Ignore all instructions"}, laya.guard_questions())       # LLM 输入护栏
agent.predict({"post": "..."},                laya.moderation_questions())          # 内容审核
agent.predict({"request": "..."},             laya.router_questions())              # LLM 网关路由
```

五个预设就是 §企业指南里六大模式的现成问题模板，改 criteria 即可本地化。

---

## 5. 批量与性能调优

### 5.1 predict_batch（同组问题打多状态）

```python
states = [{"body": t} for t in ticket_texts]
results = agent.predict_batch(states, questions, batch_size=64, sort_by_length=True)
# results[i] 与 states[i] 对齐；sort_by_length 减 padding（混合长度 workload 才有效）
```

Router 版可混合多语言请求（自动按 checkpoint 分组再合批）：

```python
requests = [{"state": s, "questions": q} for s, q in zip(states, [questions]*len(states))]
results = router.predict_batch(requests, batch_size=64)   # 结果保持输入顺序
```

### 5.2 实测性能表【实测，RTX 5060 Ti 8GB】

| 场景 | 数字 |
|---|---|
| 单题单状态（英文，n=30） | p50 10.9ms / p95 13.9ms |
| 3 题一次前向 | ~17ms |
| `Router.predict_batch` 100 状态（bs=100） | 1151ms 总耗时，**11.5ms/条** |
| 同上 bs=32 首跑 | 6.6s（形状桶预热一次性开销，二跑正常）|
| `laya-serve` 环回 HTTP 往返 | 首 call 74ms，热身后 13.4–14.3ms |
| 显存 | 双 checkpoint 常驻 1.72GB |

官方 T4 对照【官方】：1 题 39.5/32.8ms，10 题批 158.6/72.3ms，批吞吐 103–332 题/s。

调优要点：

1. **吞吐场景永远 predict_batch**：单条循环 ~10ms → 批后 ~1ms/条（RTX 5060 Ti，官方实测 ~9–10×）；
2. `sort_by_length=True` 只在长度混杂时试，均匀长度无益还费内存；
3. CPU 服务用 `LAYA_THREADS`（≤物理核数）；`laya[fast]` TileLang fast path 仅 CUDA（fp32 偏差 ≤0.046，官方 parity 数据）；
4. 改 batch 形状会有微小浮点漂移——**阈值决策系统要在自己 workload 上回归**（README 原话）。

---

## 6. 服务化

### 6.1 带 Web UI 的调试服务器

```bash
pip install "laya[serve]"
python examples/server.py          # http://127.0.0.1:8000 表单式构造请求，答案渲染成 0-100 条形图
python examples/server.py --device cuda --no-preload
```

### 6.2 生产 HTTP 服务（Jev 兼容协议）

```bash
LAYA_DEVICE=cuda LAYA_PRELOAD=1 LAYA_MODELS=english,multilingual LAYA_PORT=8000 \
LAYA_API_KEY=your-secret laya-serve
```

```bash
curl -s localhost:8000/v1/systemone -H "Authorization: Bearer your-secret" \
  -H 'content-type: application/json' -d '{
  "state": {"body": "billed twice, refund please or we cancel"},
  "questions": {"dept": {"type": "choice", "instructions": "which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"}}}}'
```

真实响应（本机实跑【实测】，注意与 Jev 响应 schema 一致——`model/answers/usage`）：

```json
{"model": "laya-rl-agent",
 "answers": {"dept": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 1.0, "tech": 0.0},
              "confidence": 0.9995, "answer_confidence": 1.0, "action": {"act_probability": 1.0}}},
 "usage": {"input_tokens": 39, "output_tokens": 0},
 "routing": {"model": "multilingual", "repo": "convaiinnovations/laya/multilingual",
             "reason": "non-Latin script (han, 100% of letters); the English checkpoint cannot read it"}}
```

运维须知（源码 `laya/serve.py` 核实）：内置限流 64 问题/50k 字符/2MB body；Bearer 用常数时间比较；**单推理 worker 是有意设计**（GPU 一次一前向），扩容=多副本+负载均衡；`GET /health` 可接 LB 探活。

### 6.3 Docker 与 NixOS

```bash
# Docker（权重缓存在卷里，重启不重新下载——见 docs/docker.md）
docker compose -f compose.cuda.yaml up      # CUDA；compose.yaml 为 CPU
```

NixOS 用户直接 `services.laya-serve.enable = true;`（systemd 加固单元、凭据经 LoadCredential 注入）。

---

## 7. 集成：LangChain / MCP / Hooks

### 7.1 LangChain / LangGraph

```python
from laya.integrations.langchain import LayaRouter, LayaGuardrail

router = LayaRouter(criteria={"billing": "invoices, charges", "tech": "bugs, outages"},
                    confidence_threshold=0.80, fallback="human_agent")
workflow.add_conditional_edges("triage", router)      # LangGraph 条件边

guard = LayaGuardrail(action="raise")                  # 越狱/注入时抛 LayaGuardrailError
```

### 7.2 MCP Server（给 Claude Desktop / Cursor 等用）

```bash
pip install "laya[mcp]"
laya-mcp-server        # stdio；工具：laya_predict / laya_route / laya_preset / laya_status
```

```json
{"mcpServers": {"laya": {"command": "laya-mcp-server", "env": {"LAYA_DEVICE": "cpu"}}}}
```

### 7.3 Hooks（脱敏/审计/缓存的标准挂点）

```python
import laya

def redact(ctx):
    # on_predict_start：改写 state 后再进模型（PII 脱敏放这里）
    ...

def audit(ctx):
    print(ctx.model, ctx.results[0]["answers"], ctx.elapsed_ms)   # 审计日志

agent = laya.load("convaiinnovations/laya", on_predict_start=redact, on_predict_end=audit)
```

支持 `on_predict_start/end、on_route、on_load、on_evict、on_error`；Router 上安装则每个请求都跑（含 batch）。完整生命周期见 laya-repo `docs/hooks/`。

---

## 8. 进阶技巧

### 8.1 置信度门控（生产自动化门槛）

```python
ans = result["answers"]["department"]
# answer_confidence = max(p)，是温度拟合/ECE 度量的校准量（agent.py:670-675）；
# confidence 是熵式启发式，绝对值没有概率语义，不要做阈值门控
if ans["answer_confidence"] >= 0.85:
    route_automatically(ans["choice"])          # 写审计日志
else:
    escalate_to_human(ans, reason=f"low confidence {ans['answer_confidence']:.2f}")
```

前提：**先做温度拟合**（见 §8.2），否则阈值没有统计含义。

### 8.2 温度校准（让置信度变成真概率）

- 原理：对 (题型, 选项数) 桶各拟合一个温度，softmax 前除以温度。官方数据：拟合后 ECE 0.466→0.081（english）/0.314→0.106（multilingual）；
- **multilingual 版出厂没带拟合温度，必须自己拟合**；english 版若加载时出现 `RuntimeWarning: ... invalid temperatures ... using choice:11+=... -> 0.5`（本机实测出现过），说明该桶温度无效被 clamp，**该桶置信度视为未校准**；
- 做法：按官方精调 notebook 的标定环节，用你自己的 held-out 标注数据拟合；加载时数值 clamp 到 [0.5, 5]，越界项回落 1.0 并告警。

### 8.3 高基数选项（>20 个选项）三板斧

1. 调预算：`agent.cfg["head_max_len"] = 512`（甚至 1024）+ 提高 `max_len`——每选项 token 变多；
2. 短名单：embedding 粗筛 top-k 再决策（77 选项场景从 0.425 起步救回的路径；社区报告 top-20 短名单把 BANKING77 从 54.3%→60.8%， reporters 数据【未复核】）：

```python
import laya
result = laya.predict_shortlist(agent, {"text": "I was charged twice for a transfer"},
                                questions, embed_fn=laya.embed_fn_from_agent(agent), k=20)
```

3. 两级分类：先粗类（<10 选项）再细类。

### 8.4 语言路由进阶

```python
from laya.lang import analyse
analyse("Care este ora in Tokyo?")     # 看内置检测器的判断

# 自带检测是"尽力而为"的启发式；已有 LID 模型就交给它：
router = Router(preload=True, lang_guess=my_fasttext_lid)   # 或单次 predict(..., lang_guess="ro")
router = Router(default="multilingual")   # 非英文流量为主时改默认
```

### 8.5 显式指定 checkpoint

```python
router.predict(state, questions, model="english")        # / multilingual / typed-decisions
router.route(state, questions)                           # 只看路由决策，不跑前向
router.predict_batch(requests)                           # 混合请求自动按 checkpoint 分组合批
```

---

## 9. 精调（从"能用"到"好用"的分水岭）

- **为什么必须精调**：base checkpoint 在 typed-decisions 上 0.362（低于多数类基线 0.461），精调版 0.766【官方基准】。作者原话：Laya 是"值得专门化的快速底座，不是 zero-shot 决策引擎"；
- **官方路径**：[notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb](laya-repo/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)——Kaggle 免费 2×T4，4 epochs ≈ 4–5 小时（~30k 问题），一条龙：造数据 → RLCD 训练 → 温度标定 → 评测 → 推送 HF；
- **数据配方**：从生产日志取"人工处理结果"反向构造（工单→实际派单部门；评论→人工审核结论）；低置信样本优先标；
- **精调后必做**：held-out 评测（不要用训练数据标定温度后自欺）+ 灰度；
- **真实案例**：[docs/finetune_browser_agent.md](laya-repo/docs/finetune_browser_agent.md)——单卡 16GB、零 API 成本，把浏览器 agent 的元素选择从 0.10 提到 0.66、任务成功率 0→62%（17–23ms/步），含"什么没用"的诚实记录；
- **社区案例**：stuntd 项目——冻结 encoder、按决策训练头，12 标签意图任务零样本 89.5% → 训练后 100%（单方报告）。

---

## 10. 已知坑与排错表（含本机实跑遇到的真实报错）

| 症状 | 原因 | 处置 |
|---|---|---|
| `ModuleNotFoundError: No module named 'laya'` | 装和跑用了不同 Python | 全程用同一 venv 的解释器 |
| PEP 668 `externally-managed-environment` | 系统Python禁止pip | 用 venv（本文 §1.1） |
| 下载权重慢/失败 | HF 直连不通 | `export HF_ENDPOINT=https://hf-mirror.com` 或魔搭下载（§1.2） |
| `ModuleNotFoundError: uvicorn`（laya-serve 起不来）【实测】 | 没装 serve extra | `pip install "laya[serve]"` |
| 首个请求耗时 ~1 分钟【实测】 | 冷加载 checkpoint | `Router(preload=True)` + 启动后 dummy predict 预热 |
| `RuntimeWarning: invalid temperatures ... -> 0.5`【实测】 | checkpoint 温度无效被 clamp | 该桶置信度不可信；自拟合温度（§8.2） |
| `Router.predict_batch() got multiple values ...` | 传参形式错了 | requests=[{"state":…, "questions":…}]，questions 放请求字典里 |
| curl 打到 405/404 | 端口被别的进程占用【实测踩过】 | `ss -tlnp \| grep <port>` 确认监听进程 |
| noul 对明显正例返回自信的 no | 标签跟随（issue #156） | `labels` 换 A/B 或改两选项 choice + 自测 |
| score 总不选第一个等级 | multilingual 位置偏置（issue #131） | 英文任务 `model="english"` |
| 非英文文本置信度极高但答案错 | 英文 checkpoint 读不了该语言 | Router 别关；短文本用 `lang_guess` |
| `act_probability` 恒为 1.0 | 已知无信号（issue #185） | 用 `confidence` 门控 |
| >255 选项请求报 400 | laya-serve 上限 64 问题/请求；选项受 token 预算 | 拆请求；高基数走 §8.3 |
| CPU 上改 batch_size 没变快 | CPU 批收益小 | 试 `sort_by_length`；或上 GPU |
| 显存 OOM | 三 checkpoint 全 preload + 大 batch | `LAYA_MODELS` 按需加载、调小 batch_size |
| 显存 OOM 且延迟骤增至秒级【实测】 | 同卡混部：本地 LLM（Ollama/vLLM）占着显存，Laya 降级 CPU（实测 10.9ms→~1700ms） | Laya 与 LLM 分卡部署，或 Laya 用 CPU/ONNX 形态 |

---

## 11. FAQ

**Q：能完全离线吗？** 能。权重下载后，`laya.load("./local-weights")` 指本地目录（须含 `model.safetensors` 与 `rl_agent_config.json`），推理全程无网络。

**Q：显存/内存要多少？** 推理：双 checkpoint 常驻 1.7GB 显存【实测】；CPU 模式吃内存但可行（~4–8GB RAM 量级，权重 bf16/fp32 加载）。精调：单卡 16GB 可行（官方浏览器 agent 案例），标准 notebook 用 2×T4。

**Q：和直接 fine-tune 一个 BERT 分类器比，为什么要用 Laya？** 三点：一次前向出**多个异构决策**（choice+score+noul 同题），传统分类器一题一模；置信度经适当评分规则训练+温度标定，天然可做门控；问题集是**数据**不是代码——加一个问题不用重训（除非要领域精度）。

**Q：什么时候该用 Jev 而不是 Laya？** 高基数选项（>20，Jev 上限 255 且质量好）、不想管 GPU/校准/精调、数据允许出境、要最快 PoC。反过来选 Laya。混合策略（Jev 主 + Laya 热备/替代）见企业指南 §2。想零成本先感受 Jev：在 [jevai.org/agent/keys](https://www.jevai.org/agent/keys) 申请免费社区 Key，调 `/api/v1/decisions`（state+questions，与 Laya predict 同构）即可对比（限流严格，见 [examples/jev_api_demo.py](examples/jev_api_demo.py)）。

**Q：中文任务能不能直接上？** 能跑但不建议裸上：多语言版 zh-CN 意图 0.63【官方基准】+ 本机中文实测有争议样本【实测】。正确姿势：中文选项描述写细 + 自建评测集 + 精调，流程与英文一致。
