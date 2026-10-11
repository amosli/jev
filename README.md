# jev 目录：System 1 决策模型研究（JEV × Laya）

> 本目录回答四个问题：**JEV 和 Laya 是什么、差在哪？企业里怎么真正用起来？怎么和大模型配合？Laya 本地怎么从入门到精通？**
> 调研与写作完成于 2026-09-24；所有性能数字标注口径：【实测】= 本机复现（RTX 5060 Ti 8GB / laya 0.3.20），【官方】= 厂商发布，【三方】= 独立第三方实测。

## 文档导航

| 文件 | 内容 | 读者 |
|---|---|---|
| [01-JEV与Laya深度对比.md](01-JEV与Laya深度对比.md) | 学术原理（RLCD/决策头架构）+ 工程对比（速度/准确率/校准/成本/部署）+ 双方不足清单 + LLM 配合模式总览 | 所有人，从这里开始 |
| [02-企业落地实战指南.md](02-企业落地实战指南.md) | 任务适配判断、选型决策树、六大 LLM 配合模式、成本测算、部署形态、质量保障、风控合规；**互联网金融**与**电商**两套行业方案；90 天路线图 | 架构师/技术负责人 |
| [03-Laya本地使用教程-入门到精通.md](03-Laya本地使用教程-入门到精通.md) | 安装（含国内镜像）→ CLI/API → 三种题型 → 批量/性能 → 服务化 → LangChain/MCP/Hooks → 校准/门控/高基数 → 精调 → 排错表 | 开发者 |
| [04-参考资料.md](04-参考资料.md) | 全部引用来源，区分"已验证（本次实抓）/未验证（仅搜索可见）" | 核查用 |
| [05-未知领域调优与模型路由实战.md](05-未知领域调优与模型路由实战.md) | 冷启动五阶段方法论（决策面工程→探针标注→基线→精调→飞轮）；金融资质审核/风控、电商退换货的定制落地；生产级模型路由（决策表/回放校准/影子模式/双跳成本） | 要把模型做到生产的人 |
| [06-Laya源码解读.md](06-Laya源码解读.md) | **总览**：七个核心思想（决策即抽取/校准双字段/RLCD 奖励/前置路由/行打包/防御式运行时）+ 一次 predict 完整生命周期 + 复刻权衡表 + 子文档索引 | 想真正理解原理与二次开发的人 |
| ↳ [06-1 序列构造与 MASK 机制](06-1-序列构造与MASK机制.md) | `build_sequence` 逐 token 布局、**`[MASK]` 槽位机制**、token 预算会计与高基数塌陷实测表（2→77 选项）、题型渲染、截断方向 | 机制层，想搞懂"答案为什么不是生成的" |
| ↳ [06-2 决策头与前向](06-2-决策头与前向.md) | `DecisionModel.forward` 五步（encoder→type_emb→2 层 head→gather→scorer）、参数分布、**act 头为何没信号**、head_checkpointing、RoPE 兼容陷阱 | 机制层，要改模型结构或自己写训练循环 |
| ↳ [06-3 温度与校准](06-3-温度与校准.md) | **温度机制**数学、clamp 与 RuntimeWarning、三个 checkpoint 温度实况、post-hoc 拟合四步与静默失效坑、**ECE 实测**、双置信度为何不能同阈值 | 机制层，要卡门控阈值的人**必读** |
| ↳ [06-4 RLCD 奖励与训练](06-4-RLCD奖励与训练.md) | **RLCD** 严格适当性证明、`proper_reward` 三项拆解与数值验证、GRPO 式训练循环全流程、软标签、TD(λ) 多轮、可直接改的训练骨架 | 机制层，要自己精调的人**必读** |

## 可运行示例（examples/）

| 脚本 | 场景 | 状态 |
|---|---|---|
| [laya_quickstart.py](examples/laya_quickstart.py) | 三种题型 + 路由 + 置信度门控 + 批量 | ✅ 实测通过 |
| [fintech_demo.py](examples/fintech_demo.py) | 互联网金融：工单分流 + 欺诈信号初筛 + 审计（含 PII 脱敏 hook 示例） | ✅ 实测通过（输出见文件头注释） |
| [ecommerce_demo.py](examples/ecommerce_demo.py) | 电商：大促多语言分流 + 舆情批量监控 + 退货自动化 | ✅ 实测通过（输出见文件头注释） |
| [llm_pipeline_demo.py](examples/llm_pipeline_demo.py) | 与 LLM 配合三模式：级联路由 / 护栏 / 后验证 | ✅ 实测通过（连 OpenAI 兼容网关） |
| [domain_tuning_demo.py](examples/domain_tuning_demo.py) | 冷启动阶段0-2：判别力探针（实测 54% 触发回改问题集）+ 软标签导出 + kappa | ✅ 实测通过 |
| [model_router_demo.py](examples/model_router_demo.py) | 生产级 LLM 路由：决策表 + fail-open + 影子模式 + 成本核算（实测暴露零样本路由陷阱） | ✅ 实测通过 |
| [jev_api_demo.py](examples/jev_api_demo.py) | Jev 社区 API（原生决策 + tool-guard + completion，含 429 退避）| ✅ 实测通过（限流时自动退避重试） |
| [bench_local.py](examples/bench_local.py) | 本机性能基准复现（须 GPU 独占） | ✅ 实测通过 |
| [laya_internals_demo.py](examples/laya_internals_demo.py) | **源码机制实测台**：序列逐 token 打印、预算塌陷表、决策头结构、温度实况与 ECE、双置信度分歧、RLCD 严格适当性数值验证（`--quick` 免模型加载） | ✅ 实测通过（06-1/06-2/06-3/06-4 的所有数字均出自此脚本） |

运行方式：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple laya typesafe-sdk
export HF_ENDPOINT=https://hf-mirror.com        # 国内加速权重下载
python examples/laya_quickstart.py
```

## 一分钟结论

- **JEV**：TypeSafe AI 的 System 1 决策模型。**两个 API 入口**：① jevai.org 社区 API（免费 Key，6 个端点：5 个预设工作流如 tool-guard/model-route/completion + 1 个与 Laya 同构的原生 `/api/v1/decisions`；限流严格，实测延迟 0.8–1.1s，适合 PoC）；② api.typesafe.ai 商业托管 API（`/v1/systemone`，255 选项上限，三方实测 p50 240–280ms，生产路径）。两套 Key 不互通。
- **Laya**：开源（Apache 2.0）对标实现，协议兼容 Jev（`laya-serve` 改 baseUrl 即迁移）。GPU 上快一个数量级（本机 p50 10.9ms【实测】）、可私有化、可微调；但 zero-shot 弱（typed-decisions 0.362），**必须精调 + 温度校准 + 置信度门控**才能上生产。
- **落地主线**：Jev 社区 API 免费验证 → 商业 API 或 Laya 私有化 → 领域精调 → 数据飞轮。金融场景的差异化价值在"数据不出内网 + 毫秒级护栏"，电商在大促分流与多语言。
- **与大模型的关系不是替代**：高频封闭判断给决策模型（毫秒、0 token），开放生成与总结留给 LLM（见 02 文档 §3 六种模式）。

## 本目录资产

- `laya-repo/`：Laya 源码克隆（对照阅读引用，如 `laya/serve.py` 的 Jev 兼容实现）
- `typesafe-sdk/`：TypeSafe 官方 SDK 克隆（协议字段出处）
- `.venv/`：本地可运行环境（Python 3.12）
- `~/.env` 中 `JEV_API_KEY`：**jevai.org 社区 API Key**（`jev_` 前缀），已实测可调用 6 个社区端点（限流严格，串行使用）；**不能**调用 api.typesafe.ai 商业端点（实测 401）。生产需另在 typesafe.ai 申请商业 Key（export 为 `TYPESAFE_API_KEY`）。
