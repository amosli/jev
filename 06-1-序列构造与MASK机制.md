# 06-1 序列构造与 [MASK] 机制：Laya 把"选项"放进序列的方式

> 本文是 `06-Laya源码解读.md` 的子文档，专注 Laya 最核心的架构决策：**为什么答案不是"生成"出来的，而是"抽"出来的？`build_sequence` 这个 50 行的函数里，每个 token 都有讲究。**
> 全部行号指向本目录 `laya-repo/`（v0.3.20）。带 `实测` 标记的输出来自 `examples/laya_internals_demo.py §1`（2026-09-26，真实 ModernBERT tokenizer + english checkpoint）。
> 这一节是理解整个包的钥匙：`06-2`（前向）、`06-3`（温度）、`06-4`（RLCD）都建立在本文的序列布局之上。

---

## 0. 一句话结论

Laya 是一个**非自回归（non-autoregressive）的抽取式决策模型**：答案不是逐 token 生成、不是 JSON 解析，而是**"答案 = 哪个位置"**。它把每个候选项前面挂一个 `[MASK]` token，然后对这 K 个 `[MASK]` 位置各跑一次打分头——**K 个选项 = K 个标记 = 一次前向全出分**。这套机制的成本全在"怎么把选项塞进序列、每个选项分到多少 token"这套显式的**预算会计**里。

---

## 1. 完整序列布局（实测逐 token 打印）

`build_sequence`（`common.py:70-120`）的 docstring 写死了格式：

```
[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
      └──────── head 区（预算 head_max_len）───────┘     └─ 剩余预算给 state ─┘
```

用真实 tokenizer 跑一个 4 选项的工单派单题（`examples/laya_internals_demo.py §1`），**逐 token 打印**的结果：

```
 0 '[CLS]'
 1 'choice'              ← 题型 token
 2 'Ġquestion'
 3 ':'
 4 'ĠWhich'  5 'Ġdepartment'  6 'Ġshould'  7 'Ġown'  8 'Ġthis'  9 'Ġticket' 10 '?'
11 '[SEP]'
12 '[MASK]'   <-- marker[0]      ← billing
13 'Ġbilling' 14 ':' 15 'Ġinvoice' 16 ',' 17 'Ġpayment' 18 ',' 19 'Ġrefund' 20 ',' 21 'Ġsubscription' 22 'Ġcharge'
23 '[MASK]'   <-- marker[1]      ← technical
24 'Ġtechnical' 25 ':' 26 'Ġcrash' 27 ',' 28 'Ġerror' 29 ',' 30 'Ġcannot' 31 'Ġlog' 32 'Ġin'
33 '[MASK]'   <-- marker[2]      ← account
34 'Ġaccount' 35 ':' 36 'Ġlogin' 37 ',' 38 'Ġpassword' 39 ',' 40 'Ġtwo' 41 '-' 42 'Ġfactor'
43 '[MASK]'   <-- marker[3]      ← sales
44 'Ġsales' 45 ':' 46 'Ġquote' 47 ',' 48 'Ġcontract' 49 ',' 50 'Ġupgrade'
51 '[SEP]'
52 'I' 53 'Ġwas' ...            ← state（工单正文）
    ...
```

对应现代 BERT 系 tokenizer 的特殊 id（实测）：`[MASK]=50284`、`[CLS]=50281`、`[SEP]=50282`。

**marker 位置记录**：`build_sequence` 返回 `(ids, markers)`，markers = `[12, 23, 33, 43]`——**这 4 个位置就是 4 个选项的"槽位"**。决策头在 `forward` 里对这 4 个位置各 `gather` 一次、出 4 个 logit（见 `06-2`）。

---

## 2. 六个布局决策，逐个拆解

### 2.1 为什么选项放 state 之前（head 区）

序列被 `build_sequence` 切成两半：

- **head 区**：`[CLS] <type> instructions [SEP] [MASK] opt0 … `，**预算上限 `head_max_len`**（英文 192，多语言/typed-decisions 256）。这里是**决策头直接读取**的区域；
- **state 区**：`[SEP] state [SEP]`，**吃满 `max_len − head 区实际长度` 的所有剩余 token**。state 是"证据"（工单正文、DOM、会话历史），可以很长、可以被整体截断，因为模型对它的需求是"读个大概"而不是"逐字定位"。

**为什么必须这样切**：决策头只从 head 区的 `[MASK]` 位置出分。如果选项被塞进 state 的 JSON 里（很多通用方案的做法），选项就落在"可以被截断"的地带——长文一截断，候选项全没了，模型压根没看见要选什么。**把候选项从 state 提到 head，是结构性防御，不是调参。**

这个判断有官方数据背书。`docs/finetune_browser_agent.md` 的 "What mattered most: the input format" 一节：把元素从 state JSON 里挪进选项区（`head_max_len` 提到 512→768，state 只留标题/URL/历史/正文片段），**收益超过任何数据改动**——Mind2Web 点击 top-1 从 0.44 → 0.51，线上套件从 6/16 → 10/16。同样的数据，纯粹因为"候选项不再被截断"。

> 这条经验是 Laya 布局哲学的最强证据：**在非自回归抽取式框架里，"选项放在哪"比"训练多少数据"更重要。**

### 2.2 每个选项 = 一个 `[MASK]` + 选项描述

`common.py:95-101`：

```python
opt_tokens = tok(" " + opts[i].replace(mask_tok, " "),
                 add_special_tokens=False,
                 truncation=True, max_length=48)["input_ids"]
opt_ids.append([tok.mask_token_id] + opt_tokens)     # [MASK] 前置
```

- **每个选项前面强制挂一个 `[MASK]`**。这个 `[MASK]` 不参与语义，它是个**占位标记**——`forward` 用 `torch.gather` 在这些位置抽出每个选项的向量，送进打分头；
- **选项描述截断到 48 token**（`truncation=True, max_length=48`，在 tokenizer 层截，注释说明这是为了避免"先全 tokenize 再切片"的浪费）。48 是**单选项上限**，但实际上多选项时会被 §2.3 的预算进一步压到远小于 48；
- **`.replace(mask_tok, " ")`**：把 state、options、instructions 里所有字面的 `[MASK]` 换成空格——**防注入**。否则用户输入里一个 `[MASK]` 就会在序列里造出一个假的 marker 位置，污染决策头的 gather。三个位置（options / instructions / state）都做了这个替换（`common.py:88, 96, 116`）。

**K 个选项 = K 个 marker = 一次前向出 K 个分**。这是整个架构的效率来源：不管 2 个选项还是 20 个选项，**都是一次 encoder 前向**，只是后面那个小 MLP/gather 头跑 K 遍。没有自回归解码，没有 KV cache，没有逐 token 生成的延迟。

### 2.3 token 预算会计：高基数的塌陷点（核心机制）

`common.py:102-107`——**这是全文最该逐行读的 6 行**：

```python
opt_budget = head_max_len - sum(len(o) for o in opt_ids)   # head 区剩余预算
if opt_budget < 16:                                        # 剩余不够放 instructions 了
    per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))  # 每选项均分预算
    opt_ids = [o[:per] for o in opt_ids]                   # 砍到每选项 per 个 token
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
head_ids = head_ids[: max(8, opt_budget)]                  # instructions 吃掉剩余
```

**`head_max_len` 是所有选项共享的预算池。** 每个选项分到 `head_max_len / K` 左右。所以：

> **选项的区分度 = `head_max_len / K`。** 选项数一多，每个选项能写进序列的描述就线性变少，模型读到的不是"选项"，是"选项的前缀"。

#### 实测：head_max_len 塌陷表（`examples/laya_internals_demo.py §1b`）

英文 checkpoint `head_max_len=192`，选项数递增：

| 选项数 K | 每选项预算 | instructions 剩余 | 说明 |
|---|---|---|---|
| 2 | 96 | 144 | 宽裕 |
| 4 | 48 | 96 | 舒服 |
| 8 | 24 | 16 | 触发兜底：每选项砍到 22 token |
| 16 | 12 | 16 | 兜底：每选项 11 token |
| 32 | 6 | 32 | 兜底：每选项 5 token |
| **77** | **2** | **8** | 兜底：每选项只剩 **4 token** |

读法：K=2/4 时预算充裕，选项描述能完整写入；K≥8 触发 `opt_budget < 16` 兜底，每个选项被砍到 `max(4, (192−16)//K)` 个 token。**K=77 时每选项只剩 `[MASK]+4 token`——选项描述几乎全被截掉。**

#### 这个塌陷直接解释了官方 benchmark 的一个"损失"

`laya-repo/BENCHMARKS.md:147-149`：

> | banking77 (77 labels) | 0.425 | 0.425 | **0.492** | 0.870 |
> banking77 is the one clear loss, and it is architectural: a choice question's options share a fixed `head_max_len` budget, so 77 labels get roughly 4 tokens each and stop being distinguishable. **Both checkpoints score exactly 0.425**, which is what you would expect from a budget ceiling rather than a capability gap. Keep choice questions under ~20 options.

**两个 checkpoint 都恰好 0.425**——作者自己指出这是"预算天花板"而不是"能力差距"：77 个选项每个只分到约 4 token，选项之间不再可区分，模型压根没法分辨它们。官方结论：**choice 问题控制在 20 个选项以内。**

> 这张表把架构缺陷和模型能力清晰分开了：0.425 不是模型不认识 banking77 的意图，是序列里根本没给选项足够的描述空间。这种"能算出根因到具体行号"的缺陷，在其他模型里很难做到。

#### 三条应对（对应 `05-未知领域调优与模型路由实战.md`）

1. **调大 `head_max_len`**：`agent.cfg["head_max_len"] = 512`。但要意识到这会压缩 state 预算（`max_len − head_max_len`），长文场景要一起调 `max_len`（english 512→1024+）。见 `03 教程 §8.3`；
2. **shortlist 粗筛**（`laya/shortlist.py`）：先用 embedding 把 77 个标签粗筛到 top-10 再进决策面，让决策问题落在 `choice:6-10` 桶（那个桶出厂校准是健康的 1.0）；
3. **两级分类**：77 类先粗分成 8-10 个大类，再在大类内细分。

### 2.4 instructions 也要挤预算

`head_ids = head_ids[: max(8, opt_budget)]`（`common.py:107`）——**指令文本和选项共享同一个 head 预算池**。选项一多，instructions 就被压缩（`max(8, ...)` 保底 8 token）。

实测表里 "instructions 剩余" 那一列就是这个：K=77 时 options 吃掉几乎全部 192 预算，instructions 只剩 8 token——"question:" 后面的话基本没了。**高基数时，问题描述本身也会被牺牲**，这是很容易被忽略的双重代价。

> 工程含义：高基数场景里，**选项描述要写得极其精炼**（每选项 4-10 token 就要能区分），而不是把长定义堆上去——反正会被砍。反而是把区分性关键词前置更有效。

### 2.5 state 的截断方向：会话从左截断（易踩坑）

`common.py:114-120`：

```python
room = max(0, max_len - len(ids) - 1)
st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
ids = ids + st + [tok.sep_token_id]
return ids[:max_len], [m for m in markers if m < max_len]
```

- **默认右截断**（`truncate_left=False`）：保留 state 开头。适合文档、正文这种"重点在前面"的；
- **会话列表要左截断**（`truncate_left=True`）：时间序 newest-last，右截断会**静默丢掉最新一轮**。`agent.py:566-568` 的注释点明了这点，`predict_batch` 对会话列表自动用左截断。这是复刻时最容易搞反的一个方向。

> 源码还留了一条防御性注释（`common.py:117`）：
> ```python
> # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
> ```
> **不能用 `state_ids[-room:]`**——当 `room=0` 时，`x[-0:]` 是**整个列表**（`x[0:]`），不是空列表。必须用 `x[max(0, len(x)-room):]` 显式处理。Python 的 `-0 == 0` 这个坑，作者显式注释挡掉了。

### 2.6 marker 也要按 max_len 过滤

`common.py:120`：`[m for m in markers if m < max_len]`。

序列末尾 `ids[:max_len]` 截断后，某些尾部选项的 `[MASK]` 位置可能已经超出 `max_len` 被切掉了——`forward` 里的 `gather` 用无效 marker 会索引错。**过滤掉越界 marker，保证 gather 的每个位置都真实存在。** 对应 `collate_items` 里的 `marker_mask`（`common.py:339`）标记哪些 marker 有效，scorer 后用 `masked_fill(~marker_mask, -1e4)` 把无效位的 logit 压掉（见 `06-2`）。

---

## 3. 题型如何渲染进同一个格式

三种题型共用同一套序列格式，只是 `render_options`（`common.py:49-67`）的渲染不同：

| 题型 | 渲染成 | 语义位置 | K |
|---|---|---|---|
| **choice** | `标签名: 描述` | 无序 | = 选项数 |
| **score** | `level i: 描述`（index 0 起） | **有序**（位置即序数） | = 档位数 |
| **noul** | `[false_label: …no, the statement does not hold, true_label: …yes, the statement holds]` | 无序二值 | 恒 2 |

（`choice` 的 `labels` 键在 `render_options` 里会报错——`labels` 只支持 noul，见 `common.py:52-53`。）

**关键设计**：题型差异被压到**渲染文本**和**解码方式**两处，**主干和决策头是共享的**。`<type>` token 放在 instructions 前面（`build_sequence`：`"%s question: %s" % (q["t"], ins)`），`forward` 里还会把它加到每个 token 的表示上（`type_emb`，见 `06-2`）。

三种题型、同一个模型、同一次前向、同一套校准——这是 Laya "一个 checkpoint 干三件事"的实现基础。

> noul 的 `criteria` 只能有 `true`/`false` 两个键（`agent.py:518-525`，issue #156/#163）：其他键会被静默替换成默认文案，用户以为自己的描述生效了其实没有。校验前置会直接报错。

---

## 4. [MASK] 机制的完整链条（一张图串起来）

```
1. render_options(q)            →  ["billing: …", "technical: …", "account: …", "sales: …"]
                                   （或 score 的 "level 0: …"，或 noul 的 [false, true]）
        ↓
2. 每项前面挂 [MASK]，截到 48 token    →  opt_ids = [[MASK]+t1, [MASK]+t2, [MASK]+t3, [MASK]+t4]
        ↓
3. 预算会计（opt_budget<16 则均分砍）  →  保证 Σ选项 ≤ head_max_len
        ↓
4. 拼成 [CLS] type instructions [SEP] + 拼好的选项 + [SEP] state [SEP]
        ↓
5. 记录 markers = 每个 [MASK] 在最终序列里的位置   →  [12, 23, 33, 43]
        ↓
6. collate_items 打包成张量：input_ids(N,L) + marker_pos(N,K) + marker_mask(N,K)
        ↓
7. DecisionModel.forward：
     encoder → +type_emb → 2层TransformerEncoder → gather(marker_pos) → scorer → (N,K) logits
        ↓
8. 解码：logits/temperature → softmax → answer_confidence = max(p) → argmax 出 choice
   （详见 06-2 前向、06-3 温度）
```

**一句话概括 [MASK] 机制**：把"选哪个"变成"每个候选项在序列里各占一个可被读取的位置"，决策头对这些位置各打一个分。**没有生成、没有解析、没有 KV cache**——这是 Laya 毫秒级延迟的架构根基。

---

## 5. 与其他非自回归/分类方案的位置对照

| 方案 | 选项放哪 | 答案怎么出 | Laya 的差异 |
|---|---|---|---|
| **Laya** | head 区独立槽位（`[MASK]`） | gather marker 位置打分 | 选项受 `head_max_len` 预算约束 |
| 生成式 LLM | 全在 prompt | 逐 token 生成 + 解析 | Laya 无生成/无解析，但选项数受限 |
| 双塔（两段式） | state、选项各自分别编码 | cosine 相似度 | Laya 只编码一次 state，双塔在 Laya 场景省不了什么 |
| cross-encoder（成对） | state+option 拼一起 | 每对一次前向 | Laya 一次前向出全部 K 个分，cross-encoder 要 K 次 |

**Laya 的取舍是"选项数换效率"**：K 个选项一次前向（不管 K 多大），代价是每个选项的描述预算被 `head_max_len/K` 压缩。**K ≲ 20 时这个取舍纯赚；K 很大时必须用 shortlist 先降维。**

---

## 6. 实操清单：序列与 [MASK] 的 5 条落地准则

1. **`head_max_len` 是你唯一必须真正理解的超参。** 选项区分度 = `head_max_len / K`。高基数（K>20）先想清楚是"调大 head_max_len（同时调 max_len）"还是"shortlist 降维"还是"两级分类"，别硬塞；
2. **选项描述精炼优先于详尽。** 反正会被砍到 `head_max_len/K` 个 token，把区分性关键词（部门名、意图名）放最前面，别堆长定义；
3. **state 只 tokenize 一次**：`build_sequence` 的 `state_ids` 参数（`common.py:78`）就是干这个的——`predict_batch` 对同一 state 的多问题复用（`agent.py:569-576` 注释："re-serializing … was pure duplicated work"）。自己拼训练循环时也照做；
4. **会话列表左截断、文档右截断**（`truncate_left`）。会话 newest-last，右截断会静默丢最新一轮；
5. **选项顺序是评测维度也是增广手段**。`build_sequence` 支持 `option_order` 参数打乱顺序（`common.py:76, 87`）——位置不变时选项顺序应该不影响答案（对比性打分靠注意力而非位置），`research/eval/metamorphic.py` 把"选项顺序扰动"作为不变性测试；反过来，这也意味着**你可以在训练时增广选项顺序增强鲁棒性**。

---

## 7. 参考文献与源码锚点

| 结论 | 出处 |
|---|---|
| 序列格式定义（docstring） | `laya/common.py:80` |
| 选项前置 [MASK] + 48 token 截断 + 防注入 | `laya/common.py:91-101` |
| 预算会计（opt_budget<16 均分砍） | `laya/common.py:102-107` |
| marker 记录 + max_len 过滤 | `laya/common.py:109-112, 120` |
| state 截断方向 + `-room` 的 `-0` 坑 | `laya/common.py:114-119`（注释 117） |
| 题型渲染（choice/score/noul） | `laya/common.py:49-67` |
| noul criteria 只能 true/false（#156/#163） | `laya/agent.py:518-525` |
| state 复用 `state_ids` 参数 | `laya/common.py:78, 115-116`；`laya/agent.py:569-576` |
| 会话左截断 | `laya/agent.py:566-568` |
| 选项顺序扰动（不变性测试） | `research/eval/metamorphic.py`；`build_sequence` 的 `option_order`（`common.py:76, 87`） |
| head_max_len 塌陷实测表 | `examples/laya_internals_demo.py §1b` |
| banking77 0.425 是预算天花板（官方定性） | `laya-repo/BENCHMARKS.md:147-149` |
| 候选项移入 head 区的收益（0.44→0.51） | `laya-repo/docs/finetune_browser_agent.md` "What mattered most" 一节 |
| 决策头如何 gather marker 出分 | 见 `06-2-决策头与前向.md`；实现 `laya/common.py:139-175` |

> 序列造好了之后怎么过决策头 → `06-2-决策头与前向.md`。
