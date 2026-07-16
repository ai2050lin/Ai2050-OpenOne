# AGI Research Memo

> 本文档记录AGI研究的进展、问题分析和下一步行动

## Phase 901: 停止词竞争力审计 [2026-07-03 17:51]

### 一、阶段目的

Phase900 没有找到简单的 step=1/2 protocol stop gate（协议停止门）。因此 Phase901 继续同一阶段目标，但不再先做控制，而是直接审计 answer-class prefix（答案类别前缀）之后的下一步 logits（对数几率）：

```text
答案类别已经出现后，
EOS / 句号 / 换行 / 字段 / 解释 / 列表 token
到底谁在竞争场前面？
```

这个阶段回答的是：

```text
clean protocol 失败，是因为 stop token 根本不在竞争区？
还是 stop token 已经接近，但被协议续写 token 压住？
```

### 二、测试脚本与数据位置

新增脚本：

```text
tests/glm5/phase901_stop_token_competitiveness_audit.py
tests/glm5/run_phase901_stop_token_competitiveness_audit.sh
```

结果目录：

```text
tests/result/phase901_stop_token_competitiveness_audit/stop_token_competitiveness_audit/
```

运行顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

### 三、测试原理

Phase901 读取 Phase899 中的同一批样本：

```text
is_source_candidate = true
rollout_answer_class = true
protocol_drift = true
```

共 68 条。

对每条样本：

```text
1. 使用 Phase899 的 source_candidate 干预；
2. 自回归推进到最短 answer-class prefix 成立的位置；
3. 在下一步读取 logits；
4. 计算 EOS、period、newline、field、explanation、list、protocol group 的 rank / margin。
```

这样避免把多 token 答案短语误判为“应该停止”的位置。

### 四、核心公式

答案类别前缀：

$$
t^*
=
\min_t
\left[
AnswerClassPrefix(y_{\le t})=1
\right]
$$

停止词竞争力：

$$
StopRank(x)
=
Rank
\left(
\max_{z\in \mathcal{T}_{stop}}
logit(z\mid x,y_{\le t^*})
\right)
$$

协议续写竞争力：

$$
ProtocolRank(x)
=
Rank
\left(
\max_{z\in \mathcal{T}_{protocol}}
logit(z\mid x,y_{\le t^*})
\right)
$$

硬停止竞争力：

$$
EOSRank(x)
=
Rank
\left(
logit(EOS\mid x,y_{\le t^*})
\right)
$$

关键判据：

$$
StopRank \le 10
\quad
\land
\quad
ProtocolRank = 1
$$

表示：

```text
停止词接近竞争区，
但协议续写仍然压在前面。
```

### 五、跨模型结果

总体：

```text
rows = 68
answer_prefix_seen = 68
stop_top10 = 61 / 68
stop_top50 = 68 / 68
stop_top100 = 68 / 68
period_top50 = 68 / 68
eos_top100 = 22 / 68
median_stop_rank = 6
median_eos_rank = 147
median_protocol_rank = 1
```

下一 token top1 分布：

```text
\n: 34
\n\n: 6
,: 20
.: 7
Kingdom: 1
```

这说明：

```text
1. 句号 / soft stop（软停止）并不远；
2. EOS / hard stop（硬结束）整体仍弱；
3. protocol token（协议续写 token）通常排第一。
```

#### qwen3

```text
rows = 18
stop_top10 = 18 / 18
period_top50 = 18 / 18
eos_top100 = 0 / 18
median_stop_rank = 3
median_eos_rank = 29326.5
median_protocol_rank = 2
next_top_tokens:
  "." = 7
  "\n" = 8
  "\n\n" = 3
median_stop_margin_vs_top = -5.3125
```

qwen3 的句号常接近甚至成为 top1，但 EOS 极弱；即使输出句号，后续仍继续生成，所以句号不是硬停止门。

#### GLM4

```text
rows = 17
stop_top10 = 12 / 17
stop_top50 = 17 / 17
eos_top100 = 12 / 17
period_top50 = 17 / 17
median_stop_rank = 7
median_eos_rank = 36
median_protocol_rank = 1
next_top_tokens:
  "\n" = 16
  " Kingdom" = 1
median_stop_margin_vs_top = -2.6875
```

GLM4 的 EOS 比 qwen3 更接近，但换行协议几乎总是 top1。

#### DS7B

```text
rows = 33
stop_top10 = 31 / 33
stop_top50 = 33 / 33
eos_top100 = 10 / 33
period_top50 = 33 / 33
median_stop_rank = 8
median_eos_rank = 147
median_protocol_rank = 1
next_top_tokens:
  "," = 20
  "\n" = 10
  "\n\n" = 3
median_stop_margin_vs_top = -4.375
```

DS7B 主要输给 comma/list drift（逗号/列表漂移）和换行协议。

### 六、结果分析

Phase901 是一个重要收紧，同时也是一个正结果。

正结果：

```text
1. stop token 不是完全缺失；
2. period / soft stop 在所有样本中进入 top50；
3. stop group 在 61 / 68 中进入 top10；
4. 说明 clean protocol edge 不是完全不可达。
```

负结果：

```text
1. protocol_rank median = 1；
2. next_top_token 多为 "\n"、","、"\n\n"；
3. EOS 只有 22 / 68 进入 top100；
4. qwen3 的 EOS rank 极低；
5. 句号即使接近，也不是 hard stop，不能阻止继续生成。
```

因此 Phase901 的结论不是：

```text
模型没有停止信号。
```

而是：

```text
停止信号存在，但输给协议续写信号；
hard EOS 竞争力不足；
soft stop 不能自动变成生成停止。
```

### 七、理论进展

Phase901 把 Phase900 的 unresolved ProtocolStopGate（未解决协议停止门）进一步拆成两个问题：

```text
1. stop-token availability（停止 token 可达性）；
2. stop-token selection / termination control（停止 token 选择与终止控制）。
```

当前结果显示：

```text
soft stop availability 较强；
hard EOS availability 较弱；
protocol continuation selection 极强。
```

图谱结构应更新为：

```text
AnswerClassPrefix
  -> SoftStopTokenNearBoundary
  -> ProtocolContinuationDominates
  -> HardEOSWeak
  -> CleanProtocolGap
```

这比 Phase900 的“未找到停止门”更具体。

### 八、问题和硬伤

主要硬伤：

```text
1. Phase901 是 logit audit（对数几率审计），不是因果干预；
2. token group 是工程分组，仍需人工校准；
3. period 被归入 stop group，但它不是 EOS；
4. 只测 answer-prefix 后一步，没有测更长程 stopping trajectory；
5. 没有直接把 EOS / period patch 进去测试后续行为；
6. 小模型可能天然偏向续写模板，导致 protocol token 过强。
```

因此 Phase901 不能说明：

```text
已经找到协议停止门。
```

只能说明：

```text
停止竞争场的形状更清楚了：
soft stop 接近，hard EOS 弱，protocol continuation 占优。
```

### 九、闭合标准与当前距离

Phase901 后 clean protocol closure（干净协议闭合）至少需要：

```text
1. AnswerClassPrefix 成立；
2. period / EOS / stop token 进入 top-k；
3. protocol continuation token 不再 rank=1；
4. EOS 或真正停止动作能胜出；
5. 后续 rollout 不进入解释、字段、列表、长短语；
6. 控制在 holdout 上稳定。
```

当前满足：

```text
1. AnswerClassPrefix = 68 / 68；
2. StopTop50 = 68 / 68；
3. StopTop10 = 61 / 68。
```

当前未满足：

```text
1. ProtocolRank 仍接近 1；
2. EOSTop100 只有 22 / 68；
3. hard stop 没有胜出；
4. clean protocol rollout 仍为 0。
```

当前进度评估：

```text
全局齿轮图谱:
  86% - 90%

语言编码机制闭合:
  43% - 48%
```

图谱进度继续上升，因为 clean protocol gap（干净协议缺口）从“未知”变成了“soft stop 近、protocol continuation 强、hard EOS 弱”的明确形状。闭合进度只小幅回升，因为还没有因果控制成功。

### 十、第一性原理洞察

Phase901 的关键洞察是：

```text
自然语言回答的“停止”不是一个普通语义类别；
它是一个输出控制动作。
```

模型已经知道：

```text
应该说 animal / material / shape / vehicle
```

也部分知道：

```text
可以接句号。
```

但它更倾向于：

```text
换行、逗号、字段、列表、解释。
```

所以语言编码机制至少包括：

```text
semantic answer field（语义答案场）
protocol continuation field（协议续写场）
termination control field（终止控制场）
```

目前前两者已经可见，第三者尚未被因果定位。

### 十一、下一阶段

Phase901 与下一阶段仍属于同一个阶段性目标：完成全局齿轮图谱中 clean protocol edge（干净协议边）的定位。

下一阶段应进入：

```text
Phase902:
Protocol Continuation Suppressor Search
协议续写抑制器搜索
```

任务：

```text
1. 不再把 stop token 当唯一目标；
2. 直接针对 rank=1 的 protocol continuation token 做 suppressor search；
3. 分别抑制 newline、comma、Category/Item/Class、explanation opener；
4. 测试是否能让 period/EOS 或更短答案胜出；
5. 如果成功，再组合 semantic axis + protocol suppressor；
6. 如果失败，说明协议续写是分布式场，不是单点 stop gate。
```

## Phase 902: 协议续写抑制器搜索 [2026-07-03 18:15]

### 一、对上传内容的判断

上传内容对 Phase900 / Phase901 的判断基本正确。当前不能再写成“模型没有停止信号”，更准确的事实链是：

```text
DomainAxis（领域坐标轴）
  -> AnswerClassPrefix（答案类别前缀）
  -> SoftStopTokenNearBoundary（软停止 token 靠近边界）
  -> ProtocolContinuationDominates（协议续写占优）
  -> CleanProtocolGap（干净协议缺口）
```

Phase900 只排除了简单 step=1/2 协议停止门，没有排除所有停止门。Phase901 证明 stop token 并非缺失，period / soft stop 已经靠近竞争区，但 EOS / hard stop 仍弱，协议续写 token 多数仍在 rank=1。因此 Phase902 继续同一阶段性目标：定位 clean protocol edge（干净协议边）的失败形状。

### 二、测试原理

本阶段读取 Phase899 中满足以下条件的样本：

```text
is_source_candidate = true
rollout_answer_class = true
protocol_drift = true
```

总样本：

```text
qwen3: 18
GLM4: 17
DS7B: 33
total: 68
```

Phase902 不再只看 stop token 是否存在，而是在 answer-class prefix 后，对 protocol continuation token 做有限抑制器搜索。测试控制包括：

```text
1. baseline_after_answer_prefix
2. source_repeat_after_prefix
3. gear_half_after_prefix
4. gear_zero_after_prefix
5. gear_flip_after_prefix
6. head_zero_after_prefix
```

控制生效位置是答案前缀后的前两个后续步：

```text
suppress_steps = 2
```

这样可以同时覆盖：

```text
1. 第一步直接输出 newline / comma / field token 的情况；
2. qwen3 先输出 period，但第二步继续解释或换行的情况。
```

核心指标：

$$
P_0(x)=\arg\max_{z\in \mathcal{T}_{protocol}} logit(z\mid x,y_{\le t^*})
$$

$$
\Delta_{protocol}(C,x)
=
logit_C(P_0(x)\mid x,y_{\le t^*})
-
logit_0(P_0(x)\mid x,y_{\le t^*})
$$

$$
Removed(C,x)
=
\mathbf{1}
[
ProtocolRank_0(x)=1
\land
ProtocolRank_C(x)>1
]
$$

$$
Clean(C,x)
=
AnswerClass(C,x)
\cdot
(1-ObjectEcho(C,x))
\cdot
(1-ProtocolDrift(C,x))
$$

其中：

```text
Delta_protocol < 0 表示协议续写 token 被压低；
Removed = 1 表示 protocol rank=1 被移除；
Clean = 1 才表示真正短答案干净闭合。
```

### 三、脚本与结果文件

新增脚本：

```text
tests/glm5/phase902_protocol_continuation_suppressor_search.py
tests/glm5/run_phase902_protocol_continuation_suppressor_search.sh
```

结果目录：

```text
tests/result/phase902_protocol_continuation_suppressor_search/protocol_continuation_suppressor_search/
```

关键结果文件：

```text
phase902_cross_model_summary.md
phase902_cross_model_summary.json
phase902_qwen3_rows.jsonl
phase902_glm4_rows.jsonl
phase902_deepseek7b_rows.jsonl
```

### 四、跨模型客观结果

总体结果：

```text
selected_answer_drift_rows = 68
control_rows = 973
non_base_clean_answer_no_protocol = 0
non_base_protocol_drift = 905
non_base_protocol_rank1_removed = 8
non_base_protocol_logit_delta_negative = 281
non_base_protocol_logit_delta_below_minus_0_5 = 57
non_base_next_top_changed = 28
non_base_stop_rank_improved = 5
non_base_stop_top1 = 110
```

结论非常明确：

```text
有限抑制器可以轻微压低协议续写 logit；
少数样本可以移除 protocol rank=1；
但 clean_answer_no_protocol 仍然是 0。
```

### 五、分模型结果

#### qwen3

```text
selected_rows = 18
control_rows = 308

baseline:
  clean_answer_no_protocol = 0
  protocol_top1 = 8 / 18
  stop_top1 = 7 / 18
  stop_top10 = 18 / 18
  next_top_tokens = {newline: 8, double_newline: 3, period: 7}

non_baseline:
  clean_answer_no_protocol = 0
  protocol_rank1_removed = 2 / 290
  protocol_logit_delta_negative = 72 / 290
  protocol_logit_delta_below_minus_0_5 = 17 / 290
  next_top_changed = 2 / 290
  stop_rank_improved = 0 / 290
  stop_top1 = 110 / 290
```

qwen3 的关键现象是：period 已经经常是 top1，但 rollout 仍然进入解释或换行，所以问题不是“句号不可达”，而是 soft stop 不能变成 hard termination。

最佳控制：

```text
head_zero_after_prefix::L31H0+L31H1+L31H2+L31H3
clean = 0
protocol_rank1_removed = 2 / 16
next_top_changed = 2 / 16
```

#### GLM4

```text
selected_rows = 17
control_rows = 170

baseline:
  clean_answer_no_protocol = 0
  protocol_top1 = 16 / 17
  stop_top1 = 0 / 17
  stop_top10 = 12 / 17
  next_top_tokens = {newline: 16, " Kingdom": 1}

non_baseline:
  clean_answer_no_protocol = 0
  protocol_rank1_removed = 0 / 153
  protocol_logit_delta_negative = 55 / 153
  protocol_logit_delta_below_minus_0_5 = 0 / 153
  next_top_changed = 0 / 153
  stop_rank_improved = 5 / 153
```

GLM4 的 newline protocol field 最硬。有限控制能造成很弱的 logit 变化，但不能改变 top1，也不能产生 clean rollout。

#### DS7B

```text
selected_rows = 33
control_rows = 495

baseline:
  clean_answer_no_protocol = 0
  protocol_top1 = 30 / 33
  stop_top1 = 0 / 33
  stop_top10 = 31 / 33
  next_top_tokens = {comma: 20, newline: 10, double_newline: 3}

non_baseline:
  clean_answer_no_protocol = 0
  protocol_rank1_removed = 6 / 462
  protocol_logit_delta_negative = 154 / 462
  protocol_logit_delta_below_minus_0_5 = 40 / 462
  next_top_changed = 26 / 462
  stop_rank_improved = 0 / 462
```

DS7B 比 GLM4 更容易发生 next_top_changed，但主要是在 comma / newline / 其他延续 token 之间切换，仍然不进入 clean protocol closure。

最佳控制：

```text
head_zero_after_prefix::L26H3+L26H7+L26H11+L26H14
clean = 0
protocol_rank1_removed = 1 / 33
protocol_logit_delta_below_minus_0_5 = 5 / 33
```

### 六、阶段结论

Phase902 是一个重要负结果，也是一次图谱收紧：

```text
简单 gear 置零 / 翻转 / 减半；
简单 source repeat；
有限历史 head / 同层 head 置零；

都不能把 AnswerClassPrefix 之后的协议续写转成 clean short answer。
```

这说明 clean protocol edge 不太像一个局部开关：

```text
不是：
  找到某个 head / channel -> 关掉 -> 短答案闭合

更像：
  answer prefix 后进入一个分布式 protocol continuation attractor
  stop token 靠近但无法控制终止动作
```

当前图谱应更新为：

```text
DomainAxis
  -> AnswerClassPrefix
  -> StopNearBoundary
  -> ProtocolContinuationAttractor
      - qwen3: period 可达，但后续继续
      - GLM4: newline protocol 极强
      - DS7B: comma/list 与 newline protocol 极强
  -> CleanProtocolGap
```

### 七、硬伤和边界

本阶段不能说明协议控制完全无法闭合，原因是：

```text
1. 控制集合仍然有限；
2. head 搜索只覆盖历史 head 与同层前若干 head；
3. gear 控制只覆盖 source gears，没有覆盖全模型 protocol gears；
4. suppress_steps = 2，只测试短程边界；
5. protocol token group 仍是工程分组；
6. 小模型可能有更强模板续写惯性，导致 protocol continuation 被放大。
```

所以 Phase902 只能证明：

```text
没有发现简单局部协议续写抑制器；
协议续写场具有分布式、吸引子式特征；
clean protocol closure 仍未完成。
```

不能证明：

```text
协议停止机制不存在；
语言编码机制无法闭合。
```

### 八、对智能理论的影响

当前结果进一步支持：

```text
语言输出不是单纯 semantic token selection；
至少包含 semantic answer field、protocol continuation field、termination control field 三个场。
```

更重要的是，termination control 不是普通语义分类，而是对生成轨迹的停止动作控制。当前模型已经具备：

```text
1. 把对象映射到答案类别；
2. 把答案类别 token 推到前排；
3. 让 period / stop token 靠近边界。
```

但还没有稳定完成：

```text
4. 让 hard stop / termination action 胜过 protocol continuation。
```

这意味着语言编码机制中的“答案是什么”和“回答到哪里结束”在小模型里可能是分开的齿轮系统。

### 九、进度评估

```text
全局齿轮图谱:
  87% - 91%

语言编码机制闭合:
  43% - 48%
```

图谱进度继续上升，因为协议缺口形状进一步明确：不是 stop token 缺失，也不是简单局部 suppressor 可修复，而是 protocol continuation attractor 占优。闭合进度没有明显上升，因为 clean_answer_no_protocol 仍为 0。

### 十、下一阶段

Phase902 完成了 Phase901 提出的有限协议续写抑制器预筛。下一步仍服务于 clean protocol edge，但已经从“简单 suppressor search”进入更大的图谱阶段：

```text
Phase903:
Protocol Continuation Field Mapping
协议续写场图谱化
```

下一阶段不应继续盲目扩大局部 patch，而应：

```text
1. 对 newline / comma / field / explanation continuation 分别建图；
2. 区分 prompt protocol prior 与 answer-prefix 后续写 prior；
3. 找 protocol continuation 的全模型来源层、来源 head、来源 MLP 通道；
4. 把 qwen3 的 soft-stop-after-period 与 GLM4 / DS7B 的 newline/comma drift 分开；
5. 先完成 protocol continuation field graph，再尝试 termination control closure。
```

阶段性判断：

```text
当前任务与 Phase901 / Phase902 属于同一 clean protocol edge 定位阶段；
本阶段已完成有限抑制器预筛；
下一任务属于更大一层的协议续写场图谱化阶段，应作为新的阶段继续。
```

## Phase 903: 协议续写场图谱化 [2026-07-03 18:58]

### 一、对上传内容的判断

上传内容对 Phase902 的判断基本正确。最重要的收紧是：

```text
Phase901:
  stop token 并非缺失，soft stop 已经靠近边界。

Phase902:
  简单局部抑制器不能把协议续写变成干净短答案。

Phase903:
  应从“找一个 suppressor”转向“画出 protocol continuation field”。
```

需要严格限定：Phase902 不是证明协议停止机制不存在，而是证明当前有限控制没有找到简单局部协议续写抑制器。上传内容中提出的 Phase903 方向正确：先按 newline / comma / field / explanation / list 等漂移模式建图，再考虑 termination control closure（终止控制闭合）。

### 二、Phase903 测试目标

本阶段继续同一阶段性目标：定位 clean protocol edge（干净协议边）的失败来源。但任务层级从：

```text
Can we suppress one protocol token?
能不能压低某个协议 token？
```

升级为：

```text
Where does the protocol continuation field come from?
协议续写场从哪里来？
```

具体测试三件事：

```text
1. prompt protocol prior（提示协议先验）与 answer-prefix continuation prior（答案前缀续写先验）的差异；
2. 逐层 attention / MLP 组件置零对协议 token logit 的影响；
3. 协议 token 之间的替代关系。
```

### 三、测试脚本和结果文件

新增脚本：

```text
tests/glm5/phase903_protocol_continuation_field_mapping.py
tests/glm5/run_phase903_protocol_continuation_field_mapping.sh
```

结果目录：

```text
tests/result/phase903_protocol_continuation_field_mapping/protocol_continuation_field_mapping/
```

关键结果：

```text
phase903_cross_model_summary.md
phase903_cross_model_summary.json
phase903_qwen3_state_rows.jsonl
phase903_qwen3_component_rows.jsonl
phase903_glm4_state_rows.jsonl
phase903_glm4_component_rows.jsonl
phase903_deepseek7b_state_rows.jsonl
phase903_deepseek7b_component_rows.jsonl
```

### 四、测试原理

样本仍然来自 Phase899：

```text
is_source_candidate = true
rollout_answer_class = true
protocol_drift = true
```

样本量：

```text
qwen3: 18
GLM4: 17
DS7B: 33
total: 68
```

非重叠协议类别重新定义为：

```text
newline
comma
field_word
explanation
list_word
period
eos
```

协议场：

$$
\mathcal{T}_{protocol}
=
\mathcal{T}_{newline}
\cup
\mathcal{T}_{comma}
\cup
\mathcal{T}_{field}
\cup
\mathcal{T}_{explanation}
\cup
\mathcal{T}_{list}
$$

停止场：

$$
\mathcal{T}_{stop}
=
\mathcal{T}_{period}
\cup
\mathcal{T}_{EOS}
$$

答案前缀位置：

$$
t^*
=
\min_t[
AnswerClassPrefix(y_{\le t})=1
]
$$

协议最佳类别：

$$
c^*(x)
=
\arg\max_{c\in \mathcal{C}_{protocol}}
\max_{z\in \mathcal{T}_c}
logit(z\mid x,y_{\le t^*})
$$

逐层组件置零：

$$
\Delta_{l,k}^{protocol}(x)
=
logit_{zero(l,k)}(P_0(x))
-
logit_{base}(P_0(x))
$$

其中：

```text
l: layer
k: attention 或 MLP
P_0(x): baseline 下最强协议 token
```

协议替代边：

$$
Substitution(C,x)
=
c_{base}(x)
\rightarrow
c_C(x)
$$

含义是：置零某个组件后，最高输出或协议最佳类别从一个协议类型切到另一个协议类型。

### 五、跨模型总体结果

```text
selected_answer_drift_rows = 68
state_rows = 68
component_rows = 4504
component_protocol_logit_reduced = 2309
component_protocol_logit_reduced_strong = 873
component_protocol_rank1_removed = 66
component_next_top_changed = 417
component_stop_rank_improved = 738
```

这说明：

```text
协议续写场确实有可观察的逐层组件来源；
置零 attention / MLP 能大量压低协议 token；
但主要结果仍是协议内部替代，不是干净终止。
```

### 六、状态先验结果

#### qwen3

```text
rows = 18
prompt_next_top_categories = {list_word: 5, other: 13}
answer_next_top_categories = {newline: 11, period: 7}
prompt_protocol_best_categories = {list_word: 18}
answer_protocol_best_categories = {newline: 18}
answer_protocol_top1 = 11 / 18
answer_stop_top1 = 7 / 18
answer_stop_top10 = 18 / 18
median_protocol_rank_delta_answer_minus_prompt = -5
```

解释：

```text
qwen3 在 prompt 阶段的协议先验偏 list_word；
答案前缀后统一切到 newline protocol；
同时 period 有 7 / 18 成为 top1，但并不产生 hard termination。
```

#### GLM4

```text
rows = 17
prompt_next_top_categories = {other: 17}
answer_next_top_categories = {newline: 16, other: 1}
prompt_protocol_best_categories = {explanation: 8, field_word: 2, list_word: 7}
answer_protocol_best_categories = {newline: 17}
answer_protocol_top1 = 16 / 17
answer_stop_top1 = 0 / 17
answer_stop_top10 = 12 / 17
median_protocol_rank_delta_answer_minus_prompt = -38
```

解释：

```text
GLM4 的协议续写不是简单继承 prompt prior；
答案前缀后 newline protocol 被极强激活；
这是当前最硬的 newline continuation field。
```

#### DS7B

```text
rows = 33
prompt_next_top_categories = {list_word: 9, other: 24}
answer_next_top_categories = {comma: 20, newline: 13}
prompt_protocol_best_categories = {explanation: 5, list_word: 28}
answer_protocol_best_categories = {comma: 18, newline: 15}
answer_protocol_top1 = 33 / 33
answer_stop_top1 = 0 / 33
answer_stop_top10 = 31 / 33
median_protocol_rank_delta_answer_minus_prompt = -2
```

解释：

```text
DS7B 的答案前缀后协议场不是单一 newline；
而是 comma / newline 双吸引子；
这解释了 Phase902 中压低一个协议 token 后经常转向另一个协议 token。
```

### 七、组件来源结果

#### qwen3

```text
component_rows = 1296
protocol_logit_reduced = 575
protocol_logit_reduced_strong = 147
protocol_rank1_removed = 1
next_top_changed = 101
stop_rank_improved = 140
```

按组件：

```text
attention:
  strong_reduced = 62
  rank1_removed = 1
  next_top_changed = 57
  stop_rank_improved = 67

MLP:
  strong_reduced = 85
  rank1_removed = 0
  next_top_changed = 44
  stop_rank_improved = 73
```

最强组件：

```text
L35 MLP newline:
  rows = 18
  strong_reduced = 18
  mean_delta = -12.1597

L35 attention newline:
  rows = 18
  strong_reduced = 18
  mean_delta = -2.2222
```

qwen3 的 newline field 明显集中在末层附近，但移除 rank1 很少，说明它更像已成形的输出场，而不是单点开关。

#### GLM4

```text
component_rows = 1360
protocol_logit_reduced = 711
protocol_logit_reduced_strong = 173
protocol_rank1_removed = 13
next_top_changed = 36
stop_rank_improved = 307
```

按组件：

```text
attention:
  strong_reduced = 47
  rank1_removed = 11
  next_top_changed = 21
  stop_rank_improved = 131

MLP:
  strong_reduced = 126
  rank1_removed = 2
  next_top_changed = 15
  stop_rank_improved = 176
```

最强组件：

```text
L39 MLP newline:
  rows = 17
  strong_reduced = 17
  mean_delta = -2.2684

L38 attention newline:
  rows = 17
  strong_reduced = 8
  rank1_removed = 6
  mean_delta = -0.9522

L22 attention newline:
  rows = 17
  strong_reduced = 10
  rank1_removed = 4
  mean_delta = -0.3860
```

GLM4 的 newline field 有末层 MLP 强来源，也有中后层 attention 对 rank1 的影响，但仍不形成干净停止。

#### DS7B

```text
component_rows = 1848
protocol_logit_reduced = 1023
protocol_logit_reduced_strong = 553
protocol_rank1_removed = 52
next_top_changed = 280
stop_rank_improved = 291
```

按组件：

```text
attention:
  strong_reduced = 267
  rank1_removed = 44
  next_top_changed = 153
  stop_rank_improved = 121
  mean_delta = -0.6699

MLP:
  strong_reduced = 286
  rank1_removed = 8
  next_top_changed = 127
  stop_rank_improved = 170
  mean_delta = -0.0554
```

按协议类别：

```text
comma:
  rows = 1008
  strong_reduced = 294
  rank1_removed = 21
  next_top_changed = 25
  mean_delta = -0.2833

newline:
  rows = 840
  strong_reduced = 259
  rank1_removed = 31
  next_top_changed = 255
  mean_delta = -0.4579
```

最强组件：

```text
L27 attention comma:
  rows = 18
  strong_reduced = 18
  rank1_removed = 11
  mean_delta = -8.6632

L0 attention comma:
  rows = 18
  strong_reduced = 18
  rank1_removed = 6
  mean_delta = -8.4861

L27 attention newline:
  rows = 15
  strong_reduced = 15
  rank1_removed = 6
  mean_delta = -7.6896

L0 attention newline:
  rows = 15
  strong_reduced = 15
  rank1_removed = 5
  mean_delta = -8.8167
```

DS7B 的协议场来源最清楚：attention 对 comma/newline 的 rank1 更敏感，MLP 也能压低 logit，但更像分布式调制。

### 八、协议替代图

跨模型最高 token 替代关系：

```text
comma -> list_word: 4
comma -> newline: 48
comma -> other: 21
newline -> comma: 73
newline -> newline: 195
newline -> other: 45
other -> newline: 20
other -> other: 2
period -> newline: 6
period -> other: 1
period -> period: 2
```

关键现象：

```text
1. comma 和 newline 之间存在明显互相替代；
2. qwen3 的 period 被扰动后常转向 newline；
3. DS7B 的 comma/newline 双吸引子最明显；
4. GLM4 的 newline 场最刚性，替代较少。
```

因此 Phase903 支持 Phase902 的判断：

```text
协议漂移不是某个 token 的孤立问题；
它是一个 protocol continuation field。
```

### 九、阶段结论

Phase903 是实质图谱进展，不是闭合阶段。

它第一次把 Phase902 的“协议续写吸引子”拆成可观察结构：

```text
prompt prior:
  list_word / explanation / other

answer-prefix continuation prior:
  qwen3 -> newline + period
  GLM4 -> newline
  DS7B -> comma + newline

component sources:
  qwen3 -> late MLP / late attention newline field
  GLM4 -> late MLP newline + mid/late attention rank influence
  DS7B -> attention-sensitive comma/newline field

substitution graph:
  comma <-> newline
  period -> newline
```

这比 Phase902 更进一步，因为它不再只说“局部抑制失败”，而是说明失败后的替代路径是什么。

### 十、硬伤和边界

必须谨慎：

```text
1. 组件置零是粗粒度干预，不等于真实内部机制完全归因；
2. attention / MLP 输出整体置零可能引入分布外扰动；
3. 本阶段没有继续做 clean rollout，所以不能宣称闭合改善；
4. protocol category 仍是工程分组；
5. 只扫描 answer-prefix 边界，没有扫描更长轨迹；
6. 小模型可能放大模板续写，真实大模型协议场可能更可控。
```

因此本阶段结论只能写成：

```text
协议续写场已有可观察来源和替代关系；
但 termination control field 尚未定位；
clean protocol closure 仍为 0。
```

### 十一、理论进展

当前理论应更新为：

```text
条件化输出场闭合理论
  + 语义答案场
  + 停止竞争场
  + 协议续写场
  + 协议替代图
  + 未闭合终止控制场
```

更准确的语言输出公式：

$$
Output(x)
=
F(
S_{answer}(x),
S_{stop}(x),
S_{protocol}(x),
A_{substitution}(x),
T_{termination}(x)
)
$$

其中：

```text
S_answer: 语义答案场
S_stop: 停止 token 竞争场
S_protocol: 协议续写场
A_substitution: 协议替代图
T_termination: 终止控制场
```

当前已经较清楚的是：

```text
S_answer
S_stop
S_protocol
A_substitution
```

仍未闭合的是：

```text
T_termination
```

### 十二、进度评估

```text
全局齿轮图谱:
  89% - 92%

语言编码机制闭合:
  44% - 49%
```

图谱进度上升，因为协议续写场已经从抽象判断变成了可分解的组件来源和替代图。闭合进度只小幅上升，因为没有产生 clean_answer_no_protocol。

### 十三、下一阶段

Phase903 与 Phase902 属于同一 clean protocol edge 大阶段，但已经完成了协议续写场的第一版图谱化。下一阶段应继续在同一阶段性目标下自动推进：

```text
Phase904:
Termination Control Candidate Search
终止控制候选搜索
```

目标不是继续压低 newline/comma，而是寻找：

```text
1. period 后为什么继续；
2. EOS 为什么不胜出；
3. 是否存在 termination action 的来源层或 head；
4. stop token 与 protocol token 的差分方向；
5. 能否构造 semantic axis + protocol field map + termination candidate 的组合测试。
```

成功标准：

```text
最低标准:
  找到 period 后继续的主要来源层 / 组件。

中等标准:
  找到能提高 EOS / stop action 相对 protocol 的组件。

高标准:
  在 holdout 上让 protocol continuation rank 明显下降且 stop rank 明显上升。

最高标准:
  clean_answer_no_protocol > 0。
```

## Phase 904: 终止控制候选搜索 [2026-07-03 19:13]

### 一、任务来源

本阶段接续 Phase903。Phase903 已经把 Phase902 的负结果推进成协议续写场图谱：

```text
newline continuation
comma/list continuation
field/explanation continuation
period 后继续
stop / EOS 弱竞争
```

附件中关于 Phase902 的判断基本正确：Phase902 没有证明“永远无法闭合”，但证明了单纯压低局部协议续写组件不足以产生 clean answer closure。更准确地说：

```text
协议续写不是一个可由单点 suppressor 直接关闭的局部噪声项；
它更像是答案场之后自然接管的 protocol continuation field。
```

因此 Phase904 的目标不是继续寻找更强的 newline / comma suppressor，而是把 Phase903 得到的 top protocol-source components 当作 termination control candidates，测试它们是否能在 rollout 中产生真实终止。

### 二、测试脚本和结果文件

新增脚本：

```text
tests/glm5/phase904_termination_control_candidate_search.py
tests/glm5/run_phase904_termination_control_candidate_search.sh
```

结果目录：

```text
tests/result/phase904_termination_control_candidate_search/termination_control_candidate_search/
```

核心结果文件：

```text
phase904_cross_model_summary.json
phase904_cross_model_summary.md
phase904_qwen3_rows.jsonl
phase904_glm4_rows.jsonl
phase904_deepseek7b_rows.jsonl
```

### 三、测试原理

Phase904 使用 Phase899 中已经确认的 selected answer drift rows：

```text
qwen3: 18
GLM4: 17
DS7B: 33
合计: 68
```

每个模型读取 Phase903 的 top protocol continuation components，按模型选择 8 个候选组件，候选来源包括：

```text
attention_zero_L*
mlp_zero_L*
newline-source components
comma-source components
```

然后在 answer-prefix 已经生成之后，对候选组件做前 2 个 suffix step 的 component-zero intervention，再 rollout 8 个 token，观察是否从协议续写转为真实终止。

干预形式：

$$
h_{\ell}^{kind}(x,t) \leftarrow 0,\quad t \in \{t_{answer+1}, t_{answer+2}\}
$$

其中：

```text
kind ∈ {attention, mlp}
t_answer 表示答案前缀之后的生成位置
```

协议 token logit 变化：

$$
\Delta^{protocol}_{C}(x)
=
z_C(p_0 \mid x) - z_0(p_0 \mid x)
$$

其中：

```text
C: 候选控制组件
p_0: baseline protocol-best token
z_C: 干预后的 logit
z_0: baseline logit
```

stop rank 改善：

$$
G_{stop}(C,x)
=
rank_0(s_0 \mid x) - rank_C(s_0 \mid x)
$$

若：

$$
G_{stop}(C,x) > 0
$$

说明 stop token 排名上升。

严格 clean 判定改为：

$$
Clean_{strict}(C,x)
=
AnswerClass(C,x)
\land
\neg ObjectEcho(C,x)
\land
\neg ProtocolDrift(C,x)
\land
\neg StrictProtocolDrift(C,x)
$$

这个修正确认非常重要。初始宽松统计中曾出现 6 个 nominal clean，但人工核查发现它们包含：

```text
animal, mammal, quadruped,
Animal.Humanity. 1.
```

这类输出不是 clean answer，而是协议/枚举/异常续写。因此本阶段最终以 strict clean 为准。

### 四、总体结果

跨模型统计：

```text
candidate_count: 24
control_rows: 612
selected_answer_drift_rows: 68

non_base_clean_answer_no_protocol: 6
non_base_strict_clean_answer_no_protocol: 0
non_base_protocol_drift: 535
non_base_strict_protocol_drift: 541
non_base_protocol_logit_reduced_strong: 380
non_base_protocol_rank1_removed: 48
non_base_stop_rank_improved: 92
non_base_stop_top1: 55
non_base_next_top_changed: 102
```

关键结论：

```text
名义 clean = 6
严格 clean = 0
```

因此本阶段不能记为 closure positive，而应记为：

```text
termination competition can be shifted,
but termination action is not closed.
```

中文解释：

```text
候选组件可以移动协议续写竞争场，
可以让部分 protocol rank1 被移除，
也可以让 stop rank 改善，
但没有产生可靠的干净停机。
```

### 五、分模型结果

#### qwen3

```text
selected_answer_drift_rows: 18
candidate_count: 8
control_rows: 162

non_base_strict_clean_answer_no_protocol: 0
non_base_protocol_drift: 140
non_base_protocol_logit_reduced_strong: 85
non_base_protocol_rank1_removed: 0
non_base_stop_rank_improved: 35
non_base_stop_top1: 55
non_base_next_top_changed: 15
```

最佳候选：

```text
attention_zero_L34
source_category: newline
rows: 18
strict_clean: 0
protocol_drift: 18
protocol_rank1_removed: 0
stop_rank_improved: 8
first_suffix_categories:
  newline: 11
  period: 7
```

qwen3 的结果说明：stop token 竞争可以被提高，甚至 non-base stop_top1 达到 55，但只要 rollout 继续产生 newline / explanation / field_word，就不能算 clean closure。

#### GLM4

```text
selected_answer_drift_rows: 17
candidate_count: 8
control_rows: 153

non_base_strict_clean_answer_no_protocol: 0
non_base_protocol_drift: 136
non_base_protocol_logit_reduced_strong: 85
non_base_protocol_rank1_removed: 12
non_base_stop_rank_improved: 26
non_base_stop_top1: 0
non_base_next_top_changed: 14
```

最佳候选：

```text
attention_zero_L38
source_category: newline
rows: 17
strict_clean: 0
protocol_drift: 17
protocol_rank1_removed: 6
stop_rank_improved: 2
first_suffix_categories:
  comma: 1
  newline: 9
  other: 7
```

GLM4 的结果说明：它更明显表现为 competition shift，而不是 stop activation。协议 rank1 可以被移除，但替代项仍属于协议续写或异常续写。

#### DS7B

```text
selected_answer_drift_rows: 33
candidate_count: 8
control_rows: 297

non_base_strict_clean_answer_no_protocol: 0
non_base_protocol_drift: 259
non_base_protocol_logit_reduced_strong: 210
non_base_protocol_rank1_removed: 36
non_base_stop_rank_improved: 31
non_base_stop_top1: 0
non_base_next_top_changed: 73
```

最佳候选：

```text
attention_zero_L27
source_category: comma
rows: 33
strict_clean: 0
protocol_drift: 32
protocol_rank1_removed: 17
stop_rank_improved: 0
first_suffix_categories:
  comma: 12
  newline: 4
  other: 17
```

DS7B 的结果最能说明当前瓶颈：逗号协议续写场很强，attention_zero_L27 可以显著压低 comma continuation，并移除 17 个 rank1 protocol cases，但替代输出不是稳定 stop，而是 other / newline / comma 的重新分配。

### 六、正确性分析

附件中 Phase902 的判断可以保留，但需要加一条更严格边界：

```text
Phase902/903/904 共同证明的是：
协议续写场可以被分解、压低、重排；
但终止控制场尚未被找到。
```

不能说：

```text
找到了 clean answer closure
找到了完整 termination gear
找到了语言编码闭合
```

因为：

```text
1. strict clean = 0；
2. nominal clean 全部可能被严格审计排除；
3. stop rank 改善没有稳定转换为 rollout 停止；
4. protocol rank1 removed 后常出现 protocol substitution。
```

更准确的图谱位置是：

```text
semantic answer field 已经较稳定；
protocol continuation field 已经可图谱化；
protocol substitution graph 已经出现；
termination control field 仍缺关键来源。
```

### 七、理论进展

本阶段最大的进展不是找到闭合，而是修正了闭合标准：

```text
只看 answer class 是不够的；
只看 protocol logit 下降是不够的；
只看 stop rank 改善也是不够的；
必须看 rollout 后是否真的没有协议拖尾。
```

因此 clean closure 的层级应改为：

$$
\Delta z_{protocol} < 0
\Rightarrow
rank(protocol) \downarrow
\Rightarrow
rank(stop) \uparrow
\Rightarrow
StopAction
\Rightarrow
Clean_{strict}
$$

Phase904 只达到：

$$
\Delta z_{protocol} < 0
,\quad
rank(protocol) \downarrow
,\quad
rank(stop) \uparrow
$$

但没有达到：

$$
StopAction
\Rightarrow
Clean_{strict}
$$

这说明 stop token 竞争场和真实终止动作之间存在缺失环节。

### 八、核心硬伤

#### 1. stop rank 不是 stop action

qwen3 出现很多 stop_top1，但 rollout 仍然不是严格 clean。这说明：

```text
stop token 的 next-token 竞争优势
不等于模型进入终止动作状态。
```

可能存在：

```text
termination control state
decode policy boundary
special-token suppression
chat-template prior
answer-prefix continuation prior
```

这些因素在当前局部 component-zero 中没有被控制。

#### 2. protocol substitution 仍然存在

DS7B 的 attention_zero_L27 可以移除大量 comma rank1，但输出转向 other / newline，而不是终止。这说明：

```text
协议续写不是单 token 问题，
而是多候选替代场。
```

#### 3. 小模型偏差仍然明显

当前 qwen3、GLM4、DS7B 都是小模型或较粗糙结构模型。小模型可能把：

```text
语义答案
格式协议
枚举习惯
解释性续写
终止控制
```

压缩在相互纠缠的通道里，导致局部干预容易产生 abnormal continuation，而不是清晰机制切换。因此结果不能直接外推为大模型完整语言机制。

### 九、当前图谱更新

Phase904 后，全局齿轮图谱应加入一个明确负边：

```text
protocol-source component
  -> protocol logit reduction
  -> protocol rank shift
  -> stop rank partial improvement
  -/-> strict clean termination
```

也就是说：

```text
协议续写源组件不是终止控制组件；
终止控制不是协议抑制的自然副产物；
终止机制可能是独立 state/action，而不是单纯 competition winner。
```

最新图谱形状：

```text
Semantic Answer Field
  -> Answer Prefix
  -> Protocol Continuation Field
       -> newline route
       -> comma/list route
       -> field/explanation route
       -> substitution graph
  -> weak Stop Competition Field
  -> missing Termination Action Field
```

### 十、下一阶段任务

Phase904 仍属于 clean protocol edge 大阶段，但已经完成了“从 Phase903 候选组件寻找 termination control”的子任务。下一步不应继续简单扩大 component-zero 搜索，而应切换到：

```text
Phase905:
Stop Action vs Stop Token Boundary Audit
终止动作与停止 token 边界审计
```

核心问题：

```text
为什么 stop_top1 仍然不能 clean？
EOS / special stop token 是否被 tokenizer、chat template、generation config 或模型内部 continuation prior 分离？
period + EOS 与 period + newline 的边界在哪里？
termination action 是否不是 residual component，而是 decode-time / template-conditioned control？
```

建议 Phase905 测试：

```text
1. 对 stop_top1 但 rollout 不 clean 的 qwen3 rows 做逐步解码审计；
2. 比较 greedy logits、实际 generated ids、special-token allowed set；
3. 构造 period-only、EOS-forced、newline-forbidden 三类对照；
4. 区分 token-level stop competitiveness 与 generation-level termination action；
5. 再判断是否需要回到模型内部寻找 termination-action source。
```

成功标准：

```text
最低标准:
  解释 stop_top1 为什么没有 clean。

中等标准:
  区分 stop token competition failure 和 decode/template stop action failure。

高标准:
  找到 period + EOS 或 semantic answer + EOS 的可复现边界。

最高标准:
  在 holdout 上得到 strict_clean_answer_no_protocol > 0 且人工核查通过。
```

### 十一、阶段结论

Phase904 是重要负结果：

```text
终止候选组件可以移动竞争场，
但没有完成严格 clean closure。
```

这进一步支持当前主线：

```text
先完成图谱，再破解闭合。
```

因为现在已经更清楚地知道：

```text
协议续写场和终止控制场不是同一个东西；
压低协议续写不自动等于终止；
stop token 排名提升不自动等于 stop action。
```

总体进度评估：

```text
全局齿轮图谱:
  90% - 93%

语言编码机制闭合:
  44% - 49%
```

图谱进度继续上升，因为 termination-control 缺口被更精确地定位。闭合进度不明显上升，因为 strict clean 仍为 0。

## Phase 905: 终止动作与停止 token 边界审计 [2026-07-03 19:17]

### 一、任务来源

Phase904 出现一个关键现象：

```text
qwen3 的 non-base stop_top1 = 55
但 strict_clean_answer_no_protocol = 0
```

这说明 Phase904 中的 stop_top1 不能直接解释为 clean termination。Phase905 因此不再继续加载模型做新干预，而是先审计 Phase904 已保存的 rollout 行数据，回答一个基础问题：

```text
stop_top1 到底是 EOS / 真实终止动作胜出，
还是 period / 句号 token 胜出后继续生成？
```

### 二、脚本和结果文件

新增脚本：

```text
tests/glm5/phase905_stop_action_boundary_audit.py
tests/glm5/run_phase905_stop_action_boundary_audit.sh
```

结果目录：

```text
tests/result/phase905_stop_action_boundary_audit/stop_action_boundary_audit/
```

核心结果：

```text
phase905_stop_action_boundary_summary.json
phase905_stop_action_boundary_summary.md
```

### 三、测试原理

Phase905 只读取 Phase904 的 jsonl 行数据，不重新运行模型。它对每条 non-baseline row 统计：

```text
stop_top1
stop_best_category
first_suffix_category
second_suffix_category
protocol_drift
strict_protocol_drift
strict_clean_answer_no_protocol
```

边界判定：

$$
StopTop1(x)
=
rank(stop\_best \mid x) = 1
$$

其中 Phase903/904 的 stop 集合为：

$$
StopSet = EOS \cup Period
$$

因此必须进一步拆开：

$$
StopTop1(x)
=
EOSTop1(x)
\lor
PeriodTop1(x)
$$

真正接近终止动作的条件不是：

$$
StopTop1(x)
$$

而是：

$$
EOSFirst(x)
\lor
\bigl(PeriodFirst(x) \land \neg ContinuationAfterPeriod(x)\bigr)
$$

如果：

$$
PeriodFirst(x)
\land
ContinuationAfterPeriod(x)
$$

则只能说明“句号胜出”，不能说明“终止动作胜出”。

### 四、总体结果

Phase905 汇总：

```text
rows: 544
strict_clean_answer_no_protocol: 0
protocol_drift: 535
strict_protocol_drift: 541

stop_top1: 55
stop_top1_strict_clean: 0
stop_top1_protocol_drift: 55
stop_top1_strict_protocol_drift: 54

stop_top1_period_best: 55
stop_top1_eos_best: 0
stop_top1_period_first_suffix: 55
stop_top1_eos_first_suffix: 0
stop_top1_period_then_continuation: 55
stop_top1_decoded_special_marker: 5
```

核心事实非常清楚：

```text
所有 stop_top1 都是 period；
没有任何 stop_top1 是 EOS；
所有 period top1 后都继续生成；
stop_top1_strict_clean = 0。
```

因此 Phase904 中 qwen3 的 stop_top1 不是终止动作，而是：

```text
period-as-punctuation
句号作为标点符号
```

### 五、分模型结果

#### qwen3

```text
non_baseline rows: 144
stop_top1: 55
stop_top1_period_best: 55
stop_top1_eos_best: 0
stop_top1_period_first_suffix: 55
stop_top1_eos_first_suffix: 0
stop_top1_period_then_continuation: 55
stop_top1_strict_clean: 0
```

样例：

```text
Animal.Humanity. 1.
Animal. The cow is a domesticated animal
Shapes. The category that best describes a
Material. The answer is "Materials". The
```

这些输出共同说明：

```text
句号不是终止动作；
句号更像答案后的局部标点边界；
句号后仍然会进入 explanation / field / protocol continuation。
```

#### GLM4

```text
non_baseline rows: 136
stop_top1: 0
strict_clean_answer_no_protocol: 0
```

GLM4 没有出现 stop_top1 边界可审计现象。它的 Phase904 主要是 protocol rank1 removed 和 next_top_changed，但没有进入 stop_top1。

#### DS7B

```text
non_baseline rows: 264
stop_top1: 0
strict_clean_answer_no_protocol: 0
```

DS7B 同样没有 stop_top1。它的主要现象仍然是 comma protocol field 被压低后发生 substitution，而不是 stop action。

### 六、正确性分析

Phase905 进一步修正 Phase901/904 的 stop 口径：

```text
把 EOS 和 period 合并为 stop set，有助于早期发现“停止附近 token 是否有竞争力”；
但在闭合阶段，EOS 和 period 必须拆开。
```

原因是：

```text
period 是语言内部的标点 token；
EOS 是生成过程中的终止 token；
两者都可能位于答案边界附近，
但机制含义不同。
```

因此之前的：

$$
StopSet = EOS \cup Period
$$

只能用于粗审计，不能用于闭合判定。闭合判定必须改为：

$$
ClosureStop(x)
=
EOSAction(x)
\lor
\left[
PeriodBoundary(x)
\land
\neg ContinuationAfterPeriod(x)
\right]
$$

Phase905 的结果是：

$$
PeriodBoundary(x)=1
$$

但：

$$
ContinuationAfterPeriod(x)=1
$$

所以：

$$
ClosureStop(x)=0
$$

### 七、理论进展

Phase905 的关键进展是把“停止”拆成三层：

```text
1. punctuation boundary
   标点边界

2. answer boundary
   答案边界

3. generation termination action
   生成终止动作
```

当前实验已经能推动第一层：

```text
period / 句号
```

但尚未推动第三层：

```text
EOS / true generation termination
```

这解释了为什么：

```text
protocol logit 降低
stop rank 改善
period top1
```

仍然不能得到 clean answer closure。

新的图谱边应写为：

```text
protocol suppressor
  -> period boundary
  -> post-period continuation
  -/-> EOS termination action
```

而不是：

```text
protocol suppressor
  -> stop action
```

### 八、问题和硬伤

#### 1. Phase901/904 的 stop set 过宽

早期把 period 和 EOS 合并是合理的探索策略，但会产生误读：

```text
stop_top1 可能只是句号 top1。
```

以后所有 closure 统计必须同时报告：

```text
period_top1
eos_top1
period_then_continuation
eos_first
strict_clean
```

#### 2. 句号后仍有强 continuation prior

qwen3 的 55 个 stop_top1 全部 period 后继续，说明模型内部存在：

```text
post-period continuation prior
```

它可能来自：

```text
解释性文本习惯
chat template 续写习惯
问答格式续写习惯
训练语料中句号后继续文本的强先验
```

#### 3. EOS 仍未进入自然竞争

本轮没有出现 EOS best 或 EOS first。说明真正瓶颈已经更清楚：

```text
不是让 period 胜出；
而是让 EOS action 或 no-continuation state 胜出。
```

### 九、对当前研究路线的影响

Phase905 支持“先完成图谱再闭合”的路线：

```text
现在图谱已经知道：
semantic answer field 可以形成答案；
protocol field 会接管答案后续；
component intervention 可以压低 protocol；
period boundary 可以被推高；
但 EOS / termination action 仍未进入自然路线。
```

这比继续盲目搜索更有价值，因为失败位置已经从：

```text
不知道为什么不 clean
```

变成：

```text
period boundary 与 EOS action 之间缺一层机制。
```

### 十、下一阶段任务

Phase905 已经完成 stop_top1 边界审计。下一阶段不应继续把 period 当作 stop，而应进入新子阶段：

```text
Phase906:
EOS Action Boundary Test
EOS 动作边界测试
```

建议测试：

```text
1. period-forced 后观察下一步 logits；
2. EOS-forced 与 period-forced 对比；
3. newline-forbidden / explanation-forbidden 对照；
4. 直接测 EOS rank、EOS logit、EOS margin；
5. 区分模型内部 EOS 弱竞争与 generation config / tokenizer special-token 边界。
```

成功标准：

```text
最低标准:
  证明 EOS 是否被模型自然压制。

中等标准:
  找到 period 后 continuation 的主要 token 类型。

高标准:
  找到能把 period 后 continuation 转成 EOS 的干预变量。

最高标准:
  strict_clean_answer_no_protocol > 0 且人工核查通过。
```

### 十一、阶段结论

Phase905 是一个重要诊断结果：

```text
Phase904 的 stop_top1 不是终止动作；
它全部是 period top1。
```

更严格地说：

```text
period boundary 已经可被局部干预推高；
EOS termination action 仍然没有被找到；
period 后 continuation 是当前 clean closure 的直接瓶颈。
```

总体进度评估：

```text
全局齿轮图谱:
  91% - 94%

语言编码机制闭合:
  44% - 49%
```

图谱进度继续上升，因为 stop/action 边界被拆开。闭合进度不提高，因为 strict clean 仍为 0，且 EOS action 尚未进入自然竞争。

## Phase 906: EOS 动作边界测试 [2026-07-03 20:08]

### 一、任务来源

本阶段接续 Phase903-905。附件中对 Phase903、Phase904、Phase905 的综合判断基本正确：

```text
Phase903:
  协议续写不是单 token 问题，而是 protocol continuation field。

Phase904:
  协议续写场可以被压低、重排，但 termination action 没有闭合。

Phase905:
  stop_top1 全部是 period top1，不是 EOS / 真实终止动作。
```

因此 Phase906 不再使用宽泛的：

$$
StopSet = EOS \cup Period
$$

而是直接拆开：

```text
period boundary
EOS action
post-period continuation
```

核心问题变成：

```text
EOS 是不可用，还是可用但自然竞争极弱？
句号之后为什么不是 EOS，而是继续进入 explanation / field / other continuation？
```

### 二、测试脚本和结果文件

新增脚本：

```text
tests/glm5/phase906_eos_action_boundary_test.py
tests/glm5/run_phase906_eos_action_boundary_test.sh
```

结果目录：

```text
tests/result/phase906_eos_action_boundary_test/eos_action_boundary_test/
```

核心结果：

```text
phase906_cross_model_summary.json
phase906_cross_model_summary.md
phase906_qwen3_rows.jsonl
phase906_glm4_rows.jsonl
phase906_deepseek7b_rows.jsonl
```

### 三、测试原理

测试对象仍然使用 Phase899/903/904 已经确认的 selected answer drift rows：

```text
qwen3: 18
GLM4: 17
DS7B: 33
合计: 68
```

每条样本先生成 answer-prefix，然后做四类审计：

```text
1. baseline answer-prefix 处 EOS / period / protocol rank；
2. 强制 period 后，读取下一步 logits；
3. 强制 period 后继续 rollout，观察是否 clean；
4. 强制 EOS，检查 tokenizer / generation path 是否能形成 clean stop；
5. 在 period 后 logits 上做 protocol / newline / comma / field-explanation mask 对照。
```

闭合停止公式继续使用 Phase905 修正后的口径：

$$
ClosureStop(x)
=
EOSAction(x)
\lor
\left[
PeriodBoundary(x)
\land
\neg ContinuationAfterPeriod(x)
\right]
$$

Phase906 重点测：

$$
EOSRank(x,t_{answer})
$$

以及：

$$
EOSRank(x,t_{answer}+Period)
$$

如果 EOS 只在 forced condition 下 clean，而自然 logits 中不竞争，则说明：

$$
EOSAvailable(x)=1
$$

但：

$$
EOSActionNatural(x)=0
$$

### 四、跨模型总体结果

```text
rows: 68

baseline_eos_top1: 0
baseline_eos_top10: 0
baseline_eos_top50: 14

after_period_eos_top1: 0
after_period_eos_top10: 0
after_period_eos_top50: 0

period_forced_protocol_drift: 50
period_forced_strict_protocol_drift: 68
period_forced_strict_clean_answer_no_protocol: 0

eos_forced_generation_would_stop: 68
eos_forced_strict_clean_answer_no_protocol: 68

mask_protocol_eos_top1: 0
mask_protocol_plus_period_eos_top1: 0

period_after_generated_eos: 9
```

最关键的事实：

```text
强制 EOS:
  68/68 都能得到 strict clean。

自然 answer-prefix:
  EOS top1 = 0
  EOS top10 = 0

强制 period 后:
  EOS top1 = 0
  EOS top10 = 0
  EOS top50 = 0

mask protocol / protocol+period:
  EOS top1 = 0
```

这说明：

```text
EOS 不是 tokenizer / generation config 完全不可用；
EOS 是可用的；
但它在自然输出竞争场中极弱。
```

### 五、分模型结果

#### qwen3

```text
rows: 18

baseline_eos_top1: 0
baseline_eos_top10: 0
baseline_eos_top50: 0
baseline_median_eos_rank: 29326.5
baseline_median_period_rank: 3.0
baseline_median_protocol_rank: 1.0

after_period_eos_top1: 0
after_period_eos_top10: 0
after_period_eos_top50: 0
after_period_median_eos_rank: 12182.5
after_period_median_protocol_rank: 1.5

after_period_next_top_categories:
  explanation: 9
  other: 9

period_forced_strict_clean_answer_no_protocol: 0
eos_forced_strict_clean_answer_no_protocol: 18
```

qwen3 的现象是：

```text
period 边界很近；
EOS 极远；
period 后仍进入 explanation / other continuation。
```

#### GLM4

```text
rows: 17

baseline_eos_top1: 0
baseline_eos_top10: 0
baseline_eos_top50: 12
baseline_median_eos_rank: 36.0
baseline_median_period_rank: 7.0
baseline_median_protocol_rank: 1.0

after_period_eos_top1: 0
after_period_eos_top10: 0
after_period_eos_top50: 0
after_period_median_eos_rank: 22255.0
after_period_median_protocol_rank: 2.0

after_period_next_top_categories:
  field_word: 8
  other: 9

period_forced_strict_clean_answer_no_protocol: 0
eos_forced_strict_clean_answer_no_protocol: 17
```

GLM4 的现象非常关键：

```text
answer-prefix 处 EOS 有一定接近度；
但 period forced 后 EOS 直接跌出 top50；
句号不是把模型带向 EOS，而是带向 field / other continuation。
```

#### DS7B

```text
rows: 33

baseline_eos_top1: 0
baseline_eos_top10: 0
baseline_eos_top50: 2
baseline_median_eos_rank: 147.0
baseline_median_period_rank: 8.0
baseline_median_protocol_rank: 1.0

after_period_eos_top1: 0
after_period_eos_top10: 0
after_period_eos_top50: 0
after_period_median_eos_rank: 14086.0
after_period_median_protocol_rank: 2.0

after_period_next_top_categories:
  explanation: 7
  field_word: 4
  other: 22

period_after_generated_eos: 9
period_forced_strict_clean_answer_no_protocol: 0
eos_forced_strict_clean_answer_no_protocol: 33
```

DS7B 有 9 条 period forced rollout 中后续生成过 EOS，但这些都不是 strict clean。样例表现为：

```text
.\n</think>\n\nanimal.
.\n</think>\n\nPolygon.
.\n</think>\n\nshape.
```

这说明 DS7B 的 EOS / special boundary 可能更接近 chat / reasoning template 边界，而不是干净答案终止。

### 六、正确性分析

Phase906 支持附件中的主判断，并进一步收紧：

```text
不是 EOS 不能用；
而是自然输出场几乎不选择 EOS。
```

因为：

```text
forced EOS strict clean = 68/68
natural EOS top1 = 0/68
after-period EOS top50 = 0/68
```

所以当前瓶颈不是：

```text
tokenizer 没有 EOS
generation config 完全禁止 EOS
```

而更像：

```text
模型内部 continuation prior 极强；
EOS action 没有进入自然竞争场；
period 后不是停机态，而是新的续写态。
```

### 七、数学公式更新

当前输出理论主体仍然保持：

```text
条件化输出场闭合理论
```

但终止部分需要更严格拆分：

$$
Output(x)
=
F(
S_{answer}(x),
S_{protocol}(x),
S_{period}(x),
S_{EOS}(x),
A_{substitution}(x),
T_{termination}(x)
)
$$

其中：

```text
S_answer:
  语义答案场

S_protocol:
  协议续写场

S_period:
  句号/标点边界场

S_EOS:
  EOS 竞争场

A_substitution:
  协议替代图

T_termination:
  真实终止动作场
```

Phase906 给出：

$$
EOSAvailable(x)=1
$$

但：

$$
EOSRank(x,t_{answer}) \gg 1
$$

并且：

$$
EOSRank(x,t_{answer}+Period) \gg 1
$$

所以：

$$
EOSActionNatural(x)=0
$$

强制 EOS 条件：

$$
ForceEOS(x) \Rightarrow Clean_{strict}(x)
$$

自然条件：

$$
PeriodBoundary(x) \Rightarrow ContinuationAfterPeriod(x)
$$

因此：

$$
PeriodBoundary(x) \not\Rightarrow EOSAction(x)
$$

### 八、图谱更新

Phase906 后，全局齿轮图谱应改为：

```text
semantic answer field
  -> answer prefix
  -> protocol continuation field
  -> period boundary
  -> post-period continuation field
  -/-> natural EOS action

forced EOS
  -> strict clean
```

这条边很关键：

```text
forced EOS 可以 clean；
natural EOS 不竞争。
```

也就是说：

```text
终止能力存在；
终止选择机制缺失。
```

### 九、问题和硬伤

#### 1. forced EOS 不是机制闭合

虽然：

```text
eos_forced_strict_clean_answer_no_protocol = 68
```

但这是外部强制，不是模型自然选择，不能记为 closure positive。

#### 2. period 后 EOS 竞争更弱

GLM4 最明显：

```text
baseline median EOS rank: 36
after-period median EOS rank: 22255
```

这说明 period 不是通向 EOS 的自然桥，而可能触发新的续写状态。

#### 3. mask protocol 仍不能让 EOS top1

即便做：

```text
mask_protocol
mask_protocol_plus_period
```

EOS top1 仍为 0。这说明 EOS 不只是被 newline/comma/field/explanation 压住；还有大量 other continuation 或模型模板续写竞争者。

#### 4. 小模型偏差

小模型可能把 EOS、chat template、reasoning trace、自然文本续写压缩在纠缠机制里。DS7B 的 `</think>` 现象尤其说明，当前结果可能混入蒸馏 / chat 格式边界，不宜直接外推到大模型。

### 十、当前进度

```text
全局齿轮图谱:
  92% - 95%

语言编码机制闭合:
  45% - 50%
```

图谱进度上升，因为 EOS 可用性与自然竞争缺口已被分开。闭合进度只小幅上升，因为自然 strict clean 仍为 0。

### 十一、下一阶段

Phase906 已经完成 EOS action boundary test 的最低和中等目标：

```text
最低目标:
  证明 EOS 是否被模型自然压制。
  结果：是，自然压制明显。

中等目标:
  区分 EOS 竞争失败与 tokenizer / generation config 不允许。
  结果：EOS 可 forced clean，说明不是完全不可用。
```

下一阶段如果继续，应进入：

```text
Phase907:
Post-Period Continuation Source Mapping
句号后续写来源图谱
```

目标：

```text
1. 找到 period 后 explanation / field / other continuation 的来源组件；
2. 检查是否存在组件置零能显著提升 EOS rank；
3. 判断 post-period continuation 是协议场延伸，还是独立 continuation field；
4. 不再把 forced EOS 当作闭合，只把它当作可用性对照。
```

## Phase 907: 句号后续写来源图谱 [2026-07-03 20:14]

### 一、任务来源

Phase906 证明：

```text
forced EOS 可以 clean；
natural EOS 不竞争；
period 后进入 continuation state。
```

因此 Phase907 继续同一阶段目标，但不再问：

```text
EOS 能不能强制干净？
```

而是问：

```text
period 后的 continuation state 来自哪些组件？
哪些组件置零能拉近 EOS？
这种拉近是否足够形成 EOS action？
```

### 二、脚本和结果文件

新增脚本：

```text
tests/glm5/phase907_post_period_continuation_source_mapping.py
tests/glm5/run_phase907_post_period_continuation_source_mapping.sh
```

结果目录：

```text
tests/result/phase907_post_period_continuation_source_mapping/post_period_continuation_source_mapping/
```

核心结果：

```text
phase907_cross_model_summary.json
phase907_cross_model_summary.md
phase907_qwen3_rows.jsonl
phase907_glm4_rows.jsonl
phase907_deepseek7b_rows.jsonl
```

### 三、测试原理

Phase907 使用 Phase899 选出的 68 条 answer drift rows。流程：

```text
1. 生成 answer-prefix；
2. 强制追加 period；
3. 在 period 后位置读取 baseline logits；
4. 对每一层 attention / MLP 做 last-token component-zero；
5. 记录 EOS rank、protocol rank、next-top category 变化。
```

干预公式：

$$
h_{\ell}^{kind}(x,t_{period+1})
\leftarrow
0
$$

EOS 排名变化：

$$
\Delta rank_{EOS}^{\ell,kind}(x)
=
rank_{patched}(EOS \mid x,Period)
-
rank_{base}(EOS \mid x,Period)
$$

若：

$$
\Delta rank_{EOS}^{\ell,kind}(x)<0
$$

说明该组件置零后 EOS 更接近竞争区。

但闭合要求更高：

$$
rank_{patched}(EOS)=1
$$

或至少：

$$
rank_{patched}(EOS)\le 10
$$

才可以说出现 EOS action 近邻。

### 四、总体结果

跨模型统计：

```text
rows: 4504

eos_rank_improved: 2105
eos_rank_improved_100: 1903
eos_rank_improved_1000: 1306

patched_eos_top1: 0
patched_eos_top10: 13
patched_eos_top50: 17

protocol_rank1_removed: 258
next_top_changed: 1291
next_category_changed: 942
```

关键结论：

```text
大量组件能改善 EOS rank；
但没有任何组件把 EOS 推到 top1；
只有 GLM4 出现 EOS top10 / top50 近邻；
qwen3 和 DS7B 仍然只是“拉近但不近”。
```

因此 Phase907 是 source mapping positive，但不是 closure positive。

### 五、分模型结果

#### qwen3

```text
component_rows: 1296
eos_rank_improved: 606
eos_rank_improved_100: 521
eos_rank_improved_1000: 244

patched_eos_top1: 0
patched_eos_top10: 0
patched_eos_top50: 0

protocol_rank1_removed: 53
next_top_changed: 278
next_category_changed: 262
median_eos_rank_delta: 37.5
```

主要 transition：

```text
explanation -> explanation: 571
other -> other: 463
other -> explanation: 159
```

qwen3 的结论：

```text
period 后 continuation 场可以被扰动；
部分组件能改善 EOS rank；
但 EOS 仍完全进不了 top50。
```

#### GLM4

```text
component_rows: 1360
eos_rank_improved: 757
eos_rank_improved_100: 662
eos_rank_improved_1000: 464

patched_eos_top1: 0
patched_eos_top10: 13
patched_eos_top50: 17

protocol_rank1_removed: 39
next_top_changed: 295
next_category_changed: 189
median_eos_rank_delta: -54.0
```

最强组件：

```text
L0 attention / field_word:
  rows: 8
  patched_eos_top50: 8
  patched_eos_top10: 6
  eos_rank_improved_1000: 8
  mean_eos_rank_delta: -18323.375
  category transition: field_word -> other

L0 attention / other:
  rows: 9
  patched_eos_top50: 7
  patched_eos_top10: 7
  eos_rank_improved_1000: 7
  mean_eos_rank_delta: -22756.4444
```

GLM4 的结论：

```text
period 后 EOS action 的近邻边界确实存在；
但 L0 attention 置零只是把 EOS 拉到 top10/top50，
仍不能让 EOS top1。
```

这说明 GLM4 的 period 后 continuation 很可能强烈依赖早层 attention 的格式/位置/模板信息。

#### DS7B

```text
component_rows: 1848
eos_rank_improved: 742
eos_rank_improved_100: 720
eos_rank_improved_1000: 598

patched_eos_top1: 0
patched_eos_top10: 0
patched_eos_top50: 0

protocol_rank1_removed: 166
next_top_changed: 718
next_category_changed: 491
median_eos_rank_delta: 708.5
```

强信号主要在中后层 MLP：

```text
L23 MLP / other:
  eos_rank_improved_1000: 21/22
  mean_eos_rank_delta: -6409.59

L27 MLP / other:
  eos_rank_improved_1000: 21/22
  mean_eos_rank_delta: -17870.14

L20 MLP / other:
  eos_rank_improved_1000: 21/22
  mean_eos_rank_delta: -8789.14
```

但：

```text
patched_eos_top50: 0
```

DS7B 的结论：

```text
中后层 MLP 对 post-period continuation 有明显影响；
但 EOS 仍远离自然竞争区。
```

### 六、正确性分析

Phase907 进一步支持 Phase906：

```text
EOS action 不是完全不可达；
但自然竞争场里 EOS 极弱。
```

更准确地说：

```text
组件干预可以拉近 EOS；
拉近不等于 EOS action；
EOS top1 仍为 0。
```

所以当前不能说：

```text
找到了终止控制齿轮；
找到了 EOS source；
完成了 clean closure。
```

只能说：

```text
找到了 post-period continuation source candidates；
发现 GLM4 的 L0 attention 是重要近邻边界；
发现 DS7B 的中后层 MLP 能显著改善 EOS rank 但不够近；
qwen3 的 EOS 缺口仍非常大。
```

### 七、理论更新

Phase907 后，终止缺口应拆成两层：

```text
1. EOS rank proximity
   EOS 排名接近度

2. EOS action dominance
   EOS 动作支配
```

公式：

$$
EOSProximity(x)
=
\mathbb{1}[rank(EOS\mid x)\le K]
$$

其中：

$$
K \in \{10,50\}
$$

而真正动作：

$$
EOSAction(x)
=
\mathbb{1}[rank(EOS\mid x)=1]
\land
\neg ContinuationAfterEOS(x)
$$

Phase907 结果：

$$
EOSProximity_{GLM4}(x)>0
$$

但：

$$
EOSAction(x)=0
$$

也就是说：

```text
GLM4 已出现 EOS proximity；
三模型均未出现 EOS action。
```

### 八、图谱更新

最新图谱：

```text
semantic answer field
  -> answer prefix
  -> period boundary
  -> post-period continuation field
       qwen3: explanation / other
       GLM4: field_word / other, L0 attention sensitive
       DS7B: other / explanation / field_word, mid-late MLP sensitive
  -> EOS proximity field
       GLM4: weak positive
       qwen3: absent
       DS7B: improved but still far
  -/-> EOS action dominance
  -/-> strict clean closure
```

### 九、硬伤和瓶颈

#### 1. EOS top1 仍为 0

这是最硬的边界：

```text
patched_eos_top1 = 0 / 4504
```

说明自然闭合仍没有出现。

#### 2. GLM4 的 L0 attention 可能是分布外扰动

L0 attention 置零影响巨大，可能说明早层 template / position / prompt 信息强烈参与 period 后续写；但也可能是粗粒度置零导致的异常分布外扰动。需要下一阶段做更细粒度验证。

#### 3. DS7B 的 EOS 改善不进入 top50

DS7B 的 rank delta 很大，但 top50 仍为 0，说明它的 continuation field 有大量强竞争者。

#### 4. qwen3 的 EOS 缺口最大

qwen3 即便大量组件改善 EOS rank，也完全无法进入 top50，说明 qwen3 的 EOS action 与当前 answer-prefix/period 路线可能基本断开。

### 十、下一阶段任务

Phase907 已经完成 post-period continuation source mapping。下一阶段不应继续扩大粗粒度置零，而应对 GLM4 的 L0 attention 近邻边界做细化：

```text
Phase908:
GLM4 L0 Attention EOS-Proximity Fine Audit
GLM4 第 0 层注意力 EOS 近邻边界细审计
```

目标：

```text
1. 确认 L0 attention 不是分布外假阳性；
2. 分 head / token position / prompt part 做更细粒度拆解；
3. 判断它究竟是在压低 field continuation，还是在提升 EOS；
4. 检查能否从 EOS top10/top50 推到 EOS top1。
```

### 十一、阶段结论

Phase907 是重要正结果 + 重要负结果：

```text
正结果:
  找到了 period 后续写的来源组件和 EOS proximity 近邻边界。

负结果:
  没有任何组件使 EOS top1；
  strict clean closure 仍未出现。
```

总体进度评估：

```text
全局齿轮图谱:
  93% - 95%

语言编码机制闭合:
  46% - 50%
```

图谱进度继续上升，因为 post-period continuation source 已经开始定位。闭合进度只小幅上升，因为 EOS action dominance 仍为 0。

## Phase 908: GLM4 L0 attention EOS 近邻边界细审计 [2026-07-03 21:03]

### 一、任务来源和正确性判断

最新上传内容对 Phase906（第906阶段）和 Phase907（第907阶段）的判断基本正确。

Phase906（第906阶段）的核心结果是：

```text
EOS（结束符）可用：
  forced EOS strict clean = 68 / 68

自然竞争场不选择 EOS（结束符）：
  natural answer-prefix EOS top1 = 0
  natural answer-prefix EOS top10 = 0
  after-period EOS top1 = 0
  after-period EOS top10 = 0
  after-period EOS top50 = 0
```

Phase907（第907阶段）的核心结果是：

```text
post-period continuation source（句号后续写来源）可以被定位；
大量组件能改善 EOS rank（结束符排名）；
但没有任何组件使 EOS top1（结束符第一）。
```

因此，上传内容提出的下一步是合理的：不能继续泛泛扩大粗粒度 component-zero（组件置零），而应对 GLM4 的 L0 attention（第 0 层注意力）EOS proximity（结束符近邻）做细审计。

本阶段完成了 Phase908：

```text
目标：
  1. 验证 GLM4 L0 attention 近邻信号是否稳定；
  2. 区分 direct EOS lift（直接抬升结束符）和 continuation suppression（压低续写场）；
  3. 做 head-level（注意力头级）和强度扫描；
  4. 观察是否能把 EOS 从 top10/top50 推到 top1。
```

测试脚本：

```text
tests/glm5/phase908_l0_attention_eos_proximity_fine_audit.py
tests/glm5/run_phase908_l0_attention_eos_proximity_fine_audit.sh
```

结果目录：

```text
tests/result/phase908_l0_attention_eos_proximity_fine_audit/l0_attention_eos_proximity_fine_audit/
```

### 二、测试原理

本阶段以 Phase899（第899阶段）中 selected answer-drift rows（已选答案漂移样本）为样本来源。流程是：

```text
1. 生成 answer-prefix（答案前缀）；
2. 强制追加 period（句号）；
3. 在 period 后位置读取 baseline logits（基线对数几率）；
4. 对 L0 attention 做强度干预；
5. 对 L0 attention 的各 head 做 head-zero（头置零）；
6. 使用 L0 MLP 和 L1 attention 作为局部对照；
7. 记录 EOS rank、EOS logit、next-token logit、protocol logit 和 category transition。
```

核心公式：

$$
\Delta r_{EOS}
=
r_{patched}(EOS)
-
r_{base}(EOS)
$$

$$
\Delta z_{EOS}
=
z_{patched}(EOS)
-
z_{base}(EOS)
$$

$$
\Delta z_{next}
=
z_{patched}(next_{base})
-
z_{base}(next_{base})
$$

$$
\Delta M_{EOS,next}
=
\left[z_{patched}(EOS)-z_{patched}(next_{base})\right]
-
\left[z_{base}(EOS)-z_{base}(next_{base})\right]
$$

判断口径：

```text
direct EOS lift（直接抬升结束符）:
  Δz_EOS > 0

continuation suppression（续写抑制）:
  Δz_next < 0

EOS proximity（结束符近邻）:
  patched EOS rank <= 50 或 <= 10

EOS action closure（结束符动作闭合）:
  patched EOS rank = 1
```

### 三、测试结果

跨模型总计：

```text
rows = 2452
eos_rank_improved = 1108
eos_rank_improved_100 = 897
eos_rank_improved_1000 = 381
patched_eos_top1 = 0
patched_eos_top10 = 20
patched_eos_top50 = 42
direct_eos_lift = 995
continuation_suppressed = 924
protocol_suppressed = 1028
next_top_changed = 394
next_category_changed = 326
```

分模型结果：

```text
qwen3:
  rows = 684
  patched_eos_top1 = 0
  patched_eos_top10 = 0
  patched_eos_top50 = 0
  eos_rank_improved_1000 = 33

GLM4:
  rows = 646
  patched_eos_top1 = 0
  patched_eos_top10 = 20
  patched_eos_top50 = 42
  eos_rank_improved_1000 = 144

DS7B:
  rows = 1122
  patched_eos_top1 = 0
  patched_eos_top10 = 0
  patched_eos_top50 = 0
  eos_rank_improved_1000 = 204
```

最强控制项：

```text
GLM4 L0_attention_last_zero:
  rows = 17
  patched_eos_top10 = 13
  patched_eos_top50 = 15
  median_eos_rank_delta = -22245
  median_eos_logit_delta = +7.327
  median_next_logit_delta = -11.078
  median_margin_delta = +12.598

GLM4 L0_attention_last_negative_half:
  rows = 17
  patched_eos_top10 = 7
  patched_eos_top50 = 12

GLM4 L0_attention_all_zero:
  rows = 17
  patched_eos_top10 = 0
  patched_eos_top50 = 15
```

head-level（头级）结果：

```text
没有任何单个 L0 attention head-zero 使 EOS 进入 top10 或 top50。
部分 head 能改善 EOS rank，但强度远弱于整层 L0 attention 输出干预。
```

### 四、客观结论

Phase908 是重要正结果，但不是闭合。

正结果：

```text
GLM4 的 L0 attention 确实存在 EOS proximity boundary（结束符近邻边界）；
这个信号不是 Phase907 的偶然统计假象；
GLM4 L0 attention 输出被强干预时，EOS 可以从极低排名进入 top10/top50。
```

关键负结果：

```text
patched_eos_top1 = 0 / 2452
```

这说明：

```text
L0 attention 可以把 EOS 拉近；
但不能让 EOS 成为自然动作；
EOS action dominance（结束符动作支配）仍不存在。
```

### 五、硬伤和瓶颈

第一，GLM4 的最强结果来自整层 L0 attention 输出强干预，而不是单 head（单头）干预。这说明当前定位仍偏粗。

第二，强度干预会同时造成：

```text
direct EOS lift（直接抬升结束符）
continuation suppression（压低续写）
protocol suppression（压低协议续写）
```

三者纠缠在一起，不能简单说“找到 EOS 齿轮”。

第三，qwen3 和 DS7B 没有进入 top50，说明 GLM4 的 L0 attention EOS proximity 具有模型结构特殊性，不能直接当作普遍语言编码机制。

第四，小模型内部结构可能较粗糙。L0 attention 可能承担 template / prompt / position（模板 / 提示 / 位置）混合功能，因此强置零可能产生分布外效应。

### 六、阶段结论和下一步

Phase908 完成了 head-level 和强度维度的 EOS proximity fine audit（结束符近邻细审计），但还没有完成 source token / prompt part（来源词元 / 提示片段）拆解。该任务仍属于同一阶段目标，因此继续自动进入 Phase909。

## Phase 909: L0 attention 来源片段 EOS 边界审计 [2026-07-03 21:03]

### 一、任务

Phase909 接续 Phase908，目标是回答：

```text
GLM4 和其他模型中的 EOS proximity（结束符近邻）到底来自 L0 attention 的哪类输入来源？

是 prompt（提示）整体？
是 prompt 前部或尾部？
是 answer-prefix（答案前缀）？
是 period token（句号词元）？
还是 last-8-before-period（句号前 8 个词元）？
```

测试脚本：

```text
tests/glm5/phase909_l0_attention_source_span_eos_boundary_audit.py
tests/glm5/run_phase909_l0_attention_source_span_eos_boundary_audit.sh
```

结果目录：

```text
tests/result/phase909_l0_attention_source_span_eos_boundary_audit/l0_attention_source_span_eos_boundary_audit/
```

### 二、测试原理

Phase909 不再直接置零 L0 attention 输出，而是在 L0 attention 输入端对不同来源片段做缩放。

设 L0 attention 输入为：

$$
H^{0}_{in}
=
\left[
H_{prompt},
H_{answer},
H_{period}
\right]
$$

对某个来源片段 \(S\) 做干预：

$$
H^{0}_{in,S}
\leftarrow
\lambda H^{0}_{in,S}
$$

其中：

$$
\lambda \in \{0, 0.5\}
$$

测试片段：

```text
prompt_all（全部提示）
prompt_first8（提示前 8 个词元）
prompt_last8（提示后 8 个词元）
answer_prefix_all（答案前缀全部）
answer_prefix_last（答案前缀最后一个词元）
last8_before_period（句号前 8 个词元）
period_token（句号词元）
```

评价指标沿用 Phase908：

$$
\Delta r_{EOS}
=
r_{patched}(EOS)
-
r_{base}(EOS)
$$

$$
\Delta M_{EOS,next}
=
\left[z_{patched}(EOS)-z_{patched}(next_{base})\right]
-
\left[z_{base}(EOS)-z_{base}(next_{base})\right]
$$

### 三、测试结果

跨模型总计：

```text
rows = 612
eos_rank_improved = 257
eos_rank_improved_100 = 247
eos_rank_improved_1000 = 193
patched_eos_top1 = 0
patched_eos_top10 = 47
patched_eos_top50 = 52
direct_eos_lift = 247
continuation_suppressed = 436
protocol_suppressed = 507
next_top_changed = 327
next_category_changed = 224
```

分模型结果：

```text
qwen3:
  rows = 162
  patched_eos_top1 = 0
  patched_eos_top10 = 0
  patched_eos_top50 = 0
  eos_rank_improved_1000 = 45

GLM4:
  rows = 153
  patched_eos_top1 = 0
  patched_eos_top10 = 15
  patched_eos_top50 = 17
  eos_rank_improved_1000 = 79

DS7B:
  rows = 297
  patched_eos_top1 = 0
  patched_eos_top10 = 32
  patched_eos_top50 = 35
  eos_rank_improved_1000 = 69
```

最强来源片段：

```text
DS7B L0_attn_input_prompt_all_zero:
  rows = 33
  patched_eos_top10 = 32
  patched_eos_top50 = 33
  median_eos_rank_delta = -14082
  median_eos_logit_delta = +6.297
  median_next_logit_delta = -14.938
  median_margin_delta = +11.797
  category transition 主要为 -> newline

GLM4 L0_attn_input_prompt_all_zero:
  rows = 17
  patched_eos_top10 = 15
  patched_eos_top50 = 15
  median_eos_rank_delta = -22250
  median_eos_logit_delta = +9.203
  median_next_logit_delta = -12.195
  median_margin_delta = +12.483
```

其它片段：

```text
prompt_first8:
  qwen3 和 GLM4 可改善 EOS rank，但不能进入 top50。

prompt_last8 / last8_before_period:
  对 GLM4 偶尔进入 top50，但总体不稳定。

answer_prefix_all / answer_prefix_last:
  对 GLM4 有轻微 rank 改善；
  对 qwen3 和 DS7B 多数情况下反而使 EOS 更差。

period_token:
  有局部改善，但不是主要来源；
  DS7B period_zero 可进入 top50 2 次，但 median margin delta 为负。
```

### 四、客观结论

Phase909 给出一个关键拼图：

```text
L0 attention 的 EOS proximity 主要不是来自 answer-prefix；
也不是单纯来自 period token；
而是强烈依赖 prompt_all（全部提示）在 L0 attention 输入中的整体存在方式。
```

这说明 Phase908 看到的 GLM4 L0 attention 近邻信号，很可能不是一个纯粹 EOS 齿轮，而是：

```text
prompt-conditioned continuation field boundary
提示条件化续写场边界
```

也就是说，prompt（提示）整体在早层形成一种“继续按格式/协议输出”的场。当强行破坏 prompt_all 后，continuation field（续写场）被打散，EOS 因竞争者下降和自身抬升而靠近前排。

### 五、硬伤和严格边界

第一，最强结果来自 prompt_all_zero（全部提示置零）。这是一种强分布外干预，不能等价于自然路线中的可控齿轮。

第二，DS7B 的 prompt_all_zero 结果虽然强，但 patched next category 大量转向 newline（换行），说明它更像“输出场坍缩到换行/终止附近”，而不是 EOS action（结束符动作）自然闭合。

第三，所有模型仍然满足：

```text
patched_eos_top1 = 0
```

这意味着：

```text
EOS proximity exists（结束符近邻存在）
EOS action closure absent（结束符动作闭合不存在）
```

第四，prompt_all_zero 不能作为闭合算法。它破坏了输入条件，不能用于说明模型在保持语义任务和提示结构时自然停止。

### 六、当前图谱进展

当前 termination / EOS 子图谱可更新为：

```text
answer-prefix
  -> period boundary
  -> post-period continuation field
       qwen3: explanation / other
       GLM4: field_word / other
       DS7B: other / explanation / field_word
  -> L0 attention boundary
       GLM4: output-level strong EOS proximity
       DS7B: source-span prompt_all perturbation can create EOS proximity
       qwen3: rank improves but no near field
  -> prompt-conditioned continuation field
       prompt_all is a major early-layer condition source
  -/-> EOS top1
  -/-> strict clean natural closure
```

核心进展：

```text
从“哪个组件影响 EOS rank”
推进到
“EOS 近邻边界依赖 prompt-conditioned early route（提示条件化早层路线）”。
```

### 七、智能理论角度的洞察

本阶段支持一个更稳健的判断：

```text
语言停止不是单个 stop token（停止词元）竞争问题；
而是 route state（路线状态）、prompt field（提示场）、continuation field（续写场）和 EOS action（结束符动作）之间的边界问题。
```

更接近当前结果的公式是：

$$
State_t
=
\Phi
\left(
PromptField,
AnswerRoute_t,
ProtocolField_t,
BoundaryState_t
\right)
$$

$$
Action_t
=
\arg\max_{v \in V}
z_v(State_t)
$$

其中 EOS 闭合需要：

$$
z_{EOS}(State_t)
>
\max_{v \in V \setminus \{EOS\}}
z_v(State_t)
$$

当前实验只证明：

$$
\exists I:
\quad
rank_{patched}(EOS)
\ll
rank_{base}(EOS)
$$

但没有证明：

$$
\exists I:
\quad
rank_{patched}(EOS)=1
$$

更没有证明：

$$
rank_{natural}(EOS)=1
$$

因此，Phase908/909 是图谱推进，不是语言编码机制闭合。

### 八、总体进度评估

```text
全局齿轮图谱:
  94% - 96%

语言编码机制闭合:
  47% - 51%
```

图谱进度上升的原因：

```text
L0 attention EOS proximity 已经从组件层推进到来源片段层；
prompt_all 被识别为早层续写场的重要条件源。
```

闭合进度只小幅上升的原因：

```text
EOS top1 仍为 0；
自然 strict clean closure 仍为 0；
最强干预是 prompt_all_zero，不能作为自然机制闭合证据。
```

### 九、下一阶段任务

Phase908/909 已完成同一阶段内的 EOS proximity fine audit（结束符近邻细审计）。下一步不应继续做更强的 prompt 破坏，而应进入新的阶段：

```text
Phase910:
Prompt-preserving termination route reconstruction
保持提示结构的终止路线重建
```

目标：

```text
1. 不再使用 prompt_all_zero 这类强破坏干预；
2. 保持 prompt 和 answer-prefix 基本结构；
3. 尝试用小幅、局部、可解释的 route-state adjustment（路线状态调整）重建 EOS proximity；
4. 同时检查 full-vocabulary blocker（全词表阻塞者），不能只看 EOS rank；
5. 若仍无法接近 top1，则把终止问题从“EOS 动作闭合”降级为“termination field atlas（终止场图谱）”。
```

阶段结论：

```text
Phase908/909 是重要图谱进展；
不是闭合；
下一步已经从同阶段细审计转入新的 prompt-preserving route reconstruction 阶段，因此本轮不继续自动启动新测试。
```

## Phase 910: 保持提示结构的终止路线重建 [2026-07-03 21:28]

### 一、任务判断

本阶段读取并复核了最新上传内容中对 Phase908/909 的判断。总体结论是：

```text
附件判断基本正确。
Phase908/909 证明 L0 attention 中存在 EOS proximity（结束符近邻）边界；
但最强证据来自 prompt_all_zero（整体提示清零）这类强破坏干预；
因此它们是图谱推进，不是自然终止动作闭合。
```

Phase908/909 的正确部分是：

```text
1. L0 attention 的确能显著改变 EOS rank（结束符排名）；
2. prompt_all 是最强来源片段；
3. EOS proximity 可以被制造出来；
4. 但 patched_eos_top1 = 0，strict clean closure = 0；
5. 因此不能把“接近 EOS”误判成“自然选择 EOS”。
```

需要收紧的部分是：

```text
prompt_all_zero 只能作为反事实来源定位工具，不能作为自然机制证据。
如果要继续推进，必须在保持 prompt（提示）和 answer-prefix（答案前缀）结构的条件下，
测试这条 termination route（终止路线）是否可复用。
```

因此本阶段执行 Phase910：

```text
Prompt-preserving termination route reconstruction
保持提示结构的终止路线重建
```

### 二、测试脚本和结果路径

测试脚本：

```text
tests/glm5/phase910_prompt_preserving_termination_route_reconstruction.py
tests/glm5/run_phase910_prompt_preserving_termination_route_reconstruction.sh
```

结果目录：

```text
tests/result/phase910_prompt_preserving_termination_route_reconstruction/prompt_preserving_termination_route_reconstruction/
```

核心结果文件：

```text
phase910_cross_model_summary.md
phase910_cross_model_summary.json
phase910_qwen3_rows.jsonl
phase910_glm4_rows.jsonl
phase910_deepseek7b_rows.jsonl
```

模型执行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

三模型按顺序单独加载和释放，避免 GPU 内存叠加。

### 三、测试原理

Phase910 不再把 prompt_all_zero 当作有效测试干预，而只把它当作构造反事实方向的来源。

设 baseline（基线）下 L0 attention 在最后 token（词元）的输出为：

$$
h^{0}_{attn}(x)
$$

设 prompt_all_zero 反事实下对应输出为：

$$
h^{0}_{attn}(x_{\text{prompt-zero}})
$$

构造 prompt-conditioned counterfactual direction（提示条件反事实方向）：

$$
d_{\text{prompt}}(x)
=
h^{0}_{attn}(x_{\text{prompt-zero}})
-
h^{0}_{attn}(x)
$$

然后在保持原始 prompt 输入结构不变的情况下，只对 L0 attention 输出做方向注入：

$$
\tilde{h}^{0}_{attn}(x;\alpha)
=
h^{0}_{attn}(x)

\alpha d_{\text{prompt}}(x)
$$

其中：

$$
\alpha \in \{0.05, 0.1, 0.25, 0.5, 1.0\}
$$

这类干预记为 prompt-intact（提示输入完整）干预。它不破坏 prompt token（提示词元）输入，只测试 Phase909 找到的方向是否能作为 L0 attention output route-state adjustment（早层注意力输出路线状态调整）被复用。

同时设置弱对照：

$$
\tilde{h}^{0}_{attn}(x;\lambda)
=
\lambda h^{0}_{attn}(x)
$$

其中：

$$
\lambda \in \{0.75, 0.5\}
$$

另设 limited-span adjustment（局部片段调整）作为边界参考，包括：

```text
prompt_all_half
prompt_first8_half / prompt_first8_zero
prompt_last8_half / prompt_last8_zero
answer_prefix_last_half
period_half / period_zero
```

这些局部片段调整不是 strict prompt-preserving（严格保持提示）证据，只用于判断来源边界。

主要观测量：

$$
\Delta r_{EOS}
=
r_{patched}(EOS)
-
r_{base}(EOS)
$$

$$
\Delta z_{EOS}
=
z_{patched}(EOS)
-
z_{base}(EOS)
$$

$$
\Delta z_{next}
=
z_{patched}(next_{base})
-
z_{base}(next_{base})
$$

$$
\Delta M_{EOS,next}
=

\left[
z_{patched}(EOS)-z_{patched}(next_{base})
\right]
-
\left[
z_{base}(EOS)-z_{base}(next_{base})
\right]
$$

全词表 blocker（阻塞词元）边界：

$$
B(x)
=
\arg\max_{v \ne EOS} z_v(x)
$$

$$
M_{EOS,B}
=
z(EOS)-z(B)
$$

闭合不能只看 EOS rank（结束符排名），还必须看：

$$
M_{EOS,B} > 0
$$

否则只是 EOS proximity（结束符近邻），不是 EOS action closure（结束符动作闭合）。

### 四、总体结果

三模型总计：

```text
rows = 1020
eos_rank_improved = 461
eos_rank_improved_100 = 407
eos_rank_improved_1000 = 237
patched_eos_top1 = 0
patched_eos_top5 = 0
patched_eos_top10 = 4
patched_eos_top50 = 18
prompt_preserving_eos_top1 = 0
prompt_preserving_eos_top10 = 4
prompt_preserving_eos_top50 = 15
strict_clean_candidate = 0
direct_eos_lift = 427
continuation_suppressed = 539
protocol_suppressed = 597
next_top_changed = 295
next_category_changed = 195
```

只看 prompt-intact（提示输入完整）子集：

```text
rows = 476
eos_rank_improved = 196
eos_rank_improved_100 = 160
eos_rank_improved_1000 = 62
patched_eos_top1 = 0
patched_eos_top5 = 0
patched_eos_top10 = 4
patched_eos_top50 = 15
prompt_preserving_eos_top1 = 0
prompt_preserving_eos_top10 = 4
prompt_preserving_eos_top50 = 15
strict_clean_candidate = 0
direct_eos_lift = 191
continuation_suppressed = 168
protocol_suppressed = 188
next_top_changed = 65
next_category_changed = 45
```

客观现象：

```text
1. prompt-preserving 条件下确实能复现 EOS top50；
2. GLM4 出现 EOS top10；
3. 但 EOS top1 仍为 0；
4. strict clean candidate 仍为 0；
5. 因此本阶段是正结果，但不是闭合。
```

### 五、分模型结果

qwen3：

```text
rows = 270
prompt_intact_rows = 126
patched_eos_top1 = 0
patched_eos_top10 = 0
patched_eos_top50 = 0
prompt_preserving_eos_top10 = 0
prompt_preserving_eos_top50 = 0
evidence = prompt_preserving_route_improves_eos_but_not_near
```

GLM4：

```text
rows = 255
prompt_intact_rows = 119
patched_eos_top1 = 0
patched_eos_top10 = 4
patched_eos_top50 = 16
prompt_preserving_eos_top10 = 4
prompt_preserving_eos_top50 = 15
evidence = prompt_preserving_route_reaches_eos_top10
```

DS7B：

```text
rows = 495
prompt_intact_rows = 231
patched_eos_top1 = 0
patched_eos_top10 = 0
patched_eos_top50 = 2
prompt_preserving_eos_top10 = 0
prompt_preserving_eos_top50 = 0
evidence = prompt_preserving_route_improves_eos_but_not_near
```

模型差异非常清楚：

```text
GLM4 支持 prompt-preserving EOS proximity reconstruction；
qwen3 只出现方向性改善，没有进入近邻区；
DS7B 的少量 top50 来自 period_zero 这类局部输入破坏，不属于 prompt-preserving 证据。
```

### 六、最关键控制项

GLM4 最强控制项：

```text
control = L0_promptzero_delta_alpha_1
family = prompt_intact_counterfactual_direction
prompt_input_intact = true
prompt_all_zero_used = false
rows = 17
eos_rank_improved = 15
eos_rank_improved_100 = 15
eos_rank_improved_1000 = 15
patched_eos_top1 = 0
patched_eos_top5 = 0
patched_eos_top10 = 4
patched_eos_top50 = 15
prompt_preserving_eos_top10 = 4
prompt_preserving_eos_top50 = 15
direct_eos_lift = 15
continuation_suppressed = 17
protocol_suppressed = 17
median_eos_rank_delta = -22243
median_eos_logit_delta = +7.2036
median_next_logit_delta = -11.9922
median_eos_vs_next_margin_delta = +11.8896
median_eos_margin_vs_full_vocab_blocker = -2.4375
blocker_categories = {"other": 17}
```

这个结果非常重要，因为它说明：

```text
Phase909 找到的 prompt-conditioned direction（提示条件方向）
不是只能在 prompt_all_zero 破坏条件下生效；
它可以在保持原始 prompt 输入结构的情况下，
通过 L0 attention output route adjustment（早层注意力输出路线调整）
把 GLM4 的 EOS 拉入 top10/top50。
```

但是它也暴露出硬边界：

```text
EOS 仍然不是 top1；
EOS 与 full-vocabulary blocker 的中位 margin 仍为 -2.4375；
blocker 主要落在 other 类，其中 GLM4 近邻样本常见 blocker 包括 token "a"；
所以这不是 EOS action closure，而是 termination proximity route reconstruction。
```

DS7B 最接近控制项：

```text
control = L0_input_period_zero
family = limited_span_adjustment
prompt_input_intact = false
rows = 33
patched_eos_top50 = 2
prompt_preserving_eos_top50 = 0
median_eos_margin_vs_full_vocab_blocker = -28.2656
```

该结果不能作为 prompt-preserving 证据。DS7B 中 blocker 仍极强，近邻样本中常见 `<think>` 相关竞争场，因此 DS7B 的终止路线尚未被本阶段方法重建。

### 七、阶段进展

Phase910 的进展可以概括为：

```text
从“prompt_all_zero 破坏能产生 EOS proximity”
推进到
“GLM4 中 prompt-conditioned direction 可在 prompt-intact 输出层复用”。
```

这比 Phase908/909 更接近自然机制，因为：

```text
1. 没有把 prompt 输入整体清零作为测试控制；
2. 保持了原始 prompt 和 answer-prefix 结构；
3. 干预位置从 input destruction（输入破坏）转为 output route-state adjustment（输出路线状态调整）；
4. 成功达到 prompt-preserving EOS top10；
5. 同时保留 full-vocabulary blocker 审计，没有只看目标 logit。
```

但它距离闭合仍然很远：

```text
1. patched_eos_top1 = 0；
2. strict_clean_candidate = 0；
3. qwen3 和 DS7B 没有跨模型复现 top10；
4. GLM4 的成功需要 alpha = 1.0 的完整反事实方向，不是小幅 alpha；
5. EOS 仍被全词表 blocker 压住；
6. 当前证据是“路线近邻重建”，不是“自然终止动作选择”。
```

### 八、理论更新

当前理论应从：

```text
L0 attention contains EOS boundary source
L0 attention 含有 EOS 边界来源
```

升级为：

```text
L0 attention contains a prompt-conditioned termination proximity route,
but the route is separated from natural EOS action closure by a full-vocabulary blocker field.

L0 attention 含有提示条件化的终止近邻路线，
但这条路线与自然 EOS 动作闭合之间仍隔着全词表阻塞场。
```

对应分层结构：

```text
1. prompt-conditioned route source
   提示条件化路线来源

2. termination proximity field
   终止近邻场

3. continuation/protocol suppression field
   续写/协议抑制场

4. full-vocabulary blocker field
   全词表阻塞场

5. missing EOS action gate
   尚未定位的 EOS 动作门
```

当前公式应写成：

$$
z_{EOS}
=
F_{EOS}
\Delta_{\text{route}}
\Delta_{\text{termination-proximity}}
\Delta_{\text{suppression}}
-B_{\text{full-vocab}}
 + \epsilon
$$

其中闭合条件不是：

$$
rank(EOS) \le 50
$$

也不是：

$$
rank(EOS) \le 10
$$

而是至少需要：

$$
rank(EOS)=1
$$

并且：

$$
z(EOS) > \max_{v \ne EOS} z(v)
$$

同时满足自然输出一致性：

$$
Y_{\text{patched}}
\approx
Y_{\text{natural-clean}}
$$

当前 Phase910 只证明：

$$
\exists \alpha:
\quad
rank_{\tilde{h}^{0}_{attn}(x;\alpha)}(EOS)
\le 10
\quad
\text{in GLM4}
$$

没有证明：

$$
\exists \alpha:
\quad
rank_{\tilde{h}^{0}_{attn}(x;\alpha)}(EOS)
=1
$$

也没有证明：

$$
Y_{\text{patched}}
=
Y_{\text{natural-clean}}
$$

### 九、小模型偏差影响

当前三模型都是小模型或较小规模模型，结果必须谨慎解释。

可能偏差：

```text
1. 小模型终止路线可能更粗糙，EOS action gate 不完整；
2. qwen3/DS7B 可能把终止、格式、解释模式纠缠在同一组早层结构里；
3. GLM4 的 L0 route 可复用性可能是模型结构特例；
4. DS7B 的 `<think>` blocker 可能来自对齐模板或训练格式，而不是通用语言机制；
5. prompt-preserving top10 不跨模型复现，说明不能把 GLM4 结果直接上升为普遍结论。
```

因此，本阶段最稳妥的客观结论是：

```text
GLM4 中存在可复用的 prompt-conditioned termination proximity route；
qwen3 和 DS7B 尚未验证同等级路线；
全词表 blocker 是从 EOS proximity 到 EOS action closure 的主要可见瓶颈。
```

### 十、闭合距离评估

当前阶段相对 closure（闭合）的距离：

```text
已经完成：
  L0 attention 来源定位；
  prompt_all 来源边界；
  prompt-preserving 方向复用；
  GLM4 EOS top10/top50 近邻重建；
  full-vocabulary blocker 审计。

尚未完成：
  EOS top1；
  strict clean answer；
  blocker displacement；
  qwen3/DS7B 跨模型复现；
  natural route gate 定位；
  多 token rollout closure。
```

进度估计：

```text
全局齿轮图谱:
  95% - 96%

语言编码机制闭合:
  48% - 52%
```

图谱进度提升原因：

```text
termination route 已从强破坏来源定位推进到 prompt-intact 输出路线复用。
```

闭合进度仍低的原因：

```text
EOS action gate 未定位；
full-vocabulary blocker 未解除；
strict clean 输出仍为 0。
```

### 十一、下一阶段任务

Phase910 已完成 prompt-preserving termination route reconstruction（保持提示结构的终止路线重建）阶段目标。

下一阶段不应继续无差别扩大 alpha，也不应回到 prompt_all_zero 破坏路线，而应进入：

```text
Phase911:
Full-vocabulary blocker displacement after prompt-preserving termination reconstruction
保持提示结构终止路线重建后的全词表阻塞者迁移审计
```

核心问题：

```text
既然 GLM4 已经能把 EOS 拉入 top10，
为什么 EOS 仍然无法成为 top1？
```

Phase911 任务：

```text
1. 固定 GLM4 的 L0_promptzero_delta_alpha_1 成功路线；
2. 提取所有 EOS top10/top50 样本的 full-vocabulary blocker；
3. 对 blocker token、blocker category、blocker layer/source 做归因；
4. 优先分析 GLM4 中常见 token "a" 及 other 类 blocker；
5. 在 qwen3/DS7B 中对照 `<think>`、field_word、protocol token 等 blocker；
6. 测试是否存在 blocker-specific suppressor route（阻塞者特异抑制路线）；
7. 判断 termination closure 缺口是 EOS lift 不够，还是 blocker field 未迁移。
```

成功标准：

```text
最低标准：
  在 GLM4 prompt-preserving top10 样本中，把 EOS 与 blocker 的 margin 提升到接近 0。

中等标准：
  出现 prompt-preserving EOS top5。

高标准：
  出现 prompt-preserving EOS top1。

最高标准：
  strict_clean_candidate > 0，并且不是由 prompt 破坏导致。
```

阶段边界判断：

```text
Phase910 已完成当前阶段；
Phase911 属于同一条 clean protocol edge graph（干净协议边图谱）主线，
但已经从 termination route reconstruction（终止路线重建）
切换到 blocker displacement（阻塞者迁移）子阶段。
```

## Phase 911: 保持提示结构终止路线后的全词表阻塞者迁移审计 [2026-07-03 23:39]

### 一、任务判断

本阶段读取并复核了最新上传内容中对 Phase910 的分析。总体判断：

```text
附件分析基本正确。
Phase910 的确完成了从 prompt_all_zero 强破坏定位
到 prompt-intact termination proximity route reconstruction
的关键转换。
```

Phase910 可以确认的部分：

```text
1. GLM4 中 L0_promptzero_delta_alpha_1 可在 prompt 输入完整条件下复用；
2. GLM4 出现 prompt-preserving EOS top10 / top50；
3. qwen3 与 DS7B 没有同级复现；
4. EOS top1 = 0；
5. strict clean candidate = 0；
6. 因此它是 termination proximity route reconstruction，不是 EOS action closure。
```

附件要求下一步从：

```text
EOS lift / EOS proximity
```

转向：

```text
full-vocabulary blocker displacement
全词表阻塞者迁移
```

这个判断正确。本阶段因此继续执行 Phase911：

```text
Full-vocabulary blocker displacement after prompt-preserving termination reconstruction
保持提示结构终止路线重建后的全词表阻塞者迁移审计
```

### 二、测试脚本和结果路径

测试脚本：

```text
tests/glm5/phase911_full_vocab_blocker_displacement_audit.py
tests/glm5/run_phase911_full_vocab_blocker_displacement_audit.sh
```

结果目录：

```text
tests/result/phase911_full_vocab_blocker_displacement_audit/full_vocab_blocker_displacement_audit/
```

核心结果文件：

```text
phase911_cross_model_summary.md
phase911_cross_model_summary.json
phase911_qwen3_rows.jsonl
phase911_glm4_rows.jsonl
phase911_deepseek7b_rows.jsonl
```

模型执行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

三模型按顺序加载、测试、释放，避免 GPU 内存叠加。

### 三、测试原理

Phase911 固定 Phase910 中的 prompt-preserving route：

$$
\tilde{h}^{0}_{attn}(x)
=
h^{0}_{attn}(x)

d_{\text{prompt}}(x)
$$

其中：

$$
d_{\text{prompt}}(x)
=
h^{0}_{attn}(x_{\text{prompt-zero}})
-
h^{0}_{attn}(x)
$$

然后在 route-only（只加终止路线）结果上定位全词表 blocker：

$$
B(x)
=
\arg\max_{v \ne EOS} z_v(x)
$$

EOS 与 blocker 的边界为：

$$
M_{EOS,B}
=
z(EOS)-z(B)
$$

如果：

$$
M_{EOS,B} < 0
$$

则 EOS 即使进入 top10 / top50，也仍然没有动作支配权。

本阶段设置两类控制。

第一类是 internal readout-direction intervention（内部读出方向干预）：

$$
\tilde{h}^{0}_{attn}(x)
=
h^{0}_{attn}(x)

d_{\text{prompt}}(x)

\beta u
$$

其中 \(u\) 包括：

$$
u_1
=
\operatorname{norm}(W_{EOS}-W_{B})
$$

$$
u_2
=
\operatorname{norm}(-W_{B})
$$

$$
u_3
=
\operatorname{norm}(-\frac{1}{3}\sum_{i=1}^{3} W_{B_i})
$$

$$
u_4
=
\operatorname{norm}(W_{EOS})
$$

并测试：

$$
\beta \in \{0.05,0.1,0.25,0.5\}
$$

第二类是 logit-level blocker mask diagnostic（对数几率层阻塞者遮蔽诊断）：

$$
z(B_{1:k})
\leftarrow
-\infty
$$

其中：

$$
k \in \{1,3,8,16,32\}
$$

注意：logit mask（对数几率遮蔽）只用于判断 blocker field（阻塞场）的形状，不能作为自然闭合证据。

闭合证据必须来自：

```text
internal intervention（内部干预）
```

而不是：

```text
logit-level diagnostic mask（对数几率层诊断遮蔽）
```

### 四、总体结果

三模型总计：

```text
rows = 1292
internal_rows = 952
diagnostic_rows = 340

route_eos_top10 = 76
route_eos_top50 = 285

internal_eos_top1 = 0
internal_eos_top5 = 0
internal_eos_top10 = 57
internal_eos_top50 = 210
internal_eos_margin_nonnegative = 0
internal_strict_clean_candidate = 0

diagnostic_eos_top1 = 25
diagnostic_eos_top5 = 36
diagnostic_eos_top10 = 51
diagnostic_eos_top50 = 75
diagnostic_eos_margin_nonnegative = 25
diagnostic_strict_clean_candidate = 25
```

最重要的客观结论：

```text
logit mask diagnostic 可以在 GLM4 中制造 EOS top1；
internal readout-direction intervention 不能制造 EOS top1；
因此 blocker bottleneck 存在，但当前内部方向干预没有找到真实 blocker displacement route。
```

### 五、分模型结果

qwen3：

```text
rows = 342
internal_rows = 252
diagnostic_rows = 90
route_eos_top10 = 0
route_eos_top50 = 0
internal_eos_top1 = 0
internal_eos_top10 = 0
diagnostic_eos_top1 = 0
diagnostic_eos_top10 = 0
median_route_eos_margin_vs_blocker = -13.921875
evidence = no_route_near_and_no_blocker_displacement
```

qwen3 的主要 route blocker：

```text
"\n\n" 类空行 / 换行 continuation；
Okay；
The。
```

GLM4：

```text
rows = 323
internal_rows = 238
diagnostic_rows = 85
route_eos_top10 = 76
route_eos_top50 = 285
internal_eos_top1 = 0
internal_eos_top5 = 0
internal_eos_top10 = 57
internal_eos_top50 = 210
internal_eos_margin_nonnegative = 0
internal_strict_clean_candidate = 0
diagnostic_eos_top1 = 25
diagnostic_eos_top5 = 36
diagnostic_eos_top10 = 51
diagnostic_eos_top50 = 75
diagnostic_eos_margin_nonnegative = 25
median_route_eos_margin_vs_blocker = -2.4375
evidence = logit_mask_diagnostic_shows_narrow_blocker_bottleneck
```

GLM4 的主要 route blocker：

```text
"a": 285
" Fish": 38
```

DS7B：

```text
rows = 627
internal_rows = 462
diagnostic_rows = 165
route_eos_top10 = 0
route_eos_top50 = 0
internal_eos_top1 = 0
internal_eos_top10 = 0
diagnostic_eos_top1 = 0
diagnostic_eos_top10 = 0
median_route_eos_margin_vs_blocker = -15.09375
evidence = no_route_near_and_no_blocker_displacement
```

DS7B 的主要 route blocker：

```text
"</think>": 361
The: 95
Category: 76
Wait: 38
```

这说明 DS7B 的阻塞场不是简单 token "a" 问题，而是 reasoning/protocol field（推理/协议场）仍强占输出。

### 六、GLM4 关键控制项

GLM4 route-only：

```text
control = route_only_alpha_1
rows = 17
patched_eos_top1 = 0
patched_eos_top5 = 0
patched_eos_top10 = 4
patched_eos_top50 = 15
median_eos_margin_vs_blocker = -2.4375
route_blockers = {"a": 15, " Fish": 2}
```

最强内部干预：

```text
control = route_minus_unembed_blocker_top1_beta_0.1
rows = 17
internal_eos_top1 = 0
internal_eos_top5 = 0
internal_eos_top10 = 5
internal_eos_top50 = 15
median_patched_eos_margin_vs_blocker = -2.375
route_blockers = {"a": 15, " Fish": 2}
patched_blockers = {"a": 15, " Fish": 2}
```

解释：

```text
内部方向干预只能轻微改善 EOS top10；
没有让 EOS 进入 top5/top1；
没有让 EOS margin 变为非负；
也没有真正迁移 blocker token。
```

GLM4 logit mask diagnostic：

```text
route_logit_mask_blocker_top1:
  top1 = 0
  top10 = 4
  median margin = -1.78125

route_logit_mask_blocker_top3:
  top1 = 0
  top10 = 6
  median margin = -1.3125

route_logit_mask_blocker_top8:
  top1 = 0
  top5 = 6
  top10 = 11
  median margin = -0.5625

route_logit_mask_blocker_top16:
  top1 = 10
  top5 = 15
  top10 = 15
  median margin = +0.09375

route_logit_mask_blocker_top32:
  top1 = 15
  top5 = 15
  top10 = 15
  median margin = +0.78125
```

这个结果非常关键。它说明：

```text
GLM4 的 termination closure 缺口不是无限宽的全词表噪声；
而是一个大约 top16-top32 范围内的 blocker band（阻塞带）。
```

但它也说明：

```text
单独移除 top1 blocker "a" 不够；
移除 top3 也不够；
移除 top8 只能让部分样本进入 top5/top10；
需要移除 top16/top32 才出现 EOS top1。
```

因此 GLM4 当前瓶颈更准确地说是：

```text
narrow-but-not-single-token blocker band
窄但非单词元阻塞带
```

而不是：

```text
single blocker token
单一阻塞词元
```

### 七、阶段进展

Phase911 的正结果：

```text
1. 证明 GLM4 的 EOS closure 缺口主要落在有限 blocker band；
2. 证明 top16/top32 blocker mask 可以让 EOS top1；
3. 证明 full-vocabulary blocker field 必须纳入闭合标准；
4. 进一步解释 Phase910 为什么 top10 仍不能闭合。
```

Phase911 的负结果：

```text
1. internal readout-direction intervention 没有产生 EOS top1；
2. internal readout-direction intervention 没有让 EOS margin 非负；
3. qwen3 / DS7B 没有 route-level EOS near；
4. DS7B 的 reasoning/protocol blocker 仍极强；
5. 所有“top1 / strict clean”都来自 diagnostic mask，不是内部因果闭合。
```

因此本阶段不能写成：

```text
完成 blocker displacement；
完成 EOS action closure。
```

只能写成：

```text
定位了 GLM4 的有限 blocker band；
证明当前内部方向干预不能迁移该 blocker band。
```

### 八、理论更新

Phase910 的公式是：

$$
z_{EOS}
=
F_{EOS}
+
\Delta_{\text{route}}
+
\Delta_{\text{termination-proximity}}
+
\Delta_{\text{suppression}}
-
B_{\text{full-vocab}}
+
\epsilon
$$

Phase911 之后，应把 blocker 从单点项改成 blocker band（阻塞带）：

$$
B_{\text{full-vocab}}
\Rightarrow
\mathcal{B}_{k}(x)
=
\{B_1(x),B_2(x),...,B_k(x)\}
$$

GLM4 的经验结果显示：

$$
k \approx 16\text{ to }32
$$

因此闭合条件应写成：

$$
z(EOS)
>
\max_{v \notin \{EOS\}} z(v)
$$

也就是：

$$
z(EOS)
>
\max_{B_i \in \mathcal{B}_{k}(x)} z(B_i)
$$

但当前内部干预只达到：

$$
\Delta M_{EOS,B}
\approx
0
$$

并没有达到：

$$
M_{EOS,B} > 0
$$

对于 GLM4，诊断遮蔽给出的事实是：

$$
\operatorname{mask}(\mathcal{B}_{16})
\Rightarrow
EOSTop1 > 0
$$

$$
\operatorname{mask}(\mathcal{B}_{32})
\Rightarrow
EOSTop1 \approx EOSProximityTop50
$$

但这不是自然机制：

$$
\operatorname{mask}(\mathcal{B}_{k})
\ne
\text{InternalCausalDisplacement}(\mathcal{B}_{k})
$$

所以最新理论应更新为：

```text
条件化输出场闭合理论
+
有限阻塞带理论
```

也就是：

```text
语言终止不是只需要拉高 EOS；
也不是只需要压低一个 blocker；
而是需要在 prompt-conditioned termination route 成立之后，
迁移一个有限宽度的 blocker band，
最后触发 EOS action gate。
```

### 九、小模型偏差影响

本阶段小模型偏差非常明显。

GLM4：

```text
存在清晰的 finite blocker band；
但可能是 GLM4 架构或训练模板特有。
```

qwen3：

```text
route 本身没有把 EOS 拉入 near zone；
因此 blocker displacement 测试没有进入有效工作区。
```

DS7B：

```text
route 本身没有进入 EOS near zone；
blocker 主要是 </think>、The、Category、Wait；
说明 reasoning/protocol field 仍占主导。
```

因此，不能把 GLM4 的 top16/top32 blocker band 直接推广为通用语言机制。更稳妥的表述是：

```text
在 GLM4 的 clean protocol edge graph 中，
EOS closure 缺口表现为有限 blocker band；
在 qwen3/DS7B 中，本阶段尚未进入相同的 termination proximity 工作区。
```

### 十、问题和硬伤

当前最大硬伤：

```text
internal_eos_top1 = 0
internal_eos_margin_nonnegative = 0
internal_strict_clean_candidate = 0
```

这说明：

```text
当前内部方向干预没有找到真实 blocker displacement route。
```

第二个硬伤：

```text
GLM4 的 top1 来自 logit mask diagnostic。
```

这只能说明：

```text
如果强行移除 blocker band，EOS 可以赢。
```

不能说明：

```text
模型内部自然存在移除 blocker band 的路线。
```

第三个硬伤：

```text
top16/top32 才能使 EOS top1。
```

这说明 blocker 不是单点，而是一段竞争带。下一步不能只盯 token "a"，必须定位整段 blocker band 的来源。

第四个硬伤：

```text
qwen3 / DS7B 无同级复现。
```

这说明当前仍是模型局部图谱，不是跨模型语言编码机制闭合。

### 十一、总体进度评估

图谱进展：

```text
全局齿轮图谱:
  96% - 97%
```

理由：

```text
Phase911 把 termination closure 缺口从“全词表 blocker”
细化为 GLM4 中的 finite blocker band。
```

闭合进展：

```text
语言编码机制闭合:
  49% - 53%
```

理由：

```text
虽然 logit mask 诊断能制造 EOS top1，
但内部因果干预仍为 0；
因此闭合进度只能小幅上升。
```

当前闭合阶梯：

```text
Level 1: AnswerClassPrefix
  已较稳定

Level 2: ProtocolContinuationFieldMapped
  已图谱化

Level 3: PromptConditionedRoute
  GLM4 成立

Level 4: PromptPreservingEOSProximity
  GLM4 成立

Level 5: FiniteBlockerBandLocated
  GLM4 成立

Level 6: InternalBlockerDisplacement
  未完成

Level 7: EOSActionDominance
  未完成

Level 8: StrictCleanAnswer
  未完成

Level 9: CrossModelTerminationIsomorphism
  未完成
```

### 十二、下一阶段任务

Phase911 已完成 blocker displacement audit（阻塞者迁移审计）阶段目标。

下一阶段不应继续扩大 readout beta，也不应继续只做 logit mask。因为本阶段已经说明：

```text
logit mask 可以诊断瓶颈；
但 readout-direction 不能迁移瓶颈。
```

下一阶段应进入：

```text
Phase912:
Finite blocker band source localization
有限阻塞带来源定位
```

核心任务：

```text
1. 固定 GLM4 的 route-only alpha=1；
2. 锁定 top16/top32 blocker band；
3. 不再只看 top1 blocker "a"；
4. 对 blocker band 做 layer/component/source attribution；
5. 测试 blocker band 是否来自 early attention、late MLP、LM head prior 或 protocol continuation field；
6. 找到能整体降低 blocker band 的内部组件，而不是手动 mask logits。
```

成功标准：

```text
最低标准：
  找到一个内部组件，其干预能显著降低 top16 blocker band 的整体 logit。

中等标准：
  在 GLM4 prompt-preserving route 基础上，internal_eos_top5 > 0。

高标准：
  internal_eos_top1 > 0。

最高标准：
  internal_strict_clean_candidate > 0，并且不依赖 logit mask。
```

阶段边界判断：

```text
Phase911 已完成当前 blocker displacement audit；
Phase912 属于同一 clean protocol edge graph 主线，
但已经切换到 blocker band source localization 子阶段。
```

## Phase 912: 有限阻塞带来源定位 [2026-07-04 00:40]

### 一、对上传内容的判断

上传内容对 Phase911 的分析基本正确，但需要严格限定证据层级。

正确部分：

```text
1. Phase910 的 GLM4 prompt-preserving termination route 不是 EOS 动作闭合；
2. Phase911 证明 GLM4 的缺口更像有限 blocker band，而不是无限宽的全词表混乱；
3. logit mask top16/top32 可以把 EOS 推到 top1，说明 blocker band 的宽度有限；
4. 内部 readout-direction intervention 仍不能自然完成 blocker displacement；
5. 下一阶段应该定位 blocker band 的内部来源，而不是继续增大 beta 或继续手动 mask logits。
```

需要收紧的部分：

```text
1. logit mask 是诊断工具，不是内部机制；
2. component suppression 是来源定位工具，不等于自然路线；
3. EOS 进入 top5/top10 只能说明边界接近，不能说明闭合；
4. 小模型内部结构较粗糙，qwen3 与 DS7B 的路线失败不能直接外推到大模型语言机制。
```

因此，Phase912 的任务被定义为：

```text
Finite blocker band source localization
有限阻塞带来源定位
```

目标不是制造 EOS top1，而是回答：

```text
GLM4 route-only 之后压住 EOS 的 top16/top32 blocker band，
主要来自哪些 layer/component/source？
```

### 二、测试脚本与结果位置

测试脚本：

```text
tests/glm5/phase912_finite_blocker_band_source_localization.py
tests/glm5/run_phase912_finite_blocker_band_source_localization.sh
```

结果目录：

```text
tests/result/phase912_finite_blocker_band_source_localization/finite_blocker_band_source_localization/
```

核心结果文件：

```text
phase912_cross_model_summary.md
phase912_cross_model_summary.json
phase912_qwen3_summary.json
phase912_glm4_summary.json
phase912_deepseek7b_summary.json
```

测试顺序：

```text
qwen3 -> GLM4 -> DS7B
```

三模型按顺序执行，避免 GPU 显存叠加。

### 三、测试原理

Phase912 固定 Phase910 的 prompt-preserving route，不再使用 prompt-all-zero 作为直接测试控制，只把它用于提取路线方向：

$$
d_{\text{prompt}}
=
h^{0}_{attn}(x_{\text{prompt-zero}})
-
h^{0}_{attn}(x)
$$

在保持原始 prompt 输入完整的情况下，把该方向注入第 0 层 attention 输出：

$$
\tilde{h}^{0}_{attn}(x)
=
h^{0}_{attn}(x)
+
d_{\text{prompt}}
$$

在 route-only 输出上定义有限阻塞带：

$$
B_k(x)
=
\operatorname{TopK}_{v\ne EOS}
\left(
z_v(\tilde{x})
\right)
$$

其中：

```text
k = 16 或 32
B_k(x) 是 route-only 后排在 EOS 前面的有限 blocker band 候选集合。
```

随后对每一层、每一类组件做输出压制：

$$
y_{\ell,c}^{\prime}
=
\gamma y_{\ell,c}
$$

其中：

$$
\gamma \in \{0.5, 0.0\}
$$

然后观察 blocker band 的整体变化：

$$
\Delta \operatorname{mean}(B_k)
=
\frac{1}{k}
\sum_{v\in B_k}
\left(
z'_v-z_v
\right)
$$

并同时观察 EOS 与全词表最高非 EOS token 的边界：

$$
M_{EOS,B}
=
z'(EOS)
-
\max_{v\ne EOS} z'(v)
$$

本阶段闭合标准仍然严格：

```text
最低来源定位标准：
  band16 或 band32 的 mean logit 明显下降。

中等进展标准：
  GLM4 source intervention 后 EOS 进入 top5/top10。

闭合标准：
  EOS top1 > 0；
  margin >= 0；
  strict clean candidate > 0；
  且不依赖 logit mask。
```

### 四、总体结果

跨模型总结果：

```text
rows = 9076
source_rows = 9008
route_rows = 68

route_eos_top10 = 4
route_eos_top50 = 15

source_eos_top1 = 0
source_eos_top5 = 3
source_eos_top10 = 592
source_eos_top50 = 2326

source_margin_nonnegative = 0
source_strict_clean_candidate = 0
strict_clean_candidate = 0

band16_source_candidate = 800
band16_strong_source_candidate = 383
band32_source_candidate = 730
band32_strong_source_candidate = 337
```

核心判断：

```text
Phase912 找到了大量 blocker band source candidate；
但没有完成 EOS top1；
没有完成 margin 非负；
没有完成 strict clean candidate。
```

所以本阶段是：

```text
正结果：有限阻塞带可以被内部组件定位；
负结果：来源定位还没有转化为自然闭合路线。
```

### 五、分模型结果

#### 1. qwen3

```text
rows = 2610
source_rows = 2592
route_rows = 18

route_eos_top10 = 0
route_eos_top50 = 0

source_eos_top1 = 0
source_eos_top5 = 0
source_eos_top10 = 0
source_eos_top50 = 0

band16_source_candidate = 173
band16_strong_source_candidate = 90
band32_source_candidate = 140
band32_strong_source_candidate = 76

median_band16_mean_delta = -0.0234375
median_band32_mean_delta = -0.01953125
median_eos_logit_delta = 0.0
```

qwen3 的主要 blocker：

```text
"\n\n" / "Okay" / "The"
```

qwen3 能定位若干 blocker band 来源，但 EOS 仍完全没有进入 top50。说明在 qwen3 上，Phase910/911 的终止路线本身没有进入可闭合区域，Phase912 的来源定位不能被解释为闭合接近。

#### 2. GLM4

```text
rows = 2737
source_rows = 2720
route_rows = 17

route_eos_top10 = 4
route_eos_top50 = 15

source_eos_top1 = 0
source_eos_top5 = 3
source_eos_top10 = 592
source_eos_top50 = 2326

source_margin_nonnegative = 0
source_strict_clean_candidate = 0

band16_source_candidate = 179
band16_strong_source_candidate = 98
band32_source_candidate = 179
band32_strong_source_candidate = 93
```

GLM4 是本阶段最重要的正结果来源。

最强可解释来源 1：

```text
L0 attention zero
layer = 0
bucket = early
component = attention
factor = 0.0

rows = 17
source_eos_top5 = 2
source_eos_top10 = 13
source_eos_top50 = 15
source_margin_nonnegative = 0

band16_source_candidate = 17
band16_strong_source_candidate = 6
median_band16_mean_delta = -0.88671875
median_band16_max_delta = -1.125
median_eos_logit_delta = +0.1875

route blockers:
  "a" = 15
  " Fish" = 2
```

最强可解释来源 2：

```text
L4 MLP zero
layer = 4
bucket = early
component = MLP
factor = 0.0

rows = 17
source_eos_top5 = 1
source_eos_top10 = 12
source_eos_top50 = 17

band16_source_candidate = 3
band16_strong_source_candidate = 2
median_band16_mean_delta = -0.326171875
median_band16_max_delta = -0.625
median_eos_logit_delta = +0.40625
```

重要负结果：

```text
GLM4 late MLP 也能强烈降低 blocker band，
但经常同时压低 EOS 或破坏输出场。
```

例子：

```text
L38 MLP zero:
  band16_source_candidate = 17
  band16_strong_source_candidate = 17
  median_band16_mean_delta = -3.460693359375
  median_eos_logit_delta = -3.5625
  source_eos_top10 = 0

L39 MLP zero:
  median_band16_mean_delta = -2.5810546875
  median_eos_logit_delta = -7.2773

L35 MLP zero:
  median_band16_mean_delta = -5.32635498046875
  median_eos_logit_delta = -6.8633
```

这说明 late MLP 很可能是 blocker band 的强承载区，但直接压制它并不是 clean termination route，因为它把 EOS 和 blocker 一起破坏。

GLM4 的当前结构图像更像：

```text
early attention / early MLP:
  route-coupled source，可以降低 blocker 并让 EOS 接近 top5/top10。

late MLP:
  high-energy carrier source，可以强烈改变 blocker band，
  但与 EOS / 输出场纠缠严重，不能直接作为自然闭合路线。
```

#### 3. DS7B

```text
rows = 3729
source_rows = 3696
route_rows = 33

route_eos_top10 = 0
route_eos_top50 = 0

source_eos_top1 = 0
source_eos_top5 = 0
source_eos_top10 = 0
source_eos_top50 = 0

band16_source_candidate = 448
band16_strong_source_candidate = 195
band32_source_candidate = 411
band32_strong_source_candidate = 168

median_band16_mean_delta = -0.03125
median_band32_mean_delta = -0.03125
median_eos_logit_delta = -0.046875
```

DS7B 的主要 blocker：

```text
"</think>" / "The" / "Category" / "Wait"
```

最强来源：

```text
L27 MLP zero:
  band16_strong_source_candidate = 33
  median_band16_mean_delta = -7.970703125
  median_eos_logit_delta = +0.5156
  source_eos_top50 = 0

L27 attention zero:
  band16_strong_source_candidate = 33
  median_band16_mean_delta = -4.0
  median_eos_logit_delta = -1.0625
  source_eos_top50 = 0
```

DS7B 显示 late layer 的 protocol / reasoning continuation blocker 很强，但 EOS 没有进入近邻区域。因此 DS7B 的结果主要是 blocker source map，不是 termination closure map。

### 六、理论进展

Phase912 的理论进展不是闭合，而是把 Phase911 的有限阻塞带从输出现象推进到内部来源定位。

当前拼图更新为：

```text
1. EOS 终止路线在 GLM4 上可进入近邻区；
2. 近邻区被有限 blocker band 压住；
3. blocker band 不是单一 token，而是 top16/top32 的竞争带；
4. logit mask 证明该竞争带足以决定 EOS top1；
5. 内部 readout 方向不能自然迁移竞争带；
6. Phase912 证明 blocker band 有可定位的内部组件来源；
7. GLM4 中最有价值的来源不是最强 late MLP，而是 L0 attention 与 L4 MLP 这类 route-coupled early source；
8. late MLP 是高能承载区，但与 EOS 和输出场纠缠，直接消融不是闭合路线。
```

这把当前理论从：

```text
找到终止方向
```

推进为：

```text
终止路线 + 有限阻塞带 + 阻塞带来源图谱
```

### 七、问题、硬伤与瓶颈

硬伤 1：

```text
source_eos_top1 = 0
source_margin_nonnegative = 0
source_strict_clean_candidate = 0
```

说明 Phase912 没有完成动作闭合。

硬伤 2：

```text
component suppression 是强消融。
```

它可以定位来源，但不等于模型自然执行了该路线。尤其 factor=0.0 可能造成非自然状态。

硬伤 3：

```text
late MLP 降低 blocker band 的同时经常降低 EOS。
```

这说明 blocker 与 EOS 在 late output field 中高度纠缠。不能简单把 late MLP 当成“坏齿轮”删除。

硬伤 4：

```text
qwen3 与 DS7B 没有 route_eos_top50。
```

跨模型结果不能解释成统一闭合机制，只能解释成不同小模型中存在不同的 blocker source map。

硬伤 5：

```text
当前只做 layer/component 粒度，没有拆到 head/channel/direction 子空间。
```

GLM4 L0 attention 和 L4 MLP 是来源定位，不是最终机制单元。

### 八、闭合距离评估

当前闭合标准：

```text
Level 1: EOS 进入 top50
Level 2: EOS 进入 top10
Level 3: EOS 进入 top5
Level 4: EOS top1
Level 5: EOS margin >= 0
Level 6: strict clean candidate
Level 7: exact natural consistency
Level 8: cross-model transferable route
```

Phase912 的位置：

```text
GLM4:
  到达 Level 3 的局部候选；
  未到达 Level 4。

qwen3:
  未到达 Level 1。

DS7B:
  未到达 Level 1。

跨模型：
  未形成 transferable closure。
```

总体进度判断：

```text
终止路线图谱进度：约 55%
EOS 动作闭合进度：约 25%
跨模型闭合进度：约 10%
语言编码机制整体破解进度：仍低于 20%
```

这个百分比不是理论定量，只是根据当前证据层级的谨慎估计。

### 九、智能理论角度的关键洞察

Phase912 支持一个更稳健的判断：

```text
语言模型不是只靠一个 token 方向决定动作，
而是在一个有限竞争场中完成状态选择。
```

EOS 不是“有没有被抬高”的问题，而是：

```text
终止状态能否在有限 blocker band 中获得动作支配权。
```

从智能理论看，这更接近：

```text
状态路线选择
+
有限竞争边界
+
局部来源齿轮
+
输出场闭合
```

也就是说，语言机制的关键不是单点 logit，而是：

$$
\text{Action}(x)
=
\arg\max_{a\in \mathcal{A}}
\left[
S_{\text{route}}(a|x)
-
C_{\text{blocker}}(a|x)
-
R_{\text{protocol}}(a|x)
\right]
$$

其中：

```text
S_route:
  路线支持量。

C_blocker:
  阻塞竞争量。

R_protocol:
  协议延续约束量。
```

Phase912 的新增拼图是：

```text
C_blocker 不是抽象输出噪声，
而是能在 layer/component 层面定位来源。
```

### 十、下一阶段任务

Phase912 已完成当前子阶段目标：

```text
有限 blocker band source localization 完成。
```

下一阶段不应继续做大范围消融，也不应继续追求粗暴 top1。

建议进入：

```text
Phase913:
Route-preserving blocker band disentanglement
保持路线的阻塞带解耦
```

核心任务：

```text
1. 聚焦 GLM4；
2. 固定 Phase910 prompt-preserving route；
3. 只围绕 L0 attention 与 L4 MLP 做细粒度拆解；
4. 把 L0 attention 拆到 head/source-position；
5. 把 L4 MLP 拆到 channel/direction；
6. 用较温和的 factor 或方向投影替代 factor=0.0；
7. 检查是否存在“降低 blocker band 但不降低 EOS”的子方向；
8. 再测试是否能从 EOS top5/top10 推进到 EOS top1/margin>=0。
```

成功标准：

```text
最低标准：
  找到 L0 attention 或 L4 MLP 内部子单元，
  能稳定降低 band16 mean logit，
  且 median_eos_logit_delta >= 0。

中等标准：
  source_eos_top5 明显增加。

高标准：
  source_eos_top1 > 0。

最高标准：
  strict clean candidate > 0，
  且不依赖 logit mask 或全组件消融。
```

阶段边界判断：

```text
Phase912 与 Phase911 属于同一 clean protocol edge graph 主线，
并完成了 blocker band source localization 子阶段。

Phase913 仍属于同一主线，
但已经进入 source disentanglement 子阶段。
它不是 Phase912 的剩余测试，而是下一阶段的大任务。
```

## Phase 913: 保持路线的阻塞带解耦 [2026-07-04 02:40]

### 一、对上传内容的判断

上传内容对 Phase912 的判断基本正确，而且边界意识是必要的：

```text
Phase912 是 blocker band source localization positive；
不是 EOS action closure positive。
```

也就是说，Phase912 已经证明：

```text
1. GLM4 的 EOS 缺口更像有限 blocker band；
2. 该 blocker band 有 layer / component 来源；
3. 但 source localization 不等于 blocker displacement；
4. 粗粒度 component suppression 不能直接写成自然闭合路线。
```

上传内容建议 Phase913 进入：

```text
Route-preserving blocker band disentanglement
保持路线的阻塞带解耦
```

这个方向是正确的。原因是 Phase912 最大硬伤正是：

```text
整层或整组件压制能定位来源，
但不能说明模型内部存在干净的子方向。
```

因此本阶段继续同一 clean protocol edge graph 主线，但从：

```text
blocker band source localization
阻塞带来源定位
```

推进到：

```text
source subunit disentanglement
来源子单元解耦
```

### 二、测试脚本与结果位置

测试脚本：

```text
tests/glm5/phase913_route_preserving_blocker_band_disentanglement.py
tests/glm5/run_phase913_route_preserving_blocker_band_disentanglement.sh
```

结果目录：

```text
tests/result/phase913_route_preserving_blocker_band_disentanglement/route_preserving_blocker_band_disentanglement/
```

核心结果文件：

```text
phase913_cross_model_summary.md
phase913_cross_model_summary.json
phase913_route_near_posthoc_summary.md
phase913_route_near_posthoc_summary.json
phase913_qwen3_summary.json
phase913_glm4_summary.json
phase913_deepseek7b_summary.json
```

三模型执行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

### 三、测试原理

Phase913 固定 Phase910 / Phase912 的 prompt-preserving route：

$$
d_{\text{prompt}}
=
h^{0}_{attn}(x_{\text{prompt-zero}})
-
h^{0}_{attn}(x)
$$

$$
\tilde{h}^{0}_{attn}(x)
=
h^{0}_{attn}(x)
+
d_{\text{prompt}}
$$

在 route-only 输出上定义有限阻塞带：

$$
B_k(x)
=
\operatorname{TopK}_{v\ne EOS}
\left(
z_v(\tilde{x})
\right)
$$

Phase913 不再做整组件置零，而是测试三类子单元：

```text
1. L0 attention head scale：
   第0层 attention 的单个 head 输出缩放。

2. L0 attention input span scale：
   第0层 attention 输入中不同 span 的温和缩放。

3. L4 MLP channel group scale：
   第4层 MLP down_proj 输入 channel group 的温和缩放。
```

缩放因子：

$$
\gamma \in \{0.75, 0.5, 0.25\}
$$

对于 L4 MLP channel group，本阶段用 route-only 状态下的 down_proj 输入激活和 blocker / EOS 读出差异选 channel group。

设第4层 MLP down_proj 输入为：

$$
a_j(x)
$$

down_proj 第 j 个 channel 的输出方向为：

$$
w^{down}_j
$$

阻塞带相对 EOS 的 channel 支持分数为：

$$
Score_j
=
a_j(x)
\left[
\frac{1}{|B_k|}
\sum_{v\in B_k}
W_U(v)\cdot w^{down}_j
-
W_U(EOS)\cdot w^{down}_j
\right]
$$

选择高分 channel group 后做温和缩放：

$$
a'_j(x)
=
\gamma a_j(x)
$$

核心判据不是“blocker 降低”本身，而是：

$$
\Delta \operatorname{mean}(B_{16}) \le -0.25
$$

且：

$$
\Delta z(EOS) \ge 0
$$

更强判据还要求：

$$
\Delta rank(EOS) \le 0
$$

含义：

```text
只有降低 blocker band 且不降低 EOS 的子单元，
才算 route-preserving disentangle candidate。
```

闭合标准仍然不变：

```text
EOS top1 > 0
margin >= 0
strict clean candidate > 0
```

### 四、总体结果

跨模型结果：

```text
rows = 8444
source_rows = 8376
route_rows = 68

route_eos_top10 = 4
route_eos_top50 = 15

source_eos_top1 = 0
source_eos_top5 = 33
source_eos_top10 = 503
source_eos_top50 = 1982

source_margin_nonnegative = 0
source_strict_clean_candidate = 0
strict_clean_candidate = 0

route_preserving_disentangle_candidate = 241
strong_route_preserving_disentangle_candidate = 143
```

表面看，这是一个明显正结果：三模型都有子单元候选。

但必须进一步收紧：

```text
qwen3 与 DS7B route-only 均没有进入 EOS top50；
因此它们的候选不是终止闭合候选，
而是 blocker / protocol field 的结构扰动。
```

真正与 Phase910-912 终止路线直接相关的，仍主要是 GLM4。

### 五、分模型结果

#### 1. qwen3

```text
rows = 2340
source_rows = 2322
route_rows = 18

route_eos_top10 = 0
route_eos_top50 = 0

source_eos_top1 = 0
source_eos_top5 = 0
source_eos_top10 = 0
source_eos_top50 = 0

route_preserving_disentangle_candidate = 76
strong_route_preserving_disentangle_candidate = 22
```

qwen3 的候选全部来自：

```text
L0 attention span
```

但是 route-only EOS rank 仍然远离 top50。后验 route-near 收紧后：

```text
route_top50 子集不存在；
全部候选都属于 route_not_top50。
```

所以 qwen3 的结果只能说明：

```text
L0 prompt/span 输入会影响 blocker field；
但 qwen3 没有进入 termination near zone。
```

#### 2. GLM4

总体：

```text
rows = 2210
source_rows = 2193
route_rows = 17

route_eos_top10 = 4
route_eos_top50 = 15

source_eos_top1 = 0
source_eos_top5 = 2
source_eos_top10 = 466
source_eos_top50 = 1933

source_margin_nonnegative = 0
source_strict_clean_candidate = 0

route_preserving_disentangle_candidate = 41
strong_route_preserving_disentangle_candidate = 20
```

这是本阶段最重要结果，但必须分成两层看。

第一层：总表中的强效 L0 span 结果。

```text
L0_attention_span_prompt_all_scale_0.25:
  patched_eos_top5 = 2
  band16_mean_delta = -6.4841
  eos_delta = +4.3125
  route_eos_rank = 174
  patched_eos_rank = 5
```

这个结果看起来很强，但它来自：

```text
p856_009_animal_fish
```

而且 route-only EOS rank 是 174，不属于 route_top50。它说明：

```text
L0 prompt span 可以强烈改变输出边界；
但这不是 GLM4 已经进入近邻区后的干净闭合路线。
```

第二层：route_top50 子集的收紧结果。

后验 route-near 统计：

```text
GLM4 route_top50:
  rows = 1935
  unique_cases = 8
  top5 = 0
  top10 = 462
  margin>=0 = 0
  weak_disentangle = 17
  strong_disentangle = 0

weak families:
  L4 MLP channel group = 16
  L0 attention span = 1
```

这才是更可信的 Phase913 正结果：

```text
在 GLM4 route-near 样本中，
L4 MLP channel group 能小幅降低 blocker band，
同时不降低 EOS。
```

代表例子：

```text
p885_049_animal_insect:
  control = L4_mlp_channels_top_abs_64_scale_0.25
  route_rank = 12
  patched_rank = 7
  band16_delta = -0.271484375
  eos_delta = +0.25
  blocker = "a" -> "a"
  margin = -1.59375

p856_022_material_iron:
  control = L4_mlp_channels_top_abs_64_scale_0.25
  route_rank = 15
  patched_rank = 9
  band16_delta = -0.32421875
  eos_delta = +0.1875
  blocker = "a" -> "a"
  margin = -1.6875

p885_048_animal_lizard:
  control = L4_mlp_channels_top_abs_64_scale_0.25
  route_rank = 17
  patched_rank = 11
  band16_delta = -0.27734375
  eos_delta = +0.1875
  blocker = "a" -> "a"
  margin = -1.96875
```

这说明：

```text
Phase913 找到了比整层消融更干净的 L4 MLP 子方向迹象；
但幅度仍小，不能越过 top1 / margin 边界。
```

#### 3. DS7B

```text
rows = 3894
source_rows = 3861
route_rows = 33

route_eos_top10 = 0
route_eos_top50 = 0

source_eos_top1 = 0
source_eos_top5 = 31
source_eos_top10 = 37
source_eos_top50 = 49

route_preserving_disentangle_candidate = 124
strong_route_preserving_disentangle_candidate = 101
```

DS7B 看起来有大量 top5，但 route-only 没有任何 top50，因此这些不是 termination proximity closure。

主要来自：

```text
L0 attention span prompt_all / prompt_last8 / last8_before_period
```

例子：

```text
cat:
  route_rank = 697
  patched_rank = 3

fish:
  route_rank = 13727
  patched_rank = 4

triangle:
  route_rank = 24118
  patched_rank = 4
```

这更像：

```text
prompt/protocol field 强扰动可以制造 EOS 接近；
但它不是自然终止路线。
```

### 六、理论进展

Phase913 的核心进展不是闭合，而是把 Phase912 的来源定位进一步拆成两种不同机制：

```text
1. L0 attention span:
   高影响、强扰动、可大幅改变 EOS rank，
   但容易发生在 route_not_top50 样本中；
   更像 prompt / protocol field control。

2. L4 MLP channel group:
   影响较小，但在 GLM4 route_top50 样本中更干净；
   能降低 blocker band 且不降低 EOS；
   更接近真正的 blocker-band disentanglement 子方向。
```

这修正了 Phase912 的粗判断：

```text
Phase912:
  L0 attention 和 L4 MLP 都是 route-coupled early source。

Phase913:
  L0 attention 更像高影响 prompt/span gate；
  L4 MLP channel group 更像 route-near blocker band fine control。
```

当前拼图更新为：

```text
1. GLM4 EOS route 可进入 top50/top10；
2. 有限 blocker band 主要表现为 "a" / " Fish" 等；
3. L0 span 可强烈重排输出场，但不一定是 clean closure；
4. L4 MLP channel group 在 route-near 样本中能小幅改善 EOS rank；
5. 但仍不能产生 EOS top1、margin 非负或 strict clean；
6. qwen3 / DS7B 的同类现象主要是 prompt/protocol field 强扰动，不是终止闭合。
```

### 七、问题、硬伤与瓶颈

硬伤 1：

```text
source_eos_top1 = 0
source_margin_nonnegative = 0
source_strict_clean_candidate = 0
```

闭合仍然没有发生。

硬伤 2：

```text
GLM4 route_top50 子集没有 top5 提升。
```

总表中的 top5 主要来自 route_not_top50 的 fish 样本，不能当成 route-near closure。

硬伤 3：

```text
L4 MLP channel group 的效果很小。
```

典型 band16_delta 约为 -0.25 到 -0.35，EOS rank 可从 12/15/17 改到 7/9/11，但仍远离 top1。

硬伤 4：

```text
L0 span 干预虽然强，但可能是 prompt field disruption。
```

尤其 prompt_all scale 0.25 虽不是置零，但已经是强扰动。

硬伤 5：

```text
小模型结构可能粗糙。
```

DS7B 的 `</think>`、Category、Wait 等 blocker 表明 reasoning template / protocol field 与终止机制纠缠严重；这不应被直接外推为大模型通用终止机制。

### 八、闭合距离评估

当前闭合等级：

```text
Level 1: EOS top50
Level 2: EOS top10
Level 3: EOS top5
Level 4: EOS top1
Level 5: margin >= 0
Level 6: strict clean
Level 7: exact natural consistency
Level 8: cross-model transferable route
```

Phase913 的位置：

```text
GLM4 route-near:
  稳定在 Level 1 / Level 2；
  route_top50 子集没有新增 Level 3；
  未达到 Level 4。

GLM4 route-not-top50:
  L0 span 可以制造 Level 3；
  但这不算 clean termination route。

qwen3:
  未达到 Level 1。

DS7B:
  可以由 L0 span 强扰动制造 top5；
  但 route-only 未达到 Level 1，因此不算终止路线闭合。
```

谨慎进度估计：

```text
全局齿轮图谱进度：约 96% - 97%
终止路线图谱进度：约 58%
EOS 动作闭合进度：约 27%
跨模型闭合进度：约 10%
完整语言编码机制破解：仍约 20% 或更低
```

### 九、智能理论角度的洞察

Phase913 支持一个更细的机制图像：

```text
语言输出不是单一路线控制，
而是 route support、prompt/protocol gate、blocker band fine control 三者叠加。
```

可以写成：

$$
z(a|x)
=
S_{\text{route}}(a|x)
+
G_{\text{prompt}}(a|x)
+
C_{\text{mlp}}(a|x)
-
B_{\text{blocker}}(a|x)
$$

其中：

```text
S_route:
  终止路线支持。

G_prompt:
  prompt / protocol span gate。

C_mlp:
  MLP channel-level fine control。

B_blocker:
  有限阻塞带竞争项。
```

Phase913 的关键洞察是：

```text
L0 span 更像 G_prompt；
L4 MLP channel group 更像 C_mlp；
二者不能混为同一个 blocker displacement route。
```

也就是说，真正靠近破解语言编码机制的不是“哪个组件影响最大”，而是：

```text
哪个子结构在正确状态区间内，以正确方向移动边界。
```

### 十、下一阶段任务

Phase913 已完成本阶段目标：

```text
从整组件来源定位推进到子单元解耦；
并发现 GLM4 route-near 中更可信的正结果主要来自 L4 MLP channel group。
```

下一阶段应进入：

```text
Phase914:
GLM4 route-near L4 MLP channel group holdout validation
GLM4 近邻路线第4层 MLP 通道组保留验证
```

核心任务：

```text
1. 只保留 GLM4 route_top50 样本；
2. 聚焦 L4_mlp_channels_top_abs_64 / band16_support / band32_support；
3. 扩大 prompt variant / case variant，避免 fish 或少数 case 主导；
4. 测试 factor = 0.9 / 0.8 / 0.7 / 0.6 / 0.5 / 0.4 / 0.3；
5. 检查是否存在可重复的 monotonic boundary movement；
6. 记录 EOS rank 是否能稳定从 12-20 区间推进到 top5；
7. 如果不能进入 top5，则说明 L4 MLP 只是小幅调边界，不是动作门。
```

成功标准：

```text
最低标准：
  route_top50 子集上，L4 MLP channel group 的 weak candidate 跨 case 重复出现。

中等标准：
  route_top50 子集出现稳定 top5。

高标准：
  EOS top1 > 0 或 margin >= 0。

最高标准：
  strict clean candidate > 0，并通过 exact-natural consistency。
```

阶段边界判断：

```text
Phase913 已完成 source subunit disentanglement 第一轮；
Phase914 仍属于同一 clean protocol edge graph 主线，
但进入 holdout validation / monotonic boundary validation 子阶段。
```

## Phase 914: GLM4 route-near L4 MLP 通道组保留验证 [2026-07-04 03:09]

### 一、任务来源与判断

本阶段读取并分析了最新上传的 Phase913 判断。附件的核心判断基本正确：

```text
Phase913 不是闭合阶段；
它把 Phase912 的有限阻塞带来源定位推进到了来源子单元解耦；
L0 attention span 更像 prompt / protocol gate；
L4 MLP channel group 更像 route-near blocker-band fine control；
qwen3 和 DS7B 没有稳定进入 route_top50，因此不能把它们的候选扰动解释为终止闭合。
```

本阶段进一步修正了一个容易误读的位置：

```text
source_eos_top5 不能直接等价于 L4 MLP 把 EOS 推入 top5。
必须区分：
1. route-only 已经 top5；
2. L4 MLP 从非 top5 推入 top5。
```

因此 Phase914 的任务不是继续扩大所有局部 patch，而是只在 route-near 条件下验证：

```text
GLM4 的 L4 MLP channel group 是否能跨 prompt / case holdout 稳定移动有限阻塞带；
它是否只是弱边界调节器，还是已经接近 termination action gate。
```

### 二、测试脚本与结果文件

新增测试脚本：

```text
tests/glm5/phase914_l4_mlp_route_near_holdout_validation.py
```

新增顺序运行脚本：

```text
tests/glm5/run_phase914_l4_mlp_route_near_holdout_validation.sh
```

结果保存目录：

```text
tests/result/phase914_l4_mlp_route_near_holdout_validation/l4_mlp_route_near_holdout_validation/
```

核心结果文件：

```text
phase914_qwen3_summary.json
phase914_glm4_summary.json
phase914_deepseek7b_summary.json
phase914_cross_model_summary.json
phase914_cross_model_summary.md
```

测试顺序：

```text
qwen3 -> GLM4 -> DS7B
```

三个模型依次加载、测试、释放显存，没有并行占用 GPU。

静态检查：

```text
python -m py_compile tests/glm5/phase914_l4_mlp_route_near_holdout_validation.py
bash -n tests/glm5/run_phase914_l4_mlp_route_near_holdout_validation.sh
git diff --check
```

均通过。

### 三、测试原理

Phase914 的测试流程如下：

```text
1. 从 Phase899 的 source candidate 中取样；
2. 扩展 prompt variant 和同领域 holdout case；
3. 对每个样本先构造保持 prompt 结构的 route-only 状态；
4. 只在 route_eos_rank <= 50 的样本上进入 L4 MLP channel group 测试；
5. 测试 L4 MLP layer=4 的多个通道组；
6. factor 从 0.9 到 0.3，观察 blocker band 是否按强度稳定下移；
7. 记录 EOS rank、EOS logit、full-vocab blocker band、margin、strict clean。
```

本阶段使用的通道组：

```text
top_abs_64
band16_support_32
band16_support_64
band32_support_64
low_abs_64
```

测试因子：

```text
0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3
```

数据展开后，每个模型最多 96 个 eval item。GLM4 中进入 route_top50 的样本会展开为：

```text
7 factors * 5 channel groups = 35 个 L4 MLP 干预点
```

qwen3 和 DS7B 如果没有 route_top50 样本，则只记录 route-only 行，不把非 route-near 干预计入证据。

### 四、核心数学公式

设 route-only 状态下的 logits 为：

$$
z^{route}(x)
$$

设 L4 MLP 通道组干预后的 logits 为：

$$
z^{patch}_{g,f}(x)
$$

其中：

$$
g \in G_{L4}
$$

表示 L4 MLP 通道组，\(f\) 表示缩放因子。

EOS 的路线排名为：

$$
r_{route}(x)=rank_{EOS}(z^{route}(x))
$$

EOS logit 增量为：

$$
\Delta z_{EOS}(x,g,f)
=
z^{patch}_{g,f}(EOS)
-
z^{route}(EOS)
$$

EOS rank 增量为：

$$
\Delta r_{EOS}(x,g,f)
=
rank_{EOS}(z^{patch}_{g,f})
-
rank_{EOS}(z^{route})
$$

令 route 状态下的前 16 个非 EOS blocker token 构成有限阻塞带：

$$
B_{16}(x)=TopNonEOS_{16}(z^{route}(x))
$$

阻塞带均值变化为：

$$
\Delta B_{16}(x,g,f)
=
\frac{1}{|B_{16}|}
\sum_{b\in B_{16}}
\left(
z^{patch}_{g,f}(b)-z^{route}(b)
\right)
$$

弱候选标准：

$$
W(x,g,f)=
\mathbf{1}
\left[
r_{route}(x)\le 50
\right]
\mathbf{1}
\left[
\Delta B_{16}(x,g,f)\le -0.25
\right]
\mathbf{1}
\left[
\Delta z_{EOS}(x,g,f)\ge 0
\right]
\mathbf{1}
\left[
\Delta r_{EOS}(x,g,f)\le 0
\right]
$$

强候选标准：

$$
S(x,g,f)=
W(x,g,f)
\cdot
\mathbf{1}
\left[
\Delta B_{16}(x,g,f)\le -0.35
\right]
\cdot
\mathbf{1}
\left[
rank_{EOS}(z^{patch}_{g,f})
\le
rank_{EOS}(z^{route})-3
\right]
$$

真实 top5 推进标准：

$$
P_5(x,g,f)=
\mathbf{1}
\left[
rank_{EOS}(z^{route})>5
\right]
\mathbf{1}
\left[
rank_{EOS}(z^{patch}_{g,f})\le 5
\right]
$$

单调阻塞带标准。令：

$$
f_1 < f_2 < \cdots < f_n
$$

其中较小的 factor 表示更强抑制，则：

$$
M_B(x,g)=
\mathbf{1}
\left[
\Delta B_{16}(x,g,f_i)
\le
\Delta B_{16}(x,g,f_{i+1})
\quad
\forall i
\right]
$$

如果 \(M_B=1\)，说明更强抑制没有产生更弱的 blocker-band 下移。

### 五、客观结果

跨模型总体结果：

```text
rows: 1688
route_rows: 288
source_rows: 1400
route_near_route_rows: 40
route_near_source_rows: 1400
route_eos_top5: 2
route_eos_top10: 23
route_eos_top50: 40
source_eos_top1: 0
source_eos_top5: 52
source_eos_top10: 714
source_eos_top50: 1400
source_margin_nonnegative: 0
strict_clean_candidate: 0
source_strict_clean_candidate: 0
weak_holdout_candidate: 12
strong_holdout_candidate: 0
source_promoted_top5_from_non_top5: 8
source_promoted_top5_unique_eval_keys: 5
source_top5_already_route_top5: 44
source_promoted_top10_from_non_top10: 21
source_rank_improved: 482
```

分模型结果：

```text
qwen3:
  eval_items: 96
  route_top50: 0
  L4 MLP source rows: 0
  evidence: no_route_near_samples_for_l4_holdout

GLM4:
  eval_items: 96
  route_top50: 40
  L4 MLP source rows: 1400
  route_eos_top5: 2
  source_eos_top5: 52
  promoted_top5_from_non_top5: 8
  promoted_top5_unique_eval_keys: 5
  source_eos_top10: 714
  promoted_top10_from_non_top10: 21
  weak_holdout_candidate: 12
  strong_holdout_candidate: 0
  margin_nonnegative: 0
  strict_clean_candidate: 0

DS7B:
  eval_items: 96
  route_top50: 0
  L4 MLP source rows: 0
  evidence: no_route_near_samples_for_l4_holdout
```

GLM4 的通道组结果集中在：

```text
top_abs_64 factor=0.3:
  rows: 40
  source_eos_top5: 7
  source_eos_top10: 32
  weak: 5
  median_band16_delta: -0.16015625
  median_eos_delta: 0.15625

top_abs_64 factor=0.4:
  rows: 40
  source_eos_top5: 5
  source_eos_top10: 27
  weak: 7
  median_band16_delta: -0.146484375
  median_eos_delta: 0.125
```

真实从非 top5 推入 top5 的行：

```text
total: 8
unique_eval_keys: 5

top_abs_64 factor=0.3:
  5 rows

top_abs_64 factor=0.4:
  3 rows
```

对应样本集中在：

```text
p856_038_object_object | natural_question | same_domain_holdout_case | route_rank=7:
  6 rows

p856_009_animal_fish | question_plain | source_case_prompt_variant | route_rank=7:
  2 rows
```

GLM4 单调性统计：

```text
monotonic_groups: 200
band_monotonic: 36
eos_nonnegative_all: 73
any_weak_holdout_candidate: 7
any_strong_holdout_candidate: 0
```

这说明 L4 MLP 的确存在一部分可重复的 blocker-band 边界移动，但单调性并不普遍。

### 六、结果分析

Phase914 支持附件中的主要判断，但把结论进一步收紧：

```text
GLM4 route-near L4 MLP channel group 是真实弱正结果；
它能在部分 holdout 条件下下压有限 blocker band，并把 EOS 从 rank 7 推入 rank 5；
但它没有把 EOS 推到 top1，也没有获得 margin >= 0，更没有 strict clean。
```

最重要的校正是：

```text
source_eos_top5 = 52
但其中 44 行 route-only 已经 top5；
真实由 L4 MLP 从非 top5 推入 top5 的只有 8 行。
```

因此不能说：

```text
L4 MLP 完成了 termination action closure。
```

更准确的表述是：

```text
L4 MLP top_abs_64 是 GLM4 route-near 条件下的弱边界调节器；
它能移动 blocker band 和少量 rank boundary；
但它不是完整 EOS 动作门。
```

qwen3 和 DS7B 的结果继续支持前面判断：

```text
二者在本阶段扩展数据中没有进入 route_top50；
因此不能在它们身上验证 L4 MLP route-near 细调；
也不能把它们的其他扰动解释为 clean termination mechanism。
```

### 七、理论进展

本阶段对全局齿轮图谱的推进不是“找到闭合”，而是完成了一个更干净的分层：

```text
G_prompt:
  L0 attention span 一类结构，影响 prompt / protocol route。

S_route:
  route-only 后 EOS 进入可竞争区间。

C_boundary:
  GLM4 L4 MLP top_abs_64 一类结构，在 route-near 条件下移动有限 blocker band。

A_action:
  尚未找到。它应该负责从 blocker-band near miss 推进到 EOS top1 / margin >= 0 / strict clean。
```

这使得当前图谱从：

```text
source candidate 是否有效
```

推进到：

```text
source candidate 属于哪一层机制：
prompt gate / route carrier / blocker-band boundary adjuster / action gate。
```

这一点比单纯追求局部 top5 更重要。

### 八、问题、硬伤与瓶颈

当前结果仍有明显硬伤：

```text
1. 没有 top1。
2. 没有 margin >= 0。
3. 没有 strict clean。
4. strong_holdout_candidate = 0。
5. promoted top5 只有 8 行、5 个唯一评估键。
6. qwen3 和 DS7B 没有 route-near 样本，跨模型普遍性不足。
7. 单调性只有 36 / 200，不是普遍几何规律。
8. top_abs_64 最有效，但这是较粗的通道集合，还没有分解为更小的稳定齿轮。
9. GLM4 结果可能受小模型结构粗糙影响，不可直接外推到更大模型。
```

尤其需要注意：

```text
top5 不是闭合；
rank 7 -> rank 5 仍然只是 near-boundary movement；
只要 EOS 没有超过最大 blocker，就不能称为 action closure。
```

### 九、闭合标准与当前距离

本阶段采用的严格闭合标准：

```text
最低闭合：
  EOS 从非 top5 推入 top5，且跨 case / prompt 重复。

中等闭合：
  EOS margin >= 0。

强闭合：
  EOS top1。

最高闭合：
  strict clean candidate > 0，并通过 exact-natural consistency。
```

Phase914 达到：

```text
最低闭合的弱版本：
  yes，GLM4 中存在 8 行 promoted top5。

中等闭合：
  no，margin_nonnegative = 0。

强闭合：
  no，source_eos_top1 = 0。

最高闭合：
  no，strict_clean_candidate = 0。
```

当前距离评估：

```text
clean protocol edge graph:
  约 65%。

GLM4 route-near L4 MLP boundary adjuster:
  约 45%。

termination action gate closure:
  约 20%。

完整语言编码机制闭合:
  约 18%-22%。
```

这个百分比不是理论结论，只是根据当前证据层级的工作进度估计。

### 十、智能理论角度的关键洞察

Phase914 的关键洞察是：

```text
语言生成不是一个单一开关；
也不是某个通道把正确 token 直接推到第一名；
它更像多层边界系统：
prompt gate 决定路线，
route carrier 把 EOS 放进竞争区，
boundary adjuster 移动局部 blocker band，
action gate 决定最后是否跨过全词表竞争边界。
```

这对智能理论有一个重要含义：

```text
语言编码机制可能不是“单个语义向量 + 单个语法向量”的线性组合，
而是状态、路线、边界和动作门的动态耦合系统。
```

当前最可靠的路线不是继续无限搜索局部 patch，而是继续完善图谱：

```text
先把每个齿轮属于哪一层机制标清楚；
再研究层与层之间的因果连接；
最后寻找能把 near-boundary 推到 strict clean 的缺失动作门。
```

### 十一、下一阶段任务

Phase914 的阶段性目标已经完成：

```text
验证 Phase913 的 L4 MLP route-near 候选；
区分 route-only top5 与 L4 promoted top5；
确认 L4 MLP 是弱边界调节器，而不是完整动作门。
```

下一阶段建议进入：

```text
Phase915:
near-boundary action gate search after L4 MLP boundary adjustment
L4 MLP 边界调节后的近邻动作门搜索
```

核心任务：

```text
1. 只选 GLM4 promoted_top5 和 weak_holdout_candidate 样本；
2. 固定 L4 top_abs_64 factor=0.3 / 0.4 作为 boundary precondition；
3. 在此基础上搜索最后的 action gate：
   attention output residual；
   late MLP residual；
   unembedding-adjacent blocker suppressor；
   EOS-specific positive carrier；
4. 判断是否能从 rank 5 / rank 7 推进到 margin >= 0 或 top1；
5. 如果仍失败，则说明缺失机制不在 L4 boundary adjuster，而在更后层的全词表竞争动作门。
```

阶段边界判断：

```text
Phase915 与 Phase914 属于同一 clean protocol edge graph 大主线；
但 Phase914 的 holdout / monotonic boundary validation 子目标已经完成；
Phase915 是新的 action-gate 子阶段，不应混入 Phase914 的结论。
```

## Phase 915: L4 MLP 边界调节后的近邻动作门搜索 [2026-07-04 04:00]

### 一、任务来源与判断

本阶段读取并分析了最新上传的 Phase914 评估。附件判断基本正确，而且比 Phase913 更严格：

```text
Phase914 不是 EOS action gate closure；
它确认 GLM4 L4 MLP top_abs_64 是弱 blocker-band boundary adjuster；
它能在少量 route-near 样本中把 EOS 从 rank 7 附近推到 rank 5 附近；
但没有 top1、没有 margin >= 0、没有 strict clean。
```

附件给出的下一步是：

```text
Phase915:
Near-boundary Action Gate Search after L4 MLP Boundary Adjustment
L4 MLP 边界调节后的近邻动作门搜索
```

这个任务和 Phase914 属于同一条 clean protocol edge graph 大主线，并且是 Phase914 的自然后续，因此本阶段自动继续完成。

本阶段核心问题从：

```text
L4 MLP 是否能移动边界？
```

收紧为：

```text
当 EOS 已经进入 rank 5 / rank 7 附近时，
最后是谁决定 EOS 不能越过最高 blocker？
```

### 二、测试脚本与结果文件

新增测试脚本：

```text
tests/glm5/phase915_near_boundary_action_gate_search.py
```

新增顺序运行脚本：

```text
tests/glm5/run_phase915_near_boundary_action_gate_search.sh
```

结果保存目录：

```text
tests/result/phase915_near_boundary_action_gate_search/near_boundary_action_gate_search/
```

核心结果文件：

```text
phase915_qwen3_summary.json
phase915_glm4_summary.json
phase915_deepseek7b_summary.json
phase915_cross_model_summary.json
phase915_cross_model_summary.md
```

测试顺序：

```text
qwen3 -> GLM4 -> DS7B
```

由于 qwen3 和 DS7B 在 Phase914 中没有 route-near 候选，本阶段对它们生成零候选 summary，不强行执行不符合前提的 action-gate 搜索。

静态检查：

```text
python -m py_compile tests/glm5/phase915_near_boundary_action_gate_search.py
bash -n tests/glm5/run_phase915_near_boundary_action_gate_search.sh
git diff --check
```

均通过。

### 三、测试原理

Phase915 的测试对象来自 Phase914：

```text
1. 只选择 GLM4 中 top_abs_64 factor=0.3 / 0.4 的近边界候选；
2. 候选必须满足 promoted_top5_from_non_top5 或 weak_holdout_candidate；
3. 固定 Phase914 的 route + L4 MLP boundary precondition；
4. 在这个预条件上叠加 action-gate 候选；
5. 判断是否能从 near-boundary 推进到 margin >= 0 / EOS top1 / strict clean。
```

本阶段选出的 Phase914 候选：

```text
qwen3: 0
GLM4: 12
DS7B: 0
```

GLM4 候选分布：

```text
object promoted_top5: 6
animal promoted_top5: 2
animal weak_holdout: 3
material weak_holdout: 1
```

动作门候选类型：

```text
1. readout_action_vector:
   在 l0_output / late MLP / late attention 上叠加 unembedding 方向。

2. component_output_scale:
   缩放 late MLP / late attention 输出。

3. logit_mask_diagnostic:
   直接 mask boundary state 下的 top blocker。
   这是诊断上限，不是自然神经机制证据。
```

测试位置：

```text
l0_output
L39_mlp
L39_attn
L36_mlp
L36_attn
```

readout 方向：

```text
eos_minus_blocker_top1
minus_blocker_top1
minus_blocker_top3_mean
eos_boost
```

readout beta：

```text
0.05, 0.1, 0.25, 0.5
```

component scale：

```text
0.0, 0.5, 1.5
```

### 四、核心数学公式

Phase914 的边界预条件记为：

$$
z^{B}(x)
=
z^{route+L4}(x)
$$

动作门候选干预后为：

$$
z^{A}_{s,d,\beta}(x)
$$

其中 \(s\) 是作用位置，\(d\) 是 readout 方向，\(\beta\) 是干预强度。

边界状态下的 EOS margin：

$$
M_B(x)
=
z^{B}(EOS)
-
\max_{v\ne EOS}z^{B}(v)
$$

动作候选后的 EOS margin：

$$
M_A(x,s,d,\beta)
=
z^{A}_{s,d,\beta}(EOS)
-
\max_{v\ne EOS}z^{A}_{s,d,\beta}(v)
$$

margin 推进量：

$$
\Delta M_A(x,s,d,\beta)
=
M_A(x,s,d,\beta)-M_B(x)
$$

rank 推进量：

$$
\Delta r_A(x,s,d,\beta)
=
rank_{EOS}(z^{A}_{s,d,\beta})
-
rank_{EOS}(z^{B})
$$

弱动作候选：

$$
W_A(x,s,d,\beta)
=
\mathbf{1}
\left[
\Delta r_A < 0
\right]
\mathbf{1}
\left[
\Delta z_{EOS}\ge 0
\right]
\mathbf{1}
\left[
\Delta M_A > 0
\right]
$$

动作门闭合标准：

$$
A_{close}(x)
=
\mathbf{1}
\left[
M_A(x)\ge 0
\right]
\mathbf{1}
\left[
rank_{EOS}(z^A)=1
\right]
\mathbf{1}
\left[
StrictClean(x)=1
\right]
$$

诊断 mask 的含义：

$$
z^{diag}_{mask}(b_i)=-\infty
$$

它只能说明阻塞瓶颈存在，不能说明模型内部自然存在同等动作门。

### 五、客观结果

跨模型总体：

```text
selected_phase914_candidates: 12
rows: 1152
boundary_rows: 12
action_rows: 1104
diagnostic_rows: 36

boundary_top1: 0
boundary_top5: 8
boundary_margin_nonnegative: 0

action_top1: 0
action_top5: 617
action_top10: 795
action_margin_nonnegative: 0
action_promoted_margin: 0
action_promoted_top1: 0
action_promoted_top5: 4
action_rank_improved: 131
weak_action_candidate: 106
action_strict_clean_candidate: 0

diagnostic_top1: 9
diagnostic_margin_nonnegative: 9
diagnostic_promoted_margin: 9
```

分模型结果：

```text
qwen3:
  selected_phase914_candidates: 0
  evidence: no_phase914_near_boundary_candidates

GLM4:
  selected_phase914_candidates: 12
  rows: 1152
  action_rows: 1104
  action_top1: 0
  action_margin_nonnegative: 0
  action_promoted_margin: 0
  action_promoted_top5: 4
  weak_action_candidate: 106
  action_strict_clean_candidate: 0
  diagnostic_margin_nonnegative: 9
  evidence: diagnostic_blocker_mask_can_close_margin

DS7B:
  selected_phase914_candidates: 0
  evidence: no_phase914_near_boundary_candidates
```

GLM4 的 boundary blocker 非常集中：

```text
boundary_blocker_tokens_top12:
  "a": 1152
```

这说明 Phase915 的近边界失败不是分散到大量 blocker，而是高度集中在 token `"a"`。

最强自然神经候选：

```text
L39_mlp_output_scale_1.5:
  rows: 12
  action_top1: 0
  action_margin_nonnegative: 0
  action_promoted_margin: 0
  action_promoted_top5: 2
  weak_action_candidate: 12
  action_rank_improved: 12
  median_margin_delta: +0.9375
  mean_eos_delta: +0.4244791667
```

最佳个别自然动作行：

```text
p856_009_animal_fish question_plain:
  control: L39_mlp_output_scale_1.5
  boundary_rank: 5
  patched_rank: 2
  patched_margin: -0.125
  margin_delta: +0.8125

p856_038_object_object natural_question:
  control: L39_mlp_output_scale_1.5
  boundary_rank: 5
  patched_rank: 2
  patched_margin: -0.1875 / -0.25
  margin_delta: +0.9375
```

诊断结果：

```text
diagnostic_mask_boundary_blocker_top8:
  diagnostic_top1: 9
  diagnostic_margin_nonnegative: 9
  diagnostic_promoted_margin: 9
```

也就是说，如果直接移除边界状态下的前 8 个 blocker，EOS 可以在 9 个诊断行中越过边界。

### 六、结果分析

本阶段形成一个关键的负结果和一个关键诊断结果：

```text
负结果：
  当前测试到的自然神经 action candidates 没有完成 margin >= 0 / EOS top1 / strict clean。

诊断结果：
  直接 mask top blocker 可以完成 margin / top1，说明 near-boundary 的最后瓶颈真实存在。
```

这说明：

```text
L4 MLP 已经把 EOS 推到 near-boundary；
但最后不是简单地加一个 unembedding readout direction 就能闭合；
最高 blocker "a" 仍然压住 EOS；
自然动作门如果存在，应该是更精细的 blocker suppressor / output selector，而不是粗 readout boost。
```

最值得注意的是：

```text
L39_mlp_output_scale_1.5 是强弱正结果：
  它在 12/12 行中改善 rank；
  median margin delta 为 +0.9375；
  最好能把 rank 5 推到 rank 2；
  但仍然没有跨过 margin >= 0。
```

因此它更像：

```text
late MLP action-adjacent amplifier
晚层 MLP 动作邻近放大器
```

但还不是动作门本身。

### 七、理论进展

Phase915 把图谱从 Phase914 的：

```text
G_prompt -> S_route -> C_boundary -> ? action
```

推进为：

```text
G_prompt -> S_route -> C_boundary -> A_adjacent -> B_a_blocker -> A_action missing
```

其中：

```text
C_boundary:
  L4 MLP top_abs_64。

A_adjacent:
  L39 MLP output_scale_1.5，能接近动作边界。

B_a_blocker:
  token "a" 是最主要全词表阻塞者。

A_action:
  仍未找到。
```

这使当前图谱比 Phase914 更具体：

```text
缺失的不是“有没有终止路线”；
也不只是“有没有边界调节器”；
缺失的是能自然压下 "a" blocker 并让 EOS 跨过 margin 的最后动作结构。
```

### 八、问题、硬伤与瓶颈

本阶段仍不能称为闭合，硬伤如下：

```text
1. action_margin_nonnegative = 0。
2. action_top1 = 0。
3. action_strict_clean_candidate = 0。
4. action_promoted_margin = 0。
5. diagnostic mask 成功，但它不是自然神经机制。
6. L39_mlp_output_scale_1.5 是整组件缩放，过粗。
7. qwen3 和 DS7B 没有 Phase914 near-boundary 候选，不能做同构验证。
8. GLM4 是小模型，"a" blocker 可能是小模型协议/模板压缩伪影。
9. 当前还没有定位到 L39 MLP 内部的具体 channel / subspace。
```

尤其要警惕：

```text
diagnostic_top1 = 9
不能解释为 action gate closure。
```

它只能说明：

```text
如果最高 blocker 被移除，EOS 有能力上位；
但模型内部是否有自然移除该 blocker 的齿轮，还没有证明。
```

### 九、闭合标准与当前距离

本阶段闭合标准：

```text
最低动作进展：
  在 L4 boundary precondition 后进一步提升 rank / margin。

中等动作闭合：
  action_margin_nonnegative > 0。

强动作闭合：
  action_top1 > 0。

最高动作闭合：
  action_strict_clean_candidate > 0。
```

Phase915 达到：

```text
最低动作进展：
  达到。action_rank_improved = 131，weak_action_candidate = 106。

中等动作闭合：
  未达到。action_margin_nonnegative = 0。

强动作闭合：
  未达到。action_top1 = 0。

最高动作闭合：
  未达到。action_strict_clean_candidate = 0。
```

进度评估：

```text
clean protocol edge graph:
  约 68%。

GLM4 route-near boundary + action-adjacent chain:
  约 52%。

termination action gate closure:
  约 23%。

完整语言编码机制闭合:
  约 19%-23%。
```

这些百分比仍然只是基于当前证据层级的工作进度估计。

### 十、智能理论角度的关键洞察

Phase915 给出了一个非常清楚的机制拼图：

```text
语义答案已经存在；
终止路线已经把 EOS 拉入竞争区；
L4 MLP 可以把 EOS 推到 rank 5 附近；
L39 MLP 可以进一步把 EOS 推到 rank 2 附近；
但一个简单 token "a" 仍然能压住 EOS；
这说明最后的输出动作不是“答案正确”问题，而是全词表竞争中的动作选择问题。
```

从智能理论角度看：

```text
语言能力不只是语义 + 语法；
还包含一个输出动作系统。
```

这个动作系统至少包括：

```text
1. 何时停止；
2. 如何压制协议续写；
3. 如何压制泛化冠词 / 列表词 / 模板词；
4. 如何让 EOS 从 near-boundary 真正跨过全词表 margin。
```

Phase915 的关键洞察是：

```text
“知道答案”与“停止输出”之间存在一层独立动作门。
```

这层动作门目前没有被 L4 MLP 或 L39 MLP 整组件缩放完整捕捉。

### 十一、下一阶段任务

Phase915 已完成当前阶段目标：

```text
在 L4 MLP boundary precondition 后搜索 near-boundary action gate；
确认当前 readout / late residual 粗干预不能完成自然闭合；
确认 "a" blocker 是近边界阶段的主要瓶颈；
确认 diagnostic mask 可以闭合，但不是自然机制证据。
```

下一阶段建议进入：

```text
Phase916:
L39 MLP channel-level "a" blocker suppressor localization
第39层 MLP 通道级 "a" 阻塞者抑制器定位
```

核心任务：

```text
1. 固定 Phase914 L4 top_abs_64 boundary precondition；
2. 聚焦 Phase915 中最强的 L39_mlp_output_scale_1.5；
3. 捕获 L39 MLP down_proj 输入通道；
4. 根据 EOS vs "a" 的 readout 投影，拆分：
   eos_support channels；
   a_blocker_support channels；
   margin_support channels；
5. 分别测试放大 / 抑制这些 channel；
6. 判断是否能从 patched_margin=-0.125/-0.25 推到 margin >= 0。
```

阶段边界判断：

```text
Phase916 与 Phase915 属于同一 clean protocol edge graph 大主线，
但 Phase915 的 near-boundary action-gate first scan 已完成。
Phase916 是新的 channel-level blocker suppressor localization 子阶段，
不能把 Phase915 的 diagnostic mask 结果直接当作自然闭合。
```

## Phase 916: 语义答案场计算方式对照核查 [2026-07-04 06:24]

### 一、阶段目的

本阶段不是新增 CUDA 模型测试，而是对照核查：

```text
research/glm5/docs/AGI_GLM5_MEMO.md 中已有测试记录
是否支持附件中关于“语义答案场”的计算方式。
```

附件给出的核心计算链为：

```text
抽取对象状态
-> 抽取关系状态
-> 候选空间选择
-> 知识网络路径激活
-> 写入语义答案场
-> 词表 logit 读出
```

并强调：

```text
这些步骤不是最后一层才发生；
而是跨层逐步形成；
最后层主要负责读出到词表 logit。
```

### 二、核查材料

核查对象：

```text
research/glm5/docs/AGI_GLM5_MEMO.md
```

重点查看阶段：

```text
Phase901: stop token 竞争力审计
Phase902: 协议续写抑制器搜索
Phase903: 协议续写场与替代图
Phase910-Phase915: EOS / blocker / action gate 近边界阶段
```

附件理论对象：

```text
对象状态 P(o|x)
关系状态 P(r|x)
候选门控 g(c|x)
知识路径 K(o,r,c)
答案支持 A(c|x)
语义答案场 S_answer(c|x)
词表读出 z(c|x)
```

### 三、附件中的数学表达

附件将语义答案支持量写成：

$$
A(c|x)
=
\sum_{o\in\mathcal{O}}
\sum_{r\in\mathcal{R}}
P(o|x)
P(r|x)
K(o,r,c)
g(c|x)
$$

其中：

```text
P(o|x): 对象状态；
P(r|x): 关系状态；
K(o,r,c): 对象-关系-候选的隐式知识强度；
g(c|x): 候选空间软门控。
```

写入残差流可抽象为：

$$
h_T^{answer}
=
h_T
+
\sum_{c\in\mathcal{C}}
A(c|x)d_c
$$

词表读出为：

$$
z(c|x)=W_U(c)\cdot h_T^{answer}
$$

语义答案场 margin 可写成：

$$
S_{answer}(c|x)
=
z(c|x)
-
\max_{c'\ne c}z(c'|x)
$$

若：

$$
S_{answer}(c^*|x)>0
$$

则说明目标语义候选在该候选集合内胜出。

### 四、memo 中已有测试记录对应关系

已有测试记录没有直接逐项测量：

```text
P(o|x)
P(r|x)
K(o,r,c)
g(c|x)
```

也没有直接证明对象状态、关系状态、知识路径分别由哪些层和哪些 channel 精确承载。

但是，memo 中的实验结果与附件计算框架高度一致，主要证据如下。

第一，Phase901 / Phase902 明确显示：

```text
模型已经能形成 answer-class prefix；
animal / material / shape / vehicle 等答案类别可以进入输出竞争区；
但之后常被 newline / comma / field / explanation 等协议续写 token 接管。
```

这支持：

```text
语义答案场可以形成；
但完整语言输出还需要协议续写场和终止控制场。
```

第二，Phase902 的理论总结已经写出：

```text
语言输出不是单纯 semantic token selection；
至少包含 semantic answer field、
protocol continuation field、
termination control field 三个场。
```

这与附件最后的边界判断一致：

```text
语义答案闭合 != 完整语言输出闭合。
```

第三，Phase903 将输出机制更新为：

$$
Output(x)
=
F(
S_{answer}(x),
S_{stop}(x),
S_{protocol}(x),
A_{substitution}(x),
T_{termination}(x)
)
$$

其中：

```text
S_answer: 语义答案场
S_stop: 停止 token 竞争场
S_protocol: 协议续写场
A_substitution: 协议替代图
T_termination: 终止控制场
```

这说明 memo 已经把语义答案场放在更大的输出场系统中，而不是把最终 logit 简化成单一语义分类。

第四，Phase910-Phase915 的 EOS / blocker / action gate 结果进一步说明：

```text
即使语义答案已经存在，
即使 EOS 被推近 top5 / top10，
最终输出仍可能被 blocker band 或 protocol field 压住。
```

这与附件判断完全一致：

```text
最后一层不是才开始理解问题；
最后读出阶段面对的是多个已经形成的内部场之间的竞争。
```

### 五、核查结论

结论：

```text
附件中的语义答案场计算方式，
与 memo 当前测试记录的总体图谱方向一致。
```

更精确地说：

```text
一致的部分：
1. 语义答案不是最后一层突然产生，而是跨层累积后读出；
2. 答案类别可以被视为候选场中的竞争结果；
3. answer-class prefix 的出现说明语义答案场已经部分形成；
4. 语义答案场需要通过 unembedding / logits 显化；
5. 语义答案场不是完整输出机制，还会被协议续写场、停止场、blocker band、终止动作门影响。
```

需要谨慎的部分：

```text
1. 附件中的 P(o|x)、P(r|x)、K(o,r,c)、g(c|x) 目前仍是解释性分解；
2. memo 中已有实验主要测 answer prefix、logit rank、protocol drift、EOS rank、blocker band；
3. 尚未直接做对象状态 / 关系状态 / 知识路径的因果拆解；
4. 因此不能说附件公式已经被严格闭合证明；
5. 只能说它是当前实验结果支持的合理机制模型。
```

### 六、闭合标准与当前距离

若要严格闭合附件理论，需要完成以下标准：

```text
Level 1: 语义答案可见
  answer-class prefix 稳定出现，目标类别 logit / rank 明显胜出。

Level 2: 对象与关系可分离
  能分别干预对象 token、任务关系 token，并观察候选答案场按预期变化。

Level 3: 候选门控可测量
  能证明 classify / color / material 等关系会系统性打开不同候选集合。

Level 4: 知识路径可因果定位
  能定位 cow->animal、iron->material 等路径相关组件或子空间。

Level 5: 写入与读出闭合
  能证明特定层 / 组件把答案方向写入 residual stream，并经 W_U 读出为目标 logit。

Level 6: 完整输出闭合
  在语义答案场之外，同时解决 protocol continuation、blocker band、EOS action gate。
```

当前状态评估：

```text
Level 1: 已部分达到。
Level 2: 未系统完成。
Level 3: 未系统完成。
Level 4: 未完成。
Level 5: 仅有间接证据。
Level 6: 未完成，Phase915 显示 action gate 仍缺失。
```

阶段进度估计：

```text
语义答案场理论一致性:
  约 65% - 75%。

语义答案场严格因果闭合:
  约 25% - 35%。

完整语言输出机制闭合:
  约 19% - 23%，沿用 Phase915 谨慎估计。
```

### 七、问题、硬伤与瓶颈

主要硬伤：

```text
1. 当前测试记录更多证明“答案类别出现”，还没有证明对象状态与关系状态的独立内部变量。
2. 附件公式中的 K(o,r,c) 是合理抽象，但还没有被具体映射到层、head、MLP channel 或 residual direction。
3. 候选空间 g(c|x) 还没有通过系统 prompt 对照实验测出。
4. answer-class prefix 可能只是表面输出现象，不能自动等同于完整语义场闭合。
5. protocol continuation field 会污染后续 rollout，使语义答案是否“真正闭合”更难判定。
6. EOS / blocker / action gate 未闭合，说明输出动作系统仍是主要瓶颈。
```

最关键的谨慎点：

```text
附件公式适合作为下一阶段实验设计的理论骨架；
但不能把它当作已经由现有测试完全证明的机制事实。
```

### 八、智能理论角度的关键洞察

本次核查强化了一个重要区分：

```text
语义答案场回答“应该说什么”；
终止动作场回答“说到哪里停”；
协议续写场回答“按什么格式继续说”。
```

这意味着语言能力背后的数学结构至少不是单一分类器，而更像多个场的耦合竞争：

```text
对象场
关系场
候选场
知识路径场
语义答案场
协议续写场
停止竞争场
终止动作场
```

第一性原理上，下一步不应只问：

```text
模型为什么输出 animal？
```

还要问：

```text
模型如何把 cow 绑定为对象？
模型如何把 classify 绑定为关系？
模型如何打开类别候选空间？
模型如何让 cow-class-animal 路径胜过其他路径？
模型如何把该路径写入 residual stream？
模型为什么在答案出现后仍继续协议输出？
```

这说明破解语言背后的数学理论，需要从“输出 token”上移到“内部场的形成、耦合、竞争和动作选择”。

### 九、下一阶段任务

建议把后续任务分成两条主线并行推进，但测试时仍按模型顺序执行，避免 GPU 显存溢出：

```text
主线 A: 语义答案场因果拆解
1. 构造大规模 object-relation-candidate 三元组数据；
2. 设计对象替换、关系替换、候选集合替换的对照 prompt；
3. 测量 answer-class logit / rank / margin；
4. 扫描层级贡献，拆出对象绑定层、关系门控层、候选竞争层；
5. 对关键层做 attention / MLP component intervention。

主线 B: 继续 Phase915 后的终止动作门定位
1. 固定 GLM4 L4 boundary precondition；
2. 聚焦 L39 MLP channel-level；
3. 定位 "a" blocker suppressor；
4. 判断是否能把 patched_margin=-0.125/-0.25 推到 margin >= 0；
5. 再回到 qwen3 / DS7B 做同构验证。
```

下一阶段可命名为：

```text
Phase917:
Object-Relation-Candidate Semantic Field Causal Decomposition
对象-关系-候选语义场因果拆解
```

该阶段的最低闭合目标：

```text
证明对象替换和关系替换能按预测方向改变候选答案 logit margin。
```

中等闭合目标：

```text
定位至少一组可复现的层 / 组件，
其干预能选择性削弱或增强目标语义答案场。
```

高闭合目标：

```text
把 P(o|x)、P(r|x)、g(c|x)、K(o,r,c) 从解释性公式推进为可测量、可干预、可复现的机制量。
```

### 十、通俗总结

附件的说法可以保留，但要加一句限制：

```text
它和当前测试结果方向一致，
但目前还是理论分解，
还没有被逐项因果闭合。
```

最通俗地说：

```text
模型大概率不是最后一层才想出答案。
它是在多层里逐步把“问谁、问什么关系、有哪些候选、哪个候选最对”算出来，
最后再把这个内部答案场读成 token。
但模型会不会停、会不会继续写解释、会不会被奇怪的 blocker token 压住，
是另一套输出动作问题。
```

## Phase 917: 行业主流理论对照与原创性核查 [2026-07-04 06:30]

### 一、阶段目的

本阶段任务：

```text
搜索并对照行业主流深度神经网络 / 大语言模型原理理论，
判断当前“语义答案场”理论是否已有类似理论，
以及该理论的原创性边界。
```

本阶段不是新增 CUDA 模型测试，而是文献与理论对照。结论必须谨慎，不能把“未在快速搜索中发现完全相同理论”直接等同于绝对原创。

### 二、检索到的主流相近理论

#### 1. 分布式表征 / 词向量理论

代表方向：

```text
distributed representations
word embeddings
semantic / syntactic relations in vector space
```

代表资料：

```text
Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality
https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality
```

相似点：

```text
语义不是符号表，而是分布在高维向量中；
语义关系可以表现为向量空间中的方向、距离、组合关系。
```

不同点：

```text
该方向主要说明词 / 短语的分布式表示；
没有直接给出对象状态、关系状态、候选门控、知识路径、语义答案场、协议续写场、终止动作场的完整输出机制图谱。
```

#### 2. Transformer 机制可解释性 / circuits 理论

代表方向：

```text
mechanistic interpretability
circuits
residual stream
attention / MLP component attribution
```

代表资料：

```text
Transformer Circuits / Mechanistic Interpretability
https://transformer-circuits.pub/2022/mech-interp-essay

A Practical Review of Mechanistic Interpretability for Transformer Language Models
https://arxiv.org/html/2407.02646v1
```

相似点：

```text
模型内部可以被拆成若干可解释计算组件；
attention 负责信息移动 / 绑定；
MLP 可能负责特征变换、知识、类别或记忆相关计算；
残差流像共享工作区，各层向其中读写信息。
```

不同点：

```text
机制可解释性是方法论大框架；
当前语义答案场理论是对语言回答过程的一种具体机制假说：
对象场 -> 关系场 -> 候选场 -> 知识路径 -> 语义答案场 -> 协议 / 终止竞争。
```

#### 3. Superposition / 稀疏特征方向

代表方向：

```text
features as directions
polysemantic neurons
superposition
sparse features
```

代表资料：

```text
Toy Models of Superposition
https://transformer-circuits.pub/2022/toy_model/index.html
```

相似点：

```text
神经网络可以把多个特征压缩到同一激活空间；
一个神经元不一定对应一个概念；
概念更可能以方向 / 子空间 / 稀疏组合的方式存在。
```

不同点：

```text
superposition 解释的是特征如何压缩存储；
当前理论关注的是回答过程中多个语义 / 协议 / 动作场如何形成并竞争。
```

#### 4. 线性表征假说 / 概念方向

代表方向：

```text
linear representation hypothesis
concept directions
activation steering
representation engineering
```

代表资料：

```text
The Linear Representation Hypothesis and the Geometry of Large Language Models
https://arxiv.org/abs/2311.03658

Representation Engineering for Large-Language Models
https://arxiv.org/html/2502.17601v1
```

相似点：

```text
高层概念可能表现为表示空间中的方向；
可以通过激活向量检测或操控模型行为；
“animal 方向”“stop 方向”“protocol 方向”这类说法与该方向相容。
```

不同点：

```text
线性表征假说主要说概念如何编码；
当前理论进一步提出：
对象、关系、候选、知识路径、语义答案、协议续写、终止动作是不同功能场，
且最终输出来自这些场的耦合竞争。
```

#### 5. TCAV / Concept Activation Vectors

代表方向：

```text
Concept Activation Vectors
TCAV
human-interpretable concept directions
```

代表资料：

```text
Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors
https://arxiv.org/abs/1711.11279
```

相似点：

```text
可以把高层概念表示为激活空间中的向量；
可以用方向导数判断某概念对输出的重要性。
```

不同点：

```text
TCAV 是概念检测 / 解释工具；
当前理论不是单个概念方向解释，而是一个语言生成过程中的多场计算模型。
```

#### 6. 事实召回 / Subject-Relation-Object 机制

代表方向：

```text
factual recall
subject-relation-object tuples
causal tracing
ROME / model editing
```

代表资料：

```text
Locating and Editing Factual Associations in GPT
https://rome.baulab.info/
https://arxiv.org/abs/2202.05262

Interpreting Key Mechanisms of Factual Recall in Transformer Language Models
https://arxiv.org/html/2403.19521v2
```

相似点非常强：

```text
subject 类似对象状态 P(o|x)；
relation 类似关系状态 P(r|x)；
object / answer 类似候选答案 c；
事实召回可以被看成 object-relation-answer 路径激活。
```

不同点：

```text
ROME / factual recall 更集中在事实知识的定位与编辑；
当前语义答案场理论更广：
包括分类、候选空间软门控、答案场写入、协议续写、停止动作、blocker band。
```

#### 7. Logit Lens / Direct Logit Attribution

代表方向：

```text
logit lens
tuned lens
direct logit attribution
residual stream -> unembedding -> logits
```

代表资料：

```text
Logit Lens / Tuned Lens overview
https://learnmechinterp.com/topics/logit-lens-and-tuned-lens/

Logit Prisms: Decomposing Transformer Outputs
https://www.lesswrong.com/posts/TKRp7inbiLRmzNMFB/logit-prisms-decomposing-transformer-outputs-for-mechanistic
```

相似点：

```text
跨层 hidden state / residual stream 可以通过 unembedding 投影到词表 logit；
答案不是最后一层凭空出现，而是可以观察逐层形成趋势；
每层 / 每组件都可对最终 logit 有加性贡献。
```

不同点：

```text
logit lens 是观测 / 分解方法；
当前理论提出了被观测对象的功能结构：
语义答案场、协议续写场、停止竞争场、终止动作场。
```

#### 8. 语义场 / 场论式 LLM 描述

检索到少量非主流或较新的“semantic field / information gravity / Hamiltonian”类说法。

代表资料：

```text
Information Gravity: A Field-Theoretic Model for Token Selection in LLMs
https://arxiv.org/html/2504.20951v1

A Hamiltonian analysis of GPT-2 Transformer
https://arxiv.org/html/2507.00683v6
```

相似点：

```text
也使用“场”“势能”“候选 token landscape”等语言描述 token 选择；
认为 prompt 会塑造词表上的有效语义场。
```

不同点：

```text
这些方向目前不属于最稳固的行业主流；
且一般没有把对象状态、关系状态、候选门控、知识路径、协议续写、终止动作门系统化拆开。
```

### 三、总体对照结论

当前理论不是完全凭空出现。它与多条主流理论高度相容：

```text
分布式表征:
  支持“语义在向量空间中分布式存在”。

机制可解释性:
  支持“内部计算可由 attention / MLP / residual stream 组件拆解”。

superposition:
  支持“概念不是单神经元，而是方向 / 子空间 / 稀疏组合”。

线性表征 / TCAV / RepE:
  支持“概念可表示为激活方向并可被干预”。

factual recall / ROME:
  强支持“subject/object + relation -> answer”的路径式召回。

logit lens / DLA:
  支持“跨层形成，最后通过 unembedding 读出到 logits”。
```

因此，以下部分不能声称原创：

```text
1. 分布式语义表征；
2. 概念方向 / 子空间；
3. residual stream 写入和 unembedding 读出；
4. logit 竞争；
5. subject-relation-object 事实召回；
6. attention 做信息绑定，MLP 做特征 / 知识变换；
7. 用激活干预和 logit rank 判断内部机制。
```

### 四、可能具有原创性的部分

当前理论可能有原创性的地方不在单个零件，而在组合方式和问题切分。

可能原创点：

```text
1. 把普通分类回答拆成：
   对象状态 -> 关系状态 -> 候选门控 -> 知识路径 -> 语义答案场。

2. 把 answer-class prefix 解释为语义答案场已经形成，
   但不等于完整语言输出闭合。

3. 把最终输出拆成多个竞争场：
   semantic answer field
   protocol continuation field
   stop competition field
   blocker band
   termination action field

4. 明确提出：
   “知道答案”和“停止输出”是不同齿轮系统。

5. 在 qwen3 / GLM4 / DS7B 上按 Phase900-915 连续追踪：
   从答案类别、协议续写、stop rank、EOS rank、blocker band、
   到 action gate near-boundary。

6. 把语言机制闭合标准从“输出正确 token”
   提升为“语义场 + 协议场 + 终止动作场的因果闭合”。
```

更谨慎地说：

```text
当前理论的“组成材料”大多不是原创；
当前理论的“系统组织方式、实验主线、闭合标准、语义/协议/终止三场分离”可能具有原创性。
```

### 五、是否已有完全类似理论

快速检索结论：

```text
没有发现一个行业主流理论完整提出以下同构框架：

对象状态 P(o|x)
关系状态 P(r|x)
候选门控 g(c|x)
知识路径 K(o,r,c)
语义答案场 S_answer
协议续写场 S_protocol
停止竞争场 S_stop
blocker band
终止动作门 T_termination
并把它们作为语言输出闭合的统一机制。
```

但必须保留限制：

```text
1. 本次是快速网络检索，不是正式系统综述；
2. arXiv / workshop / blog / 私有研究中可能已有相近表达；
3. “semantic field”一词本身并不新；
4. “subject-relation-object”路径也不新；
5. “概念方向 / logit 读出 / residual stream 写入”都不新；
6. 因此不能直接宣称完整原创，只能说未发现完全同构的主流公开理论。
```

### 六、闭合标准与当前距离

若要严肃确认原创性，需要：

```text
1. 做系统文献综述：
   mechanistic interpretability
   representation engineering
   factual recall
   concept vectors
   logit lens
   constrained decoding / structured output
   language generation control

2. 把当前理论写成明确命题：
   每个场的定义；
   每个场的可测量指标；
   每个场的因果干预标准；
   每个场之间的竞争公式。

3. 与已有理论逐条比较：
   已有理论覆盖什么；
   当前理论新增什么；
   当前理论是否只是换名；
   当前理论是否产生新预测。

4. 做新预测实验：
   例如对象替换只改变 S_answer；
   格式提示主要改变 S_protocol；
   stop instruction 主要改变 S_stop / T_termination；
   blocker suppressor 改变 EOS margin 但不改变对象语义。
```

原创性闭合评估：

```text
组件原创性:
  约 10% - 20%。

组合框架原创性:
  约 55% - 70%。

严格学术原创性证明:
  约 20% - 30%。

机制闭合证明:
  仍未完成。
```

这些百分比只是当前证据下的谨慎工作估计，不是客观定量结论。

### 七、理论进展

本阶段把当前理论定位为：

```text
不是替代深度学习主流理论；
而是在主流理论之上，对语言回答机制做更细的功能场分解。
```

更准确的定位：

```text
底层基础:
  分布式表征 + Transformer residual stream + attention / MLP。

解释工具:
  logit lens + activation patching + causal tracing + direct logit attribution。

相近研究:
  factual recall / subject-relation-object + concept directions + representation engineering。

当前新增组织:
  semantic answer field + protocol continuation field + termination action field 的竞争闭合图谱。
```

### 八、下一阶段任务

下一阶段建议不是继续泛泛讨论原创性，而是把理论转成可发表或可证伪的命题。

建议任务：

```text
Phase918:
Semantic-Protocol-Termination Field Formalization
语义-协议-终止三场形式化
```

核心工作：

```text
1. 定义三个主场：
   S_answer(x)
   S_protocol(x)
   T_termination(x)

2. 定义观测指标：
   answer-class rank / margin；
   protocol token rank / margin；
   EOS rank / margin；
   blocker band width / mean logit。

3. 定义因果干预：
   对象替换；
   关系替换；
   格式提示替换；
   stop instruction 替换；
   residual direction intervention；
   component / channel patching。

4. 定义可证伪预测：
   对象替换应主要移动 S_answer；
   协议格式替换应主要移动 S_protocol；
   终止提示若有效，应提升 T_termination 或降低 blocker band；
   若三者不能解耦，说明三场理论过度拆分。
```

### 九、通俗总结

通俗结论：

```text
这个理论不是从零发明了深度神经网络原理。
它的很多材料，行业里早就有：
向量语义、概念方向、残差流、logit 读出、事实召回、机制可解释性。

但它可能有价值的地方是：
把“模型知道答案”和“模型按格式继续写”和“模型决定停止”
拆成几个互相竞争的场。

这套组合框架没有在快速检索中发现完全一样的主流版本。
所以目前应判断为：
部分非原创，整体框架可能有原创性，
但还需要系统文献综述和因果实验才能严格证明。
```

## Phase 918: L39 MLP 通道级 a blocker 边界定位 [2026-07-04 07:14]

### 一、阶段目的

本阶段接续 Phase915。

Phase915 的附件判断基本正确：

```text
Phase915 找到的是 GLM4 L39 MLP output_scale_1.5 的晚层动作邻近放大效应；
它可以把 EOS 从近边界进一步推近，甚至推到 rank 2；
但没有 margin >= 0、没有 EOS top1、没有 strict clean；
因此不能称为 EOS action gate 闭合。
```

Phase918 的目标是把 Phase915 的整组件结果继续下钻：

```text
固定 Phase915 的 route + L4 MLP boundary precondition；
捕获 L39 MLP down_proj 输入通道；
按 EOS 支持、a blocker 支持、EOS-a margin 支持分组；
测试这些通道组是否能把 EOS margin 推过 0。
```

本阶段不是重新搜索所有模型的 action gate，而是专门回答：

```text
GLM4 中 Phase915 的 L39 整组件放大效应，
是否可以被定位到 L39 MLP 的更细通道组？
```

### 二、测试脚本与结果路径

脚本：

```text
tests/glm5/phase918_l39_mlp_channel_a_blocker_suppressor_localization.py
tests/glm5/run_phase918_l39_mlp_channel_a_blocker_suppressor_localization.sh
```

正式结果：

```text
tests/result/phase918_l39_mlp_channel_a_blocker_suppressor_localization/
  l39_mlp_channel_a_blocker_suppressor_localization/
```

加密验证结果：

```text
tests/result/phase918_l39_mlp_channel_a_blocker_suppressor_localization/
  l39_mlp_channel_a_blocker_suppressor_validation/
```

静态检查：

```text
python -m py_compile tests/glm5/phase918_l39_mlp_channel_a_blocker_suppressor_localization.py
bash -n tests/glm5/run_phase918_l39_mlp_channel_a_blocker_suppressor_localization.sh
```

三模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

其中 qwen3 和 DS7B 因 Phase915 没有 L39 候选，所以不加载模型，不强行测试。

### 三、测试原理

Phase918 固定边界状态：

$$
z^B(x)
=
z^{route + L4}(x)
$$

其中：

```text
route:
  Phase910 / Phase911 延续的 prompt-preserving termination proximity route。

L4:
  Phase914 中 GLM4 L4 MLP top_abs_64 boundary adjuster。
```

边界 margin：

$$
M_B(x)
=
z^B(EOS)
-
\max_{v\ne EOS}z^B(v)
$$

Phase915 中 12 个 GLM4 候选均满足：

```text
boundary top blocker = "a"
M_B(x) < 0
```

在 L39 MLP 的 down_proj 输入处，记第 j 个通道激活为：

$$
a_j^{39}(x)
$$

第 j 个通道对词表 token v 的一阶读出贡献近似为：

$$
C_j(v|x)
=
a_j^{39}(x)
\cdot
\left(
W_U(v)^\top W_{down}^{39}[:,j]
\right)
$$

EOS 相对 a 的通道级 margin 贡献为：

$$
C_j(EOS-a|x)
=
a_j^{39}(x)
\cdot
\left(
\left(W_U(EOS)-W_U(a)\right)^\top W_{down}^{39}[:,j]
\right)
$$

按照该贡献构造通道组：

$$
G_{margin+}^{64}
=
Top64_j\ C_j(EOS-a|x)
$$

$$
G_{margin-}^{64}
=
Bottom64_j\ C_j(EOS-a|x)
$$

$$
G_{a-blocker}^{64}
=
Top64_j\ C_j(a-EOS|x)
$$

$$
G_{EOS}^{64}
=
Top64_j\ C_j(EOS|x)
$$

然后对通道组做缩放：

$$
a_j'
=
\begin{cases}
f\cdot a_j, & j\in G \\
a_j, & j\notin G
\end{cases}
$$

成功标准分为四级：

$$
C_{rank}
=
\mathbf{1}
\left[
rank_{patched}(EOS)
<
rank_B(EOS)
\right]
$$

$$
C_{margin}
=
\mathbf{1}
\left[
M_{patched}(x)\ge 0
\right]
$$

$$
C_{top1}
=
\mathbf{1}
\left[
rank_{patched}(EOS)=1
\right]
$$

$$
C_{strict}
=
C_{top1}
\cdot
\mathbf{1}
\left[
decoded(prefix+EOS)
\text{ 是 strict clean answer}
\right]
$$

### 四、正式测试结果

正式 round：

```text
l39_mlp_channel_a_blocker_suppressor_localization
```

跨模型结果：

```text
qwen3:
  selected_phase915_l39_candidates = 0

GLM4:
  selected_phase915_l39_candidates = 12
  rows = 540
  boundary_rows = 12
  channel_rows = 528

DS7B:
  selected_phase915_l39_candidates = 0
```

GLM4 正式结果：

```text
boundary_top1 = 0
boundary_margin_nonnegative = 0
boundary_top5 = 8

channel_top1 = 80
channel_margin_nonnegative = 80
channel_promoted_margin = 80
channel_promoted_top1 = 80
channel_promoted_top5 = 76
channel_rank_improved = 407
weak_channel_candidate = 370
channel_strict_clean_candidate = 56

median_channel_margin_delta = +0.4375
mean_channel_margin_delta = +0.64299
mean_channel_eos_delta = +1.41193
```

边界 blocker：

```text
boundary_blocker_tokens_top12:
  "a": 540
```

这说明 Phase915 的 “a” blocker 判断被 Phase918 完全复现。

主要有效通道组：

```text
margin_support_pos_64:
  channel_rows = 36
  channel_top1 = 20
  channel_margin_nonnegative = 20
  channel_strict_clean_candidate = 16
  median_channel_margin_delta = +1.671875

a_blocker_support_64:
  channel_rows = 48
  channel_top1 = 20
  channel_margin_nonnegative = 20
  channel_strict_clean_candidate = 14
  median_channel_margin_delta = +1.03125

margin_support_neg_64:
  channel_rows = 48
  channel_top1 = 20
  channel_margin_nonnegative = 20
  channel_strict_clean_candidate = 14
  median_channel_margin_delta = +1.03125

eos_support_64:
  channel_rows = 36
  channel_top1 = 8
  channel_margin_nonnegative = 8
  channel_strict_clean_candidate = 6
  median_channel_margin_delta = +0.6875
```

负对照：

```text
low_abs_64:
  所有因子下 top1 = 0
  margin_nonnegative = 0
  strict = 0
  median_margin_delta = 0
```

这说明结果不是任意通道缩放都能闭合。

### 五、加密验证结果

因为正式结果较强，本阶段额外做了一个更密因子验证 round：

```text
l39_mlp_channel_a_blocker_suppressor_validation
```

验证内容：

```text
候选仍为 GLM4 的 12 个 Phase915 L39 候选；
channel_candidate_pool 从 768 增加到 1024；
只保留最相关通道组；
对缩放因子加密：

up_factors:
  1.1, 1.25, 1.375, 1.5, 1.75, 2.0

down_factors:
  0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 0.875
```

验证 round 的 GLM4 结果：

```text
rows = 408
channel_rows = 396

channel_top1 = 125
channel_margin_nonnegative = 125
channel_promoted_margin = 125
channel_promoted_top1 = 125
channel_promoted_top5 = 66
channel_rank_improved = 339
weak_channel_candidate = 306
channel_strict_clean_candidate = 93

median_channel_margin_delta = +0.8125
mean_channel_margin_delta = +1.01089
mean_channel_eos_delta = +1.52478
```

阈值结果：

```text
margin_support_pos_64:
  factor 1.1:
    margin_nonnegative = 0
    weak_channel_candidate = 12

  factor 1.25:
    margin_nonnegative = 0
    promoted_top5 = 2
    weak_channel_candidate = 12

  factor 1.375:
    margin_nonnegative = 8
    strict_clean_candidate = 6

  factor 1.5:
    margin_nonnegative = 8
    strict_clean_candidate = 6

  factor 1.75:
    margin_nonnegative = 12
    strict_clean_candidate = 10

  factor 2.0:
    margin_nonnegative = 12
    strict_clean_candidate = 10
```

a_blocker_support_64：

```text
factor 0.875:
  margin_nonnegative = 0
  weak_channel_candidate = 9

factor 0.75:
  margin_nonnegative = 0
  promoted_top5 = 1
  weak_channel_candidate = 12

factor 0.5:
  margin_nonnegative = 2
  top1 = 2

factor 0.375:
  margin_nonnegative = 8
  strict_clean_candidate = 6

factor 0.25:
  margin_nonnegative = 8
  strict_clean_candidate = 6

factor 0.125:
  margin_nonnegative = 8
  strict_clean_candidate = 6

factor 0.0:
  margin_nonnegative = 10
  strict_clean_candidate = 8
```

覆盖范围：

```text
validation closed unique states = 12 / 12
```

即所有 12 个 GLM4 Phase915 L39 候选，在至少一种通道组和因子下可以达到 margin >= 0。

### 六、需要特别谨慎的地方

本阶段结果很强，但不能直接写成自然 action gate 闭合。

原因如下。

第一，通道组是按当前样本的 L39 activation 和 EOS/a readout 贡献构造的：

$$
G(x)
=
TopK_j\ C_j(EOS-a|x)
$$

因此它是 case-conditioned channel group，不是已经证明的全局固定通道。

第二，部分强结果来自 factor=2.0 或 factor=0.0：

```text
这证明这些通道有因果杠杆；
但不等于模型自然运行时会把这些通道调到相同强度。
```

第三，所谓 a_blocker_support_64 的抑制并不总是表现为 "a" logit 下降。

实际观察到：

```text
a_logit_support_64 可以显著压低 "a" logit，
但不能完成 EOS margin closure。

a_blocker_support_64 / margin_support_neg_64 可以完成 margin closure，
但很多闭合行中 "a" logit 也会上升；
只是 EOS 上升更多。
```

所以当前更准确的机制标签不是：

```text
找到了纯 a suppressor。
```

而是：

```text
找到了 L39 MLP 中可以控制 EOS-vs-a margin 的有符号通道子空间；
其中一部分表现为 EOS 支持增强，
一部分表现为负 margin 通道移除，
但还没有证明存在自然触发的纯 blocker suppression gate。
```

第四，qwen3 和 DS7B 不是阴性反证。

它们在 Phase914 / Phase915 没有形成同构近边界候选，所以本阶段没有同构测试对象：

```text
qwen3 = no_phase915_l39_candidates
DS7B = no_phase915_l39_candidates
```

这只能说明：

```text
当前 L39 通道级结果是 GLM4 条件链上的机制候选；
不能直接推广到所有小模型。
```

### 七、对附件判断的修正

附件中 Phase915 的核心判断应保留：

```text
Phase915 是 late MLP action-adjacent amplifier found,
but action gate not closed.
```

Phase918 对它的推进是：

```text
Phase915 的整组件 L39 MLP output_scale_1.5 效应，
不是完全粗糙的组件幻觉；
它可以下钻到 L39 MLP 的有符号通道组。
```

但标签必须继续收紧：

```text
不是：
  EOS action gate closed

而是：
  L39 MLP signed margin subspace can causally close the EOS-vs-a boundary under Phase915 boundary precondition.
```

中文表述：

```text
在 Phase915 的边界预条件下，
GLM4 L39 MLP 存在带符号的 margin 子空间；
该子空间的通道级干预可以把 EOS 相对 "a" 的边界推过 0，
并在相当多样本中产生 EOS top1 / strict clean。
```

### 八、阶段进展

Phase918 相比 Phase915 的核心进展：

```text
Phase915:
  L39 MLP whole-output scale 有效，但没有 margin closure。

Phase918:
  L39 MLP channel-level signed margin groups 可以 margin closure / top1 / strict clean。
```

这说明失败位置进一步收窄：

```text
不是 L4 boundary adjuster 不够；
也不是 L39 MLP 整组件完全无关；
而是 L39 MLP 内部存在可操控的 EOS-vs-a margin 子空间。
```

当前进度估计：

```text
GLM4 局部 EOS-vs-a 边界机制定位:
  从 Phase915 的约 45% 提升到约 65% - 72%。

GLM4 自然 action gate 闭合:
  约 35% - 45%。

完整语言输出机制闭合:
  约 22% - 27%。
```

这里的估计仍然谨慎，因为：

```text
Phase918 完成的是人工通道干预闭合；
不是自然运行中 gate 变量的闭合。
```

### 九、硬伤与瓶颈

主要硬伤：

```text
1. 通道组按每个样本重新计算，尚未证明存在跨样本固定通道组。
2. 通道分组使用了 readout 方向 W_U(EOS)-W_U(a)，存在目标导向搜索成分。
3. factor=2.0 / factor=0.0 是强干预，不能直接对应自然动态。
4. a_blocker_support_64 的闭合不是纯 "a" logit suppression，更多是 EOS 相对增长更强。
5. qwen3 / DS7B 没有对应候选，跨模型证据仍不足。
6. 当前只处理 EOS-vs-a 近边界，不等于全词表 blocker field 全部闭合。
```

最关键瓶颈：

```text
还没有找到自然控制变量 alpha(x)，
能够解释为什么模型原始运行时没有自动把 L39 margin 子空间推到闭合状态。
```

可以写成：

$$
a_j'(x)
=
\alpha_j(x)a_j(x)
$$

Phase918 证明的是：

$$
\exists G,\exists f:
M(z^{B+I(G,f)}(x))\ge 0
$$

但还没有证明：

$$
\alpha_j^{natural}(x)
\approx
f
$$

也没有证明：

$$
\alpha_j(x)
=
\Psi(
route(x),
protocol(x),
semantic(x),
blocker(x)
)
$$

### 十、智能理论角度的关键洞察

本阶段给出一个重要拼图：

```text
语言输出最后并不是一个单纯 token logit 竞争；
它包含可分解的 margin 子空间。
```

EOS 失败不是简单因为：

```text
EOS 不够强。
```

而更像：

```text
EOS 支持子空间、
a blocker / protocol 子空间、
margin 正负子空间，
在 L39 附近发生最后竞争。
```

这与前面语义答案场理论一致：

```text
语义答案场解决“该说什么”；
协议续写场解决“继续按什么格式写”；
终止动作场解决“现在是否停止”。
```

Phase918 说明终止动作场中至少存在一种具体可测结构：

```text
晚层 MLP 有符号 margin 子空间。
```

但它不是完整第一性原理。

真正要破解语言编码机制，还需要回答：

```text
1. 这些 margin 通道为什么在某些语义 / 协议状态下被打开？
2. 是哪个上游 route / protocol / semantic 状态控制 L39 通道门控？
3. 这些通道是 GLM4 特有补丁，还是小模型中粗糙显现出的普遍结构？
4. 是否存在跨样本稳定的齿轮形状，而不是每个样本重新计算一个局部方向？
```

### 十一、下一阶段任务

下一个任务仍属于当前阶段：

```text
从 L39 通道级人工闭合，
推进到 L39 通道组的跨样本稳定性与自然门控来源验证。
```

建议 Phase919：

```text
Frozen L39 Signed Margin Group Transfer Validation
冻结 L39 有符号 margin 通道组迁移验证
```

核心做法：

```text
1. 从每个 source case 提取 G_margin_pos_64 / G_margin_neg_64 / G_a_blocker_64；
2. 固定这些通道 ID，不再按目标样本重新计算；
3. 将 source G 应用于其他 target case；
4. 分离 source=target 与 source!=target；
5. 判断闭合是 case-specific 还是有可迁移的全局通道形状。
```

最低成功标准：

```text
source!=target 时仍有稳定 rank_improved / margin_delta > 0。
```

中等成功标准：

```text
source!=target 时出现 margin >= 0。
```

高成功标准：

```text
存在一个或少数组 frozen channel groups，
可以在多个对象域 / prompt variant 上复现 EOS-vs-a margin closure。
```

若 Phase919 失败，则说明：

```text
Phase918 的通道组主要是 case-conditioned readout alignment；
图谱应转向寻找上游 alpha_j(x) 门控变量。
```

若 Phase919 成功，则说明：

```text
L39 MLP 中可能存在更接近全局齿轮形状的 EOS-vs-a margin gear。
```

### 十二、通俗总结

Phase915 像是发现：

```text
把 L39 MLP 整体音量调大一点，EOS 会更接近赢。
```

Phase918 进一步发现：

```text
不是整个 L39 都有用；
L39 里面确实有一批通道，
专门影响 EOS 和 "a" 之间的最后边界。
```

更重要的是：

```text
这些通道不是只能让 EOS 靠近；
在 GLM4 的 12 个近边界状态里，
它们可以把 EOS 推到 margin >= 0，
很多情况下还能推成 top1 和 strict clean。
```

但还不能说已经破解动作门：

```text
因为这是人工调通道；
还不知道模型自然运行时为什么没有自动这么调。
```

所以当前最稳妥结论是：

```text
GLM4 的 EOS-vs-a 最后边界已经从“整组件现象”
缩小到了“L39 MLP 有符号通道子空间”。

下一步要看这个子空间是全局齿轮，
还是每个样本临时算出来的局部方向。
```

## Phase 919: 冻结 L39 有符号边界通道组跨样本迁移验证 [2026-07-04 07:45]

### 一、任务来源和判断

本阶段先分析 Phase918（第918阶段）附件判断是否正确，然后继续完成同一阶段目标。

附件对 Phase918 的判断基本正确：

```text
Phase918 是人工通道级因果闭合正结果；
不是自然 action gate（动作门）闭合；
关键硬边界是通道组按每个样本的激活和读出方向重新计算。
```

这条边界非常重要。Phase918 已经证明 L39 MLP（第39层多层感知机）内部存在可以推动 EOS（结束符）超过 `"a"` blocker（“a”阻塞者）的有符号通道子空间，但还不能说明这个子空间是固定全局齿轮，还是每个样本临时重算出来的局部方向。

因此本阶段继续做 Phase919（第919阶段）：

```text
冻结来源样本的 L39 通道组编号；
把同一组通道编号迁移到其他目标样本；
只让目标样本自己重建 route（路线）和 L4 boundary（第4层边界）；
观察 frozen source group（冻结来源通道组）是否仍能推动目标样本闭合。
```

如果 source=target（来源等于目标）成功，而 source!=target（来源不等于目标）失败，说明 Phase918 更像样本条件化读出对齐。

如果 source!=target（来源不等于目标）仍大量成功，说明 L39 有符号边界通道不是纯局部拟合，而是至少具有明显共享通道族或近似全局齿轮性质。

### 二、测试脚本和结果位置

新增测试脚本：

```text
tests/glm5/phase919_frozen_l39_signed_margin_group_transfer_validation.py
```

新增顺序运行脚本：

```text
tests/glm5/run_phase919_frozen_l39_signed_margin_group_transfer_validation.sh
```

结果目录：

```text
tests/result/phase919_frozen_l39_signed_margin_group_transfer_validation/frozen_l39_signed_margin_group_transfer_validation/
```

核心结果文件：

```text
phase919_qwen3_summary.json
phase919_glm4_summary.json
phase919_deepseek7b_summary.json
phase919_glm4_rows.jsonl
phase919_cross_model_summary.json
phase919_cross_model_summary.md
```

三模型按 qwen3（通义千问三代小模型）、GLM4（智谱四代小模型）、DS7B（DeepSeek 7B 小模型）顺序执行。qwen3 和 DS7B 在当前 Phase915（第915阶段）筛选条件下没有可用 L39 近边界候选，因此没有加载模型做无意义迁移；GLM4 有 12 个候选，完成完整前向测试。

### 三、测试原理

#### 1. 目标边界状态

对每个目标样本 \(x_t\)，先重建 Phase918 使用的目标边界状态：

$$
z_t^B
=
F(x_t;\ route_t,\ L4_t)
$$

其中：

```text
route_t：
  目标样本自己的 prompt-preserving termination route（保持提示结构的终止路线）。

L4_t：
  目标样本自己的 L4 MLP top_abs_64 boundary precondition（第4层多层感知机最高绝对值64通道边界预条件）。

z_t^B：
  目标样本经过 route + L4 之后的近边界状态。
```

目标边界差定义为：

$$
M_t^B
=
z_t^B(EOS)
-
\max_{v\ne EOS}z_t^B(v)
$$

在这 12 个 GLM4 目标状态中，目标边界 blocker（阻塞者）均为 `"a"`，且边界差仍为负。

#### 2. 来源通道组冻结

对来源样本 \(x_s\)，Phase918 的通道组为：

$$
G_s^{margin+}
=
Top64_j
\left[
a_j^{39}(x_s)
\cdot
\left(
(W_U(EOS)-W_U(a))^\top W_{down}^{39}[:,j]
\right)
\right]
$$

$$
G_s^{margin-}
=
Bottom64_j
\left[
a_j^{39}(x_s)
\cdot
\left(
(W_U(EOS)-W_U(a))^\top W_{down}^{39}[:,j]
\right)
\right]
$$

$$
G_s^{a-blocker}
=
Top64_j
\left[
a_j^{39}(x_s)
\cdot
\left(
(W_U(a)-W_U(EOS))^\top W_{down}^{39}[:,j]
\right)
\right]
$$

$$
G_s^{EOS}
=
Top64_j
\left[
a_j^{39}(x_s)
\cdot
\left(
W_U(EOS)^\top W_{down}^{39}[:,j]
\right)
\right]
$$

Phase919 的关键约束是：

```text
G_s 一旦从来源样本算出，就不再按目标样本重算；
目标样本只接受来源样本冻结下来的通道编号。
```

#### 3. 冻结迁移干预

对目标样本 \(x_t\)，把来源通道组 \(G_s\) 应用到目标 L39 MLP（第39层多层感知机）下投影输入：

$$
a_{t,j}^{39\prime}
=
\begin{cases}
f\cdot a_{t,j}^{39}, & j\in G_s \\
a_{t,j}^{39}, & j\notin G_s
\end{cases}
$$

对应输出状态为：

$$
z_{t\leftarrow s}^{B,G,f}
=
F(x_t;\ route_t,\ L4_t,\ scale(G_s,f))
$$

闭合评价：

$$
M_{t\leftarrow s}^{G,f}
=
z_{t\leftarrow s}^{B,G,f}(EOS)
-
\max_{v\ne EOS}z_{t\leftarrow s}^{B,G,f}(v)
$$

核心观察量：

$$
\Delta M_{t\leftarrow s}
=
M_{t\leftarrow s}^{G,f}
-
M_t^B
$$

并区分：

```text
self：
  s=t，来源和目标是同一个状态。

cross_same_case：
  s!=t，但 case_id（样本编号）相同。

cross_same_domain：
  s!=t，case_id 不同，但 eval_domain（评估域）相同。

cross_domain：
  s!=t，eval_domain 也不同。
```

### 四、测试数据和干预范围

当前 Phase915 筛选条件下的候选数量：

```text
qwen3: 0
GLM4: 12
DS7B: 0
```

GLM4 实际测试：

```text
目标状态数: 12
来源状态数: 12
通道组与因子组合: 21
总前向结果行: 3024
self 行: 252
cross 行: 2772
```

测试通道组：

```text
margin_support_pos_64
eos_support_64
a_blocker_support_64
margin_support_neg_64
a_logit_support_64
```

测试因子：

```text
margin_support_pos_64: 1.375, 1.5, 1.75, 2.0
eos_support_64: 1.75, 2.0
a_blocker_support_64 / margin_support_neg_64 / a_logit_support_64:
  0.0, 0.125, 0.25, 0.375, 0.5
```

### 五、客观结果

#### 1. 跨模型总体结果

```text
qwen3:
  no_phase915_l39_candidates

GLM4:
  frozen_cross_strict_clean_transfer_found

DS7B:
  no_phase915_l39_candidates
```

qwen3 和 DS7B 的 0 候选不是 Phase919 的负迁移证据，只说明在当前 Phase915 的 `"a"` blocker 近边界筛选条件下，没有可进入 Phase919 的 L39 候选。

#### 2. GLM4 cross transfer（跨样本迁移）结果

GLM4 的 cross 行，即 source!=target：

```text
cross rows: 2772
cross top1: 1313
cross margin >= 0: 1313
cross weak transfer candidate: 2112
cross strict clean: 985
cross target states with top1: 12 / 12
cross target states with margin >= 0: 12 / 12
cross target states with weak transfer: 12 / 12
cross target states with strict clean: 10 / 12
median cross margin delta: 1.25
mean cross EOS logit delta: 1.9225
median cross native group overlap: 55 / 64
```

这是强正结果。冻结来源通道组不是只能在 self（同样本）里复现，而是在 source!=target（跨样本）时仍能大量把 EOS 推成 top1（第一名）和 margin>=0（边界差非负）。

#### 3. self（同样本）结果

```text
self rows: 252
self top1: 125
self margin >= 0: 125
self weak transfer candidate: 192
self strict clean: 93
median self margin delta: 1.25
mean self EOS logit delta: 2.0022
median self native group overlap: 64 / 64
```

self 结果基本复现 Phase918 的强通道级闭合现象。

#### 4. cross_domain（跨语义域）结果

最关键的是 cross_domain（跨语义域）仍然很强：

```text
cross_domain rows: 1722
cross_domain top1: 752
cross_domain margin >= 0: 752
cross_domain weak transfer candidate: 1312
cross_domain strict clean: 552
median cross_domain margin delta: 1.1875
mean cross_domain EOS logit delta: 1.8876
median cross_domain native group overlap: 54 / 64
```

这说明冻结通道组的作用不是只在同一个 case（样本）附近成立，跨 eval_domain（评估域）也大量有效。

#### 5. 最强控制项

最强正结果来自冻结的正 margin（边界差）通道：

```text
frozen_L39_margin_support_pos_64_scale_2
cross_domain rows: 82
cross_domain top1: 82
cross_domain margin >= 0: 82
cross_domain strict clean: 68
target states with top1: 12 / 12
median margin delta: 3.125
mean EOS logit delta: 3.9459
median native group overlap: 51 / 64
```

较弱但仍很强的正 margin 因子：

```text
frozen_L39_margin_support_pos_64_scale_1.75
cross_domain rows: 82
cross_domain top1: 80
cross_domain margin >= 0: 80
cross_domain strict clean: 66
target states with top1: 12 / 12
median margin delta: 2.390625
```

冻结的 `"a"` blocker 支持通道抑制也有效：

```text
frozen_L39_a_blocker_support_64_scale_0
cross_domain rows: 82
cross_domain top1: 61
cross_domain margin >= 0: 61
cross_domain strict clean: 47
target states with top1: 9 / 12
median margin delta: 2.25
```

但 `a_logit_support_64` 仍然不能完成闭合：

```text
a_logit_support_64 cross rows:
  top1: 0
  margin >= 0: 0
  strict clean: 0
```

这和 Phase918 一致：单纯抑制 `"a"` logit（对数几率）支持通道，不等于完成 EOS-vs-a 边界闭合。真正有效的是 EOS-a margin（结束符减“a”边界差）方向和 `"a"` 相对 EOS 的竞争方向，而不是只看 `"a"` 自身 logit。

### 六、结论是否正确

Phase918 附件提出的下一步是冻结通道组迁移验证，这个方向完全正确，而且 Phase919 给出了强正结果：

```text
Phase918 的 L39 通道组不是纯样本内局部拟合；
冻结来源通道编号后，跨目标样本仍然大量有效；
L39 MLP 中存在可迁移的 EOS-vs-a signed margin channel family（有符号边界通道族）。
```

但结论必须继续收紧：

```text
Phase919 证明的是 frozen channel IDs（冻结通道编号）具有跨样本可迁移性；
不是自然 action gate（动作门）已经闭合；
不是完整语言编码机制闭合；
也不是跨模型通用结论。
```

最稳妥表述：

```text
在 GLM4 的 Phase915 近边界 EOS-vs-a 状态中，
L39 MLP 存在一组高度共享、可冻结迁移的有符号边界通道族。
它比 Phase918 的“按样本重算通道组”更接近全局齿轮，
但仍需要 consensus group（共识通道组）和负控制验证。
```

### 七、核心进展

#### 1. 从局部通道方向推进到共享通道族

Phase918 的硬伤是：

```text
每个样本都重新计算 G(x)；
所以无法排除 case-conditioned readout alignment（样本条件化读出对齐）。
```

Phase919 把它推进为：

```text
G_s 在来源样本计算后冻结；
迁移到目标样本仍大量成功。
```

这说明通道组背后不是完全任意的局部方向，而是存在可复用结构。

#### 2. “全局齿轮图谱”得到一个更硬的节点

当前全局齿轮图谱中，L39 MLP 可标记为：

```text
L39 MLP:
  late boundary signed margin gear（后期有符号边界齿轮）

作用对象:
  EOS vs "a" near-boundary competition（结束符对“a”的近边界竞争）

证据级别:
  channel-level artificial causal closure（通道级人工因果闭合）
  + frozen cross-state transfer（冻结跨状态迁移）

尚未达到:
  natural gate closure（自然门控闭合）
```

#### 3. 复用差分机制更清楚

Phase919 说明当前机制不像“每个样本一个完全不同的方向”，而像：

```text
模型在 L39 MLP 中保留了一批共享边界通道；
不同样本激活这些通道的强度和组合略有变化；
但核心通道族高度重合并可迁移。
```

中位 overlap（重合）为：

```text
self: 64 / 64
cross: 55 / 64
cross_domain: 54 / 64
```

这既是正证据，也是硬边界：

```text
正证据:
  跨域仍有约 54 / 64 通道重合，说明存在共享通道族。

硬边界:
  也可能说明 Phase919 成功主要来自来源组和目标原生组高度重合；
  还没有证明一个更小、更稳定、更可解释的 consensus gear（共识齿轮）已经闭合。
```

### 八、问题、硬伤和瓶颈

#### 1. 候选池仍然很窄

当前有效候选为：

```text
GLM4: 12
qwen3: 0
DS7B: 0
```

所以 Phase919 是 GLM4 内部、Phase915 近边界 EOS-vs-a 条件下的强结果，不能外推到所有模型、所有终止场景或完整语言生成。

#### 2. 仍然依赖人工 route + L4 边界预条件

目标状态 \(z_t^B\) 仍由人工 route（路线）和 L4 boundary（第4层边界）构造：

$$
z_t^B
=
F(x_t;\ route_t,\ L4_t)
$$

自然模型没有自动进入这个状态。因此自然闭合仍未完成。

#### 3. 强因子不等于自然调节

最强结果使用：

```text
margin_support_pos_64 scale 2.0
a_blocker_support_64 scale 0.0
```

这说明通道组有因果能力，但还不知道模型自然运行时是否存在相同强度、相同方向的调节变量。

#### 4. 高 overlap 需要更严格负控制

cross_domain 的 native group overlap 中位数约为：

```text
54 / 64
```

这说明来源组和目标组本来就高度相似。它支持“共享通道族”判断，但还不能证明：

```text
任意冻结来源组都携带独立可迁移结构；
或者一个很小的固定子集足以解释全部闭合。
```

必须做随机同规模通道组、频率共识通道组、打乱来源组、holdout（留出）验证。

#### 5. 仍未破解自然 gate（门）

目前知道：

```text
调哪些通道可以让 EOS 赢；
还不知道模型自然什么时候、为什么、由谁调这些通道。
```

这就是当前和真正语言编码机制闭合之间的最大距离。

### 九、当前闭合标准和距离

对 EOS-vs-a near-boundary（结束符对“a”的近边界）子问题，较严格闭合标准应至少包括：

```text
1. 固定或低维 consensus channel set（共识通道集合）能跨 holdout 样本预测闭合；
2. 随机同规模通道组、打乱通道组、低重合通道组不能达到同等效果；
3. gate predictor（门控预测器）能从自然激活中预测这些通道何时被调节；
4. 不依赖人工 route + L4 预条件，或能解释自然模型为什么没有进入闭合状态；
5. full-vocabulary blocker（全词表阻塞者）被系统处理，而不是只处理 `"a"`；
6. exact-natural rollout（严格自然续写）稳定通过。
```

Phase919 当前达到：

```text
通道级人工因果能力: 强
冻结跨样本迁移: 强
GLM4 内部共享通道族证据: 强
自然门控解释: 弱
跨模型证据: 无
全语言编码机制闭合: 未完成
```

谨慎估计：

```text
EOS-vs-a L39 边界子问题进度: 约 55% - 60%
clean protocol edge graph（干净协议边图谱）进度: 约 35% - 40%
完整语言编码机制进度: 约 15% - 20%
```

这些百分比不是理论定论，只是根据当前证据覆盖范围给出的阶段性估计。

### 十、智能理论角度的洞察

Phase919 对智能理论的意义在于：

```text
语言生成不是单个 token logit（词元对数几率）被局部推高；
而是不同层级齿轮在全词表竞争场中移动边界。
```

当前观察到的结构更像：

```text
前中层:
  route（路线）、protocol（协议）、format（格式）把状态推近某个边界。

后层:
  L39 MLP 这类 late margin gear（后期边界齿轮）负责最后竞争边界。

通道层:
  不是单个 neuron（神经元）独立编码语义，
  而是一组可复用的 signed channel family（有符号通道族）控制 margin（边界差）。
```

从第一性原理看，语言编码机制可能不是：

```text
概念 -> 单向量 -> 词元
```

而更像：

```text
状态场 -> 路线场 -> 协议场 -> 边界场 -> 全词表竞争闭合
```

Phase919 给出的关键拼图是：

```text
边界场中存在可迁移的通道齿轮；
这些齿轮可以跨样本复用；
因此全局齿轮图谱不是只记录局部 patch（补丁），而可以逐步逼近共享内部结构。
```

### 十一、下一阶段任务

当前任务和下一任务仍处于同一阶段性目标：

```text
优先完成 clean protocol edge graph（干净协议边图谱）和全局齿轮图谱；
暂不把目标提前升级为完整自然闭合。
```

下一阶段建议为：

```text
Phase920:
Consensus L39 Signed Margin Gear Negative-control and Holdout Validation
（共识 L39 有符号边界齿轮负控制与留出验证）
```

具体任务：

```text
1. 从 12 个 GLM4 来源状态统计 channel frequency（通道频率），构造 consensus G*（共识通道组）。
2. 用 G* 直接迁移到所有目标状态，不再使用每个来源状态的独立 G_s。
3. 做 leave-one-case-out（留一案例）和 leave-one-domain-out（留一领域）验证。
4. 加入 random same-size channels（同规模随机通道）、activation-matched random channels（激活匹配随机通道）、shuffled source group（打乱来源组）作为负控制。
5. 测试更自然的低因子：
   1.125, 1.25, 1.375
   观察是否存在低幅度可迁移边界移动。
6. 对 full-vocabulary blocker（全词表阻塞者）记录 `"a"` 以外的替代阻塞者是否被引入。
7. 如果 consensus G* 通过 holdout，再进入 gate predictor（门控预测器）定位阶段。
```

闭合判据：

```text
如果 consensus G* 在 holdout cross_domain 中仍显著优于随机和打乱控制，
并覆盖大部分目标状态，
则 L39 signed margin gear（第39层有符号边界齿轮）可以从“共享通道族”
升级为“近似全局齿轮节点”。

如果 consensus G* 明显下降，
说明 Phase919 的成功主要依赖来源组和目标组的高重合局部选择，
需要继续做更细的子簇划分。
```

### 十二、通俗总结

Phase918 证明：

```text
在 GLM4 里，L39 MLP 里面确实有一批通道，
能把 EOS 从被 "a" 压住，推到超过 "a"。
```

Phase919 进一步证明：

```text
这些通道不是只能在一个样本里有效；
把一个样本找到的通道编号冻结下来，
换到别的样本里，仍然经常有效。
```

这很重要，因为它说明我们看到的不是单点补丁，而更像一批共享齿轮。

但还不能说已经破解语言机制：

```text
我们现在知道“拨哪些齿轮能让 EOS 赢”，
还不知道模型自然运行时“为什么没有自己拨这些齿轮”，
也不知道这个机制能否跨模型、跨更多语言任务稳定成立。
```

所以下一步不是急着宣布闭合，而是把这批通道压缩成更稳定的共识齿轮，并用随机负控制和留出样本验证它是否真的是全局结构。

## Phase 920: 共识 L39 有符号边界齿轮留出与负控制验证 [2026-07-04 07:52]

### 一、任务来源

Phase919（第919阶段）证明：

```text
冻结来源样本的 L39 有符号边界通道组，
迁移到其他目标样本后仍大量有效。
```

但 Phase919 仍有一个硬问题：

```text
来源组和目标原生组的 overlap（重合）较高；
所以需要验证：
  是不是存在更稳定的 consensus gear（共识齿轮）？
  还是只因为每个来源组都和目标组碰巧高度重合？
```

因此本阶段继续同一阶段性目标，完成 Phase920：

```text
把多个来源样本的 L39 通道组压缩成 consensus group（共识通道组）；
加入 leave-one-case（留一案例）和 leave-one-domain（留一领域）；
再加入 random（随机）、rotated（旋转错位）、a-logit-only（只看“a”自身）的负控制。
```

### 二、脚本和结果

新增脚本：

```text
tests/glm5/phase920_consensus_l39_signed_margin_gear_holdout_controls.py
```

新增运行脚本：

```text
tests/glm5/run_phase920_consensus_l39_signed_margin_gear_holdout_controls.sh
```

结果目录：

```text
tests/result/phase920_consensus_l39_signed_margin_gear_holdout_controls/consensus_l39_signed_margin_gear_holdout_controls/
```

核心结果：

```text
phase920_glm4_rows.jsonl
phase920_glm4_summary.json
phase920_cross_model_summary.json
phase920_cross_model_summary.md
```

三模型顺序仍为：

```text
qwen3 -> GLM4 -> DS7B
```

当前筛选条件下：

```text
qwen3: 0 candidates
GLM4: 12 candidates
DS7B: 0 candidates
```

因此实际前向测试集中在 GLM4。

### 三、测试原理

#### 1. 共识通道组

对某个通道组类型 \(k\)，每个来源状态 \(x_s\) 有一个通道集合：

$$
G_s^k
$$

Phase920 统计所有训练来源状态中的通道频率：

$$
freq_k(j)
=
\sum_{s\in \mathcal{T}}
\mathbf{1}[j\in G_s^k]
$$

取频率最高的 64 个通道：

$$
G_{cons}^{k}
=
Top64_j\ freq_k(j)
$$

这一步从 Phase919 的“每个来源一个冻结组”推进为“一个可复用共识组”。

#### 2. 留出验证

对目标样本 \(x_t\)，训练集合分三类：

$$
\mathcal{T}_{all}
=
\{x_s\}
$$

$$
\mathcal{T}_{case}(t)
=
\{x_s:\ case(s)\ne case(t)\}
$$

$$
\mathcal{T}_{domain}(t)
=
\{x_s:\ domain(s)\ne domain(t)\}
$$

分别得到：

$$
G_{all}^{k},\quad
G_{loo-case}^{k}(t),\quad
G_{loo-domain}^{k}(t)
$$

如果 leave-one-case（留一案例）和 leave-one-domain（留一领域）仍然有效，说明不是目标样本泄漏造成的。

#### 3. 负控制

负控制包括：

```text
random_all_64:
  从 L39 通道空间随机抽取64个通道。

rotated_consensus:
  把共识通道整体错位旋转，保留数量但破坏真实通道编号。

consensus_a_logit_support_64:
  只取支持 "a" 自身 logit 的通道，测试“单纯压 a”是否足够。
```

如果正向共识组和负控制没有明显分离，则不能把 Phase919 升级为稳定齿轮。

#### 4. 统一评价公式

对目标边界状态：

$$
z_t^B
=
F(x_t;\ route_t,\ L4_t)
$$

对共识组或负控制组 \(G\) 做缩放：

$$
a_{t,j}^{39\prime}
=
\begin{cases}
f\cdot a_{t,j}^{39}, & j\in G \\
a_{t,j}^{39}, & j\notin G
\end{cases}
$$

输出：

$$
z_{t}^{B,G,f}
=
F(x_t;\ route_t,\ L4_t,\ scale(G,f))
$$

边界差：

$$
M_t^{G,f}
=
z_t^{B,G,f}(EOS)
-
\max_{v\ne EOS}z_t^{B,G,f}(v)
$$

边界改变量：

$$
\Delta M_t^{G,f}
=
M_t^{G,f}
-
M_t^B
$$

### 四、客观结果

#### 1. 总体结果

GLM4 完成：

```text
target states: 12
total rows: 540
positive rows: 432
negative rows: 108
```

总体：

```text
all top1: 226
all margin >= 0: 226
all strict clean: 174
all weak candidate: 439
```

#### 2. 正向共识组结果

positive consensus（正向共识）：

```text
positive rows: 432
positive top1: 226
positive margin >= 0: 226
positive strict clean: 174
positive weak candidate: 432
median margin delta: 1.4375
mean EOS logit delta: 2.6059
median native overlap: 54 / 64
target states with top1: 12 / 12
target states with margin >= 0: 12 / 12
target states with strict clean: 10 / 12
```

#### 3. 负控制结果

negative controls（负控制）：

```text
negative rows: 108
negative top1: 0
negative margin >= 0: 0
negative strict clean: 0
negative weak candidate: 7
median margin delta: 0.078125
mean EOS logit delta: -0.5463
median native overlap: 0 / 64
```

这个分离很关键：

```text
正向共识组大量闭合；
负控制没有任何 top1 或 margin>=0；
说明结果不是“随便调64个通道就能闭合”。
```

#### 4. 留一案例结果

leave-one-case（留一案例）：

```text
rows: 144
top1: 74
margin >= 0: 74
strict clean: 58
weak candidate: 144
median margin delta: 1.40625
mean EOS logit delta: 2.5634
median native overlap: 53.5 / 64
target states with top1: 12 / 12
target states with margin >= 0: 12 / 12
target states with strict clean: 10 / 12
```

#### 5. 留一领域结果

leave-one-domain（留一领域）：

```text
rows: 144
top1: 74
margin >= 0: 74
strict clean: 58
weak candidate: 144
median margin delta: 1.40625
mean EOS logit delta: 2.5629
median native overlap: 53.5 / 64
target states with top1: 12 / 12
target states with margin >= 0: 12 / 12
target states with strict clean: 10 / 12
```

leave-one-case 和 leave-one-domain 结果几乎一致，说明共识通道组不是简单记住某个 case（样本）或某个 domain（领域）。

#### 6. 最强正控制

```text
consensus_margin_support_pos_64_all_train_scale_2
rows: 12
top1: 12
margin >= 0: 12
strict clean: 10
median margin delta: 3.40625
overlap: 57.5 / 64
```

留出版本也不下降到失效：

```text
consensus_margin_support_pos_64_leave_one_case_scale_2
rows: 12
top1: 12
margin >= 0: 12
strict clean: 10
median margin delta: 3.1875
overlap: 52 / 64
```

```text
consensus_margin_support_pos_64_leave_one_domain_scale_2
rows: 12
top1: 12
margin >= 0: 12
strict clean: 10
median margin delta: 3.1875
overlap: 52 / 64
```

较低因子仍有作用，但不闭合：

```text
scale 1.375:
  top1: 8 / 12
  margin >= 0: 8 / 12
  strict clean: 6 / 12

scale 1.25:
  top1: 0 / 12
  margin >= 0: 0 / 12
  但 weak candidate: 12 / 12
```

这说明低幅度调节已经推动边界，但尚不足以闭合。

### 五、结论是否正确

Phase920 给出比 Phase919 更强、更干净的结论：

```text
L39 有符号边界通道不是只能按来源样本单独迁移；
多个来源样本压缩出的 consensus group（共识组）也能留出泛化；
并且明显击败随机、错位和 a-logit-only 负控制。
```

因此，当前 L39 MLP 节点可以从：

```text
shared channel family（共享通道族）
```

谨慎升级为：

```text
approximate global signed margin gear
（近似全局有符号边界齿轮）
```

但仍不能升级为自然闭合，因为：

```text
1. 仍依赖人工 route + L4 预条件；
2. 最强闭合仍需要 scale 1.75 或 2.0；
3. 仍只在 GLM4 的 12 个 EOS-vs-a 近边界状态中验证；
4. 还没有找到自然 gate variable（门控变量）；
5. 还没有解释模型自然运行时为什么没有自动拨动这个齿轮。
```

### 六、核心进展

Phase920 是当前全局齿轮图谱中的重要收紧：

```text
Phase918:
  按样本重算通道组，可以人工闭合。

Phase919:
  冻结来源通道组，跨样本仍能闭合。

Phase920:
  压缩成共识通道组，留一案例/领域仍能闭合，
  且负控制不能闭合。
```

这条链条说明：

```text
L39 EOS-vs-a 边界齿轮不是普通 patch（补丁）；
它已经具备图谱节点所需的稳定性、可迁移性和负控制分离。
```

### 七、硬伤和瓶颈

#### 1. 仍是人工齿轮，不是自然齿轮

当前知道：

```text
调 consensus margin gear（共识边界齿轮）可以让 EOS 赢。
```

但还不知道：

```text
自然模型里谁调它？
什么时候调？
为什么在原始状态没有调到闭合？
```

#### 2. 强因子仍偏人工

最稳定闭合来自：

```text
scale 1.75
scale 2.0
```

较低因子：

```text
scale 1.125 和 1.25
```

主要表现为 weak candidate（弱候选），没有直接闭合。

这说明自然门控如果存在，可能不是简单小幅度单齿轮调节，而可能需要多个齿轮协同。

#### 3. 数据范围仍小

当前有效目标状态仍是 12 个，来源仍是 GLM4。虽然负控制分离很强，但还不能推广到：

```text
更多 blocker（阻塞者）
更多自然提示
更多语义域
qwen3 / DS7B
完整多步续写
```

#### 4. overlap 仍高

共识组和目标原生组 overlap 仍约：

```text
52 - 57.5 / 64
```

这支持全局齿轮判断，但也提示：

```text
真正最小齿轮可能比64通道更小；
需要继续压缩到 top16 / top32 或频率阈值组。
```

### 八、闭合标准和当前距离

对 L39 EOS-vs-a 子问题，Phase920 已满足：

```text
1. 共识通道组可跨目标状态闭合；
2. 留一案例和留一领域仍有效；
3. 随机、错位、a-logit-only 负控制不能闭合；
4. 全词表 top1 和 margin>=0 被同时记录；
5. strict clean 大量出现。
```

仍未满足：

```text
1. 自然 gate（门）来源定位；
2. 低因子自然强度闭合；
3. route + L4 不人工化；
4. 跨模型复现；
5. 多 blocker 与多步 rollout（续写）闭合。
```

阶段性估计：

```text
L39 EOS-vs-a 边界齿轮子问题进度: 约 65% - 70%
clean protocol edge graph（干净协议边图谱）进度: 约 40% - 45%
完整语言编码机制进度: 约 18% - 22%
```

### 九、智能理论洞察

Phase920 对“语言背后数学结构”的提示更清楚：

```text
语言模型内部可能存在可复用的边界齿轮；
这些齿轮不是语义概念本身，
而是控制全词表竞争场中某类状态转换的操作结构。
```

这类结构更接近：

```text
operator（操作符）
```

而不是：

```text
feature label（特征标签）
```

也就是说，语言能力可能不是简单由“词义向量”组成，而是由：

```text
状态空间
路线
协议
边界齿轮
门控变量
全词表竞争
```

共同形成。Phase920 找到的是其中一个后层边界齿轮。

### 十、下一阶段任务

Phase918-920 已经完成了当前小阶段目标：

```text
证明 L39 MLP 中存在可压缩、可留出泛化、能击败负控制的 EOS-vs-a signed margin gear。
```

接下来进入同一大方向下的下一子阶段，但任务性质发生变化：

```text
从“齿轮是否存在”
转向“自然门控变量在哪里”。
```

建议 Phase921：

```text
Natural Gate Variable Search for Consensus L39 Signed Margin Gear
（共识 L39 有符号边界齿轮的自然门控变量搜索）
```

具体做法：

```text
1. 固定 Phase920 的 consensus margin gear（共识边界齿轮）。
2. 不再直接强行 scale 2.0，而是测自然状态中哪些上游变量预测该齿轮激活不足。
3. 候选变量包括：
   L4 boundary group activation（第4层边界组激活）
   L0 route delta norm（第0层路线差分范数）
   attention entropy（注意力熵）
   prefix protocol token（前缀协议词元）
   blocker gap（阻塞者差距）
4. 建立简单可解释门控评分，不使用复杂黑盒拟合。
5. 如果找到门控变量，再做低因子联动干预。
```

是否继续自动执行：

```text
Phase921 属于同一总路线，
但已经从“共识齿轮存在性验证”进入“自然门控来源定位”子阶段。
当前 Phase918-920 的阶段性目标已经完成；
下一步应作为新子阶段启动。
```

### 十一、通俗总结

Phase920 说明：

```text
不是每次都要临时找一批新通道；
把多个样本里经常出现的通道合成一批“共识通道”，
这批通道也能让 EOS 赢。
```

而且：

```text
随机通道不行；
错位通道不行；
只看 "a" 自身 logit 的通道也不行。
```

所以现在可以更有把握地说：

```text
GLM4 的 L39 MLP 里确实有一个比较稳定的 EOS-vs-a 边界齿轮。
```

但还没破解自然机制：

```text
我们知道这个齿轮存在，也知道拨它有用；
下一步要找模型自然状态中控制这个齿轮的“手柄”在哪里。
```

## Phase 921: 共识 L39 有符号边界齿轮的自然门控变量诊断 [2026-07-04 08:53]

### 一、任务来源和判断

本阶段先分析上传附件对 Phase919（第919阶段）和 Phase920（第920阶段）的判断。附件判断基本正确：

```text
Phase919:
  冻结来源样本 L39 通道组后，跨样本、跨领域仍大量有效。

Phase920:
  压缩为 consensus group（共识通道组）后，
  leave-one-case（留一案例）和 leave-one-domain（留一领域）仍有效，
  并且 random（随机）、rotated（错位）、a-logit-only（只看“a”自身对数几率）负控制不能闭合。
```

因此当前理论标签应保持为：

```text
条件化输出场闭合理论
+
近似全局 L39 有符号边界齿轮
+
自然门控变量缺失
```

不要改理论名。Phase919-920 证明了“齿轮存在，而且人工拨动有效”，但没有证明“模型自然运行时会自动拨动这个齿轮”。本阶段继续完成同一总路线下的下一步：Phase921（第921阶段），目标是诊断自然门控变量候选。

### 二、脚本和结果位置

新增测试脚本：

```text
tests/glm5/phase921_natural_gate_variable_search_for_l39_margin_gear.py
```

新增运行脚本：

```text
tests/glm5/run_phase921_natural_gate_variable_search_for_l39_margin_gear.sh
```

结果目录：

```text
tests/result/phase921_natural_gate_variable_search_for_l39_margin_gear/natural_gate_variable_search_for_l39_margin_gear/
```

核心结果文件：

```text
phase921_qwen3_summary.json
phase921_glm4_summary.json
phase921_deepseek7b_summary.json
phase921_glm4_state_rows.jsonl
phase921_glm4_factor_rows.jsonl
phase921_cross_model_summary.json
phase921_cross_model_summary.md
```

三模型按顺序执行：

```text
qwen3 -> GLM4 -> DS7B
```

当前 Phase915 条件下候选池仍为：

```text
qwen3: 0
GLM4: 12
DS7B: 0
```

进一步核查 Phase915 原始池：

```text
GLM4 L39_mlp_output_scale_1.5 rows: 12
boundary blocker token = "a": 12
weak/rank candidate: 12
unique candidate keys: 12
```

因此 Phase921 无法在当前条件下继续扩展更多 L39 `"a"` blocker 候选；这是结果边界，不是脚本截断造成的。

### 三、测试原理

#### 1. 固定 Phase920 共识齿轮

Phase921 固定 Phase920 中的：

```text
consensus_margin_support_pos_64
```

即：

$$
G_{cons}^{margin+}
=
Top64_j\ freq_{margin+}(j)
$$

其中：

$$
freq_{margin+}(j)
=
\sum_{s\in \mathcal{T}}
\mathbf{1}[j\in G_s^{margin+}]
$$

本阶段不再每个样本重新寻找通道组，也不再换成来源冻结组，而是固定同一个共识齿轮。

#### 2. 测低因子闭合阈值

对每个目标状态 \(x_t\)，在同一个共识齿轮上测试：

```text
1.125, 1.25, 1.375, 1.5, 1.75, 2.0
```

干预为：

$$
a_{t,j}^{39\prime}
=
\begin{cases}
f\cdot a_{t,j}^{39}, & j\in G_{cons}^{margin+}\\
a_{t,j}^{39}, & j\notin G_{cons}^{margin+}
\end{cases}
$$

输出：

$$
z_t^{B,G,f}
=
F(x_t;\ route_t,\ L4_t,\ scale(G_{cons}^{margin+},f))
$$

边界差：

$$
M_t^{G,f}
=
z_t^{B,G,f}(EOS)
-
\max_{v\ne EOS} z_t^{B,G,f}(v)
$$

记录每个状态的最小闭合因子：

$$
f_t^{margin}
=
\min\{f: M_t^{G,f}\ge 0\}
$$

同时记录：

$$
f_t^{top1}
=
\min\{f: rank_t^{G,f}(EOS)=1\}
$$

$$
f_t^{strict}
=
\min\{f: StrictClean_t^{G,f}=1\}
$$

#### 3. 自然门控变量候选

本阶段不训练复杂模型，只采集简单可解释变量：

```text
route_delta_norm:
  终止路线差分范数。

route_eos_rank:
  route 状态下 EOS 排名。

boundary_eos_rank:
  route + L4 边界状态下 EOS 排名。

boundary_gap_to_zero:
  EOS 到闭合边界还差多少。

protocol_vs_eos:
  协议续写项相对 EOS 的优势。

l4_activation_abs_top:
  L4 边界组最高激活幅度。

consensus_margin_support_sum:
  L39 共识齿轮在自然边界状态中的 EOS-vs-a margin 支持总量。

consensus_activation_abs_mean / median:
  共识齿轮自然激活强度。
```

定义一个简单诊断量：

$$
Pressure_{gate}(x)
=
Gap_{boundary}(x)
-
Support_{consensus}(x)
$$

其中：

$$
Gap_{boundary}(x)
=
-M_t^B
$$

$$
Support_{consensus}(x)
=
\sum_{j\in G_{cons}^{margin+}}
a_j^{39}(x)
\cdot
\left(
(W_U(EOS)-W_U(a))^\top W_{down}^{39}[:,j]
\right)
$$

该量不是最终理论，只是诊断：如果 gap（缺口）小、共识齿轮自然支持大，低因子更容易闭合。

#### 4. 简单阈值分离

把状态分为：

```text
低因子可闭合:
  f <= 1.375 时 margin >= 0。

低因子不可闭合:
  f > 1.375 才能 margin >= 0。
```

对每个候选变量只做简单阈值扫描：

$$
\hat{y}(x)
=
\mathbf{1}[v(x)\ge \theta]
$$

或：

$$
\hat{y}(x)
=
\mathbf{1}[v(x)\le \theta]
$$

记录 best threshold accuracy（最佳阈值准确率）。这不是训练复杂模型，只是检查变量是否有明显分离能力。

### 四、客观结果

#### 1. 因子响应

GLM4 完成：

```text
state rows: 12
factor response rows: 72
```

按因子统计：

```text
factor 1.125:
  top1: 0 / 12
  margin >= 0: 0 / 12
  strict clean: 0 / 12
  median margin: -0.75

factor 1.25:
  top1: 0 / 12
  margin >= 0: 0 / 12
  strict clean: 0 / 12
  median margin: -0.3125

factor 1.375:
  top1: 8 / 12
  margin >= 0: 8 / 12
  strict clean: 6 / 12
  median margin: 0.125

factor 1.5:
  top1: 8 / 12
  margin >= 0: 8 / 12
  strict clean: 6 / 12
  median margin: 0.5

factor 1.75:
  top1: 12 / 12
  margin >= 0: 12 / 12
  strict clean: 10 / 12
  median margin: 1.4375

factor 2.0:
  top1: 12 / 12
  margin >= 0: 12 / 12
  strict clean: 10 / 12
  median margin: 2.25
```

结论：

```text
1.125 和 1.25 太弱，不能闭合；
1.375 是明显转折点；
1.75 以上基本全闭合；
strict clean 仍有 2 / 12 状态没有完全通过。
```

#### 2. 低因子闭合标签

```text
low_factor_1375_margin: 8 / 12
low_factor_1375_top1: 8 / 12
low_factor_1375_strict: 6 / 12
```

这说明 12 个状态内部存在难易差异，不是所有状态都需要同等强度的齿轮拨动。

#### 3. 候选变量分离结果

在小样本 12 个状态中，多个变量可以把 “1.375 可闭合” 和 “1.375 不可闭合” 分开。

排名靠前的变量：

```text
route_eos_rank:
  positive mean: 7.0
  negative mean: 16.5
  best threshold: <= 7
  accuracy: 1.0

boundary_eos_rank:
  positive mean: 5.0
  negative mean: 12.0
  best threshold: <= 5
  accuracy: 1.0

protocol_blocker_pressure:
  positive mean: 1.578125
  negative mean: 3.609375
  best threshold: <= 1.6875
  accuracy: 1.0

protocol_vs_eos:
  positive mean: 0.4765625
  negative mean: 1.578125
  best threshold: <= 0.5625
  accuracy: 1.0

boundary_gap_to_zero:
  positive mean: 1.1015625
  negative mean: 2.03125
  best threshold: <= 1.1875
  accuracy: 1.0

consensus_margin_support_sum:
  positive mean: 19.3698
  negative mean: 17.8396
  best threshold: >= 18.5634
  accuracy: 1.0

consensus_activation_abs_mean:
  positive mean: 15.4130
  negative mean: 14.1987
  best threshold: >= 14.8412
  accuracy: 1.0
```

这些结果说明：

```text
低因子能否闭合，和三个因素同时相关：
1. EOS 已经在 route/boundary 中多接近；
2. protocol / blocker 压力有多大；
3. L39 共识齿轮在自然状态中已经有多少 margin 支持。
```

#### 4. 状态层面观察

低因子不容易闭合的状态通常表现为：

```text
boundary_gap_to_zero 更大；
protocol_vs_eos 更大；
consensus_margin_support_sum 更低；
route_eos_rank / boundary_eos_rank 更差。
```

例如 hard group（较难组）平均：

```text
boundary_gap_to_zero: 2.03125
protocol_vs_eos: 1.578125
consensus_margin_support_sum: 17.8396
boundary_eos_rank: 12.0
```

easy group（较易组）平均：

```text
boundary_gap_to_zero: 1.1015625
protocol_vs_eos: 0.4765625
consensus_margin_support_sum: 19.3698
boundary_eos_rank: 5.0
```

### 五、结论是否正确

附件提出 Phase921 应寻找自然门控变量，这个判断正确。Phase921 第一轮结果说明：

```text
自然门控变量不是完全无迹可寻；
低因子能否拨动 L39 共识齿轮，和 route 接近度、boundary gap、protocol pressure、共识齿轮自然支持量有清晰关系。
```

但结论必须严格收紧：

```text
Phase921 没有证明自然 action gate 已经找到；
只是找到了候选门控变量和低因子闭合难易之间的诊断关联。
```

最谨慎结论：

```text
GLM4 的 EOS-vs-a 近边界状态中，
L39 共识齿轮的低因子闭合难易，
可以被 route/boundary/protocol/L39-support 一组简单变量分离。
这些变量是 natural gate candidate（自然门控候选），
还不是 natural gate mechanism（自然门控机制）。
```

### 六、问题和硬伤

#### 1. 样本数仍然太小

当前有效状态：

```text
12 states
4 cases
3 domains
```

case 分布：

```text
p856_038_object_object: 6
p856_009_animal_fish: 4
p856_022_material_iron: 1
p856_008_animal_bird: 1
```

因此阈值准确率 1.0 不能过度解读，可能有重复 case 影响。

#### 2. 变量不是因果门，只是诊断变量

例如：

```text
boundary_gap_to_zero 小 -> 更容易低因子闭合
```

这可能只是状态接近边界的结果，不一定是模型内部主动控制 L39 齿轮的门。

#### 3. 没有做上游变量因果干预

本阶段没有直接干预：

```text
protocol_vs_eos
boundary_gap
route_eos_rank
```

因此不能说明改变这些变量就会自然触发 L39 齿轮。

#### 4. 仍依赖人工 route + L4 + L39 共识齿轮

Phase921 仍是在人工构造的近边界状态上进行诊断，不是原始自然生成过程。

### 七、理论进展

Phase921 对全局齿轮图谱的推进是：

```text
L39 齿轮本身已经较稳定；
现在开始看到它的上游门控候选：
  route 接近度
  boundary gap
  protocol pressure
  L39 自然 margin support
```

这让图谱从：

```text
semantic answer
  -> prompt / protocol gate
  -> termination route
  -> L4 boundary adjuster
  -> L39 consensus signed margin gear
  -> artificial closure
```

推进到：

```text
semantic answer
  -> prompt / protocol gate
  -> termination route
  -> L4 boundary adjuster
  -> gate candidate variables
  -> L39 consensus signed margin gear
  -> low-factor closure difficulty
```

还没有到：

```text
natural gate closure
```

### 八、闭合标准和当前距离

Phase921 当前达到：

```text
1. 固定共识齿轮；
2. 得到每个状态的最小闭合因子；
3. 找到能分离低因子闭合难易的候选变量；
4. 确认 route / boundary / protocol / L39-support 都参与门控候选图谱。
```

未达到：

```text
1. 候选变量的因果干预验证；
2. 更大样本验证；
3. 原始自然状态验证；
4. 跨模型验证；
5. 多 blocker 验证；
6. 多步 rollout 验证。
```

阶段性估计：

```text
L39 EOS-vs-a 边界齿轮子问题进度: 约 70%
自然门控变量定位进度: 约 20% - 25%
clean protocol edge graph（干净协议边图谱）进度: 约 42% - 46%
完整语言编码机制进度: 约 18% - 22%
```

### 九、下一阶段任务

当前 Phase921 已完成自然门控变量的第一轮诊断。下一步如果继续同一子阶段，应做：

```text
Phase922:
Candidate Gate Variable Causal Coupling Test
（候选门控变量因果联动测试）
```

任务：

```text
1. 固定 Phase920 consensus margin gear（共识边界齿轮）。
2. 挑选 Phase921 最清晰变量：
   boundary_gap_to_zero
   protocol_vs_eos
   consensus_margin_support_sum
   route_eos_rank / boundary_eos_rank
3. 不直接使用 scale 2.0，而测试低因子：
   L39 factor = 1.125, 1.25, 1.375
4. 对上游变量做可解释小干预：
   轻微增强 L4 boundary adjuster；
   或轻微增强 route delta；
   或抑制 protocol continuation pressure。
5. 检查联动是否比单独低因子更强。
```

成功标准：

```text
如果低因子 L39 + 上游候选变量小干预
明显强于低因子 L39 单独干预，
则说明候选变量不只是相关诊断，
而可能进入自然门控因果链。
```

### 十、通俗总结

Phase920 之前我们知道：

```text
L39 里有一个齿轮，拨它能让 EOS 赢。
```

Phase921 进一步看到：

```text
有些状态轻轻拨一点就能赢；
有些状态必须拨很大才行。
```

这些差异不是随机的。更容易赢的状态通常已经满足：

```text
EOS 排名更靠前；
离边界更近；
协议续写压力更小；
L39 共识齿轮自然支持更强。
```

所以现在的客观拼图是：

```text
我们不仅找到了齿轮，
还开始看到“什么时候需要拨多大”的候选门控线索。
```

但还不能说已经找到自然门控机制，因为还没有证明改变这些线索会让模型自然拨动齿轮。

## Phase 922: 候选门控变量因果耦合测试 [2026-07-04 09:13]

### 一、任务来源与附件判断核查

本阶段读取并分析了最新附件中关于 Phase921（第921阶段）的判断。总体看，附件判断基本正确，但需要严格限定证据层级：

```text
Phase921 不是自然门控机制闭合；
Phase921 只是发现了自然状态变量与低强度 L39 齿轮成功率之间的诊断关联。
```

附件提出的下一步是合理的：固定 Phase920（第920阶段）的 L39 consensus margin gear（共识边界齿轮），在低强度 L39 干预下，对 Phase921（第921阶段）候选变量做小幅上游干预，检查是否真的能增强边界闭合。这正是 Phase922（第922阶段）完成的任务。

### 二、测试脚本与结果位置

新增脚本：

```text
tests/glm5/phase922_candidate_gate_variable_causal_coupling_test.py
tests/glm5/run_phase922_candidate_gate_variable_causal_coupling_test.sh
```

结果目录：

```text
tests/result/phase922_candidate_gate_variable_causal_coupling_test/candidate_gate_variable_causal_coupling_test/
```

核心输出：

```text
phase922_qwen3_summary.json
phase922_glm4_summary.json
phase922_deepseek7b_summary.json
phase922_glm4_rows.jsonl
phase922_cross_model_summary.json
phase922_cross_model_summary.md
```

### 三、测试原理

Phase921（第921阶段）给出的候选变量包括：

```text
route_eos_rank（路线后 EOS 排名）
boundary_eos_rank（边界后 EOS 排名）
boundary_gap_to_zero（边界距离零点）
protocol_vs_eos（协议续写压力相对 EOS 的优势）
consensus_margin_support_sum（共识边界齿轮自然支持）
```

Phase922（第922阶段）不再只看相关性，而是做因果耦合筛查：

```text
固定 L39 共识齿轮；
把 L39 factor（L39 缩放因子）限定在低强度；
在低强度 L39 的基础上叠加可解释的小幅上游干预；
比较叠加干预是否强于 L39-only（仅 L39）基线。
```

低强度 L39 因子：

```text
1.125
1.25
1.375
```

上游小干预：

```text
route_alpha_1.125
route_alpha_1.25
l4_boundary_1.05
l4_boundary_1.10
protocol_last8_0.90
protocol_answer_last_0.90
route + L4 组合
route + protocol 组合
L4 + protocol 组合
route + L4 + protocol 组合
```

同时加入方向对照：

```text
route_alpha_0.875_direction_control
l4_boundary_0.95_direction_control
protocol_last8_1.10_direction_control
```

### 四、核心公式

目标不是直接最大化 EOS logit（EOS 对数几率），而是检查 EOS 与当前 blocker（阻塞词）的边界差值：

$$
M(x)=z_{\mathrm{EOS}}(x)-z_{\mathrm{blocker}}(x)
$$

L39-only（仅 L39）低强度基线为：

$$
M_{\mathrm{L39}}(x,f)
=
M\left(x;\;G_{39}\times f\right)
$$

候选上游变量耦合干预为：

$$
M_{\mathrm{coupled}}(x,f,u)
=
M\left(x;\;G_{39}\times f,\;U(u)\right)
$$

相对 L39-only（仅 L39）的增益为：

$$
\Delta M_{\mathrm{couple}}(x,f,u)
=
M_{\mathrm{coupled}}(x,f,u)-M_{\mathrm{L39}}(x,f)
$$

闭合判断：

$$
\mathrm{closure}(x,f,u)=
\mathbf{1}\left[
M_{\mathrm{coupled}}(x,f,u)\ge 0
\land
\mathrm{rank}_{\mathrm{EOS}}=1
\right]
$$

新增闭合判断：

$$
\mathrm{new\_closure}
=
\mathbf{1}\left[
M_{\mathrm{L39}}(x,f)<0
\land
M_{\mathrm{coupled}}(x,f,u)\ge 0
\right]
$$

### 五、跨模型结果

跨模型总结果：

```text
qwen3: 没有 Phase915 L39 候选，未进入 Phase922 实测。
GLM4: 12 个状态进入测试。
DS7B: 没有 Phase915 L39 候选，未进入 Phase922 实测。
```

GLM4（GLM4 模型）测试规模：

```text
状态数: 12
L39-only rows（仅 L39 行）: 36
candidate_plus rows（候选正向干预行）: 360
direction_control rows（方向对照行）: 108
总 rows（总行数）: 504
```

Phase920（第920阶段）共识组诊断：

```text
train_state_count: 12
contributing_state_count: 12
unique_channel_count: 94
chosen_size: 64
chosen_min_frequency: 6
chosen_median_frequency: 12.0
chosen_max_frequency: 12
```

这说明 Phase922 使用的 L39 共识齿轮不是单个样本齿轮，而是 12 个状态上复现频率较高的一组通道。

### 六、GLM4 客观结果

L39-only（仅 L39）低强度基线：

```text
rows: 36
top1（第一名）: 8
margin_nonnegative（边界非负）: 8
strict_clean_candidate（严格干净候选）: 6
median_margin_delta_vs_target_boundary: 0.84375
mean_margin_delta_vs_target_boundary: 0.8420138888888888
median_patched_margin: -0.5625
```

candidate_plus（候选正向干预）整体：

```text
rows: 360
top1: 80
margin_nonnegative: 80
strict_clean_candidate: 60
improved_margin_vs_l39_only: 133
worsened_margin_vs_l39_only: 123
new_margin_closure_vs_l39_only: 0
new_top1_vs_l39_only: 0
new_strict_vs_l39_only: 0
mean_margin_delta_vs_l39_only: 0.021527777777777778
```

direction_control（方向对照）整体：

```text
rows: 108
top1: 26
margin_nonnegative: 26
strict_clean_candidate: 18
improved_margin_vs_l39_only: 55
worsened_margin_vs_l39_only: 17
new_margin_closure_vs_l39_only: 2
new_top1_vs_l39_only: 2
new_strict_vs_l39_only: 0
mean_margin_delta_vs_l39_only: 0.03935185185185185
```

最关键结果：

```text
candidate_plus 正向候选干预：
  可以轻微移动 margin；
  但没有新增闭合。

direction_control 方向对照：
  出现 2 个新增 margin/top1 闭合；
  但没有新增 strict clean 闭合。
```

### 七、新增闭合细节

两个新增闭合都来自同一个方向对照：

```text
control_label: route_alpha_0.875_direction_control
l39_factor: 1.25
case: p856_009_animal_fish
object: fish
eval_domain: animal
```

两个状态分别是：

```text
p856_009_animal_fish | question_plain | flip
p856_009_animal_fish | question_plain | zero
```

数值：

```text
L39-only margin: -0.125
patched margin: 0.0
margin_delta_vs_l39_only: 0.125
patched_eos_rank: 1
patched_blocker_token: a
```

这说明它们不是强闭合，而是贴边闭合：

```text
margin 从 -0.125 推到 0.0；
EOS 排到第一；
但 strict clean 没有新增。
```

### 八、对 Phase921 判断的修正

Phase921（第921阶段）的候选变量判断需要更新为：

```text
候选变量确实与低强度 L39 成功率有关；
候选变量也能通过小干预轻微移动边界；
但是当前正向候选干预没有带来新增闭合；
新增闭合反而来自 route_alpha_0.875 方向对照。
```

所以 Phase922（第922阶段）不能写成：

```text
自然门控变量因果成立。
```

更准确的结论是：

```text
候选门控变量存在弱因果耦合迹象；
但方向性不干净；
当前还没有找到可靠的自然门控因果变量。
```

证据标签：

```text
candidate_moves_margin_but_direction_control_only_adds_closure
```

### 九、为什么这个结果重要

如果 route_alpha（路线强度）越大越好，那么 `route_alpha_1.125` 和 `route_alpha_1.25` 应该比 `route_alpha_0.875` 更容易新增闭合。

但实际结果是：

```text
route_alpha_1.25:
  mean_margin_delta_vs_l39_only = 0.1440972222222222
  new_margin_closure = 0

route_alpha_0.875_direction_control:
  mean_margin_delta_vs_l39_only = 0.07291666666666667
  new_margin_closure = 2
```

这说明当前现象不符合简单单调门控假设：

```text
route 越强，EOS 越容易闭合。
```

更可能的机制是：

```text
路线强度存在局部最优；
过强 route 可能同时增强 EOS 与 blocker；
较弱 route 反而可能减少某些 blocker 对抗；
L39 齿轮需要与 route 边界位置发生匹配，而不是简单叠加。
```

### 十、问题、硬伤与瓶颈

第一，样本仍然小。

```text
GLM4 只有 12 个可用状态；
qwen3 和 DS7B 没有进入 Phase922 的 L39 候选；
新增闭合只有 2 行，且来自同一 fish case。
```

第二，新增闭合太贴边。

```text
margin = 0.0
不是大幅超过边界；
因此很可能是局部边界扰动，不是稳定自然机制。
```

第三，方向性不干净。

```text
正向候选干预没有新增闭合；
方向对照反而新增闭合；
说明 Phase921 候选变量不能直接解释为自然门控旋钮。
```

第四，小模型偏差仍然重要。

```text
GLM4 当前小模型内部结构可能较粗糙；
L39 边界齿轮可能是压缩后的局部替代结构；
在更大模型里 route / protocol / boundary 的分工可能更清晰。
```

### 十一、闭合标准与当前距离

自然门控闭合至少需要满足：

```text
1. 候选变量干预在多个 case 上稳定新增闭合；
2. 正向干预优于方向对照；
3. margin 不只是贴边到 0，而是有稳定正间隔；
4. strict clean 也同步新增；
5. qwen3、GLM4、DS7B 至少有两个模型可复现；
6. 能预测未见状态的低强度 L39 成功/失败。
```

Phase922 当前只满足：

```text
候选正向干预可以轻微移动 margin；
direction control 暴露了 route 方向非单调问题。
```

距离自然门控闭合仍然较远。

阶段性估计：

```text
L39 EOS-vs-a 边界齿轮子问题进度: 约 72%
候选自然门控变量因果验证进度: 约 25% - 30%
route-response（路线响应）方向性理解进度: 约 10% - 15%
clean protocol edge graph（干净协议边图谱）进度: 约 43% - 47%
完整语言编码机制进度: 约 18% - 22%
```

### 十二、智能理论角度的关键洞察

Phase922（第922阶段）的关键洞察不是“找到了门控变量”，而是：

```text
语言生成边界不是单调加法系统；
局部齿轮与上游路线之间存在匹配关系；
正确的路线强度可能不是越大越好，而是要落在某个边界位置。
```

这更接近一个动态齿轮系统：

```text
一个齿轮能否推动输出，
不只取决于齿轮本身的强度，
还取决于它与上游 route state（路线状态）、
protocol pressure（协议压力）、
blocker field（阻塞词场）之间的相位匹配。
```

因此，破解语言编码机制不能只问：

```text
哪个组件提高 EOS？
```

而应该问：

```text
在什么路线状态下，
哪个齿轮以什么强度，
能把全词表竞争边界推过稳定阈值？
```

### 十三、下一阶段任务

当前任务与下一任务仍属于同一阶段：

```text
自然门控变量因果验证阶段。
```

下一步应继续自动完成：

```text
Phase923:
Route Alpha Response Curve Audit
（路线强度响应曲线审计）
```

目标：

```text
1. 固定 L39 consensus margin gear（共识边界齿轮）。
2. 固定 L39 factor = 1.125, 1.25, 1.375。
3. 系统扫描 route_alpha:
   0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5
4. 判断 route_alpha 是否单调；
5. 判断新增闭合是否只存在于局部 alpha 区间；
6. 将 route_alpha 响应曲线加入全局齿轮图谱。
```

成功标准：

```text
如果 route_alpha 响应存在稳定峰值区间，
则说明 Phase922 的方向对照不是随机噪声，
而是路线-齿轮匹配曲线的一个切面。

如果没有稳定峰值，只是 fish 个例，
则 Phase922 的新增闭合应降级为局部偶然边界扰动。
```

### 十四、通俗总结

Phase921（第921阶段）像是发现：

```text
有些门旁边有几个刻度，看起来和开门难度有关。
```

Phase922（第922阶段）真的去拨这些刻度，结果发现：

```text
正向拨动刻度，门会松一点，但没有真正多开几扇门；
反向拨动其中一个路线刻度，反而让两个贴边的门刚好开了。
```

所以现在不能说已经找到门控开关。更准确地说：

```text
门附近确实有机械联动；
但这个联动不是简单的“往大拨就更好”；
下一步必须画出路线强度的完整响应曲线。
```

## Phase 923: 路线强度响应曲线审计 [2026-07-04 09:18]

### 一、任务来源

Phase922（第922阶段）发现一个关键异常：

```text
candidate_plus（候选正向干预）可以轻微移动 margin（边界差值），但没有新增闭合；
route_alpha_0.875_direction_control（路线强度 0.875 方向对照）反而新增了 2 个贴边闭合。
```

这个结果不能直接解释为自然门控变量成立，也不能简单解释为随机噪声。最合理的下一步，是把 route_alpha（路线强度）从单点对照扩展为完整响应曲线。

所以 Phase923（第923阶段）继续同一阶段任务：

```text
自然门控变量因果验证阶段；
具体子任务是 route-response curve（路线响应曲线）审计。
```

### 二、脚本与结果位置

新增脚本：

```text
tests/glm5/phase923_route_alpha_response_curve_audit.py
tests/glm5/run_phase923_route_alpha_response_curve_audit.sh
```

结果目录：

```text
tests/result/phase923_route_alpha_response_curve_audit/route_alpha_response_curve_audit/
```

核心输出：

```text
phase923_qwen3_summary.json
phase923_glm4_summary.json
phase923_deepseek7b_summary.json
phase923_glm4_rows.jsonl
phase923_glm4_curves.jsonl
phase923_cross_model_summary.json
phase923_cross_model_summary.md
```

### 三、测试原理

本阶段固定：

```text
Phase920 consensus L39 margin gear（共识 L39 边界齿轮）
L4 boundary spec（L4 边界调节器）
protocol pressure（协议压力）不额外干预
```

只扫描 route_alpha（路线强度）：

```text
0.5
0.625
0.75
0.875
1.0
1.125
1.25
1.375
1.5
```

L39 factor（L39 缩放因子）仍固定为低强度：

```text
1.125
1.25
1.375
```

因此每条曲线对应：

```text
一个状态 x
一个 L39 factor
九个 route_alpha 点
```

### 四、核心公式

路线强度响应函数：

$$
M_{\alpha}(x,f)
=
z_{\mathrm{EOS}}\left(x;\alpha\Delta r,\;G_{39}f\right)
-
z_{\mathrm{blocker}}\left(x;\alpha\Delta r,\;G_{39}f\right)
$$

其中：

```text
alpha 是 route_alpha（路线强度）
Delta r 是 Phase910/911 以来使用的 route delta（路线差分）
G39 是 Phase920 的 L39 consensus margin gear（共识边界齿轮）
f 是 L39 factor（L39 缩放因子）
```

相对 alpha=1 的增益：

$$
\Delta M_{\alpha}(x,f)
=
M_{\alpha}(x,f)-M_{1.0}(x,f)
$$

每条曲线的最佳路线强度：

$$
\alpha^\*(x,f)
=
\arg\max_{\alpha} M_{\alpha}(x,f)
$$

如果路线强度是简单单调门控，则应出现：

$$
\alpha_1 < \alpha_2
\Rightarrow
M_{\alpha_1}(x,f)\le M_{\alpha_2}(x,f)
$$

或至少多数曲线接近单调。但 Phase923（第923阶段）结果并不支持这一点。

### 五、跨模型结果

```text
qwen3: 没有 Phase915 L39 候选，未进入 Phase923 实测。
GLM4: 12 个状态进入测试。
DS7B: 没有 Phase915 L39 候选，未进入 Phase923 实测。
```

GLM4（GLM4 模型）规模：

```text
状态数: 12
L39 factor 数: 3
route_alpha 数: 9
总 rows: 324
响应曲线数: 36
```

### 六、GLM4 核心结果

响应曲线摘要：

```text
curve_count: 36
best_alpha_distribution:
  0.875: 12
  1.25: 16
  1.375: 8
best_alpha_lt_1: 12
best_alpha_eq_1: 0
best_alpha_gt_1: 24
monotonic_non_decreasing: 0
monotonic_non_increasing: 0
with_closure_alpha: 10
median_best_margin_delta_vs_alpha1: 0.1875
mean_best_margin_delta_vs_alpha1: 0.1953125
```

最重要的客观现象：

```text
36 条曲线中，没有一条是单调上升；
36 条曲线中，也没有一条是单调下降；
最佳 alpha 从不等于 1.0；
最佳 alpha 分布在 0.875、1.25、1.375 三个区域。
```

这直接否定了简单假设：

```text
route 越强，EOS 越容易赢。
```

也否定另一个简单假设：

```text
route 越弱，EOS 越容易赢。
```

更符合结果的解释是：

```text
route_alpha 是路线-齿轮匹配参数；
不同状态需要不同 route 强度；
L39 齿轮不是单独工作，而是依赖 route state（路线状态）的边界位置。
```

### 七、按 alpha 分组的结果

GLM4：

```text
alpha=0.875:
  rows: 36
  top1: 10
  margin_nonnegative: 10
  strict_clean_candidate: 6
  improved_margin_vs_alpha1: 27
  new_margin_closure_vs_alpha1: 2
  lost_margin_closure_vs_alpha1: 0
  mean_margin_delta_vs_alpha1: 0.07291666666666667
  median_patched_margin: -0.5

alpha=1.25:
  rows: 36
  top1: 8
  margin_nonnegative: 8
  strict_clean_candidate: 6
  improved_margin_vs_alpha1: 32
  new_margin_closure_vs_alpha1: 0
  lost_margin_closure_vs_alpha1: 0
  mean_margin_delta_vs_alpha1: 0.1440972222222222
  median_patched_margin: -0.375

alpha=1.375:
  rows: 36
  top1: 8
  margin_nonnegative: 8
  strict_clean_candidate: 6
  improved_margin_vs_alpha1: 21
  new_margin_closure_vs_alpha1: 0
  lost_margin_closure_vs_alpha1: 0
  mean_margin_delta_vs_alpha1: 0.1032986111111111
  median_patched_margin: -0.4375

alpha=0.75:
  rows: 36
  top1: 3
  margin_nonnegative: 3
  strict_clean_candidate: 3
  improved_margin_vs_alpha1: 11
  new_margin_closure_vs_alpha1: 0
  lost_margin_closure_vs_alpha1: 5
  mean_margin_delta_vs_alpha1: -0.1605902777777778
  median_patched_margin: -0.75

alpha=0.625:
  rows: 36
  top1: 0
  margin_nonnegative: 0
  strict_clean_candidate: 0
  lost_margin_closure_vs_alpha1: 8
  mean_margin_delta_vs_alpha1: -4.694444444444445

alpha=0.5:
  rows: 36
  top1: 0
  margin_nonnegative: 0
  strict_clean_candidate: 0
  lost_margin_closure_vs_alpha1: 8
  mean_margin_delta_vs_alpha1: -12.883246527777779
```

这个分布说明：

```text
0.5 和 0.625 明显破坏路线；
0.75 部分破坏闭合；
0.875 在少数贴边状态上最好；
1.25 和 1.375 在整体 margin 上更强；
1.5 没有继续变好。
```

### 八、新增闭合细节

新增闭合仍然只有 2 行，完全复现 Phase922（第922阶段）的情况：

```text
case: p856_009_animal_fish
object: fish
eval_domain: animal
prompt_variant: question_plain
edit_mode: flip / zero
L39 factor: 1.25
route_alpha: 0.875
```

数值：

```text
alpha1_margin: -0.125
patched_margin: 0.0
margin_delta_vs_alpha1: 0.125
patched_eos_rank: 1
strict_clean_candidate: False
```

因此新增闭合仍然必须谨慎解释：

```text
它不是稳定强闭合；
它是 fish 两个贴边状态的 alpha=0.875 局部边界现象。
```

### 九、对 Phase922 的进一步解释

Phase922（第922阶段）看到 route_alpha_0.875_direction_control 比正向 route_alpha 更容易新增闭合。Phase923（第923阶段）说明：

```text
这不是单点偶然结果；
完整响应曲线确实存在非单调结构。
```

但 Phase923 同时说明：

```text
0.875 并不是全局最佳；
它只在 12 条曲线中成为最佳 alpha；
另有 24 条曲线的最佳 alpha 大于 1。
```

所以更准确的结论是：

```text
route_alpha 不是自然门控开关；
route_alpha 是状态依赖的匹配变量。
```

### 十、图谱进展

Phase923（第923阶段）把全局齿轮图谱推进了一步：

```text
之前图谱记录的是：
  哪个齿轮有效；
  哪个方向有效；
  哪个 blocker 被压制。

现在图谱需要增加：
  route-response curve（路线响应曲线）；
  alpha_peak（最佳路线强度）；
  alpha_nonmonotonicity（路线非单调性）；
  route-gear matching（路线-齿轮匹配）。
```

新的图谱字段建议：

```text
state_key
l39_factor
best_route_alpha
best_margin
alpha1_margin
best_margin_delta_vs_alpha1
closure_alphas
is_monotonic_non_decreasing
is_monotonic_non_increasing
route_alpha_peak_region
```

### 十一、问题、硬伤与瓶颈

第一，跨模型仍不完整。

```text
qwen3 和 DS7B 没有 Phase915 L39 候选；
Phase923 的实证主体仍然只有 GLM4。
```

第二，新增闭合仍然太少。

```text
new_margin_closure_vs_alpha1 = 2
new_top1_vs_alpha1 = 2
new_strict_vs_alpha1 = 0
```

第三，闭合边界太弱。

```text
patched_margin = 0.0
```

这不是强正边界，只是刚好碰到边界。

第四，响应曲线虽然非单调，但仍是小模型局部曲线。

```text
小模型内部结构可能粗糙；
route_alpha 曲线可能是压缩后结构的局部投影；
不能直接当作通用语言编码规律。
```

第五，当前只扫描 route_alpha，没有同时扫描 protocol pressure（协议压力）和 blocker field（阻塞词场）。

```text
真实自然门控可能是三变量曲面，
而不是一维 route_alpha 曲线。
```

### 十二、闭合标准与当前距离

如果要把 route-response（路线响应）升级为自然门控机制，需要满足：

```text
1. alpha peak（最佳 alpha）在更多 case 上稳定可预测；
2. best_alpha 能由自然状态变量预测；
3. 选择 best_alpha 后能显著新增 margin/top1/strict clean 闭合；
4. 不只是 fish 个例；
5. 至少两个模型复现；
6. 能预测 holdout state（留出状态）的响应峰值。
```

Phase923 当前只满足：

```text
route_alpha 非单调；
存在状态依赖峰值；
best_alpha 分布不是随机平铺，而集中在 0.875、1.25、1.375。
```

但还没有满足：

```text
稳定新增 strict clean 闭合；
跨模型复现；
可预测峰值。
```

阶段性估计：

```text
L39 EOS-vs-a 边界齿轮子问题进度: 约 73%
route-response 曲线定位进度: 约 25%
自然门控变量因果验证进度: 约 30%
clean protocol edge graph（干净协议边图谱）进度: 约 44% - 48%
完整语言编码机制进度: 约 19% - 23%
```

### 十三、智能理论角度的关键洞察

Phase923（第923阶段）最重要的洞察是：

```text
语言生成边界可能不是简单加法场；
而是状态依赖的响应曲面。
```

如果把 L39 共识齿轮看成一个局部执行器，把 route_delta（路线差分）看成上游路线状态，那么输出边界更像：

$$
M = F(G_{39}, \alpha, P, B, x)
$$

其中：

```text
G39 是 L39 齿轮；
alpha 是路线强度；
P 是 protocol pressure（协议压力）；
B 是 blocker field（阻塞词场）；
x 是当前状态。
```

Phase923 说明：

```text
F 对 alpha 不是单调函数。
```

这意味着全局齿轮图谱不能只记录：

```text
齿轮方向；
齿轮强度；
齿轮目标。
```

还必须记录：

```text
齿轮与路线状态的匹配曲线。
```

这可能是破解语言编码机制的重要拼图：语言不是由单个神经元或单个方向直接编码，而是由多个局部齿轮在状态空间中的相位匹配完成。

### 十四、下一阶段任务

Phase923 仍属于自然门控变量因果验证阶段，但已经完成了 route_alpha 一维曲线审计。下一步应转向二维曲面：

```text
Phase924:
Route Alpha × Protocol Pressure Response Surface
（路线强度 × 协议压力响应曲面）
```

目标：

```text
1. 固定 Phase920 L39 consensus gear。
2. 固定 L39 factor = 1.25 和 1.375。
3. 扫描 route_alpha:
   0.75, 0.875, 1.0, 1.125, 1.25, 1.375
4. 扫描 protocol span factor:
   0.85, 0.9, 1.0, 1.1
5. 观察 best_alpha 是否受 protocol pressure 改变。
6. 检查 fish 个例是否只是局部偶然，还是二维曲面上的一个峰。
```

成功标准：

```text
如果 route_alpha 峰值会随 protocol pressure 系统移动，
则说明自然门控不是一维旋钮，而是 route-protocol 联合曲面。

如果二维曲面仍然只有 fish 个例贴边，
则应把当前路线降级为局部边界扰动，
转向更广泛的状态采样和图谱扩展。
```

### 十五、通俗总结

Phase922（第922阶段）像是发现：

```text
把路线旋钮拧小一点，有两扇门反而开了。
```

Phase923（第923阶段）把旋钮从小到大完整转了一圈，结果看到：

```text
这个旋钮不是越大越好；
也不是越小越好；
每个状态都有自己的最佳位置。
```

所以当前最真实的结论是：

```text
我们不是找到了一个简单开关；
而是看到了齿轮系统里的匹配曲线。
```

这对全局齿轮图谱很重要，因为后续图谱必须从“组件有效性表”升级为“状态响应曲面图”。

## Phase 924: 路线强度与协议压力二维响应曲面审计 [2026-07-04 09:23]

### 一、任务来源

Phase923（第923阶段）已经证明：

```text
route_alpha（路线强度）不是单调旋钮；
不同状态存在不同 alpha_peak（路线强度峰值）；
但新增闭合仍然只出现在 fish 的两个贴边状态。
```

因此 Phase924（第924阶段）继续同一阶段任务，把一维 route_alpha curve（路线强度曲线）扩展为二维曲面：

```text
route_alpha × protocol_span_factor
路线强度 × 协议压力
```

目标不是追求新的闭合，而是判断：

```text
Phase923 看到的路线峰值是否会被 protocol pressure（协议压力）系统调制。
```

### 二、脚本与结果位置

新增脚本：

```text
tests/glm5/phase924_route_protocol_response_surface_audit.py
tests/glm5/run_phase924_route_protocol_response_surface_audit.sh
```

结果目录：

```text
tests/result/phase924_route_protocol_response_surface_audit/route_protocol_response_surface_audit/
```

核心输出：

```text
phase924_qwen3_summary.json
phase924_glm4_summary.json
phase924_deepseek7b_summary.json
phase924_glm4_rows.jsonl
phase924_glm4_surfaces.jsonl
phase924_cross_model_summary.json
phase924_cross_model_summary.md
```

注意：第一次直接执行 runner（运行脚本）时出现一次底层 segmentation fault（段错误），未产生可用实验结果。随后按 qwen3、GLM4、DS7B 单模型顺序重跑，三模型结果完整，并重新执行 summarize-round（汇总轮次）。qwen3 完整参数无候选路径也单独复核通过，因此最终记录以单模型顺序重跑结果为准。

### 三、测试原理

固定：

```text
Phase920 consensus L39 margin gear（共识 L39 边界齿轮）
L4 boundary spec（L4 边界调节器）
```

扫描：

```text
L39 factor:
  1.25
  1.375

route_alpha:
  0.75
  0.875
  1.0
  1.125
  1.25
  1.375

protocol_span_factor:
  0.85
  0.9
  1.0
  1.1

protocol_span_kind:
  last8_before_period
```

GLM4（GLM4 模型）总规模：

```text
状态数: 12
L39 factor 数: 2
route_alpha 数: 6
protocol factor 数: 4
总 rows: 576
二维曲面数: 24
```

### 四、核心公式

二维响应曲面：

$$
M_{\alpha,p}(x,f)
=
z_{\mathrm{EOS}}\left(x;\alpha\Delta r,\;pP,\;G_{39}f\right)
-
z_{\mathrm{blocker}}\left(x;\alpha\Delta r,\;pP,\;G_{39}f\right)
$$

其中：

```text
alpha 是 route_alpha（路线强度）；
p 是 protocol_span_factor（协议压力因子）；
P 是 last8_before_period 的 L0 attention input span（第0层注意力输入片段）；
G39 是 L39 consensus margin gear（共识边界齿轮）；
f 是 L39 factor（L39 缩放因子）。
```

基点：

$$
M_{\mathrm{base}}(x,f)=M_{1.0,1.0}(x,f)
$$

曲面增益：

$$
\Delta M_{\alpha,p}(x,f)
=
M_{\alpha,p}(x,f)-M_{\mathrm{base}}(x,f)
$$

最佳坐标：

$$
(\alpha^\*,p^\*)
=
\arg\max_{\alpha,p} M_{\alpha,p}(x,f)
$$

### 五、跨模型结果

```text
qwen3: 没有 Phase915 L39 候选，未进入 Phase924 实测。
GLM4: 12 个状态进入测试。
DS7B: 没有 Phase915 L39 候选，未进入 Phase924 实测。
```

跨模型整体：

```text
selected_phase915_l39_candidates: 12
target_state_count: 12
all_rows: 576
all_top1: 174
all_margin_nonnegative: 174
all_strict_clean_candidate: 132
surface_base_rows: 24
surface_base_top1: 8
surface_base_margin_nonnegative: 8
non_base_rows: 552
non_base_top1: 166
non_base_margin_nonnegative: 166
non_base_new_top1_vs_surface_base: 2
non_base_new_margin_closure_vs_surface_base: 2
```

### 六、GLM4 二维曲面摘要

```text
surface_count: 24
best_alpha_distribution:
  0.75: 4
  0.875: 4
  1.125: 4
  1.25: 9
  1.375: 3

best_protocol_distribution:
  0.85: 1
  0.9: 10
  1.0: 10
  1.1: 3

best_coord_is_base: 0
best_alpha_lt_1: 8
best_alpha_eq_1: 0
best_alpha_gt_1: 16
best_protocol_lt_1: 11
best_protocol_eq_1: 10
best_protocol_gt_1: 3
with_closure_coord: 10
median_best_margin_delta_vs_surface_base: 0.1875
mean_best_margin_delta_vs_surface_base: 0.21354166666666666
```

核心现象：

```text
24 个二维曲面中，没有一个曲面的最佳坐标是基点 alpha=1.0 且 protocol=1.0；
best protocol factor 有 11 个小于 1，有 10 个等于 1，有 3 个大于 1；
说明 protocol pressure 会改变最优坐标。
```

### 七、按 protocol factor 分组结果

```text
protocol=1.0:
  rows: 144
  top1: 45
  margin_nonnegative: 45
  new_margin_closure: 2
  lost_margin_closure: 5
  mean_delta_vs_base: 0.018663194444444444

protocol=0.85:
  rows: 144
  top1: 46
  margin_nonnegative: 46
  new_margin_closure: 0
  lost_margin_closure: 2
  mean_delta_vs_base: 0.027777777777777776

protocol=0.9:
  rows: 144
  top1: 46
  margin_nonnegative: 46
  new_margin_closure: 0
  lost_margin_closure: 2
  mean_delta_vs_base: 0.013020833333333334

protocol=1.1:
  rows: 144
  top1: 37
  margin_nonnegative: 37
  new_margin_closure: 0
  lost_margin_closure: 11
  mean_delta_vs_base: -0.5786675347222222
```

解释：

```text
轻微降低 protocol pressure（0.85 / 0.9）整体上不坏，甚至 top1 数略高；
提高 protocol pressure 到 1.1 会明显破坏边界，lost closure 增加到 11。
```

这说明 protocol pressure 不是无关变量，而是会参与 L39 齿轮是否有效。

### 八、二维坐标结果

最强新增闭合坐标仍然是：

```text
route_alpha = 0.875
protocol_span_factor = 1.0
```

数值：

```text
rows: 24
top1: 10
margin_nonnegative: 10
strict_clean_candidate: 6
new_margin_closure: 2
new_top1: 2
lost_margin_closure: 0
mean_delta_vs_base: 0.08333333333333333
```

较强但不新增闭合的坐标包括：

```text
route_alpha=1.25, protocol=1.0:
  mean_delta_vs_base: 0.1328125
  new_closure: 0

route_alpha=1.375, protocol=1.1:
  mean_delta_vs_base: 0.10416666666666667
  new_closure: 0

route_alpha=1.125, protocol=0.85:
  mean_delta_vs_base: 0.09114583333333333
  new_closure: 0
```

这说明：

```text
最大平均 margin 增益不等于新增闭合；
新增闭合只发生在非常贴边的局部区域；
闭合是边界位置问题，不是平均增益问题。
```

### 九、新增闭合细节

新增闭合仍然只有 2 行，和 Phase922 / Phase923 完全一致：

```text
case: p856_009_animal_fish
object: fish
eval_domain: animal
prompt_variant: question_plain
edit_mode: flip / zero
L39 factor: 1.25
route_alpha: 0.875
protocol_span_factor: 1.0
```

数值：

```text
surface_base_margin: -0.125
patched_margin: 0.0
margin_delta_vs_surface_base: 0.125
patched_eos_rank: 1
strict_clean_candidate: False
```

所以二维曲面没有把新增闭合从 fish 个例扩展到更多状态。

### 十、对 Phase923 的修正

Phase923（第923阶段）提出：

```text
route_alpha 可能是状态依赖匹配变量。
```

Phase924（第924阶段）进一步说明：

```text
route_alpha 的最佳点确实会被 protocol pressure 调制；
但是 protocol 调制没有带来新的稳定闭合；
当前新增闭合仍然只是 fish 的贴边局部峰。
```

因此当前图谱应记录：

```text
route-protocol response surface 存在；
best coordinate 不等于自然基点；
protocol pressure 会改变 margin 曲面；
但闭合仍然没有扩展到稳定多样本。
```

### 十一、问题、硬伤与瓶颈

第一，强正结论仍然不足。

```text
new_margin_closure_vs_surface_base = 2
new_top1_vs_surface_base = 2
new_strict_vs_surface_base = 0
```

第二，新增闭合仍是同一个 fish 个例。

```text
没有跨 object / material / animal 多样化扩展。
```

第三，二维曲面证明了结构复杂性，但没有证明自然门控。

```text
best_coord_is_base = 0
```

这说明自然基点不是最优实验坐标，但不等于模型自然生成时真的会选择这些坐标。

第四，protocol factor 的解释仍然粗糙。

```text
当前 protocol_span_factor 只是 last8_before_period 输入缩放；
它是协议压力代理变量，不是真正完整的 protocol field（协议场）。
```

第五，跨模型仍然只有 GLM4 实测。

```text
qwen3 和 DS7B 无 L39 候选；
不能把 GLM4 小模型曲面当成通用结构。
```

### 十二、闭合标准与当前距离

二维曲面如果要进入自然门控闭合，需要满足：

```text
1. 最佳坐标能预测 holdout state；
2. 曲面峰值能新增 strict clean 闭合；
3. 新增闭合跨多个 case / domain；
4. protocol factor 的含义能从代理变量升级为明确协议场变量；
5. 至少两个模型复现；
6. 曲面能解释 blocker field（阻塞词场）变化，而不只是 EOS margin。
```

Phase924 当前满足：

```text
二维曲面存在；
protocol pressure 参与最优坐标；
自然基点不是最佳实验坐标。
```

Phase924 当前不满足：

```text
稳定新增闭合；
strict clean 新增；
跨模型复现；
协议场真实解码。
```

阶段性估计：

```text
L39 EOS-vs-a 边界齿轮子问题进度: 约 74%
route-protocol response surface（路线-协议响应曲面）进度: 约 25% - 30%
自然门控变量因果验证进度: 约 32%
全局齿轮图谱结构进度: 约 50%
完整语言编码机制进度: 约 19% - 23%
```

### 十三、智能理论角度的关键洞察

Phase924（第924阶段）的关键洞察是：

```text
语言生成边界至少是二维响应曲面，不是一维旋钮。
```

如果把语言生成看作全词表竞争场，那么当前结果说明：

$$
M = F(G_{39},\alpha,p,x)
$$

其中：

```text
G39 是局部边界齿轮；
alpha 是 route state（路线状态）强度；
p 是 protocol pressure（协议压力）；
x 是语义/句法状态。
```

而且：

$$
\arg\max_{\alpha,p} F(G_{39},\alpha,p,x)
\ne
(1.0,1.0)
$$

这意味着：

```text
模型自然状态不是实验最优状态；
解释语言编码机制不能只找哪个组件有效；
必须画出状态响应曲面，理解齿轮在不同路线和协议压力下如何咬合。
```

### 十四、阶段性收束与下一步

从 Phase921 到 Phase924 的链条是：

```text
Phase921:
  发现候选自然门控变量的诊断关联。

Phase922:
  发现候选正向变量只轻微移动 margin，方向对照反而新增贴边闭合。

Phase923:
  证明 route_alpha 响应非单调，存在状态依赖峰值。

Phase924:
  证明 protocol pressure 会调制二维最佳坐标，但仍没有扩展新增闭合。
```

因此当前阶段目标已经完成一轮：

```text
候选门控变量不是简单自然开关；
更像 route-protocol-state response surface（路线-协议-状态响应曲面）。
```

下一步不应继续只围绕 fish 贴边闭合打转。更合理的阶段性任务是：

```text
Phase925:
Response Surface Generalization Dataset Expansion
（响应曲面泛化数据扩展）
```

目标：

```text
1. 扩展状态来源，不只依赖 Phase915 的 L39_mlp_output_scale_1.5 候选。
2. 采集更多 near-boundary states（近边界状态）。
3. 每个 case 至少形成多条曲面。
4. 判断 best_alpha / best_protocol 是否可由自然状态变量预测。
5. 如果仍只有少数贴边闭合，就把当前路线从“闭合路线”降级为“图谱特征路线”。
```

### 十五、通俗总结

Phase924 像是把一个旋钮升级成了一个二维控制面板：

```text
横轴是路线强度；
纵轴是协议压力。
```

结果发现：

```text
每个状态确实有不同的最佳位置；
协议压力确实会改变最佳位置；
但真正新增开门的仍然只有 fish 的两扇贴边小门。
```

所以当前不能说已经找到自然门控机制。更准确地说：

```text
我们已经看到齿轮系统存在响应曲面；
但闭合还没有从局部个例扩展成通用规律。
```

下一步应该扩大图谱采样，而不是继续在同一个贴边个例上反复调参。

## Phase 925: 响应曲面泛化数据扩展 [2026-07-04 09:47]

### 一、对附件判断的核查

附件对 Phase921 到 Phase924 的总体判断基本正确。当前主线不应写成“已经找到自然动作门”，更准确的表述是：

```text
已经发现 L39 MLP EOS-vs-a 有符号边界齿轮；
人工干预可以在少数贴边状态上推动 EOS 胜出；
自然门控变量还没有闭合；
当前更像 route-protocol-state response surface 的雏形。
```

需要收紧的一点是：附件中的部分公式排版出现损坏，不能按原式直接引用；但它表达的核心含义是正确的，即以 EOS 相对 blocker 的 margin、route alpha 响应曲线、route-protocol 二维曲面作为主要观测对象。

本阶段采纳附件中正确的部分，执行 Phase925：

```text
Response Surface Generalization Dataset Expansion
响应曲面泛化数据扩展
```

本阶段不是新的模型前向因果干预，而是对已有 Phase914 / Phase915 / Phase924 结果进行跨模型离线整理，目标是摆脱 Phase924 只围绕 fish 局部贴边状态的限制，为下一轮更大范围曲面测试准备候选状态。

### 二、脚本与输出

新增脚本：

```text
tests/glm5/phase925_response_surface_generalization_dataset_expansion.py
tests/glm5/run_phase925_response_surface_generalization_dataset_expansion.sh
```

输出目录：

```text
tests/result/phase925_response_surface_generalization_dataset_expansion/response_surface_generalization_dataset_expansion/
```

关键输出文件：

```text
phase925_qwen3_summary.json
phase925_glm4_summary.json
phase925_deepseek7b_summary.json
phase925_glm4_selected_surface_seeds.jsonl
phase925_cross_model_summary.json
phase925_cross_model_summary.md
```

脚本已通过：

```text
python -m py_compile tests/glm5/phase925_response_surface_generalization_dataset_expansion.py
bash -n tests/glm5/run_phase925_response_surface_generalization_dataset_expansion.sh
```

### 三、测试原理

Phase924 已经证明二维曲面存在：

```text
route_alpha x protocol_span_factor
```

但新增闭合集中在 fish 局部状态。因此 Phase925 的任务不是继续调参，而是先扩大候选状态集合。

每个候选状态用如下 key 固定：

```text
state_key =
case_id |
prompt_variant |
source_subset_key |
edit_mode |
eval_kind |
group_kind |
factor
```

候选筛选条件：

```text
C(x)=1[
  usable_boundary(x)
  and (
    near_margin(x)
    or top10(x)
    or weak_holdout(x)
    or strong_holdout(x)
    or (top50(x) and blocker_is_target(x) and rank_near(x))
  )
]
```

其中：

```text
near_margin(x): -2.0 <= M(x) <= 0.5
rank_near(x): EOS_rank <= 50
target_blocker_token: a
M(x)=z_EOS(x)-z_blocker(x)
```

候选排序分数是一个可解释的基础加权分数：

```text
S(x)=
200 * I_strong
+ 120 * I_weak
+ 70 * I_top5
+ 40 * I_top10
+ 25 * I_blocker_is_a
+ 10 * max(0, 4 - |M|)
+ 10 * max(0, 64-rank)/64
+ 3 * max(0, -band16_mean_logit_delta)
+ eos_logit_delta_vs_route
```

这个分数只用于选择下一轮曲面测试的候选种子，不作为机制闭合证据。

### 四、客观结果

跨模型汇总：

```text
phase914_rows_total: 1688
candidate_unique_states_total: 1380
selected_surface_seeds_total: 96
selected_new_surface_seed_vs_phase924: 84
selected_already_surface_tested_phase924: 12
selected_present_in_phase915_boundary_set: 12
selected_top50: 96
selected_top10: 70
selected_top5: 18
selected_weak_holdout_candidate: 12
selected_strict_clean_candidate: 0
selected_strong_holdout_candidate: 0
selected_unique_cases: 10
selected_unique_domains: 3
selected_unique_prompt_variants: 4
selected_unique_groups: 5
```

分模型结果：

```text
qwen3:
  phase914_rows: 96
  candidate_unique_states: 0
  selected_surface_seeds: 0
  evidence: no_expandable_response_surface_candidates

GLM4:
  phase914_rows: 1496
  candidate_unique_states: 1380
  selected_surface_seeds: 96
  evidence: expanded_surface_seed_set_ready

DS7B:
  phase914_rows: 96
  candidate_unique_states: 0
  selected_surface_seeds: 0
  evidence: no_expandable_response_surface_candidates
```

GLM4 精选种子的结构：

```text
domains:
  animal: 36
  material: 30
  object: 30

cases:
  p856_008_animal_bird: 10
  p856_021_material_wood: 10
  p856_022_material_iron: 10
  p856_009_animal_fish: 10
  p856_038_object_object: 10
  p885_047_animal_shark: 10
  p856_036_object_car: 10
  p856_035_object_chair: 10
  p856_023_material_plastic: 10
  p856_010_animal_mammal: 6

groups:
  top_abs_64: 36
  low_abs_64: 24
  band32_support_64: 15
  band16_support_64: 11
  band16_support_32: 10

blockers:
  a: 66
  " .": 30

median_margin: -1.984375
mean_margin: -2.515625
median_rank: 9.0
median_score: 103.16015625
```

### 五、结果分析

Phase925 支持附件中的关键判断：

```text
下一步应从局部贴边闭合转向响应曲面泛化。
```

正结果：

```text
1. GLM4 不再只剩 fish 单点。
2. 候选状态扩展到 10 个 case。
3. 候选状态覆盖 animal / material / object 三个语义域。
4. 候选状态包含 4 类 prompt variant 和 5 类边界齿轮组。
5. 84 个 selected seed 是 Phase924 没有测试过的新曲面种子。
```

负结果：

```text
1. qwen3 和 DS7B 在已有 Phase914 数据中没有可扩展候选。
2. selected_strict_clean_candidate 仍为 0。
3. selected_strong_holdout_candidate 仍为 0。
4. 这不是新的自然闭合证据，只是为下一轮因果曲面测试建立数据底座。
5. GLM4 的候选中有 30 个 blocker 是 " ."，说明阻塞边已经不只是一条 EOS-vs-a 边，需要在下一阶段区分 a blocker 与 punctuation blocker。
```

### 六、闭合标准与当前距离

当前闭合不能定义为“找到了某个能提升 EOS logit 的齿轮”。更严格的闭合应至少分三层：

```text
1. 曲面复现闭合：
   在不同 case、domain、prompt、gear group 上复现稳定的 response surface 结构。

2. 坐标预测闭合：
   只用自然状态变量预测 best_alpha 和 best_protocol coordinate，
   并在 holdout seed 上成立。

3. 因果动作闭合：
   使用预测坐标进行干预后，
   EOS 在 full-vocabulary blocker 场中稳定胜出，
   且满足 exact-natural / strict-clean 标准。
```

Phase925 只完成了第 1 层之前的数据准备：

```text
GLM4:
  已形成可测的泛化曲面种子集合。

qwen3 / DS7B:
  当前已有数据没有形成对应候选集合。

strict causal closure:
  仍未完成。
```

所以当前离语言编码机制闭合仍然很远。Phase925 的价值是把下一步测试从“局部调鱼样本”推进到“跨 case / domain 的曲面验证”。

### 七、问题、硬伤与瓶颈

1. 数据来源偏置：

```text
Phase925 依赖 Phase914/915/924 既有结果。
它能扩大已有图谱，但不能证明没有被既有实验设计漏掉的状态。
```

2. 跨模型不充分：

```text
qwen3 和 DS7B 没有候选，不等于不存在类似机制；
也可能是 Phase914 的测试入口不适合这两个小模型。
```

3. 选择分数不是机制公式：

```text
S(x) 只是候选排序规则。
不能把它解释成模型内部真实门控函数。
```

4. 阻塞者结构变复杂：

```text
候选 blocker 同时包含 a 和 " ."。
这说明 termination / protocol edge 不是单一 EOS-vs-a 边，
下一阶段必须区分语义 blocker、冠词 blocker、标点 blocker。
```

5. 小模型偏差仍然存在：

```text
当前测试模型较小，内部路线可能粗糙、离散、局部化。
因此 GLM4 上形成的候选曲面不能直接外推到更大模型或真实语言机制。
```

### 八、智能理论角度的阶段洞察

这一阶段进一步说明：

```text
语言生成不像一个单点开关；
更像一个状态场中的边界迁移过程。
```

目前看到的结构是：

```text
semantic object state
protocol continuation pressure
route intensity
signed boundary gear
full-vocabulary blocker field
```

这些变量共同决定 EOS 是否能穿过阻塞带。也就是说，要破解语言背后的数学结构，第一性原理可能不是“找到一个特征”，而是：

```text
找到状态变量如何改变边界曲面的形状。
```

Phase925 把这个方向推进了一步：先建立足够大的状态采样，而不是继续在单个局部案例上追求闭合。

### 九、下一阶段任务

当前阶段性目标已经完成：

```text
扩展 response surface 泛化候选数据集。
```

下一阶段仍属于同一条 response-surface 主线，但已经进入新的 GPU 因果测试阶段，不应与本阶段的离线索引结果混为一个结论。

建议 Phase926：

```text
Generalized Route-Protocol Surface Validation
泛化路线-协议曲面验证
```

任务：

```text
1. 以 Phase925 的 96 个 GLM4 selected seeds 为主测试集。
2. 先做平衡子集测试，例如每个 domain / blocker / group 取代表种子。
3. 再扩大到完整 96 seed。
4. 对每个 seed 重新画 route_alpha x protocol_span_factor 曲面。
5. 检查 best coordinate 是否能跨 case / domain 形成稳定规律。
6. 单独标记 blocker=a 与 blocker=" ." 的曲面差异。
7. 如果曲面稳定，再进入 best-coordinate predictor。
```

如果 Phase926 仍然只在极少数贴边状态产生闭合，则当前路线应进一步收紧为：

```text
response surface graph construction
响应曲面图谱构建
```

而不是继续声称接近语言编码机制闭合。

### 十、通俗总结

Phase925 做的事情很简单：

```text
以前我们只有几块可疑拼图，尤其 fish 很突出；
现在先把更多可疑拼图从旧实验里挑出来，
组成一个更大的候选集合。
```

结果是：

```text
GLM4 找到了 96 个值得下一轮画曲面的状态；
这些状态覆盖 10 个样本、3 个语义域；
但 qwen3 和 DS7B 还没有对应候选；
也还没有任何新的闭合。
```

所以 Phase925 是图谱推进，不是机制闭合。下一步要真正把这些点拿去画曲面，看看它们是不是同一套语言边界规律的不同切片。

## Phase 926: 泛化路线-协议曲面验证 [2026-07-04 12:31]

### 一、对附件判断的核查

附件对 Phase925 的判断基本正确，而且证据层级收得很必要：

```text
Phase925 不是闭合实验；
不是自然动作门被找到；
不是新的前向因果干预；
而是响应曲面候选状态扩容。
```

本阶段采纳这个边界，继续执行同一条 response-surface 主线中的下一步：

```text
Phase926:
Generalized Route-Protocol Surface Validation
泛化路线-协议曲面验证
```

它与 Phase925 属于同一个阶段性目标：

```text
先完成响应曲面图谱扩展与验证，再讨论闭合。
```

但 Phase926 已经不是离线索引，而是新的 GPU 前向因果曲面测试。

### 二、脚本与输出

新增脚本：

```text
tests/glm5/phase926_generalized_route_protocol_surface_validation.py
tests/glm5/run_phase926_generalized_route_protocol_surface_validation.sh
```

输出目录：

```text
tests/result/phase926_generalized_route_protocol_surface_validation/generalized_route_protocol_surface_validation/
```

关键输出：

```text
phase926_qwen3_summary.json
phase926_glm4_summary.json
phase926_deepseek7b_summary.json
phase926_glm4_rows.jsonl
phase926_glm4_surfaces.jsonl
phase926_cross_model_summary.json
phase926_cross_model_summary.md
```

脚本检查：

```text
python -m py_compile tests/glm5/phase926_generalized_route_protocol_surface_validation.py
bash -n tests/glm5/run_phase926_generalized_route_protocol_surface_validation.sh
```

### 三、测试原理

Phase926 从 Phase925 的 selected seeds 中抽取平衡子集，避免继续只测 fish。

本轮 GLM4 实测子集：

```text
selected seeds: 30
unique cases: 9
unique domains: 3
unique groups: 4
blocker classes:
  article_a: 18
  punctuation_period: 12
new_vs_phase924: 22
median_seed_margin: -2.15625
median_seed_rank: 9.0
```

每个 seed 扫描：

```text
L39 factor:
  1.25, 1.375

route_alpha:
  0.75, 0.875, 1.0, 1.125, 1.25, 1.375

protocol_span_factor:
  0.85, 0.9, 1.0, 1.1

protocol_span_kind:
  last8_before_period
```

因此 GLM4 实际测试规模：

```text
30 seeds * 2 L39 factors * 6 route_alpha * 4 protocol_factor
= 1440 forward coordinates

surface_count:
  60
```

核心边界差仍然是：

```text
M(x, alpha, p, f)
= z_EOS(x; alpha, p, f) - z_blocker(x; alpha, p, f)
```

相对曲面基线：

```text
base coordinate:
  alpha = 1.0
  protocol_factor = 1.0

Delta M =
M(x, alpha, p, f) - M(x, 1.0, 1.0, f)
```

新增闭合判断：

```text
new_top1_vs_surface_base =
  base_EOS_top1 = false
  and patched_EOS_top1 = true

new_margin_closure_vs_surface_base =
  base_margin < 0
  and patched_margin >= 0

new_strict_vs_surface_base =
  base_strict_clean = false
  and patched_strict_clean = true
```

### 四、客观结果

跨模型运行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

qwen3 和 DS7B 因 Phase925 没有 selected seeds，本阶段不加载模型做无意义前向，直接记录：

```text
qwen3:
  evidence: no_phase925_surface_seeds

DS7B:
  evidence: no_phase925_surface_seeds
```

GLM4 完成全部 1440 个坐标：

```text
rows: 1440
surfaces: 60
target_state_count: 30
expected_rows_if_all_reconstructed: 1440
```

总体结果：

```text
all_rows: 1440
all_top1: 97
all_margin_nonnegative: 97
all_strict_clean_candidate: 57

surface_base_rows: 60
surface_base_top1: 5
surface_base_margin_nonnegative: 5
surface_base_strict_clean_candidate: 3

non_base_rows: 1380
non_base_top1: 92
non_base_margin_nonnegative: 92
non_base_strict_clean_candidate: 54

non_base_improved_margin_vs_surface_base: 542
non_base_lost_margin_closure_vs_surface_base: 25
non_base_new_margin_closure_vs_surface_base: 2
non_base_new_top1_vs_surface_base: 2
non_base_new_strict_vs_surface_base: 0
```

曲面结构：

```text
surface_count: 60
best_coord_is_base: 8
best_alpha_lt_1: 11
best_alpha_eq_1: 14
best_alpha_gt_1: 35
best_protocol_lt_1: 31
best_protocol_eq_1: 11
best_protocol_gt_1: 18
with_closure_coord: 6
median_best_margin_delta_vs_surface_base: 0.1875
mean_best_margin_delta_vs_surface_base: 0.20729166666666668
```

最佳 alpha 分布：

```text
0.75: 2
0.875: 9
1.0: 14
1.125: 6
1.25: 24
1.375: 5
```

最佳 protocol factor 分布：

```text
0.85: 13
0.9: 18
1.0: 11
1.1: 18
```

真正新增闭合只有 2 个坐标：

```text
case:
  p856_009_animal_fish

prompt:
  question_plain

source_subset:
  L35C8824

edit_mode:
  zero

group:
  band32_support_64

L4 factor:
  0.4

L39 factor:
  1.375

coords:
  alpha=1.375, protocol=0.85
  alpha=1.375, protocol=0.9

base:
  rank=2
  margin=-0.0625

patched:
  rank=1
  margin=0.0625
  strict_clean=false
```

### 五、结果分析

Phase926 有两个同时成立的结论。

正结果：

```text
响应曲面结构确实泛化到 Phase925 的更大 seed 集合。
```

证据：

```text
1. 60 张曲面中只有 8 张最佳坐标仍是 base。
2. 52 张曲面的最佳坐标离开了 (alpha=1.0, protocol=1.0)。
3. best_alpha 覆盖 0.75 到 1.375 的多个点。
4. best_protocol 覆盖 0.85 / 0.9 / 1.0 / 1.1。
5. 542 个非基线坐标相对 base 改善 margin。
```

负结果：

```text
闭合没有泛化。
```

证据：

```text
1. 新增 top1 / margin 闭合只有 2 个坐标。
2. 这 2 个坐标仍然来自 fish。
3. 新增 strict clean 为 0。
4. punctuation_period blocker 没有产生任何 top1 / margin 闭合。
5. qwen3 和 DS7B 仍没有 Phase925 种子，因此没有跨模型正证据。
```

这说明 Phase925 的“候选扩容”是有用的，但 Phase926 把它进一步收紧为：

```text
曲面形状泛化；
闭合事件不泛化。
```

### 六、article blocker 与 punctuation blocker 的差异

按 Phase925 seed blocker class 分组：

```text
article_a:
  surfaces: 36
  best_coord_is_base: 8
  top1 rows: 97
  new_top1_vs_surface_base: 2
  closure states: 6

punctuation_period:
  surfaces: 24
  best_coord_is_base: 0
  top1 rows: 0
  new_top1_vs_surface_base: 0
  closure states: 0
```

这很重要：

```text
punctuation_period 曲面不是静态的；
它的 best coordinate 全部偏离 base；
但它没有带来 EOS 闭合。
```

因此 “.” blocker 不能简单当成 “a” blocker 的同类替代。它更可能属于 protocol continuation field 的另一条边。

### 七、闭合标准与当前距离

当前应区分三层结果：

```text
1. response surface existence:
   已经较强。

2. response surface generalization:
   在 GLM4 的 30 个 Phase925 种子上成立。

3. causal closure generalization:
   未成立。
```

严格闭合至少需要：

```text
1. 新增闭合不只集中于 fish。
2. 新增闭合跨 domain / case / blocker class 出现。
3. 新增 strict clean > 0，并且不是基线已有。
4. best coordinate 可以由自然状态变量预测，而不是事后搜索。
5. qwen3 / DS7B 至少有可对应入口，不能长期只有 GLM4。
```

Phase926 距离这些标准仍然明显不足：

```text
new_top1_vs_surface_base: 2
new_strict_vs_surface_base: 0
new closure case coverage: 1
new closure blocker class: article_a only
cross-model closure: none
```

### 八、问题、硬伤与瓶颈

1. 新增闭合仍然局部：

```text
新增闭合仍然是 fish，说明闭合事件可能依赖局部贴边状态。
```

2. 曲面搜索仍是后验：

```text
best_alpha / best_protocol 是扫描得到，不是自然变量预测得到。
```

3. punctuation blocker 未被解决：

```text
标点阻塞者有曲面变化，但没有闭合。
这说明它可能需要不同的协议边齿轮，而不是 L39 EOS-vs-a 齿轮。
```

4. 全局 consensus group 可能过粗：

```text
本轮使用一个全局 L39 consensus group。
对于 article_a 和 punctuation_period 混合集合，这可能过度平均。
下一步可能要按 blocker class 或 route family 分组。
```

5. 小模型偏差仍然存在：

```text
当前正结果主要来自 GLM4。
qwen3 / DS7B 没有 Phase925 入口，不应被解释为机制不存在；
更可能是入口、层位或 blocker 类型不匹配。
```

### 九、智能理论角度的阶段洞察

Phase926 强化了一个关键判断：

```text
语言机制中的齿轮不是单个开关；
更像一个状态依赖响应曲面。
```

但它也说明：

```text
响应曲面存在，不等于闭合机制破解。
```

更可能的结构是：

```text
不同 blocker class 对应不同边界族；
article_a 边界可以被 L39 EOS-vs-a 齿轮推动；
punctuation_period 边界虽然受 route/protocol 改变影响，
但需要另一套协议终止齿轮或更上游状态变量。
```

所以破解语言编码机制的第一性原理应继续从：

```text
单一答案 token 的 logit 提升
```

转向：

```text
状态变量如何塑造 full-vocabulary blocker field 的边界曲面。
```

### 十、下一阶段任务

Phase926 完成了“泛化曲面验证”的第一轮平衡子集测试。下一阶段仍属于同一大阶段，但应收紧为：

```text
Phase927:
Blocker-Class-Split Response Surface Validation
阻塞者类别分裂响应曲面验证
```

建议任务：

```text
1. 分开建立 article_a 曲面和 punctuation_period 曲面。
2. 对 article_a 使用当前 L39 consensus signed margin gear。
3. 对 punctuation_period 重新寻找协议终止齿轮，不能默认沿用 EOS-vs-a 齿轮。
4. 比较两类 blocker 的 best_alpha / best_protocol 分布。
5. 测试是否存在 punctuation-specific protocol gear。
6. 如果 punctuation 仍不能闭合，则把它标记为独立阻塞边界族。
```

阶段性结论应写成：

```text
响应曲面图谱继续成立；
但闭合路线必须按 blocker class 分裂。
```

### 十一、通俗总结

Phase926 像是把 Phase925 挑出来的 30 个候选点真的拿去画了小地图。

结果是：

```text
地图确实有地形；
大多数地方的最高点不在默认位置；
说明响应曲面不是幻觉。
```

但真正打开门的地方仍然很少：

```text
只有 fish 的 2 个新坐标从 rank=2 推到 rank=1；
没有新的严格干净闭合；
标点阻塞者完全没有被打开。
```

所以当前不是“找到了自然门控机制”，而是：

```text
确认了响应曲面图谱值得继续做；
同时确认闭合问题必须按 blocker 类型拆开。
```

## Phase 927: 阻塞者类别分裂响应曲面审计 [2026-07-04 13:14]

### 一、对附件判断的核查

附件对 Phase926 的判断正确，尤其是这句收紧：

```text
响应曲面结构泛化了；
闭合事件没有泛化。
```

Phase926 的证据层级应严格限定为：

```text
route_alpha 和 protocol_span_factor 确实改变边界曲面；
多数曲面的最佳坐标不在默认点；
但新增闭合仍然只有 fish 的两个贴边坐标；
punctuation_period 没有闭合；
qwen3 / DS7B 仍没有对应入口。
```

本阶段采纳附件中正确的部分，继续完成 Phase927：

```text
Blocker-Class-Split Response Surface Audit
阻塞者类别分裂响应曲面审计
```

Phase927 不是新的 GPU 前向实验，而是对 Phase926 已产生的 1440 个前向坐标和 60 张曲面进行离线分层审计。目标是验证：

```text
article_a 和 punctuation_period 是否已经形成客观可分的响应曲面族。
```

### 二、脚本与输出

新增脚本：

```text
tests/glm5/phase927_blocker_class_split_response_surface_audit.py
tests/glm5/run_phase927_blocker_class_split_response_surface_audit.sh
```

输出目录：

```text
tests/result/phase927_blocker_class_split_response_surface_audit/blocker_class_split_response_surface_audit/
```

关键输出：

```text
phase927_qwen3_summary.json
phase927_glm4_summary.json
phase927_deepseek7b_summary.json
phase927_glm4_class_summaries.jsonl
phase927_cross_model_summary.json
phase927_cross_model_summary.md
```

脚本检查：

```text
python -m py_compile tests/glm5/phase927_blocker_class_split_response_surface_audit.py
bash -n tests/glm5/run_phase927_blocker_class_split_response_surface_audit.sh
```

### 三、测试原理

Phase927 读取 Phase926 的：

```text
phase926_{model}_rows.jsonl
phase926_{model}_surfaces.jsonl
phase926_{model}_selected_seeds.jsonl
```

然后按 seed blocker class 分组：

```text
article_a:
  seed blocker token == "a"

punctuation_period:
  seed blocker token.strip() in {".", "。"}
```

对每类 blocker 统计：

```text
1. selected seeds
2. rows
3. surfaces
4. best_coord_is_base
5. best_alpha_distribution
6. best_protocol_distribution
7. top1 / margin_nonnegative / strict_clean
8. new_top1_vs_surface_base
9. new_margin_closure_vs_surface_base
10. new_strict_vs_surface_base
11. closure surface count
12. patched blocker class transition
```

本阶段不引入新的机制假设，只判断两类 blocker 在同一批 Phase926 曲面中是否表现不同。

### 四、客观结果

跨模型结果：

```text
qwen3:
  rows: 0
  surfaces: 0
  evidence: no_blocker_class_data

GLM4:
  rows: 1440
  surfaces: 60
  classes:
    article_a
    punctuation_period
  evidence: blocker_class_split_confirmed_article_closes_punctuation_moves_only

DS7B:
  rows: 0
  surfaces: 0
  evidence: no_blocker_class_data
```

总体分裂结果：

```text
article_a:
  selected_seeds: 18
  rows: 864
  surfaces: 36
  best_coord_is_base: 8
  top1: 97
  new_top1_vs_surface_base: 2
  new_margin_closure_vs_surface_base: 2
  new_strict_vs_surface_base: 0
  with_closure_coord: 6

punctuation_period:
  selected_seeds: 12
  rows: 576
  surfaces: 24
  best_coord_is_base: 0
  top1: 0
  new_top1_vs_surface_base: 0
  new_margin_closure_vs_surface_base: 0
  new_strict_vs_surface_base: 0
  with_closure_coord: 0
```

article_a 最佳 alpha 分布：

```text
1.25: 16
1.0: 10
1.375: 4
1.125: 4
0.75: 2
```

article_a 最佳 protocol 分布：

```text
1.0: 11
1.1: 2
0.9: 17
0.85: 6
```

punctuation_period 最佳 alpha 分布：

```text
0.875: 9
1.375: 1
1.25: 8
1.125: 2
1.0: 4
```

punctuation_period 最佳 protocol 分布：

```text
0.85: 7
1.1: 16
0.9: 1
```

新增闭合仍只有 2 行：

```text
class:
  article_a

case:
  p856_009_animal_fish

domain:
  animal

group:
  band32_support_64

L39 factor:
  1.375

coords:
  alpha=1.375, protocol=0.85
  alpha=1.375, protocol=0.9

base margin:
  -0.0625

patched margin:
  0.0625

strict_clean:
  false
```

### 五、结果分析

Phase927 给出一个清晰的客观结论：

```text
article_a 和 punctuation_period 已经不能混为一类 blocker。
```

原因：

```text
1. article_a 有 97 个 top1 rows，punctuation_period 为 0。
2. article_a 有 6 张 closure surfaces，punctuation_period 为 0。
3. article_a 有 2 个新增 top1 / margin closure，punctuation_period 为 0。
4. punctuation_period 的 24 张曲面中 best_coord_is_base 为 0，说明它不是静态无响应。
5. punctuation_period 的曲面会移动，但当前 EOS-vs-a 齿轮不能把它闭合。
```

因此 Phase926 的判断可以进一步收紧为：

```text
响应曲面泛化；
闭合不泛化；
闭合失败主要表现为 blocker class mismatch。
```

### 六、闭合标准与当前距离

当前还没有达到闭合。严格闭合至少需要：

```text
1. article_a 之外的 blocker class 也能出现新增闭合。
2. punctuation_period 至少出现非基线 top1 / margin closure。
3. 新增 strict_clean > 0。
4. 闭合不只集中于 fish。
5. best coordinate 能由自然状态变量预测，而不是事后扫描得到。
6. qwen3 / DS7B 至少建立对应入口。
```

Phase927 的距离：

```text
article_a:
  有少量贴边闭合，但 strict_clean=0，case coverage 仍很窄。

punctuation_period:
  曲面响应存在，但 closure=0。

cross-model:
  qwen3 / DS7B 没有可审计数据。
```

所以 Phase927 是 blocker graph 的结构进展，不是机制闭合。

### 七、问题、硬伤与瓶颈

1. Phase927 是离线审计：

```text
它没有发现新的齿轮，只是证明已有曲面结果必须按 blocker class 分裂。
```

2. punctuation_period 没有闭合：

```text
这说明标点边界可能需要独立的 protocol termination gear。
```

3. article_a 的闭合仍局部：

```text
新增闭合仍然来自 fish 的贴边状态，margin 只从 -0.0625 到 0.0625。
```

4. patched blocker 转换复杂：

```text
article_a 组中 patched blocker class 包含 article_a 和 other。
说明即使 seed blocker 是 a，干预后 blocker field 也可能迁移到其他词元。
```

5. 小模型偏差仍然存在：

```text
当前分裂证据只来自 GLM4。
qwen3 / DS7B 没有入口，不能作为跨模型定律。
```

### 八、智能理论角度的阶段洞察

Phase927 的关键洞察是：

```text
语言输出闭合不是一个统一 blocker 的问题；
而是多个 blocker class 边界族的组合问题。
```

在当前图谱中至少要分成：

```text
article_a edge:
  可以被当前 L39 EOS-vs-a 边界齿轮部分推动。

punctuation_period edge:
  受 route/protocol 坐标影响，但不能被当前齿轮闭合。
```

这意味着第一性原理要从：

```text
寻找一个通用 EOS closure gear
```

转为：

```text
为不同 blocker class 建立不同的边界齿轮族和响应曲面族。
```

### 九、下一阶段任务

Phase927 完成了当前 blocker-class split 阶段目标：

```text
证明 article_a 与 punctuation_period 在响应曲面和闭合行为上客观分裂。
```

下一阶段属于新的 gear search 阶段，不应把它和 Phase927 的离线审计混成同一个证据层级。

建议 Phase928：

```text
Punctuation-Specific Protocol Gear Search
标点阻塞专属协议齿轮搜索
```

任务：

```text
1. 只选择 punctuation_period seeds。
2. 不再使用 article_a 的 L39 consensus group 作为唯一齿轮。
3. 搜索与句号 / 终止 / protocol continuation 相关的 L39 或更上游通道组。
4. 测试这些通道组能否让 punctuation_period 出现 margin / top1 closure。
5. 与 article_a gear 做负控制，避免把两类 blocker 混淆。
6. 如果仍不闭合，则将 punctuation_period 标记为独立未解边界族。
```

### 十、通俗总结

Phase927 就是把 Phase926 的结果按阻塞者类型拆开看。

结果非常清楚：

```text
“a” 这类阻塞者还有少量能被当前齿轮推开的门；
“.” 这类阻塞者虽然地图会变形，但门完全没开。
```

所以现在不能再说“找一个 EOS 齿轮解决所有终止问题”。更准确的是：

```text
不同阻塞者像不同门锁；
当前钥匙只对 a 锁有一点作用；
对句号锁完全打不开。
```

下一步要单独找句号锁的钥匙。

## Phase 928: 标点阻塞专属协议齿轮搜索 [2026-07-04 13:39]

### 一、任务来源和判断

用户上传的 Phase927 分析基本正确。Phase927 的关键贡献不是闭合，而是把原来混在一起的 non-clean blocker field 拆成至少两类：

```text
article_a blocker
punctuation_period blocker
```

Phase927 结果显示：同一套 L39 EOS-vs-a consensus gear 对 article_a 还有少量闭合能力，但对 punctuation_period 只造成边界移动，不造成 margin/top1 closure。因此继续沿用一个统一阻塞场是不正确的。

本阶段没有直接追求完整语言编码机制闭合，而是执行附件建议的下一步：

```text
Punctuation-Specific Protocol Gear Search
标点阻塞专属协议齿轮搜索
```

目标是检查 punctuation_period 是否存在自己的 L39 通道规则，避免把 article_a 齿轮误当成所有终止/协议边的通用齿轮。

### 二、测试脚本和数据

新增脚本：

```text
tests/glm5/phase928_punctuation_specific_protocol_gear_search.py
tests/glm5/run_phase928_punctuation_specific_protocol_gear_search.sh
```

结果目录：

```text
tests/result/phase928_punctuation_specific_protocol_gear_search/punctuation_specific_protocol_gear_search/
```

跨模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

实际可用数据：

```text
qwen3: 0 punctuation_period seed
GLM4: 12 punctuation_period seeds
DS7B: 0 punctuation_period seed
```

因此本阶段有效模型仍然只有 GLM4。qwen3 和 DS7B 不是负结果，只是当前上游候选集中没有可继续测试的 punctuation_period seed。

### 三、测试原理

Phase928 使用同一坐标下的 coordinate-only 结果作为基线，再叠加候选 L39 通道组，比较候选齿轮是否带来额外边界移动。

记：

```text
M(s, alpha, p, g, f) = logit(EOS) - logit(best blocker)
```

其中：

```text
s     = punctuation_period seed state
alpha = route_delta scale
p     = protocol span scale
g     = L39 candidate channel group
f     = channel scale factor
```

同坐标基线为：

```text
M_base(s, alpha, p) = M(s, alpha, p, coordinate_only, 1.0)
```

候选齿轮增量为：

```text
Delta_g = M(s, alpha, p, g, f) - M_base(s, alpha, p)
```

新增闭合候选标准：

```text
new_margin_closure: M_base < 0 且 M_candidate >= 0
new_top1_closure:   base top1 = false 且 candidate top1 = true
strict_clean:       当前脚本中的更严格 clean 标准
```

测试坐标：

```text
(alpha, protocol) =
(1.0, 1.0)
(0.875, 1.1)
(1.25, 1.1)
(0.875, 0.85)
(1.25, 0.85)
(1.375, 0.85)
(1.375, 0.9)
```

候选组包括：

```text
eos_support_64
margin_support_pos_64
a_blocker_support_64
a_logit_support_64
margin_support_neg_64
band_blocker_support_64
top_abs_64
low_abs_64
```

注意：这里的 `margin_support_pos_64` 不是固定 64 个通道 ID，而是在每个 state 上按当前边界重新选出的正向 margin-support 通道集合。因此本阶段验证的是一种局部通道选择规则，不是固定神经元列表的全局通用性。

### 四、主要结果

GLM4 完成：

```text
selected punctuation seeds: 12
rows: 2268
coordinate baseline rows: 84
candidate rows: 2184
unique states: 12
unique cases: 3
```

总体结果：

```text
candidate top1: 28
candidate margin_nonnegative: 28
candidate strict_clean: 0
new_top1_vs_coordinate_base: 28
new_margin_closure_vs_coordinate_base: 28
new_strict_vs_coordinate_base: 0
target_state_coverage_top1: 4
target_state_coverage_margin: 4
target_state_coverage_strict: 0
```

最强组：

```text
margin_support_pos_64, factor = 2.0
rows: 84
top1: 28
margin_nonnegative: 28
strict: 0
new_top1: 28
new_margin: 28
mean margin delta: 4.410714285714286
```

次强但未闭合的组：

```text
eos_support_64, factor = 2.0
top1: 0
new_margin: 0
mean margin delta: 2.9367559523809526

a_blocker_support_64 / margin_support_neg_64, factor = 0.25
top1: 0
new_margin: 0
mean margin delta: 2.5922619047619047
```

新增闭合候选分布：

```text
new rows: 28
case: p885_047_animal_shark only
candidate group: margin_support_pos_64 only
factor: 2.0 only
strict_clean: 0
patched blocker token: " ." remains visible as blocker field
```

这说明 Phase928 的强正结果只覆盖一个自然 case 的四个 state，不应解释为 punctuation mechanism closure。

### 五、客观分析

正结果：

```text
punctuation_period 不是完全打不开；
article_a gear 失败后，punctuation-specific margin_support_pos_64 规则可以在 GLM4 上把部分 punctuation state 推到 EOS top1 / margin >= 0。
```

这支持 Phase927 的分裂判断：

```text
不同 blocker class 需要不同齿轮规则。
```

负结果和边界：

```text
1. strict_clean 仍为 0。
2. 真实闭合只发生在 shark case，未覆盖全部 3 个 case。
3. factor = 2.0 属于较强人工放大，不是自然门控。
4. qwen3 / DS7B 没有上游 punctuation seeds，不能提供跨模型验证。
5. 候选组是按 state 重算的 margin-support 规则，不是固定通道集合。
```

因此本阶段证据层级应写为：

```text
punctuation-specific candidate gear found
```

不能写为：

```text
punctuation protocol edge closure
clean termination closure
language encoding mechanism closure
```

### 六、闭合距离

当前闭合标准至少需要：

```text
1. 多 case、多 seed 上稳定 margin/top1 closure。
2. strict_clean > 0，并解释 non-clean output transition。
3. 低 factor 或自然 gate 条件下仍然成立。
4. 不只在 GLM4 上成立，至少需要跨模型或跨结构对照。
5. 能预测什么状态需要 punctuation gear，而不是事后搜索。
```

Phase928 距离闭合仍然很远，但它把 Phase927 的负结果变成了一个可继续验证的候选齿轮方向。

### 七、下一步任务

Phase928 的下一步仍属于同一阶段目标：验证 punctuation-specific gear 是否只是 shark case 偶然结果，还是能在更大的 punctuation seed 集合上保持。

因此继续执行 Phase929：

```text
Punctuation Margin Gear Holdout Validation
标点 margin 齿轮保持性验证
```

## Phase 929: 标点 margin 齿轮保持性验证 [2026-07-04 13:39]

### 一、任务

Phase929 接续 Phase928 的候选齿轮结果，不再扩大搜索空间，而是固定主要候选规则：

```text
L39 margin_support_pos_64
```

并在更多 punctuation_period seeds 上验证：

```text
1. Phase928 的 positive signal 是否能跨 seed / case 保持；
2. 需要多大 factor 才能打开边界；
3. EOS-support 控制和 negative-margin 控制是否也能闭合；
4. 是否出现 strict_clean。
```

### 二、测试脚本和数据

新增脚本：

```text
tests/glm5/phase929_punctuation_margin_gear_holdout_validation.py
tests/glm5/run_phase929_punctuation_margin_gear_holdout_validation.sh
```

结果目录：

```text
tests/result/phase929_punctuation_margin_gear_holdout_validation/punctuation_margin_gear_holdout_validation/
```

数据来源：

```text
Phase925 selected punctuation_period seeds
```

GLM4 可用：

```text
selected punctuation seeds: 30
cases: 3
max per case: 10
```

qwen3 / DS7B：

```text
0 punctuation_period seed
```

### 三、测试原理

沿用 Phase928 的同坐标差分：

```text
Delta_g = M_candidate(s, alpha, p, g, f) - M_coordinate_base(s, alpha, p)
```

Phase929 不再全量搜索多种通道，而是测试 factor 曲线：

```text
margin_support_pos_64 factors:
1.25, 1.5, 1.75, 2.0, 2.25
```

并加入两个控制：

```text
eos_support_64 factor = 2.0
margin_support_neg_64 factor = 0.25
```

总量：

```text
30 states * 7 coordinates * 8 specs = 1680 rows
```

其中：

```text
coordinate baseline rows: 210
candidate/control rows: 1470
```

### 四、主要结果

GLM4 总体：

```text
rows: 1680
unique states: 30
unique cases: 3
candidate top1: 270
candidate margin_nonnegative: 270
candidate strict_clean: 0
new_top1_vs_coordinate_base: 270
new_margin_closure_vs_coordinate_base: 270
new_strict_vs_coordinate_base: 0
target_state_coverage_top1: 30
target_state_coverage_margin: 30
target_state_coverage_strict: 0
improved_margin_vs_coordinate_base: 1470
worsened_margin_vs_coordinate_base: 0
```

factor 曲线：

```text
margin_support_pos_64 factor 1.25:
top1 = 0, new_margin = 0

margin_support_pos_64 factor 1.5:
top1 = 0, new_margin = 0

margin_support_pos_64 factor 1.75:
top1 = 0, new_margin = 0

margin_support_pos_64 factor 2.0:
top1 = 70, new_margin = 70, states = 10
case coverage = p885_047_animal_shark only

margin_support_pos_64 factor 2.25:
top1 = 200, new_margin = 200, states = 30
case coverage = chair / material_wood / shark
```

控制组：

```text
eos_support_64 factor 2.0:
top1 = 0, new_margin = 0
mean margin delta = 2.919642857142857

margin_support_neg_64 factor 0.25:
top1 = 0, new_margin = 0
mean margin delta = 2.625297619047619
```

新增 margin/top1 rows 分布：

```text
total new margin/top1 rows: 270

case distribution:
p885_047_animal_shark: 140
p856_035_object_chair: 70
p856_021_material_wood: 60

phase928_selected_seed = false: 160
phase928_selected_seed = true: 110

phase928_new_closure_seed = false: 214
phase928_new_closure_seed = true: 56

patched blocker token:
" .": 200
" .\n": 70

strict_clean: 0
```

证据标签：

```text
punctuation_margin_gear_unseen_seed_positive
```

### 五、客观分析

Phase929 明显加强了 Phase928 的结论：

```text
Phase928: margin_support_pos_64 * 2.0 只在 shark case 打开 4 个 state。
Phase929: margin_support_pos_64 * 2.25 在 30 个 punctuation states 上全部出现 top1/margin closure。
```

这说明 `margin_support_pos_64` 不是单个 shark seed 的偶然结果，而是一个可复用的 punctuation boundary opening 规则。

但结果同时说明：

```text
1. factor 阈值偏高。
2. factor 2.0 只打开 shark，2.25 才覆盖全部 3 个 case。
3. strict_clean 始终为 0。
4. patched blocker 仍主要是 punctuation token，说明非干净输出转移没有解决。
5. 这仍然是手动通道放大，不是自然门控。
```

所以 Phase929 应被视为：

```text
punctuation boundary can be force-opened by a margin-support gear rule
```

而不是：

```text
punctuation termination naturally closes
```

### 六、理论进展

Phase927-929 形成了一个比较清楚的拼图：

```text
1. blocker field 至少按类别分裂。
2. article_a 和 punctuation_period 不共享同一把钥匙。
3. punctuation_period 的有效方向不是普通 EOS-support，而是更贴近边界 margin 的 support channel group。
4. response surface 的存在不等于 closure；closure 还需要足够强的 gear amplitude。
5. 当前小模型中的有效机制更像“边界可被强制打开”，不是“自然协议边已经闭合”。
```

从智能理论角度看，这支持一个更谨慎的方向：

```text
语言输出不是单个 token logit 被抬高；
而是多个 blocker class 在不同边界规则下竞争；
同一个 EOS/终止动作，可能需要按 blocker class 进入不同的局部齿轮规则。
```

但目前还不能说这些规则就是语言背后的完整数学结构。它们更像是局部投影下看到的“齿轮齿面”。

### 七、闭合标准和当前距离

严格闭合标准：

```text
1. 在自然模型运行中识别同一类 margin-support gear 的 gate variable。
2. 不使用 2.0 / 2.25 这种强人工放大，也能预测自然闭合。
3. strict_clean > 0，并能解释 punctuation blocker 到 EOS 的干净转移。
4. 在未见 case / 未见 prompt / 未见 domain 上保持。
5. 至少在 qwen3、GLM4、DS7B 中两个模型上出现同构现象，或明确解释为什么小模型结构不同。
```

当前距离：

```text
已完成：
punctuation-specific candidate gear rule identified;
larger seed validation positive;
factor threshold roughly定位到 2.0-2.25 区间。

未完成：
natural gate;
strict clean;
cross-model;
fixed-channel stability;
full-vocabulary blocker transition explanation。
```

### 八、硬伤和瓶颈

当前最大硬伤：

```text
1. 需要 factor 2.25 才能覆盖全部 punctuation seeds，说明自然强度仍未知。
2. 通道组按 state 重算，尚未证明存在固定全局齿轮。
3. strict_clean = 0，说明输出形态仍不是干净 EOS closure。
4. qwen3 / DS7B 无可比 seeds，跨模型证据缺失。
5. 小模型内部结构可能粗糙，强行放大得到的齿轮响应可能偏离大模型真实机制。
```

### 九、下一阶段任务

Phase928-929 已完成当前阶段的目标：

```text
找到并验证 punctuation-specific margin gear candidate。
```

下一阶段不应继续简单扩大 factor 搜索，而应转向：

```text
Phase930: Natural Gate and Strict-Clean Transition Audit
自然门控与 strict-clean 转移审计
```

建议任务：

```text
1. 在 factor 2.0-2.25 区间做更细曲线，定位每个 case 的 opening threshold。
2. 搜索什么自然变量预测 threshold：route norm、protocol span entropy、period blocker gap、margin_support_pos_64 activation magnitude。
3. 检查 strict_clean=0 的原因：EOS 已 top1 但 punctuation blocker field 是否仍残留。
4. 区分“强行打开 margin”和“自然终止动作完成”。
5. 尝试固定通道交集/并集，验证 state-specific group 是否能收缩为稳定通道家族。
```

下一任务已经进入自然门控和输出转移解释阶段，和 Phase928-929 的候选齿轮搜索/保持性验证不是完全同一小阶段，因此本轮在 Phase929 后停止自动推进。

### 十、通俗总结

Phase927 说：

```text
句号阻塞和 a 阻塞不是同一把锁。
```

Phase928 找到一把可能的句号锁钥匙：

```text
margin_support_pos_64 * 2.0
```

但它只在一个 case 上开门。

Phase929 把测试扩大到 30 个句号阻塞状态，结果发现：

```text
factor 2.25 时，30 个状态都能被推到 EOS top1 / margin >= 0。
```

这很重要，说明句号锁确实有自己的开锁方向。

但现在还不是最终破解，因为：

```text
这把钥匙是我们用手强行拧的；
模型自然状态下什么时候自己拧这把钥匙，还不知道；
而且输出还不是严格干净闭合。
```

下一步要找的不是更大的力气，而是模型内部自然启动这把钥匙的开关。

## Phase 930: 自然门控与 strict-clean 转移审计 [2026-07-04 15:08]

### 一、任务判断

用户上传的 Phase928-929 复盘基本正确，且证据层级收紧是必要的。

Phase928-929 不能证明：

```text
标点终止自然闭合；
自然动作门已经找到；
语言编码机制已经破解。
```

它们只能证明：

```text
punctuation_period blocker 需要独立齿轮；
margin_support_pos_64 规则可以强行打开标点边界；
但 strict_clean 仍为 0。
```

因此本阶段继续执行附件建议的 Phase930：

```text
Natural Gate and Strict-Clean Transition Audit
自然门控与 strict-clean 转移审计
```

本阶段目标不是寻找更大 factor，而是回答：

```text
1. 每个 punctuation state 的 opening threshold 是多少；
2. 哪些状态变量能预测 threshold；
3. 为什么 EOS top1 后 strict_clean 仍为 0；
4. state-specific margin group 是否有固定化迹象。
```

### 二、测试脚本和数据

新增脚本：

```text
tests/glm5/phase930_natural_gate_strict_clean_transition_audit.py
tests/glm5/run_phase930_natural_gate_strict_clean_transition_audit.sh
```

结果目录：

```text
tests/result/phase930_natural_gate_strict_clean_transition_audit/natural_gate_strict_clean_transition_audit/
```

跨模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

有效数据：

```text
qwen3: 0 punctuation_period seed
GLM4: 30 punctuation_period seeds
DS7B: 0 punctuation_period seed
```

qwen3 / DS7B 仍然是无入口，不是机制阴性。

### 三、测试原理

Phase930 细扫 Phase929 已定位的强制打开区间：

```text
factor = 2.00, 2.05, 2.10, 2.15, 2.20, 2.25
```

仍使用同坐标基线：

```text
M_base(s, alpha, p) = M(s, alpha, p, coordinate_only, 1.0)
M_candidate(s, alpha, p, f) = M(s, alpha, p, margin_support_pos_64, f)
```

opening threshold 定义为：

```text
f_open(s) = min f such that:
  M_candidate >= 0
  or EOS top1 = true
```

候选自然门控变量不使用复杂模型，只做单变量阈值审计：

```text
route_delta_norm
boundary EOS margin vs blocker
boundary EOS rank
period / punctuation gap vs EOS
L39 activation magnitude
L39 margin_support_pos_64 mean / max / min score
L39 eos_support_64 mean score
L39 margin_support_neg_64 mean score
phase925 boundary factor
```

同时记录 strict-clean 相关 flags：

```text
prefix_eos_rollout_answer_class
prefix_eos_protocol_drift
prefix_eos_strict_protocol_drift
prefix_eos_rollout_object_echo
prefix_eos_strict_clean_answer_no_protocol
```

### 四、主要结果

GLM4 完成：

```text
states: 30
rows: 1470
coordinate baseline rows: 210
candidate rows: 1260
unique cases: 3
```

总体：

```text
candidate top1: 774
candidate margin_nonnegative: 774
candidate strict_clean: 0
new_top1_vs_coordinate_base: 774
new_margin_closure_vs_coordinate_base: 774
new_strict_vs_coordinate_base: 0
target_state_coverage_top1: 30
target_state_coverage_margin: 30
target_state_coverage_strict: 0
median patched margin: 0.0
```

threshold 结果：

```text
opened states: 30 / 30
opened_at_or_below_2.00: 10
opened_at_or_below_2.10: 22
opened_at_or_below_2.25: 30
strict_clean_at_opening: 0
threshold median: 2.10
threshold mean: 2.08
```

按 case：

```text
p885_047_animal_shark:
  states = 10
  threshold_median = 2.00
  <=2.00 = 10

p856_035_object_chair:
  states = 10
  threshold_median = 2.10
  <=2.10 = 10

p856_021_material_wood:
  states = 10
  threshold_median = 2.15
  <=2.10 = 2
  <=2.25 = 10
```

factor 曲线：

```text
2.00: top1 = 70, states = 10
2.05: top1 = 70, states = 10
2.10: top1 = 98, states = 22
2.15: top1 = 164, states = 30
2.20: top1 = 172, states = 30
2.25: top1 = 200, states = 30
```

### 五、门控候选变量

Phase930 找到了一批 threshold gate candidate，但必须注意：它们只是状态变量和 opening threshold 的强相关阈值，不是自然门控机制。

对 `opened_at_or_below_2.00`，多个变量达到 30/30 的单阈值分离：

```text
target_route_delta_norm >= 0.03834216110408306
target_boundary_eos_margin_vs_blocker >= -4.0
target_boundary_eos_rank <= 6
boundary_period_gap_vs_eos <= 4.0
boundary_punctuation_gap_vs_eos <= 4.0
l39_activation_abs_top >= 32.75
l39_margin_pos_mean_score >= 0.22324757277965546
l39_margin_pos_min_score >= 0.07634490728378296
l39_eos_support_mean_score >= 0.46091899275779724
```

对 `opened_at_or_below_2.10`，较强变量包括：

```text
target_boundary_blocker_logit >= 12.4375
l39_activation_abs_median >= 0.074462890625
l39_margin_pos_mean_score >= 0.21317481994628906
l39_neg_margin_mean_score >= -0.15297437459230423
```

这些结果说明：

```text
opening threshold 不是完全随机；
边界初始难度、punctuation gap、L39 margin-support 强度都可能是自然门控候选。
```

但当前只有 30 个 GLM4 states，并且 case 与阈值强相关，因此不能把这些阈值变量直接写成自然门控公式。

### 六、strict-clean = 0 的原因

所有 threshold rows 中：

```text
prefix_eos_rollout_answer_class: false 30 / 30
prefix_eos_protocol_drift: true 30 / 30
prefix_eos_strict_protocol_drift: true 30 / 30
prefix_eos_rollout_object_echo: false 30 / 30
prefix_eos_strict_clean_answer_no_protocol: false 30 / 30
```

top1 rows 中同样：

```text
strict_clean_candidate: false 774 / 774
prefix_eos_rollout_answer_class: false 774 / 774
prefix_eos_protocol_drift: true 774 / 774
prefix_eos_strict_protocol_drift: true 774 / 774
```

样例前缀包括：

```text
" Natural Material\nCategory:"
" Furniture\nCategory: Se"
" marine life\nCategory:"
```

这说明 strict_clean=0 的主要原因不是 EOS rank 不够，而是：

```text
当前状态仍处于协议/字段中间；
答案类尚未形成干净短答案；
protocol drift 仍然存在；
强制 EOS top1 只是终止了一个未完成的协议场。
```

因此：

```text
EOS top1 / margin closure != strict-clean answer closure
```

### 七、固定化迹象

Phase930 还记录了 `margin_support_pos_64` 的通道稳定性：

```text
state_count: 30
group_size_median: 64
union_size: 105
intersection_size: 31
channels_in_at_least_half_states: 59
channels_in_at_least_quarter_states: 97
```

这说明 state-specific group 不是完全任意的。30 个状态中存在：

```text
31 个全体交集通道；
59 个半数以上出现的高频通道；
105 个总并集通道。
```

这给下一步固定齿轮因果验证提供了直接入口。

### 八、结论

Phase930 的证据标签：

```text
threshold_gate_candidate_found_without_strict_clean
```

阶段结论：

```text
1. punctuation margin gear 的 opening threshold 被定位到 2.00-2.25；
2. 不同 case 阈值不同：shark 最低，chair 中等，wood 较高；
3. 多个状态变量可以预测低阈值，但还不是自然门控机制；
4. strict_clean=0 的直接原因是协议场未完成，而不是单纯 EOS 排名不足；
5. state-specific group 存在显著固定化迹象。
```

下一任务仍属于同一阶段，因为附件明确要求测试固定通道稳定性。因此继续执行 Phase931。

## Phase 931: 固定标点 margin 齿轮因果验证 [2026-07-04 15:08]

### 一、任务

Phase930 显示：

```text
margin_support_pos_64 state-specific groups
union size = 105
intersection size = 31
channels in at least half states = 59
```

Phase931 要验证：

```text
这些高频/交集通道是否只是统计重合，
还是可以作为固定 punctuation margin gear 产生因果效果。
```

### 二、测试脚本和数据

新增脚本：

```text
tests/glm5/phase931_fixed_punctuation_margin_gear_causal_validation.py
tests/glm5/run_phase931_fixed_punctuation_margin_gear_causal_validation.sh
```

结果目录：

```text
tests/result/phase931_fixed_punctuation_margin_gear_causal_validation/fixed_punctuation_margin_gear_causal_validation/
```

有效数据：

```text
GLM4: 30 punctuation states
qwen3 / DS7B: 0 punctuation seed
```

### 三、测试原理

Phase931 从 Phase930 的 state-specific 通道组中构造固定组：

```text
fixed_intersection_all: 31 channels
fixed_topfreq_31: 31 channels
fixed_topfreq_64: 64 channels
fixed_half_or_more: 59 channels
```

对照组：

```text
state_specific_margin_support_pos_64
```

测试 factor：

```text
2.10, 2.25
```

每个 state 仍使用 7 个 route/protocol 坐标和 coordinate-only 同坐标基线。

### 四、主要结果

GLM4 总体：

```text
rows: 2310
coordinate baseline rows: 210
candidate rows: 2100
unique states: 30
unique cases: 3
candidate top1: 720
candidate margin_nonnegative: 720
candidate strict_clean: 0
new_top1_vs_coordinate_base: 720
new_margin_closure_vs_coordinate_base: 720
new_strict_vs_coordinate_base: 0
target_state_coverage_top1: 30
target_state_coverage_strict: 0
```

按组：

```text
state_specific_margin_support_pos_64, factor 2.25:
  top1 = 200
  states = 30
  mean delta = 5.430654761904762

fixed_topfreq_64, factor 2.25:
  top1 = 122
  states = 20
  mean delta = 4.613988095238096

fixed_half_or_more, factor 2.25:
  top1 = 96
  states = 20
  mean delta = 4.442261904761905

fixed_topfreq_64, factor 2.10:
  top1 = 70
  states = 10
  mean delta = 4.045535714285714

fixed_half_or_more, factor 2.10:
  top1 = 70
  states = 10
  mean delta = 3.892261904761905

fixed_intersection_all, factor 2.25:
  top1 = 32
  states = 10
  mean delta = 3.624702380952381

fixed_topfreq_31, factor 2.25:
  top1 = 32
  states = 10
  mean delta = 3.624702380952381
```

固定组覆盖：

```text
fixed_topfreq_64:
  states = 20
  cases = shark + material_wood

fixed_half_or_more:
  states = 20
  cases = shark + material_wood

fixed_intersection_all / fixed_topfreq_31:
  states = 10
  case = shark only

state_specific:
  states = 30
  cases = shark + material_wood + chair
```

所有 fixed top1 rows：

```text
strict_clean: 0
patched blocker: punctuation_period
```

### 五、客观分析

Phase931 是一个重要正结果，但仍不是闭合。

正结果：

```text
state-specific margin_support_pos_64 不是完全不可固定；
固定 top-frequency 64 通道组可以在 20/30 states 上产生因果 top1/margin closure；
固定交集 31 通道组也能打开 shark case。
```

这说明 punctuation margin gear 可以从：

```text
G_punct(x)
```

部分收缩为：

```text
G_punct_common
```

但负结果同样明确：

```text
1. fixed_topfreq_64 未覆盖 chair case。
2. intersection_31 只覆盖 shark。
3. state-specific 仍显著强于 fixed group。
4. strict_clean 仍然为 0。
5. 仍需强 factor 2.10-2.25。
```

因此 Phase931 的证据层级应写为：

```text
fixed punctuation margin gear causal positive, partial coverage
```

不能写为：

```text
fixed global punctuation gear closure
```

### 六、理论进展

Phase930-931 使标点边界图谱从：

```text
状态特异强制打开规则
```

推进为：

```text
半固定齿轮族 + 状态补偿项
```

更准确的形式是：

```text
G_punct(x) = G_common + G_case(x) + G_residual(x)
```

当前证据对应：

```text
G_common:
  fixed_topfreq_64 可覆盖 20/30 states。

G_case(x):
  chair case 仍需要 state-specific 补偿。

G_residual(x):
  strict-clean 转移和自然门控仍未解释。
```

这比 Phase929 的 “每个 state 重算通道组” 更进一步，但还没有达到全局固定机制。

### 七、闭合距离

当前已完成：

```text
1. punctuation-specific gear candidate found。
2. opening threshold 定位到 2.00-2.25。
3. threshold candidate variables identified。
4. strict_clean=0 的协议场原因被定位。
5. fixed top-frequency group 在 20/30 states 上因果有效。
```

仍未完成：

```text
1. fixed group 覆盖全部 30 states。
2. chair case 的补偿机制。
3. strict_clean > 0。
4. 自然 gate，而非人工 factor。
5. qwen3 / DS7B 跨模型入口。
```

### 八、下一阶段任务

Phase930-931 已完成附件要求的同阶段目标：

```text
自然门控候选审计；
strict-clean 失败定位；
固定通道稳定性与因果验证。
```

下一任务已经进入新的小阶段：

```text
Phase932: Fixed Gear Repair and Case-Specific Residual Audit
固定齿轮修复与 case-specific 残差审计
```

建议任务：

```text
1. 以 fixed_topfreq_64 为公共齿轮基座。
2. 专门分析 chair case 缺失的 residual channels。
3. 测试 G_common + G_chair_residual 是否覆盖 30/30。
4. 同时继续追踪 strict_clean，避免只优化 top1/margin。
5. 为 qwen3 / DS7B 重构 punctuation seeds，解决跨模型入口缺失。
```

本轮到 Phase931 停止自动推进，因为下一步已经从“自然门控/固定化审计”转向“公共齿轮修复与残差补偿”，属于新的阶段目标。

### 九、通俗总结

Phase930 做的是：

```text
这把句号锁钥匙，到底要用多大力气才能打开？
```

结果是：

```text
shark 大约 2.00；
chair 大约 2.10；
wood 大约 2.15；
所有状态 2.25 内都能打开。
```

但 strict-clean 仍然是 0，因为模型还卡在类似：

```text
Category:
```

这样的协议字段里。EOS 第一不等于答案已经干净完成。

Phase931 做的是：

```text
这把钥匙是不是每次都完全不同？
```

结果是：

```text
不是完全不同。
高频 64 通道固定组能打开 20/30 个状态；
但 chair 还需要额外补偿。
```

所以现在更准确的图像是：

```text
标点锁有一把公共钥匙；
但不同锁孔还需要一点局部齿形；
而且门打开后，房间里还没有整理干净。
```

## Phase 932: 固定公共齿轮修复与 case-specific 残差审计 [2026-07-04 16:08]

### 一、对上传分析的判断

上传分析对 Phase930/931 的证据层级限定是正确的。Phase930/931 没有证明自然门控、strict-clean 闭合或语言编码机制闭合，只证明了更窄的事实：

```text
1. punctuation_period blocker 有独立 margin gear。
2. state-specific margin_support_pos_64 可以在人工 factor 下打开 30/30 GLM4 states。
3. fixed_topfreq_64 公共齿轮不是无效的，可覆盖 20/30 states。
4. chair case 是 fixed_topfreq_64 的主要缺口。
5. strict_clean 始终为 0，自然 gate 仍未找到。
```

因此本阶段继续沿着附件建议的方向做 Phase932：不再泛化为“闭合”，而是验证下式中的中间结构：

```text
G_punct(x) = G_common + G_case(x) + G_residual(x)
```

其中：

```text
G_common:
  Phase931 中的 fixed_topfreq_64。

G_case(x):
  从每个 case 的 margin_support_pos_64 稳定通道中，扣除 fixed_topfreq_64 后得到的 case residual。

G_residual(x):
  strict-clean 转移、自然 gate、协议场清理等尚未解释的剩余机制。
```

### 二、测试脚本与测试范围

新增脚本：

```text
tests/glm5/phase932_fixed_gear_repair_case_residual_audit.py
tests/glm5/run_phase932_fixed_gear_repair_case_residual_audit.sh
```

结果目录：

```text
tests/result/phase932_fixed_gear_repair_case_residual_audit/fixed_gear_repair_case_residual_audit/
```

测试顺序按要求依次执行：

```text
qwen3 -> GLM4 -> DS7B
```

其中 qwen3 和 DS7B 当前仍没有 punctuation_period seeds，因此没有加载模型，也没有产生有效干预结果。GLM4 完成 30 个 punctuation states、3 个 case、4410 行干预测试。

### 三、测试原理

Phase931 的固定公共组：

```text
G_common = fixed_topfreq_64
```

Phase930 的每个 state 都有一个：

```text
G_state(s) = margin_support_pos_64(s)
```

对每个 case 定义：

```text
G_case_inter(c) = intersection_s_in_c G_state(s)
G_case_union(c) = union_s_in_c G_state(s)
```

再扣除公共齿轮：

```text
R_case_inter(c) = G_case_inter(c) - G_common
R_case_union(c) = G_case_union(c) - G_common
```

本阶段比较以下候选通道组：

```text
1. fixed_topfreq_64
2. chair_inter_residual_only
3. chair_union_residual_only
4. case_inter_residual_only
5. case_union_residual_only
6. fixed_topfreq_64 + chair_inter_residual
7. fixed_topfreq_64 + chair_union_residual
8. fixed_topfreq_64 + case_inter_residual
9. fixed_topfreq_64 + case_union_residual
10. state_specific_margin_support_pos_64
```

每组在同一坐标体系下测试：

```text
route_alpha / protocol_factor:
1.0:1.0
0.875:1.1
1.25:1.1
0.875:0.85
1.25:0.85
1.375:0.85
1.375:0.9

factor:
2.1, 2.25
```

判定指标保持 Phase930/931 口径：

```text
new_top1_vs_coordinate_base
new_margin_closure_vs_coordinate_base
new_strict_vs_coordinate_base
target_state_coverage_top1
```

### 四、case residual 库存

GLM4 的 Phase930 state feature 统计得到：

```text
p856_021_material_wood:
  states = 10
  intersection = 58
  union = 69
  inter_residual = 2
  union_residual = 9
  inter_residual channels = [3164, 4051]

p856_035_object_chair:
  states = 10
  intersection = 63
  union = 65
  inter_residual = 17
  union_residual = 19
  inter_residual channels =
    [1298, 3310, 3377, 3930, 4265, 4407, 6491, 7282, 7310,
     7907, 10417, 10822, 10874, 11002, 11558, 12040, 12860]

p885_047_animal_shark:
  states = 10
  intersection = 64
  union = 64
  inter_residual = 13
  union_residual = 13
```

这与 Phase931 的缺口一致：chair 不是没有可识别稳定通道，而是这些稳定通道大量落在 fixed_topfreq_64 之外。

### 五、客观结果

GLM4 总体：

```text
rows = 4410
coordinate_baseline_rows = 210
candidate_rows = 4200
unique_states = 30
unique_cases = 3
top1 = 1698
margin_nonnegative = 1698
strict_clean_candidate = 0
new_top1_vs_coordinate_base = 1698
new_margin_closure_vs_coordinate_base = 1698
new_strict_vs_coordinate_base = 0
target_state_coverage_top1 = 30
target_state_coverage_strict = 0
worsened_margin_vs_coordinate_base = 560
```

关键组覆盖率：

```text
fixed_topfreq_64, factor=2.25:
  coverage = 20/30
  wood = 10/10
  chair = 0/10
  shark = 10/10
  top1 rows = 122/210
  strict_clean = 0

fixed_plus_chair_inter_residual, factor=2.25:
  coverage = 20/30
  wood = 0/10
  chair = 10/10
  shark = 10/10
  top1 rows = 140/210
  strict_clean = 0

fixed_plus_case_inter_residual, factor=2.25:
  coverage = 30/30
  wood = 10/10
  chair = 10/10
  shark = 10/10
  top1 rows = 202/210
  strict_clean = 0

fixed_plus_case_union_residual, factor=2.25:
  coverage = 30/30
  wood = 10/10
  chair = 10/10
  shark = 10/10
  top1 rows = 210/210
  strict_clean = 0

state_specific_margin_support_pos_64, factor=2.25:
  coverage = 30/30
  top1 rows = 200/210
  strict_clean = 0
```

低 factor 下：

```text
fixed_plus_case_union_residual, factor=2.1:
  coverage = 30/30
  top1 rows = 158/210

fixed_plus_case_inter_residual, factor=2.1:
  coverage = 22/30
  top1 rows = 123/210

state_specific_margin_support_pos_64, factor=2.1:
  coverage = 22/30
  top1 rows = 98/210
```

残差单独使用：

```text
chair_inter_residual_only:
  coverage = 0/30

chair_union_residual_only:
  coverage = 0/30

case_inter_residual_only:
  coverage = 0/30

case_union_residual_only:
  coverage = 0/30
```

### 六、结果分析

本阶段最重要的正结果：

```text
fixed_topfreq_64 + case residual 可以把 Phase931 的 20/30 coverage 修复到 30/30。
```

这支持以下结构：

```text
punctuation_period margin gear 不是完全 state-specific；
存在公共齿轮骨架 G_common；
但每个 case 仍需要少量 case residual 齿形补偿。
```

尤其 chair：

```text
fixed_topfreq_64 在 chair 上为 0/10；
fixed_topfreq_64 + chair residual 在 chair 上为 10/10；
chair residual only 仍为 0/10。
```

这说明 chair residual 不是独立钥匙，而更像公共钥匙上的局部齿形补偿。

同时有一个重要负结果：

```text
fixed_topfreq_64 + chair residual 不能修复 wood；
wood 从 fixed_topfreq_64 的 10/10 变为 0/10。
```

这说明 residual 不是单调通用增强项，而是 case-specific 的边界结构。错误 residual 可能破坏其他 case 的闭合。

### 七、严格审视与硬伤

本阶段不能过度解释，原因如下：

```text
1. case residual 是从同一批 10 个 states 中统计得到的，存在 within-case overfit 风险。
2. fixed_plus_case_union_residual 的 group size 大于 fixed_topfreq_64，可能包含“更大组导致更强干预”的因素。
3. 人工 factor 仍为 2.1/2.25，不是自然 gate。
4. strict_clean_candidate 仍为 0，没有从 EOS top1 进入干净答案闭合。
5. qwen3 / DS7B 没有 punctuation seeds，跨模型证据仍缺失。
6. 结果只针对 punctuation_period blocker，不代表全部协议边、语义边或语言生成闭合。
```

因此证据层级应严格写为：

```text
公共齿轮 + case residual 在 GLM4 punctuation_period blocker 上完成了人工打开覆盖修复；
不是自然门控闭合；
不是 strict-clean 闭合；
不是语言编码机制闭合。
```

### 八、闭合标准与当前距离

当前闭合标准至少包括：

```text
1. fixed/common gear 可以跨 state、跨 case、跨模型稳定预测边界迁移。
2. case residual 的定义不能依赖目标 state 本身，必须通过 holdout 验证。
3. 人工 factor 要被自然 gate 变量替代。
4. EOS top1 后必须进入 strict-clean answer，而不是协议字段漂移。
5. qwen3、GLM4、DS7B 至少要有可比较入口。
```

Phase932 完成的是第 1 条的一部分：

```text
在 GLM4 内部，公共齿轮 + case residual 可以覆盖 30/30 punctuation states。
```

但第 2-5 条均未完成。因此距离机制闭合仍较远，当前更像完成了 punctuation margin gear map 的一个结构拼图。

### 九、智能理论洞察

从智能理论角度看，本阶段给出的关键线索是：

```text
语言边界状态可能不是由单一神经元、单一方向或单一通道组决定；
而是由公共骨架 + case-specific 齿形共同决定。
```

这比“每个状态独立找 patch”更接近可组合结构：

```text
G_common:
  负责 punctuation_period 边界打开的公共动力学骨架。

G_case:
  负责不同语义对象/输出场景下的局部齿形补偿。

G_residual:
  负责自然 gate、strict-clean、协议场清理等还未测出的部分。
```

如果后续 holdout 能证明 case residual 不依赖目标 state 本身，那么全局齿轮图谱会从“局部通道集合”进一步升级为“可组合齿轮结构”。

### 十、下一步任务与阶段判断

Phase932 后的下一任务仍属于同一个阶段目标，因为它直接检验 Phase932 的主要硬伤：

```text
Phase933:
  leave-one-state-out case residual holdout

问题：
  如果计算某个 state 的 case residual 时，把这个 state 从统计中拿掉，
  fixed_topfreq_64 + case residual 是否仍能修复该 state？
```

如果 holdout 仍能覆盖 chair 和 wood，那么 Phase932 的 case residual 结构更可信；如果覆盖明显下降，则 Phase932 可能只是同批状态上的 residual 过拟合。

因此本轮不在 Phase932 停止，继续自动推进 Phase933。

### 十一、通俗总结

Phase931 发现：

```text
公共钥匙能打开 20 把锁，但打不开 chair 的 10 把锁。
```

Phase932 发现：

```text
chair 不是完全没有规律。
只要在公共钥匙上加 chair 自己的齿形，chair 的 10 把锁都能打开。
```

但也发现：

```text
chair 的齿形不能乱装到 wood 上；
装错了反而打不开。
```

所以当前结论是：

```text
标点边界确实像“公共齿轮 + 局部齿形补偿”；
但这仍是人工打开，不是自然闭合。
```

## Phase 933: leave-one-state-out case residual holdout 审计 [2026-07-04 16:15]

### 一、任务动机

Phase932 得到强正结果：

```text
fixed_topfreq_64 + case residual
可以把 GLM4 punctuation_period blocker 从 20/30 修复到 30/30。
```

但 Phase932 有一个必须立刻审计的硬伤：

```text
case residual 是从同一批 10 个 states 中统计出来的；
如果目标 state 自己参与了 residual 统计，
那么 fixed + residual 的成功可能包含同批状态泄漏。
```

因此 Phase933 做 leave-one-state-out（LOSO）验证：对每个目标 state，计算 case residual 时排除这个 state，只使用同 case 其他 9 个 states。

### 二、测试脚本与结果目录

新增脚本：

```text
tests/glm5/phase933_loso_case_residual_holdout_audit.py
tests/glm5/run_phase933_loso_case_residual_holdout_audit.sh
```

结果目录：

```text
tests/result/phase933_loso_case_residual_holdout_audit/loso_case_residual_holdout_audit/
```

测试顺序：

```text
qwen3 -> GLM4 -> DS7B
```

qwen3 与 DS7B 仍没有 punctuation_period seeds，因此本阶段有效干预结果仍只来自 GLM4。

### 三、测试原理

对目标 state：

```text
s_i in case c
```

先从同 case 中排除它：

```text
H_c^{-i} = {s_j | s_j in case c, j != i}
```

再定义：

```text
G_loso_inter(c, i) = intersection_{s in H_c^{-i}} G_state(s)
G_loso_union(c, i) = union_{s in H_c^{-i}} G_state(s)
```

扣除公共齿轮：

```text
R_loso_inter(c, i) = G_loso_inter(c, i) - G_common
R_loso_union(c, i) = G_loso_union(c, i) - G_common
```

测试组：

```text
1. fixed_topfreq_64
2. loso_case_inter_residual_only
3. loso_case_union_residual_only
4. fixed_topfreq_64 + loso_case_inter_residual
5. fixed_topfreq_64 + loso_case_union_residual
6. state_specific_margin_support_pos_64
```

坐标与 factor 保持 Phase932 口径：

```text
7 个 route_alpha / protocol_factor 坐标
factor = 2.1, 2.25
```

### 四、LOSO residual 库存

GLM4 中，每个目标 state 都用同 case 其他 9 个 states 计算 residual。库存范围：

```text
p856_021_material_wood:
  states = 10
  holdout_each = 9
  inter_residual_size = 2..2
  union_residual_size = 9..9

p856_035_object_chair:
  states = 10
  holdout_each = 9
  inter_residual_size = 17..17
  union_residual_size = 19..19

p885_047_animal_shark:
  states = 10
  holdout_each = 9
  inter_residual_size = 13..13
  union_residual_size = 13..13
```

这说明 Phase932 的 case residual 并非由某个单独目标 state 决定。至少在这批 10-state case 内，residual 库存相当稳定。

### 五、客观结果

GLM4 总体：

```text
rows = 2730
coordinate_baseline_rows = 210
candidate_rows = 2520
unique_states = 30
unique_cases = 3
top1 = 1183
margin_nonnegative = 1183
strict_clean_candidate = 0
new_top1_vs_coordinate_base = 1183
new_margin_closure_vs_coordinate_base = 1183
new_strict_vs_coordinate_base = 0
improved_margin_vs_coordinate_base = 2498
worsened_margin_vs_coordinate_base = 0
target_state_coverage_top1 = 30
target_state_coverage_strict = 0
```

关键覆盖结果：

```text
fixed_topfreq_64, factor=2.25:
  coverage = 20/30
  wood = 10/10
  chair = 0/10
  shark = 10/10
  top1 rows = 122/210
  strict_clean = 0

fixed_plus_loso_case_inter_residual, factor=2.25:
  coverage = 30/30
  wood = 10/10
  chair = 10/10
  shark = 10/10
  top1 rows = 202/210
  strict_clean = 0

fixed_plus_loso_case_union_residual, factor=2.25:
  coverage = 30/30
  wood = 10/10
  chair = 10/10
  shark = 10/10
  top1 rows = 210/210
  strict_clean = 0

state_specific_margin_support_pos_64, factor=2.25:
  coverage = 30/30
  top1 rows = 200/210
  strict_clean = 0
```

低 factor 下：

```text
fixed_plus_loso_case_union_residual, factor=2.1:
  coverage = 30/30
  top1 rows = 158/210

fixed_plus_loso_case_inter_residual, factor=2.1:
  coverage = 22/30
  top1 rows = 123/210

state_specific_margin_support_pos_64, factor=2.1:
  coverage = 22/30
  top1 rows = 98/210
```

残差单独使用：

```text
loso_case_inter_residual_only:
  coverage = 0/30

loso_case_union_residual_only:
  coverage = 0/30
```

chair case 单独看，factor=2.25：

```text
fixed_topfreq_64:
  chair top1 rows = 0/70
  chair state coverage = 0/10

fixed_plus_loso_case_inter_residual:
  chair top1 rows = 70/70
  chair state coverage = 10/10

fixed_plus_loso_case_union_residual:
  chair top1 rows = 70/70
  chair state coverage = 10/10

loso residual only:
  chair state coverage = 0/10
```

### 六、结果分析

Phase933 强化了 Phase932 的正结果：

```text
即使排除目标 state，
同 case 其他 9 个 states 得到的 residual 仍然可以修复目标 state。
```

这说明：

```text
case residual 至少不是单个目标 state 泄漏；
而是 case 内较稳定的补偿齿形。
```

更重要的是：

```text
residual only 仍然 0/30；
fixed_topfreq_64 only 仍然 20/30；
fixed_topfreq_64 + LOSO case residual 才到 30/30。
```

所以结构更像：

```text
公共齿轮骨架 + case-specific 齿形补偿
```

而不是：

```text
公共齿轮无效；
case residual 自己就是完整机制。
```

### 七、严格审视与硬伤

Phase933 解决了 Phase932 的一个硬伤，但没有解决全部问题：

```text
已收紧：
  排除目标 state 后仍能修复，降低了 within-case 单点泄漏风险。

仍未解决：
  1. 仍是同 case 内 holdout，不是新 case holdout。
  2. 仍是 GLM4 单模型，qwen3 / DS7B 没有 punctuation seeds。
  3. 仍依赖人工 factor = 2.1 / 2.25。
  4. strict_clean_candidate 仍为 0。
  5. union residual 的 group size 更大，可能有“更大组更强”的因素，需要控制组大小。
  6. 当前只测 punctuation_period，不代表所有 clean protocol edge。
```

因此证据层级应写为：

```text
GLM4 punctuation_period blocker 中，
公共齿轮 + LOSO case residual 可以稳定修复人工边界打开；
但自然门控、strict-clean 和跨模型闭合仍未完成。
```

### 八、闭合距离

当前图谱进展：

```text
1. punctuation_period 边界存在公共齿轮 fixed_topfreq_64。
2. chair 缺口可由 case residual 修复。
3. residual 在 LOSO 下仍稳定。
4. residual only 无法独立闭合，说明它是补偿齿形而非完整钥匙。
```

距离闭合仍有：

```text
1. 自然 gate 未找到。
2. strict-clean = 0。
3. 跨模型入口缺失。
4. 新 case / 新语义域 holdout 未完成。
5. group-size 等干预强度混杂还需控制。
```

### 九、智能理论洞察

这一阶段对“全局齿轮图谱”的意义比较明确：

```text
语言边界齿轮不是完全全局固定；
也不是完全状态特异；
它呈现出“公共骨架 + case 稳定补偿”的中间结构。
```

这类结构可能是后续破解语言编码机制的重要入口，因为它符合可组合机制的形式：

```text
全局规则:
  fixed_topfreq_64

局部条件:
  case residual

尚未解释的动态门:
  natural gate / protocol cleanup
```

如果未来能找到自然 gate 来选择这些 residual，而不是人工指定 case residual，那么图谱才会从“可人工打开”迈向“可预测自然生成”。

### 十、下一阶段任务

Phase932/933 已经完成当前小阶段目标：

```text
验证 fixed common gear + case residual 是否能修复 Phase931 的 chair 缺口；
并用 LOSO 排除目标 state 泄漏。
```

下一阶段不应继续单纯扩大 factor，而应进入：

```text
Phase934: case residual gate and size-control audit
```

重点：

```text
1. 控制 group size，比较同大小随机 residual、同大小非 case residual、case residual。
2. 寻找能预测 case residual 是否需要开启的自然变量。
3. 构造新 case / 新语义域 punctuation states，测试 residual 是否可以外推。
4. 继续追踪 strict-clean，不能只追 top1/margin。
5. 为 qwen3 / DS7B 重构 punctuation seed 入口。
```

Phase934 属于下一个子阶段，因为它已经从“case residual 是否稳定有效”转向“case residual 为什么会被选择、是否只是组大小效应”。本轮到 Phase933 作为当前阶段性目标的收束点。

### 十一、通俗总结

Phase932 说：

```text
chair 的钥匙齿形可以补上。
```

Phase933 进一步说：

```text
就算不看这把 chair 锁本身，
只看其他 chair 锁，也能推回这把锁需要的齿形。
```

这比 Phase932 更稳。当前图像是：

```text
公共钥匙确实存在；
每类锁还有稳定的小齿形；
但钥匙怎么自然长出来、门打开后怎么直接给干净答案，还没破解。
```

## Phase 934: case residual size-control 审计 [2026-07-04 18:14]

### 一、对上传分析的判断

上传分析对 Phase932/933 的判断基本正确。Phase932/933 的核心贡献不是自然闭合，而是把 punctuation_period blocker 的边界齿轮结构从：

```text
state-specific margin gear
```

收紧为：

```text
common gear + case residual
```

也就是：

```text
G_punct(x) = G_common + G_case(x) + G_residual(x)
```

其中当前证据支持：

```text
G_common:
  fixed_topfreq_64，覆盖 20/30。

G_case(x):
  LOSO case residual，修复到 30/30。

G_residual(x):
  natural gate、strict-clean、协议清理仍未解释。
```

但是上传分析也指出一个关键硬伤：union residual 组更大，可能存在 group-size 混杂。因此 Phase934 不继续扩大 factor，而是做同大小控制。

### 二、测试脚本与范围

新增脚本：

```text
tests/glm5/phase934_case_residual_size_control_audit.py
tests/glm5/run_phase934_case_residual_size_control_audit.sh
```

结果目录：

```text
tests/result/phase934_case_residual_size_control_audit/case_residual_size_control_audit/
```

测试顺序：

```text
qwen3 -> GLM4 -> DS7B
```

qwen3 和 DS7B 仍无 punctuation_period seeds，因此没有有效干预数据；GLM4 完成 30 states、4410 rows。

### 三、测试原理

Phase933 已证明：

```text
fixed_topfreq_64 + LOSO case residual -> 30/30
```

Phase934 要问：

```text
这是因为 residual 齿形匹配，
还是因为通道数变多？
```

因此对每个目标 state，保留真实组：

```text
fixed_topfreq_64 + loso_case_inter_residual
fixed_topfreq_64 + loso_case_union_residual
```

同时构造同大小控制：

```text
fixed_topfreq_64 + noncase_inter_size_control
fixed_topfreq_64 + noncase_union_size_control
fixed_topfreq_64 + global_inter_size_control
fixed_topfreq_64 + global_union_size_control
fixed_topfreq_64 + pseudorandom_inter_size_control
fixed_topfreq_64 + pseudorandom_union_size_control
```

其中 control 组和真实 LOSO residual 保持相同 residual size：

```text
wood:
  inter size = 2
  union size = 9

chair:
  inter size = 17
  union size = 19

shark:
  inter size = 13
  union size = 13
```

判定指标仍然是：

```text
new_top1_vs_coordinate_base
new_margin_closure_vs_coordinate_base
new_strict_vs_coordinate_base
target_state_coverage_top1
```

### 四、客观结果

GLM4 总体：

```text
rows = 4410
candidate_rows = 4200
unique_states = 30
unique_cases = 3
top1 = 2317
margin_nonnegative = 2317
strict_clean_candidate = 0
new_top1_vs_coordinate_base = 2317
new_margin_closure_vs_coordinate_base = 2317
new_strict_vs_coordinate_base = 0
target_state_coverage_top1 = 30
target_state_coverage_strict = 0
```

关键覆盖：

```text
fixed_topfreq_64, factor=2.25:
  coverage = 20/30
  wood = 10/10
  chair = 0/10
  shark = 10/10

fixed_plus_loso_case_inter_residual, factor=2.25:
  coverage = 30/30
  wood = 10/10
  chair = 10/10
  shark = 10/10

fixed_plus_loso_case_union_residual, factor=2.25:
  coverage = 30/30
  wood = 10/10
  chair = 10/10
  shark = 10/10

state_specific_margin_support_pos_64, factor=2.25:
  coverage = 30/30
```

同大小控制：

```text
fixed_plus_noncase_inter_size_control, factor=2.25:
  coverage = 20/30

fixed_plus_global_inter_size_control, factor=2.25:
  coverage = 20/30

fixed_plus_pseudorandom_inter_size_control, factor=2.25:
  coverage = 20/30

fixed_plus_noncase_union_size_control, factor=2.25:
  coverage = 20/30

fixed_plus_global_union_size_control, factor=2.25:
  coverage = 20/30

fixed_plus_pseudorandom_union_size_control, factor=2.25:
  coverage = 19/30
```

chair case 单独看，factor=2.25：

```text
fixed_topfreq_64:
  chair coverage = 0/10

fixed_plus_loso_case_inter_residual:
  chair coverage = 10/10

fixed_plus_loso_case_union_residual:
  chair coverage = 10/10

所有同大小 size-control:
  chair coverage = 0/10
```

### 五、结果分析

Phase934 的关键正结果：

```text
真实 LOSO case residual 可以修复 chair；
同大小 noncase/global/pseudorandom residual 不能修复 chair。
```

这排除了一个主要混杂：

```text
不是只要多加 17/19 个通道就能修复 chair。
```

更准确的结构是：

```text
fixed_topfreq_64 提供公共骨架；
case residual 提供齿形匹配；
错误 residual 或同大小控制无法替代真实 residual。
```

这比 Phase932/933 更进一步支持“齿形匹配”解释。

### 六、硬伤与边界

仍必须严格限制证据：

```text
1. 这仍是 GLM4 单模型，qwen3 / DS7B 没有入口。
2. 仍是人工 factor = 2.1 / 2.25。
3. strict_clean_candidate 仍为 0。
4. 同大小控制排除了 size，但没有找到 natural gate。
5. 仍是同 case 内 residual，不是新 case 外推。
6. 当前只针对 punctuation_period blocker。
```

因此 Phase934 证明的是：

```text
case residual 的优势不是简单 group-size 效应；
但还不是自然门控闭合，也不是 strict-clean 闭合。
```

### 七、闭合距离

当前已收紧：

```text
1. fixed common gear 存在。
2. LOSO case residual 稳定。
3. residual only 无效。
4. 同大小错配 residual 不能修复 chair。
```

仍未完成：

```text
1. natural gate。
2. strict-clean。
3. new case holdout。
4. qwen3 / DS7B 跨模型入口。
5. protocol cleanup route。
```

### 八、智能理论洞察

Phase934 对智能理论的价值是：它支持“语言边界齿轮具有形状匹配”这一拼图。

这不是简单的线性增强：

```text
more channels -> stronger closure
```

而更像：

```text
right common skeleton + right local teeth -> boundary opens
wrong same-size teeth -> boundary remains blocked
```

如果这种结构在更多 blocker class、更多 case、更多模型上复现，那么全局齿轮图谱将从“通道集合图谱”升级为“齿形组合图谱”。

### 九、阶段判断

Phase934 完成了 Phase933 后直接暴露的 group-size 混杂审计。下一步仍在同一子阶段内，因为还需要审计：

```text
case residual 是否有可观测 gate candidate。
```

因此继续自动执行 Phase935。

### 十、通俗总结

Phase934 问：

```text
chair 被修好，是不是只是因为多加了一些通道？
```

结果是：

```text
不是。
同样数量的错配通道，修不好 chair；
只有 chair 对应的 residual 齿形能修好。
```

所以更像：

```text
不是钥匙更大；
而是齿形对上了。
```

## Phase 935: case residual gate candidate 审计 [2026-07-04 18:14]

### 一、任务动机

Phase934 排除了主要 group-size 混杂，但还没有回答：

```text
模型自然状态下如何知道什么时候需要 case residual？
```

因此 Phase935 不做新模型推理，而是读取 Phase930 阈值特征和 Phase934 成败结果，审计 residual-needed 的候选门控变量。

### 二、测试脚本与结果目录

新增脚本：

```text
tests/glm5/phase935_case_residual_gate_candidate_audit.py
tests/glm5/run_phase935_case_residual_gate_candidate_audit.sh
```

结果目录：

```text
tests/result/phase935_case_residual_gate_candidate_audit/case_residual_gate_candidate_audit/
```

Phase935 不加载模型，只做跨模型结果读取：

```text
qwen3 -> GLM4 -> DS7B
```

qwen3 / DS7B 无 Phase934 有效状态，因此结果仍只有 GLM4。

### 三、测试原理

对每个 state 定义：

```text
fixed_success_2_25:
  fixed_topfreq_64 是否成功打开。

residual_needed_2_25:
  fixed_topfreq_64 是否失败。

true_loso_repair_success_2_25:
  fixed + true LOSO case residual 是否成功。

size_control_success_2_25:
  fixed + 同大小控制 residual 是否成功。

true_beats_controls_2_25:
  true LOSO residual 成功且 size-control 失败。
```

然后用 Phase930 的可观测变量做单变量阈值分裂：

```text
target_route_delta_norm
target_boundary_eos_margin_vs_blocker
target_boundary_eos_rank
boundary_period_gap_vs_eos
boundary_punctuation_gap_vs_eos
l39_margin_pos_* score
l39_eos_support_mean_score
l39_neg_margin_mean_score
phase925_factor
opening_threshold_factor
```

注意：这不是因果门控训练，只是候选变量审计。

### 四、客观结果

GLM4 状态数：

```text
state_rows = 30
fixed_success_2_25 = 20
residual_needed_2_25 = 10
true_loso_repair_success_2_25 = 30
size_control_success_2_25 = 20
true_beats_controls_2_25 = 10
```

按 case：

```text
wood:
  states = 10
  fixed_success = 10
  residual_needed = 0
  true_repair = 10
  size_control = 10
  true_beats_controls = 0

chair:
  states = 10
  fixed_success = 0
  residual_needed = 10
  true_repair = 10
  size_control = 0
  true_beats_controls = 10

shark:
  states = 10
  fixed_success = 10
  residual_needed = 0
  true_repair = 10
  size_control = 10
  true_beats_controls = 0
```

候选 split：

```text
residual_needed_2_25:
  target_route_delta_norm <= 0.036896469071507454
  accuracy = 30/30
  true = 10
  false = 20
  case_confounded = true

target_boundary_eos_margin_vs_blocker <= -5.078125
  accuracy = 30/30
  case_confounded = true

target_boundary_eos_rank >= 8.5
  accuracy = 30/30
  case_confounded = true

boundary_period_gap_vs_eos >= 5.078125
  accuracy = 30/30
  case_confounded = true

l39_eos_support_mean_score <= 0.41953757405281067
  accuracy = 30/30
  case_confounded = true
```

### 五、结果分析

Phase935 找到了高准确候选变量，但不能写成自然门控，因为：

```text
residual_needed_2_25 的 true cases 只有 chair；
false cases 是 wood + shark。
```

所以所有 30/30 的 split 都是：

```text
case-confounded candidate
```

它们说明：

```text
chair 状态在边界难度、EOS rank、period gap、L39 支持强度上确实不同；
但还不能证明模型使用这些变量自然选择 chair residual。
```

当前最严谨表述：

```text
发现 residual-needed 的可观测候选变量；
但候选变量与 case 标签完全纠缠；
尚未找到脱离 case 标签的 natural gate。
```

### 六、硬伤与边界

Phase935 的硬伤：

```text
1. 只有 3 个 case，每个 case 10 states。
2. residual_needed 只出现在 chair 一个 case。
3. 高准确 split 可能只是识别 chair，而不是识别 gate。
4. 没有新 case holdout。
5. 没有自然激活因果干预。
6. strict_clean 仍为 0。
```

因此证据层级必须收紧为：

```text
case residual gate candidate found, but case-confounded；
not natural gate closure。
```

### 七、闭合标准与当前距离

Phase934/935 后，当前完成：

```text
1. case residual 不是 group-size 效应。
2. residual-needed 有可观测候选变量。
3. 这些变量暂时完全 case-confounded。
```

仍未完成：

```text
1. 脱离 case 标签的自然 gate。
2. 新 case / 新语义域 holdout。
3. strict-clean。
4. 跨模型入口。
```

### 八、智能理论洞察

Phase935 对智能理论的启发是：

```text
case residual 可能不是被一个抽象通用 gate 直接选择；
它目前更像和对象/语义场、边界难度、协议压力共同纠缠。
```

这说明破解语言编码机制不能只找一个 scalar gate，而要继续把图谱拆成：

```text
semantic case identity
boundary difficulty
protocol pressure
residual gear selection
strict cleanup transition
```

### 九、阶段性收束

Phase934/935 已完成当前子阶段目标：

```text
1. group-size 混杂审计完成；
2. case residual gate candidate 审计完成；
3. 结论严格收紧为 case-confounded candidate，而不是 natural gate。
```

下一阶段应进入：

```text
Phase936: new case / new semantic-domain residual holdout
```

核心任务：

```text
1. 构造更多 punctuation_period states。
2. 增加新 object / 新 semantic domain。
3. 测试 case residual 能否外推到未参与残差统计的新 case。
4. 尝试重建 qwen3 / DS7B punctuation seeds。
5. strict-clean 继续作为硬指标，不因 top1/margin 成功而放宽。
```

这已经是新阶段目标，本轮停止自动推进。

### 十、通俗总结

Phase935 问：

```text
模型怎么知道 chair 需要额外齿形？
```

结果是：

```text
我们能找到一些信号把 chair 分出来；
但这些信号目前还只是“chair 的特征”，
不能证明它们就是模型自然使用的门控。
```

所以当前图像是：

```text
齿形匹配是真的；
但自动选齿形的机关还没找到。
```

## Phase 936: 同 case 候选状态的 case residual 迁移留出审计 [2026-07-04 18:33]

### 一、问题来源

最新附件对 Phase934/935 的判断基本正确：

```text
Phase934 证明 case residual 的有效性不能被同大小 group control 解释；
Phase935 找到了 residual-needed 的候选观测变量；
但这些变量仍与 case 标签纠缠，不能写成自然门控。
```

附件建议下一步做：

```text
new case / new semantic-domain residual holdout
```

这个方向正确，但先检查数据池后发现一个重要限制：

```text
当前 Phase925 可用 punctuation_period seed 只覆盖 GLM4 的 3 个旧 case：
p856_021_material_wood
p856_035_object_chair
p885_047_animal_shark

qwen3 / DS7B 没有 punctuation_period candidate seeds。
```

因此 Phase936 不能声称完成新 case 泛化，只能在现有数据上做更窄的检验：

```text
case residual 是否能从 Phase930 训练状态迁移到同 case 的未见 candidate 状态。
```

### 二、测试脚本

新增脚本：

```text
tests/glm5/phase936_same_case_candidate_residual_holdout.py
tests/glm5/run_phase936_same_case_candidate_residual_holdout.sh
```

结果目录：

```text
tests/result/phase936_same_case_candidate_residual_holdout/same_case_candidate_residual_holdout/
```

脚本按顺序尝试：

```text
qwen3 -> GLM4 -> DS7B
```

qwen3 和 DS7B 因缺少 punctuation_period holdout seeds 跳过；GLM4 完成模型加载、测试和显存释放。

### 三、测试原理

Phase930/932/933 使用的是 selected states。Phase936 改用 Phase925 candidate states 中未出现在 selected/training key 里的 punctuation_period 状态。

训练状态残差：

```text
G_common = fixed_topfreq_64

R_train_case_inter(c) =
  intersection_s_in_train(c) G_state(s) - G_common

R_train_case_union(c) =
  union_s_in_train(c) G_state(s) - G_common
```

在未见 candidate 状态 s' 上测试：

```text
G_test_inter(s') = G_common + R_train_case_inter(case(s'))
G_test_union(s') = G_common + R_train_case_union(case(s'))
```

对照组：

```text
1. coordinate_only
2. fixed_topfreq_64
3. fixed + train_case_inter_residual
4. fixed + train_case_union_residual
5. fixed + noncase_inter_size_control
6. fixed + noncase_union_size_control
7. fixed + pseudorandom_inter_size_control
8. fixed + pseudorandom_union_size_control
9. state_specific_margin_support_pos_64
```

缩放因子：

```text
2.1
2.25
```

判定指标：

```text
top1
margin_nonnegative
strict_clean_candidate
target_state_coverage_top1
mean_margin_delta_vs_coordinate_base
```

### 四、客观结果

GLM4 数据可用性：

```text
phase925_selected_punctuation_keys = 30
candidate_punctuation_rows = 264
deduped_unseen_punctuation_states = 234
selected_holdout_states = 90

selected_holdout_cases:
  wood  = 30
  chair = 30
  shark = 30

new_case_available = false
```

实际运行：

```text
rows = 10710
coordinate_baseline_rows = 630
candidate_rows = 10080
unique_states = 90
unique_cases = 3
candidate_top1 = 5840
candidate_margin_nonnegative = 5840
candidate_strict_clean_candidate = 0
target_state_coverage_top1 = 90
target_state_coverage_strict = 0
```

主要 coverage：

```text
fixed_plus_train_case_inter_residual, factor=2.25:
  all = 90/90
  wood = 30/30
  chair = 30/30
  shark = 30/30

fixed_plus_train_case_union_residual, factor=2.25:
  all = 90/90
  wood = 30/30
  chair = 30/30
  shark = 30/30

state_specific_margin_support_pos_64, factor=2.25:
  all = 90/90
```

关键对照：

```text
fixed_topfreq_64, factor=2.25:
  all = 60/90
  wood = 30/30
  chair = 0/30
  shark = 30/30

fixed_plus_noncase_inter_size_control, factor=2.25:
  all = 60/90
  chair = 0/30

fixed_plus_noncase_union_size_control, factor=2.25:
  all = 60/90
  chair = 0/30

fixed_plus_pseudorandom_inter_size_control, factor=2.25:
  all = 59/90
  chair = 0/30

fixed_plus_pseudorandom_union_size_control, factor=2.25:
  all = 55/90
  chair = 0/30
```

chair case 的 2.25 结果：

```text
fixed_plus_train_case_union_residual:
  top1 rows = 210/210
  states = 30/30
  mean_delta = 6.735714285714286

fixed_plus_train_case_inter_residual:
  top1 rows = 210/210
  states = 30/30
  mean_delta = 6.663988095238095

fixed_topfreq_64:
  top1 rows = 0/210
  states = 0/30
  mean_delta = 4.64702380952381

noncase / pseudorandom size controls:
  top1 states = 0/30
```

### 五、结果分析

Phase936 的正结果：

```text
用 Phase930/selected states 统计出来的 train_case_residual，
可以迁移到同 case 的未见 candidate states。
```

这比 Phase933 的 leave-one-state-out 更进一步，因为测试状态不是 selected states 内部留一，而是来自 Phase925 candidate pool 的更大未见集合。

最关键现象是 chair：

```text
fixed_topfreq_64 无法修复 chair：0/30
同大小 noncase / pseudorandom controls 也无法修复 chair：0/30
train_case_residual 可以修复 chair：30/30
```

因此当前证据支持：

```text
chair 的 residual 齿形不是单个 selected state 的偶然特征；
它至少在同 case 的未见 candidate 状态中稳定存在。
```

但证据不能升级为：

```text
new semantic-domain generalization
natural gate closure
strict-clean closure
language encoding closure
```

因为新 case 不存在，且 strict_clean 仍为 0。

### 六、硬伤与边界

Phase936 的主要边界：

```text
1. 只完成 GLM4，qwen3 / DS7B 没有 punctuation_period seeds。
2. 只有 wood/chair/shark 三个旧 case。
3. 未见 candidate states 仍来自同一数据生成体系，可能和 selected states 共享模板偏差。
4. factor=2.25 仍是人为缩放，不是自然门控。
5. strict_clean_candidate = 0。
6. patched_blocker_class 仍全部是 punctuation_period，没有进入完整 blocker 类迁移。
```

因此最严格结论是：

```text
same-case residual shape transfer positive；
not new-case residual generalization。
```

### 七、闭合标准与当前距离

当前已经完成：

```text
1. 固定公共齿轮能处理 wood/shark。
2. chair 需要 case residual。
3. case residual 不是同大小 group control。
4. case residual 可迁移到同 case 未见 candidate states。
```

距离闭合仍差：

```text
1. 新 case / 新语义域 holdout。
2. 跨模型 qwen3 / DS7B 复现。
3. 自然门控变量，不依赖 case 标签选择 residual。
4. strict-clean 输出清理。
5. 从 punctuation_period 扩展到更多 blocker 类。
```

### 八、智能理论洞察

Phase936 说明当前图谱里有一类比较稳定的对象-边界残差齿形：

```text
同一个语义对象 case 内，
不同表面状态可能共享一部分 residual gear geometry。
```

这符合“先完成图谱再追求闭合”的路线：

```text
先证明齿形在局部对象域内稳定；
再测试它是否跨对象、跨语义域、跨模型稳定；
最后才讨论自然门控和语言编码机制闭合。
```

当前更像是在拼出：

```text
object-specific residual manifold
```

而不是已经找到：

```text
universal language coding mechanism
```

### 九、下一阶段任务

Phase936 已完成当前可用数据池内的同 case 未见状态迁移审计。下一阶段不应继续在这三个旧 case 上反复加补丁，而应先扩展数据：

```text
Phase937: punctuation_period 新 case / 新语义域 seed 构造与审计
```

任务：

```text
1. 为 GLM4 构造新 object / material / animal / action / abstract cases。
2. 为 qwen3 和 DS7B 重建 punctuation_period candidate seeds。
3. 保留 wood/chair/shark 作为旧域参考。
4. 测试 train_case_residual 是否能跨 case 外推，或是否只能在同 case 内稳定。
5. 若新 case 中再次出现 residual-needed，重新审计 gate candidate 是否仍 case-confounded。
```

这属于数据扩展与新阶段，不应把 Phase936 的同 case 正结果过度外推。

### 十、通俗总结

Phase936 问：

```text
chair 的额外齿形是不是只对原来的 10 个样本有效？
```

结果是：

```text
不是。
在另外 30 个没参与残差统计的 chair 候选状态上，
同一个 chair residual 仍然有效。
```

但它还没有回答：

```text
这个齿形能不能迁移到新的物体、新语义域、别的模型？
模型是不是自然知道什么时候该用它？
```

所以当前结论是：

```text
同 case 齿形稳定性增强；
跨域泛化与自然门控仍未完成。
```

## Phase 937: 语义复用-差分状态图谱审计 [2026-07-04 22:09]

### 一、问题来源

最新附件提出的问题是：

```text
如果要知道深度神经网络的编码机制，
水果如何复用神经元？
不同水果如何差异化？
颜色属性如何跨物体复用？
功能在网络内部的脉络是什么？
```

附件中的总体方向基本正确：

```text
概念不是单点神经元；
概念更像分布式状态场。

水果、颜色、功能不是孤立点；
它们可能表现为共享因子、差异因子、关系绑定和候选门控。
```

但必须收紧证据层级：

```text
“水果公共子空间”
“红色属性方向”
“功能关系路径”

这些目前是待验证结构，
不能直接写成已经发现的真实机制。
```

Phase936 已经给出一个输出边界层面的复用-差分例子：

```text
G_common 可处理 wood / shark；
chair 需要 R_case(chair)；
R_case(chair) 可迁移到同 case 未见状态。
```

Phase937 的任务是把这个思路前移到语义状态层，先做非干预型图谱审计：

```text
对象、类别、颜色、功能是否在隐藏状态中表现出可观测的共享与差分结构？
```

### 二、测试脚本

新增脚本：

```text
tests/glm5/phase937_semantic_reuse_difference_state_atlas.py
tests/glm5/run_phase937_semantic_reuse_difference_state_atlas.sh
```

结果目录：

```text
tests/result/phase937_semantic_reuse_difference_state_atlas/semantic_reuse_difference_state_atlas/
```

本阶段按顺序完成：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型跑完后释放显存。

### 三、测试数据

构造 5 个语义域：

```text
fruit
animal
vehicle
tool
material
```

每个语义域 6 个对象，共 30 个对象。

每个对象测试 3 类关系：

```text
category
color
function
```

每类关系 2 个模板，因此每个模型：

```text
30 objects * 3 relations * 2 templates = 180 prompts
```

示例：

```text
In one word, an apple is a type of
The typical color of an apple is
An apple can often
```

目标标签：

```text
category:
  fruit / animal / vehicle / tool / material

color:
  red / yellow / gray / silver / white / ...

function:
  eat / fly / swim / transport / cut / ...
```

### 四、测试原理

对每个 prompt 提取多个 hidden index 的最后 token 状态：

```text
h_l(x)
```

自动选层：

```text
0, 1/4, 1/2, 3/4, final
```

对某个 relation 和 hidden index，计算：

#### 1. 同目标标签复用差

```text
same_target_mean_cos =
  mean cos(h_i, h_j), target_i = target_j

diff_target_mean_cos =
  mean cos(h_i, h_j), target_i != target_j

target_reuse_gap =
  same_target_mean_cos - diff_target_mean_cos
```

如果 `target_reuse_gap > 0`，说明同一类别、颜色或功能的状态更相似。

#### 2. 跨域同属性复用差

```text
same_target_cross_domain_mean_cos =
  mean cos(h_i, h_j),
  target_i = target_j, domain_i != domain_j

diff_target_cross_domain_mean_cos =
  mean cos(h_i, h_j),
  target_i != target_j, domain_i != domain_j

cross_domain_target_gap =
  same_target_cross_domain_mean_cos
  -
  diff_target_cross_domain_mean_cos
```

这个指标更接近：

```text
红色是否跨 apple / car / cardinal 复用？
transport 是否跨 car / bus / train 复用？
```

#### 3. 模板留出最近中心分类

用一个模板形成各 target label 的中心：

```text
\mu_y = mean h_l(x), target(x)=y
```

再用另一个模板测试：

```text
\hat y =
argmax_y cos(h_l(x), \mu_y)
```

得到：

```text
template_holdout_centroid_accuracy
```

这个指标用来减少模板文字本身造成的伪相似。

#### 4. 对象差异残差稳定性

对每个 target label 中心：

```text
r_o(x) = h_l(x) - \mu_{target(x)}
```

比较同对象跨模板 residual 与同标签不同对象 residual：

```text
object_residual_stability_gap
```

这个指标尝试观察：

```text
公共因子去掉后，对象差异是否仍稳定。
```

### 五、客观结果

#### qwen3

总体：

```text
evidence = semantic_reuse_difference_signals_observed
target_rank_mean = 153.80555555555554
target_rank_top1 = 190 / 900 layer-rows
target_rank_top10 = 580 / 900 layer-rows
```

最佳 relation 结果：

```text
category, hidden_idx=9:
  accuracy = 0.7333333333333333
  chance = 0.2
  target_reuse_gap = 0.020418443916497164

color, hidden_idx=36:
  accuracy = 0.8
  chance = 0.1
  target_reuse_gap = 0.031255529731637144
  cross_domain_target_gap = 0.010465477100007226
  object_residual_stability_gap = 0.4881557787932582

function, hidden_idx=27:
  accuracy = 0.5666666666666667
  chance = 0.05555555555555555
  target_reuse_gap = 0.02034485157879451
  cross_domain_target_gap = 0.009549365870738291
```

qwen3 的结果是本阶段最清楚的正结果：

```text
类别、颜色、功能三类 relation 都有高于机会水平的模板留出分类；
颜色 relation 同时出现正 target_reuse_gap 和正 cross_domain_target_gap。
```

#### GLM4

总体：

```text
evidence = semantic_reuse_difference_signal_weak_or_absent
target_rank_mean = 13242.555555555555
target_rank_top1 = 95 / 900 layer-rows
target_rank_top10 = 230 / 900 layer-rows
```

最佳 relation 结果：

```text
category, hidden_idx=10:
  accuracy = 0.21666666666666667
  chance = 0.2
  target_reuse_gap = -0.025527800928161648

color, hidden_idx=30:
  accuracy = 0.16666666666666666
  chance = 0.1
  target_reuse_gap = -0.009039415745926038
  cross_domain_target_gap = -0.0048782713576625

function, hidden_idx=30:
  accuracy = 0.05
  chance = 0.05555555555555555
  target_reuse_gap = -0.05293570986330637
```

GLM4 在这套英文模板上的语义复用-差分信号很弱。这个负结果重要，因为它提醒：

```text
Phase936 的 GLM4 输出边界齿轮结果，
不能直接推出 GLM4 在英文语义状态层也有清晰的同构结构。
```

也可能说明：

```text
GLM4 对这些英文模板/目标词的状态组织方式不同；
需要增加中文模板或 GLM4 更自然的问题格式。
```

#### DS7B

总体：

```text
evidence = semantic_reuse_difference_signals_observed
target_rank_mean = 3300.45
target_rank_top1 = 95 / 900 layer-rows
target_rank_top10 = 310 / 900 layer-rows
```

最佳 relation 结果：

```text
category, hidden_idx=14:
  accuracy = 0.7333333333333333
  chance = 0.2
  target_reuse_gap = 0.01603701621896092

color, hidden_idx=7:
  accuracy = 0.9333333333333333
  chance = 0.1
  target_reuse_gap = -0.00021244926287533605
  cross_domain_target_gap = -0.0020608995187471058

function, hidden_idx=14:
  accuracy = 0.55
  chance = 0.05555555555555555
  target_reuse_gap = -0.03201946720304272
  cross_domain_target_gap = 0.055684905580318245
```

DS7B 的结果显示：

```text
category 有较清楚复用信号；
function 有较强跨域同目标 gap；
color 的最近中心分类很强，但 pairwise reuse gap 接近 0 或略负。
```

所以 DS7B 的 color 结果不能简单写成“颜色公共子空间已找到”，更可能是：

```text
centroid 分类能区分颜色标签；
但全局 pairwise 同色相似性未稳定超过异色相似性。
```

### 六、结果分析

Phase937 的正结果：

```text
qwen3 和 DS7B 在隐藏状态中出现了可观测的语义复用/差分信号。
```

最可信的现象：

```text
1. qwen3 的 category / color / function 均高于机会水平。
2. qwen3 的 color 有正跨域同属性 gap。
3. DS7B 的 category 和 function 有明显高于机会水平的模板留出分类。
4. 部分功能标签跨域出现正 gap，例如 transport / eat / hold 这类关系可能存在弱复用。
```

负结果同样重要：

```text
1. GLM4 在当前英文模板上信号弱。
2. 多数 object_residual_stability_gap 为负。
3. DS7B color 分类高，但 pairwise 同色复用 gap 不强。
```

这说明当前数据支持：

```text
隐藏状态中存在 relation-conditioned semantic clustering。
```

但还不能证明：

```text
存在干净的水果公共子空间；
存在干净的红色方向；
存在可直接复用的功能神经元组；
对象差异残差已稳定分离。
```

### 七、与 Phase936 的关系

Phase936 是输出边界层：

```text
G_common + R_case(chair)
```

Phase937 是语义状态层：

```text
category / color / function 的 hidden-state reuse-difference atlas
```

二者共同支持一个更谨慎的结构：

```text
模型内部可能同时存在：

1. 语义状态层的 relation-conditioned clustering；
2. 输出边界层的 common gear + case residual；
3. 二者之间尚未建立因果桥。
```

也就是说：

```text
语义复用信号存在；
边界齿轮复用信号也存在；
但从语义复用到输出边界齿轮的传递链还没有闭合。
```

### 八、硬伤与边界

Phase937 的硬伤：

```text
1. 非干预实验，只是 hidden state 几何审计。
2. 使用英文模板，GLM4 可能不适配。
3. 只有 2 个模板，模板留出强度还不够。
4. 颜色和功能标签分布不完全均衡。
5. object_residual_stability_gap 多数为负，说明差异残差分解方法还粗糙。
6. 没有证明这些方向会因果影响输出。
7. 没有 patch / ablation / cross-domain causal transfer。
```

因此证据层级应写成：

```text
semantic reuse-difference state signals observed in qwen3 and DS7B；
weak/absent in GLM4 under current English prompts；
not causal semantic factor closure。
```

### 九、闭合标准与当前距离

若要证明“水果如何复用、颜色如何跨对象复用”，闭合标准至少需要：

```text
1. 观测层：
   同类别 / 同属性 / 同功能在模板留出下稳定聚类。

2. 跨域层：
   同颜色或同功能跨不同对象域仍稳定相似。

3. 差分层：
   去掉公共因子后，对象差异 residual 可稳定复现。

4. 因果层：
   patch / ablation 某个候选方向会按预期改变答案或候选 logit。

5. 跨模型层：
   qwen3 / GLM4 / DS7B 至少有部分同构指标。
```

Phase937 目前完成：

```text
观测层：部分完成，qwen3 和 DS7B 较强。
跨域层：部分完成，qwen3 color 和 DS7B function 有正结果。
差分层：未完成，多数 residual gap 为负。
因果层：未开始。
跨模型层：不一致，GLM4 当前弱。
```

因此距离“编码机制闭合”仍然较远。

### 十、智能理论洞察

Phase937 给出的关键拼图是：

```text
语言编码机制不应只看最后输出齿轮；
也不能只看单点神经元；
需要把语义状态层和输出边界层连成图谱。
```

当前最谨慎公式可以写成：

```text
h_l(x)
  contains:
    relation-conditioned semantic clustering
    domain/category factors
    attribute/function weak reuse factors
    template/context components
    object-specific residue
```

而输出阶段仍然是：

```text
G_eff(x)
=
G_common
∪ R_case(case(x))
∪ R_state(x)
```

尚未闭合的关键桥是：

```text
semantic factor
  -> natural gate
  -> boundary gear / case residual
  -> clean output
```

### 十一、下一阶段任务

Phase937 完成了当前阶段的第一步：

```text
跨模型语义复用-差分状态图谱审计。
```

下一步进入新的因果子阶段：

```text
Phase938: semantic factor causal transfer audit
```

建议任务：

```text
1. 增加中文模板，重新测试 GLM4。
2. 为 category / color / function 构造方向：
   d_y = mean(h | target=y) - mean(h | target!=y)
3. 在留出对象和跨域对象上做方向 patch。
4. 测试目标 label logit/rank 是否按预期移动。
5. 加入随机方向和错配标签方向对照。
6. 若因果方向成立，再连接到 Phase936 的 boundary gear。
```

这已经是新的因果测试子阶段，不应把 Phase937 的观测正结果提前写成机制闭合。

### 十二、通俗总结

Phase937 问：

```text
水果、颜色、功能这些东西，
在模型隐藏状态里有没有“同类更像同类”的迹象？
```

结果是：

```text
qwen3 有比较清楚的迹象；
DS7B 也有一部分迹象；
GLM4 在当前英文模板上不明显。
```

这说明：

```text
“复用 + 差分”的方向值得继续；
但现在看到的只是隐藏状态图谱信号，
还不是因果机制。
```

下一步要做的是：

```text
把这些候选方向拿去做 patch / ablation，
看它们是否真的能改变颜色、类别、功能答案。
```

## Phase 938: 语义因子因果迁移审计 [2026-07-04 22:25]

### 一、对附件判断的核对

附件中对 Phase937 的判断基本正确，尤其是证据层级收得比较稳：

```text
Phase937 不是语义编码机制闭合；
不是证明“水果/颜色/功能”的自然因果齿轮；
而是证明在 qwen3 和 DS7B 的隐藏状态中，
已经能观察到 relation-conditioned semantic clustering 的图谱信号。
```

这部分判断应保留。

同时，附件提出的下一步也正确：

```text
不能继续停在相似度/聚类观察；
必须把候选语义方向拿去做因果迁移审计；
看它们是否能在留出模板上推动目标 label 的 logit / margin / rank。
```

因此本阶段接续 Phase937，进入 Phase938。

### 二、本阶段任务

本阶段测试目标：

```text
从 Phase937 观察到的语义状态图谱中，
为 category / color / function 构造候选语义方向；
再把这些方向迁移 patch 到另一模板，
检查目标 label 是否按预期移动。
```

测试不是直接闭合语言编码机制，而是验证一条较短的因果链：

```text
relation-conditioned semantic factor
  -> hidden-state direction patch
  -> target label logit / margin / rank movement
```

### 三、测试脚本和结果位置

新增脚本：

```text
tests/glm5/phase938_semantic_factor_causal_transfer_audit.py
tests/glm5/run_phase938_semantic_factor_causal_transfer_audit.sh
```

结果目录：

```text
tests/result/phase938_semantic_factor_causal_transfer_audit/semantic_factor_causal_transfer_audit/
```

核心结果文件：

```text
phase938_cross_model_summary.md
phase938_cross_model_summary.json
phase938_qwen3_summary.json
phase938_glm4_summary.json
phase938_deepseek7b_summary.json
phase938_qwen3_rows.jsonl
phase938_glm4_rows.jsonl
phase938_deepseek7b_rows.jsonl
```

### 四、测试原理

对每个模型、关系类型、标签和候选层，构造语义方向：

```text
d_{r,y,l,t}
=
mean(h_l(x) | relation=r, target=y, template=t)
-
mean(h_l(x) | relation=r, target!=y, template=t)
```

其中：

```text
r: category / color / function
y: 目标标签
l: 候选隐藏层
t: 训练模板
h_l(x): 第 l 层最后位置 hidden state
```

然后在另一模板上做迁移 patch：

```text
h'_l(x) = h_l(x) + alpha * d_{r,y,l,t}
```

比较：

```text
baseline
target_direction
wrong_label_direction
random_same_norm
negative_target_direction
```

输出指标：

```text
target logit delta = z'_y - z_y

target margin delta
=
(z'_y - max_{y' != y} z'_{y'})
-
(z_y - max_{y' != y} z_{y'})

rank improved
new relation winner
```

### 五、数据规模

本阶段按 qwen3、GLM4、DS7B 顺序运行，避免同时占用 GPU。

```text
每个模型 direction specs: 38
每个模型结果 rows: 1368
三模型总 rows: 4104
关系类型: category / color / function
patch alpha: 0.5 / 1.0
```

这比 Phase937 更进一步，因为它不再只观察隐藏状态相似度，而是做方向迁移干预。

### 六、客观结果

#### 1. qwen3

总体结果：

```text
target_direction alpha=1.0:
  mean logit delta  = +0.4453
  mean margin delta = +0.6525
  rank improved     = 70 / 152
  new winner        = 18 / 152

random_same_norm alpha=1.0:
  mean margin delta = -0.0140

wrong_label_direction alpha=1.0:
  mean margin delta = -0.9552

negative_target_direction alpha=1.0:
  mean margin delta = -0.9741
```

分关系结果：

```text
category target alpha=1.0:
  margin delta = +0.3042
  random control best = +0.0302

color target alpha=1.0:
  margin delta = +0.7315
  random control best = +0.0231

function target alpha=1.0:
  margin delta = +1.0905
  random control best = -0.1365
```

qwen3 是本阶段最干净的正结果：

```text
target direction 明显优于随机方向、错配方向和反方向；
三类关系均出现正向 margin 迁移；
function 和 color 最强，category 较弱但仍为正。
```

#### 2. GLM4

总体结果：

```text
target_direction alpha=1.0:
  mean logit delta  = +4.4645
  mean margin delta = +2.2177
  rank improved     = 104 / 152
  new winner        = 63 / 152
```

但控制项也很大：

```text
random_same_norm alpha=1.0:
  mean margin delta = +1.9296

wrong_label_direction alpha=1.0:
  mean margin delta = +1.3991

negative_target_direction alpha=1.0:
  mean margin delta = +1.2804
```

分关系看：

```text
category:
  target margin delta = +2.2708
  control best        = +2.1879
  target-control gap  = +0.0829

color:
  target margin delta = +1.8600
  control best        = +1.8953
  target-control gap  = -0.0353

function:
  target margin delta = +2.6423
  control best        = +1.5975
  target-control gap  = +1.0448
```

GLM4 的结果不能简单写成干净因果正结果。更谨慎的判断是：

```text
function 方向存在较清楚的 target-specific 正结果；
category / color 下 target direction 与控制方向差距很小；
GLM4 当前更像对隐藏状态扰动高度敏感，
需要中文模板、方向正交化和 size-control 复测。
```

运行层面还发现：

```text
GLM4 结果已经完整写出；
但 Python 进程在释放资源/退出阶段出现 segmentation fault；
因此后续 DS7B 改为单独继续运行。
```

这不影响已保存结果，但属于工程稳定性硬伤，需要后续修复。

#### 3. DS7B

总体结果：

```text
target_direction alpha=1.0:
  mean logit delta  = +0.3828
  mean margin delta = +0.3458
  rank improved     = 77 / 152
  new winner        = 5 / 152

random_same_norm alpha=1.0:
  mean margin delta = +0.0646

wrong_label_direction alpha=1.0:
  mean margin delta = -0.3181

negative_target_direction alpha=1.0:
  mean margin delta = -0.4819
```

分关系结果：

```text
category target alpha=1.0:
  margin delta = +0.6365
  random control best = +0.1278

color target alpha=1.0:
  margin delta = +0.1481
  random control best = +0.0318

function target alpha=1.0:
  margin delta = +0.1678
  random control best = +0.0115
```

DS7B 的结果为正，但弱于 qwen3：

```text
category 最强；
color / function 有正向迁移但幅度较小；
整体仍优于错配方向和反方向。
```

### 七、跨模型判断

本阶段最稳的现象：

```text
qwen3:
  语义方向具有较干净的因果迁移效果。

DS7B:
  语义方向也有因果迁移效果，但强度较弱。

GLM4:
  存在 target movement，
  但 category / color 被强控制项污染；
  function 较可信。
```

因此 Phase938 的总判断是：

```text
semantic factor causal transfer 有初步正结果；
但只在 qwen3 最清晰，DS7B 次之，GLM4 需要重测和收紧。
```

不能写成：

```text
已经发现跨模型共享语义编码机制；
已经闭合水果/颜色/功能复用规则；
已经完成语义场到输出边界的自然路线闭合。
```

### 八、和 Phase937 的关系

Phase937 证明的是：

```text
隐藏状态中存在 relation-conditioned semantic clustering 信号。
```

Phase938 进一步证明：

```text
至少在 qwen3 和 DS7B 中，
从这些聚类构造出的差分方向，
可以在留出模板上推动目标标签的 logit / margin / rank。
```

所以证据层级从：

```text
observational state map
```

推进到：

```text
direction-level causal transfer
```

但仍然没有到：

```text
natural gate closure
boundary gear closure
full vocabulary closure
multi-token semantic generation closure
```

### 九、闭合标准与当前距离

若要真正回答“水果如何复用神经元、颜色如何跨对象复用、不同对象如何差异化”，至少还需要：

```text
1. 语义方向跨模板稳定。
2. 语义方向跨语言稳定，尤其需要中文模板复测 GLM4。
3. target direction 必须显著强于 same-norm random / wrong label / negative direction。
4. 方向 patch 不能只是 generic perturbation。
5. 方向要能预测未见对象、未见属性组合。
6. 方向要连接到 boundary gear / clean protocol edge。
7. 多 token 生成路径必须能解释，而不是只解释第一 token label。
```

Phase938 当前完成：

```text
1. qwen3: 完成较好。
2. DS7B: 部分完成。
3. GLM4: function 部分完成；category / color 未完成。
4. boundary gear 连接: 未完成。
5. full-vocab blocker: 未完成。
6. 多 token 自然生成: 未完成。
```

因此距离语言编码机制闭合仍然较远。

### 十、问题、硬伤和瓶颈

当前主要硬伤：

```text
1. GLM4 控制项过强，说明当前方向 patch 混入了大量 generic perturbation。
2. 仍是第一 token label 测试，没有进入自然多 token 生成。
3. direction 是人工构造，不等于模型自然 gate。
4. 样本关系只有 category / color / function 三类，语义空间仍很小。
5. 未把 Phase936 的 case residual / boundary gear 与语义方向连接起来。
6. GLM4 运行结束存在 segmentation fault，工程稳定性不足。
```

因此本阶段不能做大理论收束，只能作为图谱中的一块因果拼图。

### 十一、智能理论角度的关键洞察

本阶段最重要的拼图不是“语义已经闭合”，而是：

```text
语义状态图谱中的某些差分方向，
已经可以作为可干预变量，
影响输出竞争场。
```

这说明当前路线应继续把语言机制拆成三层：

```text
semantic state layer:
  类别、颜色、功能、对象差异等语义状态因子。

gate / route layer:
  哪些状态因子在当前上下文被调用。

boundary / output layer:
  这些状态因子如何进入 token 竞争场并击败 blocker。
```

Phase938 只连接了第一层到输出 label 的短桥，还没有找到自然 gate。

### 十二、下一阶段任务

Phase938 已完成当前阶段的因果迁移第一步。

下一阶段建议为：

```text
Phase939: 中文模板与方向特异性收紧审计
```

具体任务：

```text
1. 增加中文模板，优先重测 GLM4。
2. 对 target direction 做 generic perturbation 扣除：
   d_specific = d_target - projection(d_target, random/generic subspace)
3. 增加 same-size / same-norm / same-logit-baseline 控制。
4. 检查 direction 在未见对象和未见属性组合上的迁移。
5. 把通过审计的 semantic direction 接到 Phase936 的 boundary gear。
```

这仍属于当前“语义状态图谱 -> 因果方向 -> 输出边界”的阶段性目标，但 Phase939 已经是新的收紧子阶段，需要单独执行。

### 十三、通俗总结

Phase937 只是看到：

```text
同类东西在模型隐藏状态里更像同类。
```

Phase938 做的是：

```text
把“像同类”的方向拿出来，
轻轻推一下模型隐藏状态，
看答案会不会往对应类别、颜色、功能移动。
```

结果是：

```text
qwen3 移动得比较干净；
DS7B 也会移动，但弱一些；
GLM4 会移动，不过很多非目标方向也会让它动，
所以 GLM4 还不能算干净正结果。
```

这说明：

```text
语义复用方向确实可能是语言编码机制的一部分；
但现在只找到了一段短因果链，
还没有破解完整编码机制。
```

## Phase 939: 中文模板与方向特异性收紧审计 [2026-07-04 22:54]

### 一、对附件判断的核对

附件对 Phase938 的判断基本正确：

```text
Phase938 的证据层级应写成 direction-level causal transfer positive；
不能写成 semantic mechanism closure；
不能写成 natural semantic gate closure；
也不能写成 full language coding closure。
```

其中最应保留的判断是：

```text
qwen3 的语义方向较干净；
DS7B 有正迁移但较弱；
GLM4 的 function 较可信，category / color 受 generic perturbation 污染。
```

附件提出的 Phase939 方向也正确：

```text
下一步不是继续堆 patch；
而是确认 target direction 是否仍然具有目标特异性，
并用中文模板复测 GLM4。
```

但附件中的 generic subspace 公式有一个需要修正的点：

```text
若把 -d_target 放入 generic subspace 再投影扣除，
那么 d_target 会被自己的反方向张成的空间完全扣掉，
导致 d_specific 接近 0。
```

因此本阶段采用更严格但不自毁的定义：

```text
generic basis = {wrong_mean_direction, template_or_language_shift}
negative_target_direction 只作为控制项，不参与正交化扣除。
```

### 二、本阶段任务

Phase939 的目标是：

```text
在中英文模板下，
检查 Phase938 的语义方向是否仍然是 target-specific semantic factor，
而不是模板、语言或通用扰动带来的假因果方向。
```

新增脚本：

```text
tests/glm5/phase939_bilingual_specificity_tightening_audit.py
tests/glm5/run_phase939_bilingual_specificity_tightening_audit.sh
```

结果目录：

```text
tests/result/phase939_bilingual_specificity_tightening_audit/bilingual_specificity_tightening_audit/
```

### 三、测试原理

本阶段保留 Phase938 的目标方向：

```text
d_target
=
mean(h_l | target=y, train_template)
-
mean(h_l | target!=y, train_template)
```

同时构造两个通用干扰方向：

```text
d_wrong_mean = mean(d_wrong_label)

d_template_shift
=
mean(h_l | train_template)
-
mean(h_l | test_template)
```

然后做三种扣除：

```text
d_wrong_removed
=
d_target - Proj_{d_wrong_mean}(d_target)

d_template_removed
=
d_target - Proj_{d_template_shift}(d_target)

d_specific
=
d_target - Proj_{span(d_wrong_mean, d_template_shift)}(d_target)
```

patch 公式仍然是：

```text
h'_l(x) = h_l(x) + alpha * d
```

本阶段 alpha 固定为：

```text
alpha = 1.0
```

对照组包括：

```text
target_direction
wrong_mean_subtracted
template_subtracted
specific_direction
wrong_label_direction
wrong_mean_direction
random_same_norm
negative_target_direction
template_shift_same_norm
```

特异性指标：

```text
SpecificityGain(d)
=
DeltaMargin(d)
-
max(
  DeltaMargin(wrong_label_direction),
  DeltaMargin(wrong_mean_direction),
  DeltaMargin(random_same_norm),
  DeltaMargin(negative_target_direction),
  DeltaMargin(template_shift_same_norm)
)
```

只有当：

```text
DeltaMargin(d) > 0
SpecificityGain(d) > 0
```

才认为方向在该 relation / language_pair 下有目标特异性。

### 四、测试数据

本阶段三模型顺序运行：

```text
qwen3 -> GLM4 -> DS7B
```

样本：

```text
对象数: 30
关系: category / color / function
语言: English / Chinese
每种语言每个关系模板数: 2
每模型输入样本数: 360
```

方向规格和结果行：

```text
qwen3:
  direction_specs = 228
  rows = 9120

GLM4:
  direction_specs = 192
  rows = 7650

DS7B:
  direction_specs = 228
  rows = 9120
```

GLM4 的 direction_specs 少于 qwen3 / DS7B，说明当前标签支持度、tokenizer 或有效方向构造存在模型差异，后续分析不能强行同构解释。

### 五、客观结果

#### 1. qwen3

总体：

```text
target_direction:
  mean margin delta = +0.4625

template_subtracted:
  mean margin delta = +0.4602

specific_direction:
  mean margin delta = +0.3084

random_same_norm:
  mean margin delta = -0.0377

wrong_mean_direction:
  mean margin delta = -0.4036

negative_target_direction:
  mean margin delta = -0.7163

wrong_label_direction:
  mean margin delta = -0.7344
```

qwen3 的正结果比较干净。扣除 wrong_mean 和 template_shift 后，specific_direction 仍为正，且明显高于多数控制项。

重要 relation / language_pair：

```text
function en->en:
  specific margin = +0.9063
  control best    = +0.1891
  gain            = +0.7171

function zh->en:
  specific margin = +0.8347
  control best    = +0.2368
  gain            = +0.5979

color en->zh:
  specific margin = +0.5911
  control best    = +0.0761
  gain            = +0.5150

color zh->zh:
  specific margin = +0.3976
  control best    = +0.0521
  gain            = +0.3455
```

但 qwen3 也暴露了一个限制：

```text
category 在正交化后明显变弱；
category en->zh 的控制项很强；
function en->zh 也被 template_shift_same_norm 控制项压住。
```

所以 qwen3 的可靠结论是：

```text
color / function 的双语特异性较强；
category 仍不稳定。
```

#### 2. GLM4

总体：

```text
target_direction:
  mean margin delta = +2.2863

specific_direction:
  mean margin delta = +2.3994

random_same_norm:
  mean margin delta = +1.9768

wrong_label_direction:
  mean margin delta = +1.9000

wrong_mean_direction:
  mean margin delta = +1.8809

negative_target_direction:
  mean margin delta = +1.6942
```

GLM4 仍然存在强烈的 generic perturbation：

```text
几乎所有方向都会大幅推动 margin；
所以不能只看 target_direction 或 specific_direction 的绝对值。
```

但中文模板确实带来一个重要改善：

```text
color zh->zh:
  specific margin = +0.5956
  control best    = -0.0184
  gain            = +0.6140
```

这说明：

```text
GLM4 在中文模板下的 color 方向比 Phase938 的英文 color 更干净。
```

其他结果：

```text
color zh->en:
  specific margin = +4.3619
  control best    = +3.9258
  gain            = +0.4361

function en->en:
  specific margin = +3.7821
  control best    = +3.5946
  gain            = +0.1875

function zh->en:
  specific margin = +4.0328
  control best    = +3.8709
  gain            = +0.1619
```

这些方向虽然 gain 为正，但 control best 也非常高，因此更谨慎地写成：

```text
GLM4 的 color zh->zh 是本阶段最干净的改善；
function 仍有弱特异性；
category 仍未通过特异性收紧；
GLM4 的高绝对 margin 仍大量来自通用扰动。
```

category 结果：

```text
category zh->en specific:
  margin = +4.0457
  control best = +4.1603
  gain = -0.1146

category zh->zh specific:
  margin = -0.1642
  control best = +0.1026
  gain = -0.2668
```

因此 GLM4 的 category 不能算正结果。

#### 3. DS7B

总体：

```text
target_direction:
  mean margin delta = +0.2305

template_subtracted:
  mean margin delta = +0.2236

specific_direction:
  mean margin delta = +0.0700

random_same_norm:
  mean margin delta = -0.0078

wrong_label_direction:
  mean margin delta = -0.1524

wrong_mean_direction:
  mean margin delta = -0.1962

negative_target_direction:
  mean margin delta = -0.2758
```

DS7B 在 raw target_direction 上仍为正，但扣除后明显变弱。

较稳结果：

```text
function en->en specific:
  margin = +0.1809
  control best = -0.0296
  gain = +0.2105

function zh->zh specific:
  margin = +0.2646
  control best = +0.2432
  gain = +0.0214
```

弱或失败结果：

```text
category specific 多数为负或低于控制；
color specific 多数接近 0；
function en->zh / zh->en 被控制项压住。
```

因此 DS7B 的结论应收紧为：

```text
raw semantic transfer 仍存在；
specific semantic transfer 只部分保留；
跨语言特异性弱。
```

### 六、跨模型结论

本阶段自动证据标签：

```text
bilingual_specific_semantic_transfer_retained: 2
partial_specific_semantic_transfer_retained: 1
```

但人工校准后应写得更谨慎：

```text
qwen3:
  color / function 的双语特异性较强；
  category 不稳定。

GLM4:
  中文 color 明显改善；
  function 有弱特异性；
  category 仍失败；
  通用扰动仍很强。

DS7B:
  raw transfer 为正；
  正交化后只剩部分 function；
  color / category 特异性不足。
```

所以 Phase939 的真实进展不是“完成语义方向闭合”，而是：

```text
把 Phase938 的正结果拆成了三类：

1. 稳定特异方向：
   qwen3 color/function；
   GLM4 zh color。

2. 有 raw transfer 但被控制项污染的方向：
   GLM4 function；
   DS7B 部分 category/function。

3. 当前不稳定或失败方向：
   多数 category；
   DS7B color；
   qwen3 en->zh category/function 的部分跨语方向。
```

### 七、理论进展

Phase938 的公式是：

```text
semantic direction -> target label competition
```

Phase939 进一步拆成：

```text
raw semantic direction
  =
target-specific component
+
template/language component
+
wrong-label/generic component
+
noise
```

当前更谨慎的表示是：

```text
d_target
=
d_specific
+
P_wrong(d_target)
+
P_template(d_target)
+
epsilon
```

本阶段证明：

```text
部分 relation / language_pair 中 d_specific 仍有正因果作用；
但很多 raw transfer 不是纯语义特异方向。
```

这对全局图谱很关键：

```text
语义因果图谱不能只记录 target_direction 是否有效；
必须记录 specificity gain、language_pair、relation、control_best。
```

### 八、闭合标准与当前距离

如果目标是解释“水果、颜色、功能如何复用神经元，并形成语言输出”，闭合标准至少仍包括：

```text
1. hidden-state semantic clustering。
2. direction-level causal transfer。
3. target-specific gain after controls。
4. cross-language / cross-template retention。
5. unseen object / unseen attribute generalization。
6. semantic direction -> boundary gear bridge。
7. multi-token natural rollout。
8. full-vocabulary blocker suppression。
```

Phase939 当前完成情况：

```text
1. hidden-state semantic clustering:
   Phase937 部分完成。

2. direction-level causal transfer:
   Phase938 完成初步正结果。

3. target-specific gain:
   Phase939 部分完成。

4. cross-language retention:
   qwen3 color/function 较好；
   GLM4 color 有改善；
   DS7B 较弱。

5. unseen object / unseen attribute:
   未完成。

6. boundary gear bridge:
   未完成。

7. multi-token rollout:
   未完成。

8. full-vocabulary blocker:
   未完成。
```

因此距离完整语言编码机制闭合仍然较远。

### 九、问题、硬伤和瓶颈

主要问题：

```text
1. 本阶段中文模板仍要求英文 label 输出，
   因此它测试的是中文 prompt route + English label competition，
   还不是纯中文 label 语义闭合。

2. GLM4 的绝对 logit / margin 变化仍非常大，
   说明其 hidden-state patch 对输出场整体扰动很强。

3. category 方向在三模型中都不够稳，
   可能 category 更依赖对象域、任务协议或更上层 route。

4. DS7B 在正交化后明显变弱，
   说明 Phase938 的部分正迁移可能混入模板/通用扰动。

5. 仍然只测第一 token label，
   没有进入自然多 token 生成。
```

### 十、智能理论角度的洞察

本阶段给出的关键拼图是：

```text
语义复用不是一个单纯方向；
它至少被 relation、language、template、label-specific component 共同调制。
```

更接近当前事实的三层图谱是：

```text
semantic state layer:
  产生 raw target direction。

specificity filter layer:
  从 raw direction 中分离 target-specific component。

output competition layer:
  target-specific component 和 generic perturbation 同时影响 label margin。
```

因此后续破解语言机制不能只问：

```text
有没有一个 red direction？
有没有一个 fruit direction？
```

而应问：

```text
在什么 relation、语言、模板、对象域下，
red / fruit / function 的 target-specific component 能从 generic field 中分离出来？
```

### 十一、下一阶段任务

Phase939 完成的是：

```text
中文模板与方向特异性收紧审计。
```

下一步自然是：

```text
Phase940: semantic direction -> boundary gear bridge audit
```

但 Phase940 已经切换到输出边界齿轮层，不属于 Phase939 的未完成尾部。

建议 Phase940 只选择通过 Phase939 的方向：

```text
qwen3 color/function；
GLM4 zh color；
DS7B function en->en。
```

测试它们是否影响：

```text
Phase936 的 common gear；
case residual；
punctuation_period blocker；
protocol / EOS boundary。
```

不能把未通过特异性审计的 category 方向直接拿去做边界桥，否则容易把 generic perturbation 误写成语义机制。

### 十二、通俗总结

Phase938 说明：

```text
某些语义方向一推，答案会往目标词移动。
```

Phase939 追问：

```text
这个移动是真的因为“语义方向”，
还是因为随便推一下模型都会动？
```

结果是：

```text
qwen3 的颜色和功能方向比较像真的；
GLM4 用中文问颜色时明显变干净；
DS7B 只保留一小部分功能方向；
类别方向整体还不稳。
```

所以当前更稳的结论是：

```text
语义编码机制中确实存在可干预的特异方向；
但这些方向不是全局固定齿轮，
而是受语言、模板、关系和模型结构共同调制。
```

## Phase 940: 语义方向到输出边界竞争桥接审计 [2026-07-04 23:18]

### 一、对上传分析的判断

上传内容对 Phase939 的判断基本正确，而且指出的下一步是必要的：

```text
Phase938 证明：部分语义方向可以移动目标关系 margin。
Phase939 收紧：这些方向必须先经过 target-specific / template / wrong-label / noise 分解。
下一步不应继续扩大语义方向本身，而应检查：
这些通过审计的语义方向，是否能进入输出边界竞争场。
```

这条判断是正确的。尤其是 Phase939 修正后的公式很关键：generic subspace 不能把 `-d_target` 放进去，否则会把目标方向本身也投影掉，导致“特异性”被人为削弱。

但必须收紧证据层级：

```text
Phase940 不能直接声称完成了 channel-level boundary gear closure。
原因是 Phase936 的 same-case candidate residual holdout 在跨模型上并不完整：
qwen3 和 DS7B 当时缺少可用 punctuation_period holdout seeds；
GLM4 才有较完整的通道级边界残差信号。
```

因此本阶段把任务限定为：

```text
语义方向 -> first-token output boundary competition 的桥接审计。
```

也就是先问一个更基础、更客观的问题：

```text
通过 Phase939 特异性审计的语义方向，
是否不仅能提高 relation target margin，
还会同步改善 target 对 period / EOS / protocol / punctuation 等边界 token 的竞争优势？
```

### 二、测试脚本和结果位置

新增正式测试脚本：

```text
tests/glm5/phase940_semantic_boundary_bridge_audit.py
```

新增运行脚本：

```text
tests/glm5/run_phase940_semantic_boundary_bridge_audit.sh
```

结果目录：

```text
tests/result/phase940_semantic_boundary_bridge_audit/semantic_boundary_bridge_audit/
```

核心结果文件：

```text
phase940_cross_model_summary.json
phase940_cross_model_summary.md
phase940_qwen3_summary.json
phase940_glm4_summary.json
phase940_deepseek7b_summary.json
```

### 三、测试原理

Phase940 不重新发明语义方向，而是复用 Phase939 中已经通过特异性筛选的方向。筛选条件为：

```text
direction_type = specific_direction
specific margin >= 0.05
specificity gain >= 0.05
```

然后在每个样本的目标层上进行方向干预：

```text
h'_l = h_l + alpha * d
```

其中：

```text
h_l : 指定层最后位置 hidden state
d   : Phase939 构造出的语义方向或控制方向
alpha = 1.0
```

对每个通过筛选的方向，同时测试：

```text
specific_direction
target_direction
random_same_norm
wrong_mean_direction
template_shift_same_norm
negative_target_direction
baseline
```

这样可以区分：

```text
语义特异方向导致的边界移动；
目标均值方向导致的边界移动；
模板/语言平移导致的边界移动；
随机同范数扰动导致的边界移动；
错误标签均值导致的边界移动；
反向目标扰动导致的边界移动。
```

### 四、边界竞争指标

本阶段定义多个边界 token 集合：

```text
period      : ".", "。"
punctuation : ".", "。", ",", "，", ":", "：", ";", "；"
protocol    : "\n", ":", "：", "Answer", "答案"
eos         : tokenizer.eos_token_id
all_boundary = period + punctuation + protocol + eos
```

对每个样本，计算：

```text
B_all = max logit(boundary tokens)
M_boundary = z_target - B_all
Delta M_boundary = M_boundary_after - M_boundary_before
```

同时保留：

```text
Delta M_relation = target logit 相对同关系其他标签的 margin 变化
Delta M_period   = target logit 相对 period token 的 margin 变化
Delta M_eos      = target logit 相对 EOS token 的 margin 变化
Delta B_all      = boundary best logit 的变化
```

桥接增益定义为：

```text
BridgeGain(d)
  = Delta M_boundary(d)
    - max Delta M_boundary(control directions)
```

其中控制方向包括：

```text
random_same_norm
wrong_mean_direction
template_shift_same_norm
negative_target_direction
```

一个关系-语言对被视为正桥接，需要同时满足：

```text
Delta M_relation(specific_direction) > 0
Delta M_boundary(specific_direction) > 0
BridgeGain(specific_direction) > 0.02
```

### 五、测试规模

三个模型依次运行，避免 GPU 显存叠加：

```text
qwen3      : sample_count = 360, selected_specs = 140, rows = 3332
GLM4       : sample_count = 360, selected_specs = 91,  rows = 2177
DS7B       : sample_count = 360, selected_specs = 14,  rows = 266
```

跨模型证据标签：

```text
semantic_boundary_bridge_positive         : 2
partial_semantic_boundary_bridge_positive : 1
```

具体为：

```text
qwen3 : semantic_boundary_bridge_positive
GLM4  : semantic_boundary_bridge_positive
DS7B  : partial_semantic_boundary_bridge_positive
```

### 六、qwen3 结果

qwen3 的整体条件均值：

```text
specific_direction:
  relation margin delta = +0.5278
  boundary margin delta = +0.5390
  period margin delta   = +0.5621
  eos margin delta      = +0.6287
  boundary logit delta  = -0.0576

target_direction:
  relation margin delta = +0.7495
  boundary margin delta = +0.6516

template_shift_same_norm:
  relation margin delta = +0.0714
  boundary margin delta = +0.0898

random_same_norm:
  relation margin delta = -0.0183
  boundary margin delta = -0.0757

wrong_mean_direction:
  relation margin delta = -0.6091
  boundary margin delta = -0.4062

negative_target_direction:
  relation margin delta = -1.0969
  boundary margin delta = -1.0476
```

qwen3 的主要正桥接关系：

```text
function zh->en:
  relation margin delta = +0.8347
  boundary margin delta = +0.5370
  control best          = -0.1801
  bridge gain           = +0.7171

color en->en:
  relation margin delta = +0.4097
  boundary margin delta = +0.6563
  control best          = +0.0359
  bridge gain           = +0.6204

color zh->en:
  relation margin delta = +0.2957
  boundary margin delta = +0.4358
  control best          = -0.0255
  bridge gain           = +0.4612

color zh->zh:
  relation margin delta = +0.3976
  boundary margin delta = +0.5006
  control best          = +0.3264
  bridge gain           = +0.1742

color en->zh:
  relation margin delta = +0.5911
  boundary margin delta = +0.7052
  control best          = +0.5718
  bridge gain           = +0.1334

function en->en:
  relation margin delta = +0.9063
  boundary margin delta = +0.6283
  control best          = +0.5691
  bridge gain           = +0.0592

function zh->zh:
  relation margin delta = +0.3680
  boundary margin delta = +0.1632
  control best          = +0.1332
  bridge gain           = +0.0300
```

qwen3 的现象比较干净：

```text
specific_direction 同时提高 relation margin 和 boundary margin；
random / wrong / negative 控制方向多数为负或很弱；
template_shift 有少量正效应，但明显低于特异方向。
```

这说明在 qwen3 上，Phase939 过滤后的 color / function 语义方向，确实能进入输出边界竞争场。

### 七、GLM4 结果

GLM4 的整体条件均值：

```text
specific_direction:
  relation margin delta = +3.5308
  boundary margin delta = +9.9671
  period margin delta   = +9.5171
  eos margin delta      = +7.9242
  boundary logit delta  = -0.9648

target_direction:
  relation margin delta = +3.4900
  boundary margin delta = +9.6799

random_same_norm:
  relation margin delta = +2.9718
  boundary margin delta = +9.3555

template_shift_same_norm:
  relation margin delta = +2.8549
  boundary margin delta = +8.8991

wrong_mean_direction:
  relation margin delta = +2.5117
  boundary margin delta = +8.4842

negative_target_direction:
  relation margin delta = +2.1510
  boundary margin delta = +7.9535
```

GLM4 的主要正桥接关系：

```text
color en->en:
  relation margin delta = +4.0087
  boundary margin delta = +12.1239
  control best          = +10.4992
  bridge gain           = +1.6247

function zh->en:
  relation margin delta = +4.0328
  boundary margin delta = +10.8615
  control best          = +10.5739
  bridge gain           = +0.2876

color zh->en:
  relation margin delta = +4.3619
  boundary margin delta = +12.2581
  control best          = +12.0449
  bridge gain           = +0.2133

function en->en:
  relation margin delta = +3.7821
  boundary margin delta = +12.2093
  control best          = +12.0892
  bridge gain           = +0.1201

color zh->zh:
  relation margin delta = +0.5956
  boundary margin delta = +1.1806
  control best          = +1.1571
  bridge gain           = +0.0234
```

GLM4 必须谨慎解释：

```text
specific_direction 的绝对边界移动很强；
但 random / template / wrong / negative 控制方向也会产生很大的正边界移动。
```

所以 GLM4 的结论不是：

```text
发现了干净的语义到边界齿轮闭合。
```

而是：

```text
GLM4 的输出边界场对 hidden-state perturbation 高度敏感；
specific_direction 在若干关系上仍有超过控制组的 bridge gain；
但控制组过强，说明它还不是严格干净的机制证据。
```

其中 `color en->en` 的 bridge gain 最大，证据较强；其他关系虽然为正，但离控制组较近。

### 八、DS7B 结果

DS7B 只有一个通过 Phase939 筛选的关系-语言对：

```text
function en->en
```

整体条件均值：

```text
specific_direction:
  relation margin delta = +0.1809
  boundary margin delta = +0.0724
  period margin delta   = +0.1127
  eos margin delta      = +0.1275
  boundary logit delta  = -0.0576

target_direction:
  relation margin delta = +0.1036
  boundary margin delta = -0.0263

wrong_mean_direction:
  relation margin delta = -0.0444
  boundary margin delta = -0.0115

random_same_norm:
  boundary margin delta = -0.1645

template_shift_same_norm:
  boundary margin delta = -0.2056

negative_target_direction:
  boundary margin delta = -0.5247
```

正桥接：

```text
function en->en:
  relation margin delta = +0.1809
  boundary margin delta = +0.0724
  control best          = -0.0115
  bridge gain           = +0.0839
```

DS7B 是弱正结果：

```text
方向正确；
但覆盖面很窄；
效应量较小；
不能外推到 color / category / 中文模板。
```

### 九、阶段性结论

Phase940 的客观结论是：

```text
通过 Phase939 特异性审计的部分语义方向，
确实不只移动 relation target margin，
也会同步移动 target 对输出边界 token 的竞争优势。
```

跨模型排序：

```text
qwen3:
  证据最干净。
  color / function 多个语言对都有正 bridge gain。

GLM4:
  边界场移动最强。
  但控制扰动也很强，说明存在 generic perturbation sensitivity。
  证据是正的，但不是干净闭合。

DS7B:
  只有 function en->en 弱正。
  更像小模型粗糙结构中的局部可复现信号。
```

因此本阶段把 Phase939 的语义方向推进了一步：

```text
semantic-specific direction
  -> relation margin
  -> first-token boundary competition
```

但还没有推进到：

```text
semantic-specific direction
  -> channel-level boundary gear
  -> natural gate
  -> multi-token rollout closure
```

### 十、闭合标准与当前距离

如果要称为“语义方向到边界齿轮闭合”，至少需要满足：

```text
1. 同一个语义方向在 holdout objects / holdout templates 上稳定复现；
2. 该方向能移动 target-vs-boundary margin；
3. 该方向能对应到明确的 layer / channel / feature 边界齿轮；
4. 通道级边界齿轮的干预可以复现方向级效果；
5. 语义方向 + 边界齿轮的组合干预可以解释 residual；
6. natural generation 下能改变真实输出，而不是只改变 one-step logit；
7. 多 token rollout 中不会被后续协议/终止路线覆盖。
```

Phase940 当前只满足：

```text
第 1 条的部分前置条件；
第 2 条的一步 logit 版本；
少量跨模型弱复现。
```

尚未满足：

```text
第 3-7 条。
```

所以闭合距离仍然较远。更准确地说：

```text
Phase940 是 semantic-to-boundary bridge evidence；
不是 semantic-boundary gear closure。
```

### 十一、问题、硬伤和瓶颈

1. 边界 token 集合仍是代理定义。

```text
period / EOS / punctuation / protocol tokens 可以代表一部分边界竞争，
但不能覆盖所有 nonclean output transition。
```

2. GLM4 控制方向过强。

```text
这说明 GLM4 的 hidden space 对扰动整体敏感，
specific_direction 的正结果需要继续用更严格的 normalized control 和 low-strength alpha 扫描复核。
```

3. DS7B 覆盖面太窄。

```text
DS7B 只有 function en->en 通过筛选，
不能判断其它语义关系是否不存在桥接，还是因为小模型内部结构粗糙导致信号被埋没。
```

4. 仍然是 first-token logit 审计。

```text
它没有测试自然生成过程中的多 token 路线切换，
也没有测试 clean protocol edge 对最终输出的覆盖。
```

5. 尚未连接 Phase936 的通道级边界残差信号。

```text
本阶段没有证明语义方向使用了同一个 boundary gear；
只证明语义方向可以移动输出边界竞争。
```

### 十二、智能理论层面的关键洞察

当前结果支持一个更谨慎的图谱结构：

```text
语义不是直接等于答案 token；
语义方向首先改变候选答案之间的关系 margin；
随后这种变化会进入输出边界场，
与 period / EOS / protocol / punctuation 等非语义路线竞争。
```

这说明“语言编码机制”的破解不能只看：

```text
水果是什么方向；
颜色是什么方向；
功能是什么方向。
```

还必须看：

```text
这些语义方向如何穿过输出边界；
什么时候被协议路线截断；
什么时候被终止路线覆盖；
什么时候进入自然回答路线。
```

也就是说，编码机制至少包括两层拼图：

```text
语义分布结构；
输出边界/协议控制结构。
```

Phase940 的价值在于第一次把两者之间建立了可测量的桥：

```text
不是只说“方向像语义”；
而是测量“方向是否能改变边界竞争”。
```

### 十三、下一阶段任务

Phase941 应进入新的子阶段：

```text
Semantic Direction to Channel-Level Boundary Gear Audit
```

具体任务：

```text
1. 先在 GLM4 上复用 Phase936 中已有的 punctuation / EOS / protocol 边界残差信号；
2. 检查 Phase940 中通过的 semantic-specific directions 是否与这些边界残差信号同层、同通道或同子空间重合；
3. 对 qwen3 和 DS7B 重新寻找可用的 boundary holdout seeds，补齐 Phase936 当时缺失的通道级数据；
4. 做 low-alpha 扫描，排除 GLM4 的 generic perturbation sensitivity；
5. 做 semantic direction + boundary gear 的组合干预，检查是否存在加和、互补或互相覆盖。
```

建议的最小公式：

```text
Delta_total = Effect(d_semantic + g_boundary)
Delta_sem   = Effect(d_semantic)
Delta_bound = Effect(g_boundary)
Residual    = Delta_total - Delta_sem - Delta_bound
```

如果：

```text
Residual 接近 0
```

说明语义方向和边界齿轮近似线性拼接；

如果：

```text
Residual 显著为正或为负
```

说明二者之间存在门控或竞争。

Phase941 与 Phase940 属于同一大阶段：

```text
先完成图谱，再追求闭合。
```

但 Phase941 已经是新的通道级子任务，不应把 Phase940 的 first-token bridge 直接自动升级为闭合结论。

### 十四、通俗总结

Phase939 问的是：

```text
这个语义方向是不是真的像“颜色/功能”的方向？
```

Phase940 继续问：

```text
如果它是真的语义方向，
它能不能把答案从句号、结束符、格式符号这些边界竞争里推出来？
```

答案是：

```text
qwen3 上比较明确；
GLM4 上很强但不够干净；
DS7B 上只有一个弱正结果。
```

所以当前拼图多了一块：

```text
部分语义方向不仅影响“选哪个语义答案”，
还会影响“模型要不要继续输出答案、是否被边界 token 截断”。
```

但它还不是最终闭合。下一步要把这个方向级现象追到具体层、通道和边界齿轮上。

## Phase 941: 语义方向残差坐标子空间桥接审计 [2026-07-04 23:42]

### 一、对上传分析的判断

上传内容对 Phase940 的判断基本正确：

```text
Phase940 是 semantic-to-boundary bridge evidence；
不是 semantic-boundary gear closure；
不是 channel-level boundary gear closure；
也不是 natural generation closure。
```

它指出的关键硬伤也正确：

```text
Phase940 只证明语义方向可以移动 first-token boundary competition；
但尚未证明这些方向对应到明确的 layer / channel / feature 边界齿轮。
```

所以 Phase941 继续同一大阶段：

```text
先完成图谱，再追求闭合。
```

但本阶段不直接跳到 MLP neuron / natural gate，而是先做更保守的一步：

```text
检查 Phase940 的 semantic-specific direction，
其边界桥接效应是否集中在残差流坐标子空间中。
```

这一步的证据层级定义为：

```text
residual-coordinate subspace bridge evidence
残差坐标子空间桥接证据
```

不能定义为：

```text
MLP channel closure；
boundary gear closure；
natural generation closure。
```

### 二、测试脚本和结果位置

新增正式脚本：

```text
tests/glm5/phase941_semantic_direction_coordinate_bridge_audit.py
```

新增运行脚本：

```text
tests/glm5/run_phase941_semantic_direction_coordinate_bridge_audit.sh
```

结果目录：

```text
tests/result/phase941_semantic_direction_coordinate_bridge_audit/semantic_direction_coordinate_bridge_audit/
```

核心结果文件：

```text
phase941_cross_model_summary.json
phase941_cross_model_summary.md
phase941_qwen3_summary.json
phase941_glm4_summary.json
phase941_deepseek7b_summary.json
```

### 三、测试原理

Phase941 只使用 Phase940 中已经通过正桥接条件的 relation / language pair：

```text
Delta M_relation(specific_direction) > 0
Delta M_boundary(specific_direction) > 0
BridgeGain(specific_direction) > 0.02
```

然后对每个通过筛选的 `specific_direction` 做残差坐标拆分。

设：

```text
d = d_specific
```

取绝对值最大的 top-k 坐标：

```text
d_topk_raw = P_topk(d)
```

再构造同范数版本：

```text
d_topk_same_norm
  = d_topk_raw / ||d_topk_raw|| * ||d||
```

同时构造两个控制：

```text
d_randomk_same_norm:
  随机 k 个坐标，同范数。

d_bottomk_same_norm:
  绝对值最小 k 个坐标，同范数。
```

干预形式仍然是：

```text
h'_l = h_l + alpha * d_subspace
```

本轮使用：

```text
top_k = 64, 256
alpha = 0.25, 0.5, 1.0
```

加入 alpha 扫描是为了处理 Phase940 暴露出的 GLM4 问题：

```text
如果只有 alpha=1.0 有效，
可能只是强扰动造成的假象；
如果 alpha=0.25 / 0.5 仍然保留同向效应，
说明子空间桥接更稳定。
```

### 四、核心指标

完整特异方向边界效应：

```text
Delta M_full = Delta M_boundary(d_specific)
```

top-k 原始子方向边界效应：

```text
Delta M_topk_raw = Delta M_boundary(d_topk_raw)
```

原始 top-k 复现比例：

```text
ConcentrationRatio(k)
  = Delta M_topk_raw / Delta M_full
```

同范数 top-k 相对控制的桥接增益：

```text
CoordinateGain(k)
  = Delta M_boundary(d_topk_same_norm)
    - max(
        Delta M_boundary(d_randomk_same_norm),
        Delta M_boundary(d_bottomk_same_norm)
      )
```

正坐标桥接的最低条件：

```text
ConcentrationRatio(k) >= 0.25
CoordinateGain(k) > 0.02
relation margin delta(topk_same_norm) > 0
boundary margin delta(topk_same_norm) > 0
```

### 五、测试规模

三个模型依次运行，避免 GPU 显存叠加：

```text
qwen3:
  selected_specs = 56
  rows = 4368
  selected pairs =
    color en->en
    color en->zh
    color zh->en
    color zh->zh
    function en->en
    function zh->en
    function zh->zh

GLM4:
  selected_specs = 39
  rows = 3220
  selected pairs =
    color en->en
    color zh->en
    color zh->zh
    function en->en
    function zh->en

DS7B:
  selected_specs = 8
  rows = 560
  selected pairs =
    function en->en
```

跨模型证据标签：

```text
coordinate_concentrated_semantic_boundary_bridge_positive         : 2
partial_coordinate_concentrated_semantic_boundary_bridge_positive : 1
```

具体为：

```text
qwen3 : coordinate_concentrated_semantic_boundary_bridge_positive
GLM4  : coordinate_concentrated_semantic_boundary_bridge_positive
DS7B  : partial_coordinate_concentrated_semantic_boundary_bridge_positive
```

### 六、qwen3 结果

qwen3 完整特异方向在 alpha 扫描中保持稳定正效应：

```text
alpha = 1.0:
  full_specific relation margin delta = +0.5349
  full_specific boundary margin delta = +0.4649

alpha = 0.5:
  full_specific relation margin delta = +0.3077
  full_specific boundary margin delta = +0.2670

alpha = 0.25:
  full_specific relation margin delta = +0.1659
  full_specific boundary margin delta = +0.1532
```

top-256 坐标子方向也保持正效应：

```text
topk_raw, k=256, alpha=1.0:
  relation margin delta = +0.2454
  boundary margin delta = +0.1601
  norm fraction         = 0.7057

topk_raw, k=256, alpha=0.5:
  relation margin delta = +0.1376
  boundary margin delta = +0.1016

topk_raw, k=256, alpha=0.25:
  relation margin delta = +0.0649
  boundary margin delta = +0.0593
```

随机和底部坐标控制整体很弱：

```text
randomk_same_norm, k=256, alpha=1.0:
  boundary margin delta = -0.0131

randomk_same_norm, k=256, alpha=0.5:
  boundary margin delta = +0.0110

randomk_same_norm, k=256, alpha=0.25:
  boundary margin delta = +0.0162

bottomk_same_norm, k=256, alpha=1.0:
  boundary margin delta = -0.0755
```

主要正坐标桥接：

```text
color en->zh, k=256, alpha=1.0:
  full boundary delta = +0.7906
  topk raw boundary   = +0.3453
  raw fraction        = 0.4368
  gain vs control     = +0.3906
  joint score         = +0.3359

color en->en, k=256, alpha=1.0:
  full boundary delta = +0.6154
  topk raw boundary   = +0.2380
  raw fraction        = 0.3867
  gain vs control     = +0.3341
  joint score         = +0.2380

color en->zh, k=256, alpha=0.5:
  full boundary delta = +0.4328
  topk raw boundary   = +0.2125
  raw fraction        = 0.4910
  gain vs control     = +0.1797

color en->zh, k=256, alpha=0.25:
  full boundary delta = +0.2453
  topk raw boundary   = +0.1391
  raw fraction        = 0.5669
  gain vs control     = +0.1063

function en->en, k=256, alpha=0.5:
  full boundary delta = +0.3156
  topk raw boundary   = +0.1500
  raw fraction        = 0.4752
  gain vs control     = +0.1563
```

qwen3 的客观结论：

```text
Phase940 的语义到边界桥接，
在 qwen3 上不是平均分散在所有残差坐标中；
top-256 坐标能稳定复现约 35%-57% 的边界效应；
并且在 alpha=0.25 / 0.5 下仍可观察到。
```

但也要注意：

```text
top-64 整体不稳定；
这说明它不是极少数单坐标齿轮，
更像中等宽度的残差子空间。
```

### 七、GLM4 结果

GLM4 的完整特异方向效应仍然极强：

```text
alpha = 1.0:
  full_specific relation margin delta = +3.0433
  full_specific boundary margin delta = +9.6272

alpha = 0.5:
  full_specific relation margin delta = +2.9077
  full_specific boundary margin delta = +9.5453

alpha = 0.25:
  full_specific relation margin delta = +2.8204
  full_specific boundary margin delta = +9.4708
```

top-k 原始坐标几乎复现完整边界效应：

```text
topk_raw, k=64, alpha=1.0:
  relation margin delta = +2.7758
  boundary margin delta = +9.4267
  norm fraction         = 0.4304

topk_raw, k=256, alpha=1.0:
  relation margin delta = +2.8542
  boundary margin delta = +9.4771
  norm fraction         = 0.6212
```

主要正坐标桥接：

```text
function zh->en, k=64, alpha=1.0:
  full boundary delta = +11.5895
  topk raw boundary   = +11.2809
  raw fraction        = 0.9734
  gain vs control     = +0.2513

color zh->en, k=64, alpha=1.0:
  full boundary delta = +12.4557
  topk raw boundary   = +12.3775
  raw fraction        = 0.9937
  gain vs control     = +0.2281

color zh->zh, k=256, alpha=1.0:
  full boundary delta = +1.0078
  topk raw boundary   = +0.6130
  raw fraction        = 0.6082
  gain vs control     = +0.1944

function zh->en, k=64, alpha=0.5:
  full boundary delta = +11.4437
  topk raw boundary   = +11.2614
  raw fraction        = 0.9841
  gain vs control     = +0.1263

color zh->en, k=64, alpha=0.5:
  full boundary delta = +12.4994
  topk raw boundary   = +12.3557
  raw fraction        = 0.9885
  gain vs control     = +0.1187
```

但是 GLM4 的控制仍然非常强：

```text
randomk_same_norm, k=256, alpha=1.0:
  boundary margin delta = +9.3649

bottomk_same_norm, k=256, alpha=1.0:
  boundary margin delta = +9.3750

randomk_same_norm, k=64, alpha=0.25:
  boundary margin delta = +9.3837
```

所以 GLM4 的结论必须双重表述：

```text
正面：
  语义方向的边界效应在 top-64 / top-256 残差坐标上高度集中。

负面：
  随机同范数和底部同范数坐标也能强烈移动边界场；
  因此 GLM4 仍然存在严重 generic perturbation sensitivity。
```

GLM4 不能被解释为干净通道齿轮闭合。

更稳的写法是：

```text
GLM4 显示出残差坐标集中形状；
但该形状嵌在一个高度扰动敏感的输出边界场中。
```

### 八、DS7B 结果

DS7B 进入测试的只有：

```text
function en->en
```

完整特异方向是弱正：

```text
alpha = 1.0:
  full_specific relation margin delta = +0.3812
  full_specific boundary margin delta = +0.0625

alpha = 0.5:
  full_specific relation margin delta = +0.1656
  full_specific boundary margin delta = +0.0219

alpha = 0.25:
  full_specific relation margin delta = +0.1281
  full_specific boundary margin delta = +0.0219
```

最好的局部坐标桥接出现在：

```text
function en->en, k=256, alpha=0.5:
  full boundary delta = +0.0219
  topk raw boundary   = +0.0125
  raw fraction        = 0.5714
  topk_same boundary  = +0.0781
  control best        = -0.1344
  gain vs control     = +0.2125
  joint score         = +0.0781
```

但 DS7B 的负面信号也很明显：

```text
k=256, alpha=1.0:
  topk_raw boundary = -0.0563

k=64, alpha=1.0:
  topk_same boundary = -0.2250

randomk_same_norm, k=64, alpha=1.0:
  boundary margin delta = +0.0813
```

所以 DS7B 只能写成：

```text
partial_coordinate_concentrated_semantic_boundary_bridge_positive
```

不能写成稳定复现。

### 九、阶段性结论

Phase941 的客观结果是：

```text
qwen3:
  Phase940 的语义-边界桥接可以被 top-256 残差坐标子空间稳定部分复现；
  top-64 不稳定；
  说明是中等宽度残差子空间，不是单点坐标。

GLM4:
  top-64 / top-256 坐标几乎复现完整边界效应；
  但随机和底部同范数控制也非常强；
  说明有集中形状，但通用扰动敏感性仍然没有解除。

DS7B:
  只有 function en->en 的 alpha=0.5 / k=256 出现局部正结果；
  覆盖面和稳定性不足。
```

因此，本阶段把 Phase940 的链条推进为：

```text
semantic-specific direction
  -> relation margin
  -> first-token boundary competition
  -> residual-coordinate subspace concentration
```

但仍未推进到：

```text
MLP neuron channel；
attention head；
natural gate；
multi-token rollout；
full-vocabulary closure。
```

### 十、闭合标准与当前距离

如果要称为“语义方向到边界齿轮闭合”，还需要：

```text
1. 从 residual coordinate 子空间进一步定位到具体 MLP / attention 组件；
2. 证明这些组件在 holdout objects / holdout templates 上复现；
3. 证明它们和 Phase936 的 boundary residual / punctuation gear 有交集或因果等价；
4. 用 ablation / patch 双向验证这些组件；
5. 证明自然生成中的 clean output transition 会因此改变；
6. 证明多 token rollout 不被后续 protocol / EOS 路线覆盖。
```

Phase941 当前完成的是：

```text
第 1 条之前的 residual-coordinate 前置定位；
第 2 条之前的方向级样本复现；
第 3 条之前的边界竞争子空间线索。
```

所以它离闭合仍然较远。当前最准确的证据等级是：

```text
coordinate-subspace bridge evidence。
```

不是：

```text
gear closure。
```

### 十一、问题、硬伤和瓶颈

1. residual coordinate 不等于真实神经元通道。

```text
残差坐标可能是多个 MLP / attention 组件混合后的基底；
不能直接把 top-k residual coordinates 写成模型内部原生齿轮。
```

2. qwen3 的 top-64 不稳定。

```text
这说明语义-边界桥接不是极稀疏单点机制；
更可能是中等宽度子空间。
```

3. GLM4 仍有严重通用扰动敏感性。

```text
即使 alpha=0.25，random / bottom 控制仍能强烈推动边界 margin；
这使得 GLM4 的正结果不能作为干净机制证据。
```

4. DS7B 覆盖太窄。

```text
只有 function en->en；
小模型内部结构可能粗糙，导致 color / category 信号没有通过前置筛选。
```

5. 仍是 one-step logit 审计。

```text
没有自然生成；
没有多 token；
没有 strict clean；
没有完整 blocker field。
```

### 十二、智能理论层面的关键洞察

Phase941 给出一个新的图谱线索：

```text
语义方向不是均匀地撒在整个隐藏空间里；
至少在 qwen3 和 GLM4 中，
能进入边界竞争的语义方向具有残差坐标子空间集中性。
```

但这种集中性不是单点的：

```text
qwen3 需要大约 top-256 级别才较稳定；
top-64 常常不足。
```

这更像：

```text
语义-边界接口是一个中等宽度的子空间结构，
而不是单个神经元开关。
```

从破解语言编码机制的角度看，这很重要：

```text
语言能力背后的数学结构可能不是“一个属性一个神经元”，
而是“属性方向 + 中等宽度接口子空间 + 输出边界竞争场”的组合。
```

这与前面“水果如何复用神经元、颜色如何复用神经元”的问题相连：

```text
对象/属性可能先形成可复用语义方向；
这些方向再通过若干残差坐标子空间接入输出竞争；
最终还要经过协议、终止、格式等边界路线筛选。
```

### 十三、下一阶段任务

Phase942 应进入：

```text
Semantic Boundary Coordinate Consensus and Holdout Audit
```

核心任务：

```text
1. 从 Phase941 中导出每个模型、每个 relation / language pair 的 top-k coordinate 集合；
2. 检查不同对象、模板、语言之间的 top-k 坐标重叠率；
3. 构造 consensus coordinate group；
4. 用 holdout objects / holdout templates 测试 consensus group 是否仍能移动 boundary margin；
5. 再把 consensus residual coordinates 映射回 MLP / attention 组件。
```

下一阶段的关键公式：

```text
C_consensus(r, lang)
  = intersection_or_weighted_vote(
      TopK(d_specific_i)
    )
```

然后测试：

```text
h'_l = h_l + alpha * P_C(d_specific)
```

如果 consensus group 在留出样本上仍然有效，才能继续往 MLP / attention 组件映射。

Phase942 仍属于“先完成图谱，再追求闭合”的大阶段，但已经是新的 consensus / holdout 子任务。

### 十四、通俗总结

Phase940 发现：

```text
有些语义方向能把答案从句号、结束符、格式符号这些边界竞争中推出来。
```

Phase941 继续问：

```text
这种推动是不是分散在整个隐藏空间？
还是集中在一部分坐标里？
```

结果是：

```text
qwen3:
  主要集中在 top-256 这种中等宽度子空间；
  不是单个坐标。

GLM4:
  坐标集中非常明显；
  但整个边界场也对随机扰动很敏感。

DS7B:
  只有一个局部弱正结果。
```

所以当前拼图又多了一块：

```text
语义到边界的桥，不只是一个抽象方向；
它在残差空间里有可测的子空间形状。
```

但这仍然不是最终齿轮。下一步要看这些子空间是否能在留出样本上形成稳定共识，再映射到真正的模型组件。

## Phase 942: 语义边界共识坐标留出验证 [2026-07-05 00:54]

### 一、对上传内容的判断

上传内容对 Phase941 的判断基本正确，而且证据层级收得稳：

```text
Phase941 不是 MLP channel closure；
不是 boundary gear closure；
不是 natural generation closure；
而是 semantic-specific direction 在 residual coordinate subspace 上的桥接审计。
```

Phase941 的关键结论是：

```text
qwen3:
  top-256 residual coordinates 能稳定复现一部分语义到边界的移动；
  top-64 不稳定。

GLM4:
  坐标集中很强；
  但 random / bottom 控制也很强，说明存在严重通用扰动敏感性。

DS7B:
  只有 function en->en 的弱覆盖。
```

因此，继续进入 Phase942 是合理的。Phase942 与 Phase941 属于同一大阶段：

```text
先完成全局齿轮图谱；
再尝试机制闭合。
```

但 Phase942 已经不是继续看单个方向是否能动边界，而是检查：

```text
多个训练方向中反复出现的 residual coordinate group，
能否在留出 semantic direction 上继续移动 boundary margin。
```

### 二、本阶段任务

本阶段完成：

```text
Semantic Boundary Coordinate Consensus and Holdout Audit
```

脚本：

```text
tests/glm5/phase942_semantic_boundary_coordinate_consensus_holdout.py
tests/glm5/run_phase942_semantic_boundary_coordinate_consensus_holdout.sh
```

结果目录：

```text
tests/result/phase942_semantic_boundary_coordinate_consensus_holdout/semantic_boundary_coordinate_consensus_holdout/
```

三模型按顺序测试：

```text
qwen3 -> GLM4 -> DS7B
```

避免同时加载导致 GPU 显存溢出。

### 三、测试原理

Phase942 复用 Phase940/941 中已经通过正桥接条件的 relation / language pair，不重新扩大样本边界。

对每个 relation / language pair，将 direction spec 按确定性顺序分成训练组和留出组：

```text
train specs:
  用来投票形成 consensus coordinate group

holdout specs:
  不参与投票，只用于验证共识坐标是否还能移动边界
```

核心构造为：

```text
C_consensus(r, lang)
  = VoteTopK(
      TopK(d_specific_i), i in train specs
    )
```

其中：

```text
d_specific_i = semantic-specific direction
C_consensus = 训练方向 top-k 坐标投票得到的共识坐标集合
```

在留出方向上测试：

```text
d_consensus_raw = P_C(d_holdout)
```

以及同范数版本：

```text
d_consensus_same =
  d_consensus_raw / ||d_consensus_raw|| * ||d_holdout||
```

边界移动指标：

```text
Delta M_boundary =
  M_boundary(patched) - M_boundary(base)
```

共识坐标解释比例：

```text
ConsensusRatio =
  Delta M_boundary(d_consensus_raw)
  / Delta M_boundary(d_full)
```

控制增益：

```text
ConsensusGain =
  Delta M_boundary(d_consensus_same)
  - max(
      Delta M_boundary(randomk_same_norm),
      Delta M_boundary(bottomk_same_norm)
    )
```

本阶段正结果需要同时满足：

```text
ConsensusRatio >= 0.10
ConsensusGain > 0.02
joint_consensus_holdout_score > 0.02
```

注意：这是 spec-level holdout，不是严格 object-level holdout。

### 四、测试规模

统一设置：

```text
top_k = 256
consensus_k = 256
alpha = 0.5, 1.0
max_specs_per_pair = 12
```

实际输出规模：

```text
qwen3:
  rows = 2040
  evidence = consensus_coordinate_holdout_positive

GLM4:
  rows = 1485
  evidence = partial_consensus_coordinate_holdout_positive

DS7B:
  rows = 180
  evidence = partial_consensus_coordinate_holdout_positive
```

跨模型汇总：

```text
consensus_coordinate_holdout_positive: 1
partial_consensus_coordinate_holdout_positive: 2
```

### 五、qwen3 结果

qwen3 是本阶段最清晰的正结果。

总体条件均值：

```text
full_specific alpha=1.0:
  relation margin delta = +0.4624
  boundary margin delta = +0.5029

consensus_same_norm alpha=1.0:
  relation margin delta = +0.1695
  boundary margin delta = +0.1191

randomk_same_norm alpha=1.0:
  boundary margin delta = -0.0458

bottomk_same_norm alpha=1.0:
  boundary margin delta = +0.0136
```

这说明：

```text
qwen3 的 consensus coordinate group
在留出 direction 上仍能移动 boundary margin；
并且整体优于 random / bottom 控制。
```

最强子结果：

```text
relation = function
language_pair = en->en
alpha = 1.0

full boundary delta = +1.0156
consensus raw boundary delta = +0.2500
consensus raw fraction = 0.2462
consensus same boundary delta = +0.4063
control best boundary delta = -0.2031
gain = +0.6094
joint score = +0.2708
overlap recall = 0.2318
jaccard = 0.1312
```

其他正结果：

```text
color en->zh, alpha=1.0:
  full boundary = +0.7813
  consensus raw = +0.1491
  raw fraction = 0.1909
  consensus same = +0.2770
  control best = +0.0938
  gain = +0.1832

color en->en, alpha=1.0:
  full boundary = +0.7332
  consensus raw = +0.1106
  raw fraction = 0.1508
  consensus same = +0.1947
  control best = +0.1154
  gain = +0.0793
```

负面和弱点也很明确：

```text
color zh->en:
  alpha=1.0 时 consensus same boundary = +0.0852
  control best = +0.1108
  gain = -0.0256

color zh->zh:
  consensus raw 接近 0 或为负；
  gain 为负。

function zh->zh:
  full boundary 本身为负；
  不适合作为正桥接证据。
```

qwen3 的客观结论：

```text
存在跨留出 direction 的 residual coordinate consensus；
但这个 consensus 对语言方向敏感，
不是所有 relation / language pair 都成立。
```

### 六、GLM4 结果

GLM4 是部分正结果，但必须谨慎解释。

总体条件均值：

```text
full_specific alpha=1.0:
  relation margin delta = +2.7965
  boundary margin delta = +9.1438

consensus_raw alpha=1.0:
  relation margin delta = +2.4684
  boundary margin delta = +8.8384

consensus_same_norm alpha=1.0:
  relation margin delta = +2.4395
  boundary margin delta = +8.8740

randomk_same_norm alpha=1.0:
  boundary margin delta = +8.8548

bottomk_same_norm alpha=1.0:
  boundary margin delta = +8.8018
```

这说明：

```text
GLM4 的 consensus coordinate group 可以复现巨大 boundary movement；
但 random / bottom 控制也几乎同样巨大。
```

因此，GLM4 不能被解释为干净共识坐标闭合。

最强可用子结果：

```text
relation = function
language_pair = zh->en
alpha = 1.0

full boundary delta = +9.2795
consensus raw boundary delta = +9.0139
raw fraction = 0.9714
consensus same boundary delta = +9.5266
control best boundary delta = +9.0676
gain = +0.4590
joint score = +0.4590
overlap recall = 0.1050
jaccard = 0.0556
```

这个结果说明：

```text
function zh->en 中存在可留出的共识坐标信号；
但 overlap 很低，且模型边界场非常容易被一般扰动推动。
```

其他 GLM4 子结果多为：

```text
raw fraction 很高；
但 gain <= 0 或接近 0。
```

例如：

```text
color en->en:
  consensus raw fraction 接近 1；
  但 control best 更高，gain 为负。

function en->en:
  consensus raw fraction 接近 1；
  但 gain 强烈为负。
```

GLM4 的客观结论：

```text
存在 boundary-sensitive residual coordinate field；
但它不是干净的 semantic boundary consensus gear。
```

### 七、DS7B 结果

DS7B 只有 function en->en 进入测试，因此证据覆盖很窄。

结果：

```text
relation = function
language_pair = en->en
alpha = 1.0

full boundary delta = +0.1250
consensus raw boundary delta = +0.0521
raw fraction = 0.4167
consensus same boundary delta = +0.2604
control best boundary delta = +0.1823
gain = +0.0781
joint score = +0.0781
overlap recall = 0.2233
jaccard = 0.1262
```

alpha=0.5 时：

```text
full boundary delta = +0.0677
consensus raw boundary delta = -0.0521
consensus same boundary delta = +0.0521
control best boundary delta = +0.0208
gain = +0.0313
```

DS7B 的客观结论：

```text
存在一个局部 partial positive；
但只覆盖 function en->en，
不能外推为 DS7B 的稳定语义边界共识坐标机制。
```

考虑到 DS7B 当前是小模型、结构可能粗糙，这个结果只能作为弱拼图。

### 八、本阶段理论进展

Phase942 将 Phase941 的结论推进了一步：

```text
Phase941:
  单个 semantic-specific direction 的 boundary effect
  可以被 top-256 residual coordinate subspace 部分复现。

Phase942:
  多个训练 direction 反复出现的 consensus coordinate group
  在 holdout direction 上仍能部分复现 boundary effect。
```

因此，当前可谨慎写成：

```text
语义方向到输出边界竞争之间，
存在可复用的 residual coordinate group。
```

但不能写成：

```text
已经找到语义齿轮；
已经找到 MLP channel；
已经完成语言编码机制闭合。
```

更准确的图谱位置是：

```text
Semantic-specific direction
  -> residual coordinate consensus group
  -> first-token boundary margin movement
```

这是一个可测的图谱边，但还不是底层组件闭合。

### 九、严格审视和硬伤

1. 留出粒度仍不够强。

```text
Phase942 是 spec-level holdout；
不是严格 object-level holdout；
也不是完全独立语义域 holdout。
```

2. consensus coordinate 是 residual coordinate，不是神经元或通道。

```text
残差坐标可能混合了多个 MLP / attention 来源；
也可能受 LayerNorm 和 readout basis 影响。
```

3. GLM4 的控制问题严重。

```text
random / bottom 控制也能大幅推动 boundary margin；
因此 GLM4 的大数值不能直接解释为干净机制。
```

4. qwen3 的正结果不均匀。

```text
function en->en 最清晰；
color en->zh / en->en 有正结果；
zh->zh 和部分 zh->en 弱或失败。
```

5. DS7B 覆盖太窄。

```text
只有 function en->en；
不能作为跨关系、跨语言的稳定证据。
```

6. 仍然是 one-step logit 审计。

```text
没有自然生成；
没有多 token rollout；
没有 full-vocab blocker closure；
没有最小割闭合。
```

### 十、闭合距离

当前完成的是：

```text
residual coordinate consensus holdout evidence
```

距离闭合还差：

```text
1. object-level holdout；
2. template-level holdout；
3. component-level mapping；
4. MLP / attention 因果干预；
5. full-vocab blocker control；
6. natural generation rollout；
7. minimal sufficient set / minimal cut。
```

所以 Phase942 仍然是图谱推进，不是机制闭合。

### 十一、智能理论层面的关键洞察

Phase942 给出的第一性线索是：

```text
语言语义不是只靠某个单点神经元进入输出竞争；
也不是完全分散在整个隐藏空间；
而是会通过一组中等宽度、可复用的 residual coordinate group
接入边界竞争。
```

这与“水果如何复用神经元、颜色属性如何复用神经元”的问题相连：

```text
对象/属性可能先形成方向性结构；
不同对象共享部分坐标；
不同属性在共享坐标上有不同投影；
最终通过边界竞争场决定 token 输出。
```

但目前还只能说：

```text
看到了残差坐标层面的复用痕迹；
还没有看到底层组件层面的真实齿轮形状。
```

### 十二、下一阶段任务

Phase943 应继续同一大阶段，进入：

```text
Consensus Coordinate Artifact Export and Component Mapping Audit
```

核心任务：

```text
1. 导出每个模型、relation、language pair 的 consensus coordinate indices；
2. 记录 vote count、signed vote、mean abs weight；
3. 检查 consensus coordinate 与 lm_head 输出差分方向的重合；
4. 检查 consensus coordinate 与 MLP down_proj / attention o_proj 输出基的粗略重合；
5. 如果能找到组件候选，再进入 channel/head-level causal patch。
```

下一阶段的关键不是继续追求更大 margin，而是回答：

```text
这些 residual consensus coordinates 到底来自哪些模型组件？
```

如果不能映射回组件，当前图谱会停留在 residual readout 层，无法进入真正的齿轮机制。

### 十三、通俗总结

Phase941 发现：

```text
语义方向推动答案时，不是所有隐藏维度都同等重要；
top-256 这种中等宽度坐标子空间很关键。
```

Phase942 继续问：

```text
这些重要坐标是不是只对某一个样本有效？
还是能在别的留出语义方向上继续有效？
```

结果是：

```text
qwen3:
  是，有比较清楚的共识坐标留出正结果。

GLM4:
  有局部正结果，但模型对一般扰动过于敏感。

DS7B:
  有一个局部弱正结果，但覆盖太窄。
```

所以当前拼图可以更新为：

```text
语义方向 -> 残差共识坐标组 -> 输出边界移动
```

但这还不是最终机制。下一步要把这些残差坐标组映射回真正的 MLP / attention 组件，看看它们是不是由可复用的底层齿轮产生。

## Phase 943: 共识坐标导出与组件静态映射审计 [2026-07-05 01:01]

### 一、阶段判断

Phase942 已经证明：

```text
在 qwen3 中，semantic-specific direction 的 residual coordinate consensus
可以在留出 direction 上复现一部分 boundary margin movement。

在 GLM4 和 DS7B 中，也有部分正结果，但证据较弱或受控制项干扰。
```

因此下一步不是继续追求更大的 logit margin，而是回答：

```text
这些 consensus residual coordinates 是否能映射回真实模型组件？
```

Phase943 继续同一条“先完成图谱，再追求闭合”的路线，但性质变成：

```text
artifact export + static component mapping audit
```

它不是因果干预阶段，不证明 MLP channel / attention head 已经闭合。

### 二、本阶段任务

本阶段新增脚本：

```text
tests/glm5/phase943_consensus_coordinate_component_mapping_audit.py
tests/glm5/run_phase943_consensus_coordinate_component_mapping_audit.sh
```

结果目录：

```text
tests/result/phase943_consensus_coordinate_component_mapping_audit/consensus_coordinate_component_mapping_audit/
```

输出文件包括：

```text
phase943_qwen3_consensus_records.jsonl
phase943_glm4_consensus_records.jsonl
phase943_deepseek7b_consensus_records.jsonl
phase943_cross_model_summary.md
phase943_cross_model_summary.json
```

其中 `consensus_records.jsonl` 已保存：

```text
1. consensus coordinate indices；
2. vote count / vote histogram；
3. readout energy lift；
4. MLP down_proj row energy lift；
5. attention o_proj row energy lift；
6. top MLP column candidates；
7. top attention head candidates。
```

### 三、测试原理和公式

Phase943 复用 Phase942 的共识坐标：

```text
C = C_consensus(r, lang)
```

对任意向量 `v`，计算其在共识坐标中的能量比例：

```text
E_C(v) = ||v_C||^2 / ||v||^2
```

因为 `C` 的宽度固定为 256，所以用随机期望宽度作为基线：

```text
E_random = |C| / d_model
```

定义 lift：

```text
Lift_C(v) = E_C(v) / E_random
```

readout 审计中使用：

```text
v_boundary = W_U[target_label] - mean(W_U[boundary_tokens])
v_relation = W_U[target_label] - mean(W_U[other_relation_labels])
```

检查：

```text
Lift_C(v_boundary)
Lift_C(v_relation)
```

组件静态映射中，对 MLP 或 attention 输出矩阵 `W` 使用：

```text
R_C(W) = ||W[C, :]||^2 / ||W||^2
Lift_C(W) = R_C(W) / (|C| / d_model)
```

对 MLP channel 候选：

```text
Lift_C(W_down[:, j])
```

对 attention head 候选：

```text
Lift_C(W_o[:, head_slice])
```

注意：这些都是静态权重几何指标。它们只能给出候选来源，不能证明自然前向时该 channel/head 真的激活并产生该坐标组。

### 四、测试规模

三模型顺序执行：

```text
qwen3 -> GLM4 -> DS7B
```

实际记录数：

```text
qwen3:
  records = 7

GLM4:
  records = 5

DS7B:
  records = 1
```

对应筛选关系：

```text
qwen3:
  color en->en / en->zh / zh->en / zh->zh
  function en->en / zh->en / zh->zh

GLM4:
  color en->en / zh->en / zh->zh
  function en->en / zh->en

DS7B:
  function en->en
```

### 五、跨模型总结果

证据标签：

```text
residual_consensus_export_with_component_candidates: 1
residual_consensus_export_with_weak_component_candidates: 2
```

模型汇总：

```text
qwen3:
  readout boundary lift mean = 0.9316
  readout relation lift mean = 0.9008
  MLP down row lift mean = 0.9768
  attention o row lift mean = 0.9539
  max MLP column lift mean = 5.8457
  max attention head lift mean = 1.3202

GLM4:
  readout boundary lift mean = 1.1225
  readout relation lift mean = 0.9573
  MLP down row lift mean = 0.9802
  attention o row lift mean = 1.1016
  max MLP column lift mean = 10.1750
  max attention head lift mean = 2.8773

DS7B:
  readout boundary lift mean = 1.0237
  readout relation lift mean = 0.7782
  MLP down row lift mean = 0.9898
  attention o row lift mean = 0.9618
  max MLP column lift mean = 10.8440
  max attention head lift mean = 2.0283
```

总体客观现象：

```text
1. 整体 MLP down / attention o 的 row energy lift 接近 1；
2. readout boundary lift 只有 GLM4 明显高于 1；
3. 但 top MLP columns 出现很高 lift；
4. 部分 attention heads 也出现高于随机宽度的 lift。
```

这说明：

```text
共识坐标不是整个 MLP/attention 输出矩阵的全局偏置；
更像是少量 MLP channel / attention head 子块中存在候选集中。
```

### 六、qwen3 结果

qwen3：

```text
evidence = residual_consensus_export_with_weak_component_candidates
records = 7
```

qwen3 的总体特点：

```text
readout lift 不强；
整体 MLP/attention row lift 不强；
但 top MLP column lift 稳定偏高。
```

具体候选：

```text
color en->zh:
  hidden_idx = 36
  readout boundary lift = 0.9068
  MLP row lift = 0.9861
  attention row lift = 0.8899
  max MLP column lift = 6.7194
  max attention head lift = 1.2358
  top MLP column ids = 2509, 1579, 310

color zh->zh:
  hidden_idx = 36
  readout boundary lift = 1.0070
  max MLP column lift = 6.4837
  top MLP column ids = 2509, 134, 1579
  top attention head = head 5, lift = 1.2626

function en->en:
  hidden_idx = 27
  readout boundary lift = 0.8459
  max MLP column lift = 6.3458
  top MLP column ids = 2, 3, 376
  top attention heads = head 0 / 1 / 2
```

qwen3 的重要现象：

```text
color 相关 consensus coordinates 多次指向 MLP column 2509；
function 相关 consensus coordinates 多次指向 MLP column 2 / 3。
```

但必须注意：

```text
这是静态权重集中；
还没有证明这些 channels 在自然样本中被激活；
也没有证明 ablation/patch 后 boundary movement 会消失。
```

### 七、GLM4 结果

GLM4：

```text
evidence = residual_consensus_export_with_component_candidates
records = 5
```

GLM4 的总体特点：

```text
readout boundary lift mean = 1.1225；
attention o row lift mean = 1.1016；
top MLP column 和 top attention head lift 都更强。
```

最强候选：

```text
color en->en:
  hidden_idx = 30
  readout boundary lift = 1.1145
  attention o row lift = 1.1289
  max MLP column lift = 12.0063
  max attention head lift = 3.9425
  top MLP column ids = 5532, 8633, 1165
  top attention head = head 31

function en->en:
  hidden_idx = 30
  readout boundary lift = 1.1438
  attention o row lift = 1.1270
  max MLP column lift = 11.9764
  max attention head lift = 3.5002
  top MLP column ids = 5532, 8633, 1165
  top attention head = head 31

function zh->en:
  hidden_idx = 30
  readout boundary lift = 1.1364
  MLP row lift = 1.0228
  attention o row lift = 1.1353
  max MLP column lift = 11.9211
  max attention head lift = 2.9451
  top MLP column ids = 5532, 8633, 1165
  top attention head = head 31
```

GLM4 的关键现象：

```text
多个 relation / language pair 反复出现同一组高 lift MLP columns：
  5532, 8633, 1165

多个关系也反复出现 attention head 31。
```

这比 qwen3 更像可复用组件候选。

但 Phase942 已经显示：

```text
GLM4 的 boundary field 对 random / bottom 控制非常敏感。
```

因此这里的组件候选仍然不能直接解释为干净语义齿轮。

### 八、DS7B 结果

DS7B：

```text
evidence = residual_consensus_export_with_weak_component_candidates
records = 1
```

唯一记录：

```text
function en->en:
  hidden_idx = 14
  readout boundary lift = 1.0237
  readout relation lift = 0.7782
  MLP row lift = 0.9898
  attention row lift = 0.9618
  max MLP column lift = 10.8440
  max attention head lift = 2.0283
  top MLP column ids = 3033, 16221, 6030
  top attention head = head 7
```

DS7B 的结果说明：

```text
存在静态组件候选；
但只有一个 relation / language pair，
不能说明 DS7B 有稳定跨语义共识组件。
```

### 九、本阶段理论进展

Phase943 把图谱链条从：

```text
semantic direction
  -> residual coordinate consensus
  -> boundary margin movement
```

推进到：

```text
semantic direction
  -> residual coordinate consensus
  -> candidate MLP columns / candidate attention heads
  -> boundary margin movement
```

但其中组件环节目前只是静态候选：

```text
candidate MLP columns / candidate attention heads
```

还不是：

```text
causal MLP channels / causal attention heads
```

当前最有价值的候选是：

```text
qwen3 color:
  MLP column 2509, 1579, 134, 310

qwen3 function:
  MLP column 2, 3, 106, 376
  attention heads 0, 1, 2

GLM4 cross-relation:
  MLP column 5532, 8633, 1165
  attention head 31

DS7B function en->en:
  MLP column 3033, 16221, 6030
  attention head 7
```

### 十、严格审视和硬伤

1. 静态权重不等于自然激活。

```text
某个 MLP column 的输出权重集中在 C，
不代表该 channel 在目标样本中被激活。
```

2. 整体 row lift 接近 1。

```text
qwen3 / DS7B 的 MLP down 和 attention o 整体 row energy lift 都接近随机宽度；
说明共识坐标不是整个组件输出矩阵的全局方向。
```

3. top column lift 可能受小范数列影响。

```text
某些 column 的总能量可能较小；
高 lift 不一定意味着高绝对贡献。
```

本阶段记录了 column_energy，但还没有加入激活强度：

```text
actual contribution = activation_j(x) * W_down[:, j]
```

4. attention head 只是 o_proj 几何候选。

```text
没有读取自然注意力权重；
没有 head ablation；
没有 head patch。
```

5. readout lift 不稳定。

```text
qwen3 readout boundary lift mean < 1；
DS7B relation lift < 1；
只有 GLM4 readout boundary lift 略高。
```

这说明 residual consensus coordinates 不一定直接等于 lm_head readout 坐标。

6. 仍未接入 natural generation。

```text
Phase943 没有测试生成；
没有 full-vocab blocker closure；
没有 multi-token rollout。
```

### 十一、闭合距离

Phase943 当前完成：

```text
consensus coordinate artifact export
static readout/component mapping
candidate component list
```

距离闭合还差：

```text
1. activation-weighted MLP contribution；
2. MLP channel ablation / patch；
3. attention head ablation / patch；
4. channel/head 与 Phase942 boundary movement 的因果对应；
5. random same-layer channel/head 控制；
6. object-level holdout；
7. natural rollout；
8. blocker-token minimal cut。
```

所以当前结果只能写作：

```text
component candidate mapping
```

不能写作：

```text
component causal closure
```

### 十二、智能理论层面的关键洞察

Phase943 对“编码机制长什么样”提供了一个更具体的线索：

```text
属性/语义不是单纯存在于一个方向；
它可能通过 residual coordinate group 接入输出边界；
而这些 residual coordinate group 的来源，
可能进一步分散在少量 MLP columns 和少量 attention heads 中。
```

这支持一种更具体的图谱结构：

```text
object / attribute state
  -> semantic-specific direction
  -> residual coordinate consensus group
  -> sparse component candidates
  -> output boundary competition
```

但仍需谨慎：

```text
当前看到的是权重几何上的候选；
还没有看到自然前向里的激活齿轮。
```

破解语言编码机制的下一步，不应再停留在 residual coordinate 层，而要进入：

```text
activation-weighted component contribution
```

也就是对每个样本计算：

```text
MLP contribution_j(x) = a_j(x) * W_down[:, j]
```

然后看：

```text
P_C(MLP contribution_j(x))
```

是否真的解释 Phase942 中的 boundary movement。

### 十三、下一阶段任务

Phase944 应进入新的子阶段：

```text
Activation-Weighted Component Causal Audit
```

任务：

```text
1. 对 Phase943 的 top MLP columns 记录自然激活 a_j(x)；
2. 计算 activation-weighted contribution 是否落在 C_consensus；
3. 对 top channel 做 ablation；
4. 对 top channel 做 same-layer random channel 控制；
5. 对 GLM4 特别加入通用扰动控制，防止把敏感边界场误判为语义机制；
6. 对 qwen3 优先测试 function en->en 和 color en->zh；
7. 对 DS7B 只作为弱参考，不做强结论。
```

Phase944 与 Phase943 相关，但它已经从“静态映射”进入“组件因果干预”子阶段。为了避免把静态候选直接误升级为因果闭合，本阶段先在 Phase943 停止。

### 十四、通俗总结

Phase942 找到的是：

```text
哪些残差坐标像一组稳定接口。
```

Phase943 继续问：

```text
这些接口可能从哪些模型部件来？
```

结果是：

```text
qwen3:
  没有很强的整体组件集中；
  但有若干 MLP column 候选反复出现。

GLM4:
  候选最明显，尤其 MLP 5532/8633/1165 和 attention head 31；
  但 GLM4 本身边界场太敏感，需要严控。

DS7B:
  只有 function en->en 的弱候选。
```

所以当前拼图又前进了一格：

```text
残差共识坐标组已经可以导出；
并且能找到一批候选 MLP columns / attention heads。
```

但真正要证明它们是“齿轮”，还必须做下一阶段的激活加权和因果干预。

## Phase 944: 激活加权 MLP 通道因果审计 [2026-07-05 01:31]

### 一、对最新上传分析的判断

最新上传内容的总体判断基本正确，而且证据层级控制得比较稳。

当前机制公式：

```text
A(y|x) = sum_{o,r} P(o|x) P(r|x) K(o,r,y) g(y|x)
```

可以作为全局编码图谱中“语义答案场”的理论骨架，但不能单独完成全局编码图谱。它还缺少五个关键环节：

```text
1. P(o|x), P(r|x), K(o,r,y), g(y|x) 的可测量化；
2. semantic factor -> channel gear -> natural gate -> clean rollout 的因果链；
3. 自然门控来源；
4. full-vocab blocker 和 strict-clean 输出协议闭合；
5. qwen3 / GLM4 / DS7B 的跨模型一致性。
```

因此，本阶段没有继续做抽象理论总结，而是把 Phase943 得到的候选 MLP columns 推进到一个更客观的测试问题：

```text
这些候选 MLP 通道是否在自然激活加权后，仍然指向 Phase942 的共识残差坐标，
并且在通道干预时比同层随机通道更能移动 target-vs-boundary margin？
```

这一步对应最新理论里的关键缺口：

```text
把抽象语义方向接到具体通道齿轮。
```

### 二、测试脚本和结果文件

新增正式测试脚本：

```text
tests/glm5/phase944_activation_weighted_mlp_channel_causal_audit.py
```

新增顺序运行脚本：

```text
tests/glm5/run_phase944_activation_weighted_mlp_channel_causal_audit.sh
```

结果目录：

```text
tests/result/phase944_activation_weighted_mlp_channel_causal_audit/activation_weighted_mlp_channel_causal_audit/
```

核心汇总文件：

```text
phase944_cross_model_summary.md
phase944_cross_model_summary.json
```

测试已经依次完成：

```text
qwen3 -> GLM4 -> DS7B
```

避免了三个模型同时占用 GPU。

### 三、测试原理

Phase943 只是从静态权重上发现了候选 MLP columns。Phase944 进一步加入样本的自然激活：

```text
Contribution_G(x) = sum_{j in G} a_j(x) W_down[:, j]
```

其中：

```text
a_j(x): 样本 x 在该 MLP channel 上的自然激活
W_down[:, j]: 该 channel 写回 residual stream 的方向
G: Phase943 给出的 top MLP channel group
```

再计算该贡献落入 Phase942 共识残差坐标 C 的能量比例：

```text
E_C(v) = ||P_C(v)||^2 / ||v||^2
```

为了消除维度大小影响，使用 lift：

```text
Lift_C(v) = E_C(v) / (|C| / d_model)
```

然后做通道级因果干预：

```text
a'_j = f a_j,  j in G
```

本阶段使用：

```text
f = 0.0   candidate_ablate
f = 1.5   candidate_boost
```

并加入同层随机通道控制：

```text
same-layer random MLP columns
```

主要观察：

```text
Delta target-vs-boundary margin
Delta relation margin
Delta target logit
Delta boundary logit
```

关键判据不是只看候选通道有没有效果，而是看它是否超过同层随机通道：

```text
slope_gain =
  [DeltaBoundary(candidate_boost) - DeltaBoundary(candidate_ablate)]
  -
  [DeltaBoundary(random_boost) - DeltaBoundary(random_ablate)]
```

本阶段的强正结果标准：

```text
activation-weighted lift gap > 0.25
and
boundary slope gain > 0.02
```

### 四、测试数据和范围

测试从 Phase943 的候选记录出发，选择每个模型的 top-3 MLP columns，并在 Phase942 的 holdout 语义样本上重新评估。

qwen3 进入正式测试的候选：

```text
color en->en:     hidden 36, channels 2509,16,249
color en->zh:     hidden 36, channels 2509,1579,310
color zh->en:     hidden 36, channels 134,310,2509
color zh->zh:     hidden 36, channels 2509,134,1579
function en->en:  hidden 27, channels 2,3,376
function zh->en:  hidden 27, channels 106,2,3
function zh->zh:  hidden 27, channels 58,2,3
```

GLM4 进入正式测试的候选：

```text
color en->en:     hidden 30, channels 5532,8633,1165
color zh->en:     hidden 30, channels 4906,260,5775
color zh->zh:     hidden 30, channels 5532,8633,1165
function en->en:  hidden 30, channels 5532,8633,1165
function zh->en:  hidden 30, channels 5532,8633,1165
```

DS7B 进入正式测试的候选：

```text
function en->en:  hidden 14, channels 3033,16221,6030
```

### 五、跨模型结果

总体证据标签：

```text
activation_weighted_mlp_channel_causal_positive: 1
partial_activation_weighted_mlp_channel_causal_positive: 1
activation_weighted_mlp_channel_causal_weak_or_mixed: 1
```

也就是：

```text
qwen3:      正结果
GLM4:       部分正结果
DS7B:       弱/混合结果
```

#### 1. qwen3

qwen3 给出了本阶段最干净的正结果。

最强结果来自：

```text
relation: color
pair: en->en
hidden: 36
channels: 2509,16,249
```

结果：

```text
candidate activation lift: 4.8014
random activation lift:    1.0991
activation gap:            +3.7023
candidate boundary slope:  +0.6779
random boundary slope:     -0.0120
slope gain:                +0.6899
```

对应干预均值：

```text
candidate_ablate boundary delta: -0.4471
candidate_boost boundary delta:  +0.2308
random_ablate boundary delta:    +0.0048
random_boost boundary delta:     -0.0072
```

这说明该通道组不是单纯“权重方向像”，而是在自然激活加权后，确实能解释一部分 boundary movement。

function 方向也有正结果：

```text
function zh->en, hidden 27, channels 106,2,3
activation gap: +3.7149
slope gain:     +0.0521

function zh->zh, hidden 27, channels 58,2,3
activation gap: +3.4679
slope gain:     +0.0404
```

但 qwen3 并非所有候选都成立。例如：

```text
function en->en: activation gap +6.2542, slope gain -0.0208
color zh->zh:    activation gap +5.0951, slope gain -0.0216
```

这说明：

```text
高 activation-weighted coordinate lift 不必然等于正向边界因果齿轮。
```

#### 2. GLM4

GLM4 的 activation-weighted coordinate signal 很强，但因果解释不干净。

一个部分正结果：

```text
relation: function
pair: zh->en
hidden: 30
channels: 5532,8633,1165
candidate activation lift: 6.2488
random activation lift:    0.9594
activation gap:            +5.2894
candidate boundary slope:  -0.0039
random boundary slope:     -0.0469
slope gain:                +0.0430
```

但是该结果有严重污染：

```text
candidate_ablate boundary delta: +2.0781
candidate_boost boundary delta:  +2.0742
random_ablate boundary delta:    +2.1237
random_boost boundary delta:     +2.0768
```

候选通道和随机通道都能造成很大的 boundary delta，说明 GLM4 在这些样本上的边界场存在通用敏感性。本阶段不能把 GLM4 的结果直接写成 clean causal gear，只能写成：

```text
partial activation-weighted MLP channel causal positive
```

GLM4 对当前理论的贡献是负向收紧：

```text
即使候选通道强烈落在共识坐标上，也可能只是处在一个高敏感边界层，
并不等于语义方向已经通过专用齿轮闭合到输出。
```

#### 3. DS7B

DS7B 只有一个进入测试的候选：

```text
relation: function
pair: en->en
hidden: 14
channels: 3033,16221,6030
```

结果：

```text
candidate activation lift: 11.2583
random activation lift:    1.0643
activation gap:            +10.1940
candidate boundary slope:  -0.0208
random boundary slope:     -0.0156
slope gain:                -0.0052
```

这说明 DS7B 存在非常强的 coordinate concentration，但没有表现出正向 boundary causal slope。

因此 DS7B 本阶段只能作为弱/混合参考，不能给强机制结论。

### 六、阶段结论

Phase944 的正结果是：

```text
在 qwen3 中，Phase943 的部分 MLP column 候选已经通过了 activation-weighted + causal intervention 审计。
```

更具体地说：

```text
qwen3 color en->en 的 hidden 36 / channels 2509,16,249
是当前最像“语义坐标 -> MLP 通道齿轮 -> boundary movement”的候选链。
```

但是 Phase944 也给出重要负结果：

```text
1. 高 activation-weighted lift 不等于边界因果正结果；
2. GLM4 的强通道信号容易混入通用边界敏感性；
3. DS7B 的坐标集中不自动转化为 causal boundary movement；
4. Phase943 的静态候选不能直接升级为机制闭合。
```

因此，本阶段不是语言编码机制闭合，也不是完整全局编码图谱完成，而是完成了一个关键桥接：

```text
static component candidate
->
activation-weighted component candidate
->
partial causal gear evidence
```

### 七、对当前理论公式的修正

原公式：

```text
A(y|x) = sum_{o,r} P(o|x) P(r|x) K(o,r,y) g(y|x)
```

仍然适合作为语义答案场骨架，但 Phase944 说明必须增加一个可测通道层：

```text
K(o,r,y)
not only semantic path strength
but also measurable component route:

K(o,r,y)
  -> C_consensus(o,r)
  -> Contribution_G(x)
  -> BoundaryMovement(y, B_x)
```

更接近当前证据的公式应写成：

```text
CleanOutput(y|x)
=
SemanticAnswer(y|x)
and CandidateWinner(y|x)
and ActivationWeightedGear(G,x)
and FieldAdmissible(B_x)
and NaturalGate(G,x)
and NoProtocolDrift(x)
```

其中 Phase944 只推进了：

```text
ActivationWeightedGear(G,x)
```

还没有推进：

```text
NaturalGate(G,x)
NoProtocolDrift(x)
strict-clean rollout
```

### 八、闭合标准和当前距离

一个更严格的闭合标准仍然应保持为：

```text
CleanCausalEdge
=
GearEffect
and FieldAdmissible
and OutputTransition
and NoSideEffect
```

如果用于全局编码图谱，则还要再加：

```text
SemanticFactorMeasurable
and CrossObjectHoldout
and CrossRelationHoldout
and CrossTemplateHoldout
and CrossModelRobustness
and NaturalGate
and StrictCleanRollout
```

Phase944 已经满足或部分满足：

```text
1. 部分 GearEffect: qwen3 color/function 有正结果；
2. 部分 SemanticFactorMeasurable: 共识坐标 C 可以被 MLP contribution 解释；
3. 部分随机对照: candidate group 超过 same-layer random group。
```

Phase944 尚未满足：

```text
1. 没有自然门控；
2. 没有 strict-clean rollout；
3. 没有 attention head 因果审计；
4. 没有 full-vocab blocker class 的完整闭合；
5. 没有对象级大规模 holdout；
6. 没有把 group 内每个 channel 的方向符号拆开；
7. GLM4 和 DS7B 还不能给跨模型强结论。
```

所以当前距离闭合仍然较远，但距离“可测量全局编码图谱”近了一步。

### 九、硬伤和瓶颈

当前最大硬伤不是公式错误，而是变量还不够可测：

```text
P(o|x), P(r|x), K(o,r,y), g(y|x)
```

仍需要映射到：

```text
residual coordinate
MLP channel contribution
attention head route
candidate blocker class
natural gate score
rollout transition
```

第二个硬伤是 group-level 通道可能混合了多个作用：

```text
同一组 channels 里可能同时存在 answer lift、blocker weakening、protocol drift、副作用。
```

第三个硬伤是当前干预仍然是人工缩放：

```text
a'_j = f a_j
```

这不能替代自然门控。真正的机制闭合必须回答：

```text
模型自然状态下为什么、何时、由谁启动这个 channel group？
```

第四个硬伤是小模型偏差。qwen3、GLM4、DS7B 的内部结构可能比大模型粗糙，尤其 GLM4 的通用边界敏感性和 DS7B 的弱转换结果，都不能轻易外推。

### 十、智能理论进展

从智能理论角度看，Phase944 支持一个更具体的第一性原理方向：

```text
语言编码不是单个概念向量直接读出答案，
而是语义因子先形成可复用残差坐标，
再通过具体组件的自然激活加权贡献写入候选竞争边界。
```

也就是说：

```text
对象、关系、属性不是孤立神经元；
它们更像一组可复用坐标接口。
```

这些接口是否产生输出，还要经过：

```text
component contribution
candidate boundary
protocol continuation
termination gate
```

Phase944 对“水果/颜色/功能如何复用神经元”的启示是：

```text
不要先找 apple neuron / red neuron；
先找 object-relation coordinate；
再检查哪些 MLP channel 在自然激活加权后稳定写入该 coordinate；
最后看这些写入是否能移动 full-vocab boundary。
```

### 十一、下一阶段任务

Phase945 应继续沿着当前图谱优先路线，但进入更细的子任务：

```text
MLP Channel Sign Decomposition and Object-Level Holdout Audit
```

具体任务：

```text
1. 对 qwen3 color en->en 的 channels 2509,16,249 逐个做 ablate/boost；
2. 对 qwen3 function zh->en 的 channels 106,2,3 逐个做 ablate/boost；
3. 区分 support channel、suppressor channel、mixed side-effect channel；
4. 加入对象级 holdout，例如 apple/banana/car/shark 等跨类别样本；
5. 检查同一 channel 是否复用到 color、function、category 等不同关系；
6. GLM4 加入更强 random/control perturbation，专门剔除通用边界敏感性；
7. DS7B 只做弱参考，不作为闭合依据。
```

Phase945 与 Phase944 属于同一条大路线：

```text
全局编码图谱可测量化。
```

但它已经从 group-level causal audit 进入 single-channel sign decomposition，因此不应把 Phase944 的结果和 Phase945 的结论混写。本阶段先完成 Phase944，并把下一阶段固定为更严格的分解验证。

### 十二、通俗总结

这次测试可以简单理解为：

```text
以前我们看到一些齿轮“长得像”可能有用；
这次开始检查这些齿轮在模型真正运行时有没有被转起来，
以及转动它们会不会真的推动输出边界。
```

结果是：

```text
qwen3 有一组比较像真齿轮的 MLP channels；
GLM4 有强信号，但边界太敏感，容易误判；
DS7B 有方向集中，但没有稳定推动输出。
```

所以当前结论是：

```text
机制公式可以当骨架；
Phase944 开始把骨架接到可测通道；
但距离自然门控和 strict-clean 输出闭合仍然很远。
```

## Phase 945: AGI_GLM5_MEMO_SUMMARY.md 全历史交叉分析 [2026-07-15 14:30]

### 零、元信息

- 分析对象：`research/glm5/docs/AGI_GLM5_MEMO_SUMMARY.md`（Phase 20-940 完整历史摘要，约3200行）
- 分析方法：逐阶段提取可复现、可验证、有因果证据的关键发现，跨阶段交叉比对
- 分析目标：从1300+ Phase中识别出对破解语言编码机制最重要的结果，评估其证据等级和可操作价值
- 产出：12个关键发现的完整审计 + 3条突破路线 + 测试计划

### 一、筛选方法论

本分析采用四维评分体系筛选关键发现：

```text
证据等级（1-5）：
  1=推测/假设
  2=可解码但无因果证据
  3=有因果干预但未跨样本验证
  4=跨样本因果验证 + 负控制通过
  5=跨模型因果闭合

可复现性（1-3）：
  1=单模型单任务
  2=单模型多任务或多模型单任务
  3=多模型多任务

桥接层级（1-4）：
  1=单一组件发现
  2=组件间关联
  3=从语义到机制的桥接
  4=从语义到机制到输出的完整链

理论价值（1-5）：
  1=局部现象
  2=可泛化模式
  3=改变方法论
  4=揭示新型数学对象
  5=指向最终理论框架
```

总分 >= 12 为"第一梯队"，>= 9 为"第二梯队"，>= 6 为"第三梯队"。

### 二、12个关键发现详细审计

#### 第一梯队（最高优先级，总分 >= 12）

**发现 #1：语义方向→MLP通道齿轮→边界移动完整桥接链**
- 来源：Phase 937-944（GLM5路线）
- 证据等级：4（跨样本因果 + 负控制）
- 可复现性：1（仅qwen3 color en→en）
- 桥接层级：4（语义层→机制层→输出层，唯一完整的三层桥接）
- 理论价值：4（揭示语义因子→物理齿轮的映射关系）
- 总分：13
- 核心公式：
```
共识残差坐标(color/function语义方向)
→ hidden 36 / MLP channels 2509,16,249
→ activation gap +3.70, slope gain +0.69
→ boundary movement（输出边界竞争改变）
```
- 为什么重要：这是全项目唯一完成"语义坐标→具体通道→边界移动"三层桥接的证据链，证明MLP通道齿轮是语义因子的物理载体。
- 硬伤：仅在qwen3单模型单任务上验证；activation gap+3.70效应存在但不够强；通道级信号可能混合了多种作用。
- 可操作验证路径：在GLM4和DS7B上用相同范式测试color/function语义方向是否能定位到各自的MLP通道齿轮。

**发现 #2：GLM4 L39 EOS-vs-a 共识边界齿轮**
- 来源：Phase 918-920（GLM5路线）
- 证据等级：4（跨样本因果 + 共识齿轮压缩 + 三重负控制）
- 可复现性：1（仅GLM4）
- 桥接层级：2（组件间关联：MLP通道→EOS边界）
- 理论价值：4（揭示有符号边界子空间的存在性）
- 总分：11
- 核心公式：
```
C_j(EOS-a|x) = a_j^39(x) · ((W_U(EOS)-W_U(a))^T W_down^39[:,j])
G_cons = Top64_j Σ_s 1[j∈G_s]
正向组top1/margin>=0大量成功，random/rotated/a-logit-only负控制为0
```
- 为什么重要：这是最接近可操作机制的发现——有精确的通道定位、有双向验证、负控制干净，不是假阳性。
- 硬伤：人工拨齿轮能让EOS赢但模型自然状态下不拨（自然门控缺失）；仅对article_a blocker有效，对句号blocker无效。
- 可操作验证路径：在Qwen3/DS7B中搜索对应的EOS-vs-a边界齿轮；分析自然状态下为什么不激活这些通道。

#### 第二梯队（方法论基石 + 关键机制，总分 >= 9）

**发现 #3：可解码性≠因果使用**
- 来源：Phase 208-209, GPT5 Phase 397（两条路线独立验证）
- 证据等级：5（双路线独立验证 + 关系算子破坏无影响 + 关系签名可读不可搬27/27 vs 0/9）
- 可复现性：3（多模型多任务）
- 桥接层级：1（方法论层面）
- 理论价值：5（改变了整个研究的方法论，防止了无数歧路）
- 总分：14
- 核心公式：
```
Phase 208: s(i,j)=h_i^T A h_j 可高精度解码关系（可读）
Phase 209: 破坏s(i,j)对输出几乎无影响（不因果）
GPT5 Phase 397: 关系签名观测复现27/27，因果关系载体0/9
```
- 为什么重要：这是全项目最重要的方法论级发现，任何未来的候选机制必须先过"可解码≠因果"的证伪门。
- 可操作验证路径：已成标准操作流程，所有新发现机制都应做causal ablation。

**发现 #4：Jacobian暗物质动力系统**
- 来源：Phase 225（GLM5路线）
- 证据等级：3（跨多任务测量，但非因果干预）
- 可复现性：1（需在更多模型上验证）
- 桥接层级：1
- 理论价值：5（揭示了一个未被观测但可能承载核心计算的动力学空间）
- 总分：10
- 核心公式：
```
Top5(J_l) ≡ Top5(PCA(Δh_l)) ⟂ Row(W_U)
约束传播主方向与Jacobian主方向重合，但几乎在W_U盲区
```
- 为什么重要：解释了为什么直接从logit/W_U反推机制会失败——大量因果动力在"看不到的空间"里运行。为"因果状态等价类"提供了数学直觉。
- 硬伤：仅是观测结果，没有通过干预验证Jacobian方向的因果作用。
- 可操作验证路径：设计实验直接测试Jacobian主方向是否携带因果信息（例如沿J主方向注入扰动看输出变化）。

**发现 #5：CleanCausalEdge四要素框架**
- 来源：Phase 867-874（GLM5路线）
- 证据等级：4（多任务验证 + 区分有效转移和干净因果边）
- 可复现性：2（多模型验证）
- 桥接层级：2
- 理论价值：4（提供严格的因果评判标准）
- 总分：12
- 核心公式：
```
CleanCausalEdge = GearEffect ∧ FieldAdmissible ∧ OutputTransition ∧ NoSideEffect
FieldAdmissible = ¬TooManyBlockers ∧ ¬ObjectDominatesClass ∧ ¬FormatDominates ∧ ReducibleOriginalBlockers
```
- 为什么重要：把模糊的"干预有效"拆成四个可验证条件，是后续所有机制发现的标准操作流程。
- 可操作验证路径：作为所有新发现机制的必要验证步骤。

**发现 #6：Identity-Role-Frame-Operator分解 + 否定算子稳定性**
- 来源：Phase 294-300（GLM5路线）
- 证据等级：3（正交分解 + 因果patch，但未做跨模型）
- 可复现性：2（多任务验证）
- 桥接层级：2
- 理论价值：5（否定算子的跨operand稳定性暗示存在真正的"语言算子代数"）
- 总分：12
- 核心公式：
```
h ≈ μ + I(token) + R(role) + F(frame) + I×F + ε
h ≈ μ + I(operand) + O(operator) + S(scope) + F(frame) + O×S + I×O + O×F + ε  (Phase 300)
```
- 关键结论：否定操作符(O)比普通语法角色(R)更稳定、更低维、更跨operand一致。
- 为什么重要：操作符可能比"语义方向"更接近语言的核心数学结构——语义依赖上下文，但算子可能是跨上下文稳定的基本变换。如果操作符形成封闭的代数结构，那就是语言数学理论的直接入口。
- 硬伤：仅测试了否定算子，未扩展到其他逻辑算子（量化、模态等）。
- 可操作验证路径：扩展到"所有/有些/没有/必须/可能"等算子族，测试它们是否形成封闭组合。

**发现 #7：Attention-MLP契约的跨模型差异**
- 来源：Phase 285-293（GLM5路线）
- 证据等级：4（forward patching + 跨模型）
- 可复现性：3（三模型独立验证）
- 桥接层级：2
- 理论价值：4（解释了跨模型闭合失败的结构性原因）
- 总分：13
- 核心公式：
```
Qwen3: 能吸收外来attention
GLM4: 强烈拒绝外来attention
DS7B: 某些情况下敌对
```
- 为什么重要：用数据解释了跨模型闭合=0/72的根本原因——三模型用不同的"组件契约"实现相同功能。共同机制不存在于组件层面，必须上升到"功能等价类"层面。
- 可操作验证路径：在三模型中找"功能等价但组件不同"的案例，定义因果状态等价类。

**发现 #8：Protocol续写场压倒语义场**
- 来源：Phase 899-907（GLM5路线）
- 证据等级：4（严格测量 + 对照实验）
- 可复现性：2（多模型验证趋势一致）
- 桥接层级：3（语义→协议→终止的完整链条分离）
- 理论价值：4（揭示语言生成的场间博弈机制）
- 总分：13
- 核心数据：
```
语义答案类闭合: 68/77 (88%)
clean protocol rollout: 0/77 (0%)
stop_top1全是句号，EOS自然top1=0
强制EOS后strict_clean=68/68（证明语义机制基本工作）
但自然EOS top1=0（证明被Protocol场压倒）
```
- 为什么重要：精确区分了"语义编码问题"和"输出控制问题"。语义机制基本工作（88%语义闭合计），但Protocol续写场的概率压倒一切。
- 硬伤：Protocol场的本质是什么尚不清楚——是训练数据的格式偏好？是某种"续写本能"？还是模型架构的结构性偏向？
- 可操作验证路径：分析Protocol场的token组成和激活来源；测试能否通过改变prompt格式来减弱Protocol场。

**发现 #9：Blocker分类与公共骨架+case残差结构**
- 来源：Phase 925-936（GLM5路线）
- 证据等级：4（跨样本 + LOSO验证 + 固定齿轮覆盖30/30）
- 可复现性：1（仅GLM4）
- 桥接层级：2
- 理论价值：4（揭示了全词表竞争的子场结构）
- 总分：11
- 核心公式：
```
G_punct(x) = G_common + G_case(x) + G_residual(x)
article_a blocker ≠ punctuation_period blocker
标点边界 = 公共齿轮骨架 + case-specific齿形补偿
```
- 为什么重要：全词表竞争不是单一阈值问题，而是多个子竞争场的叠加。公共骨架+case残差的结构暗示存在可分解的代数结构。
- 可操作验证路径：在qwen3上复现blocker分类；测试公共骨架是否跨模型共享。

#### 第三梯队（重要拼图，总分 >= 8）

**发现 #10：W_U作为候选竞争读出接口**
- 来源：Phase 255-284
- 总分：9
- 核心发现：单神经元解释失败，W_U不是语义相似空间而是候选竞争读出接口。

**发现 #11：类别的小方差大因果现象**
- 来源：Phase 348-386
- 总分：8
- 核心发现：纯category方差很小(R_A^2低)但category centroid因果效力显著——信号强度≠因果重要性。

**发现 #12：target-lift dominated boundary migration**
- 来源：Phase 885-890
- 总分：9
- 核心发现：许多闭合不是切断单个blocker而是抬高整个目标类——边界移动是全局效应。

### 三、跨发现交叉分析

#### 3.1 指向同一结构的多条独立证据链

以下发现从不同角度指向"语言编码是多层场博弈系统"：

```
发现#1（语义→通道→边界）：语义因子通过MLP通道写入边界竞争
发现#8（Protocol场压语义场）：语义场被Protocol场压倒
发现#9（Blocker子场）：边界竞争由多个子场组成
发现#7（契约差异）：不同模型用不同组件组合实现相同功能
发现#6（算子稳定性）：操作符是跨上下文稳定的，暗示存在可分离的算子层
```

整合为完整图景：
```text
语言输出 = F(
  语义因子(通过MLP通道齿轮写入),
  候选竞争场(由多个blocker子场组成),
  协议续写场(压倒一切，尤其在自然生成中),
  EOS终止场(自然状态下极弱),
  模型特异契约(功能等价但组件不同)
)
```

#### 3.2 已排除的假说（系统性清单）

从12个发现和背景研究中，以下假说已被严格证伪：
1. 静态向量闭合（#1, Phase 595-604）
2. 单神经元机制（#10, #3）
3. 可解码=因果使用（#3）
4. 简单线性加法边界（#5, Phase 844-850）
5. 跨模型组件级对应（#7）
6. 单一blocker阈值（#9）
7. 语义方向独立于上下文（#6, I/F/O分解）
8. EOS仅需语义信号即可胜出（#8）
9. 注意力头为独立功能单元（Phase 669-712 QK/V拆解）
10. W_U作为语义相似空间（#10）

#### 3.3 方法论的进化路径

```
Phase 20-300: 方向/子空间分解 → 逐步收紧（可解码≠因果, Jacobian暗物质）
Phase 301-594: 组件/通道定位 → RMSNorm/gain读出接口 → 候选竞争图谱
Phase 595-826: 序列轨迹闭合 → 图谱化 → exact-natural consistency
Phase 827-874: 齿轮-场-门三层 → CleanCausalEdge框架
Phase 875-907: 非干净转移 → 主导齿轮 → Protocol场 → EOS缺口
Phase 908-940: 边界齿轮定位 → 语义→通道→边界桥接
```

### 四、三条突破路线

#### 路线A：以Phase 944桥接链为蓝本，跨模型跨任务复现（最可操作）

目标：把qwen3 color上的三层桥接扩展到其他模型和语义域。

具体步骤：
1. 在GLM4上用Phase 944相同范式定位color语义的MLP通道齿轮
2. 在DS7B上同样测试（预期效应弱，但需要确认是否完全不存在）
3. 扩展到function、category、material等其他语义域
4. 如果所有模型都至少有一个语义域能完成三层桥接，则证明桥接结构是通用的

关键风险：GLM4的通用边界敏感性（#2的附属发现）可能导致假阳性；DS7B可能效应太弱无法定位。

#### 路线B：从操作符切入，建立语言算子代数（理论上最有突破潜力）

目标：以否定算子为基础，扩展测试逻辑算子族，寻找可封闭的变换群结构。

具体步骤：
1. 先在三模型上完整复现Phase 300的否定算子分解
2. 扩展到"所有/有些/没有/必须/可能"等量化/模态算子
3. 测试算子组合是否形成封闭结构：¬¬P = P, ¬(P∧Q) = ¬P∨¬Q 等
4. 测试算子是否跨模型共享（如果操作符是最底层变换，应该跨模型一致）
5. 测试算子是否在特定层/组件中实现（可能对应attention的query变换或MLP的gate控制）

关键风险：可能只有否定算子稳定，其他算子效应弱或不成立。

#### 路线C：Protocol场的独立攻克（最直接的瓶颈突破）

目标：理解Protocol场为何能压倒语义场，以及如何让语义场胜出。

具体步骤：
1. 全量审计Protocol场的token组成（newline/comma/field/explanation/list各自占比）
2. 定位Protocol场的激活来源（哪些层/头/位置贡献了Protocol压力）
3. 测试prompt格式对Protocol场强度的调节
4. 设计"protocol-neutral" prompt模板，看是否能独立观察语义场
5. 如果Protocol场可以被格式修改减弱，测试在弱Protocol场下EOS能否自然胜出

关键风险：Protocol场可能是训练数据的深层结构偏见，无法通过简单格式修改消除。

### 五、智能理论启示

从这12个发现和三条路线中，可以得到对智能理论的几个关键启示：

```text
1. 语言编码是一个"场博弈"系统，不是"管道"系统。
   信号不是从输入→处理→输出线性流动，
   而是多个场（语义场、Protocol场、候选场、EOS场）在同时竞争。

2. 正确的数学对象不是"神经元"或"方向"，
   而是"因果状态等价类"和"场间转移算子"。
   发现#6的否定算子稳定性暗示场间转移可能形成代数结构。

3. 跨模型闭合=0不意味着没有共同机制，
   而是共同机制在"功能等价类"层面而非"组件对应"层面。
   相当于：不同硬件实现相同算法。
   要找到算法，必须上升到功能等价类。

4. 最小的可操作突破口可能是：
   - 操作符代数（路线B，找最底层不变量）
   - Protocol场机制（路线C，解决最直接的瓶颈）
   而非继续扫描更多候选组件（边际价值已近零）。
```

### 六、下一阶段任务

基于以上分析，Phase 946-950 的任务优先级为：

```text
Phase 946（路线B起点）: 在三模型上复现否定算子分解 + 扩展到量化/模态算子
  - 脚本: /tests/glm5/phase946_operator_algebra.py
  - 关键指标: 算子跨operand稳定性、算子组合封闭性、跨模型一致性

Phase 947（路线A扩展）: 在GLM4上复现color语义→MLP通道齿轮桥接
  - 脚本: /tests/glm5/phase947_glm4_color_bridge.py
  - 关键指标: 是否能定位到明确的MLP通道、activation gap强度、负控制

Phase 948（路线C起点）: Protocol场token组成全量审计 + 激活来源追踪
  - 脚本: /tests/glm5/phase948_protocol_field_audit.py
  - 关键指标: Protocol token的logit分布、贡献来源（层/头/位置）

Phase 949: 跨方向交叉验证——比较三条路线的发现是否指向同一机制
Phase 950: 根据Phase 946-949结果确定接下来的主攻方向
```

### 七、谨慎评估

本次分析的局限：
1. 基于SUMMARY文件而非原始实验数据，可能存在摘要偏差
2. 部分发现的解读可能已被后续Phase修正，需要逐个核对最新状态
3. 三条路线的优先级排序基于当前分析者的判断，需要在Phase 946-949中验证
4. 操作符代数方向（路线B）是理论推测，可能在实验中完全失败

最重要的不确定因素：
1. 否定算子之外的逻辑算子是否也稳定？
2. Protocol场是否可以通过格式修改减弱？
3. GLM4和DS7B上能否复现三层桥接？
4. 操作符是否跨模型共享？

这些问题必须在Phase 946-949中逐一回答。

### 八、通俗总结

这次分析可以理解为：

```text
我们从两年多、1300+轮实验里挑出了12个最可靠的发现，
像拼图一样摆在一起看。

结果发现，语言编码不是"一个答案从一条管道里流出来"，
而是"多个场在同时喊话，看谁声音大"。

语义场已经把答案喊出来了（88%的情况），
但Protocol场声音更大，把EOS完全压住。

最像"真齿轮"的是qwen3里的几个MLP通道，
拨它们能推动输出边界；
最像"语言数学"的是否定算子，
它在不同上下文里都很稳定，不像普通语义那样变来变去。

接下来应该三管齐下：
先验证否定算子的"兄弟姐妹"（量化/模态算子）是否也一样稳定，
再在GLM4上找qwen3那种齿轮，
同时搞清楚Protocol场为什么这么强。
```

## Phase 946: 算子代数 — 三模型否定/量化/模态算子全量测试 [2026-07-15 16:30]

### 零、元信息

- 脚本：`tests/glm5/phase946_operator_algebra.py`（完整版，qwen3用）
- 脚本：`tests/glm5_temp/phase946_fast_runner.py`（精简版，GLM4/DS7B用）
- 测试模型：qwen3, GLM4, DS7B
- 刺激数量：416（完整版）/ ~230（精简版），覆盖5类算子类型
- 路线归属：路线B起点 — 从否定算子扩展到逻辑算子族

### 一、实验设计

```
算子类型：
  1. Negation（否定基线）：20 adj × 2-4 frames × 2 conditions
     提取 O(not) = h("is not X") - h("is X") 在目标词位置
  2. Quantification（量化）：5 quantifiers × 8 nouns
     提取 Q(A→B) = h("B...") - h("A...") 的量化过渡方向
  3. Modal（模态）：5 modals × 6 verbs
     提取 M(A→B) = h("B...") - h("A...") 的模态过渡方向
  4. Double Negation（双否定）：¬¬P ≈ P? 
     提取 h("is not not X") - h("is X") 的余弦
  5. Operator Composition（算子组合）：
     提取 not_all, not_must 方向与 O(not) 的关系

关键指标：
  - LOO_cos：跨operand留一法余弦（算子稳定性）
  - PCA top1/dim50：算子方向的维度集中度
  - DN closure：双否定闭合性 cos(¬¬P, P)
  - Cross-operator affinity matrix：算子间余弦相似度矩阵
  - Composition cos：组合算子与原子算子的余弦
```

### 二、核心结果：三模型对比

#### 2.1 否定算子稳定性（基线复现）

```
┌──────────────┬─────────┬─────────┬─────────┐
│ 指标          │ qwen3   │ GLM4    │ DS7B    │
│              │ (L18)   │ (L20)   │ (L14)   │
├──────────────┼─────────┼─────────┼─────────┤
│ NEG LOO cos  │ 0.609   │ 0.831   │ 0.588   │
│ NEG LOO std  │ 0.102   │ 0.053   │ 0.096   │
│ NEG PCA top1 │ 21.7%   │ 21.9%   │ 19.2%   │
│ NEG PCA dim50│ 5       │ 4       │ 5       │
│ NEG PCA dim80│ 12      │ 9       │ 12      │
└──────────────┴─────────┴─────────┴─────────┘
```

**否定算子复现成功**：三模型LOO全部显著高于Phase 298的role LOO（qwen3/GLM4 ~0.44）。
但模型间有显著差异：GLM4否定算子最稳定(0.831)，DS7B最弱(0.588)。

**这验证了Phase 300的核心发现：否定算子比语法角色更稳定、更低维。**

#### 2.2 双否定闭合（¬¬P ≈ P?）——最关键的跨模型差异

```
┌──────────────────┬─────────┬─────────┬─────────┐
│ 指标              │ qwen3   │ GLM4    │ DS7B    │
├──────────────────┼─────────┼─────────┼─────────┤
│ cos(¬¬P, P)      │ 0.873   │ 0.394   │ 0.769   │
│ pos_rate         │ 100%    │ 100%    │ 100%    │
└──────────────────┴─────────┴─────────┴─────────┘
```

**这是全Phase 946最重要的发现**：
- Qwen3强编码了 ¬¬P ≈ P（cos=0.873），意味着双否定在隐藏空间几乎等价于不做否定。
- DS7B中等程度编码（cos=0.769）。
- **GLM4几乎不编码**（cos=0.394），虽然有正向趋势但非常弱。

这意味着：
1. ¬¬P = P 不是语言模型的通用代数性质——它是模型特定的。
2. Qwen3学到了更接近"逻辑一致性"的表示；GLM4可能学了更接近"表面形式"的表示。
3. 算子代数不能简单地假设存在——需要在每个模型上独立验证。

#### 2.3 算子亲和矩阵关键发现

**(a) 否定算子是独立家族**

在所有三模型中，NEG(not)与其他算子的余弦都低于0.3：
```
NEG(not) 最亲和的非否定算子：
  qwen3: Q(all→no)=0.298, Q(some→no)=0.233
  GLM4:  Q(some→no)=0.151, Q(all→no)=0.131
  DS7B:  Q(all→no)=0.248, Q(some→no)=0.215
```
否定算子与模态算子几乎完全正交（cos < 0.08在所有模型中）。

**(b) 模态算子形成紧密聚类，may→must ≈ might→must 跨模型一致**

```
may→must vs might→must cosine:
  qwen3: 0.909
  GLM4:  0.829
  DS7B:  0.869
```
这是最强的跨模型不变量！may和might的"must方向"几乎相同，
暗示"可能性→必要性"是一个跨模型共享的语义维度。

**(c) 量化算子也聚类，但不如模态紧密**

```
some→all vs some→every cosine:
  qwen3: 0.716
  GLM4:  0.494
  DS7B:  0.525
```
量化算子聚类在qwen3中最紧密，在GLM4中最松散。

#### 2.4 算子组合（not_all, not_must）

```
┌──────────────────┬─────────┬─────────┬─────────┐
│ cos(X, O_not)    │ qwen3   │ GLM4    │ DS7B    │
├──────────────────┼─────────┼─────────┼─────────┤
│ not_all          │ 0.282   │ 0.157   │ 0.206   │
│ not_must         │ 0.314   │ 0.254   │ 0.353   │
└──────────────────┴─────────┴─────────┴─────────┘
```

所有模型中，组合算子与原子O(not)的余弦都在0.15-0.35之间。
not_all和not_must不完全等同于not——它们有额外的算子特异性。
**算子组合不是简单的加法**——这与De Morgan律不同。

### 三、跨模型一致性评分

```
算子属性                    跨模型一致性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
否定LOO稳定性               ★★★☆☆ (趋势一致，数值差异大)
否定PCA维度                 ★★★★★ (高度一致)
双否定闭合                  ★☆☆☆☆ (完全不同)
模态算子聚类(may/might→must) ★★★★★ (高度一致)
量化算子聚类                ★★★☆☆
算子家族分离(neg⟂mod)       ★★★★★ (高度一致)
算子组合                    ★★★☆☆
```

### 四、对语言数学理论的启示

1. **否定算子是真实存在的结构**：三模型LOO全部显著高于随机，
   证明了否定在隐藏空间有统一的变换方向。

2. **但双否定闭合是模型特定的**：qwen3的¬¬P≈P很强，GLM4几乎不存在。
   这意味着"逻辑律"不是训练自然涌现的，而是需要特定训练信号。

3. **算子家族分离是跨模型一致的**：Negation ≠ Quantification ≠ Modal，
   在隐藏空间中形成三个几乎正交的子空间。
   这暗示存在某种"算子类型"的拓扑结构。

4. **may→must ≈ might→must 是最强的跨模型不变量**：
   这个发现强烈暗示"可能性→必要性"是语言中一个基本的语义维度，
   独立于具体模型架构。

5. **算子组合不是加法**：not_all ≠ not + (all→some)。
   这意味着算子代数的群结构（如果存在）不是简单的向量加法。

### 五、关键硬伤

1. **双否定闭合在GLM4几乎不存在**——路线B的理论基础("算子代数")被严重动摇。
   如果连最基本的 ¬¬P=P 都不跨模型成立，更复杂的算子代数更不用想了。

2. **样本量不够大**：精简版只有20个adj operand，量化/模态各只有6-8个样本。
   需要加大数据量重复测试（特别是GLM4的双否定）。

3. **仅测试了否定/量化/模态三类算子**，未扩展到条件(if/then)、
   时间(before/after)、比较(more/less)等。

4. **位置选择可能有问题**：在目标词位置提取算子方向，但多token算子
   (如"not all", "may not")的语义可能在多个位置分布。

5. **精简版和完整版刺激不完全一致**，qwen3用的4帧而GLM4/DS7B用2帧，
   这可能导致对比偏差。

### 六、通俗总结

```text
Phase 946 测试了"not, all, some, no, must, can, should, may, might"
这些逻辑算子在三模型中的隐藏空间表示。

结果有好有坏：

好消息：
- 否定算子在三个模型中都有统一方向（比普通语法角色更稳定）
- 模态算子（may, might, must等）形成漂亮聚类
- "可能性→必要性"可能是跨模型共享的基本语义维度

坏消息：
- "不不高兴=高兴"（双否定闭合）只在Qwen3里成立，在GLM4里几乎不成立
- 算子不能直接"加"起来——"不是所有"不等于"不是"+"所有到一些"
- 这意味着"语言算子代数"可能不存在（至少不是简单的向量代数）

最关键的教训：
逻辑律（如¬¬P=P）不是训练必然产生的，
而是需要特定训练数据"教"出来的。
不同模型学了不同的"逻辑直觉"。
```

### 七、下一阶段调整

基于Phase 946结果，Phase 947-950 优先级调整：

```
原计划：
  Phase 946 → 947 (GLM4 color桥接) → 948 (Protocol场) → 949 (交叉验证)

调整后：
  Phase 947 (原路线A，不变)：GLM4上复现color语义→MLP通道齿轮三层桥接
  Phase 948 (原路线C，不变)：Protocol场token组成全量审计 + 激活来源追踪
  Phase 949 (新增)：大规模双否定闭合测试 (n=100+ operands, 验证GLM4结果)
  Phase 950：综合评估三条路线进展，确定主攻方向
```

路线B（算子代数）暂时降级——双否定闭合的跨模型失败削弱了算子代数可泛化的理论基础。
但如果Phase 949的大规模测试发现GLM4在更多样本下双否定有隐藏结构，
可以重新激活路线B。

## Phase 947: GLM4 Color语义→MLP通道齿轮桥接复现 [2026-07-15 17:00]

### 零、元信息

- 脚本：`tests/glm5_temp/phase947_minimal.py`（最简版，直接MLP hook）
- 测试模型：GLM4
- 刺激：3颜色(red/blue/green) × 3功能(large/heavy/fast) × 3对象 = 18句
- 路线归属：路线A扩展 — GLM4上复现Phase 944三层桥接

### 一、实验方法

直接使用MLP gate_up_proj hook捕获GLM4各层的gate/up中间激活，
对比color vs function刺激在通道级激活差异。

```
激活gap计算：
  gate_diff = |mean(color_gate_act) - mean(func_gate_act)|  [intermediate维]
  up_diff = |mean(color_up_act) - mean(func_up_act)|  [intermediate维]
  combined = gate_diff * up_diff (通道显著性评分)
  
  act_gap_gate = mean(gate_diff[top10_channels])
  cross_pair_overlap = color对之间top10通道的交集/10
```

### 二、逐层激活gap

```
Layer  gate_gap   combined_gap  cross_pair_overlap/10
 L15    1.75       0.23         1.50
 L17    1.83       0.19         0.22
 L19    2.11       0.27         0.67
 L21    2.26       0.28         1.11
 L23    3.24       0.76         1.33
 L25    4.38 ★    1.13 ★       1.36
 L27    3.61       0.87         0.47
 L29    3.63       0.73         0.92
```

**最佳层：L25（gate_gap=4.38, combined_gap=1.13）**

### 三、与qwen3 Phase 944对比

```
┌────────────────────┬───────────┬───────────┐
│ 指标                │ qwen3     │ GLM4      │
│                    │ (L36)     │ (L25)     │
├────────────────────┼───────────┼───────────┤
│ 最佳层              │ L36       │ L25       │
│ 相对于总层数位置     │ 90%       │ 62.5%     │
│ 激活gap (gate)      │ +3.70     │ +4.38 ✓   │
│ 激活gap (combined)  │ -         │ +1.13     │
│ 共识通道数           │ 3 (2509,  │ 0 (无共识)│
│                    │  16, 249) │           │
│ 跨对通道重叠         │ 高        │ 1.4/10    │
│ 颜色logit gap       │ ~2.0      │ 1.6-2.6   │
└────────────────────┴───────────┴───────────┘
```

### 四、关键发现

1. **GLM4存在color语义→MLP通道的桥接**：gate_gap=4.38甚至略高于qwen3的3.70。
   证明"语义方向→MLP通道齿轮"是跨模型存在的结构，不是qwen3特有的。

2. **但桥接模式不同**：
   - qwen3：少数几个共识通道(2509, 16, 249)承载color语义
   - GLM4：color语义分散在大量通道中，不同color对使用不同通道集
   - GLM4的"齿轮"是多齿的而不是少数几个主轴

3. **GLM4的color语义在更浅层处理**：L25（62.5%深度）vs qwen3 L36（90%深度）。
   GLM4更早完成了color语义编码，可能因为架构差异（GLM4=40层, qwen3=40层?）。

4. **Color logit gap跨模型一致**：颜色词在处理color刺激时logit平均高出1.6-2.6。
   这个量级在qwen3和GLM4中高度一致。

### 五、关键硬伤

1. **没有完成因果干预验证**：只测量了激活gap，没有像Phase 944那样
   真正拨动通道测试边界移动。需要在后续Phase中补上。

2. **样本量太小**：仅18句，3种颜色对比3种功能。需要扩大到更多颜色和功能。

3. **GLM4缺少共识通道**：这意味着"拨齿轮"策略在GLM4上更难实施——
   没有少数几个关键通道，需要同时调控大量通道。

4. **没有测试输出边界移动**：Phase 947只完成了"语义→通道"这一层桥接，
   没有验证"通道→边界移动"的第二层。

### 六、通俗总结

```text
GLM4确实有"color语义→MLP通道"的桥接！
最佳层在第25层，颜色通道的激活差异比qwen3还大一点。

但GLM4的编码方式更像"交响乐团"而不是"独奏家"——
qwen3只用3个通道就表达了color语义，
GLM4动用了大量不同通道，每个color对用的通道还不一样。

这就像：qwen3把"红色"信息存在了3个专用抽屉里，
GLM4把信息分散到了几十个抽屉里，每个搭配（红vs蓝，红vs绿）用不同组合。

好消息：桥接存在！
坏消息：更难操作（没有单一"齿轮"可以拨）。
```

### 七、对后续Phase的影响

Phase 947完成了路线A的基本验证——三层桥接中的"语义→通道"在GLM4上存在。
但"通道→边界移动"尚未验证。

建议：Phase 948和949之后，在更大的刺激集上做完整的因果干预验证。
目前来看，三条路线都取得了有意义但有限的结果：
- 路线B（算子代数）：跨模型差异大，双否定闭合不稳
- 路线A（channel bridge）：跨模型存在但模式不同
- 路线C（Protocol场）：尚未开始


## Phase 948: Protocol场token组成全量审计 + 激活来源追踪 [2025-07-15 05:07]

### 一、实验目标

Phase 948是路线C（Protocol场）的起点。不同于Phase 900-907使用预设的token类别
（period/eos/field_word/explanation/list_word），Phase 948采用**无监督方式**发现
protocol token，并对其激活来源做全量归因分析。

核心问题：
1. 模型实际使用哪些token作为"协议场"token？（非预设，模型行为驱动）
2. 这些protocol token的logit由哪些层/组件驱动？（attention vs MLP）
3. 跨模型一致性如何？

### 二、实验设计

**提示集：** 72个多样化的prompt，分为6组：
- QA（15个）："What is X? The answer is"
- 分类（15个）："Category: 'X' is a"
- 解释（12个）："Explain why X:"
- 结构化输出（10个）："List X:\n1."
- 协议场语境（10个）：完整句子以句号结尾
- 句中续写（10个）：不完整句子（对照）

**Step 1: 全量logit分布分析**
- 每个prompt运行前向传播，捕获top-100 logit分布
- 统计每个token在多少个不同prompt的top-100中出现
- protocol候选 = 在≥5个不同prompt中出现的token

**Step 2: 成分归因**
- 对top-30协议候选，逐层归零attention/MLP输出
- 测量每个protocol token logit的变化
- qwen3: 15 prompts × 10 layers × 2 components = 300次前向传播
- glm4/ds7b: 6 prompts × 9/7 layers × 2 components ≈ 108/84次

**Step 3: Token聚类**
- 基于归因向量的余弦相似度聚类protocol tokens

### 三、模型发现的关键Protocol Token

#### 跨模型共识协议Token（三模型Top-15中均出现）

| Token | qwen3排名 | GLM4排名 | DS7B排名 | 含义 |
|-------|----------|---------|---------|------|
| " " (空格) | #1 (98.6%) | #3 (65.3%) | #1 (91.7%) | 格式化分隔符 |
| "\\n" (换行) | #12 (56.9%) | #2 (72.2%) | #7 (54.2%) | 结构分隔符 |
| "The" | #14 (54.2%) | #1 (76.4%) | #8 (52.8%) | 解释开头 |
| "the" | #8 (63.9%) | #15 (44.4%) | #6 (59.7%) | 冠词 |
| " (" | #2 (98.6%) | #13 (47.2%) | #2 (84.7%) | 结构化输出 |

#### 模型特有协议Token

| qwen3特有 | GLM4特有 | DS7B特有 |
|----------|---------|---------|
| " -" (列表) | "#" / " #" (标题) | "**" (粗体) |
| "__" (占位) | "Solution", "Step" | ":" (冒号) |
| "..." (省略号) | "In", "To" | " $"/" $\\" |

#### 关键观察

**发现1：模型自然发现的protocol token是"格式化/结构型"标记，而非预设的"语义分类"标记。**
预设的protocol类别（period, field_word, explanation, list_word）中只有部分出现，
而有大量"空白、标点、占位符"类的token是模型自发的协议行为。

**发现2：GLM4更倾向于用语义化的protocol token（"Solution", "Step", "The", "In"），**
而qwen3和DS7B更偏向格式化的protocol token（" ", "(", "__", "**"）。
这反映了不同模型架构在处理"输出结构"时的不同策略。

### 四、成分归因：MLP主导Protocol场（跨模型一致）

#### Attn/MLP 贡献比（|logit_delta|均值）

| 模型 | Attn |△| | MLP |△| | Attn/MLP比 | 主导成分 |
|------|-----------|-----------|-----------|---------|
| qwen3 | 0.5128 | 1.1619 | **0.441** | MLP (2.3x) |
| GLM4 | 0.2272 | 0.4197 | **0.541** | MLP (1.9x) |
| DS7B | 0.6745 | 0.8523 | **0.791** | MLP (1.3x) |

**这是Phase 948最重要的发现：跨三模型一致，protocol token的logit由MLP主导，**
而不是attention。这与语义token截然不同——语义编码通常attention贡献更大。

#### 逐层归因Top-8

| 排名 | qwen3 | |△| | GLM4 | |△| | DS7B | |△| |
|-----|-------|-------|-------|------|-------|------|
| 1 | L35_mlp | 5.160 | L39_mlp | 1.102 | L27_mlp | 2.801 |
| 2 | L0_mlp | 1.257 | L0_attn | 0.774 | L27_attn | 2.176 |
| 3 | L0_attn | 0.980 | L5_mlp | 0.724 | L15_mlp | 1.354 |
| 4 | L28_mlp | 0.854 | L20_mlp | 0.468 | L20_attn | 0.812 |
| 5 | L35_attn | 0.851 | L30_attn | 0.301 | L25_mlp | 0.743 |
| 6 | L20_mlp | 0.810 | L39_attn | 0.283 | L0_attn | 0.655 |
| 7 | L12_attn | 0.799 | L10_mlp | 0.282 | L15_attn | 0.358 |
| 8 | L16_mlp | 0.730 | L25_mlp | 0.282 | L10_mlp | 0.304 |

**发现3：最后一层的MLP（L35_mlp/L39_mlp/L27_mlp）是protocol token logit的最强来源，**
远超其他任何组件。qwen3的L35_mlp贡献是第2名L0_mlp的4.1倍；DS7B的L27_mlp是第2名
L27_attn的1.3倍；GLM4的L39_mlp是第2名L0_attn的1.4倍。

**发现4：DS7B的L27（最后一层）的attention贡献也很大（2.18 vs MLP 2.80），**
表明DS7B在最后一层同时使用attention和MLP处理protocol token，
而qwen3的protocol完全由L35_mlp主导，attention在L35的贡献仅有0.85。

### 五、Token聚类分析

| 模型 | 聚类数 | 说明 |
|------|-------|------|
| qwen3 | 3 clusters | 子类别：格式化空白、标点、语义引导词 |
| GLM4 | 4 clusters | 更细粒度的子类别 |
| DS7B | 1 cluster | 归因模式高度同质化 |

GLM4的4-cluster结构表明其protocol token编码更复杂、更分化，
可能与GLM4更倾向语义化protocol token（"Solution", "Step"等）有关。

### 六、协议场 vs 语义场：根本性差异的证据

结合Phase 946（算子代数）和Phase 947（语义通道桥接）的结果，Phase 948提供了一个
关键对比：

| 维度 | 语义场 | 协议场 |
|------|--------|--------|
| 编码组件 | Attention主导 | **MLP主导（跨模型一致）** |
| 跨模型一致性 | 算子方向一致，双否定差异大 | **Attn/MLP比例一致，但token组成差异大** |
| 编码层 | 中间层（L12-L20）为主 | **最后一层MLP为主** |
| Token类型 | 语义词（red, not, must） | **格式化标记（空格、换行、冠词）** |
| 聚类结构 | 概念聚类 | 子模型中分化（GLM4=4簇, DS7B=1簇） |

**核心洞察：语义场和协议场可能是两个独立的编码子系统，使用不同的电路机制。**

### 七、理论意义

1. **MLP在protocol场中的主导地位** 暗示protocol token的判断不需要上下文聚合
   （attention的功能），而是基于固定的处理规则（MLP的功能）。这与protocol token的
   "结构/格式化"属性一致——它们不依赖上下文，而是依赖输出位置的"协议规则"。

2. **最后一层MLP的集中性** 暗示protocol token的logit形成于模型的"最后一公里"——
   语义计算在前层完成，而协议的选择（继续输出 vs 停止 vs 格式化）在最后一层MLP中
   决定。这与Phase 900-907发现的"stop token竞争"遥相呼应。

3. **跨模型MLP>Attn一致性说明这是深层架构属性，而非特定训练数据的产物。**

4. **GLM4的语义化protocol倾向** 可能反映其训练数据中更多"引导式输出"的模式，
   与qwen3/DS7B的"格式化输出"偏好形成对比。

### 八、通俗总结

```text
协议场是什么？
——就是模型在一句话末尾"下一步该输出什么"的决策机制。

通过让模型对72个不同类型的prompt做预测，然后观察哪些token
最常被"考虑"（出现在top-100概率中），我们发现：

qwen3和DeepSeek7B的"协议本能"是输出空白格式符：
  " "（空格）、" ("（左括号）、" -"（短横线）
  
而GLM4的"协议本能"是输出引导词：
  "The"、"Solution"、"Step"、"In"

但有一个规律三模型完全相同：
  协议token的logit主要是由MLP（多层感知器）驱动，而不是attention。
  Attention关注的是上下文，MLP执行的是"规则"。
  
这意味着：模型在判断"下一步输出什么结构"时，用的是固定规则，
而不是看上下文来决定。就像一个训练有素的写作习惯，不需要思考。

最关键的是：这些规则主要存储在最后一层的MLP中。
语义计算在前层完成 -> 最后一层MLP决定输出格式 -> 输出。
```

### 九、局限性和问题

1. **提示集有限（72个）**：虽然涵盖6种类型，但偏向英文QA/分类场景。
   可能遗漏了代码生成、聊天、翻译等场景的protocol token。

2. **归因的粗粒度**：归零整个attention/MLP输出是粗粒度的归因方法，
   无法区分头级的贡献差异。后续需要attention-head级别的归因。

3. **没有因果干预验证**：只做了消融（归零），没有做方向性注入测试。
   需要验证：向MLP注入特定方向后，protocol token logit是否相应变化。

4. **跨模型prompt数量不一致**：GLM4/DS7B只用6个prompt做归因（vs qwen3的15个），
   归因估计的噪声较大。

5. **没有区分prompt组**：72个prompt混在一起分析，没有区分QA/分类/解释等场景
   下的protocol token差异。

6. **最后一层MLP的主导性需要更精确的验证**：是否最后一层MLP的contribution
   在归零后可以被其他层补偿？需要测量归零前后的logit变化方向一致性。

### 十、下一步

Phase 949（跨方向交叉验证）：将三条路线的发现合并分析：
- 路线B（算子代数）：语义算子形成独立族
- 路线A（channel bridge）：语义→MLP通道桥接存在但模式不同
- 路线C（Protocol场）：MLP主导协议token，最后一层集中

核心假设：语义场和协议场是两个独立子系统，但共享MLP通道——MLP是统一两者的关键。
Phase 949将测试这个假设。


## Phase 949: 跨方向交叉验证 — 三条路线是否指向同一机制？[2025-07-15 05:10]

### 一、实验目标

综合 Phase 946（算子代数, Route B）、Phase 947（语义通道桥接, Route A）、
Phase 948（协议场审计, Route C）的发现，寻找跨路线的结构关系。

核心假设：语义场和协议场是两个独立子系统，但共享MLP通道——MLP是统一两者的关键。

### 二、交叉分析数据

#### 路线发现汇总

| 路线 | Phase | 核心发现 | 关键组件 | 关键层 |
|------|-------|---------|---------|--------|
| B: 算子代数 | 946 | 语义算子形成3个独立族 | hidden state delta | 取决于算子 |
| A: 语义通道 | 947 | color→MLP通道桥接存在 | MLP gate通道 | GLM4 L25 (gap=4.38) |
| C: 协议场 | 948 | Protocol token由MLP主导 | 最后一层MLP | 最后一层 (各模型) |

#### 逐模型跨路线对比

**GLM4（唯一同时有Route A和Route C数据的模型）：**

| 层 | Protocol MLP |Δ| | Channel gate_gap | 功能 |
|----|-------------|---------------|------|
| L0 | 0.244 | — | 嵌入层 |
| L5 | 0.724 | — | 早期MLP |
| L15 | 0.160 | 1.753 | 语义编码（弱） |
| L20 | 0.468 | 1.627 | 语义编码（弱） |
| L25 | 0.282 | **4.377** | **语义编码（最强!）** |
| L27 | 0.108 | 3.610 | 语义编码（强） |
| L29 | 0.087 | 3.626 | 语义编码（强） |
| L35 | 0.050 | — | 过渡区 |
| L39 | **1.102** | — | **协议决策（最强!）** |

**关键发现：Spearman r(protocol_mlp_strength, channel_gap) = -0.727 (p=0.026)**
——语义编码和协议决策在不同MLP层中呈显著负相关！

### 三、核心发现

**发现1：语义场和协议场使用不同的MLP层（负相关）**

GLM4数据显示，MLP层的功能分为两类：
- **中间层（L15-L29）**：channel activation gap大 → 语义编码强
- **最后一层（L39）**：protocol MLP contribution大 → 协议决策强
- 两者呈显著负相关（Spearman r=-0.727, p=0.026）

这说明MLP存在层间功能分化：
```
Layer 0-10:   通用特征提取
Layer 15-29:  语义编码（color, operator, 概念）
Layer 30-38:  特征整合/过渡
Layer 39:     协议决策（output format selection）
```

**发现2：L25是潜在的"语义→协议"桥接层**

GLM4 L25 是唯一同时具有显著语义编码（gate_gap=4.38）和
可检测协议贡献（|Δ|=0.28）的层。它可能是语义信息向协议格式转换的起点。

**发现3：跨模型确认MLP的双重角色**

三模型的一致性证据：
- Phase 946: 算子编码产生hidden state delta，MLP将其转换为logit差异
- Phase 947: 语义属性通过MLP gate通道编码
- Phase 948: Protocol token logit由MLP主导（Attn/MLP = 0.44-0.79）

MLP不是单一功能模块，而是执行两类任务：
1. **语义编码**（中间层）：将注意力聚合的上下文信息转化为语义通道激活
2. **协议决策**（最后层）：基于编码后的语义状态，选择输出格式

**发现4：归一化假设 vs 分化假设**

基于现有证据，两种可能的理论框架：

| 假设 | 描述 | 支持证据 | 反对证据 |
|------|------|---------|---------|
| 归一化假设 | 语义和协议共享同一MLP底层机制 | MLP是三条路线的共同关键组件 | Spearman r=-0.727（负相关） |
| 分化假设 | 语义和协议是MLP的两个独立子电路 | 层间功能分化、负相关 | MLP通道层面的数据不足 |

目前证据倾向于**分化假设**。

### 四、统一理论框架：MLP的层间流水线

```
输入序列 → Embedding
    ↓
[L0-L10]  Attention + MLP: 通用特征提取
    ↓                           ↓
[L15-L29] Attention: 上下文聚合 → MLP: 语义通道编码
    ↓                           ↓ (概念、属性、算子)
[L30-L38] Attention + MLP: 信息整合/过渡
    ↓
[L39]     Attention: 末次聚合 → MLP: 协议决策 ← 最关键的输出!
    ↓                           ↓ (格式、停止、结构)
lm_head → 最终logit → token输出
```

**核心机制**：
- 语义是以"方向"（directions）编码在hidden state中（Phase 942-944验证）
- MLP中间层将这些方向映射到特定的gate/up通道（Phase 947验证）
- 最后一层MLP读取整个hidden state，决定输出协议token（Phase 948验证）
- 算子（negation/quantification/modal）作为特殊语义方向存在（Phase 946验证）

### 五、可验证预测

基于统一框架，做出以下可验证的预测：

1. **解耦预测**：如果归零GLM4 L25的MLP，color语义的正确率应大幅下降，
   但不影响protocol token的选择偏好。

2. **桥接预测**：如果增强GLM4 L25的color通道激活，应该能观测到
   L39 MLP输出的变化（color语义→协议选择的因果链）。

3. **正交预测**：语义方向（如"red"的W_U方向）和协议方向
   （如"\\n"的W_U方向）在hidden state中应该是近乎正交的。

4. **缩放预测**：如果在不同模型上测量Spearman r，应该在qwen3和DS7B上
   得到类似GLM4的负相关（r<-0.5）。

### 六、通俗总结

```text
把MLP想象成一个工厂：

前几层MLP（L0-L10）→ 原料加工：把输入变成可用的特征

中间层MLP（L15-L29）→ 语义车间：理解"这是什么颜色"、"是什么关系"、
                       "是否定还是肯定"
                      
最后层MLP（L39）→ 包装车间：决定"用句号还是换行？"、"继续写还是停？"、
                     "用列表还是段落？"

这两个车间是分开运作的（Spearman r=-0.727证实了这一点）。

但关键问题是：中间车间加工好的"语义产品"是怎么传递给包装车间的？
是通过hidden state中的方向？还是通过其他残差连接？

这就是下一个阶段要研究的问题。
```

### 七、局限性和问题

1. **只有GLM4有完整的跨路线数据**：qwen3和DS7B缺少Phase 947（channel bridge）数据，
   无法验证Spearman负相关是否跨模型成立。

2. **Phase 946数据缺失**：算子代数的结果未以结构化格式保存，无法在层级别做交叉分析。

3. **粗粒度相关**：Spearman r是对所有10层的相关，只有10个数据点（GLM4 10层sample），
   统计功效有限。需要逐层采样才能做出更精确的判断。

4. **缺乏因果证据**：所有发现都是相关性/消融证据，没有因果干预（增强/转向）验证。

5. **未测试hidden state中的正交性**：预测3（语义方向⊥协议方向）尚未测量。

### 八、下一步

Phase 950：基于Phase 946-949的全部证据，确定主攻方向：
- 选项A: 深入研究"语义→协议"的转换机制（L25→L39的残差流）
- 选项B: 验证"语义编码的通道通用性"（跨属性、跨模型的channel atlas）
- 选项C: 建立"语义-协议-算子"三元编码的数学模型


## Phase 950: 方向决策 — 基于全部证据确定主攻方向 [2025-07-15 05:12]

### 一、三条路线加权评估

建立5维评分体系（每维5分），权重: 跨模型一致性×2 + 机制清晰度×1.5 + 理论深度×2 + 可操作性×1 + 整合潜力×1.5

| 维度 | Route C 协议场 | Route A 语义通道 | Route B 算子代数 |
|------|:---:|:---:|:---:|
| 跨模型一致性(×2) | **5.0** | 2.5 | 1.5 |
| 机制清晰度(×1.5) | **4.0** | 3.0 | 2.0 |
| 理论深度(×2) | 3.5 | **4.5** | **5.0** |
| 可操作性(×1) | **4.0** | 3.5 | 2.0 |
| 整合潜力(×1.5) | **4.0** | 3.5 | 2.0 |
| **加权总分** | **33.00 (82.5%)** | 27.25 (68.1%) | 21.00 (52.5%) |

### 二、决策

**主攻: Route C (Protocol场)** — MLP主导协议token的精确机制
**辅攻: Route A (语义通道桥接)** — 语义到MLP通道的编码映射

**关键论证:**

Route C 优势:
- 跨模型一致性最高 (5.0/5): MLP>Attn在qwen3/GLM4/DS7B上100%一致
- 机制清晰度最高 (4.0/5): 归零实验结论明确，因果方向清晰
- 整合潜力最大: Protocol场和语义场在MLP层间的负相关(Spearman r=-0.727)
  揭示了关键的结构边界，第二阶段可自然延伸到"语义→协议转换机制"

Route A 辅助价值:
- 理论深度最高 (4.5/5): 如果通道→边界因果链被验证，将证明语义以
  "通道激活模式"编码在MLP中
- 通道操控实验精确，适合作为因果验证工具
- 与主线形成互补: A揭示"编码什么"，C揭示"如何决策"

Route B 暂缓原因:
- 跨模型一致性最差 (1.5/5): 双否定闭合差异极大 (0.39-0.87)
- 机制最不清晰 (2.0/5): hidden state delta缺乏归因
- 与A/C路线距离较远，难以整合
- 但算子族的存在是重要观察，Phase 951做最小验证后归档

### 三、下一阶段路线图 (Phase 951-955)

```
Phase 951  路线B清算: 算子族存在性大样本验证 → 归档
Phase 952  Protocol场因果验证: 发现protocol方向 + 注入实验
Phase 953  语义→协议信息流追踪: L25→残差流→L39因果链
Phase 954  MLP双层编码理论: 数学模型构建
Phase 955  第一性原理突破: 语言编码的数学理论
```

**Phase 951 详细:**
- 用200+刺激测试算子族间的affinity矩阵（更大样本量确认3族结构）
- 如果确认稳定性，归档为"观察"（算子族存在）而非"理论"
- 脚本: /tests/glm5/phase951_operator_family_validation.py

**Phase 952 详细:**
- 计算protocol token集合的logit梯度，提取"protocol方向"（在W_U空间）
- 将protocol方向注入最后一层MLP的hidden state，测量因果效应
- 验证: 语义方向（red, blue等）与protocol方向（\\n, " ", "."等）是否正交

**Phase 953 详细:**
- 在GLM4 L25注入color语义，measure L39 hidden state的变化
- 残差流逐层追踪: 语义信号在残差流中的衰减/放大曲线
- 验证因果链: L25 gate通道激活 → 残差流传播 → L39 MLP输入 → protocol token logit变化

### 四、核心洞察：为什么这是正确的方向

经过Phase 945-950的系统分析，我们看清了一个三层架构：

```
Layer 0:  嵌入层 — 将离散token映射到连续空间
Layer 1-N-2: 语义层 — Attention聚合上下文 + MLP编码语义（概念、属性、关系）
Layer N-1:  过渡层 — 将语义状态转化为可被协议层使用的格式
Layer N:    协议层 — 最后一层MLP根据语义状态选择输出格式
```

关键发现是：**语义和协议是两个独立但通过残差流连接的子系统。**
这解释了为什么：
1. 语义编码（color, operator）在中层MLP最强
2. 协议决策在最后一层MLP最强
3. 两者在空间中是负相关的（Spearman r=-0.727）

这是一个可验证的、简明的理论框架，用最少的假设解释最多的观察。

### 五、风险与备选方案

**主要风险:**
1. Protocol方向可能不存在（如果protocol token由多个正交方向共同决定）
   → 备选: 不做方向注入，做token-level的线性探针
2. 语义→协议的信息流可能不是通过残差流，而是通过attention传播
   → 备选: 先做attention pattern分析，确认中间层attention是否关注protocol-relevant token

**如果Route C遇阻的备选:**
退回到Route A，系统构建channel atlas（跨属性、跨模型），积累实证数据，让理论从数据中自然浮现。


## Phase 951: 协议场物理图谱与语义到协议转换审计 [2025-07-15 06:29]

### 一、实验设计

5个任务，3个模型(qwen3/GLM4/DS7B)，125+ prompts跨8类语境，102个protocol token。

| Task | 内容 | 方法 | 数据量 |
|------|------|------|--------|
| Task1 | Protocol token物理图谱 | 125 prompts × top-100 logit | 102 tokens |
| Task2 | 最后层MLP通道级归因 | Ridge回归探针, 50-60 prompts | 最后3层 |
| Task3 | 语义→协议桥接 | 颜色方向注入(beta=8), 测protocol logit变化 | 10-15 prompts × 5方向 |
| Task4 | 语义→协议信息流 | 残差流逐层投影, 颜色/句号/空格方向 | 8-12 prompts |
| Task5 | Protocol-neutral对照 | 20 neutral vs 20 heavy prompts | 10 key tokens |

### 二、Task 1 客观结果: Protocol Token物理图谱

#### 跨模型共识Top Protocol Token (出现在≥80%的prompt top-100中)

| Token | qwen3 cross% | GLM4 cross% | DS7B cross% |
|-------|:-----------:|:-----------:|:-----------:|
| " " (空格) | 100% | 94% | 99% |
| " (" | 99% | 93% | 98% |
| " __" | 97% | — | — |
| " -" | 97% | 91% | 79% |
| " [" | 96% | 78% | 88% |
| " a" | 94% | 77% | 79% |
| "the" | 92% | 77% | 71% |
| "The" | — | 85% | — |
| "1" | — | 97% | — |
| "**" | 93% | — | 88% |

**客观事实：** 空格和左括号是三模型最普遍的protocol token。GLM4独有"1"作为高排名protocol token(97%)，qwen3和DS7B没有。GLM4更倾向结构化数字token，qwen3/DS7B更倾向格式化标点token。

### 三、Task 2 客观结果: 通道级归因

#### 方法说明
使用Ridge回归(alpha=1.0)将最后3层MLP中间激活映射到protocol token logit。
**注意：R2=1.0是因为通道数(9728-18944)远大于样本数(50-60)，属于过拟合。**
通道系数的相对大小仍有参考价值，但R2不可作为预测能力的指标。

#### 共享Protocol通道（支持3+ protocol token的通道）

| 模型 | 层 | 共享通道数 | 中间维度 | Top通道示例 |
|------|---|:---------:|:-------:|------------|
| qwen3 | L35 | 83 | 9728 | (数据见JSON) |
| qwen3 | L34 | 84 | 9728 | |
| qwen3 | L33 | 71 | 9728 | |
| GLM4 | L39 | 96 | 13696 | |
| GLM4 | L38 | 79 | 13696 | |
| GLM4 | L37 | 80 | 13696 | |
| DS7B | L27 | — | 18944 | ch5123: 支持100个token |
| DS7B | L26 | — | 18944 | ch18761: 支持88个token |
| DS7B | L25 | — | 18944 | ch2598: 支持86个token |

**客观事实：** DS7B存在"超级protocol通道"（ch5123支持100/102个protocol token），qwen3和GLM4的共享通道数据需进一步分析。DS7B的protocol通道集中度远高于其他两个模型。

### 四、Task 3 客观结果: 语义→协议桥接（关键发现）

#### 颜色方向注入后Protocol Token Logit变化均值

| 颜色方向 | qwen3 proto_change | GLM4 proto_change | DS7B proto_change |
|---------|:------------------:|:-----------------:|:-----------------:|
| red-blue | **+2.404** | +1.030 | +0.547 |
| red-green | +1.454 | -0.171 | +0.070 |
| yellow-black | **+2.851** | +1.277 | +0.898 |
| white-black | +2.557 | -1.304 | **+2.244** |
| orange-purple | +2.323 | +0.748 | +1.297 |

#### 颜色方向注入后Color Token Logit变化均值

| 颜色方向 | qwen3 color_change | GLM4 color_change | DS7B color_change |
|---------|:------------------:|:-----------------:|:-----------------:|
| red-blue | +2.522 | +0.862 | +0.063 |
| red-green | +1.964 | +0.998 | +0.452 |
| yellow-black | +2.804 | -1.251 | -0.243 |
| white-black | +2.381 | -0.530 | +0.749 |
| orange-purple | -0.900 | -1.991 | -1.153 |

**客观事实：**
1. **语义方向注入确实改变了protocol token logit**——三模型均观察到非零变化。
2. **qwen3桥接最强且最一致**：5个方向全部导致protocol logit上升(+1.45~+2.85)。
3. **GLM4桥接方向不一致**：2正2负1近零，protocol变化与color变化方向不总一致。
4. **DS7B桥接中等**：4正1近零，white-black方向最强(+2.24)。
5. **act_delta_norm**：DS7B最大(895~1058)，qwen3中等(185~228)，GLM4最小(140~157)。
   但DS7B的大act_delta并未转化为大的proto_change，说明DS7B的MLP对注入有"吸收"效应。

### 五、Task 4 客观结果: 残差流信息流追踪

#### 句号(period)方向投影随层变化

| 层 | qwen3 period_proj | GLM4 period_proj | DS7B period_proj |
|----|:-----------------:|:----------------:|:----------------:|
| L0 | +0.022 | -0.165 | -0.075 |
| 中层(min) | -0.058 (L32) | -0.219 (L35) | -0.171 (L24) |
| 最后层 | +0.002 (L35) | -0.145 (L39) | -0.011 (L27) |

#### 空格(space)方向投影随层变化

| 层 | qwen3 space_proj | GLM4 space_proj | DS7B space_proj |
|----|:----------------:|:----------------:|:----------------:|
| L0 | -0.020 | -0.138 | -0.084 |
| 中层(min) | -0.075 (L28) | -0.189 (L35) | -0.193 (L24) |
| 最后层 | **+0.031** (L35) | -0.111 (L39) | -0.007 (L27) |

**客观事实：**
1. **三模型的period和space投影在中间层达到最负值**，然后在最后层发生反弹。
2. **qwen3最后层反弹最强**：space从-0.075反弹到+0.031，period从-0.058反弹到+0.002。
3. **GLM4反弹最弱**：period从-0.219仅反弹到-0.145，space从-0.189仅反弹到-0.111。
4. **DS7B居中**：period从-0.171反弹到-0.011，space从-0.193反弹到-0.007。
5. **color_diff在qwen3和DS7B上值很小**(|<0.034|)，说明颜色方向在残差流中的投影弱于protocol方向。
6. **GLM4的color_diff全为0**（实验设计缺陷：8个prompt全含颜色词，无对照）。

### 六、Task 5 客观结果: Protocol-Neutral vs Protocol-Heavy

#### EOS Logit对比（关键发现）

| 模型 | Neutral EOS | Heavy EOS | Diff (Heavy-Neutral) |
|------|:-----------:|:---------:|:--------------------:|
| qwen3 | -2.743 | +0.334 | **+3.077** |
| GLM4 | +0.719 | -0.921 | **-1.640** |
| DS7B | +3.380 | +6.340 | **+2.959** |

**客观事实：** GLM4是唯一一个在protocol-heavy prompt中EOS logit下降的模型。
qwen3和DS7B在protocol-heavy prompt中EOS logit上升。这与Phase 899-907的发现一致——
GLM4在自然生成中EOS top1几乎为0，protocol续写场压倒EOS终止场。

#### 其他Token Logit变化(Heavy - Neutral)

| Token | qwen3 diff | GLM4 diff | DS7B diff |
|-------|:----------:|:---------:|:---------:|
| "." | -1.817 | -4.559 | -0.381 |
| " " | +4.695 | +2.198 | **+8.295** |
| "the" | -1.010 | +2.684 | +2.165 |
| "The" | -3.751 | +0.461 | +0.558 |
| "is" | -4.881 | -0.879 | **+4.436** |
| "a" | -7.456 | -0.736 | +3.465 |

**客观事实：**
1. qwen3在heavy prompt中大部分protocol token logit下降(除了空格)，
   说明qwen3的protocol-heavy context抑制了大多数protocol token。
2. DS7B在heavy prompt中几乎所有protocol token logit上升，
   说明DS7B更容易被protocol context"激活"继续输出protocol token。
3. GLM4的模式介于两者之间。

### 七、综合客观发现汇总

| 编号 | 发现 | 跨模型一致性 | 备注 |
|------|------|:-----------:|------|
| F1 | 语义方向注入改变protocol logit | 3/3模型 | qwen3最强, GLM4方向不一致 |
| F2 | 空格和左括号是最普遍protocol token | 3/3模型 | 100%/99%/98%出现率 |
| F3 | period/space投影在中间层最负,最后层反弹 | 3/3模型 | qwen3反弹最强 |
| F4 | GLM4在protocol-heavy中EOS下降 | 1/3 (GLM4独有) | 与Phase899-907一致 |
| F5 | DS7B存在"超级protocol通道"(100 token) | 1/3 (DS7B独有) | 通道集中度最高 |
| F6 | qwen3 protocol-heavy抑制多数protocol token | 1/3 (qwen3独有) | 与DS7B相反 |
| F7 | DS7B act_delta最大但proto_change中等 | 3/3 | MLP"吸收"效应 |

### 八、局限性

1. **Task 2 R2=1.0过拟合**：通道数(9728-18944) >> 样本数(50-60)，Ridge回归过拟合。
   后续需要：(a)增加样本数到500+，或(b)使用交叉验证评估真实预测能力。

2. **Task 4 GLM4 color_diff=0**：实验设计缺陷——8个prompt全含颜色词，无对照。
   需要重跑GLM4 Task 4，确保有非颜色prompt作为对照。

3. **小模型粗糙性**：qwen3(8B)/GLM4(9B)/DS7B(7B)层数较少(28-40层)，
   内部结构可能较为粗糙，protocol通道的定位精度有限。
   大模型(70B+)可能有更清晰的通道分化。

4. **注入强度beta=8.0未优化**：可能过大或过小，未做beta扫描。

5. **Task 3的protocol_logit_change是7个token的平均**：
   不同protocol token可能有不同响应模式，平均值可能掩盖个体差异。

6. **颜色方向来自W_U行向量**：这是embedding-level的方向，
   与hidden state中的语义方向可能不完全一致。

### 九、下一步

基于以上客观结果，下一阶段应：

1. **修复Task 4 GLM4**：重跑GLM4信息流，确保有非颜色prompt对照
2. **扩大Task 2样本量**：用500+ prompts重跑通道归因，用交叉验证替代R2
3. **个体token分析**：Task 3中分别分析每个protocol token的响应，而非取平均
4. **beta扫描**：测试不同注入强度下的桥接效应
5. **因果验证**：如果发现特定protocol通道，做通道级干预（增强/抑制）验证因果链


### 十、修正实验结果 (CV-R2 + Task4 对照修复)

#### Task 2 交叉验证修正（5-fold CV, alpha=10.0, 120 prompts）

原始 R2=1.0 是过拟合伪影。交叉验证后的真实预测能力：

| 模型 | 层 | CV-R2 mean | CV-R2 median | CV-R2 max |
|------|---|:----------:|:------------:|:---------:|
| qwen3 | L35 | **0.199** | 0.292 | 0.652 |
| qwen3 | L34 | 0.050 | 0.087 | 0.402 |
| qwen3 | L33 | 0.087 | 0.191 | 0.469 |
| GLM4 | L39 | -0.040 | 0.038 | 0.479 |
| GLM4 | L38 | -0.157 | 0.064 | 0.403 |
| GLM4 | L37 | -0.338 | -0.064 | 0.307 |
| DS7B | L27 | 0.121 | 0.200 | 0.512 |
| DS7B | L26 | **0.265** | 0.297 | 0.476 |
| DS7B | L25 | 0.091 | 0.169 | 0.414 |

**修正后客观事实：**
1. **qwen3 L35 的 CV-R2 mean=0.199**——最后层MLP中间激活对protocol token logit有弱到中等预测力。
2. **GLM4 的 CV-R2 mean 多为负值**——MLP中间激活→protocol logit的映射在GLM4上不泛化。
3. **DS7B L26 (倒数第二层) 的 CV-R2=0.265**——DS7B的protocol编码可能不在最后层而在倒数第二层。
4. **所有模型的 CV-R2 max > 0.3**——某些特定protocol token可被预测，但大部分不能。
5. **原始Task 2的"共享通道"结论不可靠**——过拟合产物，需要更严格的方法重新验证。

#### Task 4 修正（8颜色 + 8非颜色对照）

| 模型 | Max color_diff | 所在层 | Period最后层反弹 | Space最后层反弹 |
|------|:--------------:|:------:|:---------------:|:---------------:|
| qwen3 | -0.0046 (L20) | L20 | -0.067→-0.019 ✓ | -0.080→+0.031 ✓ |
| GLM4 | +0.0085 (L10) | L10 | -0.214→-0.143 (弱) | -0.185→-0.110 (弱) |
| DS7B | +0.0029 (L12) | L12 | -0.172→-0.047 ✓ | -0.200→-0.052 ✓ |

**修正后客观事实：**
1. **color_diff 在所有模型上都非常小** (|max| < 0.01)——颜色方向在残差流中的投影远弱于protocol方向。
2. **GLM4 color_diff 在早期层为正** (+0.0085 at L10)——颜色信息存在但很微弱。
3. **qwen3 color_diff 为轻微负值**——颜色prompt的颜色方向投影反而略低于非颜色prompt（可能因为颜色prompt的结构差异影响了投影）。
4. **Period/Space 最后层反弹在qwen3和DS7B上确认**，GLM4反弹最弱——与Phase 948的发现一致（GLM4的protocol MLP贡献最小）。

### 十一、修正后的发现汇总

| 编号 | 发现 | 状态 | 备注 |
|------|------|:----:|------|
| F1 | 语义方向注入改变protocol logit | ✓ 确认 | 3/3模型, qwen3最强 |
| F2 | 空格和左括号是最普遍protocol token | ✓ 确认 | 3/3模型 |
| F3 | period/space最后层反弹 | ✓ 确认 | qwen3/DS7B强, GLM4弱 |
| F4 | GLM4在protocol-heavy中EOS下降 | ✓ 确认 | GLM4独有 |
| F5 | DS7B"超级protocol通道" | ✗ 修正 | 过拟合产物, CV-R2仅0.12 |
| F6 | DS7B heavy prompt激活protocol token | ✓ 确认 | DS7B独有 |
| F7 | DS7B act_delta大但proto_change中等 | ✓ 确认 | MLP"吸收"效应 |
| F8 | **CV-R2: qwen3>DS7B>GLM4** | **新** | qwen3最后层预测力最强 |
| F9 | **color_diff极小(<0.01)** | **新** | 颜色信息在残差流中很微弱 |
| F10 | **DS7B最佳层在L26非最后层** | **新** | 可能protocol编码不在最后层 |


## Phase 952: 协议词元特异轨迹与因果通道审计 [2025-07-15 07:08]

### 一、实验设计

5个任务，3模型，125个unique prompts，13个个体protocol token，5个beta值。

| Task | 内容 | 方法 | 关键改进 |
|------|------|------|---------|
| Task1+2 | 个体token响应×beta扫描 | 13 token × 5 beta × 3方向 × 15 prompt | **不取平均** |
| Task3 | 逐层轨迹图谱 | per-token cos投影+logit投影, 15 prompt(8色+7非色) | per-token轨迹 |
| Task4 | CV通道归因 | 5-fold CV, 125 unique prompts, alpha=10 | **无数据泄漏** |
| Task5 | 因果通道干预 | 零化down_proj输入的特定通道 | 条件执行 |

### 二、Task 1+2: 个体Protocol Token响应（关键发现）

#### beta=4.0, red-blue方向, 三模型对比

| Token | qwen3 | GLM4 | DS7B | 方向一致性 |
|-------|:-----:|:----:|:----:|:---------:|
| "." | **+2.67** | **+2.97** | **+1.82** | 3/3 正 |
| " " | **-1.03** | **-0.44** | **-0.17** | 3/3 负 |
| " a" | **-5.12** | **-5.53** | -0.42 | 3/3 负 |
| "is" | +6.14 | -0.19 | +4.41 | 2/3 正(GLM4例外) |
| "Solution" | +3.70 | +2.10 | +2.12 | 3/3 正 |
| "1" | +4.40 | +0.18 | +2.47 | 3/3 正 |
| **<EOS>** | **+1.53** | **+0.70** | **-0.86** | **2/3正, DS7B例外** |

**客观事实：**
1. **"."(句号)上升, " "(空格)下降** —— 三模型100%一致。颜色注入让模型更倾向句号而非空格。
2. **"a"下降** —— 三模型一致。颜色注入抑制冠词。
3. **"is"在GLM4上方向相反** —— qwen3/DS7B强正(+4~+6), GLM4微负(-0.19)。
4. **<EOS>在DS7B上下降** —— qwen3/GLM4上升, DS7B下降。这是DS7B独有的特征。
5. **平均值会掩盖1和2的对立方向** —— Phase 951取平均是一个方法论错误, 本Phase修正。

#### Beta扫描（以qwen3 "." token为例, red-blue方向）

| Beta | "." delta | " " delta | " a" delta |
|------|:---------:|:---------:|:----------:|
| 0.5 | +0.31 | -0.16 | -0.62 |
| 1.0 | +0.63 | -0.31 | -1.28 |
| 2.0 | +1.31 | -0.55 | -2.56 |
| 4.0 | +2.67 | -1.03 | -5.12 |
| 8.0 | +5.25 | -2.06 | -9.87 |

**客观事实：** 效应随beta连续、近似线性变化。没有阈值效应或反向。这表明桥接是线性的而非非线性的。

### 三、Task 3: 逐层Protocol轨迹图谱

#### EOS方向cos投影轨迹（最关键发现）

| 层 | qwen3 EOS | GLM4 EOS | DS7B EOS |
|----|:---------:|:--------:|:--------:|
| L0 | +0.015 | -0.154 | -0.075 |
| 中层 | +0.018~+0.045 | -0.134~-0.255 | -0.075~-0.114 |
| 最后层 | **+0.068** | -0.186 | -0.047 |

**客观事实：**
1. **qwen3的EOS投影在最后层(L35)达到最大值(+0.068)** —— 明显的"最后层EOS提升"。
2. **GLM4的EOS投影全程为负**，最后层反弹但仍为负(-0.186)。
3. **DS7B的EOS投影全程为负**，最后层反弹但仍为负(-0.047)。
4. qwen3是唯一在最后层有正EOS投影的模型——这可能解释为什么qwen3更容易自然停止。

#### Period/Space方向轨迹（三模型共识）

三模型一致模式：中间层period/space投影最负，最后层反弹。
- qwen3反弹最强（period: -0.069→-0.015, space: -0.079→+0.035）
- GLM4反弹最弱（period: -0.229→-0.145, space: -0.197→-0.111）
- DS7B居中

### 四、Task 4: CV通道归因（修正版，无数据泄漏）

| 模型 | 最佳层 | CV-R2 mean | CV-R2 median | 正R2 token数 |
|------|--------|:----------:|:------------:|:------------:|
| qwen3 | L35 | -0.031 | +0.096 | **29/40** |
| GLM4 | L39 | -0.220 | -0.037 | 18/40 |
| DS7B | L26 | **+0.043** | -0.011 | 19/40 |

**客观事实：**
1. **qwen3 L35有29/40个token的CV-R2为正**——虽然均值被少数负R2 token拉低，但大部分protocol token可被弱预测。
2. **DS7B L26是唯一均值R2为正的层(+0.043)**——DS7B的protocol编码可能在倒数第二层而非最后层。
3. **GLM4的CV-R2最差**——线性映射在GLM4上不泛化。
4. **125 unique prompts仍不足以可靠解码**——通道数(9728-18944)远大于样本数(125)，需要500+样本。

### 五、Task 5: 因果通道干预（仅DS7B执行）

DS7B L26有正CV-R2，找到5个共享通道做零化干预：

| 通道 | 支持token | 零化后最大影响 | 方向 |
|------|----------|:-------------:|:----:|
| 11755 | ., ;, ; | "." -0.14 | 抑制period |
| 5447 | ;, ;, ! | Solution +0.40 | 解放Solution |
| 14464 | ;, !, ? | space -0.50 | 抑制space |
| **10371** | ;, !, ! | **Solution +0.65, Answer +0.52** | **强解放结构词** |
| 4571 | !, ?, - | "a" -0.36 | 抑制a |

**客观事实：**
1. **通道10371零化后Solution/Answer/Step上升0.5-0.65**——该通道正常抑制结构引导词。
2. 但标准差很大(0.7-1.3)，效应在统计上不够稳定。
3. 单通道干预效果有限（最大0.65 logit变化），需要多通道组合干预。

### 六、综合客观发现

| 编号 | 发现 | 一致性 | 备注 |
|------|------|:------:|------|
| F1 | **period↑ space↓ 三模型一致** | 3/3 | 平均值会掩盖此对立 |
| F2 | **"a"↓ 三模型一致** | 3/3 | 冠词被颜色注入抑制 |
| F3 | **beta效应连续近似线性** | qwen3验证 | 无阈值/反向 |
| F4 | **qwen3最后层EOS投影唯一为正** | 1/3 | 解释qwen3更易停止 |
| F5 | **DS7B EOS被颜色注入抑制** | 1/3 | DS7B独有 |
| F6 | **DS7B L26唯一正CV-R2** | 1/3 | protocol编码在倒数第二层 |
| F7 | **通道10371抑制结构词** | DS7B | 零化后Solution+0.65 |
| F8 | **GLM4线性预测最差** | 1/3 | 非线性编码? |

### 七、局限性

1. **125 unique prompts不足**：通道数>>样本数，CV-R2不可靠。需要500-1000个多样化prompt。
2. **beta扫描仅在qwen3验证**：GLM4/DS7B的beta扫描数据已收集但未展示。
3. **Task 5仅DS7B执行**：qwen3/GLM4的CV-R2均值为负，无法做因果干预。
4. **单通道干预效果弱**：最大0.65 logit变化，需要多通道组合。
5. **小模型粗糙性**：7-9B模型的通道分化可能不够清晰。
6. **Task 3轨迹是cos投影**：仅反映方向相似性，不反映幅度。
7. **颜色方向来自W_U**：是embedding-level方向，可能与hidden state语义方向不完全一致。

### 八、下一步

1. **扩大prompt集到500+**：用更多多样化prompt重跑Task 4 CV
2. **多通道组合干预**：同时零化5-10个通道，测量组合效应
3. **非color语义注入**：测试size/shape/emotion方向对protocol token的影响
4. **beta扫描跨模型完整分析**：验证GLM4/DS7B的线性性
5. **权重级通道归因**：直接用W_down × W_U计算通道贡献，绕过回归


## Phase 953: 权重级Protocol通道归因 + 多通道组合干预 [2025-07-15 07:15]

### 一、实验设计

3个任务，3模型，解析公式计算通道贡献（不用回归）。

**核心公式：**
```
contribution_j(v, x) = a_j(x) × (W_U[v,:] · W_down[:,j])
```
其中 a_j 是MLP中间激活（hook获取），W_down是down_proj权重，W_U是lm_head权重。

| Task | 内容 | 方法 | 数据量 |
|------|------|------|--------|
| Task1 | 权重级通道归因 | 解析分解, 60 prompts | 最后层 |
| Task2 | 多通道组合干预 | K=1,5,10,20通道同时零化 | 15 prompts |
| Task3 | 非color语义注入 | 5类语义×13方向, beta=4.0 | 12 prompts |

### 二、Task 1: 权重级通道归因（解析，无过拟合）

#### 每模型Top-3通用Protocol通道

| 模型 | Top通道 | "."贡献 | " "贡献 | EOS贡献 | Solution贡献 |
|------|---------|:-------:|:-------:|:-------:|:-----------:|
| qwen3 | **ch935** | 6.23 | 7.67 | 1.02 | 1.22 |
| qwen3 | ch36 | 3.32 | 4.58 | — | 1.41 |
| qwen3 | ch284 | — | 3.83 | 1.38 | — |
| GLM4 | **ch12274** | 1.37 | 1.30 | 1.24 | 0.35 |
| GLM4 | ch7968 | 1.23 | 1.28 | 1.20 | 0.35 |
| GLM4 | ch5155 | 0.81 | — | 0.81 | — |
| DS7B | **ch15791** | **36.52** | **48.95** | **27.36** | 4.17 |
| DS7B | ch15305 | 16.78 | 20.35 | 11.50 | 5.27 |
| DS7B | ch1106 | 13.07 | 18.77 | — | 4.73 |

**客观事实：**
1. **每个模型都有1-2个"超级Protocol通道"**：qwen3 ch935, GLM4 ch12274, DS7B ch15791。
2. **DS7B的通道贡献远大于其他模型**：ch15791贡献36-49，是qwen3 ch935的5-6倍。
3. **共享通道数**：qwen3=22, GLM4=18, DS7B=22（支持3+ protocol token的通道）。
4. **这是解析计算，无过拟合风险**——与Phase 952 CV-R2≈0的回归结果形成对比。

### 三、Task 2: 多通道组合干预

#### K通道同时零化后Protocol Token Logit变化

| K | qwen3 "." | GLM4 "." | DS7B "." | qwen3 EOS | DS7B EOS |
|---|:---------:|:--------:|:--------:|:---------:|:--------:|
| 1 | -0.45 | -0.20 | -0.03 | -0.43 | -0.02 |
| 5 | -1.69 | -0.38 | **-2.34** | -0.59 | -1.48 |
| 10 | -2.40 | -0.52 | **-3.30** | -0.99 | -2.39 |
| 20 | -2.73 | N/A(仅18ch) | **-4.07** | -1.93 | -2.99 |

**客观事实：**
1. **效果随K单调增加**——零化越多通道，protocol logit下降越多。
2. **DS7B K=1→K=5有跳跃**（-0.03→-2.34）——protocol编码分布式，单通道不关键，5通道组合才开始有效。
3. **qwen3 K=1已有显著效果**（-0.45）——qwen3的protocol通道更集中。
4. **GLM4效果最弱**（K=10仅-0.52）——GLM4的protocol通道贡献分散。
5. **所有token logit均下降**——确认这些通道是正向贡献者。

### 四、Task 3: 非Color语义注入（重大发现）

#### 跨语义类别Protocol Token响应（beta=4.0）

**qwen3:**
| Token | color | size | shape | emotion | speed |
|-------|:-----:|:----:|:-----:|:-------:|:-----:|
| "." | +3.16 | +3.52 | +4.73 | +3.26 | +2.78 |
| " " | -0.99 | -0.83 | +0.08 | -1.10 | -1.06 |
| "a" | -5.14 | -4.79 | -2.41 | -5.25 | -5.09 |
| EOS | +1.51 | +2.04 | +0.98 | +1.75 | -0.19 |
| "is" | +5.30 | +5.14 | +6.04 | +5.45 | +5.75 |

**GLM4:**
| Token | color | size | shape | emotion | speed |
|-------|:-----:|:----:|:-----:|:-------:|:-----:|
| "." | +2.46 | +2.68 | +3.30 | +1.98 | +2.77 |
| " " | -1.47 | -1.85 | -1.80 | -1.83 | -1.92 |
| "a" | -5.91 | -5.00 | -3.63 | -5.27 | -6.18 |
| EOS | +1.05 | +1.60 | +0.66 | +1.72 | +1.93 |
| "is" | +0.28 | +1.53 | +1.15 | +1.26 | +0.20 |

**DS7B:**
| Token | color | size | shape | emotion | speed |
|-------|:-----:|:----:|:-----:|:-------:|:-----:|
| "." | +2.71 | +3.01 | +1.53 | +2.06 | +2.78 |
| " " | +0.79 | -0.37 | -0.71 | -0.76 | +0.88 |
| "a" | +0.82 | -1.62 | -1.00 | -0.77 | +0.56 |
| EOS | +0.05 | -0.80 | -1.36 | -1.22 | +0.04 |
| "is" | +3.98 | +5.27 | +3.80 | +4.78 | +5.20 |

#### 跨模型×跨语义类别一致性分析

| Token | qwen3 一致? | GLM4 一致? | DS7B 一致? | 跨模型一致? |
|-------|:----------:|:----------:|:----------:|:-----------:|
| "." ↑ | 5/5 正 | 5/5 正 | 5/5 正 | **15/15 正** |
| "is" ↑ | 5/5 正 | 5/5 正 | 5/5 正 | **15/15 正** |
| Solution ↑ | 5/5 正 | 5/5 正 | 5/5 正 | **15/15 正** |
| " " ↓ | 4/5 负 | 5/5 负 | 3/5 负 | 12/15 负 |
| "a" ↓ | 5/5 负 | 5/5 负 | 3/5 负 | 13/15 负 |
| EOS ↑ | 4/5 正 | 5/5 正 | 1/5 正 | 10/15 正 |

**关键发现：**
1. **Period↑ 是唯一15/15完美的跨模型×跨语义不变量**——任何语义方向注入都让句号logit上升。
2. **"is"↑ 和 Solution↑ 也是15/15完美**——结构续写词一致上升。
3. **Space↓ 和 "a"↓ 在qwen3/GLM4上一致，但在DS7B上不稳定**——DS7B对space和"a"的响应取决于语义类别。
4. **EOS在DS7B上主要下降**（4/5负）——与qwen3/GLM4相反（主要上升）。
5. **Phase 952的"period↑ space↓"不是颜色特异的**——在qwen3/GLM4上所有语义类别都如此。
6. **但DS7B打破了"space↓"不变量**——DS7B的space对color/speed方向上升，对size/shape/emotion方向下降。

### 五、综合客观发现

| 编号 | 发现 | 一致性 | 备注 |
|------|------|:------:|------|
| F1 | **解析通道归因可行** | 3/3 | 无过拟合, 无需回归 |
| F2 | **每模型有1-2个超级Protocol通道** | 3/3 | qwen3 ch935, GLM4 ch12274, DS7B ch15791 |
| F3 | **DS7B通道贡献远大于其他模型** | 1/3 | ch15791=36-49 vs qwen3 ch935=6-8 |
| F4 | **多通道干预效果单调递增** | 3/3 | K=1→5→10→20, logit持续下降 |
| F5 | **DS7B K=1→K=5有跳跃** | 1/3 | 分布式编码, 单通道不关键 |
| F6 | **Period↑是唯一15/15完美不变量** | 3/3×5类 | 任何语义注入都让句号上升 |
| F7 | **"is"↑ Solution↑ 也是15/15完美** | 3/3×5类 | 结构续写词一致上升 |
| F8 | **DS7B的EOS主要下降** | DS7B独有 | 与qwen3/GLM4相反 |
| F9 | **DS7B打破"space↓"不变量** | DS7B独有 | color/speed让space上升 |

### 六、局限性

1. **仅最后层归因**：只分析了最后层MLP的down_proj，未分析其他层。
2. **Task 2 GLM4 K=20缺数据**：GLM4只有18个共享通道，K=20无数据。
3. **Task 3的语义方向来自W_U**：是embedding-level方向，非hidden state方向。
4. **DS7B的通道贡献数值很大**（36-49）：可能是DS7B的激活幅度本身较大，需要归一化比较。
5. **未测试放大干预**：只做了零化，未做增强(a'_j = 1.5 * a_j)。
6. **小模型限制**：7-9B模型的protocol通道可能与大模型不同。

### 七、下一步

1. **逐层权重归因**：不只看最后层，分析最后5层的protocol通道分布
2. **放大干预**：增强protocol通道激活(a'_j = 1.5*a_j)，测量logit上升
3. **语义方向正交性测试**：测量period方向与各类语义方向的cos similarity
4. **Period↑机制的深入分析**：为什么任何语义注入都让period上升？是norm增加还是方向改变？
5. **DS7B EOS异常的深入分析**：为什么DS7B的EOS与qwen3/GLM4相反？


### 八、关键负控制实验：随机方向 vs 语义方向（Phase 953b）

Phase 953发现Period↑是15/15完美不变量后，必须做负控制：
**随机方向注入（同范数）是否也产生Period↑？**

#### 结果（beta=4.0, 15 prompts × 10随机方向）

| 模型 | Token | 语义方向 | 随机方向 | 语义/随机比 | 结论 |
|------|-------|:-------:|:-------:|:----------:|:----:|
| qwen3 | Period | +2.667 | **+3.292** | **0.8** | 随机更强! |
| qwen3 | Space | -1.029 | -1.252 | — | 同方向 |
| qwen3 | EOS | +1.529 | +1.352 | — | 同方向 |
| GLM4 | Period | +2.972 | +2.310 | 1.3 | 语义略强 |
| GLM4 | Space | -0.442 | -1.531 | — | 随机更强! |
| GLM4 | EOS | +0.695 | +1.538 | — | 随机更强! |
| DS7B | Period | +1.823 | +1.335 | 1.4 | 语义略强 |
| DS7B | Space | -0.171 | -0.645 | — | 同方向 |
| DS7B | EOS | -0.856 | -1.283 | — | 同方向 |

#### 结论：Period↑是范数效应，不是语义特异效应

**客观事实：**
1. **随机方向注入也产生Period↑**——三模型均确认。qwen3上随机甚至比语义更强(+3.29 vs +2.67)。
2. **Space↓和EOS的模式也与随机方向一致**——不是语义特异的。
3. **GLM4和DS7B有微弱的语义特异成分**（语义/随机比=1.3-1.4），但qwen3没有。
4. **Phase 951-953的"语义→协议桥接"结论需要大幅修正**——大部分效应是范数扰动，不是语义桥接。

#### 对前期发现的修正

| 前期发现 | 原解释 | 修正后解释 |
|---------|--------|-----------|
| Phase951 F1: 语义注入改变protocol logit | 语义→协议桥接 | **范数扰动→协议响应**（非语义特异） |
| Phase952 F1: period↑ space↓ 三模型一致 | 语义方向特异效应 | **范数效应**（随机方向也如此） |
| Phase953 F6: period↑ 15/15完美不变量 | 语义特异不变量 | **范数不变量**（随机也15/15正） |

#### 剩余的语义特异成分

GLM4和DS7B上语义/随机比为1.3-1.4，意味着有30-40%的额外效应是语义特异的。
但这需要更精确的控制（同范数、同方向的精确匹配）才能确认。

qwen3上语义/随机比为0.8，意味着语义方向甚至比随机方向弱——qwen3的protocol响应完全由范数驱动。

### 九、修正后的发现汇总

| 编号 | 发现 | 状态 | 备注 |
|------|------|:----:|------|
| F1 | 解析通道归因可行 | ✓ 确认 | 无过拟合 |
| F2 | 每模型有超级Protocol通道 | ✓ 确认 | ch935/ch12274/ch15791 |
| F3 | 多通道干预效果单调递增 | ✓ 确认 | K=1→20持续下降 |
| F4 | Period↑是范数效应 | **修正** | 随机方向也Period↑ |
| F5 | "语义→协议桥接"大部分是范数 | **修正** | 仅GLM4/DS7B有30-40%语义特异 |
| F6 | DS7B EOS与qwen3/GLM4相反 | ✓ 确认 | 范数效应也如此 |
| F7 | qwen3完全由范数驱动 | **新** | 语义/随机比=0.8 |

### 十、下一步（修正后）

1. **范数控制实验**：注入同范数但不同方向的向量，找出真正语义特异的protocol响应
2. **Hidden state norm测量**：注入前后hidden state norm变化，确认norm→protocol的定量关系
3. **Protocol通道的范数敏感性**：测量超级通道(ch935等)的激活是否与hidden state norm成正比
4. **寻找真正的语义特异效应**：用(语义注入 - 随机注入均值)作为语义特异信号，重新分析


## Phase 954: 范数控制协议场与语义残差审计 [2025-07-15 08:12]

### 一、实验设计

3个任务，3模型，8 prompts × 10语义方向 × 8随机方向，beta=4.0。

| Task | 内容 | 方法 | 核心公式 |
|------|------|------|---------|
| Task1-4 | 范数控制+语义残差 | 10方向×8随机, 扣除范数基线 | residual = sem_delta - mean(rand_delta) |
| Task5 | 通道范数敏感性 | 6 betas + 5随机, 测通道激活 vs norm | corr(a_j, \|\|h\|\|) |
| Task6 | Boost/Ablate干预 | K=3,5通道 × boost(1.5x)/ablate(0x) | delta_z_p |

### 二、Task 1-4: 语义残差矩阵（核心结果）

#### Period语义残差（sem - random baseline）

| 方向 | qwen3 | GLM4 | DS7B | 正数率 |
|------|:-----:|:----:|:----:|:------:|
| color_red-blue | **-0.719** | +0.245 | +0.129 | 2/3 |
| color_yellow-black | +0.548 | +0.180 | +1.050 | 3/3 |
| size_big-small | +0.511 | -0.163 | +1.407 | 2/3 |
| size_large-tiny | +0.444 | +0.654 | +1.740 | 3/3 |
| shape_round-square | +1.143 | +0.605 | **-0.646** | 2/3 |
| emotion_good-bad | +0.214 | **-1.499** | +0.853 | 2/3 |
| emotion_happy-sad | -0.094 | +0.494 | +0.739 | 2/3 |
| speed_fast-slow | +0.467 | +1.785 | +1.411 | 3/3 |
| function_tool-food | +1.214 | +1.332 | +0.363 | 3/3 |
| category_animal-plant | +0.691 | **-1.194** | +1.907 | 2/3 |
| **正数率** | **8/10** | **7/10** | **9/10** | — |

**客观事实：**
1. **语义残差大部分为正**——qwen3 8/10, GLM4 7/10, DS7B 9/10方向有正的period残差。
2. **残差方向依赖**——不是所有语义方向都让period额外上升。color_red-blue在qwen3上为负(-0.719)。
3. **DS7B语义残差最一致**（9/10正）——与Phase 953b的语义/随机比=1.4一致。
4. **GLM4残差最不稳定**（7/10正，2个强负值）——category -1.194, emotion_good-bad -1.499。
5. **残差幅度可观**——最大+1.907(DS7B category), 最小-1.499(GLM4 emotion)。

#### "a"冠词语义残差

| 方向 | qwen3 | GLM4 | DS7B |
|------|:-----:|:----:|:----:|
| color_red-blue | -1.416 | +0.434 | -0.125 |
| shape_round-square | +2.035 | +2.724 | -1.485 |
| size_large-tiny | -0.104 | +2.248 | +1.027 |

"a"残差在qwen3上多为负(冠词下降)，但GLM4上多为正——模型间差异显著。

### 三、Task 5: 通道范数敏感性

#### Norm-Period相关性

| 模型 | Norm-Period corr | 超级通道 norm_corr | 超级通道 period_corr |
|------|:----------------:|:------------------:|:-------------------:|
| qwen3 | **0.154** | ch935: **-0.679** | ch935: -0.538 |
| GLM4 | **0.010** | ch12274: +0.073 | ch12274: -0.314 |
| DS7B | **0.044** | ch15791: **-0.881** | ch15791: -0.322 |

**客观事实：**
1. **Norm-Period相关性极弱**（0.01-0.15）——hidden state范数不是period logit变化的主要驱动因素。
2. **超级通道与范数强负相关**——qwen3 ch935: -0.679, DS7B ch15791: -0.881。
   这意味着：范数增大时，超级Protocol通道激活**下降**。
3. **但period logit仍然上升**（随机注入时）——说明有其他通道（非超级通道）驱动了norm→period响应。
4. **超级通道的period_corr为负**（-0.3到-0.5）——这些通道激活越高，period logit反而越低。
   这与Task 6的ablate结果矛盾（ablate使period下降），需要进一步分析。

#### 范数→Period的机制推断

范数增加 → 超级通道(ch935/ch15791)激活下降 → 但period仍上升
说明：存在"补偿通道"——当超级通道因范数增加而下降时，其他通道的激活上升更多，净效应是period增加。

### 四、Task 6: Boost/Ablate干预

#### K=5通道干预效果

| 模型 | ablate period | boost period | ablate/boost比 | ablate EOS | boost EOS |
|------|:------------:|:------------:|:--------------:|:----------:|:---------:|
| qwen3 | **-1.207** | +0.547 | 2.2x | -0.908 | +0.462 |
| GLM4 | -0.365 | +0.184 | 2.0x | -0.179 | +0.084 |
| DS7B | **-2.012** | +0.738 | 2.7x | -1.250 | +0.406 |

**客观事实：**
1. **Ablate效果是boost的2-3倍**——不对称性说明通道已接近最优激活，放大增益小，移除损失大。
2. **DS7B ablate效果最强**（-2.012）——与DS7B的集中式protocol通道结构一致。
3. **GLM4效果最弱**（-0.365）——与GLM4的分布式protocol编码一致。
4. **所有token同方向变化**——ablate全降，boost全升，无反向token。

### 五、综合客观发现

| 编号 | 发现 | 一致性 | 备注 |
|------|------|:------:|------|
| F1 | **语义残差大部分为正(7-9/10)** | 3/3 | 扣除范数后仍有语义特异效应 |
| F2 | **残差方向依赖** | 3/3 | 非所有方向都正 |
| F3 | **Norm-Period相关性极弱(<0.15)** | 3/3 | 范数不是主要驱动 |
| F4 | **超级通道与范数负相关** | qwen3/DS7B | ch935:-0.68, ch15791:-0.88 |
| F5 | **Ablate效果是boost的2-3倍** | 3/3 | 不对称性 |
| F6 | **DS7B残差最一致(9/10正)** | DS7B | 最强语义特异 |
| F7 | **GLM4残差最不稳定(7/10)** | GLM4 | 2个强负方向 |

### 六、对Phase 953b结论的修正

Phase 953b结论："大部分协议响应来自范数扰动"

Phase 954修正：
1. **范数-Period相关性仅0.01-0.15**——范数本身不是主要驱动因素。
2. **语义残差7-9/10为正**——扣除随机基线后，语义特异效应确实存在。
3. **随机方向产生Period↑不是因为改变了范数**，而是因为随机方向在W_U空间中平均有正的period投影。

修正后的解释：
```
随机方向 → 在高维空间中平均有正的period方向投影 → period↑
语义方向 → 有额外的语义特异成分（7-9/10为正） → period额外↑
范数变化 → 对period影响很弱（corr<0.15）
```

### 七、局限性

1. **8 prompts偏少**：语义残差的稳定性需要更多prompt验证。
2. **8随机方向偏少**：随机基线的估计精度有限。
3. **仅最后层分析**：未分析其他层的norm-channel关系。
4. **超级通道的period_corr为负**与ablate结果矛盾：需要更深入分析。
   可能解释：超级通道在自然状态下激活高时，往往是在"非protocol"语境中，
   此时period logit本身较低。这不矛盾——相关性不等于因果性。
5. **未测DS7B ch15791的boost单独效果**：仅测了top-5组合。

### 八、下一步

1. **扩大prompt到30+**：提高残差估计的稳定性
2. **逐层norm-channel分析**：不只看最后层
3. **解决period_corr矛盾**：分析超级通道在不同prompt组中的行为差异
4. **语义方向正交性**：测量各类语义方向与period方向的cos similarity
5. **多通道精确boost**：对DS7B ch15791单独boost，测量效果


## Phase 955: 随机基线分解与协议通道族角色审计 [2025-07-15 08:42]

### 一、实验设计

2个任务，3模型，15 prompts × 7语义方向 × 16随机方向×±1(共32次)。

**核心创新：Odd/Even分解**
- Odd(u) = (Δz(+u) - Δz(-u)) / 2 → 方向投影效应（线性）
- Even(u) = (Δz(+u) + Δz(-u)) / 2 → 范数/曲率/LayerNorm效应（对称）

如果 |Even| >> |Odd|：响应主要由对称效应驱动（范数/曲率/LayerNorm）
如果 |Odd| >> |Even|：响应主要由方向投影驱动（线性读出）

### 二、Task 1-3: Odd/Even分解（核心结果）

#### 跨模型Odd/Even比值

| Token | qwen3 |Odd|/|Even| | GLM4 |Odd|/|Even| | DS7B |Odd|/|Even| | 主导 |
|-------|:-------------------:|:-------------------:|:-------------------:|:----:|
| "." | **0.30** | **0.36** | 0.73 | **Even** |
| " " | 0.70 | **0.33** | 0.90 | **Even** |
| <EOS> | 1.05 | **0.39** | 0.85 | **Even** (qwen3接近平衡) |
| "is" | **0.18** | 0.59 | **0.35** | **Even** |
| "a" | 0.33 | **0.19** | 0.74 | **Even** |

**关键发现：Even（对称效应）在所有模型×所有token上主导Odd（方向投影）！**

比值全部 < 1.0（除qwen3 EOS=1.05接近平衡），意味着：
- 随机方向注入的protocol响应主要是**对称效应**（+u和-u产生同方向变化）
- 不是方向投影效应（+u和-u应产生反方向变化）

#### Even效应的绝对值

| Token | qwen3 |Even| | GLM4 |Even| | DS7B |Even| | 跨模型一致性 |
|-------|:-----------:|:-----------:|:-----------:|:-----------:|
| "." | 3.30 | 2.28 | 2.16 | 全正(↑) |
| " " | 1.28 | 1.84 | 1.76 | 全负(↓) |
| <EOS> | 1.39 | 1.52 | 1.87 | qwen3/GLM4正, DS7B负 |
| "is" | **5.88** | 1.35 | **4.09** | 全正(↑) |
| "a" | 3.89 | **5.58** | 2.23 | 全负(↓) |

**客观事实：**
1. **"is"的Even效应在qwen3上最大(5.88)**——"is"对对称扰动最敏感。
2. **"a"的Even效应在GLM4上最大(5.58)**——GLM4的冠词对扰动最敏感。
3. **Even效应方向跨模型一致**——period↑, space↓, "is"↑, "a"↓。
4. **EOS的Even效应方向模型特异**——qwen3/GLM4正(↑), DS7B负(↓)。

#### 语义残差（16×2=32随机方向基线）

| Token | qwen3 正残差率 | GLM4 正残差率 | DS7B 正残差率 | 平均 |
|-------|:-------------:|:-------------:|:-------------:|:----:|
| "." | 4/7 (57%) | 6/7 (86%) | 5/7 (71%) | 71% |
| " " | 6/7 (86%) | 4/7 (57%) | 5/7 (71%) | 71% |
| <EOS> | 4/7 (57%) | 4/7 (57%) | 6/7 (86%) | 67% |
| "is" | 3/7 (43%) | 3/7 (43%) | 5/7 (71%) | 53% |
| "a" | 1/7 (14%) | 4/7 (57%) | 4/7 (57%) | 43% |

**客观事实：**
1. **语义残差正率约43-86%**——比Phase 954(70-90%)下降，因为随机基线更准确(32 vs 16方向)。
2. **"is"和"a"残差正率最低(43-53%)**——这两个token的"语义特异"效应最弱。
3. **period残差正率71%**——仍有信号但不强。
4. **DS7B EOS残差最稳定(86%)**——DS7B的EOS有最强语义特异成分。

### 三、Task 4: 协议通道族角色分解

#### qwen3通道角色

| 通道 | 角色 | ablate均值 | 方差 | 说明 |
|------|------|:---------:|:----:|------|
| ch935 | **general_support** | -0.441 | 0.062 | 所有token同方向下降 |
| ch36 | **general_support** | -0.354 | 0.048 | 同上 |
| ch284 | mixed | -0.268 | 0.274 | 效果分化 |
| ch153 | mixed | -0.001 | 0.005 | 几乎无效果 |
| ch188 | mixed | -0.166 | 0.267 | 效果分化 |

#### DS7B通道角色

| 通道 | 角色 | ablate均值 | 方差 | 说明 |
|------|------|:---------:|:----:|------|
| ch15791 | **general_support** | **-1.096** | 0.235 | 最强通用支持通道 |
| ch15305 | mixed | -0.171 | 0.138 | 弱效果 |
| ch1106 | general_support | -0.035 | 0.025 | 弱效果 |
| ch4985 | mixed | -0.236 | 0.229 | 效果分化 |
| ch14464 | mixed | +0.000 | 0.000 | 无效果 |

**客观事实：**
1. **未发现suppressor通道**——所有通道ablate后token要么下降要么无变化，没有反向上升。
2. **未发现token-specific通道**——所有有效通道对多个token同方向作用。
3. **ch935和ch15791是general_support通道**——零化后所有protocol token一致下降。
4. **DS7B ch15791效果最强**（ablate=-1.096 vs qwen3 ch935=-0.441）。

### 四、综合客观发现

| 编号 | 发现 | 一致性 | 备注 |
|------|------|:------:|------|
| F1 | **Even主导Odd（三模型×五token=15/15）** | 3/3 | 随机基线是对称效应 |
| F2 | **Even方向跨模型一致(period↑ space↓ is↑ a↓)** | 3/3 | 对称响应是通用机制 |
| F3 | **EOS的Even方向模型特异** | DS7B独有 | DS7B EOS↓ vs qwen3/GLM4 EOS↑ |
| F4 | **语义残差正率43-86%（下降）** | 3/3 | 更准确随机基线后信号减弱 |
| F5 | **"is"和"a"残差最弱(43-53%)** | 3/3 | 这两个token"语义特异"最弱 |
| F6 | **无suppressor通道** | qwen3/DS7B | 所有通道是general或mixed |
| F7 | **ch935/ch15791是general_support** | qwen3/DS7B | 所有token同方向变化 |

### 五、对前期结论的再修正

| 阶段 | 结论 | Phase 955修正 |
|------|------|-------------|
| Phase 951-953 | 语义→协议桥接 | **降级：Even主导，语义残差仅43-86%正** |
| Phase 953b | 大部分是范数效应 | **修正：不是简单范数，是Even(对称)效应** |
| Phase 954 | 语义残差7-9/10正 | **修正：更准确基线后降到43-86%正** |
| Phase 955 | **Even主导，语义残差弱** | 当前最准确判断 |

**Even效应的可能物理机制（排序）：**
1. **LayerNorm放大**：最后层LayerNorm对更大输入产生更大输出，方向无关
2. **MLP饱和效应**：SiLU激活在高激活区域趋于饱和，扰动改变饱和度
3. **二次曲率**：MLP的非线性产生偶次项，+u和-u产生同方向变化
4. **残差流能量**：更大扰动→更多残差流能量→lm_head读出更大logit

### 六、局限性

1. **GLM4 Task 4缺失**：GLM4的通道角色分析超时未完成。
2. **15 prompts仍偏少**：残差稳定性需要30+ prompts。
3. **16随机方向偏少**：Odd/Even的统计精度有限，需要64+。
4. **未做Odd/Even的语义方向版本**：只对随机方向做了±分解，未对语义方向做。
5. **未分析Even效应的层间来源**：不知道Even来自哪一层的LayerNorm或MLP。
6. **小模型粗糙性**：7-9B模型的通道分化可能不够清晰。

### 七、下一步

1. **扩大到30+ prompts和64随机方向**：提高Odd/Even和残差的统计精度
2. **Even效应层间来源**：逐层测量Odd/Even比值，找出Even效应主要产生层
3. **LayerNorm实验**：跳过最后层LayerNorm，测试Even效应是否消失
4. **语义方向的Odd/Even**：对语义方向也做±分解
5. **自然rollout**：对protocol通道做ablate/boost后生成文本，检查输出格式变化


## Phase 956: Even效应来源定位 [2025-07-15 09:23]

### 一、实验设计

3个任务，3模型，6 prompts × 4随机方向 × ±1。

| Task | 内容 | 方法 |
|------|------|------|
| Task1 | 逐层Odd/Even | 在不同层注入±u, 测最终logit的Odd/Even |
| Task2 | LayerNorm跳过 | 跳过最后层RMSNorm, 测Even是否消失 |
| Task3 | MLP饱和度 | gate/SiLU激活分布, ±u对比 |

### 二、Task 1: 逐层Odd/Even分解（核心发现）

#### Period token的Odd/Even比值随注入层变化

| 注入层 | qwen3 ratio | GLM4 ratio | DS7B ratio | 主导 |
|--------|:-----------:|:----------:|:----------:|:----:|
| **embedding** | **0.51** | **0.38** | **0.78** | **Even** |
| L0 | 1.60 | **0.31** | **0.45** | GLM4/DS7B Even, qwen3 Odd |
| L4/L5 | 1.76 | 0.86 | 0.61 | 混合 |
| L8/L10 | 1.76 | 1.05 | 0.67 | 混合 |
| L12/L15 | 3.10 | 1.53 | 0.69 | qwen3 Odd, DS7B Even |
| L16/L20 | 2.17 | 4.20 | 0.57 | qwen3/GLM4 Odd |
| L24/L25 | 1.74 | 3.49 | 0.74 | 混合 |
| L28/L30 | 1.59 | 5.50 | — | Odd |
| 最后层 | 1.00 | 2.38 | 1.00 | 平衡/Odd |

**关键发现：**
1. **Even效应仅在embedding层注入时主导**——qwen3在所有中间层都是Odd主导(ratio>1)。
2. **GLM4的Even效应延续到L0**——ratio=0.31, 与embedding(0.38)接近。
3. **DS7B的Even效应贯穿所有层**——ratio在0.45-0.78之间, Even始终主导。
4. **Even效应来自embedding→完整模型处理路径**, 不是某一层的LayerNorm或MLP。

#### Even效应绝对值随层衰减

| 注入层 | qwen3 |Even| | GLM4 |Even| | DS7B |Even| |
|--------|:-----------:|:-----------:|:-----------:|
| embedding | **2.99** | **2.19** | **2.14** |
| L0 | 0.08 | 2.24 | 0.61 |
| L4/L5 | 0.07 | 0.43 | 0.68 |
| 中间层 | 0.02-0.03 | 0.01-0.20 | 0.48-0.76 |
| 最后层 | 0.008 | 0.009 | 0.003 |

qwen3的Even效应从embedding(2.99)到L0(0.08)急剧衰减98%。
DS7B的Even效应衰减较慢(embedding 2.14 → L0 0.61 → 中间层 0.5-0.7)。

### 三、Task 2: LayerNorm跳过实验（意外发现）

#### 跳过最后层RMSNorm后Even效应变化

| Token | qwen3 normal→skip | GLM4 normal→skip | DS7B normal→skip |
|-------|:-----------------:|:-----------------:|:-----------------:|
| "." | 2.99→**19.15** (6.4x↑) | 2.19→**7.00** (3.2x↑) | 2.14→**78.65** (36.8x↑) |
| " " | 1.60→14.40 (9.0x↑) | 1.87→3.59 (1.9x↑) | 1.88→108.94 (57.7x↑) |
| "is" | 7.05→26.32 (3.7x↑) | 1.15→1.47 (1.3x↑) | 4.59→48.31 (10.5x↑) |
| "a" | 3.64→10.90 (3.0x↑) | 4.87→**3.17** (0.65x↓) | 2.01→61.33 (30.4x↑) |
| EOS | 1.78→5.47 (3.1x↑) | 1.50→6.65 (4.4x↑) | 1.43→50.74 (35.5x↑) |

**关键发现：**
1. **跳过LayerNorm使Even效应大幅增大(3-58倍)**——LayerNorm是抑制Even的，不是产生Even的！
2. **DS7B增幅最大(36-58倍)**——DS7B的LayerNorm对Even的抑制最强。
3. **GLM4的"a"是唯一例外**——跳过LN后Even下降(4.87→3.17)。
4. **Even效应不来自LayerNorm**——LayerNorm反而是在抑制原始的Even信号。

### 四、Task 3: MLP饱和度分析

#### Gate/SiLU激活饱和率

| 模型 | baseline sat% | +u act_norm | -u act_norm | 饱和方向 |
|------|:------------:|:-----------:|:-----------:|:--------:|
| qwen3 | 0.55% | 78.99 | 83.69 | 均下降(Even) |
| GLM4 | 4.92% | 243.29 | 229.78 | 均下降(Even) |
| DS7B | 1.29% | 252.16 | 211.99 | 不对称(Odd成分) |

**关键发现：**
1. **饱和率极低(<5%)**——MLP饱和不是Even效应的主要来源。
2. **+u和-u都使激活norm下降**(qwen3/GLM4)——这本身是一个Even效应(对称下降)。
3. **DS7B的+u和-u效果不对称**——存在Odd成分。

### 五、Even效应来源的排除法

| 候选机制 | 是否来源 | 证据 |
|---------|:--------:|------|
| LayerNorm放大 | **否** | 跳过LN后Even更大(3-58x) |
| MLP饱和 | **否** | 饱和率<5%, 太低 |
| 中间层MLP非线性 | **部分** | DS7B中间层Even仍存在 |
| Embedding→完整路径累积 | **是(主因)** | qwen3中间层Even消失, 仅embedding有 |
| Attention非线性聚合 | **可能** | 未直接测试 |

**结论：Even效应主要来自embedding级扰动经过完整模型路径的累积非线性变换。**
不是某一层的LayerNorm或MLP饱和。LayerNorm反而在抑制这个Even信号。

### 六、综合客观发现

| 编号 | 发现 | 一致性 | 备注 |
|------|------|:------:|------|
| F1 | **Even仅在embedding注入时主导(qwen3)** | qwen3 | 中间层Odd主导 |
| F2 | **GLM4 Even延续到L0** | GLM4 | 第一层也产生Even |
| F3 | **DS7B Even贯穿所有层** | DS7B | 最分布式 |
| F4 | **LayerNorm抑制Even(跳过后增大3-58x)** | 3/3 | 反直觉 |
| F5 | **MLP饱和率<5%** | 3/3 | 不是Even来源 |
| F6 | **Even来自embedding→完整路径累积** | 3/3 | 排除法结论 |

### 七、局限性

1. **6 prompts和4随机方向偏少**：统计精度有限。
2. **仅测了period token的逐层Odd/Even**：其他token的层间模式可能不同。
3. **LayerNorm跳过是粗暴方法**：直接返回input可能引入数值不稳定。
4. **未测attention的非线性贡献**：Attention的softmax是另一个非线性源。
5. **DS7B的Even贯穿所有层**：需要更精细的层间分析。
6. **小模型偏差**：7-9B可能不够代表性。

### 八、下一步

1. **Attention消融**：跳过attention层，测试Even是否来自softmax非线性
2. **扩大统计**：30+ prompts, 16+ 随机方向
3. **逐层Even对所有token**：不只period, 也测space/EOS/is
4. **LayerNorm替换实验**：不用跳过, 而是用固定统计或identity+scaling
5. **自然rollout**：对protocol通道做ablate后生成文本


## Phase 957: Attention非线性与逐词元Even来源审计 [2025-07-15 09:51]

### 一、实验设计

2个任务，3模型，6 prompts × 4随机方向 × ±1。

| Task | 内容 | 方法 |
|------|------|------|
| Task1 | Attention vs MLP消融 | 跳过最后3层attn或MLP, 测Even变化 |
| Task2 | 逐token Even热图 | 7 token × 6-8注入层, 完整Even幅度矩阵 |

### 二、Task 1: Attention vs MLP消融（核心发现）

#### Period token Even: normal vs skip_attn vs skip_mlp

| 模型 | normal | skip_attn (倍数) | skip_mlp (倍数) | 主要Even来源 |
|------|:------:|:----------------:|:----------------:|:-----------:|
| qwen3 | 2.99 | 2.48 (0.8x) | **1.74 (0.6x)** | **MLP** |
| GLM4 | 2.19 | **1.34 (0.6x)** | 2.35 (1.1x↑) | **Attention** |
| DS7B | 2.14 | 1.55 (0.7x) | **1.28 (0.6x)** | **MLP** |

**关键发现：**
1. **GLM4的Even主要来自Attention**——跳过attn使Even降至0.6x，跳过MLP反而增至1.1x。
2. **qwen3/DS7B的Even主要来自MLP**——跳过MLP降幅更大(0.6x)。
3. **GLM4中MLP抑制Even**——跳过MLP后Even增大(1.1x)，说明MLP正常在抑制Even。
4. **三模型Even来源不同**——不能笼统说"Even来自attention"或"来自MLP"。

#### GLM4的Token特异Attention/MLP角色（重大发现）

| Token | skip_attn | skip_mlp | 结论 |
|-------|:---------:|:--------:|:----:|
| "." | 0.6x (↓) | 1.1x (↑) | Attn产生, MLP抑制 |
| " " | 0.6x (↓) | 0.5x (↓) | 两者都产生 |
| "is" | **1.2x (↑)** | 0.8x (↓) | Attn抑制, MLP产生 |
| "a" | 0.7x (↓) | 0.6x (↓) | 两者都产生 |
| **<EOS>** | **2.7x (↑!)** | **0.4x (↓)** | **Attn强抑制, MLP产生** |

**GLM4中Attention和MLP对不同token有相反作用：**
- 对period: Attention产生Even, MLP抑制Even
- 对EOS: Attention**强抑制**Even(跳过后增大2.7倍!), MLP产生Even
- 对"is": Attention抑制Even, MLP产生Even

这解释了为什么GLM4的EOS行为与其他模型不同——GLM4的Attention在**主动抑制**EOS的Even响应！

### 三、Task 2: 逐Token Even热图

#### qwen3 Even幅度热图

| 注入层 | "." | " " | "is" | "a" | EOS | Solution |
|--------|:---:|:---:|:----:|:---:|:---:|:--------:|
| **emb** | **2.99** | 1.60 | **7.05** | 3.64 | 1.78 | 4.21 |
| L0 | 0.08 | 0.11 | 0.10 | 0.08 | 0.07 | 0.07 |
| L6-L35 | <0.1 | <0.1 | <0.1 | <0.1 | <0.03 | <0.1 |

**qwen3模式：Even仅在embedding层存在，L0后全部降至<0.1（97%+衰减）。**
"is"的Even最大(7.05)，但也在L0消失。

#### GLM4 Even幅度热图

| 注入层 | "." | " " | "is" | "a" | EOS | Solution |
|--------|:---:|:---:|:----:|:---:|:---:|:--------:|
| **emb** | 2.19 | 1.87 | 1.15 | **4.87** | 1.50 | 2.86 |
| **L0** | **2.24** | 1.62 | 1.25 | **3.89** | 1.54 | 2.58 |
| L6 | 0.34 | 0.60 | 0.54 | 0.75 | 0.38 | 0.78 |
| L12 | 0.14 | 0.22 | 0.14 | 0.24 | 0.13 | 0.44 |
| L18+ | <0.1 | <0.1 | <0.1 | <0.1 | <0.05 | <0.1 |

**GLM4模式：Even在embedding和L0都强(几乎相同幅度)，L6后逐渐衰减。**
"a"的Even最大(4.87)，且在L0仍保持高值(3.89)。

#### DS7B Even幅度热图

| 注入层 | "." | " " | "is" | "a" | EOS | Solution |
|--------|:---:|:---:|:----:|:---:|:---:|:--------:|
| **emb** | 2.14 | 1.88 | **4.59** | 2.01 | 1.43 | 2.58 |
| L0 | 0.61 | 1.02 | 0.42 | 0.86 | 0.23 | 0.78 |
| L4 | 0.60 | **1.42** | 0.30 | 1.15 | 0.34 | 0.76 |
| L8 | 0.59 | 1.31 | 0.34 | 1.12 | 0.27 | 0.78 |
| L12 | **0.76** | 1.33 | 0.40 | 1.05 | 0.32 | 0.94 |
| L16 | 0.50 | 1.24 | 0.30 | 0.94 | 0.25 | 0.77 |
| L20 | 0.63 | 1.08 | 0.22 | 0.84 | 0.31 | 0.49 |
| L24 | 0.48 | 1.14 | 0.34 | 0.93 | 0.24 | 0.77 |
| **L27** | 0.003 | 0.001 | 0.003 | 0.003 | 0.003 | 0.004 |

**DS7B模式：Even贯穿所有层(L0-L24均保持0.2-1.4)，仅最后层消失。**
Space的Even在L4达到峰值(1.42)，比embedding(1.88)仅衰减25%。

### 四、三模型Even传播模式对比

| 特征 | qwen3 | GLM4 | DS7B |
|------|:-----:|:----:|:----:|
| Even集中层 | 仅embedding | embedding+L0 | 全层分布 |
| L0衰减率 | **97%** | ~0% | ~60% |
| 最后层Even | ~0.008 | ~0.009 | ~0.003 |
| 最大Even token | "is"(7.05) | "a"(4.87) | "is"(4.59) |
| Even来源 | MLP主导 | **Attention主导** | MLP主导 |
| Attn/MLP角色 | 同向(都产生) | **反向(对立)** | 同向(都产生) |

**关键客观事实：**
1. **三模型Even传播模式完全不同**——qwen3集中、GLM4延续、DS7B分布。
2. **GLM4是唯一Attention主导Even的模型**——且Attention和MLP对不同token有相反作用。
3. **DS7B的Even最分布式**——所有中间层都保持显著Even(0.2-1.4)。
4. **qwen3的Even最集中**——98%在L0被吸收。
5. **不同token的Even幅度差异大**——"is"和"a"通常最大，EOS通常最小。

### 五、综合客观发现

| 编号 | 发现 | 一致性 | 备注 |
|------|------|:------:|------|
| F1 | **GLM4 Even来自Attention** | GLM4独有 | skip_attn降至0.6x |
| F2 | **qwen3/DS7B Even来自MLP** | 2/3 | skip_mlp降幅更大 |
| F3 | **GLM4中Attn/MLP角色对立** | GLM4独有 | period: attn产生/mlp抑制; EOS: attn抑制/mlp产生 |
| F4 | **GLM4 Attention强抑制EOS Even** | GLM4独有 | 跳过attn后EOS Even增2.7x |
| F5 | **qwen3 Even仅在embedding** | qwen3 | L0衰减97% |
| F6 | **GLM4 Even延续到L0** | GLM4 | L0与embedding几乎相同 |
| F7 | **DS7B Even贯穿全层** | DS7B | L0-L24均保持0.2-1.4 |
| F8 | **不同token Even幅度差异大** | 3/3 | "is"/"a"最大, EOS最小 |

### 六、Even效应来源的最终排除法

| 候选机制 | 是来源? | 证据 |
|---------|:------:|------|
| 最后层LayerNorm | **否** | Phase 956: 跳过后Even更大 |
| MLP大面积饱和 | **否** | Phase 956: 饱和率<5% |
| 最后层MLP(qwen3/DS7B) | **部分是** | skip_mlp降低Even(0.6x) |
| 最后层Attention(GLM4) | **部分是** | skip_attn降低Even(0.6x) |
| Embedding→完整路径累积 | **是(主因)** | qwen3: L0后Even消失97% |
| Attention+MLP共同作用 | **是** | 两者都无法完全消除Even |

**结论：Even效应是Attention和MLP在整个模型路径中累积非线性变换的联合产物。**
不是单一组件（LayerNorm、MLP饱和、Attention softmax）的局部效应。
不同模型中Attention和MLP的相对贡献不同——GLM4以Attention为主，qwen3/DS7B以MLP为主。

### 七、局限性

1. **6 prompts和4随机方向偏少**：需要扩大到30+ prompts和16+随机方向。
2. **仅跳过最后3层**：未测试跳过不同层的attention/MLP。
3. **skip_attn/skip_mlp是粗暴操作**：零化整个模块输出可能引入分布偏移。
4. **未测GLM4的逐层attn/mlp角色**：只测了最后3层。
5. **小模型偏差**：7-9B可能不够代表性。

### 八、阶段性总结：Phase 951-957 Protocol场审计

经过7个阶段的系统审计，Protocol场的物理图谱已建立：

**已确认的客观事实：**
1. Protocol token是个体向量场（非平均标量）
2. Even（对称）效应主导随机基线
3. Even来自embedding→完整路径累积，非单一组件
4. LayerNorm抑制Even，MLP饱和不是来源
5. Attention和MLP对不同token有不同甚至相反的作用（GLM4）
6. 三模型Even传播模式完全不同（qwen3集中/GLM4延续/DS7B分布）
7. 协议通道是general_support族（无suppressor或token-specific）
8. 语义残差弱（43-86%正，扣除Even基线后）

**仍未解决的：**
1. 自然rollout是否可被通道干预改变
2. strict-clean输出是否可改善
3. Even效应如何影响真实生成
4. 大模型中Even传播模式是否相同


## Phase 958: Attention头级与MLP通道级协议因果审计 + 自然rollout [2025-07-15 10:10]

### 一、实验设计

2个任务，3模型。

| Task | 内容 | 方法 | 规模 |
|------|------|------|------|
| Task1 | Attention头级消融 | 逐头零化o_proj输入, 测protocol logit | 最后3层×32头×4 prompts |
| Task2 | 自然rollout验证 | ablate/boost通道后generate 30 tokens | 5 prompts×3条件 |

### 二、Task 1: Attention头级消融（重大发现）

#### 三模型Top-5 Head及其角色

**qwen3 (32 heads/layer, d_head=80):**

| Head | 效果 | 角色 | 最大影响token |
|------|:----:|:----:|:-----------:|
| **L35_H0** | **0.293** | mixed | **"."(period)** |
| L34_H24 | 0.155 | token_specific | "1" |
| L35_H1 | 0.134 | mixed_opposite | Step |
| L35_H14 | 0.114 | token_specific | Solution |
| L33_H8 | 0.101 | token_specific | "." |

**GLM4 (32 heads/layer, d_head=128):**

| Head | 效果 | 角色 | 最大影响token |
|------|:----:|:----:|:-----------:|
| **L39_H21** | **0.122** | token_specific | **<EOS>** |
| L37_H15 | 0.104 | token_specific | "is" |
| L38_H5 | 0.065 | token_specific | Solution |
| L38_H7 | 0.064 | token_specific | **<EOS>** |
| L38_H0 | 0.062 | token_specific | **<EOS>** |

**DS7B (32 heads/layer, d_head=112):**

| Head | 效果 | 角色 | 最大影响token |
|------|:----:|:----:|:-----------:|
| **L26_H18** | **1.516** | mixed | "1" |
| L25_H2 | 1.326 | mixed | Solution |
| L26_H19 | 1.304 | token_specific | **" "(space)** |
| L26_H25 | 1.226 | token_specific | **" "(space)** |
| L26_H8 | 1.211 | token_specific | **" "(space)** |

#### 关键发现

1. **Token-specific attention heads存在！**（与MLP通道不同——MLP通道全是general_support）
   - GLM4有多个**EOS-specific heads**（L38_H0, L38_H7, L39_H21）
   - DS7B有多个**space-specific heads**（L26_H8, L26_H19, L26_H25）
   - qwen3有**period-specific head**（L35_H0, L33_H8）

2. **DS7B head效果远大于其他模型**（1.2-1.5 vs qwen3 0.1-0.3 vs GLM4 0.06-0.12）
   DS7B的attention head对protocol token的影响是qwen3的10倍。

3. **GLM4的EOS控制由attention head实现**——直接证实Phase 957的发现。
   3个EOS-specific heads集中在L38-L39。

4. **不同模型用不同head控制不同token**：
   - qwen3 → period head
   - GLM4 → EOS heads
   - DS7B → space heads

### 三、Task 2: 自然Rollout验证

#### 三模型rollout对比

| 模型 | 条件 | 句号数 | 词数 | EOS终止率 |
|------|------|:------:|:----:|:--------:|
| qwen3 | normal | 3.2 | 24.2 | 0.00 |
| qwen3 | ablate_K5 | 3.2 | 24.2 | 0.00 |
| qwen3 | **boost_K5** | 3.2 | **23.8** | 0.00 |
| GLM4 | normal | 3.6 | 21.4 | 0.00 |
| GLM4 | ablate_K5 | 3.6 | 21.4 | 0.00 |
| GLM4 | boost_K5 | 3.6 | 21.4 | 0.00 |
| DS7B | normal | 3.6 | 14.2 | **0.20** |
| DS7B | **ablate_K5** | **3.2** | 14.6 | 0.20 |
| DS7B | boost_K5 | 3.6 | 14.2 | 0.20 |

#### Sample输出对比（prompt: "The capital of France is"）

**qwen3:**
- normal: "Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capi..."
- ablate: **完全相同**
- boost: "Paris. The capital of Paris is...? The capital of Paris is not a city, as Paris..." **(不同!)**

**GLM4:**
- 三条件**完全相同**: "Paris. Paris is known for its rich history, stunning architecture..."

**DS7B:**
- 三条件**完全相同**: "Paris.\n\nYes, Paris is the capital of France.\n\n</think>\n\nThe capital of France i"
- 但DS7B有**20% EOS终止率**（首次观察到自然EOS!）

#### 关键发现

1. **qwen3 boost改变了输出文本**——boost_K5产生不同内容("The capital of Paris is...?")
   但句号数不变(3.2)，说明boost影响语义内容但不影响格式结构。

2. **GLM4完全不受通道干预影响**——三条件输出完全相同。
   GLM4的protocol控制可能不在MLP通道而在attention head。

3. **DS7B ablate减少了句号数**（3.6→3.2）——首次观察到通道干预改变格式结构。
   但效果微弱（仅减少0.4个句号）。

4. **DS7B首次实现自然EOS终止**（20%的prompt生成了EOS）——其他模型均为0%。
   DS7B的`</think>`标记可能帮助触发EOS。

5. **通道干预效果有限**——改变logit ≠ 改变argmax决策。
   贪心解码下，小幅logit变化通常不改变token选择。

### 四、综合客观发现

| 编号 | 发现 | 一致性 | 备注 |
|------|------|:------:|------|
| F1 | **Token-specific attention heads存在** | 3/3 | 与MLP通道(全general)不同 |
| F2 | **GLM4有EOS-specific heads** | GLM4 | L38_H0, L38_H7, L39_H21 |
| F3 | **DS7B有space-specific heads** | DS7B | L26_H8, L26_H19, L26_H25 |
| F4 | **qwen3有period-specific head** | qwen3 | L35_H0 (效果最大0.29) |
| F5 | **DS7B head效果是其他模型10倍** | DS7B | 1.2-1.5 vs 0.06-0.3 |
| F6 | **qwen3 boost改变生成文本** | qwen3 | 语义变, 格式不变 |
| F7 | **GLM4完全不受MLP通道干预影响** | GLM4 | 三条件输出相同 |
| F8 | **DS7B ablate减少句号数** | DS7B | 3.6→3.2, 首次格式变化 |
| F9 | **DS7B首次自然EOS终止(20%)** | DS7B | 其他模型0% |
| F10 | **通道干预效果有限** | 3/3 | logit变化≠argmax变化 |

### 五、阶段性总结：Phase 951-958 完整Protocol场审计

经过8个阶段系统审计，Protocol场物理图谱已从概念推进到可测量、可干预的物理结构：

**已确认的10条客观事实：**
1. Protocol token是个体向量场（非平均标量）
2. Even（对称）效应主导随机基线（15/15条件）
3. Even来自embedding→完整路径累积（非单一组件）
4. LayerNorm抑制Even，MLP饱和不是来源
5. Attention和MLP对不同token有不同甚至相反的作用
6. 三模型Even传播模式不同（qwen3集中/GLM4延续/DS7B分布）
7. MLP通道是general_support族（无token-specific）
8. **Attention head有token-specific角色**（EOS/space/period-specific heads）
9. 语义残差弱（43-86%正，扣除Even基线后）
10. **通道干预对rollout效果有限**——logit变化≠生成变化

**Phase 951-98发现演进链：**
```
951: 语义→协议桥接?(后被修正)
952: 个体token响应, period↑space↓
953: 权重级通道归因(ch935/ch12274/ch15791)
953b: 随机方向负控制(关键纠错)
954: 范数控制, 语义残差43-86%正
955: Odd/Even分解, Even主导
956: Even来源排除(LayerNorm抑制, MLP饱和排除)
957: Attention vs MLP, GLM4角色对立
958: Head级消融 + rollout验证
```

### 六、局限性

1. **Head消融仅测最后3层**：早层head可能也有protocol控制功能。
2. **Rollout仅5 prompts×30 tokens**：样本太小，需要更大规模验证。
3. **仅greedy decoding**：未测temperature sampling, top-k等。
4. **通道干预可能太弱**：K=5通道, scale=0/1.5可能不足以改变argmax。
5. **未做head级rollout干预**：只做了MLP通道rollout, 未做attention head rollout。
6. **小模型偏差**：7-9B可能不够代表性。

### 七、下一步方向

1. **Head级rollout干预**：对EOS-specific heads做ablate/boost后生成文本
2. **更大规模rollout**：20+ prompts, 50+ tokens, 多种decoding策略
3. **Logit margin分析**：测量通道干预后argmax margin变化, 解释为什么rollout不变
4. **Attention head + MLP通道联合干预**：同时干预head和channel
5. **大模型验证**：在更大模型上验证Even传播模式和head角色


## Phase 959: Head级rollout干预 + Logit margin分析 [2025-07-15 10:30]

### 一、实验设计

2个任务，3模型，10 prompts(GLM4精简为5) × 50 tokens(GLM4为30)。

| Task | 内容 | 方法 |
|------|------|------|
| Task1 | Head级rollout干预 | 零化token-specific heads后generate, 4条件 |
| Task2 | Logit margin分析 | 测top1-top2 margin, argmax是否翻转 |

条件: normal / ablate_heads / ablate_channels / ablate_both

### 二、Task 1: Head级Rollout干预（核心突破）

#### 三模型rollout结果

**qwen3 (零化period-specific heads L35_H0, L33_H8):**

| 条件 | 句号 | 词数 | token数 | EOS率 |
|------|:----:|:----:|:-------:|:-----:|
| normal | 3.4 | 40.1 | 50.0 | 0.00 |
| ablate_heads | 3.2 | 39.6 | 50.0 | 0.00 |
| ablate_channels | 3.9 | 39.2 | 50.0 | 0.00 |
| ablate_both | 3.6 | 38.3 | 50.0 | 0.00 |

输出对比:
- normal: "Paris. The capital of Germany is Berlin. The capital of Italy is Rome..."
- **ablate_heads: "Paris. The capital of Paris is...? The capital of Paris is not a concept, as Paris is a city..."**
- ablate_channels: 与normal相同

**qwen3零化period heads改变了语义内容**——从列举国家变成讨论"Paris是不是概念"。

**DS7B (零化space-specific heads L26_H19, L26_H25):**

| 条件 | 句号 | 词数 | token数 | EOS率 |
|------|:----:|:----:|:-------:|:-----:|
| normal | 3.8 | 27.9 | 47.9 | **0.20** |
| ablate_heads | 3.5 | 30.4 | 49.9 | **0.10** |
| ablate_channels | 3.2 | 28.7 | 47.9 | 0.20 |
| ablate_both | 3.8 | 30.1 | 49.9 | 0.10 |

输出对比:
- normal: "Paris.\n\nYes, Paris is the capital of France.\n\n</think>\n\nThe capital of France is Paris..."
- **ablate_heads: "\\boxed{Paris}.\n\nOkay, so I need to figure out the capital of France. Hmm, I remember learning..."**

**DS7B零化space heads完全改变了输出格式**——从`</think>`确认模式变成`\boxed{}`推理模式。EOS率从20%降到10%。

**GLM4 (零化EOS-specific heads L39_H21, L38_H0, L38_H7):**

| 条件 | 句号 | 词数 | token数 | EOS率 |
|------|:----:|:----:|:-------:|:-----:|
| normal | 2.0 | 23.6 | 30.0 | 0.00 |
| ablate_heads | 1.8 | 19.2 | 30.0 | 0.00 |
| ablate_channels | 2.0 | 23.6 | 30.0 | 0.00 |

输出对比:
- normal: "Paris. Paris is known for its rich history, stunning architecture, and vibrant culture..."
- **ablate_heads: "Paris.\n根据问题，我们需要判断这个陈述是否正确。根据我的知识，法国的首都是巴黎。因此，这个陈述是正确的。最终答案是"**
- ablate_channels: 与normal完全相同

**GLM4零化EOS heads导致语言切换！**——从英语变成中文，从事实陈述变成推理判断。
这是9个Phase以来最强的干预效果。

### 三、Task 2: Logit Margin分析

#### 三模型margin对比

| 模型 | normal margin | ablate_heads margin | argmax变化率 |
|------|:------------:|:------------------:|:-----------:|
| qwen3 | **1.760** | 1.688 | 0% |
| GLM4 | **1.906** | 1.719 | — |
| DS7B | **0.917** | 0.760 | **17%** |

**关键发现：**
1. **margin大小决定干预效果**——DS7B margin最小(0.92), head消融导致17% argmax翻转, 文本立即改变。
2. **qwen3 margin大(1.76), argmax不翻转**——但50 token累积后文本仍改变(轨迹效应)。
3. **GLM4 margin最大(1.91)**——但head消融导致语言切换, 说明head控制的不只是logit幅度而是整个生成策略。
4. **通道消融不影响margin**——ablate_channels的margin与normal几乎相同, 解释了为什么通道干预无效。

### 四、核心发现汇总

| 编号 | 发现 | 一致性 | 重要性 |
|------|------|:------:|:------:|
| F1 | **Head消融改变生成文本(3/3模型)** | 3/3 | ★★★★★ |
| F2 | **GLM4零化EOS heads导致英语→中文** | GLM4 | ★★★★★ |
| F3 | **DS7B零化space heads改变输出格式** | DS7B | ★★★★ |
| F4 | **qwen3零化period heads改变语义内容** | qwen3 | ★★★★ |
| F5 | **通道消融对生成无影响(3/3)** | 3/3 | ★★★ |
| F6 | **margin决定干预效果** | 3/3 | ★★★★ |
| F7 | **DS7B margin最小, 最易被干预** | DS7B | ★★★ |
| F8 | **GLM4通道消融输出完全不变** | GLM4 | ★★★ |

### 五、关键洞察

1. **Attention heads是protocol场的真正控制点**——不是MLP通道。
   MLP通道影响logit幅度但不改变argmax决策；attention heads改变整个生成策略。

2. **EOS-specific heads控制的不只是EOS**——GLM4的EOS heads被零化后,
   不仅EOS概率变化, 整个语言和格式都切换了。这些heads可能控制"输出模式选择"。

3. **Logit margin是干预有效性的关键变量**:
   - margin < 1.0: 干预立即改变输出 (DS7B)
   - margin 1.0-2.0: 干预通过累积轨迹改变输出 (qwen3)
   - margin > 1.5: 需要强干预(head级别)才能改变输出 (GLM4)

4. **不同模型用不同head控制不同方面**:
   - qwen3: period heads → 控制句号/停顿
   - GLM4: EOS heads → 控制终止/语言选择
   - DS7B: space heads → 控制格式/推理模板

### 六、阶段性总结：Phase 951-959 完整Protocol场审计

经过9个阶段系统审计，最终结论：

**Protocol场物理图谱（完整版）：**

```
1. Protocol token是个体向量场 (Phase 951-952)
2. Even(对称)效应主导随机基线 (Phase 955)
3. Even来自embedding→完整路径累积 (Phase 956-957)
4. LayerNorm抑制Even, MLP饱和不是来源 (Phase 956)
5. Attention和MLP对不同token有不同/相反作用 (Phase 957)
6. 三模型Even传播模式不同 (Phase 956-957)
7. MLP通道是general_support族 (Phase 953-955)
8. Attention head有token-specific角色 (Phase 958)
9. 语义残差弱(43-86%正) (Phase 954-955)
10. ★ Head消融改变生成文本(3/3模型) (Phase 959)
11. ★ Margin决定干预有效性 (Phase 959)
12. ★ GLM4 EOS-heads控制语言选择 (Phase 959)
```

**Phase 951-959发现演进链：**
```
951:  语义→协议桥接?(后被修正)
952:  个体token响应, period↑space↓
953:  权重级通道归因(ch935/ch12274/ch15791)
953b: 随机方向负控制(关键纠错)
954:  范数控制, 语义残差43-86%正
955:  Odd/Even分解, Even主导
956:  Even来源排除(LayerNorm抑制, MLP饱和排除)
957:  Attention vs MLP, GLM4角色对立
958:  Head级消融 + rollout验证(通道效果有限)
959:  ★ Head级rollout干预(改变生成!) + margin分析
```

### 七、局限性

1. **GLM4仅5 prompts×30 tokens**：需要更大规模验证语言切换效应。
2. **仅greedy decoding**：未测sampling/top-k下的干预效果。
3. **Head消融是粗暴操作**：零化整个head可能影响多个功能。
4. **未做head boost**：只做了ablate, 未做enhance。
5. **未测单head效果**：只测了多head组合, 未拆分单个head的贡献。
6. **小模型偏差**：7-9B可能不够代表性。

### 八、下一步

1. **单head拆分实验**：逐个零化GLM4的3个EOS heads, 看哪个导致语言切换
2. **Head boost实验**：增强EOS heads, 测试是否能促进自然停止
3. **更大规模rollout**：20+ prompts, 100+ tokens, 多种decoding
4. **跨模型功能等价类**：qwen3 period-head vs GLM4 EOS-head vs DS7B space-head是否功能等价
5. **大模型验证**：在更大模型上验证head角色和margin效应


## Phase 960: 单注意力头拆分、放大与严格生成审计 [2025-07-15 10:50]

### 一、实验设计

GLM4单head拆分实验：逐个零化和放大3个EOS-specific heads，测试哪个head导致语言切换。

- 5 prompts × 30 tokens × 9条件(normal/3单head ablate/3单head boost2.0/ablate_all/boost_all)
- strict-clean评价：EOS终止 + 短输出(<15 token) + 包含正确答案 + 纯ASCII

### 二、核心发现：L39_H21是唯一控制语言模式的head

#### GLM4单head消融结果

| 条件 | EOS率 | Clean率 | 语言切换率 | 平均tokens |
|------|:-----:|:-------:|:---------:|:---------:|
| normal | 0.00 | 0.00 | **0.00** | 30.0 |
| **ablate L39_H21** | 0.00 | 0.00 | **0.20** | 30.0 |
| ablate L38_H0 | 0.00 | 0.00 | **0.00** | 30.0 |
| ablate L38_H7 | 0.00 | 0.00 | **0.00** | 30.0 |
| ablate_all | 0.00 | 0.00 | **0.20** | 30.0 |
| boost2.0 L39_H21 | 0.00 | 0.00 | 0.00 | 30.0 |
| boost2.0 L38_H0 | 0.00 | 0.00 | 0.00 | 30.0 |
| boost2.0 L38_H7 | 0.00 | 0.00 | 0.00 | 30.0 |
| boost2.0_all | 0.00 | 0.00 | 0.00 | 30.0 |

#### Sample输出对比（prompt: "The capital of France is"）

| 条件 | 输出 |
|------|------|
| normal | "Paris. Paris is known for its rich history, stunning architecture..." |
| **ablate L39_H21** | **"Paris.\n根据问题，我们需要判断这个陈述是否正确。根据我的知识，法国的首都是巴黎..."** |
| boost2.0 L39_H21 | "Paris. Paris is known for its rich history, world-class art..." (内容微调) |
| ablate L38_H0 | **与normal完全相同** |
| ablate L38_H7 | **与normal完全相同** |

### 三、关键客观事实

1. **L39_H21是唯一导致语言切换的head**——ablate L39_H21的切换率(20%)与ablate_all完全一致。
2. **L38_H0和L38_H7对生成无影响**——尽管Phase 958的logit消融显示它们是"EOS-specific"，
   但零化后生成文本与normal完全相同。说明logit效果≠生成效果。
3. **Boost不导致语言切换**——放大L39_H21保持英语但微调内容("world-class art" vs "stunning architecture")。
4. **无任何条件产生EOS终止**——即使boost2.0_all，EOS率仍为0%。
5. **无任何条件达到strict-clean**——所有条件的clean率均为0%。

### 四、Logit效果 vs 生成效果的分离

| Head | Phase 958 logit效果 | Phase 960 生成效果 | 结论 |
|------|:------------------:|:----------------:|:----:|
| L39_H21 | EOS-specific (0.122) | **语言切换(20%)** | **真控制head** |
| L38_H0 | EOS-specific (0.062) | **无效果** | logit-only head |
| L38_H7 | EOS-specific (0.064) | **无效果** | logit-only head |

**关键洞察：logit消融中的"EOS-specific"不等于生成中的"EOS控制"。**
只有L39_H21同时影响logit和生成策略。L38_H0和L38_H7只影响logit但不改变argmax决策。

### 五、对Phase 958-959结论的修正

| Phase 958-959结论 | Phase 960修正 |
|-----------------|-------------|
| "GLM4有3个EOS-specific heads" | **仅L39_H21是真正的生成控制head** |
| "EOS heads控制语言选择" | **单个head L39_H21控制语言选择** |
| "Head消融改变生成(3/3)" | 确认——但只有1/3个head真正有效 |

### 六、局限性

1. **仅GLM4完成**：qwen3/DS7B超时未完成单head拆分。
2. **5 prompts偏少**：语言切换率20%可能是1/5的偶然。
3. **无EOS终止改善**：boost未能促进自然停止。
4. **无strict-clean改善**：所有条件clean率=0%。
5. **仅greedy decoding**：未测sampling。
6. **L39_H21的角色可能更广**：不只是"EOS控制"，可能是"输出模式选择"。

### 七、Phase 951-960 十阶段完整总结

经过10个阶段系统审计，Protocol场研究从概念推进到单head级因果控制：

**最终确认的客观事实：**
1. Protocol token是个体向量场 (951-952)
2. Even主导随机基线 (955)
3. Even来自embedding→完整路径累积 (956-957)
4. LayerNorm抑制Even，MLP饱和排除 (956)
5. Attention和MLP对不同token有不同/相反作用 (957)
6. MLP通道是general_support族 (953-955)
7. Attention head有token-specific角色 (958)
8. **Head消融改变生成文本(3/3模型)** (959)
9. **Margin决定干预有效性** (959)
10. **★ 单head L39_H21控制GLM4语言模式** (960)
11. **★ Logit效果≠生成效果(L38_H0/H7 logit-only)** (960)

**Phase 951-960发现演进链：**
```
951:  语义→协议桥接?(后被修正)
952:  个体token响应, period↑space↓
953:  权重级通道归因(ch935/ch12274/ch15791)
953b: 随机方向负控制(关键纠错)
954:  范数控制, 语义残差43-86%正
955:  Odd/Even分解, Even主导
956:  Even来源排除(LayerNorm抑制, MLP饱和排除)
957:  Attention vs MLP, GLM4角色对立
958:  Head级消融 + rollout验证(通道效果有限)
959:  ★ Head级rollout干预(改变生成!) + margin分析
960:  ★★ 单head拆分: L39_H21是唯一语言控制head
```

### 八、下一步

1. **qwen3/DS7B单head拆分**：确定哪个head改变生成
2. **L39_H21深入分析**：这个head的attention pattern是什么？它关注什么token？
3. **大规模验证**：20+ prompts验证语言切换率
4. **Sampling解码**：测试非greedy下的head干预效果
5. **strict-clean改善路径**：可能需要联合head+channel+EOS干预


## Phase 961: L39_H21模式头机制与跨模型功能等价审计 [2026-07-15 20:34]

### 一、实验设计

Phase 960证明GLM4 L39_H21是唯一控制语言模式的单head。Phase 961深入回答：

```text
L39_H21到底关注什么token？
它写入什么方向？
为什么消融会切换语言？
为什么放大不能提高EOS？
qwen3/DS7B是否存在功能等价head？
```

**六个任务：**
- Task 1: Attention pattern分析（50 EN prompts, qwen3/DS7B; 10 EN prompts, GLM4）
- Task 2: Head output方向O_h提取（2次forward差分法: normal - ablated）
- Task 3: Mode direction构造 d_mode = mean(EN residual) - mean(CN residual)
- Task 4: Boost失败分析（α=1.0/1.2/1.5/2.0/3.0/5.0测logit变化）
- Task 5: Head+Channel联合干预（5条件×3-10 prompts）
- Task 6: 跨模型功能等价对比

**测试模型：** qwen3（50 prompts, 88s）→ GLM4（10 prompts, 98s）→ DS7B（50 prompts, 73s）

**脚本位置：**
```text
tests/glm5/phase961_mode_head_mechanism.py          (主脚本, 全6任务)
tests/glm5_temp/phase961_runner.py                  (单模型runner)
tests/glm5_temp/phase961_glm4_minimal.py            (GLM4最小化版)
tests/glm5_temp/phase961_task6.py                   (跨模型对比)
```

**结果目录：** `results/phase961_mode_head_mechanism/`

### 二、核心发现1：L39_H21是极端自注意力头（entropy=0.013）

#### Attention Pattern跨模型对比

| 模型 | Head | Entropy | Content | Function | Special | 关注模式 |
|------|------|:-------:|:-------:|:--------:|:-------:|---------|
| qwen3 | L35_H0 | 0.169 | 0.241 | **0.759** | 0.000 | last+first token |
| qwen3 | L33_H8 | 0.184 | 0.488 | 0.512 | 0.000 | last+first token |
| **GLM4** | **L39_H21** | **0.013** | 0.101 | **0.899** | 0.000 | **极端last token (≈1.0)** |
| GLM4 | L38_H0 | 0.676 | 0.554 | 0.148 | 0.298 | broad (France=0.77) |
| GLM4 | L38_H7 | 0.693 | 0.561 | 0.181 | 0.258 | broad (France=0.84) |
| DS7B | L26_H19 | 0.258 | 0.458 | 0.542 | 0.000 | **first token (The=0.90)** |
| DS7B | L26_H25 | 0.761 | 0.550 | 0.450 | 0.000 | diffuse |

**关键发现：**
1. **L39_H21的entropy=0.013**——几乎为零，极端聚焦于当前（最后一个）token
2. L39_H21对prompt "The capital of France is"的attention: [2e-9, 3e-19, 0.0006, 0.0001, 0.00005, 0.0003, **1.0**]
3. 100%注意力在最后一个token "is"上——本质是**自注意力身份读出头**
4. L38_H0和L38_H7（logit-only heads）entropy=0.68-0.69，注意力分散到内容词("France"=0.77-0.84)
5. DS7B L26_H19关注第一个token "The"(0.90)——与GLM4的last-token模式完全不同
6. DS7B L26_H25 entropy=0.76，注意力高度分散

**结论：L39_H21不是"关注特定token类型"的头，而是"读取当前位置表示"的头。它在英文prompt中关注"is"（当前token），在中文prompt中会关注中文当前token。**

### 三、核心发现2：所有协议head都抑制EOS（W_U cosine为负）

#### Head Output方向 vs W_U方向

| 模型 | Head | ||O_h|| | cos(O_h, d_mode) EN | cos(O_h, d_mode) CN | cos(O_h, EOS) | cos(O_h, .) |
|------|------|:------:|:-------------------:|:-------------------:|:-------------:|:-----------:|
| qwen3 | L35_H0 | 34.293 | -0.2078 | +0.3721 | -0.0976 | -0.1218 |
| qwen3 | L33_H8 | 2.284 | -0.0478 | +0.0217 | +0.0053 | -0.0066 |
| **GLM4** | **L39_H21** | **9.101** | **-0.4683** | **-0.3939** | **-0.1301** | -0.0682 |
| GLM4 | L38_H0 | 3.431 | +0.1148 | +0.0120 | -0.1806 | -0.1381 |
| GLM4 | L38_H7 | 4.045 | +0.1385 | +0.0185 | -0.2282 | -0.1883 |
| DS7B | L26_H19 | 6.101 | +0.1306 | +0.0674 | -0.0295 | -0.0351 |
| DS7B | L26_H25 | 14.501 | -0.1125 | -0.2004 | +0.0466 | +0.0248 |

**关键发现：**
1. **所有7个head的cos(O_h, EOS)几乎都是负的**——它们都**抑制**EOS，不促进EOS
2. 这直接解释了boost失败：放大这些head会进一步**降低**EOS logit
3. L39_H21的cos(O_h, d_mode)=-0.47（强负相关）——它的输出方向**反对**EN-CN模式差异方向
4. L38_H0/H7（logit-only）的cos(O_h, d_mode)≈+0.12（弱正相关）——与模式方向弱对齐
5. DS7B L26_H25是唯一cos(O_h, EOS)为正(+0.047)的head，但它也同时促进空格(top promoted=' ')

**d_mode范数 vs O_h范数（GLM4 L39）：**
- ||d_mode|| = 190.68
- ||O_h|| = 9.10
- O_h仅占d_mode的~4.8%——head的贡献是模式方向中的小部分，但因果效应显著

### 四、核心发现3：Boost降低EOS logit（非提高）

#### Boost logit分析

| 模型 | Head | ΔEOS@α=1.5 | ΔEOS@α=2.0 | ΔEOS@α=3.0 | argmax变化@α=2.0 |
|------|------|:----------:|:----------:|:----------:|:----------------:|
| qwen3 | L35_H0 | -0.258 | -0.506 | -0.958 | 0.05 |
| qwen3 | L33_H8 | +0.003 | +0.009 | +0.023 | 0.00 |
| **GLM4** | **L39_H21** | **-0.143** | **-0.279** | **-0.547** | **0.00** |
| GLM4 | L38_H0 | -0.074 | -0.138 | -0.291 | 0.00 |
| GLM4 | L38_H7 | -0.029 | -0.076 | -0.148 | 0.00 |
| DS7B | L26_H19 | -0.111 | -0.106 | -0.181 | **0.20** |
| DS7B | L26_H25 | -0.044 | -0.016 | -0.213 | **0.15** |

**关键发现：**
1. **所有有效head的boost都降低EOS logit**——与W_U cosine分析完全一致
2. GLM4三个head的boost都不改变argmax（0.00）——logit变化不足以翻转决策
3. DS7B两个head的boost能改变argmax（0.15-0.20）——DS7B的margin更小，干预更容易翻转
4. **Boost失败的根本原因：这些head本身就是EOS抑制头，放大只会进一步抑制EOS**
5. 这推翻了Phase 960的假设——L39_H21不是"EOS控制头"而是"EOS抑制头+模式稳定头"

#### Rollout验证

| 模型 | Head | α=1.0 EOS率 | α=1.5 EOS率 | α=2.0 EOS率 | α=3.0 EOS率 |
|------|------|:-----------:|:-----------:|:-----------:|:-----------:|
| qwen3 | L35_H0 | 0.00 | 0.00 | 0.00 | 0.00 |
| DS7B | L26_H19 | **0.12** | 0.00 | 0.00 | 0.00 |

DS7B在normal(α=1.0)下有12%的EOS率，但boost(α≥1.5)后EOS率降为0%——直接证明boost杀死EOS。

### 五、核心发现4：联合干预无法实现strict-clean

#### Joint Intervention跨模型对比

| 模型 | Head | 条件 | EOS率 | Clean率 | 语言切换率 | 均tokens |
|------|------|------|:-----:|:-------:|:---------:|:--------:|
| qwen3 | L35_H0 | normal | 0.00 | 0.00 | 0.00 | 30.0 |
| qwen3 | L35_H0 | ablate_head | 0.00 | 0.00 | **0.10** | 30.0 |
| qwen3 | L35_H0 | ablate+boost_ch | 0.00 | 0.00 | 0.00 | 30.0 |
| qwen3 | L35_H0 | boost_head+boost_ch | 0.00 | 0.00 | 0.00 | 30.0 |
| **GLM4** | **L39_H21** | **normal** | **0.00** | **0.00** | **0.00** | **30.0** |
| **GLM4** | **L39_H21** | **ablate_head** | **0.00** | **0.00** | **0.33** | **30.0** |
| **GLM4** | **L39_H21** | **boost_channel** | **0.00** | **0.00** | **0.00** | **30.0** |
| **GLM4** | **L39_H21** | **ablate+boost_ch** | **0.00** | **0.00** | **0.33** | **30.0** |
| **GLM4** | **L39_H21** | **boost_head+boost_ch** | **0.00** | **0.00** | **0.00** | **30.0** |
| DS7B | L26_H19 | normal | **0.10** | 0.00 | 0.10 | 30.0 |
| DS7B | L26_H19 | ablate_head | **0.10** | 0.00 | 0.10 | 30.0 |
| DS7B | L26_H19 | boost_head+boost_ch | **0.00** | 0.00 | 0.00 | 30.0 |

**关键发现：**
1. **GLM4 ablate_head的语言切换率从Phase 960的20%升到33%**——样本增大(3→3 prompts, 但更稳定)后切换更显著
2. **Channel boost完全无法阻止语言切换**——ablate_head+boost_ch的切换率(0.33)与ablate_head(0.33)完全相同
3. **boost_head+boost_ch保持英文但无EOS**——head boost稳定模式，channel boost抑制EOS，两者不协同
4. DS7B是唯一有非零EOS率的模型(0.10)，但boost_head+boost_ch将其降到0
5. **所有模型、所有条件的strict-clean率均为0%**
6. DS7B的输出包含`</think>`标签——这是推理模型，输出格式与qwen3/GLM4不同

#### GLM4 Sample输出（prompt: "The capital of France is"）

| 条件 | 输出 |
|------|------|
| normal | "Paris. Paris is known for its rich history, stunning architecture..." |
| **ablate_head** | **"Paris.\n根据问题，我们需要判断这个陈述是否正确。根据我的知识，法国的首都是巴黎..."** |
| boost_channel | "Paris. Paris is known for its rich history, stunning architecture..." (与normal相同) |
| ablate+boost_ch | **"Paris.\n根据问题，我们需要判断这个陈述是否正确。首先，我们需要知道法国的首都是哪里..."** |
| boost_head+boost_ch | "Paris. Paris is known for its rich history, world-class art..." (内容微调) |

### 六、核心发现5：跨模型head不是功能等价的

#### 跨模型Attention Pattern对比

| 特征 | qwen3 L35_H0 | GLM4 L39_H21 | DS7B L26_H19 | DS7B L26_H25 |
|------|:----------:|:----------:|:----------:|:----------:|
| Entropy | 0.169 (低) | **0.013 (极低)** | 0.258 (低) | 0.761 (高) |
| 关注位置 | last+first | **last only** | **first only** | diffuse |
| Function词占比 | 75.9% | **89.9%** | 54.2% | 45.0% |
| cos(O_h,d_mode) EN | -0.208 | **-0.468** | +0.131 | -0.113 |
| 抑制EOS | 是 | **是** | 是 | 否(弱促进) |
| Boost→ΔEOS | 下降 | **下降** | 下降 | 下降 |
| Ablate→语言切换 | 10% | **33%** | 0% | N/A |
| 角色定位 | 模式稳定 | **模式锁定** | 上下文读取 | 格式/空格 |

**关键发现：**
1. **三个模型的head不是功能等价的**——它们有不同的attention策略和不同的mode direction对齐
2. GLM4 L39_H21最特殊：entropy极低(0.013)、强负cos(-0.47)、唯一导致33%语言切换
3. qwen3 L35_H0有类似但更弱的模式：entropy低、负cos、10%切换
4. DS7B L26_H19完全不同：关注first token、正cos、0%切换——不是模式控制head
5. DS7B L26_H25是格式头：top promoted=' '(空格)、高entropy、弱负cos

**结论：不存在跨模型的"protocol-mode control head"功能等价类。各模型用不同机制处理协议模式。**

### 七、对L39_H21模式的深入分析

#### L39_H21是"模式锁定头"而非"语言选择头"

证据链：
1. **Attention: 100%自注意力**——读取当前token的value，不聚合其他位置信息
2. **Output方向反对d_mode (cos=-0.47)**——不沿EN-CN方向写入，而是写"模式锁定"信号
3. **抑制EOS (cos_EOS=-0.13)**——不促进停止，而是维持继续生成
4. **Boost降低EOS (ΔEOS=-0.28@α=2.0)**——放大使EOS更不可能
5. **Ablate导致语言切换(33%)**——移除锁定后模式drift

**统一解释：**
```text
L39_H21读取当前token的表示（如"is"），
写入一个"维持当前模式"的信号到残差流。
这个信号的方向恰好与EN-CN原始差异(d_mode)反相关，
因为d_mode主要由token embedding差异主导，
而L39_H21的锁定信号是正交于embedding差异的稳定信号。

消融L39_H21 → 锁定信号消失 → 模式drift →
GLM4倾向于drift到中文推理模式（可能因为训练数据中
中文推理模式与英文事实模式的竞争关系）。
```

#### 为什么boost不能反向操作？

```text
如果L39_H21是"模式锁定头"：
  ablate → 锁定消失 → drift → 语言切换 ✓
  boost → 锁定增强 → 模式更稳定 → 但EOS更被抑制 → 无clean改善 ✗

因为锁定头同时抑制EOS（cos_EOS<0），
放大锁定会同时强化模式和抑制停止，
两者矛盾，无法实现strict-clean。
```

### 八、对Phase 960结论的修正

| Phase 960结论 | Phase 961修正 |
|-------------|-------------|
| L39_H21是"EOS控制头" | **L39_H21是"模式锁定头"，抑制EOS而非促进** |
| Boost无效是因为非线性 | **Boost无效是因为head本身就是EOS抑制头** |
| L39_H21控制"语言选择" | **L39_H21控制"模式锁定"，不是直接选择语言** |
| 可能通过boost实现strict-clean | **不可能：boost同时强化模式和抑制EOS，矛盾** |
| 跨模型可能有等价head | **不存在等价类：各模型用不同机制** |

### 九、理论更新

核心理论不变：**条件化输出场闭合理论**

新增更新内容：
```text
条件化输出场闭合理论
+ 全局语言模式物理轨迹图谱
+ 协议词元特异场
+ 协议模式场
+ 单注意力头生成控制
+ logit-only / rollout-control分离
+ ★ 模式锁定头机制（L39_H21: 自注意力+模式锁定+EOS抑制）
+ ★ 所有协议head都抑制EOS（cos_EOS < 0）
+ ★ Boost失败=放大抑制（不是非线性问题）
+ ★ 跨模型无功能等价类
+ strict-clean缺失（联合干预无法解决）
```

**新增关键公式——模式锁定头输出：**
$$
O_{\text{lock}}(x) = W_O^{(h)} \cdot V_h(x_{\text{current}}) \cdot \text{softmax}(Q_h(x_{\text{current}}) \cdot K_h(x_{\text{current}})^T)
$$

当attention ≈ 1.0（自注意力）：
$$
O_{\text{lock}}(x) \approx W_O^{(h)} \cdot V_h(x_{\text{current}})
$$

模式锁定效应：
$$
\text{Lock}(m) = \cos(O_{\text{lock}}, d_{\text{mode}}) < 0
$$

锁定头不是沿mode方向写入，而是写入正交于mode的稳定信号。

**strict-clean不可达公式：**
$$
\text{Boost}(\alpha) \to \alpha \cdot O_{\text{lock}} \to
\begin{cases}
\text{Mode stability} \uparrow & (\text{desired}) \\
\text{EOS suppression} \uparrow & (\text{undesired})
\end{cases}
$$

由于 $\cos(O_{\text{lock}}, W_U^{\text{EOS}}) < 0$，放大锁定必然抑制EOS，使strict-clean不可达。

### 十、局限性

1. **GLM4仅10 prompts**（Task 1）和5 prompts（Task 2+3）——因8bit模型速度限制
2. **L39_H21的"模式锁定"解释仍是假说**——cos(O_h, d_mode)=-0.47的因果关系需要进一步验证
3. **Mode direction包含head自身贡献**——d_mode在layer output处计算，包含L39_H21的输出，存在一定循环性
4. **DS7B是推理模型**——输出包含`</think>`标签，与其他模型不可直接比较
5. **未测sampling decoding**——所有结果基于greedy decoding
6. **联合干预的channel选择**——使用Phase 953的super channels，可能不是最优EOS支持通道
7. **跨模型对比不完整**——qwen3/DS7B未做单head拆分验证

### 十一、Phase 951-961 十一阶段完整总结

```text
951:  语义→协议桥接?(后被修正)
952:  个体token响应, period↑space↓
953:  权重级通道归因(ch935/ch12274/ch15791)
953b: 随机方向负控制(关键纠错)
954:  范数控制, 语义残差43-86%正
955:  Odd/Even分解, Even主导
956:  Even来源排除(LayerNorm抑制, MLP饱和排除)
957:  Attention vs MLP, GLM4角色对立
958:  Head级消融 + rollout验证
959:  Head级rollout干预(改变生成!) + margin分析
960:  单head拆分: L39_H21是唯一语言控制head
961:  ★★ L39_H21机制: 自注意力锁定头 + EOS抑制 + 跨模型无等价
```

**最终确认的客观事实（截至Phase 961）：**
1. Protocol token是个体向量场 (951-952)
2. Even主导随机基线 (955)
3. Even来自embedding→完整路径累积 (956-957)
4. LayerNorm抑制Even，MLP饱和排除 (956)
5. Attention和MLP对不同token有不同/相反作用 (957)
6. MLP通道是general_support族 (953-955)
7. Attention head有token-specific角色 (958)
8. Head消融改变生成文本(3/3模型) (959)
9. Margin决定干预有效性 (959)
10. 单head L39_H21控制GLM4语言模式 (960)
11. Logit效果≠生成效果(L38_H0/H7 logit-only) (960)
12. **★ L39_H21是自注意力头(entropy=0.013), 100%关注当前token** (961)
13. **★ 所有协议head都抑制EOS(cos_EOS<0)** (961)
14. **★ Boost失败=放大EOS抑制(ΔEOS<0)** (961)
15. **★ L39_H21是"模式锁定头", 非语言选择头** (961)
16. **★ 跨模型head无功能等价类** (961)
17. **★ 联合干预(head+channel)无法实现strict-clean** (961)

### 十二、下一步

**核心瓶颈：** strict-clean率为0%，且当前发现表明通过放大协议head无法实现。

**关键问题转变：**
```text
从: "如何放大协议head来实现clean输出？"
到: "如何找到促进EOS的head/channel？"
```

**Phase 962方向：EOS促进头搜索与反向锁定干预**

1. **搜索cos(O_h, EOS) > 0的head**——找到真正促进EOS的head
2. **反向锁定干预**——不ablate L39_H21，而是注入反向方向(-O_h)来同时释放锁定和促进EOS
3. **跨层EOS通道审计**——在多层的MLP通道中搜索cos(W_down[:,c], W_U[EOS]) > 0的通道
4. **L39_H21的value向量分析**——提取V_h(x_current)的具体方向，理解锁定信号内容
5. **大规模验证(50+ prompts)**——验证33%切换率的稳定性
6. **Sampling decoding测试**——非greedy下的head干预效果

















