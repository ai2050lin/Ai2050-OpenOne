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
