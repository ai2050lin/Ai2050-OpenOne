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
