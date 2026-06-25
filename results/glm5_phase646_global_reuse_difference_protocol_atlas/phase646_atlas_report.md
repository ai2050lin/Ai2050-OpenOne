# Phase 646 Global Reuse-Difference Protocol Atlas

本阶段不运行模型，只把 Phase 641-645 的客观结果整理为第一版协议图谱。

## Atlas Nodes

- `mechanism:value_short_answer_protocol`: Boundary-conditioned trajectory that pushes generation toward direct category value output.
- `mechanism:newline_explanation_protocol`: Protocol tendency that opens multiline reasoning/explanation rather than direct value output.
- `mechanism:non_value_answer_protocol`: Task protocol for yes/no or other non-category-value answers.

## Boundary Matrix Highlights

### qwen3

- target_failure: n=26, original exact/newline=19/0, restore exact/newline=2/22, polarity=`harmful_or_opposite_protocol`
- original_correct: n=48, original exact/newline=28/0, restore exact/newline=8/36, polarity=`harmful_or_opposite_protocol`
- inline_bad: n=1, original exact/newline=0/0, restore exact/newline=0/0, polarity=`insufficient_data`
- relation_changed: n=48, original exact/newline=11/0, restore exact/newline=6/36, polarity=`boundary_respected_or_neutral`
- explanation_needed: n=48, original exact/newline=32/0, restore exact/newline=2/46, polarity=`boundary_respected_or_neutral`
- non_value: n=48, original exact/newline=1/0, restore exact/newline=0/48, polarity=`boundary_respected_or_neutral`

### glm4

- target_failure: n=36, original exact/newline=29/0, restore exact/newline=27/0, polarity=`weak_or_neutral`
- original_correct: n=48, original exact/newline=40/0, restore exact/newline=33/0, polarity=`harmful_or_opposite_protocol`
- inline_bad: n=6, original exact/newline=3/0, restore exact/newline=3/0, polarity=`insufficient_data`
- relation_changed: n=48, original exact/newline=11/0, restore exact/newline=8/0, polarity=`boundary_respected_or_neutral`
- explanation_needed: n=48, original exact/newline=0/0, restore exact/newline=0/0, polarity=`boundary_respected_or_neutral`
- non_value: n=48, original exact/newline=0/0, restore exact/newline=0/0, polarity=`boundary_respected_or_neutral`

### deepseek7b

- target_failure: n=48, original exact/newline=12/34, restore exact/newline=45/0, polarity=`beneficial_value_protocol`
- original_correct: n=48, original exact/newline=8/36, restore exact/newline=48/0, polarity=`beneficial_value_protocol`
- inline_bad: n=1, original exact/newline=0/1, restore exact/newline=1/0, polarity=`insufficient_data`
- relation_changed: n=48, original exact/newline=4/32, restore exact/newline=17/0, polarity=`side_effect_value_absorption`
- explanation_needed: n=48, original exact/newline=0/0, restore exact/newline=43/0, polarity=`side_effect_value_absorption`
- non_value: n=48, original exact/newline=0/28, restore exact/newline=23/8, polarity=`side_effect_value_absorption`

## Trajectory Evidence Count

- trajectory_evidence_rows: 41

## Strict Interpretation

- DS7B 的 value short-answer protocol 已有生成闭环和边界副作用证据。
- qwen3 和 GLM4 不应被硬套为同一层区间、同一 separator 字符机制。
- atlas 当前是标准化索引，不是完整全局理论。下一步应把 writer graph 和更多输出类型补进节点。

## Next Phase

Phase 647 应执行 protocol writer graph audit，把 atlas 中的 value_short_answer_protocol 节点从 layer_out trajectory 继续拆到 attention / MLP / residual update writer。
