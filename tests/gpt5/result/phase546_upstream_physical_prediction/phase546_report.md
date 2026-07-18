# Phase546 生成前上游物理观察器独立确认

生成时间：2026-07-18T06:25:32.537431+00:00

## 一、为什么必须重做

Phase545（阶段545）虽有7/9个全局事件通过独立预测，但7个全部属于答案生成后的当前位置，或第0层输入。这些事件可由“答案已经出现”或“提示词本来不同”直接解释，不能登记为上游运行脉络。

本阶段在读取新隐藏状态前冻结统一修复规则：只允许提示结束时事件，并排除第0层输入；事件仍只由 Phase545（阶段545）的发现集0-23号世界对选择。确认集使用从未做过物理采集的24-72号世界对，每个模型机制单元49个独立对。由于修复规则是在看到 Phase545（阶段545）终端问题后制定，它是独立修复确认，不冒充严格预注册。

## 二、结果

| 模型 | 家族 | 机制 | 冻结上游事件 | 方向支持率 | 中位方向对齐 | 峰层误差 | 结果 |
|---|---|---|---|---:|---:|---:|---|
| glm4 | content_knowledge | category | attention_output / current / L35 | 1.000 | 0.134 | 0.000 | 通过 |
| glm4 | content_knowledge | negated_attribute | attention_output / current / L35 | 1.000 | 0.071 | 0.000 | 通过 |
| glm4 | language_action | extract | attention_output / query / L32 | 1.000 | 0.542 | 0.000 | 通过 |
| glm4 | output_protocol | json | attention_output / current / L18 | 0.429 | -0.004 | 0.000 | 失败 |
| glm4 | state_drift | entity_drift | attention_output / source / L32 | 1.000 | 0.392 | 0.000 | 通过 |
| qwen3 | content_knowledge | category | attention_output / current / L29 | 1.000 | 0.246 | 0.000 | 通过 |
| qwen3 | content_knowledge | negated_attribute | attention_output / current / L29 | 1.000 | 0.148 | 0.000 | 通过 |
| qwen3 | language_action | extract | attention_output / current / L29 | 1.000 | 0.477 | 0.000 | 通过 |
| qwen3 | output_protocol | json | mlp_output / source / L7 | 0.612 | 0.008 | 0.000 | 失败 |

上游预测通过：7/9；跨模型共享上游拓扑：2；计算边、因果路径和严格闭合仍均为0。

## 三、证据边界

冻结事件的确认量为：

$$
P^+_e=\frac1N\sum_i\mathbf 1[\cos(\Delta h_i,\Delta w_i)>0],
\qquad
e^*=\arg\max_e\operatorname{median}(\|\Delta h\|_{norm})P^+_e.
$$

即使某事件在49个新世界对上复现，两个提示中的实体和目标本来就不同，所以它仍可能是内容身份观察器。严格关系是：

$$
G_{upstream\ observer}=1
\not\Rightarrow G_{compute\ edge}=1
\not\Rightarrow G_{causal}=1.
$$

没有执行干预、头/通道/神经元扫描，也没有读取新密封集。只有跨模型拓扑复现且通过后续必要性、充分性、错层和随机同规模控制，才可升级为粗计算路径。
