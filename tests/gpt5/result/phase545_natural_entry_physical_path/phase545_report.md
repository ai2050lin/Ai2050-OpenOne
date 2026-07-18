# Phase545 自然行为入口的全层多位置物理轨迹

生成时间：2026-07-18T06:25:32.277850+00:00

## 一、执行范围

Phase544（阶段544）行为合格的9个“模型×机制”单元进入本阶段。Qwen3（通义千问3）采集192个世界对，GLM4（智谱清言4）采集240个世界对；DS7B（深度求索7B）因行为门0/18，按预注册规则跳过模型加载。

每个世界对只保留以下聚合量：

$$
\Delta_{l,c,r,t}=
\frac{\|S^A_{l,c,r,t}-S^B_{l,c,r,t}\|}
{(\|S^A_{l,c,r,t}\|+\|S^B_{l,c,r,t}\|)/2}.
$$

同时记录世界差分与答案分叉方向的夹角、两世界余弦、组件守恒和生成前缀复现。完整隐藏向量在内存中比较后立即丢弃；没有头、通道或神经元扫描。

## 二、独立预测结果

| 模型 | 家族 | 机制 | 发现事件 | 确认方向支持率 | 峰层相对误差 | 结果 |
|---|---|---|---|---:|---:|---|
| glm4 | content_knowledge | category | after_third_generated_token / attention_output / current / L33 | 0.958 | 0.000 | 通过 |
| glm4 | content_knowledge | negated_attribute | after_third_generated_token / mlp_output / current / L25 | 1.000 | 0.000 | 失败 |
| glm4 | language_action | extract | after_third_generated_token / attention_output / current / L1 | 1.000 | 0.000 | 通过 |
| glm4 | output_protocol | json | prompt_end / layer_input / source / L0 | 0.833 | 0.000 | 通过 |
| glm4 | state_drift | entity_drift | after_first_generated_token / attention_output / current / L32 | 1.000 | 0.000 | 通过 |
| qwen3 | content_knowledge | category | after_first_generated_token / layer_input / current / L0 | 1.000 | 0.000 | 通过 |
| qwen3 | content_knowledge | negated_attribute | after_first_generated_token / layer_input / current / L0 | 1.000 | 0.000 | 通过 |
| qwen3 | language_action | extract | after_first_generated_token / layer_input / current / L0 | 1.000 | 0.000 | 通过 |
| qwen3 | output_protocol | json | after_third_generated_token / attention_output / current / L29 | 0.542 | 0.000 | 失败 |

同模型物理预测通过：7/9；其中生成前且排除第0层输入的上游候选：0/9；跨Qwen3与GLM4事件轴和相对深度同时同构：0个机制。

必须严格区分：

$$
G_{\mathrm{physical\ prediction}}=1
\not\Rightarrow
G_{\mathrm{compute\ edge}}=1
\not\Rightarrow
G_{\mathrm{causal}}=1.
$$

世界A与世界B在提示里本来就包含不同的目标内容，因此稳定差分可能是词汇/字段搬运，也可能是任务操作；它不能单独证明抽象类别、知识边或格式算子。跨模型“同构”只登记事件拓扑，不登记共享机制。

## 三、全局形状

本阶段能回答的是：在自然行为稳定的显式来源读取与格式输出任务上，世界差分首先在哪个运行时刻、组件、角色和深度形成可复现峰值。当前全局最高峰主要落在答案词元已经生成后的当前位置，或第0层来源输入，因此优先解释为终端身份事件。它不能回答：该峰值是否必要、充分、负责来源运输，或由哪些神经元实现。

严格闭合保持0/72。全局物理图谱是否提高，只按独立预测通过且客户端发布后的实际覆盖小幅调整；行为失败的推理、语法、跨语言和闭合族仍为空白，不能由这批显式读取入口外推。

## 四、下一门

只有物理预测通过的单元可进入粗路径干预：冻结来源角色、事件组件和连续层窗口，先做必要性、充分性、随机同规模与错层控制，再测试中介恢复。计算边未通过前，继续禁止头、通道和单神经元扫描。新密封集仍未读取。
