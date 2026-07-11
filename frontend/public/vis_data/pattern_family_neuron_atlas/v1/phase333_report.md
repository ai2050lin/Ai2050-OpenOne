# Phase333 动态时序路径、连续残差块与补偿图谱

## 固定执行

- 注册自然案例：648。
- 逐词元读出：35772。
- 五组件动态路径：6310960。
- 注册留出交换：108；条件生成：972。
- 动态响应：134784。

## 七门

- dynamic_sequence_stable: False
- state_block_effective: False
- competition_consistent: False
- compensation_explained: False
- free_generation_improved: False
- matched_controls_clean: False
- cross_model: True
- full_gate_pass: False

## 边界

连续残差输出块是功能时间对齐的组件级干预，不是单神经元干预。
补偿边只表示干预后按层滞后恢复或持续的候选关系，不是已闭合因果边。
无效条件指标按失败关闭：116（缺失 116，非有限 0）。
