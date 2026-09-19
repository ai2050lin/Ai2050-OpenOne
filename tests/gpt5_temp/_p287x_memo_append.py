# -*- coding: utf-8 -*-
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = """

## Phase 2875+2876+2877：属性轴通道分化三连——causal 通道阴性、mlp 通道阳性

**日期**：2026-09-18。**脚本**：`phase2875_attr_census_v2.py`（48.5 min 前向）、
`phase2876_growth_axis2_v2.py`（15.0s 零前向）、`phase2877_attr_mlp_spectrum.py`
（15.3s，126 前向）。产物：`phase2875/attr_census_v2/` {exec `9b72c85d`,
result `182d0094`, npz `aff07bb2`}；`phase2876/growth_axis2_v2/` {exec `e2b48133`,
result `3b3770bd`, npz `1b2d3faf`}；`phase2877/attr_mlp_spectrum/` {exec `0e4c799d`,
result `5c42ab92`, npz `8fc6e0cb`}。

### 2875：42 词全谱 census（2874 法定词表，钳制校验 max_cr≤0.0097）

| 判决 | 观测 | 结果 |
|---|---|---|
| **X1** 前缘存在性 | ≥6/8 轴重叠头数 = **0**（max 5） | **attr_frontedge_absent**（2872 复制，更强） |
| **X2** 机制侧分离 | ρ(attr, class) = **−0.0031**（几乎精确零） | **mechanism_side_separate** |
| **X3** 头群独立 | 交集 4，p = 0.482 | **independent_populations** |
| **X5** 词级检索 | acc 0.1667 < null p95 0.1905 | **axis_signal_absent**（42 词仍阴性） |

### 2876：增长率第二点重测（镜像 null）

sig2 仅 **1 头**（p<0.01）；D1=not_replicated / D2=axis_specific（0/1 交集）/
D3=novel_dominant（class43 迁移 0.1667 = full）/ D4=axis_signal_absent。
**功效解释弱化**：28 词（2873）与 42 词（2876）双双阴性 + 前缘零 +
谱相关零 → "头级 drop 通道不按属性轴组织"上升为候选结论。

### 2877：属性轴 mlp 响应谱（B3_attr，42×10）

Gen1 两教训：①`final_verdict` 自引用 res（UnboundLocalError，崩溃于写盘前，
E1/E2 未被观测）；②hook-vs-recompute 一致性门线错位——重算 1-D 形状 vs
前向 2-token 批形状，cuBLAS 按形状选算法致 bf16 归约序差异 ~2e-3；
**g 用 mlp_call 同形状同上下文算 ref 与 perturbed，测量自洽不受影响**。
Gen2 修正一致性检验为重算确定性（v1a：同输入两次重算，门 1e-6）。

| 判决 | 观测 | 结果 |
|---|---|---|
| **v1a** 重算确定性 | max rel err = **0.000e+00** | **true**（kernel 上下文恒定） |
| kernel 上下文噪声 | 1.98e-3（描述性登记） | — |
| **E1** 轴检索 | acc = **0.5000** vs null p95 0.1917（2.6×） | **mlp_carries_attr_axis** |
| **E2** 同轴余弦边际 | **0.1158** vs null p95 0.0563（2.1×） | **attr_margin_in_mlp** |

### 核心结论（重复三遍）

**通道分化（channel dissociation）：属性轴信号不在头级因果 drop 谱（2875/2876
双阴性，前缘不存在，谱相关 −0.003），但在 mlp 响应谱 10 维中显著存在（检索
0.5，2.6× null）——"mlp 臂是高密度机制载体"的定律从类轴跨轴复制到属性轴；
两轴的载体差异在通道维度：类轴 = drop 谱有组织 + mlp 载体（0.875），属性轴 =
drop 谱无组织 + mlp 载体（0.5）。机制基增长率曲线第二点（mlp 通道）：
acc 0 → 0.5，新组件 = 0 个新 head 集合（同一 36×32 布局内换通道读出）。**

### 方法论入账

- **重算类测量的正确一致性检验 = 重算确定性（同形状）**，不是 hook-vs-recompute
  （跨 kernel 上下文）；凡 hook 捕获 + 重算混合的协议必须区分两者。
- GPU cuBLAS 形状相关算法选择 → bf16 跨形状差异 ~2e-3 是常态，写入协议预算。

### 接续

2878 候选：A（主选）B3_attr 扩容协议推广——语法轴/翻译轴的 mlp 载体检验
（每轴 ~15s）；B 类轴+属性轴联合词坐标（B3_class ∪ B3_attr 密度门控融合）；
C 2872/2875 双词表 census 的 per-axis 对齐分析（同轴跨词表再现性）。
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
print('APPENDED')
