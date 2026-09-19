# -*- coding: utf-8 -*-
import io

SECTION = """
---

## Phase 2859：头图谱抽样稳定性——前缘稳/尾部噪，MA 战线收官（2026-09-18）

### 原理
2856-2858 把词汇敏感性量化到 cdir 层面并收束词表扩展线；本 Phase 推进到图谱层面：2846 头图谱（1152 头排序、top-10 前缘、C1-C3 判决）对词抽样的置信度。零前向，纯 2846 不可变产物（census_full.npz drops_all (80,36,32) + 36 层 census_L{li}.npz s0/s1 重装 (80,36,32)）：臂 A 前 40 vs 后 40 词分割；臂 B 200 次 40 词无放回子采样 bootstrap（SEED=2859）。

判据（冻结，execution.json `4253960c`）：S1 atlas_stable iff 分割 Spearman ≥0.7；S2 frontedge_stable iff 2846 top-10 头 bootstrap 中位排名 ≤16；S3 c2_robust iff ≥90% 抽样 frac_top64≥0.25；S4 c3_robust iff ≥90% 抽样 Spearman(direct,drop)<0.3。

### 正式判决（0.2s，脚本 `c2ab6c40`；Gen1 变量遮蔽 bug：判据变量 s1 覆盖 s0/s1 数组名，改名 s0_all/s1_all 后一次通过）
| 判据 | 值 | 结果 |
|---|---|---|
| S1 全图谱稳定 | 分割 Spearman（1152 头）**0.112**，top-10 重合 2/10，C2 前/后 0.532/0.426 | **false** |
| S2 前缘稳定 | 2846 top-10 头 bootstrap 中位排名 **11.0**（逐头 [0,2,2,3,6,7,10,8,9,16.5]） | **true** |
| S3 C2 集中稳健 | frac64 分布 p05/p50/p95 = 0.274/0.429/0.692，≥0.25 占比 ≥90% | **true** |
| S4 正交稳健 | C3 Spearman 分布 p05/p50/p95 = 0.082/0.117/0.161，200/200 全 <0.3 | **true** |
| **final_verdict** | | **atlas_robust=False（S1 挂）/ 前缘与三判决全稳健** |

### 科学结论（图谱置信度界，方法论级）
1. **头图谱 = 稳健的前缘 + 噪声的身体**：top-64 的 load 份额结构（C2）与 top-10 头身份（S2）在词抽样下稳定（中位排名 11，逐头中位全部 ≤16.5）；但 1152 头全排序的分割相关仅 0.112、top-64 身份 Jaccard 中位 0.407——尾部头（drop ±0.005 量级）的排序被词抽样噪声淹没。
2. **2846-2855 全部头级结论的置信度确认**：形成器（L13H30 中位排名 0）/放大器带（L26-35 头）都在前缘，其结论不受词抽样影响；C3 双谱正交在 200/200 bootstrap 中无一越过 0.3——**正交性是本网络最稳健的观测量之一**。
3. **图谱规范修订（入响应谱图谱规范）**：1152 头全排序**禁止**作为"头重要性排行榜"引用；图谱的合法读出单位是 top-64（尤其 top-10）前缘 + 聚合统计量（C2/C3）。尾部仅可作分布背景。
4. 至此 MA/Atlas 战线完整收官：2846（图谱+三判决）→ 2856/2857/2858（词表扩展三问：能否/为何/怎么判，答：能到 199 词、原型性梯度、自身健康度判据且未过 → 80 词为唯一正式图谱）→ 2859（图谱置信度界：前缘稳/尾部噪/判决稳）。

### 文件
- 脚本 `tests/glm5/phase2859_atlas_stability.py` sha `c2ab6c40`（7,009 B）
- 产物 `phase2859/atlas_stability/`：execution.json `4253960c`、result.json `7645d923`、atlas_stability.npz `e3f9888d`

### 接续（2860 候选）
1. **主线回归（主选）**：MA 战线收官后回到 cdir 涌现机制链的遗留缺口——2855 接续的低优先 g_direct 对照（LN2 输出点直接注入，补独立 SwiGLU 响应臂，一次 10 层×80 词批量，~40s）可顺手关闭；随后按 MASTER_PLAN 双谱规范推进 II1/II3 或开新机制问题。
2. 备选：80 词 bootstrap 词表（类内有放回抽样）下的关键判据复验（窗口 L32-34 增益谱的词抽样置信度，量化 2854/2855 结论）——零新前向可用 2854 npz 部分实现，但 2854 未存逐词矩阵，需小规模重测。
"""

path = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
with io.open(path, 'a', encoding='utf-8') as f:
    f.write(SECTION)
print('memo appended')
