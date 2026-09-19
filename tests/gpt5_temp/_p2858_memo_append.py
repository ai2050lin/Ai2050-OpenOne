# -*- coding: utf-8 -*-
import io

SECTION = """
---

## Phase 2858：G 判据重设计 + 200 词达阵——atlas_e200_legal=False，MA/Atlas-E 词表战线收束（2026-09-18）

### 原理
按 2857 结论（判据应针对扩展词表自身几何健康度），冻结 G 判据（baseline-relative 公式，数值由数据给出）：G1 ≥9/10 类满 10 新词（CAND+CAND2+CAND3 有序过滤、无剔除）；G2 separable iff max_offdiag_cos(Cm200c) ≤ max_offdiag_cos(Cm80)+0.10 且 mean ≤ mean+0.05；G3 non-degenerate iff min_c‖dW200c‖ ≥ 0.5·min_c‖dW80‖（未归一化类间对比范数）。CAND3 补池：fruit [durian, lychee, quince, plantain, mulberry]、metal [ingot, pewter, nugget, ore, foil]、tool [tongs, rasp, gouge, auger, bit]。

### 正式判决（9.8s，一次通过，脚本 `08157446`）
| 判据 | 值 | 结果 |
|---|---|---|
| G1 达阵 | [fruit 9, animal 10, metal 10, vehicle 10, country 10, food 10, nature 10, furniture 10, tool 10, clothing 10] = **199 词**（CAND3 的 durian/lychee/quince/plantain/mulberry 全灭；metal 靠 ore 补齐、tool 靠 rasp/bit 补齐） | **true** |
| G2 可分性 | offdiag max **0.4938** vs 上限 0.488（80 版 0.388+0.10，一线之差）；mean **0.3365** vs 上限 0.285（超 +0.05） | **false** |
| G3 非退化 | min‖dW‖ 0.3448 vs 0.5×0.4378=0.219，ratio **0.788** | **true** |
| **final_verdict** | | **atlas_e200_legal=False** |

### G2 失败的含义与战线收束
1. **类间中心重叠系统性上升**（max 0.39→0.49、mean 0.24→0.34）：次原型词的类边界天然更模糊（prune/date 在 fruit-food 边界、ore/nugget 在 metal-material 边界）——这是语义结构事实，不是词表构造失败。199 词版仍可用（G3 保证类方向非退化），但类条件分析的语义纯度下降。
2. **G2 判据自身缺陷登记**：baseline-relative 公式内在偏向失败——原型词表的分离度天然更高，用它作基准要求次原型词表达到同等分离度不合理。绝对阈值（如 max<0.5）事后再定则存在研究者自由度（数据已见 0.494）——不再重定阈值重跑，避免挪门柱。
3. **MA/Atlas-E 词表战线收束（2856/2857/2858 三连完整回答三问）**：①能否扩？——物理上能到 199 词，复合词/低频词大量多 token 是硬约束；②为何漂？——原型性梯度（子簇错位 0.58，非噪声非污染，2857）；③怎么判？——自身几何健康度判据（G），且 199 词版 G2 不过。**199 词普查暂缓**（判据 gating 未通过，~3.3h GPU 成本）；80 词版（2846 图谱 + 2846-2855 全链）保持唯一正式图谱。
4. 管线资产：`rdc_atlas_census.py`（`c3fcead8`）+ 199 词 e200c 清单已登记，未来若需扩展普查可直接复用。

### 文件
- 脚本 `tests/glm5/phase2858_g_criteria.py` sha `08157446`（9,585 B）
- 产物 `phase2858/g_criteria/`：execution.json `11de5c9c`、result.json `1af6bbf8`、g_criteria.npz `b2c26780`

### 接续（2859）
**图谱抽样稳定性分析（零前向，纯 2846 不可变产物）**：2846 drops_all (80,36,32) 内做前 40 vs 后 40 词分割 + 200 次随机 40 词子采样 bootstrap → 头排序 Spearman、top-10 身份分布、top-64 重合、C1/C2/C3 判据复现率——把 2856-2858 的词汇敏感性从 cdir 层面推进到图谱层面，量化 2846-2855 头级结论的词抽样置信度。
"""

path = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
with io.open(path, 'a', encoding='utf-8') as f:
    f.write(SECTION)
print('memo appended')
