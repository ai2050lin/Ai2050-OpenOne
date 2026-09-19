# -*- coding: utf-8 -*-
"""Append Phase 2864 section to AGI_GPT5_MEMO.md (append-only)."""
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

SEC = """
---

## Phase 2864 (2026-09-18) — 词级方差分解：类方差存在但微弱、集中于晚层词汇特异头，与因果前缘几乎不相交（第四轴实体化）

### 动机与协议（预注册冻结于任何观测前）
2863 三判决全 false 遗留问题：类信号是被 8 词均值稀释还是根本不存在？
2864 对 2846 drops_all 做 per-head 类间方差分解（零前向，0.5s）：
- η²[h] = SS_between/(SS_between+SS_within)（10 类 × 8 词，头级 1152）；
- R²_class = Σ_h SS_b/Σ_h SS_total（张量级）；
- null：200 次行标签随机重排（SEED=2864），max 统计做家族误差控制。
- 预注册：V1 class_variance_present iff max_h η² > null p95（max 统计）；
  V3 class_signal_in_tensor iff R²_class > null p95（2863 给不出的判决）；
  V2 描述性（η² vs mean_drop 结构、层分布、与前缘交集）。

### 结果（phase2864/class_variance/；exec ea4be587… / result aab3a157… / class_variance.npz 1e892041…）
| 判决 | 观测 | null p95 | 结果 |
|---|---|---|---|
| V1 头级类方差 | max η² = **0.393** | 0.335 | **true**（家族级显著） |
| V1 计数 | 显著头 102 | 期望假阳性 57 | 1.8×（弱-中等，未过 3× 保守线） |
| V3 张量级 | R²_class = **0.1223** | 0.1223 | **false**（边缘不超，张量无类信号） |
| V2 结构 | Spearman(η², mean_drop) = **−0.079**；η²-top20 与因果 top-64 重叠仅 **6/20** | | **类方差轴 ⊥ 因果轴** |
| 层分布 | η²-top20 集中晚层（L23-35 占 13）+ 少量早层（L5H25 0.393、L8H6） | | 晚层词汇特异头 |

η² top：L5H25(0.393)、L26H17(0.389)、L26H21(0.337)、L35H18(0.325)、L29H19(0.322)、L31H29(0.287)。

### 科学结论
1. **类信号存在但微弱且局部化**：~百个晚层头携带类间差异（V1），但张量整体组织不由类主导
   （V3，R² 12% ≈ 随机）。2863+2864 合并：**因果谱的组织单位是词/特征，不是类**；
   类只配作聚合视图。
2. **第四轴实体化**：类间方差轴（η²）与因果必要性轴（drop，ρ=−0.08）、直写份额轴（2846 C3 0.15）、
   OV 静态轴（2862 −0.026）互不预测——响应图谱四轴正交结构确立。
   η²-top20 与因果 top-64 重叠 6/20 ≈ 期望 chance 水平。
3. **晚层词汇区分器**：η² 集中 L23-35——深层头的消融损伤高度词依赖（词汇特异组件），
   早层头（L5H25/L8H6 进入 η²-top）更接近任务/结构组件。机制分层图像：
   早层=共享结构组件，晚层=词汇/特征特异组件。
4. **对战略问题的实证支撑**："2859 的 10 个 head 在其他功能维度会换成别的 head"获得直接证据——
   即使同一任务协议内，类区分头与因果头都几乎不相交；功能维度（轴）决定头子集。

### 接续（2865 候选）
- 主选（II1 收口第二刀）：L13H30 动态追踪（结构中继 vs 间接写出，~1 前向/词）——
  前缘头的类无关接口假设直接检验。
- 备选：η² 晚层词汇区分器的词级坐标化（第四轴的词坐标）。
- 备选：重启 199 词普查（G2 门线需用户决策）。
说"继续"即进入 2865 主选。
（脚本 tests/glm5/phase2864_class_variance.py；产物 phase2864/class_variance/）
"""

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_before = sum(1 for _ in f)

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SEC)

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_after = sum(1 for _ in f)

print('OK %d -> %d' % (n_before, n_after))
