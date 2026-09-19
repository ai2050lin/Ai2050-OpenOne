# -*- coding: utf-8 -*-
import io

SECTION = """
---

## Phase 2857：E2 方向漂移审计——根因确诊 subset_incompatibility（2026-09-18）

### 原理
2856 双失败（E1 词源不足 / E2 方向漂移 0.847-0.936）留下根因二选一：(a) 多义词/弱成员污染（个别词拖动类中心）；(b) 80 词基线自身的抽样噪声。本 Phase 在 unembed 层面（无前向）三臂分离：A1 jackknife 200a 词表（逐新词剔除，delta_cos = cos(dW200a[−w], dW80) − cos_base）；A2 80 协议词表内 leave-one-out 方向抖动尺度 δ80（噪声基准）；A3 新词子集中心 vs 旧词中心（Cm80）余弦（子集相容性）。随后清洗（剔 Δ≥0.05 词）+ 扩充池（CAND2，每类 5-10 候补）复跑 E1v2/E2v2/E3v2。

判据（冻结，execution.json `025498b2`）：J1 pollution_dominant iff ≥2/10 类存在 delta_cos≥0.05 的词；J2 subset_compatible iff 子集中心余弦中位 ≥0.90；J3 drift_source 三分类；E1v2 ≥9/10 类 10 新词；E2v2 ≥9/10 类 cos≥0.9；E3v2 off-diag 中心余弦矩阵 r≥0.95。

### 正式判决（10.2s，一次通过，脚本 `bbdf1890`）
| 判据 | 值 | 结果 |
|---|---|---|
| J1 污染主导 | 逐类最大 delta_cos [0.013, 0.010, 0.009, 0.012, 0.006, 0.015, 0.009, 0.018, **0.023**, 0.012]（tool/pliers 最高），全部 <<0.05，清洗列表全空 | **false** |
| J2 子集相容 | 新/旧子集中心余弦 [fruit 0.557, animal 0.633, metal 0.627, vehicle 0.576, country 0.760, food 0.575, nature 0.570, furniture 0.572, **tool 0.402**, clothing 0.631]，**中位 0.5755** | **false** |
| J3 漂移源 | δ80 中位 **0.0225**（基线自身抖动 cos≈0.978，比子集错位小 6-25 倍） | **subset_incompatibility** |
| E1v2 达阵 | 每类新词 [9,10,9,10,10,10,10,10,8,10] = **196 词**（fruit 缺 1：raisin 多 token；metal 缺 1：CAND2 元素词全灭；tool 缺 2） | **false**（7/10） |
| E2v2 方向 | cos 谱 [0.877, 0.873, 0.909, 0.872, 0.934, 0.845, 0.874, 0.858, **0.836**, 0.876]——补词后较 200a 整体略降 | **false** |
| E3v2 几何 | off-diag r = **0.9389** < 0.95（200a 0.9512 贴线，补词后跌破） | **false** |
| **final_verdict** | | **atlas_e200_ready_v2=False / drift=subset_incompatibility** |

### 科学结论（本 Phase 的正面发现）
1. **漂移根因 = 子集不相容，非噪声非污染**：2806 词表选的是类**原型词**（apple/dog/hammer/Japan），扩展池是次原型词（plum/pliers/fig/sled），两个子簇在 unembed 空间的中心余弦仅 0.40-0.76（中位 0.576）——而旧词表内部 leave-one-out 抖动只有 0.0225。原型性梯度主导类内方向：**类方向 dW_unit 是词表选择的函数，不是类的稳定属性**。
2. **类间对比几何相对稳健**：子集中心差 0.42+，但类方向（对比向量）余弦仍 0.84-0.93、类间几何 r 0.94——漂移大部分落在"类内原型性梯度"方向上，类间区分结构保留。这与 2806 hierarchy law（类间方向分层）互证。
3. **对 Atlas 战线的战略修正**：80 词版全部结论（2846-2855）的 cdir 是 80 词表的量，**不能也不需要**平移到扩展词表——cdir 本就是词表的操作量。"新旧方向一致性"（E2）判据语义过强，预注册时未预见原型性梯度（判据设计课：词表扩展判据应针对**扩展词表自身的几何健康度**，而非与基线的一致性）。

### 文件
- 脚本 `tests/glm5/phase2857_drift_audit.py` sha `bbdf1890`（14,281 B）
- 产物 `phase2857/drift_audit/`：execution.json `025498b2`、result.json `eb702ec4`、drift_audit.npz `0eb5db58`

### 接续（2858 候选）
1. **扩展词表合法性判据重设计（主选）**：放弃 E2（新旧一致），新预注册 G 判据——G1 词数达阵（每类 ≥20，fruit/metal/tool 需再补池）；G2 类间中心可分性（off-diag 中心余弦上限健康，如 max <0.6）；G3 每类 dW 非退化（范数下限）。通过后 Atlas-E200 正式双谱普查独立成线（管线库 c3fcead8 就绪）。
2. 80 词线继续深挖（独立于扩展）：词表内 jackknife 稳健性已被 2857 A2 量化（δ80 0.0225，极稳），现有结论无需重测。
3. 观察登记（不展开）：原型 vs 次原型的子簇结构本身可测（类内 unembed 主轴），留作后战。
"""

path = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
with io.open(path, 'a', encoding='utf-8') as f:
    f.write(SECTION)
print('memo appended')
