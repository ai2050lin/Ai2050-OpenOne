# -*- coding: utf-8 -*-
"""Append Phase 2863 section to AGI_GPT5_MEMO.md (append-only)."""
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

SEC = """
---

## Phase 2863 (2026-09-18) — 十类切片图谱：三判决全 false——8 词/类分辨率下"类"不是因果谱的组织单位（词级特异性主导）

### 动机与协议（预注册冻结于任何观测前）
用户图谱战略（按类推进）触发两个关键问题：类切片是否承载超出词抽样噪声的共同信号？
是否存在跨类共享前缘核心？2863 对 2846 census 张量（drops_all 80=10 类×8 词×36×32）
做 per-class 切片，**row-permutation null**（200 次，SEED=2863）对照三判决：
- J1 class_signal_above_noise：median_c Spearman(S_c, 其他 9 类均值谱) > null p95；
- J2 shared_core_real：出现于 ≥7/10 类 top-64 的头数 ≥1 且 > null p95；
- J3 formatter_role_persistent：全局形成器（120 头）在各类切片内保持 drop≥p75_c 的均值 > null p95。
零前向，0.4s。上一轮预研（_p2863_prep）已给预览（类间 offdiag mean 0.055、
重叠 13-31/64），本 Phase 加 null 后正式判决。

### 结果（phase2863/class_slices/；exec 9bfa1ade… / result 919c9f83… / class_slices.npz f8263414…）
| 判决 | 观测 | null p95 | 结果 |
|---|---|---|---|
| J1 类信号 | 0.0713 | 0.1607 | **false**（低于随机基线） |
| J2 共享核心 | 7 头 | 13.0 | **false**（真实类 top-64 重叠不超过随机预期） |
| J3 形成器保持 | 0.368 | 0.395 | **false** |
| final | **J1=False/J2=False/J3=False** | | |

描述性：核心头（虽不显著）L13H30 出现于 **9/10 类** top-64、L2H31/L4H4/L22H28/L34H15 各 8 类、
L6H12/L23H7 各 7 类；类间 offdiag mean 0.055（min −0.203 fruit×furniture，max 0.320 fruit×clothing）；
每类 top-64 与全局重叠 13-31/64。

### 科学结论（负结果，如实登记；纪律③ null 对照第 N 次修正解读）
1. **8 词/类分辨率下，"类"不是因果谱的组织单位**：类切片谱的全部结构（近正交、低自相关）
   与随机 8 词分组不可区分——per-word 机制特异性 >> 类共同性，8 词均值不足以提出类信号。
2. **上轮预研解读被正式 null 修正**："类间近正交 = 类激活独立机制组合"不成立；
   近正交只是词噪声的表现。预研无 null、正式 Phase 有 null——预注册纪律的价值再次兑现。
3. L13H30 的 9/10 类覆盖虽低于随机基线，但作为 top1 因果头（drop 0.101）的准全域出现
   与 II1 收口线一致——它是"实体档案读写接口"候选，类无关。
4. 与 2857 原型性梯度合并读：类方向（unembed 侧）是词表选择的函数；类因果谱（机制侧）
   被词噪声淹没。**类作为组织单位在 unembed 侧成立、在机制侧（8 词分辨率）不成立。**

### 对 TMA 类批次方案的修正（回应用户战略）
- 类批次推进的**前提条件 = 词分辨率提升**：8 词/类不足以检验类级机制结构。
- 2864 候选 A（零前向，优先）：词级方差分解——drops_all 的 per-word 谱做
  词间/类间/残差方差分解（ANOVA 式），定量回答"类间方差占比"，判定类信号是否存在
  但被 8 词均值稀释，还是根本不存在。
- 2864 候选 B：重启 199 词普查（2858 atlas_e200_legal=False 的 G2 门线未过；
  baseline-relative 公式缺陷已登记——是否修订 G2 需用户决策，涉及挪门柱）。
- 2864 候选 C：II1 收口第二刀（L13H30 动态追踪，前向 ~1/词）——类无关接口假设直接检验。
- 用户类别扩展（植物/器官/天体/属性轴）的排期**后置**至词分辨率问题解决——
  否则新类切片将重演 2863 的噪声主导结局。

### 接续
2864 主选 A（方差分解，零前向）→ 依结果决定 B/C。说"继续"即进入 2864。
（脚本 tests/glm5/phase2863_class_slices.py；产物 phase2863/class_slices/）
"""

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_before = sum(1 for _ in f)

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SEC)

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_after = sum(1 for _ in f)

print('OK %d -> %d' % (n_before, n_after))
