# -*- coding: utf-8 -*-
"""技能同步（Phase 20）：给 `rdc-main-axis-probe` 追加坑 60/61，并更新条数 59 -> 61。"""
import io

P = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
s = io.open(P, encoding='utf-8').read()
n = 0

NEW = u"""60. **同名不同源的两个谱必须显式区分键空间；锚复现须限定在「全尺度网格」；探针必须保留全量配对与实例才能保住秩（Phase 20 三处同轮勘误，全部由 SMOKE/探针在正式运行前抓出）**：
    - **(a) 键空间（`E-sper`）**：上一 Phase 冻结的锚谱键名是 `w_all / w_mlp / w_attn`，而本臂自算谱的键名是**组件名** `INC_ALL / INC_MLP / INC_ATTN`。首版把两者喂进同一个 `sper()` ⇒ `KeyError: 'w_all'`。**对策**：拆成两个**显式**函数 `sper_anchor(bcomp)`（w 用冻结锚谱）与 `sper_own(cw, cb)`（w 用本臂自谱）；报告里凡出现「`spearman(w, b)`」**必须标注 w 的来源**（冻结 / 自谱），两列并列。
    - **(b) 锚复现 vs 网格（`E-scope`）**：`com_layer` 族是 **α 网格**的函数 ⇒ 在 SMOKE/PROBE 的**缩幅网格**上做「逐位复现上一 Phase 冻结锚」的断言**必然报 DRIFT**（网格不同不可比）。**对策**：`FULL_SCALE = (not SMOKE) and (not PROBE)`；`E9_anchor` 增加 `applies` 字段，缩幅下记 `applies=False` + `note`，下游一律显示 **N/A** 而**不是 FAIL**。
    - **(c) 探针必须保留全量配对与实例（`E-probefull`）**：缩配对（24→4）会让 `U_ℓ = SVD(类别质心差)` 的秩从 `n_classes−1 = 5` **退化为 2**，`FULL_SWAP` 也随之偏离锚 ⇒ 探针读数**失去口径意义**（秩与锚都不成立）。**对策**：PROBE **只缩网格与 BP**，**不缩**配对集/实例集 —— 这样 Panel [B] 的输入与 α 网格无关，探针的 `com_B` 族 / `comlayer_B_all` / `share_mlp_beh_nb` / `com_V` 可与生产**逐位互证**（本轮实测 A0 探针与 P18/P16/P17 冻结锚**逐位相同**，是性价比最高的一道提前验证）。
    - **(d) 探针件命名（附带）**：seal 冻结的 `probe_files` 路径是**权威**；驱动按主脚本落盘约定写成的名字若与之不同，下游（closeout / memo / disk_verify）会**静默读不到**。**对策**：加一个**幂等发布步骤**把记录**复制**到声明路径（**不改 seal 字节**），并 `sha256` 复核。

61. **「冻结谱重算」的量在同一模型内按构造与臂无关 ⇒ 它不能充当跨口径证据（Phase 20 实证：配对 Δ ≡ 0）**：Phase 20 落盘的 `com_V_recomputed = centroid(P17 冻结 w_all 谱)` 为与 P17/P18/P19 跨 Phase 可比而**固定用锚谱**，于是**同一模型的 nf4 与 bf16 两臂给出同一个 `com_V`**（逐位相同、配对 Δ ≡ 0）⇒ 联合判据「`Δcom_V ≤ 容差`」变成**空检查**，极易被误读成「向量质心跨精度稳健」。
    - **对策**：① 报告里把量**显式分两类** ——「**锚可比口径**」（冻结谱；只证明锚一致、不构成证据）与「**本臂自谱口径**」（`com_V_own_spectrum`；**才是**跨精度证据），并让判据写在**后者**上；② 复核脚本必须加一条**事实断言**「冻结口径两臂恒等」（`max|Δ| ≤ 1e-12`）把这种构造性锁死**记录在案**，防止后人拿它当证据；③ Ledger 的 `rev_note` 也要写同等披露（本轮加 `NOTE (E-comv)`）。
    - **副产品（跨 Phase 独立互证）**：本轮探针给出的自谱位移 **26.150 → 26.057（−0.093 层）** 与 P19 **独立**测得的 Δ`com_V` = **0.0930** 吻合 ⇒ 两个 Phase、两套实现互证。**注意**：若只报冻结口径，这个可检验性会被**白白丢掉**。
    - **通用推论**：凡复用上一 Phase 的冻结中间量作为「可比基线」，都要问一句「**这个量在本 Phase 的实验变量下会不会变**」——不变即不可作证据，必须同时落一份**随臂重算**的版本并让判据落在它上面。

"""

old_head = u'## 4 已实测的坑（59 条，逐条对应数值）'
assert s.count(old_head) == 1, 'head %d' % s.count(old_head)
s = s.replace(old_head, u'## 4 已实测的坑（61 条，逐条对应数值）'); n += 1

anchor = u'\n## 5 代码骨架要点'
assert s.count(anchor) == 1, 'anchor %d' % s.count(anchor)
s = s.replace(anchor, u'\n' + NEW + u'\n## 5 代码骨架要点'); n += 1

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('PATCHED %d spots' % n)
t = io.open(P, encoding='utf-8').read()
for pr in [u'（61 条', u'`E-sper`', u'`E-scope`', u'`E-probefull`', u'NOTE (E-comv)', u'26.150 → 26.057']:
    print('  chk %-20s -> %d' % (pr, t.count(pr)))
