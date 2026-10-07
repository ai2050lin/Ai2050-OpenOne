# -*- coding: utf-8 -*-
"""Phase 13 技能同步：rdc-main-axis-probe（13->14 臂 / 44->46 坑）与
rdc-phase-closeout（12->14 条教训 / 收尾链次数）。

铁律 (o)：逐处 assert count==1 + 落盘回读复核。
"""
import io
import os

P1 = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
P2 = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'

o = []
def w(s=''):
    o.append(str(s)); print(s)


def patch(path, pairs):
    t = io.open(path, encoding='utf-8').read()
    for name, old, new in pairs:
        c = t.count(old)
        assert c == 1, '[%s] anchor %r count=%d (须为 1)' % (os.path.basename(path), name, c)
        t = t.replace(old, new)
        w('  OK %-28s (count==1 -> replaced)' % name)
    io.open(path, 'w', encoding='utf-8').write(t)
    return t


# ======================= 1. rdc-main-axis-probe =======================
w('=== rdc-main-axis-probe ===')
ARM6 = (
    '| **N2h1-α-6 位点间配对 bootstrap**（Phase 13） | 「哪些相邻位点**真的可分辨**」；集中度是**坐标系依赖**还是抽样噪声 | '
    '**零额外前向的纯再分析**：上游 Phase 已把逐对矩阵落盘（`E2_pairs` 18 位点 × 14 α × 24 对、`E6_pairs` 14×24、`E5_pairs` 4×4×17），'
    '且**所有位点/α 的 `order` 完全一致** ⇒ 直接重建 `PM_swap/PM_R/PM_conf`；分母 `FS_VEC` 由以**受体词**为键的 `FULL_SWAP_pairs` 按 `order` 取回'
    '（重建后 `mean(FS_VEC)` 必须**逐位等于**上一 Phase 的 `FULL_SWAP`）；**重放 RNG 流**（`default_rng(seed)` 的首个消费点就是主循环 `integers`，'
    '消费顺序 = 主循环 BS → 置换 BP → 确认集 BS）⇒ 逐位复现上一 Phase 的**全部带 + 2000 值置换零假设**。'
    '主量 = **相邻配对差** `Δ_b(ℓ_i) = F_b(ℓ_i) − F_b(ℓ_{i+1})`（同一 `idx_b`，F ∈ {J, xhalf}），'
    '标签 `DECISIVE_DOWN (hi<0)` / `DECISIVE_UP (lo>0)` / `TIE (lo≤0≤hi)`；'
    '**紧化归因** 硬断言 `cov>0 ⇒ sd_paired < sd_indep`（F17 方差分解代数恒等 `var_indep − var_paired == 2cov`）；'
    '**集中度改成尾部概率 + 窗口定位**：`P(share≥0.60) / P(≤0.40) / P(mid)` + **argmax 窗口索引的 bootstrap 频次直方图**，'
    '**对 xhalf 与 J 两个坐标各做一份**，判据 `coord_dep := |share_x|≥0.40 ∧ |share_j|≥0.40 ∧ |W_x − W_j|≥3`；'
    '另加 α 网格留一（12 变体）与陡度分母改 IQR 的稳健性臂 |'
)
GATE6 = (
    '**N2h1-α-6 门（qwen3-4b 实测，Phase 13）**：**G0p** 前置 = 装置锚 A0（逐位复现上一 Phase 全部带 + 两个 2000 值置换零假设）`max|d| == 0` '
    '∧ A0b（确认集带）`max|d| < 1e-12`；**D_discriminability** = 配对口径可分辨对数 `N_dec_J / N_dec_X`（对照独立区间口径 `N_dec_indep_J`）；'
    '**coord_dep_rule** = `|share_x|≥0.40 ∧ |share_j|≥0.40 ∧ |W_x − W_j|≥3`。裁决表：G0p fail → `DEVICE_ANCHOR_FAILED`；'
    '`N_dec_J == 0 ∧ N_dec_X == 0` → `PAIRED_TEST_UNINFORMATIVE`；coord_dep → `CONCENTRATION_COORDINATE_DEPENDENT`；'
    '`P_few_x ≥ 0.95 ∧ P_few_j ≥ 0.95` → `CONCENTRATION_FEW_LAYER_ROBUST`；`P_acc_x ≥ 0.95 ∧ P_acc_j ≥ 0.95` → `CONCENTRATION_ACCUMULATE_ROBUST`；'
    '其余 → `CONCENTRATION_UNDECIDED`。'
    '**实测**：A0 全部 `max|d| = 0.000e+00`（`J_ci` 18×3 / `top3_share_x` / `top3_share_recover` / `rho_recover` / `rho_xhalf` / `R_ci` / '
    '**`perm_x` 2000 值 / `perm_rec` 2000 值**）⇒ **BIT-EXACT**，且**零前向纯 CPU 2.11 s**；'
    '`N_dec_J = 10/17` 而独立区间口径只 **2/17**（`cov>0` 达 **17/17、反例 NONE**，`rho_pair ∈ [0.347, 0.894]`，**收紧比中位数 0.5475**）；`N_dec_X = 4/17`；'
    '**坐标系依赖**：`xhalf` `top3_share = 0.5745`（`P(≥0.60) = 0.379`，**argmax 窗口 = w14 = L28→L34**，freq **0.7505**）'
    'vs `J_swap` `top3_share = 0.7953`（`P(≥0.60) = 0.973`，**argmax 窗口 = w1 = L7→L10**，freq **0.739**），**两窗口相距 13** '
    '⇒ 冻结判据「少数几层承载 ≥ 60% 的 spread」在 `J` 坐标**成立**、在 `xhalf` 坐标**不成立** ⇒ **`CONCENTRATION_COORDINATE_DEPENDENT`**；'
    '**深尾反转被决断**：`L32→L34` 的 `xhalf = −0.064481`（带 `[−0.078529, −0.051705]` 排除 0）而 `L30→L32 / L28→L30` 皆 TIE ⇒ 「L30 谷」是两个不可分辨小步的叠加；'
    '确认集唯一可分辨对 `L20→L34` 与发现集**同号** ⇒ P12 `G4` 反号是**采样支不同**；'
    '**A8** 剔除 `α = 0.4` 令 `XH_RANGE` 从 0.1094 跌到 **0.1007 —— 仍高于 0.10 阈值，但 G0 裕度由 9.39% 压到 0.73%**；**A9** 分母改 IQR 后 `N_dec_J_alt = 5/17` ⇒ **`N_dec` 是统计量依赖的**。'
)
PIT45 = (
    '45. **「不可排序 / 不可分辨」是区间口径的函数，不是数据的性质（Phase 13 的判决性一条）**：'
    '边际区间**不重叠是充分条件**，其逆否——「重叠 ⇒ 不可分辨」——**不成立**。'
    'Phase 11 报「相邻位点 J 的 95% 区间 **16/17 重叠**」并据此写「位点 J 不可排序」；Phase 13 用**配对差**口径'
    '（同一重采样下 `Δ_b(ℓ_i) = J_b(ℓ_i) − J_b(ℓ_{i+1})`，`Var(Δ) = Var_i + Var_{i+1} − 2·Cov`）重算 ⇒ **10/17 可分辨**，'
    '且 `Cov > 0` 达 **17/17、无反例**（`Cov>0 ⇒ sd_paired < sd_indep` 可写成硬断言），收紧比中位数 **0.5475**。'
    '**对策**：凡出现「X/Y 重叠」的表述，必须附**配对口径的对照数字**；配对差口径（同 `idx_b`）在下游全部指标上优先。'
    '**并且**：`N_dec` 本身是**统计量依赖**的（同一份数据，陡度分母用 `median` 给 10/17、用 `IQR` 给 **5/17**）⇒ 只能读作「在该统计量定义下的可分辨对数」，'
    '**不得**读成「真实可分辨对数」；17 个相邻对**不独立**（共用位点）⇒ 只作**描述性计数**，其含义不由 p 值定义。'
)
PIT46 = (
    '46. **集中度 / 离散度型判据必须至少两个独立坐标 + 报告 argmax 位置（Phase 13 的判决性一条）**：'
    '单坐标下的「未决」可能只是**坐标系选择**。Phase 12 的 `G2_mid`（`top3_share_x = 0.5745`，带 `[0.3867, 0.8316]` **跨 0.60**）被判 '
    '`ALLOCATION_AMBIGUOUS`；Phase 13 给同一量补上**尾部概率**（`P(≥0.60) / P(≤0.40) / P(mid)`）与 **argmax 窗口频次直方图**，并**换成两个坐标**后：'
    '`xhalf` 上 `P(≥0.60) = 0.379`（未决）而 **`J_swap` 上 `P(≥0.60) = 0.973`（成立）**；两坐标的 argmax 窗口分别是 **L28→L34（深尾，freq 0.7505）** 与 '
    '**L7→L10（浅端，freq 0.739）**，**相距 13 个跳变位** ⇒ 判 `CONCENTRATION_COORDINATE_DEPENDENT`。'
    '**对策**：凡集中/离散型判据，① 至少两坐标同时报告；② **必须报 argmax 位置**（「集中在哪」与「集中度多少」是两个信息，只报取值会把结论误压在坐标系上）；'
    '③ 把判据写成「`(share, argmax_window)` 二元组在两坐标上的一致性」。'
)

pairs1 = [
    ('header 14 arms',
     '## 1 十三个臂（一次跑完，勿拆散）',
     '## 1 十四个臂（一次跑完，勿拆散）'),
    ('arm row alpha-6',
     '按铁律 (r)(s) 先断言 `spearman(xhalf, depth)` 的符号再谈 G1 |\n',
     '按铁律 (r)(s) 先断言 `spearman(xhalf, depth)` 的符号再谈 G1 |\n' + ARM6 + '\n'),
    ('gate alpha-6',
     'F1 地板比 `0.3582/18.805 = 0.0190`。\n',
     'F1 地板比 `0.3582/18.805 = 0.0190`。\n\n' + GATE6 + '\n'),
    ('header 46 pits',
     '## 4 已实测的坑（44 条，逐条对应数值）',
     '## 4 已实测的坑（46 条，逐条对应数值）'),
    ('pit 45,46',
     '**推广**：凡「跨 Phase 复用 dict / 用键做数值解析」的地方，都要先枚举键型、把非预期键显式报告。\n',
     '**推广**：凡「跨 Phase 复用 dict / 用键做数值解析」的地方，都要先枚举键型、把非预期键显式报告。\n\n' + PIT45 + '\n\n' + PIT46 + '\n'),
]
t1 = patch(P1, pairs1)
w('  -> bytes %d ; arms-hdr=%d ; alpha-6 rows=%d ; pits-hdr=%d ; pit45=%d ; pit46=%d' % (
    len(t1.encode('utf-8')), t1.count('## 1 十四个臂'), t1.count('N2h1-α-6'),
    t1.count('（46 条'), t1.count('45. **「不可排序'), t1.count('46. **集中度')))

# ======================= 2. rdc-phase-closeout =======================
w('')
w('=== rdc-phase-closeout ===')
L13 = (
    '13. **纯再分析 Phase 也必须建装置锚，并把「跨 Phase 逐位复现」写进 floors（Phase 13 实证）**：'
    'Phase 13 **零额外前向**、不加载模型，看似「无从校准」；但只要上一 Phase 把**逐对矩阵**落盘（`*_pairs`）且 bootstrap 用的是**独立生成器 + 冻结 seed**，'
    '就能**重放 RNG 流**（关键是确认「该生成器的**首个消费点**就是主循环」，且此后消费顺序固定）⇒ 逐位复现上一 Phase 的**全部带**，'
    '**包括 2000 值置换零假设**（Phase 13 实测 `J_ci` 18×3 / `top3_share_x` / `top3_share_recover` / `rho_recover` / `rho_xhalf` / `R_ci` / `perm_x` / `perm_rec` '
    '全部 `max|d| = 0.000e+00`）。**同时必须写清认识论地位**：这只证明「实现与上一 Phase 一致」（同数据、同实现谱系），**不是**对上一 Phase 结论的独立验证。'
    '**另**：复现所需的分母也要能重建（Phase 13：`FS_VEC = [FULL_SWAP_pairs[w] for w in order]`，重建后 `mean(FS_VEC)` 必须与上一 Phase 的 `FULL_SWAP` 逐位相等）。'
)
L14 = (
    '14. **把「可行性探针」前置到 seal 之前：最大技术风险用零预算排除（Phase 13 实证）**：'
    'Phase 13 最大的技术风险不是统计，而是「BRNG 流能否逐位重放」——若不成立，seal 里写死的 A0 硬断言会在**正式运行**时才崩。'
    '做法：先写一个**只读探针**（`_feas_probe.py`），**只重建上一 Phase 已发表的数量**（`J_ci`/`top3`/置换零假设…，**不计算任何新统计量**，故不构成数据窥视），'
    '全绿后才冻结 seal。**并把它写成纪律**：任何 Phase，只要判据依赖「某个前提能成立」（可复现性 / 矩阵对齐 / 键型一致），就把该前提的验证**前置**到 seal 之前。'
    '**同源做法已实证**：追加前的「源文件锚点预检」（教训 12）在 Phase 13 抓到 **4 个锚点缺失**（`CONCENTRATION_COORDINATE_DEPENDENT` / `PAIRED_TEST_INFORMATIVE` / 两个 sha8），'
    '**先修源文件再追加，一次成功、无需回滚** ⇒ 教训 11 的「按字节回滚」路径本轮**未被触发**（这正是它应当的常态）。'
)

pairs2 = [
    ('closeout chain count',
     '这条线的收尾链已跑通五次（Phase 8/9/10/11/12）',
     '这条线的收尾链已跑通六次（Phase 8/9/10/11/12/13）'),
    ('header 14 lessons',
     '**Phase 8–12 实测的 12 条收尾教训**',
     '**Phase 8–13 实测的 14 条收尾教训**'),
    ('lesson 13,14',
     '锚点要逐条对应正文里**真实出现**的字符串（实现名、数值、sha8、verdict），而不是「应该出现」的名字。\n',
     '锚点要逐条对应正文里**真实出现**的字符串（实现名、数值、sha8、verdict），而不是「应该出现」的名字。\n'
     + L13 + '\n' + L14 + '\n'),
]
t2 = patch(P2, pairs2)
w('  -> bytes %d ; chain=%d ; lessons-hdr=%d ; L13=%d ; L14=%d' % (
    len(t2.encode('utf-8')), t2.count('六次（Phase 8/9/10/11/12/13）'), t2.count('14 条收尾教训'),
    t2.count('13. **纯再分析 Phase'), t2.count('14. **把「可行性探针」')))

# ======================= 3. 回读复核 =======================
w('')
w('=== 回读复核 ===')
r1 = io.open(P1, encoding='utf-8').read()
r2 = io.open(P2, encoding='utf-8').read()
checks = [
    ('probe 14 arms', r1.count('## 1 十四个臂（一次跑完，勿拆散）') == 1),
    ('probe alpha-6 arm row', r1.count('| **N2h1-α-6 位点间配对 bootstrap**（Phase 13）') == 1),
    ('probe alpha-6 gate', r1.count('**N2h1-α-6 门（qwen3-4b 实测，Phase 13）**') == 1),
    ('probe 46 pits', r1.count('## 4 已实测的坑（46 条，逐条对应数值）') == 1),
    ('probe pit45', r1.count('45. **「不可排序 / 不可分辨」是区间口径的函数') == 1),
    ('probe pit46', r1.count('46. **集中度 / 离散度型判据必须至少两个独立坐标') == 1),
    ('probe key numbers', all(k in r1 for k in ['0.5475', '0.7505', '0.739', '0.064481', '0.1007', '5/17'])),
    ('closeout chain 6', r2.count('六次（Phase 8/9/10/11/12/13）') == 1),
    ('closeout 14 lessons', r2.count('**Phase 8–13 实测的 14 条收尾教训**') == 1),
    ('closeout L13', r2.count('13. **纯再分析 Phase 也必须建装置锚') == 1),
    ('closeout L14', r2.count('14. **把「可行性探针」前置到 seal 之前') == 1),
    ('closeout numbers', all(k in r2 for k in ['0.000e+00', 'perm_x', 'FS_VEC', '一次成功、无需回滚'])),
]
bad = [n for n, ok in checks if not ok]
for n, ok in checks:
    w('  %-24s %s' % (n, 'OK' if ok else '!! FAIL'))
assert not bad, '回读失败: %s' % bad
print('ALL SKILL CHECKS PASSED')
