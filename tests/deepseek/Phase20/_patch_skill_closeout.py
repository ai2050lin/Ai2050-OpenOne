# -*- coding: utf-8 -*-
"""技能同步（Phase 20）：给 `rdc-phase-closeout` 追加教训 31，并修正两处计数（十二次→十三次；27 条→31 条）。"""
import io

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
s = io.open(P, encoding='utf-8').read()
n = 0


def rep(old, new):
    global s, n
    c = s.count(old)
    assert c == 1, 'count=%d :: %r' % (c, old[:80])
    s = s.replace(old, new); n += 1


rep(u'这条线的收尾链已跑通十二次（Phase 8–19）',
    u'这条线的收尾链已跑通十三次（Phase 8–20）')
rep(u'**Phase 8–16 实测的 27 条收尾教训**：',
    u'**Phase 8–20 实测的 31 条收尾教训**：')

NEW = u"""31. **「冻结中间量」在同一模型内按构造与臂无关 ⇒ 不能作跨口径证据；探针/生产必须在「键空间 · 网格 · 配对」三个面上显式对齐（Phase 20 实证，收尾链第十三次）**：
    - **(a) 空检查的检出（本轮最重要）**：本轮 `com_V_recomputed` 固定用上一 Phase 的**冻结 `w_all` 谱**（为与 P17/P18/P19 跨 Phase 可比）⇒ **同模型两臂的 `com_V` 逐位相同、配对 Δ ≡ 0** ⇒ 联合判据「Δ`com_V` ≤ 容差」变成**空检查**，极易被误读成「向量质心跨精度稳健」。**处置模板**：① 报告把量**显式分两类** ——「**锚可比口径**」（冻结谱；只证锚一致、不构成证据）与「**本臂自谱口径**」（`com_V_own_spectrum`；**才是**跨精度证据），并让**判据落在后者**；② `disk_verify` 加一条**事实断言**「冻结口径两臂恒等」（`max|Δ| ≤ 1e-12`）把这种构造性锁死**记录在案**（防后人当证据）；③ Ledger `rev_note` 加 `NOTE (E-comv)` 等效披露。**通用推论**：凡复用上一 Phase 的冻结中间量作「可比基线」，都要先问「**它在本次实验变量下会不会变**」——**不变即不可作证据**，必须同时落一份**随臂重算**的版本。**副产品**：本轮自谱位移 26.150 → 26.057（−0.093 层）与 P19 独立测得的 Δ`com_V` = 0.0930 吻合 ⇒ **跨 Phase 两套实现互证**（只报冻结口径就白白丢掉这个可检验性）。
    - **(b) 探针 vs 生产的三个对齐面**（本轮三项全部由 SMOKE/探针在**正式运行前**抓出，零模型预算）：**① 键空间** —— 冻结锚谱键是 `w_all/w_mlp/w_attn`，本臂自算谱键是组件名 `INC_*`；混用同一函数 ⇒ `KeyError: 'w_all'`（SMOKE 第一跑）。修法：拆 `sper_anchor()` / `sper_own()` 两个显式函数，报告里「`spearman(w,b)`」必须标 w 的来源。**② 网格** —— `com_layer` 族是 **α 网格**的函数 ⇒ 在缩幅网格上做「逐位复现冻结锚」**必然 DRIFT**。修法：`FULL_SCALE = (not SMOKE) and (not PROBE)`；`E9_anchor` 加 `applies` 字段，缩幅下记 **N/A**（不是 FAIL）。**③ 配对与实例** —— 缩配对（24→4）会让 `U_ℓ = SVD(类别质心差)` 的秩从 `n_classes−1 = 5` **退化到 2**、`FULL_SWAP` 偏离锚 ⇒ 探针读数**失去口径意义**。修法：PROBE **只缩网格与 BP**，**不缩**配对/实例 ⇒ Panel 行为类的量与 α 网格无关，可与生产**逐位互证**（本轮 A0 探针直接把 P18/P16/P17 **三套冻结锚逐位复现**，是性价比最高的一道提前验证）。
    - **(c) 「seal 声明路径」与「实现落盘路径」必须对齐，否则下游静默读不到**：本轮 seal 的 `probe_files` 声明 `_probe20_A0_*.json`，而驱动按主脚本约定写成 `_armrec20_probe_A0_*.json` ⇒ `closeout` / `gen_memo` / `disk_verify` **全都读不到探针件**（其中一处还会因把 dict 塞进 `'%.4f'` 而 **TypeError**）。修法 = 加一个**幂等发布步**把记录**复制**到声明路径（**不改 seal 字节**）+ `sha256` 复核。**纪律**：写 seal 时就对齐两处命名，或显式落一个 publish 步并写进收尾链。
    - **(d) 独立复核的秩口径必须与主脚本同义**：主脚本 `spearman()` 用**序数秩**（`argsort(argsort)`）+ std 守卫；复核脚本若擅自改用**平均秩**并断言 `1e-9` 相等，一旦谱里出现**并列值**就**必 FAIL**（且这是复核脚本的错，不是产物的错）。修法：**断言用主脚本文义**，另算平均秩作信息量，两者差异（= 存在并列）记 **WARN** 而非 FAIL。
    - **(e) 标签重算要按「主脚本文义」而非「seal 严格文字」**：本轮 `Q9_shallow_retained` 主脚本实现是**逐配对同侧**（`gap_sign_same`），而 seal 的文字是「四臂皆 ≥2.0 层」——两者**可分离**。复核应「**重算 → 按主脚本文义导出标签 → 比对**」，并把 seal 的严格形**单列 WARN**。这正是教训 29f「不得写死预期结果」的加强版：连**标签定义**都要以**实现**为口径、并显式披露其与 seal 文字的差距。

"""

anchor = u'\n## 参照实现（Phase 3125'
assert s.count(anchor) == 1, 'anchor %d' % s.count(anchor)
s = s.replace(anchor, u'\n' + NEW + u'\n## 参照实现（Phase 3125')

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
n += 1
print('PATCHED %d spots' % n)
t = io.open(P, encoding='utf-8').read()
for pr in [u'十三次（Phase 8–20）', u'31 条收尾教训', u'31. **「冻结中间量」', u'NOTE (E-comv)',
           u'sper_anchor()', u'E9_anchor` 加 `applies`', u'幂等发布步']:
    print('  chk %-26s -> %d' % (pr, t.count(pr)))
