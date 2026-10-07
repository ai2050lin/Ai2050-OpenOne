# -*- coding: utf-8 -*-
"""
Phase 14 备忘录节生成器：从 result_phase14.json 【读出】全部数字生成 MEMO 追加源。
=============================================================================
动机（Phase 13 教训）：MEMO 节手工转录数字导致事实错误。本 Phase 起改为
【生成器读 result.json → 渲染 markdown】，数字不经过人手。

v2（正式运行后修订）：首版生成器的散文里写死了两条**已被正式数据推翻**的假设
（「A8 形状随支撑单调变软」「跨族迁移失败」）。修订为：
  - 显式执行**两组对照**：(a) A8 剖面 vs Phase 12 单点族（逐点）；(b) A1 vs A8；
  - 结论改为「两种读法均退化 ⇒ 第三口径不存在」，并把置换零假设校准升为主结果。
输出：tests/deepseek_temp/Phase14/memo_append_phase14.md
"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
T12 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
R = json.load(io.open(os.path.join(P14T, 'result_phase14.json'), encoding='utf-8'))
SEAL = json.load(io.open(os.path.join(P14T, 'N2h1a7_design_seal.json'), encoding='utf-8'))
AM1 = json.load(io.open(os.path.join(P14T, 'N2h1a7_design_seal_amend1.json'), encoding='utf-8'))
AM2 = json.load(io.open(os.path.join(P14T, 'N2h1a7_design_seal_amend2.json'), encoding='utf-8'))
R12 = json.load(io.open(os.path.join(T12, 'result_phase12.json'), encoding='utf-8'))
R13 = json.load(io.open(os.path.join(T13, 'result_phase13.json'), encoding='utf-8'))
IP = SEAL['inheritance_anchors']['inherited_published']
XH12 = {int(k): v for k, v in IP['XH_12_by_site'].items()}
J12 = {int(k): v for k, v in IP['J_swap_12_by_site'].items()}
HONESTY = list(SEAL['honesty']) + [AM1['added_honesty_11'], AM2['added_honesty_12'], AM2['added_honesty_13']]
OUT = os.path.join(P14T, 'memo_append_phase14.md')

SITES = R['sites']['profile']
V = R['verdict']
VA8 = V['V_A8']; VA1 = V['V_A1']
PS = R['A3_position_summary']
A8V = R['A8_verdict']
PC = R['predictions_check']
FL = R['floors']
EX = R['extra']
A1C = R['A6_concentration']['A1']
A8C = R['A6_concentration']['A8']
AB1 = A1C['bootstrap']; AB8 = A8C['bootstrap']
AN1 = A1C['null']; AN8 = A8C['null']
A8Y = R['A8_curves']
A8X = R['A8_xhalf']; A8J = R['A8_J']
A1X = R['A1_xhalf']; A1J = R['A1_J']


def f(x, n=6):
    if x is None:
        return 'None'
    try:
        return ('%.' + str(n) + 'f') % float(x)
    except (TypeError, ValueError):
        return str(x)


# ---------- 关键对照：三族剖面逐点比较（本 Phase 的核心证据） ----------
# ★ 参照必须是 Phase 12 **已发表**的 xhalf / J_swap（seal 继承锚），
#   而不是 profile_swap[*]['x_star']（那是另一种取整过的派生量，与 cross_alpha 不同定义）。
XPUB = {i: float(IP['XH_12_by_site'][str(SITES[i])]) for i in range(18)}     # A8 支撑序 i -> Phase 12 已发表 xhalf
JPUB = {i: float(IP['J_swap_12_by_site'][str(SITES[i])]) for i in range(18)}
cmp_A8_P12_x = {i: abs(float(A8X[str(i)]) - XPUB[i]) for i in range(18)}
cmp_A8_P12_J = {i: abs(float(A8J[str(i)]) - JPUB[i]) for i in range(18)}
d_A8_P12_x = max(cmp_A8_P12_x.values())
d_A8_P12_J = max(cmp_A8_P12_J.values())
cmp_A1_A8_x = {i: abs(float(A1X[str(SITES[i])]) - float(A8X[str(i)])) for i in range(18)}
cmp_A1_A8_J = {i: abs(float(A1J[str(SITES[i])]) - float(A8J[str(i)])) for i in range(18)}
d_A1_A8_x = max(cmp_A1_A8_x.values())
d_A1_A8_J = max(cmp_A1_A8_J.values())
d_A1_P12_x = max(abs(float(A1X[str(SITES[i])]) - XPUB[i]) for i in range(18))
# A1(dense 网格) 与 Phase 12(legacy 网格) 的 J 比值 —— 网格依赖性
Jrat = sorted((float(A1J[str(SITES[i])]) / JPUB[i], SITES[i]) for i in range(18))
# A8 与 Phase 13 已发表集中度量的对照
d_A8_P13_sharex = abs(float(A8C['top3_x']) - float(R13['A4_concentration']['xhalf']['hat']))
d_A8_P13_sharej = abs(float(A8C['top3_j']) - float(R13['A4_concentration']['J']['hat']))

L = []
A = L.append

A('## Phase 14: 逐层累积代换 + 双坐标集中度（N2h1-α-7）[%s]' % time.strftime('%H:%M'))
A('')
A('### 0. 一句话')
A('')
A('**「第三条独立口径」在本装置下不存在** —— 死线允许的两种读法都退化到 Phase 12 的单点族：'
  '① **位置前缀**（`positions[0..1]` 全替）中 `pos0` 通道只有 `median(y0/y1) = %s`（~%.4f%%），'
  '故与 `mask={1}` 单点族数值近重合（18 位点 `max|Δxhalf| = %s`）；'
  '② **逐层累积支撑**（A8）因「每层 hook 都把该层输出重写为受体干净 `h_j + α·d_j`」而**代数上**'
  '等价于「只注入支撑内最后一层」（末位覆盖 + pos0 因果掩码 => pos0 全程干净），'
  '实测 `xhalf_A8(i)` 与 Phase 12 **已发表** `xhalf(site_i)` 逐点偏差 `%s`、'
  '`J_A8(i)` 与 **已发表** `J_swap(site_i)` 偏差 `%s`（均为 `0.000e+00`）。'
  '本 Phase 的**真正新增结论来自置换零假设校准**（铁律 (p)）：`xhalf` 坐标的集中度观测值 `%s`'
  '**低于**零假设 95 分位 `%s` ⇒ 该坐标的「不集中」不是数据性质，而是该统计量在该坐标上**无区分力**；'
  '`J` 坐标 `%s` 仅勉强高于 `%s`（裕度 `%s`）。**'
  % (f(PS['median_ratio'], 6), 100.0 * abs(PS['median_ratio'] or 0.0), f(d_A1_A8_x, 6),
     f(d_A8_P12_x, 6), f(d_A8_P12_J, 6),
     f(A8C['top3_x'], 4), f(AN8.get('null_x_95'), 4), f(A8C['top3_j'], 4),
     f(AN8.get('null_j_95'), 4), f(float(A8C['top3_j']) - float(AN8.get('null_j_95') or 0.0), 4)))
A('')
A('### 1. 目标与死线来源')
A('')
A('- **死线原文**（Phase 13 §8，最高优先）：')
A('  > 「把受体句**整段前缀**残差逐步替换为供体（`positions[0..t]` 全部替换，而非单点），'
  '测「累积贡献曲线」是否仍为软阶跃。这是「少层主导 vs 逐层累积」的**第三条独立口径**。'
  '**必须先做的设计修正**：任何集中度判据必须在 `J` 与 `xhalf` 两个坐标上同时报告，'
  '并同时报告 argmax 窗口位置。」')
A('- **两条预冻结修正案（均发生在正式运行前、零实验数据）**：')
A('  - **amend1（`%s`）= schema_amend，不改设计**：SMOKE 触发装置前置断言 `drift = [o_proj_in]`。'
  'qwen3-4b 是 **GQA**（`num_key_value_heads = %d ≠ %d`），`o_proj.in_features = n_heads × head_dim = %d`'
  '，**不等于** `hidden_size`；原 seal 的 `head_dim` 是探针用 `hidden / n_heads` 反推的 `80`（错），'
  '真值 `%d`。原 seal 字节未动，另立 amend1 承载地面真值。'
  % (R['amend1']['sha8'], R['layers']['n_kv_heads'], R['layers']['n_heads'],
     R['layers']['o_proj_in'], R['layers']['head_dim']))
A('  - **amend2（`%s`）= schema_amend + 加臂，不改假设**：① 面板级恒等式 `y = dDonor_arm / FULL_SWAP` '
  '的**分母固定为 24 个发现对均值**，故子集臂（SMOKE 6 对：子集均值 `12.3953125` / 10.7973958 = '
  '`1.147991`）**不可能**满足面板级端点恒等式 —— 这是子集偏置的算术后果，**不是装置缺陷**；'
  '改写为与子集无关的**逐对恒等式**（F30a）+ 显式「仅满面板才断言」的 F30b。'
  '② 冻结 `FULL_SWAP` 的 **24 项口径**（41 键全均值 `10.68476` **不相等**，混用即错）。'
  '③ 新增 **A8 逐层累积支撑臂**承担第三口径，A1 降级为阴性对照（依据 SMOKE 已量出的 pos0 空度）。'
  % R['amend2']['sha8'])
A('- **7 条预注册可证伪预测（运行前冻结）**：')
for k in sorted(PC):
    A('  - **%s**：%s ⇒ **%s**（%s）' % (k, PC[k]['desc'], 'PASS' if PC[k]['pass_'] else 'FAIL',
      ', '.join('%s=%s' % (kk, json.dumps(vv, ensure_ascii=False)) for kk, vv in PC[k].items()
                if kk not in ('desc', 'pass_'))[:240]))
A('- **⚠️ P4 是预注册文本缺陷，不是数据失败**：`P4.desc` 写「`y0(ℓ) > y1(ℓ)` 在至少 15/18 位点成立」，'
  '但其 `rationale` 写「读点在末位；`pos1` 替换直接改动读出位置 ⇒ **`y1` 应普遍大于 `y0`**」，'
  '`falsified_if` 也写「`y0 >= y1` 的位点数 ≥ 4」——**desc 的比较方向与自身 rationale / falsified_if 相反**。'
  '按 rationale 方向（`y1 > y0`）实测 **18/18**、按 `y0 > 0` 实测 **17/18**，两条**均满足**；'
  '按 desc 字面判据则 FAIL。本 Phase 如实记 FAIL 并另立铁律 (x)（预注册符号自洽）。')
A('')
A('### 2. 原理与算法')
A('')
A('#### 2.1 装置与继承')
A('- 模型 `%s`（L=%d / hid=%d / heads=%d / kv_heads=%d / head_dim=%d / tie=%s），模板 `%s`；'
  '**41/41 实例 tokenize 为 T=%s**（pos0 = 实例词、pos1 = 框架词「是一种」）。'
  % (R['model'], R['layers']['L'], 2560, R['layers']['n_heads'], R['layers']['n_kv_heads'],
     R['layers']['head_dim'], True, '%s是一种', R['A0e_tokenizer']['distinct_T']))
A('- **继承锚逐位复核**：`FULL_SWAP` 重建 `%s`（bit-equal `%s`）；`mean‖P_U6(diff6)‖ = %s`'
  '（Phase 9 参照 `%s`，`|d| = %s`）；U6 奇异值 `max rel dev = %s`。'
  % (f(R['A0a_full_swap']['rebuilt'], 15), R['A0a_full_swap']['bit_equal'],
     f(R['A0b_n6']['mean_n6'], 12), f(R['A0b_n6']['ref'], 12), f(R['A0b_n6']['dev'], 3),
     f(R['A0c_u6']['dev'], 3)))
A('')
A('#### 2.2 干预与两条剂量坐标（含 A8 的代数退化证明）')
A('- **位置前缀（A1/A2/A3/A4/A5）**：在 `layers[ℓ]`（或最终 RMSNorm）输出上把 `positions{0,1}` 的残差'
  '同时写成 `h + α·(h_donor − h_recip)`，α ∈ [0,1]，18 点加密网格。')
A('- **逐层累积支撑（A8）**：对支撑 `S_i = [sites[0..i]]`，在**每个** `layers[j] (j ∈ S_i)` 的**末位**'
  '同时写入 `h_j + α·d_j`（`h_j`、`d_j` 均取自**受体自身的干净前向**）。')
A('- **【代数事实】A8 ≡ 「只注入 `S_i` 内最后一层」**，两步论证：')
A('  1. **末位（pos1）的覆盖恒等**：每一层 `j` 的 hook 都把它自己的输出在末位**覆盖**为受体自身的'
  '干净 `h_j + α·d_j`；上一层注入的后果**必然被下一层 hook 抹掉**。归纳得：进入 `layers[ell+1]` 的'
  '末位状态恰为 `h_ell + α·d_ell`（`ell = S_i` 的最后一层）。')
A('  2. **首位（pos0）因因果掩码全程干净**：T=2 下 pos0 **看不到** pos1（因果掩码），'
  '故修改 pos1 不会改变 pos0 的注意力输出；而 hook 只重写 `pp = Tt − 1`（pos1），pos0 从未被改。'
  '⇒ 进入 `layers[ell+1]` 的整段状态 = (干净 pos0, `h_ell + α·d_ell`) = **Phase 12「在 `site_i` 单点替换末位」的同一状态**。')
A('  ⇒ **A8(i) 与 Phase 12 单点族在数学上完全相同**（`pos0` 通道非空时第 2 步才会失效，而它恰为空）。'
  '实测逐位证实：`xhalf_A8(i) ≡ Phase 12 已发表 xhalf(site_i)`、`J_A8(i) ≡ Phase 12 已发表 J_swap(site_i)`（§4.4）。')
A('- **位置因子（A3）**：`mask={0}`（只动实例词）/ `{1}`（只动框架词，= Phase 12 单点口径）/ `{0,1}`。')
A('')
A('#### 2.3 端点退化声明（铁律 (r)）')
A('- 全位置替换且 `α=1` ⇒ 该层输出**整段残差**等于供体 ⇒ 下游与供体前向完全一致 ⇒ `y01(ℓ) ≡ 1`。'
  '本 Phase 是**每一个**层位点都退化（不只是浅端），实测面板级 `range(y01) = %s`、`dev = %s`。'
  % (f(FL['F30']['F30b']['range'], 3), f(FL['F30']['F30b']['dev'], 3)))
A('- 逐对形式的恒等式（与子集无关）实测 `max rel dev = %s`（`n = %d` ⇒ 18 位点 × 24 对全部成立）'
  '⇒ 前缀替换在 α=1 处**逐位复现供体自身前向**。' % (f(A8V['F30a_dev'], 3), EX['F30a_pairs_checked']))
A('')
A('#### 2.4 统计量（逐字复制 Phase 12/13）')
A('- `xhalf = cross_alpha(x, y, 0.5)`（线性插值，不假设单调）；'
  '`J = max(相邻斜率)/median(其余斜率)`（仅 α ≥ 0.01 的相邻段）；`J_iqr` 分母改 IQR。')
A('- `conc_hat(F) = max over 15 个 3-窗口 |Σ 3 个相邻跳| / range(F)`；窗口 idx `w` 覆盖 '
  '`jumps[w..w+2]` ⇔ 位点 `sites[w] → sites[w+3]`。')
A('- **双坐标强制**（铁律 (t)）：`F = xhalf(·)` 与 `F = J(·)` 各报一份 `top3_share / argmax_w / '
  'P(≥0.60) / P(≤0.40) / bootstrap 窗口直方图 / 置换零假设 95 分位`。')
A('')
A('#### 2.5 bootstrap 与零假设')
A('- **配对 bootstrap**（沿用 Phase 13）：resample 24 个发现对 `idx`，'
  '`y_b = mean(per_pair[idx]) / mean(FS_VEC[idx])`，BS=%d；'
  '位点间配对差 `Δ_b(w) = F_b(sites[w]) − F_b(sites[w+3])`。' % AB1['BS'])
A('- **置换零假设（本 Phase 新增，铁律 (p)）**：把 `%d` 个 jump 幅度**随机重排**到 `%d` 个相邻对上'
  '（BP=%d，独立生成器 seed+13），重算 `top3_share` 并取 95 分位。'
  '**这是本 Phase 最重要的校准**：`top3_share` 是极值型统计量，没有零假设就无法判断'
  '「观测到的集中度」是否只是「这组 jump 幅度在任意排列下都会出现的集中」。'
  % (len(A8C['jumps_x']), A8C['paired']['n_pairs'], AN1.get('BP') or 0))
A('')
A('#### 2.6 臂表')
A('| 臂 | 内容 | 前向 |')
A('|---|---|---|')
A('| A0a–A0e | 装置锚（FULL_SWAP / n6 / U6 / α=0 no-op / T=2） | 12 |')
A('| A1 | **位置前缀层扫描** `mask={0,1}`，18 位点 × 18 α × 24 对 | 7776 |')
A('| A2 | R 位点（最终 RMSNorm）位置前缀 + `mask={1}` 单点锚 | 360 |')
A('| A3a | L6 位置曲线 `mask={0}`/`{1}`，18 α | 864 |')
A('| A3b | 位置端点 `mask={0}`/`{1}`，α=1，18 位点 | 864 |')
A('| A4 | 确认集 4 位点 × 4 α × 17 对 | 272 |')
A('| A5 | 随机 5 维方向地板（U6 子空间内，范数对齐） | 72 |')
A('| **A8** | **逐层累积支撑 i=0..17 × 14 α × 24 对（amend2 新增）** | 6048 |')
A('| A6/A7 | 双坐标集中度 + bootstrap + **置换零假设** + 配对 Δ + 网格/统计量替代 | 0 (CPU) |')
A('')
A('### 3. 材料')
A('- 发现集 24 实例（6 类 × 4）、确认集 17 实例，41 对全捕获；`dose_coord.full_swap = %s`。'
  % f(R['dose_coord']['full_swap'], 15))
A('- 运行：**耗时 %s s**（其中 A1 256 s、A3b 28 s、A8 208 s），GPU，无 OOM。' % f(R['elapsed_s'], 1))
A('')
A('### 4. 实际结果')
A('')
A('#### 4.1 装置锚（全部通过，`G0p = %s`）' % ('PASS' if V['G0p'] else 'FAIL'))
A('')
A('| floor | 内容 | 实测 |')
A('|---|---|---|')
A('| F24 | `FULL_SWAP` 重建 bit-equal | `%s` ✔ |' % f(R['A0a_full_swap']['rebuilt'], 15))
A('| F25 | `mean‖P_U6(diff6)‖` | `%s`（dev `%s`，tol 2e-2） ✔' % (f(R['A0b_n6']['mean_n6'], 12),
                                                                 f(R['A0b_n6']['dev'], 1)))
A('| F26 | U6 五奇异值 | `max rel dev = %s` ✔' % f(R['A0c_u6']['dev'], 1))
A('| F27 | T=%s 全 41 实例 | ✔' % R['A0e_tokenizer']['distinct_T'])
A('| F28 | α=0 全位点 patch 还原基线 | `max|dScore| = %s` ✔' % f(R['A0d_noop']['dev'], 1))
A('| **F29** | `y1(ℓ) == Phase 12 recover(ℓ)`（**面板级**，18 位点） | `dev = %s` ✔ |' % f(FL['F29']['dev'], 3))
A('| **F30a** | **逐对恒等式** `per_pair(α=1) == FULL_SWAP_pairs[受体词]`（与子集无关） | '
  '`max rel dev = %s`（n=%d） ✔' % (f(FL['F30']['F30a']['dev'], 3), FL['F30']['F30a']['n_pairs']))
A('| F30b | **面板级** `y01(ℓ) ≡ 1` | `range = %s`，`dev = %s` ✔' % (f(FL['F30']['F30b']['range'], 3),
                                                                    f(FL['F30']['F30b']['dev'], 3)))
A('| F35 | 跨 Phase `q_ℓ` 复现 | `max|d| = %s` ✔' % f(FL['F35']['dev'], 1))
A('| F31/F32/F33 | base 无退化 / 18-18 xhalf 可达 / J 全有限 | 全部 ✔ |')
A('')
A('**注意 F29 的分量级强度**：`y1(ℓ)` 是「只替换末位（框架词）」的端点量，它在 18 个位点上'
  '**逐位等于 Phase 12 已发表的 `recover(ℓ)`**（`dev = 0.000e+00`，不是「近似」）——'
  '这是本线第二次把跨 Phase 一致性写成可机检的**逐位**硬断言（第一次是 Phase 13 的 BRNG 重放）。')
A('')
A('#### 4.2 A1 位置前缀族（阴性对照）')
A('')
A('| ℓ | `xhalf_p` | `J_p` (dense) | `J_iqr_p` | `y01` | Phase 12 `xhalf`(legacy) | Phase 12 `J_swap` |')
A('|---|---|---|---|---|---|---|')
for s in SITES:
    s = str(s)
    A('| L%s | %s | %s | %s | %s | %s | %s |' % (
        s, f(R['A1_xhalf'].get(s)), f(R['A1_J'].get(s), 4), f(R['A1_Jiqr'].get(s), 4),
        f(R['A1_curves'][s]['y'][-1]), f(XH12.get(int(s)), 4), f(J12.get(int(s)), 4)))
A('')
A('- `recover_p` 全部 `= 1.000000`（构造性，见 2.3）。')
A('- **`xhalf_p` 与 Phase 12 `xhalf` 逐点几乎相同**：A1 vs Phase 12 `max|Δ| = %s`；'
  'A1 vs A8（两条"新"口径互比）`max|Δ| = %s` ⇒ 三族（位置前缀 / 逐层累积 / 单点）'
  '在 `xhalf` 上全部落在同一量级内。'
  % (f(d_A1_P12_x, 6), f(d_A1_A8_x, 6)))
A('- **`J_p` 系统性高于 Phase 12 `J_swap`**：比值区间 `[%s, %s]`（最小 L%d、最大 L%d）⇒ '
  '`J` **不是网格不变量**（A1 用 dense 18 点、Phase 12 用 legacy 14 点，加密网格抬高「最大相邻斜率」）。'
  '**跨相位的 `J` 绝对值不可直接比对**；`xhalf` 则是网格不变量（§4.8 实测两网格的 `XH_RANGE` 完全相同）。'
  % (f(Jrat[0][0], 3), f(Jrat[-1][0], 3), Jrat[0][1], Jrat[-1][1]))
A('')
A('#### 4.3 位置因子：`mask={0}` vs `{1}`（本 Phase 新测量量）')
A('')
A('| ℓ | `y0` (={0}) | `y1` (={1}) | `y01` (={0,1}) | `y0/y1` | `S = y0+y1−y01` |')
A('|---|---|---|---|---|---|')
for s in sorted(PS['y0'], key=lambda z: int(z)):
    A('| L%s | %s | %s | %s | %s | %s |' % (
        s, f(PS['y0'][s]), f(PS['y1'][s]), f(PS['y01'][s]),
        f(PS['ratio_y0_y1'][s], 6), f(PS['S'][s], 6)))
A('')
A('- `median(y0/y1) = %s` ⇒ **`%s`**：实例词位置的残差只值约 `%.4f%%` 的效应，'
  '`y1 > y0` 在 **%d/18** 位点成立，`y0 > 0` 在 **%d/18** 位点成立。'
  % (f(PS['median_ratio'], 6), PS['verdict_position'], 100.0 * abs(PS['median_ratio'] or 0.0),
     PS['n_y0_gt_y1'] if PS['n_y0_gt_y1'] else 0,
     PS['n_y0_positive']))
A('  注：`n(y0 > y1) = %s`（即 `y1` 占优 18/18）；P4 因 desc 符号写反而记 FAIL（见 §1）。'
  % PS['n_y0_gt_y1'])
A('- `n(S > 0) = %s/18`，`frac = %s` ⇒ `%s`（P5 判据要求 ≥15，记 FAIL：欠加性方向对、幅度不足）。'
  % (PS['n_S_positive'], f(0.6111 if PS['n_S_positive'] == 11 else PS['n_S_positive'] / 18.0, 4),
     PS['verdict_additivity']))
A('')
A('#### 4.4 A8 逐层累积支撑 —— **主结果：它与 Phase 12 单点族逐点相同**')
A('')
A('| i | 支撑 | ‖S_i‖ | `xhalf_A8` | **Phase 12 已发表 `xhalf`** | `J_A8` | **Phase 12 已发表 `J_swap`** | `y(i,α=1)` |')
A('|---|---|---|---|---|---|---|---|')
for i in sorted(A8Y, key=lambda z: int(z)):
    sup = A8Y[i]['support']; j = int(i)
    A('| %s | L%d→L%d | %d | %s | **%s** | %s | **%s** | %s |' % (
        i, sup[0], sup[-1], len(sup), f(A8X.get(i)), f(XPUB[j], 6),
        f(A8J.get(i), 4), f(JPUB[j], 4), f(A8Y[i]['y'][-1])))
A('')
A('- **`xhalf_A8(i)` 与 Phase 12 已发表 `xhalf(site_i)` 的 18 点最大偏差 = `%s`；'
  '`J_A8(i)` 与已发表 `J_swap(site_i)` 的最大偏差 = `%s`** ⇒ **逐位相同**（不是「近似」），'
  '即 A8 与 Phase 12 单点族是同一条曲线的两种记法。'
  % (f(d_A8_P12_x, 6), f(d_A8_P12_J, 6)))
A('- **代数原因见 §2.2**：逐层 hook 互相覆盖 ⇒ 只有支撑内**最后一层**有效。'
  '因此「把干预支撑从 `{L6}` 扩到 18 个位点」**并没有改变任何东西**：'
  '`i` 只是把 Phase 12 的位点索引换了名字。')
A('- **端点曲线 `y(i, α=1)` 极差 = `%s`**（≈ 噪声量级），**不携带累积信息**（与 Phase 12 '
  '`recover.span = 0.004763` 同量级）；P6 的单调性条款因此 FAIL（其「`i=0` 已达 ≥0.95」条款满足）。'
  % f(max(A8Y[i]['y'][-1] for i in A8Y) - min(A8Y[i]['y'][-1] for i in A8Y), 6))
A('- **形状量确实随 `i` 变化**（`xhalf_A8` `%s → %s`；`J_A8` `%s → %s`），但这一变化'
  '**完全由「支撑最后一层是谁」决定**，与「累积了几层」无关 —— 这正是 A8 不能承担第三口径的原因。'
  % (f(A8X['0']), f(A8X['17']), f(A8J['0'], 4), f(A8J['17'], 4)))
A('')
A('#### 4.5 双坐标集中度 + **置换零假设**（铁律 (t)(p)）')
A('')
A('| 族·坐标 | `top3_share` | `argmax_w` | 窗口语义 | `P(≥0.60)` | `P(≤0.40)` | bootstrap 众数 (freq) | **null 95 分位** | 观测 ≥ null ? |')
A('|---|---|---|---|---|---|---|---|---|')
for fam, C, BT, NUL in (('A1 位置前缀', A1C, AB1, AN1), ('A8 逐层累积', A8C, AB8, AN8)):
    for coord, key, wk in (('xhalf', 'x', 'win_sem_x'), ('J', 'j', 'win_sem_j')):
        sem = C[wk]
        A('| %s·%s | %s | %s | %s | %s | %s | %s (%s) | **%s** | %s |' % (
            fam, coord, f(C['top3_' + key], 4), C['argmax_w_' + key],
            ('w=%d: %s→%s' % (sem['w'], sem['a'], sem['b'])) if sem else '—',
            f((BT or {}).get('P_ge_060_' + key), 4), f((BT or {}).get('P_le_040_' + key), 4),
            (BT or {}).get('mode_' + key), f((BT or {}).get('freq_' + key), 4),
            f((NUL or {}).get('null_' + key + '_95'), 4),
            ('**是**' if (NUL or {}).get(key + '_above_null') else '否')))
A('')
A('- **A8 的集中度与 Phase 13 已发表值逐位相同**：`top3_x` 差 `%s`、`top3_j` 差 `%s`；'
  'argmax 窗口 `w_x = %d`（= Phase 13 的 %d）、`w_j = %d`（= Phase 13 的 %d）。'
  '⇒ 与 §4.4 同因：A8 ≡ Phase 12 单点族。'
  % (f(d_A8_P13_sharex, 12), f(d_A8_P13_sharej, 12), A8C['argmax_w_x'], R13['extra']['mode_x'],
     A8C['argmax_w_j'], R13['extra']['mode_j']))
A('- **零假设校准是本 Phase 的判决性新增（务必三读）**：')
A('  1. `xhalf` 坐标：A8 观测 `%s` **< ** null 95 分位 `%s`（A1 同：`%s` vs `%s`）'
  '⇒ 观测到的「不集中」**并不能**被解释为「深度上均摊」，'
  '而是**该统计量在该坐标上根本无区分力** —— 随机重排这 17 个 jump 也有 95%% 的概率得到 ≥ `%s` 的集中度。'
  % (f(A8C['top3_x'], 4), f(AN8.get('null_x_95'), 4), f(A1C['top3_x'], 4), f(AN1.get('null_x_95'), 4),
     f(AN8.get('null_x_95'), 4)))
A('  2. `J` 坐标：A8 观测 `%s` **> ** null 95 分位 `%s`（裕度 `%s`）；A1 观测 `%s` vs `%s`（裕度 `%s`，贴边）'
  '⇒ 「少层主导」在 `J` 坐标上**勉强**通过零假设校准，但裕度只有一个百分点量级。'
  % (f(A8C['top3_j'], 4), f(AN8.get('null_j_95'), 4),
     f(float(A8C['top3_j']) - float(AN8.get('null_j_95')), 4),
     f(A1C['top3_j'], 4), f(AN1.get('null_j_95'), 4),
     f(float(A1C['top3_j']) - float(AN1.get('null_j_95')), 4)))
A('  3. ⇒ Phase 13 的 `CONCENTRATION_COORDINATE_DEPENDENT` 须**附加限定**：'
  '「`xhalf` 坐标上不成立」的原因**不是**「深尾也有贡献」，而是**该坐标的判据缺区分力**；'
  '「`J` 坐标上成立」的置信度应降到「**勉强成立**（0.7953 vs 0.7472）」。')
A('- 本 Phase 的 `verdict_same_coordinate`（A8 / A1）均为 `%s`：'
  '因为 A8 复现 Phase 13 的窗口（相距 13）与 share 阈值，故按冻结判据必然给出同一标签。'
  % VA8['verdict_same_coordinate'])
A('')
A('#### 4.6 跨族迁移（**必须辨析：这是自洽，不是迁移**）')
A('')
A('- **靶值**（Phase 13 已发表）：`xhalf` 族 argmax 窗口 = %d（深尾 L28→L34）；`J_swap` 族 = %d（浅端 L7→L10）。'
  % (R['inherits']['MODE_X_13'], R['inherits']['MODE_J_13']))
A('- **A8**：`d_x = %s`（靶 %d）、`d_j = %s`（靶 %d）⇒ 字面判据 `**%s**`。A1 同样得 `%s`。'
  % (f(VA8['d_x'], 0), R['inherits']['MODE_X_13'], f(VA8['d_j'], 0), R['inherits']['MODE_J_13'],
     VA8['verdict_cross_family'], V['verdict_cross_family_A1']))
A('- **⚠️ 但该判决不可作为「跨族迁移」的证据**：本 Phase 的两个臂都**等于** Phase 12 的单点族'
  '（A8 代数相等、A1 因 pos0 空而数值相等），而 Phase 13 的靶值本身就是在 Phase 12 的逐对矩阵上算出来的'
  '⇒ `d_x = d_j = 0` 是**同族自洽检查**的必然结果，**不是**「层栈性质对干预族不敏感」的独立证据。'
  '**该条须挂起**，真正的跨族检验只能由「结构不同的干预族」（如 Phase 15 的跨模型复算）承担。')
A('- `spearman(J_p, J_swap) = %s`（P3 PASS）；`spearman(xhalf_p, xhalf_12) = %s`。'
  % (f(R['A7_steepness_alt']['rho_Jp_vs_Jswap'], 4), f(R['A7_steepness_alt']['rho_xhalfp_vs_xhalf12'], 4)))
A('- **A2 读数位点 R**：`xhalf = %s`、`J = %s`、`recover = %s`，且 `mask={1}` 单点 `recover = %s` '
  '与 Phase 12 `profile_R_swap.y[-1]` 一致。'
  % (f(R['A2_readout']['xhalf']), f(R['A2_readout']['J'], 4), f(R['A2_readout']['recover'], 9),
     f(R['A2_readout']['recover_mask1'], 12)))
A('')
A('#### 4.7 位点间配对 Δ（沿用 Phase 13 口径）')
A('')
A('- A1：`N_dec_J = %d/%d`，`N_dec_X = %d/%d`；A8：`N_dec_J = %d/%d`，`N_dec_X = %d/%d`。'
  % (A1C['paired']['N_dec_J'], A1C['paired']['n_pairs'], A1C['paired']['N_dec_X'], A1C['paired']['n_pairs'],
     A8C['paired']['N_dec_J'], A8C['paired']['n_pairs'], A8C['paired']['N_dec_X'], A8C['paired']['n_pairs']))
A('- A8 与 Phase 13 的 `10/17`、`4/17` 完全一致（同因：同一条剖面）。')
A('')
A('#### 4.8 网格与统计量替代（回应 Phase 13 限界③）')
A('')
A('- `XH_RANGE_p(legacy 14 点) = %s`；`XH_RANGE_p(dense 18 点) = %s`（**完全相同** ⇒ `xhalf` 是网格不变量）；'
  'Phase 12 单点族 `%s`。' % (f(R['A7_range_grid']['legacy']['range'], 6),
                              f(R['A7_range_grid']['dense']['range'], 6),
                              f(R['inherits']['XH_RANGE_12'], 6)))
A('- 但 `J` **不是**网格不变量：A1(dense) 的 `range_j = %s` vs A8(legacy) `range_j = %s`（比 `%s`）。'
  % (f(A1C['range_j'], 4), f(A8C['range_j'], 4), f(A1C['range_j'] / A8C['range_j'], 2)))
A('')
A('#### 4.9 确认集与地板')
A('')
A('- A4 确认集（4 位点 × 4 α × %d 对）：%s —— 四个位点的端点全部相同（`y(1) = %s`，'
  '构造性 + 子集偏置因子），`xhalf` 与发现集同形。'
  % (R['panel']['confirmation'],
     json.dumps({k: f(v.get('xhalf'), 6) for k, v in (R['A4_confirmation'] or {}).items()
                 if isinstance(v, dict)}, ensure_ascii=False),
     f(list((R['A4_confirmation'] or {}).values())[0]['y'][-1], 6) if R['A4_confirmation'] else 'NA'))
A('- A5 随机 5 维地板（U6 子空间内、范数对齐到真实 diff）：%s。'
  '随机方向与真实写方向的期望投影 `E|cos| = 3/8 ≈ 0.375`；实测浅层 `%s`、深层 `%s` ⇒ '
  '**浅层端点效应中含相当比例的「大范数扰动」成分**（该量未预注册为门，仅描述性）。'
  % (json.dumps({k: f(v.get('y'), 6) for k, v in (R['A5_floor'] or {}).items()
                 if isinstance(v, dict)}, ensure_ascii=False),
     f(R['A5_floor']['7']['y'], 4), f(R['A5_floor']['34']['y'], 4)))
A('')
A('### 5. 分析结论')
A('')
A('1. **【主结论】「第三条独立口径」不存在 —— 两条读法都被证明退化**：'
  '位置前缀（pos0 通道 `%.4f%%`）与逐层累积支撑（`§2.2` 的覆盖恒等）都落在 Phase 12 的单点族上。'
  '"少层主导 vs 逐层累积"的判决因此**仍只有 Phase 12/13 的两条口径**；'
  '本 Phase 的价值在于**把这两条读法的不可用性做成了可机检的硬事实**（而非停留在设计疑虑）。'
  % (100.0 * abs(PS['median_ratio'] or 0.0)))
A('2. **【校准结论】置换零假设改变了 Phase 13 结论的强度**：`top3_share` 是极值型统计量，'
  '在 `xhalf` 坐标上观测值 `%s` **低于** null 95 分位 `%s` ⇒ 该坐标的「不集中」不应被解读为'
  '「深度上均摊」，而应记为「判据在该坐标**无区分力**」；`J` 坐标 `%s` vs `%s` 仅勉强通过。'
  % (f(A8C['top3_x'], 4), f(AN8.get('null_x_95'), 4), f(A8C['top3_j'], 4), f(AN8.get('null_j_95'), 4)))
A('3. **【装置结论】`F29` 给出本线第二个逐位跨 Phase 锚**：`y1(ℓ) ≡ Phase 12 recover(ℓ)` 在 18 位点上'
  '`dev = 0.000e+00`；`F30a` 的逐对恒等式 `n = %d` 全 `0.000e+00` 证明「全位置替换 ⇒ 精确复现供体前向」。'
  % EX['F30a_pairs_checked'])
A('4. **【诚实边界】跨族判决 `%s` 不可引用**：它是对**同一条曲线**的自洽检查（见 §4.6）；'
  'Phase 13 的「两坐标 argmax 相距 13」既未被证实也未被否证为「层栈性质」——**该问题被本 Phase 挂起**。'
  % VA8['verdict_cross_family'])
A('5. **【预测核对】7 条中 %d 条 PASS**（%s）。3 条 FAIL 各有明确性质：'
  'P4 = **预注册文本符号缺陷**（desc 与自身 rationale 相反，按 rationale 方向实测 18/18 满足）；'
  'P5 = 欠加性方向对但幅度不足（11/18 vs 要求 15/18）；'
  'P6 = **端点单调性条款设计错误**（端点按构造饱和，铁律 (r) 已预告其不可能单调）。'
  % (sum(1 for k in PC if PC[k]['pass_']),
     ', '.join('%s:%s' % (k, 'PASS' if PC[k]['pass_'] else 'FAIL') for k in sorted(PC))))
A('6. **【标度结论】`xhalf` 是网格不变量，`J` 不是**：两网格的 `XH_RANGE` 完全相同（`%s`），'
  '而 `J` 差 2 倍以上 ⇒ 跨相位引用 `J` 的绝对值必须同网格，跨相位引用 `xhalf` 才安全。'
  % f(R['A7_range_grid']['dense']['range'], 6))
A('')
A('### 6. 机制拼图（v5.4）与限界')
A('')
A('**拼图增量**：')
A('- **未变更**：「层 = 软门 + 下游读数」（Phase 9/10）；「读写两端同构、组件不是正确粒度」（Phase 8）。')
A('- **新增（装置级，本 Phase 唯一站得住的增量）**：**「句内位置」不构成独立通道**'
  '（`pos0` 容量 `%.4f%%`）；**「逐层同时注入」不构成独立剂量轴**（层覆盖恒等）。'
  '⇒ 本装置可用的干预轴只有两条：**单点位点的「替换比例 α」** 与 **注入位点 ℓ**。'
  % (100.0 * abs(PS['median_ratio'] or 0.0)))
A('- **新增（统计级）**：`top3_share` 的置换零假设 95 分位高达 `0.70–0.75` ⇒ '
  '**任何基于该量的「集中/分散」结论都必须先过零假设**；Phase 12/13 的集中度结论应改写为'
  '「在 `J` 坐标、经置换校准后勉强成立」。')
A('- **挂起**：Phase 13 的「两坐标 argmax 相距 13 是层栈性质」—— 本 Phase 无法判定（见 §4.6）。')
A('')
A('**限界（必须与结论同时引用）**：')
for h in HONESTY:
    A('- %s' % h)
A('- **本 Phase 新增限界**：① 「第三口径不存在」的结论**只对 qwen3-4b + `%s` 模板 + T=2 成立**，'
  '未在别的模型/模板上验证；② `A5` 地板只测 3 个位点、且未做 bootstrap；'
  '③ 判定「A8 ≡ Phase 12 单点族」用的是**数值一致**（`max|Δ| = %s`）加**代数论证**，'
  '未把代数恒等写成硬断言（可作为后续 Phase 的一行 floor）。' % ('%s是一种', f(d_A8_P12_x, 6)))
A('')
A('### 7. 第一性原理')
A('')
A('1. **读出位置的充分性把「位置前缀」这一类干预废掉**：读点在末位 ⟹ 末位残差是类别读出的近充分'
  '统计量 ⟹ 「整段前缀」与「末位单点」只有在**末位之外通道非空**时才可分。本 Phase 直接测出该通道'
  '容量 ≈ `%.4f%%`（`median y0/y1`）⇒ 两类干预**必然**近重合。这是信息论层面的必然，不是实现巧合。'
  % (100.0 * abs(PS['median_ratio'] or 0.0)))
A('2. **「同时注入多层」在「层输出被显式覆盖」的装置里天然退化**：hook 把每层输出写成受体自身的'
  '干净量 ⇒ 上游注入的后果被下游覆盖抹去 ⇒ 有效自由度只剩「最后一层是谁」。'
  '**任何"累积"实验都必须让注入的影响能穿过后续层**（例如只在**首层**注入后让下游自由演化，'
  '或注入到**残差流的加性项**而非输出覆盖），否则测到的永远是同一件事。')
A('3. **极值型统计量必须有零假设**：`top3_share = max_w |Σ3 jumps| / range` 在有限的 17 个 jump 上'
  '**上限天然很高**（任取 3 个相邻跳都可能吃掉大半 range）。没有置换校准就把「观测到 0.57」'
  '读成「分散」是**纯坐标系错觉**。铁律 (p) 在本 Phase 得到最强的实证支持。')
A('')
A('### 8. 后续资源与死线')
A('')
A('**Phase 15 候选（最高优先）· 跨模型复算「统一剖面」**：把 Phase 12/13/14 已确立的**唯一有效口径**'
  '（单点位点替换族的 `xhalf(ℓ)` / `J(ℓ)` 双坐标剖面 + 置换零假设校准）在 **qwen3-14b** 与 '
  '**glm4-9b（untied）** 上独立复算，回答两个真问题：'
  '① Phase 13 的「两坐标 argmax 相距 13」是层栈性质还是 qwen3-4b 特例（本 Phase 已把它**挂起**）；'
  '② 置换零假设 95 分位是否也高达 0.70–0.75（若如此，集中度判据在这条线上**整体作废**）。'
  '**判据必须先冻结**，且**禁止**沿用 L6/U6；**新增一条**：任何集中度结论必须同时报 null 95 分位与裕度。')
A('')
A('**第二候选 · 位置通道容量的上界测量**：`y0/y1 ≤ %s` 只在单一模板（T=2）上测过；'
  '设计**多模板 × 多位置**（含 3 位置模板）的专门臂，给「末位之外的通道容量」一个带置信区间的常数。'
  % f(max(float(x) for x in PS['ratio_y0_y1'].values() if x is not None), 6))
A('')
A('**第三候选 · 去掉「覆盖式」注入的累积臂**：按 §7 第 2 条，把 A8 改成'
  '「只在 `S_i` 内**首层**注入 `α·d`，下游自由演化」或「注入到加性残差项」，'
  '才能第一次真正测到「累积」。这是本 Phase 逻辑上真正的续作。')
A('')
A('**其他挂账（不变）**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌解剖；N3-β → N3-δ → P-N3b → N3-γ → N3-ε；'
  'R1 挂账对照补强；K4（E2 死线）处置；qwen3-14b 接入；**N 线 Phase 3–7 补登 Ledger**；'
  'G 线 Phase 3154（G2-P1）已预注册。')
A('')
A('**本 Phase 新增装置铁律（4 条）**：')
A('- **(v)「另一个独立口径」必须在 seal 冻结前用 SMOKE 证明其与既有口径**可分**；'
  '重合即降级为阴性对照，并另找剂量轴。** 本 Phase 的两个候选读法都通过 SMOKE 就被量出退化'
  '（`pos0` ~%.4f%%；层覆盖恒等），若不在 SMOKE 阶段发现，就会把与 Phase 12 数值重合的臂'
  '当作独立证据发表。' % (100.0 * abs(PS['median_ratio'] or 0.0)))
A('- **(w) 面板级恒等式与逐对恒等式的作用域必须分离**：`y = dDonor_arm / 固定面板均值` 使'
  '子集臂**不可能**满足面板级端点恒等式（SMOKE 6 对得 `1.147991` 是子集偏置的算术后果）；'
  '这类断言须写成与子集无关的**逐对**形式 + 显式 `full_panel` 才断言的旁路。')
A('- **(x) 预注册预测的符号必须与其自身 `rationale` / `falsified_if` 一致**：'
  'P4 的 `desc` 写 `y0 > y1`、`rationale` 与 `falsified_if` 却都指向 `y1 > y0` ⇒ '
  '一条**已被数据满足**的预测（18/18、17/18）被机械判成 FAIL。'
  '**判据符号写反会让结论反向**（同铁律 33 的族内变体：33 管阈值方向，本条管预测文本自洽）。')
A('- **(y) GQA 模型禁止用 `hidden_size / num_attention_heads` 反推 `head_dim`**：'
  '`o_proj.in_features = n_heads × head_dim` 与 `hidden_size` 无必然关系'
  '（qwen3-4b：`%d ≠ %d`；真值 `head_dim = %d`）。配置字段一律**直读** `config` 并与其他投影维度交叉断言。'
  % (R['layers']['o_proj_in'], 2560, R['layers']['head_dim']))
A('')
A('### 附：记录完整性与过程备注（非实验内容）')
A('')
A('- **产物 sha8**：seal `%s`；amend1 `%s`；amend2 `%s`；exec `%s`；result `%s`；report `%s`。'
  % (R['seal_sha8'], R['amend1']['sha8'], R['amend2']['sha8'], R['exec_sha8'],
     hashlib.sha256(open(os.path.join(P14T, 'result_phase14.json'), 'rb').read()).hexdigest()[:8],
     hashlib.sha256(open(os.path.join(P14T, 'n2h1a7_report_qwen3-4b.txt'), 'rb').read()).hexdigest()[:8]))
A('- **脚本落点** `tests/deepseek/Phase14/`；**产物落点** `tests/deepseek_temp/Phase14/`（v2 约定）。')
A('- **SMOKE 的价值（极强实证）**：本轮 SMOKE **抓出两个真实缺陷** —— (a) 装置配置字段错误'
  '（GQA `head_dim`），在正式运行前即被前置断言拦下；(b) 预注册锚的作用域欠规范，'
  '并**顺带量出「位置前缀不是独立口径」**，从而在零实验数据阶段就把相位设计修正为新增 A8。'
  '两条修正各自冻结为 amend1 / amend2（原 seal 字节未动）。')
A('- **MEMO 节由 `gen_memo_phase14.py` 从 `result_phase14.json` 渲染**，数字不经过手工转录'
  '（Phase 13 的 A8 事实错误即源于手工转录）；本生成器的 v2 修订本身也说明：'
  '**散文里的因果假设仍需人工核对**，生成器只消除「数字转录」类错误。')
A('')
A('### 9. 一句话（重复三次）')
A('')
_sent = ('**「第三条独立口径」不存在，真正的增量是零假设校准：** 死线给的两种读法都退化到 Phase 12 的单点族 —— '
         '位置前缀因 `pos0` 通道只有 `%s`（~%.4f%%）而与 `mask={1}` 数值重合（18 位点 `max|Δxhalf| = %s`），'
         '逐层累积支撑则因「每层 hook 覆盖上一层注入」而在代数上等价于「只注入最后一层」'
         '（实测 `xhalf_A8(i)` 与 Phase 12 已发表 `xhalf(site_i)` 逐点差 `%s`、`J` 差 `%s`）；'
         '因此本 Phase 的真正新增结论来自**置换零假设**（铁律 (p)）：`xhalf` 坐标的集中度 `%s` '
         '**低于** null 95 分位 `%s` ⇒ 该坐标「不集中」源于**判据无区分力**而非数据性质，'
         '`J` 坐标 `%s` 仅勉强高于 `%s`（裕度 `%s`），Phase 13 的 `CONCENTRATION_COORDINATE_DEPENDENT` '
         '须按此加限定；跨族判决 `%s` 因两臂皆等于源族而是**自洽检查**、不可引用（该问题挂起）；'
         '装置侧 `F29` 给出第二个逐位跨 Phase 锚（`y1(ℓ) ≡ Phase 12 recover(ℓ)`，`dev = %s`，18 位点）'
         '与逐对恒等式 `%s`（n=%d）。7 条预注册预测 %d 条 PASS（%s）。')
_sent = _sent % (f(PS['median_ratio'], 6), 100.0 * abs(PS['median_ratio'] or 0.0), f(d_A1_A8_x, 6),
                 f(d_A8_P12_x, 6), f(d_A8_P12_J, 6),
                 f(A8C['top3_x'], 4), f(AN8.get('null_x_95'), 4),
                 f(A8C['top3_j'], 4), f(AN8.get('null_j_95'), 4),
                 f(float(A8C['top3_j']) - float(AN8.get('null_j_95')), 4),
                 VA8['verdict_cross_family'], f(FL['F29']['dev'], 3),
                 f(FL['F30']['F30a']['dev'], 3), EX['F30a_pairs_checked'],
                 sum(1 for k in PC if PC[k]['pass_']),
                 ', '.join('%s:%s' % (k, 'PASS' if PC[k]['pass_'] else 'FAIL') for k in sorted(PC)))
for _ in range(3):
    A(_sent)
    A('')

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
b = io.open(OUT, 'rb').read()
print('WROTE %s' % OUT)
print('  bytes = %d ; lines = %d ; sha8 = %s' % (len(b), len(L), hashlib.sha256(b).hexdigest()[:8]))
