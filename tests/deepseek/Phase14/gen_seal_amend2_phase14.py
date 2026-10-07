# -*- coding: utf-8 -*-
"""
Phase 14 / N2h1-alpha-7 : seal amend2 —— SMOKE 触发的两项修正（正式运行前冻结）
=============================================================================
(1) F29/F30 的【作用域】欠规范（并非归一化错误）：
    y = dDonor_arm / FULL_SWAP，其中 dDonor_arm 是【该臂 pair 集】的均值。故 F29/F30 是
    【面板级】恒等式：只在臂的 pair 集 == Phase 12 的 24 个 discovery 对时成立。
    SMOKE 把 pair 集压到 order[:6] = ['苹果','香蕉','梨','西瓜','狗','猫']（水果类主导），
    其供体自身增量均值 = 12.3953125，而 FULL_SWAP = 10.797395833333333
    => ratio = 1.147991，与 SMOKE 实测 recover_p(L6) 完全一致。
    **这是子集偏置的必然结果，不是装置缺陷**（审计逐位复核，见 evidence）。
    修正：把 F29/F30 显式标注为「非 SMOKE」断言；另立一条与子集无关的【逐对恒等式】锚。
(2) 口径冻结澄清：FULL_SWAP ≡ mean_{24 个 discovery 受体词} FULL_SWAP_pairs[·] = 10.797395833333333。
    `FULL_SWAP_pairs` 是 41 键字典（含确认集 17 个键），但 Phase 12 只对 `order` 的 24 项取均值；
    41 项均值 = 10.684756097560975【不相等】。本 Phase 冻结沿用 24 项口径，不得混用。
(3) 新增 A8 逐层累积层代换：死线标题「逐层累积代换」的字面含义是【逐步把更多层的供体增量换进来】。
    SMOKE 已实证位置前缀（A1）的 pos0 贡献 ~0.07%（y0/y1 = 6.7e-4）=> A1 与 Phase 12 单点族数值近重合，
    不构成独立第三口径；A8 才是「累积」的正确剂量轴。

原 seal sha8 074fc963；amend1 sha8 0b0276b8。不改动两者字节。
输出：tests/deepseek_temp/Phase14/N2h1a7_design_seal_amend2.json
"""
import os, io, json, hashlib, time
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')


def sha(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest()


SEAL = os.path.join(P14T, 'N2h1a7_design_seal.json')
AM1 = os.path.join(P14T, 'N2h1a7_design_seal_amend1.json')
OUT = os.path.join(P14T, 'N2h1a7_design_seal_amend2.json')

R12 = json.load(io.open(os.path.join(P12T, 'result_phase12.json'), encoding='utf-8'))
FS = R12['FULL_SWAP_pairs']
ORDER = list(R12['E2']['6'][0]['order'])
M24 = float(np.mean([FS[x] for x in ORDER]))
M41 = float(np.mean([FS[k] for k in FS]))
SUB6 = list(ORDER[:6])
M6 = float(np.mean([FS[x] for x in SUB6]))
assert M24 == float(R12['FULL_SWAP'])

amend = {
    'phase': 14,
    'kind': 'schema_amend + arm_addition (no hypothesis change)',
    'name': 'N2h1-alpha-7 amend2：F29/F30 作用域界定 + FULL_SWAP 口径冻结 + 新增 A8 逐层累积层代换',
    'amend_of_seal_sha8': sha(SEAL)[:8],
    'amend_of_seal_sha256': sha(SEAL),
    'amend1_sha8': sha(AM1)[:8],
    'trigger': 'SMOKE（正式运行前，零实验数据；SMOKE 只跑 3 位点 x 6 alpha x 6 对）',

    'fix_1_anchor_scoping': {
        'issue': (
            'F29/F30 写成「y1(ell) == Phase12 recover(ell)」「y01(ell) == 1.0」，未标明它们是'
            '【面板级】恒等式：y 的分母 FULL_SWAP 是【24 个 discovery 对】的均值，'
            '而分子 dDonor_arm 是【该臂 pair 集】的均值。pair 集一变，比值即变。'
        ),
        'smoke_evidence': {
            'smoke_pair_set': SUB6,
            'mean_donor_self_increment_on_smoke_subset': M6,
            'FULL_SWAP': M24,
            'ratio': M6 / M24,
            'smoke_observed_recover_p_L6': 1.147990931455,
            'verdict': 'ratio 与 SMOKE 实测逐位一致 => 1.147991 是子集偏置的必然结果，不是装置缺陷',
        },
        'F30_new': (
            'F30（两条并列）：'
            '(a) 与子集无关的【逐对恒等式】：alpha=1 且 mask={0,1} 时，'
            'per_pair_b(ell, alpha=1) 须等于 FULL_SWAP_pairs[受体词_b]'
            '（相对误差 <= 1e-4，对 18 位点 x 24 对全查）；SMOKE 亦须通过；'
            '(b) 【面板级】y01(ell) 在 18 位点上为常数（极差 <= 1e-12），且 == 1.0（|d| <= 1e-9）；'
            '仅在臂的 pair 集 == 24 个 discovery 对时断言（SMOKE 标 skipped）。'
        ),
        'F29_new': (
            'F29：mask={1} alpha=1 的 y1(ell) 须等于 Phase 12 recover(ell)（18 位点，|d| <= 1e-9）；'
            '仅在臂的 pair 集 == 24 个 discovery 对时断言（SMOKE 标 skipped）。'
        ),
    },

    'fix_2_normalization_freeze': {
        'FULL_SWAP_definition_frozen': 'mean over the 24 discovery recipient words of FULL_SWAP_pairs[·]',
        'value': M24,
        'mean_over_all_41_keys': M41,
        'note': (
            'FULL_SWAP_pairs 是 41 键字典（41 = 24 发现 + 17 确认，以【受体词】为键），'
            '但 Phase 12 的 FULL_SWAP 只对 `order` 的 24 项取均值。41 项均值不相等。'
            '本 Phase 一律沿用 24 项口径；A4 确认集臂的分母【仍】用 FULL_SWAP（不另立分母），'
            '故 A4 的 y 端点不必然为 1（这一条写进 honesty）。'
        ),
    },

    'fix_3_new_arm': {
        'arm': 'A8_cumulative_layer_substitution',
        'definition': (
            'SITES = [6,7,8,9,10,11,12,14,16,18,20,22,24,26,28,30,32,34]；'
            '对支撑序 i = 0..17，令 S_i = SITES[0..i]，在【每个】layers[j] (j in S_i) 的 forward-hook '
            '上把【末位】写入 h_j + alpha * d_j（h_j/d_j = 供体/受体的层输出末位残差与其差）。'
            'i 增大 => 干预支撑逐层扩大 => y_A8(i, alpha) 即【累积贡献曲线】。'
        ),
        'why': (
            '死线标题为「逐层累积代换」；其字面含义是逐步把更多层的供体增量换进来。'
            'SMOKE 实证 A1（位置前缀）的 pos0 贡献仅 ~0.07%（y0/y1 = 6.7e-4）'
            '=> A1 与 Phase 12 单点族数值近重合，不构成独立第三口径。A8 才是「累积」的正确剂量轴。'
        ),
        'dose_axis': '支撑序 i = 0..17（离散等距，无量纲）；每个 i 内仍有 alpha in [0,1]',
        'grid': {'i': '全部 18 个前缀（含 i=0 = 单层 L6，与 Phase 12 的单点族在 L6 重合）',
                 'alpha': 'AL_LEG（14 点，沿用 Phase 12 同网格以控时）', 'pairs': 24},
        'fwd': 18 * 14 * 24,
        'statistics': 'xhalf_A8(i) 与 J_A8(i) 定义与 A1 完全一致（cross_alpha frac=0.5；J_only，JFL=0.01）',
        'endpoint_declaration': (
            'alpha=1 时 i=17 把末位在全部 18 个位点上都换成供体增量；但 pos0 仍是受体，'
            '故 y_A8 的端点【不】由构造等于 1（与 A1 的 y01 不同）。端点量仍只作正向判据，'
            '主量是形状量 xhalf_A8 / J_A8。'
        ),
        'predictions_for_A8': {
            'P6': 'y_A8(i, alpha=1) 随 i 单调不减；且 i=0（单层 L6）即已达 >= 0.95（端点近饱和）',
            'P7': 'xhalf_A8 的 argmax 窗口众数 in {13,14}（深尾），bootstrap 频次 >= 0.50',
        },
        'cross_family_use': (
            'A8 的 (top3_share, argmax_window) 二元组在 J 与 xhalf 两坐标上各报一份，'
            '与 Phase 13 的靶值（xhalf=14 / J=1）做同一套跨族迁移判据；'
            'A1 与 A8 各自独立给一份判决，不得合并。'
        ),
    },

    'budget_update': {'old_expected_fwd_total': 10220, 'new_expected_fwd_total': 16268,
                      'delta': {'A8_cumulative_layer': 18 * 14 * 24},
                      'note': 'A8 用 AL_LEG（14 点）；A1 仍用 AL_DEN（18 点）供 XH_RANGE 限界检验。'},

    'what_is_NOT_changed': [
        '预注册 P1-P5 —— 不变；仅新增针对新臂的 P6/P7',
        '同坐标判决表 6 行、跨族判决表 4 行 —— 不变（A1/A8 各用一次）',
        'A1/A2/A3a/A3b/A4/A5 的定义、网格、统计量 —— 全部不变',
        'inheritance anchors —— 不变',
    ],
    'added_honesty_12': (
        '12. A1（位置前缀）经 SMOKE 预判与 Phase 12 单点族数值近重合（pos0 贡献 ~0.07%），'
        '故本 Phase 的「第三独立口径」由 A8（逐层累积层代换）承担；A1 作为「首位置通道近乎为空」的'
        '阴性对照报告。两条臂的判决分别陈述，不得混为一条证据。'
    ),
    'added_honesty_13': (
        '13. A4 确认集臂的 y 分母仍用 24 项 FULL_SWAP（不另立分母），故 A4 的端点量不必然为 1，'
        'A4 只报与发现集同号性，不参与 F29/F30 断言。'
    ),
    'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}

with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(amend, f, ensure_ascii=False, indent=1)
    f.write('\n')

b = io.open(OUT, 'rb').read()
print('WROTE %s' % OUT)
print('  bytes = %d ; sha8 = %s' % (len(b), hashlib.sha256(b).hexdigest()[:8]))
print('  M24 = %.15f (== R12 FULL_SWAP: %s)' % (M24, M24 == float(R12['FULL_SWAP'])))
print('  M41 = %.15f (equal? %s)' % (M41, M41 == M24))
print('  M6(smoke subset) = %.15f ; ratio = %.6f' % (M6, M6 / M24))
