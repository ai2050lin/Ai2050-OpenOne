# -*- coding: utf-8 -*-
"""
Phase 14 / N2h1-alpha-7 : 逐层累积全位点代换（cumulative prefix substitution）
=============================================================================
生成预注册设计 seal：tests/deepseek_temp/Phase14/N2h1a7_design_seal.json
【观测前冻结】。本脚本只读 Phase 11/12/13 的【已发表】产物来确定继承锚与预测依据，
不运行任何前向、不产生任何新统计量。

Phase 13 §8 死线原文（最高优先）：
  「把受体句整段前缀残差逐步替换为供体（positions[0..t] 全部替换，而非单点），
    测「累积贡献曲线」是否仍为软阶跃。这是「少层主导 vs 逐层累积」的第三条独立口径。
    必须先做的设计修正：任何集中度判据必须在 J 与 xhalf 两个坐标上同时报告，
    并同时报告 argmax 窗口位置。」

可行性探针（tests/deepseek/Phase14/_feas_probe.py，本 seal 之前跑完）买到的三条装置事实：
  (F-A) TMPL='%s是一种' 对全部 41 实例都恰好 tokenize 成 T=2：
        pos0 = 实例词（苹果/卡车/...），pos1 = '是一种'。
        => 「整段前缀」= {pos0, pos1}；Phase 12 的单点 = {pos1}。前缀族严格是单点族的超集。
  (F-B) 费用 0.0295 s/fwd（全位点 patch 与单点 patch 等价：2 个位置 vs 1 个位置，序列长 2）。
  (F-C) 三条锚全部【逐位】成立：
        FULL_SWAP 从 Phase 12 落盘 per-pair 重算 == 10.797395833333333（bit-equal）；
        mean||P_U6(diff6)|| == 17.06125152401808（bit-equal，Phase 9 参照）；
        alpha=0 全位点 patch 还原基线 0.000e+00；
        alpha=1 全位点 patch == 供体自身前向 0.000e+00。
  (F-C3) 关键设计后果：alpha=1 的全位点替换在【每一层】都精确等于供体前向
        => 端点量 recover_p(ell) == 1.0 对全部 18 位点按构造成立
        => 端点量【完全不带深度信息】，主量必须是形状量（铁律 (r) 的又一次应用）。
"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P11T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11')
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
P13T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
os.makedirs(P14T, exist_ok=True)
SEAL_OUT = os.path.join(P14T, 'N2h1a7_design_seal.json')


def sha256(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest()


def sha8(p):
    return sha256(p)[:8]


def jload(p):
    return json.load(io.open(p, encoding='utf-8'))


E12 = jload(os.path.join(P12T, 'execution_phase12.json'))
R12 = jload(os.path.join(P12T, 'result_phase12.json'))
R13 = jload(os.path.join(P13T, 'result_phase13.json'))
R11 = jload(os.path.join(P11T, 'result_phase11.json'))

P12_RES = os.path.join(P12T, 'result_phase12.json')
P13_RES = os.path.join(P13T, 'result_phase13.json')
P12_EXEC = os.path.join(P12T, 'execution_phase12.json')

MDIR = os.path.join(ROOT, 'models', 'hf', E12['model'])
CFG_SHA = sha256(os.path.join(MDIR, 'config.json'))

SITES = [int(s) for s in R12['xhalf']['sites']]
PROFILE = [int(s) for s in E12['profile_sites']]
assert SITES == PROFILE, 'site 列表必须一致'
N_S = len(SITES)                     # 18
N_ADJ = N_S - 1                      # 17
W = 3
N_WIN = N_ADJ - W + 1                # 15
N_DISC = 24
N_CONF = 17

# 继承的已发表量（Phase 12/13）
XH_12 = {int(k): float(v) for k, v in R12['xhalf']['curve'].items()}
J_12 = {int(k): float(v['jump_ratio']) for k, v in R12['profile_swap'].items()}
REC_12 = {int(k): float(v['y'][-1]) for k, v in R12['profile_swap'].items()}
XH_RANGE_12 = float(R12['xhalf']['range'])
XH_JUMPS_12 = [float(x) for x in R12['xhalf']['jumps']]
TOP3_12 = float(R12['xhalf']['top3_share_x'])
FULL_SWAP = float(R12['FULL_SWAP'])
SING_U6 = [float(x) for x in R12['subspace']['sing_U6']]
Q_ELL_12 = {int(k): float(v) for k, v in R12['dose_coord']['q_ell'].items()}
N6_REF = float(E12['mean_n6_ref_from_phase9'])
N6_TOL = float(E12['n6_drift_tol'])
MODE_X_13 = int(R13['extra']['mode_x'])
MODE_J_13 = int(R13['extra']['mode_j'])
P_FEW_X_13 = float(R13['extra']['P_few_x'])
P_FEW_J_13 = float(R13['extra']['P_few_j'])
SHARE_X_13 = float(R13['A4_concentration']['xhalf']['hat'])
SHARE_J_13 = float(R13['A4_concentration']['J']['hat'])
SITES_ARR = SITES

ALPHA_LEGACY = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
ALPHA_DENSE = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
               0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
assert set(ALPHA_LEGACY) <= set(ALPHA_DENSE)
ALPHA_POS = list(ALPHA_DENSE)        # 位置因子臂也走 18 点网格
CONF_ALPHA = [0.25, 0.5, 0.75, 1.0]

seal = {
    'phase': 14,
    'name': 'N2h1-alpha-7 / cumulative prefix (all-position) substitution + dual-coordinate concentration report',
    'one_sentence': (
        '把 Phase 8-13 全部探针族共有的「只动末位」这一前提打破：在 18 个剖面位点上把受体 '
        'positions{0,1}（整段前缀，T=2 已由探针确证）的残差同时替换为供体，得到第三条独立口径的 '
        'J_p(ell)/xhalf_p(ell)；先由构造证明 recover_p(ell)=1.0 对 18/18 位点成立（端点量全退化），'
        '再把「少层主导 vs 逐层累积」的判决写成 (top3_share, argmax_window) 二元组在 J 与 xhalf 两坐标上的'
        '一致性，并与 Phase 13 已发表的两坐标 argmax 窗口（xhalf=14 深尾 / J_swap=1 浅端）做跨族迁移检验。'
    ),
    'motivation': {
        'deadline_source': 'Phase 13 §8「Phase 14 候选（最高优先）· 逐层累积代换（prefix swap）+ 双坐标集中度报告」',
        'why_third_family': (
            'Phase 11 的 J(ell)（固定基注入族）与 Phase 12 的 J_swap(ell)（单点替换族）都只把受体【末位】的'
            '残差换成供体。两条族给出「随深度单调变软」的一致图像，但**共享同一个结构假设**：干预只落在读出位点上。'
            '本 Phase 去掉该假设：同时替换 pos0（实例词本身）与 pos1（框架词「是一种」），'
            '干预落在**整段残差流**上。若结论仍成立，则「逐层累积」不是「只动末位」的产物。'
        ),
        'why_position_factorial': (
            'T=2 使「累积前缀曲线」恰为两级：t=0 -> {pos0}，t=1 -> {pos0,pos1}。'
            'Phase 12 从未测过 pos0，故 y0(ell)（首位置单独承载的可交换类别信号）是本 Phase 唯一'
            '真正的新测量量，直接回答「is-a 的载体在句内哪个位置、随深度如何变化」。'
        ),
        'new_observation_motivating_the_dual_coordinate_fix': (
            'Phase 13 实测：xhalf 的集中度 argmax 窗口 = 14（L28->L34，深尾，freq 0.7505，P(>=0.60)=0.379）'
            '而 J_swap 的 argmax 窗口 = 1（L7->L10，浅端，freq 0.739，P(>=0.60)=0.973），两窗口相距 13。'
            '该「坐标系依赖」是否只是单点族特有的？本 Phase 用第三条族直接检验。'
        ),
    },
    'model': {
        'name': E12['model'],
        'dir': os.path.relpath(MDIR, ROOT),
        'config_sha256': CFG_SHA,
        'n_layers': 36, 'hidden': 2560, 'n_heads': 32, 'head_dim': 80,
        'tie_word_embeddings': True,
        'template': E12['template'],
        'dtype': 'bfloat16',
        'attn_implementation': 'eager',
        'device': 'cuda (RTX 5080)',
    },
    'object': {
        'panel': {
            'discovery_instances': N_DISC,
            'confirmation_instances': N_CONF,
            'instances_all': 41,
            'pairs_all': 41,
            'classes': list(E12['classes']),
        },
        'template_tokenization': {
            'verified_by': 'tests/deepseek/Phase14/_feas_probe.py (pre-seal)',
            'distinct_T': [2],
            'pos0': '实例词（苹果/卡车/...），token id == tok.encode(w)[0]',
            'pos1': '框架词「是一种」',
            'phase12_intervention': '{pos1}（单点，末位）',
            'phase14_intervention': '{pos0, pos1}（整段前缀）',
        },
        'inheritance': {
            'capture_quantities_recomputed_not_inherited': [
                'U6 子空间（discovery 24 实例、末位 hidden_states[7]）',
                'BASE[rw].sd0 / sr0 / rd0（受体自身前向，零额外前向）',
                'FULL_SWAP（= mean_{41} FULL_SWAP_pairs，供体自身前向）',
            ],
            'inherited_for_anchors_only': [
                'XH_12 / J_12 / REC_12 / XH_RANGE_12 / XH_JUMPS_12 / TOP3_12 = Phase 12 已发表',
                'MODE_X_13=14 / MODE_J_13=1 / P_FEW_*_13 / SHARE_*_13 = Phase 13 已发表',
                'SING_U6 / Q_ELL_12 / N6_REF = Phase 8/9/12 已发表',
            ],
            'no_numeric_cross_family_compare': 'J_p 与 J_swap 只在【同一 legacy 14 点网格】上比秩；不比绝对值',
        },
    },
    'intervention': {
        'definition': (
            'h_ell_recip[pos] + alpha * ( h_ell_donor[pos] - h_ell_recip[pos] )  '
            'for pos in MASK, 在 layers[ell] 的 forward-hook（输出）上一次写入；'
            'ell == "R" 时改在最终 RMSNorm 输出的 hook 上写。'
        ),
        'mask_semantics': {
            '{0}': '只替换首位置（实例词）',
            '{1}': '只替换末位（框架词）—— 与 Phase 12 单点口径完全同构',
            '{0,1}': '整段前缀（本 Phase 主臂）',
            '{}': '不干预（= capture 基线）',
        },
        'alpha_semantics': 'alpha 属于 [0,1] 为替换比例（无量纲，主坐标）；alpha=1 即 hd[pos]（精确，F28/F29/F30）',
        'row_sites': 'PROFILE = 全部 18 个位点 = [6,7,8,9,10,11,12,14,16,18,20,22,24,26,28,30,32,34]（沿用 Phase 12，不含末层 35）',
        'readout_site': 'R = 最终 RMSNorm 输出（同 Phase 12）',
        'degeneracy_declared': (
            'alpha=1 且 mask={0,1} => 该层输出的【全序列】残差 == 供体 => 下游计算与供体前向完全一致 '
            '=> y_p(ell, alpha=1) == 1.0 对 18/18 位点成立（探针已逐位确认 0.000e+00）。'
            '故主量【必须】是曲线形状量 xhalf_p / J_p，端点量 recover_p 仅作正向判据 F30（铁律 (r)）。'
        ),
    },
    'dose_coordinate': {
        'alpha_grid_legacy': ALPHA_LEGACY,
        'alpha_grid_dense': ALPHA_DENSE,
        'alpha_grid_conf': CONF_ALPHA,
        'y_definition': 'dDonor_p / FULL_SWAP，dDonor_p = mean_pairs[ score_of(donor_logits_patched, donor_sup, donor_sid) - BASE[rw].sd0 ]',
        'xhalf_frac': 0.5,
        'jdose_floor': 0.01,
        'why_two_grids': (
            'J_p 的跨族比较【必须】与 Phase 12 用同一 14 点网格（否则斜率刻度不可比）；'
            'xhalf 的 XH_RANGE 限界检验（回应 Phase 13 限界③）另用 18 点加密网格。'
            '=> 两次报告分别标注 grid=legacy / grid=dense。'
        ),
    },
    'statistics_definition': {
        'xhalf': 'cross_alpha(x, y, frac=0.5)：y 首次达 0.5*max(y) 的 alpha（线性插值，不假设单调）——逐字复制 Phase 12/13',
        'J': 'J_only：max(相邻斜率)/median(其余斜率)，仅用 alpha >= 0.01 的相邻段 ——逐字复制 Phase 12/13',
        'J_iqr': '同 J_only，分母改 IQR(其余斜率) ——逐字复制 Phase 13（限界：N_dec 统计量依赖）',
        'xhalf_p_range': 'XH_RANGE_p = max_ell xhalf_p - min_ell xhalf_p（极差，同 Phase 12 xhalf.range 定义）',
        'concentration': 'conc_hat(F) = max over W=3 windows of |sum of 3 adjacent jumps| / range(F)，15 个窗口；窗口 idx w 覆盖 jumps[w..w+2] <=> 位点 sites[w] -> sites[w+3]',
        'two_coordinates': {
            'coord_X': 'F = xhalf_p(ell) over 18 sites',
            'coord_J': 'F = J_p(ell) over 18 sites',
            'both_must_report': ['top3_share (hat)', 'argmax_window (hat)', 'P(>=0.60)', 'P(<=0.40)', 'bootstrap window histogram', 'null 95th percentile of top3_share'],
        },
        'position_factorial': {
            'y0(ell)': 'y(ell, alpha=1, mask={0})',
            'y1(ell)': 'y(ell, alpha=1, mask={1})  —— 必须逐位等于 Phase 12 recover(ell)（F29）',
            'y01(ell)': 'y(ell, alpha=1, mask={0,1}) == 1.0（F30）',
            'additivity': 'S(ell) = y0(ell) + y1(ell) - y01(ell)；S > 0 => 次可加（饱和）；S < 0 => 超可加',
        },
    },
    'curve_classifier': {
        'logistic_k_min': 1.0, 'logistic_k_max': 60.0, 'logistic_k_step': 1.0,
        'logistic_x0_step': 0.005,
        'note': '仅在点估计上拟合（不做 bootstrap），沿用 Phase 12 的 classifier 参数',
        'UNREACH_y': 0.15,
    },
    'arms': {
        'A0a_full_swap_rebuild': {'what': '从 Phase 12 落盘 FULL_SWAP_pairs 重算 mean，须 bit-equal Phase 12 FULL_SWAP', 'fwd': 0},
        'A0b_n6_rebuild': {'what': '重算 mean||P_U6(diff6)||（discovery 24 对，末位），须 == Phase 9 参照 17.06125152401808', 'fwd': 0},
        'A0c_u6_rebuild': {'what': 'U6 奇异值须 == Phase 8 锚', 'fwd': 0},
        'A0d_noop_alpha0': {'what': 'alpha=0 全位点 patch == capture 分数（3 位点 + R）', 'fwd': 96},
        'A1_prefix_layer_sweep': {'what': 'mask={0,1}，18 位点 x 18 alpha(dense) x 24 对', 'fwd': 7776, 'main': True},
        'A2_readout_prefix': {'what': 'R 位点 mask={0,1}，14 alpha(legacy) x 24 对', 'fwd': 336},
        'A3a_position_curves_L6': {'what': 'L6，mask in {{0},{1}}，18 alpha(dense) x 24 对', 'fwd': 864},
        'A3b_position_endpoints': {'what': '18 位点，mask in {{0},{1}}，alpha=1，24 对  => y0(ell), y1(ell)', 'fwd': 864},
        'A4_confirmation': {'what': 'mask={0,1}，4 位点 [7,11,20,34] x 14 alpha x 17 确认对', 'fwd': 952},
        'A5_floor': {'what': 'mask={0,1}，alpha=1，3 位点，随机 5 维方向（||xi|| = mean||diff_ell||），24 对', 'fwd': 72},
        'A6_bootstrap': {'what': '在 A1 落盘 per_pair 上重采样（BS=2000）+ 集中度置换零假设（2000）', 'fwd': 0, 'cpu': True},
        'A7_statistic_alt': {'what': 'J_iqr 分母 + xhalf 在两套网格上的 XH_RANGE', 'fwd': 0, 'cpu': True},
    },
    'metrics': {
        'primary': ['xhalf_p(ell)', 'J_p(ell)', 'y0(ell)', 'y1(ell)', 'y01(ell)'],
        'concentration_both_coords': ['top3_share_x', 'argmax_w_x', 'P_ge_060_x', 'P_le_040_x',
                                      'top3_share_j', 'argmax_w_j', 'P_ge_060_j', 'P_le_040_j'],
        'cross_family_transfer': ['d_x = |argmax_w_x - 14|', 'd_j = |argmax_w_j - 1|'],
        'position': ['median_ell y0/y1', 'S(ell)', 'frac_sites_y0_positive'],
        'robustness': ['XH_RANGE_p(legacy)', 'XH_RANGE_p(dense)', 'rho_Jp_vs_Jswap', 'N_dec_like_J_p (paired, 沿用 Phase 13 口径)'],
    },
    'bootstrap': {
        'seed': int(E12['seed']),
        'BS': 2000,
        'scheme': 'resample discovery pair indices idx = rng.integers(0, nP, nP)；y_b(ell,alpha) = mean(per_pair[idx]) / mean(FS_VEC[idx])（与 Phase 13 完全同构）',
        'paired_delta': 'Delta_b(w) = F_b(sites[w]) - F_b(sites[w+3])，同一 idx（沿用 Phase 13）',
        'concentration_null': '把 17 个 jump 值随机重排到 17 个相邻对上（2000 次），重算 top3_share => null 分布；报 95 分位',
        'note': 'null 检验「jump 幅度是否携带位置信息」，与 bootstrap 带互补（铁律 (p)）',
    },
    'pre_registered_predictions': {
        'P1': {
            'desc': 'xhalf_p 的 argmax 窗口众数 in {13,14}（深尾），bootstrap 频次 >= 0.50',
            'rationale': 'Phase 13 已发表 xhalf argmax=14（freq 0.7505）。前缀族是单点族的超集，形状应基本继承。',
            'falsified_if': '众数 in {0..12} 且频次 > 0.50',
        },
        'P2': {
            'desc': 'J_p 的 argmax 窗口众数 in {0,1,2}（浅端），bootstrap 频次 >= 0.50',
            'rationale': 'Phase 12 J_swap 在 L7 取极大（27.11）并单调降至 L34（1.06）；J 由 L7->L10 的巨大落差支配，任何族都应继承。',
            'falsified_if': '众数 >= 3 且频次 > 0.50',
        },
        'P3': {
            'desc': 'spearman(J_p(ell), J_swap(ell)) over 18 sites >= 0.50',
            'rationale': 'Phase 12 J_swap 随深度近似单调下降（Spearman 与位点序强负相关）；前缀族应同序。',
            'falsified_if': 'rho < 0.50',
        },
        'P4': {
            'desc': '位置因子：y0(ell) > y1(ell) 在【至少 15/18】位点成立，且 y0(ell) > 0 在【至少 12/18】位点成立',
            'rationale': '读点在末位；pos1 替换直接改动读出位置，pos0 只能经下游注意力间接影响 => y1 应普遍大于 y0。但 pos0 非零（否则首位置通道为空）。',
            'falsified_if': 'y0 >= y1 的位点数 >= 4 或 y0 <= 0 的位点数 >= 7',
        },
        'P5': {
            'desc': '次可加：S(ell) = y0 + y1 - y01 > 0 在【至少 15/18】位点成立（两臂部分激活同一饱和读出，局部效应之和越过上限 1）',
            'rationale': 'y1(L6)=0.9963 已近饱和（Phase 12 已发表）；若 y0 亦非零，则 y0 + y1 > 1 = y01。',
            'falsified_if': 'S <= 0 的位点数 >= 4',
        },
    },
    'decision': {
        'gate_G0p': 'F24 ∧ F25 ∧ F26 ∧ F27 ∧ F28 ∧ F29 ∧ F30 ∧ F31（装置锚全通过）',
        'verdict_table_same_coordinate': [
            ['G0p fail', 'DEVICE_ANCHOR_FAILED'],
            ['max_ell |recover_p(ell) - 1| >= 1e-9', 'PREFIX_ENDPOINT_NOT_DEGENERATE（与探针矛盾 => 装置错误）'],
            ['P_ge_060_x >= 0.95 ∧ P_ge_060_j >= 0.95', 'CONCENTRATION_FEW_LAYER_ROBUST'],
            ['P_le_040_x >= 0.95 ∧ P_le_040_j >= 0.95', 'CONCENTRATION_ACCUMULATE_ROBUST'],
            ['argmax_w_x != argmax_w_j ∧ |argmax_w_x - argmax_w_j| >= 3 ∧ top3_share_x >= 0.40 ∧ top3_share_j >= 0.40',
             'CONCENTRATION_COORDINATE_DEPENDENT'],
            ['else', 'CONCENTRATION_UNDECIDED'],
        ],
        'verdict_table_cross_family': {
            'targets': {'xhalf_family_window': MODE_X_13, 'J_family_window': MODE_J_13,
                        'source': 'Phase 13 已发表 extra.mode_x / extra.mode_j'},
            'd_x': '|argmax_w_x - 14|',
            'd_j': '|argmax_w_j - 1|',
            'rules': [
                ['d_x <= 2 ∧ d_j <= 2', 'FAMILY_TRANSFER_BOTH（双坐标 argmax 位置是层栈性质，不依赖干预族）'],
                ['d_x <= 2 ∧ d_j > 2', 'FAMILY_TRANSFER_XHALF_ONLY'],
                ['d_j <= 2 ∧ d_x > 2', 'FAMILY_TRANSFER_J_ONLY'],
                ['else', 'FAMILY_NO_TRANSFER（Phase 13 结论为单点族特异）'],
            ],
        },
        'position_verdict': {
            'POS_LAST_POSITION_DOMINANT': 'median_ell y0/y1 <= 0.30',
            'POS_TWO_POSITION_COMPARABLE': 'median_ell y0/y1 in (0.30, 0.70]',
            'POS_FIRST_POSITION_DOMINANT': 'median_ell y0/y1 > 0.70',
            'additivity': 'SUB_ADDITIVE if frac{S>0} >= 0.5 else SUPER_ADDITIVE',
        },
    },
    'floors': {
        'F24': 'FULL_SWAP 从 Phase 12 FULL_SWAP_pairs 重算须 bit-equal Phase 12 FULL_SWAP',
        'F25': 'mean||P_U6(diff6)|| 须 == 17.06125152401808（|d| <= 2e-2）',
        'F26': 'U6 奇异值须 == Phase 8 锚（相对误差 <= 1e-6）',
        'F27': 'TMPL 对全部 41 实例 tokenize 为 T == 2',
        'F28': 'alpha=0 全位点 patch（3 位点 + R）须还原 capture 分数（max|dScore| < 1e-2）',
        'F29': 'mask={1} alpha=1 的 y1(ell) 须逐位等于 Phase 12 recover(ell)（18 位点，|d| <= 1e-9）',
        'F30': 'mask={0,1} alpha=1 的 y01(ell) 须 == 1.0（18 位点，|d| <= 1e-9）',
        'F31': 'BASE[rw].sr0 > 0 对全部 41 实例（F2 复刻）',
        'F32': 'cross_alpha(frac=0.5) 有限：18 位点全部命中（legacy 网格）',
        'F33': 'J_only 有限：18 位点全部',
        'F34': 'per-(ell,alpha) 行的 order 在 18 个 alpha 上逐元素一致',
        'F35': 'q_ell(ell) 复现 Phase 12 已发表值（18 位点，|d| <= 1e-6）',
        'F36': '每行 per_pair 长度 == 24 且 n == 24',
        'F37': '全部曲线无 NaN/Inf',
    },
    'honesty': [
        '1. 本 Phase 的「第三口径」仍是【激活级干预】，不是权重实现级证明；三条口径同属干预族，不构成对同一结论的独立证据链。',
        '2. 前缀族 α=1 精确等于供体前向（F30），端点量按【构造】退化 ⇒ 端点完全不含深度信息；主量只能取曲线形状量。',
        '3. mask={0,1} 的效应【不是】mask={0} 与 mask={1} 之和（非线性读出）；S(ell) 只作描述性可加性诊断。',
        '4. 24 个 discovery 对不独立（同一类别内 4 个实例共享上位词），bootstrap 只刻画同一受试集内的重采样不确定性，不提高跨实例/跨类别可迁移性。',
        '5. 位点间 Δ 检验沿用 Phase 13 的配对口径，但本 Phase 未重做 Phase 13 的位点配对判决表（那是 Phase 13 的产物）；本 Phase 只报两坐标的集中度与 argmax。',
        '6. XH_RANGE_p 的比较对象是 Phase 12 的 0.109388（同一 legacy 14 点网格）；dense 网格版本【不】可与 Phase 12 数值互核。',
        '7. 跨族迁移判据的靶值（14 与 1）来自 Phase 13 已发表产物，属【前视】但非本 Phase 观测；判据在运行前冻结。',
        '8. T=2 使「累积前缀曲线」只有两级，不能刻画 3 级以上的累积形状；这是 qwen3-4b tokenizer 与模板的联合产物，不可外推到其他模板。',
        '9. 确认集只有 17 实例 / 4 位点，本 Phase 只报与发现集同号性，不单独作判决。',
        '10. A5 floor 的「随机 5 维方向」与 Phase 12 的 E4 构造一致（U6 子空间内），其几何地板不是 0（随机方向在写方向上有投影）。',
    ],
    'may_falsify_the_whole_line': (
        '若前缀族的 argmax 窗口既不在深尾（>=13）也不在浅端（<=2），且 J_p 与 J_swap 的秩相关 < 0.5，'
        '则 Phase 12/13 的「逐层累积 + 坐标系依赖」是【单点族特异】的图像，'
        '「层=软门 + 下游读数」这一 v5.3 拼图中的「逐层累积」条须标注为族依赖。'
        '反之若两坐标 argmax 均迁移成功，则「深尾 xhalf / 浅端 J」是层栈的结构性质，可上提为跨族稳健结论。'
    ),
    'artifacts': {
        'scripts_dir': 'tests/deepseek/Phase14/',
        'temp_dir': 'tests/deepseek_temp/Phase14/',
        'seal': 'tests/deepseek_temp/Phase14/N2h1a7_design_seal.json',
        'execution': 'tests/deepseek_temp/Phase14/execution_phase14.json',
        'result': 'tests/deepseek_temp/Phase14/result_phase14.json',
        'report': 'tests/deepseek_temp/Phase14/n2h1a7_report_qwen3-4b.txt',
        'judgement': 'tests/deepseek_temp/Phase14/judgement_phase14.json',
        'feas_probe': 'tests/deepseek_temp/Phase14/_feas_probe.txt',
        'smoke': 'tests/deepseek_temp/Phase14/smoke/',
    },
    'inheritance_anchors': {
        'phase12_result_sha256': sha256(P12_RES),
        'phase12_result_sha8': sha8(P12_RES),
        'phase12_execution_sha256': sha256(P12_EXEC),
        'phase12_execution_sha8': sha8(P12_EXEC),
        'phase13_result_sha256': sha256(P13_RES),
        'phase13_result_sha8': sha8(P13_RES),
        'inherited_published': {
            'FULL_SWAP': FULL_SWAP,
            'mean_n6_ref_phase9': N6_REF,
            'sing_U6': SING_U6,
            'XH_12_by_site': XH_12,
            'J_swap_12_by_site': J_12,
            'recover_12_by_site': REC_12,
            'XH_RANGE_12': XH_RANGE_12,
            'XH_JUMPS_12': XH_JUMPS_12,
            'TOP3_12': TOP3_12,
            'MODE_X_13': MODE_X_13,
            'MODE_J_13': MODE_J_13,
            'P_FEW_X_13': P_FEW_X_13,
            'P_FEW_J_13': P_FEW_J_13,
            'SHARE_X_13': SHARE_X_13,
            'SHARE_J_13': SHARE_J_13,
            'Q_ELL_12': Q_ELL_12,
        },
        'phase11_result_sha8': sha8(os.path.join(P11T, 'result_phase11.json')),
    },
    'why_not_a_HARKing_violation': (
        '全部靶值（FULL_SWAP / n6 / sing_U6 / recover(ell) / xhalf(ell) / J_swap(ell) / mode_x / mode_j）'
        '都取自 Phase 8-13 的【已发表冻结产物】，仅用于 (a) 逐位装置锚 F24-F30、'
        '(b) 跨族迁移判据的固定靶值。本 Phase 的新判决量（xhalf_p、J_p、y0、S、两坐标 argmax 与 top3_share）'
        '在运行前全部无观测。判决表、5 条预测、14 条 floor 均在本文件冻结。'
    ),
    'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'phase': 14,
}

with io.open(SEAL_OUT, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
    f.write('\n')

b = io.open(SEAL_OUT, 'rb').read()
print('WROTE %s' % SEAL_OUT)
print('  bytes = %d' % len(b))
print('  sha256 = %s' % hashlib.sha256(b).hexdigest())
print('  sha8   = %s' % hashlib.sha256(b).hexdigest()[:8])
print('  top-level keys = %d' % len(seal))
print('  arms fwd total = %d' % sum(v.get('fwd', 0) for v in seal['arms'].values()))
print('  predictions = %s' % list(seal['pre_registered_predictions'].keys()))
print('  floors = %s' % list(seal['floors'].keys()))
