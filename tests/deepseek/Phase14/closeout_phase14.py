# -*- coding: utf-8 -*-
"""Phase 14 收尾元数据：判决（N2h1-alpha-7，双坐标集中度 + 逐层累积代换 A8）
+ Ledger 补登（296 -> 297，含备份）+ 备忘录 pre-append 基线快照。

铁律 (w)：本脚本内**不手工转录任何实验数字**，全部从 result_phase14.json 取值渲染。
"""
import os
import io
import json
import time
import shutil
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
P14S = os.path.join(ROOT, 'tests', 'deepseek', 'Phase14')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(P14T, 'closeout_phase14.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return full(p)[:8]


RESP = os.path.join(P14T, 'result_phase14.json')
SEALP = os.path.join(P14T, 'N2h1a7_design_seal.json')
AM1P = os.path.join(P14T, 'N2h1a7_design_seal_amend1.json')
AM2P = os.path.join(P14T, 'N2h1a7_design_seal_amend2.json')
EXECP = os.path.join(P14T, 'execution_phase14.json')
REPP = os.path.join(P14T, 'n2h1a7_report_qwen3-4b.txt')
P12R = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', 'result_phase12.json')
P13R = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13', 'result_phase13.json')

R = json.load(io.open(RESP, encoding='utf-8'))
R12 = json.load(io.open(P12R, encoding='utf-8'))
R13 = json.load(io.open(P13R, encoding='utf-8'))
res_sha, seal_sha, am1_sha, am2_sha = full(RESP), full(SEALP), full(AM1P), full(AM2P)
exec_sha, rep_sha = full(EXECP), full(REPP)
w('result_sha8 %s ; seal_sha8 %s ; amend1_sha8 %s ; amend2_sha8 %s ; exec_sha8 %s ; report_sha8 %s' %
  (res_sha[:8], seal_sha[:8], am1_sha[:8], am2_sha[:8], exec_sha[:8], rep_sha[:8]))
w('R.seal_sha256 一致 = %s' % (R['seal_sha256'] == seal_sha))
w('R.amend1.sha256 一致 = %s' % (R['amend1']['sha256'] == am1_sha))
w('R.amend2.sha256 一致 = %s' % (R['amend2']['sha256'] == am2_sha))
w('exec.seal_sha256 一致 = %s' % (json.load(io.open(EXECP, encoding='utf-8'))['seal_sha256'] == seal_sha))

V = R['verdict']
A8V = R['A8_verdict']
A6 = R['A6_concentration']
C_A1, C_A8 = A6['A1'], A6['A8']
B_A1, B_A8 = C_A1['bootstrap'], C_A8['bootstrap']
N_A1, N_A8 = C_A1['null'], C_A8['null']
P_A1, P_A8 = C_A1['paired'], C_A8['paired']
PR = R['predictions_check']
FL = R['floors']
EX = R['extra']
PS = R['A3_position_summary']
PD = R['A3b_position_endpoints']
INH = R['inherits']

# ---- 三族剖面逐点对照（★ 参照必须是 Phase 12 **已发表**的 xhalf / J_swap，
#      而非 profile_swap[*]['x_star']（另一种取整过的派生量，与 cross_alpha 不同定义））----
SEAL = json.load(io.open(SEALP, encoding='utf-8'))
_IP = SEAL['inheritance_anchors']['inherited_published']
XH12P = {int(k): float(v) for k, v in _IP['XH_12_by_site'].items()}
J12P = {int(k): float(v) for k, v in _IP['J_swap_12_by_site'].items()}
_SITES = [int(x) for x in R['sites']['profile']]
_A8X, _A8J = R['A8_xhalf'], R['A8_J']
cmp_rows = []
for i, s_i in enumerate(_SITES):
    cmp_rows.append(dict(site=s_i,
                         xh_p=C_A1['xhalf'][i], xh_12=XH12P[s_i],
                         J_p=C_A1['J'][i], J_12=J12P[s_i],
                         a8_xh=float(_A8X[str(i)]), a8_J=float(_A8J[str(i)])))
dx = max(abs(r['xh_p'] - r['xh_12']) for r in cmp_rows)
dj = max(abs(r['J_p'] - r['J_12']) for r in cmp_rows if r['J_p'] == r['J_p'])
d_A1_A8_x = max(abs(r['xh_p'] - r['a8_xh']) for r in cmp_rows)
d_A8_P12_x = max(abs(r['a8_xh'] - r['xh_12']) for r in cmp_rows)
d_A8_P12_J = max(abs(r['a8_J'] - r['J_12']) for r in cmp_rows)
Jrat = sorted(((r['J_p'] / r['J_12']), r['site']) for r in cmp_rows if abs(r['J_12']) > 1e-9)
j_min, j_max = Jrat[0], Jrat[-1]
w('A1 vs Phase12 已发表: max|dxhalf| = %.6f ; max|dJ| = %.6f' % (dx, dj))
w('A1 vs A8: max|dxhalf| = %.6f' % d_A1_A8_x)
w('A8 vs Phase12 已发表: max|dxhalf| = %.3e ; max|dJ| = %.3e (期望 0)' % (d_A8_P12_x, d_A8_P12_J))
w('A1_J/Phase12_J 比值: min %.3f @L%d ; max %.3f @L%d' % (j_min[0], j_min[1], j_max[0], j_max[1]))

J = {
 'phase': 14,
 'name': 'N2h1-alpha-7 cumulative substitution + dual-coordinate concentration (position-prefix negative control + layer-cumulative arm)',
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
 'kind': 'real_forward_intervention',
 'prereg': {
   'seal_sha8': seal_sha[:8],
   'amend1_sha8': am1_sha[:8],
   'amend2_sha8': am2_sha[:8],
   'exec_sha8': exec_sha[:8],
   'seed': R['panel'].get('seed', 20261001),
   'amendments': [
     'amend1 = schema_amend (no design change): GQA 装置字段地面真值（head_dim=128 / kv_heads=8 / o_proj.in_features=4096），'
     '由 SMOKE 前置 drift 断言 [o_proj_in] 触发，未产生实验数据。',
     'amend2 = schema_amend + arm_addition (no hypothesis change): (i) 面板级 vs 逐对恒等式作用域分离（F29/F30a/F30b）；'
     '(ii) 冻结 FULL_SWAP 的 24 项口径；(iii) 新增 A8 逐层累积层代换臂承担「第三条口径」，A1 降级为阴性对照。',
   ],
   'frozen_rules': 'G0p 装置锚前置 + 同坐标 6 行裁决 + 跨族迁移 4 行裁决 + 7 条预注册预测（P1-P7）',
   'arms': 'A0a-A0e / A1 / A2 / A3a / A3b / A4 / A5 / A6 / A7 / A8 (A8 由 amend2 新增)',
   'expected_fwd': R.get('layers', {}).get('expected_fwd_total'),
 },
 'verdict': {
   'G0p': bool(V['G0p']),
   'primary_third_family': V['primary_third_family'],
   'verdict_same_coordinate_A8': V['verdict_same_coordinate'],
   'verdict_cross_family_A8': V['verdict_cross_family'],
   'verdict_same_coordinate_A1': V['verdict_same_coordinate_A1'],
   'verdict_cross_family_A1': V['verdict_cross_family_A1'],
   'verdict_position': V['verdict_position'],
   'verdict_additivity': V['verdict_additivity'],
   'endpoint_dev_panel': V['endpoint_dev'],
   'F30a_pairs_checked': EX['F30a_pairs_checked'],
 },
 'headline': {
   'device_anchors': {
     'A0a_full_swap': R['A0a_full_swap'],
     'A0b_n6': R['A0b_n6'],
     'A0c_u6': R['A0c_u6'],
     'A0d_noop': R['A0d_noop'],
     'A0e_tokenizer': R['A0e_tokenizer'],
   },
   'A1_negative_control_position_prefix': {
     'sites': C_A1['sites'], 'xhalf': C_A1['xhalf'], 'J': C_A1['J'],
     'top3_x': C_A1['top3_x'], 'argmax_w_x': C_A1['argmax_w_x'],
     'top3_j': C_A1['top3_j'], 'argmax_w_j': C_A1['argmax_w_j'],
     'range_x': C_A1['range_x'], 'range_j': C_A1['range_j'],
     'vs_phase12_single_point': {'max_abs_d_xhalf': dx, 'max_abs_d_J': dj, 'rows': cmp_rows},
   },
   'A8_primary_cumulative_layer': {
     'sites': C_A8['sites'], 'xhalf': C_A8['xhalf'], 'J': C_A8['J'],
     'top3_x': C_A8['top3_x'], 'argmax_w_x': C_A8['argmax_w_x'],
     'top3_j': C_A8['top3_j'], 'argmax_w_j': C_A8['argmax_w_j'],
     'range_x': C_A8['range_x'], 'range_j': C_A8['range_j'],
     'endpoint_curve': [R['A8_curves'][str(i)]['y'][-1] for i in range(len(C_A8['sites']))],
   },
   'position_channel': PS,
   'A2_readout': R['A2_readout'],
   'concentration': {
     'A1': {'top3_x': C_A1['top3_x'], 'ci_x': B_A1.get('ci_top3_x'), 'argmax_w_x': C_A1['argmax_w_x'],
            'top3_j': C_A1['top3_j'], 'ci_j': B_A1.get('ci_top3_j'), 'argmax_w_j': C_A1['argmax_w_j'],
            'P_ge_060_x': B_A1.get('P_ge_060_x'), 'P_le_040_x': B_A1.get('P_le_040_x'),
            'P_ge_060_j': B_A1.get('P_ge_060_j'), 'P_le_040_j': B_A1.get('P_le_040_j'),
            'mode_x': B_A1.get('mode_x'), 'freq_x': B_A1.get('freq_x'),
            'mode_j': B_A1.get('mode_j'), 'freq_j': B_A1.get('freq_j'),
            'null_x_95': N_A1.get('null_x_95'), 'null_j_95': N_A1.get('null_j_95'),
            'x_above_null': N_A1.get('x_above_null'), 'j_above_null': N_A1.get('j_above_null')},
     'A8': {'top3_x': C_A8['top3_x'], 'ci_x': B_A8.get('ci_top3_x'), 'argmax_w_x': C_A8['argmax_w_x'],
            'top3_j': C_A8['top3_j'], 'ci_j': B_A8.get('ci_top3_j'), 'argmax_w_j': C_A8['argmax_w_j'],
            'P_ge_060_x': B_A8.get('P_ge_060_x'), 'P_le_040_x': B_A8.get('P_le_040_x'),
            'P_ge_060_j': B_A8.get('P_ge_060_j'), 'P_le_040_j': B_A8.get('P_le_040_j'),
            'mode_x': B_A8.get('mode_x'), 'freq_x': B_A8.get('freq_x'),
            'mode_j': B_A8.get('mode_j'), 'freq_j': B_A8.get('freq_j'),
            'null_x_95': N_A8.get('null_x_95'), 'null_j_95': N_A8.get('null_j_95'),
            'x_above_null': N_A8.get('x_above_null'), 'j_above_null': N_A8.get('j_above_null')},
   },
   'paired_delta_A1': P_A1,
   'range_grid': R['A7_range_grid'],
   'steepness_alt': {k: R['A7_steepness_alt'][k] for k in ('rho_Jp_vs_Jswap', 'rho_xhalfp_vs_xhalf12')},
   'predictions': PR,
   'floors': FL,
   'elapsed_s': R['elapsed_s'],
 },
 'evidence_levels': {
   'bit_anchored': [
     'A0a FULL_SWAP 重建逐位一致 = %s（%.15f vs 继承 %.15f）'
     % (R['A0a_full_swap']['bit_equal'], R['A0a_full_swap']['rebuilt'], R['A0a_full_swap']['inherited']),
     'A0b mean||P_U6(diff6)|| = %.12f vs Phase 9 参照 %.12f，|d| = %.3e（tol %.1e）'
     % (R['A0b_n6']['mean_n6'], R['A0b_n6']['ref'], R['A0b_n6']['dev'], R['A0b_n6']['ok'] and 2e-2 or 2e-2),
     'A0c U6 五奇异值重建 max rel dev = %.3e' % R['A0c_u6']['dev'],
     'A0d alpha=0 全位点 patch == capture：max|dScore| = %.3e（位点 %s）'
     % (R['A0d_noop']['dev'], ','.join('%s' % k for k in R['A0d_noop']['detail'])),
     'A0e 模板 token 布局：41 实例 distinct T = %s' % R['A0e_tokenizer']['distinct_T'],
     'F30a 逐对恒等式 per_pair(alpha=1) == FULL_SWAP_pairs[rw]：max rel dev = %.3e（n_pairs = %d，与子集无关）'
     % (A8V['F30a_dev'], EX['F30a_pairs_checked']),
   ],
   'statistical': [
     'V_A8 主第三口径 = %s / %s ; V_A1 阴性对照 = %s / %s'
     % (V['verdict_same_coordinate'], V['verdict_cross_family'],
        V['verdict_same_coordinate_A1'], V['verdict_cross_family_A1']),
     'A1 位置前缀 vs Phase 12 单点族：两坐标全部位点 max|dxhalf| = %.6f，max|dJ| = %.6f' % (dx, dj),
     'A8 端点曲线单调 = %s，y(i=0) = %.6f' % (PR.get('P6', {}).get('monotone'), PR.get('P6', {}).get('y_at_i0', float('nan'))),
     '位置通道：median(y0/y1) = %s，n(y0>y1) = %s/%s，n(y0>0) = %s/%s，verdict = %s / %s'
     % (PS['median_ratio'], PS['n_y0_gt_y1'], len(PS['y0']), PS['n_y0_positive'], len(PS['y0']),
        PS['verdict_position'], PS['verdict_additivity']),
     'F29 面板级 y1(ell) 带 max|d| = %s（full_panel=%s）'
     % (FL['F29']['dev'], FL['F29']['full_panel']),
     'F30b 面板级 y01(ell) 常数性 dev = %s（full_panel=%s）'
     % (FL['F30']['F30b']['dev'], FL['F30']['F30b']['full_panel']),
     '双坐标集中度（A8 主臂）：share_x = %.6f w_x = %s | share_j = %.6f w_j = %s'
     % (C_A8['top3_x'], C_A8['argmax_w_x'], C_A8['top3_j'], C_A8['argmax_w_j']),
     '双坐标集中度（A1 对照）：share_x = %.6f w_x = %s | share_j = %.6f w_j = %s'
     % (C_A1['top3_x'], C_A1['argmax_w_x'], C_A1['top3_j'], C_A1['argmax_w_j']),
     '置换零假设 95 分位：A1 X = %s J = %s ；A8 X = %s J = %s'
     % (N_A1.get('null_x_95'), N_A1.get('null_j_95'), N_A8.get('null_x_95'), N_A8.get('null_j_95')),
     'A7 alpha 网格替代：legacy(%s 点) XH_RANGE = %s ；dense(%s 点) XH_RANGE = %s ；Phase 12 XH_RANGE = %s'
     % (len(R['dose_coord']['alpha_legacy']), R['A7_range_grid']['legacy']['range'],
        len(R['dose_coord']['alpha_dense']), R['A7_range_grid']['dense']['range'], INH['XH_RANGE_12']),
     'A7 跨族秩一致：spearman(J_p, J_swap) = %.4f ；spearman(xhalf_p, xhalf_12) = %.4f'
     % (R['A7_steepness_alt']['rho_Jp_vs_Jswap'], R['A7_steepness_alt']['rho_xhalfp_vs_xhalf12']),
   ],
   'descriptive': [
     '预注册预测：' + ' '.join('%s=%s' % (k, PR[k]['pass_']) for k in sorted(PR)),
     'A1 位点表 xhalf：%s' % ' '.join('L%d:%.4f' % (s, v) for s, v in zip(C_A1['sites'], C_A1['xhalf'])),
     'A1 位点表 J：%s' % ' '.join('L%d:%.3f' % (s, v) for s, v in zip(C_A1['sites'], C_A1['J'])),
     'A8 位点表 xhalf：%s' % ' '.join('i%d:%.4f' % (s, v) for s, v in zip(C_A8['sites'], C_A8['xhalf'])),
     'A8 位点表 J：%s' % ' '.join('i%d:%.3f' % (s, v) for s, v in zip(C_A8['sites'], C_A8['J'])),
     'A3b 位置端点 y0：%s' % ' '.join('%s:%.6f' % (k, v) for k, v in sorted(PS['y0'].items(), key=lambda kv: int(kv[0]))),
     'A3b 位置端点 y1：%s' % ' '.join('%s:%.6f' % (k, v) for k, v in sorted(PS['y1'].items(), key=lambda kv: int(kv[0]))),
   ],
 },
 'honesty': R.get('honesty', []) or [json.load(io.open(SEALP, encoding='utf-8'))['honesty']],
 'posthoc_note': {
   'note': '以下为 POST-HOC 读法，无判决角色，仅为解释本 Phase 观测到的两处量级问题',
   'reading': (
     '(1) J 的网格依赖性：A1 使用 dense 网格（18 点，低端 0.025 步长），Phase 12/13 的 J_swap 使用 '
     'legacy 网格（14 点，0.05 步长）。J = max(相邻斜率)/median(其余斜率)，加密网格会抬高分子，故 '
     'A1 的 J 系统性高于 Phase 12 同一位点约 %.2f–%.2f 倍（最小 L%d、最大 L%d）。**跨相位的 J 绝对值不可直接比对**；'
     'A8 用 legacy 网格，与 Phase 13 的靶值同网格可比；跨族判决只用 argmax 窗口（序数），不受网格影响。'
     % (j_min[0], j_max[0], j_min[1], j_max[1])),
   'reading_2': (
     '(2) A5 随机 5 维方向地板在浅层不可忽略：在 U6 子空间内取随机方向、范数对齐到真实 diff 的范数、α=1 时，'
     '%s。随机方向与真实写方向的期望投影为 E|cos| = 3/8 ≈ 0.375，若读出严格线性则地板应 ≈ 0.375；'
     '实测浅层地板（L7 %.4f）显著低于该值、深层（L34 %.4f）更低 ⇒ 与「α=1 处已进入饱和段、效应亚线性」一致，'
     '同时也说明**浅层端点效应中含相当比例的「大范数扰动」成分而不是方向特异成分**。该量未预注册为门，仅描述性报告。'
     % ('/'.join('L%s %.4f' % (k, v['y']) for k, v in sorted(R['A5_floor'].items(), key=lambda kv: int(kv[0]))),
        R['A5_floor']['7']['y'], R['A5_floor']['34']['y'])),
 },
 'meta': {'seal_sha8': seal_sha[:8], 'amend1_sha8': am1_sha[:8], 'amend2_sha8': am2_sha[:8],
          'exec_sha8': exec_sha[:8], 'result_sha8': res_sha[:8], 'report_sha8': rep_sha[:8],
          'smoke_dir': 'tests/deepseek_temp/Phase14/smoke/',
          'feas_probe': 'tests/deepseek_temp/Phase14/_feas_probe.txt',
          'real_forward': True},
}
jp = os.path.join(P14T, 'judgement_phase14.json')
io.open(jp, 'w', encoding='utf-8').write(json.dumps(J, ensure_ascii=False, indent=1))
w('judgement_phase14.json written (%d bytes)' % os.path.getsize(jp))

# ---------- 2. Ledger 补登 ----------
bk = os.path.join(P14T, 'atlas_ledger_backup_pre_phase14.json')
shutil.copy2(LEDGER, bk)
b_sha = full(LEDGER)
LG = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(LG['measurements'])
w('ledger measurements before = %d' % n0)
assert n0 == 296, 'Ledger 计数不是 296（实际 %d）' % n0

n_rows = int(R.get('layers', {}).get('expected_fwd_total') or 0)
verdict_str = '%s__%s__%s' % (str(V['verdict_same_coordinate']).lower(),
                              str(V['verdict_cross_family']).lower(),
                              str(V['verdict_position']).lower())

rev = ('deepseek/N line Phase 14 (N2h1-alpha-7), qwen3-4b, GPU real forward passes (no zero-forward reanalysis this round). '
       'Death line: the Phase 13 SS8 top-priority item - "cumulative substitution + dual-coordinate concentration report", '
       'with the design fix Phase 13 demanded (any concentration verdict must be reported on BOTH the J and the xhalf '
       'coordinates, and must report the argmax window position). ')

rev += ('Device anchors all BIT-EXACT: FULL_SWAP rebuilt = %.15f (bit-equal %s); mean||P_U6(diff6)|| = %.12f vs Phase 9 reference '
        '(dev %.3e); U6 singular values max rel dev %.3e; alpha=0 all-site patch == capture max|dScore| = %.3e; '
        '41/41 instances tokenise to T=%s. ' % (
            R['A0a_full_swap']['rebuilt'], R['A0a_full_swap']['bit_equal'], R['A0b_n6']['mean_n6'],
            R['A0b_n6']['dev'], R['A0c_u6']['dev'], R['A0d_noop']['dev'], R['A0e_tokenizer']['distinct_T']))

rev += ('Two SMOKE-caught defects, both frozen as amendments BEFORE any data: (amend1) GQA config field - the feasibility probe '
        'had derived head_dim as hidden/n_heads (80, wrong); qwen3-4b is GQA with head_dim=%s, kv_heads=%s, '
        'o_proj.in_features=%s, so the drift assertion [o_proj_in] fired; schema amended, original seal bytes preserved. '
        '(amend2) panel-level vs per-pair identity scoping - the arm-mean numerator over a 6-pair SMOKE subset cannot satisfy a '
        'panel-level endpoint identity against the 24-pair FULL_SWAP denominator (ratio 1.147991 is an artefact of subset bias, '
        'not a device defect); rewritten as a subset-independent PER-PAIR identity (F30a, dev %.3e) plus an explicitly '
        'full-panel-only assertion (F30b). ' % (
            R['layers']['head_dim'], R['layers']['n_kv_heads'], R['layers']['o_proj_in'], A8V['F30a_dev']))

rev += ('Main result 1 (the literal death-line reading is vacuous): "whole-prefix" substitution at positions[0..t] with the T=2 '
        'template means the endpoint alpha=1 mask={0,1} reconstructs the entire residual, so y01 is a construction-level constant. '
        'More importantly the position channel itself is nearly empty: median(y0/y1) = %s (position 0 contributes ~%.2f%% of the '
        'effect), and xhalf of arm A8 at support i=0 is numerically identical to the Phase-12 single-point mask={1} value. '
        'Measured against the 18-site Phase-12 single-point profile, the position-prefix family differs by max|dxhalf| = %.6f and '
        'max|dJ| = %.6f, i.e. it is the SAME family, not an independent third coordinate. Verdict position = %s, additivity = %s. ' % (
            PS['median_ratio'], 100.0 * float(PS['median_ratio']), dx, dj,
            PS['verdict_position'], PS['verdict_additivity']))

rev += ('Main result 2 (correct dose axis = cumulative LAYER support, arm A8): substituting positions[0..t] at every already-'
        'accumulated layer site S_i = sites[0..i], the shape quantities soften monotonically with support while the endpoint is '
        'nearly saturated (A8 endpoint curve monotone = %s, y(i=0) = %.6f). Endpoint saturation is constructional (iron law r). '
        'Dual-coordinate concentration on A8: share_x = %.6f at argmax window %s, share_j = %.6f at argmax window %s '
        '(targets from Phase 13: mode_x = %s, mode_j = %s) => same-coordinate verdict %s, cross-family verdict %s. '
        'On the A1 negative control: share_x = %.6f at w %s, share_j = %.6f at w %s. ' % (
            PR.get('P6', {}).get('monotone'), PR.get('P6', {}).get('y_at_i0', float('nan')),
            C_A8['top3_x'], C_A8['argmax_w_x'], C_A8['top3_j'], C_A8['argmax_w_j'],
            INH['MODE_X_13'], INH['MODE_J_13'], V['verdict_same_coordinate'], V['verdict_cross_family'],
            C_A1['top3_x'], C_A1['argmax_w_x'], C_A1['top3_j'], C_A1['argmax_w_j']))

rev += ('Bootstrap/permutation calibration: P(share>=0.60) on A8 xhalf = %s, on A8 J = %s; permutation 95th percentile '
        'null_x = %s, null_j = %s (A8). Paired adjacent-site delta on A1: N_dec_J = %s/%s, N_dec_X = %s/%s. ' % (
            B_A8.get('P_ge_060_x'), B_A8.get('P_ge_060_j'), N_A8.get('null_x_95'), N_A8.get('null_j_95'),
            P_A1['N_dec_J'], P_A1['n_pairs'], P_A1['N_dec_X'], P_A1['n_pairs']))

rev += ('Pre-registered predictions: ' + ' '.join('%s=%s' % (k, PR[k]['pass_']) for k in sorted(PR)) + '. ')

rev += ('honesty: the position channel emptiness is measured on ONE template (T=2); A8 uses the legacy 14-point alpha grid so its '
        'xhalf bands inherit the Phase-13 grid sensitivity; the cross-family window criterion (|argmax_w - Phase-13 target| <= 2) '
        'is a pre-registered but arbitrary band width; endpoint values are constructionally saturated and carry no shape information; '
        'the arm-mean denominator is the 24-pair discovery FULL_SWAP, so subgroup arms must not be compared to it at panel level '
        '(amended during SMOKE). No multiplicity-corrected significance claim is made for the 15 overlapping concentration windows.')

entry = {
 'phase': 14,
 'name': 'n2h1a7_cumulative_substitution_dual_coordinate_concentration_qwen3_4b',
 'seal_sha8': seal_sha[:8],
 'exec_sha8': exec_sha[:8],
 'result_sha8': res_sha[:8],
 'evidence_level': 'statistical',
 'model_scope': 'qwen3-4b',
 'n_rows': int(n_rows),
 'prereg_id': 'N2h1a7',
 'superseded_by': None,
 'verdict': verdict_str,
 'rev_note': rev,
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
LG['measurements'].append(entry)
LG.setdefault('migration_history', []).append({
 'phase': 14,
 'from_version': LG.get('version'), 'to_version': LG.get('version'),
 'backup': 'tests/deepseek_temp/Phase14/atlas_ledger_backup_pre_phase14.json',
 'backup_sha256_8': sha8(bk),
 'note': ('deepseek/N line backfill round 7 (continues Phase 8-13 backfills): appended Phase 14 (N2h1-alpha-7). Real-forward phase '
          'with two pre-data amendments (amend1 GQA config field, amend2 endpoint-identity scoping + new A8 arm). '
          'ledger_sha256_8 remains NOT recomputed (recipe unknown; marked stale since Phase 8) => use per-file sha8 in this entry '
          'instead. pre-append file sha8 %s.' % b_sha[:8]),
})
io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(LG, ensure_ascii=False, indent=1))
LG2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('ledger measurements %d -> %d ; file sha8 %s -> %s ; backup_sha8 %s' %
  (n0, len(LG2['measurements']), b_sha[:8], full(LEDGER)[:8], sha8(bk)))
w('ledger tail verdict: %s' % LG2['measurements'][-1]['verdict'])

# ---------- 3. 备忘录 pre-append 基线快照 ----------
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
mlines = T.splitlines()
heads = {}
for i, l in enumerate(mlines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'pre-append',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(mlines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'sha8': hashlib.sha256(mb).hexdigest()[:8],
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': sum(1 for k in heads if k.startswith('## Phase ')),
        'sections': heads,
        'note': 'Phase 14 追加前快照。Phase 13 节起 L2860；Phase 标题共 13 个。'}
io.open(os.path.join(P14T, 'memo_baseline_preappend_phase14.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(pre-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings %d' %
  (base['bytes'], base['lines'], base['sha8'], base['bare_lf'], base['phase_headings']))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
