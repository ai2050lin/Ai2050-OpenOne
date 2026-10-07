# -*- coding: utf-8 -*-
"""
Phase 12 / N2h1-alpha-5 : 执行冻结文件生成器（v2，含 amend1）
=============================================================
从 Phase 11 的 execution_phase11.json 逐字节继承面板（再对 Phase 8/10 双向断言），
追加 Phase 12 特有字段（swap / g_family / amend1 / 新臂 / 跨 Phase 参照），写出
tests/deepseek_temp/Phase12/execution_phase12.json。

v2 变更（依 amend1）：
  * E2_swap_curve / E6_readout_swap 的 alpha 网格由 12 点加密为 14 点（加 0.15 / 0.95）
  * g_family 的 G0/G1/G2 重定义为 xhalf（经验半饱和点）口径；新增 G5 充分性判据
  * 追加 amend1 元数据与 sha
本脚本在【正式运行前】运行一次；产出后冻结。
"""
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P8T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8')
P10T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10')
P11T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11')
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
SEAL = os.path.join(P12T, 'N2h1a5_design_seal.json')
AMEND = os.path.join(P12T, 'N2h1a5_design_seal_amend1.json')
OUT = os.path.join(P12T, 'execution_phase12.json')

EXEC8 = os.path.join(P8T, 'execution_phase8.json')
EXEC10 = os.path.join(P10T, 'execution_phase10.json')
EXEC11 = os.path.join(P11T, 'execution_phase11.json')
RES11 = os.path.join(P11T, 'result_phase11.json')

GRID14 = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


E8 = json.load(io.open(EXEC8, encoding='utf-8'))
E10 = json.load(io.open(EXEC10, encoding='utf-8'))
E11 = json.load(io.open(EXEC11, encoding='utf-8'))
S12 = json.load(io.open(SEAL, encoding='utf-8'))
A12 = json.load(io.open(AMEND, encoding='utf-8'))

# ---- 面板逐字节继承（先自检） ----
PANEL = ['discovery', 'confirmation', 'instances_all', 'pairs_all', 'sup_id', 'classes',
         'PATCH_L', 'primary_layer', 'pre_layer', 'control_layer', 'template', 'seed']
for f in PANEL:
    a = json.dumps(E8[f], sort_keys=True, ensure_ascii=False)
    b = json.dumps(E10[f], sort_keys=True, ensure_ascii=False)
    c = json.dumps(E11[f], sort_keys=True, ensure_ascii=False)
    assert a == b == c, '面板继承失败 (Phase8 vs Phase10 vs Phase11): %s' % f
print('[gen12-v2] 面板 12 字段 x 3 向逐元素一致 OK')

E = {}
for f in PANEL:
    E[f] = E11[f]
for f in ['model', 'model_dir', 'config_sha256', 'tok_sha256', 'expected_cfg',
          'o_proj_in_features', 'n_heads', 'head_dim',
          'rbar_ref_from_phase9', 'mean_n6_ref_from_phase9', 'n6_drift_tol',
          'full_ref_from_phase9', 'anchor_ref_d1a_from_phase9', 'anchor_drift_tol',
          'off_manifold_pert_rel', 'classifier']:
    E[f] = E11[f]

E['depth_sites'] = E11['depth_sites']
E['profile_sites'] = E11['profile_sites']
E['readout_site'] = 'R'
E['swap_sites'] = list(E11['profile_sites'])
E['swap_rel_sites'] = [7, 12, 20, 34]
E['overshoot_sites'] = [6, 20, 34]
E['floor_sites'] = [7, 20, 34]
E['conf_sites'] = [7, 11, 20, 34]
E['f3_sites'] = [7, 20, 34, 'R']

E['swap'] = {
    'diff_ell': "diff_ell[p] = h_ell(donor) - h_ell(recip)；h_ell = hidden_states[ell+1]（层输出口径）",
    'diff_R': "diff_R[p] = hR(donor) - hR(recip)；hR = 最终 LayerNorm 输出（norm hook）",
    'intervention': "h_ell_recip + alpha * diff_ell ; alpha=1 => h_ell_donor（精确，F12）",
    'x_alpha': "alpha 属于 [0,1] —— 替换比例（主坐标，无量纲）",
    'x_rel': "a_rel —— h_ell_recip + a_rel*||h_ell_recip||*unit(diff_ell)；满替换对应 a_rel_full(ell)=||diff_ell||/||h_ell_recip||",
    'y': "dDonor_swap / FULL_SWAP",
    'FULL_SWAP': "mean_pairs[ score_of(donor_logits, donor_sup, donor_sid) - BASE[rw].sd0 ]（capture 阶段供体自身前向，零额外前向）",
    'no_cross_phase_numeric_compare': "J_swap 与 J_inject 数值不可比对（x 轴尺度不同）；只比秩（G3）。",
    'endpoint_degeneracy': "recover(alpha=1) 按构造饱和（见 amend1 A1）；主量改为 xhalf 与 J_swap。",
}

E['arms'] = {
    'E0_anchor_L6': {
        'site': 6, 'alpha_grid': [1.0], 'panel': 'discovery',
        'vector': "h6_recip + 1.0 * P_U6(diff6)  [注入口径，非替换口径]",
        'purpose': "内建跨 Phase 复现点，必须逐位等于 10.574739583333335（F6）",
    },
    'E2_swap_curve': {
        'sites': 'swap_sites (profile_sites 18 个)',
        'panel': 'discovery',
        'alpha_grid': GRID14,
        'vector': "h_ell_recip + alpha * diff_ell",
        'purpose': "主臂：逐层全残差替换 -> xhalf(ell)/x*(ell)/J_swap(ell) 剖面与 recover(ell) 诊断",
        'per_pair_landing': True,
    },
    'E2b_swap_rel': {
        'sites': 'swap_rel_sites [7,12,20,34]',
        'panel': 'discovery',
        'alpha_grid': [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6],
        'vector': "h_ell_recip + a_rel * ||h_ell_recip|| * unit(diff_ell)",
        'purpose': "第二坐标（范数归一）校验；位点集与 Phase 10 自基臂一致",
        'per_pair_landing': True,
    },
    'E3_overshoot': {
        'sites': 'overshoot_sites [6,20,34]',
        'panel': 'discovery',
        'alpha_grid': [1.25, 1.5],
        'vector': "h_ell_recip + alpha * diff_ell (alpha>1)",
        'purpose': "描述性饱和检查；不参与 G 族判决",
        'per_pair_landing': False,
    },
    'E4_floor_swap': {
        'sites': 'floor_sites [7,20,34]',
        'panel': 'discovery',
        'alpha_grid': [1.0], 'draws': 2,
        'vector': "h_ell_recip + 1.0 * ||diff_ell|| * unit(z)，z~N(0,I_HID)（全空间随机）",
        'purpose': "地板（范数匹配全空间随机方向）；F1",
    },
    'E5_conf_swap': {
        'sites': 'conf_sites [7,11,20,34]',
        'panel': 'confirmation',
        'alpha_grid': [0.25, 0.5, 0.75, 1.0],
        'vector': "h_ell_recip + alpha * diff_ell",
        'purpose': "确认集（n=17）验 G1/G2/G4/G5",
        'per_pair_landing': True,
    },
    'E6_readout_swap': {
        'site': 'R', 'panel': 'discovery',
        'alpha_grid': GRID14,
        'vector': "hR_recip + alpha * diff_R",
        'purpose': "recover(R) 与 xhalf(R)；alpha=1 是构造恒等（F11）",
        'per_pair_landing': True,
    },
    'E7_permutation_null': {'type': 'cpu_only',
                            'purpose': "把 xhalf 与 depth 标签随机置换 B_perm 次 => Spearman null 带（F7 迭代版）"},
}

E['g_family'] = {
    'amended_by': 'N2h1a5_design_seal_amend1.json',
    'G0_precondition': {'curve_ok_frac_min': 0.75, 'xh_range_min': 0.10,
                        'curve_ok': "jump_ratio 有限且 >0 且 y_sat >= UNREACH_y",
                        'xh_range': "max(xhalf) - min(xhalf) over 可算位点"},
    'G1_crystallization': {'normalize': 'XN(ell) = (xhalf(ell)-min)/(max-min)',
                           'G1a': 'x_half <= 20 AND span_10_90 <= 14',
                           'G1b': 'span_10_90 >= 24',
                           'G1_flat': 'XH_RANGE < 0.10 => 承诺点不随深度变（每层等价）',
                           'else': 'G1_mid'},
    'G2_concentration': {'window': 3,
                         'jump': 'jump_i = xhalf(ell_{i+1}) - xhalf(ell_i)',
                         'top3_share_x': 'max_i |sum(jump_i..jump_{i+2})| / XH_RANGE',
                         'max_share_x': 'max_i |jump_i| / XH_RANGE',
                         'G2a_top3_share_x_min': 0.60,
                         'G2b_max_share_x_max': 0.40,
                         'else': 'G2_mid'},
    'G3_profile_shape': {'rho_JG_same_min': 0.6, 'rho_JG_indep_max': 0.2, 'else': 'G3_weak'},
    'G4_confirmation': {'rho_abs_min': 0.5, 'same_sign': True},
    'G5_sufficiency': {'min_recover_min': 0.90,
                       'pass': 'LAST_POS_STATE_SUFFICIENT',
                       'fail': 'LAST_POS_STATE_NOT_SUFFICIENT'},
    'combined_verdict_table': A12['A3_G_renaming']['combined_verdict_table'],
}
E['decision'] = E['g_family']

E['bootstrap'] = {
    'scheme': 'pair-level percentile bootstrap (B=2000)',
    'B': 2000, 'B_perm': 2000, 'seed': 20261001, 'ci': 'percentile 2.5/97.5',
    'primary_statistics': ['xhalf per site', 'top3_share_x', 'Spearman(xhalf, depth)'],
    'diagnostic_statistics': ['recover per site', 'J_swap per site', 'Spearman(recover, depth)'],
    'per_pair_landing_arms': ['E2', 'E2b', 'E5', 'E6'],
    'zero_extra_forward': True,
    'note': 'xhalf 由逐对均值曲线的线性插值给出（不做 logistic，故 bootstrap 便宜）；纯 CPU。',
}
E['per_pair_landing'] = {'E2': True, 'E2b': True, 'E3': False, 'E4': False, 'E5': True, 'E6': True}

E['phase11_result_for_F13'] = 'tests/deepseek_temp/Phase11/result_phase11.json'
E['phase11_result_sha256'] = sha(RES11)
E['inherits_panel_from'] = 'tests/deepseek_temp/Phase11/execution_phase11.json'
E['inherits_panel_sha256'] = sha(EXEC11)
E['inherits_panel10_from'] = 'tests/deepseek_temp/Phase10/execution_phase10.json'
E['inherits_panel10_sha256'] = sha(EXEC10)
E['inherits_panel8_from'] = 'tests/deepseek_temp/Phase8/execution_phase8.json'
E['inherits_panel8_sha256'] = sha(EXEC8)
E['inherits_numbers_from'] = 'tests/deepseek_temp/Phase11/result_phase11.json'
E['inherits_numbers_sha256'] = E['phase11_result_sha256']
E['amend1'] = {'path': 'tests/deepseek_temp/Phase12/N2h1a5_design_seal_amend1.json',
               'sha256': sha(AMEND),
               'summary': 'A1 主量由 recover(alpha=1) 改为 xhalf/x*/J_swap；A2 新增 G5 充分性；'
                          'A3 G0/G1/G2 重定义；A4 alpha 网格 12->14；A5 实现修复；A6 bootstrap 增项'}

E['phase'] = 12
E['name'] = 'N2h1-alpha-5 / per-site residual swap + layer contribution allocation'
E['frozen_at'] = '2026-10-02 00:50'
E['smoke_env'] = ("SMOKE=1 -> discovery 取 6 个类各第一个实例（保证 U6 满秩）；profile/swap 位点截为 [6,7,8]；"
                  "floor/f3 位点同截；conf_sites 清空；alpha 网格截为前 3 点并强制保留 1.0；"
                  "E2b 截为前 2 点；bootstrap B 截为 200；冒烟产物落 smoke/")
E['result_keys'] = ['E0', 'full_L6', 'bit_replication', 'FULL_SWAP', 'dose_coord', 'diff_norms',
                    'proj_share_u6', 'profile_swap', 'profile_swap_rel', 'profile_R_swap',
                    'E2', 'E2b', 'E3', 'E6', 'E2_pairs', 'E2b_pairs', 'E5_pairs', 'E6_pairs',
                    'recover', 'xhalf', 'G_family', 'G_verdict', 'bootstrap_band',
                    'permutation_null', 'E4', 'E5', 'floors', 'elapsed_s', 'drift_flags']

REQUIRED = ['phase', 'name', 'frozen_at', 'model', 'model_dir', 'config_sha256', 'template', 'seed',
            'sup_id', 'classes', 'instances_all', 'discovery', 'confirmation', 'pairs_all',
            'primary_layer', 'pre_layer', 'control_layer', 'expected_cfg', 'o_proj_in_features',
            'n_heads', 'head_dim', 'depth_sites', 'profile_sites', 'readout_site', 'swap_sites',
            'swap_rel_sites', 'overshoot_sites', 'floor_sites', 'conf_sites', 'f3_sites',
            'rbar_ref_from_phase9', 'mean_n6_ref_from_phase9', 'n6_drift_tol',
            'full_ref_from_phase9', 'anchor_ref_d1a_from_phase9', 'anchor_drift_tol',
            'off_manifold_pert_rel', 'classifier', 'decision', 'g_family', 'arms', 'bootstrap',
            'per_pair_landing', 'phase11_result_for_F13', 'phase11_result_sha256',
            'inherits_panel_from', 'inherits_panel_sha256', 'inherits_panel10_from',
            'inherits_panel10_sha256', 'inherits_panel8_from', 'inherits_panel8_sha256',
            'inherits_numbers_from', 'inherits_numbers_sha256', 'amend1', 'smoke_env', 'result_keys']
_miss = [k for k in REQUIRED if k not in E]
assert not _miss, 'execution_phase12.json 缺字段: %s' % _miss

io.open(OUT, 'w', encoding='utf-8').write(json.dumps(E, ensure_ascii=False, indent=1))
print('[gen12-v2] wrote %s' % OUT)
print('[gen12-v2] bytes=%d sha256=%s' % (os.path.getsize(OUT), sha(OUT)))
print('[gen12-v2] amend1 sha256 %s' % E['amend1']['sha256'][:16])
print('[gen12-v2] inherits panel(phase11) %s' % E['inherits_panel_sha256'][:16])
print('[gen12-v2] phase11 result (F13)    %s' % E['phase11_result_sha256'][:16])
print('[gen12-v2] fields=%d ; E2 grid=%d ; profile_sites=%d' %
      (len(E), len(E['arms']['E2_swap_curve']['alpha_grid']), len(E['profile_sites'])))
