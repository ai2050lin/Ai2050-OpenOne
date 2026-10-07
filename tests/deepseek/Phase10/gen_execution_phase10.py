# -*- coding: utf-8 -*-
"""
Phase 10 / N2h1-alpha-3 : 生成执行冻结档 execution_phase10.json
------------------------------------------------------------------------
规则（与 Phase 9 同）：从 execution_phase8.json 逐字段拷贝面板并做 F5 逐元素断言，
再冻结本 Phase 的位点清单 / alpha 网格 / 判据阈值。
用法：python gen_execution_phase10.py
"""
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P8 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'execution_phase8.json')
P9 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9', 'execution_phase9.json')
RES9 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9', 'result_phase9.json')
P10T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10')
SEAL = os.path.join(P10T, 'N2h1a3_design_seal.json')
OUT = os.path.join(P10T, 'execution_phase10.json')


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


E8 = json.load(io.open(P8, encoding='utf-8'))
E9 = json.load(io.open(P9, encoding='utf-8'))
R9 = json.load(io.open(RES9, encoding='utf-8'))

execu = {
    'phase': 10,
    'name': 'N2h1-alpha-3 / soft_threshold_depth_locating',
    'frozen_at': '2026-10-01 02:42',
    'inherits_panel_from': 'tests/deepseek_temp/Phase8/execution_phase8.json',
    'inherits_panel_sha256': sha256(P8),
    'inherits_numbers_from': 'tests/deepseek_temp/Phase9/result_phase9.json',
    'inherits_numbers_sha256': sha256(RES9),
    'model': E8['model'],
    'model_dir': 'models/hf/%s' % E8['model'],
    'config_sha256': E9['config_sha256'],
    'tok_sha256': E9['tok_sha256'],
    'expected_cfg': E9['expected_cfg'],
    'o_proj_in_features': E9['o_proj_in_features'],
    'template': E8['template'],
    'seed': E8['seed'],
    'sup_id': E8['sup_id'],
    'classes': E8['classes'],
    'instances_all': E8['instances_all'],
    'discovery': E8['discovery'],
    'confirmation': E8['confirmation'],
    'pairs_all': E8['pairs_all'],
    'PATCH_L': E8['PATCH_L'],
    'primary_layer': E8['primary_layer'],
    'control_layer': E8['control_layer'],
    'pre_layer': E8['pre_layer'],
    'n_heads': E8['n_heads'],
    'head_dim': E8['head_dim'],
    # ---- Phase 10 冻结量 ----
    'depth_sites': [7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
    'profile_sites': [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
    'readout_site': 'R',
    'own_basis_sites': [7, 12, 20, 34],
    'floor_sites': [7, 20, 34],
    'conf_sites': [7, 11, 20, 34],
    'f3_sites': [7, 20, 34, 'R'],
    'arms': {
        'E0_anchor_L6': {
            'site': 6, 'alpha_grid': [1.0], 'panel': 'discovery',
            'vector': 'h6_recip + 1.0 * P_U6(diff6)',
            'purpose': '内建跨 Phase 复现点，必须逐位等于 10.574739583333335',
        },
        'E0b_anchor_rbar_L6': {
            'site': 6, 'alpha_grid': 'rbar9', 'panel': 'discovery',
            'vector': 'h6_recip + rbar9 * P_U6(diff6)',
            'purpose': '读出口径锚点，应对齐 Phase 9 D1a = +0.3335（tol 0.15）',
        },
        'E1_depth_abs': {
            'sites': 'profile_sites', 'panel': 'discovery',
            'alpha_grid': [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25],
            'vector': 'h_ell_recip + alpha * P_U6(diff6)',
            'purpose': '主臂：绝对剂量深度剖面（含 L6 参照点）',
        },
        'E1b_depth_rel': {
            'sites': 'profile_sites', 'panel': 'discovery',
            'alpha_grid': [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80],
            'vector': 'h_ell_recip + a_rel * ||h_ell_recip|| * unit(P_U6(diff6))',
            'purpose': '混淆控制：相对剂量深度剖面（amend1 A1 把网格由 4 点扩到 7 点，与 E1 对称）',
        },
        'E3_own_basis': {
            'sites': 'own_basis_sites', 'panel': 'discovery',
            'alpha_grid': [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25],
            'vector': 'h_ell_recip + alpha * P_U{ell}(diff_ell)',
            'purpose': '自基稳健性',
        },
        'E4_floor': {
            'sites': 'floor_sites', 'panel': 'discovery',
            'alpha_grid': [1.0], 'draws': 2,
            'vector': 'h_ell_recip + 1.0 * ||u6|| * unit(z @ U6), z~N(0,I5)',
            'purpose': '地板（范数匹配随机 5 维）',
        },
        'E5_conf': {
            'sites': 'conf_sites', 'panel': 'confirmation',
            'alpha_grid': [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25],
            'vector': 'h_ell_recip + alpha * P_U6(diff6)',
            'purpose': '确认集验带',
        },
        'E6_readout_grid': {
            'site': 'R', 'panel': 'discovery',
            'alpha_grid': [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            'vector': 'hR_recip + alpha * P_U6(diff6)',
            'purpose': 'amend1 A5：R 位点的扩展网格（r_R 只有深度位点的 1/3，同网格打不动）；V_readout 由本臂给出',
        },
    },
    'rbar_ref_from_phase9': R9['dose_coord']['rbar'],
    'mean_n6_ref_from_phase9': R9['dose_coord']['mean_n6'],
    'n6_drift_tol': 0.02,
    'full_ref_from_phase9': R9['full'],
    'anchor_ref_d1a_from_phase9': R9['D1a']['dDonor'],
    'anchor_drift_tol': 0.15,
    'off_manifold_pert_rel': 0.5,
    'classifier': {
        'logistic_k_min': 1.0, 'logistic_k_max': 60.0, 'logistic_k_step': 1.0,
        'logistic_x0_step': 0.005,
        'S_STRONG': {'R2_log': 0.98, 'J': 3.0, 'k_log': 2.0},
        'S_WEAK': {'R2_log': 0.95, 'J': 2.0},
        'GRADUAL': {'R2_log': 0.95, 'J_max': 2.0},
        'LINEAR': {'R2_lin': 0.97, 'R2_log_max': 0.95},
        'UNREACH_y': 0.15,
    },
    'decision': {
        'P4': {'J_anchor': 3.0, 'window': [7, 10]},
        'P1': {'J': 3.0, 'post_slack': 1.3, 'x_star_tol': 0.25},
        'P2': {'spearman_min': 0.6, 'ratio_min': 1.5},
        'P3': {'J_max': 2.0},
        'priority': ['P4', 'P1', 'P2', 'P3', 'P0_no_verdict'],
    },
    'smoke_env': 'SMOKE=1 -> discovery 前 2 实例；depth_sites 截为前 3；own_basis_sites 截为前 2；'
                 'conf_sites 清空（不跑确认集）；alpha 网格截为前 3 点并强制保留 alpha=1；'
                 'E1b 网格截为前 2 点；冒烟产物落 smoke/',
    'result_keys': [
        'rbar', 'n6_ref_check', 'full_L6', 'E0', 'E0b', 'profile_abs', 'profile_rel',
        'E3', 'E4', 'E5', 'classifier', 'verdict', 'floors', 'elapsed_s', 'drift_flags',
    ],
    'panel_fingerprint': {
        'discovery': [x[0] for x in E8['discovery']],
        'confirmation': [x[0] for x in E8['confirmation']],
        'pairs_n': len(E8['pairs_all']),
        'seed': E8['seed'],
    },
    'amend1': {
        'file': 'tests/deepseek_temp/Phase10/N2h1a3_design_seal_amend1.json',
        'sha8_at_amend': sha256(os.path.join(P10T, 'N2h1a3_design_seal_amend1.json'))[:8],
        'frozen_before_formal_run': True,
        'changes': [
            'A1 E1b 网格 4 点 -> 7 点（与 E1 对称，J 的噪声结构同阶）',
            'A2 E1/E1b 位点加入 L6 输出作为剖面第一个参照点（Phase 9 D1 站点）',
            'A3 新增次级判据族 Q1/Q2/Q3（方向修正），与封存 P1-P4 并列报告，不替换',
            'A4 代码层：r_ell 报 L6 参照、Spearman 丢无效 J、新增剖面数组行',
            'A5 新增臂 E6_readout_grid：R 位点扩展网格 [0,0.5,1,2,4,8,16]，V_readout 由该臂给出',
        ],
        'panel_unchanged': True,
        'thresholds_unchanged': True,
        'sealed_P_rules_unchanged': True,
    },
    'secondary_decisions_amend1': {
        'Q1_readout_origin': 'max(J)/min(J) <= 1.5 over 全剖面 AND R 位点 class in {S_STRONG,S_WEAK}',
        'Q2_accumulate': 'spearman(J(l), l) <= -0.6 AND J(min L) >= 1.5 * J(max L)',
        'Q3_single_layer': 'exists l0: J(l0) >= 2*J(l0+1) AND J(l0) >= 3.0 AND max/min of J over l <= l0, l in L, <= 1.5',
        'priority': ['Q3_single_layer', 'Q1_readout_origin', 'Q2_accumulate', 'Q0_no_verdict'],
    },
}

# ---- F5 面板继承断言（逐元素）----
for f in ['sup_id', 'classes', 'instances_all', 'discovery', 'confirmation', 'pairs_all',
          'PATCH_L', 'primary_layer', 'control_layer', 'pre_layer', 'template', 'seed']:
    a = json.dumps(E8[f], sort_keys=True, ensure_ascii=False)
    b = json.dumps(execu[f], sort_keys=True, ensure_ascii=False)
    assert a == b, 'F5 面板继承断言失败: %s' % f
    a9 = json.dumps(E9[f], sort_keys=True, ensure_ascii=False)
    assert a9 == b, 'F5b 与 Phase 9 执行档不一致: %s' % f
print('F5 面板继承断言通过（12 字段 x Phase8/Phase9 双向一致）')

# ---- 数字锚继承断言 ----
# ---- 数字锚继承断言（来源 = Phase 9 实测结果，不是硬编码）----
assert abs(execu['mean_n6_ref_from_phase9'] - 17.06125152401808) < 1e-9
assert abs(execu['rbar_ref_from_phase9'] - 0.23676036482771085) < 1e-12
assert abs(execu['full_ref_from_phase9'] - 10.574739583333335) < 1e-12
assert abs(execu['anchor_ref_d1a_from_phase9'] - 0.3334635416666665) < 1e-12
print('数字锚继承断言通过（Phase 9 实测值）')

io.open(OUT, 'w', encoding='utf-8').write(json.dumps(execu, ensure_ascii=False, indent=1))
print('WROTE %s (%d B) sha8=%s' % (OUT, os.path.getsize(OUT), sha256(OUT)[:8]))
print('seal sha8 =', sha256(SEAL)[:8])
