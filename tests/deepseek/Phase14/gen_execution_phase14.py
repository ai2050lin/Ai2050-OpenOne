# -*- coding: utf-8 -*-
"""
Phase 14 / N2h1-alpha-7 : 执行参数冻结
=============================================================================
生成 tests/deepseek_temp/Phase14/execution_phase14.json
依赖：seal 已冻结（N2h1a7_design_seal.json），且运行前的 seal 哈希必须与 seal 内记录一致。
"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
SEAL_P = os.path.join(P14T, 'N2h1a7_design_seal.json')
AMEND_P = os.path.join(P14T, 'N2h1a7_design_seal_amend1.json')
AMEND2_P = os.path.join(P14T, 'N2h1a7_design_seal_amend2.json')
EXEC_P = os.path.join(P14T, 'execution_phase14.json')


def sha256(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest()


def jload(p):
    return json.load(io.open(p, encoding='utf-8'))


S = jload(SEAL_P)
AM = jload(AMEND_P)
AM2 = jload(AMEND2_P)
ACF2 = AM2['fix_3_new_arm']
E12 = jload(os.path.join(P12T, 'execution_phase12.json'))
R12 = jload(os.path.join(P12T, 'result_phase12.json'))

SITES = [int(s) for s in R12['xhalf']['sites']]
MDIR = os.path.join(ROOT, 'models', 'hf', S['model']['name'])
ACF = AM['corrected_fields']
EXP = ACF['expected_cfg']

exec_ = {
    'phase': 14,
    'name': S['name'],
    'seal_sha256': sha256(SEAL_P),
    'seal_sha8': sha256(SEAL_P)[:8],
    'amend1': {
        'path': 'tests/deepseek_temp/Phase14/N2h1a7_design_seal_amend1.json',
        'sha256': sha256(AMEND_P),
        'sha8': sha256(AMEND_P)[:8],
        'kind': AM['kind'],
        'trigger': AM['trigger'],
    },
    'amend2': {
        'path': 'tests/deepseek_temp/Phase14/N2h1a7_design_seal_amend2.json',
        'sha256': sha256(AMEND2_P),
        'sha8': sha256(AMEND2_P)[:8],
        'kind': AM2['kind'],
        'trigger': AM2['trigger'],
        'FULL_SWAP_definition_frozen': AM2['fix_2_normalization_freeze']['FULL_SWAP_definition_frozen'],
        'F30_new': AM2['fix_1_anchor_scoping']['F30_new'],
        'F29_new': AM2['fix_1_anchor_scoping']['F29_new'],
    },
    'model': S['model']['name'],
    'model_dir': os.path.relpath(MDIR, ROOT),
    'config_sha256': S['model']['config_sha256'],
    'tok_sha256': sha256(os.path.join(MDIR, 'tokenizer.json')) if os.path.exists(os.path.join(MDIR, 'tokenizer.json')) else None,
    'template': S['model']['template'],
    'seed': 20261001,
    'classes': list(E12['classes']),
    'sup_id': {k: int(v) for k, v in E12['sup_id'].items()},
    'panel': {
        'discovery': [list(x) for x in E12['discovery']],
        'confirmation': [list(x) for x in E12['confirmation']],
        'instances_all': [list(x) for x in E12['instances_all']],
        'pairs_all': [list(x) for x in E12['pairs_all']],
    },
    'sites': {
        'profile': SITES,
        'layers_count': 36,
        'readout': 'R',
        'conf': [7, 11, 20, 34],
        'floor': [7, 20, 34],
        'f3': [7, 20, 34, 'R'],
        'noop_check': [7, 20, 34, 'R'],
    },
    'masks': {'all': [0, 1], 'pos0': [0], 'pos1': [1], 'none': []},
    'alpha_grids': {
        'legacy': S['dose_coordinate']['alpha_grid_legacy'],
        'dense': S['dose_coordinate']['alpha_grid_dense'],
        'conf': S['dose_coordinate']['alpha_grid_conf'],
    },
    'stats': {
        'xhalf_frac': 0.5,
        'jdose_floor': 0.01,
        'window_W': 3,
        'n_windows': 15,
    },
    'bootstrap': {'BS': 2000, 'BP': 2000, 'seed': 20261001,
                  'paired_scheme': 'Delta_b(w) = F_b(sites[w]) - F_b(sites[w+3])'},
    'arms': {
        'A1_prefix_layer_sweep': {'mask': [0, 1], 'sites': SITES, 'alpha': S['dose_coordinate']['alpha_grid_dense'], 'pairs': 24},
        'A2_readout_prefix': {'mask': [0, 1], 'sites': ['R'], 'alpha': S['dose_coordinate']['alpha_grid_legacy'], 'pairs': 24},
        'A3a_position_curves_L6': {'mask': [[0], [1]], 'sites': [6], 'alpha': S['dose_coordinate']['alpha_grid_dense'], 'pairs': 24},
        'A3b_position_endpoints': {'mask': [[0], [1]], 'sites': SITES, 'alpha': [1.0], 'pairs': 24},
        'A4_confirmation': {'mask': [0, 1], 'sites': [7, 11, 20, 34], 'alpha': S['dose_coordinate']['alpha_grid_conf'], 'pairs': 17},
        'A5_floor': {'mask': [0, 1], 'sites': [7, 20, 34], 'alpha': [1.0], 'pairs': 24, 'kind': 'random_5dim_in_U6'},
        'A8_cumulative_layer': {'support_order_i': '0..17', 'sites': SITES,
                                'alpha': S['dose_coordinate']['alpha_grid_legacy'],
                                'pairs': 24, 'def': AM2['fix_3_new_arm']['definition']},
    },
    'expected_fwd_total': 16268,
    'expected_cfg': EXP,
    'n_heads': ACF['n_heads'], 'head_dim': ACF['head_dim'],
    'n_kv_heads': ACF['n_kv_heads'],
    'o_proj_in_features': ACF['o_proj_in_features'],
    'off_manifold_pert_rel': float(E12['off_manifold_pert_rel']),
    'n6_drift_tol': float(E12['n6_drift_tol']),
    'anchor_drift_tol': 1e-9,
    'inherits': {
        'phase12_result_sha256': S['inheritance_anchors']['phase12_result_sha256'],
        'phase12_result_sha8': S['inheritance_anchors']['phase12_result_sha8'],
        'phase13_result_sha256': S['inheritance_anchors']['phase13_result_sha256'],
        'phase13_result_sha8': S['inheritance_anchors']['phase13_result_sha8'],
        'FULL_SWAP': S['inheritance_anchors']['inherited_published']['FULL_SWAP'],
        'mean_n6_ref_phase9': S['inheritance_anchors']['inherited_published']['mean_n6_ref_phase9'],
        'sing_U6': S['inheritance_anchors']['inherited_published']['sing_U6'],
        'recover_12_by_site': S['inheritance_anchors']['inherited_published']['recover_12_by_site'],
        'XH_12_by_site': S['inheritance_anchors']['inherited_published']['XH_12_by_site'],
        'J_swap_12_by_site': S['inheritance_anchors']['inherited_published']['J_swap_12_by_site'],
        'XH_RANGE_12': S['inheritance_anchors']['inherited_published']['XH_RANGE_12'],
        'Q_ELL_12': S['inheritance_anchors']['inherited_published']['Q_ELL_12'],
        'MODE_X_13': S['inheritance_anchors']['inherited_published']['MODE_X_13'],
        'MODE_J_13': S['inheritance_anchors']['inherited_published']['MODE_J_13'],
    },
    'decision': S['decision'],
    'floors': dict(S['floors'], F29=AM2['fix_1_anchor_scoping']['F29_new'],
                    F30=AM2['fix_1_anchor_scoping']['F30_new']),
    'result_keys': [
        'phase', 'name', 'model', 'smoke', 'elapsed_s',
        'seal_sha8', 'seal_sha256', 'exec_sha8',
        'inherits', 'panel', 'layers', 'sites', 'dose_coord',
        'A0a_full_swap', 'A0b_n6', 'A0c_u6', 'A0d_noop', 'A0e_tokenizer',
        'A1_curves', 'A1_xhalf', 'A1_J', 'A1_Jiqr', 'A1_logistic', 'A1_perpair',
        'A2_readout', 'A3a_position_curves', 'A3b_position_endpoints', 'A3_position_summary',
        'A4_confirmation', 'A5_floor',
        'A6_concentration', 'A6_bootstrap_bands', 'A6_null',
        'A8_curves', 'A8_xhalf', 'A8_J', 'A8_verdict', 'A8_predictions',
        'A7_range_grid', 'A7_steepness_alt', 'A7_paired_delta',
        'predictions_check', 'verdict', 'floors', 'extra',
    ],
    'smoke_env': 'SMOKE=1',
    'phase': 14,
    'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}

with io.open(EXEC_P, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(exec_, f, ensure_ascii=False, indent=1)
    f.write('\n')

b = io.open(EXEC_P, 'rb').read()
print('WROTE %s' % EXEC_P)
print('  bytes  = %d' % len(b))
print('  sha8   = %s' % hashlib.sha256(b).hexdigest()[:8])
print('  keys   = %d' % len(exec_))
print('  result_keys = %d' % len(exec_['result_keys']))
print('  seal_sha8 in exec = %s' % exec_['seal_sha8'])
