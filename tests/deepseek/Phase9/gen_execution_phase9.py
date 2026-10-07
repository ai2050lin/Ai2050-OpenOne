# -*- coding: utf-8 -*-
"""生成 Phase 9 执行冻结档 execution_phase9.json（观测前）。

纪律：面板必须与 Phase 8 逐元素相等（F5 断言），臂规格/alpha 网格在此冻结，
主脚本只读本文件，不再自带任何网格常数。
"""
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P8T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8')
P9T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9')
IN8 = os.path.join(P8T, 'execution_phase8.json')
OUT9 = os.path.join(P9T, 'execution_phase9.json')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


E8 = json.load(io.open(IN8, encoding='utf-8'))

# ---------- 源档形状检查（F5 的逐元素对账在文件末尾对 execu 执行） ----------
assert len(E8['discovery']) == 24 and len(E8['confirmation']) == 17 and len(E8['pairs_all']) == 41
assert E8['primary_layer'] == 6 and E8['pre_layer'] == 5
cfgj = json.loads(io.open(os.path.join(MDIR, 'config.json'), encoding='utf-8').read())
assert sha(os.path.join(MDIR, 'config.json')) == E8['config_sha256'], 'config 变了，F4 停'
assert cfgj['num_attention_heads'] * cfgj['head_dim'] == 4096

# ---------- 本 Phase 冻结的臂与网格 ----------
ARMS = {
    'D1_write_readout': {
        'site': 'L6out', 'alpha_grid': [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25],
        'vector': 'h6_recip + alpha * P_U6(diff6)', 'panel': 'discovery',
        'purpose': '写入->行为的传递函数；alpha=1 为 Phase 8 diff6 内建复现点',
    },
    'D1a_anchor': {
        'site': 'L6out', 'alpha_grid': 'rbar', 'vector': 'h6_recip + rbar * P_U6(diff6)',
        'panel': 'discovery', 'purpose': '锚点，应对齐 Phase 8 diff5 臂 (+0.288)',
    },
    'D2_upstream': {
        'site': 'L5out', 'alpha_grid': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.32, 6.0],
        'vector': 'h5_recip + alpha * P_U6(diff5)', 'panel': 'discovery',
        'purpose': '死线主臂：上游 U 分量剂量 + L6 输出 U-轴增益 amp(alpha)',
    },
    'D2b_basis_swap': {
        'site': 'L5out', 'alpha_grid': [1.0, 2.0, 4.32],
        'vector': 'h5_recip + alpha * P_U5(diff5)', 'panel': 'discovery',
        'purpose': '基变换稳健性',
    },
    'D3_orth_complement': {
        'site': 'L6out', 'alpha_grid': [0.5, 1.0, 1.5],
        'vector': 'h6_recip + alpha * (diff6 - P_U6(diff6))', 'panel': 'discovery',
        'purpose': '特异性：正交补应近地板',
    },
    'D4_random5': {
        'site': 'L6out', 'alpha_grid': [0.25, 0.5, 1.0], 'draws': 3,
        'vector': 'h6_recip + alpha * ||u6|| * unit(z @ U6), z~N(0,I5)', 'panel': 'discovery',
        'purpose': '地板',
    },
    'D6_upstream_orth': {
        'site': 'L5out', 'alpha_grid': [1.0, 4.32],
        'vector': 'h5_recip + alpha * (diff5 - P_U6(diff5))', 'panel': 'discovery',
        'purpose': '上游位点特异性对照',
    },
    # ---- amend1（正式运行前冻结）：补必要性对偶臂 ----
    'D7_l5_u_necessity': {
        'site': 'L5out', 'alpha_grid': [0.0, 0.5, 1.0], 'panel': 'discovery',
        'vector': 'h5_donor - (1-alpha) * P_U6(diff5)   [文本 = 供体句]',
        'baseline': '供体句自身前向的供体类分数 sD1（alpha=1 = 恒等 -> 差值 0）',
        'purpose': '上游 U 分量撤除的必要性，与 D2 的充分性构成对偶；kill_frac = drop(0)/(mean(sd0)-mean(sD1))',
    },
    'C3_conf_D7': {
        'site': 'L5out', 'alpha_grid': [0.0, 1.0], 'panel': 'confirmation',
        'vector': 'h5_donor - (1-alpha) * P_U6(diff5)   [文本 = 供体句]',
        'purpose': '确认集必要性验带',
    },
    'C1_conf_D1': {
        'site': 'L6out', 'alpha_grid': [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25],
        'vector': 'h6_recip + alpha * P_U6(diff6)', 'panel': 'confirmation',
        'purpose': '确认集验带',
    },
    'C2_conf_D2': {
        'site': 'L5out', 'alpha_grid': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.32, 6.0],
        'vector': 'h5_recip + alpha * P_U6(diff5)', 'panel': 'confirmation',
        'purpose': '确认集验带',
    },
}

execu = {
    'phase': 9,
    'name': 'N2h1-alpha-2 / threshold_gain_dose_response',
    'frozen_at': '2026-10-01 02:40',
    'inherits_panel_from': 'tests/deepseek_temp/Phase8/execution_phase8.json',
    'inherits_panel_sha256': sha(IN8),
    'model': E8['model'],
    'model_dir': E8['model_dir'],
    'config_sha256': E8['config_sha256'],
    'tok_sha256': E8['tok_sha256'],
    'expected_cfg': E8['expected_cfg'],
    'o_proj_in_features': E8['o_proj_in_features'],
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
    'arms': ARMS,
    'rbar_ref_from_phase8': 4.19 / 18.09,
    'rbar_drift_tol': 0.02,
    'anchor_ref_dDonor_phase8_diff5': 0.28815104166666655,
    'anchor_drift_tol': 0.15,
    'off_manifold_pert_rel': 0.50,
    'smoke_env': 'SMOKE=1 -> discovery 前 2 实例、alpha 网格截断为前 3 点(保留 alpha=1) + 形状/维数/NaN/alpha=0 恒等断言；冒烟产物落 smoke/',
    'amend1': {
        'file': 'tests/deepseek_temp/Phase9/N2h1a2_design_seal_amend1.json',
        'reason': 'SMOKE 暴露设计缺口：D2 只测上游 U 分量的"充分性"，若读数近零则无法区分"上游不重要"与"上游必要但不充分" => 补充必要性对偶臂 D7（供体句 + 末位撤除 P_U6(diff5) 的 alpha 梯度）。同时把 detAmp 空表守卫与 matched-dose 地板比列为代码/报告层修正，不改任何已冻结阈值。',
        'frozen_before_formal_run': True,
        'panel_unchanged': True,
        'thresholds_unchanged': True,
    },
    'result_keys': ['rbar', 'u5_u6_principals', 'D1', 'D1a', 'D2', 'D2b', 'D3', 'D4', 'D6', 'D7',
                    'amp', 'pert_rel', 'curves', 'fits', 'gates', 'verdict', 'floors',
                    'confirmation', 'elapsed_s', 'drift_flags'],
    'panel_fingerprint': {
        'discovery': [x[0] for x in E8['discovery']],
        'confirmation': [x[0] for x in E8['confirmation']],
        'pairs_n': len(E8['pairs_all']),
        'seed': E8['seed'],
    },
}

# ---------- 与 Phase 8 的显式逐元素对账（真正执行 F5） ----------
for f in ['sup_id', 'classes', 'instances_all', 'discovery', 'confirmation', 'pairs_all',
          'PATCH_L', 'primary_layer', 'control_layer', 'pre_layer', 'template', 'seed']:
    a = json.dumps(E8[f], sort_keys=True, ensure_ascii=False)
    b = json.dumps(execu[f], sort_keys=True, ensure_ascii=False)
    assert a == b, 'F5 面板继承断言失败: %s' % f

io.open(OUT9, 'w', encoding='utf-8').write(json.dumps(execu, ensure_ascii=False, indent=1))
print('wrote', OUT9, os.path.getsize(OUT9), 'bytes')
print('F5 panel inheritance OK ; discovery %d confirmation %d pairs %d' %
      (len(execu['discovery']), len(execu['confirmation']), len(execu['pairs_all'])))
print('arms', len(ARMS), '; inheritance sha256', execu['inherits_panel_sha256'][:16])
print('rbar_ref_from_phase8 = %.5f' % execu['rbar_ref_from_phase8'])
