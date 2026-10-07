# -*- coding: utf-8 -*-
"""
Phase 11 / N2h1-alpha-4 : 生成 execution_phase11.json
- 面板：对 execution_phase8.json 与 execution_phase10.json 双向逐元素断言（F5）
- 数字锚：继承 result_phase10.json（Phase 9 的 full / rbar / D1a 等）
- 唯一改动：own_basis_sites 由 4 个位点铺满全部 18 个剖面位点；新增 bootstrap 段与逐对落盘开关
"""
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T = lambda *a: os.path.join(ROOT, *a)
P8 = T('tests', 'deepseek_temp', 'Phase8', 'execution_phase8.json')
P10 = T('tests', 'deepseek_temp', 'Phase10', 'execution_phase10.json')
R10 = T('tests', 'deepseek_temp', 'Phase10', 'result_phase10.json')
OUT = T('tests', 'deepseek_temp', 'Phase11', 'execution_phase11.json')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


E8 = json.load(io.open(P8, encoding='utf-8'))
E10 = json.load(io.open(P10, encoding='utf-8'))

PANEL = ['sup_id', 'classes', 'instances_all', 'discovery', 'confirmation', 'pairs_all',
         'PATCH_L', 'primary_layer', 'control_layer', 'pre_layer', 'template', 'seed']

execu = {}
for f in PANEL:
    assert f in E8, 'Phase 8 执行档缺字段 %s' % f
    assert f in E10, 'Phase 10 执行档缺字段 %s' % f
    a = json.dumps(E8[f], sort_keys=True, ensure_ascii=False)
    b = json.dumps(E10[f], sort_keys=True, ensure_ascii=False)
    assert a == b, 'F5 面板继承断言失败 (Phase8 vs Phase10): %s' % f
    execu[f] = E10[f]

PROFILE = list(E10['profile_sites'])
DEPTH = list(E10['depth_sites'])
assert PROFILE == [6] + DEPTH, 'profile_sites 应为 [6]+depth_sites'
assert len(PROFILE) == 18, 'profile_sites 应为 18 个位点，实为 %d' % len(PROFILE)
assert E10['own_basis_sites'] == [7, 12, 20, 34], 'Phase 10 own_basis_sites 应为 4 位点'

# ---- 逐字段继承（无改动项） ----
for f in ['model', 'model_dir', 'config_sha256', 'tok_sha256', 'expected_cfg', 'o_proj_in_features',
          'n_heads', 'head_dim', 'depth_sites', 'profile_sites', 'readout_site',
          'floor_sites', 'conf_sites', 'f3_sites',
          'rbar_ref_from_phase9', 'mean_n6_ref_from_phase9', 'n6_drift_tol',
          'full_ref_from_phase9', 'anchor_ref_d1a_from_phase9', 'anchor_drift_tol',
          'off_manifold_pert_rel', 'classifier', 'decision', 'secondary_decisions_amend1']:
    assert f in E10, 'Phase 10 执行档缺字段 %s' % f
    execu[f] = E10[f]

# ---- 本轮唯一实质改动：E3 铺满全剖面 ----
execu['own_basis_sites'] = PROFILE                       # 18 位点
execu['own_basis_sites_phase10'] = E10['own_basis_sites']  # 记账：Phase 10 只覆盖这 4 个

ARMS = json.loads(json.dumps(E10['arms'], ensure_ascii=False))
ARMS['E3_own_basis']['sites'] = 'profile_sites (全 18 个位点；Phase 10 只有 [7,12,20,34])'
ARMS['E3_own_basis']['purpose'] = ('基变换稳健性（本 Phase 铺满全剖面）：用本位点自基（含本位点自 diff）'
                                   '复测，得到双基 J(ell) 剖面；ell=6 处与 E1 数学恒等（F8）')
ARMS['E3_own_basis']['per_pair_landing'] = True
execu['arms'] = ARMS

# ---- 新增加重：bootstrap / permutation ----
execu['bootstrap'] = dict(
    scheme='pair-level percentile bootstrap (B=2000)',
    B=2000, B_perm=2000, seed=20261001,
    ci='percentile 2.5/97.5', j_only=True,
    zero_extra_forward=True,
    note='E1/E1b/E3 前向次数与 Phase 10 同阶；只在循环内一并落盘逐对 dDonor',
)
execu['per_pair_landing'] = dict(E1=True, E1b=True, E3=True, E4=False, E5=True, E6=False)
execu['phase10_result_for_F9'] = 'tests/deepseek_temp/Phase10/result_phase10.json'
execu['phase10_result_sha256'] = sha(R10)

execu['smoke_env'] = ('SMOKE=1 -> discovery 取 6 个类各第一个实例（保证 U6 满秩）；profile/own_basis 位点截为 '
                      '[6,7,8]；floor/f3 位点同截；conf_sites 清空；alpha 网格截为前 3 点并强制保留 alpha=1；'
                      'E1b 网格截为前 2 点；bootstrap B 截为 200；冒烟产物落 smoke/')
execu['result_keys'] = ['E0', 'E0b', 'full_L6', 'bit_replication', 'dose_coord',
                        'profile_abs', 'profile_rel', 'profile_R_ext',
                        'E1_pairs', 'E1b_pairs', 'E3_pairs',
                        'E3', 'E3_verdict', 'bootstrap_band', 'permutation_null',
                        'E4', 'E5', 'verdict', 'floors', 'elapsed_s', 'drift_flags']

# ---- 元数据（首版遗漏，导致主脚本 KeyError: inherits_panel_sha256）----
execu['phase'] = 11
execu['name'] = 'N2h1-alpha-4 / own-basis-full-profile + noise-band'
execu['frozen_at'] = '2026-10-01 23:50'
execu['inherits_panel_from'] = 'tests/deepseek_temp/Phase10/execution_phase10.json'
execu['inherits_panel_sha256'] = sha(P10)
execu['inherits_panel8_from'] = 'tests/deepseek_temp/Phase8/execution_phase8.json'
execu['inherits_panel8_sha256'] = sha(P8)
execu['inherits_numbers_from'] = 'tests/deepseek_temp/Phase10/result_phase10.json'
execu['inherits_numbers_sha256'] = sha(R10)
execu['phase10_result_for_F9'] = 'tests/deepseek_temp/Phase10/result_phase10.json'
execu['phase10_result_sha256'] = sha(R10)

REQUIRED = ['phase', 'name', 'model', 'model_dir', 'config_sha256', 'tok_sha256', 'expected_cfg',
            'template', 'seed', 'sup_id', 'classes', 'instances_all', 'discovery', 'confirmation',
            'pairs_all', 'PATCH_L', 'primary_layer', 'control_layer', 'pre_layer', 'n_heads', 'head_dim',
            'depth_sites', 'profile_sites', 'readout_site', 'own_basis_sites', 'floor_sites', 'conf_sites',
            'f3_sites', 'arms', 'rbar_ref_from_phase9', 'mean_n6_ref_from_phase9', 'n6_drift_tol',
            'full_ref_from_phase9', 'anchor_ref_d1a_from_phase9', 'anchor_drift_tol', 'off_manifold_pert_rel',
            'classifier', 'decision', 'secondary_decisions_amend1', 'bootstrap', 'per_pair_landing',
            'inherits_panel_sha256', 'inherits_panel8_sha256', 'inherits_numbers_sha256',
            'phase10_result_sha256', 'smoke_env', 'result_keys']
missing = [f for f in REQUIRED if f not in execu]
assert not missing, 'execution 缺必需字段: %s' % missing

io.open(OUT, 'w', encoding='utf-8').write(json.dumps(execu, ensure_ascii=False, indent=1))
print('WROTE', OUT, '(%d fields, required check PASS)' % len(execu))
print('  inherits_panel_from(phase10) sha256 =', sha(P10))
print('  inherits_panel_from(phase8)  sha256 =', sha(P8))
print('  inherits_numbers_from(phase10 result) sha256 =', sha(R10))
print('  profile_sites =', PROFILE)
print('  own_basis_sites =', execu['own_basis_sites'])
print('  bootstrap =', execu['bootstrap']['scheme'], 'B=%d B_perm=%d' % (execu['bootstrap']['B'], execu['bootstrap']['B_perm']))
print('  F5 panel assert PASS (Phase8 vs Phase10, %d fields)' % len(PANEL))
