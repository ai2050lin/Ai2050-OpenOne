# -*- coding: utf-8 -*-
"""
Phase 20 (N2h1-alpha-13) execution 生成器（冻结运行配置）。
产物：tests/deepseek_temp/Phase20/execution_phase20.json
锚：P18 result（行为族）+ P16 result（com_layer 族）+ P17 result（com_V），三者 sha 现场读入。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
SEALP = os.path.join(P20T, 'N2h1a13_design_seal.json')
R18P = os.path.join(P18T, 'result_phase18.json')
R17P = os.path.join(P17T, 'result_phase17.json')
R16P = os.path.join(P16T, 'result_phase16.json')


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


S = json.load(io.open(SEALP, encoding='utf-8'))
assert S['phase'] == 20 and S['kind'] == 'design_seal'
assert os.path.exists(R18P) and os.path.exists(R16P) and os.path.exists(R17P)

EX = {
    'phase': 20, 'line': 'N2h1-alpha-13', 'kind': 'execution',
    'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seal_sha256': fsha(SEALP), 'seal_bytes': os.path.getsize(SEALP),

    'anchor_result_p18_path': 'tests\\deepseek_temp\\Phase18\\result_phase18.json',
    'anchor_result_p18_sha256': fsha(R18P),
    'anchor_result_p16_path': 'tests\\deepseek_temp\\Phase16\\result_phase16.json',
    'anchor_result_p16_sha256': fsha(R16P),
    'anchor_result_p17_path': 'tests\\deepseek_temp\\Phase17\\result_phase17.json',
    'anchor_result_p17_sha256': fsha(R17P),

    'template': S['template'], 'classes': S['classes'],
    'instances_all': S['instances_all'], 'pairs_all': S['pairs_all'],
    'discovery': S['discovery'], 'confirmation': S['confirmation'],
    'quant_nf4': S['quant_nf4'], 'quant_bf16': S['quant_bf16'],
    'components': S['components'], 'components_confirmation': S['components_confirmation'],
    'profile_sites': S['profile_sites'], 'profile_sites_legacy': S['profile_sites_legacy'],
    'alphas': S['alphas'], 'xh_frac': S['xh_frac'],
    'bootstrap': S['bootstrap'], 'floors': S['floors'],
    'neighbourhood_width': S['neighbourhood_width'],
    'arm_order': S['arm_order'], 'arms': S['arms'], 'anchors': S['anchors'],
    'revision': 'v1 (seal %s)' % fsha(SEALP)[:8],
    'note': ('唯一自变量 = 数值精度（nf4 vs bf16）。材料/域/位点/U_ℓ/口径逐字继承 P16/P18；'
             'A2（Qwen3-14B）不参与（bf16 segfault，P19 实测）。'
             'nf4 臂为校准臂（须逐位复现 P18/P16/P17 冻结锚），bf16 臂为处理臂。'),
}

OUT = os.path.join(P20T, 'execution_phase20.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(EX, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('sha8', hashlib.sha256(b).hexdigest()[:8])
print('seal', EX['seal_sha256'][:8])
print('anchors sha:', EX['anchor_result_p18_sha256'][:8], EX['anchor_result_p16_sha256'][:8],
      EX['anchor_result_p17_sha256'][:8])
print('arms', EX['arm_order'])
print('floors', json.dumps(EX['floors'], ensure_ascii=False))
print('seeds', EX['bootstrap']['seeds'], 'BP', EX['bootstrap']['BP'])
print('n_inst', len(EX['instances_all']), 'n_pairs', len(EX['pairs_all']),
      'n_disc', len(EX['discovery']), 'n_conf', len(EX['confirmation']))
