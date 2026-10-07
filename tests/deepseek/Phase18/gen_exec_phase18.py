# -*- coding: utf-8 -*-
"""
Phase 18 (N2h1-alpha-11) execution 生成器（冻结运行配置）。
产物：tests/deepseek_temp/Phase18/execution_phase18.json
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
SEALP = os.path.join(P18T, 'N2h1a11_design_seal.json')
PROBEP = os.path.join(P18T, '_probe_feasibility_A0.json')
R16P = os.path.join(P16T, 'result_phase16.json')
R17P = os.path.join(P17T, 'result_phase17.json')
EX16P = os.path.join(P16T, 'execution_phase16.json')


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


S = json.load(io.open(SEALP, encoding='utf-8'))
assert S['phase'] == 18 and S['kind'] == 'design_seal'

EX = {
    'phase': 18,
    'line': 'N2h1-alpha-11',
    'kind': 'execution',
    'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seal_sha256': fsha(SEALP),
    'seal_bytes': os.path.getsize(SEALP),
    'probe_sha256': fsha(PROBEP),

    # 冻结锚：P16 result（行为剖面）+ P17 result（向量谱/质心/邻域）
    'anchor_result_p16_path': 'tests\\deepseek_temp\\Phase16\\result_phase16.json',
    'anchor_result_p16_sha256': fsha(R16P),
    'anchor_result_p17_path': 'tests\\deepseek_temp\\Phase17\\result_phase17.json',
    'anchor_result_p17_sha256': fsha(R17P),
    'p16_anchors': S['anchor_values']['p16'],
    'p17_anchors': S['anchor_values']['p17'],

    # 运行配置（从 seal 搬运）
    'template': S['template'],
    'classes': S['classes'],
    'instances_all': S['instances_all'],
    'pairs_all': S['pairs_all'],
    'discovery': S['discovery'],
    'confirmation': S['confirmation'],
    'quant': S['quant'],
    'profile_sites': S['profile_sites'],
    'components': S['components'],
    'components_confirmation': S['components_confirmation'],
    'bootstrap': S['bootstrap'],
    'floors': S['floors'],
    'neighbourhood_width': S['neighbourhood_width'],
    'arm_order': S['arm_order'],
    'arms': json.load(io.open(EX16P, encoding='utf-8'))['arms'],
    'revision': 'v1 (seal ab0ef1b3)',
    'note': ('唯一改动 = 把 P17 的向量质量谱读成行为效应谱：注入物（Delta_inc/Delta_mlp/Delta_attn/'
             'Delta_head_h*/d_l 的 U 投影）逐字沿用 P17，读数换成 Phase 8 T 臂口径的 dDonor。'
             '位点 = 1..L-2（与 P17 的 w_all 索引对齐）；组件 5 条（discovery）/3 条（confirmation）。'),
}

OUT = os.path.join(P18T, 'execution_phase18.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(EX, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('seal', EX['seal_sha256'][:8], 'probe', EX['probe_sha256'][:8],
      'p16_result', EX['anchor_result_p16_sha256'][:8], 'p17_result', EX['anchor_result_p17_sha256'][:8])
print('floors', json.dumps(EX['floors'], ensure_ascii=False))
print('seeds', EX['bootstrap']['seeds'], 'BP', EX['bootstrap']['BP'])
print('components', EX['components'], '/', EX['components_confirmation'])
print('arm_order', EX['arm_order'])
