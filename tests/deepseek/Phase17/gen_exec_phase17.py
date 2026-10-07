# -*- coding: utf-8 -*-
"""
Phase 17 (N2h1-alpha-10) execution 生成器（冻结运行配置）。
产物：tests/deepseek_temp/Phase17/execution_phase17.json
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
SEALP = os.path.join(P17T, 'N2h1a10_design_seal.json')
PROBEP = os.path.join(P17T, '_probe_feasibility_A0.json')
R16P = os.path.join(P16T, 'result_phase16.json')


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


S = json.load(io.open(SEALP, encoding='utf-8'))
assert S['phase'] == 17 and S['kind'] == 'design_seal'

EX = {
    'phase': 17,
    'line': 'N2h1-alpha-10',
    'kind': 'execution',
    'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seal_sha256': fsha(SEALP),
    'seal_bytes': os.path.getsize(SEALP),
    'probe_sha256': fsha(PROBEP),

    # 冻结锚：Phase 16 result（P2 现场读入并逐位断言）
    'anchor_result_path': 'tests\\deepseek_temp\\Phase16\\result_phase16.json',
    'anchor_result_sha256': fsha(R16P),
    'anchor_values': S['anchor_values'],

    # 运行配置（从 seal 搬运）
    'template': S['template'],
    'classes': S['classes'],
    'instances_all': S['instances_all'],
    'pairs_all': S['pairs_all'],
    'discovery': S['discovery'],
    'confirmation': S['confirmation'],
    'quant': S['quant'],
    'profile_sites': S['profile_sites'],
    'bootstrap': S['bootstrap'],
    'floors': S['floors'],
    'span_ks': S['span_spectrum']['ks'],
    'neighbourhood_width': 2,
    'arm_order': S['arm_order'],
    'arms': json.load(io.open(os.path.join(P16T, 'execution_phase16.json'),
                              encoding='utf-8'))['arms'],
    'revision': 'v1 (seal ea0a5627)',
    'note': ('capture 扩展为 (HH, O, M, logits)；逐头贡献经**模块自身前向**的头块掩码得到；'
             '零额外前向（除 E0 自检），全部为只读测量。'),
}

OUT = os.path.join(P17T, 'execution_phase17.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(EX, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('seal', EX['seal_sha256'][:8], 'probe', EX['probe_sha256'][:8],
      'anchor_result', EX['anchor_result_sha256'][:8])
print('floors', json.dumps(EX['floors'], ensure_ascii=False))
print('seeds', EX['bootstrap']['seeds'], 'BP', EX['bootstrap']['BP'])
print('arm_order', EX['arm_order'])
