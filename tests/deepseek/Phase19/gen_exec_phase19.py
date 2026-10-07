# -*- coding: utf-8 -*-
"""
Phase 19 (N2h1-alpha-12) execution 生成器（冻结运行配置）。
产物：tests/deepseek_temp/Phase19/execution_phase19.json
锚：Phase 17 result（com_V 族，现场读入并逐位断言）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
SEALP = os.path.join(P19T, 'N2h1a12_design_seal.json')
R17P = os.path.join(P17T, 'result_phase17.json')
PROBE_A0 = os.path.join(P19T, '_probe19_A0_both.json')
PROBE_A1 = os.path.join(P19T, '_probe19_A1_bf16.json')


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


S = json.load(io.open(SEALP, encoding='utf-8'))
assert S['phase'] == 19 and S['kind'] == 'design_seal'

R17 = json.load(io.open(R17P, encoding='utf-8'))
# 该模型在 P17 冻结的 REACH（跨量化口径保持不变，用于配对）
REACH = {
    'qwen3-4b': [int(x) for x in R17['arms']['A0_calib_qwen3-4b-nf4']['E5_com_V']['reach']],
    'glm4-9b-chat-hf': [int(x) for x in R17['arms']['A1_glm4-9b-nf4']['E5_com_V']['reach']],
}

EX = {
    'phase': 19,
    'line': 'N2h1-alpha-12',
    'kind': 'execution',
    'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seal_sha256': fsha(SEALP),
    'seal_bytes': os.path.getsize(SEALP),
    'probe_sha256': {'A0_both': fsha(PROBE_A0), 'A1_bf16_loadonly': fsha(PROBE_A1)},

    # 冻结锚：Phase 17 result
    'anchor_result_path': 'tests\\deepseek_temp\\Phase17\\result_phase17.json',
    'anchor_result_sha256': fsha(R17P),
    'anchor_values': S['anchor_values'],
    'anchor_reference_only': S['anchor_reference_only'],
    'reach_by_model': REACH,

    # 材料（从 seal 搬运）
    'template': S['template'],
    'classes': S['classes'],
    'instances_all': S['instances_all'],
    'pairs_all': S['pairs_all'],
    'discovery': S['discovery'],
    'confirmation': S['confirmation'],
    'quant_nf4': S['quant_nf4'],
    'quant_bf16': S['quant_bf16'],
    'profile_sites': S['profile_sites'],
    'bootstrap': S['bootstrap'],
    'floors': S['floors'],
    'neighbourhood_width': S['neighbourhood_width'],
    'arm_order': S['arm_order'],
    'arms': S['arms'],
    'revision': 'v1 (seal %s)' % fsha(SEALP)[:8],
    'note': ('量化口径为唯一自变量；capture/质量谱/质心/区间求和全部逐字继承 P17。'
             'A2 只以 nf4 参与装置门（A2-bf16 实测 segfault）。'),
}

OUT = os.path.join(P19T, 'execution_phase19.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(EX, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('seal', EX['seal_sha256'][:8], '| anchor_result', EX['anchor_result_sha256'][:8])
print('probe', {k: v[:8] for k, v in EX['probe_sha256'].items()})
print('arms', EX['arm_order'])
print('reach_by_model:', {k: (len(v), v[:4]) for k, v in REACH.items()})
print('floors', json.dumps(EX['floors'], ensure_ascii=False))
print('seeds', EX['bootstrap']['seeds'], 'BP', EX['bootstrap']['BP'])
