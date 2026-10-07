# -*- coding: utf-8 -*-
"""把 amend1 登记进 execution_phase8.json（仅追加元数据，不动面板/配对/种子）。"""
import os, io, json, hashlib

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase8\execution_phase8.json'
A = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase8\N2h1a_design_seal_amend1.json'
E = json.load(io.open(P, encoding='utf-8'))
h_before = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
panel_fingerprint = {'discovery': [x[0] for x in E['discovery']],
                     'confirmation': [x[0] for x in E['confirmation']],
                     'pairs_n': len(E['pairs_all']), 'seed': E['seed']}
E['amend1'] = {
    'id': 'N2h1a-amend1',
    'file': 'tests/deepseek_temp/Phase8/N2h1a_design_seal_amend1.json',
    'sha256': hashlib.sha256(open(A, 'rb').read()).hexdigest(),
    'reason': 'SMOKE 暴露分量效应不可加（超可加 5.1x）；主判据改为向量预算 share_v',
    'panel_unchanged': True,
}
E['result_keys'] = ['base_ok', 'T', 'Z', 'LOO', 'L7', 'W', 'curve', 'shares_T', 'shares_Z',
                    'amend1', 'gates', 'verdict', 'confirmation', 'elapsed_s', 'jump_flag']
E['panel_fingerprint'] = panel_fingerprint
io.open(P, 'w', encoding='utf-8').write(json.dumps(E, ensure_ascii=False, indent=1))
h_after = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
E2 = json.load(io.open(P, encoding='utf-8'))
fp2 = {'discovery': [x[0] for x in E2['discovery']], 'confirmation': [x[0] for x in E2['confirmation']],
       'pairs_n': len(E2['pairs_all']), 'seed': E2['seed']}
print('exec sha8 %s -> %s ; panel_identical %s ; bytes %d' %
      (h_before, h_after, fp2 == panel_fingerprint, os.path.getsize(P)))
print('amend1 sha256', E['amend1']['sha256'][:16])
