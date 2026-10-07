# -*- coding: utf-8 -*-
"""Probe 3118/3119 npz keys and result.json for Phase 3120 design."""
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D18 = os.path.join(RDIR, 'phase3118',
                   'omega_p116_autoregressive_margin_'
                   'trajectory')
D19 = os.path.join(RDIR, 'phase3119',
                   'omega_p117_oscillation_attribution_'
                   'writemap_linearity')
OUTF = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3120_probe_out.txt'
lines = []

z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
lines.append('== 3118 traj_readout.npz keys ==')
for k in sorted(z18.files):
    lines.append('  %s : %s %s'
                 % (k, z18[k].shape, z18[k].dtype))

r18 = json.load(io.open(os.path.join(D18, 'result.json'),
                        encoding='utf-8'))
lines.append('== 3118 result.json top keys ==')
lines.append('  %s' % sorted(r18.keys()))
lines.append('  verdict=%s' % r18.get('verdict'))
for key in ('greedy', 'sampled', 'ablation', 'behavior'):
    if key in r18:
        lines.append('  %s: %s'
                     % (key, json.dumps(r18[key])[:2000]))

z19 = np.load(os.path.join(D19, 'wmap_readout.npz'),
              allow_pickle=False)
lines.append('== 3119 wmap_readout.npz keys ==')
for k in sorted(z19.files):
    lines.append('  %s : %s %s'
                 % (k, z19[k].shape, z19[k].dtype))

r19 = json.load(io.open(os.path.join(D19, 'result.json'),
                        encoding='utf-8'))
lines.append('== 3119 result.json top keys ==')
lines.append('  %s' % sorted(r19.keys()))
lines.append('  verdict=%s' % r19.get('verdict'))
lines.append('  part_b verdict=%s'
             % r19['part_b']['verdict'])
lines.append('  part_c verdict=%s'
             % r19['part_c']['verdict_lin'])

# mat5 predicates / entities for span labeling
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
lines.append('== mat5 keys ==')
lines.append('  %s' % sorted(mat5.keys()))
lines.append('  entities=%s'
             % json.dumps(mat5['entities'],
                          ensure_ascii=False))
lines.append('  predicates=%s'
             % json.dumps(mat5['predicates'],
                          ensure_ascii=False))
lines.append('  yes_id=%s no_id=%s'
             % (mat5.get('yes_id'),
                mat5.get('no_id')))
lines.append('  pair2rel sample=%s'
             % json.dumps(dict(list(
                 mat5['pair2rel'].items())[:3])))
lines.append('  false_rels sample=%s'
             % json.dumps(dict(list(
                 mat5['false_rels'].items())[:3])))

with io.open(OUTF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('OK wrote', OUTF)
