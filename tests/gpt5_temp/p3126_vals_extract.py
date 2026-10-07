# -*- coding: utf-8 -*-
"""Extract all key values from Phase 3126
result.json into p3126_vals.txt for closeout
script authoring (same pattern as 3125)."""
import io
import json
import os

OUTD = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3126'
        r'\omega_p124_glm4_anchoredlast_'
        r'regen_writechain')
VF = (r'D:\AI2050\Ai2050-OpenOne'
      r'\tests\gpt5_temp'
      r'\p3126_vals.txt')
res = json.load(io.open(
    os.path.join(OUTD, 'result.json'),
    encoding='utf-8'))
o = []
o.append('=== scalars ===')
for k in ('phase', 'name', 'smoke',
          'verdict', 'n_pairs', 'np_b',
          'np_reg', 'runtime_s'):
    o.append('%s: %r' % (k, res[k]))
o.append('')
o.append('=== part_a ===')
pa = res['part_a']
o.append('refit: %r' % (pa['refit'],))
o.append('decomp3_repl: %r'
         % (pa['decomp3_repl'],))
for dc in ('P', 'A1'):
    o.append('decomp6[%s]: %r'
             % (dc, pa['decomp6'][dc],))
    o.append('ar6[%s]: %r'
             % (dc, pa['ar6_params'][dc],))
    o.append('perm[%s]: %r'
             % (dc, pa['perm'][dc],))
o.append('long_gate: %r' % (pa['long_gate'],))
o.append('g_struct: %r' % (pa['g_struct'],))
o.append('second: %r' % (pa['second'],))
o.append('second_verdict: %r'
         % (pa['second_verdict'],))
o.append('trail6_G[P] (10x6):')
for row in pa['trail6_G']['P']:
    o.append('  %r' % (row,))
o.append('trail6_G[A1] (10x6):')
for row in pa['trail6_G']['A1']:
    o.append('  %r' % (row,))
o.append('')
o.append('=== part_b ===')
pb = res['part_b']
for k in ('interference', 'ids',
          'gen_probe_bos', 'first_tokens',
          'path', 'repro', 'readout_auc',
          'readout_verdict', 'n_span',
          'spans_verdict', 'lstar', 'sign',
          'vs_3124'):
    o.append('%s: %r' % (k, pb[k],))
cv = pb['curves']
for k in sorted(cv):
    o.append('curves.%s = %r' % (k, cv[k],))
o.append('')
o.append('=== part_c ===')
pc = res['part_c']
for k in ('np_reg', 'stats', 'shift_min',
          'verdict'):
    o.append('%s: %r' % (k, pc[k],))
o.append('')
o.append('=== part_d ===')
pd_ = res['part_d']
o.append('c3_min: %r' % (pd_['c3_min'],))
o.append('verdict: %r' % (pd_['verdict'],))
for dc in ('P', 'A1'):
    wd = pd_['layers'][dc]
    o.append('D[%s] top3: %r c3: %r peak: %r '
             'depth: %r'
             % (dc, wd['top3_layers'],
                wd['c3'], wd['peak_layer'],
                wd['peak_rel_depth']))
    o.append('D[%s] pos_band: %r'
             % (dc, wd['pos_band_ge_0.3'],))
    o.append('D[%s] neg_band: %r'
             % (dc, wd['neg_band_le_-0.3'],))
    o.append('D[%s] W_mean: %r'
             % (dc, wd['W_mean'],))
    o.append('D[%s] W_ans0: %r'
             % (dc, wd['W_ans0'],))
io.open(VF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('vals extracted')
