# -*- coding: utf-8 -*-
"""Inspect 3106 npz metadata dtypes and value distributions."""
import numpy as np

p = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3106'
     r'\omega_p104_composition_dose_depth'
     r'\capture.npz')
z = np.load(p, allow_pickle=False)
lines = []
for k in ('split', 'cond', 'truth', 'crit_rel',
          'query_rel', 'tag', 'family', 'unit'):
    try:
        a = z[k]
        vals, cnts = np.unique(a, return_counts=True)
        lines.append('%s: dtype=%s kind=%s shape=%s'
                     % (k, a.dtype, a.dtype.kind,
                        a.shape))
        for v, c in zip(vals[:12], cnts[:12]):
            lines.append('   %r -> %d' % (v, c))
    except KeyError:
        lines.append('%s: MISSING' % k)
p5 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
      r'\rdc_query_construction_20260913\phase3105'
      r'\omega_p103_incontext_truth_consistency'
      r'\capture.npz')
z5 = np.load(p5, allow_pickle=False)
lines.append('--- 3105 ---')
for k in ('split', 'tag'):
    a = z5[k]
    vals, cnts = np.unique(a, return_counts=True)
    lines.append('%s: dtype=%s kind=%s' % (k, a.dtype,
                                           a.dtype.kind))
    for v, c in zip(vals[:12], cnts[:12]):
        lines.append('   %r -> %d' % (v, c))
    # test comparison semantics
    lines.append("   (a=='train').sum()=%d "
                 "(a==b'train').sum()=%s"
                 % ((a == 'train').sum(),
                    (a == b'train').sum()
                    if a.dtype.kind == 'S' else 'n/a'))
out = open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
           r'\p3107_meta_out.txt', 'w', encoding='utf-8')
out.write('\n'.join(lines) + '\n')
out.close()
print('done')
