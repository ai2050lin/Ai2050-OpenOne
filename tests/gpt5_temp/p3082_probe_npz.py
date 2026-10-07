# -*- coding: utf-8 -*-
"""Probe: list keys/shapes/scalars of 4B npz
(3076, 3079, 3080) for 3082 design."""
import io
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = (ROOT + r'\tests\gpt5_temp'
       r'\p3082_probe_npz.txt')
o = []

for tag, rel in (
        ('3076_4B',
         r'phase3076\omega_p73_cross_prompt_'
         r'family\omega_p73_cross_prompt_family'
         r'.npz'),
        ('3079_4B',
         r'phase3079\omega_p76_migration_lock'
         r'\omega_p76_migration_lock.npz'),
        ('3080_4B',
         r'phase3080\omega_p77_ab_anatomy'
         r'\omega_p77_ab_anatomy.npz')):
    z = np.load(BASE + '\\' + rel,
                allow_pickle=True)
    o.append('== %s (%d keys) ==' % (tag,
                                     len(z.files)))
    for k in sorted(z.files):
        a = z[k]
        if a.ndim == 0:
            o.append('  %s () = %s'
                     % (k, a.item()))
        else:
            o.append('  %s %s %s'
                     % (k, a.shape, a.dtype))
io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
