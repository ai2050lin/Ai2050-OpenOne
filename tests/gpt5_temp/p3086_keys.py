# -*- coding: utf-8 -*-
"""p3086 key probe: list exact key names in the four source npz files."""
import numpy as np
import io

ROOT = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
S79 = ROOT + r'\phase3079\omega_p76_migration_lock\omega_p76_migration_lock.npz'
S81 = ROOT + r'\phase3081\omega_p78_ds7b_crossmodel\omega_p78_ds7b_crossmodel.npz'
S82 = ROOT + r'\phase3082\omega_p79_ds7b_negative_anatomy\omega_p79_ds7b_negative_anatomy.npz'
S85 = ROOT + r'\phase3085\omega_p82_l34_full_arbitration\omega_p82_l34_full_arbitration.npz'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3086_keys.txt'

with io.open(OUT, 'w', encoding='utf-8') as f:
    for tag, path in (('S79', S79), ('S81', S81),
                      ('S82', S82), ('S85', S85)):
        z = np.load(path, allow_pickle=False)
        keys = sorted(z.files)
        f.write('== %s (%d keys) ==\n' % (tag, len(keys)))
        for k in keys:
            v = z[k]
            if v.dtype.kind in 'US' or v.shape == ():
                f.write('  %s = %s\n' % (k, str(v)))
            else:
                f.write('  %s shape=%s dtype=%s\n'
                        % (k, v.shape, v.dtype))
