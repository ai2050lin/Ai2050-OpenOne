# -*- coding: utf-8 -*-
import io

import numpy as np

o = []
pool = np.array([0.67, 0.75, 2.25, 2.45, 4.53, 5.46, 7.44,
                 7.65, 8.25, 9.19, 9.67, 10.25, 10.65, 10.82,
                 11.73, 12.24, 12.26, 12.3, 13.2, 13.62, 14.68,
                 14.81, 15.09, 15.17, 15.3, 15.51, 15.67, 15.82,
                 16.05, 16.64, 16.67, 18.22, 18.65, 18.66, 20.1,
                 21.41, 21.77])
lab = np.array([1] * 15 + [0] * 22)
rng = np.random.default_rng(7)
ds = []
for k in range(20):
    pm = rng.permutation(pool.size)
    lb = lab[pm]
    d = np.median(pool[pm][lb == 1]) \
        - np.median(pool[pm][lb == 0])
    ds.append(round(float(d), 4))
o.append('sanity ds=%s' % ds)
o.append('unique=%d' % len(set(ds)))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_sanity2995.txt',
        'w', encoding='utf-8').write('\n'.join(o))
print('done')
