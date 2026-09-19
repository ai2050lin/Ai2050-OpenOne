import numpy as np
p = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2846/fullhead_census/census_L29.npz'
z = np.load(p, allow_pickle=True)
out = []
for k in z.files:
    a = z[k]
    out.append('key=%s shape=%s dtype=%s' % (k, a.shape, a.dtype))
    if a.ndim <= 1 and a.size <= 40:
        out.append('  vals=%s' % np.array2string(a, precision=5, max_line_width=200))
    elif a.ndim == 2:
        out.append('  row0[:8]=%s' % np.array2string(a[0, :8], precision=5))
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2856_npz_probe.txt'
open(rep, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('probe done')
