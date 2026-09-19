# -*- coding: utf-8 -*-
"""Phase 2904 data probe: inspect 2902/2903 npz keys/shapes/labels."""
import numpy as np
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P2902 = os.path.join(BASE, 'phase2902',
                     'glm4_channel_jacobian_asymmetry',
                     'glm4_channel_jacobian_asymmetry.npz')
P2903 = os.path.join(BASE, 'phase2903',
                     'qwen_channel_jacobian_decomposition',
                     'qwen_channel_jacobian_decomposition.npz')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_2904.txt'

lines = []
for tag, p in (('2902 glm4', P2902), ('2903 qwen', P2903)):
    lines.append('==== %s ====' % tag)
    lines.append('exists=%s' % os.path.exists(p))
    z = np.load(p, allow_pickle=True)
    for k in z.files:
        a = z[k]
        s = 'key=%-18s shape=%-14s dtype=%s' % (
            k, str(a.shape), a.dtype)
        if a.dtype == object and a.size <= 90:
            s += ' first3=%s' % (list(a[:3]),)
        elif a.ndim <= 1 and a.size <= 90:
            s += ' vals=%s' % (a.tolist()[:12],)
        lines.append(s)
    # label cross-check
    if 'labels_lang' in z.files and 'labels_concept' in z.files:
        ll = np.asarray(z['labels_lang']).astype(int)
        lc = np.asarray(z['labels_concept']).astype(int)
        lines.append('labels_lang: n=%d uniq=%s counts=%s'
                     % (len(ll), sorted(set(ll.tolist())),
                        np.bincount(ll).tolist()))
        lines.append('labels_concept: n=%d uniq=%s counts=%s'
                     % (len(lc), sorted(set(lc.tolist())),
                        np.bincount(lc).tolist()))
        if 'B_mlp' in z.files:
            B = z['B_mlp'].astype(np.float64)
            same = (ll[:, None] == ll[None, :]) & \
                (~np.eye(len(ll), dtype=bool))
            U = B / np.maximum(np.linalg.norm(
                B, axis=1, keepdims=True), 1e-30)
            Sm = U @ U.T
            m = Sm[same].mean() - Sm[(~np.eye(len(ll),
                                      dtype=bool)) & ~same].mean()
            lines.append('quick margin recompute (B_mlp, lang)='
                         '%.5f' % m)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('ok')
