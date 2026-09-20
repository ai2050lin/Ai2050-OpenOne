# -*- coding: utf-8 -*-
"""Phase 2932 seal: functional-load forensics + SHA registration."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = os.path.join(BASE, 'phase2932',
                    'skeleton_functional_ablation')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2932_seal_report.txt')

z = np.load(os.path.join(OUTD,
                         'skeleton_functional_ablation.npz'),
            allow_pickle=True)
rj = json.load(open(os.path.join(OUTD, 'result.json'),
                    encoding='utf-8'))
cells = [tuple(int(v) for v in c)
         for c in z['cells_all']]
skel = [tuple(int(v) for v in c)
        for c in z['cells_skel']]
ci = z['ci_rel'].astype(np.float64)
ci_map = dict(zip(cells, ci))
sk_set = set(skel)

out = []
out.append('verdict: %s' % rj['final_verdict'])
out.append('P1: %s' % json.dumps(rj['P1']))
out.append('P2: %s' % json.dumps(rj['P2']))
out.append('P3: %s' % json.dumps(rj['P3']))

# 1. top skeleton cells by CI vs top bg cells
sk_sorted = sorted(skel, key=lambda c: -ci_map[c])
bg = [c for c in cells if c not in sk_set]
bg_sorted = sorted(bg, key=lambda c: -ci_map[c])
out.append('')
out.append('top-10 skeleton by CI_rel: %s'
           % [(c, round(ci_map[c], 5)) for c in sk_sorted[:10]])
out.append('top-10 bg by CI_rel: %s'
           % [(c, round(ci_map[c], 5)) for c in bg_sorted[:10]])
out.append('bottom-5 skeleton: %s'
           % [(c, round(ci_map[c], 5))
              for c in sk_sorted[-5:]])

# 2. skeleton CI rank within layer (is skeleton top of layer?)
out.append('')
out.append('per-layer: n_skel / median_ci_skel / '
           'median_ci_bg / #skel_in_layer_top_nsk')
for li in range(1, 36):
    sk_l = [c for c in skel if c[1] == li]
    bg_l = [c for c in bg if c[1] == li]
    if not sk_l:
        continue
    ms = float(np.median([ci_map[c] for c in sk_l]))
    mb = float(np.median([ci_map[c] for c in bg_l])) \
        if bg_l else float('nan')
    all_l = sorted(sk_l + bg_l, key=lambda c: -ci_map[c])
    nsk = len(sk_l)
    n_top = sum(1 for c in all_l[:nsk] if c in sk_set)
    out.append('  L%02d nsk=%2d med_sk=%.5f med_bg=%.5f '
               'top_half_hit=%d/%d'
               % (li, nsk, ms, mb, n_top, nsk))

# 3. survivor vs skeleton peers
surv = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
surv_ci = [ci_map[c] for c in surv]
peer_ci = [ci_map[c] for c in skel if c not in surv]
out.append('')
out.append('survivor CI_rel median %.5f vs skeleton-peer '
           'median %.5f (survivor all above peer median: %s)'
           % (float(np.median(surv_ci)),
              float(np.median(peer_ci)),
              all(v > float(np.median(peer_ci))
                  for v in surv_ci)))

# 4. CI_logit cross-check
cl = z['ci_logit'].astype(np.float64)
cl_map = dict(zip(cells, cl))
out.append('CI_logit: skel median %.6f vs bg median %.6f'
           % (float(np.median([cl_map[c] for c in skel])),
              float(np.median([cl_map[c] for c in bg]))))

# 5. SHA registration
def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()[:8]

out.append('')
out.append('execution.json sha8: %s'
           % sha8(os.path.join(OUTD, 'execution.json')))
out.append('result.json sha8: %s'
           % sha8(os.path.join(OUTD, 'result.json')))
out.append('npz sha8: %s'
           % sha8(os.path.join(OUTD,
                'skeleton_functional_ablation.npz')))
out.append('script sha8: %s'
           % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                  r'\phase2932_skeleton_functional_ablation.py'))

open(REP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('seal done')
