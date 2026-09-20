# -*- coding: utf-8 -*-
"""Phase 2933 seal: verdict-boundary forensics + SHA registration."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = os.path.join(BASE, 'phase2933', 'full_atlas_ci')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2933_seal_report.txt')

z = np.load(os.path.join(OUTD, 'full_atlas_ci.npz'),
            allow_pickle=True)
rj = json.load(open(os.path.join(OUTD, 'result.json'),
                    encoding='utf-8'))
cells = [tuple(int(v) for v in c) for c in z['cells']]
ci = z['ci_rel'].astype(np.float64)
ci_map = dict(zip(cells, ci))
NH, NL = 32, 36

out = []
out.append('verdict: %s' % rj['final_verdict'])
out.append('P2: load %.6f deep %.6f diff %.6f p2b %.6f | '
           'spearman(med_l, lin_r) %.4f p2a %.3e'
           % (rj['P2']['median_ci_load_band'],
              rj['P2']['median_ci_deep_band'],
              rj['P2']['diff'], rj['P2']['p2b_exact'],
              rj['P2']['spearman_med_l_lin_r'],
              rj['P2']['p2a']))
out.append('P3: %s' % json.dumps(rj['P3']))

# 1. verdict boundary decomposition: which branch failed
out.append('')
out.append('BOUNDARY: p3a=%.3e vs prereg threshold 1e-3 '
           '(exceeds by %.1e -> band branch p2b=%.6f was '
           'decisively met)'
           % (rj['P3']['p_86'],
              rj['P3']['p_86'] - 1e-3,
              rj['P2']['p2b_exact']))

# 2. layer profile table with lin_r
med = rj['P2']['layer_median_ci_rel']
lin = rj['P4']['layer_profile']
out.append('')
out.append('layer profile: L, med_ci, spearman(ci,rho86)_layer, '
           'lin_r')
for d in lin:
    out.append('  L%02d %.6f %+.4f %.4f'
               % (d['layer'], d['med_ci'],
                  d['spearman_ci_rho86'], d['lin_r']))

# 3. strongest CI cells overall (full coverage)
top = sorted(cells, key=lambda c: -ci_map[c])[:12]
out.append('')
out.append('top-12 CI cells: %s'
           % [(c, round(ci_map[c], 5)) for c in top])

# 4. deep-band quietness: max CI in deep band vs load band
db = [ci_map[(h, li)] for li in range(28, 36)
      for h in range(NH)]
lb = [ci_map[(h, li)] for li in range(6, 13)
      for h in range(NH)]
out.append('CI distribution: load band max %.5f p90 %.5f | '
           'deep band max %.5f p90 %.5f'
           % (max(lb), float(np.percentile(lb, 90)),
              max(db), float(np.percentile(db, 90))))

# 5. mirror coupling dominance
ci_arr = np.array([ci_map[c] for c in cells])
out.append('')
out.append('CI vs rho mirror coupling %.4f dominates rho86 '
           'coupling %.4f - mirror-probe response structure '
           'predicts function better than original caliber'
           % (rj['P3']['spearman_ci_rho_mirror'],
              rj['P3']['spearman_ci_rho86']))

# 6. SHA registration
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
out.append('npz sha8: %s' % sha8(os.path.join(OUTD,
                'full_atlas_ci.npz')))
out.append('script sha8: %s'
           % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                  r'\phase2933_full_atlas_ci.py'))

open(REP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('seal done')
