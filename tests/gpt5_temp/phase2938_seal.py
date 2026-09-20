# -*- coding: utf-8 -*-
"""Phase 2938 seal: forensic probe + SHA registration."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2938', 'subspace_angles')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2938_seal_report.txt')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2937 = os.path.join(BASE, 'phase2937', 'scale_collapse',
                        'scale_collapse.npz')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


out = []
rj = json.load(open(os.path.join(OUT, 'result.json'),
                    encoding='utf-8'))
z = np.load(os.path.join(OUT, 'subspace_angles.npz'),
            allow_pickle=True)
out.append('verdict: %s' % rj['final_verdict'])
out.append('anchors ok: %s | a1 %s a3 %s a4 %s a7 %s'
           % (rj['anchors']['ok'], rj['anchors']['a1_diff'],
              rj['anchors']['a3_diff'],
              rj['anchors']['a4_diff'],
              rj['anchors']['a7_diff']))

# 1. P1 alignment gradient table
out.append('')
out.append('P1 median alignment gradient (func | null0 | '
           'rho_null0 | rho_median_null):')
for k in ['1', '4', '8', '16', '36', 'dir35']:
    ma = rj['P1']['median_align'][k]
    rp = rj['P1']['rho_per_null'][k]
    rm = rj['P1']['rho_median'][k]
    out.append('  k=%-5s %.4f | %.4f | %.4f | %.4f'
               % (k, ma['func'], ma['null0'],
                  rp['null0'], rm))

# 2. SVD singular value profile
sv = z['sing_vals'].astype(np.float64)
out.append('')
out.append('SVD singular values (36 dirs stack):')
out.append('  %s' % [round(float(v), 3) for v in sv])
out.append('  top-8 energy frac %.4f | top-16 %.4f'
           % (float((sv[:8] ** 2).sum()
                    / (sv ** 2).sum()),
              float((sv[:16] ** 2).sum()
                    / (sv ** 2).sum())))

# 3. per-word alpha_8 spread: func vs null0
conds = [str(s) for s in z['cond_names']]
al = z['align_k'].astype(np.float64)
ifu = conds.index('func')
in0 = conds.index('null0')
a8f = al[ifu, list([1, 4, 8, 16, 36]).index(8)]
a8n = al[in0, list([1, 4, 8, 16, 36]).index(8)]
out.append('')
out.append('alpha_8 per-word: func min %.4f p25 %.4f max %.4f'
           % (float(a8f.min()), float(np.percentile(a8f, 25)),
              float(a8f.max())))
out.append('alpha_8 per-word: null0 min %.4f p25 %.4f max %.4f'
           % (float(a8n.min()), float(np.percentile(a8n, 25)),
              float(a8n.max())))
out.append('words with alpha_8 null0 < 0.15: %d/57'
           % int((a8n < 0.15).sum()))

# 4. k=1 (SVD PC1) vs dir35 dissociation per null set
out.append('')
out.append('k=1 (SVD PC1) vs dir35 rho per null set:')
for cn in ['same', 'null0', 'null1', 'null2', 'null3']:
    r1 = rj['P1']['rho_per_null']['1'][cn]
    rd = rj['P1']['rho_per_null']['dir35'][cn]
    out.append('  %s: PC1 %.4f vs dir35 %.4f (delta %+.4f)'
               % (cn, r1, rd, r1 - rd))

# 5. P3 layer profile summary: where does dir collapse
# while alpha8 holds
r8 = z['ratio_alpha8_layers'].astype(np.float64)
rd = z['ratio_dir_layers'].astype(np.float64)
nmd = np.median(r8, axis=0)
mdd = np.median(rd, axis=0)
out.append('')
out.append('per-layer median over 4 null sets: alpha8 vs dir')
out.append('  L1  %.3f vs %.3f' % (nmd[1], mdd[1]))
out.append('  L4  %.3f vs %.3f' % (nmd[4], mdd[4]))
out.append('  L8  %.3f vs %.3f' % (nmd[8], mdd[8]))
out.append('  L12 %.3f vs %.3f' % (nmd[12], mdd[12]))
out.append('  L20 %.3f vs %.3f' % (nmd[20], mdd[20]))
out.append('  L28 %.3f vs %.3f' % (nmd[28], mdd[28]))
out.append('  L35 %.3f vs %.3f' % (nmd[35], mdd[35]))
worst = int(np.argmin(mdd))
out.append('  dir ratio min at L%d (%.3f) while alpha8 '
           'there %.3f' % (worst, mdd[worst], nmd[worst]))

# 6. SHA registration
out.append('')
out.append('SHA256-8 registration:')
for f in ['execution.json', 'result.json',
          'subspace_angles.npz']:
    out.append('  %s %s' % (f, sha8(os.path.join(OUT, f))))
out.append('  script %s'
           % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                  r'\phase2938_subspace_angles.py'))
out.append('  source 2927 npz %s' % sha8(SRC_2927))
out.append('  source 2937 npz %s' % sha8(SRC_2937))

open(REP, 'w', encoding='utf-8').write(
    chr(10).join(out) + chr(10))
print('seal done')
