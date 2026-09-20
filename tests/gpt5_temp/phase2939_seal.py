# -*- coding: utf-8 -*-
"""Phase 2939 seal: forensic probe + SHA registration."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2939', 'rotation_target')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2939_seal_report.txt')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2938 = os.path.join(BASE, 'phase2938', 'subspace_angles',
                        'subspace_angles.npz')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


out = []
rj = json.load(open(os.path.join(OUT, 'result.json'),
                    encoding='utf-8'))
z = np.load(os.path.join(OUT, 'rotation_target.npz'),
            allow_pickle=True)
out.append('verdict: %s' % rj['final_verdict'])
out.append('anchors ok: %s | a1 %s a3 %s a4 %s a5 %s'
           % (rj['anchors']['ok'], rj['anchors']['a1_diff'],
              rj['anchors']['a3_diff'],
              rj['anchors']['a4_diff'],
              rj['anchors']['a5_diff']))

# 1. energy flow table: share func vs null per basis
out.append('')
out.append('P2 energy share per basis (func | null0 | '
           'delta_e_med):')
sf = rj['P2']['share_func']
sn = rj['P2']['share_null0']
de = rj['P2']['delta_e_med']
for k in range(8):
    tag = ' <-- k*' if k == rj['P2']['k_star'] - 1 else ''
    out.append('  v%d: %.4f | %.4f | %+.4f%s'
               % (k + 1, sf[k], sn[k], de[k], tag))
out.append('  sum func %.4f sum null0 %.4f'
           % (sum(sf), sum(sn)))

# 2. v1 vs v3 basis semantics: correlate coords with
#    dir35 projection (are v1/v3 aligned to dir35?)
coords = z['coords'].astype(np.float64)
conds = [str(s) for s in z['cond_names']]
proj = z['proj_dir35'].astype(np.float64)
ifu = conds.index('func')
out.append('')
out.append('basis-diag35 coupling (func coords vs dir35 '
           'proj, Spearman over 57 words):')


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(np.asarray(a, np.float64))
    rb = rankdata(np.asarray(b, np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((ra * rb).sum() / den)


for k in range(8):
    out.append('  v%d vs dir35: %.4f'
               % (k + 1, spearman(coords[ifu][:, k],
                                  proj[ifu])))

# 3. word-level displacement norm: ||c_null - c_func|| vs
#    ||c_func|| per word (relative displacement size)
in0 = conds.index('null0')
disp = coords[in0] - coords[ifu]
dn = np.linalg.norm(disp, axis=1)
cf = np.linalg.norm(coords[ifu], axis=1)
out.append('')
out.append('word displacement in 8-dim coords: median '
           '||dc|| %.3f vs median ||c_func|| %.3f (ratio '
           '%.3f)' % (float(np.median(dn)),
                      float(np.median(cf)),
                      float(np.median(dn)
                            / max(float(np.median(cf)),
                                  1e-30))))
top_w = int(np.argmax(dn))
out.append('  max displacement word idx %d (%s): %.3f'
           % (top_w, str(list(z['words'][top_w])),
              float(dn[top_w])))

# 4. v6 outlier check: the only rho < 0.5 basis
out.append('')
out.append('v6 (only rho_med < 0.5, %.4f): share func %.4f '
           '-> null0 %.4f | delta_c overall %s'
           % (rj['P1']['rho_med']['v6'], sf[5], sn[5],
              rj['P3']['delta_c_overall']['null0'][5]))

# 5. cross-null stability of delta_c direction
d0 = np.array(rj['P3']['delta_c_overall']['null0'])
cs = []
for cn in ['null1', 'null2', 'null3']:
    d = np.array(rj['P3']['delta_c_overall'][cn])
    cs.append(float(abs(np.dot(d0, d)
                        / max(np.linalg.norm(d0)
                              * np.linalg.norm(d), 1e-30))))
out.append('delta_c cross-null cos(null0 vs null1/2/3): '
           '%s' % [round(v, 4) for v in cs])

# 6. SHA registration
out.append('')
out.append('SHA256-8 registration:')
for f in ['execution.json', 'result.json',
          'rotation_target.npz']:
    out.append('  %s %s' % (f, sha8(os.path.join(OUT, f))))
out.append('  script %s'
           % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                  r'\phase2939_rotation_target.py'))
out.append('  source 2927 npz %s' % sha8(SRC_2927))
out.append('  source 2938 npz %s' % sha8(SRC_2938))

open(REP, 'w', encoding='utf-8').write(
    chr(10).join(out) + chr(10))
print('seal done')
