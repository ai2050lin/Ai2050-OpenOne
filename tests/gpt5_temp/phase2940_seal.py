# -*- coding: utf-8 -*-
"""Phase 2940 seal: forensics for v3_decode.
P3 class-separation detail (per-word c3, top words),
P1 layer profile detail (top-10 |w|, L15-18 mass share,
sign structure), P2 per-null class split, d3 top words,
SHA registration chain. Zero forward.
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
OUT = os.path.join(BASE, 'phase2940', 'v3_decode')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2940_seal_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


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
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((ra * rb).sum() / den)


out = []
z39 = np.load(SRC_2939, allow_pickle=True)
coords = z39['coords'].astype(np.float64)
conds = [str(s) for s in z39['cond_names']]
words = z39['words']
ifu = conds.index('func')
nulls = [cn for cn in conds if cn.startswith('null')]
z87 = np.load(SRC_2887, allow_pickle=True)
lab_lang = np.asarray(z87['labels_lang']).astype(int)
z27 = np.load(SRC_2927, allow_pickle=True)
dirs_word = z27['dirs_word'].astype(np.float64)
rj = json.load(open(os.path.join(OUT, 'result.json'),
                    encoding='utf-8'))
out.append('verdict: %s' % rj['final_verdict'])
out.append('anchors: a1 %.2e a2 %.2e a4 %.2e a5 %.2e ok=%s'
           % (rj['anchors']['a1_diff'],
              rj['anchors']['a2_diff'],
              rj['anchors']['a4_diff'],
              rj['anchors']['a5_diff'],
              rj['anchors']['ok']))
out.append('runtime: %.1fs' % rj['runtime_s'])

# ---------- P1 layer profile detail ----------
Usvd, s_loc, Vt = np.linalg.svd(dirs_word,
                                full_matrices=False)
w = s_loc[2] * Usvd[:, 2]
aw = np.abs(w)
order = np.argsort(-aw)
out.append('')
out.append('P1 layer profile of v3 (w_li = s3*U[li,2]):')
out.append('  top-10: %s'
           % [(int(li), round(float(w[li]), 3))
              for li in order[:10]])
mid = aw[15:19].sum() / aw.sum()
out.append('  L15-18 mass share: %.3f' % mid)
deep = aw[20:].sum() / aw.sum()
out.append('  L20-35 mass share: %.3f' % deep)
neg = [int(li) for li in range(36) if w[li] < 0]
out.append('  negative-weight layers: %s' % neg)

# ---------- P3 per-word c3 detail ----------
c3 = coords[ifu][:, 2]
out.append('')
out.append('P3 c3(func) class separation forensics:')
m0 = lab_lang == 0
m1 = lab_lang == 1
out.append('  median lab0 %.3f | median lab1 %.3f'
           % (float(np.median(c3[m0])),
              float(np.median(c3[m1]))))
ix = np.argsort(c3)
out.append('  lowest-5 c3 words: %s'
           % [(str(words[i][2]), round(float(c3[i]), 2),
               int(lab_lang[i])) for i in ix[:5]])
out.append('  highest-5 c3 words: %s'
           % [(str(words[i][2]), round(float(c3[i]), 2),
               int(lab_lang[i])) for i in ix[-5:]])
# per-language composition of extreme deciles
lo = ix[:14]
hi = ix[-14:]
out.append('  low decile lab0 count: %d/14 | high '
           'decile lab0 count: %d/14'
           % (int(lab_lang[lo].sum()),
              int(lab_lang[hi].sum())))
# c3 vs lab within-language (en only) to rule out
# language confound
langs = np.array([str(wd[0]) for wd in words])
en = langs == 'en'
if en.sum() > 4:
    rho_en = spearman(c3[en], lab_lang[en])
    out.append('  rho(c3, lab) within en only: %.4f '
               '(n=%d)' % (rho_en, int(en.sum())))

# ---------- P2 per-null class split ----------
out.append('')
out.append('P2 per-null class split of displacement d:')
for cn in nulls:
    i = conds.index(cn)
    d = coords[i][:, 2] - coords[ifu][:, 2]
    out.append('  %s: med lab0 %+.3f lab1 %+.3f '
               'diff %+.3f'
               % (cn, float(np.median(d[m0])),
                  float(np.median(d[m1])),
                  float(np.median(d[m0])
                        - np.median(d[m1]))))
d3 = rj and np.load(os.path.join(OUT, 'v3_decode.npz'),
                    allow_pickle=True)['d3']
order_d = np.argsort(-np.abs(d3))
out.append('  top-5 |d3| words: %s'
           % [(str(words[i][2]), round(float(d3[i]), 2))
              for i in order_d[:5]])

# ---------- same-condition control ----------
out.append('')
i_s = conds.index('same')
d_same = coords[i_s][:, 2] - coords[ifu][:, 2]
out.append('same-ctx control: med lab0 %+.3f lab1 %+.3f '
           'diff %+.3f'
           % (float(np.median(d_same[m0])),
              float(np.median(d_same[m1])),
              float(np.median(d_same[m0])
                    - np.median(d_same[m1]))))

# ---------- SHA chain ----------
out.append('')
out.append('SHA256-8 chain:')
out.append('  script 2940: %s'
           % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                  r'\phase2940_v3_decode.py'))
for nm, p in [('execution', os.path.join(OUT,
                                         'execution.json')),
              ('result', os.path.join(OUT, 'result.json')),
              ('npz', os.path.join(OUT, 'v3_decode.npz'))]:
    out.append('  %s: %s' % (nm, sha8(p)))
out.append('  src 2887: %s | 2927: %s | 2937: %s | '
           '2939: %s'
           % (sha8(SRC_2887), sha8(SRC_2927),
              sha8(os.path.join(
                  BASE, 'phase2937', 'scale_collapse',
                  'scale_collapse.npz')),
              sha8(SRC_2939)))
with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('OK seal 2940', flush=True)
