# -*- coding: utf-8 -*-
"""Phase 2928 seal: SHA registry + null-overlap source + (27,24) forensics."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2928', 'survivor_core_anatomy')
SRC2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                       'probe_relativity.npz')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2928_seal_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines = []
z28 = np.load(os.path.join(OUT, 'survivor_core_anatomy.npz'),
              allow_pickle=True)
ov = z28['overlap_null']
lines.append('overlap null: median %.1f mean %.2f max %.1f '
             'min %.1f' % (np.median(ov), ov.mean(), ov.max(),
                           ov.min()))

# why is chance overlap ~7? set-size distribution drives it.
# Re-run 5 repeats quickly, recording set sizes (reuse main
# machinery inline).
z = np.load(SRC2927, allow_pickle=True)
B86 = z['B86'].astype(np.float64)
Bwd = z['B_word'].astype(np.float64)
lab = np.asarray(z['labels_lang']).astype(int)
pm86 = z['p_maxT86'].astype(np.float64)
pmwd = z['p_maxT_word'].astype(np.float64)
NH, NL, NW, FDR_Q = 32, 36, 57, 0.05

sizes = []
hot = np.zeros(NL)
for seed in range(2902, 2907):
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2928_survivor_core_anatomy import (
        gram_stack, maxT_events_fast)
    for tag, B in (('86', B86), ('wd', Bwd)):
        Gs = gram_stack(B)
        _, sig = maxT_events_fast(Gs, lab, seed)
        cells = [(int(i) // NL, int(i) % NL) for i in sig]
        sizes.append((seed, tag, len(sig)))
        for (h, li) in cells:
            hot[li] += 1
lines.append('null set sizes (5 repeats x 2 calibers): %s'
             % sizes)
top_layers = np.argsort(hot)[::-1][:6]
lines.append('null event hot layers (aggregated over 10 null '
             'sets): %s'
             % [(int(l), int(hot[l])) for l in top_layers])

# (27,24) forensics: dual-top but lost
sm86 = z['sign_M86'].astype(np.float64)
smwd = z['sign_M_word'].astype(np.float64)
lines.append('(27,24): margin86=%.5f (p_maxT86=%.4f) '
             'margin_wd=%.5f (p_maxTwd=%.4f)'
             % (sm86[27, 24], pm86[27, 24], smwd[27, 24],
                pmwd[27, 24]))
lines.append('(27,24) layer-24 pct: 86 %.3f word %.3f'
             % (float(np.mean(sm86[:, 24] <= sm86[27, 24])),
                float(np.mean(smwd[:, 24] <= smwd[27, 24]))))

# survivor vs lost: en/L word-polarity class (2918 classes)
surv = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
lost = [(1, 4), (4, 1), (4, 19), (4, 22), (6, 19), (13, 22),
        (15, 13), (17, 28), (21, 16), (22, 12), (24, 9),
        (24, 23), (25, 3), (26, 5), (26, 6), (27, 24), (28, 1)]
en_pos = {(8, 2), (22, 12), (7, 19)}
l_pos = {(26, 6), (25, 3), (24, 23), (27, 24)}
lines.append('2918 en+ class survival: %s'
             % [(e, e in surv) for e in sorted(en_pos)])
lines.append('2918 L+ class survival: %s'
             % [(e, e in surv) for e in sorted(l_pos)])

for name in ('execution.json', 'result.json',
             'survivor_core_anatomy.npz'):
    p = os.path.join(OUT, name)
    lines.append('%s sha256_8 %s' % (name, sha8(p)))
lines.append('script sha256_8 %s'
             % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                    r'\phase2928_survivor_core_anatomy.py'))
lines.append('created=%s'
             % json.load(open(os.path.join(OUT, 'execution.json'),
                              encoding='utf-8'))['created'])

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK seal report written')
