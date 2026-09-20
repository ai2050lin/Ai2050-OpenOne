# -*- coding: utf-8 -*-
"""Phase 2929 seal probe: skeleton forensics + SHA registration.

Read-only forensics on response_structure_atlas.npz:
  S1  2x2 skeleton x event contingency table
  S2  skeleton membership rate per event group
      (survivor/lost/new, rebuilt from 2927 npz thresholds)
  S3  top-20 cells by rho with skeleton/event flags
  S4  L0 forensics (all-32-head skeleton layer): rho stats
  S5  per-layer null p95 thresholds + skeleton counts
  S6  SHA256-8 registration of products + sources
Report: tests/gpt5_temp/phase2929_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2929', 'response_structure_atlas')
SRC2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                       'probe_relativity.npz')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2929_seal_report.txt')
NH, NL = 32, 36
FDR_Q = 0.05


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines = []
z = np.load(os.path.join(OUT, 'response_structure_atlas.npz'),
            allow_pickle=True)
rho = z['rho_grid']
skel = z['skel_mask'].astype(bool)
evf = z['event_flag'].astype(bool).reshape(NH, NL)

# S1 contingency
both = int((skel & evf).sum())
skel_only = int(skel.sum()) - both
ev_only = int(evf.sum()) - both
neither = int((~skel & ~evf).sum())
lines.append('S1 2x2: skel&ev=%d skel_only=%d ev_only=%d '
             'neither=%d' % (both, skel_only, ev_only, neither))
lines.append('    event->skeleton rate %.3f | background->skeleton '
             'rate %.3f' % (both / float(evf.sum()),
                            skel_only / float((~evf).sum())))

# S2 per-group skeleton membership (rebuild E17/Ewd from 2927)
z27 = np.load(SRC2927, allow_pickle=True)
pm86 = z27['p_maxT86'].astype(np.float64)
pmwd = z27['p_maxT_word'].astype(np.float64)
E17 = set((h, li) for h in range(NH) for li in range(NL)
          if pm86[h, li] <= FDR_Q)
Ewd = set((h, li) for h in range(NH) for li in range(NL)
          if pmwd[h, li] <= FDR_Q)
for name, cells in (('survivor', E17 & Ewd), ('lost', E17 - Ewd),
                    ('new', Ewd - E17)):
    k = sum(1 for (h, li) in cells if skel[h, li])
    lines.append('S2 %s: %d/%d in skeleton (%.3f)'
                 % (name, k, len(cells), k / float(len(cells))))

# S3 top-20 by rho
order = np.argsort(rho.ravel())[::-1][:20]
lines.append('S3 top-20 cells (h, l, rho, skel, event):')
for i in order:
    h, li = int(i) // NL, int(i) % NL
    lines.append('    (%d,%2d) rho=%.4f skel=%s ev=%s'
                 % (h, li, rho[h, li], bool(skel[h, li]),
                    bool(evf[h, li])))

# S4 L0 forensics
lines.append('S4 L0: rho min %.4f median %.4f max %.4f | all-32 '
             'in skeleton: %s'
             % (rho[:, 0].min(), np.median(rho[:, 0]),
                rho[:, 0].max(), bool(skel[:, 0].all())))
lines.append('    L0 non-skel rho values: %s'
             % sorted(round(float(v), 4)
                      for v in rho[:, 0][~skel[:, 0]]))

# S5 per-layer profile
z29 = np.load(os.path.join(OUT, 'response_structure_atlas.npz'),
              allow_pickle=True)
p95 = z29['null_p95']
prof = [(l, int(skel[:, l].sum()), round(float(p95[l]), 3))
        for l in range(NL)]
lines.append('S5 layer profile (l, n_skel_of_32, null_p95): %s'
             % prof)
lines.append('S5 peak layers by skeleton count: %s'
             % sorted(prof, key=lambda t: -t[1])[:6])

# S6 SHA registration
sha = {'execution.json': sha8(os.path.join(OUT, 'execution.json')),
       'result.json': sha8(os.path.join(OUT, 'result.json')),
       'response_structure_atlas.npz':
           sha8(os.path.join(OUT,
                             'response_structure_atlas.npz')),
       'src2927_probe_relativity.npz': sha8(SRC2927),
       'src2928_survivor_core_anatomy.npz':
           sha8(os.path.join(BASE, 'phase2928',
                             'survivor_core_anatomy',
                             'survivor_core_anatomy.npz')),
       'src2917_event_atlas.npz':
           sha8(os.path.join(BASE, 'phase2917', 'event_atlas',
                             'event_atlas.npz')),
       'script_phase2929': sha8(
           r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
           r'\phase2929_response_structure_atlas.py')}
lines.append('S6 SHA256-8: %s' % json.dumps(sha, indent=1))

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('seal OK')
