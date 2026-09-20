# -*- coding: utf-8 -*-
"""Phase 2930 seal probe: forensics + SHA registration.

S1  survivor-core rho robustness quantification (7 cells,
    |delta rho| mirror vs 2929, sign preservation)
S2  E_mirror layer/head structure; L+/en+ class membership
    in Ewd27 vs E_mirror
S3  layer profile of r-level nonlinearity (lin_r) and its
    correlation with mir_err
S4  lost-survivor forensics: (8,2) and (20,8) p_maxT mirror
    vs 2927 values; boundary band check
S5  L0 degenerate-tie replication in mirror caliber
S6  SHA256-8 registration
Report: tests/gpt5_temp/phase2930_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2930', 'direction_flip_control')
SRC2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                       'probe_relativity.npz')
SRC2929 = os.path.join(BASE, 'phase2929',
                       'response_structure_atlas',
                       'response_structure_atlas.npz')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2930_seal_report.txt')
NH, NL = 32, 36
FDR_Q = 0.05
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
L_POS = [(24, 23), (25, 3), (26, 6), (27, 24)]
EN_POS = [(7, 19), (8, 2), (22, 12)]


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines = []
z = np.load(os.path.join(OUT, 'direction_flip_control.npz'),
            allow_pickle=True)
rho_mir = z['rho_mirror_grid']
pmir = z['p_maxT_mirror'].astype(np.float64)
skel_mir = z['skel_mask_mirror'].astype(bool)
lin_r = z['lin_r_profile']
mir_err = z['mir_err_grid']
z27 = np.load(SRC2927, allow_pickle=True)
p_mir27 = z27['p_maxT_word'].astype(np.float64)
z29 = np.load(SRC2929, allow_pickle=True)
rho_29 = z29['rho_grid'].astype(np.float64)

# S1 survivor rho robustness
lines.append('S1 survivor-core rho robustness:')
for (h, li) in SURV:
    lines.append('    (%d,%2d) rho29 %.4f -> rho_mir %.4f '
                 '(delta %+.4f) | p_maxT 2927 %.4f -> mir %.4f'
                 % (h, li, rho_29[h, li], rho_mir[h, li],
                    rho_mir[h, li] - rho_29[h, li],
                    p_mir27[h, li], pmir[h, li]))
deltas = [abs(rho_mir[h, li] - rho_29[h, li])
          for (h, li) in SURV]
lines.append('    |delta| median %.4f max %.4f | all-7 rho>0.7 '
             'in mirror: %s'
             % (float(np.median(deltas)), float(max(deltas)),
                all(rho_mir[h, li] > 0.7 for (h, li) in SURV)))

# S2 class membership across calibers
lines.append('S2 L+ class (2928 l_pos):')
for e in L_POS:
    lines.append('    %s in Ewd27=%s in E_mirror=%s'
                 % (e, bool(p_mir27[e[0], e[1]] <= FDR_Q),
                    bool(pmir[e[0], e[1]] <= FDR_Q)))
lines.append('S2 en+ class (2918 en_pos):')
for e in EN_POS:
    lines.append('    %s in Ewd27=%s in E_mirror=%s'
                 % (e, bool(p_mir27[e[0], e[1]] <= FDR_Q),
                    bool(pmir[e[0], e[1]] <= FDR_Q)))
Emir = [(h, li) for h in range(NH) for li in range(NL)
        if pmir[h, li] <= FDR_Q]
lay = {}
hd = {}
for (h, li) in Emir:
    lay[li] = lay.get(li, 0) + 1
    hd[h] = hd.get(h, 0) + 1
lines.append('S2 E_mirror layers: %s'
             % sorted(lay.items()))
lines.append('S2 E_mirror heads(top): %s'
             % sorted(hd.items(), key=lambda t: -t[1])[:8])

# S3 nonlinearity layer profile + coupling
lines.append('S3 lin_r per layer: %s'
             % [(li, round(float(lin_r[li]), 3))
                for li in range(NL)])
c = np.corrcoef(lin_r,
                np.median(mir_err, axis=0))[0, 1]
lines.append('S3 corr(lin_r_layer, median mir_err_layer) '
             '= %.4f' % float(c))

# S4 lost-survivor boundary forensics
lines.append('S4 lost survivors p_maxT (2927 -> mirror):')
for e in ((8, 2), (20, 8)):
    lines.append('    %s %.4f -> %.4f (boundary band '
                 '0.05-0.10: %s/%s)'
                 % (e, p_mir27[e[0], e[1]], pmir[e[0], e[1]],
                    bool(0.05 < p_mir27[e[0], e[1]] <= 0.10),
                    bool(0.05 < pmir[e[0], e[1]] <= 0.10)))

# S5 L0 replication
lines.append('S5 L0 mirror: all-32 in skeleton %s | rho max '
             '%.6f (degenerate-tie artifact replicated)'
             % (bool(skel_mir[:, 0].all()),
                float(rho_mir[:, 0].max())))

# S6 SHA
sha = {'execution.json': sha8(os.path.join(OUT,
                                           'execution.json')),
       'result.json': sha8(os.path.join(OUT, 'result.json')),
       'direction_flip_control.npz':
           sha8(os.path.join(OUT,
                             'direction_flip_control.npz')),
       'src2927': sha8(SRC2927),
       'src2929': sha8(SRC2929),
       'script_phase2930': sha8(
           r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
           r'\phase2930_direction_flip_control.py')}
lines.append('S6 SHA256-8: %s' % json.dumps(sha, indent=1))

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('seal OK')
