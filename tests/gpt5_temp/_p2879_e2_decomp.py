# -*- coding: utf-8 -*-
"""Post-hoc descriptive analysis (zero forward): per-axis decomposition
of the 2879 E2 margin_absent verdict, from the frozen 2879 npz.

2879 E2 tested the GLOBAL same-axis cosine margin over all 3 syntax
axes (0.0744 < null p95 0.0889).  This decomposition asks WHICH axis
lacks the margin: for each axis separately, same-axis pairs within that
axis vs all cross-axis pairs, compared against axis-matched permutation
nulls (200 perms, SEED=2880 posthoc).  NOTE: post-hoc, no gate
decision - registered as descriptive evidence for 2880 interpretation.
"""
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC = os.path.join(BASE, 'phase2879', 'syntax_mlp_spectrum',
                   'syntax_mlp_spectrum.npz')
OUT = os.path.join(BASE, 'phase2879', 'e2_posthoc_decomp.txt')
SEED = 2880
N_NULL = 200

zv = np.load(SRC, allow_pickle=True)
B3 = zv['B3_spec'].astype(np.float64)
lab = zv['labels'].astype(int)
tl = [str(t) for t in zv['target_list']]
axes = sorted(set(lab.tolist()))
U = B3 / np.maximum(np.linalg.norm(B3, axis=1, keepdims=True), 1e-30)
S = U @ U.T
n = len(lab)

lines = []
lines.append('P2879 E2 post-hoc per-axis decomposition (descriptive)')


def margins_for(pl):
    out = {}
    for a in axes:
        rows = [i for i in range(n) if pl[i] == a]
        if len(rows) < 2:
            out[a] = (float('nan'), 0)
            continue
        same_vals = []
        for x in range(len(rows)):
            for y in range(x + 1, len(rows)):
                same_vals.append(S[rows[x], rows[y]])
        cross = []
        # cross pairs: one in axis a, one outside
        in_a = set(rows)
        for i in range(n):
            for j in range(n):
                if i < j and ((i in in_a) != (j in in_a)):
                    cross.append(S[i, j])
        out[a] = (float(np.mean(same_vals)) - float(np.mean(cross)),
                  len(same_vals))
    return out


obs = margins_for(lab)
rng = np.random.default_rng(SEED)
null = {a: [] for a in axes}
for _ in range(N_NULL):
    pl = rng.permutation(lab)
    m = margins_for(pl)
    for a in axes:
        null[a].append(m[a][0])

lines.append('')
for a in axes:
    words = [tl[i].split(':')[1] for i in range(n) if lab[i] == a]
    nul = np.array([x for x in null[a] if not np.isnan(x)], dtype=float)
    m_obs, n_pairs = obs[a]
    p95 = float(np.percentile(nul, 95)) if len(nul) else float('nan')
    exceeds = bool(m_obs > p95)
    lines.append('axis %-12s n_words=%d n_pairs=%d margin=%+.4f '
                 'null_p95=%+.4f null_mean=%+.4f exceeds=%s'
                 % (tl[[i for i in range(n) if lab[i] == a][0]].split(
                     ':')[0], len(words), n_pairs, m_obs, p95,
                    float(nul.mean()) if len(nul) else float('nan'),
                    exceeds))
    lines.append('  words: %s' % ' '.join(words))

# also pairwise-axis cos means (3x3) for structure
lines.append('')
lines.append('mean cos between axis-centroid spectra:')
cents = {}
for a in axes:
    rows = [i for i in range(n) if lab[i] == a]
    c = U[rows].mean(axis=0)
    cents[a] = c / max(float(np.linalg.norm(c)), 1e-30)
for a in axes:
    for b in axes:
        if a < b:
            lines.append('  %d vs %d: %+.4f'
                         % (a, b, float(cents[a] @ cents[b])))

with open(OUT, 'w', encoding='utf-8') as g:
    g.write('\n'.join(lines) + '\n')
print('written')
