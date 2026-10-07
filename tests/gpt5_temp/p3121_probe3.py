# -*- coding: utf-8 -*-
"""p3121 probe3:
(a) L26/L31/L33 direction-split first-token fork
    from 3118 npz (re-interpret 3120 'opposite to
    L26' claim).
(b) Part D per-t syntax recomputed CORRECTLY
    (element-wise column selection, not row
    selection) + gate re-verdict + pooled
    cross-check.
(c) Part A span-inside vs after-span decomposition
    of D_c."""
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D18 = os.path.join(RDIR, 'phase3118',
                   'omega_p116_autoregressive_margin_'
                   'trajectory')
D20 = os.path.join(RDIR, 'phase3120',
                   'omega_p118_content_attr_amplifier_'
                   'behavior_opshape')
D21 = os.path.join(RDIR, 'phase3121',
                   'omega_p119_repl_causality_erase_'
                   'polarity_recon')
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3121_probe3_out.txt')
o = []

z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
z20 = np.load(os.path.join(D20, 'p118_readout.npz'),
              allow_pickle=False)
z21 = np.load(os.path.join(D21, 'p119_readout.npz'),
              allow_pickle=False)

from transformers import AutoTokenizer  # noqa: E402
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
tok = AutoTokenizer.from_pretrained(MDIR)
seal20 = json.load(io.open(
    os.path.join(D20, 'design_seal.json'),
    encoding='utf-8'))
YF = set()
NF = set()
for ys in seal20['token_families']['yes']:
    YF |= set(tok(ys, add_special_tokens=False)
              ['input_ids'])
for ns in seal20['token_families']['no']:
    NF |= set(tok(ns, add_special_tokens=False)
              ['input_ids'])

# (a) L26/L31/L33 direction split
o.append('== (a) 3118 ablation direction split ==')
for L in ('L26', 'L31', 'L33'):
    for d in ('P', 'A1'):
        arr = z18['gen_abl_%s__%s' % (L, d)]
        fy = float(np.mean([int(t) in YF
                            for t in arr[:, 0]]))
        fn = float(np.mean([int(t) in NF
                            for t in arr[:, 0]]))
        yr = float(np.mean([
            any(int(t) in YF for t in row[:8])
            for row in arr]))
        o.append('%s_%s: first_yes=%.4f '
                 'first_no=%.4f yes8=%.4f'
                 % (L, d, fy, fn, yr))

# (b) Part D corrected per-t syntax
o.append('== (b) Part D corrected ==')
mP = z18['gt_cleanseq_clean__P'].astype(np.float64)
mA = z18['gt_cleanseq_clean__A1'].astype(np.float64)
d_ok = True
for dcode, m, ann in (('P', mP, z20['annP_c']),
                      ('A1', mA, z20['annA_c'])):
    dm = (m[:, 1:] - m[:, :-1])[:, 1:]
    a = ann[:, 1:]
    per_t = []
    all_neg = True
    n_zero = 0
    wn = 0
    wsum = 0.0
    for t in range(dm.shape[1]):
        s = (a[:, t] == 1)
        n = int(s.sum())
        if n == 0:
            per_t.append(None)
            n_zero += 1
            continue
        v = float(dm[:, t][s].mean())
        per_t.append(round(v, 4))
        wn += n
        wsum += v * n
        if v >= 0:
            all_neg = False
    pooled_chk = wsum / wn if wn else None
    o.append('%s per_t=%s n_zero=%d all_neg=%s '
             'pooled_check=%.4f'
             % (dcode, per_t, n_zero, all_neg,
                pooled_chk))
    if not all_neg:
        d_ok = False
o.append('D-POS corrected verdict: %s'
         % ('position_independent_confirmed'
            if d_ok else 'position_confounded'))

# (c) Part A inside vs after span (P side)
o.append('== (c) Part A inside/after span ==')
pa = {c: z21['pa_%s_dn__P' % c]
      for c in ('c0', 'c1', 'c2', 'c3')}
sp_len = z21['span_info_len']
sp_dir = z21['span_info_dir']
# need k1/k2: not in npz; recompute from ann
annP = z20['annP_c']
ins = {c: [] for c in pa}
aft = {c: [] for c in pa}
for j in range(672):
    if sp_dir[j] != 1 or sp_len[j] < 2:
        continue
    k1 = None
    best = None
    for k in range(1, 12):
        isf = int(annP[j, k]) in (3, 4)
        if isf and k1 is None:
            k1 = k
        if (not isf or k == 11) \
                and k1 is not None:
            k2 = k if (isf and k == 11) else k - 1
            if best is None or (k2 - k1) > \
                    (best[1] - best[0]):
                best = (k1, k2)
            k1 = None
    if best is None:
        continue
    k1, k2 = best
    for c in pa:
        diff = (pa[c][j].astype(np.float64)
                - pa['c0'][j].astype(np.float64))
        ins[c].append(float(diff[k1 + 1:k2 + 1]
                            .mean()))
        if k2 + 1 < 13:
            aft[c].append(float(diff[k2 + 1:]
                                .mean()))
for c in ('c1', 'c2', 'c3'):
    o.append('%s inside=%.4f after=%.4f '
             '(n=%d/%d)'
             % (c, float(np.mean(ins[c]))
                if ins[c] else float('nan'),
                float(np.mean(aft[c]))
                if aft[c] else float('nan'),
                len(ins[c]), len(aft[c])))

io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
