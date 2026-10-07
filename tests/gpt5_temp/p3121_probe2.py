# -*- coding: utf-8 -*-
"""p3121 probe2:
(a) Part D self-consistency: pooled syntax vs
    per-t syntax from the same arrays.
(b) Fixed fork stats: per-direction first-token
    rates for clean/L30/L32/joint.
(c) Part A D_c(t) profiles per condition.
(d) L26 direction split if 3118 npz has per-dir
    ablation gens."""
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
        r'\p3121_probe2_out.txt')
o = []

z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
z20 = np.load(os.path.join(D20, 'p118_readout.npz'),
              allow_pickle=False)
z21 = np.load(os.path.join(D21, 'p119_readout.npz'),
              allow_pickle=False)
mP = z18['gt_cleanseq_clean__P'].astype(np.float64)
mA = z18['gt_cleanseq_clean__A1'].astype(np.float64)
annP = z20['annP_c']
annA = z20['annA_c']

# (a) Part D self-consistency
o.append('== (a) syntax pooled vs per-t ==')
for dcode, m, ann in (('P', mP, annP),
                      ('A1', mA, annA)):
    dm = (m[:, 1:] - m[:, :-1])[:, 1:]
    a = ann[:, 1:]
    sel = (a == 1)
    pooled = float(dm[sel].mean())
    n = int(sel.sum())
    per_t = []
    for t in range(dm.shape[1]):
        s = (a[:, t] == 1)
        if s.any():
            per_t.append(round(float(dm[s].mean()),
                               3))
        else:
            per_t.append(None)
    o.append('%s pooled=%.4f n=%d per_t=%s'
             % (dcode, pooled, n, per_t))
    # also check OTHER class alignment as sanity
    sel0 = (a == 0)
    o.append('%s other pooled=%.4f n=%d'
             % (dcode, float(dm[sel0].mean()),
                int(sel0.sum())))

# (b) fixed fork stats per direction
o.append('== (b) fork per-direction ==')


def stats(seq2d, yf, nf):
    fy = float(np.mean([int(t) in yf
                        for t in seq2d[:, 0]]))
    fn = float(np.mean([int(t) in nf
                        for t in seq2d[:, 0]]))
    yr = float(np.mean([
        any(int(t) in yf for t in row[:8])
        for row in seq2d]))
    nr = float(np.mean([
        any(int(t) in nf for t in row[:8])
        for row in seq2d]))
    return fy, fn, yr, nr


YF = set()
NF = set()
seal20 = json.load(io.open(
    os.path.join(D20, 'design_seal.json'),
    encoding='utf-8'))
from transformers import AutoTokenizer  # noqa: E402
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
tok = AutoTokenizer.from_pretrained(MDIR)
for ys in seal20['token_families']['yes']:
    YF |= set(tok(ys, add_special_tokens=False)
              ['input_ids'])
for ns in seal20['token_families']['no']:
    NF |= set(tok(ns, add_special_tokens=False)
              ['input_ids'])

srcs = {
    'clean_P': z18['gen_clean__P'],
    'clean_A1': z18['gen_clean__A1'],
    'L30_P': z20['gen_abl_L30__P'],
    'L30_A1': z20['gen_abl_L30__A1'],
    'L32_P': z20['gen_abl_L32__P'],
    'L32_A1': z20['gen_abl_L32__A1'],
    'joint_P': z21['gj_P'],
    'joint_A1': z21['gj_A1'],
}
for nm, arr in srcs.items():
    fy, fn, yr, nr = stats(arr, YF, NF)
    o.append('%s: first_yes=%.4f first_no=%.4f '
             'yes8=%.4f no8=%.4f n=%d'
             % (nm, fy, fn, yr, nr, arr.shape[0]))

# (c) Part A D_c(t) profiles (P side)
o.append('== (c) Part A D_c(t) profile (P, '
         'span-pairs mean) ==')
pa_dn = {c: z21['pa_%s_dn__P' % c]
         for c in ('c0', 'c1', 'c2', 'c3')}
sp_len = z21['span_info_len']
sp_dir = z21['span_info_dir']
idxP = [j for j in range(len(sp_len))
        if sp_dir[j] == 1 and sp_len[j] >= 2]
prof = {c: np.zeros(13) for c in pa_dn}
for c in pa_dn:
    prof[c] = (pa_dn[c][idxP].astype(np.float64)
               - pa_dn['c0'][idxP]
               .astype(np.float64)).mean(0)
    o.append('%s D(t)=%s'
             % (c, np.round(prof[c], 3).tolist()))
# span length distribution
o.append('span len dist P: %s'
         % np.bincount(sp_len[sp_dir == 1])
         .tolist())
o.append('span len dist A1: %s'
         % np.bincount(sp_len[sp_dir == 0])
         .tolist())

# (d) L26 direction split in 3118 npz?
o.append('== (d) 3118 npz keys with abl/gen ==')
for k in sorted(z18.files):
    if 'abl' in k or 'gen' in k:
        o.append('  %s : %s' % (k, z18[k].shape))

io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
