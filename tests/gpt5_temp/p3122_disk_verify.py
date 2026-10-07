# -*- coding: utf-8 -*-
"""Phase 3122 independent disk verify.
Re-derives from frozen inputs + npz:
  B1 s0 bit-exact + wrec anchors
  B2 find_span replication + E_cont/E_syn
     recompute from npz sb_* arrays
  B3 Part C FULL re-simulation (rng 3122)
     -> auc_sim + rank PIT + KS bit-check
  B4 ledger n/sha recompute
  B5 MEMO tail structure + key numbers
  B6 dual wlog + MEMORY.md chain
"""
import hashlib
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
OUTD = os.path.join(RDIR, 'phase3122',
                    'omega_p120_write_content_'
                    'readout_sentence_causal_'
                    'dist_recon')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
N_NEW = 12
checks = 0
fails = []


def ck(tag, cond):
    global checks
    checks += 1
    if not cond:
        fails.append(tag)


# ---------- A. result.json ----------
res = json.load(io.open(
    os.path.join(OUTD, 'result.json'),
    encoding='utf-8'))
V = ('mixed|write_polarity_diverged|'
     'replay_bit_exact|'
     'sentence_content_push_down|'
     'sentence_content_push_down|'
     'syntax_effect_present|'
     'syntax_effect_present|'
     'oscillation_persist|'
     'distribution_reconstruction_failed|'
     'pit_marginal')
ck('A.verdict', res['verdict'] == V)
ck('A.smoke', res['smoke'] is False)
ck('A.pairs', res['n_pairs'] == 672
   and res['np_a'] == 672)
ck('A.runtime', abs(res['runtime_s'] - 238.2)
   < 0.05)

# ---------- B. npz ----------
z = np.load(os.path.join(OUTD, 'p120_readout.npz'),
            allow_pickle=False)
z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
z20 = np.load(os.path.join(D20, 'p118_readout.npz'),
              allow_pickle=False)
ck('B.keys', all(k in z for k in (
    'wrec_dn_P', 'wrec_dn_A1', 'wrec_pd_P',
    'wrec_pd_A1', 'wrec_pf_P', 'wrec_pn_P',
    'sb_s0_dn_P', 'sb_s1_dn_P', 'sb_s2_dn_P',
    'sb_s3_dn_P', 'sb_s0_dn_A1', 'sb_s1_dn_A1',
    'sb_s2_dn_A1', 'sb_s3_dn_A1', 'auc_sim',
    'pit')))
ck('B.wrec_shape',
   z['wrec_dn_P'].shape == (672, 13)
   and z['wrec_pd_P'].shape == (36, 672, 13))
ck('B.sb_shape', z['sb_s0_dn_P'].shape == (672, 13))
# B1: s0 replay bit-exact + wrec vs 3118
for dc in ('P', 'A1'):
    ref = z18['gt_cleanseq_clean__%s' % dc]
    w = z['wrec_dn_%s' % dc]
    s0 = z['sb_s0_dn_%s' % dc]
    ck('B1.replay_' + dc,
       float(np.abs(w.astype(np.float64)
                    - ref.astype(np.float64)).max())
       == 0.0)
    ck('B1.s0_' + dc,
       float(np.abs(s0.astype(np.float64)
                    - w.astype(np.float64)).max())
       == 0.0)
# write polarity recompute from npz
p30 = float(np.concatenate([
    z['wrec_pd_P'][30].ravel(),
    z['wrec_pd_A1'][30].ravel()]).mean())
p32 = float(np.concatenate([
    z['wrec_pd_P'][32].ravel(),
    z['wrec_pd_A1'][32].ravel()]).mean())
ck('B1.pol_L30',
   abs(p30 - res['part_a']['write_polarity']
       ['p_dn_L30']) < 1e-9)
ck('B1.pol_L32',
   abs(p32 - res['part_a']['write_polarity']
       ['p_dn_L32']) < 1e-9)
# L35 anchors
m35P = float(z['wrec_pd_P'][35].mean())
m35A = float(z['wrec_pd_A1'][35].mean())
ck('B1.L35_P',
   abs(m35P - (-11.44801139831543)) < 1e-4)
ck('B1.L35_A',
   abs(m35A - (-11.520125389099121)) < 1e-4)

# B2: find_span replication + E_cont/E_syn
annP = z20['annP_c']
annA = z20['annA_c']
ck('B2.ann_shape',
   annP.shape == (672, 12)
   and annP.dtype == np.int8)


def find_span(row):
    best = None
    k1 = None
    for k in range(1, N_NEW):
        isf = int(row[k]) in (3, 4)
        if isf and k1 is None:
            k1 = k
        if (not isf or k == N_NEW - 1) \
                and k1 is not None:
            k2 = (k if isf and k == N_NEW - 1
                  else k - 1)
            if best is None or \
                    (k2 - k1) > (best[1]
                                 - best[0]):
                best = (k1, k2)
            k1 = None
    return best


n_span_tot = 0
for dc in ('P', 'A1'):
    ann = annP if dc == 'P' else annA
    e = res['part_b']['effects'][dc]
    Ds = {'s1': [], 's2': [], 's3': []}
    lens = []
    n_sp = 0
    for j in range(672):
        best = find_span(ann[j])
        if best is None:
            continue
        n_sp += 1
        (k1, k2) = best
        lens.append(k2 - k1 + 1)
        base = z18['gt_cleanseq_clean__%s' % dc][j]
        base = base.astype(np.float64)
        for c in ('s1', 's2', 's3'):
            m = z['sb_%s_dn_%s' % (c, dc)][j]
            d = m.astype(np.float64) - base
            Ds[c].append(float(d[k1 + 1:].mean()))
    n_span_tot += n_sp
    ck('B2.nspan_' + dc, n_sp == e['n_span'])
    E_cont = float(np.mean([a - b for (a, b)
                            in zip(Ds['s1'],
                                   Ds['s3'])]))
    E_syn = float(np.mean([a - b for (a, b)
                           in zip(Ds['s1'],
                                  Ds['s2'])]))
    ck('B2.Econt_' + dc,
       abs(E_cont - e['E_cont']) < 1e-9)
    ck('B2.Esyn_' + dc,
       abs(E_syn - e['E_syn']) < 1e-9)
ck('B2.nspan_total', n_span_tot == 625)

# B3: Part C full re-simulation
mP = z18['gt_cleanseq_clean__P'][:672] \
    .astype(np.float64)
mA = z18['gt_cleanseq_clean__A1'][:672] \
    .astype(np.float64)
MSEQ = {'P': mP, 'A1': mA}
ANN = {'P': annP[:672], 'A1': annA[:672]}
S = -0.6801486439276976
MS = -6.052991245322225
content = {}
for dc in ('P', 'A1'):
    dm = MSEQ[dc][:, 1:] - MSEQ[dc][:, :-1]
    tab = np.zeros((10, N_NEW), dtype=np.float64)
    for k in range(N_NEW):
        col = ANN[dc][:, k].astype(np.int64)
        dv = dm[:, k]
        allm = dv.mean()
        for cc in range(10):
            sel = (col == cc)
            if sel.any():
                tab[cc, k] = float(
                    dv[sel].mean() - allm)
    content[dc] = tab
resid = []
for dc in ('P', 'A1'):
    dm = MSEQ[dc][:, 1:] - MSEQ[dc][:, :-1]
    lin = S * (MSEQ[dc][:, :-1] - MS)
    col = ANN[dc].astype(np.int64)
    dev = content[dc][col, np.arange(N_NEW)]
    resid.append((dm - lin - dev).ravel())
res_pool = np.concatenate(resid)
ck('B3.pool', len(res_pool) == 16128)
ck('B3.sigma',
   abs(float(np.std(res_pool))
       - res['part_c']['sigma']) < 1e-9)
N_REPS = 200
rng = np.random.default_rng(3122)
snap = {d: [np.zeros((672, N_REPS))
            for _ in range(N_NEW + 1)]
        for d in ('P', 'A1')}
for dc in ('P', 'A1'):
    seq = MSEQ[dc]
    mh = np.zeros((672, N_REPS),
                  dtype=np.float64)
    mh[:] = seq[:, 0][:, None]
    snap[dc][0] = mh.copy()
    col_all = ANN[dc].astype(np.int64)
    for k in range(N_NEW):
        noise = res_pool[rng.integers(
            0, len(res_pool),
            size=(672, N_REPS))]
        step = S * (mh - MS) \
            + content[dc][col_all[:, k], k][:, None] \
            + noise
        mh = mh + step
        snap[dc][k + 1] = mh.copy()


def auc_mw(pos_vals, neg_vals):
    x = np.concatenate([pos_vals, neg_vals])
    n1 = len(pos_vals)
    n2 = len(neg_vals)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) \
                and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0)
                 / (n1 * n2))


auc_sim = np.zeros(N_NEW + 1, dtype=np.float64)
for t in range(N_NEW + 1):
    auc_sim[t] = auc_mw(snap['P'][t].ravel(),
                        snap['A1'][t].ravel())
d_auc = float(np.abs(
    auc_sim - z['auc_sim'].astype(np.float64)).max())
ck('B3.auc_bit', d_auc < 1e-10)
pit = np.zeros((2, 672, N_NEW + 1),
               dtype=np.float64)
for di, dc in enumerate(('P', 'A1')):
    seq = MSEQ[dc]
    pit[di, :, 0] = 0.5
    for t in range(1, N_NEW + 1):
        s_i = snap[dc][t]
        tr = seq[:, t][:, None]
        pit[di, :, t] = (
            (s_i < tr).sum(axis=1)
            + 0.5 * (s_i == tr).sum(axis=1)) \
            / float(N_REPS)
d_pit = float(np.abs(
    pit - z['pit'].astype(np.float64)).max())
ck('B3.pit_bit', d_pit < 1e-12)
u = pit[:, :, 1:].ravel()
us = np.sort(u)
ecdf = np.arange(1, len(us) + 1) / float(len(us))
ks = float(np.max(np.abs(ecdf - us)))
ck('B3.ks',
   abs(ks - res['part_c']['pit_ks']) < 1e-12)
ck('B3.pit_t0', float(np.abs(
    z['pit'][:, :, 0] - 0.5).max()) == 0.0)
r_dist = float(np.corrcoef(
    auc_sim, z18['auc_curve'].astype(np.float64)
    )[0, 1])
ck('B3.rdist',
   abs(r_dist - res['part_c']['r_dist']) < 1e-9)
ck('B3.auc0_anchor',
   abs(float(z['auc_sim'][0])
       - float(z18['auc_curve'][0])) < 1e-12)

# ---------- C. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ck('C.n', len(led['measurements']) == 259)
m12 = [m for m in led['measurements']
       if m.get('phase') == 3122]
ck('C.meas', len(m12) == 1)
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ck('C.l14', len(l14['connects']) == 227)
ck('C.l14_has', any('meas3122' in c
                    for c in l14['connects']))
sha8 = led['ledger_sha256_8']
led2 = dict(led)
led2.pop('ledger_sha256_8', None)
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
re8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ck('C.sha8', re8 == sha8)

# ---------- D. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
tail = memo[-6500:]
ck('D.head', '## Phase 3122:' in tail)
import re
mm = re.search(
    r'## Phase 3122:[^\n]*\[(\d{4}-\d{2}-\d{2} '
    r'\d{2}:\d{2})\]', tail)
ck('D.bracket', mm is not None)
for s in (u'语法使内容可读',
          u'L35 \u5927\u8d1f\u5199 \u221211.4',
          'E_cont −3.68/−2.23',
          'PIT KS 0.053',
          'AUC→0.50',
          u'缺失持久轨迹锚点',
          u'### 1. \u4e09\u5927\u53d1\u73b0\uff08\u91cd\u590d\u4e09\u904d\uff09',
          u'### 2. \u5173\u952e\u6570\u503c',
          u'### 3. \u786c\u4f24',
          u'### 4. \u673a\u5236\u62fc\u56fe\u66f4\u65b0',
          u'### 5. 3123 \u9884\u6ce8\u518c',
          '`p3122_patch4.py`'):
    ck('D.has:' + s[:14], s in tail)
ck('D.tail_artifact',
   tail.rstrip().endswith(
       u'`p3122_patch4.py`\u3002'))

# ---------- E. wlog x2 ----------
for wdir in (ROOT + r'\.workbuddy\memory',
             r'C:\Users\Admin\WorkBuddy'
             r'\2026-09-17-01-30-05'
             r'\.workbuddy\memory'):
    wl = wdir + '\\2026-09-23.md'
    txt = io.open(wl, encoding='utf-8').read()
    ck('E.exp:' + wdir[0:6],
       'Phase 3122 Omega-P120' in txt)
    ck('E.clo:' + wdir[0:6],
       'Phase 3122 closeout' in txt)
    ck('E.sha:' + wdir[0:6],
       sha8 in txt)

# ---------- F. MEMORY.md ----------
memw = (ROOT + r'\.workbuddy\memory\MEMORY.md')
mtxt = io.open(memw, encoding='utf-8').read()
ck('F.len', len(mtxt) == 2899)
ck('F.max', 'max=3122' in mtxt)
_title3122 = [l for l in mtxt.splitlines()
              if l.startswith('## ')
              and '3122' in l
              and u'\u673a\u5236\u94fe\u72b6\u6001' in l]
ck('F.title', len(_title3122) == 1)
ck('F.l3122', u'\u8bed\u6cd5\u4f7f\u5185\u5bb9\u53ef\u8bfb' in mtxt
   and u'\u7f3a\u6301\u4e45\u8f68\u8ff9\u951a\u70b9' in mtxt)
ck('F.nol3121title',
   '## Mechanism chain state (3121)' not in mtxt)

# ---------- report ----------
out = ['VERIFY %d checks, %d fail'
       % (checks, len(fails))]
out += ['FAIL ' + f for f in fails]
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp'
        r'\p3122_verify_out.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('\n'.join(out))
