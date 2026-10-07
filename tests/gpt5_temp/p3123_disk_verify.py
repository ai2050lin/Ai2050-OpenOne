# -*- coding: utf-8 -*-
"""Phase 3123 disk verify (independent):
A result asserts -> B recompute from npz
(anchors/spec/l35/l30q/ml/E-curves) ->
C Part A full simulation re-run (rng 3123)
-> D ledger -> E MEMO -> F wlogs -> G MEMORY."""
import hashlib
import io
import json
import re

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D18 = RDIR + r'\phase3118' \
      r'\omega_p116_autoregressive_margin_' \
      'trajectory'
D20 = RDIR + r'\phase3120' \
      r'\omega_p118_content_attr_amplifier_' \
      'behavior_opshape'
D22 = RDIR + r'\phase3122' \
      r'\omega_p120_write_content_readout_' \
      'sentence_causal_dist_recon'
OUTD = RDIR + r'\phase3123' \
       r'\omega_p121_dirfit_anchor_l35loc_' \
       'syntax_trace'
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
VF = (ROOT + r'\tests\gpt5_temp'
      r'\p3123_verify_out.txt')
fails = []
o = []


def sec(name, conds):
    bad = [i for i, c in enumerate(conds)
           if not c]
    o.append('%s: %d/%d pass%s'
             % (name, len(conds) - len(bad),
                len(conds),
                (' FAIL at ' + str(bad))
                if bad else ''))
    if bad:
        fails.append((name, bad))


V = ('dirfit_failed|anchor_failed|'
     'pit_calibrated|pit_calibrated|'
     'anchor_separation_present|'
     'anchor_unreliable|final_brake_global|'
     'assertion_write_global_positive|'
     'replay_bit_exact|syntax_emerges_L21|'
     'syntax_emerges_L21|'
     'content_emerges_L20|'
     'content_emerges_L20')

# ---------- A. result asserts ----------
res = json.load(io.open(
    OUTD + r'\result.json', encoding='utf-8'))
z22 = np.load(os.path.join(D22,
                           'p120_readout.npz'),
              allow_pickle=False) \
    if False else None
import os  # noqa: E402

z22 = np.load(os.path.join(D22,
                           'p120_readout.npz'),
              allow_pickle=False)
z18 = np.load(os.path.join(D18,
                           'traj_readout.npz'),
              allow_pickle=False)
z20 = np.load(os.path.join(D20,
                           'p118_readout.npz'),
              allow_pickle=False)
z23 = np.load(os.path.join(OUTD,
                           'p121_readout.npz'),
              allow_pickle=False)
sec('A0_load', [res['verdict'] == V,
                res['smoke'] is False,
                res['n_pairs'] == 672,
                res['np_a'] == 672,
                abs(res['runtime_s']
                    - 230.4) < 0.05])
pa = res['part_a']
chk = [abs(pa['dirfit']['P']['S']
           - (-0.601611613690753)) < 1e-12,
       abs(pa['dirfit']['A1']['S']
           - (-0.5540798866844862)) < 1e-12,
       abs(pa['dirfit']['P']['MS']
           - (-4.955594255793292)) < 1e-12,
       abs(pa['dirfit']['A1']['MS']
           - (-7.154421990932929)) < 1e-12,
       abs(pa['anchors']['gap']
           - 2.198827735139629) < 1e-12,
       abs(pa['sim']['r_dir']
           - 0.2285152966483294) < 1e-12,
       abs(pa['sim']['r_anchor']
           - 0.23528388458123164) < 1e-12,
       abs(pa['sim']['pit_ks_dir']
           - 0.03650545634920632) < 1e-12,
       abs(pa['sim']['pit_ks_anchor']
           - 0.03262400793650794) < 1e-12]
pc = res['part_b']
chk += [abs(pc['l35']['ans_mean_pooled']
            - (-9.701520158776216)) < 1e-12,
        abs(pc['l35']['per_dir']['A1']
            ['ans_mean']
            - (-8.029358918468157)) < 1e-12,
        abs(pc['l30q']['q_mean_pooled']
            - 1.482873646914959) < 1e-12]
pc3 = res['part_c']
chk += [pc3['repro']['max_diff'] == 0.0,
        pc3['n_span'] == {'P': 305, 'A1': 320},
        pc3['lstar']['P'] == {'syn_L': 21,
                              'cont_L': 20},
        abs(pc3['curves']['E_syn_P'][36]
            - (-2.2558336193932855)) < 1e-12,
        abs(pc3['curves']['E_cont_A1'][36]
            - (-2.2313470710720873)) < 1e-12]
r22 = json.load(io.open(
    os.path.join(D22, 'result.json'),
    encoding='utf-8'))
chk += [abs(pc3['curves']['E_syn_P'][36]
            - r22['part_b']['effects']['P']
            ['E_syn']) < 1e-9,
        abs(pc3['curves']['E_cont_P'][36]
            - r22['part_b']['effects']['P']
            ['E_cont']) < 1e-9,
        abs(pc3['curves']['E_cont_A1'][36]
            - r22['part_b']['effects']['A1']
            ['E_cont']) < 1e-9]
sec('A_result', chk)

# ---------- B. recompute from npz ----------
mP18 = z18['gt_cleanseq_clean__P']
mA18 = z18['gt_cleanseq_clean__A1']
MSEQ = {'P': mP18[:672].astype(np.float64),
        'A1': mA18[:672].astype(np.float64)}
ANN = {'P': z20['annP_c'][:672],
       'A1': z20['annA1_c'][:672]
       if 'annA1_c' in z20.files
       else z20['annA_c'][:672]}
N_NEW = 12
content = {}
for dc in ('P', 'A1'):
    dm = MSEQ[dc][:, 1:] - MSEQ[dc][:, :-1]
    tab = np.zeros((10, N_NEW))
    for k in range(N_NEW):
        col = ANN[dc][:, k].astype(np.int64)
        dv = dm[:, k]
        allm = dv.mean()
        for cc in range(10):
            sel_ = (col == cc)
            if sel_.any():
                tab[cc, k] = float(
                    dv[sel_].mean() - allm)
    content[dc] = tab
fit = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    dm = m[:, 1:] - m[:, :-1]
    col = ANN[dc].astype(np.int64)
    dev = content[dc][col,
                      np.arange(N_NEW)]
    dm_res = dm - dev
    X = np.stack([m[:, :-1].ravel(),
                  np.ones(m[:, :-1].size)], 1)
    y = dm_res.ravel()
    bb, *_ = np.linalg.lstsq(X, y,
                             rcond=None)
    S_dc = float(bb[0])
    MS_dc = float(-bb[1] / bb[0])
    a_full = m[:, :-1] - dm_res / S_dc
    a_i = a_full.mean(1)
    rel_r = float(np.corrcoef(
        a_full[:, :6].mean(1),
        a_full[:, 6:].mean(1))[0, 1])
    fit[dc] = {'S': S_dc, 'MS': MS_dc,
               'a': a_i, 'rel': rel_r}
c1 = [abs(fit['P']['S']
          - pa['dirfit']['P']['S']) < 1e-9,
      abs(fit['A1']['S']
          - pa['dirfit']['A1']['S']) < 1e-9,
      abs(fit['P']['MS']
          - pa['dirfit']['P']['MS']) < 1e-9,
      abs(fit['A1']['MS']
          - pa['dirfit']['A1']['MS']) < 1e-9,
      abs(fit['P']['rel']
          - pa['dirfit']['P']['rel_r'])
      < 1e-9,
      abs(fit['A1']['rel']
          - pa['dirfit']['A1']['rel_r'])
      < 1e-9,
      np.max(np.abs(
          fit['P']['a']
          - z23['anchors_P'].astype(
              np.float64))) < 1e-5,
      np.max(np.abs(
          fit['A1']['a']
          - z23['anchors_A1'].astype(
              np.float64))) < 1e-5]
sec('B1_refit_anchors', c1)

CLS_LIST = [0, 1, 2, 3, 4, 9]
c2 = []
sp = {}
for dc in ('P', 'A1'):
    w = z22['wrec_pd_%s' % dc].astype(
        np.float64)
    ann = ANN[dc]
    tab = np.full((36, 7), np.nan)
    for ci, c in enumerate(CLS_LIST):
        for L in range(36):
            vals = []
            for k in range(N_NEW):
                sel_ = (ann[:, k] == c)
                if sel_.any():
                    vals.append(float(
                        w[L][:, k][sel_]
                        .mean()))
            if vals:
                tab[L, ci] = float(
                    np.mean(vals))
    for L in range(36):
        tab[L, 6] = float(
            w[L][:, 12].mean())
    sp[dc] = tab
c2 += [abs(sp['P'][35, 5]
           - z23['spec_dn_P'].astype(
               np.float64)[35, 5]) < 1e-4,
       abs(sp['A1'][30, 2]
           - z23['spec_dn_A1'].astype(
               np.float64)[30, 2]) < 1e-4,
       abs(sp['P'][34, 6]
           - z23['spec_dn_P'].astype(
               np.float64)[34, 6]) < 1e-4]
l35 = pc['l35']
w35s = 0.0
w35o = 0.0
na = 0
no = 0
for dc in ('P', 'A1'):
    w = z22['wrec_pd_%s' % dc].astype(
        np.float64)
    w35 = w[35][:, :N_NEW]
    am_ = (ANN[dc] == 9)
    w35s += float(w35[am_].sum())
    w35o += float(w35[~am_].sum())
    na += int(am_.sum())
    no += int((~am_).sum())
c2 += [abs(w35s / na
           - l35['ans_mean_pooled']) < 1e-9,
       abs(w35o / no
           - l35['oth_mean_pooled']) < 1e-9,
       na == 1344 and no == 14784]
lq = pc['l30q']
qs = 0.0
qn = 0
nqs = 0.0
nqn = 0
for dc in ('P', 'A1'):
    w = z22['wrec_pd_%s' % dc].astype(
        np.float64)
    for LL in (30, 32):
        wq = w[LL][:, :N_NEW]
        qm_ = (ANN[dc] == 2)
        qs += float(wq[qm_].sum())
        qn += int(qm_.sum())
        nqs += float(wq[~qm_].sum())
        nqn += int((~qm_).sum())
c2 += [abs(qs / qn
           - lq['q_mean_pooled']) < 1e-9,
       abs(nqs / nqn
           - lq['nq_mean_pooled']) < 1e-9,
       qn == 140]
sec('B2_spec_l35_l30q', c2)

# B3 ml/E curves/n_span from npz
c3 = []
mlP0 = z23['ml_s0_P'].astype(np.float64)
ref0 = z22['sb_s0_dn_P'].astype(np.float64)
c3.append(np.max(np.abs(
    mlP0[:, 36, :] - ref0)) < 1e-3)
SCOND = ('s0', 's1', 's2', 's3')


def find_span(ann_row):
    best = None
    k1 = None
    for k in range(1, N_NEW):
        isf = int(ann_row[k]) in (3, 4)
        if isf and k1 is None:
            k1 = k
        if (not isf or k == N_NEW - 1) \
                and k1 is not None:
            k2 = (k if isf
                  and k == N_NEW - 1
                  else k - 1)
            if best is None or \
                    (k2 - k1) > (best[1]
                                 - best[0]):
                best = (k1, k2)
            k1 = None
    return best


for dc in ('P', 'A1'):
    ann = ANN[dc]
    ML = {c: z23['ml_%s_%s' % (c, dc)]
          .astype(np.float64)
          for c in SCOND}
    syn = np.zeros(37)
    con = np.zeros(37)
    n_sp = 0
    for j in range(672):
        best = find_span(ann[j])
        if best is None:
            continue
        n_sp += 1
        (k1, k2) = best
        s0 = ML['s0'][j]
        d1 = ML['s1'][j] - s0
        d2 = ML['s2'][j] - s0
        d3 = ML['s3'][j] - s0
        for L in range(37):
            syn[L] += float(
                d1[L, k1 + 1:].mean()
                - d2[L, k1 + 1:].mean())
            con[L] += float(
                d1[L, k1 + 1:].mean()
                - d3[L, k1 + 1:].mean())
    syn /= max(n_sp, 1)
    con /= max(n_sp, 1)
    eS = np.array(pc3['curves']
                  ['E_syn_%s' % dc])
    eC = np.array(pc3['curves']
                  ['E_cont_%s' % dc])
    c3.append(np.max(np.abs(syn - eS))
              < 5e-3)
    c3.append(np.max(np.abs(con - eC))
              < 5e-3)
    c3.append(n_sp == pc3['n_span'][dc])
sec('B3_ml_Ecurves_nspan', c3)

# ---------- C. Part A sim re-run ----------
N_REPS = 200
rng = np.random.default_rng(3123)
sim_dir = {}
sim_anch = {}
pit_dir = {}
pit_anch = {}


def auc_mw2(pos_vals, neg_vals):
    x = np.concatenate([pos_vals, neg_vals])
    n1 = len(pos_vals)
    n2 = len(neg_vals)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x),
                     dtype=np.float64)
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


def rank_pit(snap_d, seq, n_reps):
    n, _ = seq.shape
    pit = np.zeros((n, N_NEW + 1))
    pit[:, 0] = 0.5
    for t in range(1, N_NEW + 1):
        s_i = snap_d[t]
        tr = seq[:, t][:, None]
        pit[:, t] = (
            (s_i < tr).sum(axis=1)
            + 0.5 * (s_i == tr).sum(axis=1)) \
            / float(n_reps)
    return pit


for dc in ('P', 'A1'):
    m = MSEQ[dc]
    col = ANN[dc].astype(np.int64)
    S_dc = fit[dc]['S']
    MS_dc = fit[dc]['MS']
    dm = m[:, 1:] - m[:, :-1]
    dev = content[dc][col,
                      np.arange(N_NEW)]
    dm_res = dm - dev
    X = np.stack([m[:, :-1].ravel(),
                  np.ones(m[:, :-1].size)], 1)
    y = dm_res.ravel()
    bb, *_ = np.linalg.lstsq(X, y,
                             rcond=None)
    r_d = y - X @ bb
    a_i = fit[dc]['a']
    r_a = (dm_res
           - S_dc * (m[:, :-1]
                     - a_i[:, None])).ravel()
    mh = np.zeros((672, N_REPS))
    mh[:] = m[:, 0][:, None]
    snap = [mh.copy()]
    for k in range(N_NEW):
        noise = r_d[rng.integers(
            0, len(r_d),
            size=(672, N_REPS))]
        mh = mh + S_dc * (mh - MS_dc) \
            + content[dc][col[:, k], k][:, None] \
            + noise
        snap.append(mh.copy())
    sim_dir[dc] = snap
    pit_dir[dc] = rank_pit(snap, m, N_REPS)
    a_draw = a_i[rng.integers(
        0, 672, size=(672, N_REPS))]
    mh = np.zeros((672, N_REPS))
    mh[:] = m[:, 0][:, None]
    snap = [mh.copy()]
    for k in range(N_NEW):
        noise = r_a[rng.integers(
            0, len(r_a),
            size=(672, N_REPS))]
        mh = mh + S_dc * (mh - a_draw) \
            + content[dc][col[:, k], k][:, None] \
            + noise
        snap.append(mh.copy())
    sim_anch[dc] = snap
    pit_anch[dc] = rank_pit(snap, m, N_REPS)
auc_dir = np.zeros(N_NEW + 1)
auc_anch = np.zeros(N_NEW + 1)
for t in range(N_NEW + 1):
    auc_dir[t] = auc_mw2(
        sim_dir['P'][t].ravel(),
        sim_dir['A1'][t].ravel())
    auc_anch[t] = auc_mw2(
        sim_anch['P'][t].ravel(),
        sim_anch['A1'][t].ravel())
md1 = float(np.max(np.abs(
    auc_dir - z23['auc_sim_dir'])))
md2 = float(np.max(np.abs(
    auc_anch - z23['auc_sim_anchor'])))
md3 = float(np.max(np.abs(
    np.stack([pit_dir['P'],
              pit_dir['A1']])
    - z23['pit_dir'])))
md4 = float(np.max(np.abs(
    np.stack([pit_anch['P'],
              pit_anch['A1']])
    - z23['pit_anchor'])))
u_d = np.concatenate(
    [pit_dir['P'][:, 1:].ravel(),
     pit_dir['A1'][:, 1:].ravel()])
u_a = np.concatenate(
    [pit_anch['P'][:, 1:].ravel(),
     pit_anch['A1'][:, 1:].ravel()])


def ks_uniform(u):
    us = np.sort(u)
    n_u = len(us)
    ecdf = np.arange(1, n_u + 1) \
        / float(n_u)
    return float(np.max(np.abs(ecdf - us)))


ks_d = ks_uniform(u_d)
ks_a = ks_uniform(u_a)
rd = float(np.corrcoef(
    auc_dir, z18['auc_curve'])[0, 1])
ra = float(np.corrcoef(
    auc_anch, z18['auc_curve'])[0, 1])
sec('C_sim_rerun',
    [md1 < 1e-10, md2 < 1e-10,
     md3 < 1e-12, md4 < 1e-12,
     abs(ks_d - pa['sim']['pit_ks_dir'])
     < 1e-12,
     abs(ks_a - pa['sim']['pit_ks_anchor'])
     < 1e-12,
     abs(rd - pa['sim']['r_dir']) < 1e-12,
     abs(ra - pa['sim']['r_anchor'])
     < 1e-12])
o.append('C maxdiff auc_dir=%.3e auc_anch='
         '%.3e pit=%.3e/%.3e' % (md1, md2,
                                 md3, md4))

# ---------- D. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m3 = [m for m in led['measurements']
      if m.get('phase') == 3123]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
led.pop('ledger_sha256_8', None)
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
sec('D_ledger',
    [len(led['measurements']) == 260,
     len(m3) == 1,
     len(l14['connects']) == 228,
     m3 and m3[0]['meas_id']
     == 'meas3123_omega_p121_dirfit_anchor_'
     'l35loc_syntax_trace',
     sha8 == 'e9d06fec'])

# ---------- E. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
cnt = memo.count('## Phase 3123:')
tail = memo[memo.index('## Phase 3123:'):] \
    if cnt else ''
mline = [l for l in tail.splitlines()
         if l.startswith('## Phase 3123:')][0] \
    if cnt else ''
has_ts = bool(re.search(
    r'\[2026-\d{2}-\d{2} \d{2}:\d{2}\]',
    mline)) if cnt else False
f1 = (u'\u9759\u6001\u6301\u4e45\u951a\u70b9'
      u'\u5047\u8bbe\u5426\u5b9a')
f2 = u'\u6269\u6563\u4f2a\u5f71'
f3 = (u'\u8bed\u6cd5\u95e8\u63a7 L21'
      u'\u3001\u5185\u5bb9\u6548\u5e94 L20')
f4 = (u'\u5199\u5165\u94fe\u4e0a\u6e38')
f5 = (u'\u7b54\u6848\u6b65\u91ca\u538b')
f6 = (u'\u6b8b\u5dee\u5fc5\u4e3a\u5171\u6a21')
sec('E_memo',
    [cnt == 1,
     has_ts,
     f1 in tail, f2 in tail, f3 in tail,
     f4 in tail, f5 in tail, f6 in tail,
     tail.rstrip().endswith(
         u'`tests/gpt5_temp/p3123_patch1.py`'
         u'\u3002'),
     '3124' in tail])

# ---------- F. wlogs ----------
cf = []
for wdir in (WLOG_D, WLOG_C):
    t = io.open(wdir + '\\' + '2026-09-23.md',
                encoding='utf-8').read()
    cf += ['Phase 3123 Omega-P121' in t,
           'Phase 3123 closeout' in t,
           'e9d06fec' in t]
sec('F_wlogs', cf)

# ---------- G. MEMORY ----------
mem = io.open(MEMO_W,
              encoding='utf-8').read()
sec('G_memory',
    [len(mem) == 2860,
     mem.count(u'\u673a\u5236\u94fe\u72b6'
               u'\u6001\uff083123\uff09') == 1,
     mem.count(u'- 3123\uff08T4\uff09') == 1,
     mem.count('max=3123') == 1,
     mem.count(u'- 3114\u20133117\uff1a') == 1,
     mem.count(u'- 3116\uff1a') == 0,
     mem.count(u'- 3115\uff1a') == 0])

# ---------- report ----------
o.append('TOTAL FAILS: %d' % len(fails))
io.open(VF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('verify done, fails=%d' % len(fails))
