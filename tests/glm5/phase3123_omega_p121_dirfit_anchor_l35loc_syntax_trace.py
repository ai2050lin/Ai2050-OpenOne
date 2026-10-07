"""Phase 3123 (Omega-P121): per-direction operator
refit + trajectory-anchor state search + L35 final-
write localization + write-spectrum class binning +
syntax/content readability layer tracing.

Inputs (frozen): phase3118 traj npz (margins, gen
tokens), phase3120 ann npz, phase3120 result.json
operator, phase3121 result.json, phase3122 result +
npz (write spectra, s0/s1/s2/s3 margins, auc_sim),
phase3105 material.json, phase3113 capture_b.npz.

Parts:
  A offline: per-direction linear refit (S_dir,
    MS_dir) on content-residual transitions; anchor
    a_i = mean_t[m - dm_res/S]; split-half
    reliability; two simulations (dirfit operator,
    per-trajectory anchor) with empirical-residual
    bootstrap -> auc_sim recovery vs auc18 + rank
    PIT.
  B offline: L35 write localization by annotation
    class (answer vs other) from 3122 wrec_pd;
    L30/L32 query-step polarity; full 36-layer x
    class mean tables (w_dn and write/norm).
  C gpu: 4 conditions x 672 pairs x 2 dirs, all-37
    hidden-state w_dn readout (logit lens on model
    .norm); per-layer E_syn(L)=D_s1-D_s2 and
    E_cont(L)=D_s1-D_s3 emergence layers; s0 final-
    layer bit-exact check vs 3122 sb_s0.
"""
import hashlib
import io
import json
import os
import random as _rnd
import time
import zlib

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p121_dirfit_anchor_l35loc_' \
       'syntax_trace'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D18 = os.path.join(RDIR, 'phase3118',
                   'omega_p116_autoregressive_margin_'
                   'trajectory')
D20 = os.path.join(RDIR, 'phase3120',
                   'omega_p118_content_attr_amplifier_'
                   'behavior_opshape')
D22 = os.path.join(RDIR, 'phase3122',
                   'omega_p120_write_content_readout_'
                   'sentence_causal_dist_recon')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3123', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


N_NEW = 12
NP_A = 8 if SMOKE else 672

# ================================================================
# frozen inputs + integrity assertions
# ================================================================
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13, 'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
NB = len(pkB)
res18 = json.load(io.open(
    os.path.join(D18, 'result.json'),
    encoding='utf-8'))
assert res18['verdict'] == \
    'belief_decays_in_generation|' \
    'temporal_compensation|' \
    'top_ablation_changes_behavior'
res20 = json.load(io.open(
    os.path.join(D20, 'result.json'),
    encoding='utf-8'))
assert res20['verdict'] == \
    'attribution_unresolved|amplifier_partial|' \
    'not_applicable|mean_reversion_confirmed|' \
    'linear_operator'
assert res20['smoke'] is False
S_3120 = float(
    res20['part_c']['primary']['slope_lin'])
MS_3120 = float(
    res20['part_c']['fixed_point_raw_lin'])
assert abs(S_3120
           - (-0.6801486439276976)) < 1e-12
assert abs(MS_3120
           - (-6.052991245322225)) < 1e-12
res22 = json.load(io.open(
    os.path.join(D22, 'result.json'),
    encoding='utf-8'))
V22 = ('mixed|write_polarity_diverged|'
       'replay_bit_exact|'
       'sentence_content_push_down|'
       'sentence_content_push_down|'
       'syntax_effect_present|'
       'syntax_effect_present|'
       'oscillation_persist|'
       'distribution_reconstruction_failed|'
       'pit_marginal')
assert res22['verdict'] == V22
assert res22['smoke'] is False
z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
auc18 = z18['auc_curve']
assert len(auc18) == N_NEW + 1
assert abs(float(auc18[0])
           - 0.9809094210600907) < 1e-12
z20 = np.load(os.path.join(D20, 'p118_readout.npz'),
              allow_pickle=False)
annP_c = z20['annP_c']
annA_c = z20['annA_c']
assert annP_c.shape == (672, N_NEW)
assert annP_c.dtype == np.int8
z22 = np.load(os.path.join(D22, 'p120_readout.npz'),
              allow_pickle=False)
assert z22['wrec_pd_P'].shape == (36, 672, 13)
assert z22['sb_s0_dn_P'].shape == (672, 13)
assert abs(float(z22['auc_sim'][0])
           - float(auc18[0])) < 1e-12

# ================================================================
# design seal (frozen BEFORE observation)
# ================================================================
seal = {
    'phase': 3123,
    'name': NAME,
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'np_a': NP_A,
    'data_sources': {
        'margins_and_tokens':
            'phase3118 traj_readout.npz '
            '(gt_cleanseq_clean, gen_clean, '
            'auc_curve)',
        'annotation': 'phase3120 p118_readout.npz '
                      'annP_c/annA_c (0 other, 1 '
                      'syntax, 2 query_rest, 3 '
                      'fact_shaped, 4 fact_strict, '
                      '9 answer)',
        'operator_3120': 'slope_lin -0.6801 / '
                         'fixed_point_raw_lin '
                         '-6.0530 (reference only; '
                         'Part A refits per-dir)',
        'write_spectra_3122': 'phase3122 '
                              'p120_readout.npz '
                              'wrec_pd/wrec_pn '
                              '(36,672,13) + '
                              'sb_s0_dn + '
                              'sb_s1/s2/s3_dn',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz'},
    'part_a': {
        'refit': 'per direction: dm_res = dm - '
                 'content[dir][cls,k]; lstsq '
                 'dm_res ~ [m, 1] -> S_dir=b0, '
                 'MS_dir=-b1/b0',
        'anchor': 'a_i = mean_t[m_i(t) - '
                  'dm_res_i(t)/S_dir] over t=0..11; '
                  'split-half reliability = pearson '
                  '(t0..5 vs t6..11)',
        'sim_dirfit': 'm(t+1)=m(t)+S_dir*(m(t)-'
                      'MS_dir)+content+bootstrap'
                      '(resid_dir), init m0=seq[:,0]',
        'sim_anchor': 'per-trajectory anchor drawn '
                      'from empirical anchor pool '
                      '(with replacement, per rep); '
                      'm(t+1)=m(t)+S_dir*(m(t)-a)+'
                      'content+bootstrap(resid_anch)',
        'mc': {'n_reps': 200 if not SMOKE else 5,
               'seed': 3123,
               'noise': 'empirical residual '
                        'bootstrap per model, per '
                        'direction'}},
    'part_b': {
        'l35_loc': 'pooled dirs, steps k=0..11: '
                   'ans = ann==9 positions, oth = '
                   'ann!=9; sign pattern -> '
                   'final_brake_global / '
                   'at_answer / off_answer / '
                   'absent',
        'l30q_pol': 'pooled L30+L32 writes, k=0..11: '
                    'q = ann==2; q>0 & nq<=0 -> '
                    'assertion_write_query_specific; '
                    'q>0 & nq>0 -> '
                    'assertion_write_global_positive;'
                    ' else mixed',
        'spec_tables': 'mean write and mean '
                       'write/norm per (layer x '
                       'class[0,1,2,3,4,9,tail]) '
                       'per direction'},
    'part_c': {
        'readout': 'logit lens: model.model.norm '
                   'applied to every hidden_state '
                   '(0..36), project on w_dn, '
                   'margins at pos0+k, k=0..12; '
                   'float32 numpy dot identical to '
                   '3122 path',
        'conditions': ['s0_replay', 's1_swap',
                       's2_shuffle', 's3_dots'],
        'tracing': 'E_syn(L)=mean(D_s1-D_s2), '
                   'E_cont(L)=mean(D_s1-D_s3) over '
                   'span pairs, post-span steps '
                   'k>k1'},
    'gates': {
        'A_dirfit': 'corr(auc_sim_dir, auc18) >= '
                    '0.7 dirfit_confirmed; >= 0.5 '
                    'partial; else failed',
        'A_anchor': 'corr(auc_sim_anchor, auc18) '
                    '>= 0.7 anchor_confirmed; '
                    '>= 0.5 partial; else failed',
        'A_pit': 'KS(u, uniform) t>=1 pooled: '
                 '< 0.05 pit_calibrated; >= 0.15 '
                 'pit_miscalibrated; else '
                 'pit_marginal (per model)',
        'A_anchor_sep': 'mean(a_P) - mean(a_A1) > '
                        '0 -> '
                        'anchor_separation_present '
                        'else absent',
        'A_anchor_rel': 'split-half r >= 0.8 BOTH '
                        'dirs -> anchor_reliable; '
                        'one dir -> '
                        'anchor_reliable_partial; '
                        'else anchor_unreliable',
        'B_l35': 'sign pattern of ans/oth means '
                 '(pooled, k=0..11)',
        'B_l30q': 'sign pattern of q/nq means '
                  '(pooled L30+L32)',
        'C_repro': 'max|s0 final-layer margins - '
                   '3122 sb_s0| == 0 -> '
                   'replay_bit_exact; < 1e-6 -> '
                   'replay_ok; else diverged '
                   '(FATAL)',
        'C_syn_trace': 'L* = min L>=20 with '
                       'E_syn(L) <= -th (P -0.05, '
                       'A1 -0.025) -> '
                       'syntax_emerges_L{L*}; '
                       'none -> '
                       'syntax_below_threshold',
        'C_cont_trace': 'same with E_cont <= -th '
                        '-> content_emerges_L{L*} '
                        '/ '
                        'content_below_threshold'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('seal frozen (%s)' % seal['created'])

# ================================================================
# shared frozen arrays
# ================================================================
mP18 = z18['gt_cleanseq_clean__P']
mA18 = z18['gt_cleanseq_clean__A1']
MSEQ = {'P': mP18[:NP_A].astype(np.float64),
        'A1': mA18[:NP_A].astype(np.float64)}
ANN = {'P': annP_c[:NP_A], 'A1': annA_c[:NP_A]}


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


def rank_pit(snap_d, seq, n_reps):
    """rank-based PIT (same simulation), t>=1
    filled; t=0 set 0.5 (deterministic anchor)."""
    n, _ = seq.shape
    pit = np.zeros((n, N_NEW + 1),
                   dtype=np.float64)
    pit[:, 0] = 0.5
    for t in range(1, N_NEW + 1):
        s_i = snap_d[t]
        tr = seq[:, t][:, None]
        pit[:, t] = (
            (s_i < tr).sum(axis=1)
            + 0.5 * (s_i == tr).sum(axis=1)) \
            / float(n_reps)
    return pit


def ks_uniform(u):
    us = np.sort(u)
    n_u = len(us)
    ecdf = np.arange(1, n_u + 1) / float(n_u)
    return float(np.max(np.abs(ecdf - us)))


# ================================================================
# PART A: per-direction refit + anchor model
# ================================================================
log('== PART A: dirfit + anchor ==')
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

dirfit = {}
anchors = {'P': None, 'A1': None}
resid_by = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    dm = m[:, 1:] - m[:, :-1]
    col = ANN[dc].astype(np.int64)
    dev = content[dc][col, np.arange(N_NEW)]
    dm_res = dm - dev
    X = np.stack([m[:, :-1].ravel(),
                  np.ones(m[:, :-1].size)], 1)
    y = dm_res.ravel()
    bb, *_ = np.linalg.lstsq(X, y, rcond=None)
    S_dc = float(bb[0])
    MS_dc = float(-bb[1] / bb[0])
    resid_dc = y - X @ bb
    a_full = m[:, :-1] - dm_res / S_dc
    a_i = a_full.mean(1)
    rel_r = float(np.corrcoef(
        a_full[:, :6].mean(1),
        a_full[:, 6:].mean(1))[0, 1])
    resid_anch = (dm_res
                  - S_dc * (m[:, :-1]
                            - a_i[:, None])).ravel()
    dirfit[dc] = {'S': S_dc, 'MS': MS_dc,
                  'resid_std': float(
                      np.std(resid_dc)),
                  'anchor_mean': float(a_i.mean()),
                  'anchor_std': float(a_i.std()),
                  'rel_r': rel_r}
    anchors[dc] = a_i
    resid_by[dc] = {'dir': resid_dc,
                    'anch': resid_anch}
    log('A refit %s: S=%.4f MS=%.4f resid_std='
        '%.4f anchor=%.4f+-%.4f rel_r=%.4f'
        % (dc, S_dc, MS_dc,
           dirfit[dc]['resid_std'],
           dirfit[dc]['anchor_mean'],
           dirfit[dc]['anchor_std'], rel_r))

N_REPS = 5 if SMOKE else 200
rng_mc = np.random.default_rng(3123)
sim_dir = {}
sim_anch = {}
pit_dir = {}
pit_anch = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    col = ANN[dc].astype(np.int64)
    S_dc = dirfit[dc]['S']
    MS_dc = dirfit[dc]['MS']
    r_d = resid_by[dc]['dir']
    r_a = resid_by[dc]['anch']
    # dirfit simulation
    mh = np.zeros((NP_A, N_REPS),
                  dtype=np.float64)
    mh[:] = m[:, 0][:, None]
    snap = [mh.copy()]
    for k in range(N_NEW):
        noise = r_d[rng_mc.integers(
            0, len(r_d),
            size=(NP_A, N_REPS))]
        mh = mh + S_dc * (mh - MS_dc) \
            + content[dc][col[:, k], k][:, None] \
            + noise
        snap.append(mh.copy())
    sim_dir[dc] = snap
    pit_dir[dc] = rank_pit(snap, m, N_REPS)
    # anchor simulation
    a_draw = anchors[dc][rng_mc.integers(
        0, NP_A, size=(NP_A, N_REPS))]
    mh = np.zeros((NP_A, N_REPS),
                  dtype=np.float64)
    mh[:] = m[:, 0][:, None]
    snap = [mh.copy()]
    for k in range(N_NEW):
        noise = r_a[rng_mc.integers(
            0, len(r_a),
            size=(NP_A, N_REPS))]
        mh = mh + S_dc * (mh - a_draw) \
            + content[dc][col[:, k], k][:, None] \
            + noise
        snap.append(mh.copy())
    sim_anch[dc] = snap
    pit_anch[dc] = rank_pit(snap, m, N_REPS)

auc_dir = np.zeros(N_NEW + 1, dtype=np.float64)
auc_anch = np.zeros(N_NEW + 1, dtype=np.float64)
for t in range(N_NEW + 1):
    auc_dir[t] = auc_mw(
        np.concatenate([sim_dir['P'][t].ravel()]),
        np.concatenate([sim_dir['A1'][t].ravel()]))
    auc_anch[t] = auc_mw(
        np.concatenate(
            [sim_anch['P'][t].ravel()]),
        np.concatenate(
            [sim_anch['A1'][t].ravel()]))
r_dir = float(np.corrcoef(auc_dir, auc18)[0, 1])
r_anch = float(np.corrcoef(auc_anch, auc18)[0, 1])
dirfit_v = ('dirfit_confirmed' if r_dir >= 0.7
            else ('dirfit_partial' if r_dir >= 0.5
                  else 'dirfit_failed'))
anchor_v = ('anchor_confirmed' if r_anch >= 0.7
            else ('anchor_partial' if r_anch >= 0.5
                  else 'anchor_failed'))
u_d = np.concatenate([pit_dir['P'][:, 1:].ravel(),
                      pit_dir['A1'][:, 1:].ravel()])
u_a = np.concatenate(
    [pit_anch['P'][:, 1:].ravel(),
     pit_anch['A1'][:, 1:].ravel()])
ks_dir = ks_uniform(u_d)
ks_anch = ks_uniform(u_a)


def pit_v(ksv):
    return ('pit_calibrated' if ksv < 0.05
            else ('pit_miscalibrated'
                  if ksv >= 0.15
                  else 'pit_marginal'))


pit_dir_v = pit_v(ks_dir)
pit_anch_v = pit_v(ks_anch)
gap = (float(anchors['P'].mean())
       - float(anchors['A1'].mean()))
anchor_sep_v = ('anchor_separation_present'
                if gap > 0
                else 'anchor_separation_absent')
relP = dirfit['P']['rel_r']
relA = dirfit['A1']['rel_r']
if relP >= 0.8 and relA >= 0.8:
    anchor_rel_v = 'anchor_reliable'
elif relP >= 0.8 or relA >= 0.8:
    anchor_rel_v = 'anchor_reliable_partial'
else:
    anchor_rel_v = 'anchor_unreliable'
log('A-DIRFIT: r=%.4f -> %s; PIT KS=%.4f -> %s'
    % (r_dir, dirfit_v, ks_dir, pit_dir_v))
log('A-ANCHOR: r=%.4f -> %s; PIT KS=%.4f -> %s'
    % (r_anch, anchor_v, ks_anch, pit_anch_v))
log('A-SEP: gap=%.4f -> %s; rel P=%.4f A1=%.4f '
    '-> %s' % (gap, anchor_sep_v, relP, relA,
               anchor_rel_v))

# ================================================================
# PART B: L35 localization + spectrum binning
# (offline on 3122 npz, full 672)
# ================================================================
log('== PART B: L35 loc + spectrum bins ==')
CLS_LIST = [0, 1, 2, 3, 4, 9]
spec_dn = {}
spec_rel = {}
l35_stats = {}
l30q_stats = {}
for dc in ('P', 'A1'):
    w = z22['wrec_pd_%s' % dc].astype(np.float64)
    wn = z22['wrec_pn_%s' % dc].astype(np.float64)
    ann = annP_c if dc == 'P' else annA_c
    tab = np.full((36, 7), np.nan)
    rtab = np.full((36, 7), np.nan)
    for ci, c in enumerate(CLS_LIST):
        for L in range(36):
            vals = []
            rvals = []
            for k in range(N_NEW):
                sel = (ann[:, k] == c)
                if sel.any():
                    vals.append(
                        float(w[L][:, k][sel]
                              .mean()))
                    rvals.append(float(
                        (w[L][:, k][sel]
                         / wn[L][:, k][sel])
                        .mean()))
            if vals:
                tab[L, ci] = float(
                    np.mean(vals))
                rtab[L, ci] = float(
                    np.mean(rvals))
    for L in range(36):
        tab[L, 6] = float(w[L][:, 12].mean())
        rtab[L, 6] = float(
            (w[L][:, 12] / wn[L][:, 12]).mean())
    spec_dn[dc] = tab
    spec_rel[dc] = rtab
    # L35 answer vs other (k=0..11, pooled below)
    w35 = w[35][:, :N_NEW]
    am = (ann == 9)
    l35_stats[dc] = {
        'ans_mean': float(w35[am].mean()),
        'oth_mean': float(w35[~am].mean()),
        'ans_sum': float(w35[am].sum()),
        'oth_sum': float(w35[~am].sum()),
        'n_ans': int(am.sum()),
        'n_oth': int((~am).sum())}
    # L30/L32 query polarity
    for LL in (30, 32):
        wq = w[LL][:, :N_NEW]
        qm = (ann == 2)
        key = 'L%d' % LL
        st = l30q_stats.setdefault(key, {
            'q_sum': 0.0, 'q_n': 0,
            'nq_sum': 0.0, 'nq_n': 0,
            'q_mean_per_dir': [],
            'nq_mean_per_dir': []})
        st['q_sum'] += float(wq[qm].sum())
        st['q_n'] += int(qm.sum())
        st['nq_sum'] += float(wq[~qm].sum())
        st['nq_n'] += int((~qm).sum())
        st['q_mean_per_dir'].append(
            float(wq[qm].mean()))
        st['nq_mean_per_dir'].append(
            float(wq[~qm].mean()))

_ans_sum = sum(l35_stats[d]['ans_sum']
               for d in ('P', 'A1'))
_oth_sum = sum(l35_stats[d]['oth_sum']
               for d in ('P', 'A1'))
_n_ans = sum(l35_stats[d]['n_ans']
             for d in ('P', 'A1'))
_n_oth = sum(l35_stats[d]['n_oth']
             for d in ('P', 'A1'))
ans_all = _ans_sum / max(_n_ans, 1)
oth_all = _oth_sum / max(_n_oth, 1)
if ans_all < 0 and oth_all < 0:
    l35_v = 'final_brake_global'
elif ans_all < 0:
    l35_v = 'final_brake_at_answer'
elif oth_all < 0:
    l35_v = 'final_brake_off_answer'
else:
    l35_v = 'final_brake_absent'
_q_sum = sum(l30q_stats[k]['q_sum']
             for k in ('L30', 'L32'))
_q_n = sum(l30q_stats[k]['q_n']
           for k in ('L30', 'L32'))
_nq_sum = sum(l30q_stats[k]['nq_sum']
              for k in ('L30', 'L32'))
_nq_n = sum(l30q_stats[k]['nq_n']
            for k in ('L30', 'L32'))
q_all = _q_sum / max(_q_n, 1)
nq_all = _nq_sum / max(_nq_n, 1)
if q_all > 0 and nq_all <= 0:
    l30q_v = 'assertion_write_query_specific'
elif q_all > 0 and nq_all > 0:
    l30q_v = 'assertion_write_global_positive'
else:
    l30q_v = 'assertion_write_mixed'
log('B-L35: ans=%.4f oth=%.4f -> %s'
    % (ans_all, oth_all, l35_v))
log('B-L30Q: q=%.4f nq=%.4f -> %s'
    % (q_all, nq_all, l30q_v))

# ================================================================
# PART C: GPU layer tracing
# ================================================================
log('== PART C: layer tracing ==')


def build_prompt(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o)).encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents[s], PREDS[qrel], ents[o]))
    return text


p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
texts = []
for i in range(NB):
    pk = str(pkB[i])
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    cc = str(condB[i])
    if cc == 'P':
        (qrel, lrel) = (r, r)
    elif cc == 'A1':
        (qrel, lrel) = (r, ri1)
    else:
        (qrel, lrel) = (ri2, ri1)
    texts.append(build_prompt(mat5, s, o, lrel,
                              qrel))
hP = {}
hA1 = {}
for i in range(NB):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
    elif str(condB[i]) == 'A1':
        hA1[pk] = i
pks = sorted(hP.keys())
NP_ = len(pks)
assert NP_ == 672
NP_C = min(NP_A, NP_)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NL = len(model.model.layers)
log('model loaded NL=%d' % NL)
WU = model.lm_head.weight.detach()
assert model.lm_head.bias is None
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])
w_dn = (WU[YES_ID] - WU[NO_ID]) \
    .float().cpu().numpy()
log('readout ready: |w_dn|=%.4f'
    % float(np.linalg.norm(w_dn)))
norm_mod = model.model.norm


def forward_trackL(prompt_ids, gen_tokens):
    """teacher-forced forward; w_dn margins at
    every hidden state 0..NL (logit lens via
    model.model.norm); float32 numpy dot path
    identical to 3122."""
    gen_tokens = [int(x) for x in gen_tokens]
    ids = list(prompt_ids) + gen_tokens
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(gen_tokens) + 1
    ml = np.zeros((NL + 1, npts),
                  dtype=np.float64)
    with torch.inference_mode():
        out = model(t_in,
                    output_hidden_states=True,
                    use_cache=False)
        for L in range(NL + 1):
            h = norm_mod(
                out.hidden_states[L][0]
            ).float().cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dn)
        del out
    return ml


gP18 = z18['gen_clean__P']
gA18 = z18['gen_clean__A1']
PID_P = [tok(texts[hP[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]
PID_A = [tok(texts[hA1[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]

DOT_ID = tok('.', add_special_tokens=False)
DOT_ID = int(DOT_ID['input_ids'][0])
_line_tok = {}


def line_tokens(s2, o2, lrel):
    key = (s2, o2, lrel)
    if key not in _line_tok:
        txt = ' The %s %s the %s.' % (
            ents_all[s2], PREDS_all[lrel],
            ents_all[o2])
        _line_tok[key] = list(tok(
            txt, add_special_tokens=False)
            ['input_ids'])
    return _line_tok[key]


def context_triples(pk, dcode):
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    if dcode == 'P':
        lrel = r
    else:
        lrel = ri1
    D = [tuple(d) for d in
         mat5['distractors']['%d_%d' % (s, o)]]
    ctx = set((a, rr, b) for (a, rr, b) in
              ([(s, lrel, o)] + list(D)))
    return s, o, lrel, ctx


def pick_replacement(pk, dcode):
    (s, o, lrel, ctx) = context_triples(
        pk, dcode)
    rng1 = _rnd.Random(zlib.crc32(
        ('s1|%s|%s' % (pk, dcode))
        .encode('ascii')))
    ents_n = len(ents_all)
    for _try in range(200):
        s2 = rng1.randrange(ents_n)
        o2 = rng1.randrange(ents_n)
        if s2 == s or o2 == o:
            continue
        if (s2, lrel, o2) in ctx:
            continue
        if s2 == o2:
            continue
        return s2, o2
    raise RuntimeError('no replacement found')


def find_span(ann_row):
    best = None
    k1 = None
    for k in range(1, N_NEW):
        isf = int(ann_row[k]) in (3, 4)
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


SCOND = ('s0', 's1', 's2', 's3')
ML = {d: {c: None for c in SCOND}
      for d in ('P', 'A1')}
t0c = time.time()
for dcode, ann, gen18, PID in (
        ('P', annP_c, gP18, PID_P),
        ('A1', annA_c, gA18, PID_A)):
    store = {c: [] for c in SCOND}
    n_done = 0
    for j in range(NP_C):
        pk = pks[j]
        prompt_ids = PID[j]
        base = [int(t) for t in gen18[j]]
        best = find_span(ann[j])
        toks = {c: list(base) for c in SCOND}
        if best is not None:
            (k1, k2) = best
            Lspan = k2 - k1 + 1
            (s2, o2) = pick_replacement(
                pk, dcode)
            (_, _, lrel, _) = context_triples(
                pk, dcode)
            sub = line_tokens(s2, o2, lrel)
            n_pad = max(0, Lspan - len(sub))
            sub = sub[:Lspan]
            sub = sub + [DOT_ID] * n_pad
            rng2 = _rnd.Random(zlib.crc32(
                ('s2|%s|%s' % (pk, dcode))
                .encode('ascii')))
            shuf = list(sub)
            rng2.shuffle(shuf)
            toks['s1'][k1:k2 + 1] = sub
            toks['s2'][k1:k2 + 1] = shuf
            toks['s3'][k1:k2 + 1] = \
                [DOT_ID] * Lspan
        for c in SCOND:
            store[c].append(forward_trackL(
                prompt_ids, toks[c]))
        n_done += 1
        if n_done % 128 == 0:
            log('C %s %d/%d (%.1fs)'
                % (dcode, n_done, NP_C,
                   time.time() - t0c))
    for c in SCOND:
        ML[dcode][c] = np.array(store[c],
                                dtype=np.float64)
    log('partC %s done (%.1fs)'
        % (dcode, time.time() - t0c))

# s0 integrity vs 3122 (final layer)
drep = 0.0
for dc in ('P', 'A1'):
    ref = z22['sb_s0_dn_%s' % dc][:NP_C]
    drep = max(drep, float(np.abs(
        ML[dc]['s0'][:, NL, :]
        - ref.astype(np.float64)).max()))
if drep == 0.0:
    c_repro_v = 'replay_bit_exact'
elif drep < 1e-6:
    c_repro_v = 'replay_ok'
else:
    c_repro_v = 'replay_diverged'
log('C-REPRO: max diff %.3e -> %s'
    % (drep, c_repro_v))
assert c_repro_v != 'replay_diverged', \
    's0 replay diverged - FATAL'

# per-layer effects
curves = {}
lstars = {}
th_map = {'P': 0.05, 'A1': 0.025}
for dcode in ('P', 'A1'):
    ann = annP_c if dcode == 'P' else annA_c
    syn = np.zeros(NL + 1, dtype=np.float64)
    con = np.zeros(NL + 1, dtype=np.float64)
    n_sp = 0
    for j in range(NP_C):
        best = find_span(ann[j])
        if best is None:
            continue
        n_sp += 1
        (k1, k2) = best
        s0 = ML[dcode]['s0'][j]
        d1 = ML[dcode]['s1'][j] - s0
        d2 = ML[dcode]['s2'][j] - s0
        d3 = ML[dcode]['s3'][j] - s0
        for L in range(NL + 1):
            syn[L] += float(
                d1[L, k1 + 1:].mean()
                - d2[L, k1 + 1:].mean())
            con[L] += float(
                d1[L, k1 + 1:].mean()
                - d3[L, k1 + 1:].mean())
    syn /= max(n_sp, 1)
    con /= max(n_sp, 1)
    th = th_map[dcode]
    ls = None
    lc = None
    for L in range(20, NL + 1):
        if ls is None and syn[L] <= -th:
            ls = L
        if lc is None and con[L] <= -th:
            lc = L
    lstars[dcode] = {'syn_L': ls, 'cont_L': lc,
                     'n_span': n_sp}
    curves[dcode] = {'syn': syn, 'cont': con}
    syn_v = ('syntax_emerges_L%d' % ls if ls
             is not None
             else 'syntax_below_threshold')
    con_v = ('content_emerges_L%d' % lc if lc
             is not None
             else 'content_below_threshold')
    curves[dcode]['syn_v'] = syn_v
    curves[dcode]['cont_v'] = con_v
    log('C-TRACE %s: n_span=%d syn_L=%s cont_L='
        '%s' % (dcode, n_sp, ls, lc))

# ================================================================
# verdict + save
# ================================================================
verdict = '|'.join([
    dirfit_v, anchor_v, pit_dir_v, pit_anch_v,
    anchor_sep_v, anchor_rel_v,
    l35_v, l30q_v, c_repro_v,
    curves['P']['syn_v'], curves['A1']['syn_v'],
    curves['P']['cont_v'],
    curves['A1']['cont_v']])
runtime = round(time.time() - T0, 1)
results = {
    'phase': 3123,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'n_pairs': NP_,
    'np_a': NP_A,
    'runtime_s': runtime,
    'part_a': {
        'dirfit': dirfit,
        'anchors': {
            'mean_P': float(anchors['P'].mean()),
            'mean_A1': float(
                anchors['A1'].mean()),
            'std_P': float(anchors['P'].std()),
            'std_A1': float(anchors['A1'].std()),
            'gap': gap,
            'rel_r_P': relP, 'rel_r_A1': relA,
            'verdict_sep': anchor_sep_v,
            'verdict_rel': anchor_rel_v},
        'sim': {
            'r_dir': r_dir,
            'dirfit_verdict': dirfit_v,
            'r_anchor': r_anch,
            'anchor_verdict': anchor_v,
            'pit_ks_dir': ks_dir,
            'pit_dir_verdict': pit_dir_v,
            'pit_ks_anchor': ks_anch,
            'pit_anchor_verdict': pit_anch_v,
            'auc_sim_dir': [float(x)
                            for x in auc_dir],
            'auc_sim_anchor': [float(x)
                               for x in auc_anch],
            'n_reps': N_REPS}},
    'part_b': {
        'l35': {'ans_mean_pooled': float(ans_all),
                'oth_mean_pooled': float(oth_all),
                'per_dir': l35_stats,
                'verdict': l35_v},
        'l30q': {'q_mean_pooled': float(q_all),
                 'nq_mean_pooled': float(nq_all),
                 'per_layer': l30q_stats,
                 'verdict': l30q_v}},
    'part_c': {
        'repro': {'max_diff': drep,
                  'verdict': c_repro_v},
        'n_span': {'P': lstars['P']['n_span'],
                   'A1': lstars['A1']['n_span']},
        'lstar': {d: {'syn_L': lstars[d]['syn_L'],
                      'cont_L': lstars[d]
                      ['cont_L']}
                  for d in ('P', 'A1')},
        'curves': {
            'E_syn_P': [float(x) for x in
                        curves['P']['syn']],
            'E_syn_A1': [float(x) for x in
                         curves['A1']['syn']],
            'E_cont_P': [float(x) for x in
                         curves['P']['cont']],
            'E_cont_A1': [float(x) for x in
                          curves['A1']['cont']]},
        'verdicts': {
            'P': {'syn': curves['P']['syn_v'],
                  'cont': curves['P']['cont_v']},
            'A1': {'syn': curves['A1']['syn_v'],
                   'cont': curves['A1']
                   ['cont_v']}}},
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False,
              indent=1)
npz_out = {}
for dc in ('P', 'A1'):
    npz_out['anchors_%s' % dc] = \
        anchors[dc].astype(np.float32)
    npz_out['spec_dn_%s' % dc] = \
        spec_dn[dc].astype(np.float32)
    npz_out['spec_rel_%s' % dc] = \
        spec_rel[dc].astype(np.float32)
    for c in SCOND:
        npz_out['ml_%s_%s' % (c, dc)] = \
            ML[dc][c].astype(np.float32)
npz_out['auc_sim_dir'] = auc_dir
npz_out['auc_sim_anchor'] = auc_anch
npz_out['pit_dir'] = np.stack(
    [pit_dir['P'], pit_dir['A1']])
npz_out['pit_anchor'] = np.stack(
    [pit_anch['P'], pit_anch['A1']])
for dc in ('P', 'A1'):
    npz_out['E_syn_%s' % dc] = \
        curves[dc]['syn'].astype(np.float32)
    npz_out['E_cont_%s' % dc] = \
        curves[dc]['cont'].astype(np.float32)
np.savez(os.path.join(OUT, 'p121_readout.npz'),
         **npz_out)
log('verdict=%s' % verdict)
log('done (%.1fs)' % runtime)
print('phase3123 done: %s' % verdict)
