"""Phase 3124 (Omega-P122): residual common-mode
decomposition + Kalman anchor-drift test + L35
release decomposition (offline) + GLM4 cross-model
sentence paradigm + layer tracing (GPU).

Inputs (frozen): phase3118 traj npz, phase3120 ann
npz, phase3122 result + npz (wrec_pd/pn), phase3123
result + npz (fits, spec tables), phase3105
material.json, phase3113 capture_b.npz.

Parts:
  A offline: per-direction refit (assert vs 3123);
    3-way resid decomposition (step-dummy common-
    mode / m-structure / remainder); deterministic
    no-noise simulation -> auc_det vs auc18; median
    per-trajectory R2; analytic iid plateau check
    vs 3123 simulated plateau.
  B offline: lag-1 autocorr of implied anchor;
    Kalman filter random-walk anchor with global q
    MLE (grid x sigma_a2) + RTS smoother; own-
    smoothed-anchor deterministic simulation.
  C offline: L35 A1 answer-step release first-order
    decomposition (norm-gain vs semantic via
    wn = w/pn); per-class L30/L32/L35 spec readout.
  D gpu (glm4-9b-chat-hf): same 672x2 material,
    greedy gen 12 tok (batched), re-tokenize,
    query-line span replacement s0/s1/s2/s3, all-
    layer logit-lens w_dn margins (41 states),
    E_syn/E_cont curves with RELATIVE L* gates;
    path-valid + s0 repro FATAL gates; readout
    sanity AUC.
"""
import hashlib
import io
import json
import math
import os
import random as _rnd
import time
import zlib

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p122_resid_cm_kalman_l35rel_glm4x'
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
D23 = os.path.join(RDIR, 'phase3123',
                   'omega_p121_dirfit_anchor_l35loc_'
                   'syntax_trace')
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3124', NAME)
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
NP = 672
NP_D = 8 if SMOKE else NP
NP_CHECK = 8 if SMOKE else 32
NP_REPRO = 4 if SMOKE else 16

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
res23 = json.load(io.open(
    os.path.join(D23, 'result.json'),
    encoding='utf-8'))
V23 = ('dirfit_failed|anchor_failed|'
       'pit_calibrated|pit_calibrated|'
       'anchor_separation_present|'
       'anchor_unreliable|final_brake_global|'
       'assertion_write_global_positive|'
       'replay_bit_exact|syntax_emerges_L21|'
       'syntax_emerges_L21|content_emerges_L20|'
       'content_emerges_L20')
assert res23['verdict'] == V23
assert res23['smoke'] is False
z18 = np.load(os.path.join(D18,
                           'traj_readout.npz'),
              allow_pickle=False)
auc18 = z18['auc_curve']
assert len(auc18) == N_NEW + 1
assert abs(float(auc18[0])
           - 0.9809094210600907) < 1e-12
z20 = np.load(os.path.join(D20,
                           'p118_readout.npz'),
              allow_pickle=False)
annP_c = z20['annP_c']
annA_c = z20['annA_c']
assert annP_c.shape == (NP, N_NEW)
z22 = np.load(os.path.join(D22,
                           'p120_readout.npz'),
              allow_pickle=False)
assert z22['wrec_pd_P'].shape == (36, 672, 13)
z23 = np.load(os.path.join(D23,
                           'p121_readout.npz'),
              allow_pickle=False)
assert z23['auc_sim_dir'].shape == (N_NEW + 1,)

# ================================================================
# design seal (frozen BEFORE observation)
# ================================================================
seal = {
    'phase': 3124,
    'name': NAME,
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'np_d': NP_D,
    'data_sources': {
        'margins_and_tokens':
            'phase3118 traj_readout.npz '
            '(gt_cleanseq_clean, auc_curve)',
        'annotation': 'phase3120 annP_c/annA_c',
        'write_spectra_3122': 'wrec_pd/wrec_pn '
                              '(36,672,13)',
        'fits_3123': 'phase3123 result.json '
                     'part_a.dirfit (cross-check '
                     'only; refit here)',
        'spec_3123': 'phase3123 p121_readout.npz '
                     'spec_dn/spec_rel',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz'},
    'part_a': {
        'refit': 'identical to 3123: content '
                 'table -> dm_res -> lstsq '
                 '[m,1] -> S,MS; assert vs 3123 '
                 'result (1e-9)',
        'decomp': 'resid (bootstrap pool) 3-way '
                  'sequential: (1) 12 step-dummy '
                  'block -> common-mode share; '
                  '(2) x=(m-MS) block on '
                  'remainder -> m-structure '
                  'share; (3) remainder share',
        'cm_gate': 'min(cm_share_P, cm_share_A1) '
                   '>= 0.2 -> '
                   'common_mode_dominant; '
                   '>= 0.05 -> '
                   'common_mode_partial; else '
                   'iid_like',
        'det_sim': 'deterministic no-noise: '
                   'm(t+1)=m(t)+S_dir*(m(t)-'
                   'MS_dir)+content[cls,k], '
                   'init m0 actual; auc_det vs '
                   'auc18: r>=0.7 '
                   'det_core_sufficient; >=0.5 '
                   'partial; else failed',
        'r2_gate': 'median per-trajectory R2 '
                   '(det sim, both steps 0..12) '
                   '>= 0.5 -> r2_ok else r2_weak',
        'analytic': 'iid plateau Phi(gap/sqrt('
                    'sigP^2+sigA1^2)), sig=resid_'
                    'std/sqrt(1-(1+S)^2); sanity '
                    '|phi - auc_sim_dir3123[12]| '
                    '< 0.03'},
    'part_b': {
        'lag1': 'mean_i corr(a_full[i,:-1], '
                'a_full[i,1:]); mean of dirs '
                '>= 0.05 lag1_positive; '
                '<= -0.05 lag1_negative; else '
                'lag1_near_zero',
        'kalman': 'z_t = dm_res - S*m_t = '
                  '-S*a_t + e; a random walk '
                  'q; prior N(MS, sig_a2); sig_e2 '
                  '= static resid_std^2; q grid '
                  '= sig_a2*logspace(-4,1,22); '
                  'ratio=q_hat/sig_a2: max over '
                  'dirs > 0.1 anchor_drifts; '
                  'max < 0.02 anchor_static; '
                  'else anchor_mixed',
        'rts': 'RTS smoother at q_hat -> '
               'a_s[i,t] (672,12)',
        'own_sim': 'deterministic with own '
                   'smoothed anchors: '
                   'm(t+1)=m(t)+S*(m(t)-a_s[:,t])'
                   '+content; r>=0.7 '
                   'own_anchor_sufficient; '
                   '>=0.5 own_anchor_partial; '
                   'else own_anchor_insufficient '
                   '(DIAGNOSTIC, in-sample)'},
    'part_c': {
        'l35rel': 'wn=w/pn elementwise (|pn|>'
                  '1e-9); first-order split of '
                  'W_oth-W_ans = Pn_oth*'
                  '(Wn_oth-Wn_ans) [norm term] '
                  '+ Wn_ans*(Pn_oth-Pn_ans) '
                  '[sem term] + leak; '
                  'norm_share_A1 > 0.5 -> '
                  'norm_gain_dominant else '
                  'semantic_component_present',
        'spec_readout': 'rows L30/L32/L35 of '
                        '3123 spec_dn/spec_rel '
                        '(36,7) x 2 dirs -> '
                        'result (descriptive)'},
    'part_d': {
        'model': 'glm4-9b-chat-hf (GlmForCausal'
                 'LM, 40 layers, BF16, eager, '
                 'batch1 forwards)',
        'readout': 'w_dn = WU[yes]-WU[no], '
                   'yes/no = first ids of " yes"'
                   ' / " no"; logit lens via '
                   'model.model.norm on every '
                   'hidden state 0..40; margins '
                   'at pos0+k, k=0..12',
        'gen': 'greedy 12 new tokens, batch '
               '32, left padding; decode -> '
               're-tokenize (add_special='
               'False) -> trim/pad to 12 with '
               '"."',
        'span': 'first occurrence of query '
                'line "The {s} {qrel} the {o}."'
                ' in decoded gen; token span '
                'via offset mapping; k1>=1, '
                '2<=len<=12; s1 = distractor-'
                'entity line (dot-padded to '
                'span len), s2 = within-span '
                'shuffle (crc32 seed), s3 = '
                'dots',
        'gates': {
            'D_path': 'corr(manual-norm margin, '
                      'model-logits margin) > '
                      '0.9999 on first NP_CHECK '
                      'pairs -> path_valid else '
                      'FATAL',
            'D_repro': 's0 forward twice, max '
                       'diff == 0 on first '
                       'NP_REPRO pairs -> '
                       'replay_bit_exact; <1e-6 '
                       'replay_ok; else FATAL',
            'D_readout': 'final-step margin AUC '
                         '(P vs A1) > 0.6 -> '
                         'readout_ok else '
                         'readout_warn',
            'D_spans': 'n_span >= 300 BOTH dirs '
                       '-> spans_ok; >= 100 both '
                       '-> spans_sparse; else '
                       'spans_absent (E gates '
                       'skipped_sparse)',
            'D_lstar_rel': 'L* = min L>=20 with '
                           'E(L) <= -0.3*|E(final)|'
                           ' -> glm_syntax_L{L*} / '
                           'glm_below_threshold '
                           '(per dir per effect); '
                           'absolute-gate L* '
                           '(qwen th -0.05/-0.025) '
                           'also reported'},
        'no_mc': 'no Monte-Carlo in this phase; '
                 'only crc32-seeded shuffles '
                 '(deterministic)'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('seal frozen (%s)' % seal['created'])

# ================================================================
# shared frozen arrays + refit (assert vs 3123)
# ================================================================
mP18 = z18['gt_cleanseq_clean__P']
mA18 = z18['gt_cleanseq_clean__A1']
MSEQ = {'P': mP18[:NP].astype(np.float64),
        'A1': mA18[:NP].astype(np.float64)}
ANN = {'P': annP_c[:NP], 'A1': annA_c[:NP]}


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


# ================================================================
# PART A: refit + 3-way decomp + det sim + analytic
# ================================================================
log('== PART A: refit + decomp + det sim ==')
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

fit = {}
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
    resid = (y - X @ bb).reshape(NP, N_NEW)
    a_full = m[:, :-1] - dm_res / S_dc
    a_i = a_full.mean(1)
    fit[dc] = {'S': S_dc, 'MS': MS_dc,
               'resid': resid,
               'a_full': a_full, 'a': a_i,
               'col': col, 'dm_res': dm_res}
    r23 = res23['part_a']['dirfit'][dc]
    assert abs(S_dc - r23['S']) < 1e-9, dc
    assert abs(MS_dc - r23['MS']) < 1e-9, dc
    a23 = z23['anchors_%s' % dc].astype(
        np.float64)
    assert np.max(np.abs(a_i - a23)) < 1e-5, dc
log('A refit assert vs 3123 ok')

decomp = {}
for dc in ('P', 'A1'):
    R = fit[dc]['resid'].ravel()
    ss_tot = float((R * R).sum())
    NIK = R.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, R, rcond=None)
    r1 = R - Z1 @ b1
    ss_step = ss_tot - float((r1 * r1).sum())
    x = (MSEQ[dc][:, :-1]
         - fit[dc]['MS']).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1, rcond=None)
    r2 = r1 - Z2 @ b2
    ss_m = float((r1 * r1).sum()) \
        - float((r2 * r2).sum())
    ss_rem = float((r2 * r2).sum())
    Rm = fit[dc]['resid']
    s_k = Rm.std(axis=0)
    decomp[dc] = {
        'ss_tot': ss_tot,
        'cm_share': ss_step / ss_tot,
        'm_share': ss_m / ss_tot,
        'rem_share': ss_rem / ss_tot,
        's_k': [float(v) for v in s_k],
        'leak': 1.0 - ss_step / ss_tot
        - ss_m / ss_tot - ss_rem / ss_tot}
    log('A decomp %s: cm=%.4f m=%.4f rem=%.4f'
        % (dc, decomp[dc]['cm_share'],
           decomp[dc]['m_share'],
           decomp[dc]['rem_share']))
cm_min = min(decomp['P']['cm_share'],
             decomp['A1']['cm_share'])
a_cm_v = ('common_mode_dominant' if cm_min >= 0.2
          else ('common_mode_partial'
                if cm_min >= 0.05
                else 'iid_like'))
log('A-CM: min share %.4f -> %s'
    % (cm_min, a_cm_v))

auc_det = np.zeros(N_NEW + 1)
det_arr = {}
r2_med = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    col = fit[dc]['col']
    S_dc = fit[dc]['S']
    MS_dc = fit[dc]['MS']
    mh = m[:, 0].copy()
    snap = [mh.copy()]
    for k in range(N_NEW):
        mh = mh + S_dc * (mh - MS_dc) \
            + content[dc][col[:, k], k]
        snap.append(mh.copy())
    SArr = np.stack(snap, 1)
    det_arr[dc] = SArr
    denom = ((m - m.mean(1, keepdims=True))
             ** 2).sum(1)
    r2i = 1.0 - ((SArr - m) ** 2).sum(1) \
        / np.maximum(denom, 1e-12)
    r2_med[dc] = float(np.median(r2i))
for t in range(N_NEW + 1):
    auc_det[t] = auc_mw(det_arr['P'][:, t],
                        det_arr['A1'][:, t])
r_det = float(np.corrcoef(auc_det, auc18)[0, 1])
a_det_v = ('det_core_sufficient' if r_det >= 0.7
           else ('det_core_partial'
                 if r_det >= 0.5
                 else 'det_core_failed'))
r2_min = min(r2_med['P'], r2_med['A1'])
a_r2_v = 'r2_ok' if r2_min >= 0.5 else 'r2_weak'
log('A-DET: r=%.4f -> %s; R2med P=%.4f A1=%.4f '
    '-> %s' % (r_det, a_det_v, r2_med['P'],
               r2_med['A1'], a_r2_v))

sigP = (float(np.std(fit['P']['resid']))
        / math.sqrt(1.0 - (1.0
                           + fit['P']['S'])
                    ** 2))
sigA1 = (float(np.std(fit['A1']['resid']))
         / math.sqrt(1.0 - (1.0
                            + fit['A1']['S'])
                     ** 2))
gap = (float(fit['P']['a'].mean())
       - float(fit['A1']['a'].mean()))
phi = 0.5 * (1.0 + math.erf(
    (gap / math.sqrt(sigP ** 2 + sigA1 ** 2))
    / math.sqrt(2.0)))
phi_3123 = float(z23['auc_sim_dir'][12])
assert abs(phi - phi_3123) < 0.03, \
    'analytic plateau mismatch'
log('A-ANALYTIC: phi=%.4f vs 3123 plateau %.4f '
    'ok' % (phi, phi_3123))

# ================================================================
# PART B: lag1 + Kalman q MLE + RTS + own sim
# ================================================================
log('== PART B: lag1 + kalman + own sim ==')
lag1 = {}
kal = {}
a_s_store = {}
for dc in ('P', 'A1'):
    af = fit[dc]['a_full']
    afc = af - af.mean(1, keepdims=True)
    num = (afc[:, :-1] * afc[:, 1:]).sum(1)
    den = np.sqrt((afc[:, :-1] ** 2).sum(1)
                  * (afc[:, 1:] ** 2).sum(1))
    lag1[dc] = float((num
                      / np.maximum(den, 1e-12))
                     .mean())
    m = MSEQ[dc]
    S_dc = fit[dc]['S']
    MS_dc = fit[dc]['MS']
    dm_res = fit[dc]['dm_res']
    sig_e2 = float(np.std(
        fit[dc]['resid'])) ** 2
    sig_a2 = float(fit[dc]['a'].var())
    z = dm_res - S_dc * m[:, :-1]
    q_grid = sig_a2 * np.logspace(-4, 1, 22)
    lls = np.zeros(len(q_grid))
    for qi, q in enumerate(q_grid):
        a = np.full(NP, MS_dc)
        P = np.full(NP, sig_a2)
        ll = 0.0
        for t in range(N_NEW):
            Pm = P + q
            v = S_dc * S_dc * Pm + sig_e2
            innov = z[:, t] + S_dc * a
            ll += -0.5 * float(
                (np.log(2.0 * math.pi * v)
                 + innov * innov / v).sum())
            K = -S_dc * Pm / v
            a = a + K * innov
            P = (1.0 - S_dc * S_dc * Pm / v) \
                * Pm
        lls[qi] = ll
    q_hat = float(q_grid[int(np.argmax(lls))])
    ratio = q_hat / sig_a2
    # RTS smoother at q_hat
    a = np.full(NP, MS_dc)
    P = np.full(NP, sig_a2)
    af_store = np.zeros((NP, N_NEW))
    Pf_store = np.zeros((NP, N_NEW))
    for t in range(N_NEW):
        Pm = P + q_hat
        v = S_dc * S_dc * Pm + sig_e2
        innov = z[:, t] + S_dc * a
        K = -S_dc * Pm / v
        a = a + K * innov
        P = (1.0 - S_dc * S_dc * Pm / v) * Pm
        af_store[:, t] = a
        Pf_store[:, t] = P
    a_s = np.zeros((NP, N_NEW))
    a_s[:, N_NEW - 1] = af_store[:, N_NEW - 1]
    for t in range(N_NEW - 2, -1, -1):
        Pp_next = Pf_store[:, t] + q_hat
        C = Pf_store[:, t] \
            / np.maximum(Pp_next, 1e-12)
        a_s[:, t] = af_store[:, t] + C * (
            a_s[:, t + 1] - af_store[:, t])
    kal[dc] = {'q_hat': q_hat, 'ratio': ratio,
               'sig_a2': sig_a2,
               'sig_e2': sig_e2,
               'll_grid': [float(v)
                           for v in lls],
               'q_grid': [float(v)
                          for v in q_grid]}
    a_s_store[dc] = a_s
    log('B kalman %s: q_hat=%.6g ratio=%.4f '
        'sig_a2=%.4f' % (dc, q_hat, ratio,
                         sig_a2))
lag1_mean = 0.5 * (lag1['P'] + lag1['A1'])
b_lag1_v = ('lag1_positive' if lag1_mean >= 0.05
            else ('lag1_negative'
                  if lag1_mean <= -0.05
                  else 'lag1_near_zero'))
ratio_max = max(kal['P']['ratio'],
                kal['A1']['ratio'])
b_kalman_v = ('anchor_drifts' if ratio_max > 0.1
              else ('anchor_static'
                    if ratio_max < 0.02
                    else 'anchor_mixed'))
log('B-LAG1: P=%.4f A1=%.4f -> %s'
    % (lag1['P'], lag1['A1'], b_lag1_v))
log('B-KALMAN: ratio P=%.4f A1=%.4f -> %s'
    % (kal['P']['ratio'], kal['A1']['ratio'],
       b_kalman_v))

auc_own = np.zeros(N_NEW + 1)
own_arr = {}
r2_own_med = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    col = fit[dc]['col']
    S_dc = fit[dc]['S']
    a_s = a_s_store[dc]
    mh = m[:, 0].copy()
    snap = [mh.copy()]
    for k in range(N_NEW):
        mh = mh + S_dc * (mh - a_s[:, k]) \
            + content[dc][col[:, k], k]
        snap.append(mh.copy())
    SArr = np.stack(snap, 1)
    own_arr[dc] = SArr
    denom = ((m - m.mean(1, keepdims=True))
             ** 2).sum(1)
    r2i = 1.0 - ((SArr - m) ** 2).sum(1) \
        / np.maximum(denom, 1e-12)
    r2_own_med[dc] = float(np.median(r2i))
for t in range(N_NEW + 1):
    auc_own[t] = auc_mw(own_arr['P'][:, t],
                        own_arr['A1'][:, t])
r_own = float(np.corrcoef(auc_own, auc18)[0, 1])
b_owns_v = ('own_anchor_sufficient'
            if r_own >= 0.7
            else ('own_anchor_partial'
                  if r_own >= 0.5
                  else 'own_anchor_insufficient'))
log('B-OWNS: r=%.4f -> %s; R2med P=%.4f A1=%.4f'
    % (r_own, b_owns_v, r2_own_med['P'],
       r2_own_med['A1']))

# ================================================================
# PART C: L35 release decomposition + spec readout
# ================================================================
log('== PART C: L35 release + spec ==')
l35rel = {}
for dc in ('P', 'A1'):
    w = z22['wrec_pd_%s' % dc].astype(
        np.float64)[35][:, :N_NEW]
    p = z22['wrec_pn_%s' % dc].astype(
        np.float64)[35][:, :N_NEW]
    ok = np.abs(p) > 1e-9
    wn = np.where(ok, w / np.where(ok, p, 1.0),
                  np.nan)
    ann = ANN[dc]
    am = (ann == 9) & ok
    om = (~am) & ok
    W_ans = float(np.nanmean(w[am]))
    W_oth = float(np.nanmean(w[om]))
    Pn_ans = float(np.nanmean(p[am]))
    Pn_oth = float(np.nanmean(p[om]))
    Wn_ans = float(np.nanmean(wn[am]))
    Wn_oth = float(np.nanmean(wn[om]))
    t_norm = Pn_oth * (Wn_oth - Wn_ans)
    t_sem = Wn_ans * (Pn_oth - Pn_ans)
    leak = (W_oth - W_ans) - t_norm - t_sem
    denom2 = abs(t_norm) + abs(t_sem)
    l35rel[dc] = {
        'W_ans': W_ans, 'W_oth': W_oth,
        'Pn_ans': Pn_ans, 'Pn_oth': Pn_oth,
        'Wn_ans': Wn_ans, 'Wn_oth': Wn_oth,
        't_norm': t_norm, 't_sem': t_sem,
        'leak': leak,
        'norm_share': (abs(t_norm) / denom2
                       if denom2 > 0 else None)}
    log('C-L35 %s: raw dW=%.4f t_norm=%.4f '
        't_sem=%.4f leak=%.4f share=%.4f'
        % (dc, W_oth - W_ans, t_norm, t_sem,
           leak, l35rel[dc]['norm_share']))
share_A1 = l35rel['A1']['norm_share']
c_l35rel_v = ('norm_gain_dominant'
              if share_A1 > 0.5
              else 'semantic_component_present')
log('C-L35REL: A1 norm share %.4f -> %s'
    % (share_A1, c_l35rel_v))

CLS_LIST = [0, 1, 2, 3, 4, 9]
spec_rows = {}
for dc in ('P', 'A1'):
    tab = z23['spec_dn_%s' % dc].astype(
        np.float64)
    rtab = z23['spec_rel_%s' % dc].astype(
        np.float64)
    spec_rows[dc] = {}
    for LL in (30, 32, 35):
        spec_rows[dc]['L%d_dn' % LL] = [
            float(tab[LL, ci])
            for ci in range(7)]
        spec_rows[dc]['L%d_rel' % LL] = [
            float(rtab[LL, ci])
            for ci in range(7)]
log('C spec rows extracted (descriptive)')

# ================================================================
# PART D: GLM4 GPU
# ================================================================
log('== PART D: glm4 ==')
p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']


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


hP = {}
hA1 = {}
for i in range(NB):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
    elif str(condB[i]) == 'A1':
        hA1[pk] = i
pks = sorted(hP.keys())
assert len(pks) == NP
NP_C = min(NP_D, NP)
texts = {}
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
    texts[i] = build_prompt(mat5, s, o, lrel,
                            qrel)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

tokG = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
modelG = AutoModelForCausalLM.from_pretrained(
    MDIR_G, torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    trust_remote_code=True).to('cuda').eval()
NLG = len(modelG.model.layers)
log('glm4 loaded NLG=%d' % NLG)
WUG = modelG.lm_head.weight.detach()
YES_G = int(tokG(' yes', add_special_tokens=False)
            ['input_ids'][0])
NO_G = int(tokG(' no', add_special_tokens=False)
           ['input_ids'][0])
w_dnG = (WUG[YES_G] - WUG[NO_G]) \
    .float().cpu().numpy()
DOT_G = int(tokG('.', add_special_tokens=False)
            ['input_ids'][0])
log('readout ready: yes=%d no=%d dot=%d '
    '|w_dn|=%.4f' % (YES_G, NO_G, DOT_G,
                     float(np.linalg.norm(
                         w_dnG))))
normG = modelG.model.norm
PADG = tokG.pad_token_id
if PADG is None:
    PADG = modelG.config.pad_token_id
tokG.padding_side = 'left'

GEN_BATCH = 8 if SMOKE else 32
gen_ids_store = {}
for dc, hmap in (('P', hP), ('A1', hA1)):
    plist = [texts[hmap[p]] for p in pks][:NP_C]
    store = []
    for b0 in range(0, len(plist), GEN_BATCH):
        chunk = plist[b0:b0 + GEN_BATCH]
        enc = tokG(chunk, return_tensors='pt',
                   padding=True)
        enc = {k: v.to('cuda')
               for k, v in enc.items()}
        with torch.inference_mode():
            outg = modelG.generate(
                **enc, max_new_tokens=N_NEW,
                do_sample=False, num_beams=1,
                pad_token_id=PADG)
        newg = outg[:, enc['input_ids'].shape[1]:]
        for row in newg:
            ids_r = [int(t) for t in row]
            if PADG in ids_r:
                ids_r = ids_r[:ids_r.index(PADG)]
            for e in modelG.config.eos_token_id \
                    if isinstance(
                        modelG.config.eos_token_id,
                        list) else [
                modelG.config.eos_token_id]:
                if e in ids_r:
                    ids_r = ids_r[:ids_r.index(e)]
                    break
            store.append(ids_r)
        if (b0 // GEN_BATCH) % 8 == 0:
            log('D gen %s %d/%d (%.1fs)'
                % (dc, b0 + len(chunk), len(plist),
                   time.time() - T0))
    gen_ids_store[dc] = store
ftok = {}
for dc in ('P', 'A1'):
    cnt = {}
    for ids_r in gen_ids_store[dc]:
        if ids_r:
            t0g = ids_r[0]
            cnt[t0g] = cnt.get(t0g, 0) + 1
    top = sorted(cnt.items(),
                 key=lambda kv: -kv[1])[:8]
    ftok[dc] = [(int(t), n, tokG.decode([t]))
                for (t, n) in top]
    log('D first-tok %s: %s' % (dc, ftok[dc]))


def traj_tokens(dc, j):
    # GLM4 answers Yes/No directly and never
    # echoes the query line (SMOKE n_span=0
    # on the generated stream). Interference
    # is therefore injected into the PROMPT
    # copy of the query line (equal-length
    # replacement, pos0 unchanged) while the
    # generated stream is replayed unchanged.
    ids_r = gen_ids_store[dc][j]
    ptxt = texts[hmap[pks[j]]]
    encp = tokG(ptxt, add_special_tokens=False,
                return_offsets_mapping=True)
    poffs = [tuple(v) for v in
             encp['offset_mapping']]
    ids2 = list(ids_r)[:N_NEW]
    while len(ids2) < N_NEW:
        ids2.append(DOT_G)
    return ptxt, poffs, ids2


def find_span_g(dec, offs, s, o, qrel):
    qline = 'The %s %s the %s.' % (
        ents_all[s], PREDS_all[qrel],
        ents_all[o])
    cs = dec.find(qline)
    ce_want = cs + len(qline)
    while cs != -1:
        idxs = [k for k in range(len(offs))
                if offs[k][0] >= cs
                and offs[k][1] <= ce_want
                and offs[k][1] > offs[k][0]]
        if len(idxs) >= 2 \
                and idxs[0] >= 1 \
                and idxs[-1] - idxs[0] + 1 <= N_NEW:
            return idxs[0], idxs[-1]
        cs = dec.find(qline, cs + 1)
        ce_want = cs + len(qline)
    return None


def line_tokens_g(s2, o2, lrel):
    txt = ' The %s %s the %s.' % (
        ents_all[s2], PREDS_all[lrel],
        ents_all[o2])
    return list(tokG(txt,
                     add_special_tokens=False)
                ['input_ids'])


def context_triples(pk, dcode):
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    lrel = r if dcode == 'P' else ri1
    D = [tuple(d) for d in
         mat5['distractors']['%d_%d' % (s, o)]]
    ctx = set((a, rr, b) for (a, rr, b) in
              ([(s, lrel, o)] + list(D)))
    return s, o, lrel, ctx


def pick_replacement_g(pk, dcode):
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
    raise RuntimeError('no replacement')


def forward_trackG(prompt_ids, traj_ids,
                   want_logits=False):
    # GLM4 hidden_states trap (probe-verified):
    # hs = [emb, L1..L39 out, final-norm(h)]
    # -- L40 pre-norm output missing; hs[-1]
    # is post-norm. Use hooks to collect
    # emb + all L1..L40 pre-norm outputs,
    # same readout semantics as Qwen 3123.
    traj_ids = [int(x) for x in traj_ids]
    ids = list(prompt_ids) + traj_ids
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(traj_ids) + 1
    ml = np.zeros((NLG + 1, npts),
                  dtype=np.float64)
    lm = None
    feats = []
    hooks = []

    def _mk():
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) \
                else out
            feats.append(o.detach())
        return hook

    for lyr in modelG.model.layers:
        hooks.append(
            lyr.register_forward_hook(_mk()))
    with torch.inference_mode():
        out = modelG(t_in,
                     output_hidden_states=False,
                     use_cache=False)
        for hk in hooks:
            hk.remove()
        seq = [modelG.model.embed_tokens(
            t_in)] + feats
        for L in range(NLG + 1):
            h = normG(seq[L][0]).float() \
                .cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dnG)
        if want_logits:
            lg = out.logits[0].float().cpu() \
                .numpy()
            lm = np.array([
                float(lg[pos0 + k][YES_G])
                - float(lg[pos0 + k][NO_G])
                for k in range(npts)])
        del out, seq, feats
    return ml, lm


PID_G = {}
for dc, hmap in (('P', hP), ('A1', hA1)):
    PID_G[dc] = [
        tokG(texts[hmap[p]],
             add_special_tokens=False)
        ['input_ids'] for p in pks[:NP_C]]

# path-valid + s0 repro gates (subset)
path_rs = []
path_d = []
drep = 0.0
SCOND = ('s0', 's1', 's2', 's3')
MLG = {d: {c: None for c in SCOND}
       for d in ('P', 'A1')}
t0d = time.time()
span_count = {}
for dcode in ('P', 'A1'):
    store = {c: [] for c in SCOND}
    n_span = 0
    n_done = 0
    for j in range(NP_C):
        pk = pks[j]
        prompt_ids0 = list(PID_G[dcode][j])
        ptxt, poffs, base = traj_tokens(
            dcode, j)
        (s, o) = (int(v) for v in
                  pk.split('_'))
        r = p2r[pk]
        span = find_span_g(ptxt, poffs,
                           s, o, r)
        pids = {c: list(prompt_ids0)
                for c in SCOND}
        if span is not None:
            (k1, k2) = span
            Lspan = k2 - k1 + 1
            (s2, o2) = pick_replacement_g(
                pk, dcode)
            (_, _, lrel, _) = context_triples(
                pk, dcode)
            sub = line_tokens_g(s2, o2, lrel)
            n_pad = max(0, Lspan - len(sub))
            sub = sub[:Lspan]
            sub = sub + [DOT_G] * n_pad
            rng2 = _rnd.Random(zlib.crc32(
                ('s2|%s|%s' % (pk, dcode))
                .encode('ascii')))
            shuf = list(sub)
            rng2.shuffle(shuf)
            pids['s1'][k1:k2 + 1] = sub
            pids['s2'][k1:k2 + 1] = shuf
            pids['s3'][k1:k2 + 1] = \
                [DOT_G] * Lspan
            n_span += 1
        for c in SCOND:
            wl, lm = forward_trackG(
                pids[c], base,
                want_logits=(j < NP_CHECK
                             and c == 's0'))
            store[c].append(wl)
            if lm is not None:
                path_rs.append(lm)
                path_d.append(
                    wl[NLG, :].copy())
        if j < NP_REPRO:
            wl2, _ = forward_trackG(
                pids['s0'], base)
            drep = max(drep, float(np.abs(
                store['s0'][j] - wl2).max()))
        n_done += 1
        if n_done % 128 == 0:
            log('D %s %d/%d (%.1fs)'
                % (dcode, n_done, NP_C,
                   time.time() - t0d))
    for c in SCOND:
        MLG[dcode][c] = np.array(
            store[c], dtype=np.float64)
    log('D %s done n_span=%d (%.1fs)'
        % (dcode, n_span, time.time() - t0d))
    span_count[dcode] = n_span

n_span_g = span_count
pr = np.concatenate(path_rs)
pd = np.concatenate(path_d)
if np.std(pd) > 0:
    r_path = float(np.corrcoef(pr, pd)[0, 1])
else:
    r_path = 0.0
d_path_v = 'path_valid' if r_path > 0.9999 \
    else 'path_diverged'
assert d_path_v == 'path_valid', \
    'logit-lens path diverged - FATAL'
if drep == 0.0:
    d_repro_v = 'replay_bit_exact'
elif drep < 1e-6:
    d_repro_v = 'replay_ok'
else:
    d_repro_v = 'replay_diverged'
assert d_repro_v != 'replay_diverged', \
    's0 repro diverged - FATAL'
log('D-PATH: r=%.6f -> %s; D-REPRO: %.3e -> %s'
    % (r_path, d_path_v, drep, d_repro_v))

auc_fin = auc_mw(
    MLG['P']['s0'][:, NLG, -1],
    MLG['A1']['s0'][:, NLG, -1])
d_readout_v = 'readout_ok' if auc_fin > 0.6 \
    else 'readout_warn'
log('D-READOUT: final-step AUC=%.4f -> %s'
    % (auc_fin, d_readout_v))
if SMOKE:
    s_ok = max(1, NP_C // 2)
    s_sp = s_ok
else:
    s_ok = 300
    s_sp = 100
if n_span_g['P'] >= s_ok \
        and n_span_g['A1'] >= s_ok:
    d_spans_v = 'spans_ok'
elif n_span_g['P'] >= s_sp \
        and n_span_g['A1'] >= s_sp:
    d_spans_v = 'spans_sparse'
else:
    d_spans_v = 'spans_absent'
log('D-SPANS: P=%d A1=%d -> %s'
    % (n_span_g['P'], n_span_g['A1'],
       d_spans_v))

curves_g = {}
lstar_g = {}
if d_spans_v != 'spans_absent':
    for dcode in ('P', 'A1'):
        syn = np.zeros(NLG + 1)
        con = np.zeros(NLG + 1)
        n_sp = 0
        for j in range(NP_C):
            pk = pks[j]
            (s, o) = (int(v) for v in
                      pk.split('_'))
            ptxt, poffs, _base = traj_tokens(
                dcode, j)
            span = find_span_g(
                ptxt, poffs, s, o, p2r[pk])
            if span is None:
                continue
            n_sp += 1
            s0 = MLG[dcode]['s0'][j]
            d1 = MLG[dcode]['s1'][j] - s0
            d2 = MLG[dcode]['s2'][j] - s0
            d3 = MLG[dcode]['s3'][j] - s0
            for L in range(NLG + 1):
                syn[L] += float(
                    d1[L].mean()
                    - d2[L].mean())
                con[L] += float(
                    d1[L].mean()
                    - d3[L].mean())
        syn /= max(n_sp, 1)
        con /= max(n_sp, 1)
        curves_g[dcode] = {'syn': syn,
                           'cont': con,
                           'n_span': n_sp}
        lstar_g[dcode] = {}
        for nm, cv in (('syn', syn),
                       ('cont', con)):
            eff = abs(float(cv[NLG]))
            lr = None
            la = None
            for L in range(20, NLG + 1):
                if lr is None and eff > 0 \
                        and cv[L] <= -0.3 * eff:
                    lr = L
                th = 0.05 if dcode == 'P' \
                    else 0.025
                if la is None \
                        and cv[L] <= -th:
                    la = L
            lstar_g[dcode][nm] = {
                'rel': lr, 'abs': la}
        log('D-TRACE %s: n_span=%d syn_rel=%s '
            'cont_rel=%s' % (
                dcode, n_sp,
                lstar_g[dcode]['syn']['rel'],
                lstar_g[dcode]['cont']['rel']))
else:
    curves_g = {'P': {'n_span': 0},
                'A1': {'n_span': 0}}
    lstar_g = {'P': {}, 'A1': {}}


def lstar_v(dc, nm):
    if d_spans_v == 'spans_absent':
        return 'skipped_sparse'
    e = lstar_g.get(dc, {}).get(nm, {})
    lr = e.get('rel')
    tag = 'syntax' if nm == 'syn' else 'content'
    if lr is None:
        return 'glm_%s_below_threshold' % tag
    return 'glm_%s_L%d' % (tag, lr)


d_syn_P_v = lstar_v('P', 'syn')
d_syn_A1_v = lstar_v('A1', 'syn')
d_cont_P_v = lstar_v('P', 'cont')
d_cont_A1_v = lstar_v('A1', 'cont')

# ================================================================
# verdict + save
# ================================================================
verdict = '|'.join([
    a_cm_v, a_det_v, a_r2_v, b_lag1_v,
    b_kalman_v, b_owns_v, c_l35rel_v,
    d_path_v, d_repro_v, d_readout_v,
    d_spans_v, d_syn_P_v, d_syn_A1_v,
    d_cont_P_v, d_cont_A1_v])
runtime = round(time.time() - T0, 1)
results = {
    'phase': 3124,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'n_pairs': NP,
    'np_d': NP_D,
    'runtime_s': runtime,
    'part_a': {
        'refit': {
            'P': {'S': fit['P']['S'],
                  'MS': fit['P']['MS']},
            'A1': {'S': fit['A1']['S'],
                   'MS': fit['A1']['MS']}},
        'decomp': decomp,
        'cm_min': cm_min,
        'cm_verdict': a_cm_v,
        'det_sim': {
            'r_det': r_det,
            'verdict': a_det_v,
            'r2_median': r2_med,
            'r2_verdict': a_r2_v,
            'auc_det': [float(x)
                        for x in auc_det]},
        'analytic': {
            'sig_P': sigP, 'sig_A1': sigA1,
            'gap': gap, 'phi': phi,
            'phi_3123_plateau': phi_3123}},
    'part_b': {
        'lag1': lag1,
        'lag1_verdict': b_lag1_v,
        'kalman': {dc: {'q_hat': kal[dc]
                        ['q_hat'],
                        'ratio': kal[dc]
                        ['ratio'],
                        'sig_a2': kal[dc]
                        ['sig_a2']}
                   for dc in ('P', 'A1')},
        'kalman_verdict': b_kalman_v,
        'own_sim': {
            'r_own': r_own,
            'verdict': b_owns_v,
            'r2_median': r2_own_med,
            'auc_own': [float(x)
                        for x in auc_own]}},
    'part_c': {
        'l35rel': l35rel,
        'verdict': c_l35rel_v,
        'spec_rows': spec_rows},
    'part_d': {
        'interference':
            'input_prompt_span_equal_len',
        'interference_note': (
            'GLM4 answers Yes/No directly '
            'without echoing the query line, '
            'so s1/s2/s3 replace the prompt '
            'copy of the query line '
            '(equal-length, pos0 preserved) '
            'and the generated stream is '
            'replayed unchanged'),
        'ids': {'yes': YES_G, 'no': NO_G,
                'dot': DOT_G, 'n_layers': NLG},
        'first_tokens': {dc: [[t, n, txt]
                              for (t, n, txt)
                              in ftok[dc]]
                         for dc in ('P', 'A1')},
        'path': {'r': r_path,
                 'verdict': d_path_v},
        'repro': {'max_diff': drep,
                  'verdict': d_repro_v},
        'readout_auc': auc_fin,
        'readout_verdict': d_readout_v,
        'n_span': n_span_g,
        'spans_verdict': d_spans_v,
        'lstar': {d: {nm: lstar_g[d].get(
            nm, {'rel': None, 'abs': None})
            for nm in ('syn', 'cont')}
            for d in ('P', 'A1')},
        'curves': {
            'E_syn_P': [float(x) for x in
                        curves_g['P']['syn']]
            if 'syn' in curves_g['P'] else [],
            'E_syn_A1': [float(x) for x in
                         curves_g['A1']['syn']]
            if 'syn' in curves_g['A1'] else [],
            'E_cont_P': [float(x) for x in
                         curves_g['P']['cont']]
            if 'cont' in curves_g['P'] else [],
            'E_cont_A1': [float(x) for x in
                          curves_g['A1']
                          ['cont']]
            if 'cont' in curves_g['A1']
            else []}},
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False,
              indent=1)
npz_out = {
    'auc_det': auc_det,
    'auc_own': auc_own,
    'anchors_P': fit['P']['a'].astype(np.float32),
    'anchors_A1': fit['A1']['a'].astype(
        np.float32),
    'a_s_P': a_s_store['P'].astype(np.float32),
    'a_s_A1': a_s_store['A1'].astype(np.float32),
}
for dc in ('P', 'A1'):
    npz_out['rts_%s' % dc] = \
        a_s_store[dc].astype(np.float32)
    for c in SCOND:
        npz_out['mlg_%s_%s' % (c, dc)] = \
            MLG[dc][c].astype(np.float32)
    if 'syn' in curves_g[dc]:
        npz_out['E_syn_%s' % dc] = \
            curves_g[dc]['syn'].astype(np.float32)
        npz_out['E_cont_%s' % dc] = \
            curves_g[dc]['cont'].astype(
                np.float32)
np.savez(os.path.join(OUT, 'p122_readout.npz'),
         **npz_out)
log('verdict=%s' % verdict)
log('done (%.1fs)' % runtime)
print('phase3124 done: %s' % verdict)
