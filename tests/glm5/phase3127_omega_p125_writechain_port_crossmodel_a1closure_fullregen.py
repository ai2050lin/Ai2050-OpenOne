"""Phase 3127 (Omega-P125): write-chain PORT
cross-model functional test (GPU, qwen3-4b then
glm4-9b-chat-hf, sequential) + A1 long-range
closure + cross-trajectory second-order /
common-mode / sparse-event spectra (offline) +
full-672 counterfactual regeneration (GLM4).

Inputs (frozen): phase3118 traj npz, phase3120
ann npz, phase3122 p120 npz, phase3124 result +
p122_readout.npz, phase3125 result + p123 npz,
phase3126 result + p124 npz, phase3105
material.json, phase3113 capture_b.npz.

Parts:
  A offline: replication asserts (refit S/MS vs
    3125 1e-12; D6 shares + G6 vs 3126 1e-12);
    A1 long-range closure: lag4-6 subblock fit +
    permutation z (rng 3127) + cross-direction
    kernel transfer; cross-trajectory second-
    order (cls x m interaction on r4_d6);
    common-mode PCA; coordinate-level sparse-
    event spectrum (3113 capture_b h_out, 5
    layers x 2560 coords).
  B gpu qwen3-4b (36L): s0 single-layer swap
    ablation WRITE {26,28,30,32,34} / PORT
    {20,21} / CTRL {2,8,14}, 672 x 2 directions;
    repro vs frozen p123 mlg_s0 (<=1e-4); path
    check; delta-margin gates + wspec alignment
    (wspec_Q recomputed from p123 mlg_s0 with
    the 3126 formula).
  C gpu glm4-9b (40L): same structure, WRITE
    {8,9,13,29} / PORT {20} / CTRL {4,14,34};
    repro vs p124 mlg_s0; wspec from p124 npz;
    cross-model relative-depth alignment.
  D gpu glm4: full-672 counterfactual regen
    s1..s3 (s0 tokens frozen from p124
    gen_base); first 96 pairs are batch-
    identical to 3126 -> token-exact assertion;
    s0 replay probe (8 pairs); decode-polarity
    flip with No/no separation + multi-flip.
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
NAME = 'omega_p125_writechain_port_crossmodel_' \
       'a1closure_fullregen'
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
D22 = RDIR + r'\phase3122' \
      r'\omega_p120_write_content_readout_' \
      r'sentence_causal_dist_recon'
D24 = RDIR + r'\phase3124' \
      r'\omega_p122_resid_cm_kalman_l35rel_' \
      r'glm4x'
D25 = RDIR + r'\phase3125' \
      r'\omega_p123_third_comp_qwen_inputstream'
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
MDIR_Q = os.path.join(ROOT, 'models', 'hf',
                      'qwen3-4b')
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3127', NAME)
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
NP_B = 8 if SMOKE else NP
NP_REPRO = 4 if SMOKE else 16
NP_CHECK = 4 if SMOKE else 16
NP_REG = 4 if SMOKE else NP
TRAIL_D = 6
N_PERM = 8 if SMOKE else 200
PERM_SEED = 3127
GEN_BATCH = 4 if SMOKE else 32

# ablation layer sets (0-based)
LAY_Q = {'write': [26, 28, 30, 32, 34],
         'port': [20, 21],
         'ctrl': [2, 8, 14]}
LAY_G = {'write': [8, 9, 13, 29],
         'port': [20],
         'ctrl': [4, 14, 34]}
PORT_NORM_GATE = 2.0
SPEC_CORR_GATE = 0.6
DEPTH_CORR_GATE = 0.5

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
res24 = json.load(io.open(
    os.path.join(D24, 'result.json'),
    encoding='utf-8'))
assert res24['smoke'] is False
res25 = json.load(io.open(
    os.path.join(D25, 'result.json'),
    encoding='utf-8'))
V25 = ('common_mode_partial|trail_present|'
       'ar_absent|trajectory_transient|'
       'third_partial|path_valid|'
       'replay_bit_exact|readout_ok|spans_ok|'
       'qis_syntax_L23|qis_syntax_L22|'
       'qis_content_L28|qis_content_L32|'
       'qis_syn_final_negative|'
       'qis_syn_final_negative|'
       'qis_cont_final_negative|'
       'qis_cont_final_negative')
assert res25['verdict'] == V25
assert res25['smoke'] is False
pa25 = res25['part_a']
assert abs(pa25['decomp3']['P']['trail_share']
           - 0.08599867672568678) < 1e-12
res26 = json.load(io.open(
    os.path.join(D26, 'result.json'),
    encoding='utf-8'))
V26 = res26['verdict']
assert res26['smoke'] is False
assert 'baseline_replicated' in V26
z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
auc18 = z18['auc_curve']
assert len(auc18) == N_NEW + 1
z20 = np.load(os.path.join(D20, 'p118_readout.npz'),
              allow_pickle=False)
annP_c = z20['annP_c']
annA_c = z20['annA_c']
assert annP_c.shape == (672, N_NEW)
z22 = np.load(os.path.join(D22, 'p120_readout.npz'),
              allow_pickle=False)
z24 = np.load(os.path.join(D24, 'p122_readout.npz'),
              allow_pickle=False)
z25 = np.load(os.path.join(D25, 'p123_readout.npz'),
              allow_pickle=False)
assert z25['mlg_s0_P'].shape == (672, 37, 13)
z26 = np.load(os.path.join(D26, 'p124_readout.npz'),
              allow_pickle=False)
assert z26['mlg_s0_P'].shape == (672, 41, 13)
assert z26['G6_P'].shape == (10, 6)
assert z26['gen_base_P'].shape == (672, 12)
log('frozen inputs ok (05/13/18/20/22/24/25/26)')

# ================================================================
# design seal (frozen BEFORE observation)
# ================================================================
seal = {
    'phase': 3127,
    'name': NAME,
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'np_a': NP,
    'np_b': NP_B,
    'np_reg': NP_REG,
    'data_sources': {
        'margins_annotation': 'phase3118 + 3120',
        'fits_3125_3126': 'refit/D6 asserts '
                          '1e-12 vs frozen '
                          'results',
        'qwen_field': 'phase3125 p123 mlg_s0 '
                      '(672x37x13 f32)',
        'glm4_field': 'phase3126 p124 mlg_s0 '
                      '(672x41x13 f32) + '
                      'gen_base tokens + '
                      'wspec + regen96',
        'coords': 'phase3113 capture_b h_out '
                  '(2016x5x2560 f16) + m',
        'qwen_write_ref': 'phase3122 p120 '
                          'wrec_pd (MLP write '
                          'projection, '
                          'secondary '
                          'descriptor)'},
    'part_a': {
        'a1_closure': 'lag4-6 subblock (30 cols) '
                      'fit on r2 per direction; '
                      'gain46 = 1 - SS(res)/'
                      'SS_tot; perm z (cls '
                      'row-permute, argsort '
                      'rng 3127, 200 reps, '
                      'full pipeline per rep); '
                      'min z >= 4 -> '
                      'lag46_significant',
        'transfer': 'A1 rows fitted with FROZEN '
                    'P lag4-6 kernel (3126 '
                    'G6_P[:, 3:6], no refit) '
                    'and vice versa; '
                    'gain_transfer >= 0.02 AND '
                    '>= 2x max(gain46_own, '
                    '0.005) -> '
                    'a1_transferable (material '
                    'side) else '
                    'a1_short_range_intrinsic',
        'cross_second': 'on r4_d6 rows: additive '
                        '[cls10, m_c, 1] vs '
                        'interactive '
                        '[+cls10*m_c]; gain = '
                        'SS_add - SS_int over '
                        'SS_tot; min over dirs '
                        '>= 0.03 -> '
                        'cross_second_candidate',
        'common_mode': 'stack r4_d6 (1344x12), '
                       'column-center, SVD; '
                       'PC1 var share >= 0.30 '
                       '-> '
                       'common_mode_candidate',
        'sparse_events': 'capture_b h_out per '
                         '(layer,coord) z over '
                         '2016 records; event '
                         '|z|>=4; overall rate '
                         '>= 1e-3 AND top-layer '
                         'share >= 0.4 -> '
                         'sparse_events_present',
        'no_mc': 'deterministic perms rng 3127'},
    'part_b': {
        'model': 'qwen3-4b (36L BF16 eager '
                 'batch1; norm applied ONLY to '
                 'L<NL, transformers 5.14 '
                 'append-before trap, 3125 '
                 'semantics)',
        'traj': '3118 gen_clean tokens '
                '(3125-identical)',
        'ablation': 'single-layer swap '
                    '(layer output := input, '
                    'in-place) on s0 prompts; '
                    'layers WRITE {26,28,30,32,'
                    '34} PORT {20,21} CTRL '
                    '{2,8,14}; readout '
                    'delta-margin at final '
                    'position, 13-point field '
                    'stored',
        'repro': 's0 forwards vs frozen p123 '
                 'mlg_s0 atol 1e-4 (16 pairs); '
                 'path r >= 0.9999 on 16 pairs '
                 '(FATAL if not)',
        'gates': 'write_functional: '
                 'median|dM_final| WRITE >= '
                 '%.1f x CTRL median; '
                 'port_functional: PORT >= '
                 '%.1f x CTRL; spectrum: '
                 'Spearman(wspec_Q[l], '
                 'median|dM(l)|) >= %.1f over '
                 'the 10 ablated layers; '
                 'wspec_Q recomputed from '
                 'p123 mlg_s0 with the 3126 '
                 'formula'
                 % (PORT_NORM_GATE,
                    PORT_NORM_GATE,
                    SPEC_CORR_GATE)},
    'part_c': {
        'model': 'glm4-9b-chat-hf (40L BF16 '
                 'eager batch1; hook-collected, '
                 'normG on ALL states, 3126 '
                 'semantics)',
        'traj': '3126 gen_base tokens',
        'ablation': 'same swap, layers WRITE '
                    '{8,9,13,29} PORT {20} CTRL '
                    '{4,14,34}',
        'gates': 'same with wspec_G from p124 '
                 'npz; plus cross-model '
                 'relative-depth alignment: '
                 'Spearman(depth-profile_Q, '
                 'depth-profile_G) >= %.1f '
                 '(normalized med|dM| / ctrl '
                 'median, layer/NL)'
                 % DEPTH_CORR_GATE},
    'part_d': {
        'model': 'glm4 (same load as C)',
        's0': 'tokens frozen from p124 '
              'gen_base; 8-pair replay probe '
              'token-agree >= 0.95 -> '
              'baseline_s0_replay_ok',
        'regen': 'full 672 x s1..s3 x 2 dirs, '
                 'greedy, GEN_BATCH=%d; first '
                 '96 pairs are batch-composition-'
                 'identical to 3126 (3x32) -> '
                 'token-exact vs p124 '
                 'regen_s1/s2/s3 REQUIRED ('
                 'FATAL if not)' % GEN_BATCH,
        'polarity': 'decoded first token: '
                    'yes* -> +1, no* -> -1, '
                    'No/no counted separately '
                    '(n_no_cap/n_no_low), other '
                    '-> 0; flip = sign change '
                    'with both nonzero',
        'multi_flip': 'polarity changes within '
                      'first 3 tokens',
        'gates': 'shift_min_full >= 0.10 -> '
                 'behavior_shift_full; coverage '
                 'full672 (degrade rule: if '
                 'projected > 6.5h at SMOKE, '
                 's2/s3 to even pairs 336, '
                 'coverage_degraded)'}
}
SEALF = os.path.join(OUT, 'design_seal.json')
with io.open(SEALF, 'w',
             encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('seal frozen: %s' % SEALF)

# ================================================================
# PART A: offline
# ================================================================
log('== PART A: offline ==')
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


def content_table(m, cls):
    dm = m[:, 1:] - m[:, :-1]
    tab = np.zeros((10, N_NEW),
                   dtype=np.float64)
    for k in range(N_NEW):
        col = cls[:, k]
        dv = dm[:, k]
        allm = dv.mean()
        for cc in range(10):
            sel = (col == cc)
            if sel.any():
                tab[cc, k] = float(
                    dv[sel].mean() - allm)
    return tab


def refit(m, cls, tab):
    dm = m[:, 1:] - m[:, :-1]
    dev = tab[cls, np.arange(N_NEW)]
    dm_res = dm - dev
    X = np.stack([m[:, :-1].ravel(),
                  np.ones(m[:, :-1].size)], 1)
    y = dm_res.ravel()
    bb, *_ = np.linalg.lstsq(X, y,
                             rcond=None)
    S_dc = float(bb[0])
    MS_dc = float(-bb[1] / bb[0])
    resid = (y - X @ bb).reshape(NP, N_NEW)
    return S_dc, MS_dc, resid


def trailX(cls, D, lags=None):
    """one-hot trail design; lags=None ->
    all d=1..D (3126-identical); else the
    given lag list (subblock)."""
    if lags is None:
        lags = list(range(1, D + 1))
    NIK = cls.size
    X = np.zeros((NIK, 10 * len(lags)))
    for ci, d in enumerate(lags):
        rows_k = np.arange(d, N_NEW)
        idx_rows = (np.repeat(
            np.arange(NP) * N_NEW,
            len(rows_k))
            + np.tile(rows_k, NP))
        cls_d = cls[:, :N_NEW - d].ravel()
        X[idx_rows, cls_d * len(lags)
          + ci] = 1.0
    return X


def seq_shares(m, MS, resid, cls, D):
    ss_tot = float((resid * resid).sum())
    NIK = resid.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, resid.ravel(),
                             rcond=None)
    r1 = (resid.ravel() - Z1 @ b1).reshape(
        NP, N_NEW)
    ss_step = ss_tot - float((r1 * r1).sum())
    x = (m[:, :-1] - MS).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1.ravel(),
                             rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(
        NP, N_NEW)
    ss_m = float((r1 * r1).sum()) \
        - float((r2 * r2).sum())
    X_tr = trailX(cls, D)
    btr, *_ = np.linalg.lstsq(X_tr, r2.ravel(),
                              rcond=None)
    r3 = (r2.ravel() - X_tr @ btr).reshape(
        NP, N_NEW)
    ss_trail = float((r2 * r2).sum()) \
        - float((r3 * r3).sum())
    return (ss_step, ss_m, ss_trail, ss_tot,
            r3, X_tr, btr)


fit = {}
for dc in ('P', 'A1'):
    tab = content_table(MSEQ[dc], ANN[dc])
    S_dc, MS_dc, resid = refit(MSEQ[dc],
                               ANN[dc], tab)
    fit[dc] = {'S': S_dc, 'MS': MS_dc,
               'resid': resid, 'col': ANN[dc],
               'tab': tab}
    r25f = pa25['refit'][dc]
    assert abs(S_dc - r25f['S']) < 1e-12, dc
    assert abs(MS_dc - r25f['MS']) < 1e-12, dc
log('A refit assert vs 3125 ok')

dec6 = {}
ar6 = {}
G6 = {}
r4_store = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    R = fit[dc]['resid']
    cls = fit[dc]['col']
    (ss_step, ss_m, ss_tr6, ss_tot,
     r3, X_tr, btr) = seq_shares(
        m, fit[dc]['MS'], R, cls, TRAIL_D)
    y_ar = r3[:, 2:].ravel()
    X_ar = np.stack(
        [r3[:, 1:N_NEW - 1].ravel(),
         r3[:, 0:N_NEW - 2].ravel(),
         np.ones(y_ar.size)], 1)
    bar, *_ = np.linalg.lstsq(X_ar, y_ar,
                              rcond=None)
    phi1, phi2, c_ar = (float(bar[0]),
                        float(bar[1]),
                        float(bar[2]))
    r4 = r3.copy()
    r4[:, 2:] = r3[:, 2:] - (
        phi1 * r3[:, 1:N_NEW - 1]
        + phi2 * r3[:, 0:N_NEW - 2] + c_ar)
    ss_ar = float(((r3[:, 2:] - r4[:, 2:])
                   ** 2).sum())
    ss_rem = float((r4 * r4).sum())
    leak = (1.0 - ss_step / ss_tot
            - ss_m / ss_tot
            - ss_tr6 / ss_tot
            - ss_ar / ss_tot
            - ss_rem / ss_tot)
    dec6[dc] = {
        'ss_tot': ss_tot,
        'cm_share': ss_step / ss_tot,
        'm_share': ss_m / ss_tot,
        'trail6_share': ss_tr6 / ss_tot,
        'ar6_share': ss_ar / ss_tot,
        'rem6_share': ss_rem / ss_tot,
        'leak': leak}
    ar6[dc] = {'phi1': phi1, 'phi2': phi2,
               'c': c_ar}
    G6[dc] = btr.reshape(10, TRAIL_D)
    r4_store[dc] = r4
    d26 = res26['part_a']['decomp6'][dc]
    for k in ('cm_share', 'm_share',
              'trail6_share', 'ar6_share',
              'rem6_share'):
        assert abs(dec6[dc][k]
                   - d26[k]) < 1e-12, (dc, k)
    assert np.abs(G6[dc]
                  - z26['G6_%s' % dc]) \
        .max() < 1e-10, dc
log('A D6 decomp asserts vs 3126 ok '
    '(1e-12 shares / 1e-10 G6)')

# ---- A1 closure: lag4-6 subblock ----
LAGS46 = (4, 5, 6)
a1clo = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    cls = fit[dc]['col']
    R = fit[dc]['resid']
    ss_tot = dec6[dc]['ss_tot']
    # r2 (pre-trail residual), same as
    # seq_shares internals
    NIK = R.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, R.ravel(),
                             rcond=None)
    r1 = (R.ravel() - Z1 @ b1).reshape(
        NP, N_NEW)
    x = (m[:, :-1] - fit[dc]['MS']).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1.ravel(),
                             rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(
        NP, N_NEW)
    ss_r2 = float((r2 * r2).sum())
    X46 = trailX(cls, TRAIL_D,
                 lags=LAGS46)
    b46, *_ = np.linalg.lstsq(X46, r2.ravel(),
                              rcond=None)
    r46 = r2.ravel() - X46 @ b46
    gain46 = (ss_r2 - float((r46 * r46)
                            .sum())) / ss_tot
    G46 = b46.reshape(10, len(LAGS46))
    # frozen cross-direction transfer
    GT = z26['G6_%s' % ('A1' if dc == 'P'
                        else 'P')][:, 3:6]
    gflat = GT.reshape(-1)
    r2t = r2.ravel() - X46 @ gflat
    gain_tr = (ss_r2 - float((r2t * r2t)
                             .sum())) / ss_tot
    # permutation null (cls row-permute,
    # full pipeline per rep, seed 3127)
    rgA = np.random.default_rng(PERM_SEED)
    nulls = np.zeros(N_PERM)
    for rep in range(N_PERM):
        pm = np.argsort(
            rgA.random((NP, N_NEW)), axis=1)
        cls_p = np.take_along_axis(
            cls, pm, axis=1)
        tab_p = content_table(m, cls_p)
        S_p, MS_p, resid_p = refit(
            m, cls_p, tab_p)
        Z1p = np.zeros((NIK, N_NEW))
        Z1p[np.arange(NIK),
            np.arange(NIK) % N_NEW] = 1.0
        b1p, *_ = np.linalg.lstsq(
            Z1p, resid_p.ravel(), rcond=None)
        r1p = (resid_p.ravel()
               - Z1p @ b1p).reshape(NP, N_NEW)
        xp = (m[:, :-1] - MS_p).ravel()
        Z2p = np.stack([xp, np.ones(NIK)], 1)
        b2p, *_ = np.linalg.lstsq(
            Z2p, r1p.ravel(), rcond=None)
        r2p = (r1p.ravel() - Z2p @ b2p) \
            .reshape(NP, N_NEW)
        ss_r2p = float((r2p * r2p).sum())
        X46p = trailX(cls_p, TRAIL_D,
                      lags=LAGS46)
        b46p, *_ = np.linalg.lstsq(
            X46p, r2p.ravel(), rcond=None)
        r46p = r2p.ravel() - X46p @ b46p
        ss_tot_p = float(
            (resid_p * resid_p).sum())
        nulls[rep] = (ss_r2p
                      - float((r46p * r46p)
                              .sum())) / ss_tot_p
    mu = float(nulls.mean())
    sd = float(nulls.std())
    z46 = (gain46 - mu) / sd if sd > 0 else 0.0
    a1clo[dc] = {
        'gain46': float(gain46),
        'G46': G46.tolist(),
        'gain_transfer': float(gain_tr),
        'perm_mean': mu, 'perm_std': sd,
        'z46': float(z46),
        'reps': N_PERM, 'seed': PERM_SEED}
    log('A-A1CLO %s: gain46=%.4f z=%.2f '
        'gain_tr=%.4f'
        % (dc, gain46, z46, gain_tr))
z46_min = min(a1clo['P']['z46'],
              a1clo['A1']['z46'])
a_lag_v = ('lag46_significant'
           if z46_min >= 4.0
           else 'lag46_notsig')
own_a1 = max(a1clo['A1']['gain46'], 0.005)
tr_a1 = a1clo['A1']['gain_transfer']
a_tr_v = ('a1_transferable'
          if (tr_a1 >= 0.02
              and tr_a1 >= PORT_NORM_GATE
              * 0.0 + 2.0 * own_a1)
          else 'a1_short_range_intrinsic')
log('A-GATE lag46: %s | transfer: %s'
    % (a_lag_v, a_tr_v))

# ---- A2: cross-second + PCA + sparse ----
second2 = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    r4 = r4_store[dc]
    cls = fit[dc]['col']
    ss_tot = dec6[dc]['ss_tot']
    NIK = r4.size
    A_cls = np.zeros((NIK, 10))
    A_cls[np.arange(NIK),
          cls.ravel()] = 1.0
    m_c = (m[:, :-1]
           - fit[dc]['MS']).ravel()
    m_c = m_c - m_c.mean()
    ones = np.ones(NIK)
    X_add = np.column_stack(
        [A_cls, m_c, ones])
    X_int = np.column_stack(
        [A_cls, m_c, A_cls * m_c[:, None],
         ones])
    ba, *_ = np.linalg.lstsq(X_add,
                             r4.ravel(),
                             rcond=None)
    ra = r4.ravel() - X_add @ ba
    bi, *_ = np.linalg.lstsq(X_int,
                             r4.ravel(),
                             rcond=None)
    ri = r4.ravel() - X_int @ bi
    gain_x = (float((ra * ra).sum())
              - float((ri * ri).sum())) / ss_tot
    second2[dc] = {'cross_gain': float(gain_x)}
    log('A-CROSS2 %s: gain=%.4f'
        % (dc, gain_x))
gx_min = min(second2['P']['cross_gain'],
             second2['A1']['cross_gain'])
a_cross_v = ('cross_second_candidate'
             if gx_min >= 0.03
             else 'cross_second_absent')

Zpca = np.stack([r4_store['P'],
                 r4_store['A1']],
                0).reshape(2 * NP, N_NEW)
Zc = Zpca - Zpca.mean(0, keepdims=True)
U, sv, Vt = np.linalg.svd(Zc,
                          full_matrices=False)
var_exp = (sv * sv) / float((Zc * Zc).sum())
pc1 = U[:, 0] * sv[0]
pc1_share = float(var_exp[0])
m_mean = np.concatenate([
    MSEQ['P'].mean(1), MSEQ['A1'].mean(1)])
dir_ind = np.concatenate([
    np.zeros(NP), np.ones(NP)])
corr_pc1_m = float(np.corrcoef(pc1, m_mean)
                   [0, 1])
corr_pc1_dir = float(np.corrcoef(pc1, dir_ind)
                     [0, 1])
a_cm_v = ('common_mode_factor_candidate'
          if pc1_share >= 0.30
          else 'common_mode_absent')
log('A-PCA: PC1=%.4f corr(m)=%.4f '
    'corr(dir)=%.4f -> %s'
    % (pc1_share, corr_pc1_m, corr_pc1_dir,
       a_cm_v))

h_out = capb['h_out'].astype(np.float32)
m_cap = capb['m'].astype(np.float64)
LAY5 = capb['layers']
n_rec = h_out.shape[0]
ev_counts = np.zeros(len(LAY5),
                     dtype=np.int64)
for li in range(len(LAY5)):
    hl = h_out[:, li, :]
    mu_l = hl.mean(0, keepdims=True)
    sd_l = hl.std(0, keepdims=True)
    sd_l[sd_l == 0] = 1.0
    zl = (hl - mu_l) / sd_l
    ev_counts[li] = int(
        (np.abs(zl) >= 4.0).sum())
ev_total = int(ev_counts.sum())
ev_rate = ev_total / float(n_rec
                           * h_out.shape[1]
                           * h_out.shape[2])
ev_share_top = float(
    ev_counts.max() / max(ev_total, 1))
a_sp_v = ('sparse_events_present'
          if (ev_rate >= 1e-3
              and ev_share_top >= 0.4)
          else 'sparse_events_absent')
log('A-SPARSE: rate=%.5f top-share=%.3f '
    'counts=%s -> %s'
    % (ev_rate, ev_share_top,
       ev_counts.tolist(), a_sp_v))
log('A-GATES: %s | %s | %s | %s | %s'
    % (a_lag_v, a_tr_v, a_cross_v, a_cm_v,
       a_sp_v))

# ================================================================
# materials factory (3125/3126-identical logic,
# tokenizer parameterized)
# ================================================================
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


def make_materials(tok, gens):
    """gens[dc] = (672, 12) int token array
    (frozen trajectory tokens)."""
    texts = {}
    PID_T = {}
    for dc in ('P', 'A1'):
        texts[dc] = {}
        ids_l = []
        for pk in pks:
            (s, o) = (int(v)
                      for v in pk.split('_'))
            r = p2r[pk]
            ri1, ri2 = frel[pk]
            if dc == 'P':
                (qrel, lrel) = (r, r)
            else:
                (qrel, lrel) = (r, ri1)
            t_ = build_prompt(mat5, s, o,
                              lrel, qrel)
            texts[dc][pk] = t_
            ids_l.append(list(
                tok(t_, add_special_tokens=False)
                ['input_ids']))
        PID_T[dc] = ids_l
    DOT = int(tok('.', add_special_tokens=False)
              ['input_ids'][0])
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
        (s, o) = (int(v)
                  for v in pk.split('_'))
        r = p2r[pk]
        ri1, ri2 = frel[pk]
        lrel = r if dcode == 'P' else ri1
        D = [tuple(d) for d in
             mat5['distractors']
             ['%d_%d' % (s, o)]]
        ctx = set((a, rr, b)
                  for (a, rr, b) in
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
        raise RuntimeError('no replacement')

    def traj_tokens(dc, j):
        ids_r = [int(t) for t in gens[dc][j]]
        ptxt = texts[dc][pks[j]]
        encp = tok(ptxt, add_special_tokens=False,
                   return_offsets_mapping=True)
        poffs = [tuple(v) for v in
                 encp['offset_mapping']]
        ids2 = list(ids_r)[:N_NEW]
        while len(ids2) < N_NEW:
            ids2.append(DOT)
        return ptxt, poffs, ids2

    def find_span(dec, offs, s, o, qrel):
        """Anchored LAST occurrence."""
        qline = 'The %s %s the %s.' % (
            ents_all[s], PREDS_all[qrel],
            ents_all[o])
        ce = dec.rfind(qline)
        while ce != -1:
            cs = ce
            ce_want = cs + len(qline)
            if dec[ce_want:ce_want + 8] \
                    == ' Is this':
                idxs = [k for k in
                        range(len(offs))
                        if offs[k][0] >= cs
                        and offs[k][1]
                        <= ce_want
                        and offs[k][1]
                        > offs[k][0]]
                if len(idxs) >= 2 \
                        and idxs[0] >= 1 \
                        and idxs[-1] - idxs[0] \
                        + 1 <= N_NEW:
                    return idxs[0], idxs[-1]
            ce = dec.rfind(qline, 0, ce)
        return None

    span_idx = {dc: np.full((NP, 2), -1,
                            dtype=np.int16)
                for dc in ('P', 'A1')}
    pids_all = {dc: [None] * NP
                for dc in ('P', 'A1')}

    def build_pids(dc, j):
        pk = pks[j]
        prompt_ids0 = list(PID_T[dc][j])
        ptxt, poffs, _base = traj_tokens(dc, j)
        (s, o) = (int(v)
                  for v in pk.split('_'))
        r = p2r[pk]
        span = find_span(ptxt, poffs, s, o, r)
        pids = {c: list(prompt_ids0)
                for c in ('s0', 's1', 's2',
                          's3')}
        if span is not None:
            (k1, k2) = span
            Lspan = k2 - k1 + 1
            (s2, o2) = pick_replacement(pk, dc)
            (_, _, lrel, _) = context_triples(
                pk, dc)
            sub = line_tokens(s2, o2, lrel)
            n_pad = max(0, Lspan - len(sub))
            sub = sub[:Lspan]
            sub = sub + [DOT] * n_pad
            rng2 = _rnd.Random(zlib.crc32(
                ('s2|%s|%s' % (pk, dc))
                .encode('ascii')))
            shuf = list(sub)
            rng2.shuffle(shuf)
            pids['s1'][k1:k2 + 1] = sub
            pids['s2'][k1:k2 + 1] = shuf
            pids['s3'][k1:k2 + 1] = \
                [DOT] * Lspan
            span_idx[dc][j] = (k1, k2)
        pids_all[dc][j] = pids
        return pids, span

    for dc in ('P', 'A1'):
        for j in range(NP):
            build_pids(dc, j)
    return {'texts': texts, 'PID_T': PID_T,
            'DOT': DOT, 'build_pids': build_pids,
            'span_idx': span_idx,
            'pids_all': pids_all,
            'traj_tokens': traj_tokens}


# ================================================================
# PART B: GPU qwen3-4b port ablation
# ================================================================
log('== PART B: qwen port ablation ==')
gens_q = {'P': z18['gen_clean__P'],
          'A1': z18['gen_clean__A1']}

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

tok_q = AutoTokenizer.from_pretrained(MDIR_Q)
model_q = AutoModelForCausalLM.from_pretrained(
    MDIR_Q, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NLQ = len(model_q.model.layers)
assert NLQ == 36
log('qwen3-4b loaded NLQ=%d' % NLQ)
WU_q = model_q.lm_head.weight.detach()
assert model_q.lm_head.bias is None
YES_Q = int(mat5['yes_id'])
NO_Q = int(mat5['no_id'])
w_dn_q = (WU_q[YES_Q] - WU_q[NO_Q]) \
    .float().cpu().numpy()
norm_q = model_q.model.norm
matq = make_materials(tok_q, gens_q)
for dc in ('P', 'A1'):
    assert np.array_equal(
        matq['span_idx'][dc],
        z25['span_idx_%s' % dc]), dc
log('B span_idx asserts vs 3125 ok (672x2)')


def forward_track_q(prompt_ids, traj_ids,
                    want_logits=False,
                    swap_layer=None):
    """3125 semantics: hidden_states[NL] already
    final-normed -> norm applied ONLY to L<NL;
    swap_layer: replace that layer output by
    its input (in-place, before collection)."""
    traj_ids = [int(x) for x in traj_ids]
    ids = list(prompt_ids) + traj_ids
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(traj_ids) + 1
    ml = np.zeros((NLQ + 1, npts),
                  dtype=np.float64)
    lmv = None
    hk = None
    if swap_layer is not None:
        lyr = model_q.model.layers[swap_layer]

        def _swap(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            i2 = inp[0] \
                if isinstance(inp, tuple) \
                else inp
            o2.copy_(i2.to(o2.dtype))

        hk = lyr.register_forward_hook(_swap)
    with torch.inference_mode():
        out = model_q(t_in,
                      output_hidden_states=True,
                      use_cache=False)
        if hk is not None:
            hk.remove()
        for L in range(NLQ + 1):
            hL = out.hidden_states[L][0]
            if L < NLQ:
                hL = norm_q(hL)
            h = hL.float().cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dn_q)
        if want_logits:
            lg = out.logits[0,
                            pos0:pos0 + npts] \
                .float().cpu().numpy()
            lmv = lg[:, YES_Q] - lg[:, NO_Q]
        del out
    return ml, lmv


# repro vs frozen p123 mlg_s0 + path check
drep_q = 0.0
path_rq = []
for dc in ('P', 'A1'):
    for j in range(NP_REPRO):
        pids, _sp = matq['build_pids'](dc, j)
        _ptxt, _poffs, base = matq['traj_tokens'](
            dc, j)
        wl, lm = forward_track_q(
            pids['s0'], base,
            want_logits=(j < NP_CHECK))
        ref = z25['mlg_s0_%s' % dc][j]
        drep_q = max(drep_q, float(
            np.abs(wl - ref).max()))
        if lm is not None:
            path_rq.append(float(np.corrcoef(
                wl[NLQ, :], lm)[0, 1]))
path_q_min = min(path_rq)
assert drep_q <= 1e-4, drep_q
assert path_q_min >= 0.9999, path_q_min
log('B repro max|d|=%.2e (<=1e-4) | path r '
    'min=%.6f' % (drep_q, path_q_min))

# swap ablation over 672 x 2 x 10 layers
ABL_Q = ([(l, 'write') for l in LAY_Q['write']]
         + [(l, 'port') for l in LAY_Q['port']]
         + [(l, 'ctrl') for l in LAY_Q['ctrl']])
dm_field_q = {}
t0b = time.time()
for (l, tag) in ABL_Q:
    for dc in ('P', 'A1'):
        store = np.zeros((NP_B, N_NEW + 1),
                         dtype=np.float64)
        for j in range(NP_B):
            pids, _sp = matq['build_pids'](dc, j)
            _ptxt, _poffs, base = \
                matq['traj_tokens'](dc, j)
            wl, _lm = forward_track_q(
                pids['s0'], base,
                swap_layer=l)
            store[j] = wl[NLQ, :]
        dm_field_q['%s_L%02d' % (dc, l)] = \
            store.astype(np.float32)
    done = sum(1 for _ in ABL_Q)
    log('B abl L%02d (%s) done (%.1fs)'
        % (l, tag, time.time() - T0))
base_final = {
    dc: z25['mlg_s0_%s' % dc][:NP_B, NLQ, :]
    for dc in ('P', 'A1')}
dm_final_q = {}
for (l, tag) in ABL_Q:
    for dc in ('P', 'A1'):
        key = '%s_L%02d' % (dc, l)
        dm_final_q[key] = (
            dm_field_q[key][:, -1]
            - base_final[dc][:, -1]).astype(
                np.float64)

# wspec_Q from p123 mlg_s0 (3126 formula)
wspec_q = {}
W_q = {}
for dc in ('P', 'A1'):
    s0m = z25['mlg_s0_%s' % dc][:NP] \
        .astype(np.float64)
    D_l = s0m[:, 1:, :] - s0m[:, :-1, :]
    ws = D_l.mean(0)
    wspec_q[dc] = ws
    W_q[dc] = ws.mean(1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(
        np.float64)
    rb = np.argsort(np.argsort(b)).astype(
        np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


b_gates = {}
med_c_q = np.median(np.abs(np.concatenate(
    [dm_final_q['P_L%02d' % l]
     for l in LAY_Q['ctrl']]
    + [dm_final_q['A1_L%02d' % l]
       for l in LAY_Q['ctrl']])))
for grp in ('write', 'port'):
    meds = []
    for l in LAY_Q[grp]:
        for dc in ('P', 'A1'):
            meds.append(np.median(np.abs(
                dm_final_q['%s_L%02d'
                           % (dc, l)])))
    med_g = float(np.median(meds))
    b_gates[grp] = med_g / max(med_c_q, 1e-12)
log('B GATES: ctrl_med=%.4f write x%.2f port '
    'x%.2f' % (med_c_q, b_gates['write'],
               b_gates['port']))
spec_pairs = []
for (l, tag) in ABL_Q:
    for dc in ('P', 'A1'):
        key = '%s_L%02d' % (dc, l)
        spec_pairs.append((
            float(np.abs(W_q[dc][l])),
            float(np.median(np.abs(
                dm_final_q[key])))))
sp_x = np.array([p[0] for p in spec_pairs])
sp_y = np.array([p[1] for p in spec_pairs])
spec_corr_q = spearman(sp_x, sp_y)
b_write_v = ('qwen_write_functional'
             if b_gates['write']
             >= PORT_NORM_GATE
             else 'qwen_write_not')
b_port_v = ('qwen_port_functional'
            if b_gates['port']
            >= PORT_NORM_GATE
            else 'qwen_port_not')
b_spec_v = ('qwen_spectrum_aligned'
            if spec_corr_q >= SPEC_CORR_GATE
            else 'qwen_spectrum_weak')
log('B-GATES: %s | %s | %s (spec r=%.3f)'
    % (b_write_v, b_port_v, b_spec_v,
       spec_corr_q))

# secondary descriptor: MLP write projection
# depth profile from 3122 wrec_pd (recorded,
# not gated)
wp_q = {}
for dc in ('P', 'A1'):
    wp = z22['wrec_pd_%s' % dc].astype(
        np.float64)
    wp_q[dc] = wp.mean(axis=(1, 2))
log('B wrec_pd mean-profile top3 P: %s'
    % list(np.argsort(np.abs(wp_q['P']))[::-1]
           [:3]))

del model_q
import gc  # noqa: E402

gc.collect()
torch.cuda.empty_cache()
log('qwen unloaded; cuda cache cleared')

# ================================================================
# PART C: GPU glm4 port ablation
# ================================================================
log('== PART C: glm4 port ablation ==')
gens_g = {'P': z26['gen_base_P'],
          'A1': z26['gen_base_A1']}
tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
model_g = AutoModelForCausalLM.from_pretrained(
    MDIR_G, torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    trust_remote_code=True).to('cuda').eval()
NLG = len(model_g.model.layers)
assert NLG == 40
log('glm4 loaded NLG=%d' % NLG)
WUG = model_g.lm_head.weight.detach()
YES_G = int(tok_g(' yes', add_special_tokens=False)
            ['input_ids'][0])
NO_G = int(tok_g(' no', add_special_tokens=False)
           ['input_ids'][0])
w_dn_g = (WUG[YES_G] - WUG[NO_G]) \
    .float().cpu().numpy()
DOT_G = int(tok_g('.', add_special_tokens=False)
            ['input_ids'][0])
norm_g = model_g.model.norm
PADG = tok_g.pad_token_id
if PADG is None:
    PADG = model_g.config.pad_token_id
tok_g.padding_side = 'left'
matg = make_materials(tok_g, gens_g)
for dc in ('P', 'A1'):
    assert np.array_equal(
        matg['span_idx'][dc],
        z26['span_idx_%s' % dc]), dc
log('C span_idx asserts vs 3126 ok (672x2)')
# gen-encoding probe (3126-identical)
_genenc = tok_g(matg['texts']['P'][pks[0]])[
    'input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']['P'][0])]]
assert len(PREFIX_IDS) in (0, 2)
log('C gen-prefix: %s (decoded %r)'
    % (PREFIX_IDS,
       tok_g.decode(PREFIX_IDS)
       if PREFIX_IDS else ''))


def forward_track_g(prompt_ids, traj_ids,
                    want_logits=False,
                    swap_layer=None):
    """3126 semantics: hook-collected layer
    outputs, normG applied to ALL states."""
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
    if swap_layer is not None:
        lyr = model_g.model.layers[swap_layer]

        def _swap(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            i2 = inp[0] \
                if isinstance(inp, tuple) \
                else inp
            o2.copy_(i2.to(o2.dtype))

        hooks.append(
            lyr.register_forward_hook(_swap))

    def _mk():
        def hook(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            feats.append(o2.detach())
        return hook

    for lyr in model_g.model.layers:
        hooks.append(
            lyr.register_forward_hook(_mk()))
    with torch.inference_mode():
        out = model_g(t_in,
                      output_hidden_states=False,
                      use_cache=False)
        for hk in hooks:
            hk.remove()
        seq = [model_g.model.embed_tokens(t_in)] \
            + feats
        for L in range(NLG + 1):
            h = norm_g(seq[L][0]).float() \
                .cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dn_g)
        if want_logits:
            lg = out.logits[0].float().cpu() \
                .numpy()
            lm = np.array([
                float(lg[pos0 + k][YES_G])
                - float(lg[pos0 + k][NO_G])
                for k in range(npts)])
        del out, seq, feats
    return ml, lm


drep_g = 0.0
path_rg = []
for dc in ('P', 'A1'):
    for j in range(NP_REPRO):
        pids, _sp = matg['build_pids'](dc, j)
        _ptxt, _poffs, base = matg['traj_tokens'](
            dc, j)
        wl, lm = forward_track_g(
            pids['s0'], base,
            want_logits=(j < NP_CHECK))
        ref = z26['mlg_s0_%s' % dc][j]
        drep_g = max(drep_g, float(
            np.abs(wl - ref).max()))
        if lm is not None:
            path_rg.append(float(np.corrcoef(
                wl[NLG, :], lm)[0, 1]))
path_g_min = min(path_rg)
assert drep_g <= 1e-4, drep_g
assert path_g_min >= 0.9999, path_g_min
log('C repro max|d|=%.2e (<=1e-4) | path r '
    'min=%.6f' % (drep_g, path_g_min))

ABL_G = ([(l, 'write') for l in LAY_G['write']]
         + [(l, 'port') for l in LAY_G['port']]
         + [(l, 'ctrl') for l in LAY_G['ctrl']])
dm_field_g = {}
for (l, tag) in ABL_G:
    for dc in ('P', 'A1'):
        store = np.zeros((NP_B, N_NEW + 1),
                         dtype=np.float64)
        for j in range(NP_B):
            pids, _sp = matg['build_pids'](dc, j)
            _ptxt, _poffs, base = \
                matg['traj_tokens'](dc, j)
            wl, _lm = forward_track_g(
                pids['s0'], base,
                swap_layer=l)
            store[j] = wl[NLG, :]
        dm_field_g['%s_L%02d' % (dc, l)] = \
            store.astype(np.float32)
    log('C abl L%02d (%s) done (%.1fs)'
        % (l, tag, time.time() - T0))
base_final_g = {
    dc: z26['mlg_s0_%s' % dc][:NP_B, NLG, :]
    for dc in ('P', 'A1')}
dm_final_g = {}
for (l, tag) in ABL_G:
    for dc in ('P', 'A1'):
        key = '%s_L%02d' % (dc, l)
        dm_final_g[key] = (
            dm_field_g[key][:, -1]
            - base_final_g[dc][:, -1]).astype(
                np.float64)

med_c_g = np.median(np.abs(np.concatenate(
    [dm_final_g['P_L%02d' % l]
     for l in LAY_G['ctrl']]
    + [dm_final_g['A1_L%02d' % l]
       for l in LAY_G['ctrl']])))
c_gates = {}
for grp in ('write', 'port'):
    meds = []
    for l in LAY_G[grp]:
        for dc in ('P', 'A1'):
            meds.append(np.median(np.abs(
                dm_final_g['%s_L%02d'
                           % (dc, l)])))
    med_g = float(np.median(meds))
    c_gates[grp] = med_g / max(med_c_g, 1e-12)
log('C GATES: ctrl_med=%.4f write x%.2f port '
    'x%.2f' % (med_c_g, c_gates['write'],
               c_gates['port']))
spec_pairs_g = []
for (l, tag) in ABL_G:
    for dc in ('P', 'A1'):
        key = '%s_L%02d' % (dc, l)
        Wg = z26['wspec_%s' % dc].astype(
            np.float64).mean(1)
        spec_pairs_g.append((
            float(np.abs(Wg[l])),
            float(np.median(np.abs(
                dm_final_g[key])))))
sp_xg = np.array([p[0] for p in spec_pairs_g])
sp_yg = np.array([p[1] for p in spec_pairs_g])
spec_corr_g = spearman(sp_xg, sp_yg)
c_write_v = ('glm4_write_functional'
             if c_gates['write']
             >= PORT_NORM_GATE
             else 'glm4_write_not')
c_port_v = ('glm4_port_functional'
            if c_gates['port']
            >= PORT_NORM_GATE
            else 'glm4_port_not')
c_spec_v = ('glm4_spectrum_aligned'
            if spec_corr_g >= SPEC_CORR_GATE
            else 'glm4_spectrum_weak')
log('C-GATES: %s | %s | %s (spec r=%.3f)'
    % (c_write_v, c_port_v, c_spec_v,
       spec_corr_g))

# cross-model relative-depth alignment:
# normalized effect profiles interpolated onto
# a common depth grid
def depth_profile(ABL, dm_final, NL, med_c):
    xs = []
    ys = []
    for (l, tag) in ABL:
        vals = []
        for dc in ('P', 'A1'):
            vals.extend(np.abs(
                dm_final['%s_L%02d' % (dc, l)]))
        xs.append((l + 0.5) / NL)
        ys.append(float(np.median(vals))
                  / max(med_c, 1e-12))
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    return xs, ys


xs_q, ys_q = depth_profile(ABL_Q, dm_final_q,
                           NLQ, med_c_q)
xs_g, ys_g = depth_profile(ABL_G, dm_final_g,
                           NLG, med_c_g)
grid = np.linspace(0.0, 1.0, 21)
yq_i = np.interp(grid, xs_q, ys_q)
yg_i = np.interp(grid, xs_g, ys_g)
depth_corr = spearman(yq_i, yg_i)
depth_v = ('port_depth_aligned'
           if depth_corr >= DEPTH_CORR_GATE
           else 'port_depth_divergent')
log('C-DEPTH: profile corr=%.3f -> %s'
    % (depth_corr, depth_v))

# ================================================================
# PART D: full-672 counterfactual regeneration
# ================================================================
log('== PART D: full regen ==')


def pad12(ids):
    ids2 = list(ids)[:N_NEW]
    return ids2 + [DOT_G] * (N_NEW
                             - len(ids2))


def gen_from_ids(id_lists):
    outs = []
    id_lists = [list(PREFIX_IDS) + list(p)
                for p in id_lists]
    for b0 in range(0, len(id_lists),
                    GEN_BATCH):
        chunk = id_lists[b0:b0 + GEN_BATCH]
        maxlen = max(len(p) for p in chunk)
        ids = np.full((len(chunk), maxlen),
                      PADG, dtype=np.int64)
        mask = np.zeros((len(chunk), maxlen),
                        dtype=np.int64)
        for i, p in enumerate(chunk):
            ids[i, maxlen - len(p):] = p
            mask[i, maxlen - len(p):] = 1
        t_ids = torch.tensor(ids, device='cuda')
        t_mask = torch.tensor(mask,
                              device='cuda')
        with torch.inference_mode():
            outg = model_g.generate(
                input_ids=t_ids,
                attention_mask=t_mask,
                max_new_tokens=N_NEW,
                do_sample=False, num_beams=1,
                pad_token_id=PADG)
        newg = outg[:, maxlen:]
        for row in newg:
            ids_r = [int(t) for t in row]
            if PADG in ids_r:
                ids_r = ids_r[:ids_r.index(
                    PADG)]
            for e in model_g.config.eos_token_id \
                    if isinstance(
                        model_g.config.eos_token_id,
                        list) else [
                model_g.config.eos_token_id]:
                if e in ids_r:
                    ids_r = ids_r[:ids_r.index(
                        e)]
                    break
            outs.append(ids_r)
    return outs


def pol_raw(tok_id):
    """(polarity, cap) from decoded token:
    polarity +1 yes* / -1 no* / 0 other; cap=1
    when the negative token starts with capital
    N ('No'), 0 for lower-case 'no'."""
    if not tok_id:
        return (0, 0)
    t = tok_g.decode([int(tok_id)])
    ts = t.strip()
    tl = ts.lower()
    if tl.startswith('yes'):
        return (1, 0)
    if tl.startswith('no'):
        return (-1, 1 if ts.startswith('No')
                else 0)
    return (0, 0)


# s0 replay probe (8 pairs, batch=8: NOT batch-
# identical to 3126 -> token-agree gate, not
# bit gate)
probe_ids = [matg['pids_all']['P'][j]['s0']
             for j in range(8)]
probe_out = gen_from_ids(probe_ids)
agree_tok = 0
n_tok = 0
for j in range(8):
    b12 = pad12(z26['gen_base_P'][j])
    g12 = pad12(probe_out[j])
    agree_tok += sum(int(a == b)
                     for a, b in zip(g12, b12))
    n_tok += N_NEW
s0_agree = agree_tok / float(n_tok)
s0_probe_v = ('baseline_s0_replay_ok'
              if s0_agree >= 0.95
              else 'baseline_s0_drift')
log('D-S0PROBE: token agree %.4f -> %s'
    % (s0_agree, s0_probe_v))

# full s1-s3 regeneration (s0 frozen from
# p124); degrade rule after s1 timing
SCOND3 = ('s1', 's2', 's3')
regen_full = {dc: {} for dc in ('P', 'A1')}
regen_idx = {dc: {} for dc in ('P', 'A1')}
degraded = False
t_b = None
for dc in ('P', 'A1'):
    for c in SCOND3:
        if (c == 's2' and t_b is not None
                and not degraded
                and NP_REG == NP):
            proj_s = 4 * (NP_REG // GEN_BATCH) \
                * t_b
            if proj_s > 6.5 * 3600:
                degraded = True
                log('D-DEGRADE: projected %.0fs '
                    '> 6.5h -> even-pair subset '
                    'for s2/s3' % proj_s)
        if degraded and c in ('s2', 's3'):
            idx = list(range(0, NP_REG, 2))
        else:
            idx = list(range(NP_REG))
        regen_idx[dc][c] = idx
        idl = [matg['pids_all'][dc][j][c]
               for j in idx]
        tb0 = time.time()
        regen_full[dc][c] = gen_from_ids(idl)
        if t_b is None:
            t_b = (time.time() - tb0) \
                / max(1, NP_REG // GEN_BATCH)
            log('D-TIMING: per-batch %.1fs '
                '(s1 %s)' % (t_b, dc))
        log('D gen %s %s done (%.1fs)'
            % (dc, c, time.time() - T0))

# bit-exact assertion: first 96 pairs are
# batch-composition-identical to 3126 (3x32)
# -> token-exact vs p124 regen (s2/s3 only
# when not degraded; skipped in SMOKE where
# batch composition necessarily differs)
bit_mism = 0
if not SMOKE:
    bit_conds = ['s1'] + ([] if degraded
                          else ['s2', 's3'])
    for dc in ('P', 'A1'):
        for c in bit_conds:
            ref = z26['regen_%s_%s' % (c, dc)]
            for j in range(min(96, NP_REG)):
                g12 = pad12(
                    regen_full[dc][c][j])
                r12 = [int(v) for v in ref[j]]
                if g12 != r12:
                    bit_mism += 1
    assert bit_mism == 0, \
        ('regen bit mismatch', bit_mism)
    regen_bit_v = 'regen_replay_bit_exact'
    log('D-BITEXACT: %s token-exact vs p124 '
        '(mism=0)' % bit_conds)
else:
    regen_bit_v = 'regen_bit_smoke_skip'
    log('D-BITEXACT: skipped in SMOKE '
        '(batch composition differs)')

# polarity / flip / multi-flip stats
d_stats = {}
flip_all_neg = {dc: {c: [] for c in SCOND3}
                for dc in ('P', 'A1')}
flip_no_only = {dc: {c: [] for c in SCOND3}
                for dc in ('P', 'A1')}
n_no_cap = 0
n_no_low = 0
for dc in ('P', 'A1'):
    agree = {c: [] for c in SCOND3}
    fdiv = {c: [] for c in SCOND3}
    flip = {c: [] for c in SCOND3}
    mflip = {c: [] for c in SCOND3}
    for c in SCOND3:
        idx = regen_idx[dc][c]
        for jj in range(len(idx)):
            j = idx[jj]
            base12 = pad12(
                z26['gen_base_%s' % dc][j])
            bpol, _ = pol_raw(base12[0])
            g12 = pad12(regen_full[dc][c][jj])
            ag = float(np.mean(
                [a == b for a, b
                 in zip(g12, base12)]))
            agree[c].append(ag)
            fd = N_NEW
            for k in range(N_NEW):
                if g12[k] != base12[k]:
                    fd = k
                    break
            fdiv[c].append(fd)
            gpol, cap = pol_raw(g12[0])
            if cap and gpol < 0:
                n_no_cap += 1
            elif gpol < 0:
                n_no_low += 1
            if bpol != 0 and gpol != 0 \
                    and gpol != bpol:
                flip[c].append(1.0)
                flip_all_neg[dc][c].append(1.0)
                flip_no_only[dc][c].append(
                    1.0 if cap == 1 else 0.0)
            else:
                flip[c].append(0.0)
                flip_all_neg[dc][c].append(0.0)
                flip_no_only[dc][c].append(0.0)
            # multi-flip within first 3 tokens
            pols = []
            for k in range(3):
                p_k, _ = pol_raw(g12[k])
                if p_k != 0:
                    pols.append(p_k)
            ch = sum(1 for a, b in
                     zip(pols, pols[1:])
                     if a != b)
            mflip[c].append(
                1.0 if ch >= 1 else 0.0)
    cs = {}
    for c in SCOND3:
        cs[c] = {
            'agree_mean': float(
                np.mean(agree[c])),
            'first_div_mean': float(
                np.mean(fdiv[c])),
            'flip_rate': float(
                np.mean(flip[c])),
            'multi_flip_rate': float(
                np.mean(mflip[c]))}
    d_stats[dc] = cs
    log('D-STATS %s: %s'
        % (dc, {c: {k: round(v, 4)
                    for k, v in cs[c].items()}
                for c in SCOND3}))
shift_min_full = min(
    d_stats[dc]['s1']['agree_mean']
    - float(np.mean([d_stats[dc][c]
                     ['agree_mean']
                     for c in SCOND3]))
    for dc in ('P', 'A1'))
shift_v = ('behavior_shift_full'
           if shift_min_full >= 0.10
           else 'behavior_shift_weak')
log('D-GATE: shift_min_full=%.4f -> %s'
    % (shift_min_full, shift_v))
# flip polarity separation robustness: all-neg
# vs No-only flip rates must agree
diffs = []
for dc in ('P', 'A1'):
    for c in SCOND3:
        a_all = float(np.mean(
            flip_all_neg[dc][c]))
        a_no = float(np.mean(
            flip_no_only[dc][c]))
        diffs.append(abs(a_all - a_no))
flip_sep = max(diffs) if diffs else 0.0
flip_v = ('flip_polarity_separated'
          if flip_sep <= 0.05
          else 'flip_polarity_mixed')
coverage_v = ('coverage_full'
              if NP_REG >= NP
              else 'coverage_degraded')
log('D-GATES: %s | %s | %s (max flip diff '
    '=%.4f, No cap/low=%d/%d)'
    % (shift_v, flip_v, coverage_v, flip_sep,
       n_no_cap, n_no_low))

# ================================================================
# verdict + dumps
# ================================================================
verdict = '|'.join([
    a_lag_v, a_tr_v, a_cross_v, a_cm_v,
    a_sp_v, 'qwen_path_valid', b_write_v,
    b_port_v, b_spec_v, 'glm4_path_valid',
    c_write_v, c_port_v, c_spec_v, depth_v,
    s0_probe_v, regen_bit_v, shift_v,
    flip_v, coverage_v])
assert len(verdict.split('|')) == 19
log('VERDICT: %s' % verdict)

result = {
    'phase': 3127,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'runtime_s': float(time.time() - T0),
    'verdict': verdict,
    'part_a': {
        'a1_closure': a1clo,
        'cross_second': second2,
        'pca': {
            'pc1_share': pc1_share,
            'var_top5': [float(v) for v
                         in var_exp[:5]],
            'corr_pc1_m': corr_pc1_m,
            'corr_pc1_dir': corr_pc1_dir},
        'sparse_events': {
            'rate': float(ev_rate),
            'top_share': float(ev_share_top),
            'counts': [int(v)
                       for v in ev_counts],
            'layers': [int(v) for v in LAY5]}},
    'part_b': {
        'repro_max_abs': drep_q,
        'path_r_min': path_q_min,
        'gates': {k: float(v) for k, v
                  in b_gates.items()},
        'spec_corr': spec_corr_q,
        'ctrl_median': float(med_c_q),
        'dm_final': {k: [float(x) for x
                         in np.abs(v)]
                     for k, v
                     in dm_final_q.items()}},
    'part_c': {
        'repro_max_abs': drep_g,
        'path_r_min': path_g_min,
        'gates': {k: float(v) for k, v
                  in c_gates.items()},
        'spec_corr': spec_corr_g,
        'ctrl_median': float(med_c_g),
        'dm_final': {k: [float(x) for x
                         in np.abs(v)]
                     for k, v
                     in dm_final_g.items()}},
    'depth': {
        'xs_q': [float(v) for v in xs_q],
        'ys_q': [float(v) for v in ys_q],
        'xs_g': [float(v) for v in xs_g],
        'ys_g': [float(v) for v in ys_g],
        'corr': depth_corr},
    'part_d': {
        's0_probe_agree': float(s0_agree),
        'bit_mismatch': int(bit_mism),
        'stats': d_stats,
        'shift_min_full': float(
            shift_min_full),
        'flip_sep_max': float(flip_sep),
        'n_no_cap': int(n_no_cap),
        'n_no_low': int(n_no_low),
        'np_reg': NP_REG}}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)
npz_out = {}
for k, v in dm_field_q.items():
    npz_out['dmq_' + k] = v
for k, v in dm_field_g.items():
    npz_out['dmg_' + k] = v
for dc in ('P', 'A1'):
    npz_out['wspecQ_%s' % dc] = \
        wspec_q[dc].astype(np.float32)
    npz_out['wprofQ_pd_%s' % dc] = \
        wp_q[dc].astype(np.float32)
for dc in ('P', 'A1'):
    for c in SCOND3:
        idx = regen_idx[dc][c]
        arr = np.full((len(idx), N_NEW), DOT_G,
                      dtype=np.int32)
        for jj in range(len(idx)):
            g12 = pad12(regen_full[dc][c][jj])
            arr[jj] = g12
        npz_out['regen_%s_%s' % (c, dc)] = arr
        npz_out['regenidx_%s_%s' % (c, dc)] = \
            np.array(idx, dtype=np.int64)
NPZF = os.path.join(OUT, 'p125_readout.npz')
np.savez(NPZF, **npz_out)
log('dumps: result.json + p125_readout.npz '
    '(%d keys)' % len(npz_out))
log('RUNTIME %.1fs' % (time.time() - T0))
print('PHASE3127_DONE verdict=%s' % verdict)




