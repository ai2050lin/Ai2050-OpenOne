"""Phase 3125 (Omega-P123): residual THIRD-component
localization (offline: trail kernel + AR structure +
trajectory persistence + deterministic re-simulation)
+ Qwen3-4B input-stream interference control (GPU).

Inputs (frozen): phase3118 traj npz, phase3120 ann
npz, phase3122 result + npz, phase3123 result + npz,
phase3124 result + npz, phase3105 material.json,
phase3113 capture_b.npz.

Parts:
  A offline (full 672): inherit 3124 refit/resid
    (assert vs 3124); sequential decomposition
    step-dummy -> m-structure -> trail kernel (D=3)
    -> AR(2) -> remainder; trajectory-persistence
    permutation test (deterministic rng 3125, 200
    perms); full deterministic re-simulation with
    the third component -> auc_sim3 vs auc18.
  B gpu (qwen3-4b): INPUT-stream interference
    control. s1/s2/s3 replace the PROMPT copy of
    the QUERY line (located by anchored LAST
    occurrence: 3124 used first occurrence which
    under P collides with the identical Facts
    line - fixed here), generated stream (3118
    gen_clean) replayed unchanged via teacher
    forcing; 37-state logit lens w_dn margins;
    E_syn/E_cont curves (all-position means,
    3124-GLM4 isomorphic); rel+abs L* gates; path
    FATAL gate; s0 repro bit-exact gate; readout
    AUC; final-layer SIGN separation across
    {3123 qwen output-stream, 3124 glm4
    input-stream, 3125 qwen input-stream}.
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
NAME = 'omega_p123_third_comp_qwen_inputstream'
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
D24 = RDIR + r'\phase3124' \
      r'\omega_p122_resid_cm_kalman_l35rel_' \
      'glm4x'
MDIR = os.path.join(ROOT, 'models', 'hf',
                    'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3125', NAME)
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
NP_CHECK = 8 if SMOKE else 32
NP_REPRO = 4 if SMOKE else 16
TRAIL_D = 3
N_PERM = 200
PERM_SEED = 3125

V23 = ('dirfit_failed|anchor_failed|'
       'pit_calibrated|pit_calibrated|'
       'anchor_separation_present|'
       'anchor_unreliable|final_brake_global|'
       'assertion_write_global_positive|'
       'replay_bit_exact|syntax_emerges_L21|'
       'syntax_emerges_L21|'
       'content_emerges_L20|'
       'content_emerges_L20')
V24 = ('common_mode_partial|det_core_failed|'
       'r2_weak|lag1_negative|anchor_static|'
       'own_anchor_insufficient|'
       'semantic_component_present|path_valid|'
       'replay_bit_exact|readout_ok|spans_ok|'
       'glm_syntax_L20|glm_syntax_L20|'
       'glm_content_L20|glm_content_L20')

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
res22 = json.load(io.open(
    os.path.join(D22, 'result.json'),
    encoding='utf-8'))
assert res22['verdict'] == \
    'mixed|write_polarity_diverged|' \
    'replay_bit_exact|' \
    'sentence_content_push_down|' \
    'sentence_content_push_down|' \
    'syntax_effect_present|' \
    'syntax_effect_present|' \
    'oscillation_persist|' \
    'distribution_reconstruction_failed|' \
    'pit_marginal'
assert res22['smoke'] is False
res23 = json.load(io.open(
    os.path.join(D23, 'result.json'),
    encoding='utf-8'))
assert res23['verdict'] == V23
assert res23['smoke'] is False
res24 = json.load(io.open(
    os.path.join(D24, 'result.json'),
    encoding='utf-8'))
assert res24['verdict'] == V24
assert res24['smoke'] is False
pa24 = res24['part_a']
assert abs(pa24['decomp']['P']['cm_share']
           - 0.19001305759179912) < 1e-12
assert abs(pa24['det_sim']['auc_det'][12]
           - 0.9714405293367347) < 1e-12
assert abs(res24['part_b']['lag1']['P']
           - (-0.1674141723780209)) < 1e-12
assert res24['part_d']['lstar']['P']['syn'] \
    == {'rel': 20, 'abs': 20}
assert abs(res24['part_d']['readout_auc']
           - 0.8313137755102041) < 1e-12
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
z23 = np.load(os.path.join(D23, 'p121_readout.npz'),
              allow_pickle=False)
z24 = np.load(os.path.join(D24, 'p122_readout.npz'),
              allow_pickle=False)
log('frozen inputs ok (18/20/22/23/24)')

# ================================================================
# design seal (frozen BEFORE observation)
# ================================================================
seal = {
    'phase': 3125,
    'name': NAME,
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'np_a': NP,
    'np_b': NP_B,
    'data_sources': {
        'margins_tokens_gen': 'phase3118 '
                              'traj_readout.npz '
                              '(gt_cleanseq_clean, '
                              'gen_clean, auc_curve)',
        'annotation': 'phase3120 annP_c/annA_c',
        'fits_3124': 'phase3124 result.json '
                     'part_a.refit (assert '
                     '1e-12) + p122_readout.npz '
                     'anchors (assert 1e-5)',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz'},
    'part_a': {
        'inherit': 'content table -> dm_res -> '
                   'lstsq [m,1] -> S,MS (assert '
                   'vs 3124 1e-12); resid = '
                   'dm_res - [m,1]@b',
        'decomp3_seq': 'R -> Z1 step-dummy -> r1 '
                       '-> Z2 [m-MS,1] -> r2 -> '
                       'trail kernel X(30: '
                       'cls[k-d]*3+(d-1), d=1..3, '
                       'missing rows zero) -> r3 '
                       '-> AR(2) on k>=2 rows '
                       '[r3[k-1],r3[k-2],1] -> r4 '
                       '(r4=r3 on k<2); shares '
                       'relative to ss_tot(R)',
        'trail_gate': 'ss_trail/ss_tot >= 0.05 -> '
                      'trail_present else '
                      'trail_absent',
        'ar_gate': 'ss_ar/ss_tot >= 0.05 -> '
                   'ar_present else ar_absent',
        'traj_gate': 'rho_within = pooled pearson '
                     'r4[:, :-1] vs r4[:, 1:]; '
                     'null: row permutation (200 '
                     'reps, numpy default_rng(3125), '
                     'deterministic); '
                     'rho_within >= 0.1 AND > '
                     'perm_mean + 4*perm_std -> '
                     'trajectory_persistent else '
                     'trajectory_transient',
        'sim3': 'deterministic no-noise: m(t+1)='
                'm+S*(m-MS)+content[cls_k]+'
                'trail(G[cls_{k-1..k-3}])+'
                'AR(e_k=phi1*e_{k-1}+phi2*e_{k-2}'
                '+c, e0=e1=0); gate r(auc_sim3, '
                'auc18) >= 0.5 third_sufficient; '
                '>= 0.3 third_partial; else '
                'third_insufficient',
        'no_mc': 'only deterministic row '
                 'permutations (seed 3125)'},
    'part_b': {
        'model': 'qwen3-4b (36L, BF16, eager, '
                 'batch1 forwards)',
        'hs_trap': 'transformers 5.14 qwen3 '
                   'hidden_states[NL] ALREADY '
                   'final-normed (probe: raw '
                   'r=0.99999 vs logit diff, '
                   're-norm r=0.854); norm applied '
                   'only to L<NL; 3122/3123 L36 '
                   'was double-normed (L0..35 '
                   'correct, L* unaffected)',
        'interference': 'input_prompt_span_'
                        'equal_len_anchored_last',
        'span_fix': 'query line located by '
                    "suffix-anchored LAST "
                    "occurrence (' Is this' after "
                    'the line); 3124 find() first '
                    'occurrence collides with the '
                    'identical Facts line under P '
                    '- documented as 3124 caveat',
        'gen_replay': '3118 gen_clean teacher-'
                      'forced unchanged (input-'
                      'stream only)',
        'curves': 'E_syn/E_cont = mean over ALL '
                  '13 readout positions of '
                  'd1-d2 / d1-d3 (3124-GLM4 '
                  'isomorphic)',
        'gates': {
            'B_path': 'corr(manual-norm margin, '
                      'model-logits margin) > '
                      '0.9999 on first NP_CHECK '
                      'pairs -> path_valid else '
                      'FATAL',
            'B_repro': 's0 forward twice, max '
                       'diff == 0 on first '
                       'NP_REPRO pairs -> '
                       'replay_bit_exact; <1e-6 '
                       'replay_ok; else FATAL',
            'B_readout': 'final-step margin AUC '
                         '(P vs A1) > 0.6 -> '
                         'readout_ok else warn',
            'B_spans': 'n_span >= 300 BOTH dirs '
                       '-> spans_ok; >= 100 both '
                       '-> spans_sparse; else '
                       'spans_absent (SMOKE: '
                       'max(1, NP_B//2))',
            'B_lstar': 'L* = min L>=20 with '
                       'E(L) <= -0.3*|E(final)| '
                       '(rel) and <= -0.05/-0.025 '
                       '(abs), 37 states',
            'B_sign': 'sign(E(NL)) final-layer '
                      'polarity per effect per '
                      'dir -> qis_{syn|cont}_'
                      'final_{positive|negative}'},
        'no_mc': 'only crc32-seeded entity '
                 'choice/shuffle (deterministic)'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('seal frozen (%s)' % seal['created'])

# ================================================================
# shared frozen arrays + refit (assert vs 3124)
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
# PART A: third-component decomposition
# ================================================================
log('== PART A: third-component localization ==')
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
               'resid': resid, 'col': col}
    r24f = pa24['refit'][dc]
    assert abs(S_dc - r24f['S']) < 1e-12, dc
    assert abs(MS_dc - r24f['MS']) < 1e-12, dc
    a24 = z24['anchors_%s' % dc].astype(
        np.float64)
    assert np.max(np.abs(a_i - a24)) < 1e-5, dc
log('A refit assert vs 3124 ok')

dec3 = {}
ar_par = {}
G_tab = {}
traj_stat = {}
sim3 = {}
for dc in ('P', 'A1'):
    R = fit[dc]['resid']
    ss_tot = float((R * R).sum())
    NIK = R.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, R.ravel(),
                             rcond=None)
    r1 = (R.ravel() - Z1 @ b1).reshape(NP, N_NEW)
    ss_step = ss_tot - float((r1 * r1).sum())
    x = (MSEQ[dc][:, :-1]
         - fit[dc]['MS']).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1.ravel(),
                             rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(NP, N_NEW)
    ss_m = float((r1 * r1).sum()) \
        - float((r2 * r2).sum())
    # trail kernel (D=3)
    X_tr = np.zeros((NIK, 10 * TRAIL_D))
    col = fit[dc]['col']
    for d in range(1, TRAIL_D + 1):
        rows_k = np.arange(d, N_NEW)
        idx_rows = (np.repeat(
            np.arange(NP) * N_NEW, len(rows_k))
            + np.tile(rows_k, NP))
        cls_d = col[:, :N_NEW - d].ravel()
        X_tr[idx_rows,
             cls_d * TRAIL_D + (d - 1)] = 1.0
    btr, *_ = np.linalg.lstsq(X_tr, r2.ravel(),
                              rcond=None)
    r3 = (r2.ravel() - X_tr @ btr).reshape(
        NP, N_NEW)
    ss_trail = float((r2 * r2).sum()) \
        - float((r3 * r3).sum())
    # AR(2) on k>=2 rows
    y_ar = r3[:, 2:].ravel()
    X_ar = np.stack([r3[:, 1:N_NEW - 1].ravel(),
                     r3[:, 0:N_NEW - 2].ravel(),
                     np.ones(y_ar.size)], 1)
    bar, *_ = np.linalg.lstsq(X_ar, y_ar,
                              rcond=None)
    phi1, phi2, c_ar = (float(bar[0]),
                        float(bar[1]),
                        float(bar[2]))
    r4 = r3.copy()
    pred = (phi1 * r3[:, 1:N_NEW - 1]
            + phi2 * r3[:, 0:N_NEW - 2]
            + c_ar)
    r4[:, 2:] = r3[:, 2:] - pred
    ss_ar = float(((r3[:, 2:] - r4[:, 2:])
                   ** 2).sum())
    ss_rem = float((r4 * r4).sum())
    leak = (1.0 - ss_step / ss_tot
            - ss_m / ss_tot
            - ss_trail / ss_tot
            - ss_ar / ss_tot
            - ss_rem / ss_tot)
    dec3[dc] = {
        'ss_tot': ss_tot,
        'cm_share': ss_step / ss_tot,
        'm_share': ss_m / ss_tot,
        'trail_share': ss_trail / ss_tot,
        'ar_share': ss_ar / ss_tot,
        'rem_share': ss_rem / ss_tot,
        'leak': leak}
    G_tab[dc] = btr.reshape(10, TRAIL_D)
    ar_par[dc] = {'phi1': phi1, 'phi2': phi2,
                  'c': c_ar}
    # trajectory persistence
    xx = r4[:, :-1].ravel()
    yy = r4[:, 1:].ravel()
    rho_w = float(np.corrcoef(xx, yy)[0, 1])
    rg = np.random.default_rng(PERM_SEED)
    rho_p = np.zeros(N_PERM)
    for rep in range(N_PERM):
        pm = rg.permutation(NP)
        rho_p[rep] = float(np.corrcoef(
            r4[pm, :-1].ravel(), yy)[0, 1])
    traj_stat[dc] = {
        'rho_within': rho_w,
        'rho_perm_mean': float(rho_p.mean()),
        'rho_perm_std': float(rho_p.std()),
        'perm_reps': N_PERM}
log('A decomposition pass done')

auc_sim3 = np.zeros(N_NEW + 1)
sim_arr = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    col = fit[dc]['col']
    S_dc = fit[dc]['S']
    MS_dc = fit[dc]['MS']
    phi1 = ar_par[dc]['phi1']
    phi2 = ar_par[dc]['phi2']
    c_ar = ar_par[dc]['c']
    mh = m[:, 0].copy()
    snap = [mh.copy()]
    e1 = np.zeros(NP)
    e2 = np.zeros(NP)
    for k in range(N_NEW):
        t_tr = np.zeros(NP)
        if k >= 1:
            t_tr += G_tab[dc][col[:, k - 1], 0]
        if k >= 2:
            t_tr += G_tab[dc][col[:, k - 2], 1]
        if k >= 3:
            t_tr += G_tab[dc][col[:, k - 3], 2]
        e_k = phi1 * e1 + phi2 * e2 + c_ar
        mh = mh + S_dc * (mh - MS_dc) \
            + content[dc][col[:, k], k] \
            + t_tr + e_k
        snap.append(mh.copy())
        e2 = e1
        e1 = e_k
    sim_arr[dc] = np.stack(snap, 1)
for t in range(N_NEW + 1):
    auc_sim3[t] = auc_mw(sim_arr['P'][:, t],
                         sim_arr['A1'][:, t])
r3v = float(np.corrcoef(auc_sim3, auc18)[0, 1])
a_sim_v = ('third_sufficient' if r3v >= 0.5
           else ('third_partial' if r3v >= 0.3
                 else 'third_insufficient'))
log('A-SIM3: r=%.4f vs det 3124 r=-0.123 -> %s'
    % (r3v, a_sim_v))

cm_min = min(dec3['P']['cm_share'],
             dec3['A1']['cm_share'])
a_cm_v = ('common_mode_dominant' if cm_min >= 0.2
          else ('common_mode_partial'
                if cm_min >= 0.05
                else 'iid_like'))
tr_min = min(dec3['P']['trail_share'],
             dec3['A1']['trail_share'])
a_trail_v = ('trail_present' if tr_min >= 0.05
             else 'trail_absent')
ar_min = min(dec3['P']['ar_share'],
             dec3['A1']['ar_share'])
a_ar_v = ('ar_present' if ar_min >= 0.05
          else 'ar_absent')
tw = traj_stat
tw_min = min(tw['P']['rho_within'],
             tw['A1']['rho_within'])
tw_ok = True
for dc in ('P', 'A1'):
    st = tw[dc]
    if not (st['rho_within'] >= 0.1
            and st['rho_within']
            > st['rho_perm_mean']
            + 4.0 * st['rho_perm_std']):
        tw_ok = False
a_traj_v = ('trajectory_persistent' if tw_ok
            else 'trajectory_transient')
log('A-DECOMP P: cm=%.4f m=%.4f trail=%.4f '
    'ar=%.4f rem=%.4f'
    % (dec3['P']['cm_share'],
       dec3['P']['m_share'],
       dec3['P']['trail_share'],
       dec3['P']['ar_share'],
       dec3['P']['rem_share']))
log('A-DECOMP A1: cm=%.4f m=%.4f trail=%.4f '
    'ar=%.4f rem=%.4f'
    % (dec3['A1']['cm_share'],
       dec3['A1']['m_share'],
       dec3['A1']['trail_share'],
       dec3['A1']['ar_share'],
       dec3['A1']['rem_share']))
log('A-TRAJ: rho_w P=%.4f A1=%.4f -> %s'
    % (tw['P']['rho_within'],
       tw['A1']['rho_within'], a_traj_v))

# ================================================================
# PART B: GPU qwen input-stream control
# ================================================================
log('== PART B: qwen input-stream ==')
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
assert len(pks) == 672
texts = {}
PID = {}
hmap = {'P': hP, 'A1': hA1}
for dc in ('P', 'A1'):
    texts[dc] = {}
    PID[dc] = []
    for pk in pks:
        (s, o) = (int(v) for v in pk.split('_'))
        r = p2r[pk]
        ri1, ri2 = frel[pk]
        if dc == 'P':
            (qrel, lrel) = (r, r)
        else:
            (qrel, lrel) = (r, ri1)
        t_ = build_prompt(mat5, s, o, lrel, qrel)
        texts[dc][pk] = t_
        PID[dc].append(t_)

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


def forward_trackL_q(prompt_ids, gen_tokens,
                     want_logits=False):
    """teacher-forced forward (prompt possibly
    span-replaced + gen replayed unchanged);
    w_dn margins at every hidden state 0..NL.
    PATH PROBE 3125: transformers 5.14 qwen3
    hidden_states[NL] ALREADY carries the final
    norm (raw r=0.99999 vs logit diff; re-norm
    gives double-norm r=0.854) - norm is applied
    ONLY to L<NL here. NOTE: 3122/3123 curves
    used norm() on all L incl. NL -> their L36
    point is double-normed (L0..L35 correct,
    L* verdicts unaffected)."""
    gen_tokens = [int(x) for x in gen_tokens]
    ids = list(prompt_ids) + gen_tokens
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(gen_tokens) + 1
    ml = np.zeros((NL + 1, npts),
                  dtype=np.float64)
    lmv = None
    with torch.inference_mode():
        out = model(t_in,
                    output_hidden_states=True,
                    use_cache=False)
        for L in range(NL + 1):
            hL = out.hidden_states[L][0]
            if L < NL:
                hL = norm_mod(hL)
            h = hL.float().cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dn)
        if want_logits:
            lg = out.logits[0,
                            pos0:pos0 + npts] \
                .float().cpu().numpy()
            lmv = lg[:, YES_ID] - lg[:, NO_ID]
        del out
    return ml, lmv


PID_T = {dc: [tok(t, add_special_tokens=False)
              ['input_ids'] for t in texts[dc]
              .values()]
         for dc in ('P', 'A1')}
g18 = {'P': z18['gen_clean__P'],
       'A1': z18['gen_clean__A1']}
DOT_ID = int(tok('.', add_special_tokens=False)
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
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    lrel = r if dcode == 'P' else ri1
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
    raise RuntimeError('no replacement')


def traj_tokens_q(dc, j):
    ids_r = [int(t) for t in g18[dc][j]]
    ptxt = texts[dc][pks[j]]
    encp = tok(ptxt, add_special_tokens=False,
               return_offsets_mapping=True)
    poffs = [tuple(v) for v in
             encp['offset_mapping']]
    ids2 = list(ids_r)[:N_NEW]
    while len(ids2) < N_NEW:
        ids2.append(DOT_ID)
    return ptxt, poffs, ids2


def find_span_q(dec, offs, s, o, qrel):
    """Anchored LAST occurrence: query line is
    followed by ' Is this'. Fixes the 3124
    first-occurrence Facts-line collision under P."""
    qline = 'The %s %s the %s.' % (
        ents_all[s], PREDS_all[qrel],
        ents_all[o])
    ce = dec.rfind(qline)
    while ce != -1:
        cs = ce
        ce_want = cs + len(qline)
        if dec[ce_want:ce_want + 8] == ' Is this':
            idxs = [k for k in range(len(offs))
                    if offs[k][0] >= cs
                    and offs[k][1] <= ce_want
                    and offs[k][1]
                    > offs[k][0]]
            if len(idxs) >= 2 \
                    and idxs[0] >= 1 \
                    and idxs[-1] - idxs[0] + 1 \
                    <= N_NEW:
                return idxs[0], idxs[-1]
        ce = dec.rfind(qline, 0, ce)
    return None


# span sanity probe: first vs anchored-last
for dc in ('P', 'A1'):
    pk0 = pks[0]
    (s0_, o0_) = (int(v)
                  for v in pk0.split('_'))
    qrel0 = p2r[pk0]
    ptxt0, poffs0, _ = traj_tokens_q(dc, 0)
    ql0 = 'The %s %s the %s.' % (
        ents_all[s0_], PREDS_all[qrel0],
        ents_all[o0_])
    n_occ = ptxt0.count(ql0)
    log('B-SPANPROBE %s %s: occurrences=%d '
        'first=%d anchored=%d'
        % (dc, pk0, n_occ, ptxt0.find(ql0),
           ptxt0.rfind(ql0)))

SCOND = ('s0', 's1', 's2', 's3')
MLG = {d: {c: None for c in SCOND}
       for d in ('P', 'A1')}
t0b = time.time()
span_count = {}
span_idx_store = {}
path_rs = []
path_ds = []
drep = 0.0
for dcode in ('P', 'A1'):
    store = {c: [] for c in SCOND}
    n_span = 0
    n_done = 0
    sidx = np.full((NP_B, 2), -1,
                   dtype=np.int16)
    for j in range(NP_B):
        pk = pks[j]
        prompt_ids0 = list(PID_T[dcode][j])
        ptxt, poffs, base = traj_tokens_q(
            dcode, j)
        (s, o) = (int(v) for v in
                  pk.split('_'))
        r = p2r[pk]
        span = find_span_q(ptxt, poffs,
                           s, o, r)
        pids = {c: list(prompt_ids0)
                for c in SCOND}
        if span is not None:
            (k1, k2) = span
            Lspan = k2 - k1 + 1
            sidx[j] = (k1, k2)
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
            pids['s1'][k1:k2 + 1] = sub
            pids['s2'][k1:k2 + 1] = shuf
            pids['s3'][k1:k2 + 1] = \
                [DOT_ID] * Lspan
            n_span += 1
        for c in SCOND:
            wl, lm = forward_trackL_q(
                pids[c], base,
                want_logits=(j < NP_CHECK
                             and c == 's0'))
            store[c].append(wl)
            if lm is not None:
                path_rs.append(wl[NL, :].copy())
                path_ds.append(lm)
        if j < NP_REPRO:
            wl2, _ = forward_trackL_q(
                pids['s0'], base)
            drep = max(drep, float(np.abs(
                store['s0'][j] - wl2).max()))
        n_done += 1
        if n_done % 128 == 0:
            log('B %s %d/%d (%.1fs)'
                % (dcode, n_done, NP_B,
                   time.time() - t0b))
    for c in SCOND:
        MLG[dcode][c] = np.array(
            store[c], dtype=np.float64)
    span_idx_store[dcode] = sidx
    log('B %s done n_span=%d (%.1fs)'
        % (dcode, n_span, time.time() - t0b))
    span_count[dcode] = n_span

n_span_g = span_count
pr = np.concatenate(path_rs)
pdd = np.concatenate(path_ds)
if np.std(pdd) > 0:
    r_path = float(np.corrcoef(pr, pdd)[0, 1])
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
log('B-PATH: r=%.6f -> %s; B-REPRO: %.3e -> %s'
    % (r_path, d_path_v, drep, d_repro_v))

auc_fin = auc_mw(
    MLG['P']['s0'][:, NL, -1],
    MLG['A1']['s0'][:, NL, -1])
d_readout_v = 'readout_ok' if auc_fin > 0.6 \
    else 'readout_warn'
log('B-READOUT: final-step AUC=%.4f -> %s'
    % (auc_fin, d_readout_v))
if SMOKE:
    s_ok = max(1, NP_B // 2)
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
log('B-SPANS: P=%d A1=%d -> %s'
    % (n_span_g['P'], n_span_g['A1'],
       d_spans_v))

curves_b = {}
lstar_b = {}
sign_b = {}
if d_spans_v != 'spans_absent':
    for dcode in ('P', 'A1'):
        syn = np.zeros(NL + 1)
        con = np.zeros(NL + 1)
        n_sp = 0
        for j in range(NP_B):
            pk = pks[j]
            (s, o) = (int(v) for v in
                      pk.split('_'))
            ptxt, poffs, _b = traj_tokens_q(
                dcode, j)
            span = find_span_q(
                ptxt, poffs, s, o, p2r[pk])
            if span is None:
                continue
            n_sp += 1
            s0 = MLG[dcode]['s0'][j]
            d1 = MLG[dcode]['s1'][j] - s0
            d2 = MLG[dcode]['s2'][j] - s0
            d3 = MLG[dcode]['s3'][j] - s0
            for L in range(NL + 1):
                syn[L] += float(
                    d1[L].mean()
                    - d2[L].mean())
                con[L] += float(
                    d1[L].mean()
                    - d3[L].mean())
        syn /= max(n_sp, 1)
        con /= max(n_sp, 1)
        curves_b[dcode] = {'syn': syn,
                           'cont': con,
                           'n_span': n_sp}
        lstar_b[dcode] = {}
        sign_b[dcode] = {}
        for nm, cvv in (('syn', syn),
                        ('cont', con)):
            eff = abs(float(cvv[NL]))
            lr = None
            la = None
            for L in range(20, NL + 1):
                if lr is None and eff > 0 \
                        and cvv[L] <= -0.3 * eff:
                    lr = L
                th = 0.05 if dcode == 'P' \
                    else 0.025
                if la is None \
                        and cvv[L] <= -th:
                    la = L
            lstar_b[dcode][nm] = {
                'rel': lr, 'abs': la}
            sign_b[dcode][nm] = (
                'positive'
                if cvv[NL] > 0 else 'negative')
        log('B-TRACE %s: n_span=%d syn rel=%s '
            'abs=%s final=%+.4f | cont rel=%s '
            'abs=%s final=%+.4f'
            % (dcode, n_sp,
               lstar_b[dcode]['syn']['rel'],
               lstar_b[dcode]['syn']['abs'],
               syn[NL],
               lstar_b[dcode]['cont']['rel'],
               lstar_b[dcode]['cont']['abs'],
               con[NL]))
else:
    curves_b = {'P': {'n_span': 0},
                'A1': {'n_span': 0}}
    lstar_b = {'P': {}, 'A1': {}}
    sign_b = {'P': {}, 'A1': {}}


def lstar_v(dc, nm):
    if d_spans_v == 'spans_absent':
        return 'skipped_sparse'
    e = lstar_b.get(dc, {}).get(nm, {})
    lr = e.get('rel')
    tag = 'syntax' if nm == 'syn' else 'content'
    if lr is None:
        return 'qis_%s_below_threshold' % tag
    return 'qis_%s_L%d' % (tag, lr)


def sign_v(dc, nm):
    if d_spans_v == 'spans_absent':
        return 'skipped_sparse'
    return 'qis_%s_final_%s' % (
        'syn' if nm == 'syn' else 'cont',
        sign_b[dc][nm])


d_syn_P_v = lstar_v('P', 'syn')
d_syn_A1_v = lstar_v('A1', 'syn')
d_cont_P_v = lstar_v('P', 'cont')
d_cont_A1_v = lstar_v('A1', 'cont')
d_synP_s_v = sign_v('P', 'syn')
d_synA_s_v = sign_v('A1', 'syn')
d_contP_s_v = sign_v('P', 'cont')
d_contA_s_v = sign_v('A1', 'cont')

# ================================================================
# verdict + save
# ================================================================
verdict = '|'.join([
    a_cm_v, a_trail_v, a_ar_v, a_traj_v,
    a_sim_v, d_path_v, d_repro_v,
    d_readout_v, d_spans_v,
    d_syn_P_v, d_syn_A1_v,
    d_cont_P_v, d_cont_A1_v,
    d_synP_s_v, d_synA_s_v,
    d_contP_s_v, d_contA_s_v])
runtime = round(time.time() - T0, 1)
results = {
    'phase': 3125,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'n_pairs': NP,
    'np_b': NP_B,
    'runtime_s': runtime,
    'part_a': {
        'refit': {
            'P': {'S': fit['P']['S'],
                  'MS': fit['P']['MS']},
            'A1': {'S': fit['A1']['S'],
                   'MS': fit['A1']['MS']}},
        'decomp3': dec3,
        'cm_verdict': a_cm_v,
        'trail_verdict': a_trail_v,
        'ar_verdict': a_ar_v,
        'traj_verdict': a_traj_v,
        'trail_G': {
            dc: [[float(v) for v in row]
                 for row in G_tab[dc]]
            for dc in ('P', 'A1')},
        'ar_params': ar_par,
        'traj_stat': traj_stat,
        'sim3': {
            'r_sim3': r3v,
            'verdict': a_sim_v,
            'r_det_3124_ref':
                -0.12297439326221062,
            'auc_sim3': [float(x) for x
                         in auc_sim3]}},
    'part_b': {
        'interference':
            'input_prompt_span_equal_len_'
            'anchored_last',
        'interference_note': (
            'query line located by '
            "suffix-anchored LAST occurrence "
            "(' Is this' follows the query "
            'line); 3124 used FIRST occurrence '
            'which under P collides with the '
            'identical Facts line - 3125 fix; '
            'generated stream replayed '
            'unchanged (teacher-forced 3118 '
            'gen_clean)'),
        'span_first_vs_last_probe': {
            dc: 'see run_log B-SPANPROBE'
            for dc in ('P', 'A1')},
        'ids': {'yes': YES_ID, 'no': NO_ID,
                'dot': DOT_ID, 'n_layers': NL},
        'path': {'r': r_path,
                 'verdict': d_path_v},
        'repro': {'max_diff': drep,
                  'verdict': d_repro_v},
        'readout_auc': auc_fin,
        'readout_verdict': d_readout_v,
        'n_span': n_span_g,
        'spans_verdict': d_spans_v,
        'lstar': {d: {nm: lstar_b[d].get(
            nm, {'rel': None, 'abs': None})
            for nm in ('syn', 'cont')}
            for d in ('P', 'A1')},
        'sign': {d: dict(sign_b[d])
                 for d in ('P', 'A1')},
        'curves': {
            'E_syn_P': [float(x) for x in
                        curves_b['P']['syn']]
            if 'syn' in curves_b['P'] else [],
            'E_syn_A1': [float(x) for x in
                         curves_b['A1']['syn']]
            if 'syn' in curves_b['A1']
            else [],
            'E_cont_P': [float(x) for x in
                         curves_b['P']['cont']]
            if 'cont' in curves_b['P'] else [],
            'E_cont_A1': [float(x) for x in
                          curves_b['A1']
                          ['cont']]
            if 'cont' in curves_b['A1']
            else []}},
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False,
              indent=1)
npz_out = {
    'auc_sim3': auc_sim3,
    'r4_P': np.zeros((NP, N_NEW)),
    'r4_A1': np.zeros((NP, N_NEW)),
}
for dc in ('P', 'A1'):
    R = fit[dc]['resid']
    ss_tot = float((R * R).sum())
    NIK = R.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, R.ravel(),
                             rcond=None)
    r1 = (R.ravel() - Z1 @ b1).reshape(NP, N_NEW)
    x = (MSEQ[dc][:, :-1]
         - fit[dc]['MS']).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1.ravel(),
                             rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(NP, N_NEW)
    X_tr = np.zeros((NIK, 10 * TRAIL_D))
    col = fit[dc]['col']
    for d in range(1, TRAIL_D + 1):
        rows_k = np.arange(d, N_NEW)
        idx_rows = (np.repeat(
            np.arange(NP) * N_NEW, len(rows_k))
            + np.tile(rows_k, NP))
        cls_d = col[:, :N_NEW - d].ravel()
        X_tr[idx_rows,
             cls_d * TRAIL_D + (d - 1)] = 1.0
    btr, *_ = np.linalg.lstsq(X_tr, r2.ravel(),
                              rcond=None)
    r3 = (r2.ravel() - X_tr @ btr).reshape(
        NP, N_NEW)
    phi1 = ar_par[dc]['phi1']
    phi2 = ar_par[dc]['phi2']
    c_ar = ar_par[dc]['c']
    r4 = r3.copy()
    r4[:, 2:] = r3[:, 2:] - (
        phi1 * r3[:, 1:N_NEW - 1]
        + phi2 * r3[:, 0:N_NEW - 2] + c_ar)
    npz_out['r4_%s' % dc] = r4
    npz_out['span_idx_%s' % dc] = \
        span_idx_store[dc]
    for c in SCOND:
        npz_out['mlg_%s_%s' % (c, dc)] = \
            MLG[dc][c].astype(np.float32)
    if 'syn' in curves_b[dc]:
        npz_out['E_syn_%s' % dc] = \
            curves_b[dc]['syn'].astype(
                np.float32)
        npz_out['E_cont_%s' % dc] = \
            curves_b[dc]['cont'].astype(
                np.float32)
np.savez(os.path.join(OUT, 'p123_readout.npz'),
         **npz_out)
log('verdict=%s' % verdict)
log('done (%.1fs)' % runtime)
print('phase3125 done: %s' % verdict)
