"""Phase 3126 (Omega-P124): GLM4 anchored-last
corrected re-run + counterfactual regeneration
control + write-chain layer identification
(GPU, single model glm4-9b-chat-hf) + trail
deepening (offline).

Inputs (frozen): phase3118 traj npz, phase3120
ann npz, phase3124 result + p122_readout.npz,
phase3125 result, phase3105 material.json,
phase3113 capture_b.npz.

Parts:
  A offline (full 672): inherit 3125 refit
    (assert 1e-12); D=6 long-range trail kernel
    sequential decomposition; within-trajectory
    cls permutation significance (rng 3126, 200
    reps); G-table cross-direction structure;
    second-order decomposition of the remainder
    (joint cls kernel + m-quadratic).
  B gpu (glm4): anchored-LAST span fix (3124
    first-occurrence collided with the identical
    Facts line under P); own-stream base
    generation; s0-s3 input-stream interference;
    41-state hook-collected logit lens (3124
    append-before trap fix); E curves / L* /
    final sign; vs-3124 curve correlation
    (A1 unconfounded must replicate >= 0.9).
  C gpu: counterfactual regeneration subset
    (NP_REG pairs x s0..s3 actually generated
    from the perturbed prompts; agreement /
    first-divergence / answer-flip vs base).
  D offline: write-chain spectrum from s0
    margins (per-layer margin deltas); top-3
    concentration + positive/negative write
    bands; Qwen 3122 (L28-34 pos / L35 neg)
    qualitative reference.
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
NAME = 'omega_p124_glm4_anchoredlast_' \
       'regen_writechain'
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
D24 = RDIR + r'\phase3124' \
      r'\omega_p122_resid_cm_kalman_l35rel_' \
      'glm4x'
D25 = RDIR + r'\phase3125' \
      r'\omega_p123_third_comp_qwen_inputstream'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3126', NAME)
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
NP_REG = 4 if SMOKE else 96
TRAIL_D = 6
N_PERM = 20 if SMOKE else 200
PERM_SEED = 3126
GEN_BATCH = 4 if SMOKE else 32

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
assert abs(res24['part_a']['decomp']['P']
           ['cm_share']
           - 0.19001305759179912) < 1e-12
res25 = json.load(io.open(
    os.path.join(D25, 'result.json'),
    encoding='utf-8'))
assert res25['verdict'] == V25
assert res25['smoke'] is False
pa25 = res25['part_a']
assert abs(pa25['decomp3']['P']['trail_share']
           - 0.08599867672568678) < 1e-12
assert abs(pa25['decomp3']['A1']['trail_share']
           - 0.05022617239377846) < 1e-12
z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
auc18 = z18['auc_curve']
assert len(auc18) == N_NEW + 1
z20 = np.load(os.path.join(D20, 'p118_readout.npz'),
              allow_pickle=False)
annP_c = z20['annP_c']
annA_c = z20['annA_c']
assert annP_c.shape == (672, N_NEW)
z24 = np.load(os.path.join(D24, 'p122_readout.npz'),
              allow_pickle=False)
log('frozen inputs ok (18/20/24/25)')

# ================================================================
# design seal (frozen BEFORE observation)
# ================================================================
seal = {
    'phase': 3126,
    'name': NAME,
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'np_a': NP,
    'np_b': NP_B,
    'np_reg': NP_REG,
    'data_sources': {
        'margins_annotation': 'phase3118 '
                              'traj_readout.npz + '
                              'phase3120 ann npz',
        'fits_3125': 'phase3125 result.json '
                     'part_a.refit (assert 1e-12) '
                     '+ trail3 shares (assert '
                     '1e-12)',
        'glm4_3124_ref': 'phase3124 '
                         'p122_readout.npz E '
                         'curves (float32)',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz'},
    'part_a': {
        'inherit': 'content table -> dm_res -> '
                   'lstsq [m,1] -> S,MS (assert '
                   'vs 3125 1e-12); D3 replication '
                   'assert (trail share 1e-12)',
        'decomp6_seq': 'R -> Z1 step-dummy -> r1 '
                       '-> Z2 [m-MS,1] -> r2 -> '
                       'trail kernel X(60: '
                       'cls[k-d]*6+(d-1), d=1..6, '
                       'missing rows zero) -> r3 '
                       '-> AR(2) on k>=2 rows -> '
                       'r4; shares relative to '
                       'ss_tot(R)',
        'long_gate': 'min(trail6_P, trail6_A1) - '
                     'min(trail3_P, trail3_A1) '
                     '>= 0.02 -> '
                     'long_range_trail_present '
                     'else '
                     'long_range_trail_absent',
        'perm_null': 'within-trajectory cls '
                     'permutation (argsort of '
                     'rng.random((672,12)) rows, '
                     'numpy default_rng(3126), '
                     '200 reps, deterministic); '
                     'full pipeline recompute per '
                     'rep (content table -> refit '
                     '-> resid -> Z1 -> Z2 -> '
                     'trail6); z = (obs - mean)/'
                     'std; min z >= 4 -> '
                     'trail_significant else '
                     'trail_notsig',
        'g_gate': 'corr(G6_P.ravel(), '
                  'G6_A1.ravel()) >= 0.5 -> '
                  'trail_shared_directions else '
                  'trail_dirspecific',
        'second_order': 'on r4: (a) joint '
                        '(cls_k, cls_{k-1}) '
                        'one-hot 100 cells on '
                        'k>=1 rows -> gain_j; '
                        '(b) [m-MS, (m-MS)^2, 1] '
                        '-> gain_q; min over '
                        'dirs of max(gain_j, '
                        'gain_q) >= 0.05 -> '
                        'second_order_candidate_'
                        'present else absent',
        'no_mc': 'only deterministic permutations '
                 '(seed 3126)'},
    'part_b': {
        'model': 'glm4-9b-chat-hf (40L, BF16, '
                 'eager, trust_remote_code, '
                 'batch1 forwards, hook-'
                 'collected per-layer states, '
                 'normG applied to ALL states '
                 '(3124 append-before trap fix))',
        'interference': 'input_prompt_span_'
                        'equal_len_anchored_last',
        'span_fix': "query line located by "
                    "suffix-anchored LAST "
                    "occurrence (' Is this' "
                    'after the line); fixes the '
                    '3124 P-direction '
                    'first-occurrence Facts-line '
                    'collision; A1 was and is '
                    'unconfounded',
        'base_gen': 'own-stream greedy '
                    'generation (batched, '
                    'max_new_tokens=12) from '
                    'prompt ids prepended with '
                    'the tokenizer default '
                    'special-token prefix '
                    '(3124 text-encode '
                    'identical, probe-verified; '
                    'keeps the Yes/No answer '
                    'style so the yes/no '
                    'readout matches); '
                    'teacher-forced forwards '
                    'stay plain-encoded '
                    '(3124-identical)',
        'curves': 'E_syn/E_cont = mean over ALL '
                  '13 readout positions of '
                  'd1-d2 / d1-d3 (3124/3125 '
                  'isomorphic)',
        'gates': {
            'B_path': 'corr(hook margin, '
                      'model-logits margin) > '
                      '0.9999 on first NP_CHECK '
                      'pairs -> path_valid else '
                      'FATAL',
            'B_repro': 's0 forward twice, max '
                       'diff == 0 on first '
                       'NP_REPRO pairs -> '
                       'replay_bit_exact; else '
                       'FATAL',
            'B_readout': 'final-step margin AUC '
                         '> 0.6 -> readout_ok',
            'B_spans': 'n_span >= 300 BOTH dirs '
                       '-> spans_ok (SMOKE: '
                       'max(1, NP_B//2))',
            'B_base3124': 'corr(E_3126, '
                          'E_3124) >= 0.9 for '
                          'A1 syn AND cont '
                          '(unconfounded) -> '
                          'baseline_replicated; '
                          'P corr reported as '
                          'collision-contribution '
                          'probe',
            'B_lstar': 'L* = min L>=20 with '
                       'E(L) <= -0.3*|E(final)| '
                       '(rel) and <= -0.05/-0.025 '
                       '(abs), 41 states',
            'B_sign': 'sign(E(NL)) per effect '
                      'per dir'},
        'no_mc': 'crc32-seeded entity choice/'
                 'shuffle (deterministic)'},
    'part_c': {
        'design': 'counterfactual regeneration: '
                  'NP_REG pairs per dir x '
                  's0..s3 actually generated '
                  '(greedy, batched) from the '
                  'perturbed prompt token ids; '
                  'padded to 12 with dot',
        'metrics': 'agreement = mean token '
                   'equality vs base gen; '
                   'first_div = first differing '
                   'step; answer flip = first-'
                   'token polarity change (yes/'
                   'no base only)',
        'gate': 'min over dirs of '
                '(agree_s0 - mean(agree_s1..3)) '
                '>= 0.10 -> '
                'behavior_shift_present else '
                'behavior_shift_absent'},
    'part_d': {
        'design': 'write-chain spectrum from '
                  'part-B s0 margins: '
                  'wspec[L,k] = mean_j '
                  '(ml[L+1,k] - ml[L,k]), '
                  'W[L] = mean_k wspec[L,k]',
        'gate': 'min over dirs of sum top-3 '
                '|W| / sum |W| >= 0.4 -> '
                'writechain_located else '
                'writechain_diffuse',
        'ref': 'qwen 3122 qualitative: L28-34 '
               'positive write, L35 large '
               'negative write (rel depth '
               '0.78-0.97)'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('seal frozen (%s)' % seal['created'])

# ================================================================
# PART A: offline trail deepening
# ================================================================
log('== PART A: trail deepening ==')
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


def trailX(cls, D):
    NIK = cls.size
    X = np.zeros((NIK, 10 * D))
    for d in range(1, D + 1):
        rows_k = np.arange(d, N_NEW)
        idx_rows = (np.repeat(
            np.arange(NP) * N_NEW,
            len(rows_k))
            + np.tile(rows_k, NP))
        cls_d = cls[:, :N_NEW - d].ravel()
        X[idx_rows, cls_d * D + (d - 1)] = 1.0
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

# D=3 replication + D=6 decomposition
dec6 = {}
ar6 = {}
G6 = {}
dec3_chk = {}
r4_store = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    R = fit[dc]['resid']
    cls = fit[dc]['col']
    (ss_step, ss_m, ss_tr3, ss_tot,
     _, _, _) = seq_shares(m, fit[dc]['MS'],
                           R, cls, 3)
    dec3_chk[dc] = ss_tr3 / ss_tot
    assert abs(dec3_chk[dc]
               - pa25['decomp3'][dc]
               ['trail_share']) < 1e-12, dc
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
log('A D6 decomposition done '
    '(D3 replication asserted)')

# permutation significance
perm_stat = {}
rgA = np.random.default_rng(PERM_SEED)
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    R = fit[dc]['resid']
    cls = fit[dc]['col']
    obs = dec6[dc]['trail6_share']
    nulls = np.zeros(N_PERM)
    for rep in range(N_PERM):
        pm = np.argsort(
            rgA.random((NP, N_NEW)), axis=1)
        cls_p = np.take_along_axis(
            cls, pm, axis=1)
        tab_p = content_table(m, cls_p)
        S_p, MS_p, resid_p = refit(m, cls_p,
                                   tab_p)
        (_, _, ss_tr, ss_tot_p,
         _, _, _) = seq_shares(
            m, MS_p, resid_p, cls_p, TRAIL_D)
        nulls[rep] = ss_tr / ss_tot_p
    mu = float(nulls.mean())
    sd = float(nulls.std())
    zsc = (obs - mu) / sd if sd > 0 else 0.0
    p_hat = float((nulls >= obs).mean())
    perm_stat[dc] = {
        'obs': obs, 'null_mean': mu,
        'null_std': sd, 'z': zsc,
        'p_hat': p_hat, 'reps': N_PERM,
        'seed': PERM_SEED}
    log('A-PERM %s: obs=%.4f null=%.4f+-%.4f '
        'z=%.2f p>=%.4f'
        % (dc, obs, mu, sd, zsc, p_hat))

# G-table cross-direction structure
gp = G6['P'].ravel()
ga = G6['A1'].ravel()
g_corr = float(np.corrcoef(gp, ga)[0, 1])
lag_e = {dc: [float((G6[dc][:, d] ** 2).sum())
              for d in range(TRAIL_D)]
         for dc in ('P', 'A1')}
log('A-GSTRUCT: corr(G_P, G_A1)=%.4f' % g_corr)

# second-order decomposition of r4
second = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    r4 = r4_store[dc]
    cls = fit[dc]['col']
    ss_r4 = float((r4 * r4).sum())
    ss_tot = dec6[dc]['ss_tot']
    # joint adjacent-class kernel (k>=1 rows)
    rows = r4[:, 1:].ravel()
    N2 = rows.size
    Xj = np.zeros((N2, 100))
    ck = cls[:, 1:].ravel()
    ckm1 = cls[:, :-1].ravel()
    Xj[np.arange(N2), ck * 10 + ckm1] = 1.0
    bj, *_ = np.linalg.lstsq(Xj, rows,
                             rcond=None)
    rj = rows - Xj @ bj
    gain_j = (float((r4[:, 1:] ** 2).sum())
              - float((rj * rj).sum())) / ss_tot
    # m-quadratic block
    x = (m[:, :-1] - fit[dc]['MS']).ravel()
    Xq = np.stack([x, x * x,
                   np.ones(x.size)], 1)
    bq, *_ = np.linalg.lstsq(Xq, r4.ravel(),
                             rcond=None)
    rq = r4.ravel() - Xq @ bq
    gain_q = (ss_r4
              - float((rq * rq).sum())) / ss_tot
    second[dc] = {'joint_gain': float(gain_j),
                  'mquad_gain': float(gain_q)}
    log('A-SECOND %s: joint=%.4f mquad=%.4f'
        % (dc, gain_j, gain_q))

# Part A gates
t3_min = min(pa25['decomp3']['P']
             ['trail_share'],
             pa25['decomp3']['A1']
             ['trail_share'])
t6_min = min(dec6['P']['trail6_share'],
             dec6['A1']['trail6_share'])
a_long_v = ('long_range_trail_present'
            if t6_min - t3_min >= 0.02
            else 'long_range_trail_absent')
z_min = min(perm_stat['P']['z'],
            perm_stat['A1']['z'])
a_sig_v = ('trail_significant'
           if z_min >= 4.0
           else 'trail_notsig')
a_share_v = ('trail_shared_directions'
             if g_corr >= 0.5
             else 'trail_dirspecific')
sec_min = min(max(second['P']['joint_gain'],
                  second['P']['mquad_gain']),
              max(second['A1']['joint_gain'],
                  second['A1']['mquad_gain']))
a_second_v = ('second_order_candidate_present'
              if sec_min >= 0.05
              else 'second_order_absent')
log('A-GATES: %s | %s | %s | %s'
    % (a_long_v, a_sig_v, a_share_v,
       a_second_v))

# ================================================================
# PART B: GPU glm4 anchored-last re-run
# ================================================================
log('== PART B: glm4 anchored-last ==')
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
texts = {}
PID = {}
hmap = {'P': hP, 'A1': hA1}
for dc in ('P', 'A1'):
    texts[dc] = {}
    PID[dc] = []
    for pk in pks:
        (s, o) = (int(v) for v in
                  pk.split('_'))
        r = p2r[pk]
        ri1, ri2 = frel[pk]
        if dc == 'P':
            (qrel, lrel) = (r, r)
        else:
            (qrel, lrel) = (r, ri1)
        t_ = build_prompt(mat5, s, o, lrel,
                          qrel)
        texts[dc][pk] = t_
        PID[dc].append(t_)

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
YES_G = int(tokG(' yes',
                 add_special_tokens=False)
            ['input_ids'][0])
NO_G = int(tokG(' no',
                add_special_tokens=False)
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

PID_T = {dc: [tokG(texts[dc][p],
                   add_special_tokens=False)
              ['input_ids']
              for p in pks[:NP_B]]
         for dc in ('P', 'A1')}
# gen-encode probe: does tokG(text) add BOS?
_pk0 = pks[0]
_genenc_ids = tokG(texts['P'][_pk0])['input_ids']
_len_genenc = len(_genenc_ids)
_len_plain = len(PID_T['P'][0])
_prefix_n = _len_genenc - _len_plain
assert _prefix_n >= 0
PREFIX_IDS = [int(t) for t in
              _genenc_ids[:_prefix_n]]
log('B-GENPROBE: gen-enc len=%d plain len=%d '
    'bos_added=%s prefix=%s decoded=%r'
    % (_len_genenc, _len_plain,
       _prefix_n > 0, PREFIX_IDS,
       tokG.decode(PREFIX_IDS)
       if PREFIX_IDS else ''))

# base generation (batched, greedy)
gen_base = {}
ftok = {}
for dc in ('P', 'A1'):
    plist = PID_T[dc]
    store = []
    for b0 in range(0, len(plist), GEN_BATCH):
        chunk = [list(PREFIX_IDS) + list(p)
                 for p in plist[b0:b0 + GEN_BATCH]]
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
            outg = modelG.generate(
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
            for e in modelG.config.eos_token_id \
                    if isinstance(
                        modelG.config.eos_token_id,
                        list) else [
                modelG.config.eos_token_id]:
                if e in ids_r:
                    ids_r = ids_r[:ids_r.index(
                        e)]
                    break
            store.append(ids_r)
        if (b0 // GEN_BATCH) % 16 == 0:
            log('B gen %s %d/%d (%.1fs)'
                % (dc, b0 + len(chunk),
                   len(plist), time.time() - T0))
    gen_base[dc] = store
    cnt = {}
    for ids_r in store:
        if ids_r:
            cnt[ids_r[0]] = cnt.get(ids_r[0],
                                    0) + 1
    top = sorted(cnt.items(),
                 key=lambda kv: -kv[1])[:8]
    ftok[dc] = [(int(t), n, tokG.decode([t]))
                for (t, n) in top]
    log('B first-tok %s: %s' % (dc, ftok[dc]))


def traj_tokens(dc, j):
    ids_r = gen_base[dc][j]
    ptxt = texts[dc][pks[j]]
    encp = tokG(ptxt, add_special_tokens=False,
                return_offsets_mapping=True)
    poffs = [tuple(v) for v in
             encp['offset_mapping']]
    ids2 = list(ids_r)[:N_NEW]
    while len(ids2) < N_NEW:
        ids2.append(DOT_G)
    return ptxt, poffs, ids2


def find_span_gq(dec, offs, s, o, qrel):
    """Anchored LAST occurrence (3125 fix)."""
    qline = 'The %s %s the %s.' % (
        ents_all[s], PREDS_all[qrel],
        ents_all[o])
    ce = dec.rfind(qline)
    while ce != -1:
        cs = ce
        ce_want = cs + len(qline)
        if dec[ce_want:ce_want + 8] \
                == ' Is this':
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


def line_tokens_g(s2, o2, lrel):
    txt = ' The %s %s the %s.' % (
        ents_all[s2], PREDS_all[lrel],
        ents_all[o2])
    return list(tokG(txt,
                     add_special_tokens=False)
                ['input_ids'])


def context_triples(pk, dcode):
    (s, o) = (int(v) for v in
              pk.split('_'))
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


def build_pids(dc, j):
    """s0..s3 prompt token ids for pair j
    (anchored-last span; 3125-identical logic)."""
    pk = pks[j]
    prompt_ids0 = list(PID_T[dc][j])
    ptxt, poffs, _base = traj_tokens(dc, j)
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    span = find_span_gq(ptxt, poffs, s, o, r)
    pids = {c: list(prompt_ids0)
            for c in SCOND}
    if span is not None:
        (k1, k2) = span
        Lspan = k2 - k1 + 1
        (s2, o2) = pick_replacement_g(pk, dc)
        (_, _, lrel, _) = context_triples(
            pk, dc)
        sub = line_tokens_g(s2, o2, lrel)
        n_pad = max(0, Lspan - len(sub))
        sub = sub[:Lspan]
        sub = sub + [DOT_G] * n_pad
        rng2 = _rnd.Random(zlib.crc32(
            ('s2|%s|%s' % (pk, dc))
            .encode('ascii')))
        shuf = list(sub)
        rng2.shuffle(shuf)
        pids['s1'][k1:k2 + 1] = sub
        pids['s2'][k1:k2 + 1] = shuf
        pids['s3'][k1:k2 + 1] = \
            [DOT_G] * Lspan
    return pids, span


def forward_trackG(prompt_ids, traj_ids,
                   want_logits=False):
    """GLM4 hook-collected per-layer states
    (append-before trap fix, 3124-identical);
    normG applied to ALL states; margin =
    normG(h) @ w_dnG at 13 stream positions."""
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
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            feats.append(o2.detach())
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
        seq = [modelG.model.embed_tokens(t_in)] \
            + feats
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


SCOND = ('s0', 's1', 's2', 's3')
MLG = {d: {c: None for c in SCOND}
       for d in ('P', 'A1')}
span_idx_store = {}
t0b = time.time()
span_count = {}
path_rs = []
path_ds = []
drep = 0.0
# span sanity probe: first vs anchored-last
for dc in ('P', 'A1'):
    pk0 = pks[0]
    (s0_, o0_) = (int(v)
                  for v in pk0.split('_'))
    qrel0 = p2r[pk0]
    ptxt0, poffs0, _ = traj_tokens(dc, 0)
    ql0 = 'The %s %s the %s.' % (
        ents_all[s0_], PREDS_all[qrel0],
        ents_all[o0_])
    n_occ = ptxt0.count(ql0)
    log('B-SPANPROBE %s %s: occurrences=%d '
        'first=%d anchored=%d'
        % (dc, pk0, n_occ, ptxt0.find(ql0),
           ptxt0.rfind(ql0)))
for dcode in ('P', 'A1'):
    store = {c: [] for c in SCOND}
    n_span = 0
    n_done = 0
    sidx = np.full((NP_B, 2), -1,
                   dtype=np.int16)
    for j in range(NP_B):
        pids, span = build_pids(dcode, j)
        _ptxt, _poffs, base = traj_tokens(
            dcode, j)
        if span is not None:
            (k1, k2) = span
            sidx[j] = (k1, k2)
            n_span += 1
        for c in SCOND:
            wl, lm = forward_trackG(
                pids[c], base,
                want_logits=(j < NP_CHECK
                             and c == 's0'))
            store[c].append(wl)
            if lm is not None:
                path_rs.append(wl[NLG, :].copy())
                path_ds.append(lm)
        if j < NP_REPRO:
            wl2, _ = forward_trackG(
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
    MLG['P']['s0'][:, NLG, -1],
    MLG['A1']['s0'][:, NLG, -1])
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
        syn = np.zeros(NLG + 1)
        con = np.zeros(NLG + 1)
        n_sp = 0
        for j in range(NP_B):
            pk = pks[j]
            (s, o) = (int(v)
                      for v in pk.split('_'))
            span = find_span_gq(
                *traj_tokens(dcode, j)[:2],
                s, o, p2r[pk])
            if span is None:
                continue
            n_sp += 1
            s0m = MLG[dcode]['s0'][j]
            d1 = MLG[dcode]['s1'][j] - s0m
            d2 = MLG[dcode]['s2'][j] - s0m
            d3 = MLG[dcode]['s3'][j] - s0m
            for L in range(NLG + 1):
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
            eff = abs(float(cvv[NLG]))
            lr = None
            la = None
            for L in range(20, NLG + 1):
                if lr is None and eff > 0 \
                        and cvv[L] \
                        <= -0.3 * eff:
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
                if cvv[NLG] > 0
                else 'negative')
        log('B-TRACE %s: n_span=%d syn rel=%s '
            'abs=%s final=%+.4f | cont rel=%s '
            'abs=%s final=%+.4f'
            % (dcode, n_sp,
               lstar_b[dcode]['syn']['rel'],
               lstar_b[dcode]['syn']['abs'],
               syn[NLG],
               lstar_b[dcode]['cont']['rel'],
               lstar_b[dcode]['cont']['abs'],
               con[NLG]))
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
    tag = 'syntax' if nm == 'syn' \
        else 'content'
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

# vs-3124 curve comparison (collision probe)
vs24 = {}
if d_spans_v != 'spans_absent':
    corr_map = {}
    for dc in ('P', 'A1'):
        for nm in ('syn', 'cont'):
            c_new = curves_b[dc][nm]
            c_old = z24['E_%s_%s' % (nm, dc)] \
                .astype(np.float64)
            corr_map['%s_%s' % (nm, dc)] = \
                float(np.corrcoef(c_new,
                                  c_old)[0, 1])
    a1_min_corr = min(corr_map['syn_A1'],
                      corr_map['cont_A1'])
    b_base_v = ('baseline_replicated'
                if a1_min_corr >= 0.9
                else 'baseline_diverged')
    ls24 = res24['part_d']['lstar']
    sg24 = {}
    for dc in ('P', 'A1'):
        for nm in ('syn', 'cont'):
            c_old = z24['E_%s_%s'
                        % (nm, dc)] \
                .astype(np.float64)
            sg24['%s_%s' % (nm, dc)] = (
                'positive'
                if c_old[NLG] > 0
                else 'negative')
    vs24 = {'corr': corr_map,
            'a1_min_corr': a1_min_corr,
            'lstar_3124': ls24,
            'sign_3124': sg24,
            'verdict': b_base_v}
    log('B-VS3124: corr synP=%.4f synA1=%.4f '
        'contP=%.4f contA1=%.4f -> %s'
        % (corr_map['syn_P'],
           corr_map['syn_A1'],
           corr_map['cont_P'],
           corr_map['cont_A1'], b_base_v))
else:
    b_base_v = 'skipped_sparse'
    vs24 = {'verdict': b_base_v}

# ================================================================
# PART C: counterfactual regeneration subset
# ================================================================
log('== PART C: regen subset ==')


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
            outg = modelG.generate(
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
            for e in modelG.config.eos_token_id \
                    if isinstance(
                        modelG.config.eos_token_id,
                        list) else [
                modelG.config.eos_token_id]:
                if e in ids_r:
                    ids_r = ids_r[:ids_r.index(
                        e)]
                    break
            outs.append(ids_r)
    return outs


def _pol(tok_id):
    """answer polarity from decoded first
    token ('Yes'/'yes' -> +1, 'No'/'no'
    -> -1; GLM4 emits 9450 'Yes' /
    2753 'No')."""
    if not tok_id:
        return 0
    t = tokG.decode([int(tok_id)])
    t = t.strip().lower()
    if t.startswith('yes'):
        return 1
    if t.startswith('no'):
        return -1
    return 0


regen_store = {}
c_stats = {}
for dc in ('P', 'A1'):
    pids_all = {}
    spans_c = {}
    for j in range(NP_REG):
        pids_c, span_c = build_pids(dc, j)
        for c in SCOND:
            pids_all.setdefault(c, []) \
                .append(pids_c[c])
        spans_c[j] = span_c
    gens_c = {}
    for c in SCOND:
        gens_c[c] = gen_from_ids(pids_all[c])
        log('C gen %s %s done (%.1fs)'
            % (dc, c, time.time() - T0))
    regen_store[dc] = gens_c
    n_flip = {c: 0 for c in SCOND}
    n_ans = {c: 0 for c in SCOND}
    agree = {c: [] for c in SCOND}
    fdiv = {c: [] for c in SCOND}
    for j in range(NP_REG):
        base12 = pad12(gen_base[dc][j])
        bpol = _pol(base12[0])
        for c in SCOND:
            g12 = pad12(gens_c[c][j])
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
            gpol = _pol(g12[0])
            if bpol != 0 and gpol != 0:
                n_ans[c] += 1
                if gpol != bpol:
                    n_flip[c] += 1
    cs = {
        'agree_mean': {c: float(
            np.mean(agree[c]))
            for c in SCOND},
        'first_div_mean': {c: float(
            np.mean(fdiv[c]))
            for c in SCOND},
        'flip_rate': {c: (float(
            n_flip[c]) / n_ans[c]
            if n_ans[c] else None)
            for c in SCOND},
        'n_ans': n_ans,
        'n_span_sub': int(sum(
            1 for v in spans_c.values()
            if v is not None))}
    c_stats[dc] = cs
    log('C-STATS %s: agree %s | fdiv %s | '
        'flip %s'
        % (dc,
           {c: round(v, 4) for c, v
            in cs['agree_mean'].items()},
           {c: round(v, 2) for c, v
            in cs['first_div_mean'].items()},
           {c: (round(v, 4) if v is not None
                else None) for c, v
            in cs['flip_rate'].items()}))
shift_min = min(
    c_stats[dc]['agree_mean']['s0']
    - float(np.mean([
        c_stats[dc]['agree_mean'][c]
        for c in ('s1', 's2', 's3')]))
    for dc in ('P', 'A1'))
c_shift_v = ('behavior_shift_present'
             if shift_min >= 0.10
             else 'behavior_shift_absent')
log('C-GATE: shift_min=%.4f -> %s'
    % (shift_min, c_shift_v))

# ================================================================
# PART D: write-chain spectrum (offline)
# ================================================================
log('== PART D: write-chain ==')
wspec_store = {}
wd = {}
for dc in ('P', 'A1'):
    s0m = MLG[dc]['s0']
    D_l = s0m[:, 1:, :] - s0m[:, :-1, :]
    ws = D_l.mean(0)
    wspec_store[dc] = ws
    W = ws.mean(1)
    W0 = ws[:, 0]
    aw = np.abs(W)
    top3 = np.argsort(aw)[::-1][:3]
    c3 = float(aw[top3].sum() / max(
        aw.sum(), 1e-12))
    pos_band = [int(L) for L in range(NLG)
                if W[L] >= 0.3]
    neg_band = [int(L) for L in range(NLG)
                if W[L] <= -0.3]
    peak = int(np.argmax(aw))
    wd[dc] = {
        'W_mean': [float(v) for v in W],
        'W_ans0': [float(v) for v in W0],
        'top3_layers': [int(v) for v in top3],
        'c3': c3,
        'pos_band_ge_0.3': pos_band,
        'neg_band_le_-0.3': neg_band,
        'peak_layer': peak,
        'peak_rel_depth': peak / float(NLG)}
    log('D %s: top3=%s c3=%.3f pos=%s neg=%s '
        'peak=L%d (%.2f depth)'
        % (dc, wd[dc]['top3_layers'], c3,
           pos_band, neg_band, peak,
           peak / float(NLG)))
c3_min = min(wd['P']['c3'], wd['A1']['c3'])
d_chain_v = ('writechain_located'
             if c3_min >= 0.4
             else 'writechain_diffuse')
log('D-GATE: c3_min=%.3f -> %s'
    % (c3_min, d_chain_v))

# ================================================================
# verdict + save
# ================================================================
verdict = '|'.join([
    a_long_v, a_sig_v, a_share_v, a_second_v,
    d_path_v, d_repro_v, d_readout_v,
    d_spans_v, b_base_v,
    d_syn_P_v, d_syn_A1_v,
    d_cont_P_v, d_cont_A1_v,
    d_synP_s_v, d_synA_s_v,
    d_contP_s_v, d_contA_s_v,
    c_shift_v, d_chain_v])
runtime = round(time.time() - T0, 1)
results = {
    'phase': 3126,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'n_pairs': NP,
    'np_b': NP_B,
    'np_reg': NP_REG,
    'runtime_s': runtime,
    'part_a': {
        'refit': {
            'P': {'S': fit['P']['S'],
                  'MS': fit['P']['MS']},
            'A1': {'S': fit['A1']['S'],
                   'MS': fit['A1']['MS']}},
        'decomp3_repl': dec3_chk,
        'decomp6': dec6,
        'ar6_params': ar6,
        'trail6_G': {
            dc: [[float(v) for v in row]
                 for row in G6[dc]]
            for dc in ('P', 'A1')},
        'long_gate': {'trail3_min': t3_min,
                      'trail6_min': t6_min,
                      'delta': t6_min - t3_min,
                      'verdict': a_long_v},
        'perm': perm_stat,
        'g_struct': {'corr': g_corr,
                     'lag_energy': lag_e,
                     'verdict': a_share_v},
        'second': second,
        'second_verdict': a_second_v},
    'part_b': {
        'interference':
            'input_prompt_span_equal_len_'
            'anchored_last',
        'interference_note': (
            'GLM4 anchored-LAST span fix; '
            '3124 used FIRST occurrence '
            'which under P collided with '
            'the identical Facts line - '
            'P-direction 3124 curves carry '
            'Facts-line-replacement '
            'semantics; A1 unconfounded '
            'both'),
        'ids': {'yes': YES_G, 'no': NO_G,
                'dot': DOT_G,
                'n_layers': NLG,
                'pad': int(PADG)},
        'gen_probe_bos': {
            'gen_enc_len': _len_genenc,
            'plain_len': _len_plain,
            'prefix_ids': PREFIX_IDS,
            'prefix_decoded':
                tokG.decode(PREFIX_IDS)
                if PREFIX_IDS else ''},
        'first_tokens': ftok,
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
            if 'syn' in curves_b['P']
            else [],
            'E_syn_A1': [float(x) for x in
                         curves_b['A1']['syn']]
            if 'syn' in curves_b['A1']
            else [],
            'E_cont_P': [float(x) for x in
                         curves_b['P']['cont']]
            if 'cont' in curves_b['P']
            else [],
            'E_cont_A1': [float(x) for x in
                          curves_b['A1']
                          ['cont']]
            if 'cont' in curves_b['A1']
            else []},
        'vs_3124': vs24},
    'part_c': {
        'np_reg': NP_REG,
        'stats': c_stats,
        'shift_min': shift_min,
        'verdict': c_shift_v,
        'note': 's0 agreement vs base is the '
                'greedy-consistency baseline; '
                's1..s3 agreement drop = '
                'behavioral effect of '
                'input-stream perturbation'},
    'part_d': {
        'wspec_note': 'wspec[L,k] = mean_j '
                      '(ml[L+1,k] - ml[L,k]) '
                      'from s0 margins; normed '
                      'states; all pairs',
        'layers': wd,
        'c3_min': c3_min,
        'verdict': d_chain_v,
        'qwen_3122_ref': 'L28-34 positive '
                         'write, L35 large '
                         'negative write '
                         '(rel depth 0.78-0.97)'},
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False,
              indent=1)
npz_out = {
    'G6_P': G6['P'],
    'G6_A1': G6['A1'],
    'wspec_P': wspec_store['P'].astype(
        np.float32),
    'wspec_A1': wspec_store['A1'].astype(
        np.float32),
    'gen_base_P': np.array(
        [pad12(g) for g in gen_base['P']],
        dtype=np.int32),
    'gen_base_A1': np.array(
        [pad12(g) for g in gen_base['A1']],
        dtype=np.int32),
}
for dc in ('P', 'A1'):
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
    for c in SCOND:
        arr = np.zeros((NP_REG, N_NEW),
                       dtype=np.int32)
        for j in range(NP_REG):
            g12 = pad12(
                regen_store[dc][c][j])
            arr[j] = g12
        npz_out['regen_%s_%s' % (c, dc)] = arr
np.savez(os.path.join(OUT, 'p124_readout.npz'),
         **npz_out)
log('verdict=%s' % verdict)
log('done (%.1fs)' % runtime)
print('phase3126 done: %s' % verdict)
