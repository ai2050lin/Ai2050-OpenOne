# -*- coding: utf-8 -*-
"""Phase 3077 Omega-P74: write-routing function
(NO forwards - pure re-analysis of frozen npz).

Question (3076 A menu): WHAT decides which heads
become focal in a prompt family?  We have three
families x 32 heads of causal swap spectra
(R1_ALL32_A/B/C, 3076 npz) plus frozen geometry
(3071 DOH/DHH, 3072 M34/COS_MED_H/ATT, positions).
Protocol:
  E1 assembly + cross-source bit anchors
  E2 intra-family observation-causation correlation
     (spearman + permutation p, seed 3020)
  E3 two-way (head x family) variance decomposition
     ICC + sign-flip quantification of focal heads
  E4 preregistered feature bidding (g-features x
     3 families: spearman + top8 overlap vs
     hypergeometric p)
  E5 stable component alpha_i vs eps_fi structure
  E6 family-A attention features (nanmedian)
  verdict tree with preregistered gates G1-G4.
All inputs are frozen npz files; zero model calls."""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P76 = BASE + (r'\phase3076\omega_p73_cross_prompt_'
              r'family')
P71 = BASE + (r'\phase3071\omega_p68_attn_head_'
              r'decomp')
P72 = BASE + (r'\phase3072\omega_p69_focal_head_'
              r'lineage')
P73 = BASE + (r'\phase3073\omega_p70_head_'
              r'interaction')
NAME = 'omega_p74_write_routing'
OUT = (BASE + r'\phase3077' + '\\' + NAME
       + (r'\smoke' if SMOKE else ''))
SCRIPT = (ROOT + r'\tests\glm5\phase3077_omega_'
          r'p74_write_routing.py')
FK = ('A', 'B', 'C')
SEED = 3020
N_PERM = 200 if SMOKE else 10000
TOL_BIT = 0.0

PREREG = {
    'mode': 'NO forwards; frozen npz re-analysis '
            'of 3076 (R1_ALL32_A/B/C 32-head causal '
            'swap spectra x 3 prompt families) + '
            '3071 (DOH/DHH/R34/R35/C34H) + 3072 '
            '(M34_MED/COS_MED_H/ATT34B/ATT34I) + '
            '3073 (I_BOOK); single model qwen3-4b '
            'inherited; all statistics fp64; '
            'spearman = stable argsort rank '
            'correlation; permutation p two-sided, '
            'rng default_rng(3020), n_perm=%d; '
            'hypergeometric tail P(X>=k), N=32, '
            'K=n=8; SMOKE=1 reduces n_perm to 200'
            % N_PERM,
    'question': '3077 A (menu of 3076): what '
                'decides which heads become FOCAL '
                '(top-8 most-negative causal swap '
                'effect) in a prompt family?  Is '
                'focality predictable from '
                'family-independent (global) head '
                'features, or is it family-local '
                '(head x family interaction)?  '
                'Subsidiary: why does the '
                'observational spectrum (|DAH|) '
                'fail to predict the causal '
                'spectrum (r1)?',
    'features': [
        'g1 mean_f |DAH34_MED| (composite over '
        'all 3 families; dir -1; eval all)',
        'g2 max_f |DAH34_MED| (dir -1; eval all)',
        'g3 mean_f |DAH35_MED| (transfer layer; '
        'dir -1; eval all)',
        'g4 |R1_A| (family-A causal prior, '
        'magnitude; dir -1; SOURCE=A -> eval '
        'B/C only)',
        'g5 R1_A (family-A causal prior, '
        'signed; dir +1; SOURCE=A -> eval B/C '
        'only)',
        'g6 |M34_MED| (3072 single-head '
        'marginal IE, family A measurement; '
        'dir -1; SOURCE=A -> eval B/C only)',
        'g7 head index i (dir BOTH, '
        'exploratory; eval all)',
        'g8 i//4 GQA group (dir BOTH, '
        'exploratory; eval all)',
        'g9 i//8 quarter (dir BOTH, '
        'exploratory; eval all)',
        'g10 min_f |DAH34_MED| (dir -1; eval '
        'all)',
        'g11 |DOH34_MED| (3071 head-output '
        'delta norm, family A; dir -1; '
        'SOURCE=A -> eval B/C only)',
        'g12 |DHH34_MED| (3071 head-input '
        'delta norm, family A; dir -1; '
        'SOURCE=A -> eval B/C only)'],
    'direction_rule': 'amplitude features predict '
                      'MORE-negative R1 (dir -1); '
                      'signed prior g5 predicts '
                      'same-sign (dir +1); position '
                      'features g7-g9 have no '
                      'natural direction -> both '
                      'directions computed, the '
                      'larger |sp| reported and '
                      'flagged exploratory; '
                      'top8-overlap uses the '
                      'preregistered direction '
                      '(for BOTH features the '
                      'reported direction)',
    'evaluation_rule': 'SMOKE correction frozen '
                      'BEFORE the authoritative run: '
                      'features measured on ONE '
                      'family (g4-g6, g11-g12, '
                      'SOURCE=A) are evaluated ONLY '
                      'on the non-source families '
                      '(B/C) - self-prediction of a '
                      'family by its own measurement '
                      'is excluded (g5 on A would be '
                      'the identity); composite '
                      '(g1-g3, g10) and position '
                      '(g7-g9) features are '
                      'evaluated on all three '
                      'families; mean|sp| and '
                      'mean overlap for G1/G2 use '
                      'each best feature\'s own '
                      'evaluation set',
    'gates': 'G1 routing_predictable_global: best '
             'g-feature mean |spearman| over the 3 '
             'families >= 0.6 (measured in the '
             'preregistered direction); G2 '
             'top8_predictable: same best feature '
             'mean top8 overlap >= 5 of 8; G3 '
             'observation_causation_decoupled: '
             'mean over families of '
             'spearman(|DAH34_MED_f|, R1_f) has '
             '|mean| < 0.3 (weak coupling); G4 '
             'causal_stable_component_dominant: '
             'ICC_head of the R matrix >= 0.5 '
             '(two-way head x family ANOVA)',
    'verdict': 'anchors fail -> named mismatch; '
               'any family n_neg < 8 -> '
               'spectrum_degenerate; G1 AND G2 -> '
               'routing_predictable_global; elif '
               'G4 AND (not G1) -> '
               'causal_stable_component_dominant; '
               'elif G3 AND (not G1) -> '
               'observation_causation_decoupled; '
               'elif (not G1) -> '
               'routing_family_local; else -> '
               'routing_mixed',
    'statistics_discipline': 'gates preregistered '
             'before analysis; multiple features '
             'compete - the verdict uses only the '
             'preregistered directions and the '
             'BEST feature is reported with its '
             'permutation p (selection noted as a '
             'caveat); top8 overlap p is '
             'hypergeometric (selection-free); '
             'no post-hoc feature additions',
    'limitations': '3 families x 32 heads = 96 '
             'observations; ICC with 3 family '
             'levels has wide uncertainty (only 2 '
             'df for family); permutation p on '
             'best-of-12 features is optimistic '
             '(selection); family-A-only features '
             '(g4-g6, g11-g12) measured on 24 '
             'pairs of ONE family generalize to '
             'other families only as hypotheses; '
             'ATT features family A only and may '
             'contain padding NaN',
    'memory_discipline': 'no model load; peak '
             'memory = 3076/3071/3072/3073 npz '
             'TT/LG arrays (~100 MB); everything '
             'else < 10 MB',
}

os.makedirs(OUT, exist_ok=True)
LOGF = OUT + r'\run_log.txt'
_o = []


def log(msg):
    _o.append(str(msg))


t0 = time.time()

# ---------- execution.json (prereg freeze) ----------
exep = OUT + r'\execution.json'
exe = {'phase': 3077, 'name': NAME,
       'created': time.strftime(
           '%Y-%m-%d %H:%M:%S'),
       'smoke': SMOKE, 'prereg': PREREG}
with io.open(exep, 'w', encoding='utf-8') as f:
    json.dump(exe, f, ensure_ascii=False, indent=1)
log('execution.json written (prereg frozen) %s '
    'smoke=%s' % (exe['created'], SMOKE))

# ---------- load frozen data ----------
z76 = np.load(
    P76 + r'\omega_p73_cross_prompt_family.npz')
z71 = np.load(P71 + r'\omega_p68_attn_head_decomp.npz')
z72 = np.load(P72 + r'\omega_p69_focal_head_lineage.npz')
z73 = np.load(P73 + r'\omega_p70_head_interaction.npz')
res76 = json.load(io.open(
    P76 + r'\result.json', encoding='utf-8'))

R1 = {f: z76['R1_ALL32_' + f].astype(np.float64)
      for f in FK}
D34 = {f: z76['DAH34_MED_' + f].astype(np.float64)
       for f in FK}
D35 = {f: z76['DAH35_MED_' + f].astype(np.float64)
       for f in FK}
T8 = {f: [int(x) for x in z76['TOP8_' + f]]
      for f in FK}
MEDC = {f: float(z76['MED_C_34_' + f]) for f in FK}
CS1H = {f: z76['CS1H_' + f].astype(np.float64)
        for f in FK}


# ---------- stats helpers ----------
def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def perm_p(a, b, n_perm=N_PERM, seed=SEED):
    """Two-sided permutation p for spearman,
    vectorized (independent row permutations)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
    if n_perm <= 0:
        return 1.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    ra -= ra.mean()
    B = np.tile(b, (n_perm, 1))
    B = rng.permuted(B, axis=1)
    rb = np.argsort(np.argsort(B, axis=1),
                    axis=1).astype(np.float64)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum(axis=1))
    sp = (rb @ ra) / den
    cnt = int((np.abs(sp) >= obs - 1e-12).sum())
    return (cnt + 1) / (n_perm + 1)


def hyper_p(k, N=32, K=8, n=8):
    from math import comb
    tot = comb(N, n)
    num = sum(comb(K, j) * comb(N - K, n - j)
              for j in range(k, min(K, n) + 1))
    return num / tot


def anova2(M):
    """Two-way (row=family, col=head) no-rep
    decomposition.  Returns dict with alpha
    (head, per column), beta (family, per row),
    eps, variances."""
    M = np.asarray(M, dtype=np.float64)
    gm = M.mean()
    alpha = M.mean(axis=0) - gm   # per head (nh,)
    beta = M.mean(axis=1) - gm    # per family (nf,)
    eps = M - gm - alpha[None, :] - beta[:, None]
    nf, nh = M.shape
    v_head = float((alpha ** 2).sum() / (nh - 1))
    v_fam = float((beta ** 2).sum() / (nf - 1))
    v_eps = float((eps ** 2).sum()
                  / ((nf - 1) * (nh - 1)))
    return {'alpha': alpha, 'beta': beta,
            'eps': eps, 'gm': gm,
            'v_head': v_head, 'v_fam': v_fam,
            'v_eps': v_eps,
            'icc_head': v_head / (v_head + v_eps)}


# ---------- E1 anchors ----------
log('[E1] anchors (cross-source bit)')
an = {}
an['a1_diff'] = float(np.abs(
    R1['A'] - z71['R34'].astype(np.float64)).max())
an['a1_ok'] = an['a1_diff'] == TOL_BIT
an['a2_diff'] = float(np.abs(
    D34['A'] - z71['DAH34_MED'].astype(
        np.float64)).max())
an['a2_ok'] = an['a2_diff'] == TOL_BIT
an['a3_diff'] = float(np.abs(
    D34['A'] - z72['DAH34_MED'].astype(
        np.float64)).max())
an['a3_ok'] = an['a3_diff'] == TOL_BIT
an['a4_diff'] = float(np.abs(
    CS1H['A'] - z71['C34H'].astype(
        np.float64)).max())
an['a4_ok'] = an['a4_diff'] == TOL_BIT
dmax = 0.0
for f in FK:
    rec = np.median(CS1H[f], axis=1) - MEDC[f]
    dmax = max(dmax, float(np.abs(
        rec - R1[f]).max()))
an['a5_diff'] = dmax
an['a5_ok'] = dmax == TOL_BIT
dmax = 0.0
for f in FK:
    order = np.argsort(R1[f])
    topk = min(8, int((R1[f] < 0).sum()))
    rec = [int(h) for h in order[:topk]]
    dmax = max(dmax, 0.0 if rec == T8[f] else 1.0)
an['a6_diff'] = dmax
an['a6_ok'] = dmax == TOL_BIT
dmax = 0.0
for f in FK:
    rj = res76['stats']['families'][f]['top8']
    if rj != T8[f]:
        dmax = 1.0
    if abs(res76['stats']['families'][f]
           ['med_c_34'] - MEDC[f]) > 0:
        dmax = 1.0
an['a7_diff'] = dmax
an['a7_ok'] = dmax == TOL_BIT
for k in sorted(an.keys()):
    log('  %s=%s' % (k, an[k]))
anchors_ok = all(v for k, v in an.items()
                 if k.endswith('_ok'))
log('[E1] anchors_ok=%s' % anchors_ok)

n_neg = {f: int((R1[f] < 0).sum()) for f in FK}
degenerate = any(v < 8 for v in n_neg.values())
log('[E1] n_neg=%s degenerate=%s'
    % (n_neg, degenerate))

if not anchors_ok:
    verdict = 'anchor_mismatch_frozen_data'
elif degenerate:
    verdict = 'spectrum_degenerate'
else:
    # ---------- E2 intra-family ----------
    log('[E2] intra-family observation-causation')
    intra = {}
    for f in FK:
        r1 = R1[f]
        e = {}
        e['sp_absd34'] = spearman(np.abs(D34[f]), r1)
        e['p_absd34'] = perm_p(np.abs(D34[f]), r1)
        e['sp_absd35'] = spearman(np.abs(D35[f]), r1)
        e['p_absd35'] = perm_p(np.abs(D35[f]), r1)
        e['sp_sd34'] = spearman(D34[f], r1)
        e['p_sd34'] = perm_p(D34[f], r1)
        intra[f] = e
        log('  %s sp(|D34|,R1)=%+.4f p=%.4f | '
            'sp(|D35|,R1)=%+.4f p=%.4f | '
            'sp(sD34,R1)=%+.4f p=%.4f'
            % (f, e['sp_absd34'], e['p_absd34'],
               e['sp_absd35'], e['p_absd35'],
               e['sp_sd34'], e['p_sd34']))
    # family-A extras (3071/3072/3073 geometry)
    r1a = R1['A']
    xa = {}
    xa['sp_absdoh34'] = spearman(
        np.abs(z71['DOH34_MED'].astype(np.float64)),
        r1a)
    xa['sp_absdhh34'] = spearman(
        np.abs(z71['DHH34_MED'].astype(np.float64)),
        r1a)
    xa['sp_absm34'] = spearman(
        np.abs(z72['M34_MED'].astype(np.float64)),
        r1a)
    xa['sp_cosmed'] = spearman(
        z72['COS_MED_H'].astype(np.float64), r1a)
    # head-level interaction strength from 3073:
    # I_PAIR (28,24) -> per-head mean involvement
    pairs = z73['PAIRS']
    ip = z73['I_PAIR'].astype(np.float64)
    invol = np.zeros(32)
    cnt = np.zeros(32)
    for j in range(pairs.shape[0]):
        h1, h2 = int(pairs[j, 0]), int(pairs[j, 1])
        v = abs(float(np.median(ip[j])))
        invol[h1] += v
        invol[h2] += v
        cnt[h1] += 1
        cnt[h2] += 1
    invol = invol / np.maximum(cnt, 1)
    xa['sp_invol'] = spearman(invol, r1a)
    for k in ('sp_absdoh34', 'sp_absdhh34',
              'sp_absm34', 'sp_cosmed',
              'sp_invol'):
        log('  A-extra %s=%+.4f'
            % (k, xa[k]))
    intra['A_extra'] = xa

    # ---------- E3 ICC + sign flip ----------
    log('[E3] variance decomposition')
    M_R = np.stack([R1[f] for f in FK])
    M_D = np.stack([np.abs(D34[f]) for f in FK])
    aR = anova2(M_R)
    aD = anova2(M_D)
    # permutation p for ICC: shuffle heads within
    # each family row independently
    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(N_PERM):
        Mp = np.stack([rng.permutation(row)
                       for row in M_R])
        if anova2(Mp)['icc_head'] \
                >= aR['icc_head'] - 1e-12:
            cnt += 1
    icc_p = (cnt + 1) / (N_PERM + 1)
    log('  ICC(R): v_head=%.5f v_fam=%.5f '
        'v_eps=%.5f icc=%.4f p_perm=%.4f'
        % (aR['v_head'], aR['v_fam'],
           aR['v_eps'], aR['icc_head'], icc_p))
    log('  ICC(|D34|): v_head=%.4f v_fam=%.4f '
        'v_eps=%.4f icc=%.4f'
        % (aD['v_head'], aD['v_fam'],
           aD['v_eps'], aD['icc_head']))
    # sign flip: own-family top8 R1 vs same heads
    # in other families
    own = []
    oth = []
    for f in FK:
        for h in T8[f]:
            own.append(R1[f][h])
            oth.extend(R1[g][h] for g in FK
                       if g != f)
    own = np.array(own)
    oth = np.array(oth)
    sf_obs = float(oth.mean() - own.mean())
    rng = np.random.default_rng(SEED + 1)
    cnt = 0
    heads_own = [h for f in FK for h in T8[f]]
    vals = {h: np.array([R1[f][h] for f in FK])
            for h in set(heads_own)}
    for _ in range(N_PERM):
        o = []
        t = []
        for h in heads_own:
            v = vals[h]
            k = int(rng.integers(0, 3))
            o.append(v[k])
            t.extend(np.delete(v, k))
        if float(np.array(t).mean()
                 - np.array(o).mean()) \
                >= sf_obs - 1e-12:
            cnt += 1
    sf_p = (cnt + 1) / (N_PERM + 1)
    log('  sign-flip: mean_own=%.4f mean_other='
        '%.4f delta=%.4f p_perm=%.4f'
        % (own.mean(), oth.mean(), sf_obs, sf_p))

    # ---------- E4 feature bidding ----------
    log('[E4] feature bidding (preregistered)')
    absD = {f: np.abs(D34[f]) for f in FK}
    absD35 = {f: np.abs(D35[f]) for f in FK}
    feats = [
        ('g1', 'mean|D34|',
         np.stack([absD[f] for f in FK]).mean(0), -1),
        ('g2', 'max|D34|',
         np.stack([absD[f] for f in FK]).max(0), -1),
        ('g3', 'mean|D35|',
         np.stack([absD35[f] for f in FK]).mean(0),
         -1),
        ('g4', '|R1_A|', np.abs(R1['A']), -1),
        ('g5', 'R1_A', R1['A'].copy(), +1),
        ('g6', '|M34|',
         np.abs(z72['M34_MED'].astype(np.float64)),
         -1),
        ('g7', 'idx', np.arange(32, dtype=np.float64),
         0),
        ('g8', 'gqa',
         (np.arange(32) // 4).astype(np.float64), 0),
        ('g9', 'qtr',
         (np.arange(32) // 8).astype(np.float64), 0),
        ('g10', 'min|D34|',
         np.stack([absD[f] for f in FK]).min(0), -1),
        ('g11', '|DOH34|',
         np.abs(z71['DOH34_MED'].astype(np.float64)),
         -1),
        ('g12', '|DHH34|',
         np.abs(z71['DHH34_MED'].astype(np.float64)),
         -1)]
    fstats = []
    for fid, desc, fv, dr in feats:
        e = {'id': fid, 'desc': desc,
             'dir': int(dr), 'sp': {},
             'ov': {}, 'p_ov': {}}
        # evaluation set: source-family features
        # are evaluated on B/C only (no
        # self-prediction); smoke correction
        # frozen before the authoritative run
        if fid in ('g4', 'g5', 'g6', 'g11',
                   'g12'):
            e['eval_f'] = ['B', 'C']
            e['source'] = 'A'
        else:
            e['eval_f'] = ['A', 'B', 'C']
            e['source'] = None
        for f in e['eval_f']:
            s = spearman(fv, R1[f])
            e['sp'][f] = s
            # top8 predicted: heads with the most
            # negative R1 under the preregistered
            # direction
            if dr > 0:
                pred = np.argsort(fv)[:8]
            elif dr < 0:
                pred = np.argsort(-fv)[:8]
            else:
                # BOTH: direction chosen by |sp|
                pred = (np.argsort(fv)[:8]
                        if s > 0
                        else np.argsort(-fv)[:8])
            ov = len(set(int(x) for x in pred)
                     & set(T8[f]))
            e['ov'][f] = ov
            e['p_ov'][f] = hyper_p(ov)
        e['mean_abs_sp'] = float(np.mean(
            [abs(e['sp'][f])
             for f in e['eval_f']]))
        e['mean_ov'] = float(np.mean(
            [e['ov'][f] for f in e['eval_f']]))
        fstats.append(e)
        log('  %s %-9s src=%s sp=%s mean|sp|='
            '%.3f ov=%s mean=%.2f'
            % (fid, desc, e['source'],
               {f: round(e['sp'][f], 3)
                for f in e['eval_f']},
               e['mean_abs_sp'],
               [e['ov'][f] for f in e['eval_f']],
               e['mean_ov']))
    # best feature by mean_abs_sp among
    # preregistered-direction features only
    # (dr != 0); BOTH-direction features are
    # exploratory and excluded from G1/G2
    core = [e for e in fstats if e['dir'] != 0]
    best = max(core,
               key=lambda e: e['mean_abs_sp'])
    log('  best=%s mean|sp|=%.3f mean_ov=%.2f'
        % (best['id'], best['mean_abs_sp'],
           best['mean_ov']))

    # ---------- E5 stable vs interaction ----------
    log('[E5] stable component vs interaction')
    alpha = aR['alpha']
    eps = aR['eps']
    al8 = [int(h) for h in np.argsort(alpha)[:8]]
    al_ov = {f: len(set(al8) & set(T8[f]))
             for f in FK}
    log('  alpha_top8=%s overlap=%s'
        % (al8, al_ov))
    mean_d34 = np.stack(
        [absD[f] for f in FK]).mean(0)
    sp_alpha_g1 = spearman(alpha, mean_d34)
    eps_ov = {}
    for fi, f in enumerate(FK):
        e8 = [int(h) for h in
              np.argsort(eps[fi])[:8]]
        eps_ov[f] = {'eps_top8': e8,
                     'overlap': len(set(e8)
                                    & set(T8[f]))}
        log('  %s eps_top8=%s overlap=%d'
            % (f, e8, eps_ov[f]['overlap']))
    log('  sp(alpha, mean|D34|)=%+.4f'
        % sp_alpha_g1)

    # ---------- E6 family-A attention ----------
    log('[E6] family-A attention features')
    att = {}
    for nm, arr in (('ATT34I',
                     z72['ATT34I'].astype(
                         np.float64)),
                    ('ATT34B',
                     z72['ATT34B'].astype(
                         np.float64))):
        nan_r = float(np.isnan(arr).mean())
        # last-position feature: many entries are
        # padding NaN -> keep only heads with a
        # valid nanmedian over pairs
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            last = np.nanmedian(arr[:, :, -1],
                                axis=0)
            inj = np.nanmedian(arr[:, :, :4],
                               axis=(0, 2))
        mlast = ~np.isnan(last)
        e_last = (spearman(last[mlast], r1a[mlast])
                  if mlast.sum() >= 8 else None)
        minj = ~np.isnan(inj)
        e_inj = (spearman(inj[minj], r1a[minj])
                 if minj.sum() >= 8 else None)
        att[nm] = {
            'nan_frac': nan_r,
            'n_head_valid_last': int(mlast.sum()),
            'sp_last_vs_R1A': e_last,
            'sp_inj_vs_R1A': e_inj,
            'sp_inj_vs_absD34A': (
                spearman(inj[minj], absD['A'][minj])
                if minj.sum() >= 8 else None),
            'head_inj_top3': [int(h) for h in
                              np.argsort(
                                  -np.nan_to_num(
                                      inj))[:3]]}
        log('  %s nan=%.3f valid_last=%d '
            'sp(last,R1A)=%s sp(inj,R1A)=%s '
            'inj_top3=%s'
            % (nm, nan_r, int(mlast.sum()),
               ('%.4f' % e_last
                if e_last is not None else 'NA'),
               ('%.4f' % e_inj
                if e_inj is not None else 'NA'),
               att[nm]['head_inj_top3']))

    # ---------- gates + verdict ----------
    g1 = best['mean_abs_sp'] >= 0.6
    g2 = best['mean_ov'] >= 5.0
    mean_intra = float(np.mean(
        [intra[f]['sp_absd34'] for f in FK]))
    g3 = abs(mean_intra) < 0.3
    g4 = aR['icc_head'] >= 0.5
    log('[G] G1=%s G2=%s G3=%s (mean_intra=%.4f) '
        'G4=%s (icc=%.4f)'
        % (g1, g2, g3, mean_intra, g4,
           aR['icc_head']))
    if g1 and g2:
        verdict = 'routing_predictable_global'
    elif g4 and not g1:
        verdict = ('causal_stable_component_'
                   'dominant')
    elif g3 and not g1:
        verdict = 'observation_causation_decoupled'
    elif not g1:
        verdict = 'routing_family_local'
    else:
        verdict = 'routing_mixed'
    log('[G] VERDICT: %s' % verdict)

stats = {
    'n_neg': n_neg,
    'anchors': an,
    'intra': {f: intra[f] for f in FK},
    'intra_A_extra': intra.get('A_extra', {}),
    'icc': {
        'R': {'v_head': aR['v_head'],
              'v_fam': aR['v_fam'],
              'v_eps': aR['v_eps'],
              'icc_head': aR['icc_head'],
              'p_perm': icc_p},
        'absD34': {'v_head': aD['v_head'],
                   'v_fam': aD['v_fam'],
                   'v_eps': aD['v_eps'],
                   'icc_head': aD['icc_head']}},
    'sign_flip': {'mean_own': float(own.mean()),
                  'mean_other': float(oth.mean()),
                  'delta': sf_obs,
                  'p_perm': sf_p},
    'features': fstats,
    'best_feature': best,
    'alpha_top8': al8,
    'alpha_overlap': al_ov,
    'sp_alpha_meand34': sp_alpha_g1,
    'eps_overlap': eps_ov,
    'att_a': att,
    'gates': {'G1': bool(g1), 'G2': bool(g2),
              'G3': bool(g3),
              'mean_intra_sp': mean_intra,
              'G4': bool(g4),
              'icc_head': aR['icc_head']},
    'forwards': 0,
}

# ---------- npz save ----------
npz_path = OUT + '\\' + NAME + '.npz'
np.savez_compressed(
    npz_path,
    R1_3x32=M_R.astype(np.float64),
    ABS_D34_3x32=M_D.astype(np.float64),
    ALPHA=aR['alpha'].astype(np.float64),
    BETA=aR['beta'].astype(np.float64),
    EPS=aR['eps'].astype(np.float64),
    GAMMA_M_D34=aD['alpha'].astype(np.float64),
    FEAT_IDS=np.array([e['id'] for e in fstats]),
    FEAT_SP=np.array([[e['sp'].get(f, np.nan)
                       for f in FK]
                      for e in fstats],
                     dtype=np.float64),
    FEAT_OV=np.array([[e['ov'].get(f, -1)
                       for f in FK]
                      for e in fstats],
                     dtype=np.int64),
    VERDICT=np.array(verdict),
    ICC_P=np.array(icc_p),
    SF_DELTA=np.array(sf_obs),
    SF_P=np.array(sf_p),
    MEAN_INTRA=np.array(mean_intra),
    TOP8_A=np.array(T8['A'], dtype=np.int64),
    TOP8_B=np.array(T8['B'], dtype=np.int64),
    TOP8_C=np.array(T8['C'], dtype=np.int64))
log('npz saved')

result = {
    'phase': 3077, 'name': NAME,
    'created': exe['created'],
    'elapsed': time.time() - t0,
    'forwards': 0,
    'run': 'run1 authoritative (NO forwards; '
           'frozen npz re-analysis)'
    if not SMOKE else 'smoke',
    'prereg': PREREG,
    'stats': stats,
    'verdict': verdict,
}
res_path = OUT + r'\result.json'
with io.open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)


def fsha8(p):
    return hashlib.sha256(
        io.open(p, 'rb').read()).hexdigest()[:8]


seal = {
    'phase': 3077, 'name': NAME,
    'created': exe['created'],
    'npz_sha256_8': fsha8(npz_path),
    'result_sha256_8': fsha8(res_path),
    'exec_sha256_8': fsha8(exep),
    'script_sha256_8': fsha8(SCRIPT),
    'verdict': verdict,
    'setup_ok': bool(anchors_ok and not degenerate),
}
with io.open(OUT + r'\seal.json', 'w',
             encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'],
       time.time() - t0))
log('RUN_COMPLETE %s' % verdict)

with io.open(LOGF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(_o) + '\n')
print('RUN_COMPLETE %s' % verdict)
