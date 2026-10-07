"""
Phase 3080 A (menu of 3079): omega_p77_ab_anatomy.
NO forwards - frozen-npz re-analysis.

Question (3079 leftovers): (1) AB ANOMALY - the AB
family pair migrates STRONGEST (spearman R1_A->R1_B
= 0.676) yet the preregistered f1 rank-shape
similarity FAILS there (+0.214, p=0.32) - what
locks AB migration?  (2) UPGRADE CHECK - the
exploratory f2 cos(TT) predictor beat f1
everywhere in 3079 (T_AC +0.771 p=1e-4); is f2 a
consistent carrier across ALL six tests?

HONESTY DISCIPLINE (preregistered): f2 values were
already observed in 3079 on the SAME frozen 3076
npz.  E2 below is a preregistered CONFIRMATORY
RE-CHECK with a gate fixed BEFORE this run - it is
NOT an independent confirmation.  Independent
confirmation requires new data (DS7B replication
or new families).

Data (frozen): 3076 npz (TT/LG/CS/CS1H/A_S/R1/
SP_R1/MASKS), 3079 npz (T/U/F1-F5 arrays + E3
spearman/p), 3078 npz (DM_L34 base readout
spectra), 3074 npz (A_S), 3071 result (r34).

Design:
  E1 anchors (bit, no forwards):
    a1 T/U/F1-F5 recomputed from 3076 vs 3079 npz
       bit 0.0 (5 arrays x 3 pairs x 2 responses
       -> T/U + F, i.e. 25 arrays)
    a2 SP_R1 replay (spearman76) vs 3076 npz bit
    a3 R1_ALL32_A vs 3071 r34 bit
    a4 DM_L34 cross-family spearman(|.|,|.|) replay
       vs 3078 result.json bit
    a5 A_S_A vs 3074 npz A_S bit
  E2 f2 confirmatory re-check (preregistered gate
     G2, seed 3080, 20000 perms):
    6 tests sp(f2_cTT, response) for response in
    {T,U} x pair in {AB,AC,BC}
    G2: count(sp > 0 AND p < 0.05) >= 4 AND
        min(sp) > 0
    references: Bonferroni 0.05/6 = 0.00833;
    Stouffer one-sided z from the 6 p values
  E3 AB anatomy (exploratory, NOT gated):
    E3.1 per-pair table for AB: k, bi, ci, norms,
       f1, f2, f5, T_AB, U_AB; top/bottom pairs
    E3.2 f1-f2 coupling sp(f1, f2) x 3 pairs
       (is f1 a coarse proxy of f2?)
    E3.3 first-order partial spearman (ranks):
       f2 | f1 and f5 | f2 on T/U for AB and AC
       (8 tests, perm seed 3080) - orientation
       evidence only (n=24)
    E3.4 family level (n=3, orientation only):
       TT norm medians; DM_L34 base-spectrum
       cross-similarity vs migration (3078
       structure sharing does NOT order with
       migration, unlike A_S which ANTI-orders)
  E4 verdict:
    ab_cos_locked  : G2 passes AND f2~T_AB
        significant (perm p < 0.05) - AB locked
        by direction cos; f1 failure explained
        as rank-shape coarsening
    ab_multi_source: G2 passes but f2~T_AB not
        significant - multiple sources
    ab_unexplained : G2 fails

limitations (recorded): same frozen data as 3079
(f2 not independent); n=24 per family pair;
partial spearman first-order on n=24 is
orientation evidence; single model qwen3-4b;
causal-connective paradigm, shared syntactic
frame; migration defined on the focal top-8
subset frame; family-level statements are n=3
orientation only.

memory discipline: no model load, no forwards;
TT64/LG64 float64 copies freed before npz save.
"""
import gc
import hashlib
import io
import json
import os
import time
from statistics import NormalDist

import numpy as np

PHASE = 3080
NAME = 'omega_p77_ab_anatomy'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3080', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
SCRIPT = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3080_omega_p77_ab_anatomy.py')

SEED = 3080
N_PERM = 1000 if SMOKE else 20000
FKEYS = ('A', 'B', 'C')
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
RDIR = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913')
P76 = os.path.join(
    RDIR, 'phase3076',
    'omega_p73_cross_prompt_family')
P79 = os.path.join(
    RDIR, 'phase3079',
    'omega_p76_migration_lock')
P78 = os.path.join(
    RDIR, 'phase3078',
    'omega_p75_routing_timing')
P74 = os.path.join(
    RDIR, 'phase3074',
    'omega_p71_capacity_law')
P71 = os.path.join(
    RDIR, 'phase3071',
    'omega_p68_attn_head_decomp')

PREREG = {
    'mode': 'no forwards; frozen re-analysis '
            'of 3076/3078/3079/3074 npz + 3071 '
            'result; smoke mode optional '
            '(SMOKE=1: n_perm=1000)',
    'question': '3080 A (menu of 3079): AB '
                'migrates strongest (0.676) but '
                'f1 rank-shape fails (+0.214 '
                'ns).  What locks AB migration, '
                'and is f2 cos(TT) a consistent '
                'carrier across all six '
                'f2-response tests?',
    'honesty': 'f2 values were already observed '
               'in 3079 on the SAME frozen data; '
               'E2 is a preregistered gate fixed '
               'before this run - a confirmatory '
               'RE-CHECK, not independent '
               'confirmation; independent '
               'confirmation requires new data',
    'definitions': {
        'f1_sTT[k]': 'sp(TT_f[k], TT_g[k]) '
                     'rank shape (3079 main)',
        'f2_cTT[k]': 'cos(TT_f[k], TT_g[k]) '
                     'direction angle (3079 '
                     'exploratory, upgraded '
                     'here)',
        'f5_amp[k]': 'min/max TT norm ratio',
        'T_fg[k]': 'sp(CS1H_f[:,k], CS1H_g[:,k])',
        'U_fg[k]': 'sp(CS_f[:,k], CS_g[:,k])',
        'partial': 'first-order spearman partial '
                   '(pearson on ranks)',
    },
    'anchors': {
        'a1': 'T/U/F1-F5 recompute vs 3079 npz '
              'bit 0.0 (25 arrays)',
        'a2': 'SP_R1 replay (spearman76) vs '
              '3076 npz bit 0.0',
        'a3': 'R1_ALL32_A vs 3071 r34 bit 0.0',
        'a4': 'DM_L34 cross replay vs 3078 '
              'result bit 0.0',
        'a5': 'A_S_A vs 3074 npz bit 0.0',
    },
    'gates': {
        'G2': 'count over 6 tests of (sp > 0 '
              'AND perm p < 0.05) >= 4 AND '
              'min(sp) > 0; seed 3080, 20000 '
              'perms',
        'ab_cos_locked': 'G2 passes AND f2~T_AB '
                         'p < 0.05',
        'ab_multi_source': 'G2 passes, f2~T_AB '
                           'not significant',
        'ab_unexplained': 'G2 fails',
    },
    'statistics_discipline': 'manual 3077-series '
        'spearman for all new statistics; '
        'spearman76 (3076 corrcoef path) ONLY '
        'for the a2 SP_R1 replay; two-sided '
        'permutation p (seed 3080, vectorized); '
        'Bonferroni 0.05/6 reference reported; '
        'Stouffer one-sided z reported as a '
        'summary; partial spearman permuted on '
        'the response column only; all AB '
        'anatomy is exploratory',
    'limitations': 'same frozen data as 3079 '
        '(f2 re-check not independent); n=24 '
        'per pair (moderate power); first-order '
        'partial on n=24 is orientation only; '
        'single model; shared syntactic frame; '
        'focal top-8 subset frame; family '
        'level n=3 orientation only',
    'memory_discipline': 'no model load; TT64/'
        'LG64 copies freed before npz save',
}

os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
_o = []


def log(msg):
    _o.append(str(msg))


t0 = time.time()

# ---------- execution.json (prereg freeze) ----------
exep = os.path.join(OUT, 'execution.json')
resp = os.path.join(OUT, 'result.json')
for p_ in (exep, resp):
    if os.path.exists(p_):
        raise SystemExit(
            'stale %s exists; delete it before '
            'rerun (prereg discipline)' % p_)
exe = {'phase': PHASE, 'name': NAME,
       'created': time.strftime(
           '%Y-%m-%d %H:%M:%S'),
       'smoke': SMOKE, 'prereg': PREREG}
with io.open(exep, 'w', encoding='utf-8') as f:
    json.dump(exe, f, ensure_ascii=False,
              indent=1)
log('execution.json written (prereg frozen) %s '
    'smoke=%s' % (exe['created'], SMOKE))

# ---------- frozen data ----------
z76 = np.load(os.path.join(
    P76, 'omega_p73_cross_prompt_family.npz'))
z79 = np.load(os.path.join(
    P79, 'omega_p76_migration_lock.npz'))
z78 = np.load(os.path.join(
    P78, 'omega_p75_routing_timing.npz'))
z74 = np.load(os.path.join(
    P74, 'omega_p71_capacity_law.npz'))
j71 = json.load(io.open(
    os.path.join(P71, 'result.json'),
    encoding='utf-8'))
res78 = json.load(io.open(
    os.path.join(P78, 'result.json'),
    encoding='utf-8'))

A_S = {f: z76['A_S_' + f].astype(np.float64)
       for f in FKEYS}
CS = {f: z76['CS_' + f].astype(np.float64)
      for f in FKEYS}
CS1H = {f: z76['CS1H_' + f].astype(np.float64)
        for f in FKEYS}
R1 = {f: z76['R1_ALL32_' + f]
      .astype(np.float64) for f in FKEYS}
TT64 = {f: z76['TT_' + f].astype(np.float64)
        for f in FKEYS}
LG64 = {f: z76['LG_' + f].astype(np.float64)
        for f in FKEYS}

# ---------- statistics helpers ----------


def spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def spearman76(a, b):
    """Bit-exact replica of the 3076 spearman
    (corrcoef path) - ONLY for cross-phase
    replays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.empty(len(a))
    rb = np.empty(len(b))
    ra[np.argsort(a, kind='stable')] = \
        np.arange(len(a), dtype=np.float64)
    rb[np.argsort(b, kind='stable')] = \
        np.arange(len(b), dtype=np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def cosv(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b) / (na * nb)


def perm_p(a, b, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
    if n_perm <= 0:
        return 1.0
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    ra -= ra.mean()
    B = np.tile(b, (n_perm, 1))
    B = rng.permuted(B, axis=1)
    rb = np.argsort(np.argsort(B, axis=1),
                    axis=1).astype(np.float64)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (rb * ra[None, :]).sum(axis=1)
    den = np.sqrt(
        (rb * rb).sum(axis=1)
        * float((ra * ra).sum()))
    den[den == 0] = 1.0
    stats = np.abs(num / den)
    return float((stats >= obs - 1e-12)
                 .mean())


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()
                  * (b * b).sum())
    if den == 0:
        return 0.0
    return float((a * b).sum() / den)


def ranks(x):
    x = np.asarray(x, np.float64)
    return np.argsort(np.argsort(x)) \
        .astype(np.float64)


def partial_sp(x, y, z):
    """First-order partial spearman: pearson on
    ranks of x,y controlling rank(z)."""
    rx = ranks(x)
    ry = ranks(y)
    rz = ranks(z)
    r_xy = pearson(rx, ry)
    r_xz = pearson(rx, rz)
    r_yz = pearson(ry, rz)
    den = np.sqrt(
        (1.0 - r_xz * r_xz)
        * (1.0 - r_yz * r_yz))
    if den == 0:
        return 0.0
    return float((r_xy - r_xz * r_yz) / den)


def partial_sp_perm_p(x, y, z,
                      n_perm=N_PERM,
                      seed=SEED):
    """Perm p for partial_sp: permute y only
    (x, z structure fixed)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    obs = abs(partial_sp(x, y, z))
    if n_perm <= 0:
        return 1.0
    Y = np.tile(y, (n_perm, 1))
    Y = rng.permuted(Y, axis=1)
    rx = ranks(x)
    rz = ranks(z)
    r_xz = pearson(rx, rz)
    cnt = 0
    for i in range(n_perm):
        ry = ranks(Y[i])
        r_xy = pearson(rx, ry)
        r_yz = pearson(ry, rz)
        den = np.sqrt(
            (1.0 - r_xz * r_xz)
            * (1.0 - r_yz * r_yz))
        val = 0.0 if den == 0 else \
            (r_xy - r_xz * r_yz) / den
        if abs(val) >= obs - 1e-12:
            cnt += 1
    return float(cnt) / n_perm


# ---------- E1 anchors ----------
adiff = {}
aok = {}

d1 = 0.0
T = {}
U = {}
F = {fid: {} for fid in
     ('f1_sTT', 'f2_cTT', 'f3_sLGp',
      'f4_sLGb', 'f5_amp')}
CIDX = np.array([k // 8 + 1
                 for k in range(24)])
BIDX = np.array([k % 8
                 for k in range(24)])
PREF_K = BIDX * 4 + CIDX
BASE_K = BIDX * 4
for fa, fb in CPAIRS:
    key = fa + fb
    T[key] = np.array([
        spearman(CS1H[fa][:, k],
                 CS1H[fb][:, k])
        for k in range(24)])
    U[key] = np.array([
        spearman(CS[fa][:, k], CS[fb][:, k])
        for k in range(24)])
    F['f1_sTT'][key] = np.array([
        spearman(TT64[fa][k], TT64[fb][k])
        for k in range(24)])
    F['f2_cTT'][key] = np.array([
        cosv(TT64[fa][k], TT64[fb][k])
        for k in range(24)])
    F['f3_sLGp'][key] = np.array([
        spearman(LG64[fa][PREF_K[k]],
                 LG64[fb][PREF_K[k]])
        for k in range(24)])
    F['f4_sLGb'][key] = np.array([
        spearman(LG64[fa][BASE_K[k]],
                 LG64[fb][BASE_K[k]])
        for k in range(24)])
    n_fa = np.linalg.norm(TT64[fa], axis=1)
    n_fb = np.linalg.norm(TT64[fb], axis=1)
    F['f5_amp'][key] = np.minimum(
        n_fa, n_fb) / np.maximum(n_fa, n_fb)
    FUP = {'f1_sTT': 'F1_STT',
           'f2_cTT': 'F2_CTT',
           'f3_sLGp': 'F3_SLGP',
           'f4_sLGb': 'F4_SLGB',
           'f5_amp': 'F5_AMP'}
    d1 = max(d1, float(np.max(np.abs(
        T[key] - z79['T_' + key]))))
    d1 = max(d1, float(np.max(np.abs(
        U[key] - z79['U_' + key]))))
    for fid in F:
        d1 = max(d1, float(np.max(np.abs(
            F[fid][key]
            - z79['%s_%s' % (FUP[fid],
                             key)]))))
adiff['a1'] = d1
aok['a1'] = bool(adiff['a1'] == 0.0)

sp_re = {}
d2 = 0.0
for fa, fb in CPAIRS:
    sp_re[fa + fb] = spearman76(R1[fa],
                                R1[fb])
    d2 = max(d2, abs(sp_re[fa + fb]
                     - float(z76['SP_R1_'
                                 + fa + fb])))
adiff['a2'] = d2
aok['a2'] = bool(adiff['a2'] == 0.0)

r34_71 = np.array(
    j71['stats']['head']['r34'],
    dtype=np.float64)
adiff['a3'] = float(np.max(np.abs(
    R1['A'] - r34_71)))
aok['a3'] = bool(adiff['a3'] == 0.0)

re8 = [spearman(
    np.abs(z78['DM_L34_' + fa]),
    np.abs(z78['DM_L34_' + fb]))
    for fa, fb in CPAIRS]
adiff['a4'] = float(np.max(np.abs(
    np.array(re8)
    - np.array(res78['stats']
               ['sp_cross'][7]))))
aok['a4'] = bool(adiff['a4'] == 0.0)

adiff['a5'] = float(np.max(np.abs(
    A_S['A'] - z74['A_S']
    .astype(np.float64))))
aok['a5'] = bool(adiff['a5'] == 0.0)

for k in ('a1', 'a2', 'a3', 'a4', 'a5'):
    log('%s diff=%s ok=%s'
        % (k, adiff[k], aok[k]))
if not all(aok.values()):
    log('FATAL: anchor failed; aborting '
        'before statistics')
    with io.open(LOGF, 'w',
                 encoding='utf-8') as f:
        f.write('\n'.join(_o) + '\n')
    raise SystemExit(1)

# ---------- E2 f2 confirmatory re-check ----------
log('E2 f2 confirmatory re-check '
    '(preregistered gate G2, seed %d, '
    'n_perm %d):' % (SEED, N_PERM))
E2 = {}
for rn, RESP in (('T', T), ('U', U)):
    for key in ('AB', 'AC', 'BC'):
        s_ = spearman(F['f2_cTT'][key],
                      RESP[key])
        p_ = perm_p(F['f2_cTT'][key],
                    RESP[key])
        E2['%s_%s' % (rn, key)] = {
            'sp': s_, 'p': p_}
        log('  f2~%s_%s: %+.4f (p=%.5f)%s'
            % (rn, key, s_, p_,
               ' [Bonf]' if p_ < 0.05 / 6
               else ''))
cnt_pos = sum(
    1 for v in E2.values()
    if v['sp'] > 0 and v['p'] < 0.05)
min_sp2 = min(v['sp'] for v in E2.values())
g2 = bool(cnt_pos >= 4 and min_sp2 > 0)
zlist = [NormalDist().inv_cdf(
    min(1.0 - 1e-12, 1.0 - v['p']))
    for v in E2.values()]
stouffer = float(np.sum(zlist)
                 / np.sqrt(len(zlist)))
log('  G2: count=%d/6 min_sp=%.4f -> %s | '
    'Stouffer z(one-sided)=%.3f'
    % (cnt_pos, min_sp2, g2, stouffer))
p_f2_TAB = E2['T_AB']['p']
sp_f2_TAB = E2['T_AB']['sp']

# ---------- E3.1 AB per-pair table ----------
log('E3.1 AB per-pair table: '
    'k(bi,ci) normA normB f5 f1 f2 '
    'T_AB U_AB')
NORM = {f: np.linalg.norm(TT64[f], axis=1)
        for f in FKEYS}
for k in range(24):
    log('  %2d(%d,%d) %7.1f %7.1f '
        '%.3f %+.3f %+.3f %+.3f %+.3f'
        % (k, BIDX[k], CIDX[k],
           NORM['A'][k], NORM['B'][k],
           F['f5_amp']['AB'][k],
           F['f1_sTT']['AB'][k],
           F['f2_cTT']['AB'][k],
           T['AB'][k], U['AB'][k]))
kmax = int(np.argmax(T['AB']))
kmin = int(np.argmin(T['AB']))
log('  top T_AB pair: k=%d (bi=%d ci=%d) '
    'f1=%+.3f f2=%+.3f f5=%.3f'
    % (kmax, BIDX[kmax], CIDX[kmax],
       F['f1_sTT']['AB'][kmax],
       F['f2_cTT']['AB'][kmax],
       F['f5_amp']['AB'][kmax]))
log('  bottom T_AB pair: k=%d (bi=%d ci=%d) '
    'f1=%+.3f f2=%+.3f f5=%.3f'
    % (kmin, BIDX[kmin], CIDX[kmin],
       F['f1_sTT']['AB'][kmin],
       F['f2_cTT']['AB'][kmin],
       F['f5_amp']['AB'][kmin]))

# ---------- E3.2 f1-f2 coupling ----------
sp_f1f2 = {}
for key in ('AB', 'AC', 'BC'):
    sp_f1f2[key] = spearman(
        F['f1_sTT'][key],
        F['f2_cTT'][key])
log('E3.2 f1-f2 coupling: %s'
    % ' | '.join(
        '%s %.3f' % (k, v)
        for k, v in sp_f1f2.items()))

# ---------- E3.3 partial spearman ----------
log('E3.3 partial spearman '
    '(exploratory, orientation only):')
PSP = {}
for key in ('AB', 'AC'):
    for rn, RESP in (('T', T), ('U', U)):
        tag = '%s_%s' % (rn, key)
        p21 = partial_sp(F['f2_cTT'][key],
                         RESP[key],
                         F['f1_sTT'][key])
        p21p = partial_sp_perm_p(
            F['f2_cTT'][key], RESP[key],
            F['f1_sTT'][key])
        p52 = partial_sp(F['f5_amp'][key],
                         RESP[key],
                         F['f2_cTT'][key])
        p52p = partial_sp_perm_p(
            F['f5_amp'][key], RESP[key],
            F['f2_cTT'][key])
        PSP['f2g1_' + tag] = p21
        PSP['f2g1p_' + tag] = p21p
        PSP['f5g2_' + tag] = p52
        PSP['f5g2p_' + tag] = p52p
        log('  %s: sp(f2|f1)=%+.3f (p=%.4f) '
            'sp(f5|f2)=%+.3f (p=%.4f)'
            % (tag, p21, p21p, p52, p52p))

# ---------- E3.4 family level ----------
tt_med = {f: float(np.median(NORM[f]))
          for f in FKEYS}
mig = {'AB': sp_re['AB'], 'AC': sp_re['AC'],
       'BC': sp_re['BC']}
dmc = {'AB': re8[0], 'AC': re8[1],
       'BC': re8[2]}
sp_dmmig = spearman(
    [dmc[k] for k in ('AB', 'AC', 'BC')],
    [mig[k] for k in ('AB', 'AC', 'BC')])
sp_asmig = spearman(
    [spearman(A_S[fa], A_S[fb])
     for fa, fb in CPAIRS],
    [mig[k] for k in ('AB', 'AC', 'BC')])
log('E3.4 family level (n=3, orientation): '
    'TT norm med %.1f/%.1f/%.1f | '
    'DM_L34 cross %.4f/%.4f/%.4f | '
    'mig %.4f/%.4f/%.4f'
    % (tt_med['A'], tt_med['B'],
       tt_med['C'], dmc['AB'], dmc['AC'],
       dmc['BC'], mig['AB'], mig['AC'],
       mig['BC']))
log('  orientation: sp(DM_L34 cross, mig)'
    '=%+.3f  sp(A_S sim, mig)=%+.3f'
    % (sp_dmmig, sp_asmig))

# ---------- E4 verdict ----------
gates = {}
if not SMOKE:
    if g2 and p_f2_TAB < 0.05:
        verdict = 'ab_cos_locked'
    elif g2:
        verdict = 'ab_multi_source'
    else:
        verdict = 'ab_unexplained'
    gates = {'G2': g2,
             'count_sig_pos': cnt_pos,
             'min_sp': min_sp2,
             'stouffer_z': stouffer,
             'f2_TAB_sp': sp_f2_TAB,
             'f2_TAB_p': p_f2_TAB}
    log('VERDICT: %s (G2=%s count=%d/6 '
        'min_sp=%.4f stouffer=%.3f; '
        'f2~T_AB sp=%+.4f p=%.5f)'
        % (verdict, g2, cnt_pos, min_sp2,
           stouffer, sp_f2_TAB, p_f2_TAB))
else:
    verdict = 'smoke_pending'
    gates = {'G2': None}
    log('VERDICT: smoke_pending')

# ---------- npz ----------
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'N_PERM': np.int64(N_PERM),
}
AK = {'a1': 'A1', 'a2': 'A2', 'a3': 'A3',
      'a4': 'A4', 'a5': 'A5'}
for k in ('a1', 'a2', 'a3', 'a4', 'a5'):
    save[AK[k] + '_DIFF'] = \
        np.float64(adiff[k])
    save[AK[k] + '_OK'] = np.bool_(aok[k])
for rn in ('T', 'U'):
    for key in ('AB', 'AC', 'BC'):
        save['F2SP_%s_%s' % (rn, key)] = \
            np.float64(
                E2['%s_%s' % (rn, key)]['sp'])
        save['F2P_%s_%s' % (rn, key)] = \
            np.float64(
                E2['%s_%s' % (rn, key)]['p'])
save['G2_COUNT'] = np.int64(cnt_pos)
save['G2_MIN_SP'] = np.float64(min_sp2)
save['STOUFFER_Z'] = np.float64(stouffer)
for key in ('AB', 'AC', 'BC'):
    save['SP_F1F2_' + key] = np.float64(
        sp_f1f2[key])
for tag in PSP:
    save['PSP_' + tag.upper()] = np.float64(
        PSP[tag])
for f in FKEYS:
    save['TT_NORM_' + f] = NORM[f].astype(
        np.float64)
    save['TT_NORM_MED_' + f] = np.float64(
        tt_med[f])
for key in ('AB', 'AC', 'BC'):
    save['T_' + key] = T[key]
    save['U_' + key] = U[key]
    save['MIG_' + key] = np.float64(
        mig[key])
    save['DM_CROSS_' + key] = np.float64(
        dmc[key])
save['SP_DMMIG'] = np.float64(sp_dmmig)
save['SP_ASMIG'] = np.float64(sp_asmig)
del TT64, LG64
gc.collect()
np.savez(npz_path, **save)
log('npz saved %s' % npz_path)

# ---------- result.json ----------
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


res = {
    'phase': PHASE,
    'name': NAME,
    'smoke': SMOKE,
    'elapsed_s': f64(time.time() - t0),
    'forwards': 0,
    'verdict': verdict,
    'prereg': PREREG,
    'anchors': {
        'diffs': {AK[k]: f64(adiff[k])
                  for k in adiff},
        'oks': {AK[k]: bool(aok[k])
                for k in aok},
    },
    'stats': {
        'f2_tests': {tk: {'sp': f64(
                             E2[tk]['sp']),
                          'p': f64(
                              E2[tk]['p'])}
                     for tk in E2},
        'g2_count': int(cnt_pos),
        'g2_min_sp': f64(min_sp2),
        'stouffer_z': f64(stouffer),
        'sp_f1f2': {k: f64(v)
                    for k, v in
                    sp_f1f2.items()},
        'partial': {k: f64(v)
                    for k, v in PSP.items()},
        'family': {
            'tt_norm_med': {k: f64(v)
                            for k, v in
                            tt_med.items()},
            'dm_cross': {k: f64(v)
                         for k, v in
                         dmc.items()},
            'mig': {k: f64(v)
                    for k, v in mig.items()},
            'sp_dmmig': f64(sp_dmmig),
            'sp_asmig': f64(sp_asmig),
        },
        'ab_top_bottom': {
            'kmax': int(kmax),
            'bi_max': int(BIDX[kmax]),
            'ci_max': int(CIDX[kmax]),
            'kmin': int(kmin),
            'bi_min': int(BIDX[kmin]),
            'ci_min': int(CIDX[kmin]),
        },
    },
    'gates': gates,
    'note': '3080 A: AB-anomaly anatomy + '
            'preregistered confirmatory '
            're-check of f2 cos(TT) on the '
            'SAME frozen 3076 data (not '
            'independent).  AB anatomy is '
            'exploratory; the honest '
            'independent test is a DS7B or '
            'new-family replication.',
}
with io.open(resp, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False,
              indent=1)
log('result.json written')

# ---------- seal ----------
def fsha8(p):
    return hashlib.sha256(
        io.open(p, 'rb').read()) \
        .hexdigest()[:8]


seal = {
    'phase': PHASE,
    'name': NAME,
    'npz_sha256_8': fsha8(npz_path),
    'result_sha256_8': fsha8(resp),
    'exec_sha256_8': fsha8(exep),
    'script_sha256_8': fsha8(SCRIPT),
}
with io.open(os.path.join(OUT, 'seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8']))

with io.open(LOGF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(_o) + '\n')
print('PHASE%d DONE verdict=%s elapsed=%.1fs'
      % (PHASE, verdict, time.time() - t0))
