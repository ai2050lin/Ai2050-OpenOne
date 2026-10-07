"""
Phase 3079 A (menu of 3078): omega_p76_migration_
lock.  NO forwards - frozen-npz re-analysis.

Question: 3077 found that the causal spectrum R1
migrates asymmetrically across prompt families
(spearman R1_A->R1_B = 0.676, R1_A->R1_C = 0.130,
R1_B->R1_C = -0.088): everyday and science causal
paradigms share routing structure, social-emotional
does not.  What explains the asymmetry?  Candidate:
the family SPECTRAL SHAPE similarity (how alike two
families' subset-response spectra are) locks how
much the causal routing structure transfers.

Data (3076 npz, three families with an identical
255-subset frame over the focal top-8 heads):
  A_S_f (255,)  joint-swap amplitude spectrum
                (|readout drop|, family f)
  R_S_f (255,)  signed readout spectrum (= -A_S)
  CS_f  (255,24) per-subset per-pair readout cos
  CS1H_f(32,24) per-head single-swap cos per pair
  MASKS (255,)  canonical subset codes 1..255
                (bit-identical across families ->
                spectra are index-aligned)
  R1_ALL32_f(32,) causal per-head readout
  TT_f (24,NVOC) per-pair family direction
                 (pref-base logits difference,
                 float32 stored)
  LG_f (32,NVOC) per-prompt logits (float32)
  prompt index map: assembled[bi*4+ci], pair k =
  (ci-1)*8+bi -> pref_i = bi*4+ci, base_i = bi*4.

Design:
  E1 anchors (bit):
    a1 A_S_A vs 3074 npz A_S = 0.0
    a2 R1_ALL32_A vs 3071 r34 = 0.0
    a3 SP_R1_AB/AC/BC replay vs npz = 0.0
    a4 MASKS identical across families
    a5 R1 = median(CS1H, axis=1) - med_c replay
       bit (3077 a5 re-check)
    a6 median(DAH34, axis=0) vs DAH34_MED bit
    a7 top8 = argsort(R1)[:8] replay bit
  E2 family level (n=3 pairs, orientation check):
    sp(A_S_f, A_S_g) vs migration SP_R1_fg;
    family-median TT similarity vs migration.
  E3 per-pair migration (CORE, n=24 per family
  pair):
    head level    T_fg[k] = sp(CS1H_f[:,k],
                              CS1H_g[:,k])
    subset level  U_fg[k] = sp(CS_f[:,k],
                               CS_g[:,k])
    predictors (preregistered):
      f1 sTT_fg[k]  = sp(TT_f[k], TT_g[k])
      f2 cTT_fg[k]  = cos(TT_f[k], TT_g[k])
      f3 sLGp_fg[k] = sp(LG_f[pref_k],
                         LG_g[pref_k])
      f4 sLGb_fg[k] = sp(LG_f[base_k],
                         LG_g[base_k])
      f5 amp_fg[k]  = min(|TT_f[k]|,|TT_g[k]|)
                      / max(...)  (control)
    test: spearman(predictor, response) over the
    24 pairs + two-sided permutation p (20000,
    seed 3079).
  E4 structure coupling: sp(U_fg, T_fg) over 24
  pairs (head configuration vs joint-subset
  migration); top8 cross-family means.

Gates (preregistered):
  G1 (main, f1 only): max over family pairs and
      responses of sp(sTT, response) >= 0.5 AND
      perm p < 0.05
  verdict:
    migration_tt_locked : G1 passes with
        positive sign
    migration_tt_anti   : G1 passes with
        negative sign
    migration_uncoupled : G1 fails and ALL
        predictors on ALL responses have
        max |sp| < 0.5
    migration_partial   : otherwise

limitations (recorded): n=24 pairs per family
pair (spearman power moderate); family level
n=3 is orientation only; TT/LG are float32
stored copies (all family comparisons use the
same rounding, no anchor depends on them);
paradigm = causal connectives, shared syntactic
frame; single model qwen3-4b.

memory discipline: no model load, no forwards;
peak memory = TT/LG float64 copies (~300 MB);
freed before npz save.
"""
import gc
import hashlib
import io
import json
import os
import time

import numpy as np

PHASE = 3079
NAME = 'omega_p76_migration_lock'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3079', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
SCRIPT = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3079_omega_p76_migration_lock.py')

SEED = 3079
N_PERM = 1000 if SMOKE else 20000
G1_SP = 0.5
G1_P = 0.05
FKEYS = ('A', 'B', 'C')
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
RDIR = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913')
P76 = os.path.join(
    RDIR, 'phase3076',
    'omega_p73_cross_prompt_family')
P74 = os.path.join(
    RDIR, 'phase3074',
    'omega_p71_capacity_law')
P71 = os.path.join(
    RDIR, 'phase3071',
    'omega_p68_attn_head_decomp')

PREREG = {
    'mode': 'no forwards; frozen-npz '
            're-analysis of 3076 (A_S/R_S/CS/'
            'MASKS/CS1H/R1_ALL32/TT/LG/SP_R1/'
            'TOP8/DAH34/DAH34_MED/MED_C_34) + '
            '3074 (A_S) + 3071 (r34); float32 '
            'TT/LG promoted to float64 for '
            'similarity features (no anchor '
            'depends on them); smoke mode '
            'optional (SMOKE=1: n_perm=1000)',
    'question': '3079 A (menu of 3078): R1 '
                'migrates asymmetrically across '
                'prompt families (A->B 0.676, '
                'A->C 0.130, B->C -0.088).  Is '
                'the migration locked by '
                'spectral-shape similarity '
                '(family-level 255-subset '
                'spectra, per-pair TT direction '
                'similarity, per-pair logits '
                'shape similarity)?  '
                'Family-level comparison is '
                'orientation only (n=3); the '
                'main test is per-pair (n=24): '
                'does the per-pair TT similarity '
                'predict the per-pair head-'
                'configuration migration?',
    'definitions': {
        'T_fg[k]': 'sp(CS1H_f[:,k], CS1H_g[:,k]) '
                   'head-level migration of '
                   'pair k (32 heads)',
        'U_fg[k]': 'sp(CS_f[:,k], CS_g[:,k]) '
                   'subset-level migration '
                   '(255 subsets)',
        'sTT_fg[k]': 'sp(TT_f[k], TT_g[k]) '
                     'per-pair TT rank-shape '
                     'similarity',
        'cTT_fg[k]': 'cos(TT_f[k], TT_g[k])',
        'sLGp/sLGb': 'logits shape similarity '
                     'at the pref/base prompt '
                     'of pair k',
        'amp_fg[k]': 'TT norm ratio control',
    },
    'anchors': {
        'a1': 'A_S_A vs 3074 npz A_S bit 0.0',
        'a2': 'R1_ALL32_A vs 3071 r34 bit 0.0',
        'a3': 'SP_R1_AB/AC/BC replay vs 3076 '
              'npz bit 0.0',
        'a4': 'MASKS identical across families',
        'a5': 'R1 = median(CS1H)-med_c replay '
              'bit 0.0 (3 families)',
        'a6': 'median(DAH34) vs DAH34_MED '
              'replay bit 0.0 (3 families)',
        'a7': 'top8 = argsort(R1)[:8] replay '
              'bit (3 families)',
    },
    'gates': {
        'G1': 'max over (family pair, response '
              'in {T,U}) of sp(f1 sTT, resp) '
              '>= 0.5 AND perm p < 0.05; '
              'sign of the best sp decides '
              'locked vs anti',
        'uncoupled': 'G1 fails AND max |sp| '
                     'over ALL predictors '
                     '(f1-f5) x responses < 0.5',
    },
    'statistics_discipline': 'spearman via '
        'stable argsort ranks; two-sided '
        'permutation p (seed 3079, vectorized '
        'row permutations); G1 evaluated on '
        'f1 only (preregistered main '
        'predictor); f2-f5 exploratory '
        'controls recorded without gating; '
        'family level n=3 orientation only; '
        'a3 anchor replays SP_R1 with the '
        'bit-exact 3076 corrcoef-path '
        'spearman (spearman76) because 3076 '
        'stored SP_R1 via np.corrcoef - the '
        'manual 3077-series spearman agrees '
        'to 1.4e-17 (1 ulp) but not bit',
    'limitations': 'n=24 pairs per family '
        'pair; float32 TT/LG stored copies '
        '(comparisons use identical rounding); '
        'causal-connective paradigm, shared '
        'syntactic frame; single model; '
        'spectral shape is defined on the '
        'focal top-8 subset frame (255 '
        'subsets) - migration of structures '
        'outside the focal frame is not '
        'measured',
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
z74 = np.load(os.path.join(
    P74, 'omega_p71_capacity_law.npz'))
j71 = json.load(io.open(
    os.path.join(P71, 'result.json'),
    encoding='utf-8'))

A_S = {f: z76['A_S_' + f].astype(np.float64)
       for f in FKEYS}
R_S = {f: z76['R_S_' + f].astype(np.float64)
       for f in FKEYS}
CS = {f: z76['CS_' + f].astype(np.float64)
      for f in FKEYS}
CS1H = {f: z76['CS1H_' + f].astype(np.float64)
        for f in FKEYS}
MASKS = {f: z76['MASKS_' + f]
         for f in FKEYS}
R1 = {f: z76['R1_ALL32_' + f]
      .astype(np.float64) for f in FKEYS}
TT32 = {f: z76['TT_' + f].astype(np.float32)
        for f in FKEYS}
LG32 = {f: z76['LG_' + f].astype(np.float32)
        for f in FKEYS}
DAH34 = {f: z76['DAH34_' + f]
         .astype(np.float64) for f in FKEYS}
DAH34M = {f: z76['DAH34_MED_' + f]
          .astype(np.float64) for f in FKEYS}
MEDC = {f: float(z76['MED_C_34_' + f])
        for f in FKEYS}
TOP8 = {f: z76['TOP8_' + f] for f in FKEYS}

# ---------- statistics helpers ----------


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
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


def cosv(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b) / (na * nb)


def spearman76(a, b):
    """Bit-exact replica of the 3076 spearman
    (corrcoef path) - used ONLY for the a3
    replay anchor; E3 statistics use the
    3077-series manual spearman above."""
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


# ---------- E1 anchors ----------
adiff = {}
aok = {}

adiff['a1'] = float(np.max(np.abs(
    A_S['A'] - z74['A_S']
    .astype(np.float64))))
aok['a1'] = bool(adiff['a1'] == 0.0)
r34_71 = np.array(
    j71['stats']['head']['r34'],
    dtype=np.float64)
adiff['a2'] = float(np.max(np.abs(
    R1['A'] - r34_71)))
aok['a2'] = bool(adiff['a2'] == 0.0)
sp_npz = {'AB': float(z76['SP_R1_AB']),
          'AC': float(z76['SP_R1_AC']),
          'BC': float(z76['SP_R1_BC'])}
sp_re = {}
for fa, fb in CPAIRS:
    sp_re[fa + fb] = spearman76(R1[fa],
                                R1[fb])
adiff['a3'] = max(
    abs(sp_re[k] - sp_npz[k])
    for k in sp_re)
aok['a3'] = bool(adiff['a3'] == 0.0)
adiff['a4'] = float(np.max(np.abs(
    MASKS['A'].astype(np.float64)
    - MASKS['B'].astype(np.float64)))) \
    + float(np.max(np.abs(
        MASKS['A'].astype(np.float64)
        - MASKS['C'].astype(np.float64))))
aok['a4'] = bool(adiff['a4'] == 0.0)
d5 = 0.0
for f in FKEYS:
    r5 = np.median(CS1H[f], axis=1) \
        - MEDC[f]
    d5 = max(d5, float(np.max(np.abs(
        r5 - R1[f]))))
adiff['a5'] = d5
aok['a5'] = bool(adiff['a5'] == 0.0)
d6 = 0.0
for f in FKEYS:
    d6 = max(d6, float(np.max(np.abs(
        np.median(DAH34[f], axis=0)
        - DAH34M[f]))))
adiff['a6'] = d6
aok['a6'] = bool(adiff['a6'] == 0.0)
d7 = 0
for f in FKEYS:
    t7 = np.argsort(R1[f])[:8].astype(np.int64)
    d7 += int(np.sum(t7 != TOP8[f]))
adiff['a7'] = d7
aok['a7'] = bool(adiff['a7'] == 0)
for k in ('a1', 'a2', 'a3', 'a4', 'a5', 'a6',
          'a7'):
    log('%s diff=%s ok=%s'
        % (k, adiff[k], aok[k]))

# ---------- pair index map ----------
CIDX = np.array([k // 8 + 1
                 for k in range(24)])
BIDX = np.array([k % 8
                 for k in range(24)])
PREF_K = BIDX * 4 + CIDX
BASE_K = BIDX * 4

TT64 = {f: TT32[f].astype(np.float64)
        for f in FKEYS}
LG64 = {f: LG32[f].astype(np.float64)
        for f in FKEYS}

# ---------- E2 family level ----------
fam_sim = {}
for fa, fb in CPAIRS:
    key = fa + fb
    fam_sim[key] = {
        'sp_AS': spearman(A_S[fa], A_S[fb]),
        'sp_RS': spearman(R_S[fa], R_S[fb]),
        'mig': sp_re[key],
        'tt_med': float(np.median([
            spearman(TT64[fa][k], TT64[fb][k])
            for k in range(24)])),
    }
log('E2 family level:')
for key in ('AB', 'AC', 'BC'):
    v = fam_sim[key]
    log('  %s: sp(A_S)=%.4f mig=%.4f '
        'tt_med=%.4f'
        % (key, v['sp_AS'], v['mig'],
           v['tt_med']))
sp_fam = spearman(
    [fam_sim[k]['sp_AS']
     for k in ('AB', 'AC', 'BC')],
    [fam_sim[k]['mig']
     for k in ('AB', 'AC', 'BC')])
sp_fam_tt = spearman(
    [fam_sim[k]['tt_med']
     for k in ('AB', 'AC', 'BC')],
    [fam_sim[k]['mig']
     for k in ('AB', 'AC', 'BC')])
log('E2 orientation: sp(sp_AS, mig)=%.4f '
    'sp(tt_med, mig)=%.4f (n=3, orientation '
    'only)' % (sp_fam, sp_fam_tt))

# ---------- E3 per-pair migration ----------
T = {}
U = {}
F = {fid: {} for fid in
     ('f1_sTT', 'f2_cTT', 'f3_sLGp',
      'f4_sLGb', 'f5_amp')}
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
        n_fa, n_fb) / np.maximum(
        n_fa, n_fb)
log('E3 per-pair migration:')
for key in ('AB', 'AC', 'BC'):
    log('  %s: T med=%.4f [%+.3f, %+.3f] '
        'n_pos=%d/24 | U med=%.4f '
        '[%+.3f, %+.3f]'
        % (key, float(np.median(T[key])),
           float(T[key].min()),
           float(T[key].max()),
           int((T[key] > 0).sum()),
           float(np.median(U[key])),
           float(U[key].min()),
           float(U[key].max())))

# E3 tests: predictors x responses
test_keys = []
E3 = {}
for resp_name, RESP in (('T', T), ('U', U)):
    for fid in ('f1_sTT', 'f2_cTT',
                'f3_sLGp', 'f4_sLGb',
                'f5_amp'):
        for key in ('AB', 'AC', 'BC'):
            s_ = spearman(F[fid][key],
                          RESP[key])
            p_ = perm_p(F[fid][key],
                        RESP[key])
            E3['%s~%s_%s' % (fid, resp_name,
                             key)] = {
                'sp': s_, 'p': p_}
            test_keys.append((fid, resp_name,
                              key))
log('E3 tests (predictor~response_family '
    'pair): sp (p)')
for tk in test_keys:
    e = E3['%s~%s_%s' % tk]
    log('  %s~%s_%s: %+.4f (%.4f)%s'
        % (tk[0], tk[1], tk[2], e['sp'],
           e['p'],
           ' <-- MAIN' if tk[0] == 'f1_sTT'
           else ''))

# ---------- E4 structure coupling ----------
E4 = {'sp_U_T': {}, 'cross8': {}}
for fa, fb in CPAIRS:
    key = fa + fb
    E4['sp_U_T'][key] = spearman(
        U[key], T[key])
    t8 = TOP8[fa].astype(np.int64)
    rest = np.setdiff1d(
        np.arange(32), t8)
    E4['cross8'][key] = {
        'mean_R1g_top8f': float(
            R1[fb][t8].mean()),
        'mean_R1g_rest': float(
            R1[fb][rest].mean())}
for key in ('AB', 'AC', 'BC'):
    log('E4 %s: sp(U,T)=%.4f | '
        'mean R1_g on top8_f=%.4f vs '
        'rest=%.4f'
        % (key, E4['sp_U_T'][key],
           E4['cross8'][key]
           ['mean_R1g_top8f'],
           E4['cross8'][key]
           ['mean_R1g_rest']))

# ---------- verdict ----------
gates = {}
if not SMOKE:
    best = None
    for key in ('AB', 'AC', 'BC'):
        for rn in ('T', 'U'):
            e = E3['f1_sTT~%s_%s'
                   % (rn, key)]
            cand = (abs(e['sp']), e['sp'],
                    e['p'], rn, key)
            if best is None \
                    or cand[0] > best[0]:
                best = cand
    g1 = bool(best[0] >= G1_SP
              and best[2] < G1_P)
    gmax_all = max(
        abs(E3[k]['sp']) for k in E3)
    gates = {'G1': g1,
             'G1_sp': best[1], 'G1_p': best[2],
             'G1_resp': best[3],
             'G1_pair': best[4],
             'max_abs_sp_all': gmax_all}
    if g1 and best[1] > 0:
        verdict = 'migration_tt_locked'
    elif g1:
        verdict = 'migration_tt_anti'
    elif (not g1) and gmax_all < 0.5:
        verdict = 'migration_uncoupled'
    else:
        verdict = 'migration_partial'
    log('VERDICT: %s (G1=%s best f1 '
        'sp=%+.4f p=%.4f on %s_%s; '
        'max|sp| all=%.4f)'
        % (verdict, g1, best[1], best[2],
           best[3], best[4], gmax_all))
else:
    verdict = 'smoke_pending'
    gates = {'G1': None}
    log('VERDICT: smoke_pending')

# ---------- npz ----------
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'N_PERM': np.int64(N_PERM),
}
for k in ('a1', 'a2', 'a3', 'a4', 'a5', 'a6',
          'a7'):
    save['A' + k[1:].upper() + '_DIFF'] = \
        np.float64(adiff[k])
    save['A' + k[1:].upper() + '_OK'] = \
        np.bool_(aok[k])
for key in ('AB', 'AC', 'BC'):
    save['SP_R1_REPLAY_' + key] = np.float64(
        sp_re[key])
    save['T_' + key] = T[key]
    save['U_' + key] = U[key]
    for fid in F:
        save['%s_%s' % (fid.upper(), key)] = \
            F[fid][key]
    save['SP_AS_' + key] = np.float64(
        fam_sim[key]['sp_AS'])
    save['SP_RS_' + key] = np.float64(
        fam_sim[key]['sp_RS'])
    save['TT_MED_' + key] = np.float64(
        fam_sim[key]['tt_med'])
save['SP_FAM_AS_MIG'] = np.float64(sp_fam)
save['SP_FAM_TT_MIG'] = np.float64(sp_fam_tt)
for tk in test_keys:
    e = E3['%s~%s_%s' % tk]
    save['E3_%s_%s_%s'
         % (tk[0].upper(), tk[1], tk[2])] = \
        np.float64(e['sp'])
    save['E3P_%s_%s_%s'
         % (tk[0].upper(), tk[1], tk[2])] = \
        np.float64(e['p'])
for key in ('AB', 'AC', 'BC'):
    save['SP_UT_' + key] = np.float64(
        E4['sp_U_T'][key])
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
        'diffs': {k: f64(adiff[k])
                  for k in adiff},
        'oks': {k: bool(aok[k])
                for k in aok},
    },
    'stats': {
        'family_level': {
            key: {'sp_AS': f64(
                      fam_sim[key]['sp_AS']),
                  'sp_RS': f64(
                      fam_sim[key]['sp_RS']),
                  'mig': f64(
                      fam_sim[key]['mig']),
                  'tt_med': f64(
                      fam_sim[key]['tt_med'])}
            for key in ('AB', 'AC', 'BC')},
        'sp_fam_AS_mig': f64(sp_fam),
        'sp_fam_tt_mig': f64(sp_fam_tt),
        'T': {key: [f64(v) for v in T[key]]
              for key in ('AB', 'AC', 'BC')},
        'U': {key: [f64(v) for v in U[key]]
              for key in ('AB', 'AC', 'BC')},
        'e3': {k: {'sp': f64(E3[k]['sp']),
                   'p': f64(E3[k]['p'])}
               for k in E3},
        'sp_U_T': {key: f64(E4['sp_U_T'][key])
                   for key in ('AB', 'AC',
                               'BC')},
        'cross8': E4['cross8'],
    },
    'gates': gates,
    'note': '3079 A: spectral-shape similarity '
            'vs R1 migration.  Family level '
            '(n=3): 255-subset shape similarity '
            'does NOT order with migration '
            '(BC most similar, least '
            'migration).  Per-pair (n=24): TT '
            'similarity vs head-config '
            'migration is the preregistered '
            'main test.',
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
