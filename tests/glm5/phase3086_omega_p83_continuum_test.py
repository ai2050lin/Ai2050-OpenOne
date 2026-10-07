# -*- coding: utf-8 -*-
"""Phase 3086: Omega-P83 quantitative spectrum-
migration continuum test (forward-free, numpy
only).

Question (3086 A, menu of 3085): 3082
introduced the CS-spectrum diagnosis (top3 /
PR) and 3085 closed the four-way arbitration
with THREE model-level points: qwen3-4b
trunk x migrates, DS7B dispersed x
no_migrate, qwen2.5-3b@L34 mixed x absent.
Direction was consistent at all three
points.  This phase upgrades that 3-point
direction claim into a QUANTITATIVE
preregistered test: over the 27 (model,
family-pair) units, does the spectral
low-dimensionality position s_lo = min(CS
top3 of the two families) predict the
migration strength?

Units (frozen): 27 = {4B, DS7B, 3B} x
{AB, AC, BC}.  Responses: T_med = median over
the 24 causal pairs of the head-level
migration sp(CS1H_fa[:,k], CS1H_fb[:,k]);
U_med = median of the subset-level sp(CS
columns); MIG = sp(R1_fa, R1_fb) family
migration reference (3079 SP_R1_REPLAY for
4B, MIG keys for DS7B/3B).  Predictor:
s_lo (main, bottleneck definition) and
s_mean (secondary, recorded).

Sources (frozen, all pre-existing npz):
S79 = phase3079 omega_p76_migration_lock
(4B T/U/F2/SP_R1_REPLAY);
S81 = phase3081 omega_p78_ds7b_crossmodel
(DS7B full set);
S82 = phase3082 omega_p79_ds7b_negative_
anatomy (4B/DS7B spectra + INPUT_SHA chain);
S85 = phase3085 omega_p82_l34_full_
arbitration (3B full set at L_INJ=34).
No new forwards (FORWARDS=0).

Anchors (frozen):
a1 sha chain: S82 INPUT_SHA79 == sha8(S79)
and S82 INPUT_SHA81 == sha8(S81) (the 3082
frozen identity chain must still hold);
a2 spectrum bands: all 4B top3 in (0.95,
0.97), all DS7B top3 in (0.26, 0.33), all
3B top3 in (0.82, 0.88);
a3 shapes: every T_/U_ key is a length-24
vector; FORWARDS: S81 > 20000, S85 > 19000
(S79 is forward-free, no key by design).

Statistics (frozen): manual 3077-series
spearman; permutation p with n_perm=20000,
seed 3086, vectorized (3079-series
implementation); main test rho_T =
sp(s_lo, T_med) over the 27 units;
secondary rho_U, rho_MIG (recorded, not
gating); secondary predictor s_mean recorded
for all three responses.

verdict (preregistered):
  any anchor fail -> continuum_setup_failed
  rho_T > 0 and p < 0.05 ->
      continuum_confirmed
  rho_T > 0 and p >= 0.05 ->
      continuum_positive_ns
  rho_T <= 0 -> continuum_falsified

limitations (recorded): 27 units from 3
models sharing the Qwen lineage (4B and 3B
Qwen family, DS7B Qwen2-backbone distill);
T/U/MIG definitions are isomorphic across
models but head counts differ (28 DS7B vs
16); s_lo threshold-free by construction;
the test is directional (monotone
association), not a causal model; same
frozen 3076 texts across all units;
historical data - the preregistration
covers the TEST DEFINITION and verdict
tree, not blind data (the 3-point direction
was already known from 3082/3085).
"""
import hashlib
import io
import json
import os
import time

import numpy as np

PHASE = 3086
NAME = 'omega_p83_continuum_test'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913')
S79 = os.path.join(
    BASE, 'phase3079',
    'omega_p76_migration_lock',
    'omega_p76_migration_lock.npz')
S81 = os.path.join(
    BASE, 'phase3081',
    'omega_p78_ds7b_crossmodel',
    'omega_p78_ds7b_crossmodel.npz')
S82 = os.path.join(
    BASE, 'phase3082',
    'omega_p79_ds7b_negative_anatomy',
    'omega_p79_ds7b_negative_anatomy.npz')
S85 = os.path.join(
    BASE, 'phase3085',
    'omega_p82_l34_full_arbitration',
    'omega_p82_l34_full_arbitration.npz')
SMOKE = os.environ.get('SMOKE', '0') == '1'
OUT = os.path.join(BASE, 'phase3086', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
SEED = 3086
N_PERM = 1000 if SMOKE else 20000
PAIRS = ('AB', 'AC', 'BC')
FKEYS = ('A', 'B', 'C')
MODELS = (('4B', S79, S82, '4B_',
           'SP_R1_REPLAY'),
          ('DS7B', S81, S82, 'DS7B_', 'MIG'),
          ('3B', S85, S85, '', 'MIG'))
BANDS = {'4B': (0.95, 0.97),
         'DS7B': (0.26, 0.33),
         '3B': (0.82, 0.88)}

PREREG = {
    'mode': 'forward-free numpy analysis; '
            '27 units = 3 models x 3 family '
            'pairs; manual 3077-series '
            'spearman + vectorized '
            'permutation (n_perm %d, seed '
            '%d); sources S79/S81/S82/S85 '
            'frozen with sha8 identity '
            'chain; no model loads, no '
            'GPU' % (N_PERM, SEED),
    'question': '3086 A (menu of 3085): the '
                'four-way arbitration closed '
                'with three direction-'
                'consistent points (4B '
                'trunk x migrates, DS7B '
                'dispersed x no_migrate, 3B '
                'mixed x absent).  Quantify: '
                'does spectral low-'
                'dimensionality s_lo = '
                'min(CS top3 of the pair '
                'families) predict migration '
                'strength T_med over the 27 '
                '(model, pair) units?',
    'units': 'T_med(m,p) = median_k '
             'sp(CS1H_fa[:,k], CS1H_fb[:,k]); '
             'U_med(m,p) = median_k '
             'sp(CS_fa[:,k], CS_fb[:,k]); '
             'MIG(m,p) = sp(R1_fa, R1_fb); '
             's_lo = min(top3_fa, top3_fb) '
             'main; s_mean = mean secondary',
    'anchors': {
        'a1': 'S82 INPUT_SHA79 == sha8(S79) '
              'and S82 INPUT_SHA81 == '
              'sha8(S81) (3082 identity '
              'chain still holds)',
        'a2': 'spectrum bands: 4B top3 in '
              '(0.95, 0.97), DS7B in (0.26, '
              '0.33), 3B in (0.82, 0.88), '
              'all three families',
        'a3': 'T_/U_ keys are length-24 '
              'vectors; FORWARDS S81 > '
              '20000, S85 > 19000 (S79 is '
              'forward-free by design)',
    },
    'verdict': 'anchor fail -> '
               'continuum_setup_failed; '
               'rho_T > 0 and p < 0.05 -> '
               'continuum_confirmed; rho_T '
               '> 0 and p >= 0.05 -> '
               'continuum_positive_ns; '
               'rho_T <= 0 -> '
               'continuum_falsified',
    'statistics_discipline': 'manual '
        'spearman identical to the 3077 '
        'series; perm p = cnt/n_perm no '
        '+1 smoothing; vectorized '
        'permutation on the response '
        'column; s_mean and PR-based '
        'variants recorded NOT gating; '
        'no post-hoc unit exclusions',
    'limitations': '3 models share the '
        'Qwen lineage; head counts differ '
        '(DS7B 28 heads); directional '
        'monotone test, not a causal '
        'model; same 3076 texts; test '
        'definition preregistered over '
        'historical data (direction '
        'already known from 3082/3085)',
    'memory_discipline': 'single process, '
        'numpy only, peak RSS < 1 GB',
}


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


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)
for fn in (NAME + '.npz', 'run_log.txt',
           'execution.json', 'result.json',
           'seal.json'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG,
             'smoke': SMOKE}
with io.open(os.path.join(OUT, 'execution.json'),
             'w', encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False,
              indent=1)

lines = []


def log(msg):
    lines.append(str(msg))
    with io.open(os.path.join(
            OUT, 'run_log.txt'), 'a',
            encoding='utf-8') as f:
        f.write(str(msg) + '\n')


log('execution.json written (prereg frozen) '
    '%s smoke=%s' % (created, SMOKE))

# ==== load sources ====
z79 = np.load(S79, allow_pickle=False)
z81 = np.load(S81, allow_pickle=False)
z82 = np.load(S82, allow_pickle=False)
z85 = np.load(S85, allow_pickle=False)
log('sources loaded')

# ==== anchors ====
anchors = {}
a1 = (str(z82['INPUT_SHA79']) == sha8(S79)
      and str(z82['INPUT_SHA81'])
      == sha8(S81))
anchors['a1_sha_chain'] = bool(a1)
log('a1 sha chain: SHA79=%s vs sha8(S79)=%s '
    '| SHA81=%s vs sha8(S81)=%s -> %s'
    % (str(z82['INPUT_SHA79']), sha8(S79),
       str(z82['INPUT_SHA81']), sha8(S81),
       a1))

ZN = {'4B': z79, 'DS7B': z81, '3B': z85}
ZT = {'4B': z82, 'DS7B': z82, '3B': z85}
top3 = {}
spec_ok = True
for mn, zn, zt, pfk, mref in MODELS:
    s = {fk: float(ZT[mn]['E3_TOP3_CS_' + pfk
                       + fk])
         for fk in FKEYS}
    top3[mn] = s
    lo, hi = BANDS[mn]
    inb = all(lo < v < hi
              for v in s.values())
    spec_ok = spec_ok and inb
    log('a2 %s top3 %s band (%.2f, %.2f) -> '
        'in=%s'
        % (mn, ' '.join('%s=%.4f'
                        % (fk, s[fk])
                        for fk in FKEYS),
           lo, hi, inb))
anchors['a2_spectrum_bands'] = bool(spec_ok)

shape_ok = True
for mn, zn, zt, pfk, mref in MODELS:
    for p in PAIRS:
        shape_ok = shape_ok and \
            len(ZN[mn]['T_' + p]) == 24 and \
            len(ZN[mn]['U_' + p]) == 24
fwd_ok = (int(z81['FORWARDS']) > 20000
          and int(z85['FORWARDS']) > 19000)
anchors['a3_shapes_forwards'] = bool(
    shape_ok and fwd_ok)
log('a3 shapes24=%s forwards(S81=%d S85=%d) '
    '-> %s'
    % (shape_ok, int(z81['FORWARDS']),
       int(z85['FORWARDS']),
       shape_ok and fwd_ok))

setup_ok = all(anchors.values())
log('SETUP_OK=%s' % setup_ok)

# ==== units ====
units = {}
if setup_ok:
    for mn, zn, zt, pfk, mref in MODELS:
        for p in PAIRS:
            fa, fb = p[0], p[1]
            t_med = float(np.median(
                ZN[mn]['T_' + p]))
            u_med = float(np.median(
                ZN[mn]['U_' + p]))
            mig = float(ZN[mn][mref + '_' + p])
            s_lo = float(min(top3[mn][fa],
                             top3[mn][fb]))
            s_mean = float(
                (top3[mn][fa]
                 + top3[mn][fb]) / 2.0)
            key = '%s_%s' % (mn, p)
            units[key] = {
                's_lo': s_lo,
                's_mean': s_mean,
                'T_med': t_med,
                'U_med': u_med,
                'MIG': mig}
            log('unit %s: s_lo=%.4f s_mean='
                '%.4f T_med=%+.4f U_med='
                '%+.4f MIG=%+.4f'
                % (key, s_lo, s_mean,
                   t_med, u_med, mig))

    s_lo = np.array([units[k]['s_lo']
                     for k in sorted(units)])
    s_mean = np.array([units[k]['s_mean']
                       for k in sorted(units)])
    T_med = np.array([units[k]['T_med']
                      for k in sorted(units)])
    U_med = np.array([units[k]['U_med']
                      for k in sorted(units)])
    MIG = np.array([units[k]['MIG']
                    for k in sorted(units)])

    # ==== main test ====
    rho_T = spearman(s_lo, T_med)
    p_T = perm_p(s_lo, T_med)
    rho_U = spearman(s_lo, U_med)
    p_U = perm_p(s_lo, U_med)
    rho_M = spearman(s_lo, MIG)
    p_M = perm_p(s_lo, MIG)
    log('MAIN rho_T=%.4f p=%.6f | '
        'rho_U=%.4f p=%.6f | rho_MIG='
        '%.4f p=%.6f'
        % (rho_T, p_T, rho_U, p_U,
           rho_M, p_M))
    # secondary predictor s_mean
    smT = spearman(s_mean, T_med)
    smU = spearman(s_mean, U_med)
    smM = spearman(s_mean, MIG)
    log('secondary s_mean: sp(T)=%+.4f '
        'sp(U)=%+.4f sp(MIG)=%+.4f'
        % (smT, smU, smM))
    # per-model sanity (recorded)
    for mn in ('4B', 'DS7B', '3B'):
        idx = [i for i, k
               in enumerate(sorted(units))
               if k.startswith(mn)]
        r_in = spearman(s_lo[idx],
                        T_med[idx])
        log('  within-%s (n=3): sp(s_lo, '
            'T_med)=%+.4f'
            % (mn, r_in))

    if rho_T > 0 and p_T < 0.05:
        verdict = 'continuum_confirmed'
    elif rho_T > 0:
        verdict = 'continuum_positive_ns'
    else:
        verdict = 'continuum_falsified'
else:
    s_lo = s_mean = T_med = U_med = MIG = \
        np.array([])
    rho_T = rho_U = rho_M = float('nan')
    p_T = p_U = p_M = float('nan')
    smT = smU = smM = float('nan')
    verdict = 'continuum_setup_failed'
log('VERDICT: %s' % verdict)

# ==== npz ====
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'FORWARDS': np.int64(0),
    'SEED': np.int64(SEED),
    'N_PERM': np.int64(N_PERM),
    'SETUP_OK': np.bool_(setup_ok),
    'N_UNITS': np.int64(len(units)),
    'RHO_T': np.float64(rho_T),
    'P_T': np.float64(p_T),
    'RHO_U': np.float64(rho_U),
    'P_U': np.float64(p_U),
    'RHO_MIG': np.float64(rho_M),
    'P_MIG': np.float64(p_M),
    'SP_MEAN_T': np.float64(smT),
    'SP_MEAN_U': np.float64(smU),
    'SP_MEAN_MIG': np.float64(smM),
    'ANCH_A1': np.bool_(
        anchors['a1_sha_chain']),
    'ANCH_A2': np.bool_(
        anchors['a2_spectrum_bands']),
    'ANCH_A3': np.bool_(
        anchors['a3_shapes_forwards']),
    'SHA79': np.array(sha8(S79)),
    'SHA81': np.array(sha8(S81)),
    'SHA82': np.array(sha8(S82)),
    'SHA85': np.array(sha8(S85)),
}
if setup_ok:
    for i, k in enumerate(
            sorted(units)):
        u = units[k]
        tag = k.replace('DS7B', 'MDS7B') \
            .replace('4B', 'M4B') \
            .replace('3B', 'M3B')
        save['S_LO_' + tag] = np.float64(
            u['s_lo'])
        save['S_MEAN_' + tag] = np.float64(
            u['s_mean'])
        save['T_MED_' + tag] = np.float64(
            u['T_med'])
        save['U_MED_' + tag] = np.float64(
            u['U_med'])
        save['MIG_' + tag] = np.float64(
            u['MIG'])
    save['TOP3_4B'] = np.array(
        [top3['4B'][fk] for fk in FKEYS])
    save['TOP3_DS7B'] = np.array(
        [top3['DS7B'][fk] for fk in FKEYS])
    save['TOP3_3B'] = np.array(
        [top3['3B'][fk] for fk in FKEYS])
np.savez(npz_path, **save)
log('npz saved %s' % npz_path)

# ==== result.json ====
result = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'elapsed': time.time() - t0,
    'forwards': 0,
    'run': 'run1 authoritative (forward-free '
           'numpy analysis over frozen '
           'sources S79/S81/S82/S85)',
    'prereg': PREREG,
    'stats': {
        'anchors': anchors,
        'n_units': len(units),
        'units': {k: {kk: float(vv)
                      for kk, vv in
                      v.items()}
                  for k, v in
                  sorted(units.items())},
        'top3': {mn: {fk: float(v)
                      for fk, v in
                      top3[mn].items()}
                 for mn in top3},
        'tests': {
            'rho_T': float(rho_T),
            'p_T': float(p_T),
            'rho_U': float(rho_U),
            'p_U': float(p_U),
            'rho_MIG': float(rho_M),
            'p_MIG': float(p_M),
            'sp_mean_T': float(smT),
            'sp_mean_U': float(smU),
            'sp_mean_MIG': float(smM),
        },
        'sha': {'S79': sha8(S79),
                'S81': sha8(S81),
                'S82': sha8(S82),
                'S85': sha8(S85)},
    },
    'verdict': verdict,
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)

seal = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(
        os.path.join(OUT, 'result.json')),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(
        __file__)),
    'verdict': verdict,
    'setup_ok': setup_ok,
}
with io.open(os.path.join(OUT, 'seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'],
       time.time() - t0))
log('sealed')
print('RUN_COMPLETE %s' % verdict)
