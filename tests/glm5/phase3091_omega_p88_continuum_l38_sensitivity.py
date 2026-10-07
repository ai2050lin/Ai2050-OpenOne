# -*- coding: utf-8 -*-
"""Phase 3091 - Omega-P88: continuum L38 layer
sensitivity (forward-free).

Question: 3088 established rho(s_lo, T_med)=+0.9021
(n=12, p<5e-5) with GLM4 units at L37.  3089 ran the
full arbitration at L38 (the E1 amplitude-criterion
layer) and landed the SAME verdict cell
(fourth_l38_mixed_absent).  Does the continuum
conclusion survive when the GLM4 units are moved from
L37 to L38 values?

Arms (preregistered in execution.json before any rho
computation):
  A (main, n=12 replacement): replace the 3 GLM4
     units' s_lo / T_med / U_med / MIG with 3089 L38
     values; recompute rho_T + perm p (20000, seed
     3091).  Secondary: rho_U, rho_MIG, s_mean
     variant.
  B (n=15 augmentation): keep the original 12 units
     AND append 3 GLM4-L38 units; recompute rho_T.
  C (descriptive, no gate): Stouffer sign
     sensitivity on 3089's 6 f2_cTT tests - the
     preregistered unsigned Stouffer (reproduced
     bit-level) vs a sign-aware variant.

Verdict tree:
  robust_l38           both arms rho_T>0 & p<0.05
  robust_repl_only     only arm A significant
  fragile_repl         only arm B significant
  broken_l38           neither

Inputs (frozen, sha8 recorded): 3088 npz, 3087 A2
npz, 3089 npz.  No model forwards; GPU untouched.

Preregistration honesty note: the 3089 L38 marginal
spectrum lines (CS top3 / T med / U med) were
observed in run_log before this script was written;
the JOINT continuum recomputation (rho values) was
not observed.  The replacement design was frozen
before any recomputed rho was seen.
"""
import hashlib
import io
import json
import os
import time
from statistics import NormalDist

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P86 = (RES + r'\phase3088\omega_p86_continuum_n12'
       r'\omega_p86_continuum_n12.npz')
P87 = (RES + r'\phase3087'
       r'\omega_p85_glm4_l37_full_arbitration'
       r'\omega_p85_glm4_l37_full_arbitration.npz')
P89 = (RES + r'\phase3089'
       r'\omega_p87_glm4_l38_full_arbitration'
       r'\omega_p87_glm4_l38_full_arbitration.npz')
NAME = 'omega_p88_continuum_l38_sensitivity'
OUT = RES + r'\phase3091' + '\\' + NAME
SEED = 3091
N_PERM = 20000
PAIRS = ('AB', 'AC', 'BC')
# Unit order MUST match 3088's sorted(units)
# key order ('3B'<'4B'<'DS7B'<'GLM4'): the
# frozen spearman ranks ties by argsort
# position (no average-rank correction),
# so rho is order-sensitive under the 4
# tied s_lo groups.  Verified bit-exact
# against stored RHO_T by ANCH_86 below.
MODELS = ('M3B', 'M4B', 'MDS7B', 'MGLM4')

o = []


def log(msg):
    o.append(msg)
    print(msg)


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


# ---- frozen test functions, extracted verbatim
# from the 3088 authoritative source (defaults
# bound at def time; ns pre-seeded) ----
SRC88 = (ROOT + r'\tests\glm5'
         r'\phase3088_omega_p86_continuum_n12.py')
src = io.open(SRC88, encoding='utf-8').read()
i0 = src.index('def spearman(a, b):')
i1 = src.index('def perm_p(a, b', i0)
i2 = src.index('\ndef ', i1 + 10)
frozen = src[i0:i2]
ns = {'np': np, 'N_PERM': N_PERM, 'SEED': SEED}
exec(compile(frozen, 'frozen_3088', 'exec'), ns)
spearman = ns['spearman']
perm_p = ns['perm_p']
log('frozen spearman/perm_p extracted from 3088 '
    'source chars %d-%d' % (i0, i2))

# ---- load frozen inputs ----
z86 = np.load(P86, allow_pickle=False)
z87 = np.load(P87, allow_pickle=False)
z89 = np.load(P89, allow_pickle=False)
sha_in = {'p86': sha8(P86), 'p87': sha8(P87),
          'p89': sha8(P89)}
log('inputs sha8 %s' % sha_in)

# ---- cleanup stale artifacts (rerun rule) ----
os.makedirs(OUT, exist_ok=True)
for fn in (NAME + '.npz', 'result.json',
           'execution.json', 'seal.json',
           'run_log.txt'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')

# ---- preregistration: freeze design BEFORE
# computing any rho ----
design = {
    'phase': 3091,
    'name': 'Omega-P88 continuum L38 layer '
            'sensitivity (forward-free)',
    'created': created,
    'forward_free': True,
    'seed': SEED,
    'n_perm': N_PERM,
    'inputs': sha_in,
    'unit_order': 'sorted-key convention of '
                  '3088: M3B,M4B,MDS7B,MGLM4 '
                  'x AB,AC,BC (frozen argsort '
                  'spearman is tie-order '
                  'sensitive); n=15 arm '
                  'appends MGLM4L38 AB,AC,BC '
                  'at the end',
    'prereg_note': '3089 L38 marginal spectrum '
                   'lines observed in run_log '
                   'before design freeze; joint '
                   'rho recomputation not '
                   'observed. Design frozen '
                   'before any recomputed rho.',
    'arms': {
        'A_repl_n12': 'replace GLM4 units '
                      '(s_lo/T_med/U_med/MIG) '
                      'with 3089 L38 values; '
                      'rho_T + perm_p main',
        'B_aug_n15': 'keep original 12 units, '
                     'append 3 GLM4-L38 units; '
                     'rho_T + perm_p',
        'C_stouffer': 'descriptive: unsigned '
                      '(preregistered 3089 form, '
                      'bit-check vs z89) vs '
                      'sign-aware variant',
    },
    'verdict_tree': [
        'continuum_robust_l38: A and B both '
        'rho_T>0 and p<0.05',
        'continuum_robust_repl_only: A only',
        'continuum_fragile_repl: B only',
        'continuum_broken_l38: neither',
    ],
}
with io.open(os.path.join(OUT, 'execution.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, ensure_ascii=False,
              indent=1)
log('execution.json frozen (before rho '
    'computation)')


def unit_triplets(z, model):
    """per-pair (s_lo, T_med, U_med, MIG) for one
    model tag, reading z86-style keys."""
    out = {}
    for pr in PAIRS:
        out[pr] = (
            float(z['S_LO_%s_%s' % (model, pr)]),
            float(z['T_MED_%s_%s' % (model, pr)]),
            float(z['U_MED_%s_%s' % (model, pr)]),
            float(z['MIG_%s_%s' % (model, pr)]))
    return out


# ---- anchors on frozen data (old results only) --
assert int(z86['N_UNITS']) == 12
assert int(z86['N_PERM']) == 20000
rho86 = float(z86['RHO_T'])

# internal consistency: recompute rho_T from z86
# stored unit arrays -> bit-equal stored value
s_lo_86 = np.array([unit_triplets(z86, m)[pr][0]
                    for m in MODELS
                    for pr in PAIRS])
t_med_86 = np.array([unit_triplets(z86, m)[pr][1]
                     for m in MODELS
                     for pr in PAIRS])
rho86_re = float(spearman(s_lo_86, t_med_86))
anch86 = (rho86_re == rho86)
log('ANCHOR z86 rho_T recompute %.10f vs stored '
    '%.10f bit_eq=%s'
    % (rho86_re, rho86, anch86))
assert anch86

# GLM4 L37 provenance: z86 GLM4 units must equal
# z87-derived values bit-level
top3_37 = [float(z87['E3_TOP3_CS_' + fk])
           for fk in 'ABC']
s_lo_37 = {'AB': min(top3_37[0], top3_37[1]),
           'AC': min(top3_37[0], top3_37[2]),
           'BC': min(top3_37[1], top3_37[2])}
g37 = unit_triplets(z86, 'MGLM4')
anch_glm = all(
    abs(s_lo_37[pr] - g37[pr][0]) == 0.0
    and float(np.median(z87['T_' + pr]))
    == g37[pr][1]
    and float(np.median(z87['U_' + pr]))
    == g37[pr][2]
    and float(z87['MIG_' + pr]) == g37[pr][3]
    for pr in PAIRS)
log('ANCHOR z86 GLM4 units == z87(L37)-derived '
    'bit_eq=%s' % anch_glm)
assert anch_glm

# z89 sanity
assert bool(z89['SMOKE']) is False
assert int(z89['FORWARDS']) == 20922
assert str(z89['VERDICT']) == \
    'fourth_l38_mixed_absent'
assert int(z89['L_INJ']) == 38
assert bool(z89['SETUP_OK'])
assert bool(z89['REPRO_OK'])
log('ANCHOR z89 sanity ok (20922 fw, L38, '
    'mixed_absent)')

# ---- L38 replacement values from 3089 ----
top3_38 = [float(z89['E3_TOP3_CS_' + fk])
           for fk in 'ABC']
s_lo_38 = {'AB': min(top3_38[0], top3_38[1]),
           'AC': min(top3_38[0], top3_38[2]),
           'BC': min(top3_38[1], top3_38[2])}
t38 = {pr: float(np.median(z89['T_' + pr]))
       for pr in PAIRS}
u38 = {pr: float(np.median(z89['U_' + pr]))
       for pr in PAIRS}
mig38 = {pr: float(z89['MIG_' + pr])
         for pr in PAIRS}
s_mean_38 = {pr: float(np.mean(
    [top3_38[0], top3_38[1], top3_38[2]]))
    for pr in PAIRS}
log('L38 values: top3=%s s_lo=%s T_med=%s '
    'U_med=%s MIG=%s'
    % (['%.4f' % v for v in top3_38],
       {k: '%.4f' % v for k, v in
        s_lo_38.items()},
       {k: '%.4f' % v for k, v in t38.items()},
       {k: '%.4f' % v for k, v in u38.items()},
       {k: '%.4f' % v for k, v in
        mig38.items()}))

# ---- Arm A: n=12 replacement ----
non_glm = [m for m in MODELS if m != 'MGLM4']
s_lo_A = list(s_lo_86.copy())
t_med_A = list(t_med_86.copy())
u_med_A = np.array([unit_triplets(z86, m)[pr][2]
                    for m in MODELS
                    for pr in PAIRS]).copy()
mig_A = np.array([unit_triplets(z86, m)[pr][3]
                  for m in MODELS
                  for pr in PAIRS]).copy()
s_mean_A = np.array([
    float(z86['S_MEAN_%s_%s' % (m, pr)])
    for m in MODELS for pr in PAIRS]).copy()
for i, m in enumerate(MODELS):
    if m == 'MGLM4':
        for j, pr in enumerate(PAIRS):
            k = i * 3 + j
            s_lo_A[k] = s_lo_38[pr]
            t_med_A[k] = t38[pr]
            u_med_A[k] = u38[pr]
            mig_A[k] = mig38[pr]
            s_mean_A[k] = s_mean_38[pr]
s_lo_A = np.array(s_lo_A)
t_med_A = np.array(t_med_A)

rho_A = float(spearman(s_lo_A, t_med_A))
p_A = float(perm_p(s_lo_A, t_med_A,
                   N_PERM, SEED))
rho_A_u = float(spearman(s_lo_A, u_med_A))
p_A_u = float(perm_p(s_lo_A, u_med_A,
                     N_PERM, SEED))
rho_A_m = float(spearman(s_lo_A, mig_A))
p_A_m = float(perm_p(s_lo_A, mig_A,
                     N_PERM, SEED))
rho_A_sm = float(spearman(s_mean_A, t_med_A))
p_A_sm = float(perm_p(s_mean_A, t_med_A,
                      N_PERM, SEED))
log('ARM A (n=12 repl): rho_T=%+.4f p=%.5f | '
    'rho_U=%+.4f p=%.5f | rho_MIG=%+.4f p=%.5f '
    '| rho_smean=%+.4f p=%.5f'
    % (rho_A, p_A, rho_A_u, p_A_u, rho_A_m,
       p_A_m, rho_A_sm, p_A_sm))

# ---- Arm B: n=15 augmentation ----
u_med_86 = np.array([unit_triplets(z86, m)[pr][2]
                     for m in MODELS
                     for pr in PAIRS])
mig_86 = np.array([unit_triplets(z86, m)[pr][3]
                   for m in MODELS
                   for pr in PAIRS])
s_lo_B = np.concatenate(
    [s_lo_86,
     np.array([s_lo_38[pr] for pr in PAIRS])])
t_med_B = np.concatenate(
    [t_med_86,
     np.array([t38[pr] for pr in PAIRS])])
u_med_B = np.concatenate(
    [u_med_86,
     np.array([u38[pr] for pr in PAIRS])])
mig_B = np.concatenate(
    [mig_86,
     np.array([mig38[pr] for pr in PAIRS])])
rho_B = float(spearman(s_lo_B, t_med_B))
p_B = float(perm_p(s_lo_B, t_med_B,
                   N_PERM, SEED))
rho_B_u = float(spearman(s_lo_B, u_med_B))
p_B_u = float(perm_p(s_lo_B, u_med_B,
                     N_PERM, SEED))
rho_B_m = float(spearman(s_lo_B, mig_B))
p_B_m = float(perm_p(s_lo_B, mig_B,
                     N_PERM, SEED))
log('ARM B (n=15 aug): rho_T=%+.4f p=%.5f | '
    'rho_U=%+.4f p=%.5f | rho_MIG=%+.4f p=%.5f'
    % (rho_B, p_B, rho_B_u, p_B_u, rho_B_m,
       p_B_m))

# ---- Arm C: Stouffer sign sensitivity ----
ps = [float(z89['E3P_F2_CTT_' + rn + '_' + pr])
      for rn in ('T', 'U') for pr in PAIRS]
sps = [float(z89['E3_F2_CTT_' + rn + '_' + pr])
       for rn in ('T', 'U') for pr in PAIRS]
st_un = float(np.sum(
    [NormalDist().inv_cdf(
        min(1.0 - 1e-12, 1.0 - p)) for p in ps])
    / np.sqrt(6))
st_sig = float(np.sum(
    [(1.0 if s > 0 else -1.0)
     * NormalDist().inv_cdf(
         min(1.0 - 1e-12, 1.0 - p))
     for s, p in zip(sps, ps)])
    / np.sqrt(6))
anch_st = (st_un == float(z89['STOUFFER_Z']))
log('ARM C: stouffer unsigned=%.4f (bit_eq z89 '
    '%s) signed=%+.4f'
    % (st_un, anch_st, st_sig))
assert anch_st

# ---- verdict ----
sig_A = (rho_A > 0 and p_A < 0.05)
sig_B = (rho_B > 0 and p_B < 0.05)
if sig_A and sig_B:
    verdict = 'continuum_robust_l38'
elif sig_A:
    verdict = 'continuum_robust_repl_only'
elif sig_B:
    verdict = 'continuum_fragile_repl'
else:
    verdict = 'continuum_broken_l38'
log('VERDICT: %s (A rho=%+.4f p=%.5f sig=%s; '
    'B rho=%+.4f p=%.5f sig=%s)'
    % (verdict, rho_A, p_A, sig_A, rho_B, p_B,
       sig_B))

elapsed = time.time() - t0

# ---- npz ----
save = {
    'SEED': np.int64(SEED),
    'N_PERM': np.int64(N_PERM),
    'FORWARD_FREE': np.bool_(True),
    'SETUP_OK': np.bool_(True),
    'ANCH_86_RHOT_BIT': np.bool_(anch86),
    'ANCH_GLM4_L37_BIT': np.bool_(anch_glm),
    'ANCH_STOUFFER_BIT': np.bool_(anch_st),
    'N_UNITS_A': np.int64(12),
    'N_UNITS_B': np.int64(15),
    'RHO_86_STORED': np.float64(rho86),
    'TOP3_38': np.array(top3_38,
                        dtype=np.float64),
    'TOP3_37': np.array(top3_37,
                        dtype=np.float64),
    'S_LO_REPL': s_lo_A,
    'T_MED_REPL': t_med_A,
    'U_MED_REPL': u_med_A,
    'MIG_REPL': mig_A,
    'S_MEAN_REPL': s_mean_A,
    'S_LO_N15': s_lo_B,
    'T_MED_N15': t_med_B,
    'RHO_T_REPL': np.float64(rho_A),
    'P_T_REPL': np.float64(p_A),
    'RHO_U_REPL': np.float64(rho_A_u),
    'P_U_REPL': np.float64(p_A_u),
    'RHO_MIG_REPL': np.float64(rho_A_m),
    'P_MIG_REPL': np.float64(p_A_m),
    'RHO_SMEAN_REPL': np.float64(rho_A_sm),
    'P_SMEAN_REPL': np.float64(p_A_sm),
    'RHO_T_N15': np.float64(rho_B),
    'P_T_N15': np.float64(p_B),
    'RHO_U_N15': np.float64(rho_B_u),
    'P_U_N15': np.float64(p_B_u),
    'RHO_MIG_N15': np.float64(rho_B_m),
    'P_MIG_N15': np.float64(p_B_m),
    'STOUFFER_UNSIGNED': np.float64(st_un),
    'STOUFFER_SIGNED': np.float64(st_sig),
    'ELAPSED': np.float64(elapsed),
    'VERDICT': np.str_(verdict),
}
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path, **save)
log('npz saved %s' % npz_path)

# ---- result.json ----
result = {
    'phase': 3091,
    'name': NAME,
    'created': created,
    'forward_free': True,
    'forwards': 0,
    'elapsed': elapsed,
    'seed': SEED,
    'n_perm': N_PERM,
    'inputs_sha8': sha_in,
    'l38_values': {
        'top3_cs': top3_38,
        's_lo': s_lo_38,
        't_med': t38,
        'u_med': u38,
        'mig': mig38,
    },
    'arm_a_repl_n12': {
        'rho_T': rho_A, 'p_T': p_A,
        'rho_U': rho_A_u, 'p_U': p_A_u,
        'rho_MIG': rho_A_m, 'p_MIG': p_A_m,
        'rho_smean': rho_A_sm,
        'p_smean': p_A_sm,
    },
    'arm_b_aug_n15': {
        'rho_T': rho_B, 'p_T': p_B,
        'rho_U': rho_B_u, 'p_U': p_B_u,
        'rho_MIG': rho_B_m, 'p_MIG': p_B_m,
    },
    'arm_c_stouffer': {
        'unsigned': st_un,
        'signed': st_sig,
        'unsigned_bit_eq_3089': bool(anch_st),
    },
    'rho_3088_stored': rho86,
    'verdict': verdict,
    'verdict_tree': design['verdict_tree'],
}
rj = os.path.join(OUT, 'result.json')
with io.open(rj, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)
log('result.json saved')

# ---- seal (key names follow the 3089
# convention: npz_sha256_8/result_sha256_8/
# exec8/script_sha256_8) ----
seal = {
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(rj),
    'exec8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8':
        sha8(os.path.abspath(__file__)),
}
with io.open(os.path.join(OUT, 'seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec8'], seal['script_sha256_8'],
       elapsed))

with io.open(os.path.join(OUT, 'run_log.txt'),
             'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('RUN_COMPLETE %s' % verdict)
