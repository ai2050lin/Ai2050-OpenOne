# -*- coding: utf-8 -*-
"""Phase 3094  Ω-P92: trunk-anatomy -- why does
the G_DS gate split between two trunk models?

4B  (phase 3080, ab_cos_locked):  f2 six-test
    5/6 significant, min_sp=+0.359 -> gate PASS.
14B (phase 3093, fifth_trunk_no_migrate):
    f2 T-side 3/3 significant, U-side 0/3,
    min_sp=-0.2383 -> gate FAIL.
Both models are spec_class=trunk, both were
 arbitrated at the same criterion layer band.
The gate split is therefore located EXACTLY in
 the U-side (subset-level) migration response.

Forward-free: only sealed npz from 3076 (4B)
 and 3093 (14B) are consumed.

Preregistered hypotheses (frozen in
 execution.json BEFORE any computation):
  H1 u_compression:
     14B subset-level CS columns are
     intra-family redundant -> U_fg[k] =
     sp(CS_f[:,k], CS_g[:,k]) loses dynamic
     range -> f2~U necessarily ns.
     Criteria (ALL required, >=2 of 3 family
     pairs):
       (a) intra_redund_14B - intra_redund_4B
           >= +0.15 (mean off-diagonal
           spearman between CS columns within
           a family)
       (b) std(U_14B) / std(U_4B) <= 0.6
  H2 tt_factor_failed:
     CS structure comparable (redundancy gap
     < 0.15 for >=2 pairs) but the TT angle
     factor f2 degrades on 14B (median f2
     lower by >= 0.1 on >=2 pairs) while
     f2~U replays ns.
  H3 mixed_degradation:
     (a) passes on some pairs, (b)-side
     evidence on others, neither clean.
  fallback inconclusive.

Anchors (bit-level, preregistered):
  a1 14B replay: T/U/F2 per-pair arrays
     recomputed from CS/CS1H/TT npz keys with
     the 3077-series manual spearman +
     cosv on TT64 must equal the sealed
     3093 npz arrays <= 1e-9.
  a2 4B replay: same vs 3079 npz arrays.
  a3 gate replay: 14B GDS_COUNT==3,
     |GDS_MIN_SP - (-0.2383)|<=1e-9;
     4B G2_COUNT==5 (3080 npz).
  a4 SP_UT replay (spearman(U,T) from
     replayed arrays vs 3079 stored)
     <=1e-12.

All new statistics: n_perm=20000, seed=3094.

Verdict naming: {ordinal}_{state} with
 ordinal 'fifth' retained (fifth-spectrum
 follow-up anatomy).
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

PHASE = 3094
NAME = 'omega_p92_trunk_anatomy'
SEED = 3094
N_PERM = 20000
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P4B = (R13 + r'\phase3076'
       r'\omega_p73_cross_prompt_family'
       r'\omega_p73_cross_prompt_family.npz')
P4B79 = (R13 + r'\phase3079'
         r'\omega_p76_migration_lock'
         r'\omega_p76_migration_lock.npz')
P4B80 = (R13 + r'\phase3080'
         r'\omega_p77_ab_anatomy'
         r'\omega_p77_ab_anatomy.npz')
P14B = (R13 + r'\phase3093'
        r'\omega_p91_qwen14b_l37_full_arbitration'
        r'\omega_p91_qwen14b_l37_full_'
        r'arbitration.npz')
OUT = R13 + r'\phase3094' + '\\' + NAME
LOGF = OUT + r'\run_log.txt'
CPAIRS = ('AB', 'AC', 'BC')
FKEYS = ('A', 'B', 'C')
LOGS = []


def log(msg):
    LOGS.append(msg)


def h8(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20),
                        b''):
            h.update(blk)
    return h.hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)

# ---------- prereg freeze ----------
z4 = np.load(P4B, allow_pickle=False)
z4_79 = np.load(P4B79, allow_pickle=False)
z4_80 = np.load(P4B80, allow_pickle=False)
z14 = np.load(P14B, allow_pickle=False)
exec_doc = {
    'phase': PHASE,
    'name': NAME,
    'frozen_before_compute': True,
    'question': ('why does the G_DS gate split '
                 'between two trunk models '
                 '(4B 5/6 pass vs 14B 3/6 fail); '
                 'where does the 14B U-side '
                 'response die'),
    'hypotheses': {
        'H1_u_compression': {
            'a_intra_redund_gap_ge': 0.15,
            'b_std_ratio_le': 0.6,
            'pairs_required': 2},
        'H2_tt_factor_failed': {
            'redund_gap_lt': 0.15,
            'f2_median_drop_ge': 0.10,
            'pairs_required': 2},
        'H3_mixed_degradation': {
            'note': 'mixed evidence'},
        'fallback': 'inconclusive'},
    'decision_rule': (
        'H1 if (a)&(b) on >=2/3 pairs; else H2 '
        'if CS-redundancy gap<0.15 on >=2/3 '
        'AND f2-median drop>=0.10 on >=2/3 '
        'AND 14B f2~U replay 3/3 ns; else H3 '
        'if either mechanism shows >=1 pair '
        'evidence; else inconclusive'),
    'anchors': {
        'a1': '14B T/U/F2 replay vs 3093 npz '
              '<=1e-9',
        'a2': '4B T/U/F2 replay vs 3079 npz '
              '<=1e-9',
        'a3': 'gate replay 14B count=3 '
              'min_sp=-0.2383 (4-digit render, '
              'tol 1e-3); 4B G2_COUNT=5',
        'a4': 'SP_UT 3076 vs 3079 <=1e-12'},
    'stats': {'n_perm': N_PERM, 'seed': SEED},
    'inputs': {
        'p4b_npz8': h8(P4B),
        'p4b79_npz8': h8(P4B79),
        'p4b80_npz8': h8(P4B80),
        'p14b_npz8': h8(P14B)},
    'forward_free': True}
with io.open(OUT + r'\execution.json', 'w',
             encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=1,
              ensure_ascii=False)
log('execution.json written (prereg frozen) '
    'smoke=False')

# ---------- stats helpers (3077 series) --
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


def intra_redund(M):
    """mean off-diagonal spearman between
    columns of M (col = one pair's response
    profile across rows)."""
    k = M.shape[1]
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(spearman(M[:, i],
                                 M[:, j]))
    return float(np.mean(vals)), float(
        np.std(vals))


# ---------- load sealed arrays ----------
CS4 = {f: z4['CS_' + f].astype(np.float64)
       for f in FKEYS}
CS1H4 = {f: z4['CS1H_' + f]
         .astype(np.float64) for f in FKEYS}
TT4 = {f: z4['TT_' + f]
       .astype(np.float32).astype(np.float64)
       for f in FKEYS}
CS14 = {f: z14['CS_' + f].astype(np.float64)
        for f in FKEYS}
CS1H14 = {f: z14['CS1H_' + f]
          .astype(np.float64) for f in FKEYS}
TT14 = {f: z14['TT_' + f]
        .astype(np.float32).astype(np.float64)
        for f in FKEYS}
log('loaded 4B CS %s CS1H %s | 14B CS %s '
    'CS1H %s'
    % (CS4['A'].shape, CS1H4['A'].shape,
       CS14['A'].shape, CS1H14['A'].shape))

# ---------- anchors ----------
adiff = {}
aok = {}


def replay_TU_F2(CS, CS1H, TT):
    T = {}
    U = {}
    F2 = {}
    F1 = {}
    for fa, fb in (('A', 'B'), ('A', 'C'),
                   ('B', 'C')):
        key = fa + fb
        T[key] = np.array([
            spearman(CS1H[fa][:, k],
                     CS1H[fb][:, k])
            for k in range(24)])
        U[key] = np.array([
            spearman(CS[fa][:, k],
                     CS[fb][:, k])
            for k in range(24)])
        F2[key] = np.array([
            cosv(TT[fa][k], TT[fb][k])
            for k in range(24)])
        F1[key] = np.array([
            spearman(TT[fa][k], TT[fb][k])
            for k in range(24)])
    return T, U, F2, F1


T14, U14, F2_14, F1_14 = replay_TU_F2(
    CS14, CS1H14, TT14)
T4, U4, F2_4, F1_4 = replay_TU_F2(
    CS4, CS1H4, TT4)

for key in CPAIRS:
    for nm, rep, sea in (
            ('T', T14[key],
             z14['T_' + key]),
            ('U', U14[key],
             z14['U_' + key]),
            ('F2', F2_14[key],
             z14['F2_CTT_' + key]),
            ('F1', F1_14[key],
             z14['F1_STT_' + key])):
        d = float(np.max(np.abs(
            rep - sea.astype(np.float64))))
        adiff['a1_%s_%s' % (nm, key)] = d
        aok['a1_%s_%s' % (nm, key)] = (
            d <= 1e-9)
    for nm, rep, sea in (
            ('T', T4[key],
             z4_79['T_' + key]),
            ('U', U4[key],
             z4_79['U_' + key]),
            ('F2', F2_4[key],
             z4_79['F2_CTT_' + key]),
            ('F1', F1_4[key],
             z4_79['F1_STT_' + key])):
        d = float(np.max(np.abs(
            rep - sea.astype(np.float64))))
        adiff['a2_%s_%s' % (nm, key)] = d
        aok['a2_%s_%s' % (nm, key)] = (
            d <= 1e-9)
A1_OK = all(v for k, v in aok.items()
            if k.startswith('a1_'))
A2_OK = all(v for k, v in aok.items()
            if k.startswith('a2_'))
log('a1 14B replay ok=%s (max diff %.3e)'
    % (A1_OK, max(adiff[k] for k in adiff
                  if k.startswith('a1_'))))
log('a2 4B replay ok=%s (max diff %.3e)'
    % (A2_OK, max(adiff[k] for k in adiff
                  if k.startswith('a2_'))))

# a3 gate replay (count exact; min_sp vs
# the 4-digit MEMO rendering -0.2383 at
# rendering tolerance 1e-3)
g14_count = int(z14['GDS_COUNT'])
g14_minsp = float(z14['GDS_MIN_SP'])
g4_count = int(z4_80['G2_COUNT'])
a3_ok = (g14_count == 3
         and abs(g14_minsp - (-0.2383))
         <= 1e-3 and g4_count == 5)
log('a3 gate replay: 14B count=%d min_sp=%s '
    '| 4B G2 count=%d ok=%s'
    % (g14_count, g14_minsp, g4_count,
       a3_ok))

# a4 SP_UT replay from the replayed T/U
# arrays vs 3079 sealed values (proves the
# T/U replay path is the same one that
# produced the sealed statistics)
a4_d = max(
    abs(spearman(U4[k], T4[k])
        - float(z4_79['SP_UT_' + k]))
    for k in CPAIRS)
a4_ok = a4_d <= 1e-12
log('a4 SP_UT cross-npz max diff %.3e ok=%s'
    % (a4_d, a4_ok))

assert A1_OK and A2_OK and a3_ok and a4_ok, \
    'anchor failure'
log('anchors all ok')

# ---------- E1 CS structure ----------
E1 = {}
log('E1 CS/CS1H structure (intra-family '
    'column redundancy + U/T column spread):')
for key in CPAIRS:
    fa, fb = key[0], key[1]
    for side, MAT4, MAT14 in (
            ('CS', CS4, CS14),
            ('CS1H', CS1H4, CS1H14)):
        r4, s4 = intra_redund(MAT4[fa])
        r14, s14 = intra_redund(MAT14[fa])
        E1['redund_%s_%s_4B' % (side, key)] \
            = r4
        E1['redund_%s_%s_14B' % (side, key)] \
            = r14
        E1['redund_gap_%s_%s'
           % (side, key)] = r14 - r4
    U4k = U4[key]
    U14k = U14[key]
    T4k = T4[key]
    T14k = T14[key]
    E1['std_U_%s_4B' % key] = float(
        np.std(U4k))
    E1['std_U_%s_14B' % key] = float(
        np.std(U14k))
    E1['std_ratio_U_%s'
       % key] = (float(np.std(U14k))
                 / max(float(np.std(U4k)),
                       1e-12))
    E1['std_T_%s_4B' % key] = float(
        np.std(T4k))
    E1['std_T_%s_14B' % key] = float(
        np.std(T14k))
    E1['med_f2_%s_4B' % key] = float(
        np.median(F2_4[key]))
    E1['med_f2_%s_14B' % key] = float(
        np.median(F2_14[key]))
    E1['f2_med_drop_%s'
       % key] = (float(np.median(F2_4[key]))
                 - float(np.median(
                     F2_14[key])))
    E1['sp_f2U_%s_4B' % key] = spearman(
        F2_4[key], U4k)
    E1['sp_f2U_%s_14B' % key] = spearman(
        F2_14[key], U14k)
    E1['p_f2U_%s_4B' % key] = perm_p(
        F2_4[key], U4k)
    E1['p_f2U_%s_14B' % key] = perm_p(
        F2_14[key], U14k)
    log('  %s: CS redund 4B %+.4f -> 14B '
        '%+.4f (gap %+.4f) | std_U 4B %.4f '
        '-> 14B %.4f (ratio %.3f) | med_f2 '
        '4B %+.4f -> 14B %+.4f | sp(f2,U) '
        '4B %+.3f (p=%.4f) 14B %+.3f '
        '(p=%.4f)'
        % (key,
           E1['redund_CS_%s_4B' % key],
           E1['redund_CS_%s_14B' % key],
           E1['redund_gap_CS_%s' % key],
           E1['std_U_%s_4B' % key],
           E1['std_U_%s_14B' % key],
           E1['std_ratio_U_%s' % key],
           E1['med_f2_%s_4B' % key],
           E1['med_f2_%s_14B' % key],
           E1['sp_f2U_%s_4B' % key],
           E1['p_f2U_%s_4B' % key],
           E1['sp_f2U_%s_14B' % key],
           E1['p_f2U_%s_14B' % key]))

# ---------- E2 TT norm field ----------
E2 = {}
for f in FKEYS:
    n4 = np.linalg.norm(TT4[f], axis=1)
    n14 = np.linalg.norm(TT14[f], axis=1)
    E2['med_ttnorm_%s_4B' % f] = float(
        np.median(n4))
    E2['med_ttnorm_%s_14B' % f] = float(
        np.median(n14))
    E2['ttnorm_ratio_%s'
       % f] = float(np.median(n14)) / float(
        np.median(n4))
log('E2 TT norm field: ' + '; '.join(
    '%s 4B %.1f 14B %.1f (x%.2f)'
    % (f, E2['med_ttnorm_%s_4B' % f],
       E2['med_ttnorm_%s_14B' % f],
       E2['ttnorm_ratio_%s' % f])
    for f in FKEYS))

# ---------- decision ----------
h1_pairs = sum(
    1 for key in CPAIRS
    if E1['redund_gap_CS_%s' % key] >= 0.15
    and E1['std_ratio_U_%s' % key] <= 0.6)
h2_redund_ok = sum(
    1 for key in CPAIRS
    if abs(E1['redund_gap_CS_%s' % key])
    < 0.15)
h2_f2_ok = sum(
    1 for key in CPAIRS
    if E1['f2_med_drop_%s' % key] >= 0.10)
h2_u_ns = all(
    E1['p_f2U_%s_14B' % key] > 0.05
    for key in CPAIRS)
log('decision: h1_pairs=%d h2_redund_ok=%d '
    'h2_f2_ok=%d h2_u_ns=%s'
    % (h1_pairs, h2_redund_ok, h2_f2_ok,
       h2_u_ns))
if h1_pairs >= 2:
    VERDICT = 'fifth_u_compression'
elif (h2_redund_ok >= 2 and h2_f2_ok >= 2
      and h2_u_ns):
    VERDICT = 'fifth_tt_factor_failed'
elif h1_pairs >= 1 or h2_f2_ok >= 1:
    VERDICT = 'fifth_mixed_degradation'
else:
    VERDICT = 'fifth_inconclusive'
log('VERDICT: %s' % VERDICT)

# ---------- result.json ----------
res = {
    'phase': PHASE, 'name': NAME,
    'verdict': VERDICT,
    'forwards': 0,
    'elapsed': 0.0,
    'stats': {
        'E1': E1, 'E2': E2,
        'decision': {
            'h1_pairs': h1_pairs,
            'h2_redund_ok': h2_redund_ok,
            'h2_f2_ok': h2_f2_ok,
            'h2_u_ns': h2_u_ns},
        'T14': {k: T14[k].tolist()
                for k in CPAIRS},
        'U14': {k: U14[k].tolist()
                for k in CPAIRS},
        'U4': {k: U4[k].tolist()
               for k in CPAIRS}},
    'anchors': {
        'diffs': adiff, 'ok': aok,
        'a1_ok': A1_OK, 'a2_ok': A2_OK,
        'a3_ok': a3_ok, 'a4_ok': a4_ok}}
with io.open(OUT + r'\result.json', 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=1,
              ensure_ascii=False)

# ---------- npz ----------
npz_path = OUT + r'\%s.npz' % NAME
np.savez(
    npz_path,
    VERDICT=np.array(VERDICT),
    SMOKE=np.bool_(False),
    PHASE=np.int64(PHASE),
    SEED=np.int64(SEED),
    N_PERM=np.int64(N_PERM),
    REDUND_CS_AB_4B=np.float64(
        E1['redund_CS_AB_4B']),
    REDUND_CS_AB_14B=np.float64(
        E1['redund_CS_AB_14B']),
    REDUND_CS_AC_4B=np.float64(
        E1['redund_CS_AC_4B']),
    REDUND_CS_AC_14B=np.float64(
        E1['redund_CS_AC_14B']),
    REDUND_CS_BC_4B=np.float64(
        E1['redund_CS_BC_4B']),
    REDUND_CS_BC_14B=np.float64(
        E1['redund_CS_BC_14B']),
    STD_U_AB_4B=np.float64(
        E1['std_U_AB_4B']),
    STD_U_AB_14B=np.float64(
        E1['std_U_AB_14B']),
    STD_U_AC_4B=np.float64(
        E1['std_U_AC_4B']),
    STD_U_AC_14B=np.float64(
        E1['std_U_AC_14B']),
    STD_U_BC_4B=np.float64(
        E1['std_U_BC_4B']),
    STD_U_BC_14B=np.float64(
        E1['std_U_BC_14B']),
    U14_AB=U14['AB'], U14_AC=U14['AC'],
    U14_BC=U14['BC'],
    U4_AB=U4['AB'], U4_AC=U4['AC'],
    U4_BC=U4['BC'],
    T14_AB=T14['AB'], T14_AC=T14['AC'],
    T14_BC=T14['BC'],
    T4_AB=T4['AB'], T4_AC=T4['AC'],
    T4_BC=T4['BC'])

# ---------- seal ----------
seal = {
    'npz_sha256_8': h8(npz_path),
    'result_sha256_8': h8(
        OUT + r'\result.json'),
    'script_sha256_8': h8(os.path.abspath(
        __file__)),
    'execution_sha256_8': h8(
        OUT + r'\execution.json')}
with io.open(OUT + r'\seal.json', 'w',
             encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('sealed npz8=%s result8=%s'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8']))

with io.open(LOGF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(LOGS) + '\n')
print('RUN_COMPLETE %s' % VERDICT)
