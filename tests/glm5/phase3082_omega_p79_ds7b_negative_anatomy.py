# -*- coding: utf-8 -*-
"""Phase 3082 / Omega-P79: DS7B negative-result
anatomy (no-forward npz re-analysis).

QUESTION (menu 3081 A): WHAT is the shape of the
DS7B cross-family migration absence?
  H_regroup : within-family focal structure is
              healthy but the focal-head sets are
              family-specific (routing regrouped).
  H_decorr  : the within-family causal spectrum
              itself is weak / dispersed.
  H_scale   : spectra healthy, only magnitude
              layer differs.

EVIDENCE BLOCKS (all analytic, no RNG):
  E1 within-family health, DS7B vs 4B per family:
     med_c, r1 min/med, n_neg rate, capture8,
     R_ALL, depth=|r1min|/med_c.
  E2 focal-top8 cross-family overlap (3071
     criterion replayed from R1 vectors):
     observed overlaps vs hypergeometric SF
     (N=heads, K=n=8), random expectation K*n/N.
  E3 causal-spectrum structure: column-centered
     SVD of CS (255x24) and CS1H (Hx24):
     participation ratio, effective rank (exp of
     spectral entropy), top-3 energy share.
  E4 T/U migration-spectrum dispersion vs the
     H0 spearman noise floor 1/sqrt(23).
  E5 TT-direction spectrum: med f2 per pair
     block, TT norm scale, sp(f1,f2) coupling.
  E6 cross-model conservative test: f2~U_AB on
     BOTH models (4B 3079 npz vs DS7B 3081 npz).

PREREGISTERED VERDICT TREE:
  D  = (PR_CS_DS7B[f] - PR_CS_4B[f] >= 0.15 for
        all three families)
  R1 = (hypergeom SF > 0.05 for all 3 DS7B
        overlaps) AND (SF < 0.05 for 4B AB or AC)
  R2 = (SP_AS < 0.2 for all DS7B pairs) AND
       (SP_AS > 0.4 for all 4B pairs)
  R3 = (capture8 >= 0.5 for all DS7B families)
  verdict:
    ds7b_decorrelated            if D
    ds7b_family_specific_routing if R1 and R2
                                 and R3
    ds7b_mixed_anatomy           otherwise

INPUTS (frozen by sha8):
  3081 DS7B npz, 3076 4B npz, 3079 4B npz,
  3080 4B npz (TT norm medians).
No model load, no forwards; deterministic.

Memory discipline: np.load lazily; only the
needed keys are materialized (TT/LG/ZH tensors
are skipped entirely; already-reduced F arrays
from 3079 are used for f1/f2).
"""
import hashlib
import io
import json
import os
import time
from math import comb, log, sqrt

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P81 = BASE + (r'\phase3081\omega_p78_'
              r'ds7b_crossmodel')
P76 = BASE + (r'\phase3076\omega_p73_'
              r'cross_prompt_family')
P79 = BASE + (r'\phase3079\omega_p76_'
              r'migration_lock')
NPZ81 = P81 + r'\omega_p78_ds7b_crossmodel.npz'
NPZ76 = P76 + (r'\omega_p73_cross_prompt_'
               r'family.npz')
NPZ79 = P79 + r'\omega_p76_migration_lock.npz'
P80 = BASE + (r'\phase3080\omega_p77_'
              r'ab_anatomy')
NPZ80 = P80 + r'\omega_p77_ab_anatomy.npz'
NAME = 'omega_p79_ds7b_negative_anatomy'
OUT = BASE + r'\phase3082' + '\\' + NAME
PHASE = 3082
SEED = 3082

FK = ('A', 'B', 'C')
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
NOISE_SP = 1.0 / sqrt(23.0)

lines = []


def log(msg):
    lines.append(str(msg))
    with io.open(OUT + r'\run_log.txt', 'a',
                 encoding='utf-8') as f:
        f.write(str(msg) + '\n')


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


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


def hypergeom_sf(x, N, K, n):
    """P(X >= x), X~Hypergeom(N, K, n)."""
    lo = max(x, 0)
    hi = min(K, n)
    tot = comb(N, n)
    s = 0
    for k in range(lo, hi + 1):
        s += comb(K, k) * comb(N - K, n - k)
    return s / tot


def spectrum_struct(M):
    """Column-centered SVD structure of M."""
    M = np.asarray(M, dtype=np.float64)
    Mc = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Mc,
                      compute_uv=False)
    lam = s * s
    tot = float(lam.sum())
    pr = float(lam.sum() ** 2
               / (lam * lam).sum())
    p = lam / tot
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    keff = float(np.exp(H))
    top3 = float(lam[:3].sum() / tot)
    return pr, keff, top3


# ==== execution freeze ====
os.makedirs(OUT, exist_ok=True)
for fn in ('result.json', NAME + '.npz',
           'seal.json'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
created = time.strftime(
    '%Y-%m-%d %H:%M:%S')
t0 = time.time()

in_sha = {
    'npz3081_ds7b': sha8(NPZ81),
    'npz3076_4b': sha8(NPZ76),
    'npz3079_4b': sha8(NPZ79),
    'npz3080_4b': sha8(NPZ80),
}
prereg = {
    'question':
        '3082 A (menu of 3081): what is the '
        'SHAPE of the DS7B cross-family '
        'migration absence - focal-head '
        'regrouping (family-specific routing), '
        'spectrum decorrelation, or magnitude '
        'scale?',
    'mode':
        'no-forward npz re-analysis; inputs = '
        '3081 DS7B authoritative npz + 3076 4B '
        'authoritative npz + 3079 4B npz + '
        '3080 4B npz; '
        'deterministic analytic statistics '
        '(hypergeometric SF, column-centered '
        'SVD structure, descriptive dispersion); '
        'no RNG; seed %d unused but fixed'
        % SEED,
    'evidence_blocks': {
        'E1': 'within-family health per family '
              'and model: med_c, r1 min/med, '
              'n_neg rate, capture8, R_ALL, '
              'depth=|r1min|/med_c',
        'E2': 'focal top8 cross-family overlap '
              '(3071 criterion replayed from '
              'R1 vectors, checked against npz '
              'TOP8) vs hypergeometric SF '
              'P(X>=obs), N=heads, K=n=8; '
              'random expectation K*n/N',
        'E3': 'column-centered SVD of CS '
              '(255x24) and CS1H (Hx24): '
              'participation ratio, effective '
              'rank, top-3 energy share',
        'E4': 'T/U (24,) dispersion per pair '
              'and model vs H0 spearman noise '
              'floor 1/sqrt(23)=%.4f'
              % NOISE_SP,
        'E5': 'med f2 per pair block, TT norm '
              'scale, sp(f1,f2) coupling',
        'E6': 'cross-model conservative test: '
              'f2~U_AB on both models (4B '
              '3079 E3 npz vs DS7B 3081 E3 '
              'npz)',
    },
    'verdict_tree': {
        'D': 'all(PR_CS_DS7B[f]-PR_CS_4B[f] '
             '>= 0.15 for f in A,B,C)',
        'R1': 'hypergeom SF > 0.05 for all 3 '
              'DS7B overlaps AND SF < 0.05 for '
              '4B AB or AC',
        'R2': 'SP_AS < 0.2 all DS7B pairs AND '
              'SP_AS > 0.4 all 4B pairs '
              '(recomputed spearman on A_S '
              'vectors, checked against npz '
              'SP_AS)',
        'R3': 'capture8 >= 0.5 all DS7B '
              'families',
        'verdict':
            'ds7b_decorrelated if D; elif '
            'R1 and R2 and R3 -> '
            'ds7b_family_specific_routing; '
            'else ds7b_mixed_anatomy',
    },
    'independence':
        'all inputs are frozen sealed npz '
        'artifacts of completed phases; no '
        'activation data is generated; verdict '
        'tree fixed before evaluation',
    'limitations':
        'two models only (form decided on this '
        'pair); layer positions differ (4B '
        'L34/35 of 36 vs DS7B L25/26 of 28); '
        'head counts differ (32 vs 28) so '
        'overlap statistics use per-model N; '
        'same shared syntactic frame as 3076; '
        'E3 SVD structure is descriptive, not '
        'a mechanism claim',
    'input_sha8': in_sha,
}
exe = {'name': NAME, 'phase': PHASE,
       'created': created, 'prereg': prereg,
       'mode': 'full'}
with io.open(OUT + r'\execution.json', 'w',
             encoding='utf-8') as f:
    json.dump(exe, f, ensure_ascii=False,
              indent=1)
log('execution.json written (prereg frozen) %s'
    % created)

# ==== load inputs (lazy, key-selected) ====
Z81 = np.load(NPZ81, allow_pickle=True)
Z76 = np.load(NPZ76, allow_pickle=True)
Z79 = np.load(NPZ79, allow_pickle=True)
Z80 = np.load(NPZ80, allow_pickle=True)

R = {'DS7B': {}, '4B': {}}
HDR = {'DS7B': 28, '4B': 32}
ZS = {'DS7B': Z81, '4B': Z76}

for mdl in ('DS7B', '4B'):
    z = ZS[mdl]
    h = HDR[mdl]
    R[mdl]['heads'] = h
    for fk in FK:
        R[mdl]['r1_' + fk] = np.asarray(
            z['R1_ALL%d_%s' % (h, fk)],
            dtype=np.float64)
        R[mdl]['top8_' + fk] = [
            int(v) for v
            in z['TOP8_' + fk]]
        R[mdl]['n_neg_' + fk] = int(
            z['N_NEG_' + fk])
        R[mdl]['cap8_' + fk] = float(
            z['CAPTURE8_' + fk])
        R[mdl]['rall_' + fk] = float(
            z['R_ALL_' + fk])
        R[mdl]['cs_' + fk] = np.asarray(
            z['CS_' + fk], dtype=np.float64)
        R[mdl]['cs1h_' + fk] = np.asarray(
            z['CS1H_' + fk],
            dtype=np.float64)
        R[mdl]['as_' + fk] = np.asarray(
            z['A_S_' + fk], dtype=np.float64)
    if mdl == 'DS7B':
        for fk in FK:
            R[mdl]['medc_' + fk] = float(
                z['MED_C_' + fk])
    else:
        for fk in FK:
            R[mdl]['medc_' + fk] = float(
                z['MED_C_34_' + fk])

stats = {'E1': {}, 'E2': {}, 'E3': {},
         'E4': {}, 'E5': {}, 'E6': {}}

# ==== E1 within-family health ====
log('E1 within-family health (DS7B vs 4B):')
log('  mdl  fam  med_c   r1min   r1med  '
    'n_neg  cap8   R_ALL  depth')
for mdl in ('DS7B', '4B'):
    for fk in FK:
        r1 = R[mdl]['r1_' + fk]
        medc = R[mdl]['medc_' + fk]
        r1min = float(r1.min())
        r1med = float(np.median(r1))
        depth = abs(r1min) / medc
        e = {'med_c': medc,
             'r1_min': r1min,
             'r1_med': r1med,
             'n_neg': R[mdl]['n_neg_' + fk],
             'n_heads': R[mdl]['heads'],
             'capture8': R[mdl]['cap8_' + fk],
             'R_ALL': R[mdl]['rall_' + fk],
             'depth': depth}
        stats['E1'][mdl + '_' + fk] = e
        log('  %-4s %-4s %7.4f %+7.4f %+7.4f '
            '%3d/%2d %.4f %+7.4f %.3f'
            % (mdl, fk, medc, r1min, r1med,
               e['n_neg'], e['n_heads'],
               e['capture8'], e['R_ALL'],
               depth))

# ==== E2 focal-top8 overlap ====
log('E2 focal top8 cross-family overlap '
    '(3071 criterion replay):')
for mdl in ('DS7B', '4B'):
    h = R[mdl]['heads']
    for fk in FK:
        r1 = R[mdl]['r1_' + fk]
        n_neg = int((r1 < 0).sum())
        assert n_neg == R[mdl]['n_neg_' + fk]
        topk = min(8, n_neg)
        order = np.argsort(r1)
        top8 = [int(v) for v
                in order[:topk]]
        assert top8 == R[mdl]['top8_' + fk], \
            (mdl, fk, top8)
        assert all(r1[top8] < 0) if topk == 8 \
            else True
for mdl in ('DS7B', '4B'):
    h = R[mdl]['heads']
    exp = 8.0 * 8.0 / h
    for fa, fb in CPAIRS:
        key = fa + fb
        ov = len(set(R[mdl]['top8_' + fa])
                 & set(R[mdl]['top8_' + fb]))
        p = hypergeom_sf(ov, h, 8, 8)
        stats['E2'][mdl + '_' + key] = {
            'overlap': ov, 'N': h,
            'expect': exp, 'sf_p': p}
        log('  %s %s: ov=%d (expect %.2f) '
            'hypergeom SF p=%.4f'
            % (mdl, key, ov, exp, p))

# ==== E3 spectrum structure ====
log('E3 column-centered SVD structure '
    '(PR / k_eff / top3):')
for mdl in ('DS7B', '4B'):
    for fk in FK:
        pr, keff, t3 = spectrum_struct(
            R[mdl]['cs_' + fk])
        prh, keffh, t3h = spectrum_struct(
            R[mdl]['cs1h_' + fk])
        stats['E3'][mdl + '_CS_' + fk] = {
            'PR': pr, 'keff': keff,
            'top3': t3}
        stats['E3'][mdl + '_CS1H_' + fk] = {
            'PR': prh, 'keff': keffh,
            'top3': t3h}
        log('  %s %s CS  : PR=%.3f keff=%.3f '
            'top3=%.3f'
            % (mdl, fk, pr, keff, t3))
        log('  %s %s CS1H: PR=%.3f keff=%.3f '
            'top3=%.3f'
            % (mdl, fk, prh, keffh, t3h))

# ==== E4 T/U dispersion ====
log('E4 T/U dispersion (std vs H0 noise '
    'floor %.4f):' % NOISE_SP)
for tag, zp in (('4B', Z79),
                ('DS7B', Z81)):
    for rn in ('T', 'U'):
        for fa, fb in CPAIRS:
            key = fa + fb
            arr = np.asarray(
                zp[rn + '_' + key],
                dtype=np.float64)
            sd = float(arr.std(ddof=1))
            stats['E4'][
                '%s_%s_%s' % (tag, rn, key)] = {
                'std': sd,
                'med': float(np.median(arr)),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'n_pos': int((arr > 0).sum()),
                'snr': sd / NOISE_SP}
            log('  %s %s_%s: std=%.4f snr=%.2f '
                'med=%+.4f [%+.3f,%+.3f] '
                'n_pos=%d/24'
                % (tag, rn, key, sd,
                   sd / NOISE_SP,
                   float(np.median(arr)),
                   arr.min(), arr.max(),
                   int((arr > 0).sum())))

# ==== E5 TT direction spectrum ====
log('E5 f2 med / TT norm / sp(f1,f2):')
for tag, zp, zt in (('4B', Z79, None),
                    ('DS7B', Z81, Z81)):
    for fa, fb in CPAIRS:
        key = fa + fb
        f2 = np.asarray(
            zp['F2_CTT_' + key],
            dtype=np.float64)
        f1 = np.asarray(
            zp['F1_STT_' + key],
            dtype=np.float64)
        sf12 = spearman(f1, f2)
        stats['E5'][
            'med_f2_%s_%s' % (tag, key)] = \
            float(np.median(f2))
        stats['E5'][
            'sp_f1f2_%s_%s' % (tag, key)] = \
            sf12
        log('  %s %s: med f2=%.4f '
            'sp(f1,f2)=%.4f'
            % (tag, key, float(np.median(f2)),
               sf12))
for tag, zp, tkey in (
        ('4B', Z80, 'TT_NORM_MED_'),
        ('DS7B', Z81, 'MED_TT_NORM_')):
    for fk in FK:
        mn = float(zp[tkey + fk])
        stats['E5'][
            'tt_norm_med_%s_%s'
            % (tag, fk)] = mn
        log('  %s %s: med TT norm = %.1f'
            % (tag, fk, mn))

# ==== E6 cross-model conservative test ====
f2u_4b_sp = float(
    Z79['E3_F2_CTT_U_AB'])
f2u_4b_p = float(
    Z79['E3P_F2_CTT_U_AB'])
f2u_ds_sp = float(
    Z81['E3_F2_CTT_U_AB'])
f2u_ds_p = float(
    Z81['E3P_F2_CTT_U_AB'])
stats['E6'] = {
    'f2u_AB_4B': {'sp': f2u_4b_sp,
                  'p': f2u_4b_p},
    'f2u_AB_DS7B': {'sp': f2u_ds_sp,
                    'p': f2u_ds_p},
    'both_significant': bool(
        f2u_4b_sp > 0 and f2u_4b_p < 0.05
        and f2u_ds_sp > 0
        and f2u_ds_p < 0.05),
}
log('E6 f2~U_AB: 4B %+.4f (p=%.5f) | DS7B '
    '%+.4f (p=%.5f) -> both significant: %s'
    % (f2u_4b_sp, f2u_4b_p, f2u_ds_sp,
       f2u_ds_p,
       stats['E6']['both_significant']))

# ==== SP_AS recompute (R2 input) ====
# reference values: 4B from 3079 npz, DS7B
# from 3081 npz
SP_AS_REF = {'4B': Z79, 'DS7B': Z81}
sp_as = {'DS7B': {}, '4B': {}}
for mdl in ('DS7B', '4B'):
    for fa, fb in CPAIRS:
        key = fa + fb
        v = spearman(R[mdl]['as_' + fa],
                     R[mdl]['as_' + fb])
        ref = float(
            SP_AS_REF[mdl]['SP_AS_' + key])
        d = abs(v - ref)
        assert d < 1e-10, (mdl, key, v, ref)
        sp_as[mdl][key] = v
        log('SP_AS %s %s: %.4f (npz %.4f '
            'replay diff %.1e)'
            % (mdl, key, v, ref, d))
stats['SP_AS'] = {m: dict(sp_as[m])
                  for m in sp_as}

# ==== verdict tree ====
D = all(
    stats['E3']['DS7B_CS_' + fk]['PR']
    - stats['E3']['4B_CS_' + fk]['PR']
    >= 0.15 for fk in FK)
R1 = all(
    stats['E2']['DS7B_' + fa + fb]['sf_p']
    > 0.05 for fa, fb in CPAIRS) and any(
    stats['E2']['4B_' + fa + fb]['sf_p']
    < 0.05 for fa, fb in CPAIRS)
R2 = all(sp_as['DS7B'][fa + fb] < 0.2
         for fa, fb in CPAIRS) and all(
    sp_as['4B'][fa + fb] > 0.4
    for fa, fb in CPAIRS)
R3 = all(R['DS7B']['cap8_' + fk] >= 0.5
         for fk in FK)
if D:
    verdict = 'ds7b_decorrelated'
elif R1 and R2 and R3:
    verdict = 'ds7b_family_specific_routing'
else:
    verdict = 'ds7b_mixed_anatomy'
stats['tree'] = {'D': bool(D), 'R1': bool(R1),
                 'R2': bool(R2), 'R3': bool(R3)}
log('VERDICT TREE: D=%s R1=%s R2=%s R3=%s'
    % (D, R1, R2, R3))
log('VERDICT: %s' % verdict)

elapsed = time.time() - t0

# ==== persist npz ====
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(elapsed),
    'PHASE': np.int64(PHASE),
    'INPUT_SHA81': np.array(
        in_sha['npz3081_ds7b']),
    'INPUT_SHA76': np.array(
        in_sha['npz3076_4b']),
    'INPUT_SHA79': np.array(
        in_sha['npz3079_4b']),
    'TREE_D': np.bool_(D),
    'TREE_R1': np.bool_(R1),
    'TREE_R2': np.bool_(R2),
    'TREE_R3': np.bool_(R3),
}
for mdl in ('DS7B', '4B'):
    for fk in FK:
        pfx = mdl + '_' + fk
        e = stats['E1'][pfx]
        for kk, vv in e.items():
            save['E1_' + pfx + '_' + kk] = \
                np.float64(vv) \
                if not isinstance(vv, int) \
                else np.int64(vv)
        pr, keff, t3 = spectrum_struct(
            R[mdl]['cs_' + fk])
        save['E3_PR_CS_' + pfx] = \
            np.float64(pr)
        save['E3_KEFF_CS_' + pfx] = \
            np.float64(keff)
        save['E3_TOP3_CS_' + pfx] = \
            np.float64(t3)
        prh, keffh, t3h = spectrum_struct(
            R[mdl]['cs1h_' + fk])
        save['E3_PR_CS1H_' + pfx] = \
            np.float64(prh)
        save['E3_KEFF_CS1H_' + pfx] = \
            np.float64(keffh)
        save['E3_TOP3_CS1H_' + pfx] = \
            np.float64(t3h)
for mdl in ('DS7B', '4B'):
    h = R[mdl]['heads']
    for fa, fb in CPAIRS:
        key = fa + fb
        e = stats['E2'][mdl + '_' + key]
        save['E2_OV_' + mdl + '_' + key] = \
            np.int64(e['overlap'])
        save['E2_P_' + mdl + '_' + key] = \
            np.float64(e['sf_p'])
for key in ('AB', 'AC', 'BC'):
    save['E4_STD_T_4B_' + key] = np.float64(
        stats['E4']['4B_T_' + key]['std'])
    save['E4_STD_U_4B_' + key] = np.float64(
        stats['E4']['4B_U_' + key]['std'])
    save['E4_STD_T_DS7B_' + key] = \
        np.float64(
            stats['E4']['DS7B_T_' + key]
            ['std'])
    save['E4_STD_U_DS7B_' + key] = \
        np.float64(
            stats['E4']['DS7B_U_' + key]
            ['std'])
    for mdl in ('DS7B', '4B'):
        save['E5_MED_F2_' + mdl + '_' + key] \
            = np.float64(
                stats['E5']['med_f2_%s_%s'
                            % (mdl, key)])
        save['E5_SPF1F2_' + mdl + '_' + key] \
            = np.float64(
                stats['E5']['sp_f1f2_%s_%s'
                            % (mdl, key)])
        save['SP_AS_' + mdl + '_' + key] = \
            np.float64(sp_as[mdl][key])
for mdl in ('DS7B', '4B'):
    for fk in FK:
        save['E5_TTNORM_' + mdl + '_' + fk] = \
            np.float64(
                stats['E5']['tt_norm_med_%s_%s'
                            % (mdl, fk)])
save['E6_SP_4B'] = np.float64(f2u_4b_sp)
save['E6_P_4B'] = np.float64(f2u_4b_p)
save['E6_SP_DS7B'] = np.float64(f2u_ds_sp)
save['E6_P_DS7B'] = np.float64(f2u_ds_p)
npz_path = OUT + '\\' + NAME + '.npz'
np.savez_compressed(npz_path, **save)

# ==== result.json ====
result = {
    'phase': PHASE,
    'name': NAME,
    'created': created,
    'forwards': 0,
    'elapsed': elapsed,
    'verdict': verdict,
    'stats': stats,
    'inputs': in_sha,
}
res_path = OUT + r'\result.json'
with io.open(res_path, 'w',
             encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)

# ==== seal ====
seal = {
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'script_sha256_8': sha8(
        ROOT + r'\tests\glm5\phase3082_'
        r'omega_p79_ds7b_negative_anatomy.py'),
    'exec_sha256_8': sha8(
        OUT + r'\execution.json'),
}
with io.open(OUT + r'\seal.json', 'w',
             encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('sealed npz8=%s result8=%s script8=%s '
    'exec8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['script_sha256_8'],
       seal['exec_sha256_8'], elapsed))
with io.open(OUT + r'\run_log.txt', 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('PHASE3082_OK ' + verdict)
