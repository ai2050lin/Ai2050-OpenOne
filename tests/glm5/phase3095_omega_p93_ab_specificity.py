# -*- coding: utf-8 -*-
"""Phase 3095  Ω-P93: AB-specificity anatomy.

3094 found: on qwen3-14b the TT angle
factor f2 collapses on family pairs
AC/BC (median 0.595->0.169, 0.598->0.183)
while AB is preserved (0.534->0.550), and
the sp(f2,U) coupling dies.  Why does AB
survive on the larger model?

Forward-free.  Inputs: sealed npz from
3076 (4B), 3079 (4B stats), 3093 (14B),
3094 (replayed arrays).

KEY ALIGNMENT ARGUMENT (recorded in
execution.json): qwen3-4b and qwen3-14b
share the Qwen3 tokenizer (both vocab
151936), so the TT direction fields live
in the SAME 151936-dim vocab coordinate
system with aligned token semantics.
Vocab-space coordinates are therefore
directly comparable across these two
models (unlike hidden/residual
coordinates; AGENTS.md restriction
applies to hidden states, not to
shared-vocab logits space).  All
cross-model comparisons here are either
vocab-space (TT top-token overlap) or
structure statistics (per-prefix-group
f2 medians) -- never hidden-state
indices.

Preregistered hypotheses (frozen before
compute):
  H_A1 ab_prefix_locked:
     14B f2_AB survival concentrates in
     one prefix group: some ci in
     {1,2,3} has group median >= other
     ci groups' max + 0.15 and >= 0.45,
     while all 14B f2_AC/BC group medians
     < 0.35.
  H_A3 ab_residual (whole-pair
     preservation):
     ALL 14B f2_AB ci-group medians
     >= 0.45 AND all 14B f2_AC/BC group
     medians < 0.35.
  H_A2 ab_token_content:
     top-token overlap AB >= mean(AC,BC)
     + 0.05 on BOTH models (content
     similarity drives alignment).
  fallback: fifth_ab_mixed /
     fifth_inconclusive.

Decision order (frozen): H_A1 ->
  H_A3 -> H_A2 -> mixed -> inconclusive.

Anchors:
  b1: 14B F2 replay identical to the
      3093-sealed F2_CTT arrays (<=1e-9,
      the same comparison 3094 a1 passed)
      plus same-path determinism recompute
      (bit 0); 4B f2 medians cross-checked
      against 3094 sealed result.json
      med_f2_*_4B (<=1e-9).
  b2: CIDX/BIDX definition replay
      (k//8+1, k%8) self-check.
  b3: gate-key replay 14B count=3 /
      G2 4B count=5 (same as 3094 a3).

Stats: n_perm 20000, seed 3095.
"""
import hashlib
import io
import json
import os

import numpy as np

PHASE = 3095
NAME = 'omega_p93_ab_specificity'
SEED = 3095
N_PERM = 20000
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P4B = (R13 + r'\phase3076'
       r'\omega_p73_cross_prompt_family'
       r'\omega_p73_cross_prompt_family.npz')
P14B = (R13 + r'\phase3093'
        r'\omega_p91_qwen14b_l37_full_'
        r'arbitration'
        r'\omega_p91_qwen14b_l37_full_'
        r'arbitration.npz')
P3094 = (R13 + r'\phase3094'
         r'\omega_p92_trunk_anatomy'
         r'\omega_p92_trunk_anatomy.npz')
OUT = (R13 + r'\phase3095' + '\\' + NAME)
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

z4 = np.load(P4B, allow_pickle=False)
z14 = np.load(P14B, allow_pickle=False)
z94 = np.load(P3094, allow_pickle=False)
exec_doc = {
    'phase': PHASE,
    'name': NAME,
    'frozen_before_compute': True,
    'question': ('why does the 14B f2 angle '
                 'factor survive on family '
                 'pair AB but collapse on '
                 'AC/BC'),
    'vocab_alignment': (
        'qwen3-4b and qwen3-14b share the '
        'Qwen3 tokenizer (vocab 151936); '
        'TT direction fields are in the '
        'same vocab coordinate system with '
        'aligned token semantics; no '
        'hidden-state coordinate is '
        'compared cross-model'),
    'hypotheses': {
        'H_A1_ab_prefix_locked': {
            'gap_ge': 0.15, 'abs_ge': 0.45,
            'others_lt': 0.35},
        'H_A3_ab_residual': {
            'all_ab_ge': 0.45,
            'all_acbc_lt': 0.35},
        'H_A2_ab_token_content': {
            'topk': 64,
            'margin_ge': 0.05,
            'both_models': True}},
    'decision_order': ['H_A1', 'H_A3',
                       'H_A2', 'mixed',
                       'inconclusive'],
    'stats': {'n_perm': N_PERM,
              'seed': SEED},
    'inputs': {
        'p4b_npz8': h8(P4B),
        'p14b_npz8': h8(P14B),
        'p3094_npz8': h8(P3094)},
    'forward_free': True}
with io.open(OUT + r'\execution.json', 'w',
             encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=1,
              ensure_ascii=False)
log('execution.json written (prereg '
    'frozen) smoke=False')

CIDX = np.array([k // 8 + 1
                 for k in range(24)])
BIDX = np.array([k % 8
                 for k in range(24)])
PREF_K = BIDX * 4 + CIDX
BASE_K = BIDX * 4
log('pair map: CIDX=%s' % (CIDX.tolist(),))

# ---------- anchors ----------
# b2 definition self-check
assert (CIDX[8] == 2 and CIDX[16] == 3
        and BIDX[9] == 1 and PREF_K[0] == 1
        and BASE_K[0] == 0)
log('b2 pair-map self-check ok')

# b1 arrays vs 3094 npz (which were
# themselves bit-0 vs sealed 3079/3093)
def cosv(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b) / (na * nb)


TT4 = {f: z4['TT_' + f]
       .astype(np.float32)
       .astype(np.float64) for f in FKEYS}
TT14 = {f: z14['TT_' + f]
        .astype(np.float32)
        .astype(np.float64) for f in FKEYS}
F2 = {'4B': {}, '14B': {}}
for fa, fb in (('A', 'B'), ('A', 'C'),
               ('B', 'C')):
    key = fa + fb
    F2['4B'][key] = np.array([
        cosv(TT4[fa][k], TT4[fb][k])
        for k in range(24)])
    F2['14B'][key] = np.array([
        cosv(TT14[fa][k], TT14[fb][k])
        for k in range(24)])
b1_diff = 0.0
# same-path determinism sanity (3094 a1/a2
# proved this path bit-matches sealed F2)
F2_94 = {}
for key in CPAIRS:
    F2_94[key] = np.array([
        cosv(TT14[key[0]][k],
             TT14[key[1]][k])
        for k in range(24)])
    b1_diff = max(b1_diff, float(
        np.max(np.abs(F2['14B'][key]
                      - F2_94[key]))))
    # 14B F2 replay vs 3093-sealed F2_CTT
    b1_diff = max(b1_diff, float(np.max(
        np.abs(F2['14B'][key]
               - z14['F2_CTT_' + key]
               .astype(np.float64)))))
b1_ok = b1_diff <= 1e-9
log('b1 14B F2 vs sealed F2_CTT + '
    'determinism diff=%.3e ok=%s'
    % (b1_diff, b1_ok))

# 4B f2 medians vs 3094 sealed result.json
P3094_RES = (R13 + r'\phase3094'
             r'\omega_p92_trunk_anatomy'
             r'\result.json')
with io.open(P3094_RES, 'r',
             encoding='utf-8') as f:
    z94_res = json.load(f)
b1_4b_diff = 0.0
for key in CPAIRS:
    b1_4b_diff = max(b1_4b_diff, abs(
        float(np.median(F2['4B'][key]))
        - z94_res['stats']['E1']
        ['med_f2_%s_4B' % key]))
b1_4b_ok = b1_4b_diff <= 1e-9
log('b1-4b f2 medians vs 3094 result '
    'diff=%.3e ok=%s'
    % (b1_4b_diff, b1_4b_ok))

b3_ok = (int(z14['GDS_COUNT']) == 3
         and int(np.load(
             R13 + r'\phase3080'
             r'\omega_p77_ab_anatomy'
             r'\omega_p77_ab_anatomy.npz')
             ['G2_COUNT']) == 5)
log('b3 gate replay ok=%s' % b3_ok)
assert b1_ok and b1_4b_ok and b3_ok
log('anchors ok')

# ---------- E1 per-prefix f2 ----------
E1 = {}
log('E1 f2 by prefix group (ci=1,2,3; '
    '8 pairs each):')
for key in CPAIRS:
    for mi, mk in (('4B', '4B'),
                   ('14B', '14B')):
        for ci in (1, 2, 3):
            m = CIDX == ci
            E1['f2_%s_%s_ci%d_med'
               % (mk, key, ci)] = float(
                np.median(F2[mk][key][m]))
    log('  %s: 4B %s | 14B %s'
        % (key,
           ' '.join('%.3f'
                    % E1['f2_4B_%s_ci%d_med'
                         % (key, ci)]
                    for ci in (1, 2, 3)),
           ' '.join('%.3f'
                    % E1['f2_14B_%s_ci%d_med'
                         % (key, ci)]
                    for ci in (1, 2, 3))))
ab_meds = [E1['f2_14B_AB_ci%d_med'
              % ci] for ci in (1, 2, 3)]
h1_best = int(np.argmax(ab_meds))
h1_best_med = max(ab_meds)
h1_other_max = max(
    m for i, m in enumerate(ab_meds)
    if i != h1_best)
acbc_meds = [
    E1['f2_14B_%s_ci%d_med'
       % (key, ci)]
    for key in ('AC', 'BC')
    for ci in (1, 2, 3)]
H_A1 = (h1_best_med
        >= h1_other_max + 0.15
        and h1_best_med >= 0.45
        and max(acbc_meds) < 0.35)
H_A3 = (min(ab_meds) >= 0.45
        and max(acbc_meds) < 0.35)
log('E1 decision: h1_best=ci%d '
    'med=%.3f other_max=%.3f '
    'acbc_max=%.3f | H_A1=%s H_A3=%s'
    % (h1_best + 1, h1_best_med,
       h1_other_max, max(acbc_meds),
       H_A1, H_A3))

# ---------- E2 vocab content ----------
E2 = {}
log('E2 top-64 token overlap '
    '(4B vs 14B, same tokenizer):')
TOPK = 64
for key in CPAIRS:
    fa, fb = key[0], key[1]
    ov = []
    for k in range(24):
        tops = []
        for TT in (TT4, TT14):
            d = TT[fa][k] - TT[fb][k]
            # TT direction per model is
            # TT_f[k] vs TT_g[k] angle --
            # for content, use the mean
            # direction magnitude profile:
            # rank tokens by |TT_f[k]] +
            # TT_g[k]| normalized (the
            # shared movement profile)
            prof = (np.abs(TT[fa][k])
                    + np.abs(TT[fb][k]))
            tops.append(set(
                np.argsort(
                    prof)[-TOPK:]
                .tolist()))
        ov.append(len(
            tops[0] & tops[1]) / TOPK)
    ov = np.array(ov)
    E2['overlap_%s_med' % key] = float(
        np.median(ov))
    E2['overlap_%s_mean' % key] = float(
        np.mean(ov))
    log('  %s: overlap med=%.3f '
        'mean=%.3f'
        % (key, E2['overlap_%s_med'
                   % key],
           E2['overlap_%s_mean'
              % key]))
ab_o = E2['overlap_AB_mean']
oth = (E2['overlap_AC_mean']
       + E2['overlap_BC_mean']) / 2
H_A2 = ab_o >= oth + 0.05
log('E2 decision: AB mean %.3f vs '
    'others %.3f -> H_A2=%s'
    % (ab_o, oth, H_A2))

# ---------- E3 f2~U_AB sign flip ----
E3 = {}
U14_AB = z94['U14_AB']
F2_14_AB = F2['14B']['AB']
for ci in (1, 2, 3):
    m = CIDX == ci
    E3['U14_AB_ci%d_med' % ci] = float(
        np.median(U14_AB[m]))
    E3['f2_14B_AB_ci%d' % ci] = float(
        np.median(F2_14_AB[m]))
log('E3 14B AB by ci: f2 %s | U %s'
    % (' '.join('%.3f'
                % E3['f2_14B_AB_ci%d'
                     % ci]
                for ci in (1, 2, 3)),
       ' '.join('%.3f'
                % E3['U14_AB_ci%d_med'
                     % ci]
                for ci in (1, 2, 3))))
# leave-one-out spearman stability
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


sp_full = spearman(F2_14_AB, U14_AB)
sp_loo = [
    spearman(np.delete(F2_14_AB, k),
             np.delete(U14_AB, k))
    for k in range(24)]
E3['sp_full'] = sp_full
E3['sp_loo_min'] = float(min(sp_loo))
E3['sp_loo_max'] = float(max(sp_loo))
log('E3 f2~U_AB full sp=%+.3f | LOO '
    'range [%+.3f, %+.3f]'
    % (sp_full, min(sp_loo),
       max(sp_loo)))

# ---------- decision ----------
if H_A1:
    VERDICT = 'fifth_ab_prefix_locked'
elif H_A3:
    VERDICT = 'fifth_ab_residual'
elif H_A2:
    VERDICT = 'fifth_ab_token_content'
elif not (H_A1 or H_A3 or H_A2):
    VERDICT = 'fifth_ab_mixed'
else:
    VERDICT = 'fifth_inconclusive'
log('VERDICT: %s' % VERDICT)

# ---------- persist ----------
res = {
    'phase': PHASE, 'name': NAME,
    'verdict': VERDICT, 'forwards': 0,
    'elapsed': 0.0,
    'stats': {'E1': E1, 'E2': E2,
              'E3': E3,
              'decision': {
                  'H_A1': bool(H_A1),
                  'H_A3': bool(H_A3),
                  'H_A2': bool(H_A2),
                  'h1_best_ci': h1_best + 1}},
    'anchors': {'b1_diff': b1_diff,
                'b1_ok': b1_ok,
                'b1_4b_diff': b1_4b_diff,
                'b1_4b_ok': b1_4b_ok,
                'b3_ok': b3_ok}}
with io.open(OUT + r'\result.json', 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=1,
              ensure_ascii=False)
npz_path = OUT + r'\%s.npz' % NAME
np.savez(
    npz_path,
    VERDICT=np.array(VERDICT),
    SMOKE=np.bool_(False),
    PHASE=np.int64(PHASE),
    SEED=np.int64(SEED),
    F2_14_AB=F2['14B']['AB'],
    F2_14_AC=F2['14B']['AC'],
    F2_14_BC=F2['14B']['BC'],
    F2_4_AB=F2['4B']['AB'],
    F2_4_AC=F2['4B']['AC'],
    F2_4_BC=F2['4B']['BC'],
    OV_AB=E2['overlap_AB_med'],
    OV_AC=E2['overlap_AC_med'],
    OV_BC=E2['overlap_BC_med'])
seal = {
    'npz_sha256_8': h8(npz_path),
    'result_sha256_8': h8(
        OUT + r'\result.json'),
    'script_sha256_8': h8(
        os.path.abspath(__file__)),
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
