# -*- coding: utf-8 -*-
"""Phase 3075: Omega-P72 supermodular structure -
localize WHERE the 26.5 percent submodularity
violations of 3074 live, via the full Moebius
spectrum of the measured set function A(S)
(qwen3-4b single model, NO forwards: pure
re-analysis of the frozen 3074 npz).

Question (3075 A, menu of 3074): the head-level
causal amplitude A(S) = -R(S) over all 255
non-empty subsets of the focal top-8 violated
4359/16472 submodularity inequalities.  A set
function is submodular iff ALL its Moebius
coefficients of order >= 2 are <= 0.  So the
violations must come from positive-order-2+
Mobius mass.  WHERE is it?  WHICH heads form
cooperative write cliques, and do they follow
GQA/quarter position structure or effect-size
structure?

E1 ANCHORS (bit-exact replay of 3074): rebuild
the deterministic (S, x, T) enumeration (length
16472), recompute every marginal difference and
compare bit-exact vs the 3074 npz PAIRS4 (a1);
recompute n_viol/viol_rate vs the 3074 result
(a2); A(255) vs result a_u8 (a3); npz R_ALL vs
3071 gA recov -0.5717521069904176 (a4); 3073
r1[0] file anchor (a5); Mobius pair coefficients
mu2 vs R1_73[i]+R1_73[j]-R2_73[p] from the 3073
result (a6); fast-Moebius vs brute-force sum on
all 255 masks bit 0.0 (b0).
E2 MOEBIUS SPECTRUM: brute-force mu(S) =
sum_{T subset S} (-1)^(|S|-|T|) A(T) for all
255 masks (3^8 = 6561 terms); order-wise stats
(2..8): n_pos/n_neg at TOL_MOB = 0.02, max/min;
top-12 positive coefficients with head names;
concentration c10 = top-10 positive sum over
total positive sum.
E3 MARGINAL VIOLATION LOCALIZATION: per added
head x (viol count/sum/max), per |S| bucket,
per budget bucket x(S) (quartiles of the Hill
input), worst-10 inequalities.
E4 CONDITIONAL SYNERGY PROFILES: for each of
the 28 pairs, SM(i,j|S) = A(S+ij)-A(S+i)-A(S+j)
+A(S) over all 64 bases S (i,j not in S) - is
synergy born at small bases or emergent at
large ones?  pooled spearman(|S|, SM); count
SM > TOL_MOB.
E5 POSITION vs SIZE: GQA-group (h//4) and
quarter (h//8) same/different rates among the
top synergy triples vs all triples; spearman
(m_i, viol_sum as added head); spearman
(DAH34_MED head spectra from the 3074 npz,
m_i) and spearman (DAH34_MED, viol_sum).
verdict: setup fail -> setup_failed_
supermodular; a1..a5 fail -> anchor_mismatch_
3074; a6 fail -> anchor_mismatch_3073_ibook;
b0 fail -> mobius_transform_mismatch; no
positive mu beyond tol -> submodular_strict
(impossible: 3074 measured 26.5 percent
violations); no positive mu at all ->
residual_noise_only; c10 >= 0.5 ->
supermodular_localized_mu<top1 mask>;
else supermodular_diffuse.
memory discipline: no model load; npz fp64
re-analysis; single process; seconds runtime.
"""
import hashlib
import itertools
import json
import os
import time

import numpy as np

PHASE = 3075
NAME = 'omega_p72_supermodular_structure'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3075', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
LOG = os.path.join(OUT, 'run_log.txt')

R74 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3074', 'omega_p71_capacity_law')
NPZ74 = os.path.join(
    R74, 'omega_p71_capacity_law.npz')
RES74 = os.path.join(R74, 'result.json')
RES73 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3073', 'omega_p70_head_interaction',
    'result.json')
GALL_3071 = -0.5717521069904176
TOL_MOB = 0.02
NT = 8
HEAD_NAMES = [20, 7, 1, 14, 26, 0, 2, 24]


def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


t0 = time.time()
os.makedirs(OUT, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)
for fn in (NAME + '.npz', 'execution.json',
           'result.json', 'seal.json'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)

created = time.strftime('%Y-%m-%d %H:%M:%S')
PREREG = {
    'mode': 'no forwards; pure re-analysis of '
            'the frozen 3074 npz (A_S over '
            'MASKS = 1..255, PAIRS4 in '
            'enumeration order) plus the 3073 '
            'result r1/r2; no model load; '
            'smoke mode optional (SMOKE=1: '
            'anchors only, analysis skipped)',
    'question': '3075 A (menu of 3074): WHERE '
                'do the 26.5 percent '
                'submodularity violations '
                'live?  Full Moebius spectrum '
                'of the measured set function '
                'A(S) over all 255 subsets of '
                'the focal top-8: order-wise '
                'positive-mass statistics, '
                'top cooperative cliques, '
                'conditional synergy profiles '
                'SM(i,j|S) over 64 bases per '
                'pair, budget-bucket and '
                'per-head violation '
                'localization, GQA/quarter '
                'position vs effect-size '
                'structure.',
    'E1_anchors': 'deterministic enumeration '
                  'replay (S, x, T) length '
                  '16472; marginal diffs bit '
                  '0.0 vs npz PAIRS4 (a1); '
                  'n_viol=4359 and viol_rate '
                  'bit vs 3074 result (a2); '
                  'A(255)=0.5322981028358011 '
                  '(a3); npz R_ALL bit vs '
                  '3071 gA '
                  '-0.5717521069904176 (a4); '
                  '3073 r1[0] file anchor '
                  '(a5); mu2 pairs bit vs '
                  'R1[j]-R2[p]+R1[i] from '
                  '3073 result with the '
                  'brute-force summation '
                  'order matched (a6; A '
                  'values bit-identical to '
                  '-(R1/R2) via the 3074 a10 '
                  'family); fast Moebius vs '
                  'brute force agree to '
                  '<=1e-12 on all 255 (b0; '
                  'different fp summation '
                  'orders, algorithmic '
                  'equivalence not a same-'
                  'path bit anchor)',
    'E2_spectrum': 'brute-force Mobius mu(S) = '
                   'sum_{T subset S} '
                   '(-1)^(|S|-|T|) A(T), all '
                   '255 masks (3^8 terms); '
                   'order 2..8 n_pos/n_neg at '
                   'TOL_MOB = 0.02 (~24-pair '
                   'median SE, same scale as '
                   '3074 TOL_SUB); top-12 '
                   'positive with head names; '
                   'concentration c10 = '
                   'top-10 positive sum / '
                   'total positive sum',
    'E3_violations': 'per added head x: '
                     'count/sum/max of '
                     'violations (Delta > '
                     'TOL_MOB); per |S| in '
                     '{2,3,4,5,6,7} violation '
                     'rate; per budget bucket '
                     '(x(S) quartiles of the '
                     '255 subsets); worst-10 '
                     'inequalities recorded',
    'E4_profiles': 'SM(i,j|S) = A(S+ij)-A(S+i)-'
                   'A(S+j)+A(S) for 28 pairs x '
                   '64 bases; per-pair '
                   'med/max; count SM > '
                   'TOL_MOB; pooled '
                   'spearman(|S|, SM)',
    'E5_position': 'GQA (h//4) and quarter '
                   '(h//8) same-group rates '
                   'among top-12 positive '
                   'Mobius masks vs all 2nd-'
                   'order masks; spearman(m_i,'
                   ' viol_sum[x]); spearman('
                   'DAH34_MED[TOP8], m_i); '
                   'spearman(DAH34_MED[TOP8], '
                   'viol_sum[x])',
    'gates': 'TOL_MOB = 0.02; localized if '
             'c10 >= 0.5 (top-10 positive '
             'coefficients carry at least '
             'half the positive Moebius '
             'mass); recorded not gating: '
             'bucket rates, spearman, '
             'profiles',
    'verdict': 'setup fail -> setup_failed_'
               'supermodular; a1..a5 fail -> '
               'anchor_mismatch_3074; a6 '
               'fail -> anchor_mismatch_3073_'
               'ibook; b0 fail -> mobius_'
               'transform_mismatch; max mu '
               '(order >= 2) <= TOL_MOB -> '
               'submodular_strict; no mu > '
               'TOL_MOB at all -> residual_'
               'noise_only; c10 >= 0.5 -> '
               'supermodular_localized_mu'
               '<top1 mask>; else '
               'supermodular_diffuse',
    'statistics_discipline': 'same-precision bit '
                             'anchors on frozen '
                             'data; TOL_MOB = '
                             '3074 TOL_SUB noise '
                             'scale; no new '
                             'forwards so no '
                             'protocol drift is '
                             'possible; all '
                             'thresholds frozen '
                             'before the '
                             're-analysis',
}


def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


def spearman(a, b):
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


log('execution.json written (prereg frozen) '
    '%s smoke=%s' % (created, SMOKE))
z = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'prereg': PREREG,
    'env': {'smoke': bool(SMOKE)},
    'inputs': {'npz74': NPZ74,
               'res74': RES74,
               'res73': RES73},
}
with open(os.path.join(OUT, 'execution.json'),
          'w', encoding='utf-8') as f:
    json.dump(z, f, ensure_ascii=False, indent=1)

# ==== load frozen inputs ====
d74 = np.load(NPZ74)
A_S = d74['A_S'].astype(np.float64)
MASKS74 = d74['MASKS'].astype(np.int64)
PAIRS4 = d74['PAIRS4'].astype(np.float64)
R_ALL = float(d74['R_ALL'])
DAH34_74 = d74['DAH34'].astype(np.float64)
TOP8_74 = d74['TOP8'].astype(np.int64)
j74 = json.load(open(RES74, encoding='utf-8'))
st74 = j74['stats']
j73 = json.load(open(RES73, encoding='utf-8'))
st73 = j73['stats']
R1_73 = np.array(st73['r1'], dtype=np.float64)
R2_73 = np.array(st73['r2'], dtype=np.float64)
log('loaded 3074 npz: A_S %s PAIRS4 %s '
    '| 3073 r1/r2 (%d/%d)'
    % (A_S.shape, PAIRS4.shape,
       len(R1_73), len(R2_73)))

AMP = {int(m): float(A_S[i])
       for i, m in enumerate(MASKS74)}
AMP[0] = 0.0


def A(m):
    return AMP.get(int(m))


assert set(MASKS74.tolist()) == set(
    range(1, 256)), 'MASKS not 1..255'
TOP8 = [int(v) for v in TOP8_74]
assert TOP8 == [20, 7, 1, 14, 26, 0, 2, 24]

# ==== E1 anchor replay ====
log('E1 enumeration replay (a1/a2)')


def popcount(m):
    return bin(m).count('1')


idx_seq = []
replay = []
for S in range(1, 256):
    for x in range(NT):
        if S >> x & 1:
            continue
        base_m = A(S | (1 << x)) - A(S)
        TT_ = S
        while True:
            TT_ = (TT_ - 1) & S
            if TT_ < 0:
                break
            if TT_ == S:
                continue
            mt = A(TT_ | (1 << x)) - A(TT_)
            idx_seq.append((S, x, TT_))
            replay.append(base_m - mt)
            if TT_ == 0:
                break
replay = np.array(replay, dtype=np.float64)
a1_diff = None
a1_ok = False
if not SMOKE:
    a1_diff = float(np.max(np.abs(
        replay - PAIRS4)))
    a1_ok = bool(a1_diff == 0.0
                 and len(replay) == len(PAIRS4))
log('a1 replay vs PAIRS4: n=%d/%d '
    'max|d|=%s ok=%s'
    % (len(replay), len(PAIRS4),
       a1_diff, a1_ok))

viol_mask = replay > TOL_MOB
n_viol_re = int(viol_mask.sum())
viol_rate_re = float(n_viol_re) / len(replay)
a2_ok = False
if not SMOKE:
    a2_ok = bool(
        n_viol_re == int(st74['n_viol'])
        and viol_rate_re
        == float(st74['viol_rate'])
        and int(st74['n_sub_checked']) == 16472)
log('a2 n_viol=%d rate=%.17g vs 3074 result '
    'ok=%s' % (n_viol_re, viol_rate_re, a2_ok))

a3_ok = bool(A(255)
             == 0.5322981028358011)
log('a3 A(255)=%.17g ok=%s'
    % (A(255), a3_ok))
a4_ok = bool(R_ALL == GALL_3071)
log('a4 npz R_ALL=%.17g vs 3071 gA ok=%s'
    % (R_ALL, a4_ok))
a5_ok = bool(len(R1_73) == 8
             and R1_73[0]
             == -0.22156993894701427)
log('a5 3073 r1[0] file anchor ok=%s' % a5_ok)

# ==== E2 Moebius spectrum ====
log('E2 Moebius spectrum (brute force)')
Aarr = np.zeros(256, dtype=np.float64)
for m in range(256):
    Aarr[m] = A(m)

MU = np.zeros(256, dtype=np.float64)
for m in range(1, 256):
    s = 0.0
    t = m
    while True:
        sign = 1.0 if (popcount(m)
                       - popcount(t)) % 2 == 0 \
            else -1.0
        s += sign * Aarr[t]
        if t == 0:
            break
        t = (t - 1) & m
    MU[m] = s

# b0: fast Moebius transform vs brute
# force.  Different fp summation orders,
# so agreement is asserted at 1e-12
# (algorithmic equivalence), not bit.
FAST = Aarr.copy()
for i in range(NT):
    for m in range(256):
        if m >> i & 1:
            FAST[m] -= FAST[m ^ (1 << i)]
b0_diff = float(np.max(np.abs(
    FAST[1:] - MU[1:])))
b0_ok = bool(b0_diff <= 1e-12)
log('b0 fast-vs-brute Moebius max|d|='
    '%.3e ok=%s' % (b0_diff, b0_ok))

# a6: mu2 pairs vs 3073 bookkeeping.
# The brute-force mu2 summation order for
# mask m=(1<<i)|(1<<j), i<j, is
# A[m] - A[1<<j] - A[1<<i] (lowest bit i
# removed first); A values are bit-identical
# to -(R1_73/R2_73) via the 3074 a10 family
# anchor, so the reference below reorders the
# sum identically -> same-path bit anchor.
mu2_re = np.array(
    [MU[(1 << i) | (1 << j)]
     for i, j in itertools.combinations(
         range(NT), 2)],
    dtype=np.float64)
mu2_ref = np.array(
    [(R1_73[j] - R2_73[p]) + R1_73[i]
     for p, (i, j) in enumerate(
         itertools.combinations(range(NT), 2))],
    dtype=np.float64)
a6_diff = None
a6_ok = False
if not SMOKE:
    a6_diff = float(np.max(np.abs(
        mu2_re - mu2_ref)))
    a6_ok = bool(a6_diff == 0.0)
log('a6 mu2 vs -(3073 I_BOOK) chain: '
    'max|d|=%s ok=%s' % (a6_diff, a6_ok))

orders = np.array([popcount(m)
                   for m in range(256)])
spec = {}
for k in range(2, NT + 1):
    sel = MU[1:][orders[1:] == k]
    spec[k] = {
        'n': int(len(sel)),
        'n_pos': int((sel > TOL_MOB).sum()),
        'n_neg': int((sel < -TOL_MOB).sum()),
        'max': float(sel.max()),
        'min': float(sel.min()),
        'sum': float(sel.sum())}
    log('E2 mu order %d: n=%d n_pos=%d '
        'n_neg=%d max=%+.4f min=%+.4f'
        % (k, spec[k]['n'], spec[k]['n_pos'],
           spec[k]['n_neg'], spec[k]['max'],
           spec[k]['min']))

masks_hi = [m for m in range(1, 256)
            if popcount(m) >= 2]
pos_pairs = [(float(MU[m]), m)
             for m in masks_hi if MU[m] > TOL_MOB]
pos_pairs.sort(reverse=True)
pos_vals = np.array(
    [v for v, _ in pos_pairs],
    dtype=np.float64) \
    if pos_pairs else np.zeros(0)
if len(pos_vals):
    c10 = float(
        pos_vals[:10].sum() / pos_vals.sum())
else:
    c10 = float('nan')
max_mu_hi = float(max(
    (MU[m] for m in masks_hi), default=0.0))
log('E2 positive Mobius mass (|S|>=2, '
    '>TOL): n=%d total=%.4f c10=%s '
    'max=%+.4f'
    % (len(pos_vals),
       float(pos_vals.sum())
       if len(pos_vals) else 0.0,
       c10, max_mu_hi))
top12 = pos_pairs[:12]
for v, m in top12:
    heads = [HEAD_NAMES[i]
             for i in range(NT) if m >> i & 1]
    log('E2 top mu mask=%d order=%d '
        'mu=%+.4f heads=%s'
        % (m, popcount(m), v, heads))

# ==== E3 violation localization ====
log('E3 violation localization')
viol_x = np.zeros(NT, dtype=np.int64)
viol_sum_x = np.zeros(NT, dtype=np.float64)
viol_max_x = np.zeros(NT, dtype=np.float64)
viol_S = []
if not SMOKE:
    for (S, x, T), dv in zip(idx_seq,
                             replay):
        if dv > TOL_MOB:
            viol_x[x] += 1
            viol_sum_x[x] += dv
            viol_max_x[x] = max(
                viol_max_x[x], dv)
            viol_S.append((float(dv), S, x, T))
m_i = np.abs(R1_73)
log('E3 per added head x (head, count, '
    'sum, max, m_i):')
for x in range(NT):
    log('E3 x=h%02d: n=%d sum=%.4f '
        'max=%.4f m=%.4f'
        % (HEAD_NAMES[x], viol_x[x],
           viol_sum_x[x], viol_max_x[x],
           m_i[x]))
sp_m_viol = float(spearman(m_i, viol_sum_x)) \
    if not SMOKE else float('nan')
log('E3 spearman(m_i, viol_sum_x)=%.4f'
    % sp_m_viol)

rate_by_size = {}
if not SMOKE:
    sizes_arr = np.array(
        [popcount(S) for S, _, _ in idx_seq])
    for sz in range(2, NT):
        sel = sizes_arr == sz
        n_sz = int(sel.sum())
        v_sz = int((replay[sel]
                    > TOL_MOB).sum())
        rate_by_size[sz] = {
            'n': n_sz, 'viol': v_sz,
            'rate': v_sz / n_sz}
        log('E3 |S|=%d: n=%d viol=%d '
            'rate=%.3f'
            % (sz, n_sz, v_sz, v_sz / n_sz))

budget = np.zeros(256, dtype=np.float64)
for m in range(256):
    s = 0.0
    for ti in range(NT):
        if m >> ti & 1:
            s += m_i[ti]
    budget[m] = s
rate_by_budget = {}
if not SMOKE:
    buds = np.array(
        [budget[S] for S, _, _ in idx_seq])
    qs = np.quantile(
        [budget[m] for m in range(1, 256)],
        [0.25, 0.5, 0.75])
    edges = [0.0, qs[0], qs[1], qs[2],
             float(budget[255]) + 1e-9]
    for bi in range(4):
        sel = (buds >= edges[bi]) \
            & (buds < edges[bi + 1])
        n_b = int(sel.sum())
        v_b = int((replay[sel]
                   > TOL_MOB).sum())
        rate_by_budget[bi] = {
            'lo': float(edges[bi]),
            'hi': float(edges[bi + 1]),
            'n': n_b, 'viol': v_b,
            'rate': (v_b / n_b)
            if n_b else float('nan')}
        log('E3 budget[%d] %.3f-%.3f: n=%d '
            'viol=%d rate=%.3f'
            % (bi, edges[bi], edges[bi + 1],
               n_b, v_b, rate_by_budget[bi]
               ['rate']))

worst = sorted(viol_S, reverse=True)[:10] \
    if not SMOKE else []
for dv, S, x, T in worst:
    hs = [HEAD_NAMES[i] for i in range(NT)
          if S >> i & 1]
    ht = [HEAD_NAMES[i] for i in range(NT)
          if T >> i & 1]
    log('E3 worst: Delta=%+.4f S=%s x=h%02d '
        'T=%s' % (dv, hs, HEAD_NAMES[x], ht))

# ==== E4 conditional synergy profiles ====
log('E4 conditional synergy SM(i,j|S)')
PAIRS = list(itertools.combinations(
    range(NT), 2))
SM_prof = np.full((len(PAIRS), 64), np.nan)
sm_cnt = 0
sm_med = np.zeros(len(PAIRS))
sm_max = np.zeros(len(PAIRS))
if not SMOKE:
    for p_i, (i, j) in enumerate(PAIRS):
        others = [k for k in range(NT)
                  if k != i and k != j]
        c = 0
        vals = []
        for r in range(64):
            S = 0
            for q, k in enumerate(others):
                if r >> q & 1:
                    S |= 1 << k
            v = (A(S | (1 << i) | (1 << j))
                 - A(S | (1 << i))
                 - A(S | (1 << j)) + A(S))
            SM_prof[p_i, c] = v
            vals.append(v)
            if v > TOL_MOB:
                sm_cnt += 1
            c += 1
        sm_med[p_i] = float(np.median(vals))
        sm_max[p_i] = float(np.max(vals))
    sizes_prof = np.array(
        [popcount(S) for S in range(64)])
    flat_sm = SM_prof.reshape(-1)
    flat_sz = np.tile(sizes_prof, len(PAIRS))
    sp_sz_sm = float(spearman(flat_sz,
                              flat_sm))
    log('E4 SM>TOL count=%d/1792; pooled '
        'spearman(|S|, SM)=%.4f'
        % (sm_cnt, sp_sz_sm))
    for p_i, (i, j) in enumerate(PAIRS):
        if sm_max[p_i] > TOL_MOB or \
                sm_med[p_i] > 0:
            log('E4 pair h%02d+h%02d: '
                'med=%+.4f max=%+.4f'
                % (HEAD_NAMES[i],
                   HEAD_NAMES[j],
                   sm_med[p_i], sm_max[p_i]))
else:
    sp_sz_sm = float('nan')

# ==== E5 position vs size ====
log('E5 position vs size')
gqa = [h // 4 for h in HEAD_NAMES]
qtr = [h // 8 for h in HEAD_NAMES]
if not SMOKE:
    # weighted pair census inside positive
    # Mobius masks (any order >= 2) vs the
    # 28-pair baseline
    pair_pos = {}
    for v, m in pos_pairs:
        tis = [ti for ti in range(NT)
               if m >> ti & 1]
        for i, j in itertools.combinations(
                tis, 2):
            pair_pos[(i, j)] = \
                pair_pos.get((i, j), 0) + 1
    top_pairs = sorted(
        pair_pos.items(),
        key=lambda kv: (-kv[1], kv[0]))[:12]
    w_tot = sum(pair_pos.values())
    w_gqa = sum(c for (i, j), c
                in pair_pos.items()
                if gqa[i] == gqa[j])
    w_qtr = sum(c for (i, j), c
                in pair_pos.items()
                if qtr[i] == qtr[j])
    base_gqa = sum(
        1 for i, j in PAIRS
        if gqa[i] == gqa[j])
    base_qtr = sum(
        1 for i, j in PAIRS
        if qtr[i] == qtr[j])
    log('E5 weighted pair census inside '
        'positive mu masks: total=%d GQA-'
        'same=%.3f (baseline %.3f = %d/28) '
        'quarter-same=%.3f (baseline %.3f '
        '= %d/28)'
        % (w_tot,
           w_gqa / w_tot if w_tot else
           float('nan'),
           base_gqa / 28.0, base_gqa,
           w_qtr / w_tot if w_tot else
           float('nan'),
           base_qtr / 28.0, base_qtr))
    for (i, j), c in top_pairs:
        log('E5 top pair h%02d+h%02d '
            '(gqa %d/%d, qtr %d/%d): '
            'weight=%d'
            % (HEAD_NAMES[i], HEAD_NAMES[j],
               gqa[i], gqa[j],
               qtr[i], qtr[j], c))
else:
    w_tot = w_gqa = w_qtr = 0
    base_gqa = base_qtr = 0
sp_dah_m = float('nan')
sp_dah_v = float('nan')
if not SMOKE:
    dah_top = np.abs(np.median(
        DAH34_74, axis=0))[TOP8]
    sp_dah_m = float(spearman(dah_top, m_i))
    sp_dah_v = float(spearman(dah_top,
                              viol_sum_x))
    log('E5 spearman(|DAH34_MED|, m_i)=%.4f; '
        'spearman(|DAH34_MED|, viol_sum)='
        '%.4f' % (sp_dah_m, sp_dah_v))

# ==== verdict ====
verdict = 'smoke_pending'
if not SMOKE:
    setup_ok = bool(a1_ok and a2_ok and a3_ok
                    and a4_ok and a5_ok)
    if not setup_ok:
        verdict = 'setup_failed_supermodular'
    elif not a6_ok:
        verdict = 'anchor_mismatch_3073_ibook'
    elif not b0_ok:
        verdict = 'mobius_transform_mismatch'
    elif max_mu_hi <= TOL_MOB:
        verdict = 'submodular_strict'
    elif len(pos_vals) == 0:
        verdict = 'residual_noise_only'
    elif c10 >= 0.5:
        m_top = max(masks_hi,
                    key=lambda m: MU[m])
        verdict = ('supermodular_localized_mu'
                   '%d' % m_top)
    else:
        verdict = 'supermodular_diffuse'
    log('setup_ok=%s a6_ok=%s b0_ok=%s '
        'max_mu_hi=%.4f n_pos=%d c10=%.4f'
        % (setup_ok, a6_ok, b0_ok,
           max_mu_hi, len(pos_vals), c10))

# ==== npz ====
save = {
    'MU': MU,
    'A_S': A_S,
    'MASKS': MASKS74,
    'PAIRS4': PAIRS4,
    'TOP8': np.array(TOP8, dtype=np.int64),
    'SM_PROF': SM_prof,
    'PAIRS_IDX': np.array(PAIRS,
                          dtype=np.int64),
    'A1_DIFF': np.float64(
        a1_diff if a1_diff is not None
        else np.nan),
    'A1_OK': np.bool_(a1_ok),
    'A2_OK': np.bool_(a2_ok),
    'A3_OK': np.bool_(a3_ok),
    'A4_OK': np.bool_(a4_ok),
    'A5_OK': np.bool_(a5_ok),
    'A6_DIFF': np.float64(
        a6_diff if a6_diff is not None
        else np.nan),
    'A6_OK': np.bool_(a6_ok),
    'B0_DIFF': np.float64(b0_diff),
    'B0_OK': np.bool_(b0_ok),
    'N_VIOL_RE': np.int64(n_viol_re),
    'VIOL_RATE_RE': np.float64(viol_rate_re),
    'C10': np.float64(c10
                      if np.isfinite(c10)
                      else np.nan),
    'MAX_MU_HI': np.float64(max_mu_hi),
    'R_ALL': np.float64(R_ALL),
}
if not SMOKE:
    save['VIOL_SUM_X'] = viol_sum_x
    save['VIOL_MAX_X'] = viol_max_x
    save['M_I'] = m_i
np.savez(os.path.join(OUT, NAME + '.npz'),
         **save)

# ==== result.json ====
res = {
    'phase': PHASE,
    'name': NAME,
    'created': created,
    'verdict': verdict,
    'forwards': 0,
    'elapsed': float(time.time() - t0),
    'anchors': {
        'a1_diff': a1_diff,
        'a1_ok': a1_ok,
        'a2_ok': a2_ok,
        'a3_ok': a3_ok,
        'a4_ok': a4_ok,
        'a5_ok': a5_ok,
        'a6_diff': a6_diff,
        'a6_ok': a6_ok,
        'b0_diff': b0_diff,
        'b0_ok': b0_ok,
        'setup_ok': bool(a1_ok and a2_ok
                         and a3_ok and a4_ok
                         and a5_ok),
    },
    'stats': {
        'n_viol_re': n_viol_re,
        'viol_rate_re': viol_rate_re,
        'max_mu_hi': max_mu_hi,
        'n_pos_mu': int(len(pos_vals)),
        'pos_sum': float(pos_vals.sum())
        if len(pos_vals) else 0.0,
        'c10': c10,
        'spectrum': {
            str(k): v
            for k, v in spec.items()},
        'top12_pos_mu': [
            {'mask': int(m), 'order':
             popcount(m), 'mu': v,
             'heads': [HEAD_NAMES[i]
                       for i in range(NT)
                       if m >> i & 1]}
            for v, m in top12],
        'viol_sum_x': viol_sum_x.tolist()
        if not SMOKE else None,
        'viol_max_x': viol_max_x.tolist()
        if not SMOKE else None,
        'sp_m_viol': sp_m_viol,
        'rate_by_size': {
            str(k): v for k, v
            in rate_by_size.items()},
        'rate_by_budget': {
            str(k): v for k, v
            in rate_by_budget.items()},
        'worst10': [
            {'delta': dv, 'S_mask': int(S),
             'x': int(x), 'T_mask': int(T)}
            for dv, S, x, T in worst],
        'sm_count': sm_cnt,
        'sp_size_sm': sp_sz_sm,
        'sm_med': sm_med.tolist()
        if not SMOKE else None,
        'sm_max': sm_max.tolist()
        if not SMOKE else None,
        'sp_dah_m': sp_dah_m,
        'sp_dah_v': sp_dah_v,
        'mu2_min': float(mu2_re.min()),
        'mu2_max': float(mu2_re.max()),
        'w_pair_total': w_tot,
        'w_gqa': w_gqa,
        'w_qtr': w_qtr,
        'base_gqa': base_gqa,
        'base_qtr': base_qtr,
        'top_pairs': [
            {'i': int(i), 'j': int(j),
             'weight': int(c),
             'heads': [HEAD_NAMES[i],
                       HEAD_NAMES[j]]}
            for (i, j), c in top_pairs]
            if not SMOKE else [],
    },
}
with open(os.path.join(OUT, 'result.json'),
          'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False,
              indent=1)

# ==== seal ====
seal = {
    'verdict': verdict,
    'setup_ok': res['anchors']['setup_ok'],
    'npz_sha256_8': sha8(os.path.join(
        OUT, NAME + '.npz')),
    'result_sha256_8': sha8(os.path.join(
        OUT, 'result.json')),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(
        os.path.abspath(__file__)),
    'created': created,
}
with open(os.path.join(OUT, 'seal.json'),
          'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'],
       res['elapsed']))
log('VERDICT: %s' % verdict)
log('sealed')
print('RUN_COMPLETE %s' % verdict)
