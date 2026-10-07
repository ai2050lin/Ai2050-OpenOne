# -*- coding: utf-8 -*-
"""Phase 3076: Omega-P73 cross-prompt family
stability of the focal-head capacity law
(qwen3-4b single model bf16).

Question (3076 A, menu of 3075): 3074 measured
the saturation curve R(S) over ALL 255 subsets
of the focal top-8 on ONE prompt family and
found a Hill capacity law (fit on |S|<=2
extrapolates the 8-head joint and the full
32-head swap within 0.05); 3075 showed the
Moebius spectrum has a dual-zone structure
(pair-level Mobius coefficients all <= 0 =
competition zone at small bases; positive
high-order mass at orders 4-8 = completion
zone at large bases).  IS THIS STRUCTURE
PROMPT-INDEPENDENT?  Three prompt families,
full protocol each, one process, sequential:
  A everyday-causal = the exact 3074 texts
    RERUN as the cross-run bit-anchor family
    (a1 COS_LAD_34 vs 3066 npz; a8 ZH34/ZH35
    vs 3071 npz; a9 DAH medians vs 3071 npz;
    a10s all-32 singles vs 3071 r34; a10p
    pairs/triplet/top8 vs 3073 r2/r_t3/r_u8;
    a11 full-32 swap vs 3071 gA; a12 A_S vs
    3074 npz; a13 PAIRS4 vs 3074 npz; a14 MU
    vs 3075 npz; a15 top8 == 3071 list; a16
    Hill p_le2 vs 3074 result);
  B science-causal (new texts, same 8
    connectives);
  C social-emotional-causal (new texts).
Per family: banks + b0/b1/b3/b5/b6 recapture
anchors + E1 inj@34 ladder (repV 3065-identical,
24 pairs, med_c_34 reference) + E2H per-head
observation decomposition + 32-head single-swap
scan (per-family r1_all32) + per-family focal
top8 by the 3071 criterion (argsort(r1)
ascending, topk = min(8, n_neg)) + ALL 255
non-empty subsets swapped jointly (24 pairs
each) + full 32-head swap + all 16472
submodularity inequalities (tol 0.02) +
capacity-law fits (fit1 all 255 descriptive;
fit2 |S|<=2 only, extrapolate S8 and all-32,
gates 0.05) + brute-force Moebius spectrum
(order stats 2..8, c10 concentration, top
cliques) + fast-vs-brute Moebius check.
E8 cross-family: top8 overlaps/Jaccard,
spearman(r1_all32), spearman(|DAH34_MED|),
Hill parameter table, spectrum shape, viol
rates, capture8, med_tt_norm scales.
structure_stable(family) = hill passes BOTH
gates AND mu2 n_pos == 0 AND max mu(order>=4)
> TOL_MOB (the 3074/3075 dual-zone signature).
verdict: setup fail -> setup_failed_cross_
prompt; a-anchor fail -> anchor_mismatch_<ref>;
b8 fail -> block_output_mismatch; per-family
top8 selection degenerate -> top8_selection_
degenerate; Moebius transform mismatch ->
mobius_transform_mismatch; else n_stable of 3:
3/3 and min pairwise top8 overlap >= 5 ->
cross_prompt_stable; 3/3 else ->
cross_prompt_stable_head_shift; 2/3 ->
cross_prompt_partial; else cross_prompt_
unstable.
limitations (recorded): absolute 0.05 gates
are scale-dependent (3074 gate ~ 8.7 percent
of the full recovery; relative errors
recorded); causal-connective paradigm only,
families share the syntactic frame so the
comparison isolates domain/content not syntax;
single model qwen3-4b.
memory discipline: single model; families
sequential, big banks (VB/PB/BH/...) fp64 CPU
freed after each family; Wo34/Wo35 fp32 in
E2H then deleted; W32 fp32 resident; del +
gc + empty_cache at family end and run end.
"""
import gc
import hashlib
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

PHASE = 3076
NAME = 'omega_p73_cross_prompt_family'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3076', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
LOG = os.path.join(OUT, 'run_log.txt')

MDIR = os.path.join(ROOT, 'models', 'hf',
                    'qwen3-4b')
NL, HID, KV_HEAD, NQ = 36, 2560, 8, 32
HDIM = 128
KVW = KV_HEAD * HDIM
NQW = NQ * HDIM
FRONT = 4
SEED_MAIN = 3020
L34 = 34
L35 = 35
NH = NQ
NG = KV_HEAD
MEDC34_REF = 0.1487826048372403
PA34_REF = 0.29685845971107483
PF34_REF = 0.2371114194393158
GALL_3071 = -0.5717521069904176
RDIR = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913')
NPZ71 = os.path.join(
    RDIR, 'phase3071', 'omega_p68_attn_head_decomp',
    'omega_p68_attn_head_decomp.npz')
RES71 = os.path.join(
    RDIR, 'phase3071', 'omega_p68_attn_head_decomp',
    'result.json')
RES73 = os.path.join(
    RDIR, 'phase3073', 'omega_p70_head_interaction',
    'result.json')
R74 = os.path.join(RDIR, 'phase3074',
                   'omega_p71_capacity_law')
NPZ74 = os.path.join(R74,
                     'omega_p71_capacity_law.npz')
RES74 = os.path.join(R74, 'result.json')
NPZ75 = os.path.join(
    RDIR, 'phase3075',
    'omega_p72_supermodular_structure',
    'omega_p72_supermodular_structure.npz')
NPZ66 = os.path.join(
    RDIR, 'phase3066',
    'omega_p63_last_layer_flip_anatomy',
    'omega_p63_last_layer_flip_anatomy.npz')
GATE_PRED = 0.05
TOL_SUB = 0.02
TOL_MOB = 0.02
FKEYS = ('A', 'B', 'C')
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
FAM = {
    'A': {
        'domain': 'everyday-causal (3074 texts '
                  'rerun, bit-anchor family)',
        'bodies': (
            'The weather was cold, so',
            'He studied every night because',
            'The experiment failed, therefore',
            'He missed the train, however',
            'The garden grows quickly while',
            'The price was high, yet',
            'She speaks French, although',
            'The road was closed, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the weather,'),
    },
    'B': {
        'domain': 'science-causal',
        'bodies': (
            'The solution turned acidic, so',
            'The sample was heated because',
            'The catalyst degraded, therefore',
            'The vacuum leaked, however',
            'The crystals formed while',
            'The pressure dropped, yet',
            'The alloy expanded, although',
            'The circuit overheated, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the experiment,'),
    },
    'C': {
        'domain': 'social-emotional-causal',
        'bodies': (
            'She felt deeply betrayed, so',
            'He apologized to her because',
            'They reconciled after the '
            'quarrel, therefore',
            'She stormed out of the room, '
            'however',
            'He listened quietly to every '
            'word while',
            'The gift was cheap and hasty, '
            'yet',
            'She forgave him in the end, '
            'although',
            'The friendship ended without '
            'warning, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the '
                     'conversation,'),
    },
}

PREREG = {
    'mode': 'single model qwen3-4b bf16 eager '
            'seed 3020; three prompt families '
            'processed sequentially in one '
            'process (family order A, B, C '
            'frozen; big banks freed between '
            'families); per-family protocol '
            '3074-identical (repV 3065-identical '
            'E1 ladder; 32-head single swap '
            'scan; ALL 255 non-empty subsets of '
            'the per-family focal top-8 swapped '
            'jointly at L34, 24 pairs each; '
            'full 32-head swap; lens probes '
            'fp32 W32 resident; verdict scalars '
            'fp64 cosv); smoke mode optional '
            '(SMOKE=1: 12 masks, 4 pairs, 8 '
            'ladder pairs, a-anchors off, '
            'b-anchors on)',
    'question': '3076 A (menu of 3075): is the '
                'focal-head capacity law + '
                'dual-zone Moebius structure of '
                '3074/3075 prompt-independent?  '
                'Full per-family protocol on '
                'three prompt families (A = 3074 '
                'texts rerun as bit-anchor; B = '
                'science-causal; C = social-'
                'emotional-causal): per-family '
                'focal top-8, 255-subset '
                'saturation sweep, capacity-law '
                'fit2 extrapolation gates, '
                'Moebius spectrum, submodularity '
                'violations; cross-family top8 '
                'overlap, r1/DAH spearman, Hill '
                'parameters, spectrum shape.',
    'families': {
        fk: {'domain': FAM[fk]['domain'],
             'bodies': list(FAM[fk]['bodies']),
             'targets': list(TARGETS),
             'prefixes': list(FAM[fk]
                              ['prefixes'])}
        for fk in FKEYS},
    'top8_criterion': 'per family (3071-'
                      'identical): order = '
                      'np.argsort(r1_all32) '
                      'ASCENDING (most negative '
                      'first), topk = min(8, '
                      'n_neg), top8 = '
                      'order[:topk]; recorded '
                      'per family: n_neg, '
                      'capture8 = '
                      '|sum(r1[top8])| / '
                      '|sum(r1[r1<0])|; '
                      'top8_sel_ok = (topk == 8 '
                      'and all r1[top8] < 0); '
                      'family A must reproduce '
                      '3071 top8 '
                      '[20,7,1,14,26,0,2,24] '
                      '(a15).  NOTE: the '
                      'criterion is signed, '
                      'not |r1| - 3076 probe '
                      'confirmed argsort(-|r1|) '
                      'would give a different '
                      'set (h12 vs h0/h24).  '
                      'If n_neg < 8 the family '
                      'is DEGENERATE: the '
                      'sweep/fits/spectrum run '
                      'over the reduced set '
                      '(empty at NT=0) and '
                      'top8_sel_ok=False forces '
                      'verdict top8_selection_'
                      'degenerate (frozen '
                      'before the run; smoke '
                      'showed B/C can be '
                      'positive-r1 dominated '
                      'at 4 pairs - the '
                      'authoritative 24-pair '
                      'measurement decides).',
    'E1_ladder': 'injection layer 34 x 24 pairs '
                 'per family, repV 3065-'
                 'identical; COS_LAD_34; '
                 'PA34/PF34 lens probes; zH/'
                 'zA captured at L34/L35 per '
                 'pair; family A: a1 bit vs '
                 '3066 npz row 34, med_c_34 '
                 'reference assert '
                 '0.1487826048372403, aref '
                 'PA34/PF34 bit vs 3069 '
                 'constants',
    'E2H_perhead': 'dzH 32 slices of 128 per '
                   'family; dAh TT projection '
                   'medians; family A: a8 ZH34/'
                   'ZH35 bit vs 3071 npz, a9 '
                   'DAH34_MED/DAH35_MED bit vs '
                   '3071 npz',
    'E3_scan': '32-head single swap scan per '
               'family (K3 pairs each) -> '
               'r1_all32 = median cos - '
               'med_c_34; family A: a10s '
               'r1_all32 bit vs 3071 r34 (all '
               '32, stronger than the 3074 '
               '8-single anchor)',
    'E4_sweep': 'ALL 255 non-empty subsets of '
                'the per-family top8 (bitmask, '
                'bit i = top8 position i), 24 '
                'pairs each; R(S) = median cos '
                '- med_c_34; A(S) = -R(S); '
                'family A: a10p pairs/triplet/'
                'top8 bit vs 3073 r2/r_t3/r_u8, '
                'a12 full A_S bit vs 3074 npz, '
                'a11 full 32-head swap R_ALL '
                'bit vs 3071 gA '
                '-0.5717521069904176',
    'E5_submodularity': 'amplitude A = -R; all '
                        '16472 inequalities '
                        'A(S+x)-A(S) <= A(T+x)-'
                        'A(T) for T proper '
                        'subset of S (empty '
                        'included), x not in S; '
                        'tol 0.02 (~24-pair '
                        'median SE); family A: '
                        'a13 PAIRS4 bit vs 3074 '
                        'npz; viol rate '
                        'recorded per family',
    'E6_fits': 'budget x(S) = sum of single-'
               'head amplitudes m_i = |r1_i| '
               'over the per-family top8; '
               'candidates add/exp/log/hill; '
               'grid+refinement least squares; '
               'fit1 = all 255 points '
               '(descriptive); fit2 = |S|<=2 '
               'only (36 points) -> extrapolate '
               'S8 (measured A(255)) and '
               'all-32 (measured -R_ALL); '
               'gates BOTH |pred - meas| < '
               '0.05; family A: a16 Hill p_le2 '
               'bit vs 3074 result '
               '[0.7142857142857143, '
               '0.45409444455802084, '
               '1.1857142857142857]; relative '
               'errors recorded (scale caveat)',
    'E7_spectrum': 'per family brute-force '
                   'Moebius mu(S) = sum_(T '
                   'subset S) (-1)^(|S|-|T|) '
                   'A(T), all 255 masks (3^8 '
                   'terms); fast transform '
                   'cross-check <= 1e-12 (b0m, '
                   'algorithmic equivalence); '
                   'order-wise n_pos/n_neg at '
                   'TOL_MOB = 0.02, max/min; '
                   'top-12 positive with head '
                   'names; c10 concentration; '
                   'family A: a14 MU bit vs '
                   '3075 npz',
    'E8_cross': 'top8 pairwise overlaps and '
                'Jaccard; spearman(r1_all32) '
                'and spearman(|DAH34_MED|) '
                'pairwise over 32 heads; Hill '
                'parameter table; spectrum '
                'shape per order; viol rates; '
                'capture8; med_tt_norm scales; '
                'asymptote gap a/(measured '
                'all-32 amplitude)',
    'gates': 'structure_stable(family) = hill '
             'pass_u8 AND pass_all32 AND mu2 '
             'n_pos == 0 AND max mu(order >= 4) '
             '> TOL_MOB; stable verdict needs '
             'all three families stable AND '
             'min pairwise top8 overlap >= 5 '
             '(of 8)',
    'verdict': 'setup fail -> setup_failed_'
               'cross_prompt; a1 fail -> '
               'anchor_mismatch_3066_ladder; '
               'a8 fail -> anchor_mismatch_'
               '3071_zh; a9 fail -> anchor_'
               'mismatch_3071_dah; a10s/a10p '
               'fail -> anchor_mismatch_3073_'
               'sweep; a11 fail -> anchor_'
               'mismatch_3071_allswap; a12 '
               'fail -> anchor_mismatch_3074_'
               'asweep; a13 fail -> anchor_'
               'mismatch_3074_pairs; a14 fail '
               '-> anchor_mismatch_3075_mu; '
               'a15 fail -> anchor_mismatch_'
               '3071_top8; a16 fail -> anchor_'
               'mismatch_3074_hill; b8 fail -> '
               'block_output_mismatch; top8 '
               'selection degenerate -> '
               'top8_selection_degenerate; '
               'Moebius transform mismatch -> '
               'mobius_transform_mismatch; '
               'else n_stable = 3 and min '
               'overlap >= 5 -> cross_prompt_'
               'stable; n_stable = 3 -> '
               'cross_prompt_stable_head_'
               'shift; n_stable = 2 -> cross_'
               'prompt_partial; else cross_'
               'prompt_unstable',
    'statistics_discipline': 'same-precision '
                             'bit anchors on '
                             'frozen seeds; TT '
                             'logit-space '
                             '3065-identical; '
                             'R(S) referenced to '
                             'the per-family '
                             'med_c_34; subset '
                             'order fixed (mask '
                             '1..255, bit i = '
                             'top8 position i); '
                             'fit input = '
                             'single-head '
                             'amplitudes measured '
                             'in the same run; '
                             'family order frozen '
                             'A, B, C; no post-'
                             'hoc model changes',
    'memory_discipline': 'single model; families '
                         'sequential with big '
                         'banks freed between '
                         'families; Wo34/Wo35 '
                         'fp32 resident in E2H '
                         'then deleted; W32 '
                         'resident; del + gc + '
                         'empty_cache at family '
                         'end and run end',
    'limitations': 'absolute 0.05 gates are '
                   'scale-dependent (3074 gate '
                   '~ 8.7 percent of full '
                   'recovery; relative errors '
                   'recorded); causal-connective '
                   'paradigm only, families '
                   'share the syntactic frame '
                   'so the comparison isolates '
                   'domain/content not syntax; '
                   'single model qwen3-4b',
}

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
with open(os.path.join(OUT, 'execution.json'),
          'w', encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False,
              indent=1)

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


log('execution.json written (prereg frozen) %s '
    'smoke=%s' % (created, SMOKE))
torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)


def cosv(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


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


def model_pred(name, x, p):
    if name == 'add':
        return x
    a, b, g = p
    if name == 'exp':
        return a * (1.0 - np.exp(-x / b))
    if name == 'log':
        return a * np.log1p(x / b)
    if name == 'hill':
        return a * x ** g / (b ** g + x ** g)
    raise ValueError(name)


def grid_fit(name, xs, ys):
    if name == 'add':
        return (0.0, 0.0, 0.0), float(
            ((xs - ys) ** 2).sum())
    ag = np.linspace(0.3, 2.0, 35)
    bg = np.geomspace(0.03, 5.0, 45)
    gg = np.linspace(0.3, 4.5, 43)
    X = np.asarray(xs, dtype=np.float64)[
        None, None, None, :]
    Y = np.asarray(ys, dtype=np.float64)[
        None, None, None, :]
    best = None
    for gi, g in enumerate(gg):
        if name == 'exp':
            P = ag[:, None, None, None] \
                * (1.0 - np.exp(-X
                                / bg[None, :,
                                     None, None]))
        elif name == 'log':
            P = ag[:, None, None, None] \
                * np.log1p(X
                           / bg[None, :,
                                None, None])
        else:
            Xg = X ** g
            P = ag[:, None, None, None] * Xg \
                / (bg[None, :, None, None] ** g
                   + Xg)
        sse = ((P - Y) ** 2).sum(-1)
        i = np.unravel_index(
            np.argmin(sse), sse.shape)
        v = float(sse[i])
        if best is None or v < best[1]:
            best = ((float(ag[i[0]]),
                     float(bg[i[1]]),
                     float(gg[gi])), v)
    # refine twice around the best
    p0, v0 = best
    for _ in range(2):
        da = ag[1] - ag[0]
        db = bg[1] / bg[0]
        dg = gg[1] - gg[0] if len(gg) > 1 else 0
        ag2 = np.linspace(max(0.05, p0[0] - da),
                          p0[0] + da, 15)
        bg2 = np.geomspace(max(1e-3, p0[1] / db),
                           p0[1] * db, 20)
        gg2 = np.linspace(
            max(0.1, p0[2] - dg),
            p0[2] + dg, 15) \
            if name == 'hill' else np.array(
                [p0[2]])
        Xg = X ** gg2[0] if name == 'hill' else X
        best2 = None
        for g in gg2:
            if name == 'exp':
                P = ag2[:, None, None, None] \
                    * (1.0 - np.exp(-X
                                    / bg2[None, :,
                                         None, None]))
            elif name == 'log':
                P = ag2[:, None, None, None] \
                    * np.log1p(X
                               / bg2[None, :,
                                    None, None])
            else:
                Xg = X ** g
                P = ag2[:, None, None, None] \
                    * Xg / (bg2[None, :, None,
                                None] ** g + Xg)
            sse = ((P - Y) ** 2).sum(-1)
            i = np.unravel_index(
                np.argmin(sse), sse.shape)
            v = float(sse[i])
            if best2 is None or v < best2[1]:
                best2 = ((float(ag2[i[0]]),
                          float(bg2[i[1]]),
                          float(g)), v)
        if best2[1] < best[1]:
            best = best2
        p0 = best[0]
    return best


# ==== load model ====
tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.num_key_value_heads) \
    == KV_HEAD
assert int(model.config.num_attention_heads) == NQ
assert int(model.config.hidden_size) == HID
INTER = int(model.config.intermediate_size)
NVOC = int(model.config.vocab_size)
Wemb = model.get_output_embeddings().weight
final_norm = model.model.norm
assert int(Wemb.shape[0]) == NVOC
assert int(Wemb.shape[1]) == HID
W32 = Wemb.float()
log('qwen3-4b loaded bf16 (vocab=%d inter=%d) '
    'gpu=%.2f GB (+W32 fp32 %.2f GB)'
    % (NVOC, INTER,
       torch.cuda.memory_allocated() / 1e9,
       W32.numel() * 4 / 1e9))

FW = [0]
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
stateACT = {li: {'repl': None, 'mask': None}
            for li in range(NL)}
stateATN = {li: {'repl': None, 'mask': None}
            for li in range(NL)}
capV = {li: {'rec': False, 'orig': None}
        for li in range(NL)}
capP = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capX = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capA = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capM = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capH = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capZ = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capG = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capU = {li: {'rec': False, 'v': None}
        for li in range(NL)}
capACT = {li: {'rec': False, 'v': None}
          for li in range(NL)}


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


def hook_post(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['v'] = out[0].detach().clone()
        return out
    return h


def hook_last(cp):
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] \
                if isinstance(out, tuple) else out
            cp['v'] = t[0, -1].detach().clone()
        return out
    return h


def hook_in_last(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['v'] = inp[0][0, -1] \
                .detach().clone()
        return out
    return h


def hook_pre_last(st):
    def h(module, args):
        if st['repl'] is None:
            return None
        a = args[0].clone()
        a[0, -1, st['mask']] = st['repl']
        return (a,)
    return h


for li in range(NL):
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
    layers[li].register_forward_hook(hook_post(
        capP[li]))
    layers[li].register_forward_hook(hook_in_last(
        capX[li]))
    layers[li].self_attn \
        .register_forward_hook(hook_last(capA[li]))
    layers[li].mlp \
        .register_forward_hook(hook_last(capM[li]))
    layers[li].self_attn.o_proj \
        .register_forward_hook(hook_in_last(
            capH[li]))
    layers[li].post_attention_layernorm \
        .register_forward_hook(hook_last(
            capZ[li]))
    layers[li].mlp.gate_proj \
        .register_forward_hook(hook_last(
            capG[li]))
    layers[li].mlp.up_proj \
        .register_forward_hook(hook_last(
            capU[li]))
    layers[li].mlp.down_proj \
        .register_forward_hook(hook_in_last(
            capACT[li]))
    layers[li].mlp.down_proj \
        .register_forward_pre_hook(hook_pre_last(
            stateACT[li]))
    layers[li].self_attn.o_proj \
        .register_forward_pre_hook(hook_pre_last(
            stateATN[li]))


def reset_all():
    for li in range(NL):
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capV[li]['rec'] = False
        capV[li]['orig'] = None
        capP[li]['rec'] = False
        capP[li]['v'] = None
        capX[li]['rec'] = False
        capX[li]['v'] = None
        capA[li]['rec'] = False
        capA[li]['v'] = None
        capM[li]['rec'] = False
        capM[li]['v'] = None
        capH[li]['rec'] = False
        capH[li]['v'] = None
        capZ[li]['rec'] = False
        capZ[li]['v'] = None
        capG[li]['rec'] = False
        capG[li]['v'] = None
        capU[li]['rec'] = False
        capU[li]['v'] = None
        capACT[li]['rec'] = False
        capACT[li]['v'] = None
        stateACT[li]['repl'] = None
        stateACT[li]['mask'] = None
        stateATN[li]['repl'] = None
        stateATN[li]['mask'] = None


def forward_gen(ids, repl=None, act_swaps=None,
                attn_swaps=None):
    """3065-identical V replacement; optional
    act swaps [(layer, idx, vals)] at down_proj
    inputs and attn swaps [(layer, idx, vals)]
    at o_proj inputs (last position).  Returns
    bf16 last-position captures zX/zA/zM/zP
    (NL, HID), zH (NL, NQW), zZ (NL, HID),
    zG/zU/zACT (NL, INTER) on GPU."""
    reset_all()
    FW[0] += 1
    m_all = torch.ones(
        len(ids), dtype=torch.bool,
        device='cuda')
    if repl is not None:
        rt = torch.tensor(
            np.ascontiguousarray(repl),
            dtype=torch.bfloat16,
            device='cuda')
        for li in range(NL):
            stateV[li]['repl'] = rt[li]
            stateV[li]['mask'] = m_all
    if act_swaps is not None:
        for li_, midx, mval in act_swaps:
            stateACT[li_]['mask'] = torch.tensor(
                np.ascontiguousarray(midx),
                dtype=torch.long,
                device='cuda')
            stateACT[li_]['repl'] = torch.tensor(
                np.ascontiguousarray(mval),
                dtype=torch.bfloat16,
                device='cuda')
    if attn_swaps is not None:
        for li_, midx, mval in attn_swaps:
            stateATN[li_]['mask'] = torch.tensor(
                np.ascontiguousarray(midx),
                dtype=torch.long,
                device='cuda')
            stateATN[li_]['repl'] = torch.tensor(
                np.ascontiguousarray(mval),
                dtype=torch.bfloat16,
                device='cuda')
    for li in range(NL):
        capV[li]['rec'] = True
        capP[li]['rec'] = True
        capX[li]['rec'] = True
        capA[li]['rec'] = True
        capM[li]['rec'] = True
        capH[li]['rec'] = True
        capZ[li]['rec'] = True
        capG[li]['rec'] = True
        capU[li]['rec'] = True
        capACT[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor(
            [ids], device='cuda'),
            use_cache=False)
    lg = out.logits[0, -1].detach() \
        .double().cpu().numpy()
    lg_gpu = out.logits[0, -1].detach()
    vb = np.stack([capV[li]['orig'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    po = np.stack([capP[li]['v'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    zX = torch.stack([capX[li]['v']
                      for li in range(NL)])
    zA = torch.stack([capA[li]['v']
                      for li in range(NL)])
    zM = torch.stack([capM[li]['v']
                      for li in range(NL)])
    zP = torch.stack([capP[li]['v'][-1]
                      for li in range(NL)])
    zH = torch.stack([capH[li]['v']
                      for li in range(NL)])
    zZ = torch.stack([capZ[li]['v']
                      for li in range(NL)])
    zG = torch.stack([capG[li]['v']
                      for li in range(NL)])
    zU = torch.stack([capU[li]['v']
                      for li in range(NL)])
    zACT = torch.stack([capACT[li]['v']
                        for li in range(NL)])
    reset_all()
    return lg, lg_gpu, vb, po, zX, zA, zM, zP, \
        zH, zZ, zG, zU, zACT


ridx = np.arange(FRONT)
ALLH = np.arange(NQW)
ALLI = np.arange(INTER)
HEAD_IDX = [np.arange(h * HDIM, (h + 1) * HDIM)
            for h in range(NH)]


def head_decomp(dzH, WoT, Wo, dzA, t, tn):
    dhr = torch.tensor(
        np.ascontiguousarray(
            dzH.reshape(NQ, HDIM)),
        device='cuda').float()
    with torch.no_grad():
        outh = torch.einsum(
            'hd,hdj->hj', dhr, WoT)
        un = F.linear(outh, W32)
    out_np = outh.float().cpu().numpy()
    un_np = un.double().cpu().numpy()
    dAh = (un_np @ t) / tn
    tot = F.linear(
        dhr.reshape(1, NQW), Wo)[0]
    tot_np = tot.double().cpu().numpy()
    rel = float(np.linalg.norm(
        tot_np - dzA)) / max(
        float(np.linalg.norm(dzA)), 1e-12)
    return dAh, np.linalg.norm(
        out_np, axis=1), rel


# ==== anchor files (family A) ====
ANCH = {}
if not SMOKE:
    z66 = np.load(NPZ66)
    ANCH['cos66_34'] = \
        z66['COS_LAD'][L34].astype(
            np.float64).copy()
    z71 = np.load(NPZ71)
    ANCH['zh34_71'] = z71['ZH34'].astype(
        np.float64).copy()
    ANCH['zh35_71'] = z71['ZH35'].astype(
        np.float64).copy()
    ANCH['dah34m_71'] = z71['DAH34_MED'].astype(
        np.float64).copy()
    ANCH['dah35m_71'] = z71['DAH35_MED'].astype(
        np.float64).copy()
    j71 = json.load(open(RES71,
                         encoding='utf-8'))
    ANCH['r34_71'] = np.array(
        j71['stats']['head']['r34'],
        dtype=np.float64)
    ANCH['top8_71'] = [int(v) for v
                       in j71['stats']['head']
                       ['top8']]
    j73 = json.load(open(RES73,
                         encoding='utf-8'))
    ANCH['r1_73'] = np.array(
        j73['stats']['r1'], dtype=np.float64)
    ANCH['r2_73'] = np.array(
        j73['stats']['r2'], dtype=np.float64)
    ANCH['r_t3_73'] = float(
        j73['stats']['r_t3'])
    ANCH['r_u8_73'] = float(
        j73['stats']['r_u8'])
    z74 = np.load(NPZ74)
    ANCH['a_s_74'] = z74['A_S'].astype(
        np.float64).copy()
    ANCH['pairs4_74'] = z74['PAIRS4'].astype(
        np.float64).copy()
    z75 = np.load(NPZ75)
    ANCH['mu_75'] = z75['MU'].astype(
        np.float64).copy()
    j74 = json.load(open(RES74,
                         encoding='utf-8'))
    ANCH['hill74_le2'] = list(
        j74['stats']['fits']['hill']['p_le2'])
    log('anchor files loaded: cos66_34 %s '
        'r34_71 %s top8_71 %s a_s_74 %s '
        'pairs4_74 %s mu_75 %s hill74_le2 %s'
        % (ANCH['cos66_34'].shape,
           ANCH['r34_71'].shape,
           ANCH['top8_71'],
           ANCH['a_s_74'].shape,
           ANCH['pairs4_74'].shape,
           ANCH['mu_75'].shape,
           ANCH['hill74_le2']))
else:
    log('anchor files skipped (smoke)')

AKEYS = ('a1', 'aref', 'a8', 'a9', 'a10s',
         'a10p', 'a11', 'a12', 'a13', 'a14',
         'a15', 'a16')


def popcount(m):
    return bin(m).count('1')


def run_family(fkey):
    fam = FAM[fkey]
    bodies = fam['bodies']
    prefixes = fam['prefixes']
    is_anchor = (fkey == 'A')
    log('[%s] ==== family begin (%s) ===='
        % (fkey, fam['domain']))
    adiff = {k: None for k in AKEYS}
    aok = {k: True for k in AKEYS}
    if is_anchor and not SMOKE:
        aok = {k: False for k in AKEYS}

    # ==== assembly (3065/3074-identical) ====
    word_tok = {}
    for w in TARGETS:
        wi = tok(' ' + w,
                 add_special_tokens=False)[
            'input_ids']
        assert len(wi) == 1, (fkey, w, wi)
        word_tok[w] = int(wi[0])
    assembled = []
    for bi in range(len(bodies)):
        for ci in range(len(prefixes)):
            s = (prefixes[ci] + ' '
                 + bodies[bi]) if prefixes[ci] \
                else bodies[bi]
            ids = [int(x) for x in tok(
                s, add_special_tokens=False)[
                'input_ids']]
            t = word_tok[TARGETS[bi]]
            assert ids.count(t) == 1, \
                (fkey, bi, ci)
            assembled.append(
                {'ids': ids, 'cond': ci,
                 'body': bi})
    n_pr = len(assembled)
    assert n_pr == 32
    idx_of = {}
    for i in range(n_pr):
        idx_of[(assembled[i]['cond'],
                assembled[i]['body'])] = i
    for i in range(n_pr):
        ci = assembled[i]['cond']
        if ci == 0:
            assembled[i]['off'] = 0
        else:
            bid = assembled[idx_of[(0,
                assembled[i]['body'])]]['ids']
            pid = assembled[i]['ids']
            off = len(pid) - len(bid)
            assert off > 0, (fkey, i)
            assert list(pid[off + 1:]) \
                == list(bid[1:]), (fkey, i)
            w0b = tok.decode([bid[0]]).strip()
            w0p = tok.decode(
                [pid[off]]).strip()
            assert w0b == w0p, (fkey, i)
            assembled[i]['off'] = off
    LENS = np.array([len(assembled[i]['ids'])
                     for i in range(n_pr)])
    NMAX = int(LENS.max())
    log('[%s] assembled 32 prompts (lens %d-%d)'
        % (fkey, int(LENS.min()), NMAX))

    cidx = []
    bidx = []
    for ci in (1, 2, 3):
        for bi in range(len(bodies)):
            cidx.append(ci)
            bidx.append(bi)
    cidx = np.array(cidx)
    bidx = np.array(bidx)
    NP_ = 24
    NP_USE = 8 if SMOKE else NP_
    K3 = 4 if SMOKE else NP_

    def pair_idx(k):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        pref_i = idx_of[(c, b)]
        off = assembled[pref_i]['off']
        return b, base_i, pref_i, off

    BASE_K = np.array([pair_idx(k)[1]
                       for k in range(NP_)])
    for k in range(NP_):
        b, base_i, pref_i, off = pair_idx(k)
        assert int(LENS[base_i]) >= FRONT, \
            (fkey, k, int(LENS[base_i]))
    log('[%s] base length check ok (all >= '
        'FRONT=%d)' % (fkey, FRONT))

    # ==== banks ====
    LG = np.zeros((n_pr, NVOC))
    VB = np.zeros((n_pr, NL, NMAX, KVW))
    PB = np.zeros((n_pr, NL, NMAX, HID))
    BX = np.zeros((n_pr, NL, HID))
    BA = np.zeros((n_pr, NL, HID))
    BM = np.zeros((n_pr, NL, HID))
    BH = np.zeros((n_pr, NL, NQW))
    BZ35 = np.zeros((n_pr, HID))
    BG35 = np.zeros((n_pr, INTER))
    BU35 = np.zeros((n_pr, INTER))
    BACT34 = np.zeros((n_pr, INTER))
    BACT35 = np.zeros((n_pr, INTER))
    a2_max = 0.0
    for i in range(n_pr):
        lg, lg_gpu, vb, po, zX, zA, zM, zP, \
            zH, zZ, zG, zU, zACT = forward_gen(
                assembled[i]['ids'])
        n = int(LENS[i])
        LG[i] = lg
        VB[i, :, :n, :] = vb
        PB[i, :, :n, :] = po
        BX[i] = zX.double().cpu().numpy()
        BA[i] = zA.double().cpu().numpy()
        BM[i] = zM.double().cpu().numpy()
        BH[i] = zH.double().cpu().numpy()
        BZ35[i] = zZ[NL - 1].double() \
            .cpu().numpy()
        BG35[i] = zG[NL - 1].double() \
            .cpu().numpy()
        BU35[i] = zU[NL - 1].double() \
            .cpu().numpy()
        BACT34[i] = zACT[L34].double() \
            .cpu().numpy()
        BACT35[i] = zACT[L35].double() \
            .cpu().numpy()
        with torch.no_grad():
            lgt32 = F.linear(
                final_norm(zP[NL - 1])
                .float().unsqueeze(0), W32)
        d32 = float(
            (lgt32[0] - lg_gpu.float())
            .abs().max())
        a2_max = max(a2_max, d32)
    a2lens_ok = bool(a2_max <= 0.125)
    log('[%s] banks: LG%s VB%s BH%s BACT34%s '
        '(forwards=%d)'
        % (fkey, LG.shape, VB.shape, BH.shape,
           BACT34.shape, FW[0]))
    log('[%s] a2lens lens(final)=logits: fp32 '
        'max|d|=%.3e (<=0.125 ok)'
        % (fkey, a2_max))

    b0_diff = 0.0
    for si in (0, 9, 17, 31):
        lg, lg_gpu, vb, po, zX, zA, zM, zP, \
            zH, zZ, zG, zU, zACT = forward_gen(
                assembled[si]['ids'])
        n2 = int(LENS[si])
        b0_diff = max(b0_diff, float(np.max(
            np.abs(LG[si] - lg))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(VB[si, :, :n2, :] - vb))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(PB[si, :, :n2, :] - po))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BX[si] - zX.double().cpu()
                   .numpy()))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BA[si] - zA.double().cpu()
                   .numpy()))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BM[si] - zM.double().cpu()
                   .numpy()))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BH[si] - zH.double().cpu()
                   .numpy()))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BZ35[si] - zZ[NL - 1].double()
                   .cpu().numpy()))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BACT34[si]
                   - zACT[L34].double().cpu()
                   .numpy()))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BACT35[si]
                   - zACT[L35].double().cpu()
                   .numpy()))))
    b0_diff = float(b0_diff)
    b0_ok = bool(b0_diff == 0.0)
    log('[%s] b0 recapture diff=%.3e ok=%s'
        % (fkey, b0_diff, b0_ok))

    TT = np.stack([
        LG[idx_of[(int(cidx[k]),
                   int(bidx[k]))]]
        - LG[idx_of[(0, int(bidx[k]))]]
        for k in range(NP_)])
    TTG = torch.tensor(TT, dtype=torch.float32,
                       device='cuda')
    TTn = torch.tensor(
        np.linalg.norm(TT, axis=1),
        dtype=torch.float32, device='cuda')
    med_tn = float(np.median(
        np.linalg.norm(TT, axis=1)))

    base0 = idx_of[(0, int(bidx[0]))]
    n0 = int(LENS[base0])
    selfV = VB[base0, :, :n0, :].copy()
    lg_s = forward_gen(
        assembled[base0]['ids'], repl=selfV)[0]
    b1_diff = float(np.max(np.abs(lg_s
                                  - LG[base0])))
    b1_ok = bool(b1_diff == 0.0)
    log('[%s] b1 sham self-replacement diff='
        '%.3e ok=%s' % (fkey, b1_diff, b1_ok))

    b3_ok = bool(np.isfinite(LG).all()
                 and np.isfinite(VB).all()
                 and np.isfinite(PB).all()
                 and np.isfinite(TT).all()
                 and np.isfinite(BX).all()
                 and np.isfinite(BA).all()
                 and np.isfinite(BM).all()
                 and np.isfinite(BH).all()
                 and np.isfinite(BACT34).all()
                 and np.isfinite(BACT35).all())
    log('[%s] b3 finite=%s' % (fkey, b3_ok))

    b5_diff = 0.0
    for i in range(n_pr):
        g_ = torch.tensor(BG35[i],
                          device='cuda') \
            .to(torch.bfloat16)
        u_ = torch.tensor(BU35[i],
                          device='cuda') \
            .to(torch.bfloat16)
        act_ = torch.tensor(BACT35[i],
                            device='cuda') \
            .to(torch.bfloat16)
        rec = F.silu(g_) * u_
        b5_diff = max(b5_diff, float(
            (rec - act_).abs().max()))
    b5_diff = float(b5_diff)
    log('[%s] b5 act=silu(g)*u bf16 '
        'reconstruction diff=%.3e (recorded)'
        % (fkey, b5_diff))

    b6_diff = 0.0
    for i in range(n_pr):
        n = int(LENS[i])
        x_ = torch.tensor(BX[i],
                          device='cuda') \
            .to(torch.bfloat16)
        a_ = torch.tensor(BA[i],
                          device='cuda') \
            .to(torch.bfloat16)
        m_ = torch.tensor(BM[i],
                          device='cuda') \
            .to(torch.bfloat16)
        p_ = torch.tensor(PB[i, :, n - 1, :],
                          device='cuda') \
            .to(torch.bfloat16)
        h2 = (x_ + a_) + m_
        b6_diff = max(b6_diff, float(
            (h2 - p_).abs().max()))
    b6_diff = float(b6_diff)
    b6_ok = bool(b6_diff == 0.0)
    log('[%s] b6 (x+a)+m=h2 bf16 diff=%.3e '
        'ok=%s' % (fkey, b6_diff, b6_ok))

    def base_z_gpu(base_i, nb):
        bx = torch.tensor(BX[base_i],
                          device='cuda') \
            .to(torch.bfloat16)
        ba = torch.tensor(BA[base_i],
                          device='cuda') \
            .to(torch.bfloat16)
        bp = torch.tensor(
            PB[base_i][:, nb - 1, :],
            device='cuda').to(torch.bfloat16)
        return torch.cat([bx, bx + ba, bp],
                         dim=0)

    def lens_cos(zinj, zbase, k):
        with torch.no_grad():
            li_ = F.linear(
                final_norm(zinj).float(), W32)
            lb_ = F.linear(
                final_norm(zbase).float(), W32)
            d = li_ - lb_
            t = TTG[k]
            num = (d * t.unsqueeze(0)) \
                .sum(dim=1)
            den = d.norm(dim=1) \
                * float(TTn[k])
            c = (num / den.clamp_min(1e-12)) \
                .double().cpu().numpy()
        return c

    # ==== E1 inj@34 ladder ====
    COS_LAD_34 = np.full(NP_USE, np.nan)
    PA34 = np.full(NP_USE, np.nan)
    PF34 = np.full(NP_USE, np.nan)
    ZH34 = np.zeros((NP_USE, NQW))
    ZH35 = np.zeros((NP_USE, NQW))
    ZA34 = np.zeros((NP_USE, HID))
    ZA35 = np.zeros((NP_USE, HID))
    b4_diff = 0.0
    b8_diff = 0.0
    for k in range(NP_USE):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[L34, ridx, :] = VB[pref_i][
            L34, off + ridx, :]
        lg, _, _, _, zX, zA, zM, zP, zH, \
            zZ, zG, zU, zACT = forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
        dlg = lg - LG[base_i]
        COS_LAD_34[k] = cosv(dlg, TT[k])
        b4_diff = max(b4_diff, float(np.max(
            np.abs(zX[L34].double().cpu()
                   .numpy()
                   - BX[base_i, L34]))))
        PA34[k] = lens_cos(
            torch.cat([zX, zX + zA, zP], dim=0),
            base_z_gpu(base_i, nb), k)[NL + L34]
        PF34[k] = lens_cos(
            torch.cat([zX, zX + zA, zP], dim=0),
            base_z_gpu(base_i, nb),
            k)[2 * NL + L34]
        ZA34[k] = zA[L34].double().cpu().numpy()
        ZA35[k] = zA[L35].double().cpu().numpy()
        ZH34[k] = zH[L34].double().cpu().numpy()
        ZH35[k] = zH[L35].double().cpu().numpy()
        d8 = float(np.max(np.abs(
            zX[L35].double().cpu().numpy()
            - zP[L34].double().cpu().numpy())))
        b8_diff = max(b8_diff, d8)
    b4_ok = bool(b4_diff == 0.0)
    b8_diff = float(b8_diff)
    b8_ok = bool(b8_diff == 0.0)
    log('[%s] b4 delta-x at L34 diff=%.3e ok=%s'
        % (fkey, b4_diff, b4_ok))
    log('[%s] b8 dzX_35 vs dzP_34 diff=%.3e '
        'ok=%s' % (fkey, b8_diff, b8_ok))

    med_c_34 = float(np.median(COS_LAD_34))
    pa34 = float(np.median(PA34))
    pf34 = float(np.median(PF34))
    log('[%s] E1 med_c(34)=%.4f PA=%.4f PF='
        '%.4f (forwards=%d)'
        % (fkey, med_c_34, pa34, pf34, FW[0]))

    if is_anchor and not SMOKE:
        adiff['a1'] = float(np.max(np.abs(
            COS_LAD_34 - ANCH['cos66_34'])))
        aok['a1'] = bool(adiff['a1'] == 0.0)
        log('[%s] a1 ladder row 34 vs 3066 npz: '
            'max|d|=%.3e ok=%s'
            % (fkey, adiff['a1'], aok['a1']))
        assert abs(med_c_34 - MEDC34_REF) \
            < 1e-12, med_c_34
        log('[%s] med_c_34 reference assert ok'
            % fkey)
        adiff['aref'] = max(
            abs(pa34 - PA34_REF),
            abs(pf34 - PF34_REF))
        aok['aref'] = bool(
            adiff['aref'] == 0.0)
        log('[%s] aref PA34/PF34 medians vs '
            '3069 constants: max|d|=%.3e ok=%s'
            % (fkey, adiff['aref'],
               aok['aref']))
    elif is_anchor:
        log('[%s] a1/med_c/aref skipped (smoke)'
            % fkey)

    # ==== E2H per-head observation ====
    Wo34 = layers[L34].self_attn.o_proj.weight \
        .detach().float()
    Wo35 = layers[L35].self_attn.o_proj.weight \
        .detach().float()
    WoT34 = Wo34.t().reshape(NQ, HDIM, HID)
    WoT35 = Wo35.t().reshape(NQ, HDIM, HID)
    DAH34 = np.zeros((NP_USE, NQ))
    DAH35 = np.zeros((NP_USE, NQ))
    HEADLIN34 = np.zeros(NP_USE)
    BX34b = np.zeros((NP_USE, HID))
    BA34b = np.zeros((NP_USE, HID))
    for k in range(NP_USE):
        base_i = int(BASE_K[k])
        BX34b[k] = BX[base_i, L34]
        BA34b[k] = BA[base_i, L34]
    for k in range(NP_USE):
        base_i = int(BASE_K[k])
        t = TT[k]
        tn = max(float(np.linalg.norm(t)), 1e-12)
        dzH34 = ZH34[k] - BH[base_i, L34]
        dzA34 = ZA34[k] - BA34b[k]
        DAH34[k], _, HEADLIN34[k] = head_decomp(
            dzH34, WoT34, Wo34, dzA34, t, tn)
        dzH35 = ZH35[k] - BH[base_i, L35]
        dzA35 = ZA35[k] - BA[base_i, L35]
        DAH35[k], _, _ = head_decomp(
            dzH35, WoT35, Wo35, dzA35, t, tn)
    DAH34_MED = np.median(DAH34, axis=0)
    DAH35_MED = np.median(DAH35, axis=0)
    hl34_med = float(np.median(HEADLIN34))
    log('[%s] E2H L34: headlin rel med=%.2e | '
        'top obs heads %s'
        % (fkey, hl34_med,
           np.argsort(DAH34_MED)[::-1][:6]
           .tolist()))
    del Wo34, Wo35, WoT34, WoT35
    gc.collect()
    torch.cuda.empty_cache()

    if is_anchor and not SMOKE:
        adiff['a8'] = max(
            float(np.max(np.abs(
                ZH34 - ANCH['zh34_71']))),
            float(np.max(np.abs(
                ZH35 - ANCH['zh35_71']))))
        aok['a8'] = bool(adiff['a8'] == 0.0)
        log('[%s] a8 ZH34/ZH35 vs 3071 npz: '
            'max|d|=%.3e ok=%s'
            % (fkey, adiff['a8'], aok['a8']))
        adiff['a9'] = max(
            float(np.max(np.abs(
                DAH34_MED
                - ANCH['dah34m_71']))),
            float(np.max(np.abs(
                DAH35_MED
                - ANCH['dah35m_71']))))
        aok['a9'] = bool(adiff['a9'] == 0.0)
        log('[%s] a9 DAH medians vs 3071 npz: '
            'max|d|=%.3e ok=%s'
            % (fkey, adiff['a9'], aok['a9']))
    elif is_anchor:
        log('[%s] a8/a9 skipped (smoke)' % fkey)

    # ==== b7 identity self-swaps ====
    b, base_i, pref_i, off = pair_idx(0)
    lg7a = forward_gen(
        assembled[base_i]['ids'],
        attn_swaps=[(L34, ALLH,
                     BH[base_i, L34])])[0]
    b7a_diff = float(np.max(np.abs(
        lg7a - LG[base_i])))
    lg7c = forward_gen(
        assembled[base_i]['ids'],
        act_swaps=[(L34, ALLI,
                    BACT34[base_i])])[0]
    b7c_diff = float(np.max(np.abs(
        lg7c - LG[base_i])))
    b7_ok = bool(b7a_diff == 0.0
                 and b7c_diff == 0.0)
    log('[%s] b7 identity self-swaps (attn '
        '%.1e act %.1e) ok=%s'
        % (fkey, b7a_diff, b7c_diff, b7_ok))

    def repv_of(k):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[L34, ridx, :] = VB[pref_i][
            L34, off + ridx, :]
        return base_i, repV

    def run_cond(mk_kwargs):
        cs = np.full(K3, np.nan)
        for k in range(K3):
            base_i, repV = repv_of(k)
            lg = forward_gen(
                assembled[base_i]['ids'],
                repl=repV,
                **mk_kwargs(base_i))[0]
            cs[k] = cosv(lg - LG[base_i],
                         TT[k])
        return cs

    # ==== E3 32-head single swap scan ====
    CS1H = np.full((NH, K3), np.nan)
    for h in range(NH):
        idx = HEAD_IDX[h]
        CS1H[h] = run_cond(
            lambda bi, idx=idx: {
                'attn_swaps': [
                    (L34, idx,
                     BH[bi, L34][idx])]})
        if (h + 1) % 8 == 0 or h == NH - 1:
            log('[%s] E3 scan progress %d/%d '
                '(forwards=%d)'
                % (fkey, h + 1, NH, FW[0]))
    r1_all32 = np.median(CS1H, axis=1) \
        - med_c_34
    log('[%s] E3 scan r1_all32: min=%.4f '
        'max=%.4f n_neg=%d'
        % (fkey, float(r1_all32.min()),
           float(r1_all32.max()),
           int((r1_all32 < 0).sum())))

    if is_anchor and not SMOKE:
        adiff['a10s'] = float(np.max(np.abs(
            r1_all32 - ANCH['r34_71'])))
        aok['a10s'] = bool(
            adiff['a10s'] == 0.0)
        log('[%s] a10s r1_all32 vs 3071 r34 '
            '(all 32): max|d|=%.3e ok=%s'
            % (fkey, adiff['a10s'],
               aok['a10s']))
    elif is_anchor:
        log('[%s] a10s skipped (smoke)' % fkey)

    # ==== per-family focal top8 (3071
    # criterion: argsort(r1) ascending) ====
    n_neg = int((r1_all32 < 0).sum())
    order = np.argsort(r1_all32)
    topk = min(8, n_neg)
    top8 = [int(h) for h in order[:topk]]
    top8_sel_ok = bool(
        topk == 8
        and bool((r1_all32[top8] < 0).all()))
    neg_sum = float(r1_all32[r1_all32 < 0]
                    .sum())
    capture8 = abs(float(
        r1_all32[top8].sum())) \
        / abs(neg_sum) if abs(neg_sum) > 1e-9 \
        else float('nan')
    log('[%s] top8 (3071 criterion)=%s '
        'n_neg=%d capture8=%.4f sel_ok=%s'
        % (fkey, top8, n_neg, capture8,
           top8_sel_ok))
    if is_anchor and not SMOKE:
        adiff['a15'] = float(
            0.0 if top8 == ANCH['top8_71']
            else 1.0)
        aok['a15'] = bool(
            top8 == ANCH['top8_71'])
        log('[%s] a15 top8 vs 3071 %s: ok=%s'
            % (fkey, ANCH['top8_71'],
               aok['a15']))

    # ==== E4 subset saturation sweep ====
    NT = len(top8)
    NM_ALL = 1 << NT
    M_FULL = NM_ALL - 1
    if SMOKE:
        MASKS = [m for m in (1, 2, 3, 5, 7, 15,
                             31, 63, 127, 255, 24,
                             85) if m < NM_ALL]
    else:
        MASKS = list(range(1, NM_ALL))
    NM = len(MASKS)
    log('[%s] E4 sweep: NT=%d masks=%d K3=%d'
        % (fkey, NT, NM, K3))

    CS = np.full((NM, K3), np.nan)
    for mi, m in enumerate(MASKS):
        sel = [ti for ti in range(NT)
               if m >> ti & 1]
        idx = np.concatenate(
            [HEAD_IDX[top8[ti]] for ti in sel])
        CS[mi] = run_cond(
            lambda bi, idx=idx: {
                'attn_swaps': [
                    (L34, idx,
                     BH[bi, L34][idx])]})
        if (mi + 1) % 32 == 0 or mi == NM - 1:
            log('[%s] E4 sweep progress %d/%d '
                '(forwards=%d)'
                % (fkey, mi + 1, NM, FW[0]))
    R_S = np.median(CS, axis=1) - med_c_34
    A_S = -R_S
    y_u8 = float(-R_S[MASKS.index(M_FULL)]) \
        if M_FULL in MASKS else float('nan')
    if NM > 0:
        log('[%s] E4 sweep R(S) collected: min='
            '%.4f max=%.4f (most negative at '
            'mask %d)'
            % (fkey, float(R_S.min()),
               float(R_S.max()),
               int(MASKS[int(np.argmin(R_S))])))
    else:
        log('[%s] E4 sweep empty (NT=0, '
            'degenerate family)' % fkey)

    if is_anchor and not SMOKE and NT == 8:
        mask_of = {m: i for i, m
                   in enumerate(MASKS)}
        d10 = 0.0
        for p_i, (a_, b_) in enumerate(
                itertools.combinations(
                    range(NT), 2)):
            m = (1 << a_) | (1 << b_)
            d10 = max(d10, abs(
                R_S[mask_of[m]]
                - ANCH['r2_73'][p_i]))
        d10 = max(d10, abs(
            R_S[mask_of[7]]
            - ANCH['r_t3_73']))
        d10 = max(d10, abs(
            R_S[mask_of[255]]
            - ANCH['r_u8_73']))
        adiff['a10p'] = float(d10)
        aok['a10p'] = bool(
            adiff['a10p'] == 0.0)
        log('[%s] a10p pairs/triplet/top8 vs '
            '3073 r2/r_t3/r_u8: max|d|=%.3e '
            'ok=%s' % (fkey, adiff['a10p'],
                       aok['a10p']))

    # a11: full 32-head swap
    lg_all = run_cond(lambda bi: {
        'attn_swaps': [(L34, ALLH,
                        BH[bi, L34])]})
    R_ALL = float(np.median(lg_all)
                  - med_c_34)
    if is_anchor and not SMOKE:
        adiff['a11'] = abs(R_ALL - GALL_3071)
        aok['a11'] = bool(
            adiff['a11'] == 0.0)
        log('[%s] a11 all-32 swap recov=%+.4f '
            'vs 3071 gA %+.4f: |d|=%.3e ok=%s'
            % (fkey, R_ALL, GALL_3071,
               adiff['a11'], aok['a11']))
    else:
        log('[%s] all-32 swap recov=%+.4f '
            '(a11 skipped)' % (fkey, R_ALL))

    # free the big banks before analysis
    del VB, PB, BX, BA, BM, BH, BZ35, BG35, \
        BU35, BACT34, BACT35, TTG, TTn, selfV
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] big banks freed (forwards=%d)'
        % (fkey, FW[0]))

    # ==== E5 submodularity ====
    AMP = {}
    for i, m in enumerate(MASKS):
        AMP[m] = float(-R_S[i])
    AMP[0] = 0.0

    def amp(m):
        return AMP.get(m)

    pairs4 = []
    if not SMOKE:
        for S in range(1, NM_ALL):
            for x in range(NT):
                if S >> x & 1:
                    continue
                mSx = S | (1 << x)
                base_m = amp(mSx) - amp(S)
                TT_ = S
                while True:
                    TT_ = (TT_ - 1) & S
                    if TT_ < 0:
                        break
                    if TT_ == S:
                        continue
                    mt = amp(TT_ | (1 << x)) \
                        - amp(TT_)
                    pairs4.append(base_m - mt)
                    if TT_ == 0:
                        break
        par = np.array(pairs4)
        n_sub_checked = len(par)
        if n_sub_checked:
            n_viol = int(
                (par > TOL_SUB).sum())
            viol_rate = float(n_viol) \
                / n_sub_checked
            worst_excess = float(par.max())
            log('[%s] E5 submodularity: %d '
                'inequalities, violations %d '
                '(%.3f percent) | worst excess='
                '%.4f'
                % (fkey, n_sub_checked, n_viol,
                   100.0 * viol_rate,
                   worst_excess))
        else:
            n_viol = 0
            viol_rate = float('nan')
            worst_excess = float('nan')
            log('[%s] E5 submodularity empty '
                '(degenerate)' % fkey)
        if is_anchor and NT == 8:
            PAIRS4_ARR = np.array(
                pairs4, dtype=np.float64)
            adiff['a13'] = float(np.max(
                np.abs(PAIRS4_ARR
                       - ANCH['pairs4_74'])))
            aok['a13'] = bool(
                adiff['a13'] == 0.0)
            log('[%s] a13 PAIRS4 vs 3074 npz: '
                'max|d|=%.3e ok=%s'
                % (fkey, adiff['a13'],
                   aok['a13']))
    else:
        n_sub_checked = 0
        n_viol = 0
        viol_rate = float('nan')
        worst_excess = float('nan')
        log('[%s] E5 submodularity skipped '
            '(smoke)' % fkey)

    if is_anchor and not SMOKE and NT == 8:
        mask_of = {m: i for i, m
                   in enumerate(MASKS)}
        adiff['a12'] = float(np.max(np.abs(
            A_S - ANCH['a_s_74'])))
        aok['a12'] = bool(
            adiff['a12'] == 0.0)
        log('[%s] a12 full A_S vs 3074 npz: '
            'max|d|=%.3e ok=%s'
            % (fkey, adiff['a12'],
               aok['a12']))

    # ==== E6 capacity-law fits ====
    fits = {}
    if not SMOKE and NT >= 2:
        MAG = np.abs(r1_all32[top8])
        x_u8 = float(MAG.sum())
        x_all32 = float(np.abs(r1_all32).sum())
        xs_all = []
        ys_all = []
        sizes = []
        for m in range(1, NM_ALL):
            x = float(sum(
                MAG[ti] for ti in range(NT)
                if m >> ti & 1))
            xs_all.append(x)
            ys_all.append(amp(m))
            sizes.append(popcount(m))
        xs_all = np.array(xs_all)
        ys_all = np.array(ys_all)
        sizes = np.array(sizes)
        le2 = sizes <= 2
        y_u8 = float(-R_S[MASKS.index(M_FULL)])
        y_all32 = float(-R_ALL)
        for name in ('add', 'exp', 'log',
                     'hill'):
            p_all, sse_all = grid_fit(
                name, xs_all, ys_all)
            p_2, sse_2 = grid_fit(
                name, xs_all[le2], ys_all[le2])
            pr8 = float(model_pred(
                name, x_u8, p_2))
            pr32 = float(model_pred(
                name, x_all32, p_2))
            e8 = abs(pr8 - y_u8)
            e32 = abs(pr32 - y_all32)
            fits[name] = {
                'p_all': list(p_all),
                'sse_all': sse_all,
                'p_le2': list(p_2),
                'sse_le2': sse_2,
                'pred_u8': pr8,
                'pred_all32': pr32,
                'err_u8': float(e8),
                'err_all32': float(e32),
                'rel_u8': float(
                    e8 / max(abs(y_u8),
                             1e-12)),
                'rel_all32': float(
                    e32 / max(abs(y_all32),
                              1e-12)),
                'pass_u8': bool(
                    e8 < GATE_PRED),
                'pass_all32': bool(
                    e32 < GATE_PRED)}
            log('[%s] E6 fit %-4s | all: p=%s '
                'sse=%.4f | le2: p=%s sse='
                '%.4f | pred S8=%.4f (err '
                '%.4f rel %.3f) all32=%.4f '
                '(err %.4f rel %.3f)'
                % (fkey, name,
                   np.round(p_all, 4).tolist(),
                   sse_all,
                   np.round(p_2, 4).tolist(),
                   sse_2, pr8, e8,
                   fits[name]['rel_u8'],
                   pr32, e32,
                   fits[name]['rel_all32']))
        passed = [nm for nm in fits
                  if fits[nm]['pass_u8']
                  and fits[nm]['pass_all32']]
        partial = [nm for nm in fits
                   if nm not in passed
                   and (fits[nm]['pass_u8']
                        or fits[nm]
                        ['pass_all32'])]
        if passed:
            best_nm = min(
                passed,
                key=lambda nm:
                fits[nm]['err_u8']
                + fits[nm]['err_all32'])
            fam_fit_verdict = \
                'capacity_law_' + best_nm
        elif partial:
            best_nm = min(
                partial,
                key=lambda nm:
                fits[nm]['err_u8']
                + fits[nm]['err_all32'])
            fam_fit_verdict = \
                'capacity_law_partial_' \
                + best_nm
        else:
            fam_fit_verdict = \
                'capacity_law_undetermined'
        log('[%s] E6 gate: passed=%s partial='
            '%s -> %s'
            % (fkey, passed, partial,
               fam_fit_verdict))
        if is_anchor and NT == 8:
            h_le2 = fits['hill']['p_le2']
            adiff['a16'] = float(max(
                abs(h_le2[i]
                    - ANCH['hill74_le2'][i])
                for i in range(3)))
            aok['a16'] = bool(
                h_le2 == ANCH['hill74_le2'])
            log('[%s] a16 hill p_le2 vs 3074 '
                'result %s: max|d|=%.3e ok=%s'
                % (fkey, ANCH['hill74_le2'],
                   adiff['a16'], aok['a16']))
    else:
        MAG = None
        x_u8 = float('nan')
        x_all32 = float('nan')
        if SMOKE:
            fam_fit_verdict = 'smoke_pending'
            log('[%s] E6 fits skipped (smoke)'
                % fkey)
        else:
            fam_fit_verdict = \
                'skipped_degenerate_nt%d' % NT
            log('[%s] E6 fits skipped '
                '(degenerate NT=%d)'
                % (fkey, NT))

    # ==== E7 Moebius spectrum ====
    b0m_diff = None
    b0m_ok = True
    spec = {}
    c10 = float('nan')
    max_mu_hi = float('nan')
    pos_pairs = []
    if not SMOKE:
        Aarr = np.zeros(NM_ALL, dtype=np.float64)
        for m in range(NM_ALL):
            Aarr[m] = amp(m)
        MU = np.zeros(NM_ALL, dtype=np.float64)
        for m in range(1, NM_ALL):
            s = 0.0
            t = m
            while True:
                sign = 1.0 \
                    if (popcount(m)
                        - popcount(t)) % 2 == 0 \
                    else -1.0
                s += sign * Aarr[t]
                if t == 0:
                    break
                t = (t - 1) & m
            MU[m] = s
        FAST = Aarr.copy()
        for i in range(NT):
            for m in range(NM_ALL):
                if m >> i & 1:
                    FAST[m] -= FAST[
                        m ^ (1 << i)]
        if NM_ALL > 1:
            b0m_diff = float(np.max(np.abs(
                FAST[1:] - MU[1:])))
        else:
            b0m_diff = 0.0
        b0m_ok = bool(b0m_diff <= 1e-12)
        log('[%s] b0m fast-vs-brute Moebius '
            'max|d|=%.3e ok=%s'
            % (fkey, b0m_diff, b0m_ok))
        if is_anchor and NT == 8:
            adiff['a14'] = float(np.max(
                np.abs(MU - ANCH['mu_75'])))
            aok['a14'] = bool(
                adiff['a14'] == 0.0)
            log('[%s] a14 MU vs 3075 npz: '
                'max|d|=%.3e ok=%s'
                % (fkey, adiff['a14'],
                   aok['a14']))
        orders = np.array(
            [popcount(m)
             for m in range(NM_ALL)])
        for k in range(2, NT + 1):
            sel = MU[1:][orders[1:] == k]
            spec[k] = {
                'n': int(len(sel)),
                'n_pos': int(
                    (sel > TOL_MOB).sum()),
                'n_neg': int(
                    (sel < -TOL_MOB).sum()),
                'max': float(sel.max()),
                'min': float(sel.min()),
                'sum': float(sel.sum())}
            log('[%s] E7 mu order %d: n=%d '
                'n_pos=%d n_neg=%d max=%+.4f '
                'min=%+.4f'
                % (fkey, k, spec[k]['n'],
                   spec[k]['n_pos'],
                   spec[k]['n_neg'],
                   spec[k]['max'],
                   spec[k]['min']))
        masks_hi = [m for m
                    in range(1, NM_ALL)
                    if popcount(m) >= 2]
        pos_pairs = [(float(MU[m]), m)
                     for m in masks_hi
                     if MU[m] > TOL_MOB]
        pos_pairs.sort(reverse=True)
        pos_vals = np.array(
            [v for v, _ in pos_pairs],
            dtype=np.float64) \
            if pos_pairs else np.zeros(0)
        if len(pos_vals):
            c10 = float(
                pos_vals[:10].sum()
                / pos_vals.sum())
        max_mu_hi = float(max(
            (MU[m] for m in masks_hi),
            default=0.0))
        log('[%s] E7 positive Mobius mass '
            '(>=2, >TOL): n=%d total=%.4f '
            'c10=%s max=%+.4f'
            % (fkey, len(pos_vals),
               float(pos_vals.sum())
               if len(pos_vals) else 0.0,
               c10, max_mu_hi))
        for v, m in pos_pairs[:12]:
            heads = [top8[i]
                     for i in range(NT)
                     if m >> i & 1]
            log('[%s] E7 top mu mask=%d order='
                '%d mu=%+.4f heads=%s'
                % (fkey, m, popcount(m), v,
                   heads))
    else:
        MU = None
        log('[%s] E7 spectrum skipped (smoke)'
            % fkey)

    # structure_stable (3074/3075 signature)
    if not SMOKE:
        hi_keys = [k for k in (4, 5, 6, 7, 8)
                   if k in spec]
        hi4_max = max(spec[k]['max']
                      for k in hi_keys) \
            if hi_keys else -1.0
        hill_u8 = fits.get('hill', {}).get(
            'pass_u8', False)
        hill_all32 = fits.get('hill', {}).get(
            'pass_all32', False)
        mu2_npos = spec[2]['n_pos'] \
            if 2 in spec else -1
        stable = bool(
            hill_u8 and hill_all32
            and mu2_npos == 0
            and hi4_max > TOL_MOB)
        log('[%s] structure_stable=%s (hill '
            'u8=%s all32=%s mu2_npos=%d '
            'hi4_max=%.4f)'
            % (fkey, stable, hill_u8,
               hill_all32, mu2_npos, hi4_max))
    else:
        stable = None

    setup_ok_f = bool(b0_ok and b1_ok and b3_ok
                      and b4_ok and b6_ok
                      and b7_ok
                      and aok['aref'])
    log('[%s] ==== family end: setup_ok=%s '
        'a_ok=%s forwards=%d ===='
        % (fkey, setup_ok_f,
           {k: aok[k] for k in AKEYS
            if not aok[k]} or 'all',
           FW[0]))

    return {
        'fkey': fkey,
        'domain': fam['domain'],
        'med_c_34': med_c_34,
        'pa34': pa34, 'pf34': pf34,
        'hl34_med': hl34_med,
        'med_tt_norm': med_tn,
        'COS_LAD_34': COS_LAD_34,
        'PA34': PA34, 'PF34': PF34,
        'ZH34': ZH34, 'ZH35': ZH35,
        'DAH34': DAH34, 'DAH35': DAH35,
        'DAH34_MED': DAH34_MED,
        'DAH35_MED': DAH35_MED,
        'r1_all32': r1_all32,
        'CS1H': CS1H,
        'n_neg': n_neg, 'topk': topk,
        'top8': top8,
        'top8_sel_ok': top8_sel_ok,
        'capture8': capture8,
        'MASKS': MASKS, 'CS': CS,
        'R_S': R_S, 'A_S': A_S,
        'R_ALL': R_ALL,
        'PAIRS4': (np.array(pairs4,
                            dtype=np.float64)
                   if not SMOKE else None),
        'n_sub_checked': n_sub_checked,
        'n_viol': n_viol,
        'viol_rate': viol_rate,
        'worst_excess': worst_excess,
        'MAG': MAG,
        'x_u8': x_u8, 'x_all32': x_all32,
        'y_u8': y_u8,
        'y_all32': (float(-R_ALL)
                    if not SMOKE
                    else float('nan')),
        'fits': fits,
        'fam_fit_verdict': fam_fit_verdict,
        'MU': MU, 'spec': spec, 'c10': c10,
        'max_mu_hi': max_mu_hi,
        'pos_pairs': pos_pairs,
        'stable': stable,
        'b0m_diff': b0m_diff,
        'b0m_ok': b0m_ok,
        'TT32': TT.astype(np.float32),
        'LG32': LG.astype(np.float32),
        'b0_diff': b0_diff, 'b0_ok': b0_ok,
        'b1_diff': b1_diff, 'b1_ok': b1_ok,
        'b3_ok': b3_ok,
        'b4_diff': b4_diff, 'b4_ok': b4_ok,
        'b5_diff': b5_diff,
        'b6_diff': b6_diff, 'b6_ok': b6_ok,
        'b7a_diff': b7a_diff,
        'b7c_diff': b7c_diff,
        'b7_ok': b7_ok,
        'b8_diff': b8_diff, 'b8_ok': b8_ok,
        'a2_max': a2_max,
        'a2lens_ok': a2lens_ok,
        'adiff': adiff, 'aok': aok,
        'setup_ok_f': setup_ok_f,
    }


RES_F = {}
for fk in FKEYS:
    RES_F[fk] = run_family(fk)
    gc.collect()
    torch.cuda.empty_cache()

# ==== E8 cross-family ====
log('E8 cross-family comparison')
ov = {}
jac = {}
sp_r1 = {}
sp_dah = {}
for a_, b_ in (('A', 'B'), ('A', 'C'),
               ('B', 'C')):
    key = a_ + b_
    ov[key] = len(set(RES_F[a_]['top8'])
                  & set(RES_F[b_]['top8']))
    jac[key] = float(ov[key])
    jac[key] = ov[key] / (16 - ov[key])
    sp_r1[key] = float(spearman(
        RES_F[a_]['r1_all32'],
        RES_F[b_]['r1_all32']))
    sp_dah[key] = float(spearman(
        np.abs(RES_F[a_]['DAH34_MED']),
        np.abs(RES_F[b_]['DAH34_MED'])))
    log('E8 %s-%s: top8 overlap=%d/8 '
        'jaccard=%.3f spearman(r1)=%.4f '
        'spearman(|DAH34_MED|)=%.4f'
        % (a_, b_, ov[key], jac[key],
           sp_r1[key], sp_dah[key]))
for fk in FKEYS:
    log('E8 %s: med_c=%.4f R_ALL=%+.4f '
        'a_u8=%.4f viol=%.3f capture8=%.4f '
        'hill=%s fit=%s'
        % (fk, RES_F[fk]['med_c_34'],
           RES_F[fk]['R_ALL'],
           RES_F[fk]['y_u8'],
           RES_F[fk]['viol_rate'],
           RES_F[fk]['capture8'],
           (np.round(
               RES_F[fk]['fits']['hill']
               ['p_le2'], 4).tolist()
            if RES_F[fk]['fits'] else None),
           RES_F[fk]['fam_fit_verdict']))

# ==== verdict ====
setup_ok_all = bool(all(
    RES_F[fk]['setup_ok_f'] for fk in FKEYS))
b8_all_ok = bool(all(
    RES_F[fk]['b8_ok'] for fk in FKEYS))
b0m_all_ok = bool(all(
    RES_F[fk]['b0m_ok'] for fk in FKEYS))
top8_sel_all_ok = bool(all(
    RES_F[fk]['top8_sel_ok']
    for fk in FKEYS))
verdict = 'smoke_pending'
if not SMOKE:
    if not setup_ok_all:
        verdict = 'setup_failed_cross_prompt'
    elif not RES_F['A']['aok']['a1']:
        verdict = 'anchor_mismatch_3066_ladder'
    elif not RES_F['A']['aok']['a8']:
        verdict = 'anchor_mismatch_3071_zh'
    elif not RES_F['A']['aok']['a9']:
        verdict = 'anchor_mismatch_3071_dah'
    elif not (RES_F['A']['aok']['a10s']
              and RES_F['A']['aok']['a10p']):
        verdict = 'anchor_mismatch_3073_sweep'
    elif not RES_F['A']['aok']['a11']:
        verdict = 'anchor_mismatch_3071_allswap'
    elif not RES_F['A']['aok']['a12']:
        verdict = 'anchor_mismatch_3074_asweep'
    elif not RES_F['A']['aok']['a13']:
        verdict = 'anchor_mismatch_3074_pairs'
    elif not RES_F['A']['aok']['a14']:
        verdict = 'anchor_mismatch_3075_mu'
    elif not RES_F['A']['aok']['a15']:
        verdict = 'anchor_mismatch_3071_top8'
    elif not RES_F['A']['aok']['a16']:
        verdict = 'anchor_mismatch_3074_hill'
    elif not b8_all_ok:
        verdict = 'block_output_mismatch'
    elif not top8_sel_all_ok:
        verdict = 'top8_selection_degenerate'
    elif not b0m_all_ok:
        verdict = 'mobius_transform_mismatch'
    else:
        n_stable = sum(
            1 for fk in FKEYS
            if RES_F[fk]['stable'])
        ov_min = min(ov['AB'], ov['AC'],
                     ov['BC'])
        if n_stable == 3 and ov_min >= 5:
            verdict = 'cross_prompt_stable'
        elif n_stable == 3:
            verdict = \
                'cross_prompt_stable_head_shift'
        elif n_stable == 2:
            verdict = 'cross_prompt_partial'
        else:
            verdict = 'cross_prompt_unstable'
    log('VERDICT: %s (setup_ok=%s b8=%s '
        'top8_sel=%s b0m=%s n_stable=%s '
        'ov=%s smoke=%s)'
        % (verdict, setup_ok_all, b8_all_ok,
           top8_sel_all_ok, b0m_all_ok,
           [RES_F[fk]['stable']
            for fk in FKEYS], ov, SMOKE))
else:
    log('VERDICT: smoke_pending')

# ==== npz ====
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'FORWARDS': np.int64(FW[0]),
    'SETUP_OK': np.bool_(setup_ok_all),
    'B8_OK': np.bool_(b8_all_ok),
    'TOP8_SEL_OK': np.bool_(top8_sel_all_ok),
    'B0M_OK': np.bool_(b0m_all_ok),
}
for fk in FKEYS:
    R = RES_F[fk]
    save['MED_C_34_' + fk] = np.float64(
        R['med_c_34'])
    save['PA34_' + fk] = R['PA34']
    save['PF34_' + fk] = R['PF34']
    save['COS_LAD_34_' + fk] = R['COS_LAD_34']
    save['ZH34_' + fk] = R['ZH34']
    save['ZH35_' + fk] = R['ZH35']
    save['DAH34_' + fk] = R['DAH34']
    save['DAH35_' + fk] = R['DAH35']
    save['DAH34_MED_' + fk] = R['DAH34_MED']
    save['DAH35_MED_' + fk] = R['DAH35_MED']
    save['R1_ALL32_' + fk] = R['r1_all32']
    save['CS1H_' + fk] = R['CS1H']
    save['TOP8_' + fk] = np.array(
        R['top8'], dtype=np.int64)
    save['N_NEG_' + fk] = np.int64(R['n_neg'])
    save['CAPTURE8_' + fk] = np.float64(
        R['capture8'])
    save['TOP8_SEL_OK_' + fk] = np.bool_(
        R['top8_sel_ok'])
    save['MASKS_' + fk] = np.array(
        R['MASKS'], dtype=np.int64)
    save['CS_' + fk] = R['CS']
    save['R_S_' + fk] = R['R_S']
    save['A_S_' + fk] = R['A_S']
    save['R_ALL_' + fk] = np.float64(R['R_ALL'])
    save['N_SUB_CHECKED_' + fk] = np.int64(
        R['n_sub_checked'])
    save['N_VIOL_' + fk] = np.int64(
        R['n_viol'])
    save['VIOL_RATE_' + fk] = np.float64(
        R['viol_rate'])
    save['X_U8_' + fk] = np.float64(R['x_u8'])
    save['X_ALL32_' + fk] = np.float64(
        R['x_all32'])
    save['B0_DIFF_' + fk] = np.float64(
        R['b0_diff'])
    save['B1_DIFF_' + fk] = np.float64(
        R['b1_diff'])
    save['B3_OK_' + fk] = np.bool_(R['b3_ok'])
    save['B4_DIFF_' + fk] = np.float64(
        R['b4_diff'])
    save['B4_OK_' + fk] = np.bool_(R['b4_ok'])
    save['B5_DIFF_' + fk] = np.float64(
        R['b5_diff'])
    save['B6_DIFF_' + fk] = np.float64(
        R['b6_diff'])
    save['B6_OK_' + fk] = np.bool_(R['b6_ok'])
    save['B7A_DIFF_' + fk] = np.float64(
        R['b7a_diff'])
    save['B7C_DIFF_' + fk] = np.float64(
        R['b7c_diff'])
    save['B7_OK_' + fk] = np.bool_(R['b7_ok'])
    save['B8_DIFF_' + fk] = np.float64(
        R['b8_diff'])
    save['B8_OK_' + fk] = np.bool_(R['b8_ok'])
    save['A2LENS_MAX_' + fk] = np.float64(
        R['a2_max'])
    save['STABLE_' + fk] = (
        np.bool_(R['stable'])
        if R['stable'] is not None
        else np.bool_(False))
    for ak in AKEYS:
        d_ = R['adiff'][ak]
        save[(ak.upper() + '_DIFF_' + fk)] = \
            np.float64(
                d_ if d_ is not None
                else np.nan)
        save[(ak.upper() + '_OK_' + fk)] = \
            np.bool_(R['aok'][ak])
    if not SMOKE:
        save['MU_' + fk] = R['MU']
        save['PAIRS4_' + fk] = R['PAIRS4']
if not SMOKE:
    for fk in FKEYS:
        save['TT_' + fk] = RES_F[fk]['TT32']
        save['LG_' + fk] = RES_F[fk]['LG32']
    save['OV_AB'] = np.int64(ov['AB'])
    save['OV_AC'] = np.int64(ov['AC'])
    save['OV_BC'] = np.int64(ov['BC'])
    save['SP_R1_AB'] = np.float64(sp_r1['AB'])
    save['SP_R1_AC'] = np.float64(sp_r1['AC'])
    save['SP_R1_BC'] = np.float64(sp_r1['BC'])
    save['SP_DAH_AB'] = np.float64(
        sp_dah['AB'])
    save['SP_DAH_AC'] = np.float64(
        sp_dah['AC'])
    save['SP_DAH_BC'] = np.float64(
        sp_dah['BC'])
np.savez(npz_path, **save)

# ==== result.json ====
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


fam_stats = {}
for fk in FKEYS:
    R = RES_F[fk]
    d_ = {
        'domain': R['domain'],
        'med_c_34': f64(R['med_c_34']),
        'pa34': f64(R['pa34']),
        'pf34': f64(R['pf34']),
        'hl34_med': f64(R['hl34_med']),
        'med_tt_norm': f64(R['med_tt_norm']),
        'r_all32': f64(R['R_ALL']),
        'a_u8': f64(R['y_u8']),
        'a_best_subset': (
            {'mask': int(R['MASKS'][int(
                np.argmax(R['A_S']))]),
             'amp': f64(float(
                 R['A_S'].max()))}
            if len(R['A_S']) else
            {'mask': None, 'amp': None}),
        'n_neg': int(R['n_neg']),
        'top8': R['top8'],
        'top8_sel_ok': bool(R['top8_sel_ok']),
        'capture8': f64(R['capture8']),
        'n_sub_checked': int(
            R['n_sub_checked']),
        'n_viol': int(R['n_viol']),
        'viol_rate': f64(R['viol_rate']),
        'worst_excess': f64(R['worst_excess']),
        'x_u8': f64(R['x_u8']),
        'x_all32': f64(R['x_all32']),
        'y_u8': f64(R['y_u8']),
        'y_all32': f64(R['y_all32']),
        'fam_fit_verdict': R['fam_fit_verdict'],
        'stable': (None
                   if R['stable'] is None
                   else bool(R['stable'])),
        'b_anchors': {
            'b0_diff': f64(R['b0_diff']),
            'b0_ok': bool(R['b0_ok']),
            'b1_diff': f64(R['b1_diff']),
            'b1_ok': bool(R['b1_ok']),
            'b3_ok': bool(R['b3_ok']),
            'b4_diff': f64(R['b4_diff']),
            'b4_ok': bool(R['b4_ok']),
            'b5_diff': f64(R['b5_diff']),
            'b6_diff': f64(R['b6_diff']),
            'b6_ok': bool(R['b6_ok']),
            'b7a_diff': f64(R['b7a_diff']),
            'b7c_diff': f64(R['b7c_diff']),
            'b7_ok': bool(R['b7_ok']),
            'b8_diff': f64(R['b8_diff']),
            'b8_ok': bool(R['b8_ok']),
            'a2lens_max': f64(R['a2_max']),
            'a2lens_ok': bool(R['a2lens_ok']),
            'setup_ok': bool(R['setup_ok_f'])},
        'anchors': {
            ak: {'diff': R['adiff'][ak],
                 'ok': bool(R['aok'][ak])}
            for ak in AKEYS},
    }
    if not SMOKE:
        d_['fits'] = {
            nm: {
                'p_all': [f64(v)
                          for v in dd['p_all']],
                'sse_all': f64(dd['sse_all']),
                'p_le2': [f64(v)
                          for v in dd['p_le2']],
                'sse_le2': f64(dd['sse_le2']),
                'pred_u8': f64(dd['pred_u8']),
                'pred_all32': f64(
                    dd['pred_all32']),
                'err_u8': f64(dd['err_u8']),
                'err_all32': f64(
                    dd['err_all32']),
                'rel_u8': f64(dd['rel_u8']),
                'rel_all32': f64(
                    dd['rel_all32']),
                'pass_u8': dd['pass_u8'],
                'pass_all32': dd[
                    'pass_all32'],
            } for nm, dd in R['fits'].items()}
        d_['spectrum'] = {
            str(k): v
            for k, v in R['spec'].items()}
        d_['c10'] = f64(R['c10'])
        d_['max_mu_hi'] = f64(R['max_mu_hi'])
        d_['n_pos_mu'] = int(
            len(R['pos_pairs']))
        d_['pos_sum'] = float(
            sum(v for v, _ in R['pos_pairs']))
        d_['top12_pos_mu'] = [
            {'mask': int(m),
             'order': popcount(m), 'mu': v,
             'heads': [R['top8'][i]
                       for i in range(len(
                           R['top8']))
                       if m >> i & 1]}
            for v, m in R['pos_pairs'][:12]]
        d_['asymptote_gap'] = f64(
            R['fits']['hill']['p_le2'][0]
            / max(abs(R['y_all32']), 1e-12))
    fam_stats[fk] = d_

result = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'elapsed': time.time() - t0,
    'forwards': int(FW[0]),
    'run': 'run1 authoritative (qwen3-4b bf16 '
           'single model, families A/B/C '
           'sequential)' if not SMOKE
    else 'smoke',
    'prereg': PREREG,
    'stats': {
        'families': fam_stats,
        'cross': {
            'ov_ab': int(ov['AB']),
            'ov_ac': int(ov['AC']),
            'ov_bc': int(ov['BC']),
            'jac_ab': f64(jac['AB']),
            'jac_ac': f64(jac['AC']),
            'jac_bc': f64(jac['BC']),
            'sp_r1_ab': f64(sp_r1['AB']),
            'sp_r1_ac': f64(sp_r1['AC']),
            'sp_r1_bc': f64(sp_r1['BC']),
            'sp_dah_ab': f64(sp_dah['AB']),
            'sp_dah_ac': f64(sp_dah['AC']),
            'sp_dah_bc': f64(sp_dah['BC']),
            'n_stable': (sum(
                1 for fk in FKEYS
                if RES_F[fk]['stable'])
                if not SMOKE else None),
        },
    },
    'verdict': verdict,
}
with open(os.path.join(OUT, 'result.json'),
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
    'setup_ok': setup_ok_all,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
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

del model, layers, tok, Wemb, final_norm, W32
gc.collect()
torch.cuda.empty_cache()
