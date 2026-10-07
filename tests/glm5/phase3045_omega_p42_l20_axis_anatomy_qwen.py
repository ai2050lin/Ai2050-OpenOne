# -*- coding: utf-8 -*-
# Phase 3045 - Omega-P42: L20 axis potency anatomy
# (run2 = fp32 redesign after the run1 bf16 noise-floor
# discovery; see corrections).
# RUN1 FINDINGS (registered): (i) a112 chain anchor
# FAILED -> probe showed the 3044 T5 "L20 inversion
# 3.42x" was a scale-mismatch artifact: raw med eff_u
# 3.4178 (= stored obs_t5) vs raw med eff_rand 3.4362,
# honest ratio 1.0154 - the null was computed on
# divided-pool ratios (~1 by construction) while obs was
# raw; (ii) bf16 flip-noise floor: ||dlg|| is FLAT at
# 10-15 across injection norms 0.1-3.0 and layers L3/L20
# (P2), +-symmetry cos +0.51 (P4) - 3044's logit
# responses were measured BELOW the bf16 rounding-flip
# noise floor; static quantization is only RMS 0.009 per
# logit (P1); (iii) run1 J columns (med_col 13.2,
# layer-invariant) were the same noise.
# RUN2 DESIGN (fp32): load the model in float32 (no
# rounding flips), rebuild the field axes in fp32 from
# fp32 re-extraction (cross-precision cos gate vs the
# bf16 3044 axes >= 0.999), and measure cleanly:
# (T1) efficiency at natural scale g: 8 old bodies x
#     (ubar + 12 fresh random dirs) at L3 (ubar3, g =
#     GBAR3) and L20 (ubar20, g = GBAR20); eff =
#     ||dlg||/g; stat = med over bodies of eff(axis)/
#     med(eff_rand); null = Monte-Carlo label
#     permutation (R=20000); spec iff p<0.05.
# (T2) dose ladder (body 0): norms {0.05,0.1,0.2,0.4,
#     0.8,1.6,3.2} x (ubar + 6 rand) x 2 layers;
#     descriptive growth curve (fp32 linearity).
# (T3) Jacobian anatomy: J = d lg/d V[pos] via 128
#     one-hot columns (h=0.25), probe bodies (0,3,6),
#     layers (3,20): (a) linear efficiency ubar vs 24
#     RND20 / 120 RND3 dirs (verbatim 3044 npz unit
#     dirs), stat/null as T1; (b) singular alignment:
#     frac_k(u) = energy in top-8 right singular dirs,
#     pooled axis-vs-random label permutation; (c)
#     linearity: cos(J (g ubar), dlg_actual) med >=
#     0.95 gate (flag).
# (T4) attenuation curve (body 0, layers 8/27): sigma_
#     max and Frobenius of J; descriptive.
# (T5) out-of-bank replication: 4 NEW bodies x (ubar20
#     + 12 fresh rand) at L20, g20_b from their own
#     prefix displacements; same scheme; rep iff
#     p<0.05.
# verdict_tree: spec = (p_T1_L20<0.05); align = (p_T3b
# _L20<0.05); rep = (p_T5<0.05); spec AND align ->
# faxis20_linear_readout_qwen; spec only ->
# faxis20_linear_local_qwen; (not spec) AND rep ->
# faxis20_nonlinear_qwen; else -> faxis20_null_qwen;
# single branch; L3 contrasts reported as flags.
# anchors: a107 duplicate prefill prompt0 (fp32) bit
# 0.0; a108 full duplicate extraction all 48 prompts
# bit 0.0; a109 source seals 3037..3044; a110 cross-
# precision: cos(V3_fp32_base, V3_bf16_ref) >= 0.999
# (8/8); a111 cross-precision axes: cos(ubar3_fp32,
# ubar3_bf16) >= 0.999 and cos(ubar20_fp32,
# ubar20_bf16) >= 0.999; a112 derived chain on the
# 3044 npz rec arrays: raw med eff_u == stored obs_t5
# (bit) - registering the scale-mismatch diagnosis;
# a113 J nondegenerate (med_i ||J e_i|| > 0.05) AND
# fp32 linearity med cos >= 0.95.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3045
NAME = 'omega_p42_l20_axis_anatomy_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 7
CURVE = (3, 8, 20, 27)
LAYERS_EX = CURVE
INJ_LAYERS = CURVE
H_STEP = 0.25
PROBE_BODIES = (0, 3, 6)
N_RND_T1 = 12
N_RND_NEW = 12
NORMS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
R_MC = 20000
SEED_RND_L3 = 9500
SEED_RND_L20 = 9501
SEED_RND_NEW = 9502
SEED_MC_T1_3 = 9600
SEED_MC_T1_20 = 9601
SEED_MC_T3 = 9602
SEED_MC_T3B_20 = 9603
SEED_MC_T3B_3 = 9604
SEED_MC_T5 = 9605
SEED_MAIN = 3009
DEGEN_NORM = 1e-6
COS_GATE = 0.999
LIN_GATE = 0.95
A113_GATE = 0.05
K_TOP = 8

BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
BODY_IDX = (0, 1, 3, 6, 7, 8, 9, 10)
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')
NEW_BODIES = (
    'The game was delayed, so',
    'She stayed home because',
    'The engine failed, therefore',
    'He kept smiling, although',)
NEW_TARGETS = ('so', 'because', 'therefore',
               'although')

PREREG = {
    'mode': 'fp32 MODEL (torch.float32, eager, seed '
            '3009) - run2 redesign after the run1 bf16 '
            'flip-noise-floor discovery; 8 old bodies '
            'x 4 conditions + 4 NEW bodies x 4 '
            'conditions; V read at the target position '
            'from past.layers[li].values[0,kv7]; layers '
            '(3,8,20,27); V PRIMARY',
    'question': 'L20 axis potency anatomy, fp32 (3045 '
                'A main line): with the bf16 flip-noise '
                'floor eliminated, does the L20 shared '
                'field axis move downstream logits more '
                'efficiently than size-matched random '
                'directions, is it aligned with high-'
                'gain singular directions of the '
                'Jacobian, does the effect replicate '
                'out-of-bank, and how does readout gain '
                'decay across layers?',
    'run1_findings': 'a112 chain anchor FAILED -> '
                     'probe (gpt5_temp/probe3045*): '
                     '(i) 3044 T5 obs_t5 3.4178 was the '
                     'RAW med eff_u while its null was '
                     'divided-pool ratios (~1); raw med '
                     'eff_rand 3.4362 -> honest ratio '
                     '1.0154, the L20 inversion claim is '
                     'RETRACTED; (ii) bf16 flip noise: '
                     '||dlg|| flat 10-15 for norms '
                     '0.1-3.0 at L3 and L20, +-symmetry '
                     'cos +0.51 -> all 3044 logit '
                     'statistics (dose rho -0.40, '
                     'antisym +0.56, T2 p 0.64, T3 p '
                     '0.40, T5) were noise-bound; '
                     'static bf16 quantization RMS '
                     '0.009/logit (norm 3.5); run1 J '
                     'columns med_col 13.2 layer-'
                     'invariant = same noise',
    'axes': 'rebuilt IN FP32 from fp32 re-extraction '
            '(D3/D20 from Vbase recompute); cross-'
            'precision gates vs the bf16 3044 npz '
            'arrays: cos >= 0.999 (a111); natural '
            'norms g = mean_c ||D(c,b)|| in fp32',
    'T1': 'efficiency at natural scale: 8 bodies x '
          '(ubar + N_RND_T1=12 fresh random dirs, '
          'seeds %d/%d) at L3 (ubar3, g=GBAR3) and '
          'L20 (ubar20, g=GBAR20), delta = g*dir; '
          'eff = ||dlg||/g; stat = med over bodies of '
          'eff(axis)/med(eff_rand); null = Monte-'
          'Carlo label permutation (R=%d, seeds '
          '%d/%d); spec iff p<0.05'
          % (SEED_RND_L3, SEED_RND_L20, R_MC,
             SEED_MC_T1_3, SEED_MC_T1_20),
    'T2': 'dose ladder (body 0): norms {0.05,0.1,0.2,'
          '0.4,0.8,1.6,3.2} x (ubar + 6 rand) x 2 '
          'layers; descriptive fp32 growth curve',
    'T3': 'Jacobian: J = d lg/d V[pos] via 128 one-'
          'hot columns (h=0.25), probe bodies (0,3,6), '
          'layers (3,20); (a) linear efficiency: '
          'ubar vs RND20 (24, verbatim 3044 npz) / '
          'RND3 (120) unit dirs, stat/null as T1 '
          '(seed %d); (b) singular alignment: frac_k '
          '= energy in top-%d right singular dirs, '
          'pooled axis-vs-random label permutation '
          '(R=%d, seeds %d/%d); (c) linearity gate: '
          'med cos(J (g ubar), dlg_actual) >= %.2f '
          % (SEED_MC_T3, K_TOP, R_MC,
             SEED_MC_T3B_20, SEED_MC_T3B_3,
             LIN_GATE),
    'T4': 'attenuation curve (body 0, layers 8/27): '
          'sigma_max and Frobenius norm of J; '
          'descriptive',
    'T5': 'out-of-bank replication: 4 NEW bodies x '
          '(ubar20 + 12 fresh rand, seed base %d) at '
          'L20, g20_b = mean_c ||D20_new(c,b)|| from '
          'their own fp32 prefix displacements; same '
          'scheme (seed %d); rep iff p<0.05'
          % (SEED_RND_NEW, SEED_MC_T5),
    'verdict_tree': 'spec = (p_T1_L20<0.05); align = '
                    '(p_T3b_L20<0.05); rep = (p_T5<'
                    '0.05); spec AND align -> '
                    'faxis20_linear_readout_qwen; '
                    'spec only -> '
                    'faxis20_linear_local_qwen; (not '
                    'spec) AND rep -> '
                    'faxis20_nonlinear_qwen; else -> '
                    'faxis20_null_qwen; single branch '
                    'assignment; L3 contrasts and '
                    'linearity reported as flags',
    'anchors': 'a107 duplicate prefill prompt0 (fp32, '
               'lg AND K/V all 4 layers) bit 0.0; '
               'a108 full duplicate extraction all 48 '
               'prompts (probs, lg, target-pos V at '
               'L3/L20) bit 0.0; a109 source seals '
               '3037..3044 sha8 match; a110 cross-'
               'precision base V3: cos vs 3037 bf16 '
               'npz >= 0.999 (8/8); a111 cross-'
               'precision axes: cos(ubar3, ubar3_bf16) '
               '>= 0.999 AND cos(ubar20, ubar20_bf16) '
               '>= 0.999; a112 derived chain on 3044 '
               'rec arrays: raw med eff_u == stored '
               'obs_t5 (<=1e-12), registering the '
               'scale-mismatch diagnosis; a113 J '
               'nondegenerate (med_i ||J e_i|| > 0.05) '
               'AND fp32 linearity med cos >= 0.95',
    'control': 'unit-norm random directions (fresh '
               'seeds fp32; verbatim 3044 banks for '
               'the Jacobian tests); label permutation '
               'Monte-Carlo; no other intervention',
    'statistics_discipline': 'arrays pre-initialized; '
                             'degenerate-norm gates; '
                             'nulls and obs on the SAME '
                             'scale (the 3044 T5 '
                             'lesson); nulls never '
                             'placed on intervened '
                             'quantities of the '
                             'treated arm; verdict in '
                             'one branch; small-n '
                             'caveat: 3 probe bodies, '
                             'exploratory margin',
    'corrections': 'run1 (bf16) crashed pre-'
                   'verdict on RND indexing AND '
                   'failed a112; probe '
                   'established the bf16 flip-'
                   'noise floor and the 3044 T5 '
                   'scale mismatch; run2 = fp32 '
                   'redesign; run2 completed but '
                   'T3a had a reduction-axis bug '
                   '(norm over the wrong axis of '
                   'the (NVOC,24) matrix -> per-'
                   'coordinate norms ~0.001, '
                   'inflated ratios 66/31 that '
                   'violate the sigma_max bound); '
                   'T3a fixed to axis=0 and fully '
                   'rerun as run3; T1/T2/T3b/T3c/'
                   'T4/T5 unaffected (reproduced '
                   'identically); a111 (0.99879 < '
                   '0.999) and the a113 med_col '
                   'gate (body0 L20 med_col 0.0150 '
                   'is REAL linear signal at the '
                   'measured L20 gain 0.052 per '
                   'unit, threshold miscalibrated) '
                   'documented as-is per prereg',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


os.makedirs(OUT, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)
for fn in ('execution.json', 'result.json', 'seal.json',
           NAME + '.npz'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG}
with open(os.path.join(OUT, 'execution.json'), 'w',
          encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False, indent=1)
log('execution.json written (prereg frozen) %s' % created)

torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
HDIM_V = int(model.config.head_dim) \
    if hasattr(model.config, 'head_dim') else 128
assert HDIM_V == 128
assert int(model.config.num_key_value_heads) == 8
log('model loaded fp32 head_dim=%d' % HDIM_V)

state = {li: {'on': False, 'pos': -1, 'delta': None}
         for li in INJ_LAYERS}
SL = slice(KV_HEAD * HDIM_V, (KV_HEAD + 1) * HDIM_V)


def make_hook(li):
    def h(module, inp, out):
        st = state[li]
        if st['on']:
            out[0, st['pos'], SL] += st['delta']
        return out
    return h


for li in INJ_LAYERS:
    layers[li].self_attn.v_proj.register_forward_hook(
        make_hook(li))


def forward_full(ids):
    for li in INJ_LAYERS:
        state[li]['on'] = False
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    past = out.past_key_values
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    l = lg - lg.max()
    p = np.exp(l)
    p = p / p.sum()
    kv = {}
    for li in LAYERS_EX:
        k = past.layers[li].keys[0, KV_HEAD] \
            .detach().double().cpu().numpy()
        v = past.layers[li].values[0, KV_HEAD] \
            .detach().double().cpu().numpy()
        kv[li] = (k, v)
    return p, lg, kv


def forward_inj(ids, li, pos, delta128):
    for lj in INJ_LAYERS:
        state[lj]['on'] = False
    d = torch.tensor(np.asarray(delta128),
                     dtype=torch.float32,
                     device='cuda')
    state[li].update(on=True, pos=pos, delta=d)
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    state[li]['on'] = False
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    past = out.past_key_values
    vrb = past.layers[li].values[0, KV_HEAD, pos] \
        .detach().double().cpu().numpy()
    return lg, vrb


# ---------- chain sources ----------
z43 = np.load(os.path.join(
    BASE, 'phase3043',
    'omega_p40_field_variance_qwen',
    'omega_p40_field_variance_qwen.npz'),
    allow_pickle=True)
z44 = np.load(os.path.join(
    BASE, 'phase3044',
    'omega_p41_field_axis_injection_qwen',
    'omega_p41_field_axis_injection_qwen.npz'),
    allow_pickle=True)

# ---------- assemble prompts ----------
word_tok = {}
for w in set(TARGETS) | set(NEW_TARGETS):
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    assert len(wi) == 1, (w, wi)
    word_tok[w] = int(wi[0])
assembled = []
for bi in range(len(BODIES)):
    for ci in range(len(PREFIXES)):
        s = (PREFIXES[ci] + ' ' + BODIES[bi]) \
            if PREFIXES[ci] else BODIES[bi]
        ids = [int(x) for x in tok(
            s, add_special_tokens=False)[
            'input_ids']]
        t = word_tok[TARGETS[bi]]
        assert ids.count(t) == 1, (bi, ci)
        assembled.append({'ids': ids,
                          'pos': ids.index(t),
                          'cond': ci, 'body': bi,
                          'new': False})
for bi in range(len(NEW_BODIES)):
    for ci in range(len(PREFIXES)):
        s = (PREFIXES[ci] + ' ' + NEW_BODIES[bi]) \
            if PREFIXES[ci] else NEW_BODIES[bi]
        ids = [int(x) for x in tok(
            s, add_special_tokens=False)[
            'input_ids']]
        t = word_tok[NEW_TARGETS[bi]]
        assert ids.count(t) == 1, (bi, ci)
        assembled.append({'ids': ids,
                          'pos': ids.index(t),
                          'cond': ci, 'body': bi,
                          'new': True})
n_pr = len(assembled)
n_old = len(BODIES) * len(PREFIXES)
assert n_old == 32 and n_pr == 48
idx_of = {}
for i in range(n_pr):
    idx_of[(assembled[i]['cond'],
            assembled[i]['body'],
            assembled[i]['new'])] = i
log('assembled %d prompts (%d old + %d new)'
    % (n_pr, n_old, n_pr - n_old))

# a107: duplicate prefill of prompt0
p_b0, lg_b0, kv_b0 = forward_full(assembled[0]['ids'])
p_d0, lg_d0, kv_d0 = forward_full(assembled[0]['ids'])
a107_diff = float(np.max(np.abs(lg_b0 - lg_d0)))
for li in LAYERS_EX:
    for a, b in zip(kv_b0[li], kv_d0[li]):
        a107_diff = max(a107_diff, float(
            np.max(np.abs(a - b))))

# ---------- main extraction ----------
Ps = [None] * n_pr
LGs = [None] * n_pr
KVs = {li: [None] * n_pr for li in LAYERS_EX}
for i in range(n_pr):
    p, lg, kv = forward_full(assembled[i]['ids'])
    Ps[i] = p
    LGs[i] = lg
    for li in LAYERS_EX:
        KVs[li][i] = kv[li]

# a108: full duplicate extraction
a108_diff = 0.0
for i in range(n_pr):
    p2, lg2, kv2 = forward_full(assembled[i]['ids'])
    a108_diff = max(a108_diff, float(
        np.max(np.abs(Ps[i] - p2))))
    a108_diff = max(a108_diff, float(
        np.max(np.abs(LGs[i] - lg2))))
    for li in (3, 20):
        pos = assembled[i]['pos']
        for a, b in zip(KVs[li][i], kv2[li]):
            a108_diff = max(a108_diff, float(
                np.max(np.abs(a[pos] - b[pos]))))
a108_diff = float(a108_diff)
log('a107=%.3e a108=%.3e' % (a107_diff, a108_diff))

# a109: source seals
a109_detail = []
for ph, nm in (
        (3037, 'omega_p34_kv_situational_'
               'specificity_qwen'),
        (3038, 'omega_p35_reentrant_readout_qwen'),
        (3039, 'omega_p36_direct_logistic_'
               'replication_qwen'),
        (3040, 'omega_p37_situational_'
               'component_qwen'),
        (3041, 'omega_p38_situational_axis_qwen'),
        (3042, 'omega_p39_style_field_probe_qwen'),
        (3043, 'omega_p40_field_variance_qwen'),
        (3044, 'omega_p41_field_axis_injection_'
               'qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a109_detail.append(bool(
        s == sealj['result_sha256_8']))
a109_ok = bool(a109_detail) and all(a109_detail)

Vbase = {li: np.zeros((n_pr, HDIM_V))
         for li in LAYERS_EX}
for i in range(n_pr):
    pos = assembled[i]['pos']
    for li in LAYERS_EX:
        Vbase[li][i] = KVs[li][i][1][pos]

# a110: cross-precision base V3 vs 3037 bf16 npz
z37 = np.load(os.path.join(
    BASE, 'phase3037',
    'omega_p34_kv_situational_specificity_qwen',
    'omega_p34_kv_situational_specificity_qwen.npz'),
    allow_pickle=True)
w37 = [str(x) for x in z37['occ_word']]
pr37 = [int(x) for x in z37['occ_prompt']]
po37 = [int(x) for x in z37['occ_pos']]
V37 = z37['V3']
ref = {}
for i in range(len(w37)):
    ref[(w37[i], pr37[i], po37[i])] = V37[i]
a110_min = 1.0
a110_matched = 0
for i in range(n_old):
    if assembled[i]['cond'] != 0:
        continue
    bi = assembled[i]['body']
    key = (TARGETS[bi], int(BODY_IDX[bi]),
           int(assembled[i]['pos']))
    if key not in ref:
        continue
    v32 = Vbase[3][i]
    vb = ref[key]
    cs = float(v32 @ vb) / (
        np.linalg.norm(v32)
        * np.linalg.norm(vb))
    a110_min = min(a110_min, cs)
    a110_matched += 1
a110_min = float(a110_min)
a110_ok = bool(a110_matched == len(BODIES)
               and a110_min >= COS_GATE)
log('a110 matched=%d/%d min cos=%.8f'
    % (a110_matched, len(BODIES), a110_min))

# ---------- fp32 axes ----------
c43 = z43['c_old']
b43 = z43['b_old']
D3_old = np.zeros((24, HDIM_V))
D20_old = np.zeros((24, HDIM_V))
for k in range(24):
    ci = int(c43[k])
    bi = int(b43[k])
    ic = idx_of[(ci, bi, False)]
    ib = idx_of[(0, bi, False)]
    D3_old[k] = Vbase[3][ic] - Vbase[3][ib]
    D20_old[k] = Vbase[20][ic] - Vbase[20][ib]


def alphas_from(D, conds):
    out = []
    for c in (1, 2, 3):
        m = conds == c
        a = D[m].mean(axis=0)
        nn = np.linalg.norm(a)
        out.append(a / nn if nn > 0 else a)
    return np.stack(out)


def unit(v):
    nn = np.linalg.norm(v)
    return v / nn if nn > 0 else v


ALPHAS3 = alphas_from(D3_old, c43)
ALPHAS20 = alphas_from(D20_old, c43)
UBAR3 = unit(ALPHAS3.mean(axis=0))
UBAR20 = unit(ALPHAS20.mean(axis=0))
G3 = np.zeros((3, len(BODIES)))
G20 = np.zeros((3, len(BODIES)))
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        m = (c43 == c) & (b43 == b)
        if m.any():
            G3[c - 1, b] = float(
                np.linalg.norm(D3_old[m][0]))
            G20[c - 1, b] = float(
                np.linalg.norm(D20_old[m][0]))
GBAR3 = G3.mean(axis=0)
GBAR20 = G20.mean(axis=0)
D20_new = np.zeros((12, HDIM_V))
GBAR20N = np.zeros(len(NEW_BODIES))
k_new = 0
for b in range(len(NEW_BODIES)):
    vals = []
    for c in (1, 2, 3):
        ic = idx_of[(c, b, True)]
        ib = idx_of[(0, b, True)]
        d = Vbase[20][ic] - Vbase[20][ib]
        D20_new[k_new] = d
        vals.append(float(np.linalg.norm(d)))
        k_new += 1
    GBAR20N[b] = float(np.mean(vals))

# a111: cross-precision axes vs bf16 3044 arrays
a111_min = 1.0
for mine, refn in ((UBAR3, 'ubar3'),
                   (UBAR20, 'ubar20'),
                   (ALPHAS3, 'alpha3'),
                   (ALPHAS20, 'alpha20')):
    rb = z44[refn]
    if rb.ndim == 2:
        for r in range(rb.shape[0]):
            cs = float(mine[r] @ rb[r]) \
                / (np.linalg.norm(mine[r])
                   * np.linalg.norm(rb[r]))
            a111_min = min(a111_min, cs)
    else:
        cs = float(mine @ rb) / (
            np.linalg.norm(mine)
            * np.linalg.norm(rb))
        a111_min = min(a111_min, cs)
a111_min = float(a111_min)
a111_ok = bool(a111_min >= COS_GATE)
log('a111 min cross-precision cos=%.8f ok=%s'
    % (a111_min, a111_ok))

# a112: derived chain - 3044 raw med eff_u == obs_t5
rec44_kind = z44['rec_kind']
rec44_body = z44['rec_body']
rec44_dlg = z44['rec_dlg']
rec44_iok = z44['rec_iok']
GB_bf16 = z44['G20'].mean(axis=0)
eff_u44 = []
for k in range(len(rec44_kind)):
    if int(rec44_kind[k]) != 3:
        continue
    if not bool(rec44_iok[k]):
        continue
    b = int(rec44_body[k])
    eff_u44.append(float(rec44_dlg[k])
                   / float(GB_bf16[b]))
a112_raw = float(np.median(eff_u44)) \
    if eff_u44 else float('nan')
a112_d = float(abs(a112_raw
                   - float(z44['obs_t5'])))
a112_ok = bool(a112_d <= 1e-12)
log('a112 raw med eff_u=%.6f vs stored obs_t5=%.6f '
    'd=%.2e ok=%s (scale-mismatch diagnosis '
    'registered)' % (a112_raw, float(z44['obs_t5']),
                     a112_d, a112_ok))

RND20 = z44['rnd_dirs20']
RND3 = z44['rnd_dirs3']

# ---------- run records ----------
rec_layer = []
rec_kind = []
rec_body = []
rec_dlg = []
n_integ_fail = 0


def record(li, kind, b, nd, ok):
    rec_layer.append(li)
    rec_kind.append(kind)
    rec_body.append(b)
    rec_dlg.append(nd)
    if not ok:
        globals()['n_integ_fail'] += 1


def integrity(vinj, vbase, delta):
    diff = vinj - vbase
    nd = float(np.linalg.norm(diff))
    ndl = float(np.linalg.norm(delta))
    if nd < 1e-9 or ndl < 1e-9:
        return False
    cos = float(diff @ delta) / (nd * ndl)
    ratio = nd / ndl
    return bool(cos > 0.9 and 0.5 <= ratio <= 1.5)


# ---------- T1: efficiency at natural scale ----------
log('=== T1 efficiency (fp32, natural scale) ===')
EFF_A3 = {}
EFF_A20 = {}
EFF_R3 = {}
EFF_R20 = {}
for b in range(len(BODIES)):
    base_i = idx_of[(0, b, False)]
    pos = assembled[base_i]['pos']
    ids = assembled[base_i]['ids']
    for li, gvec, seed, ea, er in (
            (3, GBAR3, SEED_RND_L3, EFF_A3,
             EFF_R3),
            (20, GBAR20, SEED_RND_L20, EFF_A20,
             EFF_R20)):
        g = float(gvec[b])
        if g < DEGEN_NORM:
            continue
        axis = UBAR3 if li == 3 else UBAR20
        rng = np.random.default_rng(seed + b)
        R = rng.standard_normal((N_RND_T1,
                                 HDIM_V))
        R = R / np.linalg.norm(R, axis=1)[:, None]
        dlg_a, vrb = forward_inj(
            ids, li, pos, g * axis)
        ok = integrity(vrb, Vbase[li][base_i],
                       g * axis)
        na = float(np.linalg.norm(
            dlg_a - LGs[base_i]))
        record(li, 0 if li == 3 else 1, b, na,
               ok)
        ea[b] = na / g
        effs = []
        for r in range(N_RND_T1):
            dr = R[r] * g
            lg_i, vrbr = forward_inj(
                ids, li, pos, dr)
            okr = integrity(vrbr,
                            Vbase[li][base_i],
                            dr)
            nr = float(np.linalg.norm(
                lg_i - LGs[base_i]))
            effs.append(nr / g)
            record(li, 2 if li == 3 else 3, b,
                   nr, okr)
        er[b] = np.array(effs)
    log('body%d done' % b)


def mc_med_ratio(pools, seed):
    mc = np.random.default_rng(seed)
    obs = float(np.median(
        [p[0] / float(np.median(p[1:]))
         for p in pools]))
    null = np.zeros(R_MC)
    for it in range(R_MC):
        vals = np.zeros(len(pools))
        for pi, effs in enumerate(pools):
            i = mc.integers(0, len(effs))
            rest = np.delete(effs, i)
            vals[pi] = effs[i] / max(
                float(np.median(rest)), 1e-30)
        null[it] = float(np.median(vals))
    p = float(np.mean(null >= obs))
    return obs, float(np.median(null)), p


pools3 = []
pools20 = []
for b in range(len(BODIES)):
    if b in EFF_A3 and b in EFF_R3:
        pools3.append(np.concatenate(
            ([EFF_A3[b]], EFF_R3[b])))
    if b in EFF_A20 and b in EFF_R20:
        pools20.append(np.concatenate(
            ([EFF_A20[b]], EFF_R20[b])))
obs_t1_3, nul3, p_t1_3 = mc_med_ratio(
    pools3, SEED_MC_T1_3)
obs_t1_20, nul20, p_t1_20 = mc_med_ratio(
    pools20, SEED_MC_T1_20)
log('T1: L20 obs=%.4f null=%.4f p=%.5f | L3 '
    'obs=%.4f null=%.4f p=%.5f'
    % (obs_t1_20, nul20, p_t1_20, obs_t1_3, nul3,
       p_t1_3))

# ---------- T2: dose ladder (body 0) ----------
log('=== T2 dose ladder (body 0) ===')
b = 0
base_i = idx_of[(0, b, False)]
pos = assembled[base_i]['pos']
ids = assembled[base_i]['ids']
ladder = {}
for li, gvec, axis in ((3, GBAR3, UBAR3),
                       (20, GBAR20, UBAR20)):
    rows = []
    for nrm in NORMS:
        dlg_a, _ = forward_inj(
            ids, li, pos, nrm * axis)
        na = float(np.linalg.norm(
            dlg_a - LGs[base_i]))
        vals = [na]
        rng = np.random.default_rng(
            9700 + int(nrm * 100) + li)
        R = rng.standard_normal((6, HDIM_V))
        R = R / np.linalg.norm(
            R, axis=1)[:, None]
        for r in range(6):
            lg_i, _ = forward_inj(
                ids, li, pos, nrm * R[r])
            vals.append(float(np.linalg.norm(
                lg_i - LGs[base_i])))
        rows.append((nrm, na,
                     float(np.median(vals[1:]))))
        record(li, 4, b, na, True)
    ladder[li] = rows
    log('T2 L%d: %s' % (li, [
        'n=%.2f a=%.3f r=%.3f' % rw
        for rw in rows]))

# ---------- T3: Jacobian anatomy ----------
log('=== T3 Jacobian (fp32) ===')
NVOC = len(lg_b0)
JEFF = {}
JFRAC = {}
JLINE = {}
a113_j = True
for b in PROBE_BODIES:
    base_i = idx_of[(0, b, False)]
    pos = assembled[base_i]['pos']
    ids = assembled[base_i]['ids']
    for li in (3, 20):
        J = np.zeros((NVOC, HDIM_V))
        for i in range(HDIM_V):
            d = np.zeros(HDIM_V)
            d[i] = H_STEP
            lg_i, vrb = forward_inj(
                ids, li, pos, d)
            J[:, i] = lg_i - LGs[base_i]
            ok = integrity(
                vrb, Vbase[li][base_i], d)
            record(li, 5, b,
                   float(np.linalg.norm(
                       J[:, i])), ok)
        coln = np.linalg.norm(J, axis=0)
        med_col = float(np.median(coln))
        if med_col <= A113_GATE:
            a113_j = False
        GS = J.T @ J
        evals, evecs = np.linalg.eigh(GS)
        order = np.argsort(-evals)
        evals = evals[order]
        evecs = evecs[:, order]
        smax = float(np.sqrt(max(evals[0], 0.0)))
        fro = float(np.sqrt(max(evals.sum(),
                                0.0)))
        Vt = evecs[:, :K_TOP]
        if li == 20:
            axis = UBAR20
            rnds = RND20[b]
            g = float(GBAR20[b])
        else:
            axis = UBAR3
            rnds = RND3[:, b].reshape(
                -1, HDIM_V)
            g = float(GBAR3[b])
        fr_axis = float(np.sum(
            (Vt.T @ axis) ** 2))
        fr_rnd = np.sum((Vt.T @ rnds.T) ** 2,
                        axis=0)
        JFRAC[(b, li)] = (fr_axis, fr_rnd)
        eff_ax = float(np.linalg.norm(J @ axis))
        eff_rnd = np.linalg.norm(J @ rnds.T,
                                 axis=0)
        JEFF[(b, li)] = (eff_ax, eff_rnd)
        pred = J @ (g * axis)
        dlg_a, vrb = forward_inj(
            ids, li, pos, g * axis)
        dact = dlg_a - LGs[base_i]
        na = float(np.linalg.norm(dact))
        npd = float(np.linalg.norm(pred))
        cs = float(pred @ dact) / (npd * na) \
            if na > 1e-12 and npd > 1e-12 \
            else float('nan')
        JLINE[(b, li)] = (cs, npd / na)
        del J
        log('J body%d L%d: med_col=%.4f smax=%.2f '
            'fro=%.1f lin cos=%.4f'
            % (b, li, med_col, smax, fro, cs))

pools_j20 = []
pools_j3 = []
for b in PROBE_BODIES:
    ea, er = JEFF[(b, 20)]
    pools_j20.append(np.concatenate(([ea], er)))
    ea3, er3 = JEFF[(b, 3)]
    pools_j3.append(np.concatenate(([ea3], er3)))


def mc_med_ratio_raw(pools, seed):
    return mc_med_ratio(pools, seed)


obs_t3a_20, nulj20, p_t3a_20 = mc_med_ratio_raw(
    pools_j20, SEED_MC_T3)
obs_t3a_3, nulj3, p_t3a_3 = mc_med_ratio_raw(
    pools_j3, SEED_MC_T3)
log('T3a: L20 obs=%.4f null=%.4f p=%.5f | L3 '
    'obs=%.4f null=%.4f p=%.5f'
    % (obs_t3a_20, nulj20, p_t3a_20, obs_t3a_3,
       nulj3, p_t3a_3))


def t3b_stat(layer):
    ax_vals = []
    rnd_vals = []
    for b in PROBE_BODIES:
        fa, fr = JFRAC[(b, layer)]
        ax_vals.append(fa)
        rnd_vals.extend(list(fr))
    ax_vals = np.array(ax_vals)
    rnd_vals = np.array(rnd_vals)
    obs = float(ax_vals.mean()
                - rnd_vals.mean())
    pool = np.concatenate((ax_vals, rnd_vals))
    nax = len(ax_vals)
    seed = SEED_MC_T3B_20 if layer == 20 \
        else SEED_MC_T3B_3
    mcx = np.random.default_rng(seed)
    null = np.zeros(R_MC)
    for it in range(R_MC):
        idx = mcx.permutation(len(pool))
        null[it] = float(pool[idx[:nax]].mean()
                         - pool[idx[nax:]].mean())
    p = float(np.mean(null >= obs))
    return obs, p, float(rnd_vals.mean())


obs_t3b_20, p_t3b_20, rndm20 = t3b_stat(20)
obs_t3b_3, p_t3b_3, rndm3 = t3b_stat(3)
log('T3b: L20 axis=%.4f rand=%.4f (chance %.4f) '
    'p=%.5f | L3 axis=%.4f rand=%.4f p=%.5f'
    % (obs_t3b_20 + rndm20, rndm20,
       K_TOP / float(HDIM_V), p_t3b_20,
       obs_t3b_3 + rndm3, rndm3, p_t3b_3))

lin_cs20 = [JLINE[(b, 20)][0]
            for b in PROBE_BODIES]
med_lin20 = float(np.nanmedian(lin_cs20))
log('T3c: med linearity cos L20=%.4f' % med_lin20)

# ---------- T4: attenuation curve ----------
log('=== T4 curve (body 0, L8/L27) ===')
curve = {}
for li in (8, 27):
    b = 0
    base_i = idx_of[(0, b, False)]
    pos = assembled[base_i]['pos']
    ids = assembled[base_i]['ids']
    J = np.zeros((NVOC, HDIM_V))
    for i in range(HDIM_V):
        d = np.zeros(HDIM_V)
        d[i] = H_STEP
        lg_i, _ = forward_inj(ids, li, pos, d)
        J[:, i] = lg_i - LGs[base_i]
    GS = J.T @ J
    evals = np.linalg.eigvalsh(GS)
    smax = float(np.sqrt(max(evals[-1], 0.0)))
    fro = float(np.sqrt(max(evals.sum(), 0.0)))
    curve[li] = (smax, fro)
    del J
    log('T4: L%d sigma_max=%.4f fro=%.2f'
        % (li, smax, fro))

# ---------- T5: out-of-bank replication ----------
log('=== T5 new bodies (L20) ===')
pools_new = []
for b in range(len(NEW_BODIES)):
    base_i = idx_of[(0, b, True)]
    pos = assembled[base_i]['pos']
    ids = assembled[base_i]['ids']
    g = float(GBAR20N[b])
    if g < DEGEN_NORM:
        continue
    dlg_a, vrb = forward_inj(ids, 20, pos,
                             g * UBAR20)
    ok = integrity(vrb, Vbase[20][base_i],
                   g * UBAR20)
    na = float(np.linalg.norm(
        dlg_a - LGs[base_i]))
    record(20, 6, b, na, ok)
    eff_u = na / g
    effs = []
    rng = np.random.default_rng(
        SEED_RND_NEW + b)
    R = rng.standard_normal((N_RND_NEW, HDIM_V))
    R = R / np.linalg.norm(R, axis=1)[:, None]
    for r in range(N_RND_NEW):
        dr = R[r] * g
        lg_i, vrbr = forward_inj(ids, 20, pos,
                                 dr)
        okr = integrity(vrbr, Vbase[20][base_i],
                        dr)
        nr = float(np.linalg.norm(
            lg_i - LGs[base_i]))
        effs.append(nr / g)
        record(20, 7, b, nr, okr)
    pools_new.append(np.array([eff_u] + effs))
if pools_new:
    obs_t5, nul5, p_t5 = mc_med_ratio_raw(
        pools_new, SEED_MC_T5)
else:
    obs_t5, nul5, p_t5 = (float('nan'),
                          float('nan'),
                          float('nan'))
log('T5: obs=%.4f null=%.4f p=%.5f (bodies=%d)'
    % (obs_t5, nul5, p_t5, len(pools_new)))

# ---------- verdict ----------
a113_ok = bool(a113_j and med_lin20 >= LIN_GATE)
spec = bool(p_t1_20 < 0.05)
align = bool(p_t3b_20 < 0.05)
rep = bool(p_t5 < 0.05)
if spec and align:
    verdict = 'faxis20_linear_readout_qwen'
elif spec:
    verdict = 'faxis20_linear_local_qwen'
elif rep:
    verdict = 'faxis20_nonlinear_qwen'
else:
    verdict = 'faxis20_null_qwen'
anchor_core_ok = bool(
    a107_diff == 0.0 and a108_diff == 0.0
    and a109_ok and a110_ok and a111_ok
    and a112_ok)

log('=== verdict ===')
log('a107=%r a108=%r a109=%s a110_ok=%s (%.6f,%d) '
    'a111_ok=%s (%.6f) a112_ok=%s (%.2e) a113=%s '
    'integ_fail=%d'
    % (a107_diff, a108_diff, a109_ok, a110_ok,
       a110_min, a110_matched, a111_ok, a111_min,
       a112_ok, a112_d, a113_ok, n_integ_fail))
log('T1 p20=%.5f p3=%.5f | T3a p20=%.5f | T3b '
    'p20=%.5f | T5 p=%.5f | lin=%.4f'
    % (p_t1_20, p_t1_3, p_t3a_20, p_t3b_20, p_t5,
       med_lin20))
log('VERDICT=%s anchor_core_ok=%s'
    % (verdict, anchor_core_ok))

elapsed = time.time() - t0

npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    bodies=np.array(BODIES),
    new_bodies=np.array(NEW_BODIES),
    alpha3=ALPHAS3, ubar3=UBAR3,
    alpha20=ALPHAS20, ubar20=UBAR20,
    G3=G3, G20=G20, GBAR3=GBAR3, GBAR20=GBAR20,
    GBAR20N=GBAR20N,
    eff_a3=np.array([EFF_A3.get(b,
                     float('nan'))
                     for b in range(8)]),
    eff_a20=np.array([EFF_A20.get(b,
                      float('nan'))
                      for b in range(8)]),
    obs_t1_20=np.float64(obs_t1_20),
    p_t1_20=np.float64(p_t1_20),
    obs_t1_3=np.float64(obs_t1_3),
    p_t1_3=np.float64(p_t1_3),
    ladder3=np.array([[rw[0], rw[1], rw[2]]
                      for rw in ladder[3]]),
    ladder20=np.array([[rw[0], rw[1], rw[2]]
                       for rw in ladder[20]]),
    obs_t3a_20=np.float64(obs_t3a_20),
    p_t3a_20=np.float64(p_t3a_20),
    obs_t3a_3=np.float64(obs_t3a_3),
    p_t3a_3=np.float64(p_t3a_3),
    obs_t3b_20=np.float64(obs_t3b_20),
    rndm20=np.float64(rndm20),
    p_t3b_20=np.float64(p_t3b_20),
    obs_t3b_3=np.float64(obs_t3b_3),
    rndm3=np.float64(rndm3),
    p_t3b_3=np.float64(p_t3b_3),
    med_lin20=np.float64(med_lin20),
    curve8=np.array(curve[8]),
    curve27=np.array(curve[27]),
    obs_t5=np.float64(obs_t5),
    p_t5=np.float64(p_t5),
    n_integ_fail=np.int64(n_integ_fail),
    a107_diff=np.float64(a107_diff),
    a108_diff=np.float64(a108_diff),
    a109_ok=np.bool_(a109_ok),
    a110_min=np.float64(a110_min),
    a111_min=np.float64(a111_min),
    a112_raw=np.float64(a112_raw),
    a112_d=np.float64(a112_d),
    a113_ok=np.bool_(a113_ok),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'run': 'run3 authoritative (fp32; run2 T3a '
           'reduction-axis bug fixed; T1/T2/T3b/'
           'T3c/T4/T5 reproduced identically; '
           'run1 bf16 crashed pre-verdict, see '
           'run1_findings)',
    'anchor_core_ok': anchor_core_ok,
    'anchors': {
        'a107_dup_prefill_bit': a107_diff,
        'a108_dup_all_bit': a108_diff,
        'a109_source_seals': a109_ok,
        'a110_cross_precision_min_cos': a110_min,
        'a110_matched': a110_matched,
        'a111_cross_precision_min_cos': a111_min,
        'a112_raw_med_eff_u': a112_raw,
        'a112_d_vs_stored': a112_d,
        'a113_j_and_linearity': a113_ok,
        'a113_gate': A113_GATE,
        'n_integ_fail': n_integ_fail,
    },
    'T1_efficiency': {'obs_L20': obs_t1_20,
                      'p_L20': p_t1_20,
                      'obs_L3': obs_t1_3,
                      'p_L3': p_t1_3,
                      'n_bodies': len(pools20)},
    'T2_dose_ladder': {'L3': ladder[3],
                       'L20': ladder[20]},
    'T3a_linear_eff': {'obs_L20': obs_t3a_20,
                       'p_L20': p_t3a_20,
                       'obs_L3': obs_t3a_3,
                       'p_L3': p_t3a_3},
    'T3b_singular_alignment': {
        'axis_frac_L20': obs_t3b_20 + rndm20,
        'rand_frac_L20': rndm20,
        'p_L20': p_t3b_20,
        'axis_frac_L3': obs_t3b_3 + rndm3,
        'rand_frac_L3': rndm3,
        'p_L3': p_t3b_3,
        'chance': K_TOP / float(HDIM_V)},
    'T3c_linearity': {'med_cos_L20': med_lin20,
                      'gate': LIN_GATE},
    'T4_curve': {'L8': curve[8], 'L27': curve[27]},
    'T5_out_of_bank': {'obs': obs_t5, 'p': p_t5,
                       'n_bodies': len(pools_new)},
    'flags': {'spec': spec, 'align': align,
              'rep': rep,
              'lin_gate': bool(med_lin20
                               >= LIN_GATE),
              'n_integ_fail': n_integ_fail},
    'prereg': PREREG,
    'elapsed_s': round(elapsed, 1),
}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)


def sha8(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()) \
            .hexdigest()[:8]


seal = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(__file__)),
    'verdict': verdict,
    'anchor_core_ok': anchor_core_ok,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('sealed npz8=%s result8=%s exec8=%s script8=%s '
    'elapsed=%.1fs'
    % (seal['npz_sha256_8'], seal['result_sha256_8'],
       seal['exec_sha256_8'], seal['script_sha256_8'],
       elapsed))
log('sealed')
