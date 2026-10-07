# -*- coding: utf-8 -*-
# Phase 3044 - Omega-P41: field-axis causal injection
# Phase 3043 (fieldvar_prefix_plus_body_qwen) established:
# prefix-induced V-write displacements decompose into a
# body-content main effect (64%) + a shared, transferable
# prefix-field component (15%) with a single dominant
# common axis (pairwise med |cos| = 0.65, 10.8x Gaussian)
# that generalizes to NEW bodies (11.4x). This phase moves
# the field from correlational to CAUSAL: inject the
# measured axes directly into the L3 V write (v_proj
# output, kv head 7 slice) of base-condition prompts and
# measure downstream logit effects.
# (T1) dose and antisymmetry: per (prefix c, body b) pair
#     inject alpha_c at scales s in {0.5,1,2,4} x g_bc
#     (g_bc = the natural displacement norm from the 3043
#     npz) and -s; stat1 = med Spearman rho(||dlg||, s)
#     over positive scales (one-sided exact binomial sign
#     test); stat2 = med cos(dlg(+s), dlg(-s)) at s=1 and
#     s=2 (expect ~ -1).
# (T2) axis efficiency specificity: eff = ||dlg(s=+1)||
#     / g_bc for alpha_c vs 40 size-matched random unit
#     directions per pair; stat = med over pairs of
#     eff_alpha / med(eff_rand); null = Monte-Carlo label
#     permutation within each pair's pool (R=20000,
#     seed 9102); spec iff p<0.05.
# (T3) causal reproduction of the prefix: does injecting
#     alpha_c at s=+1 reproduce the prefix's OWN down-
#     stream logit shift? stat = med over pairs of
#     cos(dlg_inj(alpha_c,+1), dlg_pref(c,b)) where
#     dlg_pref = lg(prefix cond) - lg(base cond); null =
#     the random-dir meds over dirs valid in ALL pairs
#     (deterministic); repro iff p<0.05; pairs with
#     ||dlg_pref|| < 1e-6 excluded.
# (T5) L20 control: shared axis ubar20 at s=+1 x g20_b
#     vs 24 random dirs per body; same label-permutation
#     scheme (seed 9103); ratio20 vs ratio2 reported as
#     a flag only.
# Integrity (bf16 reality): injected V readback at the
# target pos must satisfy cos(vinj - vbase, delta) > 0.9
# and norm ratio in [0.5, 1.5] (bf16 rounding tolerance);
# failing runs excluded and counted; non-target positions
# bit-identical (0.0) on 3 probe runs using the FULL
# injected L3 cache. Global sham gate: max ||dlg|| over
# alpha |s|>=2 runs > 0.05.
# Extraction verbatim 3042/3043 (8 old bodies x 4 conds).
# PREREG frozen below BEFORE any observation.
import os
import json
import time
import math
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3044
NAME = 'omega_p41_field_axis_injection_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 7
LAYERS_EX = (3, 20)
INJ_LAYERS = (3, 20)
N_RND_L3 = 40
N_RND_L20 = 24
SCALES = (0.5, 1.0, 2.0, 4.0)
S1 = 1
S2 = 2
R_MC = 20000
SEED_MAIN = 3009
SEED_MC_L3 = 9102
SEED_MC_L20 = 9103
DEGEN_NORM = 1e-6
INTEG_COS = 0.9
INTEG_RATIO = (0.5, 1.5)
A106_GATE = 0.05
A104_GATE = 1e-12

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

PREREG = {
    'mode': 'pure prefill, eager attention, bf16, seed '
            '3009; 8 old bodies x 4 conditions (verbatim '
            '3042/3043 assembly); V read at the target '
            'position from past.layers[li].values[0,kv7]; '
            'layers (3, 20); V PRIMARY',
    'question': 'field-axis causal injection (3044 A '
                'main line): does injecting the measured '
                'prefix-field axes into the L3 V write '
                'move downstream logits efficiently '
                '(axis specificity), dose-dependently, '
                'antisymmetrically, and in the direction '
                'the real prefix moves them (causal '
                'reproduction)? L20 as layer control.',
    'injection': 'forward hook on layers[li].self_attn.'
                 'v_proj; output slice [kv7*128:(kv7+1)*'
                 '128] at the target position gets += '
                 'delta (bf16); delta = s * g * unit_dir; '
                 'alpha_c / ubar3 rebuilt verbatim from '
                 'the 3043 npz (D3_old, c_old); g_bc = '
                 'natural displacement norms from the '
                 'same npz; ubar20 / g20 from our own '
                 'D20 recompute (base bit-anchored); '
                 'random dirs = per-pair seeded unit '
                 'Gaussians, size-matched',
    'T1': 'dose/antisymmetry: per (c,b) pair alpha_c at '
          's in +-{0.5,1,2,4} x g_bc; stat1 = med '
          'Spearman rho(||dlg||, s) over positive scales '
          '(pairs with <3 valid positive scales '
          'excluded), one-sided exact binomial sign '
          'test vs 0; stat2 = med cos(dlg(+s), dlg(-s)) '
          'at s=1 and s=2',
    'T2': 'axis efficiency: eff = ||dlg(+1)||/g_bc; '
          'alpha_c vs N_RND_L3=40 size-matched random '
          'dirs per pair; stat = med over valid pairs '
          'of eff_alpha/med(eff_rand); null = Monte-'
          'Carlo label permutation within each pool '
          '(R=%d, seed %d), one-sided p; spec iff '
          'p<0.05' % (R_MC, SEED_MC_L3),
    'T3': 'causal reproduction: stat = med over valid '
          'pairs of cos(dlg_inj(alpha_c,+1), '
          'dlg_pref(c,b)); dlg_pref = lg(cond c) - '
          'lg(cond 0) same body; pairs with '
          '||dlg_pref||<1e-6 or failed integrity '
          'excluded; null = random-dir meds over dirs '
          'valid in ALL valid pairs (deterministic); '
          'repro iff p<0.05',
    'T5': 'L20 control: ubar20 at s=+1 x g20_b (8 '
          'bodies) vs N_RND_L20=24 random dirs per '
          'body; same Monte-Carlo scheme (seed %d); '
          'ratio20 vs ratio3 comparison reported as a '
          'flag only' % SEED_MC_L20,
    'integrity': 'readback gate: cos(vinj-vbase, delta) '
                 '> 0.9 and ||ratio|| in [0.5,1.5] '
                 '(bf16 tolerance); failing runs '
                 'excluded from all statistics and '
                 'counted; non-target positions bit '
                 '0.0 on 3 probe runs against the FULL '
                 'injected L3 cache; global sham: '
                 'max||dlg|| over alpha |s|>=2 > 0.05',
    'verdict_tree': 'spec_ok = (p_T2<0.05); repro_ok = '
                    '(p_T3<0.05); spec_ok AND repro_ok '
                    '-> fieldaxis_causal_readout_qwen; '
                    'spec_ok only -> '
                    'fieldaxis_local_logit_qwen; else '
                    '-> fieldaxis_null_qwen; single '
                    'branch assignment; dose/antisym/'
                    'sham/L20 reported as flags',
    'anchors': 'a100 duplicate prefill prompt0 (lg AND '
               'K3/V3/K20/V20) bit 0.0; a101 full '
               'duplicate extraction all 32 prompts '
               '(probs, lg AND target-pos V) bit 0.0; '
               'a102 source seals 3037..3043 sha8 '
               'match; a103 base-condition V3 vs 3037 '
               'npz bit 0.0 (8/8); a104 displacement-'
               'chain vs 3043 npz: recomputed D3 == '
               'D3_old bit 0.0, norms == norms_old bit '
               '0.0, recomputed alpha pairwise med|cos| '
               '== obs_t3 (<=1e-12); a105 injection '
               'integrity all included runs pass + 3 '
               'non-target bit probes; a106 sham '
               'effect present',
    'control': 'size-matched random directions for T2/'
               'T3/T5; exact label permutation Monte-'
               'Carlo; no other intervention',
    'statistics_discipline': 'arrays pre-initialized; '
                             'degenerate-norm gates; '
                             'nulls never placed on '
                             'intervened quantities of '
                             'the treated arm; verdict '
                             'in one branch; small-n '
                             'caveat: <=24 pairs, '
                             'exploratory margin',
    'corrections': 'run1 completed (verdict '
                   'fieldaxis_null_qwen) but T3 '
                   'was mis-normalized: missing '
                   '||dlg_pref|| denominator, obs '
                   'value 26.97>1 exposed it; the '
                   'projection-vs-null comparison '
                   'was internally consistent '
                   '(p=0.45) but not the '
                   'preregistered cosine; T3 '
                   'corrected to the cosine '
                   '(per-pair denominators differ '
                   'across pairs so the med '
                   'ordering can change -> full '
                   'rerun); T1/T2/T5 are norm-'
                   'based, unaffected, and '
                   'reproduce identically under '
                   'the frozen seeds; verdict '
                   'tree unchanged',
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
    MODEL_DIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
HDIM_V = int(model.config.head_dim) \
    if hasattr(model.config, 'head_dim') else 128
assert HDIM_V == 128
assert int(model.config.num_key_value_heads) == 8
log('model loaded head_dim=%d' % HDIM_V)

W_U = model.lm_head.weight.detach().float() \
    .cpu().numpy()
VOCAB = W_U.shape[0]
assert VOCAB == int(model.config.vocab_size)

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


layers[3].self_attn.v_proj.register_forward_hook(
    make_hook(3))
layers[20].self_attn.v_proj.register_forward_hook(
    make_hook(20))


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
            .detach().float().cpu().numpy()
        v = past.layers[li].values[0, KV_HEAD] \
            .detach().float().cpu().numpy()
        kv[li] = (k, v)
    return p, lg, kv


def forward_inj(ids, li, pos, delta128, full=False):
    for lj in INJ_LAYERS:
        state[lj]['on'] = False
    d = torch.tensor(np.asarray(delta128),
                     dtype=torch.bfloat16,
                     device='cuda')
    state[li].update(on=True, pos=pos, delta=d)
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    state[li]['on'] = False
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    past = out.past_key_values
    v3p = past.layers[3].values[0, KV_HEAD, pos] \
        .detach().float().cpu().numpy()
    v20p = past.layers[20].values[0, KV_HEAD, pos] \
        .detach().float().cpu().numpy()
    if full:
        v3f = past.layers[3].values[0, KV_HEAD] \
            .detach().float().cpu().numpy()
        return lg, v3p, v20p, v3f
    return lg, v3p, v20p


# ---------- assemble prompts (old bank only) ----------
word_tok = {}
for w in set(TARGETS):
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
                          'cond': ci, 'body': bi})
n_pr = len(assembled)
assert n_pr == 32
idx_of = {}
for i in range(n_pr):
    idx_of[(assembled[i]['cond'],
            assembled[i]['body'])] = i
log('assembled %d prompts' % n_pr)

# a100: duplicate prefill of prompt0
p_b0, lg_b0, kv_b0 = forward_full(assembled[0]['ids'])
p_d0, lg_d0, kv_d0 = forward_full(assembled[0]['ids'])
a100_diff = float(np.max(np.abs(lg_b0 - lg_d0)))
for li in LAYERS_EX:
    for a, b in zip(kv_b0[li], kv_d0[li]):
        a100_diff = max(a100_diff, float(
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

# a101: full duplicate extraction
a101_diff = 0.0
for i in range(n_pr):
    p2, lg2, kv2 = forward_full(assembled[i]['ids'])
    a101_diff = max(a101_diff, float(
        np.max(np.abs(Ps[i] - p2))))
    a101_diff = max(a101_diff, float(
        np.max(np.abs(LGs[i] - lg2))))
    for li in LAYERS_EX:
        pos = assembled[i]['pos']
        for a, b in zip(KVs[li][i], kv2[li]):
            a101_diff = max(a101_diff, float(
                np.max(np.abs(a[pos] - b[pos]))))
a101_diff = float(a101_diff)
log('a100=%.3e a101=%.3e' % (a100_diff, a101_diff))

# a102: source seals
a102_detail = []
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
        (3043, 'omega_p40_field_variance_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a102_detail.append(bool(
        s == sealj['result_sha256_8']))
a102_ok = bool(a102_detail) and all(a102_detail)

V3o = np.zeros((n_pr, HDIM_V))
V20o = np.zeros((n_pr, HDIM_V))
for i in range(n_pr):
    pos = assembled[i]['pos']
    V3o[i] = KVs[3][i][1][pos]
    V20o[i] = KVs[20][i][1][pos]

# a103: cross-phase bit anchor vs 3037 npz (base cond)
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
a103_diff = 0.0
a103_matched = 0
for i in range(n_pr):
    if assembled[i]['cond'] != 0:
        continue
    bi = assembled[i]['body']
    key = (TARGETS[bi], int(BODY_IDX[bi]),
           int(assembled[i]['pos']))
    if key not in ref:
        continue
    a103_diff = max(a103_diff, float(np.max(
        np.abs(V3o[i] - ref[key]))))
    a103_matched += 1
a103_diff = float(a103_diff)
a103_ok = bool(a103_matched == len(BODIES)
               and a103_diff == 0.0)
log('a103 matched=%d/%d max|dV3|=%.3e'
    % (a103_matched, len(BODIES), a103_diff))

# a104: displacement-chain anchor vs 3043 npz
z43 = np.load(os.path.join(
    BASE, 'phase3043',
    'omega_p40_field_variance_qwen',
    'omega_p40_field_variance_qwen.npz'),
    allow_pickle=True)
c43 = z43['c_old']
b43 = z43['b_old']
D3_old = np.zeros((24, HDIM_V))
for k in range(24):
    ci = int(c43[k])
    bi = int(b43[k])
    D3_old[k] = V3o[idx_of[(ci, bi)]] \
        - V3o[idx_of[(0, bi)]]
a104_dD = float(np.max(np.abs(D3_old - z43['D3_old'])))
nrm_rec = np.linalg.norm(D3_old, axis=1)
a104_dnorm = float(np.max(np.abs(
    nrm_rec - z43['norms_old'])))


def alphas_from(D, conds):
    out = []
    for c in (1, 2, 3):
        m = conds == c
        a = D[m].mean(axis=0)
        nn = np.linalg.norm(a)
        out.append(a / nn if nn > 0 else a)
    return np.stack(out)


ALPHAS3 = alphas_from(D3_old, c43)
iu3 = np.triu_indices(3, 1)
mc_rec = float(np.median(np.abs(
    ALPHAS3 @ ALPHAS3.T)[iu3]))
a104_dmed = float(abs(mc_rec - float(z43['obs_t3'])))
a104_ok = bool(a104_dD == 0.0 and a104_dnorm == 0.0
               and a104_dmed <= A104_GATE)
log('a104 dD=%.3e dnorm=%.3e dmed=%.3e ok=%s'
    % (a104_dD, a104_dnorm, a104_dmed, a104_ok))

UBAR3 = ALPHAS3.mean(axis=0)
nn = np.linalg.norm(UBAR3)
UBAR3 = UBAR3 / nn if nn > 0 else UBAR3

G3 = np.zeros((3, len(BODIES)))
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        m = (c43 == c) & (b43 == b)
        if m.any():
            G3[c - 1, b] = float(
                np.linalg.norm(D3_old[m][0]))

# L20 displacements (own recompute; base bit-anchored)
D20_old = np.zeros((24, HDIM_V))
for k in range(24):
    ci = int(c43[k])
    bi = int(b43[k])
    D20_old[k] = V20o[idx_of[(ci, bi)]] \
        - V20o[idx_of[(0, bi)]]
ALPHAS20 = alphas_from(D20_old, c43)
UBAR20 = ALPHAS20.mean(axis=0)
nn = np.linalg.norm(UBAR20)
UBAR20 = UBAR20 / nn if nn > 0 else UBAR20
G20 = np.zeros((3, len(BODIES)))
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        m = (c43 == c) & (b43 == b)
        if m.any():
            G20[c - 1, b] = float(
                np.linalg.norm(D20_old[m][0]))
GBAR20 = G20.mean(axis=0)

# ---------- prefix logit shifts ----------
DLG_PREF = np.zeros((3, len(BODIES), VOCAB))
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        DLG_PREF[c - 1, b] = LGs[idx_of[(c, b)]] \
            - LGs[idx_of[(0, b)]]
pref_norm = np.linalg.norm(DLG_PREF, axis=2)
log('dlg_pref norm med=%.4f min=%.4f'
    % (float(np.median(pref_norm)),
       float(pref_norm.min())))


def integrity(vinj, vbase, delta):
    diff = vinj - vbase
    nd = float(np.linalg.norm(diff))
    ndl = float(np.linalg.norm(delta))
    if nd < 1e-9 or ndl < 1e-9:
        return False, 0.0, nd / max(ndl, 1e-12)
    cos = float(diff @ delta) / (nd * ndl)
    ratio = nd / ndl
    ok = bool(cos > INTEG_COS
              and INTEG_RATIO[0] <= ratio
              <= INTEG_RATIO[1])
    return ok, cos, ratio


# ---------- random direction banks ----------
RND3 = np.zeros((3, len(BODIES), N_RND_L3, HDIM_V))
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        rng = np.random.default_rng(
            SEED_MC_L3 * 7 + 1000 * b + c)
        R = rng.standard_normal((N_RND_L3, HDIM_V))
        R = R / np.maximum(
            np.linalg.norm(R, axis=1), 1e-12)[:, None]
        RND3[c - 1, b] = R
RND20 = np.zeros((len(BODIES), N_RND_L20, HDIM_V))
for b in range(len(BODIES)):
    rng = np.random.default_rng(
        SEED_MC_L20 * 7 + b)
    R = rng.standard_normal((N_RND_L20, HDIM_V))
    R = R / np.maximum(
        np.linalg.norm(R, axis=1), 1e-12)[:, None]
    RND20[b] = R

# ---------- run records ----------
rec_layer = []
rec_kind = []
rec_body = []
rec_cond = []
rec_scale = []
rec_dlg = []
rec_iok = []
rec_icos = []
rec_iratio = []
n_integ_fail = 0


def record(li, kind, b, c, sg, nd, ok, cs, rt):
    rec_layer.append(li)
    rec_kind.append(kind)
    rec_body.append(b)
    rec_cond.append(c)
    rec_scale.append(sg)
    rec_dlg.append(nd)
    rec_iok.append(bool(ok))
    rec_icos.append(cs)
    rec_iratio.append(rt)


# ---------- L3 alpha dose runs ----------
log('=== L3 alpha dose injections ===')
ALPHA_N = np.zeros((3, 8, 2, len(SCALES)))
ALPHA_OK = np.zeros((3, 8, 2, len(SCALES)),
                    dtype=bool)
ALPHA_D1 = np.zeros((3, 8, 2, VOCAB))
ALPHA_D2 = np.zeros((3, 8, 2, VOCAB))
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        g = G3[c - 1, b]
        if g < DEGEN_NORM:
            log('skip pair c%d_b%d (degenerate g)'
                % (c, b))
            continue
        base_i = idx_of[(0, b)]
        pos = assembled[base_i]['pos']
        ids = assembled[base_i]['ids']
        a_dir = ALPHAS3[c - 1]
        for si, s in enumerate(SCALES):
            for sign in (0, 1):
                sg = -s if sign else s
                delta = sg * g * a_dir
                lg_i, v3p, v20p = forward_inj(
                    ids, 3, pos, delta)
                ok, cs, rt = integrity(
                    v3p, V3o[base_i], delta)
                d = lg_i - LGs[base_i]
                nd = float(np.linalg.norm(d))
                ALPHA_N[c - 1, b, sign, si] = nd
                ALPHA_OK[c - 1, b, sign, si] = ok
                if s == S1:
                    ALPHA_D1[c - 1, b, sign] = d
                if s == S2:
                    ALPHA_D2[c - 1, b, sign] = d
                record(3, 0, b, c, sg, nd, ok,
                       cs, rt)
                if not ok:
                    n_integ_fail += 1
        log('c%d_b%d done (g=%.4f integ_fail=%d)'
            % (c, b, g, n_integ_fail))
log('alpha dose complete (integ_fail=%d)'
    % n_integ_fail)

# probe: non-target positions bit-identical (3 runs,
# full injected L3 cache vs base cache)
probe_max = 0.0
for c in (1, 2, 3):
    b = 0
    if G3[c - 1, b] < DEGEN_NORM:
        continue
    base_i = idx_of[(0, b)]
    pos = assembled[base_i]['pos']
    delta = G3[c - 1, b] * ALPHAS3[c - 1]
    _, _, _, v3f = forward_inj(
        assembled[base_i]['ids'], 3, pos, delta,
        full=True)
    dfull = v3f - KVs[3][base_i][1]
    mask = np.ones(dfull.shape[0], dtype=bool)
    mask[pos] = False
    probe_max = max(probe_max, float(
        np.max(np.abs(dfull[mask]))))
probe_max = float(probe_max)
log('non-target probe max=%.3e' % probe_max)
a105_probe_ok = bool(probe_max == 0.0)

# ---------- L3 random control runs (s=+1) ----------
log('=== L3 random control injections ===')
RAND_EFF = np.zeros((3, 8, N_RND_L3))
RAND_COS = np.zeros((3, 8, N_RND_L3))
RAND_OK = np.ones((3, 8, N_RND_L3), dtype=bool)
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        g = G3[c - 1, b]
        if g < DEGEN_NORM:
            RAND_OK[c - 1, b] = False
            continue
        base_i = idx_of[(0, b)]
        pos = assembled[base_i]['pos']
        ids = assembled[base_i]['ids']
        for r in range(N_RND_L3):
            delta = g * RND3[c - 1, b, r]
            lg_i, v3p, v20p = forward_inj(
                ids, 3, pos, delta)
            ok, cs, rt = integrity(
                v3p, V3o[base_i], delta)
            d = lg_i - LGs[base_i]
            nd = float(np.linalg.norm(d))
            RAND_EFF[c - 1, b, r] = nd / g
            RAND_COS[c - 1, b, r] = float(
                d @ DLG_PREF[c - 1, b]) \
                / max(nd, 1e-12) \
                / pref_norm[c - 1, b]
            RAND_OK[c - 1, b, r] = ok
            record(3, 2, b, c, 1.0, nd, ok,
                   cs, rt)
            if not ok:
                n_integ_fail += 1
log('random L3 done (integ_fail total=%d)'
    % n_integ_fail)

# ---------- L3 ubar runs ----------
log('=== L3 ubar injections ===')
UBAR_N = np.zeros((8, 2, len(SCALES)))
UBAR_OK = np.ones((8, 2, len(SCALES)), dtype=bool)
UBAR_D1 = np.zeros((8, VOCAB))
for b in range(len(BODIES)):
    g = float(G3[:, b].mean())
    if g < DEGEN_NORM:
        UBAR_OK[b] = False
        continue
    base_i = idx_of[(0, b)]
    pos = assembled[base_i]['pos']
    ids = assembled[base_i]['ids']
    for si, s in enumerate(SCALES):
        for sign in (0, 1):
            sg = -s if sign else s
            delta = sg * g * UBAR3
            lg_i, v3p, v20p = forward_inj(
                ids, 3, pos, delta)
            ok, cs, rt = integrity(
                v3p, V3o[base_i], delta)
            d = lg_i - LGs[base_i]
            nd = float(np.linalg.norm(d))
            UBAR_N[b, sign, si] = nd
            UBAR_OK[b, sign, si] = ok
            if s == S1:
                UBAR_D1[b] = d
            record(3, 1, b, -1, sg, nd, ok,
                   cs, rt)
            if not ok:
                n_integ_fail += 1
log('ubar L3 done (integ_fail total=%d)'
    % n_integ_fail)

# ---------- statistics ----------
def spearman(y):
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 3 or np.std(y) == 0:
        return float('nan')
    r = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(
        r, np.arange(len(y),
                     dtype=np.float64))[0, 1])


log('=== T1 dose (L3 alpha) ===')
rho_list = []
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        ys = []
        ss = []
        for si, s in enumerate(SCALES):
            if ALPHA_OK[c - 1, b, 0, si]:
                ys.append(ALPHA_N[c - 1, b, 0, si])
                ss.append(s)
        if len(ys) >= 3:
            rho_list.append(spearman(ys))
rho_list = [r for r in rho_list if r == r]
med_rho = float(np.median(rho_list)) \
    if rho_list else float('nan')
k_pos = int(np.sum(np.array(rho_list) > 0))
n_rho = len(rho_list)
p_dose = 0.0
for j in range(k_pos, n_rho + 1):
    p_dose += math.comb(n_rho, j)
p_dose = p_dose / (2.0 ** n_rho) if n_rho \
    else float('nan')
log('T1 dose: med rho=%.4f (n=%d, k_pos=%d) p=%.5f'
    % (med_rho, n_rho, k_pos, p_dose))

log('=== T1 antisymmetry ===')
anti = {S1: [], S2: []}
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        for s, arr in ((S1, ALPHA_D1),
                       (S2, ALPHA_D2)):
            si = SCALES.index(float(s))
            if G3[c - 1, b] < DEGEN_NORM:
                continue
            if ALPHA_OK[c - 1, b, 0, si] \
                    and ALPHA_OK[c - 1, b, 1, si]:
                dp = arr[c - 1, b, 0]
                dm = arr[c - 1, b, 1]
                npn = np.linalg.norm(dp)
                nmn = np.linalg.norm(dm)
                if npn > 1e-12 and nmn > 1e-12:
                    anti[s].append(
                        float(dp @ dm)
                        / (npn * nmn))
med_anti1 = float(np.median(anti[S1])) \
    if anti[S1] else float('nan')
med_anti2 = float(np.median(anti[S2])) \
    if anti[S2] else float('nan')
log('T1 antisym: med cos(d+,d-) s=1: %.4f (n=%d) | '
    's=2: %.4f (n=%d)'
    % (med_anti1, len(anti[S1]), med_anti2,
       len(anti[S2])))

log('=== T2 axis efficiency specificity ===')
mc = np.random.default_rng(SEED_MC_L3)
pair_ratio = []
pair_pools = []
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        if G3[c - 1, b] < DEGEN_NORM:
            continue
        if not ALPHA_OK[c - 1, b, 0, S1]:
            continue
        g = G3[c - 1, b]
        eff_a = ALPHA_N[c - 1, b, 0, S1] / g
        effs = [RAND_EFF[c - 1, b, r]
                for r in range(N_RND_L3)
                if RAND_OK[c - 1, b, r]]
        if len(effs) < 10:
            continue
        pool = np.array([eff_a] + effs)
        med_rest = np.zeros(len(pool))
        for i in range(len(pool)):
            med_rest[i] = np.median(
                np.delete(pool, i))
        v = pool / np.maximum(med_rest, 1e-30)
        pair_pools.append(v)
        pair_ratio.append(v[0])
obs_t2 = float(np.median(pair_ratio)) \
    if pair_ratio else float('nan')
null2 = np.zeros(R_MC)
for it in range(R_MC):
    vals = np.zeros(len(pair_pools))
    for pi, v in enumerate(pair_pools):
        vals[pi] = v[mc.integers(0, len(v))]
    null2[it] = float(np.median(vals))
p_t2 = float(np.mean(null2 >= obs_t2))
log('T2: obs ratio=%.4f null med=%.4f p=%.5f '
    '(pairs=%d)' % (obs_t2,
                    float(np.median(null2)), p_t2,
                    len(pair_ratio)))

log('=== T3 causal reproduction ===')
cos_a = []
cos_null_mat = []
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        if G3[c - 1, b] < DEGEN_NORM:
            continue
        if not ALPHA_OK[c - 1, b, 0, S1]:
            continue
        if pref_norm[c - 1, b] < DEGEN_NORM:
            continue
        dp = ALPHA_D1[c - 1, b, 0]
        npn = np.linalg.norm(dp)
        if npn < 1e-12:
            continue
        cos_a.append(float(dp @ DLG_PREF[c - 1, b])
                     / (npn
                        * pref_norm[c - 1,
                                   b]))
        vals = [RAND_COS[c - 1, b, r]
                for r in range(N_RND_L3)
                if RAND_OK[c - 1, b, r]]
        cos_null_mat.append(vals)
# dirs valid in ALL valid pairs
n_valid = [len(v) for v in cos_null_mat]
dirs_common = []
if cos_a:
    okmask = np.ones(N_RND_L3, dtype=bool)
    for c in (1, 2, 3):
        for b in range(len(BODIES)):
            if G3[c - 1, b] >= DEGEN_NORM \
                    and ALPHA_OK[c - 1, b, 0, S1] \
                    and pref_norm[c - 1, b] \
                    >= DEGEN_NORM:
                okmask &= RAND_OK[c - 1, b]
    dirs_common = list(np.where(okmask)[0])
obs_t3 = float(np.median(cos_a)) \
    if cos_a else float('nan')
null3 = np.zeros(len(dirs_common))
for j, r in enumerate(dirs_common):
    null3[j] = float(np.median(
        [RAND_COS[c - 1, b, r]
         for c in (1, 2, 3)
         for b in range(len(BODIES))
         if G3[c - 1, b] >= DEGEN_NORM
         and ALPHA_OK[c - 1, b, 0, S1]
         and pref_norm[c - 1, b] >= DEGEN_NORM]))
p_t3 = float(np.mean(null3 >= obs_t3)) \
    if len(dirs_common) else float('nan')
log('T3: obs cos=%.4f null med=%.4f p=%.5f '
    '(pairs=%d, dirs=%d)'
    % (obs_t3,
       float(np.median(null3))
       if len(dirs_common) else float('nan'),
       p_t3, len(cos_a), len(dirs_common)))

log('=== T5 L20 control ===')
ratio20_pairs = []
pools20 = []
for b in range(len(BODIES)):
    g = float(GBAR20[b])
    if g < DEGEN_NORM:
        continue
    base_i = idx_of[(0, b)]
    pos = assembled[base_i]['pos']
    ids = assembled[base_i]['ids']
    delta = g * UBAR20
    lg_i, v3p, v20p = forward_inj(
        ids, 20, pos, delta)
    ok, cs, rt = integrity(v20p, V20o[base_i],
                           delta)
    d = lg_i - LGs[base_i]
    eff_u = float(np.linalg.norm(d)) / g
    record(20, 3, b, -1, 1.0,
           float(np.linalg.norm(d)), ok, cs, rt)
    if not ok:
        n_integ_fail += 1
    effs = []
    for r in range(N_RND_L20):
        dr = RND20[b, r] * g
        lg_r, _, v20r = forward_inj(
            ids, 20, pos, dr)
        okr, csr, rtr = integrity(
            v20r, V20o[base_i], dr)
        ndr = float(np.linalg.norm(
            lg_r - LGs[base_i]))
        effs.append(ndr / g)
        record(20, 4, b, -1, 1.0, ndr, okr,
               csr, rtr)
        if not okr:
            n_integ_fail += 1
    pool = np.array([eff_u] + effs)
    med_rest = np.zeros(len(pool))
    for i in range(len(pool)):
        med_rest[i] = np.median(
            np.delete(pool, i))
    pools20.append(pool
                   / np.maximum(med_rest, 1e-30))
    ratio20_pairs.append(pool[0])
obs_t5 = float(np.median(ratio20_pairs)) \
    if ratio20_pairs else float('nan')
mc20 = np.random.default_rng(SEED_MC_L20)
null5 = np.zeros(R_MC)
for it in range(R_MC):
    vals = np.zeros(len(pools20))
    for pi, v in enumerate(pools20):
        vals[pi] = v[mc20.integers(0, len(v))]
    null5[it] = float(np.median(vals))
p_t5 = float(np.mean(null5 >= obs_t5)) \
    if ratio20_pairs else float('nan')
log('T5: L20 ratio=%.4f null med=%.4f p=%.5f '
    '(bodies=%d)' % (obs_t5,
                     float(np.median(null5)),
                     p_t5, len(ratio20_pairs)))

# a105 / a106
rec_iok_arr = np.array(rec_iok, dtype=bool)
a105_ok = bool(np.all(rec_iok_arr)
               and a105_probe_ok)
alpha_s2_max = 0.0
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        for sign in (0, 1):
            for si in (2, 3):
                if ALPHA_OK[c - 1, b, sign, si]:
                    alpha_s2_max = max(
                        alpha_s2_max,
                        ALPHA_N[c - 1, b, sign,
                                si])
a106_ok = bool(alpha_s2_max > A106_GATE)
log('a105 integ_all=%s probe=%.1e | a106 sham '
    'max=%.4f gate=%.2f ok=%s'
    % (bool(np.all(rec_iok_arr)), probe_max,
       alpha_s2_max, A106_GATE, a106_ok))

# ---------- verdict ----------
a_ok = bool(a100_diff == 0.0 and a101_diff == 0.0
            and a102_ok and a103_ok and a104_ok
            and a105_ok and a106_ok)
spec_ok = bool(p_t2 < 0.05)
repro_ok = bool(p_t3 < 0.05)
if spec_ok and repro_ok:
    verdict = 'fieldaxis_causal_readout_qwen'
elif spec_ok:
    verdict = 'fieldaxis_local_logit_qwen'
else:
    verdict = 'fieldaxis_null_qwen'
dose_mono = bool(med_rho == med_rho and med_rho > 0
                 and p_dose < 0.05)
antisym_ok = bool(med_anti1 == med_anti1
                  and med_anti1 <= -0.9)
l20_weaker = bool(obs_t5 == obs_t5
                  and obs_t2 == obs_t2
                  and obs_t5 < obs_t2)

log('=== verdict ===')
log('a100=%r a101=%r a102=%s a103_ok=%s (%.3e,%d) '
    'a104_ok=%s (%.1e/%.1e/%.1e) a105=%s a106=%s'
    % (a100_diff, a101_diff, a102_ok, a103_ok,
       a103_diff, a103_matched, a104_ok, a104_dD,
       a104_dnorm, a104_dmed, a105_ok, a106_ok))
log('T2 p=%.5f | T3 p=%.5f | T5 p=%.5f'
    % (p_t2, p_t3, p_t5))
log('VERDICT=%s anchor_all_ok=%s' % (verdict, a_ok))

elapsed = time.time() - t0

npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    bodies=np.array(BODIES),
    prefixes=np.array(PREFIXES),
    targets=np.array(TARGETS),
    body_idx=np.array(BODY_IDX),
    alpha3=ALPHAS3, ubar3=UBAR3,
    alpha20=ALPHAS20, ubar20=UBAR20,
    rnd_dirs3=RND3, rnd_dirs20=RND20,
    G3=G3, G20=G20,
    dlg_pref_norm=pref_norm,
    alpha_n=ALPHA_N, alpha_ok=ALPHA_OK,
    alpha_d1=ALPHA_D1, alpha_d2=ALPHA_D2,
    ubar_n=UBAR_N, ubar_ok=UBAR_OK,
    ubar_d1=UBAR_D1,
    rand_eff=RAND_EFF, rand_cos=RAND_COS,
    rand_ok=RAND_OK,
    rec_layer=np.array(rec_layer),
    rec_kind=np.array(rec_kind),
    rec_body=np.array(rec_body),
    rec_cond=np.array(rec_cond),
    rec_scale=np.array(rec_scale),
    rec_dlg=np.array(rec_dlg),
    rec_iok=np.array(rec_iok, dtype=bool),
    med_rho=np.float64(med_rho),
    p_dose=np.float64(p_dose),
    med_anti1=np.float64(med_anti1),
    med_anti2=np.float64(med_anti2),
    obs_t2=np.float64(obs_t2),
    null2_med=np.float64(np.median(null2)),
    p_t2=np.float64(p_t2),
    obs_t3=np.float64(obs_t3),
    null3_med=np.float64(np.median(null3))
    if len(dirs_common) else np.float64('nan'),
    p_t3=np.float64(p_t3),
    obs_t5=np.float64(obs_t5),
    null5_med=np.float64(np.median(null5)),
    p_t5=np.float64(p_t5),
    probe_max=np.float64(probe_max),
    alpha_s2_max=np.float64(alpha_s2_max),
    n_integ_fail=np.int64(n_integ_fail),
    a100_diff=np.float64(a100_diff),
    a101_diff=np.float64(a101_diff),
    a102_ok=np.bool_(a102_ok),
    a103_diff=np.float64(a103_diff),
    a103_matched=np.int64(a103_matched),
    a104_dD=np.float64(a104_dD),
    a104_dnorm=np.float64(a104_dnorm),
    a104_dmed=np.float64(a104_dmed),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'run': 'run2 authoritative (run1 T3 '
           'cosine-definition bug: missing '
           '||dlg_pref|| denominator, obs>1 '
           'exposed it; T1/T2/T5 norm-based, '
           'reproduced identically)',
    'anchor_all_ok': a_ok,
    'anchors': {
        'a100_dup_prefill_bit': a100_diff,
        'a101_dup_all_bit': a101_diff,
        'a102_source_seals': a102_ok,
        'a103_cross_phase_bit': a103_diff,
        'a103_matched': a103_matched,
        'a104_dD_bit': a104_dD,
        'a104_dnorm_bit': a104_dnorm,
        'a104_dmed': a104_dmed,
        'a104_gate': A104_GATE,
        'a105_integrity_all': a105_ok,
        'a105_n_integ_fail': n_integ_fail,
        'a105_probe_nontarget_max': probe_max,
        'a106_sham_max': alpha_s2_max,
        'a106_gate': A106_GATE,
    },
    'T1_dose': {'med_rho': med_rho,
                'n_pairs': n_rho, 'k_pos': k_pos,
                'p_binom': p_dose},
    'T1_antisym': {'med_cos_s1': med_anti1,
                   'med_cos_s2': med_anti2,
                   'n_pairs_s1': len(anti[S1]),
                   'n_pairs_s2': len(anti[S2])},
    'T2_specificity': {'obs_ratio': obs_t2,
                       'null_med':
                       float(np.median(null2)),
                       'p_t2': p_t2,
                       'n_pairs': len(pair_ratio)},
    'T3_reproduction': {'obs_cos': obs_t3,
                        'null_med':
                        float(np.median(null3))
                        if len(dirs_common)
                        else None,
                        'p_t3': p_t3,
                        'n_pairs': len(cos_a),
                        'n_dirs': len(dirs_common)},
    'T5_layer20': {'obs_ratio': obs_t5,
                   'null_med':
                   float(np.median(null5)),
                   'p_t5': p_t5,
                   'n_bodies': len(ratio20_pairs)},
    'flags': {'dose_mono': dose_mono,
              'antisym_ok': antisym_ok,
              'l20_weaker': l20_weaker,
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
    'anchor_all_ok': a_ok,
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
