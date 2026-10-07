# -*- coding: utf-8 -*-
# Phase 3046 - Omega-P43: K-side field injection
# (routing-channel causal test, fp32).
# RUN HISTORY (all registered in corrections):
# run1 crashed pre-anchor: NEW-bodies assembly
# else-branch used BODIES[bi] (transcription slip).
# run2 crashed mid-T4: attention tensors left on
# cuda. run3 completed but a118 failed 208/208 and
# T5 dose was flat -> probe (gpt5_temp/probe3046c)
# root-caused: (i) make_inj captured the k_proj
# slice BEFORE adding delta (dpre = max|delta|
# 6.84, not ~0); (ii) Qwen3 has qk-norm: k_norm
# (weight mean 1.7633) sits between the k_proj
# output and RoPE; the pre-norm kv7 key has norm
# 1.7076 while the post-norm+RoPE key has norm
# 20.16, so the post-norm displacement scale
# gK = 15.7 injected pre-norm OVERDRIVES the key
# ~9x - attention saturates (T5 flat 0.611 across
# 16x dose), direction effects are washed out
# (T2 1.00x), and the post-norm ratio check
# (1.7325) is architecturally wrong. The T1
# post-norm K field itself is REAL and huge
# (L3 0.7842 = 13x null, L20 0.3350 = 5.6x,
# both p=0.00000, 84 pairs).
# RUN4 REDESIGN (pre-norm injection at the
# architecture-correct scale): capture pre-norm
# kv7 keys KPpre for all 48 prompts during
# extraction; the intervention point is the
# k_proj output (pre k_norm); natural scale and
# axes defined PRE-NORM: dpre(c,b) = KPpre(c,b) -
# KPpre(0,b); gpre_b = mean_c ||dpre(c,b)||;
# ubarPre = unit(mean_c unit(mean_b dpre));
# (T1pre) same-prefix cross-body alignment of
# dpre (primary field test; T1post kept as
# descriptive measurement-level result);
# (T2) inject gpre_b * ubarPre vs 12 fresh random
# unit dirs x gpre_b; eff = ||dlg||/gpre_b; stat =
# med over bodies of eff(axis)/med(eff_rand);
# null = within-pool label permutation (R=20000)
# on RAW efficiencies;
# (T3a) axis replay: cos(response, t_bc), t_bc =
# LG(c,b) - LG(0,b); pools per (b,c) = [cos_axis]
# + [cos_rand d=1..12]; within-pool permutation;
# (T3b) EXACT replay: inject the exact pre-norm
# displacement dpre(c,b) (k_norm(x0+d) = k_norm(xc)
# holds pointwise) - the single-layer faithful
# replay of the prefix key change; vs 12 random
# dirs matched to ||dpre(c,b)||; stat = med over
# (b,c) of (cos_exact - med(cos_rand)); frac =
# ||resp||/||t_bc|| descriptive;
# (T4) routing readout (descriptive): body 0,
# output_attentions, dAttn = sum over q heads
# 28-31 of L1(attn[-1 row] inj - base) at the
# injected layer; Spearman(dAttn, ||dlg||);
# (T5) dose ladder: gpre x {0.25,0.5,1,2,4} x
# (ubar + 3 rand) x 2 layers.
# verdict_tree: fieldK = (p_T1pre_L3<0.05 or
# p_T1pre_L20<0.05); causK = (p_T2_L3<0.05 or
# p_T2_L20<0.05); routK = (p_T3a<0.05 at either
# layer) or (p_T3b<0.05 at either layer);
# fieldK AND causK AND routK ->
# kfield_causal_route_qwen; fieldK AND causK ->
# kfield_causal_local_qwen; fieldK AND routK ->
# kfield_route_specific_qwen; fieldK ->
# kfield_correlational_qwen; else ->
# kfield_null_qwen; single branch.
# anchors: a114 duplicate prefill prompt0 (lg AND
# K/V full rows AND KPpre at L3/L20) bit 0.0;
# a115 full duplicate extraction all 48 prompts
# (probs, lg, K full rows, KPpre) bit 0.0; a116
# source seals 3037..3045; a117 cross-precision
# post-norm K base: cos vs 3037 bf16 npz K3/K20
# (matched 8/8 per layer) >= 0.999; a118v2
# injection integrity: dpre == delta exactly
# (capture AFTER modification), non-target past
# keys bit 0.0, post-norm diff norm in (0, 60]
# (loose qk-norm-aware bound; measured ratio
# recorded descriptive), zero failures; a119
# chain-entry gates: max ||dlg|| over T2
# injections >= 0.05 AND max dAttn over T4 >=
# 1e-6.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3046
NAME = 'omega_p43_kfield_injection_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 7
QH = (28, 29, 30, 31)
LAYERS_EX = (3, 20)
N_RND_T2 = 12
N_RND_T3B = 12
N_RND_T4 = 6
N_RND_T5 = 3
R_MC = 20000
SEED_RND_L3 = 9800
SEED_RND_L20 = 9801
SEED_RND_T3B3 = 9804
SEED_RND_T3B20 = 9805
SEED_RND_T5 = 9802
SEED_RND_T4 = 9803
SEED_MC_T1P_3 = 9818
SEED_MC_T1P_20 = 9819
SEED_MC_T1POST_3 = 9820
SEED_MC_T1POST_20 = 9821
SEED_MC_T2_3 = 9812
SEED_MC_T2_20 = 9813
SEED_MC_T3A_3 = 9814
SEED_MC_T3A_20 = 9815
SEED_MC_T3B_3 = 9816
SEED_MC_T3B_20 = 9817
SEED_MAIN = 3009
POST_BOUND = 60.0
COS_GATE = 0.999
DLG_GATE = 0.05
ATTN_GATE = 1e-6
T5_MULT = (0.25, 0.5, 1.0, 2.0, 4.0)

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
            '3009); 48 prompts verbatim 3042-3045 '
            'bank; run4 pre-norm redesign: injection '
            'into k_proj output (kv7 slice, pre '
            'k_norm) at the pre-norm natural scale '
            'gpre and along the pre-norm field axis '
            'ubarPre; layers (3,20); K PRIMARY',
    'question': '3046 A main line, run4: does the '
                'prefix field act through K '
                '(attention routing) when injected '
                'at the architecture-correct pre-'
                'norm scale? (i) does the pre-norm '
                'key carry a same-prefix cross-body '
                'displacement field (T1pre, '
                'primary); (ii) is the pre-norm '
                'field axis more efficient than '
                'size-matched random directions at '
                'moving logits (T2); (iii) does '
                'axis injection (T3a) or the EXACT '
                'per-condition pre-norm key '
                'displacement (T3b, faithful '
                'single-layer replay) reproduce the '
                'prefix logit displacement; (iv) '
                'attention routing readout (T4, '
                'descriptive); (v) dose response '
                'spanning subnatural to overdriven '
                '(T5, descriptive)',
    'run3_findings': 'a118 failed 208/208 and T5 '
                     'dose flat; probe: (i) make_inj '
                     'capture-order bug (mod recorded '
                     'pre-modification, dpre = '
                     'max|delta| 6.84); (ii) qk-norm: '
                     'k_norm weight mean 1.7633, pre-'
                     'norm kv7 key norm 1.7076 vs '
                     'post-norm+RoPE 20.16 -> the '
                     'post-norm displacement scale '
                     'gK 15.7 injected pre-norm '
                     'overdrives ~9x, saturating '
                     'attention (T5 flat 0.611 across '
                     '16x dose) and washing out '
                     'direction effects (T2 1.00x); '
                     '(iii) the post-norm T1 K field '
                     'is real: L3 0.7842 (13x null) '
                     'L20 0.3350 (5.6x), both '
                     'p=0.00000; (iv) ratio check '
                     '[0.9,1.1] architecturally '
                     'wrong (measured 1.7325)',
    'T1pre': 'dpre(c,b) = KPpre(c,b) - KPpre(0,b) '
             'at the target position (pre-norm), '
             'layers (3,20); stat = med |cos| over '
             'same-prefix cross-body pairs (84); '
             'null = Gaussian 128-dim unit pairs, '
             'R=%d (seeds %d/%d); spec iff p<0.05'
             % (R_MC, SEED_MC_T1P_3, SEED_MC_T1P_20),
    'T1post': 'same statistic on post-norm+RoPE '
              'displacements (measurement level, '
              'descriptive; seeds %d/%d)'
              % (SEED_MC_T1POST_3,
                 SEED_MC_T1POST_20),
    'T2': 'inject gpre_b * ubarPre into k_proj '
          'output kv7 slice at the target position '
          'of the base-condition prompt; gpre_b = '
          'mean_c ||dpre(c,b)||; vs N_RND_T2=12 '
          'fresh random unit dirs x gpre_b (seeds '
          '%d/%d); eff = ||dlg||/gpre_b; stat = med '
          'over bodies of eff(axis)/med(eff_rand); '
          'null = within-pool label permutation '
          '(R=%d, seeds %d/%d) on RAW efficiencies; '
          'spec iff p<0.05'
          % (SEED_RND_L3, SEED_RND_L20, R_MC,
             SEED_MC_T2_3, SEED_MC_T2_20),
    'T3a': 'cos(response, t_bc), t_bc = LG(c,b) - '
           'LG(0,b); pools per (b,c) = [cos_axis] '
           '+ [cos_rand d=1..12] (responses shared '
           'with T2 injections); stat = med over '
           'pools of (cos_axis - med(cos_rand)); '
           'null = within-pool index permutation '
           '(R=%d, seeds %d/%d); spec iff p<0.05'
           % (R_MC, SEED_MC_T3A_3, SEED_MC_T3A_20),
    'T3b': 'EXACT replay: inject dpre(c,b) itself '
           '(k_norm(x0+d) = k_norm(xc) pointwise) '
           'at the base prompt; vs N_RND_T3B=12 '
           'random unit dirs matched to '
           '||dpre(c,b)|| (seeds %d/%d); stat = '
           'med over 24 (b,c) of (cos_exact - '
           'med(cos_rand)); null = within-pool '
           'permutation (R=%d, seeds %d/%d); frac '
           '= ||resp||/||t_bc|| descriptive; spec '
           'iff p<0.05'
           % (SEED_RND_T3B3, SEED_RND_T3B20, R_MC,
              SEED_MC_T3B_3, SEED_MC_T3B_20),
    'T4': 'body 0, output_attentions=True, eager; '
          'dAttn = sum over q heads 28-31 of '
          'L1(attn[injected layer][0,qh,-1,:] inj - '
          'base) at pre-norm natural scale; axis + '
          'N_RND_T4=6 rand (seed %d); med axis vs '
          'med rand; Spearman(dAttn, ||dlg||); '
          'descriptive' % SEED_RND_T4,
    'T5': 'body 0 dose ladder: gpre x '
          '{0.25,0.5,1,2,4} x (ubar + 3 rand, seed '
          '%d) x 2 layers; descriptive'
          % SEED_RND_T5,
    'verdict_tree': 'fieldK = (p_T1pre_L3<0.05 or '
                    'p_T1pre_L20<0.05); causK = '
                    '(p_T2_L3<0.05 or p_T2_L20<'
                    '0.05); routK = (p_T3a_L3<0.05 '
                    'or p_T3a_L20<0.05 or p_T3b_L3<'
                    '0.05 or p_T3b_L20<0.05); '
                    'fieldK AND causK AND routK -> '
                    'kfield_causal_route_qwen; '
                    'fieldK AND causK -> '
                    'kfield_causal_local_qwen; '
                    'fieldK AND routK -> '
                    'kfield_route_specific_qwen; '
                    'fieldK -> '
                    'kfield_correlational_qwen; '
                    'else -> kfield_null_qwen; '
                    'single branch',
    'anchors': 'a114 duplicate prefill prompt0 (lg '
               'AND K/V full rows AND KPpre L3/L20) '
               'bit 0.0; a115 full duplicate '
               'extraction all 48 prompts (probs, '
               'lg, K full rows, KPpre) bit 0.0; '
               'a116 source seals 3037..3045; a117 '
               'cross-precision post-norm K base '
               'cos vs 3037 bf16 K3/K20 >= 0.999 '
               '(16 matched); a118v2 integrity: '
               'dpre == delta exactly (capture '
               'AFTER modification), non-target '
               'past keys bit 0.0, post-norm diff '
               'norm in (0,60] (qk-norm-aware '
               'loose bound, ratio descriptive), '
               'zero failures; a119 chain-entry: '
               'max ||dlg|| over T2 >= 0.05 AND '
               'max dAttn over T4 >= 1e-6',
    'control': 'unit-norm random directions (fresh '
               'seeds fp32; norm-matched for T3b); '
               'within-pool label permutation; '
               'Gaussian pair null for T1; no '
               'other intervention',
    'statistics_discipline': 'obs and null on the '
                             'SAME scale (raw '
                             'efficiencies and raw '
                             'cos differences); nulls '
                             'never on intervened '
                             'quantities of the '
                             'treated arm; arrays '
                             'pre-initialized; '
                             'verdict in one branch',
    'corrections': 'run4 crashed in T5: forward_inj '
                   'was reduced to a 2-value return in the '
                   'run4 rewrite but the T5 unpacks kept 3 '
                   'values; T1pre/T1post/T2/T3a/T3b/T4 were '
                   'observed in run4 and reproduce under '
                   'frozen seeds; unpacks fixed; run5 '
                   'authoritative; run1 crashed pre-anchor at '
                   'prompt assembly: the NEW-bodies '
                   'loop else-branch used BODIES[bi] '
                   '(transcription slip; 3045 line '
                   '409 correctly reads '
                   'NEW_BODIES[bi]); no anchor or '
                   'statistic observed; run2 crashed '
                   'mid-T4: attention tensors left '
                   'on cuda (T1/T2/T3 observed, '
                   'reproduce under frozen seeds); '
                   'run3 completed but a118 failed '
                   '208/208 and T5 dose flat; probe '
                   'root-caused the capture-order '
                   'bug AND the qk-norm overdrive '
                   '(see run3_findings); run4 = '
                   'pre-norm redesign at the '
                   'architecture-correct scale with '
                   'T3b exact replay added; verdict '
                   'tree extended (routK includes '
                   'T3b)',
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
HDIM = 128
assert int(model.config.num_key_value_heads) == 8
assert int(model.config.num_attention_heads) == 32
assert hasattr(layers[3].self_attn, 'k_norm')
log('model loaded fp32 (qk-norm confirmed)')

state = {li: {'on': False, 'pos': -1, 'delta': None}
         for li in LAYERS_EX}
cap = {li: {'rec': False, 'pos': -1, 'orig': None,
            'mod': None} for li in LAYERS_EX}
SL = slice(KV_HEAD * HDIM, (KV_HEAD + 1) * HDIM)


def make_cap(li):
    def h(module, inp, out):
        cs = cap[li]
        if cs['rec']:
            cs['orig'] = out[0, cs['pos'], SL] \
                .detach().clone()
        return out
    return h


def make_inj(li):
    def h(module, inp, out):
        st = state[li]
        if st['on']:
            out[0, st['pos'], SL] += st['delta']
            cs = cap[li]
            if cs['rec']:
                cs['mod'] = out[0, st['pos'], SL] \
                    .detach().clone()
        return out
    return h


for li in LAYERS_EX:
    layers[li].self_attn.k_proj.register_forward_hook(
        make_cap(li))
    layers[li].self_attn.k_proj.register_forward_hook(
        make_inj(li))


def forward_full(ids, pos):
    for lj in LAYERS_EX:
        state[lj]['on'] = False
        cap[lj]['rec'] = True
        cap[lj]['pos'] = pos
        cap[lj]['orig'] = None
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    for lj in LAYERS_EX:
        cap[lj]['rec'] = False
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
    for lj in LAYERS_EX:
        state[lj]['on'] = False
        cap[lj]['rec'] = False
        cap[lj]['orig'] = None
        cap[lj]['mod'] = None
    d = torch.tensor(np.asarray(delta128),
                     dtype=torch.float32,
                     device='cuda')
    state[li].update(on=True, pos=pos, delta=d)
    cap[li]['rec'] = True
    cap[li]['pos'] = pos
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    state[li]['on'] = False
    cap[li]['rec'] = False
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    past = out.past_key_values
    kinj = past.layers[li].keys[0, KV_HEAD] \
        .detach().double().cpu().numpy()
    return lg, kinj


def forward_attn(ids):
    for lj in LAYERS_EX:
        state[lj]['on'] = False
        cap[lj]['rec'] = False
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=False,
                    output_attentions=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    attns = [a.detach().double().cpu()
             .numpy() for a in out.attentions]
    return lg, attns


def forward_inj_attn(ids, li, pos, delta128):
    for lj in LAYERS_EX:
        state[lj]['on'] = False
        cap[lj]['rec'] = False
    d = torch.tensor(np.asarray(delta128),
                     dtype=torch.float32,
                     device='cuda')
    state[li].update(on=True, pos=pos, delta=d)
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=False,
                    output_attentions=True)
    state[li]['on'] = False
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    attns = [a.detach().double().cpu()
             .numpy() for a in out.attentions]
    return lg, attns


# ---------- chain sources ----------
z37 = np.load(os.path.join(
    BASE, 'phase3037',
    'omega_p34_kv_situational_specificity_qwen',
    'omega_p34_kv_situational_specificity_qwen.npz'),
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

# a114: duplicate prefill of prompt0
p_b0, lg_b0, kv_b0 = forward_full(
    assembled[0]['ids'], assembled[0]['pos'])
pre_b0 = {li: cap[li]['orig'].double().cpu().numpy()
          for li in LAYERS_EX}
p_d0, lg_d0, kv_d0 = forward_full(
    assembled[0]['ids'], assembled[0]['pos'])
pre_d0 = {li: cap[li]['orig'].double().cpu().numpy()
          for li in LAYERS_EX}
a114_diff = float(np.max(np.abs(lg_b0 - lg_d0)))
for li in LAYERS_EX:
    for a, b in zip(kv_b0[li], kv_d0[li]):
        a114_diff = max(a114_diff, float(
            np.max(np.abs(a - b))))
    a114_diff = max(a114_diff, float(
        np.max(np.abs(pre_b0[li] - pre_d0[li]))))

# ---------- main extraction ----------
Ps = [None] * n_pr
LGs = [None] * n_pr
Kfull = {li: [None] * n_pr for li in LAYERS_EX}
KPpre = {li: np.zeros((n_pr, HDIM))
         for li in LAYERS_EX}
for i in range(n_pr):
    p, lg, kv = forward_full(
        assembled[i]['ids'], assembled[i]['pos'])
    Ps[i] = p
    LGs[i] = lg
    for li in LAYERS_EX:
        Kfull[li][i] = kv[li][0]
        KPpre[li][i] = cap[li]['orig'].double() \
            .cpu().numpy()

# a115: full duplicate extraction
a115_diff = 0.0
for i in range(n_pr):
    p2, lg2, kv2 = forward_full(
        assembled[i]['ids'], assembled[i]['pos'])
    a115_diff = max(a115_diff, float(
        np.max(np.abs(Ps[i] - p2))))
    a115_diff = max(a115_diff, float(
        np.max(np.abs(LGs[i] - lg2))))
    for li in LAYERS_EX:
        a115_diff = max(a115_diff, float(
            np.max(np.abs(Kfull[li][i] - kv2[li][0]))))
        a115_diff = max(a115_diff, float(np.max(
            np.abs(KPpre[li][i]
                   - cap[li]['orig'].double()
                   .cpu().numpy()))))
a115_diff = float(a115_diff)
log('a114=%.3e a115=%.3e' % (a114_diff, a115_diff))

# a116: source seals
a116_detail = []
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
               'qwen'),
        (3045, 'omega_p42_l20_axis_anatomy_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a116_detail.append(bool(
        s == sealj['result_sha256_8']))
a116_ok = bool(a116_detail) and all(a116_detail)

# K base at target position (post-norm+RoPE)
Kbase = {li: np.zeros((n_pr, HDIM))
         for li in LAYERS_EX}
for i in range(n_pr):
    pos = assembled[i]['pos']
    for li in LAYERS_EX:
        Kbase[li][i] = Kfull[li][i][pos]

# a117: cross-precision post-norm K base vs 3037
w37 = [str(x) for x in z37['occ_word']]
pr37 = [int(x) for x in z37['occ_prompt']]
po37 = [int(x) for x in z37['occ_pos']]
a117_min = 1.0
a117_matched = 0
for li, refn in ((3, 'K3'), (20, 'K20')):
    refarr = z37[refn]
    ref = {}
    for j in range(len(w37)):
        ref[(w37[j], pr37[j], po37[j])] = refarr[j]
    for i in range(n_old):
        if assembled[i]['cond'] != 0:
            continue
        bi = assembled[i]['body']
        key = (TARGETS[bi], int(BODY_IDX[bi]),
               int(assembled[i]['pos']))
        if key not in ref:
            continue
        k32 = Kbase[li][i]
        kb = ref[key]
        cs = float(k32 @ kb) / (
            np.linalg.norm(k32)
            * np.linalg.norm(kb))
        a117_min = min(a117_min, cs)
        a117_matched += 1
a117_min = float(a117_min)
a117_matched = int(a117_matched)
a117_ok = bool(a117_matched == 2 * len(BODIES)
               and a117_min >= COS_GATE)
log('a117 matched=%d min cos=%.8f ok=%s'
    % (a117_matched, a117_min, a117_ok))

# ---------- field displacements ----------
cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)


def unit(v):
    nn = np.linalg.norm(v)
    return v / nn if nn > 0 else v


def displacements(Base):
    D = np.zeros((24, HDIM))
    for k in range(24):
        ic = idx_of[(int(cidx[k]), int(bidx[k]),
                     False)]
        ib = idx_of[(0, int(bidx[k]), False)]
        D[k] = Base[ic] - Base[ib]
    return D


D3K = displacements(Kbase[3])
D20K = displacements(Kbase[20])
DPRE3 = displacements(KPpre[3])
DPRE20 = displacements(KPpre[20])


def alphas_from(D):
    out = []
    for c in (1, 2, 3):
        m = cidx == c
        a = D[m].mean(axis=0)
        out.append(unit(a))
    return np.stack(out)


ALPHAK3 = alphas_from(D3K)
ALPHAK20 = alphas_from(D20K)
UBARK3 = unit(ALPHAK3.mean(axis=0))
UBARK20 = unit(ALPHAK20.mean(axis=0))
ALPHAPRE3 = alphas_from(DPRE3)
ALPHAPRE20 = alphas_from(DPRE20)
UBARPRE3 = unit(ALPHAPRE3.mean(axis=0))
UBARPRE20 = unit(ALPHAPRE20.mean(axis=0))
GK3 = np.zeros((3, len(BODIES)))
GK20 = np.zeros((3, len(BODIES)))
GPRE3 = np.zeros((3, len(BODIES)))
GPRE20 = np.zeros((3, len(BODIES)))
for c in (1, 2, 3):
    for b in range(len(BODIES)):
        m = (cidx == c) & (bidx == b)
        GK3[c - 1, b] = float(
            np.linalg.norm(D3K[m][0]))
        GK20[c - 1, b] = float(
            np.linalg.norm(D20K[m][0]))
        GPRE3[c - 1, b] = float(
            np.linalg.norm(DPRE3[m][0]))
        GPRE20[c - 1, b] = float(
            np.linalg.norm(DPRE20[m][0]))
GBARK3 = GK3.mean(axis=0)
GBARK20 = GK20.mean(axis=0)
GBARPRE3 = GPRE3.mean(axis=0)
GBARPRE20 = GPRE20.mean(axis=0)
log('post-norm K disp norms: GBARK3 med=%.4f '
    'GBARK20 med=%.4f' % (float(np.median(GBARK3)),
                          float(np.median(GBARK20))))
log('pre-norm K disp norms: GBARPRE3 med=%.4f '
    'GBARPRE20 med=%.4f'
    % (float(np.median(GBARPRE3)),
       float(np.median(GBARPRE20))))
log('pre-norm key norms: L3 med=%.4f L20 med=%.4f'
    % (float(np.median(np.linalg.norm(KPpre[3],
                                      axis=1))),
       float(np.median(np.linalg.norm(KPpre[20],
                                      axis=1)))))

# ---------- Gaussian pair null ----------
def gauss_null_medcos(n_pairs, seed):
    rng = np.random.default_rng(seed)
    null = np.zeros(R_MC)
    for it in range(R_MC):
        x = rng.standard_normal((2 * n_pairs, HDIM))
        x = x / np.linalg.norm(x, axis=1)[:, None]
        cs = np.sum(x[0::2] * x[1::2], axis=1)
        null[it] = float(np.median(np.abs(cs)))
    return null


def t1_stat(D):
    pairs = []
    for c in (1, 2, 3):
        rows = np.where(cidx == c)[0]
        for ai in range(len(rows)):
            for bj in range(ai + 1, len(rows)):
                u = D[rows[ai]]
                v = D[rows[bj]]
                cs = float(u @ v) / (
                    np.linalg.norm(u)
                    * np.linalg.norm(v))
                pairs.append(abs(cs))
    return float(np.median(pairs)), len(pairs)


# ---------- T1pre (primary) ----------
log('=== T1pre pre-norm field ===')
obs_t1p_3, npair = t1_stat(DPRE3)
null1p_3 = gauss_null_medcos(npair, SEED_MC_T1P_3)
p_t1p_3 = float(np.mean(null1p_3 >= obs_t1p_3))
obs_t1p_20, _ = t1_stat(DPRE20)
null1p_20 = gauss_null_medcos(npair, SEED_MC_T1P_20)
p_t1p_20 = float(np.mean(null1p_20 >= obs_t1p_20))
log('T1pre: L3 obs=%.4f null=%.4f p=%.5f | L20 '
    'obs=%.4f null=%.4f p=%.5f (pairs=%d)'
    % (obs_t1p_3, float(np.median(null1p_3)),
       p_t1p_3, obs_t1p_20,
       float(np.median(null1p_20)), p_t1p_20,
       npair))

# ---------- T1post (descriptive) ----------
log('=== T1post post-norm field (descriptive) ===')
obs_t1o_3, _ = t1_stat(D3K)
null1o_3 = gauss_null_medcos(npair,
                             SEED_MC_T1POST_3)
p_t1o_3 = float(np.mean(null1o_3 >= obs_t1o_3))
obs_t1o_20, _ = t1_stat(D20K)
null1o_20 = gauss_null_medcos(npair,
                              SEED_MC_T1POST_20)
p_t1o_20 = float(np.mean(null1o_20 >= obs_t1o_20))
log('T1post: L3 obs=%.4f p=%.5f | L20 obs=%.4f '
    'p=%.5f' % (obs_t1o_3, p_t1o_3, obs_t1o_20,
                p_t1o_20))

# ---------- integrity bookkeeping ----------
n_integ_fail = 0
rec_kind = []
rec_body = []
rec_dlg = []
max_dlg_t2 = 0.0
ratio_sum = 0.0
ratio_n = 0


def check_integ(kinj, li, i, pos, delta):
    global n_integ_fail, ratio_sum, ratio_n
    ok = True
    cs = cap[li]
    if cs['orig'] is None or cs['mod'] is None:
        ok = False
        dpre = float('nan')
    else:
        orig = cs['orig'].double().cpu().numpy()
        mod = cs['mod'].double().cpu().numpy()
        dpre = float(np.max(np.abs(
            mod - orig - delta)))
        if dpre > 1e-5:
            ok = False
    kb = Kfull[li][i]
    mask = np.ones(kb.shape[0], dtype=bool)
    mask[pos] = False
    dnt = float(np.max(np.abs(
        kinj[mask] - kb[mask]))) if mask.any() \
        else 0.0
    if dnt != 0.0:
        ok = False
    nd = float(np.linalg.norm(
        kinj[pos] - kb[pos]))
    if not (0.0 < nd <= POST_BOUND):
        ok = False
    ratio = nd / max(float(np.linalg.norm(delta)),
                     1e-30)
    ratio_sum += ratio
    ratio_n += 1
    if not ok:
        n_integ_fail += 1
    return ok, dpre, dnt, ratio


# ---------- T2 + T3a injections (pre-norm) ----------
log('=== T2/T3a pre-norm axis injections ===')
EFF_A = {3: {}, 20: {}}
EFF_R = {3: {}, 20: {}}
COS_A = {3: np.zeros(24), 20: np.zeros(24)}
COS_R = {3: np.zeros((8, 3, N_RND_T2)),
         20: np.zeros((8, 3, N_RND_T2))}
t_targets = {}
for b in range(len(BODIES)):
    for c in (1, 2, 3):
        ic = idx_of[(c, b, False)]
        ib = idx_of[(0, b, False)]
        t_targets[(b, c)] = LGs[ic] - LGs[ib]

for b in range(len(BODIES)):
    base_i = idx_of[(0, b, False)]
    pos = assembled[base_i]['pos']
    ids = assembled[base_i]['ids']
    for li, gvec, axis, seed in (
            (3, GBARPRE3, UBARPRE3, SEED_RND_L3),
            (20, GBARPRE20, UBARPRE20,
             SEED_RND_L20)):
        g = float(gvec[b])
        if g < 1e-9:
            continue
        dlg_a, kinj = forward_inj(
            ids, li, pos, g * axis)
        ok, dpre, dnt, ratio = check_integ(
            kinj, li, base_i, pos, g * axis)
        na = float(np.linalg.norm(
            dlg_a - LGs[base_i]))
        rec_kind.append(0 if li == 3 else 1)
        rec_body.append(b)
        rec_dlg.append(na)
        max_dlg_t2 = max(max_dlg_t2, na)
        EFF_A[li][b] = na / g
        for c in (1, 2, 3):
            t = t_targets[(b, c)]
            r = dlg_a - LGs[base_i]
            COS_A[li][b * 3 + (c - 1)] = float(
                r @ t) / (np.linalg.norm(r)
                          * np.linalg.norm(t)) \
                if np.linalg.norm(r) > 1e-12 \
                and np.linalg.norm(t) > 1e-12 \
                else 0.0
        rng = np.random.default_rng(seed + b)
        R = rng.standard_normal((N_RND_T2, HDIM))
        R = R / np.linalg.norm(R, axis=1)[:, None]
        effs = []
        for rd in range(N_RND_T2):
            dr = R[rd] * g
            dlg_r, kinjr = forward_inj(
                ids, li, pos, dr)
            okr, _, _, _ = check_integ(
                kinjr, li, base_i, pos, dr)
            nr = float(np.linalg.norm(
                dlg_r - LGs[base_i]))
            rec_kind.append(2 if li == 3 else 3)
            rec_body.append(b)
            rec_dlg.append(nr)
            max_dlg_t2 = max(max_dlg_t2, nr)
            effs.append(nr / g)
            rr = dlg_r - LGs[base_i]
            for c in (1, 2, 3):
                t = t_targets[(b, c)]
                COS_R[li][b, c - 1, rd] = float(
                    rr @ t) / (np.linalg.norm(rr)
                               * np.linalg.norm(t)) \
                    if np.linalg.norm(rr) > 1e-12 \
                    and np.linalg.norm(t) > 1e-12 \
                    else 0.0
        EFF_R[li][b] = np.array(effs)
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


def mc_med_diff(pools, seed):
    mc = np.random.default_rng(seed)
    obs = float(np.median(
        [p[0] - float(np.median(p[1:]))
         for p in pools]))
    null = np.zeros(R_MC)
    for it in range(R_MC):
        vals = np.zeros(len(pools))
        for pi, p in enumerate(pools):
            i = mc.integers(0, len(p))
            rest = np.delete(p, i)
            vals[pi] = p[i] - float(
                np.median(rest))
        null[it] = float(np.median(vals))
    p = float(np.mean(null >= obs))
    return obs, float(np.median(null)), p


pools2_3 = []
pools2_20 = []
for b in range(len(BODIES)):
    if b in EFF_A[3] and b in EFF_R[3]:
        pools2_3.append(np.concatenate(
            ([EFF_A[3][b]], EFF_R[3][b])))
    if b in EFF_A[20] and b in EFF_R[20]:
        pools2_20.append(np.concatenate(
            ([EFF_A[20][b]], EFF_R[20][b])))
obs_t2_3, nul2_3, p_t2_3 = mc_med_ratio(
    pools2_3, SEED_MC_T2_3)
obs_t2_20, nul2_20, p_t2_20 = mc_med_ratio(
    pools2_20, SEED_MC_T2_20)
log('T2: L3 obs=%.4f null=%.4f p=%.5f | L20 '
    'obs=%.4f null=%.4f p=%.5f'
    % (obs_t2_3, nul2_3, p_t2_3,
       obs_t2_20, nul2_20, p_t2_20))

pools3a_3 = []
pools3a_20 = []
for b in range(len(BODIES)):
    for c in (1, 2, 3):
        k = b * 3 + (c - 1)
        pools3a_3.append(np.concatenate(
            ([COS_A[3][k]], COS_R[3][b, c - 1])))
        pools3a_20.append(np.concatenate(
            ([COS_A[20][k]], COS_R[20][b, c - 1])))
obs_t3a_3, nul3a_3, p_t3a_3 = mc_med_diff(
    pools3a_3, SEED_MC_T3A_3)
obs_t3a_20, nul3a_20, p_t3a_20 = mc_med_diff(
    pools3a_20, SEED_MC_T3A_20)
log('T3a: L3 obs=%.4f null=%.4f p=%.5f | L20 '
    'obs=%.4f null=%.4f p=%.5f'
    % (obs_t3a_3, nul3a_3, p_t3a_3,
       obs_t3a_20, nul3a_20, p_t3a_20))

# ---------- T3b: exact replay ----------
log('=== T3b exact pre-norm replay ===')
COS_E = {3: np.zeros(24), 20: np.zeros(24)}
COS_ER = {3: np.zeros((24, N_RND_T3B)),
          20: np.zeros((24, N_RND_T3B))}
FRAC_E = {3: np.zeros(24), 20: np.zeros(24)}
for li, seedbase in ((3, SEED_RND_T3B3),
                     (20, SEED_RND_T3B20)):
    for k in range(24):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b, False)]
        pos = assembled[base_i]['pos']
        ids = assembled[base_i]['ids']
        d = DPRE3[k] if li == 3 else DPRE20[k]
        nd_ = float(np.linalg.norm(d))
        dlg_e, kinje = forward_inj(ids, li, pos, d)
        oke, _, _, _ = check_integ(
            kinje, li, base_i, pos, d)
        t = t_targets[(b, c)]
        r = dlg_e - LGs[base_i]
        nr = float(np.linalg.norm(r))
        nt = float(np.linalg.norm(t))
        cs_e = float(r @ t) / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
        COS_E[li][k] = cs_e
        FRAC_E[li][k] = nr / nt if nt > 1e-12 \
            else float('nan')
        rec_kind.append(12 if li == 3 else 13)
        rec_body.append(b)
        rec_dlg.append(nr)
        rng = np.random.default_rng(
            seedbase + k)
        R = rng.standard_normal((N_RND_T3B, HDIM))
        R = R / np.linalg.norm(R, axis=1)[:, None]
        for rd in range(N_RND_T3B):
            dlg_r, kinjr = forward_inj(
                ids, li, pos, R[rd] * nd_)
            okr, _, _, _ = check_integ(
                kinjr, li, base_i, pos,
                R[rd] * nd_)
            rr = dlg_r - LGs[base_i]
            nrr = float(np.linalg.norm(rr))
            cs_r = float(rr @ t) / (nrr * nt) \
                if nrr > 1e-12 and nt > 1e-12 \
                else 0.0
            COS_ER[li][k, rd] = cs_r
            rec_kind.append(14 if li == 3 else 15)
            rec_body.append(b)
            rec_dlg.append(nrr)
    log('T3b L%d done' % li)

pools3b_3 = []
pools3b_20 = []
for k in range(24):
    pools3b_3.append(np.concatenate(
        ([COS_E[3][k]], COS_ER[3][k])))
    pools3b_20.append(np.concatenate(
        ([COS_E[20][k]], COS_ER[20][k])))
obs_t3b_3, nul3b_3, p_t3b_3 = mc_med_diff(
    pools3b_3, SEED_MC_T3B_3)
obs_t3b_20, nul3b_20, p_t3b_20 = mc_med_diff(
    pools3b_20, SEED_MC_T3B_20)
log('T3b: L3 obs=%.4f null=%.4f p=%.5f | L20 '
    'obs=%.4f null=%.4f p=%.5f'
    % (obs_t3b_3, nul3b_3, p_t3b_3,
       obs_t3b_20, nul3b_20, p_t3b_20))
log('T3b frac ||resp||/||t||: L3 med=%.4f L20 '
    'med=%.4f' % (float(np.nanmedian(FRAC_E[3])),
                  float(np.nanmedian(FRAC_E[20]))))

# ---------- T4: attention routing readout ----------
log('=== T4 attention routing (body 0) ===')


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-30)
    rb = (rb - rb.mean()) / (rb.std() + 1e-30)
    return float(np.mean(ra * rb))


b0 = 0
base_i = idx_of[(0, b0, False)]
pos = assembled[base_i]['pos']
ids = assembled[base_i]['ids']
lg0_attn, attn0 = forward_attn(ids)
T4 = {}
for li, gvec, axis, seed in (
        (3, GBARPRE3, UBARPRE3, SEED_RND_T4),
        (20, GBARPRE20, UBARPRE20,
         SEED_RND_T4 + 1)):
    g = float(gvec[b0])
    a_dlg = []
    a_att = []
    r_dlg = []
    r_att = []
    dlg_a, attn_a = forward_inj_attn(
        ids, li, pos, g * axis)
    da = float(sum(float(np.sum(np.abs(
        attn_a[li][0, qh, -1, :]
        - attn0[li][0, qh, -1, :])))
        for qh in QH))
    na = float(np.linalg.norm(dlg_a - lg0_attn))
    a_dlg.append(na)
    a_att.append(da)
    rec_kind.append(8 if li == 3 else 9)
    rec_body.append(b0)
    rec_dlg.append(na)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((N_RND_T4, HDIM))
    R = R / np.linalg.norm(R, axis=1)[:, None]
    for rd in range(N_RND_T4):
        dlg_r, attn_r = forward_inj_attn(
            ids, li, pos, R[rd] * g)
        dr_att = float(sum(float(np.sum(np.abs(
            attn_r[li][0, qh, -1, :]
            - attn0[li][0, qh, -1, :])))
            for qh in QH))
        nr = float(np.linalg.norm(
            dlg_r - lg0_attn))
        r_dlg.append(nr)
        r_att.append(dr_att)
        rec_kind.append(10 if li == 3 else 11)
        rec_body.append(b0)
        rec_dlg.append(nr)
    all_dlg = np.array(a_dlg + r_dlg)
    all_att = np.array(a_att + r_att)
    T4[li] = {'med_axis_attn': float(
                  np.median(a_att)),
              'med_rand_attn': float(
                  np.median(r_att)),
              'med_axis_dlg': float(
                  np.median(a_dlg)),
              'med_rand_dlg': float(
                  np.median(r_dlg)),
              'spearman': spearman(all_att,
                                   all_dlg)}
    log('T4 L%d: attn axis=%.3e rand=%.3e | dlg '
        'axis=%.3f rand=%.3f | rho=%.3f'
        % (li, T4[li]['med_axis_attn'],
           T4[li]['med_rand_attn'],
           T4[li]['med_axis_dlg'],
           T4[li]['med_rand_dlg'],
           T4[li]['spearman']))

# ---------- T5: dose ladder ----------
log('=== T5 dose ladder (body 0) ===')
ladder = {}
for li, gvec, axis in ((3, GBARPRE3, UBARPRE3),
                       (20, GBARPRE20,
                        UBARPRE20)):
    g = float(gvec[b0])
    rows = []
    for mult in T5_MULT:
        nrm = g * mult
        dlg_a, _ = forward_inj(
            ids, li, pos, nrm * axis)
        na = float(np.linalg.norm(
            dlg_a - LGs[base_i]))
        rng = np.random.default_rng(
            SEED_RND_T5 + int(mult * 100) + li)
        R = rng.standard_normal((N_RND_T5, HDIM))
        R = R / np.linalg.norm(
            R, axis=1)[:, None]
        vals = [na]
        for rd in range(N_RND_T5):
            dlg_r, _ = forward_inj(
                ids, li, pos, nrm * R[rd])
            vals.append(float(np.linalg.norm(
                dlg_r - LGs[base_i])))
        rows.append((mult, na,
                     float(np.median(vals[1:]))))
        rec_kind.append(4 if li == 3 else 5)
        rec_body.append(b0)
        rec_dlg.append(na)
    ladder[li] = rows
    log('T5 L%d: %s' % (li, [
        'm=%.2f a=%.3f r=%.3f' % rw
        for rw in rows]))

# ---------- verdict ----------
max_att_t4 = max(max(T4[3]['med_axis_attn'],
                     T4[3]['med_rand_attn']),
                 max(T4[20]['med_axis_attn'],
                     T4[20]['med_rand_attn']))
med_ratio = float(ratio_sum / max(ratio_n, 1))
a118_ok = bool(n_integ_fail == 0)
a119_ok = bool(max_dlg_t2 >= DLG_GATE
               and max_att_t4 >= ATTN_GATE)
fieldK = bool(p_t1p_3 < 0.05 or p_t1p_20 < 0.05)
causK = bool(p_t2_3 < 0.05 or p_t2_20 < 0.05)
routK = bool(p_t3a_3 < 0.05 or p_t3a_20 < 0.05
             or p_t3b_3 < 0.05 or p_t3b_20 < 0.05)
if fieldK and causK and routK:
    verdict = 'kfield_causal_route_qwen'
elif fieldK and causK:
    verdict = 'kfield_causal_local_qwen'
elif fieldK and routK:
    verdict = 'kfield_route_specific_qwen'
elif fieldK:
    verdict = 'kfield_correlational_qwen'
else:
    verdict = 'kfield_null_qwen'
anchor_core_ok = bool(
    a114_diff == 0.0 and a115_diff == 0.0
    and a116_ok and a117_ok and a118_ok)

log('=== verdict ===')
log('a114=%r a115=%r a116=%s a117_ok=%s (%.6f,%d) '
    'a118=%s (fail=%d med_ratio=%.4f) a119=%s '
    '(maxdlg=%.3f maxattn=%.3e)'
    % (a114_diff, a115_diff, a116_ok, a117_ok,
       a117_min, a117_matched, a118_ok,
       n_integ_fail, med_ratio, a119_ok,
       max_dlg_t2, max_att_t4))
log('T1pre p3=%.5f p20=%.5f | T2 p3=%.5f p20=%.5f '
    '| T3a p3=%.5f p20=%.5f | T3b p3=%.5f p20=%.5f'
    % (p_t1p_3, p_t1p_20, p_t2_3, p_t2_20,
       p_t3a_3, p_t3a_20, p_t3b_3, p_t3b_20))
log('VERDICT=%s anchor_core_ok=%s'
    % (verdict, anchor_core_ok))

elapsed = time.time() - t0

npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    bodies=np.array(BODIES),
    Kbase3=Kbase[3], Kbase20=Kbase[20],
    KPpre3=KPpre[3], KPpre20=KPpre[20],
    D3K=D3K, D20K=D20K,
    DPRE3=DPRE3, DPRE20=DPRE20,
    alphaK3=ALPHAK3, ubarK3=UBARK3,
    alphaK20=ALPHAK20, ubarK20=UBARK20,
    alphaPre3=ALPHAPRE3, ubarPre3=UBARPRE3,
    alphaPre20=ALPHAPRE20, ubarPre20=UBARPRE20,
    GK3=GK3, GK20=GK20,
    GPRE3=GPRE3, GPRE20=GPRE20,
    GBARK3=GBARK3, GBARK20=GBARK20,
    GBARPRE3=GBARPRE3, GBARPRE20=GBARPRE20,
    obs_t1p_3=np.float64(obs_t1p_3),
    p_t1p_3=np.float64(p_t1p_3),
    obs_t1p_20=np.float64(obs_t1p_20),
    p_t1p_20=np.float64(p_t1p_20),
    obs_t1o_3=np.float64(obs_t1o_3),
    p_t1o_3=np.float64(p_t1o_3),
    obs_t1o_20=np.float64(obs_t1o_20),
    p_t1o_20=np.float64(p_t1o_20),
    n_pairs=np.int64(npair),
    eff_a3=np.array([EFF_A[3].get(b, float('nan'))
                     for b in range(8)]),
    eff_a20=np.array([EFF_A[20].get(b, float('nan'))
                      for b in range(8)]),
    eff_r3=np.stack([EFF_R[3].get(
        b, np.full(N_RND_T2, np.nan))
        for b in range(8)]),
    eff_r20=np.stack([EFF_R[20].get(
        b, np.full(N_RND_T2, np.nan))
        for b in range(8)]),
    obs_t2_3=np.float64(obs_t2_3),
    p_t2_3=np.float64(p_t2_3),
    obs_t2_20=np.float64(obs_t2_20),
    p_t2_20=np.float64(p_t2_20),
    cos_a3=COS_A[3], cos_a20=COS_A[20],
    cos_r3=COS_R[3], cos_r20=COS_R[20],
    obs_t3a_3=np.float64(obs_t3a_3),
    p_t3a_3=np.float64(p_t3a_3),
    obs_t3a_20=np.float64(obs_t3a_20),
    p_t3a_20=np.float64(p_t3a_20),
    cos_e3=COS_E[3], cos_e20=COS_E[20],
    cos_er3=COS_ER[3], cos_er20=COS_ER[20],
    frac_e3=FRAC_E[3], frac_e20=FRAC_E[20],
    obs_t3b_3=np.float64(obs_t3b_3),
    p_t3b_3=np.float64(p_t3b_3),
    obs_t3b_20=np.float64(obs_t3b_20),
    p_t3b_20=np.float64(p_t3b_20),
    t4_med_axis_attn3=np.float64(
        T4[3]['med_axis_attn']),
    t4_med_rand_attn3=np.float64(
        T4[3]['med_rand_attn']),
    t4_med_axis_dlg3=np.float64(
        T4[3]['med_axis_dlg']),
    t4_med_rand_dlg3=np.float64(
        T4[3]['med_rand_dlg']),
    t4_spearman3=np.float64(T4[3]['spearman']),
    t4_med_axis_attn20=np.float64(
        T4[20]['med_axis_attn']),
    t4_med_rand_attn20=np.float64(
        T4[20]['med_rand_attn']),
    t4_med_axis_dlg20=np.float64(
        T4[20]['med_axis_dlg']),
    t4_med_rand_dlg20=np.float64(
        T4[20]['med_rand_dlg']),
    t4_spearman20=np.float64(T4[20]['spearman']),
    ladder3=np.array([[rw[0], rw[1], rw[2]]
                      for rw in ladder[3]]),
    ladder20=np.array([[rw[0], rw[1], rw[2]]
                       for rw in ladder[20]]),
    rec_kind=np.array(rec_kind),
    rec_body=np.array(rec_body),
    rec_dlg=np.array(rec_dlg),
    n_integ_fail=np.int64(n_integ_fail),
    med_ratio=np.float64(med_ratio),
    max_dlg_t2=np.float64(max_dlg_t2),
    max_att_t4=np.float64(max_att_t4),
    a114_diff=np.float64(a114_diff),
    a115_diff=np.float64(a115_diff),
    a116_ok=np.bool_(a116_ok),
    a117_min=np.float64(a117_min),
    a117_matched=np.int64(a117_matched),
    a118_ok=np.bool_(a118_ok),
    a119_ok=np.bool_(a119_ok),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'run': 'run5 authoritative (pre-norm redesign '
           'after the run3 qk-norm overdrive '
           'diagnosis; run1 assembly slip and run2 '
           'cuda attention crash registered; see '
           'corrections)',
    'anchor_core_ok': anchor_core_ok,
    'anchors': {
        'a114_dup_prefill_bit': a114_diff,
        'a115_dup_all_bit': a115_diff,
        'a116_source_seals': a116_ok,
        'a117_cross_precision_min_cos': a117_min,
        'a117_matched': a117_matched,
        'a118_integrity_ok': a118_ok,
        'n_integ_fail': n_integ_fail,
        'a119_chain_entry_ok': a119_ok,
        'max_dlg_t2': max_dlg_t2,
        'max_att_t4': max_att_t4,
    },
    'T1pre_field': {'obs_L3': obs_t1p_3,
                    'p_L3': p_t1p_3,
                    'obs_L20': obs_t1p_20,
                    'p_L20': p_t1p_20,
                    'n_pairs': int(npair)},
    'T1post_field': {'obs_L3': obs_t1o_3,
                     'p_L3': p_t1o_3,
                     'obs_L20': obs_t1o_20,
                     'p_L20': p_t1o_20},
    'T2_efficiency': {'obs_L3': obs_t2_3,
                      'p_L3': p_t2_3,
                      'obs_L20': obs_t2_20,
                      'p_L20': p_t2_20},
    'T3a_axis_replay': {'obs_L3': obs_t3a_3,
                        'p_L3': p_t3a_3,
                        'obs_L20': obs_t3a_20,
                        'p_L20': p_t3a_20},
    'T3b_exact_replay': {'obs_L3': obs_t3b_3,
                         'p_L3': p_t3b_3,
                         'obs_L20': obs_t3b_20,
                         'p_L20': p_t3b_20,
                         'frac_med_L3': float(
                             np.nanmedian(
                                 FRAC_E[3])),
                         'frac_med_L20': float(
                             np.nanmedian(
                                 FRAC_E[20]))},
    'T4_attention': {'L3': T4[3], 'L20': T4[20]},
    'T5_ladder': {'L3': ladder[3],
                  'L20': ladder[20]},
    'flags': {'fieldK': fieldK, 'causK': causK,
              'routK': routK,
              'n_integ_fail': n_integ_fail,
              'med_post_ratio': med_ratio},
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
