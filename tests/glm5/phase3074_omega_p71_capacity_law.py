# -*- coding: utf-8 -*-
"""Phase 3074: Omega-P71 capacity law - the
saturation curve R(S) over focal-head swap
subsets (qwen3-4b single model bf16).

Question (3074 A, menu of 3073): 3073 showed
the head-level causal effect is a SET function
of the swapped-head set S with strong
diminishing returns (1 head max 0.222, best
pair 0.389, top-8 joint 0.532, full 32-head
0.572).  WHAT is the functional form of the
saturation?

E1 inj@34 ladder: 24 pairs, repV protocol
3065/.../3073-identical (a1 bit anchor vs the
3066 npz; med_c_34 reference assert; aref
PA34/PF34 bit vs 3069; a8/a9 bit vs the 3071
npz).
E3 SATURATION SWEEP: ALL 255 non-empty subsets
of the focal top-8 (bitmask over positions
[20,7,1,14,26,0,2,24]) swapped jointly to
base, 24 pairs each.  Anchors embedded: a10
family = every 3073 condition is a subset
(singles bit vs 3071 r34[top8]; 28 pairs +
triplet + top-8 bit vs 3073 r2/r_t3/r_u8);
a11 = full 32-head swap bit vs 3071 recov
gA -0.5717521069904176.
E4 SUBMODULARITY: all 16472 inequalities
A(S+x)-A(S) <= A(T+x)-A(T), T a proper
subset of S (empty included), x not in S
(amplitude A = -R, tol 0.02 ~ 24-pair
median SE scale); violation rate
recorded, not gating.
E5 CAPACITY-LAW FIT: input budget x(S) =
sum of single-head amplitudes m_i = |r1_i|
(3071/3073 bit-identical); candidates
A(x) = x (add), a(1-exp(-x/b)) (exp),
a*ln(1+x/b) (log), a x^g/(b^g+x^g) (hill);
grid + refinement, least squares on ALL
255 amplitude points (descriptive) and on
|S| <= 2 only (36 points = 8 singles + 28
pairs) -> extrapolate
the top-8 joint (measured -0.5322981028358011)
and the full 32-head (measured -0.5717521069904176).
verdict: setup fail -> setup_failed_
saturation; a1 fail -> anchor_mismatch_3066_
ladder; a8 fail -> anchor_mismatch_3071_zh;
a9 fail -> anchor_mismatch_3071_dah; a10
fail -> anchor_mismatch_3073_sweep; a11
fail -> anchor_mismatch_3071_allswap; b8
fail -> block_output_mismatch; then among
candidates passing BOTH gates (|pred(S8) -
0.5323| < 0.05 AND |pred(all32) - 0.5718|
< 0.05, ~9 percent of the full recovery)
pick the smallest total extrapolation
error -> capacity_law_<name>; if exactly
one gate passed -> capacity_law_partial_
<name>; else capacity_law_undetermined.
Submodularity rate and the fit table are
recorded, not gating.
memory discipline: single model; banks fp64
CPU; Wo34/Wo35 fp32 resident in E2H then
deleted; W32 resident; del + gc +
empty_cache at the end.
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

PHASE = 3074
NAME = 'omega_p71_capacity_law'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3074', NAME)
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
NPZ71 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3071', 'omega_p68_attn_head_decomp',
    'omega_p68_attn_head_decomp.npz')
RES71 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3071', 'omega_p68_attn_head_decomp',
    'result.json')
RES73 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3073', 'omega_p70_head_interaction',
    'result.json')
GATE_PRED = 0.05
TOL_SUB = 0.02

BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')

PREREG = {
    'mode': 'single model qwen3-4b bf16 eager '
            'seed 3020; E1 protocol 3065/.../'
            '3073-identical (a1 bit anchor vs '
            '3066 npz COS_LAD row 34, hard; '
            'med_c_34 reference assert; aref '
            'PA34/PF34 bit vs 3069; a8 ZH34/'
            'ZH35 bit vs 3071 npz; a9 DAH '
            'medians bit vs 3071 npz; a10 '
            'family = all 38 3073 conditions '
            'are subsets of the sweep, bit vs '
            '3073 r1/r2/r_t3/r_u8 and 3071 '
            'r34[top8]; a11 full 32-head swap '
            'bit vs 3071 recov gA); lens '
            'probes fp32 W32 resident; verdict '
            'scalars fp64 cosv; smoke mode '
            'optional (SMOKE=1: 12 masks, 4 '
            'pairs, anchors a1/a8/a9/a10/a11 '
            'off, b-anchors on)',
    'question': '3074 A (menu of 3073): the '
                'head-level causal effect is a '
                'set function R(S) with strong '
                'diminishing returns - WHAT is '
                'the functional form?  Full '
                'measurement: all 255 non-empty '
                'subsets of the focal top-8 '
                'swapped jointly (24 pairs '
                'each); submodularity checked '
                'on all 16472 inequalities; '
                'capacity-law candidates fit '
                'on |S|<=2 and extrapolated to '
                'the 8-head and 32-head joints.',
    'E1_ladder': 'injection layer 34 x 24 pairs, '
                 'repV 3065-identical; COS_LAD_34 '
                 '(a1 bit anchor vs 3066 npz row '
                 '34); PA34/PF34 lens probes; zH '
                 'captured at L34/L35 per pair',
    'E2H_perhead': 'dzH 32 slices of 128; dAh TT '
                   'projection medians (a9 bit '
                   'anchor vs 3071 npz); a8 ZH '
                   'bit anchor vs 3071 npz',
    'E3_sweep': 'ALL 255 non-empty subsets of '
                'TOP8=[20,7,1,14,26,0,2,24] '
                '(bitmask, subset = swap those '
                'head slices to base at the '
                'last position), 24 pairs each; '
                'R(S) = median cos - med_c_34; '
                'a10 family: 8 singles vs 3071 '
                'r34[top8], 28 pairs + triplet '
                '(mask 7) + top-8 (mask 255) vs '
                '3073 r2/r_t3/r_u8, all bit 0.0; '
                'a11: full 32-head swap vs 3071 '
                'recov gA -0.5717521069904176 '
                'bit 0.0',
    'E4_submodularity': 'amplitude A = -R; all '
                        '17496 inequalities '
                        'A(S+x)-A(S) <= A(T+x)-'
                        'A(T) for T subset S '
                        '(incl. T = empty), x '
                        'not in S; tol 0.02 '
                        '(~24-pair median SE); '
                        'violation rate recorded, '
                        'not gating',
    'E5_fit': 'budget x(S) = sum m_i over '
              'swapped heads, m_i = |r1_i| '
              '(bit-identical 3071/3073 '
              'singles); candidates: add A=x; '
              'exp A=a(1-exp(-x/b)); log A=a*'
              'ln(1+x/b); hill A=a x^g/(b^g+'
              'x^g); grid+refinement least '
              'squares; fit1 = all 255 points '
              '(descriptive); fit2 = |S|<=2 '
              'only (36 points) -> extrapolate '
              'S8 (measured -0.5322981028358011)'
              ' and all-32 (measured '
              '-0.5717521069904176)',
    'gates': 'capacity law: BOTH |pred(S8) - '
             '0.5323| < 0.05 AND |pred(all32) - '
             '0.5718| < 0.05 (~9 percent of the '
             'full recovery); submodularity tol '
             '0.02',
    'verdict': 'setup fail -> setup_failed_'
               'saturation; a1 fail -> anchor_'
               'mismatch_3066_ladder; a8 fail '
               '-> anchor_mismatch_3071_zh; a9 '
               'fail -> anchor_mismatch_3071_'
               'dah; a10 fail -> anchor_mismatch'
               '_3073_sweep; a11 fail -> anchor_'
               'mismatch_3071_allswap; b8 fail '
               '-> block_output_mismatch; among '
               'candidates passing BOTH gates '
               'pick smallest total '
               'extrapolation error -> '
               'capacity_law_<name>; exactly '
               'one gate -> capacity_law_partial'
               '_<name>; else capacity_law_'
               'undetermined.  Submodularity '
               'rate and fit table recorded, '
               'not gating.',
    'statistics_discipline': 'same-precision bit '
                             'anchors on frozen '
                             'seeds; TT logit-space '
                             '3065-identical; R(S) '
                             'referenced to med_c_'
                             '34; subset order fixed '
                             '(mask 1..255, bit i = '
                             'TOP8 position i); fit '
                             'input = single-head '
                             'amplitudes measured in '
                             'the same run (a10-'
                             'anchored); no post-hoc '
                             'model changes',
    'memory_discipline': 'single model; banks '
                         'fp64 CPU; Wo34/Wo35 '
                         'fp32 resident in E2H '
                         'then deleted; W32 '
                         'resident; del + gc + '
                         'empty_cache at end',
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
    at o_proj inputs (last position; idx may
    be any head-slice index array).  Returns
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
    n = len(ids)
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


# ==== assembly (3065-identical) ====
word_tok = {}
for w in TARGETS:
    wi = tok(' ' + w,
             add_special_tokens=False)['input_ids']
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
                          'cond': ci,
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
        assert off > 0, (i,)
        assert list(pid[off + 1:]) \
            == list(bid[1:]), (i,)
        w0b = tok.decode([bid[0]]).strip()
        w0p = tok.decode([pid[off]]).strip()
        assert w0b == w0p, (i, w0b, w0p)
        assembled[i]['off'] = off
LENS = np.array([len(assembled[i]['ids'])
                 for i in range(n_pr)])
NMAX = int(LENS.max())
log('assembled 32 prompts (lens %d-%d)'
    % (int(LENS.min()), NMAX))

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
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
    lg, lg_gpu, vb, po, zX, zA, zM, zP, zH, \
        zZ, zG, zU, zACT = forward_gen(
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
log('banks: LG%s VB%s BH%s BACT34%s (forwards='
    '%d)'
    % (LG.shape, VB.shape, BH.shape,
       BACT34.shape, FW[0]))
log('a2lens lens(final)=logits: fp32 max|d|='
    '%.3e (<=0.125 ok)' % a2_max)

b0_diff = 0.0
for si in (0, 9, 17, 31):
    lg, lg_gpu, vb, po, zX, zA, zM, zP, zH, \
        zZ, zG, zU, zACT = forward_gen(
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
log('b0 recapture diff=%.3e ok=%s'
    % (b0_diff, b0_ok))

TT = np.stack([
    LG[idx_of[(int(cidx[k]), int(bidx[k]))]]
    - LG[idx_of[(0, int(bidx[k]))]]
    for k in range(NP_)])
TTG = torch.tensor(TT, dtype=torch.float32,
                   device='cuda')
TTn = torch.tensor(
    np.linalg.norm(TT, axis=1),
    dtype=torch.float32, device='cuda')

base0 = idx_of[(0, int(bidx[0]))]
n0 = int(LENS[base0])
selfV = VB[base0, :, :n0, :].copy()
lg_s = forward_gen(
    assembled[base0]['ids'], repl=selfV)[0]
b1_diff = float(np.max(np.abs(lg_s
                              - LG[base0])))
b1_ok = bool(b1_diff == 0.0)
log('b1 sham self-replacement diff=%.3e ok=%s'
    % (b1_diff, b1_ok))

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
log('b3 finite=%s' % b3_ok)

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
log('b5 act=silu(g)*u bf16 reconstruction '
    'diff=%.3e (recorded)' % b5_diff)

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
log('b6 (x+a)+m=h2 bf16 diff=%.3e ok=%s'
    % (b6_diff, b6_ok))

ridx = np.arange(FRONT)
ALLH = np.arange(NQW)
ALLI = np.arange(INTER)
HEAD_IDX = [np.arange(h * HDIM, (h + 1) * HDIM)
            for h in range(NH)]


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
    return torch.cat([bx, bx + ba, bp], dim=0)


def lens_cos(zinj, zbase, k):
    with torch.no_grad():
        li_ = F.linear(
            final_norm(zinj).float(), W32)
        lb_ = F.linear(
            final_norm(zbase).float(), W32)
        d = li_ - lb_
        t = TTG[k]
        num = (d * t.unsqueeze(0)).sum(dim=1)
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
        np.abs(zX[L34].double().cpu().numpy()
               - BX[base_i, L34]))))
    PA34[k] = lens_cos(
        torch.cat([zX, zX + zA, zP], dim=0),
        base_z_gpu(base_i, nb), k)[NL + L34]
    PF34[k] = lens_cos(
        torch.cat([zX, zX + zA, zP], dim=0),
        base_z_gpu(base_i, nb), k)[2 * NL + L34]
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
log('b4 delta-x at L34 diff=%.3e ok=%s'
    % (b4_diff, b4_ok))
log('b8 dzX_35 vs dzP_34 diff=%.3e ok=%s'
    % (b8_diff, b8_ok))

med_c_34 = float(np.median(COS_LAD_34))
pa34 = float(np.median(PA34))
pf34 = float(np.median(PF34))
log('E1 med_c(34)=%.4f PA=%.4f PF=%.4f '
    '(forwards=%d)'
    % (med_c_34, pa34, pf34, FW[0]))

# a1: row 34 bit anchor vs 3066 npz
a1_diff = None
a1_ok = False
npz66 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3066',
    'omega_p63_last_layer_flip_anatomy',
    'omega_p63_last_layer_flip_anatomy.npz')
if not SMOKE:
    z = np.load(npz66)
    c66 = z['COS_LAD']
    a1_diff = float(np.max(
        np.abs(COS_LAD_34 - c66[L34])))
    a1_ok = bool(a1_diff == 0.0)
    log('a1 ladder row 34 vs 3066 npz: '
        'max|d|=%.3e ok=%s'
        % (a1_diff, a1_ok))
else:
    log('a1 skipped (smoke)')

if not SMOKE:
    assert abs(med_c_34 - MEDC34_REF) \
        < 1e-12, med_c_34
    log('med_c_34 reference assert ok')

aref_diff = None
aref_ok = False
if not SMOKE:
    aref_diff = max(abs(pa34 - PA34_REF),
                    abs(pf34 - PF34_REF))
    aref_ok = bool(aref_diff == 0.0)
    log('aref PA34/PF34 medians vs 3069 '
        'per-layer authoritative: max|d|='
        '%.3e ok=%s' % (aref_diff, aref_ok))
else:
    aref_ok = True
    log('aref skipped (smoke)')

# ==== E2H per-head observation decomposition ====
Wo34 = layers[L34].self_attn.o_proj.weight \
    .detach().float()
Wo35 = layers[L35].self_attn.o_proj.weight \
    .detach().float()
WoT34 = Wo34.t().reshape(NQ, HDIM, HID)
WoT35 = Wo35.t().reshape(NQ, HDIM, HID)
DAH34 = np.zeros((NP_USE, NQ))
DAH35 = np.zeros((NP_USE, NQ))
HEADLIN34 = np.zeros(NP_USE)


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
log('E2H L34: headlin rel med=%.2e | top obs '
    'heads %s'
    % (hl34_med,
       np.argsort(DAH34_MED)[::-1][:6]
       .tolist()))
del Wo34, Wo35, WoT34, WoT35
gc.collect()
torch.cuda.empty_cache()

# a8: ZH34/ZH35 bit anchor vs 3071 npz
a8_diff = None
a8_ok = False
z71 = None
if not SMOKE:
    z71 = np.load(NPZ71)
    a8_diff = max(
        float(np.max(np.abs(
            ZH34 - z71['ZH34']))),
        float(np.max(np.abs(
            ZH35 - z71['ZH35']))))
    a8_ok = bool(a8_diff == 0.0)
    log('a8 ZH34/ZH35 vs 3071 npz: max|d|='
        '%.3e ok=%s' % (a8_diff, a8_ok))
else:
    log('a8 skipped (smoke)')

# a9: DAH medians bit anchor vs 3071 npz
a9_diff = None
a9_ok = False
if not SMOKE:
    a9_diff = max(
        float(np.max(np.abs(
            DAH34_MED - z71['DAH34_MED']))),
        float(np.max(np.abs(
            DAH35_MED - z71['DAH35_MED']))))
    a9_ok = bool(a9_diff == 0.0)
    log('a9 DAH medians vs 3071 npz: max|d|='
        '%.3e ok=%s' % (a9_diff, a9_ok))
else:
    log('a9 skipped (smoke)')

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
b7_ok = bool(b7a_diff == 0.0 and b7c_diff == 0.0)
log('b7 identity self-swaps (attn %.1e act '
    '%.1e) ok=%s'
    % (b7a_diff, b7c_diff, b7_ok))


# ==== E3 saturation sweep ====
j71 = json.load(open(RES71, encoding='utf-8'))
R34_71 = np.array(
    j71['stats']['head']['r34'],
    dtype=np.float64)
TOP8 = [int(v) for v
        in j71['stats']['head']['top8']]
j73 = json.load(open(RES73, encoding='utf-8'))
R1_73 = np.array(j73['stats']['r1'],
                 dtype=np.float64)
R2_73 = np.array(j73['stats']['r2'],
                 dtype=np.float64)
R_T3_73 = float(j73['stats']['r_t3'])
R_U8_73 = float(j73['stats']['r_u8'])
NT = len(TOP8)
if SMOKE:
    MASKS = [1, 2, 3, 5, 7, 15, 31, 63, 127,
             255, 24, 85]
else:
    MASKS = list(range(1, 256))
NM = len(MASKS)
log('E3 sweep: NT=%d masks=%d (1..%s) '
    'K3=%d'
    % (NT, NM, MASKS[-1], K3))


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
            repl=repV, **mk_kwargs(base_i))[0]
        cs[k] = cosv(lg - LG[base_i], TT[k])
    return cs


CS = np.full((NM, K3), np.nan)
for mi, m in enumerate(MASKS):
    sel = [ti for ti in range(NT)
           if m >> ti & 1]
    idx = np.concatenate(
        [HEAD_IDX[TOP8[ti]] for ti in sel])
    CS[mi] = run_cond(lambda bi, idx=idx: {
        'attn_swaps': [(L34, idx,
                        BH[bi, L34][idx])]})
    if (mi + 1) % 32 == 0 or mi == NM - 1:
        log('E3 sweep progress %d/%d '
            '(forwards=%d)'
            % (mi + 1, NM, FW[0]))
R_S = np.median(CS, axis=1) - med_c_34
A_S = -R_S
log('E3 sweep R(S) collected: min=%.4f '
    'max=%.4f (most negative at mask %d)'
    % (float(R_S.min()), float(R_S.max()),
       int(MASKS[int(np.argmin(R_S))])))

# a10 family: 3073 conditions are subsets
a10_diff = None
a10_ok = False
if not SMOKE:
    mask_of = {m: i for i, m
               in enumerate(MASKS)}
    refs = []
    d10 = 0.0
    for ti in range(NT):
        m = 1 << ti
        d10 = max(d10, abs(
            R_S[mask_of[m]] - R1_73[ti]))
    for p_i, (a_, b_) in enumerate(
            itertools.combinations(
                range(NT), 2)):
        m = (1 << a_) | (1 << b_)
        d10 = max(d10, abs(
            R_S[mask_of[m]] - R2_73[p_i]))
    d10 = max(d10, abs(
        R_S[mask_of[7]] - R_T3_73))
    d10 = max(d10, abs(
        R_S[mask_of[255]] - R_U8_73))
    a10_diff = float(d10)
    a10_ok = bool(a10_diff == 0.0)
    log('a10 family (38 3073 conditions) vs '
        '3071/3073 refs: max|d|=%.3e ok=%s'
        % (a10_diff, a10_ok))
else:
    log('a10 skipped (smoke)')

# a11: full 32-head swap vs 3071 gA recov
lg_all = run_cond(lambda bi: {
    'attn_swaps': [(L34, ALLH,
                    BH[bi, L34])]})
R_ALL = float(np.median(lg_all) - med_c_34)
a11_diff = None
a11_ok = False
if not SMOKE:
    a11_diff = abs(R_ALL - GALL_3071)
    a11_ok = bool(a11_diff == 0.0)
    log('a11 all-32 swap recov=%+.4f vs 3071 '
        'gA %+.4f: |d|=%.3e ok=%s'
        % (R_ALL, GALL_3071, a11_diff,
           a11_ok))
else:
    log('a11 skipped (smoke)')

# ==== E4 submodularity ====
AMP = {}
for i, m in enumerate(MASKS):
    AMP[m] = float(-R_S[i])
AMP[0] = 0.0


def amp(m):
    return AMP.get(m)


pairs4 = []
if not SMOKE:
    full_masks = list(range(1, 256))
    for S in full_masks:
        for x in range(NT):
            if S >> x & 1:
                continue
            mSx = S | (1 << x)
            base_m = amp(mSx) - amp(S)
            # enumerate all T subset S
            # (incl. T = empty)
            TT_ = S
            while True:
                TT_ = (TT_ - 1) & S
                if TT_ < 0:
                    break
                if TT_ == S:
                    continue
                mt = amp(TT_ | (1 << x)) \
                    - amp(TT_)
                pairs4.append(
                    base_m - mt)
                if TT_ == 0:
                    break
    par = np.array(pairs4)
    n_sub_checked = len(par)
    n_viol = int((par > TOL_SUB).sum())
    viol_rate = float(n_viol) / n_sub_checked
    sub_ok = bool(viol_rate < 0.10)
    log('E4 submodularity: %d inequalities, '
        'violations %d (%.3f percent), rate '
        '< 0.10 -> %s | worst excess=%.4f'
        % (n_sub_checked, n_viol,
           100.0 * viol_rate, sub_ok,
           float(par.max())))
else:
    n_sub_checked = 0
    n_viol = 0
    viol_rate = float('nan')
    sub_ok = None
    log('E4 submodularity skipped (smoke)')

# ==== E5 capacity-law fit ====
MAG = np.abs(R1_73)
X32 = float(np.abs(R34_71).sum())


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


# build the design arrays
fits = {}
if not SMOKE:
    mask_of = {m: i for i, m
               in enumerate(MASKS)}
    xs_all = []
    ys_all = []
    sizes = []
    for m in range(1, 256):
        x = float(sum(
            MAG[ti] for ti in range(NT)
            if m >> ti & 1))
        xs_all.append(x)
        ys_all.append(amp(m))
        sizes.append(bin(m).count('1'))
    xs_all = np.array(xs_all)
    ys_all = np.array(ys_all)
    sizes = np.array(sizes)
    le2 = sizes <= 2
    x_u8 = float(MAG.sum())
    y_u8 = float(-R_S[mask_of[255]])
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
            name, X32, p_2))
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
            'pass_u8': bool(e8 < GATE_PRED),
            'pass_all32': bool(
                e32 < GATE_PRED)}
        log('E5 fit %-4s | all: p=%s sse=%.4f '
            '| le2: p=%s sse=%.4f | pred '
            'S8=%.4f (err %.4f) all32=%.4f '
            '(err %.4f)'
            % (name,
               np.round(p_all, 4).tolist(),
               sse_all,
               np.round(p_2, 4).tolist(),
               sse_2, pr8, e8, pr32, e32))
    passed = [nm for nm in fits
              if fits[nm]['pass_u8']
              and fits[nm]['pass_all32']]
    partial = [nm for nm in fits
               if nm not in passed
               and (fits[nm]['pass_u8']
                    or fits[nm]['pass_all32'])]
    if passed:
        best_nm = min(passed,
                      key=lambda nm:
                      fits[nm]['err_u8']
                      + fits[nm]['err_all32'])
        verdict = 'capacity_law_' + best_nm
    elif partial:
        best_nm = min(partial,
                      key=lambda nm:
                      fits[nm]['err_u8']
                      + fits[nm]['err_all32'])
        verdict = ('capacity_law_partial_'
                   + best_nm)
    else:
        verdict = 'capacity_law_undetermined'
    log('E5 gate: passed=%s partial=%s -> %s'
        % (passed, partial, verdict))
else:
    verdict = 'smoke_pending'
    log('E5 skipped (smoke)')

# ==== verdict ====
setup_ok = bool(b0_ok and b1_ok and b3_ok
                and b4_ok and b6_ok and b7_ok
                and aref_ok)
if verdict is not None:
    if not setup_ok:
        verdict = 'setup_failed_saturation'
    elif (not SMOKE) and (not a1_ok):
        verdict = 'anchor_mismatch_3066_ladder'
    elif (not SMOKE) and (not a8_ok):
        verdict = 'anchor_mismatch_3071_zh'
    elif (not SMOKE) and (not a9_ok):
        verdict = 'anchor_mismatch_3071_dah'
    elif (not SMOKE) and (not a10_ok):
        verdict = 'anchor_mismatch_3073_sweep'
    elif (not SMOKE) and (not a11_ok):
        verdict = 'anchor_mismatch_3071_allswap'
    elif (not SMOKE) and (not b8_ok):
        verdict = 'block_output_mismatch'
log('VERDICT: %s (setup_ok=%s a1_ok=%s a8_ok='
    '%s a9_ok=%s a10_ok=%s a11_ok=%s b8_ok=%s '
    'viol_rate=%s smoke=%s)'
    % (verdict, setup_ok, a1_ok, a8_ok, a9_ok,
       a10_ok, a11_ok, b8_ok, viol_rate, SMOKE))

# ==== npz ====
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'A1_DIFF': np.float64(
        a1_diff if a1_diff is not None
        else np.nan),
    'A1_OK': np.bool_(a1_ok),
    'A8_DIFF': np.float64(
        a8_diff if a8_diff is not None
        else np.nan),
    'A8_OK': np.bool_(a8_ok),
    'A9_DIFF': np.float64(
        a9_diff if a9_diff is not None
        else np.nan),
    'A9_OK': np.bool_(a9_ok),
    'A10_DIFF': np.float64(
        a10_diff if a10_diff is not None
        else np.nan),
    'A10_OK': np.bool_(a10_ok),
    'A11_DIFF': np.float64(
        a11_diff if a11_diff is not None
        else np.nan),
    'A11_OK': np.bool_(a11_ok),
    'B8_DIFF': np.float64(b8_diff),
    'B8_OK': np.bool_(b8_ok),
    'A2LENS_MAX': np.float64(a2_max),
    'AREF_DIFF': np.float64(
        aref_diff if aref_diff is not None
        else np.nan),
    'AREF_OK': np.bool_(aref_ok),
    'B0_DIFF': np.float64(b0_diff),
    'B1_DIFF': np.float64(b1_diff),
    'B4_DIFF': np.float64(b4_diff),
    'B5_DIFF': np.float64(b5_diff),
    'B6_DIFF': np.float64(b6_diff),
    'B7A_DIFF': np.float64(b7a_diff),
    'B7C_DIFF': np.float64(b7c_diff),
    'B3_OK': np.bool_(b3_ok),
    'B4_OK': np.bool_(b4_ok),
    'B6_OK': np.bool_(b6_ok),
    'B7_OK': np.bool_(b7_ok),
    'SETUP_OK': np.bool_(setup_ok),
    'COS_LAD_34': COS_LAD_34,
    'MED_C_34': np.float64(med_c_34),
    'PA34': PA34, 'PF34': PF34,
    'ZH34': ZH34, 'ZH35': ZH35,
    'DAH34': DAH34, 'DAH35': DAH35,
    'HEADLIN34': HEADLIN34,
    'DAH34_MED': DAH34_MED,
    'DAH35_MED': DAH35_MED,
    'HL34_MED': np.float64(hl34_med),
    'TOP8': np.array(TOP8, dtype=np.int64),
    'MASKS': np.array(MASKS,
                      dtype=np.int64),
    'CS': CS, 'R_S': R_S, 'A_S': A_S,
    'R_ALL': np.float64(R_ALL),
    'N_SUB_CHECKED': np.int64(n_sub_checked),
    'N_VIOL': np.int64(n_viol),
    'VIOL_RATE': np.float64(viol_rate),
    'TT': TT.astype(np.float32),
    'LG': LG.astype(np.float32),
}
if not SMOKE:
    save['PAIRS4'] = np.array(
        pairs4, dtype=np.float64)
np.savez(npz_path, **save)


# ==== result.json ====
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


result = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'elapsed': time.time() - t0,
    'forwards': int(FW[0]),
    'run': 'run1 authoritative (qwen3-4b bf16 '
           'single model)' if not SMOKE
    else 'smoke',
    'prereg': PREREG,
    'stats': {
        'med_c_34': f64(med_c_34),
        'pa34': f64(pa34), 'pf34': f64(pf34),
        'hl34_med': f64(hl34_med),
        'top8': TOP8,
        'r_all32': f64(R_ALL),
        'a_u8': f64(-R_S[MASKS.index(255)]),
        'a_best_subset': {
            'mask': int(MASKS[int(
                np.argmax(A_S))]),
            'amp': f64(float(A_S.max()))},
        'n_sub_checked': int(n_sub_checked),
        'n_viol': int(n_viol),
        'viol_rate': f64(viol_rate),
        'submodular_ok': (None if sub_ok
                          is None
                          else bool(sub_ok)),
        'x_u8': f64(float(MAG.sum())),
        'x_all32': f64(X32),
        'fits': {nm: {
            'p_all': [f64(v)
                      for v in d['p_all']],
            'sse_all': f64(d['sse_all']),
            'p_le2': [f64(v)
                      for v in d['p_le2']],
            'sse_le2': f64(d['sse_le2']),
            'pred_u8': f64(d['pred_u8']),
            'pred_all32': f64(
                d['pred_all32']),
            'err_u8': f64(d['err_u8']),
            'err_all32': f64(d['err_all32']),
            'pass_u8': d['pass_u8'],
            'pass_all32': d['pass_all32'],
        } for nm, d in fits.items()},
    },
    'anchors': {
        'a1_diff': a1_diff, 'a1_ok': a1_ok,
        'a8_diff': a8_diff, 'a8_ok': a8_ok,
        'a9_diff': a9_diff, 'a9_ok': a9_ok,
        'a10_diff': a10_diff, 'a10_ok': a10_ok,
        'a11_diff': a11_diff, 'a11_ok': a11_ok,
        'b8_diff': b8_diff, 'b8_ok': b8_ok,
        'a2lens_max': a2_max,
        'a2lens_ok': a2lens_ok,
        'aref_diff': aref_diff,
        'aref_ok': aref_ok,
        'b0_diff': b0_diff, 'b0_ok': b0_ok,
        'b1_diff': b1_diff, 'b1_ok': b1_ok,
        'b3_ok': b3_ok,
        'b4_diff': b4_diff, 'b4_ok': b4_ok,
        'b5_diff': b5_diff,
        'b6_diff': b6_diff, 'b6_ok': b6_ok,
        'b7a_diff': b7a_diff,
        'b7c_diff': b7c_diff,
        'b7_ok': b7_ok,
        'setup_ok': setup_ok},
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
    'setup_ok': setup_ok,
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
