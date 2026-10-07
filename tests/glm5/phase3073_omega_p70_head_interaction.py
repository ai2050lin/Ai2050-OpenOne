# -*- coding: utf-8 -*-
"""Phase 3073: Omega-P70 head interaction matrix
(qwen3-4b single model bf16).

Question (3073 A, menu of 3072): 3071 showed the
per-head causal medians are non-additive (top-5
sum -0.968 vs full swap -0.572).  WHERE does the
interaction come from?  Pairwise joint swaps of
the focal top-8 heads decompose it.

E1 inj@34 ladder: 24 pairs, repV protocol
3065/3066/3067/3069/3070/3071/3072-identical
(a1 anchor COS_LAD row 34 bit 0.0 vs the 3066
npz; med_c_34 reference assert; aref PA34/PF34
bit vs the 3069 authoritative values; zH
captured at L34/L35).
E2H per-head observation decomposition
(a9 anchor bit vs the 3071 npz medians; a8
anchor ZH34/ZH35 bit vs the 3071 npz) - zero
extra forwards, reuses E1 captures.
E3 interaction matrix on the 3071 focal top-8
(h20/7/1/14/26/0/2/24): (i) single-head
resweep x24 pairs (a10 anchor: recov medians
bit 0.0 vs the 3071 result r34 at top8);
(ii) all 28 pairwise joint swaps x24 (I(A,B) =
recov(AB) - recov(A) - recov(B), sample level);
(iii) full top-8 joint swap; (iv) triplet test
(the 3 most negative single heads) for third-
order residuals.  Second-order completeness:
pred_u8 = sum_i recov_i + sum_p I_p vs measured
top-8 joint.
verdict: setup fail -> setup_failed_interaction;
a1 fail -> anchor_mismatch_3066_ladder; a8 fail
-> anchor_mismatch_3071_zh; a9 fail ->
anchor_mismatch_3071_dah; a10 fail ->
anchor_mismatch_3071_r34; b8 fail ->
block_output_mismatch; then (thresholds: I
significance 0.02 ~ 10-30 percent of single-head
effects; pred err 0.05 ~ 9 percent of the -0.572
full recovery; triplet residual 0.02 ~ 5 percent
of the 3071 top-5 interaction total 0.396):
pred_err_med >= 0.05 -> higher_order_required;
max|I_med| < 0.02 -> additivity_holds; err3_med
< 0.02 -> pairwise_complete; else
pairwise_partial.  Sign of I (sub-additive if
positive given negative recovs) and attribute
correlations recorded, not gating.
memory discipline: single model; banks fp64
CPU; Wo34/Wo35 fp32 resident in E2H then
deleted; W32 resident; del + gc + empty_cache
at the end.
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

PHASE = 3073
NAME = 'omega_p70_head_interaction'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3073', NAME)
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
GATE_I = 0.02
GATE_PRED = 0.05
GATE_TRI = 0.02

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
            'seed 3020; E1 protocol 3065/3066/'
            '3067/3069/3070/3071/3072-identical '
            '(a1 bit anchor vs 3066 npz COS_LAD '
            'row 34, hard; med_c_34 reference '
            'assert; aref PA34/PF34 bit vs 3069; '
            'a8 ZH34/ZH35 bit vs 3071 npz; a9 '
            'DAH medians bit vs 3071 npz; a10 '
            'single-head recov medians bit vs '
            '3071 result r34 at top8); lens '
            'probes fp32 W32 resident; verdict '
            'scalars fp64 cosv; smoke mode '
            'optional (SMOKE=1: 4 pairs per '
            'condition, anchors a1/a8/a9/a10 '
            'off, b-anchors on)',
    'question': '3073 A (menu of 3072): 3071 '
                'showed per-head causal medians '
                'are non-additive (top-5 sum '
                '-0.968 vs full swap -0.572) - '
                'WHERE does the interaction come '
                'from?  Pairwise joint swaps of '
                'the focal top-8 heads decompose '
                'the interaction; second-order '
                'expansion tests its '
                'completeness; triplet test '
                'bounds third-order residuals.',
    'E1_ladder': 'injection layer 34 x 24 pairs, '
                 'repV 3065-identical; COS_LAD_34 '
                 '(a1 bit anchor vs 3066 npz row '
                 '34); PA34/PF34 lens probes; zH '
                 'captured at L34/L35 per pair; '
                 'captures x/a/m/p/act',
    'E2H_perhead': 'dzH 32 slices of 128; out_h '
                   '= dzH_h @ Wo slice; headlin '
                   'rel err; dAh TT projection; '
                   'medians (a9 bit anchor vs '
                   '3071 npz DAH medians); a8 '
                   'ZH34/ZH35 bit anchor vs 3071 '
                   'npz',
    'E3_interaction': 'top-8 = 3071 focal set '
                      '[20,7,1,14,26,0,2,24]; '
                      '(i) single-head resweep '
                      '24 pairs each (a10 bit '
                      'anchor vs 3071 r34); '
                      '(ii) 28 pairwise joint '
                      'swaps 24 pairs each; '
                      '(iii) full top-8 joint '
                      'swap; (iv) triplet (3 '
                      'most negative singles) '
                      'joint swap.  ALL effects '
                      'are median recovs (24-'
                      'pair medians, 3071 '
                      'definition); interaction '
                      'bookkeeping at the '
                      'median scale: I(A,B) = '
                      'R(AB) - R(A) - R(B) '
                      '(sample-level interaction '
                      'medians recorded as '
                      'observation only - summing '
                      '28 noisy sample-level '
                      'terms amplifies noise '
                      'sqrt(28) x and is useless '
                      'for prediction).  '
                      'Completeness: R_U8_PRED = '
                      'sum R_i + sum I_p vs '
                      'measured R_U8.',
    'gates': 'I significance max|I_BOOK| >= '
             '0.02 (~10-30 percent of '
             'single-head effects 0.06-0.22); '
             'second-order completeness '
             'PRED_ERR = |R_U8_PRED - R_U8| '
             '< 0.05 (~9 percent of the full '
             'recovery -0.572, ~ SE scale of '
             '24-pair medians); triplet '
             'residual ERR3 < 0.02 (~5 percent '
             'of the 3071 top-5 interaction '
             'total 0.396)',
    'verdict': 'setup fail -> setup_failed_'
               'interaction; a1 fail -> anchor_'
               'mismatch_3066_ladder; a8 fail '
               '-> anchor_mismatch_3071_zh; a9 '
               'fail -> anchor_mismatch_3071_'
               'dah; a10 fail -> anchor_mismatch'
               '_3071_r34; b8 fail -> block_'
               'output_mismatch; PRED_ERR >= '
               '0.05 -> higher_order_required; '
               'max|I_BOOK| < 0.02 -> '
               'additivity_holds; ERR3 < 0.02 '
               '-> pairwise_complete; else '
               'pairwise_partial.  I sign '
               '(sub-additivity), GQA-group / '
               'quarter / magnitude attribute '
               'correlations recorded, not '
               'gating.',
    'statistics_discipline': 'same-precision bit '
                             'anchors on frozen '
                             'seeds; TT logit-space '
                             '3065-identical; recov '
                             'referenced to med_c_'
                             '34; median-scale '
                             'bookkeeping: R(x) = '
                             'median cos - med_c_34 '
                             'over the same 24 pairs '
                             'for every condition; '
                             'pair order fixed '
                             'lexicographic; triplet '
                             'selected by the 3 most '
                             'negative single-head '
                             'recovs (no post-hoc '
                             'choice)',
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
    at o_proj inputs (last position; idx may be
    any head-slice index array for multi-head
    swaps).  Returns bf16 last-position captures
    zX/zA/zM/zP (NL, HID), zH (NL, NQW), zZ
    (NL, HID), zG/zU/zACT (NL, INTER) on GPU."""
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
a2_bit = 0.0
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
        lgt = F.linear(
            final_norm(zP[NL - 1]
                       .unsqueeze(0)), Wemb)
        lgt32 = F.linear(
            final_norm(zP[NL - 1])
            .float().unsqueeze(0), W32)
    d_bit = float(
        (lgt[0] - lg_gpu).abs().max())
    d32 = float(
        (lgt32[0] - lg_gpu.float())
        .abs().max())
    a2_bit = max(a2_bit, d_bit)
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
ZX34 = np.zeros((NP_USE, HID))
ZA34 = np.zeros((NP_USE, HID))
ZM34 = np.zeros((NP_USE, HID))
ZP34 = np.zeros((NP_USE, HID))
ZX35 = np.zeros((NP_USE, HID))
ZA35 = np.zeros((NP_USE, HID))
ZM35 = np.zeros((NP_USE, HID))
ZP35 = np.zeros((NP_USE, HID))
ZH34 = np.zeros((NP_USE, NQW))
ZH35 = np.zeros((NP_USE, NQW))
ZACT34 = np.zeros((NP_USE, INTER))
ZACT35 = np.zeros((NP_USE, INTER))
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
    zinj = torch.cat(
        [zX, zX + zA, zP], dim=0)
    zbase = base_z_gpu(base_i, nb)
    c = lens_cos(zinj, zbase, k)
    PA34[k] = c[NL + L34]
    PF34[k] = c[2 * NL + L34]
    ZX34[k] = zX[L34].double().cpu().numpy()
    ZA34[k] = zA[L34].double().cpu().numpy()
    ZM34[k] = zM[L34].double().cpu().numpy()
    ZP34[k] = zP[L34].double().cpu().numpy()
    ZX35[k] = zX[L35].double().cpu().numpy()
    ZA35[k] = zA[L35].double().cpu().numpy()
    ZM35[k] = zM[L35].double().cpu().numpy()
    ZP35[k] = zP[L35].double().cpu().numpy()
    ZH34[k] = zH[L34].double().cpu().numpy()
    ZH35[k] = zH[L35].double().cpu().numpy()
    ZACT34[k] = zACT[L34].double() \
        .cpu().numpy()
    ZACT35[k] = zACT[L35].double() \
        .cpu().numpy()
    d8 = float(np.max(np.abs(
        ZX35[k] - ZP34[k])))
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
has_opb = bool(
    layers[L34].self_attn.o_proj.bias
    is not None
    or layers[L35].self_attn.o_proj.bias
    is not None)
WoT34 = Wo34.t().reshape(NQ, HDIM, HID)
WoT35 = Wo35.t().reshape(NQ, HDIM, HID)
DAH34 = np.zeros((NP_USE, NQ))
DAH35 = np.zeros((NP_USE, NQ))
HEADLIN34 = np.zeros(NP_USE)
HEADLIN35 = np.zeros(NP_USE)


def head_decomp(dzH, WoT, Wo, dzA, t, tn):
    dhr = torch.tensor(
        np.ascontiguousarray(
            dzH.reshape(NQ, HDIM)),
        device='cuda').float()
    with torch.no_grad():
        outh = torch.einsum(
            'hd,hdj->hj', dhr, WoT)
        tot = F.linear(
            dhr.reshape(1, NQW), Wo)[0]
        un = F.linear(outh, W32)
    out_np = outh.float().cpu().numpy()
    tot_np = tot.double().cpu().numpy()
    un_np = un.double().cpu().numpy()
    dAh = (un_np @ t) / tn
    rel = float(np.linalg.norm(
        tot_np - dzA)) / max(
        float(np.linalg.norm(dzA)), 1e-12)
    return dAh, np.linalg.norm(
        out_np, axis=1), rel


for k in range(NP_USE):
    base_i = int(BASE_K[k])
    t = TT[k]
    tn = max(float(np.linalg.norm(t)), 1e-12)
    dzH34 = ZH34[k] - BH[base_i, L34]
    dzA34 = ZA34[k] - BA[base_i, L34]
    DAH34[k], _, HEADLIN34[k] = head_decomp(
        dzH34, WoT34, Wo34, dzA34, t, tn)
    dzH35 = ZH35[k] - BH[base_i, L35]
    dzA35 = ZA35[k] - BA[base_i, L35]
    DAH35[k], _, HEADLIN35[k] = head_decomp(
        dzH35, WoT35, Wo35, dzA35, t, tn)
DAH34_MED = np.median(DAH34, axis=0)
DAH35_MED = np.median(DAH35, axis=0)
hl34_med = float(np.median(HEADLIN34))
hl35_med = float(np.median(HEADLIN35))
log('E2H L34: headlin rel med=%.2e | top obs '
    'heads %s'
    % (hl34_med,
       np.argsort(DAH34_MED)[::-1][:6]
       .tolist()))
log('E2H L35: headlin rel med=%.2e'
    % hl35_med)
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


# ==== E3 interaction matrix ====
j71 = json.load(open(RES71, encoding='utf-8'))
TOP8 = [int(v) for v
        in j71['stats']['head']['top8']]
R34_71 = np.array(
    j71['stats']['head']['r34'],
    dtype=np.float64)
NT = len(TOP8)
PAIRS = list(itertools.combinations(
    range(NT), 2))
NP_P = len(PAIRS)
log('E3 top8=%s pairs=%d gate_I=%.3f '
    'gate_pred=%.3f gate_tri=%.3f'
    % (TOP8, NP_P, GATE_I, GATE_PRED,
       GATE_TRI))


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


# (i) single-head resweep on the top-8
C1H = np.full((NT, K3), np.nan)
for ti in range(NT):
    h = TOP8[ti]
    idx = HEAD_IDX[h]
    C1H[ti] = run_cond(lambda bi, idx=idx: {
        'attn_swaps': [(L34, idx,
                        BH[bi, L34][idx])]})
R1 = np.median(C1H, axis=1) - med_c_34
log('E3 singles recov: %s'
    % np.round(R1, 4).tolist())

# a10: singles bit anchor vs 3071 r34
a10_diff = None
a10_ok = False
if not SMOKE:
    a10_diff = float(np.max(np.abs(
        R1 - R34_71[TOP8])))
    a10_ok = bool(a10_diff == 0.0)
    log('a10 singles vs 3071 r34[top8]: '
        'max|d|=%.3e ok=%s'
        % (a10_diff, a10_ok))
else:
    log('a10 skipped (smoke)')

# (ii) pairwise joint swaps
C2H = np.full((NP_P, K3), np.nan)
for p_i, (a_, b_) in enumerate(PAIRS):
    ia = TOP8[a_]
    ib = TOP8[b_]
    idx = np.concatenate([HEAD_IDX[ia],
                          HEAD_IDX[ib]])
    C2H[p_i] = run_cond(lambda bi, idx=idx: {
        'attn_swaps': [(L34, idx,
                        BH[bi, L34][idx])]})
    log('E3 pair %s (h%02d+h%02d) med=%.4f'
        % ((a_, b_), ia, ib,
           float(np.median(C2H[p_i]))))
R2 = np.median(C2H, axis=1) - med_c_34
# median-scale interaction bookkeeping
pa_i = np.array([a_ for a_, b_ in PAIRS])
pb_i = np.array([b_ for a_, b_ in PAIRS])
I_BOOK = R2 - R1[pa_i] - R1[pb_i]
# sample-level interaction medians
# (observation only: summing 28 noisy
# sample-level terms amplifies noise)
I_PAIR = (C2H - C1H[pa_i] - C1H[pb_i]
          + med_c_34)
I_SMED = np.median(I_PAIR, axis=1)
n_ipos = int((I_BOOK > 0).sum())
n_ineg = int((I_BOOK < 0).sum())
i_max = float(np.max(np.abs(I_BOOK)))
i_argmax = int(np.argmax(np.abs(I_BOOK)))
log('E3 interactions (median book): '
    'n>0=%d n<0=%d max|I|=%.4f at pair %s '
    '(h%02d+h%02d)'
    % (n_ipos, n_ineg, i_max,
       PAIRS[i_argmax],
       TOP8[PAIRS[i_argmax][0]],
       TOP8[PAIRS[i_argmax][1]]))
log('E3 I_BOOK: %s'
    % np.round(I_BOOK, 4).tolist())
log('E3 I sample-medians (obs): %s'
    % np.round(I_SMED, 4).tolist())

# (iii) full top-8 joint swap
idx_u8 = np.concatenate(
    [HEAD_IDX[h] for h in TOP8])
C_U8 = run_cond(lambda bi, idx=idx_u8: {
    'attn_swaps': [(L34, idx,
                    BH[bi, L34][idx])]})
R_U8 = float(np.median(C_U8) - med_c_34)
log('E3 top-8 joint: med=%.4f recov=%+.4f'
    % (float(np.median(C_U8)), R_U8))

# median-scale second-order prediction
R_U8_PRED = float(R1.sum() + I_BOOK.sum())
PRED_ERR = float(abs(R_U8_PRED - R_U8))
log('E3 2nd-order book: sum singles=%+.4f '
    'sum I=%+.4f pred=%+.4f vs measured '
    '%+.4f | PRED_ERR=%.4f (gate %.3f)'
    % (float(R1.sum()), float(I_BOOK.sum()),
       R_U8_PRED, R_U8, PRED_ERR, GATE_PRED))

# (iv) triplet test: 3 most negative singles
t3 = list(np.argsort(R1)[:3])
idx_t3 = np.concatenate(
    [HEAD_IDX[TOP8[ti]] for ti in t3])
C_T3 = run_cond(lambda bi, idx=idx_t3: {
    'attn_swaps': [(L34, idx,
                    BH[bi, L34][idx])]})
R_T3 = float(np.median(C_T3) - med_c_34)
t3_pairs = [p_i for p_i, (a_, b_)
            in enumerate(PAIRS)
            if a_ in t3 and b_ in t3]
R_T3_PRED = float(R1[t3].sum()
                  + I_BOOK[t3_pairs].sum())
ERR3 = float(abs(R_T3_PRED - R_T3))
log('E3 triplet %s (h%s): measured %+.4f '
    'pred %+.4f | ERR3=%.4f (gate %.3f)'
    % (t3, [TOP8[ti] for ti in t3], R_T3,
       R_T3_PRED, ERR3, GATE_TRI))

# ==== E4 attribute correlates (recorded) ====


def _rank(a):
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty(len(a), dtype=np.int64)
    inv[sorter] = np.arange(len(a))
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (cnt[dense]
                  + cnt[dense - 1] + 1.0)


def spearman(x, y):
    if len(x) != len(y) or len(x) < 3:
        return float('nan')
    rx = _rank(x)
    ry = _rank(y)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx < 1e-12 or sy < 1e-12:
        return float('nan')
    return float(np.mean(
        (rx - rx.mean()) * (ry - ry.mean()))
        / (sx * sy))


pa = pa_i
pb = pb_i
same_gqa = np.array(
    [TOP8[a_] // NG == TOP8[b_] // NG
     for a_, b_ in zip(pa, pb)])
same_q = np.array(
    [TOP8[a_] // 8 == TOP8[b_] // 8
     for a_, b_ in zip(pa, pb)])
sp_I_r1 = spearman(np.abs(R1[pa])
                   * np.abs(R1[pb]),
                   np.abs(I_BOOK))
sp_I_dah = spearman(
    np.abs(DAH34_MED[TOP8][pa])
    * np.abs(DAH34_MED[TOP8][pb]),
    np.abs(I_BOOK))
ig_gqa = [float(np.mean(I_BOOK[same_gqa])),
          float(np.mean(I_BOOK[~same_gqa]))]
ig_q = [float(np.mean(I_BOOK[same_q])),
        float(np.mean(I_BOOK[~same_q]))]
log('E4 correlates: spearman(|r1|prod,|I|)='
    '%.3f spearman(|dah|prod,|I|)=%.3f | '
    'I mean sameGQA=%.4f diffGQA=%.4f | '
    'sameQ=%.4f diffQ=%.4f'
    % (sp_I_r1, sp_I_dah, ig_gqa[0],
       ig_gqa[1], ig_q[0], ig_q[1]))

# ==== verdict ====
setup_ok = bool(b0_ok and b1_ok and b3_ok
                and b4_ok and b6_ok and b7_ok
                and aref_ok)
inter_sig = bool(i_max >= GATE_I)
if not setup_ok:
    verdict = 'setup_failed_interaction'
elif (not SMOKE) and (not a1_ok):
    verdict = 'anchor_mismatch_3066_ladder'
elif (not SMOKE) and (not a8_ok):
    verdict = 'anchor_mismatch_3071_zh'
elif (not SMOKE) and (not a9_ok):
    verdict = 'anchor_mismatch_3071_dah'
elif (not SMOKE) and (not a10_ok):
    verdict = 'anchor_mismatch_3071_r34'
elif (not SMOKE) and (not b8_ok):
    verdict = 'block_output_mismatch'
elif PRED_ERR >= GATE_PRED:
    verdict = 'higher_order_required'
elif not inter_sig:
    verdict = 'additivity_holds'
elif ERR3 < GATE_TRI:
    verdict = 'pairwise_complete'
else:
    verdict = 'pairwise_partial'
log('VERDICT: %s (setup_ok=%s a1_ok=%s a8_ok='
    '%s a9_ok=%s a10_ok=%s b8_ok=%s i_max=%.4f '
    'PRED_ERR=%.4f ERR3=%.4f smoke=%s)'
    % (verdict, setup_ok, a1_ok, a8_ok, a9_ok,
       a10_ok, b8_ok, i_max, PRED_ERR,
       ERR3, SMOKE))

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
    'HEADLIN35': HEADLIN35,
    'DAH34_MED': DAH34_MED,
    'DAH35_MED': DAH35_MED,
    'HL34_MED': np.float64(hl34_med),
    'HL35_MED': np.float64(hl35_med),
    'HAS_OPB': np.bool_(has_opb),
    'TOP8': np.array(TOP8, dtype=np.int64),
    'C1H': C1H, 'R1': R1,
    'PAIRS': np.array(PAIRS,
                      dtype=np.int64),
    'C2H': C2H, 'R2': R2,
    'I_BOOK': I_BOOK, 'I_PAIR': I_PAIR,
    'I_SMED': I_SMED,
    'C_U8': C_U8, 'R_U8': np.float64(R_U8),
    'R_U8_PRED': np.float64(R_U8_PRED),
    'PRED_ERR': np.float64(PRED_ERR),
    'T3': np.array(t3, dtype=np.int64),
    'C_T3': C_T3, 'R_T3': np.float64(R_T3),
    'R_T3_PRED': np.float64(R_T3_PRED),
    'ERR3': np.float64(ERR3),
    'N_IPOS': np.int64(n_ipos),
    'N_INEG': np.int64(n_ineg),
    'SP_I_R1': np.float64(sp_I_r1),
    'SP_I_DAH': np.float64(sp_I_dah),
    'IG_GQA': np.array(ig_gqa),
    'IG_Q': np.array(ig_q),
    'TT': TT.astype(np.float32),
    'LG': LG.astype(np.float32),
}
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
        'hl35_med': f64(hl35_med),
        'has_o_proj_bias': has_opb,
        'top8': TOP8,
        'r1': [f64(v) for v in R1],
        'r2': [f64(v) for v in R2],
        'pairs': [[int(a_), int(b_)]
                  for a_, b_ in PAIRS],
        'i_book': [f64(v) for v in I_BOOK],
        'i_smed': [f64(v) for v in I_SMED],
        'i_pair_max': f64(i_max),
        'i_argmax_pair': [int(PAIRS[i_argmax][0]),
                          int(PAIRS[i_argmax][1])],
        'n_ipos': int(n_ipos),
        'n_ineg': int(n_ineg),
        'r_u8': f64(R_U8),
        'r_u8_pred': f64(R_U8_PRED),
        'pred_err': f64(PRED_ERR),
        't3': [int(v) for v in t3],
        'r_t3': f64(R_T3),
        'r_t3_pred': f64(R_T3_PRED),
        'err3': f64(ERR3),
        'inter_significant': inter_sig,
        'sp_I_r1prod': f64(sp_I_r1),
        'sp_I_dahprod': f64(sp_I_dah),
        'i_gqa_means': [f64(v) for v in ig_gqa],
        'i_q_means': [f64(v) for v in ig_q],
    },
    'anchors': {
        'a1_diff': a1_diff, 'a1_ok': a1_ok,
        'a8_diff': a8_diff, 'a8_ok': a8_ok,
        'a9_diff': a9_diff, 'a9_ok': a9_ok,
        'a10_diff': a10_diff, 'a10_ok': a10_ok,
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
