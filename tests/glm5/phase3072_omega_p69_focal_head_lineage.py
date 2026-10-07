# -*- coding: utf-8 -*-
"""Phase 3072: Omega-P69 focal head deep anatomy
(qwen3-4b single model bf16).

Question (3072 A, menu of 3071): the L34
same-block attention suppressor resolves to a
focal 5-head core (h20/7/1/14/26 from 3071 R34).
WHY are these heads the responders?  Three
sub-questions: (i) LISTENER - do they attend to
the injected positions (V injection at L34
positions 0..3, readout at last position)?
(ii) LINEAGE - is each head's observed write
dzH_h exactly the linear image of the injected
V signature, dzH_h = sum_p w_h(p) * dV_p[g(h)]
with w_h the attention weights (last row) and
g(h) = h // 4 the GQA value group?  (iii) OV -
which vocabulary directions does each focal
head's write promote (W32 @ W_O slice @ dzH_h)?

Protocol fact used (preregistered): the 3065
V-clamp replaces v_proj outputs at ALL layers
with base values except L34 positions 0..3;
the L34 block input is unchanged (b4 bit 0.0
in 3070/3071), so L34 Q/K are unchanged and
the attention WEIGHTS at L34 must be bit-
identical between base and inj forwards
(b10 anchor, measured via output_attentions
probes; b9 anchor: output_attentions does not
change logits, bit 0.0).  Under b10 the
lineage identity is exact up to bf16 matmul
rounding.

E1 inj@34 ladder: 24 pairs, repV protocol
3065/3066/3067/3069/3070/3071-identical
(a1 anchor COS_LAD row 34 bit vs 3066 npz;
a7 E2 cproj medians bit vs 3070 refs; a8
ZH34/ZH35 bit vs the 3071 npz; a9 per-head
obs medians bit vs the 3071 npz; med_c_34
reference assert; aref PA34/PF34 bit vs
3069; zH captured at L34/L35).
E2 3070-identical diff-chain decomposition.
E2H per-head observation decomposition
(o_proj linear, no bias; headlin rel err;
per-head TT projection dAh; medians).
E3 attention probes: output_attentions=True
forwards on the 24 base prompts and the 24
inj forwards; last query row captured at
L34/L35; b9 logits bit anchor; b10 L34
weights bit anchor (base vs inj); L35
weight shift recorded (expected nonzero -
descriptive).
E4 lineage: pred_h = sum_p w_h(p) * dV_p
[g(h)] from banks; joint rel error
||obs - pred|| / ||obs|| per pair; per-head
cos(pred, obs) medians.
E5 listener: attention mass at positions
0..3 from the last position, per head;
Spearman vs |R34| (3071 causal) and vs
|dAh| (3071 obs); L35 same vs |R35|.
E6 OV circuit: for focal5 + obs-top heads
h6/h24: per-pair dlogit_h = W32 @ out_h;
median across pairs; top-10 tokens; median
rank of the target token and of the TT-
argmax token.
verdict: setup fail -> setup_failed_focal;
a1 fail -> anchor_mismatch_3066_ladder; a7
fail -> anchor_mismatch_3070_e2; a8 fail ->
anchor_mismatch_3071_zh; a9 fail ->
anchor_mismatch_3071_dah; b9 fail ->
attn_probe_mismatch; b10 fail ->
attn_weight_shift_mismatch; b8 fail ->
block_output_mismatch; med_rel_joint > 0.05
-> lineage_linear_fail; else min over
focal5 of per-head median cos >= 0.99 ->
focal_lineage_full; elif min over top8 >=
0.99 -> focal_lineage_top8; elif median
over all 32 heads >= 0.99 ->
lineage_linear_allheads; else
lineage_partial.  Listener/OV results are
recorded, not gating.
memory discipline: single model; banks
fp64 CPU; W32 resident; del + gc +
empty_cache at the end.
"""
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

PHASE = 3072
NAME = 'omega_p69_focal_head_lineage'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3072', NAME)
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
MEDC34_REF = 0.1487826048372403
PA34_REF = 0.29685845971107483
PF34_REF = 0.2371114194393158
D34A_REF = 639.8104587682893
D34M_REF = -95.91242909117048
D34O_REF = 735.8797832320352
D35A_REF = -14.567961614698053
D35M_REF = -924.1991208657355
D35O_REF = -191.78616219035163
NPZ71 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3071', 'omega_p68_attn_head_decomp',
    'omega_p68_attn_head_decomp.npz')

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
            '3067/3069/3070/3071-identical (a1 '
            'bit anchor vs 3066 npz COS_LAD row '
            '34, hard; med_c_34 reference '
            'assert; aref PA34/PF34 bit vs '
            '3069; a7 E2 cproj medians bit vs '
            '3070 refs; a8 ZH34/ZH35 bit vs '
            '3071 npz; a9 per-head obs medians '
            'bit vs 3071 npz); attention probes '
            'output_attentions=True (b9 logits '
            'bit anchor; b10 L34 weights base/'
            'inj bit anchor); lens probes fp32 '
            'W32 resident; verdict scalars '
            'fp64 cosv; smoke mode optional '
            '(SMOKE=1: 8 pairs, a1/a7/a8/a9 '
            'skipped, b9/b10 active)',
    'question': '3072 A (menu of 3071): why are '
                'h20/7/1/14/26 the focal '
                'responders?  (i) listener: '
                'attention mass at the injected '
                'positions 0..3; (ii) lineage: '
                'dzH_h = sum_p w_h(p) * dV_p[g(h)]'
                ' exactly (V-clamp protocol + b4 '
                'dzX_34 = 0 + b10 weights bit-'
                'identical); (iii) OV: which '
                'tokens each focal head write '
                'promotes.',
    'E1_ladder': 'injection layer 34 x 24 pairs, '
                 'repV 3065-identical; COS_LAD_34 '
                 '(a1 bit anchor vs 3066 npz row '
                 '34); PA34/PF34 lens probes; zH '
                 'captured at L34/L35 per pair',
    'E2_decomposition': '3070/3071-identical '
                        'diff chain (a7 bit '
                        'anchor); E2H per-head '
                        'o_proj-linear '
                        'decomposition (a9 bit '
                        'anchor vs 3071 '
                        'medians)',
    'E3_probes': 'output_attentions=True on 24 '
                 'base prompts + 24 inj '
                 'forwards; last query row at '
                 'L34/L35; b9 = probe logits bit '
                 '0.0 vs bank (oatt numerics '
                 'unchanged); b10 = L34 weights '
                 'bit 0.0 between base and inj '
                 '(V-only injection leaves Q/K '
                 'unchanged); L35 shift '
                 'descriptive',
    'E4_lineage': 'pred_h = sum_p w_h(p) * '
                  'dV_p[g(h)], g(h) = h // 4, '
                  'dV from banks (prefix minus '
                  'base at positions 0..3); '
                  'joint rel err per pair; '
                  'per-head cos medians; gates: '
                  'med_rel_joint > 0.05 -> '
                  'lineage_linear_fail; focal5 '
                  'min med-cos >= 0.99 -> '
                  'focal_lineage_full; top8 min '
                  '>= 0.99 -> focal_lineage_'
                  'top8; all-32 median >= 0.99 '
                  '-> lineage_linear_allheads; '
                  'else lineage_partial',
    'E5_listener': 'attention mass at positions '
                   '0..3 from last position per '
                   'head; Spearman vs |R34| (3071 '
                   'npz) and vs |dAh|; L35 mass '
                   'vs |R35| recorded',
    'E6_ov': 'focal5 + h6 + h24; per-pair '
             'dlogit_h = W32 @ out_h; median '
             'across pairs; top-10 tokens; '
             'median rank of target token and '
             'TT-argmax token; recorded, not '
             'gating',
    'anchors': 'a1 row 34 bit 0.0 vs 3066 npz '
               '(hard); med_c_34 reference '
               'assert 0.1487826048372403 '
               '(hard); a7 E2 medians bit vs '
               '3070 refs (hard); a8 ZH34/ZH35 '
               'bit 0.0 vs 3071 npz (hard); a9 '
               'DAH34_MED/DAH35_MED bit 0.0 vs '
               '3071 npz (hard); b9 probe '
               'logits bit 0.0 (hard); b10 L34 '
               'weights bit 0.0 base vs inj '
               '(hard); aref PA34/PF34 bit vs '
               '3069 (hard); b8 dzX_35 = dzP_34 '
               'bit 0.0 (hard); b0 recapture '
               'bit 0.0 (incl. BH); b1 sham bit '
               '0.0; b3 finite; b4 delta-x at '
               'L34 bit 0.0; b5 act = silu(g)*u '
               'bf16 (recorded, L35); b6 (x+a)'
               '+m = h2 bf16 bit 0.0; b7a/b7c '
               'identity self-swaps bit 0.0',
    'verdict': 'setup fail -> setup_failed_'
               'focal; a1 fail -> anchor_mismatch'
               '_3066_ladder; a7 fail -> anchor'
               '_mismatch_3070_e2; a8 fail -> '
               'anchor_mismatch_3071_zh; a9 '
               'fail -> anchor_mismatch_3071_'
               'dah; b9 fail -> attn_probe_'
               'mismatch; b10 fail -> attn_'
               'weight_shift_mismatch; b8 fail '
               '-> block_output_mismatch; '
               'med_rel_joint > 0.05 -> '
               'lineage_linear_fail; else min '
               'focal5 med-cos >= 0.99 -> '
               'focal_lineage_full; elif min '
               'top8 >= 0.99 -> focal_lineage_'
               'top8; elif all-32 median >= '
               '0.99 -> lineage_linear_'
               'allheads; else lineage_partial',
    'statistics_discipline': 'same-precision bit '
                             'anchors on frozen '
                             'seeds; TT logit-space '
                             '3065-identical; '
                             'lineage prediction is '
                             'exact in real '
                             'arithmetic (bf16 '
                             'rounding only); '
                             'listener/OV recorded '
                             'not gating',
    'memory_discipline': 'single model; banks '
                         'fp64 CPU; DLG fp32; '
                         'W32 resident; del + gc '
                         '+ empty_cache at end',
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

z71 = np.load(NPZ71)
R34_3071 = z71['R34'].astype(np.float64)
R35_3071 = z71['R35'].astype(np.float64)
TOP8_3071 = z71['TOP8'].astype(np.int64)
FOCAL5 = TOP8_3071[:5].copy()
log('3071 npz loaded: focal5=%s top8=%s'
    % (FOCAL5.tolist(), TOP8_3071.tolist()))


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
PROBE_ATT = {'ok': False, 'w34': None,
             'w35': None}
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
                attn_swaps=None, oatt=False):
    """3065-identical V replacement; optional
    act swaps [(layer, idx, vals)] at down_proj
    inputs and attn swaps [(layer, idx, vals)]
    at o_proj inputs (last position).  oatt=True
    requests output_attentions and stashes the
    L34/L35 last query rows into PROBE_ATT.
    Returns bf16 last-position captures zX/zA/
    zM/zP (NL, HID), zH (NL, NQW), zZ (NL, HID),
    zG/zU/zACT (NL, INTER) on GPU."""
    reset_all()
    PROBE_ATT['ok'] = False
    PROBE_ATT['w34'] = None
    PROBE_ATT['w35'] = None
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
            use_cache=False,
            output_attentions=oatt)
    if oatt:
        atts = getattr(out, 'attentions', None)
        if atts is not None and len(atts) == NL:
            PROBE_ATT['ok'] = True
            PROBE_ATT['w34'] = atts[L34][
                0, :, -1, :].double() \
                .cpu().numpy()
            PROBE_ATT['w35'] = atts[L35][
                0, :, -1, :].double() \
                .cpu().numpy()
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
    '%.3e (<=0.125 ok)'
    % a2_max)

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

# a8/a9: bit anchors vs the 3071 npz
a8_diff = None
a8_ok = False
a9_diff = None
a9_ok = False
if not SMOKE:
    a8_diff = max(float(np.max(np.abs(
        ZH34 - z71['ZH34']))),
        float(np.max(np.abs(
            ZH35 - z71['ZH35']))))
    a8_ok = bool(a8_diff == 0.0)
    log('a8 ZH34/ZH35 vs 3071 npz: max|d|='
        '%.3e ok=%s' % (a8_diff, a8_ok))
else:
    log('a8 skipped (smoke)')

# ==== E2 observed diff decomposition ====
Wd34 = layers[L34].mlp.down_proj.weight \
    .detach().float()
CPX34 = np.zeros(NP_USE)
CPA34 = np.zeros(NP_USE)
CPM34 = np.zeros(NP_USE)
CPP34 = np.zeros(NP_USE)
CPA35 = np.zeros(NP_USE)
CPM35 = np.zeros(NP_USE)
CPP35 = np.zeros(NP_USE)
NX34 = np.zeros(NP_USE)
NA34 = np.zeros(NP_USE)
NM34 = np.zeros(NP_USE)
NP34o = np.zeros(NP_USE)
NA35 = np.zeros(NP_USE)
NM35 = np.zeros(NP_USE)
NP35o = np.zeros(NP_USE)
LIN34 = np.zeros(NP_USE)
for k in range(NP_USE):
    base_i = int(BASE_K[k])
    t = TT[k]
    tn = max(float(np.linalg.norm(t)), 1e-12)
    dx = ZX34[k] - BX[base_i, L34]
    da = ZA34[k] - BA[base_i, L34]
    dm = ZM34[k] - BM[base_i, L34]
    dp = ZP34[k] - PB[base_i, L34,
                      int(LENS[base_i]) - 1]
    da5 = ZA35[k] - BA[base_i, L35]
    dm5 = ZM35[k] - BM[base_i, L35]
    dp5 = ZP35[k] - PB[base_i, L35,
                       int(LENS[base_i]) - 1]
    dact34 = ZACT34[k] - BACT34[base_i]
    for nm, v, arr in (
            ('x34', dx, CPX34),
            ('a34', da, CPA34),
            ('m34', dm, CPM34),
            ('p34', dp, CPP34),
            ('a35', da5, CPA35),
            ('m35', dm5, CPM35),
            ('p35', dp5, CPP35)):
        vt = torch.tensor(
            np.ascontiguousarray(v),
            device='cuda').float()
        with torch.no_grad():
            u = F.linear(
                vt.unsqueeze(0), W32)[0]
        u_np = u.double().cpu().numpy()
        arr[k] = float(u_np @ t) / tn
    NX34[k] = float(np.linalg.norm(dx))
    NA34[k] = float(np.linalg.norm(da))
    NM34[k] = float(np.linalg.norm(dm))
    NP34o[k] = float(np.linalg.norm(dp))
    NA35[k] = float(np.linalg.norm(da5))
    NM35[k] = float(np.linalg.norm(dm5))
    NP35o[k] = float(np.linalg.norm(dp5))
    with torch.no_grad():
        mrec = F.linear(
            torch.tensor(
                np.ascontiguousarray(dact34),
                device='cuda').float()
            .unsqueeze(0), Wd34)[0]
    LIN34[k] = cosv(
        mrec.double().cpu().numpy(), dm)
d34_attn = float(np.median(CPA34))
d34_mlp = float(np.median(CPM34))
d34_out = float(np.median(CPP34))
d34_x = float(np.median(CPX34))
d35_attn = float(np.median(CPA35))
d35_mlp = float(np.median(CPM35))
d35_out = float(np.median(CPP35))
lin34_med = float(np.median(LIN34))
del Wd34
gc.collect()
torch.cuda.empty_cache()
log('E2 L34: cproj x=%.1f a=%.1f m=%.1f p=%.1f '
    '| norms x=%.3f a=%.3f m=%.3f p=%.3f '
    '| lincheck=%.4f'
    % (d34_x, d34_attn, d34_mlp, d34_out,
       float(np.median(NX34)),
       float(np.median(NA34)),
       float(np.median(NM34)),
       float(np.median(NP34o)), lin34_med))
log('E2 L35: cproj a=%.1f m=%.1f p=%.1f '
    '| norms a=%.3f m=%.3f p=%.3f'
    % (d35_attn, d35_mlp, d35_out,
       float(np.median(NA35)),
       float(np.median(NM35)),
       float(np.median(NP35o))))

# a7: E2 medians bit anchor vs 3070 refs
a7_diff = None
a7_ok = False
if not SMOKE:
    a7_diff = max(
        abs(d34_attn - D34A_REF),
        abs(d34_mlp - D34M_REF),
        abs(d34_out - D34O_REF),
        abs(d35_attn - D35A_REF),
        abs(d35_mlp - D35M_REF),
        abs(d35_out - D35O_REF))
    a7_ok = bool(a7_diff == 0.0)
    log('a7 E2 cproj medians vs 3070 refs: '
        'max|d|=%.3e ok=%s'
        % (a7_diff, a7_ok))
else:
    log('a7 skipped (smoke)')

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
DOH34N = np.zeros((NP_USE, NQ))
DOH35N = np.zeros((NP_USE, NQ))
DHH34N = np.zeros((NP_USE, NQ))
DHH35N = np.zeros((NP_USE, NQ))
HEADLIN34 = np.zeros(NP_USE)
HEADLIN35 = np.zeros(NP_USE)
OUTH34 = np.zeros((NP_USE, NQ, HID))


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
    out_np = outh.double().cpu().numpy()
    tot_np = tot.double().cpu().numpy()
    un_np = un.double().cpu().numpy()
    dAh = (un_np @ t) / tn
    rel = float(np.linalg.norm(
        tot_np - dzA)) / max(
        float(np.linalg.norm(dzA)), 1e-12)
    return dAh, np.linalg.norm(
        out_np, axis=1), np.linalg.norm(
        dzH.reshape(NQ, HDIM), axis=1), rel, \
        out_np


for k in range(NP_USE):
    base_i = int(BASE_K[k])
    t = TT[k]
    tn = max(float(np.linalg.norm(t)), 1e-12)
    dzH34 = ZH34[k] - BH[base_i, L34]
    dzA34 = ZA34[k] - BA[base_i, L34]
    DAH34[k], DOH34N[k], DHH34N[k], \
        HEADLIN34[k], OUTH34[k] = head_decomp(
            dzH34, WoT34, Wo34, dzA34, t, tn)
    dzH35 = ZH35[k] - BH[base_i, L35]
    dzA35 = ZA35[k] - BA[base_i, L35]
    DAH35[k], DOH35N[k], DHH35N[k], \
        HEADLIN35[k], _ = head_decomp(
            dzH35, WoT35, Wo35, dzA35, t, tn)
DAH34_MED = np.median(DAH34, axis=0)
DAH35_MED = np.median(DAH35, axis=0)
DOH34_MED = np.median(DOH34N, axis=0)
DOH35_MED = np.median(DOH35N, axis=0)
DHH34_MED = np.median(DHH34N, axis=0)
DHH35_MED = np.median(DHH35N, axis=0)
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

# a9: per-head obs medians bit vs 3071 npz
if not SMOKE:
    a9_diff = max(float(np.max(np.abs(
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


# ==== E3 attention probes ====
ATT34B = np.full((NP_USE, NQ, NMAX), np.nan)
ATT35B = np.full((NP_USE, NQ, NMAX), np.nan)
ATT34I = np.full((NP_USE, NQ, NMAX), np.nan)
ATT35I = np.full((NP_USE, NQ, NMAX), np.nan)
b9_diff = 0.0
b10_diff = 0.0
b10_35_max = 0.0
probe_ok = True
for k in range(NP_USE):
    b, base_i, pref_i, off = pair_idx(k)
    nb = int(LENS[base_i])
    n = int(LENS[base_i])
    lgp = forward_gen(
        assembled[base_i]['ids'],
        oatt=True)[0]
    b9_diff = max(b9_diff, float(np.max(
        np.abs(lgp - LG[base_i]))))
    if not PROBE_ATT['ok']:
        probe_ok = False
        continue
    ATT34B[k, :, :n] = PROBE_ATT['w34']
    ATT35B[k, :, :n] = PROBE_ATT['w35']
    repV = VB[base_i, :, :nb, :].copy()
    repV[L34, ridx, :] = VB[pref_i][
        L34, off + ridx, :]
    forward_gen(
        assembled[base_i]['ids'],
        repl=repV, oatt=True)
    if not PROBE_ATT['ok']:
        probe_ok = False
        continue
    ATT34I[k, :, :n] = PROBE_ATT['w34']
    ATT35I[k, :, :n] = PROBE_ATT['w35']
    b10_diff = max(b10_diff, float(np.max(
        np.abs(ATT34I[k, :, :n]
               - ATT34B[k, :, :n]))))
    b10_35_max = max(b10_35_max, float(
        np.max(np.abs(ATT35I[k, :, :n]
                      - ATT35B[k, :, :n]))))
b9_diff = float(b9_diff)
b10_diff = float(b10_diff)
b10_35_max = float(b10_35_max)
b9_ok = bool(probe_ok and b9_diff == 0.0)
b10_ok = bool(probe_ok and b10_diff == 0.0)
log('b9 probe logits diff=%.3e ok=%s '
    '(probe_ok=%s)'
    % (b9_diff, b9_ok, probe_ok))
log('b10 L34 weights base-vs-inj diff=%.3e '
    'ok=%s | L35 shift (descriptive)=%.3e'
    % (b10_diff, b10_ok, b10_35_max))

# ==== E4 lineage ====
NG = NQ // KV_HEAD
OBS_H = np.zeros((NP_USE, NQ, HDIM))
PRED_H = np.zeros((NP_USE, NQ, HDIM))
REL_JOINT = np.full(NP_USE, np.nan)
COS_H = np.full((NP_USE, NQ), np.nan)
for k in range(NP_USE):
    b, base_i, pref_i, off = pair_idx(k)
    obs = (ZH34[k]
           - BH[base_i, L34]).reshape(NQ, HDIM)
    OBS_H[k] = obs
    dv = (VB[pref_i, L34, off:off + FRONT, :]
          - VB[base_i, L34, :FRONT, :])
    w = ATT34B[k, :, :FRONT]
    pred = np.zeros((NQ, HDIM))
    for h in range(NQ):
        g = h // NG
        sl = slice(g * HDIM, (g + 1) * HDIM)
        pred[h] = np.tensordot(
            w[h], dv[:, sl], axes=(0, 0))
    PRED_H[k] = pred
    REL_JOINT[k] = float(np.linalg.norm(
        obs - pred)) / max(
        float(np.linalg.norm(obs)), 1e-12)
    for h in range(NQ):
        COS_H[k, h] = cosv(pred[h], obs[h])
med_rel = float(np.median(REL_JOINT))
cos_med_h = np.array([
    np.nanmedian(COS_H[:, h])
    if np.isfinite(COS_H[:, h]).any()
    else np.nan for h in range(NQ)])
focal5_cos = cos_med_h[FOCAL5]
top8_cos = cos_med_h[TOP8_3071]
all32_med = float(np.nanmedian(cos_med_h))
rel_focal5 = []
for h in FOCAL5:
    rr = REL_H = np.linalg.norm(
        OBS_H[:, h, :] - PRED_H[:, h, :],
        axis=1) / np.maximum(
        np.linalg.norm(OBS_H[:, h, :], axis=1),
        1e-12)
    rel_focal5.append(float(np.median(rr)))
log('E4 lineage: med_rel_joint=%.4f | '
    'focal5 cos med=%s | focal5 rel med=%s | '
    'top8 min cos=%.4f | all32 med cos=%.4f'
    % (med_rel,
       np.round(focal5_cos, 5).tolist(),
       np.round(rel_focal5, 4).tolist(),
       float(np.nanmin(top8_cos)), all32_med))

# ==== E5 listener ====
MH34 = np.nansum(ATT34B[:, :, :FRONT], axis=2)
MH35 = np.nansum(ATT35B[:, :, :FRONT], axis=2)
m34_med = np.median(MH34, axis=0)
m35_med = np.median(MH35, axis=0)


def _rank(a):
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty(len(a), dtype=np.int64)
    inv[sorter] = np.arange(len(a))
    s = a[sorter]
    obsm = np.r_[True, s[1:] != s[:-1]]
    dense = obsm.cumsum()[inv]
    cnt = np.r_[np.nonzero(obsm)[0], len(obsm)]
    return 0.5 * (cnt[dense]
                  + cnt[dense - 1] + 1.0)


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
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


sp_m_absR = spearman(m34_med, np.abs(R34_3071))
sp_m_absD = spearman(m34_med,
                     np.abs(DAH34_MED))
sp_m35_absR35 = spearman(m35_med,
                         np.abs(R35_3071))
log('E5 listener: m34 med=%s | spearman(m34,'
    '|R34|)=%.3f spearman(m34,|dAh|)=%.3f | '
    'm35 med=%s spearman(m35,|R35|)=%.3f'
    % (np.round(m34_med, 3).tolist(),
       sp_m_absR, sp_m_absD,
       np.round(m35_med, 3).tolist(),
       sp_m35_absR35))

# ==== E6 OV circuit ====
SEL = sorted(set(FOCAL5.tolist()
                 + [6, 24]))
NS = len(SEL)
DLG = np.zeros((NP_USE, NS, NVOC),
               dtype=np.float32)
for k in range(NP_USE):
    osel = OUTH34[k][SEL]
    with torch.no_grad():
        dlg = F.linear(
            torch.tensor(osel,
                         device='cuda',
                         dtype=torch.float32),
            W32)
    DLG[k] = dlg.float().cpu().numpy()
OV_MED = np.median(DLG, axis=0)
TTARG = np.array([int(np.argmax(TT[k]))
                  for k in range(NP_USE)])
top_ids = np.zeros((NS, 10), dtype=np.int64)
top_vals = np.zeros((NS, 10))
tgrank = np.zeros(NS)
tttop_rank = np.zeros(NS)
for i in range(NS):
    order = np.argsort(-OV_MED[i])
    top_ids[i] = order[:10]
    top_vals[i] = OV_MED[i][order[:10]]
    rr = []
    rr2 = []
    for k in range(NP_USE):
        b = int(bidx[k])
        tg = word_tok[TARGETS[b]]
        o = np.argsort(-DLG[k, i])
        rr.append(int(np.where(o == tg)[0][0]))
        rr2.append(int(np.where(
            o == TTARG[k])[0][0]))
    tgrank[i] = float(np.median(rr))
    tttop_rank[i] = float(np.median(rr2))
log('E6 OV: sel=%s | target rank med=%s | '
    'ttarg rank med=%s'
    % (SEL, tgrank.tolist(),
       tttop_rank.tolist()))
for i, h in enumerate(SEL):
    toks = [tok.decode([int(t)]).strip()
            or repr(int(t))
            for t in top_ids[i]]
    log('E6 OV h=%02d top10: %s'
        % (h, toks))

# ==== verdict ====
setup_ok = bool(b0_ok and b1_ok and b3_ok
                and b4_ok and b6_ok and b7_ok
                and aref_ok)
if not setup_ok:
    verdict = 'setup_failed_focal'
elif (not SMOKE) and (not a1_ok):
    verdict = 'anchor_mismatch_3066_ladder'
elif (not SMOKE) and (not a7_ok):
    verdict = 'anchor_mismatch_3070_e2'
elif (not SMOKE) and (not a8_ok):
    verdict = 'anchor_mismatch_3071_zh'
elif (not SMOKE) and (not a9_ok):
    verdict = 'anchor_mismatch_3071_dah'
elif (not SMOKE) and (not b9_ok):
    verdict = 'attn_probe_mismatch'
elif (not SMOKE) and (not b10_ok):
    verdict = 'attn_weight_shift_mismatch'
elif (not SMOKE) and (not b8_ok):
    verdict = 'block_output_mismatch'
elif med_rel > 0.05:
    verdict = 'lineage_linear_fail'
elif float(np.nanmin(focal5_cos)) >= 0.99:
    verdict = 'focal_lineage_full'
elif float(np.nanmin(top8_cos)) >= 0.99:
    verdict = 'focal_lineage_top8'
elif all32_med >= 0.99:
    verdict = 'lineage_linear_allheads'
else:
    verdict = 'lineage_partial'
log('VERDICT: %s (setup_ok=%s a1_ok=%s a7_ok='
    '%s a8_ok=%s a9_ok=%s b9_ok=%s b10_ok=%s '
    'b8_ok=%s med_rel=%.4f focal5_min_cos='
    '%.4f smoke=%s)'
    % (verdict, setup_ok, a1_ok, a7_ok, a8_ok,
       a9_ok, b9_ok, b10_ok, b8_ok, med_rel,
       float(np.nanmin(focal5_cos)), SMOKE))

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
    'A7_DIFF': np.float64(
        a7_diff if a7_diff is not None
        else np.nan),
    'A7_OK': np.bool_(a7_ok),
    'A8_DIFF': np.float64(
        a8_diff if a8_diff is not None
        else np.nan),
    'A8_OK': np.bool_(a8_ok),
    'A9_DIFF': np.float64(
        a9_diff if a9_diff is not None
        else np.nan),
    'A9_OK': np.bool_(a9_ok),
    'B9_DIFF': np.float64(b9_diff),
    'B9_OK': np.bool_(b9_ok),
    'B10_DIFF': np.float64(b10_diff),
    'B10_OK': np.bool_(b10_ok),
    'B10_35_MAX': np.float64(b10_35_max),
    'PROBE_OK': np.bool_(probe_ok),
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
    'B8_OK': np.bool_(b8_ok),
    'SETUP_OK': np.bool_(setup_ok),
    'COS_LAD_34': COS_LAD_34,
    'MED_C_34': np.float64(med_c_34),
    'PA34': PA34, 'PF34': PF34,
    'CPX34': CPX34, 'CPA34': CPA34,
    'CPM34': CPM34, 'CPP34': CPP34,
    'CPA35': CPA35, 'CPM35': CPM35,
    'CPP35': CPP35,
    'NX34': NX34, 'NA34': NA34,
    'NM34': NM34, 'NP34o': NP34o,
    'NA35': NA35, 'NM35': NM35,
    'NP35o': NP35o,
    'LIN34': LIN34,
    'D34_ATTN': np.float64(d34_attn),
    'D34_MLP': np.float64(d34_mlp),
    'D34_OUT': np.float64(d34_out),
    'D34_X': np.float64(d34_x),
    'D35_ATTN': np.float64(d35_attn),
    'D35_MLP': np.float64(d35_mlp),
    'D35_OUT': np.float64(d35_out),
    'LIN34_MED': np.float64(lin34_med),
    'ZH34': ZH34, 'ZH35': ZH35,
    'DAH34': DAH34, 'DAH35': DAH35,
    'DAH34_MED': DAH34_MED,
    'DAH35_MED': DAH35_MED,
    'HEADLIN34': HEADLIN34,
    'HEADLIN35': HEADLIN35,
    'HL34_MED': np.float64(hl34_med),
    'HL35_MED': np.float64(hl35_med),
    'HAS_OPB': np.bool_(has_opb),
    'ATT34B': ATT34B.astype(np.float32),
    'ATT35B': ATT35B.astype(np.float32),
    'ATT34I': ATT34I.astype(np.float32),
    'ATT35I': ATT35I.astype(np.float32),
    'OBS_H': OBS_H.astype(np.float32),
    'PRED_H': PRED_H.astype(np.float32),
    'REL_JOINT': REL_JOINT,
    'COS_H': COS_H,
    'MED_REL': np.float64(med_rel),
    'COS_MED_H': cos_med_h,
    'FOCAL5': FOCAL5,
    'TOP8_3071': TOP8_3071,
    'REL_FOCAL5': np.array(rel_focal5),
    'M34_MED': m34_med, 'M35_MED': m35_med,
    'SP_M_ABSR': np.float64(sp_m_absR),
    'SP_M_ABSD': np.float64(sp_m_absD),
    'SP_M35_ABSR35': np.float64(
        sp_m35_absR35),
    'SEL_HEADS': np.array(SEL,
                          dtype=np.int64),
    'OV_TOP_IDS': top_ids,
    'OV_TOP_VALS': top_vals,
    'OV_TGRANK': tgrank,
    'OV_TTTOP_RANK': tttop_rank,
    'TTARG': TTARG,
    'TT': TT.astype(np.float32),
    'LG': LG.astype(np.float32),
}
np.savez(npz_path, **save)


# ==== result.json ====
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


ov_toks = []
for i in range(NS):
    ov_toks.append(
        [tok.decode([int(t)]).strip()
         or '#' + str(int(t))
         for t in top_ids[i]])

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
        'd34_cproj': {'x': f64(d34_x),
                      'attn': f64(d34_attn),
                      'mlp': f64(d34_mlp),
                      'out': f64(d34_out)},
        'd35_cproj': {'attn': f64(d35_attn),
                      'mlp': f64(d35_mlp),
                      'out': f64(d35_out)},
        'lincheck_34': f64(lin34_med),
        'probe': {'b9_diff': f64(b9_diff),
                  'b10_diff': f64(b10_diff),
                  'b10_35_max': f64(b10_35_max),
                  'probe_ok': bool(probe_ok)},
        'lineage': {
            'focal5': [int(v) for v
                       in FOCAL5],
            'med_rel_joint': f64(med_rel),
            'focal5_cos_med':
                [f64(v) for v
                 in focal5_cos],
            'focal5_rel_med':
                [f64(v) for v
                 in rel_focal5],
            'top8_min_cos':
                f64(np.nanmin(top8_cos)),
            'all32_med_cos': f64(all32_med)},
        'listener': {
            'm34_med': [f64(v) for v
                        in m34_med],
            'm35_med': [f64(v) for v
                        in m35_med],
            'sp_m34_absR': f64(sp_m_absR),
            'sp_m34_absDah': f64(sp_m_absD),
            'sp_m35_absR35':
                f64(sp_m35_absR35)},
        'ov': {
            'sel_heads': [int(v) for v
                          in SEL],
            'top10_ids':
                [[int(t) for t in row]
                 for row in top_ids],
            'top10_tok': ov_toks,
            'top10_val':
                [[f64(v) for v in row]
                 for row in top_vals],
            'target_rank_med':
                [f64(v) for v in tgrank],
            'tttop_rank_med':
                [f64(v) for v
                 in tttop_rank]},
    },
    'anchors': {
        'a1_diff': a1_diff, 'a1_ok': a1_ok,
        'a7_diff': a7_diff, 'a7_ok': a7_ok,
        'a8_diff': a8_diff, 'a8_ok': a8_ok,
        'a9_diff': a9_diff, 'a9_ok': a9_ok,
        'b9_diff': b9_diff, 'b9_ok': b9_ok,
        'b10_diff': b10_diff,
        'b10_ok': b10_ok,
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
