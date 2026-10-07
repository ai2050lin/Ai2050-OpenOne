# -*- coding: utf-8 -*-
"""Phase 3069: Omega-P66 cross-layer MLP pool
generalization (qwen3-4b single model bf16).

Question (3069 A, menu of 3068): 3067/3068
established at L35 that the conditional reversal
is carried by ONE top-128 neuron pool (96.5 pct
of the ALL-swap upper bound; same pool serving
both state conditions).  Is this layer-internal
concentration GENERAL across layers, and what is
the write polarity of the L30-34 MLPs (3066
called the upstream band 'distributed positive
normalization' - not presupposed here)?

E1 per-layer injection ladder: injection layer
l in {30..35}, 24 pairs each, repV protocol
3065/3066/3067-identical; per injection layer
record COS_LAD_l (a1 anchor vs the 3066 npz
rows 34/35, hard; rows 30-35 recorded as a1ext),
PA/PF lens probes, and the layer's OWN act/m
diff (ZACT_SELF/ZM_SELF at the injected layer).
E1.5 per-layer scores: score_l[j] = median_k(
|dact_l[k][j]| * ||W_down_l[:,j]||); S_TOP_l =
top-128; a2 anchor: S_TOP(L35) set equality vs
the 3067 npz S_A; S_R_l = random 128 from the
complement of S_TOP_l (seed 3025+li, sorted).
E1.6 observed polarity: cproj_l[k] = (W32 @ dm_l)
. TT[k]/||TT[k]|| medians over pairs (fp32
unembed WITHOUT final norm - declared
approximation).
E2 causal swap-to-base: down_proj forward-PRE-
hook at layer l, last position only; base
self-swap = identity (b7l per layer); groups
per layer = {TOP, RAND, ALL} x 24 pairs; ALL ->
m_l = m_l_base bit-exact (that layer's MLP diff
fully neutralized, upper anchor); a3 anchor:
PERM(L35, TOP) bit-equal vs the 3067 npz
PERM_A35; L35 positive control: recov(L35,TOP)
must reproduce the 3067 reference 0.7062164995
552644.
recov_l(g) = median PERM_l(g) - med_c_l (same
layer, same injection condition); capture_l =
|recov_top|/|recov_all| when |recov_all| >=
0.02 and signs agree.
verdict: setup fail -> setup_failed_crosslayer;
a1 fail -> anchor_mismatch_3066_ladder; a2 fail
-> anchor_mismatch_3067_sA; a3 fail ->
anchor_mismatch_3067_perm; L35 control fail ->
anchor_mismatch_3067_l35ctl; n_conc = layers
with capture >= 0.5: >= 5 -> cross_layer_
concentrated; <= 2 -> cross_layer_distributed;
else cross_layer_mixed.
memory discipline: single model; banks fp64
CPU; Wd32 fp32 copies per layer deleted after
scoring; W32 resident; del + gc + empty_cache
at the end.
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

PHASE = 3069
NAME = 'omega_p66_cross_layer_mlp_pool'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3069', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
LOG = os.path.join(OUT, 'run_log.txt')

MDIR = os.path.join(ROOT, 'models', 'hf',
                    'qwen3-4b')
NL, HID, KV_HEAD, NQ = 36, 2560, 8, 32
HDIM = 128
KVW = KV_HEAD * HDIM
FRONT = 4
SEED_MAIN = 3020
SEED_RAND_BASE = 3025
K_TOP = 128
LAYERS = (30, 31, 32, 33, 34, 35)
NLAY = len(LAYERS)
L35_RECOV_REF = 0.7062164995552644

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
            '3067-identical per injection layer '
            '(a1 bit anchor vs 3066 npz COS_LAD '
            'rows 34/35, hard; a1ext rows 30-35 '
            'recorded; a2 S_TOP(L35) set equality '
            'vs 3067 npz; a3 PERM(L35,TOP) bit '
            '0.0 vs 3067 npz; L35 positive '
            'control recov ref 0.7062164995552644)'
            '; lens probes fp32 W32 resident; '
            'verdict scalars fp64 cosv; smoke '
            'mode optional (SMOKE=1: 8 E1 pairs, '
            '4 E2 pairs per group, anchors off)',
    'question': '3069 A (menu of 3068): is the '
                'L35 layer-internal concentration '
                '(top-128 carries 96.5 pct of the '
                'ALL-swap upper bound, one pool '
                'serving both state conditions) '
                'GENERAL across layers, and what '
                'is the write polarity of the '
                'L30-34 MLPs (not presupposed)?',
    'E1_ladder': 'injection layer l in '
                 '{30,31,32,33,34,35} x 24 '
                 'pairs, repV 3065-identical; '
                 'COS_LAD_l per layer (a1 rows '
                 '34/35 bit anchor vs 3066 npz; '
                 'a1ext rows 30-35 recorded); '
                 'PA/PF lens probes per layer; '
                 'ZACT_SELF/ZM_SELF = the '
                 'injected layer own act/m diff',
    'E1_5_scores': 'score_l[j] = median_k(|dact_'
                   'l[k][j]| * ||W_down_l[:,j]||'
                   '); S_TOP_l = top-128 (a2: '
                   'S_TOP(L35) vs 3067 npz S_A '
                   'set equality); S_R_l = random '
                   '128 from complement of S_TOP_'
                   'l (seed 3025+li, sorted)',
    'E1_6_polarity': 'cproj_l[k] = (W32 @ dm_l[k]'
                     ' . TT[k])/||TT[k]||, dm_l '
                     '= ZM_SELF - BM[base, l]; '
                     'medians over pairs; fp32 '
                     'unembed without final norm '
                     '(declared approximation)',
    'E2_swaps': 'down_proj forward-PRE-hook at '
                'layer l, swap-to-base at the '
                'last position; per layer groups '
                '{TOP, RAND, ALL} x 24 pairs; '
                'ALL -> m_l = m_l_base bit-exact '
                '(layer MLP diff neutralized, '
                'upper anchor); base self-swap = '
                'identity (b7l per layer); a3: '
                'PERM(L35,TOP) vs 3067 npz '
                'PERM_A35 bit 0.0',
    'anchors': 'a1 rows 34/35 bit 0.0 vs 3066 '
               'npz (hard); a2 S_TOP(L35) set '
               'equality vs 3067 npz (hard); '
               'a3 PERM(L35,TOP) bit 0.0 vs '
               '3067 npz (hard); l35ctl |recov'
               '(L35,TOP) - 0.7062164995552644| '
               '< 1e-9 (hard); med_c reference '
               'assert (0.1487826048372403 / '
               '-0.35579100779974404); a2b '
               '|PF(L35) - med_c(L35)| <= 0.01; '
               'b0 recapture bit 0.0 (incl. '
               'BACTL 6 layers); b1 sham bit '
               '0.0; b3 finite; b4 delta-x per '
               'injection layer bit 0.0; b5 act '
               '= silu(g)*u bf16 (recorded, '
               'L35); b6 (x+a)+m = h2 bf16 bit '
               '0.0; b7l identity self-swap per '
               'layer bit 0.0',
    'verdict': 'setup fail -> '
               'setup_failed_crosslayer; a1 '
               'fail -> anchor_mismatch_3066_'
               'ladder; a2 fail -> '
               'anchor_mismatch_3067_sA; a3 '
               'fail -> anchor_mismatch_3067_'
               'perm; l35ctl fail -> '
               'anchor_mismatch_3067_l35ctl; '
               'n_conc = #layers with |recov_'
               'all| >= 0.02, sign(top) == '
               'sign(all), capture >= 0.5: '
               '>= 5 -> cross_layer_'
               'concentrated; <= 2 -> '
               'cross_layer_distributed; else '
               'cross_layer_mixed',
    'statistics_discipline': 'same-precision '
                             'bit anchors on '
                             'frozen seeds; TT '
                             'logit-space 3065-'
                             'identical; recov '
                             'referenced to the '
                             'same injection '
                             'condition med_c_l; '
                             'random control per '
                             'layer disjoint from '
                             'S_TOP_l; capture '
                             'requires sign '
                             'agreement and a '
                             '0.02 floor',
    'memory_discipline': 'single model; banks '
                         'fp64 CPU; Wd32 fp32 '
                         'copies per layer '
                         'deleted after scoring; '
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


def hook_act_pre(st):
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
        .register_forward_pre_hook(hook_act_pre(
            stateACT[li]))


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


def forward_gen(ids, repl=None, act_swap=None):
    """3065-identical V replacement; optional
    act swap = (layer, idx, vals) at that
    layer's down_proj input (last position).
    Returns bf16 last-position captures zX/zA/
    zM/zP (NL, HID), zH (NL, NQ*HDIM), zZ (NL,
    HID), zG/zU/zACT (NL, INTER) on GPU."""
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
    if act_swap is not None:
        li_, midx, mval = act_swap
        stateACT[li_]['mask'] = torch.tensor(
            np.ascontiguousarray(midx),
            dtype=torch.long, device='cuda')
        stateACT[li_]['repl'] = torch.tensor(
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
PREF_K = np.array([pair_idx(k)[2]
                   for k in range(NP_)])
OFF_K = np.array([pair_idx(k)[3]
                  for k in range(NP_)])


# ==== banks ====
LG = np.zeros((n_pr, NVOC))
VB = np.zeros((n_pr, NL, NMAX, KVW))
PB = np.zeros((n_pr, NL, NMAX, HID))
BX = np.zeros((n_pr, NL, HID))
BA = np.zeros((n_pr, NL, HID))
BM = np.zeros((n_pr, NL, HID))
BH = np.zeros((n_pr, NL, NQ * HDIM))
BZ35 = np.zeros((n_pr, HID))
BG35 = np.zeros((n_pr, INTER))
BU35 = np.zeros((n_pr, INTER))
BACTL = np.zeros((NLAY, n_pr, INTER))
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
    for li in range(NLAY):
        BACTL[li, i] = zACT[LAYERS[li]] \
            .double().cpu().numpy()
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
BACT35 = BACTL[NLAY - 1]
log('banks: LG%s VB%s BACTL%s (forwards=%d)'
    % (LG.shape, VB.shape, BACTL.shape, FW[0]))
log('a2lens lens(final)=logits: fp32 max|d|='
    '%.3e (<=0.125 ok) bit_attempt=%.3e'
    % (a2_max, a2_bit))

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
        np.abs(BG35[si] - zG[NL - 1].double()
               .cpu().numpy()))))
    b0_diff = max(b0_diff, float(np.max(
        np.abs(BU35[si] - zU[NL - 1].double()
               .cpu().numpy()))))
    for li in range(NLAY):
        b0_diff = max(b0_diff, float(np.max(
            np.abs(BACTL[li, si]
                   - zACT[LAYERS[li]].double()
                   .cpu().numpy()))))
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
             and np.isfinite(BZ35).all()
             and np.isfinite(BG35).all()
             and np.isfinite(BU35).all()
             and np.isfinite(BACTL).all())
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


# ==== E1 per-layer injection ladder ====
COS_LAD_L = np.full((NLAY, NP_USE), np.nan)
PA_L = np.full((NLAY, NP_USE), np.nan)
PF_L = np.full((NLAY, NP_USE), np.nan)
ZACT_SELF = np.zeros((NLAY, NP_USE, INTER))
ZM_SELF = np.zeros((NLAY, NP_USE, HID))
b4_diff = 0.0
for li in range(NLAY):
    l = LAYERS[li]
    for k in range(NP_USE):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[l, ridx, :] = VB[pref_i][
            l, off + ridx, :]
        lg, _, _, _, zX, zA, zM, zP, zH, \
            zZ, zG, zU, zACT = forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
        dlg = lg - LG[base_i]
        COS_LAD_L[li, k] = cosv(dlg, TT[k])
        b4_diff = max(b4_diff, float(np.max(
            np.abs(zX[l].double().cpu().numpy()
                   - BX[base_i, l]))))
        zinj = torch.cat(
            [zX, zX + zA, zP], dim=0)
        zbase = base_z_gpu(base_i, nb)
        c = lens_cos(zinj, zbase, k)
        PA_L[li, k] = c[NL + l]
        PF_L[li, k] = c[2 * NL + l]
        ZACT_SELF[li, k] = zACT[l].double() \
            .cpu().numpy()
        ZM_SELF[li, k] = zM[l].double() \
            .cpu().numpy()
b4_ok = bool(b4_diff == 0.0)
log('b4 delta-x per injection layer diff='
    '%.3e ok=%s' % (b4_diff, b4_ok))

MED_C_L = np.array([float(np.median(COS_LAD_L[li]))
                    for li in range(NLAY)])
PA_M = np.array([float(np.median(PA_L[li]))
                 for li in range(NLAY)])
PF_M = np.array([float(np.median(PF_L[li]))
                 for li in range(NLAY)])
for li in range(NLAY):
    log('E1 l=%d med_c=%.4f PA=%.4f PF=%.4f '
        '(forwards=%d)'
        % (LAYERS[li], MED_C_L[li], PA_M[li],
           PF_M[li], FW[0]))

# a1: ladder rows 34/35 bit anchor vs 3066 npz
a1_diff = None
a1_ok = False
a1ext_diff = None
npz66 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3066',
    'omega_p63_last_layer_flip_anatomy',
    'omega_p63_last_layer_flip_anatomy.npz')
if not SMOKE:
    z = np.load(npz66)
    c66 = z['COS_LAD']
    i34 = LAYERS.index(34)
    i35 = LAYERS.index(35)
    a1_diff = float(max(
        np.max(np.abs(COS_LAD_L[i34] - c66[34])),
        np.max(np.abs(COS_LAD_L[i35] - c66[35]))))
    a1_ok = bool(a1_diff == 0.0)
    if c66.shape[0] >= 36:
        a1ext_diff = float(max(
            np.max(np.abs(COS_LAD_L[li]
                          - c66[LAYERS[li]]))
            for li in range(NLAY)))
    log('a1 ladder rows 34/35 vs 3066 npz: '
        'max|d|=%.3e ok=%s | a1ext rows 30-35 '
        'max|d|=%s'
        % (a1_diff, a1_ok,
           ('%.3e' % a1ext_diff)
           if a1ext_diff is not None else 'n/a'))
else:
    log('a1 skipped (smoke)')

if not SMOKE:
    i34 = LAYERS.index(34)
    i35 = LAYERS.index(35)
    assert abs(MED_C_L[i34]
               - 0.1487826048372403) < 1e-12, \
        MED_C_L[i34]
    assert abs(MED_C_L[i35]
               - (-0.35579100779974404)) \
        < 1e-12, MED_C_L[i35]
    log('med_c reference assert ok')

a2b_diff = abs(float(PF_M[NLAY - 1])
               - float(MED_C_L[NLAY - 1]))
a2b_ok = bool(a2b_diff <= 0.01)
log('a2b lens calibration |PF(35) - med_c(35)|='
    '%.4f ok=%s' % (a2b_diff, a2b_ok))

# ==== E1.5 per-layer scores ====
SCORE_L = np.zeros((NLAY, INTER))
for li in range(NLAY):
    l = LAYERS[li]
    Wd = layers[l].mlp.down_proj.weight \
        .detach().float()
    cn = Wd.norm(dim=0).double().cpu().numpy()
    DA = np.zeros((NP_USE, INTER))
    for k in range(NP_USE):
        base_i = int(BASE_K[k])
        DA[k] = np.abs(ZACT_SELF[li, k]
                       - BACTL[li][base_i])
    SCORE_L[li] = np.median(
        DA * cn[None, :], axis=0)
    del Wd
gc.collect()
torch.cuda.empty_cache()
S_TOP = np.zeros((NLAY, K_TOP), dtype=np.int64)
S_R_L = np.zeros((NLAY, K_TOP), dtype=np.int64)
for li in range(NLAY):
    S_TOP[li] = np.argsort(-SCORE_L[li])[:K_TOP] \
        .astype(np.int64)
    rng = np.random.RandomState(
        SEED_RAND_BASE + li)
    comp = np.setdiff1d(
        np.arange(INTER), S_TOP[li])
    S_R_L[li] = np.sort(rng.choice(
        comp, size=K_TOP,
        replace=False)).astype(np.int64)
    log('E1.5 l=%d score max=%.4f med=%.4f '
        'top16=%s'
        % (LAYERS[li], float(SCORE_L[li].max()),
           float(np.median(SCORE_L[li])),
           S_TOP[li][:16].tolist()))

# a2: S_TOP(L35) set equality vs 3067 npz
a2set_diff = None
a2set_ok = False
npz67 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3067',
    'omega_p64_mlp_conditional_reversal',
    'omega_p64_mlp_conditional_reversal'
    '.npz')
if not SMOKE:
    z67 = np.load(npz67)
    sa67 = z67['S_A']
    a2set_diff = int(len(np.setdiff1d(
        S_TOP[NLAY - 1], sa67))
        + len(np.setdiff1d(sa67,
                           S_TOP[NLAY - 1])))
    a2set_ok = bool(a2set_diff == 0)
    log('a2 S_TOP(L35) set vs 3067 npz: '
        '|diff|=%d ok=%s'
        % (a2set_diff, a2set_ok))
else:
    log('a2 skipped (smoke)')

# ==== E1.6 observed polarity ====
OBS_CPROJ = np.zeros((NLAY, NP_USE))
for li in range(NLAY):
    l = LAYERS[li]
    for k in range(NP_USE):
        base_i = int(BASE_K[k])
        dm = torch.tensor(
            ZM_SELF[li, k] - BM[base_i, l],
            device='cuda').float()
        with torch.no_grad():
            u = F.linear(
                dm.unsqueeze(0), W32)[0]
        u_np = u.double().cpu().numpy()
        t = TT[k]
        tn = max(float(np.linalg.norm(t)),
                 1e-12)
        OBS_CPROJ[li, k] = float(
            u_np @ t) / tn
OBS_M = np.array([float(np.median(OBS_CPROJ[li]))
                  for li in range(NLAY)])
for li in range(NLAY):
    log('E1.6 l=%d obs cproj med=%.1f '
        '(dm_l TT projection, unnorm)'
        % (LAYERS[li], OBS_M[li]))

# ==== b7l identity self-swap per layer ====
b7l_diff = 0.0
for li in range(NLAY):
    l = LAYERS[li]
    b, base_i, pref_i, off = pair_idx(0)
    lg7 = forward_gen(
        assembled[base_i]['ids'],
        act_swap=(l, S_TOP[li],
                  BACTL[li][base_i][
                      S_TOP[li]]))[0]
    b7l_diff = max(b7l_diff, float(np.max(
        np.abs(lg7 - LG[base_i]))))
b7l_diff = float(b7l_diff)
b7l_ok = bool(b7l_diff == 0.0)
log('b7l identity self-swap (6 layers) diff='
    '%.3e ok=%s' % (b7l_diff, b7l_ok))

# ==== E2 per-layer causal swaps ====
ALLN = np.arange(INTER)
GN = ('TOP', 'RAND', 'ALL')
PERM_L = np.full((NLAY, 3, K3), np.nan)
for li in range(NLAY):
    l = LAYERS[li]
    for gi, sset in enumerate(
            (S_TOP[li], S_R_L[li], ALLN)):
        for k in range(K3):
            b, base_i, pref_i, off = \
                pair_idx(k)
            nb = int(LENS[base_i])
            repV = VB[base_i, :, :nb, :].copy()
            repV[l, ridx, :] = VB[pref_i][
                l, off + ridx, :]
            lg = forward_gen(
                assembled[base_i]['ids'],
                repl=repV,
                act_swap=(l, sset,
                          BACTL[li][base_i][
                              sset]))[0]
            PERM_L[li, gi, k] = cosv(
                lg - LG[base_i], TT[k])
    log('E2 l=%d TOP med=%.4f | RAND med='
        '%.4f | ALL med=%.4f'
        % (l,
           float(np.median(PERM_L[li, 0])),
           float(np.median(PERM_L[li, 1])),
           float(np.median(PERM_L[li, 2]))))

# a3: PERM(L35, TOP) vs 3067 npz PERM_A35
a3_diff = None
a3_ok = False
if not SMOKE:
    p67 = np.load(npz67)
    a3_diff = float(np.max(np.abs(
        PERM_L[NLAY - 1, 0]
        - p67['PERM_A35'])))
    a3_ok = bool(a3_diff == 0.0)
    log('a3 perm L35 TOP vs 3067 npz PERM_A35: '
        'max|d|=%.3e ok=%s'
        % (a3_diff, a3_ok))
else:
    log('a3 skipped (smoke)')

CP_L = np.array([[float(np.median(PERM_L[li, gi]))
                  for gi in range(3)]
                 for li in range(NLAY)])
RECOV_L = CP_L - MED_C_L[:, None]
CAPTURE = np.full(NLAY, np.nan)
for li in range(NLAY):
    rt = float(RECOV_L[li, 0])
    rr = float(RECOV_L[li, 1])
    ra = float(RECOV_L[li, 2])
    if abs(ra) >= 0.02 \
            and (rt >= 0) == (ra >= 0) \
            and rt != 0.0:
        CAPTURE[li] = abs(rt) / abs(ra)
    log('E2 l=%d recov top=%.4f rand=%.4f '
        'all=%.4f capture=%s'
        % (LAYERS[li], rt, rr, ra,
           ('%.3f' % CAPTURE[li])
           if np.isfinite(CAPTURE[li])
           else 'n/a'))
log('E2 recovery table done (forwards=%d)'
    % FW[0])

# ==== L35 positive control vs 3067 ref ====
l35ctl_diff = None
l35ctl_ok = False
if not SMOKE:
    l35ctl_diff = abs(float(RECOV_L[NLAY - 1, 0])
                      - L35_RECOV_REF)
    l35ctl_ok = bool(l35ctl_diff < 1e-9)
    log('l35ctl recov(L35,TOP)=%.12f vs ref '
        '0.7062164995552644: diff=%.3e ok=%s'
        % (float(RECOV_L[NLAY - 1, 0]),
           l35ctl_diff, l35ctl_ok))
else:
    log('l35ctl skipped (smoke)')

# ==== verdict ====
n_conc = int(sum(
    1 for li in range(NLAY)
    if np.isfinite(CAPTURE[li])
    and CAPTURE[li] >= 0.5))
setup_ok = bool(b0_ok and b1_ok and b3_ok
                and b4_ok and b6_ok and b7l_ok
                and a2b_ok)
if not setup_ok:
    verdict = 'setup_failed_crosslayer'
elif (not SMOKE) and (not a1_ok):
    verdict = 'anchor_mismatch_3066_ladder'
elif (not SMOKE) and (not a2set_ok):
    verdict = 'anchor_mismatch_3067_sA'
elif (not SMOKE) and (not a3_ok):
    verdict = 'anchor_mismatch_3067_perm'
elif (not SMOKE) and (not l35ctl_ok):
    verdict = 'anchor_mismatch_3067_l35ctl'
elif n_conc >= 5:
    verdict = 'cross_layer_concentrated'
elif n_conc <= 2:
    verdict = 'cross_layer_distributed'
else:
    verdict = 'cross_layer_mixed'
log('VERDICT: %s (setup_ok=%s a1_ok=%s a2_ok='
    '%s a3_ok=%s l35ctl_ok=%s n_conc=%d '
    'smoke=%s)'
    % (verdict, setup_ok,
       a1_ok, a2set_ok, a3_ok, l35ctl_ok,
       n_conc, SMOKE))

# ==== npz ====
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'LAYERS': np.array(LAYERS, dtype=np.int64),
    'A1_DIFF': np.float64(
        a1_diff if a1_diff is not None
        else np.nan),
    'A1_OK': np.bool_(a1_ok),
    'A1EXT_DIFF': np.float64(
        a1ext_diff if a1ext_diff is not None
        else np.nan),
    'A2SET_DIFF': np.float64(
        a2set_diff if a2set_diff is not None
        else np.nan),
    'A2SET_OK': np.bool_(a2set_ok),
    'A3_DIFF': np.float64(
        a3_diff if a3_diff is not None
        else np.nan),
    'A3_OK': np.bool_(a3_ok),
    'L35CTL_DIFF': np.float64(
        l35ctl_diff if l35ctl_diff is not None
        else np.nan),
    'L35CTL_OK': np.bool_(l35ctl_ok),
    'A2LENS_MAX': np.float64(a2_max),
    'A2B_DIFF': np.float64(a2b_diff),
    'B0_DIFF': np.float64(b0_diff),
    'B1_DIFF': np.float64(b1_diff),
    'B4_DIFF': np.float64(b4_diff),
    'B5_DIFF': np.float64(b5_diff),
    'B6_DIFF': np.float64(b6_diff),
    'B7L_DIFF': np.float64(b7l_diff),
    'B3_OK': np.bool_(b3_ok),
    'B4_OK': np.bool_(b4_ok),
    'B6_OK': np.bool_(b6_ok),
    'B7L_OK': np.bool_(b7l_ok),
    'SETUP_OK': np.bool_(setup_ok),
    'COS_LAD_L': COS_LAD_L,
    'PA_L': PA_L, 'PF_L': PF_L,
    'MED_C_L': MED_C_L,
    'PA_M': PA_M, 'PF_M': PF_M,
    'SCORE_L': SCORE_L.astype(np.float32),
    'S_TOP': S_TOP, 'S_R_L': S_R_L,
    'OBS_CPROJ_L': OBS_CPROJ,
    'OBS_M': OBS_M,
    'PERM_L': PERM_L,
    'CP_L': CP_L,
    'RECOV_L': RECOV_L,
    'CAPTURE': CAPTURE,
    'N_CONC': np.int64(n_conc),
    'TT': TT.astype(np.float32),
    'LG': LG.astype(np.float32),
}
np.savez(npz_path, **save)


# ==== result.json ====
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


per_layer = []
for li in range(NLAY):
    per_layer.append({
        'layer': int(LAYERS[li]),
        'med_c': f64(MED_C_L[li]),
        'pa': f64(PA_M[li]), 'pf': f64(PF_M[li]),
        'obs_cproj_med': f64(OBS_M[li]),
        'score_max': f64(SCORE_L[li].max()),
        'score_med': f64(float(
            np.median(SCORE_L[li]))),
        'c_perm': {'top': f64(CP_L[li, 0]),
                   'rand': f64(CP_L[li, 1]),
                   'all': f64(CP_L[li, 2])},
        'recov': {'top': f64(RECOV_L[li, 0]),
                  'rand': f64(RECOV_L[li, 1]),
                  'all': f64(RECOV_L[li, 2])},
        'capture': f64(CAPTURE[li]),
    })
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
        'layers': [int(x) for x in LAYERS],
        'med_c_l': {str(int(LAYERS[li])):
                    f64(MED_C_L[li])
                    for li in range(NLAY)},
        'obs_cproj_l': {str(int(LAYERS[li])):
                        f64(OBS_M[li])
                        for li in range(NLAY)},
        'per_layer': per_layer,
        'n_conc': int(n_conc),
        'l35_control_recov':
            f64(RECOV_L[NLAY - 1, 0]),
        'l35_control_ref': L35_RECOV_REF,
    },
    'anchors': {
        'a1_diff': a1_diff, 'a1_ok': a1_ok,
        'a1ext_diff': a1ext_diff,
        'a2set_diff': a2set_diff,
        'a2set_ok': a2set_ok,
        'a3_diff': a3_diff, 'a3_ok': a3_ok,
        'l35ctl_diff': l35ctl_diff,
        'l35ctl_ok': l35ctl_ok,
        'a2lens_max': a2_max,
        'a2lens_ok': a2lens_ok,
        'a2b_diff': a2b_diff, 'a2b_ok': a2b_ok,
        'b0_diff': b0_diff, 'b0_ok': b0_ok,
        'b1_diff': b1_diff, 'b1_ok': b1_ok,
        'b3_ok': b3_ok,
        'b4_diff': b4_diff, 'b4_ok': b4_ok,
        'b5_diff': b5_diff,
        'b6_diff': b6_diff, 'b6_ok': b6_ok,
        'b7l_diff': b7l_diff, 'b7l_ok': b7l_ok,
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
