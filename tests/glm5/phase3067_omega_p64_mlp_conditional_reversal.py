# -*- coding: utf-8 -*-
"""Phase 3067: Omega-P64 L35 MLP conditional
reversal anatomy (qwen3-4b single model bf16).

Question (3067 A, menu of 3066): WHICH parameter
carriers inside the L35 MLP implement the
conditional sign reversal found in 3066 (raw
+0.199 -> final -0.356 for direct V injection at
L35, but +0.237 -> +0.149 preserved for the
L34-propagated displacement)?

E1 dual-state capture: l in {34, 35} x N pairs,
repV 3065/3066-identical (a1 bit anchor vs the
3066 npz COS_LAD rows 34/35); per forward
capture at the L35 last position: z (post_
attention_layernorm out), g/u (gate/up_proj
out), act (down_proj in), m (mlp out), a, x;
COS_LAD rows 34/35 recomputed; PA/PF diag lens
probes recorded (final_norm bf16 + fp32 W32,
3066 lens chain).
E2 linear Jacobian (recorded, no threshold):
around the BASE L35 MLP state (z0, g0 = W_gate
z0, u0 = W_up z0), fp32 weight copies:
  dact_lin = (silu'(g0)*u0)*(W_gate dz)
             + silu(g0)*(W_up dz)
  dm_lin = W_down dact_lin
per pair x {state A = inj@35, state B = inj@34}:
cos + rel err vs the actual dact/dm; plus the
MLP output direction angle across the two
states (cos(dm_A, dm_B), cos(dz_A, dz_B)) and
norms; CMREAD = m-channel readout polarity
W32 @ (norm(h1+m) - norm(h1)) vs TT per state
(base/A/B; h1 = x + a bf16, model-identical).
E1.5 neuron score: score_j = median_k(|dact_k[j]
| * ||W_down[:,j]||) at L35 state A; S_A = top-
128 (K_TOP); S_R = random 128 from the
complement (seed 3023, sorted).
E3 causal swap-to-base: down_proj FORWARD-PRE-
hook at L35 replaces act[0, -1, S] with the
BASE act values (last position only - the L35
MLP at earlier positions cannot reach the
last-position readout: L35 attention reads
x_35 = block input, not block output). Groups:
inj@35 + S_A, inj@35 + S_R, inj@34 + S_A;
c_perm = med_k cosv fp64; base self-swap =
identity (b7), so no matched base forward
needed.
anchors: a1 rows 34/35 bit 0.0 vs 3066 npz
(hard, non-smoke); med_c reference assert
(0.1487826048372403 / -0.35579100779974404);
a2 lens(final h2) = logits fp32 <= 0.125 (bit
attempt recorded); a2b |PF35 med - med_c(35)|
<= 0.01; b0 recapture bit 0.0; b1 sham bit 0.0;
b3 finite; b4 delta-x at the injection layer
bit 0.0; b5 act = silu(g)*u bf16 reconstruction
(bit attempt, recorded); b6 (x+a)+m = h2 bf16
bit 0.0 (all layers); b7 identity self-swap
bit 0.0.
verdict: setup fail -> setup_failed_mlp_
reversal; a1 fail -> anchor_mismatch_3066_
ladder; recov35 = c_perm_A35 - med_c(35);
recov_rand = c_perm_R35 - med_c(35); delta34 =
c_perm_B34 - med_c(34); recov35 < 0.10 ->
mlp_reversal_distributed_topk; elif
|recov_rand| > 0.05 -> mlp_reversal_nonspecific
_topk; elif |delta34| <= 0.05 ->
mlp_reversal_localized_statemode; else
mlp_reversal_localized_shared.
memory discipline: single model; banks fp64
CPU; Jacobian fp32 weight copies (~300 MB)
deleted after E1.5; W32 resident (+1.55 GB);
del + gc + empty_cache at the end.
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

PHASE = 3067
NAME = 'omega_p64_mlp_conditional_reversal'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3067', NAME)
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
SEED_RAND = 3023
K_TOP = 128

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
            'seed 3020; E1 protocol 3066-identical '
            '(a1 bit anchor vs 3066 npz COS_LAD '
            'rows 34/35, hard); lens probes: '
            'final_norm bf16 (model path) then '
            'fp32 matmul against resident W32 = '
            'Wemb.float() (3066 lens chain); '
            'verdict scalars via fp64 numpy cosv '
            'on final logits; smoke mode optional '
            '(SMOKE=1: 8 E1 pairs + 4 E3 pairs, '
            'a1 off)',
    'question': '3067 A (menu of 3066): which '
                'parameter carriers inside the '
                'L35 MLP implement the '
                'conditional sign reversal (raw '
                '+0.199 -> final -0.356 for '
                'direct V injection at L35; '
                '+0.237 -> +0.149 preserved for '
                'the L34-propagated '
                'displacement)? Localize: top-128 '
                'down_proj-input neurons, linear-'
                'Jacobian explainability, state '
                'specificity.',
    'E1_dual_state': 'l in {34,35} x N pairs, '
                     'repV 3065-identical; '
                     'captures at the L35 last '
                     'position: z (ln2 out), g/u '
                     '(gate/up out), act (down '
                     'in), m, a, x; COS_LAD rows '
                     '34/35 = a1 anchor; PA/PF '
                     'diag lens probes recorded',
    'E2_jacobian': 'linearization around the '
                   'BASE L35 MLP state (z0, g0 = '
                   'W_gate z0, u0 = W_up z0), '
                   'fp32 weight copies: dact_lin '
                   '= (silu_prime(g0)*u0)*(W_gate '
                   'dz) + silu(g0)*(W_up dz); '
                   'dm_lin = W_down dact_lin; per '
                   'pair x state A/B: cos + rel '
                   'err vs actual (recorded, no '
                   'threshold); direction angles '
                   'cos(dm_A, dm_B), cos(dz_A, '
                   'dz_B) + norms; CMREAD = '
                   'm-channel readout polarity '
                   'per state (base/A/B)',
    'E1_5_score': 'score_j = median_k(|dact_k[j]'
                  '| * ||W_down[:,j]||) at L35 '
                  'state A; S_A = top-128; S_R = '
                  'random 128 from the complement '
                  '(seed 3023, sorted)',
    'E3_swap': 'down_proj forward-PRE-hook at '
               'L35 replaces act[0,-1,S] with '
               'the BASE act values (last '
               'position only: the L35 MLP at '
               'earlier positions cannot reach '
               'the last-position readout - L35 '
               'attention reads x_35 = block '
               'input, not block output); '
               'groups: inj@35+S_A, inj@35+S_R, '
               'inj@34+S_A; c_perm = med_k cosv '
               'fp64; base self-swap = identity '
               '(b7) so no matched base forward '
               'needed',
    'anchors': 'a1 rows 34/35 bit 0.0 vs 3066 '
               'npz (hard, non-smoke); med_c '
               'reference assert (0.1487826048372'
               '403 / -0.35579100779974404); a2 '
               'lens(final h2) = logits fp32 <= '
               '0.125 (bit attempt recorded); '
               'a2b |PF35 med - med_c(35)| <= '
               '0.01; b0 recapture bit 0.0; b1 '
               'sham bit 0.0; b3 finite; b4 '
               'delta-x at injection layer bit '
               '0.0; b5 act = silu(g)*u bf16 '
               'reconstruction (bit attempt, '
               'recorded); b6 (x+a)+m = h2 bf16 '
               'bit 0.0 (all layers); b7 '
               'identity self-swap bit 0.0',
    'verdict': 'setup fail -> '
               'setup_failed_mlp_reversal; a1 '
               'fail -> anchor_mismatch_3066_'
               'ladder; recov35 = c_perm_A35 - '
               'med_c(35); recov_rand = '
               'c_perm_R35 - med_c(35); delta34 '
               '= c_perm_B34 - med_c(34); '
               'recov35 < 0.10 -> '
               'mlp_reversal_distributed_topk; '
               'elif |recov_rand| > 0.05 -> '
               'mlp_reversal_nonspecific_topk; '
               'elif |delta34| <= 0.05 -> '
               'mlp_reversal_localized_statemode'
               '; else '
               'mlp_reversal_localized_shared',
    'statistics_discipline': 'same-precision '
                             'bit anchors on '
                             'frozen seeds; TT '
                             'logit-space 3065-'
                             'identical; lens '
                             'probes declared '
                             'fp32; Jacobian '
                             'comparison '
                             'descriptive (no '
                             'threshold '
                             'preregistered); '
                             'random control '
                             'judges swap '
                             'specificity',
    'memory_discipline': 'single model; banks '
                         'fp64 CPU; Jacobian '
                         'fp32 weight copies '
                         '(~300 MB) deleted '
                         'after E1.5; W32 '
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
# fp32 unembed resident for lens probes (3066
# lens chain; bf16-linear output rounding
# pollutes lens cosines at the ~0.25 level).
W32 = Wemb.float()
log('qwen3-4b loaded bf16 (vocab=%d inter=%d) '
    'gpu=%.2f GB (+W32 fp32 %.2f GB)'
    % (NVOC, INTER,
       torch.cuda.memory_allocated() / 1e9,
       W32.numel() * 4 / 1e9))

FW = [0]
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
stateACT = {'repl': None, 'mask': None}
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
layers[NL - 1].mlp.down_proj \
    .register_forward_pre_hook(hook_act_pre(
        stateACT))


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
    stateACT['repl'] = None
    stateACT['mask'] = None


def forward_gen(ids, repl=None, act_swap=None):
    """3065-identical V replacement; optional
    act swap at the L35 down_proj input (last
    position).  Returns bf16 last-position
    captures zX/zA/zM/zP (NL, HID), zH (NL,
    NQ*HDIM), zZ (NL, HID), zG/zU/zACT (NL,
    INTER) on GPU."""
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
        midx, mval = act_swap
        stateACT['mask'] = torch.tensor(
            np.ascontiguousarray(midx),
            dtype=torch.long, device='cuda')
        stateACT['repl'] = torch.tensor(
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
    BACT35[i] = zACT[NL - 1].double() \
        .cpu().numpy()
    # a2: lens(final h2) must match actual
    # logits (bit attempt + 1-bf16-ulp fp32)
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
a2_ok = bool(a2_max <= 0.125)
a2_recorded_only = bool(a2_ok and a2_bit != 0.0)
log('banks: LG%s VB%s BG35%s (forwards=%d)'
    % (LG.shape, VB.shape, BG35.shape, FW[0]))
log('a2 lens(final)=logits: fp32 max|d|=%.3e '
    '(<=0.125 ok) bit_attempt=%.3e '
    'recorded_only=%s'
    % (a2_max, a2_bit, a2_recorded_only))

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
    b0_diff = max(b0_diff, float(np.max(
        np.abs(BACT35[si] - zACT[NL - 1]
               .double().cpu().numpy()))))
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
             and np.isfinite(BACT35).all())
log('b3 finite=%s' % b3_ok)

# b5: act = silu(g) * u (bf16 reconstruction,
# L35, all samples; bit attempt, recorded)
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

# b6: (x + a) + m = h2 (bf16, all layers)
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
    """(3*NL, HID) bf16 GPU: x, h1 = x + a, h2;
    last position is nb-1 (prompt length)."""
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
    """fp32 lens cos for all 3*NL rows of zinj
    vs zbase against TT[k].  norm in bf16 (the
    model's own path), matmul in fp32 against
    the resident W32 (no bf16 output rounding)."""
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


# ==== E1 dual-state capture ====
COS_LAD_34 = np.full(NP_USE, np.nan)
COS_LAD_35 = np.full(NP_USE, np.nan)
PA34 = np.full(NP_USE, np.nan)
PA35 = np.full(NP_USE, np.nan)
PF34 = np.full(NP_USE, np.nan)
PF35 = np.full(NP_USE, np.nan)
# state 0 = A (inj@35), state 1 = B (inj@34)
ZX = np.zeros((2, NP_USE, HID))
ZA = np.zeros((2, NP_USE, HID))
ZM = np.zeros((2, NP_USE, HID))
ZZ = np.zeros((2, NP_USE, HID))
ZG = np.zeros((2, NP_USE, INTER))
ZU = np.zeros((2, NP_USE, INTER))
ZACT = np.zeros((2, NP_USE, INTER))
b4_diff = 0.0
for l in (34, 35):
    si_ = 0 if l == 35 else 1
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
        if l == 34:
            COS_LAD_34[k] = cosv(dlg, TT[k])
        else:
            COS_LAD_35[k] = cosv(dlg, TT[k])
        b4_diff = max(b4_diff, float(np.max(
            np.abs(zX[l].double().cpu().numpy()
                   - BX[base_i, l]))))
        zinj = torch.cat(
            [zX, zX + zA, zP], dim=0)
        zbase = base_z_gpu(base_i, nb)
        c = lens_cos(zinj, zbase, k)
        if l == 34:
            PA34[k] = c[NL + l]
            PF34[k] = c[2 * NL + l]
        else:
            PA35[k] = c[NL + l]
            PF35[k] = c[2 * NL + l]
        ZX[si_, k] = zX[NL - 1].double() \
            .cpu().numpy()
        ZA[si_, k] = zA[NL - 1].double() \
            .cpu().numpy()
        ZM[si_, k] = zM[NL - 1].double() \
            .cpu().numpy()
        ZZ[si_, k] = zZ[NL - 1].double() \
            .cpu().numpy()
        ZG[si_, k] = zG[NL - 1].double() \
            .cpu().numpy()
        ZU[si_, k] = zU[NL - 1].double() \
            .cpu().numpy()
        ZACT[si_, k] = zACT[NL - 1].double() \
            .cpu().numpy()
b4_ok = bool(b4_diff == 0.0)
log('b4 delta-x at injection layer diff=%.3e '
    'ok=%s' % (b4_diff, b4_ok))

med_c_34 = float(np.median(COS_LAD_34))
med_c_35 = float(np.median(COS_LAD_35))
pa34 = float(np.median(PA34))
pa35 = float(np.median(PA35))
pf34 = float(np.median(PF34))
pf35 = float(np.median(PF35))
log('E1 med_c(34)=%.4f med_c(35)=%.4f '
    'PA=%.4f/%.4f PF=%.4f/%.4f (forwards=%d)'
    % (med_c_34, med_c_35, pa34, pa35,
       pf34, pf35, FW[0]))

# a1: ladder rows 34/35 bit anchor vs 3066 npz
a1_diff = None
a1_ok = False
if not SMOKE:
    npz66 = os.path.join(
        ROOT, 'tests', 'glm5', 'result',
        'rdc_query_construction_20260913',
        'phase3066',
        'omega_p63_last_layer_flip_anatomy',
        'omega_p63_last_layer_flip_anatomy'
        '.npz')
    z = np.load(npz66)
    c66 = z['COS_LAD']
    a1_diff = float(max(
        np.max(np.abs(COS_LAD_34 - c66[34])),
        np.max(np.abs(COS_LAD_35 - c66[35]))))
    a1_ok = bool(a1_diff == 0.0)
    log('a1 ladder rows 34/35 vs 3066 npz: '
        'max|d|=%.3e ok=%s'
        % (a1_diff, a1_ok))
else:
    log('a1 skipped (smoke)')

if not SMOKE:
    assert abs(med_c_34
               - 0.1487826048372403) < 1e-12, \
        med_c_34
    assert abs(med_c_35
               - (-0.35579100779974404)) \
        < 1e-12, med_c_35
    log('med_c reference assert ok '
        '(3066 result.json values)')

# a2b: lens calibration at the last layer
a2b_diff = abs(pf35 - med_c_35)
a2b_ok = bool(a2b_diff <= 0.01)
log('a2b lens calibration |PF35 - med_c(35)|='
    '%.4f ok=%s' % (a2b_diff, a2b_ok))

# ==== E2 linear Jacobian (recorded) ====
mlp35 = layers[NL - 1].mlp
Wg32 = mlp35.gate_proj.weight.detach().float()
Wu32 = mlp35.up_proj.weight.detach().float()
Wd32 = mlp35.down_proj.weight.detach().float()
colnorm_np = Wd32.norm(dim=0).double() \
    .cpu().numpy()
log('jacobian weights fp32: %.1f MB'
    % ((Wg32.numel() + Wu32.numel()
        + Wd32.numel()) * 4 / 1e6))
JC_ACT_A = np.full(NP_USE, np.nan)
JC_ACT_B = np.full(NP_USE, np.nan)
JR_ACT_A = np.full(NP_USE, np.nan)
JR_ACT_B = np.full(NP_USE, np.nan)
JC_M_A = np.full(NP_USE, np.nan)
JC_M_B = np.full(NP_USE, np.nan)
JR_M_A = np.full(NP_USE, np.nan)
JR_M_B = np.full(NP_USE, np.nan)
ANG_AB = np.full(NP_USE, np.nan)
ANG_ZAB = np.full(NP_USE, np.nan)
NDZ_A = np.full(NP_USE, np.nan)
NDZ_B = np.full(NP_USE, np.nan)
NDM_A = np.full(NP_USE, np.nan)
NDM_B = np.full(NP_USE, np.nan)
for k in range(NP_USE):
    b, base_i, pref_i, off = pair_idx(k)
    z0 = torch.tensor(BZ35[base_i],
                      device='cuda').float()
    g0 = torch.tensor(BG35[base_i],
                      device='cuda').float()
    u0 = torch.tensor(BU35[base_i],
                      device='cuda').float()
    sig = torch.sigmoid(g0)
    sp = sig * (1.0 + g0 * (1.0 - sig))
    sg = sig * g0
    dzs = []
    for s in (0, 1):
        dz = torch.tensor(
            ZZ[s, k] - BZ35[base_i],
            device='cuda').float()
        dzs.append(dz)
        wgdz = F.linear(dz, Wg32)
        wudz = F.linear(dz, Wu32)
        dact_lin = (sp * u0) * wgdz \
            + sg * wudz
        dm_lin = F.linear(dact_lin, Wd32)
        dact_act = ZACT[s, k] \
            - BACT35[base_i]
        dm_true = ZM[s, k] - BM[base_i, 35]
        dact_lin_np = dact_lin.double() \
            .cpu().numpy()
        dm_lin_np = dm_lin.double() \
            .cpu().numpy()
        jc = cosv(dact_lin_np, dact_act)
        jm = cosv(dm_lin_np, dm_true)
        ra = float(np.linalg.norm(
            dact_lin_np - dact_act)) \
            / max(float(np.linalg.norm(
                dact_act)), 1e-12)
        rm = float(np.linalg.norm(
            dm_lin_np - dm_true)) \
            / max(float(np.linalg.norm(
                dm_true)), 1e-12)
        if s == 0:
            JC_ACT_A[k] = jc
            JR_ACT_A[k] = ra
            JC_M_A[k] = jm
            JR_M_A[k] = rm
            NDZ_A[k] = float(np.linalg.norm(
                ZZ[s, k] - BZ35[base_i]))
            NDM_A[k] = float(np.linalg.norm(
                dm_true))
        else:
            JC_ACT_B[k] = jc
            JR_ACT_B[k] = ra
            JC_M_B[k] = jm
            JR_M_B[k] = rm
            NDZ_B[k] = float(np.linalg.norm(
                ZZ[s, k] - BZ35[base_i]))
            NDM_B[k] = float(np.linalg.norm(
                dm_true))
    ANG_ZAB[k] = cosv(
        (ZZ[0, k] - BZ35[base_i]),
        (ZZ[1, k] - BZ35[base_i]))
    ANG_AB[k] = cosv(
        (ZM[0, k] - BM[base_i, 35]),
        (ZM[1, k] - BM[base_i, 35]))
jac_ca_a = float(np.median(JC_ACT_A))
jac_ca_b = float(np.median(JC_ACT_B))
jac_ra_a = float(np.median(JR_ACT_A))
jac_ra_b = float(np.median(JR_ACT_B))
jac_cm_a = float(np.median(JC_M_A))
jac_cm_b = float(np.median(JC_M_B))
jac_rm_a = float(np.median(JR_M_A))
jac_rm_b = float(np.median(JR_M_B))
ang_ab = float(np.median(ANG_AB))
ang_zab = float(np.median(ANG_ZAB))
log('E2 jacobian: cos_act A/B=%.4f/%.4f '
    'rel A/B=%.4f/%.4f | cos_m A/B=%.4f/%.4f '
    'rel A/B=%.4f/%.4f'
    % (jac_ca_a, jac_ca_b, jac_ra_a,
       jac_ra_b, jac_cm_a, jac_cm_b,
       jac_rm_a, jac_rm_b))
log('E2 angles: cos(dm_A,dm_B)=%.4f '
    'cos(dz_A,dz_B)=%.4f | norms dz '
    'A/B=%.4f/%.4f dm A/B=%.4f/%.4f'
    % (ang_ab, ang_zab,
       float(np.median(NDZ_A)),
       float(np.median(NDZ_B)),
       float(np.median(NDM_A)),
       float(np.median(NDM_B))))

# CMREAD: m-channel readout polarity per state
# W32 @ (norm(h1+m) - norm(h1)) vs TT[k];
# h1 = x + a bf16 (model-identical add)


def mlp_read(xs64, as64, m64, k):
    x_ = torch.tensor(xs64,
                      device='cuda') \
        .to(torch.bfloat16)
    a_ = torch.tensor(as64,
                      device='cuda') \
        .to(torch.bfloat16)
    m_ = torch.tensor(m64,
                      device='cuda') \
        .to(torch.bfloat16)
    h1 = x_ + a_
    with torch.no_grad():
        d = (final_norm(h1 + m_).float()
             - final_norm(h1).float()) \
            .unsqueeze(0)
        u = F.linear(d, W32)[0]
        t = TTG[k]
        num = float((u * t).sum())
        den = float(u.norm()) * float(TTn[k])
        return num / max(den, 1e-12)


CMREAD_BASE = np.full(NP_USE, np.nan)
CMREAD_A = np.full(NP_USE, np.nan)
CMREAD_B = np.full(NP_USE, np.nan)
for k in range(NP_USE):
    b, base_i, pref_i, off = pair_idx(k)
    CMREAD_BASE[k] = mlp_read(
        BX[base_i, 35], BA[base_i, 35],
        BM[base_i, 35], k)
    CMREAD_A[k] = mlp_read(
        ZX[0, k], ZA[0, k], ZM[0, k], k)
    CMREAD_B[k] = mlp_read(
        ZX[1, k], ZA[1, k], ZM[1, k], k)
cm_base = float(np.median(CMREAD_BASE))
cm_a = float(np.median(CMREAD_A))
cm_b = float(np.median(CMREAD_B))
log('CMREAD m-channel polarity base/A/B='
    '%.4f/%.4f/%.4f' % (cm_base, cm_a, cm_b))

del Wg32, Wu32, Wd32, mlp35
gc.collect()
torch.cuda.empty_cache()

# ==== E1.5 neuron score -> S_A / S_R ====
DA = np.zeros((NP_USE, INTER))
for k in range(NP_USE):
    b, base_i, pref_i, off = pair_idx(k)
    DA[k] = np.abs(ZACT[0, k] - BACT35[base_i])
score = np.median(
    DA * colnorm_np[None, :], axis=0)
S_A = np.argsort(-score)[:K_TOP].astype(
    np.int64)
rng = np.random.RandomState(SEED_RAND)
comp = np.setdiff1d(np.arange(INTER), S_A)
S_R = np.sort(rng.choice(
    comp, size=K_TOP,
    replace=False)).astype(np.int64)
log('E1.5 score: max=%.4f med=%.4f | top16 '
    'idx=%s'
    % (float(score.max()), float(np.median(
        score)),
       S_A[:16].tolist()))
log('E1.5 top16 scores=%s'
    % [round(float(score[j]), 5)
       for j in S_A[:16]])

# ==== b7 identity self-swap sham ====
b, base_i, pref_i, off = pair_idx(0)
lg7 = forward_gen(
    assembled[base_i]['ids'],
    act_swap=(S_A, BACT35[base_i][S_A]))[0]
b7_diff = float(np.max(np.abs(
    lg7 - LG[base_i])))
b7_ok = bool(b7_diff == 0.0)
log('b7 identity self-swap diff=%.3e ok=%s'
    % (b7_diff, b7_ok))

# ==== E3 causal swap-to-base ====
GROUPS = (('A35', 35, S_A), ('R35', 35, S_R),
          ('B34', 34, S_A))
PERM = {}
for gname, l, sset in GROUPS:
    cs = np.full(K3, np.nan)
    for k in range(K3):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[l, ridx, :] = VB[pref_i][
            l, off + ridx, :]
        lg = forward_gen(
            assembled[base_i]['ids'],
            repl=repV,
            act_swap=(sset,
                      BACT35[base_i][sset]))[0]
        cs[k] = cosv(lg - LG[base_i], TT[k])
    PERM[gname] = cs
    log('E3 %s: perm=%s med=%.4f'
        % (gname, np.round(cs, 4).tolist(),
           float(np.median(cs))))
c_perm_A35 = float(np.median(PERM['A35']))
c_perm_R35 = float(np.median(PERM['R35']))
c_perm_B34 = float(np.median(PERM['B34']))
recov35 = c_perm_A35 - med_c_35
recov_rand = c_perm_R35 - med_c_35
delta34 = c_perm_B34 - med_c_34
log('E3 recovery: recov35=%.4f recov_rand='
    '%.4f delta34=%.4f (forwards=%d)'
    % (recov35, recov_rand, delta34, FW[0]))

# ==== verdict ====
setup_ok = bool(b0_ok and b1_ok and b3_ok
                and b4_ok and b6_ok and b7_ok
                and a2b_ok)
if not setup_ok:
    verdict = 'setup_failed_mlp_reversal'
elif (not SMOKE) and (not a1_ok):
    verdict = 'anchor_mismatch_3066_ladder'
elif recov35 < 0.10:
    verdict = 'mlp_reversal_distributed_topk'
elif abs(recov_rand) > 0.05:
    verdict = 'mlp_reversal_nonspecific_topk'
elif abs(delta34) <= 0.05:
    verdict = 'mlp_reversal_localized_statemode'
else:
    verdict = 'mlp_reversal_localized_shared'
log('VERDICT: %s (setup_ok=%s a1_ok=%s '
    'smoke=%s)'
    % (verdict, setup_ok, a1_ok, SMOKE))

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
    'A2_MAX': np.float64(a2_max),
    'A2_BIT': np.float64(a2_bit),
    'A2B_DIFF': np.float64(a2b_diff),
    'B0_DIFF': np.float64(b0_diff),
    'B1_DIFF': np.float64(b1_diff),
    'B4_DIFF': np.float64(b4_diff),
    'B5_DIFF': np.float64(b5_diff),
    'B6_DIFF': np.float64(b6_diff),
    'B7_DIFF': np.float64(b7_diff),
    'B3_OK': np.bool_(b3_ok),
    'B4_OK': np.bool_(b4_ok),
    'B6_OK': np.bool_(b6_ok),
    'B7_OK': np.bool_(b7_ok),
    'SETUP_OK': np.bool_(setup_ok),
    'COS_LAD_34': COS_LAD_34,
    'COS_LAD_35': COS_LAD_35,
    'MED_C_34': np.float64(med_c_34),
    'MED_C_35': np.float64(med_c_35),
    'PA34': PA34, 'PA35': PA35,
    'PF34': PF34, 'PF35': PF35,
    'JC_ACT_A': JC_ACT_A, 'JC_ACT_B': JC_ACT_B,
    'JR_ACT_A': JR_ACT_A, 'JR_ACT_B': JR_ACT_B,
    'JC_M_A': JC_M_A, 'JC_M_B': JC_M_B,
    'JR_M_A': JR_M_A, 'JR_M_B': JR_M_B,
    'ANG_AB': ANG_AB, 'ANG_ZAB': ANG_ZAB,
    'NDZ_A': NDZ_A, 'NDZ_B': NDZ_B,
    'NDM_A': NDM_A, 'NDM_B': NDM_B,
    'CMREAD_BASE': CMREAD_BASE,
    'CMREAD_A': CMREAD_A,
    'CMREAD_B': CMREAD_B,
    'SCORE': score.astype(np.float32),
    'S_A': S_A, 'S_R': S_R,
    'PERM_A35': PERM['A35'],
    'PERM_R35': PERM['R35'],
    'PERM_B34': PERM['B34'],
    'C_PERM_A35': np.float64(c_perm_A35),
    'C_PERM_R35': np.float64(c_perm_R35),
    'C_PERM_B34': np.float64(c_perm_B34),
    'RECOV35': np.float64(recov35),
    'RECOV_RAND': np.float64(recov_rand),
    'DELTA34': np.float64(delta34),
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
        'med_c_35': f64(med_c_35),
        'pa34': f64(pa34), 'pa35': f64(pa35),
        'pf34': f64(pf34), 'pf35': f64(pf35),
        'jac_cos_act_a': f64(jac_ca_a),
        'jac_cos_act_b': f64(jac_ca_b),
        'jac_rel_act_a': f64(jac_ra_a),
        'jac_rel_act_b': f64(jac_ra_b),
        'jac_cos_m_a': f64(jac_cm_a),
        'jac_cos_m_b': f64(jac_cm_b),
        'jac_rel_m_a': f64(jac_rm_a),
        'jac_rel_m_b': f64(jac_rm_b),
        'ang_ab': f64(ang_ab),
        'ang_zab': f64(ang_zab),
        'norm_dz_a': f64(float(np.median(
            NDZ_A))),
        'norm_dz_b': f64(float(np.median(
            NDZ_B))),
        'norm_dm_a': f64(float(np.median(
            NDM_A))),
        'norm_dm_b': f64(float(np.median(
            NDM_B))),
        'cmread_base': f64(cm_base),
        'cmread_a': f64(cm_a),
        'cmread_b': f64(cm_b),
        'score_max': f64(float(score.max())),
        'score_top16': [
            [int(j), f64(score[j])]
            for j in S_A[:16]],
        'c_perm_a35': f64(c_perm_A35),
        'c_perm_r35': f64(c_perm_R35),
        'c_perm_b34': f64(c_perm_B34),
        'recov35': f64(recov35),
        'recov_rand': f64(recov_rand),
        'delta34': f64(delta34),
    },
    'anchors': {
        'a1_diff': a1_diff, 'a1_ok': a1_ok,
        'a2_max': a2_max, 'a2_ok': a2_ok,
        'a2_recorded_only': a2_recorded_only,
        'a2b_diff': a2b_diff, 'a2b_ok': a2b_ok,
        'b0_diff': b0_diff, 'b0_ok': b0_ok,
        'b1_diff': b1_diff, 'b1_ok': b1_ok,
        'b3_ok': b3_ok,
        'b4_diff': b4_diff, 'b4_ok': b4_ok,
        'b5_diff': b5_diff,
        'b6_diff': b6_diff, 'b6_ok': b6_ok,
        'b7_diff': b7_diff, 'b7_ok': b7_ok,
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
