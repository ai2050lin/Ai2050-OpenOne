# -*- coding: utf-8 -*-
"""Phase 3068: Omega-P65 S_B symmetric selection +
competition decomposition (qwen3-4b single model
bf16).

Question (3068 A, menu of 3067): the 3067 verdict
mlp_reversal_localized_shared left the competition
quantitative picture open - is there ONE shared
neuron pool whose gain is state-dependent, or TWO
partially distinct sub-populations (S_A negative
writer vs S_B positive writer)?  Decompose.

E1 dual-state capture (3067-identical): l in
{34,35} x N pairs, repV 3065-identical (a1 bit
anchor vs the 3066 npz COS_LAD rows 34/35);
captures at the L35 last position: act (down_proj
in) and m (mlp out) banked per state (A = inj@35,
B = inj@34); PA/PF diag lens probes recorded.
E1.5 symmetric scores: score_s[j] = median_k(
|dact_s[k][j]| * ||W_down[:,j]||) for s in {A,B};
S_A = top-128 (a2 anchor: set equality vs the
3067 npz S_A); S_B = top-128 by score_B (new);
OVL = |S_A n S_B| + overlap curve |top_m(A) n
top_m(B)| for m in {32,64,128,256,512,1024,2048};
S_R = random 128 from the complement of S_A u S_B
(seed 3024, sorted).
E2 exact linear decomposition (recorded): m =
W_down act is LINEAR, so with the partition
P1 = S_A - S_B, P2 = S_B - S_A, P3 = S_A ^ S_B,
P4 = rest, the diff splits exactly:
  v_i = W_down[:,P_i] @ dact[P_i]
  sum_i v_i = W_down @ dact ~= dm (lincheck cos)
per state per pair: norm shares ||v_i||/||dm|| and
signed TT projections cproj_i = (W32 @ v_i . TT)/
||TT|| (fp32 unembed WITHOUT final norm -
declared approximation); medians over pairs.
E3 causal swap-to-base (down_proj forward-PRE-
hook at L35, last position only; base self-swap
= identity, b7): EIGHT groups = {A35, B34} x
{S_A, S_B, S_R, ALL}; ALL = arange(9728) ->
m = m_base bit-exact (L35 MLP diff fully
neutralized, the rest-of-system anchor); c_perm =
med_k cosv fp64.  a3 anchors: PERM(A35,S_A) and
PERM(B34,S_A) bit-equal vs the 3067 npz
PERM_A35 / PERM_B34 (same swaps, same forwards).
anchors: a1 rows 34/35 bit 0.0 vs 3066 npz
(hard); a2 S_A set equality vs 3067 npz (hard);
a3 perm bit 0.0 vs 3067 npz (hard); med_c
reference assert; a2b |PF35 - med_c(35)| <= 0.01;
b0/b1/b4/b5/b6/b7 as 3067; lincheck recorded.
verdict: setup fail -> setup_failed_competition;
a1 fail -> anchor_mismatch_3066_ladder; a2 fail
-> anchor_mismatch_3067_sA; a3 fail ->
anchor_mismatch_3067_perm; ov = |S_A n S_B|;
pool = sharedpool if ov >= 96 else splitpool;
recov_B_Sb = c_perm(B34,S_B) - med_c(34):
>= 0.10 -> top_negative_both (top-dact neurons
negative-aligned in BOTH states); <= -0.10 ->
dedicated_positive_sb (S_B carries the positive
write in state B); else sb_diffuse; verdict =
competition_{pool}_{eff}.
memory discipline: single model; banks fp64
CPU; Wd32 fp32 copy (~100 MB) deleted after
E2; W32 resident; del + gc + empty_cache at
the end.
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

PHASE = 3068
NAME = 'omega_p65_sb_symmetric_competition'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3068', NAME)
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
SEED_RAND = 3024
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
            'seed 3020; E1 protocol 3067/3066-'
            'identical (a1 bit anchor vs 3066 '
            'npz COS_LAD rows 34/35, hard; a2 '
            'S_A set equality vs 3067 npz; a3 '
            'PERM(A35,S_A)/PERM(B34,S_A) bit 0.0 '
            'vs 3067 npz); lens probes fp32 W32 '
            'resident; verdict scalars fp64 '
            'cosv; smoke mode optional (SMOKE=1: '
            '8 E1 pairs + 4 E3 pairs per group, '
            'a1/a2/a3 off)',
    'question': '3068 A (menu of 3067): one '
                'shared neuron pool with state-'
                'dependent gain, or two partially '
                'distinct sub-populations (S_A '
                'negative writer vs S_B positive '
                'writer)?  Symmetric S_B selection, '
                '8-group causal swaps incl. the '
                'ALL-swap rest anchor, and the '
                'exact linear competition '
                'decomposition.',
    'E1_dual_state': '3067-identical: l in '
                     '{34,35} x N pairs, repV '
                     '3065-identical; bank act/m '
                     'at the L35 last position '
                     'per state (A=inj@35, '
                     'B=inj@34); COS_LAD rows '
                     '34/35 = a1 anchor',
    'E1_5_scores': 'score_s[j] = median_k(|dact_'
                   's[k][j]| * ||W_down[:,j]||), '
                   's in {A,B}; S_A = top-128 (a2 '
                   'anchor vs 3067 npz); S_B = '
                   'top-128 by score_B; OVL = '
                   '|S_A n S_B| + overlap curve '
                   'm in {32,64,128,256,512,1024,'
                   '2048}; S_R = random 128 from '
                   'complement of S_A u S_B (seed '
                   '3024, sorted)',
    'E2_decomposition': 'm = W_down act is '
                        'linear: partition P1 = '
                        'S_A\\S_B, P2 = S_B\\S_A, '
                        'P3 = S_A n S_B, P4 = '
                        'rest; v_i = W_down[:,P_i]'
                        ' @ dact[P_i]; per state '
                        'per pair norm shares + '
                        'signed TT projections '
                        'cproj_i = (W32 @ v_i . '
                        'TT)/||TT|| (fp32 unembed '
                        'without final norm, '
                        'declared); lincheck '
                        'cos(sum v_i, dm) '
                        'recorded',
    'E3_swaps': 'down_proj forward-PRE-hook at '
                'L35, swap-to-base at the last '
                'position; EIGHT groups = {A35, '
                'B34} x {S_A, S_B, S_R, ALL}; '
                'ALL -> m = m_base bit-exact '
                '(MLP diff neutralized, '
                'rest-of-system anchor); '
                'base self-swap = identity (b7) '
                'so no matched base forward '
                'needed',
    'anchors': 'a1 rows 34/35 bit 0.0 vs 3066 '
               'npz (hard); a2 S_A set equality '
               'vs 3067 npz (hard); a3 '
               'PERM(A35,S_A)/PERM(B34,S_A) bit '
               '0.0 vs 3067 npz (hard); med_c '
               'reference assert (0.1487826048372'
               '403 / -0.35579100779974404); '
               'a2b |PF35 - med_c(35)| <= 0.01; '
               'b0 recapture bit 0.0; b1 sham '
               'bit 0.0; b3 finite; b4 delta-x '
               'at injection layer bit 0.0; b5 '
               'act = silu(g)*u bf16 '
               'reconstruction (recorded); b6 '
               '(x+a)+m = h2 bf16 bit 0.0; b7 '
               'identity self-swap bit 0.0; '
               'lincheck recorded (soft)',
    'verdict': 'setup fail -> '
               'setup_failed_competition; a1 '
               'fail -> anchor_mismatch_3066_'
               'ladder; a2 fail -> '
               'anchor_mismatch_3067_sA; a3 '
               'fail -> anchor_mismatch_3067_'
               'perm; pool = sharedpool if '
               '|S_A n S_B| >= 96 else '
               'splitpool; recov_B_Sb >= 0.10 '
               '-> top_negative_both; <= -0.10 '
               '-> dedicated_positive_sb; else '
               'sb_diffuse; verdict = '
               'competition_{pool}_{eff}',
    'statistics_discipline': 'same-precision '
                             'bit anchors on '
                             'frozen seeds; TT '
                             'logit-space 3065-'
                             'identical; cproj is '
                             'a declared unnorm '
                             'approximation (no '
                             'final norm); '
                             'random control '
                             'disjoint from both '
                             'selected sets',
    'memory_discipline': 'single model; banks '
                         'fp64 CPU; Wd32 fp32 '
                         'copy deleted after E2; '
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
log('banks: LG%s VB%s BACT35%s (forwards=%d)'
    % (LG.shape, VB.shape, BACT35.shape, FW[0]))
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


# ==== E1 dual-state capture ====
COS_LAD_34 = np.full(NP_USE, np.nan)
COS_LAD_35 = np.full(NP_USE, np.nan)
PA34 = np.full(NP_USE, np.nan)
PA35 = np.full(NP_USE, np.nan)
PF34 = np.full(NP_USE, np.nan)
PF35 = np.full(NP_USE, np.nan)
# state 0 = A (inj@35), state 1 = B (inj@34)
ZACT = np.zeros((2, NP_USE, INTER))
ZM = np.zeros((2, NP_USE, HID))
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
        ZACT[si_, k] = zACT[NL - 1].double() \
            .cpu().numpy()
        ZM[si_, k] = zM[NL - 1].double() \
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
npz66 = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3066',
    'omega_p63_last_layer_flip_anatomy',
    'omega_p63_last_layer_flip_anatomy.npz')
if not SMOKE:
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
    log('med_c reference assert ok')

a2b_diff = abs(pf35 - med_c_35)
a2b_ok = bool(a2b_diff <= 0.01)
log('a2b lens calibration |PF35 - med_c(35)|='
    '%.4f ok=%s' % (a2b_diff, a2b_ok))

# ==== E1.5 symmetric scores ====
Wd32 = layers[NL - 1].mlp.down_proj.weight \
    .detach().float()
colnorm_np = Wd32.norm(dim=0).double() \
    .cpu().numpy()
DA_A = np.zeros((NP_USE, INTER))
DA_B = np.zeros((NP_USE, INTER))
for k in range(NP_USE):
    b, base_i, pref_i, off = pair_idx(k)
    DA_A[k] = np.abs(ZACT[0, k]
                     - BACT35[base_i])
    DA_B[k] = np.abs(ZACT[1, k]
                     - BACT35[base_i])
score_A = np.median(
    DA_A * colnorm_np[None, :], axis=0)
score_B = np.median(
    DA_B * colnorm_np[None, :], axis=0)
S_A = np.argsort(-score_A)[:K_TOP].astype(
    np.int64)
S_B = np.argsort(-score_B)[:K_TOP].astype(
    np.int64)
ovl = int(len(np.intersect1d(S_A, S_B)))
OVL_M = (32, 64, 128, 256, 512, 1024, 2048)
ovl_curve = []
for m in OVL_M:
    ta = np.argsort(-score_A)[:m]
    tb = np.argsort(-score_B)[:m]
    ovl_curve.append(int(len(
        np.intersect1d(ta, tb))))
rng = np.random.RandomState(SEED_RAND)
union = np.union1d(S_A, S_B)
comp = np.setdiff1d(np.arange(INTER), union)
S_R = np.sort(rng.choice(
    comp, size=K_TOP,
    replace=False)).astype(np.int64)
log('E1.5 scores: A max=%.4f med=%.4f | B '
    'max=%.4f med=%.4f | OVL=%d curve=%s'
    % (float(score_A.max()),
       float(np.median(score_A)),
       float(score_B.max()),
       float(np.median(score_B)), ovl,
       ovl_curve))
log('E1.5 S_B top16 idx=%s scores=%s'
    % (S_B[:16].tolist(),
       [round(float(score_B[j]), 5)
        for j in S_B[:16]]))

# a2: S_A set equality vs 3067 npz
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
        S_A, sa67)) + len(np.setdiff1d(
            sa67, S_A)))
    a2set_ok = bool(a2set_diff == 0)
    log('a2 S_A set vs 3067 npz: |diff|=%d '
        'ok=%s' % (a2set_diff, a2set_ok))
else:
    log('a2 skipped (smoke)')

# ==== E2 exact linear decomposition ====
P1 = np.setdiff1d(S_A, S_B)
P2 = np.setdiff1d(S_B, S_A)
P3 = np.intersect1d(S_A, S_B)
P4 = np.setdiff1d(np.arange(INTER), union)
log('E2 partition sizes: P1(S_A only)=%d '
    'P2(S_B only)=%d P3(both)=%d P4(rest)=%d'
    % (len(P1), len(P2), len(P3), len(P4)))
PARTS = (('P1', P1), ('P2', P2), ('P3', P3),
         ('P4', P4))
SHARE = {nm: np.zeros((2, NP_USE))
         for nm, _ in PARTS}
CPROJ = {nm: np.zeros((2, NP_USE))
         for nm, _ in PARTS}
COSU = {nm: np.zeros((2, NP_USE))
        for nm, _ in PARTS}
LIN = np.zeros((2, NP_USE))
for k in range(NP_USE):
    b, base_i, pref_i, off = pair_idx(k)
    for s in (0, 1):
        dact = torch.tensor(
            ZACT[s, k] - BACT35[base_i],
            device='cuda').float()
        dm = ZM[s, k] - BM[base_i, 35]
        dm_norm = max(float(np.linalg.norm(
            dm)), 1e-12)
        vsum = torch.zeros(HID,
                           device='cuda')
        for nm, P in PARTS:
            if len(P) == 0:
                SHARE[nm][s, k] = 0.0
                CPROJ[nm][s, k] = 0.0
                COSU[nm][s, k] = 0.0
                continue
            v = F.linear(
                dact[torch.tensor(
                    np.ascontiguousarray(P),
                    device='cuda')],
                Wd32[:, torch.tensor(
                    np.ascontiguousarray(P),
                    device='cuda')])
            vsum = vsum + v
            SHARE[nm][s, k] = float(
                v.norm()) / dm_norm
            with torch.no_grad():
                u = F.linear(
                    v.unsqueeze(0), W32)[0]
            u_np = u.double().cpu().numpy()
            t = TT[k]
            tn = max(float(np.linalg.norm(
                t)), 1e-12)
            CPROJ[nm][s, k] = float(
                u_np @ t) / tn
            COSU[nm][s, k] = cosv(u_np, t)
        lin = cosv(vsum.double().cpu()
                   .numpy(), dm)
        LIN[s, k] = lin
lin_med = [float(np.median(LIN[s]))
           for s in (0, 1)]
for s, sname in ((0, 'A'), (1, 'B')):
    log('E2 state %s: lincheck=%.4f | shares '
        'P1/P2/P3/P4=%.3f/%.3f/%.3f/%.3f'
        % (sname, lin_med[s],
           float(np.median(SHARE['P1'][s])),
           float(np.median(SHARE['P2'][s])),
           float(np.median(SHARE['P3'][s])),
           float(np.median(SHARE['P4'][s]))))
    log('E2 state %s: cproj P1/P2/P3/P4='
        '%.1f/%.1f/%.1f/%.1f (cosu '
        '%.3f/%.3f/%.3f/%.3f)'
        % (sname,
           float(np.median(CPROJ['P1'][s])),
           float(np.median(CPROJ['P2'][s])),
           float(np.median(CPROJ['P3'][s])),
           float(np.median(CPROJ['P4'][s])),
           float(np.median(COSU['P1'][s])),
           float(np.median(COSU['P2'][s])),
           float(np.median(COSU['P3'][s])),
           float(np.median(COSU['P4'][s]))))
del Wd32
gc.collect()
torch.cuda.empty_cache()

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

# ==== E3 eight-group causal swaps ====
ALLN = np.arange(INTER)
GROUPS = (('A35_SA', 35, S_A),
          ('A35_SB', 35, S_B),
          ('A35_SR', 35, S_R),
          ('A35_ALL', 35, ALLN),
          ('B34_SA', 34, S_A),
          ('B34_SB', 34, S_B),
          ('B34_SR', 34, S_R),
          ('B34_ALL', 34, ALLN))
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
    log('E3 %s: med=%.4f %s'
        % (gname, float(np.median(cs)),
           np.round(cs, 4).tolist()
           if K3 <= 8 else ''))

# a3: perm replication anchors vs 3067 npz
a3_diff = None
a3_ok = False
if not SMOKE:
    p67 = np.load(npz67)
    a3_diff = float(max(
        np.max(np.abs(PERM['A35_SA']
                      - p67['PERM_A35'])),
        np.max(np.abs(PERM['B34_SA']
                      - p67['PERM_B34']))))
    a3_ok = bool(a3_diff == 0.0)
    log('a3 perm vs 3067 npz (A35_SA/B34_SA): '
        'max|d|=%.3e ok=%s'
        % (a3_diff, a3_ok))
else:
    log('a3 skipped (smoke)')

cp = {g: float(np.median(PERM[g]))
      for g, _, _ in GROUPS}
recov = {
    'A_Sa': cp['A35_SA'] - med_c_35,
    'A_Sb': cp['A35_SB'] - med_c_35,
    'A_Sr': cp['A35_SR'] - med_c_35,
    'A_all': cp['A35_ALL'] - med_c_35,
    'B_Sa': cp['B34_SA'] - med_c_34,
    'B_Sb': cp['B34_SB'] - med_c_34,
    'B_Sr': cp['B34_SR'] - med_c_34,
    'B_all': cp['B34_ALL'] - med_c_34,
}
log('E3 recovery: A_Sa=%.4f A_Sb=%.4f '
    'A_Sr=%.4f A_all=%.4f | B_Sa=%.4f '
    'B_Sb=%.4f B_Sr=%.4f B_all=%.4f '
    '(forwards=%d)'
    % (recov['A_Sa'], recov['A_Sb'],
       recov['A_Sr'], recov['A_all'],
       recov['B_Sa'], recov['B_Sb'],
       recov['B_Sr'], recov['B_all'],
       FW[0]))

# ==== verdict ====
setup_ok = bool(b0_ok and b1_ok and b3_ok
                and b4_ok and b6_ok and b7_ok
                and a2b_ok)
pool = 'sharedpool' if ovl >= 96 \
    else 'splitpool'
if recov['B_Sb'] >= 0.10:
    eff = 'top_negative_both'
elif recov['B_Sb'] <= -0.10:
    eff = 'dedicated_positive_sb'
else:
    eff = 'sb_diffuse'
if not setup_ok:
    verdict = 'setup_failed_competition'
elif (not SMOKE) and (not a1_ok):
    verdict = 'anchor_mismatch_3066_ladder'
elif (not SMOKE) and (not a2set_ok):
    verdict = 'anchor_mismatch_3067_sA'
elif (not SMOKE) and (not a3_ok):
    verdict = 'anchor_mismatch_3067_perm'
else:
    verdict = 'competition_' + pool + '_' + eff
log('VERDICT: %s (setup_ok=%s a1_ok=%s '
    'a2_ok=%s a3_ok=%s ovl=%d recov_B_Sb='
    '%.4f smoke=%s)'
    % (verdict, setup_ok, a1_ok, a2set_ok,
       a3_ok, ovl, recov['B_Sb'], SMOKE))

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
    'A2SET_DIFF': np.float64(
        a2set_diff if a2set_diff is not None
        else np.nan),
    'A2SET_OK': np.bool_(a2set_ok),
    'A3_DIFF': np.float64(
        a3_diff if a3_diff is not None
        else np.nan),
    'A3_OK': np.bool_(a3_ok),
    'A2LENS_MAX': np.float64(a2_max),
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
    'SCORE_A': score_A.astype(np.float32),
    'SCORE_B': score_B.astype(np.float32),
    'S_A': S_A, 'S_B': S_B, 'S_R': S_R,
    'OVL': np.int64(ovl),
    'OVL_M': np.array(OVL_M, dtype=np.int64),
    'OVL_CURVE': np.array(ovl_curve,
                          dtype=np.int64),
    'LIN': LIN,
    'SHARE_P1': SHARE['P1'],
    'SHARE_P2': SHARE['P2'],
    'SHARE_P3': SHARE['P3'],
    'SHARE_P4': SHARE['P4'],
    'CPROJ_P1': CPROJ['P1'],
    'CPROJ_P2': CPROJ['P2'],
    'CPROJ_P3': CPROJ['P3'],
    'CPROJ_P4': CPROJ['P4'],
    'COSU_P1': COSU['P1'],
    'COSU_P2': COSU['P2'],
    'COSU_P3': COSU['P3'],
    'COSU_P4': COSU['P4'],
    'PERM_A35_SA': PERM['A35_SA'],
    'PERM_A35_SB': PERM['A35_SB'],
    'PERM_A35_SR': PERM['A35_SR'],
    'PERM_A35_ALL': PERM['A35_ALL'],
    'PERM_B34_SA': PERM['B34_SA'],
    'PERM_B34_SB': PERM['B34_SB'],
    'PERM_B34_SR': PERM['B34_SR'],
    'PERM_B34_ALL': PERM['B34_ALL'],
    'RECOV_A_SA': np.float64(recov['A_Sa']),
    'RECOV_A_SB': np.float64(recov['A_Sb']),
    'RECOV_A_SR': np.float64(recov['A_Sr']),
    'RECOV_A_ALL': np.float64(recov['A_all']),
    'RECOV_B_SA': np.float64(recov['B_Sa']),
    'RECOV_B_SB': np.float64(recov['B_Sb']),
    'RECOV_B_SR': np.float64(recov['B_Sr']),
    'RECOV_B_ALL': np.float64(recov['B_all']),
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
        'ovl': int(ovl),
        'ovl_curve': {
            str(m): int(v) for m, v in
            zip(OVL_M, ovl_curve)},
        'partition_sizes': {
            'P1': int(len(P1)), 'P2': int(len(P2)),
            'P3': int(len(P3)), 'P4': int(len(P4))},
        'lincheck_a': f64(lin_med[0]),
        'lincheck_b': f64(lin_med[1]),
        'share': {nm: {'a': f64(float(
                        np.median(SHARE[nm][0]))),
                    'b': f64(float(
                        np.median(SHARE[nm][1])))}
                  for nm, _ in PARTS},
        'cproj': {nm: {'a': f64(float(
                       np.median(CPROJ[nm][0]))),
                   'b': f64(float(
                       np.median(CPROJ[nm][1])))}
                  for nm, _ in PARTS},
        'cosu': {nm: {'a': f64(float(
                      np.median(COSU[nm][0]))),
                  'b': f64(float(
                      np.median(COSU[nm][1])))}
                 for nm, _ in PARTS},
        'c_perm': {g: f64(v) for g, v
                   in cp.items()},
        'recov': {kk: f64(v) for kk, v
                  in recov.items()},
    },
    'anchors': {
        'a1_diff': a1_diff, 'a1_ok': a1_ok,
        'a2set_diff': a2set_diff,
        'a2set_ok': a2set_ok,
        'a3_diff': a3_diff, 'a3_ok': a3_ok,
        'a2lens_max': a2_max,
        'a2lens_ok': a2lens_ok,
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
