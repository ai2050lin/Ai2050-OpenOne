# -*- coding: utf-8 -*-
"""Phase 3066: Omega-P63 last-layer flip anatomy
(qwen3-4b single model bf16).

Question (3066 A, menu of 3065): WHERE is the
qwen3-4b last-layer sign flip written - raw
V-readout polarity at the injection layer, down-
stream attention readout of the injected positions,
MLP reshaping, or the final norm?  Ladder band
L30-34 all positive (+0.101..+0.531) vs L35
-0.356 (3065).

E1 lens ladder: full ladder 36x24, forwards
  3065-identical (a1 bit anchor vs 3065 npz
  qwen3_4b_COS_LAD); per forward capture at the
  last position the per-layer quadruple (x block
  input, a attn out, m mlp out, h2 block out);
  h1 = x + a (bf16 add, bit-equal to the model's
  internal h1); lens(z) = model.model.norm(z) then
  F.linear(z, Wemb); PROP_ATTN[l,lp] = med_k
  cos(lens(h1)_inj - lens(h1)_base, TT[k]);
  PROP_FULL[l,lp] same on h2; C_RAW[l] =
  PROP_ATTN[l,l] = raw V-readout polarity.
  Lens probes: GPU bf16 weights, fp32-accumulated
  (declared precision ~1e-3; thresholds >= 0.05).
E2 causal silencing: V0 = zero the V rows at
  positions 0..3 of chosen downstream layer(s);
  for l in {25,30,34}: one-at-a-time V0@lp
  (lp in l+1..35) and cumulative V0@(l+1..35);
  every variant paired with a matched base-V0
  forward; c_sil = med_k cosv fp64 (lg_inj_var -
  lg_base_var, TT[k]); share35 = (med_c(34) -
  c_sil(34,V0@35)) / (med_c(34) - raw_pos_med),
  guard |denom| >= 0.05.
E3 head map (descriptive): o_proj input capture
  at the last position; per-head contribution
  diff c_h = W_o[:, h*128:(h+1)*128] @ (vec_inj -
  vec_base); norm shares and cos(c_h, mean final
  displacement); for inj@34 and inj@35.
E4 norm channel: c_unnorm on the h2 diff without
  final norm, l in {34,35} (recorded).
anchors: a1 ladder bit 0.0 vs 3065 npz (hard);
  a2 lens(final)=logits bit 0.0 per bank prompt
  (bit hard; |d|<=1e-3 -> recorded-only);
  a2b |PROP_FULL[35,35] - med_c(35)| <= 0.01
  (lens calibration); b0 recapture bit 0.0 (LG/
  VB/PB/BX/BA/BM/BH); b1 sham bit 0.0; b3 finite;
  b4 delta-x at the injection layer bit 0.0
  (block input unaffected); b5 V0@35-on-base
  ablation magnitude recorded.
verdict: setup fail -> setup_failed_flip_anatomy;
  a1 fail -> anchor_mismatch_3065_ladder;
  raw_pos_med = median C_RAW over band {30..34};
  raw_pos_med >= 0 -> sign_flip_rawband_lastlayer_
  {mlp_shaped|raw_pass} (|C_RAW[35] -
  C_FULL_DIAG[35]| >= 0.1 -> mlp_shaped);
  raw_pos_med < 0 and |med_c(34) - raw_pos_med|
  >= 0.05 -> share35 >= 0.5 ->
  sign_flip_downstream_L35readout_localized else
  sign_flip_downstream_distributed; else
  flip_mixed_recorded; head tag appended:
  top1 head share >= 0.5 -> _head{h} else
  _heads_dist (max over the inj@34/35 maps).
memory discipline: single model; banks fp64 CPU;
  no fp64 W_U copies (GPU bf16 lm_head reused);
  del + gc + empty_cache at the end.
"""
import gc
import hashlib
import io
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

PHASE = 3066
NAME = 'omega_p63_last_layer_flip_anatomy'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3066', NAME)
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
BAND = (30, 31, 32, 33, 34)
SIL_LAYERS = (25, 30, 34)

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
            'seed 3020; ladder protocol 3065-'
            'identical (a1 bit anchor vs 3065 '
            'npz qwen3_4b_COS_LAD, hard); lens '
            'probes: final_norm bf16 (model '
            'path) then fp32 matmul against '
            'resident W32 = Wemb.float() '
            '(+1.55 GB GPU; bf16-linear output '
            'rounding ~1 ulp/logit pollutes '
            'lens cos at ~0.25 level, removed '
            'by fp32; residual precision '
            '~1e-5); verdict scalars via fp64 '
            'numpy cosv on final logits '
            '(COS_LAD path); smoke mode '
            'optional (SMOKE=1, ladder subset, '
            'a1 off)',
    'question': '3066 A (menu of 3065): where is '
                'the qwen3-4b last-layer sign '
                'flip written - raw V-readout '
                'polarity at the injection '
                'layer, downstream attention '
                'readout of the injected '
                'positions, MLP reshaping, or '
                'the final norm? Band L30-34 '
                'all positive (+0.101..+0.531) '
                'vs L35 -0.356.',
    'E1_lens_ladder': 'full ladder 36x24 3065-'
                      'identical forwards; per '
                      'forward capture at the '
                      'last position the per-'
                      'layer quadruple (x, a, m, '
                      'h2); h1 = x + a bf16 (bit '
                      'equal to model internal); '
                      'lens = final_norm then '
                      'F.linear(z, Wemb); '
                      'PROP_ATTN[l,lp]/PROP_FULL'
                      '[l,lp] = med_k cos(lens '
                      'diff, TT[k]); C_RAW[l] = '
                      'PROP_ATTN[l,l]',
    'E2_silence': 'V0 = zero V rows at '
                  'positions 0..3 of chosen '
                  'downstream layer(s); l in '
                  '{25,30,34}; one-at-a-time '
                  'V0@lp and cumulative '
                  'V0@(l+1..35); matched base-'
                  'V0 forward per variant; '
                  'c_sil = med_k cosv fp64; '
                  'share35 = (med_c(34) - '
                  'c_sil(34,V0@35)) / (med_c(34)'
                  ' - raw_pos_med), guard '
                  '|denom| >= 0.05',
    'E3_heads': 'descriptive: o_proj input '
                'capture at last position; '
                'per-head contribution diff '
                'c_h = W_o[:, h*128:(h+1)*128] '
                '@ dv_h; norm shares + '
                'cos(c_h, total attn-write '
                'direction) at lp=35; inj@34 '
                'and inj@35',
    'E4_norm_channel': 'c_unnorm on h2 diff '
                       'without final norm, l '
                       'in {34,35} (recorded)',
    'anchors': 'a1 ladder bit 0.0 vs 3065 npz '
               '(hard); a2 lens(final h2) vs '
               'model logits fp32 max|d| <= '
               '0.125 (1 bf16 ulp) ok, bit '
               'attempt recorded; a2b '
               '|PROP_FULL[35,35] - med_c(35)| '
               '<= 0.01 (lens calibration); '
               'b0 recapture bit 0.0; b1 sham '
               'bit 0.0; b3 finite; b4 delta-x '
               'at injection layer bit 0.0; b5 '
               'V0@35-on-base magnitude '
               'recorded',
    'verdict': 'setup fail -> '
               'setup_failed_flip_anatomy; a1 '
               'fail -> '
               'anchor_mismatch_3065_ladder; '
               'raw_pos_med = median C_RAW '
               'over band {30..34}; '
               'raw_pos_med >= 0 -> '
               'sign_flip_rawband_lastlayer_{'
               'mlp_shaped|raw_pass} '
               '(|C_RAW[35]-C_FULL_DIAG[35]| '
               '>= 0.1 -> mlp_shaped); '
               'raw_pos_med < 0 and '
               '|med_c(34)-raw_pos_med| >= '
               '0.05 -> share35 >= 0.5 -> '
               'sign_flip_downstream_L35readout'
               '_localized else '
               'sign_flip_downstream_distributed'
               '; else flip_mixed_recorded; '
               'head tag appended: top1 head '
               'share >= 0.5 -> _head{h} else '
               '_heads_dist (max over inj@34/'
               '35 maps)',
    'statistics_discipline': 'same-precision '
                             'bit anchors on '
                             'frozen seeds; no '
                             'cross-space cosine '
                             'except TT (logit '
                             'space, 3065-'
                             'identical) and '
                             'lens probes '
                             '(declared); head '
                             'map descriptive',
    'memory_discipline': 'single model; banks '
                         'fp64 CPU; no fp64 W_U '
                         'copies; GPU bf16 lm_head '
                         'reused; del + gc + '
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
NVOC = int(model.config.vocab_size)
Wemb = model.get_output_embeddings().weight
final_norm = model.model.norm
assert int(Wemb.shape[0]) == NVOC
assert int(Wemb.shape[1]) == HID
# fp32 unembed resident for lens probes: bf16
# linear output rounding (~1 ulp per logit)
# pollutes lens cosines at the ~0.25 level
# (smoke a2b failure); fp32 matmul removes it.
W32 = Wemb.float()
log('qwen3-4b loaded bf16 (vocab=%d) gpu=%.2f '
    'GB (+W32 fp32 %.2f GB)'
    % (NVOC,
       torch.cuda.memory_allocated() / 1e9,
       W32.numel() * 4 / 1e9))

stateV = {li: {'repl': None, 'mask': None}
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


def forward_gen(ids, repl=None):
    """3065-identical V replacement; additionally
    returns bf16 last-position captures zX/zA/zM/
    zP (NL, HID) and zH (NL, NQ*HDIM) on GPU."""
    reset_all()
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
    for li in range(NL):
        capV[li]['rec'] = True
        capP[li]['rec'] = True
        capX[li]['rec'] = True
        capA[li]['rec'] = True
        capM[li]['rec'] = True
        capH[li]['rec'] = True
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
    reset_all()
    return lg, lg_gpu, vb, po, zX, zA, zM, zP, zH


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
a2_max = 0.0
a2_bit = 0.0
for i in range(n_pr):
    lg, lg_gpu, vb, po, zX, zA, zM, zP, zH = \
        forward_gen(assembled[i]['ids'])
    n = int(LENS[i])
    LG[i] = lg
    VB[i, :, :n, :] = vb
    PB[i, :, :n, :] = po
    BX[i] = zX.double().cpu().numpy()
    BA[i] = zA.double().cpu().numpy()
    BM[i] = zM.double().cpu().numpy()
    BH[i] = zH.double().cpu().numpy()
    # a2: lens(final h2) must match actual
    # logits (bit attempt + 1-bf16-ulp fp32)
    with torch.no_grad():
        lgt = torch.nn.functional.linear(
            final_norm(zP[NL - 1]
                       .unsqueeze(0)), Wemb)
        lgt32 = torch.nn.functional.linear(
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
log('banks: LG%s VB%s BX%s BH%s'
    % (LG.shape, VB.shape, BX.shape, BH.shape))
log('a2 lens(final)=logits: fp32 max|d|=%.3e '
    '(<=0.125 ok) bit_attempt=%.3e '
    'recorded_only=%s'
    % (a2_max, a2_bit, a2_recorded_only))

b0_diff = 0.0
for si in (0, 9, 17, 31):
    lg, lg_gpu, vb, po, zX, zA, zM, zP, zH = \
        forward_gen(assembled[si]['ids'])
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
lg_s, _, _, _, _, _, _, _, _ = forward_gen(
    assembled[base0]['ids'], repl=selfV)
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
             and np.isfinite(BH).all())
log('b3 finite=%s' % b3_ok)

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
        li_ = torch.nn.functional.linear(
            final_norm(zinj).float(), W32)
        lb_ = torch.nn.functional.linear(
            final_norm(zbase).float(), W32)
        d = li_ - lb_
        t = TTG[k]
        num = (d * t.unsqueeze(0)).sum(dim=1)
        den = d.norm(dim=1) \
            * float(TTn[k])
        c = (num / den.clamp_min(1e-12)) \
            .double().cpu().numpy()
    return c


# ==== E1 lens ladder (+ a1 anchor, + head caps)
L_E1 = list(range(NL))
if SMOKE:
    L_E1 = [30, 34, 35]
COS_LAD = np.zeros((NL, NP_))
PROP_ATTN_S = np.full((NL, NP_, NL), np.nan)
PROP_FULL_S = np.full((NL, NP_, NL), np.nan)
PROP_ATTN = np.full((NL, NL), np.nan)
PROP_FULL = np.full((NL, NL), np.nan)
C_RAW = np.full(NL, np.nan)
C_FULL_DIAG = np.full(NL, np.nan)
HI = {34: np.zeros((NP_, NL, NQ * HDIM)),
      35: np.zeros((NP_, NL, NQ * HDIM))}
b4_diff = 0.0
for l in L_E1:
    for k in range(NP_):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[l, ridx, :] = VB[pref_i][
            l, off + ridx, :]
        lg, _, _, _, zX, zA, zM, zP, zH = \
            forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
        dlg = lg - LG[base_i]
        COS_LAD[l, k] = cosv(dlg, TT[k])
        b4_diff = max(b4_diff, float(np.max(
            np.abs(zX[l].double().cpu().numpy()
                   - BX[base_i, l]))))
        zinj = torch.cat(
            [zX, zX + zA, zP], dim=0)
        zbase = base_z_gpu(base_i, nb)
        c = lens_cos(zinj, zbase, k)
        PROP_ATTN_S[l, k, :] = c[NL:2 * NL]
        PROP_FULL_S[l, k, :] = c[2 * NL:3 * NL]
        if l in HI:
            HI[l][k] = zH.double() \
                .cpu().numpy()
    PROP_ATTN[l, :] = np.nanmedian(
        PROP_ATTN_S[l], axis=0)
    PROP_FULL[l, :] = np.nanmedian(
        PROP_FULL_S[l], axis=0)
    C_RAW[l] = PROP_ATTN[l, l]
    C_FULL_DIAG[l] = PROP_FULL[l, l]
    log('E1 l=%2d med_c=%.4f raw=%.4f '
        'full_diag=%.4f'
        % (l, float(np.median(COS_LAD[l])),
           float(C_RAW[l]),
           float(C_FULL_DIAG[l])))
med_c = np.median(COS_LAD, axis=1)
b4_ok = bool(b4_diff == 0.0)
log('b4 delta-x at injection layer diff=%.3e '
    'ok=%s' % (b4_diff, b4_ok))

# a1: ladder bit anchor vs 3065 npz
a1_diff = None
a1_ok = False
if not SMOKE:
    npz65 = os.path.join(
        ROOT, 'tests', 'glm5', 'result',
        'rdc_query_construction_20260913',
        'phase3065',
        'omega_p62_v_sign_orchestration',
        'omega_p62_v_sign_orchestration.npz')
    z = np.load(npz65)
    a1_diff = float(np.max(np.abs(
        COS_LAD - z['qwen3_4b_COS_LAD'])))
    a1_ok = bool(a1_diff == 0.0)
    log('a1 ladder vs 3065 npz: max|d|=%.3e '
        'ok=%s' % (a1_diff, a1_ok))
else:
    log('a1 skipped (smoke)')

# a2b: lens calibration at the last layer
a2b_diff = abs(float(PROP_FULL[NL - 1, NL - 1])
               - float(med_c[NL - 1]))
a2b_ok = bool(a2b_diff <= 0.01)
log('a2b lens calibration |PROP_FULL[35,35] - '
    'med_c(35)|=%.4f ok=%s'
    % (a2b_diff, a2b_ok))

# ==== E2 causal silencing ====
SIL_ONE = np.full((len(SIL_LAYERS), NL), np.nan)
SIL_CUM = np.zeros(len(SIL_LAYERS))
b5_mag = None
share35 = None
c_sil35 = None
for si_, l in enumerate(SIL_LAYERS):
    lps = list(range(l + 1, NL))
    for lp in lps:
        cs = []
        for k in range(NP_):
            b, base_i, pref_i, off = \
                pair_idx(k)
            nb = int(LENS[base_i])
            repI = VB[base_i, :, :nb, :].copy()
            repI[l, ridx, :] = VB[pref_i][
                l, off + ridx, :]
            repI[lp, ridx, :] = 0.0
            repB = VB[base_i, :, :nb, :].copy()
            repB[lp, ridx, :] = 0.0
            lgI, _, _, _, _, _, _, _, _ = \
                forward_gen(
                    assembled[base_i]['ids'],
                    repl=repI)
            lgB, _, _, _, _, _, _, _, _ = \
                forward_gen(
                    assembled[base_i]['ids'],
                    repl=repB)
            cs.append(cosv(lgI - lgB, TT[k]))
            if l == 34 and lp == 35 \
                    and b5_mag is None:
                b5_mag = float(np.linalg.norm(
                    lgB - LG[base_i]))
        SIL_ONE[si_, lp] = float(np.median(cs))
    cs = []
    for k in range(NP_):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repI = VB[base_i, :, :nb, :].copy()
        repI[l, ridx, :] = VB[pref_i][
            l, off + ridx, :]
        repI[l + 1:, ridx, :] = 0.0
        repB = VB[base_i, :, :nb, :].copy()
        repB[l + 1:, ridx, :] = 0.0
        lgI, _, _, _, _, _, _, _, _ = \
            forward_gen(
                assembled[base_i]['ids'],
                repl=repI)
        lgB, _, _, _, _, _, _, _, _ = \
            forward_gen(
                assembled[base_i]['ids'],
                repl=repB)
        cs.append(cosv(lgI - lgB, TT[k]))
        if l == 34 and b5_mag is None:
            b5_mag = float(np.linalg.norm(
                lgB - LG[base_i]))
    SIL_CUM[si_] = float(np.median(cs))
    log('E2 l=%d: one-at-a-time %s cum=%.4f'
        % (l, np.round(
            SIL_ONE[si_, l + 1:], 4).tolist(),
           SIL_CUM[si_]))
c_sil35 = float(SIL_ONE[
    SIL_LAYERS.index(34), NL - 1])
log('b5 V0@35-on-base ablation magnitude '
    '||dlg||=%.3e (recorded)' % (b5_mag or 0.0))

# ==== E3 head map (descriptive) ====
HEAD_SHARE = {}
HEAD_COS = {}
for l in (34, 35):
    if l not in L_E1:
        continue
    lp = NL - 1
    contrib = np.zeros((NQ,))
    ch_cos = np.zeros((NQ,))
    for k in range(NP_):
        b, base_i, pref_i, off = pair_idx(k)
        dH = torch.tensor(
            HI[l][k] - BH[base_i],
            device='cuda').to(torch.bfloat16)
        with torch.no_grad():
            dv = dH[lp].reshape(NQ, HDIM)
            wo = layers[lp].self_attn \
                .o_proj.weight
            total = (dv.reshape(-1)
                     @ wo.t()).float()
            tn = max(float(total.norm()),
                     1e-12)
            for h in range(NQ):
                sl = slice(h * HDIM,
                           (h + 1) * HDIM)
                c_h = (dv[h]
                       @ wo[:, sl].t()).float()
                cn = max(float(c_h.norm()),
                         1e-12)
                contrib[h] += cn / NP_
                ch_cos[h] += float(
                    (c_h * total).sum()) \
                    / (cn * tn) / NP_
    HEAD_SHARE[l] = contrib \
        / max(contrib.sum(), 1e-12)
    HEAD_COS[l] = ch_cos
    log('E3 heads l=%d: top1=h%d share=%.3f '
        'top5=%s'
        % (l, int(np.argmax(HEAD_SHARE[l])),
           float(HEAD_SHARE[l].max()),
           np.argsort(HEAD_SHARE[l])[::-1][:5]
           .tolist()))
s34 = HEAD_SHARE.get(34)
s35 = HEAD_SHARE.get(35)
m34 = float(s34.max()) if s34 is not None \
    else -1.0
m35 = float(s35.max()) if s35 is not None \
    else -1.0
HEAD_TOP1 = max(m34, m35)
HEAD_ARG = 34 if m34 >= m35 else 35
HEAD_ARGH = int(np.argmax(
    HEAD_SHARE[HEAD_ARG]))

# ==== E4 norm channel ====
C_UNNORM = {}
for l in (34, 35):
    if l not in L_E1:
        continue
    cs = []
    for k in range(NP_):
        b, base_i, pref_i, off = pair_idx(k)
        nb = int(LENS[base_i])
        repV = VB[base_i, :, :nb, :].copy()
        repV[l, ridx, :] = VB[pref_i][
            l, off + ridx, :]
        lg, _, _, _, zX, zA, zM, zP, zH = \
            forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
        hb = torch.tensor(
            PB[base_i][NL - 1, nb - 1, :],
            device='cuda').to(torch.bfloat16)
        d2 = (zP[NL - 1] - hb).float() \
            .unsqueeze(0)
        with torch.no_grad():
            du = torch.nn.functional.linear(
                d2, W32)[0]
            t = TTG[k]
            num = float((du * t).sum())
            den = float(du.norm()) \
                * float(TTn[k])
            cs.append(num / max(den, 1e-12))
    C_UNNORM[l] = float(np.median(cs))
    log('E4 l=%d c_unnorm=%.4f vs lens '
        'full=%.4f'
        % (l, C_UNNORM[l],
           float(PROP_FULL[l, NL - 1])))

# ==== verdict ====
setup_ok = bool(b0_ok and b1_ok and b3_ok
                and b4_ok and a2b_ok)
band_use = [l for l in BAND
            if np.isfinite(C_RAW[l])]
raw_pos_med = float(np.median(
    [C_RAW[l] for l in band_use]))
den35 = float(med_c[34]) - raw_pos_med
if share35 is None and den35 != 0:
    share35 = (float(med_c[34]) - c_sil35) \
        / den35
head_tag = '_head%d' % HEAD_ARGH \
    if HEAD_TOP1 >= 0.5 else '_heads_dist'
if not setup_ok:
    verdict = 'setup_failed_flip_anatomy'
elif (not SMOKE) and (not a1_ok):
    verdict = 'anchor_mismatch_3065_ladder'
elif raw_pos_med >= 0:
    sub = 'mlp_shaped' \
        if abs(float(C_RAW[NL - 1])
               - float(C_FULL_DIAG[NL - 1])) \
        >= 0.1 else 'raw_pass'
    verdict = ('sign_flip_rawband_lastlayer_'
               + sub) + head_tag
elif abs(den35) >= 0.05:
    if share35 >= 0.5:
        verdict = ('sign_flip_downstream_'
                   'L35readout_localized'
                   + head_tag)
    else:
        verdict = ('sign_flip_downstream_'
                   'distributed' + head_tag)
else:
    verdict = 'flip_mixed_recorded'
log('raw_pos_med=%.4f med_c(34)=%.4f '
    'c_sil35=%.4f share35=%.4f top1=%.3f'
    % (raw_pos_med, float(med_c[34]),
       c_sil35, share35, HEAD_TOP1))
log('VERDICT: %s (setup_ok=%s a1_ok=%s)'
    % (verdict, setup_ok, a1_ok))

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
    'A2B_DIFF': np.float64(a2b_diff),
    'B0_DIFF': np.float64(b0_diff),
    'B1_DIFF': np.float64(b1_diff),
    'B4_DIFF': np.float64(b4_diff),
    'B5_MAG': np.float64(b5_mag or 0.0),
    'SETUP_OK': np.bool_(setup_ok),
    'COS_LAD': COS_LAD,
    'MED_C': med_c,
    'PROP_ATTN': PROP_ATTN,
    'PROP_FULL': PROP_FULL,
    'PROP_ATTN_S': PROP_ATTN_S,
    'PROP_FULL_S': PROP_FULL_S,
    'C_RAW': C_RAW,
    'C_FULL_DIAG': C_FULL_DIAG,
    'SIL_ONE': SIL_ONE,
    'SIL_CUM': SIL_CUM,
    'SHARE35': np.float64(share35),
    'RAW_POS_MED': np.float64(raw_pos_med),
    'C_UNNORM_34': np.float64(
        C_UNNORM.get(34, np.nan)),
    'C_UNNORM_35': np.float64(
        C_UNNORM.get(35, np.nan)),
    'HEAD_SHARE_34': np.array(
        HEAD_SHARE.get(34,
                       np.full(NQ, np.nan))),
    'HEAD_SHARE_35': np.array(
        HEAD_SHARE.get(35,
                       np.full(NQ, np.nan))),
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
    'run': 'run1 authoritative (qwen3-4b bf16 '
           'single model)' if not SMOKE
    else 'smoke',
    'prereg': PREREG,
    'stats': {
        'med_c_30_35': [f64(med_c[l])
                        for l in range(30, 36)],
        'c_raw_30_35': [f64(C_RAW[l])
                        for l in range(30, 36)],
        'c_full_diag_30_35': [
            f64(C_FULL_DIAG[l])
            for l in range(30, 36)],
        'raw_pos_med': f64(raw_pos_med),
        'med_c_34': f64(med_c[34]),
        'c_sil35': f64(c_sil35),
        'share35': f64(share35),
        'sil_cum': {
            str(l): float(SIL_CUM[i])
            for i, l in
            enumerate(SIL_LAYERS)},
        'c_unnorm_35': f64(C_UNNORM.get(35,
                                        np.nan)),
        'head_top1': f64(HEAD_TOP1),
        'head_arg_layer': HEAD_ARG,
        'head_arg_h': HEAD_ARGH,
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
        'b5_mag': b5_mag,
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

del model, layers, tok, Wemb, final_norm
gc.collect()
torch.cuda.empty_cache()
