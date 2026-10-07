# -*- coding: utf-8 -*-
# Phase 3054 - Omega-P51: norm projection
# mechanism. 3053 T3 showed the assembly chain:
# attention-stage diff cos 0.4305 -> full-layer
# diff cos -0.1373 (L35 MLP counter-shaping) ->
# post-final-norm cos 0.6805. Question: HOW does
# the final RMSNorm lift -0.14 to 0.68?
# Hypothesis (preregistered): the layer diff d
# decomposes into a tangential component (perp to
# x_b = post_b) with high target alignment and a
# DOMINANT radial component (amplitude modulation,
# low/negative alignment); RMSNorm kills the
# radial part to first order (Jacobian = gamma *
# P_perp / rms), keeping only the tangential
# projection -> alignment jumps. The -0.1373 cos
# is radial dilution, not a tangential reversal.
# T2 decomposition (per pair, canonical h7 joint
# diag arm = 3053 T3 arm, a163 bit chain anchor):
# d = post_i - post_b; d_tan = d - (d.xb)xb;
# cos_WU(d) [= STAGE col2, bit anchor vs z53],
# cos_WU(d_tan), cos_WU(d_rad); radial fraction
# |d.xb|/||d||.
# T3 first-order prediction (MAIN): pred = gamma *
# d_tan / rms_b; actual = postn_i - postn_b;
# cos_lg = cos(W_U pred, logit diff r); rel_err =
# ||W_U pred - r|| / ||r||; cos_postn = cos(pred,
# actual). verdict: med cos_lg >= 0.99 ->
# norm_tangential_qwen; >= 0.95 ->
# norm_approx_qwen; else norm_higher_order_qwen.
# T3b offline counterfactuals (no forward):
# actual_tan = norm(x_b + d_tan) - norm(x_b);
# actual_rad = norm(x_b + d_rad) - norm(x_b);
# cos(W_U actual_tan, r) and cos(W_U actual_rad,
# t_k) - direct demonstration that the radial
# part passes through the norm direction-neutral
# (dilution) while the tangential part carries.
# T4 h0 exact K-only arm (3053 COS_KCTRL_h0 med
# -0.2492): same decomposition - is the negative
# alignment tangential-adversarial (survives the
# projection, cos_WU(d_tan) still negative) or
# radial-diluted (washed out)?
# anchors: a162 sampled re-capture (4 prompts)
# bit 0.0 vs z48 + TT 0.0; a163 COS_H3 vs z51
# COS_H[7] bit 0.0; a164 postn consistency
# cos(W_U actual, r) vs COS_H3 < 1e-3; a165 sham
# bit identity; a166 stage-pre identity 0.0
# (h7 arm); a167 per-pair cos_WU(d) bit 0.0 vs
# z53 STAGE col2; a168 per-pair cos_WU(actual)
# bit 0.0 vs z53 STAGE col3; a116c source seals
# 3044-3053.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3054
NAME = 'omega_p51_norm_projection_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 8
HDIM = 128
SEED_MAIN = 3010
FRONT = 4
L_TGT = 35
HEAD_H = 7
HEAD_H0 = 0
T_TANG = 0.99
T_APPROX = 0.95
BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')

PREREG = {
    'mode': 'fp32 MODEL (torch.float32, eager, '
            'seed 3010); capture bank loaded from '
            'the phase3048 npz (KPpost/VP/LG/LENS/'
            'TT); chain anchors vs the phase3051 '
            'npz (COS_H[7]) and the phase3053 npz '
            '(STAGE cols 2/3 per-pair bit); '
            'prompts reassembled locally with the '
            '3048 tail-alignment rule; statistics '
            'on the 24 old pairs',
    'question': '3054 A main line: the final '
                'RMSNorm lifts the full-layer '
                'diff alignment from -0.1373 to '
                '0.6805 (3053 T3). Is this radial '
                'dilution (d decomposes into a '
                'high-alignment tangential part '
                'plus a dominant low/negative-'
                'alignment radial amplitude part; '
                'RMSNorm Jacobian = gamma*P_perp/'
                'rms keeps only tangential) - and '
                'is the h0 exact-field negative '
                'alignment tangential-adversarial '
                'or radial-diluted?',
    'T2_decomposition': 'per pair k (24 old pairs): '
                        'canonical h7 joint diag '
                        'replacement (K+V head-7 '
                        'block at L35 x FRONT, '
                        '= 3053 T3 arm); d = post_i '
                        '- post_b (last position, '
                        'L35 layer output); xb = '
                        'post_b; d_tan = d - (d.xb/'
                        '|xb|^2) xb; d_rad = d - '
                        'd_tan; per pair: cos_WU(d) '
                        '(bit anchor vs z53 STAGE '
                        'col2), cos_WU(d_tan), '
                        'cos_WU(d_rad), frac_rad = '
                        '|d.xb|/||d||, rms_b, '
                        '|d_tan|/|d|',
    'T3_first_order': 'MAIN: pred = gamma * d_tan / '
                      'rms_b (RMSNorm first-order '
                      'Jacobian, gamma = final-norm '
                      'weight, eps = '
                      'config.rms_norm_eps); actual '
                      '= postn_i - postn_b; per pair '
                      'cos_lg = cos(W_U pred, '
                      'logit diff r), rel_err = '
                      '||W_U pred - r||/||r||, '
                      'cos_postn = cos(pred, '
                      'actual) (bit anchor vs z53 '
                      'STAGE col3 via cos(W_U '
                      'actual, r)); verdict: med '
                      'cos_lg >= 0.99 -> '
                      'norm_tangential_qwen; >= '
                      '0.95 -> norm_approx_qwen; '
                      'else -> '
                      'norm_higher_order_qwen; '
                      'single branch',
    'T3b_counterfactual': 'offline (no forward): '
                          'actual_tan = norm(x_b + '
                          'd_tan) - norm(x_b); '
                          'actual_rad = norm(x_b + '
                          'd_rad) - norm(x_b); '
                          'cos(W_U actual_tan, r) '
                          'and cos(W_U actual_rad, '
                          't_k) per pair - direct '
                          'demonstration of radial '
                          'dilution vs tangential '
                          'carrying; descriptive',
    'T4_h0_arm': 'h0 exact K-only diag arm (K head-0 '
                 'block at L35 x FRONT, V self; = '
                 '3053 COS_KCTRL_h0 arm): same T2/T3 '
                 'decomposition; question: is med '
                 'cos_WU(d_tan_h0) still negative '
                 '(tangential-adversarial, the '
                 'adversarial component survives the '
                 'norm) or washed toward 0 (radial '
                 'dilution)? Also quantifies why '
                 'random K (+0.49) beats the exact '
                 'h0 field (-0.25). Descriptive',
    'anchors': 'a162 sampled re-capture (4 prompts '
               '0/9/17/31) KPpre/KPpost/VP/LG bit '
               '0.0 vs z48 + TT diff 0.0; a163 '
               'COS_H3 diff 0.0 vs z51 COS_H[7]; '
               'a164 postn consistency max |cos(W_U '
               'actual, r) - COS_H3| < 1e-3; a165 '
               'sham self-replacement bit identity; '
               'a166 stage-pre identity diff 0.0 '
               '(h7 arm); a167 max |cos_WU(d) - '
               'z53 STAGE[:,2]| == 0.0; a168 max '
               '|cos_WU(actual) - z53 STAGE[:,3]| '
               '== 0.0; a116c source seals '
               '3044-3053',
    'statistics_discipline': 'all decomposition '
                             'quantities computed per '
                             'pair on the same 24 '
                             'pairs, same replacement '
                             'machinery, same base '
                             'forward (self '
                             'replacement = bit '
                             'identity, a157-style); '
                             'no null needed - the '
                             'main claim is an '
                             'approximation-quality '
                             'criterion on the exact '
                             'intervention; verdict '
                             'in one branch',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


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
assert int(model.config.num_key_value_heads) == KV_HEAD
assert int(model.config.num_attention_heads) == 32
rot = model.model.rotary_emb
INV_FREQ = rot.inv_freq.detach().cpu().numpy() \
    .astype(np.float32)
assert INV_FREQ.shape == (64,), INV_FREQ.shape
assert float(rot.attention_scaling) == 1.0
NVOC = int(model.config.vocab_size)
EPS_NORM = float(model.config.rms_norm_eps)
log('model loaded fp32 (vocab=%d rms_eps=%g)'
    % (NVOC, EPS_NORM))


def rot_apply(x, delta):
    """Rotate (...,8,128) by delta*inv_freq
    (NeoX half-pairing, fp32)."""
    ang = INV_FREQ * np.float32(delta)
    emb = np.concatenate([ang, ang]).astype(
        np.float32)
    c = np.cos(emb)[None, None, :]
    s = np.sin(emb)[None, None, :]
    x1 = x[..., :64]
    x2 = x[..., 64:]
    rh = np.concatenate([-x2, x1], axis=-1)
    return x * c + rh * s


stateKn = {li: {'repl': None, 'mask': None}
           for li in range(NL)}
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
capKn = {li: {'rec': False, 'orig': None,
              'mod': None} for li in range(NL)}
capV = {li: {'rec': False, 'orig': None,
             'mod': None} for li in range(NL)}
capKpre = {li: {'rec': False, 'orig': None}
           for li in range(NL)}
cap_stage = {'pre': {'rec': False, 'v': None},
             'op': {'rec': False, 'v': None},
             'post': {'rec': False, 'v': None}}


def hook_norm(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_kpre(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        return out
    return h


def hook_stage(cp):
    """Capture the LAST-position vector of a
    bare-tensor output (batch dim stripped by
    indexing [0])."""
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] if isinstance(out, tuple) \
                else out
            cp['v'] = t[0][-1].detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_norm \
        .register_forward_hook(hook_norm(
            stateKn[li], capKn[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
    layers[li].self_attn.k_proj \
        .register_forward_hook(hook_kpre(
            capKpre[li]))
layers[34].register_forward_hook(
    hook_stage(cap_stage['pre']))
layers[35].self_attn.o_proj.register_forward_hook(
    hook_stage(cap_stage['op']))
layers[35].register_forward_hook(
    hook_stage(cap_stage['post']))


def reset_all():
    for li in range(NL):
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKn[li]['rec'] = False
        capV[li]['rec'] = False
        capKpre[li]['rec'] = False
    for kk in cap_stage:
        cap_stage[kk]['rec'] = False
        cap_stage[kk]['v'] = None


def forward_cap(ids):
    reset_all()
    for li in range(NL):
        capKn[li]['rec'] = True
        capV[li]['rec'] = True
        capKpre[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    n = len(ids)
    kp = np.stack([capKpre[li]['orig']
                   .cpu().numpy()
                   for li in range(NL)])
    kn = np.stack([capKn[li]['orig'].reshape(
        n, HDIM * 8).cpu().numpy()
        for li in range(NL)])
    vp = np.stack([capV[li]['orig'].cpu().numpy()
                   for li in range(NL)])
    reset_all()
    return lg, kp, kn, vp


def forward_stage(ids, replK, maskK, replV, maskV):
    """Joint replacement with stage capture;
    returns logits + last-position stage vectors
    (pre, op, post) and the post-final-norm post
    vector."""
    reset_all()
    rk = torch.tensor(np.ascontiguousarray(
        replK.reshape(NL, -1, 8, HDIM)),
        dtype=torch.float32, device='cuda')
    mk = torch.tensor(np.asarray(maskK),
                      device='cuda')
    rv = torch.tensor(np.ascontiguousarray(
        replV),
        dtype=torch.float32, device='cuda')
    mv = torch.tensor(np.asarray(maskV),
                      device='cuda')
    for li in range(NL):
        stateKn[li]['repl'] = rk[li]
        stateKn[li]['mask'] = mk
        stateV[li]['repl'] = rv[li]
        stateV[li]['mask'] = mv
    for kk in cap_stage:
        cap_stage[kk]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    pre = cap_stage['pre']['v'].double() \
        .cpu().numpy()
    op = cap_stage['op']['v'].double() \
        .cpu().numpy()
    post_t = cap_stage['post']['v']
    post = post_t.double().cpu().numpy()
    with torch.no_grad():
        postn_t = model.model.norm(
            post_t.unsqueeze(0))
    postn = postn_t[0].double().cpu().numpy()
    reset_all()
    return lg, pre, op, post, postn


# ---------- chain: load capture banks ----------
z48 = np.load(os.path.join(
    BASE, 'phase3048',
    'omega_p45_kvpos_full_replay_qwen',
    'omega_p45_kvpos_full_replay_qwen.npz'),
    allow_pickle=True)
KPpre = z48['KPpre']
KPpost = z48['KPpost']
VP = z48['VP']
LG = z48['LG']
LENS = z48['LENS']
TT = z48['TT']
log('z48 bank loaded: KPpre%s KPpost%s VP%s LG%s'
    % (KPpre.shape, KPpost.shape, VP.shape,
       LG.shape))
z51 = np.load(os.path.join(
    BASE, 'phase3051',
    'omega_p48_l35_anatomy_qwen',
    'omega_p48_l35_anatomy_qwen.npz'),
    allow_pickle=True)
COS_H51 = z51['COS_H']
z53 = np.load(os.path.join(
    BASE, 'phase3053',
    'omega_p50_gate_source_qwen',
    'omega_p50_gate_source_qwen.npz'),
    allow_pickle=True)
STAGE53 = z53['STAGE']
assert STAGE53.shape == (24, 4), STAGE53.shape
log('z51/z53 anchors loaded: COS_H%s STAGE53%s'
    % (COS_H51.shape, STAGE53.shape))

seal_detail = []
for ph, nm in (
        (3044, 'omega_p41_field_axis_injection_'
               'qwen'),
        (3045, 'omega_p42_l20_axis_anatomy_qwen'),
        (3046, 'omega_p43_kfield_injection_qwen'),
        (3047, 'omega_p44_kv_joint_replay_qwen'),
        (3048, 'omega_p45_kvpos_full_replay_qwen'),
        (3049, 'omega_p46_kvload_localization_'
               'qwen'),
        (3050, 'omega_p47_kvdeep_dissection_'
               'qwen'),
        (3051, 'omega_p48_l35_anatomy_qwen'),
        (3052, 'omega_p49_kvhead_identity_qwen'),
        (3053, 'omega_p50_gate_source_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a116_ok = bool(seal_detail) and all(seal_detail)
log('a116c source seals ok=%s' % a116_ok)

# ---------- assemble prompts (3048 verbatim) ----------
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
for i in range(n_pr):
    ci = assembled[i]['cond']
    if ci == 0:
        assembled[i]['off'] = 0
    else:
        bid = assembled[idx_of[(0,
            assembled[i]['body'])]]['ids']
        pid = assembled[i]['ids']
        off = len(pid) - len(bid)
        assert off > 0, i
        assert list(pid[off + 1:]) \
            == list(bid[1:]), i
        w0b = tok.decode([bid[0]]).strip()
        w0p = tok.decode([pid[off]]).strip()
        assert w0b == w0p, (i, w0b, w0p)
        assembled[i]['off'] = off
for i in range(n_pr):
    assert LENS[i] == len(assembled[i]['ids']), i
log('assembled %d prompts (lens consistent '
    'with z48)' % n_pr)

# a162: sampled re-capture, bit-exact vs z48
a162_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a162_diff = max(a162_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a162_diff = max(a162_diff, float(np.max(
        np.abs(KPpre[si, :, :n2, :] - kp2))))
    a162_diff = max(a162_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a162_diff = max(a162_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a162_diff = float(a162_diff)

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)
NP_ = 24

t_targets = {}
for b in range(len(BODIES)):
    for c in (1, 2, 3):
        t_targets[(b, c)] = LG[idx_of[(c, b)]] \
            - LG[idx_of[(0, b)]]
TTl = np.stack([t_targets[(int(bidx[k]),
                          int(cidx[k]))]
                for k in range(NP_)])
a_tt = float(np.max(np.abs(TTl - TT)))
a162_ok = bool(a162_diff == 0.0 and a_tt == 0.0)
log('a162 re-capture diff=%.3e TT diff=%.3e ok=%s'
    % (a162_diff, a_tt, a162_ok))

# a165: sham self-replacement (bit identity)
b0 = int(bidx[0])
base0 = idx_of[(0, b0)]
ids0 = assembled[base0]['ids']
n0 = len(ids0)
selfK = KPpost[base0, :, :n0, :].copy()
selfV = VP[base0, :, :n0, :].copy()
m_all = np.ones(n0, dtype=bool)
lg_s, _, _, _, _ = forward_stage(
    ids0, selfK, m_all, selfV, m_all)
a165_diff = float(np.max(np.abs(lg_s
                                - LG[base0])))
a165_ok = bool(a165_diff == 0.0)
log('a165 sham self-replacement diff=%.3e ok=%s'
    % (a165_diff, a165_ok))


def front_rows(nb):
    return list(range(min(FRONT, nb)))


def l35_fields(k):
    """Per pair: base self fields (NL, nb, 1024)
    plus the L35 pref fields for the FRONT rows
    (K rotated by off, V raw)."""
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    nb = int(LENS[base_i])
    off = assembled[pref_i]['off']
    repK = KPpost[base_i, :, :nb, :].copy()
    repV = VP[base_i, :, :nb, :].copy()
    rows = front_rows(nb)
    ridx = np.asarray(rows)
    src = KPpost[pref_i][:, off + ridx, :] \
        .reshape(NL, len(rows), 8, HDIM)
    rotK = rot_apply(src, off)  # (NL,nfr,8,128)
    srcV = VP[pref_i][:, off + ridx, :]
    return repK, repV, rotK, srcV, rows, base_i


def cos_frac(lg_base, r, t):
    nr = float(np.linalg.norm(r))
    nt = float(np.linalg.norm(t))
    cs = float(r @ t) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    fr = nr / nt if nt > 1e-12 else float('nan')
    return cs, fr


h = HEAD_H
sl = slice(h * HDIM, (h + 1) * HDIM)
NFR_ALL = None
for k in range(NP_):
    nb = int(LENS[idx_of[(0, int(bidx[k]))]])
    if NFR_ALL is None:
        NFR_ALL = len(front_rows(nb))
    assert len(front_rows(nb)) == NFR_ALL, k
assert NFR_ALL == 4

WU = model.lm_head.weight.detach()  # (V, 2560)
wud = WU.double().cpu().numpy()
GAMMA = model.model.norm.weight.detach() \
    .double().cpu().numpy()
assert GAMMA.shape == (2560,)

a166_pre_id_max = 0.0
COS_H3 = np.zeros(NP_)
STAGE_MLP = np.zeros(NP_)
STAGE_MLPN = np.zeros(NP_)
# T2/T3 per-pair decomposition arrays
COS_D = np.zeros(NP_)
COS_DTAN = np.zeros(NP_)
COS_DRAD = np.zeros(NP_)
FRAC_RAD = np.zeros(NP_)
RMS_B = np.zeros(NP_)
TAN_FRAC = np.zeros(NP_)
COS_LG = np.zeros(NP_)
REL_ERR = np.zeros(NP_)
COS_POSTN = np.zeros(NP_)
# T3b counterfactuals
COS_TAN_ONLY = np.zeros(NP_)
COS_RAD_ONLY = np.zeros(NP_)

log('=== T2/T3 h7 joint arm: capture + '
    'decomposition ===')
for k in range(NP_):
    b = int(bidx[k])
    repK_k, repV_k, rotK_k, srcV_k, rows_k, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    # base (self replacement) stages
    _, pre_b, op_b, post_b, postn_b = forward_stage(
        ids_b, repK_k, mask, repV_k, mask)
    # intervened: h7 joint (K+V head block)
    rK = repK_k.copy()
    rV = repV_k.copy()
    rK[L_TGT, rows_k, sl] = rotK_k[L_TGT, :, h, :]
    svk = srcV_k[L_TGT].reshape(
        len(rows_k), KV_HEAD, HDIM)
    rV[L_TGT, rows_k, sl] = svk[:, h, :]
    lg_i, pre_i, op_i, post_i, postn_i \
        = forward_stage(ids_b, rK, mask, rV, mask)
    r = lg_i - LG[base_i]
    t_k = t_targets[(b, int(cidx[k]))]
    COS_H3[k] = cos_frac(LG[base_i], r, t_k)[0]
    dp = pre_i - pre_b
    a166_pre_id_max = max(a166_pre_id_max,
                          float(np.max(np.abs(dp))))
    d = post_i - post_b
    # STAGE-consistent quantities (bit anchors)
    STAGE_MLP[k] = cos_frac(LG[base_i],
                            wud @ d, t_k)[0]
    STAGE_MLPN[k] = cos_frac(LG[base_i],
                             wud @ (postn_i
                                    - postn_b),
                             t_k)[0]
    # T2 decomposition
    xb = post_b
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    ddot = float(d @ xhat)
    d_tan = d - ddot * xhat
    d_rad = ddot * xhat
    rms_b = float(np.sqrt(nx2 / 2560.0 + EPS_NORM))
    COS_D[k] = STAGE_MLP[k]
    nd = float(np.linalg.norm(d))
    COS_DTAN[k] = cos_frac(LG[base_i],
                           wud @ d_tan, t_k)[0]
    COS_DRAD[k] = cos_frac(LG[base_i],
                           wud @ d_rad, t_k)[0]
    FRAC_RAD[k] = abs(ddot) / nd if nd > 1e-12 \
        else 0.0
    TAN_FRAC[k] = float(np.linalg.norm(d_tan)) / nd \
        if nd > 1e-12 else 0.0
    RMS_B[k] = rms_b
    # T3 first-order prediction
    pred = GAMMA * d_tan / rms_b
    z_pred = wud @ pred
    nz = float(np.linalg.norm(z_pred))
    nr = float(np.linalg.norm(r))
    COS_LG[k] = float(z_pred @ r) / (nz * nr) \
        if nz > 1e-12 and nr > 1e-12 else 0.0
    REL_ERR[k] = float(np.linalg.norm(
        z_pred - r)) / nr if nr > 1e-12 else 0.0
    actual = postn_i - postn_b
    na = float(np.linalg.norm(actual))
    npd = float(np.linalg.norm(pred))
    COS_POSTN[k] = float(pred @ actual) \
        / (npd * na) \
        if npd > 1e-12 and na > 1e-12 else 0.0
    # T3b counterfactuals (offline exact norm)
    with torch.no_grad():
        xbt = torch.tensor(xb, dtype=torch.float32,
                           device='cuda').unsqueeze(0)
        t1 = model.model.norm(
            xbt + torch.tensor(d_tan,
                               dtype=torch.float32,
                               device='cuda'))
        r1 = model.model.norm(
            xbt + torch.tensor(d_rad,
                               dtype=torch.float32,
                               device='cuda'))
    at = t1[0].double().cpu().numpy() \
        - postn_b
    ar = r1[0].double().cpu().numpy() \
        - postn_b
    COS_TAN_ONLY[k] = cos_frac(LG[base_i],
                               wud @ at, t_k)[0]
    COS_RAD_ONLY[k] = cos_frac(LG[base_i],
                               wud @ ar, t_k)[0]
    if k % 8 == 0:
        log('  k=%02d cos_d=%.4f cos_dtan=%.4f '
            'cos_lg=%.5f rel=%.4f frac_rad=%.3f'
            % (k, COS_D[k], COS_DTAN[k],
               COS_LG[k], REL_ERR[k],
               FRAC_RAD[k]))

a166_ok = bool(a166_pre_id_max == 0.0)
log('a166 stage-pre identity max diff=%.3e ok=%s'
    % (a166_pre_id_max, a166_ok))
a163_diff = float(np.max(np.abs(
    COS_H3 - COS_H51[h])))
a163_ok = bool(a163_diff == 0.0)
log('a163 joint diag repro vs z51 COS_H[7]: '
    'diff=%.3e ok=%s' % (a163_diff, a163_ok))
a164_dev = float(np.max(np.abs(
    STAGE_MLPN - COS_H3)))
a164_ok = bool(a164_dev < 1e-3)
log('a164 postn consistency dev max=%.3e ok=%s'
    % (a164_dev, a164_ok))
a167_diff = float(np.max(np.abs(
    STAGE_MLP - STAGE53[:, 2])))
a167_ok = bool(a167_diff == 0.0)
a168_diff = float(np.max(np.abs(
    STAGE_MLPN - STAGE53[:, 3])))
a168_ok = bool(a168_diff == 0.0)
log('a167 stage-mlp bit vs z53: diff=%.3e ok=%s | '
    'a168 stage-mlpn bit vs z53: diff=%.3e ok=%s'
    % (a167_diff, a167_ok, a168_diff, a168_ok))

# T2/T3 summaries
med_cos_d = float(np.median(COS_D))
med_cos_dtan = float(np.median(COS_DTAN))
med_cos_drad = float(np.median(COS_DRAD))
med_frac_rad = float(np.median(FRAC_RAD))
med_tan_frac = float(np.median(TAN_FRAC))
med_rms = float(np.median(RMS_B))
med_cos_lg = float(np.median(COS_LG))
med_rel = float(np.median(REL_ERR))
med_cos_postn = float(np.median(COS_POSTN))
med_cos_tonly = float(np.median(COS_TAN_ONLY))
med_cos_ronly = float(np.median(COS_RAD_ONLY))
log('T2: med cos_d=%.4f cos_dtan=%.4f '
    'cos_drad=%.4f | frac_rad=%.4f tan_frac=%.4f '
    'rms_b=%.2f'
    % (med_cos_d, med_cos_dtan, med_cos_drad,
       med_frac_rad, med_tan_frac, med_rms))
log('T3: med cos_lg=%.5f rel_err=%.4f '
    'cos_postn=%.5f'
    % (med_cos_lg, med_rel, med_cos_postn))
log('T3b: med cos_tan_only=%.4f cos_rad_only=%.4f'
    % (med_cos_tonly, med_cos_ronly))

# ---------- T4 h0 exact K-only arm ----------
log('=== T4 h0 exact K-only decomposition ===')
slh = slice(HEAD_H0 * HDIM,
            (HEAD_H0 + 1) * HDIM)
COS_K0 = np.zeros(NP_)
H0_COS_D = np.zeros(NP_)
H0_COS_DTAN = np.zeros(NP_)
H0_COS_DTAN_ONLY = np.zeros(NP_)
H0_FRAC_RAD = np.zeros(NP_)
for k in range(NP_):
    b = int(bidx[k])
    repK_k, repV_k, rotK_k, _, rows_k, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    _, pre_b, op_b, post_b, postn_b = forward_stage(
        ids_b, repK_k, mask, repV_k, mask)
    rK = repK_k.copy()
    rK[L_TGT, rows_k, slh] \
        = rotK_k[L_TGT, :, HEAD_H0, :]
    lg_i, pre_i, op_i, post_i, postn_i \
        = forward_stage(ids_b, rK, mask,
                        repV_k, mask)
    r = lg_i - LG[base_i]
    t_k = t_targets[(b, int(cidx[k]))]
    COS_K0[k] = cos_frac(LG[base_i], r, t_k)[0]
    d = post_i - post_b
    H0_COS_D[k] = cos_frac(LG[base_i],
                           wud @ d, t_k)[0]
    xb = post_b
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    ddot = float(d @ xhat)
    d_tan = d - ddot * xhat
    d_rad = ddot * xhat
    H0_COS_DTAN[k] = cos_frac(LG[base_i],
                              wud @ d_tan, t_k)[0]
    H0_FRAC_RAD[k] = abs(ddot) / float(
        np.linalg.norm(d))
    with torch.no_grad():
        xbt = torch.tensor(xb, dtype=torch.float32,
                           device='cuda').unsqueeze(0)
        t1 = model.model.norm(
            xbt + torch.tensor(d_tan,
                               dtype=torch.float32,
                               device='cuda'))
    at = t1[0].double().cpu().numpy() \
        - postn_b
    H0_COS_DTAN_ONLY[k] = cos_frac(
        LG[base_i], wud @ at, t_k)[0]
med_cos_k0 = float(np.median(COS_K0))
med_h0_d = float(np.median(H0_COS_D))
med_h0_dtan = float(np.median(H0_COS_DTAN))
med_h0_dtonly = float(np.median(H0_COS_DTAN_ONLY))
med_h0_frad = float(np.median(H0_FRAC_RAD))
log('T4 h0: med cos_K0=%.4f cos_d=%.4f '
    'cos_dtan=%.4f cos_dtan_only=%.4f '
    'frac_rad=%.4f'
    % (med_cos_k0, med_h0_d, med_h0_dtan,
       med_h0_dtonly, med_h0_frad))
log('  (z53 ref: med COS_KCTRL_h0 = -0.2492)')

# ---------- verdict ----------
if med_cos_lg >= T_TANG:
    verdict = 'norm_tangential_qwen'
elif med_cos_lg >= T_APPROX:
    verdict = 'norm_approx_qwen'
else:
    verdict = 'norm_higher_order_qwen'
log('VERDICT: %s (med cos_lg=%.5f rel=%.4f)'
    % (verdict, med_cos_lg, med_rel))

anchor_core_ok = bool(a116_ok and a162_ok
                      and a163_ok and a164_ok
                      and a165_ok and a166_ok
                      and a167_ok and a168_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_H3=COS_H3,
         STAGE_MLP=STAGE_MLP,
         STAGE_MLPN=STAGE_MLPN,
         COS_D=COS_D, COS_DTAN=COS_DTAN,
         COS_DRAD=COS_DRAD,
         FRAC_RAD=FRAC_RAD, TAN_FRAC=TAN_FRAC,
         RMS_B=RMS_B,
         COS_LG=COS_LG, REL_ERR=REL_ERR,
         COS_POSTN=COS_POSTN,
         COS_TAN_ONLY=COS_TAN_ONLY,
         COS_RAD_ONLY=COS_RAD_ONLY,
         COS_K0=COS_K0,
         H0_COS_D=H0_COS_D,
         H0_COS_DTAN=H0_COS_DTAN,
         H0_COS_DTAN_ONLY=H0_COS_DTAN_ONLY,
         H0_FRAC_RAD=H0_FRAC_RAD,
         a162_diff=np.float64(a162_diff),
         a163_diff=np.float64(a163_diff),
         a164_dev=np.float64(a164_dev),
         a165_diff=np.float64(a165_diff),
         a166_pre_id_max=np.float64(
             a166_pre_id_max),
         a167_diff=np.float64(a167_diff),
         a168_diff=np.float64(a168_diff),
         a116_ok=np.bool_(a116_ok),
         a162_ok=np.bool_(a162_ok),
         a163_ok=np.bool_(a163_ok),
         a164_ok=np.bool_(a164_ok),
         a165_ok=np.bool_(a165_ok),
         a166_ok=np.bool_(a166_ok),
         a167_ok=np.bool_(a167_ok),
         a168_ok=np.bool_(a168_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_decomposition': {
        'med_cos_d': med_cos_d,
        'med_cos_dtan': med_cos_dtan,
        'med_cos_drad': med_cos_drad,
        'med_frac_rad': med_frac_rad,
        'med_tan_frac': med_tan_frac,
        'med_rms_b': med_rms},
    'T3_first_order': {
        'med_cos_lg': med_cos_lg,
        'med_rel_err': med_rel,
        'med_cos_postn': med_cos_postn},
    'T3b_counterfactual': {
        'med_cos_tan_only': med_cos_tonly,
        'med_cos_rad_only': med_cos_ronly},
    'T4_h0_arm': {
        'med_cos_K0': med_cos_k0,
        'med_cos_d': med_h0_d,
        'med_cos_dtan': med_h0_dtan,
        'med_cos_dtan_only': med_h0_dtonly,
        'med_frac_rad': med_h0_frad,
        'z53_ref_med_COS_KCTRL_h0': -0.2492},
    'anchors': {'a116_seals_ok': a116_ok,
                'a162_recapture_diff': a162_diff,
                'a163_jdiag_diff': a163_diff,
                'a164_postn_dev': a164_dev,
                'a165_sham_diff': a165_diff,
                'a166_pre_id_max':
                    a166_pre_id_max,
                'a167_stage_mlp_diff':
                    a167_diff,
                'a168_stage_mlpn_diff':
                    a168_diff,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run1 authoritative (fp32; capture '
                 'bank from the phase3048 npz; '
                 'chain anchors vs the phase3051 '
                 'and phase3053 npz)',
          'prereg': PREREG, 'stats': stats,
          'verdict': verdict}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)

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
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'], elapsed))
log('sealed')
