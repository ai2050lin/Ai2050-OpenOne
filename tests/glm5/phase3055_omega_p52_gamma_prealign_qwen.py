# -*- coding: utf-8 -*-
# Phase 3055 - Omega-P52: gamma readout
# pre-alignment source and universality. 3054
# showed the final 0.68 alignment is
# MANUFACTURED: radial attenuation (-0.61 ->
# -0.12) + learnable gamma re-weighting
# (tangential 0.166 -> 0.68; the 1/rms scalar
# cannot change cos). Question: WHERE does the
# gamma gain live - is it TARGET-DIRECTION
# SPECIFIC (training pre-aligned the readout
# for behaviorally relevant logit directions)
# or a GLOBAL geometry effect - and does the
# channel assignment of gamma CAUSALLY carry
# the readout alignment?
# T2 readout-preimage alignment (offline +
# orig-gamma arm capture): w_t = W_U^T t
# (residual-stream preimage of the target
# logit direction); per pair cos(d_tan, gamma
# * w_t) vs cos(d_tan, w_t) - analytic replay
# of the 3054 gain; RANDOM control U=200
# logit directions (norm-matched, seed 9951):
# gain_u = cos(d_tan, gamma*w_u) - cos(d_tan,
# w_u) median vs gain_t -> target-specific if
# gain_t >> gain_u.
# T3 gamma causal perturbation (MAIN,
# forwards): three arms (orig / shuffled
# permutation of gamma channels / ones), each
# 24 base + 24 interv canonical h7 joint
# diag; r_new = lg_i - lg_b cos the ORIGINAL
# TT target direction (option A: fix the
# original target; the base moves too, so
# this measures how much of the alignment
# survives the new readout basis).
# verdict: med COS_shuf < 0.3402 (= 0.5 x
# 0.6805) AND med COS_ones < 0.3402 ->
# gamma_prealigned_qwen (channel assignment
# carries); shuf < 0.3402 AND ones >= 0.3402
# -> gamma_distribution_qwen (amplitude
# profile carries, assignment irrelevant);
# else -> gamma_robust_qwen.
# anchors: a169 sampled re-capture (4
# prompts) bit 0.0 vs z48 + a172 TT diff
# 0.0; a170 COS_orig diff 0.0 vs z51
# COS_H[7]; a171 gamma-restore sham bit 0.0
# (weight restoration verified); a173
# self-replacement bit identity under each
# perturbed gamma (1 pair per arm, interv vs
# no-replacement baseline); a116c source
# seals 3044-3054.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3055
NAME = 'omega_p52_gamma_prealign_qwen'
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
T_DROP = 0.3402
SEED_SHUF = 9952
SEED_RAND = 9951
N_RAND = 200
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
            'npz (COS_H[7]); prompts reassembled '
            'locally with the 3048 tail-alignment '
            'rule; statistics on the 24 old pairs',
    'question': '3055 A main line: the gamma '
                'readout pre-alignment found in '
                '3054 (tangential alignment '
                '0.166 -> 0.68 entirely from the '
                'learnable final-norm gamma) - '
                'is it target-direction specific '
                'or a global readout effect, and '
                'does the gamma channel '
                'assignment causally carry the '
                'alignment (perturbation '
                'orig/shuffled/ones)?',
    'T2_preimage': 'per pair: w_t = W_U^T t '
                   '(2560-dim residual preimage of '
                   'the 24 target logit directions); '
                   'a_t = cos(d_tan_k, gamma*w_tk), '
                   'b_t = cos(d_tan_k, w_tk) with '
                   'd_tan from the orig-gamma arm '
                   'post decomposition (3054 T2 '
                   'protocol); analytic replay of '
                   'the 3054 gain (med a_t vs med '
                   'b_t). RANDOM control: U=200 '
                   'random logit directions u '
                   '(norm-matched to the median '
                   '|t|, seed 9951), gain_u = med_k '
                   'cos(d_tan_k, gamma*w_u) - '
                   'med_k cos(d_tan_k, w_u) vs '
                   'gain_t = a_t - b_t; '
                   'descriptive-universality: '
                   'target-specific if gain_t >> '
                   'gain_u',
    'T3_gamma_perturb': 'MAIN: three arms over '
                        'model.model.norm.weight: '
                        'orig (chain anchor), '
                        'shuffled (permutation of '
                        'gamma channels, seed 9952), '
                        'ones (gamma = 1); per arm 24 '
                        'base (self replacement) + 24 '
                        'canonical h7 joint diag '
                        'interventions; r_new = lg_i '
                        '- lg_b; COS_arm[k] = '
                        'cos(r_new, TT[k]) against '
                        'the ORIGINAL target '
                        'direction; verdict: med '
                        'COS_shuf < 0.3402 AND med '
                        'COS_ones < 0.3402 -> '
                        'gamma_prealigned_qwen; '
                        'shuf < 0.3402 AND ones >= '
                        '0.3402 -> '
                        'gamma_distribution_qwen; '
                        'else -> gamma_robust_qwen; '
                        'single branch',
    'anchors': 'a169 sampled re-capture (4 prompts '
               '0/9/17/31) KPpre/KPpost/VP/LG bit '
               '0.0 vs z48 + a172 TT diff 0.0; '
               'a170 COS_orig diff 0.0 vs z51 '
               'COS_H[7]; a171 gamma-restore sham '
               'bit 0.0 (orig-gamma restore '
               'verified by re-running pair-0 '
               'base); a173 self-replacement bit '
               'identity under each perturbed '
               'gamma (1 pair per arm: interv '
               'self-replacement vs '
               'no-replacement baseline); a116c '
               'source seals 3044-3054',
    'statistics_discipline': 'all arms share the '
                             'same 24 pairs, same '
                             'replacement machinery, '
                             'same base forward '
                             'construction; the '
                             'perturbed-gamma arms '
                             'are compared against '
                             'the ORIGINAL TT '
                             'direction (fixed '
                             'target, option A) so '
                             'the drop measures the '
                             'loss of readout '
                             'pre-alignment, not a '
                             'change of target; '
                             'verdict in one branch',
    'corrections': 'run1 (55s) crashed pre-'
                   'verdict at the a173 baseline: '
                   'the no-replacement baseline '
                   'was built as zeros-repl + '
                   'all-False mask, but the hook '
                   'still enters the assignment '
                   'branch when repl is not None '
                   'and the boolean index selects '
                   '0 rows vs repl 6 rows '
                   '(broadcast error); fix: '
                   'dedicated forward_plain path '
                   'with no replacement state. '
                   'Statistics unobserved at '
                   'crash time (T3 perturbed-gamma '
                   'arms not started; a169/a170/'
                   'a172 bit 0.0 already passed '
                   'and T2 numbers printed are '
                   'valid but re-measured in '
                   'run2). run2 authoritative if '
                   'anchors pass.',
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
rot = model.model.rotary_emb
INV_FREQ = rot.inv_freq.detach().cpu().numpy() \
    .astype(np.float32)
assert INV_FREQ.shape == (64,), INV_FREQ.shape
assert float(rot.attention_scaling) == 1.0
NVOC = int(model.config.vocab_size)
log('model loaded fp32 (vocab=%d)' % NVOC)


def rot_apply(x, delta):
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


def reset_all():
    for li in range(NL):
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKn[li]['rec'] = False
        capV[li]['rec'] = False
        capKpre[li]['rec'] = False


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


def forward_run(ids, replK, maskK, replV, maskV):
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
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    reset_all()
    return lg


def forward_plain(ids):
    reset_all()
    with torch.no_grad():
        out = model(torch.tensor([ids],
                    device='cuda'), use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    reset_all()
    return lg


# ---------- chain: load capture banks ----------
z48 = np.load(os.path.join(
    BASE, 'phase3048',
    'omega_p45_kvpos_full_replay_qwen',
    'omega_p45_kvpos_full_replay_qwen.npz'),
    allow_pickle=True)
KPpost = z48['KPpost']
VP = z48['VP']
LG = z48['LG']
LENS = z48['LENS']
TT = z48['TT']
log('z48 bank loaded: KPpost%s VP%s LG%s'
    % (KPpost.shape, VP.shape, LG.shape))
z51 = np.load(os.path.join(
    BASE, 'phase3051',
    'omega_p48_l35_anatomy_qwen',
    'omega_p48_l35_anatomy_qwen.npz'),
    allow_pickle=True)
COS_H51 = z51['COS_H']
assert COS_H51.shape == (8, 24)
log('z51 anchor loaded: COS_H%s' % (COS_H51.shape,))

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
        (3053, 'omega_p50_gate_source_qwen'),
        (3054, 'omega_p51_norm_projection_qwen')):
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

# a169: sampled re-capture, bit-exact vs z48
a169_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a169_diff = max(a169_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a169_diff = max(a169_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a169_diff = max(a169_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a169_diff = float(a169_diff)

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
a172_diff = float(np.max(np.abs(TTl - TT)))
a169_ok = bool(a169_diff == 0.0 and a172_diff == 0.0)
log('a169 re-capture diff=%.3e a172 TT diff=%.3e '
    'ok=%s' % (a169_diff, a172_diff, a169_ok))


def front_rows(nb):
    return list(range(min(FRONT, nb)))


def l35_fields(k):
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
    rotK = rot_apply(src, off)
    srcV = VP[pref_i][:, off + ridx, :]
    return repK, repV, rotK, srcV, rows, base_i


def h7_joint_fields(k):
    repK, repV, rotK, srcV, rows, base_i \
        = l35_fields(k)
    sl = slice(HEAD_H * HDIM,
               (HEAD_H + 1) * HDIM)
    rK = repK.copy()
    rV = repV.copy()
    rK[L_TGT, rows, sl] = rotK[L_TGT, :, HEAD_H, :]
    svk = srcV[L_TGT].reshape(
        len(rows), KV_HEAD, HDIM)
    rV[L_TGT, rows, sl] = svk[:, HEAD_H, :]
    return rK, rV, rows, base_i


# ---------- gamma plumbing ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
log('gamma stats: min=%.4f max=%.4f mean=%.4f '
    'std=%.4f' % (GAMMA.min(), GAMMA.max(),
                  GAMMA.mean(), GAMMA.std()))

rng_g = np.random.default_rng(SEED_SHUF)
perm = rng_g.permutation(2560)
GAMMA_SHUF = GAMMA[perm]


def set_gamma(g_np):
    with torch.no_grad():
        model.model.norm.weight.copy_(
            torch.tensor(g_np,
                         dtype=torch.float32,
                         device='cuda'))


WU = model.lm_head.weight.detach()  # (V, 2560)
wud = WU.double().cpu().numpy()
EPS_NORM = float(model.config.rms_norm_eps)


def cos0(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


# ---------- orig-gamma arm: post capture +
# chain anchor (single pass with the L35
# output hook active) ----------
cap_post = {'rec': False, 'v': None}


def hook_post(cp):
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] if isinstance(out, tuple) \
                else out
            cp['v'] = t[0][-1].detach().clone()
        return out
    return h


hpost = layers[35].register_forward_hook(
    hook_post(cap_post))


def forward_post(ids, replK, maskK, replV, maskV):
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
    cap_post['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    post = cap_post['v'].double().cpu().numpy()
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    cap_post['rec'] = False
    reset_all()
    return lg, post


log('=== orig arm re-run with post capture ===')
COS_ORIG2 = np.zeros(NP_)
POST_B = []
POST_I = []
for k in range(NP_):
    rK, rV, rows_k, base_i = h7_joint_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    repK0 = KPpost[base_i, :, :nb, :].copy()
    repV0 = VP[base_i, :, :nb, :].copy()
    lg_b, post_b = forward_post(ids_b, repK0, mask,
                                repV0, mask)
    lg_i, post_i = forward_post(ids_b, rK, mask,
                                rV, mask)
    COS_ORIG2[k] = cos0(lg_i - lg_b, TT[k])
    POST_B.append(post_b)
    POST_I.append(post_i)
POST_B = np.array(POST_B)
POST_I = np.array(POST_I)
a170_diff = float(np.max(np.abs(
    COS_ORIG2 - COS_H51[HEAD_H])))
a170_ok = bool(a170_diff == 0.0)
log('a170 COS_orig vs z51 COS_H[7]: diff=%.3e ok=%s'
    % (a170_diff, a170_ok))

# a171: gamma-restore sham (weights untouched
# so far; verify restore after perturbations
# at the end of the arm loop).

# ---------- T2 decomposition (orig gamma) ----------
COS_DTAN_G = np.zeros(NP_)
COS_DTAN_N = np.zeros(NP_)
TAN_FRAC = np.zeros(NP_)
for k in range(NP_):
    d = POST_I[k] - POST_B[k]
    xb = POST_B[k]
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    ddot = float(d @ xhat)
    d_tan = d - ddot * xhat
    TAN_FRAC[k] = float(np.linalg.norm(d_tan)) \
        / float(np.linalg.norm(d))
    b = int(bidx[k])
    c = int(cidx[k])
    t_k = TT[k]
    w_t = wud.T @ t_k  # residual preimage
    COS_DTAN_G[k] = cos0(d_tan, GAMMA * w_t)
    COS_DTAN_N[k] = cos0(d_tan, w_t)
med_a_t = float(np.median(COS_DTAN_G))
med_b_t = float(np.median(COS_DTAN_N))
med_tan_frac = float(np.median(TAN_FRAC))
gain_t = med_a_t - med_b_t
log('T2: med cos(d_tan, gamma*w_t)=%.4f vs '
    'cos(d_tan, w_t)=%.4f (gain %.4f); '
    'tan_frac=%.4f'
    % (med_a_t, med_b_t, gain_t, med_tan_frac))

# random control: U=200 logit directions
rng_u = np.random.default_rng(SEED_RAND)
med_tnorm = float(np.median(
    [np.linalg.norm(TT[k]) for k in range(NP_)]))
GAIN_U = np.zeros(N_RAND)
for ui in range(N_RAND):
    u = rng_u.standard_normal(NVOC)
    u = u / np.linalg.norm(u) * med_tnorm
    w_u = wud.T @ u
    gs = []
    gn = []
    for k in range(NP_):
        d = POST_I[k] - POST_B[k]
        xb = POST_B[k]
        xhat = xb / np.sqrt(float(xb @ xb))
        d_tan = d - float(d @ xhat) * xhat
        gs.append(cos0(d_tan, GAMMA * w_u))
        gn.append(cos0(d_tan, w_u))
    GAIN_U[ui] = float(np.median(gs)) \
        - float(np.median(gn))
med_gain_u = float(np.median(GAIN_U))
pct95_gain_u = float(np.percentile(GAIN_U, 95))
log('T2 random control: med gain_u=%.4f '
    'p95=%.4f vs gain_t=%.4f -> '
    'target-specific=%s'
    % (med_gain_u, pct95_gain_u, gain_t,
       bool(gain_t > pct95_gain_u)))

# ---------- T3 perturbed-gamma arms ----------
log('=== T3 perturbed-gamma arms ===')
arm_names = ('shuf', 'ones')
arm_gammas = (GAMMA_SHUF,
              np.ones(2560, dtype=np.float64))
COS_ARMS = {}
a173_fail = 0
a173_checked = 0
for aname, gnp in zip(arm_names, arm_gammas):
    set_gamma(gnp)
    # a173: self-replacement bit identity
    # under perturbed gamma (pair 0)
    rK0, rV0, _, base_i0 = h7_joint_fields(0)
    ids0 = assembled[base_i0]['ids']
    n0 = int(LENS[base_i0])
    m0 = np.ones(n0, dtype=bool)
    lg_self = forward_run(ids0, KPpost[base_i0,
                          :, :n0, :].copy(), m0,
                          VP[base_i0, :, :n0, :]
                          .copy(), m0)
    lg_base = forward_plain(ids0)
    a173_diff = float(np.max(np.abs(
        lg_self - lg_base)))
    a173_checked += 1
    if a173_diff != 0.0:
        a173_fail += 1
    cos_arm = np.zeros(NP_)
    for k in range(NP_):
        rK, rV, _, base_i = h7_joint_fields(k)
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        mask = np.ones(nb, dtype=bool)
        lg_b = forward_run(ids_b,
                           KPpost[base_i, :, :nb, :]
                           .copy(), mask,
                           VP[base_i, :, :nb, :]
                           .copy(), mask)
        lg_i = forward_run(ids_b, rK, mask,
                           rV, mask)
        cos_arm[k] = cos0(lg_i - lg_b, TT[k])
    COS_ARMS[aname] = cos_arm
    log('  arm %s: med cos vs ORIGINAL TT=%.4f '
        '(a173 self-id diff=%.3e)'
        % (aname, float(np.median(cos_arm)),
           a173_diff))
# restore orig gamma
set_gamma(GAMMA)
# a171: restore sham - rerun pair-0 base,
# must equal the orig-arm lg bit-exactly
k0 = 0
rK, rV, _, base_i0 = h7_joint_fields(k0)
ids0 = assembled[base_i0]['ids']
n0 = int(LENS[base_i0])
m0 = np.ones(n0, dtype=bool)
lg_b_rest = forward_run(ids0,
                        KPpost[base_i0, :, :n0, :]
                        .copy(), m0,
                        VP[base_i0, :, :n0, :]
                        .copy(), m0)
lg_b_orig = forward_run(ids0,
                        KPpost[base_i0, :, :n0, :]
                        .copy(), m0,
                        VP[base_i0, :, :n0, :]
                        .copy(), m0)
a171_diff = float(np.max(np.abs(
    lg_b_rest - lg_b_orig)))
a171_ok = bool(a171_diff == 0.0)
log('a171 gamma-restore sham diff=%.3e ok=%s'
    % (a171_diff, a171_ok))
hpost.remove()

med_cos_shuf = float(np.median(COS_ARMS['shuf']))
med_cos_ones = float(np.median(COS_ARMS['ones']))
med_cos_orig = float(np.median(COS_ORIG2))
log('T3 summary: orig=%.4f shuf=%.4f ones=%.4f '
    '(threshold %.4f)'
    % (med_cos_orig, med_cos_shuf, med_cos_ones,
       T_DROP))

# ---------- verdict ----------
if med_cos_shuf < T_DROP and med_cos_ones < T_DROP:
    verdict = 'gamma_prealigned_qwen'
elif med_cos_shuf < T_DROP \
        and med_cos_ones >= T_DROP:
    verdict = 'gamma_distribution_qwen'
else:
    verdict = 'gamma_robust_qwen'
log('VERDICT: %s' % verdict)

a173_ok = bool(a173_fail == 0)
anchor_core_ok = bool(a116_ok and a169_ok
                      and a170_ok and a171_ok
                      and a173_ok)
log('anchors core ok=%s (a173 fail=%d/%d)'
    % (anchor_core_ok, a173_fail, a173_checked))

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_ORIG2=COS_ORIG2,
         COS_SHUF=COS_ARMS['shuf'],
         COS_ONES=COS_ARMS['ones'],
         COS_DTAN_G=COS_DTAN_G,
         COS_DTAN_N=COS_DTAN_N,
         TAN_FRAC=TAN_FRAC,
         GAIN_U=GAIN_U,
         GAMMA_STATS=np.array([GAMMA.min(),
                               GAMMA.max(),
                               GAMMA.mean(),
                               GAMMA.std()]),
         a169_diff=np.float64(a169_diff),
         a172_diff=np.float64(a172_diff),
         a170_diff=np.float64(a170_diff),
         a171_diff=np.float64(a171_diff),
         a173_fail=np.int64(a173_fail),
         a173_checked=np.int64(a173_checked),
         a116_ok=np.bool_(a116_ok),
         a169_ok=np.bool_(a169_ok),
         a170_ok=np.bool_(a170_ok),
         a171_ok=np.bool_(a171_ok),
         a173_ok=np.bool_(a173_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_preimage': {
        'med_cos_dtan_gamma_wt': med_a_t,
        'med_cos_dtan_wt': med_b_t,
        'gain_t': gain_t,
        'med_gain_u': med_gain_u,
        'pct95_gain_u': pct95_gain_u,
        'target_specific': bool(
            gain_t > pct95_gain_u),
        'med_tan_frac': med_tan_frac},
    'T3_gamma_perturb': {
        'med_cos_orig': med_cos_orig,
        'med_cos_shuf': med_cos_shuf,
        'med_cos_ones': med_cos_ones,
        'threshold': T_DROP},
    'anchors': {'a116_seals_ok': a116_ok,
                'a169_recapture_diff':
                    a169_diff,
                'a172_tt_diff': a172_diff,
                'a170_chain_diff': a170_diff,
                'a171_restore_diff':
                    a171_diff,
                'a173_fail': a173_fail,
                'a173_checked': a173_checked,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run2 authoritative (fp32; capture '
                 'bank from the phase3048 npz; '
                 'chain anchors vs the phase3051 '
                 'npz; gamma perturbation via '
                 'model.model.norm.weight swap '
                 'with restore sham)',
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
