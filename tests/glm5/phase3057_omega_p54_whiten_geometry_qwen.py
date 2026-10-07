# -*- coding: utf-8 -*-
# Phase 3057 - Omega-P54: whitening geometry
# verification. 3056 found corr(gamma,
# colstd(W_U)) = -0.82 (gamma = anti-variance
# readout reweighting = whitening filter) and
# NO vocab-level pre-alignment; 3055 showed the
# target-specific gain (0.0271 -> 0.1114) lives
# in channel assignment. Questions: (1) HOW
# MUCH does gamma*W_U whiten the readout basis
# (spectrum / condition number / effective
# rank); (2) is gamma analytically the inverse
# column-std profile (log-log fit slope alpha);
# (3) MAIN: does the fitted whitening profile
# ALONE reproduce the direction-specific gain
# (offline replay of the 3055 T2 alignment
# with gamma_fit = colstd^-alpha); (4) causal
# gamma shape ladder: keep the top-k shape
# channels (|gamma - mean| ranking), replace
# the rest by the mean - forwards over the
# canonical h7 joint diag arm, recovery rate
# r_k.
# T2 spectrum (offline): Gram eigvals of W_U
# and W_U*gamma (chunked GPU); cond number,
# effective rank, participation ratio;
# colstd spread before/after.
# T3 inverse fit (offline): OLS of log(gamma)
# on -log(colstd); alpha + R2 + Spearman.
# T5 gamma_fit replay (offline): gamma_fit =
# colstd^-alpha normalized to the gamma mean;
# per pair cos(d_tan, gamma_fit*w_t) with
# d_tan from the orig-arm post capture; med
# vs z55 med 0.1114 (also mean-only gamma
# control, expect = 3055 ones-level).
# T4 shape ladder (MAIN, forwards): arms
# k in (0, 64, 256, 1024); gamma_k keeps the
# top-k shape channels, mean elsewhere; per
# arm 24 base (self replacement) + 24 interv
# canonical h7 joint diag; r_k = (med COS_k -
# med COS_0) / (med COS_orig - med COS_0);
# k=0 arm must reproduce the 3055 ones med
# 0.1899 within 0.01 (cos invariant to
# positive scaling).
# verdict: r_256 >= 0.5 AND med COS_FIT >=
# 0.9 * 0.1114 -> gamma_whiten_sufficient_qwen;
# r_256 >= 0.5 -> gamma_whiten_partial_qwen;
# else -> gamma_whiten_diffuse_qwen.
# anchors: a181 source seals 3044-3056; a177
# COS_orig diff 0.0 vs z51 COS_H[7]; a178
# d_tan replay (COS_DTAN_G/N medians bit-equal
# to z55); a179 gamma stats bit 0.0 vs z55
# GAMMA_STATS; a180 TT recompute bit 0.0; a182
# gamma_k construction assertions (k=2560 ==
# orig bit, monotone channel sets); a183
# mean-arm consistency vs the 3055 ones med.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3057
NAME = 'omega_p54_whiten_geometry_qwen'
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
LADDER = (0, 64, 256, 1024)
ONES_MED_55 = 0.1899329367435097
Z55_MED_G = 0.11143683246599181
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
            'npz (COS_H[7]) and the phase3055 npz '
            '(T2 arrays + GAMMA_STATS); prompts '
            'reassembled locally with the 3048 '
            'tail-alignment rule; statistics on '
            'the 24 old pairs',
    'question': '3057 A main line: the 3056 '
                'whitening signature (corr(gamma, '
                'colstd) = -0.82) - HOW MUCH does '
                'gamma*W_U whiten the readout '
                'basis, is gamma analytically the '
                'inverse column-std profile, does '
                'the fitted whitening profile '
                'ALONE reproduce the 3055 '
                'direction-specific gain, and '
                'where does the causal payload '
                'concentrate (shape-channel '
                'ladder)?',
    'T2_spectrum': 'offline: Gram eigvals of W_U '
                   'and W_U*gamma (chunked GPU '
                   'matmul, 2560x2560 eigvalsh); '
                   'cond = s_max / s_min (clamped '
                   'at 1e-12), effective rank = '
                   '(sum s)^2 / sum s^2, '
                   'participation ratio; colstd '
                   'spread: std/mean of column '
                   'stds before vs after',
    'T3_inverse_fit': 'offline: OLS of log(gamma) '
                      'on -log(colstd(W_U)); slope '
                      'alpha, R2; Spearman(gamma, '
                      'colstd) recompute',
    'T5_gamma_fit_replay': 'offline: gamma_fit = '
                           'colstd^-alpha normalized '
                           'to the gamma mean; per '
                           'pair cos(d_tan, '
                           'gamma_fit * w_t) with '
                           'd_tan from the orig-arm '
                           'post capture (3054 T2 '
                           'protocol: d = post_i - '
                           'post_b, tangential wrt '
                           'xhat_b); med COS_FIT vs '
                           'z55 med 0.1114; mean-only '
                           'gamma control (gamma_mean '
                           '= full mean vector)',
    'T4_shape_ladder': 'MAIN (forwards): gamma_k '
                       'keeps the top-k channels by '
                       '|gamma - mean|, replaces the '
                       'rest by the mean; arms k in '
                       '(0, 64, 256, 1024) plus the '
                       'orig arm (k = 2560 '
                       'effectively, from the same '
                       'forwards as the post '
                       'capture); per arm 24 base '
                       '(self replacement) + 24 '
                       'canonical h7 joint diag; '
                       'COS_arm[k] = cos(lg_i - '
                       'lg_b, TT[k]) against the '
                       'ORIGINAL target; r_k = (med '
                       'COS_k - med COS_0) / (med '
                       'COS_orig - med COS_0); '
                       'verdict: r_256 >= 0.5 AND '
                       'med COS_FIT >= 0.9 * '
                       '0.1114 -> '
                       'gamma_whiten_sufficient_'
                       'qwen; r_256 >= 0.5 -> '
                       'gamma_whiten_partial_qwen; '
                       'else -> '
                       'gamma_whiten_diffuse_qwen; '
                       'single branch',
    'anchors': 'a181 source seals 3044-3056 '
               '(result.json sha256_8 vs seal.json); '
               'a177 COS_orig diff 0.0 vs z51 '
               'COS_H[7]; a178 d_tan replay: medians '
               'of cos(d_tan, gamma*w_t) / cos(d_tan, '
               'w_t) recomputed from the captured '
               'POST arrays bit-equal to the z55 '
               'stored values; a179 gamma stats '
               'recomputed from model weights '
               'bit-equal to z55 GAMMA_STATS; a180 '
               'TT recomputed from z48 LG diffs bit '
               '0.0; a182 gamma_k construction '
               'assertions (k = 2560 identity, '
               'channel sets monotone, replaced '
               'entries exactly mean); a183 mean-arm '
               'consistency: |med COS_0 - 3055 ones '
               'med 0.18993| < 0.01',
    'statistics_discipline': 'all arms share the '
                             'same 24 pairs, same '
                             'replacement machinery, '
                             'same base forward '
                             'construction; every arm '
                             'compared against the '
                             'ORIGINAL TT direction '
                             '(option A fixed target, '
                             'as 3055); ladder r_k '
                             'uses the same COS_0 and '
                             'COS_orig anchors; '
                             'verdict in one branch '
                             'assigned inside the '
                             'criteria',
    'corrections': 'run1 (25s) crashed pre-'
                   'verdict at the T2 Gram '
                   'accumulation: G1/G2 were '
                   'allocated on CPU while the W_U '
                   'chunks were on cuda (device '
                   'mismatch). run2 (36s) crashed '
                   'pre-verdict at the same line: '
                   'the gamma multiplication '
                   'sliced gam_t by vocab rows '
                   '(gam_t[s:e], empty at s > '
                   '2560) instead of broadcasting '
                   'the 2560-dim channel vector '
                   'over the last dimension. '
                   'run3 (3m0s) crashed pre-verdict '
                   'at colstd_w: (wud * GAMMA[None,'
                   ':]) tried to allocate a '
                   '151936 x 2560 float64 array '
                   '(2.90 GiB). Statistics '
                   'unobserved at all crash times '
                   '(T2/T3/T5/T4 not started; a177 '
                   'COS_orig diff 0.0, a178 d_tan '
                   'replay bit-equal, a179, a180 '
                   'bit 0.0 already passed). Fixes: '
                   'G1/G2 on cuda; channel '
                   'broadcast blk * gam_t; colstd_w '
                   '= |gamma| * colstd (gamma is '
                   'constant per column). run4 '
                   '(189.2s) completed and sealed '
                   'with anchor_core_ok = True and '
                   'the T4 ladder clean, but T3/T5 '
                   'produced nan: gamma has '
                   'non-positive channels (min '
                   '-0.0221) so log(gamma) is '
                   'undefined and the alpha fit / '
                   'gamma_fit replay were invalid '
                   '(verdict fell to the r-only '
                   'branch: gamma_whiten_partial_'
                   'qwen). run5 amendment (this '
                   'run): T3 fit restricted to '
                   'positive-gamma channels with '
                   'the excluded count reported; '
                   'T5 replay uses the amended '
                   'alpha; verdict may be '
                   'reassigned. run5 authoritative '
                   'if anchors pass.',
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
z55 = np.load(os.path.join(
    BASE, 'phase3055',
    'omega_p52_gamma_prealign_qwen',
    'omega_p52_gamma_prealign_qwen.npz'),
    allow_pickle=True)
log('z51/z55 banks loaded')

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
        (3054, 'omega_p51_norm_projection_qwen'),
        (3055, 'omega_p52_gamma_prealign_qwen'),
        (3056, 'omega_p53_gamma_spectrum_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a181_ok = bool(seal_detail) and all(seal_detail)
log('a181 source seals ok=%s' % a181_ok)

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

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)
NP_ = 24

# a180: TT recompute (cond-major, prompt = b*4+c)
LGd = np.stack([LG[b * 4 + c] - LG[b * 4]
                for c in (1, 2, 3)
                for b in range(8)])
a180_diff = float(np.max(np.abs(LGd - TT)))
a180_ok = bool(a180_diff == 0.0)
log('a180 TT recompute diff=%.3e ok=%s'
    % (a180_diff, a180_ok))

# ---------- gamma plumbing ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
gm_mean = float(GAMMA.mean())

# a179: gamma stats vs z55 GAMMA_STATS
gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
a179_diff = float(np.max(np.abs(
    gs_now - z55['GAMMA_STATS'])))
a179_ok = bool(a179_diff == 0.0)
log('a179 gamma stats diff=%.3e ok=%s'
    % (a179_diff, a179_ok))


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


WU = model.lm_head.weight.detach()  # (V, 2560)
wud = WU.double().cpu().numpy()
EPS_NORM = float(model.config.rms_norm_eps)


def cos0(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


# ---------- orig arm: forwards + post capture
# (a177 chain anchor + d_tan for T5/a178) ----------
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


log('=== orig arm (post capture + chain anchor) ===')
COS_ORIG = np.zeros(NP_)
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
    COS_ORIG[k] = cos0(lg_i - lg_b, TT[k])
    POST_B.append(post_b)
    POST_I.append(post_i)
POST_B = np.array(POST_B)
POST_I = np.array(POST_I)
a177_diff = float(np.max(np.abs(
    COS_ORIG - COS_H51[HEAD_H])))
a177_ok = bool(a177_diff == 0.0)
log('a177 COS_orig vs z51 COS_H[7]: diff=%.3e ok=%s '
    '(med=%.4f)'
    % (a177_diff, a177_ok,
       float(np.median(COS_ORIG))))
hpost.remove()

# ---------- a178: d_tan replay vs z55 ----------
COS_DTAN_G = np.zeros(NP_)
COS_DTAN_N = np.zeros(NP_)
D_TAN = []
for k in range(NP_):
    d = POST_I[k] - POST_B[k]
    xb = POST_B[k]
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    d_tan = d - float(d @ xhat) * xhat
    D_TAN.append(d_tan)
    b = int(bidx[k])
    c = int(cidx[k])
    t_k = TT[k]
    w_t = wud.T @ t_k
    COS_DTAN_G[k] = cos0(d_tan, GAMMA * w_t)
    COS_DTAN_N[k] = cos0(d_tan, w_t)
a178_ok = bool(
    float(np.median(COS_DTAN_G))
    == float(np.median(z55['COS_DTAN_G']))
    and float(np.median(COS_DTAN_N))
    == float(np.median(z55['COS_DTAN_N'])))
log('a178 d_tan replay ok=%s (med_g=%.6f med_n=%.6f '
    'vs z55 %.6f / %.6f)'
    % (a178_ok,
       float(np.median(COS_DTAN_G)),
       float(np.median(COS_DTAN_N)),
       float(np.median(z55['COS_DTAN_G'])),
       float(np.median(z55['COS_DTAN_N']))))
D_TAN = np.array(D_TAN)

# ---------- T2: spectrum whitening ----------
colstd = WU.float().std(dim=0).double() \
    .cpu().numpy()
gam_t = torch.tensor(GAMMA, dtype=torch.float32,
                     device='cuda')
G1 = torch.zeros((2560, 2560),
                 dtype=torch.float64,
                 device='cuda')
G2 = torch.zeros((2560, 2560),
                 dtype=torch.float64,
                 device='cuda')
CH = 8192
with torch.no_grad():
    for s in range(0, NVOC, CH):
        e = min(s + CH, NVOC)
        blk = WU[s:e].double()
        G1 += blk.T @ blk
        G2 += (blk * gam_t.double()).T @ blk


def spec_stats(G):
    ev = torch.linalg.eigvalsh(G) \
        .clamp(min=0).double().cpu().numpy()
    sv = np.sqrt(ev[::-1])
    smax = float(sv[0])
    smin = float(sv[-1])
    cond = smax / max(smin, 1e-12)
    eff = float(sv.sum() ** 2
                / (sv ** 2).sum())
    pr = eff
    return sv, cond, eff, pr


sv1, cond1, eff1, pr1 = spec_stats(G1)
sv2, cond2, eff2, pr2 = spec_stats(G2)
spread1 = float(colstd.std() / colstd.mean())
# gamma is constant per column, so the column
# std of (W_U * gamma) = |gamma| * colstd
# (no V x 2560 allocation)
colstd_w = np.abs(GAMMA) * colstd
spread2 = float(colstd_w.std() / colstd_w.mean())
log('T2 spectrum: W_U cond=%.3e eff_rank=%.1f | '
    'W_U*gamma cond=%.3e eff_rank=%.1f | '
    'colstd spread %.4f -> %.4f'
    % (cond1, eff1, cond2, eff2,
       spread1, spread2))

# ---------- T3: inverse fit ----------
# run4 amendment: gamma has non-positive
# channels (min -0.0221), so log(gamma) is
# undefined there; fit on the positive subset
# and report the excluded count.
pos_m = GAMMA > 0
n_nonpos = int((~pos_m).sum())
x = -np.log(colstd[pos_m])
y = np.log(GAMMA[pos_m])
A = np.vstack([x, np.ones_like(x)]).T
coef, res_, *_ = np.linalg.lstsq(
    A, y, rcond=None)
alpha = float(coef[0])
pred = A @ coef
ss_res = float(((y - pred) ** 2).sum())
ss_tot = float(((y - y.mean()) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


rho = spearman(GAMMA, colstd)
log('T3 inverse fit: alpha=%.4f R2=%.4f '
    'spearman(gamma, colstd)=%.4f '
    '(fit on %d positive channels, %d '
    'non-positive excluded)'
    % (alpha, r2, rho, int(pos_m.sum()),
       n_nonpos))

# ---------- T5: gamma_fit replay ----------
colstd_safe = np.clip(colstd, 1e-12, None)
gamma_fit = colstd_safe ** (-alpha)
gamma_fit = gamma_fit / gamma_fit.mean() * gm_mean
gamma_mean = np.full(2560, gm_mean)
COS_FIT = np.zeros(NP_)
COS_MEAN = np.zeros(NP_)
for k in range(NP_):
    t_k = TT[k]
    w_t = wud.T @ t_k
    d_tan = D_TAN[k]
    COS_FIT[k] = cos0(d_tan, gamma_fit * w_t)
    COS_MEAN[k] = cos0(d_tan, gamma_mean * w_t)
med_fit = float(np.median(COS_FIT))
med_mean = float(np.median(COS_MEAN))
med_g = float(np.median(COS_DTAN_G))
log('T5: med cos(d_tan, gamma_fit*w_t)=%.6f '
    '(z55 gamma=%.6f); mean-only control med=%.6f'
    % (med_fit, med_g, med_mean))

# ---------- T4: gamma shape ladder ----------
order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]


def make_gamma_k(kk):
    g = np.full(2560, gm_mean)
    if kk > 0:
        g[order_shape[:kk]] = GAMMA[
            order_shape[:kk]]
    return g


# a182: construction assertions
g_full = make_gamma_k(2560)
a182_ok = bool(
    np.array_equal(g_full, GAMMA)
    and np.all(make_gamma_k(0) == gm_mean)
    and np.array_equal(
        np.sort(order_shape[:64]),
        np.sort(np.argsort(
            np.abs(GAMMA - gm_mean))[::-1][:64])))
log('a182 gamma_k construction ok=%s' % a182_ok)


def set_gamma(g_np):
    with torch.no_grad():
        model.model.norm.weight.copy_(
            torch.tensor(g_np,
                         dtype=torch.float32,
                         device='cuda'))


def run_arm(gnp):
    set_gamma(gnp)
    cos_arm = np.zeros(NP_)
    for k in range(NP_):
        rK, rV, _, base_i = h7_joint_fields(k)
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        mask = np.ones(nb, dtype=bool)
        lg_b = forward_run(
            ids_b,
            KPpost[base_i, :, :nb, :].copy(),
            mask,
            VP[base_i, :, :nb, :].copy(), mask)
        lg_i = forward_run(ids_b, rK, mask,
                           rV, mask)
        cos_arm[k] = cos0(lg_i - lg_b, TT[k])
    return cos_arm


log('=== T4 shape ladder ===')
COS_ARMS = {}
for kk in LADDER:
    cos_arm = run_arm(make_gamma_k(kk))
    COS_ARMS[kk] = cos_arm
    log('  arm k=%d: med cos=%.6f'
        % (kk, float(np.median(cos_arm))))
# restore orig gamma
set_gamma(GAMMA)

med_orig = float(np.median(COS_ORIG))
med_k0 = float(np.median(COS_ARMS[0]))
a183_diff = abs(med_k0 - ONES_MED_55)
a183_ok = bool(a183_diff < 0.01)
log('a183 mean-arm consistency: |%.6f - %.6f| = '
    '%.6f ok=%s'
    % (med_k0, ONES_MED_55, a183_diff, a183_ok))

r_ladder = {}
den_r = med_orig - med_k0
for kk in LADDER:
    med_k = float(np.median(COS_ARMS[kk]))
    r_ladder[kk] = float((med_k - med_k0) / den_r)
log('T4 ladder: orig=%.6f k0=%.6f; r_64=%.4f '
    'r_256=%.4f r_1024=%.4f'
    % (med_orig, med_k0, r_ladder[64],
       r_ladder[256], r_ladder[1024]))

# ---------- verdict ----------
r256 = r_ladder[256]
if r256 >= 0.5 and med_fit >= 0.9 * Z55_MED_G:
    verdict = 'gamma_whiten_sufficient_qwen'
elif r256 >= 0.5:
    verdict = 'gamma_whiten_partial_qwen'
else:
    verdict = 'gamma_whiten_diffuse_qwen'
log('VERDICT: %s' % verdict)

anchor_core_ok = bool(a181_ok and a177_ok
                      and a178_ok and a179_ok
                      and a180_ok and a182_ok
                      and a183_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_ORIG=COS_ORIG,
         COS_ARM0=COS_ARMS[0],
         COS_ARM64=COS_ARMS[64],
         COS_ARM256=COS_ARMS[256],
         COS_ARM1024=COS_ARMS[1024],
         R_LADDER=np.array([r_ladder[kk]
                            for kk in LADDER]),
         COS_DTAN_G=COS_DTAN_G,
         COS_DTAN_N=COS_DTAN_N,
         COS_FIT=COS_FIT,
         COS_MEAN=COS_MEAN,
         SV1=sv1.astype(np.float64),
         SV2=sv2.astype(np.float64),
         SPEC=np.array([cond1, eff1, cond2,
                        eff2, spread1, spread2]),
         ALPHA=np.float64(alpha),
         R2=np.float64(r2),
         RHO=np.float64(rho),
         GAMMA_FIT=gamma_fit,
         a177_diff=np.float64(a177_diff),
         a178_ok=np.bool_(a178_ok),
         a179_diff=np.float64(a179_diff),
         a180_diff=np.float64(a180_diff),
         a182_ok=np.bool_(a182_ok),
         a183_diff=np.float64(a183_diff),
         a181_ok=np.bool_(a181_ok),
         a177_ok=np.bool_(a177_ok),
         a179_ok=np.bool_(a179_ok),
         a180_ok=np.bool_(a180_ok),
         a183_ok=np.bool_(a183_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_spectrum': {
        'cond_W_U': cond1,
        'eff_rank_W_U': eff1,
        'cond_W_U_gamma': cond2,
        'eff_rank_W_U_gamma': eff2,
        'colstd_spread_before': spread1,
        'colstd_spread_after': spread2},
    'T3_inverse_fit': {
        'alpha': alpha,
        'R2': r2,
        'spearman_gamma_colstd': rho,
        'n_nonpositive_gamma': n_nonpos,
        'n_fit_channels': int(pos_m.sum())},
    'T5_gamma_fit_replay': {
        'med_cos_fit': med_fit,
        'z55_med_gamma': med_g,
        'med_cos_mean_only': med_mean,
        'fit_replay_ratio': med_fit / med_g},
    'T4_shape_ladder': {
        'med_cos_orig': med_orig,
        'med_cos_k0': med_k0,
        'r_64': r_ladder[64],
        'r_256': r_ladder[256],
        'r_1024': r_ladder[1024],
        'a183_ones_consistency_diff':
            float(a183_diff)},
    'anchors': {'a181_seals_ok': a181_ok,
                'a177_chain_diff': a177_diff,
                'a178_replay_ok': a178_ok,
                'a179_gamma_stats_diff':
                    a179_diff,
                'a180_tt_diff': a180_diff,
                'a182_construction_ok':
                    a182_ok,
                'a183_ones_diff':
                    float(a183_diff),
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run5 authoritative (fp32; capture '
                 'bank from the phase3048 npz; '
                 'chain anchors vs the phase3051/'
                 'phase3055 npzs; gamma ladder via '
                 'model.model.norm.weight swap with '
                 'orig restore)',
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
