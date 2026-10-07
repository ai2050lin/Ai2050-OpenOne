# -*- coding: utf-8 -*-
# Phase 3059 - Omega-P56: combinatorial payload
# subspace geometry. 3058 showed the gamma
# payload is combinatorial: rank-cumulative
# r_cum = 0.08 / 0.39 / 0.87 / 0.99 (k = 1/8/
# 16/32) with full one-at-a-time redundancy.
# Question: WHY are 16 channels enough? If the
# 24 target readout preimages v_k = gamma *
# W_U^T t_k live (mostly) in a low-dim
# subspace whose channel support is S16, the
# head16 result is a dimensionality
# consequence. T2 (offline): SVD of the
# v_k family -> effective dimension d_eff +
# per-PC energy in S16/S64. T3 (MAIN,
# offline): med ||v_k[S16]||^2 / ||v_k||^2 vs
# 16-of-64 and 16-of-2560 permutation nulls
# (R = 2000, seeds 9981/9982). T5 (48 capture
# forwards + offline): d_tan energies in
# S16/S64 vs the same nulls - do the write
# side and the read side share the subspace?
# verdict (single branch): p_T3 < 0.05 AND
# d_eff <= 16 -> payload_subspace16_qwen;
# p_T3 < 0.05 -> payload_subspace_partial_
# qwen; else -> payload_subspace_null_qwen.
# anchors: a192 source seals 3044-3058; a193
# gamma stats bit 0.0 vs z55 GAMMA_STATS; a194
# TT recompute bit 0.0 vs z48; a195 TOP64/
# FLAT64 recompute bit 0.0 vs z58; a196
# COS_orig vs z51 COS_H[7] bit 0.0; a197
# d_tan replay: med cos(d_tan, gamma*w_t) /
# cos(d_tan, w_t) bit-equal to z55.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3059
NAME = 'omega_p56_payload_subspace_qwen'
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
N_PERM = 2000
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
            'TT); chain anchors vs the phase3051/'
            'phase3055/phase3058 npzs; prompts '
            'reassembled locally with the 3048 '
            'tail-alignment rule; statistics on '
            'the 24 old pairs',
    'question': '3059 A main line: WHY is the '
                'gamma payload combinatorial '
                'head16 (3058: r_cum8 = 0.39, '
                'r_cum16 = 0.87, full one-at-a-'
                'time redundancy)? Do the 24 '
                'target readout preimages and '
                'the d_tan write directions live '
                'in a low-dim subspace with '
                'channel support S16 - i.e. is '
                'head16 a dimensionality '
                'consequence?',
    'T2_family_spectrum': 'offline: V = stack of '
                          'v_k = gamma * w_t '
                          '(w_t = W_U^T t_k, 24 x '
                          '2560); SVD -> d_eff = '
                          '(sum s)^2 / sum s^2; '
                          'per-PC energy in S16 = '
                          'TOP64[:16] and S64',
    'T3_E16_enrichment': 'MAIN (offline): E16_k = '
                         '||v_k[S16]||^2 / ||v_k||^2; '
                         'med E16 vs 16-of-64 (seed '
                         '9981) and 16-of-2560 (seed '
                         '9982) permutation nulls, R '
                         '= 2000 each, one-sided p = '
                         'P(null >= obs)',
    'T5_dtan_subspace': '48 capture forwards (orig '
                        'h7 joint arm replay with '
                        'post capture) -> d_tan per '
                        'pair (3054 protocol: d = '
                        'post_i - post_b tangential '
                        'wrt xhat_b); energy of '
                        'd_tan in S16/S64 vs the '
                        'same nulls; a197 replay '
                        'anchor: med cos(d_tan, '
                        'gamma*w_t) / cos(d_tan, '
                        'w_t) bit-equal to z55',
    'verdict': 'p_T3_64 < 0.05 AND d_eff <= 16 -> '
               'payload_subspace16_qwen; p_T3_64 < '
               '0.05 -> payload_subspace_partial_'
               'qwen; else -> payload_subspace_'
               'null_qwen; single branch assigned '
               'inside the criteria',
    'anchors': 'a192 source seals 3044-3058 '
               '(result.json sha256_8 vs seal.json); '
               'a193 gamma stats bit 0.0 vs z55 '
               'GAMMA_STATS; a194 TT recompute bit '
               '0.0 vs z48; a195 TOP64/FLAT64 '
               'recompute bit 0.0 vs z58; a196 '
               'COS_orig diff 0.0 vs z51 COS_H[7]; '
               'a197 d_tan replay medians bit-equal '
               'to z55 COS_DTAN_G/COS_DTAN_N',
    'statistics_discipline': 'all energies '
                             'computed on the same '
                             'gamma-weighted vectors; '
                             'nulls are channel-set '
                             'permutations on frozen '
                             'seeds; verdict in one '
                             'branch',
    'corrections': 'none (first run).',
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


for li in range(NL):
    layers[li].self_attn.k_norm \
        .register_forward_hook(hook_norm(
            stateKn[li], capKn[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))


def reset_all():
    for li in range(NL):
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKn[li]['rec'] = False
        capV[li]['rec'] = False


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
z58 = np.load(os.path.join(
    BASE, 'phase3058',
    'omega_p55_payload_channel_identity_qwen',
    'omega_p55_payload_channel_identity_qwen.npz'),
    allow_pickle=True)
log('z51/z55/z58 banks loaded')

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
        (3056, 'omega_p53_gamma_spectrum_qwen'),
        (3057, 'omega_p54_whiten_geometry_qwen'),
        (3058, 'omega_p55_payload_channel_'
               'identity_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a192_ok = bool(seal_detail) and all(seal_detail)
log('a192 source seals ok=%s' % a192_ok)

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

# a194: TT recompute (cond-major, prompt = b*4+c)
LGd = np.stack([LG[b * 4 + c] - LG[b * 4]
                for c in (1, 2, 3)
                for b in range(8)])
a194_diff = float(np.max(np.abs(LGd - TT)))
a194_ok = bool(a194_diff == 0.0)
log('a194 TT recompute diff=%.3e ok=%s'
    % (a194_diff, a194_ok))

# ---------- gamma plumbing ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
gm_mean = float(GAMMA.mean())

# a193: gamma stats vs z55 GAMMA_STATS
gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
a193_diff = float(np.max(np.abs(
    gs_now - z55['GAMMA_STATS'])))
a193_ok = bool(a193_diff == 0.0)
log('a193 gamma stats diff=%.3e ok=%s'
    % (a193_diff, a193_ok))

# a195: TOP64/FLAT64 recompute vs z58
order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]
TOP64 = order_shape[:64].astype(np.int64)
FLAT64 = order_shape[::-1][:64].astype(np.int64)
S16 = TOP64[:16].astype(np.int64)
a195_diff = max(
    float(np.max(np.abs(TOP64
                        - z58['TOP64']))),
    float(np.max(np.abs(FLAT64
                        - z58['FLAT64']))))
a195_ok = bool(a195_diff == 0.0)
log('a195 TOP64/FLAT64 recompute diff=%.3e ok=%s'
    % (a195_diff, a195_ok))


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


def cos0(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


# ---------- orig arm: forwards + post capture
# (a196 chain anchor + d_tan for T5/a197) ----------
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
a196_diff = float(np.max(np.abs(
    COS_ORIG - COS_H51[HEAD_H])))
a196_ok = bool(a196_diff == 0.0)
log('a196 COS_orig vs z51 COS_H[7]: diff=%.3e ok=%s '
    '(med=%.4f)'
    % (a196_diff, a196_ok,
       float(np.median(COS_ORIG))))
hpost.remove()

# ---------- a197: d_tan replay vs z55 ----------
D_TAN = []
COS_DTAN_G = np.zeros(NP_)
COS_DTAN_N = np.zeros(NP_)
for k in range(NP_):
    d = POST_I[k] - POST_B[k]
    xb = POST_B[k]
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    d_tan = d - float(d @ xhat) * xhat
    D_TAN.append(d_tan)
    t_k = TT[k]
    w_t = wud.T @ t_k
    COS_DTAN_G[k] = cos0(d_tan, GAMMA * w_t)
    COS_DTAN_N[k] = cos0(d_tan, w_t)
a197_ok = bool(
    float(np.median(COS_DTAN_G))
    == float(np.median(z55['COS_DTAN_G']))
    and float(np.median(COS_DTAN_N))
    == float(np.median(z55['COS_DTAN_N'])))
log('a197 d_tan replay ok=%s (med_g=%.6f med_n=%.6f)'
    % (a197_ok, float(np.median(COS_DTAN_G)),
       float(np.median(COS_DTAN_N))))
D_TAN = np.array(D_TAN)

# ---------- T2: family spectrum ----------
Vmat = np.zeros((NP_, 2560))
for k in range(NP_):
    w_t = wud.T @ TT[k]
    Vmat[k] = GAMMA * w_t
# SVD of the family
U, sv, Vt = np.linalg.svd(Vmat,
                          full_matrices=False)
d_eff = float(sv.sum() ** 2 / (sv ** 2).sum())
# per-PC energy in S16 / S64 (columns of Vt are
# right singular vectors in channel space)
pc_e16 = (Vt[:, S16] ** 2).sum(axis=1)
pc_e64 = (Vt[:, TOP64] ** 2).sum(axis=1)
log('T2 family: d_eff=%.2f (sv[0..5]=%s); '
    'PC energy in S16: top-8 PCs %s; in S64: %s'
    % (d_eff,
       np.array2string(sv[:6], precision=2),
       np.array2string(pc_e16[:8], precision=3),
       np.array2string(pc_e64[:8], precision=3)))
# cumulative energy of the family captured by
# the S16-channel coordinates
fam_e16 = float(
    (sv ** 2 * pc_e16).sum() / (sv ** 2).sum())
fam_e64 = float(
    (sv ** 2 * pc_e64).sum() / (sv ** 2).sum())
log('T2 family energy: S16 share=%.4f S64 share='
    '%.4f' % (fam_e16, fam_e64))

# ---------- T3: E16 enrichment ----------
E16 = np.array([
    float((Vmat[k, S16] ** 2).sum()
          / (Vmat[k] ** 2).sum())
    for k in range(NP_)])
E64 = np.array([
    float((Vmat[k, TOP64] ** 2).sum()
          / (Vmat[k] ** 2).sum())
    for k in range(NP_)])
med_e16 = float(np.median(E16))
med_e64 = float(np.median(E64))
rng1 = np.random.default_rng(9981)
pm1 = np.argsort(
    rng1.random((N_PERM, 64)), axis=1)[:, :16]
null_16of64 = np.array([
    float(np.median(
        (Vmat[:, TOP64[pm]] ** 2).sum(axis=1)
        / (Vmat ** 2).sum(axis=1)))
    for pm in pm1])
p16_64 = float((null_16of64 >= med_e16).sum()
               + 1) / (N_PERM + 1)
rng2 = np.random.default_rng(9982)
pm2 = np.argsort(
    rng2.random((N_PERM, 2560)), axis=1)[:, :16]
null_16of2560 = np.array([
    float(np.median(
        (Vmat[:, pm] ** 2).sum(axis=1)
        / (Vmat ** 2).sum(axis=1)))
    for pm in pm2])
p16_2560 = float((null_16of2560 >= med_e16).sum()
                 + 1) / (N_PERM + 1)
null_64of2560 = np.array([
    float(np.median(
        (Vmat[:, pm] ** 2).sum(axis=1)
        / (Vmat ** 2).sum(axis=1)))
    for pm in np.argsort(
        rng2.random((N_PERM, 2560)),
        axis=1)[:, :64]])
p64_2560 = float((null_64of2560 >= med_e64).sum()
                 + 1) / (N_PERM + 1)
log('T3: med E16=%.4f vs 16-of-64 null med=%.4f '
    '(p=%.4f), 16-of-2560 null med=%.4f (p=%.4f); '
    'med E64=%.4f vs 64-of-2560 null med=%.4f '
    '(p=%.4f)'
    % (med_e16, float(np.median(null_16of64)),
       p16_64, float(np.median(null_16of2560)),
       p16_2560, med_e64,
       float(np.median(null_64of2560)), p64_2560))

# ---------- T5: d_tan subspace ----------
DE16 = np.array([
    float((D_TAN[k, S16] ** 2).sum()
          / (D_TAN[k] ** 2).sum())
    for k in range(NP_)])
DE64 = np.array([
    float((D_TAN[k, TOP64] ** 2).sum()
          / (D_TAN[k] ** 2).sum())
    for k in range(NP_)])
med_de16 = float(np.median(DE16))
med_de64 = float(np.median(DE64))
null_d_16of64 = np.array([
    float(np.median(
        (D_TAN[:, TOP64[pm]] ** 2).sum(axis=1)
        / (D_TAN ** 2).sum(axis=1)))
    for pm in pm1])
p_d16_64 = float((null_d_16of64 >= med_de16).sum()
                 + 1) / (N_PERM + 1)
log('T5 d_tan: med E16=%.4f vs 16-of-64 null '
    'med=%.4f (p=%.4f); med E64=%.4f'
    % (med_de16, float(np.median(null_d_16of64)),
       p_d16_64, med_de64))

# ---------- verdict ----------
if p16_64 < 0.05 and d_eff <= 16:
    verdict = 'payload_subspace16_qwen'
elif p16_64 < 0.05:
    verdict = 'payload_subspace_partial_qwen'
else:
    verdict = 'payload_subspace_null_qwen'
log('VERDICT: %s' % verdict)

anchor_core_ok = bool(a192_ok and a193_ok
                      and a194_ok and a195_ok
                      and a196_ok and a197_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_ORIG=COS_ORIG,
         COS_DTAN_G=COS_DTAN_G,
         COS_DTAN_N=COS_DTAN_N,
         D_TAN=D_TAN,
         Vmat=Vmat,
         SV=sv,
         D_EFF=np.float64(d_eff),
         PC_E16=pc_e16,
         PC_E64=pc_e64,
         FAM_E=np.array([fam_e16, fam_e64]),
         E16=E16,
         E64=E64,
         NULL_16OF64=null_16of64,
         NULL_16OF2560=null_16of2560,
         NULL_64OF2560=null_64of2560,
         DE16=DE16,
         DE64=DE64,
         NULL_D_16OF64=null_d_16of64,
         S16=S16,
         a192_ok=np.bool_(a192_ok),
         a193_diff=np.float64(a193_diff),
         a194_diff=np.float64(a194_diff),
         a195_diff=np.float64(a195_diff),
         a196_diff=np.float64(a196_diff),
         a197_ok=np.bool_(a197_ok),
         anchor_core_ok=np.bool_(anchor_core_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_family_spectrum': {
        'd_eff': d_eff,
        'fam_energy_S16_share': fam_e16,
        'fam_energy_S64_share': fam_e64,
        'pc_energy_S16_top8':
            [float(v) for v in pc_e16[:8]]},
    'T3_E16_enrichment': {
        'med_E16': med_e16,
        'null16of64_med':
            float(np.median(null_16of64)),
        'p16_of64': p16_64,
        'null16of2560_med':
            float(np.median(null_16of2560)),
        'p16_of2560': p16_2560,
        'med_E64': med_e64,
        'null64of2560_med':
            float(np.median(null_64of2560)),
        'p64_of2560': p64_2560},
    'T5_dtan_subspace': {
        'med_DE16': med_de16,
        'null_d16of64_med':
            float(np.median(null_d_16of64)),
        'p_d16_of64': p_d16_64,
        'med_DE64': med_de64},
    'anchors': {'a192_seals_ok': a192_ok,
                'a193_gamma_stats_diff':
                    a193_diff,
                'a194_tt_diff': a194_diff,
                'a195_top64_diff': a195_diff,
                'a196_chain_diff': a196_diff,
                'a197_dtan_replay_ok': a197_ok,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run1 authoritative (fp32; capture '
                 'bank from the phase3048 npz; '
                 'chain anchors vs the phase3051/'
                 'phase3055/phase3058 npzs; 48 '
                 'capture forwards for d_tan, rest '
                 'offline)',
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
