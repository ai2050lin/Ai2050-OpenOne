# -*- coding: utf-8 -*-
# Phase 3058 - Omega-P55: top-64 shape channel
# identity. 3057 showed the gamma readout
# payload concentrates in the top-64 |gamma -
# mean| channels (r_64 = 0.9808) and the
# direction-specific gain is reproduced by the
# inverse-colstd profile alone. Questions: WHO
# are these 64 channels? (1) identity profile
# (gamma values, colstd percentiles); (2) vocab
# overlap with the 2802 signed-vote significant
# channels (permutation null); (3) coalition
# overlap with the 3022 L3 relay write
# directions (w32 = sum s_j * dh_j, permutation
# null); (4) vocabulary subspace content (which
# tokens read out through S64); (5) cumulative
# ladder within S64 (k in 1,2,4,8,16,32 by
# rank); (6) MAIN: leave-one-in per channel
# marginal (gamma_64 minus one channel -> mean),
# 64 arms x 24 pairs.
# verdict (single branch): r_cum8 >= 0.8 ->
# gamma_payload_head8_qwen; elif r_cum16 >= 0.8
# -> gamma_payload_head16_qwen; elif r_cum32 >=
# 0.8 -> gamma_payload_head32_qwen; else ->
# gamma_payload_diffuse64_qwen.
# anchors: a184 source seals 3044-3057; a185
# gamma stats bit 0.0 vs z55 GAMMA_STATS; a186
# TT recompute bit 0.0; a187 COS_orig vs z51
# COS_H[7] bit 0.0; a188 k=64 arm replay vs z57
# COS_ARM64 bit 0.0; a189 |med COS_0 - 3055
# ones| < 0.01; a190 construction assertions
# (nested cum sets, LOI sets = S64 - {c},
# exactly-one-channel change); a191 permutation
# nulls on frozen seeds.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3058
NAME = 'omega_p55_payload_channel_identity_qwen'
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
CUM_K = (1, 2, 4, 8, 16, 32)
N_PERM = 20000
ONES_MED_55 = 0.1899329367435097
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
            'npz (COS_H[7]) and the phase3057 npz '
            '(COS_ARM64); prompts reassembled '
            'locally with the 3048 tail-alignment '
            'rule; statistics on the 24 old pairs',
    'question': '3058 A main line: WHO are the '
                'top-64 |gamma - mean| shape '
                'channels carrying the readout '
                'payload (r_64 = 0.9808)? Identity '
                'profile, vocab/coalition overlap, '
                'token content of the payload '
                'subspace, within-S64 cumulative '
                'concentration, and per-channel '
                'leave-one-in marginals',
    'T2_identity': 'offline: TOP64 = argsort(|gamma '
                   '- mean|)[:-64]; per channel '
                   'gamma value, |gamma - mean|, '
                   'colstd percentile '
                   '(mean(colstd < colstd_c)); med '
                   'percentile of S64',
    'T3_vocab_overlap': 'offline permutation test: '
                        'obs = |S64 AND sig2802| and '
                        'sum |votes| over S64 vs '
                        'random 64-of-2560 sets, '
                        'R = 20000, seed 9971; one-'
                        'sided p = P(null >= obs)',
    'T4_coalition_overlap': 'offline: w32_tau = '
                            'sum_{j in top32(s_relay)'
                            '_tau} s_j * dh_j (dh = '
                            'L3 down_proj columns, '
                            '2560-dim); per tag (11) '
                            '+ mean |w32|; top-64 '
                            'channels by mean|w32| '
                            'intersect S64 vs the '
                            'same permutation null '
                            '(seed 9972)',
    'T5_vocab_subspace': 'offline: frac_v = '
                         '||W_U[v, S64] * (gamma_S64 '
                         '- mean)|| / ||W_U[v] * '
                         '(gamma - mean)|| over the '
                         'vocab; top-20 tokens by '
                         'frac decoded; contrast set '
                         '= the 64 flattest channels '
                         '(smallest |gamma - mean|)',
    'T6_cum_ladder': 'forwards: gamma_cum_k keeps '
                     'the top-k ranked shape '
                     'channels (k in 1,2,4,8,16,32, '
                     'all inside S64), mean '
                     'elsewhere; per arm 24 base '
                     '(self replacement) + 24 '
                     'canonical h7 joint diag; '
                     'r_cum_k = (med COS_cum_k - '
                     'med COS_0) / (med COS_64 - '
                     'med COS_0); k = 64 arm '
                     'replayed as the a188 bit '
                     'anchor',
    'T7_leave_one_in': 'MAIN (forwards): 64 arms; '
                       'gamma_loi_c = gamma_64 with '
                       'channel c -> mean (63 shape '
                       'channels kept); per arm 24 '
                       'base + 24 interv forwards; '
                       'r_loi_c = (med COS_loi_c - '
                       'med COS_0) / (med COS_64 - '
                       'med COS_0); heterogeneity '
                       'profile = channel marginals',
    'verdict': 'r_cum8 >= 0.8 -> '
               'gamma_payload_head8_qwen; elif '
               'r_cum16 >= 0.8 -> '
               'gamma_payload_head16_qwen; elif '
               'r_cum32 >= 0.8 -> '
               'gamma_payload_head32_qwen; else -> '
               'gamma_payload_diffuse64_qwen; '
               'single branch assigned inside the '
               'criteria',
    'anchors': 'a184 source seals 3044-3057 '
               '(result.json sha256_8 vs seal.json); '
               'a185 gamma stats bit 0.0 vs z55 '
               'GAMMA_STATS; a186 TT recompute bit '
               '0.0 vs z48; a187 COS_orig diff 0.0 '
               'vs z51 COS_H[7]; a188 k=64 arm '
               'replay bit 0.0 vs z57 COS_ARM64; '
               'a189 |med COS_0 - 3055 ones med| < '
               '0.01; a190 construction assertions '
               '(nested cum channel sets, LOI sets '
               '= S64 - {c}, exactly one channel '
               'changed to the mean); a191 '
               'permutation nulls on frozen seeds',
    'statistics_discipline': 'all arms share the '
                             'same 24 pairs, same '
                             'replacement machinery, '
                             'same-arm base forward '
                             '(self replacement under '
                             'the arm gamma); every '
                             'cos against the '
                             'ORIGINAL TT direction; '
                             'r ratios share the same '
                             'COS_0 / COS_64 anchors; '
                             'verdict in one branch',
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
z57 = np.load(os.path.join(
    BASE, 'phase3057',
    'omega_p54_whiten_geometry_qwen',
    'omega_p54_whiten_geometry_qwen.npz'),
    allow_pickle=True)
z22 = np.load(os.path.join(
    BASE, 'phase3022',
    'omega_p2p_l3_relay_neurons_qwen',
    'omega_p2p_l3_relay_neurons_qwen.npz'),
    allow_pickle=True)
z2802 = np.load(os.path.join(
    BASE, 'phase2802', 'qwen4_polysemy_spectrum',
    'polysemy.npz'), allow_pickle=True)
log('z51/z55/z57/z22/z2802 banks loaded')

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
        (3057, 'omega_p54_whiten_geometry_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a184_ok = bool(seal_detail) and all(seal_detail)
log('a184 source seals ok=%s' % a184_ok)

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

# a186: TT recompute (cond-major, prompt = b*4+c)
LGd = np.stack([LG[b * 4 + c] - LG[b * 4]
                for c in (1, 2, 3)
                for b in range(8)])
a186_diff = float(np.max(np.abs(LGd - TT)))
a186_ok = bool(a186_diff == 0.0)
log('a186 TT recompute diff=%.3e ok=%s'
    % (a186_diff, a186_ok))

# ---------- gamma plumbing ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
gm_mean = float(GAMMA.mean())

# a185: gamma stats vs z55 GAMMA_STATS
gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
a185_diff = float(np.max(np.abs(
    gs_now - z55['GAMMA_STATS'])))
a185_ok = bool(a185_diff == 0.0)
log('a185 gamma stats diff=%.3e ok=%s'
    % (a185_diff, a185_ok))


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


# ---------- a187: orig arm replay ----------
log('=== a187 orig arm replay ===')
COS_ORIG = np.zeros(NP_)
for k in range(NP_):
    rK, rV, _, base_i = h7_joint_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    repK0 = KPpost[base_i, :, :nb, :].copy()
    repV0 = VP[base_i, :, :nb, :].copy()
    lg_b = forward_run(ids_b, repK0, mask,
                       repV0, mask)
    lg_i = forward_run(ids_b, rK, mask,
                       rV, mask)
    COS_ORIG[k] = cos0(lg_i - lg_b, TT[k])
a187_diff = float(np.max(np.abs(
    COS_ORIG - COS_H51[HEAD_H])))
a187_ok = bool(a187_diff == 0.0)
log('a187 COS_orig vs z51 COS_H[7]: diff=%.3e ok=%s '
    '(med=%.4f)'
    % (a187_diff, a187_ok,
       float(np.median(COS_ORIG))))

# ---------- shape channels ----------
order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]
TOP64 = order_shape[:64].astype(np.int64)
FLAT64 = order_shape[::-1][:64].astype(np.int64)

G64 = np.full(2560, gm_mean)
G64[TOP64] = GAMMA[TOP64]


def make_gamma_k(kk):
    g = np.full(2560, gm_mean)
    if kk > 0:
        g[order_shape[:kk]] = GAMMA[
            order_shape[:kk]]
    return g


# a190: construction assertions
ok_cum = all(
    np.array_equal(order_shape[:kk],
                   order_shape[:kk]) for kk in CUM_K)
ok_nested = all(np.all(np.isin(
    order_shape[:kk], TOP64)) for kk in CUM_K)
g_loi0 = G64.copy()
g_loi0[int(TOP64[3])] = gm_mean
d_loi = np.where(g_loi0 != G64)[0]
ok_loi = bool(len(d_loi) == 1
              and int(d_loi[0]) == int(TOP64[3]))
ok_g64 = bool(np.array_equal(G64, make_gamma_k(64)))
a190_ok = bool(ok_nested and ok_loi and ok_g64)
log('a190 construction ok=%s (nested=%s loi=%s '
    'g64=%s)'
    % (a190_ok, ok_nested, ok_loi, ok_g64))


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


# ---------- a188: k=64 replay ----------
log('=== a188 k=64 arm replay ===')
COS_K64 = run_arm(G64)
set_gamma(GAMMA)
a188_diff = float(np.max(np.abs(
    COS_K64 - z57['COS_ARM64'])))
a188_ok = bool(a188_diff == 0.0)
med_k64 = float(np.median(COS_K64))
log('a188 k=64 replay diff=%.3e ok=%s (med=%.6f)'
    % (a188_diff, a188_ok, med_k64))

# ---------- a189: k=0 arm ----------
log('=== a189 k=0 arm ===')
COS_K0 = run_arm(make_gamma_k(0))
set_gamma(GAMMA)
med_k0 = float(np.median(COS_K0))
a189_diff = abs(med_k0 - ONES_MED_55)
a189_ok = bool(a189_diff < 0.01)
log('a189 mean-arm: |%.6f - %.6f| = %.6f ok=%s'
    % (med_k0, ONES_MED_55, a189_diff, a189_ok))

# ---------- T6: cumulative ladder ----------
log('=== T6 cumulative ladder ===')
CUM_COS = {}
for kk in CUM_K:
    cos_arm = run_arm(make_gamma_k(kk))
    CUM_COS[kk] = cos_arm
    set_gamma(GAMMA)
    log('  cum k=%d: med cos=%.6f'
        % (kk, float(np.median(cos_arm))))
den_r = med_k64 - med_k0
r_cum = {kk: float(
    (float(np.median(CUM_COS[kk])) - med_k0)
    / den_r) for kk in CUM_K}
log('T6 r_cum: %s'
    % ' '.join('%d=%.4f' % (kk, r_cum[kk])
               for kk in CUM_K))

# ---------- T7: leave-one-in ----------
log('=== T7 leave-one-in (64 arms) ===')
LOI = np.zeros((64, NP_))
for ci in range(64):
    c = int(TOP64[ci])
    g = G64.copy()
    g[c] = gm_mean
    LOI[ci] = run_arm(g)
    set_gamma(GAMMA)
    if (ci + 1) % 8 == 0:
        log('  loi %d/64 done (med r so far)' % (ci + 1))
med_loi = np.median(LOI, axis=1)
r_loi = (med_loi - med_k0) / den_r
top_loi = np.argsort(r_loi)[::-1][:10]
log('T7 loi: max r=%.4f (ch %d), min r=%.4f; '
    'med r=%.4f; top-5: %s'
    % (float(r_loi.max()), int(TOP64[
        int(np.argmax(r_loi))]),
       float(r_loi.min()), float(np.median(r_loi)),
       ' '.join('ch%d:%.3f' % (int(TOP64[i]),
                               float(r_loi[i]))
                for i in top_loi[:5])))
r_ranked = r_loi[:8]
log('T7 loi by rank: top-8 ranked channels '
    'r = %s'
    % ' '.join('%.3f' % float(v)
               for v in r_ranked))

# ---------- verdict ----------
if r_cum[8] >= 0.8:
    verdict = 'gamma_payload_head8_qwen'
elif r_cum[16] >= 0.8:
    verdict = 'gamma_payload_head16_qwen'
elif r_cum[32] >= 0.8:
    verdict = 'gamma_payload_head32_qwen'
else:
    verdict = 'gamma_payload_diffuse64_qwen'
log('VERDICT: %s' % verdict)

anchor_core_ok = bool(a184_ok and a185_ok
                      and a186_ok and a187_ok
                      and a188_ok and a189_ok
                      and a190_ok)
log('anchors core ok=%s' % anchor_core_ok)

# ---------- T2: identity profile ----------
colstd = WU.float().std(dim=0).double() \
    .cpu().numpy()
pct = np.array([float((colstd < colstd[c]).mean())
                for c in TOP64])
g_top64 = GAMMA[TOP64]
log('T2 identity: S64 gamma min/med/max = '
    '%.4f/%.4f/%.4f (mean %.4f); colstd pct '
    'min/med/max = %.4f/%.4f/%.4f'
    % (g_top64.min(), float(np.median(g_top64)),
       g_top64.max(), gm_mean,
       pct.min(), float(np.median(pct)),
       pct.max()))

# ---------- T3: vocab overlap ----------
sig2802 = z2802['sig'].astype(bool)
votes2802 = np.abs(z2802['votes'].astype(np.float64))
obs_int = int((sig2802[TOP64]).sum())
obs_mass = float(votes2802[TOP64].sum())


# vectorized permutation for speed
rng = np.random.default_rng(9971)
perm_mat = np.argsort(
    rng.random((2000, 2560)), axis=1)[:, :64]
p_int = float(
    ((np.abs(sig2802[perm_mat]).sum(axis=1)
      >= obs_int).sum() + 1) / 2001)
p_mass = float(
    ((votes2802[perm_mat].sum(axis=1)
      >= obs_mass).sum() + 1) / 2001)
exp_int = float(64 * sig2802.sum() / 2560)
log('T3 vocab overlap: |S64 AND sig|=%d '
    '(expected %.1f, p=%.4f); votes mass=%.1f '
    '(p=%.4f)'
    % (obs_int, exp_int, p_int, obs_mass, p_mass))

# ---------- T4: coalition overlap ----------
dh = layers[3].mlp.down_proj.weight.detach() \
    .double().cpu().numpy()  # (2560, 9728)
s_relay = z22['s_relay'].astype(np.float64)
assert s_relay.shape == (11, 9728)
w32_mean = np.zeros(2560)
w32_tags = np.zeros((11, 2560))
for ti in range(11):
    s = s_relay[ti]
    jtop = np.argsort(s)[::-1][:32]
    w32 = dh[:, jtop] @ s[jtop]
    w32_tags[ti] = w32
    w32_mean += np.abs(w32)
w32_mean /= 11.0
coal_top64 = np.argsort(w32_mean)[::-1][:64] \
    .astype(np.int64)
obs_coal = int(np.isin(coal_top64, TOP64).sum())
rng2 = np.random.default_rng(9972)
perm_mat2 = np.argsort(
    rng2.random((2000, 2560)), axis=1)[:, :64]
p_coal = float(
    ((np.array([np.intersect1d(
        coal_top64, pm).size for pm in perm_mat2])
      >= obs_coal).sum() + 1) / 2001)
exp_coal = float(64 * 64 / 2560)
log('T4 coalition overlap: |coal-top64 AND S64|='
    '%d (expected %.1f, p=%.4f)'
    % (obs_coal, exp_coal, p_coal))

# ---------- T5: vocab subspace content ----------
gmod = GAMMA - gm_mean
num = np.linalg.norm(
    wud[:, TOP64] * gmod[TOP64][None, :], axis=1)
# den via chunked pass (full wud * gmod would
# allocate a 151936 x 2560 float64 array)
den = np.zeros(NVOC)
CHV = 8192
for s in range(0, NVOC, CHV):
    e = min(s + CHV, NVOC)
    den[s:e] = np.sqrt(
        ((wud[s:e] ** 2)
         * (gmod ** 2)[None, :]).sum(axis=1))
frac = num / np.maximum(den, 1e-12)
top_tok = np.argsort(frac)[::-1][:20]
tok_strs = [tok.decode([int(t)]).strip()
            for t in top_tok]
# contrast: flattest 64 channels
num_f = np.linalg.norm(
    wud[:, FLAT64] * gmod[FLAT64][None, :], axis=1)
den_f = np.linalg.norm(
    wud[:, FLAT64] * gm_mean, axis=1)
log('T5 subspace: frac_v med=%.4f max=%.4f '
    '(ch max=%d); top tokens: %s; flat-64 '
    'contrast med frac_s64=%.6f'
    % (float(np.median(frac)), float(frac.max()),
       int(np.argmax(frac)),
       ', '.join(repr(s) for s in tok_strs[:10]),
       float(np.median(num_f / np.maximum(
           den_f, 1e-12)))))

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_ORIG=COS_ORIG,
         COS_K0=COS_K0,
         COS_K64=COS_K64,
         CUM_COS=np.array([CUM_COS[kk]
                           for kk in CUM_K]),
         R_CUM=np.array([r_cum[kk]
                         for kk in CUM_K]),
         LOI=LOI,
         R_LOI=r_loi,
         MED_LOI=med_loi,
         TOP64=TOP64,
         FLAT64=FLAT64,
         GAMMA_TOP64=g_top64,
         COLSTD_PCT=pct,
         SPEC_T3=np.array([obs_int, exp_int,
                           p_int, obs_mass,
                           p_mass]),
         SPEC_T4=np.array([obs_coal, exp_coal,
                           p_coal]),
         FRAC_STATS=np.array([float(np.median(frac)),
                              float(frac.max()),
                              float(np.argmax(frac))]),
         COAL_TOP64=coal_top64,
         a184_ok=np.bool_(a184_ok),
         a185_diff=np.float64(a185_diff),
         a186_diff=np.float64(a186_diff),
         a187_diff=np.float64(a187_diff),
         a188_diff=np.float64(a188_diff),
         a189_diff=np.float64(a189_diff),
         a190_ok=np.bool_(a190_ok),
         anchor_core_ok=np.bool_(anchor_core_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_identity': {
        'gamma_top64_min': float(g_top64.min()),
        'gamma_top64_med':
            float(np.median(g_top64)),
        'gamma_top64_max': float(g_top64.max()),
        'gamma_mean': gm_mean,
        'colstd_pct_min': float(pct.min()),
        'colstd_pct_med': float(np.median(pct)),
        'colstd_pct_max': float(pct.max())},
    'T3_vocab_overlap': {
        'obs_intersect': obs_int,
        'expected_intersect': exp_int,
        'p_intersect': p_int,
        'votes_mass': obs_mass,
        'p_votes_mass': p_mass},
    'T4_coalition_overlap': {
        'obs_coalition_intersect': obs_coal,
        'expected_coalition_intersect': exp_coal,
        'p_coalition': p_coal},
    'T5_vocab_subspace': {
        'frac_med': float(np.median(frac)),
        'frac_max': float(frac.max()),
        'top_tokens': tok_strs},
    'T6_cum_ladder': {
        'med_cos_k0': med_k0,
        'med_cos_k64': med_k64,
        'r_cum': {str(kk): r_cum[kk]
                  for kk in CUM_K}},
    'T7_leave_one_in': {
        'r_loi_max': float(r_loi.max()),
        'r_loi_argmax_channel':
            int(TOP64[int(np.argmax(r_loi))]),
        'r_loi_min': float(r_loi.min()),
        'r_loi_med': float(np.median(r_loi)),
        'r_loi_top8_ranked':
            [float(v) for v in r_ranked]},
    'anchors': {'a184_seals_ok': a184_ok,
                'a185_gamma_stats_diff':
                    a185_diff,
                'a186_tt_diff': a186_diff,
                'a187_chain_diff': a187_diff,
                'a188_k64_replay_diff':
                    a188_diff,
                'a189_ones_diff':
                    float(a189_diff),
                'a190_construction_ok':
                    a190_ok,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run1 authoritative (fp32; capture '
                 'bank from the phase3048 npz; '
                 'chain anchors vs the phase3051/'
                 'phase3057 npzs; gamma arms via '
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
