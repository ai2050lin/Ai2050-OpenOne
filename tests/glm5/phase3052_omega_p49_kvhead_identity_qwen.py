# -*- coding: utf-8 -*-
# Phase 3052 - Omega-P49: KV head identity (h7/h6).
# 3051 localized the L35 KV payload to the K field
# (95.4pct) carried by heads h7 (0.6805) / h6
# (0.6367) with h1 negative (-0.1624) and
# readout-carrying separation. Here: WHO ARE h7/h6?
# (i) T2 cross-pair transfer matrices (PRIMARY):
# for h in {7,6,1}, full 24x24 matrix - insert
# pair j's head-h fields (K rotated + V, same
# head block, 3051 T3 protocol) into base k's
# L35 x FRONT rows; A[k,j] = cos(r, t_k),
# B[k,j] = cos(r, t_j); diagonal = 3051 COS_H
# chain anchor (bit-exact); off-diagonal paired
# difference D = B - A, sign-flip permutation
# (R=2000, one-sided) - B > A means the values
# carry pair-specific content (content-carrier);
# A >= B means context-assembled / generic slot.
# (ii) T3 static anatomy (descriptive): uniform-
# attention OV write u_h = M_h @ vbar_h (M_h =
# sum of the 4 GQA Q-head column blocks of
# o_proj), lm_head projection into vocab space
# aligned with t_k (K routing has no direct
# unembed projection; head identity probed via
# the OV write - preregistered interpretation);
# top-10 written tokens for h7/h6/h1; (iii) 3022
# coalition cross-chain linkage: w32_tau =
# sum_top32 s_relay[tau,j] * dh_j (L3 residual
# write direction of the relay coalition,
# dh = down_proj columns), cos(w32_tau, ubar_h)
# 8x11 matrix + w32/w_all ratio per head.
# verdict_tree (h7 primary): p_perm < 0.05 AND
# med_B_off > med_A_off -> kvhead_content_qwen;
# else -> kvhead_generic_qwen; single branch;
# h6/h1 descriptive.
# anchors: a150 sampled re-capture (4 prompts)
# bit 0.0 vs z48 + TT diff 0.0; a151 diagonal
# repro A_diag vs z51 COS_H[h] diff 0.0 for
# h in {7,6,1}; a152 sham self-replacement bit
# identity; a153 integrity bit-exact on the h7
# diagonal band (24 forwards, integ=True);
# a154 chain-entry max||dlg|| >= 0.05; a116c
# source seals 3044-3051.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3052
NAME = 'omega_p49_kvhead_identity_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 8
HDIM = 128
R_PERM = 2000
SEED_PERM = 9933
SEED_MAIN = 3010
T_GATE = 0.05
FRONT = 4
L_TGT = 35
HEADS_T = (7, 6, 1)
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
    'mode': 'fp32 MODEL (torch.float32, eager, seed '
            '3010); capture bank loaded from the '
            'phase3048 npz (KPpost/VP/LG/LENS/TT); '
            'chain anchors vs the phase3051 npz '
            '(COS_H) and the phase3022 npz '
            '(s_relay); prompts reassembled locally '
            'with the 3048 tail-alignment rule; '
            'statistics on the 24 old pairs',
    'question': '3052 A main line: what IS the '
                'h7/h6 carrier head - a content-'
                'carrying slot (values encode which '
                'prefix) or a generic routing slot '
                '(context assembles the direction)? '
                'Cross-pair transfer matrices (A '
                'vs B) decide; static OV/unembed '
                'alignment and the 3022 L3-coalition '
                'linkage are descriptive',
    'T2_transfer': 'for h in (7,6,1): all 24x24 '
                   'ordered pairs (k dest, j src): '
                   'insert pair j head-h fields '
                   '(rotK_j[L35] rotated by off_j + '
                   'srcV_j[L35], head block sl = '
                   'h*128..(h+1)*128) into base_k '
                   'L35 FRONT rows (other rows/'
                   'blocks self-replaced, all-True '
                   'mask); A[k,j] = cos(r, t_k), '
                   'B[k,j] = cos(r, t_j); diagonal '
                   'must reproduce 3051 COS_H[h] '
                   'bit-exactly (a151); off-diag '
                   'stats: med_A_off, med_B_off; '
                   'paired sign-flip permutation on '
                   'D = B - A (R=%d, seed %d + rep, '
                   'one-sided p = P(med D_flip >= '
                   'med D)); descriptive: '
                   'cos(t_j,t_k) off-diag baseline, '
                   'corr(A, ctgt), corr(B, ctgt)'
                   % (R_PERM, SEED_PERM),
    'T3_static': 'descriptive: uniform-attention '
                 'OV write u_hk = M_h @ vbar_hk '
                 '(vbar = mean V over FRONT rows of '
                 'base_k at L35; M_h = sum of the 4 '
                 'GQA Q-head 128-col blocks of '
                 'o_proj.weight, all 8 KV heads); '
                 'z_hk = lm_head(u_hk) vs t_k per '
                 'pair -> med per head; rank corr '
                 'with 3051 med_h; top-10 tokens of '
                 'lm_head(ubar_h) for h7/h6/h1; '
                 '3022 linkage: w32_tau = '
                 'down_proj[:, top32] @ s_relay[tau,'
                 'top32] (L3 coalition residual '
                 'write, 11 tags), cos(w32_tau, '
                 'ubar_h) 8x11 + w32/w_all ratio',
    'verdict_tree': 'h7 primary: p_perm < 0.05 AND '
                    'med_B_off > med_A_off -> '
                    'kvhead_content_qwen; else -> '
                    'kvhead_generic_qwen; single '
                    'branch; h6/h1 descriptive',
    'anchors': 'a150 sampled re-capture (4 prompts '
               '0/9/17/31) KPpre/KPpost/VP/LG bit '
               '0.0 vs z48 + TT diff 0.0; a151 '
               'diagonal repro A_diag vs z51 '
               'COS_H[h] diff 0.0 (h in 7/6/1); '
               'a152 sham self-replacement bit '
               'identity; a153 integrity bit-exact '
               '(mod == where(mask,repl,orig)) on '
               'the h7 diagonal band (24 forwards, '
               'integ=True); a154 chain-entry '
               'max||dlg|| >= %g; a116c source '
               'seals 3044-3051' % T_GATE,
    'control': 'paired within-entry contrast (B vs '
               'A on the same forward), no extra '
               'intervention; the diagonal band is '
               'the bit-exact 3051 protocol',
    'statistics_discipline': 'obs and perm null on '
                             'the SAME scale (paired '
                             'off-diag entries, same '
                             'forwards, sign-flip '
                             'label null); null never '
                             'on intervened quantities '
                             'beyond the label swap; '
                             'verdict in one branch',
    'corrections': 'none yet (run1 authoritative '
                   'if anchors pass)',
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
log('model loaded fp32 (vocab=%d)' % NVOC)


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


def forward_run(ids, replK=None, maskK=None,
                replV=None, maskV=None, integ=False):
    reset_all()
    rk = rv = None
    mk = mv = None
    if replK is not None:
        assert maskK is not None
        rk = torch.tensor(np.ascontiguousarray(
            replK.reshape(NL, -1, 8, HDIM)),
            dtype=torch.float32, device='cuda')
        mk = torch.tensor(np.asarray(maskK),
                          device='cuda')
        for li in range(NL):
            stateKn[li]['repl'] = rk[li]
            stateKn[li]['mask'] = mk
    if replV is not None:
        assert maskV is not None
        rv = torch.tensor(np.ascontiguousarray(
            replV),
            dtype=torch.float32, device='cuda')
        mv = torch.tensor(np.asarray(maskV),
                          device='cuda')
        for li in range(NL):
            stateV[li]['repl'] = rv[li]
            stateV[li]['mask'] = mv
    if integ:
        for li in range(NL):
            capKn[li]['rec'] = True
            capV[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    res = {'lg': lg}
    if integ:
        e = 0.0
        for li in range(NL):
            if rk is not None:
                comb = capKn[li]['orig'].clone()
                comb[mk] = rk[li]
                e = max(e, float(
                    (capKn[li]['mod'] - comb)
                    .abs().max()))
            if rv is not None:
                comb = capV[li]['orig'].clone()
                comb[mv] = rv[li]
                e = max(e, float(
                    (capV[li]['mod'] - comb)
                    .abs().max()))
        res['bit'] = e
    reset_all()
    return res


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
log('z51 anchors loaded: COS_H%s'
    % (COS_H51.shape,))
z22 = np.load(os.path.join(
    BASE, 'phase3022',
    'omega_p2p_l3_relay_neurons_qwen',
    'omega_p2p_l3_relay_neurons_qwen.npz'),
    allow_pickle=True)
S_RELAY = z22['s_relay']
assert S_RELAY.shape == (11, 9728)
log('z22 loaded: s_relay%s' % (S_RELAY.shape,))

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
        (3051, 'omega_p48_l35_anatomy_qwen')):
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

# a150: sampled re-capture, bit-exact vs z48
a150_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a150_diff = max(a150_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a150_diff = max(a150_diff, float(np.max(
        np.abs(KPpre[si, :, :n2, :] - kp2))))
    a150_diff = max(a150_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a150_diff = max(a150_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a150_diff = float(a150_diff)

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
a150_ok = bool(a150_diff == 0.0 and a_tt == 0.0)
log('a150 re-capture diff=%.3e TT diff=%.3e ok=%s'
    % (a150_diff, a_tt, a150_ok))

# a152: sham self-replacement (bit identity)
b0 = int(bidx[0])
base0 = idx_of[(0, b0)]
ids0 = assembled[base0]['ids']
n0 = len(ids0)
selfK = KPpost[base0, :, :n0, :].copy()
selfV = VP[base0, :, :n0, :].copy()
m_all = np.ones(n0, dtype=bool)
res = forward_run(ids0, replK=selfK, maskK=m_all,
                  replV=selfV, maskV=m_all,
                  integ=True)
a152_diff = float(np.max(np.abs(res['lg']
                                - LG[base0])))
a152_diff = float(max(a152_diff, res['bit']))
a152_ok = bool(a152_diff == 0.0)
log('a152 sham self-replacement diff=%.3e ok=%s'
    % (a152_diff, a152_ok))


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


NFR_ALL = None
for k in range(NP_):
    nb = int(LENS[idx_of[(0, int(bidx[k]))]])
    if NFR_ALL is None:
        NFR_ALL = len(front_rows(nb))
    assert len(front_rows(nb)) == NFR_ALL, k
assert NFR_ALL == 4
log('all base prompts have %d FRONT rows' % NFR_ALL)

a153_fail = 0
a153_checked = 0
rec_dlg = []

# ---------- T2 cross-pair transfer matrices ----------
log('=== T2 cross-pair transfer (heads %s) ==='
    % (HEADS_T,))
MAT_A = np.zeros((len(HEADS_T), NP_, NP_))
MAT_B = np.zeros((len(HEADS_T), NP_, NP_))
for hi, h in enumerate(HEADS_T):
    sl = slice(h * HDIM, (h + 1) * HDIM)
    for k in range(NP_):
        b = int(bidx[k])
        repK_k, repV_k, _, _, rows_k, base_i \
            = l35_fields(k)
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        mask = np.ones(nb, dtype=bool)
        for j in range(NP_):
            _, _, rotK_j, srcV_j, rows_j, _ \
                = l35_fields(j)
            rK = repK_k.copy()
            rV = repV_k.copy()
            rK[L_TGT, rows_k, sl] \
                = rotK_j[L_TGT, :, h, :]
            svj = srcV_j[L_TGT].reshape(
                len(rows_j), KV_HEAD, HDIM)
            rV[L_TGT, rows_k, sl] = svj[:, h, :]
            integ = bool(h == 7 and j == k)
            res = forward_run(ids_b, replK=rK,
                              maskK=mask,
                              replV=rV, maskV=mask,
                              integ=integ)
            if integ:
                a153_checked += 1
                if res.get('bit', 0.0) != 0.0:
                    a153_fail += 1
            r = res['lg'] - LG[base_i]
            t_k = t_targets[(b, int(cidx[k]))]
            cs, _ = cos_frac(LG[base_i], r, t_k)
            MAT_A[hi, k, j] = cs
            b_j = int(bidx[j])
            t_j = t_targets[(b_j, int(cidx[j]))]
            csj, _ = cos_frac(LG[base_i], r, t_j)
            MAT_B[hi, k, j] = csj
            rec_dlg.append(float(
                np.linalg.norm(r)))
    ad = float(np.median(np.diag(MAT_A[hi])))
    off = ~np.eye(NP_, dtype=bool)
    log('  head %d: diag med=%.4f | A_off med='
        '%.4f B_off med=%.4f'
        % (h, ad,
           float(np.median(MAT_A[hi][off])),
           float(np.median(MAT_B[hi][off]))))

# a151: diagonal repro vs z51 COS_H
a151_diffs = {}
a151_ok = True
for hi, h in enumerate(HEADS_T):
    d = float(np.max(np.abs(
        np.diag(MAT_A[hi]) - COS_H51[h])))
    a151_diffs['h%d' % h] = d
    a151_ok = bool(a151_ok and d == 0.0)
log('a151 diagonal repro vs z51 COS_H: %s ok=%s'
    % (a151_diffs, a151_ok))

# off-diag stats + paired sign-flip permutation
off = ~np.eye(NP_, dtype=bool)
tcos = np.zeros((NP_, NP_))
for k in range(NP_):
    for j in range(NP_):
        tk = TTl[k]
        tj = TTl[j]
        nk = float(np.linalg.norm(tk))
        nj = float(np.linalg.norm(tj))
        tcos[k, j] = float(tk @ tj) / (nk * nj) \
            if nk > 1e-12 and nj > 1e-12 else 0.0
T2 = {}
perm_res = {}
for hi, h in enumerate(HEADS_T):
    D = (MAT_B[hi] - MAT_A[hi])[off]
    medD = float(np.median(D))
    reps = np.empty(R_PERM)
    for rep in range(R_PERM):
        rng = np.random.default_rng(
            SEED_PERM + rep)
        sgn = rng.choice((-1.0, 1.0), size=D.shape)
        reps[rep] = float(np.median(D * sgn))
    p_perm = float(
        (1 + int(np.sum(reps >= medD)))
        / (R_PERM + 1))
    ct = tcos[off]
    Av = MAT_A[hi][off]
    Bv = MAT_B[hi][off]
    c_a = float(np.corrcoef(Av, ct)[0, 1]) \
        if float(np.std(Av)) > 1e-12 else float('nan')
    c_b = float(np.corrcoef(Bv, ct)[0, 1]) \
        if float(np.std(Bv)) > 1e-12 else float('nan')
    T2['h%d' % h] = {
        'med_A_diag': float(np.median(
            np.diag(MAT_A[hi]))),
        'med_B_diag': float(np.median(
            np.diag(MAT_B[hi]))),
        'med_A_off': float(np.median(Av)),
        'med_B_off': float(np.median(Bv)),
        'med_D_off': medD,
        'p_perm': p_perm,
        'ctgt_off_med': float(np.median(ct)),
        'corr_A_ctgt': c_a,
        'corr_B_ctgt': c_b}
    perm_res['h%d' % h] = reps
    log('  head %d: D med=%.4f p_perm=%.5f | '
        'ctgt med=%.4f corrA=%.3f corrB=%.3f'
        % (h, medD, p_perm, float(np.median(ct)),
           c_a, c_b))

# ---------- T3 static anatomy ----------
log('=== T3 static OV / unembed / coalition ===')
Wo = layers[L_TGT].self_attn.o_proj.weight \
    .detach()  # (2560, 4096)
WU = model.lm_head.weight.detach()  # (V, 2560)
M_h = np.zeros((KV_HEAD, 2560, HDIM),
               dtype=np.float64)
for h in range(KV_HEAD):
    blk = Wo[:, (4 * h) * HDIM:
             (4 * h + 4) * HDIM]  # (2560, 512)
    blk = blk.reshape(2560, 4, HDIM)
    M_h[h] = blk.sum(axis=1) \
        .double().cpu().numpy()
VBAR = np.zeros((NP_, KV_HEAD, HDIM))
for k in range(NP_):
    base_i = idx_of[(0, int(bidx[k]))]
    v35 = VP[base_i, L_TGT, :NFR_ALL, :] \
        .reshape(NFR_ALL, KV_HEAD, HDIM)
    VBAR[k] = v35.mean(axis=0)
COS_OV = np.zeros((KV_HEAD, NP_))
TOP_TOK = {}
for h in range(KV_HEAD):
    ubar = np.zeros(2560)
    for k in range(NP_):
        u = M_h[h] @ VBAR[k, h].astype(np.float64)
        z = WU.double().cpu().numpy() @ u
        tk = TTl[k]
        nt = float(np.linalg.norm(tk))
        nz = float(np.linalg.norm(z))
        COS_OV[h, k] = float(z @ tk) / (nz * nt) \
            if nz > 1e-12 and nt > 1e-12 else 0.0
        ubar += u
    ubar /= NP_
    zbar = WU.double().cpu().numpy() @ ubar
    top = np.argsort(-zbar)[:10]
    TOP_TOK['h%d' % h] = [tok.decode([int(i)])
                          .strip() for i in top]
med_ov = [float(np.median(COS_OV[h]))
          for h in range(KV_HEAD)]
sa = float(np.std(med_ov))
sb = float(np.std(med_h51 := [
    float(np.median(COS_H51[h]))
    for h in range(KV_HEAD)]))
rank_corr = float(np.corrcoef(
    _r := np.argsort(np.argsort(med_ov)),
    _r2 := np.argsort(np.argsort(med_h51)))[0, 1]) \
    if sa > 1e-12 and sb > 1e-12 else float('nan')
log('T3: OV-vs-target med per head %s'
    % np.array2string(np.array(med_ov),
                      precision=4))
log('T3: rank corr(OV med, 3051 causal med)='
    '%.4f' % rank_corr)
log('T3: top tokens h7=%s h6=%s h1=%s'
    % (TOP_TOK['h7'], TOP_TOK['h6'],
       TOP_TOK['h1']))

# 3022 coalition linkage (L3 residual write)
dh = layers[3].mlp.down_proj.weight.detach() \
    .double().cpu().numpy()  # (2560, 9728)
COS_COAL = np.zeros((KV_HEAD, 11))
RATIO_COAL = np.zeros((KV_HEAD, 11))
UBAR = np.zeros((KV_HEAD, 2560))
for h in range(KV_HEAD):
    UBAR[h] = (M_h[h] @ VBAR[:, h].T) \
        .mean(axis=1)
for tau in range(11):
    s = S_RELAY[tau].astype(np.float64)
    top32 = np.argsort(-np.abs(s))[:32]
    w32 = dh[:, top32] @ s[top32]
    w_all = dh @ s
    nw32 = float(np.linalg.norm(w32))
    nwall = float(np.linalg.norm(w_all))
    for h in range(KV_HEAD):
        nu = float(np.linalg.norm(UBAR[h]))
        if nu > 1e-12 and nw32 > 1e-12:
            COS_COAL[h, tau] = float(
                UBAR[h] @ w32) / (nu * nw32)
        if nu > 1e-12 and nwall > 1e-12:
            RATIO_COAL[h, tau] = COS_COAL[h, tau] \
                / (float(UBAR[h] @ w_all)
                   / (nu * nwall) + 1e-12)
med_coal = [float(np.median(COS_COAL[h]))
            for h in range(KV_HEAD)]
log('T3: cos(w32_coalition, ubar_h) med per '
    'head %s'
    % np.array2string(np.array(med_coal),
                      precision=4))
log('T3: h7 cos per tag %s'
    % np.array2string(COS_COAL[7], precision=3))
log('T3: h7 w32/w_all ratio per tag %s'
    % np.array2string(RATIO_COAL[7],
                      precision=3))

# a153 / a154
a153_ok = bool(a153_fail == 0
               and a153_checked >= 1)
max_dlg = float(np.max(np.array(rec_dlg)))
a154_ok = bool(max_dlg >= T_GATE)
log('a153 integrity fails=%d checked=%d ok=%s | '
    'a154 max||dlg||=%.4f ok=%s'
    % (a153_fail, a153_checked, a153_ok,
       max_dlg, a154_ok))

# ---------- verdict (h7 primary) ----------
t7 = T2['h7']
if t7['p_perm'] < 0.05 \
        and t7['med_B_off'] > t7['med_A_off']:
    verdict = 'kvhead_content_qwen'
else:
    verdict = 'kvhead_generic_qwen'
log('VERDICT: %s (h7: p=%.5f med_B_off=%.4f '
    'med_A_off=%.4f)'
    % (verdict, t7['p_perm'], t7['med_B_off'],
       t7['med_A_off']))

anchor_core_ok = bool(a116_ok and a150_ok
                      and a151_ok and a152_ok
                      and a153_ok and a154_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         MAT_A=MAT_A, MAT_B=MAT_B,
         COS_OV=COS_OV, COS_COAL=COS_COAL,
         RATIO_COAL=RATIO_COAL, UBAR=UBAR,
         tcos=tcos,
         a150_diff=np.float64(a150_diff),
         a151_diffs=np.array(
             [a151_diffs['h%d' % h]
              for h in HEADS_T]),
         a152_diff=np.float64(a152_diff),
         a153_fail=np.int64(a153_fail),
         a153_checked=np.int64(a153_checked),
         a154_max_dlg=np.float64(max_dlg),
         a116_ok=np.bool_(a116_ok),
         a150_ok=np.bool_(a150_ok),
         a151_ok=np.bool_(a151_ok),
         a152_ok=np.bool_(a152_ok),
         a153_ok=np.bool_(a153_ok),
         a154_ok=np.bool_(a154_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_transfer': T2,
    'T3_static': {
        'med_cos_ov_per_head': med_ov,
        'rank_corr_ov_vs_causal': rank_corr,
        'med_cos_coalition_per_head': med_coal,
        'cos_coal_h7_per_tag':
            [float(x) for x in COS_COAL[7]],
        'ratio_coal_h7_per_tag':
            [float(x) for x in RATIO_COAL[7]],
        'top_tokens': {kk: TOP_TOK[kk]
                       for kk in
                       ('h7', 'h6', 'h1')}},
    'anchors': {'a116_seals_ok': a116_ok,
                'a150_recapture_diff': a150_diff,
                'a151_diag_diffs': a151_diffs,
                'a152_sham_diff': a152_diff,
                'a153_fail': a153_fail,
                'a153_checked': a153_checked,
                'a154_max_dlg': max_dlg,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run1 authoritative (fp32; capture '
                 'bank from the phase3048 npz; '
                 'chain anchors vs the phase3051 '
                 'and phase3022 npz)',
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
