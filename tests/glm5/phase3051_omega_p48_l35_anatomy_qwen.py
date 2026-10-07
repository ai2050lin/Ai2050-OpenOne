# -*- coding: utf-8 -*-
# Phase 3051 - Omega-P48: L35 final-layer anatomy.
# 3050 localized the carrying KV payload to the
# final layer L35 (single-layer med cos 0.7026 =
# full deep band 0.6954). Here L35 is dissected:
# (i) K/V split (3045/3046 protocol): K-only,
# V-only, joint exact replacement at L35 x FRONT
# band, 24 pairs (joint = 3050 COS_SL[35] chain
# anchor, bit-exact); (ii) per-KV-head joint
# replacement (8 heads individually, K+V same
# head) + leave-one-out (all heads but one);
# (iii) attention readout alignment (descriptive):
# L35 attention weights captured with
# output_attentions=True on the base prompts,
# per-GQA-group (4 query heads per KV head) mass
# from the last position onto the FRONT positions,
# correlated with the per-head replacement effect;
# (iv) PRIMARY null: norm-matched random K+V
# replacement restricted to the L35 x FRONT rows
# (other rows self-replaced), R=200, 24-pair
# median, p = P(null >= obs).
# verdict_tree: p_null >= 0.05 -> kv35_null_qwen;
# else max_h med_h >= 0.8 * med_joint ->
# kv35_headfocus_qwen; else -> kv35_mixed_qwen;
# single branch. K/V dominance descriptive
# (k_only vs v_only med).
# anchors: a145 sampled re-capture (4 prompts)
# bit 0.0 vs z48; a146a joint-L35 repro per-pair
# cos vs z50 COS_SL[35] diff 0.0; a146b FRONT
# all-layer repro per-pair cos vs z50 COS_FR
# diff 0.0; a147 sham self-replacement bit
# identity; a148 integrity bit-exact on the
# joint-L35 band (24 forwards, integ=True) plus
# the sham; a149 chain-entry max||dlg|| >= 0.05.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3051
NAME = 'omega_p48_l35_anatomy_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 8
HDIM = 128
R_NULL = 200
SEED_NULL = 9931
SEED_MAIN = 3010
T_GATE = 0.05
FRONT = 4
L_TGT = 35
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
            'phase3048 npz (KPpost/VP/LG/LENS/TT, 32 '
            'old-body prompts); per-pair chain '
            'anchors vs the phase3050 npz '
            '(COS_SL[35], COS_FR); prompts '
            'reassembled locally with the 3048 '
            'tail-alignment rule; statistics on the '
            '24 old pairs',
    'question': '3051 A main line: INSIDE L35, which '
                'component carries the KV payload - '
                'K field, V field, or specific KV '
                'heads - and does the per-head '
                'effect align with the attention '
                'readout mass? (i) K/V split '
                '(K-only, V-only, joint); (ii) 8 '
                'per-head joint replacements + 8 '
                'leave-one-out; (iii) attention '
                'readout alignment (descriptive, '
                'GQA group mass vs head effect); '
                '(iv) PRIMARY null: norm-matched '
                'random K+V restricted to L35 x '
                'FRONT rows (R=200, seed 9931+mc, '
                '24-pair median, p = P(null >= '
                'obs))',
    'T2_kvsplit': 'per pair: exact replacement at '
                  'layer 35 FRONT band only '
                  '(other rows self-replaced): '
                  'K-only (post-norm K, RoPE offset '
                  'rotation), V-only, joint; med cos '
                  'over 24 pairs each; joint must '
                  'reproduce the 3050 COS_SL[35] '
                  'per-pair cos bit-exactly (a146a)',
    'T3_heads': 'per-head joint replacement: for '
                'each KV head h in 0..7, replace '
                'ONLY head h block (K rotated + V) '
                'at L35 x FRONT rows, other head '
                'blocks keep base self values; med '
                'cos per head (8); leave-one-out: '
                'all heads except h replaced (8); '
                'descriptive; dominance = max_h '
                'med_h vs med_joint',
    'T4_attn_readout': 'descriptive: base-prompt '
                       'forward with '
                       'output_attentions=True; at '
                       'L35, per GQA group g (query '
                       'heads 4g..4g+3) attention '
                       'mass from the last position '
                       'summed over the FRONT '
                       'positions (24, 8); Pearson '
                       'corr per pair across heads '
                       'with COS_H, and global corr '
                       'of mean mass vs med_h',
    'T5_null': 'PRIMARY: R=%d MC, seed %d + mc; '
               'per draw, rows of layer 35 at the '
               'FRONT positions replaced by '
               'norm-matched gaussian random '
               '(K and V, per-position norms '
               'matched to the joint repl field), '
               'all other rows keep the exact '
               'self-replacement values; stat = '
               '24-pair median cos, p = '
               'P(null >= obs), obs = med_joint'
               % (R_NULL, SEED_NULL),
    'verdict_tree': 'p_null >= 0.05 -> '
                    'kv35_null_qwen; else max_h '
                    'med_h >= 0.8 * med_joint -> '
                    'kv35_headfocus_qwen; else -> '
                    'kv35_mixed_qwen; single '
                    'branch; K/V dominance '
                    'descriptive',
    'anchors': 'a145 sampled re-capture (4 prompts '
               '0/9/17/31) KPpre/KPpost/VP/LG bit '
               '0.0 vs z48; a146a joint-L35 repro '
               'per-pair cos vs z50 COS_SL[35] '
               'diff 0.0; a146b FRONT all-layer '
               'repro per-pair cos vs z50 COS_FR '
               'diff 0.0; a147 sham self-'
               'replacement bit identity; a148 '
               'integrity bit-exact (mod == '
               'where(mask,repl,orig)) on the '
               'joint-L35 band (24 forwards, '
               'integ=True) plus the sham; a149 '
               'chain-entry max||dlg|| >= %g'
               % T_GATE,
    'control': 'norm-matched random replacement '
               'restricted to the same rows as the '
               'exact field (L35 x FRONT); no '
               'other intervention',
    'statistics_discipline': 'obs and null on the '
                             'SAME scale (raw cos, '
                             'same 24 pairs, same '
                             'replacement machinery, '
                             'null randomized only '
                             'inside the L35 FRONT '
                             'rows); null never on '
                             'intervened quantities; '
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
assert hasattr(layers[3].self_attn, 'k_norm')
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


def forward_attn(ids):
    """Attention weights at L35 (base forward,
    output_attentions=True); per GQA group mass
    from the last position onto the FRONT rows."""
    reset_all()
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True,
                    output_attentions=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    aw = out.attentions[L_TGT][0].detach() \
        .double().cpu().numpy()  # (32, s, s)
    reset_all()
    return lg, aw


# ---------- chain: load z48 capture bank ----------
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
z50 = np.load(os.path.join(
    BASE, 'phase3050',
    'omega_p47_kvdeep_dissection_qwen',
    'omega_p47_kvdeep_dissection_qwen.npz'),
    allow_pickle=True)
COS_SL50 = z50['COS_SL']
COS_FR50 = z50['COS_FR']
log('z50 anchors loaded: COS_SL%s COS_FR%s'
    % (COS_SL50.shape, COS_FR50.shape))

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
               'qwen')):
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

# a145: sampled re-capture, bit-exact vs z48
a145_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a145_diff = max(a145_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a145_diff = max(a145_diff, float(np.max(
        np.abs(KPpre[si, :, :n2, :] - kp2))))
    a145_diff = max(a145_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a145_diff = max(a145_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a145_diff = float(a145_diff)
a145_ok = bool(a145_diff == 0.0)
log('a145 re-capture diff=%.3e ok=%s'
    % (a145_diff, a145_ok))

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
a145_ok = bool(a145_ok and a_tt == 0.0)
log('a145b TT vs z48 diff=%.3e' % a_tt)

# a147: sham self-replacement (bit identity)
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
a147_diff = float(np.max(np.abs(res['lg']
                                - LG[base0])))
a147_diff = float(max(a147_diff, res['bit']))
a147_ok = bool(a147_diff == 0.0)
log('a147 sham self-replacement diff=%.3e ok=%s'
    % (a147_diff, a147_ok))


def front_rows(nb):
    return list(range(min(FRONT, nb)))


def l35_fields(k):
    """Per pair: base self fields (NL, nb, 1024)
    plus the L35 pref fields for the FRONT rows
    (K rotated by off, V raw), returned as
    (repK, repV, rotK35 (nfr,8,128),
    srcV35 (nfr,1024), rows)."""
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


# integrity bookkeeping (a148): the joint-L35
# band forwards run with integ=True; every arm
# records ||dlg|| for a149.
a148_fail = 0
a148_checked = 0
rec_dlg = []

# ---------- T2 K/V split + joint (chain anchor) ----------
log('=== T2 K/V split at L35 x FRONT ===')
COS_J = np.zeros(NP_)
FRAC_J = np.zeros(NP_)
COS_K = np.zeros(NP_)
FRAC_K = np.zeros(NP_)
COS_V = np.zeros(NP_)
FRAC_V = np.zeros(NP_)
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    repK, repV, rotK, srcV, rows, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    # joint (both K and V at L35 rows)
    rjK = repK.copy()
    rjV = repV.copy()
    rjK[L_TGT, rows, :] = rotK[L_TGT] \
        .reshape(len(rows), HDIM * 8)
    rjV[L_TGT, rows, :] = srcV[L_TGT]
    integ = True
    res = forward_run(ids_b, replK=rjK, maskK=mask,
                      replV=rjV, maskV=mask,
                      integ=integ)
    a148_checked += 1
    if res.get('bit', 0.0) != 0.0:
        a148_fail += 1
    r = res['lg'] - LG[base_i]
    t = t_targets[(b, c)]
    cs, fr = cos_frac(LG[base_i], r, t)
    COS_J[k] = cs
    FRAC_J[k] = fr
    rec_dlg.append(float(np.linalg.norm(r)))
    # K-only
    rkK = repK.copy()
    rkK[L_TGT, rows, :] = rotK[L_TGT] \
        .reshape(len(rows), HDIM * 8)
    res = forward_run(ids_b, replK=rkK, maskK=mask,
                      replV=repV, maskV=mask)
    r = res['lg'] - LG[base_i]
    cs, fr = cos_frac(LG[base_i], r, t)
    COS_K[k] = cs
    FRAC_K[k] = fr
    rec_dlg.append(float(np.linalg.norm(r)))
    # V-only
    rvV = repV.copy()
    rvV[L_TGT, rows, :] = srcV[L_TGT]
    res = forward_run(ids_b, replK=repK, maskK=mask,
                      replV=rvV, maskV=mask)
    r = res['lg'] - LG[base_i]
    cs, fr = cos_frac(LG[base_i], r, t)
    COS_V[k] = cs
    FRAC_V[k] = fr
    rec_dlg.append(float(np.linalg.norm(r)))
med_j = float(np.median(COS_J))
med_k = float(np.median(COS_K))
med_v = float(np.median(COS_V))
log('T2: joint med=%.4f K-only med=%.4f '
    'V-only med=%.4f' % (med_j, med_k, med_v))

# a146a: joint-L35 repro vs z50 COS_SL[35]
a146a_diff = float(np.max(np.abs(
    COS_J - COS_SL50[L_TGT])))
a146a_ok = bool(a146a_diff == 0.0)
log('a146a joint-L35 repro vs z50 COS_SL[35]: '
    'diff=%.3e ok=%s' % (a146a_diff, a146a_ok))

# ---------- T3 per-head + leave-one-out ----------
log('=== T3 per-head joint replacement (8) ===')
COS_H = np.zeros((KV_HEAD, NP_))
COS_LOO = np.zeros((KV_HEAD, NP_))
for h in range(KV_HEAD):
    sl = slice(h * HDIM, (h + 1) * HDIM)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        repK, repV, rotK, srcV, rows, base_i \
            = l35_fields(k)
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        mask = np.ones(nb, dtype=bool)
        # head h only (K+V)
        rhK = repK.copy()
        rhV = repV.copy()
        rhK[L_TGT, rows, sl] = rotK[L_TGT, :, h, :]
        sv = srcV[L_TGT].reshape(len(rows),
                                 KV_HEAD, HDIM)
        rhV[L_TGT, rows, sl] = sv[:, h, :]
        res = forward_run(ids_b, replK=rhK,
                          maskK=mask, replV=rhV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        t = t_targets[(b, c)]
        cs, _ = cos_frac(LG[base_i], r, t)
        COS_H[h, k] = cs
        rec_dlg.append(float(np.linalg.norm(r)))
        # leave-one-out (all heads but h)
        loK = repK.copy()
        loV = repV.copy()
        loK[L_TGT, rows, :] = rotK[L_TGT] \
            .reshape(len(rows), HDIM * KV_HEAD)
        loV[L_TGT, rows, :] = srcV[L_TGT]
        loK[L_TGT, rows, sl] = \
            repK[L_TGT, rows, sl]
        loV[L_TGT, rows, sl] = \
            repV[L_TGT, rows, sl]
        res = forward_run(ids_b, replK=loK,
                          maskK=mask, replV=loV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        cs, _ = cos_frac(LG[base_i], r, t)
        COS_LOO[h, k] = cs
        rec_dlg.append(float(np.linalg.norm(r)))
    log('  head %d: only med=%.4f loo med=%.4f'
        % (h, float(np.median(COS_H[h])),
           float(np.median(COS_LOO[h]))))
med_h = [float(np.median(COS_H[h]))
         for h in range(KV_HEAD)]
med_loo = [float(np.median(COS_LOO[h]))
           for h in range(KV_HEAD)]
h_star = int(np.argmax(med_h))
loo_drop = med_j - med_loo[h_star]
log('T3: best head h*=%d med=%.4f (joint %.4f); '
    'loo drop at h*=%.4f'
    % (h_star, med_h[h_star], med_j, loo_drop))

# ---------- T4 attention readout (descriptive) ----------
log('=== T4 attention readout mass (GQA groups) ===')
MASS = np.zeros((NP_, KV_HEAD))
for k in range(NP_):
    b = int(bidx[k])
    base_i = idx_of[(0, b)]
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    rows = front_rows(nb)
    _, aw = forward_attn(ids_b)
    for g in range(KV_HEAD):
        MASS[k, g] = float(
            aw[4 * g:4 * g + 4, -1, :][:, rows]
            .sum())
per_pair_corr = []
for k in range(NP_):
    a = MASS[k]
    e = COS_H[:, k]
    sa = float(np.std(a))
    se = float(np.std(e))
    if sa > 1e-12 and se > 1e-12:
        per_pair_corr.append(float(np.corrcoef(
            a, e)[0, 1]))
    else:
        per_pair_corr.append(float('nan'))
mean_mass = MASS.mean(axis=0)
if float(np.std(mean_mass)) > 1e-12 \
        and float(np.std(med_h)) > 1e-12:
    corr_global = float(np.corrcoef(mean_mass,
                                    med_h)[0, 1])
else:
    corr_global = float('nan')
log('T4: per-pair corr med=%.4f | global '
    'corr(mean mass, med_h)=%.4f'
    % (float(np.nanmedian(per_pair_corr)),
       corr_global))
log('T4: group mass %s'
    % np.array2string(mean_mass, precision=4))

# ---------- FRONT all-layer repro (a146b) ----------
log('=== FRONT all-layer repro (a146b) ===')
COS_FR = np.zeros(NP_)
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    off = assembled[pref_i]['off']
    repK = KPpost[base_i, :, :nb, :].copy()
    repV = VP[base_i, :, :nb, :].copy()
    for j in front_rows(nb):
        src = KPpost[pref_i, :, off + j, :] \
            .reshape(NL, 8, HDIM)
        rot_full = rot_apply(src, off) \
            .reshape(NL, HDIM * 8)
        repK[:, j, :] = rot_full
        repV[:, j, :] = VP[pref_i, :, off + j, :]
    mask = np.ones(nb, dtype=bool)
    res = forward_run(ids_b, replK=repK,
                      maskK=mask, replV=repV,
                      maskV=mask)
    r = res['lg'] - LG[base_i]
    t = t_targets[(b, c)]
    cs, _ = cos_frac(LG[base_i], r, t)
    COS_FR[k] = cs
    rec_dlg.append(float(np.linalg.norm(r)))
a146b_diff = float(np.max(np.abs(
    COS_FR - COS_FR50)))
a146b_ok = bool(a146b_diff == 0.0)
log('a146b FRONT repro vs z50 COS_FR: diff=%.3e '
    'ok=%s' % (a146b_diff, a146b_ok))

# ---------- T5 null restricted to L35 x FRONT ----------
log('=== T5 null L35 x FRONT (R=%d) ===' % R_NULL)
null35 = np.zeros(R_NULL)
for mc in range(R_NULL):
    rng = np.random.default_rng(SEED_NULL + mc)
    csm = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        repK, repV, rotK, srcV, rows, base_i \
            = l35_fields(k)
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        rjK = repK.copy()
        rjV = repV.copy()
        rjK[L_TGT, rows, :] = rotK[L_TGT] \
            .reshape(len(rows), HDIM * KV_HEAD)
        rjV[L_TGT, rows, :] = srcV[L_TGT]
        NK = np.linalg.norm(
            rjK[L_TGT, rows, :], axis=1)
        NV_ = np.linalg.norm(
            rjV[L_TGT, rows, :], axis=1)
        RK = rng.standard_normal(
            (len(rows), HDIM * KV_HEAD)) \
            .astype(np.float32)
        RK *= (NK / np.linalg.norm(RK, axis=1)
               )[:, None]
        RV = rng.standard_normal(
            (len(rows), HDIM * KV_HEAD)) \
            .astype(np.float32)
        RV *= (NV_ / np.linalg.norm(RV, axis=1)
               )[:, None]
        fullK = rjK.copy()
        fullV = rjV.copy()
        fullK[L_TGT, rows, :] = RK
        fullV[L_TGT, rows, :] = RV
        mask = np.ones(nb, dtype=bool)
        res = forward_run(ids_b, replK=fullK,
                          maskK=mask, replV=fullV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        t = t_targets[(b, c)]
        cs, _ = cos_frac(LG[base_i], r, t)
        csm[k] = cs
        rec_dlg.append(float(np.linalg.norm(r)))
    null35[mc] = float(np.median(csm))
    if (mc + 1) % 25 == 0:
        log('  mc %d/%d (med=%.4f)'
            % (mc + 1, R_NULL, null35[mc]))
p_null = float((1 + int(np.sum(
    null35 >= med_j))) / (R_NULL + 1))
log('L35: obs=%.4f null med=%.4f max=%.4f '
    'p=%.5f'
    % (med_j, float(np.median(null35)),
       float(null35.max()), p_null))

# a148 / a149
a148_ok = bool(a148_fail == 0
               and a148_checked >= 1)
max_dlg = float(np.max(np.array(rec_dlg)))
a149_ok = bool(max_dlg >= T_GATE)
log('a148 integrity fails=%d checked=%d ok=%s | '
    'a149 max||dlg||=%.4f ok=%s'
    % (a148_fail, a148_checked, a148_ok,
       max_dlg, a149_ok))

# ---------- verdict ----------
sig = bool(p_null < 0.05)
if not sig:
    verdict = 'kv35_null_qwen'
elif med_h[h_star] >= 0.8 * med_j:
    verdict = 'kv35_headfocus_qwen'
else:
    verdict = 'kv35_mixed_qwen'
log('VERDICT: %s (p=%.5f h*=%d med_h*=%.4f '
    'med_j=%.4f med_k=%.4f med_v=%.4f)'
    % (verdict, p_null, h_star, med_h[h_star],
       med_j, med_k, med_v))

anchor_core_ok = bool(a116_ok and a145_ok
                      and a146a_ok and a146b_ok
                      and a147_ok and a148_ok
                      and a149_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_J=COS_J, FRAC_J=FRAC_J,
         COS_K=COS_K, FRAC_K=FRAC_K,
         COS_V=COS_V, FRAC_V=FRAC_V,
         COS_H=COS_H, COS_LOO=COS_LOO,
         MASS=MASS,
         null35=null35,
         COS_FR=COS_FR,
         a145_diff=np.float64(a145_diff),
         a146a_diff=np.float64(a146a_diff),
         a146b_diff=np.float64(a146b_diff),
         a147_diff=np.float64(a147_diff),
         a148_fail=np.int64(a148_fail),
         a148_checked=np.int64(a148_checked),
         a149_max_dlg=np.float64(max_dlg),
         a116_ok=np.bool_(a116_ok),
         a145_ok=np.bool_(a145_ok),
         a146a_ok=np.bool_(a146a_ok),
         a146b_ok=np.bool_(a146b_ok),
         a147_ok=np.bool_(a147_ok),
         a148_ok=np.bool_(a148_ok),
         a149_ok=np.bool_(a149_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_kvsplit': {'med_joint': med_j,
                   'med_k_only': med_k,
                   'med_v_only': med_v,
                   'frac_joint_med':
                       float(np.median(FRAC_J)),
                   'frac_k_med':
                       float(np.median(FRAC_K)),
                   'frac_v_med':
                       float(np.median(FRAC_V))},
    'T3_heads': {'med_per_head': med_h,
                 'med_loo': med_loo,
                 'best_head': h_star,
                 'med_at_best': med_h[h_star],
                 'loo_drop_at_best': loo_drop,
                 'dominance_ratio':
                     med_h[h_star] / med_j
                     if abs(med_j) > 1e-12
                     else float('nan')},
    'T4_attn_readout': {'per_pair_corr_med':
                            float(np.nanmedian(
                                per_pair_corr)),
                        'corr_global': corr_global,
                        'mean_group_mass':
                            [float(x) for x in
                             mean_mass]},
    'T5_null': {'R': R_NULL, 'seed': SEED_NULL,
                'obs': med_j,
                'med_null':
                    float(np.median(null35)),
                'max_null':
                    float(null35.max()),
                'p_null': p_null},
    'anchors': {'a116_seals_ok': a116_ok,
                'a145_recapture_diff': a145_diff,
                'a146a_joint35_diff': a146a_diff,
                'a146b_front_diff': a146b_diff,
                'a147_sham_diff': a147_diff,
                'a148_fail': a148_fail,
                'a148_checked': a148_checked,
                'a149_max_dlg': max_dlg,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run1 authoritative (fp32; capture '
                 'bank loaded from the phase3048 '
                 'npz; per-pair chain anchors vs '
                 'the phase3050 npz)',
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
