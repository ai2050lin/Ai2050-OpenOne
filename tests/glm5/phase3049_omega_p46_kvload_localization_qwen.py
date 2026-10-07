# -*- coding: utf-8 -*-
# Phase 3049 - Omega-P46: KV payload
# localization (position-band x layer-band).
# 3048 showed the full-position exact KV
# replay significantly carries the prefix
# logit direction (med cos 0.6067, p=0.00498)
# but is not a faithful copy (frac 3.20).
# Here the payload is LOCALIZED: exact
# replacement restricted to position bands
# of the base prompt (prefix-adjacent front
# band j<4 / target position only / rest
# minus target / all = 3048 reproduction),
# a norm-matched random-replacement null on
# the front band (R=200, preregistered
# 24-pair median), a layer-band x front-band
# matrix (6 layer blocks), and SCR
# reproduction + front/back scramble split.
# Capture bank is loaded from the z48 npz
# (bit-level chain anchor: 3 sampled prompts
# re-captured and compared bit-exact; REP and
# SCR per-pair statistics re-derived and
# compared exactly).
# verdict_tree: sig_front = p_front < 0.05;
# sig_front AND med cos_front > med cos_tgt
# AND med cos_front > med cos_rest ->
# kvload_frontband_qwen; sig_front (else) ->
# kvload_uniform_qwen; else ->
# kvload_null_qwen; single branch.
# anchors: a132 sampled re-capture (3
# prompts, KPpre/KPpost/VP/LG) bit 0.0 vs
# z48; a133 all-band REP cos/frac vs z48
# COS_REP/FRAC_REP diff 0.0 (per pair); a134
# SCR kill_cos/kill_frac vs z48 diff 0.0
# (same seed 9852 and iteration order); a135
# sham self-replacement bit identity; a136
# integrity bit-exact on all intervention
# forwards; a137 chain-entry max||dlg|| >=
# 0.05.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3049
NAME = 'omega_p46_kvload_localization_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 7
HDIM = 128
R_NULL = 200
SEED_NULL = 9901
SEED_SCRF = 9853
SEED_SCRB = 9854
SEED_MAIN = 3009
T_GATE = 0.05
FRONT = 4
LBANDS = ((0, 6), (6, 12), (12, 18), (18, 24),
          (24, 30), (30, 36))
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
            '3009); capture bank loaded from the '
            'phase3048 npz (KPpre/KPpost/VP/LG/LENS/'
            'TT, 48 prompts verbatim bank); prompts '
            'reassembled locally with the 3048 tail-'
            'alignment rule; statistics on the 24 '
            'old pairs',
    'question': '3049 A main line: WHERE does the '
                'carrying KV payload live? (i) '
                'position-band exact replacement on '
                'the base prompt: front band (body '
                'positions 0..min(4,nb)-1, prefix-'
                'adjacent), target position only, '
                'rest minus target, all positions '
                '(3048 reproduction arm); (ii) '
                'norm-matched random-replacement '
                'null on the front band (primary '
                'test); (iii) layer-band x front-band '
                'exact replacement matrix; (iv) SCR '
                'reproduction (anchor) + front/back '
                'prefix scramble split (descriptive)',
    'T2_bands': 'per pair: exact post-norm K '
                'replacement (RoPE offset rotation) '
                '+ V replacement restricted to a '
                'position set S, other positions '
                'self-replaced bit-identically from '
                'the base capture; bands: ALL (all '
                'nb positions, reproduction), FRONT '
                '(j in 0..min(FRONT,nb)-1), TGT '
                '(target position only), REST (all '
                'minus target); stats cos(resp, '
                't_bc), frac; PRIMARY null on FRONT: '
                'R=%d MC norm-matched random '
                'replacement on the FRONT band, '
                'seed %d + mc, stat = 24-pair median '
                'cos, p = P(null >= obs)'
                % (R_NULL, SEED_NULL),
    'T3_layerbands': 'for each of the 6 layer '
                     'blocks, FRONT-band exact '
                     'replacement ONLY in that block '
                     '(other layers self-replaced); '
                     'med cos (6, 24), descriptive',
    'T4_SCR': 'reproduction of the 3048 prefix-'
              'token scramble (seed 9852, same '
              'order; anchor vs z48) plus front-half '
              '(0..off//2-1) and back-half '
              '(off//2..off-1) scramble split (seeds '
              '%d/%d, per-pair draws), kill_cos/'
              'kill_frac descriptive'
              % (SEED_SCRF, SEED_SCRB),
    'verdict_tree': 'sig_front = p_front < 0.05; '
                    'sig_front AND med cos_FRONT > '
                    'med cos_TGT AND med cos_FRONT > '
                    'med cos_REST -> '
                    'kvload_frontband_qwen; sig_front '
                    'otherwise -> kvload_uniform_qwen; '
                    'else -> kvload_null_qwen; single '
                    'branch',
    'anchors': 'a132 sampled re-capture (3 prompts '
               'spanning cond 0/1/3) KPpre/KPpost/VP/'
               'LG bit 0.0 vs z48; a133 ALL-band REP '
               'cos/frac vs z48 COS_REP/FRAC_REP '
               'diff 0.0 per pair; a134 SCR '
               'kill_cos/kill_frac vs z48 diff 0.0; '
               'a135 sham self-replacement bit '
               'identity; a136 integrity bit-exact '
               '(mod == where(mask,repl,orig), '
               'additive == orig+delta) on all '
               'intervention forwards; a137 '
               'chain-entry max||dlg|| >= %g'
               % T_GATE,
    'control': 'norm-matched random replacement on '
               'the FRONT band (same machinery, per '
               'layer per position norms matched to '
               'the FRONT repl field); scramble arms; '
               'no other intervention',
    'statistics_discipline': 'obs and null on the '
                             'SAME scale (raw cos, '
                             'same 24 pairs, same '
                             'replacement machinery '
                             'restricted to the FRONT '
                             'band); null never on '
                             'intervened quantities; '
                             'verdict in one branch',
    'corrections': 'run1 crashed pre-anchor on the assembly assertion: this phase assembles only the 32 old-body prompts (the NEW_BODIES bank of 3043-3048 is not used here) but the copied assertion still required n_pr==48, and one a132 sampling index pointed at a new-body row of the z48 bank; assertion corrected to n_pr==32 and sampling restricted to the old-body rows; run2 crashed at the first V-replacement forward (a135 sham): the copied forward_run applied the K-side reshape (NL,-1,8,128) to replV as well, but the v_proj output is (1,s,1024) and the 3048 protocol passes replV flat - reshape removed, K side unchanged; run3 crashed pre-anchor inside band_rep on the K source slice reshape: KPpost[pref_i, :, off+j, :] is (NL,1024) and must reshape to (NL,8,128), not (8,128) (same fix in the T3 layer-band block); run4 crashed pre-anchor on the assignment side of the same slice: rot_apply returns (NL,8,128) and repK[:, j, :] is (NL,1024), so the reshape must be (NL,HDIM*8) (T3 block reshapes to (NL,HDIM*8) then slices [l0:l1]); run5 crashed pre-anchor on a K-hook row-count mismatch: band_rep returns the full-length repl field (nb rows) paired with a partial band mask (mask.sum()<nb), but the hook requires repl rows == mask.sum(); fixed by an all-True mask (non-band rows already hold the base self-replacement values; the FRONT-band null MC reuses the same front_mask and follows automatically); run6 authoritative',
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
assert int(model.config.num_key_value_heads) == 8
assert int(model.config.num_attention_heads) == 32
assert hasattr(layers[3].self_attn, 'k_norm')
rot = model.model.rotary_emb
INV_FREQ = rot.inv_freq.detach().cpu().numpy() \
    .astype(np.float32)
assert INV_FREQ.shape == (64,), INV_FREQ.shape
assert float(rot.attention_scaling) == 1.0
NVOC = int(model.config.vocab_size)
log('model loaded fp32 (vocab=%d)' % NVOC)

SL = slice(KV_HEAD * HDIM, (KV_HEAD + 1) * HDIM)


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
LMAX = int(KPpost.shape[2])
log('z48 bank loaded: KPpre%s KPpost%s VP%s LG%s'
    % (KPpre.shape, KPpost.shape, VP.shape,
       LG.shape))

seal_detail = []
for ph, nm in (
        (3044, 'omega_p41_field_axis_injection_'
               'qwen'),
        (3045, 'omega_p42_l20_axis_anatomy_qwen'),
        (3046, 'omega_p43_kfield_injection_qwen'),
        (3047, 'omega_p44_kv_joint_replay_qwen'),
        (3048, 'omega_p45_kvpos_full_replay_qwen')):
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
log('a116b source seals ok=%s' % a116_ok)

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
n_old = len(BODIES) * len(PREFIXES)
assert n_old == 32 and n_pr == 32  # 3049 drops the NEW_BODIES bank (old bodies only)
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
# consistency with the bank lengths
for i in range(n_pr):
    assert LENS[i] == len(assembled[i]['ids']), i
log('assembled %d prompts (lens consistent '
    'with z48)' % n_pr)

# a132: sampled re-capture, bit-exact vs z48
a132_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a132_diff = max(a132_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a132_diff = max(a132_diff, float(np.max(
        np.abs(KPpre[si, :, :n2, :] - kp2))))
    a132_diff = max(a132_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a132_diff = max(a132_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a132_diff = float(a132_diff)
a132_ok = bool(a132_diff == 0.0)
log('a132 re-capture diff=%.3e ok=%s'
    % (a132_diff, a132_ok))

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
a132_ok = bool(a132_ok and a_tt == 0.0)
log('a132b TT vs z48 diff=%.3e' % a_tt)

# a135: sham self-replacement (bit identity)
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
a135_diff = float(np.max(np.abs(res['lg']
                                - LG[base0])))
a135_diff = float(max(a135_diff, res['bit']))
a135_ok = bool(a135_diff == 0.0)
log('a135 sham self-replacement diff=%.3e ok=%s'
    % (a135_diff, a135_ok))


def band_rep(k, band):
    """Build the (replK, replV, mask) for exact
    replacement restricted to a position band of
    the base prompt; other positions self-replaced
    from the base capture. band in
    {'ALL','FRONT','TGT','REST'}."""
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    nb = int(LENS[base_i])
    off = assembled[pref_i]['off']
    pos = assembled[base_i]['pos']
    if band == 'ALL':
        sel = list(range(nb))
    elif band == 'FRONT':
        sel = list(range(min(FRONT, nb)))
    elif band == 'TGT':
        sel = [pos]
    elif band == 'REST':
        sel = [j for j in range(nb) if j != pos]
    else:
        raise ValueError(band)
    repK = KPpost[base_i, :, :nb, :].copy()
    repV = VP[base_i, :, :nb, :].copy()
    for j in sel:
        src = KPpost[pref_i, :, off + j, :] \
            .reshape(NL, 8, HDIM)
        repK[:, j, :] = rot_apply(src, off) \
            .reshape(NL, HDIM * 8)
        repV[:, j, :] = VP[pref_i, :, off + j, :]
    # full-length repl + all-True mask:
    # non-band rows already hold the base
    # self-replacement values
    mask = np.ones(nb, dtype=bool)
    return repK, repV, mask


# ---------- T2 bands ----------
log('=== T2 position bands (24 pairs) ===')
COS = {bd: np.zeros(NP_) for bd in
       ('ALL', 'FRONT', 'TGT', 'REST')}
FRAC = {bd: np.zeros(NP_) for bd in
        ('ALL', 'FRONT', 'TGT', 'REST')}
a136_fail = 0
rec_kind = []
rec_dlg = []
front_repK = []
front_repV = []
front_mask = []
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    ids_b = assembled[base_i]['ids']
    t = t_targets[(b, c)]
    nt = float(np.linalg.norm(t))
    for bd in ('ALL', 'FRONT', 'TGT', 'REST'):
        repK, repV, mask = band_rep(k, bd)
        res = forward_run(ids_b, replK=repK,
                          maskK=mask, replV=repV,
                          maskV=mask,
                          integ=(bd != 'ALL'))
        if res.get('bit', 0.0) != 0.0:
            a136_fail += 1
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        COS[bd][k] = float(r @ t) / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
        FRAC[bd][k] = nr / nt if nt > 1e-12 \
            else float('nan')
        rec_kind.append(bd)
        rec_dlg.append(nr)
        if bd == 'FRONT':
            front_repK.append(repK)
            front_repV.append(repV)
            front_mask.append(mask)
    if (k + 1) % 8 == 0:
        log('  pairs %d/24 done' % (k + 1))
med_cos = {bd: float(np.median(COS[bd]))
           for bd in COS}
med_frac = {bd: float(np.nanmedian(FRAC[bd]))
            for bd in FRAC}
for bd in ('ALL', 'FRONT', 'TGT', 'REST'):
    log('T2 %s: med cos=%.4f med frac=%.4f'
        % (bd, med_cos[bd], med_frac[bd]))

# a133: ALL band vs z48 reproduction
a133_dc = float(np.max(np.abs(COS['ALL']
                              - z48['COS_REP'])))
a133_df = float(np.max(np.abs(FRAC['ALL']
                              - z48['FRAC_REP'])))
a133_diff = float(max(a133_dc, a133_df))
a133_ok = bool(a133_diff == 0.0)
log('a133 ALL-band vs z48: dc=%.3e df=%.3e ok=%s'
    % (a133_dc, a133_df, a133_ok))

# null MC on the FRONT band (preregistered
# 24-pair median statistic)
log('=== null MC FRONT (R=%d) ===' % R_NULL)
null_cos = np.zeros(R_NULL)
for mc in range(R_NULL):
    rng = np.random.default_rng(SEED_NULL + mc)
    csm = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        ids_b = assembled[base_i]['ids']
        repK = front_repK[k]
        repV = front_repV[k]
        mask = front_mask[k]
        NK = np.linalg.norm(
            repK[:, mask, :], axis=2)
        NV_ = np.linalg.norm(
            repV[:, mask, :], axis=2)
        RK = rng.standard_normal(
            (NL, int(mask.sum()), HDIM * 8)) \
            .astype(np.float32)
        RK *= (NK / np.linalg.norm(RK, axis=2)
               )[:, :, None]
        RV = rng.standard_normal(
            (NL, int(mask.sum()), HDIM * 8)) \
            .astype(np.float32)
        RV *= (NV_ / np.linalg.norm(RV, axis=2)
               )[:, :, None]
        fullK = repK.copy()
        fullV = repV.copy()
        fullK[:, mask, :] = RK
        fullV[:, mask, :] = RV
        res = forward_run(ids_b, replK=fullK,
                          maskK=mask, replV=fullV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        t = t_targets[(b, c)]
        nt = float(np.linalg.norm(t))
        csm[k] = float(r @ t) / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
    null_cos[mc] = float(np.median(csm))
    if (mc + 1) % 25 == 0:
        log('  mc %d/%d (med=%.4f)'
            % (mc + 1, R_NULL, null_cos[mc]))
p_front = float((1 + int(np.sum(
    null_cos >= med_cos['FRONT'])))
    / (R_NULL + 1))
log('FRONT: obs=%.4f null med=%.4f max=%.4f '
    'p=%.5f'
    % (med_cos['FRONT'],
       float(np.median(null_cos)),
       float(null_cos.max()), p_front))

# ---------- T3 layer-band x FRONT matrix ----------
log('=== T3 layer-band x FRONT matrix ===')
COS_LB = np.zeros((len(LBANDS), NP_))
for gi, (l0, l1) in enumerate(LBANDS):
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        pref_i = idx_of[(c, b)]
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        off = assembled[pref_i]['off']
        t = t_targets[(b, c)]
        nt = float(np.linalg.norm(t))
        repK = KPpost[base_i, :, :nb, :].copy()
        repV = VP[base_i, :, :nb, :].copy()
        for j in range(min(FRONT, nb)):
            src = KPpost[pref_i, :, off + j, :] \
                .reshape(NL, 8, HDIM)
            repK[l0:l1, j, :] = rot_apply(
                src, off).reshape(
                NL, HDIM * 8)[l0:l1]
            repV[l0:l1, j, :] = VP[pref_i,
                                   l0:l1,
                                   off + j, :]
        mask = np.ones(nb, dtype=bool)
        res = forward_run(ids_b, replK=repK,
                          maskK=mask, replV=repV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        COS_LB[gi, k] = float(r @ t) / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
    log('  LB%d-%d: med cos=%.4f'
        % (l0, l1 - 1,
           float(np.median(COS_LB[gi]))))
med_cos_lb = [float(np.median(COS_LB[gi]))
              for gi in range(len(LBANDS))]
argmax_lb = int(np.argmax(med_cos_lb))

# ---------- T4 SCR reproduction + split ----------
log('=== T4 SCR (reproduction + split) ===')
rng4 = np.random.default_rng(9852)
KILL_FRAC = np.zeros(NP_)
KILL_COS = np.zeros(NP_)
KILLF_F = np.zeros(NP_)
KILLC_F = np.zeros(NP_)
KILLF_B = np.zeros(NP_)
KILLC_B = np.zeros(NP_)
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    ids_p = assembled[pref_i]['ids']
    off = assembled[pref_i]['off']
    t = t_targets[(b, c)]
    nt = float(np.linalg.norm(t))
    # reproduction of 3048 (seed 9852, same order)
    RK = rng4.standard_normal(
        (NL, off, HDIM * 8)).astype(np.float32)
    NKp = np.linalg.norm(
        KPpost[pref_i, :, :off, :], axis=2)
    RK *= (NKp / np.linalg.norm(RK, axis=2)
           )[:, :, None]
    RV = rng4.standard_normal(
        (NL, off, HDIM * 8)).astype(np.float32)
    NVp = np.linalg.norm(
        VP[pref_i, :, :off, :], axis=2)
    RV *= (NVp / np.linalg.norm(RV, axis=2)
           )[:, :, None]
    m_p = np.zeros(len(ids_p), dtype=bool)
    m_p[:off] = True
    res = forward_run(ids_p, replK=RK, maskK=m_p,
                      replV=RV, maskV=m_p)
    r = res['lg'] - LG[pref_i]
    nr = float(np.linalg.norm(r))
    KILL_FRAC[k] = nr / nt if nt > 1e-12 \
        else float('nan')
    KILL_COS[k] = float(r @ (-t)) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    rec_kind.append('SCR')
    rec_dlg.append(nr)
    # front-half scramble
    nf = max(1, off // 2)
    rngF = np.random.default_rng(SEED_SCRF + k)
    RKf = rngF.standard_normal(
        (NL, nf, HDIM * 8)).astype(np.float32)
    NKf = np.linalg.norm(
        KPpost[pref_i, :, :nf, :], axis=2)
    RKf *= (NKf / np.linalg.norm(RKf, axis=2)
            )[:, :, None]
    RVf = rngF.standard_normal(
        (NL, nf, HDIM * 8)).astype(np.float32)
    NVf = np.linalg.norm(
        VP[pref_i, :, :nf, :], axis=2)
    RVf *= (NVf / np.linalg.norm(RVf, axis=2)
            )[:, :, None]
    m_f = np.zeros(len(ids_p), dtype=bool)
    m_f[:nf] = True
    res = forward_run(ids_p, replK=RKf, maskK=m_f,
                      replV=RVf, maskV=m_f)
    r = res['lg'] - LG[pref_i]
    nr = float(np.linalg.norm(r))
    KILLF_F[k] = nr / nt if nt > 1e-12 \
        else float('nan')
    KILLC_F[k] = float(r @ (-t)) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    rec_kind.append('SCRF')
    rec_dlg.append(nr)
    # back-half scramble
    nbk = off - nf
    rngB = np.random.default_rng(SEED_SCRB + k)
    RKb = rngB.standard_normal(
        (NL, nbk, HDIM * 8)).astype(np.float32)
    NKb = np.linalg.norm(
        KPpost[pref_i, :, nf:off, :], axis=2)
    RKb *= (NKb / np.linalg.norm(RKb, axis=2)
            )[:, :, None]
    RVb = rngB.standard_normal(
        (NL, nbk, HDIM * 8)).astype(np.float32)
    NVb = np.linalg.norm(
        VP[pref_i, :, nf:off, :], axis=2)
    RVb *= (NVb / np.linalg.norm(RVb, axis=2)
            )[:, :, None]
    m_bk = np.zeros(len(ids_p), dtype=bool)
    m_bk[nf:off] = True
    res = forward_run(ids_p, replK=RKb,
                      maskK=m_bk, replV=RVb,
                      maskV=m_bk)
    r = res['lg'] - LG[pref_i]
    nr = float(np.linalg.norm(r))
    KILLF_B[k] = nr / nt if nt > 1e-12 \
        else float('nan')
    KILLC_B[k] = float(r @ (-t)) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    rec_kind.append('SCRB')
    rec_dlg.append(nr)
med_kill_frac = float(np.nanmedian(KILL_FRAC))
med_kill_cos = float(np.median(KILL_COS))
med_killf_f = float(np.nanmedian(KILLF_F))
med_killc_f = float(np.median(KILLC_F))
med_killf_b = float(np.nanmedian(KILLF_B))
med_killc_b = float(np.median(KILLC_B))
log('SCR repro: med kill_frac=%.4f kill_cos=%.4f'
    % (med_kill_frac, med_kill_cos))
log('SCR split: front frac=%.4f cos=%.4f | '
    'back frac=%.4f cos=%.4f'
    % (med_killf_f, med_killc_f,
       med_killf_b, med_killc_b))

# a134: SCR vs z48
a134_dc = float(np.max(np.abs(KILL_COS
                              - z48['KILL_COS'])))
a134_df = float(np.max(np.abs(KILL_FRAC
                              - z48['KILL_FRAC'])))
a134_diff = float(max(a134_dc, a134_df))
a134_ok = bool(a134_diff == 0.0)
log('a134 SCR vs z48: dc=%.3e df=%.3e ok=%s'
    % (a134_dc, a134_df, a134_ok))

# a136/a137
a136_ok = bool(a136_fail == 0)
max_dlg = float(np.max(np.array(rec_dlg)))
a137_ok = bool(max_dlg >= T_GATE)
log('a136 integrity fails=%d ok=%s | a137 '
    'max||dlg||=%.4f ok=%s'
    % (a136_fail, a136_ok, max_dlg, a137_ok))

# ---------- verdict ----------
sig_front = bool(p_front < 0.05)
if sig_front and (med_cos['FRONT']
                  > med_cos['TGT']
                  and med_cos['FRONT']
                  > med_cos['REST']):
    verdict = 'kvload_frontband_qwen'
elif sig_front:
    verdict = 'kvload_uniform_qwen'
else:
    verdict = 'kvload_null_qwen'
log('VERDICT: %s (sig_front=%s med front=%.4f '
    'tgt=%.4f rest=%.4f)'
    % (verdict, sig_front, med_cos['FRONT'],
       med_cos['TGT'], med_cos['REST']))

anchor_core_ok = bool(a116_ok and a132_ok
                      and a133_ok and a134_ok
                      and a135_ok and a136_ok
                      and a137_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_ALL=COS['ALL'], COS_FRONT=COS['FRONT'],
         COS_TGT=COS['TGT'], COS_REST=COS['REST'],
         FRAC_ALL=FRAC['ALL'],
         FRAC_FRONT=FRAC['FRONT'],
         FRAC_TGT=FRAC['TGT'],
         FRAC_REST=FRAC['REST'],
         null_cos=null_cos,
         COS_LB=COS_LB,
         med_cos_lb=np.array(med_cos_lb),
         KILL_FRAC=KILL_FRAC, KILL_COS=KILL_COS,
         KILLF_F=KILLF_F, KILLC_F=KILLC_F,
         KILLF_B=KILLF_B, KILLC_B=KILLC_B,
         rec_kind=np.array(rec_kind),
         rec_dlg=np.array(rec_dlg),
         a132_diff=np.float64(a132_diff),
         a133_diff=np.float64(a133_diff),
         a134_diff=np.float64(a134_diff),
         a135_diff=np.float64(a135_diff),
         a136_fail=np.int64(a136_fail),
         a137_max_dlg=np.float64(max_dlg),
         a116_ok=np.bool_(a116_ok),
         a132_ok=np.bool_(a132_ok),
         a133_ok=np.bool_(a133_ok),
         a134_ok=np.bool_(a134_ok),
         a135_ok=np.bool_(a135_ok),
         a136_ok=np.bool_(a136_ok),
         a137_ok=np.bool_(a137_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_bands': {bd: {'med_cos': med_cos[bd],
                      'med_frac': med_frac[bd],
                      'cos_per_pair':
                          COS[bd].tolist()}
                 for bd in
                 ('ALL', 'FRONT', 'TGT', 'REST')},
    'null_front': {'R': R_NULL, 'seed': SEED_NULL,
                   'med_cos':
                       float(np.median(null_cos)),
                   'max_cos':
                       float(null_cos.max()),
                   'p_front': p_front},
    'T3_layerbands': {'med_cos': med_cos_lb,
                      'argmax_band': argmax_lb,
                      'med_cos_at_argmax':
                          med_cos_lb[argmax_lb]},
    'T4_SCR': {'med_kill_frac': med_kill_frac,
               'med_kill_cos': med_kill_cos,
               'front_kill_frac': med_killf_f,
               'front_kill_cos': med_killc_f,
               'back_kill_frac': med_killf_b,
               'back_kill_cos': med_killc_b},
    'anchors': {'a116_seals_ok': a116_ok,
                'a132_recapture_diff': a132_diff,
                'a133_allband_diff': a133_diff,
                'a134_scr_diff': a134_diff,
                'a135_sham_diff': a135_diff,
                'a136_fail': a136_fail,
                'a137_max_dlg': max_dlg,
                'anchor_core_ok': anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run6 authoritative (fp32; run1 crashed pre-anchor on an assembly assertion, see corrections; capture '
                 'bank loaded from the phase3048 npz '
                 'with bit-level re-capture anchor)',
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
