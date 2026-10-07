# -*- coding: utf-8 -*-
# Phase 3050 - Omega-P47: KV deep-band (LB30-35)
# causal dissection. 3049 localized the carrying
# KV payload: body-wide over positions, deep-layer
# concentrated (LB30-35 alone med cos 0.695 vs
# 0.11-0.28 elsewhere). Here the deep band is
# dissected: (i) single-layer exact replacement
# profile (each of the 36 layers individually on
# the FRONT band, 24 pairs); (ii) suffix cumulative
# bands L(l..35), l=28..35 (the l=30 band is the
# 3049 LB30-35 reproduction arm, chain anchor);
# (iii) PRIMARY null: norm-matched random
# replacement restricted to layers 30..35 x FRONT
# positions (other rows self-replaced), R=200,
# 24-pair median, p = P(null >= obs); (iv) FRONT-
# band restricted null (retro-validation of the
# 3049 p_front, whose all-True-mask fix actually
# randomized ALL positions); (v) logit-lens
# alignment: per-layer lens direction
# (lens[pref] - lens[base]) cos with t_bc, final
# layer bit-anchored vs LG. Capture bank loaded
# from the z48 npz; per-pair chain anchors vs the
# z49 npz (COS_FRONT, COS_LB[5]).
# verdict_tree: sig_deep = p_deep < 0.05;
# sig_deep AND argmax single-layer med cos >= 30
# -> kvdeep_local_qwen; sig_deep otherwise ->
# kvdeep_broad_qwen; else -> kvdeep_null_qwen;
# single branch.
# anchors: a139 sampled re-capture (4 prompts)
# bit 0.0 vs z48; a140a FRONT repro per-pair cos
# vs z49 COS_FRONT diff 0.0; a140b L30-35 repro
# per-pair cos vs z49 COS_LB[5] diff 0.0; a141
# sham self-replacement bit identity; a142
# integrity bit-exact on a deterministic subset
# (every 20th intervention forward + sham + L30-35
# repro band); a143 lens final layer vs LG bit
# 0.0; a144 chain-entry max||dlg|| >= 0.05.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3050
NAME = 'omega_p47_kvdeep_dissection_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 7
HDIM = 128
R_NULL = 200
SEED_NULL_D = 9911
SEED_NULL_F = 9921
SEED_MAIN = 3009
T_GATE = 0.05
FRONT = 4
DEEP0 = 30
SUFFIX = (28, 29, 30, 31, 32, 33, 34, 35)
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
            'TT, 32 old-body prompts); per-pair '
            'chain anchors vs the phase3049 npz '
            '(COS_FRONT, COS_LB); prompts '
            'reassembled locally with the 3048 '
            'tail-alignment rule; statistics on the '
            '24 old pairs',
    'question': '3050 A main line: WHERE inside the '
                'deep band does the carrying KV '
                'payload live, and does it align '
                'with the logit-lens readout? (i) '
                'single-layer exact replacement '
                'profile (36 layers individually on '
                'the FRONT band); (ii) suffix '
                'cumulative bands L(l..35) l=28..35 '
                '(l=30 = 3049 LB30-35 reproduction); '
                '(iii) PRIMARY null: norm-matched '
                'random replacement RESTRICTED to '
                'layers 30..35 x FRONT positions, '
                'other rows self-replaced (R=200, '
                'seed 9911+mc, 24-pair median, '
                'p = P(null >= obs)); (iv) FRONT-'
                'band restricted null (seed 9921+mc) '
                'retro-validating the 3049 p_front '
                'whose all-True-mask null randomized '
                'ALL positions; (v) logit-lens '
                'direction profile (descriptive)',
    'T2_single_layer': 'per pair: exact post-norm K '
                       'replacement (RoPE offset '
                       'rotation) + V replacement at '
                       'the FRONT band (j in '
                       '0..min(4,nb)-1) in ONE layer '
                       'l only (other layers self-'
                       'replaced from the base '
                       'capture); med cos over 24 '
                       'pairs per layer (36), '
                       'descriptive; peak layer l* = '
                       'argmax',
    'T3_suffix': 'suffix cumulative bands L(l..35) '
                 'for l in 28..35, FRONT band, '
                 'exact replacement; med cos (8); '
                 'the l=30 band must reproduce the '
                 '3049 LB30-35 per-pair cos bit-'
                 'exactly (a140b); onset = min l '
                 'with med >= 0.8 * med(l=30)',
    'T4a_null_deep': 'PRIMARY: R=%d MC, seed %d + '
                     'mc; per draw, rows 30..35 of '
                     'the FRONT positions replaced '
                     'by norm-matched gaussian '
                     'random (per layer per '
                     'position norms matched to '
                     'the L30-35 repl field), all '
                     'other rows keep the exact '
                     'repl field (self-replaced '
                     'non-band rows preserved); '
                     'stat = 24-pair median cos, '
                     'p = P(null >= obs), obs = '
                     'med cos of the L30-35 exact '
                     'reproduction'
                     % (R_NULL, SEED_NULL_D),
    'T4b_null_front': 'retro-validation of 3049: '
                      'R=%d MC, seed %d + mc; rows '
                      '0..35 of the FRONT positions '
                      'randomized (norm-matched), '
                      'other rows self-replaced; '
                      'obs = med cos FRONT (must '
                      'reproduce z49 COS_FRONT '
                      'bit-exactly, a140a); p = '
                      'P(null >= obs)'
                      % (R_NULL, SEED_NULL_F),
    'T5_lens': 'logit-lens capture for all 32 '
               'prompts: decoder-layer output -> '
               'model.model.norm -> lm_head, per '
               'layer (36); anchor: lens layer 35 '
               'vs LG bit 0.0 (a143); per pair '
               'direction d_l = lens[pref,l] - '
               'lens[base,l], cos(d_l, t_bc) '
               'per layer; med profile over 24 '
               'pairs, argmax and onset (min l '
               'with med >= 0.8) descriptive; '
               'alignment with l* and the suffix '
               'onset reported',
    'verdict_tree': 'sig_deep = p_deep < 0.05; '
                    'sig_deep AND argmax single-'
                    'layer med cos >= 30 -> '
                    'kvdeep_local_qwen; sig_deep '
                    'otherwise -> kvdeep_broad_'
                    'qwen; else -> kvdeep_null_'
                    'qwen; single branch',
    'anchors': 'a139 sampled re-capture (4 prompts '
               '0/9/17/31) KPpre/KPpost/VP/LG bit '
               '0.0 vs z48; a140a FRONT repro '
               'per-pair cos vs z49 COS_FRONT '
               'diff 0.0; a140b L30-35 repro '
               'per-pair cos vs z49 COS_LB[5] '
               'diff 0.0; a141 sham self-'
               'replacement bit identity; a142 '
               'integrity bit-exact (mod == '
               'where(mask,repl,orig)) on the '
               'full L30-35 repro band (24 '
               'forwards, integ=True) plus the '
               'sham; a143 lens '
               'final layer vs LG bit 0.0; a144 '
               'chain-entry max||dlg|| >= %g'
               % T_GATE,
    'control': 'norm-matched random replacement '
               'restricted to the same rows as the '
               'exact field (deep band x FRONT for '
               'T4a, all layers x FRONT for T4b); '
               'no other intervention',
    'statistics_discipline': 'obs and null on the '
                             'SAME scale (raw cos, '
                             'same 24 pairs, same '
                             'replacement machinery, '
                             'null randomized only '
                             'inside the band rows); '
                             'null never on intervened '
                             'quantities; verdict in '
                             'one branch',
    'corrections': 'run1 crashed pre-anchor at '
                   'the T5 lens capture: LENSLOG '
                   'was declared (n_pr, NL) while '
                   'forward_lens returned (NL, n, '
                   'NVOC), and full per-layer lens '
                   'storage is infeasible anyway '
                   '(~1.4TB for 32 prompts); '
                   'forward_lens now keeps last-'
                   'position logits only (NL, '
                   'NVOC), a143 compares per '
                   'prompt and discards, and the '
                   'COS_LENS loop re-captures '
                   'base/pref per pair on the fly; '
                   'run2 crashed pre-anchor inside '
                   'forward_lens: the Qwen3 decoder '
                   'layer returns a bare tensor so '
                   'the hook out[0] already stripped '
                   'the batch dim and the extra [0] '
                   'left a 1-D hidden (lg 1-D -> '
                   'IndexError); dim guard added '
                   '(hH.dim()==3 -> hH[0]); '
                   'run3 COMPLETED BUT FAILED the '
                   'a143 anchor (lens final layer '
                   'vs LG diff 22.4, lens profile '
                   'degenerate): lg[0,-1] on the '
                   '2-D lm_head output (n,NVOC) is '
                   'a SCALAR (token 0, last vocab '
                   'entry), not the last-token '
                   'vector - lens rows were '
                   'constant vectors; fixed to '
                   'lg[-1] (a full-matrix probe '
                   'lm_head(norm(cap35)) vs logits '
                   'diff 0.0 at every position '
                   'proves the capture+composition '
                   'is otherwise exact); run4 '
                   'authoritative',
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
capH = {li: {'rec': False, 'orig': None}
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


def hook_h(cp):
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
    layers[li].register_forward_hook(hook_h(
        capH[li]))


def reset_all():
    for li in range(NL):
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKn[li]['rec'] = False
        capV[li]['rec'] = False
        capKpre[li]['rec'] = False
        capH[li]['rec'] = False


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


def forward_lens(ids):
    """Per-layer logit-lens logits (final norm +
    lm_head applied to each decoder-layer output)."""
    reset_all()
    for li in range(NL):
        capH[li]['rec'] = True
    with torch.no_grad():
        model(torch.tensor([ids], device='cuda'),
              use_cache=True)
    lens = np.zeros((NL, NVOC), dtype=np.float64)
    with torch.no_grad():
        for li in range(NL):
            hH = capH[li]['orig']
            if hH.dim() == 3:
                hH = hH[0]
            ln = model.model.norm(hH)
            lg = model.lm_head(ln)
            lens[li] = lg[-1].detach() \
                .double().cpu().numpy()
    reset_all()
    return lens


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
log('z48 bank loaded: KPpre%s KPpost%s VP%s LG%s'
    % (KPpre.shape, KPpost.shape, VP.shape,
       LG.shape))
z49 = np.load(os.path.join(
    BASE, 'phase3049',
    'omega_p46_kvload_localization_qwen',
    'omega_p46_kvload_localization_qwen.npz'),
    allow_pickle=True)
COS_LB49 = z49['COS_LB']
COS_FRONT49 = z49['COS_FRONT']
log('z49 anchors loaded: COS_LB%s COS_FRONT%s'
    % (COS_LB49.shape, COS_FRONT49.shape))

seal_detail = []
for ph, nm in (
        (3044, 'omega_p41_field_axis_injection_'
               'qwen'),
        (3045, 'omega_p42_l20_axis_anatomy_qwen'),
        (3046, 'omega_p43_kfield_injection_qwen'),
        (3047, 'omega_p44_kv_joint_replay_qwen'),
        (3048, 'omega_p45_kvpos_full_replay_qwen'),
        (3049, 'omega_p46_kvload_localization_'
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

# a139: sampled re-capture, bit-exact vs z48
a139_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a139_diff = max(a139_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a139_diff = max(a139_diff, float(np.max(
        np.abs(KPpre[si, :, :n2, :] - kp2))))
    a139_diff = max(a139_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a139_diff = max(a139_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a139_diff = float(a139_diff)
a139_ok = bool(a139_diff == 0.0)
log('a139 re-capture diff=%.3e ok=%s'
    % (a139_diff, a139_ok))

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
a139_ok = bool(a139_ok and a_tt == 0.0)
log('a139b TT vs z48 diff=%.3e' % a_tt)

# a141: sham self-replacement (bit identity)
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
a141_diff = float(np.max(np.abs(res['lg']
                                - LG[base0])))
a141_diff = float(max(a141_diff, res['bit']))
a141_ok = bool(a141_diff == 0.0)
log('a141 sham self-replacement diff=%.3e ok=%s'
    % (a141_diff, a141_ok))


def front_rows(nb):
    return list(range(min(FRONT, nb)))


def build_repl(k, layers_repl):
    """Exact FRONT-band replacement restricted to
    layers_repl (list of layer indices); all other
    (layer, position) entries hold the base
    self-replacement values. Returns full-length
    (NL, nb, 1024) replK/replV (all-True mask)."""
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    nb = int(LENS[base_i])
    off = assembled[pref_i]['off']
    repK = KPpost[base_i, :, :nb, :].copy()
    repV = VP[base_i, :, :nb, :].copy()
    if layers_repl:
        lmin = min(layers_repl)
        lmax = max(layers_repl) + 1
        for j in front_rows(nb):
            src = KPpost[pref_i, :, off + j, :] \
                .reshape(NL, 8, HDIM)
            rot_full = rot_apply(src, off) \
                .reshape(NL, HDIM * 8)
            repK[lmin:lmax, j, :] = \
                rot_full[lmin:lmax]
            repV[lmin:lmax, j, :] = \
                VP[pref_i, lmin:lmax, off + j, :]
    return repK, repV


def cos_frac(lg_base, r, t):
    nr = float(np.linalg.norm(r))
    nt = float(np.linalg.norm(t))
    cs = float(r @ t) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    fr = nr / nt if nt > 1e-12 else float('nan')
    return cs, fr


# integrity bookkeeping (a142): the L30-35 repro
# band forwards run with integ=True; every arm
# records ||dlg|| for a144.
a142_fail = 0
a142_checked = 0
rec_dlg = []


# ---------- T5 lens capture (before forwards) ----------
log('=== T5 lens capture (32 prompts) ===')
a143_diff = 0.0
for i in range(n_pr):
    ll = forward_lens(assembled[i]['ids'])
    a143_diff = max(a143_diff, float(np.max(
        np.abs(ll[NL - 1, :]
               - LG[i].astype(np.float64)))))
a143_diff = float(a143_diff)
a143_ok = bool(a143_diff == 0.0)
log('a143 lens final layer vs LG diff=%.3e ok=%s'
    % (a143_diff, a143_ok))

COS_LENS = np.zeros((NL, NP_))
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    t = t_targets[(b, c)].astype(np.float64)
    nt = float(np.linalg.norm(t))
    lb = forward_lens(assembled[base_i]['ids'])
    lp = forward_lens(assembled[pref_i]['ids'])
    for li in range(NL):
        d = lp[li] - lb[li]
        nr = float(np.linalg.norm(d))
        COS_LENS[li, k] = float(d @ t) \
            / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
med_lens = [float(np.median(COS_LENS[li]))
            for li in range(NL)]
argmax_lens = int(np.argmax(med_lens))
onset_lens = None
for li in range(NL):
    if med_lens[li] >= 0.8 * med_lens[NL - 1]:
        onset_lens = li
        break
log('lens profile: argmax L%d onset L%d '
    'med[L30]=%.4f med[L33]=%.4f med[L35]=%.4f'
    % (argmax_lens, onset_lens, med_lens[30],
       med_lens[33], med_lens[35]))

# ---------- T3 suffix bands + repro anchors ----------
log('=== T3 suffix cumulative bands ===')
COS_SUF = np.zeros((len(SUFFIX), NP_))
COS_L30 = np.zeros(NP_)
FRAC_L30 = np.zeros(NP_)
for si_, l0 in enumerate(SUFFIX):
    layers_repl = list(range(l0, NL))
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        ids_b = assembled[base_i]['ids']
        repK, repV = build_repl(k, layers_repl)
        mask = np.ones(int(LENS[base_i]),
                       dtype=bool)
        integ = (l0 == DEEP0)
        res = forward_run(ids_b, replK=repK,
                          maskK=mask, replV=repV,
                          maskV=mask, integ=integ)
        if integ:
            a142_checked += 1
            if res.get('bit', 0.0) != 0.0:
                a142_fail += 1
        r = res['lg'] - LG[base_i]
        t = t_targets[(b, c)]
        cs, fr = cos_frac(LG[base_i], r, t)
        COS_SUF[si_, k] = cs
        if l0 == DEEP0:
            COS_L30[k] = cs
            FRAC_L30[k] = fr
        rec_dlg.append(float(np.linalg.norm(r)))
    log('  L%d-35: med cos=%.4f'
        % (l0, float(np.median(COS_SUF[si_]))))
med_suf = [float(np.median(COS_SUF[si_]))
           for si_ in range(len(SUFFIX))]
obs_deep = float(np.median(COS_L30))
onset_kv = None
for si_, l0 in enumerate(SUFFIX):
    if med_suf[si_] >= 0.8 * obs_deep:
        onset_kv = l0
        break
log('suffix: obs_deep(l=%d)=%.6f onset_kv=L%s'
    % (DEEP0, obs_deep, onset_kv))

# a140b: L30-35 repro vs z49 COS_LB[5]
a140b_diff = float(np.max(np.abs(
    COS_L30 - COS_LB49[5])))
a140b_ok = bool(a140b_diff == 0.0)
log('a140b L30-35 repro vs z49 LB5: diff=%.3e '
    'ok=%s' % (a140b_diff, a140b_ok))

# ---------- T4a deep-band restricted null ----------
log('=== T4a null deep L30-35 x FRONT (R=%d) ==='
    % R_NULL)
null_deep = np.zeros(R_NULL)
for mc in range(R_NULL):
    rng = np.random.default_rng(SEED_NULL_D + mc)
    csm = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        rows = front_rows(nb)
        repK, repV = build_repl(
            k, list(range(DEEP0, NL)))
        NK = np.linalg.norm(
            repK[DEEP0:NL, rows, :], axis=2)
        NV_ = np.linalg.norm(
            repV[DEEP0:NL, rows, :], axis=2)
        RK = rng.standard_normal(
            (NL - DEEP0, len(rows),
             HDIM * 8)).astype(np.float32)
        RK *= (NK / np.linalg.norm(RK, axis=2)
               )[:, :, None]
        RV = rng.standard_normal(
            (NL - DEEP0, len(rows),
             HDIM * 8)).astype(np.float32)
        RV *= (NV_ / np.linalg.norm(RV, axis=2)
               )[:, :, None]
        fullK = repK.copy()
        fullV = repV.copy()
        fullK[DEEP0:NL, rows, :] = RK
        fullV[DEEP0:NL, rows, :] = RV
        mask = np.ones(nb, dtype=bool)
        res = forward_run(ids_b, replK=fullK,
                          maskK=mask, replV=fullV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        t = t_targets[(b, c)]
        cs, _ = cos_frac(LG[base_i], r, t)
        csm[k] = cs
        rec_dlg.append(float(np.linalg.norm(r)))
    null_deep[mc] = float(np.median(csm))
    if (mc + 1) % 25 == 0:
        log('  mc %d/%d (med=%.4f)'
            % (mc + 1, R_NULL, null_deep[mc]))
p_deep = float((1 + int(np.sum(
    null_deep >= obs_deep))) / (R_NULL + 1))
log('DEEP: obs=%.4f null med=%.4f max=%.4f '
    'p=%.5f'
    % (obs_deep, float(np.median(null_deep)),
       float(null_deep.max()), p_deep))

# ---------- FRONT repro + T4b restricted null ----------
log('=== FRONT repro + T4b null (R=%d) ==='
    % R_NULL)
COS_FR = np.zeros(NP_)
FRAC_FR = np.zeros(NP_)
FR_replK = []
FR_replV = []
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    ids_b = assembled[base_i]['ids']
    repK, repV = build_repl(
        k, list(range(NL)))
    mask = np.ones(int(LENS[base_i]),
                   dtype=bool)
    res = forward_run(ids_b, replK=repK,
                      maskK=mask, replV=repV,
                      maskV=mask)
    r = res['lg'] - LG[base_i]
    t = t_targets[(b, c)]
    cs, fr = cos_frac(LG[base_i], r, t)
    COS_FR[k] = cs
    FRAC_FR[k] = fr
    FR_replK.append(repK)
    FR_replV.append(repV)
    rec_dlg.append(float(np.linalg.norm(r)))
obs_front = float(np.median(COS_FR))
log('FRONT repro: med cos=%.6f (z49 %.6f)'
    % (obs_front,
       float(np.median(COS_FRONT49))))
a140a_diff = float(np.max(np.abs(
    COS_FR - COS_FRONT49)))
a140a_ok = bool(a140a_diff == 0.0)
log('a140a FRONT repro vs z49: diff=%.3e ok=%s'
    % (a140a_diff, a140a_ok))

null_front = np.zeros(R_NULL)
for mc in range(R_NULL):
    rng = np.random.default_rng(SEED_NULL_F + mc)
    csm = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        rows = front_rows(nb)
        repK = FR_replK[k]
        repV = FR_replV[k]
        NK = np.linalg.norm(
            repK[:, rows, :], axis=2)
        NV_ = np.linalg.norm(
            repV[:, rows, :], axis=2)
        RK = rng.standard_normal(
            (NL, len(rows), HDIM * 8)) \
            .astype(np.float32)
        RK *= (NK / np.linalg.norm(RK, axis=2)
               )[:, :, None]
        RV = rng.standard_normal(
            (NL, len(rows), HDIM * 8)) \
            .astype(np.float32)
        RV *= (NV_ / np.linalg.norm(RV, axis=2)
               )[:, :, None]
        fullK = repK.copy()
        fullV = repV.copy()
        fullK[:, rows, :] = RK
        fullV[:, rows, :] = RV
        mask = np.ones(nb, dtype=bool)
        res = forward_run(ids_b, replK=fullK,
                          maskK=mask, replV=fullV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        t = t_targets[(b, c)]
        cs, _ = cos_frac(LG[base_i], r, t)
        csm[k] = cs
        rec_dlg.append(float(np.linalg.norm(r)))
    null_front[mc] = float(np.median(csm))
    if (mc + 1) % 25 == 0:
        log('  mc %d/%d (med=%.4f)'
            % (mc + 1, R_NULL, null_front[mc]))
p_front_r = float((1 + int(np.sum(
    null_front >= obs_front))) / (R_NULL + 1))
log('FRONT-restricted: obs=%.4f null med=%.4f '
    'max=%.4f p=%.5f'
    % (obs_front, float(np.median(null_front)),
       float(null_front.max()), p_front_r))

# ---------- T2 single-layer profile ----------
log('=== T2 single-layer profile (36 x 24) ===')
COS_SL = np.zeros((NL, NP_))
for li in range(NL):
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        ids_b = assembled[base_i]['ids']
        repK, repV = build_repl(k, [li])
        mask = np.ones(int(LENS[base_i]),
                       dtype=bool)
        res = forward_run(ids_b, replK=repK,
                          maskK=mask, replV=repV,
                          maskV=mask)
        r = res['lg'] - LG[base_i]
        t = t_targets[(b, c)]
        cs, _ = cos_frac(LG[base_i], r, t)
        COS_SL[li, k] = cs
        rec_dlg.append(float(np.linalg.norm(r)))
    if (li + 1) % 6 == 0:
        log('  layers %d/36 done' % (li + 1))
med_sl = [float(np.median(COS_SL[li]))
          for li in range(NL)]
l_star = int(np.argmax(med_sl))
log('single-layer peak: L%d med cos=%.4f '
    '(L30=%.4f L33=%.4f L35=%.4f)'
    % (l_star, med_sl[l_star], med_sl[30],
       med_sl[33], med_sl[35]))

# a142 / a144
a142_ok = bool(a142_fail == 0
               and a142_checked >= 1)
max_dlg = float(np.max(np.array(rec_dlg)))
a144_ok = bool(max_dlg >= T_GATE)
log('a142 integrity fails=%d checked=%d ok=%s | '
    'a144 max||dlg||=%.4f ok=%s'
    % (a142_fail, a142_checked, a142_ok,
       max_dlg, a144_ok))

# ---------- verdict ----------
sig_deep = bool(p_deep < 0.05)
if sig_deep and l_star >= DEEP0:
    verdict = 'kvdeep_local_qwen'
elif sig_deep:
    verdict = 'kvdeep_broad_qwen'
else:
    verdict = 'kvdeep_null_qwen'
log('VERDICT: %s (sig_deep=%s l*=%d p_deep=%.5f '
    'p_front_r=%.5f)'
    % (verdict, sig_deep, l_star, p_deep,
       p_front_r))

anchor_core_ok = bool(a116_ok and a139_ok
                      and a140a_ok and a140b_ok
                      and a141_ok and a142_ok
                      and a143_ok and a144_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_SL=COS_SL,
         COS_SUF=COS_SUF,
         COS_L30=COS_L30, FRAC_L30=FRAC_L30,
         COS_FR=COS_FR, FRAC_FR=FRAC_FR,
         COS_LENS=COS_LENS,
         null_deep=null_deep,
         null_front=null_front,
         med_sl=np.array(med_sl),
         med_suf=np.array(med_suf),
         med_lens=np.array(med_lens),
         a139_diff=np.float64(a139_diff),
         a140a_diff=np.float64(a140a_diff),
         a140b_diff=np.float64(a140b_diff),
         a141_diff=np.float64(a141_diff),
         a142_fail=np.int64(a142_fail),
         a142_checked=np.int64(a142_checked),
         a143_diff=np.float64(a143_diff),
         a144_max_dlg=np.float64(max_dlg),
         a116_ok=np.bool_(a116_ok),
         a139_ok=np.bool_(a139_ok),
         a140a_ok=np.bool_(a140a_ok),
         a140b_ok=np.bool_(a140b_ok),
         a141_ok=np.bool_(a141_ok),
         a142_ok=np.bool_(a142_ok),
         a143_ok=np.bool_(a143_ok),
         a144_ok=np.bool_(a144_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_single_layer': {'med_cos': med_sl,
                        'peak_layer': l_star,
                        'med_at_peak':
                            med_sl[l_star],
                        'med_L30': med_sl[30],
                        'med_L33': med_sl[33],
                        'med_L35': med_sl[35]},
    'T3_suffix': {'layers': list(SUFFIX),
                  'med_cos': med_suf,
                  'obs_deep': obs_deep,
                  'onset_layer': onset_kv},
    'T4a_null_deep': {'R': R_NULL,
                      'seed': SEED_NULL_D,
                      'obs': obs_deep,
                      'med_null':
                          float(np.median(
                              null_deep)),
                      'max_null':
                          float(null_deep.max()),
                      'p_deep': p_deep},
    'T4b_null_front': {'R': R_NULL,
                       'seed': SEED_NULL_F,
                       'obs': obs_front,
                       'med_null':
                           float(np.median(
                               null_front)),
                       'max_null':
                           float(null_front.max()),
                       'p_front_restricted':
                           p_front_r,
                       'note': 'retro-validation '
                               'of the 3049 p_front '
                               '(3049 null randomized '
                               'ALL positions via the '
                               'all-True mask)'},
    'T5_lens': {'med_cos': med_lens,
                'argmax_layer': argmax_lens,
                'onset_layer': onset_lens,
                'med_L30': med_lens[30],
                'med_L33': med_lens[33],
                'med_L35': med_lens[35]},
    'anchors': {'a116_seals_ok': a116_ok,
                'a139_recapture_diff': a139_diff,
                'a140a_front_diff': a140a_diff,
                'a140b_l30_diff': a140b_diff,
                'a141_sham_diff': a141_diff,
                'a142_fail': a142_fail,
                'a142_checked': a142_checked,
                'a143_lens_diff': a143_diff,
                'a144_max_dlg': max_dlg,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run4 authoritative (fp32; run1-2 '
                 'crashed pre-anchor at the T5 '
                 'lens capture and run3 failed the '
                 'a143 anchor, see corrections; '
                 'capture bank loaded from the '
                 'phase3048 npz; per-pair chain '
                 'anchors vs the phase3049 npz)',
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
