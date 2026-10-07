# -*- coding: utf-8 -*-
# Phase 3048 - Omega-P45: full-position KV replay
# and decomposition (fp32). Final verdict on the
# "KV carries the prefix effect" hypothesis:
# 3047 replayed the TARGET-position KV
# displacement jointly at all 36 layers and got
# null (med frac 0.0756, p_cos 0.124). Here the
# replay extends to ALL body positions (all 8 kv
# heads, all 36 layers) with an EXACT replay arm:
# post-norm keys captured at the k_norm output
# (hook point of 3046 probe: (1,s,8,128),
# pre-RoPE), rotated by the prompt position
# offset (RoPE relative correction) so that the
# model's own RoPE at the base position maps them
# onto the prefix's post-RoPE keys; values are
# copied verbatim (V has no norm/RoPE). Arms per
# pair k=(c-1)*8+b (c 1..3, b 0..7, 24 old pairs):
# ADD: + pre-norm displacement at all positions
#   (z47-protocol companion, descriptive);
# REP: exact replacement at all body positions
#   (PRIMARY; null = norm-matched random
#   replacement, R=200, seed 9851+mc);
# SCR: scramble the PREFIX-token positions on
#   the prefix prompt (norm-matched random,
#   seed 9852) - complementary removal arm;
# T3: single-layer exact replacement profile
#   (36 layers x 24 pairs, descriptive; other
#   layers self-replaced bit-identically).
# verdict_tree: sig = p_REP < 0.05; sig AND med
# frac_REP >= 0.5 -> kvpos_carries_qwen; sig
# AND med frac_REP < 0.5 -> kvpos_partial_qwen;
# else -> kvpos_null_qwen; single branch.
# anchors: a126 chain vs z47 (KPRE/VPRE kv7
# slice at target pos, DPREK/DPREV) bit 0.0;
# a127 duplicate capture prompt0 full field bit
# 0.0; a128 sham self-replacement bit identity
# (lg + fields); a129 exact-replay empirical
# check: replaced base past keys match the
# prefix's own past keys (gate 1e-3), replaced
# values bit 0.0, rotation group property gate
# 1e-4; a130 integrity: every intervention
# forward bit-exact (mod == orig+delta, mod ==
# where(mask,repl,orig) inside torch; zero
# failures); a131 chain-entry max ||dlg|| >=
# 0.05. Capture bank: all 48 prompts (chain),
# statistics on the 24 old pairs.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3048
NAME = 'omega_p45_kvpos_full_replay_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 7
HDIM = 128
R_NULL = 200
SEED_NULL = 9851
SEED_SCR = 9852
SEED_MAIN = 3009
T_GATE = 0.05
FRAC_CARRY = 0.5
A129_GATE = 1e-3
A129B_GATE = 1e-4
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
NEW_BODIES = (
    'The game was delayed, so',
    'She stayed home because',
    'The engine failed, therefore',
    'He kept smiling, although',)
NEW_TARGETS = ('so', 'because', 'therefore',
               'although')

PREREG = {
    'mode': 'fp32 MODEL (torch.float32, eager, seed '
            '3009); 48 prompts verbatim 3042-3047 '
            'bank; capture at THREE hook points for '
            'all 36 layers: k_proj output (pre-norm, '
            'chain), k_norm output (post-norm '
            '(1,s,8,128), exact-replay source), '
            'v_proj output; all positions, all 8 kv '
            'heads (1024 dims); statistics on the '
            '24 old pairs',
    'question': '3048 A main line: KV-carries '
                'hypothesis final verdict - does '
                'the prefix logit effect flow '
                'through the KV writes at ALL '
                'positions when replayed exactly? '
                '(i) full-position capture (T1); '
                '(ii) ADD arm: all-position '
                'pre-norm displacement (z47 '
                'companion); (iii) REP arm: exact '
                'post-norm replacement with RoPE '
                'offset rotation (primary, '
                'norm-matched random-replacement '
                'null); (iv) SCR arm: prefix-token '
                'scramble on the prefix prompt '
                '(removal); (v) T3 single-layer '
                'replacement profile',
    'T1': 'capture: for each of the 48 prompts one '
          'fp32 prefill, full-position fields '
          'KPpre/KPpost/VP (48,36,Lmax,1024) fp32 '
          'padded, LG (48,) double; a126 chain vs '
          'z47 bit 0.0; a127 duplicate prompt0',
    'T2_ADD': 'per pair: dpre = KPpre[pref, pos+off] '
              '- KPpre[base, pos] at all positions '
              'all layers (1024 dims), injected '
              'additively into the base prompt; '
              'resp = lg - LG[base]; cos(resp, t_bc), '
              'frac = ||resp||/||t_bc||; descriptive '
              'companion of z47 (no null)',
    'T2_REP': 'per pair: replace k_norm output at '
              'all base positions with '
              'rot_apply(KPpost[pref, pos+off], off) '
              'and v_proj output with '
              'VP[pref, pos+off]; PRIMARY stat = '
              'med over 24 pairs of cos(resp, t_bc); '
              'null = R=%d MC: replace with '
              'norm-matched random vectors (per '
              'layer per position, matched to the '
              'REP norms), seed %d + mc; p = '
              'P(null med cos >= obs); frac = '
              '||resp||/||t_bc||'
              % (R_NULL, SEED_NULL),
    'T4_SCR': 'on the prefix prompt: replace '
              'k_norm/v_proj outputs at the '
              'prefix-token positions (0..off-1, '
              'pair-dependent) with norm-matched '
              'random vectors (seed %d); resp = lg '
              '- LG[pref]; kill_frac = '
              '||resp||/||t_bc||, kill_cos = '
              'cos(resp, -t_bc); descriptive '
              'removal complement' % SEED_SCR,
    'T3': 'single-layer exact replacement: for '
          'each layer l, replace ONLY layer l with '
          'the REP field (other layers '
          'self-replaced bit-identically from the '
          'captured base field); frac profile (36 '
          'x 24), descriptive',
    'verdict_tree': 'sig = p_REP < 0.05; sig AND '
                    'med frac_REP >= 0.5 -> '
                    'kvpos_carries_qwen; sig AND '
                    'med frac_REP < 0.5 -> '
                    'kvpos_partial_qwen; else -> '
                    'kvpos_null_qwen; single branch',
    'anchors': 'a126 chain KPRE/VPRE kv7-slice '
               'target-pos + DPREK/DPREV vs z47 bit '
               '0.0; a127 duplicate capture prompt0 '
               'full field bit 0.0; a128 sham '
               'self-replacement bit identity; a129 '
               'empirical exact replay: replaced '
               'base past keys vs prefix past keys '
               'gate %g, replaced values bit 0.0, '
               'rotation group property gate %g; '
               'a130 integrity bit-exact on all '
               'intervention forwards; a131 '
               'chain-entry max ||dlg|| >= %g'
               % (A129_GATE, A129B_GATE, T_GATE),
    'control': 'norm-matched random replacement '
               'per layer per position (same '
               'machinery); prefix scramble arm; '
               'no other intervention',
    'statistics_discipline': 'obs and null on the '
                             'SAME scale (raw cos, '
                             'same 24 pairs, same '
                             'replacement machinery); '
                             'null never on intervened '
                             'quantities; verdict in '
                             'one branch; RoPE '
                             'convention verified '
                             'empirically via past-key '
                             'match before statistics',
    'corrections': 'run1 crashed pre-anchor on '
                   'the prompt-assembly alignment '
                   'assertion: BPE leading-space '
                   'boundary effect (first body '
                   'token is the no-space variant in '
                   'the base prompt but the '
                   'leading-space variant inside the '
                   'prefix prompt, so naive '
                   'subsequence matching fails); '
                   'alignment redefined as tail '
                   'matching (pref ids end with base '
                   'ids[1:]) plus a first-token '
                   'semantic strip-equality check; '
                   'offset = length difference. '
                   'run2 crashed mid-T4 on a '
                   'replacement-array length '
                   'mismatch (partial-position repl '
                   'passed where a full-length '
                   'reshape was assumed; '
                   'forward_run now reshapes to '
                   '(-1,8,128) and integrity scatters '
                   'onto masked rows); the run2 null '
                   'printout (p=0.00498) was found '
                   'to be single-pair (accidental '
                   'reuse of the last T2 iteration) '
                   'instead of the preregistered '
                   'per-mc 24-pair median - null '
                   'implementation corrected to the '
                   'preregistered statistic; ADD/REP '
                   'observed statistics are '
                   'frozen-seed deterministic and '
                   're-derived unchanged; run3 '
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
log('model loaded fp32 (qk-norm + rotary buffers '
    'confirmed, vocab=%d)' % NVOC)

SL = slice(KV_HEAD * HDIM, (KV_HEAD + 1) * HDIM)


def rot_apply(x, delta):
    """Rotate (...,8,128) by delta*inv_freq
    (NeoX half-pairing, fp32, matches the model's
    RoPE linear map up to rounding)."""
    ang = INV_FREQ * np.float32(delta)
    emb = np.concatenate([ang, ang]).astype(
        np.float32)
    c = np.cos(emb)[None, None, :]
    s = np.sin(emb)[None, None, :]
    x1 = x[..., :64]
    x2 = x[..., 64:]
    rh = np.concatenate([-x2, x1], axis=-1)
    return x * c + rh * s


# hook states: k_proj (pre-norm): additive only
stateKpre = {li: {'delta': None} for li in range(NL)}
capKpre = {li: {'rec': False, 'orig': None,
                'mod': None} for li in range(NL)}
# k_norm (post-norm): replacement only
stateKn = {li: {'repl': None, 'mask': None}
           for li in range(NL)}
capKn = {li: {'rec': False, 'orig': None,
              'mod': None} for li in range(NL)}
# v_proj: additive + replacement
stateV = {li: {'delta': None, 'repl': None,
               'mask': None} for li in range(NL)}
capV = {li: {'rec': False, 'orig': None,
             'mod': None} for li in range(NL)}


def hook_pre(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['delta'] is not None:
            n = out.shape[1]
            out[0, :n, :] += st['delta']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


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
        if st['delta'] is not None:
            n = out.shape[1]
            out[0, :n, :] += st['delta']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_proj \
        .register_forward_hook(hook_pre(
            stateKpre[li], capKpre[li]))
    layers[li].self_attn.k_norm \
        .register_forward_hook(hook_norm(
            stateKn[li], capKn[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))


def reset_all():
    for li in range(NL):
        stateKpre[li]['delta'] = None
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['delta'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKpre[li]['rec'] = False
        capKn[li]['rec'] = False
        capV[li]['rec'] = False


def forward_cap(ids):
    reset_all()
    for li in range(NL):
        capKpre[li]['rec'] = True
        capKn[li]['rec'] = True
        capV[li]['rec'] = True
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


def past_numpy(past):
    pk = {}
    pv = {}
    for li in range(NL):
        pk[li] = past.layers[li].keys[0] \
            .detach().cpu().numpy()
        pv[li] = past.layers[li].values[0] \
            .detach().cpu().numpy()
    return pk, pv


def forward_run(ids, dpreK=None, dpreV=None,
                replK=None, maskK=None, replV=None,
                maskV=None, integ=False,
                want_past=False):
    """dpreK/dpreV: (NL,n,1024) additive at
    k_proj/v_proj; replK: (NL,n,8*128) post-norm
    replacement with mask (n,); replV:
    (NL,n,1024) with maskV."""
    reset_all()
    n = len(ids)
    tk = tv = rk = rv = None
    mk = mv = None
    if dpreK is not None:
        tk = torch.tensor(np.ascontiguousarray(dpreK),
                          dtype=torch.float32,
                          device='cuda')
        for li in range(NL):
            stateKpre[li]['delta'] = tk[li]
    if dpreV is not None:
        tv = torch.tensor(np.ascontiguousarray(dpreV),
                          dtype=torch.float32,
                          device='cuda')
        for li in range(NL):
            stateV[li]['delta'] = tv[li]
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
        rv = torch.tensor(np.ascontiguousarray(replV),
                          dtype=torch.float32,
                          device='cuda')
        mv = torch.tensor(np.asarray(maskV),
                          device='cuda')
        for li in range(NL):
            stateV[li]['repl'] = rv[li]
            stateV[li]['mask'] = mv
    if integ:
        for li in range(NL):
            capKpre[li]['rec'] = True
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
            if tk is not None:
                e = max(e, float(
                    (capKpre[li]['mod']
                     - (capKpre[li]['orig'] + tk[li]))
                    .abs().max()))
            if rk is not None:
                comb = capKn[li]['orig'].clone()
                comb[mk] = rk[li]
                e = max(e, float(
                    (capKn[li]['mod'] - comb)
                    .abs().max()))
            if tv is not None:
                e = max(e, float(
                    (capV[li]['mod']
                     - (capV[li]['orig'] + tv[li]))
                    .abs().max()))
            if rv is not None:
                comb = capV[li]['orig'].clone()
                comb[mv] = rv[li]
                e = max(e, float(
                    (capV[li]['mod'] - comb)
                    .abs().max()))
        res['bit'] = e
    reset_all()
    if want_past:
        res['pk'], res['pv'] = past_numpy(
            out.past_key_values)
    return res


# ---------- chain sources ----------
z47 = np.load(os.path.join(
    BASE, 'phase3047',
    'omega_p44_kv_joint_replay_qwen',
    'omega_p44_kv_joint_replay_qwen.npz'),
    allow_pickle=True)

seal_detail = []
for ph, nm in (
        (3037, 'omega_p34_kv_situational_'
               'specificity_qwen'),
        (3038, 'omega_p35_reentrant_readout_qwen'),
        (3039, 'omega_p36_direct_logistic_'
               'replication_qwen'),
        (3040, 'omega_p37_situational_'
               'component_qwen'),
        (3041, 'omega_p38_situational_axis_qwen'),
        (3042, 'omega_p39_style_field_probe_qwen'),
        (3043, 'omega_p40_field_variance_qwen'),
        (3044, 'omega_p41_field_axis_injection_'
               'qwen'),
        (3045, 'omega_p42_l20_axis_anatomy_qwen'),
        (3046, 'omega_p43_kfield_injection_qwen'),
        (3047, 'omega_p44_kv_joint_replay_qwen')):
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

# ---------- assemble prompts (3047 verbatim) ----------
word_tok = {}
for w in set(TARGETS) | set(NEW_TARGETS):
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
                          'cond': ci, 'body': bi,
                          'new': False})
for bi in range(len(NEW_BODIES)):
    for ci in range(len(PREFIXES)):
        s = (PREFIXES[ci] + ' ' + NEW_BODIES[bi]) \
            if PREFIXES[ci] else NEW_BODIES[bi]
        ids = [int(x) for x in tok(
            s, add_special_tokens=False)[
            'input_ids']]
        t = word_tok[NEW_TARGETS[bi]]
        assert ids.count(t) == 1, (bi, ci)
        assembled.append({'ids': ids,
                          'pos': ids.index(t),
                          'cond': ci, 'body': bi,
                          'new': True})
n_pr = len(assembled)
n_old = len(BODIES) * len(PREFIXES)
assert n_old == 32 and n_pr == 48
idx_of = {}
for i in range(n_pr):
    idx_of[(assembled[i]['cond'],
            assembled[i]['body'],
            assembled[i]['new'])] = i
# offsets: body-token subsequence position inside
# the prefix prompt (prefix strictly prepended)
for i in range(n_pr):
    ci = assembled[i]['cond']
    if ci == 0:
        assembled[i]['off'] = 0
    else:
        bid = assembled[idx_of[(0, assembled[i]
                               ['body'],
                               assembled[i]['new'])]][
            'ids']
        pid = assembled[i]['ids']
        off = len(pid) - len(bid)
        assert off > 0, i
        # tail alignment: body tokens 1..end match
        assert list(pid[off + 1:]) \
            == list(bid[1:]), i
        # first body token: leading-space variant of
        # the same word (BPE boundary effect)
        w0b = tok.decode([bid[0]]).strip()
        w0p = tok.decode([pid[off]]).strip()
        assert w0b == w0p, (i, w0b, w0p)
        assembled[i]['off'] = off
LMAX = max(len(assembled[i]['ids'])
           for i in range(n_pr))
log('assembled %d prompts, Lmax=%d' % (n_pr, LMAX))

# ---------- T1 capture ----------
log('=== T1 capture (48 prompts, all positions) ===')
LG = [None] * n_pr
KPpre = np.zeros((n_pr, NL, LMAX, HDIM * 8),
                 dtype=np.float32)
KPpost = np.zeros((n_pr, NL, LMAX, HDIM * 8),
                  dtype=np.float32)
VP = np.zeros((n_pr, NL, LMAX, HDIM * 8),
              dtype=np.float32)
LENS = np.zeros(n_pr, dtype=np.int64)
for i in range(n_pr):
    lg, kp, kn, vp = forward_cap(assembled[i]['ids'])
    assert lg.shape[0] == NVOC
    LG[i] = lg
    LENS[i] = len(assembled[i]['ids'])
    KPpre[i, :, :LENS[i], :] = kp
    KPpost[i, :, :LENS[i], :] = kn
    VP[i, :, :LENS[i], :] = vp
LG = np.stack(LG)
log('capture done')

# a127: duplicate capture prompt0
lg2, kp2, kn2, vp2 = forward_cap(assembled[0]['ids'])
a127_diff = float(np.max(np.abs(LG[0] - lg2)))
a127_diff = max(a127_diff, float(np.max(np.abs(
    KPpre[0, :, :LENS[0], :] - kp2))))
a127_diff = max(a127_diff, float(np.max(np.abs(
    KPpost[0, :, :LENS[0], :] - kn2))))
a127_diff = max(a127_diff, float(np.max(np.abs(
    VP[0, :, :LENS[0], :] - vp2))))
a127_diff = float(a127_diff)
log('a127 dup capture diff=%.3e' % a127_diff)

# a126: chain vs z47 (target pos, kv7 slice)
pos0 = [assembled[i]['pos'] for i in range(n_pr)]
KSL = np.stack([KPpre[i, :, pos0[i], SL]
                .astype(np.float64)
                for i in range(n_pr)])
VSL = np.stack([VP[i, :, pos0[i], SL]
                .astype(np.float64)
                for i in range(n_pr)])
a126_d1 = float(np.max(np.abs(KSL - z47['KPRE'])))
a126_d2 = float(np.max(np.abs(VSL - z47['VPRE'])))

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)
NP_ = 24
DPREK_sl = np.zeros((NP_, NL, HDIM))
DPREV_sl = np.zeros((NP_, NL, HDIM))
for k in range(NP_):
    ic = idx_of[(int(cidx[k]), int(bidx[k]), False)]
    ib = idx_of[(0, int(bidx[k]), False)]
    DPREK_sl[k] = KSL[ic] - KSL[ib]
    DPREV_sl[k] = VSL[ic] - VSL[ib]
a126_d3 = float(np.max(np.abs(
    DPREK_sl - z47['DPREK'])))
a126_d4 = float(np.max(np.abs(
    DPREV_sl - z47['DPREV'])))
a126_diff = float(max(a126_d1, a126_d2,
                      a126_d3, a126_d4))
a126_ok = bool(a126_diff == 0.0)
log('a126 chain Kpre=%.3e Vpre=%.3e DPREK=%.3e '
    'DPREV=%.3e ok=%s'
    % (a126_d1, a126_d2, a126_d3, a126_d4, a126_ok))

# targets
t_targets = {}
for b in range(len(BODIES)):
    for c in (1, 2, 3):
        t_targets[(b, c)] = LG[idx_of[(c, b, False)]] \
            - LG[idx_of[(0, b, False)]]
TT = np.stack([t_targets[(int(bidx[k]),
                          int(cidx[k]))]
               for k in range(NP_)])
NTT = np.linalg.norm(TT, axis=1)
log('t norms: med=%.4f max=%.4f'
    % (float(np.median(NTT)), float(NTT.max())))

# a128: sham self-replacement (bit identity)
b0 = int(bidx[0])
base0 = idx_of[(0, b0, False)]
ids0 = assembled[base0]['ids']
n0 = len(ids0)
selfK = KPpost[base0, :, :n0, :].copy()
selfV = VP[base0, :, :n0, :].copy()
m_all = np.ones(n0, dtype=bool)
res = forward_run(ids0, replK=selfK, maskK=m_all,
                  replV=selfV, maskV=m_all, integ=True)
a128_diff = float(np.max(np.abs(res['lg']
                                - LG[base0])))
a128_diff = float(max(a128_diff, res['bit']))
a128_ok = bool(a128_diff == 0.0)
log('a128 sham self-replacement diff=%.3e ok=%s'
    % (a128_diff, a128_ok))

# a129: empirical exact replay (pair 0)
k0 = 0
ic0 = idx_of[(int(cidx[0]), int(bidx[0]), False)]
ib0 = idx_of[(0, int(bidx[0]), False)]
off0 = assembled[ic0]['off']
nb0 = len(assembled[ib0]['ids'])
repK_129 = np.zeros((NL, nb0, HDIM * 8),
                    dtype=np.float32)
for l in range(NL):
    src = KPpost[ic0, l, off0:off0 + nb0, :] \
        .reshape(nb0, 8, HDIM)
    repK_129[l] = rot_apply(src, off0) \
        .reshape(nb0, HDIM * 8)
repV_129 = VP[ic0, :, off0:off0 + nb0, :].copy()
m_b0 = np.ones(nb0, dtype=bool)
res_inj = forward_run(ids0, replK=repK_129,
                      maskK=m_b0, replV=repV_129,
                      maskV=m_b0, want_past=True)
res_ref = forward_run(assembled[ic0]['ids'],
                      want_past=True)
a129_key = 0.0
for li in range(NL):
    a129_key = max(a129_key, float(np.max(np.abs(
        res_inj['pk'][li][:, :nb0, :]
        - res_ref['pk'][li][:, off0:off0 + nb0,
          :]))))
a129_vbit = 0.0
for li in range(NL):
    a129_vbit = max(a129_vbit, float(np.max(np.abs(
        res_inj['pv'][li][:, :nb0, :]
        - res_ref['pv'][li][:, off0:off0 + nb0,
          :]))))
xr = np.random.default_rng(77).standard_normal(
    (4, 8, HDIM)).astype(np.float32)
a129_grp = float(np.max(np.abs(
    rot_apply(rot_apply(xr, 3.0), 5.0)
    - rot_apply(xr, 8.0))))
a129_ok = bool(a129_key <= A129_GATE
               and a129_vbit == 0.0
               and a129_grp <= A129B_GATE)
log('a129 exact replay: key=%.3e vbit=%.3e '
    'grp=%.3e ok=%s'
    % (a129_key, a129_vbit, a129_grp, a129_ok))

# ---------- T2 arms (24 pairs) ----------
log('=== T2 arms (ADD + REP, 24 pairs) ===')
COS_ADD = np.zeros(NP_)
FRAC_ADD = np.zeros(NP_)
COS_REP = np.zeros(NP_)
FRAC_REP = np.zeros(NP_)
RESP_ADD = np.zeros((NP_, NVOC), dtype=np.float32)
RESP_REP = np.zeros((NP_, NVOC), dtype=np.float32)
RESP_SCR = np.zeros((NP_, NVOC), dtype=np.float32)
KILL_FRAC = np.zeros(NP_)
KILL_COS = np.zeros(NP_)
a130_fail = 0
rec_kind = []
rec_body = []
rec_dlg = []
repK_list = []
repV_list = []
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b, False)]
    pref_i = idx_of[(c, b, False)]
    ids_b = assembled[base_i]['ids']
    ids_p = assembled[pref_i]['ids']
    nb = len(ids_b)
    off = assembled[pref_i]['off']
    t = t_targets[(b, c)]
    nt = float(np.linalg.norm(t))
    # ADD: pre-norm all-position displacement
    dK = (KPpre[pref_i, :, off:off + nb, :]
          - KPpre[base_i, :, :nb, :]).copy()
    dV = (VP[pref_i, :, off:off + nb, :]
          - VP[base_i, :, :nb, :]).copy()
    res = forward_run(ids_b, dpreK=dK, dpreV=dV,
                      integ=True)
    if res['bit'] != 0.0:
        a130_fail += 1
    r = res['lg'] - LG[base_i]
    nr = float(np.linalg.norm(r))
    COS_ADD[k] = float(r @ t) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    FRAC_ADD[k] = nr / nt if nt > 1e-12 \
        else float('nan')
    RESP_ADD[k] = r
    rec_kind.append(1)
    rec_body.append(b)
    rec_dlg.append(nr)
    # REP: exact post-norm replacement
    repK = np.zeros((NL, nb, HDIM * 8),
                    dtype=np.float32)
    for l in range(NL):
        src = KPpost[pref_i, l, off:off + nb, :] \
            .reshape(nb, 8, HDIM)
        repK[l] = rot_apply(src, off) \
            .reshape(nb, HDIM * 8)
    repV = VP[pref_i, :, off:off + nb, :].copy()
    repK_list.append(repK)
    repV_list.append(repV)
    m_b = np.ones(nb, dtype=bool)
    res = forward_run(ids_b, replK=repK, maskK=m_b,
                      replV=repV, maskV=m_b,
                      integ=True)
    if res['bit'] != 0.0:
        a130_fail += 1
    r = res['lg'] - LG[base_i]
    nr = float(np.linalg.norm(r))
    COS_REP[k] = float(r @ t) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    FRAC_REP[k] = nr / nt if nt > 1e-12 \
        else float('nan')
    RESP_REP[k] = r
    rec_kind.append(2)
    rec_body.append(b)
    rec_dlg.append(nr)
    if (k + 1) % 8 == 0:
        log('  pairs %d/24 done' % (k + 1))
obs_cos_add = float(np.median(COS_ADD))
obs_frac_add = float(np.nanmedian(FRAC_ADD))
obs_cos_rep = float(np.median(COS_REP))
obs_frac_rep = float(np.nanmedian(FRAC_REP))
log('T2 ADD: med cos=%.4f med frac=%.4f'
    % (obs_cos_add, obs_frac_add))
log('T2 REP: med cos=%.4f med frac=%.4f'
    % (obs_cos_rep, obs_frac_rep))

# null MC for REP (per-mc 24-pair median,
# preregistered statistic)
log('=== null MC (R=%d random replacement, '
    '24-pair median) ===' % R_NULL)
null_cos = np.zeros(R_NULL)
null_frac = np.zeros(R_NULL)
for mc in range(R_NULL):
    rng = np.random.default_rng(SEED_NULL + mc)
    csm = np.zeros(NP_)
    frm = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b, False)]
        ids_b = assembled[base_i]['ids']
        nb = len(ids_b)
        repK = repK_list[k]
        repV = repV_list[k]
        m_b = np.ones(nb, dtype=bool)
        NK = np.linalg.norm(repK, axis=2)
        NV_ = np.linalg.norm(repV, axis=2)
        RK = rng.standard_normal(
            (NL, nb, HDIM * 8)).astype(np.float32)
        RK *= (NK / np.linalg.norm(RK, axis=2)
               )[:, :, None]
        RV = rng.standard_normal(
            (NL, nb, HDIM * 8)).astype(np.float32)
        RV *= (NV_ / np.linalg.norm(RV, axis=2)
               )[:, :, None]
        res = forward_run(ids_b, replK=RK,
                          maskK=m_b, replV=RV,
                          maskV=m_b)
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        t = t_targets[(b, c)]
        nt = float(np.linalg.norm(t))
        csm[k] = float(r @ t) / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
        frm[k] = nr / nt if nt > 1e-12 \
            else float('nan')
        rec_kind.append(3)
        rec_body.append(b)
        rec_dlg.append(nr)
    null_cos[mc] = float(np.median(csm))
    null_frac[mc] = float(np.nanmedian(frm))
    if (mc + 1) % 25 == 0:
        log('  mc %d/%d (null med cos=%.4f)'
            % (mc + 1, R_NULL, null_cos[mc]))
p_rep = float((1 + int(np.sum(
    null_cos >= obs_cos_rep))) / (R_NULL + 1))
log('REP: obs med cos=%.4f null med=%.4f max=%.4f '
    'p=%.5f'
    % (obs_cos_rep, float(np.median(null_cos)),
       float(null_cos.max()), p_rep))

# ---------- T4 SCR (prefix-token scramble) ----------
log('=== T4 SCR (prefix-token scramble) ===')
rng4 = np.random.default_rng(SEED_SCR)
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b, False)]
    pref_i = idx_of[(c, b, False)]
    ids_p = assembled[pref_i]['ids']
    off = assembled[pref_i]['off']
    t = t_targets[(b, c)]
    nt = float(np.linalg.norm(t))
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
                      replV=RV, maskV=m_p, integ=True)
    if res['bit'] != 0.0:
        a130_fail += 1
    r = res['lg'] - LG[pref_i]
    nr = float(np.linalg.norm(r))
    KILL_FRAC[k] = nr / nt if nt > 1e-12 \
        else float('nan')
    KILL_COS[k] = float(r @ (-t)) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    RESP_SCR[k] = r
    rec_kind.append(4)
    rec_body.append(b)
    rec_dlg.append(nr)
med_kill_frac = float(np.nanmedian(KILL_FRAC))
med_kill_cos = float(np.median(KILL_COS))
log('SCR: med kill_frac=%.4f med kill_cos=%.4f'
    % (med_kill_frac, med_kill_cos))

# ---------- T3 single-layer profile ----------
log('=== T3 single-layer replacement profile ===')
FRAC_L = np.zeros((NL, NP_))
for l in range(NL):
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b, False)]
        pref_i = idx_of[(c, b, False)]
        ids_b = assembled[base_i]['ids']
        nb = len(ids_b)
        off = assembled[pref_i]['off']
        t = t_targets[(b, c)]
        nt = float(np.linalg.norm(t))
        rpK = KPpost[base_i, :, :nb, :].copy()
        src = KPpost[pref_i, l, off:off + nb, :] \
            .reshape(nb, 8, HDIM)
        rpK[l] = rot_apply(src, off) \
            .reshape(nb, HDIM * 8)
        rpV = VP[base_i, :, :nb, :].copy()
        rpV[l] = VP[pref_i, l, off:off + nb, :]
        m_b = np.ones(nb, dtype=bool)
        res = forward_run(ids_b, replK=rpK,
                          maskK=m_b, replV=rpV,
                          maskV=m_b)
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        FRAC_L[l, k] = nr / nt if nt > 1e-12 \
            else float('nan')
        rec_kind.append(5)
        rec_body.append(b)
        rec_dlg.append(nr)
    if (l + 1) % 9 == 0:
        log('  layers %d/36 done (med frac=%.4f)'
            % (l + 1,
               float(np.nanmedian(FRAC_L[l]))))
prof_med = np.array([float(np.nanmedian(FRAC_L[l]))
                     for l in range(NL)])
argmax_layer = int(np.argmax(prof_med))
low_med = float(np.nanmedian(prof_med[:18]))
high_med = float(np.nanmedian(prof_med[18:]))
log('T3 profile: argmax L%d (med frac=%.4f), '
    'low=%.4f high=%.4f'
    % (argmax_layer, float(prof_med[argmax_layer]),
       low_med, high_med))

# a130/a131
a130_ok = bool(a130_fail == 0)
max_dlg = float(np.max(np.array(rec_dlg)))
a131_ok = bool(max_dlg >= T_GATE)
log('a130 integrity fails=%d ok=%s | a131 '
    'max||dlg||=%.4f ok=%s'
    % (a130_fail, a130_ok, max_dlg, a131_ok))

# ---------- verdict ----------
sig = bool(p_rep < 0.05)
if sig and obs_frac_rep >= FRAC_CARRY:
    verdict = 'kvpos_carries_qwen'
elif sig:
    verdict = 'kvpos_partial_qwen'
else:
    verdict = 'kvpos_null_qwen'
log('VERDICT: %s (sig=%s med frac_REP=%.4f)'
    % (verdict, sig, obs_frac_rep))

anchor_core_ok = bool(a116_ok and a126_ok
                      and a127_diff == 0.0
                      and a128_ok and a129_ok
                      and a130_ok and a131_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         bodies=np.array(BODIES),
         LENS=LENS,
         KPpre=KPpre, KPpost=KPpost, VP=VP, LG=LG,
         TT=TT,
         COS_ADD=COS_ADD, FRAC_ADD=FRAC_ADD,
         COS_REP=COS_REP, FRAC_REP=FRAC_REP,
         RESP_ADD=RESP_ADD, RESP_REP=RESP_REP,
         RESP_SCR=RESP_SCR,
         KILL_FRAC=KILL_FRAC, KILL_COS=KILL_COS,
         null_cos=null_cos, null_frac=null_frac,
         FRAC_L=FRAC_L, prof_med=prof_med,
         rec_kind=np.array(rec_kind),
         rec_body=np.array(rec_body),
         rec_dlg=np.array(rec_dlg),
         a126_diff=np.float64(a126_diff),
         a127_diff=np.float64(a127_diff),
         a128_diff=np.float64(a128_diff),
         a129_key=np.float64(a129_key),
         a129_vbit=np.float64(a129_vbit),
         a129_grp=np.float64(a129_grp),
         a130_fail=np.int64(a130_fail),
         a131_max_dlg=np.float64(max_dlg),
         a116_ok=np.bool_(a116_ok),
         a126_ok=np.bool_(a126_ok),
         a128_ok=np.bool_(a128_ok),
         a129_ok=np.bool_(a129_ok),
         a130_ok=np.bool_(a130_ok),
         a131_ok=np.bool_(a131_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_ADD': {'med_cos': obs_cos_add,
               'med_frac': obs_frac_add},
    'T2_REP': {'med_cos': obs_cos_rep,
               'med_frac': obs_frac_rep,
               'cos_per_pair': COS_REP.tolist(),
               'frac_per_pair': FRAC_REP.tolist()},
    'null': {'R': R_NULL, 'seed': SEED_NULL,
             'med_cos': float(np.median(null_cos)),
             'max_cos': float(null_cos.max()),
             'med_frac': float(np.nanmedian(
                 null_frac)),
             'p_rep': p_rep},
    'T4_SCR': {'med_kill_frac': med_kill_frac,
               'med_kill_cos': med_kill_cos},
    'T3_profile': {'argmax_layer': argmax_layer,
                   'med_frac_at_argmax': float(
                       prof_med[argmax_layer]),
                   'low_half_med': low_med,
                   'high_half_med': high_med},
    't_norms': {'med': float(np.median(NTT)),
                'max': float(NTT.max())},
    'anchors': {'a116_seals_ok': a116_ok,
                'a126_chain_diff': a126_diff,
                'a127_dup_diff': a127_diff,
                'a128_sham_diff': a128_diff,
                'a129_key_diff': a129_key,
                'a129_v_bit': a129_vbit,
                'a129_group_diff': a129_grp,
                'a130_fail': a130_fail,
                'a131_max_dlg': max_dlg,
                'anchor_core_ok': anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run3 authoritative (fp32; run1 crashed pre-anchor on a tokenization alignment assertion; run2 crashed mid-T4 on a repl length mismatch and its null was single-pair instead of the preregistered 24-pair median; see corrections)',
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
