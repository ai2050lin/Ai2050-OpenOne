# -*- coding: utf-8 -*-
# Phase 3047 - Omega-P44: joint multi-layer KV
# replay (fp32). Upper-bound test of the "KV
# routing carries the prefix effect" hypothesis.
# Prior results: single-layer V replay has no
# readout privilege (3045) and single-layer K
# replay carries only 0.08-0.30 pct of the prefix
# logit displacement (3046 T3b). Here the EXACT
# per-layer pre-norm KV displacements dpre(c,b,l)
# = capture(c,b,l) - capture(0,b,l) measured on
# the 48-prompt bank (verbatim 3042-3046) are
# injected SIMULTANEOUSLY at ALL 36 layers (kv7
# slice, k_proj / v_proj outputs, pre qk-norm)
# into the base-condition prompt, and removed
# from the prefix-condition prompt.
# Arms per pair k=(c,b), c in 1..3, b in 0..7,
# k = (c-1)*8+b (3046 order):
# K: +dpreK all layers on base; V: +dpreV;
# KV: both; RM: -both on the prefix prompt.
# T1 (capture): LG (48,), KPRE (48,36,128),
# VPRE (48,36,128) at target position.
# T2 (primary): resp = lg_inj - LG[base];
# cos(resp, t_bc) and frac = ||resp||/||t_bc||
# per pair; t_bc = LG(c,b) - LG(0,b).
# null: R=200 MC, random unit dirs (36,128)
# norm-matched per layer per pair, injected KV
# jointly; stat = med over 24 pairs of cos;
# p = P(null med >= obs med).
# T3 (dose, descriptive): layer subsets
# {3}, {20}, 0..17, 18..35 (KV for halves,
# K-only for {3}/{20} = 3046 T3b replication).
# T4 (removal, descriptive): rem_frac =
# ||lgRM - lgA||/||t_bc||; rem_cos =
# cos(lgRM - lgA, -t_bc); keys+values carry
# everything iff lgRM ~ lgB.
# verdict_tree: sig = p_cos < 0.05;
# sig AND med frac_KV >= 0.5 ->
# kvjoint_carries_qwen; sig AND med frac_KV
# < 0.5 -> kvjoint_partial_qwen; else ->
# kvjoint_null_qwen; single branch.
# anchors: a120 chain: KPRE[:,3,:]/[:,20,:] vs
# z46 KPpre3/KPpre20 bit 0.0; DPREK3/DPREK20
# vs z46 DPRE3/DPRE20 bit 0.0; a121 duplicate
# capture prompt0 (lg + KPRE + VPRE all 36)
# bit 0.0; a122 integrity over all 96 main
# injections: mod-orig == delta bit at every
# layer (K and V), non-target past positions
# bit 0.0 (full 36-layer K and V vs same-
# prompt reference), zero failures; a123 sham
# (state forced on, delta = 0, all layers):
# lg bit == base AND integ diffs bit 0.0;
# a124 chain-entry: max ||dlg|| over all
# recorded injections >= 0.05; a125 cross-
# phase: dose L3/K frac vs z46 frac_e3 and
# L20/K vs frac_e20, max |diff| <= 1e-9
# (bit-level expected, same fp32 pipeline).
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3047
NAME = 'omega_p44_kv_joint_replay_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 7
HDIM = 128
R_NULL = 200
SEED_NULL = 9850
SEED_MAIN = 3009
T_GATE = 0.05
FRAC_CARRY = 0.5
A125_GATE = 1e-9
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
            '3009); 48 prompts verbatim 3042-3046 '
            'bank; capture k_proj AND v_proj kv7 '
            'slices at the target position for ALL '
            '36 layers; joint injection of the '
            'exact per-layer pre-norm displacements',
    'question': '3047 A main line: does the prefix '
                'logit effect flow through the KV '
                'writes at the target position when '
                'ALL layers are replayed jointly? '
                '(i) capture the full per-layer '
                'pre-norm KV displacement field '
                '(T1); (ii) joint K / V / KV replay '
                'into the base prompt - does the '
                'response reproduce the prefix '
                'logit displacement in direction '
                '(cos, primary null) and magnitude '
                '(frac); (iii) layer dose subsets '
                '(T3, descriptive; {3}/{20} '
                'K-only replicate 3046 T3b); (iv) '
                'removal from the prefix prompt '
                '(T4, descriptive): does subtracting '
                'the KV displacement erase the '
                'prefix effect',
    'T1': 'capture: for each of the 48 prompts one '
          'fp32 prefill with capture hooks on all '
          '36 k_proj/v_proj outputs (kv7 slice) at '
          'the target position; LG (48,) double, '
          'KPRE/VPRE (48,36,128); a120 chain vs '
          'z46 bit 0.0; a121 duplicate prompt0',
    'T2': 'per pair k (c-major, 3046 order): arms '
          'K (+DPREK all layers), V (+DPREV), KV '
          '(both), RM (-both on prefix prompt) '
          'into the base prompt at the target '
          'position; resp = lg - LG[base] (RM: '
          'lg - LG[prefix]); cos(resp, t_bc) and '
          'frac = ||resp||/||t_bc||; PRIMARY stat '
          '= med over 24 pairs of cos_KV; null = '
          'R=%d MC random unit dirs (36,128) '
          'norm-matched per layer per pair, joint '
          'KV injection, seed %d + mc; p = '
          'P(null med cos >= obs); a122 integrity '
          'on all 96 main injections'
          % (R_NULL, SEED_NULL),
    'T3': 'dose subsets (descriptive): L3 K-only, '
          'L20 K-only (replicate 3046 T3b; a125 '
          'max |frac - frac_e| <= %g), low 0..17 '
          'KV, high 18..35 KV; frac and med cos '
          'per subset' % A125_GATE,
    'T4': 'removal (descriptive): rem_frac = '
          '||lgRM - lgA||/||t_bc||, rem_cos = '
          'cos(lgRM - lgA, -t_bc); keys+values '
          'carry the prefix effect iff lgRM '
          'approaches lgB (rem_frac ~ 1, rem_cos '
          '~ 1); if lgRM ~ lgA the KV writes '
          'carry nothing',
    'verdict_tree': 'sig = p_cos < 0.05; sig AND '
                    'med frac_KV >= 0.5 -> '
                    'kvjoint_carries_qwen; sig AND '
                    'med frac_KV < 0.5 -> '
                    'kvjoint_partial_qwen; else -> '
                    'kvjoint_null_qwen; single '
                    'branch',
    'anchors': 'a120 chain KPRE L3/L20 + DPREK3/'
               'DPREK20 vs z46 bit 0.0; a121 '
               'duplicate capture prompt0 bit 0.0; '
               'a122 integrity 96 main injections '
               '(mod == orig+delta bit-exact all layers '
               'K+V; non-target past positions bit '
               '0.0 full 36-layer K and V); a123 '
               'sham forced-on zero-delta bit '
               'identity; a124 chain-entry max '
               '||dlg|| >= 0.05; a125 dose '
               'replication vs z46 frac_e gate %g'
               % A125_GATE,
    'control': 'norm-matched random directions '
               'per layer (same joint injection '
               'machinery); removal arm; no other '
               'intervention',
    'statistics_discipline': 'obs and null on the '
                             'SAME scale (raw cos '
                             'over the same 24 '
                             'pairs, same machinery); '
                             'null never on '
                             'intervened quantities; '
                             'arrays pre-initialized; '
                             'verdict in one branch',
    'corrections': 'run1 completed with a122 '
                   'failing 96/96 (past_fail=0); '
                   'probe gpt5_temp/probe3047d '
                   'root-caused the CHECK '
                   'formulation, not the injection: '
                   'max|mod-orig-delta| = 1.86e-09 '
                   '(fp32 subtraction rounding of '
                   '(x+d)-x, rel 3.5e-08) while '
                   'mod == orig+delta is BIT-EXACT '
                   '(e_bit = 0.0) and the f32 cast '
                   'is exact (e_cast = 0.0); '
                   'integrity reformulated to the '
                   'bit-exact mod == orig+delta '
                   'check (subtraction form kept '
                   'as diagnostic a122_diag); '
                   'statistics unchanged and '
                   'reproduce bit-level under '
                   'frozen seeds; run2 '
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
NVOC = int(model.config.vocab_size)
log('model loaded fp32 (qk-norm confirmed, '
    'vocab=%d)' % NVOC)

SL = slice(KV_HEAD * HDIM, (KV_HEAD + 1) * HDIM)
stateK = {li: {'on': False, 'pos': -1,
               'delta': None} for li in range(NL)}
stateV = {li: {'on': False, 'pos': -1,
               'delta': None} for li in range(NL)}
capK = {li: {'rec': False, 'pos': -1,
             'orig': None, 'mod': None}
        for li in range(NL)}
capV = {li: {'rec': False, 'pos': -1,
             'orig': None, 'mod': None}
        for li in range(NL)}


def make_hook(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0, cp['pos'], SL] \
                .detach().clone()
        if st['on']:
            out[0, st['pos'], SL] += st['delta']
        if cp['rec']:
            cp['mod'] = out[0, cp['pos'], SL] \
                .detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_proj \
        .register_forward_hook(make_hook(stateK[li],
                                         capK[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(make_hook(stateV[li],
                                         capV[li]))


def reset_all():
    for li in range(NL):
        stateK[li]['on'] = False
        stateV[li]['on'] = False
        capK[li]['rec'] = False
        capV[li]['rec'] = False


def clear_caps():
    for li in range(NL):
        capK[li]['orig'] = None
        capK[li]['mod'] = None
        capV[li]['orig'] = None
        capV[li]['mod'] = None


def forward_cap(ids, pos):
    reset_all()
    clear_caps()
    for li in range(NL):
        capK[li].update(rec=True, pos=pos)
        capV[li].update(rec=True, pos=pos)
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    kp = np.stack([capK[li]['orig'].double().cpu()
                   .numpy() for li in range(NL)])
    vp = np.stack([capV[li]['orig'].double().cpu()
                   .numpy() for li in range(NL)])
    reset_all()
    return lg, kp, vp


def past_numpy(past):
    pk = {}
    pv = {}
    for li in range(NL):
        pk[li] = past.layers[li].keys[0] \
            .detach().double().cpu().numpy()
        pv[li] = past.layers[li].values[0] \
            .detach().double().cpu().numpy()
    return pk, pv


def forward_past(ids):
    reset_all()
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    pk, pv = past_numpy(out.past_key_values)
    return lg, pk, pv


def forward_inj(ids, pos, dK, dV, force_on=False,
                integ=False, want_past=False):
    reset_all()
    clear_caps()
    for li in range(NL):
        onk = force_on or float(
            np.max(np.abs(dK[li]))) > 0.0
        onv = force_on or float(
            np.max(np.abs(dV[li]))) > 0.0
        if onk:
            stateK[li].update(
                on=True, pos=pos,
                delta=torch.tensor(
                    np.ascontiguousarray(dK[li]),
                    dtype=torch.float32,
                    device='cuda'))
        if onv:
            stateV[li].update(
                on=True, pos=pos,
                delta=torch.tensor(
                    np.ascontiguousarray(dV[li]),
                    dtype=torch.float32,
                    device='cuda'))
        if integ:
            capK[li].update(rec=True, pos=pos)
            capV[li].update(rec=True, pos=pos)
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    res = {'lg': lg}
    if integ:
        ik = np.stack([(capK[li]['mod']
                        - capK[li]['orig'])
                       .double().cpu().numpy()
                       for li in range(NL)])
        iv = np.stack([(capV[li]['mod']
                        - capV[li]['orig'])
                       .double().cpu().numpy()
                       for li in range(NL)])
        res['integK'] = ik
        res['integV'] = iv
        ek = 0.0
        ev = 0.0
        for li in range(NL):
            dtk = stateK[li]['delta'] \
                if stateK[li]['on'] else None
            dtv = stateV[li]['delta'] \
                if stateV[li]['on'] else None
            if dtk is not None:
                ek = max(ek, float(
                    (capK[li]['mod']
                     - (capK[li]['orig']
                        + dtk)).abs().max()))
            else:
                ek = max(ek, float(
                    (capK[li]['mod']
                     - capK[li]['orig'])
                    .abs().max()))
            if dtv is not None:
                ev = max(ev, float(
                    (capV[li]['mod']
                     - (capV[li]['orig']
                        + dtv)).abs().max()))
            else:
                ev = max(ev, float(
                    (capV[li]['mod']
                     - capV[li]['orig'])
                    .abs().max()))
        res['bitK'] = ek
        res['bitV'] = ev
    reset_all()
    if want_past:
        res['pk'], res['pv'] = past_numpy(
            out.past_key_values)
    return res


# ---------- chain sources ----------
z46 = np.load(os.path.join(
    BASE, 'phase3046',
    'omega_p43_kfield_injection_qwen',
    'omega_p43_kfield_injection_qwen.npz'),
    allow_pickle=True)

# a116b: source seals 3037..3046
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
        (3046, 'omega_p43_kfield_injection_qwen')):
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

# ---------- assemble prompts (3046 verbatim) ----------
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
log('assembled %d prompts' % n_pr)

# ---------- T1 capture ----------
log('=== T1 capture (48 prompts x 36 layers) ===')
LG = [None] * n_pr
KPRE = np.zeros((n_pr, NL, HDIM))
VPRE = np.zeros((n_pr, NL, HDIM))
for i in range(n_pr):
    lg, kp, vp = forward_cap(assembled[i]['ids'],
                             assembled[i]['pos'])
    assert lg.shape[0] == NVOC
    LG[i] = lg
    KPRE[i] = kp
    VPRE[i] = vp
LG = np.stack(LG)
log('capture done')

# a121: duplicate capture prompt0
lg2, kp2, vp2 = forward_cap(assembled[0]['ids'],
                            assembled[0]['pos'])
a121_diff = float(np.max(np.abs(LG[0] - lg2)))
a121_diff = max(a121_diff, float(
    np.max(np.abs(KPRE[0] - kp2))))
a121_diff = max(a121_diff, float(
    np.max(np.abs(VPRE[0] - vp2))))
a121_diff = float(a121_diff)
log('a121 dup capture diff=%.3e' % a121_diff)

# a120: chain vs z46
a120_d1 = float(np.max(np.abs(
    KPRE[:, 3, :] - z46['KPpre3'])))
a120_d2 = float(np.max(np.abs(
    KPRE[:, 20, :] - z46['KPpre20'])))
a120_diff = max(a120_d1, a120_d2)

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)
NP_ = 24


def displacements(A):
    D = np.zeros((NP_,) + A.shape[1:])
    for k in range(NP_):
        ic = idx_of[(int(cidx[k]), int(bidx[k]),
                     False)]
        ib = idx_of[(0, int(bidx[k]), False)]
        D[k] = A[ic] - A[ib]
    return D


DPREK = displacements(KPRE)
DPREV = displacements(VPRE)
a120_d3 = float(np.max(np.abs(
    DPREK[:, 3, :] - z46['DPRE3'])))
a120_d4 = float(np.max(np.abs(
    DPREK[:, 20, :] - z46['DPRE20'])))
a120_diff = float(max(a120_diff, a120_d3, a120_d4))
a120_ok = bool(a120_diff == 0.0)
log('a120 chain Kpre3=%.3e Kpre20=%.3e '
    'DPRE3=%.3e DPRE20=%.3e ok=%s'
    % (a120_d1, a120_d2, a120_d3, a120_d4,
       a120_ok))

# targets and norms
t_targets = {}
for b in range(len(BODIES)):
    for c in (1, 2, 3):
        t_targets[(b, c)] = LG[idx_of[(c, b, False)]] \
            - LG[idx_of[(0, b, False)]]
TT = np.stack([t_targets[(int(bidx[k]),
                          int(cidx[k]))]
               for k in range(NP_)])
NTT = np.linalg.norm(TT, axis=1)
NORMK = np.linalg.norm(DPREK, axis=2)
NORMV = np.linalg.norm(DPREV, axis=2)
log('t norms: med=%.4f max=%.4f | dpreK norms '
    'med=%.4f' % (float(np.median(NTT)),
                  float(NTT.max()),
                  float(np.median(NORMK))))

# a123: sham (forced on, zero delta)
b0 = int(bidx[0])
base0 = idx_of[(0, b0, False)]
pos0 = assembled[base0]['pos']
res = forward_inj(assembled[base0]['ids'], pos0,
                  np.zeros((NL, HDIM)),
                  np.zeros((NL, HDIM)),
                  force_on=True, integ=True)
a123_diff = float(np.max(np.abs(
    res['lg'] - LG[base0])))
a123_diff = max(a123_diff, float(
    np.max(np.abs(res['integK']))))
a123_diff = max(a123_diff, float(
    np.max(np.abs(res['integV']))))
a123_diff = float(a123_diff)
a123_ok = bool(a123_diff == 0.0)
log('a123 sham diff=%.3e ok=%s'
    % (a123_diff, a123_ok))

# ---------- T2 main arms ----------
log('=== T2 main arms (24 pairs x 4) ===')
RESP = {'K': np.zeros((NP_, NVOC)),
        'V': np.zeros((NP_, NVOC)),
        'KV': np.zeros((NP_, NVOC)),
        'RM': np.zeros((NP_, NVOC))}
COS = {a: np.zeros(NP_) for a in RESP}
FRAC = {a: np.zeros(NP_) for a in RESP}
n_integ_fail = 0
n_past_fail = 0
a122_diag = 0.0
rec_kind = []
rec_body = []
rec_dlg = []
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b, False)]
    pref_i = idx_of[(c, b, False)]
    pos = assembled[base_i]['pos']
    ids_b = assembled[base_i]['ids']
    ids_p = assembled[pref_i]['ids']
    lgB_ref, pkB, pvB = forward_past(ids_b)
    lgA_ref, pkA, pvA = forward_past(ids_p)
    t = t_targets[(b, c)]
    nt = float(np.linalg.norm(t))
    arms = (
        ('K', ids_b, base_i, DPREK[k],
         np.zeros((NL, HDIM)), pkB, pvB),
        ('V', ids_b, base_i, np.zeros((NL, HDIM)),
         DPREV[k], pkB, pvB),
        ('KV', ids_b, base_i, DPREK[k], DPREV[k],
         pkB, pvB),
        ('RM', ids_p, pref_i, -DPREK[k], -DPREV[k],
         pkA, pvA))
    for an, ids, ref_i, dK, dV, pkref, pvref \
            in arms:
        res = forward_inj(ids, pos, dK, dV,
                          integ=True, want_past=True)
        # integrity: mod-orig == delta bit
        e1 = float(res['bitK'])
        e2 = float(res['bitV'])
        e1d = float(np.max(np.abs(
            res['integK'] - dK)))
        e2d = float(np.max(np.abs(
            res['integV'] - dV)))
        a122_diag = max(a122_diag, e1d, e2d)
        # non-target past positions bit 0.0
        e3 = 0.0
        for li in range(NL):
            e3 = max(e3, float(np.max(np.abs(
                res['pk'][li][:, :pos, :]
                - pkref[li][:, :pos, :]))))
            e3 = max(e3, float(np.max(np.abs(
                res['pv'][li][:, :pos, :]
                - pvref[li][:, :pos, :]))))
        if e1 != 0.0 or e2 != 0.0 or e3 != 0.0:
            n_integ_fail += int(e1 != 0.0
                                or e2 != 0.0)
            n_past_fail += int(e3 != 0.0)
        r = res['lg'] - LG[ref_i]
        nr = float(np.linalg.norm(r))
        cs = float(r @ t) / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
        RESP[an][k] = r
        COS[an][k] = cs
        FRAC[an][k] = nr / nt if nt > 1e-12 \
            else float('nan')
        rec_kind.append({'K': 1, 'V': 2, 'KV': 3,
                         'RM': 4}[an])
        rec_body.append(b)
        rec_dlg.append(nr)
    if (k + 1) % 8 == 0:
        log('  arms %d/24 done' % (k + 1))
a122_ok = bool(n_integ_fail == 0
               and n_past_fail == 0)
log('a122 integrity: integ_fail=%d past_fail=%d '
    'ok=%s' % (n_integ_fail, n_past_fail,
               a122_ok))

obs_cos = {a: float(np.median(COS[a]))
           for a in RESP}
obs_frac = {a: float(np.median(FRAC[a]))
            for a in RESP}
log('T2 med cos: K=%.4f V=%.4f KV=%.4f RM=%.4f'
    % (obs_cos['K'], obs_cos['V'],
       obs_cos['KV'], obs_cos['RM']))
log('T2 med frac: K=%.4f V=%.4f KV=%.4f RM=%.4f'
    % (obs_frac['K'], obs_frac['V'],
       obs_frac['KV'], obs_frac['RM']))

# ---------- T3 dose subsets ----------
log('=== T3 dose subsets ===')
SUBSETS = (('L3', (3,), 'K'), ('L20', (20,), 'K'),
           ('low', tuple(range(18)), 'KV'),
           ('high', tuple(range(18, NL)), 'KV'))
dose_frac = {}
dose_cos = {}
for sn, lay, kind in SUBSETS:
    fs = np.zeros(NP_)
    cs_ = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b, False)]
        pos = assembled[base_i]['pos']
        dK = np.zeros((NL, HDIM))
        dV = np.zeros((NL, HDIM))
        if kind in ('K', 'KV'):
            dK[list(lay)] = DPREK[k, list(lay)]
        if kind in ('V', 'KV'):
            dV[list(lay)] = DPREV[k, list(lay)]
        res = forward_inj(assembled[base_i]['ids'],
                          pos, dK, dV)
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        nt = float(np.linalg.norm(
            t_targets[(b, c)]))
        fs[k] = nr / nt if nt > 1e-12 else float('nan')
        cs_[k] = float(r @ t_targets[(b, c)]) \
            / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
        rec_kind.append(5)
        rec_body.append(b)
        rec_dlg.append(nr)
    dose_frac[sn] = fs
    dose_cos[sn] = cs_
    log('  %s (%s): med frac=%.4f med cos=%.4f'
        % (sn, kind, float(np.nanmedian(fs)),
           float(np.median(cs_))))

# a125: cross-phase replication vs z46
def nan_maxdiff(a, b):
    same_nan = np.isnan(a) == np.isnan(b)
    d = np.where(np.isnan(a) | np.isnan(b),
                 0.0, np.abs(a - b))
    return float(np.max(d)), bool(same_nan.all())


a125_d1, a125_n1 = nan_maxdiff(
    dose_frac['L3'], z46['frac_e3'])
a125_d2, a125_n2 = nan_maxdiff(
    dose_frac['L20'], z46['frac_e20'])
a125_diff = max(a125_d1, a125_d2)
a125_ok = bool(a125_diff <= A125_GATE
               and a125_n1 and a125_n2)
log('a125 dose replication: L3=%.3e L20=%.3e '
    'nan_ok=%s/%s ok=%s'
    % (a125_d1, a125_d2, a125_n1, a125_n2,
       a125_ok))

# ---------- null MC ----------
log('=== null MC (R=%d joint KV random) ==='
    % R_NULL)
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
        pos = assembled[base_i]['pos']
        RK = rng.standard_normal((NL, HDIM))
        RK = RK / np.linalg.norm(
            RK, axis=1)[:, None]
        RV = rng.standard_normal((NL, HDIM))
        RV = RV / np.linalg.norm(
            RV, axis=1)[:, None]
        dK = RK * NORMK[k][:, None]
        dV = RV * NORMV[k][:, None]
        res = forward_inj(
            assembled[base_i]['ids'], pos,
            dK, dV)
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        t = t_targets[(b, c)]
        nt = float(np.linalg.norm(t))
        csm[k] = float(r @ t) / (nr * nt) \
            if nr > 1e-12 and nt > 1e-12 else 0.0
        frm[k] = nr / nt if nt > 1e-12 \
            else float('nan')
        rec_kind.append(6)
        rec_body.append(b)
        rec_dlg.append(nr)
    null_cos[mc] = float(np.median(csm))
    null_frac[mc] = float(np.nanmedian(frm))
    if (mc + 1) % 40 == 0:
        log('  mc %d/%d (null med cos=%.4f)'
            % (mc + 1, R_NULL, null_cos[mc]))
p_cos = float((1 + int(np.sum(
    null_cos >= obs_cos['KV']))) / (R_NULL + 1))
log('T2 KV: obs med cos=%.4f null med cos '
    'med=%.4f max=%.4f p=%.5f'
    % (obs_cos['KV'], float(np.median(null_cos)),
       float(null_cos.max()), p_cos))

# a124: chain-entry gate
max_dlg = float(np.max(np.array(rec_dlg)))
a124_ok = bool(max_dlg >= T_GATE)
log('a124 chain-entry max||dlg||=%.4f ok=%s'
    % (max_dlg, a124_ok))

# ---------- verdict ----------
sig = bool(p_cos < 0.05)
if sig and obs_frac['KV'] >= FRAC_CARRY:
    verdict = 'kvjoint_carries_qwen'
elif sig:
    verdict = 'kvjoint_partial_qwen'
else:
    verdict = 'kvjoint_null_qwen'
log('VERDICT: %s (sig=%s med frac_KV=%.4f)'
    % (verdict, sig, obs_frac['KV']))

anchor_core_ok = bool(a120_ok and a121_diff == 0.0
                      and a122_ok and a123_ok
                      and a124_ok and a125_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         bodies=np.array(BODIES),
         KPRE=KPRE, VPRE=VPRE, LG=LG,
         DPREK=DPREK, DPREV=DPREV,
         NORMK=NORMK, NORMV=NORMV, TT=TT,
         RESP_K=RESP['K'], RESP_V=RESP['V'],
         RESP_KV=RESP['KV'], RESP_RM=RESP['RM'],
         COS_K=COS['K'], COS_V=COS['V'],
         COS_KV=COS['KV'], COS_RM=COS['RM'],
         FRAC_K=FRAC['K'], FRAC_V=FRAC['V'],
         FRAC_KV=FRAC['KV'], FRAC_RM=FRAC['RM'],
         dose_frac_L3=dose_frac['L3'],
         dose_frac_L20=dose_frac['L20'],
         dose_frac_low=dose_frac['low'],
         dose_frac_high=dose_frac['high'],
         dose_cos_L3=dose_cos['L3'],
         dose_cos_L20=dose_cos['L20'],
         dose_cos_low=dose_cos['low'],
         dose_cos_high=dose_cos['high'],
         null_cos=null_cos, null_frac=null_frac,
         rec_kind=np.array(rec_kind),
         rec_body=np.array(rec_body),
         rec_dlg=np.array(rec_dlg),
         n_integ_fail=np.int64(n_integ_fail),
         n_past_fail=np.int64(n_past_fail),
         a122_diag=np.float64(a122_diag),
         a120_diff=np.float64(a120_diff),
         a121_diff=np.float64(a121_diff),
         a123_diff=np.float64(a123_diff),
         a125_diff=np.float64(a125_diff),
         a116_ok=np.bool_(a116_ok),
         a122_ok=np.bool_(a122_ok),
         a124_ok=np.bool_(a124_ok),
         a125_ok=np.bool_(a125_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_arms': {
        'med_cos': obs_cos, 'med_frac': obs_frac,
        'cos_KV_per_pair': COS['KV'].tolist(),
        'frac_KV_per_pair': FRAC['KV'].tolist()},
    'null': {'R': R_NULL, 'seed': SEED_NULL,
             'med_cos': float(np.median(null_cos)),
             'max_cos': float(null_cos.max()),
             'med_frac': float(np.nanmedian(
                 null_frac)),
             'p_cos': p_cos},
    'T3_dose': {sn: {'kind': kd,
                     'med_frac': float(
                         np.nanmedian(dose_frac[sn])),
                     'med_cos': float(
                         np.median(dose_cos[sn]))}
                for sn, la, kd in SUBSETS},
    'T4_removal': {'med_rem_frac': float(
        np.median(FRAC['RM'])),
        # rem_cos = cos(lgRM - lgA, -t_bc);
        # +1 = full erasure (lgRM ~ lgB)
        'med_rem_cos': float(
            np.median(-COS['RM']))},
    'anchors': {'a116_seals_ok': a116_ok,
                'a120_chain_diff': a120_diff,
                'a121_dup_diff': a121_diff,
                'a122_ok': a122_ok,
                'n_integ_fail': n_integ_fail,
                'n_past_fail': n_past_fail,
                'a122_diag_subform': float(
                    a122_diag),
                'a123_sham_diff': a123_diff,
                'a124_max_dlg': max_dlg,
                'a125_dose_diff': a125_diff,
                'a125_ok': a125_ok,
                'anchor_core_ok': anchor_core_ok},
    't_norms': {'med': float(np.median(NTT)),
                'max': float(NTT.max())},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run2 authoritative (fp32; run1 a122 failed on a subtraction-rounding artifact of the check itself, probe-root-caused, injection bit-exact; see corrections)',
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
