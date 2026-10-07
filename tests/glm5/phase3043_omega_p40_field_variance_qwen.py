# -*- coding: utf-8 -*-
# Phase 3043 - Omega-P40: field variance decomposition
# Phase 3042 (stylefield_global_identity_qwen) established:
# a prefix displaces the V write of DIFFERENT downstream body
# sentences in a shared direction (4.5x Gaussian), with a
# prefix-specific component on top, identity-channel leak,
# flat dose, and a semantic-overlap candidate (topic prefix
# 'Regarding the weather,' x weather body had the largest
# displacement). This phase decomposes the field:
# (T1) two-way additive variance decomposition of the
#     displacement matrix Delta_{c,b} (3 prefixes x 8 old
#     bodies, 128-dim): OLS with intercept + prefix dummies
#     + body dummies (no interaction); SS_prefix = marginal
#     gain of the prefix block (full fit minus body-only
#     fit), SS_body symmetric; fractions of the centered
#     total SS; null = exact label permutation (prefix
#     labels for SS_prefix, body labels for SS_body,
#     50000 each) - separates SHARED FIELD (prefix main
#     effect) from CONTENT INTERACTION (body main effect).
# (T2) content-interaction outlier: obs = |Delta(topic,
#     weather-body)| among the 24 displacement norms;
#     null = prefix-label permutation (body labels fixed),
#     stat = norm of the (label-3, body-0) cell; tests
#     whether the semantically overlapping pair is an
#     outlier beyond its marginal effects.
# (T3) field-direction commonality: med |cos| among the
#     three prefix-mean directions alpha_c (old bodies) vs
#     size-matched Gaussian null - is there ONE field axis
#     plus per-prefix identity, or fully separate fields?
# (T4) out-of-bank generalization: 4 NEW body sentences
#     (not in the 3037 bank) x the same 3 prefixes;
#     alignment cos(Delta_new, alpha_c) vs Gaussian null -
#     does the field estimated on old bodies transfer?
# (T5) layer control: T1 and T4 repeated at L20.
# Extraction verbatim 3042 (8 old bodies x 4 conditions,
# bit-anchored vs 3037 npz and 3042 npz) plus 16 new
# prompts. PREREG frozen below BEFORE any observation.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3043
NAME = 'omega_p40_field_variance_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
HID = 2560
KV_HEAD = 7
LAYERS_EX = (3, 20)
N_PERM = 50000
N_RND = 20000
SEED_MAIN = 3009
SEED_PFX = 9070
SEED_BODY = 9071
SEED_T2 = 9072
SEED_T3 = 9073
SEED_T4 = 9074
SEED_L20 = 9075
SEED_L20_T4 = 9076
DEGEN_NORM = 1e-6
A95_GATE = 0.15
A98_GATE = 1e-12

BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
BODY_IDX = (0, 1, 3, 6, 7, 8, 9, 10)
NEW_BODIES = (
    'The game was delayed, so',
    'She stayed home because',
    'The engine failed, therefore',
    'He kept smiling, although',)
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')
NEW_TARGETS = ('so', 'because', 'therefore',
               'although')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')
WEATHER_PAIR = (3, 0)  # (topic prefix, weather body)

PREREG = {
    'mode': 'pure prefill + KV-cache read (no '
            'intervention, eager attention, bf16, seed '
            '3009); 8 old bodies x 4 conditions '
            '(verbatim 3042, same assembly order) + '
            '4 NEW bodies x 4 conditions = 48 prompts; '
            'V read at the target position from '
            'past.layers[li].values[0, kv7]; layers '
            '(3, 20); V PRIMARY',
    'question': 'field variance decomposition (3043 A '
                'main line): of the prefix-induced '
                'displacement field, how much is '
                'prefix main effect (shared field) '
                'vs body main effect (content '
                'interaction) vs residual; is the '
                'semantically overlapping pair an '
                'outlier; do the three prefix-mean '
                'directions share a common axis; and '
                'does the field generalize to NEW '
                'body sentences?',
    'displacement': 'Delta(c,b) = V_c[b] - V_base[b] at '
                    'the target occurrence, per layer; '
                    'DEGENERATE RULE: |Delta| < 1e-6 '
                    'excluded from all statistics '
                    '(3040 lesson)',
    'T1': 'two-way additive variance decomposition: '
          'design = intercept + prefix dummies + body '
          'dummies (balanced 3x8, no interaction), '
          'OLS per dimension; SS_prefix = ||fit_full||'
          '^2 - ||fit_body_only||^2 (marginal gain of '
          'the prefix block), SS_body symmetric, '
          'total = centered ||D - mean||^2; '
          'fractions f_prefix, f_body, f_resid = 1 - '
          'explained; null = exact label permutation '
          '(prefix labels N_PERM=%d seed %d; body '
          'labels N_PERM=%d seed %d), one-sided p = '
          'P(null >= obs) on each fraction; shared-'
          'field claim iff p_prefix<0.05; content-'
          'interaction claim iff p_body<0.05'
          % (N_PERM, SEED_PFX, N_PERM, SEED_BODY),
    'T2': 'content-interaction outlier: obs = norm of '
          'the (topic-prefix, weather-body) cell; '
          'null = WITHIN-BLOCK body-label permuta- '
          'tion (the 8 rows of the topic-prefix '
          'block; exactly one row carries body 0 '
          'per draw, so the cell always exists); '
          'N_PERM=%d seed %d; '
          'one-sided p = P(null >= obs); outlier '
          'iff p<0.05' % (N_PERM, SEED_T2),
    'T3': 'field-direction commonality: alpha_c = mean '
          'old-body displacement per prefix '
          '(c=1,2,3), normalized; stat = med |cos| '
          'over the 3 pairs vs size-matched Gaussian '
          'null (N_RND=%d seed %d); common axis iff '
          'p<0.05' % (N_RND, SEED_T3),
    'T4': 'out-of-bank generalization: 4 NEW bodies '
          '(not in the 3037 bank, one logic target '
          'each) x 3 prefixes; stat = med |cos| '
          '(Delta_new, alpha_c) over 12 pairs vs '
          'size-matched Gaussian null (N_RND=%d seed '
          '%d); field transfer iff p<0.05'
          % (N_RND, SEED_T4),
    'T5': 'layer control: T1 (both permutations, seed '
          '%d) and T4 (seed %d) repeated at L20'
          % (SEED_L20, SEED_L20_T4),
    'verdict_tree': 'prefixfx = (p_T1_prefix<0.05); '
                    'bodyfx = (p_T1_body<0.05); flags '
                    'reported but NOT in the verdict '
                    'name: contenthit = (p_T2<0.05), '
                    'fieldcommon = (p_T3<0.05), '
                    'generalizes = (p_T4<0.05); '
                    'prefixfx AND bodyfx -> '
                    'fieldvar_prefix_plus_body_qwen; '
                    'prefixfx AND NOT bodyfx -> '
                    'fieldvar_prefix_dominant_qwen; '
                    'else -> fieldvar_null_qwen; '
                    'single branch assignment',
    'anchors': 'a93 duplicate prefill prompt0 K3/V3 '
               'bit-identical (0.0); a94 full duplicate '
               'extraction all 48 prompts max abs diff '
               '0.0; a95 manual final-norm+lm_head '
               'recompute (top-2 identity AND '
               'max|dlogit| <= 0.15; near-tie skip '
               'note if gap<0.05); a96 source seals '
               '3037/3038/3039/3040/3041/3042 sha8 '
               'match; a97 cross-phase bit anchor: '
               'base-condition old-body targets vs '
               '3037 npz (word, prompt, pos), '
               'max|dV3| == 0.0, 8/8; a98 '
               'displacement-chain anchor vs 3042 npz: '
               'D3 AND D20 bit-identical over the 24 '
               'old displacements (same build order); a99 '
               'channel-basis anchor vs 3041 npz: '
               'recomputed SIT3 AND SIT20 max diff '
               '<= 1e-12',
    'control': 'exact label permutations for T1/T2; '
               'size-matched Gaussian nulls for T3/'
               'T4; no intervention',
    'statistics_discipline': 'all arrays pre-'
                             'initialized; degenerate-'
                             'norm gate on Delta; '
                             'marginal SS definition '
                             'fixed in prereg (full '
                             'fit minus reduced fit); '
                             'verdict criteria in one '
                             'branch assignment; '
                             'small-n caveat: 24 old '
                             'displacements per layer, '
                             'exploratory margin',
    'corrections': 'run2 crashed mid-T2 '
                   '(T1 statistics were observed and '
                   'unchanged: f_prefix/f_body printed; '
                   'T3/T4/T5 not yet observed): the '
                   'prefix-label permutation leaves '
                   'the (label-3, body-0) cell empty '
                   'with probability (2/3)^3 -> IndexError; '
                   'T2 null redefined as within-block '
                   'body-label permutation (exact, cell '
                   'always exists); run1 crashed '
                   'PRE-verdict (anchor stage only, no '
                   'test statistic observed): '
                   'a98 referenced V3/V20 keys '
                   'absent from the 3042 npz '
                   '(which stores displacement '
                   'arrays D3/D20); corrected to '
                   'a displacement-chain anchor '
                   'comparing our D3/D20 vs the '
                   '3042 npz D3/D20 bit-exact '
                   '(stronger derived-quantity '
                   'chain check); verdict tree '
                   'unchanged',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


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
    MODEL_DIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.hidden_size) == HID
HDIM = int(model.config.head_dim) \
    if hasattr(model.config, 'head_dim') else 128
EPS = float(getattr(model.config, 'rms_norm_eps',
                    1e-6))
log('model loaded head_dim=%d eps=%g'
    % (HDIM, EPS))

W_U = model.lm_head.weight.detach().float() \
    .cpu().numpy()
assert W_U.shape == (int(model.config.vocab_size),
                     HID), W_U.shape

state_fin = {'on': False}
fin_cap = {}


def pre_norm(module, args, kwargs):
    if state_fin['on']:
        fin_cap['x'] = args[0][:, -1, :] \
            .detach().float().cpu().numpy().copy()
    return None


model.model.norm.register_forward_pre_hook(
    pre_norm, with_kwargs=True)


def prefill_extract(ids, cap_fin=False):
    fin_cap.pop('x', None)
    state_fin['on'] = bool(cap_fin)
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
    state_fin['on'] = False
    past = out.past_key_values
    lg = out.logits[0, -1].detach() \
        .double().cpu().numpy()
    l = lg - lg.max()
    p = np.exp(l)
    p = p / p.sum()
    kv = {}
    for li in LAYERS_EX:
        k = past.layers[li].keys[0, KV_HEAD] \
            .detach().float().cpu().numpy()
        v = past.layers[li].values[0, KV_HEAD] \
            .detach().float().cpu().numpy()
        kv[li] = (k, v)
    xf = fin_cap['x'][0].copy() \
        if 'x' in fin_cap else None
    return p, lg, kv, xf


# ---------- assemble prompts ----------
word_tok = {}
for w in set(TARGETS) | set(NEW_TARGETS):
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    assert len(wi) == 1, (w, wi)
    word_tok[w] = int(wi[0])
assembled = []
for bi in range(len(BODIES)):
    for ci in range(len(PREFIXES)):
        body = BODIES[bi]
        s = (PREFIXES[ci] + ' ' + body) \
            if PREFIXES[ci] else body
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
        body = NEW_BODIES[bi]
        s = (PREFIXES[ci] + ' ' + body) \
            if PREFIXES[ci] else body
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
log('assembled %d prompts (%d old + %d new)'
    % (n_pr, n_old, n_pr - n_old))

# a93: duplicate prefill of prompt0
p_b0, lg_b0, kv_b0, xf_b0 = prefill_extract(
    assembled[0]['ids'], cap_fin=True)
p_d0, lg_d0, kv_d0, _ = prefill_extract(
    assembled[0]['ids'])
a93_diff = 0.0
for li in LAYERS_EX:
    for a, b in ((kv_b0[li][0], kv_d0[li][0]),
                 (kv_b0[li][1], kv_d0[li][1])):
        a93_diff = max(a93_diff, float(
            np.max(np.abs(a - b))))
a93_diff = float(a93_diff)

# ---------- main extraction ----------
Ps = [None] * n_pr
KVs = {li: [None] * n_pr for li in LAYERS_EX}
for i in range(n_pr):
    p, _, kv, _ = prefill_extract(assembled[i]['ids'])
    Ps[i] = p
    for li in LAYERS_EX:
        KVs[li][i] = kv[li]

# a94: full duplicate extraction
a94_diff = 0.0
for i in range(n_pr):
    p2, _, kv2, _ = prefill_extract(
        assembled[i]['ids'])
    a94_diff = max(a94_diff, float(np.max(
        np.abs(Ps[i] - p2))))
    for li in LAYERS_EX:
        for a, b in zip(KVs[li][i], kv2[li]):
            a94_diff = max(a94_diff, float(
                np.max(np.abs(a - b))))
a94_diff = float(a94_diff)
log('a93=%.3e a94=%.3e' % (a93_diff, a94_diff))

# a95: manual final-norm+lm_head recompute (prompt0)
w_norm = model.model.norm.weight.detach() \
    .float().cpu().numpy()
h_fin = xf_b0
var = float((h_fin ** 2).mean())
hn = h_fin / np.sqrt(var + EPS)
lg_man = (W_U @ (hn * w_norm)).astype(np.float64)
a95_maxdiff = float(np.max(np.abs(lg_man - lg_b0)))
ord_m = np.argsort(-lg_man)
ord_o = np.argsort(-lg_b0)
srt = np.sort(lg_b0)
gap0 = float(srt[-1] - srt[-2])
if gap0 < 0.05:
    a95_top2_ok = True
    a95_note = 'near-tie skip (gap=%.4f)' % gap0
else:
    a95_top2_ok = bool(ord_m[0] == ord_o[0]
                       and ord_m[1] == ord_o[1])
    a95_note = ''
a95_ok = bool(a95_top2_ok
              and a95_maxdiff <= A95_GATE)
log('a95 top2=%s maxdiff=%.4f (gate %.2f) %s'
    % (a95_top2_ok, a95_maxdiff, A95_GATE,
       a95_note))

# a96: source seals
a96_detail = []
for ph, nm in (
        (3037, 'omega_p34_kv_situational_'
               'specificity_qwen'),
        (3038, 'omega_p35_reentrant_readout_qwen'),
        (3039, 'omega_p36_direct_logistic_'
               'replication_qwen'),
        (3040, 'omega_p37_situational_'
               'component_qwen'),
        (3041, 'omega_p38_situational_axis_qwen'),
        (3042, 'omega_p39_style_field_probe_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a96_detail.append(bool(s == sealj['result_sha256_8']))
a96_ok = bool(a96_detail) and all(a96_detail)

# ---------- occurrence vectors ----------
V3o = np.zeros((n_pr, HDIM))
V20o = np.zeros((n_pr, HDIM))
for i in range(n_pr):
    pos = assembled[i]['pos']
    V3o[i] = KVs[3][i][1][pos]
    V20o[i] = KVs[20][i][1][pos]

# a97: cross-phase bit anchor vs 3037 npz (old base)
z37 = np.load(os.path.join(
    BASE, 'phase3037',
    'omega_p34_kv_situational_specificity_qwen',
    'omega_p34_kv_situational_specificity_qwen.npz'),
    allow_pickle=True)
w37 = [str(x) for x in z37['occ_word']]
pr37 = [int(x) for x in z37['occ_prompt']]
po37 = [int(x) for x in z37['occ_pos']]
V37 = z37['V3']
ref = {}
for i in range(len(w37)):
    ref[(w37[i], pr37[i], po37[i])] = V37[i]
a97_diff = 0.0
a97_matched = 0
for i in range(n_old):
    if assembled[i]['cond'] != 0:
        continue
    bi = assembled[i]['body']
    key = (TARGETS[bi], int(BODY_IDX[bi]),
           int(assembled[i]['pos']))
    if key not in ref:
        continue
    a97_diff = max(a97_diff, float(np.max(
        np.abs(V3o[i] - ref[key]))))
    a97_matched += 1
a97_diff = float(a97_diff)
a97_ok = bool(a97_matched == len(BODIES)
              and a97_diff == 0.0)
log('a97 matched=%d/%d max|dV3|=%.3e'
    % (a97_matched, len(BODIES), a97_diff))

# a98: full old-bank anchor vs 3042 npz
z42 = np.load(os.path.join(
    BASE, 'phase3042',
    'omega_p39_style_field_probe_qwen',
    'omega_p39_style_field_probe_qwen.npz'),
    allow_pickle=True)

# a99: channel basis from 3041 npz
z41 = np.load(os.path.join(
    BASE, 'phase3041',
    'omega_p38_situational_axis_qwen',
    'omega_p38_situational_axis_qwen.npz'),
    allow_pickle=True)
V3_41 = z41['V3']
V20_41 = z41['V20']
occ_w_41 = z41['occ_w']
n_types = int(occ_w_41.max()) + 1


def basis_from(V41):
    WM = np.zeros((n_types, HDIM))
    for wi in range(n_types):
        WM[wi] = V41[occ_w_41 == wi].mean(axis=0)
    U, S, Vt = np.linalg.svd(WM,
                             full_matrices=False)
    r = int(np.sum(S > 1e-8 * S[0]))
    B = Vt[:r].T.copy()
    orth = float(np.max(np.abs(
        B.T @ B - np.eye(r))))
    return B, orth, r


B3, o3, r3 = basis_from(V3_41)
B20, o20, r20 = basis_from(V20_41)
SIT3_41 = z41['SIT3']
SIT20_41 = z41['SIT20']
a99_d3 = float(np.max(np.abs(
    V3_41 - (V3_41 @ B3) @ B3.T - SIT3_41)))
a99_d20 = float(np.max(np.abs(
    V20_41 - (V20_41 @ B20) @ B20.T - SIT20_41)))
a99_ok = bool(max(a99_d3, a99_d20) <= A98_GATE)
log('a99 dSIT3=%.3e dSIT20=%.3e ok=%s'
    % (a99_d3, a99_d20, a99_ok))

# ---------- displacements (old + new) ----------
idx_of = {}
for i in range(n_pr):
    idx_of[(assembled[i]['cond'],
            assembled[i]['body'],
            assembled[i]['new'])] = i


def build_D(new_flag):
    rows = []
    conds = []
    bods = []
    for ci in range(1, len(PREFIXES)):
        for bi in range(len(BODIES if not new_flag
                                else NEW_BODIES)):
            ic = idx_of[(ci, bi, new_flag)]
            ib = idx_of[(0, bi, new_flag)]
            rows.append(V3o[ic] - V3o[ib])
            conds.append(ci)
            bods.append(bi)
    return np.array(rows), np.array(conds,
                                    dtype=np.int64), \
        np.array(bods, dtype=np.int64)


D3_old, c_old, b_old = build_D(False)
D3_new, c_new, b_new = build_D(True)
ok_old = np.linalg.norm(D3_old, axis=1) >= DEGEN_NORM
ok_new = np.linalg.norm(D3_new, axis=1) >= DEGEN_NORM
nD_old = D3_old.shape[0]
nD_new = D3_new.shape[0]
log('old displacements %d (degenerate %d); new %d '
    '(degenerate %d)'
    % (nD_old, int((~ok_old).sum()), nD_new,
       int((~ok_new).sum())))


def fit_additive(D, conds, bods):
    """OLS additive fit; marginal SS fractions."""
    n = D.shape[0]
    cols = [np.ones(n)]
    for c in (2, 3):
        cols.append((conds == c).astype(np.float64))
    for b in range(1, D.shape[0] // 3):
        cols.append((bods == b).astype(np.float64))
    X = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(X, D, rcond=None)
    fit_full = X @ beta
    Xb = X[:, [0] + list(range(3, X.shape[1]))]
    bb, *_ = np.linalg.lstsq(Xb, D, rcond=None)
    fit_body = Xb @ bb
    Xp = X[:, :3]
    bp, *_ = np.linalg.lstsq(Xp, D, rcond=None)
    fit_pref = Xp @ bp
    Dm = D - D.mean(axis=0, keepdims=True)
    tot = float((Dm ** 2).sum())
    ss_p = float(((fit_full ** 2).sum()
                  - (fit_body ** 2).sum()))
    ss_b = float(((fit_full ** 2).sum()
                  - (fit_pref ** 2).sum()))
    return ss_p, ss_b, tot


def t1_tests(D, conds, bods, seed_pfx, seed_body):
    ss_p, ss_b, tot = fit_additive(D, conds, bods)
    f_p = ss_p / max(tot, 1e-30)
    f_b = ss_b / max(tot, 1e-30)
    rngp = np.random.default_rng(seed_pfx)
    null_p = np.zeros(N_PERM)
    for it in range(N_PERM):
        pc = rngp.permutation(conds)
        s1, s2, t = fit_additive(D, pc, bods)
        null_p[it] = s1 / max(t, 1e-30)
    p_pfx = float(np.mean(null_p >= f_p))
    rngb = np.random.default_rng(seed_body)
    null_b = np.zeros(N_PERM)
    for it in range(N_PERM):
        pb = rngb.permutation(bods)
        s1, s2, t = fit_additive(D, conds, pb)
        null_b[it] = s2 / max(t, 1e-30)
    p_body = float(np.mean(null_b >= f_b))
    return f_p, f_b, p_pfx, p_body


log('=== T1 variance decomposition (L3) ===')
f_p3, f_b3, p_pfx3, p_body3 = t1_tests(
    D3_old, c_old, b_old, SEED_PFX, SEED_BODY)
log('T1: f_prefix=%.4f p=%.5f | f_body=%.4f p=%.5f '
    '(resid=%.4f)'
    % (f_p3, p_pfx3, f_b3, p_body3,
       1.0 - min(f_p3 + f_b3, 1.0)))

log('=== T2 content-interaction outlier (L3) ===')
nrmD = np.linalg.norm(D3_old, axis=1)
pos_wb = int(np.where((c_old == WEATHER_PAIR[0])
                      & (b_old == WEATHER_PAIR[1]))
             [0][0])
obs_t2 = float(nrmD[pos_wb])
rng2 = np.random.default_rng(SEED_T2)
blk = np.where(c_old == WEATHER_PAIR[0])[0]
assert len(blk) == 8
null2 = np.zeros(N_PERM)
for it in range(N_PERM):
    pb = rng2.permutation(b_old[blk])
    j = int(np.where(
        pb == WEATHER_PAIR[1])[0][0])
    null2[it] = float(nrmD[blk[j]])
p_t2 = float(np.mean(null2 >= obs_t2))
log('T2: obs=%.4f (rank %d/24) null med=%.4f p=%.5f'
    % (obs_t2, int(np.sum(nrmD >= obs_t2)),
       float(np.median(null2)), p_t2))
log('T2 norms by cell: %s' % json.dumps(
    {'c%d_b%d' % (int(c_old[k]), int(b_old[k])):
     float(nrmD[k]) for k in range(nD_old)}))

log('=== T3 field-direction commonality (L3) ===')
ALPHAS = []
for c in (1, 2, 3):
    m = c_old == c
    a = D3_old[m].mean(axis=0)
    n = np.linalg.norm(a)
    ALPHAS.append(a / n if n > 0 else a)
A = np.stack(ALPHAS)
iu3 = np.triu_indices(3, 1)
obs_t3 = float(np.median(np.abs(
    A @ A.T)[iu3]))
rng3 = np.random.default_rng(SEED_T3)
null3 = np.zeros(N_RND)
for it in range(N_RND):
    G = rng3.standard_normal((3, HDIM))
    Gn = G / np.maximum(
        np.linalg.norm(G, axis=1), 1e-12)[:, None]
    null3[it] = float(np.median(np.abs(
        Gn @ Gn.T)[iu3]))
p_t3 = float(np.mean(null3 >= obs_t3))
log('T3: obs=%.4f null=%.4f p=%.5f'
    % (obs_t3, float(np.median(null3)), p_t3))

log('=== T4 out-of-bank generalization (L3) ===')
pairs_cos = []
for k in range(nD_new):
    if not ok_new[k]:
        continue
    a = ALPHAS[int(c_new[k]) - 1]
    d = D3_new[k]
    nd = np.linalg.norm(d)
    pairs_cos.append(abs(float(d @ a))
                     / max(nd, 1e-12))
obs_t4 = float(np.median(pairs_cos))
rng4 = np.random.default_rng(SEED_T4)
null4 = np.zeros(N_RND)
for it in range(N_RND):
    G = rng4.standard_normal((len(pairs_cos), HDIM))
    Gn = G / np.maximum(
        np.linalg.norm(G, axis=1), 1e-12)[:, None]
    C = np.abs(Gn @ A.T)
    null4[it] = float(np.median(C))
p_t4 = float(np.mean(null4 >= obs_t4))
log('T4: %d pairs obs=%.4f null=%.4f p=%.5f'
    % (len(pairs_cos), obs_t4,
       float(np.median(null4)), p_t4))

log('=== T5 L20 control ===')
D20_old = np.zeros_like(D3_old)
for k in range(nD_old):
    ci, bi = int(c_old[k]), int(b_old[k])
    ic = idx_of[(ci, bi, False)]
    ib = idx_of[(0, bi, False)]
    D20_old[k] = V20o[ic] - V20o[ib]
a98_shape_ok = bool(
    z42['D3'].shape == D3_old.shape
    and z42['D20'].shape == D20_old.shape)
a98_d3 = float(np.max(np.abs(
    D3_old - z42['D3']))) if a98_shape_ok \
    else float('nan')
a98_d20 = float(np.max(np.abs(
    D20_old - z42['D20']))) if a98_shape_ok \
    else float('nan')
a98_ok = bool(a98_shape_ok and a98_d3 == 0.0
              and a98_d20 == 0.0)
log('a98 shape_ok=%s dD3=%.3e dD20=%.3e'
    % (a98_shape_ok, a98_d3, a98_d20))
ok20_old = np.linalg.norm(D20_old, axis=1) \
    >= DEGEN_NORM
D20_use = D20_old[ok20_old]
c_use = c_old[ok20_old]
b_use = b_old[ok20_old]
f_p20, f_b20, p_pfx20, p_body20 = t1_tests(
    D20_use, c_use, b_use, SEED_L20, SEED_L20)
D20_new = np.zeros((nD_new, HDIM))
for k in range(nD_new):
    ci, bi = int(c_new[k]), int(b_new[k])
    ic = idx_of[(ci, bi, True)]
    ib = idx_of[(0, bi, True)]
    D20_new[k] = V20o[ic] - V20o[ib]
A20 = []
for c in (1, 2, 3):
    m = c_use == c
    a = D20_use[m].mean(axis=0)
    n = np.linalg.norm(a)
    A20.append(a / n if n > 0 else a)
A20 = np.stack(A20)
cos20 = []
for k in range(nD_new):
    if not ok_new[k]:
        continue
    d = D20_new[k]
    nd = np.linalg.norm(d)
    cos20.append(abs(float(d @ A20[int(c_new[k])
                                   - 1]))
                 / max(nd, 1e-12))
obs_t4_20 = float(np.median(cos20))
rng5 = np.random.default_rng(SEED_L20_T4)
null5 = np.zeros(N_RND)
for it in range(N_RND):
    G = rng5.standard_normal((len(cos20), HDIM))
    Gn = G / np.maximum(
        np.linalg.norm(G, axis=1), 1e-12)[:, None]
    C = np.abs(Gn @ A20.T)
    null5[it] = float(np.median(C))
p_t4_20 = float(np.mean(null5 >= obs_t4_20))
log('T5: L20 f_prefix=%.4f p=%.5f f_body=%.4f '
    'p=%.5f; T4 L20 obs=%.4f null=%.4f p=%.5f'
    % (f_p20, p_pfx20, f_b20, p_body20,
       obs_t4_20, float(np.median(null5)),
       p_t4_20))

# ---------- verdict ----------
a_ok = bool(a93_diff == 0.0 and a94_diff == 0.0
            and a95_ok and a96_ok and a97_ok
            and a98_ok and a99_ok)
prefixfx = bool(p_pfx3 < 0.05)
bodyfx = bool(p_body3 < 0.05)
contenthit = bool(p_t2 < 0.05)
fieldcommon = bool(p_t3 < 0.05)
generalizes = bool(p_t4 < 0.05)
if prefixfx and bodyfx:
    verdict = 'fieldvar_prefix_plus_body_qwen'
elif prefixfx:
    verdict = 'fieldvar_prefix_dominant_qwen'
else:
    verdict = 'fieldvar_null_qwen'

log('=== verdict ===')
log('a93=%r a94=%r a95_ok=%s (%.4f) a96=%s '
    'a97_ok=%s (%.3e, %d) a98_ok=%s (%.1e/%.1e) '
    'a99_ok=%s (%.1e/%.1e)'
    % (a93_diff, a94_diff, a95_ok, a95_maxdiff,
       a96_ok, a97_ok, a97_diff, a97_matched,
       a98_ok, a98_d3, a98_d20, a99_ok,
       a99_d3, a99_d20))
log('T1 p_pfx=%.5f p_body=%.5f | T2 p=%.5f | '
    'T3 p=%.5f | T4 p=%.5f'
    % (p_pfx3, p_body3, p_t2, p_t3, p_t4))
log('VERDICT=%s anchor_all_ok=%s' % (verdict, a_ok))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    bodies=np.array(BODIES),
    new_bodies=np.array(NEW_BODIES),
    prefixes=np.array(PREFIXES),
    targets=np.array(TARGETS),
    new_targets=np.array(NEW_TARGETS),
    body_idx=np.array(BODY_IDX),
    D3_old=D3_old, D3_new=D3_new,
    c_old=c_old, b_old=b_old,
    c_new=c_new, b_new=b_new,
    ok_old=ok_old, ok_new=ok_new,
    f_prefix=np.float64(f_p3),
    f_body=np.float64(f_b3),
    p_pfx=np.float64(p_pfx3),
    p_body=np.float64(p_body3),
    obs_t2=np.float64(obs_t2),
    p_t2=np.float64(p_t2),
    norms_old=nrmD,
    obs_t3=np.float64(obs_t3),
    null3_med=np.float64(np.median(null3)),
    p_t3=np.float64(p_t3),
    obs_t4=np.float64(obs_t4),
    null4_med=np.float64(np.median(null4)),
    p_t4=np.float64(p_t4),
    f_prefix_20=np.float64(f_p20),
    f_body_20=np.float64(f_b20),
    p_pfx_20=np.float64(p_pfx20),
    p_body_20=np.float64(p_body20),
    obs_t4_20=np.float64(obs_t4_20),
    p_t4_20=np.float64(p_t4_20),
    a93_diff=np.float64(a93_diff),
    a94_diff=np.float64(a94_diff),
    a95_maxdiff=np.float64(a95_maxdiff),
    a95_top2_ok=np.bool_(a95_top2_ok),
    a96_ok=np.bool_(a96_ok),
    a97_diff=np.float64(a97_diff),
    a97_matched=np.int64(a97_matched),
    a98_d3=np.float64(a98_d3),
    a98_d20=np.float64(a98_d20),
    a99_d3=np.float64(a99_d3),
    a99_d20=np.float64(a99_d20),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'run': 'run3 authoritative (run1 crash pre-'
           'verdict a98 npz-key bug; run2 crash '
           'mid-T2 empty-cell under prefix label '
           'permutation; T2 null redefined to the '
           'within-block body-label permutation; '
           'T1 observed in run2, design unchanged)',
    'anchor_all_ok': a_ok,
    'anchors': {
        'a93_dup_prefill_bit': a93_diff,
        'a94_dup_all_bit': a94_diff,
        'a95_top2_ok': a95_top2_ok,
        'a95_maxdiff': a95_maxdiff,
        'a95_gate': A95_GATE,
        'a95_note': a95_note,
        'a96_source_seals': a96_ok,
        'a97_cross_phase_bit': a97_diff,
        'a97_matched': a97_matched,
        'a98_d3_bit': a98_d3,
        'a98_d20_bit': a98_d20,
        'a99_sit3_diff': a99_d3,
        'a99_sit20_diff': a99_d20,
        'a99_gate': A98_GATE,
    },
    'T1_variance': {
        'f_prefix': f_p3, 'p_prefix': p_pfx3,
        'f_body': f_b3, 'p_body': p_body3,
        'f_resid': 1.0 - min(f_p3 + f_b3, 1.0),
    },
    'T2_content_outlier': {
        'obs_norm': obs_t2,
        'rank': int(np.sum(nrmD >= obs_t2)),
        'p_t2': p_t2,
        'norms_by_cell': {
            'c%d_b%d' % (int(c_old[k]),
                         int(b_old[k])):
            float(nrmD[k])
            for k in range(nD_old)},
    },
    'T3_field_commonality': {
        'obs_med_abs_cos': obs_t3,
        'null_med': float(np.median(null3)),
        'p_t3': p_t3,
    },
    'T4_generalization': {
        'n_pairs': len(pairs_cos),
        'obs_med_abs_cos': obs_t4,
        'null_med': float(np.median(null4)),
        'p_t4': p_t4,
    },
    'T5_layer20': {
        'f_prefix': f_p20, 'p_prefix': p_pfx20,
        'f_body': f_b20, 'p_body': p_body20,
        't4_obs': obs_t4_20, 't4_p': p_t4_20,
    },
    'flags': {'prefixfx': prefixfx,
              'bodyfx': bodyfx,
              'contenthit': contenthit,
              'fieldcommon': fieldcommon,
              'generalizes': generalizes},
    'prereg': PREREG,
    'elapsed_s': round(elapsed, 1),
}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)


def sha8(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()) \
            .hexdigest()[:8]


seal = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(__file__)),
    'verdict': verdict,
    'anchor_all_ok': a_ok,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('sealed npz8=%s result8=%s exec8=%s script8=%s '
    'elapsed=%.1fs'
    % (seal['npz_sha256_8'], seal['result_sha256_8'],
       seal['exec_sha256_8'], seal['script_sha256_8'],
       elapsed))
log('sealed')
