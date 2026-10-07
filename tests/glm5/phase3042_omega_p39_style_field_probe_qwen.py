# -*- coding: utf-8 -*-
# Phase 3042 - Omega-P39: style-field probe (situational
# code globality test)
# Phase 3041 (sitaxis_multirank_qwen) established: the L3 kv7
# situational residual is multidimensional, strongly
# left-prefix deterministic (same-prefix pairs reproduce
# cos ~ +1), weakly shared across words. The HDMCC review
# flagged the attachment claim "style prompt = GLOBAL
# context gravity field distorting attention routing" as
# UNTESTED. This phase tests it directly: prepend style /
# topic prefixes to 8 body prompts (each containing exactly
# one logic target word), extract the V write of the SAME
# target occurrence under each condition, and dissect the
# displacement Delta = V(cond) - V(base):
# (T1) global field: alignment of same-prefix cross-body
#     displacements (median |cos|) vs size-matched Gaussian
#     null - a global style field would align displacements
#     across DIFFERENT body sentences;
# (T1b) same on the complement-projected displacement
#     (word-identity subspace removed) - field inside the
#     situational channel specifically;
# (T2) prefix identity: med cos(same-prefix pairs) - med
#     cos(diff-prefix pairs), exact prefix-label permutation
#     null (bodies fixed) - does each prefix imprint its OWN
#     direction;
# (T3) descriptive dose: per-prefix med |Delta| and prefix
#     token lengths (bootstrap CI on the |Delta| ~ length
#     slope, descriptive only);
# (T4) channel decomposition: energy fraction of Delta in
#     the base-bank word-identity subspace B3 (reconstructed
#     VERBATIM from the 3041 npz) vs the Gaussian expectation
#     r3/HDIM; descriptive complement-dominance flag.
# (T5) layer control: T1/T2 repeated at L20 with B20.
# Word-identity subspace reconstructed from 3041 npz V3/V20
# (bit-anchored) keeps the channel definition identical to
# 3040/3041 without re-extracting the 28-prompt bank.
# PREREG frozen below BEFORE any observation.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3042
NAME = 'omega_p39_style_field_probe_qwen'
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
N_BOOT = 10000
SEED_MAIN = 3009
SEED_T1 = 9060
SEED_T1B = 9061
SEED_T2 = 9062
SEED_BOOT = 9063
SEED_L20 = 9064
DEGEN_NORM = 1e-6
A88_GATE = 0.15
A89_GATE = 1e-5
A92_GATE = 1e-12

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
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')

PREREG = {
    'mode': 'pure prefill + KV-cache read (no '
            'intervention, eager attention, bf16, seed '
            '3009); 8 body prompts x 4 prefix '
            'conditions (base + 3 prefixes), one '
            'prefill per assembled prompt; V read at '
            'the target-word position from '
            'past.layers[li].values[0, kv7]; layers '
            '(3, 20); V PRIMARY',
    'question': 'style-field probe (3042 A main '
                'line): does a style/topic prefix '
                'displace the V write of a DOWNSTREAM '
                'body word (i) in a prefix-shared '
                'direction across different body '
                'sentences (global field), (ii) in a '
                'prefix-specific direction (prefix '
                'identity), (iii) inside the word-'
                'identity subspace or the situational '
                'complement (channel), and (iv) how '
                'large is the displacement (dose)? '
                'Direct test of the HDMCC attachment '
                'claim "style = global gravity field '
                'on attention routing"',
    'bodies': '8 GEN prompts of the 3037 bank, each '
              'containing EXACTLY one logic target '
              '(so/because/therefore/however/while/'
              'yet/although/thus, one per body); '
              'assembled prompt = prefix + " " + body '
              'for non-empty prefixes; assert: target '
              'token id occurs EXACTLY once per '
              'assembled prompt',
    'displacement': 'Delta(body, prefix) = V_c[pos_c] - '
                    'V_base[pos_base] at the target '
                    'occurrence, per layer; DEGENERATE '
                    'RULE: |Delta| < 1e-6 excluded from '
                    'all pair statistics (3040 lesson)',
    'channel_basis': 'B3/B20 reconstructed VERBATIM '
                     'from the 3041 npz V3/V20 (26-type '
                     'word-mean row-space SVD basis, '
                     'B = Vt[:r].T); anchor a92: '
                     'recomputed SIT3/SIT20 must match '
                     'the npz SIT3/SIT20 within 1e-12 '
                     'and orthonormality <= 1e-5 - keeps '
                     'the channel definition identical '
                     'to 3040/3041',
    'T1': 'global field: same-prefix cross-body '
          'displacement pairs (3 prefixes x C(8,2) '
          'pairs, norm-gated); stat = median |cos| '
          '(abs: field alignment is sign-agnostic at '
          'first order; signed cosines reported '
          'too); null = size-matched Gaussian '
          '(N_RND=%d seed %d, median over equal pair '
          'count per draw); one-sided p = '
          'P(null >= obs); global iff p<0.05'
          % (N_RND, SEED_T1),
    'T1b': 'complement field: T1 repeated on '
           'Delta_c = Delta - B3 B3^T Delta '
           '(complement, dim HDIM - r3); Gaussian '
           'null in the complement dimension '
           '(N_RND=%d seed %d); globcomp iff p<0.05'
           % (N_RND, SEED_T1B),
    'T2': 'prefix identity: d2 = med cos(same-prefix '
          'cross-body pairs) - med cos(diff-prefix '
          'pairs), SIGNED cosines (displacement null '
          '~ 0); null = exact permutation of prefix '
          'labels among the 3x8 displacements (bodies '
          'fixed), N_PERM=%d seed %d; one-sided p = '
          'P(null >= obs); prefixid iff p<0.05 AND '
          'd2>0' % (N_PERM, SEED_T2),
    'T3': 'DESCRIPTIVE dose: per-prefix med |Delta| '
          'and prefix token lengths; bootstrap 95%% '
          'CI on the |Delta| ~ prefix-length slope '
          '(%d resamples seed %d); no verdict gate'
          % (N_BOOT, SEED_BOOT),
    'T4': 'channel decomposition: energy fraction '
          'ew = ||B3^T Delta||^2 / ||Delta||^2 per '
          'displacement; med ew reported vs the '
          'Gaussian expectation r3/HDIM; '
          'complement-dominance flag = med ew < '
          'r3/HDIM (descriptive, not gated)',
    'T5': 'layer control: T1 and T2 repeated at L20 '
          'with B20 (seeds %d for T2; Gaussian null '
          'size-matched); relay specificity of any '
          'field' % SEED_L20,
    'verdict_tree': 'global = (p_T1<0.05); prefixid = '
                    '(p_T2<0.05 AND d2>0); globcomp = '
                    '(p_T1b<0.05) reported as flag; '
                    'global AND prefixid -> '
                    'stylefield_global_identity_qwen; '
                    'global AND NOT prefixid -> '
                    'stylefield_global_qwen; NOT '
                    'global AND prefixid -> '
                    'stylefield_prefix_identity_only_'
                    'qwen; else -> stylefield_null_'
                    'qwen; single branch assignment',
    'anchors': 'a86 duplicate prefill prompt0 K3/V3 '
               'bit-identical (0.0); a87 full duplicate '
               'extraction all 32 prompts max abs diff '
               '0.0; a88 manual final-norm+lm_head '
               'recompute vs prefill logits (top-2 '
               'identity AND max|dlogit| <= 0.15; '
               'near-tie skip note if gap<0.05); a89 '
               'reconstructed basis orthonormality '
               'max|B^T B - I| <= 1e-5 at BOTH layers; '
               'a90 source seals 3037/3038/3039/3040/'
               '3041 sha8 match seal.json; a91 '
               'cross-phase bit anchor: base-condition '
               'target occurrences matched to 3037 npz '
               'by (word, prompt, pos), max|dV3| == '
               '0.0; a92 channel-basis anchor vs 3041 '
               'npz: recomputed SIT3 AND SIT20 max '
               'diff <= 1e-12',
    'control': 'size-matched Gaussian nulls for T1/'
               'T1b/T5-T1; exact prefix-label '
               'permutation for T2/T5-T2; diff-prefix '
               'pairs are the within-phase empirical '
               'null; no intervention',
    'statistics_discipline': 'abs cosines for field '
                             'alignment (sign-agnostic), '
                             'signed for identity (null '
                             '~0); all arrays '
                             'pre-initialized; '
                             'degenerate-norm gate on '
                             'Delta; verdict criteria '
                             'in one branch assignment; '
                             'small-n caveat: 24 '
                             'displacements per layer, '
                             'exploratory margin',
    'corrections': 'run1 crashed PRE-verdict '
                   '(assembly logging only, '
                   'no test statistic '
                   'observed): log format '
                   'string had 5 placeholders '
                   'but 4 arguments (missing '
                   'arg for the '
                   'assembled-prompt %r); '
                   'corrected to pass the '
                   'assembled string; verdict '
                   'tree unchanged',
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
for w in TARGETS:
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    assert len(wi) == 1, (w, wi)
    word_tok[w] = int(wi[0])
nB = len(BODIES)
nC = len(PREFIXES)
assembled = []
for ci in range(nC):
    for bi in range(nB):
        if PREFIXES[ci]:
            s = PREFIXES[ci] + ' ' + BODIES[bi]
        else:
            s = BODIES[bi]
        ids = tok(s, add_special_tokens=False)[
            'input_ids']
        ids = [int(x) for x in ids]
        t = word_tok[TARGETS[bi]]
        assert ids.count(t) == 1, (ci, bi, s)
        assembled.append({'ids': ids,
                          'pos': ids.index(t),
                          'cond': ci, 'body': bi})
        log('cond=%d body=%d pos=%d len=%d %r'
            % (ci, bi, ids.index(t), len(ids),
               s))
n_pr = len(assembled)
plen = np.array([len(a['ids']) for a in assembled],
                dtype=np.int64)

# a86: duplicate prefill of prompt0 (bit determinism)
p_b0, lg_b0, kv_b0, xf_b0 = prefill_extract(
    assembled[0]['ids'], cap_fin=True)
p_d0, lg_d0, kv_d0, _ = prefill_extract(
    assembled[0]['ids'])
a86_diff = 0.0
for li in LAYERS_EX:
    for a, b in ((kv_b0[li][0], kv_d0[li][0]),
                 (kv_b0[li][1], kv_d0[li][1])):
        a86_diff = max(a86_diff, float(
            np.max(np.abs(a - b))))
a86_diff = float(a86_diff)

# ---------- main extraction ----------
Ps = [None] * n_pr
KVs = {li: [None] * n_pr for li in LAYERS_EX}
for i in range(n_pr):
    p, _, kv, _ = prefill_extract(assembled[i]['ids'])
    Ps[i] = p
    for li in LAYERS_EX:
        KVs[li][i] = kv[li]

# a87: full duplicate extraction
a87_diff = 0.0
for i in range(n_pr):
    p2, _, kv2, _ = prefill_extract(
        assembled[i]['ids'])
    a87_diff = max(a87_diff, float(np.max(
        np.abs(Ps[i] - p2))))
    for li in LAYERS_EX:
        for a, b in zip(KVs[li][i], kv2[li]):
            a87_diff = max(a87_diff, float(
                np.max(np.abs(a - b))))
a87_diff = float(a87_diff)
log('a86=%.3e a87=%.3e' % (a86_diff, a87_diff))

# a88: manual final-norm+lm_head recompute (prompt0)
w_norm = model.model.norm.weight.detach() \
    .float().cpu().numpy()
h_fin = xf_b0
var = float((h_fin ** 2).mean())
hn = h_fin / np.sqrt(var + EPS)
lg_man = (W_U @ (hn * w_norm)).astype(np.float64)
a88_maxdiff = float(np.max(np.abs(lg_man - lg_b0)))
ord_m = np.argsort(-lg_man)
ord_o = np.argsort(-lg_b0)
srt = np.sort(lg_b0)
gap0 = float(srt[-1] - srt[-2])
if gap0 < 0.05:
    a88_top2_ok = True
    a88_note = 'near-tie skip (gap=%.4f)' % gap0
else:
    a88_top2_ok = bool(ord_m[0] == ord_o[0]
                       and ord_m[1] == ord_o[1])
    a88_note = ''
a88_ok = bool(a88_top2_ok
              and a88_maxdiff <= A88_GATE)
log('a88 top2=%s maxdiff=%.4f (gate %.2f) %s'
    % (a88_top2_ok, a88_maxdiff, A88_GATE,
       a88_note))

# a90: source seals
a90_detail = []
for ph, nm in (
        (3037, 'omega_p34_kv_situational_'
               'specificity_qwen'),
        (3038, 'omega_p35_reentrant_readout_qwen'),
        (3039, 'omega_p36_direct_logistic_'
               'replication_qwen'),
        (3040, 'omega_p37_situational_'
               'component_qwen'),
        (3041, 'omega_p38_situational_axis_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a90_detail.append(bool(s == sealj['result_sha256_8']))
a90_ok = bool(a90_detail) and all(a90_detail)

# ---------- occurrence vectors ----------
V3o = np.zeros((n_pr, HDIM))
V20o = np.zeros((n_pr, HDIM))
for i in range(n_pr):
    pos = assembled[i]['pos']
    V3o[i] = KVs[3][i][1][pos]
    V20o[i] = KVs[20][i][1][pos]

# a91: cross-phase bit anchor vs 3037 npz (base cond)
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
a91_diff = 0.0
a91_matched = 0
for i in range(n_pr):
    if assembled[i]['cond'] != 0:
        continue
    bi = assembled[i]['body']
    key = (TARGETS[bi], int(BODY_IDX[bi]),
           int(assembled[i]['pos']))
    if key not in ref:
        continue
    a91_diff = max(a91_diff, float(np.max(
        np.abs(V3o[i] - ref[key]))))
    a91_matched += 1
a91_diff = float(a91_diff)
a91_ok = bool(a91_matched == nB
              and a91_diff == 0.0)
log('a91 matched=%d/%d max|dV3|=%.3e'
    % (a91_matched, nB, a91_diff))

# ---------- channel basis from 3041 npz ----------
z41 = np.load(os.path.join(
    BASE, 'phase3041',
    'omega_p38_situational_axis_qwen',
    'omega_p38_situational_axis_qwen.npz'),
    allow_pickle=True)
V3_41 = z41['V3']
V20_41 = z41['V20']
occ_w_41 = z41['occ_w']
n41 = V3_41.shape[0]
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
a89_ok = bool(max(o3, o20) <= A89_GATE)
log('basis: r3=%d orth3=%.2e r20=%d orth20=%.2e ok=%s'
    % (r3, o3, r20, o20, a89_ok))

# a92: SIT reproduction vs 3041 npz
SIT3_41 = z41['SIT3']
SIT20_41 = z41['SIT20']
SIT3_rec = V3_41 - (V3_41 @ B3) @ B3.T
SIT20_rec = V20_41 - (V20_41 @ B20) @ B20.T
a92_d3 = float(np.max(np.abs(SIT3_rec - SIT3_41)))
a92_d20 = float(np.max(np.abs(SIT20_rec
                              - SIT20_41)))
a92_ok = bool(max(a92_d3, a92_d20) <= A92_GATE)
log('a92 dSIT3=%.3e dSIT20=%.3e ok=%s'
    % (a92_d3, a92_d20, a92_ok))

# ---------- displacements ----------
idx_of = {}
for i in range(n_pr):
    idx_of[(assembled[i]['cond'],
            assembled[i]['body'])] = i
d_list = []
for ci in range(1, nC):
    for bi in range(nB):
        ic = idx_of[(ci, bi)]
        ib = idx_of[(0, bi)]
        d_list.append({
            'cond': ci, 'body': bi,
            'd3': V3o[ic] - V3o[ib],
            'd20': V20o[ic] - V20o[ib]})
nD = len(d_list)
D3 = np.zeros((nD, HDIM))
D20 = np.zeros((nD, HDIM))
d_cond = np.zeros(nD, dtype=np.int64)
d_body = np.zeros(nD, dtype=np.int64)
for k in range(nD):
    D3[k] = d_list[k]['d3']
    D20[k] = d_list[k]['d20']
    d_cond[k] = d_list[k]['cond']
    d_body[k] = d_list[k]['body']
nrm3 = np.linalg.norm(D3, axis=1)
nrm20 = np.linalg.norm(D20, axis=1)
ok3 = nrm3 >= DEGEN_NORM
ok20 = nrm20 >= DEGEN_NORM
log('displacements: %d; degenerate L3 %d, L20 %d'
    % (nD, int((~ok3).sum()), int((~ok20).sum())))
plen_pf = np.array(
    [len(tok(PREFIXES[c], add_special_tokens=False)
         ['input_ids']) for c in range(nC)],
    dtype=np.int64)
log('prefix token lengths: %s'
    % json.dumps({int(c): int(plen_pf[c])
                  for c in range(nC)}))


def field_tests(Dmat, ok, B, seed_t1, seed_t1b,
                seed_t2, d_amb_r):
    """T1 field alignment + T1b complement + T2 prefix
    identity + T4 energy fraction, norm-gated rows."""
    rows = np.where(ok)[0]
    m = len(rows)
    X = Dmat[rows]
    Xn = X / np.maximum(
        np.linalg.norm(X, axis=1), 1e-12)[:, None]
    iu, ju = np.triu_indices(m, 1)
    cos_abs = np.abs(np.einsum('ij,ij->i',
                               Xn[iu], Xn[ju]))
    cos_sgn = np.einsum('ij,ij->i', Xn[iu], Xn[ju])
    c_r = d_cond[rows]
    same = c_r[iu] == c_r[ju]
    diff = ~same
    obs_t1 = float(np.median(cos_abs))
    n_pairs = int(len(cos_abs))
    rng = np.random.default_rng(seed_t1)
    null1 = np.zeros(N_RND)
    for it in range(N_RND):
        G = rng.standard_normal((m, HDIM))
        Gn = G / np.maximum(
            np.linalg.norm(G, axis=1),
            1e-12)[:, None]
        ca = np.abs(np.einsum('ij,ij->i',
                              Gn[iu], Gn[ju]))
        null1[it] = float(np.median(ca))
    p_t1 = float(np.mean(null1 >= obs_t1))
    # T1b: complement-projected field
    Xc = X - (X @ B) @ B.T
    Xcn = Xc / np.maximum(
        np.linalg.norm(Xc, axis=1), 1e-12)[:, None]
    cosc = np.abs(np.einsum('ij,ij->i',
                            Xcn[iu], Xcn[ju]))
    obs_t1b = float(np.median(cosc))
    rngb = np.random.default_rng(seed_t1b)
    null1b = np.zeros(N_RND)
    damb = HDIM - d_amb_r
    for it in range(N_RND):
        G = rngb.standard_normal((m, damb))
        Gn = G / np.maximum(
            np.linalg.norm(G, axis=1),
            1e-12)[:, None]
        ca = np.abs(np.einsum('ij,ij->i',
                              Gn[iu], Gn[ju]))
        null1b[it] = float(np.median(ca))
    p_t1b = float(np.mean(null1b >= obs_t1b))
    # T2: prefix identity (signed)
    med_same = float(np.median(cos_sgn[same])) \
        if same.any() else float('nan')
    med_diff = float(np.median(cos_sgn[diff])) \
        if diff.any() else float('nan')
    obs_d2 = med_same - med_diff
    rng2 = np.random.default_rng(seed_t2)
    perm2 = np.zeros(N_PERM)
    for it in range(N_PERM):
        lab = rng2.permutation(c_r)
        sp = lab[iu] == lab[ju]
        ms = cos_sgn[sp]
        md = cos_sgn[~sp]
        perm2[it] = (float(np.median(ms))
                     - float(np.median(md))
                     if len(ms) and len(md)
                     else -1.0)
    p_t2 = float(np.mean(perm2 >= obs_d2)) \
        if same.any() and diff.any() else float('nan')
    # T4: energy fraction in word subspace
    ew = ((X @ B) ** 2).sum(axis=1) \
        / np.maximum((X ** 2).sum(axis=1), 1e-30)
    return {
        'm': m, 'n_pairs': n_pairs,
        'obs_t1': obs_t1,
        'null1_med': float(np.median(null1)),
        'p_t1': p_t1,
        'obs_t1b': obs_t1b,
        'null1b_med': float(np.median(null1b)),
        'p_t1b': p_t1b,
        'obs_d2': obs_d2,
        'med_same': med_same,
        'med_diff': med_diff,
        'p_t2': p_t2,
        'med_ew': float(np.median(ew)),
        'ew': ew, 'rows': rows,
        'n_same': int(same.sum()),
        'n_diff': int(diff.sum()),
    }


log('=== L3 tests ===')
R3 = field_tests(D3, ok3, B3, SEED_T1, SEED_T1B,
                 SEED_T2, r3)
log('L3 ok=%d/%d pairs=%d (same %d diff %d)'
    % (R3['m'], nD, R3['n_pairs'], R3['n_same'],
       R3['n_diff']))
log('T1: obs=%.4f null=%.4f p=%.5f'
    % (R3['obs_t1'], R3['null1_med'], R3['p_t1']))
log('T1b: obs=%.4f null=%.4f p=%.5f'
    % (R3['obs_t1b'], R3['null1b_med'],
       R3['p_t1b']))
log('T2: d2=%.4f (same %.4f diff %.4f) p=%.5f'
    % (R3['obs_d2'], R3['med_same'],
       R3['med_diff'], R3['p_t2']))
log('T4: med ew=%.4f vs Gaussian expectation %.4f'
    % (R3['med_ew'], r3 / HDIM))

log('=== T3 dose (descriptive) ===')
dose = []
for c in range(1, nC):
    msk = ok3 & (d_cond == c)
    dose.append({
        'prefix': PREFIXES[c],
        'n_tok': int(plen_pf[c]),
        'med_absD3': float(np.median(nrm3[msk]))
        if msk.any() else float('nan')})
log('dose: %s' % json.dumps(dose))
mD = nrm3[ok3]
lD = plen_pf[d_cond[ok3]].astype(np.float64)
sl_dose = float(np.polyfit(lD, mD, 1)[0]) \
    if len(set(lD)) > 1 else float('nan')
rngd = np.random.default_rng(SEED_BOOT)
boot_d = np.zeros(N_BOOT)
for b in range(N_BOOT):
    idx = rngd.integers(0, len(mD), len(mD))
    boot_d[b] = np.polyfit(lD[idx], mD[idx], 1)[0]
ci_dose = (float(np.percentile(boot_d, 2.5)),
           float(np.percentile(boot_d, 97.5)))
log('dose slope=%.4f CI[%.4f,%.4f]'
    % (sl_dose, ci_dose[0], ci_dose[1]))

log('=== T5 L20 control ===')
R20 = field_tests(D20, ok20, B20, SEED_T1, SEED_T1B,
                  SEED_L20, r20)
log('L20 ok=%d/%d; T1 obs=%.4f null=%.4f p=%.5f; '
    'T2 d2=%.4f p=%.5f'
    % (R20['m'], nD, R20['obs_t1'],
       R20['null1_med'], R20['p_t1'],
       R20['obs_d2'], R20['p_t2']))

# ---------- verdict ----------
a_ok = bool(a86_diff == 0.0 and a87_diff == 0.0
            and a88_ok and a89_ok and a90_ok
            and a91_ok and a92_ok)
global_f = bool(R3['p_t1'] < 0.05)
globcomp = bool(R3['p_t1b'] < 0.05)
prefixid = bool(R3['p_t2'] < 0.05
                and R3['obs_d2'] > 0)
if global_f and prefixid:
    verdict = 'stylefield_global_identity_qwen'
elif global_f:
    verdict = 'stylefield_global_qwen'
elif prefixid:
    verdict = 'stylefield_prefix_identity_only_qwen'
else:
    verdict = 'stylefield_null_qwen'

log('=== verdict ===')
log('a86=%r a87=%r a88_ok=%s (%.4f) a89_ok=%s '
    '(%.2e) a90=%s a91_ok=%s (%.3e, %d) a92_ok=%s '
    '(d3=%.1e d20=%.1e)'
    % (a86_diff, a87_diff, a88_ok, a88_maxdiff,
       a89_ok, max(o3, o20), a90_ok, a91_ok,
       a91_diff, a91_matched, a92_ok,
       a92_d3, a92_d20))
log('T1 p=%.5f | T1b p=%.5f | T2 p=%.5f d2=%.4f'
    % (R3['p_t1'], R3['p_t1b'], R3['p_t2'],
       R3['obs_d2']))
log('VERDICT=%s anchor_all_ok=%s' % (verdict, a_ok))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    bodies=np.array(BODIES),
    prefixes=np.array(PREFIXES),
    targets=np.array(TARGETS),
    body_idx=np.array(BODY_IDX),
    d_cond=d_cond, d_body=d_body,
    plen=plen, plen_pf=plen_pf,
    D3=D3, D20=D20, ok3=ok3, ok20=ok20,
    obs_t1=np.float64(R3['obs_t1']),
    null1_med=np.float64(R3['null1_med']),
    p_t1=np.float64(R3['p_t1']),
    obs_t1b=np.float64(R3['obs_t1b']),
    null1b_med=np.float64(R3['null1b_med']),
    p_t1b=np.float64(R3['p_t1b']),
    obs_d2=np.float64(R3['obs_d2']),
    med_same=np.float64(R3['med_same']),
    med_diff=np.float64(R3['med_diff']),
    p_t2=np.float64(R3['p_t2']),
    med_ew3=np.float64(R3['med_ew']),
    ew3=R3['ew'],
    dose_med=np.array([d['med_absD3']
                       for d in dose]),
    slope_dose=np.float64(sl_dose),
    ci_dose=np.array(ci_dose),
    obs_t1_20=np.float64(R20['obs_t1']),
    p_t1_20=np.float64(R20['p_t1']),
    obs_d2_20=np.float64(R20['obs_d2']),
    p_t2_20=np.float64(R20['p_t2']),
    med_ew20=np.float64(R20['med_ew']),
    a86_diff=np.float64(a86_diff),
    a87_diff=np.float64(a87_diff),
    a88_maxdiff=np.float64(a88_maxdiff),
    a88_top2_ok=np.bool_(a88_top2_ok),
    a89_max_orth=np.float64(max(o3, o20)),
    a90_ok=np.bool_(a90_ok),
    a91_diff=np.float64(a91_diff),
    a91_matched=np.int64(a91_matched),
    a92_d3=np.float64(a92_d3),
    a92_d20=np.float64(a92_d20),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'run': 'run1 authoritative',
    'anchor_all_ok': a_ok,
    'anchors': {
        'a86_dup_prefill_bit': a86_diff,
        'a87_dup_all_bit': a87_diff,
        'a88_top2_ok': a88_top2_ok,
        'a88_maxdiff': a88_maxdiff,
        'a88_gate': A88_GATE,
        'a88_note': a88_note,
        'a89_basis_orth_max': max(o3, o20),
        'a89_gate': A89_GATE,
        'a90_source_seals': a90_ok,
        'a91_cross_phase_bit': a91_diff,
        'a91_matched': a91_matched,
        'a92_sit3_diff': a92_d3,
        'a92_sit20_diff': a92_d20,
        'a92_gate': A92_GATE,
    },
    'displacements': {
        'n': nD,
        'degenerate_L3': int((~ok3).sum()),
        'degenerate_L20': int((~ok20).sum()),
        'prefix_tok_lens': {int(c): int(plen_pf[c])
                            for c in range(nC)},
    },
    'T1_global_field': {
        'n_pairs': R3['n_pairs'],
        'obs_med_abs_cos': R3['obs_t1'],
        'null_med': R3['null1_med'],
        'p_t1': R3['p_t1'],
    },
    'T1b_complement_field': {
        'obs_med_abs_cos': R3['obs_t1b'],
        'null_med': R3['null1b_med'],
        'p_t1b': R3['p_t1b'],
        'd_ambient': HDIM - r3,
    },
    'T2_prefix_identity': {
        'n_same': R3['n_same'],
        'n_diff': R3['n_diff'],
        'med_same': R3['med_same'],
        'med_diff': R3['med_diff'],
        'd2': R3['obs_d2'], 'p_t2': R3['p_t2'],
    },
    'T3_dose': {
        'per_prefix': dose,
        'slope': sl_dose, 'ci_slope': ci_dose,
    },
    'T4_channel': {
        'med_ew_L3': R3['med_ew'],
        'gauss_expectation': r3 / HDIM,
        'r3': r3,
        'complement_dominant':
            bool(R3['med_ew'] < r3 / HDIM),
    },
    'T5_layer20': {
        'obs_t1': R20['obs_t1'],
        'p_t1': R20['p_t1'],
        'd2': R20['obs_d2'], 'p_t2': R20['p_t2'],
        'med_ew': R20['med_ew'],
    },
    'flags': {'global': global_f,
              'globcomp': globcomp,
              'prefixid': prefixid},
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
