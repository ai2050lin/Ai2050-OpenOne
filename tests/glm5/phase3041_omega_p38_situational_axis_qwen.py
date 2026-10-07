# -*- coding: utf-8 -*-
# Phase 3041 - Omega-P38: anatomy of the rank-1 situational axis
# Phase 3040 (sitcomp_context_pure_qwen) established: after
# removing the word-identity subspace from the L3 kv7 V write,
# the residual SIT is context-locked (same-prompt d1=+0.0563,
# p=0.0124, L20-null) with NO positive word-residual code, yet
# same-word deviations follow an anti-parallel rank-1 geometry
# beyond the construction-matched null (p=0.9998), and one
# 'was' pair with an identical left prefix reproduced cos=+1.0
# exactly. This phase dissects that situational axis:
# (T1) per-word residual spectrum: for every word with >=3
#     non-degenerate occurrences, participation ratio
#     PR_w = (sum lam)^2 / sum(lam^2) of the gram eigenvalues
#     of its residual matrix; stat = median PR over words.
#     Null = construction-matched label permutation (3040 T2
#     machinery): pseudo-group means in V space, deviations
#     re-projected onto the SAME fixed complement basis B.
#     rank-1 claim = obs median PR significantly BELOW null.
# (T2) same-prefix minimal-pair control: for same-word
#     cross-prompt pairs, Lshare = length of the common
#     immediate-left token prefix before the occurrence.
#     stat = OLS slope of cos_dev ~ Lshare over ALL same-word
#     cross-prompt pairs (bootstrap CI); plus subset test
#     med(cos | Lshare>=2) - med(cos | all sw pairs) with an
#     exact random-subset null of equal size; pre-registered
#     fallback threshold Lshare>=1 if fewer than 3 pairs at
#     Lshare>=2; if still <3 pairs: NaN + insufficient note
#     (criterion unreachability declared, not post hoc).
# (T3) axis identity across words: principal axis a_w = first
#     left singular direction of each word's residual matrix;
#     stat = median off-diagonal |cos(a_w, a_v)| over word
#     pairs; null = random Gaussian residual sets of the SAME
#     group sizes in the same ambient dimension (Monte Carlo).
#     shared axis = obs significantly ABOVE null.
# (T4) layer control: T1 repeated at L20 with the L20 word-mean
#     subspace, its own degenerate mask, and the L3-defined
#     word set (relay specificity of the rank structure).
# Bank, extraction protocol, decomposition and degenerate-row
# rule are VERBATIM 3040 run3 (which is verbatim 3037).
# PREREG frozen below BEFORE any observation.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3041
NAME = 'omega_p38_situational_axis_qwen'
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
N_SUB = 100000
N_RND = 20000
N_BOOT = 10000
SEED_MAIN = 3009
SEED_T1 = 9050
SEED_T2 = 9051
SEED_T3 = 9052
SEED_L20 = 9053
SEED_BOOT = 9054
MIN_OCC = 2
MIN_GRP = 3
A81_GATE = 0.15
A82_GATE = 1e-5
DEGEN_NORM = 1e-6
LSHARE_MAIN = 2
LSHARE_FALLBACK = 1

GEN_PROMPTS = (
    'The weather was cold, so',
    'He studied every night because',
    'She wanted to buy the car, but',
    'The experiment failed, therefore',
    'You should take an umbrella if',
    'The meeting was long, and',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',
    'We left early because',)
EXTRA_PROMPTS = (
    'He passed the exam because he studied',
    'The match stopped because of rain',
    'He was tired, however he kept going',
    'The plan failed, however slowly they tried',
    'He sang while she cooked dinner',
    'We stayed inside while it rained',
    'She passed the test, although barely',
    'He kept running, although his legs hurt',
    'The data were clear, therefore we',
    'He apologized, therefore she forgave him',
    'The price was high, yet fair',
    'He was tired, yet he continued',
    'The bridge was out, thus we',
    'She saved money, thus she traveled',
    'It was late, so we left',
    'The store closed, so he went home',)
ALL_PROMPTS = GEN_PROMPTS + EXTRA_PROMPTS
TARGETS = ('because', 'however', 'while', 'although',
           'therefore', 'yet', 'thus', 'so')

PREREG = {
    'mode': 'pure prefill + KV-cache read (no '
            'intervention, eager attention, bf16, seed '
            '3009); VERBATIM 3040 run3 extraction: 28 '
            'prompts, extended alphabetic-token bank '
            '(strip().isalpha(), len>=2, >=2 '
            'occurrences), V read from '
            'past.layers[li].values[0, kv7] per '
            'occurrence position, layers (3, 20), V '
            'PRIMARY',
    'question': 'situational axis anatomy (3041 A main '
                'line): is the 3040 situational residual '
                'a rank-1 axis per word (T1), is its '
                'sign/magnitude locked to the shared '
                'left prefix (T2 same-prefix minimal '
                'pairs), and do different words share '
                'one situational direction or keep '
                'private axes (T3)?',
    'decomposition': 'VERBATIM 3040: word-mean matrix '
                     'WM (n_types x head_dim), SVD '
                     'rank-r row-space basis B = '
                     'Vt[:r].T (r = #sv > 1e-8*s0), '
                     'SIT = V - B B^T V (fp32)',
    'exclusion': 'VERBATIM 3040 degenerate-row rule: '
                 'residual norm < 1e-6 excluded from '
                 'ALL statistics and permutation '
                 'universes',
    'T1': 'per-word residual spectrum: word set W3 = '
          'words with >= %d non-degenerate occurrences '
          'at L3; per word, eigenvalues lam of the gram '
          'matrix SIT_w SIT_w^T; PR_w = (sum lam)^2 / '
          'sum(lam^2); stat_T1 = median PR_w over W3; '
          'also e1_w = lam_max/sum lam reported; null = '
          'construction-matched label permutation '
          '(multiset preserved): pseudo-group means in '
          'V space, deviations re-projected onto the '
          'SAME fixed complement basis B3, per-word PR '
          'on the same group sizes; N_PERM=%d seed %d; '
          'one-sided p = P(null <= obs); rank-1 claim '
          'iff p<0.05 (obs below null)'
          % (MIN_GRP, N_PERM, SEED_T1),
    'T2': 'same-prefix minimal pairs: Lshare(i,j) = '
          'length of the common immediate-left token '
          'prefix of the two occurrence positions '
          '(identical consecutive tokens strictly '
          'before the occurrence across the two '
          'prompts); population = same-word '
          'cross-prompt non-degenerate pairs (sw); '
          'stat_T2a = OLS slope of cos_dev ~ Lshare '
          'over ALL sw pairs, bootstrap 95%% CI '
          '(%d resamples seed %d); stat_T2b = '
          'med(cos | Lshare>=%d) - med(cos | all sw), '
          'exact random-subset null (equal-size subsets '
          'of sw pairs, N_SUB=%d seed %d, empty subset '
          'scores -1.0); pre-registered fallback '
          'threshold Lshare>=%d if fewer than 3 pairs '
          'at Lshare>=%d; if still <3 pairs: NaN + '
          'insufficient-pairs note (declared '
          'unreachability); prediction from the 3040 '
          "'was' cos=+1.0 observation: slope>0 and "
          'subset stat>0 (identical left context '
          'reproduces the situational residual)'
          % (N_BOOT, SEED_BOOT, LSHARE_MAIN, N_SUB,
             SEED_T2, LSHARE_FALLBACK, LSHARE_MAIN),
    'T3': 'axis identity across words: per word in W3, '
          'principal axis a_w = first left singular '
          'direction of its residual matrix (via '
          'eigenvector of the gram matrix, sign-fixed '
          'positive sum); stat_T3 = median '
          'off-diagonal |cos(a_w, a_v)| over '
          'distinct word pairs in W3; null = random '
          'Gaussian residual sets of the SAME group '
          'sizes in the same ambient dimension '
          '(N_RND=%d seed %d); one-sided p = '
          'P(null >= obs); shared axis iff p<0.05 '
          '(obs above null); private axes = obs at or '
          'below null' % (N_RND, SEED_T3),
    'T4': 'layer control: T1 repeated at L20 with the '
          'L20 word-mean subspace, the L20 degenerate '
          'mask, and the SAME L3-defined word set W3 '
          '(words with <3 non-degenerate L20 rows are '
          'skipped and counted); seed %d; relay '
          'specificity of the rank structure'
          % SEED_L20,
    'verdict_tree': 'rank1 = (p_T1<0.05); shared = '
                    '(p_T3<0.05); prefix_flag = '
                    '(p_T2b<0.05 AND stat_T2b>0), '
                    'reported but NOT in the verdict '
                    'name; rank1 AND shared -> '
                    'sitaxis_shared_rank1_qwen; rank1 '
                    'AND NOT shared -> '
                    'sitaxis_private_rank1_qwen; NOT '
                    'rank1 -> sitaxis_multirank_qwen; '
                    'single branch assignment',
    'anchors': 'a79 duplicate prefill prompt0 K3/V3 '
               'bit-identical (0.0); a80 full duplicate '
               'extraction all 28 prompts max abs diff '
               '0.0; a81 manual final-norm+lm_head '
               'recompute vs prefill logits (top-2 '
               'identity AND max|dlogit| <= 0.15, a51 '
               'family; near-tie skip note if gap<0.05); '
               'a82 projection basis orthonormality '
               'max|B^T B - I| <= 1e-5 at BOTH layers; '
               'a83 source seals 3037/3038/3039/3040 '
               'sha8 match seal.json; a84 cross-phase '
               'bit anchor: 25 target-word occurrences '
               'matched to 3037 npz by (word, prompt, '
               'pos), max|dV3| == 0.0; a85 full-bank '
               'chain anchor vs 3040 npz: V3 AND V20 '
               'bit-identical over ALL occurrences '
               '(max diff == 0.0) and SIT3 max diff '
               '<= 1e-12 (derived-quantity tolerance; '
               'extraction quantities require exact 0)',
    'control': 'construction-matched permutation is the '
               'exact null for T1/T4; random-subset '
               'randomization is the exact null for T2b; '
               'size-matched Gaussian sets are the '
               'Monte Carlo null for T3; no intervention',
    'statistics_discipline': 'all arrays pre-initialized '
                             '(3020 lesson); degenerate-row '
                             'exclusion everywhere (3040 '
                             'run2 lesson); construction-'
                             'matched reconstruction for '
                             'group-centered quantities '
                             '(3040 run3 lesson); verdict '
                             'criteria evaluated in one '
                             'branch assignment; word '
                             'sign-fixing (positive sum) '
                             'before axis cosines so the '
                             'statistic is sign-invariant',
    'corrections': 'none (first run; run1 of 3040 and '
                   'run2/run3 lessons are baked into '
                   'the verbatim-3040 protocol)',
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
# rerun discipline: clear old artifacts
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


# ---------- tokenization + occurrence scan ----------
nP = len(ALL_PROMPTS)
tok_ids = []
for pr in ALL_PROMPTS:
    ids = tok(pr, add_special_tokens=False)[
        'input_ids']
    tok_ids.append(list(int(x) for x in ids))
plen = np.array([len(x) for x in tok_ids],
                dtype=np.int64)
word_tok = {}
for w in TARGETS:
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    assert len(wi) == 1, (w, wi)
    word_tok[w] = int(wi[0])
tid2word = {v: k for k, v in word_tok.items()}
log('word token ids=%s' % word_tok)

ext_occ = []
for pi in range(nP):
    ids = tok_ids[pi]
    for pos, t in enumerate(ids):
        s = tok.decode([t]).strip()
        if s.isalpha() and len(s) >= 2:
            ext_occ.append({'tid': int(t),
                            'prompt': pi,
                            'pos': pos})
cnt = {}
for x in ext_occ:
    cnt[x['tid']] = cnt.get(x['tid'], 0) + 1
keep = sorted(t for t, c in cnt.items() if c >= MIN_OCC)
occ = [x for x in ext_occ if x['tid'] in set(keep)]
n_occ = len(occ)
n_types = len(keep)
occ_tid = np.array([x['tid'] for x in occ],
                   dtype=np.int64)
occ_pr = np.array([x['prompt'] for x in occ],
                  dtype=np.int64)
occ_pos = np.array([x['pos'] for x in occ],
                   dtype=np.int64)
tidx_of = {t: i for i, t in enumerate(keep)}
occ_w = np.array([tidx_of[t] for t in occ_tid],
                 dtype=np.int64)
log('ext bank: %d occurrences, %d types '
    '(>= %d occ); target occ=%d'
    % (n_occ, n_types, MIN_OCC,
       int(sum(1 for t in occ_tid
               if t in tid2word))))

# a79: duplicate prefill prompt0 (bit determinism)
p_b0, lg_b0, kv_b0, xf_b0 = prefill_extract(
    tok_ids[0], cap_fin=True)
p_d0, lg_d0, kv_d0, _ = prefill_extract(
    tok_ids[0])
a79_diff = 0.0
for li in LAYERS_EX:
    for a, b in ((kv_b0[li][0], kv_d0[li][0]),
                 (kv_b0[li][1], kv_d0[li][1])):
        a79_diff = max(a79_diff, float(
            np.max(np.abs(a - b))))
a79_diff = float(a79_diff)

# ---------- main extraction (pass 1) ----------
Ps = [None] * nP
KVs = {li: [None] * nP for li in LAYERS_EX}
for pi in range(nP):
    p, _, kv, _ = prefill_extract(tok_ids[pi])
    Ps[pi] = p
    for li in LAYERS_EX:
        KVs[li][pi] = kv[li]

# a80: full duplicate extraction (bit determinism)
a80_diff = 0.0
for pi in range(nP):
    p2, _, kv2, _ = prefill_extract(tok_ids[pi])
    a80_diff = max(a80_diff, float(np.max(
        np.abs(Ps[pi] - p2))))
    for li in LAYERS_EX:
        for a, b in zip(KVs[li][pi], kv2[li]):
            a80_diff = max(a80_diff, float(
                np.max(np.abs(a - b))))
a80_diff = float(a80_diff)
log('a79=%.3e a80=%.3e' % (a79_diff, a80_diff))

# a81: manual final-norm+lm_head recompute (prompt0)
w_norm = model.model.norm.weight.detach() \
    .float().cpu().numpy()
h_fin = xf_b0
var = float((h_fin ** 2).mean())
hn = h_fin / np.sqrt(var + EPS)
lg_man = (W_U @ (hn * w_norm)).astype(np.float64)
a81_maxdiff = float(np.max(np.abs(lg_man - lg_b0)))
ord_m = np.argsort(-lg_man)
ord_o = np.argsort(-lg_b0)
srt = np.sort(lg_b0)
gap0 = float(srt[-1] - srt[-2])
if gap0 < 0.05:
    a81_top2_ok = True
    a81_note = 'near-tie skip (gap=%.4f)' % gap0
else:
    a81_top2_ok = bool(ord_m[0] == ord_o[0]
                       and ord_m[1] == ord_o[1])
    a81_note = ''
a81_ok = bool(a81_top2_ok
              and a81_maxdiff <= A81_GATE)
log('a81 top2=%s maxdiff=%.4f (gate %.2f) %s'
    % (a81_top2_ok, a81_maxdiff, A81_GATE,
       a81_note))

# a83: source seals
a83_detail = []
for ph, nm in (
        (3037, 'omega_p34_kv_situational_'
               'specificity_qwen'),
        (3038, 'omega_p35_reentrant_readout_qwen'),
        (3039, 'omega_p36_direct_logistic_'
               'replication_qwen'),
        (3040, 'omega_p37_situational_'
               'component_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a83_detail.append(bool(s == sealj['result_sha256_8']))
a83_ok = bool(a83_detail) and all(a83_detail)

# ---------- occurrence vectors ----------
V3o = np.zeros((n_occ, HDIM))
V20o = np.zeros((n_occ, HDIM))
for oi in range(n_occ):
    pi, pos = int(occ_pr[oi]), int(occ_pos[oi])
    V3o[oi] = KVs[3][pi][1][pos]
    V20o[oi] = KVs[20][pi][1][pos]

# a84: cross-phase bit anchor vs 3037 npz
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
a84_diff = 0.0
a84_matched = 0
for oi in range(n_occ):
    t = int(occ_tid[oi])
    if t not in tid2word:
        continue
    key = (tid2word[t], int(occ_pr[oi]),
           int(occ_pos[oi]))
    if key not in ref:
        continue
    a84_diff = max(a84_diff, float(np.max(
        np.abs(V3o[oi] - ref[key]))))
    a84_matched += 1
a84_diff = float(a84_diff)
a84_ok = bool(a84_matched == len(w37)
              and a84_diff == 0.0)
log('a84 matched=%d/%d max|dV3|=%.3e'
    % (a84_matched, len(w37), a84_diff))

# a85: full-bank chain anchor vs 3040 npz
z40 = np.load(os.path.join(
    BASE, 'phase3040',
    'omega_p37_situational_component_qwen',
    'omega_p37_situational_component_qwen.npz'),
    allow_pickle=True)
V3_40 = z40['V3']
V20_40 = z40['V20']
SIT3_40 = z40['SIT3']
a85_shape_ok = bool(V3_40.shape == V3o.shape
                    and V20_40.shape == V20o.shape)
a85_v_diff = float(np.max(np.abs(V3o - V3_40))) \
    if a85_shape_ok else float('nan')
a85_v20_diff = float(np.max(np.abs(V20o - V20_40))) \
    if a85_shape_ok else float('nan')


def make_sub(Vo):
    WM = np.zeros((n_types, HDIM))
    for wi in range(n_types):
        WM[wi] = Vo[occ_w == wi].mean(axis=0)
    U, S, Vt = np.linalg.svd(WM,
                             full_matrices=False)
    r = int(np.sum(S > 1e-8 * S[0]))
    B = Vt[:r].T.copy()
    orth = float(np.max(np.abs(
        B.T @ B - np.eye(r))))
    proj = (Vo @ B) @ B.T
    SIT = Vo - proj
    ew = ((Vo @ B) ** 2).sum(axis=1) \
        / np.maximum((Vo ** 2).sum(axis=1), 1e-30)
    return B, orth, SIT, ew, r


B3, a82_o3, SIT3, ew3, r3 = make_sub(V3o)
B20, a82_o20, SIT20, ew20, r20 = make_sub(V20o)
a82_ok = bool(max(a82_o3, a82_o20) <= A82_GATE)
log('subspace: r3=%d orth3=%.2e r20=%d orth20=%.2e '
    'ok=%s' % (r3, a82_o3, r20, a82_o20, a82_ok))
log('energy share (word subspace): med_L3=%.4f '
    'med_L20=%.4f' % (float(np.median(ew3)),
                      float(np.median(ew20))))

if a85_shape_ok:
    a85_sit_diff = float(np.max(
        np.abs(SIT3 - SIT3_40)))
a85_ok = bool(a85_shape_ok
              and a85_v_diff == 0.0
              and a85_v20_diff == 0.0
              and a85_sit_diff <= 1e-12)
log('a85 shape_ok=%s dV3=%.3e dV20=%.3e dSIT3=%.3e'
    % (a85_shape_ok, a85_v_diff, a85_v20_diff,
       a85_sit_diff))

# ---------- degenerate-row masks ----------
nrm3 = np.linalg.norm(SIT3, axis=1)
nrm20 = np.linalg.norm(SIT20, axis=1)
ok3 = nrm3 >= DEGEN_NORM
ok20 = nrm20 >= DEGEN_NORM
log('degenerate rows: L3 %d/%d, L20 %d/%d '
    '(norm < %g)'
    % (int((~ok3).sum()), n_occ,
       int((~ok20).sum()), n_occ, DEGEN_NORM))

rows3 = np.where(ok3)[0]
m3 = len(rows3)
X3 = SIT3[rows3]
X3n = X3 / np.maximum(
    np.linalg.norm(X3, axis=1), 1e-12)[:, None]
lab3 = occ_w[rows3]
pr3 = occ_pr[rows3]
counts3 = np.bincount(lab3, minlength=n_types)
W3 = [wi for wi in range(n_types)
      if counts3[wi] >= MIN_GRP]
log('W3 (>= %d non-degenerate occ at L3): %d words'
    % (MIN_GRP, len(W3)))
log('W3 words: %s' % json.dumps(
    [tok.decode([int(keep[wi])]).strip()
     for wi in W3]))


def word_prs(resid, labs, word_list):
    """per-word participation ratio + e1 + axis."""
    PR = {}
    E1 = {}
    AX = {}
    for wi in word_list:
        Rw = resid[labs == wi]
        lam, U = np.linalg.eigh(Rw @ Rw.T)
        lam = lam[::-1]
        lam = np.maximum(lam, 0.0)
        s = lam.sum()
        PR[wi] = float(s * s
                       / np.maximum((lam ** 2).sum(),
                                    1e-30)) \
            if s > 0 else float('nan')
        E1[wi] = float(lam[0] / s) if s > 0 \
            else float('nan')
        v1 = U[:, -1]
        ax = v1 @ Rw
        n = np.linalg.norm(ax)
        ax = ax / n if n > 0 else ax
        if ax.sum() < 0:
            ax = -ax
        AX[wi] = ax
    return PR, E1, AX


PR3, E13, AX3 = word_prs(X3, lab3, W3)
obs_pr_t1 = float(np.median([PR3[wi] for wi in W3]))
obs_e1_t1 = float(np.median([E13[wi] for wi in W3]))
log('T1 obs: med PR=%.4f med e1=%.4f over %d words'
    % (obs_pr_t1, obs_e1_t1, len(W3)))

# ---------- T1 null (construction-matched) ----------
rng1 = np.random.default_rng(SEED_T1)
Vr3 = V3o[rows3]
perm_pr = np.zeros(N_PERM)
for it in range(N_PERM):
    lab = rng1.permutation(lab3)
    M = np.zeros((n_types, HDIM))
    np.add.at(M, lab, Vr3)
    Mu = M / np.maximum(counts3, 1)[:, None] \
        .astype(np.float64)
    dev = Vr3 - Mu[lab]
    TS = dev - (dev @ B3) @ B3.T
    prs_w = []
    for wi in W3:
        Rw = TS[lab == wi]
        lam = np.linalg.eigvalsh(Rw @ Rw.T)
        lam = np.maximum(lam[::-1], 0.0)
        s = lam.sum()
        prs_w.append(float(s * s
                           / np.maximum(
                               (lam ** 2).sum(),
                               1e-30)) if s > 0
                     else -1.0)
    perm_pr[it] = float(np.median(prs_w))
p_t1 = float(np.mean(perm_pr <= obs_pr_t1))
null_pr_med = float(np.median(perm_pr))
log('T1: obs med PR=%.4f null med=%.4f '
    'p(P(null<=obs))=%.5f'
    % (obs_pr_t1, null_pr_med, p_t1))

# ---------- T2: same-prefix minimal pairs ----------
iu, ju = np.triu_indices(m3, 1)
cos_r = np.einsum('ij,ij->i', X3n[iu], X3n[ju])
sw_m = (lab3[iu] == lab3[ju]) \
    & (pr3[iu] != pr3[ju])
sw_idx = np.where(sw_m)[0]


def lshare(a_row, b_row):
    ia = int(rows3[a_row])
    ib = int(rows3[b_row])
    pa = tok_ids[int(occ_pr[ia])][:int(occ_pos[ia])]
    pb = tok_ids[int(occ_pr[ib])][:int(occ_pos[ib])]
    k = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        k += 1
    return k


ls = np.array([lshare(int(iu[t]), int(ju[t]))
               for t in sw_idx], dtype=np.float64)
cs = cos_r[sw_idx]
log('T2: %d same-word cross-prompt pairs; Lshare '
    'hist=%s' % (len(cs), json.dumps(
        {int(v): int((ls == v).sum())
         for v in np.unique(ls)})))
sl_t2 = float(np.polyfit(ls, cs, 1)[0])
rngb = np.random.default_rng(SEED_BOOT)
boot_sl = np.zeros(N_BOOT)
for b in range(N_BOOT):
    idx = rngb.integers(0, len(cs), len(cs))
    boot_sl[b] = np.polyfit(ls[idx], cs[idx], 1)[0]
ci_t2 = (float(np.percentile(boot_sl, 2.5)),
         float(np.percentile(boot_sl, 97.5)))
thr = LSHARE_MAIN
hi_m = ls >= thr
if int(hi_m.sum()) < 3:
    thr = LSHARE_FALLBACK
    hi_m = ls >= thr
if int(hi_m.sum()) >= 3:
    obs_t2b = float(np.median(cs[hi_m])) \
        - float(np.median(cs))
    rng2 = np.random.default_rng(SEED_T2)
    nh = int(hi_m.sum())
    sub = np.zeros(N_SUB)
    for it in range(N_SUB):
        pick = rng2.choice(len(cs), nh, replace=False)
        sub[it] = float(np.median(cs[pick])) \
            - float(np.median(cs))
    p_t2b = float(np.mean(sub >= obs_t2b))
    n_hi = nh
else:
    obs_t2b = float('nan')
    p_t2b = float('nan')
    n_hi = int(hi_m.sum())
log('T2: slope=%.4f CI[%.4f,%.4f]; subset thr=%d '
    'n_hi=%d stat=%.4f p=%.5f'
    % (sl_t2, ci_t2[0], ci_t2[1], thr, n_hi,
       obs_t2b, p_t2b))

# ---------- T3: axis identity across words ----------
DAMB = HDIM - r3
AW = np.stack([AX3[wi] for wi in W3])
off = AW @ AW.T
off = np.abs(off)
iuw = np.triu_indices(len(W3), 1)
obs_t3 = float(np.median(off[iuw]))
rng3 = np.random.default_rng(SEED_T3)
sizes = [int(counts3[wi]) for wi in W3]
null_t3 = np.zeros(N_RND)
for it in range(N_RND):
    axes = []
    for nw in sizes:
        Rw = rng3.standard_normal((nw, DAMB))
        lam, U = np.linalg.eigh(Rw @ Rw.T)
        v1 = U[:, -1]
        ax = v1 @ Rw
        n = np.linalg.norm(ax)
        ax = ax / n if n > 0 else ax
        axes.append(ax)
    A = np.stack(axes)
    C = np.abs(A @ A.T)
    null_t3[it] = float(np.median(C[iuw]))
p_t3 = float(np.mean(null_t3 >= obs_t3))
log('T3: obs med |cos|=%.4f null med=%.4f '
    'p(P(null>=obs))=%.5f (ambient dim %d, %d words)'
    % (obs_t3, float(np.median(null_t3)), p_t3,
       DAMB, len(W3)))

# ---------- T4: L20 control (T1 only) ----------
rows20 = np.where(ok20)[0]
X20 = SIT20[rows20]
lab20 = occ_w[rows20]
cnt20 = np.bincount(lab20, minlength=n_types)
W3_20 = [wi for wi in W3 if cnt20[wi] >= MIN_GRP]
PR20, E120, _ = word_prs(X20, lab20, W3_20)
obs_pr_20 = float(np.median([PR20[wi]
                             for wi in W3_20])) \
    if W3_20 else float('nan')
rng4 = np.random.default_rng(SEED_L20)
Vr20 = V20o[rows20]
counts20 = np.bincount(lab20, minlength=n_types)
perm20 = np.zeros(N_PERM)
skipped20 = 0
for it in range(N_PERM):
    lab = rng4.permutation(lab20)
    M = np.zeros((n_types, HDIM))
    np.add.at(M, lab, Vr20)
    Mu = M / np.maximum(counts20, 1)[:, None] \
        .astype(np.float64)
    dev = Vr20 - Mu[lab]
    TS = dev - (dev @ B20) @ B20.T
    prs_w = []
    for wi in W3_20:
        Rw = TS[lab == wi]
        lam = np.linalg.eigvalsh(Rw @ Rw.T)
        lam = np.maximum(lam[::-1], 0.0)
        s = lam.sum()
        prs_w.append(float(s * s
                           / np.maximum(
                               (lam ** 2).sum(),
                               1e-30)) if s > 0
                     else -1.0)
    perm20[it] = float(np.median(prs_w))
p_t4 = float(np.mean(perm20 <= obs_pr_20)) \
    if W3_20 else float('nan')
log('T4 L20: %d/%d W3 words with >=%d ok rows; '
    'obs med PR=%.4f null med=%.4f p=%.5f'
    % (len(W3_20), len(W3), MIN_GRP, obs_pr_20,
       float(np.median(perm20)), p_t4))

# descriptive: per-word PR table
pr_tab = []
for wi in W3:
    pr_tab.append({
        'word': tok.decode([int(keep[wi])]).strip(),
        'n_ok': int(counts3[wi]),
        'PR': PR3[wi], 'e1': E13[wi]})
log('per-word spectrum: %s' % json.dumps(pr_tab))

# ---------- verdict ----------
a_ok = bool(a79_diff == 0.0 and a80_diff == 0.0
            and a81_ok and a82_ok and a83_ok
            and a84_ok and a85_ok)
rank1 = bool(p_t1 < 0.05)
shared = bool(p_t3 < 0.05)
prefix_flag = bool(p_t2b == p_t2b and p_t2b < 0.05
                   and obs_t2b == obs_t2b
                   and obs_t2b > 0)
if rank1 and shared:
    verdict = 'sitaxis_shared_rank1_qwen'
elif rank1:
    verdict = 'sitaxis_private_rank1_qwen'
else:
    verdict = 'sitaxis_multirank_qwen'

log('=== verdict ===')
log('a79=%r a80=%r a81_ok=%s (%.4f) a82_ok=%s '
    '(%.2e) a83=%s a84_ok=%s (%.3e, %d) a85_ok=%s '
    '(dV3=%.1e dV20=%.1e dSIT=%.1e)'
    % (a79_diff, a80_diff, a81_ok, a81_maxdiff,
       a82_ok, max(a82_o3, a82_o20), a83_ok,
       a84_ok, a84_diff, a84_matched, a85_ok,
       a85_v_diff, a85_v20_diff, a85_sit_diff))
log('T1 p=%.5f | T3 p=%.5f | T2b p=%.5f flag=%s'
    % (p_t1, p_t3, p_t2b, prefix_flag))
log('VERDICT=%s anchor_all_ok=%s' % (verdict, a_ok))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    prompts=np.array(ALL_PROMPTS),
    occ_tid=occ_tid, occ_prompt=occ_pr,
    occ_pos=occ_pos, occ_w=occ_w,
    types=np.array(keep), plen=plen,
    V3=V3o, V20=V20o, SIT3=SIT3, SIT20=SIT20,
    ok3=ok3, ok20=ok20,
    w3=np.array(W3),
    pr3_word=np.array([PR3[wi] for wi in W3]),
    e1_word=np.array([E13[wi] for wi in W3]),
    counts3_word=np.array([counts3[wi]
                           for wi in W3]),
    obs_pr_t1=np.float64(obs_pr_t1),
    obs_e1_t1=np.float64(obs_e1_t1),
    null_pr_med=np.float64(null_pr_med),
    p_t1=np.float64(p_t1),
    sw_pair_cos=cs, sw_lshare=ls,
    slope_t2=np.float64(sl_t2),
    ci_t2=np.array(ci_t2),
    t2_thr=np.int64(thr), n_hi=np.int64(n_hi),
    obs_t2b=np.float64(obs_t2b),
    p_t2b=np.float64(p_t2b),
    obs_t3=np.float64(obs_t3),
    null_t3_med=np.float64(np.median(null_t3)),
    p_t3=np.float64(p_t3),
    d_ambient=np.int64(DAMB),
    obs_pr_20=np.float64(obs_pr_20),
    p_t4=np.float64(p_t4),
    n_w3_20=np.int64(len(W3_20)),
    a79_diff=np.float64(a79_diff),
    a80_diff=np.float64(a80_diff),
    a81_maxdiff=np.float64(a81_maxdiff),
    a81_top2_ok=np.bool_(a81_top2_ok),
    a82_max_orth=np.float64(max(a82_o3, a82_o20)),
    a83_ok=np.bool_(a83_ok),
    a84_diff=np.float64(a84_diff),
    a84_matched=np.int64(a84_matched),
    a85_v_diff=np.float64(a85_v_diff),
    a85_v20_diff=np.float64(a85_v20_diff),
    a85_sit_diff=np.float64(a85_sit_diff),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'run': 'run1 authoritative',
    'anchor_all_ok': a_ok,
    'anchors': {
        'a79_dup_prefill_bit': a79_diff,
        'a80_dup_all_bit': a80_diff,
        'a81_top2_ok': a81_top2_ok,
        'a81_maxdiff': a81_maxdiff,
        'a81_gate': A81_GATE,
        'a81_note': a81_note,
        'a82_basis_orth_max': max(a82_o3, a82_o20),
        'a82_gate': A82_GATE,
        'a83_source_seals': a83_ok,
        'a84_cross_phase_bit': a84_diff,
        'a84_matched': a84_matched,
        'a85_v3_bit': a85_v_diff,
        'a85_v20_bit': a85_v20_diff,
        'a85_sit3_diff': a85_sit_diff,
        'a85_gate_sit': 1e-12,
    },
    'W3': {
        'n_words': len(W3),
        'words': [tok.decode([int(keep[wi])]).strip()
                  for wi in W3],
        'group_sizes': [int(counts3[wi])
                        for wi in W3],
    },
    'T1_rank_spectrum': {
        'obs_med_PR': obs_pr_t1,
        'obs_med_e1': obs_e1_t1,
        'null_med_PR': null_pr_med,
        'p_t1': p_t1,
        'per_word': pr_tab,
    },
    'T2_prefix_pairs': {
        'n_sw_pairs': int(len(cs)),
        'lshare_hist': {int(v): int((ls == v).sum())
                        for v in np.unique(ls)},
        'slope': sl_t2, 'ci_slope': ci_t2,
        'threshold': thr, 'n_high': n_hi,
        'stat_subset': obs_t2b, 'p_subset': p_t2b,
    },
    'T3_axis_identity': {
        'obs_med_abs_cos': obs_t3,
        'null_med_abs_cos':
            float(np.median(null_t3)),
        'p_t3': p_t3, 'd_ambient': DAMB,
        'n_words': len(W3),
    },
    'T4_layer20': {
        'n_words_ok': len(W3_20),
        'obs_med_PR': obs_pr_20,
        'null_med_PR': float(np.median(perm20)),
        'p_t4': p_t4,
    },
    'flags': {'rank1': rank1, 'shared': shared,
              'prefix_flag': prefix_flag},
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
