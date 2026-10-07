# -*- coding: utf-8 -*-
# Phase 3040 - Omega-P37: situational component of the L3 KV write
# Phase 3037 established kv_mixed_qwen: the L3 V write of a given
# word is word-identity dominated (same-word cos 0.848-0.981 vs
# different-word null 0.554, grand_ratio 1.718) with a residual
# context modulation. This phase EXTRACTS that situational
# component: build the word-identity subspace (span of the
# per-token-id mean V vectors over the bank, SVD-orthonormalized),
# project it out, and test the residual SIT = V - P(V) for
# (T1) context lock: same-prompt different-word residual cosines
#     vs cross-prompt null (exact prompt-label permutation);
# (T2) residual word specificity with a CONSTRUCTION-MATCHED
#     permutation null: relabel word labels (multiset preserved),
#     recompute pseudo-group means in V space, re-project the
#     deviations onto the SAME fixed complement basis - this
#     reproduces the within-group centering structure (sum-to-
#     zero; forced cos=-1 for size-2 groups) exactly, which a
#     plain pair-subset permutation does NOT;
# (T3) decay anatomy: cos_sit vs token distance dpos / dpfrac
#     (OLS + bootstrap CI + dpos bins);
# (T4) layer control at L20 (relay specificity).
# Bank and extraction protocol are VERBATIM 3037 (28 prompts, 8
# logic words, eager attention, bf16, kv head 7, layers 3/20, V
# PRIMARY); extended alphabetic-token bank (token ids with >=2
# occurrences, 3037 T2 scan convention) supplies the same-prompt
# different-word pairs that the 8-word target set cannot.
# DEGENERATE-ROW EXCLUSION: occurrences whose residual norm is
# < 1e-6 (stereotyped writes with V exactly in the word-mean
# span at fp32) are excluded from all pair statistics and
# permutation universes - their cosine is undefined and an
# exact-0.0 guard would poison every median (run2 lesson).
# PREREG frozen below BEFORE any observation.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3040
NAME = 'omega_p37_situational_component_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
HID = 2560
KV_HEAD = 7
LAYERS_EX = (3, 20)
N_PERM = 100000
SEED_MAIN = 3009
SEED_T1 = 9040
SEED_T2 = 9041
SEED_BOOT = 9042
SEED_T1_20 = 9043
SEED_T2_20 = 9044
MIN_OCC = 2
A75_GATE = 0.15
A76_GATE = 1e-5
DEGEN_NORM = 1e-6

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
            '3009); per prompt ONE prefill with '
            'use_cache=True; V read from '
            'past.layers[li].values[0, kv7] per '
            'occurrence position; layers (3, 20); V '
            'PRIMARY (K RoPE-confounded, not used in '
            'tests); bank + scan convention VERBATIM '
            '3037',
    'question': 'situational component extraction '
                '(3040 A main line): after removing the '
                'word-identity subspace from the L3 V '
                'write, does the residual encode the '
                'prompt context (context lock), and does '
                'it retain word-specific structure? '
                'Direct quantification of the 3037 '
                'kv_mixed residual modulation',
    'bank': 'extended alphabetic-token bank: every '
            'token of the 28-prompt 3037 bank whose '
            'decoded form strip().isalpha() and len>=2 '
            '(3037 T2 convention), restricted to token '
            'ids with >=2 occurrences (word means '
            'well-defined); capitalized variants are '
            'distinct token ids (distinct words); 8 '
            'logic targets are a subset (a78 bit anchor '
            'vs 3037 npz)',
    'decomposition': 'per layer: word-mean matrix WM '
                     '(n_types x head_dim) of V vectors; '
                     'SVD rank-r orthonormal basis B of '
                     'the ROW space of WM (B = Vt[:r].T, '
                     'r = #singular values > 1e-8*s0); '
                     'SIT = V - B B^T V (signed residual, '
                     'fp32); energy share ew = ||B^T '
                     'V||^2 / ||V||^2 per occurrence',
    'exclusion': 'DEGENERATE-ROW RULE: occurrences with '
                 'residual norm < 1e-6 (V exactly in the '
                 'word-mean span at fp32 = stereotyped '
                 'write) are excluded from ALL pair '
                 'statistics and from the permutation '
                 'universes; per-type degenerate counts '
                 'reported; motivation: undefined cosine '
                 'and exact-0.0 guard poisoning (run2)',
    'T1': 'PRIMARY context lock: different-word pairs '
          'among NON-degenerate rows split same-prompt '
          'vs cross-prompt; stat d1 = med(cos_sit '
          'same-prompt) - med(cos_sit cross-prompt), '
          'SIGNED cosine; null = exact permutation of '
          'prompt labels among non-degenerate rows '
          '(word labels fixed; positions travel with '
          'occurrences; empty same-prompt subset scores '
          '-1.0); N_PERM=100000 seed 9040; one-sided p; '
          'T1 significant iff p<0.05 AND d1>0',
    'T2': 'residual word specificity with '
          'CONSTRUCTION-MATCHED null: stat d2 = '
          'med(cos same-word cross-prompt) - med(cos '
          'different-word cross-prompt) on REAL '
          'residuals over non-degenerate rows; null = '
          'word-label permutation (multiset preserved) '
          'with FULL reconstruction: pseudo-group means '
          'in V space, deviations re-projected onto the '
          'SAME fixed complement basis B, pair cosines '
          'on pseudo-residuals - reproduces the within-'
          'group centering structure (sum-to-zero; '
          'forced cos=-1 for size-2 groups) exactly, '
          'which a plain pair-subset permutation does '
          'NOT; N_PERM=100000 seed 9041; one-sided p; '
          'word-residual significant iff p<0.05 AND '
          'd2>0',
    'T3': 'DESCRIPTIVE decay anatomy on cross-prompt '
          'different-word non-degenerate pairs: OLS '
          'cos_sit ~ dpos and cos_sit ~ |dpfrac| '
          '(dpfrac = |pos/L_i - pos/L_j|), bootstrap '
          '95% CI (10000 resamples seed 9042); median '
          'cos_sit in dpos bins [1-3]/[4-8]/[9-16]/'
          '[17+]; distinguishes global prompt-identity '
          'encoding (flat in dpos) from local '
          'attention-window mixing (decaying in dpos)',
    'T4': 'layer control: T1 and T2 repeated at L20 '
          'with the L20 word-mean subspace and its own '
          'degenerate-row mask (seeds 9043/9044); relay '
          'specificity of any context lock',
    'verdict_tree': 'ctx = (p1<0.05 AND d1>0); wres = '
                    '(p2<0.05 AND d2>0); ctx AND NOT wres '
                    '-> sitcomp_context_pure_qwen; ctx '
                    'AND wres -> '
                    'sitcomp_context_plus_wordres_qwen; '
                    'NOT ctx AND wres -> '
                    'sitcomp_wordres_only_qwen; else -> '
                    'sitcomp_null_qwen',
    'anchors': 'a73 duplicate prefill prompt0 K3/V3 '
               'bit-identical (0.0); a74 full duplicate '
               'extraction all 28 prompts max abs diff '
               '0.0; a75 manual final-norm+lm_head '
               'recompute vs prefill logits (top-2 '
               'identity AND max|dlogit| <= 0.15, a51 '
               'family; near-tie skip note if gap<0.05); '
               'a76 projection basis orthonormality '
               'max|B^T B - I| <= 1e-5 at BOTH layers; '
               'a77 source seals 3037/3038/3039 sha8 '
               'match seal.json; a78 cross-phase bit '
               'anchor: 25 target-word occurrences '
               'matched to 3037 npz by (word, prompt, '
               'pos), max|dV3| == 0.0',
    'control': 'cross-prompt different-word pairs are '
               'the within-phase empirical null for T1; '
               'label-preserved permutations (T1) and '
               'construction-matched reconstruction '
               'permutations (T2) are the exact nulls; '
               'no intervention, no sham prompts needed',
    'statistics_discipline': 'signed cosines for '
                             'residuals (residual null '
                             'is ~0, abs would bias '
                             'positive); all arrays '
                             'pre-initialized (3020 '
                             'lesson); permutation '
                             'statistic defined on '
                             'precomputed pair cosines '
                             '(T1) or per-permutation '
                             'reconstruction (T2); '
                             'verdict criteria evaluated '
                             'in one branch assignment '
                             '(no sequential re-judging)',
    'corrections': 'run1 crashed at the subspace build '
                   '(make_sub) BEFORE any verdict '
                   'statistic was observed: SVD basis '
                   'taken from U (token-type space) '
                   'instead of Vt (residual-stream row '
                   'space); corrected to B = Vt[:r].T. '
                   'run2 completed with verdict '
                   'sitcomp_null_qwen but the verdict is '
                   'VOIDED by a demonstrated array '
                   'defect (probe3040): 32/92 occurrences '
                   'have residual norm < 1e-6 (stereotyped '
                   'writes, e.g. The x12) and the cosine '
                   'guard returned EXACT 0.0 for 2416/4186 '
                   'pairs, degenerating every median and '
                   'the permutation null (std=0) - p=1.0 '
                   'was a guard artifact, not a finding; '
                   'probe also showed strong nonzero '
                   'residual structure (|cos| std 0.125, '
                   'same-prompt diff-word 75th pct +0.31) '
                   'and a centering baseline for same-'
                   'word pairs (targets med -0.476 ~ '
                   '-1/(n-1)); run3 fixes: (i) degenerate-'
                   'row exclusion from all pair '
                   'statistics and permutation universes, '
                   '(ii) T2 null upgraded to the '
                   'construction-matched reconstruction '
                   'permutation; verdict tree, T1/T3/T4 '
                   'structure and anchors unchanged; '
                   'run3 authoritative',
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

# a73: duplicate prefill prompt0 (bit determinism)
p_b0, lg_b0, kv_b0, xf_b0 = prefill_extract(
    tok_ids[0], cap_fin=True)
p_d0, lg_d0, kv_d0, _ = prefill_extract(
    tok_ids[0])
a73_diff = 0.0
for li in LAYERS_EX:
    for a, b in ((kv_b0[li][0], kv_d0[li][0]),
                 (kv_b0[li][1], kv_d0[li][1])):
        a73_diff = max(a73_diff, float(
            np.max(np.abs(a - b))))
a73_diff = float(a73_diff)

# ---------- main extraction (pass 1) ----------
Ps = [None] * nP
KVs = {li: [None] * nP for li in LAYERS_EX}
for pi in range(nP):
    p, _, kv, _ = prefill_extract(tok_ids[pi])
    Ps[pi] = p
    for li in LAYERS_EX:
        KVs[li][pi] = kv[li]

# a74: full duplicate extraction (bit determinism)
a74_diff = 0.0
for pi in range(nP):
    p2, _, kv2, _ = prefill_extract(tok_ids[pi])
    a74_diff = max(a74_diff, float(np.max(
        np.abs(Ps[pi] - p2))))
    for li in LAYERS_EX:
        for a, b in zip(KVs[li][pi], kv2[li]):
            a74_diff = max(a74_diff, float(
                np.max(np.abs(a - b))))
a74_diff = float(a74_diff)
log('a73=%.3e a74=%.3e' % (a73_diff, a74_diff))

# a75: manual final-norm+lm_head recompute (prompt0)
w_norm = model.model.norm.weight.detach() \
    .float().cpu().numpy()
h_fin = xf_b0
var = float((h_fin ** 2).mean())
hn = h_fin / np.sqrt(var + EPS)
lg_man = (W_U @ (hn * w_norm)).astype(np.float64)
a75_maxdiff = float(np.max(np.abs(lg_man - lg_b0)))
ord_m = np.argsort(-lg_man)
ord_o = np.argsort(-lg_b0)
srt = np.sort(lg_b0)
gap0 = float(srt[-1] - srt[-2])
if gap0 < 0.05:
    a75_top2_ok = True
    a75_note = 'near-tie skip (gap=%.4f)' % gap0
else:
    a75_top2_ok = bool(ord_m[0] == ord_o[0]
                       and ord_m[1] == ord_o[1])
    a75_note = ''
a75_ok = bool(a75_top2_ok
              and a75_maxdiff <= A75_GATE)
log('a75 top2=%s maxdiff=%.4f (gate %.2f) %s'
    % (a75_top2_ok, a75_maxdiff, A75_GATE,
       a75_note))

# a77: source seals
a77_detail = []
for ph, nm in (
        (3037, 'omega_p34_kv_situational_'
               'specificity_qwen'),
        (3038, 'omega_p35_reentrant_readout_qwen'),
        (3039, 'omega_p36_direct_logistic_'
               'replication_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a77_detail.append(bool(s == sealj['result_sha256_8']))
a77_ok = bool(a77_detail) and all(a77_detail)

# ---------- occurrence vectors ----------
V3o = np.zeros((n_occ, HDIM))
V20o = np.zeros((n_occ, HDIM))
for oi in range(n_occ):
    pi, pos = int(occ_pr[oi]), int(occ_pos[oi])
    V3o[oi] = KVs[3][pi][1][pos]
    V20o[oi] = KVs[20][pi][1][pos]

# a78: cross-phase bit anchor vs 3037 npz
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
a78_diff = 0.0
a78_matched = 0
for oi in range(n_occ):
    t = int(occ_tid[oi])
    if t not in tid2word:
        continue
    key = (tid2word[t], int(occ_pr[oi]),
           int(occ_pos[oi]))
    if key not in ref:
        continue
    a78_diff = max(a78_diff, float(np.max(
        np.abs(V3o[oi] - ref[key]))))
    a78_matched += 1
a78_diff = float(a78_diff)
a78_ok = bool(a78_matched == len(w37)
              and a78_diff == 0.0)
log('a78 matched=%d/%d max|dV3|=%.3e'
    % (a78_matched, len(w37), a78_diff))


def make_sub(Vo):
    WM = np.zeros((n_types, HDIM))
    for wi in range(n_types):
        WM[wi] = Vo[occ_w == wi].mean(axis=0)
    U, S, Vt = np.linalg.svd(WM,
                             full_matrices=False)
    r = int(np.sum(S > 1e-8 * S[0]))
    # basis of the ROW space of WM lives in
    # R^HDIM: WM = U S Vt, Vt is (n_types x
    # HDIM), so the orthonormal basis columns
    # are the first r rows of Vt transposed
    B = Vt[:r].T.copy()
    orth = float(np.max(np.abs(
        B.T @ B - np.eye(r))))
    proj = (Vo @ B) @ B.T
    SIT = Vo - proj
    ew = ((Vo @ B) ** 2).sum(axis=1) \
        / np.maximum((Vo ** 2).sum(axis=1), 1e-30)
    G = WM @ WM.T
    nrm = np.sqrt(np.diag(G))
    wm_cos = G / np.maximum(
        np.outer(nrm, nrm), 1e-30)
    off = wm_cos[~np.eye(n_types, dtype=bool)]
    return B, orth, SIT, ew, r, \
        float(np.median(off))


B3, a76_o3, SIT3, ew3, r3, wmcos3 = make_sub(V3o)
B20, a76_o20, SIT20, ew20, r20, wmcos20 = \
    make_sub(V20o)
a76_ok = bool(max(a76_o3, a76_o20) <= A76_GATE)
log('subspace: r3=%d orth3=%.2e r20=%d orth20=%.2e '
    'ok=%s' % (r3, a76_o3, r20, a76_o20, a76_ok))
log('energy share (word subspace): med_L3=%.4f '
    'med_L20=%.4f' % (float(np.median(ew3)),
                      float(np.median(ew20))))
log('word-mean pairwise cos: med_L3=%.4f med_L20=%.4f'
    % (wmcos3, wmcos20))

# ---------- degenerate-row masks ----------
nrm3 = np.linalg.norm(SIT3, axis=1)
nrm20 = np.linalg.norm(SIT20, axis=1)
ok3 = nrm3 >= DEGEN_NORM
ok20 = nrm20 >= DEGEN_NORM
log('degenerate rows: L3 %d/%d, L20 %d/%d '
    '(norm < %g)'
    % (int((~ok3).sum()), n_occ,
       int((~ok20).sum()), n_occ, DEGEN_NORM))
deg_by_type3 = {}
deg_by_type20 = {}
for wi in range(n_types):
    msk = occ_w == wi
    nd3 = int((msk & ~ok3).sum())
    nd20 = int((msk & ~ok20).sum())
    if nd3:
        deg_by_type3[int(keep[wi])] = nd3
    if nd20:
        deg_by_type20[int(keep[wi])] = nd20
log('degenerate by tid L3: %s'
    % json.dumps(deg_by_type3))
log('degenerate by tid L20: %s'
    % json.dumps(deg_by_type20))


def layer_tests(Vmat, SIT, ok, B, seed_t1, seed_t2):
    """T1 context lock + T2 construction-matched word
    residual + pair arrays, non-degenerate rows only."""
    rows = np.where(ok)[0]
    m = len(rows)
    X = SIT[rows]
    Xn = X / np.maximum(
        np.linalg.norm(X, axis=1), 1e-12)[:, None]
    iu, ju = np.triu_indices(m, 1)
    cos_r = np.einsum('ij,ij->i', Xn[iu], Xn[ju])
    pr_r = occ_pr[rows]
    lab0 = occ_w[rows]
    pos_r = occ_pos[rows]
    pr_same0 = pr_r[iu] == pr_r[ju]
    pr_diff0 = ~pr_same0
    wd_same0 = lab0[iu] == lab0[ju]
    wd_diff0 = ~wd_same0
    sd_m = wd_diff0 & pr_same0
    cd_m = wd_diff0 & pr_diff0
    sw_m = wd_same0 & pr_diff0
    # T1: context lock
    if sd_m.any():
        obs_d1 = float(np.median(cos_r[sd_m])) \
            - float(np.median(cos_r[cd_m]))
    else:
        obs_d1 = float('nan')
    rng = np.random.default_rng(seed_t1)
    perm1 = np.zeros(N_PERM)
    for it in range(N_PERM):
        pl = rng.permutation(pr_r)
        sp = pl[iu] == pl[ju]
        ms = wd_diff0 & sp
        if ms.any():
            perm1[it] = float(np.median(cos_r[ms])) \
                - float(np.median(
                    cos_r[wd_diff0 & ~sp]))
        else:
            perm1[it] = -1.0
    p1 = float(np.mean(perm1 >= obs_d1)) \
        if sd_m.any() else float('nan')
    # T2: construction-matched null
    obs_d2 = float('nan')
    p2 = float('nan')
    ratio2 = float('nan')
    if sw_m.any() and cd_m.any():
        obs_d2 = float(np.median(cos_r[sw_m])) \
            - float(np.median(cos_r[cd_m]))
        Vr = Vmat[rows]
        counts0 = np.bincount(lab0,
                              minlength=n_types)
        rng2 = np.random.default_rng(seed_t2)
        perm2 = np.zeros(N_PERM)
        for it in range(N_PERM):
            lab = rng2.permutation(lab0)
            M = np.zeros((n_types, HDIM))
            np.add.at(M, lab, Vr)
            Mu = M / np.maximum(
                counts0, 1)[:, None].astype(
                np.float64)
            dev = Vr - Mu[lab]
            TS = dev - (dev @ B) @ B.T
            TSn = TS / np.maximum(
                np.linalg.norm(TS, axis=1),
                1e-12)[:, None]
            cn = np.einsum('ij,ij->i', TSn[iu],
                           TSn[ju])
            sp2 = lab[iu] == lab[ju]
            ms2 = sp2 & pr_diff0
            if ms2.any():
                perm2[it] = float(
                    np.median(cn[ms2])) \
                    - float(np.median(
                        cn[(~sp2) & pr_diff0]))
            else:
                perm2[it] = -1.0
        p2 = float(np.mean(perm2 >= obs_d2))
        ratio2 = float(np.median(cos_r[sw_m])) \
            / max(abs(float(np.median(
                cos_r[cd_m]))), 1e-30)
    med_same_sd = float(np.median(cos_r[sd_m])) \
        if sd_m.any() else float('nan')
    med_cross = float(np.median(cos_r[cd_m])) \
        if cd_m.any() else float('nan')
    med_sw = float(np.median(cos_r[sw_m])) \
        if sw_m.any() else float('nan')
    return {
        'rows': rows, 'iu': iu, 'ju': ju,
        'cos_r': cos_r, 'pr_same0': pr_same0,
        'wd_same0': wd_same0,
        'sd_m': sd_m, 'cd_m': cd_m, 'sw_m': sw_m,
        'pr_r': pr_r, 'lab0': lab0,
        'pos_r': pos_r,
        'obs_d1': obs_d1, 'p1': p1,
        'obs_d2': obs_d2, 'p2': p2,
        'ratio2': ratio2,
        'med_same_sd': med_same_sd,
        'med_cross': med_cross,
        'med_sw': med_sw,
    }


log('=== L3 tests ===')
R3 = layer_tests(V3o, SIT3, ok3, B3,
                 SEED_T1, SEED_T2)
log('L3 rows ok=%d/%d; pairs: sd=%d cd=%d sw=%d'
    % (len(R3['rows']), n_occ, int(R3['sd_m'].sum()),
       int(R3['cd_m'].sum()), int(R3['sw_m'].sum())))
log('T1 L3: d1=%.4f p1=%.5f (med_same=%.4f '
    'med_cross=%.4f)'
    % (R3['obs_d1'], R3['p1'], R3['med_same_sd'],
       R3['med_cross']))
log('T2 L3: d2=%.4f p2=%.5f ratio2=%.3f '
    '(med_sw=%.4f med_dw=%.4f)'
    % (R3['obs_d2'], R3['p2'], R3['ratio2'],
       R3['med_sw'], R3['med_cross']))

log('=== T3 decay anatomy (L3) ===')
cdx = np.where(R3['cd_m'])[0]
c3 = R3['cos_r'][cdx]
rows3 = R3['rows']
iu3, ju3 = R3['iu'], R3['ju']
dp = np.abs(R3['pos_r'][iu3] - R3['pos_r'][ju3])[cdx]
prs3 = R3['pr_r']
df_all = np.abs(
    R3['pos_r'][iu3].astype(np.float64)
    / plen[prs3[iu3]]
    - R3['pos_r'][ju3].astype(np.float64)
    / plen[prs3[ju3]])[cdx]
sl_pos = float(np.polyfit(dp.astype(np.float64),
                          c3, 1)[0])
sl_frac = float(np.polyfit(df_all, c3, 1)[0])
rngb = np.random.default_rng(SEED_BOOT)
n_cd = len(cdx)
boot_pos = np.zeros(10000)
boot_frac = np.zeros(10000)
for b in range(10000):
    idx = rngb.integers(0, n_cd, n_cd)
    boot_pos[b] = np.polyfit(dp[idx].astype(
        np.float64), c3[idx], 1)[0]
    boot_frac[b] = np.polyfit(df_all[idx], c3[idx],
                              1)[0]
ci_pos = (float(np.percentile(boot_pos, 2.5)),
          float(np.percentile(boot_pos, 97.5)))
ci_frac = (float(np.percentile(boot_frac, 2.5)),
           float(np.percentile(boot_frac, 97.5)))
bins = [(1, 3), (4, 8), (9, 16), (17, 10 ** 9)]
bin_med = []
for lo, hi in bins:
    msk = (dp >= lo) & (dp <= hi)
    bin_med.append(float(np.median(c3[msk]))
                   if msk.any() else float('nan'))
log('T3: slope_dpos=%.5f CI[%.5f,%.5f] '
    'slope_dpfrac=%.5f CI[%.5f,%.5f]'
    % (sl_pos, ci_pos[0], ci_pos[1],
       sl_frac, ci_frac[0], ci_frac[1]))
log('T3 bins: %s'
    % ' '.join('%.4f' % v for v in bin_med))

log('=== L20 control ===')
R20 = layer_tests(V20o, SIT20, ok20, B20,
                  SEED_T1_20, SEED_T2_20)
log('L20 rows ok=%d/%d; pairs: sd=%d cd=%d sw=%d'
    % (len(R20['rows']), n_occ,
       int(R20['sd_m'].sum()), int(R20['cd_m'].sum()),
       int(R20['sw_m'].sum())))
log('T4 L20: d1=%.4f p1=%.5f d2=%.4f p2=%.5f'
    % (R20['obs_d1'], R20['p1'], R20['obs_d2'],
       R20['p2']))

# descriptive: top |cos| residual pairs (L3, ok rows)
absc = np.abs(R3['cos_r'])
top5 = np.argsort(-absc)[:5]
top_desc = []
for t in top5:
    a = int(rows3[iu3[t]])
    b = int(rows3[ju3[t]])
    top_desc.append({
        'cos': float(R3['cos_r'][t]),
        'tok_a': tok.decode([int(occ_tid[a])]).strip(),
        'tok_b': tok.decode([int(occ_tid[b])]).strip(),
        'prompt_a': int(occ_pr[a]),
        'prompt_b': int(occ_pr[b]),
        'pos_a': int(occ_pos[a]),
        'pos_b': int(occ_pos[b])})
log('top |cos| pairs (L3): %s'
    % json.dumps(top_desc))

# ---------- verdict ----------
a_ok = bool(a73_diff == 0.0 and a74_diff == 0.0
            and a75_ok and a76_ok and a77_ok
            and a78_ok)
ctx = bool(R3['p1'] < 0.05 and R3['obs_d1'] > 0)
wres = bool(R3['p2'] < 0.05 and R3['obs_d2'] > 0)
if ctx and not wres:
    verdict = 'sitcomp_context_pure_qwen'
elif ctx and wres:
    verdict = 'sitcomp_context_plus_wordres_qwen'
elif wres:
    verdict = 'sitcomp_wordres_only_qwen'
else:
    verdict = 'sitcomp_null_qwen'

log('=== verdict ===')
log('a73=%r a74=%r a75_ok=%s (%.4f) a76_ok=%s '
    '(%.2e) a77=%s a78_ok=%s (%.3e, matched %d)'
    % (a73_diff, a74_diff, a75_ok, a75_maxdiff,
       a76_ok, max(a76_o3, a76_o20), a77_ok,
       a78_ok, a78_diff, a78_matched))
log('T1 d1=%.4f p1=%.5f | T2 d2=%.4f p2=%.5f '
    'ratio2=%.3f' % (R3['obs_d1'], R3['p1'],
                     R3['obs_d2'], R3['p2'],
                     R3['ratio2']))
log('VERDICT=%s anchor_all_ok=%s' % (verdict, a_ok))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    prompts=np.array(ALL_PROMPTS),
    occ_tid=occ_tid, occ_prompt=occ_pr,
    occ_pos=occ_pos, occ_w=occ_w,
    types=np.array(keep),
    plen=plen,
    V3=V3o, V20=V20o, SIT3=SIT3, SIT20=SIT20,
    ok3=ok3, ok20=ok20,
    rows3=rows3, rows20=R20['rows'],
    cos3_ok=R3['cos_r'], cos20_ok=R20['cos_r'],
    pair_i3=iu3, pair_j3=ju3,
    pair_i20=R20['iu'], pair_j20=R20['ju'],
    sd_m3=R3['sd_m'], cd_m3=R3['cd_m'],
    sw_m3=R3['sw_m'],
    obs_d1=np.float64(R3['obs_d1']),
    p1=np.float64(R3['p1']),
    obs_d2=np.float64(R3['obs_d2']),
    p2=np.float64(R3['p2']),
    ratio2=np.float64(R3['ratio2']),
    n_sd=int(R3['sd_m'].sum()),
    n_cd=int(R3['cd_m'].sum()),
    n_sw=int(R3['sw_m'].sum()),
    slope_dpos=np.float64(sl_pos),
    ci_pos=np.array(ci_pos),
    slope_dpfrac=np.float64(sl_frac),
    ci_frac=np.array(ci_frac),
    bin_med=np.array(bin_med),
    obs_d1_20=np.float64(R20['obs_d1']),
    p1_20=np.float64(R20['p1']),
    obs_d2_20=np.float64(R20['obs_d2']),
    p2_20=np.float64(R20['p2']),
    med_ew3=np.float64(np.median(ew3)),
    med_ew20=np.float64(np.median(ew20)),
    ew3=ew3, ew20=ew20,
    wmcos3=np.float64(wmcos3),
    wmcos20=np.float64(wmcos20),
    a73_diff=np.float64(a73_diff),
    a74_diff=np.float64(a74_diff),
    a75_maxdiff=np.float64(a75_maxdiff),
    a75_top2_ok=np.bool_(a75_top2_ok),
    a76_max_orth=np.float64(max(a76_o3, a76_o20)),
    a77_ok=np.bool_(a77_ok),
    a78_diff=np.float64(a78_diff),
    a78_matched=np.int64(a78_matched),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'run': 'run3 authoritative (run1 crash: SVD basis '
           'space bug, pre-verdict; run2 verdict '
           'VOIDED: exact-0.0 guard poisoned 2416/4186 '
           'pairs via 32 degenerate rows, probe3040 '
           'evidence)',
    'anchor_all_ok': a_ok,
    'anchors': {
        'a73_dup_prefill_bit': a73_diff,
        'a74_dup_all_bit': a74_diff,
        'a75_top2_ok': a75_top2_ok,
        'a75_maxdiff': a75_maxdiff,
        'a75_gate': A75_GATE,
        'a75_note': a75_note,
        'a76_basis_orth_max': max(a76_o3, a76_o20),
        'a76_gate': A76_GATE,
        'a77_source_seals': a77_ok,
        'a78_cross_phase_bit': a78_diff,
        'a78_matched': a78_matched,
    },
    'decomposition': {
        'n_occ': n_occ, 'n_types': n_types,
        'rank_L3': r3, 'rank_L20': r20,
        'med_energy_share_L3': float(np.median(ew3)),
        'med_energy_share_L20':
            float(np.median(ew20)),
        'med_wordmean_paircos_L3': wmcos3,
        'med_wordmean_paircos_L20': wmcos20,
        'degenerate_L3': int((~ok3).sum()),
        'degenerate_L20': int((~ok20).sum()),
        'degenerate_by_tid_L3': deg_by_type3,
        'degenerate_by_tid_L20': deg_by_type20,
    },
    'T1_context_lock': {
        'n_same_prompt_diff_word':
            int(R3['sd_m'].sum()),
        'n_cross_prompt_diff_word':
            int(R3['cd_m'].sum()),
        'med_same_prompt': R3['med_same_sd'],
        'med_cross_prompt': R3['med_cross'],
        'd1': R3['obs_d1'], 'p1': R3['p1'],
    },
    'T2_word_residual': {
        'n_same_word_cross': int(R3['sw_m'].sum()),
        'med_same_word': R3['med_sw'],
        'med_diff_word': R3['med_cross'],
        'd2': R3['obs_d2'], 'p2': R3['p2'],
        'ratio2': R3['ratio2'],
        'null': 'construction-matched reconstruction '
                'permutation',
    },
    'T3_decay': {
        'slope_dpos': sl_pos, 'ci_dpos': ci_pos,
        'slope_dpfrac': sl_frac,
        'ci_dpfrac': ci_frac,
        'bin_med': bin_med,
        'bin_edges': ['1-3', '4-8', '9-16', '17+'],
    },
    'T4_layer20': {
        'd1': R20['obs_d1'], 'p1': R20['p1'],
        'd2': R20['obs_d2'], 'p2': R20['p2'],
        'n_sd': int(R20['sd_m'].sum()),
        'n_sw': int(R20['sw_m'].sum()),
    },
    'top_abs_cos_pairs_L3': top_desc,
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
