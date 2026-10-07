# -*- coding: utf-8 -*-
# Phase 3037 - Omega-P34: situational specificity of L3 KV
# Same-word different-context K/V similarity at the L3 relay layer
# (kv head 7, the 3011 logic-gate head). Direct test of the 3013
# episodic-KV claim: are the L3 writes of a given word stereotyped
# (fixed direction, high same-word cosine) or episodic (context-
# conditioned, same-word cosine at the different-word null level)?
# K carries RoPE rotation -> position-confounded, so V is PRIMARY.
# Machine: pure prefill + cache read (no intervention chains).
# PREREG frozen below BEFORE any observation (execution.json
# written pre-run).
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3037
NAME = 'omega_p34_kv_situational_specificity_qwen'
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
PERM_SEED = 9037
RATIO_STEREO = 2.0
RATIO_EPI = 1.3

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
            'intervention, no injection hooks, eager '
            'attention, bf16); per prompt ONE prefill '
            'with use_cache=True; K/V read from '
            'past.layers[li].keys/values at last-position '
            'axis per occurrence; layers (3, 20); primary '
            'kv head 7 (3011 logic-gate group)',
    'question': 'situational specificity (3013 direct '
                'test): is the L3 KV write of a given '
                'word stereotyped (same fixed direction '
                'across contexts) or episodic (context-'
                'conditioned)? V is PRIMARY (K includes '
                'RoPE rotation -> position-confounded, '
                'reported descriptively)',
    'bank': '28 prompts = 12 GEN_PROMPTS (logic-word '
            'final) + 16 EXTRA minimal-pair prompts; 8 '
            'target words (because x4, however/while/'
            'although/therefore/yet/thus/so x3 each) '
            'located by exact lowercase space-prefixed '
            'token id scan (capitalized variants '
            'excluded by construction)',
    'T1': 'PRIMARY stereotypy: per target word, med '
          'cosine of V3 (kv7) over its cross-prompt '
          'same-word pairs; null = empirical different-'
          'word cross-prompt pair cosines; exact label '
          'permutation (N_PERM=100000, seed 9037) per '
          'word + maxT family correction across 8 '
          'words; ratio = med(same-word) / med(diff-'
          'word)',
    'T2': 'within-prompt repeats: automatic scan of all '
          '28 prompts for repeated alphabetic token '
          'ids; cosV3 / cosK3 vs position distance '
          '(descriptive, RoPE effect on K)',
    'T3': 'layer control: identical T1 statistics at '
          'L20 for every target word (relay-specificity '
          'of any stereotypy)',
    'verdict_tree': 'if family-pass (maxT p<0.05) >= 1 '
                    'word AND grand ratio >= 2.0 -> '
                    'kv_stereotyped_qwen (3013 '
                    'episodic falsified for V); elif '
                    'family-pass == 0 AND grand ratio '
                    '<= 1.3 -> kv_episodic_qwen (3013 '
                    'confirmed); else -> kv_mixed_qwen',
    'anchors': 'a58 duplicate prefill prompt0 K3/V3 '
               'bit-identical (0.0); a59 full duplicate '
               'extraction across all 28 prompts max '
               'abs diff 0.0; a60 manual final-norm+'
               'lm_head recompute vs prefill logits '
               '(top-2 identity AND max|dlogit| <= '
               '0.15, a51 bf16-family gate; near-tie '
               'skip note if gap<0.05); a61 source '
               'seals 3035/3036 sha8(result) match '
               'seal.json',
    'control': 'different-word cross-prompt pairs are '
               'the within-phase empirical null; no '
               'sham prompts needed (no intervention)',
    'corrections': 'run1 crashed pre-verdict '
                   '(grand_ratio pair-vs-word array '
                   'shape bug) and its preregistered '
                   'a60 was MIS-SPECIFIED: it compared '
                   'the 3036 TWO-STEP re-entrant '
                   'softmax (step-2 re-feeds the last '
                   'token at position L, attending to '
                   '0..L) against the direct prefill '
                   'readout at position L-1 - '
                   'different quantities by design '
                   '(run1 measured max dp 0.764, '
                   'registered as observation, not '
                   'anchor); corrected a60 = manual '
                   'final-norm+lm_head recompute of '
                   'the prefill readout (a51 family, '
                   'gate 0.15); grand_ratio fixed to '
                   'pair-level median; T1 statistics '
                   'unchanged by the correction; V '
                   'primary / K descriptive (RoPE '
                   'confound); all arrays pre-'
                   'initialized (3020 lesson)',
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

torch.manual_seed(3009)
np.random.seed(3009)

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
word_tok = {}
for w in TARGETS:
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    assert len(wi) == 1, (w, wi)
    word_tok[w] = int(wi[0])
log('word token ids=%s' % word_tok)

occ = []
for pi in range(nP):
    ids = tok_ids[pi]
    for w in TARGETS:
        tid = word_tok[w]
        for pos, t in enumerate(ids):
            if t == tid:
                occ.append({'word': w,
                            'prompt': pi,
                            'pos': pos})
n_occ = len(occ)
log('occurrences=%d over %d prompts' % (n_occ, nP))
for w in TARGETS:
    nw = sum(1 for x in occ if x['word'] == w)
    log('  %s: %d occurrences' % (w, nw))

# a58: duplicate prefill prompt0 (bit determinism)
p_b0, lg_b0, kv_b0, xf_b0 = prefill_extract(
    tok_ids[0], cap_fin=True)
p_d0, lg_d0, kv_d0, _ = prefill_extract(
    tok_ids[0])
a58_diff = 0.0
for li in LAYERS_EX:
    for a, b in ((kv_b0[li][0], kv_d0[li][0]),
                 (kv_b0[li][1], kv_d0[li][1])):
        a58_diff = max(a58_diff, float(
            np.max(np.abs(a - b))))
a58_diff = float(a58_diff)

# ---------- main extraction (pass 1) ----------
Ps = [None] * nP
KVs = {li: [None] * nP for li in LAYERS_EX}
for pi in range(nP):
    p, _, kv, _ = prefill_extract(tok_ids[pi])
    Ps[pi] = p
    for li in LAYERS_EX:
        KVs[li][pi] = kv[li]

# a59: full duplicate extraction (bit determinism)
a59_diff = 0.0
for pi in range(nP):
    p2, _, kv2, _ = prefill_extract(tok_ids[pi])
    a59_diff = max(a59_diff, float(np.max(
        np.abs(Ps[pi] - p2))))
    for li in LAYERS_EX:
        for a, b in zip(KVs[li][pi], kv2[li]):
            a59_diff = max(a59_diff, float(
                np.max(np.abs(a - b))))
a59_diff = float(a59_diff)
log('a58=%.3e a59=%.3e' % (a58_diff, a59_diff))

# a60: manual final-norm+lm_head recompute (prompt0)
w_norm = model.model.norm.weight.detach() \
    .float().cpu().numpy()
h_fin = xf_b0
var = float((h_fin ** 2).mean())
hn = h_fin / np.sqrt(var + EPS)
lg_man = (W_U @ (hn * w_norm)).astype(np.float64)
a60_maxdiff = float(np.max(np.abs(lg_man - lg_b0)))
ord_m = np.argsort(-lg_man)
ord_o = np.argsort(-lg_b0)
srt = np.sort(lg_b0)
gap0 = float(srt[-1] - srt[-2])
if gap0 < 0.05:
    a60_top2_ok = True
    a60_note = 'near-tie skip (gap=%.4f)' % gap0
else:
    a60_top2_ok = bool(ord_m[0] == ord_o[0]
                       and ord_m[1] == ord_o[1])
    a60_note = ''
a60_ok = bool(a60_top2_ok
              and a60_maxdiff <= 0.15)
log('a60 top2=%s maxdiff=%.4f (gate 0.15) %s'
    % (a60_top2_ok, a60_maxdiff, a60_note))
# registered observation (not an anchor): re-entrant
# step-2 readout (3036 protocol) vs direct prefill
# readout differ - quantified per GEN prompt
z36 = np.load(os.path.join(
    BASE, 'phase3036',
    'omega_p33_fingerprint_curvature_map_qwen',
    'omega_p33_fingerprint_curvature_map_qwen.npz'),
    allow_pickle=True)
pidx36 = [int(x) for x in z36['prompt_idx'][:11]]
tokA36 = [int(x) for x in z36['tok_top'][:11, 0]]
pA36 = z36['p0_top'][:11, 0]
a60_re_dp = np.zeros(11)
a60_re_tok = np.zeros(11, dtype=bool)
for row in range(11):
    pi = pidx36[row]
    a60_re_tok[row] = int(np.argmax(Ps[pi])) \
        == tokA36[row]
    a60_re_dp[row] = abs(
        float(Ps[pi][tokA36[row]])
        - float(pA36[row]))
log('reentrant-vs-direct: tok_match=%d/11 '
    'max_dp=%.3f (registered observation)'
    % (int(a60_re_tok.sum()),
       float(a60_re_dp.max())))

# a61: source seals
a61_detail = []
for ph, nm in ((3035,
                'omega_p32_fingerprint_competition_qwen'),
               (3036,
                'omega_p33_fingerprint_curvature_map_'
                'qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a61_detail.append(bool(s == sealj['result_sha256_8']))
a61_ok = bool(a61_detail) and all(a61_detail)

# ---------- occurrence vectors ----------
K3 = np.zeros((n_occ, HDIM))
V3 = np.zeros((n_occ, HDIM))
K20 = np.zeros((n_occ, HDIM))
V20 = np.zeros((n_occ, HDIM))
for oi, x in enumerate(occ):
    pi, pos = x['prompt'], x['pos']
    K3[oi] = KVs[3][pi][0][pos]
    V3[oi] = KVs[3][pi][1][pos]
    K20[oi] = KVs[20][pi][0][pos]
    V20[oi] = KVs[20][pi][1][pos]


def cosp(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(abs(np.dot(a, b))
                 / max(na * nb, 1e-30))


words = [x['word'] for x in occ]
prs = [x['prompt'] for x in occ]
n_w = len(TARGETS)
w_idx = {w: i for i, w in enumerate(TARGETS)}

# cross-prompt pair cosines (same-prompt pairs
# reserved for T2)
cross_pairs = []
for i in range(n_occ):
    for j in range(i + 1, n_occ):
        if prs[i] != prs[j]:
            cross_pairs.append((i, j))
n_cp = len(cross_pairs)
cosV3 = np.zeros(n_cp)
cosK3 = np.zeros(n_cp)
cosV20 = np.zeros(n_cp)
same_flag = np.zeros(n_cp, dtype=int)
wpair = np.full(n_cp, -1, dtype=int)
for ci, (i, j) in enumerate(cross_pairs):
    cosV3[ci] = cosp(V3[i], V3[j])
    cosK3[ci] = cosp(K3[i], K3[j])
    cosV20[ci] = cosp(V20[i], V20[j])
    if words[i] == words[j]:
        same_flag[ci] = 1
        wpair[ci] = w_idx[words[i]]

diff_mask = same_flag == 0
same_mask = same_flag == 1
med_diff = float(np.median(cosV3[diff_mask]))
med_diff_K = float(np.median(cosK3[diff_mask]))
med_diff_20 = float(np.median(cosV20[diff_mask]))

# per-word stats + exact permutation maxT
rng = np.random.default_rng(PERM_SEED)
occ_idx_by_word = [[i for i in range(n_occ)
                    if words[i] == w]
                   for w in TARGETS]
pair_idx_by_word = [[ci for ci in range(n_cp)
                     if same_flag[ci] == 1
                     and wpair[ci] == wi]
                    for wi in range(n_w)]
obs_med = np.zeros(n_w)
for wi in range(n_w):
    idxs = pair_idx_by_word[wi]
    if idxs:
        obs_med[wi] = float(np.median(
            cosV3[idxs]))
# permutation: relabel word among cross-prompt
# occurrences only
cross_occ = [i for i in range(n_occ)
             if any(prs[i] != prs[j]
                    for j in range(n_occ))]
lab0 = np.array([w_idx[words[i]]
                 for i in cross_occ])
pre_pairs = []
for a in range(len(cross_occ)):
    for b in range(a + 1, len(cross_occ)):
        pre_pairs.append((a, b))
pre_pairs = np.array(pre_pairs)
lab0_pair = lab0[pre_pairs[:, 0]] == \
    lab0[pre_pairs[:, 1]]
# map cross_occ index -> occ index for cos lookup
co2oi = {k: i for k, i in enumerate(cross_occ)}
cp_index = {}
for ci, (i, j) in enumerate(cross_pairs):
    cp_index[(i, j)] = ci
pre_ci = np.array([cp_index[(co2oi[a], co2oi[b])]
                   for a, b in pre_pairs])
same_cos_pool = cosV3[pre_ci]

perm_max = np.zeros(N_PERM)
obs_max = float(np.max(obs_med))
for it in range(N_PERM):
    lab = rng.permutation(lab0)
    lp = lab[pre_pairs[:, 0]] == lab[pre_pairs[:, 1]]
    mx = -1.0
    for wi in range(n_w):
        sel = lp & (lab[pre_pairs[:, 0]] == wi)
        if not sel.any():
            continue
        m = float(np.median(same_cos_pool[sel]))
        if m > mx:
            mx = m
    perm_max[it] = mx
p_maxT = float(np.mean(perm_max >= obs_max))

per_word = []
for wi, w in enumerate(TARGETS):
    idxs = pair_idx_by_word[wi]
    n_pairs = len(idxs)
    med_w = float(obs_med[wi])
    ratio_w = med_w / max(med_diff, 1e-30)
    # per-word uncorrected permutation p
    cnt = 0
    nw_occ = len(occ_idx_by_word[wi])
    if n_pairs:
        for it in range(N_PERM):
            lab = rng.permutation(lab0)
            lp = lab[pre_pairs[:, 0]] == \
                lab[pre_pairs[:, 1]]
            sel = lp & (lab[pre_pairs[:, 0]] == wi)
            if sel.any() and float(np.median(
                    same_cos_pool[sel])) >= med_w:
                cnt += 1
        p_w = float(cnt + 1) / (N_PERM + 1)
    else:
        p_w = float('nan')
    per_word.append({
        'word': w, 'n_occ': nw_occ,
        'n_pairs': n_pairs, 'med_cosV3': med_w,
        'ratio': ratio_w, 'p_uncorrected': p_w})
    log('%s n_occ=%d n_pairs=%d med_cosV3=%.4f '
        'ratio=%.2f p=%.4f'
        % (w, nw_occ, n_pairs, med_w, ratio_w, p_w))

grand_ratio = (float(np.median(cosV3[same_mask]))
               / max(med_diff, 1e-30)
               if same_mask.any()
               else float('nan'))
n_pass = 0
# family pass judged via maxT p (phase-level) plus
# per-word ratios >= RATIO_STEREO among words with
# pairs
n_ratio_hi = sum(1 for pw in per_word
                 if pw['n_pairs'] > 0
                 and pw['ratio'] >= RATIO_STEREO)

# T2: within-prompt repeats (alphabetic tokens)
rep_rows = []
for pi in range(nP):
    ids = tok_ids[pi]
    seen = {}
    for pos, t in enumerate(ids):
        s = tok.decode([t]).strip().lower()
        if not s.isalpha() or len(s) < 2:
            continue
        if t in seen:
            for pos0 in seen[t]:
                rep_rows.append({
                    'prompt': pi, 'tok': s,
                    'pos0': pos0, 'pos1': pos,
                    'cosV3': cosp(
                        KVs[3][pi][1][pos0],
                        KVs[3][pi][1][pos]),
                    'cosK3': cosp(
                        KVs[3][pi][0][pos0],
                        KVs[3][pi][0][pos])})
            seen[t].append(pos)
        else:
            seen[t] = [pos]
med_repV = float(np.median([r['cosV3']
                            for r in rep_rows])) \
    if rep_rows else float('nan')
med_repK = float(np.median([r['cosK3']
                            for r in rep_rows])) \
    if rep_rows else float('nan')

# T3: L20 per-word ratios
per_word_20 = []
for wi, w in enumerate(TARGETS):
    idxs = pair_idx_by_word[wi]
    if idxs:
        m = float(np.median(cosV20[idxs]))
    else:
        m = float('nan')
    per_word_20.append(m)
med_same_20 = float(np.nanmedian(per_word_20)) \
    if per_word_20 else float('nan')
ratio_20 = med_same_20 / max(med_diff_20, 1e-30)

# ---------- verdict ----------
a_ok = bool(a58_diff == 0.0 and a59_diff == 0.0
            and a60_ok and a61_ok)
if p_maxT < 0.05 and n_ratio_hi >= 1 \
        and grand_ratio >= RATIO_STEREO:
    verdict = 'kv_stereotyped_qwen'
elif p_maxT >= 0.05 and grand_ratio <= RATIO_EPI:
    verdict = 'kv_episodic_qwen'
else:
    verdict = 'kv_mixed_qwen'

log('=== verdict ===')
log('a58=%r a59=%r a60_ok=%s a60_maxdiff=%.4f '
    'a61=%r'
    % (a58_diff, a59_diff, a60_ok, a60_maxdiff,
       a61_ok))
log('med_diff_cosV3=%.4f med_diff_K=%.4f '
    'med_diff_20=%.4f'
    % (med_diff, med_diff_K, med_diff_20))
log('grand_ratio=%.3f n_ratio_hi=%d p_maxT=%.5f'
    % (grand_ratio, n_ratio_hi, p_maxT))
log('T2 repeats: n=%d med_cosV3=%.4f med_cosK3=%.4f'
    % (len(rep_rows), med_repV, med_repK))
log('T3 L20: med_same=%.4f ratio=%.3f'
    % (med_same_20, ratio_20))
log('VERDICT=%s anchor_all_ok=%s'
    % (verdict, a_ok))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    prompts=np.array(ALL_PROMPTS),
    targets=np.array(TARGETS),
    occ_word=np.array(words),
    occ_prompt=np.array([x['prompt'] for x in occ]),
    occ_pos=np.array([x['pos'] for x in occ]),
    K3=K3, V3=V3, K20=K20, V20=V20,
    cross_i=np.array([i for i, j in cross_pairs]),
    cross_j=np.array([j for i, j in cross_pairs]),
    cosV3=cosV3, cosK3=cosK3, cosV20=cosV20,
    same_flag=same_flag, wpair=wpair,
    obs_med=obs_med,
    med_diff=np.float64(med_diff),
    grand_ratio=np.float64(grand_ratio),
    n_ratio_hi=np.int64(n_ratio_hi),
    p_maxT=np.float64(p_maxT),
    med_repV=np.float64(med_repV),
    med_repK=np.float64(med_repK),
    med_same_20=np.float64(med_same_20),
    ratio_20=np.float64(ratio_20),
    repV=np.array([r['cosV3'] for r in rep_rows]),
    repK=np.array([r['cosK3'] for r in rep_rows]),
    rep_dpos=np.array([r['pos1'] - r['pos0']
                       for r in rep_rows]),
    a58_diff=np.float64(a58_diff),
    a59_diff=np.float64(a59_diff),
    a60_maxdiff=np.float64(a60_maxdiff),
    a60_top2_ok=np.bool_(a60_top2_ok),
    a60_re_dp=a60_re_dp,
    a60_re_tok=a60_re_tok,
    a61_ok=np.bool_(a61_ok),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'anchor_all_ok': a_ok,
    'anchors': {
        'a58_dup_prefill_bit': a58_diff,
        'a59_dup_all_bit': a59_diff,
        'a60_top2_ok': a60_top2_ok,
        'a60_maxdiff': a60_maxdiff,
        'a60_gate': 0.15,
        'a60_note': a60_note,
        'a61_source_seals': a61_ok,
    },
    'a60_reentrant_observation': {
        'note': 'run1 preregistered a60 compared '
                'the 3036 two-step re-entrant '
                'softmax against the direct prefill '
                'readout - different quantities by '
                'design; registered as observation, '
                'replaced by manual-recompute anchor',
        'tok_match': int(a60_re_tok.sum()),
        'max_dp': float(a60_re_dp.max()),
        'dp_per_row': [float(v)
                       for v in a60_re_dp],
    },
    'T1_stereotypy': {
        'med_diff_cosV3': med_diff,
        'grand_ratio': grand_ratio,
        'n_ratio_hi': n_ratio_hi,
        'p_maxT': p_maxT,
        'per_word': per_word,
    },
    'T2_repeats': {
        'n_pairs': len(rep_rows),
        'med_cosV3': med_repV,
        'med_cosK3': med_repK,
    },
    'T3_layer20': {
        'med_same_cosV20': med_same_20,
        'med_diff_cosV20': med_diff_20,
        'ratio': ratio_20,
        'per_word': per_word_20,
    },
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
