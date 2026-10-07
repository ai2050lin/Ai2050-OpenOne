# -*- coding: utf-8 -*-
"""Phase 3118 (Omega-P116): autoregressive margin
trajectory (T4, first phase).

Question: is the margin m = h_final . (w_yes - w_no)
- the balanced controlled quantity established in
3114-3117 - maintained across free-generation steps,
or does it decay as the model's own tokens replace
the prompt?  And what is the TEMPORAL structure of
the top-causal-layer ablation effect (L26/L33/L31)
along generation steps?

Preregistered frame (3117 MEMO section 7), gates
frozen in this seal:
  1. TRACK gate (clean greedy trajectory): per
     generation step t = 0..12 read m(t) on the
     concatenated [prompt, g_1..g_12] sequence in
     ONE teacher-forced forward (causal attention ->
     prefix states identical to step-wise decode up
     to kernel-shape noise).  AUC(t) = Mann-Whitney
     AUC of P-group m(t) vs A1-group m(t) over 672
     pairs.  decay = AUC(0) - AUC(12):
       >= 0.10 -> belief_decays_in_generation
       <= 0.03 -> belief_tracked_in_generation
       else     -> belief_partially_tracked
  2. TEMPORAL gate (state effect on frozen clean
     sequences): replay the clean greedy sequence
     under each TOP3 ablation (L26/L33/L31) and
     read the trajectory difference
     dm_X(t) = m_abl(t) - m_clean(t) on the SAME
     tokens (pure state effect, no sequence
     confound).  ratio_X = mean|dm(9..12)| /
     mean|dm(0..2)|; mean over X:
       <= 0.7  -> temporal_compensation
       >= 1.3  -> ablation_amplifies_with_steps
       else    -> ablation_effect_static
  3. BEHAV gate (TOP3 ablation free-generation
     behavior): max over X of |yes_rate_X -
     yes_rate_clean| on first 8 greedy tokens:
       >= 0.05 -> top_ablation_changes_behavior
       <= 0.02 -> top_ablation_behavior_neutral
       else     -> top_ablation_behavior_mixed
  Sampled sub-study: 300 pairs x {P, A1} x
  {clean, abl_L26} x K=2 reps (temp 0.7, seed
  crc32(pk|dir|rep) shared across conditions, same
  rule as 3117); trajectories of both sequences
  replayed under both conditions (closed-loop vs
  state-effect separation at the sampled regime).
  greedy recheck: teacher-forced argmax at
  positions n_prompt-1+k must equal the generated
  token (reported per condition; KV-cache vs full-
  sequence kernel differences may make it < 1.0).

CONDITIONS (greedy trajectories, all 672 pairs):
  own sequences: clean, abl_L26, abl_L33, abl_L31
  replays: clean-seq under L26/L33/L31; each
  abl-seq under clean.
CONDITIONS (sampled, 300 pairs): clean + abl_L26,
  K=2, both sequences replayed under both
  conditions.
"""
import io
import json
import os
import time
import zlib
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p116_autoregressive_margin_trajectory'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D17 = os.path.join(RDIR, 'phase3117',
                   'omega_p115_pair_cancellation_'
                   'sampling')
D16 = os.path.join(RDIR, 'phase3116',
                   'omega_p114_full_mlp_sweep_decouple')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3118', NAME)
if SMOKE:
    D13 = os.path.join(D13, 'smoke')
    D05 = os.path.join(D05, 'smoke')
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ================================================================
# frozen inputs
# ================================================================
res13 = json.load(io.open(
    os.path.join(D13, 'result.json'), encoding='utf-8'))
assert res13['verdict'] == \
    'belief_robust|within_unit_replicated|' \
    'write_in_concentrated'
res17 = json.load(io.open(
    os.path.join(D17, 'result.json'), encoding='utf-8'))
assert res17['verdict'] == \
    'cancellation_pairwise_buffered|' \
    'sampled_consequence_confirmed'
res16 = json.load(io.open(
    os.path.join(D16, 'result.json'), encoding='utf-8'))
assert res16['verdict'] == \
    'interaction_dominant|no_behavioral_decoupling'
TOP3 = [int(t['layer'])
        for t in res16['top3_causal']]
assert TOP3 == [26, 33, 31]
N_NEW = 12
N_MATCH = 8
NSAMP_PAIRS = 300
K_REPS = 2
TEMP = 0.7
if SMOKE:
    NSAMP_PAIRS = 4
    K_REPS = 2
    N_NEW = 6
    N_MATCH = 4
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13, 'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
truthB = capb['truth']
m_base = capb['m'].astype(np.float32)
NB = len(pkB)

seal = {
    'phase': 3118,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'top3': TOP3,
    'n_new': N_NEW,
    'readout': 'm(t) = norm(h_NL[pos0+t]) . '
               '(w_yes - w_no), float32, ONE '
               'teacher-forced forward per '
               '(sequence, condition); pos0 = '
               'len(prompt)-1; t = 0..n_new',
    'greedy_conditions': ['clean', 'abl_L26',
                          'abl_L33', 'abl_L31'],
    'replay_matrix': ['clean-seq x {clean, L26, '
                      'L33, L31}; L26/L33/L31-seq '
                      'x {own, clean}'],
    'sampled': {'temp': TEMP, 'K': K_REPS,
                'n_pairs': NSAMP_PAIRS,
                'conds': ['clean', 'abl_L26'],
                'seed_rule': 'crc32(pk|dir|rep) & '
                             '0x7fffffff - identical '
                             'seed for clean and abl, '
                             'no cond term (same as '
                             '3117)'},
    'gates': {
        'track': 'decay = AUC_m(0) - AUC_m(12) over '
                 'P/A1 groups on clean greedy '
                 'trajectories: >=0.10 -> '
                 'belief_decays_in_generation; '
                 '<=0.03 -> '
                 'belief_tracked_in_generation; '
                 'else belief_partially_tracked',
        'temporal': 'ratio = mean|dm_state(9..12)| / '
                    'mean|dm_state(0..2)|, dm on '
                    'frozen clean sequences, mean '
                    'over TOP3: <=0.7 -> '
                    'temporal_compensation; >=1.3 '
                    '-> ablation_amplifies_with_'
                    'steps; else '
                    'ablation_effect_static',
        'behav': 'max over TOP3 of |yes_rate_cond - '
                 'yes_rate_clean| (first 8 greedy '
                 'tokens): >=0.05 -> '
                 'top_ablation_changes_behavior; '
                 '<=0.02 -> '
                 'top_ablation_behavior_neutral; '
                 'else top_ablation_behavior_mixed'},
    'greedy_recheck': 'teacher-forced argmax at '
                      'pos0+k equals generated token '
                      'g_k+1; reported per condition '
                      '(KV-cache vs full-sequence '
                      'kernel noise allowed)',
    'yes_family': ['yes', 'Yes', ' yes', ' Yes',
                   'YES'],
    'note': 'T4 first phase: the margin is read at '
            'every generation step; prefix states '
            'under causal attention are identical '
            'to step-wise decode states (up to '
            'kernel-shape noise), so one forward '
            'yields the full 13-point trajectory',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3118 Omega-P116 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed; TOP3=%s n_new=%d sampled '
    'pairs=%d K=%d temp=%.2f'
    % (TOP3, N_NEW, NSAMP_PAIRS, K_REPS, TEMP))

# --- rebuilt records (identical to 3113 B / 3114-3117) ---
import random as _rnd  # noqa: E402


def build_prompt_rb(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o)).encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this query '
             'true? Answer:' % (ents[s], PREDS[qrel],
                                ents[o]))
    return text


p2r = mat5['pair2rel']
frel = mat5['false_rels']
texts = []
for i in range(NB):
    pk = str(pkB[i])
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    cc = str(condB[i])
    if cc == 'P':
        (qrel, lrel) = (r, r)
    elif cc == 'A1':
        (qrel, lrel) = (r, ri1)
    else:
        (qrel, lrel) = (ri2, ri1)
    texts.append(build_prompt_rb(mat5, s, o, lrel,
                                 qrel))
hP = {}
hA1 = {}
for i in range(NB):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
    elif str(condB[i]) == 'A1':
        hA1[pk] = i
pks = sorted(hP.keys())
ip = np.array([hP[p] for p in pks])
ia = np.array([hA1[p] for p in pks])
NP_ = len(pks)
log('records rebuilt: %d (%d pairs)'
    % (NB, NP_))

# ================================================================
# model + hooks
# ================================================================
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.is_available()
tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NL = len(model.model.layers)
HIDb = int(model.config.hidden_size)
NH = int(model.config.num_attention_heads)
HD = int(getattr(model.config, 'head_dim', 0) or 0) \
    or HIDb // NH
WU = model.lm_head.weight.detach()
assert not (model.lm_head.bias is not None), \
    'lm_head bias unexpected'
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])
w_dn = (WU[YES_ID] - WU[NO_ID]).float().cpu().numpy()
YES_FAMILY = []
for ys in seal['yes_family']:
    YES_FAMILY += list(tok(ys,
                           add_special_tokens=False)
                       ['input_ids'])
YES_FAMILY = sorted(set(YES_FAMILY))
log('model loaded NL=%d HID=%d heads=%d head_dim=%d; '
    'yes_family ids=%s'
    % (NL, HIDb, NH, HD, YES_FAMILY))

ABL = {'mode': None, 'layers': set()}


def mk_mlp_post(L):
    def hook(mod, mod_in, out):
        if ABL['mode'] == 'mlp' \
                and L in ABL['layers']:
            return torch.zeros_like(out)
        return None
    return hook


ABL_LAYERS = set(TOP3)
for L in sorted(ABL_LAYERS):
    model.model.layers[L].mlp \
        .register_forward_hook(mk_mlp_post(L))


def gen_greedy(prompt_ids, n_new, layers_abl):
    """Greedy generation with KV cache; returns
    list of token ids."""
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl)
    cur = torch.tensor([prompt_ids], device='cuda')
    out_tokens = []
    with torch.inference_mode():
        out = model(cur, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1, :].argmax(-1)
        for _ in range(n_new):
            out_tokens.append(int(nxt))
            if len(out_tokens) >= n_new:
                break
            out = model(nxt.view(1, 1),
                        past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1, :].argmax(-1)
    ABL['mode'] = None
    ABL['layers'] = set()
    return out_tokens


def gen_sampled(prompt_ids, n_new, layers_abl, temp,
                seed):
    """Temperature sampling with KV cache; frozen
    seed per (pk, direction, rep) shared across
    conditions."""
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl)
    cur = torch.tensor([prompt_ids], device='cuda')
    g = torch.Generator(device='cuda')
    g.manual_seed(int(seed))
    out_tokens = []
    with torch.inference_mode():
        out = model(cur, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        nxt = torch.multinomial(
            torch.softmax(logits / temp, -1), 1,
            generator=g)
        for _ in range(n_new):
            out_tokens.append(int(nxt.item()))
            if len(out_tokens) >= n_new:
                break
            out = model(nxt.view(1, 1),
                        past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()
            nxt = torch.multinomial(
                torch.softmax(logits / temp, -1), 1,
                generator=g)
    ABL['mode'] = None
    ABL['layers'] = set()
    return out_tokens


def forward_track(prompt_ids, gen_tokens,
                  layers_abl):
    """One teacher-forced forward over
    [prompt, gen]; returns m(t) for t=0..n_new
    (13 points) and greedy-recheck rate."""
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl)
    gen_tokens = [int(x) for x in gen_tokens]
    ids = list(prompt_ids) + gen_tokens
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(gen_tokens) + 1
    ms = np.zeros(npts, dtype=np.float64)
    rechk = 0
    with torch.inference_mode():
        out = model(t_in,
                    output_hidden_states=True,
                    use_cache=False)
        hs = out.hidden_states[NL][0]
        lg = out.logits[0]
        norm = model.model.norm
        h_all = norm(hs)
        for k in range(npts):
            pos = pos0 + k
            ms[k] = float(
                (h_all[pos].float().cpu().numpy()
                 @ w_dn).item())
            if k >= 1 \
                    and int(lg[pos - 1].argmax(-1)
                            .item()) \
                    == int(gen_tokens[k - 1]):
                rechk += 1
        del out
    ABL['mode'] = None
    ABL['layers'] = set()
    return ms, rechk / float(max(len(gen_tokens), 1))


def auc_mw(pos_vals, neg_vals):
    """Mann-Whitney AUC with average ranks for
    ties (no scipy)."""
    x = np.concatenate([pos_vals, neg_vals])
    n1 = len(pos_vals)
    n2 = len(neg_vals)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0)
                 / (n1 * n2))


# ================================================================
# PART A: greedy trajectories (all 672 pairs)
# ================================================================
PID_P = [tok(texts[hP[pk]],
             add_special_tokens=False)['input_ids']
         for pk in pks]
PID_A = [tok(texts[hA1[pk]],
             add_special_tokens=False)['input_ids']
         for pk in pks]
COND_G = [('clean', ()), ('abl_L26', (26,)),
          ('abl_L33', (33,)), ('abl_L31', (31,))]
GEN = {}
for (cname, layers) in COND_G:
    t0 = time.time()
    gP = np.zeros((NP_, N_NEW), dtype=np.int32)
    gA = np.zeros((NP_, N_NEW), dtype=np.int32)
    for j in range(NP_):
        gP[j] = gen_greedy(PID_P[j], N_NEW, layers)
        gA[j] = gen_greedy(PID_A[j], N_NEW, layers)
        if j % 200 == 0:
            log('gen %s %d/%d (%.1fs)'
                % (cname, j, NP_,
                   time.time() - t0))
    GEN[cname] = {'P': gP, 'A1': gA}
    log('%s generation done (%.1fs)'
        % (cname, time.time() - t0))

TR = {}
rechk_log = {}
traj_conds = [('cleanseq_clean', 'clean', ()),
              ('cleanseq_L26', 'clean', (26,)),
              ('cleanseq_L33', 'clean', (33,)),
              ('cleanseq_L31', 'clean', (31,)),
              ('L26seq_L26', 'abl_L26', (26,)),
              ('L26seq_clean', 'abl_L26', ()),
              ('L33seq_L33', 'abl_L33', (33,)),
              ('L33seq_clean', 'abl_L33', ()),
              ('L31seq_L31', 'abl_L31', (31,)),
              ('L31seq_clean', 'abl_L31', ())]
for (tname, gcond, layers) in traj_conds:
    t0 = time.time()
    arrP = np.zeros((NP_, N_NEW + 1),
                    dtype=np.float32)
    arrA = np.zeros((NP_, N_NEW + 1),
                    dtype=np.float32)
    rc = []
    for j in range(NP_):
        arrP[j], r1 = forward_track(
            PID_P[j], GEN[gcond]['P'][j], layers)
        arrA[j], r2 = forward_track(
            PID_A[j], GEN[gcond]['A1'][j], layers)
        rc.append(r1)
        rc.append(r2)
        if j % 200 == 0:
            log('traj %s %d/%d (%.1fs)'
                % (tname, j, NP_,
                   time.time() - t0))
    TR[tname] = {'P': arrP, 'A1': arrA}
    rechk_log[tname] = float(np.mean(rc))
    log('%s done (%.1fs) recheck=%.4f'
        % (tname, time.time() - t0,
           rechk_log[tname]))

# --- TRACK gate ---
aucs = []
for t in range(N_NEW + 1):
    pv = TR['cleanseq_clean']['P'][:, t] \
        .astype(np.float64)
    nv = TR['cleanseq_clean']['A1'][:, t] \
        .astype(np.float64)
    aucs.append(auc_mw(pv, nv))
decay = aucs[0] - aucs[-1]
if decay >= 0.10:
    track_v = 'belief_decays_in_generation'
elif decay <= 0.03:
    track_v = 'belief_tracked_in_generation'
else:
    track_v = 'belief_partially_tracked'
log('TRACK: AUC(0)=%.4f AUC(12)=%.4f decay=%.4f '
    '-> %s'
    % (aucs[0], aucs[-1], decay, track_v))
log('TRACK curve: %s'
    % ' '.join('%.3f' % a for a in aucs))

# --- TEMPORAL gate (state effect on frozen
#     clean sequences) ---
ratios = []
temporal_by_layer = {}
for X in ('L26', 'L33', 'L31'):
    dmP = (TR['cleanseq_%s' % X]['P']
           - TR['cleanseq_clean']['P'])
    dmA = (TR['cleanseq_%s' % X]['A1']
           - TR['cleanseq_clean']['A1'])
    dm = np.concatenate([dmP, dmA], axis=0) \
        .astype(np.float64)
    head = np.abs(dm[:, 0:3]).mean()
    tail = np.abs(dm[:, N_NEW - 3:N_NEW + 1]).mean()
    ratio = tail / max(head, 1e-9)
    ratios.append(ratio)
    temporal_by_layer[X] = {
        'head_mean_abs_dm': float(head),
        'tail_mean_abs_dm': float(tail),
        'ratio': float(ratio)}
    log('TEMPORAL %s: head=%.4f tail=%.4f '
        'ratio=%.3f' % (X, head, tail, ratio))
ratio_mean = float(np.mean(ratios))
if ratio_mean <= 0.7:
    temp_v = 'temporal_compensation'
elif ratio_mean >= 1.3:
    temp_v = 'ablation_amplifies_with_steps'
else:
    temp_v = 'ablation_effect_static'
log('TEMPORAL: mean ratio=%.3f over TOP3 -> %s'
    % (ratio_mean, temp_v))

# --- BEHAV gate (free-generation behavior) ---
yf = set(YES_FAMILY)
yr = {}
agr = {}
for (cname, _) in COND_G:
    yc = 0
    nseq = 2 * NP_
    ag = 0
    for j, pk in enumerate(pks):
        gp = list(GEN[cname]['P'][j])
        ga = list(GEN[cname]['A1'][j])
        if any(t in yf for t in gp[:N_MATCH]):
            yc += 1
        if any(t in yf for t in ga[:N_MATCH]):
            yc += 1
        if gp[:N_MATCH] == ga[:N_MATCH]:
            ag += 1
    yr[cname] = yc / float(nseq)
    agr[cname] = ag / float(NP_)
beh_diffs = {c: abs(yr[c] - yr['clean'])
             for (c, _) in COND_G if c != 'clean'}
beh_max = max(beh_diffs.values())
if beh_max >= 0.05:
    beh_v = 'top_ablation_changes_behavior'
elif beh_max <= 0.02:
    beh_v = 'top_ablation_behavior_neutral'
else:
    beh_v = 'top_ablation_behavior_mixed'
log('BEHAV: yes_rate clean=%.4f L26=%.4f L33=%.4f '
    'L31=%.4f | agree clean=%.4f L26=%.4f L33=%.4f '
    'L31=%.4f | max_diff=%.4f -> %s'
    % (yr['clean'], yr['abl_L26'], yr['abl_L33'],
       yr['abl_L31'], agr['clean'], agr['abl_L26'],
       agr['abl_L33'], agr['abl_L31'], beh_max,
       beh_v))

# --- closed-loop total effect ---
closed = {}
for X in ('L26', 'L33', 'L31'):
    dP = (TR['%sseq_%s' % (X, X)]['P']
          - TR['cleanseq_clean']['P'])
    dA = (TR['%sseq_%s' % (X, X)]['A1']
          - TR['cleanseq_clean']['A1'])
    dm = np.concatenate([dP, dA], axis=0) \
        .astype(np.float64)
    head = np.abs(dm[:, 0:3]).mean()
    tail = np.abs(dm[:, N_NEW - 3:N_NEW + 1]).mean()
    closed[X] = {
        'head_mean_abs': float(head),
        'tail_mean_abs': float(tail),
        'ratio': float(tail / max(head, 1e-9))}
    log('CLOSED %s: head=%.4f tail=%.4f ratio=%.3f'
        % (X, head, tail, closed[X]['ratio']))

# ================================================================
# PART B: sampled trajectories (300 pairs)
# ================================================================
samp_pks = pks[:NSAMP_PAIRS]
assert len(samp_pks) == NSAMP_PAIRS


def seed_for(pk, direction, rep):
    return zlib.crc32(('%s|%s|%d'
                       % (pk, direction, rep))
                      .encode('ascii')) & 0x7fffffff


SGEN = {}
t0 = time.time()
for j, pk in enumerate(samp_pks):
    for direction, pid in (('P', PID_P[j]),
                           ('A1', PID_A[j])):
        for rep in range(K_REPS):
            sd = seed_for(pk, direction, rep)
            sc = gen_sampled(pid, N_NEW, (), TEMP,
                             sd)
            sa = gen_sampled(pid, N_NEW, (26,),
                             TEMP, sd)
            SGEN[(pk, direction, rep)] = (sc, sa)
    if j % 50 == 0:
        log('samp gen %d/%d (%.1fs)'
            % (j, NSAMP_PAIRS,
               time.time() - t0))
log('sampled generation done (%.1fs)'
    % (time.time() - t0))

ST = {}
t0 = time.time()
for (tname, seqkey, layers) in (
        ('samp_cleanseq_clean', 'c', ()),
        ('samp_cleanseq_L26', 'c', (26,)),
        ('samp_L26seq_L26', 'a', (26,)),
        ('samp_L26seq_clean', 'a', ())):
    n_tot = NSAMP_PAIRS * 2 * K_REPS
    arr = np.zeros((n_tot, N_NEW + 1),
                   dtype=np.float32)
    idx = 0
    for j in range(NSAMP_PAIRS):
        for pid in (PID_P[j], PID_A[j]):
            for rep in range(K_REPS):
                pk = samp_pks[j]
                direction = ('P'
                             if pid is PID_P[j]
                             else 'A1')
                sc, sa = SGEN[(pk, direction,
                               rep)]
                seq = sc if seqkey == 'c' else sa
                ms, _ = forward_track(pid, seq,
                                      layers)
                arr[idx] = ms
                idx += 1
    ST[tname] = arr
    log('%s done (%.1fs)'
        % (tname, time.time() - t0))
# sampled state-effect ratio (abl_L26)
dms = (ST['samp_cleanseq_L26']
       - ST['samp_cleanseq_clean']) \
    .astype(np.float64)
head_s = np.abs(dms[:, 0:3]).mean()
tail_s = np.abs(dms[:, N_NEW - 3:N_NEW + 1]).mean()
samp_ratio = float(tail_s / max(head_s, 1e-9))
log('SAMPLED TEMPORAL: head=%.4f tail=%.4f '
    'ratio=%.3f' % (head_s, tail_s, samp_ratio))
# sampled yes rates + seq agreement (regime
# consistency with 3117)
ys_c = 0
ys_a = 0
sag = 0
n_seq = NSAMP_PAIRS * 2 * K_REPS
for (pk, direction, rep), (sc, sa) in \
        SGEN.items():
    if sc == sa:
        sag += 1
    if any(t in yf for t in sc[:N_MATCH]):
        ys_c += 1
    if any(t in yf for t in sa[:N_MATCH]):
        ys_a += 1
samp_res = {
    'seq_agree': sag / float(n_seq),
    'yes_rate_clean': ys_c / float(n_seq),
    'yes_rate_abl_L26': ys_a / float(n_seq),
    'state_ratio': samp_ratio,
    'n_seq': n_seq}
log('SAMPLED: seq_agree=%.4f yes clean=%.4f '
    'abl26=%.4f | %s'
    % (samp_res['seq_agree'],
       samp_res['yes_rate_clean'],
       samp_res['yes_rate_abl_L26'],
       json.dumps(samp_res)))
log('VERDICT: %s|%s|%s'
    % (track_v, temp_v, beh_v))

# ================================================================
# save
# ================================================================
npz_dict = {'m_base_check': m_base,
            'pk': pkB, 'cond': condB,
            'truth': truthB,
            'auc_curve': np.array(aucs)}
for tname in TR:
    npz_dict['gt_%s__P' % tname] = TR[tname]['P']
    npz_dict['gt_%s__A1' % tname] = TR[tname]['A1']
for (cname, _) in COND_G:
    npz_dict['gen_%s__P' % cname] = \
        GEN[cname]['P']
    npz_dict['gen_%s__A1' % cname] = \
        GEN[cname]['A1']
for tname in ST:
    npz_dict['st_%s' % tname] = ST[tname]
npz_dict['samp_pks'] = np.array(samp_pks)
np.savez(os.path.join(OUT, 'traj_readout.npz'),
         **npz_dict)
log('readout saved')

results = {
    'verdict': '%s|%s|%s'
    % (track_v, temp_v, beh_v),
    'top3': TOP3,
    'auc_curve': aucs,
    'track': {'auc0': aucs[0],
              'auc_last': aucs[-1],
              'decay': decay,
              'gate': track_v},
    'temporal_state': {'by_layer':
                       temporal_by_layer,
                       'ratio_mean': ratio_mean,
                       'gate': temp_v},
    'closed_loop': closed,
    'greedy_recheck': rechk_log,
    'behavior': {'yes_rate': yr,
                 'agree_P_A1': agr,
                 'max_diff': beh_max,
                 'gate': beh_v},
    'sampled': samp_res,
    'gates': seal['gates'],
    'smoke': SMOKE, 'n_records': NB,
    'n_pairs': NP_,
    'n_samp_pairs': NSAMP_PAIRS,
    'k_reps': K_REPS, 'temp': TEMP,
    'n_new': N_NEW,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1,
              ensure_ascii=False)
log('result.json written')
log('Phase 3118 done (%.1fs)' % (time.time() - T0))
print('PHASE3118_DONE verdict=%s|%s|%s'
      % (track_v, temp_v, beh_v))
