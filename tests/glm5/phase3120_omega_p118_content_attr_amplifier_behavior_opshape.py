# -*- coding: utf-8 -*-
"""Phase 3120 (Omega-P118): content-step attribution,
amplifier behavioral transmission, operator shape
(T4, third phase).

Preregistered in 3119 MEMO section 5; gates frozen
in this seal BEFORE any observation.

3119 established: (i) answer tokens appear ONLY at
t=1, so AUC oscillation is driven by CONTENT-token
steps, but the semantic identity of those content
tokens was NOT attributed; (ii) compensation is
layer-specific with exactly 2 amplifiers L30 (ratio
1.377, peak_t=9) / L32 (1.148, peak_t=5) = the
erase chain, whose BEHAVIORAL role is untested;
(iii) the one-step rewrite is state-dependent
nonlinear (R2 0.19, embedding injection zero,
interaction +10.3pp) - the QUANTITATIVE SHAPE of
the mean-reversion operator is unknown.

Part A (offline, reuses frozen 3118 npz - no GPU):
  content-token span attribution.
  For every record the 11 post-answer tokens
  (k=1..11, steps t=2..12) are reconstructed as a
  string (byte-level BPE: decode(seq) is the char
  concat of per-token decodes) and annotated with
  regex spans:
    fact spans  : 'The E PRED the E.' with E in the
                  28-entity vocab, PRED in the 8
                  predicate vocab; sub-tag 'queried'
                  if the triple == the queried
                  statement (s,qrel,o), 'context_'
                  'other' if it matches one of the 8
                  context lines, else 'novel'.
    query spans : 'Is this query true' / 'Query' /
                  'true?' / 'Answer'.
    token class : syntax (no alnum) > fact_strict /
                  fact_shaped (inside a fact span) >
                  query_rest > other.
  fact = fact_strict U fact_shaped; neutral = the
  rest.  Hypothesis: fact-restatement steps drive
  gap recovery (P margin up, A1 margin down),
  neutral steps drive collapse.
  Gates (units = (t in 2..12) x quintile of the
  conditioning margin, 55 units; valid unit needs
  n_fact>=15 AND n_neutral>=15; pooled_* computed
  on the samples of valid units):
   A-FACT-P: pooled_diff_P >= +0.10 AND
     unit_rate_P >= 0.6 (diff_u>0 per unit)
   A-FACT-A1: pooled_diff_A1 <= -0.10 AND
     unit_rate_A1 >= 0.6 (diff_u<0 per unit)
   A-GAP2: classes FF = P-fact AND A1-fact,
     NN = both neutral, on quintiles of gap(t-1);
     pooled_dgap_FF >= +0.10 AND contrast
     (FF-NN) >= +0.10 AND unit_rate(contrast_u>0)
     >= 0.55 -> gap_attribution_confirmed
   A verdict: factP AND factA1 AND gap ->
     fact_restatement_drives_recovery; factP AND
     factA1 -> fact_restatement_margin_only; gap ->
     gap_attribution_only; else
     attribution_unresolved.
  Descriptive readouts: per-class mean dm per
  direction; sub-tag table (queried / context_other
  / novel) - for A1 the queried statement is NOT a
  context line while for P it IS, so 'queried'
  restatement separates restating-the-assertion
  from restating-the-context.

Part B (GPU): amplifier behavioral transmission.
  Greedy generation under single-layer MLP ablation
  at L30 and L32 (the 3119 amplifiers), 672 pairs x
  {P,A1} each; clean metrics recomputed offline
  from the 3118 npz and asserted bit-level vs the
  3118 result.json.  yes_rate = any yes-family
  token in first N_MATCH=8 greedy tokens (3118
  definition); agree_P_A1 = P and A1 first-8 token
  sequences identical.
   B-BEH: beh_max = max |yes_cond - yes_clean|
     over {L30,L32}: >= 0.05 -> amplifier_
     behavioral; <= 0.02 -> amplifier_neutral;
     else amplifier_partial.
  Sampled sub-study (temp 0.7, K=2, seed rule
  crc32(pk|dir|rep) & 0x7fffffff, SAME as 3118):
  300 frozen 3118 samp pairs x {clean, abl_L30,
  abl_L32}.  Clean re-run is a pipeline integrity
  probe: margins must reproduce 3118
  st_samp_cleanseq_clean bit-exactly.
   B-REPRO: max|dst| == 0.0 -> sampled_pipeline_
     reproduced; else sampled_pipeline_diverged
     (non-fatal: comparisons then use the internal
     clean re-run as baseline).
   B-SAMP: max |yes_cond - yes_clean| over
     amplifiers >= 0.03 -> sampled_confirms_
     behavioral; < 0.03 -> sampled_not_confirm;
     greedy-neutral/partial -> not_applicable.
  Also: 2x2 tracking per amplifier (clean tokens
  tracked under ablation; own tokens tracked clean
  and under own ablation) -> state_ratio (3118
  definition) and closed-loop ratio; own-recheck
  on 64 stride-selected sequences per condition
  (greedy expected ~1.0, sampled descriptive).
  B verdict: '<beh>|<samp>'.

Part C (offline): quantitative shape of the
  mean-reversion rewrite operator.
  Samples: all (pair j, step t=1..12, direction)
  = 672*12*2 = 16128.  y = m(t) - m(t-1),
  x = m(t-1).  PRIMARY: per-direction centering
  (xc = x - mean(x|dir), yc = y - mean(y|dir);
  equivalent to direction-specific intercepts with
  a shared shape).  20 equal-frequency bins with
  PAIR-LEVEL bootstrap (1000 reps, seed 3120,
  percentile CI).  Fits (in-sample R2; shape
  question, not prediction):
    lin : y ~ [1, xc]
    quad: y ~ [1, xc, xc^2]
    pw  : y ~ [1, xc, relu(xc - b)], b over the 9
          interior deciles of xc, best b kept
  Gates:
   C-MONO: spearman(xc, yc) <= -0.5 ->
     mean_reversion_confirmed; in (-0.5,-0.2] ->
     mean_reversion_weak; > -0.2 ->
     mean_reversion_absent.
   C-QUAD: d_quad = R2_quad - R2_lin >= 0.05 ->
     nonlinear_curvature_present else absent.
   C-GATE: d_pw = R2_pw_best - R2_lin >= 0.05 ->
     gated_operator else linear_operator.
   C verdict: '<mono>|<gate>'.
  Secondary (logged, gates re-evaluated
  descriptively): raw pooled fits without
  centering; per-direction separate fits; t>=2
  sensitivity (13440 samples) - the A1 answer-step
  relaxation jump (+8.77) is inside the primary
  window by design.

Overall verdict = A | B | C.
"""
import io
import json
import os
import re
import time
import zlib
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p118_content_attr_amplifier_' \
       'behavior_opshape'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D18 = os.path.join(RDIR, 'phase3118',
                   'omega_p116_autoregressive_margin_'
                   'trajectory')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3120', NAME)
if SMOKE:
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
res18 = json.load(io.open(
    os.path.join(D18, 'result.json'), encoding='utf-8'))
assert res18['verdict'] == \
    'belief_decays_in_generation|' \
    'temporal_compensation|' \
    'top_ablation_changes_behavior'
if not SMOKE:
    D19 = os.path.join(RDIR, 'phase3119',
                       'omega_p117_oscillation_'
                       'attribution_writemap_'
                       'linearity')
    res19 = json.load(io.open(
        os.path.join(D19, 'result.json'),
        encoding='utf-8'))
    assert res19['verdict'] == \
        'attribution_selection_dominant|' \
        'compensation_layer_specific|' \
        'rewrite_nonlinear'
AMP_LAYERS = [30, 32]
TOP3 = [26, 33, 31]
N_NEW = 12
N_MATCH = 8
NP_B = 24 if SMOKE else 672
NSAMP_S = 4 if SMOKE else 300
K_REPS = 2
TEMP = 0.7
N_BOOT = 1000
N_BINS = 20
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13, 'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
NB = len(pkB)
z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
auc18 = z18['auc_curve']
assert abs(float(auc18[0])
           - 0.9809094210600907) < 1e-12

seal = {
    'phase': 3120,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'amp_layers': AMP_LAYERS,
    'n_new': N_NEW,
    'n_match': N_MATCH,
    'sampled': {'temp': TEMP, 'K': K_REPS,
                'n_pairs': NSAMP_S,
                'seed_rule': 'crc32(pk|dir|rep) & '
                             '0x7fffffff, identical '
                             'seed across conditions '
                             '(3118 rule)'},
    'data_sources': {
        'trajectories_and_tokens':
            'phase3118/traj_readout.npz bit-exact '
            'reuse',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz '
                    '(same rebuild as 3114-3119)'},
    'token_families': {
        'yes': ['yes', 'Yes', ' yes', ' Yes',
                'YES'],
        'no': ['no', 'No', ' no', ' No', 'NO']},
    'annotation': {
        'fact_span': 'The E PRED the E.? with E in '
                     '28 entities, PRED in 8 '
                     'predicates (regex alternation, '
                     'longest first)',
        'sub_tags': 'queried (== queried statement '
                    's,qrel,o) > context_other (in '
                    'the 8 context lines) > novel',
        'query_span': 'Is this query true / Query / '
                      'true? / Answer',
        'token_class_priority': 'syntax (no alnum) > '
                                'fact_strict/fact_'
                                'shaped > query_rest '
                                '> other',
        'fact_for_gates': 'fact_strict U '
                          'fact_shaped; neutral = '
                          'the rest',
        'note': 'byte-level BPE: decode(seq) = char '
                'concat of per-token decodes; token '
                '-> char offsets -> span membership'},
    'gates': {
        'A_fact_P': 'units=(t 2..12)xquintile(m_P(t-1)'
                    ', pooled edges); valid unit '
                    'n_fact>=15 and n_neutral>=15; '
                    'pass iff pooled_diff_P >= +0.10 '
                    'AND unit_rate_P >= 0.6 '
                    '(diff_u>0) -> fact_margin_P_up',
        'A_fact_A1': 'same on A1; pass iff '
                     'pooled_diff_A1 <= -0.10 AND '
                     'unit_rate_A1 >= 0.6 (diff_u<0)'
                     ' -> fact_margin_A1_down',
        'A_gap2': 'units=(t 2..12)xquintile(gap(t-1));'
                  ' FF=both-fact, NN=both-neutral; '
                  'valid n>=15 both; pass iff '
                  'pooled_dgap_FF >= +0.10 AND '
                  'contrast(FF-NN) >= +0.10 AND '
                  'unit_rate(contrast_u>0) >= 0.55 '
                  '-> gap_attribution_confirmed',
        'A_verdict': 'factP AND factA1 AND gap -> '
                     'fact_restatement_drives_'
                     'recovery; factP AND factA1 -> '
                     'fact_restatement_margin_only; '
                     'gap -> gap_attribution_only; '
                     'else attribution_unresolved',
        'B_beh': 'beh_max = max |yes_cond-yes_clean| '
                 'over {L30,L32}, yes_rate = any '
                 'yes-family token in first 8 greedy '
                 'tokens: >=0.05 -> '
                 'amplifier_behavioral; <=0.02 -> '
                 'amplifier_neutral; else '
                 'amplifier_partial',
        'B_repro': 'clean sampled margins bit-exact '
                   'vs 3118 st_samp_cleanseq_clean '
                   '(max==0.0) -> '
                   'sampled_pipeline_reproduced; '
                   'else diverged (non-fatal, '
                   'internal baseline)',
        'B_samp': 'max |yes_cond-yes_clean| sampled '
                  '>= 0.03 -> '
                  'sampled_confirms_behavioral; '
                  '< 0.03 -> sampled_not_confirm; '
                  'greedy neutral/partial -> '
                  'not_applicable',
        'C_mono': 'spearman(xc,yc) <= -0.5 -> '
                  'mean_reversion_confirmed; '
                  '(-0.5,-0.2] -> '
                  'mean_reversion_weak; > -0.2 -> '
                  'mean_reversion_absent',
        'C_quad': 'd_quad = R2_quad - R2_lin >= 0.05 '
                  '-> nonlinear_curvature_present '
                  'else absent',
        'C_gate': 'd_pw = R2_pw_best - R2_lin '
                  '>= 0.05 -> gated_operator else '
                  'linear_operator',
        'C_verdict': 'mono|gate'},
    'part_c_spec': {
        'n_samples': 16128,
        'primary': 'per-direction centering '
                   '(direction intercepts, shared '
                   'shape)',
        'secondary': ['raw pooled (no centering)',
                      'per-direction separate fits',
                      't>=2 sensitivity (13440)'],
        'bins': '20 equal-frequency on xc',
        'bootstrap': 'pair-level, 1000 reps, seed '
                     '3120, percentile 2.5/97.5',
        'r2_mode': 'in-sample (shape question, not '
                   'prediction; noted in seal)'},
    'note': '3119 left three holes: content-step '
            'semantic identity, amplifier behavioral '
            'role, operator shape.  This phase '
            'closes all three on the same frozen '
            '3118 material + 2 new GPU conditions.',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3120 Omega-P118 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed; AMP=%s NP_B=%d samp_pairs=%d '
    'K=%d temp=%.2f'
    % (AMP_LAYERS, NP_B, NSAMP_S, K_REPS, TEMP))

# ================================================================
# rebuilt records (identical to 3113 B / 3114-3119)
# ================================================================
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
NP_ = len(pks)
assert NP_ == 672
NP_B = min(NP_B, NP_)
log('records rebuilt: %d (%d pairs)' % (NB, NP_))

# ================================================================
# PART A: content-token span attribution (offline)
# ================================================================
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MDIR)
YES_FAMILY = []
for ys in seal['token_families']['yes']:
    YES_FAMILY += list(tok(
        ys, add_special_tokens=False)
        ['input_ids'])
YES_FAMILY = sorted(set(YES_FAMILY))
NO_FAMILY = []
for ns in seal['token_families']['no']:
    NO_FAMILY += list(tok(
        ns, add_special_tokens=False)
        ['input_ids'])
NO_FAMILY = sorted(set(NO_FAMILY))
YF = np.array(YES_FAMILY)
NF = np.array(NO_FAMILY)
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])

mP = z18['gt_cleanseq_clean__P']   # (672,13) f32
mA = z18['gt_cleanseq_clean__A1']
gP = z18['gen_clean__P']           # (672,12) i32
gA = z18['gen_clean__A1']
assert mP.shape[0] == NP_ and gP.shape[0] == NP_
N_NEW = int(mP.shape[1] - 1)
assert N_NEW == 12

# AUC curve self-check vs 3118
def auc_mw(pos_vals, neg_vals):
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


auc_rc = [auc_mw(mP[:, t].astype(np.float64),
                 mA[:, t].astype(np.float64))
          for t in range(N_NEW + 1)]
sc_curve = max(abs(a - float(b))
               / max(abs(float(b)), 1e-12)
               for (a, b) in zip(auc_rc, auc18))
assert sc_curve < 1e-6, \
    'AUC curve does not reproduce 3118'
log('consistency: AUC curve reproduced, max rel '
    'diff %.2e' % sc_curve)

# --- regex vocab ---
ENTS = [str(e) for e in mat5['entities']]
PREDS = [str(p) for p in mat5['predicates']]
ent_alt = '|'.join(re.escape(e) for e in
                   sorted(ENTS, key=len,
                          reverse=True))
pred_alt = '|'.join(re.escape(p) for p in
                    sorted(PREDS, key=len,
                           reverse=True))
FACT_RE = re.compile(
    r'[Tt]he (%s) (%s) the (%s)\.?'
    % (ent_alt, pred_alt, ent_alt))
QUERY_RE = re.compile(
    r'(?:[Ii]s this query true|[Qq]uery|true\?'
    r'|[Aa]nswer)')
CLS_OTHER = 0
CLS_SYNTAX = 1
CLS_QUERY = 2
CLS_FSHAPED = 3
CLS_FSTRICT = 4
CLS_ANSWER = 9
SUB_NONE = 0
SUB_QUERIED = 1
SUB_CTX = 2
SUB_NOVEL = 3
CLS_NAME = {0: 'other', 1: 'syntax',
            2: 'query_rest', 3: 'fact_shaped',
            4: 'fact_strict', 9: 'answer'}


def classify_record(gtoks, pk, dircode):
    """Annotate content tokens k=1..N_NEW-1 of one
    record.  Returns (cls[k], sub[k], dec[k])."""
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    if dircode == 'P':
        (qrel, lrel) = (r, r)
    else:
        (qrel, lrel) = (r, ri1)
    D = [tuple(d) for d in
         mat5['distractors']['%d_%d' % (s, o)]]
    ctx = set()
    for (a, rr, b) in ([(s, lrel, o)] + list(D)):
        ctx.add((ENTS[a], PREDS[rr], ENTS[b]))
    queried = (ENTS[s], PREDS[qrel], ENTS[o])
    gtoks = [int(x) for x in gtoks]
    dec = [tok.decode([t]) for t in gtoks]
    S = ''.join(dec[1:])
    starts = []
    pos = 0
    for k in range(1, len(gtoks)):
        starts.append(pos)
        pos += len(dec[k])
    spans_f = []
    for mt in FACT_RE.finditer(S):
        trip = (mt.group(1), mt.group(2),
                mt.group(3))
        spans_f.append((mt.start(), mt.end(), trip))
    spans_q = [(m.start(), m.end())
               for m in QUERY_RE.finditer(S)]
    n = len(gtoks)
    cls = np.full(n, CLS_OTHER, dtype=np.int8)
    sub = np.zeros(n, dtype=np.int8)
    cls[0] = CLS_ANSWER
    for k in range(1, n):
        txt = dec[k]
        if not any(ch.isalnum() for ch in txt):
            cls[k] = CLS_SYNTAX
            continue
        cs = starts[k - 1]
        ce = cs + len(txt)
        hit_f = None
        for (a0, a1, trip) in spans_f:
            if a0 < ce and cs < a1:
                hit_f = trip
                break
        if hit_f is not None:
            if hit_f in ctx:
                cls[k] = CLS_FSTRICT
                sub[k] = (SUB_QUERIED
                          if hit_f == queried
                          else SUB_CTX)
            else:
                cls[k] = CLS_FSHAPED
                sub[k] = (SUB_QUERIED
                          if hit_f == queried
                          else SUB_NOVEL)
            continue
        hit_q = False
        for (a0, a1) in spans_q:
            if a0 < ce and cs < a1:
                hit_q = True
                break
        if hit_q:
            cls[k] = CLS_QUERY
    return cls, sub, dec


annP_c = np.zeros((NP_, N_NEW), dtype=np.int8)
annP_s = np.zeros((NP_, N_NEW), dtype=np.int8)
annA_c = np.zeros((NP_, N_NEW), dtype=np.int8)
annA_s = np.zeros((NP_, N_NEW), dtype=np.int8)
ynyn = 0
sample_strings = []
for j in range(NP_):
    pk = pks[j]
    c1, s1, d1 = classify_record(gP[j], pk, 'P')
    c2, s2, d2 = classify_record(gA[j], pk, 'A1')
    annP_c[j] = c1
    annP_s[j] = s1
    annA_c[j] = c2
    annA_s[j] = s2
    for k in range(1, N_NEW):
        if int(gP[j, k]) in set(YES_FAMILY) \
                or int(gP[j, k]) in set(NO_FAMILY) \
                or int(gA[j, k]) in set(YES_FAMILY) \
                or int(gA[j, k]) in set(NO_FAMILY):
            ynyn += 1
    if j < 4 and not SMOKE:
        sample_strings.append(
            {'pk': pk, 'dir': 'P',
             'answer': d1[0],
             'content': ''.join(d1[1:])})
        sample_strings.append(
            {'pk': pk, 'dir': 'A1',
             'answer': d2[0],
             'content': ''.join(d2[1:])})
log('annotation done; yes/no-family tokens found '
    'at k>=1: %d (3119 predicted 0)' % ynyn)
if sample_strings:
    for ss in sample_strings:
        log('SAMPLE %s %s ans=%r content=%r'
            % (ss['pk'], ss['dir'], ss['answer'],
               ss['content'][:160]))

FACTOR = np.isin(annP_c, [3, 4])
FACTA = np.isin(annA_c, [3, 4])
log('fact-token rate: P %.4f A1 %.4f'
    % (float(FACTOR[:, 1:].mean()),
       float(FACTA[:, 1:].mean())))

# --- per-direction margin gate (55 units) ---
def units_margin_gate(mD, factmask, want_pos):
    """units=(t 2..12)xquintile(m(t-1)); returns
    dict with pooled_diff, unit_rate, valid
    counts, per-unit rows."""
    dm = (mD[:, 1:] - mD[:, :-1]) \
        .astype(np.float64)[:, 1:]
    nt = dm.shape[1]
    x = mD[:, :-1].astype(np.float64)[:, 1:]
    fv = factmask[:, 1:]
    edges = np.quantile(x.ravel(),
                        [0.2, 0.4, 0.6, 0.8])
    qb = np.searchsorted(edges, x.ravel(),
                         side='right')
    tarr = np.tile(np.arange(nt), x.shape[0])
    rows = []
    diffs = []
    dP = dm.ravel()
    for t in range(nt):
        for b in range(5):
            sel = (qb == b) & (tarr == t)
            sf = sel & fv.ravel()
            sn = sel & (~fv.ravel())
            nf = int(sf.sum())
            nn = int(sn.sum())
            if nf < 15 or nn < 15:
                rows.append({'t': t + 2, 'bin': b,
                             'n_fact': nf,
                             'n_neutral': nn,
                             'valid': False})
                continue
            mf = float(dP[sf].mean())
            mn = float(dP[sn].mean())
            diff = mf - mn
            diffs.append(diff)
            rows.append({'t': t + 2, 'bin': b,
                         'n_fact': nf,
                         'n_neutral': nn,
                         'dm_fact': mf,
                         'dm_neutral': mn,
                         'diff': diff,
                         'valid': True})
    nval = len(diffs)
    npos = sum(1 for d in diffs
               if (d > 0 if want_pos else d < 0))
    sf_all = np.zeros(dP.shape, dtype=bool)
    sn_all = np.zeros(dP.shape, dtype=bool)
    fv_r = fv.ravel()
    for row in rows:
        if not row['valid']:
            continue
        t = row['t'] - 2
        b = row['bin']
        sel = (qb == b) & (tarr == t)
        sf_all |= (sel & fv_r)
        sn_all |= (sel & (~fv_r))
    pooled = float(dP[sf_all].mean()
                   - dP[sn_all].mean()) \
        if sf_all.any() and sn_all.any() else None
    rate = (npos / float(nval)) if nval else None
    return {'pooled_diff': pooled,
            'unit_rate': rate,
            'n_valid_units': nval,
            'n_units': nt * 5,
            'units': rows}


gATE_P = units_margin_gate(mP, FACTOR, True)
gATE_A = units_margin_gate(mA, FACTA, False)
factP = (gATE_P['pooled_diff'] is not None
         and gATE_P['pooled_diff'] >= 0.10
         and gATE_P['unit_rate'] is not None
         and gATE_P['unit_rate'] >= 0.6)
factA1 = (gATE_A['pooled_diff'] is not None
          and gATE_A['pooled_diff'] <= -0.10
          and gATE_A['unit_rate'] is not None
          and gATE_A['unit_rate'] >= 0.6)
log('A-FACT-P: pooled_diff=%s unit_rate=%s '
    'valid=%d/%d -> %s'
    % (('%.4f' % gATE_P['pooled_diff'])
       if gATE_P['pooled_diff'] is not None
       else 'nan',
       ('%.3f' % gATE_P['unit_rate'])
       if gATE_P['unit_rate'] is not None
       else 'nan',
       gATE_P['n_valid_units'],
       gATE_P['n_units'],
       'fact_margin_P_up' if factP else 'fail'))
log('A-FACT-A1: pooled_diff=%s unit_rate=%s '
    'valid=%d/%d -> %s'
    % (('%.4f' % gATE_A['pooled_diff'])
       if gATE_A['pooled_diff'] is not None
       else 'nan',
       ('%.3f' % gATE_A['unit_rate'])
       if gATE_A['unit_rate'] is not None
       else 'nan',
       gATE_A['n_valid_units'],
       gATE_A['n_units'],
       'fact_margin_A1_down' if factA1
       else 'fail'))

# --- gap gate (FF vs NN, 55 units) ---
gap = (mP - mA)
dgap = (gap[:, 1:] - gap[:, :-1]) \
    .astype(np.float64)[:, 1:]
xg = gap[:, :-1].astype(np.float64)[:, 1:]
edges_g = np.quantile(xg.ravel(),
                      [0.2, 0.4, 0.6, 0.8])
qbg = np.searchsorted(edges_g, xg.ravel(),
                      side='right')
tarrg = np.tile(np.arange(xg.shape[1]),
                xg.shape[0])
nt = xg.shape[1]
ffv = (FACTOR & FACTA)[:, 1:].ravel()
nnv = ((~FACTOR) & (~FACTA))[:, 1:].ravel()
dgv = dgap.ravel()
rows_g = []
contrasts = []
for t in range(nt):
    for b in range(5):
        sel = (qbg == b) & (tarrg == t)
        sff = sel & ffv
        snn = sel & nnv
        nff = int(sff.sum())
        nnn = int(snn.sum())
        if nff < 15 or nnn < 15:
            rows_g.append({'t': t + 2, 'bin': b,
                           'n_ff': nff, 'n_nn': nnn,
                           'valid': False})
            continue
        mff = float(dgv[sff].mean())
        mnn = float(dgv[snn].mean())
        contrasts.append(mff - mnn)
        rows_g.append({'t': t + 2, 'bin': b,
                       'n_ff': nff, 'n_nn': nnn,
                       'dgap_ff': mff,
                       'dgap_nn': mnn,
                       'contrast': mff - mnn,
                       'valid': True})
nval_g = len(contrasts)
npos_g = sum(1 for d in contrasts if d > 0)
sfF = np.zeros(dgv.shape, dtype=bool)
snN = np.zeros(dgv.shape, dtype=bool)
for row in rows_g:
    if not row['valid']:
        continue
    t = row['t'] - 2
    b = row['bin']
    sel = (qbg == b) & (tarrg == t)
    sfF |= (sel & ffv)
    snN |= (sel & nnv)
pdg_ff = float(dgv[sfF].mean()) if sfF.any() \
    else None
pdg_nn = float(dgv[snN].mean()) if snN.any() \
    else None
contrast_pool = (pdg_ff - pdg_nn) \
    if (pdg_ff is not None and pdg_nn is not None) \
    else None
rate_g = (npos_g / float(nval_g)) if nval_g \
    else None
gap_ok = (pdg_ff is not None
          and contrast_pool is not None
          and pdg_ff >= 0.10
          and contrast_pool >= 0.10
          and rate_g is not None and rate_g >= 0.55)
log('A-GAP2: dgap_FF=%s dgap_NN=%s contrast=%s '
    'rate=%s valid=%d -> %s'
    % (('%.4f' % pdg_ff) if pdg_ff is not None
       else 'nan',
       ('%.4f' % pdg_nn) if pdg_nn is not None
       else 'nan',
       ('%.4f' % contrast_pool)
       if contrast_pool is not None else 'nan',
       ('%.3f' % rate_g) if rate_g is not None
       else 'nan', nval_g,
       'gap_attribution_confirmed' if gap_ok
       else 'rejected'))

if factP and factA1 and gap_ok:
    verdict_a = 'fact_restatement_drives_recovery'
elif factP and factA1:
    verdict_a = 'fact_restatement_margin_only'
elif gap_ok:
    verdict_a = 'gap_attribution_only'
else:
    verdict_a = 'attribution_unresolved'
log('PART A VERDICT: %s' % verdict_a)

# --- per-class + sub-tag descriptives ---
def class_table(mD, ann_c):
    dm = (mD[:, 1:] - mD[:, :-1]) \
        .astype(np.float64)[:, 1:]
    out = {}
    for code in (0, 1, 2, 3, 4):
        sel = (ann_c[:, 1:] == code)
        n = int(sel.sum())
        out[CLS_NAME[code]] = {
            'n': n,
            'mean_dm': float(dm[sel].mean())
            if n else None}
    return out


def subtag_table(mD, ann_c, ann_s):
    dm = (mD[:, 1:] - mD[:, :-1]) \
        .astype(np.float64)[:, 1:]
    out = {}
    for (scode, sname) in ((1, 'queried'),
                           (2, 'context_other'),
                           (3, 'novel')):
        sel = (ann_s[:, 1:] == scode)
        n = int(sel.sum())
        out[sname] = {'n': n,
                      'mean_dm': float(dm[sel]
                                       .mean())
                      if n else None}
    return out


cls_tab_P = class_table(mP, annP_c)
cls_tab_A = class_table(mA, annA_c)
sub_tab_P = subtag_table(mP, annP_c, annP_s)
sub_tab_A = subtag_table(mA, annA_c, annA_s)
log('class table P: %s'
    % json.dumps(cls_tab_P))
log('class table A1: %s'
    % json.dumps(cls_tab_A))
log('subtag table P: %s' % json.dumps(sub_tab_P))
log('subtag table A1: %s' % json.dumps(sub_tab_A))
per_step_fact = {
    'P': [float(FACTOR[:, k + 1].mean())
          for k in range(N_NEW - 1)],
    'A1': [float(FACTA[:, k + 1].mean())
           for k in range(N_NEW - 1)]}

part_a = {
    'verdict': verdict_a,
    'fact_margin_P_up': factP,
    'fact_margin_A1_down': factA1,
    'gap_attribution_confirmed': gap_ok,
    'gate_P': {**{k: gATE_P[k] for k in
                  ('pooled_diff', 'unit_rate',
                   'n_valid_units',
                   'n_units')},
               'units': gATE_P['units']},
    'gate_A1': {**{k: gATE_A[k] for k in
                   ('pooled_diff', 'unit_rate',
                    'n_valid_units',
                    'n_units')},
                'units': gATE_A['units']},
    'gate_gap': {'dgap_ff': pdg_ff,
                 'dgap_nn': pdg_nn,
                 'contrast': contrast_pool,
                 'unit_rate': rate_g,
                 'n_valid_units': nval_g,
                 'units': rows_g},
    'class_table': {'P': cls_tab_P,
                    'A1': cls_tab_A},
    'subtag_table': {'P': sub_tab_P,
                     'A1': sub_tab_A},
    'fact_token_rate': {
        'P': float(FACTOR[:, 1:].mean()),
        'A1': float(FACTA[:, 1:].mean())},
    'per_step_fact_rate': per_step_fact,
    'yes_no_at_content_steps': ynyn,
    'auc_curve_recomputed': auc_rc,
    'selfcheck_curve_rel_max': sc_curve,
    'sample_strings': sample_strings,
}

# ================================================================
# model + hooks (Part B needs GPU)
# ================================================================
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.is_available()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NL = len(model.model.layers)
log('model loaded NL=%d' % NL)
WU = model.lm_head.weight.detach()
assert model.lm_head.bias is None
w_dn = (WU[YES_ID] - WU[NO_ID]) \
    .float().cpu().numpy()

ABL = {'mode': None, 'layers': set()}


def mk_mlp_post(L):
    def hook(mod, mod_in, out):
        if ABL['mode'] == 'mlp' \
                and L in ABL['layers']:
            return torch.zeros_like(out)
        return None
    return hook


for L in AMP_LAYERS:
    model.model.layers[L].mlp \
        .register_forward_hook(mk_mlp_post(L))


def gen_greedy(prompt_ids, n_new, layers_abl):
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


def gen_sampled(prompt_ids, n_new, layers_abl,
                temp, seed):
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
                torch.softmax(logits / temp, -1),
                1, generator=g)
    ABL['mode'] = None
    ABL['layers'] = set()
    return out_tokens


def forward_track(prompt_ids, gen_tokens,
                  layers_abl):
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
    return ms, rechk / float(
        max(len(gen_tokens), 1))


PID_P = [tok(texts[hP[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]
PID_A = [tok(texts[hA1[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]

# ================================================================
# PART B: amplifier behavioral transmission (GPU)
# ================================================================
yf = set(YES_FAMILY)

# --- clean baseline recomputed from 3118 npz ---
yr_clean = 0
ag_clean = 0
for j in range(NP_):
    gp = list(gP[j])
    ga = list(gA[j])
    if any(t in yf for t in gp[:N_MATCH]):
        yr_clean += 1
    if any(t in yf for t in ga[:N_MATCH]):
        yr_clean += 1
    if gp[:N_MATCH] == ga[:N_MATCH]:
        ag_clean += 1
yr_clean = yr_clean / float(2 * NP_)
ag_clean = ag_clean / float(NP_)
assert abs(yr_clean
           - res18['behavior']['yes_rate']
           ['clean']) < 1e-12
assert abs(ag_clean
           - res18['behavior']['agree_P_A1']
           ['clean']) < 1e-12
log('clean behavior recomputed bit-level: '
    'yes=%.10f agree=%.10f' % (yr_clean, ag_clean))

GENX = {}
for (cname, layers) in (('abl_L30', (30,)),
                        ('abl_L32', (32,))):
    t0b = time.time()
    gxP = np.zeros((NP_B, N_NEW), dtype=np.int32)
    gxA = np.zeros((NP_B, N_NEW), dtype=np.int32)
    for j in range(NP_B):
        gxP[j] = gen_greedy(PID_P[j], N_NEW, layers)
        gxA[j] = gen_greedy(PID_A[j], N_NEW, layers)
        if j % 100 == 0:
            log('gen %s %d/%d (%.1fs)'
                % (cname, j, NP_B,
                   time.time() - t0b))
    GENX[cname] = {'P': gxP, 'A1': gxA}
    log('%s greedy done (%.1fs)'
        % (cname, time.time() - t0b))

yr = {'clean': yr_clean}
agr = {'clean': ag_clean}
for cname in ('abl_L30', 'abl_L32'):
    yc = 0
    ag = 0
    for j in range(NP_B):
        gp = list(GENX[cname]['P'][j])
        ga = list(GENX[cname]['A1'][j])
        if any(t in yf for t in gp[:N_MATCH]):
            yc += 1
        if any(t in yf for t in ga[:N_MATCH]):
            yc += 1
        if gp[:N_MATCH] == ga[:N_MATCH]:
            ag += 1
    yr[cname] = yc / float(2 * NP_B)
    agr[cname] = ag / float(NP_B)
beh_diffs = {c: abs(yr[c] - yr['clean'])
             for c in ('abl_L30', 'abl_L32')}
beh_max = max(beh_diffs.values())
if beh_max >= 0.05:
    beh_v = 'amplifier_behavioral'
elif beh_max <= 0.02:
    beh_v = 'amplifier_neutral'
else:
    beh_v = 'amplifier_partial'
log('B-BEH: yes clean=%.4f L30=%.4f L32=%.4f | '
    'agree clean=%.4f L30=%.4f L32=%.4f | '
    'max_diff=%.4f -> %s'
    % (yr['clean'], yr['abl_L30'],
       yr['abl_L32'], agr['clean'],
       agr['abl_L30'], agr['abl_L32'], beh_max,
       beh_v))

# --- greedy own-recheck (64 stride pairs) ---
rechk_greedy = {}
stride = max(1, NP_B // 64)
idxs = list(range(0, NP_B, stride))[:64]
for (cname, layers) in (('abl_L30', (30,)),
                        ('abl_L32', (32,))):
    rc = []
    for j in idxs:
        _, r1 = forward_track(PID_P[j],
                              GENX[cname]['P'][j],
                              layers)
        _, r2 = forward_track(PID_A[j],
                              GENX[cname]['A1'][j],
                              layers)
        rc.append(r1)
        rc.append(r2)
    rechk_greedy[cname] = float(np.mean(rc))
    log('own-recheck greedy %s: %.4f (%d seqs)'
        % (cname, rechk_greedy[cname], len(rc)))

# --- sampled sub-study ---
samp_pks_all = [str(x) for x in z18['samp_pks']]
samp_pks = samp_pks_all[:NSAMP_S]
assert samp_pks == pks[:NSAMP_S]


def seed_for(pk, direction, rep):
    return zlib.crc32(('%s|%s|%d'
                       % (pk, direction, rep))
                      .encode('ascii')) \
        & 0x7fffffff


SGEN = {}
t0s = time.time()
n_tot = NSAMP_S * 2 * K_REPS
for j in range(NSAMP_S):
    pk = samp_pks[j]
    for direction, pid in (('P', PID_P[j]),
                           ('A1', PID_A[j])):
        for rep in range(K_REPS):
            sd = seed_for(pk, direction, rep)
            sc = gen_sampled(pid, N_NEW, (), TEMP,
                             sd)
            s30 = gen_sampled(pid, N_NEW, (30,),
                              TEMP, sd)
            s32 = gen_sampled(pid, N_NEW, (32,),
                              TEMP, sd)
            SGEN[(pk, direction, rep)] = (sc, s30,
                                          s32)
    if j % 20 == 0:
        log('samp gen %d/%d (%.1fs)'
            % (j, NSAMP_S, time.time() - t0s))
log('sampled generation done: %d seqs x 3 conds '
    '(%.1fs)' % (n_tot, time.time() - t0s))

ST = {}


def track_all(seqkey, layers, tag):
    arr = np.zeros((n_tot, N_NEW + 1),
                   dtype=np.float32)
    idx = 0
    for j in range(NSAMP_S):
        pk = samp_pks[j]
        for pid, dname in ((PID_P[j], 'P'),
                           (PID_A[j], 'A1')):
            for rep in range(K_REPS):
                sc, s30, s32 = SGEN[(pk, dname,
                                     rep)]
                if seqkey == 'c':
                    seq = sc
                elif seqkey == 30:
                    seq = s30
                else:
                    seq = s32
                ms, _ = forward_track(pid, seq,
                                      layers)
                arr[idx] = ms
                idx += 1
    ST[tag] = arr
    log('%s tracked (%.1fs)'
        % (tag, time.time() - T0))


track_all('c', (), 'st_clean_clean')
drep = np.abs(ST['st_clean_clean']
              - z18['st_samp_cleanseq_clean'][:n_tot]) \
    .max() if n_tot <= len(
        z18['st_samp_cleanseq_clean']) else None
if drep is None:
    repro_v = 'sampled_pipeline_diverged'
    drep = float('nan')
    log('B-REPRO: n_tot exceeds 3118 array - '
        'diverged (config mismatch)')
elif float(drep) == 0.0:
    repro_v = 'sampled_pipeline_reproduced'
    log('B-REPRO: clean sampled margins bit-exact '
        'vs 3118 (max diff 0.00e+00)')
else:
    repro_v = 'sampled_pipeline_diverged'
    log('B-REPRO: clean sampled margins DIVERGE '
        'from 3118 (max diff %.3e) - internal '
        'clean used as baseline' % float(drep))

for X in (30, 32):
    track_all('c', (X,), 'st_clean_L%d' % X)
    track_all(X, (), 'st_L%d_clean' % X)
    track_all(X, (X,), 'st_L%d_L%d' % (X, X))

state_ratio = {}
closed_ratio = {}
for X in (30, 32):
    dms = (ST['st_clean_L%d' % X]
           - ST['st_clean_clean']) \
        .astype(np.float64)
    head = float(np.abs(dms[:, 0:3]).mean())
    tail = float(np.abs(
        dms[:, N_NEW - 3:N_NEW + 1]).mean())
    state_ratio['L%d' % X] = {
        'head': head, 'tail': tail,
        'ratio': tail / max(head, 1e-9)}
    dmc = (ST['st_L%d_L%d' % (X, X)]
           - ST['st_clean_clean']) \
        .astype(np.float64)
    head_c = float(np.abs(dmc[:, 0:3]).mean())
    tail_c = float(np.abs(
        dmc[:, N_NEW - 3:N_NEW + 1]).mean())
    closed_ratio['L%d' % X] = {
        'head': head_c, 'tail': tail_c,
        'ratio': tail_c / max(head_c, 1e-9)}
    log('SAMPLED STATE L%d: head=%.4f tail=%.4f '
        'ratio=%.3f | closed head=%.4f tail=%.4f '
        'ratio=%.3f'
        % (X, head, tail,
           state_ratio['L%d' % X]['ratio'],
           head_c, tail_c,
           closed_ratio['L%d' % X]['ratio']))

ys = {}
sag = {}
for (cname, seqidx) in (('clean', 0),
                        ('abl_L30', 1),
                        ('abl_L32', 2)):
    yc = 0
    agn = 0
    for (pk, dname, rep), seqs in SGEN.items():
        seq = seqs[seqidx]
        if any(t in yf for t in seq[:N_MATCH]):
            yc += 1
        if cname != 'clean' and seq == seqs[0]:
            agn += 1
    ys[cname] = yc / float(n_tot)
    if cname != 'clean':
        sag[cname] = agn / float(n_tot)
samp_diffs = {c: abs(ys[c] - ys['clean'])
              for c in ('abl_L30', 'abl_L32')}
samp_max = max(samp_diffs.values())
if beh_v == 'amplifier_behavioral':
    samp_v = ('sampled_confirms_behavioral'
              if samp_max >= 0.03
              else 'sampled_not_confirm')
else:
    samp_v = 'not_applicable'
log('B-SAMP: yes sampled clean=%.4f L30=%.4f '
    'L32=%.4f | seq_agree L30=%.4f L32=%.4f | '
    'max_diff=%.4f -> %s'
    % (ys['clean'], ys['abl_L30'],
       ys['abl_L32'], sag.get('abl_L30', -1),
       sag.get('abl_L32', -1), samp_max, samp_v))

rechk_samp = {}
stride_s = max(1, n_tot // 64)
idxs_s = list(range(0, n_tot, stride_s))[:64]
for X in (30, 32):
    rc = []
    for idx in idxs_s:
        j = idx // (2 * K_REPS)
        rem = idx % (2 * K_REPS)
        pid = PID_P[j] if rem < K_REPS \
            else PID_A[j]
        dname = 'P' if rem < K_REPS else 'A1'
        rep = rem % K_REPS
        sc, s30, s32 = SGEN[(samp_pks[j], dname,
                             rep)]
        seq = {30: s30, 32: s32}[X]
        _, r1 = forward_track(pid, seq, (X,))
        rc.append(r1)
    rechk_samp['L%d' % X] = float(np.mean(rc))
    log('own-recheck sampled L%d: %.4f (%d seqs)'
        % (X, rechk_samp['L%d' % X], len(rc)))

part_b = {
    'verdict': '%s|%s' % (beh_v, samp_v),
    'beh_verdict': beh_v,
    'samp_verdict': samp_v,
    'repro_verdict': repro_v,
    'repro_max_diff': (float(drep)
                       if drep is not None
                       else None),
    'yes_rate_greedy': yr,
    'agree_greedy': agr,
    'beh_max': beh_max,
    'yes_rate_sampled': ys,
    'seq_agree_sampled': sag,
    'samp_max': samp_max,
    'state_ratio': state_ratio,
    'closed_ratio': closed_ratio,
    'recheck_greedy': rechk_greedy,
    'recheck_sampled': rechk_samp,
    'n_pairs_greedy': NP_B,
    'n_samp_pairs': NSAMP_S,
    'k_reps': K_REPS,
    'temp': TEMP,
}

# ================================================================
# PART C: operator shape (offline)
# ================================================================
def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    return float(np.corrcoef(rankdata(a),
                             rankdata(b))[0, 1])


def ols_fit(cols, y):
    X = np.column_stack(
        [np.ones(len(y))] + list(cols))
    th, _, _, _ = np.linalg.lstsq(
        X, y, rcond=None)
    pred = X @ th
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return (1.0 - sse / sst), th, pred


def shape_fits(x, y, pair_ids):
    """lin/quad/pw fits + spearman; returns
    dict."""
    sp = spearman(x, y)
    r2l, th_l, _ = ols_fit([x], y)
    r2q, th_q, _ = ols_fit([x, x * x], y)
    bs = np.quantile(x, np.arange(1, 10) / 10.0)
    best = {'r2': -9e9, 'b': None}
    for b in bs:
        h = np.maximum(x - b, 0.0)
        r2b, _, _ = ols_fit([x, h], y)
        if r2b > best['r2']:
            best = {'r2': float(r2b),
                    'b': float(b)}
    d_quad = r2q - r2l
    d_pw = best['r2'] - r2l
    return {'spearman': sp, 'r2_lin': r2l,
            'r2_quad': float(r2q),
            'r2_pw_best': best['r2'],
            'pw_breakpoint': best['b'],
            'slope_lin': float(th_l[1]),
            'd_quad': float(d_quad),
            'd_pw': float(d_pw),
            'n': int(len(y))}


def gate_verdicts(fx):
    sp = fx['spearman']
    if sp <= -0.5:
        mono = 'mean_reversion_confirmed'
    elif sp <= -0.2:
        mono = 'mean_reversion_weak'
    else:
        mono = 'mean_reversion_absent'
    quad = ('nonlinear_curvature_present'
            if fx['d_quad'] >= 0.05
            else 'nonlinear_curvature_absent')
    gate = ('gated_operator'
            if fx['d_pw'] >= 0.05
            else 'linear_operator')
    return mono, quad, gate


# build sample arrays
XS = []
YS = []
DIRS = []
PIDS = []
TS = []
for (dname, mD) in (('P', mP), ('A1', mA)):
    isp = 1.0 if dname == 'P' else 0.0
    for j in range(NP_):
        for t in range(1, N_NEW + 1):
            XS.append(float(mD[j, t - 1]))
            YS.append(float(mD[j, t]
                            - mD[j, t - 1]))
            DIRS.append(isp)
            PIDS.append(j)
            TS.append(t)
XS = np.array(XS, dtype=np.float64)
YS = np.array(YS, dtype=np.float64)
DIRS = np.array(DIRS, dtype=np.float64)
PIDS = np.array(PIDS)
TS = np.array(TS)
assert len(XS) == NP_ * N_NEW * 2

mx_P = float(XS[DIRS == 1].mean())
my_P = float(YS[DIRS == 1].mean())
mx_A = float(XS[DIRS == 0].mean())
my_A = float(YS[DIRS == 0].mean())
xc = np.where(DIRS == 1, XS - mx_P, XS - mx_A)
yc = np.where(DIRS == 1, YS - my_P, YS - my_A)

fits_primary = shape_fits(xc, yc, PIDS)
mono_v, quad_v, gate_v = gate_verdicts(
    fits_primary)
log('PART C primary (centered): spearman=%.4f '
    'r2_lin=%.4f r2_quad=%.4f r2_pw=%.4f (b=%.3f) '
    'd_quad=%+.4f d_pw=%+.4f'
    % (fits_primary['spearman'],
       fits_primary['r2_lin'],
       fits_primary['r2_quad'],
       fits_primary['r2_pw_best'],
       fits_primary['pw_breakpoint'],
       fits_primary['d_quad'],
       fits_primary['d_pw']))
log('PART C GATES: %s | %s | %s'
    % (mono_v, quad_v, gate_v))

# 20-bin curve + pair bootstrap
edges_c = np.quantile(
    xc, np.arange(1, N_BINS) / float(N_BINS))
binc = np.searchsorted(edges_c, xc,
                       side='right')
bin_stats = []
bin_idx = [np.where(binc == b)[0]
           for b in range(N_BINS)]
rng = np.random.default_rng(3120)
boot = np.zeros((N_BOOT, N_BINS),
                dtype=np.float64)
for rep in range(N_BOOT):
    draws = rng.integers(0, NP_, NP_)
    w = np.bincount(draws, minlength=NP_)
    for b in range(N_BINS):
        ii = bin_idx[b]
        sw = w[PIDS[ii]]
        if sw.sum() == 0:
            boot[rep, b] = np.nan
        else:
            boot[rep, b] = float(
                (yc[ii] * sw).sum()
                / sw.sum())
for b in range(N_BINS):
    ii = bin_idx[b]
    bm = float(yc[ii].mean())
    lo, hi = np.nanpercentile(boot[:, b],
                              [2.5, 97.5])
    bin_stats.append({
        'bin': b, 'n': int(len(ii)),
        'x_center': float(xc[ii].mean()),
        'y_mean': bm, 'y_lo': float(lo),
        'y_hi': float(hi)})
log('C bin curve: %s'
    % ' '.join('%.3f' % r['y_mean']
               for r in bin_stats))

# secondary: raw pooled, per-direction, t>=2
fits_raw = shape_fits(XS, YS, PIDS)
mono_r, quad_r, gate_r = gate_verdicts(fits_raw)
log('C secondary raw pooled: spearman=%.4f '
    'r2_lin=%.4f d_quad=%+.4f d_pw=%+.4f -> %s|%s'
    % (fits_raw['spearman'],
       fits_raw['r2_lin'], fits_raw['d_quad'],
       fits_raw['d_pw'], mono_r, gate_r))
fits_by_dir = {}
for (dname, mD) in (('P', mP), ('A1', mA)):
    sel = (DIRS == (1.0 if dname == 'P' else 0.0))
    fx = shape_fits(XS[sel], YS[sel], PIDS[sel])
    fits_by_dir[dname] = fx
    log('C per-direction %s: spearman=%.4f '
        'r2_lin=%.4f d_quad=%+.4f d_pw=%+.4f'
        % (dname, fx['spearman'], fx['r2_lin'],
           fx['d_quad'], fx['d_pw']))
sel2 = (TS >= 2)
fits_t2 = shape_fits(xc[sel2], yc[sel2],
                     PIDS[sel2])
mono_2, quad_2, gate_2 = gate_verdicts(fits_t2)
log('C sensitivity t>=2 (n=%d): spearman=%.4f '
    'r2_lin=%.4f d_quad=%+.4f d_pw=%+.4f -> %s|%s'
    % (fits_t2['n'], fits_t2['spearman'],
       fits_t2['r2_lin'], fits_t2['d_quad'],
       fits_t2['d_pw'], mono_2, gate_2))
mono_flip = (mono_2 != mono_v)
gate_flip = (gate_2 != gate_v)
if mono_flip or gate_flip:
    log('C WARNING: sensitivity flips verdict '
        '(mono_flip=%d gate_flip=%d)'
        % (mono_flip, gate_flip))

# raw pooled fixed point (descriptive)
th_raw = ols_fit([XS], YS)[1]
fixed_raw = float(-th_raw[0] / th_raw[1]) \
    if abs(th_raw[1]) > 1e-9 else None

part_c = {
    'verdict': '%s|%s' % (mono_v, gate_v),
    'mono_verdict': mono_v,
    'quad_verdict': quad_v,
    'gate_verdict': gate_v,
    'primary': fits_primary,
    'raw_pooled': fits_raw,
    'raw_pooled_verdict': '%s|%s'
    % (mono_r, gate_r),
    'per_direction': fits_by_dir,
    't2_sensitivity': fits_t2,
    't2_verdict': '%s|%s' % (mono_2, gate_2),
    't2_flip': {'mono': bool(mono_flip),
                'gate': bool(gate_flip)},
    'bin_curve': bin_stats,
    'centering': {'mx_P': mx_P, 'my_P': my_P,
                  'mx_A1': mx_A, 'my_A1': my_A},
    'fixed_point_raw_lin': fixed_raw,
    'n_samples': int(len(XS)),
}

# ================================================================
# save
# ================================================================
npz_out = {
    'annP_c': annP_c, 'annP_s': annP_s,
    'annA_c': annA_c, 'annA_s': annA_s,
    'gen_abl_L30__P': GENX['abl_L30']['P'],
    'gen_abl_L30__A1': GENX['abl_L30']['A1'],
    'gen_abl_L32__P': GENX['abl_L32']['P'],
    'gen_abl_L32__A1': GENX['abl_L32']['A1'],
    'samp_tokens_clean': np.array(
        [SGEN[(pk, d, r)][0]
         for pk in samp_pks
         for d in ('P', 'A1')
         for r in range(K_REPS)], dtype=np.int32),
    'samp_tokens_L30': np.array(
        [SGEN[(pk, d, r)][1]
         for pk in samp_pks
         for d in ('P', 'A1')
         for r in range(K_REPS)], dtype=np.int32),
    'samp_tokens_L32': np.array(
        [SGEN[(pk, d, r)][2]
         for pk in samp_pks
         for d in ('P', 'A1')
         for r in range(K_REPS)], dtype=np.int32),
    'samp_pks': np.array(samp_pks),
    'st_clean_clean': ST['st_clean_clean'],
}
for X in (30, 32):
    npz_out['st_clean_L%d' % X] = \
        ST['st_clean_L%d' % X]
    npz_out['st_L%d_clean' % X] = \
        ST['st_L%d_clean' % X]
    npz_out['st_L%d_L%d' % (X, X)] = \
        ST['st_L%d_L%d' % (X, X)]
npz_out['c_xy_x'] = XS
npz_out['c_xy_y'] = YS
npz_out['c_xy_dir'] = DIRS
npz_out['c_xy_pair'] = PIDS
npz_out['c_xy_t'] = TS
npz_out['c_bin_edges'] = edges_c
npz_out['c_boot'] = boot
np.savez(os.path.join(OUT, 'p118_readout.npz'),
         **npz_out)

verdict_full = '%s|%s|%s' % (
    verdict_a, part_b['verdict'],
    part_c['verdict'])
results = {
    'verdict': verdict_full,
    'amp_layers': AMP_LAYERS,
    'n_records': NB,
    'n_pairs': NP_,
    'n_pairs_partb': NP_B,
    'n_new': N_NEW,
    'n_match': N_MATCH,
    'smoke': SMOKE,
    'part_a': part_a,
    'part_b': part_b,
    'part_c': part_c,
    'gates': seal['gates'],
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('VERDICT: %s' % verdict_full)
log('result.json written')
log('Phase 3120 done (%.1fs)'
    % (time.time() - T0))
print('PHASE3120_DONE verdict=%s' % verdict_full)
