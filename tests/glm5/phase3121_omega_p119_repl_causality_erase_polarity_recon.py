# -*- coding: utf-8 -*-
"""Phase 3121 (Omega-P119): counterfactual
restatement replacement (causality), joint L30+L32
erase-chain behavioral polarity + first-token fork +
yes/no-family readout robustness, two-component
forward reconstruction of the margin oscillation,
and position-by-class cross control.

Inputs (frozen): 3118 traj_readout.npz (greedy gen
tokens + margin trajectories), 3120 p118_readout.npz
(content-token annotations annP_c/annA_c/annP_s/
annA_s), 3105 material.json, 3113 capture_b.npz
(prompt rebuild), 3120 result.json (operator
params for Part C).

Part A (GPU, teacher-forced): per (pair, direction)
take the longest contiguous fact_strict span of the
greedy continuation; replay the SAME token sequence
under 4 conditions:
  c0 replay (baseline, must reproduce 3118 margins)
  c1 span -> queried-line tokens of another pair
  c2 span -> shuffled span tokens (same multiset)
  c3 span -> '.' punctuation fill
Effects D_c = mean over t in (k1..12] of
m_cond(t) - m_c0(t); E_10 = D_c1 - D_c2 (content
specificity), E_31 = D_c3 - D_c1 (syntax polarity).
Gates (frozen in design_seal.json BEFORE any run):
  A-REPRO: max|c0 - 3118| == 0 -> replay_bit_exact;
           < 1e-6 -> replay_ok; else diverged
           (non-fatal, c0 used as baseline).
  A-CAUS-CONTENT (P): E_10 >= +0.10 ->
      content_specific_confirmed; < +0.05 ->
      content_nonspecific; else content_partial.
      (A1: +0.05 / +0.025.)
  A-CAUS-SYNTAX (P): E_31 <= -0.05 ->
      syntax_polarity_confirmed; else
      syntax_polarity_absent.  (A1: -0.025.)
  Spans with length < 2 excluded from E_10 (shuffle
  degenerate); pairs without fact span excluded
  from all Part A effect stats (count reported).

Part B (GPU): joint ablation gen_greedy(30,32) on
all 672 pairs x 2 directions; yes_rate gate
(identical 3120 band 0.05/0.02); first-token fork
stats vs clean/L30/L32 (offline from npz); joint
tracking with TWO readouts w_dn = WU[yes]-WU[no]
and w_fam = mean(WU[yes_family])-mean(WU[no_family])
(clean tracking = Part A c0).  Gate B-FAM: pooled
pearson r of per-step increments (dm_dn vs dm_fam
over clean+joint, all pairs/dirs/steps) >= 0.8 ->
readout_robust; < 0.5 -> readout_sensitive; else
readout_partial.

Part C (offline): two-component forward
reconstruction m_hat(t+1) = m_hat(t) + s*(m_hat(t)
- m*) + content[dir][cls(j,t)][t], where content is
the per-(dir,class,t) class DEVIATION of dm
(class mean minus all-token mean at that step),
s and m* from 3120 raw primary fit.  Controls:
lin-only (no content), persistence (m_hat = m(0)).
Gates: R2_two >= 0.5 and R2_two - R2_lin >= 0.05 ->
reconstruction_confirmed; R2_two < 0.2 ->
reconstruction_failed; else reconstruction_partial.
AUC curve gate: corr(auc_two(t), auc_3118(t)) >=
0.9 -> curve_shape_confirmed.  MC noise gate:
sigma from teacher-forced residuals; coverage of
true m in +/-1.96*sigma*sqrt(t) band over 20 MC
reps (seed 3121) >= 0.90 -> coverage_ok.

Part D (offline): per-(dir,class,t) table for
content steps; gate D-POS: syntax per-t mean dm
< 0 for EVERY t and BOTH directions ->
position_independent_confirmed; else
position_confounded (min/max reported).
"""
import hashlib
import io
import json
import os
import random as _rnd
import time
import zlib
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p119_repl_causality_erase_' \
       'polarity_recon'
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
D20 = os.path.join(RDIR, 'phase3120',
                   'omega_p118_content_attr_amplifier_'
                   'behavior_opshape')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3121', NAME)
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
# frozen inputs + integrity assertions
# ================================================================
res13 = json.load(io.open(
    os.path.join(D13, 'result.json'),
    encoding='utf-8'))
assert res13['verdict'] == \
    'belief_robust|within_unit_replicated|' \
    'write_in_concentrated'
res18 = json.load(io.open(
    os.path.join(D18, 'result.json'),
    encoding='utf-8'))
assert res18['verdict'] == \
    'belief_decays_in_generation|' \
    'temporal_compensation|' \
    'top_ablation_changes_behavior'
res20 = json.load(io.open(
    os.path.join(D20, 'result.json'),
    encoding='utf-8'))
assert res20['verdict'] == \
    'attribution_unresolved|amplifier_partial|' \
    'not_applicable|mean_reversion_confirmed|' \
    'linear_operator'
assert res20['smoke'] is False
AMP_LAYERS = [30, 32]
N_NEW = 12
N_MATCH = 8
NP_A = 8 if SMOKE else 672      # Part A + tracking
NP_G = 24 if SMOKE else 672     # Part B greedy gen
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
assert len(auc18) == N_NEW + 1
assert abs(float(auc18[0])
           - 0.9809094210600907) < 1e-12
z20 = np.load(os.path.join(D20, 'p118_readout.npz'),
              allow_pickle=False)
annP_c = z20['annP_c']
annA_c = z20['annA_c']
assert annP_c.shape == (672, N_NEW)
assert annP_c.dtype == np.int8

SLOPE_3120 = float(
    res20['part_c']['primary']['slope_lin'])
MSTAR_3120 = float(
    res20['part_c']['fixed_point_raw_lin'])
assert abs(SLOPE_3120
           - (-0.6801486439276976)) < 1e-12
assert abs(MSTAR_3120
           - (-6.052991245322225)) < 1e-9

# ================================================================
# seal (frozen BEFORE any computation)
# ================================================================
seal = {
    'phase': 3121,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'amp_layers': AMP_LAYERS,
    'n_new': N_NEW,
    'n_match': N_MATCH,
    'np_a': NP_A, 'np_g': NP_G,
    'data_sources': {
        'tokens_and_trajectories':
            'phase3118 traj_readout.npz (gen_clean, '
            'gt_cleanseq margins)',
        'annotation': 'phase3120 p118_readout.npz '
                      'annP_c/annA_c (class codes '
                      '0/1/2/3/4/9)',
        'operator_params': 'phase3120 result.json '
                           'primary slope_lin + '
                           'fixed_point_raw_lin',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz'},
    'conditions_part_a': [
        'c0_replay', 'c1_other_fact_line',
        'c2_shuffle_span', 'c3_punct_fill'],
    'readouts': {
        'w_dn': 'WU[YES_ID] - WU[NO_ID]',
        'w_fam': 'mean(WU[yes_family]) - '
                 'mean(WU[no_family])'},
    'gates': {
        'A_repro': 'max|c0-3118| == 0 -> '
                   'replay_bit_exact; < 1e-6 -> '
                   'replay_ok; else diverged '
                   '(non-fatal, c0 baseline)',
        'A_caus_content_P': 'E_10 = D_c1 - D_c2 '
                            '>= +0.10 -> '
                            'content_specific_'
                            'confirmed; < +0.05 -> '
                            'content_nonspecific; '
                            'else content_partial',
        'A_caus_content_A1': 'same with +0.05 / '
                             '+0.025',
        'A_caus_syntax_P': 'E_31 = D_c3 - D_c1 '
                           '<= -0.05 -> '
                           'syntax_polarity_'
                           'confirmed; else '
                           'syntax_polarity_absent',
        'A_caus_syntax_A1': 'same with -0.025',
        'A_exclusions': 'spans len<2 excluded from '
                        'E_10 (shuffle degenerate); '
                        'pairs w/o fact span '
                        'excluded from Part A '
                        'effects (count reported)',
        'B_joint': '|yes_joint - yes_clean| >= 0.05 '
                   '-> erase_joint_behavioral; '
                   '<= 0.02 -> erase_joint_neutral; '
                   'else erase_joint_partial',
        'B_fam': 'pooled pearson r(dm_dn, dm_fam) '
                 '>= 0.8 -> readout_robust; '
                 '< 0.5 -> readout_sensitive; '
                 'else readout_partial',
        'C_fit': 'R2_two >= 0.5 AND dR2 >= 0.05 -> '
                 'reconstruction_confirmed; '
                 'R2_two < 0.2 -> '
                 'reconstruction_failed; else '
                 'reconstruction_partial',
        'C_curve': 'corr(auc_two, auc_3118) >= 0.9 '
                   '-> curve_shape_confirmed',
        'C_mc': 'coverage(true m in +/-1.96*sigma*'
                'sqrt(t) band, 20 reps seed 3121) '
                '>= 0.90 -> coverage_ok',
        'D_pos': 'syntax per-t mean dm < 0 for '
                 'every t in both directions -> '
                 'position_independent_confirmed; '
                 'else position_confounded'},
    'mc': {'n_reps': 20, 'seed': 3121},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('seal frozen (%s)' % seal['created'])

# ================================================================
# prompt rebuild (identical to 3114-3120)
# ================================================================


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
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
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
NP_A = min(NP_A, NP_)
NP_G = min(NP_G, NP_)
log('records rebuilt: %d (%d pairs); NP_A=%d NP_G=%d'
    % (NB, NP_, NP_A, NP_G))

# queried-line text per pair (for c1 replacement)
q_text = {}
for pk in pks:
    (s, o) = (int(v) for v in pk.split('_'))
    q_text[pk] = 'The %s %s the %s.' % (
        ents_all[s], PREDS_all[p2r[pk]],
        ents_all[o])

# ================================================================
# GPU setup
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
log('model loaded NL=%d' % NL)
WU = model.lm_head.weight.detach()
assert model.lm_head.bias is None
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])
w_dn = (WU[YES_ID] - WU[NO_ID]) \
    .float().cpu().numpy()
YES_FAMILY = []
for ys in ['yes', 'Yes', ' yes', ' Yes',
           'YES']:
    YES_FAMILY += list(tok(
        ys, add_special_tokens=False)['input_ids'])
YES_FAMILY = sorted(set(YES_FAMILY))
NO_FAMILY = []
for ns in ['no', 'No', ' no', ' No', 'NO']:
    NO_FAMILY += list(tok(
        ns, add_special_tokens=False)['input_ids'])
NO_FAMILY = sorted(set(NO_FAMILY))
yf_ids = np.array(YES_FAMILY)
nf_ids = np.array(NO_FAMILY)
w_fam = (WU[yf_ids].mean(0)
         - WU[nf_ids].mean(0)) \
    .float().cpu().numpy()
log('readouts ready: |w_dn|=%.4f |w_fam|=%.4f'
    % (float(np.linalg.norm(w_dn)),
       float(np.linalg.norm(w_fam))))

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


def forward_track2(prompt_ids, gen_tokens,
                   layers_abl):
    """teacher-forced forward; returns per-step
    (N_NEW+1,) margins under BOTH readouts."""
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl)
    gen_tokens = [int(x) for x in gen_tokens]
    ids = list(prompt_ids) + gen_tokens
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(gen_tokens) + 1
    ms_dn = np.zeros(npts, dtype=np.float64)
    ms_fam = np.zeros(npts, dtype=np.float64)
    with torch.inference_mode():
        out = model(t_in,
                    output_hidden_states=True,
                    use_cache=False)
        hs = out.hidden_states[NL][0]
        norm = model.model.norm
        h_all = norm(hs).float().cpu().numpy()
        for k in range(npts):
            v = h_all[pos0 + k]
            ms_dn[k] = float(v @ w_dn)
            ms_fam[k] = float(v @ w_fam)
        del out
    ABL['mode'] = None
    ABL['layers'] = set()
    return ms_dn, ms_fam


PID_P = [tok(texts[hP[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]
PID_A = [tok(texts[hA1[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]
gP18 = z18['gen_clean__P']
gA18 = z18['gen_clean__A1']

# ================================================================
# PART A: counterfactual restatement replacement
# ================================================================
DOT_ID = tok('.', add_special_tokens=False)
DOT_ID = int(DOT_ID['input_ids'][0])
log('DOT_ID=%d' % DOT_ID)

CONDS = ('c0', 'c1', 'c2', 'c3')
PA = {c: {'P': None, 'A1': None} for c in CONDS}
PA_FAM = {c: {'P': None, 'A1': None}
          for c in CONDS}
span_info = []
t0a = time.time()
for dcode, ann, gen18, PID, hmap in (
        ('P', annP_c, gP18, PID_P, hP),
        ('A1', annA_c, gA18, PID_A, hA1)):
    m_dn = {c: np.zeros((NP_A, N_NEW + 1),
                        dtype=np.float32)
            for c in CONDS}
    m_fm = {c: np.zeros((NP_A, N_NEW + 1),
                        dtype=np.float32)
            for c in CONDS}
    for j in range(NP_A):
        pk = pks[j]
        prompt_ids = PID[j]
        base = [int(t) for t in gen18[j]]
        # longest contiguous fact span in
        # content steps (ann cols 1..11)
        best = None
        k1 = None
        for k in range(1, N_NEW):
            isf = int(ann[j, k]) in (3, 4)
            if isf and k1 is None:
                k1 = k
            if (not isf or k == N_NEW - 1) \
                    and k1 is not None:
                k2 = (k if isf and k == N_NEW - 1
                      else k - 1)
                if best is None or \
                        (k2 - k1) > (best[1]
                                     - best[0]):
                    best = (k1, k2)
                k1 = None
        span_info.append(
            {'pk': pk, 'dir': dcode,
             'k1': None if best is None
             else int(best[0]),
             'k2': None if best is None
             else int(best[1]),
             'len': 0 if best is None
             else int(best[1] - best[0] + 1)})
        toks = {c: list(base) for c in CONDS}
        if best is not None:
            (k1, k2) = best
            L = k2 - k1 + 1
            # c1: another pair's queried line
            rng1 = _rnd.Random(zlib.crc32(
                ('c1|%s|%s' % (pk, dcode))
                .encode('ascii')))
            cand = [q for q in pks if q != pk]
            line_pk = cand[rng1.randrange(
                len(cand))]
            lt = tok(q_text[line_pk],
                     add_special_tokens=False)
            lt = [int(x) for x in lt['input_ids']]
            fill1 = (lt * ((L // len(lt)) + 1))[:L]
            toks['c1'][k1:k2 + 1] = fill1
            # c2: shuffle span tokens
            rng2 = _rnd.Random(zlib.crc32(
                ('c2|%s|%s' % (pk, dcode))
                .encode('ascii')))
            seg = base[k1:k2 + 1]
            if L >= 2:
                rng2.shuffle(seg)
            toks['c2'][k1:k2 + 1] = seg
            # c3: punctuation fill
            toks['c3'][k1:k2 + 1] = [DOT_ID] * L
        for c in CONDS:
            md, mf = forward_track2(
                prompt_ids, toks[c], ())
            m_dn[c][j] = md
            m_fm[c][j] = mf
        if j % 100 == 0:
            log('partA %s %d/%d (%.1fs)'
                % (dcode, j, NP_A,
                   time.time() - t0a))
    for c in CONDS:
        PA[c][dcode] = m_dn[c]
        PA_FAM[c][dcode] = m_fm[c]
log('partA done (%.1fs)' % (time.time() - t0a))

# repro gate: c0 vs 3118 margins
mP18 = z18['gt_cleanseq_clean__P']
mA18 = z18['gt_cleanseq_clean__A1']
ref18 = {'P': mP18[:NP_A], 'A1': mA18[:NP_A]}
drep = 0.0
for dcode in ('P', 'A1'):
    drep = max(drep, float(np.abs(
        PA['c0'][dcode] - ref18[dcode]).max()))
if drep == 0.0:
    repro_v = 'replay_bit_exact'
elif drep < 1e-6:
    repro_v = 'replay_ok'
else:
    repro_v = 'replay_diverged'
log('A-REPRO: max diff %.3e -> %s'
    % (drep, repro_v))

# Part A effect stats
A_eff = {}
for dcode in ('P', 'A1'):
    ann = annP_c if dcode == 'P' else annA_c
    Ds = {c: [] for c in CONDS}
    lens = []
    n_span = 0
    n_len2 = 0
    for j in range(NP_A):
        si = span_info[j + (0 if dcode == 'P'
                            else NP_A)]
        if si['k1'] is None:
            continue
        n_span += 1
        lens.append(si['len'])
        k1, k2 = si['k1'], si['k2']
        if si['len'] >= 2:
            n_len2 += 1
        for c in CONDS:
            diff = (PA[c][dcode][j]
                    .astype(np.float64)
                    - PA['c0'][dcode][j]
                    .astype(np.float64))
            Ds[c].append(
                float(diff[k1 + 1:].mean()))
    if n_span == 0:
        A_eff[dcode] = {'n_span': 0}
        continue
    Dm = {c: float(np.mean(Ds[c])) for c in CONDS}
    e10_all = [a - b
               for (a, b, ln) in
               zip(Ds['c1'], Ds['c2'], lens)
               if ln >= 2]
    assert len(e10_all) == n_len2
    e31_all = [a - b for (a, b) in
               zip(Ds['c3'], Ds['c1'])]
    e10 = float(np.mean(e10_all))
    e31 = float(np.mean(e31_all))
    A_eff[dcode] = {
        'n_span': n_span, 'n_len2': n_len2,
        'D_mean': Dm,
        'E_10_mean': e10, 'E_31_mean': e31,
        'E_10_sd': float(np.std(e10_all)),
        'E_31_sd': float(np.std(e31_all))}
th_c = {'P': (0.10, 0.05), 'A1': (0.05, 0.025)}
th_s = {'P': -0.05, 'A1': -0.025}
vc = {}
vs = {}
for dcode in ('P', 'A1'):
    if A_eff[dcode].get('n_span', 0) == 0:
        vc[dcode] = 'no_spans'
        vs[dcode] = 'no_spans'
        continue
    hi, lo = th_c[dcode]
    e10 = A_eff[dcode]['E_10_mean']
    vc[dcode] = ('content_specific_confirmed'
                 if e10 >= hi else
                 ('content_nonspecific'
                  if e10 < lo
                  else 'content_partial'))
    e31 = A_eff[dcode]['E_31_mean']
    vs[dcode] = ('syntax_polarity_confirmed'
                 if e31 <= th_s[dcode] else
                 'syntax_polarity_absent')
log('A-CAUS: P E10=%.4f E31=%.4f -> %s|%s; '
    'A1 E10=%.4f E31=%.4f -> %s|%s'
    % (A_eff['P'].get('E_10_mean', float('nan')),
       A_eff['P'].get('E_31_mean', float('nan')),
       vc['P'], vs['P'],
       A_eff['A1'].get('E_10_mean', float('nan')),
       A_eff['A1'].get('E_31_mean', float('nan')),
       vc['A1'], vs['A1']))

# ================================================================
# PART B: joint ablation behavior + family readout
# ================================================================
GJ = {'P': None, 'A1': None}
MJ_DN = {'P': None, 'A1': None}
MJ_FAM = {'P': None, 'A1': None}
t0b = time.time()
for dcode, PID, hmap in (('P', PID_P, hP),
                         ('A1', PID_A, hA1)):
    gj = np.zeros((NP_G, N_NEW), dtype=np.int32)
    for j in range(NP_G):
        gj[j] = gen_greedy(PID[j], N_NEW,
                           (30, 32))
        if j % 100 == 0:
            log('partB gen %s %d/%d (%.1fs)'
                % (dcode, j, NP_G,
                   time.time() - t0b))
    GJ[dcode] = gj
log('partB joint greedy done (%.1fs)'
    % (time.time() - t0b))
t0b2 = time.time()
for dcode, PID, hmap in (('P', PID_P, hP),
                         ('A1', PID_A, hA1)):
    mjd = np.zeros((NP_A, N_NEW + 1),
                   dtype=np.float32)
    mjf = np.zeros((NP_A, N_NEW + 1),
                   dtype=np.float32)
    for j in range(NP_A):
        md, mf = forward_track2(
            PID[j], list(GJ[dcode][j]), (30, 32))
        mjd[j] = md
        mjf[j] = mf
        if j % 100 == 0:
            log('partB track %s %d/%d (%.1fs)'
                % (dcode, j, NP_A,
                   time.time() - t0b2))
    MJ_DN[dcode] = mjd
    MJ_FAM[dcode] = mjf
log('partB joint tracking done (%.1fs)'
    % (time.time() - t0b2))

yf = set(YES_FAMILY)


def any_yes(seq2d, nmatch=N_MATCH):
    cnt = 0
    for row in seq2d:
        if any(int(t) in yf
               for t in row[:nmatch]):
            cnt += 1
    return cnt / float(seq2d.shape[0])


yr_clean = any_yes(np.vstack([gP18[:NP_G],
                              gA18[:NP_G]]))
yr_joint = any_yes(np.vstack([GJ['P'], GJ['A1']]))
db = abs(yr_joint - yr_clean)
b_joint_v = ('erase_joint_behavioral'
             if db >= 0.05 else
             ('erase_joint_neutral'
              if db <= 0.02 else
              'erase_joint_partial'))
log('B-JOINT: clean %.4f -> joint %.4f '
    '(d=%.4f) -> %s'
    % (yr_clean, yr_joint, db, b_joint_v))

# first-token fork stats (clean/L30/L32 from npz,
# joint fresh)
def first_rates(seq2d):
    fy = float(np.mean([int(t) in yf
                        for t in seq2d[:, 0]]))
    fn = float(np.mean([int(t) in set(NO_FAMILY)
                        for t in seq2d[:, 0]]))
    return fy, fn


z20g30 = np.vstack([z20['gen_abl_L30__P'],
                    z20['gen_abl_L30__A1']])
z20g32 = np.vstack([z20['gen_abl_L32__P'],
                    z20['gen_abl_L32__A1']])
fork = {}
for (nm, arr) in (('clean',
                   np.vstack([gP18[:NP_G],
                              gA18[:NP_G]])),
                  ('abl_L30', z20g30[:NP_G]),
                  ('abl_L32', z20g32[:NP_G]),
                  ('abl_joint',
                   np.vstack([GJ['P'],
                              GJ['A1']]))):
    fy, fn = first_rates(arr)
    fork[nm] = {'first_yes': fy,
                'first_no': fn,
                'yes_rate8': any_yes(arr)}

# B-FAM: pooled increment correlation dn vs fam
inc_dn = []
inc_fm = []
for dcode in ('P', 'A1'):
    for src_m, tag in ((PA['c0'][dcode], 'c'),
                       (MJ_DN[dcode], 'j')):
        a = src_m.astype(np.float64)
        if tag == 'c':
            b = PA_FAM['c0'][dcode] \
                .astype(np.float64)
        else:
            b = MJ_FAM[dcode] \
                .astype(np.float64)
        inc_dn.append(a[:, 1:] - a[:, :-1])
        inc_fm.append(b[:, 1:] - b[:, :-1])
xd = np.concatenate([x.ravel() for x in inc_dn])
yd = np.concatenate([x.ravel() for x in inc_fm])
r_fam = float(np.corrcoef(xd, yd)[0, 1])
b_fam_v = ('readout_robust' if r_fam >= 0.8 else
           ('readout_sensitive'
            if r_fam < 0.5 else 'readout_partial'))
log('B-FAM: pooled r=%.4f -> %s'
    % (r_fam, b_fam_v))

# ================================================================
# PART C: two-component forward reconstruction
# ================================================================
mP = z18['gt_cleanseq_clean__P'].astype(np.float64)
mA = z18['gt_cleanseq_clean__A1'].astype(np.float64)
MSEQ = {'P': mP, 'A1': mA}
ANN = {'P': annP_c, 'A1': annA_c}
# per-(dir, cls, t) content deviation table
content = {}
for dcode in ('P', 'A1'):
    dm = (MSEQ[dcode][:, 1:]
          - MSEQ[dcode][:, :-1])  # (672,12) f64
    tab = np.zeros((10, N_NEW), dtype=np.float64)
    cnt = np.zeros((10, N_NEW), dtype=np.int64)
    for k in range(N_NEW):
        col = ANN[dcode][:, k].astype(np.int64)
        dv = dm[:, k]
        allm = dv.mean()
        for cc in range(10):
            sel = (col == cc)
            if sel.any():
                tab[cc, k] = float(
                    dv[sel].mean() - allm)
                cnt[cc, k] = int(sel.sum())
    content[dcode] = tab
S = SLOPE_3120
MS = MSTAR_3120


def recon(use_content):
    out = {}
    for dcode in ('P', 'A1'):
        seq = MSEQ[dcode]
        mh = np.zeros((NP_, N_NEW + 1),
                      dtype=np.float64)
        mh[:, 0] = seq[:, 0]
        for k in range(N_NEW):
            step = S * (mh[:, k] - MS)
            if use_content:
                col = ANN[dcode][:, k] \
                    .astype(np.int64)
                dev = content[dcode][col, k]
                step = step + dev
            mh[:, k + 1] = mh[:, k] + step
        out[dcode] = mh
    return out


rec_two = recon(True)
rec_lin = recon(False)
rec_per = {d: np.repeat(
    MSEQ[d][:, :1], N_NEW + 1, axis=1)
    for d in ('P', 'A1')}


def r2_all(rec):
    num = 0.0
    den = 0.0
    gm = 0.0
    n = 0
    for dcode in ('P', 'A1'):
        X = MSEQ[dcode][:, 1:].ravel()
        Y = rec[dcode][:, 1:].ravel()
        gm += X.sum()
        n += len(X)
        num += ((X - Y) ** 2).sum()
    gm /= n
    for dcode in ('P', 'A1'):
        X = MSEQ[dcode][:, 1:].ravel()
        den += ((X - gm) ** 2).sum()
    return float(1.0 - num / den)


r2_two = r2_all(rec_two)
r2_lin = r2_all(rec_lin)
r2_per = r2_all(rec_per)
c_fit_v = ('reconstruction_confirmed'
           if (r2_two >= 0.5
               and r2_two - r2_lin >= 0.05) else
           ('reconstruction_failed'
            if r2_two < 0.2 else
            'reconstruction_partial'))
log('C-FIT: r2_two=%.4f r2_lin=%.4f '
    'r2_pers=%.4f -> %s'
    % (r2_two, r2_lin, r2_per, c_fit_v))

# AUC curve comparison


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
        while j + 1 < len(x) \
                and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0)
                 / (n1 * n2))


auc_true = np.array(
    [float(auc18[t]) for t in range(N_NEW + 1)])
auc_two = np.array([
    auc_mw(rec_two['P'][:, t],
           rec_two['A1'][:, t])
    for t in range(N_NEW + 1)])
auc_lin = np.array([
    auc_mw(rec_lin['P'][:, t],
           rec_lin['A1'][:, t])
    for t in range(N_NEW + 1)])
r_curve = float(np.corrcoef(auc_two,
                            auc_true)[0, 1])
r_curve_lin = float(np.corrcoef(auc_lin,
                                auc_true)[0, 1])
c_curve_v = ('curve_shape_confirmed'
             if r_curve >= 0.9 else
             'curve_shape_weak')
log('C-CURVE: r=%.4f (lin %.4f) -> %s'
    % (r_curve, r_curve_lin, c_curve_v))

# MC noise coverage
resid = []
for dcode in ('P', 'A1'):
    dm = (MSEQ[dcode][:, 1:]
          - MSEQ[dcode][:, :-1])
    lin = S * (MSEQ[dcode][:, :-1] - MS)
    col = ANN[dcode].astype(np.int64)
    dev = content[dcode][col,
                         np.arange(N_NEW)]
    resid.append((dm - lin - dev).ravel())
sigma = float(np.std(np.concatenate(resid)))
rng_mc = np.random.default_rng(3121)
covered = 0
tot = 0
for rep in range(seal['mc']['n_reps']):
    for dcode in ('P', 'A1'):
        seq = MSEQ[dcode]
        mh = seq[:, 0].copy()
        for k in range(N_NEW):
            ck = ANN[dcode][:, k] \
                .astype(np.int64)
            step = S * (mh - MS) \
                + content[dcode][ck, k] \
                + rng_mc.normal(0.0, sigma,
                                len(mh))
            mh = mh + step
            lo_b = mh - 1.96 * sigma * np.sqrt(k + 1)
            hi_b = mh + 1.96 * sigma * np.sqrt(k + 1)
            covered += int(((seq[:, k + 1] >= lo_b)
                            & (seq[:, k + 1] <= hi_b))
                           .sum())
            tot += len(mh)
cov = covered / float(tot)
c_mc_v = 'coverage_ok' if cov >= 0.90 \
    else 'coverage_low'
log('C-MC: sigma=%.4f coverage=%.4f -> %s'
    % (sigma, cov, c_mc_v))

# ================================================================
# PART D: position x class cross control
# ================================================================
pertab = {}
d_ok = True
for dcode in ('P', 'A1'):
    dm = (MSEQ[dcode][:, 1:]
          - MSEQ[dcode][:, :-1])[:, 1:]
    ann = ANN[dcode][:, 1:]
    rows = []
    for t in range(dm.shape[1]):
        row = {'t': t + 2}
        for cc, nm in ((1, 'syntax'),
                       (4, 'fact_strict'),
                       (0, 'other')):
            sel = (ann[:, t] == cc)
            row[nm] = {
                'n': int(sel.sum()),
                'mean_dm': float(dm[sel].mean())
                if sel.any() else None}
        rows.append(row)
        if row['syntax']['mean_dm'] is not None \
                and row['syntax']['mean_dm'] >= 0:
            d_ok = False
    pertab[dcode] = rows
d_pos_v = ('position_independent_confirmed'
           if d_ok else 'position_confounded')
log('D-POS: syntax per-t all negative=%s -> %s'
    % (d_ok, d_pos_v))

# ================================================================
# verdict + save
# ================================================================
verdict = '%s|%s|%s|%s|%s|%s' % (
    vc['P'] if vc['P'] == vc['A1']
    else '%s/%s' % (vc['P'], vc['A1']),
    vs['P'] if vs['P'] == vs['A1']
    else '%s/%s' % (vs['P'], vs['A1']),
    b_joint_v, b_fam_v,
    '%s|%s|%s' % (c_fit_v, c_curve_v, c_mc_v),
    d_pos_v)
log('VERDICT: %s' % verdict)

npz_out = {
    'span_info_len': np.array(
        [s['len'] for s in span_info],
        dtype=np.int32),
    'span_info_dir': np.array(
        [1 if s['dir'] == 'P' else 0
         for s in span_info], dtype=np.int32),
}
for c in CONDS:
    npz_out['pa_%s_dn__P' % c] = PA[c]['P']
    npz_out['pa_%s_dn__A1' % c] = PA[c]['A1']
    npz_out['pa_%s_fam__P' % c] = PA_FAM[c]['P']
    npz_out['pa_%s_fam__A1' % c] = \
        PA_FAM[c]['A1']
npz_out['gj_P'] = GJ['P']
npz_out['gj_A1'] = GJ['A1']
npz_out['mj_dn_P'] = MJ_DN['P']
npz_out['mj_dn_A1'] = MJ_DN['A1']
npz_out['mj_fam_P'] = MJ_FAM['P']
npz_out['mj_fam_A1'] = MJ_FAM['A1']
npz_out['content_P'] = content['P']
npz_out['content_A1'] = content['A1']
npz_out['rec_two_P'] = rec_two['P']
npz_out['rec_two_A1'] = rec_two['A1']
npz_out['auc_two'] = auc_two
def _pt(rows, nm):
    return [np.nan
            if r[nm]['mean_dm'] is None
            else r[nm]['mean_dm']
            for r in rows]


npz_out['pertab_dm_P'] = np.array(
    [_pt(pertab['P'], 'syntax'),
     _pt(pertab['P'], 'fact_strict'),
     _pt(pertab['P'], 'other')],
    dtype=np.float64)
npz_out['pertab_dm_A1'] = np.array(
    [_pt(pertab['A1'], 'syntax'),
     _pt(pertab['A1'], 'fact_strict'),
     _pt(pertab['A1'], 'other')],
    dtype=np.float64)
np.savez(os.path.join(OUT, 'p119_readout.npz'),
         **npz_out)

results = {
    'phase': 3121,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'n_pairs': NP_,
    'np_a': NP_A,
    'np_g': NP_G,
    'runtime_s': round(time.time() - T0, 1),
    'part_a': {
        'verdict': '%s|%s' % (
            vc['P'] if vc['P'] == vc['A1']
            else '%s/%s' % (vc['P'], vc['A1']),
            vs['P'] if vs['P'] == vs['A1']
            else '%s/%s' % (vs['P'], vs['A1'])),
        'repro': {'verdict': repro_v,
                  'max_diff': drep},
        'effects': A_eff,
        'n_span_P': sum(
            1 for s in span_info
            if s['dir'] == 'P'
            and s['k1'] is not None),
        'n_span_A1': sum(
            1 for s in span_info
            if s['dir'] == 'A1'
            and s['k1'] is not None),
        'gates': {'P': vc['P'], 'A1': vc['A1'],
                  'syntax_P': vs['P'],
                  'syntax_A1': vs['A1']},
    },
    'part_b': {
        'verdict': '%s|%s' % (b_joint_v, b_fam_v),
        'yes_rate_clean': yr_clean,
        'yes_rate_joint': yr_joint,
        'yes_delta': yr_joint - yr_clean,
        'joint_verdict': b_joint_v,
        'first_token': fork,
        'fam_r': r_fam,
        'fam_verdict': b_fam_v,
    },
    'part_c': {
        'verdict': '%s|%s|%s' % (c_fit_v,
                                 c_curve_v,
                                 c_mc_v),
        'r2_two': r2_two,
        'r2_lin': r2_lin,
        'r2_persistence': r2_per,
        'd_r2': r2_two - r2_lin,
        'fit_verdict': c_fit_v,
        'auc_r': r_curve,
        'auc_r_lin': r_curve_lin,
        'curve_verdict': c_curve_v,
        'sigma': sigma,
        'coverage': cov,
        'mc_verdict': c_mc_v,
        'slope_3120': S,
        'mstar_3120': MS,
    },
    'part_d': {
        'verdict': d_pos_v,
        'syntax_n_zero_P': sum(
            1 for r in pertab['P']
            if r['syntax']['mean_dm']
            is None),
        'syntax_n_zero_A1': sum(
            1 for r in pertab['A1']
            if r['syntax']['mean_dm']
            is None),
        'syntax_min_P': min(
            (r['syntax']['mean_dm']
             for r in pertab['P']
             if r['syntax']['mean_dm']
             is not None), default=None),
        'syntax_max_P': max(
            (r['syntax']['mean_dm']
             for r in pertab['P']
             if r['syntax']['mean_dm']
             is not None), default=None),
        'syntax_min_A1': min(
            (r['syntax']['mean_dm']
             for r in pertab['A1']
             if r['syntax']['mean_dm']
             is not None), default=None),
        'syntax_max_A1': max(
            (r['syntax']['mean_dm']
             for r in pertab['A1']
             if r['syntax']['mean_dm']
             is not None), default=None),
    },
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False,
              indent=1)
log('saved result.json + p119_readout.npz')
print('PHASE3121 DONE verdict=%s (smoke=%s)'
      % (verdict, SMOKE))
