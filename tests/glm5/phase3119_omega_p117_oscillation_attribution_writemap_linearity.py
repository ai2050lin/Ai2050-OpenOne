# -*- coding: utf-8 -*-
"""Phase 3119 (Omega-P117): oscillation mechanism,
three-pronged test (T4, second phase).

Preregistered in 3118 MEMO section 5; gates frozen
in this seal BEFORE any observation.

Question: 3118 found the belief margin AUC decays
0.981 -> 0.672 over 12 greedy steps with STRONG
oscillation (0.518..0.972).  WHY does it oscillate,
WHERE in the layer stack does the state effect
shrink vs amplify, and is the per-step rewrite
operator linear?

Part A (offline, reuses 3118 npz - no GPU):
  token attribution of the oscillation.
  Hypothesis: steps where the model consumes a yes-
  family token drive AUC recovery; content-token
  steps drive collapse.  AUC(t) is a monotone
  functional of the per-pair gap gap_j(t) =
  m_P_j(t) - m_A1_j(t), so the right object is
  dGap_j(t) grouped by (class of g_t^P, class of
  g_t^A1).  Self-selection confound (high margin
  causes yes generation) is controlled by decile
  bins of m(t-1) within each direction.
  Gates:
   A-SEM-yes: per direction, deciles of m(t-1);
     valid bin = n_yes>=30 & n_other>=30; pass if
     >=8 valid bins AND >=70% of valid bins have
     mean dm(yes) > mean dm(other) in BOTH P and A1
     -> yes_semantic_confirmed
   A-SEM-no: same, mean dm(no) < mean dm(other)
     -> no_semantic_confirmed; <8 valid bins ->
     not_evaluable
   A-SEL: AUC(m(t-1) | g_t=yes vs other) per
     direction; mean(P,A1) >= 0.65 ->
     self_selection_present; <= 0.55 absent;
     else partial
   A-GAP: pooled mean dGap(yes,other) > 0 AND mean
     dGap(other,yes) < 0 AND per-step sign
     consistency (steps with n>=30; >=70% positive
     for (yes,other), >=70% negative for
     (other,yes); if <5 valid steps consistency is
     skipped) -> gap_attribution_confirmed
   A verdict: yes_semantic_confirmed AND
     (no_semantic_confirmed OR not_evaluable) AND
     gap_attribution_confirmed
     -> attribution_semantic_confirmed
     elif NOT yes_semantic_confirmed AND A-SEL
     present -> attribution_selection_dominant
     else -> attribution_unresolved

Part B (GPU): time x layer write map.
  Replay the FROZEN 3118 clean greedy sequences
  (gen_clean from 3118 npz, bit-exact reuse) under
  single-layer MLP ablation for every layer
  L14..L35 (22 conditions, L26 first as integrity
  probe).  dm_L(t) = m_ablL(t) - m_clean(t) on the
  same tokens (pure state effect).  ratio_L =
  mean|dm(9..12)| / mean|dm(0..2)|.
  Gates:
   B-OVL: L26/L33/L31 replay trajectories
     bit-exact vs 3118 npz (max abs diff == 0.0)
     -> replay_integrity_confirmed; else abort
     remaining replays, verdict
     replay_integrity_fail
   B-MAP: TOP3 all ratio < 0.7 AND >= 1 non-TOP3
     layer ratio > 1.1 -> compensation_layer_
     specific; TOP3 all < 0.7 AND no layer > 1.1
     -> compensation_global; else
     compensation_mixed
   B-L35: ratio_L35 > 1.0 -> rebuild_layer_
     confirmed (3116: L35 is the unique strong
     positive rebuilder, +0.376); else
     rebuild_layer_absent

Part C (GPU-lite): one-step predictability of
  margin(t) - linearity of the rewrite operator.
  Regress m(t) on m(t-1) and e_yes(g_t) =
  emb(g_t) . (w_yes - w_no) (embedding-side
  projection of the consumed token).
  Samples: all (pair j, step t=1..12, direction)
  = 672*12*2 = 16128.  Split by PAIR PARITY
  (odd j train / even j test, deterministic).
  Models: persist pred = m(t-1);
   base  [1, m(t-1), is_P]
   full  [1, m(t-1), e_yes, is_P]
   inter [1, m(t-1), e_yes, m(t-1)*e_yes, is_P]
  Gates:
   C-TOK: dR2 = R2_full - R2_base >= 0.10 ->
     token_injection_tracked; <= 0.02 ->
     token_injection_untracked; else partial
   C-LIN: R2_full >= 0.7 -> rewrite_mostly_linear;
     0.4-0.7 -> rewrite_partially_linear; < 0.4 ->
     rewrite_nonlinear
   C-INT: dR2_int = R2_inter - R2_full >= 0.05 ->
     multiplicative_component_present else
     additive_dominant
  CAUTION (3107): unembed and embedding are
  different matrices; e_yes measures the DIRECT
  embedding-side injection only, not the deep
  rewrite.  C failing does NOT mean no rewrite.

Overall verdict = A | B | C-LIN.
"""
import io
import json
import os
import time
import zlib
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p117_oscillation_attribution_' \
       'writemap_linearity'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D16 = os.path.join(RDIR, 'phase3116',
                   'omega_p114_full_mlp_sweep_decouple')
D18 = os.path.join(RDIR, 'phase3118',
                   'omega_p116_autoregressive_margin_'
                   'trajectory')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3119', NAME)
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
res16 = json.load(io.open(
    os.path.join(D16, 'result.json'), encoding='utf-8'))
assert res16['verdict'] == \
    'interaction_dominant|no_behavioral_decoupling'
res18 = json.load(io.open(
    os.path.join(D18, 'result.json'), encoding='utf-8'))
assert res18['verdict'] == \
    'belief_decays_in_generation|' \
    'temporal_compensation|' \
    'top_ablation_changes_behavior'
TOP3 = [int(t['layer'])
        for t in res16['top3_causal']]
assert TOP3 == [26, 33, 31]
N_NEW = 12
N_MATCH = 8
NP_B = 24 if SMOKE else 672
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

LAYERS = [26] + [L for L in range(14, 36)
                 if L != 26]
assert len(LAYERS) == 22

seal = {
    'phase': 3119,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'top3': TOP3,
    'n_new': N_NEW,
    'layers_scan': list(range(14, 36)),
    'data_sources': {
        'trajectories_and_tokens':
            'phase3118/traj_readout.npz '
            '(gt_cleanseq_clean__P/A1, gen_clean__P/'
            'A1, auc_curve) - bit-exact reuse',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz '
                    '(same rebuild as 3114-3118)'},
    'token_families': {
        'yes': ['yes', 'Yes', ' yes', ' Yes',
                'YES'],
        'no': ['no', 'No', ' no', ' No', 'NO']},
    'readout': 'm(t) = norm(h_NL[pos0+t]) . '
               '(w_yes - w_no); part B replay: ONE '
               'teacher-forced forward per (sequence, '
               'layer), identical code path to 3118',
    'part_c_spec': {
        'features': ['1', 'm(t-1)', 'e_yes(g_t)',
                     'is_P'],
        'interaction_model': 'adds m(t-1)*e_yes',
        'split': 'pair parity: odd j train, even j '
                 'test (deterministic)',
        'e_yes': 'emb(g_t) . (WU[yes_id] - '
                 'WU[no_id]), float32',
        'n_samples': 16128},
    'gates': {
        'A_sem_yes': 'per direction deciles of '
                     'm(t-1); valid bin = n_yes>=30 '
                     'and n_other>=30; pass if >=8 '
                     'valid bins AND >=70pct of valid '
                     'bins have mean dm(yes) > mean '
                     'dm(other) in BOTH P and A1 -> '
                     'yes_semantic_confirmed',
        'A_sem_no': 'same with mean dm(no) < mean '
                    'dm(other) -> '
                    'no_semantic_confirmed; <8 valid '
                    'bins -> not_evaluable',
        'A_sel': 'AUC(m(t-1)|g_t=yes vs other) per '
                 'direction; mean(P,A1) >=0.65 -> '
                 'self_selection_present; <=0.55 '
                 'self_selection_absent; else partial',
        'A_gap': 'pooled mean dGap(yes,other) > 0 '
                 'AND mean dGap(other,yes) < 0 AND '
                 'per-step sign consistency (steps '
                 'with n>=30; >=70pct positive for '
                 '(yes,other), >=70pct negative for '
                 '(other,yes); <5 valid steps -> '
                 'consistency skipped) -> '
                 'gap_attribution_confirmed',
        'A_verdict': 'yes_sem AND (no_sem OR '
                     'not_evaluable) AND gap -> '
                     'attribution_semantic_confirmed;'
                     ' elif NOT yes_sem AND sel '
                     'present -> '
                     'attribution_selection_dominant;'
                     ' else attribution_unresolved',
        'B_ovl': 'L26/L33/L31 replay trajectories '
                 'bit-exact vs 3118 npz (max abs '
                 'diff == 0.0) -> '
                 'replay_integrity_confirmed; else '
                 'abort -> replay_integrity_fail',
        'B_map': 'TOP3 all ratio<0.7 AND >=1 '
                 'non-TOP3 ratio>1.1 -> '
                 'compensation_layer_specific; TOP3 '
                 'all <0.7 AND none >1.1 -> '
                 'compensation_global; else '
                 'compensation_mixed',
        'B_l35': 'ratio_L35 > 1.0 -> '
                 'rebuild_layer_confirmed else '
                 'rebuild_layer_absent',
        'C_tok': 'dR2 = R2_full - R2_base >=0.10 -> '
                 'token_injection_tracked; <=0.02 -> '
                 'token_injection_untracked; else '
                 'token_injection_partial',
        'C_lin': 'R2_full held-out: >=0.7 -> '
                 'rewrite_mostly_linear; 0.4-0.7 -> '
                 'rewrite_partially_linear; <0.4 -> '
                 'rewrite_nonlinear',
        'C_int': 'dR2_int = R2_inter - R2_full '
                 '>=0.05 -> '
                 'multiplicative_component_present '
                 'else additive_dominant'},
    'note': 'AUC(t) is a monotone functional of the '
            'per-pair gap distribution, so gap '
            'attribution IS AUC attribution; '
            'self-selection (m(t-1) causes yes '
            'generation) is controlled by decile '
            'bins, not assumed away',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3119 Omega-P117 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed; TOP3=%s layers=L14..L35 '
    'NP_B=%d n_new=%d'
    % (TOP3, NP_B, N_NEW))

# ================================================================
# rebuilt records (identical to 3113 B / 3114-3118)
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
assert NP_ == 672 or SMOKE
if SMOKE:
    NP_B = min(NP_B, NP_)
log('records rebuilt: %d (%d pairs; part B '
    'uses %d)' % (NB, NP_, NP_B))

# ================================================================
# PART A: token attribution (offline)
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
log('token families: yes=%s no=%s'
    % (YES_FAMILY, NO_FAMILY))
YF = np.array(YES_FAMILY)
NF = np.array(NO_FAMILY)
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])

mP = z18['gt_cleanseq_clean__P']   # (672,13) f32
mA = z18['gt_cleanseq_clean__A1']
gP = z18['gen_clean__P']           # (672,12) i32
gA = z18['gen_clean__A1']
assert mP.shape == (672, 13)
assert gP.shape == (672, 12)


def auc_mw(pos_vals, neg_vals):
    """Mann-Whitney AUC with average ranks for
    ties (no scipy) - identical to 3118."""
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


def classify(g):
    c = np.zeros(g.shape, dtype=np.int8)
    c[np.isin(g, NF)] = 2
    c[np.isin(g, YF)] = 1
    return c


cP = classify(gP)   # 1=yes 2=no 0=other
cA = classify(gA)
auc_rc = [auc_mw(mP[:, t].astype(np.float64),
                 mA[:, t].astype(np.float64))
          for t in range(13)]
sc_curve = max(abs(a - float(b))
               / max(abs(float(b)), 1e-12)
               for (a, b) in zip(auc_rc, auc18))
assert sc_curve < 1e-6, \
    'AUC curve does not reproduce 3118'
log('consistency: AUC curve reproduced from npz, '
    'max rel diff %.2e' % sc_curve)

# --- step table ---
step_tab = []
for t in range(1, 13):
    row = {
        't': t,
        'yes_frac_P': float((cP[:, t - 1] == 1)
                            .mean()),
        'yes_frac_A1': float((cA[:, t - 1] == 1)
                             .mean()),
        'no_frac_P': float((cP[:, t - 1] == 2)
                           .mean()),
        'no_frac_A1': float((cA[:, t - 1] == 2)
                            .mean()),
        'auc_t': float(auc_rc[t]),
        'mean_gap_t': float(
            (mP[:, t] - mA[:, t])
            .astype(np.float64).mean())}
    step_tab.append(row)
log('STEP TABLE (t, yesP, yesA1, noP, noA1, '
    'AUC, gap):')
for row in step_tab:
    log('  t=%2d yesP=%.3f yesA1=%.3f noP=%.3f '
        'noA1=%.3f AUC=%.3f gap=%+.3f'
        % (row['t'], row['yes_frac_P'],
           row['yes_frac_A1'], row['no_frac_P'],
           row['no_frac_A1'], row['auc_t'],
           row['mean_gap_t']))

# --- pooled dm per class, per direction ---
part_a = {}
dir_data = {}
for (dname, mD, cD) in (('P', mP, cP),
                        ('A1', mA, cA)):
    dm = (mD[:, 1:] - mD[:, :-1]) \
        .astype(np.float64)
    x = mD[:, :-1].astype(np.float64).ravel()
    d = dm.ravel()
    c = cD.ravel()
    cls_stats = {}
    for (cn, cv) in (('yes', 1), ('no', 2),
                     ('other', 0)):
        sel = (c == cv)
        cls_stats[cn] = {
            'n': int(sel.sum()),
            'mean_dm': float(d[sel].mean())
            if sel.any() else None,
            'std_dm': float(d[sel].std())
            if sel.any() else None}
    sel_y = (c == 1)
    sel_o = (c == 0)
    if sel_y.any() and sel_o.any():
        auc_sel = auc_mw(x[sel_y], x[sel_o])
    else:
        auc_sel = float('nan')
    edges = np.quantile(x, np.arange(1, 10) / 10.0)
    bins = np.searchsorted(edges, x, side='right')
    bin_rows = []
    for b in range(10):
        s = (bins == b)
        ny = int(((c == 1) & s).sum())
        nn = int(((c == 2) & s).sum())
        no_ = int(((c == 0) & s).sum())
        dy = (float(d[(c == 1) & s].mean())
              if ny > 0 else None)
        dn = (float(d[(c == 2) & s].mean())
              if nn > 0 else None)
        do = (float(d[(c == 0) & s].mean())
              if no_ > 0 else None)
        bin_rows.append({'bin': b, 'n_yes': ny,
                         'n_no': nn, 'n_other': no_,
                         'dm_yes': dy, 'dm_no': dn,
                         'dm_other': do})
    dir_data[dname] = {
        'dm': dm, 'x': x, 'd': d, 'c': c,
        'cls_stats': cls_stats,
        'auc_sel': auc_sel,
        'bin_rows': bin_rows}
    log('A %s: dm yes=%s no=%s other=%s | '
        'AUC(m(t-1)|yes vs other)=%.4f'
        % (dname,
           ('%+.4f n=%d' % (cls_stats['yes']
                            ['mean_dm'],
                            cls_stats['yes']['n']))
           if cls_stats['yes']['mean_dm']
           is not None else 'n=0',
           ('%+.4f n=%d' % (cls_stats['no']
                            ['mean_dm'],
                            cls_stats['no']['n']))
           if cls_stats['no']['mean_dm']
           is not None else 'n=0',
           ('%+.4f n=%d' % (cls_stats['other']
                            ['mean_dm'],
                            cls_stats['other']
                            ['n'])),
           auc_sel))

# decile gate per direction
def decile_gate(bin_rows, want_yes):
    """valid bin: n_yes>=30 & n_other>=30 (for
    yes test) / n_no>=30 & n_other>=30 (no test);
    returns (n_valid, n_pass, pass_rate)."""
    nv = 0
    npass = 0
    for row in bin_rows:
        if want_yes:
            n1 = row['n_yes']
        else:
            n1 = row['n_no']
        if row['n_other'] < 30 or n1 < 30:
            continue
        if row['dm_other'] is None:
            continue
        if want_yes:
            diff = row['dm_yes'] - row['dm_other']
        else:
            diff = row['dm_no'] - row['dm_other']
        if diff is None:
            continue
        nv += 1
        if want_yes and diff > 0:
            npass += 1
        if (not want_yes) and diff < 0:
            npass += 1
    return nv, npass, (npass / float(nv)
                       if nv else None)


sem_yes = {}
sem_no = {}
for dname in ('P', 'A1'):
    nv, npp, pr = decile_gate(
        dir_data[dname]['bin_rows'], True)
    sem_yes[dname] = {'n_valid': nv,
                      'n_pass': npp,
                      'pass_rate': pr}
    nv, npp, pr = decile_gate(
        dir_data[dname]['bin_rows'], False)
    sem_no[dname] = {'n_valid': nv,
                     'n_pass': npp,
                     'pass_rate': pr}
yes_ok = all(
    sem_yes[d]['n_valid'] >= 8
    and sem_yes[d]['pass_rate'] >= 0.7
    for d in ('P', 'A1'))
no_evaluable = all(
    sem_no[d]['n_valid'] >= 8
    for d in ('P', 'A1'))
no_ok = no_evaluable and all(
    sem_no[d]['pass_rate'] >= 0.7
    for d in ('P', 'A1'))
log('A-SEM yes: P %d/%d bins A1 %d/%d bins -> %s'
    % (sem_yes['P']['n_pass'],
       sem_yes['P']['n_valid'],
       sem_yes['A1']['n_pass'],
       sem_yes['A1']['n_valid'],
       'yes_semantic_confirmed' if yes_ok
       else 'yes_semantic_rejected'))
if not no_evaluable:
    log('A-SEM no: not_evaluable (bins with '
        'n_no>=30 & n_other>=30 fewer than 8 in '
        'some direction)')
else:
    log('A-SEM no: P %d/%d bins A1 %d/%d bins -> %s'
        % (sem_no['P']['n_pass'],
           sem_no['P']['n_valid'],
           sem_no['A1']['n_pass'],
           sem_no['A1']['n_valid'],
           'no_semantic_confirmed' if no_ok
           else 'no_semantic_rejected'))

sel_mean = (dir_data['P']['auc_sel']
            + dir_data['A1']['auc_sel']) / 2.0
if sel_mean >= 0.65:
    sel_v = 'self_selection_present'
elif sel_mean <= 0.55:
    sel_v = 'self_selection_absent'
else:
    sel_v = 'self_selection_partial'
log('A-SEL: P=%.4f A1=%.4f mean=%.4f -> %s'
    % (dir_data['P']['auc_sel'],
       dir_data['A1']['auc_sel'], sel_mean, sel_v))

# --- gap attribution ---
gap = (mP - mA)
dgap = (gap[:, 1:] - gap[:, :-1]) \
    .astype(np.float64)
combo_tab = {}
for a_cls in (1, 2, 0):
    for b_cls in (1, 2, 0):
        mask = (cP == a_cls) & (cA == b_cls)
        n = int(mask.sum())
        combo_tab['%d_%d' % (a_cls, b_cls)] = {
            'n': n,
            'mean_dgap': float(dgap[mask].mean())
            if n else None}
key_yo = combo_tab['1_0']
key_oy = combo_tab['0_1']
cons_yo = None
cons_oy = None
nval_yo = 0
npos_yo = 0
nval_oy = 0
nneg_oy = 0
for t in range(12):
    s_yo = (cP[:, t] == 1) & (cA[:, t] == 0)
    if int(s_yo.sum()) >= 30:
        nval_yo += 1
        if float(dgap[s_yo, t].mean()) > 0:
            npos_yo += 1
    s_oy = (cP[:, t] == 0) & (cA[:, t] == 1)
    if int(s_oy.sum()) >= 30:
        nval_oy += 1
        if float(dgap[s_oy, t].mean()) < 0:
            nneg_oy += 1
if nval_yo >= 5:
    cons_yo = npos_yo / float(nval_yo)
if nval_oy >= 5:
    cons_oy = nneg_oy / float(nval_oy)
gap_ok = (key_yo['mean_dgap'] is not None
          and key_oy['mean_dgap'] is not None
          and key_yo['mean_dgap'] > 0
          and key_oy['mean_dgap'] < 0)
if gap_ok:
    if cons_yo is not None and cons_yo < 0.7:
        gap_ok = False
    if cons_oy is not None and cons_oy < 0.7:
        gap_ok = False
log('A-GAP: (yes,other) mean=%s n=%d; '
    '(other,yes) mean=%s n=%d; consistency '
    'yo=%s (%d/%d) oy=%s (%d/%d) -> %s'
    % (('%+.4f' % key_yo['mean_dgap'])
       if key_yo['mean_dgap'] is not None
       else 'nan', key_yo['n'],
       ('%+.4f' % key_oy['mean_dgap'])
       if key_oy['mean_dgap'] is not None
       else 'nan', key_oy['n'],
       ('%.2f' % cons_yo) if cons_yo is not None
       else 'skip', npos_yo, nval_yo,
       ('%.2f' % cons_oy) if cons_oy is not None
       else 'skip', nneg_oy, nval_oy,
       'gap_attribution_confirmed' if gap_ok
       else 'gap_attribution_rejected'))

if yes_ok and (no_ok or (not no_evaluable)) \
        and gap_ok:
    verdict_a = 'attribution_semantic_confirmed'
elif (not yes_ok) and sel_v == \
        'self_selection_present':
    verdict_a = 'attribution_selection_dominant'
else:
    verdict_a = 'attribution_unresolved'
log('PART A VERDICT: %s' % verdict_a)

part_a = {
    'verdict': verdict_a,
    'yes_family': YES_FAMILY,
    'no_family': NO_FAMILY,
    'step_table': step_tab,
    'cls_stats': {d: dir_data[d]['cls_stats']
                  for d in ('P', 'A1')},
    'sel_auc': {'P': dir_data['P']['auc_sel'],
                'A1': dir_data['A1']['auc_sel'],
                'mean': sel_mean,
                'verdict': sel_v},
    'decile_yes': sem_yes,
    'decile_no': sem_no,
    'no_evaluable': no_evaluable,
    'yes_semantic_confirmed': yes_ok,
    'no_semantic_confirmed': no_ok
    if no_evaluable else None,
    'combo_table': combo_tab,
    'gap_consistency': {
        'yes_other': {'n_val': nval_yo,
                      'n_pos': npos_yo,
                      'rate': cons_yo},
        'other_yes': {'n_val': nval_oy,
                      'n_neg': nneg_oy,
                      'rate': cons_oy}},
    'gap_attribution_confirmed': gap_ok,
    'auc_curve_recomputed': auc_rc,
    'selfcheck_curve_rel_max': sc_curve,
}

# ================================================================
# model + hooks
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


for L in LAYERS:
    model.model.layers[L].mlp \
        .register_forward_hook(mk_mlp_post(L))


def forward_track(prompt_ids, gen_tokens,
                  layers_abl):
    """One teacher-forced forward over
    [prompt, gen]; returns m(t) for t=0..n_new
    (13 points) and greedy-recheck rate.
    Identical to 3118."""
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
# PART B: time x layer write map (GPU)
# ================================================================
cleanP = z18['gt_cleanseq_clean__P'][:NP_B]
cleanA = z18['gt_cleanseq_clean__A1'][:NP_B]
o18P = {'L26': z18['gt_cleanseq_L26__P'][:NP_B],
        'L33': z18['gt_cleanseq_L33__P'][:NP_B],
        'L31': z18['gt_cleanseq_L31__P'][:NP_B]}
o18A = {'L26': z18['gt_cleanseq_L26__A1'][:NP_B],
        'L33': z18['gt_cleanseq_L33__A1'][:NP_B],
        'L31': z18['gt_cleanseq_L31__A1'][:NP_B]}
wmP = {}
wmA = {}
rechk_b = {}
overlap = {}
integrity = True
t0b = time.time()
for L in LAYERS:
    gtP = np.zeros((NP_B, N_NEW + 1),
                   dtype=np.float32)
    gtA = np.zeros((NP_B, N_NEW + 1),
                   dtype=np.float32)
    rc = []
    for j in range(NP_B):
        ms, r1 = forward_track(
            PID_P[j], gP[j], (L,))
        gtP[j] = ms
        rc.append(r1)
        ms, r2 = forward_track(
            PID_A[j], gA[j], (L,))
        gtA[j] = ms
        rc.append(r2)
    wmP[L] = gtP
    wmA[L] = gtA
    rechk_b[L] = float(np.mean(rc))
    key = 'L%d' % L
    if L in (26, 33, 31):
        dP = float(np.abs(gtP - o18P[key]).max())
        dA = float(np.abs(gtA - o18A[key]).max())
        overlap[key] = {'dP': dP, 'dA': dA}
        log('OVERLAP %s: dP=%.2e dA=%.2e'
            % (key, dP, dA))
        if dP > 0.0 or dA > 0.0:
            if SMOKE:
                log('INTEGRITY MISMATCH in SMOKE '
                    '(subset alignment); continuing '
                    '(path test only)')
            else:
                integrity = False
                log('INTEGRITY FAIL at %s - '
                    'aborting remaining replays'
                    % key)
                break
    log('replay %s done (%.1fs) recheck=%.4f'
        % (key, time.time() - t0b, rechk_b[L]))

write_map = {}
if integrity:
    for L in LAYERS:
        dmP = (wmP[L] - cleanP)
        dmA = (wmA[L] - cleanA)
        dm = np.concatenate([dmP, dmA], axis=0) \
            .astype(np.float64)
        head = float(np.abs(dm[:, 0:3]).mean())
        tail = float(
            np.abs(dm[:, N_NEW - 3:N_NEW + 1])
            .mean())
        per_step = np.abs(dm).mean(axis=0)
        write_map['L%d' % L] = {
            'head_mean_abs': head,
            'tail_mean_abs': tail,
            'ratio': float(tail
                           / max(head, 1e-9)),
            'peak_t': int(per_step.argmax()),
            'per_step_mean_abs': [
                float(x) for x in per_step]}
        log('WMAP L%d: head=%.4f tail=%.4f '
            'ratio=%.3f peak_t=%d'
            % (L, head, tail,
               write_map['L%d' % L]['ratio'],
               write_map['L%d' % L]['peak_t']))
    top3_ok = all(write_map['L%d' % X]['ratio']
                  < 0.7 for X in TOP3)
    amp = [L for L in LAYERS
           if write_map['L%d' % L]['ratio'] > 1.1
           and L not in TOP3]
    amp_all = [L for L in LAYERS
               if write_map['L%d' % L]['ratio']
               > 1.1]
    shr = [L for L in LAYERS
           if write_map['L%d' % L]['ratio'] < 0.7]
    log('WMAP CLASS: shrinkers(<0.7)=%s '
        'amplifiers(>1.1)=%s (non-TOP3 amps=%s)'
        % (shr, amp_all, amp))
    r35 = write_map['L35']['ratio']
    l35_v = ('rebuild_layer_confirmed'
             if r35 > 1.0
             else 'rebuild_layer_absent')
    log('L35: ratio=%.3f -> %s' % (r35, l35_v))
    if not top3_ok:
        verdict_b = 'compensation_mixed'
    elif top3_ok and amp:
        verdict_b = 'compensation_layer_specific'
    elif top3_ok and not amp_all:
        verdict_b = 'compensation_global'
    else:
        verdict_b = 'compensation_mixed'
    log('PART B VERDICT: %s' % verdict_b)
else:
    verdict_b = 'replay_integrity_fail'
    l35_v = 'not_evaluable'
    amp = []
    shr = []
    log('PART B VERDICT: %s' % verdict_b)

part_b = {
    'verdict': verdict_b,
    'integrity_confirmed': integrity,
    'overlap': overlap,
    'recheck': {'L%d' % L: rechk_b[L]
                for L in wmP},
    'write_map': write_map,
    'shrinkers': shr,
    'amplifiers': amp_all,
    'amplifiers_non_top3': amp,
    'l35_verdict': l35_v,
    'n_pairs_replay': NP_B,
}

# ================================================================
# PART C: one-step predictability (linearity)
# ================================================================
emb = model.model.embed_tokens.weight
VOC = emb.shape[0]
w_t = torch.tensor(w_dn, device='cuda',
                   dtype=torch.float32)
e_all = np.zeros(int(VOC), dtype=np.float32)
with torch.inference_mode():
    for s0 in range(0, int(VOC), 32768):
        e_all[s0:s0 + 32768] = (
            emb[s0:s0 + 32768].float() @ w_t) \
            .cpu().numpy()
log('e_all computed: e_yes(yes_id)=%+.4f '
    'e_yes(no_id)=%+.4f'
    % (float(e_all[YES_ID]), float(e_all[NO_ID])))

X1 = []
X2 = []
YV = []
ISP = []
JJ = []
TT = []
for (dname, mD, gD) in (('P', mP, gP),
                        ('A1', mA, gA)):
    isp = 1.0 if dname == 'P' else 0.0
    for j in range(672):
        for t in range(1, N_NEW + 1):
            X1.append(float(mD[j, t - 1]))
            X2.append(float(e_all[int(gD[j, t - 1])]))
            YV.append(float(mD[j, t]))
            ISP.append(isp)
            JJ.append(j)
            TT.append(t)
X1 = np.array(X1, dtype=np.float64)
X2 = np.array(X2, dtype=np.float64)
YV = np.array(YV, dtype=np.float64)
ISP = np.array(ISP, dtype=np.float64)
JJ = np.array(JJ)
TRN = (JJ % 2 == 1)
TST = ~TRN
assert TRN.sum() == 8064 and TST.sum() == 8064


def ols_r2(cols, trn, tst):
    Xtr = np.column_stack(
        [np.ones(trn.sum())]
        + [c[trn] for c in cols])
    ytr = YV[trn]
    th, _, _, _ = np.linalg.lstsq(
        Xtr, ytr, rcond=None)
    Xte = np.column_stack(
        [np.ones(tst.sum())]
        + [c[tst] for c in cols])
    pred = Xte @ th
    yte = YV[tst]
    sse = float(((yte - pred) ** 2).sum())
    sst = float(((yte - yte.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst
    return r2, pred


one = np.ones(len(YV), dtype=np.float64)
r2_persist = 1.0 - float(
    ((YV[TST] - X1[TST]) ** 2).sum()) \
    / float(((YV[TST] - YV[TST].mean()) ** 2).sum())
r2_base, _ = ols_r2([X1, ISP], TRN, TST)
r2_full, pred_full = ols_r2([X1, X2, ISP],
                            TRN, TST)
r2_int, _ = ols_r2([X1, X2, X1 * X2, ISP],
                   TRN, TST)
d_tok = r2_full - r2_base
d_int = r2_int - r2_full


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


sp = float(np.corrcoef(
    rankdata(pred_full),
    rankdata(YV[TST]))[0, 1])
if d_tok >= 0.10:
    tok_v = 'token_injection_tracked'
elif d_tok <= 0.02:
    tok_v = 'token_injection_untracked'
else:
    tok_v = 'token_injection_partial'
if r2_full >= 0.7:
    lin_v = 'rewrite_mostly_linear'
elif r2_full >= 0.4:
    lin_v = 'rewrite_partially_linear'
else:
    lin_v = 'rewrite_nonlinear'
int_v = ('multiplicative_component_present'
         if d_int >= 0.05
         else 'additive_dominant')
log('PART C: R2 persist=%.4f base=%.4f '
    'full=%.4f inter=%.4f | d_tok=%+.4f '
    'd_int=%+.4f spearman=%.4f'
    % (r2_persist, r2_base, r2_full, r2_int,
       d_tok, d_int, sp))
log('PART C VERDICT: %s | %s | %s'
    % (lin_v, tok_v, int_v))

ey_yes = float(e_all[YF].mean())
ey_no = float(e_all[NF].mean())
other_mask = (cP == 0)
ey_other = float(
    e_all[gP[other_mask]].mean())
log('e_yes class means: yes=%+.4f no=%+.4f '
    'other=%+.4f' % (ey_yes, ey_no, ey_other))

part_c = {
    'verdict_lin': lin_v,
    'verdict_tok': tok_v,
    'verdict_int': int_v,
    'r2_persist': r2_persist,
    'r2_base': r2_base,
    'r2_full': r2_full,
    'r2_inter': r2_int,
    'd_r2_token': d_tok,
    'd_r2_int': d_int,
    'spearman_pred_true': sp,
    'n_train': int(TRN.sum()),
    'n_test': int(TST.sum()),
    'e_yes_class_means': {'yes': ey_yes,
                          'no': ey_no,
                          'other': ey_other},
}

# ================================================================
# save
# ================================================================
avail = [L for L in LAYERS if L in wmP]
npz_out = {'layers': np.array(avail,
                              dtype=np.int64),
           'wmP': np.stack(
               [wmP[L] for L in avail], 0),
           'wmA': np.stack(
               [wmA[L] for L in avail], 0),
           'cleanP_ref': cleanP,
           'cleanA_ref': cleanA}
if integrity:
    npz_out['wmap_per_step'] = np.stack(
        [np.array(write_map['L%d' % L]
                  ['per_step_mean_abs'],
                 dtype=np.float64)
         for L in LAYERS], 0)
np.savez(os.path.join(OUT, 'wmap_readout.npz'),
         **npz_out)

verdict_full = '%s|%s|%s' % (verdict_a, verdict_b,
                             lin_v)
results = {
    'verdict': verdict_full,
    'top3': TOP3,
    'layers_scan': LAYERS,
    'n_records': NB,
    'n_pairs': NP_,
    'n_pairs_partb': NP_B,
    'n_new': N_NEW,
    'smoke': SMOKE,
    'part_a': part_a,
    'part_b': part_b,
    'part_c': part_c,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('VERDICT: %s' % verdict_full)
log('result.json written')
log('Phase 3119 done (%.1fs)'
    % (time.time() - T0))
print('VERDICT: %s' % verdict_full)
