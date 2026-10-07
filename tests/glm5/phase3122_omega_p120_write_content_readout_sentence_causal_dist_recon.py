# -*- coding: utf-8 -*-
"""Phase 3122 (Omega-P120): write-content readout
(direction-split MLP write projection spectrum)
+ sentence-level coherent replacement causality
+ distribution-level reconstruction.

T4 fifth phase.  Frozen inputs: 3118 traj_readout
(gen + ablation teacher-forced margins), 3120
p118_readout (ann codes), 3121 p119_readout
(c0 margins for oscillation cross-check), 3113
capture_b, 3105 material.json.

Parts:
  A-GPU  all-layer MLP write projection (p_dn /
         p_fam / norm per layer per step, both
         dirs) on clean teacher-forced replay;
         sanity gate L30/L32 p_dn < 0.
  A-off  ablation teacher-forced margin direction
         split from 3118 npz (gt_cleanseq_L{26,31,33}
         - gt_cleanseq_clean), no GPU.
  B-GPU  sentence-level replacement inside fact
         spans: s0 replay / s1 same-predicate
         entity-swapped sentence / s2 shuffled s1 /
         s3 '.' fill (equal-length, n_pad logged).
         Gates B2-CONT / B2-SYN / B2-OSC.
  C-off  distribution-level reconstruction:
         empirical-residual bootstrap MC (200
         reps, seed 3122), simulated AUC curve vs
         3118 auc_curve + PIT calibration.
"""
import hashlib
import io
import json
import os
import random as _rnd
import re
import time
import zlib

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p120_write_content_readout_' \
       'sentence_causal_dist_recon'
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
D21 = os.path.join(RDIR, 'phase3121',
                   'omega_p119_repl_causality_erase_'
                   'polarity_recon')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3122', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


N_NEW = 12
NP_A = 8 if SMOKE else 672
NP_G = 8 if SMOKE else 672

# ================================================================
# frozen inputs + integrity assertions
# ================================================================
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13, 'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
NB = len(pkB)
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
res21 = json.load(io.open(
    os.path.join(D21, 'result.json'),
    encoding='utf-8'))
assert res21['verdict'] == \
    'content_nonspecific|syntax_polarity_absent|' \
    'erase_joint_behavioral|readout_robust|' \
    'reconstruction_failed|curve_shape_weak|' \
    'coverage_ok|position_independent_confirmed'
assert res21['smoke'] is False
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
z21 = np.load(os.path.join(D21, 'p119_readout.npz'),
              allow_pickle=True)
SLOPE_3120 = float(
    res20['part_c']['primary']['slope_lin'])
MSTAR_3120 = float(
    res20['part_c']['fixed_point_raw_lin'])
assert abs(SLOPE_3120
           - (-0.6801486439276976)) < 1e-12
assert abs(MSTAR_3120
           - (-6.052991245322225)) < 1e-12

# ================================================================
# design seal (frozen BEFORE observation)
# ================================================================
seal = {
    'phase': 3122,
    'name': NAME,
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'np_a': NP_A,
    'data_sources': {
        'tokens_and_margins':
            'phase3118 traj_readout.npz (gen_clean, '
            'gt_cleanseq_{clean,L26,L31,L33} margins)',
        'annotation': 'phase3120 p118_readout.npz '
                      'annP_c/annA_c (0 other, 1 '
                      'syntax, 2 query_rest, 3 '
                      'fact_shaped, 4 fact_strict, '
                      '9 answer; fact span = code in '
                      '{3,4}, longest contiguous, '
                      'cols 1..11)',
        'osc_cross_check': 'phase3121 p119_readout.npz '
                           'pa_c1_dn__P/A1 (token-level '
                           'other-fact fill)',
        'operator_params': 'phase3120 result.json '
                           'primary slope_lin '
                           '-0.6801486439276976 + '
                           'fixed_point_raw_lin '
                           '-6.052991245322225',
        'material': 'phase3105 material.json + '
                    'phase3113 capture_b.npz'},
    'conditions_part_b': [
        's0_replay', 's1_entity_swap_sentence',
        's2_shuffled_s1', 's3_punct_fill'],
    'replacement_rule': 's1 keeps the PREDICATE of '
                        'the replaced line, swaps '
                        'BOTH entities to a random '
                        'unused (s2,o2) pair with '
                        's2!=s, o2!=o, (s2,lrel,o2) '
                        'not among the 8 context '
                        'lines; token sequence '
                        "tok(' The s2 lrel the o2.') "
                        'truncated or dot-padded to '
                        'the span length L (n_pad '
                        'logged); s2 = s1 tokens '
                        'shuffled (seed crc32); '
                        's3 = dot x L',
    'readouts': {
        'w_dn': 'WU[YES_ID] - WU[NO_ID]',
        'w_fam': 'mean(WU[yes_family]) - '
                 'mean(WU[no_family])'},
    'mc': {'n_reps': 200 if not SMOKE else 5,
           'seed': 3122,
           'noise': 'empirical residual bootstrap '
                    '(with replacement) from 3121 '
                    'residual definition'},
    'gates': {
        'A_repro': 'max|s0-3118| == 0 -> '
                   'replay_bit_exact; < 1e-6 -> '
                   'replay_ok; else diverged '
                   '(non-fatal, s0 baseline)',
        'W_san': 'mean p_dn over steps/pairs of L30 '
                 'AND L32 < 0 (pooled dirs) -> '
                 'write_polarity_replicated; else '
                 'write_polarity_diverged',
        'A1_dir': 'a1_share = sum|dm_A1| / '
                  '(sum|dm_P| + sum|dm_A1|) over '
                  'L26/L31/L33 pooled: >= 0.6 -> '
                  'ablation_margin_a1_dominant; '
                  '0.4..0.6 -> mixed; < 0.4 -> '
                  'p_dominant',
        'B2_cont_P': 'E_cont = D_s1 - D_s3 >= +0.05 '
                     '-> sentence_content_pull_up; '
                     '<= -0.05 -> '
                     'sentence_content_push_down; '
                     'else sentence_content_neutral',
        'B2_cont_A1': 'same with +/-0.025',
        'B2_syn_P': '|E_syn| = |D_s1 - D_s2| >= '
                    '0.05 -> syntax_effect_present '
                    '(sign logged); else '
                    'syntax_effect_absent',
        'B2_syn_A1': 'same with 0.025',
        'B2_osc': 'r_osc = mean|diff_s1[t>k2]| / '
                  'mean|diff_c1_3121[t>k2]|: <= 0.7 '
                  '-> oscillation_reduced; >= 1.0 '
                  '-> oscillation_persist; else '
                  'oscillation_partial',
        'B2_pad_sensitivity': 'E_cont recomputed on '
                              'n_pad==0 & no-trunc '
                              'subsample; direction '
                              'flip reported (not a '
                              'gate)',
        'C_dist': 'corr(auc_sim, auc18) >= 0.7 -> '
                  'distribution_reconstruction_'
                  'confirmed; >= 0.5 -> '
                  'distribution_reconstruction_'
                  'partial; else '
                  'distribution_reconstruction_'
                  'failed',
        'C_pit': 'KS(u, uniform) < 0.05 -> '
                 'pit_calibrated; >= 0.15 -> '
                 'pit_miscalibrated; else '
                 'pit_marginal'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('seal frozen (%s)' % seal['created'])

# ================================================================
# prompt rebuild (identical to 3114-3121)
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
log('records rebuilt: %d (%d pairs); NP_A=%d'
    % (NB, NP_, NP_A))

# ================================================================
# GPU load
# ================================================================
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

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

# hooks: two modes
ABL = {'mode': None, 'layers': set()}
WREC = {'on': False, 'store': {}}


def mk_mlp_hook(L):
    def hook(mod, mod_in, out):
        if ABL['mode'] == 'mlp' \
                and L in ABL['layers']:
            return torch.zeros_like(out)
        if WREC['on'] and L not in WREC['store']:
            WREC['store'][L] = \
                out.detach().float().cpu()
        return None
    return hook


for L in range(NL):
    model.model.layers[L].mlp \
        .register_forward_hook(mk_mlp_hook(L))

w_dn_t = torch.tensor(w_dn, device='cpu')
w_fam_t = torch.tensor(w_fam, device='cpu')


def gen_greedy(prompt_ids, n_new, layers_abl):
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl) if layers_abl else set()
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
    """teacher-forced forward; per-step margins
    under BOTH readouts (3121-identical)."""
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl) if layers_abl else set()
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


def forward_wrec(prompt_ids, gen_tokens):
    """teacher-forced forward; margins + all-layer
    MLP write projections (per layer per step)."""
    ABL['mode'] = None
    ABL['layers'] = set()
    WREC['on'] = True
    WREC['store'] = {}
    gen_tokens = [int(x) for x in gen_tokens]
    ids = list(prompt_ids) + gen_tokens
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(gen_tokens) + 1
    ms_dn = np.zeros(npts, dtype=np.float64)
    with torch.inference_mode():
        out = model(t_in,
                    output_hidden_states=True,
                    use_cache=False)
        hs = out.hidden_states[NL][0]
        norm = model.model.norm
        h_all = norm(hs).float().cpu().numpy()
        for k in range(npts):
            ms_dn[k] = float(h_all[pos0 + k]
                             @ w_dn)
        del out
        stacked = torch.stack(
            [WREC['store'][L]
             for L in range(NL)]).squeeze(1)
        proj_dn = torch.einsum('lsh,h->ls',
                               stacked, w_dn_t)
        proj_fm = torch.einsum('lsh,h->ls',
                               stacked, w_fam_t)
        pn = stacked.norm(dim=-1)
        pd = proj_dn[:, pos0:pos0 + npts].numpy()
        pf = proj_fm[:, pos0:pos0 + npts].numpy()
        pn = pn[:, pos0:pos0 + npts].numpy()
    WREC['on'] = False
    WREC['store'] = {}
    return ms_dn, pd, pf, pn


PID_P = [tok(texts[hP[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]
PID_A = [tok(texts[hA1[p]],
             add_special_tokens=False)['input_ids']
         for p in pks]
gP18 = z18['gen_clean__P']
gA18 = z18['gen_clean__A1']

# ================================================================
# PART A-GPU: write-content projection spectrum
# ================================================================
log('== PART A-GPU: write-content readout ==')
wrec = {'P': {}, 'A1': {}}
for dcode, gen18, PID in (('P', gP18, PID_P),
                          ('A1', gA18, PID_A)):
    m_dn = np.zeros((NP_A, N_NEW + 1),
                    dtype=np.float64)
    p_dn = np.zeros((NL, NP_A, N_NEW + 1),
                    dtype=np.float32)
    p_fm = np.zeros((NL, NP_A, N_NEW + 1),
                    dtype=np.float32)
    p_nm = np.zeros((NL, NP_A, N_NEW + 1),
                    dtype=np.float32)
    t0a = time.time()
    for j in range(NP_A):
        prompt_ids = PID[j]
        base = [int(t) for t in gen18[j]]
        md, pd_, pf_, pn_ = forward_wrec(
            prompt_ids, base)
        m_dn[j] = md
        p_dn[:, j, :] = pd_
        p_fm[:, j, :] = pf_
        p_nm[:, j, :] = pn_
    wrec[dcode] = {'m_dn': m_dn, 'p_dn': p_dn,
                   'p_fm': p_fm, 'p_nm': p_nm}
    log('wrec %s done (%.1fs)'
        % (dcode, time.time() - t0a))

# replay repro: wrec m_dn vs 3118 gt_cleanseq_clean
mP18 = z18['gt_cleanseq_clean__P']
mA18 = z18['gt_cleanseq_clean__A1']
ref18 = {'P': mP18[:NP_A], 'A1': mA18[:NP_A]}
drep = 0.0
for dcode in ('P', 'A1'):
    drep = max(drep, float(np.abs(
        wrec[dcode]['m_dn']
        - ref18[dcode].astype(np.float64)).max()))
if drep == 0.0:
    repro_v = 'replay_bit_exact'
elif drep < 1e-6:
    repro_v = 'replay_ok'
else:
    repro_v = 'replay_diverged'
log('A-REPRO: max diff %.3e -> %s'
    % (drep, repro_v))

# write polarity sanity gate (L30/L32 pooled)
p30 = float(np.concatenate([
    wrec['P']['p_dn'][30].ravel(),
    wrec['A1']['p_dn'][30].ravel()]).mean())
p32 = float(np.concatenate([
    wrec['P']['p_dn'][32].ravel(),
    wrec['A1']['p_dn'][32].ravel()]).mean())
w_san = ('write_polarity_replicated'
         if (p30 < 0 and p32 < 0)
         else 'write_polarity_diverged')
log('W-SAN: p_dn L30=%.3f L32=%.3f -> %s'
    % (p30, p32, w_san))

# write spectrum table (per layer per dir)
spec = {}
for dcode in ('P', 'A1'):
    pd_ = wrec[dcode]['p_dn']
    spec[dcode] = {
        'mean_dn': [float(pd_[L].mean())
                    for L in range(NL)],
        'mean_fam': [float(wrec[dcode]['p_fm'][L]
                           .mean())
                     for L in range(NL)],
        'mean_norm': [float(wrec[dcode]['p_nm'][L]
                            .mean())
                      for L in range(NL)]}

# ================================================================
# PART A-offline: ablation margin direction split
# ================================================================
log('== PART A-offline: ablation margin split ==')
adirs = {}
sum_abs = {'P': 0.0, 'A1': 0.0}
for Lay in (26, 31, 33):
    for dc in ('P', 'A1'):
        dm = (z18['gt_cleanseq_L%d__%s' % (Lay, dc)]
              .astype(np.float64)
              - z18['gt_cleanseq_clean__%s' % dc]
              .astype(np.float64))
        sum_abs[dc] += float(np.abs(dm).sum())
        adirs['L%d_%s' % (Lay, dc)] = {
            'mean_total': float(dm.mean()),
            'mean_first': float(dm[:, 0].mean()),
            'mean_rest': float(dm[:, 1:].mean()),
            'per_step': [float(dm[:, t].mean())
                         for t in
                         range(N_NEW + 1)]}
a1_share = sum_abs['A1'] / (sum_abs['P']
                            + sum_abs['A1'])
a1_dir_v = ('ablation_margin_a1_dominant'
            if a1_share >= 0.6 else
            ('p_dominant' if a1_share < 0.4
             else 'mixed'))
log('A1-DIR: a1_share=%.4f -> %s'
    % (a1_share, a1_dir_v))

# ================================================================
# PART B-GPU: sentence-level replacement
# ================================================================
log('== PART B-GPU: sentence replacement ==')
DOT_ID = tok('.', add_special_tokens=False)
DOT_ID = int(DOT_ID['input_ids'][0])


def find_span(ann_row):
    best = None
    k1 = None
    for k in range(1, N_NEW):
        isf = int(ann_row[k]) in (3, 4)
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
    return best


# line-token cache for replacement sentences
_line_tok = {}


def line_tokens(s2, o2, lrel):
    key = (s2, o2, lrel)
    if key not in _line_tok:
        txt = ' The %s %s the %s.' % (
            ents_all[s2], PREDS_all[lrel],
            ents_all[o2])
        _line_tok[key] = list(tok(
            txt, add_special_tokens=False)
            ['input_ids'])
    return _line_tok[key]


def context_triples(pk, dcode):
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    if dcode == 'P':
        lrel = r
    else:
        lrel = ri1
    D = [tuple(d) for d in
         mat5['distractors']['%d_%d' % (s, o)]]
    ctx = set((a, rr, b) for (a, rr, b) in
              ([(s, lrel, o)] + list(D)))
    return s, o, lrel, ctx


def pick_replacement(pk, dcode):
    (s, o, lrel, ctx) = context_triples(
        pk, dcode)
    rng1 = _rnd.Random(zlib.crc32(
        ('s1|%s|%s' % (pk, dcode))
        .encode('ascii')))
    ents_n = len(ents_all)
    for _try in range(200):
        s2 = rng1.randrange(ents_n)
        o2 = rng1.randrange(ents_n)
        if s2 == s or o2 == o:
            continue
        if (s2, lrel, o2) in ctx:
            continue
        if s2 == o2:
            continue
        return s2, o2
    raise RuntimeError('no replacement found')


SCOND = ('s0', 's1', 's2', 's3')
SB = {c: {'P': None, 'A1': None} for c in SCOND}
pad_info = []
t0b = time.time()
for dcode, ann, gen18, PID in (
        ('P', annP_c, gP18, PID_P),
        ('A1', annA_c, gA18, PID_A)):
    m_dn = {c: [] for c in SCOND}
    m_fm = {c: [] for c in SCOND}
    n_done = 0
    for j in range(NP_A):
        pk = pks[j]
        prompt_ids = PID[j]
        base = [int(t) for t in gen18[j]]
        best = find_span(ann[j])
        toks = {c: list(base) for c in SCOND}
        if best is not None:
            (k1, k2) = best
            L = k2 - k1 + 1
            (s2, o2) = pick_replacement(
                pk, dcode)
            (_, _, lrel, _) = context_triples(
                pk, dcode)
            sub = line_tokens(s2, o2, lrel)
            n_trunc = max(0, len(sub) - L)
            n_pad = max(0, L - len(sub))
            sub = sub[:L]
            sub = sub + [DOT_ID] * n_pad
            rng2 = _rnd.Random(zlib.crc32(
                ('s2|%s|%s' % (pk, dcode))
                .encode('ascii')))
            shuf = list(sub)
            rng2.shuffle(shuf)
            toks['s1'][k1:k2 + 1] = sub
            toks['s2'][k1:k2 + 1] = shuf
            toks['s3'][k1:k2 + 1] = \
                [DOT_ID] * L
            pad_info.append(
                {'pk': pk, 'dir': dcode,
                 'len': L, 'n_pad': n_pad,
                 'n_trunc': n_trunc})
        for c in SCOND:
            md, mf = forward_track2(
                prompt_ids, toks[c], None)
            m_dn[c].append(md)
            m_fm[c].append(mf)
        n_done += 1
        if n_done % 128 == 0:
            log('B %s %d/%d (%.1fs)'
                % (dcode, n_done, NP_A,
                   time.time() - t0b))
    for c in SCOND:
        SB[c][dcode] = {
            'dn': np.array(m_dn[c],
                           dtype=np.float64),
            'fam': np.array(m_fm[c],
                            dtype=np.float64)}
    log('partB %s done (%.1fs)'
        % (dcode, time.time() - t0b))

# s0 replay repro
drep_b = 0.0
for dcode in ('P', 'A1'):
    ref = ref18[dcode].astype(np.float64)
    drep_b = max(drep_b, float(np.abs(
        SB['s0'][dcode]['dn'] - ref).max()))
log('B-REPRO: max diff %.3e' % drep_b)

# Part B effects (spans only)
b_eff = {}
for dcode in ('P', 'A1'):
    ann = annP_c if dcode == 'P' else annA_c
    Ds = {c: [] for c in ('s1', 's2', 's3')}
    lens = []
    osc_num = []
    osc_den = []
    for j in range(NP_A):
        best = find_span(ann[j])
        if best is None:
            continue
        (k1, k2) = best
        lens.append(k2 - k1 + 1)
        base = ref18[dcode][j].astype(np.float64)
        diffs = {}
        for c in SCOND:
            m = SB[c][dcode]['dn'][j]
            diffs[c] = m.astype(np.float64) - base
        for c in ('s1', 's2', 's3'):
            Ds[c].append(float(
                diffs[c][k1 + 1:].mean()))
        osc_num.append(float(
            np.abs(diffs['s1'][k2 + 1:]).mean())
            if k2 + 1 < N_NEW + 1 else 0.0)
        mc1 = z21['pa_c1_dn__%s' % dcode][j] \
            .astype(np.float64)
        osc_den.append(float(
            np.abs((mc1 - base)[k2 + 1:]).mean())
            if k2 + 1 < N_NEW + 1 else 0.0)
    e_cont = float(np.mean(
        [a - b for (a, b) in
         zip(Ds['s1'], Ds['s3'])]))
    e_syn_pairs = [(a, b, ln) for (a, b, ln) in
                   zip(Ds['s1'], Ds['s2'], lens)
                   if ln >= 2]
    e_syn = float(np.mean(
        [a - b for (a, b, _) in e_syn_pairs]))
    Dm = {c: float(np.mean(Ds[c]))
          for c in ('s1', 's2', 's3')}
    num = float(np.sum(osc_num))
    den = float(np.sum(osc_den))
    r_osc = num / den if den > 0 else float('nan')
    b_eff[dcode] = {
        'n_span': len(lens),
        'n_len2': len(e_syn_pairs),
        'D_mean': Dm,
        'E_cont': e_cont,
        'E_syn': e_syn,
        'osc_num_mean': num / max(len(osc_num), 1),
        'osc_den_mean': den / max(len(osc_den), 1),
        'r_osc': r_osc}

th_c = {'P': 0.05, 'A1': 0.025}
b2_verdicts = {}
for dcode in ('P', 'A1'):
    e = b_eff[dcode]
    th = th_c[dcode]
    if e['E_cont'] >= th:
        vc = 'sentence_content_pull_up'
    elif e['E_cont'] <= -th:
        vc = 'sentence_content_push_down'
    else:
        vc = 'sentence_content_neutral'
    vs = ('syntax_effect_present'
          if abs(e['E_syn']) >= th
          else 'syntax_effect_absent')
    b2_verdicts[dcode] = {'cont': vc,
                          'syn': vs}
ro = b_eff['P']['r_osc']
osc_v = ('oscillation_reduced' if ro <= 0.7
         else ('oscillation_persist'
               if ro >= 1.0
               else 'oscillation_partial'))
log('B2: contP %s (E=%.4f) contA1 %s (E=%.4f) '
    'osc r=%.4f -> %s'
    % (b2_verdicts['P']['cont'],
       b_eff['P']['E_cont'],
       b2_verdicts['A1']['cont'],
       b_eff['A1']['E_cont'], ro, osc_v))

# pad sensitivity (n_pad==0 and n_trunc==0)
pad_sens = {}
for dcode in ('P', 'A1'):
    ann = annP_c if dcode == 'P' else annA_c
    pads = [p for p in pad_info
            if p['dir'] == dcode]
    keep = set()
    pad_j = 0
    for j in range(NP_A):
        best = find_span(ann[j])
        if best is None:
            continue
        info = pads[pad_j]
        pad_j += 1
        if info['n_pad'] == 0 \
                and info['n_trunc'] == 0:
            keep.add(j)
    e_c = []
    base_all = ref18[dcode].astype(np.float64)
    pad_j = 0
    # (pads already filtered above for this dcode)
    for j in range(NP_A):
        best = find_span(ann[j])
        if best is None:
            continue
        (k1, k2) = best
        info = pads[pad_j]
        pad_j += 1
        if j not in keep:
            continue
        d1 = (SB['s1'][dcode]['dn'][j]
              .astype(np.float64)
              - base_all[j])
        d3 = (SB['s3'][dcode]['dn'][j]
              .astype(np.float64)
              - base_all[j])
        e_c.append(float(
            d1[k1 + 1:].mean()
            - d3[k1 + 1:].mean()))
    pad_sens[dcode] = {
        'n_clean': len(e_c),
        'E_cont_clean': float(np.mean(e_c))
        if e_c else None}

# ================================================================
# PART C-offline: distribution-level reconstruction
# ================================================================
log('== PART C-off: distribution reconstruction ==')
mP = mP18[:NP_A].astype(np.float64)
mA = mA18[:NP_A].astype(np.float64)
MSEQ = {'P': mP, 'A1': mA}
ANN = {'P': annP_c[:NP_A], 'A1': annA_c[:NP_A]}
S = SLOPE_3120
MS = MSTAR_3120
content = {}
for dcode in ('P', 'A1'):
    dm = (MSEQ[dcode][:, 1:]
          - MSEQ[dcode][:, :-1])
    tab = np.zeros((10, N_NEW),
                   dtype=np.float64)
    for k in range(N_NEW):
        col = ANN[dcode][:, k].astype(np.int64)
        dv = dm[:, k]
        allm = dv.mean()
        for cc in range(10):
            sel = (col == cc)
            if sel.any():
                tab[cc, k] = float(
                    dv[sel].mean() - allm)
    content[dcode] = tab
resid = []
for dcode in ('P', 'A1'):
    dm = (MSEQ[dcode][:, 1:]
          - MSEQ[dcode][:, :-1])
    lin = S * (MSEQ[dcode][:, :-1] - MS)
    col = ANN[dcode].astype(np.int64)
    dev = content[dcode][col, np.arange(N_NEW)]
    resid.append((dm - lin - dev).ravel())
sigma = float(np.std(np.concatenate(resid)))
res_pool = np.concatenate(resid)
log('C: sigma=%.4f pool=%d' % (sigma,
                               len(res_pool)))
N_REPS = 5 if SMOKE else 200
rng_mc = np.random.default_rng(3122)
auc_sim = np.zeros(N_NEW + 1,
                   dtype=np.float64)
pit = np.zeros((2, NP_A, N_NEW + 1),
               dtype=np.float64)


def auc_mw(pos_vals, neg_vals):
    x = np.concatenate([pos_vals, neg_vals])
    n1 = len(pos_vals)
    n2 = len(neg_vals)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x),
                     dtype=np.float64)
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


# single simulation pass below: per-step
# snapshots; auc_sim and rank-based PIT are
# both computed on THIS simulation (seed 3122,
# one continuous rng stream, no re-simulation).
snap = {d: [np.zeros((NP_A, N_REPS))
            for _ in range(N_NEW + 1)]
        for d in ('P', 'A1')}
for dcode in ('P', 'A1'):
    seq = MSEQ[dcode]
    mh = np.zeros((NP_A, N_REPS),
                  dtype=np.float64)
    mh[:] = seq[:, 0][:, None]
    snap[dcode][0] = mh.copy()
    col_all = ANN[dcode].astype(np.int64)
    for k in range(N_NEW):
        noise = res_pool[rng_mc.integers(
            0, len(res_pool),
            size=(NP_A, N_REPS))]
        step = S * (mh - MS) \
            + content[dcode][col_all[:, k], k][:, None] \
            + noise
        mh = mh + step
        snap[dcode][k + 1] = mh.copy()
for t in range(N_NEW + 1):
    auc_sim[t] = auc_mw(
        snap['P'][t].ravel(),
        snap['A1'][t].ravel())
# rank-based PIT on the SAME simulation:
# u = (#{sim<true} + 0.5*#{sim==true}) / N_REPS;
# t=0 is the deterministic anchor (u=0.5),
# KS is computed over t>=1 only.
for dcode_i, dcode in enumerate(('P', 'A1')):
    seq = MSEQ[dcode]
    pit[dcode_i, :, 0] = 0.5
    for t in range(1, N_NEW + 1):
        s_i = snap[dcode][t]
        tr = seq[:, t][:, None]
        pit[dcode_i, :, t] = (
            (s_i < tr).sum(axis=1)
            + 0.5 * (s_i == tr).sum(axis=1)) \
            / float(N_REPS)
u = pit[:, :, 1:].ravel()
u_sorted = np.sort(u)
n_u = len(u_sorted)
ecdf = np.arange(1, n_u + 1) / float(n_u)
ks = float(np.max(np.abs(ecdf - u_sorted)))
r_dist = float(np.corrcoef(auc_sim, auc18)[0, 1])
c_dist_v = ('distribution_reconstruction_confirmed'
            if r_dist >= 0.7 else
            ('distribution_reconstruction_partial'
             if r_dist >= 0.5 else
             'distribution_reconstruction_failed'))
c_pit_v = ('pit_calibrated' if ks < 0.05 else
           ('pit_miscalibrated' if ks >= 0.15
            else 'pit_marginal'))
log('C-DIST: r=%.4f -> %s; PIT KS=%.4f -> %s'
    % (r_dist, c_dist_v, ks, c_pit_v))

# ================================================================
# verdict + save
# ================================================================
verdict = '|'.join([
    a1_dir_v, w_san, repro_v,
    b2_verdicts['P']['cont'],
    b2_verdicts['A1']['cont'],
    b2_verdicts['P']['syn'],
    b2_verdicts['A1']['syn'],
    osc_v, c_dist_v, c_pit_v])
runtime = round(time.time() - T0, 1)
results = {
    'phase': 3122,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'n_pairs': NP_,
    'np_a': NP_A,
    'runtime_s': runtime,
    'part_a': {
        'repro': {'verdict': repro_v,
                  'max_diff': drep},
        'write_polarity': {
            'p_dn_L30': p30, 'p_dn_L32': p32,
            'verdict': w_san},
        'spectrum': spec,
        'ablation_margin_split': {
            'effects': adirs,
            'sum_abs': sum_abs,
            'a1_share': a1_share,
            'verdict': a1_dir_v}},
    'part_b': {
        'repro_max_diff': drep_b,
        'effects': b_eff,
        'verdicts': b2_verdicts,
        'osc_verdict': osc_v,
        'pad_sensitivity': pad_sens,
        'n_pad_info': {
            'n_span_total': len(pad_info),
            'n_pad0': sum(1 for p in pad_info
                          if p['n_pad'] == 0),
            'n_trunc0': sum(1 for p in pad_info
                            if p['n_trunc'] == 0)}},
    'part_c': {
        'sigma': sigma,
        'n_reps': N_REPS,
        'r_dist': r_dist,
        'dist_verdict': c_dist_v,
        'pit_ks': ks,
        'pit_verdict': c_pit_v,
        'auc_sim': [float(x) for x in auc_sim]},
}
with io.open(os.path.join(OUT, 'result.json'),
             'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False,
              indent=1)
npz_out = {}
for dcode in ('P', 'A1'):
    npz_out['wrec_dn_%s' % dcode] = \
        wrec[dcode]['m_dn'].astype(np.float32)
    npz_out['wrec_pd_%s' % dcode] = \
        wrec[dcode]['p_dn']
    npz_out['wrec_pf_%s' % dcode] = \
        wrec[dcode]['p_fm']
    npz_out['wrec_pn_%s' % dcode] = \
        wrec[dcode]['p_nm']
for c in SCOND:
    for dcode in ('P', 'A1'):
        npz_out['sb_%s_dn_%s' % (c, dcode)] = \
            SB[c][dcode]['dn'].astype(np.float32)
        npz_out['sb_%s_fam_%s' % (c, dcode)] = \
            SB[c][dcode]['fam'].astype(np.float32)
npz_out['auc_sim'] = auc_sim
npz_out['pit'] = pit
np.savez(os.path.join(OUT, 'p120_readout.npz'),
         **npz_out)
log('verdict=%s' % verdict)
log('done (%.1fs)' % runtime)
print('phase3122 done: %s' % verdict)
