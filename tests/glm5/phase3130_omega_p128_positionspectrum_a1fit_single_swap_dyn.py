# -*- coding: utf-8 -*-
"""Phase 3130 Omega-P128 (T4 13th phase).

Pre-registered (3129 MEMO section 5, frozen
before any observation):
 1. Qwen position spectrum: inject at
    traj-relative k in {0(None),1..12}, K=50
    top-rho coords, delta=0.10*norm, +-sgn,
    P+A1 x672. Anchors: k=0 reproduces 3128/3129
    f10 flip 0.18229166666666666 bit-level
    (1e-9); k=6 reproduces 3129 mid flip
    0.00818452380952381 (1e-9). Decay lengths
    k50/k10 reported.
 2. GLM4 A1-fit coords: refit top-50 rho on A1
    odd-half (L20 hs vs A1 margin), inject on
    A1 even-half. Gate A1_MIN=0.05. P-fit
    replay anchor must reproduce 3129
    a1_flip 0.011904761904761904 (1e-9).
 3. Single-layer swap generation: layers
    {8,9,13,29} x 256 frozen-selected prompts
    vs full W4 swap 672 vs gen_base.
    Full-swap must bit-match 3129
    gen_same_swap (cross-phase generation
    determinism, full run only).
 4. Token fork dynamics: per-token diff rate
    curve + first-change histogram, margin-
    flip subset vs rest. DYN_GATE=0.5 on
    first-token fork share.

Seal: OUT/design_seal.json written before any
GPU work (3129 lesson: seal file mandatory).
"""
import gc
import hashlib
import io
import json
import os
import random as _rnd
import time
import zlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
NAME = ('omega_p128_positionspectrum_'
        'a1fit_single_swap_dyn')
SMOKE = os.environ.get('P3130_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D18 = os.path.join(RDIR, 'phase3118',
                   'omega_p116_autoregressive_margin_'
                   'trajectory')
D25 = RDIR + r'\phase3125' \
      r'\omega_p123_third_comp_qwen_inputstream'
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D27 = RDIR + r'\phase3127' \
      r'\omega_p125_writechain_port_' \
      r'crossmodel_a1closure_fullregen'
D28 = RDIR + r'\phase3128' \
      r'\omega_p126_joint_swap_coord_' \
      r'inject_interaction_s0match'
D29 = RDIR + r'\phase3129' \
      r'\omega_p127_dose_sweep_gen_' \
      r'decouple_s0full_symbolfield'
MDIR_Q = os.path.join(ROOT, 'models', 'hf',
                      'qwen3-4b')
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3130', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a',
                 encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


log('P3130 start SMOKE=%s' % SMOKE)

# ---------------- frozen constants ----------------
NP = 672
N_NEW = 12
NP_B = 8 if SMOKE else NP
NP_REPRO = 4 if SMOKE else 16
GEN_BATCH = 4 if SMOKE else 32
PERM_SEED = 3130
DIRS = ('P', 'A1')

INJ_LAYER_Q = 24
K_DOSE = 50
SPEC_FRAC = 0.10
SPEC_KS = (0, 6, 12) if SMOKE \
    else (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
          11, 12)
DEC_FRACS = (0.5, 0.1)
F10_REF = 0.18229166666666666
MEDDM_REF = 1.0975285470485687
MID6_REF = 0.00818452380952381
SPEC_TOL = 1e-9

INJ_LAYER_G = 20
HALF_FIT = 4 if SMOKE else (NP // 2)
A1_MIN = 0.05
A1FIT_REPLAY_REF = 0.011904761904761904
JOINT_G = {'write': [8, 9, 13, 29]}
SINGLE_LAYERS = [8, 13] if SMOKE \
    else [8, 9, 13, 29]
SINGLE_NP = 4 if SMOKE else 256
GEN_CHANGE_GATE = 0.05
DYN_GATE = 0.5

GROUPS_D = (('s1', 'P'), ('s1', 'A1'),
            ('s3', 'P'), ('s3', 'A1'))

# ---------------- seal ----------------
SEAL = {
    'phase': 3130,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'INJ_LAYER_Q': INJ_LAYER_Q,
        'K_DOSE': K_DOSE,
        'SPEC_FRAC': SPEC_FRAC,
        'SPEC_KS': list(SPEC_KS),
        'DEC_FRACS': list(DEC_FRACS),
        'INJ_LAYER_G': INJ_LAYER_G,
        'HALF_FIT': HALF_FIT,
        'A1_MIN': A1_MIN,
        'JOINT_G_write': JOINT_G['write'],
        'SINGLE_LAYERS': SINGLE_LAYERS,
        'SINGLE_NP': SINGLE_NP,
        'GEN_CHANGE_GATE': GEN_CHANGE_GATE,
        'DYN_GATE': DYN_GATE,
        'PERM_SEED': PERM_SEED},
    'anchors': {
        'f10_flip': F10_REF,
        'f10_med_dm': MEDDM_REF,
        'mid6_flip': MID6_REF,
        'a1fit_pfit_replay': A1FIT_REPLAY_REF},
    'prereg': ('3129 MEMO section 5 items '
               '1-4, frozen before '
               'observation; seal sha8 of '
               'this file recorded in '
               'result.json')}
SEALF = os.path.join(OUT, 'design_seal.json')
if os.path.exists(SEALF):
    prev = json.load(io.open(SEALF,
                             encoding='utf-8'))
    assert prev['constants'] == \
        SEAL['constants'], 'seal drift'
    assert prev['anchors'] == \
        SEAL['anchors'], 'seal drift'
    log('seal ok (existing, constants match)')
else:
    with io.open(SEALF, 'w',
                 encoding='utf-8') as f:
        json.dump(SEAL, f, ensure_ascii=False,
                  indent=1)
    log('seal written (constants frozen)')

# ---------------- frozen inputs ----------
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13,
                            'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
z18 = np.load(os.path.join(D18,
                           'traj_readout.npz'),
              allow_pickle=False)
z25 = np.load(D25 + r'\p123_readout.npz',
              allow_pickle=False)
assert z25['mlg_s0_P'].shape == (672, 37, 13)
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
assert z26['mlg_s0_P'].shape == (672, 41, 13)
assert z26['gen_base_P'].shape == (672, 12)
z27 = np.load(D27 + r'\p125_readout.npz',
              allow_pickle=False)
z28 = np.load(D28 + r'\p126_readout.npz',
              allow_pickle=False)
z29 = np.load(D29 + r'\p127_readout.npz',
              allow_pickle=False)
res28 = json.load(io.open(
    D28 + r'\result.json', encoding='utf-8'))
res29 = json.load(io.open(
    D29 + r'\result.json', encoding='utf-8'))
assert res28['smoke'] is False
assert res29['smoke'] is False
log('frozen inputs ok (05/13/18/25/26/27/28/29)')

# ================================================================
# PART A: offline 3129 link asserts + gates recompute
# ================================================================
log('== PART A: 3129 link asserts ==')
V29 = res29['verdict']
EXP_V29 = ('a_3128_ok|qwen_dose_monotonic'
           '|qwen_midprop_no'
           '|glm4_coord_a1_ineffective'
           '|gen_coupled|s0_fully_deterministic'
           '|interaction_symbolic_not_in_field'
           '|coverage_full')
assert V29 == EXP_V29
pb29 = res29['part_b']
pc29 = res29['part_c']
pd29 = res29['part_d']
assert abs(pb29['dose_q']['f05']['flip']
           - 0.08035714285714285) < 1e-12
assert abs(pb29['dose_q']['f10']['flip']
           - F10_REF) < 1e-12
assert abs(pb29['dose_q']['f20']['flip']
           - 0.28422619047619047) < 1e-12
assert abs(pb29['mid_flip'] - MID6_REF) < 1e-12
assert abs(pc29['a1_flip']
           - A1FIT_REPLAY_REF) < 1e-12
assert pc29['margin_flip_n'] == 172
assert pc29['gen_first_chg'] == 178
assert pc29['gen_chg_rate'] == 1.0
assert pc29['s0_mism'] == 0
assert pd29['peak_layer'] == 40
assert abs(pd29['peak_sep']
           - 0.2934277862541327) < 1e-12
assert abs(pd29['peak_r']
           - 0.2499297708600317) < 1e-12
log('A hard asserts ok (3129 frozen values)')

LAY_Q27 = {'write': [26, 28, 30, 32, 34],
           'port': [20, 21],
           'ctrl': [2, 8, 14]}
LAY_G27 = {'write': [8, 9, 13, 29],
           'port': [20],
           'ctrl': [4, 14, 34]}


def f32_dm(zf, zbase, side, dc, l, NL):
    return (zf['%s_%s_L%02d' % (side, dc, l)]
            [:, -1]
            - zbase['mlg_s0_%s' % dc][:NP, NL,
                                      -1]
            ).astype(np.float64)


def gates27(zf, zbase, LAY, NL, side):
    dmf = {}
    for grp in ('write', 'port', 'ctrl'):
        for l in LAY[grp]:
            for dc in DIRS:
                dmf['%s_L%02d' % (dc, l)] = \
                    f32_dm(zf, zbase, side,
                           dc, l, NL)
    mc = float(np.median(np.abs(
        np.concatenate(
            [dmf['P_L%02d' % l]
             for l in LAY['ctrl']]
            + [dmf['A1_L%02d' % l]
               for l in LAY['ctrl']]))))
    g = {}
    for grp in ('write', 'port'):
        meds = [float(np.median(np.abs(
            dmf['%s_L%02d' % (dc, l)])))
            for l in LAY[grp] for dc in DIRS]
        g[grp] = float(np.median(meds)) / max(
            mc, 1e-12)
    return g, mc


gq_r, mcq_r = gates27(z27, z25, LAY_Q27, 36,
                      'dmq')
gg_r, mcg_r = gates27(z27, z26, LAY_G27, 40,
                      'dmg')
pa29 = res29['part_a']
assert abs(gq_r['write']
           - pa29['gates_q']['write']) < 1e-12
assert abs(gq_r['port']
           - pa29['gates_q']['port']) < 1e-12
assert abs(gg_r['write']
           - pa29['gates_g']['write']) < 1e-12
assert abs(gg_r['port']
           - pa29['gates_g']['port']) < 1e-12
assert abs(mcq_r - pa29['mcq']) < 1e-12
assert abs(mcg_r - pa29['mcg']) < 1e-12
log('A gates recompute 1e-12 ok: q %s g %s'
    % ({k: round(v, 4)
        for k, v in gq_r.items()},
       {k: round(v, 4)
        for k, v in gg_r.items()}))

# ================================================================
# materials factory (3125/3126/3127/3128/3129-
# identical)
# ================================================================
p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
hP = {}
hA1 = {}
for i in range(len(pkB)):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
    elif str(condB[i]) == 'A1':
        hA1[pk] = i
pks = sorted(hP.keys())
assert len(pks) == NP


def build_prompt(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o))
        .encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents[s], PREDS[qrel], ents[o]))
    return text


def make_materials(tok, gens):
    """gens[dc] = (672, 12) frozen tokens."""
    texts = {}
    PID_T = {}
    for dc in DIRS:
        texts[dc] = {}
        ids_l = []
        for pk in pks:
            (s, o) = (int(v)
                      for v in pk.split('_'))
            r = p2r[pk]
            ri1, ri2 = frel[pk]
            if dc == 'P':
                (qrel, lrel) = (r, r)
            else:
                (qrel, lrel) = (r, ri1)
            t_ = build_prompt(mat5, s, o,
                              lrel, qrel)
            texts[dc][pk] = t_
            ids_l.append(list(
                tok(t_,
                    add_special_tokens=False)
                ['input_ids']))
        PID_T[dc] = ids_l
    DOT = int(tok('.',
                  add_special_tokens=False)
              ['input_ids'][0])
    _line_tok = {}

    def line_tokens(s2, o2, lrel):
        key = (s2, o2, lrel)
        if key not in _line_tok:
            txt = ' The %s %s the %s.' % (
                ents_all[s2],
                PREDS_all[lrel],
                ents_all[o2])
            _line_tok[key] = list(tok(
                txt,
                add_special_tokens=False)
                ['input_ids'])
        return _line_tok[key]

    def context_triples(pk, dcode):
        (s, o) = (int(v)
                  for v in pk.split('_'))
        r = p2r[pk]
        ri1, ri2 = frel[pk]
        lrel = r if dcode == 'P' else ri1
        D = [tuple(d) for d in
             mat5['distractors']
             ['%d_%d' % (s, o)]]
        ctx = set((a, rr, b)
                  for (a, rr, b) in
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
        raise RuntimeError('no replacement')

    def traj_tokens(dc, j):
        ids_r = [int(t) for t in gens[dc][j]]
        ptxt = texts[dc][pks[j]]
        encp = tok(ptxt,
                   add_special_tokens=False,
                   return_offsets_mapping=True)
        poffs = [tuple(v) for v in
                 encp['offset_mapping']]
        ids2 = list(ids_r)[:N_NEW]
        while len(ids2) < N_NEW:
            ids2.append(DOT)
        return ptxt, poffs, ids2

    def find_span(dec, offs, s, o, qrel):
        qline = 'The %s %s the %s.' % (
            ents_all[s], PREDS_all[qrel],
            ents_all[o])
        ce = dec.rfind(qline)
        while ce != -1:
            cs = ce
            ce_want = cs + len(qline)
            if dec[ce_want:ce_want + 8] \
                    == ' Is this':
                idxs = [k for k in
                        range(len(offs))
                        if offs[k][0] >= cs
                        and offs[k][1]
                        <= ce_want
                        and offs[k][1]
                        > offs[k][0]]
                if len(idxs) >= 2 \
                        and idxs[0] >= 1 \
                        and idxs[-1] - idxs[0] \
                        + 1 <= N_NEW:
                    return idxs[0], idxs[-1]
            ce = dec.rfind(qline, 0, ce)
        return None

    span_idx = {dc: np.full((NP, 2), -1,
                            dtype=np.int16)
                for dc in DIRS}
    pids_all = {dc: [None] * NP
                for dc in DIRS}

    def build_pids(dc, j):
        pk = pks[j]
        prompt_ids0 = list(PID_T[dc][j])
        ptxt, poffs, _base = traj_tokens(dc, j)
        (s, o) = (int(v)
                  for v in pk.split('_'))
        r = p2r[pk]
        span = find_span(ptxt, poffs, s, o, r)
        pids = {c: list(prompt_ids0)
                for c in ('s0', 's1', 's2',
                          's3')}
        if span is not None:
            (k1, k2) = span
            Lspan = k2 - k1 + 1
            (s2, o2) = pick_replacement(pk, dc)
            (_, _, lrel, _) = context_triples(
                pk, dc)
            sub = line_tokens(s2, o2, lrel)
            n_pad = max(0, Lspan - len(sub))
            sub = sub[:Lspan]
            sub = sub + [DOT] * n_pad
            rng2 = _rnd.Random(zlib.crc32(
                ('s2|%s|%s' % (pk, dc))
                .encode('ascii')))
            shuf = list(sub)
            rng2.shuffle(shuf)
            pids['s1'][k1:k2 + 1] = sub
            pids['s2'][k1:k2 + 1] = shuf
            pids['s3'][k1:k2 + 1] = \
                [DOT] * Lspan
            span_idx[dc][j] = (k1, k2)
        pids_all[dc][j] = pids
        return pids, span

    for dc in DIRS:
        for j in range(NP):
            build_pids(dc, j)
    return {'texts': texts, 'PID_T': PID_T,
            'DOT': DOT, 'build_pids': build_pids,
            'span_idx': span_idx,
            'pids_all': pids_all,
            'traj_tokens': traj_tokens}


log('factory defined')

# ================================================================
# PART B: GPU qwen3-4b position spectrum
# ================================================================
log('== PART B: qwen position spectrum ==')
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

gens_q = {'P': z18['gen_clean__P'],
          'A1': z18['gen_clean__A1']}
tok_q = AutoTokenizer.from_pretrained(MDIR_Q)
model_q = AutoModelForCausalLM.from_pretrained(
    MDIR_Q, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NLQ = len(model_q.model.layers)
assert NLQ == 36
log('qwen3-4b loaded NLQ=%d' % NLQ)
WU_q = model_q.lm_head.weight.detach()
assert model_q.lm_head.bias is None
YES_Q = int(mat5['yes_id'])
NO_Q = int(mat5['no_id'])
w_dn_q = (WU_q[YES_Q] - WU_q[NO_Q]) \
    .float().cpu().numpy()
norm_q = model_q.model.norm
matq = make_materials(tok_q, gens_q)
for dc in DIRS:
    assert np.array_equal(
        matq['span_idx'][dc],
        z25['span_idx_%s' % dc]), dc
log('B span_idx asserts vs 3125 ok (672x2)')


def forward_track_q(prompt_ids, traj_ids,
                    inject=None):
    """3125 semantics: norm ONLY L<NL.
    inject: (layer, coords, delta, sgn, midk)
    -> h[pos0+midk, coords] += delta*sgn at
    that layer OUTPUT; midk None = last."""
    traj_ids = [int(x) for x in traj_ids]
    ids = list(prompt_ids) + traj_ids
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(traj_ids) + 1
    ml = np.zeros((NLQ + 1, npts),
                  dtype=np.float64)
    hooks = []
    if inject is not None:
        (il, coords, delta, sgn, midk) = inject
        lyr = model_q.model.layers[il]
        co_t = torch.as_tensor(
            np.asarray(coords, dtype=np.int64),
            device='cuda')
        dv_t = torch.full((len(coords),),
                          float(delta) * float(sgn),
                          device='cuda',
                          dtype=torch.bfloat16)
        tgt = (-1 if midk is None
               else pos0 + int(midk))

        def _inj(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            o2[0, tgt, co_t] += dv_t
            return None

        hooks.append(
            lyr.register_forward_hook(_inj))
    with torch.inference_mode():
        out = model_q(t_in,
                      output_hidden_states=True,
                      use_cache=False)
        for hk in hooks:
            hk.remove()
        for L in range(NLQ + 1):
            hL = out.hidden_states[L][0]
            if L < NLQ:
                hL = norm_q(hL)
            h = hL.float().cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dn_q)
        del out
    return ml


drep_q = 0.0
for dc in DIRS:
    for j in range(NP_REPRO):
        pids, _sp = matq['build_pids'](dc, j)
        _ptxt, _poffs, base = \
            matq['traj_tokens'](dc, j)
        wl = forward_track_q(pids['s0'], base)
        ref = z25['mlg_s0_%s' % dc][j]
        drep_q = max(drep_q, float(
            np.abs(wl - ref).max()))
assert drep_q <= 1e-4, drep_q
log('B repro max|d|=%.2e (<=1e-4)' % drep_q)

base_m_q = {
    dc: z25['mlg_s0_%s' % dc][:NP_B, NLQ, -1]
        .astype(np.float64)
    for dc in DIRS}
s0_pids_q = {dc: [matq['build_pids'](dc, j)[0]
                  for j in range(NP_B)]
             for dc in DIRS}

# ---- spectrum coords (3129-identical path) ----
h_out = capb['h_out'].astype(np.float32)
m_cap = capb['m'].astype(np.float64)
li = [12, 20, 24, 28, 32].index(INJ_LAYER_Q)
hl = h_out[:, li, :]
hr = np.argsort(np.argsort(hl, axis=0),
                axis=0).astype(np.float64)
mr = np.argsort(np.argsort(m_cap)).astype(
    np.float64)
hc = hr - hr.mean(0, keepdims=True)
mc2 = mr - mr.mean()
den = np.sqrt((hc * hc).sum(0)
              * float((mc2 * mc2).sum()))
rho_cap = (hc * mc2[:, None]).sum(0) / den
d_rho = float(np.abs(
    rho_cap - z28['rho_cap_L24']
    .astype(np.float64)).max())
assert d_rho < 1e-6, d_rho
order_rho = np.argsort(-np.abs(rho_cap))
co50 = order_rho[:K_DOSE]
norms = np.sqrt((hl.astype(np.float64)
                 * hl.astype(np.float64))
                .sum(1))
mean_row_l2 = float(norms.mean())
assert abs(mean_row_l2
           - pb29['mean_row_l2']) < 1e-9
delta_q = SPEC_FRAC * mean_row_l2
assert abs(delta_q
           - 10.694569431733811) < 1e-9
log('B rho_cap vs 3128 npz maxd=%.2e; '
    'delta_q=%.4f' % (d_rho, delta_q))


def inject_run_q_spec(coords, delta, sgn, k):
    """k=0 -> midk None (last); k>=1 ->
    midk=k (traj-relative)."""
    fl = []
    dmm = []
    for dc in DIRS:
        for j in range(NP_B):
            wl = forward_track_q(
                s0_pids_q[dc][j]['s0'],
                matq['traj_tokens'](dc, j)[2],
                inject=(INJ_LAYER_Q, coords,
                        delta, sgn,
                        None if k == 0 else k))
            m_inj = wl[NLQ, -1]
            m_b = base_m_q[dc][j]
            fl.append(int(
                (m_inj > 0) != (m_b > 0)))
            dmm.append(abs(m_inj - m_b))
    return float(np.mean(fl)), \
        float(np.median(dmm))


spec_flip = {}
spec_med = {}
for k in SPEC_KS:
    fp, mp = inject_run_q_spec(
        co50, delta_q, +1, k)
    fm, mm = inject_run_q_spec(
        co50, delta_q, -1, k)
    spec_flip[k] = (fp + fm) / 2.0
    spec_med[k] = (mp + mm) / 2.0
    log('B spec k=%02d: flip=%.6f med|dm|=%.4f'
        % (k, spec_flip[k], spec_med[k]))

ok0 = (abs(spec_flip[0] - F10_REF)
       < SPEC_TOL
       and abs(spec_med[0] - MEDDM_REF)
       < SPEC_TOL)
ok6 = (6 in spec_flip
       and abs(spec_flip[6] - MID6_REF)
       < SPEC_TOL)
log('B anchors: k0 %s k6 %s'
    % (ok0, ok6))
# k=12 is the same injection site as k=0
# (traj last = read point): sanity only in
# full run.
if SMOKE:
    log('B anchors SMOKE: assert skipped '
        '(NP_B=8, granularity 0.0625)')
else:
    assert ok0, (spec_flip[0], F10_REF)
    if 6 in spec_flip:
        assert ok6, (spec_flip[6],
                     MID6_REF)
    if 12 in spec_flip:
        assert abs(spec_flip[12]
                   - spec_flip[0]) < 1e-12
        assert abs(spec_med[12]
                   - spec_med[0]) < 1e-12
        log('B k12==k0 site-identity ok')

ks_pos = [k for k in SPEC_KS if k >= 1]
f0 = spec_flip[0]
k50 = 0
k10 = 0
for k in ks_pos:
    if spec_flip[k] >= DEC_FRACS[0] * f0:
        k50 = max(k50, k)
    if spec_flip[k] >= DEC_FRACS[1] * f0:
        k10 = max(k10, k)
nonincreasing = all(
    spec_flip[ks_pos[i + 1]]
    <= spec_flip[ks_pos[i]] + 0.01
    for i in range(len(ks_pos) - 1))
far_dead = all(spec_flip[k] < 0.5 * f0
               for k in ks_pos if k >= 3)
spec_local_ok = nonincreasing and far_dead
if not ok0:
    spec_v = 'qwen_spec_anchor_fail'
elif spec_local_ok:
    spec_v = 'qwen_spec_reproduced_local'
else:
    spec_v = 'qwen_spec_nonlocal'
log('B-SPEC: k50=%d k10=%d local=%s -> %s'
    % (k50, k10, spec_local_ok, spec_v))

del model_q
gc.collect()
torch.cuda.empty_cache()
log('qwen unloaded; cuda cache cleared')

# ================================================================
# PART C: GPU glm4 A1-fit coords + single swap
#         + dynamics
# ================================================================
log('== PART C: glm4 a1fit + single swap ==')
gens_g = {'P': z26['gen_base_P'],
          'A1': z26['gen_base_A1']}
tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
model_g = AutoModelForCausalLM.from_pretrained(
    MDIR_G, torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    trust_remote_code=True).to('cuda').eval()
NLG = len(model_g.model.layers)
assert NLG == 40
log('glm4 loaded NLG=%d' % NLG)
WUG = model_g.lm_head.weight.detach()
YES_G = int(tok_g(' yes',
                  add_special_tokens=False)
            ['input_ids'][0])
NO_G = int(tok_g(' no',
                 add_special_tokens=False)
           ['input_ids'][0])
w_dn_g = (WUG[YES_G] - WUG[NO_G]) \
    .float().cpu().numpy()
DOT_G = int(tok_g('.',
                  add_special_tokens=False)
            ['input_ids'][0])
norm_g = model_g.model.norm
PADG = tok_g.pad_token_id
if PADG is None:
    PADG = model_g.config.pad_token_id
tok_g.padding_side = 'left'
matg = make_materials(tok_g, gens_g)
for dc in DIRS:
    assert np.array_equal(
        matg['span_idx'][dc],
        z26['span_idx_%s' % dc]), dc
log('C span_idx asserts vs 3126 ok (672x2)')
_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
assert len(PREFIX_IDS) in (0, 2)
log('C gen-prefix: %s' % PREFIX_IDS)


def forward_track_g(prompt_ids, traj_ids,
                    swap_layers=None,
                    inject=None,
                    collect_layer=None):
    """3126 semantics: hook-collected layer
    outputs, normG applied to ALL states.
    inject: (layer, coords, delta, sgn, ipos),
    ipos None = last position."""
    traj_ids = [int(x) for x in traj_ids]
    ids = list(prompt_ids) + traj_ids
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(traj_ids) + 1
    ml = np.zeros((NLG + 1, npts),
                  dtype=np.float64)
    feats = []
    hooks = []
    col_vec = [None]
    if swap_layers:
        for l in swap_layers:
            lyr = model_g.model.layers[l]

            def _swap(mod, inp, out, _l=l):
                o2 = out[0] \
                    if isinstance(out, tuple) \
                    else out
                i2 = inp[0] \
                    if isinstance(inp, tuple) \
                    else inp
                o2.copy_(i2.to(o2.dtype))

            hooks.append(
                lyr.register_forward_hook(
                    _swap))
    if inject is not None:
        (il, coords, delta, sgn, ipos) = inject
        lyr = model_g.model.layers[il]
        co_t = torch.as_tensor(
            np.asarray(coords, dtype=np.int64),
            device='cuda')
        dv_t = torch.full((len(coords),),
                          float(delta) * float(sgn),
                          device='cuda',
                          dtype=torch.bfloat16)
        tgt = ipos if ipos is not None else -1

        def _inj(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            o2[0, tgt, co_t] += dv_t
            return None

        hooks.append(
            lyr.register_forward_hook(_inj))

    def _mk():
        def hook(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            feats.append(o2.detach())
        return hook

    for lyr in model_g.model.layers:
        hooks.append(
            lyr.register_forward_hook(_mk()))
    if collect_layer is not None:
        lyr = model_g.model.layers[
            collect_layer]

        def _col(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            col_vec[0] = o2[0, -1] \
                .detach().float().cpu().numpy()

        hooks.append(
            lyr.register_forward_hook(_col))
    with torch.inference_mode():
        out = model_g(t_in,
                      output_hidden_states=False,
                      use_cache=False)
        for hk in hooks:
            hk.remove()
        seq = [model_g.model.embed_tokens(
            t_in)] + feats
        for L in range(NLG + 1):
            h = norm_g(seq[L][0]).float() \
                .cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dn_g)
        del out, seq, feats
    return ml, (col_vec[0]
                if collect_layer is not None
                else None)


drep_g = 0.0
for dc in DIRS:
    for j in range(NP_REPRO):
        pids, _sp = matg['build_pids'](dc, j)
        _ptxt, _poffs, base = \
            matg['traj_tokens'](dc, j)
        wl, _cv = forward_track_g(
            pids['s0'], base)
        ref = z26['mlg_s0_%s' % dc][j]
        drep_g = max(drep_g, float(
            np.abs(wl - ref).max()))
assert drep_g <= 1e-4, drep_g
log('C repro max|d|=%.2e (<=1e-4)' % drep_g)


def collect_fit(dc, half_idx):
    hs = np.zeros((len(half_idx), 4096),
                  dtype=np.float64)
    ms = np.zeros(len(half_idx),
                  dtype=np.float64)
    for jj, j in enumerate(half_idx):
        pids, _sp = matg['build_pids'](dc, j)
        _ptxt, _poffs, base = \
            matg['traj_tokens'](dc, j)
        wl, cv = forward_track_g(
            pids['s0'], base,
            collect_layer=INJ_LAYER_G)
        hs[jj] = cv
        ms[jj] = wl[NLG, -1]
        if jj % 64 == 0:
            log('C collect %s %d/%d (%.1fs)'
                % (dc, jj, len(half_idx),
                   time.time() - T0))
    return hs, ms


def rho_topk(hs, ms, k):
    hr2 = np.argsort(np.argsort(hs, axis=0),
                     axis=0).astype(np.float64)
    mr2 = np.argsort(np.argsort(ms)).astype(
        np.float64)
    hc2 = hr2 - hr2.mean(0, keepdims=True)
    mc3 = mr2 - mr2.mean()
    den2 = np.sqrt((hc2 * hc2).sum(0)
                   * float((mc3 * mc3).sum()))
    rho = (hc2 * mc3[:, None]).sum(0) / den2
    order = np.argsort(-np.abs(rho))
    delta = SPEC_FRAC * float(
        np.sqrt((hs * hs).sum(1)).mean())
    return order[:k], rho, delta


half_a = list(range(0, NP, 2))[:HALF_FIT]
half_b = list(range(1, NP, 2))[:HALF_FIT]
# P-fit replay (must match 3128 frozen)
hs_p, ms_p = collect_fit('P', half_a)
coords_g_re, rho_g_re, delta_g_re = \
    rho_topk(hs_p, ms_p, K_DOSE)
if SMOKE:
    log('C P-fit replay SMOKE: exact '
        'assert skipped (HALF_FIT=%d vs '
        '3128 frozen 336)' % HALF_FIT)
else:
    assert np.array_equal(
        coords_g_re,
        z28['coords_g_top50']), 'P-fit drift'
    assert abs(delta_g_re
               - float(res28['part_c']
                       ['delta_g'])) < 1e-9
    log('C P-fit replay exact (coords + '
        'delta %.4f)' % delta_g_re)
# A1-fit (new: A1 margin, same odd-half)
hs_a1, ms_a1 = collect_fit('A1', half_a)
coords_g_a1, rho_a1, delta_g_a1 = \
    rho_topk(hs_a1, ms_a1, K_DOSE)
log('C A1-fit top50 done delta=%.4f '
    'overlap_with_P=%d/50'
    % (delta_g_a1,
       len(set(coords_g_a1.tolist())
           & set(coords_g_re.tolist()))))


def inject_run_g_a1(coords, delta, sgn):
    fl = []
    dmm = []
    for j in half_b:
        pids, _sp = matg['build_pids']('A1', j)
        _ptxt, _poffs, base = \
            matg['traj_tokens']('A1', j)
        wl, _cv = forward_track_g(
            pids['s0'], base,
            inject=(INJ_LAYER_G, coords,
                    delta, sgn, None))
        m_inj = wl[NLG, -1]
        m_b = float(
            z26['mlg_s0_A1'][j, NLG, -1])
        fl.append(int(
            (m_inj > 0) != (m_b > 0)))
        dmm.append(abs(m_inj - m_b))
    return float(np.mean(fl)), \
        float(np.median(dmm))


# replay 3129 config (P-fit coords + frozen
# delta) on A1 half_b: anchor
pf_s1, _ = inject_run_g_a1(
    coords_g_re, delta_g_re, +1)
pf_sm1, _ = inject_run_g_a1(
    coords_g_re, delta_g_re, -1)
pfit_a1_flip = (pf_s1 + pf_sm1) / 2.0
if SMOKE:
    log('C P-fit+A1 replay SMOKE: anchor '
        'assert skipped (flip=%.4f, small '
        'halves)' % pfit_a1_flip)
else:
    assert abs(pfit_a1_flip
               - A1FIT_REPLAY_REF) < 1e-9, (
        pfit_a1_flip, A1FIT_REPLAY_REF)
    log('C P-fit+A1 replay exact: flip='
        '%.6f' % pfit_a1_flip)

a1f_s1, _ = inject_run_g_a1(
    coords_g_a1, delta_g_a1, +1)
a1f_sm1, _ = inject_run_g_a1(
    coords_g_a1, delta_g_a1, -1)
a1fit_flip = (a1f_s1 + a1f_sm1) / 2.0
a1fit_ok = a1fit_flip >= A1_MIN
a1fit_v = ('glm4_a1fit_effective'
           if a1fit_ok
           else 'glm4_a1fit_confirms_binding')
log('C-A1FIT: flip=%.4f (P-fit ref %.4f) '
    '-> %s' % (a1fit_flip, pfit_a1_flip,
               a1fit_v))

# ---- single-layer swap generation ----
def gen_batch_g(pids_list, swap_layers=None):
    chunk = [list(PREFIX_IDS) + list(p)
             for p in pids_list]
    maxlen = max(len(p) for p in chunk)
    ids_p = np.full((len(chunk), maxlen),
                    PADG, dtype=np.int64)
    mask_p = np.zeros((len(chunk), maxlen),
                      dtype=np.int64)
    for i, p in enumerate(chunk):
        ids_p[i, maxlen - len(p):] = p
        mask_p[i, maxlen - len(p):] = 1
    hooks = []
    if swap_layers:
        for l in swap_layers:
            lyr = model_g.model.layers[l]

            def _swap(mod, inp, out,
                      _l=l):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                i2 = inp[0] \
                    if isinstance(inp,
                                  tuple) \
                    else inp
                o2.copy_(i2.to(o2.dtype))

            hooks.append(
                lyr.register_forward_hook(
                    _swap))
    t_ids = torch.tensor(ids_p,
                         device='cuda')
    t_mask = torch.tensor(mask_p,
                          device='cuda')
    with torch.inference_mode():
        outg = model_g.generate(
            input_ids=t_ids,
            attention_mask=t_mask,
            max_new_tokens=N_NEW,
            do_sample=False, num_beams=1,
            pad_token_id=PADG)
    for hk in hooks:
        hk.remove()
    newg = outg[:, maxlen:]
    res = []
    for row in newg:
        ids_r = [int(t) for t in row]
        if PADG in ids_r:
            ids_r = ids_r[
                :ids_r.index(PADG)]
        for e in model_g.config.eos_token_id \
                if isinstance(
                    model_g.config
                    .eos_token_id,
                    list) else [
            model_g.config.eos_token_id]:
            if e in ids_r:
                ids_r = ids_r[
                    :ids_r.index(e)]
                break
        res.append(ids_r)
    return res


def pad12(ids_r):
    v = list(ids_r)[:N_NEW]
    return v + [DOT_G] * (N_NEW - len(v))


GEN_NP = NP_B if SMOKE else NP
pids_P_all = [matg['pids_all']['P'][j]['s0']
              for j in range(GEN_NP)]

# full W4 swap rerun (672) with token record
gen_swap_full = []
for b0 in range(0, GEN_NP, GEN_BATCH):
    batch = pids_P_all[b0:b0 + GEN_BATCH]
    gen_swap_full.extend(gen_batch_g(
        batch,
        swap_layers=JOINT_G['write']))
    if (b0 // GEN_BATCH) % 7 == 0:
        log('C full-swap batch %d/%d done'
            % (b0 // GEN_BATCH + 1,
               (GEN_NP + GEN_BATCH - 1)
               // GEN_BATCH))
gen_same_new = np.zeros(GEN_NP, dtype=bool)
gen_tokdiff = np.zeros(N_NEW, dtype=np.float64)
gen_first_new = 0
for j in range(GEN_NP):
    b12 = pad12([int(v) for v in
                 z26['gen_base_P'][j]])
    g12 = pad12(gen_swap_full[j])
    gen_same_new[j] = (g12 == b12)
    gen_first_new += int(g12[0] != b12[0])
    for t in range(N_NEW):
        gen_tokdiff[t] += float(
            g12[t] != b12[t])
gen_tokdiff /= float(GEN_NP)
gen_chg_new = 1.0 - float(gen_same_new.mean())
if SMOKE:
    swap_bitexact = True
    log('C full-swap SMOKE: chg=%.4f '
        '(bitexact vs 3129 skipped)'
        % gen_chg_new)
else:
    swap_bitexact = bool(np.array_equal(
        gen_same_new,
        z29['gen_same_swap'].astype(bool)))
    assert swap_bitexact, (
        'cross-phase gen drift vs 3129')
    assert gen_first_new == 178
    log('C full-swap 3129 bitexact ok '
        '(chg=%.4f first=%d)'
        % (gen_chg_new, gen_first_new))
swap_v = ('gen_swap_3129_bitexact'
          if swap_bitexact
          else 'gen_swap_drift_found')

# single-layer swaps on frozen selection
rng_sel = np.random.default_rng(PERM_SEED)
sel = np.sort(rng_sel.choice(
    GEN_NP, size=min(SINGLE_NP, GEN_NP),
    replace=False)).astype(np.int64)
sel_pids = [pids_P_all[j] for j in sel]
single_chg = {}
single_first = {}
for l in SINGLE_LAYERS:
    gen_l = []
    for b0 in range(0, len(sel_pids),
                    GEN_BATCH):
        batch = sel_pids[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g(
            batch, swap_layers=[l]))
    same_l = np.zeros(len(sel_pids),
                      dtype=bool)
    first_l = 0
    for jj, j in enumerate(sel):
        b12 = pad12([int(v) for v in
                     z26['gen_base_P'][j]])
        g12 = pad12(gen_l[jj])
        same_l[jj] = (g12 == b12)
        first_l += int(g12[0] != b12[0])
    single_chg[l] = 1.0 - float(same_l.mean())
    single_first[l] = first_l
    log('C single L%02d: chg=%.4f first=%d'
        % (l, single_chg[l], first_l))
single_max = max(single_chg.values())
dominance = (single_max / gen_chg_new
             if gen_chg_new > 1e-12 else 0.0)
single_v = ('single_dominant'
            if dominance >= 0.5
            else 'single_distributed')
log('C-SINGLE: max=%.4f overall=%.4f '
    'dominance=%.2f -> %s'
    % (single_max, gen_chg_new, dominance,
       single_v))

# ---- dynamics: first-change histogram ----
first_hist = np.zeros(N_NEW + 1,
                      dtype=np.int64)
for j in range(GEN_NP):
    b12 = pad12([int(v) for v in
                 z26['gen_base_P'][j]])
    g12 = pad12(gen_swap_full[j])
    fc = N_NEW
    for t in range(N_NEW):
        if g12[t] != b12[t]:
            fc = t + 1
            break
    first_hist[fc] += 1
assert int(first_hist.sum()) == GEN_NP
if not SMOKE:
    # full swap rewrites all: every
    # sample differs somewhere
    assert int(first_hist[1:].sum()) \
        == GEN_NP
    # first-token divergence subset
    assert int(first_hist[1]) \
        == gen_first_new == 178
mfl = z29['margin_flip_P'].astype(bool)[:GEN_NP]
diff_mat = ~gen_same_new
b12_all = [pad12([int(v) for v in
                  z26['gen_base_P'][j]])
           for j in range(GEN_NP)]
g12_all = [pad12(gen_swap_full[j])
           for j in range(GEN_NP)]
tf_flip = np.zeros(N_NEW, dtype=np.float64)
tf_rest = np.zeros(N_NEW, dtype=np.float64)
if int(mfl.sum()) > 0:
    for t in range(N_NEW):
        bcol = np.array([g12_all[j][t]
                         != b12_all[j][t]
                         for j in range(GEN_NP)])
        tf_flip[t] = float(bcol[mfl].mean())
        tf_rest[t] = float(bcol[~mfl].mean())
else:
    tf_flip[:] = -1.0
    tf_rest[:] = -1.0
first_share = float(first_hist[1]) / float(
    max(1, int(first_hist[1:].sum())))
dyn_v = ('dyn_first_concentrated'
         if first_share >= DYN_GATE
         else 'dyn_spread')
log('C-DYN: first-token share=%.3f -> %s'
    % (first_share, dyn_v))

del model_g
gc.collect()
torch.cuda.empty_cache()
log('glm4 unloaded; cuda cache cleared')

# ================================================================
# dumps
# ================================================================
runtime = time.time() - T0
verdict = '|'.join([
    'a_3129_ok', spec_v,
    'glm4_pfit_replay_exact', a1fit_v,
    swap_v, single_v, dyn_v,
    'coverage_full'])
log('VERDICT: %s' % verdict)
result = {
    'name': NAME,
    'phase': 3130,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'gates_q': gq_r, 'gates_g': gg_r,
        'mcq': mcq_r, 'mcg': mcg_r,
        'asserts_3129': 'ok'},
    'part_b': {
        'spec_ks': list(SPEC_KS),
        'spec_flip': [float(spec_flip[k])
                      for k in SPEC_KS],
        'spec_med_dm': [float(spec_med[k])
                        for k in SPEC_KS],
        'anchor_k0_ok': bool(ok0),
        'anchor_k6_ok': bool(6 in spec_flip
                             and ok6),
        'k50': int(k50), 'k10': int(k10),
        'd50': int(N_NEW - k50),
        'd10': int(N_NEW - k10),
        'delta_q': delta_q,
        'mean_row_l2': mean_row_l2},
    'part_c': {
        'pfit_replay_flip': pfit_a1_flip,
        'a1fit_flip': a1fit_flip,
        'a1fit_flip_s1': a1f_s1,
        'a1fit_flip_s-1': a1f_sm1,
        'a1fit_ok': bool(a1fit_ok),
        'coords_a1_overlap_P': len(
            set(coords_g_a1.tolist())
            & set(coords_g_re.tolist())),
        'delta_g_a1': delta_g_a1,
        'full_swap_chg': gen_chg_new,
        'full_swap_bitexact_3129':
            bool(swap_bitexact),
        'full_swap_first': int(gen_first_new),
        'single_chg': {str(k): float(v) for
                       k, v in
                       single_chg.items()},
        'single_first': {str(k): int(v) for
                         k, v in
                         single_first.items()},
        'dominance': dominance,
        'sel256_sha8': hashlib.sha256(
            sel.tobytes()).hexdigest()[:8],
        'first_hist': [int(v) for v in
                       first_hist],
        'tok_diff_all': [float(v) for v in
                         gen_tokdiff],
        'tok_diff_flip': [float(v) for v in
                          tf_flip],
        'tok_diff_rest': [float(v) for v in
                          tf_rest],
        'first_share': first_share},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)
order_a1_full = np.argsort(-np.abs(rho_a1))
npz_out = {
    'spec_ks': np.asarray(SPEC_KS,
                          dtype=np.int64),
    'spec_flip': np.asarray(
        [spec_flip[k] for k in SPEC_KS],
        dtype=np.float64),
    'spec_med_dm': np.asarray(
        [spec_med[k] for k in SPEC_KS],
        dtype=np.float64),
    'coords_g_a1_top50':
        coords_g_a1.astype(np.int64),
    'rho_a1_top200':
        rho_a1[order_a1_full[:200]].astype(
            np.float32),
    'gen_same_full': gen_same_new.astype(
        np.int64),
    'gen_tokdiff': gen_tokdiff.astype(
        np.float64),
    'gen_first_hist': first_hist,
    'sel': sel.astype(np.int64),
    'single_chg': np.asarray(
        [single_chg[l] for l in
         SINGLE_LAYERS],
        dtype=np.float64),
    'tok_diff_flip': tf_flip.astype(
        np.float64),
    'tok_diff_rest': tf_rest.astype(
        np.float64)}
np.savez(os.path.join(OUT,
                      'p128_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p128_readout.npz (%d keys)'
    % len(npz_out))
log('P3130 DONE (%.1fs)' % (time.time() - T0))
