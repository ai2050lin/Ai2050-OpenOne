# -*- coding: utf-8 -*-
"""Phase 3131 (Omega-P129): layer-window +
full-layer single-swap spectrum + A1-half
generation symmetry + fork-decision layer.
Prereg: 3130 MEMO section 6 items 1-4.
Seal frozen BEFORE any observation (3129
lesson). Gate design excludes identity
points (3130 errata lesson)."""
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
NAME = ('omega_p129_layerwindow_'
        'single40_a1gen_forklayer')
SMOKE = os.environ.get('P3131_SMOKE',
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
D28 = RDIR + r'\phase3128' \
      r'\omega_p126_joint_swap_coord_' \
      r'inject_interaction_s0match'
D30 = RDIR + r'\phase3130' \
      r'\omega_p128_positionspectrum_' \
      r'a1fit_single_swap_dyn'
MDIR_Q = os.path.join(ROOT, 'models', 'hf',
                      'qwen3-4b')
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3131', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0,
                            msg)
    with io.open(LOGF, 'a',
                 encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


# ---------------- frozen constants ------
NP = 672
N_NEW = 12
NP_B = 8 if SMOKE else NP
NP_REPRO = 4 if SMOKE else 16
GEN_BATCH = 4 if SMOKE else 32
DIRS = ('P', 'A1')

INJ_LAYER_Q = 24
K_DOSE = 50
SPEC_FRAC = 0.10
LAYER_WIN_Q = [20, 22, 24, 26, 28]
F10_REF = 0.18229166666666666
MEDDM_REF = 1.0975285470485687
DELTA_Q_REF = 10.694569431733811
SPEC_TOL = 1e-9
WIN_GATE = 0.5

INJ_LAYER_G = 20
JOINT_G = {'write': [8, 9, 13, 29]}
SINGLE40_NP = 4 if SMOKE else 64
SEL_SEED = 3131
SYM_GATE = 0.95
DOM_GATE = 0.5
FORK_LATE_L = 38
FIRST_P_REF = 178

# ---------------- seal ------------------
SEAL = {
    'phase': 3131,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'INJ_LAYER_Q': INJ_LAYER_Q,
        'K_DOSE': K_DOSE,
        'SPEC_FRAC': SPEC_FRAC,
        'LAYER_WIN_Q': LAYER_WIN_Q,
        'INJ_LAYER_G': INJ_LAYER_G,
        'JOINT_G_write': JOINT_G['write'],
        'SINGLE40_NP': SINGLE40_NP,
        'SEL_SEED': SEL_SEED,
        'SYM_GATE': SYM_GATE,
        'DOM_GATE': DOM_GATE,
        'FORK_LATE_L': FORK_LATE_L,
        'WIN_GATE': WIN_GATE},
    'anchors': {
        'l24_flip': F10_REF,
        'l24_med_dm': MEDDM_REF,
        'delta_q': DELTA_Q_REF,
        'full_swap_first': FIRST_P_REF},
    'prereg': ('3130 MEMO section 6 items '
               '1-4; single40 adjusted 256->'
               '64 prompts before any '
               'observation (runtime budget '
               '10.2h->2.6h, grain 0.0156 '
               'sufficient for layer gaps '
               '0.17-0.36); 3130 4-layer '
               '256-values kept as soft '
               'reference. Frozen before '
               'observation.')}
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
        json.dump(SEAL, f,
                  ensure_ascii=False,
                  indent=1)
    log('seal written (constants frozen)')

# ---------------- frozen inputs ---------
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
z28 = np.load(D28 + r'\p126_readout.npz',
              allow_pickle=False)
z30 = np.load(D30 + r'\p128_readout.npz',
              allow_pickle=False)
res30 = json.load(io.open(
    D30 + r'\result.json', encoding='utf-8'))
assert res30['smoke'] is False
log('frozen inputs ok (05/13/18/25/26/28/30)')

# ================================================================
# PART A: offline 3130 link asserts
# ================================================================
log('== PART A: 3130 link asserts ==')
V30 = res30['verdict']
EXP_V30 = ('a_3129_ok|qwen_spec_nonlocal'
           '|glm4_pfit_replay_exact'
           '|glm4_a1fit_confirms_binding'
           '|gen_swap_3129_bitexact'
           '|single_dominant|dyn_spread'
           '|coverage_full')
assert V30 == EXP_V30
pb30 = res30['part_b']
pc30 = res30['part_c']
sf30 = pb30['spec_flip']
assert abs(sf30[0] - F10_REF) < 1e-12
assert abs(sf30[6]
           - 0.00818452380952381) < 1e-12
assert abs(sf30[12] - sf30[0]) < 1e-12
assert abs(pb30['spec_med_dm'][0]
           - MEDDM_REF) < 1e-12
assert pb30['k50'] == 12 and pb30['d50'] == 0
assert abs(pc30['dominance']
           - 208 / 256.0) < 1e-9
assert abs(pc30['first_share']
           - FIRST_P_REF / 672.0) < 1e-12
assert pc30['full_swap_chg'] == 1.0
assert pc30['full_swap_first'] == 178
assert pc30['full_swap_bitexact_3129'] \
    is True
raw30 = io.open(
    D30 + r'\result.json', 'rb').read()
sha30 = hashlib.sha256(
    raw30).hexdigest()[:8]
assert res30['seal_sha8'] == '2bfa1e2b'
seal30 = json.load(io.open(
    D30 + r'\design_seal.json',
    encoding='utf-8'))
sha30b = hashlib.sha256(io.open(
    D30 + r'\design_seal.json', 'rb'
).read()).hexdigest()[:8]
assert sha30b == '2bfa1e2b'
log('A hard asserts ok (3130 frozen, '
    'result sha8 %s)' % sha30)

# materials factory (3125/3126/3130-identical)
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
# PART B: qwen layer-window injection
# ================================================================
log('== PART B: qwen layer window ==')
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

# coords: 3130-identical path (capture L24)
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
delta_q = SPEC_FRAC * mean_row_l2
assert abs(delta_q - DELTA_Q_REF) < 1e-9
log('B rho_cap vs 3128 npz maxd=%.2e; '
    'delta_q=%.4f' % (d_rho, delta_q))


def inject_run_q(coords, delta, sgn, layer):
    fl = []
    dmm = []
    for dc in DIRS:
        for j in range(NP_B):
            wl = forward_track_q(
                s0_pids_q[dc][j]['s0'],
                matq['traj_tokens'](dc, j)[2],
                inject=(layer, coords, delta,
                        sgn, None))
            m_inj = wl[NLQ, -1]
            m_b = base_m_q[dc][j]
            fl.append(int(
                (m_inj > 0) != (m_b > 0)))
            dmm.append(abs(m_inj - m_b))
    return float(np.mean(fl)), \
        float(np.median(dmm))


win_flip = {}
win_med = {}
for lw in LAYER_WIN_Q:
    fp, mp = inject_run_q(co50, delta_q, +1,
                          lw)
    fm, mm = inject_run_q(co50, delta_q, -1,
                          lw)
    win_flip[lw] = (fp + fm) / 2.0
    win_med[lw] = (mp + mm) / 2.0
    log('B win L%02d: flip=%.6f med|dm|=%.4f'
        % (lw, win_flip[lw], win_med[lw]))

if not SMOKE:
    assert abs(win_flip[INJ_LAYER_Q]
               - F10_REF) < SPEC_TOL
    assert abs(win_med[INJ_LAYER_Q]
               - MEDDM_REF) < SPEC_TOL
    l24_ok = True
    log('B anchors: L24 flip+med exact '
        '(1e-9, 3128/3130 chain)')
else:
    l24_ok = False
    log('B anchors: SKIPPED in SMOKE '
        '(NP_B=%d coarse)' % NP_B)

f0 = F10_REF
n_win = sum(1 for lw in LAYER_WIN_Q
            if win_flip[lw] >= WIN_GATE * f0)
win_v = ('qwen_win_wide' if n_win >= 3
         else ('qwen_win_narrow'
               if n_win <= 1
               else 'qwen_win_partial'))
log('B-WIN: n_win=%d/5 (gate %.3f) -> %s'
    % (n_win, WIN_GATE * f0, win_v))

del model_q
gc.collect()
torch.cuda.empty_cache()
log('qwen unloaded; cuda cache cleared')

# ================================================================
# PART C: glm4 single40 + a1gen + fork layer
# ================================================================
log('== PART C: glm4 single40/a1gen/fork ==')
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


def forward_track_g(prompt_ids, traj_ids,
                    swap_layers=None):
    traj_ids = [int(x) for x in traj_ids]
    ids = list(prompt_ids) + traj_ids
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(traj_ids) + 1
    ml = np.zeros((NLG + 1, npts),
                  dtype=np.float64)
    feats = []
    hooks = []
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
    return ml


drep_g = 0.0
for dc in DIRS:
    for j in range(NP_REPRO):
        pids, _sp = matg['build_pids'](dc, j)
        _ptxt, _poffs, base = \
            matg['traj_tokens'](dc, j)
        wl = forward_track_g(pids['s0'], base)
        ref = z26['mlg_s0_%s' % dc][j]
        drep_g = max(drep_g, float(
            np.abs(wl - ref).max()))
assert drep_g <= 1e-4, drep_g
log('C repro max|d|=%.2e (<=1e-4)' % drep_g)

_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
assert len(PREFIX_IDS) in (0, 2)
log('C gen-prefix: %s' % PREFIX_IDS)


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


GEN_NP = NP_B
pids_P_all = [matg['pids_all']['P'][j]['s0']
              for j in range(GEN_NP)]

# ---- C0: full W4 swap rerun (P 672) ----
log('C0: full-swap P rerun %d' % GEN_NP)
gen_swap_full = []
for b0 in range(0, GEN_NP, GEN_BATCH):
    batch = pids_P_all[b0:b0 + GEN_BATCH]
    gen_swap_full.extend(gen_batch_g(
        batch,
        swap_layers=JOINT_G['write']))
    if (b0 // GEN_BATCH) % 7 == 0:
        log('C0 full-swap batch %d/%d done'
            % (b0 // GEN_BATCH + 1,
               (GEN_NP + GEN_BATCH - 1)
               // GEN_BATCH))
gen_same_new = np.zeros(GEN_NP, dtype=bool)
gen_first_new = 0
for j in range(GEN_NP):
    b12 = pad12([int(v) for v in
                 z26['gen_base_P'][j]])
    g12 = pad12(gen_swap_full[j])
    gen_same_new[j] = (g12 == b12)
    gen_first_new += int(g12[0] != b12[0])
gen_chg_new = 1.0 - float(gen_same_new.mean())
if not SMOKE:
    swap_bitexact = bool(np.array_equal(
        gen_same_new,
        z30['gen_same_full'].astype(bool)))
    assert swap_bitexact, 'cross-phase drift'
    assert gen_first_new == FIRST_P_REF
    assert gen_chg_new == 1.0
    swap_v = 'gen_swap_3130_bitexact'
    log('C0 3130 bitexact ok (chg=%.4f '
        'first=%d)' % (gen_chg_new,
                       gen_first_new))
else:
    swap_bitexact = False
    swap_v = 'gen_swap_smoke_subset'
    log('C0 SMOKE subset chg=%.4f '
        'first=%d (bitexact deferred)'
        % (gen_chg_new, gen_first_new))

# ---- C1: full-layer single-swap spectrum
sel = np.sort(np.random.default_rng(
    SEL_SEED).choice(
    GEN_NP, size=min(SINGLE40_NP, GEN_NP),
    replace=False)).astype(np.int64)
sel_sha8 = hashlib.sha256(
    sel.tobytes()).hexdigest()[:8]
log('C1: single40 spectrum sel64 sha8=%s'
    % sel_sha8)
sel_pids = [pids_P_all[j] for j in sel]
chg40 = {}
first40 = {}
for l in range(NLG):
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
    chg40[l] = 1.0 - float(same_l.mean())
    first40[l] = first_l
    log('C1 single L%02d: chg=%.4f first=%d'
        % (l, chg40[l], first_l))
c40 = np.asarray([chg40[l]
                  for l in range(NLG)],
                 dtype=np.float64)
f40 = np.asarray([first40[l]
                  for l in range(NLG)],
                 dtype=np.int64)
peak40 = int(np.argmax(c40))
single_max = float(c40.max())
dom40 = single_max / gen_chg_new
single_v = ('single40_dominant'
            if dom40 >= DOM_GATE
            else 'single40_distributed')
# soft reference vs 3130 (256-sample)
soft_ref = {8: 208 / 256.0,
            9: 205 / 256.0,
            13: 137 / 256.0,
            29: 45 / 256.0}
soft_dev = {str(l): float(
    abs(chg40[l] - soft_ref[l]))
    for l in soft_ref}
log('C-SINGLE40: peak=L%02d max=%.4f '
    'dom=%.2f -> %s; soft dev vs 3130 '
    '%s' % (peak40, single_max, dom40,
            single_v,
            ' '.join('%s:%.3f' % (k, v)
                     for k, v in
                     sorted(soft_dev.items(),
                            key=lambda t:
                            -t[1]))))

# ---- C2: A1-half full swap generation --
pids_A1_all = [matg['pids_all']['A1'][j]['s0']
               for j in range(GEN_NP)]
log('C2: full-swap A1 rerun %d' % GEN_NP)
gen_swap_a1 = []
for b0 in range(0, GEN_NP, GEN_BATCH):
    batch = pids_A1_all[b0:b0 + GEN_BATCH]
    gen_swap_a1.extend(gen_batch_g(
        batch,
        swap_layers=JOINT_G['write']))
    if (b0 // GEN_BATCH) % 7 == 0:
        log('C2 a1-swap batch %d/%d done'
            % (b0 // GEN_BATCH + 1,
               (GEN_NP + GEN_BATCH - 1)
               // GEN_BATCH))
gen_same_a1 = np.zeros(GEN_NP, dtype=bool)
gen_first_a1 = 0
hist_a1 = np.zeros(N_NEW + 1,
                   dtype=np.int64)
for j in range(GEN_NP):
    b12 = pad12([int(v) for v in
                 z26['gen_base_A1'][j]])
    g12 = pad12(gen_swap_a1[j])
    gen_same_a1[j] = (g12 == b12)
    gen_first_a1 += int(g12[0] != b12[0])
    fc = N_NEW
    for t in range(N_NEW):
        if g12[t] != b12[t]:
            fc = t + 1
            break
    hist_a1[fc] += 1
assert int(hist_a1.sum()) == GEN_NP
chg_a1 = 1.0 - float(gen_same_a1.mean())
a1_v = ('a1_gen_symmetric'
        if chg_a1 >= SYM_GATE
        else 'a1_gen_asymmetric')
log('C2-A1: chg=%.4f first=%d '
    '(P ref %.4f/%d) -> %s'
    % (chg_a1, gen_first_a1, gen_chg_new,
       gen_first_new, a1_v))

# ---- C3: fork-decision layer ----------
log('C3: fork layer (swap traj %d)'
    % GEN_NP)
ml_swap_last = np.zeros((GEN_NP, NLG + 1),
                        dtype=np.float64)
for j in range(GEN_NP):
    pids = pids_P_all[j]
    ml = forward_track_g(
        pids, gen_swap_full[j])
    ml_swap_last[j] = ml[:, -1]
    if j % 128 == 0:
        log('C3 swap traj %d/%d' % (j,
                                    GEN_NP))
ml_base_last = z26['mlg_s0_P'][:GEN_NP,
                               :, -1].astype(
    np.float64)
assert ml_base_last.shape == (GEN_NP,
                              NLG + 1)
dm = ml_swap_last - ml_base_last
med_abs_dm = np.median(np.abs(dm), axis=0)
mfl30 = z30['gen_same_full'].astype(bool)
first_set = np.array(
    [int(pad12(gen_swap_full[j])[0]
         != pad12([int(v) for v in
                   z26['gen_base_P'][j]])[0])
     for j in range(GEN_NP)],
    dtype=bool)
n_first = int(first_set.sum())
if n_first > 0:
    med_dm_first = np.median(
        np.abs(dm[first_set]), axis=0)
    med_dm_rest = np.median(
        np.abs(dm[~first_set]), axis=0)
else:
    med_dm_first = np.full(NLG + 1, -1.0)
    med_dm_rest = np.full(NLG + 1, -1.0)
pk_dm = int(np.argmax(med_abs_dm))
pk_first = (int(np.argmax(med_dm_first))
            if n_first > 0 else -1)
fork_v = ('fork_late' if pk_dm
          >= FORK_LATE_L
          else ('fork_mid' if pk_dm >= 20
                else 'fork_early'))
log('C3-FORK: peak L%02d (first-set peak '
    'L%02d, n=%d) -> %s'
    % (pk_dm, pk_first, n_first, fork_v))

del model_g
gc.collect()
torch.cuda.empty_cache()
log('glm4 unloaded; cuda cache cleared')

# ================================================================
# dumps
# ================================================================
runtime = time.time() - T0
verdict = '|'.join([
    'a_3130_ok', win_v, swap_v, single_v,
    a1_v, fork_v, 'coverage_full'])
log('VERDICT: %s' % verdict)
result = {
    'name': NAME,
    'phase': 3131,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'res30_sha8': sha30,
        'asserts_3130': 'ok'},
    'part_b': {
        'layers': list(LAYER_WIN_Q),
        'win_flip': [float(win_flip[l])
                     for l in LAYER_WIN_Q],
        'win_med_dm': [float(win_med[l])
                       for l in LAYER_WIN_Q],
        'l24_anchor_ok': bool(l24_ok),
        'delta_q': delta_q,
        'n_win': int(n_win)},
    'part_c': {
        'full_swap_chg': gen_chg_new,
        'full_swap_first':
            int(gen_first_new),
        'full_swap_bitexact_3130':
            bool(swap_bitexact),
        'sel64_sha8': sel_sha8,
        'chg40': [float(v) for v in c40],
        'first40': [int(v) for v in f40],
        'peak40': int(peak40),
        'dominance40': float(dom40),
        'soft_dev_3130': soft_dev,
        'chg_a1': float(chg_a1),
        'first_a1': int(gen_first_a1),
        'hist_a1': [int(v)
                    for v in hist_a1],
        'pk_dm': int(pk_dm),
        'pk_dm_first': int(pk_first),
        'n_first': int(n_first),
        'med_abs_dm': [float(v) for v in
                       med_abs_dm],
        'med_dm_first': [float(v) for v in
                         med_dm_first],
        'med_dm_rest': [float(v) for v in
                        med_dm_rest]},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'win_ks': np.asarray(LAYER_WIN_Q,
                         dtype=np.int64),
    'win_flip': np.asarray(
        [win_flip[l] for l in
         LAYER_WIN_Q],
        dtype=np.float64),
    'win_med_dm': np.asarray(
        [win_med[l] for l in
         LAYER_WIN_Q],
        dtype=np.float64),
    'gen_same_p': gen_same_new.astype(
        np.int64),
    'gen_same_a1': gen_same_a1.astype(
        np.int64),
    'hist_a1': hist_a1,
    'sel': sel.astype(np.int64),
    'chg40': c40,
    'first40': f40,
    'dm': dm.astype(np.float32),
    'med_abs_dm': med_abs_dm.astype(
        np.float64),
    'med_dm_first': med_dm_first.astype(
        np.float64),
    'med_dm_rest': med_dm_rest.astype(
        np.float64)}
np.savez(os.path.join(OUT,
                      'p129_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p129_readout.npz (%d keys)'
    % len(npz_out))
log('P3131 DONE (%.1fs)'
    % (time.time() - T0))
