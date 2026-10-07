# -*- coding: utf-8 -*-
"""Phase 3128 (Omega-P126): joint multi-layer swap +
coordinate-level sparse injection + direction-by-
condition interaction localization + s0 probe with
matched batch composition.

Preregistered in 3127 MEMO section 5 (frozen before
any observation):
  B gpu qwen3-4b: joint swap W5 {26,28,30,32,34} vs
    C5 {2,8,14,4,10} (gate 2.0x); coordinate
    injection at layer-24 output (top-K |rho| coords
    from 3113 capture_b L24 hs x margin, K in
    {5,50,200}, +-delta, random-50 control).
  C gpu glm4-9b: joint swap W4 {8,9,13,29} vs C4
    {4,14,34,24}; L20-output hs fit on odd half ->
    top-50 +-delta on even half; s0 probe: first 32
    prompts of P in ONE batch-32 (composition
    identical to 3126 gen_base first batch) ->
    token-exact assert (formal run).
  A offline: 3127 gates recompute (f32 path,
    1e-12); interaction localization: GLM4
    dmg_final field x regen flip labels per layer
    per condition (needs tok_g -> runs after C).
"""
import gc
import io
import json
import os
import random as _rnd
import time
import zlib

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p126_joint_swap_coord_' \
       'inject_interaction_s0match'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
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
MDIR_Q = os.path.join(ROOT, 'models', 'hf',
                      'qwen3-4b')
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3128', NAME)
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


log('P3128 start SMOKE=%s' % SMOKE)

# ---------------- constants ----------------
NP = 672
N_NEW = 12
NP_B = 8 if SMOKE else NP
NP_REPRO = 4 if SMOKE else 16
NP_CHECK = 4 if SMOKE else 16
GEN_BATCH = 4 if SMOKE else 32
PERM_SEED = 3128
SCOND3 = ('s1', 's2', 's3')
DIRS = ('P', 'A1')

JOINT_Q = {'write': [26, 28, 30, 32, 34],
           'ctrl': [2, 8, 14, 4, 10]}
JOINT_G = {'write': [8, 9, 13, 29],
           'ctrl': [4, 14, 34, 24]}
JOINT_GATE = 2.0

CB_LAYERS = [12, 20, 24, 28, 32]
INJ_LAYER_Q = 24
K_SET_Q = (5, 50, 200)
K_RANDOM_CTRL = 50
INJ_FRAC = 0.10
INJ_LAYER_G = 20
K_SET_G = (50,)
HALF_FIT = 4 if SMOKE else 336

V_GATE = 0.30
V_SEP = 0.15

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
res26 = json.load(io.open(
    D26 + r'\result.json', encoding='utf-8'))
assert res26['smoke'] is False
res27 = json.load(io.open(
    D27 + r'\result.json', encoding='utf-8'))
assert res27['smoke'] is False
assert len(res27['verdict'].split('|')) == 19
log('frozen inputs ok (05/13/18/25/26/27)')

# ---------------- seal ----------------
seal = {
    'phase': 3128,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'np_a': NP, 'np_b': NP_B,
    'perm_seed': PERM_SEED,
    'data_sources': {
        'margins_annotation': 'phase3118+3120',
        'swap_field_3127': 'p125 dmq/dmg + '
                           'regen tokens',
        'coords_qwen': 'phase3113 capture_b '
                       'h_out L24 x m -> '
                       'per-coord Spearman',
        'coords_glm4': 'live L20-output hs, '
                       'odd-half fit',
        'gen_ref': 'p124 gen_base (batch-32 '
                   'x21)'},
    'part_b': {
        'joint_swap': 'swap_layers list hook;'
                      ' W5 %s vs C5 %s; gate '
                      'median|dM| ratio >= %s'
                      % (JOINT_Q['write'],
                         JOINT_Q['ctrl'],
                         JOINT_GATE),
        'coord_inject': 'hook at layers[%d] '
                        'output, LAST position '
                        'only: h[coords] += '
                        'delta*sgn; delta=%s*'
                        'mean_row_L2(capture_b '
                        'L24); K in %s + random'
                        '-%d (rng 3128); flip = '
                        'sign(m_inj)!=sign(m_'
                        'base frozen)'
                        % (INJ_LAYER_Q,
                           INJ_FRAC, K_SET_Q,
                           K_RANDOM_CTRL)},
    'part_c': {
        'joint_swap': 'W4 %s vs C4 %s same '
                      'gate' % (JOINT_G['write'],
                                JOINT_G['ctrl']),
        'coord_inject': 'L%d-output hs fit on '
                        'odd half (%d) -> top-%s '
                        'coords, even half +-delta'
                        % (INJ_LAYER_G, HALF_FIT,
                           K_SET_G),
        's0_probe': 'first %d P prompts of '
                    '3126-identical PID_T in ONE'
                    ' batch-%d -> token-exact vs '
                    'gen_base[0:n] (formal); '
                    'agree>=0.95 (smoke)'
                    % (GEN_BATCH, GEN_BATCH)},
    'part_d': {
        'interaction': 'flip labels from p125 '
                       'regen (pol_raw vs gen_base'
                       '); per-layer Spearman(|dmg'
                       '_final|, flip); interaction'
                       ' profile = rho(P,s1) - rho'
                       '(A1,s3); gates r>=%s '
                       'sep>=%s' % (V_GATE, V_SEP)}}
with io.open(os.path.join(OUT,
                          'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('seal frozen (%s)' % seal['created'])

# ================================================================
# PART A0: offline recompute of 3127 gates (f32 path)
# ================================================================
log('== PART A0: 3127 recompute ==')
LAY_Q27 = {'write': [26, 28, 30, 32, 34],
           'port': [20, 21],
           'ctrl': [2, 8, 14]}
LAY_G27 = {'write': [8, 9, 13, 29],
           'port': [20],
           'ctrl': [4, 14, 34]}
DIRS = ('P', 'A1')


def f32_dm(zf, zbase, side, dc, l, NL):
    # main-script path: f32 subtraction then
    # exact widen to f64 (median runs in f64)
    return (zf['%s_%s_L%02d' % (side, dc, l)]
            [:, -1]
            - zbase['mlg_s0_%s' % dc][:NP,
                                      NL, -1]
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
pb27 = res27['part_b']
pc27 = res27['part_c']
a_repl_ok = (
    abs(gq_r['write'] - pb27['gates']['write'])
    < 1e-12
    and abs(gq_r['port'] - pb27['gates']['port'])
    < 1e-12
    and abs(mcq_r - pb27['ctrl_median']) < 1e-12
    and abs(gg_r['write'] - pc27['gates']['write'])
    < 1e-12
    and abs(gg_r['port'] - pc27['gates']['port'])
    < 1e-12
    and abs(mcg_r - pc27['ctrl_median']) < 1e-12)
assert a_repl_ok, (gq_r, gg_r)
a_repl_v = 'repl_3127_ok'
log('A0 repl gates ok: q %s g %s'
    % ({k: round(v, 4)
        for k, v in gq_r.items()},
       {k: round(v, 4)
        for k, v in gg_r.items()}))

# ================================================================
# materials factory (3125/3126/3127-identical logic)
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
# PART B: GPU qwen3-4b joint swap + coord inject
# ================================================================
log('== PART B: qwen joint + inject ==')
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
                    swap_layers=None,
                    inject=None):
    """3125 semantics: norm ONLY L<NL.
    swap_layers: list of layer indices whose
    output := input (in-place).
    inject: (layer, coords np, delta float,
    sgn) -> h[last_pos, coords] += delta*sgn
    at that layer OUTPUT."""
    traj_ids = [int(x) for x in traj_ids]
    ids = list(prompt_ids) + traj_ids
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(traj_ids) + 1
    ml = np.zeros((NLQ + 1, npts),
                  dtype=np.float64)
    hooks = []
    if swap_layers:
        for l in swap_layers:
            lyr = model_q.model.layers[l]

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
        (il, coords, delta, sgn) = inject
        lyr = model_q.model.layers[il]
        co_t = torch.as_tensor(
            np.asarray(coords, dtype=np.int64),
            device='cuda')
        dv_t = torch.full((len(coords),),
                          float(delta) * float(sgn),
                          device='cuda',
                          dtype=torch.bfloat16)

        def _inj(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            o2[0, -1, co_t] += dv_t
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


# repro vs frozen p123 (pipeline validity)
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

# base final margins (frozen) + s0 prompts
base_m_q = {
    dc: z25['mlg_s0_%s' % dc][:NP_B, NLQ, -1]
        .astype(np.float64)
    for dc in DIRS}
s0_pids_q = {dc: [matq['build_pids'](dc, j)[0]
                  for j in range(NP_B)]
             for dc in DIRS}


def run_joint_q(layers):
    dm = {}
    for dc in DIRS:
        store = np.zeros((NP_B, N_NEW + 1),
                         dtype=np.float64)
        for j in range(NP_B):
            wl = forward_track_q(
                s0_pids_q[dc][j]['s0'],
                matq['traj_tokens'](dc, j)[2],
                swap_layers=layers)
            store[j] = wl[NLQ, :]
        dm[dc] = store[:, -1] - base_m_q[dc]
    return dm


jres_q = {}
for grp in ('write', 'ctrl'):
    dm = run_joint_q(JOINT_Q[grp])
    jres_q[grp] = dm
    log('B joint %s done (%.1fs)'
        % (grp, time.time() - T0))
med_cq = float(np.median(np.abs(
    np.concatenate([jres_q['ctrl']['P'],
                    jres_q['ctrl']['A1']]))))
med_wq = float(np.median(np.abs(
    np.concatenate([jres_q['write']['P'],
                    jres_q['write']['A1']]))))
joint_ratio_q = med_wq / max(med_cq, 1e-12)
q_joint_v = ('qwen_joint_redundant'
             if joint_ratio_q >= JOINT_GATE
             else 'qwen_joint_independent')
log('B-JOINT: ctrl_med=%.4f write_med=%.4f '
    'ratio=%.3f -> %s'
    % (med_cq, med_wq, joint_ratio_q,
       q_joint_v))

# ---- coordinate injection (capture_b) ----
h_out = capb['h_out'].astype(np.float32)
m_cap = capb['m'].astype(np.float64)
li = CB_LAYERS.index(INJ_LAYER_Q)
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
order_rho = np.argsort(
    -np.abs(rho_cap))
norms = np.sqrt((hl.astype(np.float64)
                 * hl.astype(np.float64))
                .sum(1))
delta_q = INJ_FRAC * float(norms.mean())
log('B delta_q=%.4f (mean row L2 of '
    'capture_b L%d)' % (delta_q, INJ_LAYER_Q))
rng_i = np.random.default_rng(PERM_SEED)
rand_coords = rng_i.choice(
    2560, size=K_RANDOM_CTRL, replace=False)


def inject_run_q(coords, sgn):
    fl = []
    dmm = []
    for dc in DIRS:
        for j in range(NP_B):
            wl = forward_track_q(
                s0_pids_q[dc][j]['s0'],
                matq['traj_tokens'](dc, j)[2],
                inject=(INJ_LAYER_Q, coords,
                        delta_q, sgn))
            m_inj = wl[NLQ, -1]
            m_b = base_m_q[dc][j]
            fl.append(int(
                (m_inj > 0) != (m_b > 0)))
            dmm.append(abs(m_inj - m_b))
    return float(np.mean(fl)), \
        float(np.median(dmm))


coord_q = {}
for K in K_SET_Q:
    co = order_rho[:K]
    fp, mp = inject_run_q(co, +1)
    fm, mm = inject_run_q(co, -1)
    coord_q['top%d' % K] = {
        'flip': (fp + fm) / 2.0,
        'med_dm': (mp + mm) / 2.0,
        'flip_p': fp, 'flip_m': fm}
    log('B inject top%d: flip=%.4f '
        'med|dm|=%.4f' % (K, (fp + fm) / 2.0,
                          (mp + mm) / 2.0))
co_r = rand_coords
fp, mp = inject_run_q(co_r, +1)
fm, mm = inject_run_q(co_r, -1)
coord_q['rand50'] = {
    'flip': (fp + fm) / 2.0,
    'med_dm': (mp + mm) / 2.0}
log('B inject rand50: flip=%.4f med|dm|=%.4f'
    % ((fp + fm) / 2.0, (mp + mm) / 2.0))
fr = [coord_q['top%d' % K]['flip']
      for K in K_SET_Q]
q_coord_eff = (
    coord_q['top200']['flip']
    >= 2.0 * max(coord_q['rand50']['flip'],
                 1e-9)
    and coord_q['top200']['flip'] >= 0.05)
q_coord_v = ('qwen_coord_effective'
             if q_coord_eff
             else 'qwen_coord_ineffective')
dose_mono = (fr[2] >= fr[1] - 0.01
             and fr[1] >= fr[0] - 0.01)
q_dose_v = ('qwen_coord_dose_monotonic'
            if dose_mono
            else 'qwen_coord_dose_flat')
log('B-COORD: %s | %s (flips %s rand %.4f)'
    % (q_coord_v, q_dose_v,
       ['%.3f' % v for v in fr],
       coord_q['rand50']['flip']))

del model_q
gc.collect()
torch.cuda.empty_cache()
log('qwen unloaded; cuda cache cleared')

# ================================================================
# PART C: GPU glm4 joint swap + L20 inject + s0
# ================================================================
log('== PART C: glm4 joint + inject + s0 ==')
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
    collect_layer: also return that layer
    OUTPUT last-position vector."""
    traj_ids = [int(x) for x in traj_ids]
    ids = list(prompt_ids) + traj_ids
    t_in = torch.tensor([ids], device='cuda')
    pos0 = len(prompt_ids) - 1
    npts = len(traj_ids) + 1
    ml = np.zeros((NLG + 1, npts),
                  dtype=np.float64)
    lm = None
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
        (il, coords, delta, sgn) = inject
        lyr = model_g.model.layers[il]
        co_t = torch.as_tensor(
            np.asarray(coords, dtype=np.int64),
            device='cuda')
        dv_t = torch.full((len(coords),),
                          float(delta) * float(sgn),
                          device='cuda',
                          dtype=torch.bfloat16)

        def _inj(mod, inp, out):
            o2 = out[0] \
                if isinstance(out, tuple) \
                else out
            o2[0, -1, co_t] += dv_t
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

base_m_g = {
    dc: z26['mlg_s0_%s' % dc][:NP_B, NLG, -1]
        .astype(np.float64)
    for dc in DIRS}
s0_pids_g = {dc: [matg['build_pids'](dc, j)[0]
                  for j in range(NP_B)]
             for dc in DIRS}


def run_joint_g(layers):
    dm = {}
    for dc in DIRS:
        store = np.zeros((NP_B, N_NEW + 1),
                         dtype=np.float64)
        for j in range(NP_B):
            wl, _cv = forward_track_g(
                s0_pids_g[dc][j]['s0'],
                matg['traj_tokens'](dc, j)[2],
                swap_layers=layers)
            store[j] = wl[NLG, :]
        dm[dc] = store[:, -1] - base_m_g[dc]
    return dm


jres_g = {}
for grp in ('write', 'ctrl'):
    jres_g[grp] = run_joint_g(JOINT_G[grp])
    log('C joint %s done (%.1fs)'
        % (grp, time.time() - T0))
med_cg = float(np.median(np.abs(
    np.concatenate([jres_g['ctrl']['P'],
                    jres_g['ctrl']['A1']]))))
med_wg = float(np.median(np.abs(
    np.concatenate([jres_g['write']['P'],
                    jres_g['write']['A1']]))))
joint_ratio_g = med_wg / max(med_cg, 1e-12)
g_joint_v = ('glm4_joint_redundant'
             if joint_ratio_g >= JOINT_GATE
             else 'glm4_joint_independent')
log('C-JOINT: ctrl_med=%.4f write_med=%.4f '
    'ratio=%.3f -> %s'
    % (med_cg, med_wg, joint_ratio_g,
       g_joint_v))

# ---- GLM4 L20 hs: odd-half fit, even inject --
half_a = list(range(0, NP, 2))[:HALF_FIT]
half_b = list(range(1, NP, 2))[:HALF_FIT]
D_G = 4096
hs_fit = np.zeros((HALF_FIT, D_G),
                  dtype=np.float64)
m_fit = np.zeros(HALF_FIT, dtype=np.float64)
for jj, j in enumerate(half_a):
    pids, _sp = matg['build_pids']('P', j)
    _ptxt, _poffs, base = \
        matg['traj_tokens']('P', j)
    wl, cv = forward_track_g(
        pids['s0'], base,
        collect_layer=INJ_LAYER_G)
    hs_fit[jj] = cv
    m_fit[jj] = wl[NLG, -1]
    if jj % 64 == 0:
        log('C collect %d/%d (%.1fs)'
            % (jj, HALF_FIT, time.time()
               - T0))
hr2 = np.argsort(np.argsort(hs_fit, axis=0),
                 axis=0).astype(np.float64)
mr2 = np.argsort(np.argsort(m_fit)).astype(
    np.float64)
hc2 = hr2 - hr2.mean(0, keepdims=True)
mc3 = mr2 - mr2.mean()
den2 = np.sqrt((hc2 * hc2).sum(0)
               * float((mc3 * mc3).sum()))
rho_g = (hc2 * mc3[:, None]).sum(0) / den2
order_g = np.argsort(-np.abs(rho_g))
coords_g = order_g[:K_SET_G[0]]
hv20 = hs_fit
delta_g = INJ_FRAC * float(
    np.sqrt((hv20 * hv20).sum(1)).mean())
log('C delta_g=%.4f top50 fitted on %d odd'
    ' records' % (delta_g, HALF_FIT))


def inject_run_g(coords, sgn):
    fl = []
    dmm = []
    for j in half_b:
        pids, _sp = matg['build_pids']('P', j)
        _ptxt, _poffs, base = \
            matg['traj_tokens']('P', j)
        wl, _cv = forward_track_g(
            pids['s0'], base,
            inject=(INJ_LAYER_G, coords,
                    delta_g, sgn))
        m_inj = wl[NLG, -1]
        m_b = float(
            z26['mlg_s0_P'][j, NLG, -1])
        fl.append(int(
            (m_inj > 0) != (m_b > 0)))
        dmm.append(abs(m_inj - m_b))
    return float(np.mean(fl)), \
        float(np.median(dmm))


coord_g = {}
for K in K_SET_G:
    fp, mp = inject_run_g(coords_g[:K], +1)
    fm, mm = inject_run_g(coords_g[:K], -1)
    coord_g['top%d' % K] = {
        'flip': (fp + fm) / 2.0,
        'med_dm': (mp + mm) / 2.0}
    log('C inject top%d: flip=%.4f '
        'med|dm|=%.4f' % (K, (fp + fm) / 2.0,
                          (mp + mm) / 2.0))
g_coord_v = ('glm4_coord_effective'
             if coord_g['top50']['flip']
             >= 0.05
             else 'glm4_coord_ineffective')
log('C-COORD: %s' % g_coord_v)

# cross-model coord note: Qwen dose (3 K) vs
# GLM4 single-K are descriptive only this
# phase (different fit sources); verdict
# carries each model's own coord segment.
coord_v = 'coord_cross_descriptive'

# ---- s0 probe: batch-32 matched composition --
n_probe = GEN_BATCH
probe_ids = [matg['pids_all']['P'][j]['s0']
             for j in range(n_probe)]
probe_out = []
chunk = [list(PREFIX_IDS) + list(p)
         for p in probe_ids]
maxlen = max(len(p) for p in chunk)
ids_p = np.full((len(chunk), maxlen), PADG,
                dtype=np.int64)
mask_p = np.zeros((len(chunk), maxlen),
                  dtype=np.int64)
for i, p in enumerate(chunk):
    ids_p[i, maxlen - len(p):] = p
    mask_p[i, maxlen - len(p):] = 1
t_ids = torch.tensor(ids_p, device='cuda')
t_mask = torch.tensor(mask_p, device='cuda')
with torch.inference_mode():
    outg = model_g.generate(
        input_ids=t_ids,
        attention_mask=t_mask,
        max_new_tokens=N_NEW,
        do_sample=False, num_beams=1,
        pad_token_id=PADG)
newg = outg[:, maxlen:]
for row in newg:
    ids_r = [int(t) for t in row]
    if PADG in ids_r:
        ids_r = ids_r[:ids_r.index(PADG)]
    for e in model_g.config.eos_token_id \
            if isinstance(
                model_g.config.eos_token_id,
                list) else [
        model_g.config.eos_token_id]:
        if e in ids_r:
            ids_r = ids_r[:ids_r.index(e)]
            break
    probe_out.append(ids_r)
agree_tok = 0
bit_mism = 0
for j in range(n_probe):
    b12 = [int(v) for v in
           z26['gen_base_P'][j]]
    g12 = list(probe_out[j])[:N_NEW]
    g12 = g12 + [DOT_G] * (N_NEW - len(g12))
    b12 = b12 + [DOT_G] * (N_NEW - len(b12))
    agree_tok += sum(int(a == b)
                     for a, b in zip(g12, b12))
    bit_mism += int(g12 != b12)
s0_agree = agree_tok / float(n_probe * N_NEW)
if not SMOKE:
    assert bit_mism == 0, bit_mism
    s0_v = 's0_probe_matched'
    log('C-S0PROBE: bit-exact vs gen_base '
        '[0:%d] (batch-%d matched)'
        % (n_probe, GEN_BATCH))
else:
    s0_v = ('s0_probe_matched'
            if s0_agree >= 0.95
            else 's0_probe_drift')
    log('C-S0PROBE: agree %.4f (smoke, '
        'non-matched batch)' % s0_agree)

del model_g
gc.collect()
torch.cuda.empty_cache()
log('glm4 unloaded; cuda cache cleared')

# ================================================================
# PART D: interaction localization (offline, tok_g)
# ================================================================
log('== PART D: interaction localization ==')
dmg_final = {}
for grp in ('write', 'port', 'ctrl'):
    for l in LAY_G27[grp]:
        for dc in DIRS:
            key = '%s_L%02d' % (dc, l)
            dmg_final[key] = (
                z27['dmg_' + key][:, -1]
                - z26['mlg_s0_%s' % dc][:NP,
                                        NLG, -1])


def pol_raw(tok_id):
    if not tok_id:
        return (0, 0)
    t = tok_g.decode([int(tok_id)])
    ts = t.strip()
    tl = ts.lower()
    if tl.startswith('yes'):
        return (1, 0)
    if tl.startswith('no'):
        return (-1,
                1 if ts.startswith('No')
                else 0)
    return (0, 0)


flip_lab = {}
for dc in DIRS:
    for c in SCOND3:
        fl = np.zeros(NP, dtype=np.int64)
        reg = z27['regen_%s_%s' % (c, dc)]
        for j in range(NP):
            b12 = [int(v) for v in
                   z26['gen_base_%s' % dc][j]]
            bp, _cap = pol_raw(b12[0])
            rp, _cap2 = pol_raw(reg[j, 0])
            fl[j] = int(rp != bp)
        flip_lab[(c, dc)] = fl


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(
        np.float64)
    rb = np.argsort(np.argsort(b)).astype(
        np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


rho_prof = {}
for c in SCOND3:
    for dc in DIRS:
        for l in (LAY_G27['write']
                  + LAY_G27['port']
                  + LAY_G27['ctrl']):
            key = '%s_L%02d' % (dc, l)
            rho_prof['%s_%s_L%02d'
                     % (c, dc, l)] = spearman(
                np.abs(dmg_final[key]),
                flip_lab[(c, dc)])
prof_P = np.array([rho_prof['s1_P_L%02d' % l]
                   for l in LAY_G27['write']])
prof_A = np.array([rho_prof['s3_A1_L%02d' % l]
                   for l in LAY_G27['write']])
diff_prof = prof_P - prof_A
peak_i = int(np.argmax(np.abs(diff_prof)))
peak_layer = LAY_G27['write'][peak_i]
peak_sep = float(np.abs(diff_prof[peak_i]))
peak_r = float(max(abs(prof_P[peak_i]),
                   abs(prof_A[peak_i])))
interact_v = ('interaction_found'
              if (peak_r >= V_GATE
                  and peak_sep >= V_SEP)
              else 'interaction_not_in_field')
log('D-INTERACT: peak layer L%02d sep=%.3f '
    'r=%.3f -> %s' % (peak_layer, peak_sep,
                      peak_r, interact_v))

# ---------------- verdict + dumps -------------
verdict = '|'.join([
    a_repl_v,
    interact_v,
    q_joint_v,
    q_coord_v,
    q_dose_v,
    g_joint_v,
    g_coord_v,
    coord_v,
    s0_v,
    'coverage_full'])
assert len(verdict.split('|')) == 10
result = {
    'phase': 3128,
    'name': NAME,
    'smoke': SMOKE,
    'runtime_s': time.time() - T0,
    'verdict': verdict,
    'part_a': {'repl_gates_q': gq_r,
               'repl_gates_g': gg_r},
    'part_b': {
        'joint_ratio_q': joint_ratio_q,
        'med_ctrl': med_cq,
        'med_write': med_wq,
        'delta_q': delta_q,
        'coord_q': coord_q,
        'rho_top200_L24': [
            float(rho_cap[i])
            for i in order_rho[:200]],
        'repro_max_abs': drep_q},
    'part_c': {
        'joint_ratio_g': joint_ratio_g,
        'med_ctrl': med_cg,
        'med_write': med_wg,
        'delta_g': delta_g,
        'coord_g': coord_g,
        's0_agree': s0_agree,
        's0_bit_mism': bit_mism,
        'repro_max_abs': drep_g},
    'part_d': {
        'rho_prof': rho_prof,
        'peak_layer': peak_layer,
        'peak_sep': peak_sep,
        'peak_r': peak_r}}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)
npz_out = {}
for grp in ('write', 'ctrl'):
    for dc in DIRS:
        npz_out['joint_%s_%s' % (grp, dc)] = \
            jres_q[grp][dc].astype(np.float32)
        npz_out['jointg_%s_%s' % (grp, dc)] = \
            jres_g[grp][dc].astype(np.float32)
npz_out['rho_cap_L24'] = rho_cap.astype(
    np.float32)
npz_out['rho_g_L20'] = rho_g.astype(np.float32)
npz_out['coords_g_top50'] = coords_g.astype(
    np.int64)
npz_out['flip_lab_s1_P'] = flip_lab[
    ('s1', 'P')]
npz_out['flip_lab_s3_A1'] = flip_lab[
    ('s3', 'A1')]
np.savez(os.path.join(OUT,
                      'p126_readout.npz'),
         **npz_out)
log('VERDICT: %s' % verdict)
log('dumps done: result.json + '
    'p126_readout.npz (%d keys)'
    % (len(npz_out) + 1))
log('P3128 DONE (%.1fs)'
    % (time.time() - T0))
