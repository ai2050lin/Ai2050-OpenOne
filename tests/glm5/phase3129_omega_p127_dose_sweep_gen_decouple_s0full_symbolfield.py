# -*- coding: utf-8 -*-
"""Phase 3129 (Omega-P127): dose-response sweep +
mid-position injection + margin-generation decouple
+ s0 full 21-batch determinism + regen teacher-
forced margin sign-field interaction localization.

Preregistered in 3128 MEMO section 5 (frozen
before any observation):
  A offline: 3128 hard-value asserts + 3127 gates
    f32 recompute (1e-12).
  B qwen3-4b: dose sweep K=50 top-rho coords,
    frac in {0.05,0.10,0.20} x +-sgn, 672x2
    (frac=0.10 must reproduce 3128 top50);
    mid-position injection (traj token 6).
  C glm4-9b: A1-direction coord inject (even half,
    3128 top50 coords, frac=0.10); W4 joint swap
    generation 672 vs gen_base (margin-generation
    decouple); s0 full 21-batch bit-exact.
  D glm4 (before unload): regen teacher-forced
    margin sign-field, 4 groups x 672, per-layer
    phi profile, peak sep gate 0.30/0.15.
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
NAME = 'omega_p127_dose_sweep_gen_' \
       'decouple_s0full_symbolfield'
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
D28 = RDIR + r'\phase3128' \
      r'\omega_p126_joint_swap_coord_' \
      r'inject_interaction_s0match'
MDIR_Q = os.path.join(ROOT, 'models', 'hf',
                      'qwen3-4b')
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3129', NAME)
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


log('P3129 start SMOKE=%s' % SMOKE)

# ---------------- constants ----------------
NP = 672
N_NEW = 12
NP_B = 8 if SMOKE else NP
NP_REPRO = 4 if SMOKE else 16
GEN_BATCH = 4 if SMOKE else 32
PERM_SEED = 3129
DIRS = ('P', 'A1')

INJ_LAYER_Q = 24
K_DOSE = 50
DOSE_FRACS = (0.10,) if SMOKE \
    else (0.05, 0.10, 0.20)
MIDK = 6
DOSE_TOL = 0.01
MID_MIN = 0.05
MID_RATIO = 0.5

INJ_LAYER_G = 20
A1_MIN = 0.05
JOINT_G = {'write': [8, 9, 13, 29]}
GEN_CHANGE_GATE = 0.05

V_GATE = 0.30
V_SEP = 0.15
GROUPS_D = (('s1', 'P'), ('s1', 'A1'),
            ('s3', 'P'), ('s3', 'A1'))

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
res27 = json.load(io.open(
    D27 + r'\result.json', encoding='utf-8'))
assert res27['smoke'] is False
res28 = json.load(io.open(
    D28 + r'\result.json', encoding='utf-8'))
assert res28['smoke'] is False
z28 = np.load(D28 + r'\p126_readout.npz',
              allow_pickle=False)
log('frozen inputs ok (05/13/18/25/26/27/28)')

# ================================================================
# PART A: offline 3128 link asserts + gates recompute
# ================================================================
log('== PART A: 3128 link asserts ==')
V28 = ('repl_3127_ok|interaction_not_in_field'
       '|qwen_joint_independent'
       '|qwen_coord_effective'
       '|qwen_coord_dose_monotonic'
       '|glm4_joint_independent'
       '|glm4_coord_effective'
       '|coord_cross_descriptive'
       '|s0_probe_matched|coverage_full')
assert res28['verdict'] == V28
pb28 = res28['part_b']
pc28 = res28['part_c']
assert abs(pb28['joint_ratio_q']
           - 0.8969961609372126) < 1e-9
assert abs(pc28['joint_ratio_g']
           - 0.8957927785460357) < 1e-9
cq28 = pb28['coord_q']
assert abs(cq28['top50']['flip']
           - 0.18229166666666666) < 1e-12
assert abs(cq28['top200']['flip']
           - 0.328125) < 1e-12
assert abs(cq28['rand50']['flip']
           - 0.09523809523809523) < 1e-12
assert pc28['s0_bit_mism'] == 0
assert abs(pb28['delta_q']
           - 10.694569431733811) < 1e-9
assert abs(pc28['delta_g']
           - 0.9953634794440709) < 1e-9
log('A hard asserts ok (3128 frozen values)')

LAY_Q27 = {'write': [26, 28, 30, 32, 34],
           'port': [20, 21],
           'ctrl': [2, 8, 14]}
LAY_G27 = {'write': [8, 9, 13, 29],
           'port': [20],
           'ctrl': [4, 14, 34]}


def f32_dm(zf, zbase, side, dc, l, NL):
    return (zf['%s_%s_L%02d' % (side, dc, l)]
            [:, -1]
            - zbase['mlg_s0_%s' % dc][:NP, NL, -1]
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
pa28 = res28['part_a']
a_ok = (
    abs(gq_r['write']
        - pa28['repl_gates_q']['write']) < 1e-12
    and abs(gq_r['port']
            - pa28['repl_gates_q']['port'])
    < 1e-12
    and abs(gg_r['write']
            - pa28['repl_gates_g']['write'])
    < 1e-12
    and abs(gg_r['port']
            - pa28['repl_gates_g']['port'])
    < 1e-12)
assert a_ok, (gq_r, gg_r)
a_v = 'a_3128_ok'
log('A gates recompute 1e-12 ok: q %s g %s'
    % ({k: round(v, 4)
        for k, v in gq_r.items()},
       {k: round(v, 4)
        for k, v in gg_r.items()}))

# ================================================================
# materials factory (3125/3126/3127/3128-identical)
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
# PART B: GPU qwen3-4b dose sweep + mid injection
# ================================================================
log('== PART B: qwen dose + mid inject ==')
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

# ---- dose sweep (capture_b top-50 coords) ----
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
order_rho = np.argsort(-np.abs(rho_cap))
norms = np.sqrt((hl.astype(np.float64)
                 * hl.astype(np.float64))
                .sum(1))
mean_row_l2 = float(norms.mean())
log('B mean_row_L2=%.4f (L%d)'
    % (mean_row_l2, INJ_LAYER_Q))
co50 = order_rho[:K_DOSE]


def inject_run_q(coords, frac, sgn,
                 mid=False):
    delta = frac * mean_row_l2
    midk = None
    if mid:
        midk = MIDK
    fl = []
    dmm = []
    for dc in DIRS:
        for j in range(NP_B):
            wl = forward_track_q(
                s0_pids_q[dc][j]['s0'],
                matq['traj_tokens'](dc, j)[2],
                inject=(INJ_LAYER_Q, coords,
                        delta, sgn, midk))
            m_inj = wl[NLQ, -1]
            m_b = base_m_q[dc][j]
            fl.append(int(
                (m_inj > 0) != (m_b > 0)))
            dmm.append(abs(m_inj - m_b))
    return float(np.mean(fl)), \
        float(np.median(dmm))


dose_q = {}
for frac in DOSE_FRACS:
    fp, mp = inject_run_q(co50, frac, +1)
    fm, mm = inject_run_q(co50, frac, -1)
    dose_q['f%02d' % round(frac * 100)] = {
        'flip': (fp + fm) / 2.0,
        'med_dm': (mp + mm) / 2.0,
        'flip_p': fp, 'flip_m': fm}
    log('B dose %.2f: flip=%.4f med|dm|=%.4f'
        % (frac, (fp + fm) / 2.0,
           (mp + mm) / 2.0))

fr = [dose_q['f%02d' % round(f * 100)]['flip']
      for f in DOSE_FRACS]
dose_mono = all(
    fr[i + 1] >= fr[i] - DOSE_TOL
    for i in range(len(fr) - 1)) \
    and fr[-1] > fr[0]
dose_v = ('qwen_dose_monotonic'
          if dose_mono
          else 'qwen_dose_flat')
if not SMOKE:
    ref50 = cq28['top50']['flip']
    ref_dm = cq28['top50']['med_dm']
    f10 = dose_q['f10']['flip']
    d10 = dose_q['f10']['med_dm']
    assert abs(f10 - ref50) < 1e-9, (f10,
                                     ref50)
    assert abs(d10 - ref_dm) < 1e-9, (d10,
                                      ref_dm)
    log('B dose 0.10 reproduces 3128 top50 '
        'bit-level (flip %.6f)' % f10)

fp, mp = inject_run_q(co50, 0.10, +1,
                      mid=True)
fm, mm = inject_run_q(co50, 0.10, -1,
                      mid=True)
mid_flip = (fp + fm) / 2.0
mid_dm = (mp + mm) / 2.0
last_flip = dose_q['f10']['flip']
mid_ok = (mid_flip >= MID_MIN
          and mid_flip >= MID_RATIO
          * max(last_flip, 1e-9))
mid_v = ('qwen_midprop_yes' if mid_ok
         else 'qwen_midprop_no')
log('B-MID: mid flip=%.4f last flip=%.4f '
    '-> %s' % (mid_flip, last_flip, mid_v))

del model_q
gc.collect()
torch.cuda.empty_cache()
log('qwen unloaded; cuda cache cleared')

# ================================================================
# PART C: GPU glm4 A1 inject + gen decouple + s0 full
# ================================================================
log('== PART C: glm4 a1 + decouple + s0 ==')
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

# ---- C1: GLM4 A1-direction coord inject ----
# Reuse 3128 frozen top50 coords + delta_g
# directly (comparable with P-side 0.1399).
coords_g = z28['coords_g_top50']
delta_g = float(res28['part_c']['delta_g'])
half_b = list(range(1, NP, 2))[:NP_B // 2] \
    if SMOKE else list(range(1, NP, 2))


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


a1_res = {}
for sgn in (+1, -1):
    fp, mp = inject_run_g_a1(coords_g,
                             delta_g, sgn)
    a1_res['flip_s%d' % (1 if sgn > 0
                         else -1)] = fp
    a1_res['dm_s%d' % (1 if sgn > 0
                       else -1)] = mp
    log('C a1 inject sgn%+d: flip=%.4f '
        'med|dm|=%.4f' % (sgn, fp, mp))
a1_flip = (a1_res['flip_s1']
           + a1_res['flip_s-1']) / 2.0
a1_dm = (a1_res['dm_s1']
         + a1_res['dm_s-1']) / 2.0
a1_ok = a1_flip >= A1_MIN
a1_v = ('glm4_coord_a1_effective' if a1_ok
        else 'glm4_coord_a1_ineffective')
log('C-A1COORD: flip=%.4f (P-side ref '
    '%.4f) -> %s' % (a1_flip,
                     res28['part_c']
                     ['coord_g']['top50']
                     ['flip'], a1_v))

# ---- C2: margin-generation decouple ----
# W4 joint swap (3128 write set) generation
# 672 P prompts, same pipeline as gen_base.
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

# C3 first: no-swap full generation (s0 full)
gen_full = []
for b0 in range(0, GEN_NP, GEN_BATCH):
    batch = pids_P_all[b0:b0 + GEN_BATCH]
    gen_full.extend(
        gen_batch_g(batch, swap_layers=None))
    if (b0 // GEN_BATCH) % 7 == 0:
        log('C s0-full batch %d/%d done'
            % (b0 // GEN_BATCH + 1,
               (GEN_NP + GEN_BATCH - 1)
               // GEN_BATCH))
s0_mism = 0
s0_agree_tok = 0
for j in range(GEN_NP):
    b12 = pad12([int(v) for v in
                 z26['gen_base_P'][j]])
    g12 = pad12(gen_full[j])
    s0_agree_tok += sum(int(a == b)
                        for a, b in zip(g12,
                                        b12))
    s0_mism += int(g12 != b12)
s0_rate = s0_agree_tok / float(GEN_NP * N_NEW)
s0_full_v = ('s0_fully_deterministic'
             if s0_mism == 0
             else 's0_drift_found')
log('C-S0FULL: bit_mism=%d/%d token '
    'agree=%.6f -> %s'
    % (s0_mism, GEN_NP, s0_rate,
       s0_full_v))

# W4 joint swap generation (decouple test)
gen_swap = []
for b0 in range(0, GEN_NP, GEN_BATCH):
    batch = pids_P_all[b0:b0 + GEN_BATCH]
    gen_swap.extend(gen_batch_g(
        batch,
        swap_layers=JOINT_G['write']))
    if (b0 // GEN_BATCH) % 7 == 0:
        log('C swap-gen batch %d/%d done'
            % (b0 // GEN_BATCH + 1,
               (GEN_NP + GEN_BATCH - 1)
               // GEN_BATCH))
m_base_P = z26['mlg_s0_P'][:GEN_NP, NLG, -1] \
    .astype(np.float64)
m_swap_P = z28['joint_write_P'] \
    .astype(np.float64)[:GEN_NP] + m_base_P
margin_flip = ((m_swap_P > 0)
               != (m_base_P > 0))
gen_same = np.zeros(GEN_NP, dtype=bool)
gen_first_chg = 0
for j in range(GEN_NP):
    b12 = pad12([int(v) for v in
                 z26['gen_base_P'][j]])
    g12 = pad12(gen_swap[j])
    gen_same[j] = (g12 == b12)
    gen_first_chg += int(g12[0] != b12[0])
gen_chg_rate = 1.0 - float(gen_same.mean())
mf_n = int(margin_flip.sum())
if mf_n > 0:
    same_given_flip = float(
        gen_same[margin_flip].mean())
else:
    same_given_flip = -1.0
same_overall = float(gen_same.mean())
decouple_ok = (
    gen_chg_rate < GEN_CHANGE_GATE
    and mf_n > 0
    and abs(same_given_flip - same_overall)
    < GEN_CHANGE_GATE)
gen_v = ('gen_decoupled' if decouple_ok
         else 'gen_coupled')
log('C-DECOUPLE: gen_chg=%.4f margin_flip='
    '%d/%d same|flip=%.4f same_all=%.4f '
    'first_chg=%d -> %s'
    % (gen_chg_rate, mf_n, NP,
       same_given_flip, same_overall,
       gen_first_chg, gen_v))

# ================================================================
# PART D: regen teacher-forced margin sign field
# ================================================================
log('== PART D: sign-field interaction ==')


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(
        np.float64)
    rb = np.argsort(np.argsort(b)).astype(
        np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


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
    for c in ('s1', 's2', 's3'):
        fl = np.zeros(NP, dtype=np.int64)
        reg = z27['regen_%s_%s' % (c, dc)]
        for j in range(NP):
            b12 = [int(v) for v in
                   z26['gen_base_%s' % dc][j]]
            bp, _cap = pol_raw(b12[0])
            rp, _cap2 = pol_raw(reg[j, 0])
            fl[j] = int(rp != bp)
        flip_lab[(c, dc)] = fl
log('D flip labels: %s'
    % {('%s_%s' % k): round(float(v.mean()),
                            4)
       for k, v in flip_lab.items()})

NP_D = 8 if SMOKE else NP
sym_phi = {}
for (c, dc) in GROUPS_D:
    fl = flip_lab[(c, dc)][:NP_D]
    pred = np.zeros((NLG + 1, NP_D),
                    dtype=np.float64)
    for j in range(NP_D):
        pids, _sp = matg['build_pids'](dc, j)
        reg = [int(t) for t in
               z27['regen_%s_%s'
                   % (c, dc)][j]]
        wl, _cv = forward_track_g(
            pids['s0'], reg)
        base_L = z26['mlg_s0_%s' % dc][
            j, :, -1].astype(np.float64)
        pred[:, j] = (
            (wl[:, -1] > 0)
            != (base_L > 0)).astype(
            np.float64)
    phi = np.zeros(NLG + 1)
    for L in range(NLG + 1):
        phi[L] = spearman(pred[L], fl)
    sym_phi['%s_%s' % (c, dc)] = phi
    log('D phi %s_%s: peak %.3f @L%02d'
        % (c, dc, float(np.max(np.abs(phi))),
           int(np.argmax(np.abs(phi)))))

prof_P = sym_phi['s1_P']
prof_A = sym_phi['s3_A1']
diff_prof = prof_P - prof_A
peak_i = int(np.argmax(np.abs(diff_prof)))
peak_layer = int(peak_i)
peak_sep = float(np.abs(diff_prof[peak_i]))
peak_r = float(max(abs(prof_P[peak_i]),
                   abs(prof_A[peak_i])))
sym_v = ('interaction_symbolic_found'
         if (peak_r >= V_GATE
             and peak_sep >= V_SEP)
         else 'interaction_symbolic_'
              'not_in_field')
log('D-SYMINTERACT: peak L%02d sep=%.3f '
    'r=%.3f -> %s'
    % (peak_layer, peak_sep, peak_r,
       sym_v))

del model_g
gc.collect()
torch.cuda.empty_cache()
log('glm4 unloaded; cuda cache cleared')

# ---------------- verdict + dumps -------------
verdict = '|'.join([
    a_v,
    dose_v,
    mid_v,
    a1_v,
    gen_v,
    s0_full_v,
    sym_v,
    'coverage_full'])
assert len(verdict.split('|')) == 8
result = {
    'phase': 3129,
    'name': NAME,
    'smoke': SMOKE,
    'runtime_s': time.time() - T0,
    'verdict': verdict,
    'part_a': {
        'gates_q': gq_r,
        'gates_g': gg_r,
        'mcq': mcq_r, 'mcg': mcg_r},
    'part_b': {
        'mean_row_l2': mean_row_l2,
        'dose_q': dose_q,
        'mid_flip': mid_flip,
        'mid_dm': mid_dm,
        'last_flip_ref': last_flip,
        'repro_max_abs': drep_q},
    'part_c': {
        'a1_flip': a1_flip,
        'a1_dm': a1_dm,
        'a1_by_sgn': a1_res,
        'gen_chg_rate': gen_chg_rate,
        'gen_first_chg': gen_first_chg,
        'margin_flip_n': mf_n,
        'same_given_flip': same_given_flip,
        'same_overall': same_overall,
        's0_mism': s0_mism,
        's0_token_agree': s0_rate,
        'repro_max_abs': drep_g},
    'part_d': {
        'np_d': NP_D,
        'flip_rates': {
            '%s_%s' % k: float(v.mean())
            for k, v in flip_lab.items()},
        'phi_prof': {
            k: [float(x) for x in v]
            for k, v in sym_phi.items()},
        'peak_layer': peak_layer,
        'peak_sep': peak_sep,
        'peak_r': peak_r}}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)
npz_out = {
    'coords_g_top50': coords_g.astype(
        np.int64),
    'm_swap_P': m_swap_P.astype(
        np.float32),
    'margin_flip_P': margin_flip.astype(
        np.int64),
    'gen_same_swap': gen_same.astype(
        np.int64),
    'flip_lab_s1_P': flip_lab[
        ('s1', 'P')],
    'flip_lab_s1_A1': flip_lab[
        ('s1', 'A1')],
    'flip_lab_s3_P': flip_lab[
        ('s3', 'P')],
    'flip_lab_s3_A1': flip_lab[
        ('s3', 'A1')],
    'phi_s1_P': sym_phi['s1_P'],
    'phi_s1_A1': sym_phi['s1_A1'],
    'phi_s3_P': sym_phi['s3_P'],
    'phi_s3_A1': sym_phi['s3_A1']}
np.savez(os.path.join(OUT,
                      'p127_readout.npz'),
         **npz_out)
log('VERDICT: %s' % verdict)
log('dumps done: result.json + '
    'p127_readout.npz (%d keys)'
    % len(npz_out))
log('P3129 DONE (%.1fs)'
    % (time.time() - T0))
