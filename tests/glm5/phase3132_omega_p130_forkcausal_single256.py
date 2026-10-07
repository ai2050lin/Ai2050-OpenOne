# -*- coding: utf-8 -*-
"""Phase 3132 (Omega-P130): fork-layer L17
causalization + single-swap spectrum 256.
Prereg: 3131 MEMO section 5 (items 1-3).
Chain: L17 coords captured at prompt-last
pos (margin-rho, 3131-B method) -> +/-delta
inject at prompt-last -> 12-tok rewrite
rate (vs swap chg=1.0) -> swap4+inject
rescue (step0/allstep) -> ctrl L38 peak /
L33 trough -> teacher-forced margin
profile. Spectrum: sel64-nested 256
samples x 8 layers; hard gate vs 3130
256-sample refs (tol 0.08), soft 64-subset
vs 3131 refs (tol 0.033, record only:
batch-composition drift has no prior).
Seal frozen BEFORE any observation."""
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
NAME = ('omega_p130_forkcausal_'
        'single256')
SMOKE = os.environ.get('P3132_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D31 = RDIR + r'\phase3131' \
      r'\omega_p129_layerwindow_' \
      r'single40_a1gen_forklayer'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3132', NAME)
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
NP_CAP = 32 if SMOKE else NP
SPEC_NP = 4 if SMOKE else 256
NP_TF = 4 if SMOKE else 128
GEN_BATCH = 4 if SMOKE else 32
CAP_BATCH = 8 if SMOKE else 32
DIRS = ('P', 'A1')

INJ_L = 17
CTRL_PEAK_L = 38
CTRL_LOW_L = 33
SWAP_L = [8, 9, 13, 29]
K_DOSE = 50
SPEC_FRAC = 0.10

TF_SEED = 3133
SEL_SEED_OLD = 3131
SEL_SEED_NEW = 3132

SPEC256_TOL = 0.08
SUB64_TOL = 0.033
INJ_ABS_GATE = 0.10
RESC_GATE = 0.10
FORK_LO = 15
FORK_HI = 19

REF30 = {8: 208 / 256.0,
         9: 205 / 256.0,
         13: 137 / 256.0,
         29: 45 / 256.0}
CHG40_REF = {4: 0.96875,
             8: 0.734375,
             9: 0.78125,
             13: 0.421875,
             17: 0.84375,
             29: 0.109375,
             33: 0.09375,
             38: 1.0}
FIRST_A1_REF = 614
N_FIRST_REF = 178
FULL_SWAP_FIRST_REF = 178
SEL64_SHA_REF = '9efd0f88'
RES31_SHA_REF = '25613e9e'
SEAL31_SHA_REF = 'a55569e6'
EXP_V31 = ('a_3130_ok|qwen_win_wide'
           '|gen_swap_3130_bitexact'
           '|single40_dominant'
           '|a1_gen_symmetric|fork_early'
           '|coverage_full')

# ---------------- seal ------------------
SEAL = {
    'phase': 3132,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'INJ_L': INJ_L,
        'CTRL_PEAK_L': CTRL_PEAK_L,
        'CTRL_LOW_L': CTRL_LOW_L,
        'SWAP_L': SWAP_L,
        'K_DOSE': K_DOSE,
        'SPEC_FRAC': SPEC_FRAC,
        'TF_SEED': TF_SEED,
        'SEL_SEED_OLD': SEL_SEED_OLD,
        'SEL_SEED_NEW': SEL_SEED_NEW,
        'SPEC256_TOL': SPEC256_TOL,
        'SUB64_TOL': SUB64_TOL,
        'INJ_ABS_GATE': INJ_ABS_GATE,
        'RESC_GATE': RESC_GATE,
        'FORK_LO': FORK_LO,
        'FORK_HI': FORK_HI},
    'anchors': {
        'res31_verdict': EXP_V31,
        'full_swap_first':
            FULL_SWAP_FIRST_REF,
        'first_a1': FIRST_A1_REF,
        'n_first': N_FIRST_REF,
        'pk_dm': INJ_L,
        'chg40_ref': CHG40_REF,
        'ref30': REF30,
        'sel64_sha8': SEL64_SHA_REF,
        'res31_sha8': RES31_SHA_REF,
        'seal31_sha8': SEAL31_SHA_REF},
    'prereg': ('3131 MEMO section 5: '
               '(1) L17 fork-layer '
               'causalization: capture '
               'margin-rho coords at '
               'prompt-last pos (3131-B '
               'method, block-output '
               'residual state), +/-delta '
               'inject -> 12-tok rewrite '
               'rate; swap4+inject rescue '
               'step0/allstep; ctrl L38 '
               'peak + L33 trough; tf '
               'margin profile gate '
               '[15,19]. (2) spectrum '
               'refinement: sel64-nested '
               '256 x layers '
               '{4,8,9,13,17,29,33,38}; '
               'hard gate vs 3130 256-ref '
               '(tol 0.08), soft 64-subset '
               'vs 3131 64-ref (tol 0.033, '
               'record only: batch-'
               'composition drift has no '
               'prior). (3) L17/L33 added '
               'beyond prereg core '
               '{4,8,9,13,29,38} for '
               'fork-theme closure. '
               'Frozen before observation.')}
SEALF = os.path.join(OUT, 'design_seal.json')
if os.path.exists(SEALF):
    prev = json.load(io.open(SEALF,
                             encoding='utf-8'))
    seal_rt = json.loads(json.dumps(SEAL))
    assert prev['constants'] == \
        seal_rt['constants'], 'seal drift'
    assert prev['anchors'] == \
        seal_rt['anchors'], 'seal drift'
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
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
assert z26['mlg_s0_P'].shape == (672, 41, 13)
assert z26['gen_base_P'].shape == (672, 12)
z31 = np.load(D31 + r'\p129_readout.npz',
              allow_pickle=False)
assert z31['dm'].shape == (672, 41)
res31 = json.load(io.open(
    D31 + r'\result.json', encoding='utf-8'))
assert res31['smoke'] is False
log('frozen inputs ok (05/26/31)')

# ================================================================
# PART A: offline 3131 link asserts
# ================================================================
log('== PART A: 3131 link asserts ==')
V31 = res31['verdict']
assert V31 == EXP_V31, V31
pc31 = res31['part_c']
pb31 = res31['part_b']
assert pc31['full_swap_chg'] == 1.0
assert pc31['full_swap_first'] \
    == FULL_SWAP_FIRST_REF
assert pc31['full_swap_bitexact_3130'] \
    is True
assert pc31['first_a1'] == FIRST_A1_REF
assert pc31['chg_a1'] >= 0.95
assert pc31['pk_dm'] == INJ_L
assert pc31['pk_dm_first'] == INJ_L
assert pc31['n_first'] == N_FIRST_REF
assert pb31['n_win'] == 5
assert pb31['l24_anchor_ok'] is True
assert pc31['sel64_sha8'] == SEL64_SHA_REF
assert pc31['peak40'] == 0
for l, v in CHG40_REF.items():
    assert abs(pc31['chg40'][l] - v) < 1e-12, l
raw31 = io.open(
    D31 + r'\result.json', 'rb').read()
sha31 = hashlib.sha256(
    raw31).hexdigest()[:8]
assert sha31 == RES31_SHA_REF, sha31
seal31 = hashlib.sha256(io.open(
    D31 + r'\design_seal.json', 'rb'
).read()).hexdigest()[:8]
assert seal31 == SEAL31_SHA_REF, seal31
log('A hard asserts ok (3131 frozen, '
    'result sha8 %s)' % sha31)

# ================================================================
# materials factory (3125/3126/3130/3131-
# identical)
# ================================================================
p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
hP = {}
for i in range(len(pkB)):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
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
# model load
# ================================================================
log('== load glm4-9b ==')
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

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
HIDG = int(model_g.config.hidden_size)
log('glm4 loaded NLG=%d HIDG=%d'
    % (NLG, HIDG))
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
log('span_idx asserts vs 3126 ok (672x2)')

pids_P_all = [matg['pids_all']['P'][j]['s0']
              for j in range(NP)]
_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
assert len(PREFIX_IDS) in (0, 2)
log('gen-prefix: %s' % PREFIX_IDS)


def pad12(ids_r):
    v = list(ids_r)[:N_NEW]
    return v + [DOT_G] * (N_NEW - len(v))


def _pad_batch(rows):
    chunk = [list(PREFIX_IDS) + list(p)
             for p in rows]
    maxlen = max(len(p) for p in chunk)
    ids_p = np.full((len(chunk), maxlen),
                    PADG, dtype=np.int64)
    mask_p = np.zeros((len(chunk), maxlen),
                      dtype=np.int64)
    for i, p in enumerate(chunk):
        ids_p[i, maxlen - len(p):] = p
        mask_p[i, maxlen - len(p):] = 1
    return ids_p, mask_p


def gen_batch_g(pids_list, swap_layers=None,
                inj=None):
    ids_p, mask_p = _pad_batch(pids_list)
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
    if inj is not None:
        (il, coords, delta, sgn,
         mode) = inj
        lyr = model_g.model.layers[il]
        co_t = torch.as_tensor(
            np.asarray(coords,
                       dtype=np.int64),
            device='cuda')
        dv_t = torch.full(
            (len(coords),),
            float(delta) * float(sgn),
            device='cuda',
            dtype=torch.bfloat16)
        st = {'step': 0}

        def _inj(mod, inp, out):
            o2 = out[0] \
                if isinstance(out,
                              tuple) \
                else out
            if st['step'] == 0 \
                    or mode == 'allstep':
                o2[:, -1, co_t] += dv_t
            st['step'] += 1
            return None

        hooks.append(
            lyr.register_forward_hook(
                _inj))
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
    newg = outg[:, ids_p.shape[1]:]
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


def capture_L17():
    HOUT = np.zeros((NP_CAP, HIDG),
                    dtype=np.float32)
    lyr = model_g.model.layers[INJ_L]
    for b0 in range(0, NP_CAP, CAP_BATCH):
        rows = [pids_P_all[j] for j in
                range(b0,
                      min(b0 + CAP_BATCH,
                          NP_CAP))]
        ids_p, mask_p = _pad_batch(rows)
        feats = []

        def _cap(mod, inp, out):
            o2 = out[0] \
                if isinstance(out,
                              tuple) \
                else out
            feats.append(
                o2[:, -1, :].detach())

        hk = lyr.register_forward_hook(
            _cap)
        t_ids = torch.tensor(ids_p,
                             device='cuda')
        t_mask = torch.tensor(mask_p,
                              device='cuda')
        with torch.inference_mode():
            model_g(input_ids=t_ids,
                    attention_mask=t_mask,
                    use_cache=False)
        hk.remove()
        HOUT[b0:b0 + len(rows)] = \
            feats[0].float().cpu().numpy()
        del feats
    return HOUT


def margin_track_single(prompt_ids,
                        traj_ids,
                        inject=None):
    """Single-sample, NO prefix, FULL-TRACK
    margin (3131 forward_track_g / 3126 z26
    convention: ids = prompt + traj,
    pos0 = len(prompt)-1, npts = 13,
    teacher-forced). Inject delta at pos0
    (track step k=0) after layer il output.
    Returns (NLG+1, npts) float64.
    [rev-3132a dbg4/dbg5: replicates z26
    mlg_s0_P to 0.0 exactly on all (L,k);
    prompt-only last-pos REJECTED - bf16
    compute-path chaos vs z26 up to 0.63.]"""
    pos0 = len(prompt_ids) - 1
    ids = list(prompt_ids) \
        + [int(x) for x in traj_ids]
    t_in = torch.tensor([ids],
                        device='cuda')
    npts = len(traj_ids) + 1
    feats = []
    hooks = []
    if inject is not None:
        (il, coords, delta, sgn) = inject
        lyr = model_g.model.layers[il]
        co_t = torch.as_tensor(
            np.asarray(coords,
                       dtype=np.int64),
            device='cuda')
        dv_t = torch.full(
            (len(coords),),
            float(delta) * float(sgn),
            device='cuda',
            dtype=torch.bfloat16)

        def _inj(mod, inp, out):
            o2 = out[0] \
                if isinstance(out,
                              tuple) \
                else out
            o2[0, pos0, co_t] += dv_t
            return None

        hooks.append(
            lyr.register_forward_hook(
                _inj))

    def _mk():
        def hook(mod, inp, out):
            o2 = out[0] \
                if isinstance(out,
                              tuple) \
                else out
            feats.append(o2.detach())
        return hook

    for lyr in model_g.model.layers:
        hooks.append(
            lyr.register_forward_hook(
                _mk()))
    with torch.inference_mode():
        model_g(t_in, use_cache=False)
        for hk in hooks:
            hk.remove()
        seq = [model_g.model.embed_tokens(
            t_in)] + feats
        ml = np.zeros((NLG + 1, npts),
                      dtype=np.float64)
        for L in range(NLG + 1):
            h = norm_g(seq[L][0]).float() \
                .cpu().numpy()
            for k in range(npts):
                ml[L, k] = float(
                    h[pos0 + k] @ w_dn_g)
        del feats
    return ml


# ================================================================
# PART B0: dm sign stats (3131 frozen npz)
# ================================================================
log('== PART B0: dm sign stats ==')
dm31 = z31['dm'].astype(np.float64)
med_dm17 = float(np.median(dm31[:, INJ_L]))
frac_neg17 = float(
    (dm31[:, INJ_L] < 0).mean())
sgn_exp = 1 if med_dm17 < 0 else -1
log('B0: med dm[L17]=%.4f frac_neg=%.4f '
    '-> rescue expected sgn %+d'
    % (med_dm17, frac_neg17, sgn_exp))

# ================================================================
# PART B1: L17 coord capture + rho + delta
# ================================================================
log('== PART B1: L17 capture ==')
HOUT = capture_L17()
m_ref = z26['mlg_s0_P'][:NP_CAP, NLG, 0] \
    .astype(np.float64)
log('B1: captured (%d,%d); m_ref '
    'mean=%.4f' % (HOUT.shape[0],
                   HOUT.shape[1],
                   float(m_ref.mean())))
Hf = HOUT.astype(np.float64)
hr = np.argsort(np.argsort(Hf, axis=0),
                axis=0).astype(np.float64)
mr = np.argsort(np.argsort(m_ref)).astype(
    np.float64)
hc = hr - hr.mean(0, keepdims=True)
mc2 = mr - mr.mean()
den = np.sqrt((hc * hc).sum(0)
              * float((mc2 * mc2).sum()))
rho_l17 = (hc * mc2[:, None]).sum(0) / den
order_rho = np.argsort(-np.abs(rho_l17))
co50 = order_rho[:K_DOSE].astype(np.int64)
row_l2 = np.sqrt((Hf * Hf).sum(1))
mean_row_l2 = float(row_l2.mean())
delta_l17 = SPEC_FRAC * mean_row_l2
co_sha8 = hashlib.sha256(
    co50.tobytes()).hexdigest()[:8]
top_abs = [float(abs(rho_l17[c]))
           for c in co50[:10]]
log('B1: co50 sha8=%s delta=%.4f '
    'mean_row_l2=%.4f rho_top10=%s'
    % (co_sha8, delta_l17, mean_row_l2,
       ' '.join('%.3f' % v
                for v in top_abs)))

# ================================================================
# PART B2: L17 inject -> generate rewrite
# ================================================================
log('== PART B2: inject trials (%d) =='
    % NP_B)
base12_P = [pad12([int(v) for v in
                   z26['gen_base_P'][j]])
            for j in range(NP_B)]
pids_B = [pids_P_all[j]
          for j in range(NP_B)]
TRIALS = [(INJ_L, 'step0', 1),
          (INJ_L, 'step0', -1),
          (INJ_L, 'allstep', 1),
          (INJ_L, 'allstep', -1),
          (CTRL_PEAK_L, 'step0', 1),
          (CTRL_PEAK_L, 'step0', -1),
          (CTRL_LOW_L, 'step0', 1),
          (CTRL_LOW_L, 'step0', -1)]
inj_res = {}
same_store = {}
for (L, mode, sgn) in TRIALS:
    gen_l = []
    for b0 in range(0, NP_B, GEN_BATCH):
        batch = pids_B[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g(
            batch,
            inj=(L, co50, delta_l17, sgn,
                 mode)))
    same_l = np.zeros(NP_B, dtype=bool)
    first_l = 0
    for j in range(NP_B):
        g12 = pad12(gen_l[j])
        same_l[j] = (g12 == base12_P[j])
        first_l += int(g12[0]
                       != base12_P[j][0])
    chg_l = 1.0 - float(same_l.mean())
    key = '%d|%s|%+d' % (L, mode, sgn)
    inj_res[key] = {'chg': chg_l,
                    'first': first_l,
                    'layer': L, 'mode': mode,
                    'sgn': sgn}
    same_store['same_L%d_%s_%s'
               % (L, mode,
                  ('p1' if sgn > 0
                   else 'm1'))] = \
        same_l.astype(np.int64)
    log('B2 %s: chg=%.4f first=%d'
        % (key, chg_l, first_l))


def best_chg(keys):
    return max(inj_res[k]['chg']
               for k in keys)


keys_l17 = ['%d|%s|%+d' % (INJ_L, m, s)
            for m in ('step0', 'allstep')
            for s in (1, -1)]
keys_l38 = ['%d|step0|%+d'
            % (CTRL_PEAK_L, s)
            for s in (1, -1)]
keys_l33 = ['%d|step0|%+d'
            % (CTRL_LOW_L, s)
            for s in (1, -1)]
chg_l17_best = best_chg(keys_l17)
chg_l38_best = best_chg(keys_l38)
chg_l33_best = best_chg(keys_l33)
if not SMOKE:
    if chg_l17_best >= INJ_ABS_GATE \
            and chg_l17_best \
            > chg_l33_best:
        l17_v = 'l17_inj_rewrites'
    elif chg_l17_best >= INJ_ABS_GATE:
        l17_v = 'l17_inj_ambiguous'
    else:
        l17_v = 'l17_inj_null'
    ctrl_v = ('ctrl33_quiescent'
              if chg_l33_best
              < 0.5 * max(chg_l17_best,
                          INJ_ABS_GATE)
              else 'ctrl33_active')
    log('B2-GATE: l17_best=%.4f l38=%.4f '
        'l33=%.4f -> %s / %s'
        % (chg_l17_best, chg_l38_best,
           chg_l33_best, l17_v, ctrl_v))
else:
    l17_v = 'l17_smoke_subset'
    ctrl_v = 'ctrl_smoke_subset'
    log('B2-GATE: SKIPPED in SMOKE '
        '(NP_B=%d)' % NP_B)

# ================================================================
# PART B3: swap4 + inject rescue
# ================================================================
log('== PART B3: rescue trials (%d) =='
    % NP_B)
RESC = [(INJ_L, 'step0', 1),
        (INJ_L, 'step0', -1),
        (INJ_L, 'allstep', 1),
        (INJ_L, 'allstep', -1),
        (CTRL_PEAK_L, 'step0', 1),
        (CTRL_PEAK_L, 'step0', -1)]
resc_res = {}
for (L, mode, sgn) in RESC:
    gen_l = []
    for b0 in range(0, NP_B, GEN_BATCH):
        batch = pids_B[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g(
            batch, swap_layers=SWAP_L,
            inj=(L, co50, delta_l17, sgn,
                 mode)))
    same_l = np.zeros(NP_B, dtype=bool)
    first_l = 0
    for j in range(NP_B):
        g12 = pad12(gen_l[j])
        same_l[j] = (g12 == base12_P[j])
        first_l += int(g12[0]
                       != base12_P[j][0])
    chg_l = 1.0 - float(same_l.mean())
    key = '%d|%s|%+d' % (L, mode, sgn)
    resc_res[key] = {'chg': chg_l,
                     'rescue': float(
                         same_l.mean()),
                     'first': first_l,
                     'layer': L, 'mode': mode,
                     'sgn': sgn}
    same_store['sameR_L%d_%s_%s'
               % (L, mode,
                  ('p1' if sgn > 0
                   else 'm1'))] = \
        same_l.astype(np.int64)
    log('B3 %s: chg=%.4f rescue=%.4f '
        'first=%d' % (key, chg_l,
                      resc_res[key]['rescue'],
                      first_l))
rescue_l17 = max(resc_res[k]['rescue']
                 for k in keys_l17)
rescue_l38 = max(
    resc_res['%d|step0|%+d'
             % (CTRL_PEAK_L, s)]['rescue']
    for s in (1, -1))
if not SMOKE:
    if rescue_l17 >= RESC_GATE \
            and rescue_l17 > rescue_l38:
        rescue_v = 'rescue_l17_present'
    elif rescue_l38 >= RESC_GATE:
        rescue_v = 'rescue_not_specific'
    else:
        rescue_v = 'rescue_absent'
    log('B3-GATE: rescue_l17=%.4f '
        'rescue_l38=%.4f -> %s'
        % (rescue_l17, rescue_l38,
           rescue_v))
else:
    rescue_v = 'rescue_smoke_subset'
    log('B3-GATE: SKIPPED in SMOKE')

# ================================================================
# PART B4: teacher-forced margin profile
# ================================================================
log('== PART B4: tf margin profile ==')
# [rev-3132a] margin semantics corrected
# to the 3131 full-track convention
# (dbg4/dbg5): baseline = z26 mlg_s0_P
# (prompt+traj, 13 positions, exact
# replica); inject delta at pos0; layer
# profile = df[:, -1] (last track
# position, same column convention as
# 3131 dm = ml_swap_last - ml_base_last).
# prompt-only last-pos REJECTED: bf16
# compute-path chaos vs z26 up to 0.63.
log('B4 NOTE: full-track margin '
    'convention (pos0 inject, last-col '
    'profile, 3131-aligned); prompt-only '
    'rejected by dbg5 (|d|<=0.63)')
tf_idx = np.sort(np.random.default_rng(
    TF_SEED).choice(NP, NP_TF,
                    replace=False)).astype(
    np.int64)
tf_rows = [pids_P_all[j] for j in tf_idx]
tf_trajs = [matg['traj_tokens'](
    'P', int(j))[2] for j in tf_idx]
m0_tf = z26['mlg_s0_P'][tf_idx] \
    .astype(np.float64)
assert m0_tf.shape == (NP_TF, NLG + 1,
                       N_NEW + 1)
_drep = np.stack(
    [margin_track_single(tf_rows[jj],
                         tf_trajs[jj])
     for jj in range(4)])
drep_p = float(np.abs(
    _drep - m0_tf[:4]).max())
assert drep_p <= 1e-4, drep_p
log('B4 repro full-track max|d|=%.2e '
    '(<=1e-4, all L,k)' % drep_p)
med_tf = {}
dk0_med = {}
df_last_store = {}
for sgn in (1, -1):
    df_all = np.zeros(
        (NP_TF, NLG + 1, N_NEW + 1),
        dtype=np.float64)
    for jj in range(NP_TF):
        ml = margin_track_single(
            tf_rows[jj], tf_trajs[jj],
            inject=(INJ_L, co50,
                    delta_l17, sgn))
        df_all[jj] = ml - m0_tf[jj]
        if jj % 32 == 0:
            log('B4 sgn %+d row %d/%d'
                % (sgn, jj, NP_TF))
    df_last_store[sgn] = \
        df_all[:, :, -1].astype(np.float32)
    zmax = float(np.abs(
        df_all[:, :INJ_L + 2, -1]).max())
    assert zmax == 0.0, zmax
    med_tf[sgn] = np.median(
        np.abs(df_all[:, :, -1]), axis=0)
    dk0_med[sgn] = np.median(
        np.abs(df_all[:, :, 0]), axis=0)
    log('B4 sgn %+d: med|df| L17o=%.4f '
        'L18o=%.4f L40=%.4f peak=L%02d '
        'dk0peak=L%02d'
        % (sgn, med_tf[sgn][INJ_L + 1],
           med_tf[sgn][INJ_L + 2],
           med_tf[sgn][NLG],
           int(np.argmax(med_tf[sgn])),
           int(np.argmax(dk0_med[sgn]))))
med_tf_avg = 0.5 * (med_tf[1]
                    + med_tf[-1])
peak_tf = int(np.argmax(med_tf_avg))
med_abs_dm31 = z31['med_abs_dm'] \
    .astype(np.float64)


def rankv(x):
    return np.argsort(
        np.argsort(x)).astype(np.float64)


sp_tf = float(np.corrcoef(
    rankv(med_tf_avg),
    rankv(med_abs_dm31))[0, 1])
if not SMOKE:
    if FORK_LO <= peak_tf <= FORK_HI:
        fork_v = 'fork_l17_gate'
    elif peak_tf > FORK_HI:
        fork_v = 'fork_l17_amp'
    else:
        fork_v = 'fork_anom'
    log('B4-GATE: peak_tf=L%02d '
        'spearman_vs_3131=%.3f -> %s'
        % (peak_tf, sp_tf, fork_v))
else:
    fork_v = 'fork_smoke_subset'
    log('B4-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C: sel64-nested 256 spectrum
# ================================================================
log('== PART C: spectrum 256 ==')
sel64 = np.sort(np.random.default_rng(
    SEL_SEED_OLD).choice(NP, 64,
                         replace=False)) \
    .astype(np.int64)
sha64 = hashlib.sha256(
    sel64.tobytes()).hexdigest()[:8]
assert sha64 == SEL64_SHA_REF, sha64
rest = np.setdiff1d(np.arange(NP), sel64)
sel192 = np.sort(np.random.default_rng(
    SEL_SEED_NEW).choice(rest, 192,
                         replace=False)) \
    .astype(np.int64)
sel256 = np.sort(np.concatenate(
    [sel64, sel192])).astype(np.int64)
assert len(sel256) == 256
assert np.all(np.isin(sel64, sel256))
sel_sha8 = hashlib.sha256(
    sel256.tobytes()).hexdigest()[:8]
log('C: sel256 sha8=%s (sel64 nested, '
    'sha64=%s)' % (sel_sha8, sha64))
LAYERS_C = [4, 8, 9, 13, 17, 29, 33, 38]
chg256 = {}
first256 = {}
same256 = {}
for l in LAYERS_C:
    rows_l = [pids_P_all[j]
              for j in sel256[:SPEC_NP]]
    gen_l = []
    for b0 in range(0, len(rows_l),
                    GEN_BATCH):
        batch = rows_l[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g(
            batch, swap_layers=[l]))
    same_l = np.zeros(len(rows_l),
                      dtype=bool)
    first_l = 0
    for jj, j in enumerate(
            sel256[:SPEC_NP]):
        b12 = pad12([int(v) for v in
                     z26['gen_base_P'][j]])
        g12 = pad12(gen_l[jj])
        same_l[jj] = (g12 == b12)
        first_l += int(g12[0] != b12[0])
    chg256[l] = 1.0 - float(same_l.mean())
    first256[l] = first_l
    same256[l] = same_l
    log('C L%02d: chg=%.4f first=%d'
        % (l, chg256[l], first_l))
if not SMOKE:
    dev30 = {str(l): float(
        abs(chg256[l] - REF30[l]))
        for l in REF30}
    spec_ok = all(v <= SPEC256_TOL
                  for v in dev30.values())
    spec_v = ('spec256_ok' if spec_ok
              else 'spec256_dev')
    pos64 = np.array(
        [int(np.where(sel256 == j)[0][0])
         for j in sel64], dtype=np.int64)
    dev31 = {str(l): float(abs(
        (1.0 - float(same256[l][pos64]
                     .mean()))
        - CHG40_REF[l]))
        for l in LAYERS_C}
    n_dev31_over = sum(
        1 for v in dev31.values()
        if v > SUB64_TOL)
    log('C-GATE: dev30=%s -> %s; '
        'dev31(max=%.4f, over%d) soft'
        % (' '.join('%s:%.3f' % (k, v)
                    for k, v in
                    sorted(dev30.items())),
           spec_v, max(dev31.values()),
           n_dev31_over))
else:
    dev30 = {}
    dev31 = {}
    n_dev31_over = -1
    spec_v = 'spec_smoke_subset'
    log('C-GATE: SKIPPED in SMOKE')

# ================================================================
# dumps
# ================================================================
runtime = time.time() - T0
verdict = '|'.join([
    'a_3131_ok', l17_v, rescue_v, ctrl_v,
    spec_v, fork_v, 'coverage_full'])
log('VERDICT: %s' % verdict)
result = {
    'name': NAME,
    'phase': 3132,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'res31_sha8': sha31,
        'asserts_3131': 'ok'},
    'part_b': {
        'dm_med17': med_dm17,
        'dm_frac_neg17': frac_neg17,
        'sgn_exp': int(sgn_exp),
        'co50_sha8': co_sha8,
        'delta_l17': delta_l17,
        'mean_row_l2': mean_row_l2,
        'rho_top10_abs': top_abs,
        'inj_trials': inj_res,
        'chg_l17_best': chg_l17_best,
        'chg_l38_best': chg_l38_best,
        'chg_l33_best': chg_l33_best,
        'l17_v': l17_v,
        'ctrl_v': ctrl_v,
        'resc_trials': resc_res,
        'rescue_l17': rescue_l17,
        'rescue_l38': rescue_l38,
        'rescue_v': rescue_v,
        'tf_idx': [int(v)
                   for v in tf_idx],
        'peak_tf': int(peak_tf),
        'spearman_tf_vs_3131': sp_tf,
        'med_tf_avg': [float(v) for v
                       in med_tf_avg],
        'med_tf_p1': [float(v) for v
                      in med_tf[1]],
        'med_tf_m1': [float(v) for v
                      in med_tf[-1]],
        'fork_v': fork_v},
    'part_c': {
        'sel_sha8': sel_sha8,
        'sel64_sha8': sha64,
        'layers_c': LAYERS_C,
        'chg256': {str(l): float(
            chg256[l]) for l in LAYERS_C},
        'first256': {str(l): int(
            first256[l])
            for l in LAYERS_C},
        'dev30': dev30,
        'dev31': dev31,
        'n_dev31_over': int(n_dev31_over),
        'spec_v': spec_v},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'co50': co50,
    'rho_l17': rho_l17.astype(np.float32),
    'hout_l17': HOUT,
    'dm31_med17': np.float64(med_dm17),
    'dm31_frac_neg17': np.float64(
        frac_neg17),
    'tf_idx': tf_idx,
    'med_tf_p1': med_tf[1],
    'med_tf_m1': med_tf[-1],
    'med_tf_avg': med_tf_avg,
    'tf_df_last_p1': df_last_store[1],
    'tf_df_last_m1': df_last_store[-1],
    'tf_dk0_med_p1': dk0_med[1],
    'tf_dk0_med_m1': dk0_med[-1],
    'sel64': sel64,
    'sel256': sel256}
for k, v in same_store.items():
    npz_out[k] = v
for l in LAYERS_C:
    npz_out['chg256_%d' % l] = np.float64(
        chg256[l])
    npz_out['same256_%d' % l] = \
        same256[l].astype(np.int64)
np.savez(os.path.join(OUT,
                      'p130_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p130_readout.npz (%d keys)'
    % len(npz_out))
log('P3132 DONE (%.1fs)'
    % (time.time() - T0))
