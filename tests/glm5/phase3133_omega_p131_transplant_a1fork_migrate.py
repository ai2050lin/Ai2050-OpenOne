# -*- coding: utf-8 -*-
"""Phase 3133 (Omega-P131): strong-intervention
disambiguation of the L17 gate + A1 first-token
fork layer profile + write-chain spectrum
migration to the A1 direction.
Prereg: 3132 MEMO section 5 (items 1-3).
Part B: full-dimensional swap-state transplant
(dvec = h_swap4 - h_base at layers 17/29/33/38,
injected at prompt-last pos during generation)
dose {0.5,1,2} x step0/allstep + L38 single +
joint {17,38} and {17,29,33,38} -> additivity.
Alternative-explanation test: if the L17 dm
correlation were non-mediational, transplant
should fail where top50 scalar injection
succeeded; if mediation holds at state level,
transplant should reach swap-level rewrite.
Part C1: teacher-forced swap4 fork-layer
profiles, A1 half vs P half (first-divergence
depth at k=0 + med|dm| profile vs 3131
med_dm_first). Part C2: single-layer swap
spectrum on A1 prompts (same sel256 as 3132,
sha e34d2588) vs P chg256 -> transfer rho.
Part C3: L17 top50 coords injection on A1
(co50 sha 52b126af reuse). Seal frozen BEFORE
any observation."""
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
NAME = ('omega_p131_transplant_'
        'a1fork_migrate')
SMOKE = os.environ.get('P3133_SMOKE',
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
D32 = RDIR + r'\phase3132' \
      r'\omega_p130_forkcausal_single256'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3133', NAME)
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


# ---------------- crash-resume ckpt -----
# [rev-3133h] two full runs died at the
# same logical point (~7.4h, C2 L08->L13
# transition; run1 cause unresolved, run2
# host power outage) -> all expensive
# stages now checkpoint to disk after
# completion and are skipped on restart.
# B1 (16 min) is cheap and always rerun;
# resumed stages: b2_base / b2_trials /
# c1_prof / c2_base / c2_layers / c3_base
# / c3.
import pickle  # noqa: E402
CKPTF = os.path.join(OUT, 'p131_ckpt.pkl')
CK = {'done': [], 'data': {}, 'meta': {}}
if os.path.exists(CKPTF):
    try:
        with io.open(CKPTF, 'rb') as _fh:
            CK = pickle.load(_fh)
        log('RESUME: ckpt stages=%s'
            % sorted(CK['done']))
    except Exception as _e:
        log('RESUME: ckpt load failed (%s), '
            'starting fresh' % _e)
        CK = {'done': [], 'data': {},
              'meta': {}}


def ck_save(stage, data):
    CK['data'][stage] = data
    if stage not in CK['done']:
        CK['done'].append(stage)
    _tmp = CKPTF + '.tmp'
    with io.open(_tmp, 'wb') as _fh:
        pickle.dump(CK, _fh, protocol=4)
    os.replace(_tmp, CKPTF)
    log('CKPT saved: %s' % stage)


# ---------------- frozen constants ------
NP = 672
N_NEW = 12
NP_B = 8 if SMOKE else NP
NP_CAP = 32 if SMOKE else NP
TF_N = 4 if SMOKE else 128
SPEC_NP = 4 if SMOKE else 256
C3_N = 8 if SMOKE else 128
GEN_BATCH = 4 if SMOKE else 32
CAP_BATCH = 8 if SMOKE else 32
DIRS = ('P', 'A1')

SWAP_L = [8, 9, 13, 29]
CAP_L = [17, 29, 33, 38]
LAYERS_C = [4, 8, 9, 13, 17, 29, 33, 38]
DOSES = (0.5, 1.0, 2.0)

TF_SEED = 3134
SEL_SEED_OLD = 3131
SEL_SEED_NEW = 3132

TR_FULL = 0.90
TR_PART = 0.10
ADD_TOL = 0.15
DOSE_SLACK = 0.02
RHO_PROF = 0.70
RHO_TRANS = 0.70
MED_E_TOL = 3
EMERGE_FRAC = 0.10
FLIP_GUARD = 0.1
DM40_MIN = 0.05
P31_LINK_RHO = 0.95

# ---------------- frozen 3132 anchors --
EXP_V32 = ('a_3131_ok|l17_inj_rewrites'
           '|rescue_absent|ctrl33_quiescent'
           '|spec256_ok|fork_l17_amp'
           '|coverage_full')
RES32_SHA = 'e4a0c198'
SEAL32_SHA = '81c4820c'
SEL256_SHA = 'e34d2588'
SEL64_SHA = '9efd0f88'
CO50_SHA = '52b126af'
CHG17BEST_REF = 0.9196
RESC32_L17 = 0.0089
PEAK_TF32 = 28
DEV30_MAX32 = 0.043
REF30 = {8: 208 / 256.0,
         9: 205 / 256.0,
         13: 137 / 256.0,
         29: 45 / 256.0}
REF32_CHG256 = {4: 0.96875,
                8: 0.7890625,
                9: 0.76171875,
                13: 0.4921875,
                17: 0.8828125,
                29: 0.1875,
                33: 0.1640625,
                38: 0.9921875}
FIRST_A1_31 = 614
NFIRST_31 = 178
PK_DM31 = 17

# ---------------- seal ------------------
SEAL = {
    'phase': 3133,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'SWAP_L': SWAP_L, 'CAP_L': CAP_L,
        'LAYERS_C': LAYERS_C,
        'DOSES': list(DOSES),
        'TF_SEED': TF_SEED,
        'SEL_SEED_OLD': SEL_SEED_OLD,
        'SEL_SEED_NEW': SEL_SEED_NEW,
        'TR_FULL': TR_FULL,
        'TR_PART': TR_PART,
        'ADD_TOL': ADD_TOL,
        'DOSE_SLACK': DOSE_SLACK,
        'RHO_PROF': RHO_PROF,
        'RHO_TRANS': RHO_TRANS,
        'MED_E_TOL': MED_E_TOL,
        'EMERGE_FRAC': EMERGE_FRAC,
        'FLIP_GUARD': FLIP_GUARD,
        'DM40_MIN': DM40_MIN,
        'P31_LINK_RHO': P31_LINK_RHO},
    'anchors': {
        'res32_verdict': EXP_V32,
        'res32_sha8': RES32_SHA,
        'seal32_sha8': SEAL32_SHA,
        'sel256_sha8': SEL256_SHA,
        'sel64_sha8': SEL64_SHA,
        'co50_sha8': CO50_SHA,
        'chg17best_ref': CHG17BEST_REF,
        'resc32_l17': RESC32_L17,
        'peak_tf32': PEAK_TF32,
        'dev30_max32': DEV30_MAX32,
        'ref32_chg256': REF32_CHG256,
        'first_a1_31': FIRST_A1_31,
        'nfirst_31': NFIRST_31,
        'pk_dm31': PK_DM31},
    'prereg': ('3132 MEMO section 5: '
               '(1) L17 correlation-vs-'
               'mediation alternative '
               'test: full-dim swap-state '
               'transplant (dvec h_swap4-'
               'h_base at L{17,29,33,38}) '
               'doses 0.5/1/2 step0 + '
               'allstep + L38 single + '
               'joint {17,38},{17,29,33,'
               '38} -> 12-tok chg + '
               'additivity; generation '
               'step-level first-divergence '
               'trajectories. (2) A1 '
               'first-token fork asymmetry '
               '(614 vs 178) layer profile: '
               'tf swap4 fork-layer '
               'profiles A1 vs P halves '
               '(first-divergence depth '
               'k=0 + med|dm| vs 3131 '
               'med_dm_first). (3) '
               'write-chain spectrum '
               'migration to A1: '
               'single-layer swap x 8 '
               'layers on A1 (sel256 '
               'paired) vs P chg256 + '
               'L17 coords injection on '
               'A1. Frozen before '
               'observation.')}
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
assert z26['mlg_s0_A1'].shape == (672, 41, 13)
assert z26['gen_base_P'].shape == (672, 12)
assert z26['gen_base_A1'].shape == (672, 12)
z31 = np.load(D31 + r'\p129_readout.npz',
              allow_pickle=False)
assert z31['dm'].shape == (672, 41)
assert z31['med_dm_first'].shape == (41,)
z32 = np.load(D32 + r'\p130_readout.npz',
              allow_pickle=False)
assert z32['hout_l17'].shape == (672, 4096)
assert z32['co50'].shape == (50,)
res32 = json.load(io.open(
    D32 + r'\result.json', encoding='utf-8'))
assert res32['smoke'] is False
log('frozen inputs ok (05/26/31/32)')

# ================================================================
# PART A: offline 3132 link asserts
# ================================================================
log('== PART A: 3132 link asserts ==')
V32 = res32['verdict']
assert V32 == EXP_V32, V32
pb32 = res32['part_b']
pc32 = res32['part_c']
assert abs(pb32['chg_l17_best']
           - CHG17BEST_REF) < 1e-3
assert abs(pb32['rescue_l17']
           - RESC32_L17) < 1e-3
assert pb32['rescue_l38'] == 0.0
assert pb32['peak_tf'] == PEAK_TF32
assert pc32['sel_sha8'] == SEL256_SHA
assert pc32['sel64_sha8'] == SEL64_SHA
for l, v in REF32_CHG256.items():
    assert abs(pc32['chg256'][str(l)]
               - v) < 1e-12, l
dev30_chk = max(
    abs(pc32['chg256'][str(l)] - REF30[l])
    for l in REF30)
assert dev30_chk <= 0.08
raw32 = io.open(
    D32 + r'\result.json', 'rb').read()
sha32 = hashlib.sha256(
    raw32).hexdigest()[:8]
assert sha32 == RES32_SHA, sha32
seal32h = hashlib.sha256(io.open(
    D32 + r'\design_seal.json', 'rb'
).read()).hexdigest()[:8]
assert seal32h == SEAL32_SHA, seal32h
co50 = z32['co50'].astype(np.int64)
co50_sha = hashlib.sha256(
    co50.tobytes()).hexdigest()[:8]
assert co50_sha == CO50_SHA, co50_sha
log('A hard asserts ok (3132 frozen, '
    'result sha8 %s, co50 sha8 %s)'
    % (sha32, co50_sha))

# ================================================================
# materials factory (3125/3126/3130/3131/
# 3132-identical)
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
assert HIDG == 4096
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
pids_A1_all = [matg['pids_all']['A1'][j]['s0']
               for j in range(NP)]
_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
assert len(PREFIX_IDS) in (0, 2)
for j in (0, 1, 2):
    _ae = tok_g(
        matg['texts']['A1'][pks[j]]
    )['input_ids']
    assert list(_ae[:len(PREFIX_IDS)]) \
        == list(PREFIX_IDS), 'A1 prefix'
    assert len(_ae) - len(PREFIX_IDS) \
        == len(matg['PID_T']['A1'][j])
log('gen-prefix: %s (A1 rows checked)'
    % PREFIX_IDS)


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


def gen_batch_g2(pids_list,
                 swap_layers=None,
                 inj=None,
                 inj_vec=None):
    """Generate N_NEW tokens per row.
    inj: scalar coord injection
      (il, coords, delta, sgn, mode).
    inj_vec: list of per-layer transplants
      (il, dvec_batch (n,HID) fp32, scale,
       mode). step0 = prompt forward only;
      allstep = every forward."""
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
    if inj_vec:
        for (il, dvec_batch, scale,
             mode) in inj_vec:
            lyr = model_g.model.layers[il]
            dv_t = torch.as_tensor(
                np.asarray(dvec_batch,
                           dtype=np.float32),
                device='cuda') \
                .to(torch.bfloat16) \
                * float(scale)
            st = {'step': 0}

            def _injv(mod, inp, out,
                      _dv=dv_t, _m=mode,
                      _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if _st['step'] == 0 \
                        or _m == 'allstep':
                    o2[:, -1, :] += _dv
                _st['step'] += 1
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _injv))
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


def capture_states(rows_all, cap_layers,
                   swap_layers=None):
    """Last-position states at cap_layers.
    [rev-3133f] SINGLE-SAMPLE loop (batch=1,
    zero padding): REPRO proved batched
    left-padded plain-forwards measurably
    change row states (row0 margin +0.52
    when padded against a different-length
    row; position_ids fix did NOT remove
    it). Single-sample forwards replicate
    z26/3131 semantics exactly (row0
    full-track max|d|=0.000e+00). Cost:
    N forwards x2 states, acceptable."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    for j in range(len(rows_all)):
        ids_r = rows_all[j]
        t_in = torch.tensor(
            [list(ids_r)], device='cuda')
        feats = {l: None
                 for l in cap_layers}
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

        def _mk(_l):
            def hook(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                feats[_l] = \
                    o2[0, -1, :].detach()
            return hook

        for l in cap_layers:
            lyr = model_g.model.layers[l]
            hooks.append(
                lyr.register_forward_hook(
                    _mk(l)))
        with torch.inference_mode():
            model_g(t_in, use_cache=False)
            for hk in hooks:
                hk.remove()
            for l in cap_layers:
                OUTS[l][j] = \
                    feats[l].float() \
                    .cpu().numpy()
        del feats
    return OUTS


def margin_track_single(prompt_ids,
                        traj_ids,
                        swap_layers=None,
                        inject=None):
    """Single-sample, NO prefix, FULL-TRACK
    margin (3131 forward_track_g / 3126 z26
    convention: ids = prompt + traj,
    pos0 = len(prompt)-1, npts = 13,
    teacher-forced). Inject delta at pos0
    (track step k=0) after layer il output.
    swap_layers: bypass those layers
    (output := input) at all positions.
    Returns (NLG+1, npts) float64.
    [rev-3132a: replicates z26 mlg_s0_P to
    0.0 exactly on all (L,k).]"""
    pos0 = len(prompt_ids) - 1
    ids = list(prompt_ids) \
        + [int(x) for x in traj_ids]
    t_in = torch.tensor([ids],
                        device='cuda')
    npts = len(traj_ids) + 1
    feats = []
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


def trial_metrics(gen_l, base12_l):
    same_l = np.zeros(len(gen_l),
                      dtype=bool)
    fstep = np.full(len(gen_l), -1,
                    dtype=np.int8)
    first_l = 0
    for j in range(len(gen_l)):
        g12 = pad12(gen_l[j])
        b12 = base12_l[j]
        same_l[j] = (g12 == b12)
        if g12 != b12:
            for t in range(N_NEW):
                if g12[t] != b12[t]:
                    fstep[j] = t
                    break
        first_l += int(g12[0] != b12[0])
    chg_l = 1.0 - float(same_l.mean())
    cum = [float(((fstep >= 0)
                  & (fstep <= t)).mean())
           for t in range(N_NEW)]
    return chg_l, first_l, fstep, cum


# ================================================================
# PART B1: swap-state capture -> dvec
# ================================================================
if os.environ.get('P3133_REPRO_ONLY',
                  '') == '1':
    # [rev-3133e] decisive diagnostics:
    # (a) single-sample full-track repro
    # vs z26 (session anchor stability);
    # (b) single-sample capture at
    # layers[17] output vs z26 profile
    # (isolates batched-padding effect).
    _r0 = margin_track_single(
        pids_P_all[0],
        matg['traj_tokens']('P', 0)[2])
    _d0 = float(np.abs(
        _r0 - z26['mlg_s0_P'][0]
        .astype(np.float64)).max())
    log('REPRO row0 full-track '
        'max|d|=%.3e' % _d0)
    _ids1 = torch.tensor(
        [list(pids_P_all[0])],
        device='cuda')
    _f1 = []

    def _cap1(mod, inp, out):
        _o = out[0] if isinstance(
            out, tuple) else out
        _f1.append(
            _o[:, -1, :].detach())

    _hk1 = model_g.model.layers[17] \
        .register_forward_hook(_cap1)
    with torch.inference_mode():
        model_g(_ids1, use_cache=False)
    _hk1.remove()
    with torch.inference_mode():
        _h1 = norm_g(_f1[0]).float() \
            .cpu().numpy()
    _m1 = float((_h1 @ w_dn_g)[0])
    log('REPRO single-capture '
        'margin=%.4f | z26 idx17=%.4f '
        'idx18=%.4f idx40=%.4f'
        % (_m1, float(z26['mlg_s0_P'][0, 17, 0]),
           float(z26['mlg_s0_P'][0, 18, 0]),
           float(z26['mlg_s0_P'][0, 40, 0])))
    # (c) batched padded forward: does row0
    # state change when batched with a
    # different-length row?
    _ids2, _mk2 = _pad_batch(
        [pids_P_all[0], pids_P_all[1]])
    _f2 = []

    def _cap2(mod, inp, out):
        _o = out[0] if isinstance(
            out, tuple) else out
        _f2.append(
            _o[:, -1, :].detach())

    _hk2 = model_g.model.layers[17] \
        .register_forward_hook(_cap2)
    with torch.inference_mode():
        model_g(torch.tensor(
            _ids2, device='cuda'),
            attention_mask=torch.tensor(
                _mk2, device='cuda'),
            use_cache=False)
    _hk2.remove()
    with torch.inference_mode():
        _h2 = norm_g(_f2[0]).float() \
            .cpu().numpy()
    _m2 = _h2 @ w_dn_g
    log('REPRO batch2 margins=%.4f/%.4f '
        '(row0 batched vs single %.4f; '
        'len0=%d len1=%d)'
        % (float(_m2[0]), float(_m2[1]),
           _m1, len(pids_P_all[0]),
           len(pids_P_all[1])))
    # (d) unperturbed generation vs z26
    # frozen base (bit-exact?)
    _g0 = gen_batch_g2(
        [pids_P_all[j] for j in range(4)])
    _same = [pad12(_g0[j]) == pad12(
        [int(v) for v in
         z26['gen_base_P'][j]])
        for j in range(4)]
    log('REPRO gen bit-exact vs z26 '
        'base: %s' % _same)
    import sys
    sys.exit(0)
log('== PART B1: state capture ==')
rows_cap = [pids_P_all[j]
            for j in range(NP_CAP)]
base_states = capture_states(
    rows_cap, CAP_L, swap_layers=None)
log('B1 base capture done (%d rows)'
    % len(rows_cap))
swap_states = capture_states(
    rows_cap, CAP_L, swap_layers=SWAP_L)
log('B1 swap4 capture done')
dvec = {}
norms = {}
for l in CAP_L:
    d = (swap_states[l][:NP_B]
         - base_states[l][:NP_B]) \
        if SMOKE else \
        (swap_states[l] - base_states[l])
    dvec[l] = d.astype(np.float32)
    norms[l] = float(np.median(
        np.linalg.norm(dvec[l], axis=1)))
    log('B1 dvec L%02d med||d||=%.4f'
        % (l, norms[l]))
# L17 baseline cross-check vs 3132 npz
# [rev-3133a] state rel-L2 is noisy across
# sessions (SMOKE: 0.0537, bf16 kernel-path
# spread) -> record only. Semantic check at
# margin level: norm_g(state)@w_dn vs z26
# mlg[L17, k=0]; margin dot averages the
# element noise -> tight gate meaningful.
# dvec/transplant are same-session, so
# cross-session noise cancels internally.
_b17 = base_states[17][:len(rows_cap)] \
    - z32['hout_l17'][:len(rows_cap)]
_rel = np.linalg.norm(_b17, axis=1) \
    / np.maximum(np.linalg.norm(
        z32['hout_l17'][:len(rows_cap)]
        .astype(np.float64), axis=1), 1e-9)
dl = float(_rel.max())
log('B1 base L17 vs 3132 hout max '
    'rel-L2=%.4f (record only)' % dl)
_st = torch.tensor(
    base_states[17][:len(rows_cap)],
    device='cuda', dtype=torch.bfloat16)
with torch.inference_mode():
    _h = norm_g(_st).float().cpu().numpy()
_mine = _h @ w_dn_g
_zp = z26['mlg_s0_P'][:len(rows_cap), :, 0] \
    .astype(np.float64)
# [rev-3133g] mapping DECIDED by R7 32-row
# median: capture at layers[17] OUTPUT =
# profile idx 18 (seq[L] = layers[L-1]
# output). REPRO row0's 0.0106 match to
# idx17 was a kernel-offset coincidence
# (its true idx18 distance was 0.19, the
# noise tail); idx17 row spread is
# 0.01-0.43 (off-layer). Residual capture
# vs z26 difference is bf16 prompt-only
# vs full-track kernel-path noise, med
# ~0.09 margin units.
log('B1 diag: _mine[:4]=%s var=%.4f '
    'nan=%d'
    % (' '.join('%.3f' % v
                for v in _mine[:4]),
       float(np.nanvar(_mine)),
       int(np.isnan(_mine).sum())))
log('B1 diag: _zp[:,18][:4]=%s'
    % ' '.join('%.3f' % v
               for v in _zp[:4, 18]))
_med_diff = np.array(
    [float(np.median(np.abs(
        _mine - _zp[:, L])))
     for L in range(NLG + 1)])
best_i = int(np.argmin(_med_diff))
log('B1 capture best-matching profile '
    'idx %d (med|d| %.4f)'
    % (best_i, _med_diff[best_i]))
assert best_i == 18, best_i
assert _med_diff[18] < 0.15, \
    _med_diff[18]
dmar = float(np.abs(_mine
                    - _zp[:, 18]).max())
log('B1 base L17 margin vs z26 idx18 '
    'max|d|=%.4f (record, kernel-noise '
    'tail)' % dmar)
# energy of dvec17 in co50 subspace
d17 = dvec[17].astype(np.float64)
e_full = float((d17 * d17).sum(1).mean())
e_co = float(
    (d17[:, co50] ** 2).sum(1).mean())
frac_co = e_co / max(e_full, 1e-12)
log('B1 dvec17 energy in co50: %.4f '
    '(scalar top50 injection touched '
    '%.1f%% of state diff L2)'
    % (frac_co, 100.0 * frac_co))
del base_states, swap_states
gc.collect()
# [rev-3133h] checkpoint B1 (dvec drives
# all downstream transplant stages)
_meta = {'smoke': SMOKE, 'np': NP,
         'np_cap': NP_CAP, 'np_b': NP_B}
if CK['meta'] and CK['meta'] != _meta:
    log('CKPT meta mismatch %s vs %s -> '
        'discard stale ckpt'
        % (CK['meta'], _meta))
    CK = {'done': [], 'data': {},
          'meta': _meta}
CK['meta'] = _meta
ck_save('b1', {
    'dvec': {str(l): dvec[l]
             for l in CAP_L},
    'norms': norms, 'frac_co': frac_co,
    'dl': dl, 'dmar': dmar})

# ================================================================
# PART B2/B3: transplant trials
# ================================================================
# [rev-3133f] session-internal generation
# baseline: REPRO showed unperturbed
# generation vs the z26 frozen base is
# only 3/4 bit-exact (cross-phase drift
# at kernel level) -> same-verdicts must
# compare against an in-session
# unperturbed run; z26 base kept as a
# drift RECORD only (hard-limit note).
rows_B = [pids_P_all[j]
          for j in range(NP_B)]
_B2B = CK['data'].get('b2_base')
if _B2B is not None:
    gen_base_sess_P = _B2B['gens']
    xphase_P = _B2B['xphase']
    log('B2 base RESUMED from ckpt '
        '(xphase_P=%.4f)' % xphase_P)
else:
    gen_base_sess_P = []
    for b0 in range(0, NP_B, GEN_BATCH):
        gen_base_sess_P.extend(gen_batch_g2(
            rows_B[b0:b0 + GEN_BATCH]))
    _xp = [pad12(gen_base_sess_P[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_P'][j]])
           for j in range(NP_B)]
    xphase_P = float(np.mean(_xp))
    log('B2 xphase drift record: '
        'session-base vs z26 base '
        'bit-match %.4f (%d/%d)'
        % (xphase_P, int(np.sum(_xp)),
           NP_B))
    ck_save('b2_base', {
        'gens': gen_base_sess_P,
        'xphase': xphase_P})
base12_P = [pad12(gen_base_sess_P[j])
            for j in range(NP_B)]
TRIALS_B = [
    ('t17_s0_d05',
     [(17, dvec[17], 0.5, 'step0')]),
    ('t17_s0_d10',
     [(17, dvec[17], 1.0, 'step0')]),
    ('t17_s0_d20',
     [(17, dvec[17], 2.0, 'step0')]),
    ('t17_as_d10',
     [(17, dvec[17], 1.0, 'allstep')]),
    ('t38_s0_d10',
     [(38, dvec[38], 1.0, 'step0')]),
    ('tJ1_s0',
     [(17, dvec[17], 1.0, 'step0'),
      (38, dvec[38], 1.0, 'step0')]),
    ('tJ2_s0',
     [(17, dvec[17], 1.0, 'step0'),
      (29, dvec[29], 1.0, 'step0'),
      (33, dvec[33], 1.0, 'step0'),
      (38, dvec[38], 1.0, 'step0')])]
tr_res = {}
fstep_store = {}
for (tname, ivecs) in TRIALS_B:
    if tname in CK['data'].get(
            'b2_trials', {}):
        _rec = CK['data']['b2_trials'][
            tname]
        tr_res[tname] = _rec['tr']
        fstep_store['fstep_%s' % tname] \
            = _rec['fstep']
        log('B %s RESUMED chg=%.4f'
            % (tname, _rec['tr']['chg']))
        continue
    gen_l = []
    for b0 in range(0, NP_B, GEN_BATCH):
        rows = rows_B[b0:b0 + GEN_BATCH]
        iv = [(l, dv[b0:b0 + len(rows)],
               s, m) for (l, dv, s, m)
              in ivecs]
        gen_l.extend(gen_batch_g2(
            rows, inj_vec=iv))
    chg_l, first_l, fstep, cum = \
        trial_metrics(gen_l, base12_P)
    tr_res[tname] = {
        'chg': chg_l, 'first': int(first_l),
        'cum': [float(v) for v in cum]}
    fstep_store['fstep_%s' % tname] = fstep
    log('B %s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    CK['data'].setdefault(
        'b2_trials', {})[tname] = {
            'tr': tr_res[tname],
            'fstep': fstep}
    ck_save('b2_trials',
            CK['data']['b2_trials'])
    torch.cuda.empty_cache()
    gc.collect()

chg17 = tr_res['t17_s0_d10']['chg']
chg17_05 = tr_res['t17_s0_d05']['chg']
chg17_20 = tr_res['t17_s0_d20']['chg']
chg17_as = tr_res['t17_as_d10']['chg']
chg38 = tr_res['t38_s0_d10']['chg']
chgJ1 = tr_res['tJ1_s0']['chg']
chgJ2 = tr_res['tJ2_s0']['chg']
predJ1 = 1.0 - (1.0 - chg17) \
    * (1.0 - chg38)
if not SMOKE:
    if chg17 >= TR_FULL:
        b_trans = 'transplant_l17_full'
    elif chg17 >= TR_PART:
        b_trans = 'transplant_l17_partial'
    else:
        b_trans = 'transplant_l17_null'
    dose_ok = (chg17_05 <= chg17
               + DOSE_SLACK) and \
              (chg17 <= chg17_20
               + DOSE_SLACK)
    b_dose = ('dose_monotone' if dose_ok
              else 'dose_nonmonotone')
    b_step = ('step_transplant_allstep_higher'
              if chg17_as - chg17 > 0.10
              else 'step_transplant_equivalent')
    b38 = ('l38_transplant_lower'
           if chg38 < 0.9 * max(chg17, 1e-9)
           else 'l38_transplant_comparable')
    if abs(chgJ1 - predJ1) <= ADD_TOL:
        b_joint = 'joint_additive'
    elif chgJ1 > predJ1:
        b_joint = 'joint_superadditive'
    else:
        b_joint = 'joint_subadditive'
    log('B-GATE: t17=%.4f t38=%.4f J1=%.4f '
        '(pred %.4f) J2=%.4f -> %s/%s/%s/%s/%s'
        % (chg17, chg38, chgJ1, predJ1, chgJ2,
           b_trans, b_dose, b38, b_joint,
           b_step))
else:
    b_trans = b_dose = b_step = b38 \
        = b_joint = 'smoke_subset'
    log('B-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C1: tf fork-layer profiles A1 vs P
# ================================================================
log('== PART C1: tf fork profiles ==')
tf_idx = np.sort(np.random.default_rng(
    TF_SEED).choice(NP, TF_N,
                    replace=False)).astype(
    np.int64)
log('C1 tf_idx sha8=%s' % hashlib.sha256(
    tf_idx.tobytes()).hexdigest()[:8])
ml_base_chk = {}
for dc in DIRS:
    m0 = z26['mlg_s0_%s' % dc][tf_idx[:4]] \
        .astype(np.float64)
    _rep = np.stack([
        margin_track_single(
            (pids_P_all if dc == 'P'
             else pids_A1_all)[int(j)],
            matg['traj_tokens'](dc, int(j))[2])
        for j in tf_idx[:4]])
    dchk = float(np.abs(_rep - m0).max())
    log('C1 repro %s max|d|=%.2e'
        % (dc, dchk))
    assert dchk <= 1e-4, (dc, dchk)


def fork_stats(ml_base, ml_swap):
    """ml_*: (NLG+1, npts). Returns dict of
    per-depth fork descriptors at k=0."""
    b0 = ml_base[:, 0].astype(np.float64)
    s0 = ml_swap[:, 0].astype(np.float64)
    dm0 = s0 - b0
    div = [L for L in range(NLG + 1)
           if abs(b0[L]) > FLIP_GUARD
           and (s0[L] > 0) != (b0[L] > 0)]
    first_div = min(div) if div else -1
    last_div = max(div) if div else -1
    a40 = abs(dm0[NLG])
    if a40 < DM40_MIN:
        emerge = -2
    else:
        thr = EMERGE_FRAC * a40
        idxs = [L for L in range(NLG + 1)
                if abs(dm0[L]) > thr]
        emerge = min(idxs) if idxs else -1
    return {'first_div': first_div,
            'last_div': last_div,
            'emerge': emerge,
            'dm0': dm0}


prof = {}
for dc in DIRS:
    if dc in CK['data'].get('c1_prof',
                            {}):
        _rec = CK['data']['c1_prof'][dc]
        prof[dc] = {
            'first_div': _rec['first_div'],
            'last_div': _rec['last_div'],
            'emerge': _rec['emerge'],
            'dm0': _rec['dm0']}
        log('C1 %s RESUMED from ckpt'
            % dc)
        continue
    rows_dc = (pids_P_all if dc == 'P'
               else pids_A1_all)
    m0_all = z26['mlg_s0_%s' % dc][tf_idx] \
        .astype(np.float64)
    fd = np.zeros(TF_N, dtype=np.int32)
    ld = np.zeros(TF_N, dtype=np.int32)
    em = np.zeros(TF_N, dtype=np.int32)
    dm0_all = np.zeros((TF_N, NLG + 1),
                       dtype=np.float64)
    for jj in range(TF_N):
        j = int(tf_idx[jj])
        ml_sw = margin_track_single(
            rows_dc[j],
            matg['traj_tokens'](dc, j)[2],
            swap_layers=SWAP_L)
        fs = fork_stats(m0_all[jj], ml_sw)
        fd[jj] = fs['first_div']
        ld[jj] = fs['last_div']
        em[jj] = fs['emerge']
        dm0_all[jj] = fs['dm0']
        if jj % 32 == 0:
            log('C1 %s row %d/%d'
                % (dc, jj, TF_N))
    prof[dc] = {'first_div': fd,
                'last_div': ld,
                'emerge': em,
                'dm0': dm0_all}
    CK['data'].setdefault('c1_prof',
                          {})[dc] = {
        'first_div': fd, 'last_div': ld,
        'emerge': em, 'dm0': dm0_all}
    ck_save('c1_prof',
            CK['data']['c1_prof'])
    log('C1 %s: med first_div=%s med '
        'last_div=%s med emerge=%s'
        % (dc, int(np.median(
            fd[fd >= 0])) if (fd >= 0)
           .any() else -1,
           int(np.median(
            ld[ld >= 0])) if (ld >= 0)
           .any() else -1,
           int(np.median(
            em[em >= 0])) if (em >= 0)
           .any() else -1))

# P-side link to 3131 frozen dm (k=12)
med_abs_0 = np.median(
    np.abs(prof['P']['dm0']), axis=0)
med_abs_dm31 = z31['med_abs_dm'] \
    .astype(np.float64)
sp31 = float(np.corrcoef(
    np.argsort(np.argsort(med_abs_0)),
    np.argsort(np.argsort(
        med_abs_dm31)))[0, 1])
log('C1 P-link: spearman(med|dm_k0|, '
    'z31 med_abs_dm[k=12])=%.3f (record; '
    'k differs)' % sp31)


def rankv(x):
    return np.argsort(
        np.argsort(x)).astype(np.float64)


medA1 = np.median(np.abs(
    prof['A1']['dm0']), axis=0)
medP = med_abs_0
sp_prof = float(np.corrcoef(
    rankv(medA1), rankv(medP))[0, 1])
fdA1 = prof['A1']['first_div']
fdP = prof['P']['first_div']
medE_A1 = int(np.median(
    fdA1[fdA1 >= 0])) if (fdA1 >= 0) \
    .any() else -1
medE_P = int(np.median(
    fdP[fdP >= 0])) if (fdP >= 0) \
    .any() else -1
if not SMOKE:
    sym_ok = (sp_prof >= RHO_PROF) and \
             (abs(medE_A1 - medE_P)
              <= MED_E_TOL)
    c_prof = ('a1_forklayer_symmetric'
              if sym_ok
              else 'a1_forklayer_asymmetric')
    log('C1-GATE: sp_prof=%.3f medE A1=%d '
        'P=%d -> %s'
        % (sp_prof, medE_A1, medE_P,
           c_prof))
else:
    c_prof = 'smoke_subset'
    log('C1-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C2: A1 single-layer spectrum
# ================================================================
log('== PART C2: A1 spectrum ==')
sel64 = np.sort(np.random.default_rng(
    SEL_SEED_OLD).choice(NP, 64,
                         replace=False)) \
    .astype(np.int64)
sha64 = hashlib.sha256(
    sel64.tobytes()).hexdigest()[:8]
assert sha64 == SEL64_SHA, sha64
rest = np.setdiff1d(np.arange(NP), sel64)
sel192 = np.sort(np.random.default_rng(
    SEL_SEED_NEW).choice(rest, 192,
                         replace=False)) \
    .astype(np.int64)
sel256 = np.sort(np.concatenate(
    [sel64, sel192])).astype(np.int64)
sel_sha = hashlib.sha256(
    sel256.tobytes()).hexdigest()[:8]
assert sel_sha == SEL256_SHA, sel_sha
log('C2 sel256 sha8=%s (paired with '
    '3132 P spectrum)' % sel_sha)
chgA1 = {}
firstA1 = {}
sameA1 = {}
# [rev-3133f] session-internal A1
# baseline (same rationale as B2),
# computed once outside the layer loop.
rows_l = [pids_A1_all[j]
          for j in sel256[:SPEC_NP]]
_C2B = CK['data'].get('c2_base')
if _C2B is not None:
    gen_base_sess_A = _C2B['gens']
    xphase_A = _C2B['xphase']
    log('C2 base RESUMED from ckpt '
        '(xphase_A=%.4f)' % xphase_A)
else:
    gen_base_sess_A = []
    for b0 in range(0, len(rows_l),
                    GEN_BATCH):
        gen_base_sess_A.extend(gen_batch_g2(
            rows_l[b0:b0 + GEN_BATCH]))
    _xa = [pad12(gen_base_sess_A[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_A1']
                  [int(sel256[j])]])
           for j in range(len(rows_l))]
    xphase_A = float(np.mean(_xa))
    log('C2 xphase drift record: '
        'session-base vs z26 A1 base '
        'bit-match %.4f (%d/%d)'
        % (xphase_A, int(np.sum(_xa)),
           len(rows_l)))
    ck_save('c2_base', {
        'gens': gen_base_sess_A,
        'xphase': xphase_A})
base12_A = [pad12(gen_base_sess_A[j])
            for j in range(len(rows_l))]
for l in LAYERS_C:
    if str(l) in CK['data'].get(
            'c2_layers', {}):
        _rec = CK['data']['c2_layers'][
            str(l)]
        chgA1[l] = _rec['chg']
        firstA1[l] = _rec['first']
        sameA1[l] = 1 - _rec['chg']
        log('C2 A1 L%02d RESUMED chg=%.4f'
            % (l, _rec['chg']))
        continue
    gen_l = []
    for b0 in range(0, len(rows_l),
                    GEN_BATCH):
        batch = rows_l[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch, swap_layers=[l]))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base12_A)
    chgA1[l] = chg_l
    firstA1[l] = int(first_l)
    sameA1[l] = 1 - chg_l
    log('C2 A1 L%02d: chg=%.4f first=%d'
        % (l, chg_l, first_l))
    CK['data'].setdefault(
        'c2_layers', {})[str(l)] = {
            'chg': chg_l,
            'first': int(first_l)}
    ck_save('c2_layers',
            CK['data']['c2_layers'])
    torch.cuda.empty_cache()
    gc.collect()
vecA1 = np.array([chgA1[l]
                  for l in LAYERS_C])
vecP = np.array([REF32_CHG256[l]
                 for l in LAYERS_C])
sp_spec = float(np.corrcoef(
    rankv(vecA1), rankv(vecP))[0, 1])
if not SMOKE:
    c_spec = ('a1_transfer_high'
              if sp_spec >= RHO_TRANS
              else 'a1_transfer_low')
    log('C2-GATE: spearman(A1, P chg256)'
        '=%.3f -> %s' % (sp_spec, c_spec))
else:
    c_spec = 'smoke_subset'
    log('C2-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C3: L17 coords injection on A1
# ================================================================
log('== PART C3: L17 coords on A1 ==')
rows_C3 = [pids_A1_all[j]
           for j in sel256[:C3_N]]
_C3B = CK['data'].get('c3_base')
if _C3B is not None:
    gen_base_sess_C3 = _C3B['gens']
    xphase_C3 = _C3B['xphase']
    log('C3 base RESUMED from ckpt '
        '(xphase_C3=%.4f)' % xphase_C3)
else:
    gen_base_sess_C3 = []
    for b0 in range(0, len(rows_C3),
                    GEN_BATCH):
        gen_base_sess_C3.extend(gen_batch_g2(
            rows_C3[b0:b0 + GEN_BATCH]))
    _xc = [pad12(gen_base_sess_C3[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_A1']
                  [int(sel256[j])]])
           for j in range(len(rows_C3))]
    xphase_C3 = float(np.mean(_xc))
    log('C3 xphase drift record: '
        'session-base vs z26 A1 base '
        'bit-match %.4f (%d/%d)'
        % (xphase_C3, int(np.sum(_xc)),
           len(rows_C3)))
    ck_save('c3_base', {
        'gens': gen_base_sess_C3,
        'xphase': xphase_C3})
base12_C3 = [pad12(gen_base_sess_C3[j])
             for j in range(len(rows_C3))]
# delta reuse: 0.10 x mean row-l2 of 3132
# hout_l17 (frozen, sha-checked)
row_l2 = np.linalg.norm(
    z32['hout_l17'][:NP].astype(np.float64),
    axis=1)
delta_l17 = 0.10 * float(row_l2.mean())
log('C3 delta_l17=%.4f (0.10 x mean '
    'row_l2, 3132-frozen)' % delta_l17)
c3_res = {}
for sgn in (1, -1):
    _k = 'sgn%+d' % sgn
    if _k in CK['data'].get('c3', {}):
        c3_res[_k] = CK['data']['c3'][_k]
        fstep_store['fstep_c3_%s' % _k] \
            = CK['data']['c3_fs'][_k]
        log('C3 %s RESUMED chg=%.4f'
            % (_k, c3_res[_k]['chg']))
        continue
    gen_l = []
    for b0 in range(0, len(rows_C3),
                    GEN_BATCH):
        batch = rows_C3[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=(17, co50, delta_l17, sgn,
                 'step0')))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base12_C3)
    c3_res['sgn%+d' % sgn] = {
        'chg': chg_l, 'first': int(first_l)}
    fstep_store['fstep_c3_sgn%+d' % sgn] \
        = fstep
    log('C3 sgn %+d: chg=%.4f first=%d'
        % (sgn, chg_l, first_l))
    CK['data'].setdefault('c3', {})[_k] \
        = c3_res[_k]
    CK['data'].setdefault('c3_fs', {})[_k] \
        = fstep
    ck_save('c3', CK['data']['c3'])
if not SMOKE:
    c3_best = max(v['chg']
                  for v in c3_res.values())
    c_coord = ('a1_l17coord_rewrites'
               if c3_best >= TR_PART
               else 'a1_l17coord_null')
    log('C3-GATE: best=%.4f -> %s'
        % (c3_best, c_coord))
else:
    c_coord = 'smoke_subset'
    log('C3-GATE: SKIPPED in SMOKE')

# ================================================================
# dumps
# ================================================================
# [rev-3133h] integrity: resumed ckpt
# must cover every gated stage before
# verdict assembly
if not SMOKE:
    assert len(tr_res) == len(TRIALS_B), \
        (sorted(tr_res), len(TRIALS_B))
    assert len(c3_res) == 2, c3_res
    assert all(dc in prof for dc in DIRS)
    assert all(l in chgA1
               for l in LAYERS_C)
    log('CKPT integrity assert ok')
runtime = time.time() - T0
verdict = '|'.join([
    'a_3132_ok', b_trans, b_dose, b38,
    b_joint, b_step, c_prof, c_spec,
    c_coord, 'coverage_full'])
log('VERDICT: %s' % verdict)
result = {
    'name': NAME,
    'phase': 3133,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'res32_sha8': sha32,
        'asserts_3132': 'ok',
        'co50_sha8': co50_sha},
    'part_b': {
        'dvec_med_norm': {
            str(l): norms[l]
            for l in CAP_L},
        'dvec17_co50_energy': frac_co,
        'base17_vs_3132_maxrel': dl,
        'xphase_base_match_P': xphase_P,
        'trials': tr_res,
        'chg17_s0_d10': chg17,
        'chg17_s0_d05': chg17_05,
        'chg17_s0_d20': chg17_20,
        'chg17_as_d10': chg17_as,
        'chg38_s0_d10': chg38,
        'chgJ1_s0': chgJ1,
        'chgJ2_s0': chgJ2,
        'predJ1_indep': predJ1,
        'b_trans': b_trans,
        'b_dose': b_dose,
        'b38': b38,
        'b_joint': b_joint,
        'b_step': b_step},
    'part_c': {
        'tf_idx_sha8': hashlib.sha256(
            tf_idx.tobytes())
        .hexdigest()[:8],
        'tf_idx': [int(v)
                   for v in tf_idx],
        'p31_link_spearman': sp31,
        'sp_prof': sp_prof,
        'medE_A1': medE_A1,
        'medE_P': medE_P,
        'med_dm0_A1': [float(v)
                       for v in medA1],
        'med_dm0_P': [float(v)
                      for v in medP],
        'first_div_A1': [int(v)
                         for v in fdA1],
        'first_div_P': [int(v)
                        for v in fdP],
        'c_prof': c_prof,
        'sel_sha8': sel_sha,
        'xphase_base_match_A1': xphase_A,
        'xphase_base_match_c3': xphase_C3,
        'chgA1': {str(l): float(chgA1[l])
                  for l in LAYERS_C},
        'firstA1': {str(l): int(firstA1[l])
                    for l in LAYERS_C},
        'sp_spec': sp_spec,
        'c_spec': c_spec,
        'delta_l17': delta_l17,
        'c3_trials': c3_res,
        'c_coord': c_coord},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'co50': co50,
    'tf_idx': tf_idx,
    'dvec17': dvec[17][:NP_B],
    'dvec29': dvec[29][:NP_B],
    'dvec33': dvec[33][:NP_B],
    'dvec38': dvec[38][:NP_B],
    'dvec_med_norm': np.array(
        [norms[l] for l in CAP_L]),
    'sel256': sel256}
for k, v in fstep_store.items():
    npz_out[k] = v
for l in CAP_L:
    npz_out['dvec_full_%d' % l] = \
        dvec[l].astype(np.float16)
for dc in DIRS:
    npz_out['fdm0_%s' % dc] = \
        prof[dc]['dm0'].astype(np.float32)
    npz_out['fdiv_%s' % dc] = \
        prof[dc]['first_div']
    npz_out['emerge_%s' % dc] = \
        prof[dc]['emerge']
for l in LAYERS_C:
    npz_out['chgA1_%d' % l] = np.float64(
        chgA1[l])
np.savez(os.path.join(OUT,
                      'p131_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p131_readout.npz (%d keys)'
    % len(npz_out))
# [rev-3133h] all data persisted -> drop
# ckpt so a future run starts fresh
# (P3133_CKPT_KEEP=1 keeps it for a
# full-RESUME smoke validation)
if os.path.exists(CKPTF) \
        and not os.environ.get(
            'P3133_CKPT_KEEP'):
    os.remove(CKPTF)
    log('CKPT removed (data fully '
        'persisted)')
log('P3133 DONE (%.1fs)'
    % (time.time() - T0))
