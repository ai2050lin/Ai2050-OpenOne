# -*- coding: utf-8 -*-
"""Phase 3135 (Omega-P133): cross-layer
conduction of non-L17 carrier dvecs +
co36 necessity subset ablation & co50
overlap + multi-step injection windows.

Preregistered in 3134 MEMO section 5:
(1) layer-to-layer conduction: inject
each CAP_L dvec (d2) teacher-forced at
its own layer, read downstream full-layer
state shifts, compare against the swap4
dvec field (cos + amplitude ratio) ->
decoupled vs attenuated vs mixed;
(2) co36 (L35 A1-P top50) necessity:
subset ablation (top25/bottom25/random25)
on A1 + overlap decomposition of L17
co50 (full set minus co36, and the
intersection cap) on A1;
(3) multi-step injection windows of L17
dvec (joint steps, full & dose-shared
scales) vs same-session allstep ceiling.

Reuses verified 3133/3134 infrastructure
(materials factory, gen_batch_g2,
single-sample capture, trial_metrics,
ckpt/RESUME)."""
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
        + r'\rdc_query_construction_20260913')
NAME = ('omega_p133_conduction_'
        'co36ablation_window')
SMOKE = os.environ.get('P3135_SMOKE',
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
D33 = RDIR + r'\phase3133' \
      r'\omega_p131_transplant_a1fork_migrate'
D34 = RDIR + r'\phase3134' \
      r'\omega_p132_carrier_matrix_' \
      r'forkcoord_stepscan'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3135', NAME)
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
# [3134a] same discipline as rev-3133h:
# every expensive stage checkpoints after
# completion and is skipped on restart.
import pickle  # noqa: E402
CKPTF = os.path.join(OUT, 'p133_ckpt.pkl')
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
SCAN_N = 4 if SMOKE else 128
TF_N = 4 if SMOKE else 128
GEN_BATCH = 4 if SMOKE else 32
DIRS = ('P', 'A1')

# carrier layers (dvec injection layers;
# conduction readout = all 40 layer
# outputs, profile idx 1..40)
SWAP_L = [8, 9, 13, 29]
CAP_L = [17, 29, 33, 38]
DOSE_COND = 2.0          # conduction inj dose (3134 d2 anchor)
ALL_L = list(range(40))  # capture all layer outputs
FORK_PROF_IDX = 36       # medE A1 (3133)
FORK_LAYER = FORK_PROF_IDX - 1  # layers[35]
TOPK_COORD = 50
SUBSET_K = 25
N_RANDOM = 3
RNG_SEED = 3135
# conduction gates (cos/amp med over
# direct downstream layers r=il+1..il+3,
# non-L17 carriers; frozen before obs)
COND_COS_HI = 0.40
COND_RHO_HI = 0.30
COND_COS_LO = 0.20
COND_RHO_LO = 0.10
# subset-ablation gates (fractions of
# full50 same-session chg)
CONC_FRAC_HI = 0.6
CONC_FRAC_LO = 0.3
NEC_FRAC = 0.5
SUFF_FRAC = 0.8
# multi-step windows: forward indices,
# 0 = prompt forward, k>=1 = decode step
# k-1 (per-step full dose 1.0)
WINDOWS = [[0, 1, 2], [0, 1, 2, 3],
           [0, 1, 2, 3, 4, 5],
           list(range(0, 9))]
WIN_NAMES = ['w2', 'w3', 'w5', 'w8']
W_SHARE = 1.0 / 3.0      # dose-shared scale
ALLSTEP_FRAC = 0.8       # window-near-allstep
CUM_GAP = 0.05
SHARE_GAP = 0.05

TR_PART = 0.10
ADD_TOL = 0.15
FLIP_GUARD = 0.1

TF_SEED = 3135

# ---------------- frozen 3134 anchors --
EXP_V34 = ('a_3133_ok|carrier_l17_max|'
           'dose_monotone_all|j2_session_'
           'match|pair_additive|forkcoord_'
           'p_null|stepscan_effective|'
           'traj_swap_early30|coverage_full')
RES34_SHA = 'd331bd1b'
CO36_34_SHA = '84a1e1a3'
SEL256_SHA = 'e34d2588'
CO50_SHA = '52b126af'
# exact fractions from the 3134 full run
CHG17_D20 = 172 / 672.0
CHG29_D20 = 42 / 672.0
CHG33_D20 = 33 / 672.0
CHG38_D20 = 60 / 672.0
C1P_34 = 5 / 128.0
C1A1_34 = 1.0
DELTA36_34 = 1.694018277446071
SCAN1_34 = 35 / 128.0
CFULL_34 = 184 / 672.0
K0SWAP_34 = 30
K0INJ_34 = 38
C3_BEST = 41 / 128.0
DELTA_L17 = 0.637683315669971

# ---------------- seal ------------------
SEAL = {
    'phase': 3135,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'SWAP_L': SWAP_L, 'CAP_L': CAP_L,
        'DOSE_COND': DOSE_COND,
        'ALL_L': ALL_L,
        'FORK_PROF_IDX': FORK_PROF_IDX,
        'FORK_LAYER': FORK_LAYER,
        'TOPK_COORD': TOPK_COORD,
        'SUBSET_K': SUBSET_K,
        'N_RANDOM': N_RANDOM,
        'RNG_SEED': RNG_SEED,
        'COND_COS_HI': COND_COS_HI,
        'COND_RHO_HI': COND_RHO_HI,
        'COND_COS_LO': COND_COS_LO,
        'COND_RHO_LO': COND_RHO_LO,
        'CONC_FRAC_HI': CONC_FRAC_HI,
        'CONC_FRAC_LO': CONC_FRAC_LO,
        'NEC_FRAC': NEC_FRAC,
        'SUFF_FRAC': SUFF_FRAC,
        'WINDOWS': WINDOWS,
        'WIN_NAMES': WIN_NAMES,
        'W_SHARE': W_SHARE,
        'ALLSTEP_FRAC': ALLSTEP_FRAC,
        'CUM_GAP': CUM_GAP,
        'SHARE_GAP': SHARE_GAP,
        'TR_PART': TR_PART,
        'ADD_TOL': ADD_TOL,
        'FLIP_GUARD': FLIP_GUARD,
        'TF_SEED': TF_SEED},
    'anchors': {
        'res34_verdict': EXP_V34,
        'res34_sha8': RES34_SHA,
        'co36_34_sha8': CO36_34_SHA,
        'sel256_sha8': SEL256_SHA,
        'co50_sha8': CO50_SHA,
        'chg17_d20': CHG17_D20,
        'chg29_d20': CHG29_D20,
        'chg33_d20': CHG33_D20,
        'chg38_d20': CHG38_D20,
        'c1p_34': C1P_34,
        'c1a1_34': C1A1_34,
        'delta36_34': DELTA36_34,
        'scan1_34': SCAN1_34,
        'cfull_34': CFULL_34,
        'k0swap_34': K0SWAP_34,
        'k0inj_34': K0INJ_34,
        'c3_best': C3_BEST,
        'delta_l17': DELTA_L17},
    'prereg': ('3134 MEMO section 5: '
               '(1) cross-layer conduction '
               'of non-L17 carrier dvecs: '
               'inject each CAP_L dvec (d2) '
               'teacher-forced at its own '
               'layer, read downstream '
               'full-layer state shifts, '
               'compare vs swap4 dvec field '
               '(cos + amplitude ratio) -> '
               'decoupled/attenuated/mixed. '
               '(2) co36 necessity subset '
               'ablation (top25/bottom25/'
               'random25) on A1 + co50 '
               'overlap decomposition '
               '(full\\co36, cap) on A1. '
               '(3) multi-step injection '
               'windows of L17 dvec (joint '
               'steps full dose + dose-'
               'shared scale) vs same-'
               'session allstep ceiling. '
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
assert z26['mlg_s0_A1'].shape == (672, 41, 13)
assert z26['gen_base_P'].shape == (672, 12)
assert z26['gen_base_A1'].shape == (672, 12)
z32 = np.load(D32 + r'\p130_readout.npz',
              allow_pickle=False)
assert z32['hout_l17'].shape == (672, 4096)
assert z32['co50'].shape == (50,)
z34 = np.load(D34 + r'\p132_readout.npz',
              allow_pickle=False)
assert z34['co36'].shape == (50,)
assert z34['dvec_full_17'].shape == (672, 4096)
res34 = json.load(io.open(
    D34 + r'\result.json', encoding='utf-8'))
assert res34['smoke'] is False
log('frozen inputs ok (05/26/32/34)')

# ================================================================
# PART A: offline 3134 link asserts
# ================================================================
log('== PART A: 3134 link asserts ==')
V34 = res34['verdict']
assert V34 == EXP_V34, V34
pb34 = res34['part_b']
pc34 = res34['part_c']
pd34 = res34['part_d']
_cm = pb34['chg_matrix']
assert abs(_cm['17'][2] - CHG17_D20) < 1e-9
assert abs(_cm['29'][2] - CHG29_D20) < 1e-9
assert abs(_cm['33'][2] - CHG33_D20) < 1e-9
assert abs(_cm['38'][2] - CHG38_D20) < 1e-9
_c1 = pc34['c1_trials']
assert abs(_c1['P']['chg'] - C1P_34) < 1e-9
assert abs(_c1['A1']['chg'] - C1A1_34) < 1e-9
assert abs(pc34['delta36']
           - DELTA36_34) < 1e-9
assert pc34['best_step'] == '1'
assert abs(pc34['scan']['1']['chg']
           - SCAN1_34) < 1e-9
assert abs(pc34['c3_full']['chg']
           - CFULL_34) < 1e-9
assert pd34['k0_swap_peak'] == K0SWAP_34
assert pd34['k0_inj_peak'] == K0INJ_34
raw34 = io.open(
    D34 + r'\result.json', 'rb').read()
sha34 = hashlib.sha256(
    raw34).hexdigest()[:8]
assert sha34 == RES34_SHA, sha34
co50 = z32['co50'].astype(np.int64)
co50_sha = hashlib.sha256(
    co50.tobytes()).hexdigest()[:8]
assert co50_sha == CO50_SHA, co50_sha
co36_34 = z34['co36'].astype(np.int64)
c36_sha = hashlib.sha256(
    co36_34.tobytes()).hexdigest()[:8]
assert c36_sha == CO36_34_SHA, c36_sha
log('A hard asserts ok (3134 frozen, '
    'result sha8 %s, co50 sha8 %s, '
    'co36-34 sha8 %s)'
    % (sha34, co50_sha, c36_sha))

# ================================================================
# materials factory (3125/.../3133-
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
    inj_vec: per-layer transplants
      (il, dvec_batch (n,HID) fp32, scale,
       mode). mode: 'allstep' = every
       forward; int s = prompt forward +
       decode step s (s==0 -> step0)."""
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
            if mode == 'allstep' \
                    or st['step'] == 0 \
                    or st['step'] == mode:
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
                if isinstance(_m,
                              (list, tuple)):
                    # [3135] window semantics:
                    # explicit forward-index
                    # list (0 = prompt
                    # forward, k>=1 = decode
                    # step k-1); NO implicit
                    # prompt add-on.
                    _hit = _st['step'] in _m
                else:
                    _hit = (_m == 'allstep'
                            or _st['step'] == 0
                            or _st['step']
                            == _m)
                if _hit:
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
    SINGLE-SAMPLE loop (batch=1, zero
    padding): batched left-padded forwards
    measurably change row states (3133
    REPRO; row0 margin +0.52). Single-
    sample forwards replicate z26/3131
    semantics exactly."""
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
    margin (3131/3126 z26 convention:
    ids = prompt + traj, pos0 =
    len(prompt)-1, npts = 13, teacher-
    forced). inject (il, coords, delta,
    sgn) at pos0 (track step k=0) after
    layer il output. Returns (NLG+1, npts)
    float64. Replicates z26 mlg_s0_P to
    0.0 exactly (3132 rev-a)."""
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


def capture_states_inject(rows_all,
                          cap_layers,
                          inj_layer,
                          dvec_batch,
                          scale):
    """Single-sample forwards with a
    per-row dvec injected at inj_layer
    OUTPUT (last position), capturing all
    cap_layers states. Teacher-forced
    prompt-only (no generation). Injection
    is in-place on the layer output, so
    downstream capture hooks always read
    the post-injection field."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    lyr = model_g.model.layers[inj_layer]
    for j in range(len(rows_all)):
        dv_t = torch.as_tensor(
            np.asarray(dvec_batch[j],
                       dtype=np.float32),
            device='cuda') \
            .to(torch.bfloat16) \
            * float(scale)

        def _inj(mod, inp, out,
                 _dv=dv_t):
            o2 = out[0] \
                if isinstance(out,
                              tuple) \
                else out
            o2[0, -1, :] += _dv
            return None

        feats = {l: None
                 for l in cap_layers}
        hooks = [lyr.register_forward_hook(
            _inj)]

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
            hooks.append(
                model_g.model.layers[l]
                .register_forward_hook(
                    _mk(l)))
        try:
            with torch.inference_mode():
                model_g(
                    torch.tensor(
                        [list(rows_all[j])],
                        device='cuda'),
                    use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        for l in cap_layers:
            OUTS[l][j] = feats[l].float() \
                .cpu().numpy()
        del feats, dv_t
    return OUTS


# ================================================================
# PART B1: swap-state capture -> dvec
# [3135] FULL-layer capture (all 40 layer
# outputs, single-sample convention):
# needed as the downstream baseline field
# for the conduction spectra. CAP_L dvecs
# remain the injection vectors (3133/3134
# convention).
# ================================================================
log('== PART B1: state capture (full) ==')
rows_cap = [pids_P_all[j]
            for j in range(NP_CAP)]
CKM = {'smoke': SMOKE, 'np': NP,
       'np_cap': NP_CAP, 'np_b': NP_B}
if CK['meta'] and CK['meta'] != CKM:
    log('CKPT meta mismatch -> discard')
    CK = {'done': [], 'data': {},
          'meta': {}}
CK['meta'] = CKM
_B1K = CK['data'].get('b1')
if _B1K is not None:
    dvec = {int(k): v
            for k, v in
            _B1K['dvec'].items()}
    dvec_full = {
        int(k): v.astype(np.float32)
        for k, v in
        _B1K['dvec_full_fp16'].items()}
    base_states = {
        int(k): v.astype(np.float32)
        for k, v in
        _B1K['base_fp16'].items()}
    norms = {int(k): v for k, v in
             _B1K['norms'].items()}
    log('B1 RESUMED from ckpt '
        '(norms L17=%.4f)' % norms[17])
else:
    base_states = capture_states(
        rows_cap, ALL_L, swap_layers=None)
    log('B1 base capture done (%d rows, '
        '%d layers)' % (len(rows_cap),
                        len(ALL_L)))
    swap_states = capture_states(
        rows_cap, ALL_L,
        swap_layers=SWAP_L)
    log('B1 swap4 capture done')
    dvec = {}
    norms = {}
    dvec_full = {}
    for l in ALL_L:
        d = (swap_states[l][:NP_B]
             - base_states[l][:NP_B]) \
            if SMOKE else \
            (swap_states[l]
             - base_states[l])
        dvec_full[l] = d.astype(
            np.float32)
        if l in CAP_L:
            dvec[l] = d.astype(
                np.float32)
            norms[l] = float(np.median(
                np.linalg.norm(
                    dvec[l], axis=1)))
            log('B1 dvec L%02d '
                'med||d||=%.4f'
                % (l, norms[l]))
    # margin-level semantic check vs z26
    # profile idx 18 (3133 rev-g mapping:
    # capture at layers[17] output =
    # profile idx 18); batched capture is
    # the polluter -> single-sample here
    # must match.
    _st = torch.tensor(
        base_states[17][:NP_CAP],
        device='cuda',
        dtype=torch.bfloat16)
    with torch.inference_mode():
        _h = norm_g(_st).float() \
            .cpu().numpy()
    _mine = _h @ w_dn_g
    _zp = z26['mlg_s0_P'][:NP_CAP, :, 0] \
        .astype(np.float64)
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
    del swap_states
    gc.collect()
    ck_save('b1', {
        'dvec': {str(l): dvec[l]
                 for l in CAP_L},
        'dvec_full_fp16': {
            str(l): dvec_full[l]
            .astype(np.float16)
            for l in ALL_L},
        'base_fp16': {
            str(l): base_states[l]
            .astype(np.float16)
            for l in ALL_L},
        'norms': norms})

# --- session-internal generation base ---
# (same-session baseline discipline,
# rev-3133f; xphase drift record vs z26)
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

# ================================================================
# PART B2: cross-layer conduction
# [3135 prereg 1] inject each CAP_L dvec
# (d2) at its OWN layer, teacher-forced
# prompt-only, capture all 40 downstream
# states; compare the induced shift vs
# the swap4 dvec field (cos + amplitude
# ratio). Direct-downstream gate on
# non-L17 carriers -> decoupled /
# attenuated / mixed.
# ================================================================
log('== PART B2: conduction ==')
cos_spec = {}
rho_spec = {}
for il in CAP_L:
    _K = CK['data'].get('cond_%d' % il)
    if _K is not None:
        cos_spec[il] = _K['cos']
        rho_spec[il] = _K['rho']
        log('B2 cond L%02d RESUMED' % il)
        continue
    n_cap = dvec[il].shape[0]
    h_inj = capture_states_inject(
        rows_cap[:n_cap], ALL_L, il,
        dvec[il], DOSE_COND)
    log('B2 inj L%02d capture done '
        '(%d rows)' % (il, n_cap))
    cos_m = np.full((len(ALL_L), n_cap),
                    np.nan,
                    dtype=np.float32)
    rho_m = np.full((len(ALL_L), n_cap),
                    np.nan,
                    dtype=np.float32)
    for r in ALL_L:
        if r <= il:
            continue
        dh = (h_inj[r].astype(np.float64)
              - base_states[r][:n_cap]
              .astype(np.float64))
        dv = dvec_full[r].astype(
            np.float64)
        ndh = np.linalg.norm(dh, axis=1)
        ndv = np.linalg.norm(dv, axis=1)
        num = np.sum(dh * dv, axis=1)
        den = ndh * ndv
        with np.errstate(
                invalid='ignore',
                divide='ignore'):
            cos_m[r] = np.where(
                den > 1e-12,
                num / den, 0.0)
            rho_m[r] = np.where(
                ndv > 1e-12, ndh / ndv,
                0.0)
    del h_inj
    gc.collect()
    torch.cuda.empty_cache()
    cos_spec[il] = cos_m
    rho_spec[il] = rho_m
    ck_save('cond_%d' % il, {
        'cos': cos_m, 'rho': rho_m})

gmap = {}

if not SMOKE:
    def _direct(il):
        rs = [r for r in (il + 1, il + 2,
                          il + 3) if r < 40]
        cd = float(np.median(
            [float(np.nanmedian(
                cos_spec[il][r]))
             for r in rs]))
        rd = float(np.median(
            [float(np.nanmedian(
                rho_spec[il][r]))
             for r in rs]))
        return cd, rd

    for il in (29, 33, 38):
        cd, rd = _direct(il)
        if cd >= COND_COS_HI \
                and rd >= COND_RHO_HI:
            gmap[il] = 'decoupled'
        elif cd < COND_COS_LO \
                or rd < COND_RHO_LO:
            gmap[il] = 'attenuated'
        else:
            gmap[il] = 'mixed'
        log('B2 gate L%02d: cos=%.4f '
            'rho=%.4f -> %s'
            % (il, cd, rd, gmap[il]))
    _vals = set(gmap.values())
    if _vals == {'decoupled'}:
        c_cond = 'cond_readout_decoupled'
    elif _vals == {'attenuated'}:
        c_cond = 'cond_signal_attenuated'
    else:
        c_cond = 'cond_mixed'
    l17_down = {r: float(np.nanmedian(
        cos_spec[17][r]))
        for r in (18, 19, 20, 25, 30,
                  35, 39)}
    log('B2-GATE: %s; L17 downstream '
        'cos %s (positive control)'
        % (c_cond, ' '.join(
            'L%d:%.3f' % (k, v)
            for k, v in
            sorted(l17_down.items()))))
else:
    c_cond = 'smoke_subset'
    l17_down = {}
    log('B2-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C: co36 necessity ablation +
# co50 overlap decomposition
# [3135 prereg 2]
# NOTE (3134 lesson): 3134 C1 compared A1
# injection trials against the P session
# baseline (prompt mismatch) -> its A1
# chg=1.0 is inflated; the P-side gate
# was unaffected. ALL A1-direction trials
# here use a dedicated same-session A1
# baseline.
# ================================================================
log('== PART C: co36 ablation ==')
rows_c1 = {'P': [pids_P_all[j]
                 for j in range(SCAN_N)],
           'A1': [pids_A1_all[j]
                  for j in range(SCAN_N)]}
_C0K = CK['data'].get('c0_states')
if _C0K is not None:
    co36 = _C0K['co36']
    co36_rank = _C0K['rank']
    delta36 = _C0K['delta']
    j36 = _C0K['jaccard34']
    log('C0 states RESUMED '
        '(jaccard34=%.2f)' % j36)
else:
    _hA = capture_states(rows_c1['A1'],
                         [FORK_LAYER])
    _hP = capture_states(rows_c1['P'],
                         [FORK_LAYER])
    hA1_36 = _hA[FORK_LAYER]
    hP_36 = _hP[FORK_LAYER]
    d36 = (hA1_36 - hP_36).astype(
        np.float64)
    med_abs = np.median(np.abs(d36),
                        axis=0)
    # 3134 convention (sorted ascending
    # tail of argsort) for the sha anchor;
    # med-descending rank for subsets.
    co36 = np.argsort(med_abs)[
        -TOPK_COORD:].astype(np.int64)
    co36_rank = np.argsort(med_abs)[::-1][
        :TOPK_COORD].astype(np.int64)
    row_l2 = np.linalg.norm(d36, axis=1)
    delta36 = 0.10 * float(row_l2.mean())
    j36 = float(len(set(co36.tolist())
                    & set(co36_34.tolist()))
                / len(set(co36.tolist())
                      | set(
                          co36_34.tolist())))
    log('C0 d36 med||d||=%.4f delta=%.4f '
        'co36 jaccard vs 3134 = %.3f'
        % (float(np.median(row_l2)),
           delta36, j36))
    del _hA, _hP, hA1_36, hP_36
    gc.collect()
    ck_save('c0_states', {
        'co36': co36, 'rank': co36_rank,
        'delta': delta36,
        'jaccard34': j36})
if not SMOKE:
    assert j36 >= 0.9, j36
else:
    log('C0 j36 smoke record %.3f '
        '(full assert gated)' % j36)

# same-session A1 generation baseline
_CAK = CK['data'].get('c_base_A1')
if _CAK is not None:
    gen_base_sess_A1 = _CAK['gens']
    log('C base A1 RESUMED')
else:
    gen_base_sess_A1 = []
    for b0 in range(0, SCAN_N,
                    GEN_BATCH):
        gen_base_sess_A1.extend(
            gen_batch_g2(
                rows_c1['A1'][
                    b0:b0 + GEN_BATCH]))
    ck_save('c_base_A1', {
        'gens': gen_base_sess_A1})
base12_A1 = [pad12(gen_base_sess_A1[j])
             for j in range(SCAN_N)]

rngC = np.random.default_rng(RNG_SEED)
top25 = np.sort(co36_rank[:SUBSET_K])
bot25 = np.sort(
    co36_rank[SUBSET_K:2 * SUBSET_K])
rand25s = [np.sort(
    rngC.choice(co36_rank, SUBSET_K,
                replace=False))
    .astype(np.int64)
    for _ in range(N_RANDOM)]

C1_SUBS = [('full50', co36),
           ('top25', top25),
           ('bot25', bot25)]
for _ri in range(N_RANDOM):
    C1_SUBS.append(
        ('rand25_%d' % _ri,
         rand25s[_ri]))

c1_trials = {}
_C1K = CK['data'].get('c1_trials')
if _C1K is not None:
    c1_trials = _C1K['res']
    log('C1 trials RESUMED')
else:
    for (sname, coords) in C1_SUBS:
        gen_l = []
        for b0 in range(0, SCAN_N,
                        GEN_BATCH):
            batch = rows_c1['A1'][
                b0:b0 + GEN_BATCH]
            gen_l.extend(gen_batch_g2(
                batch,
                inj=(FORK_LAYER, coords,
                     delta36, 1, 0)))
        chg_l, first_l, fstep, _cum = \
            trial_metrics(gen_l, base12_A1)
        c1_trials[sname] = {
            'chg': chg_l,
            'first': int(first_l)}
        log('C1 %s: chg=%.4f first=%d'
            % (sname, chg_l, first_l))
    ck_save('c1_trials',
            {'res': c1_trials})

if not SMOKE:
    f_full = c1_trials['full50']['chg']
    f_top = c1_trials['top25']['chg']
    f_weak = max(c1_trials['bot25']['chg'],
                 max(c1_trials[
                     'rand25_%d' % _ri]
                     ['chg']
                     for _ri in range(
                         N_RANDOM)))
    if f_top >= CONC_FRAC_HI * f_full \
            and f_weak \
            <= CONC_FRAC_LO * f_full:
        c_abl = 'co36_concentrated'
    elif f_top < CONC_FRAC_LO * f_full \
            or f_weak \
            >= CONC_FRAC_HI * f_full:
        c_abl = 'co36_dispersed'
    else:
        c_abl = 'co36_partial_split'
    log('C1-GATE: full=%.4f top25=%.4f '
        'weak_max=%.4f -> %s'
        % (f_full, f_top, f_weak, c_abl))
else:
    c_abl = 'smoke_subset'
    log('C1-GATE: SKIPPED in SMOKE')

# ---- C2: co50 overlap decomposition ----
ov50 = np.intersect1d(co50, co36)
only50 = np.setdiff1d(co50, co36)
n_ov = int(len(ov50))
jac5036 = float(n_ov
                / len(set(co50.tolist())
                      | set(
                          co36.tolist())))
log('C2 overlap: |co50&co36|=%d '
    'jaccard=%.3f' % (n_ov, jac5036))
C2_SUBS = [('co50_full', co50),
           ('co50_no36', only50),
           ('co50_cap', ov50)]
c2_trials = {}
_C2K = CK['data'].get('c2_trials')
if _C2K is not None:
    c2_trials = _C2K['res']
    log('C2 trials RESUMED')
else:
    for (sname, coords) in C2_SUBS:
        gen_l = []
        for b0 in range(0, SCAN_N,
                        GEN_BATCH):
            batch = rows_c1['A1'][
                b0:b0 + GEN_BATCH]
            gen_l.extend(gen_batch_g2(
                batch,
                inj=(17, coords, DELTA_L17,
                     -1, 0)))
        chg_l, first_l, fstep, _cum = \
            trial_metrics(gen_l, base12_A1)
        c2_trials[sname] = {
            'chg': chg_l,
            'first': int(first_l)}
        log('C2 %s: chg=%.4f first=%d'
            % (sname, chg_l, first_l))
    ck_save('c2_trials',
            {'res': c2_trials})

if not SMOKE:
    g_full = c2_trials['co50_full']['chg']
    g_no = c2_trials['co50_no36']['chg']
    g_cap = c2_trials['co50_cap']['chg']
    if g_no <= NEC_FRAC * g_full:
        c_ovr = 'overlap_necessary'
    elif g_cap >= SUFF_FRAC * g_full:
        c_ovr = 'overlap_sufficient'
    else:
        c_ovr = 'overlap_mixed'
    log('C2-GATE: full=%.4f no36=%.4f '
        'cap=%.4f -> %s'
        % (g_full, g_no, g_cap, c_ovr))
else:
    c_ovr = 'smoke_subset'
    log('C2-GATE: SKIPPED in SMOKE')

# ================================================================
# PART D: multi-step injection windows
# [3135 prereg 3] L17 dvec joint injection
# over forward-index windows (0 = prompt,
# k>=1 = decode step k-1), full per-step
# dose 1.0 + dose-shared scale; same-
# session allstep ceiling reference.
# ================================================================
log('== PART D: windows ==')
rows_c2 = [pids_P_all[j]
           for j in range(SCAN_N)]
base_D = base12_P[:SCAN_N]
dvec17_scan = dvec[17][:SCAN_N]
fstep_store = {}
D_TRIALS = []
for _wn, _ws in zip(WIN_NAMES,
                    WINDOWS):
    D_TRIALS.append(
        (_wn, [(17, dvec17_scan, 1.0,
                list(_ws))]))
D_TRIALS.append(
    ('w2share',
     [(17, dvec17_scan, W_SHARE,
       list(WINDOWS[0]))]))
D_TRIALS.append(
    ('allstep',
     [(17, dvec17_scan, 1.0, 'allstep')]))

d_res = {}
_DK = CK['data'].get('d_trials')
if _DK is not None:
    d_res = _DK['res']
    log('D trials RESUMED')
else:
    for (tname, ivecs) in D_TRIALS:
        gen_l = []
        for b0 in range(0, SCAN_N,
                        GEN_BATCH):
            batch = rows_c2[
                b0:b0 + GEN_BATCH]
            iv = [(l, dv[b0:b0
                         + len(batch)],
                   s, m)
                  for (l, dv, s, m)
                  in ivecs]
            gen_l.extend(gen_batch_g2(
                batch, inj_vec=iv))
        chg_l, first_l, fstep, cum = \
            trial_metrics(gen_l, base_D)
        d_res[tname] = {
            'chg': chg_l,
            'first': int(first_l),
            'cum': [float(v)
                    for v in cum]}
        fstep_store['fstep_%s' % tname] \
            = fstep
        log('D %s: chg=%.4f first=%d'
            % (tname, chg_l, first_l))
    ck_save('d_trials', {'res': d_res})

if not SMOKE:
    chg_w = {wn: d_res[wn]['chg']
             for wn in WIN_NAMES}
    chg_share = d_res['w2share']['chg']
    chg_all = d_res['allstep']['chg']
    mono_w = all(
        chg_w[WIN_NAMES[i + 1]]
        >= chg_w[WIN_NAMES[i]]
        - 0.02
        for i in range(len(WIN_NAMES)
                       - 1))
    if mono_w and (chg_w['w8']
                   >= chg_w['w2']
                   + CUM_GAP):
        c_win = 'window_cumulative'
    elif chg_w['w8'] >= chg_w['w2'] \
            + CUM_GAP:
        c_win = 'window_jumpy'
    else:
        c_win = 'window_flat'
    c_near = ('window_near_allstep'
              if chg_w['w8']
              >= ALLSTEP_FRAC * chg_all
              else 'window_below_allstep')
    if chg_share <= chg_w['w2'] \
            - SHARE_GAP:
        c_share = 'dose_share_penalizes'
    elif chg_share >= chg_w['w2'] \
            + SHARE_GAP:
        c_share = 'dose_share_better'
    else:
        c_share = 'dose_share_neutral'
    log('D-GATE: w2=%.4f w3=%.4f w5=%.4f '
        'w8=%.4f share=%.4f all=%.4f -> '
        '%s|%s|%s'
        % (chg_w['w2'], chg_w['w3'],
           chg_w['w5'], chg_w['w8'],
           chg_share, chg_all, c_win,
           c_near, c_share))
else:
    c_win = c_near = c_share \
        = 'smoke_subset'
    log('D-GATE: SKIPPED in SMOKE')

# ================================================================
# dumps
# ================================================================
if not SMOKE:
    assert set(d_res.keys()) == \
        set(dict(D_TRIALS).keys())
    assert len(c1_trials) == len(C1_SUBS)
    assert len(c2_trials) == 3
    log('CKPT integrity assert ok')
runtime = time.time() - T0
verdict = '|'.join([
    'a_3134_ok', c_cond, c_abl, c_ovr,
    c_win, c_near, c_share,
    'coverage_full'])
log('VERDICT: %s' % verdict)


def _direct_med(il):
    rs = [r for r in (il + 1, il + 2,
                      il + 3) if r < 40]
    return (
        float(np.median(
            [float(np.nanmedian(
                cos_spec[il][r]))
             for r in rs])),
        float(np.median(
            [float(np.nanmedian(
                rho_spec[il][r]))
             for r in rs])))


cond_summary = {}
for il in CAP_L:
    cd, rd = _direct_med(il)
    cond_summary[str(il)] = {
        'cos_direct': cd,
        'rho_direct': rd,
        'gate': (gmap[il]
                 if (il in gmap
                     and not SMOKE)
                 else 'na')}
result = {
    'name': NAME,
    'phase': 3135,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'res34_sha8': sha34,
        'asserts_3134': 'ok',
        'co50_sha8': co50_sha,
        'co36_34_sha8': c36_sha},
    'part_b': {
        'dvec_med_norm': {
            str(l): norms[l]
            for l in CAP_L},
        'xphase_base_match_P': xphase_P,
        'dose_cond': DOSE_COND,
        'conduction': cond_summary,
        'cos_spec_med': {
            str(il): [None if np.isnan(v)
                      else float(v)
                      for v in np.nanmedian(
                          cos_spec[il],
                          axis=1)]
            for il in CAP_L},
        'rho_spec_med': {
            str(il): [None if np.isnan(v)
                      else float(v)
                      for v in np.nanmedian(
                          rho_spec[il],
                          axis=1)]
            for il in CAP_L},
        'l17_downstream_cos': {
            str(k): float(v) for k, v
            in l17_down.items()},
        'b_cond': c_cond},
    'part_c': {
        'fork_layer': FORK_LAYER,
        'fork_prof_idx': FORK_PROF_IDX,
        'co36_sha8': hashlib.sha256(
            co36.tobytes()).hexdigest()[:8],
        'jaccard34': j36,
        'delta36': delta36,
        'c1_trials': c1_trials,
        'c_abl': c_abl,
        'overlap': {
            'n_ov': n_ov,
            'jaccard_50_36': jac5036},
        'c2_trials': c2_trials,
        'c_ovr': c_ovr},
    'part_d': {
        'windows': WIN_NAMES,
        'w_share': W_SHARE,
        'd_trials': d_res,
        'd_win': c_win,
        'd_near': c_near,
        'd_share': c_share},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'co36': co36,
    'co36_rank': co36_rank,
    'top25': top25,
    'bot25': bot25,
    'ov50': ov50,
    'only50': only50,
    'dvec17': dvec[17][:NP_B],
    'dvec29': dvec[29][:NP_B],
    'dvec33': dvec[33][:NP_B],
    'dvec38': dvec[38][:NP_B],
    'dvec_med_norm': np.array(
        [norms[l] for l in CAP_L])}
for il in CAP_L:
    npz_out['cos_%d' % il] = cos_spec[il]
    npz_out['rho_%d' % il] = rho_spec[il]
    npz_out['dvec_full_%d' % il] = \
        dvec_full[il].astype(np.float16)
for k, v in fstep_store.items():
    npz_out[k] = v
np.savez(os.path.join(OUT,
                      'p133_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p133_readout.npz (%d keys)'
    % len(npz_out))
# all data persisted -> drop ckpt
# (P3135_CKPT_KEEP=1 keeps it for a
# full-RESUME smoke validation)
if os.path.exists(CKPTF) \
        and not os.environ.get(
            'P3135_CKPT_KEEP'):
    os.remove(CKPTF)
    log('CKPT removed (data fully '
        'persisted)')
log('P3135 DONE (%.1fs)'
    % (time.time() - T0))
