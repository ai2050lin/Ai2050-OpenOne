# -*- coding: utf-8 -*-
"""Phase 3136 (Omega-P134): conduction
fidelity dose-response + co36/co50 cross
injection matrix + w8 drop-one spectrum.

Preregistered in 3135 closeout (task 79):
(1) dose-response of the decoupled
conduction: inject CAP_L dvecs (29/33/38,
frozen 3135 fp32) at their own layer at
doses {0.5,1,2,4}xDOSE_COND, read
downstream cos/rho curves vs the swap4
field + behavior chg (allstep gen);
(2) co36 u co50 cross injection matrix
(layers {17,35,both} x sets {co50,co36,
union}) on A1: diagonal dominance +
union additivity (within-layer and
cross-layer);
(3) w8 stepwise drop-one ablation:
remove one forward index k from the
window [0..8], per-k contribution
spectrum + top1 dominance.

Reuses verified 3133-3135 infrastructure
(materials factory, gen_batch_g2 with
list-mode windows, single-sample capture,
trial_metrics, ckpt/RESUME).  dvecs are
loaded frozen from the 3135 readout npz
(sha-anchored) so no swap4 re-capture is
needed; only the session-internal base
capture is redone (xphase probe vs z26
profile idx 18)."""
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
NAME = ('omega_p134_conddose_'
        'crossmatrix_w8drop')
SMOKE = os.environ.get('P3136_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D32 = RDIR + r'\phase3132' \
      r'\omega_p130_forkcausal_single256'
D35 = RDIR + r'\phase3135' \
      r'\omega_p133_conduction_' \
      r'co36ablation_window'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3136', NAME)
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
import pickle  # noqa: E402
CKPTF = os.path.join(OUT, 'p134_ckpt.pkl')
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
GEN_BATCH = 4 if SMOKE else 32
DIRS = ('P', 'A1')

# dvec carrier layers (frozen 3135 fp32)
CAP_L_D = [29, 33, 38]
# swap4 layers for the reference field
# (3134/3135 frozen convention)
SWAP_L = [8, 9, 13, 29]
L17 = 17
L35 = 35
DOSE_COND = 2.0
DOSES = [0.5, 1.0, 2.0, 4.0]
ALL_L = list(range(40))
W8_IDX = list(range(9))
# conduction gates
FID_GAP = 0.06
RHO_SLOPE_LO = 0.75
RHO_SLOPE_HI = 1.25
CHG_MONO_TOL = 0.02
# cross-matrix gates
DIAG_GAP = 0.10
ADD_TOL = 0.15
# drop-one gates
TOP1_MIN = 0.15
FLAT_MAX = 0.05
# soft repro tolerance (record only)
W8_REPRO_TOL = 0.08

TF_SEED = 3136
RNG_SEED = 3136

# ---------------- frozen 3135 anchors --
EXP_V35 = ('a_3134_ok|cond_readout_'
           'decoupled|co36_dispersed|'
           'overlap_mixed|window_cumulative|'
           'window_near_allstep|'
           'dose_share_penalizes|'
           'coverage_full')
RES35_SHA = '0f1fdf95'
CO36_SHA = '84a1e1a3'
CO36_RANK_SHA = '0f4c25b1'
CO50_SHA = '52b126af'
UNION_SHA = '1e7edb1a'
DV17_SHA = '5e4c3085'
DV29_SHA = 'ee9484b2'
DV33_SHA = '59fbe0d3'
DV38_SHA = 'aced803b'
DELTA36_34 = 1.694018277446071
DELTA_L17 = 0.637683315669971
DVEC_MED_35 = [4.253485202789307,
               31.643726348876953,
               44.65449905395508,
               100.42410278320312]
# soft anchors (record-only, no crash)
W8_35 = 83 / 128.0
ALL_35 = 0.6875
M17_50_35 = 0.421875
M35_36_35 = 0.078125
COSDIR_35 = {'29': 0.9558479189872742,
             '33': 0.9602330923080444,
             '38': 0.9704782962799072}

# ---------------- seal ------------------
SEAL = {
    'phase': 3136,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'CAP_L_D': CAP_L_D, 'SWAP_L': SWAP_L,
        'L17': L17,
        'L35': L35, 'DOSE_COND': DOSE_COND,
        'DOSES': DOSES, 'ALL_L': ALL_L,
        'W8_IDX': W8_IDX,
        'FID_GAP': FID_GAP,
        'RHO_SLOPE_LO': RHO_SLOPE_LO,
        'RHO_SLOPE_HI': RHO_SLOPE_HI,
        'CHG_MONO_TOL': CHG_MONO_TOL,
        'DIAG_GAP': DIAG_GAP,
        'ADD_TOL': ADD_TOL,
        'TOP1_MIN': TOP1_MIN,
        'FLAT_MAX': FLAT_MAX,
        'W8_REPRO_TOL': W8_REPRO_TOL,
        'TF_SEED': TF_SEED,
        'RNG_SEED': RNG_SEED},
    'anchors': {
        'res35_verdict': EXP_V35,
        'res35_sha8': RES35_SHA,
        'co36_sha8': CO36_SHA,
        'co36_rank_sha8': CO36_RANK_SHA,
        'co50_sha8': CO50_SHA,
        'union_sha8': UNION_SHA,
        'dv17_sha8': DV17_SHA,
        'dv29_sha8': DV29_SHA,
        'dv33_sha8': DV33_SHA,
        'dv38_sha8': DV38_SHA,
        'delta36_34': DELTA36_34,
        'delta_l17': DELTA_L17,
        'dvec_med_35': DVEC_MED_35,
        'w8_35_soft': W8_35,
        'allstep_35_soft': ALL_35,
        'l17co50_35_soft': M17_50_35,
        'l35co36_35_soft': M35_36_35,
        'cosdir_35_soft': COSDIR_35},
    'prereg': ('3135 closeout task 79: '
               '(1) conduction fidelity '
               'dose-response: CAP_L dvecs '
               '(29/33/38, frozen 3135 fp32) '
               'self-layer injection at '
               'doses {0.5,1,2,4}xDOSE_COND, '
               'downstream cos/rho curves '
               'vs swap4 field + allstep '
               'behavior chg. (2) co36/co50 '
               'cross injection matrix: '
               'layers {17,35,both} x sets '
               '{co50,co36,union} on A1, '
               'diagonal dominance + union '
               'additivity (within-layer + '
               'cross-layer). (3) w8 drop-one '
               'ablation: per-k contribution '
               'spectrum + top1 dominance. '
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
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
assert z26['mlg_s0_P'].shape == (672, 41, 13)
assert z26['gen_base_P'].shape == (672, 12)
assert z26['gen_base_A1'].shape == (672, 12)
z32 = np.load(D32 + r'\p130_readout.npz',
              allow_pickle=False)
assert z32['co50'].shape == (50,)
z35 = np.load(D35 + r'\p133_readout.npz',
              allow_pickle=False)
for k, sha in (('dvec17', DV17_SHA),
               ('dvec29', DV29_SHA),
               ('dvec33', DV33_SHA),
               ('dvec38', DV38_SHA),
               ('co36', CO36_SHA),
               ('co36_rank', CO36_RANK_SHA)):
    a = z35[k]
    got = hashlib.sha256(
        a.tobytes()).hexdigest()[:8]
    assert got == sha, (k, got, sha)
co50 = z32['co50'].astype(np.int64)
co50_sha = hashlib.sha256(
    co50.tobytes()).hexdigest()[:8]
assert co50_sha == CO50_SHA, co50_sha
union3650 = np.union1d(
    z35['co36'].astype(np.int64), co50)
u_sha = hashlib.sha256(
    union3650.tobytes()).hexdigest()[:8]
assert u_sha == UNION_SHA, u_sha
assert len(union3650) == 100
res35 = json.load(io.open(
    D35 + r'\result.json', encoding='utf-8'))
assert res35['smoke'] is False
log('frozen inputs ok (26/32/35 npz, '
    'dvec sha anchored)')

# ================================================================
# PART A: offline 3135 link asserts
# ================================================================
log('== PART A: 3135 link asserts ==')
V35 = res35['verdict']
assert V35 == EXP_V35, V35
raw35 = io.open(
    D35 + r'\result.json', 'rb').read()
sha35 = hashlib.sha256(
    raw35).hexdigest()[:8]
assert sha35 == RES35_SHA, sha35
pb35 = res35['part_b']
pc35 = res35['part_c']
pd35 = res35['part_d']
assert abs(pc35['delta36']
           - DELTA36_34) < 1e-9
_med35 = [float(v) for v in
          z35['dvec_med_norm']]
assert all(abs(a - b) < 1e-6 for a, b
           in zip(_med35, DVEC_MED_35)), _med35
assert abs(pd35['d_trials']['w8']['chg']
           - W8_35) < 1e-9
assert abs(pd35['d_trials']['allstep']
           ['chg'] - ALL_35) < 1e-9
assert abs(pc35['c2_trials']['co50_full']
           ['chg'] - M17_50_35) < 1e-9
assert abs(pc35['c1_trials']['full50']
           ['chg'] - M35_36_35) < 1e-9
log('A hard asserts ok (3135 frozen, '
    'result sha8 %s, verdict match)' % sha35)

# ================================================================
# materials factory (3125/.../3135
# identical)
# ================================================================
p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
capb = np.load(RDIR + r'\phase3113'
               r'\omega_p111_artifact_writein'
               r'\capture_b.npz',
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
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
      (il, coords, delta, sgn, mode), OR a
      LIST of such tuples for simultaneous
      multi-layer injection (3136: cross
      matrix both-layer trials).
    inj_vec: per-layer transplants
      (il, dvec_batch (n,HID) fp32, scale,
       mode). mode: 'allstep' = every
       forward; int s = prompt forward +
       decode step s (s==0 -> step0);
       list = explicit forward-index
       window (3135 semantics)."""
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
        inj_list = inj if isinstance(
            inj, (list, tuple)) \
            and inj and isinstance(
                inj[0], (list, tuple)) \
            else [inj]
        for (il, coords, delta, sgn,
             mode) in inj_list:
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

            def _inj(mod, inp, out,
                     _co=co_t, _dv=dv_t,
                     _m=mode, _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if _m == 'allstep' \
                        or _st['step'] == 0 \
                        or _st['step'] == _m:
                    o2[:, -1, _co] += _dv
                _st['step'] += 1
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
    padding) replicating z26/3131
    semantics exactly (3133 rev-f)."""
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
    prompt-only (no generation)."""
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
# PART B1: frozen dvec load + session
# base capture (full 40 layers)
# [3136] dvecs come from the 3135 npz
# (sha-anchored fp32) -> only the
# session-internal base capture is redone.
# dvec_full (swap4 field, fp16 in z35) is
# the reference for cos/rho spectra.
# ================================================================
log('== PART B1: frozen dvecs + base ==')
dvec = {int(l): z35['dvec%d' % l]
        .astype(np.float32)
        for l in ([L17] + CAP_L_D)}
norms = {l: float(np.median(np.linalg.norm(
    dvec[l], axis=1)))
    for l in ([L17] + CAP_L_D)}
for l in ([L17] + CAP_L_D):
    log('B1 frozen dvec L%02d '
        'med||d||=%.4f' % (l, norms[l]))
rows_cap = [pids_P_all[j]
            for j in range(NP_CAP)]
CKM = {'smoke': SMOKE, 'np': NP,
       'np_cap': NP_CAP, 'np_b': NP_B,
       'scan_n': SCAN_N}
if CK['meta'] and CK['meta'] != CKM:
    log('CKPT meta mismatch -> discard')
    CK = {'done': [], 'data': {},
          'meta': {}}
CK['meta'] = CKM
_B1K = CK['data'].get('b1_base')
if _B1K is not None:
    base_states = {
        int(k): v.astype(np.float32)
        for k, v in
        _B1K['base_fp16'].items()}
    log('B1 base RESUMED from ckpt')
else:
    base_states = capture_states(
        rows_cap, ALL_L, swap_layers=None)
    log('B1 base capture done (%d rows, '
        '%d layers)' % (len(rows_cap),
                        len(ALL_L)))
    # xphase probe vs z26 profile idx 18
    # (capture at layers[17] output =
    # profile idx 18; 3133 rev-g)
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
    ck_save('b1_base', {
        'base_fp16': {
            str(l): base_states[l]
            .astype(np.float16)
            for l in ALL_L}})
_S1K = CK['data'].get('b1_swap')
if _S1K is not None:
    swap_states = {
        int(k): v.astype(np.float32)
        for k, v in
        _S1K['swap_fp16'].items()}
    log('B1 swap RESUMED from ckpt')
else:
    swap_states = capture_states(
        rows_cap, ALL_L,
        swap_layers=SWAP_L)
    log('B1 swap4 capture done')
    ck_save('b1_swap', {
        'swap_fp16': {
            str(l): swap_states[l]
            .astype(np.float16)
            for l in ALL_L}})
dvec_full = {}
for l in ALL_L:
    _d = (swap_states[l]
          - base_states[l])
    dvec_full[l] = _d.astype(
        np.float32)
    if l in CAP_L_D:
        log('B1 local dvec L%02d '
            'med||d||=%.4f (3135 '
            'anchor %.4f)'
            % (l, float(np.median(
                np.linalg.norm(
                    dvec_full[l],
                    axis=1))),
               DVEC_MED_35[
                   CAP_L_D.index(l)
                   + 1]))
del swap_states
gc.collect()

# --- session-internal generation base ---
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
# PART B2: conduction dose-response
# [3136 prereg 1] for il in {29,33,38} and
# dose in {0.5,1,2,4}xDOSE_COND: inject
# frozen dvec[il] at its own layer,
# teacher-forced, capture all 40 states;
# cos/rho vs the frozen swap4 field.
# Behavior chg: allstep dvec injection on
# SCAN_N P rows vs same-session base.
# ================================================================
log('== PART B2: dose-response ==')
cos_dr = {}
rho_dr = {}
for il in CAP_L_D:
    cos_dr[il] = {}
    rho_dr[il] = {}
    for dose in DOSES:
        _K = CK['data'].get(
            'dosecap_%d_%s' % (il, dose))
        if _K is not None:
            cos_dr[il][dose] = _K['cos']
            rho_dr[il][dose] = _K['rho']
            log('B2 dose L%02d d%.1f '
                'RESUMED' % (il, dose))
            continue
        n_cap = min(NP_CAP,
                    dvec[il].shape[0])
        h_inj = capture_states_inject(
            rows_cap[:n_cap], ALL_L, il,
            dvec[il][:n_cap],
            dose * DOSE_COND)
        cos_m = np.full((len(ALL_L),
                         n_cap),
                        np.nan,
                        dtype=np.float32)
        rho_m = np.full((len(ALL_L),
                         n_cap),
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
            ndh = np.linalg.norm(
                dh, axis=1)
            ndv = np.linalg.norm(
                dv, axis=1)
            num = np.sum(dh * dv, axis=1)
            den = ndh * ndv
            with np.errstate(
                    invalid='ignore',
                    divide='ignore'):
                cos_m[r] = np.where(
                    den > 1e-12,
                    num / den, 0.0)
                rho_m[r] = np.where(
                    ndv > 1e-12,
                    ndh / ndv, 0.0)
        del h_inj
        gc.collect()
        torch.cuda.empty_cache()
        cos_dr[il][dose] = cos_m
        rho_dr[il][dose] = rho_m
        ck_save('dosecap_%d_%s'
                % (il, dose),
                {'cos': cos_m,
                 'rho': rho_m})


def _direct_med(il, dose):
    rs = [r for r in (il + 1, il + 2,
                      il + 3) if r < 40]
    return (
        float(np.median(
            [float(np.nanmedian(
                cos_dr[il][dose][r]))
             for r in rs])),
        float(np.median(
            [float(np.nanmedian(
                rho_dr[il][dose][r]))
             for r in rs])))


cosdir_curve = {str(il): {
    str(d): _direct_med(il, d)
    for d in DOSES} for il in CAP_L_D}
for il in CAP_L_D:
    _line = ' '.join(
        'd%.1f:(%.3f,%.3f)'
        % (d, cosdir_curve[str(il)]
           [str(d)][0],
           cosdir_curve[str(il)]
           [str(d)][1])
        for d in DOSES)
    log('B2 curve L%02d %s' % (il, _line))

# ---- behavior chg dose response ----
rows_dose = [pids_P_all[j]
             for j in range(SCAN_N)]
base_D = base12_P[:SCAN_N]
dchg = {}
_DK = CK['data'].get('dose_chg')
if _DK is not None:
    dchg = _DK['res']
    log('dose chg RESUMED')
else:
    for il in CAP_L_D:
        for dose in DOSES:
            gen_l = []
            for b0 in range(0, SCAN_N,
                            GEN_BATCH):
                batch = rows_dose[
                    b0:b0 + GEN_BATCH]
                gen_l.extend(gen_batch_g2(
                    batch,
                    inj_vec=[(il, dvec[il][
                        b0:b0
                        + len(batch)],
                        dose * DOSE_COND,
                        'allstep')]))
            chg_l, first_l, fstep, _cum = \
                trial_metrics(gen_l, base_D)
            dchg['%d_%.1f' % (il, dose)] = {
                'chg': chg_l,
                'first': int(first_l)}
            log('B2 chg L%02d d%.1f: '
                'chg=%.4f first=%d'
                % (il, dose, chg_l,
                   first_l))
    ck_save('dose_chg', {'res': dchg})

if not SMOKE:
    gmap_dose = {}
    for il in CAP_L_D:
        cs = [cosdir_curve[str(il)]
              [str(d)][0] for d in DOSES]
        rs = [cosdir_curve[str(il)]
              [str(d)][1] for d in DOSES]
        gap = max(cs) - min(cs)
        # log-log slope of rho vs dose
        lx = np.log(np.array(DOSES))
        ly = np.log(np.array(rs))
        slope = float(np.polyfit(
            lx, ly, 1)[0])
        chgs = [dchg['%d_%.1f'
                     % (il, d)]['chg']
                for d in DOSES]
        mono = all(
            chgs[i + 1] >= chgs[i]
            - CHG_MONO_TOL
            for i in range(
                len(DOSES) - 1))
        gmap_dose[il] = {
            'fid_gap': float(gap),
            'rho_slope': slope,
            'mono': bool(mono),
            'chg': chgs}
        log('B2-GATE L%02d: fid_gap=%.4f '
            'rho_slope=%.3f mono=%s '
            'chg=%s'
            % (il, gap, slope, mono,
               ['%.4f' % c
                for c in chgs]))
    _gaps = [gmap_dose[il]['fid_gap']
             for il in CAP_L_D]
    _slopes = [gmap_dose[il]['rho_slope']
               for il in CAP_L_D]
    _monos = [gmap_dose[il]['mono']
              for il in CAP_L_D]
    if max(_gaps) <= FID_GAP:
        c_fid = 'fidelity_flat'
    else:
        c_fid = 'fidelity_dose_sensitive'
    med_slope = float(np.median(_slopes))
    if med_slope < RHO_SLOPE_LO:
        c_rho = 'rho_sublinear'
    elif med_slope > RHO_SLOPE_HI:
        c_rho = 'rho_superlinear'
    else:
        c_rho = 'rho_linear'
    c_dmono = ('dose_resp_monotone'
               if all(_monos)
               else 'dose_resp_nonmono')
    log('B2-GATE: %s|%s|%s (med slope '
        '%.3f, max gap %.4f)'
        % (c_fid, c_rho, c_dmono,
           med_slope, max(_gaps)))
else:
    gmap_dose = {}
    c_fid = c_rho = c_dmono \
        = 'smoke_subset'
    log('B2-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C: co36/co50 cross injection
# matrix (A1 direction, SCAN_N rows)
# [3136 prereg 2] layers {17,35,both} x
# sets {co50,co36,union}.  17 uses
# (DELTA_L17, sgn=-1); 35 uses
# (DELTA36_34, sgn=+1) -- exactly the
# 3135 C2/C1 parameters.  Diagonal
# dominance + union additivity.
# ================================================================
log('== PART C: cross matrix ==')
rows_c = [pids_A1_all[j]
          for j in range(SCAN_N)]
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
                rows_c[b0:b0 + GEN_BATCH]))
    ck_save('c_base_A1', {
        'gens': gen_base_sess_A1})
base12_A1 = [pad12(gen_base_sess_A1[j])
             for j in range(SCAN_N)]

co36 = z35['co36'].astype(np.int64)
SETS = {'co50': co50,
        'co36': co36,
        'union': union3650}
C_TRIALS = []
for sname, coords in SETS.items():
    C_TRIALS.append(
        ('L17_%s' % sname,
         [(L17, coords, DELTA_L17, -1, 0)]))
for sname, coords in SETS.items():
    C_TRIALS.append(
        ('L35_%s' % sname,
         [(L35, coords, DELTA36_34, 1, 0)]))
for sname, coords in SETS.items():
    C_TRIALS.append(
        ('both_%s' % sname,
         [(L17, coords, DELTA_L17, -1, 0),
          (L35, coords, DELTA36_34, 1, 0)]))

cmat = {}
_CK2 = CK['data'].get('c_matrix')
if _CK2 is not None:
    cmat = _CK2['res']
    log('C matrix RESUMED')
else:
    for (tname, inj_list) in C_TRIALS:
        gen_l = []
        for b0 in range(0, SCAN_N,
                        GEN_BATCH):
            batch = rows_c[
                b0:b0 + GEN_BATCH]
            gen_l.extend(gen_batch_g2(
                batch, inj=inj_list))
        chg_l, first_l, fstep, _cum = \
            trial_metrics(gen_l, base12_A1)
        cmat[tname] = {
            'chg': chg_l,
            'first': int(first_l)}
        log('C %s: chg=%.4f first=%d'
            % (tname, chg_l, first_l))
    ck_save('c_matrix', {'res': cmat})

if not SMOKE:
    diag_min = min(cmat['L17_co50']['chg'],
                   cmat['L35_co36']['chg'])
    off_max = max(cmat['L17_co36']['chg'],
                  cmat['L35_co50']['chg'])
    if diag_min >= off_max + DIAG_GAP:
        c_diag = 'diagonal_dominant'
    else:
        c_diag = 'diagonal_weak'
    # union additivity
    a17 = abs(cmat['L17_union']['chg']
              - (cmat['L17_co50']['chg']
                 + cmat['L17_co36']['chg']))
    a35 = abs(cmat['L35_union']['chg']
              - (cmat['L35_co50']['chg']
                 + cmat['L35_co36']['chg']))
    aboth = abs(cmat['both_union']['chg']
                - (cmat['L17_union']['chg']
                   + cmat['L35_union']
                   ['chg']))
    add_flags = [a17 <= ADD_TOL,
                 a35 <= ADD_TOL,
                 aboth <= ADD_TOL]
    if all(add_flags):
        c_uni = 'union_add_all'
    elif not add_flags[0] \
            and not add_flags[1] \
            and not add_flags[2]:
        c_uni = 'union_nonadd'
    elif add_flags[0] and add_flags[1]:
        c_uni = 'union_add_withinlayer'
    else:
        c_uni = 'union_mixed'
    log('C-GATE: diag_min=%.4f off_max='
        '%.4f -> %s; add resid l17=%.4f '
        'l35=%.4f both=%.4f -> %s'
        % (diag_min, off_max, c_diag,
           a17, a35, aboth, c_uni))
    # soft anchors vs 3135
    log('C-SOFT: L17_co50 %.4f vs 3135 '
        '%.4f | L35_co36 %.4f vs 3135 '
        '%.4f'
        % (cmat['L17_co50']['chg'],
           M17_50_35,
           cmat['L35_co36']['chg'],
           M35_36_35))
else:
    c_diag = c_uni = 'smoke_subset'
    log('C-GATE: SKIPPED in SMOKE')

# ================================================================
# PART D: w8 drop-one ablation (P
# direction, SCAN_N rows)
# [3136 prereg 3] window [0..8], remove
# one forward index k at a time; per-k
# contribution = chg(w8) - chg(w8\k).
# ================================================================
log('== PART D: w8 drop-one ==')
rows_d2 = [pids_P_all[j]
           for j in range(SCAN_N)]
dvec17_scan = dvec[17][:SCAN_N]
D_TRIALS = [('w8full',
             [(L17, dvec17_scan, 1.0,
               list(W8_IDX))])]
for k in W8_IDX:
    D_TRIALS.append(
        ('drop%d' % k,
         [(L17, dvec17_scan, 1.0,
           [i for i in W8_IDX if i != k])]))

d_res = {}
fstep_store = {}
_DK2 = CK['data'].get('d_drop1')
if _DK2 is not None:
    d_res = _DK2['res']
    log('D trials RESUMED')
else:
    for (tname, ivecs) in D_TRIALS:
        gen_l = []
        for b0 in range(0, SCAN_N,
                        GEN_BATCH):
            batch = rows_d2[
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
            'first': int(first_l)}
        fstep_store['fstep_%s' % tname] \
            = fstep
        log('D %s: chg=%.4f first=%d'
            % (tname, chg_l, first_l))
    ck_save('d_drop1', {'res': d_res})

if not SMOKE:
    chg_w8 = d_res['w8full']['chg']
    contrib = {}
    for k in W8_IDX:
        contrib[k] = chg_w8 \
            - d_res['drop%d' % k]['chg']
    top1_k = max(contrib,
                 key=lambda k: contrib[k])
    top1_v = contrib[top1_k]
    min_v = min(contrib.values())
    if top1_v >= TOP1_MIN:
        c_d1 = 'w8_top1_dominant'
    elif top1_v <= FLAT_MAX:
        c_d1 = 'w8_flat'
    else:
        c_d1 = 'w8_distributed'
    drift = abs(chg_w8 - W8_35)
    c_rep = ('w8_repro_ok'
             if drift <= W8_REPRO_TOL
             else 'w8_repro_drift')
    log('D-GATE: w8full=%.4f (3135 %.4f, '
        'drift %.4f -> %s); top1 k=%d '
        'contrib=%.4f; contrib range '
        '[%.4f, %.4f] -> %s'
        % (chg_w8, W8_35, drift, c_rep,
           top1_k, top1_v, min_v, top1_v,
           c_d1))
    # submodularity record: sum of
    # drop-one contributions vs chg(w8)
    sum_contrib = float(
        sum(contrib.values()))
    log('D-RECORD: sum_contrib=%.4f '
        'ratio=%.3f (sum/chg_w8)'
        % (sum_contrib,
           sum_contrib / max(chg_w8,
                             1e-9)))
else:
    c_d1 = c_rep = 'smoke_subset'
    top1_k = -1
    top1_v = min_v = 0.0
    sum_contrib = 0.0
    log('D-GATE: SKIPPED in SMOKE')

# ================================================================
# dumps
# ================================================================
if not SMOKE:
    assert set(d_res.keys()) == \
        set(dict(D_TRIALS).keys())
    assert len(cmat) == len(C_TRIALS)
    assert len(cmat) == 9
    assert len(dchg) == len(CAP_L_D) \
        * len(DOSES)
    log('CKPT integrity assert ok')
runtime = time.time() - T0
verdict = '|'.join([
    'a_3135_ok', c_fid, c_rho, c_dmono,
    c_diag, c_uni, c_d1, c_rep,
    'coverage_full'])
log('VERDICT: %s' % verdict)

result = {
    'name': NAME,
    'phase': 3136,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'res35_sha8': sha35,
        'asserts_3135': 'ok',
        'co50_sha8': co50_sha,
        'union_sha8': u_sha},
    'part_b': {
        'dvec_med_norm': {
            str(l): norms[l]
            for l in ([L17] + CAP_L_D)},
        'xphase_base_match_P': xphase_P,
        'doses': DOSES,
        'dose_cond': DOSE_COND,
        'cosdir_curve': cosdir_curve,
        'dose_gates': {
            str(il): gmap_dose[il]
            for il in CAP_L_D
            if il in gmap_dose},
        'dose_chg': dchg,
        'b_fid': c_fid,
        'b_rho': c_rho,
        'b_mono': c_dmono},
    'part_c': {
        'sets': {k: int(len(v))
                 for k, v in
                 SETS.items()},
        'delta_l17': DELTA_L17,
        'delta36': DELTA36_34,
        'matrix': cmat,
        'c_diag': c_diag,
        'c_uni': c_uni},
    'part_d': {
        'w8_idx': W8_IDX,
        'd_res': d_res,
        'drop1_contrib': {
            str(k): float(contrib[k])
            for k in W8_IDX}
            if not SMOKE else {},
        'top1_k': int(top1_k),
        'top1_v': float(top1_v),
        'sum_contrib': float(sum_contrib),
        'w8_35_soft': W8_35,
        'd_drop1': c_d1,
        'd_repro': c_rep},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'co50': co50,
    'co36': co36,
    'union': union3650,
    'dvec_med_norm': np.array(
        [norms[l]
         for l in ([L17] + CAP_L_D)])}
for il in CAP_L_D:
    for dose in DOSES:
        npz_out['cos_%d_d%s' % (
            il, str(dose).replace(
                '.', ''))] = \
            cos_dr[il][dose]
        npz_out['rho_%d_d%s' % (
            il, str(dose).replace(
                '.', ''))] = \
            rho_dr[il][dose]
for k, v in fstep_store.items():
    npz_out[k] = v
np.savez(os.path.join(OUT,
                      'p134_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p134_readout.npz (%d keys)'
    % len(npz_out))
# all data persisted -> drop ckpt
# (P3136_CKPT_KEEP=1 keeps it for a
# full-RESUME smoke validation)
if os.path.exists(CKPTF) \
        and not os.environ.get(
            'P3136_CKPT_KEEP'):
    os.remove(CKPTF)
    log('CKPT removed (data fully '
        'persisted)')
log('P3136 DONE (%.1fs)'
    % (time.time() - T0))
