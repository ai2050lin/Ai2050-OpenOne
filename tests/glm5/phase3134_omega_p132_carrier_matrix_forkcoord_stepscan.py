# -*- coding: utf-8 -*-
"""Phase 3134 (Omega-P132): dvec carrier
matrix + A1 fork-layer coordimization +
generation step-scan + trajectory profiles.

Preregistered in 3133 MEMO section 5:
(1) swap4 per-layer dvec dose-response
matrix (carrier decomposition) + pairwise
additivity + same-session J2 anchor;
(2) A1 first-divergence layer (profile
idx 36 = layers[35] output) coordimiza-
tion: A1-P state-diff top50 injected into
P/A1 generation; generation step-scan of
L17 dvec (injection steps 0..11);
(3) trajectory layer profiles (k=0..12)
P vs A1 + swap4/inject dm trajectories +
descriptive second-difference interaction.

Reuses verified 3133 infrastructure
(materials factory, gen_batch_g2,
single-sample capture, margin_track,
trial_metrics, ckpt/RESUME)."""
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
NAME = ('omega_p132_carrier_matrix_'
        'forkcoord_stepscan')
SMOKE = os.environ.get('P3134_SMOKE',
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
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3134', NAME)
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
CKPTF = os.path.join(OUT, 'p132_ckpt.pkl')
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

# carrier decomposition (3133 dvec
# convention: capture at CAP_L under
# SWAP_L bypass; injection layer indices
# are the CAP_L layer indices themselves)
SWAP_L = [8, 9, 13, 29]
CAP_L = [17, 29, 33, 38]
DOSES = (0.5, 1.0, 2.0)
PAIRS = [(17, 29), (17, 33), (29, 33)]
STEPS_SCAN = [0, 1, 2, 3, 5, 8, 11]
FORK_PROF_IDX = 36       # medE A1 (3133)
FORK_LAYER = FORK_PROF_IDX - 1  # layers[35]
TOPK_COORD = 50

TR_PART = 0.10
ADD_TOL = 0.15
DOSE_SLACK = 0.02
J2_TOL = 0.05
STEP_GATE = 0.20
EMERGE_FRAC = 0.10
FLIP_GUARD = 0.1
DM40_MIN = 0.05
FLIP_TOL = 3

TF_SEED = 3134

# ---------------- frozen 3133 anchors --
EXP_V33 = ('a_3132_ok|transplant_l17_'
           'partial|dose_monotone|l38_'
           'transplant_lower|joint_additive'
           '|step_transplant_allstep_higher'
           '|a1_forklayer_asymmetric|a1_'
           'transfer_high|a1_l17coord_'
           'rewrites|coverage_full')
RES33_SHA = '73f29f4e'
SEL256_SHA = 'e34d2588'
CO50_SHA = '52b126af'
# exact fractions (n/672) from the 3133
# full run (bit-stable across 3 sessions)
CHG17_D05 = 54 / 672.0
CHG17_D10 = 90 / 672.0
CHG17_D20 = 172 / 672.0
CHG17_AS = 0.6875
CHG38_D10 = 24 / 672.0
CHGJ1 = 92 / 672.0
CHGJ2 = 258 / 672.0
MED_E_A1 = 36
MED_E_P = -1
SP_SPEC = 0.8571428571428572
C3_BEST = 41 / 128.0
DELTA_L17 = 0.637683315669971

# ---------------- seal ------------------
SEAL = {
    'phase': 3134,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'SWAP_L': SWAP_L, 'CAP_L': CAP_L,
        'DOSES': list(DOSES),
        'PAIRS': PAIRS,
        'STEPS_SCAN': STEPS_SCAN,
        'FORK_PROF_IDX': FORK_PROF_IDX,
        'FORK_LAYER': FORK_LAYER,
        'TOPK_COORD': TOPK_COORD,
        'TF_SEED': TF_SEED,
        'TR_PART': TR_PART,
        'ADD_TOL': ADD_TOL,
        'DOSE_SLACK': DOSE_SLACK,
        'J2_TOL': J2_TOL,
        'STEP_GATE': STEP_GATE,
        'EMERGE_FRAC': EMERGE_FRAC,
        'FLIP_GUARD': FLIP_GUARD,
        'DM40_MIN': DM40_MIN,
        'FLIP_TOL': FLIP_TOL},
    'anchors': {
        'res33_verdict': EXP_V33,
        'res33_sha8': RES33_SHA,
        'sel256_sha8': SEL256_SHA,
        'co50_sha8': CO50_SHA,
        'chg17_d05': CHG17_D05,
        'chg17_d10': CHG17_D10,
        'chg17_d20': CHG17_D20,
        'chg17_as': CHG17_AS,
        'chg38_d10': CHG38_D10,
        'chgj1': CHGJ1,
        'chgj2': CHGJ2,
        'med_e_a1': MED_E_A1,
        'med_e_p': MED_E_P,
        'sp_spec': SP_SPEC,
        'c3_best': C3_BEST,
        'delta_l17': DELTA_L17},
    'prereg': ('3133 MEMO section 5: '
               '(1) swap4 per-layer dvec '
               'dose-response matrix '
               '(carrier decomposition) '
               '+ pairwise additivity + '
               'same-session J2 anchor. '
               '(2) A1 first-divergence '
               'layer coordimization: '
               'A1-P L35 state-diff top50 '
               '-> P/A1 generation; '
               'generation step-scan of '
               'L17 dvec (steps 0..11) + '
               'full-N confirmation. '
               '(3) trajectory layer '
               'profiles P vs A1 + swap4/'
               'inject dm trajectories + '
               'descriptive 2nd-diff. '
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
z33 = np.load(D33 + r'\p131_readout.npz',
              allow_pickle=False)
assert z33['fdm0_A1'].shape == (128, 41)
assert z33['fdm0_P'].shape == (128, 41)
res33 = json.load(io.open(
    D33 + r'\result.json', encoding='utf-8'))
assert res33['smoke'] is False
log('frozen inputs ok (05/26/31/32/33)')

# ================================================================
# PART A: offline 3133 link asserts
# ================================================================
log('== PART A: 3133 link asserts ==')
V33 = res33['verdict']
assert V33 == EXP_V33, V33
pb33 = res33['part_b']
pc33 = res33['part_c']
assert abs(pb33['chg17_s0_d05']
           - CHG17_D05) < 1e-9
assert abs(pb33['chg17_s0_d10']
           - CHG17_D10) < 1e-9
assert abs(pb33['chg17_s0_d20']
           - CHG17_D20) < 1e-9
assert abs(pb33['chg17_as_d10']
           - CHG17_AS) < 1e-9
assert abs(pb33['chg38_s0_d10']
           - CHG38_D10) < 1e-9
assert abs(pb33['chgJ2_s0']
           - CHGJ2) < 1e-9
assert pc33['medE_A1'] == MED_E_A1
assert pc33['medE_P'] == MED_E_P
assert abs(pc33['sp_spec'] - SP_SPEC) < 1e-9
assert abs(max(pc33['c3_trials'][k]['chg']
               for k in pc33['c3_trials'])
           - C3_BEST) < 1e-9
raw33 = io.open(
    D33 + r'\result.json', 'rb').read()
sha33 = hashlib.sha256(
    raw33).hexdigest()[:8]
assert sha33 == RES33_SHA, sha33
co50 = z32['co50'].astype(np.int64)
co50_sha = hashlib.sha256(
    co50.tobytes()).hexdigest()[:8]
assert co50_sha == CO50_SHA, co50_sha
log('A hard asserts ok (3133 frozen, '
    'result sha8 %s, co50 sha8 %s)'
    % (sha33, co50_sha))

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
                if _m == 'allstep' \
                        or _st['step'] == 0 \
                        or _st['step'] == _m:
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


# ================================================================
# PART B1: swap-state capture -> dvec
# (3133 convention: capture at CAP_L
# under SWAP_L bypass)
# ================================================================
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
# margin-level semantic check vs z26
# profile idx 18 (3133 rev-g mapping:
# capture at layers[17] output = profile
# idx 18); batched capture is the polluter
# -> single-sample here must match.
_st = torch.tensor(
    base_states[17][:NP_CAP],
    device='cuda', dtype=torch.bfloat16)
with torch.inference_mode():
    _h = norm_g(_st).float().cpu().numpy()
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
del base_states, swap_states
gc.collect()
CKM = {'smoke': SMOKE, 'np': NP,
       'np_cap': NP_CAP, 'np_b': NP_B}
if CK['meta'] and CK['meta'] != CKM:
    log('CKPT meta mismatch -> discard')
    CK = {'done': [], 'data': {},
          'meta': CKM}
CK['meta'] = CKM
ck_save('b1', {
    'dvec': {str(l): dvec[l]
             for l in CAP_L},
    'norms': norms})

# ================================================================
# PART B2: carrier decomposition matrix
# 4 layers x 3 doses + 3 pairs + J2
# (16 trials, per-trial ckpt)
# ================================================================
log('== PART B2: carrier matrix ==')
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

TRIALS_B = []
for l in CAP_L:
    for di, dose in enumerate(DOSES):
        TRIALS_B.append(
            ('l%02d_d%d%d' % (l, 0, int(
                dose * 10)),
             [(l, dvec[l], dose, 0)]))
for (a, b) in PAIRS:
    TRIALS_B.append(
        ('p%02d_%02d' % (a, b),
         [(a, dvec[a], 1.0, 0),
          (b, dvec[b], 1.0, 0)]))
TRIALS_B.append(
    ('j2_s0',
     [(l, dvec[l], 1.0, 0)
      for l in CAP_L]))

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

# --- carrier gates ---
chg_mat = {}
for l in CAP_L:
    chg_mat[l] = [tr_res[
        'l%02d_d0%d' % (l, int(d * 10))
    ]['chg'] for d in DOSES]
if not SMOKE:
    mono_ok = all(
        chg_mat[l][0] <= chg_mat[l][1]
        + DOSE_SLACK
        and chg_mat[l][1] <= chg_mat[l][2]
        + DOSE_SLACK
        for l in CAP_L)
    c_dose = ('dose_monotone_all'
              if mono_ok
              else 'dose_nonmonotone')
    d2s = {l: chg_mat[l][2] for l in CAP_L}
    l17_max = d2s[17] >= max(
        d2s[l] for l in CAP_L)
    if l17_max:
        c_carrier = 'carrier_l17_max'
    elif max(d2s[l] for l in CAP_L
             if l != 17) > 2.0 * d2s[17]:
        c_carrier = 'carrier_dispersed'
    else:
        c_carrier = 'carrier_mixed'
    c_j2 = ('j2_session_match'
            if abs(tr_res['j2_s0']['chg']
                   - CHGJ2) <= J2_TOL
            else 'j2_session_drift')
    pair_ok = True
    pair_pred = {}
    pair_pred = {}
    for (a, b) in PAIRS:
        pa = tr_res['l%02d_d010' % a]['chg']
        pb = tr_res['l%02d_d010' % b]['chg']
        pred = 1.0 - (1.0 - pa) \
            * (1.0 - pb)
        pair_pred['%02d_%02d' % (a, b)] = \
            pred
        if abs(tr_res[
                'p%02d_%02d' % (a, b)
            ]['chg'] - pred) > ADD_TOL:
            pair_ok = False
    c_pair = ('pair_additive' if pair_ok
              else 'pair_superadditive')
    log('B-GATE: d2=%s -> %s|%s|%s|%s'
        % (' '.join('L%d:%.4f'
                    % (l, d2s[l])
                    for l in CAP_L),
           c_carrier, c_dose, c_j2, c_pair))
else:
    c_carrier = c_dose = c_j2 = c_pair \
        = 'smoke_subset'
    pair_pred = {}
    log('B-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C1: fork-layer coordimization
# profile idx 36 = layers[35] output
# (3133 mapping seq[L] = layers[L-1] out)
# ================================================================
log('== PART C1: fork coord ==')
rows_c1 = {'P': [pids_P_all[j]
                 for j in range(SCAN_N)],
           'A1': [pids_A1_all[j]
                  for j in range(SCAN_N)]}
_C1K = CK['data'].get('c1_states')
if _C1K is not None:
    hA1_36 = _C1K['hA1']
    hP_36 = _C1K['hP']
    co36 = _C1K['co36']
    delta36 = _C1K['delta']
    log('C1 states RESUMED from ckpt')
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
    co36 = np.argsort(med_abs)[
        -TOPK_COORD:].astype(np.int64)
    row_l2 = np.linalg.norm(d36, axis=1)
    delta36 = 0.10 * float(row_l2.mean())
    log('C1 d36 med||d||=%.4f delta=%.4f'
        % (float(np.median(row_l2)),
           delta36))
    ck_save('c1_states', {
        'hA1': hA1_36, 'hP': hP_36,
        'co36': co36, 'delta': delta36})
_C1T = CK['data'].get('c1_trials')
if _C1T is not None:
    c1_res = _C1T['res']
    log('C1 trials RESUMED')
else:
    c1_res = {}
    for dc in DIRS:
        gen_l = []
        for b0 in range(0, SCAN_N,
                        GEN_BATCH):
            batch = rows_c1[dc][
                b0:b0 + GEN_BATCH]
            gen_l.extend(gen_batch_g2(
                batch,
                inj=(FORK_LAYER, co36,
                     delta36, 1, 0)))
        base_l = [pad12(gen_base_sess_P[j])
                  for j in range(SCAN_N)]
        chg_l, first_l, fstep, _cum = \
            trial_metrics(gen_l, base_l)
        c1_res[dc] = {'chg': chg_l,
                      'first': int(first_l)}
        log('C1 inj->%s: chg=%.4f first=%d'
            % (dc, chg_l, first_l))
    ck_save('c1_trials', {'res': c1_res})
if not SMOKE:
    c_forkp = ('forkcoord_p_rewrite'
               if c1_res['P']['chg']
               >= TR_PART
               else 'forkcoord_p_null')
    log('C1-GATE: P chg=%.4f A1 chg=%.4f '
        '-> %s'
        % (c1_res['P']['chg'],
           c1_res['A1']['chg'], c_forkp))
else:
    c_forkp = 'smoke_subset'
    log('C1-GATE: SKIPPED in SMOKE')

# ================================================================
# PART C2: generation step-scan of L17
# dvec (d1.0): injection steps 0..11
# ================================================================
log('== PART C2: step scan ==')
rows_c2 = [pids_P_all[j]
           for j in range(SCAN_N)]
_C2K = CK['data'].get('c2_scan')
if _C2K is not None:
    scan_res = _C2K['res']
    log('C2 scan RESUMED')
else:
    scan_res = {}
    for s in STEPS_SCAN:
        gen_l = []
        for b0 in range(0, SCAN_N,
                        GEN_BATCH):
            batch = rows_c2[b0:b0
                            + GEN_BATCH]
            gen_l.extend(gen_batch_g2(
                batch,
                inj_vec=[(17,
                          dvec[17][b0:b0
                                   + len(batch)],
                          1.0, s)]))
        base_l = [pad12(gen_base_sess_P[j])
                  for j in range(SCAN_N)]
        chg_l, first_l, fstep, _cum = \
            trial_metrics(gen_l, base_l)
        scan_res[str(s)] = {
            'chg': chg_l,
            'first': int(first_l)}
        log('C2 step %02d: chg=%.4f first=%d'
            % (s, chg_l, first_l))
    ck_save('c2_scan', {'res': scan_res})
best_s = max(scan_res.keys(),
             key=lambda k: scan_res[k]['chg'])
if not SMOKE:
    c_step = ('stepscan_effective'
              if scan_res[best_s]['chg']
              >= STEP_GATE
              else 'stepscan_weak')
    log('C2-GATE: best step %s chg=%.4f '
        '-> %s'
        % (best_s, scan_res[best_s]['chg'],
           c_step))
else:
    c_step = 'smoke_subset'
    log('C2-GATE: SKIPPED in SMOKE')

# C3: full-N confirmation of best step
_C3K = CK['data'].get('c3_full')
if _C3K is not None:
    c3_full = _C3K
    fstep_store['fstep_stepscan_full'] \
        = c3_full['fstep']
    log('C3 full RESUMED chg=%.4f'
        % c3_full['chg'])
else:
    gen_l = []
    bs = int(best_s)
    for b0 in range(0, NP_B, GEN_BATCH):
        rows = rows_B[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            rows,
            inj_vec=[(17, dvec[17][b0:b0
                                   + len(rows)],
                      1.0, bs)]))
    chg_l, first_l, fstep, cum = \
        trial_metrics(gen_l, base12_P)
    c3_full = {'chg': chg_l,
               'first': int(first_l),
               'step': bs,
               'cum': [float(v) for v
                       in cum],
               'fstep': fstep}
    fstep_store['fstep_stepscan_full'] \
        = fstep
    log('C3 full step %d: chg=%.4f'
        % (bs, chg_l))
    ck_save('c3_full', c3_full)

# ================================================================
# PART D: trajectory layer profiles
# P vs A1 (base) + A1 swap4 dm + A1
# inject dm (descriptive 2nd diff)
# ================================================================
log('== PART D: traj profiles ==')
tf_idx = np.sort(np.random.default_rng(
    TF_SEED).choice(NP, TF_N,
                    replace=False)).astype(
    np.int64)
log('D tf_idx sha8=%s' % hashlib.sha256(
    tf_idx.tobytes()).hexdigest()[:8])
_DK = CK['data'].get('d_traj')
if _DK is not None:
    med_base = {'P': _DK['medP'],
                'A1': _DK['medA1']}
    dm_swap = _DK['dmSwap']
    dm_inj = _DK['dmInj']
    log('D traj RESUMED')
else:
    med_base = {}
    ml_store = {}
    for dc in DIRS:
        rows_dc = (pids_P_all if dc == 'P'
                   else pids_A1_all)
        mls = np.zeros((TF_N, NLG + 1,
                        N_NEW + 1),
                       dtype=np.float64)
        for jj in range(TF_N):
            j = int(tf_idx[jj])
            mls[jj] = margin_track_single(
                rows_dc[j],
                matg['traj_tokens'](dc, j)[2])
            if jj % 32 == 0:
                log('D %s row %d/%d'
                    % (dc, jj, TF_N))
        ml_store[dc] = mls
        med_base[dc] = np.median(
            np.abs(mls), axis=0)
    # A1 swap4 dm trajectory
    rows_A1 = [pids_A1_all[int(j)]
               for j in tf_idx]
    dms = np.zeros((TF_N, NLG + 1,
                    N_NEW + 1),
                   dtype=np.float64)
    for jj in range(TF_N):
        j = int(tf_idx[jj])
        ml_sw = margin_track_single(
            rows_A1[jj],
            matg['traj_tokens']('A1', j)[2],
            swap_layers=SWAP_L)
        dms[jj] = ml_sw - ml_store['A1'][jj]
        if jj % 32 == 0:
            log('D swap row %d/%d'
                % (jj, TF_N))
    dm_swap = np.median(dms, axis=0)
    # A1 L36-coord inject dm trajectory
    dmi = np.zeros((TF_N, NLG + 1,
                    N_NEW + 1),
                   dtype=np.float64)
    for jj in range(TF_N):
        j = int(tf_idx[jj])
        ml_in = margin_track_single(
            rows_A1[jj],
            matg['traj_tokens']('A1', j)[2],
            inject=(FORK_LAYER, co36,
                    delta36, 1))
        dmi[jj] = ml_in - ml_store['A1'][jj]
        if jj % 32 == 0:
            log('D inj row %d/%d'
                % (jj, TF_N))
    dm_inj = np.median(dmi, axis=0)
    ck_save('d_traj', {
        'medP': med_base['P'],
        'medA1': med_base['A1'],
        'dmSwap': dm_swap,
        'dmInj': dm_inj})
# descriptive second-difference:
# (swap - base-effect) - (inject - base-
# effect) interaction per (layer, step)
second_diff = dm_swap - dm_inj
k0_swap = int(np.argmax(
    np.abs(dm_swap[:, 0])))
k0_inj = int(np.argmax(
    np.abs(dm_inj[:, 0])))
# [rev-3134a] soft gate (amended BEFORE
# part-D observation, B2 still running):
# the swap4 dm k=0 profile ACCUMULATES
# over swap points (L38 dvec norm 100),
# so the |dm| PEAK is expected at/after
# the fork region up to the last profile
# idx - not AT 36 (36 is the EMERGENCE
# layer, 3133 medE). Peak in [33, 40]
# passes; an early peak would refute the
# fork-region attribution and is
# RECORDED (no crash). Emergence index
# (first crossing of 0.10*max, fork_
# stats convention) logged descriptively.
if not SMOKE:
    _amp = np.abs(dm_swap[:, 0])
    _emax = float(_amp.max())
    k_emerge = int(np.argmax(
        _amp >= 0.10 * _emax))
    if 33 <= k0_swap <= 40:
        c_traj = ('traj_swap_peak%d'
                  % k0_swap)
    else:
        c_traj = ('traj_swap_early%d'
                  % k0_swap)
    log('D-GATE: swap dm k=0 peak %d '
        'emerge(0.10max) %d (3133 medE='
        '36); inj peak %d -> %s'
        % (k0_swap, k_emerge, k0_inj,
           c_traj))
else:
    c_traj = 'smoke_subset'
    log('D-GATE: SKIPPED in SMOKE')

# ================================================================
# dumps
# ================================================================
if not SMOKE:
    assert len(tr_res) == len(TRIALS_B)
    assert all(l in chg_mat
               for l in CAP_L)
    log('CKPT integrity assert ok')
runtime = time.time() - T0
verdict = '|'.join([
    'a_3133_ok', c_carrier, c_dose, c_j2,
    c_pair, c_forkp, c_step, c_traj,
    'coverage_full'])
log('VERDICT: %s' % verdict)
result = {
    'name': NAME,
    'phase': 3134,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'res33_sha8': sha33,
        'asserts_3133': 'ok',
        'co50_sha8': co50_sha},
    'part_b': {
        'dvec_med_norm': {
            str(l): norms[l]
            for l in CAP_L},
        'xphase_base_match_P': xphase_P,
        'chg_matrix': {
            str(l): [float(v) for v in
                     chg_mat[l]]
            for l in CAP_L},
        'trials': tr_res,
        'pair_pred': {
            k: float(v) for k, v in
            pair_pred.items()},
        'b_carrier': c_carrier,
        'b_dose': c_dose,
        'b_j2': c_j2,
        'b_pair': c_pair},
    'part_c': {
        'fork_layer': FORK_LAYER,
        'fork_prof_idx': FORK_PROF_IDX,
        'co36_sha8': hashlib.sha256(
            co36.tobytes()).hexdigest()[:8],
        'delta36': delta36,
        'c1_trials': c1_res,
        'c_forkp': c_forkp,
        'scan': scan_res,
        'best_step': best_s,
        'c_step': c_step,
        'c3_full': {k: v for k, v in
                    c3_full.items()
                    if k != 'fstep'}},
    'part_d': {
        'tf_idx': [int(v) for v in
                   tf_idx],
        'k0_swap_peak': k0_swap,
        'k0_inj_peak': k0_inj,
        'c_traj': c_traj,
        'second_diff_summary': {
            'med_abs': float(np.median(
                np.abs(second_diff))),
            'max_abs': float(np.max(
                np.abs(second_diff)))}},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'tf_idx': tf_idx,
    'co36': co36,
    'dvec17': dvec[17][:NP_B],
    'dvec29': dvec[29][:NP_B],
    'dvec33': dvec[33][:NP_B],
    'dvec38': dvec[38][:NP_B],
    'dvec_med_norm': np.array(
        [norms[l] for l in CAP_L]),
    'med_traj_P': med_base['P'],
    'med_traj_A1': med_base['A1'],
    'dm_swap_traj': dm_swap,
    'dm_inj_traj': dm_inj,
    'second_diff': second_diff}
for k, v in fstep_store.items():
    npz_out[k] = v
for l in CAP_L:
    npz_out['dvec_full_%d' % l] = \
        dvec[l].astype(np.float16)
np.savez(os.path.join(OUT,
                      'p132_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p132_readout.npz (%d keys)'
    % len(npz_out))
# all data persisted -> drop ckpt
# (P3134_CKPT_KEEP=1 keeps it for a
# full-RESUME smoke validation)
if os.path.exists(CKPTF) \
        and not os.environ.get(
            'P3134_CKPT_KEEP'):
    os.remove(CKPTF)
    log('CKPT removed (data fully '
        'persisted)')
log('P3134 DONE (%.1fs)'
    % (time.time() - T0))
