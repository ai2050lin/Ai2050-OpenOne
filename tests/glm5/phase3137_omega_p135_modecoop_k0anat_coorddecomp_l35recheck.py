# -*- coding: utf-8 -*-
"""Phase 3137 (Omega-P135): prompt-only vs
allstep mode x layer x dose matrix + k0
negative-contribution anatomy + co36@L17
coordinate decomposition + L35 union dose
recheck.

Preregistered in 3136 closeout (task 80):
(1) mode x layer x dose matrix: layers
{17,29,33,38} x modes {prompt-only,
allstep} x absolute scales {1,2,4} on
SCAN_N=128 P rows, session base; quantify
the timing-layer interaction law (3134
chg_matrix was prompt-only 672 rows, 3136
B2 was allstep 128 rows).  Plus one 672-row
prompt-only L38 s2.0 trial to bit-link the
3134 anchor (0.08929);
(2) k0 anatomy: w8full[0..8] vs w8_nok0
[1..8] vs w8_k0only [0] + drop3 anchor
replication; row-level flip decomposition;
(3) co36@L17 coordinate decomposition:
co36_rank top25/bot25 @L17 (delta
DELTA_L17, sgn -1, mode 0, A1 rows) +
co36_full dose {0.5,1,2,4} + co50_full
dose {0.5,1,2};
(4) L35 union dose recheck: L35_union
deltas {1,2,4}xDELTA36_34 + L35_co36 s2.0
(A1 rows, mode 0).

No swap capture / cos-rho readouts needed
(behavior-only phase): dvecs frozen from
3135 npz (sha-anchored); session base
capture all-40-layers for the xphase probe
record only."""
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
NAME = ('omega_p135_modecoop_'
        'k0anat_coorddecomp_l35recheck')
SMOKE = os.environ.get('P3137_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D32 = RDIR + r'\phase3132' \
      r'\omega_p130_forkcausal_single256'
D34 = RDIR + r'\phase3134' \
      r'\omega_p132_carrier_matrix_' \
      r'forkcoord_stepscan'
D35 = RDIR + r'\phase3135' \
      r'\omega_p133_conduction_' \
      r'co36ablation_window'
D36 = RDIR + r'\phase3136' \
      r'\omega_p134_conddose_' \
      r'crossmatrix_w8drop'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3137', NAME)
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
CKPTF = os.path.join(OUT, 'p135_ckpt.pkl')
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

# carrier layers (frozen 3135 fp32)
CAP_L_D = [17, 29, 33, 38]
L17 = 17
L35 = 35
# absolute injection scales for PART E
# (3134 chg_matrix semantics: scale is
# the direct dvec multiplier)
SCALES = [1.0, 2.0, 4.0]
ALL_L = list(range(40))
W8_IDX = list(range(9))
# mode gap gate (chg_all - chg_po at
# scale 2.0)
MODE_GAP = 0.30
PO_MONO_TOL = 0.02
PO_FLAT_MAX = 0.15
# k0 anatomy gates
K0_TOL = 0.05
K0_ALONE_MAX = 0.10
# coordinate decomposition gates
TOPBOT_RATIO = 2.0
TOPBOT_MIN = 0.08
CD_MONO_TOL = 0.02
CD_FLAT_MAX = 0.10
# L35 recheck gates
L35_ZERO_MAX = 0.12
L35_EMERGENT = 0.25
# soft repro tolerance (record only)
SOFT_TOL = 0.08

TF_SEED = 3137
RNG_SEED = 3137

# ---------------- frozen 3136 anchors --
EXP_V36 = ('a_3135_ok|fidelity_dose_'
           'sensitive|rho_linear|'
           'dose_resp_monotone|'
           'diagonal_weak|union_add_all|'
           'w8_top1_dominant|w8_repro_ok|'
           'coverage_full')
RES36_SHA = '3903af46'
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
# 3136 soft anchors (record-only)
W8FULL_36 = 0.6484375
DROP3_36 = 0.40625
DROP0_36 = 0.75
K0_CONTRIB_36 = -0.1015625
K3_CONTRIB_36 = 0.2421875
CMAT36 = {'L17_co50': 0.421875,
          'L17_co36': 0.140625,
          'L17_union': 0.5546875,
          'L35_co50': 0.0625,
          'L35_co36': 0.078125,
          'L35_union': 0.0546875,
          'both_co50': 0.390625,
          'both_co36': 0.1328125,
          'both_union': 0.5546875}
DOSECHG36_D10 = {'29': 0.640625,
                 '33': 0.515625,
                 '38': 0.9453125}
# 3134 prompt-only anchor (672 rows,
# scale 2.0, L38)
L38_PO_672_34 = 0.0892857142857143
# 3134 dose-2.0 cells (all four layers,
# MEMORY anchors) used to self-locate
# the scale-2.0 column in chg_matrix
CAND34 = [0.25595238095238093,
          0.0625,
          0.049107142857142905,
          0.0892857142857143]

# ---------------- seal ------------------
SEAL = {
    'phase': 3137,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_NEW': N_NEW,
        'CAP_L_D': CAP_L_D, 'L17': L17,
        'L35': L35, 'SCALES': SCALES,
        'ALL_L': ALL_L, 'W8_IDX': W8_IDX,
        'MODE_GAP': MODE_GAP,
        'PO_MONO_TOL': PO_MONO_TOL,
        'PO_FLAT_MAX': PO_FLAT_MAX,
        'K0_TOL': K0_TOL,
        'K0_ALONE_MAX': K0_ALONE_MAX,
        'TOPBOT_RATIO': TOPBOT_RATIO,
        'TOPBOT_MIN': TOPBOT_MIN,
        'CD_MONO_TOL': CD_MONO_TOL,
        'CD_FLAT_MAX': CD_FLAT_MAX,
        'L35_ZERO_MAX': L35_ZERO_MAX,
        'L35_EMERGENT': L35_EMERGENT,
        'SOFT_TOL': SOFT_TOL,
        'TF_SEED': TF_SEED,
        'RNG_SEED': RNG_SEED},
    'anchors': {
        'res36_verdict': EXP_V36,
        'res36_sha8': RES36_SHA,
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
        'w8full_36_soft': W8FULL_36,
        'drop3_36_soft': DROP3_36,
        'drop0_36_soft': DROP0_36,
        'k0_contrib_36_soft': K0_CONTRIB_36,
        'k3_contrib_36_soft': K3_CONTRIB_36,
        'cmat36_soft': CMAT36,
        'dosechg36_d10_soft': DOSECHG36_D10,
        'l38_po_672_34_soft': L38_PO_672_34,
        'cand34_d2_cells': CAND34},
    'prereg': ('3136 closeout task 80: '
               '(1) prompt-only vs allstep '
               'mode x layer {17,29,33,38} '
               'x absolute scale {1,2,4} '
               'matrix on 128 P rows + one '
               '672-row prompt-only L38 s2 '
               'trial linking the 3134 '
               'anchor; (2) k0 negative '
               'contribution anatomy: '
               'w8full vs w8_nok0 vs '
               'w8_k0only + drop3 anchor, '
               'row-level flip decomposition; '
               '(3) co36@L17 coordinate '
               'decomposition: co36_rank '
               'top25/bot25 + co36_full dose '
               '{0.5,1,2,4} + co50_full dose '
               '{0.5,1,2} (A1 rows, delta '
               'DELTA_L17 sgn -1, mode 0); '
               '(4) L35 union dose recheck: '
               'deltas {1,2,4}xDELTA36_34 + '
               'L35_co36 s2.0. Frozen before '
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
res36 = json.load(io.open(
    D36 + r'\result.json', encoding='utf-8'))
assert res36['smoke'] is False
res34 = json.load(io.open(
    D34 + r'\result.json', encoding='utf-8'))
assert res34['smoke'] is False
log('frozen inputs ok (26/32/34/35/36, '
    'dvec sha anchored)')

# ================================================================
# PART A: offline 3136/3134 link asserts
# ================================================================
log('== PART A: 3136/3134 link asserts ==')
V36 = res36['verdict']
assert V36 == EXP_V36, V36
raw36 = io.open(
    D36 + r'\result.json', 'rb').read()
sha36 = hashlib.sha256(
    raw36).hexdigest()[:8]
assert sha36 == RES36_SHA, sha36
pd36 = res36['part_d']
pc36 = res36['part_c']
pb36 = res36['part_b']
assert abs(pd36['d_res']['w8full']['chg']
           - W8FULL_36) < 1e-9
assert abs(pd36['d_res']['drop3']['chg']
           - DROP3_36) < 1e-9
assert abs(pd36['d_res']['drop0']['chg']
           - DROP0_36) < 1e-9
assert abs(pd36['drop1_contrib']['0']
           - K0_CONTRIB_36) < 1e-9
assert abs(pd36['drop1_contrib']['3']
           - K3_CONTRIB_36) < 1e-9
for k, v in CMAT36.items():
    assert abs(pc36['matrix'][k]['chg']
               - v) < 1e-9, k
for l, v in DOSECHG36_D10.items():
    assert abs(pb36['dose_chg']
               ['%s_1.0' % l]['chg']
               - v) < 1e-9, l
# 3134 prompt-only anchor: locate the
# scale-2.0 column by the four MEMORY
# anchor cells (L17 0.2560 / L29 0.0625 /
# L33 0.0491 / L38 0.0893)
b34 = res34['part_b']
cm34 = b34['chg_matrix']
i2 = None
for _i in range(3):
    _vals = [float(cm34[l][_i])
             for l in ('17', '29', '33',
                       '38')]
    if all(abs(x - y) < 1e-9
           for x, y in zip(_vals, CAND34)):
        i2 = _i
        break
assert i2 is not None, 'no dose-2 cell'
got38 = float(cm34['38'][i2])
assert abs(got38 - L38_PO_672_34) < 1e-9, \
    got38
log('A hard asserts ok (3136 frozen sha8 '
    '%s verdict match; 3134 chg_matrix '
    'scale-2.0 col=%d cell=%.6f)'
    % (sha36, i2, got38))

# ================================================================
# materials factory (3125/.../3136
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
      LIST of such tuples.  mode: 0 =
      prompt forward only; 'allstep' =
      every forward.
    inj_vec: per-layer transplants
      (il, dvec_batch (n,HID) fp32, scale,
       mode). mode: 'allstep' = every
       forward; int s = prompt forward +
       decode step s; list = explicit
       forward-index window (3135
       semantics)."""
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


# ================================================================
# PART B1: frozen dvec load + session
# base capture (all 40 layers, xphase
# probe record).  No swap capture: 3137
# is behavior-only (no cos/rho).
# ================================================================
log('== PART B1: frozen dvecs + base ==')
dvec = {int(l): z35['dvec%d' % l]
        .astype(np.float32)
        for l in CAP_L_D}
norms = {l: float(np.median(np.linalg.norm(
    dvec[l], axis=1)))
    for l in CAP_L_D}
for l in CAP_L_D:
    log('B1 frozen dvec L%02d med||d||='
        '%.4f (3135 anchor %.4f, drift '
        '%.2e)'
        % (l, norms[l],
           DVEC_MED_35[CAP_L_D.index(l)],
           abs(norms[l]
               - DVEC_MED_35[
                   CAP_L_D.index(l)])))
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
    del base_states
    gc.collect()
    torch.cuda.empty_cache()

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

# --- A1 session base (SCAN_N rows) ----
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

# ================================================================
# PART E (prereg 1): mode x layer x
# scale matrix.  P rows, SCAN_N, frozen
# dvec carriers at own layer, scale =
# absolute dvec multiplier (3134
# semantics).  modes: po (prompt-only,
# mode 0) / all (allstep).
# ================================================================
log('== PART E: mode x layer x scale ==')
rows_scan = [pids_P_all[j]
             for j in range(SCAN_N)]
base_E = base12_P[:SCAN_N]
emode = {}


def _run_vec_trial(tname, il, mode,
                   scale, rows, base12,
                   store):
    """dvec carrier trial with per-trial
    ckpt."""
    _K = CK['data'].get(tname)
    if _K is not None:
        store[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname,
               store[tname]['chg']))
        return _K.get('fstep')
    gen_l = []
    for b0 in range(0, len(rows),
                    GEN_BATCH):
        batch = rows[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il,
                      dvec[il][b0:b0
                               + len(batch)],
                      scale, mode)]))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base12)
    store[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': store[tname],
                    'fstep': fstep})
    return fstep


fstep_E = {}
for il in CAP_L_D:
    for mode, mtag in ((0, 'po'),
                       ('allstep', 'all')):
        for s in SCALES:
            tname = '%s_l%02d_s%.1f' % (
                mtag, il, s)
            fstep_E[tname] = _run_vec_trial(
                tname, il, mode, s,
                rows_scan, base_E, emode)
    ga = emode['all_l%02d_s2.0' % il]['chg']
    gp = emode['po_l%02d_s2.0' % il]['chg']
    log('E mode-gap L%02d: all=%.4f po='
        '%.4f gap=%.4f ratio=%.2f'
        % (il, ga, gp, ga - gp,
           ga / max(gp, 1e-9)))

# 672-row prompt-only L38 s2.0 anchor
# (links 3134 chg_matrix scale-2.0 cell)
_A672 = CK['data'].get('a672_l38_po_s2')
if _A672 is not None:
    a672 = _A672['res']
    log('a672 RESUMED chg=%.4f'
        % a672['chg'])
else:
    gen_672 = []
    _N672 = SCAN_N if SMOKE else NP
    for b0 in range(0, _N672, GEN_BATCH):
        batch = pids_P_all[b0:b0 + GEN_BATCH]
        gen_672.extend(gen_batch_g2(
            batch,
            inj_vec=[(38, dvec[38][b0:b0
                                   + len(batch)],
                      2.0, 0)]))
    chg_l, first_l, _fs, _cum = \
        trial_metrics(gen_672, base12_P)
    a672 = {'chg': chg_l,
            'first': int(first_l)}
    log('a672_l38_po_s2 (672 rows): '
        'chg=%.4f first=%d (3134 anchor '
        '%.6f)' % (chg_l, first_l,
                   L38_PO_672_34))
    ck_save('a672_l38_po_s2', {
        'res': a672})

if not SMOKE:
    gaps = []
    for il in CAP_L_D:
        ga = emode['all_l%02d_s2.0'
                   % il]['chg']
        gp = emode['po_l%02d_s2.0'
                   % il]['chg']
        gaps.append(ga - gp)
    n_big = sum(1 for g in gaps
                if g >= MODE_GAP)
    if n_big == len(CAP_L_D):
        c_mode = 'mode_gap_all'
    elif n_big >= 2:
        c_mode = 'mode_gap_majority'
    else:
        c_mode = 'mode_gap_sub'
    monos = []
    for il in CAP_L_D:
        cs = [emode['po_l%02d_s%.1f'
                    % (il, s)]['chg']
              for s in SCALES]
        m_i = all(
            cs[i + 1] >= cs[i] - PO_MONO_TOL
            for i in range(len(cs) - 1))
        monos.append(m_i or
                     max(cs) <= PO_FLAT_MAX)
    if all(monos):
        c_po = 'po_dose_monotone'
    else:
        c_po = 'po_dose_mixed'
    log('E-GATE: gaps=%s -> %s; po '
        'monotone=%s -> %s'
        % (['%.4f' % g for g in gaps],
           c_mode, monos, c_po))
    # soft anchors vs 3136 allstep d1.0
    # (= absolute scale 2.0)
    for l, v in DOSECHG36_D10.items():
        got = emode['all_l%02d_s2.0'
                    % int(l)]['chg']
        log('E-SOFT: all L%s s2.0 %.6f vs '
            '3136 %.6f (drift %.2e)'
            % (l, got, v, abs(got - v)))
    log('E-SOFT: a672 L38 po s2.0 %.6f vs '
        '3134 %.6f (drift %.2e)'
        % (a672['chg'], L38_PO_672_34,
           abs(a672['chg']
               - L38_PO_672_34)))
else:
    c_mode = c_po = 'smoke_subset'
    log('E-GATE: SKIPPED in SMOKE')

# ================================================================
# PART F (prereg 2): k0 anatomy.  P
# rows, dvec17 window trials (scale
# 1.0): w8full [0..8], w8_nok0 [1..8],
# w8_k0only [0], drop3 [0,1,2,4..8].
# Row-level flip decomposition.
# ================================================================
log('== PART F: k0 anatomy ==')
dvec17_scan = dvec[17][:SCAN_N]
F_TRIALS = [('w8full', list(W8_IDX)),
            ('w8_nok0',
             [i for i in W8_IDX if i != 0]),
            ('w8_k0only', [0]),
            ('drop3',
             [i for i in W8_IDX
              if i != 3])]
fres = {}
fstep_F = {}
for (tname, win) in F_TRIALS:
    _K = CK['data'].get('f_%s' % tname)
    if _K is not None:
        fres[tname] = _K['res']
        fstep_F[tname] = _K['fstep']
        log('F %s RESUMED chg=%.4f'
            % (tname, fres[tname]['chg']))
        continue
    gen_l = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(17,
                      dvec17_scan[b0:b0
                                  + len(batch)],
                      1.0, win)]))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base_E)
    fres[tname] = {'chg': chg_l,
                   'first': int(first_l)}
    fstep_F[tname] = fstep
    log('F %s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save('f_%s' % tname,
            {'res': fres[tname],
             'fstep': fstep})

if not SMOKE:
    chg_full = fres['w8full']['chg']
    chg_nok0 = fres['w8_nok0']['chg']
    chg_k0o = fres['w8_k0only']['chg']
    diff_k0 = chg_nok0 - chg_full
    if diff_k0 > K0_TOL:
        c_k0 = 'k0_negative_confirmed'
    elif diff_k0 < -K0_TOL:
        c_k0 = 'k0_positive'
    else:
        c_k0 = 'k0_neutral'
    c_k0a = ('k0_alone_inert'
             if chg_k0o <= K0_ALONE_MAX
             else 'k0_alone_active')
    # row-level flip decomposition
    fl_full = fstep_F['w8full'] >= 0
    fl_nok0 = fstep_F['w8_nok0'] >= 0
    n_both = int((fl_full & fl_nok0).sum())
    n_only_full = int(
        (fl_full & ~fl_nok0).sum())
    n_only_nok0 = int(
        (~fl_full & fl_nok0).sum())
    log('F-GATE: w8full=%.4f nok0=%.4f '
        '(diff %.4f -> %s); k0only=%.4f '
        '-> %s; flips full=%d nok0=%d '
        'both=%d only_full=%d '
        'only_nok0=%d'
        % (chg_full, chg_nok0, diff_k0,
           c_k0, chg_k0o, c_k0a,
           int(fl_full.sum()),
           int(fl_nok0.sum()), n_both,
           n_only_full, n_only_nok0))
    log('F-SOFT: w8full %.6f vs 3136 '
        '%.6f (drift %.2e) | drop3 %.6f '
        'vs 3136 %.6f (drift %.2e)'
        % (chg_full, W8FULL_36,
           abs(chg_full - W8FULL_36),
           fres['drop3']['chg'], DROP3_36,
           abs(fres['drop3']['chg']
               - DROP3_36)))
else:
    c_k0 = c_k0a = 'smoke_subset'
    n_both = n_only_full = n_only_nok0 = 0
    log('F-GATE: SKIPPED in SMOKE')

# ================================================================
# PART G (prereg 3): co36@L17 coordinate
# decomposition.  A1 rows, L17, delta
# DELTA_L17 x dose_scale, sgn -1, mode
# 0 (prompt-only).  top25/bot25 from
# co36_rank ordering.
# ================================================================
log('== PART G: co36@L17 decomposition ==')
co36 = z35['co36'].astype(np.int64)
cr = z35['co36_rank'].astype(np.int64)
if set(cr.tolist()) == set(co36.tolist()):
    ORDER = cr
    ordtag = 'coords'
else:
    assert set(cr.tolist()) == \
        set(range(len(co36))), cr[:8]
    ORDER = co36[cr]
    ordtag = 'perm'
assert len(ORDER) == 50
top25 = ORDER[:25]
bot25 = ORDER[-25:]
log('G co36_rank semantics=%s '
    'len=%d (top25=%s... bot25=%s...)'
    % (ordtag, len(ORDER),
       top25[:5].tolist(),
       bot25[:5].tolist()))
G_TRIALS = [('g_top25', L17, top25, 1.0),
            ('g_bot25', L17, bot25, 1.0),
            ('g_co36_d05', L17, co36, 0.5),
            ('g_co36_d1', L17, co36, 1.0),
            ('g_co36_d2', L17, co36, 2.0),
            ('g_co36_d4', L17, co36, 4.0),
            ('g_co50_d05', L17, co50, 0.5),
            ('g_co50_d1', L17, co50, 1.0),
            ('g_co50_d2', L17, co50, 2.0)]
gres = {}
for (tname, il, coords, ds) in G_TRIALS:
    _K = CK['data'].get(tname)
    if _K is not None:
        gres[tname] = _K['res']
        log('G %s RESUMED chg=%.4f'
            % (tname, gres[tname]['chg']))
        continue
    gen_l = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        batch = rows_c[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=[(il, coords,
                  ds * DELTA_L17, -1, 0)]))
    chg_l, first_l, _fs, _cum = \
        trial_metrics(gen_l, base12_A1)
    gres[tname] = {'chg': chg_l,
                   'first': int(first_l)}
    log('G %s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': gres[tname]})

if not SMOKE:
    chg_top = gres['g_top25']['chg']
    chg_bot = gres['g_bot25']['chg']
    if chg_top >= max(
            TOPBOT_RATIO * chg_bot,
            TOPBOT_MIN):
        c_tb = 'co36_l17_top_heavy'
    elif chg_bot >= max(
            TOPBOT_RATIO * chg_top,
            TOPBOT_MIN):
        c_tb = 'co36_l17_bot_heavy'
    else:
        c_tb = 'co36_l17_flat'
    cds = [gres['g_co36_d%s'
                % ('05' if d == 0.5
                   else str(int(d)))]['chg']
           for d in (0.5, 1.0, 2.0, 4.0)]
    cd_mono = all(
        cds[i + 1] >= cds[i] - CD_MONO_TOL
        for i in range(len(cds) - 1))
    if max(cds) <= CD_FLAT_MAX:
        c_cd = 'coord_dose_flat'
    elif cd_mono:
        c_cd = 'coord_dose_monotone'
    else:
        c_cd = 'coord_dose_mixed'
    log('G-GATE: top25=%.4f bot25=%.4f '
        '-> %s; co36 dose curve=%s -> %s'
        % (chg_top, chg_bot, c_tb,
           ['%.4f' % c for c in cds], c_cd))
    log('G-SOFT: co36 d1 %.6f vs 3136 '
        'L17_co36 %.6f (drift %.2e) | '
        'co50 d1 %.6f vs 3136 L17_co50 '
        '%.6f (drift %.2e)'
        % (gres['g_co36_d1']['chg'],
           CMAT36['L17_co36'],
           abs(gres['g_co36_d1']['chg']
               - CMAT36['L17_co36']),
           gres['g_co50_d1']['chg'],
           CMAT36['L17_co50'],
           abs(gres['g_co50_d1']['chg']
               - CMAT36['L17_co50'])))
else:
    c_tb = c_cd = 'smoke_subset'
    log('G-GATE: SKIPPED in SMOKE')

# ================================================================
# PART H (prereg 4): L35 union dose
# recheck.  A1 rows, L35, delta
# scale x DELTA36_34, sgn +1, mode 0.
# ================================================================
log('== PART H: L35 union recheck ==')
H_TRIALS = [('h_l35union_d1', union3650,
             1.0),
            ('h_l35union_d2', union3650,
             2.0),
            ('h_l35union_d4', union3650,
             4.0),
            ('h_l35co36_d2', co36, 2.0)]
hres = {}
for (tname, coords, ds) in H_TRIALS:
    _K = CK['data'].get(tname)
    if _K is not None:
        hres[tname] = _K['res']
        log('H %s RESUMED chg=%.4f'
            % (tname, hres[tname]['chg']))
        continue
    gen_l = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        batch = rows_c[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=[(L35, coords,
                  ds * DELTA36_34, 1, 0)]))
    chg_l, first_l, _fs, _cum = \
        trial_metrics(gen_l, base12_A1)
    hres[tname] = {'chg': chg_l,
                   'first': int(first_l)}
    log('H %s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': hres[tname]})

if not SMOKE:
    hu = [hres['h_l35union_d%s'
              % ('1' if d == 1.0
                 else str(int(d)))]['chg']
          for d in (1.0, 2.0, 4.0)]
    if max(hu) <= L35_ZERO_MAX:
        c_l35 = 'l35_union_zero_dose_robust'
    elif max(hu) >= L35_EMERGENT:
        c_l35 = 'l35_union_dose_emergent'
    else:
        c_l35 = 'l35_union_weak_partial'
    log('H-GATE: l35union d1/2/4=%s -> '
        '%s; l35co36 d2=%.4f (3136 d1 '
        'anchor %.6f)'
        % (['%.4f' % c for c in hu], c_l35,
           hres['h_l35co36_d2']['chg'],
           CMAT36['L35_co36']))
    log('H-SOFT: l35union d1 %.6f vs 3136 '
        '%.6f (drift %.2e)'
        % (hres['h_l35union_d1']['chg'],
           CMAT36['L35_union'],
           abs(hres['h_l35union_d1']['chg']
               - CMAT36['L35_union'])))
else:
    c_l35 = 'smoke_subset'
    log('H-GATE: SKIPPED in SMOKE')

# ================================================================
# dumps
# ================================================================
if not SMOKE:
    assert len(emode) == len(CAP_L_D) * 2 \
        * len(SCALES)
    assert len(fres) == len(F_TRIALS)
    assert len(gres) == len(G_TRIALS)
    assert len(hres) == len(H_TRIALS)
    log('CKPT integrity assert ok')
runtime = time.time() - T0
verdict = '|'.join([
    'a_3136_ok', c_mode, c_po, c_k0, c_k0a,
    c_tb, c_cd, c_l35, 'coverage_full'])
log('VERDICT: %s' % verdict)

result = {
    'name': NAME,
    'phase': 3137,
    'smoke': SMOKE,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        io.open(SEALF, 'rb').read())
    .hexdigest()[:8],
    'part_a': {
        'res36_sha8': sha36,
        'asserts_3136_3134': 'ok',
        'co50_sha8': co50_sha,
        'union_sha8': u_sha,
        'l38_po_672_34': L38_PO_672_34},
    'part_e': {
        'scales': SCALES,
        'xphase_base_match_P': xphase_P,
        'mode_matrix': emode,
        'a672_l38_po_s2': a672,
        'e_mode': c_mode,
        'e_po': c_po},
    'part_f': {
        'w8_idx': W8_IDX,
        'f_res': fres,
        'f_k0': c_k0,
        'f_k0alone': c_k0a,
        'flip_decomp': {
            'n_full': int(
                (fstep_F['w8full'] >= 0)
                .sum()),
            'n_nok0': int(
                (fstep_F['w8_nok0'] >= 0)
                .sum()),
            'n_both': n_both,
            'n_only_full': n_only_full,
            'n_only_nok0': n_only_nok0}},
    'part_g': {
        'ordtag': ordtag,
        'top25': top25.tolist(),
        'bot25': bot25.tolist(),
        'g_res': gres,
        'g_topbot': c_tb,
        'g_cdose': c_cd},
    'part_h': {
        'h_res': hres,
        'h_l35': c_l35},
    'verdict': verdict}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {}
for k, v in fstep_E.items():
    if v is not None:
        npz_out['fstep_%s' % k] = v
for k, v in fstep_F.items():
    npz_out['fstep_%s' % k] = v
npz_out['co36_rank_order'] = ORDER
np.savez(os.path.join(OUT,
                      'p135_readout.npz'),
         **npz_out)
log('dumps done: result.json + seal + '
    'p135_readout.npz (%d keys)'
    % len(npz_out))
# all data persisted -> drop ckpt
# (P3137_CKPT_KEEP=1 keeps it for a
# full-RESUME smoke validation)
if os.path.exists(CKPTF) \
        and not os.environ.get(
            'P3137_CKPT_KEEP'):
    os.remove(CKPTF)
    log('CKPT removed (data fully '
        'persisted)')
log('P3137 DONE (%.1fs)'
    % (time.time() - T0))
