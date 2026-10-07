# -*- coding: utf-8 -*-
"""Phase 3141 (Omega-P139): cinj
functional-peak refinement (step1 x
dose) + WR-PC1 x dvec joint-injection
superposition law + co36 identity-energy
enrichment causal split + write-induced
retrieval.

Preregistered in 3140 closeout (aligned
with FINGERPRINT_PARADIGM_PLAN.md
Omega-P139):
(1) cinj peak refinement: C(T2) template
vector injection at CL7 = L19/23/24/25/
26/27/28 (3140 d2 plateau 0.273-0.289
plus L19 0.492 outlier) x doses
{1,2,4} x step1 mode (3135 semantics:
prompt forward + decode step 1) vs
allstep d2 controls (4 bit-repro anchors
from 3140: L23 0.2890625, L25 0.2734375,
L26 0.2890625, L27 0.28125) + iinj L17
d1 allstep bit-anchor 0.328125;
(2) WR-PC1 x dvec29 joint injection at
L29: pc1-alone d2 bit-anchor 0.203125,
dvec29-alone d{1,2}, joint d{1,2} ->
superposition law (additive vs
probability-independence vs blocking);
(3) co36 50-coordinate identity-energy
enrichment causal split: rank coords by
mean per-row energy fraction fr
(3140 D2 semantics, z=6.85) -> high25/
low25 injected at L17 (3137 G semantics:
DELTA_L17 x dose, sgn -1, mode 0, A1
rows) x doses {1,2} + co36 d1/d2 bit
anchors (0.140625 / 0.2265625) + co50
d1 bit anchor 0.421875;
(4) write-induced retrieval: 84 unseen
pairs, V1/V2 prompts, forward-state
retrieval (V1 must bit-replicate 3140
retr_v1) vs post-generation state
retrieval (write history induced) ->
tests 3140 finding 5 (retrieval failure
= missing write history).

Bank shards REUSED from 3138 full run
(no recapture). dvecs frozen from 3135
(sha-anchored). Session baselines P+A1
with xphase==1.0 asserts vs z26."""
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
NAME = ('omega_p139_cinjrefine_wrjoint_'
        'coenrich_writeinduce')
SMOKE = os.environ.get('P3141_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D35 = RDIR + r'\phase3135' \
      r'\omega_p133_conduction_' \
      r'co36ablation_window'
D36 = RDIR + r'\phase3136' \
      r'\omega_p134_conddose_crossmatrix_w8drop'
D37 = RDIR + r'\phase3137' \
      r'\omega_p135_modecoop_k0anat_' \
      r'coorddecomp_l35recheck'
D38 = RDIR + r'\phase3138' \
      r'\omega_p136_statebank_' \
      r'bicdecomp_probecontrast'
D39 = RDIR + r'\phase3139' \
      r'\omega_p137_idinteract_portconsume_' \
      r'rewrite_xmat'
D40 = RDIR + r'\phase3140' \
      r'\omega_p138_layerspec_owndecay_' \
      r'wrpath_retrattr'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3141', NAME)
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
CKPTF = os.path.join(OUT, 'p139_ckpt.pkl')
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
N_T = 4
DIRS = ('P', 'A1')
ALL_L = list(range(40))
KEY_L = (17, 29, 33, 38)
# decomposition layer set: cinj
# refinement needs L19/23/24/25/26/27/28
# vectors, PC needs 26/29, PART E needs
# IDEINT_P[17], retrieval needs KEY_L
CL = (17, 19, 23, 24, 25, 26, 27, 28,
      29, 33, 38)
CL7 = (19, 23, 24, 25, 26, 27, 28)
STEP1 = 1
C_DOSES = (1.0, 2.0, 4.0)
SCAN_N = 4 if SMOKE else 128
GEN_BATCH = 4 if SMOKE else 32
N_NEW = 12
D_DOSES = (1.0, 2.0)
E_DOSES = (1.0, 2.0)
V2_GAIN = 0.05
XMAT_RETR_MIN = 0.30
WI_GAIN = 0.05
JOINT_TOL = 0.05
BLOCK_TOL = 0.02
RNG_SEED = 3141

# ---------------- frozen 3140 anchors --
EXP_V14 = ('a_3139_ok|spec_repro_3139_ok|'
           'iinj_peak_l19|cinj_peak_l38|'
           'peaks_separated|'
           'spec_dose_monotone|own_below_nb|'
           'own_rank_chance|co_enriched|'
           'wr_active|wr_pcvar_recorded|'
           'retr_v2_flat|retr_v2_below|'
           'retr_v1_repro_ok|xphase_ok|'
           'coverage_full')
RES14_SHA = '6d699902'
XPHASE14 = 1.0
N_REPRO14 = 9
CINJ_D2 = {'23': 0.2890625,
           '25': 0.2734375,
           '26': 0.2890625,
           '27': 0.28125}
IINJ17_D1 = 0.328125
WRPC1_L29_D2 = 0.203125
PC_VAR29 = [0.22551806089162557,
            0.09090308053237295,
            0.07110304798769929,
            0.04049961556265634]
RETR14_V1 = {'17': 0.05952380952380952,
             '29': 0.03571428571428571,
             '38': 0.05952380952380952}
CO36_Z = 6.853524885985136
# ---------------- frozen 3137 anchors --
EXP_V37 = ('a_3136_ok|mode_gap_all|'
           'po_dose_monotone|'
           'k0_negative_confirmed|'
           'k0_alone_active|co36_l17_flat|'
           'coord_dose_monotone|'
           'l35_union_zero_dose_robust|'
           'coverage_full')
RES37_SHA = '4f8055bb'
G37 = {'g_co36_d1': 0.140625,
       'g_co36_d2': 0.2265625,
       'g_co50_d1': 0.421875}
DELTA_L17 = 0.637683315669971
RES36_SHA = '3903af46'
# dvec sha8 (3135 npz, fp32)
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
DVEC_MEDNORM = {17: 4.253485202789307,
                29: 31.643726348876953,
                33: 44.65449905395504,
                38: 100.42410278320312}
LEDGER_N = 277

# ---------------- seal ------------------
SEAL = {
    'phase': 3141,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_T': N_T,
        'DIRS': list(DIRS),
        'ALL_L': ALL_L,
        'KEY_L': list(KEY_L),
        'CL': list(CL),
        'CL7': list(CL7),
        'STEP1': STEP1,
        'C_DOSES': list(C_DOSES),
        'SCAN_N': SCAN_N,
        'GEN_BATCH': GEN_BATCH,
        'D_DOSES': list(D_DOSES),
        'E_DOSES': list(E_DOSES),
        'V2_GAIN': V2_GAIN,
        'XMAT_RETR_MIN': XMAT_RETR_MIN,
        'WI_GAIN': WI_GAIN,
        'JOINT_TOL': JOINT_TOL,
        'BLOCK_TOL': BLOCK_TOL,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res14_verdict': EXP_V14,
        'res14_sha8': RES14_SHA,
        'xphase14': XPHASE14,
        'n_repro14': N_REPRO14,
        'cinj_d2_anchors': CINJ_D2,
        'iinj17_d1': IINJ17_D1,
        'wrpc1_l29_d2': WRPC1_L29_D2,
        'pc_var29': PC_VAR29,
        'retr14_v1': RETR14_V1,
        'co36_z': CO36_Z,
        'res37_verdict': EXP_V37,
        'res37_sha8': RES37_SHA,
        'g37': G37,
        'delta_l17': DELTA_L17,
        'res36_sha8': RES36_SHA,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3140 closeout + '
               'FINGERPRINT_PARADIGM_PLAN '
               'Omega-P139: (1) cinj step1 '
               'refinement CL7 x doses 1/2/4 '
               '+ allstep d2 controls with 4 '
               'bit anchors (3140) + iinj17 '
               'bit anchor; (2) WR-PC1 x '
               'dvec29 joint at L29, pc1 d2 '
               'bit anchor 0.203125, '
               'superposition law; (3) co36 '
               'energy-enrichment high25/low25 '
               'causal split at L17 (3137 G '
               'semantics, A1 rows) + 3 bit '
               'anchors (co36 d1/d2, co50 '
               'd1); (4) write-induced '
               'retrieval 84 pairs V1/V2 '
               'fwd vs post-gen states, V1 '
               'fwd bit-replicates 3140 '
               'retr_v1. Bank reused 3138. '
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
z35 = np.load(D35 + r'\p133_readout.npz',
              allow_pickle=False)
dvec = {}
for l in KEY_L:
    a = z35['dvec%d' % l].astype(np.float32)
    got = hashlib.sha256(
        a.tobytes()).hexdigest()[:8]
    assert got == DVEC_SHA[l], (l, got)
    assert a.shape == (NP, 4096)
    dvec[l] = a
    medn = float(np.median(
        np.linalg.norm(a, axis=1)))
    drift = abs(medn - DVEC_MEDNORM[l])
    log('frozen dvec L%02d sha8 %s '
        'med||d||=%.4f (drift %.2e)'
        % (l, DVEC_SHA[l], medn, drift))
    assert drift < 1e-3, drift
res14 = json.load(io.open(
    D40 + r'\result.json', encoding='utf-8'))
assert res14['smoke'] is False
res37 = json.load(io.open(
    D37 + r'\result.json', encoding='utf-8'))
assert res37['smoke'] is False
res36 = json.load(io.open(
    D36 + r'\result.json', encoding='utf-8'))
assert res36['smoke'] is False
z136 = np.load(D36 + r'\p134_readout.npz',
               allow_pickle=False)
CO_SETS = {}
for cn in ('co36', 'co50', 'union'):
    _c = np.asarray(z136[cn]).astype(np.int64)
    assert _c.ndim == 1, (cn, _c.shape)
    assert _c.min() >= 0 and _c.max() < 4096
    CO_SETS[cn] = _c
    log('co set %s: len=%d range=[%d,%d]'
        % (cn, len(_c), int(_c.min()),
           int(_c.max())))
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
assert len(led['measurements']) == LEDGER_N
log('frozen inputs ok (26/35/36/37/40 + '
    'ledger n=%d)' % LEDGER_N)

# ================================================================
# PART A: 3140 + 3137 link asserts
# ================================================================
log('== PART A: link asserts ==')
V14 = res14['verdict']
assert V14 == EXP_V14, V14
raw14 = io.open(
    D40 + r'\result.json', 'rb').read()
sha14 = hashlib.sha256(raw14).hexdigest()[:8]
assert sha14 == RES14_SHA, sha14
pc14 = res14['part_c']
pe14 = res14['part_e']
pf14 = res14['part_f']
assert abs(float(pc14['xphase'])
           - XPHASE14) < 1e-12
assert int(pc14['n_repro_match']) \
    == N_REPRO14
repro14 = pc14['repro_3139']
assert all(v['match']
           for v in repro14.values())
for l, v in CINJ_D2.items():
    got = float(pc14['cinj_d2'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
assert abs(float(pc14['iinj_d1']['17'])
           - IINJ17_D1) < 1e-9
assert abs(float(
    pe14['wr_trials']['wrpc1_l29_d2.0'])
    - WRPC1_L29_D2) < 1e-9
for a, b in zip(PC_VAR29,
                pe14['pc_var']['29']):
    assert abs(a - float(b)) < 1e-12
for l, v in RETR14_V1.items():
    got = float(pf14['retr_v1'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
assert pf14['v1_repro_3139'] is True
assert abs(float(
    res14['part_d']['co_z']['co36']['z'])
    - CO36_Z) < 1e-9
V37 = res37['verdict']
assert V37 == EXP_V37, V37
raw37 = io.open(
    D37 + r'\result.json', 'rb').read()
sha37 = hashlib.sha256(raw37).hexdigest()[:8]
assert sha37 == RES37_SHA, sha37
g37res = res37['part_g']['g_res']
for tn, v in G37.items():
    got = float(g37res[tn]['chg'])
    assert abs(got - v) < 1e-9, (tn, got, v)
raw36 = io.open(
    D36 + r'\result.json', 'rb').read()
sha36 = hashlib.sha256(raw36).hexdigest()[:8]
assert sha36 == RES36_SHA, sha36
log('A hard asserts ok (3140 sha8 %s with '
    'xphase/9-repro/cinj-d2/iinj17/wrpc1/'
    'pcvar/retr/co36z anchors; 3137 sha8 '
    '%s with 3 G-trial anchors; 3136 sha8 '
    '%s)' % (sha14, sha37, sha36))

# ================================================================
# materials factory (3140 lineage)
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
            (s2, o2) = pick_replacement(
                pk, dc)
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
log('gen-prefix ready (%d ids)'
    % len(PREFIX_IDS))


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
    inj_vec: per-layer transplants
    (il, dvec_batch (n,HID) fp32, scale,
    mode). mode: 'allstep' = every
    forward; int s = prompt forward +
    decode step s (3135 semantics)."""
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


def capture_states(rows_all, cap_layers):
    """Last-position states at cap_layers.
    SINGLE-SAMPLE loop (batch=1, zero
    padding) replicating z26/3140
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
# PART B: bank reuse (3138 full shards)
# ================================================================
log('== PART B: bank reuse from 3138 ==')
H_ld = {}
H_sc = {}
for dc in DIRS:
    H_ld[dc] = {}
    H_sc[dc] = {}
    for t in range(N_T):
        fp = os.path.join(
            D38, 'bank_%s_T%d.npz'
            % (dc, t))
        assert os.path.exists(fp), fp
        zt = np.load(fp,
                     allow_pickle=False)
        _a = zt['states'].astype(np.float32)
        assert np.isfinite(_a).all(), \
            'bank shard non-finite'
        assert _a.shape == (NP, 40, HIDG)
        H_ld[dc][t] = _a
        H_sc[dc][t] = zt['scale'] \
            .astype(np.float32)
log('bank loaded (8 shards fp16, reused)')

# ================================================================
# PART C0: session baselines P+A1 + xphase
# ================================================================
log('== PART C0: baselines P+A1 ==')
CKM = {'smoke': SMOKE, 'scan_n': SCAN_N}
if CK['meta'] and CK['meta'] != CKM:
    log('CKPT meta mismatch -> discard')
    CK = {'done': [], 'data': {},
          'meta': {}}
CK['meta'] = CKM
rows_scan = [pids_P_all[j]
             for j in range(SCAN_N)]
_EB = CK['data'].get('e_base_P')
if _EB is not None:
    gen_base_P = _EB['gens']
    xphase_P = _EB['xphase']
    log('base P RESUMED (xphase=%.4f)'
        % xphase_P)
else:
    gen_base_P = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        gen_base_P.extend(gen_batch_g2(
            rows_scan[b0:b0 + GEN_BATCH]))
    _xp = [pad12(gen_base_P[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_P'][j]])
           for j in range(SCAN_N)]
    xphase_P = float(np.mean(_xp))
    log('xphase P: session-base vs z26 '
        'bit-match %.4f (%d/%d)'
        % (xphase_P, int(np.sum(_xp)),
           SCAN_N))
    ck_save('e_base_P', {
        'gens': gen_base_P,
        'xphase': xphase_P})
base12_P = [pad12(gen_base_P[j])
            for j in range(SCAN_N)]
rows_A1 = [pids_A1_all[j]
           for j in range(SCAN_N)]
_EB1 = CK['data'].get('e_base_A1')
if _EB1 is not None:
    gen_base_A1 = _EB1['gens']
    xphase_A1 = _EB1['xphase']
    log('base A1 RESUMED (xphase=%.4f)'
        % xphase_A1)
else:
    gen_base_A1 = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        gen_base_A1.extend(gen_batch_g2(
            rows_A1[b0:b0 + GEN_BATCH]))
    _xa = [pad12(gen_base_A1[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_A1'][j]])
           for j in range(SCAN_N)]
    xphase_A1 = float(np.mean(_xa))
    log('xphase A1: session-base vs z26 '
        'bit-match %.4f (%d/%d)'
        % (xphase_A1, int(np.sum(_xa)),
           SCAN_N))
    ck_save('e_base_A1', {
        'gens': gen_base_A1,
        'xphase': xphase_A1})
base12_A1 = [pad12(gen_base_A1[j])
             for j in range(SCAN_N)]
xphase_ok = (xphase_P == 1.0) \
    and (xphase_A1 == 1.0)

# ================================================================
# PART C1: bank decomposition at CL
# ================================================================
log('== PART C1: decomposition ==')
IDEINT_P = {}
CVEC = {}
PC = {}
WRPCA = {}
for l in CL:
    X = np.zeros((2, N_T, NP, HIDG),
                 dtype=np.float32)
    for di, dc in enumerate(DIRS):
        for t in range(N_T):
            X[di, t] = \
                H_ld[dc][t][:, l, :] \
                * H_sc[dc][t][l]
    B = X.mean(axis=(0, 1, 2))
    Ideint = X.mean(axis=1) - B
    C = X.mean(axis=(0, 2)) - B
    IDEINT_P[l] = Ideint[0].copy()
    CVEC[l] = C.copy()
    if l in (26, 29):
        WR = X - B[None, None, None, :] \
            - Ideint[:, None, :, :] \
            - C[None, :, None, :]
        WRm = WR.reshape(-1, HIDG)
        mu = WRm.mean(axis=0)[None, :]
        WRc = WRm - mu
        _U, _S, _Vt = np.linalg.svd(
            WRc.astype(np.float64),
            full_matrices=False)
        _var = (_S[:4] ** 2
                / np.sum(_S ** 2)).tolist()
        PC[l] = {
            'V': _Vt[:4].astype(np.float32),
            'var': _var,
            'mnorm': float(np.median(
                np.linalg.norm(
                    WRc, axis=1)))}
        WRPCA[l] = WR.reshape(
            2, N_T, NP, HIDG)
        log('C1 L%02d WR PC var %s mnorm '
            '%.3f' % (l,
                      json.dumps([round(v, 4)
                                  for v in
                                  _var]),
                      PC[l]['mnorm']))
        for a, b in zip(_var, PC_VAR29) \
                if l == 29 else []:
            assert abs(a - b) < 1e-9
        del WR, WRm, mu, WRc, _U, _S, _Vt
    else:
        WRPCA[l] = None
    del X, B, Ideint, C
    gc.collect()
log('C1 done (%d layers)' % len(CL))

# ================================================================
# PART C2: cinj step1 refinement
# ================================================================
log('== PART C2: cinj step1 trials ==')
E_res = {}


def _run_vec_trial(tname, il, dv_batch,
                   scale, mode, rows, base12):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        return
    gen_l = []
    for b0 in range(0, len(rows),
                    GEN_BATCH):
        batch = rows[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il,
                      dv_batch[b0:b0
                               + len(batch)],
                      scale, mode)]))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname],
                    'fstep': fstep})


# step1: CL7 x doses {1,2,4}
for l in CL7:
    _cv = CVEC[l][2][None, :]
    _dv_full = np.repeat(
        _cv, SCAN_N, axis=0) \
        .astype(np.float32)
    for dose in C_DOSES:
        tname = 'c_step1_l%02d_d%.1f' % (
            l, dose)
        _run_vec_trial(tname, l, _dv_full,
                       dose, STEP1,
                       rows_scan, base12_P)
# allstep d2 controls (bit anchors from
# 3140 at L23/25/26/27)
for l in CL7:
    _cv = CVEC[l][2][None, :]
    _dv_full = np.repeat(
        _cv, SCAN_N, axis=0) \
        .astype(np.float32)
    tname = 'c_all_l%02d_d2.0' % l
    _run_vec_trial(tname, l, _dv_full,
                   2.0, 'allstep',
                   rows_scan, base12_P)
# iinj L17 d1 allstep bit anchor (3140)
_dv_iinj = np.roll(
    IDEINT_P[17], -1, axis=0)[:SCAN_N] \
    .astype(np.float32)
_run_vec_trial('c_iinj17_d1.0', 17,
               _dv_iinj, 1.0, 'allstep',
               rows_scan, base12_P)

# bit-repro checks vs 3140
n_bit = 0
bit_anchors = {}
if not SMOKE:
    for l, v in CINJ_D2.items():
        got = E_res['c_all_l%02d_d2.0'
                    % int(l)]['chg']
        m = abs(got - v) < 1e-9
        n_bit += int(m)
        bit_anchors['c_all_l%s_d2.0' % l] = {
            'got': got, 'want': v,
            'match': bool(m)}
    got = E_res['c_iinj17_d1.0']['chg']
    m = abs(got - IINJ17_D1) < 1e-9
    n_bit += int(m)
    bit_anchors['c_iinj17_d1.0'] = {
        'got': got, 'want': IINJ17_D1,
        'match': bool(m)}
    log('C2 3140 repro: %d/%d bit-match'
        % (n_bit, len(CINJ_D2) + 1))
else:
    log('C2 3140 repro SKIPPED (smoke)')

# step1 spectrum verdicts
step1_d = {dose: {l: E_res['c_step1_l%02d_'
                           'd%.1f' % (l, dose)]
                  ['chg'] for l in CL7}
           for dose in C_DOSES}
all_d2 = {l: E_res['c_all_l%02d_d2.0'
                   % l]['chg']
          for l in CL7}
peak_step1_d2 = max(step1_d[2.0],
                    key=step1_d[2.0].get)
peak_all_d2 = max(all_d2,
                  key=all_d2.get)
s1_max = step1_d[2.0][peak_step1_d2]
al_max = all_d2[peak_all_d2]
step1_beats = s1_max > al_max
mono_rows = []
for l in CL7:
    cds = [step1_d[d][l]
           for d in C_DOSES]
    mono_rows.append(all(
        cds[i + 1] >= cds[i] - 1e-12
        for i in range(len(cds) - 1)))
s1_mono = float(np.mean(mono_rows))
med_s1 = float(np.median(
    list(step1_d[2.0].values())))
med_al = float(np.median(list(all_d2.values())))
sharp_s1 = s1_max / max(med_s1, 1e-12)
sharp_al = al_max / max(med_al, 1e-12)
sharpened = sharp_s1 > sharp_al
log('C2-SOFT: step1 d2 %s; allstep d2 %s; '
    'peaks step1 L%02d (%.4f) vs all L%02d '
    '(%.4f); mono %.2f; sharp %.2f vs '
    '%.2f (sharpened %s)'
    % (json.dumps({str(l): round(v, 4)
                   for l, v in
                   step1_d[2.0].items()}),
       json.dumps({str(l): round(v, 4)
                   for l, v in
                   all_d2.items()}),
       peak_step1_d2, s1_max,
       peak_all_d2, al_max, s1_mono,
       sharp_s1, sharp_al, sharpened))

# ================================================================
# PART D: WR-PC1 x dvec29 joint injection
# ================================================================
log('== PART D: wr x dvec joint ==')
LJ = 29
_pc = PC[LJ]
v1 = _pc['V'][0]
dv_pc1 = {dose: np.tile(
    (v1 * _pc['mnorm'] * dose)[None, :],
    (SCAN_N, 1)).astype(np.float32)
    for dose in D_DOSES}
dv_dvec = {dose: (dvec[LJ][:SCAN_N]
                  * dose).astype(np.float32)
           for dose in D_DOSES}
# pc1 d2 bit anchor (3140 wrpc1_l29_d2.0)
_run_vec_trial('d_wrpc1_l29_d2.0', LJ,
               dv_pc1[2.0], 1.0, 'allstep',
               rows_scan, base12_P)
for dose in D_DOSES:
    _run_vec_trial('d_dvec29_l29_d%.1f'
                   % dose, LJ,
                   dv_dvec[dose], 1.0,
                   'allstep', rows_scan,
                   base12_P)
    # joint: pc1 first, then dvec (fixed
    # hook order; addition commutes but
    # float order fixed for repro)
    _K = CK['data'].get('d_joint_l29_d%.1f'
                        % dose)
    if _K is not None:
        E_res['d_joint_l29_d%.1f' % dose] = \
            _K['res']
        log('joint d%.1f RESUMED chg=%.4f'
            % (dose, _K['res']['chg']))
        continue
    gen_l = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[
                (LJ,
                 dv_pc1[dose][b0:b0
                              + len(batch)],
                 1.0, 'allstep'),
                (LJ,
                 dv_dvec[dose][b0:b0
                               + len(batch)],
                 1.0, 'allstep')]))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base12_P)
    E_res['d_joint_l29_d%.1f' % dose] = {
        'chg': chg_l, 'first': int(first_l)}
    log('d_joint_l29_d%.1f: chg=%.4f '
        'first=%d' % (dose, chg_l, first_l))
    ck_save('d_joint_l29_d%.1f' % dose,
            {'res': E_res[
                'd_joint_l29_d%.1f' % dose],
             'fstep': fstep})

if not SMOKE:
    got = E_res['d_wrpc1_l29_d2.0']['chg']
    wr_bit = abs(got - WRPC1_L29_D2) < 1e-9
    log('D pc1 d2 bit anchor: got %.10f '
        'want %.10f match %s'
        % (got, WRPC1_L29_D2, wr_bit))
else:
    wr_bit = None
a_d2 = E_res['d_wrpc1_l29_d2.0']['chg']
b_d2 = E_res['d_dvec29_l29_d2.0']['chg']
j_d2 = E_res['d_joint_l29_d2.0']['chg']
p_ind = 1.0 - (1.0 - a_d2) * (1.0 - b_d2)
p_add = min(a_d2 + b_d2, 1.0)
if j_d2 >= p_ind + JOINT_TOL:
    law_d2 = 'joint_superadditive'
elif abs(j_d2 - p_add) <= JOINT_TOL:
    law_d2 = 'joint_additive'
elif j_d2 <= max(a_d2, b_d2) + BLOCK_TOL:
    law_d2 = 'joint_blocking'
else:
    law_d2 = 'joint_subadditive'
a_d1 = E_res['d_wrpc1_l29_d1.0']['chg'] \
    if 'd_wrpc1_l29_d1.0' in E_res else None
log('D-SOFT: d2 pc1 %.4f + dvec %.4f -> '
    'joint %.4f (p_ind %.4f p_add %.4f) '
    '-> %s'
    % (a_d2, b_d2, j_d2, p_ind, p_add,
       law_d2))
dvec29_active = b_d2 >= 0.05

# ================================================================
# PART E: co36 energy-enrichment split
# ================================================================
log('== PART E: co36 enrichment split ==')
I17 = IDEINT_P[17]
fr = I17 ** 2 / np.maximum(
    np.sum(I17 ** 2, axis=1,
           keepdims=True), 1e-12)
co36 = CO_SETS['co36']
assert len(co36) == 50
score = np.mean(fr[:, co36], axis=0)
order_e = co36[np.argsort(-score)]
high25 = order_e[:25]
low25 = order_e[25:]
log('E co36 enrichment order: high25=%s '
    'low25=%s (score hi %.6f.. lo %.6f)'
    % (high25[:5].tolist(),
       low25[:5].tolist(),
       float(score[np.argsort(-score)[0]]),
       float(score[np.argsort(-score)[-1]])))
E_TRIALS = [('e_high_d1', high25, 1.0),
            ('e_low_d1', low25, 1.0),
            ('e_high_d2', high25, 2.0),
            ('e_low_d2', low25, 2.0),
            ('e_co36_d1', co36, 1.0),
            ('e_co36_d2', co36, 2.0),
            ('e_co50_d1', CO_SETS['co50'],
             1.0)]
for (tname, coords, ds) in E_TRIALS:
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        continue
    gen_l = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        batch = rows_A1[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=[(17, coords,
                  ds * DELTA_L17, -1, 0)]))
    chg_l, first_l, _fs, _cum = \
        trial_metrics(gen_l, base12_A1)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})

e_bit = 0
if not SMOKE:
    for tn, v in (('e_co36_d1',
                   G37['g_co36_d1']),
                  ('e_co36_d2',
                   G37['g_co36_d2']),
                  ('e_co50_d1',
                   G37['g_co50_d1'])):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        e_bit += int(m)
        log('E bit anchor %s: got %.10f '
            'want %.10f match %s'
            % (tn, got, v, m))
    log('E 3137 repro: %d/3 bit-match'
        % e_bit)
else:
    log('E 3137 repro SKIPPED (smoke)')
hi_d2 = E_res['e_high_d2']['chg']
lo_d2 = E_res['e_low_d2']['chg']
hi_d1 = E_res['e_high_d1']['chg']
lo_d1 = E_res['e_low_d1']['chg']
ratio2 = hi_d2 / max(lo_d2, 1e-12)
if hi_d2 >= max(1.5 * lo_d2, lo_d2 + 0.05):
    enrich_tag = 'co_enrich_causal_heavy'
elif lo_d2 >= max(1.5 * hi_d2,
                  hi_d2 + 0.05):
    enrich_tag = 'co_enrich_causal_reversed'
elif abs(hi_d2 - lo_d2) <= 0.02:
    enrich_tag = 'co_enrich_causal_flat'
else:
    enrich_tag = 'co_enrich_causal_mixed'
log('E-SOFT: d1 hi %.4f lo %.4f | d2 hi '
    '%.4f lo %.4f (ratio %.2f) -> %s; '
    'vs 3137 rank-order top25 0.1328/'
    'bot25 0.1172 (flat)'
    % (hi_d1, lo_d1, hi_d2, lo_d2,
       ratio2, enrich_tag))

# ================================================================
# PART F: write-induced retrieval
# ================================================================
log('== PART F: write-induced retrieval ==')
used_pairs = set(pks)
ents_n = len(ents_all)
new_pairs = []
for s in range(ents_n):
    for o in range(ents_n):
        if s == o:
            continue
        pk = '%d_%d' % (s, o)
        if pk not in used_pairs:
            new_pairs.append(pk)
new_pairs = sorted(new_pairs)
if SMOKE:
    new_pairs = new_pairs[:6]
XMAT_N = len(new_pairs)
log('F new pairs: %d unseen (s,o) pairs; '
    'ents_n=%d (chance-s %.4f)'
    % (XMAT_N, ents_n, 1.0 / ents_n))


def build_prompt_new(s, o, lrel, qrel):
    """V1 (3139-identical): fixed-seed
    distractor lines, query first
    candidate, k=0."""
    rng3 = _rnd.Random(zlib.crc32(
        ('xmat|%d_%d_%d_%d'
         % (s, o, lrel, qrel))
        .encode('ascii')))
    lines = [(s, lrel, o)]
    seen = {(s, lrel, o)}
    while len(lines) < 8:
        a = rng3.randrange(ents_n)
        b = rng3.randrange(ents_n)
        rr = rng3.randrange(len(PREDS_all))
        if a == b or (a, rr, b) in seen:
            continue
        seen.add((a, rr, b))
        lines.append((a, rr, b))
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents_all[ls], PREDS_all[lr],
            ents_all[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents_all[s], PREDS_all[qrel],
                ents_all[o]))
    return text


def build_prompt_v2(s, o, lrel, qrel):
    """V2: same-distractor-family prompt
    (3140 semantics)."""
    o0 = None
    for pk in pks:
        (s2, o2) = (int(v) for v in
                    pk.split('_'))
        if s2 == s and o2 != o:
            o0 = o2
            break
    assert o0 is not None, (s, o)
    D0 = [tuple(d) for d in
          mat5['distractors']
          ['%d_%d' % (s, o0)]]
    Df = [d for d in D0
          if o not in (d[0], d[2])]
    lines = [(s, lrel, o)] + Df[:7]
    _grng = _rnd.Random(zlib.crc32(
        ('v2fill|%d_%d' % (s, o))
        .encode('ascii')))
    while len(lines) < 8:
        a = _grng.randrange(ents_n)
        b = _grng.randrange(ents_n)
        rr = _grng.randrange(
            len(PREDS_all))
        if a == b or (a, rr, b) in lines:
            continue
        if o in (a, b):
            continue
        lines.append((a, rr, b))
    k0 = int(mat5['kline']
             ['%d_%d' % (s, o0)]) % 8
    rng4 = _rnd.Random(zlib.crc32(
        ('v2|%d_%d' % (s, o))
        .encode('ascii')))
    order = list(range(8))
    rng4.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k0] = \
        lines[k0], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents_all[ls], PREDS_all[lr],
            ents_all[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents_all[s], PREDS_all[qrel],
                ents_all[o]))
    return text


rows_v = {v: [] for v in ('V1', 'V2')}
for pk in new_pairs:
    (s, o) = (int(v)
              for v in pk.split('_'))
    used_rels = {p2r.get(pk)}
    if pk in frel:
        used_rels.update(frel[pk])
    rp = None
    for rr in range(len(PREDS_all)):
        if rr not in used_rels:
            rp = rr
            break
    assert rp is not None, pk
    r = rp
    for v in ('V1', 'V2'):
        if v == 'V1':
            txt = build_prompt_new(
                s, o, r, r)
        else:
            txt = build_prompt_v2(
                s, o, r, r)
        rows_v[v].append(list(tok_g(
            txt, add_special_tokens=False)
            ['input_ids']))
RL = (17, 29, 38)
_FK = CK['data'].get('fwd_cap')
if _FK is not None:
    vst = {int(k): v for k, v in
           _FK['states'].items()}
    log('F fwd capture RESUMED')
else:
    _cap = capture_states(
        rows_v['V1'] + rows_v['V2'], RL)
    _half = len(rows_v['V1'])
    vst = {}
    for l in RL:
        vst[l] = {'V1': _cap[l][:_half],
                  'V2': _cap[l][_half:]}
    ck_save('fwd_cap', {
        'states': {
            str(l): {'V1': vst[l]['V1'],
                     'V2': vst[l]['V2']}
            for l in RL}})


def retr_from_states(q, l):
    bank = IDEINT_P[l]
    bank_n = bank / (np.linalg.norm(
        bank, axis=1, keepdims=True)
        + 1e-12)
    q_n = q / (np.linalg.norm(
        q, axis=1, keepdims=True)
        + 1e-12)
    sim = q_n @ bank_n.T
    top1 = np.argmax(sim, axis=1)
    hit_s = 0
    for i, pk in enumerate(new_pairs):
        s = int(pk.split('_')[0])
        s2 = int(pks[int(top1[i])]
                 .split('_')[0])
        hit_s += int(s2 == s)
    return hit_s / XMAT_N


RETR = {}
for v in ('V1', 'V2'):
    retr_s = {}
    for l in RL:
        retr_s[l] = retr_from_states(
            vst[l][v], l)
        log('F %s fwd retr L%02d: same-s '
            '%.4f' % (v, l, retr_s[l]))
    RETR[v] = retr_s
if not SMOKE:
    v1_fwd_bit = all(
        abs(RETR['V1'][l]
            - RETR14_V1[str(l)]) < 1e-9
        for l in RL)
    log('F V1 fwd bit anchor vs 3140 '
        'retr_v1: %s' % v1_fwd_bit)
else:
    v1_fwd_bit = None

# write-induced: generate then capture
_GK = CK['data'].get('gen_cap')
if _GK is not None:
    gst = {int(k): v for k, v in
           _GK['states'].items()}
    log('F gen capture RESUMED')
else:
    gst = {}
    for v in ('V1', 'V2'):
        gen_w = []
        for b0 in range(0, XMAT_N,
                        GEN_BATCH):
            gen_w.extend(gen_batch_g2(
                rows_v[v][b0:b0 + GEN_BATCH]))
        full_rows = [rows_v[v][i] + gen_w[i]
                     for i in range(XMAT_N)]
        _capg = capture_states(
            full_rows, RL)
        for l in RL:
            gst.setdefault(l, {})[v] = \
                _capg[l]
    ck_save('gen_cap', {
        'states': {
            str(l): {v: gst[l][v]
                     for v in ('V1', 'V2')}
            for l in RL}})
RETR_G = {}
for v in ('V1', 'V2'):
    retr_sg = {}
    for l in RL:
        retr_sg[l] = retr_from_states(
            gst[l][v], l)
        log('F %s gen retr L%02d: same-s '
            '%.4f' % (v, l, retr_sg[l]))
    RETR_G[v] = retr_sg
wi_gains = {}
for v in ('V1', 'V2'):
    for l in RL:
        wi_gains['%s_L%02d' % (v, l)] = \
            RETR_G[v][l] - RETR[v][l]
wi_med_gain = float(np.median(
    list(wi_gains.values())))
wi_gain_ok = wi_med_gain >= WI_GAIN
all_retr = [RETR_G[v][l]
            for v in ('V1', 'V2')
            for l in RL]
gen_best = float(max(all_retr))
gen_reach = gen_best >= XMAT_RETR_MIN
log('F-SOFT: write-induced gains %s '
    '(med %+.4f, >=0.05 %s); gen best '
    '%.4f (>=0.30 %s); fwd V1 med %.4f '
    'V2 med %.4f'
    % (json.dumps({k: round(v, 4)
                   for k, v in
                   wi_gains.items()}),
       wi_med_gain, wi_gain_ok, gen_best,
       gen_reach,
       float(np.median(list(
           RETR['V1'].values()))),
       float(np.median(list(
           RETR['V2'].values())))))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3140_ok']
if not SMOKE:
    n_bit_all = n_bit + int(wr_bit) + e_bit
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == len(CINJ_D2)
                + 1 + 1 + 3
                else 'repro_bit_drift')
else:
    tags.append('repro_smoke_skipped')
tags.append('step1_peak_l%02d'
            % peak_step1_d2)
tags.append('step1_beats_all'
            if step1_beats
            else 'step1_below_all')
tags.append('step1_dose_mono'
            if s1_mono >= 0.6
            else 'step1_dose_flat')
tags.append('peak_sharpened'
            if sharpened
            else 'peak_not_sharpened')
tags.append('dvec29_active'
            if dvec29_active
            else 'dvec29_quiet')
tags.append(law_d2)
tags.append(enrich_tag)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
if not SMOKE:
    tags.append('v1fwd_bit_ok'
                if v1_fwd_bit
                else 'v1fwd_drift')
else:
    tags.append('v1fwd_smoke_skipped')
tags.append('writeinduce_gain'
            if wi_gain_ok
            else 'writeinduce_flat')
tags.append('gen_retr_reach' if gen_reach
            else 'gen_retr_below')
tags.append('coverage_full')
verdict = '|'.join(tags)
runtime = time.time() - T0
log('VERDICT: %s' % verdict)
log('DONE (%.1fs)' % runtime)

result = {
    'phase': 3141,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        json.dumps(SEAL, sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8],
    'part_c': {
        'xphase_P': xphase_P,
        'xphase_A1': xphase_A1,
        'step1': {str(dose): {
            str(l): step1_d[dose][l]
            for l in CL7}
            for dose in C_DOSES},
        'allstep_d2': {str(l): all_d2[l]
                       for l in CL7},
        'iinj17_d1': E_res[
            'c_iinj17_d1.0']['chg'],
        'peak_step1_d2': int(peak_step1_d2),
        'peak_all_d2': int(peak_all_d2),
        's1_max': s1_max,
        'al_max': al_max,
        's1_mono_rate': s1_mono,
        'sharp_s1': sharp_s1,
        'sharp_all': sharp_al,
        'bit_anchors_3140': bit_anchors},
    'part_d': {
        'pc1_d2': a_d2,
        'dvec29_d2': b_d2,
        'joint_d2': j_d2,
        'p_ind': p_ind,
        'p_add': p_add,
        'law_d2': law_d2,
        'dvec29_d1': E_res[
            'd_dvec29_l29_d1.0']['chg'],
        'joint_d1': E_res[
            'd_joint_l29_d1.0']['chg'],
        'wr_bit_3140': wr_bit},
    'part_e': {
        'high25': high25.tolist(),
        'low25': low25.tolist(),
        'hi_d1': hi_d1, 'lo_d1': lo_d1,
        'hi_d2': hi_d2, 'lo_d2': lo_d2,
        'ratio_d2': ratio2,
        'tag': enrich_tag,
        'co36_d1': E_res['e_co36_d1']['chg'],
        'co36_d2': E_res['e_co36_d2']['chg'],
        'co50_d1': E_res['e_co50_d1']['chg'],
        'e_bit_3137': e_bit},
    'part_f': {
        'n_pairs': XMAT_N,
        'retr_fwd_v1': {str(l): RETR['V1'][l]
                        for l in RL},
        'retr_fwd_v2': {str(l): RETR['V2'][l]
                        for l in RL},
        'retr_gen_v1': {str(l): RETR_G['V1'][l]
                        for l in RL},
        'retr_gen_v2': {str(l): RETR_G['V2'][l]
                        for l in RL},
        'wi_gains': wi_gains,
        'wi_med_gain': wi_med_gain,
        'gen_best': gen_best,
        'v1_fwd_bit_3140': v1_fwd_bit}}
with io.open(os.path.join(
        OUT, 'result.json'), 'w',
        encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False, indent=1)
np.savez(os.path.join(
    OUT, 'p139_readout.npz'),
    c_step1_d1=np.array(
        [step1_d[1.0][l] for l in CL7]),
    c_step1_d2=np.array(
        [step1_d[2.0][l] for l in CL7]),
    c_step1_d4=np.array(
        [step1_d[4.0][l] for l in CL7]),
    c_all_d2=np.array(
        [all_d2[l] for l in CL7]),
    cl7=np.array(CL7),
    d_trial=np.array(
        [a_d2, b_d2, j_d2,
         E_res['d_dvec29_l29_d1.0']['chg'],
         E_res['d_joint_l29_d1.0']['chg']]),
    e_highlow=np.array(
        [hi_d1, lo_d1, hi_d2, lo_d2]),
    e_co36=np.array(
        [E_res['e_co36_d1']['chg'],
         E_res['e_co36_d2']['chg']]),
    retr_fwd=np.array(
        [RETR['V1'][l] for l in RL]
        + [RETR['V2'][l] for l in RL]),
    retr_gen=np.array(
        [RETR_G['V1'][l] for l in RL]
        + [RETR_G['V2'][l] for l in RL]))
log('dumps done: result.json + seal + '
    'p139_readout.npz')
if not SMOKE:
    if os.path.exists(CKPTF):
        os.remove(CKPTF)
        log('ckpt cleared')
log('PHASE 3141 COMPLETE')
