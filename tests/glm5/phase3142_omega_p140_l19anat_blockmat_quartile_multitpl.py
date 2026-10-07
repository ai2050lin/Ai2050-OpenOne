# -*- coding: utf-8 -*-
"""Phase 3142 (Omega-P140): L19 true-peak
anatomy (new dvec19 capture + step1 fine
scan L17-21) + blocking mechanism
localization (dose matrix + cross-layer +
order reversal) + co36 enrichment
quartile x low-dose window + in-context
multi-template write-induced retrieval.

Preregistered in 3141 closeout (aligned
with FINGERPRINT_PARADIGM_PLAN.md
Omega-P140):
(1) L19 anatomy: dvec19 newly captured
(3135 semantics: P-row base vs swap4
[8,9,13,29] layer-skip state diff, L19,
single-sample) + step1 fine scan
L17-21 x doses {1,2,4} (CVEC[l][2]
template vector, 3141 semantics) +
allstep d2 controls L17-21 + dvec19
injection trials (allstep d1 / step1
d1) -> single-step sensitive zone
boundary;
(2) blocking localization at L29:
WR-PC1 x dvec29 dose matrix
(pc1 d{0.5,1,2} x dvec d{0.5,1,2},
9 joint cells; singles incl. 3141 bit
anchors pc1_d2 0.203125 / dvec29_d2
0.640625 / dvec29_d1 0.25 / joint_d2
0.4765625) + cross-layer joint
(pc1@L29 d2 + dvec33@L33 d1) +
order-reversal joint d2 -> readout
competition vs same-layer state
destruction;
(3) co36 enrichment quartiles:
order_e (3141 semantics) split into
Q1..Q4, injected at L17 (3137 G
semantics: DELTA_L17, sgn -1, mode 0,
A1 rows) x doses {0.5,1.0} + co36 d1/d2
+ co50 d1 bit anchors (3137: 0.140625 /
0.2265625 / 0.421875) -> enrichment->
causal window;
(4) in-context multi-template write:
84 unseen pairs, variants v0..v3
(new rng seeds) each generate 12 tokens,
f3 = quad write [W0+G0..W3+G3]+V1query,
f4 = single write [W2+G2]+V1query,
f2 = V1 gen write (3141 bit anchors
retr_fwd_v1 / retr_gen_v1) ->
write-dose vs template-diversity vs
structural absence.

Bank shards REUSED from 3138 full run.
dvecs frozen from 3135 (sha-anchored).
Session baselines P+A1 with xphase
recording. order_e must equal 3141
high25/low25 exactly (pure-numpy)."""
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
NAME = ('omega_p140_l19anat_blockmat_'
        'quartile_multitpl')
SMOKE = os.environ.get('P3142_SMOKE',
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
D41 = RDIR + r'\phase3141' \
      r'\omega_p139_cinjrefine_wrjoint_' \
      r'coenrich_writeinduce'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3142', NAME)
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
CKPTF = os.path.join(OUT, 'p140_ckpt.pkl')
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
SWAP_L = [8, 9, 13, 29]
D19_L = 19
# decomposition layers: fine scan 17-21,
# cinj anchors 23-28, PC 26/29, E 17,
# retrieval 17/29/38
CL = (17, 18, 19, 20, 21, 23, 24, 25,
      26, 27, 28, 29, 33, 38)
SCAN_L = (17, 18, 19, 20, 21)
STEP1 = 1
C_DOSES = (1.0, 2.0, 4.0)
D_DOSES = (0.5, 1.0, 2.0)
E_DOSES = (0.5, 1.0)
SCAN_N = 4 if SMOKE else 128
GEN_BATCH = 4 if SMOKE else 32
N_NEW = 12
JOINT_TOL = 0.05
BLOCK_TOL = 0.02
REV_TOL = 1e-9
WI_GAIN = 0.05
XMAT_RETR_MIN = 0.30
RNG_SEED = 3142

# ---------------- frozen 3141 anchors --
EXP_V41 = ('a_3140_ok|repro_bit_9|'
           'repro_bit_ok|step1_peak_l19|'
           'step1_below_all|step1_dose_flat|'
           'peak_sharpened|dvec29_active|'
           'joint_blocking|'
           'co_enrich_causal_mixed|'
           'xphase_ok|v1fwd_bit_ok|'
           'writeinduce_flat|gen_retr_below|'
           'coverage_full')
RES41_SHA = 'a93c8892'
SEAL41 = '7ccf7ffc'
XPHASE41 = 1.0
S1_D2_19 = 0.140625
ALL_D2_41 = {'19': 0.4921875,
             '23': 0.2890625,
             '24': 0.3203125,
             '25': 0.2734375,
             '26': 0.2890625,
             '27': 0.28125,
             '28': 0.2578125}
IINJ17_D1 = 0.328125
PC1_D2 = 0.203125
DVEC29_D1 = 0.25
DVEC29_D2 = 0.640625
JOINT_D2 = 0.4765625
HI_D1 = 0.21875
LO_D1 = 0.09375
HI_D2 = 0.203125
LO_D2 = 0.171875
HIGH25_41 = [2319, 83, 2309, 3140, 3099,
             2491, 2097, 999, 2702, 1939,
             1235, 2617, 161, 3426, 1877,
             551, 3301, 2011, 2560, 1008,
             2605, 78, 1357, 3818, 207]
LOW25_41 = [381, 1179, 1004, 1578, 2911,
            1626, 1973, 642, 1284, 2442,
            1306, 2890, 2664, 317, 1757,
            2185, 1991, 3131, 3756, 294,
            1634, 807, 1059, 3499, 3432]
RETR_FWD_V1 = {'17': 0.05952380952380952,
               '29': 0.03571428571428571,
               '38': 0.05952380952380952}
RETR_GEN_V1 = {'17': 0.047619047619047616,
               '29': 0.03571428571428571,
               '38': 0.05952380952380952}
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
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
DVEC_MEDNORM = {17: 4.253485202789307,
                29: 31.643726348876953,
                33: 44.65449905395504,
                38: 100.42410278320312}
LEDGER_N = 278

# ---------------- seal ------------------
SEAL = {
    'phase': 3142,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_T': N_T,
        'DIRS': list(DIRS),
        'ALL_L': ALL_L,
        'KEY_L': list(KEY_L),
        'SWAP_L': SWAP_L,
        'D19_L': D19_L,
        'CL': list(CL),
        'SCAN_L': list(SCAN_L),
        'STEP1': STEP1,
        'C_DOSES': list(C_DOSES),
        'D_DOSES': list(D_DOSES),
        'E_DOSES': list(E_DOSES),
        'SCAN_N': SCAN_N,
        'GEN_BATCH': GEN_BATCH,
        'JOINT_TOL': JOINT_TOL,
        'BLOCK_TOL': BLOCK_TOL,
        'REV_TOL': REV_TOL,
        'WI_GAIN': WI_GAIN,
        'XMAT_RETR_MIN': XMAT_RETR_MIN,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res41_verdict': EXP_V41,
        'res41_sha8': RES41_SHA,
        'seal41': SEAL41,
        'xphase41': XPHASE41,
        's1_d2_19': S1_D2_19,
        'all_d2_41': ALL_D2_41,
        'iinj17_d1': IINJ17_D1,
        'pc1_d2': PC1_D2,
        'dvec29_d1': DVEC29_D1,
        'dvec29_d2': DVEC29_D2,
        'joint_d2': JOINT_D2,
        'hi_d1': HI_D1, 'lo_d1': LO_D1,
        'hi_d2': HI_D2, 'lo_d2': LO_D2,
        'high25_41': HIGH25_41,
        'low25_41': LOW25_41,
        'retr_fwd_v1': RETR_FWD_V1,
        'retr_gen_v1': RETR_GEN_V1,
        'res37_verdict': EXP_V37,
        'res37_sha8': RES37_SHA,
        'g37': G37,
        'delta_l17': DELTA_L17,
        'res36_sha8': RES36_SHA,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3141 closeout + '
               'FINGERPRINT_PARADIGM_PLAN '
               'Omega-P140: (1) dvec19 new '
               'capture (3135 swap4 semantics, '
               'L19) + step1 fine scan L17-21 '
               'x d{1,2,4} + allstep d2 ctrl + '
               'dvec19 inj trials -> sensitive '
               'zone boundary; (2) blocking '
               'matrix pc1 x dvec29 9 cells + '
               'cross-layer dvec33@L33 + '
               'order reversal, 4 bit anchors '
               '(3141 D); (3) co36 quartiles '
               'Q1-Q4 x d{0.5,1.0} at L17 '
               '(3137 G semantics) + 3 bit '
               'anchors; (4) in-context '
               'multi-template write 84 pairs: '
               'f2 V1 gen (3141 bit anchors), '
               'f3 quad-write, f4 single-write '
               '(variant 2). order_e must '
               'equal 3141 high25/low25. Bank '
               'reused 3138. Frozen before '
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
res41 = json.load(io.open(
    D41 + r'\result.json', encoding='utf-8'))
assert res41['smoke'] is False
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
log('frozen inputs ok (26/35/36/37/41 + '
    'ledger n=%d)' % LEDGER_N)

# ================================================================
# PART A: 3141 + 3137 link asserts
# ================================================================
log('== PART A: link asserts ==')
V41 = res41['verdict']
assert V41 == EXP_V41, V41
raw41 = io.open(
    D41 + r'\result.json', 'rb').read()
sha41 = hashlib.sha256(raw41).hexdigest()[:8]
assert sha41 == RES41_SHA, sha41
assert str(res41['seal_sha8']) == SEAL41
pc41 = res41['part_c']
pd41 = res41['part_d']
pe41 = res41['part_e']
pf41 = res41['part_f']
assert abs(float(pc41['xphase_P'])
           - XPHASE41) < 1e-12
assert abs(float(pc41['xphase_A1'])
           - XPHASE41) < 1e-12
ba41 = pc41['bit_anchors_3140']
assert all(v['match']
           for v in ba41.values())
assert abs(float(pc41['step1']['2.0']['19'])
           - S1_D2_19) < 1e-9
for l, v in ALL_D2_41.items():
    got = float(pc41['allstep_d2'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
assert abs(float(pc41['iinj17_d1'])
           - IINJ17_D1) < 1e-9
assert abs(float(pd41['pc1_d2'])
           - PC1_D2) < 1e-9
assert abs(float(pd41['dvec29_d1'])
           - DVEC29_D1) < 1e-9
assert abs(float(pd41['dvec29_d2'])
           - DVEC29_D2) < 1e-9
assert abs(float(pd41['joint_d2'])
           - JOINT_D2) < 1e-9
assert str(pd41['law_d2']) == \
    'joint_blocking'
assert abs(float(pe41['hi_d1'])
           - HI_D1) < 1e-9
assert abs(float(pe41['lo_d1'])
           - LO_D1) < 1e-9
assert abs(float(pe41['hi_d2'])
           - HI_D2) < 1e-9
assert abs(float(pe41['lo_d2'])
           - LO_D2) < 1e-9
assert list(pe41['high25']) == HIGH25_41
assert list(pe41['low25']) == LOW25_41
for l, v in RETR_FWD_V1.items():
    got = float(pf41['retr_fwd_v1'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
for l, v in RETR_GEN_V1.items():
    got = float(pf41['retr_gen_v1'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
assert pf41['v1_fwd_bit_3140'] is True
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
log('A hard asserts ok (3141 sha8 %s seal '
    '%s with xphase/7bit/s1/all/hilo/'
    'high25/retr anchors; 3137 sha8 %s; '
    '3136 sha8 %s)'
    % (sha41, SEAL41, sha37, sha36))

# ================================================================
# materials factory (3141 lineage)
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


def capture_states(rows_all, cap_layers,
                   swap_layers=None):
    """Last-position states at cap_layers.
    SINGLE-SAMPLE loop (batch=1, zero
    padding) replicating z26/3140
    semantics exactly. swap_layers:
    layer-skip hooks (3135 dvec
    semantics)."""
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
        if swap_layers:
            for l in swap_layers:
                lyr = model_g.model.layers[l]

                def _swap(mod, inp, out):
                    o2 = out[0] \
                        if isinstance(
                            out, tuple) \
                        else out
                    i2 = inp[0] \
                        if isinstance(
                            inp, tuple) \
                        else inp
                    o2.copy_(
                        i2.to(o2.dtype))

                hooks.append(
                    lyr.register_forward_hook(
                        _swap))
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
# PART C1: dvec19 capture (3135 semantics)
# ================================================================
log('== PART C1: dvec19 capture ==')
_N19 = 4 if SMOKE else NP
rows_d19 = [pids_P_all[j]
            for j in range(_N19)]
_K19 = CK['data'].get('dvec19')
if _K19 is not None:
    dvec19 = _K19['dvec']
    dvec19_sha = _K19['sha8']
    dvec19_medn = _K19['medn']
    log('dvec19 RESUMED sha8=%s' %
        dvec19_sha)
else:
    base19 = capture_states(
        rows_d19, [D19_L],
        swap_layers=None)[D19_L]
    log('dvec19 base capture done')
    swap19 = capture_states(
        rows_d19, [D19_L],
        swap_layers=SWAP_L)[D19_L]
    dvec19 = (swap19 - base19) \
        .astype(np.float32)
    dvec19_sha = hashlib.sha256(
        dvec19.tobytes()).hexdigest()[:8]
    dvec19_medn = float(np.median(
        np.linalg.norm(dvec19, axis=1)))
    ck_save('dvec19', {
        'dvec': dvec19,
        'sha8': dvec19_sha,
        'medn': dvec19_medn})
log('dvec19: sha8=%s med||d||=%.4f '
    'max||d||=%.4f (n=%d, 3135 swap4 '
    'semantics)' % (dvec19_sha,
                    dvec19_medn,
                    float(np.max(
                        np.linalg.norm(
                            dvec19,
                            axis=1))),
                    _N19))

# ================================================================
# PART C2: bank decomposition at CL
# ================================================================
log('== PART C2: decomposition ==')
IDEINT_P = {}
CVEC = {}
PC = {}
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
        log('C2 L%02d WR PC var %s mnorm '
            '%.3f' % (l,
                      json.dumps([round(v, 4)
                                  for v in
                                  _var]),
                      PC[l]['mnorm']))
        del WR, WRm, mu, WRc, _U, _S, _Vt
    del X, B, Ideint, C
    gc.collect()
log('C2 done (%d layers)' % len(CL))

# ================================================================
# PART C3: step1 fine scan + dvec19 trials
# ================================================================
log('== PART C3: step1 fine scan ==')
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


# step1 fine scan: SCAN_L x doses {1,2,4}
for l in SCAN_L:
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
# allstep d2 controls L17-21 (+ L23 anchor)
for l in SCAN_L:
    _cv = CVEC[l][2][None, :]
    _dv_full = np.repeat(
        _cv, SCAN_N, axis=0) \
        .astype(np.float32)
    tname = 'c_all_l%02d_d2.0' % l
    _run_vec_trial(tname, l, _dv_full,
                   2.0, 'allstep',
                   rows_scan, base12_P)
# iinj L17 d1 allstep bit anchor (3140/3141)
_dv_iinj = np.roll(
    IDEINT_P[17], -1, axis=0)[:SCAN_N] \
    .astype(np.float32)
_run_vec_trial('c_iinj17_d1.0', 17,
               _dv_iinj, 1.0, 'allstep',
               rows_scan, base12_P)
# dvec19 injection trials (new carrier)
assert _N19 >= SCAN_N, \
    'dvec19 rows insufficient'
_dv_d19 = dvec19[:SCAN_N].copy()
_run_vec_trial('c_dvec19_l19_d1.0', 19,
               _dv_d19, 1.0, 'allstep',
               rows_scan, base12_P)
_run_vec_trial('c_dvec19_l19_s1_d1.0', 19,
               _dv_d19, 1.0, STEP1,
               rows_scan, base12_P)

# bit-repro checks vs 3141 (subset)
n_bitC = 0
bit_anchors_C = {}
if not SMOKE:
    got = E_res['c_all_l19_d2.0']['chg']
    m = abs(got - ALL_D2_41['19']) < 1e-9
    n_bitC += int(m)
    bit_anchors_C['c_all_l19_d2.0'] = {
        'got': got, 'want': ALL_D2_41['19'],
        'match': bool(m)}
    got = E_res['c_iinj17_d1.0']['chg']
    m = abs(got - IINJ17_D1) < 1e-9
    n_bitC += int(m)
    bit_anchors_C['c_iinj17_d1.0'] = {
        'got': got, 'want': IINJ17_D1,
        'match': bool(m)}
    log('C3 3141 repro: %d/2 bit-match'
        % n_bitC)
else:
    log('C3 3141 repro SKIPPED (smoke)')

# step1 fine-scan spectrum
step1_d = {dose: {l: E_res['c_step1_l%02d_'
                           'd%.1f' % (l, dose)]
                  ['chg'] for l in SCAN_L}
           for dose in C_DOSES}
all_d2 = {l: E_res['c_all_l%02d_d2.0'
                   % l]['chg']
          for l in SCAN_L}
peak_s1 = max(step1_d[2.0],
              key=step1_d[2.0].get)
s1_max = step1_d[2.0][peak_s1]
mono_rows = []
for l in SCAN_L:
    cds = [step1_d[d][l]
           for d in C_DOSES]
    mono_rows.append(all(
        cds[i + 1] >= cds[i] - 1e-12
        for i in range(len(cds) - 1)))
s1_mono = float(np.mean(mono_rows))
# sensitive zone: layers where step1 d4
# >= 2x d1 (dose-responsive single-step)
sens_zone = [l for l in SCAN_L
             if step1_d[4.0][l]
             >= 2.0 * step1_d[1.0][l]]
d19_all = E_res['c_dvec19_l19_d1.0']['chg']
d19_s1 = E_res['c_dvec19_l19_s1_d1.0'][
    'chg']
d19_active = d19_all >= 0.05
log('C3-SOFT: step1 d2 %s; allstep d2 %s; '
    'peak L%02d (%.4f); mono %.2f; sens '
    'zone %s; dvec19 all_d1 %.4f s1_d1 '
    '%.4f (active %s)'
    % (json.dumps({str(l): round(v, 4)
                   for l, v in
                   step1_d[2.0].items()}),
       json.dumps({str(l): round(v, 4)
                   for l, v in
                   all_d2.items()}),
       peak_s1, s1_max, s1_mono,
       sens_zone, d19_all, d19_s1,
       d19_active))

# ================================================================
# PART D: blocking localization matrix
# ================================================================
log('== PART D: blocking matrix ==')
LJ = 29
_pc = PC[LJ]
v1 = _pc['V'][0]
dv_pc1 = {dose: np.tile(
    (v1 * _pc['mnorm'] * dose)[None, :],
    (SCAN_N, 1)).astype(np.float32)
    for dose in D_DOSES}
dv_dv29 = {dose: (dvec[LJ][:SCAN_N]
                  * dose).astype(np.float32)
           for dose in D_DOSES}
DV33_D1 = 1.0
dv_dv33 = (dvec[33][:SCAN_N]
           * DV33_D1).astype(np.float32)


def _run_joint(tname, spec):
    """spec: list of (il, dv_batch, scale)."""
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        return
    gen_l = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il, dv[b0:b0
                              + len(batch)],
                      1.0, 'allstep')
                     for (il, dv, sc)
                     in spec]))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base12_P)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname],
                    'fstep': fstep})


# singles (rows of the matrix)
for dose in D_DOSES:
    _run_vec_trial('d_pc1_l29_d%.1f' % dose,
                   LJ, dv_pc1[dose], 1.0,
                   'allstep', rows_scan,
                   base12_P)
for dose in D_DOSES:
    _run_vec_trial('d_dvec29_l29_d%.1f'
                   % dose, LJ,
                   dv_dv29[dose], 1.0,
                   'allstep', rows_scan,
                   base12_P)
# joint matrix 9 cells (pc1 dose a x
# dvec dose b; fixed hook order pc1->dvec)
for a in D_DOSES:
    for b in D_DOSES:
        _run_joint('d_joint_l29_a%.1f_b%.1f'
                   % (a, b),
                   [(LJ, dv_pc1[a], 1.0),
                    (LJ, dv_dv29[b], 1.0)])
# cross-layer: pc1@L29 d2 + dvec33@L33 d1
_run_vec_trial('d_dvec33_l33_d1.0', 33,
               dv_dv33, 1.0, 'allstep',
               rows_scan, base12_P)
_run_joint('d_cross_l29pc1_d2_l33dvec_d1',
           [(LJ, dv_pc1[2.0], 1.0),
            (33, dv_dv33, 1.0)])
# order reversal: dvec first, pc1 second
_run_joint('d_rev_l29_d2',
           [(LJ, dv_dv29[2.0], 1.0),
            (LJ, dv_pc1[2.0], 1.0)])

if not SMOKE:
    n_bitD = 0
    bit_anchors_D = {}
    for tn, v in (
            ('d_pc1_l29_d2.0', PC1_D2),
            ('d_dvec29_l29_d1.0', DVEC29_D1),
            ('d_dvec29_l29_d2.0', DVEC29_D2),
            ('d_joint_l29_a2.0_b2.0',
             JOINT_D2)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitD += int(m)
        bit_anchors_D[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('D 3141 repro: %d/4 bit-match'
        % n_bitD)
else:
    n_bitD = 0
    bit_anchors_D = {}
    log('D 3141 repro SKIPPED (smoke)')

# blocking index surface
BI = {}
for a in D_DOSES:
    for b in D_DOSES:
        jv = E_res['d_joint_l29_a%.1f_b%.1f'
                   % (a, b)]['chg']
        sa = E_res['d_pc1_l29_d%.1f'
                   % a]['chg']
        sb = E_res['d_dvec29_l29_d%.1f'
                   % b]['chg']
        p_add = min(sa + sb, 1.0)
        p_ind = 1.0 - (1.0 - sa) \
            * (1.0 - sb)
        bi = 1.0 - jv / max(p_add, 1e-12)
        BI['%.1f_%.1f' % (a, b)] = {
            'j': jv, 'p_add': p_add,
            'p_ind': p_ind, 'BI': bi}
# row scan: pc1 fixed d2, dvec dose up
row_b = [BI['2.0_%.1f' % b]['BI']
         for b in D_DOSES]
col_b = [BI['%.1f_2.0' % a]['BI']
         for a in D_DOSES]
if row_b[-1] > row_b[0] + 0.1 \
        and row_b[1] < row_b[-1] + 0.02:
    mat_tag = 'bi_dvec_dose_dominant'
elif col_b[-1] > col_b[0] + 0.1:
    mat_tag = 'bi_pc1_dose_dominant'
elif abs(row_b[0] - col_b[0]) < 0.1:
    mat_tag = 'bi_symmetric'
else:
    mat_tag = 'bi_mixed'
# cross-layer law
j_x = E_res['d_cross_l29pc1_d2_l33dvec_d1'][
    'chg']
s_x = E_res['d_dvec33_l33_d1.0']['chg']
pa_x = min(PC1_D2 + s_x, 1.0)
if j_x >= pa_x - JOINT_TOL:
    cross_tag = 'crosslayer_additive'
elif j_x <= max(PC1_D2, s_x) + BLOCK_TOL:
    cross_tag = 'crosslayer_blocking'
else:
    cross_tag = 'crosslayer_subadditive'
# order reversal
j_fwd = BI['2.0_2.0']['j']
j_rev = E_res['d_rev_l29_d2']['chg']
d_rev = abs(j_rev - j_fwd)
if d_rev < REV_TOL:
    rev_tag = 'order_bit_invariant'
elif d_rev < 0.02:
    rev_tag = 'order_minor'
else:
    rev_tag = 'order_sensitive'
log('D-SOFT: BI surface %s; row(2.0,*) '
    '%s col(*,2.0) %s -> %s; crosslayer '
    'j=%.4f (pc1 %.4f + dvec33 %.4f, '
    'p_add %.4f) -> %s; reversal d=%.2e '
    '-> %s'
    % (json.dumps({k: round(v['BI'], 3)
                   for k, v in BI.items()}),
       json.dumps([round(v, 3)
                   for v in row_b]),
       json.dumps([round(v, 3)
                   for v in col_b]),
       mat_tag, j_x, PC1_D2, s_x, pa_x,
       cross_tag, d_rev, rev_tag))

# ================================================================
# PART E: co36 enrichment quartiles
# ================================================================
log('== PART E: quartile window ==')
I17 = IDEINT_P[17]
fr = I17 ** 2 / np.maximum(
    np.sum(I17 ** 2, axis=1,
           keepdims=True), 1e-12)
co36 = CO_SETS['co36']
assert len(co36) == 50
score = np.mean(fr[:, co36], axis=0)
order_e = co36[np.argsort(-score)]
assert list(order_e[:25]) == HIGH25_41, \
    'order_e drift vs 3141 high25'
assert list(order_e[25:]) == LOW25_41, \
    'order_e drift vs 3141 low25'
log('E order_e equals 3141 high25/low25 '
    '(pure-numpy exact)')
q_sets = [np.asarray(q)
          for q in np.array_split(order_e, 4)]
q_sizes = [len(q) for q in q_sets]
assert q_sizes == [13, 13, 12, 12]
E_TRIALS = []
for qi, q in enumerate(q_sets):
    for ds in E_DOSES:
        E_TRIALS.append(
            ('e_q%d_d%.1f' % (qi + 1, ds),
             q, ds))
E_TRIALS += [('e_co36_d1', co36, 1.0),
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

n_bitE = 0
bit_anchors_E = {}
if not SMOKE:
    for tn, v in (('e_co36_d1',
                   G37['g_co36_d1']),
                  ('e_co36_d2',
                   G37['g_co36_d2']),
                  ('e_co50_d1',
                   G37['g_co50_d1'])):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitE += int(m)
        bit_anchors_E[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('E 3137 repro: %d/3 bit-match'
        % n_bitE)
else:
    log('E 3137 repro SKIPPED (smoke)')
qchg = {ds: [E_res['e_q%d_d%.1f'
                        % (qi + 1, ds)]['chg']
             for qi in range(4)]
        for ds in E_DOSES}
seps = {ds: qchg[ds][0] - qchg[ds][3]
        for ds in E_DOSES}
best_d = max(seps, key=seps.get)
qmono = all(
    qchg[best_d][i]
    >= qchg[best_d][i + 1] - 1e-12
    for i in range(3))
log('E-SOFT: Q chg d0.5 %s d1.0 %s; '
    'sep %s; best d%.1f (mono %s); vs '
    '3141 hi/lo d1 0.2188/0.0938 d2 '
    '0.2031/0.1719'
    % (json.dumps([round(v, 4)
                   for v in qchg[0.5]]),
       json.dumps([round(v, 4)
                   for v in qchg[1.0]]),
       json.dumps({('%.1f' % k):
                   round(v, 4)
                   for k, v in
                   seps.items()}),
       best_d, qmono))

# ================================================================
# PART F: in-context multi-template write
# ================================================================
log('== PART F: multi-template write ==')
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


def build_prompt_variant(s, o, lrel, qrel,
                         vi):
    """Variant vi: same semantics, new
    rng-seeded distractor lines."""
    rng3 = _rnd.Random(zlib.crc32(
        ('xmv|%d|%d_%d_%d_%d'
         % (vi, s, o, lrel, qrel))
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


rows_v1 = []
rows_var = {v: [] for v in range(4)}
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
    txt = build_prompt_new(s, o, r, r)
    rows_v1.append(list(tok_g(
        txt, add_special_tokens=False)
        ['input_ids']))
    for v in range(4):
        txtv = build_prompt_variant(
            s, o, r, r, v)
        rows_var[v].append(list(tok_g(
            txtv,
            add_special_tokens=False)
            ['input_ids']))
RL = (17, 29, 38)
# f1: fwd V1 states (3141 bit anchor)
_FK = CK['data'].get('fwd_cap')
if _FK is not None:
    vst = {int(k): v for k, v in
           _FK['states'].items()}
    log('F fwd capture RESUMED')
else:
    _cap = capture_states(rows_v1, RL)
    vst = {l: _cap[l] for l in RL}
    ck_save('fwd_cap', {
        'states': {str(l): vst[l]
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


RETR_F1 = {l: retr_from_states(vst[l], l)
           for l in RL}
for l in RL:
    log('F f1 fwd V1 retr L%02d: same-s '
        '%.4f' % (l, RETR_F1[l]))
if not SMOKE:
    f1_bit = all(
        abs(RETR_F1[l]
            - RETR_FWD_V1[str(l)]) < 1e-9
        for l in RL)
    log('F f1 bit anchor vs 3141 '
        'retr_fwd_v1: %s' % f1_bit)
else:
    f1_bit = None

# generations: V1 + 4 variants
gens_all = {}
for key, rows_k in (
        ('V1', rows_v1),
        ('V0', rows_var[0]),
        ('V1x', rows_var[1]),
        ('V2', rows_var[2]),
        ('V3', rows_var[3])):
    _GK = CK['data'].get('gen_%s' % key)
    if _GK is not None:
        gens_all[key] = _GK['gens']
        log('gen %s RESUMED' % key)
        continue
    gw = []
    for b0 in range(0, XMAT_N, GEN_BATCH):
        gw.extend(gen_batch_g2(
            rows_k[b0:b0 + GEN_BATCH]))
    gens_all[key] = gw
    ck_save('gen_%s' % key, {'gens': gw})

# f2: V1 gen write -> capture (3141 sem)
_F2 = CK['data'].get('cap_f2')
if _F2 is not None:
    st_f2 = {int(k): v for k, v in
             _F2['states'].items()}
    log('F f2 capture RESUMED')
else:
    full2 = [rows_v1[i]
             + list(gens_all['V1'][i])
             for i in range(XMAT_N)]
    _c2 = capture_states(full2, RL)
    st_f2 = {l: _c2[l] for l in RL}
    ck_save('cap_f2', {
        'states': {str(l): st_f2[l]
                   for l in RL}})
RETR_F2 = {l: retr_from_states(st_f2[l], l)
           for l in RL}
# f4: single variant-2 write -> capture
_F4 = CK['data'].get('cap_f4')
if _F4 is not None:
    st_f4 = {int(k): v for k, v in
             _F4['states'].items()}
    log('F f4 capture RESUMED')
else:
    full4 = [rows_var[2][i]
             + list(gens_all['V2'][i])
             + rows_v1[i]
             for i in range(XMAT_N)]
    _c4 = capture_states(full4, RL)
    st_f4 = {l: _c4[l] for l in RL}
    ck_save('cap_f4', {
        'states': {str(l): st_f4[l]
                   for l in RL}})
RETR_F4 = {l: retr_from_states(st_f4[l], l)
           for l in RL}
# f3: quad write -> capture
_F3 = CK['data'].get('cap_f3')
if _F3 is not None:
    st_f3 = {int(k): v for k, v in
             _F3['states'].items()}
    log('F f3 capture RESUMED')
else:
    vkeys = ('V0', 'V1x', 'V2', 'V3')
    vidx = {'V0': 0, 'V1x': 1, 'V2': 2,
            'V3': 3}
    full3 = []
    for i in range(XMAT_N):
        seq = []
        for vk in vkeys:
            seq += (rows_var[vidx[vk]][i]
                    + list(gens_all[vk][i]))
        seq += rows_v1[i]
        full3.append(seq)
    _c3 = capture_states(full3, RL)
    st_f3 = {l: _c3[l] for l in RL}
    ck_save('cap_f3', {
        'states': {str(l): st_f3[l]
                   for l in RL}})
RETR_F3 = {l: retr_from_states(st_f3[l], l)
           for l in RL}
for l in RL:
    log('F retr L%02d: f2_v1gen %.4f | '
        'f4_single %.4f | f3_quad %.4f'
        % (l, RETR_F2[l], RETR_F4[l],
           RETR_F3[l]))
if not SMOKE:
    f2_bit = all(
        abs(RETR_F2[l]
            - RETR_GEN_V1[str(l)]) < 1e-9
        for l in RL)
    log('F f2 bit anchor vs 3141 '
        'retr_gen_v1: %s' % f2_bit)
else:
    f2_bit = None
g_f2 = float(np.median(
    [RETR_F2[l] - RETR_F1[l]
     for l in RL]))
g_f4 = float(np.median(
    [RETR_F4[l] - RETR_F1[l]
     for l in RL]))
g_f3 = float(np.median(
    [RETR_F3[l] - RETR_F1[l]
     for l in RL]))
all_retr = [RETR_F2[l] for l in RL] \
    + [RETR_F3[l] for l in RL] \
    + [RETR_F4[l] for l in RL]
gen_best = float(max(all_retr))
gen_reach = gen_best >= XMAT_RETR_MIN
if g_f3 >= WI_GAIN \
        and g_f4 < 0.5 * max(g_f3, 1e-12):
    wtag = 'write_diversity_keyed'
elif g_f4 >= WI_GAIN \
        and g_f3 < g_f4 + WI_GAIN:
    wtag = 'write_dose_keyed'
elif g_f3 >= WI_GAIN:
    wtag = 'write_quad_gain'
elif max(g_f2, g_f3, g_f4) < WI_GAIN:
    wtag = 'write_structural_absent'
else:
    wtag = 'write_partial'
log('F-SOFT: gains med f2 %+.4f f4 '
    '%+.4f f3 %+.4f (WI_GAIN %.2f); '
    'gen best %.4f (>=0.30 %s) -> %s'
    % (g_f2, g_f4, g_f3, WI_GAIN,
       gen_best, gen_reach, wtag))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3141_ok']
if not SMOKE:
    n_bit_all = n_bitC + n_bitD + n_bitE
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == 9
                else 'repro_bit_drift')
else:
    tags.append('repro_smoke_skipped')
tags.append('dvec19_sha_%s' % dvec19_sha[:6])
tags.append('dvec19_active' if d19_active
            else 'dvec19_quiet')
tags.append('s1_peak_l%02d' % peak_s1)
tags.append('s1_zone_%s'
            % '_'.join(str(l)
                       for l in sens_zone))
tags.append('s1_dose_mono'
            if s1_mono >= 0.6
            else 's1_dose_flat')
tags.append(mat_tag)
tags.append(cross_tag)
tags.append(rev_tag)
tags.append('quartile_mono'
            if qmono
            else 'quartile_mono_fail')
tags.append('window_d%.1f' % best_d)
if not SMOKE:
    tags.append('f1fwd_bit_ok'
                if f1_bit
                else 'f1fwd_drift')
    tags.append('f2gen_bit_ok'
                if f2_bit
                else 'f2gen_drift')
else:
    tags.append('f1fwd_smoke_skipped')
    tags.append('f2gen_smoke_skipped')
tags.append(wtag)
tags.append('gen_retr_reach' if gen_reach
            else 'gen_retr_below')
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
runtime = time.time() - T0
log('VERDICT: %s' % verdict)
log('DONE (%.1fs)' % runtime)

result = {
    'phase': 3142,
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
        'dvec19': {'sha8': dvec19_sha,
                   'med_norm': dvec19_medn,
                   'n': _N19,
                   'swap_l': SWAP_L},
        'step1_scan': {str(dose): {
            str(l): step1_d[dose][l]
            for l in SCAN_L}
            for dose in C_DOSES},
        'allstep_ctrl': {str(l): all_d2[l]
                         for l in SCAN_L},
        'iinj17_d1': E_res[
            'c_iinj17_d1.0']['chg'],
        'dvec19_trials': {
            'all_d1': d19_all,
            's1_d1': d19_s1},
        'peak_s1': int(peak_s1),
        's1_max': s1_max,
        's1_mono_rate': s1_mono,
        'sens_zone': sens_zone,
        'bit_anchors_3141': bit_anchors_C},
    'part_d': {
        'singles': {
            'pc1': {str(dose): E_res[
                'd_pc1_l29_d%.1f'
                % dose]['chg']
                for dose in D_DOSES},
            'dvec29': {str(dose): E_res[
                'd_dvec29_l29_d%.1f'
                % dose]['chg']
                for dose in D_DOSES},
            'dvec33_d1': s_x},
        'matrix': {k: v for k, v
                   in BI.items()},
        'crosslayer_joint': j_x,
        'reversal_joint': j_rev,
        'rev_delta': d_rev,
        'mat_tag': mat_tag,
        'cross_tag': cross_tag,
        'rev_tag': rev_tag,
        'bit_anchors_3141': bit_anchors_D},
    'part_e': {
        'q_sizes': q_sizes,
        'q_chg': {('%.1f' % ds):
                  [float(v) for v in
                   qchg[ds]]
                  for ds in E_DOSES},
        'seps': {('%.1f' % k): float(v)
                 for k, v in seps.items()},
        'best_d': float(best_d),
        'quartile_mono': bool(qmono),
        'co36_d1': E_res['e_co36_d1']['chg'],
        'co36_d2': E_res['e_co36_d2']['chg'],
        'co50_d1': E_res['e_co50_d1']['chg'],
        'bit_anchors_3137': bit_anchors_E},
    'part_f': {
        'n_pairs': XMAT_N,
        'retr_f1_fwd_v1': {str(l): RETR_F1[l]
                           for l in RL},
        'retr_f2_gen_v1': {str(l): RETR_F2[l]
                           for l in RL},
        'retr_f4_single': {str(l): RETR_F4[l]
                           for l in RL},
        'retr_f3_quad': {str(l): RETR_F3[l]
                         for l in RL},
        'gains_med': {'f2': g_f2, 'f4': g_f4,
                      'f3': g_f3},
        'gen_best': gen_best,
        'write_tag': wtag,
        'f1_bit_3141': f1_bit,
        'f2_bit_3141': f2_bit}}
with io.open(os.path.join(
        OUT, 'result.json'), 'w',
        encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False, indent=1)
np.savez(os.path.join(
    OUT, 'p140_readout.npz'),
    c_step1_d1=np.array(
        [step1_d[1.0][l] for l in SCAN_L]),
    c_step1_d2=np.array(
        [step1_d[2.0][l] for l in SCAN_L]),
    c_step1_d4=np.array(
        [step1_d[4.0][l] for l in SCAN_L]),
    c_all_d2=np.array(
        [all_d2[l] for l in SCAN_L]),
    scan_l=np.array(SCAN_L),
    d19=np.array([d19_all, d19_s1]),
    d_matrix=np.array(
        [BI['%.1f_%.1f' % (a, b)]['j']
         for a in D_DOSES
         for b in D_DOSES]),
    d_singles=np.array(
        [E_res['d_pc1_l29_d%.1f' % dz]['chg']
         for dz in D_DOSES]
        + [E_res['d_dvec29_l29_d%.1f'
                 % dz]['chg']
           for dz in D_DOSES]
        + [s_x, j_x, j_rev]),
    e_q=np.array(
        [qchg[ds][qi]
         for ds in E_DOSES
         for qi in range(4)]),
    retr=np.array(
        [RETR_F1[l] for l in RL]
        + [RETR_F2[l] for l in RL]
        + [RETR_F4[l] for l in RL]
        + [RETR_F3[l] for l in RL]))
log('dumps done: result.json + seal + '
    'p140_readout.npz')
if not SMOKE:
    if os.path.exists(CKPTF):
        os.remove(CKPTF)
        log('ckpt cleared')
log('PHASE 3142 COMPLETE')
