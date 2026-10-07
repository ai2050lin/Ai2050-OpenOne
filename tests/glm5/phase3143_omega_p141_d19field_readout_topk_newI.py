# -*- coding: utf-8 -*-
"""Phase 3143 (Omega-P141): dvec19
downstream conduction field (is the L19
carrier consumed through the L17 pathway?)
+ direct readout-competition observation
(pc1 projection shifts + w_dn synthetic
angle under dvec/joint injection)
+ co36 enrichment top-k sliding threshold
+ new-material I-component existence
(IDEINT_new self-retrieval vs old-bank
cross: identity = old-material weight
memory vs generic encoding mechanism).

Preregistered in 3142 closeout (aligned
with FINGERPRINT_PARADIGM_PLAN.md
Omega-P141):
(1) dvec19 re-capture (3135 swap4
semantics, sha-anchored 6a0332a6) +
full-field rebuild (128 rows x 40 layers
base/swap) with self-consistency checks
(field[17] vs frozen dvec17, field[19]
vs dvec19, z35 cos_17 spectra soft
anchor) + injection captures dvec19@L19
and dvec17@L17 (DOSE_COND 2.0, 3135 B2
semantics) -> cos/rho spectra vs swap4
field, cross-field cos dh19 vs dh17,
spectral Spearman -> pathway verdict;
(2) readout competition: 2 generation
bit anchors (pc1_l29_d2 0.203125,
dvec29_l29_d2 0.640625) + 3-condition
teacher-forced injection captures
(pc1/dvec29/joint at L29, layers
29/33/38) -> pc1 projection, w_dn
projection, dh.w_dn, state-additivity
cos(dh_joint, dh_pc1+dh_dv29) ->
competition mechanism tags;
(3) top-k sliding: order_e rebuilt
(== 3141 high25/low25 asserted) k in
{5,10,13,15,20,25,30} + co36 full-50
anchor injected at L17 (3137 G
semantics: DELTA_L17, sgn -1, mode 0,
A1 rows, dose 1.0) -> k* head
concentration; k=13 == 3142 Q1 bit
anchor 0.15625;
(4) new-material I: 84 unseen pairs,
V1 prompt + 4 variant templates (3142 F
rng semantics, prompt-only captures at
17/29/38) -> IDEINT_new from 4 variants
minus old-bank layer base B_l, self
retrieval (held-out V1 query) vs old
IDEINT_P cross retrieval (f1 bit anchor
retr_fwd_v1) -> identity encoding
verdict.

Bank shards REUSED from 3138 full run.
Frozen dvecs from 3135 (sha-anchored).
Session baselines P+A1 with xphase
recording."""
import gc
import hashlib
import io
import json
import os
import pickle
import random as _rnd
import time
import zlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
NAME = ('omega_p141_d19field_readout_'
        'topk_newI')
SMOKE = os.environ.get('P3143_SMOKE',
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
D38 = RDIR + r'\phase3138' \
      r'\omega_p136_statebank_' \
      r'bicdecomp_probecontrast'
D41 = RDIR + r'\phase3141' \
      r'\omega_p139_cinjrefine_wrjoint_' \
      r'coenrich_writeinduce'
D42 = RDIR + r'\phase3142' \
      r'\omega_p140_l19anat_blockmat_' \
      r'quartile_multitpl'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3143', NAME)
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
CKPTF = os.path.join(OUT, 'p141_ckpt.pkl')
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
D17_L = 17
NCAP = 4 if SMOKE else 128      # field/conduction rows
N19_ROWS = 4 if SMOKE else NP   # dvec19 capture rows
GEN_BATCH = 4 if SMOKE else 32
N_NEW = 12
DOSE_COND = 2.0                 # 3135 B2 semantics
DEC_L = (17, 29, 38)            # bank decomposition layers
CAP_B = (29, 33, 38)            # readout competition capture
RL = (17, 29, 38)               # retrieval layers
K_GRID = (5, 10, 13, 15, 20, 25, 30)
E_DOSE = 1.0
FIELD_SELF_COS = 0.99
Z35_TOL = 0.08
PATH_CORR_HI = 0.7
PATH_CORR_LO = 0.3
CROSS_COS_HI = 0.8
DIRECT_COS_HI = 0.6
STAT_ADD_COS = 0.9
COMP_TOL = 0.02
PC1_DARK_RATIO = 0.25
HEAD_FRAC = 0.9
SELF_MULT = 3.0
RNG_SEED = 3143

# ---------------- frozen 3142 anchors --
EXP_V42 = ('a_3141_ok|repro_bit_9|'
           'repro_bit_ok|dvec19_sha_6a0332|'
           'dvec19_active|s1_peak_l21|'
           's1_zone_19_20_21|s1_dose_mono|'
           'bi_pc1_dose_dominant|'
           'crosslayer_blocking|'
           'order_bit_invariant|'
           'quartile_mono_fail|'
           'window_d1.0|f1fwd_bit_ok|'
           'f2gen_bit_ok|'
           'write_structural_absent|'
           'gen_retr_below|xphase_ok|'
           'coverage_full')
RES42_SHA = '7291cfc5'
SEAL42 = '449f8161'
XPHASE42 = 1.0
DVEC19_SHA = '6a0332a6'
MEDNORM19 = 5.643608093261719
D19_ALL_D1 = 0.7890625
D19_S1_D1 = 0.1640625
S1_D2_L21 = 0.203125
ALL_D2_L19 = 0.4921875
IINJ17_D1 = 0.328125
PEAK_S1_42 = 21
SENS_ZONE_42 = [19, 20, 21]
PC1_D2 = 0.203125
DVEC29_D1 = 0.25
DVEC29_D2 = 0.640625
JOINT_D2 = 0.4765625
CROSS_JOINT_42 = 0.1953125
CO36_D1 = 0.140625
CO36_D2 = 0.2265625
CO50_D1 = 0.421875
Q1_D1 = 0.15625
QCHG_D1_42 = [0.15625, 0.1640625,
              0.109375, 0.1015625]
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
# ---------------- frozen 3135/3137 -----
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
DELTA_L17 = 0.637683315669971
RES36_SHA = '3903af46'
LEDGER_N = 279

# ---------------- seal ------------------
SEAL = {
    'phase': 3143,
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
        'D19_L': D19_L, 'D17_L': D17_L,
        'NCAP': NCAP,
        'N19_ROWS': N19_ROWS,
        'GEN_BATCH': GEN_BATCH,
        'N_NEW': N_NEW,
        'DOSE_COND': DOSE_COND,
        'DEC_L': list(DEC_L),
        'CAP_B': list(CAP_B),
        'RL': list(RL),
        'K_GRID': list(K_GRID),
        'E_DOSE': E_DOSE,
        'FIELD_SELF_COS': FIELD_SELF_COS,
        'Z35_TOL': Z35_TOL,
        'PATH_CORR_HI': PATH_CORR_HI,
        'PATH_CORR_LO': PATH_CORR_LO,
        'CROSS_COS_HI': CROSS_COS_HI,
        'DIRECT_COS_HI': DIRECT_COS_HI,
        'STAT_ADD_COS': STAT_ADD_COS,
        'COMP_TOL': COMP_TOL,
        'PC1_DARK_RATIO': PC1_DARK_RATIO,
        'HEAD_FRAC': HEAD_FRAC,
        'SELF_MULT': SELF_MULT,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res42_verdict': EXP_V42,
        'res42_sha8': RES42_SHA,
        'seal42': SEAL42,
        'xphase42': XPHASE42,
        'dvec19_sha8': DVEC19_SHA,
        'mednorm19': MEDNORM19,
        'd19_all_d1': D19_ALL_D1,
        'd19_s1_d1': D19_S1_D1,
        's1_d2_l21': S1_D2_L21,
        'all_d2_l19': ALL_D2_L19,
        'iinj17_d1': IINJ17_D1,
        'peak_s1_42': PEAK_S1_42,
        'sens_zone_42': SENS_ZONE_42,
        'pc1_d2': PC1_D2,
        'dvec29_d1': DVEC29_D1,
        'dvec29_d2': DVEC29_D2,
        'joint_d2': JOINT_D2,
        'cross_joint_42': CROSS_JOINT_42,
        'co36_d1': CO36_D1,
        'co36_d2': CO36_D2,
        'co50_d1': CO50_D1,
        'q1_d1': Q1_D1,
        'qchg_d1_42': QCHG_D1_42,
        'high25_41': HIGH25_41,
        'low25_41': LOW25_41,
        'retr_fwd_v1': RETR_FWD_V1,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'delta_l17': DELTA_L17,
        'res36_sha8': RES36_SHA,
        'ledger_n': LEDGER_N},
    'prereg': ('3142 closeout + '
               'FINGERPRINT_PARADIGM_PLAN '
               'Omega-P141: (1) dvec19 re-capture '
               '(sha 6a0332a6) + 128-row full '
               'field rebuild + self-consistency '
               '(field17 vs dvec17, field19 vs '
               'dvec19, z35 cos_17 soft) + '
               'injection captures dvec19@L19 / '
               'dvec17@L17 at dose 2.0 -> '
               'cos/rho spectra, cross-field '
               'cos, spectral Spearman -> '
               'pathway verdict; (2) readout '
               'competition: 2 generation bit '
               'anchors + 3-condition captures '
               '(pc1/dvec29/joint @L29, layers '
               '29/33/38) -> w_dn projection, '
               'state-additivity cos, pc1-dark '
               'test; (3) top-k sliding k in '
               '{5,10,13,15,20,25,30} @L17 A1 '
               'dose 1.0 (3137 G semantics) + '
               'co36 full-50 anchor + Q1(k=13) '
               'bit anchor -> k*; (4) new-'
               'material I: 84 pairs V1+4 '
               'variant prompt captures @RL, '
               'IDEINT_new self-retrieval '
               '(held-out V1) vs IDEINT_P cross '
               '(f1 bit anchor) -> identity '
               'encoding verdict. Bank reused '
               '3138. Frozen before observation.')}
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
    log('frozen dvec L%02d sha8 %s'
        % (l, DVEC_SHA[l]))
z136 = np.load(D36 + r'\p134_readout.npz',
               allow_pickle=False)
CO_SETS = {}
for cn in ('co36', 'co50', 'union'):
    _c = np.asarray(z136[cn]).astype(np.int64)
    assert _c.ndim == 1, (cn, _c.shape)
    assert _c.min() >= 0 and _c.max() < 4096
    CO_SETS[cn] = _c
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
assert len(led['measurements']) == LEDGER_N
log('frozen inputs ok (26/35/36 + ledger '
    'n=%d)' % LEDGER_N)

# ================================================================
# PART A: 3142 link asserts
# ================================================================
log('== PART A: link asserts ==')
res42 = json.load(io.open(
    D42 + r'\result.json', encoding='utf-8'))
assert res42['smoke'] is False
V42 = res42['verdict']
assert V42 == EXP_V42, V42
raw42 = io.open(
    D42 + r'\result.json', 'rb').read()
sha42 = hashlib.sha256(raw42).hexdigest()[:8]
assert sha42 == RES42_SHA, sha42
assert str(res42['seal_sha8']) == SEAL42
pc42 = res42['part_c']
pd42 = res42['part_d']
pe42 = res42['part_e']
pf42 = res42['part_f']
assert abs(float(pc42['xphase_P'])
           - XPHASE42) < 1e-12
assert abs(float(pc42['xphase_A1'])
           - XPHASE42) < 1e-12
_d19m = pc42['dvec19']
assert str(_d19m['sha8']) == DVEC19_SHA
assert abs(float(_d19m['med_norm'])
           - MEDNORM19) < 1e-9
_d19t = pc42['dvec19_trials']
assert abs(float(_d19t['all_d1'])
           - D19_ALL_D1) < 1e-9
assert abs(float(_d19t['s1_d1'])
           - D19_S1_D1) < 1e-9
assert int(pc42['peak_s1']) == PEAK_S1_42
assert list(pc42['sens_zone']) == \
    SENS_ZONE_42
assert abs(float(pc42['step1_scan']
                 ['2.0']['21'])
           - S1_D2_L21) < 1e-9
assert abs(float(pc42['allstep_ctrl']['19'])
           - ALL_D2_L19) < 1e-9
assert abs(float(pc42['iinj17_d1'])
           - IINJ17_D1) < 1e-9
assert all(v['match']
           for v in
           pc42['bit_anchors_3141'].values())
_s = pd42['singles']
assert abs(float(_s['pc1']['2.0'])
           - PC1_D2) < 1e-9
assert abs(float(_s['dvec29']['1.0'])
           - DVEC29_D1) < 1e-9
assert abs(float(_s['dvec29']['2.0'])
           - DVEC29_D2) < 1e-9
assert abs(float(pd42['matrix']['2.0_2.0']
                 ['j']) - JOINT_D2) < 1e-9
assert abs(float(pd42['crosslayer_joint'])
           - CROSS_JOINT_42) < 1e-9
assert abs(float(pd42['rev_delta'])) < 1e-12
assert str(pd42['mat_tag']) == \
    'bi_pc1_dose_dominant'
assert str(pd42['cross_tag']) == \
    'crosslayer_blocking'
assert str(pd42['rev_tag']) == \
    'order_bit_invariant'
assert all(v['match']
           for v in
           pd42['bit_anchors_3141'].values())
assert abs(float(pe42['co36_d1'])
           - CO36_D1) < 1e-9
assert abs(float(pe42['co36_d2'])
           - CO36_D2) < 1e-9
assert abs(float(pe42['co50_d1'])
           - CO50_D1) < 1e-9
assert [float(v) for v in
        pe42['q_chg']['1.0']] == QCHG_D1_42
assert list(pe42['q_sizes']) == \
    [13, 13, 12, 12]
assert pe42['quartile_mono'] is False
assert all(v['match']
           for v in
           pe42['bit_anchors_3137'].values())
assert str(pf42['write_tag']) == \
    'write_structural_absent'
assert pf42['f1_bit_3141'] is True
assert pf42['f2_bit_3141'] is True
for l, v in RETR_FWD_V1.items():
    got = float(pf42['retr_f1_fwd_v1'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
log('A hard asserts ok (3142 sha8 %s seal '
    '%s: xphase/dvec19/trials/s1/all/pc1/'
    'dvec29/joint/cross/co36/qchg/retr '
    'anchors)' % (sha42, SEAL42))

# ================================================================
# materials factory (3142 lineage)
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
ENTS_N = len(ents_all)


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
    return {'texts': texts, 'PID_T': PID_T,
            'DOT': DOT}


log('factory core defined')

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
pids_P_all = matg['PID_T']['P']
pids_A1_all = matg['PID_T']['A1']
_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
log('gen-prefix ready (%d ids; 3142 '
    'BOS semantics)' % len(PREFIX_IDS))
log('materials ready (672x2)')


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
                 inj=None,
                 inj_vec=None):
    """Generate N_NEW tokens per row from
    prompt ids. inj_vec: (il, dvec_batch
    (n,HID) fp32, scale, mode). mode:
    'allstep' = every forward; int s =
    prompt forward + decode step s (3135
    semantics). inj: (il, coords, delta,
    sgn, mode) coordinate injection
    (3137 G semantics)."""
    ids_p, mask_p = _pad_batch(pids_list)
    hooks = []
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
                if (_m == 'allstep'
                        or _st['step'] == 0
                        or _st['step']
                        == _m):
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


def capture_states_inject2(rows_all,
                           cap_layers,
                           inj_list):
    """Single-sample forwards with per-row
    dvec injections (3135
    capture_states_inject semantics,
    multi-injection extended). inj_list:
    list of (il, dv_rows (n,HID) fp32,
    scale). Injection is in-place on the
    layer output; capture hooks registered
    after injection hooks read the
    post-injection field."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    specs = [(model_g.model.layers[il],
              dvs, sc)
             for (il, dvs, sc) in inj_list]
    for j in range(len(rows_all)):
        feats = {l: None
                 for l in cap_layers}
        hooks = []
        for (lyr, dvs, sc) in specs:
            dv_t = torch.as_tensor(
                np.asarray(dvs[j],
                           dtype=np.float32),
                device='cuda') \
                .to(torch.bfloat16) \
                * float(sc)

            def _inj(mod, inp, out,
                     _dv=dv_t):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                o2[0, -1, :] += _dv
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _inj))

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
    return chg_l, first_l, fstep


def rankdata_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x),
                     dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) \
                and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata_avg(a)
    rb = rankdata_avg(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum()
                  * (rb ** 2).sum())
    if den < 1e-12:
        return 0.0
    return float((ra * rb).sum() / den)


# ================================================================
# PART C0: session baselines P+A1 + xphase
# ================================================================
log('== PART C0: baselines P+A1 ==')
CKM = {'smoke': SMOKE, 'ncap': NCAP,
       'n19': N19_ROWS,
       'gen_prefix': True}
_KEEP_BOS_FREE = ('dvec19', 'field',
                  'cond19', 'cond17')
if CK['meta'] and CK['meta'] != CKM:
    _nd = {k: v for k, v in
           CK['data'].items()
           if k in _KEEP_BOS_FREE}
    _nold = len(CK['data'])
    CK = {'done': sorted(_nd.keys()),
          'data': _nd, 'meta': {}}
    log('CKPT meta mismatch -> kept %d '
        'BOS-independent capture stages, '
        'dropped %d gen stages'
        % (len(_nd), _nold - len(_nd)))
CK['meta'] = CKM
rows_scan = [pids_P_all[j]
             for j in range(NCAP)]
_EB = CK['data'].get('e_base_P')
if _EB is not None:
    gen_base_P = _EB['gens']
    xphase_P = _EB['xphase']
    log('base P RESUMED (xphase=%.4f)'
        % xphase_P)
else:
    gen_base_P = []
    for b0 in range(0, NCAP, GEN_BATCH):
        gen_base_P.extend(gen_batch_g2(
            rows_scan[b0:b0 + GEN_BATCH]))
    _xp = [pad12(gen_base_P[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_P'][j]])
           for j in range(NCAP)]
    xphase_P = float(np.mean(_xp))
    log('xphase P: session-base vs z26 '
        'bit-match %.4f (%d/%d)'
        % (xphase_P, int(np.sum(_xp)),
           NCAP))
    ck_save('e_base_P', {
        'gens': gen_base_P,
        'xphase': xphase_P})
base12_P = [pad12(gen_base_P[j])
            for j in range(NCAP)]
rows_A1 = [pids_A1_all[j]
           for j in range(NCAP)]
_EB1 = CK['data'].get('e_base_A1')
if _EB1 is not None:
    gen_base_A1 = _EB1['gens']
    xphase_A1 = _EB1['xphase']
    log('base A1 RESUMED (xphase=%.4f)'
        % xphase_A1)
else:
    gen_base_A1 = []
    for b0 in range(0, NCAP, GEN_BATCH):
        gen_base_A1.extend(gen_batch_g2(
            rows_A1[b0:b0 + GEN_BATCH]))
    _xa = [pad12(gen_base_A1[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_A1'][j]])
           for j in range(NCAP)]
    xphase_A1 = float(np.mean(_xa))
    log('xphase A1: session-base vs z26 '
        'bit-match %.4f (%d/%d)'
        % (xphase_A1, int(np.sum(_xa)),
           NCAP))
    ck_save('e_base_A1', {
        'gens': gen_base_A1,
        'xphase': xphase_A1})
base12_A1 = [pad12(gen_base_A1[j])
             for j in range(NCAP)]
xphase_ok = (xphase_P == 1.0) \
    and (xphase_A1 == 1.0)

# ================================================================
# PART C1: dvec19 re-capture (sha assert)
# ================================================================
log('== PART C1: dvec19 re-capture ==')
_K19 = CK['data'].get('dvec19')
if _K19 is not None \
        and _K19.get('sha8') == DVEC19_SHA:
    dvec19 = _K19['dvec']
    dvec19_sha = _K19['sha8']
    log('dvec19 RESUMED sha8=%s'
        % dvec19_sha)
else:
    rows_d19 = [pids_P_all[j]
                for j in range(N19_ROWS)]
    base19 = capture_states(
        rows_d19, [D19_L],
        swap_layers=None)[D19_L]
    log('dvec19 base capture done '
        '(%d rows)' % N19_ROWS)
    swap19 = capture_states(
        rows_d19, [D19_L],
        swap_layers=SWAP_L)[D19_L]
    dvec19 = (swap19 - base19) \
        .astype(np.float32)
    dvec19_sha = hashlib.sha256(
        dvec19.tobytes()).hexdigest()[:8]
    ck_save('dvec19', {
        'dvec': dvec19,
        'sha8': dvec19_sha})
dvec19_medn = float(np.median(
    np.linalg.norm(dvec19, axis=1)))
d19_sha_ok = (dvec19_sha == DVEC19_SHA)
d19_medn_drift = abs(
    dvec19_medn - MEDNORM19)
log('dvec19: sha8=%s (want %s, match %s) '
    'med||d||=%.4f (drift %.2e, n=%d)'
    % (dvec19_sha, DVEC19_SHA, d19_sha_ok,
       dvec19_medn, d19_medn_drift,
       dvec19.shape[0]))
assert d19_medn_drift < 0.05, \
    d19_medn_drift

# ================================================================
# PART C2: bank load + decomposition @DEC_L
# ================================================================
log('== PART C2: bank + decomposition ==')
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
IDEINT_P = {}
BANK_B = {}
PC = {}
for l in DEC_L:
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
    BANK_B[l] = B.copy()
    if l == 29:
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
        PC[29] = {
            'V': _Vt[:4].astype(np.float32),
            'var': _var,
            'mnorm': float(np.median(
                np.linalg.norm(
                    WRc, axis=1)))}
        log('C2 L29 WR PC var %s mnorm '
            '%.3f' % (json.dumps(
                [round(v, 4)
                 for v in _var]),
                PC[29]['mnorm']))
        del WR, WRm, mu, WRc, _U, _S, _Vt
    del X, B, Ideint, C
    gc.collect()
log('C2 done (%d layers)' % len(DEC_L))

# ================================================================
# PART D1: full-field rebuild (128 rows)
# ================================================================
log('== PART D1: field rebuild ==')
rows_cap = [pids_P_all[j]
            for j in range(NCAP)]
_KF = CK['data'].get('field')
if _KF is not None:
    base_states = {int(k):
                   v.astype(np.float32)
                   for k, v in
                   _KF['base'].items()}
    swap_states = {int(k):
                   v.astype(np.float32)
                   for k, v in
                   _KF['swap'].items()}
    log('field RESUMED')
else:
    base_states = capture_states(
        rows_cap, ALL_L, swap_layers=None)
    log('field base capture done (%d rows '
        'x %d layers)' % (NCAP, len(ALL_L)))
    swap_states = capture_states(
        rows_cap, ALL_L,
        swap_layers=SWAP_L)
    log('field swap4 capture done')
    ck_save('field', {
        'base': {str(l): base_states[l]
                 .astype(np.float16)
                 for l in ALL_L},
        'swap': {str(l): swap_states[l]
                 .astype(np.float16)
                 for l in ALL_L}})
field = {}
for l in ALL_L:
    field[l] = (swap_states[l]
                - base_states[l]) \
        .astype(np.float32)
# self-consistency: field[17] vs frozen
# dvec17; field[19] vs dvec19


def _rowcos(A, Bm):
    num = np.sum(A.astype(np.float64)
                 * Bm.astype(np.float64),
                 axis=1)
    den = (np.linalg.norm(
        A.astype(np.float64), axis=1)
        * np.linalg.norm(
            Bm.astype(np.float64), axis=1))
    with np.errstate(invalid='ignore',
                     divide='ignore'):
        return np.where(den > 1e-12,
                        num / den, 0.0)


fs17 = float(np.median(
    _rowcos(field[17], dvec[17][:NCAP])))
fs19 = float(np.median(
    _rowcos(field[19], dvec19[:NCAP])))
field_self_ok = (fs17 >= FIELD_SELF_COS
                 and fs19 >= FIELD_SELF_COS)
log('field self-consistency: cos(field17,'
    'dvec17)=%.4f cos(field19,dvec19)='
    '%.4f (gate %.2f -> %s)'
    % (fs17, fs19, FIELD_SELF_COS,
       field_self_ok))
# z35 cos_17 spectra soft anchor (full:
# med over 672; ours: med over NCAP)
z35_c17_med = np.array(
    [float(np.nanmedian(z35['cos_17'][r]))
     for r in range(20, 40)])
# injection captures (conduction)
_KA = CK['data'].get('cond19')
if _KA is not None:
    h19 = {int(k):
           v.astype(np.float32)
           for k, v in
           _KA['h'].items()}
    log('cond19 RESUMED')
else:
    h19 = capture_states_inject2(
        rows_cap, ALL_L,
        [(D19_L, dvec19[:NCAP], DOSE_COND)])
    log('cond19 injection capture done')
    ck_save('cond19', {
        'h': {str(l): h19[l]
              .astype(np.float16)
              for l in ALL_L}})
_KA7 = CK['data'].get('cond17')
if _KA7 is not None:
    h17 = {int(k):
           v.astype(np.float32)
           for k, v in
           _KA7['h'].items()}
    log('cond17 RESUMED')
else:
    h17 = capture_states_inject2(
        rows_cap, ALL_L,
        [(D17_L, dvec[17][:NCAP],
          DOSE_COND)])
    log('cond17 injection capture done')
    ck_save('cond17', {
        'h': {str(l): h17[l]
              .astype(np.float16)
              for l in ALL_L}})


def _spec(h_inj, il):
    cos_m = np.full((len(ALL_L), NCAP),
                    np.nan,
                    dtype=np.float32)
    rho_m = np.full((len(ALL_L), NCAP),
                    np.nan,
                    dtype=np.float32)
    for r in ALL_L:
        if r <= il:
            continue
        dh = (h_inj[r].astype(np.float64)
              - base_states[r]
              .astype(np.float64))
        dv = field[r].astype(np.float64)
        ndh = np.linalg.norm(dh, axis=1)
        ndv = np.linalg.norm(dv, axis=1)
        num = np.sum(dh * dv, axis=1)
        den = ndh * ndv
        with np.errstate(
                invalid='ignore',
                divide='ignore'):
            cos_m[r] = np.where(
                den > 1e-12, num / den,
                0.0)
            rho_m[r] = np.where(
                ndv > 1e-12, ndh / ndv,
                0.0)
    return cos_m, rho_m


cos19, rho19 = _spec(h19, D19_L)
cos17, rho17 = _spec(h17, D17_L)
# z35 soft anchor on the L17 spectra
ours_c17_med = np.array(
    [float(np.nanmedian(cos17[r]))
     for r in range(20, 40)])
z35_diff = float(np.max(np.abs(
    ours_c17_med - z35_c17_med)))
z35_ok = z35_diff < Z35_TOL
log('z35 cos_17 soft anchor: max|diff| '
    'over L20-39 = %.4f (tol %.2f -> %s)'
    % (z35_diff, Z35_TOL, z35_ok))
# cross-field: cos(dh19, dh17) on common
# downstream r >= 20
cross_cos = {}
for r in range(20, 40):
    dh19r = (h19[r].astype(np.float64)
             - base_states[r]
             .astype(np.float64))
    dh17r = (h17[r].astype(np.float64)
             - base_states[r]
             .astype(np.float64))
    cross_cos[r] = _rowcos(dh19r, dh17r)
cross_med = {r: float(np.median(v))
             for r, v in
             cross_cos.items()}
cross_near = float(np.median(
    [cross_med[r] for r in range(20, 28)]))
# spectral correlation over L20-39
spec_corr = spearman(
    ours_c17_med.tolist(),
    [float(np.nanmedian(cos19[r]))
     for r in range(20, 40)])
# direct downstream gates (3135
# semantics r=il+1..il+3)
d19_direct = float(np.median(
    [float(np.nanmedian(cos19[r]))
     for r in (20, 21, 22)]))
d19_direct_rho = float(np.median(
    [float(np.nanmedian(rho19[r]))
     for r in (20, 21, 22)]))
d17_direct = float(np.median(
    [float(np.nanmedian(cos17[r]))
     for r in (18, 19, 20)]))
# pathway verdict
if spec_corr >= PATH_CORR_HI \
        and cross_near >= CROSS_COS_HI:
    path_tag = 'd19_path_same'
elif spec_corr < PATH_CORR_LO \
        or cross_near < 0.3:
    path_tag = 'd19_path_indep'
else:
    path_tag = 'd19_path_partial'
d19_direct_hi = d19_direct >= \
    DIRECT_COS_HI
log('D1-SOFT: spec_corr(L20-39)=%.3f '
    'cross_near(L20-27)=%.3f; d19 direct '
    'cos %.3f rho %.3f (hi %s); d17 '
    'direct cos %.3f -> %s'
    % (spec_corr, cross_near, d19_direct,
       d19_direct_rho, d19_direct_hi,
       d17_direct, path_tag))

# ================================================================
# PART D2: readout competition
# ================================================================
log('== PART D2: readout competition ==')
v1 = PC[29]['V'][0]
mn29 = PC[29]['mnorm']
dv_pc1_pos = np.tile(
    (v1 * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_pc1_neg = -dv_pc1_pos
dv_pc1 = dv_pc1_pos
pc1_sign = None  # resolved below
dv_dv29 = (dvec[29][:NCAP] * 2.0) \
    .astype(np.float32)
# generation bit anchors (3142 D
# semantics; hook order pc1->dvec in
# joint is irrelevant for singles)
E_res = {}


def _run_vec_trial(tname, il, dv_batch,
                   scale, mode, rows,
                   base12):
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
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})


_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P)
# sign disambiguation: the preregistered
# 3142 anchor resolves the SVD sign free
# parameter (LAPACK signs are not stable
# across sessions). If neither sign
# matches, the SVD drifted numerically
# -> hard fail.
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - PC1_D2) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg'] - PC1_D2) < 1e-9:
        pc1_sign = -1
    else:
        raise AssertionError(
            'pc1 sign unresolved: pos '
            '%.6f neg %.6f want %.6f'
            % (E_res['b_pc1_l29_d2.0']
               ['chg'],
               E_res['b_pc1neg_l29_d2.0']
               ['chg'], PC1_D2))
else:
    if not SMOKE:
        pc1_sign = 1
if pc1_sign == -1:
    dv_pc1 = dv_pc1_neg
    log('D2 pc1 sign resolved: -1 '
        '(neg reproduces 3142 anchor '
        '%.6f)' % PC1_D2)
elif pc1_sign == 1:
    log('D2 pc1 sign resolved: +1 '
        '(pos reproduces 3142 anchor)')
_run_vec_trial('b_dvec29_l29_d2.0', 29,
               dv_dv29, 1.0, 'allstep',
               rows_scan, base12_P)
n_bitB = 0
bit_anchors_B = {}
if not SMOKE:
    for tn, v in (('b_pc1_l29_d2.0',
                   PC1_D2),
                  ('b_dvec29_l29_d2.0',
                   DVEC29_D2)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitB += int(m)
        bit_anchors_B[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('D2 3142 repro: %d/2 bit-match'
        % n_bitB)
else:
    log('D2 3142 repro SKIPPED (smoke)')
# 3-condition captures @L29, layers CAP_B
_cond_specs = {
    'pc1': [(29, dv_pc1, 1.0)],
    'dv29': [(29, dv_dv29, 1.0)],
    'joint': [(29, dv_pc1, 1.0),
              (29, dv_dv29, 1.0)]}
HB = {}
for cn, spec in _cond_specs.items():
    _K = CK['data'].get('bcap_%s' % cn)
    if _K is not None:
        HB[cn] = {int(k):
                  v.astype(np.float32)
                  for k, v in
                  _K['h'].items()}
        log('bcap %s RESUMED' % cn)
        continue
    HB[cn] = capture_states_inject2(
        rows_cap, CAP_B, spec)
    log('bcap %s done' % cn)
    ck_save('bcap_%s' % cn, {
        'h': {str(l): HB[cn][l]
              .astype(np.float16)
              for l in CAP_B}})
# projection analysis
proj = {}
for cn in ('pc1', 'dv29', 'joint'):
    h29 = HB[cn][29]
    dh = (h29.astype(np.float64)
          - base_states[29]
          .astype(np.float64))
    dhj_ref = None
    proj[cn] = {
        'pc1_proj_med': float(np.median(
            h29.astype(np.float64) @ v1
            .astype(np.float64))),
        'wdn29_med': float(np.median(
            h29.astype(np.float64)
            @ w_dn_g.astype(
                np.float64))),
        'wdn38_med': float(np.median(
            HB[cn][38].astype(np.float64)
            @ w_dn_g.astype(
                np.float64))),
        'dh_wdn_med': float(np.median(
            dh @ w_dn_g.astype(
                np.float64))),
        'dh_v1_med': float(np.median(
            dh @ v1.astype(
                np.float64))),
        'dh_norm_med': float(np.median(
            np.linalg.norm(dh, axis=1)))}
# base references
b_pc1 = float(np.median(
    base_states[29].astype(np.float64)
    @ v1.astype(np.float64)))
b_wdn29 = float(np.median(
    base_states[29].astype(np.float64)
    @ w_dn_g.astype(np.float64)))
b_wdn38 = float(np.median(
    base_states[38].astype(np.float64)
    @ w_dn_g.astype(np.float64)))
# state additivity: dh_joint vs
# dh_pc1 + dh_dv29
dh_pc1 = (HB['pc1'][29].astype(np.float64)
          - base_states[29]
          .astype(np.float64))
dh_dv29 = (HB['dv29'][29].astype(
    np.float64)
    - base_states[29].astype(np.float64))
dh_joint = (HB['joint'][29].astype(
    np.float64)
    - base_states[29].astype(np.float64))
dh_sum = dh_pc1 + dh_dv29
stat_add_cos = float(np.median(
    _rowcos(dh_joint, dh_sum)))
# readout competition: joint dh.w_dn vs
# max single
wdn_pc1 = proj['pc1']['dh_wdn_med']
wdn_dv29 = proj['dv29']['dh_wdn_med']
wdn_joint = proj['joint']['dh_wdn_med']
comp_conf = wdn_joint <= max(
    wdn_pc1, wdn_dv29) + COMP_TOL
pc1_dark = abs(wdn_pc1) < \
    PC1_DARK_RATIO * abs(wdn_dv29)
stat_add = stat_add_cos >= STAT_ADD_COS
log('D2-SOFT: base pc1 %.3f wdn29 %.3f '
    'wdn38 %.3f | pc1_proj med pc1 %.3f '
    'dv29 %.3f joint %.3f | dh.wdn pc1 '
    '%.3f dv29 %.3f joint %.3f | '
    'stat_add cos %.3f -> comp %s / '
    'pc1_dark %s / stat %s'
    % (b_pc1, b_wdn29, b_wdn38,
       proj['pc1']['pc1_proj_med'],
       proj['dv29']['pc1_proj_med'],
       proj['joint']['pc1_proj_med'],
       wdn_pc1, wdn_dv29, wdn_joint,
       stat_add_cos,
       'confirmed' if comp_conf
       else 'absent',
       'dark' if pc1_dark else 'lit',
       'add' if stat_add
       else 'nonlin'))

# ================================================================
# PART D3: top-k enrichment sliding
# ================================================================
log('== PART D3: top-k sliding ==')
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
log('order_e equals 3141 high25/low25 '
    '(pure-numpy exact)')
C_TRIALS = []
for k in K_GRID:
    C_TRIALS.append(
        ('c_topk%02d_d1.0' % k,
         order_e[:k], E_DOSE))
C_TRIALS.append(('c_co36_d1.0', co36,
                 E_DOSE))
for (tname, coords, ds) in C_TRIALS:
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        continue
    gen_l = []
    for b0 in range(0, NCAP, GEN_BATCH):
        batch = rows_A1[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=[(17, coords,
                  ds * DELTA_L17, -1, 0)]))
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12_A1)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})
n_bitC = 0
bit_anchors_C = {}
if not SMOKE:
    for tn, v in (('c_co36_d1.0',
                   CO36_D1),
                  ('c_topk13_d1.0',
                   Q1_D1)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitC += int(m)
        bit_anchors_C[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('D3 repro: %d/2 bit-match '
        '(co36 full-50 vs 3137/3142; '
        'topk13 == 3142 Q1)' % n_bitC)
else:
    log('D3 repro SKIPPED (smoke)')
kchg = {k: E_res['c_topk%02d_d1.0' % k]
        ['chg'] for k in K_GRID}
co36_full = E_res['c_co36_d1.0']['chg']
k_star = None
for k in sorted(K_GRID):
    if kchg[k] >= HEAD_FRAC * co36_full:
        k_star = k
        break
head_tag = ('head_conc_k%02d' % k_star
            if k_star is not None
            else 'head_gradual')
log('D3-SOFT: k chg %s; co36 full %.4f; '
    'k* (>= %.0f%% full) = %s -> %s'
    % (json.dumps({str(k): round(v, 4)
                   for k, v in
                   kchg.items()}),
       co36_full, HEAD_FRAC * 100,
       k_star, head_tag))

# ================================================================
# PART D4: new-material I component
# ================================================================
log('== PART D4: new-material I ==')
used_pairs = set(pks)
new_pairs = []
for s in range(ENTS_N):
    for o in range(ENTS_N):
        if s == o:
            continue
        pk = '%d_%d' % (s, o)
        if pk not in used_pairs:
            new_pairs.append(pk)
new_pairs = sorted(new_pairs)
if SMOKE:
    new_pairs = new_pairs[:6]
XMAT_N = len(new_pairs)
log('new pairs: %d unseen (s,o) pairs; '
    'ents_n=%d (chance-s %.4f)'
    % (XMAT_N, ENTS_N, 1.0 / ENTS_N))


def build_prompt_new(s, o, lrel, qrel):
    rng3 = _rnd.Random(zlib.crc32(
        ('xmat|%d_%d_%d_%d'
         % (s, o, lrel, qrel))
        .encode('ascii')))
    lines = [(s, lrel, o)]
    seen = {(s, lrel, o)}
    while len(lines) < 8:
        a = rng3.randrange(ENTS_N)
        b = rng3.randrange(ENTS_N)
        rr = rng3.randrange(
            len(PREDS_all))
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
    rng3 = _rnd.Random(zlib.crc32(
        ('xmv|%d|%d_%d_%d_%d'
         % (vi, s, o, lrel, qrel))
        .encode('ascii')))
    lines = [(s, lrel, o)]
    seen = {(s, lrel, o)}
    while len(lines) < 8:
        a = rng3.randrange(ENTS_N)
        b = rng3.randrange(ENTS_N)
        rr = rng3.randrange(
            len(PREDS_all))
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
HN = {}
for key, rows_k in (('V1', rows_v1),
                    ('V0', rows_var[0]),
                    ('V1x', rows_var[1]),
                    ('V2', rows_var[2]),
                    ('V3', rows_var[3])):
    _K = CK['data'].get('ncap_%s' % key)
    if _K is not None:
        HN[key] = {int(l):
                   _K['h'][str(l)]
                   .astype(np.float32)
                   for l in RL}
        log('ncap %s RESUMED' % key)
        continue
    HN[key] = capture_states(rows_k, RL)
    log('ncap %s done (%d rows)'
        % (key, len(rows_k)))
    ck_save('ncap_%s' % key, {
        'h': {str(l): HN[key][l]
              .astype(np.float16)
              for l in RL}})
# IDEINT_new from 4 variants minus old
# bank layer base
I_new = {}
for l in RL:
    Hm = np.mean([HN[vk][l]
                  for vk in ('V0', 'V1x',
                             'V2', 'V3')],
                 axis=0)
    I_new[l] = (Hm - BANK_B[l][None, :]) \
        .astype(np.float32)


def retr_from_states(q, l, bank):
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


hit_self = {l: retr_from_states(
    HN['V1'][l], l, I_new[l]) for l in RL}
hit_cross = {l: retr_from_states(
    HN['V1'][l], l, IDEINT_P[l])
    for l in RL}
chance = 1.0 / ENTS_N
for l in RL:
    log('retr L%02d: self(I_new) %.4f | '
        'cross(IDEINT_P) %.4f | chance '
        '%.4f' % (l, hit_self[l],
                  hit_cross[l], chance))
n_bitD = 0
bit_anchors_D = {}
if not SMOKE:
    f1_bit = all(
        abs(hit_cross[l]
            - RETR_FWD_V1[str(l)]) < 1e-9
        for l in RL)
    if f1_bit:
        n_bitD += 1
    bit_anchors_D['f1_cross_retr'] = {
        'got': {str(l): hit_cross[l]
                for l in RL},
        'want': RETR_FWD_V1,
        'match': bool(f1_bit)}
    log('D4 f1 cross-retr bit anchor vs '
        '3142 retr_fwd_v1: %s' % f1_bit)
else:
    f1_bit = None
    log('D4 f1 bit anchor SKIPPED (smoke)')
self_mult = {l: hit_self[l] / chance
             for l in RL}
best_self = max(self_mult.values())
if best_self >= SELF_MULT:
    itag = 'iself_general'
elif max(hit_cross.values()) >= \
        SELF_MULT * chance:
    itag = 'iself_family'
else:
    itag = 'iself_session'
log('D4-SOFT: self mult %s; best %.2fx '
    'chance (gate %.1fx) -> %s'
    % (json.dumps({str(l): round(v, 2)
                   for l, v in
                   self_mult.items()}),
       best_self, SELF_MULT, itag))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3142_ok']
n_bit_all = n_bitB + n_bitC + \
    (n_bitD if not SMOKE else 0)
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == 5
                else 'repro_bit_drift')
else:
    tags.append('repro_smoke_skipped')
tags.append('dvec19_repro_%s'
            % dvec19_sha[:6])
tags.append('field_self_ok'
            if field_self_ok
            else 'field_self_drift')
tags.append('z35_cos17_ok' if z35_ok
            else 'z35_cos17_drift')
tags.append(path_tag)
tags.append('d19_direct_hi'
            if d19_direct_hi
            else 'd19_direct_lo')
tags.append('stat_add' if stat_add
            else 'stat_nonlin')
tags.append('readout_comp_confirmed'
            if comp_conf
            else 'readout_comp_absent')
tags.append('pc1_readout_dark'
            if pc1_dark
            else 'pc1_readout_lit')
tags.append(head_tag)
tags.append(itag)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3143,
    'name': NAME,
    'smoke': SMOKE,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'runtime_s': time.time() - T0,
    'seal_sha8': hashlib.sha256(
        json.dumps(SEAL, sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8],
    'verdict': verdict,
    'constants': SEAL['constants'],
    'part_a': {
        'res42_sha8': sha42,
        'seal42': SEAL42,
        'xphase_P': xphase_P,
        'xphase_A1': xphase_A1},
    'part_field': {
        'dvec19_sha8': dvec19_sha,
        'dvec19_sha_ok': bool(d19_sha_ok),
        'dvec19_medn': dvec19_medn,
        'field17_vs_dvec17_cos': fs17,
        'field19_vs_dvec19_cos': fs19,
        'field_self_ok':
            bool(field_self_ok),
        'z35_c17_maxdiff': z35_diff,
        'z35_ok': bool(z35_ok)},
    'part_cond': {
        'spec_corr': spec_corr,
        'cross_med': {str(r): cross_med[r]
                      for r in
                      sorted(cross_med)},
        'cross_near': cross_near,
        'cos19_med': {str(r):
                      float(np.nanmedian(
                          cos19[r]))
                      for r in range(
                          20, 40)},
        'cos17_med': {str(r):
                      float(np.nanmedian(
                          cos17[r]))
                      for r in range(
                          20, 40)},
        'rho19_med': {str(r):
                      float(np.nanmedian(
                          rho19[r]))
                      for r in range(
                          20, 40)},
        'rho17_med': {str(r):
                      float(np.nanmedian(
                          rho17[r]))
                      for r in range(
                          20, 40)},
        'd19_direct_cos': d19_direct,
        'd19_direct_rho': d19_direct_rho,
        'd17_direct_cos': d17_direct,
        'path_tag': path_tag,
        'd19_direct_hi':
            bool(d19_direct_hi)},
    'part_readout': {
        'pc1_sign': pc1_sign,
        'bit_anchors_3142': bit_anchors_B,
        'proj': proj,
        'base': {'pc1_proj': b_pc1,
                 'wdn29': b_wdn29,
                 'wdn38': b_wdn38},
        'stat_add_cos': stat_add_cos,
        'stat_add': bool(stat_add),
        'comp_confirmed': bool(comp_conf),
        'pc1_dark': bool(pc1_dark)},
    'part_topk': {
        'bit_anchors': bit_anchors_C,
        'kchg': {str(k): kchg[k]
                 for k in K_GRID},
        'co36_full': co36_full,
        'k_star': k_star,
        'head_tag': head_tag},
    'part_newI': {
        'n_pairs': XMAT_N,
        'ents_n': ENTS_N,
        'chance': chance,
        'hit_self': {str(l): hit_self[l]
                     for l in RL},
        'hit_cross': {str(l):
                      hit_cross[l]
                      for l in RL},
        'self_mult': {str(l):
                      self_mult[l]
                      for l in RL},
        'bit_anchors': bit_anchors_D,
        'itag': itag}}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'cos19': cos19, 'rho19': rho19,
    'cos17': cos17, 'rho17': rho17,
    'cross_cos': np.array(
        [cross_med[r]
         for r in sorted(cross_med)]),
    'cross_layers': np.array(
        sorted(cross_med)),
    'dvec19_sha': np.array([dvec19_sha]),
    'kchg': np.array([kchg[k]
                      for k in
                      sorted(K_GRID)]),
    'kgrid': np.array(sorted(K_GRID))}
npz_out['dvec19_out'] = dvec19.astype(
    np.float16)
np.savez(os.path.join(OUT,
                      'p141_readout.npz'),
         **npz_out)
with io.open(os.path.join(OUT, 'p141_'
                          'ckpt.pkl'),
             'wb') as f:
    pass
if not SMOKE:
    try:
        os.remove(CKPTF)
        log('ckpt cleaned (final)')
    except OSError:
        pass
log('result.json + npz written. DONE '
    '(verdict %s)' % verdict)
