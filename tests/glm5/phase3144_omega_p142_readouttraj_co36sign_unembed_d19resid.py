# -*- coding: utf-8 -*-
"""Phase 3144 (Omega-P142): high-order
readout competition localization (L29-39
per-layer w_dn trajectories under joint +
logit-level decomposition of the
interfering token) + co36 positive/negative
coordinate separation (k25 head set vs
rank 30-50 tail set, crossed with 3137
co36_rank top/bot25 and co50 extras +
tail sign-flip injection) + old-material
identity unembed tracing (is IDEINT_P
aligned to entity-token output geometry?)
+ dh19 residual field (dh19 minus its
projection on the dh17 pathway: where does
the independent component go?).

Preregistered in 3143 closeout (aligned
with FINGERPRINT_PARADIGM_PLAN.md
Omega-P142):
(1) readout trajectory: pc1/dv29/joint
injections at L29 (3142 D semantics,
P rows, dose 2.0 vectors) captured on
L29-39 -> per-layer dh.w_dn + dh.v1 +
dh.norm trajectories; blocking layer =
argmax gap(wdn_dv29 - wdn_joint) if >
GAP_TOL; 3 generation bit anchors
(b_pc1 0.203125 / b_dvec29 0.640625 /
b_joint 0.4765625 vs 3142) + interfering
token classification (answer vs format)
+ teacher-forced logit decomposition on
diverged rows (pc1 condition);
(2) co36 sign separation: HEAD25/TAIL25
(3143 order_e == 3141 high25/low25
asserted) vs 3137 co36_rank top25/bot25
(exact lists asserted) overlap analysis +
enrichment-score stats + injections
@L17 A1 dose 1.0 (3137 G semantics,
sgn -1, mode 0): d_head (bit anchor ==
k25 0.21875), d_tail, d_tailpos (sgn
+1), d_gbot (bit anchor == 3137 g_bot25
0.1171875), d_co36full (bit anchor ==
0.140625), d_co50ex;
(3) unembed tracing: entity vectors =
mean unembed rows of ' '+entity tokens;
self vs object vs random cos against
IDEINT_P[17] rows; top1 logit-lens token
hit-in-entities rate; top-25 coordinate
energy share;
(4) dh19 residual: field rebuild (128
rows base/swap4, self-consistency gates
fs17/fs19 >= 0.99, z35 cos_17 soft
anchor) + cond19/cond17 captures (dose
2.0) -> per-layer row-level projection
dh19_perp = dh19 - proj_dh17(dh19);
perp share, resid-field cos spectrum,
perp@w_dn at L38/39, perp top-PC
alignment -> destination verdict.

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
NAME = ('omega_p142_readouttraj_'
        'co36sign_unembed_d19resid')
SMOKE = os.environ.get('P3144_SMOKE',
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
D43 = RDIR + r'\phase3143' \
      r'\omega_p141_d19field_readout_' \
      r'topk_newI'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3144', NAME)
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
CKPTF = os.path.join(OUT, 'p142_ckpt.pkl')
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
NCAP = 4 if SMOKE else 128      # field/cond/traj rows
N19_ROWS = 4 if SMOKE else NP   # dvec19 capture rows
GEN_BATCH = 4 if SMOKE else 32
B3_ROWS = 4 if SMOKE else 64    # logit-decomp rows
DOSE_COND = 2.0                 # 3135 B2 semantics
DOSE_C = 1.0                    # coordinate injections
DEC_L = (17, 29, 38)            # bank decomposition layers
TRAJ_L = tuple(range(29, 40))   # trajectory capture layers
FIELD_SELF_COS = 0.99
Z35_TOL = 0.08
GAP_TOL = 0.15
TAIL_TOL = 0.03
FORMAT_GATE = 0.5
UNEMB_MARGIN = 0.02
PERP_DEAD = 0.15
READOUT_FRAC = 0.3
N_RAND_ENT = 20
RNG_SEED = 3144

# ---------------- frozen 3143 anchors --
EXP_V43 = ('a_3142_ok|repro_bit_5|'
           'repro_bit_ok|dvec19_repro_6a0332|'
           'field_self_ok|z35_cos17_ok|'
           'd19_path_partial|d19_direct_hi|'
           'stat_add|readout_comp_absent|'
           'pc1_readout_dark|head_conc_k13|'
           'iself_session|xphase_ok|'
           'coverage_full')
RES43_SHA = 'ce348ff5'
SEAL43 = 'e7f52d03'
XPHASE43 = 1.0
DVEC19_SHA = '6a0332a6'
MEDNORM19 = 5.643608093261719
STAT_ADD_43 = 0.9999981374
PC1_SIGN_43 = 1
DH_WDN_43 = {'pc1': 0.219909273320809,
             'dv29': 6.507396821863949,
             'joint': 6.725984403863549}
DH_V1_43 = {'pc1': 10.48045490672731,
            'dv29': -20.869472232950635,
            'joint': -10.39192377088787}
KCHG_43 = {'5': 0.0859375,
           '10': 0.109375,
           '13': 0.15625,
           '15': 0.1328125,
           '20': 0.1640625,
           '25': 0.21875,
           '30': 0.2109375}
CO36FULL_43 = 0.140625
SPEC_CORR_43 = 0.881203
CROSS_NEAR_43 = 0.777693
D19_DIRECT_43 = 0.854632
# ---------------- frozen 3142/3137 ------
PC1_D2 = 0.203125
DVEC29_D2 = 0.640625
JOINT_D2 = 0.4765625
K25_CHG = 0.21875
GB25_37 = 0.1171875
GTOP25_37 = 0.1328125
CO36_D1 = 0.140625
DELTA_L17 = 0.637683315669971
HIGH25_43 = [2319, 83, 2309, 3140, 3099,
             2491, 2097, 999, 2702, 1939,
             1235, 2617, 161, 3426, 1877,
             551, 3301, 2011, 2560, 1008,
             2605, 78, 1357, 3818, 207]
LOW25_43 = [381, 1179, 1004, 1578, 2911,
            1626, 1973, 642, 1284, 2442,
            1306, 2890, 2664, 317, 1757,
            2185, 1991, 3131, 3756, 294,
            1634, 807, 1059, 3499, 3432]
G_TOP25 = [2319, 642, 1634, 2097, 1008,
           1357, 2664, 2617, 83, 294,
           3818, 1877, 3499, 2011, 3301,
           161, 317, 999, 3099, 1235,
           807, 1626, 2309, 381, 1939]
G_BOT25 = [3432, 3426, 2911, 3131, 3140,
           1004, 1578, 1284, 2185, 1973,
           2442, 1059, 551, 207, 2702,
           3756, 78, 2560, 1306, 1179,
           1757, 2890, 2491, 1991, 2605]
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
LEDGER_N = 280

# ---------------- seal ------------------
SEAL = {
    'phase': 3144,
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
        'B3_ROWS': B3_ROWS,
        'DOSE_COND': DOSE_COND,
        'DOSE_C': DOSE_C,
        'DEC_L': list(DEC_L),
        'TRAJ_L': list(TRAJ_L),
        'FIELD_SELF_COS': FIELD_SELF_COS,
        'Z35_TOL': Z35_TOL,
        'GAP_TOL': GAP_TOL,
        'TAIL_TOL': TAIL_TOL,
        'FORMAT_GATE': FORMAT_GATE,
        'UNEMB_MARGIN': UNEMB_MARGIN,
        'PERP_DEAD': PERP_DEAD,
        'READOUT_FRAC': READOUT_FRAC,
        'N_RAND_ENT': N_RAND_ENT,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res43_verdict': EXP_V43,
        'res43_sha8': RES43_SHA,
        'seal43': SEAL43,
        'xphase43': XPHASE43,
        'dvec19_sha8': DVEC19_SHA,
        'mednorm19': MEDNORM19,
        'stat_add_43': STAT_ADD_43,
        'pc1_sign_43': PC1_SIGN_43,
        'dh_wdn_43': DH_WDN_43,
        'dh_v1_43': DH_V1_43,
        'kchg_43': KCHG_43,
        'co36full_43': CO36FULL_43,
        'spec_corr_43': SPEC_CORR_43,
        'cross_near_43': CROSS_NEAR_43,
        'd19_direct_43': D19_DIRECT_43,
        'pc1_d2': PC1_D2,
        'dvec29_d2': DVEC29_D2,
        'joint_d2': JOINT_D2,
        'k25_chg': K25_CHG,
        'gb25_37': GB25_37,
        'gtop25_37': GTOP25_37,
        'co36_d1': CO36_D1,
        'delta_l17': DELTA_L17,
        'high25_43': HIGH25_43,
        'low25_43': LOW25_43,
        'g_top25': G_TOP25,
        'g_bot25': G_BOT25,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3143 closeout + '
               'FINGERPRINT_PARADIGM_PLAN '
               'Omega-P142: (1) readout '
               'trajectory: pc1/dv29/joint '
               '@L29 (3142 D semantics, P '
               'rows) captured L29-39 -> '
               'per-layer dh.w_dn/v1/norm '
               'trajectories, blocking layer '
               '= argmax gap(wdn_dv29-'
               'wdn_joint); 3 gen bit anchors '
               '(pc1/dvec29/joint vs 3142) + '
               'interfering-token class + '
               'teacher-forced logit '
               'decomposition (pc1 rows); '
               '(2) co36 sign: HEAD25/TAIL25 '
               '(order_e) vs 3137 co36_rank '
               'top25/bot25 overlap + '
               'injections @L17 A1 d1.0 sgn '
               '-1 (d_head==k25 bit anchor, '
               'd_tail, d_tailpos sgn +1, '
               'd_gbot==3137 g_bot25 bit '
               'anchor, d_co36full==0.140625 '
               'bit anchor, d_co50ex); '
               '(3) unembed tracing: entity '
               'vectors mean unembed rows, '
               'self/obj/rand cos vs '
               'IDEINT_P[17], top1 logit-lens '
               'hit rate, top-25 energy '
               'share; (4) dh19 residual: '
               'field rebuild + cond19/17 '
               'captures (dose 2.0) -> '
               'row-level dh19_perp, perp '
               'share / resid-field cos / '
               'perp@w_dn L38-39 / top-PC '
               'alignment -> destination. '
               'Bank reused 3138. Frozen '
               'before observation.')}
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
G_ORDER = z35['co36_rank'].astype(np.int64)
assert len(G_ORDER) == 50
assert list(G_ORDER[:25]) == G_TOP25, \
    'co36_rank top25 drift vs 3137'
assert list(G_ORDER[25:]) == G_BOT25, \
    'co36_rank bot25 drift vs 3137'
log('co36_rank order exact == 3137 G '
    'top25/bot25 lists')
z136 = np.load(D36 + r'\p134_readout.npz',
               allow_pickle=False)
CO_SETS = {}
for cn in ('co36', 'co50', 'union'):
    _c = np.asarray(z136[cn]).astype(np.int64)
    assert _c.ndim == 1, (cn, _c.shape)
    assert _c.min() >= 0 and _c.max() < 4096
    CO_SETS[cn] = _c
assert len(CO_SETS['co36']) == 50
assert set(G_ORDER.tolist()) == \
    set(CO_SETS['co36'].tolist()), \
    'co36_rank set != co36 set'
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
assert len(led['measurements']) == LEDGER_N
log('frozen inputs ok (26/35/36 + ledger '
    'n=%d)' % LEDGER_N)

# ================================================================
# PART A: 3143 link asserts
# ================================================================
log('== PART A: link asserts ==')
res43 = json.load(io.open(
    D43 + r'\result.json', encoding='utf-8'))
assert res43['smoke'] is False
V43 = res43['verdict']
assert V43 == EXP_V43, V43
raw43 = io.open(
    D43 + r'\result.json', 'rb').read()
sha43 = hashlib.sha256(raw43).hexdigest()[:8]
assert sha43 == RES43_SHA, sha43
assert str(res43['seal_sha8']) == SEAL43
pa43 = res43['part_a']
assert abs(float(pa43['xphase_P'])
           - XPHASE43) < 1e-12
assert abs(float(pa43['xphase_A1'])
           - XPHASE43) < 1e-12
pf43 = res43['part_field']
assert str(pf43['dvec19_sha8']) == DVEC19_SHA
assert abs(float(pf43['dvec19_medn'])
           - MEDNORM19) < 1e-9
assert abs(float(pf43['field17_vs_dvec17_cos'])
           - 1.0) < 1e-12
assert abs(float(pf43['field19_vs_dvec19_cos'])
           - 1.0) < 1e-12
assert float(pf43['z35_c17_maxdiff']) < Z35_TOL
pcn43 = res43['part_cond']
assert abs(float(pcn43['spec_corr'])
           - SPEC_CORR_43) < 1e-4
assert abs(float(pcn43['cross_near'])
           - CROSS_NEAR_43) < 1e-4
assert abs(float(pcn43['d19_direct_cos'])
           - D19_DIRECT_43) < 1e-4
assert str(pcn43['path_tag']) == \
    'd19_path_partial'
pr43 = res43['part_readout']
assert int(pr43['pc1_sign']) == PC1_SIGN_43
assert abs(float(pr43['stat_add_cos'])
           - STAT_ADD_43) < 1e-6
assert pr43['stat_add'] is True
assert pr43['comp_confirmed'] is False
assert pr43['pc1_dark'] is True
for cn in ('pc1', 'dv29', 'joint'):
    got_w = float(pr43['proj'][cn]
                  ['dh_wdn_med'])
    assert abs(got_w - DH_WDN_43[cn]) < 1e-6, \
        (cn, got_w)
    got_v = float(pr43['proj'][cn]
                  ['dh_v1_med'])
    assert abs(got_v - DH_V1_43[cn]) < 1e-6, \
        (cn, got_v)
assert all(v['match']
           for v in
           pr43['bit_anchors_3142'].values())
pt43 = res43['part_topk']
for k in KCHG_43:
    got = float(pt43['kchg'][k])
    assert abs(got - KCHG_43[k]) < 1e-9, \
        (k, got)
assert abs(float(pt43['co36_full'])
           - CO36FULL_43) < 1e-9
assert int(pt43['k_star']) == 13
assert all(v['match']
           for v in pt43['bit_anchors'].values())
pn43 = res43['part_newI']
assert str(pn43['itag']) == 'iself_session'
chance43 = float(pn43['chance'])
for l in ('17', '29', '38'):
    assert abs(float(pn43['hit_self'][l])
               - chance43) < 1e-9
nb43 = 0
for v in pr43['bit_anchors_3142'].values():
    nb43 += int(v['match'])
for v in pt43['bit_anchors'].values():
    nb43 += int(v['match'])
for v in pn43['bit_anchors'].values():
    nb43 += int(v['match'])
assert 'repro_bit_5' in V43 and nb43 == 5
log('A hard asserts ok (3143 sha8 %s seal '
    '%s: xphase/dvec19/field/cond/readout/'
    'topk/newI anchors; 5-bit chain '
    'verified)' % (sha43, SEAL43))

# ================================================================
# materials factory (3142 lineage)
# ================================================================
p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
pks = sorted(p2r.keys())
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
VOCAB = int(WUG.shape[0])
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


N_NEW = 12


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
                  'cond19', 'cond17',
                  'traj', 'tfbase', 'tfinj')
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
# PART C3: full-field rebuild (128 rows)
# ================================================================
log('== PART C3: field rebuild ==')
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

# ================================================================
# PART B1: generation bit anchors + pc1
# sign resolution + interfering-token
# classification
# ================================================================
log('== PART B1: gen anchors + sign ==')
v1 = PC[29]['V'][0]
mn29 = PC[29]['mnorm']
dv_pc1_pos = np.tile(
    (v1 * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_pc1_neg = -dv_pc1_pos
dv_pc1 = dv_pc1_pos
dv_dv29 = (dvec[29][:NCAP] * 2.0) \
    .astype(np.float32)
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
                    'first': int(first_l),
                    'gens': gen_l}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})


_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P)
pc1_sign = None
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
    log('B1 pc1 sign resolved: -1')
elif pc1_sign == 1:
    log('B1 pc1 sign resolved: +1')
_run_vec_trial('b_dvec29_l29_d2.0', 29,
               dv_dv29, 1.0, 'allstep',
               rows_scan, base12_P)
_KJ = CK['data'].get('b_joint_l29_d2.0')
if _KJ is None:
    gen_l = []
    for b0 in range(0, NCAP, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(29, dv_pc1[b0:b0
                                + len(batch)],
                      1.0, 'allstep'),
                     (29, dv_dv29[b0:b0
                                  + len(batch)],
                      1.0, 'allstep')]))
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12_P)
    E_res['b_joint_l29_d2.0'] = {
        'chg': chg_l, 'first': int(first_l),
        'gens': gen_l}
    log('b_joint_l29_d2.0: chg=%.4f '
        'first=%d' % (chg_l, first_l))
    ck_save('b_joint_l29_d2.0', {
        'res': E_res['b_joint_l29_d2.0']})
else:
    E_res['b_joint_l29_d2.0'] = _KJ['res']
    log('b_joint_l29_d2.0 RESUMED '
        'chg=%.4f'
        % E_res['b_joint_l29_d2.0']['chg'])
n_bitB = 0
bit_anchors_B = {}
if not SMOKE:
    for tn, v in (('b_pc1_l29_d2.0',
                   PC1_D2),
                  ('b_dvec29_l29_d2.0',
                   DVEC29_D2),
                  ('b_joint_l29_d2.0',
                   JOINT_D2)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitB += int(m)
        bit_anchors_B[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('B1 3142 repro: %d/3 bit-match'
        % n_bitB)
else:
    log('B1 3142 repro SKIPPED (smoke)')


def tok_class(tid):
    if tid in (YES_G, NO_G):
        return 'answer'
    if tid == DOT_G:
        return 'format'
    return 'other'


tokcls = {}
for tn in ('b_pc1_l29_d2.0',
           'b_dvec29_l29_d2.0',
           'b_joint_l29_d2.0'):
    gens = E_res[tn]['gens']
    counts = {'answer': 0, 'format': 0,
              'other': 0, 'none': 0}
    tstars = []
    for j in range(len(gens)):
        g12 = pad12(gens[j])
        b12 = base12_P[j]
        tstar = -1
        for t in range(N_NEW):
            if g12[t] != b12[t]:
                tstar = t
                break
        if tstar < 0:
            counts['none'] += 1
            continue
        tstars.append(tstar)
        counts[tok_class(g12[tstar])] += 1
    nint = sum(counts[k] for k in
               ('answer', 'format',
                'other'))
    tokcls[tn] = {
        'counts': counts,
        'answer_frac':
            counts['answer'] / max(nint, 1),
        'format_frac':
            counts['format'] / max(nint, 1),
        'med_tstar':
            float(np.median(tstars))
            if tstars else -1.0}
    log('B1 tokcls %s: %s answer %.3f '
        'format %.3f med_t* %s'
        % (tn, json.dumps(counts),
           tokcls[tn]['answer_frac'],
           tokcls[tn]['format_frac'],
           tokcls[tn]['med_tstar']))
_af = tokcls['b_pc1_l29_d2.0']['answer_frac']
_ff = tokcls['b_pc1_l29_d2.0']['format_frac']
if _af > FORMAT_GATE:
    pc1_channel = 'pc1_channel_answer'
elif _ff > FORMAT_GATE:
    pc1_channel = 'pc1_channel_format'
else:
    pc1_channel = 'pc1_channel_mixed'
log('B1-GATE: pc1 interfering-token '
    'channel -> %s' % pc1_channel)

# ================================================================
# PART B2: readout trajectory captures
# (pc1/dv29/joint @L29, layers 29-39)
# ================================================================
log('== PART B2: readout trajectory ==')
_traj_specs = {
    'pc1': [(29, dv_pc1, 1.0)],
    'dv29': [(29, dv_dv29, 1.0)],
    'joint': [(29, dv_pc1, 1.0),
              (29, dv_dv29, 1.0)]}
HT = {}
for cn, spec in _traj_specs.items():
    _K = CK['data'].get('traj_%s' % cn)
    if _K is not None:
        HT[cn] = {int(k):
                  v.astype(np.float32)
                  for k, v in
                  _K['h'].items()}
        log('traj %s RESUMED' % cn)
        continue
    HT[cn] = capture_states_inject2(
        rows_cap, list(TRAJ_L), spec)
    log('traj %s done (%d layers)'
        % (cn, len(TRAJ_L)))
    ck_save('traj_%s' % cn, {
        'h': {str(l): HT[cn][l]
              .astype(np.float16)
              for l in TRAJ_L}})
traj = {}
for cn in ('pc1', 'dv29', 'joint'):
    traj[cn] = {}
    for r in TRAJ_L:
        dh = (HT[cn][r].astype(np.float64)
              - base_states[r]
              .astype(np.float64))
        traj[cn][r] = {
            'wdn': float(np.median(
                dh @ w_dn_g.astype(
                    np.float64))),
            'v1': float(np.median(
                dh @ v1.astype(
                    np.float64))),
            'norm': float(np.median(
                np.linalg.norm(dh,
                               axis=1)))}
gap = {r: traj['dv29'][r]['wdn']
       - traj['joint'][r]['wdn']
       for r in TRAJ_L}
max_gap = max(gap.values())
blk_layer = max(gap, key=lambda r: gap[r])
traj_tag = ('traj_located_l%02d'
            % blk_layer
            if max_gap > GAP_TOL
            else 'traj_gradual')
log('B2-SOFT: w_dn traj pc1 %s' %
    json.dumps({str(r): round(
        traj['pc1'][r]['wdn'], 3)
        for r in TRAJ_L}))
log('B2-SOFT: w_dn traj dv29 %s' %
    json.dumps({str(r): round(
        traj['dv29'][r]['wdn'], 3)
        for r in TRAJ_L}))
log('B2-SOFT: w_dn traj joint %s' %
    json.dumps({str(r): round(
        traj['joint'][r]['wdn'], 3)
        for r in TRAJ_L}))
log('B2-SOFT: v1 traj joint %s' %
    json.dumps({str(r): round(
        traj['joint'][r]['v1'], 3)
        for r in TRAJ_L}))
log('B2-GATE: gap(dv29-joint) max %.4f at '
    'L%02d (tol %.2f) -> %s'
    % (max_gap, blk_layer, GAP_TOL,
       traj_tag))

# ================================================================
# PART B3: teacher-forced logit
# decomposition (pc1 condition, diverged
# rows)
# ================================================================
log('== PART B3: logit decomposition ==')
gens_pc1 = E_res['b_pc1_l29_d2.0']['gens']
div_rows = []
for j in range(NCAP):
    g12 = pad12(gens_pc1[j])
    if g12 != base12_P[j]:
        div_rows.append(j)
    if len(div_rows) >= B3_ROWS:
        break
B3N = len(div_rows)
log('B3: %d diverged rows (cap %d)'
    % (B3N, B3_ROWS))
tf_rows = []
tstar_tok = []
for j in div_rows:
    tstar = -1
    g12 = pad12(gens_pc1[j])
    for t in range(N_NEW):
        if g12[t] != base12_P[j][t]:
            tstar = t
            break
    assert tstar >= 0
    tf_ids = (list(PREFIX_IDS)
              + list(rows_scan[j])
              + list(base12_P[j][:tstar]))
    tf_rows.append(tf_ids)
    tstar_tok.append(int(base12_P[j][tstar]))
_KT = CK['data'].get('tfbase')
if _KT is not None:
    h39_base = _KT['h'].astype(np.float32)
    log('tfbase RESUMED')
else:
    h39_base = capture_states(
        tf_rows, [39])[39]
    ck_save('tfbase', {
        'h': h39_base.astype(np.float16)})
_KI = CK['data'].get('tfinj')
if _KI is not None:
    h39_inj = _KI['h'].astype(np.float32)
    log('tfinj RESUMED')
else:
    h39_inj = capture_states_inject2(
        tf_rows, [39],
        [(29, dv_pc1[:B3N], 1.0)])[39]
    ck_save('tfinj', {
        'h': h39_inj.astype(np.float16)})
dh39 = (h39_inj.astype(np.float64)
        - h39_base.astype(np.float64))


def chunked_logit(DH, topk=5):
    """DH (n,4096) @ WUG.T in chunks ->
    per-row global argmax id + topk
    (ids, logits)."""
    n = DH.shape[0]
    W = WUG.detach()
    best_val = np.full(n, -np.inf)
    am = np.zeros(n, dtype=np.int64)
    top_ids = np.zeros((n, topk),
                       dtype=np.int64)
    top_vals = np.full((n, topk),
                       -np.inf)
    CH = 16384
    for s0 in range(0, VOCAB, CH):
        s1 = min(s0 + CH, VOCAB)
        Wc = W[s0:s1].to(torch.float32) \
            .cpu().numpy() \
            .astype(np.float64)
        LG = DH @ Wc.T
        mx = LG.max(axis=1)
        upd = mx > best_val
        am[upd] = s0 + np.argmax(
            LG, axis=1)[upd]
        best_val[upd] = mx[upd]
        pids = np.argsort(
            -LG, axis=1)[:, :topk]
        pvals = np.take_along_axis(
            LG, pids, axis=1)
        allv = np.concatenate(
            [top_vals, pvals], axis=1)
        alli = np.concatenate(
            [top_ids, pids + s0], axis=1)
        sel = np.argsort(
            -allv, axis=1)[:, :topk]
        top_vals = np.take_along_axis(
            allv, sel, axis=1)
        top_ids = np.take_along_axis(
            alli, sel, axis=1)
    return am, top_ids, top_vals


am39, top5_ids, top5_vals = \
    chunked_logit(dh39, 5)
dlogit_star = np.array([
    dh39[i] @ WUG[int(tstar_tok[i])]
    .to(torch.float32).cpu().numpy()
    .astype(np.float64)
    for i in range(B3N)])
wdn39 = dh39 @ w_dn_g.astype(np.float64)
cls_top1 = [tok_class(int(t))
            for t in am39]
cls_star = [tok_class(int(t))
            for t in tstar_tok]
b3 = {
    'n_rows': B3N,
    'med_dlogit_star':
        float(np.median(dlogit_star)),
    'frac_dlogit_star_pos':
        float(np.mean(dlogit_star > 0)),
    'star_cls_counts': {
        c: int(cls_star.count(c))
        for c in ('answer', 'format',
                  'other')},
    'top1d_cls_counts': {
        c: int(cls_top1.count(c))
        for c in ('answer', 'format',
                  'other')},
    'med_dh_wdn39':
        float(np.median(wdn39)),
    'top1d_token_ids':
        [int(t) for t in am39[:8].tolist()]}
log('B3-SOFT: med dlogit(t*) %.3f '
    'frac>0 %.2f | star cls %s | top1d '
    'cls %s | med dh.wdn39 %.3f'
    % (b3['med_dlogit_star'],
       b3['frac_dlogit_star_pos'],
       json.dumps(b3['star_cls_counts']),
       json.dumps(b3['top1d_cls_counts']),
       b3['med_dh_wdn39']))

# ================================================================
# PART D: co36 positive/negative separation
# ================================================================
log('== PART D: co36 sign separation ==')
I17 = IDEINT_P[17]
fr = I17 ** 2 / np.maximum(
    np.sum(I17 ** 2, axis=1,
           keepdims=True), 1e-12)
co36 = CO_SETS['co36']
assert len(co36) == 50
score = np.mean(fr[:, co36], axis=0)
order_e = co36[np.argsort(-score)]
assert list(order_e[:25]) == HIGH25_43, \
    'order_e drift vs 3141 high25'
assert list(order_e[25:]) == LOW25_43, \
    'order_e drift vs 3141 low25'
log('order_e equals 3141 high25/low25 '
    '(pure-numpy exact)')
HEAD25 = list(order_e[:25])
TAIL25 = list(order_e[25:])
GT25 = list(G_TOP25)
GB25 = list(G_BOT25)
co50 = CO_SETS['co50']
co50ex = sorted(set(co50.tolist())
                - set(co36.tolist()))
head_gt = sorted(set(HEAD25)
                 & set(GT25))
head_gb = sorted(set(HEAD25)
                 & set(GB25))
tail_gt = sorted(set(TAIL25)
                 & set(GT25))
tail_gb = sorted(set(TAIL25)
                 & set(GB25))
jac_hg = len(head_gt) / float(
    len(set(HEAD25) | set(GT25)))
score_by_coord = {int(c): float(score[i])
                  for i, c in
                  enumerate(co36)}
head_score = [score_by_coord[c]
              for c in HEAD25]
tail_score = [score_by_coord[c]
              for c in TAIL25]
# rank correlation between the two
# orderings restricted to co36
rank_e = rankdata_avg(
    [-score_by_coord[int(c)]
     for c in co36])
rank_g = rankdata_avg(
    [float(np.where(G_ORDER == int(c))[0][0])
     for c in co36])
order_rho = spearman(rank_e, rank_g)
ident = {
    'n_head_gt': len(head_gt),
    'n_head_gb': len(head_gb),
    'n_tail_gt': len(tail_gt),
    'n_tail_gb': len(tail_gb),
    'jaccard_head_gt': jac_hg,
    'co50ex_n': len(co50ex),
    'co50ex': co50ex,
    'head_score_mean':
        float(np.mean(head_score)),
    'tail_score_mean':
        float(np.mean(tail_score)),
    'head_score_med':
        float(np.median(head_score)),
    'tail_score_med':
        float(np.median(tail_score)),
    'order_rho_co36': order_rho}
log('D-ID: |HEAD∩GT|=%d |HEAD∩GB|=%d '
    '|TAIL∩GT|=%d |TAIL∩GB|=%d '
    'J(HEAD,GT)=%.3f co50ex=%d score '
    'head_med %.5f tail_med %.5f '
    'order_rho %.3f'
    % (ident['n_head_gt'],
       ident['n_head_gb'],
       ident['n_tail_gt'],
       ident['n_tail_gb'],
       jac_hg, len(co50ex),
       ident['head_score_med'],
       ident['tail_score_med'],
       order_rho))
D_TRIALS = [('d_head_d1.0', HEAD25, -1),
            ('d_tail_d1.0', TAIL25, -1),
            ('d_tailpos_d1.0', TAIL25, 1),
            ('d_gbot_d1.0', GB25, -1),
            ('d_co36full_d1.0',
             [int(c) for c in co36], -1),
            ('d_co50ex_d1.0', co50ex, -1)]
for (tname, coords, sgn) in D_TRIALS:
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
                  DOSE_C * DELTA_L17, sgn,
                  0)]))
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12_A1)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})
n_bitD = 0
bit_anchors_D = {}
if not SMOKE:
    for tn, v in (('d_head_d1.0',
                   K25_CHG),
                  ('d_gbot_d1.0',
                   GB25_37),
                  ('d_co36full_d1.0',
                   CO36_D1)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitD += int(m)
        bit_anchors_D[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('D repro: %d/3 bit-match '
        '(head==k25 3143; gbot==3137; '
        'co36full triple-anchored)' % n_bitD)
else:
    log('D repro SKIPPED (smoke)')
d_head = E_res['d_head_d1.0']['chg']
d_tail = E_res['d_tail_d1.0']['chg']
d_tailpos = E_res['d_tailpos_d1.0']['chg']
tail_neg = d_tail < d_head - TAIL_TOL
sign_asym = abs(d_tailpos - d_tail) \
    > TAIL_TOL
tail_tag = ('tail_neg_confirmed'
            if tail_neg
            else 'tail_neg_absent')
sign_tag = ('tail_sign_asym'
            if sign_asym
            else 'tail_sign_sym')
log('D-GATE: head %.4f tail %.4f '
    'tailpos %.4f (tol %.2f) -> %s / %s'
    % (d_head, d_tail, d_tailpos,
       TAIL_TOL, tail_tag, sign_tag))

# ================================================================
# PART E: identity unembed tracing
# ================================================================
log('== PART E: unembed tracing ==')
ent_tokens = {}
for s in range(ENTS_N):
    ids_s = tok_g(' ' + ents_all[s],
                  add_special_tokens=False)
    ent_tokens[s] = [int(t) for t in
                     ids_s['input_ids']]
E_vec = np.zeros((ENTS_N, HIDG),
                 dtype=np.float64)
for s in range(ENTS_N):
    rows = [WUG[t].to(torch.float32)
            .cpu().numpy().astype(np.float64)
            for t in ent_tokens[s]]
    E_vec[s] = np.mean(rows, axis=0)
ip17 = IDEINT_P[17].astype(np.float64)
rng_e = _rnd.Random(RNG_SEED)
self_cos = np.zeros(NP)
obj_cos = np.zeros(NP)
rand_cos = np.zeros(NP)
ent_id_set = set()
for s in range(ENTS_N):
    ent_id_set.update(ent_tokens[s])
ip17_top1, _, _ = chunked_logit(
    ip17, 1)
hit_top1 = 0
for i, pk in enumerate(pks):
    (s, o) = (int(v)
              for v in pk.split('_'))
    self_cos[i] = float(ip17[i] @ E_vec[s])
    obj_cos[i] = float(ip17[i] @ E_vec[o])
    cands = [x for x in range(ENTS_N)
             if x != s]
    pick = [cands[rng_e.randrange(
        len(cands))]
        for _ in range(N_RAND_ENT)]
    rand_cos[i] = float(np.median(
        [ip17[i] @ E_vec[x]
         for x in pick]))
    top1 = int(ip17_top1[i])
    if top1 in ent_id_set:
        hit_top1 += 1
ipn = np.linalg.norm(ip17, axis=1)
self_c = self_cos / ipn
obj_c = obj_cos / ipn
rand_c = rand_cos / ipn
top25_share = float(np.median(
    np.sum(fr[:, HEAD25], axis=1)))
unemb = {
    'self_cos_med': float(np.median(self_c)),
    'obj_cos_med': float(np.median(obj_c)),
    'rand_cos_med':
        float(np.median(rand_c)),
    'self_minus_rand':
        float(np.median(self_c)
              - np.median(rand_c)),
    'top1_in_ents_frac':
        hit_top1 / float(NP),
    'top1_chance':
        len(ent_id_set) / float(VOCAB),
    'top25_energy_share': top25_share,
    'vocab': VOCAB,
    'n_ent_token_ids': len(ent_id_set)}
aligned = (unemb['self_minus_rand']
           > UNEMB_MARGIN)
unemb_tag = ('identity_unembed_aligned'
             if aligned else
             'identity_unembed_orthogonal')
log('E-SOFT: self %.5f obj %.5f rand %.5f '
    '(margin %.2f) top1-in-ents %.4f '
    '(chance %.6f) top25 share %.4f -> %s'
    % (unemb['self_cos_med'],
       unemb['obj_cos_med'],
       unemb['rand_cos_med'],
       UNEMB_MARGIN,
       unemb['top1_in_ents_frac'],
       unemb['top1_chance'],
       top25_share, unemb_tag))

# ================================================================
# PART F: dh19 residual field destination
# ================================================================
log('== PART F: dh19 residual ==')
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
z35_c17_med = np.array(
    [float(np.nanmedian(z35['cos_17'][r]))
     for r in range(20, 40)])
ours_c17_med = np.array(
    [float(np.nanmedian(cos17[r]))
     for r in range(20, 40)])
z35_diff = float(np.max(np.abs(
    ours_c17_med - z35_c17_med)))
z35_ok = z35_diff < Z35_TOL
log('z35 cos_17 soft anchor: max|diff| '
    'over L20-39 = %.4f (tol %.2f -> %s)'
    % (z35_diff, Z35_TOL, z35_ok))
# residual field: dh19 minus projection on
# dh17 (row-level, per layer r >= 20)
resid = {}
for r in range(20, 40):
    dh19r = (h19[r].astype(np.float64)
             - base_states[r]
             .astype(np.float64))
    dh17r = (h17[r].astype(np.float64)
             - base_states[r]
             .astype(np.float64))
    n17 = np.sum(dh17r * dh17r, axis=1,
                 keepdims=True)
    coef = np.where(
        n17 > 1e-12,
        np.sum(dh19r * dh17r, axis=1,
               keepdims=True)
        / np.maximum(n17, 1e-12), 0.0)
    perp = dh19r - coef * dh17r
    n19 = np.linalg.norm(dh19r, axis=1)
    npp = np.linalg.norm(perp, axis=1)
    share = np.where(
        n19 > 1e-12, npp / np.maximum(
            n19, 1e-12), 0.0)
    rfc = _rowcos(perp,
                  field[r].astype(
                      np.float64))
    wdn_p = perp @ w_dn_g.astype(
        np.float64)
    resid[r] = {
        'perp_share':
            float(np.median(share)),
        'resid_field_cos':
            float(np.median(rfc)),
        'perp_wdn_med':
            float(np.median(wdn_p)),
        'dh19_wdn_med': float(np.median(
            dh19r @ w_dn_g.astype(
                np.float64))),
        'perp_norm_med':
            float(np.median(npp))}
share38 = resid[38]['perp_share']
wdn19_38 = abs(resid[38]['dh19_wdn_med'])
wdnpp38 = abs(resid[38]['perp_wdn_med'])
if share38 < PERP_DEAD:
    resid_tag = 'd19resid_dead'
elif wdnpp38 > READOUT_FRAC * max(
        wdn19_38, 1e-9):
    resid_tag = 'd19resid_readout'
else:
    resid_tag = 'd19resid_diffuse'
# top-PC of the L38 residual and its
# alignment to known directions
perp38 = None
dh19r38 = (h19[38].astype(np.float64)
           - base_states[38]
           .astype(np.float64))
dh17r38 = (h17[38].astype(np.float64)
           - base_states[38]
           .astype(np.float64))
n17b = np.sum(dh17r38 * dh17r38, axis=1,
              keepdims=True)
coefb = np.where(
    n17b > 1e-12,
    np.sum(dh19r38 * dh17r38, axis=1,
           keepdims=True)
    / np.maximum(n17b, 1e-12), 0.0)
perp38 = dh19r38 - coefb * dh17r38
pc = perp38 - perp38.mean(axis=0)
_U, _S, _Vt = np.linalg.svd(
    pc, full_matrices=False)
pc1_res = _Vt[0]
dv29_mean = dvec[29][:NCAP].astype(
    np.float64).mean(axis=0)
resid_align = {
    'pc1_res_vs_v1':
        float(abs(pc1_res @ v1.astype(
            np.float64)) /
            max(np.linalg.norm(pc1_res),
                1e-12)),
    'pc1_res_vs_wdn':
        float(abs(pc1_res @ w_dn_g.astype(
            np.float64)) /
            max(np.linalg.norm(pc1_res),
                1e-12)),
    'pc1_res_vs_dv29mean':
        float(abs(pc1_res @ dv29_mean)
              / max(np.linalg.norm(pc1_res)
                    * np.linalg.norm(
                        dv29_mean), 1e-12))}
log('F-SOFT: perp share L38 %.4f | '
    'perp@wdn38 %.4f (dh19 %.4f) | '
    'resid-field cos38 %.4f | pc1res '
    'align v1 %.3f wdn %.3f dv29 %.3f '
    '-> %s'
    % (share38,
       resid[38]['perp_wdn_med'],
       resid[38]['dh19_wdn_med'],
       resid[38]['resid_field_cos'],
       resid_align['pc1_res_vs_v1'],
       resid_align['pc1_res_vs_wdn'],
       resid_align['pc1_res_vs_dv29mean'],
       resid_tag))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3143_ok']
n_bit_all = n_bitB + n_bitD
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == 6
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
tags.append(traj_tag)
tags.append(pc1_channel)
tags.append(tail_tag)
tags.append(sign_tag)
tags.append(unemb_tag)
tags.append(resid_tag)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3144,
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
        'res43_sha8': sha43,
        'seal43': SEAL43,
        'xphase_P': xphase_P,
        'xphase_A1': xphase_A1},
    'part_traj': {
        'pc1_sign': pc1_sign,
        'bit_anchors_3142': bit_anchors_B,
        'traj': {cn: {str(r): traj[cn][r]
                      for r in TRAJ_L}
                 for cn in traj},
        'gap': {str(r): gap[r]
                for r in TRAJ_L},
        'max_gap': max_gap,
        'blocking_layer': int(blk_layer),
        'traj_tag': traj_tag},
    'part_tokdec': {
        'tokcls': tokcls,
        'pc1_channel': pc1_channel,
        'b3': b3},
    'part_co36sign': {
        'ident': ident,
        'bit_anchors': bit_anchors_D,
        'd_head': d_head,
        'd_tail': d_tail,
        'd_tailpos': d_tailpos,
        'd_gbot':
            E_res['d_gbot_d1.0']['chg'],
        'd_co36full':
            E_res['d_co36full_d1.0']['chg'],
        'd_co50ex':
            E_res['d_co50ex_d1.0']['chg'],
        'tail_tag': tail_tag,
        'sign_tag': sign_tag},
    'part_unembed': unemb,
    'part_resid': {
        'resid': {str(r): resid[r]
                  for r in sorted(resid)},
        'resid_align': resid_align,
        'resid_tag': resid_tag,
        'cos19_med': {str(r):
                      float(np.nanmedian(
                          cos19[r]))
                      for r in range(
                          20, 40)},
        'cos17_med': {str(r):
                      float(np.nanmedian(
                          cos17[r]))
                      for r in range(
                          20, 40)}},
    'part_field': {
        'dvec19_sha8': dvec19_sha,
        'dvec19_sha_ok': bool(d19_sha_ok),
        'dvec19_medn': dvec19_medn,
        'field17_vs_dvec17_cos': fs17,
        'field19_vs_dvec19_cos': fs19,
        'field_self_ok':
            bool(field_self_ok),
        'z35_c17_maxdiff': z35_diff,
        'z35_ok': bool(z35_ok)}}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
traj_out = {}
for cn in ('pc1', 'dv29', 'joint'):
    for key in ('wdn', 'v1', 'norm'):
        traj_out['traj_%s_%s' % (cn, key)] = \
            np.array([traj[cn][r][key]
                      for r in TRAJ_L])
npz_out = {
    'traj_layers': np.array(sorted(TRAJ_L)),
    'gap': np.array([gap[r]
                     for r in TRAJ_L]),
    'cos19': cos19, 'rho19': rho19,
    'cos17': cos17, 'rho17': rho17,
    'perp_share': np.array(
        [resid[r]['perp_share']
         for r in range(20, 40)]),
    'resid_field_cos': np.array(
        [resid[r]['resid_field_cos']
         for r in range(20, 40)]),
    'perp_wdn': np.array(
        [resid[r]['perp_wdn_med']
         for r in range(20, 40)]),
    'dvec19_sha': np.array([dvec19_sha]),
    'd_head_tail_tailpos': np.array(
        [d_head, d_tail, d_tailpos])}
npz_out.update(traj_out)
npz_out['dvec19_out'] = dvec19.astype(
    np.float16)
np.savez(os.path.join(OUT,
                      'p142_readout.npz'),
         **npz_out)
with io.open(os.path.join(OUT, 'p142_'
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
