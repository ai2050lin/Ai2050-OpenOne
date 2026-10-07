# -*- coding: utf-8 -*-
"""Phase 3149 (Omega-P147): flip-carrier
dlogit per-token decomposition (neg_d0.5
8 flip rows + pos_d1 13 flip rows, step
0-1 dlogit top-20 token sets -> common
elevated tokens beyond 131401) + pos_d1
late-onset mechanism (per-step L38/39
capture: w_dn proj vs dlogit131401 vs
|dh|, + firstk prefix-cut gen k{0..4} ->
readout competition vs injection delay
vs format growth) + coordinate-dose
interchange (co50ex top-5/top-10 @
d{0.5,1,2} vs full dose curve ->
breadth-strength exchangeability) + v1
micro-amp fill alpha{-0.05,-0.10,-0.15}
@L38 all -> bias dose dependence vs
3147 micro clip.

Preregistered in 3148 closeout
(Omega-P147). 8 bit-anchor replays:
b_pc1/b_dvec29/b_joint (3142 chain),
d_co50ex_d2.0 (3147), s2_tailpos_d1/
s2_tailneg_d1 (3146/3147), v2_clip_a025
(3146), n_neg_d0.5 (3146). Bank shards
REUSED from 3138. Frozen dvecs from
3135 (sha-anchored). Session baselines
P+A1 with xphase recording. Frozen
before observation."""
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
NAME = ('omega_p147_carrier_dlogit_'
        'poslate_kdose_v3amp')
SMOKE = os.environ.get('P3149_SMOKE',
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
D47 = RDIR + r'\phase3147' \
      r'\omega_p145_tbsym_co50exk_' \
      r'negmech_v1micro'
D48 = RDIR + r'\phase3148' \
      r'\omega_p146_xcross_headsrc_' \
      r'uncancel_v1sym'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3149', NAME)
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
CKPTF = os.path.join(OUT, 'p147_ckpt.pkl')
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
NCAP = 4 if SMOKE else 128
N19_ROWS = 4 if SMOKE else NP
GEN_BATCH = 4 if SMOKE else 32
SPEC_ROWS = 4 if SMOKE else 16
DOSE_COND = 2.0
DOSE_C = 1.0
DEC_L = (17, 29, 38)
FIELD_SELF_COS = 0.99
Z35_TOL = 0.08
SHARE38_TOL = 0.02
WDN39_TOL = 0.15
ALIGN_TOL = 0.03
PNORM38_TOL = 1.0
SPEC_TOKS = [131401, 134772, 23386]
SIGN_TOL = 0.03
T2_STEPS = 2
T2_TOPK = 20
T2_SHARE = 0.5
L_STEPS = 6
L_CUTS = (0, 1, 2, 3, 4)
K_SUBS = (5, 10)
K_DOSES = (0.5, 1.0, 2.0)
V2_ALPHA = ((-0.25, 'v2_amp_a025'),
            (-0.5, 'v2_amp_a050'),
            (0.25, 'v2_clip_a025'),
            (0.5, 'v2_clip_a050'))
V3_ALPHA = ((-0.05, 'v3_ampan_a005'),
            (-0.10, 'v3_ampan_a010'),
            (-0.15, 'v3_ampan_a015'))
RNG_SEED = 3149

# ---------------- frozen 3148 anchors --
EXP_V48 = ('a_3147_ok|repro_bit_18|'
           'repro_bit_ok|dvec19_repro_6a0332'
           '|field_self_ok|z35_cos17_ok|'
           'resid_anchor_ok|'
           'tail_xcross_located|'
           'xcross_fstep_poslate|h_pair_sub|'
           'head_conc_moderate|'
           'src_top_partial|'
           'uncancel_insufficient|'
           'v1_sym_mixed|xphase_ok|'
           'coverage_full')
RES48_SHA = '29327d0e'
SEAL48 = '87842206'
XPHASE48 = 1.0
DVEC19_SHA = '6a0332a6'
MEDNORM19 = 5.643608093261719
# 8 bit anchors replayed (cross-phase)
BIT8 = {'b_pc1_l29_d2.0': 0.203125,
        'b_dvec29_l29_d2.0': 0.640625,
        'b_joint_l29_d2.0': 0.4765625,
        'd_co50ex_d2.0': 0.8515625,
        's2_tailpos_d1': 0.1015625,
        's2_tailneg_d1': 0.09375,
        'v2_clip_a025': 0.296875,
        'n_neg_d0.5': 0.2421875}
N_BIT_TOT = 8
# 3148 part_x2 exact curves (link)
X248 = {'pos_1': 0.1015625,
        'pos_2': 0.2578125,
        'pos_25': 0.265625,
        'pos_3': 0.2265625,
        'pos_35': 0.1640625,
        'pos_4': 0.234375,
        'neg_1': 0.09375,
        'neg_2': 0.171875,
        'neg_25': 0.15625,
        'neg_3': 0.1875,
        'neg_35': 0.2578125,
        'neg_4': 0.3046875}
XCROSS48 = [3.0, 3.5]
FLIP_P1_48 = {'n': 13, 'med': 4.0}
FLIP_N4_48 = {'n': 39, 'med': 0.0}
# 3148 part_h link
H48 = {'solo1': 0.09375, 'solo2': 0.09375,
       'pair': 0.1484375,
       'resid': -0.0390625,
       'frac_top2': 0.2689969539642334,
       'rank19_2530': 10}
# 3148 part_u link
U48 = {'only': 0.2421875,
       'cancel': 0.8203125,
       'rand': 0.203125,
       'wdn': 0.9921875}
# 3148 part_v2 link
V248 = {'amp025': 0.1875,
        'amp050': 0.3828125,
        'clip025': 0.296875,
        'clip050': 0.6640625}
# 3147 micro clip values (V3 sym link)
V47MICRO = {'005': 0.0703125,
            '010': 0.078125,
            '015': 0.1328125}
# 3147 co50ex dose + k-top d4 (K link)
X47K = {'full_05': 0.140625,
        'full_1': 0.421875,
        'full_2': 0.8515625,
        'ktop5_d4': 0.5078125,
        'ktop10_d4': 0.890625}
# 3147 frozen neg flip rows (N2 link)
N47FROZEN = {'rows': [0, 3, 4, 10, 14, 17,
                      20, 28],
             'fs': [4, 6, 4, 4, 4, 4, 6,
                    4]}
ORDER_EX2 = [2530, 3755]
# 3144 resid anchors (C4 recompute gates)
PSHARE_38 = 0.640474
PWDN_39 = 2.495525
PNORM_38 = 55.263357
RA_V1_44 = 0.228672
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
LEDGER_N = 285

# ---------------- seal ------------------
SEAL = {
    'phase': 3149,
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
        'SPEC_ROWS': SPEC_ROWS,
        'DOSE_COND': DOSE_COND,
        'DOSE_C': DOSE_C,
        'DEC_L': list(DEC_L),
        'SIGN_TOL': SIGN_TOL,
        'T2_STEPS': T2_STEPS,
        'T2_TOPK': T2_TOPK,
        'T2_SHARE': T2_SHARE,
        'L_STEPS': L_STEPS,
        'L_CUTS': list(L_CUTS),
        'K_SUBS': list(K_SUBS),
        'K_DOSES': list(K_DOSES),
        'V2_ALPHA': [list(v) for v
                     in V2_ALPHA],
        'V3_ALPHA': [list(v) for v
                     in V3_ALPHA],
        'SPEC_TOKS': SPEC_TOKS,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res48_verdict': EXP_V48,
        'res48_sha8': RES48_SHA,
        'seal48': SEAL48,
        'xphase48': XPHASE48,
        'dvec19_sha8': DVEC19_SHA,
        'mednorm19': MEDNORM19,
        'bit8': BIT8,
        'n_bit_tot': N_BIT_TOT,
        'x248': X248,
        'xcross48': XCROSS48,
        'flip_p1_48': FLIP_P1_48,
        'flip_n4_48': FLIP_N4_48,
        'h48': H48, 'u48': U48,
        'v248': V248,
        'v47micro': V47MICRO,
        'x47k': X47K,
        'n47_frozen': N47FROZEN,
        'order_ex2': ORDER_EX2,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3148 closeout Omega-P147: '
               '(1) flip-carrier dlogit '
               'per-token decomposition: '
               'neg_d0.5 8 flip rows + '
               'pos_d1 13 flip rows, step '
               '0-1 dlogit top-20 token '
               'sets -> common elevated '
               'tokens beyond 131401; '
               '(2) pos_d1 late-onset '
               'mechanism: per-step L38/39 '
               'capture (wdn proj vs '
               'dlogit131401 vs |dh|) + '
               'firstk prefix-cut gen '
               'k{0..4} -> readout '
               'competition vs injection '
               'delay vs format growth; '
               '(3) coordinate-dose '
               'interchange: co50ex '
               'top-5/top-10 @d{0.5,1,2} '
               'vs full dose curve -> '
               'breadth-strength '
               'exchangeability; (4) v1 '
               'micro-amp fill alpha'
               '{-0.05,-0.10,-0.15} @L38 '
               'all -> bias dose '
               'dependence vs 3147 micro '
               'clip. 8 bit replays. Bank '
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
assert z26['gen_base_A1'].shape == (672, 12)
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
# PART A: 3148 link asserts
# ================================================================
log('== PART A: link asserts ==')
res48 = json.load(io.open(
    D48 + r'\result.json',
    encoding='utf-8'))
assert res48['smoke'] is False
V48g = res48['verdict']
assert V48g == EXP_V48, V48g
raw48 = io.open(
    D48 + r'\result.json', 'rb').read()
sha48 = hashlib.sha256(
    raw48).hexdigest()[:8]
assert sha48 == RES48_SHA, sha48
assert str(res48['seal_sha8']) == SEAL48
pa48 = res48['part_a']
assert abs(float(pa48['xphase_P'])
           - XPHASE48) < 1e-12
assert abs(float(pa48['xphase_A1'])
           - XPHASE48) < 1e-12
px248 = res48['part_x2']
for k, v in (('1', 'pos_1'),
             ('2', 'pos_2'),
             ('2.5', 'pos_25'),
             ('3', 'pos_3'),
             ('3.5', 'pos_35'),
             ('4', 'pos_4')):
    assert abs(float(
        px248['pos_curve'][k])
        - X248[v]) < 1e-9, v
for k, v in (('1', 'neg_1'),
             ('2', 'neg_2'),
             ('2.5', 'neg_25'),
             ('3', 'neg_3'),
             ('3.5', 'neg_35'),
             ('4', 'neg_4')):
    assert abs(float(
        px248['neg_curve'][k])
        - X248[v]) < 1e-9, v
assert px248['xcross_tag'] == \
    'tail_xcross_located'
assert [float(v) for v
        in px248['xcross_interval']] == \
    XCROSS48
fp148 = px248['flip_pos_d1']
assert int(fp148['n']) == \
    FLIP_P1_48['n']
assert abs(float(fp148['med_fstep'])
           - FLIP_P1_48['med']) < 1e-9
fn448 = px248['flip_neg_d4']
assert int(fn448['n']) == \
    FLIP_N4_48['n']
assert abs(float(fn448['med_fstep'])
           - FLIP_N4_48['med']) < 1e-9
ph48 = res48['part_h']
assert abs(float(ph48['solo1_d2'])
           - H48['solo1']) < 1e-9
assert abs(float(ph48['solo2_d2'])
           - H48['solo2']) < 1e-9
assert abs(float(ph48['pair_d2'])
           - H48['pair']) < 1e-9
assert abs(float(ph48['resid_pair'])
           - H48['resid']) < 1e-9
assert abs(float(ph48['frac_top2'])
           - H48['frac_top2']) < 1e-9
assert int(ph48['rank_dv19']['2530']) == \
    H48['rank19_2530']
pu48 = res48['part_u']
for k in ('only', 'cancel', 'rand',
          'wdn'):
    assert abs(float(pu48['chg_' + k])
               - U48[k]) < 1e-9, k
pv248 = res48['part_v2']
for k in ('amp025', 'amp050',
          'clip025', 'clip050'):
    assert abs(float(pv248['chg_' + k])
               - V248[k]) < 1e-9, k
log('A hard asserts ok (3148 sha8 %s '
    'seal %s: xphase/part_x2 curves/'
    'xcross/flip rows/part_h/part_u/'
    'part_v2 all verified)'
    % (sha48, SEAL48))



# p3148 patch1 applied

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
                 inj_vec=None,
                 clip=None):
    """Generate N_NEW tokens per row from
    prompt ids. inj_vec: (il, dvec_batch
    (n,HID) fp32, scale, mode). mode:
    'allstep' = every forward; int s =
    prompt forward + decode step s (3135
    semantics). inj: (il, coords, delta,
    sgn, mode) coordinate injection
    (3137 G semantics). clip: list of
    (il, direction np[HID], alpha, cmode)
    v1-projection hooks on layer outputs;
    cmode 'all' = every forward (3145
    semantics at alpha 1.0), 'decode' =
    skip prompt forward (step 0)."""
    ids_p, mask_p = _pad_batch(pids_list)
    hooks = []
    if clip:
        for (il, direction, alpha,
             cmode) in clip:
            lyr = model_g.model.layers[il]
            v_t = torch.as_tensor(
                np.asarray(direction,
                           dtype=np.float32),
                device='cuda')
            v_t = v_t / (v_t.norm() + 1e-12)
            _af = float(alpha)
            st = {'step': 0}

            def _clip(mod, inp, out,
                      _v=v_t, _a=_af,
                      _m=cmode, _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if _m == 'decode' \
                        and _st['step'] == 0:
                    _st['step'] += 1
                    return None
                h = o2[:, -1, :].float()
                proj = (h @ _v) \
                    .unsqueeze(-1)
                o2[:, -1, :] = (
                    h - _a * proj
                    * _v.unsqueeze(0)) \
                    .to(o2.dtype)
                _st['step'] += 1
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _clip))
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
                        or _st['step'] == _m \
                        or (isinstance(_m, str)
                            and _m.startswith(
                                'cut')
                            and _st['step']
                            <= int(_m[3:])):
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
                  'nbase', 'ninj', 'nstepb',
                  'nstepi')
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
# PART C4: cond19/cond17 captures -> perp38
# top-PC + resid anchors
# ================================================================
log('== PART C4: cond captures + resid ==')
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
cos19m = {}
for r in range(20, 40):
    dh = (h19[r].astype(np.float64)
          - base_states[r].astype(np.float64))
    dv = field[r].astype(np.float64)
    cos19m[r] = _rowcos(dh, dv)
z35_c17_med = np.array(
    [float(np.nanmedian(z35['cos_17'][r]))
     for r in range(20, 40)])
h17_cos = {}
for r in range(20, 40):
    dh = (h17[r].astype(np.float64)
          - base_states[r].astype(np.float64))
    dv = field[r].astype(np.float64)
    h17_cos[r] = _rowcos(dh, dv)
ours_c17_med = np.array(
    [float(np.nanmedian(h17_cos[r]))
     for r in range(20, 40)])
z35_diff = float(np.max(np.abs(
    ours_c17_med - z35_c17_med)))
z35_ok = z35_diff < Z35_TOL
log('z35 cos_17 soft anchor: max|diff| '
    'over L20-39 = %.4f (tol %.2f -> %s)'
    % (z35_diff, Z35_TOL, z35_ok))


def _perp_at(r):
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
    return perp, share, dh19r, dh17r


perp38, share38_arr, dh19r38, dh17r38 = \
    _perp_at(38)
perp39, share39_arr, dh19r39, dh17r39 = \
    _perp_at(39)
share38 = float(np.median(share38_arr))
wdn_p38 = float(np.median(
    perp38 @ w_dn_g.astype(np.float64)))
wdn_p39 = float(np.median(
    perp39 @ w_dn_g.astype(np.float64)))
dh19_wdn39 = float(np.median(
    dh19r39 @ w_dn_g.astype(np.float64)))
pnorm38 = float(np.median(
    np.linalg.norm(perp38, axis=1)))
pc = perp38 - perp38.mean(axis=0)
_U, _S, _Vt = np.linalg.svd(
    pc, full_matrices=False)
pc1_res = _Vt[0].astype(np.float64)
dv29_mean = dvec[29][:NCAP].astype(
    np.float64).mean(axis=0)
v1 = PC[29]['V'][0].astype(np.float64)
ra_v1 = float(abs(pc1_res @ v1)
              / max(np.linalg.norm(pc1_res),
                    1e-12))
ra_wdn = float(abs(pc1_res
                   @ w_dn_g.astype(np.float64))
               / max(np.linalg.norm(pc1_res),
                     1e-12))
ra_dv29 = float(abs(pc1_res @ dv29_mean)
                / max(np.linalg.norm(pc1_res)
                      * np.linalg.norm(
                          dv29_mean), 1e-12))
# sign convention: median projection of
# perp38 rows onto pc1_res is positive
proj_sign = float(np.median(
    perp38 @ pc1_res))
if proj_sign < 0:
    pc1_res = -pc1_res
    log('C4 pc1_res sign flipped '
        '(median proj %.3e -> +)'
        % proj_sign)
resid_anchor_ok = (
    abs(share38 - PSHARE_38)
    < SHARE38_TOL
    and abs(wdn_p39 - PWDN_39)
    < WDN39_TOL
    and abs(ra_v1 - RA_V1_44) < ALIGN_TOL
    and abs(pnorm38 - PNORM_38)
    < PNORM38_TOL)
log('C4 resid anchors: share38 %.4f '
    '(want %.4f) wdn_p39 %.4f (want %.4f) '
    'ra_v1 %.4f (want %.4f) pnorm38 %.3f '
    '(want %.3f) -> %s'
    % (share38, PSHARE_38, wdn_p39,
       PWDN_39, ra_v1, RA_V1_44, pnorm38,
       PNORM_38, resid_anchor_ok))
ra_all = {'pc1_res_vs_v1': ra_v1,
          'pc1_res_vs_wdn': ra_wdn,
          'pc1_res_vs_dv29mean': ra_dv29}

# ================================================================
# PART B: bit replays (3 gen trials)
# ================================================================
log('== PART B: bit replays ==')
mn29 = PC[29]['mnorm']
dv_pc1_pos = np.tile(
    (PC[29]['V'][0] * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_pc1_neg = -dv_pc1_pos
dv_pc1 = dv_pc1_pos
dv_dv29 = (dvec[29][:NCAP] * 2.0) \
    .astype(np.float32)
E_res = {}
fstep_store = {}


def _run_vec_trial(tname, il, dv_batch,
                   scale, mode, rows,
                   base12, save_gens=False,
                   clip=None):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        fstep_store[tname] = _K.get(
            'fstep')
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
                      scale, mode)]
            if dv_batch is not None
            else None,
            clip=clip))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    fstep_store[tname] = fs
    if save_gens:
        E_res[tname]['gens'] = gen_l
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    _rec = {'res': {'chg': chg_l,
                    'first': int(first_l)},
            'fstep': fs.astype(np.int8)
            .tolist()}
    if save_gens:
        _rec['res']['gens'] = gen_l
    ck_save(tname, _rec)


def _run_coord_trial(tname, coords, sgn,
                     dsc, rows, base12,
                     il=D17_L, mode=0,
                     save_gens=False):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        fstep_store[tname] = _K.get(
            'fstep')
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        return
    gen_l = []
    for b0 in range(0, len(rows),
                    GEN_BATCH):
        batch = rows[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=[(il, coords,
                  dsc * DELTA_L17, sgn,
                  mode)]))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    fstep_store[tname] = fs
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    _rec = {'res': E_res[tname],
            'fstep': fs.astype(np.int8)
            .tolist()}
    if save_gens:
        _rec['res']['gens'] = gen_l
    ck_save(tname, _rec)


# B1: replay 3 generation bit anchors
_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P,
               save_gens=True)
pc1_sign = None
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - BIT8['b_pc1_l29_d2.0']) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg']
           - BIT8['b_pc1_l29_d2.0']) < 1e-9:
        pc1_sign = -1
    else:
        raise AssertionError(
            'pc1 sign unresolved')
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
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12_P)
    E_res['b_joint_l29_d2.0'] = {
        'chg': chg_l, 'first': int(first_l),
        'gens': gen_l}
    fstep_store['b_joint_l29_d2.0'] = fs
    log('b_joint_l29_d2.0: chg=%.4f '
        'first=%d' % (chg_l, first_l))
    ck_save('b_joint_l29_d2.0', {
        'res': E_res['b_joint_l29_d2.0']})
else:
    E_res['b_joint_l29_d2.0'] = _KJ['res']
    log('b_joint_l29_d2.0 RESUMED '
        'chg=%.4f'
        % E_res['b_joint_l29_d2.0']['chg'])
n_bit_all = 0
bit_anchors = {}
if not SMOKE:
    for tn in ('b_pc1_l29_d2.0',
               'b_dvec29_l29_d2.0',
               'b_joint_l29_d2.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT8[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT8[tn],
            'match': bool(m)}
    log('B1 3142 repro: %d/3 bit-match'
        % n_bit_all)
else:
    log('B1 3142 repro SKIPPED (smoke)')

# ================================================================
# PART S: bot15 sign matrix + subdivision
# ================================================================
log('== PART S: tb sign matrix ==')
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
TTOP10 = list(order_e[25:35])
TBOT15 = list(order_e[35:])
TB5 = list(order_e[35:40])
TB10 = list(order_e[40:])
# 3149: tail d2..d4 curve dropped
# (xcross located in 3148); d1 pair
# retained in PART X2 (s2_* namespace)


def _chg(tn):
    return E_res[tn]['chg']


# ================================================================
# PART X: co50ex superlinear anatomy
# ================================================================
log('== PART X: co50ex anatomy ==')
co50 = CO_SETS['co50']
co50ex = np.asarray(
    sorted(set(co50.tolist())
           - set(co36.tolist())),
    dtype=np.int64)
assert co50ex.ndim == 1
assert len(co50ex) == 50, len(co50ex)
assert len(set(co50ex.tolist())
           & set(co36.tolist())) == 0, \
    'co50ex must be disjoint from co36' \
    ' (3135)'
score_ex = np.mean(fr[:, co50ex], axis=0)
order_ex = co50ex[np.argsort(-score_ex)]
log('X order_ex (IDEINT@L17 enrichment '
    'desc): %s' % json.dumps(
        [int(c) for c in order_ex]))
# 1 bit replay (K baseline)
_run_coord_trial('d_co50ex_d2.0', co50ex,
                 -1, 2.0, rows_A1, base12_A1)
if not SMOKE:
    tn = 'd_co50ex_d2.0'
    got = E_res[tn]['chg']
    m = abs(got - BIT8[tn]) < 1e-9
    n_bit_all += int(m)
    bit_anchors[tn] = {
        'got': got, 'want': BIT8[tn],
        'match': bool(m)}
    log('X replay: +1 bit anchor '
        '(total %d/%d)'
        % (n_bit_all, N_BIT_TOT))
assert [int(c) for c in order_ex[:2]] == ORDER_EX2, list(order_ex[:2])
log('order_ex top2 == 3148 link %s (enrichment desc)' % ORDER_EX2)

# ================================================================
# PART N: neg_d0.5 bit + flip rows
# ================================================================
log('== PART N: neg_d0.5 + flips ==')
# N1: neg_d0.5 replay (bit anchor
# 0.2421875 from 3146)
dv_n = (pc1_res
        / max(np.linalg.norm(pc1_res),
              1e-12) * pnorm38 * 0.5) \
    .astype(np.float32)
dv_n_tile = np.tile(dv_n[None, :],
                    (NCAP, 1))
_run_vec_trial('n_neg_d0.5', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               save_gens=True)
chg_n = E_res['n_neg_d0.5']['chg']
_gen_n = E_res['n_neg_d0.5']['gens']
fs_n = fstep_store['n_neg_d0.5']
log('N1 n_neg_d0.5 done (3146 want '
    '0.2422)')
n_repro_ok = abs(chg_n - 0.2421875) < 1e-9
if not SMOKE:
    got = chg_n
    m = abs(got - BIT8['n_neg_d0.5']) \
        < 1e-9
    n_bit_all += int(m)
    bit_anchors['n_neg_d0.5'] = {
        'got': got,
        'want': BIT8['n_neg_d0.5'],
        'match': bool(m)}
    log('N replay: +1 bit anchor (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
log('N1 anchor vs 3146: %s'
    % n_repro_ok)
# flip rows (3147 frozen list)
flip_rows = []
for j in range(NCAP):
    if fstep_store['n_neg_d0.5'][j] >= 0:
        flip_rows.append(j)
flip_rows = flip_rows[:8]
NF = len(flip_rows)
log('N2: %d flip rows %s'
    % (NF, flip_rows))
if not SMOKE:
    assert flip_rows == \
        N47FROZEN['rows'], flip_rows
    assert [int(fstep_store['n_neg_d0.5'][j])
            for j in flip_rows] == \
        N47FROZEN['fs'], 'fsteps drift'
    log('N2 frozen-list match ok (3147)')

# ================================================================
# PART X2: tail d1 pair (bit + T2/L rows)
# ================================================================
log('== PART X2: tail d1 pair ==')
TAIL_DOSES = (1.0,)
for dd in TAIL_DOSES:
    _run_coord_trial('s2_tailpos_d%g' % dd,
                     TAIL25, 1, dd, rows_A1,
                     base12_A1,
                     save_gens=True)
    _run_coord_trial('s2_tailneg_d%g' % dd,
                     TAIL25, -1, dd, rows_A1,
                     base12_A1)
if not SMOKE:
    for tn in ('s2_tailpos_d1',
               's2_tailneg_d1'):
        got = E_res[tn]['chg']
        m = abs(got - BIT8[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT8[tn],
            'match': bool(m)}
    log('X2 replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
# flip-row extraction (T2/L inputs)
fs_p1 = fstep_store['s2_tailpos_d1']
fs_n1 = fstep_store['s2_tailneg_d1']
flp_rows = [j for j in range(NCAP)
            if fs_p1[j] >= 0]
fln_rows = [j for j in range(NCAP)
            if fs_n1[j] >= 0]
log('X2: pos_d1 flips n=%d (fs %s); '
    'neg_d1 flips n=%d'
    % (len(flp_rows),
       [int(fs_p1[j]) for j in flp_rows],
       len(fln_rows)))
if not SMOKE:
    assert len(flp_rows) == \
        FLIP_P1_48['n'], len(flp_rows)
    assert float(np.median(
        [int(fs_p1[j])
         for j in flp_rows])) == \
        FLIP_P1_48['med']


# ================================================================
# PART V2: v1 perturbation symmetry
# ================================================================
log('== PART V2: v1 symmetry ==')
v1f = v1.astype(np.float32)
for (a, tname) in V2_ALPHA:
    _run_vec_trial(tname, 29, None, 1.0,
                   'allstep', rows_scan,
                   base12_P,
                   clip=[(38, v1f, a,
                          'all')])
if not SMOKE:
    tn = 'v2_clip_a025'
    got = E_res[tn]['chg']
    m = abs(got - BIT8[tn]) < 1e-9
    n_bit_all += int(m)
    bit_anchors[tn] = {
        'got': got, 'want': BIT8[tn],
        'match': bool(m)}
    log('V2 replay: +1 bit anchor (total '
        '%d/%d)' % (n_bit_all, N_BIT_TOT))
sym25 = _chg('v2_amp_a025') / max(
    _chg('v2_clip_a025'), 1e-9)
sym50 = _chg('v2_amp_a050') / max(
    _chg('v2_clip_a050'), 1e-9)
if sym25 > 0.8 and sym50 > 0.8:
    v_sym = 'v1_sym_amplitude'
elif sym25 < 0.5 and sym50 < 0.5:
    v_sym = 'v1_sym_directional'
else:
    v_sym = 'v1_sym_mixed'
log('V-GATE2: amp025 %.4f vs clip025 '
    '%.4f (sym %.2f); amp050 %.4f vs '
    'clip050 %.4f (sym %.2f) -> %s'
    % (_chg('v2_amp_a025'),
       _chg('v2_clip_a025'), sym25,
       _chg('v2_amp_a050'),
       _chg('v2_clip_a050'), sym50,
       v_sym))

# ================================================================
# PART T2: flip-carrier dlogit decomposition
# ================================================================
log('== PART T2: carrier dlogit ==')


def _dl_vec(dh):
    """dlogit vector via chunked
    fp32 matmul against WUG (GPU
    mem-safe, 268MB peak per
    chunk)."""
    ots = []
    for c0 in range(0, VOCAB,
                    16384):
        Wc = WUG[c0:c0 + 16384] \
            .float()
        ots.append(dh @ Wc.T)
        del Wc
    return torch.cat(ots)


def _dlogit_steps(row_ids, base_tok,
                  inj_tok, dv_row,
                  inj_coord, n_steps):
    """Per-step dlogit (inj - base) for
    steps 0..n_steps-1. dv_row: L38 vec
    injection or None. inj_coord:
    (coords, delta) L17 injection or
    None. Returns list of vocab np
    vectors (fp64)."""
    outs = []
    for side in range(2):
        toks = base_tok if side == 0 \
            else inj_tok
        step_h = []
        for k in range(n_steps):
            feats = {39: None}
            hooks = []

            def _mk39(mod, inp, out,
                      _f=feats):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                _f[39] = o2[0, -1, :] \
                    .detach()
                return None

            hooks.append(
                model_g.model.layers[39]
                .register_forward_hook(
                    _mk39))
            if side == 1:
                if dv_row is not None:
                    dv_t = torch.as_tensor(
                        dv_row,
                        dtype=torch.float32,
                        device='cuda') \
                        .to(torch.bfloat16)

                    def _injv2(mod, inp,
                               out,
                               _dv=dv_t):
                        o2 = out[0] \
                            if isinstance(
                                out,
                                tuple) \
                            else out
                        o2[0, -1, :] += _dv
                        return None

                    hooks.append(
                        model_g.model
                        .layers[38]
                        .register_forward_hook(
                            _injv2))
                if inj_coord is not None:
                    (co_l, dl_v) = inj_coord
                    co_t = torch.as_tensor(
                        np.asarray(
                            co_l,
                            dtype=np.int64),
                        device='cuda')

                    def _injc2(mod, inp,
                               out,
                               _co=co_t,
                               _dl=dl_v):
                        o2 = out[0] \
                            if isinstance(
                                out,
                                tuple) \
                            else out
                        o2[0, -1, _co] += \
                            float(_dl)
                        return None

                    hooks.append(
                        model_g.model
                        .layers[17]
                        .register_forward_hook(
                            _injc2))
            try:
                with torch.inference_mode():
                    model_g(
                        torch.tensor(
                            [list(row_ids)
                             + [int(t)
                                for t in
                                toks[:k]]],
                            device='cuda'),
                        use_cache=False)
            finally:
                for hk in hooks:
                    hk.remove()
            step_h.append(feats[39]
                          .float())
        outs.append(step_h)
    res_dlog = []
    for k in range(n_steps):
        hb = norm_g(outs[0][k].to(
            torch.bfloat16)
            .unsqueeze(0))[0].float().detach()
        hi = norm_g(outs[1][k].to(
            torch.bfloat16)
            .unsqueeze(0))[0].float().detach()
        dl = _dl_vec(hi - hb) \
            .cpu().numpy()
        res_dlog.append(
            dl.astype(np.float64))
    return res_dlog


_T2CK = CK['data'].get('t2')
if _T2CK is not None:
    T2_DATA = _T2CK['t2']
    log('T2 RESUMED')
else:
    T2_DATA = {'neg': {}, 'pos': {}}
    _tail_p = [int(c) for c in TAIL25]
    for j in flip_rows:
        _bt = base12_P[j]
        _it = pad12(_gen_n[j])
        dls = _dlogit_steps(
            rows_scan[j], _bt, _it,
            dv_n[j], None, T2_STEPS)
        _rec = {}
        for k in range(T2_STEPS):
            dl = dls[k]
            top = np.argsort(
                -np.abs(dl))[:T2_TOPK]
            r131 = int(np.sum(
                dl > dl[SPEC_TOKS[0]])) + 1
            _rec['s%d' % k] = {
                'top50': [[int(t),
                           float(dl[t])]
                          for t in top],
                'd131': float(
                    dl[SPEC_TOKS[0]]),
                'r131': r131}
        T2_DATA['neg'][str(j)] = _rec
        log('T2 neg row %d done' % j)
    for j in flp_rows:
        _bt = base12_A1[j]
        _it = pad12(
            E_res['s2_tailpos_d1']
            ['gens'][j])
        dls = _dlogit_steps(
            rows_A1[j], _bt, _it, None,
            (_tail_p, 1.0 * DELTA_L17),
            T2_STEPS)
        _rec = {}
        for k in range(T2_STEPS):
            dl = dls[k]
            top = np.argsort(
                -np.abs(dl))[:T2_TOPK]
            r131 = int(np.sum(
                dl > dl[SPEC_TOKS[0]])) + 1
            _rec['s%d' % k] = {
                'top50': [[int(t),
                           float(dl[t])]
                          for t in top],
                'd131': float(
                    dl[SPEC_TOKS[0]]),
                'r131': r131}
        T2_DATA['pos'][str(j)] = _rec
        log('T2 pos row %d done' % j)
    ck_save('t2', {'t2': T2_DATA})
common_tok = {}
t2_stats = {}
for grp in ('neg', 'pos'):
    rows_l = flip_rows if grp == 'neg' \
        else flp_rows
    all_t = {}
    d131_all = []
    r131_all = []
    for j in rows_l:
        d = T2_DATA[grp][str(j)]
        for k in range(T2_STEPS):
            r = d['s%d' % k]
            d131_all.append(r['d131'])
            r131_all.append(r['r131'])
            for (tk, vl) in r['top50']:
                if vl > 0:
                    all_t[int(tk)] = \
                        all_t.get(int(tk),
                                  0) + 1
    thr = int(np.ceil(len(rows_l)
                      * T2_SHARE))
    common = sorted(
        [t for t, c in all_t.items()
         if c >= thr
         and t != SPEC_TOKS[0]])
    common_tok[grp] = common
    t2_stats[grp] = {
        'n_rows': len(rows_l),
        'thr': thr,
        'n_common': len(common),
        'common': common[:30],
        'd131_med': float(np.median(
            d131_all)),
        'r131_med': float(np.median(
            r131_all))}
shared = sorted(set(common_tok['neg'])
                & set(common_tok['pos']))
if len(common_tok['neg']) >= 3 \
        or len(common_tok['pos']) >= 3:
    t2_tag = 'carrier_common_found'
elif len(common_tok['neg']) >= 1 \
        or len(common_tok['pos']) >= 1:
    t2_tag = 'carrier_common_sparse'
else:
    t2_tag = 'carrier_131401_only'
log('T2-GATE: neg common %s | pos '
    'common %s | shared %s | d131 med '
    'neg %.3f (r %.0f) pos %.3f '
    '(r %.0f) -> %s'
    % (json.dumps(common_tok['neg']
                  [:15]),
       json.dumps(common_tok['pos']
                  [:15]),
       json.dumps(shared[:10]),
       t2_stats['neg']['d131_med'],
       t2_stats['neg']['r131_med'],
       t2_stats['pos']['d131_med'],
       t2_stats['pos']['r131_med'],
       t2_tag))

# ================================================================
# PART L: pos_d1 late-onset mechanism
# ================================================================
log('== PART L: pos_d1 late mech ==')
_LCK = CK['data'].get('late')
if _LCK is not None:
    L_traj = _LCK['traj']
    L_cuts = _LCK['cuts']
    log('L RESUMED')
else:
    L_traj = {}
    wdn_np = w_dn_g.astype(np.float64)
    w131 = WUG[SPEC_TOKS[0]] \
        .float()
    _tail_p = [int(c) for c in TAIL25]
    for j in flp_rows:
        _bt = base12_A1[j]
        _it = pad12(
            E_res['s2_tailpos_d1']
            ['gens'][j])
        outs = []
        for side in range(2):
            toks = _bt if side == 0 \
                else _it
            step_h = []
            for k in range(L_STEPS):
                feats = {39: None}
                hooks = []

                def _mk39(mod, inp, out,
                          _f=feats):
                    o2 = out[0] \
                        if isinstance(out,
                                      tuple) \
                        else out
                    _f[39] = o2[0, -1, :] \
                        .detach()
                    return None

                hooks.append(
                    model_g.model
                    .layers[39]
                    .register_forward_hook(
                        _mk39))
                if side == 1:

                    def _injp(mod, inp,
                              out,
                              _tp=_tail_p):
                        o2 = out[0] \
                            if isinstance(
                                out,
                                tuple) \
                            else out
                        o2[0, -1, _tp] += \
                            1.0 * DELTA_L17
                        return None

                    hooks.append(
                        model_g.model
                        .layers[17]
                        .register_forward_hook(
                            _injp))
                try:
                    with torch.inference_mode():
                        model_g(
                            torch.tensor(
                                [list(rows_A1[j])
                                 + [int(t)
                                    for t in
                                    toks[:k]]],
                                device='cuda'),
                            use_cache=False)
                finally:
                    for hk in hooks:
                        hk.remove()
                step_h.append(
                    feats[39].float())
            outs.append(step_h)
        steps = []
        for k in range(L_STEPS):
            hb = norm_g(outs[0][k].to(
                torch.bfloat16)
                .unsqueeze(0))[0].float().detach()
            hi = norm_g(outs[1][k].to(
                torch.bfloat16)
                .unsqueeze(0))[0].float().detach()
            dh = ((hi - hb).cpu().numpy()
                  .astype(np.float64))
            d131 = float(
                ((hi - hb) @ w131)
                .cpu())
            steps.append({
                'wdn': float(dh @ wdn_np),
                'dh': float(np.linalg
                            .norm(dh)),
                'd131': d131})
        L_traj[str(j)] = steps
        log('L traj row %d done' % j)
    for cut in L_CUTS:
        _run_coord_trial(
            'l_cut%d' % cut, _tail_p, 1,
            1.0,
            [rows_A1[j]
             for j in flp_rows],
            [base12_A1[j]
             for j in flp_rows],
            mode='cut%d' % cut)
    L_cuts = {}
    for jj, j in enumerate(flp_rows):
        kmin = -1
        for cut in L_CUTS:
            fs_c = fstep_store[
                'l_cut%d' % cut]
            if fs_c[jj] >= 0:
                kmin = cut
                break
        L_cuts[str(j)] = kmin
    ck_save('late', {'traj': L_traj,
                     'cuts': L_cuts})
fs_j = {j: int(fs_p1[j])
        for j in flp_rows}
km_list = [int(L_cuts[str(j)])
           for j in flp_rows]
fs_list = [fs_j[j] for j in flp_rows]
frac_fixed = float(np.mean(
    [0 <= L_cuts[str(j)] <= 1
     for j in flp_rows]))
wdn_early = float(np.median(
    [L_traj[str(j)][0]['wdn']
     for j in flp_rows]))
wdn_late = float(np.median(
    [L_traj[str(j)][min(L_STEPS - 1,
                        fs_j[j])]
     ['wdn'] for j in flp_rows]))
d131_early = float(np.median(
    [L_traj[str(j)][0]['d131']
     for j in flp_rows]))
d131_late = float(np.median(
    [L_traj[str(j)][min(L_STEPS - 1,
                        fs_j[j])]
     ['d131'] for j in flp_rows]))
log('L-SOFT: kmin %s vs fstep %s | wdn '
    'early %.3f late %.3f | d131 early '
    '%.3f late %.3f | frac_fixed %.2f'
    % (json.dumps(km_list),
       json.dumps(fs_list), wdn_early,
       wdn_late, d131_early, d131_late,
       frac_fixed))
if frac_fixed >= 0.5:
    l_tag = 'poslate_fixed_early'
elif wdn_early > 0.3 \
        and wdn_late > wdn_early:
    l_tag = 'poslate_readout_comp'
elif d131_early > 0.3 \
        and d131_late > d131_early:
    l_tag = 'poslate_format_grow'
else:
    l_tag = 'poslate_inj_delay'
log('L-GATE: %s' % l_tag)

# ================================================================
# PART K: coordinate-dose interchange
# ================================================================
log('== PART K: coord-dose interchange ==')
for kk in K_SUBS:
    for dd in K_DOSES:
        _run_coord_trial(
            'k_top%d_d%g' % (kk, dd),
            [int(c) for c in
             order_ex[:kk]], -1, dd,
            rows_A1, base12_A1)
kx = {(kk, dd): _chg('k_top%d_d%g'
                     % (kk, dd))
      for kk in K_SUBS
      for dd in K_DOSES}
full_curve = {0.5: X47K['full_05'],
              1.0: X47K['full_1'],
              2.0: X47K['full_2']}
best_gap = None
best_pair = None
for kk in K_SUBS:
    for dd in K_DOSES:
        for df, vf in full_curve.items():
            g = abs(kx[(kk, dd)] - vf)
            if best_gap is None \
                    or g < best_gap:
                best_gap = g
                best_pair = (kk, dd, df)
if best_gap < 0.05:
    kx_tag = 'kx_interchangeable'
elif best_gap < 0.15:
    kx_tag = 'kx_partial'
else:
    kx_tag = 'kx_breadth_required'
log('K-GATE: kx %s | best match '
    'top%d@d%g~full@d%g gap %.4f -> %s'
    % (json.dumps({'top%d_d%g'
                   % (kk, dd):
                   round(float(v), 4)
                   for (kk, dd), v
                   in sorted(kx.items())}),
        best_pair[0], best_pair[1],
        best_pair[2], best_gap, kx_tag))

# ================================================================
# PART V3: v1 micro-amp fill (bias dose)
# ================================================================
log('== PART V3: v1 micro-amp fill ==')
v1f = v1.astype(np.float32)
for (a, tname) in V3_ALPHA:
    _run_vec_trial(tname, 29, None, 1.0,
                   'allstep', rows_scan,
                   base12_P,
                   clip=[(38, v1f, a,
                          'all')])
sym_curve = {
    '0.05': _chg('v3_ampan_a005')
    / max(V47MICRO['005'], 1e-9),
    '0.1': _chg('v3_ampan_a010')
    / max(V47MICRO['010'], 1e-9),
    '0.15': _chg('v3_ampan_a015')
    / max(V47MICRO['015'], 1e-9),
    '0.25': _chg('v2_amp_a025')
    / max(_chg('v2_clip_a025'), 1e-9),
    '0.5': _chg('v2_amp_a050')
    / max(_chg('v2_clip_a050'), 1e-9)}
sv = [sym_curve[k] for k in
      ('0.05', '0.1', '0.15', '0.25',
       '0.5')]
if max(sv) - min(sv) < 0.1:
    v3_tag = 'v3_bias_constant'
elif sv[0] > sv[-1] + 0.1:
    v3_tag = 'v3_bias_growing'
elif sv[-1] > sv[0] + 0.1:
    v3_tag = 'v3_bias_shrinking'
else:
    v3_tag = 'v3_bias_irregular'
log('V3-GATE: sym curve %s -> %s'
    % (json.dumps({k: round(float(v), 3)
                   for k, v
                   in sym_curve.items()}),
       v3_tag))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3148_ok']
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == N_BIT_TOT
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
tags.append('resid_anchor_ok'
            if resid_anchor_ok
            else 'resid_anchor_drift')
tags.append(t2_tag)
tags.append(l_tag)
tags.append(kx_tag)
tags.append(v3_tag)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3148,
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
        'res48_sha8': sha48,
        'seal48': SEAL48,
        'xphase_P': xphase_P,
        'xphase_A1': xphase_A1},
    'part_field': {
        'dvec19_sha8': dvec19_sha,
        'dvec19_sha_ok':
            bool(d19_sha_ok),
        'dvec19_medn': dvec19_medn,
        'field17_vs_dvec17_cos': fs17,
        'field19_vs_dvec19_cos': fs19,
        'field_self_ok':
            bool(field_self_ok),
        'z35_c17_maxdiff': z35_diff,
        'z35_ok': bool(z35_ok)},
    'part_resid': {
        'share38': share38,
        'wdn_p38': wdn_p38,
        'wdn_p39': wdn_p39,
        'dh19_wdn39': dh19_wdn39,
        'pnorm38': pnorm38,
        'resid_align': ra_all,
        'anchor_ok':
            bool(resid_anchor_ok)},
    'part_bits': {'bit_anchors':
                  bit_anchors},
    'part_t2': {
        'stats': t2_stats,
        'common_neg': common_tok['neg'],
        'common_pos': common_tok['pos'],
        'shared': [int(t)
                   for t in shared],
        't2_tag': t2_tag},
    'part_l': {
        'flp_rows': [int(j)
                     for j in flp_rows],
        'fsteps': [fs_j[j]
                   for j in flp_rows],
        'kmin_cuts': [int(L_cuts[str(j)])
                      for j in flp_rows],
        'wdn_early': wdn_early,
        'wdn_late': wdn_late,
        'd131_early': d131_early,
        'd131_late': d131_late,
        'frac_fixed': frac_fixed,
        'l_tag': l_tag},
    'part_k': {
        'kx': {'top%d_d%g' % (kk, dd):
               float(v)
               for (kk, dd), v
               in sorted(kx.items())},
        'best_pair': [int(best_pair[0]),
                      float(best_pair[1]),
                      float(best_pair[2])],
        'best_gap': float(best_gap),
        'kx_tag': kx_tag},
    'part_v3': {
        'chg_ampan005': _chg(
            'v3_ampan_a005'),
        'chg_ampan010': _chg(
            'v3_ampan_a010'),
        'chg_ampan015': _chg(
            'v3_ampan_a015'),
        'sym_curve': {k: float(v)
                      for k, v
                      in sym_curve.items()},
        'v3_tag': v3_tag},
    }
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'dvec19_sha': np.array([dvec19_sha]),
    't2_common_neg': np.array(
        [int(t) for t in
         common_tok['neg']],
        dtype=np.int64),
    't2_common_pos': np.array(
        [int(t) for t in
         common_tok['pos']],
        dtype=np.int64),
    'l_traj_wdn': np.array(
        [[st['wdn']
          for st in L_traj[str(j)]]
         for j in flp_rows],
        dtype=np.float64),
    'l_traj_d131': np.array(
        [[st['d131']
          for st in L_traj[str(j)]]
         for j in flp_rows],
        dtype=np.float64),
    'l_kmin': np.array(
        [int(L_cuts[str(j)])
         for j in flp_rows],
        dtype=np.int64),
    'kx_curve': np.array(
        [float(kx[(kk, dd)])
         for kk in K_SUBS
         for dd in K_DOSES],
        dtype=np.float64),
    'sym_curve': np.array(
        [float(sym_curve[k]) for k in
         ('0.05', '0.1', '0.15', '0.25',
          '0.5')],
        dtype=np.float64),
    'pc1_res': pc1_res.astype(
        np.float32)}
np.savez(os.path.join(OUT,
                      'p147_readout.npz'),
         **npz_out)
if not SMOKE:
    try:
        os.remove(CKPTF)
        log('ckpt cleaned (final)')
    except OSError:
        pass
log('result.json + npz written. DONE '
    '(verdict %s)' % verdict)

# p3148 patch2 applied
# p3148 patch3 applied
# rev-3148a patch1: flip_rows cap 8 (3147 NFLIP_CAP semantics)

# rev-3149a patch1: norm_g outputs detach (grad leak in T2/L dlogit path)
# rev-3149b patch2: _injp tuple guard (L17 out is tensor -> out[0] sliced dim0)
# rev-3149c patch3: L d131 scalar dot (1D inner product -> 0-dim tensor)
# rev-3149d patch4: L cut gen passes row contents not row indices
# rev-3149e patch5: kmin via fstep_store
