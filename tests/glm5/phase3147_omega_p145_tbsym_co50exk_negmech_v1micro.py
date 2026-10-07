# -*- coding: utf-8 -*-
"""Phase 3147 (Omega-P145): bot15 sign
matrix (TBOT15 x +/-sgn x d{1,2} + TAIL25
pos d{1,2} fill + bot15 subdivision top5/
bot10 @d4 -> tail sign structure: 3145
showed pos_d2 0.2578 > neg_d2 0.1719 while
3146 showed pos_d4 0.2344 < neg_d4 0.3047
-> sign-crossing test) + co50ex
superlinear anatomy (k-subsets top/bot
k{2,5,10,15} by IDEINT enrichment @d4 +
d0.5 low-dose fill -> locus + threshold
structure of the 0.992 near-total
disruption) + neg late-flip mechanism
(neg_d0.5 flip rows teacher-forced
step-by-step captures @L38/39: w_dn
projection trajectory vs dlogit(131401)
-> readout competition vs format shift)
+ v1 micro-clip descent (alpha
{0.05,0.10,0.15} + base-generation v1
amplitude per-step trajectory -> harmless
threshold or continuous dependence).

Preregistered in 3146 closeout
(Omega-P145). 11 bit-anchor replays:
b_pc1/b_dvec29/b_joint (3142, 5th),
d_tbot15_d4/d_ttop10_d4 (3146, 2nd),
d_co50ex d{1,2,4} (3146/3144, 3rd),
t_pc1_inst (3145, 3rd), e_pcres38_d1
pos/neg (3145, 3rd). Bank shards REUSED
from 3138. Frozen dvecs from 3135
(sha-anchored). Session baselines P+A1
with xphase recording. Frozen before
observation."""
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
NAME = ('omega_p145_tbsym_co50exk_'
        'negmech_v1micro')
SMOKE = os.environ.get('P3147_SMOKE',
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
D46 = RDIR + r'\phase3146' \
      r'\omega_p144_taillocus_histfield_' \
      r'pcresdose_cleanclip'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3147', NAME)
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
CKPTF = os.path.join(OUT, 'p145_ckpt.pkl')
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
CLEAN_GATE = 0.05
ONSET_TOL = 0.10
SIGN_TOL = 0.03
Z35_TOL = 0.08
FIELD_SELF_COS = 0.99
SHARE38_TOL = 0.02
WDN39_TOL = 0.15
ALIGN_TOL = 0.03
PNORM38_TOL = 1.0
SPEC_TOKS = [131401, 134772, 23386]
K_GRID = (2, 5, 10, 15)
NFLIP_CAP = 2 if SMOKE else 8
NSTEP_CAP = 3 if SMOKE else 6
ALPHA_MICRO = (0.05, 0.10, 0.15)
RNG_SEED = 3147

# ---------------- frozen 3146 anchors --
EXP_V46 = ('a_3145_ok|repro_bit_14|'
           'repro_bit_ok|dvec19_repro_6a0332|'
           'field_self_ok|z35_cos17_ok|'
           'resid_anchor_ok|head_dose_mixed|'
           'co36full_dose_mono|'
           'co50ex_dose_mono|gbot_dose_mono|'
           'tail_dose_mono|tail_locus_bot15|'
           'hist_onset_l29|l39_jump_downstream|'
           'pcres_gap_mono|neg_flip_late|'
           'v1_no_clean_window|xphase_ok|'
           'coverage_full')
RES46_SHA = 'e5ed3181'
SEAL46 = 'ed8c8ede'
XPHASE46 = 1.0
DVEC19_SHA = '6a0332a6'
MEDNORM19 = 5.643608093261719
# 11 bit anchors to replay on-site
BIT11 = {'b_pc1_l29_d2.0': 0.203125,
         'b_dvec29_l29_d2.0': 0.640625,
         'b_joint_l29_d2.0': 0.4765625,
         'd_tbot15_d4.0': 0.359375,
         'd_ttop10_d4.0': 0.2421875,
         'd_co50ex_d1.0': 0.421875,
         'd_co50ex_d2.0': 0.8515625,
         'd_co50ex_d4.0': 0.9921875,
         't_pc1_l29_inst': 0.0625,
         'e_pcres38_d1.0_pos': 0.984375,
         'e_pcres38_d1.0_neg': 0.421875}
# 3146 part_dose dvals (all 21)
DVALS46 = {
    'd_head_d1.0': 0.21875,
    'd_gbot_d1.0': 0.1171875,
    'd_co36full_d1.0': 0.140625,
    'd_co50ex_d1.0': 0.421875,
    'd_tail_d1.0': 0.09375,
    'd_tail_d2.0': 0.171875,
    'd_tail_d4.0': 0.3046875,
    'd_headpos_d1.0': 0.140625,
    'd_head_d2.0': 0.203125,
    'd_head_d4.0': 0.6875,
    'd_headpos_d2.0': 0.125,
    'd_headpos_d4.0': 0.890625,
    'd_gbot_d2.0': 0.15625,
    'd_gbot_d4.0': 0.2265625,
    'd_co36full_d2.0': 0.2265625,
    'd_co36full_d4.0': 0.65625,
    'd_co50ex_d2.0': 0.8515625,
    'd_co50ex_d4.0': 0.9921875,
    'd_tailpos_d4.0': 0.234375,
    'd_ttop10_d4.0': 0.2421875,
    'd_tbot15_d4.0': 0.359375}
SHAPES46 = {'head': 'mixed',
            'headpos': 'mixed',
            'gbot': 'mono',
            'co36full': 'mono',
            'co50ex': 'mono',
            'tail': 'mono'}
TAILLOCUS46 = 'tail_locus_bot15'
HDR46 = 3.142857142857143
TDR46 = 3.25
# 3146 part_cleanclip anchors
WINDOWS46 = {'a025': 0.296875,
             'a050': 0.6640625,
             'deco': 0.78125}
CLEANTAG46 = 'v1_no_clean_window'
# 3146 part_pcres anchors
PCRES46 = {'e_pcres38_d0.25_pos': 0.296875,
           'e_pcres38_d0.25_neg': 0.1484375,
           'e_pcres38_d0.5_pos': 0.78125,
           'e_pcres38_d0.5_neg': 0.2421875,
           'e_pcres38_d1.0_pos': 0.984375,
           'e_pcres38_d1.0_neg': 0.421875}
GAP46 = {'0.25': 0.1484375,
         '0.5': 0.5390625,
         '1.0': 0.5625}
GAPTAG46 = 'pcres_gap_mono'
NEGFLIPTAG46 = 'neg_flip_late'
MEDFS46 = 4.0
# 3146 part_hist anchors
ONSET46 = 29
JUMP_RAW46 = 1.920691922716944
JUMP_NORM46 = 1.7738666588641274
JUMP_HIST46 = 1.920691922716944
L39TAG46 = 'l39_jump_downstream'
ONSETTAG46 = 'hist_onset_l29'
# 3144 resid anchors (C4 recompute gates)
PSHARE_38 = 0.640474
PWDN_39 = 2.495525
PNORM_38 = 55.263357
RA_V1_44 = 0.228672
# 3143/3144 coordinate lists
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
LEDGER_N = 283

# ---------------- seal ------------------
SEAL = {
    'phase': 3147,
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
        'CLEAN_GATE': CLEAN_GATE,
        'ONSET_TOL': ONSET_TOL,
        'SIGN_TOL': SIGN_TOL,
        'Z35_TOL': Z35_TOL,
        'FIELD_SELF_COS': FIELD_SELF_COS,
        'SPEC_TOKS': SPEC_TOKS,
        'K_GRID': list(K_GRID),
        'NFLIP_CAP': NFLIP_CAP,
        'NSTEP_CAP': NSTEP_CAP,
        'ALPHA_MICRO': list(ALPHA_MICRO),
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res46_verdict': EXP_V46,
        'res46_sha8': RES46_SHA,
        'seal46': SEAL46,
        'xphase46': XPHASE46,
        'dvec19_sha8': DVEC19_SHA,
        'mednorm19': MEDNORM19,
        'bit11': BIT11,
        'dvals46': DVALS46,
        'shapes46': SHAPES46,
        'tail_locus46': TAILLOCUS46,
        'hdr46': HDR46, 'tdr46': TDR46,
        'windows46': WINDOWS46,
        'cleantag46': CLEANTAG46,
        'pcres46': PCRES46,
        'gap46': GAP46,
        'gaptag46': GAPTAG46,
        'negfliptag46': NEGFLIPTAG46,
        'medfs46': MEDFS46,
        'onset46': ONSET46,
        'jump_raw46': JUMP_RAW46,
        'jump_norm46': JUMP_NORM46,
        'jump_hist46': JUMP_HIST46,
        'l39tag46': L39TAG46,
        'onsettag46': ONSETTAG46,
        'pshare_38': PSHARE_38,
        'pwdn_39': PWDN_39,
        'pnorm_38': PNORM_38,
        'ra_v1_44': RA_V1_44,
        'delta_l17': DELTA_L17,
        'high25_43': HIGH25_43,
        'low25_43': LOW25_43,
        'g_top25': G_TOP25,
        'g_bot25': G_BOT25,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3146 closeout Omega-P145: '
               '(1) bot15 sign matrix: TBOT15 '
               'x +/-sgn x d{1,2} + TAIL25 pos '
               'd{1,2} fill + bot15 subdivision '
               'top5/bot10 @d4 -> tail sign '
               'structure (3145 pos_d2 0.2578 > '
               'neg_d2 0.1719 vs 3146 pos_d4 '
               '0.2344 < neg_d4 0.3047 -> sign-'
               'crossing test); (2) co50ex '
               'superlinear: k-subsets top/bot '
               'k{2,5,10,15} by IDEINT '
               'enrichment @d4 + d0.5 fill -> '
               'locus + threshold of 0.992; '
               '(3) neg late-flip: neg_d0.5 '
               'flip rows tf step-by-step '
               'captures @L38/39 w_dn traj vs '
               'dlogit -> readout_comp vs '
               'format_shift; (4) v1 micro '
               'clip alpha{0.05,0.10,0.15} + '
               'base v1 amplitude per-step '
               'traj. 11 bit replays. Bank '
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
# PART A: 3146 link asserts
# ================================================================
log('== PART A: link asserts ==')
res46 = json.load(io.open(
    D46 + r'\result.json', encoding='utf-8'))
assert res46['smoke'] is False
V46 = res46['verdict']
assert V46 == EXP_V46, V46
raw46 = io.open(
    D46 + r'\result.json', 'rb').read()
sha46 = hashlib.sha256(raw46).hexdigest()[:8]
assert sha46 == RES46_SHA, sha46
assert str(res46['seal_sha8']) == SEAL46
pa46 = res46['part_a']
assert abs(float(pa46['xphase_P'])
           - XPHASE46) < 1e-12
assert abs(float(pa46['xphase_A1'])
           - XPHASE46) < 1e-12
pcc46 = res46['part_cleanclip']
for tn, v in BIT11.items():
    if tn in pcc46['bit_anchors']:
        assert abs(float(
            pcc46['bit_anchors'][tn]
            ['got']) - v) < 1e-9, tn
        assert pcc46['bit_anchors'][tn][
            'match'] is True, tn
pd46 = res46['part_dose']
got_dvals = pd46['dvals']
for k, v in DVALS46.items():
    assert abs(float(got_dvals[k]) - v) \
        < 1e-9, k
assert pd46['shapes'] == SHAPES46
assert pd46['tail_locus'] == TAILLOCUS46
assert abs(float(pd46['chg_ttop10_d4'])
           - 0.2421875) < 1e-9
assert abs(float(pd46['chg_tbot15_d4'])
           - 0.359375) < 1e-9
assert abs(float(pd46['head_dose_ratio'])
           - HDR46) < 1e-9
assert abs(float(pd46['tail_dose_ratio'])
           - TDR46) < 1e-9
for wn, v in WINDOWS46.items():
    assert abs(float(
        pcc46['windows'][wn]) - v) < 1e-9, wn
assert pcc46['clean_windows'] == []
assert pcc46['clean_tag'] == CLEANTAG46
pp46 = res46['part_pcres']
for tn, v in PCRES46.items():
    assert abs(float(pp46['trials'][tn])
               - v) < 1e-9, tn
for k, v in GAP46.items():
    assert abs(float(pp46['gap_by_dose'][k])
               - v) < 1e-9, k
assert pp46['gap_tag'] == GAPTAG46
assert pp46['neg_flip_tag'] == NEGFLIPTAG46
assert abs(float(pp46['med_fstep_neg'])
           - MEDFS46) < 1e-9
ph46 = res46['part_hist']
assert int(ph46['onset_hist']) == ONSET46
assert ph46['onset_hist_tag'] == ONSETTAG46
assert abs(float(ph46['jump_raw'])
           - JUMP_RAW46) < 1e-9
assert abs(float(ph46['jump_norm'])
           - JUMP_NORM46) < 1e-9
assert abs(float(ph46['jump_hist'])
           - JUMP_HIST46) < 1e-9
assert ph46['l39_tag'] == L39TAG46
pf46 = res46['part_field']
assert str(pf46['dvec19_sha8']) == \
    DVEC19_SHA
assert abs(float(pf46['dvec19_medn'])
           - MEDNORM19) < 1e-9
assert pf46['field_self_ok'] is True
assert pf46['z35_ok'] is True
pr46 = res46['part_resid']
assert abs(float(pr46['share38'])
           - PSHARE_38) < 1e-4
assert abs(float(pr46['wdn_p39'])
           - PWDN_39) < 1e-4
assert abs(float(pr46['pnorm38'])
           - PNORM_38) < 1e-3
assert pr46['anchor_ok'] is True
log('A hard asserts ok (3146 sha8 %s seal '
    '%s: xphase/bit14/dvals21/shapes/'
    'windows/pcres/hist/field/resid all '
    'verified)' % (sha46, SEAL46))

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
                     il=D17_L):
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
                  0)]))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    fstep_store[tname] = fs
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {
        'res': E_res[tname],
        'fstep': fs.astype(np.int8)
        .tolist()})


# B1: replay 3 generation bit anchors
_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P,
               save_gens=True)
pc1_sign = None
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - BIT11['b_pc1_l29_d2.0']) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg']
           - BIT11['b_pc1_l29_d2.0']) < 1e-9:
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
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
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
# 2 bit replays @d4 sgn-1 (3146, 2nd)
_run_coord_trial('d_tbot15_d4.0', TBOT15,
                 -1, 4.0, rows_A1,
                 base12_A1)
_run_coord_trial('d_ttop10_d4.0', TTOP10,
                 -1, 4.0, rows_A1,
                 base12_A1)
if not SMOKE:
    for tn in ('d_tbot15_d4.0',
               'd_ttop10_d4.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('S replay: +2 bit anchors (total '
        '%d/11)' % n_bit_all)
# sign matrix: TAIL25 pos d{1,2} fill
# (neg d{1,2,4} + pos_d4 known bits)
_run_coord_trial('s_tailpos_d1.0', TAIL25,
                 1, 1.0, rows_A1, base12_A1)
_run_coord_trial('s_tailpos_d2.0', TAIL25,
                 1, 2.0, rows_A1, base12_A1)
# bot15 sign matrix: +/- x d{1,2}
_run_coord_trial('s_tb15_pos_d1.0', TBOT15,
                 1, 1.0, rows_A1, base12_A1)
_run_coord_trial('s_tb15_pos_d2.0', TBOT15,
                 1, 2.0, rows_A1, base12_A1)
_run_coord_trial('s_tb15_neg_d1.0', TBOT15,
                 -1, 1.0, rows_A1, base12_A1)
_run_coord_trial('s_tb15_neg_d2.0', TBOT15,
                 -1, 2.0, rows_A1, base12_A1)
# bot15 subdivision @d4 sgn-1
_run_coord_trial('s_tb5_neg_d4.0', TB5,
                 -1, 4.0, rows_A1, base12_A1)
_run_coord_trial('s_tb10_neg_d4.0', TB10,
                 -1, 4.0, rows_A1, base12_A1)


def _chg(tn):
    return E_res[tn]['chg']


# tail (TAIL25) sign curve:
# neg {0.094, 0.172, 0.305} (bits) vs
# pos {s1, s2, 0.234} (3146 tailpos_d4)
p1 = _chg('s_tailpos_d1.0')
p2 = _chg('s_tailpos_d2.0')
n1t = DVALS46['d_tail_d1.0']
n2t = DVALS46['d_tail_d2.0']
n4t = DVALS46['d_tail_d4.0']
p4t = DVALS46['d_tailpos_d4.0']
if p2 > n2t + SIGN_TOL \
        and p4t < n4t - SIGN_TOL:
    tail_sign = 'tail_sign_cross'
elif p2 > n2t + SIGN_TOL:
    tail_sign = 'tail_sign_pos'
elif n2t > p2 + SIGN_TOL \
        and n4t > p4t + SIGN_TOL:
    tail_sign = 'tail_sign_neg'
elif abs(p2 - n2t) < SIGN_TOL \
        and abs(p4t - n4t) < SIGN_TOL:
    tail_sign = 'tail_sign_symmetric'
else:
    tail_sign = 'tail_sign_mixed'
log('S-GATE: TAIL25 sign curve pos %s vs '
    'neg %s -> %s'
    % (json.dumps({'1': round(p1, 4),
                   '2': round(p2, 4),
                   '4': round(p4t, 4)}),
       json.dumps({'1': round(n1t, 4),
                   '2': round(n2t, 4),
                   '4': round(n4t, 4)}),
       tail_sign))
p2_want = 0.2578125
tailpos2_repro = abs(p2 - p2_want) < 1e-9
log('S-SOFT: s_tailpos_d2 %.4f vs 3145 '
    '0.2578 repro %s' % (p2,
                         tailpos2_repro))
# bot15 sign @d4: neg (bit 0.359) vs pos
b15n = BIT11['d_tbot15_d4.0']
# bot15 pos @d4 not in grid; use neg d2
# vs pos d2 for sign, and full d4 curve
bp2 = _chg('s_tb15_pos_d2.0')
bn2 = _chg('s_tb15_neg_d2.0')
if bp2 > bn2 + SIGN_TOL:
    tb15_sym = 'tb15_sym_pos'
elif bn2 > bp2 + SIGN_TOL:
    tb15_sym = 'tb15_sym_neg'
else:
    tb15_sym = 'tb15_sym_symmetric'
log('S-SOFT: TBOT15 @d2 pos %.4f vs neg '
    '%.4f -> %s; @d1 pos %.4f vs neg %.4f'
    % (bp2, bn2, tb15_sym,
       _chg('s_tb15_pos_d1.0'),
       _chg('s_tb15_neg_d1.0')))
# subdivision @d4 neg
c5 = _chg('s_tb5_neg_d4.0')
c10 = _chg('s_tb10_neg_d4.0')
if c5 > c10 + SIGN_TOL:
    tail_sub = 'tail_sub_top5'
elif c10 > c5 + SIGN_TOL:
    tail_sub = 'tail_sub_bot10'
else:
    tail_sub = 'tail_sub_diffuse'
log('S-GATE: bot15 subdivision @d4: tb5 '
    '%.4f vs tb10 %.4f (tb15 %.4f) -> %s'
    % (c5, c10, b15n, tail_sub))

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
# 3 bit replays + low-dose fill
_run_coord_trial('d_co50ex_d1.0', co50ex,
                 -1, 1.0, rows_A1, base12_A1)
_run_coord_trial('d_co50ex_d2.0', co50ex,
                 -1, 2.0, rows_A1, base12_A1)
_run_coord_trial('d_co50ex_d4.0', co50ex,
                 -1, 4.0, rows_A1, base12_A1)
_run_coord_trial('x_co50ex_d0.5', co50ex,
                 -1, 0.5, rows_A1, base12_A1)
if not SMOKE:
    for tn in ('d_co50ex_d1.0',
               'd_co50ex_d2.0',
               'd_co50ex_d4.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('X replay: +3 bit anchors (total '
        '%d/11)' % n_bit_all)
# k-subsets top/bot @d4 sgn-1
XK_TRIALS = []
for k in K_GRID:
    XK_TRIALS.append(
        ('x_top%d_d4' % k,
         list(order_ex[:k]), -1, 4.0))
    XK_TRIALS.append(
        ('x_bot%d_d4' % k,
         list(order_ex[::-1][:k]), -1, 4.0))
for (tname, coords, sgn, dsc) in XK_TRIALS:
    _run_coord_trial(tname, coords, sgn,
                     dsc, rows_A1, base12_A1)
# dose curve {0.5,1,2,4}
cd05 = _chg('x_co50ex_d0.5')
cd1 = BIT11['d_co50ex_d1.0']
cd2 = BIT11['d_co50ex_d2.0']
cd4 = BIT11['d_co50ex_d4.0']
curve = {'0.5': cd05, '1': cd1,
         '2': cd2, '4': cd4}
if cd05 < 0.5 * cd1 - 0.02:
    x_dose = 'co50ex_threshold'
elif cd05 > 0.5 * cd1 + 0.02 \
        and cd2 > 2 * cd1 + 0.02:
    x_dose = 'co50ex_superlinear'
else:
    x_dose = 'co50ex_graded'
log('X-GATE: co50ex dose curve %s -> %s'
    % (json.dumps({k: round(v, 4)
                   for k, v
                   in curve.items()}),
       x_dose))
# locus: top15 vs bot15 @d4
ct15 = _chg('x_top15_d4')
cb15 = _chg('x_bot15_d4')
if ct15 > cb15 + SIGN_TOL:
    x_locus = 'co50ex_locus_top'
elif cb15 > ct15 + SIGN_TOL:
    x_locus = 'co50ex_locus_bot'
else:
    x_locus = 'co50ex_locus_diffuse'
log('X-GATE: co50ex k-curve @d4 top %s '
    'bot %s -> %s'
    % (json.dumps({str(k): round(
        _chg('x_top%d_d4' % k), 4)
        for k in K_GRID}),
       json.dumps({str(k): round(
           _chg('x_bot%d_d4' % k), 4)
           for k in K_GRID}), x_locus))
# enrichment-direction cross-set test:
# bot15-of-co50ex stronger than top15?
enr_inv = (cb15 > ct15 + SIGN_TOL)
log('X-SOFT: enrichment inverse across '
    'sets (bot15 co50ex > top15): %s'
    % enr_inv)

# ================================================================
# PART T: pc1 instant replay (1 bit)
# ================================================================
log('== PART T: instant replay ==')
_run_vec_trial('t_pc1_l29_inst', 29,
               dv_pc1, 1.0, 0,
               rows_scan, base12_P)
if not SMOKE:
    got = E_res['t_pc1_l29_inst']['chg']
    m = abs(got - BIT11['t_pc1_l29_inst']) \
        < 1e-9
    n_bit_all += int(m)
    bit_anchors['t_pc1_l29_inst'] = {
        'got': got,
        'want': BIT11['t_pc1_l29_inst'],
        'match': bool(m)}
    log('T replay: +1 bit anchor (total '
        '%d/11)' % n_bit_all)

# ================================================================
# PART N: pcres d1 replays + neg mechanism
# ================================================================
log('== PART N: pcres replay + neg ==')
dv100 = (pc1_res
         / max(np.linalg.norm(pc1_res),
               1e-12) * pnorm38 * 1.0)     .astype(np.float32)
dv100_tile = np.tile(dv100[None, :],
                     (NCAP, 1))
_run_vec_trial('e_pcres38_d1.0_pos', 38,
               dv100_tile, 1.0, 'allstep',
               rows_scan, base12_P)
_run_vec_trial('e_pcres38_d1.0_neg', 38,
               -dv100_tile, 1.0, 'allstep',
               rows_scan, base12_P)
if not SMOKE:
    for tn in ('e_pcres38_d1.0_pos',
               'e_pcres38_d1.0_neg'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('N replay: +2 bit anchors (total '
        '%d/11)' % n_bit_all)
# N1: neg_d0.5 replay (soft anchor
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
log('N1 soft anchor vs 3146: %s'
    % n_repro_ok)
# flip rows
flip_rows = []
for j in range(NCAP):
    if fs_n[j] >= 0:
        flip_rows.append(j)
    if len(flip_rows) >= NFLIP_CAP:
        break
NF = len(flip_rows)
log('N2: %d flip rows (cap %d)'
    % (NF, NFLIP_CAP))
# tf sequences: per row, steps 0..t*
# (capped), injected vs base
n_seqs = []
n_meta = []
for j in flip_rows:
    tstar = int(fs_n[j])
    steps = list(range(
        0, min(tstar, NSTEP_CAP) + 1))
    for t in steps:
        n_seqs.append(
            list(PREFIX_IDS)
            + list(rows_scan[j])
            + list(_gen_n[j][:t]))
        n_meta.append((j, t, tstar))
NS = len(n_seqs)
log('N3: %d step-captures' % NS)
_KNB = CK['data'].get('nstepb')
if _KNB is not None:
    h_nb = {int(k):
            v.astype(np.float32)
            for k, v in _KNB['h'].items()}
    log('nstepb RESUMED')
else:
    h_nb = capture_states(
        n_seqs, [38, 39])
    ck_save('nstepb', {
        'h': {str(l): h_nb[l]
              .astype(np.float16)
              for l in (38, 39)}})
_KNI = CK['data'].get('nstepi')
if _KNI is not None:
    h_ni = {int(k):
            v.astype(np.float32)
            for k, v in _KNI['h'].items()}
    log('nstepi RESUMED')
else:
    _dv_all = np.tile(
        (-dv_n)[None, :], (NS, 1))
    h_ni = capture_states_inject2(
        n_seqs, [38, 39],
        [(38, _dv_all, 1.0)])
    ck_save('nstepi', {
        'h': {str(l): h_ni[l]
              .astype(np.float16)
              for l in (38, 39)}})
w_rows_np = {}
for tok in SPEC_TOKS:
    w_rows_np[tok] = \
        WUG[tok].to(torch.float32) \
        .cpu().numpy() \
        .astype(np.float64)
wdn_traj = {}
dlog_traj = {}
nrm_traj = {}
for si in range(NS):
    t = n_meta[si][1]
    dh = (h_ni[38][si].astype(np.float64)
          - h_nb[38][si].astype(np.float64))
    dh39 = (h_ni[39][si].astype(np.float64)
            - h_nb[39][si]
            .astype(np.float64))
    wdn_traj.setdefault(t, []).append(
        float(dh @ w_dn_g.astype(
            np.float64)))
    dlog_traj.setdefault(t, []).append(
        float(dh39 @ w_rows_np[SPEC_TOKS[0]]))
    nrm_traj.setdefault(t, []).append(
        float(np.linalg.norm(dh39)))
wdn_med = {str(t): float(np.median(v))
           for t, v in sorted(
               wdn_traj.items())}
dlog_med = {str(t): float(np.median(v))
            for t, v in sorted(
                dlog_traj.items())}
nrm_med = {str(t): float(np.median(v))
           for t, v in sorted(
               nrm_traj.items())}
log('N3-SOFT: wdn_traj %s'
    % json.dumps({k: round(v, 3)
                  for k, v
                  in wdn_med.items()}))
log('N3-SOFT: dlogit(%d)_traj %s'
    % (SPEC_TOKS[0],
       json.dumps({k: round(v, 3)
                   for k, v
                   in dlog_med.items()})))
log('N3-SOFT: ||dh39||_traj %s'
    % json.dumps({k: round(v, 2)
                  for k, v
                  in nrm_med.items()}))
ts_list = sorted(int(k)
                 for k in wdn_med)
neg_tag = 'neg_late_insufficient'
if len(ts_list) >= 2:
    t_early = ts_list[0]
    t_late = ts_list[-1]
    w_e = abs(wdn_med[str(t_early)])
    w_l = abs(wdn_med[str(t_late)])
    d_e = dlog_med[str(t_early)]
    d_l = dlog_med[str(t_late)]
    wdn_drop = w_l < 0.5 * max(w_e, 1e-9)
    dlog_grow = d_l > 2.0 * max(abs(d_e),
                                 0.05)
    if wdn_drop and not dlog_grow:
        neg_tag = 'neg_late_readout_comp'
    elif dlog_grow and not wdn_drop:
        neg_tag = 'neg_late_format_shift'
    elif wdn_drop and dlog_grow:
        neg_tag = 'neg_late_mixed'
    else:
        neg_tag = 'neg_late_static'
log('N-GATE: wdn |%.3f|->|%.3f| dlogit '
    '%.3f->%.3f -> %s'
    % (wdn_med[str(ts_list[0])]
       if ts_list else 0,
       wdn_med[str(ts_list[-1])]
       if ts_list else 0,
       dlog_med[str(ts_list[0])]
       if ts_list else 0,
       dlog_med[str(ts_list[-1])]
       if ts_list else 0, neg_tag))

# ================================================================
# PART V: v1 micro clip + amplitude traj
# ================================================================
log('== PART V: v1 micro ==')
v1f = v1.astype(np.float32)
V_TRIALS = [('v_base_a%03d' % int(a * 100),
             [(38, v1f, a, 'all')])
            for a in ALPHA_MICRO]
for (tname, cl) in V_TRIALS:
    _run_vec_trial(tname, 29, None, 1.0,
                   'allstep', rows_scan,
                   base12_P, clip=cl)
micro_vals = {tn: _chg(tn)
              for tn, _ in V_TRIALS}
micro_min = min(micro_vals.values())
if micro_min < CLEAN_GATE:
    v_micro = 'v1_micro_threshold'
else:
    v_micro = 'v1_micro_continuous'
log('V-GATE: micro clips %s (gate %.2f, '
    'min %.4f) -> %s'
    % (json.dumps({k: round(v, 4)
                   for k, v
                   in micro_vals.items()}),
       CLEAN_GATE, micro_min, v_micro))
# v1 amplitude trajectory (no
# intervention): base generation, hook
# L29 last-position projection on v1-hat
TRJ_ROWS = 4 if SMOKE else 16
_vn = v1f / (np.linalg.norm(v1f) + 1e-12)
_v_t = torch.as_tensor(
    np.asarray(_vn, dtype=np.float32),
    device='cuda')
_proj_store = []


def _traj_hook(mod, inp, out):
    o2 = out[0] \
        if isinstance(out, tuple) else out
    h = o2[:, -1, :].float()
    _proj_store.append(
        (h @ _v_t).detach().cpu()
        .numpy().astype(np.float64))
    return None


_hk = model_g.model.layers[29] \
    .register_forward_hook(_traj_hook)
try:
    trj_batches = []
    for b0 in range(0, TRJ_ROWS,
                    GEN_BATCH):
        batch = rows_scan[
            b0:b0 + GEN_BATCH]
        _proj_store.clear()
        ids_p, mask_p = _pad_batch(batch)
        t_ids = torch.tensor(ids_p,
                             device='cuda')
        t_mask = torch.tensor(mask_p,
                              device='cuda')
        with torch.inference_mode():
            model_g.generate(
                input_ids=t_ids,
                attention_mask=t_mask,
                max_new_tokens=N_NEW,
                do_sample=False,
                num_beams=1,
                pad_token_id=PADG)
        trj = np.array(_proj_store)
        trj_batches.append(trj.T)
finally:
    _hk.remove()
traj_all = np.concatenate(trj_batches,
                          axis=0)
n_fwd = traj_all.shape[1]
traj_med = [float(np.median(np.abs(
    traj_all[:, t])))
    for t in range(n_fwd)]
half = max(n_fwd // 2, 1)
amp_early = float(np.median(
    traj_med[1:half + 1])) \
    if n_fwd > 2 else traj_med[0]
amp_late = float(np.median(
    traj_med[half:]))
ratio_amp = amp_late / max(amp_early, 1e-9)
if ratio_amp > 1.3:
    v_traj = 'v1_amp_growing'
elif ratio_amp < 0.7:
    v_traj = 'v1_amp_decaying'
else:
    v_traj = 'v1_amp_stable'
log('V-SOFT: |h.v1| med traj %s -> %s '
    '(ratio %.2f)'
    % (json.dumps([round(v, 2)
                   for v in traj_med]),
       v_traj, ratio_amp))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3146_ok']
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == 11
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
tags.append(tail_sign)
tags.append(tb15_sym)
tags.append(tail_sub)
tags.append(x_dose)
tags.append(x_locus)
tags.append('enrich_inverse_crossset'
            if enr_inv
            else 'enrich_inverse_absent')
tags.append(neg_tag)
tags.append('neg_repro_3146'
            if n_repro_ok
            else 'neg_repro_drift')
tags.append(v_micro)
tags.append(v_traj)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3147,
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
        'res46_sha8': sha46,
        'seal46': SEAL46,
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
    'part_s': {
        'tail_pos': {'1': p1, '2': p2,
                     '4': p4t},
        'tail_neg': {'1': n1t, '2': n2t,
                     '4': n4t},
        'tail_sign': tail_sign,
        'tb15_pos_d1': _chg(
            's_tb15_pos_d1.0'),
        'tb15_pos_d2': bp2,
        'tb15_neg_d1': _chg(
            's_tb15_neg_d1.0'),
        'tb15_neg_d2': bn2,
        'tb15_sym': tb15_sym,
        'chg_tb5_d4': c5,
        'chg_tb10_d4': c10,
        'chg_tbot15_d4': b15n,
        'tail_sub': tail_sub,
        'tailpos_d2_got': p2,
        'tailpos_d2_repro':
            bool(tailpos2_repro)},
    'part_x': {
        'order_ex': [int(c)
                     for c in order_ex],
        'dose_curve': curve,
        'x_dose': x_dose,
        'k_top': {str(k): _chg(
            'x_top%d_d4' % k)
            for k in K_GRID},
        'k_bot': {str(k): _chg(
            'x_bot%d_d4' % k)
            for k in K_GRID},
        'x_locus': x_locus,
        'enrich_inverse': bool(enr_inv)},
    'part_n': {
        'chg_neg_d05': chg_n,
        'n_repro_ok': bool(n_repro_ok),
        'flip_rows': flip_rows,
        'fsteps': [int(fs_n[j])
                   for j in flip_rows],
        'n_seqs': NS,
        'wdn_traj': wdn_med,
        'dlog_traj': dlog_med,
        'nrm_traj': nrm_med,
        'neg_tag': neg_tag},
    'part_v': {
        'micro': micro_vals,
        'micro_min': micro_min,
        'v_micro': v_micro,
        'traj_med': traj_med,
        'amp_ratio': ratio_amp,
        'v_traj': v_traj},
    }
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'dvec19_sha': np.array([dvec19_sha]),
    'traj_abs_v1': traj_all,
    'wdn_traj': np.array(
        [wdn_med[k] for k
         in sorted(wdn_med,
                   key=lambda s: int(s))]),
    'dlog_traj': np.array(
        [dlog_med[k] for k
         in sorted(dlog_med,
                   key=lambda s: int(s))]),
    'nrm_traj': np.array(
        [nrm_med[k] for k
         in sorted(nrm_med,
                   key=lambda s: int(s))]),
    'step_axis': np.array(
        sorted(int(k) for k in wdn_med)),
    'dose_curve_co50ex': np.array(
        [curve['0.5'], curve['1'],
         curve['2'], curve['4']]),
    'order_ex': order_ex.astype(np.int64),
    'pc1_res': pc1_res.astype(
        np.float32)}
np.savez(os.path.join(OUT,
                      'p145_readout.npz'),
         **npz_out)
if not SMOKE:
    try:
        os.remove(CKPTF)
        log('ckpt cleaned (final)')
    except OSError:
        pass
log('result.json + npz written. DONE '
    '(verdict %s)' % verdict)
