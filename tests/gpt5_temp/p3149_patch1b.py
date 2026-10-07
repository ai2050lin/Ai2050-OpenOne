# -*- coding: utf-8 -*-
"""p3149 patch1b: constants block, anchors
block, SEAL block. Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()


def rep(old, new, tag):
    global s
    n = s.count(old)
    assert n == 1, (tag, n)
    s = s.replace(old, new)
    print('OK', tag)


# 1) constants: SPEC_TOKS..RNG_SEED
OLD_CONST = """SPEC_TOKS = [131401, 134772, 23386]
SIGN_TOL = 0.03
XDOSE_MID = (2.5, 3.0, 3.5)
H_DOSES = (0.5, 1.0, 2.0)
HEAD_COORDS = [2530, 3755]
NHEADS = 32
HEAD_ROWS = 4 if SMOKE else 32
V2_ALPHA = ((-0.25, 'v2_amp_a025'),
            (-0.5, 'v2_amp_a050'),
            (0.25, 'v2_clip_a025'),
            (0.5, 'v2_clip_a050'))
UCANCEL_TOL = 0.05
RNG_SEED = 3148"""
NEW_CONST = """SPEC_TOKS = [131401, 134772, 23386]
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
RNG_SEED = 3149"""
rep(OLD_CONST, NEW_CONST, 'constants')

# 2) anchors block (EXP_V47..LEDGER_N)
i1 = s.index(
    "# ---------------- frozen 3147 "
    "anchors --")
i2 = s.index("# ---------------- seal "
             "------------------")
NEW_ANCH = """# ---------------- frozen 3148 anchors --
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
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
LEDGER_N = 285

"""
s = s[:i1] + NEW_ANCH + s[i2:]
print('OK anchors block')

# 3) SEAL block
j1 = s.index('SEAL = {')
j2 = s.index("SEALF = os.path.join(OUT, "
             "'design_seal.json')")
NEW_SEAL = """SEAL = {
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
"""
s = s[:j1] + NEW_SEAL + s[j2:]
print('OK SEAL block')

io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH1B_DONE')
