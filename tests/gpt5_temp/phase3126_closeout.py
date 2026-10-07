# -*- coding: utf-8 -*-
"""Phase 3126 closeout (idempotent):
result asserts -> Ledger -> MEMO Phase 3126 ->
workspace logs (x2 entries) -> MEMORY.md.
@@...@@ placeholders MUST be filled from
p3126_vals.txt before execution."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3126'
        r'\omega_p124_glm4_anchoredlast_'
        'regen_writechain')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now()
NOWS = NOW.strftime('%Y-%m-%d %H:%M')
WDATE = NOW.strftime('%Y-%m-%d')
o = []

# ======== FILL CONSTANTS AFTER FULL RUN
# ======== (source: p3126_vals.txt) ========
VERDICT = ('long_range_trail_absent|'
           'trail_significant|'
           'trail_shared_directions|'
           'second_order_absent|'
           'path_valid|replay_bit_exact|'
           'readout_ok|spans_ok|'
           'baseline_replicated|'
           'qis_syntax_L20|qis_syntax_L20|'
           'qis_content_L20|qis_content_L20|'
           'qis_syn_final_positive|'
           'qis_syn_final_negative|'
           'qis_cont_final_positive|'
           'qis_cont_final_positive|'
           'behavior_shift_present|'
           'writechain_diffuse')
RUNTIME = float('10650.0')
# part_a refit
S_P = float('-0.601611613690753')
MS_P = float('-4.955594255793292')
S_A1 = float('-0.5540798866844862')
MS_A1 = float('-7.154421990932929')
# decomp6 P
D6P_CM = float('0.19001305759179912')
D6P_M = float('0.0033137063011249046')
D6P_TR = float('0.14541402213417262')
D6P_AR = float('0.008850062321321888')
D6P_REM = float('0.6524091516515815')
D6P_SSTOT = float('69737.75834882268')
# decomp6 A1
D6A_CM = float('0.35574441732458434')
D6A_M = float('3.0830390620402785e-06')
D6A_TR = float('0.05445240009894038')
D6A_AR = float('0.007852489562643112')
D6A_REM = float('0.5819476099747701')
D6A_SSTOT = float('103458.4879387006')
# perm
PERM_P_OBS = float('0.14541402213417262')
PERM_P_Z = float('28.41408758211333')
PERM_A_OBS = float('0.05445240009894038')
PERM_A_Z = float('10.863519222963433')
G_CORR = float('0.5763224540872617')
# G6 samples
G6P_0_0 = float('1.99187686456906')
G6P_0_1 = float('-1.7657426415876498')
G6A_0_1 = float('-1.5199730940106253')
# second order
SEC_P_J = float('0.029279717355663')
SEC_P_Q = float('0.00016060415524816302')
SEC_A_J = float('0.011587394070033114')
SEC_A_Q = float('0.0016012286768257377')
# part_b
PATH_R = float('0.9999857905405053')
DREP = float('0.0')
AUC_FIN = float('0.8313137755102041')
NSPAN_P = int('672')
NSPAN_A1 = int('672')
# lstar (ints; 'None' allowed when
# below threshold)
def _li(s):
    return None if s == 'None' \
        else int(s)


LS_P_SYN_REL = _li('20')
LS_P_SYN_ABS = _li('20')
LS_P_CONT_REL = _li('20')
LS_P_CONT_ABS = _li('20')
LS_A_SYN_REL = _li('20')
LS_A_SYN_ABS = _li('20')
LS_A_CONT_REL = _li('20')
LS_A_CONT_ABS = _li('20')
# E curve samples
E_SYN_P_AT_LSTAR = float('-0.12357999268414346')
E_SYN_P_FINAL = float('0.0593891150570533')
E_SYN_A1_AT = float('-0.10777419752820006')
E_SYN_A1_FINAL = float('-0.1140573490683959')
E_CONT_P_AT = float('-0.3004961357086951')
E_CONT_P_FINAL = float('0.3408039071573277')
E_CONT_A1_AT = float('-0.2806829876836132')
E_CONT_A1_FINAL = float('0.08595561432300791')
# vs3124
CORR_SYN_P = float('0.9999999999999998')
CORR_SYN_A1 = float('0.999999999999999')
CORR_CONT_P = float('0.9999999999999997')
CORR_CONT_A1 = float('0.9999999999999998')
# part_d wspec
C3_P = float('0.21191781524575162')
C3_A1 = float('0.21518741851620496')
TOP3_P = '[8, 29, 9]'
TOP3_A1 = '[29, 8, 13]'
PEAK_P = int('8')
PEAK_A1 = int('29')
# part_c
SHIFT_MIN = float('0.7526041666666666')
AGREE_S0_P = float('1.0')
AGREE_MIN_P = float('0.18923611111111108')
AGREE_S0_A1 = float('1.0')
AGREE_MIN_A1 = float('0.13020833333333331')
FLIP_S1_P = '0.8913'
FLIP_S2_P = '0.7333'
FLIP_S3_P = '0.3846'
FLIP_S1_A1 = '0.2247'
FLIP_S2_A1 = '0.2976'
FLIP_S3_A1 = '0.6667'
FTOK_P = 'Yes 670 / No 2'
FTOK_A1 = ('No 555 / Yes 90 / no 16 / '
           'The 11')
# ======== END FILL CONSTANTS ========
_ALL_CONSTS = [VERDICT, RUNTIME,
               S_P, MS_P, S_A1, MS_A1,
               D6P_CM, D6P_M, D6P_TR, D6P_AR,
               D6P_REM, D6P_SSTOT,
               D6A_CM, D6A_M, D6A_TR, D6A_AR,
               D6A_REM, D6A_SSTOT,
               PERM_P_OBS, PERM_P_Z,
               PERM_A_OBS, PERM_A_Z, G_CORR,
               G6P_0_0, G6P_0_1, G6A_0_1,
               SEC_P_J, SEC_P_Q, SEC_A_J,
               SEC_A_Q, PATH_R, DREP, AUC_FIN,
               NSPAN_P, NSPAN_A1,
               LS_P_SYN_REL, LS_P_SYN_ABS,
               LS_P_CONT_REL, LS_P_CONT_ABS,
               LS_A_SYN_REL, LS_A_SYN_ABS,
               LS_A_CONT_REL, LS_A_CONT_ABS,
               E_SYN_P_AT_LSTAR, E_SYN_P_FINAL,
               E_SYN_A1_AT, E_SYN_A1_FINAL,
               E_CONT_P_AT, E_CONT_P_FINAL,
               E_CONT_A1_AT, E_CONT_A1_FINAL,
               CORR_SYN_P, CORR_SYN_A1,
               CORR_CONT_P, CORR_CONT_A1,
               C3_P, C3_A1, TOP3_P, TOP3_A1,
               PEAK_P, PEAK_A1, SHIFT_MIN,
               AGREE_S0_P, AGREE_MIN_P,
               AGREE_S0_A1, AGREE_MIN_A1,
               FLIP_S1_P, FLIP_S2_P,
               FLIP_S3_P, FLIP_S1_A1,
               FLIP_S2_A1, FLIP_S3_A1,
               FTOK_P, FTOK_A1]


def _no_ph(vals):
    for v in vals:
        if '@@' in str(v):
            return False
    return True


V = VERDICT
PA_V = V.split('|')
A_LONG_V = PA_V[0]
A_SIG_V = PA_V[1]
A_SHARE_V = PA_V[2]
A_SECOND_V = PA_V[3]
B_BASE_V = PA_V[8]
C_SHIFT_V = PA_V[17]
D_CHAIN_V = PA_V[18]

# ---------- 1. result.json asserts ----------
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
_n = [0]


def chk(cond):
    assert cond, 'assert #%d failed' % len(_n)
    _n.append(1)


chk(_no_ph(_ALL_CONSTS))
chk(res['phase'] == 3126)
chk(res['name'] == 'omega_p124_glm4_'
    'anchoredlast_regen_writechain')
chk(res['verdict'] == V)
chk(res['smoke'] is False)
chk(res['n_pairs'] == 672)
chk(res['np_b'] == 672)
chk(res['np_reg'] == 96)
chk(abs(res['runtime_s'] - RUNTIME) < 0.05)
pa = res['part_a']
chk(abs(pa['refit']['P']['S'] - S_P) < 1e-12)
chk(abs(pa['refit']['P']['MS'] - MS_P)
    < 1e-12)
chk(abs(pa['refit']['A1']['S'] - S_A1)
    < 1e-12)
chk(abs(pa['refit']['A1']['MS'] - MS_A1)
    < 1e-12)
chk(abs(pa['decomp3_repl']['P']
        - 0.08599867672568678) < 1e-12)
chk(abs(pa['decomp3_repl']['A1']
        - 0.05022617239377846) < 1e-12)
d6P = pa['decomp6']['P']
chk(abs(d6P['cm_share'] - D6P_CM) < 1e-12)
chk(abs(d6P['m_share'] - D6P_M) < 1e-12)
chk(abs(d6P['trail6_share'] - D6P_TR)
    < 1e-12)
chk(abs(d6P['ar6_share'] - D6P_AR) < 1e-12)
chk(abs(d6P['rem6_share'] - D6P_REM)
    < 1e-12)
chk(abs(d6P['ss_tot'] - D6P_SSTOT) < 1e-9)
chk(abs(d6P['leak']) < 1e-9)
d6A = pa['decomp6']['A1']
chk(abs(d6A['cm_share'] - D6A_CM) < 1e-12)
chk(abs(d6A['m_share'] - D6A_M) < 1e-12)
chk(abs(d6A['trail6_share'] - D6A_TR)
    < 1e-12)
chk(abs(d6A['ar6_share'] - D6A_AR) < 1e-12)
chk(abs(d6A['rem6_share'] - D6A_REM)
    < 1e-12)
chk(abs(d6A['ss_tot'] - D6A_SSTOT) < 1e-9)
chk(abs(d6A['leak']) < 1e-9)
lg = pa['long_gate']
chk(abs(lg['trail3_min']
        - 0.05022617239377846) < 1e-12)
chk(lg['verdict'] == A_LONG_V)
chk(pa['perm']['P']['z'] == PA_V[1]
    or abs(pa['perm']['P']['z'] - PERM_P_Z)
    < 1e-9)
chk(abs(pa['perm']['A1']['z'] - PERM_A_Z)
    < 1e-9)
chk(pa['perm']['P']['reps'] == 200)
chk(pa['perm']['P']['seed'] == 3126)
chk(abs(pa['perm']['P']['obs']
        - PERM_P_OBS) < 1e-9)
chk(abs(pa['perm']['A1']['obs']
        - PERM_A_OBS) < 1e-9)
chk(abs(pa['g_struct']['corr'] - G_CORR)
    < 1e-9)
chk(pa['g_struct']['verdict'] == A_SHARE_V)
chk(abs(pa['trail6_G']['P'][0][0]
        - G6P_0_0) < 1e-12)
chk(abs(pa['trail6_G']['P'][0][1]
        - G6P_0_1) < 1e-12)
chk(abs(pa['trail6_G']['A1'][0][1]
        - G6A_0_1) < 1e-12)
chk(abs(pa['second']['P']['joint_gain']
        - SEC_P_J) < 1e-9)
chk(abs(pa['second']['P']['mquad_gain']
        - SEC_P_Q) < 1e-9)
chk(abs(pa['second']['A1']['joint_gain']
        - SEC_A_J) < 1e-9)
chk(abs(pa['second']['A1']['mquad_gain']
        - SEC_A_Q) < 1e-9)
chk(pa['second_verdict'] == A_SECOND_V)
pb = res['part_b']
chk(pb['interference']
    == 'input_prompt_span_equal_len_'
    'anchored_last')
chk(pb['ids']['n_layers'] == 40)
gpb = pb['gen_probe_bos']
chk(gpb['prefix_ids'] == [151331, 151333])
chk(gpb['prefix_decoded']
    == '[gMASK]<sop>')
chk(gpb['gen_enc_len'] - gpb['plain_len'] == 2)
chk(pb['path']['verdict'] == 'path_valid')
chk(abs(pb['path']['r'] - PATH_R) < 1e-9)
chk(pb['repro']['max_diff'] == DREP)
chk(pb['repro']['verdict'] in
    ('replay_bit_exact', 'replay_ok'))
chk(abs(pb['readout_auc'] - AUC_FIN)
    < 1e-12)
chk(pb['readout_verdict'] == 'readout_ok')
chk(pb['n_span'] == {'P': NSPAN_P,
                     'A1': NSPAN_A1})
chk(pb['spans_verdict'] in
    ('spans_ok', 'spans_sparse'))
lsp = pb['lstar']
chk(lsp['P']['syn'] == {'rel': LS_P_SYN_REL,
                        'abs': LS_P_SYN_ABS})
chk(lsp['P']['cont']
    == {'rel': LS_P_CONT_REL,
        'abs': LS_P_CONT_ABS})
chk(lsp['A1']['syn']
    == {'rel': LS_A_SYN_REL,
        'abs': LS_A_SYN_ABS})
chk(lsp['A1']['cont']
    == {'rel': LS_A_CONT_REL,
        'abs': LS_A_CONT_ABS})
chk(pb['sign']['P']['syn']
    == ('positive' if E_SYN_P_FINAL > 0
        else 'negative'))
chk(pb['sign']['A1']['syn']
    == ('positive' if E_SYN_A1_FINAL > 0
        else 'negative'))
chk(pb['sign']['P']['cont']
    == ('positive' if E_CONT_P_FINAL > 0
        else 'negative'))
chk(pb['sign']['A1']['cont']
    == ('positive' if E_CONT_A1_FINAL > 0
        else 'negative'))
cv = pb['curves']
chk(cv['E_syn_P'][0] == 0.0)
chk(abs(cv['E_syn_P'][40]
        - E_SYN_P_FINAL) < 1e-12)
chk(abs(cv['E_syn_A1'][40]
        - E_SYN_A1_FINAL) < 1e-12)
chk(abs(cv['E_cont_P'][40]
        - E_CONT_P_FINAL) < 1e-12)
chk(abs(cv['E_cont_A1'][40]
        - E_CONT_A1_FINAL) < 1e-12)
vs = pb['vs_3124']
chk(abs(vs['corr']['syn_P'] - CORR_SYN_P)
    < 1e-9)
chk(abs(vs['corr']['syn_A1']
        - CORR_SYN_A1) < 1e-9)
chk(abs(vs['corr']['cont_P']
        - CORR_CONT_P) < 1e-9)
chk(abs(vs['corr']['cont_A1']
        - CORR_CONT_A1) < 1e-9)
chk(vs['verdict'] == B_BASE_V)
pc = res['part_c']
chk(pc['np_reg'] == 96)
chk(abs(pc['shift_min'] - SHIFT_MIN)
    < 1e-12)
chk(pc['verdict'] == C_SHIFT_V)
pd_ = res['part_d']
chk(abs(pd_['c3_min']
        - min(C3_P, C3_A1)) < 1e-9)
chk(pd_['verdict'] == D_CHAIN_V)
o.append('asserts ok (%d checks)' % len(_n))

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3126
           for m in led['measurements']):
    claim = (
        'Omega-P124 (3126, T4 ninth phase: '
        'GLM4 anchored-last re-run + '
        'counterfactual regeneration + '
        'write-chain spectrum + trail '
        'deepening, offline 672 + glm4-9b '
        'GPU 672x4 forwards + 96x4x2 regen, '
        '10650.0s) - verdict ' + V + '.  '
        'Part A (offline, frozen '
        '3118/3120/3124/3125): D=6 trail '
        'kernel shares P 0.145 / A1 0.054 '
        'with permutation z 28.4 / 10.9 '
        '(200 within-trajectory perms, '
        'full-pipeline recompute per rep) '
        '- the 3125 trail kernel is NOT '
        'overfitting; D3->D6 gain P +0.059 '
        '(0.086->0.145, real long-range '
        'component at lag 4-6) vs A1 '
        '+0.004 -> direction-asymmetric '
        'long-range tail (min-gate '
        'long_range_trail_absent is the '
        'conservative A1-driven call); G '
        'table cross-direction corr 0.576 '
        '(shared), lag-2 energy dominates '
        'both; second-order (joint '
        '(cls_k,cls_{k-1}) 100-cell + '
        'm-quadratic) gains 0.029/0.012 -> '
        'absent, first-order conditional '
        'structure already captures the '
        'systematic part; remainder still '
        '0.652/0.582.  Part B (GPU glm4-9b '
        '40L, [gMASK]<sop> prefix fix + '
        'decode-polarity fix, path r '
        '0.999986, replay bit-exact, '
        'n_span 672/672, readout AUC '
        '0.831): GLM4 E_syn/E_cont L*=20/20 '
        'ALL FOUR (identical to 3124), '
        'final signs P syn + / A1 syn - / '
        'cont both + -> sign PATTERN is '
        'MODEL-SPECIFIC (Qwen 3125 '
        'all-negative); CRITICAL: vs-3124 '
        'curve corr = 1.0000000 x4 -> '
        'baseline_replicated: the 3125 '
        'SPANPROBE first-occurrence warning '
        '(P first=185 Facts line) did NOT '
        'affect the 3124 curves - the 3124 '
        'suffix-checked span location '
        'actually hit the query line '
        '(185 has no " Is this" suffix and '
        'was skipped) -> the 3124 '
        'P-direction Facts-line-collision '
        'concern is WITHDRAWN and 3124 P '
        'vs 3126 P are directly comparable. '
        ' Part C (96 pairs x 4 conditions '
        'real counterfactual generation): '
        's0 agreement 1.0 (greedy-consistent '
        'baseline), s1-s3 agreement '
        '0.13-0.34, first divergence 0.4-3.1 '
        'steps, answer flip 0.22-0.89 -> '
        'shift_min 0.753 '
        'behavior_shift_present: input-'
        'stream perturbations strongly '
        'change generation, so the '
        'fixed-replay readout conclusions '
        '(E curves / L* / signs) are '
        'behaviorally grounded.  Part D '
        '(write-chain spectrum wspec[L,k] '
        'from s0 margins): c3 0.212/0.215 '
        '-> writechain_diffuse (GLM4 margin '
        'write energy is dispersed, unlike '
        'Qwen L28-34 concentration); shared '
        'peaks L8 (0.20 depth) + L29 (0.72 '
        'depth) in both directions top3.  '
        'NEXT 3127: cross-model write-chain '
        'ports (GLM4 L8/L29 vs Qwen '
        'L26-L35, gradient attribution '
        'fallback), A1-direction long-range '
        'trail closure, cross-trajectory '
        'conditional second-order on pooled '
        'residuals, full-672 regen with '
        'No/no flip separation.')
    assert '@@' not in claim, \
        'CLAIM placeholder not filled'
    meas = {
        'meas_id': 'meas3126_omega_p124_glm4_'
                   'anchoredlast_regen_writechain',
        'phase': 3126,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen '
                   'before computation: A long '
                   'gate min(trail6)-min(trail3) '
                   '>=0.02 present else absent, '
                   'A_sig min z >=4 (200 '
                   'deterministic perms rng '
                   '3126), A_g corr >=0.5 '
                   'shared, A_second min-max '
                   'gain >=0.05, B_path corr '
                   '>0.9999 FATAL, B_repro '
                   '<=1e-6, B_readout AUC >0.6, '
                   'B_spans >=300, vs3124 '
                   'a1_min_corr >=0.9 '
                   'replicated, C_shift '
                   's0-vs-mean(s1..s3) >=0.10, '
                   'D_chain c3 >=0.4 located; '
                   'deterministic: crc32 seeds '
                   '+ fixed rng 3126, no MC',
        'artifacts': {
            'result': 'phase3126/omega_p124_'
                      'glm4_anchoredlast_regen_'
                      'writechain/result.json',
            'seal': 'phase3126/omega_p124_'
                    'glm4_anchoredlast_regen_'
                    'writechain/seal.json',
            'readout': 'phase3126/omega_p124_'
                       'glm4_anchoredlast_regen_'
                       'writechain/'
                       'p124_readout.npz'},
        'hashes': {},
        'note': 'GPU glm4-9b-chat-hf only '
                '(40L BF16 eager, batch1 '
                'forwards + batched greedy '
                'gen with [gMASK]<sop> prefix; '
                'teacher-forced forwards stay '
                'plain-encoded; Part A offline '
                'on frozen qwen3-4b data; '
                '2 SMOKE iterations before '
                'full run: (1) plain-ids gen '
                'lost the [gMASK]<sop> prefix '
                '-> True/False style, readout '
                'AUC 0.44, vs3124 corr '
                'collapsed -> prepend '
                'tokenizer default prefix '
                '[151331,151333], (2) first-'
                'token polarity via id '
                'equality missed 9450 Yes / '
                '2753 No -> decode-text '
                'startswith polarity',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][
        0]
    l14['connects'].append(
        'meas3126_omega_p124_glm4_anchoredlast_'
        'regen_writechain')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- 3. MEMO Phase 3126 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3126:' not in memo:
    sec = u'''## Phase 3126: Ω-P124 GLM4 anchored-last 修正重跑 + 反事实生成对照 + 写入链谱 + trail 长程核深化（T4 第9Phase）——**vs3124 四方向 corr≈1.0000=baseline_replicated：3125 警告的 P 方向 Facts 行碰撞被证伪（3124 后缀校验跳过无 ' Is this' 后缀的 185、实际命中的就是查询行 255）——3124/3126 P 完全可比；GLM4 L\\*=20/20 四方向复现、终层符号 P 正/A1 负（syn）+ cont 全正=模型特异指纹；trail 核置换 z 28.4/10.9 高度显著（非过拟合）但 A1 无长程（P D3→D6 +0.059）；行为层面 s1–s3 扰动强烈改变生成（shift 0.753、flip 0.22–0.89）=固定重放读出结论行为有效；写入链谱弥散 c3≈0.21=writechain_diffuse 但双方向共享峰 L8+L29** [2026-09-24 04:50]

**性质**：T4 第 9 Phase，3125 MEMO 第 5 节预注册四项全部执行（①GLM4 anchored-last 重跑 ②反事实生成 ③写入链层识别 ④trail 深化）。Part A offline 全量 672（冻结 3118/3120/3124/3125，refit 与 D3 复现断言 1e-12）；Part B/C/D GPU glm4-9b-chat-hf（40 层 BF16）10650.0s。SMOKE 2 轮修复后全绿：①plain-ids 生成丢 tokenizer 默认前缀 [gMASK]<sop>（[151331,151333]）→ True/False 风格、读出 AUC 0.44、vs3124 corr 崩塌 → 生成输入统一加前缀（teacher-forced forward 保持 plain 编码，与 3124 一致）；②首 token 极性判定 id 相等漏配 GLM4 实际发射 9450 'Yes'/2753 'No' → decode 文本 startswith 判定。

### 1. 三大发现（重复三遍）
1. **3125 的"P 方向 Facts 行碰撞"警告被证伪——GLM4 曲线与行身份无关**。SPANPROBE：P occurrences=2（first=185 Facts 行、anchored=255 查询行）、A1 唯一 255；但 3126（anchored-last 替换查询行）与 3124 曲线四方向 corr=0.9999999999999998/0.9999999999999997/0.999999999999999/0.9999999999999998=**baseline_replicated**。机理：3124 的 span 定位同样带 ' Is this' 后缀校验，Facts 行（185）后无该后缀被跳过、实际命中的就是查询行 255——3125 SPANPROBE 的 first=185 只是纯文本 find 结果，不代表干预位置。**结论：3124 P 方向无混杂，3124 与 3126 P 方向曲线完全可比**（3125 硬伤⑦撤销；"行身份是实验设计变量"保留为方法论纪律）。
2. **GLM4 读出几何：L\\*=20/20 四方向全复现 + 终层符号 P/A1 分裂=模型特异指纹**。E_syn 与 E_cont 在 P/A1 两方向的 rel/abs 层号全部 =20（0.50 深度，与 3124 完全一致）；终层符号 syn_P +0.0594（正）/syn_A1 −0.1141（负）/cont_P +0.3408（正）/cont_A1 +0.0860（正）。对照 Qwen 3125 全负 → **符号模式（不只是单方向符号）是模型特异指纹**；跨模型不变量进一步收窄为"相对深度带（~0.5 深度写入链上游涌现）"。readout AUC 0.8313（P/A1 s0 末层末位分离良好，3125 trail_present 判决之外读出侧亦稳健）。
3. **trail 核确证显著 + 方向不对称长程 + 二阶缺席 + 行为有效性**。(a) 置换显著性（轨迹内 200 次全管线重算、seed 3126）：z=28.41(P)/10.86(A1)、p̂=0——**3125 trail 核非过拟合（硬伤②回应）**；(b) D3→D6：P 0.086→0.145（+0.059 真实长程成分 lag 4–6，主项 cls1 lag6 −2.07、cls9 lag2/4 +2.35/+1.27）vs A1 0.050→0.054（+0.004）→ **P 方向内容尾迹延伸至 lag 6、A1 截断于短程**（min 门保守判 long_range_trail_absent）；(c) 二阶（joint 100-cell + m²）gain 0.0293/0.0116 absent——一阶条件结构已捕获系统成分，余项 0.652/0.582 主体仍需新候选项；(d) **反事实生成（96 对×4 条件真实重生成）：s0 agreement=1.0、扰动后 0.13–0.34、首分叉 0.4–3.1 步、答案翻转 0.22–0.89、shift_min=0.753 → behavior_shift_present——输入流扰动的行为效应强烈，固定重放读出结论（E 曲线/L\\*/符号）行为有效**。

### 2. 关键数值
Part A：decomp6 P {cm 0.19001305759179912, m 0.0033137063011249046, trail6 0.14541402213417262, ar6 0.008850062321321888, rem 0.6524091516515815}、A1 {cm 0.35574441732458434, m 3.0830390620402785e-06, trail6 0.05445240009894038, ar6 0.007852489562643112, rem 0.5819476099747701}；AR(2) φ1 −0.1229/−0.1168、φ2 −0.0406/−0.0779；perm z 28.41408758211333/10.863519222963433（null 0.0520±0.0033/0.0314±0.0021、p̂=0）；G corr 0.5763224540872617；lag 能量 P [11.70,18.99,11.18,8.22,6.24,8.18]、A1 [0.72,11.04,10.66,4.42,5.94,5.00]（lag2 主导）；second joint 0.0293/0.0116、mquad 0.0002/0.0016。Part B：path r 0.9999857905405053、repro 0.0、readout_auc 0.8313137755102041、n_span 672/672；L\\* 全 20/20；E_syn_P [20] −0.1236/[40] +0.0594；E_syn_A1 [20] −0.1078/[40] −0.1141；E_cont_P [20] −0.3005/[40] +0.3408；E_cont_A1 [20] −0.2807/[40] +0.0860；vs3124 corr 四方向 ≈1.0000。Part C：agree P {s0 1.0, s1 0.2118, s2 0.25, s3 0.1892}、A1 {s0 1.0, s1 0.2682, s2 0.3438, s3 0.1302}；fdiv P {s1 0.5, s2 1.46, s3 1.13}、A1 {s1 2.43, s2 3.09, s3 0.36}；flip P {s1 0.8913, s2 0.7333, s3 0.3846}、A1 {s1 0.2247, s2 0.2976, s3 0.6667}；shift_min 0.7526041666666666。Part D：c3 P 0.21191781524575162（top3 [8,29,9]、peak L8 depth 0.20）、A1 0.21518741851620496（top3 [29,8,13]、peak L29 depth 0.725）。first-tok：P {Yes 670, No 2}、A1 {No 555, Yes 90, no 16, The 11}。

### 3. 硬伤
① trail6 的 D=6 核 60 参数在 8064 样本上无正则拟合——虽经置换显著性检验（z 28.4/10.9）仍是个案内显著、跨材料泛化未测；② long_range 门用 min(P,A1)——A1 无长程拖累判决，P 方向 +0.059 单独看是 present（方向不对称本身是发现，但门设计保守）；③ 写入链谱 c3≈0.21 diffuse——wspec[L,k]=mean_j 邻层 margin 差把 672 对全摊平，可能稀释稀疏写入带（Qwen 3122 同法 c3≥0.4 集中，GLM4 无集中带可能是模型差异也可能是度量敏感度不足）；④ 反事实 flip 度量首 token decode 极性把 'no'(2152) 与 'No'(2753) 混计为负极性；⑤ Part C 只测 96 对（全量 14%）；⑥ trail 核与写入链谱在两模型上用不同轨迹材料（Qwen 3118/3120 冻结 vs GLM4 Part B）——跨模型 trail 比较未做；⑦ 二阶 joint 核 100 参数同样无正则（absent 结论因此保守可信——过拟合会高估 gain）。

### 4. 机制拼图更新
内部响应图谱：① GLM4 全 41 层 × 4 条件 × 672 × 13 lens margin 场（mlg npz float32）+ E 曲线 + L\\* + 终层符号（跨模型对照第三块拼图：Qwen 输出流/Qwen 输入流/GLM4 输入流）；② D=6 trail6 核 G 表（10 类 × 6 lag × 2 方向）+ 置换 z + lag 能量谱——**内容尾迹时间结构定稿：lag2 主导、P 长程延伸 A1 短程**；③ span_idx 672×2×2 存档 + gen/regen 序列（96×12×4×2）——反事实行为对照数据集；④ 写入链谱 wspec (40×13×2) + top3/c3/pos/neg 带。三图谱关联：**写入链上游涌现带（~0.5 深度）跨模型稳健（GLM4 L20 = Qwen L21/L20/L23），精确层号与符号模式模型特异**；RDC 更新：① 残差条件结构一阶完备性初步成立（二阶 absent、AR 弱、m²≈0）——剩余 0.58–0.65 主体是"非条件性"成分（候选：跨轨迹共模子空间结构、坐标级稀疏事件）；② E_cont 符号指纹表更新：GLM4 {syn P+, A1 −, cont P+, A1 +} vs Qwen 全负；③ 行为有效性闭环首次建立（读出结论 ↔ 真实生成翻转）。

### 5. 3127 预注册（T4 继续，观测前冻结框架）
① **写入链端口跨模型对照（GPU Qwen + GLM4）**：对 L8/L29（GLM4）与 L26–L35（Qwen）做 per-layer 干预或梯度归因，检验"写入链上游带"的功能等价性（不满足于谱相似）；② **A1 长程尾迹闭合（offline）**：A1 方向为何 lag≥4 无能量——检验是材料（A1 关系分布）还是编码（短程化）原因；③ **剩余 0.58–0.65 主体的新候选（offline）**：跨轨迹条件二阶统计（区别于本 Phase 轨迹内 joint）+ 坐标级稀疏事件谱；④ **反事实生成扩展（GPU）**：96→全量 672 + flip 度量分离 'No'/'no' + 多步翻转追踪。具体门在 3127 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3126/omega_p124_glm4_anchoredlast_regen_writechain/`（result.json、design_seal.json、run_log.txt、p124_readout.npz）；脚本 `tests/glm5/phase3126_omega_p124_glm4_anchoredlast_regen_writechain.py`；复核 `tests/gpt5_temp/p3126_disk_verify.py`。
'''
    assert '@@' not in sec, \
        'MEMO_SECTION placeholder not filled'
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    _memo_delta = len(sec)
    o.append('memo +%d chars (Phase 3126)'
             % len(sec))
else:
    _memo_delta = 0
    o.append('memo already appended')

# ---------- 4. workspace logs (x2) ----------
line_exp = ('- Phase 3126 Omega-P124 (T4 ninth '
            'phase: GLM4 anchored-last re-run '
            '+ counterfactual regen + '
            'write-chain spectrum + trail '
            'deepening, offline 672 + GPU '
            '672x4 + 96x4x2, 10650.0s): '
            'verdict ' + V + '. '
            '(A) Trail kernel confirmed: perm '
            'z 28.41/10.86 (p_hat=0), NOT '
            'overfitting; D3->D6 P +0.059 '
            '(real long-range lag 4-6) vs A1 '
            '+0.004 -> direction-asymmetric; '
            'G corr 0.576 shared, lag-2 '
            'dominates; second-order absent '
            '(joint 0.029/0.012). (B) GLM4 '
            'L*=20/20 all four, final signs '
            'P+/A1- (syn), cont both + = '
            'model-specific fingerprint; '
            'vs3124 corr 1.0000000 x4 = '
            'baseline_replicated -> 3125 '
            'P-direction Facts-line collision '
            'WITHDRAWN (3124 suffix check '
            'actually hit the query line '
            '255). (C) Counterfactual regen: '
            's0 agree 1.0, s1-s3 0.13-0.34, '
            'fdiv 0.4-3.1, flip 0.22-0.89, '
            'shift_min 0.753 -> '
            'behavior_shift_present, '
            'fixed-replay conclusions '
            'behaviorally grounded. (D) '
            'Write-chain c3 0.212/0.215 '
            'diffuse, shared peaks L8+L29. '
            'NEXT 3127: cross-model '
            'write-chain ports + A1 long-'
            'range closure + pooled-residual '
            'second-order + full-672 regen.\n')
assert '@@' not in line_exp, \
    'WLOG_EXP placeholder not filled'
line_clo_tpl = ('- Phase 3126 closeout finished: '
                'five-write chain ok (ledger n='
                '%LGN% l14=%L14N% sha=%SHA8%, '
                'MEMO +%MEMOC% chars, dual wlog, '
                'MEMORY.md update); disk verify '
                'next. 2 SMOKE iterations '
                '([gMASK]<sop> prefix fix for '
                'gen; decode-text polarity fix '
                'for Yes 9450 / No 2753); full '
                'run %RT%s.\n')
try:
    led2 = json.load(io.open(LEDGER,
                             encoding='utf-8'))
    _sha8 = led2['ledger_sha256_8']
    _lgn = len(led2['measurements'])
    _l14n = len([l for l in led2['linkage']
                 if l.get('link_id')
                 == 'L14_readout_spectrum_'
                    'cross_model'][0]
                ['connects'])
except Exception:
    _sha8 = 'unknown'
    _lgn = 0
    _l14n = 0
line_clo = (line_clo_tpl
            .replace('%LGN%', str(_lgn))
            .replace('%L14N%', str(_l14n))
            .replace('%SHA8%', _sha8)
            .replace('%MEMOC%', str(_memo_delta))
            .replace('%RT%',
                     ('%.1f' % RUNTIME)))
for wdir in (WLOG_D, WLOG_C):
    for tag, line in (('exp', line_exp),
                      ('clo', line_clo)):
        wl = wdir + '\\' + WDATE + '.md'
        try:
            prev = io.open(wl,
                           encoding='utf-8').read()
        except IOError:
            prev = ''
        marker = ('Phase 3126 Omega-P124' if tag
                  == 'exp'
                  else 'Phase 3126 closeout')
        if marker not in prev:
            try:
                with io.open(wl, 'a',
                             encoding='utf-8') as f:
                    f.write(line)
                o.append('wlog %s appended %s'
                         % (tag, wl))
            except Exception as e:
                o.append('wlog %s fail %s: %r'
                         % (tag, wl, e))
        else:
            o.append('wlog %s already %s'
                     % (tag, wl))

# ---------- 5. MEMORY.md ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3125' in mem_old:
    NEW_3125_COMPACT = (
        u'- 3125（T4）：第三成分=内容尾迹（trail '
        u'0.086/0.050；AR≈0）；sim3 r −0.12→+0.38；'
        u'Qwen 输入流 E_syn L23/22、E_cont L28/32、'
        u'终层全负=模型特异；Qwen3 hs 末项已 norm。')
    NEW_3126 = (u'- 3126（T4）：GLM4 L\\*=20/20 四方向复现'
                u'+终层符号 P/A1 分裂=模型特异；vs3124 '
                u'corr≈1.0=3125 行碰撞警告证伪；trail z '
                u'28.4/10.9 显著、P 长程 +0.059/A1 无；'
                u'反事实 shift 0.75 行为有效；写入链 '
                u'diffuse c3 0.21（共享峰 L8+L29）。')
    NEW_3118_COMPACT = (
        u'- 3118–3120：AUC 振荡 0.981→0.672、状态'
        u'补偿 0.52；答案 token 仅 t=1；重述步上推、'
        u'标点恢复。')
    NEW_NEXT = (u'- max=3126，下一 3127：**写入链端口'
                u'跨模型对照（L8/L29 vs L26–L35）+ A1 '
                u'长程尾迹闭合 + 跨轨迹条件二阶 + 全量'
                u'反事实生成**。')
    assert '@@' not in NEW_3126, \
        'MEM_NEW_3126 not filled'
    assert '@@' not in NEW_NEXT, \
        'MEM_NEW_NEXT not filled'
    lines = mem_old.splitlines()
    out = []
    for ln in lines:
        if ln.startswith(u'## 机制链状态'):
            out.append(u'## 机制链状态（3126）')
        elif ln.startswith(u'- 3125'):
            out.append(NEW_3125_COMPACT)
            out.append(NEW_3126)
        elif ln.startswith(u'- 3118'):
            out.append(NEW_3118_COMPACT)
        elif ln.startswith(u'- max=3125'):
            out.append(NEW_NEXT)
        else:
            out.append(ln)
    mem_new = u'\n'.join(out) + u'\n'
    assert mem_new.count(
        u'## 机制链状态（3126）') == 1
    assert mem_new.count(u'- 3126（T4）') == 1
    assert mem_new.count(u'- 3125（T4）') == 1
    assert mem_new.count(u'- 3124（T4）') == 1
    assert mem_new.count(u'max=3126') == 1
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars'
             % len(mem_new))
else:
    o.append('memory already updated')

with io.open(LOGF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('CLOSEOUT_OK (%d steps)' % len(o))
