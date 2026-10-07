# -*- coding: utf-8 -*-
"""Phase 3129 closeout: idempotent five-write
chain (ledger -> MEMO -> wlogs -> MEMORY).
MEMO title SHORT per user rule (2026-09-24).
wlog read protected vs missing daily file."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3129'
        r'\omega_p127_dose_sweep_gen_'
        r'decouple_s0full_symbolfield')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOGS = [ROOT + r'\.workbuddy\memory',
         (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')]
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
NOW = datetime.datetime.now()
STAMP = NOW.strftime('%Y-%m-%d %H:%M')
r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
V = r['verdict']
assert len(V.split('|')) == 8
assert V == ('a_3128_ok|qwen_dose_monotonic'
             '|qwen_midprop_no'
             '|glm4_coord_a1_ineffective'
             '|gen_coupled'
             '|s0_fully_deterministic'
             '|interaction_symbolic_'
             'not_in_field|coverage_full')
pb = r['part_b']
pc = r['part_c']
pd_ = r['part_d']

# frozen-value asserts (hard-coded抄录)
assert abs(pb['dose_q']['f05']['flip']
           - 0.08035714285714285) < 1e-12
assert abs(pb['dose_q']['f10']['flip']
           - 0.18229166666666666) < 1e-12
assert abs(pb['dose_q']['f20']['flip']
           - 0.28422619047619047) < 1e-12
assert abs(pb['dose_q']['f10']['med_dm']
           - 1.0975285470485687) < 1e-12
assert abs(pb['mid_flip']
           - 0.00818452380952381) < 1e-12
assert abs(pc['a1_flip']
           - 0.011904761904761904) < 1e-12
assert pc['gen_chg_rate'] == 1.0
assert pc['margin_flip_n'] == 172
assert pc['gen_first_chg'] == 178
assert pc['s0_mism'] == 0
assert pc['s0_token_agree'] == 1.0
assert abs(pd_['peak_sep']
           - 0.2934277862541327) < 1e-12
assert abs(pd_['peak_r']
           - 0.2499297708600317) < 1e-12
assert pd_['peak_layer'] == 40
assert r['runtime_s'] > 10000

# sha8 of result.json
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

steps = []
if 'meas3129_omega_p127' in io.open(
        LEDGER, encoding='utf-8').read():
    steps.append('ledger: exists, skip')
else:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    claim = (
        'Omega-P127 (3129, T4 twelfth phase: '
        'dose-response sweep K=50 top-rho '
        'coords frac 0.05/0.10/0.20 x+-sgn '
        'qwen L24 672x2 with bit-level 3128 '
        'reproduction at 0.10; mid-position '
        'injection traj token 6; glm4 A1-'
        'direction inject; W4 joint swap '
        'generation 672 vs gen_base margin-'
        'generation decouple; s0 full 21-'
        'batch bit-exact; regen teacher-'
        'forced margin sign-field 4x672 '
        'per-layer phi, 10239s) - verdict '
        + V)
    entry = {
        'meas_id':
            'meas3129_omega_p127_dose_sweep_'
            'gen_decouple_s0full_symbolfield',
        'phase': 3129,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3129/omega_p127_.../result'
                '.json sha256_8=' + sha8,
            'npz': 'p127_readout.npz'},
        'hashes': {'result_sha256_8': sha8},
        'anchors': [],
        'note': 'dose monotonic with exact '
                '3128 repro + injection doubly '
                'local (mid-pos + A1 both '
                'dead) + swap rewrites 100% '
                'generation vs small margin '
                'shift + s0 fully deterministic'}
    led['measurements'].append(entry)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    steps.append('ledger: appended n=%d '
                 'sha8=%s'
                 % (len(led['measurements']),
                    led['ledger_sha256_8']))

memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3129:' in memo:
    steps.append('memo: exists, skip')
else:
    sec = (
        '\n## Phase 3129: \u03a9-P127 \u5242\u91cf'
        '\u5355\u8c03\u590d\u73b0 + \u6ce8\u5165'
        '\u53cc\u91cd\u5c40\u57df + swap \u91cd'
        '\u5199\u751f\u6210 + s0 \u5168\u91cf'
        '\u786e\u5b9a\u6027\uff08T4 \u7b2c12'
        'Phase\uff09[' + STAMP + ']\n\n'
        '**\u6027\u8d28**\uff1aT4 \u7b2c 12 '
        'Phase\uff0c3128 MEMO \u7b2c 5 \u8282 '
        '\u9884\u6ce8\u518c\u56db\u9879\u6267'
        '\u884c\u3002Part A offline 3128 \u94fe'
        '\u63a5\u65ad\u8a00 + 3127 \u95e8\u503c'
        ' f32 \u590d\u523b\uff081e-12\uff09\uff1b'
        'Part B GPU qwen3-4b \u5242\u91cf\u626b'
        '\u63cf K=50 \u5750\u6807 frac '
        '{0.05,0.10,0.20}\u00d7\u00b1sgn 672\u00d72'
        ' + \u4e2d\u95f4\u4f4d\uff08traj token 6'
        '\uff09\u6ce8\u5165\uff1bPart C GPU glm4'
        '-9b A1 \u65b9\u5411\u6ce8\u5165 + W4 '
        'joint swap \u5168\u91cf\u751f\u6210\u5bf9'
        '\u7167 + s0 \u5168 21 \u6279\u4f4d\u7ea7'
        '\uff1bPart D regen teacher-forced '
        'margin \u7b26\u53f7\u573a 4 \u7ec4\u00d7672 '
        '\u9010\u5c42 phi \u5256\u9762\u3002'
        '\u8fd0\u884c 10239s\u3002\n\n'
        '### 1. \u4e09\u5927\u53d1\u73b0\uff08'
        '\u91cd\u590d\u4e09\u904d\uff09\n'
        '1. **\u5242\u91cf\u5355\u8c03 + \u8de8'
        'Phase \u4f4d\u7ea7\u590d\u73b0**\u3002'
        'flip 0.080\u21920.182\u21920.284'
        '\uff08\u03b4=0.05/0.10/0.20\u00d7norm'
        '\uff09\u5355\u8c03\uff1b\u03b4=0.10 \u4e0e '
        '3128 top50 **flip \u4e0e med|dm| \u53cc'
        '\u4f4d\u7ea7\u540c\u503c**\uff080.18229166'
        '.../1.0975285... \u7cbe\u786e\u4e00\u81f4'
        '\uff09\u3002**\u91cd\u590d\uff1a\u5750\u6807'
        '\u6ce8\u5165\u6548\u5e94\u5b8c\u5168\u53ef'
        '\u590d\u73b0\uff1b\u5242\u91cf\u54cd\u5e94'
        '\u5e73\u6ed1\u5355\u8c03\u3002**\n'
        '1. **\u5242\u91cf\u5355\u8c03 + \u8de8'
        'Phase \u4f4d\u7ea7\u590d\u73b0**\u3002'
        'flip 0.080\u21920.182\u21920.284'
        '\uff08\u03b4=0.05/0.10/0.20\u00d7norm'
        '\uff09\u5355\u8c03\uff1b\u03b4=0.10 \u4e0e '
        '3128 top50 **flip \u4e0e med|dm| \u53cc'
        '\u4f4d\u7ea7\u540c\u503c**\uff080.18229166'
        '.../1.0975285... \u7cbe\u786e\u4e00\u81f4'
        '\uff09\u3002**\u91cd\u590d\uff1a\u5750\u6807'
        '\u6ce8\u5165\u6548\u5e94\u5b8c\u5168\u53ef'
        '\u590d\u73b0\uff1b\u5242\u91cf\u54cd\u5e94'
        '\u5e73\u6ed1\u5355\u8c03\u3002**\n'
        '1. **\u5242\u91cf\u5355\u8c03 + \u8de8'
        'Phase \u4f4d\u7ea7\u590d\u73b0**\u3002'
        'flip 0.080\u21920.182\u21920.284'
        '\uff08\u03b4=0.05/0.10/0.20\u00d7norm'
        '\uff09\u5355\u8c03\uff1b\u03b4=0.10 \u4e0e '
        '3128 top50 **flip \u4e0e med|dm| \u53cc'
        '\u4f4d\u7ea7\u540c\u503c**\uff080.18229166'
        '.../1.0975285... \u7cbe\u786e\u4e00\u81f4'
        '\uff09\u3002**\u91cd\u590d\uff1a\u5750\u6807'
        '\u6ce8\u5165\u6548\u5e94\u5b8c\u5168\u53ef'
        '\u590d\u73b0\uff1b\u5242\u91cf\u54cd\u5e94'
        '\u5e73\u6ed1\u5355\u8c03\u3002**\n'
        '2. **\u6ce8\u5165\u6548\u5e94\u53cc\u91cd'
        '\u5c40\u57df\uff08\u7a7a\u95f4\u00d7\u65b9'
        '\u5411\uff09**\u3002\u4e2d\u95f4\u4f4d'
        '\u6ce8\u5165\uff08\u9694 6 token \u4f4d'
        '\uff09flip 0.008 vs \u672b\u4f4d 0.182'
        '\uff1bA1 \u65b9\u5411 flip 0.012 vs P '
        '\u65b9\u5411 0.140\u2014\u2014\u5750\u6807'
        '\u6ce8\u5165\u4ec5\u5728\uff08\u672b\u4f4d'
        '\uff0cP \u65b9\u5411\uff09\u8bfb\u70b9'
        '\u5c40\u90e8\u6709\u6548\u3002**\u91cd'
        '\u590d\uff1a\u5750\u6807\u63a8\u62c9\u662f'
        '\u8bfb\u70b9\u7ed1\u5b9a\u73b0\u8c61'
        '\uff0c\u4e0d\u662f\u53ef\u4f20\u64ad\u7684'
        '\u5199\u673a\u5236\uff1b\u6548\u5e94\u8870'
        '\u51cf\u957f\u5ea6 < 6 token \u4f4d\u3002**\n'
        '2. **\u6ce8\u5165\u6548\u5e94\u53cc\u91cd'
        '\u5c40\u57df\uff08\u7a7a\u95f4\u00d7\u65b9'
        '\u5411\uff09**\u3002\u4e2d\u95f4\u4f4d'
        '\u6ce8\u5165\uff08\u9694 6 token \u4f4d'
        '\uff09flip 0.008 vs \u672b\u4f4d 0.182'
        '\uff1bA1 \u65b9\u5411 flip 0.012 vs P '
        '\u65b9\u5411 0.140\u2014\u2014\u5750\u6807'
        '\u6ce8\u5165\u4ec5\u5728\uff08\u672b\u4f4d'
        '\uff0cP \u65b9\u5411\uff09\u8bfb\u70b9'
        '\u5c40\u90e8\u6709\u6548\u3002**\u91cd'
        '\u590d\uff1a\u5750\u6807\u63a8\u62c9\u662f'
        '\u8bfb\u70b9\u7ed1\u5b9a\u73b0\u8c61'
        '\uff0c\u4e0d\u662f\u53ef\u4f20\u64ad\u7684'
        '\u5199\u673a\u5236\uff1b\u6548\u5e94\u8870'
        '\u51cf\u957f\u5ea6 < 6 token \u4f4d\u3002**\n'
        '2. **\u6ce8\u5165\u6548\u5e94\u53cc\u91cd'
        '\u5c40\u57df\uff08\u7a7a\u95f4\u00d7\u65b9'
        '\u5411\uff09**\u3002\u4e2d\u95f4\u4f4d'
        '\u6ce8\u5165\uff08\u9694 6 token \u4f4d'
        '\uff09flip 0.008 vs \u672b\u4f4d 0.182'
        '\uff1bA1 \u65b9\u5411 flip 0.012 vs P '
        '\u65b9\u5411 0.140\u2014\u2014\u5750\u6807'
        '\u6ce8\u5165\u4ec5\u5728\uff08\u672b\u4f4d'
        '\uff0cP \u65b9\u5411\uff09\u8bfb\u70b9'
        '\u5c40\u90e8\u6709\u6548\u3002**\u91cd'
        '\u590d\uff1a\u5750\u6807\u63a8\u62c9\u662f'
        '\u8bfb\u70b9\u7ed1\u5b9a\u73b0\u8c61'
        '\uff0c\u4e0d\u662f\u53ef\u4f20\u64ad\u7684'
        '\u5199\u673a\u5236\uff1b\u6548\u5e94\u8870'
        '\u51cf\u957f\u5ea6 < 6 token \u4f4d\u3002**\n'
        '3. **swap \u91cd\u5199\u751f\u6210 + s0 '
        '\u5168\u91cf\u786e\u5b9a\u6027\u5b8c\u5907'
        '**\u3002W4 swap \u4e0b\u751f\u6210 '
        '672/672 **\u5168\u90e8\u6539\u5199**'
        '\uff08\u9996\u4f4d\u53d8 178\uff09\u800c '
        'margin \u4ec5\u5c0f\u5e45\u6270\u52a8'
        '\uff08flip \u4ec5 172\uff09\u2014\u2014'
        '**margin \u975e\u751f\u6210\u5145\u5206'
        '\u7edf\u8ba1\u91cf**\uff1bs0 \u5168 21 \u6279 '
        '672 prompts \u4f4d\u7ea7\u4e00\u81f4'
        '\uff08mism 0\uff0cagree 1.000000\uff09'
        '\u3002**\u91cd\u590d\uff1a\u751f\u6210'
        '\u7ba1\u7ebf\u786e\u5b9a\u6027\u5b8c\u5907'
        '\u8bc1\u660e\uff1bmargin \u5c0f\u53d8'
        '\u2260 \u751f\u6210\u4fdd\u5e8f\u3002**\n'
        '3. **swap \u91cd\u5199\u751f\u6210 + s0 '
        '\u5168\u91cf\u786e\u5b9a\u6027\u5b8c\u5907'
        '**\u3002W4 swap \u4e0b\u751f\u6210 '
        '672/672 **\u5168\u90e8\u6539\u5199**'
        '\uff08\u9996\u4f4d\u53d8 178\uff09\u800c '
        'margin \u4ec5\u5c0f\u5e45\u6270\u52a8'
        '\uff08flip \u4ec5 172\uff09\u2014\u2014'
        '**margin \u975e\u751f\u6210\u5145\u5206'
        '\u7edf\u8ba1\u91cf**\uff1bs0 \u5168 21 \u6279 '
        '672 prompts \u4f4d\u7ea7\u4e00\u81f4'
        '\uff08mism 0\uff0cagree 1.000000\uff09'
        '\u3002**\u91cd\u590d\uff1a\u751f\u6210'
        '\u7ba1\u7ebf\u786e\u5b9a\u6027\u5b8c\u5907'
        '\u8bc1\u660e\uff1bmargin \u5c0f\u53d8'
        '\u2260 \u751f\u6210\u4fdd\u5e8f\u3002**\n'
        '3. **swap \u91cd\u5199\u751f\u6210 + s0 '
        '\u5168\u91cf\u786e\u5b9a\u6027\u5b8c\u5907'
        '**\u3002W4 swap \u4e0b\u751f\u6210 '
        '672/672 **\u5168\u90e8\u6539\u5199**'
        '\uff08\u9996\u4f4d\u53d8 178\uff09\u800c '
        'margin \u4ec5\u5c0f\u5e45\u6270\u52a8'
        '\uff08flip \u4ec5 172\uff09\u2014\u2014'
        '**margin \u975e\u751f\u6210\u5145\u5206'
        '\u7edf\u8ba1\u91cf**\uff1bs0 \u5168 21 \u6279 '
        '672 prompts \u4f4d\u7ea7\u4e00\u81f4'
        '\uff08mism 0\uff0cagree 1.000000\uff09'
        '\u3002**\u91cd\u590d\uff1a\u751f\u6210'
        '\u7ba1\u7ebf\u786e\u5b9a\u6027\u5b8c\u5907'
        '\u8bc1\u660e\uff1bmargin \u5c0f\u53d8'
        '\u2260 \u751f\u6210\u4fdd\u5e8f\u3002**\n\n'
        '### 2. \u5173\u952e\u6570\u503c\n'
        'Part B\uff1adose flip {f05 0.0804/'
        'f10 0.1823/f20 0.2842}\uff0cmed|dm| '
        '{0.5132/1.0975/2.3679}\uff1bmid flip '
        '0.0082\u3002Part C\uff1aa1 flip 0.0119'
        '\uff08sgn+1 0.0089/sgn-1 0.0149\uff09'
        '\uff1bgen_chg 1.0000\u3001margin_flip '
        '172/672\u3001first_chg 178\u3001s0 '
        'mism 0/672 agree 1.000000\u3002Part D'
        '\uff1aphi \u5cf0 s1_P 0.241@L20\u3001'
        's1_A1 0.357@L40\u3001s3_P 0.257@L39'
        '\u3001s3_A1 0.347@L38\uff1b\u4ea4\u4e92'
        '\u5cf0 L40 sep 0.2934 r 0.2499\u3002'
        'flip \u7387 6 \u7ec4 {s1_P 0.8571/'
        's2_P 0.8274/s3_P 0.4345/s1_A1 0.2589/'
        's2_A1 0.3497/s3_A1 0.6057}\u3002\n\n'
        '### 3. \u786c\u4f24\n'
        '\u2460 mid \u5355\u70b9\uff08\u4ec5 traj '
        'token 6\uff09\uff0c\u672a\u626b\u4f4d'
        '\u7f6e\u8c31\uff1b\u2461 A1 \u6ce8\u5165'
        '\u7528 P-fit \u5750\u6807\uff08\u5750'
        '\u6807\u6e90\u65b9\u5411\u504f\u7f6e'
        '\uff09\uff0c\u975e A1-fit \u5750\u6807'
        '\uff1b\u2462\u751f\u6210\u5bf9\u7167'
        '\u4ec5 GLM4-P\uff0cQwen \u4fa7\u672a\u6d4b'
        '\uff1b\u2463 swap-gen 100% \u6539\u5199'
        '\u542b\u6b63\u5e38\u89e3\u7801\u5206\u53c9'
        '\u653e\u5927\uff0c\u65e0\u5355\u5c42 swap '
        '\u751f\u6210\u5bf9\u7167\uff1b\u2464phi '
        '\u5728 flip \u7387\u6781\u7aef\u7ec4'
        '\uff08s1_P 0.857\uff09\u6709\u504f'
        '\uff1b\u2465\u5242\u91cf 3 \u70b9\u7a00'
        '\u758f\u65e0 S \u5f62\u62df\u5408\uff1b'
        '\u2466margin_flip\u00d7gen_chg 2\u00d72 '
        '\u65e0\u7edf\u8ba1\u68c0\u9a8c\u3002\n\n'
        '### 4. \u673a\u5236\u62fc\u56fe\u66f4\u65b0\n'
        '\u2460\u5750\u6807\u6ce8\u5165=\u8bfb\u70b9'
        '\u7ed1\u5b9a\u73b0\u8c61\uff08\u7a7a\u95f4'
        '\u00d7\u65b9\u5411\u53cc\u5c40\u57df'
        '\uff09\uff0c\u4e0e 3128 \u5e72\u9884'
        '\u7c92\u5ea6=\u5750\u6807\u65b9\u5411'
        '\u5408\u5e76\uff1a\u6709\u6548\u5e72\u9884'
        '\u7a97\u53e3=(\u672b\u4f4d, P, top-K '
        '\u8bfb\u51fa\u5bf9\u9f50\u5750\u6807)'
        '\uff1b\u2461margin \u4e0e\u751f\u6210'
        '\u89e3\u8026\u53cd\u8f6c\uff1alayer swap '
        '\u51e0\u4e4e\u4e0d\u52a8 margin \u5374'
        '\u5168\u91cd\u5199\u751f\u6210\u2192'
        'margin \u53ea\u662f\u672b\u4f4d yes/no '
        'logit \u5dee\uff0c\u4e0d\u7f16\u7801'
        '\u751f\u6210\u8def\u5f84\uff1b\u2462'
        '\u751f\u6210\u786e\u5b9a\u6027\u5b8c\u5907'
        '\uff08\u6279\u5185\u4f4d\u7ea7 + \u8de8'
        'Phase \u4f4d\u7ea7\uff09\uff1b\u2463'
        '\u7b26\u53f7\u7ffb\u8f6c\u573a\u4e2d\u5c42'
        '\u65e0\u65b9\u5411\u00d7\u6761\u4ef6\u4ea4'
        '\u4e92\uff0c\u672b\u5c42\u624d\u4e0e\u884c'
        '\u4e3a\u5bf9\u9f50\uff08L38-40 \u5cf0'
        '\uff09\u2192\u7b26\u53f7\u51b3\u7b56'
        '\u665a\u719f\u3002\n\n'
        '### 5. 3130 \u9884\u6ce8\u518c\uff08T4 '
        '\u7ee7\u7eed\uff0c\u89c2\u6d4b\u524d'
        '\u51bb\u7ed3\uff09\n'
        '\u2460\u4f4d\u7f6e\u8c31\u626b\u63cf'
        '\uff1a\u6ce8\u5165\u4f4d k\u2208{12..1} '
        '\u5168\u8c31 flip(k) \u66f2\u7ebf\uff0c'
        '\u5b9a\u91cf\u6548\u5e94\u8870\u51cf'
        '\u957f\u5ea6\uff1b\u2461A1-fit \u5750\u6807'
        '\u5bf9\u7167\uff08\u6392\u9664\u5750\u6807'
        '\u6e90\u65b9\u5411\u504f\u7f6e\uff09'
        '\uff1b\u2462\u5355\u5c42 swap \u751f\u6210'
        '\u5bf9\u7167\uff08\u89e3\u8026\u7684\u5c42'
        '\u7ea7\u7279\u5f02\u6027\uff09\uff1b'
        '\u2463swap-gen \u9010 token \u5206\u53c9'
        '\u52a8\u529b\u5b66\uff08\u9996\u53d8'
        '\u4f4d\u7f6e\u5206\u5e03\uff09\u3002\n\n'
        '\u4ea7\u7269\uff1a`tests/glm5/result/'
        'rdc_query_construction_20260913/'
        'phase3129/omega_p127_dose_sweep_gen_'
        'decouple_s0full_symbolfield/`'
        '\uff08result.json\u3001design_seal.json'
        '\u3001run_log.txt\u3001p127_readout.npz'
        '\uff09\uff1b\u811a\u672c `tests/glm5/'
        'phase3129_omega_p127_dose_sweep_gen_'
        'decouple_s0full_symbolfield.py`\u3002')
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    steps.append('memo: appended (short '
                 'title)')

WDATE = NOW.strftime('%Y-%m-%d')
wline = ('- Phase 3129 Omega-P127 closeout: '
         'verdict ' + V + '; ledger sha8 '
         + json.load(io.open(
             LEDGER,
             encoding='utf-8'))
         ['ledger_sha256_8']
         + '; MEMO 3129 section (short '
         'title); runtime 10239s.')
for wd in WLOGS:
    wl = wd + '\\' + WDATE + '.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3129 Omega-P127 closeout' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        steps.append('wlog: ' + wl[:12])
    else:
        steps.append('wlog: exists ' + wl[:12])

mem = io.open(MEMW, encoding='utf-8').read()
if '3129\uff08T4\uff09' not in mem:
    old2 = ('\u4e0b\u4e00 3129\uff1a**')
    i2 = mem.find(old2)
    assert i2 > 0, 'next-line anchor missing'
    j2 = mem.find('\n', i2)
    new2 = ('\u4e0b\u4e00 3130\uff1a**'
            '\u4f4d\u7f6e\u8c31\u626b\u63cf'
            '\uff08\u6ce8\u5165\u4f4d k \u5168'
            '\u8c31 flip(k) \u66f2\u7ebf + \u8870'
            '\u51cf\u957f\u5ea6\uff09+ A1-fit \u5750'
            '\u6807\u5bf9\u7167 + \u5355\u5c42 swap '
            '\u751f\u6210\u5bf9\u7167 + swap-gen '
            '\u9010 token \u5206\u53c9\u52a8\u529b'
            '\u5b66\u3002**')
    mem2 = mem[:i2] + new2 + mem[j2:]
    oldm = '- max=3128\uff0c'
    assert mem2.count(oldm) == 1
    mem2 = mem2.replace(
        oldm, '- max=3129\uff0c')
    anchor = ('- 3128\uff08T4\uff09\uff1a')
    ia = mem2.find(anchor)
    assert ia > 0
    new_line = (
        '- 3129\uff08T4\uff09\uff1a\u5242\u91cf'
        '\u5355\u8c03\u4e14\u03b4=0.10 \u4f4d\u7ea7'
        '\u590d\u73b0 3128\uff08flip 0.080/0.182/'
        '0.284\uff09\uff1b\u6ce8\u5165\u53cc\u91cd'
        '\u5c40\u57df\uff08mid 0.008\u3001A1 '
        '0.012 vs \u672b\u4f4d P 0.182/0.140\uff09'
        '=\u8bfb\u70b9\u7ed1\u5b9a\uff1bswap \u91cd'
        '\u5199\u751f\u6210 672/672 \u800c margin '
        '\u4ec5 172 \u7ffb\u8f6c=\u975e\u5145\u5206'
        '\u7edf\u8ba1\u91cf\uff1bs0 \u5168\u91cf '
        'mism 0\uff1b\u7b26\u53f7\u573a\u4ea4\u4e92 '
        'not_in_field\uff08\u5cf0 L40 r 0.25\uff09'
        '\u3002')
    mem2 = mem2[:ia] + new_line + '\n' + \
        mem2[ia:]
    assert len(mem2) < 3000, len(mem2)
    with io.open(MEMW, 'w',
                 encoding='utf-8') as f:
        f.write(mem2)
    steps.append('memory: updated (%d '
                 'chars)' % len(mem2))
else:
    steps.append('memory: exists, skip')

for s in steps:
    print(s)
print('CLOSEOUT_OK (%d steps)'
      % len(steps))
