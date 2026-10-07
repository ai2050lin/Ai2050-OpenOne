# -*- coding: utf-8 -*-
"""Phase 3130 closeout: idempotent five-write
chain (ledger -> MEMO -> wlogs -> MEMORY).
MEMO title SHORT per user rule (2026-09-24).
Includes qwen_spec_nonlocal verdict-word
errata (gate artifact: k=12 site-identity).
All numeric claims read from result.json;
hard frozen asserts before any write."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3130'
        r'\omega_p128_positionspectrum_'
        r'a1fit_single_swap_dyn')
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

F10_REF = 0.18229166666666666
MEDDM_REF = 1.0975285470485687
MID6_REF = 0.00818452380952381
A1FIT_REPLAY_REF = 0.011904761904761904

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
V = r['verdict']
vd = V.split('|')
assert len(vd) == 8
assert vd[0] == 'a_3129_ok'
assert vd[1] == 'qwen_spec_nonlocal'
assert vd[2] == 'glm4_pfit_replay_exact'
assert vd[3] == 'glm4_a1fit_confirms_binding'
assert vd[4] == 'gen_swap_3129_bitexact'
assert vd[7] == 'coverage_full'
pb = r['part_b']
pc = r['part_c']

# ---- frozen-value asserts ----
sf = pb['spec_flip']
sm = pb['spec_med_dm']
assert len(sf) == 13 and len(sm) == 13
assert abs(sf[0] - F10_REF) < 1e-12
assert abs(sf[6] - MID6_REF) < 1e-12
assert abs(sf[12] - sf[0]) < 1e-12
assert abs(sm[0] - MEDDM_REF) < 1e-12
assert pb['k50'] == 12 and pb['k10'] == 12
assert pb['d50'] == 0 and pb['d10'] == 0
assert 10.694 < pb['delta_q'] < 10.695
cliff = max(sf[k] for k in range(1, 12))
assert cliff < 0.035
assert abs(pc['pfit_replay_flip']
           - A1FIT_REPLAY_REF) < 1e-12
assert 0.014 < pc['a1fit_flip'] < 0.016
assert pc['a1fit_ok'] is False
assert pc['coords_a1_overlap_P'] == 0
assert 0.99 < pc['delta_g_a1'] < 1.02
assert abs((pc['a1fit_flip_s1']
            + pc['a1fit_flip_s-1']) / 2.0
           - pc['a1fit_flip']) < 1e-12
assert pc['full_swap_chg'] == 1.0
assert pc['full_swap_first'] == 178
assert pc['full_swap_bitexact_3129'] is True
fh = pc['first_hist']
assert sum(fh) == 672 and fh[0] == 0
assert sum(fh[1:]) == 672
assert fh[1] == 178
assert pc['dominance'] >= 0.0
fs = pc['first_share']
dom = pc['dominance']
single_v_expect = ('single_dominant'
                   if dom >= 0.5
                   else 'single_distributed')
assert vd[5] == single_v_expect
dyn_v_expect = ('dyn_first_concentrated'
                if fs >= 0.5 else 'dyn_spread')
assert vd[6] == dyn_v_expect
assert r['runtime_s'] > 7000

sc = pc['single_chg']
# frozen single-layer values from run 1
# log (deterministic pipeline, 256-grain)
assert abs(sc['8'] - 208 / 256.0) < 1e-9
assert abs(sc['9'] - 205 / 256.0) < 1e-9
assert abs(sc['13'] - 137 / 256.0) < 1e-9
assert abs(sc['29'] - 45 / 256.0) < 1e-9
assert pc['single_first'] == {'8': 3,
                              '9': 2,
                              '13': 0,
                              '29': 0}
assert abs(dom - 208 / 256.0) < 1e-9
assert abs(fs - 178 / 672.0) < 1e-12
sc_s = 'L8 %.3f/L9 %.3f/L13 %.3f/L29 %.3f' \
    % (sc['8'], sc['9'], sc['13'], sc['29'])
fh_s = ' '.join(str(v) for v in fh)

# sha8 of result.json
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

steps = []
if 'meas3130_omega_p128' in io.open(
        LEDGER, encoding='utf-8').read():
    steps.append('ledger: exists, skip')
else:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    claim = (
        'Omega-P128 (3130, T4 thirteenth '
        'phase: position spectrum k=0..12 '
        'x+-sgn 672 qwen L24 K=50 coords '
        'delta=10.69 with k0/k6 1e-9 '
        'anchors and k12==k0 site-identity; '
        'glm4 A1-fit top50 coords 0/50 '
        'overlap vs P-fit, P-fit replay '
        'bit-exact 0.0119, A1-fit inject '
        '0.0149 < A1_MIN=0.05; W4 joint '
        'swap generation 672 cross-phase '
        'bit-exact vs 3129 first=178; '
        'single-layer swap {8,9,13,29}x256 '
        'dominance %.2f; first-change '
        'dynamics share %.2f) - verdict '
        % (dom, fs) + V)
    entry = {
        'meas_id':
            'meas3130_omega_p128_position'
            'spectrum_a1fit_single_swap_dyn',
        'phase': 3130,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3130/omega_p128_.../'
                'result.json sha256_8=' + sha8,
            'npz': 'p128_readout.npz'},
        'hashes': {'result_sha256_8': sha8},
        'anchors': [],
        'note': 'position spectrum '
                'cliff-local: k=1..11 all '
                '<=%.3f vs f0 0.182, d50=d10=0 '
                '; verdict word qwen_spec_'
                'nonlocal is a gate artifact '
                '(k=12 site-identity pollutes '
                'far_dead), corrected in MEMO '
                'errata; A1-fit coords '
                'disjoint from P-fit (0/50) '
                'and A1-half injection weak '
                'in both fits = readout '
                'binding; full swap gen '
                'cross-phase bit-exact'
                % cliff}
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
if '## Phase 3130:' in memo:
    steps.append('memo: exists, skip')
else:
    sec = (
        '\n## Phase 3130: \u03a9-P128 \u4f4d'
        '\u7f6e\u8c31\u60ac\u5d16 d50=0 + '
        'A1-fit 0/50 \u91cd\u53e0 + swap '
        '\u4f4d\u7ea7\u590d\u73b0 + \u5355'
        '\u5c42/\u52a8\u529b\u5b66\uff08T4 '
        '\u7b2c13Phase\uff09[' + STAMP + ']\n\n'
        '**\u6027\u8d28**\uff1aT4 \u7b2c 13 '
        'Phase\uff0c3129 MEMO \u7b2c 5 \u8282 '
        '\u9884\u6ce8\u518c\u56db\u9879\u6267'
        '\u884c\uff0cdesign_seal.json \u89c2'
        '\u6d4b\u524d\u51bb\u7ed3\u3002Part A '
        'offline 3129 \u94fe\u63a5\u65ad\u8a00'
        ' + 3127 \u95e8\u503c f32 \u590d\u523b'
        '\uff081e-12\uff09\uff1bPart B GPU '
        'qwen3-4b \u4f4d\u7f6e\u8c31\u626b\u63cf'
        ' k\u2208{0..12}\u00d7\u00b1sgn\u00d7672'
        '\uff08K=50 \u5750\u6807 \u03b4=0.10\u00d7'
        'norm=10.6946\uff0cL24\uff09+ k0/k6 '
        '\u53cc\u951a + k12\u2261k0 site-'
        'identity\uff1bPart C GPU glm4-9b '
        'A1-fit \u5750\u6807\u5bf9\u7167 + '
        'P-fit replay \u951a + \u5168 swap '
        '672 \u8de8 Phase \u4f4d\u7ea7 + \u5355'
        '\u5c42 swap {8,9,13,29}\u00d7256 + '
        '\u9010 token \u5206\u53c9\u52a8\u529b'
        '\u5b66\u3002\u8fd0\u884c RUNTIMEs\u3002\n\n'
        '### 1. \u4e09\u5927\u53d1\u73b0\uff08'
        '\u91cd\u590d\u4e09\u904d\uff09\n'
        'REPEATS\n\n'
        '### 2. \u5173\u952e\u6570\u503c\n'
        'NUMS\n\n'
        '### 3. \u786c\u4f24\n'
        'HARDS\n\n'
        '### 4. \u673a\u5236\u62fc\u56fe\u66f4'
        '\u65b0\n'
        'MECH\n\n'
        '### 5. errata\uff08\u5224\u51b3\u8bcd'
        '\u52d8\u8bef\uff09\n'
        'ERRATA\n\n'
        '### 6. 3131 \u9884\u6ce8\u518c\uff08'
        'T4 \u7ee7\u7eed\uff0c\u89c2\u6d4b\u524d'
        '\u51bb\u7ed3\uff09\n'
        'PREREG\n\n'
        '\u4ea7\u7269\uff1a`tests/glm5/result/'
        'rdc_query_construction_20260913/'
        'phase3130/omega_p128_positionspectrum'
        '_a1fit_single_swap_dyn/`'
        '\uff08result.json\u3001design_seal'
        '.json\u3001run_log.txt\u3001p128_'
        'readout.npz\uff09\uff1b\u811a\u672c '
        '`tests/glm5/phase3130_omega_p128_'
        'positionspectrum_a1fit_single_swap_'
        'dyn.py`\u3002')

    f1 = (
        '1. **\u4f4d\u7f6e\u8c31\u60ac\u5d16'
        '\u5f0f\u5c40\u57df\uff0c\u6ce8\u5165'
        '\u6548\u5e94\u4e25\u683c\u7ed1\u5b9a'
        '\u8bfb\u70b9\u4f4d\uff08d50=d10=0'
        '\uff09**\u3002k=0 flip 0.182292'
        '\uff08=3128 f10 \u4f4d\u7ea7\u590d'
        '\u73b0 1e-9\uff09\uff1bk=1..11 \u5168'
        '\u90e8 %.4f\u2013%.4f\uff08\u22640.17'
        '\u00d7\u672c\u4f4d\uff0c\u4f4e\u4e8e '
        'rand50 \u57fa\u7ebf 0.095 \u7684 1/3'
        '\uff09\uff1bk=12\u2261k0 \u4f4d\u7ea7'
        '\u76f8\u540c\uff08site-identity\uff0c'
        '\u7ba1\u7ebf\u786e\u5b9a\u6027\u8bc1'
        '\u636e\uff09\u3002\u60ac\u5d16\u843d'
        '\u5dee >5.9\u00d7 \u53d1\u751f\u5728 1 '
        'token \u4f4d\u5185\u3002**\u91cd\u590d'
        '\uff1a\u6548\u5e94\u8870\u51cf\u957f'
        '\u5ea6 <1 token \u4f4d\uff1b\u5750'
        '\u6807\u6ce8\u5165\u662f\u8bfb\u70b9'
        '\u4f4d\u7ed1\u5b9a\u73b0\u8c61\uff0c'
        '\u4e0d\u662f\u53ef\u4f20\u64ad\u5199'
        '\u5165\u3002**\n'
        % (min(sf[1:12]), cliff))
    f2 = (
        '2. **A1-fit \u5750\u6807\u4e0e P-fit '
        '0/50 \u91cd\u53e0 + A1 \u534a\u6ce8'
        '\u5165\u53cc\u5411\u5f31**\u3002A1 '
        'odd-half\uff08336 prompts L20 hs\uff09'
        '+ A1 margin \u91cd\u7b97 top-50 \u5750'
        '\u6807\uff0c\u4e0e 3128 P-fit \u5750'
        '\u6807\u4ea4\u96c6 **0/50**\uff08\u65b9'
        '\u5411\u7279\u5f02\u5750\u6807\u96c6'
        '\uff0cSMOKE \u7684 2/50 \u7cfb 4 \u6837'
        '\u672c\u4f2a\u8c61\uff09\uff1bdelta_g'
        '_a1=%.4f \u4e0e P 0.9954 \u540c\u91cf'
        '\u7ea7\u3002\u4f46 A1-fit \u6ce8\u5165 '
        'A1 \u534a flip \u4ec5 %.4f\uff08< '
        'A1_MIN=0.05\uff09\uff0cP-fit \u6ce8'
        '\u5165 A1 \u534a 0.0119\uff08=3129 '
        '\u4f4d\u7ea7\u590d\u73b0\uff09\u2014'
        '\u2014\u53cc\u5411\u90fd\u5f31\u3002'
        '**\u91cd\u590d\uff1aA1 \u65b9\u5411'
        '\u4e0d\u5b58\u5728\u53ef\u6ce8\u5165'
        '\u901a\u9053\uff1b\u7ed1\u5b9a\u7684'
        '\u4e0d\u6b62\u5750\u6807\u96c6\uff0c'
        '\u8fd8\u6709\u65b9\u5411\u00d7\u534a'
        '\u533a\u7ec4\u5408\u3002**\n'
        % (pc['delta_g_a1'], pc['a1fit_flip']))
    f3 = (
        '3. **\u5168 swap \u751f\u6210\u8de8 '
        'Phase \u4f4d\u7ea7\u590d\u73b0 + \u5355'
        '\u5c42/\u52a8\u529b\u5b66\u5224\u51b3'
        '**\u3002W4 joint swap \u751f\u6210 '
        '672 prompts \u4e0e 3129 \u4f4d\u7ea7'
        '\u4e00\u81f4\uff08np.array_equal\uff0c'
        'chg=1.0\u3001first=178\uff09\u2014\u2014'
        '\u751f\u6210\u7ba1\u7ebf\u8de8 Phase '
        '\u786e\u5b9a\u6027\u5b8c\u5907\u3002'
        '\u5355\u5c42 swap {8,9,13,29}\u00d7256'
        '\uff08PERM_SEED=3130 \u51bb\u7ed3\u9009'
        '\u62e9 sel256_sha8=%s\uff09\uff1a%s'
        '\uff0cdominance %.2f \u2192 %s\u3002'
        '\u9010 token \u5206\u53c9\uff1afirst_'
        'share %.3f \u2192 %s\uff08first_hist '
        '%s\uff09\u3002**\u91cd\u590d\uff1a%s'
        '**\n'
        % (pc['sel256_sha8'], sc_s, dom,
           single_v_expect, fs, dyn_v_expect,
           fh_s,
           ('\u5355\u5c42\u4e3b\u5bfc\uff0c\u6539'
            '\u5199\u51b3\u7b56\u53ef\u5c40\u90e8'
            '\u5316' if dom >= 0.5
            else '\u6539\u5199\u9700\u591a\u5c42'
                 '\u8054\u5408\uff0c\u5355\u5c42'
                 '\u65e0\u4e3b\u5bfc')
           + '\uff1b'
           + ('\u5206\u53c9\u96c6\u4e2d\u4e8e'
              '\u9996 token\uff0c\u51b3\u7b56'
              '\u5728\u751f\u6210\u8d77\u70b9'
              '\u4e00\u6b65\u5b8c\u6210'
              if fs >= 0.5
              else '\u5206\u53c9\u5f25\u6563'
                   '\uff0c\u9010\u6b65\u7d2f\u79ef')))

    nums = (
        'Part B\uff1aflip(k) = %s\uff1b'
        'med|dm| k0 %.4f\uff1b\u03b4_q=%.4f'
        '\uff1bk50=k10=12\uff0cd50=d10=0\u3002'
        'Part C\uff1apfit_replay_flip '
        '0.011905\uff08\u4f4d\u7ea7\uff09\uff1b'
        'a1fit_flip %.4f\uff08s1 %.4f / s-1 '
        '%.4f\uff09\u3001overlap 0/50\u3001'
        'delta_g_a1 %.4f\uff1bfull_swap chg '
        '%.1f\u3001first %d\u3001bitexact '
        'True\uff1bsingle %s\uff1bdominance '
        '%.2f\uff1bfirst_share %.3f\u3001'
        'first_hist %s\u3002'
        % ('/'.join(['%.4f' % v
                     for v in sf]),
           sm[0], pb['delta_q'],
           pc['a1fit_flip'], pc['a1fit_flip_s1'],
           pc['a1fit_flip_s-1'],
           pc['delta_g_a1'], pc['full_swap_chg'],
           pc['full_swap_first'], sc_s,
           dom, fs, fh_s))

    hards = (
        '\u2460\u8c31\u626b\u63cf flip \u4e3a '
        '\u00b1sgn \u5747\u503c\uff0c\u672a\u9010 '
        'sgn \u62a5\u5256\u9762\uff1b\u2461'
        'k12\u2261k0 \u6052\u7b49\u4f7f far_dead/'
        'nonincreasing \u95e8\u5931\u6548\uff08'
        '\u5224\u51b3\u8bcd\u52d8\u8bef\u89c1 '
        '\u00a75\uff0c\u811a\u672c immutable \u95e8'
        '\u672a\u4fee\uff09\uff1b\u2462A1-fit \u4ec5 '
        'top-50 \u5355\u4e00 K\uff0c\u672a\u626b '
        'K \u6df1\u5ea6\uff1b\u2463single \u4ec5 4 '
        '\u5c42\uff083127 \u95e8\u5c42\uff09\u975e'
        '\u5168\u5c42\uff1b\u2464\u52a8\u529b\u5b66'
        '\u4ec5\u9996\u5206\u53c9\u4f4d\u7f6e\u76f4'
        '\u65b9\u56fe\uff0c\u672a\u5b9a\u4f4d\u5206'
        '\u53c9\u51b3\u7b56\u5c42\uff1b\u2465gen_'
        'tokdiff \u6743\u91cd\u5206\u89e3\u4f9d'
        '\u8d56 3129 margin_flip \u6807\u7b7e'
        '\uff08\u89c2\u6d4b\u540e\u5212\u5206\uff09'
        '\u3002')

    mech = (
        '\u2460\u7ed1\u5b9a\u7a97\u53e3\u4e09'
        '\u7ef4\u5316\uff1a\u4f4d\u7f6e\u7ef4'
        '\uff083130 \u60ac\u5d16 d50=0\uff09\u00d7'
        '\u65b9\u5411\u7ef4\uff083129 A1 \u6b7b'
        '\u533a + 3130 \u5750\u6807\u96c6 0/50 '
        '\u5206\u79bb\uff09\u2014\u2014\u5750\u6807'
        '\u6ce8\u5165\u6548\u5e94\u9700\uff08\u672b'
        '\u4f4d\uff0cP \u534a\u533a\uff0c\u5bf9'
        '\u9f50\u5750\u6807\uff09\u4e09\u5143\u7ec4'
        '\u540c\u65f6\u6210\u7acb\uff1b\u2461A1 '
        '\u534a\u533a hs \u4e0e margin \u53ef fit'
        '\uff08rho \u5b58\u5728\uff09\u4f46\u4e0d'
        '\u53ef\u6ce8\u5165\u2014\u2014\u63cf\u8ff0'
        '\u6027\u76f8\u5173\u4e0e\u673a\u5236\u901a'
        '\u9053\u518d\u6b21\u5206\u79bb\uff08\u547c'
        '\u5e94 3102 \u539f\u5219 4\uff09\uff1b'
        '\u2462\u751f\u6210\u4e0e margin \u53cc'
        '\u8f68\uff1a\u5168 swap \u4f4d\u7ea7\u590d'
        '\u73b0 + \u5355\u5c42 %s\u2014\u2014margin '
        '\u975e\u5145\u5206\u7edf\u8ba1\u91cf\u518d'
        '\u8bc1\uff1b\u2463\u5206\u53c9\u52a8\u529b'
        '\uff1a%s\u3002'
        % (('\u4e3b\u5bfc L%d chg %.3f'
            % (int(max(sc, key=lambda k:
                       sc[k])),
               max(sc.values())))
           if dom >= 0.5
           else '\u5f25\u6563\uff08max %.3f / 1.0'
                % max(sc.values()),
           ('\u9996 token \u96c6\u4e2d\uff08share '
            '%.2f\uff09=\u51b3\u7b56\u5728\u751f'
            '\u6210\u8d77\u70b9' % fs
            if fs >= 0.5
            else '\u5f25\u6563\uff08share %.2f'
                 '=\u9010\u6b65\u7d2f\u79ef' % fs)))

    errata = (
        '\u5224\u51b3\u8bcd `qwen_spec_nonlocal` '
        '\u4e3a\u95e8\u4f2a\u8c61\uff1afar_dead '
        '\u68c0\u67e5\uff08k\u22653 \u5168\u90e8 '
        '<0.5\u00d7f0\uff09\u672a\u6392\u9664 '
        'k=12\u2261k0 \u7684 site-identity \u6052'
        '\u7b49\u70b9\uff08\u5176 flip=0.182292 '
        '\u6052\u4e0d\u6ee1\u8db3\uff09\uff0c'
        'nonincreasing \u4ea6\u88ab k11\u2192k12 '
        '\u62ac\u5347\u6c61\u67d3\u3002\u6b63\u786e'
        '\u89e3\u8bfb\u4e3a**\u5f3a\u5c40\u57df'
        '\uff08cliff-local\uff09**\uff1ak\u2208{1..11}'
        ' \u5168\u90e8 \u2264%.4f\u3002verdict \u5b57'
        '\u6bb5 immutable\uff0c\u4ee5\u672c\u52d8'
        '\u8bef\u4e3a\u51c6\uff1b3131 \u8d77\u95e8'
        '\u8bbe\u8ba1\u5148\u6392\u9664\u6052\u7b49'
        '\u70b9\u3002' % cliff)

    prereg = (
        '\u2460\u5168\u5c42\u5355\u5c42 swap \u751f'
        '\u6210\u8c31\uff1aL0\u2013L39 \u9010\u5c42'
        ' swap\uff0840\u00d7256\uff0cSMOKE \u7f29'
        '\u51cf\uff09\u5b9a\u4f4d\u6539\u5199\u5c42'
        '\u8d21\u732e\u5256\u9762\u4e0e 3127 \u95e8'
        '\u5c42\uff088/9/13/29\uff09\u5173\u7cfb'
        '\uff1b\u2461\u6ce8\u5165\u5c42\u7ef4\u7a97'
        '\u53e3\uff1a\u672b\u4f4d\u8bfb\u51fa\u4e0b'
        '\u6ce8\u5165\u5c42 L\u2208{20,22,24,26,28}'
        '\u00d7P-fit \u5750\u6807\u00d7\u03b4=10.69'
        '\uff0c\u8865\u5168\u7ed1\u5b9a\u7a97\u53e3'
        '\u7684\u5c42\u7ef4\u5256\u9762\uff08\u4f4d'
        '\u7f6e\u00d7\u65b9\u5411\u5df2\u5c01\uff09'
        '\uff1b\u2462A1 \u534a\u533a\u751f\u6210'
        '\u5bf9\u7167\uff1aA1 half \u5168 swap \u751f'
        '\u6210 672\uff08chg/first \u52a8\u529b\u5b66'
        '\uff09\uff0c\u68c0\u9a8c swap \u91cd\u5199'
        '\u751f\u6210\u7684\u65b9\u5411\u5bf9\u79f0'
        '\u6027\uff1b\u2463\u5206\u53c9\u51b3\u7b56'
        '\u5c42\u5b9a\u4f4d\uff1aswap \u751f\u6210'
        '\u4e0b\u9996\u5206\u53c9 token \u7684 per-'
        'layer margin \u8f68\u8ff9\uff08b vs g\uff0c'
        'L0\u201340\uff09\uff0c\u4e0e 3129 \u7b26'
        '\u53f7\u573a L38\u201340 \u665a\u719f\u5bf9'
        '\u7167\u3002')

    sec = (sec.replace('RUNTIME',
                       '%.0f' % r['runtime_s'])
              .replace('REPEATS',
                       f1 + f1 + f1 + f2 + f2
                       + f2 + f3 + f3 + f3)
              .replace('NUMS', nums)
              .replace('HARDS', hards)
              .replace('MECH', mech)
              .replace('ERRATA', errata)
              .replace('PREREG', prereg))
    assert 'REPEAT' not in sec
    assert len(sec) > 2500
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    steps.append('memo: appended (short '
                 'title + errata)')

WDATE = NOW.strftime('%Y-%m-%d')
wline = ('- Phase 3130 Omega-P128 closeout: '
         'verdict ' + V + '; ledger sha8 '
         + json.load(io.open(
             LEDGER,
             encoding='utf-8'))
         ['ledger_sha256_8']
         + '; MEMO 3130 section (short '
         'title + verdict errata); runtime '
         + ('%.0fs' % r['runtime_s']) + '.')
for wd in WLOGS:
    wl = wd + '\\' + WDATE + '.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3130 Omega-P128 closeout' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        steps.append('wlog: ' + wl[:12])
    else:
        steps.append('wlog: exists ' + wl[:12])

mem = io.open(MEMW, encoding='utf-8').read()
if '3130\uff08T4\uff09' not in mem:
    # pre-compress old lines (budget for
    # the 3130 line under 3000 cap)
    pairs = [
        ('- 3123\uff08T4\uff09\uff1a\u5206\u65b9'
         '\u5411\u7b97\u5b50\uff08S \u22120.60/'
         '\u22120.55\u3001MS \u22124.96/\u2212'
         '7.15 \u5206\u79bb\uff09+\u6c60\u62bd'
         '\u951a\u70b9\u5747\u672a\u6062\u590d '
         'AUC \u5f62\u72b6\uff08r 0.23\uff09\u4f46'
         '\u5e73\u53f0 0.50\u21920.66\u3001PIT '
         '0.033 \u6821\u51c6=**i.i.d. \u6269'
         '\u6563\u4f2a\u5f71\u3001\u6b8b\u5dee'
         '\u5fc5\u4e3a\u5171\u6a21**\uff1bL35 '
         '\u5239\u8f66\u5168\u5c40\u4f46 A1 \u7b54'
         '\u6848\u6b65\u91ca\u538b +3.7\uff1b**'
         '\u8bed\u6cd5\u95e8\u63a7 L21/\u5185'
         '\u5bb9 L20 \u6d8c\u73b0=\u5199\u5165'
         '\u94fe\u4e0a\u6e38**\u3002',
         '- 3123\uff08T4\uff09\uff1a\u5206\u65b9'
         '\u5411\u7b97\u5b50+\u6c60\u62bd\u951a'
         '\u70b9\u672a\u6062\u590d AUC \u5f62'
         '\u72b6\u4f46\u5e73\u53f0 0.50\u21920.66'
         '\u3001PIT \u6821\u51c6=**i.i.d. \u6269'
         '\u6563\u4f2a\u5f71\u3001\u6b8b\u5dee'
         '\u5fc5\u4e3a\u5171\u6a21**\uff1bL35 '
         '\u5239\u8f66\u5168\u5c40\u4f46 A1 \u91ca'
         '\u538b\uff1b**\u8bed\u6cd5 L21/\u5185'
         '\u5bb9 L20 \u6d8c\u73b0=\u5199\u5165'
         '\u94fe\u4e0a\u6e38**\u3002'),
        ('- 3121\u20133122\uff1atoken \u7ea7'
         '\u66ff\u6362\u5931\u6548\uff08\u632f'
         '\u8361\u6df9\u6ca1\u5747\u503c\uff09'
         '\uff1b\u5199\u5165\u8c31 L28\u201334 '
         '\u6b63\u5199+L35 \u5927\u8d1f\u5199 '
         '\u221211.4\uff1b\u53e5\u7ea7\u66ff\u6362 '
         'E_cont \u22123.68/\u22122.23\u3001'
         '\u8bed\u6cd5\u6df7\u6392\u6062\u590d'
         '\u4e00\u534a=**\u8bed\u6cd5\u4f7f'
         '\u5185\u5bb9\u53ef\u8bfb**\u3002',
         '- 3121\u20133122\uff1atoken \u7ea7'
         '\u66ff\u6362\u5931\u6548\uff1b\u5199'
         '\u5165\u8c31 L28\u201334 \u6b63\u5199'
         '+L35 \u8d1f\u5199 \u221211.4\uff1b'
         '\u53e5\u7ea7\u8bed\u6cd5\u6df7\u6392'
         '\u6062\u590d\u4e00\u534a=**\u8bed'
         '\u6cd5\u4f7f\u5185\u5bb9\u53ef\u8bfb'
         '**\u3002'),
        ('- 3127\uff08T4\uff09\uff1a\u5199\u5165'
         '\u94fe\u529f\u80fd\u5426\u5b9a\uff08'
         '\u00d71.04/\u00d70.87 \u5168 <2 \u95e8'
         '\uff09=\u5355\u5c42\u65e0\u5fc5\u8981'
         '\u5199\u5165\uff1btrail \u5b9a\u7a3f '
         'lag1-3\uff1b\u5168\u91cf regen \u524d '
         '96 \u4f4d\u7ea7\u590d\u73b0\u3001flip '
         '\u590d\u73b0\u3002',
         '- 3127\uff08T4\uff09\uff1a\u5199\u5165'
         '\u94fe\u529f\u80fd\u5426\u5b9a\uff08'
         '\u00d71.04/\u00d70.87\uff09=\u5355'
         '\u5c42\u65e0\u5fc5\u8981\u5199\u5165'
         '\uff1btrail lag1-3\uff1b\u5168\u91cf '
         'regen \u524d 96 \u4f4d\u7ea7\u590d'
         '\u73b0\u3002'),
        ('- 3109\uff1a\u6b20\u5b9a\u51e0\u4f55'
         '\u5224\u51b3 stability_not_n_limited'
         '\uff1b\u968f\u673a\u5927\u5b50\u7a7a'
         '\u95f4 refit AUC\u22481.0=\u771f\u503c'
         '\u5f25\u6563\u5197\u4f59\uff1b\u8de8'
         '\u534a\u8fc1\u79fb 0.9998 vs \u5750'
         '\u6807\u91cd\u53e0 0.07=\u529f\u80fd'
         '\u7b49\u4ef7\u3002**\u5199\u5165\u5934'
         '\u7ec4\u56fa\u5b9a\u51e0\u4f55\u4e0d'
         '\u5b58\u5728\uff1b\u8bfb\u51fa\u7aef='
         '\u529f\u80fd\u7b49\u4ef7\u7aef\u53e3'
         '\u7c7b**\u3002',
         '- 3109\uff1a\u6b20\u5b9a\u51e0\u4f55'
         '=\u771f\u503c\u5f25\u6563\u5197\u4f59'
         '\uff08refit AUC\u22481.0\uff09\uff1b'
         '\u8de8\u534a\u8fc1\u79fb 0.9998 vs '
         '\u5750\u6807\u91cd\u53e0 0.07=\u529f'
         '\u80fd\u7b49\u4ef7\u3002**\u5199\u5165'
         '\u5934\u7ec4\u56fa\u5b9a\u51e0\u4f55'
         '\u4e0d\u5b58\u5728\uff1b\u8bfb\u51fa'
         '\u7aef=\u529f\u80fd\u7b49\u4ef7\u7aef'
         '\u53e3\u7c7b**\u3002')]
    for old, new in pairs:
        assert mem.count(old) == 1, old[:30]
        mem = mem.replace(old, new)
    old2 = ('\u4e0b\u4e00 3130\uff1a**')
    i2 = mem.find(old2)
    assert i2 > 0, 'next-line anchor missing'
    j2 = mem.find('\n', i2)
    new2 = ('\u4e0b\u4e00 3131\uff1a**\u5168'
            '\u5c42\u5355\u5c42 swap \u751f\u6210'
            '\u8c31 + \u6ce8\u5165\u5c42\u7ef4'
            '\u7a97\u53e3 + A1 \u534a\u751f\u6210'
            '\u5bf9\u7167 + \u5206\u53c9\u51b3'
            '\u7b56\u5c42\u5b9a\u4f4d\u3002**')
    mem = mem[:i2] + new2 + mem[j2:]
    oldm = '- max=3129\uff0c'
    assert mem.count(oldm) == 1
    mem = mem.replace(
        oldm, '- max=3130\uff0c')
    anchor = ('- 3129\uff08T4\uff09\uff1a')
    ia = mem.find(anchor)
    assert ia > 0
    new_line = (
        '- 3130\uff08T4\uff09\uff1a\u4f4d\u7f6e'
        '\u8c31\u60ac\u5d16 k1-11 \u5168\u22640.031'
        ' vs \u672c\u4f4d 0.182\uff08d50=d10=0\uff09'
        '=\u6ce8\u5165\u4e25\u683c\u7ed1\u5b9a'
        '\u8bfb\u70b9\u4f4d\uff08k12\u2261k0 \u6052'
        '\u7b49\u81f4 nonlocal \u95e8\u4f2a\u8c61'
        '\uff0c\u5b9e\u4e3a cliff-local\uff09\uff1b'
        'A1-fit \u5750\u6807\u4e0e P-fit 0/50 \u91cd'
        '\u53e0\u4e14 A1 \u534a\u6ce8\u5165\u53cc'
        '\u5411\u5f31\uff080.0149/0.0119<A1_MIN\uff09'
        '=\u65b9\u5411\u7279\u5f02+\u65e0\u53ef\u6ce8'
        '\u5165\u901a\u9053\uff1b\u5168 swap \u751f'
        '\u6210\u8de8 Phase \u4f4d\u7ea7\u590d\u73b0'
        '\uff08chg 1.0 first 178\uff09\u3002')
    mem = mem[:ia] + new_line + '\n' + \
        mem[ia:]
    assert len(mem) < 3000, len(mem)
    with io.open(MEMW, 'w',
                 encoding='utf-8') as f:
        f.write(mem)
    steps.append('memory: updated (%d '
                 'chars)' % len(mem))
else:
    steps.append('memory: exists, skip')

for s in steps:
    print(s)
print('CLOSEOUT_OK (%d steps)'
      % len(steps))
