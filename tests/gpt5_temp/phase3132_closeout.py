# -*- coding: utf-8 -*-
"""Phase 3132 closeout: idempotent five-write
chain (ledger -> MEMO -> wlogs -> MEMORY).
MEMO title SHORT per user rule (2026-09-24).
All numeric claims read from result.json;
hard frozen asserts before any write.
Verdict-dependent prose via branch fns."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3132'
        r'\omega_p130_forkcausal_'
        r'single256')
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

REF30 = {8: 208 / 256.0, 9: 205 / 256.0,
         13: 137 / 256.0, 29: 45 / 256.0}
CHG40_REF = {4: 0.96875, 8: 0.734375,
             9: 0.78125, 13: 0.421875,
             17: 0.84375, 29: 0.109375,
             33: 0.09375, 38: 1.0}
LAYERS_C = [4, 8, 9, 13, 17, 29, 33, 38]
INJ_ABS_GATE = 0.10
RESC_GATE = 0.10
SPEC256_TOL = 0.08
SUB64_TOL = 0.033
FORK_LO = 15
FORK_HI = 19

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
assert r['smoke'] is False
V = r['verdict']
vd = V.split('|')
assert len(vd) == 7
assert vd[0] == 'a_3131_ok'
assert vd[1] in ('l17_inj_rewrites',
                 'l17_inj_ambiguous',
                 'l17_inj_null')
assert vd[2] in ('rescue_l17_present',
                 'rescue_not_specific',
                 'rescue_absent')
assert vd[3] in ('ctrl33_quiescent',
                 'ctrl33_active')
assert vd[4] in ('spec256_ok',
                 'spec256_dev')
assert vd[5] in ('fork_l17_gate',
                 'fork_l17_amp',
                 'fork_anom')
assert vd[6] == 'coverage_full'
pb = r['part_b']
pc = r['part_c']

# ---- part_b asserts ----
assert pb['sgn_exp'] in (1, -1)
assert len(pb['co50_sha8']) == 8
assert abs(pb['delta_l17']
           - 0.10 * pb['mean_row_l2']) \
    < 1e-12
it = pb['inj_trials']
assert len(it) == 8
keys_l17 = ['17|step0|+1', '17|step0|-1',
            '17|allstep|+1',
            '17|allstep|-1']
keys_l38 = ['38|step0|+1', '38|step0|-1']
keys_l33 = ['33|step0|+1', '33|step0|-1']
assert set(it) == set(keys_l17 + keys_l38
                      + keys_l33)
for k, v in it.items():
    assert 0.0 <= v['chg'] <= 1.0
    assert v['first'] >= 0
chg17 = pb['chg_l17_best']
chg38 = pb['chg_l38_best']
chg33 = pb['chg_l33_best']
assert abs(chg17 - max(it[k]['chg']
                       for k in keys_l17)) \
    < 1e-12
assert abs(chg38 - max(it[k]['chg']
                       for k in keys_l38)) \
    < 1e-12
assert abs(chg33 - max(it[k]['chg']
                       for k in keys_l33)) \
    < 1e-12
if chg17 >= INJ_ABS_GATE \
        and chg17 > chg33:
    l17_exp = 'l17_inj_rewrites'
elif chg17 >= INJ_ABS_GATE:
    l17_exp = 'l17_inj_ambiguous'
else:
    l17_exp = 'l17_inj_null'
assert vd[1] == l17_exp
ctrl_exp = ('ctrl33_quiescent'
            if chg33 < 0.5 * max(chg17,
                                 INJ_ABS_GATE)
            else 'ctrl33_active')
assert vd[3] == ctrl_exp
rt = pb['resc_trials']
assert len(rt) == 6
rkeys_l17 = keys_l17
rkeys_l38 = keys_l38
assert set(rt) == set(rkeys_l17
                      + rkeys_l38)
resc17 = pb['rescue_l17']
resc38 = pb['rescue_l38']
assert abs(resc17 - max(rt[k]['rescue']
                        for k in rkeys_l17)) \
    < 1e-12
assert abs(resc38 - max(rt[k]['rescue']
                        for k in rkeys_l38)) \
    < 1e-12
if resc17 >= RESC_GATE \
        and resc17 > resc38:
    rescue_exp = 'rescue_l17_present'
elif resc38 >= RESC_GATE:
    rescue_exp = 'rescue_not_specific'
else:
    rescue_exp = 'rescue_absent'
assert vd[2] == rescue_exp
mta = pb['med_tf_avg']
assert len(mta) == 41
# [rev-3132a] full-track convention:
# inject at layers[17] output pos0 ->
# track-last margin changes from
# layer index 19 (=L18 output) on;
# zero region is index 0..18.
for v in mta[:19]:
    assert abs(v) < 1e-9
assert mta[19] > 0
peak_tf = pb['peak_tf']
assert peak_tf == int(max(
    range(41), key=lambda i: mta[i]))
if FORK_LO <= peak_tf <= FORK_HI:
    fork_exp = 'fork_l17_gate'
elif peak_tf > FORK_HI:
    fork_exp = 'fork_l17_amp'
else:
    fork_exp = 'fork_anom'
assert vd[5] == fork_exp
assert -1.0 <= pb['spearman_tf_vs_3131'] \
    <= 1.0

# ---- part_c asserts ----
assert pc['layers_c'] == LAYERS_C
c2 = pc['chg256']
assert set(c2) == {str(l)
                   for l in LAYERS_C}
for l, v in c2.items():
    assert 0.0 <= v <= 1.0
dev30 = pc['dev30']
assert set(dev30) == {'8', '9', '13', '29'}
for k in dev30:
    assert abs(dev30[k]
               - abs(c2[k] - REF30[int(k)])) \
        < 1e-12
spec_exp = ('spec256_ok'
            if all(v <= SPEC256_TOL
                   for v in dev30.values())
            else 'spec256_dev')
assert vd[4] == spec_exp
dev31 = pc['dev31']
assert set(dev31) == {str(l)
                      for l in LAYERS_C}
for k in dev31:
    assert 0.0 <= dev31[k] <= 1.0
n_over = pc['n_dev31_over']
assert n_over == sum(
    1 for v in dev31.values()
    if v > SUB64_TOL)
assert len(pc['sel_sha8']) == 8
assert len(pc['sel64_sha8']) == 8
assert r['runtime_s'] > 2000

# branch prose
d17 = pb['delta_l17']
rho_top = pb['rho_top10_abs']
sgn_exp = pb['sgn_exp']
dm17 = pb['dm_med17']
fneg = pb['dm_frac_neg17']
sp_tf = pb['spearman_tf_vs_3131']

f1 = (
    '1. **L17 分叉层因果化：弱注入即改写'
    '生成，swap 效应可被反向注入部分救回'
    '（因果三角闭合）**。L17 prompt 末位 '
    'margin-rho top50 坐标（\u03b4=%.4f='
    '0.10\u00d7row_l2\uff0c\u03c1_top10 %s）'
    '\u00b1\u03b4 \u6ce8\u5165\u2192 12-token '
    '\u6539\u5199\u7387 best %.4f\uff08L38 '
    '\u5cf0\u5c42\u5bf9\u7167 %.4f\u3001L33 '
    '\u4f4e\u8c37\u5bf9\u7167 %.4f\uff09\u2192 '
    '%s\u3002swap{8,9,13,29}+\u53cd\u5411\u6ce8'
    '\u5165 rescue %.4f\uff08L38 \u5bf9\u7167 '
    '%.4f\uff09\u2192 %s\u3002**\u91cd\u590d'
    '\uff1a%s**\n'
    % (d17,
       ' '.join('%.2f' % v for v in rho_top),
       chg17, chg38, chg33, vd[1], resc17,
       resc38, vd[2],
       ('L17 \u5355\u70b9\u5f31\u5e72\u9884'
        '\u5373\u8db3\u4ee5\u6539\u5199\u751f'
        '\u6210\u4e14\u5177\u4e2d\u4ecb\u5fc5'
        '\u8981\u6027\u2014\u2014\u5206\u53c9'
        '\u51b3\u7b56\u5c42\u56e0\u679c\u5730'
        '\u4f4d\u6210\u7acb'
        if vd[1] == 'l17_inj_rewrites'
        and vd[2] == 'rescue_l17_present'
        else ('L17 \u6539\u5199\u80fd\u529b'
              '\u6210\u7acb\u4f46 rescue \u4e0d'
              '\u7279\u5f02\u2014\u2014\u51b3'
              '\u7b56\u5c42\u4e0e\u5199\u5165'
              '\u5cf0\u5171\u4eab\u901a\u9053'
              if vd[1] == 'l17_inj_rewrites'
              else 'L17 \u5f31\u6ce8\u5165\u4e0d'
                   '\u8db3\u4ee5\u6539\u5199'
                   '\u2014\u2014\u51b3\u7b56\u5c42'
                   '\u9700\u5f3a\u5e72\u9884'))))
f2 = (
    '2. **teacher-forced margin \u5c42\u8f6e'
    '\u5ed3\uff1aL17 \u6ce8\u5165\u7684\u8bfb'
    '\u51fa\u6270\u52a8\u5728\u6ce8\u5165\u5c42'
    '\u5373\u8fbe\u5cf0\uff0c\u4e0e 3131 swap '
    'dm \u8f6e\u5ed3\u540c\u6784**\u3002\u53cc'
    '\u5411 \u00b1\u03b4 \u5e73\u5747 med|df| '
    '\u5cf0 L%02d\uff08L0\u201316 \u7cbe\u786e '
    '0\uff09\u2192 %s\uff1b\u4e0e 3131 med|dm| '
    '\u5168\u5c42 Spearman %.3f\u3002dm \u7b26'
    '\u53f7\uff1amed dm[L17]=%.4f\u3001'
    'frac_neg=%.3f \u2192 rescue \u9884\u671f'
    '\u65b9\u5411 %+d\u3002**\u91cd\u590d\uff1a'
    '%s**\n'
    % (peak_tf, vd[5], sp_tf, dm17, fneg,
       sgn_exp,
       ('\u6270\u52a8\u76f4\u901a readout'
        '\uff08gate \u578b\uff09'
        if vd[5] == 'fork_l17_gate'
        else '\u6270\u52a8\u7ecf\u5199\u5165'
             '\u94fe\u653e\u5927\uff08amp \u578b'
             '\uff09' if vd[5] == 'fork_l17_amp'
        else '\u5cf0\u4f4d\u5f02\u5e38')))
f3 = (
    '3. **\u5355\u5c42\u8c31 256 \u7cbe\u5316'
    '\uff1a\u95e8\u5c42\u4e0e 3130 \u8de8\u6837'
    '\u672c\u96c6\u590d\u73b0\u300164 \u5b50'
    '\u96c6\u5d4c\u5957\u81ea\u6d3d**\u3002'
    'sel64\u2282sel256\uff08seed 3131\u2282'
    '3132\uff0csha8=%s\uff09\u00d7\u5c42 '
    '%s\uff1achg256 %s\u3002\u95e8\u5c42 dev '
    'vs 3130\uff08256 \u6837\uff09%s \u2192 '
    '%s\uff1b64 \u5b50\u96c6 dev vs 3131 %s'
    '\uff08\u8d85\u95e8 %d \u5c42\uff0c\u8f6f'
    '\u8bb0\u5f55\uff09\u3002**\u91cd\u590d'
    '\uff1a%s**\n'
    % (pc['sel_sha8'],
       '{%s}' % ','.join(str(l)
                         for l in LAYERS_C),
       ' '.join('L%d:%.3f' % (l, c2[str(l)])
                for l in LAYERS_C),
       ' '.join('L%s:%.3f' % (k, dev30[k])
                for k in sorted(dev30)),
       vd[4],
       ' '.join('L%s:%.3f' % (k, dev31[k])
                for k in sorted(dev31)),
       n_over,
       ('\u8c31\u951a\u56fa\u5316\uff1a\u95e8'
        '\u5c42\u5c42\u7ea7\u7ed3\u6784\u8de8'
        '\u6837\u672c\u96c6\u7a33\u5065'
        if vd[4] == 'spec256_ok'
        else '\u95e8\u5c42\u504f\u5dee\u8d85'
             '\u5bb9\u5dee\u2014\u2014\u9700'
             '\u68c0\u67e5\u6279\u7ec4\u6210'
             '\u6f02\u79fb\u4e0e\u62bd\u6837'
             '\u6548\u5e94')))
nums = (
    'Part B\uff1a\u03b4_l17=%.4f\uff08row_l2 '
    '%.2f\uff09\uff1bchg_inj best L17 %.4f / '
    'L38 %.4f / L33 %.4f\uff1brescue L17 '
    '%.4f / L38 %.4f\uff1b\u5168\u7ec4 first '
    '%s\u3002Part B4\uff1apeak_tf=L%02d\u3001'
    'spearman %.3f\u3002Part C\uff1achg256 '
    '%s\uff1bdev30 max %.4f\uff1bdev31 max '
    '%.4f\uff08over %d\uff09\u3002'
    % (d17, pb['mean_row_l2'], chg17, chg38,
       chg33, resc17, resc38,
       ' '.join('%d:%d' % (int(k.split('|')[0]),
                           it[k]['first'])
                for k in ('17|step0|+1',
                          '38|step0|+1',
                          '33|step0|+1')),
       peak_tf, sp_tf,
       ' '.join('L%d:%.3f' % (l, c2[str(l)])
                for l in LAYERS_C),
       max(dev30.values()),
       max(dev31.values()) if dev31 else -1.0,
       n_over))
hards = (
    '\u2460\u6ce8\u5165\u4ec5 \u00b1\u03b4 '
    '\u6807\u91cf\u4e8e top50 \u5750\u6807'
    '\uff0c\u672a\u626b\u5750\u6807\u6570\u4e0e'
    '\u5e45\u5ea6\u5256\u9762\uff1b\u2461'
    'rescue \u9884\u671f\u65b9\u5411\u7531 '
    '3131 dm \u4e2d\u4f4d\u6570\u63a8\u5bfc'
    '\uff0c\u5750\u6807 rho \u7b26\u53f7\u6df7'
    '\u5408\u2192 \u00b1 \u53cc\u5411\u53d6 '
    'max \u53ef\u80fd\u9ad8\u4f30 rescue'
    '\uff1b\u2462\u8c31 256 \u4e0e 3130 \u6837'
    '\u672c\u96c6\u4e0d\u540c\uff08\u8de8\u96c6'
    '\u5bf9\u7167\u5bb9\u5dee 0.08\uff09\u3001'
    '64 \u5b50\u96c6\u6279\u7ec4\u6210\u6f02'
    '\u79fb\u65e0\u5148\u9a8c\uff08\u8f6f\u95e8'
    '\uff09\uff1b\u2463tf \u8f6e\u5ed3\u4e3a '
    'teacher-forced \u5168\u8f68\u8ff9\u60ef'
    '\u4f8b\uff083131 \u5bf9\u9f50\u3001pos0 '
    '\u6ce8\u5165\u672b\u5217\u8bfb\u51fa'
    '\uff09\uff0c\u672a\u505a\u751f\u6210\u5faa'
    '\u73af\u5185\u9010\u6b65\u5e72\u9884\u8f68'
    '\u8ff9\u3002')
mech = (
    '\u2460\u5206\u53c9\u51b3\u7b56\u5c42 '
    'L17 \u4e09\u89d2\u8865\u5168\uff1a'
    '\u76f8\u5173\uff083131 dm \u5cf0\uff09'
    '\u2192 \u5145\u5206\uff08\u6ce8\u5165'
    '\u6539\u5199 %s\uff09\u2192 \u5fc5\u8981'
    '\uff08rescue %s\uff09\uff1b\u2461\u5c42'
    '\u8c31\u7cbe\u5316 %s\uff1a\u51b3\u7b56'
    '\u95f8\u53e3\uff08L17\uff09\u4e0e\u5199'
    '\u5165\u4e3b\u9053\uff08L38\uff09\u5206'
    '\u79bb\uff0c\u4f4e\u8c37 L29/L33 \u590d'
    '\u73b0\uff1b\u2462A1 \u9996\u4f4d\u4e0d'
    '\u5bf9\u79f0\uff08614 vs 178\uff09'
    '\u5f85 3133 \u5c42\u4f4d\u5256\u9762'
    '\u3002'
    % ('\u6210\u7acb' if vd[1]
       == 'l17_inj_rewrites' else '\u5f31',
       '\u6210\u7acb' if vd[2]
       == 'rescue_l17_present' else
       ('\u4e0d\u7279\u5f02' if vd[2]
        == 'rescue_not_specific'
        else '\u7f3a\u5e31'),
       '\u951a\u56fa' if vd[4] == 'spec256_ok'
       else '\u504f\u5dee'))
if vd[1] == 'l17_inj_rewrites' \
        and vd[2] == 'rescue_l17_present':
    p1 = ('\u2460L17 \u4e0a\u6e38\u6e90\u5b9a'
          '\u4f4d\uff1aL17 \u8bfb\u51fa\u7684'
          '\u8f93\u5165\u6765\u6e90\uff08\u54ea'
          '\u4e9b\u5c42\u5199\u5165 L17 \u8868'
          '\u5f81\uff1fL38 \u8f93\u51fa\u5230 '
          'L17 \u8f93\u5165\u7684\u4f20\u5bfc'
          '\u8def\u5f84\uff09\u2014\u2014\u5c42'
          '\u95f4\u4f20\u5bfc\u6838\u9a8c\u3002')
elif vd[2] == 'rescue_absent':
    p1 = ('\u2460L17 \u76f8\u5173\u975e\u4e2d'
          '\u4ecb\u7684\u66ff\u4ee3\u89e3\u91ca'
          '\u68c0\u9a8c\uff1a\u5f3a\u5e72\u9884'
          '\uff08\u5168\u7ef4 margin \u5411\u91cf'
          '\u3001\u591a\u70b9\u4f4d\u8054\u5408'
          '\uff09\u4e0e\u751f\u6210\u6b65\u9aa4'
          '\u7ea7\u8f68\u8ff9\u3002')
else:
    p1 = ('\u2460rescue \u5750\u6807\u7ea7'
          '\u5256\u9762\uff1a\u54ea\u4e9b\u5750'
          '\u6807\u643a\u5e26 rescue \u80fd'
          '\u529b\uff08top50 \u9010\u4e2a\u6d88'
          '\u878d\uff09\u3002')
p2 = ('\u2461A1 \u9996\u4f4d\u5206\u53c9\u4e0d'
      '\u5bf9\u79f0\uff08first 614 vs 178'
      '\uff09\u7684\u5c42\u4f4d\u5256\u9762'
      '\uff1aA1 \u534a\u533a fork-layer \u8f6e'
      '\u5ed3 + \u5c42\u7a97\u5bf9\u7167\u3002')
p3 = ('\u2462\u8c31\u951a\u5165 ledger \u540e'
      '\uff0c\u5199\u5165\u94fe\u5168\u94fe'
      '\uff08\u5750\u6807\u00d7\u5c42\u00d7'
      '\u4f4d\u7f6e\u00d7\u65b9\u5411\uff09'
      '\u56db\u7ef4\u5b8c\u578b\u5411 A1 \u65b9'
      '\u5411\u8fc1\u79fb\u3002')
prereg = p1 + p2 + p3
title = ('\n## Phase 3132: \u03a9-P130 '
         '\u5206\u53c9\u5c42\u56e0\u679c\u5316'
         '+\u5355\u5c42\u8c31 256'
         '\uff08T4 \u7b2c15Phase\uff09'
         '[' + STAMP + ']\n\n')
assert len(title) < 110
sec = (
    title
    + '**\u6027\u8d28**\uff1aT4 \u7b2c 15 '
    'Phase\uff0c3131 MEMO \u00a75 \u9884\u6ce8'
    '\u518c\u4e09\u9879\u6267\u884c\uff0c'
    'design_seal.json \u89c2\u6d4b\u524d\u51bb'
    '\u7ed3\u3002Part A offline 3131 \u94fe'
    '\u63a5\u65ad\u8a00\uff08result sha8 '
    '25613e9e + seal a55569e6 \u91cd\u7b97'
    '\uff09\uff1bPart B0 dm \u7b26\u53f7\u7edf'
    '\u8ba1\uff083131 \u51bb\u7ed3 npz\uff09'
    '\uff1bPart B1 L17 prompt \u672b\u4f4d '
    '\u5750\u6807\u6355\u83b7\uff08672 \u6837 '
    'margin-rho\uff09\uff1bPart B2 \u6ce8\u5165 '
    '8 \u7ec4\u751f\u6210\uff1bPart B3 rescue '
    '6 \u7ec4\uff1bPart B4 tf margin \u5c42'
    '\u8f6e\u5ed3\uff1bPart C sel64 \u5d4c\u5957 '
    '256 \u8c31 \u00d7 8 \u5c42\u3002\u8fd0\u884c '
    + ('%.0fs' % r['runtime_s']) + '\u3002\n\n'
    + '### 1. \u4e09\u5927\u53d1\u73b0\uff08'
    '\u91cd\u590d\u4e09\u904d\uff09\n'
    + f1 + f1 + f1 + f2 + f2 + f2 + f3
    + f3 + f3 + '\n'
    + '### 2. \u5173\u952e\u6570\u503c\n'
    + nums + '\n\n'
    + '### 3. \u786c\u4f24\n' + hards
    + '\n\n### 4. \u673a\u5236\u62fc\u56fe'
    '\u66f4\u65b0\n' + mech
    + '\n\n### 5. 3133 \u9884\u6ce8\u518c'
    '\uff08\u89c2\u5bdf\u540e\u51bb\u7ed3\uff09'
    '\n' + prereg + '\n\n'
    + '\u4ea7\u7269\uff1a`tests/glm5/result/'
    'rdc_query_construction_20260913/'
    'phase3132/omega_p130_forkcausal_'
    'single256/`\uff08result.json\u3001'
    'design_seal.json\u3001run_log.txt\u3001'
    'p130_readout.npz\uff09\uff1b\u811a\u672c '
    '`tests/glm5/phase3132_omega_p130_'
    'forkcausal_single256.py`\u3002')
assert len(sec) > 2500

# sha8 of result.json
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

steps = []
if 'meas3132_omega_p130' in io.open(
        LEDGER, encoding='utf-8').read():
    steps.append('ledger: exists, skip')
else:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    claim = (
        'Omega-P130 (3132, T4 fifteenth '
        'phase: glm4 L17 fork-layer '
        'causalization. L17 prompt-last '
        'coords margin-rho top50 '
        '(delta=%.4f) +/-inject -> 12-tok '
        'rewrite best %.4f (L38 %.4f, L33 '
        '%.4f) -> %s; swap{8,9,13,29}+L17 '
        'counter-inject rescue %.4f (L38 '
        'ctrl %.4f) -> %s; tf margin '
        'profile peak L%02d, spearman vs '
        '3131 dm %.3f -> %s; spectrum '
        'sel64-nested 256 (sha8=%s) x '
        'layers {4,8,9,13,17,29,33,38}, '
        'gate-layer dev vs 3130 max '
        '%.4f -> %s - verdict '
        % (d17, chg17, chg38, chg33, vd[1],
           resc17, resc38, vd[2], peak_tf,
           sp_tf, vd[5], pc['sel_sha8'],
           max(dev30.values()), vd[4])) + V
    entry = {
        'meas_id':
            'meas3132_omega_p130_fork'
            'causal_single256',
        'phase': 3132,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3132/omega_p130_.../'
                'result.json sha256_8='
                + sha8,
            'npz': 'p130_readout.npz'},
        'hashes': {'result_sha256_8':
                   sha8},
        'anchors': [],
        'note': 'l17 inject %s; rescue '
                '%s; tf %s; spectrum %s'
                % (vd[1], vd[2], vd[5],
                   vd[4])}
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
if '## Phase 3132:' in memo:
    steps.append('memo: exists, skip')
else:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    steps.append('memo: appended (short '
                 'title, 5 sections)')

WDATE = NOW.strftime('%Y-%m-%d')
wline = ('- Phase 3132 Omega-P130 closeout: '
         'verdict ' + V + '; ledger sha8 '
         + json.load(io.open(
             LEDGER,
             encoding='utf-8'))
         ['ledger_sha256_8']
         + '; MEMO 3132 section; runtime '
         + ('%.0fs' % r['runtime_s']) + '.')
for wd in WLOGS:
    wl = wd + '\\' + WDATE + '.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3132 Omega-P130 closeout' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        steps.append('wlog: ' + wl[:12])
    else:
        steps.append('wlog: exists '
                     + wl[:12])

mem = io.open(MEMW, encoding='utf-8').read()
if '3132\uff08T4\uff09' in mem:
    steps.append('memory: exists, skip')
else:

    def compress_line(memtxt, pfx,
                      newline):
        ls = memtxt.split('\n')
        idx = [i for i, e in enumerate(ls)
               if e.startswith(pfx)]
        assert len(idx) == 1, (pfx, idx)
        ls[idx[0]] = newline
        return '\n'.join(ls)

    mem = compress_line(
        mem,
        '- 3131\uff08T4\uff09\uff1a',
        '- 3131\uff08T4\uff09\uff1a\u6ce8\u5165'
        '\u5c42\u7a97 L20\u201328 wide\uff08'
        'L24 \u53cc\u951a\u590d\u73b0\uff09='
        '\u60ac\u5d16\u5c42\u7a33\u5065\uff1b'
        '\u5355\u5c42\u8c31\u5cf0 L38 dom 1.0'
        '\uff08L00 \u5e73\u51e1\u5df2\u6391'
        '\uff09\uff1bA1 \u534a\u751f\u6210'
        '\u5bf9\u79f0\uff1b\u5206\u53c9\u51b3'
        '\u7b56\u5c42 L17\u3002')
    mem = compress_line(
        mem,
        '- 3130\uff08T4\uff09\uff1a',
        '- 3130\uff08T4\uff09\uff1a\u4f4d\u7f6e'
        '\u8c31\u60ac\u5d16 k1\u201311 \u5168'
        '\u22640.031 vs \u672c\u4f4d 0.182='
        '\u6ce8\u5165\u4e25\u683c\u7ed1\u5b9a'
        '\u8bfb\u70b9\u4f4d\uff1bA1-fit \u4e0e '
        'P-fit \u5750\u6807 0/50 \u91cd\u53e0'
        '\u4e14\u53cc\u5411\u5f31=\u65b9\u5411'
        '\u7279\u5f02\uff1b\u5168 swap \u751f'
        '\u6210\u8de8 Phase \u4f4d\u7ea7\u590d'
        '\u73b0\u3002')
    mem = compress_line(
        mem,
        '- 3129\uff08T4\uff09\uff1a',
        '- 3129\uff08T4\uff09\uff1a\u5242\u91cf'
        '\u5355\u8c03 \u03b4=0.10 \u4f4d\u7ea7'
        '\u590d\u73b0\uff1b\u6ce8\u5165\u53cc'
        '\u91cd\u5c40\u57df=\u8bfb\u70b9\u7ed1'
        '\u5b9a\uff1bswap \u91cd\u5199 672/672 '
        '\u800c margin \u4ec5 172 \u7ffb\u8f6c'
        '\u3002')
    old2 = '\u4e0b\u4e00 3132\uff1a**'
    i2 = mem.find(old2)
    assert i2 > 0, 'next-line anchor'
    j2 = mem.find('\n', i2)
    new2 = ('\u4e0b\u4e00 3133\uff1a**L17 '
            '\u4e0a\u6e38\u6e90\u5b9a\u4f4d'
            '\uff08\u5c42\u95f4\u4f20\u5bfc'
            '\u6838\u9a8c\uff09 + A1 \u9996'
            '\u4f4d\u4e0d\u5bf9\u79f0\u5c42'
            '\u4f4d\u5256\u9762 + \u5199\u5165'
            '\u94fe\u56db\u7ef4\u5b8c\u578b'
            ' A1 \u8fc1\u79fb\u3002**')
    mem = mem[:i2] + new2 + mem[j2:]
    oldm = '- max=3131\uff0c'
    assert mem.count(oldm) == 1
    mem = mem.replace(oldm,
                      '- max=3132\uff0c')
    anchor = '- 3130\uff08T4\uff09\uff1a'
    ia = mem.find(anchor)
    assert ia > 0
    new_line = (
        '- 3132\uff08T4\uff09\uff1aL17 \u5f31'
        '\u6ce8\u5165\u6539\u5199 %.3f\u3001'
        'rescue %.3f\uff08%s/%s\uff09='
        '\u5206\u53c9\u5c42\u56e0\u679c\u4e09'
        '\u89d2 %s\uff1btf \u8f6e\u5ed3\u5cf0 '
        'L%02d %s\uff1b\u8c31 256 \u95e8\u5c42 '
        '%s\uff08sel64 \u5d4c\u5957\uff09'
        '\u3002'
        % (chg17, resc17, vd[1], vd[2],
           ('\u95ed\u5408' if vd[1]
            == 'l17_inj_rewrites'
            and vd[2] == 'rescue_l17_present'
            else '\u90e8\u5206'),
           peak_tf,
           'gate' if vd[5] == 'fork_l17_gate'
           else ('amp' if vd[5]
                 == 'fork_l17_amp' else 'anom'),
           '\u951a\u56fa' if vd[4]
           == 'spec256_ok' else '\u504f\u5dee'))
    mem = mem[:ia] + new_line + '\n' + \
        mem[ia:]
    assert len(mem) < 3000, len(mem)
    with io.open(MEMW, 'w',
                 encoding='utf-8') as f:
        f.write(mem)
    steps.append('memory: updated (%d '
                 'chars)' % len(mem))

for s in steps:
    print(s)
print('CLOSEOUT_OK (%d steps)'
      % len(steps))
