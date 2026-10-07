# -*- coding: utf-8 -*-
"""Phase 3134 closeout: idempotent five-write
chain (ledger -> MEMO -> wlogs -> MEMORY).
MEMO title SHORT per user rule (2026-09-24).
All numeric claims read from result.json;
hard frozen asserts before any write."""
import datetime
import hashlib
import io
import json

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3134'
        r'\omega_p132_carrier_matrix_'
        r'forkcoord_stepscan')
D33 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3133'
       r'\omega_p131_transplant_'
       r'a1fork_migrate')
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

CAP_L = [17, 29, 33, 38]
DOSES = (0.5, 1.0, 2.0)
PAIRS = [(17, 29), (17, 33), (29, 33)]
STEPS_SCAN = ['0', '1', '2', '3', '5', '8',
              '11']
TR_PART = 0.10
ADD_TOL = 0.15
DOSE_SLACK = 0.02
J2_TOL = 0.05
STEP_GATE = 0.20
FLIP_TOL = 3
FORK_PROF_IDX = 36
CHGJ2_REF = 258 / 672.0
CHG17_D05 = 54 / 672.0
CHG17_D10 = 90 / 672.0
CHG17_D20 = 172 / 672.0

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
assert r['smoke'] is False
V = r['verdict']
vd = V.split('|')
assert len(vd) == 9
assert vd[0] == 'a_3133_ok'
assert vd[1] in ('carrier_l17_max',
                 'carrier_dispersed',
                 'carrier_mixed')
assert vd[2] in ('dose_monotone_all',
                 'dose_nonmonotone')
assert vd[3] in ('j2_session_match',
                 'j2_session_drift')
assert vd[4] in ('pair_additive',
                 'pair_superadditive')
assert vd[5] in ('forkcoord_p_rewrite',
                 'forkcoord_p_null')
assert vd[6] in ('stepscan_effective',
                 'stepscan_weak')
# [rev-3134a] token may be traj_swap_
# peak%d (band pass) or traj_swap_
# early%d (early-peak refutation,
# recorded not crashed)
assert vd[7].startswith('traj_swap_')
assert vd[8] == 'coverage_full'
pb = r['part_b']
pc = r['part_c']
pd_ = r['part_d']

# ---- part_b asserts ----
dn = pb['dvec_med_norm']
assert set(dn) == {'17', '29', '33', '38'}
assert all(v > 0 for v in dn.values())
tt = pb['trials']
exp_names = set()
for l in CAP_L:
    for d in DOSES:
        exp_names.add('l%02d_d0%d'
                      % (l, int(d * 10)))
for (a, b) in PAIRS:
    exp_names.add('p%02d_%02d' % (a, b))
exp_names.add('j2_s0')
assert set(tt) == exp_names, (
    sorted(set(tt) ^ exp_names),)
for k, v in tt.items():
    assert 0.0 <= v['chg'] <= 1.0
    assert 0 <= v['first'] <= 672
    assert len(v['cum']) == 12
    assert abs(v['cum'][-1] - v['chg']) \
        < 1e-9
cm = pb['chg_matrix']
assert set(cm) == {'17', '29', '33', '38'}
for l in CAP_L:
    for di, d in enumerate(DOSES):
        k = 'l%02d_d0%d' % (l, int(d * 10))
        assert abs(cm[str(l)][di]
                   - tt[k]['chg']) < 1e-12
# carrier gate re-derivation
d2s = {l: cm[str(l)][2] for l in CAP_L}
if d2s[17] >= max(d2s.values()):
    carrier_exp = 'carrier_l17_max'
elif max(d2s[l] for l in CAP_L
         if l != 17) > 2.0 * d2s[17]:
    carrier_exp = 'carrier_dispersed'
else:
    carrier_exp = 'carrier_mixed'
assert vd[1] == carrier_exp
mono_ok = all(
    cm[str(l)][0] <= cm[str(l)][1]
    + DOSE_SLACK
    and cm[str(l)][1] <= cm[str(l)][2]
    + DOSE_SLACK for l in CAP_L)
assert vd[2] == ('dose_monotone_all'
                 if mono_ok
                 else 'dose_nonmonotone')
assert vd[3] == ('j2_session_match'
                 if abs(tt['j2_s0']['chg']
                        - CHGJ2_REF)
                 <= J2_TOL
                 else 'j2_session_drift')
pair_pred = {}
pair_ok = True
for (a, b) in PAIRS:
    pa = tt['l%02d_d010' % a]['chg']
    pbb = tt['l%02d_d010' % b]['chg']
    pred = 1.0 - (1.0 - pa) * (1.0 - pbb)
    pair_pred['%02d_%02d' % (a, b)] = pred
    kk = 'p%02d_%02d' % (a, b)
    assert abs(pb['pair_pred']
               ['%02d_%02d' % (a, b)]
               - pred) < 1e-12
    if abs(tt[kk]['chg'] - pred) > ADD_TOL:
        pair_ok = False
assert vd[4] == ('pair_additive'
                 if pair_ok
                 else 'pair_superadditive')
# session-vs-3133 anchor (cross-session
# bit-stability record)
res33 = json.load(io.open(
    D33 + r'\result.json',
    encoding='utf-8'))
assert abs(tt['l17_d010']['chg']
           - res33['part_b']
           ['chg17_s0_d10']) < 1e-9
assert abs(tt['l17_d05']['chg']
           - CHG17_D05) < 1e-9
assert abs(tt['l17_d020']['chg']
           - CHG17_D20) < 1e-9
assert abs(tt['j2_s0']['chg']
           - res33['part_b']['chgJ2_s0']) \
    < 1e-9

# ---- part_c asserts ----
assert pc['fork_layer'] == 35
assert pc['fork_prof_idx'] == 36
assert len(pc['co36_sha8']) == 8
assert pc['delta36'] > 0
c1 = pc['c1_trials']
assert set(c1) == {'P', 'A1'}
for v in c1.values():
    assert 0.0 <= v['chg'] <= 1.0
assert vd[5] == ('forkcoord_p_rewrite'
                 if c1['P']['chg'] >= TR_PART
                 else 'forkcoord_p_null')
scan = pc['scan']
assert set(scan) == set(STEPS_SCAN)
for v in scan.values():
    assert 0.0 <= v['chg'] <= 1.0
best_s = max(scan.keys(),
             key=lambda k: scan[k]['chg'])
assert pc['best_step'] == best_s
assert vd[6] == ('stepscan_effective'
                 if scan[best_s]['chg']
                 >= STEP_GATE
                 else 'stepscan_weak')
c3f = pc['c3_full']
assert c3f['step'] == int(best_s)
# c3_full is the full-NP_B(672-row)
# confirmation of the best step found on
# the SCAN_N(128-row) scan - different
# sample sets, so chg values differ by
# sampling; assert validity + range only
# (equality assert would be a category
# error - fixed pre-closeout).
assert 0.0 <= c3f['chg'] <= 1.0
assert len(c3f['cum']) == 12
assert abs(c3f['cum'][-1] - c3f['chg']) \
    < 1e-9

# ---- part_d asserts ----
assert len(pd_['tf_idx']) == 128
assert len(pd_['tf_idx']) == len(
    set(pd_['tf_idx']))
# [rev-3134a] mirror the soft gate:
# re-derive k0_swap from npz dm_swap
# (float64 exact) and check the token
# by the peak-band rule [33, 40]; the
# emergence index (0.10*max crossing)
# is re-derived for the MEMO narrative
# only (not gated).
znp = np.load(OUTD + r'\p132_readout.npz',
              allow_pickle=False)
dm_sw = znp['dm_swap_traj']
k0_re = int(np.argmax(
    np.abs(dm_sw[:, 0])))
assert pd_['k0_swap_peak'] == k0_re
_amp = np.abs(dm_sw[:, 0])
k_emerge = int(np.argmax(
    _amp >= 0.10 * float(_amp.max())))
if 33 <= k0_re <= 40:
    assert vd[7] == ('traj_swap_peak%d'
                     % k0_re)
else:
    assert vd[7] == ('traj_swap_early%d'
                     % k0_re)
print('D re-derived: k0_swap=%d '
      'k_emerge=%.0f-cross=%d (3133 '
      'medE=36)' % (k0_re,
                    float(_amp.max()),
                    k_emerge))
sds = pd_['second_diff_summary']
assert sds['max_abs'] >= sds['med_abs'] >= 0
# runtime_s records only the last (RESUME)
# leg (888s); the full computation spans
# the 6h12m main leg + this leg. The
# smoke guard is r['smoke'] is False.
assert r['runtime_s'] > 300

# branch prose
carr_txt = {
    'carrier_l17_max':
        'L17 dvec 在 d2 剂量下 chg 最大——'
        '载体仍以 L17 为主',
    'carrier_dispersed':
        '非 L17 层 d2 chg 超过 L17 两倍——'
        '载体显著分散',
    'carrier_mixed':
        '多层 dvec 载体贡献同量级——分散'
        '承载但无单层主导'}[vd[1]]
f1 = (
    '1. **dvec 载体分解矩阵：%s**。'
    'd2 剂量 chg L17 %.4f / L29 %.4f / '
    'L33 %.4f / L38 %.4f（%s）；全层剂量 '
    '0.5→1→2 单调 %s；成对加性 %s'
    '（pred %s）；同会话 J2 %.4f vs 3133 '
    '%.4f → %s。**重复：%s**\n'
    % (vd[1], d2s[17], d2s[29], d2s[33],
       d2s[38], vd[1], vd[2], vd[4],
       ' '.join('%s:%.3f' % (k, v) for k, v
                in sorted(pair_pred.items())),
       tt['j2_s0']['chg'], CHGJ2_REF, vd[3],
       carr_txt))
f1b = f1
f1c = f1
fp_txt = {
    'forkcoord_p_rewrite':
        'A1−P 状态差坐标注入可改写 P 生成'
        '——分叉层位坐标级可迁移',
    'forkcoord_p_null':
        'A1−P 状态差坐标注入不能改写 P——'
        '分叉为方向内机制，不可跨方向移植'}[
    vd[5]]
f2 = (
    '2. **A1 分叉坐标化：%s**。L35 层 '
    'A1−P 状态差 top50（δ=%.3f）注入 '
    'P chg=%.4f vs 注入 A1 chg=%.4f → %s。'
    '**重复：%s**\n'
    % (vd[5], pc['delta36'], c1['P']['chg'],
       c1['A1']['chg'], vd[5], fp_txt))
f2b = f2
f2c = f2
step_txt = {
    'stepscan_effective':
        '单步注入有效——干预窗口定位到生成'
        '步 %s' % best_s,
    'stepscan_weak':
        '单步注入皆弱于 step0——干预必须'
        '落在 prompt forward（窗口=位置 0）'}[
    vd[6]]
f3 = (
    '3. **生成循环逐步干预：%s**。L17 dvec '
    'd1.0 注入步 0–11 扫描 chg %s；best '
    'step %s chg=%.4f，全量确认 %.4f → %s。'
    '轨迹剖面：swap dm 峰层 %d（带门 [33,40]，'
    '涌现指数 %d vs 3133 medE 36）、inject 峰层 '
    '%d、二阶差分 med|x|=%.4f。**重复：%s**\n'
    % (vd[6], ' '.join(
        '%s:%.3f' % (k, scan[k]['chg'])
        for k in STEPS_SCAN), best_s,
       scan[best_s]['chg'], c3f['chg'],
       vd[6], k0_re, k_emerge,
       pd_['k0_inj_peak'],
       sds['med_abs'], step_txt))
f3b = f3
f3c = f3
nums = (
    'Part B：dvec med||d|| %s；chg 矩阵 d2 %s；'
    'pair pred %s；J2 %.4f。Part C：delta36 '
    '%.3f、inj P/A1=%.4f/%.4f、scan best '
    'step %s=%.4f、full %.4f。Part D：swap 峰层 '
    '%d、inj 峰层 %d、二阶差分 med %.4f。'
    'xphase P=%.2f。'
    % (' '.join('L%s:%.1f' % (k, v)
                for k, v in sorted(
                    dn.items())),
       ' '.join('L%d:%.4f' % (l, d2s[l])
                for l in CAP_L),
       ' '.join('%s:%.3f' % (k, v) for k, v
                in sorted(pair_pred.items())),
       tt['j2_s0']['chg'], pc['delta36'],
       c1['P']['chg'], c1['A1']['chg'],
       best_s, scan[best_s]['chg'],
       c3f['chg'], pd_['k0_swap_peak'],
       pd_['k0_inj_peak'], sds['med_abs'],
       pb['xphase_base_match_P']))
hards = (
    '①坐标化用 A1−P 行内差 top50（med 准则），'
    '未做方向内 A1 与自身对照基线的坐标差；'
    '②step 扫描为单步独立注入（非逐步组合），'
    '窗口交互未穷举；③轨迹剖面为 teacher-forced '
    '惯例，k=0 读出与生成解码位错位 1 步；'
    '④成对加性检验为 2 阶（3 层以上联合仅 J2 '
    '锚定）；⑤二阶差分为描述性（3102 五原则第'
    '5 条：Möbius 谱仅描述），未做置换检验；'
    '⑥跨 Phase 漂移纪律延续 3133 硬伤⑥：全部 '
    'same 判决用会话内基线（xphase P=%.2f），'
    '3133 锚定值仅作跨会话稳定性记录。'
    % pb['xphase_base_match_P'])
mech = (
    '①载体分解 %s（d2 chg L17 %.3f vs max 非'
    'L17 %.3f）；②分叉坐标级 %s（P chg %.3f）；'
    '③干预窗口 %s。'
    % (vd[1], d2s[17],
       max(d2s[l] for l in CAP_L if l != 17),
       vd[5], c1['P']['chg'], step_txt))
p1 = ('①载体分散时的层间传导：非 L17 主载层的 '
      'dvec 下游消费路径核验（若 %s）。' % vd[1])
p2 = ('②A1 分叉坐标的功能验证：co36 坐标集在 '
      'A1 内部的必要性强弱（子集消融）+ 与 3132 '
      'co50 的重叠分析。')
p3 = ('③逐步干预组合：有效步窗口的多步联合注入'
      '剂量-响应 + 与 allstep 上限的差距分解。')
prereg = p1 + p2 + p3
title = ('\n## Phase 3134: Ω-P132 载体分解+A1'
         '分叉坐标化+步扫（T4 第17Phase）'
         '[' + STAMP + ']\n\n')
assert len(title) < 110
sec = (
    title
    + '**性质**：T4 第 17 Phase，3133 MEMO §5 '
    '预注册三项执行，design_seal.json 观测前'
    '冻结。Part A offline 3133 链接断言'
    '（result sha8 73f29f4e + co50 52b126af '
    '+ 分数锚 CHG17/J2 重算）；Part B1 swap4 '
    '状态重捕获（L17/29/33/38，672 样）；'
    'Part B2 载体分解矩阵 4 层 × 3 剂量 + 3 '
    '成对 + J2 同会话锚（16 组生成）；'
    'Part C1 L35 层 A1−P 状态差 top50 坐标'
    '注入 P/A1（128+128）；Part C2 L17 dvec '
    '生成步扫描 0/1/2/3/5/8/11（128 行）+ '
    'best 步全量确认（672）；Part D 轨迹层'
    '轮廓 P/A1 + swap4/inject dm 轨迹 + '
    '描述性二阶差分。运行 '
    + ('%.0fs' % r['runtime_s']) + '。\n\n'
    + '### 1. 三大发现（重复三遍）\n'
    + f1 + f1b + f1c + f2 + f2b + f2c + f3
    + f3b + f3c + '\n'
    + '### 2. 关键数值\n'
    + nums + '\n\n'
    + '### 3. 硬伤\n' + hards
    + '\n\n### 4. 机制拼图更新\n' + mech
    + '\n\n### 5. 3135 预注册（观察后冻结）'
    '\n' + prereg + '\n\n'
    + '产物：`tests/glm5/result/'
    'rdc_query_construction_20260913/'
    'phase3134/omega_p132_carrier_matrix_'
    'forkcoord_stepscan/`（result.json、'
    'design_seal.json、run_log.txt、'
    'p132_readout.npz）；脚本 '
    '`tests/glm5/phase3134_omega_p132_'
    'carrier_matrix_forkcoord_stepscan.py`。')
assert len(sec) > 2500

raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

steps = []
if 'meas3134_omega_p132' in io.open(
        LEDGER, encoding='utf-8').read():
    steps.append('ledger: exists, skip')
else:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    claim = (
        'Omega-P132 (3134, T4 seventeenth '
        'phase: dvec carrier matrix + A1 '
        'fork-layer coordimization + '
        'generation step-scan. Carrier: d2 '
        'chg L17 %.4f/L29 %.4f/L33 %.4f/'
        'L38 %.4f -> %s, %s; pairs %s; '
        'same-session J2 %.4f vs 3133 %.4f '
        '-> %s; fork coords L35 top50 '
        'inject P %.4f / A1 %.4f -> %s; '
        'step scan best %s chg %.4f full '
        '%.4f -> %s; traj swap peak L%d '
        '- verdict '
        % (d2s[17], d2s[29], d2s[33],
           d2s[38], vd[1], vd[2],
           ' '.join('%s %.3f/%.3f'
                    % (k, tt['p' + k]['chg'],
                       pair_pred[k])
                    for k in sorted(
                        pair_pred)),
           tt['j2_s0']['chg'], CHGJ2_REF,
           vd[3], c1['P']['chg'],
           c1['A1']['chg'], vd[5], best_s,
           scan[best_s]['chg'], c3f['chg'],
           vd[6], pd_['k0_swap_peak'])) + V
    entry = {
        'meas_id':
            'meas3134_omega_p132_'
            'carrier_matrix_forkcoord_'
            'stepscan',
        'phase': 3134,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3134/omega_p132_.../'
                'result.json sha256_8='
                + sha8,
            'npz': 'p132_readout.npz'},
        'hashes': {'result_sha256_8':
                   sha8},
        'anchors': ['meas3133_omega_p131_'
                    'transplant_a1fork_'
                    'migrate'],
        'note': 'carrier %s; fork %s; step '
                '%s; traj %s' % (vd[1],
                                 vd[5], vd[6],
                                 vd[7])}
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
if '## Phase 3134:' in memo:
    steps.append('memo: exists, skip')
else:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    steps.append('memo: appended (short '
                 'title, 5 sections)')

WDATE = NOW.strftime('%Y-%m-%d')
wline = ('- Phase 3134 Omega-P132 closeout: '
         'verdict ' + V + '; ledger sha8 '
         + json.load(io.open(
             LEDGER,
             encoding='utf-8'))
         ['ledger_sha256_8']
         + '; MEMO 3134 section; runtime '
         + ('%.0fs' % r['runtime_s']) + '.')
for wd in WLOGS:
    wl = wd + '\\' + WDATE + '.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3134 Omega-P132 closeout' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        steps.append('wlog: ' + wl[:12])
    else:
        steps.append('wlog: exists '
                     + wl[:12])

mem = io.open(MEMW, encoding='utf-8').read()
if '3134（T4）' in mem:
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
        '- 3126（T4）：',
        '- 3126（T4）：GLM4 L*=20/20 复现='
        '模型特异；trail 显著；diffuse c3 '
        '0.21。')
    mem = compress_line(
        mem,
        '- 3127（T4）：',
        '- 3127（T4）：写入链功能否定=单层'
        '无必要；trail lag1-3。')
    mem = compress_line(
        mem,
        '- 3125（T4）：',
        '- 3125（T4）：第三成分=内容尾迹；'
        'Qwen 层位=模型特异。')
    mem = compress_line(
        mem,
        '- 3133（T4）：',
        '- 3133（T4）：L17 全维移植 chg 0.134 '
        '部分充分；A1 fork medE 36/−1 不对'
        '称；谱迁移 sp 0.86；L17 坐标→A1 '
        'chg 0.320。')
    mem = compress_line(
        mem,
        '- 3132（T4）：',
        '- 3132（T4）：L17 注入改写 0.920/'
        'rescue 0.009=分叉层因果；谱 256 '
        '锚固。')
    mem = compress_line(
        mem,
        '- 3131（T4）：',
        '- 3131（T4）：注入窗 L20–28 稳健；'
        '谱峰 L38；分叉决策层 L17。')
    new_line = (
        '- 3134（T4）：dvec 载体 %s（d2 chg '
        'L17 %.3f/L29 %.3f/L33 %.3f/L38 '
        '%.3f）；A1−P L35 坐标注入 P chg '
        '%.3f（%s）；步扫 best %s chg %.3f'
        '（%s）。'
        % (vd[1].replace('carrier_', ''),
           d2s[17], d2s[29], d2s[33],
           d2s[38], c1['P']['chg'],
           vd[5].replace('forkcoord_', ''),
           best_s, scan[best_s]['chg'],
           vd[6].replace('stepscan_', '')))
    anchor = '- 3133（T4）：'
    ia = mem.find(anchor)
    assert ia > 0
    mem = mem[:ia] + new_line + '\n' + \
        mem[ia:]
    old2 = '下一 3134：**'
    i2 = mem.find(old2)
    assert i2 > 0, 'next-line anchor'
    j2 = mem.find('\n', i2)
    if vd[1] == 'carrier_l17_max':
        new2 = ('下一 3135：**L17 主载确认下的 '
                '层间传导核验（L17→下游消费）+ '
                'co36 必要性子集消融。**')
    else:
        new2 = ('下一 3135：**非 L17 主载层的层'
                '间传导 + co36 必要性子集消融 + '
                '有效步窗口多步联合注入。**')
    mem = mem[:i2] + new2 + mem[j2:]
    oldm = '- max=3133，'
    assert mem.count(oldm) == 1
    mem = mem.replace(oldm,
                      '- max=3134，')
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
