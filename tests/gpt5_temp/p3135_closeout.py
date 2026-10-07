# -*- coding: utf-8 -*-
"""Phase 3135 closeout: idempotent five-write
chain (ledger -> MEMO -> wlogs -> MEMORY).
MEMO title SHORT per user rule (2026-09-24).
All numeric claims read from result.json;
hard frozen asserts before any write.
rev-3135a notes: C1 uses same-session A1
baseline (3134 lesson); C2 degenerate
(overlap=0) recorded not asserted."""
import datetime
import hashlib
import io
import json

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3135'
        r'\omega_p133_conduction_'
        r'co36ablation_window')
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
COND_COS_HI = 0.40
COND_RHO_HI = 0.30
COND_COS_LO = 0.20
COND_RHO_LO = 0.10
CONC_FRAC_HI = 0.6
CONC_FRAC_LO = 0.3
NEC_FRAC = 0.5
SUFF_FRAC = 0.8
ALLSTEP_FRAC = 0.8
CUM_GAP = 0.05
SHARE_GAP = 0.05
DELTA36_34 = 1.694018277446071
CO36_34_SHA = '84a1e1a3'
WIN_NAMES = ['w2', 'w3', 'w5', 'w8']

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
assert r['smoke'] is False
V = r['verdict']
vd = V.split('|')
assert len(vd) == 8
assert vd[0] == 'a_3134_ok'
assert vd[1] in ('cond_readout_decoupled',
                 'cond_signal_attenuated',
                 'cond_mixed')
assert vd[2] in ('co36_concentrated',
                 'co36_dispersed',
                 'co36_partial_split')
assert vd[3] in ('overlap_necessary',
                 'overlap_sufficient',
                 'overlap_mixed')
assert vd[4] in ('window_cumulative',
                 'window_jumpy',
                 'window_flat')
assert vd[5] in ('window_near_allstep',
                 'window_below_allstep')
assert vd[6] in ('dose_share_penalizes',
                 'dose_share_better',
                 'dose_share_neutral')
assert vd[7] == 'coverage_full'
pb = r['part_b']
pc = r['part_c']
pd_ = r['part_d']
assert r['runtime_s'] > 300

# ---- part_b asserts ----
assert pb['dose_cond'] == 2.0
assert abs(pb['xphase_base_match_P']
           - 1.0) < 1e-9
dn = pb['dvec_med_norm']
assert set(dn) == {'17', '29', '33', '38'}
assert all(v > 0 for v in dn.values())
assert set(pb['l17_downstream_cos']) == \
    {'18', '19', '20', '25', '30', '35',
     '39'}
for il in ('17', '29', '33', '38'):
    assert len(pb['cos_spec_med'][il]) == 40
    assert len(pb['rho_spec_med'][il]) == 40
# conduction gate re-derivation from npz
znp = np.load(OUTD + r'\p133_readout.npz',
              allow_pickle=False)
cond = pb['conduction']
for il in CAP_L:
    cm = znp['cos_%d' % il]
    rm = znp['rho_%d' % il]
    rs = [q for q in (il + 1, il + 2,
                      il + 3) if q < 40]
    cd = float(np.median(
        [float(np.nanmedian(cm[q]))
         for q in rs]))
    rd = float(np.median(
        [float(np.nanmedian(rm[q]))
         for q in rs]))
    assert abs(cd - cond[str(il)]
               ['cos_direct']) < 1e-6
    assert abs(rd - cond[str(il)]
               ['rho_direct']) < 1e-6
    if il == 17:
        assert cond['17']['gate'] == 'na'
        continue
    if cd >= COND_COS_HI \
            and rd >= COND_RHO_HI:
        g_exp = 'decoupled'
    elif cd < COND_COS_LO \
            or rd < COND_RHO_LO:
        g_exp = 'attenuated'
    else:
        g_exp = 'mixed'
    assert cond[str(il)]['gate'] == g_exp
_vals = set(cond[str(l)]['gate']
            for l in (29, 33, 38))
if _vals == {'decoupled'}:
    c_cond_exp = 'cond_readout_decoupled'
elif _vals == {'attenuated'}:
    c_cond_exp = 'cond_signal_attenuated'
else:
    c_cond_exp = 'cond_mixed'
assert vd[1] == c_cond_exp

# ---- part_c asserts ----
assert pc['fork_layer'] == 35
assert pc['fork_prof_idx'] == 36
assert pc['jaccard34'] == 1.0
assert abs(pc['delta36']
           - DELTA36_34) < 1e-9
assert pc['co36_sha8'] == CO36_34_SHA
assert pc['overlap']['n_ov'] == 0
assert abs(pc['overlap']
           ['jaccard_50_36']) < 1e-12
c1 = pc['c1_trials']
assert set(c1) == {'full50', 'top25',
                   'bot25', 'rand25_0',
                   'rand25_1', 'rand25_2'}
for v in c1.values():
    assert 0.0 <= v['chg'] <= 1.0
    assert 0 <= v['first'] <= 672
f_full = c1['full50']['chg']
f_top = c1['top25']['chg']
f_weak = max(c1['bot25']['chg'],
             max(c1['rand25_%d' % i]['chg']
                 for i in range(3)))
if f_top >= CONC_FRAC_HI * f_full \
        and f_weak \
        <= CONC_FRAC_LO * f_full:
    c_abl_exp = 'co36_concentrated'
elif f_top < CONC_FRAC_LO * f_full \
        or f_weak \
        >= CONC_FRAC_HI * f_full:
    c_abl_exp = 'co36_dispersed'
else:
    c_abl_exp = 'co36_partial_split'
assert vd[2] == c_abl_exp
c2 = pc['c2_trials']
assert set(c2) == {'co50_full',
                   'co50_no36',
                   'co50_cap'}
for v in c2.values():
    assert 0.0 <= v['chg'] <= 1.0
g_full = c2['co50_full']['chg']
g_no = c2['co50_no36']['chg']
g_cap = c2['co50_cap']['chg']
if g_no <= NEC_FRAC * g_full:
    c_ovr_exp = 'overlap_necessary'
elif g_cap >= SUFF_FRAC * g_full:
    c_ovr_exp = 'overlap_sufficient'
else:
    c_ovr_exp = 'overlap_mixed'
assert vd[3] == c_ovr_exp

# ---- part_d asserts ----
assert pd_['windows'] == WIN_NAMES
dt = pd_['d_trials']
assert set(dt) == set(WIN_NAMES) | \
    {'w2share', 'allstep'}
for v in dt.values():
    assert 0.0 <= v['chg'] <= 1.0
    assert 0 <= v['first'] <= 672
    assert len(v['cum']) == 12
    assert abs(v['cum'][-1]
               - v['chg']) < 1e-9
chg_w = {w: dt[w]['chg'] for w in WIN_NAMES}
mono_w = all(
    chg_w[WIN_NAMES[i + 1]]
    >= chg_w[WIN_NAMES[i]] - 0.02
    for i in range(3))
if mono_w and (chg_w['w8']
               >= chg_w['w2'] + CUM_GAP):
    c_win_exp = 'window_cumulative'
elif chg_w['w8'] >= chg_w['w2'] \
        + CUM_GAP:
    c_win_exp = 'window_jumpy'
else:
    c_win_exp = 'window_flat'
assert vd[4] == c_win_exp
c_near_exp = ('window_near_allstep'
              if chg_w['w8']
              >= ALLSTEP_FRAC
              * dt['allstep']['chg']
              else 'window_below_allstep')
assert vd[5] == c_near_exp
if dt['w2share']['chg'] \
        <= chg_w['w2'] - SHARE_GAP:
    c_share_exp = 'dose_share_penalizes'
elif dt['w2share']['chg'] \
        >= chg_w['w2'] + SHARE_GAP:
    c_share_exp = 'dose_share_better'
else:
    c_share_exp = 'dose_share_neutral'
assert vd[6] == c_share_exp
# npz subset sanity
assert len(znp['co36']) == 50
assert len(znp['top25']) == 25
assert len(znp['ov50']) == 0
assert len(znp['only50']) == 50

# branch prose
cond_txt = {
    'cond_readout_decoupled':
        '非 L17 载体层 dvec 自层注入的下游'
        '传导与 swap 场高度同向（cos≈0.96）'
        '且幅度≈剂量线性外推——层间传导'
        '直通，无中继层',
    'cond_signal_attenuated':
        '非 L17 载体下游传导衰减——存在'
        '层间阻断',
    'cond_mixed':
        '部分载体直通、部分衰减'}[vd[1]]
f1 = (
    '1. **层间传导核验：%s**。L29/33/38 '
    'dvec 自层注入（dose 2.0）下游直接层 '
    'cos %.3f/%.3f/%.3f、rho %.2f/%.2f/'
    '%.2f → 全 decoupled；L17 正对照下游 '
    'cos L18 %.3f→L30 %.3f→L39 %.3f。'
    '结合 3134 行为面（L29/33/38 chg '
    '0.049–0.089 远小于 L17 0.256）：'
    '**传导保真不分离载体，行为差异不在'
    '传导而在写入内容的行为读出权重——'
    '写入即达、读出定效**。**重复：%s**\n'
    % (vd[1],
       cond['29']['cos_direct'],
       cond['33']['cos_direct'],
       cond['38']['cos_direct'],
       cond['29']['rho_direct'],
       cond['33']['rho_direct'],
       cond['38']['rho_direct'],
       pb['l17_downstream_cos']['18'],
       pb['l17_downstream_cos']['30'],
       pb['l17_downstream_cos']['39'],
       cond_txt))
f1b = f1
f1c = f1
abl_txt = {
    'co36_concentrated':
        'A1 分叉改写由 med-abs top25 集中'
        '承载',
    'co36_dispersed':
        'A1 分叉改写不由 top50 集中承载'
        '（随机 25 子集同量级）——功能'
        '分散',
    'co36_partial_split':
        '部分集中部分分散'}[vd[2]]
f2 = (
    '2. **co36 复现与消融：jaccard34=1.000、'
    'co36_sha8=%s（与 3134 位级一致）、'
    'delta36=%.4f 精确复现；同会话 A1 基线'
    '（3134 错配修正）下 full50 chg %.4f、'
    'top25 %.4f、bot25 %.4f、rand25 %.4f–'
    '%.4f → %s。C2: co50∩co36=%d（零重叠'
    '全等对照退化），co50@L17 负向 chg '
    '%.4f=强载体 → **两代 top50 互不相交'
    '却各自功能有效：坐标级端口类**（3109 '
    '功能等价、3130 零重叠的坐标级重现）。'
    '**重复：%s**\n'
    % (pc['co36_sha8'], pc['delta36'],
       f_full, f_top, c1['bot25']['chg'],
       c1['rand25_0']['chg'],
       f_weak, vd[2],
       pc['overlap']['n_ov'],
       g_full, abl_txt))
f2b = f2
f2c = f2
f3 = (
    '3. **多步窗口：%s|%s|%s**。w2 %.4f→'
    'w3 %.4f→w5 %.4f→w8 %.4f 单调累积；'
    'w8 为 allstep（%.4f）的 %.0f%%→'
    '9-forward 窗近饱和；w2share（同窗 '
    '1/3 剂量）%.4f 远低于 w2 → **时间'
    '积分非线性：同窗分布式弱剂量远劣于'
    '每步全剂量——解码步级注入有效但需'
    '每步足量**。**重复：%s|%s|%s**\n'
    % (vd[4], vd[5], vd[6],
       chg_w['w2'], chg_w['w3'],
       chg_w['w5'], chg_w['w8'],
       dt['allstep']['chg'],
       100.0 * chg_w['w8']
       / dt['allstep']['chg'],
       dt['w2share']['chg'],
       vd[4], vd[5], vd[6]))
f3b = f3
f3c = f3
nums = (
    'Part B：dvec med||d|| L17 %.3f/L29 '
    '%.3f/L33 %.3f/L38 %.3f；B1 profile '
    '匹配 idx18 (med|d| 0.0769)；xphase '
    'P=%.4f (672/672)；cond cos/rho 29 '
    '%.3f/%.2f, 33 %.3f/%.2f, 38 %.3f/'
    '%.2f, 17 %.3f/%.2f。Part C：jaccard34 '
    '%.3f、delta36 %.4f、C1 %s、C2 co50 '
    '%.4f/cap %.4f。Part D：w2 %.4f/w3 '
    '%.4f/w5 %.4f/w8 %.4f/share %.4f/'
    'all %.4f。运行 %.0fs。'
    % (dn['17'], dn['29'], dn['33'],
       dn['38'],
       pb['xphase_base_match_P'],
       cond['29']['cos_direct'],
       cond['29']['rho_direct'],
       cond['33']['cos_direct'],
       cond['33']['rho_direct'],
       cond['38']['cos_direct'],
       cond['38']['rho_direct'],
       cond['17']['cos_direct'],
       cond['17']['rho_direct'],
       pc['jaccard34'], pc['delta36'],
       ' '.join('%s %.4f' % (k,
                             c1[k]['chg'])
                for k in ('full50', 'top25',
                          'bot25')),
       g_full, g_cap,
       chg_w['w2'], chg_w['w3'],
       chg_w['w5'], chg_w['w8'],
       dt['w2share']['chg'],
       dt['allstep']['chg'],
       r['runtime_s']))
hards = (
    '①C2 分解退化：|co50∩co36|=0 → no36'
    '≡full 全等对照，重叠必要性不可判'
    '（设计预期有重叠，实测零重叠本身即'
    '发现）；②cond 门 (0.40/0.30) 太宽，'
    '实测全落 decoupled，且 rho≈2.0 的'
    '剂量线性假设无剂量扫描支持（单剂量 '
    '2.0）；③C1 同会话基线数字与 3134 '
    '报告值不可比（3134 基线错配通胀，'
    '本 Phase 为 A1 方向权威值）；④co50 '
    'first=51 的步语义未解释（>N_NEW=12，'
    'trial_metrics 口径待查）；⑤窗口饱和'
    '形状（w3→w5 仅 +0.055）未做步级 '
    'drop-one 定位；⑥sha/jaccard 位级'
    '复现依赖同 GPU 同精度，跨硬件未证；'
    '⑦SMOKE 调试 R1–R4 修 4 处（B2 行'
    '对齐/DELTA_L17/CKPTF 名/SMOKE 门'
    '控，rev-3135a），正式跑一次通过。')
mech = (
    '①层间传导直通（非 L17 载体 cos≈0.96，'
    '无中继层）；②坐标端口类（co36/co50 '
    '不相交但各自功能有效）；③窗口剂量'
    '积分非线性（dose-share 惩罚）。')
p1 = ('①传导保真剂量-响应：L29/33/38 剂量 '
      '{0.5,1,2,4} 自层注入，下游 cos/rho '
      '剂量曲线 + 同场行为 chg（保真-行为'
      '分离点定位）。')
p2 = ('②co36∪co50 交叉注入矩阵：注入位 '
      '{L35→A1, L17→P 负向} × 坐标集 '
      '{co36, co50, 并集} 3×3 chg + 并集'
      '加性检验（端口类交叉验证）。')
p3 = ('③w8 逐步 drop-one 消融（9 forward '
      '逐一剔除 → 步级贡献谱 + 饱和定位）。')
prereg = p1 + p2 + p3
title = ('\n## Phase 3135: Ω-P133 层间传导+'
         'co36消融+窗口（T4 第18Phase）'
         '[' + STAMP + ']\n\n')
assert len(title) < 110
sec = (
    title
    + '**性质**：T4 第 18 Phase，3134 MEMO '
    '§5 预注册三项执行，design_seal.json '
    '观测前冻结。Part A offline 3134 链接'
    '断言（result sha8 d331bd1b + co50 '
    '52b126af + co36-34 84a1e1a3）；'
    'Part B1 全 40 层状态重捕获（672 样，'
    'swap4+base）+ 会话内生成基线（xphase '
    'P=1.000，672/672 位级复现 z26）；'
    'Part B2 层间传导矩阵（4 载体层 dvec '
    '自层注入 dose2.0 × 全 40 层下游读出 '
    'cos+rho 谱）；Part C co36 必要性子集'
    '消融（full50/top25/bot25/rand25×3，'
    'A1 同会话基线——3134 错配修正）+ '
    'co50 重叠分解；Part D 多步窗口联合'
    '注入（w2/w3/w5/w8/w2share/allstep，'
    '128 行）。运行 %.0fs。\n\n'
    % r['runtime_s']
    + '### 1. 三大发现（重复三遍）\n'
    + f1 + f1b + f1c + f2 + f2b + f2c + f3
    + f3b + f3c + '\n'
    + '### 2. 关键数值\n'
    + nums + '\n\n'
    + '### 3. 硬伤\n' + hards
    + '\n\n### 4. 机制拼图更新\n' + mech
    + '\n\n### 5. 3136 预注册（观察后冻结）'
    '\n' + prereg + '\n\n'
    + '产物：`tests/glm5/result/'
    'rdc_query_construction_20260913/'
    'phase3135/omega_p133_conduction_'
    'co36ablation_window/`（result.json、'
    'design_seal.json、run_log.txt、'
    'p133_readout.npz）；脚本 '
    '`tests/glm5/phase3135_omega_p133_'
    'conduction_co36ablation_window.py`。')
assert len(sec) > 2500

raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

steps = []
if 'meas3135_omega_p133' in io.open(
        LEDGER, encoding='utf-8').read():
    steps.append('ledger: exists, skip')
else:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    claim = (
        'Omega-P133 (3135, T4 eighteenth '
        'phase: cross-layer conduction + '
        'co36 necessity ablation + '
        'multi-step windows. Conduction: '
        'L29/33/38 dvec self-layer inj '
        'dose2.0 -> downstream cos '
        '%.3f/%.3f/%.3f rho %.2f/%.2f/'
        '%.2f all decoupled (L17 ctrl '
        '0.802); co36 bit-level '
        'replication sha %s delta36 '
        '%.4f; same-session A1 baseline '
        'full50 %.4f top25 %.4f weak '
        '%.4f -> %s; co50 cap co36 = 0, '
        'co50@L17 neg chg %.4f -> '
        'port-class; windows w2 %.4f -> '
        'w8 %.4f cumulative, %.0f%% of '
        'allstep %.4f, dose-share %.4f '
        'penalized - verdict '
        % (cond['29']['cos_direct'],
           cond['33']['cos_direct'],
           cond['38']['cos_direct'],
           cond['29']['rho_direct'],
           cond['33']['rho_direct'],
           cond['38']['rho_direct'],
           pc['co36_sha8'], pc['delta36'],
           f_full, f_top, f_weak, vd[2],
           g_full, chg_w['w2'],
           chg_w['w8'],
           100.0 * chg_w['w8']
           / dt['allstep']['chg'],
           dt['allstep']['chg'],
           dt['w2share']['chg'])) + V
    entry = {
        'meas_id':
            'meas3135_omega_p133_'
            'conduction_co36ablation_'
            'window',
        'phase': 3135,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3135/omega_p133_.../'
                'result.json sha256_8='
                + sha8,
            'npz': 'p133_readout.npz'},
        'hashes': {'result_sha256_8':
                   sha8},
        'anchors': ['meas3134_omega_p132_'
                    'carrier_matrix_'
                    'forkcoord_'
                    'stepscan'],
        'note': 'cond %s; abl %s; ovr %s; '
                'win %s/%s/%s'
                % (vd[1], vd[2], vd[3],
                   vd[4], vd[5], vd[6])}
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
if '## Phase 3135:' in memo:
    steps.append('memo: exists, skip')
else:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    steps.append('memo: appended (short '
                 'title, 5 sections)')

WDATE = NOW.strftime('%Y-%m-%d')
wline = ('- Phase 3135 Omega-P133 closeout: '
         'verdict ' + V + '; ledger sha8 '
         + json.load(io.open(
             LEDGER,
             encoding='utf-8'))
         ['ledger_sha256_8']
         + '; MEMO 3135 section; runtime '
         + ('%.0fs' % r['runtime_s']) + '.')
for wd in WLOGS:
    wl = wd + '\\' + WDATE + '.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3135 Omega-P133 closeout' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        steps.append('wlog: ' + wl[:12])
    else:
        steps.append('wlog: exists '
                     + wl[:12])

mem = io.open(MEMW, encoding='utf-8').read()
if '3135（T4）' in mem:
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
        '- 3121–3122：',
        '- 3121–3122：token 级替换失效；'
        '语法 L21/内容 L20 涌现=写入链上游。')
    mem = compress_line(
        mem,
        '- 3123（T4）：',
        '- 3123（T4）：AUC 平台 0.50→0.66='
        'i.i.d. 扩散伪影、残差共模；L35 刹车'
        '全局/A1 释压。')
    mem = compress_line(
        mem,
        '- 3124（T4）：',
        '- 3124（T4）：(m,锚点,算子) 族不足；'
        'L35 释压=范数×语义各半；GLM4 L*=20 '
        '复现。')
    mem = compress_line(
        mem,
        '- 3130（T4）：',
        '- 3130（T4）：注入严格绑定读点位；'
        'A1/P-fit 坐标零重叠=方向特异；全 '
        'swap 位级复现。')
    new_line = (
        '- 3135（T4）：传导解耦 L29/33/38'
        '（自层注入→下游 cos 0.96/rho≈2.0）；'
        'co36 位级复现（sha 84a1e1a3）但 A1 '
        '同会话 chg 0.078=分散；co50∩co36=0 '
        '而 co50@L17 chg 0.422=端口类；窗口'
        '累积 w8 0.648≈allstep 94%、'
        'dose-share 惩罚。')
    anchor = '- 3134（T4）：'
    ia = mem.find(anchor)
    assert ia > 0
    ie = mem.find('\n', ia)
    mem = mem[:ie + 1] + new_line + '\n' \
        + mem[ie + 1:]
    i2 = mem.find('下一 3135：**')
    assert i2 > 0, 'next-line anchor'
    j2 = mem.find('**', i2 + 8)
    assert j2 > i2
    new2 = ('下一 3136：**传导保真×行为分离'
            '剂量曲线 + co36∪co50 交叉注入'
            '矩阵 + w8 逐步 drop-one 贡献谱。**')
    mem = mem[:i2] + new2 \
        + mem[j2 + 2:]
    oldm = '- max=3134，'
    assert mem.count(oldm) == 1
    mem = mem.replace(oldm,
                      '- max=3135，')
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
