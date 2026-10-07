# -*- coding: utf-8 -*-
"""Phase 3133 closeout: idempotent five-write
chain (ledger -> MEMO -> wlogs -> MEMORY).
MEMO title SHORT per user rule (2026-09-24).
All numeric claims read from result.json;
hard frozen asserts before any write.
Verdict-dependent prose via branch fns."""
import datetime
import hashlib
import io
import json

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3133'
        r'\omega_p131_transplant_'
        r'a1fork_migrate')
D32 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3132'
       r'\omega_p130_forkcausal_single256')
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

REF32_CHG256 = {4: 0.96875, 8: 0.7890625,
                9: 0.76171875, 13: 0.4921875,
                17: 0.8828125, 29: 0.1875,
                33: 0.1640625, 38: 0.9921875}
LAYERS_C = [4, 8, 9, 13, 17, 29, 33, 38]
TR_FULL = 0.90
TR_PART = 0.10
ADD_TOL = 0.15
DOSE_SLACK = 0.02
RHO_PROF = 0.70
RHO_TRANS = 0.70
MED_E_TOL = 3

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
assert r['smoke'] is False
V = r['verdict']
vd = V.split('|')
assert len(vd) == 10
assert vd[0] == 'a_3132_ok'
assert vd[1] in ('transplant_l17_full',
                 'transplant_l17_partial',
                 'transplant_l17_null')
assert vd[2] in ('dose_monotone',
                 'dose_nonmonotone')
assert vd[3] in ('l38_transplant_lower',
                 'l38_transplant_comparable')
assert vd[4] in ('joint_additive',
                 'joint_superadditive',
                 'joint_subadditive')
assert vd[5] in ('step_transplant_'
                 'allstep_higher',
                 'step_transplant_'
                 'equivalent')
assert vd[6] in ('a1_forklayer_symmetric',
                 'a1_forklayer_asymmetric')
assert vd[7] in ('a1_transfer_high',
                 'a1_transfer_low')
assert vd[8] in ('a1_l17coord_rewrites',
                 'a1_l17coord_null')
assert vd[9] == 'coverage_full'
pb = r['part_b']
pc = r['part_c']

# ---- part_b asserts ----
dn = pb['dvec_med_norm']
assert set(dn) == {'17', '29', '33', '38'}
assert all(v > 0 for v in dn.values())
assert 0.0 <= pb['dvec17_co50_energy'] \
    <= 1.0
TR = ['t17_s0_d05', 't17_s0_d10',
      't17_s0_d20', 't17_as_d10',
      't38_s0_d10', 'tJ1_s0', 'tJ2_s0']
tt = pb['trials']
assert set(tt) == set(TR)
for k, v in tt.items():
    assert 0.0 <= v['chg'] <= 1.0
    assert 0 <= v['first'] <= 672
    assert len(v['cum']) == 12
    assert abs(v['cum'][-1] - v['chg']) \
        < 1e-9
chg17 = pb['chg17_s0_d10']
chg17_05 = pb['chg17_s0_d05']
chg17_20 = pb['chg17_s0_d20']
chg17_as = pb['chg17_as_d10']
chg38 = pb['chg38_s0_d10']
chgJ1 = pb['chgJ1_s0']
chgJ2 = pb['chgJ2_s0']
assert abs(chg17
           - tt['t17_s0_d10']['chg']) < 1e-12
assert abs(chgJ1 - tt['tJ1_s0']['chg']) \
    < 1e-12
predJ1 = 1.0 - (1.0 - chg17) \
    * (1.0 - chg38)
assert abs(predJ1 - pb['predJ1_indep']) \
    < 1e-12
if chg17 >= TR_FULL:
    b_trans_exp = 'transplant_l17_full'
elif chg17 >= TR_PART:
    b_trans_exp = 'transplant_l17_partial'
else:
    b_trans_exp = 'transplant_l17_null'
assert vd[1] == b_trans_exp
dose_ok = (chg17_05 <= chg17 + DOSE_SLACK
           and chg17 <= chg17_20
           + DOSE_SLACK)
assert vd[2] == ('dose_monotone'
                 if dose_ok
                 else 'dose_nonmonotone')
assert vd[3] == ('l38_transplant_lower'
                 if chg38 < 0.9
                 * max(chg17, 1e-9)
                 else 'l38_transplant_'
                      'comparable')
if abs(chgJ1 - predJ1) <= ADD_TOL:
    joint_exp = 'joint_additive'
elif chgJ1 > predJ1:
    joint_exp = 'joint_superadditive'
else:
    joint_exp = 'joint_subadditive'
assert vd[4] == joint_exp
assert vd[5] == ('step_transplant_'
                 'allstep_higher'
                 if chg17_as - chg17 > 0.10
                 else 'step_transplant_'
                      'equivalent')

# ---- part_c asserts ----
assert len(pc['tf_idx']) == 128
assert len(pc['tf_idx']) == len(
    set(pc['tf_idx']))
assert -1.0 <= pc['p31_link_spearman'] \
    <= 1.0
assert -1.0 <= pc['sp_prof'] <= 1.0
assert -1.0 <= pc['sp_spec'] <= 1.0
for k in ('med_dm0_A1', 'med_dm0_P'):
    arr = pc[k]
    assert len(arr) == 41
    assert all(abs(v) < 1e6 for v in arr)
fdA1 = pc['first_div_A1']
fdP = pc['first_div_P']
assert len(fdA1) == len(fdP) == 128
# replicate main-script medE exactly:
# int(np.median(int32 array)) truncation
_a1pos = np.array([x for x in fdA1
                   if x >= 0],
                  dtype=np.int32)
_ppos = np.array([x for x in fdP
                  if x >= 0],
                 dtype=np.int32)
medE_A1_re = (int(np.median(_a1pos))
              if _a1pos.size else -1)
medE_P_re = (int(np.median(_ppos))
             if _ppos.size else -1)
assert pc['medE_A1'] == medE_A1_re
assert pc['medE_P'] == medE_P_re
c2a = pc['chgA1']
assert set(c2a) == {str(l)
                    for l in LAYERS_C}
assert all(0.0 <= c2a[str(l)] <= 1.0
           for l in LAYERS_C)
fa = pc['firstA1']
assert all(0 <= fa[str(l)] <= 256
           for l in LAYERS_C)
assert pc['sel_sha8'] == 'e34d2588'


def rankv(x):
    return np.argsort(
        np.argsort(np.asarray(
            x, dtype=np.float64)))


sp_prof_re = float(np.corrcoef(
    rankv(pc['med_dm0_A1']),
    rankv(pc['med_dm0_P']))[0, 1])
assert abs(sp_prof_re - pc['sp_prof']) \
    < 1e-6
vecA1 = [c2a[str(l)] for l in LAYERS_C]
vecP = [REF32_CHG256[l] for l in LAYERS_C]
sp_spec_re = float(np.corrcoef(
    rankv(vecA1), rankv(vecP))[0, 1])
assert abs(sp_spec_re - pc['sp_spec']) \
    < 1e-6
sym_exp = ('a1_forklayer_symmetric'
           if pc['sp_prof'] >= RHO_PROF
           and abs(pc['medE_A1']
                   - pc['medE_P'])
           <= MED_E_TOL
           else 'a1_forklayer_asymmetric')
assert vd[6] == sym_exp
assert vd[7] == ('a1_transfer_high'
                 if pc['sp_spec']
                 >= RHO_TRANS
                 else 'a1_transfer_low')
c3 = pc['c3_trials']
assert set(c3) == {'sgn+1', 'sgn-1'}
for v in c3.values():
    assert 0.0 <= v['chg'] <= 1.0
c3_best = max(v['chg'] for v in
              c3.values())
assert vd[8] == ('a1_l17coord_rewrites'
                 if c3_best >= TR_PART
                 else 'a1_l17coord_null')
z32 = np.load(D32 + r'\p130_readout.npz',
              allow_pickle=False)
row_l2 = np.linalg.norm(
    z32['hout_l17'][:672].astype(
        np.float64), axis=1)
delta_re = 0.10 * float(row_l2.mean())
assert abs(pc['delta_l17'] - delta_re) \
    < 1e-3
assert r['runtime_s'] > 2000

# branch prose
f1 = (
    '1. **L17 全维移植判别：%s**。'
    'dvec=h_swap4−h_base（med||d|| L17 %.2f'
    '/L29 %.2f/L33 %.2f/L38 %.2f；L17 能量 '
    'co50 份额 %.3f）。移植剂量 0.5/1/2 → '
    'chg %.4f/%.4f/%.4f（%s）；allstep '
    '%.4f（%s）；L38 移植 %.4f（%s）；'
    '联合 J1 {17,38}=%.4f vs 独立预测 '
    '%.4f → %s，J2 {17,29,33,38}=%.4f。'
    '**重复：%s**\n'
    % (vd[1], dn['17'], dn['29'], dn['33'],
       dn['38'], pb['dvec17_co50_energy'],
       chg17_05, chg17, chg17_20, vd[2],
       chg17_as, vd[5], chg38, vd[3],
       chgJ1, predJ1, vd[4], chgJ2,
       {'transplant_l17_full':
        'L17 状态级充分——swap 决策由 '
        'L17 状态承载，3132 相关=中介',
        'transplant_l17_partial':
        'L17 贡献真实但部分——写入链'
        '分散承载，相关=部分中介',
        'transplant_l17_null':
        '状态级移植无效——3132 注入'
        '改写非 L17 状态承载'}[vd[1]]))
f2 = (
    '2. **A1 首位分叉层位剖面：%s**。'
    'tf swap4 first-divergence 层位 '
    'medE A1=%d vs P=%d（容 %d 层）；'
    'med|dm_k0| 剖面 Spearman %.3f；'
    'first_div>=0 计数 A1 %d/128 vs P '
    '%d/128。结合 3131 first 614 vs 178 '
    '不对称。**重复：%s**\n'
    % (vd[6], pc['medE_A1'], pc['medE_P'],
       MED_E_TOL, pc['sp_prof'],
       sum(1 for x in fdA1 if x >= 0),
       sum(1 for x in fdP if x >= 0),
       ('两方向 fork 决策同层位成形——'
        '不对称来自幅度/坐标而非深度'
        if vd[6] == 'a1_forklayer_symmetric'
        else 'A1 首位分叉在不同深度成形'
             '——层位级不对称实锤')))
f3 = (
    '3. **写入链谱 A1 迁移：Spearman '
    '%.3f → %s**。chgA1 %s vs P chg256 '
    '%s；firstA1 %s；C3 L17 坐标注入 '
    'A1 chg best %.4f → %s。**重复：%s**\n'
    % (pc['sp_spec'], vd[7],
       ' '.join('L%d:%.3f' % (l,
                              c2a[str(l)])
                for l in LAYERS_C),
       ' '.join('L%d:%.3f' % (l,
                              REF32_CHG256[l])
                for l in LAYERS_C),
       ' '.join('L%d:%d' % (l, fa[str(l)])
                for l in LAYERS_C),
       c3_best, vd[8],
       ('层位层级结构跨方向迁移成立'
        if vd[7] == 'a1_transfer_high'
        else '层级结构方向特异——A1 迁移'
             '失败，写入链组织随方向变化')))
nums = (
    'Part B：dvec med||d|| %s；co50 能量份额 '
    '%.3f；chg 移植 0.5/1/2=%.4f/%.4f/%.4f、'
    'allstep %.4f、L38 %.4f、J1 %.4f（预测 '
    '%.4f）、J2 %.4f。Part C：medE A1/P=%d/%d、'
    'sp_prof %.3f、sp_spec %.3f、chgA1 峰 '
    'L%d %.3f、C3 best %.4f。xphase 漂移记录 '
    'P/A1/C3=%.2f/%.2f/%.2f。'
    % (' '.join('L%s:%.1f' % (k, v)
                for k, v in sorted(
                    dn.items())),
       pb['dvec17_co50_energy'],
       chg17_05, chg17, chg17_20, chg17_as,
       chg38, chgJ1, predJ1, chgJ2,
       pc['medE_A1'], pc['medE_P'],
       pc['sp_prof'], pc['sp_spec'],
       max(LAYERS_C,
           key=lambda l: c2a[str(l)]),
       max(c2a[str(l)] for l in LAYERS_C),
       c3_best,
       pb['xphase_base_match_P'],
       pc['xphase_base_match_A1'],
       pc['xphase_base_match_c3']))
hards = (
    '①移植仅 prompt 末位单点（allstep 一组'
    '对照），未做多位置联合移植；②dvec 为 '
    'bf16 状态差，剂量尺度与 3132 标量 δ '
    '坐标系不可直接比较；③fork-layer 剖面为 '
    'teacher-forced swap4 惯例（k=0 读出），'
    'first-divergence 判据依赖 margin 符号'
    '翻转（guard 0.1）；④A1 谱迁移对照为 '
    'P 侧 3132 跨 Phase 数值，无方向内重测'
    '基线；⑤J1 加性检验样本 672 但单次'
    '测量，无重复方差估计；⑥跨 Phase 漂移'
    '实锤：无扰动生成 vs 3126 冻结 base 位级'
    '匹配仅 P %.2f/A1 %.2f/C3 %.2f，捕获态 '
    'vs 3132 hout rel-L2 %.2f——本 Phase 全部 '
    'same 判决改用会话内基线（z26 base 降级'
    '为漂移记录 xphase_*），跨 Phase 状态/'
    '生成数值不可直接比较（单样本捕获 vs '
    'full-track kernel 噪声 med|d| 0.08 margin '
    '单位）；捕获索引映射 idx18 经 REPRO+'
    'SMOKE R7 32 行中位双锁定（REPRO row0 '
    'idx17 匹配 0.0106 为 kernel 偏移巧合）。'
    % (pb['xphase_base_match_P'],
       pc['xphase_base_match_A1'],
       pc['xphase_base_match_c3'],
       pb['base17_vs_3132_maxrel']))
mech = (
    '①L17 因果三角第四角：状态级移植 %s'
    '（top50 标量 0.92 改写 + 全维移植 '
    '%.3f）→ 中介判定；②A1 首位不对称'
    '（614 vs 178）层位剖面 medE %d vs '
    '%d；③谱层级结构 A1 迁移 sp %.2f'
    '（%s）。'
    % (vd[1], chg17, pc['medE_A1'],
       pc['medE_P'], pc['sp_spec'], vd[7]))
if vd[1] == 'transplant_l17_full':
    p1 = ('①L17 状态下游消费路径：L17 移植'
          '态如何被 L18–L38 写入链消费至 '
          'readout（层间传导核验）。')
elif vd[1] == 'transplant_l17_partial':
    p1 = ('①swap4 各层 dvec 逐层移植剂量-'
          '响应矩阵（载体分解）。')
else:
    p1 = ('①3132 top50 改写伪象排查'
          '（坐标方向×层位双扫）。')
p2 = ('②A1 首位分叉坐标化：first_div 差异'
      '层位的坐标级归因 + 生成循环内逐步'
      '干预轨迹（回应 3132 硬伤④）。')
p3 = ('③四维完型第二象限：A1 半区 k=0–12 '
      '层轮廓位置维展开 + 跨方向条件二阶'
      '差分。')
prereg = p1 + p2 + p3
title = ('\n## Phase 3133: Ω-P131 全维移植'
         '判别+A1首位剖面（T4 第16Phase）'
         '[' + STAMP + ']\n\n')
assert len(title) < 110
sec = (
    title
    + '**性质**：T4 第 16 Phase，3132 MEMO '
    '§5 预注册三项执行，design_seal.json '
    '观测前冻结。Part A offline 3132 链接'
    '断言（result sha8 e4a0c198 + seal '
    '81c4820c + co50 52b126af 重算）；'
    'Part B1 swap4 全维状态捕获（L17/29/'
    '33/38，672 样）；Part B2/B3 全维移植 '
    '7 组生成（剂量 0.5/1/2 + allstep + '
    'L38 + J1/J2 联合）；Part C1 A1/P tf '
    'swap4 fork 剖面（128+128）；Part C2 '
    'A1 谱 8 层 × 256（sel256 配对）；'
    'Part C3 L17 坐标注入 A1。运行 '
    + ('%.0fs' % r['runtime_s']) + '。\n\n'
    + '### 1. 三大发现（重复三遍）\n'
    + f1 + f1 + f1 + f2 + f2 + f2 + f3
    + f3 + f3 + '\n'
    + '### 2. 关键数值\n'
    + nums + '\n\n'
    + '### 3. 硬伤\n' + hards
    + '\n\n### 4. 机制拼图更新\n' + mech
    + '\n\n### 5. 3134 预注册（观察后冻结）'
    '\n' + prereg + '\n\n'
    + '产物：`tests/glm5/result/'
    'rdc_query_construction_20260913/'
    'phase3133/omega_p131_transplant_'
    'a1fork_migrate/`（result.json、'
    'design_seal.json、run_log.txt、'
    'p131_readout.npz）；脚本 '
    '`tests/glm5/phase3133_omega_p131_'
    'transplant_a1fork_migrate.py`。')
assert len(sec) > 2500

# sha8 of result.json
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

steps = []
if 'meas3133_omega_p131' in io.open(
        LEDGER, encoding='utf-8').read():
    steps.append('ledger: exists, skip')
else:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    claim = (
        'Omega-P131 (3133, T4 sixteenth '
        'phase: glm4 strong-intervention '
        'disambiguation of L17 gate + A1 '
        'first-token fork profile + '
        'write-chain spectrum A1 '
        'migration. Full-dim swap-state '
        'transplant dvec(h_swap4-h_base) '
        'L17/29/33/38: L17 dose 0.5/1/2 '
        'chg %.4f/%.4f/%.4f (%s), allstep '
        '%.4f, L38 %.4f, J1 {17,38} %.4f '
        'vs pred %.4f -> %s, J2 %.4f; tf '
        'fork profile A1 vs P medE %d/%d '
        'sp %.3f -> %s; A1 spectrum '
        '(sel256 e34d2588) sp %.3f -> %s, '
        'L17-coords-on-A1 best %.4f -> %s '
        '- verdict '
        % (chg17_05, chg17, chg17_20,
           vd[2], chg17_as, chg38, chgJ1,
           predJ1, vd[4], chgJ2,
           pc['medE_A1'], pc['medE_P'],
           pc['sp_prof'], vd[6],
           pc['sp_spec'], vd[7], c3_best,
           vd[8])) + V
    entry = {
        'meas_id':
            'meas3133_omega_p131_'
            'transplant_a1fork_migrate',
        'phase': 3133,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3133/omega_p131_.../'
                'result.json sha256_8='
                + sha8,
            'npz': 'p131_readout.npz'},
        'hashes': {'result_sha256_8':
                   sha8},
        'anchors': [],
        'note': 'transplant %s; profile '
                '%s; migration %s; coords '
                '%s' % (vd[1], vd[6], vd[7],
                        vd[8])}
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
if '## Phase 3133:' in memo:
    steps.append('memo: exists, skip')
else:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    steps.append('memo: appended (short '
                 'title, 5 sections)')

WDATE = NOW.strftime('%Y-%m-%d')
wline = ('- Phase 3133 Omega-P131 closeout: '
         'verdict ' + V + '; ledger sha8 '
         + json.load(io.open(
             LEDGER,
             encoding='utf-8'))
         ['ledger_sha256_8']
         + '; MEMO 3133 section; runtime '
         + ('%.0fs' % r['runtime_s']) + '.')
for wd in WLOGS:
    wl = wd + '\\' + WDATE + '.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3133 Omega-P131 closeout' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        steps.append('wlog: ' + wl[:12])
    else:
        steps.append('wlog: exists '
                     + wl[:12])

mem = io.open(MEMW, encoding='utf-8').read()
if '3133（T4）' in mem:
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
        '- 3129（T4）：',
        '- 3129（T4）：剂量单调复现；swap '
        '全量重写而 margin 仅 172 翻转='
        '行为读出解耦。')
    mem = compress_line(
        mem,
        '- 3128（T4）：',
        '- 3128（T4）：多层 swap 阴性=写入链'
        '纯相关；坐标注入 dose 单调。')
    trans_sum = {
        'transplant_l17_full':
            'L17 全维移植 chg %.3f=swap 级'
            '充分（相关=中介）' % chg17,
        'transplant_l17_partial':
            'L17 全维移植 chg %.3f 部分充分'
            '（写入链分散）' % chg17,
        'transplant_l17_null':
            'L17 全维移植 chg %.3f 无效'
            '（伪象嫌疑）' % chg17}[vd[1]]
    new_line = (
        '- 3133（T4）：%s；A1 fork 剖面 %s'
        '（medE %d/%d sp %.2f）；A1 谱迁移 '
        'sp %.2f（%s）；L17 坐标→A1 chg '
        '%.3f（%s）。'
        % (trans_sum, vd[6].replace(
            'a1_forklayer_', ''),
           pc['medE_A1'], pc['medE_P'],
           pc['sp_prof'], pc['sp_spec'],
           vd[7].replace('a1_transfer_', ''),
           c3_best,
           vd[8].replace('a1_l17coord_', '')))
    anchor = '- 3132（T4）：'
    ia = mem.find(anchor)
    assert ia > 0
    mem = mem[:ia] + new_line + '\n' + \
        mem[ia:]
    old2 = '下一 3133：**'
    i2 = mem.find(old2)
    assert i2 > 0, 'next-line anchor'
    j2 = mem.find('\n', i2)
    if vd[1] == 'transplant_l17_full':
        new2 = ('下一 3134：**L17 状态下游'
                '消费路径（L17→L38→readout '
                '层间传导核验）+ A1 首位分叉'
                '坐标化 + 生成循环内逐步干预'
                '。**')
    else:
        new2 = ('下一 3134：**swap4 各层 '
                'dvec 逐层移植剂量矩阵 + A1 '
                '首位分叉坐标化 + 生成循环内'
                '逐步干预。**')
    mem = mem[:i2] + new2 + mem[j2:]
    oldm = '- max=3132，'
    assert mem.count(oldm) == 1
    mem = mem.replace(oldm,
                      '- max=3133，')
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
