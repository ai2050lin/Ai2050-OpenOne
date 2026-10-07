# -*- coding: utf-8 -*-
"""Phase 3120 closeout (idempotent):
Ledger -> MEMO Phase 3120 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3120'
        r'\omega_p118_content_attr_amplifier_'
        'behavior_opshape')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
V = res['verdict']
assert V == 'attribution_unresolved|' \
    'amplifier_partial|not_applicable|' \
    'mean_reversion_confirmed|linear_operator', V
assert res['smoke'] is False
assert res['n_records'] == 2016
assert res['n_pairs'] == 672
assert res['n_pairs_partb'] == 672
pa = res['part_a']
assert pa['verdict'] == 'attribution_unresolved'
assert pa['fact_margin_P_up'] is True
assert pa['fact_margin_A1_down'] is False
assert pa['gap_attribution_confirmed'] is False
assert abs(pa['gate_P']['pooled_diff']
           - 1.0774126996805224) < 1e-9
assert abs(pa['gate_P']['unit_rate']
           - 0.7586206896551724) < 1e-12
assert pa['gate_P']['n_valid_units'] == 29
assert abs(pa['gate_A1']['pooled_diff']
           - 2.0069585943974846) < 1e-9
assert abs(pa['gate_A1']['unit_rate'] - 0.25) < 1e-12
assert abs(pa['gate_gap']['dgap_ff']
           - (-0.07847756274049038)) < 1e-9
assert abs(pa['gate_gap']['dgap_nn']
           - 0.6537013573023083) < 1e-9
assert abs(pa['gate_gap']['contrast']
           - (-0.7321789200427986)) < 1e-9
assert abs(pa['fact_token_rate']['P']
           - 0.3364448051948052) < 1e-12
assert abs(pa['fact_token_rate']['A1']
           - 0.3476731601731602) < 1e-12
ctP = pa['class_table']['P']
ctA = pa['class_table']['A1']
assert abs(ctP['syntax']['mean_dm']
           - (-2.3066124154259664)) < 1e-12
assert abs(ctP['fact_strict']['mean_dm']
           - 0.618677744531804) < 1e-12
assert ctP['fact_strict']['n'] == 2487
assert abs(ctA['syntax']['mean_dm']
           - (-4.249947949893155)) < 1e-12
assert abs(ctA['fact_strict']['mean_dm']
           - 0.22660683546548688) < 1e-12
assert ctA['fact_strict']['n'] == 2570
assert pa['subtag_table']['P']['queried']['n'] == 2463
assert abs(pa['subtag_table']['P']['queried']
           ['mean_dm'] - 0.6192993277996186) < 1e-12
assert pa['subtag_table']['A1']['context_other']['n'] == 2570
assert pa['subtag_table']['A1']['novel']['n'] == 0
assert pa['yes_no_at_content_steps'] == 0
pb = res['part_b']
assert pb['verdict'] == \
    'amplifier_partial|not_applicable'
assert pb['beh_verdict'] == 'amplifier_partial'
assert pb['samp_verdict'] == 'not_applicable'
assert pb['repro_verdict'] == \
    'sampled_pipeline_reproduced'
assert pb['repro_max_diff'] == 0.0
assert abs(pb['yes_rate_greedy']['clean']
           - 0.5885416666666666) < 1e-12
assert abs(pb['yes_rate_greedy']['abl_L30']
           - 0.5729166666666666) < 1e-12
assert abs(pb['yes_rate_greedy']['abl_L32']
           - 0.546875) < 1e-12
assert abs(pb['beh_max']
           - 0.04166666666666663) < 1e-12
assert abs(pb['agree_greedy']['clean']
           - 0.028273809523809524) < 1e-12
assert abs(pb['yes_rate_sampled']['clean']
           - 0.5941666666666666) < 1e-12
assert abs(pb['yes_rate_sampled']['abl_L30']
           - 0.5666666666666667) < 1e-12
assert abs(pb['yes_rate_sampled']['abl_L32']
           - 0.5566666666666666) < 1e-12
assert abs(pb['samp_max']
           - 0.03749999999999998) < 1e-12
assert abs(pb['state_ratio']['L30']['ratio']
           - 1.3606021421196623) < 1e-9
assert abs(pb['closed_ratio']['L30']['ratio']
           - 3.027212089110517) < 1e-9
assert abs(pb['recheck_greedy']['abl_L30']
           - 0.9889322916666667) < 1e-9
pc = res['part_c']
assert pc['verdict'] == \
    'mean_reversion_confirmed|linear_operator'
assert pc['mono_verdict'] == \
    'mean_reversion_confirmed'
assert pc['quad_verdict'] == \
    'nonlinear_curvature_absent'
assert pc['gate_verdict'] == 'linear_operator'
assert abs(pc['primary']['spearman']
           - (-0.6256644090477388)) < 1e-12
assert abs(pc['primary']['r2_lin']
           - 0.3429356255118402) < 1e-12
assert abs(pc['primary']['d_quad']
           - 7.139464048599997e-05) < 1e-12
assert abs(pc['primary']['d_pw']
           - 0.002277820072898673) < 1e-12
assert abs(pc['primary']['slope_lin']
           - (-0.6801486439276976)) < 1e-12
assert abs(pc['primary']['pw_breakpoint']
           - (-2.1063614659351377)) < 1e-9
assert abs(pc['t2_sensitivity']['spearman']
           - (-0.6683672673127151)) < 1e-12
assert abs(pc['t2_sensitivity']['r2_lin']
           - 0.40101512829355523) < 1e-12
assert pc['t2_flip'] == {'mono': False,
                         'gate': False}
assert abs(pc['fixed_point_raw_lin']
           - (-6.052991245322225)) < 1e-9
assert abs(pc['bin_curve'][0]['y_mean']
           - 4.363031010886607) < 1e-9
assert abs(pc['bin_curve'][19]['y_mean']
           - (-4.65047143540284)) < 1e-9
o.append('asserts ok (%d lines checked)'
         % len(o))

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3120
           for m in led['measurements']):
    claim = (
        'Omega-P118 (3120, T4 third phase: '
        'content-step attribution + amplifier '
        'behavioral transmission + operator shape, '
        'qwen3-4b, 2360s: 2x1344 greedy ablation '
        'gens + 3x1200 sampled gens + 8400 tracking '
        'forwards) - verdict '
        'attribution_unresolved|'
        'amplifier_partial|not_applicable|'
        'mean_reversion_confirmed|'
        'linear_operator.  Part A (content-token '
        'span attribution, 14784 content tokens on '
        'frozen 3118 sequences): preregistered '
        '"fact-restatement drives gap recovery" '
        'DIRECTION REFUTED - the model restates '
        'facts at ~1/3 of content tokens (P 33.6pct '
        '/ A1 34.8pct; sub-tags: P restates the '
        'QUERIED statement n=2463, A1 restates '
        'context lines n=2570, NOVEL facts zero), '
        'and restatement pushes BOTH margins UP (P '
        '+0.62, A1 +0.23 per token) = assertion-'
        'pull that erodes the A1 no-belief; gap '
        'recovery is driven by NEUTRAL steps, '
        'especially punctuation (syntax dm P '
        '-2.31 / A1 -4.25) -> dgap(FF) -0.078 vs '
        'dgap(NN) +0.654 (contrast -0.732).  '
        'A-FACT-P passed (+1.077, unit_rate 0.759, '
        '29/55 units); A-FACT-A1 failed (sign '
        '+2.007, not negative); A-GAP2 rejected '
        '(direction reversed).  Fact-restatement '
        'rate constant 0.45-0.48 for t=2..8 then '
        'collapses (0.36/0.14/0.02 at t=9/10/11). '
        ' Part B (L30/L32 greedy 672 pairs + '
        'sampled 300 pairs x 2 dirs x 2 reps x 3 '
        'conds; clean sampled margins BIT-EXACT '
        'vs 3118, max diff 0.00e+00): amplifier '
        'behavioral transmission is PARTIAL and '
        'OPPOSITE to L26 - yes_rate drops L30 '
        '-1.56pp / L32 -4.16pp (L26 was +7.6pp), '
        'beh_max 0.0417 -> amplifier_partial; '
        'sampled yes drops same direction '
        '(-2.75/-3.75pp, samp_max 0.0375) though '
        'preregistered B-SAMP gate not applicable '
        'under partial; sampled closed-loop ratio '
        '3.03/2.47 vs state ratio 1.36/1.06 '
        '(closed-state separation holds).  Part C '
        '(operator shape, 16128 samples, '
        'direction-centered): spearman(x, dm) '
        '-0.626 -> mean_reversion_confirmed; '
        'linear R2 0.343, quadratic +0.00007, '
        'piecewise +0.0023 -> LINEAR OPERATOR, '
        'slope -0.680 (68pct reversion per step), '
        'raw fixed point m*=-6.05; t>=2 '
        'sensitivity stronger (-0.668, R2 0.401), '
        'no flips.  3119 rewrite_nonlinear '
        'RECONCILED: the conditional-mean shape '
        'is linear mean reversion; the low '
        'predictability comes from large residual '
        'variance (content-driven perturbations = '
        'the Part A token-class effects), not '
        'curvature.  NEXT 3121: counterfactual '
        'restatement replacement (correlation -> '
        'causality), erase-chain behavioral '
        'polarity (readout reversal / joint '
        'ablation), two-component forward '
        'reconstruction of the oscillation.')
    meas = {
        'meas_id': 'meas3120_omega_p118_content_'
                   'attr_amplifier_behavior_'
                   'opshape',
        'phase': 3120,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen before '
                   'computation: A-FACT '
                   '+/-0.10 & unit_rate 0.6, A-GAP2 '
                   'FF>=+0.10 & contrast>=+0.10 & '
                   'rate 0.55; B-BEH 0.05/0.02, '
                   'B-REPRO ==0.0, B-SAMP 0.03 '
                   '(only if greedy behavioral); '
                   'C-MONO -0.5/-0.2, C-QUAD '
                   '0.05, C-GATE 0.05',
        'artifacts': {
            'result': 'phase3120/omega_p118_'
                      'content_attr_amplifier_'
                      'behavior_opshape/'
                      'result.json',
            'seal': 'phase3120/omega_p118_'
                    'content_attr_amplifier_'
                    'behavior_opshape/'
                    'design_seal.json',
            'readout': 'phase3120/omega_p118_'
                       'content_attr_amplifier_'
                       'behavior_opshape/'
                       'p118_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 2360s); Part A/C offline '
                'on 3118 frozen npz; clean sampled '
                'margins bit-exact vs 3118 (0.0); '
                'AUC curve cross-check 0.00e+00',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3120_omega_p118_content_attr_'
        'amplifier_behavior_opshape')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- MEMO Phase 3120 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3120:' not in memo:
    sec = u'''## Phase 3120: Ω-P118 内容步归因 + 放大器行为传导 + 回归均值算子形状（T4 第3Phase）——**预注册"事实重述驱动恢复"方向否定：重述步把两方向 margin 一齐上推（对 A1=信念侵蚀），标点/中性步才是 gap 恢复主力**；**L30/L32 擦除链获得行为因果角色：贪心 yes 率 −1.6/−4.2pp、方向与 L26 主写链相反（partial 级）**；**回归均值算子均值形状=线性（斜率 −0.680、R² 0.343、无曲率无门控），3119"非线性"实质=大方差内容扰动** [[NOW]]

**性质**：T4 第 3 Phase，3119 MEMO 第 5 节预注册、门在 seal 观测前冻结。qwen3-4b BF16，2360s（贪心 2×1344 + 采样 3×1200 + 追踪 8400 前向）。A 部分（离线，3118 冻结序列的 14784 个内容 token 字节级 BPE 重建 + 正则 span 标注：fact_strict/fact_shaped/query_rest/syntax/other + 子标签 queried/context_other/novel）；B 部分（GPU）：L30/L32 单层 MLP 消融贪心 672 对 + 采样 300 对×{P,A1}×2 reps×{clean,L30,L32}（种子规则同 3118，**clean 采样 margin 与 3118 bit-exact 0.00e+00**）+ 2×2 追踪；C 部分：16128 个 (对,步,方向) 样本 y=Δm 对 x=m(t−1) 的形状拟合（方向中心化主分析 + 原始/分方向/t≥2 三个次级分析，20 等频 bin + 对级 bootstrap 1000 次）。

### 1. 三大发现（重复三遍）
1. **预注册"事实重述驱动恢复"方向否定：重述步把两方向 margin 一齐上推，标点/中性步才是恢复主力**。模型在 ~1/3 内容 token 上重述事实（P 33.6%、A1 34.8%）；子标签揭示重述结构：**P 重述被查询句（queried n=2463/2487），A1 重述上下文真实行（context_other n=2570/2570），novel 事实=0**（模型绝不编造新事实，只复述）。重述把两方向 margin 都上推（P +0.619、A1 +0.227/token）——对 A1（正确答案=no）这是**断言牵引=信念侵蚀**；gap 恢复由中性步驱动，标点步最猛（syntax dm：P −2.31、A1 −4.25）→ dgap(FF)=−0.078 vs dgap(NN)=+0.654（contrast −0.732）。门结果：A-FACT-P 过（+1.077、rate 0.759、29/55 单元）；A-FACT-A1 败（+2.007 而非负、rate 0.25）；A-GAP2 反向 rejected → attribution_unresolved（预注册符号预测错误=方向否定，本身即发现）。
2. **L30/L32 擦除链获得行为因果角色：部分级、方向与 L26 主写链相反**。全量贪心 yes 率：clean 0.5885 → L30 0.5729（−1.56pp）、L32 0.5469（−4.16pp）→ beh_max 0.0417 → amplifier_partial（0.02–0.05 带）；**L26 消融是 +7.6pp（升 yes），L30/L32 消融是降 yes——擦除链的正常写入在行为极性上与主写链对峙**。采样同向确认：yes 0.5942 → L30 0.5667（−2.75pp）、L32 0.5567（−3.75pp），samp_max 0.0375（≥0.03 但预注册 B-SAMP 门在 greedy partial 下 not_applicable——数值如实记录）。**clean 采样 margin 与 3118 bit-exact（max diff 0.00e+00）**——采样管线+种子规则+GPU 确定性完整复现。采样闭环比 3.03/2.47 vs 状态比 1.36/1.06——闭环-状态分离在放大器上同样成立。
3. **回归均值算子的条件均值形状=线性：每步向固定点收缩 68%，无曲率无门控**。spearman(m(t−1), Δm)=−0.626 → mean_reversion_confirmed；线性 R²=0.3429、二次仅 +0.00007、分段线性仅 +0.00228（最优断点 −2.11）→ **linear_operator**；斜率 −0.680，原始尺度固定点 m*=−6.05（no 侧吸引子）。t≥2 敏感性更强（−0.668、R² 0.401）无翻转；分方向一致（P −0.614/A1 −0.626）；20 bin 曲线从 +4.36 单调降到 −4.65。**3119 rewrite_nonlinear 的调和：3119 预测的是 m(t)（低 R² 0.19），3120 拟合的是 Δm 的条件均值形状——形状线性，低可预测性来自大方差残差（=Part A 的 token 类内容扰动），不来自均值弯曲**。

### 2. 关键数值
类表（mean_dm/token）：P other −0.692(n=3692)、syntax −2.307(1147)、query_rest +2.213(66)、fact_strict +0.619(2487)；A1 other −0.222(3841)、syntax −4.250(977)、fact_strict +0.227(2570)。重述率时间谱：t=2–8 恒定 0.45–0.48 → t=9 0.36 → t=10 0.14 → t=11 0.02（生成后期停止重述）。agree_P_A1：clean 0.0283、L30 0.0193、L32 0.0119（消融降低两方向续写一致性）。own-recheck：贪心 0.989/0.993，采样 0.846/0.882。内容步 yes/no token=0（复证 3119）。

### 3. 硬伤
① 预注册符号预测错误（A-FACT-A1 与 A-GAP2 方向反）——门机制正常但假说错，attribution_unresolved 是诚实的无判决；② "断言牵引"是机制性解读，未做反事实验证（重述 span 替换为中性内容的因果测试未做——3121 首任务）；③ B-SAMP 门因 greedy partial 而 not_applicable，采样同向数值（0.0375）未被正式门采纳（门结构教训：应允许 partial 时也评估）；④ beh_max 0.0417 距 behavioral 线仅 0.0083，门边界敏感；⑤ syntax 类巨幅负 dm 可能与句尾位置韵律混杂（未做位置×类别交叉控制）；⑥ 12 步窗口尾部重述骤停造成类构成漂移；⑦ 固定点/斜率依赖 w_yes−w_no 单一读出投影；⑧ 单模型、放大层仅 2 个。

### 4. 机制拼图更新
内部响应图谱新增：**内容步 token 类别→Δm 映射表**（5 类×2 方向+子标签）+ **放大器行为传导符号**（L30/L32 反向于 L26）。RDC 更新：① **信念振荡双源分解**：m(t+1)=m(t)+λ(μ−m(t))+content(t)+noise，λ=0.680、μ=−6.05 为线性均值回归核，content(t)=重述上推（+0.2~+0.6）/标点下坠（−2.3~−4.3）的类别加性扰动——**AUC 振荡=回归均值与内容扰动的相位交替**，A1 侧扰动幅度约为 P 的 2 倍（syntax −4.25 vs −2.31），故振荡主要由 A1 侧驱动；② 与 3119"答案后弛豫"（A1 吃 no 后 +8.77）拼接：A1 信念双相动力学=弛豫上跳→重述侵蚀→标点巩固；③ 擦除链（L30/L32）不只是状态现象，参与行为极性维持（方向对抗主写链）；④ 重述内容 100% 来自身份记忆（queried/context_other、零 novel）——生成=记忆检索的复述而非新事实合成。

### 5. 3121 预注册（T4 继续，观测前冻结框架）
① **反事实重述替换测试**（相关性→因果性）：teacher-forced 重放时把重述 span 的 token 替换为（a）其他上下文事实行（b）乱序词序列（c）标点填充，检验 Δm 是否按替换内容类预测移动；② **擦除链行为极性**：L30/L32 联合消融 + 读出方向反转（w_no−w_yes 对照）+ A1 首 token 分叉统计，判定擦除链写入内容的极性归属；③ **振荡双源前向重构**：用「线性回归核+类别扰动」两成分模型从 m(0) 前向模拟 672 对轨迹，对比真实 AUC(t) 曲线（定量拟合优度门在 seal 冻结）；④ 位置×类别交叉控制解耦标点韵律。具体门在 3121 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3120/omega_p118_content_attr_amplifier_behavior_opshape/`（result.json、design_seal.json、run_log.txt、p118_readout.npz）；脚本 `tests/glm5/phase3120_omega_p118_content_attr_amplifier_behavior_opshape.py`。
'''
    sec = sec.replace('[[NOW]]', '[' + NOW + ']')
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3120)'
             % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3120 Omega-P118 (T4 third phase: '
          'content-step attribution + amplifier '
          'behavior + operator shape, qwen3-4b, '
          '2360s): verdict '
          'attribution_unresolved|'
          'amplifier_partial|not_applicable|'
          'mean_reversion_confirmed|'
          'linear_operator. (A) Preregistered '
          '"fact-restatement drives recovery" '
          'DIRECTION REFUTED: restatement (~1/3 of '
          'content tokens; P restates the QUERIED '
          'statement n=2463, A1 restates context '
          'lines n=2570, novel=0) pushes BOTH '
          'margins UP (P +0.62, A1 +0.23) = '
          'assertion-pull eroding the A1 no-belief;'
          ' recovery comes from NEUTRAL steps, '
          'esp. punctuation (syntax dm P -2.31 / '
          'A1 -4.25); dgap(FF) -0.078 vs dgap(NN) '
          '+0.654. (B) L30/L32 behavioral role '
          'PARTIAL and OPPOSITE to L26: yes_rate '
          '-1.56pp/-4.16pp (L26 was +7.6pp); '
          'sampled same direction; clean sampled '
          'margins BIT-EXACT vs 3118 (0.00e+00); '
          'closed-loop ratio 3.03/2.47 vs state '
          '1.36/1.06. (C) Operator shape LINEAR: '
          'spearman -0.626, R2 0.343, quad '
          '+0.00007, piecewise +0.0023, slope '
          '-0.680 (68pct/step reversion), fixed '
          'point m*=-6.05; 3119 rewrite_nonlinear '
          'reconciled as large-variance content '
          'perturbation, not mean curvature. NEXT '
          '3121: counterfactual restatement '
          'replacement + erase-chain behavioral '
          'polarity + two-component forward '
          'reconstruction.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3120 Omega-P118' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md update ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3120' not in mem_old:
    mem_new = mem_old.replace(
        u'## 机制链状态（3119）\n'
        u'- 3119（T4）：振荡机制三叉。**答案 token '
        u'仅 t=1，AUC 振荡由内容 token 步驱动'
        u'（假说方向否定）；A1 吃 no 后 +8.77=答案'
        u'后弛豫；自选择 AUC 0.84**。**补偿层特异：'
        u'10 收缩层（L24 0.36 最强、early-peak）vs '
        u'L30/L32 放大（1.38/1.15、late-peak）——'
        u'擦除链=生成后期持续去信念化**；L35 重建'
        u'否定。**重写算子非线性：R² 0.19、embedding '
        u'注入零贡献、交互 +10.3pp**。\n',
        u'## 机制链状态（3120）\n'
        u'- 3120（T4）：内容步归因+放大器行为+算子'
        u'形状。**预注册方向否定：重述步把两方向 '
        u'margin 一齐上推（P +0.62/A1 +0.23=对 A1 '
        u'信念侵蚀、novel=0 纯复述），标点/中性步才'
        u'恢复 gap（A1 syntax −4.25 最猛）**。'
        u'**L30/L32 行为角色反向于 L26（yes −1.6/'
        u'−4.2pp、partial）；采样管线 bit-exact '
        u'复现 3118**。**Δm 算子均值形状线性'
        u'（slope −0.68、R² 0.343、无曲率无门控）'
        u'——3119"非线性"=大方差内容扰动**。\n'
        u'- 3119（T4）：答案 token 仅 t=1、振荡由'
        u'内容步驱动；补偿层特异（10 收缩 early-peak '
        u'vs L30/L32 放大 late-peak）；重写非线性 '
        u'R² 0.19、交互 +10.3pp。\n')
    mem_new = mem_new.replace(
        u'max=3119', u'max=3120').replace(
        u'下一 3120：**内容 token 步振荡归因'
        u'（事实重述 vs 中性步）+ L30/L32 放大器'
        u'行为传导 + 回归均值算子定量形状**。',
        u'下一 3121：**反事实重述替换测试'
        u'（相关性→因果性）+ 擦除链行为极性'
        u'（读出反转/联合消融）+ 振荡双源前向'
        u'重构**。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars'
             % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
