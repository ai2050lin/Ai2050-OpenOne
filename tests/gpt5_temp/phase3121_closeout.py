# -*- coding: utf-8 -*-
"""Phase 3121 closeout (idempotent):
result asserts -> Ledger -> MEMO Phase 3121 ->
workspace logs (x2 entries) -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3121'
        r'\omega_p119_repl_causality_erase_'
        'polarity_recon')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
WDAYS = sorted(set([TODAY, '2026-09-23']))
o = []

V = ('content_nonspecific|'
     'syntax_polarity_absent|'
     'erase_joint_behavioral|'
     'readout_robust|'
     'reconstruction_failed|'
     'curve_shape_weak|coverage_ok|'
     'position_independent_confirmed')

# ---------- 1. result.json asserts ----------
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
assert res['verdict'] == V, res['verdict']
assert res['smoke'] is False
assert res['n_pairs'] == 672
assert res['np_a'] == 672
assert res['np_g'] == 672
assert abs(res['runtime_s'] - 669.4) < 0.05
pa = res['part_a']
assert pa['verdict'] == \
    'content_nonspecific|syntax_polarity_absent'
assert pa['repro']['verdict'] == 'replay_bit_exact'
assert pa['repro']['max_diff'] == 0.0
assert pa['n_span_P'] == 305
assert pa['n_span_A1'] == 320
assert pa['gates'] == {
    'P': 'content_nonspecific',
    'A1': 'content_nonspecific',
    'syntax_P': 'syntax_polarity_absent',
    'syntax_A1': 'syntax_polarity_absent'}
eP = pa['effects']['P']
assert eP['n_span'] == 305 and eP['n_len2'] == 305
assert abs(eP['D_mean']['c0']) == 0.0
assert abs(eP['D_mean']['c1']
           - (-2.0944437736859087)) < 1e-9
assert abs(eP['D_mean']['c2']
           - 0.4315817940430563) < 1e-9
assert abs(eP['D_mean']['c3']
           - 1.1288977380170198) < 1e-9
assert abs(eP['E_10_mean']
           - (-2.526025567728965)) < 1e-9
assert abs(eP['E_31_mean']
           - 3.2233415117029285) < 1e-9
assert abs(eP['E_10_sd'] - 1.42277025689462) < 1e-9
assert abs(eP['E_31_sd']
           - 0.8628208373949953) < 1e-9
eA = pa['effects']['A1']
assert eA['n_span'] == 320 and eA['n_len2'] == 320
assert abs(eA['D_mean']['c1']
           - 0.5678164122719318) < 1e-9
assert abs(eA['D_mean']['c2']
           - 1.844489301871508) < 1e-9
assert abs(eA['D_mean']['c3']
           - 2.423060708269477) < 1e-9
assert abs(eA['E_10_mean']
           - (-1.2766728895995765)) < 1e-9
assert abs(eA['E_31_mean']
           - 1.855244295997545) < 1e-9
pb = res['part_b']
assert pb['verdict'] == \
    'erase_joint_behavioral|readout_robust'
assert abs(pb['yes_rate_clean']
           - 0.5885416666666666) < 1e-12
assert abs(pb['yes_rate_joint']
           - 0.5275297619047619) < 1e-12
assert abs(pb['yes_delta']
           - (-0.06101190476190477)) < 1e-12
assert pb['joint_verdict'] == \
    'erase_joint_behavioral'
assert abs(pb['fam_r'] - 0.8499039671426715) < 1e-9
assert pb['fam_verdict'] == 'readout_robust'
ft = pb['first_token']
assert abs(ft['clean']['first_yes']
           - 0.5885416666666666) < 1e-12
assert abs(ft['abl_L30']['first_yes']
           - 0.9955357142857143) < 1e-12
assert abs(ft['abl_L32']['first_yes']
           - 0.9955357142857143) < 1e-12
assert abs(ft['abl_joint']['first_yes']
           - 0.5275297619047619) < 1e-12
pc = res['part_c']
assert pc['verdict'] == \
    'reconstruction_failed|curve_shape_weak|' \
    'coverage_ok'
assert abs(pc['r2_two']
           - 0.12843996911885736) < 1e-12
assert abs(pc['r2_lin']
           - 0.04183500685653896) < 1e-12
assert abs(pc['r2_persistence']
           - (-0.9618412331055202)) < 1e-12
assert abs(pc['d_r2']
           - 0.0866049622623184) < 1e-12
assert pc['fit_verdict'] == 'reconstruction_failed'
assert abs(pc['auc_r']
           - (-0.07890096967521298)) < 1e-12
assert abs(pc['auc_r_lin']
           - 2.276099300428211e-16) < 1e-18
assert pc['curve_verdict'] == 'curve_shape_weak'
assert abs(pc['sigma']
           - 3.3923936726908717) < 1e-9
assert abs(pc['coverage']
           - 0.9671347966269841) < 1e-9
assert pc['mc_verdict'] == 'coverage_ok'
assert abs(pc['slope_3120']
           - (-0.6801486439276976)) < 1e-12
assert abs(pc['mstar_3120']
           - (-6.052991245322225)) < 1e-12
pd_ = res['part_d']
assert pd_['verdict'] == \
    'position_independent_confirmed'  # buggy, kept
assert pd_['syntax_n_zero_P'] == 6
assert pd_['syntax_n_zero_A1'] == 5
assert abs(pd_['syntax_min_P']
           - (-0.7617262028314017)) < 1e-9
assert abs(pd_['syntax_max_P']
           - (-0.08661466046546472)) < 1e-9
o.append('asserts ok (%d checks)' % 40)

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3121
           for m in led['measurements']):
    claim = (
        'Omega-P119 (3121, T4 fourth phase: '
        'counterfactual restatement replacement '
        'causality + erase-chain joint-ablation '
        'polarity + two-component forward '
        'reconstruction, qwen3-4b, 669.4s: 4 '
        'conditions x 2 dirs x 2 readouts '
        'teacher-forced tracking + joint greedy '
        'gens + offline reconstruction/MC) - '
        'verdict ' + V + '.  Part A (token-level '
        'counterfactual replacement inside '
        'restatement spans; replay c0 BIT-EXACT '
        'vs 3118, max diff 0.0; 367/352 pairs '
        'without fact spans, effects on P 305 / '
        'A1 320 span pairs): A-CAUS gates fail '
        'with signs OPPOSITE to preregistration - '
        'E_10 = D_c1-D_c2 = -2.53 (P) / -1.28 '
        '(A1) vs gate +0.10, E_31 = D_c3-D_c1 = '
        '+3.22 / +1.86 vs gate -0.05 -> '
        'content_nonspecific|syntax_polarity_'
        'absent.  Probe refinement: replacement '
        'responses are +/-3~5 STRONG OSCILLATIONS '
        '(c1 trough -4.7 at t=5-6, c2 spike +4.55 '
        'at t=3) with misaligned phases across '
        'pairs, swamping the ~2-sized condition-'
        'mean differences; inside-span means DO '
        'separate (c1 -1.66 / c2 +1.43 / c3 '
        '+1.89) but AFTER the span all three '
        'converge to a shared negative drift '
        '(-3.41/-2.27/-0.73) -> the '
        'content_nonspecific verdict is a '
        'TOKEN-LEVEL PARADIGM FAILURE (OOD '
        'fragmented-text confound), NOT positive '
        'evidence of content independence.  Part '
        'B (joint L30+L32 greedy ablation): yes '
        '0.5885 -> 0.5275 (-6.1pp >= 0.05) -> '
        'erase_joint_behavioral; family-readout '
        'increment correlation r=0.8499 >= 0.8 -> '
        'readout_robust.  Implementation bug '
        '(vstack[:NP_G] slice made abl_L30/L32 '
        'fork rows P-only 672 instead of mixed '
        '1344; 0.5885x1344=791 integer proves the '
        'mixed caliber) accidentally exposed the '
        'DIRECTION SPLIT that REVERSES the 3120 '
        'behavioral reading: clean_P first_yes '
        '0.9970 (saturated) vs clean_A1 0.1801; '
        'after ablation L30_A1 0.1503 / L32_A1 '
        '0.0982 / joint_A1 0.0655 (monotone '
        'toward the CORRECT no, joint strongest) '
        'with P side saturated ~0.99 -> erase-'
        'chain write = assertion-pull (pulls '
        'inconsistent pairs toward yes/wrong), '
        'ablating it RELEASES the no prior and '
        'IMPROVES discrimination; 3120 mixed '
        '-1.56/-4.16pp was the P-saturation + '
        'A1-down composite.  3118 re-split (L26/'
        'L31/L33): L26_P 0.9970 / L26_A1 0.3318 '
        '(vs clean_A1 0.1801 -> +15.2pp yes-ward) '
        '-> the 3118 mixed +7.6pp comes ENTIRELY '
        'from the A1 side; L26 write = maintain '
        'no-belief, polarity-opposed to L30/L32 '
        'writes - the 3120 confrontation claim '
        'holds and sharpens under direction '
        'split.  Part C (two-component '
        'deterministic reconstruction with 3120-'
        'frozen slope -0.680 / m*=-6.05 + per-'
        '(dir,cls,t) content deviations): R2_two '
        '0.128 < 0.2 -> reconstruction_failed '
        '(linear-only 0.042, persistence -0.962); '
        'AUC-curve corr -0.079 -> curve_shape_'
        'weak; BUT 20-rep seed-3121 MC noise band '
        '(sigma 3.392, +/-1.96*sigma*sqrt(t)) '
        'coverage 0.967 >= 0.90 -> coverage_ok -> '
        'the oscillation is a DISTRIBUTIONAL '
        'phenomenon, not a per-trajectory '
        'deterministic one; the 3120 two-source '
        'decomposition holds at the conditional-'
        'mean level only.  Part D bug (1-D '
        'boolean dm[sel] selected ROWS not '
        'elements -> per-t table and gate '
        'invalid; corrected probe: 6/5 zero-'
        'sample steps, positive +4.85/+7.95 at '
        't=5) -> corrected verdict '
        'position_confounded (result.json kept '
        'immutable, bug + correction recorded).  '
        'NEXT 3122: sentence-level coherent '
        'replacement, direction-split behavioral '
        'gates as standard, distribution-level '
        'reconstruction (transition-distribution '
        'fit), L26/L31/L33 write-content readout.')
    meas = {
        'meas_id': 'meas3121_omega_p119_repl_'
                   'causality_erase_polarity_'
                   'recon',
        'phase': 3121,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen before '
                   'computation: A-CAUS P +0.10/'
                   '+0.05 (A1 +0.05/+0.025), '
                   'A-SYNTAX -0.05/-0.025, '
                   'A-REPRO ==0.0 / <1e-6, '
                   'B-JOINT 0.05/0.02, B-FAM r '
                   '0.8/0.5, C-FIT R2>=0.5 & '
                   'dR2>=0.05 / fail <0.2, '
                   'C-CURVE 0.9, C-MC coverage '
                   '0.90 (20 reps seed 3121), '
                   'D-POS all-step negative',
        'artifacts': {
            'result': 'phase3121/omega_p119_repl_'
                      'causality_erase_polarity_'
                      'recon/result.json',
            'seal': 'phase3121/omega_p119_repl_'
                    'causality_erase_polarity_'
                    'recon/design_seal.json',
            'readout': 'phase3121/omega_p119_'
                       'repl_causality_erase_'
                       'polarity_recon/'
                       'p119_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 669.4s); Parts C/D '
                'offline on frozen annotations; '
                'replay bit-exact 0.0; TWO '
                'implementation bugs found post-'
                'run (Part B mixed-caliber fork '
                'slice P-only; Part D 1-D boolean '
                'row selection) - corrections via '
                'probes p3121_probe2/probe3, '
                'result.json immutable',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3121_omega_p119_repl_causality_'
        'erase_polarity_recon')
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

# ---------- 3. MEMO Phase 3121 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3121:' not in memo:
    sec = u'''## Phase 3121: Ω-P119 反事实重述替换因果性 + 擦除链联合消融极性 + 振荡双源前向重构（T4 第4Phase）——**token 级替换范式失效：替换响应是 ±3~5 强振荡，条件均值差被相位淹没（content_nonspecific=范式失效而非内容无关的正面证明）**；**方向分解翻转擦除链行为结论：擦除链写入=断言牵引、消融=释放 no 先验提高判别（joint A1 first_yes 0.0655 vs clean 0.1801，P 侧饱和不动）**；**两成分确定性重构失败（R² 0.128）但 MC 噪声带校准良好（coverage 0.967）——振荡是分布现象而非逐轨迹确定性现象** [[NOW]]

**性质**：T4 第 4 Phase，3120 MEMO 第 5 节预注册、门在 seal 观测前冻结（design_seal.json created 2026-09-23 17:23:54）。qwen3-4b BF16，669.4s。Part A：teacher-forced 4 条件（c0 重放/c1 其他对 queried 行填充/c2 span 乱序/c3 '.' 填充）×双方向×双读出（w_dn 与家族均值差 w_fam）；672 对中 367/352 无 fact span，有效效应样本 P 305/A1 320（span len 集中 7–10）。Part B：L30+L32 联合消融贪心 672 对+首 token 分叉+家族读出增量相关（设计升级：w_no−w_yes=−w_dn 平凡镜像，改家族级读出）。Part C：两成分前向重构 m̂(t+1)=m̂(t)+S(m̂(t)−MS)+content[dir][cls(j,t)][t]（3120 冻结 slope −0.680/m* −6.05）+20 reps seed 3121 MC 噪声带。Part D：位置×类别交叉（结果见硬伤③）。

### 1. 三大发现（重复三遍）
1. **token 级替换范式失效：替换响应是 ±3~5 强振荡，淹没条件均值差**。A-CAUS 门全线失败且方向与预注册相反：P 侧 D_c1=−2.094（其他对真实 token 填充，预测 yes 上推→实测 no 下推）、D_c2=+0.432（乱序）、D_c3=+1.129（'.' 填充），E_10=−2.526（门 ≥+0.10）、E_31=+3.223（门 ≤−0.05）；A1 侧 E_10=−1.277、E_31=+1.855 同向。诊断探针揭示原因：替换 token 引发 ±3~5 强振荡（c1 深谷 t=5–6 达 −4.7、c2 尖峰 t=3 达 +4.55、c3 峰 +3.75），不同对相位不同 → ~±2 量级的条件均值差被相位混合淹没。span 内/外分解：span 内三条件均值确实可分（c1 −1.66 vs c2 +1.43 vs c3 +1.89——内容条件依赖存在），span 后三条件收敛为共同负漂（−3.41/−2.27/−0.73）。**判决读法修正：content_nonspecific 应读作"token 级替换范式失效"（OOD 破碎文本混杂），不是"内容无关"的正面证明；相关→因果的桥梁需要句级连贯替换范式**。
2. **方向分解翻转擦除链行为结论：写入=断言牵引，消融=提高判别**。Part B 联合消融 yes 率 0.5885→0.5275（−6.1pp ≥0.05 → erase_joint_behavioral）；家族读出 r=0.8499 ≥0.8 → readout_robust。实现 bug（vstack[:NP_G] 切片使 abl_L30/L32 fork 为 P-only 672；0.5885×1344=791 整数实锤混合口径）意外暴露方向分解：**clean_P first_yes=0.9970（饱和）、clean_A1=0.1801；消融后 L30_A1=0.1503、L32_A1=0.0982、joint_A1=0.0655——朝正确答案 no 单调移动、joint 最强，P 侧全部 ~0.99 饱和不动**。即擦除链（L30/L32）的正常写入把不一致对往 yes/错误方向拉（=断言牵引，与 3120 Part A 重述上推一致），消融它=释放 no 先验=提高判别；3120 的混合 −1.56/−4.16pp 是 P 饱和+A1 下移的合成。3118 L26/L31/L33 分方向重审（3118 npz）：L26_P 0.9970/L26_A1 0.3318（clean_A1 0.1801→yes 向 +15.2pp）、L31_A1 0.1161（no 向）、L33_A1 0.1994——**3118 混合 +7.6pp 完全来自 A1 侧；L26 写入=维持 no 信念，与 L30/L32 写入极性对抗在方向分解下成立且更尖锐**。
3. **两成分确定性重构失败，但噪声带校准良好——振荡是分布现象**。Part C 用 3120 冻结算子+类偏离表确定性前向重构 672×2 轨迹：R²_two=0.128 <0.2 → reconstruction_failed（线性核单独 0.042、持久性基线 −0.962）；AUC 曲线相关 r=−0.079 → curve_shape_weak（重构轨迹连 AUC 振荡相位都无法复现）。但同参数 MC 噪声模拟（σ=3.392、20 reps seed 3121、±1.96σ√t 带）覆盖率 0.9671 ≥0.90 → coverage_ok。**解读：3120 双源分解（回归核+类扰动）只在条件均值层面成立；逐轨迹行为受未建模大噪声（σ=3.39，与 margin 动态范围同量级）支配——振荡是统计/分布现象，不是逐轨迹确定性现象**。

### 2. 关键数值
Part A：D_mean P c0 0/c1 −2.0944/c2 +0.4316/c3 +1.1289；A1 c1 +0.5678/c2 +1.8445/c3 +2.4231；E_10 sd 1.423/1.349、E_31 sd 0.863/0.952；repro bit-exact 0.0；span len 分布 P [367,0×6,63,146,82,14]。Part B：yes_clean 0.5885（=791/1344）、yes_joint 0.5275（=709/1344）、joint_P 0.9896/joint_A1 0.0655、fam_r 0.8499。Part C：σ=3.3924、coverage 0.9671、auc_r −0.0789、auc_r_lin 2.3e−16。Part D 修正（探针）：per-t syntax P 6/11 步零样本、非零含 +4.846（t=5）；A1 5/11 步零样本、+7.951（t=5）；pooled 交叉核对 −2.307/−4.250 与 3120 一致 → 修正判决 position_confounded。

### 3. 硬伤
① token 级替换引入 OOD 破碎文本混杂（范式失效的正解释不唯一——句级连贯替换才能排除）；② Part B fork 口径 bug：abl_L30/L32 实为 P-only 672（vstack[:NP_G] 切片），混合口径表不可用，修正值由探针重算（result.json 保持 immutable，诚实记录）；③ Part D 一维布尔行选择 bug：dm[sel] 选行非元素，per-t 表与门判决无效，result.json 的 position_independent_confirmed 作废，修正判决 position_confounded；④ 367/352 对无 fact span，效应样本仅一半且 span len 集中 7–10（选择偏差未控）；⑤ content 表同数据既拟合又评估（in-sample 重构，R² 上偏——真实泛化重构会更差）；⑥ MC σ 假设高斯独立，真实残差自相关未检验；⑦ fam_r 口径混合 c0+joint 增量（方向稳健但口径不纯）；⑧ 单模型（qwen3-4b）单材料族，擦除链极性结论待跨模型。

### 4. 机制拼图更新
内部响应图谱：① 擦除链行为极性方向分解表（P 饱和 ~0.99 / A1 单调 no 向 0.150→0.098→0.066）+ L26/L31/L33 分方向重审表（A1：0.332/0.116/0.199）；② 替换响应振荡剖面（±3~5、跨对相位不对齐、span 后收敛）；③ span 内/外效应分解表。RDC 更新：① **擦除链功能重定义——其写入内容对不一致对构成假牵引（往 yes 方向），消融释放 no 先验提高判别；主写链 L26 写入维持 no 信念——写入链内部存在方向对抗结构（L26/L31 维持 vs L30/L32 假牵引），功能意义待句级范式判别**；② **振荡本体=分布现象**：确定性两成分模型失败+MC 带校准 → 逐轨迹演化含本质随机成分，3120 类别扰动表是条件均值而非轨迹预测器，AUC 振荡的复现必须在分布层面做；③ **方法论判决：token 级替换范式不可用（相位淹没+OOD 混杂），句级连贯替换是相关→因果的唯一可用桥梁**。

### 5. 3122 预注册（T4 继续，观测前冻结框架）
① **句级连贯替换**：用完整句法正确的替代事实句替换重述 span（保长度保位置），检验内容特异性的因果存在性与方向；② **方向分解行为门标准化**：所有行为消融实验必须分方向报告 first_yes/first_no 并设分方向门（P 侧饱和使混合口径失效——门定义升级）；③ **分布级重构**：放弃确定性轨迹重构，改拟合单步转移分布 p(m(t+1)|m(t),cls)（分位数回归/混合密度），检验 AUC 振荡能否从分布层面重现；④ **L26/L31/L33 写入内容读出**：分方向消融差向量投影到 w_dn/w_fam 与类别轴，定位主写链写入的语义内容。具体门在 3122 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3121/omega_p119_repl_causality_erase_polarity_recon/`（result.json、design_seal.json、run_log.txt、p119_readout.npz）；脚本 `tests/glm5/phase3121_omega_p119_repl_causality_erase_polarity_recon.py`；探针 `tests/gpt5_temp/p3121_probe2.py`、`p3121_probe3.py`。
'''
    sec = sec.replace('[[NOW]]', '[' + NOW + ']')
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    _memo_delta = len(sec)
    o.append('memo +%d chars (Phase 3121)'
             % len(sec))
else:
    _memo_delta = 0
    o.append('memo already appended')

# ---------- 4. workspace logs (x2) ----------
line_exp = ('- Phase 3121 Omega-P119 (T4 fourth '
            'phase: counterfactual replacement + '
            'erase-chain joint polarity + two-'
            'component reconstruction, qwen3-4b, '
            '669.4s): verdict ' + V + '. (A) '
            'Token-level replacement paradigm '
            'FAILS: E_10 -2.53/-1.28 (gate +0.10) '
            'and E_31 +3.22/+1.86 (gate -0.05) '
            'BOTH opposite to preregistration; '
            'probe: replacement responses are '
            '+/-3~5 oscillations with misaligned '
            'phases swamping ~2-sized condition '
            'means; inside-span means separate (c1 '
            '-1.66/c2 +1.43/c3 +1.89) but after-'
            'span converges (-3.41/-2.27/-0.73) -> '
            'content_nonspecific = paradigm '
            'failure, not content independence. '
            '(B) Joint L30+L32 yes 0.5885->0.5275 '
            '(-6.1pp, behavioral); fam r 0.8499 '
            'robust; fork-slice bug (P-only '
            'abl_L30/L32) exposed DIRECTION SPLIT: '
            'clean_A1 0.1801 -> L30 0.1503/L32 '
            '0.0982/joint 0.0655 (toward correct '
            'no, P saturated) -> erase write = '
            'assertion-pull, ablation IMPROVES '
            'discrimination; 3118 re-split: L26 '
            '+7.6pp entirely from A1 (0.180->'
            '0.332 yes-ward). (C) Deterministic '
            'two-component reconstruction fails '
            '(R2 0.128, auc_r -0.079) but MC '
            'coverage 0.967 -> oscillation is '
            'DISTRIBUTIONAL, conditional-mean '
            'decomposition only. NEXT 3122: '
            'sentence-level coherent replacement + '
            'direction-split gates + distribution-'
            'level reconstruction + write-content '
            'readout.\n')
line_clo = ('- Phase 3121 closeout finished: '
            'five-write chain ok (ledger n=258 '
            'l14=226 sha=%SHA8%, MEMO +%MEMOC% '
            'chars, dual wlog, MEMORY.md update); '
            'disk verify next. Two implementation '
            'bugs honestly recorded (fork caliber '
            'P-only; Part D row-selection), '
            'corrected verdicts via probes, '
            'result.json immutable.\n')
try:
    led2 = json.load(io.open(LEDGER,
                             encoding='utf-8'))
    _sha8 = led2['ledger_sha256_8']
except Exception:
    _sha8 = 'unknown'
line_clo = line_clo.replace(
    '%SHA8%', _sha8).replace(
    '%MEMOC%', str(_memo_delta))
for wdir in (WLOG_D, WLOG_C):
    for tag, line in (('exp', line_exp),
                      ('clo', line_clo)):
        wl = wdir + '\\' + '2026-09-23.md'
        try:
            prev = io.open(wl,
                           encoding='utf-8').read()
        except IOError:
            prev = ''
        marker = ('Phase 3121 Omega-P119' if tag
                  == 'exp'
                  else 'Phase 3121 closeout')
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
if 'max=3120' in mem_old:
    sha8 = '?'
    try:
        led2 = json.load(io.open(LEDGER,
                                 encoding='utf-8'))
        sha8 = led2['ledger_sha256_8']
    except Exception:
        pass
    r1o = u'## 机制链状态（3120）'
    r1n = u'## 机制链状态（3121）'
    assert mem_old.count(r1o) == 1
    mem_new = mem_old.replace(r1o, r1n)
    r2o = (u'- 3120（T4）：内容步归因+放大器行为+算子'
           u'形状。**预注册方向否定：重述步把两方向 '
           u'margin 一齐上推（P +0.62/A1 +0.23=对 A1 '
           u'信念侵蚀、novel=0 纯复述），标点/中性步才'
           u'恢复 gap（A1 syntax −4.25 最猛）**。'
           u'**L30/L32 行为角色反向于 L26（yes −1.6/'
           u'−4.2pp、partial）；采样管线 bit-exact '
           u'复现 3118**。**Δm 算子均值形状线性'
           u'（slope −0.68、R² 0.343、无曲率无门控）'
           u'——3119"非线性"=大方差内容扰动**。')
    r2n = (u'- 3121（T4）：反事实替换+擦除链极性+双源'
           u'重构。**token 级替换范式失效：响应=±3~5 '
           u'振荡淹没条件均值（E_10 −2.53/E_31 +3.22 '
           u'双反向）；span 后三条件收敛（−3.41/−2.27/'
           u'−0.73）**。**方向分解翻转行为结论：擦除链'
           u'写入=断言牵引、消融=释放 no 提高判别（A1 '
           u'first_yes 0.150/0.098/joint 0.066 vs '
           u'clean 0.180、P 侧饱和）；L26 +7.6pp 全来'
           u'自 A1（0.180→0.332）**。**重构失败 R² '
           u'0.128 但 MC 覆盖 0.967=振荡是分布现象**。\n'
           u'- 3120（T4）：重述步双向 margin 上推'
           u'（P +0.62/A1 +0.23=断言侵蚀）、标点步恢复 '
           u'gap（A1 −4.25）；L30/L32 行为反向于 L26'
           u'（−1.6/−4.2pp）；Δm 均值形状线性'
           u'（slope −0.68、R² 0.343）。')
    assert mem_new.count(r2o) == 1
    mem_new = mem_new.replace(r2o, r2n)
    r3o = (u'- 3119（T4）：答案 token 仅 t=1、振荡由'
           u'内容步驱动；补偿层特异（10 收缩 early-peak '
           u'vs L30/L32 放大 late-peak）；重写非线性 '
           u'R² 0.19、交互 +10.3pp。')
    r3n = (u'- 3119：答案 token 仅 t=1、振荡由内容步'
           u'驱动；补偿层特异（L30/L32 late-peak）；'
           u'重写 R² 0.19=方差非曲率。')
    assert mem_new.count(r3o) == 1
    mem_new = mem_new.replace(r3o, r3n)
    r4o = (u'- 3116：TOP3=L26/L33/L31；sum 单点=2.28×'
           u'联合→margin=多层调节平衡态（全移除仍留 '
           u'5.4%）；L32 贪心统计零改变→分布熵调节'
           u'（3117 精化为方向×regime 传导）。')
    r4n = (u'- 3116：TOP3=L26/L33/L31；margin=多层'
           u'调节平衡态；L32 贪心零改变→分布熵调节。')
    assert mem_new.count(r4o) == 1
    mem_new = mem_new.replace(r4o, r4n)
    r6o = (u'- 3117：成对消融 buffered（add_err_rel '
           u'1.627、cann_eff 0.02–0.995）；L32 消融 A1 '
           u'首 token 20.2% 分叉；采样传导确认。')
    r6n = (u'- 3117：成对消融 buffered；L32 消融 A1 '
           u'首 token 20.2% 分叉；采样传导确认。')
    assert mem_new.count(r6o) == 1
    mem_new = mem_new.replace(r6o, r6n)
    r5o = (u'- max=3120，下一 3121：**反事实重述替换'
           u'测试（相关性→因果性）+ 擦除链行为极性'
           u'（读出反转/联合消融）+ 振荡双源前向'
           u'重构**。')
    r5n = (u'- max=3121，下一 3122：**句级连贯替换'
           u'检验内容特异性 + 方向分解行为门标准化 + '
           u'分布级重构（转移分布拟合）+ L26/L31/L33 '
           u'写入内容读出**。')
    assert mem_new.count(r5o) == 1
    mem_new = mem_new.replace(r5o, r5n)
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars (sha8=%s)'
             % (len(mem_new), sha8))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
