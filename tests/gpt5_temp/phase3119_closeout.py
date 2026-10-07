# -*- coding: utf-8 -*-
"""Phase 3119 closeout (idempotent):
Ledger -> MEMO Phase 3119 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3119'
        r'\omega_p117_oscillation_attribution_'
        'writemap_linearity')
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
assert V == 'attribution_selection_dominant|' \
    'compensation_layer_specific|' \
    'rewrite_nonlinear', V
assert res['smoke'] is False
assert res['n_records'] == 2016
assert res['n_pairs'] == 672
assert res['n_pairs_partb'] == 672
pa = res['part_a']
assert pa['verdict'] == \
    'attribution_selection_dominant'
assert abs(pa['sel_auc']['mean']
           - 0.8365936518210295) < 1e-9
assert pa['sel_auc']['verdict'] == \
    'self_selection_present'
assert abs(pa['cls_stats']['A1']['no']['mean_dm']
           - 8.765673626986416) < 1e-9
assert abs(pa['combo_table']['1_2']['mean_dgap']
           - (-9.199765738779611)) < 1e-9
assert pa['step_table'][5]['auc_t'] == \
    0.9720339958900227
assert pa['selfcheck_curve_rel_max'] == 0.0
pb = res['part_b']
assert pb['verdict'] == \
    'compensation_layer_specific'
assert pb['integrity_confirmed'] is True
assert pb['overlap']['L26']['dP'] == 0.0
assert pb['overlap']['L26']['dA'] == 0.0
assert pb['overlap']['L31']['dP'] == 0.0
assert pb['overlap']['L33']['dP'] == 0.0
wm = pb['write_map']
assert abs(wm['L26']['ratio']
           - 0.44719892847360415) < 1e-12
assert abs(wm['L33']['ratio']
           - 0.5306656233054948) < 1e-12
assert abs(wm['L31']['ratio']
           - 0.59268886598429) < 1e-12
assert abs(wm['L30']['ratio']
           - 1.3773594868400552) < 1e-12
assert abs(wm['L32']['ratio']
           - 1.1479748121428726) < 1e-12
assert abs(wm['L35']['ratio']
           - 0.9193717473737328) < 1e-12
assert wm['L30']['peak_t'] == 9
assert wm['L32']['peak_t'] == 5
assert pb['l35_verdict'] == 'rebuild_layer_absent'
assert pb['amplifiers_non_top3'] == [30, 32]
pc = res['part_c']
assert pc['verdict_lin'] == 'rewrite_nonlinear'
assert pc['verdict_tok'] == \
    'token_injection_untracked'
assert pc['verdict_int'] == \
    'multiplicative_component_present'
assert abs(pc['r2_full']
           - 0.18936137607996972) < 1e-12
assert abs(pc['d_r2_token']
           - 9.915125662518509e-05) < 1e-12
assert abs(pc['d_r2_int']
           - 0.10251094437756858) < 1e-12
assert abs(pc['r2_persist']
           - (-0.2453164373288632)) < 1e-12

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3119
           for m in led['measurements']):
    claim = (
        'Omega-P117 (3119, T4 second phase: '
        'oscillation mechanism three-pronged test, '
        'qwen3-4b, 22x1344 teacher-forced layer-'
        'ablation replays + offline attribution + '
        'OLS predictability, 936s) - verdict '
        'attribution_selection_dominant|'
        'compensation_layer_specific|'
        'rewrite_nonlinear.  Part A (token '
        'attribution, offline on frozen 3118 '
        'trajectories): answer tokens appear ONLY '
        'at t=1 (yesP=0.997, noA1=0.818); t>=2 is '
        '100% content tokens, so the preregistered '
        '"yes/no-token steps drive AUC recovery" '
        'hypothesis is DIRECTIONALLY REFUTED - '
        'recovery happens at content-token steps '
        '(t=2/4/5/6, AUC up to 0.972), collapse at '
        'the answer step (t=1, AUC 0.559) and '
        'content steps (t=3/8).  Answer-token '
        'relaxation: A1 consuming no shifts margin '
        '+8.77 (post-answer release, not belief '
        'reinforcement); A1 consuming its wrong '
        'yes shifts -1.64 (self-correction).  '
        'Self-selection strong: AUC(m(t-1)|yes vs '
        'other)=0.837.  Part B (time x layer write '
        'map, 22 layers L14-L35, overlap L26/L31/'
        'L33 bit-exact 0.00e+00 vs 3118): '
        'compensation is LAYER-SPECIFIC - 10 '
        'shrinkers (L24 0.364 strongest, L19 0.439, '
        'L26 0.447, L22 0.474, L18 0.484, L16 '
        '0.488, L33 0.531, L14 0.548, L31 0.593, '
        'L23 0.628; nearly all early-peak peak_t=0)'
        ' vs exactly 2 amplifiers L30 1.377 and '
        'L32 1.148 (the 3113-3116 ERASE layer and '
        'its neighbor) with LATE peaks (L30 '
        'peak_t=9, L32 peak_t=5) - the erase '
        'chain CONTINUES to de-belief the margin '
        'during late generation.  L35 rebuild '
        'hypothesis rejected (ratio 0.919 < 1.0) '
        'though L34/L35 have the largest absolute '
        'amplitude (3.5-6.2) = large slow-relaxing '
        'writers.  Part C (one-step predictability,'
        ' 8064 train / 8064 test by pair parity): '
        'R2 persist=-0.245, base(m(t-1),is_P)=0.189'
        ', +e_yes only +0.0001 (embedding-side '
        'readout projection does NOT carry the '
        'rewrite, despite correct sign: yes +0.95 '
        'no -0.60), +interaction m(t-1)*e_yes '
        '+0.103 - the rewrite operator is a '
        'STATE-DEPENDENT NONLINEAR map, not '
        'additive token injection.  CAVEATS: '
        'preregistered A-GAP cross combos '
        'structurally empty (n=1/0) - answer '
        'tokens concentrated at t=1; decile '
        'semantic channel direction-consistent '
        '(P 3/3, A1 2/2) but underpowered (<8 '
        'valid bins); content-step token identity '
        'not yet attributed; single model.  '
        'NEXT 3120: content-step oscillation '
        'attribution (fact-restatement vs neutral '
        'steps) + L30/L32 amplifier behavioral '
        'transmission + quantitative shape of the '
        'mean-reversion rewrite operator.')
    meas = {
        'meas_id': 'meas3119_omega_p117_'
                   'oscillation_attribution_'
                   'writemap_linearity',
        'phase': 3119,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen before '
                   'computation: A decile gate 8 '
                   'bins/70pct, A-SEL 0.65/0.55, '
                   'A-GAP pooled sign + 70pct step '
                   'consistency; B overlap bit-exact '
                   '== 0.0, B-MAP 0.7/1.1, B-L35 '
                   '1.0; C-TOK 0.10/0.02, C-LIN '
                   '0.7/0.4, C-INT 0.05',
        'artifacts': {
            'result': 'phase3119/omega_p117_'
                      'oscillation_attribution_'
                      'writemap_linearity/'
                      'result.json',
            'seal': 'phase3119/omega_p117_'
                    'oscillation_attribution_'
                    'writemap_linearity/'
                    'design_seal.json',
            'readout': 'phase3119/omega_p117_'
                       'oscillation_attribution_'
                       'writemap_linearity/'
                       'wmap_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 936s: 22x1344 layer-'
                'ablation teacher-forced replays); '
                'Part A/C offline on 3118 frozen '
                'npz; L26/L31/L33 replays bit-exact '
                'vs 3118 (dP=dA=0.00e+00)',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3119_omega_p117_oscillation_attribution'
        '_writemap_linearity')
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

# ---------- MEMO Phase 3119 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3119:' not in memo:
    sec = u'''## Phase 3119: Ω-P117 振荡机制三叉验证（T4 第2Phase）——**答案 token 只在 t=1 出现：AUC 振荡由内容 token 步驱动，"yes/no token 步驱动恢复"预注册假说方向被否定**；**时间补偿层特异：10 收缩层 vs L30/L32 双放大层（擦除层效应随生成步增强、峰值在后期）——L35 重建假说否定**；**重写算子高度非线性：单步 R² 仅 0.19、embedding 直接注入零贡献（d_tok≈0.0001）、交互项 +10.3pp** [[NOW]]

**性质**：T4 第 2 Phase，3118 MEMO 第 5 节预注册、门在 seal 观测前冻结。qwen3-4b BF16，936s。A 部分（离线，3118 冻结 npz 的 16128 个 (对,步,方向) 样本）：生成 token 按 yes/no/other 分类做 Δm 归因 + m(t-1) 十分位控制自选择 + Δgap 类组合表；B 部分（GPU）：clean 贪心序列在 L14–L35 每层单层 MLP 消融下重放（22 条件 × 1344 teacher-forced 前向），**L26/L31/L33 与 3118 overlap 全部 0.00e+00（bit-exact，完整性确认）**；C 部分：m(t) ~ m(t-1) + e_yes(g_t) + is_P 按对奇偶分割 OLS（8064 训练/8064 测试），e_yes=emb(g_t)·(w_yes−w_no)。

### 1. 三大发现（重复三遍）
1. **答案 token 只在 t=1 出现，AUC 振荡由内容 token 步驱动**：yesP(t=1)=0.997、noA1(t=1)=0.818；t≥2 两组 100% 内容 token。塌缩：t=1（AUC 0.559——**A1 消费 no 后 dm=+8.77 大幅回正=答案后弛豫**，非信念强化；A1 错答 yes 后 dm=−1.64 自纠）、t=3/8（内容步）；恢复：t=2/4/5/6（内容步，AUC 最高 0.972）——预注册假说"yes/no 步驱动恢复、内容步驱动塌缩"**方向否定**。自选择强：AUC(m(t-1)|yes vs other)=0.837 → self_selection_present。decile 桶内 yes 步 Δm 方向全正（P 3/3、A1 2/2）但有效桶 <8（功效不足）。
2. **时间补偿层特异（写入链的层分工）**：TOP3 复现（0.447/0.531/0.593，bit 级）+ **10 收缩层**（L24 0.364 最强、L19 0.439、L22 0.474、L18 0.484、L16 0.488、L33 0.531、L14 0.548、L31 0.593、L23 0.628、L26 0.447，几乎全部 early-peak peak_t=0）vs **仅 2 放大层：L30 1.377、L32 1.148（=3113–3116 擦除层及其邻居），峰值在后期（L30 peak_t=9、L32 peak_t=5）** → compensation_layer_specific。**擦除链的真实时间角色=生成后期持续去信念化，而非一次性擦除**。L35 重建假说否定（ratio 0.919<1.0），但 L34/L35 绝对幅度全图最大（3.5–6.2）=大幅值慢弛豫写入者。
3. **重写算子高度非线性、embedding 直接注入不参与**：R² persist=−0.245（m(t) 比 m(t-1) 的均值预测还差——单步剧烈重写）；R² base=0.189；加 e_yes 只 +0.0001（token_injection_untracked）——尽管 e_yes 方向正确（yes token +0.95、no token −0.60），embedding 在读出方向上的投影对重写无预测力（3107 embedding/unembed 近正交的生成侧新证据）；交互项 m(t-1)×e_yes 再 +10.3pp（multiplicative_component_present）——重写是**状态依赖的非线性算子**（边际效应取决于当前信念），不是加性 token 注入。

### 2. 关键数值
Δgap 组合表：(yes,no) n=548 Δgap=−9.20（答案步 gap 收缩主项）；(yes,yes) n=121 Δgap=+1.01（双方同答 yes 时 gap 反而扩大）；(other,other) n=7392 Δgap=+0.12（内容步净小幅恢复）；预注册交叉组合 (yes,other)/(other,yes) 结构性稀少（n=1/0）。写入地图 22 层完整谱见 write_map；重放 recheck 谱 0.86–0.97；Part C spearman(pred,true)=0.455。

### 3. 硬伤
① A-GAP 预注册交叉组合结构性稀少（n=1/0），归因门功效不足（设计教训：答案 token 集中于 t=1 使交叉组合不存在——预注册时未预见答案结构）；② decile 语义通道方向一致但有效桶 <8，未达预注册门；③ 内容步振荡的 token 身份归因未做（t=2–12 未标注语义——3120）；④ 放大器仅 2 层，依赖单层消融可加性假设；⑤ 单模型、12 步窗口。

### 4. 机制拼图更新
内部响应图谱新增：**时间×层写入地图**（22 层 × 13 步 |Δm| 谱：收缩层 early-peak、放大层 late-peak 的清晰层分工）+ **答案后弛豫现象**（A1 消费 no 后 margin +8.77 回正）。RDC 更新：① 信念重写是**状态依赖非线性算子**（交互 +10.3pp、embedding 注入零）——"每步重写"不是线性滤波而是门控式计算；② 补偿网络有层分工：**主写链（L14–L28）随生成收缩（状态补偿），擦除链（L30–L32）随生成增强（去信念化）**——3113"擦除相"获得时间维度新解释，与 3118 行为雪球指向同一机制家族：生成后期模型主动降低信念判别；③ 答案 token 的作用是"弛豫触发器"而非"信念强化器"。

### 5. 3120 预注册（T4 继续，观测前冻结框架）
① **内容 token 步振荡归因**：对 t=2–12 内容 token 按身份/位置标注（句首/句中/事实重述/重复问句），检验"事实重述步驱动恢复、中性步驱动塌缩"；② **放大器行为传导**：L30/L32 消融的贪心+采样行为测试（yes 率、P/A1 agree——放大层是否像 L26 一样是行为杠杆）；③ **回归均值算子定量形状**：Δm 对 m(t-1) 分箱曲线的非参数拟合（分段线性/门控），给出重写算子的经验形式；具体门在 3120 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3119/omega_p117_oscillation_attribution_writemap_linearity/`（result.json、design_seal.json、run_log.txt、wmap_readout.npz）；脚本 `tests/glm5/phase3119_omega_p117_oscillation_attribution_writemap_linearity.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3119)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3119 Omega-P117 (T4 second phase: '
          'oscillation mechanism three-pronged test, '
          'qwen3-4b, 936s): verdict '
          'attribution_selection_dominant|'
          'compensation_layer_specific|'
          'rewrite_nonlinear. Overlap L26/L31/L33 '
          'bit-exact 0.00e+00 vs 3118. (A) Answer '
          'tokens ONLY at t=1 (yesP=0.997, noA1='
          '0.818); t>=2 is 100% content tokens; '
          'recovery at content steps (t=2/4/5/6 up '
          'to AUC 0.972), collapse at answer step '
          '(t=1: A1 consuming no shifts margin '
          '+8.77 = post-answer relaxation, not '
          'belief reinforcement) and content steps '
          't=3/8 - preregistered yes/no-recovery '
          'hypothesis DIRECTIONALLY REFUTED; '
          'self-selection AUC 0.837. (B) Time x '
          'layer write map: compensation is LAYER-'
          'SPECIFIC - 10 shrinkers (L24 0.364 '
          'strongest, early-peak) vs exactly 2 '
          'amplifiers L30 1.377 / L32 1.148 (erase '
          'chain) with LATE peaks (9/5) - erase '
          'chain keeps de-believing during late '
          'generation; L35 rebuild rejected (0.919) '
          'but largest absolute amplitude (3.5-6.2).'
          ' (C) Rewrite operator: R2 persist -0.245,'
          ' base 0.189, e_yes +0.0001 (embedding '
          'injection does NOT carry rewrite), '
          'interaction +0.103 -> STATE-DEPENDENT '
          'NONLINEAR map. NEXT 3120: content-step '
          'token attribution + L30/L32 behavioral '
          'transmission + mean-reversion operator '
          'shape.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3119 Omega-P117' not in prev:
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
if 'max=3119' not in mem_old:
    mem_new = mem_old.replace(
        u'## 机制链状态（3118）\n'
        u'- 3118（T4）：自回归轨迹。**AUC(m 判别) '
        u'0.981→0.672 强振荡（0.52–0.97，每步被'
        u'生成 token 重写）**；**状态效应收缩 '
        u'tail/head 0.52（时间补偿）vs 闭环放大 '
        u'1.08–1.73（token 反馈雪球）——补偿作用于'
        u'状态、不作用于行为**；L26 消融 yes +7.6pp '
        u'（vs L32 零改变）→**因果秩预测行为杠杆**。\n',
        u'## 机制链状态（3119）\n'
        u'- 3119（T4）：振荡机制三叉。**答案 token '
        u'仅 t=1，AUC 振荡由内容 token 步驱动'
        u'（假说方向否定）；A1 吃 no 后 +8.77=答案'
        u'后弛豫；自选择 AUC 0.84**。**补偿层特异：'
        u'10 收缩层（L24 0.36 最强、early-peak）vs '
        u'L30/L32 放大（1.38/1.15、late-peak）——'
        u'擦除链=生成后期持续去信念化**；L35 重建'
        u'否定。**重写算子非线性：R² 0.19、embedding '
        u'注入零贡献、交互 +10.3pp**。\n'
        u'- 3118（T4）：AUC 0.981→0.672 强振荡；'
        u'状态补偿 0.52 vs 闭环放大 1.08–1.73；'
        u'L26 消融 yes +7.6pp（因果秩预测行为'
        u'杠杆）。\n')
    mem_new = mem_new.replace(
        u'max=3118', u'max=3119').replace(
        u'下一 3119：**振荡 token 归因 + 时间×层'
        u'写入地图 + margin(t) 单步可预测性**。',
        u'下一 3120：**内容 token 步振荡归因'
        u'（事实重述 vs 中性步）+ L30/L32 放大器'
        u'行为传导 + 回归均值算子定量形状**。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
