# -*- coding: utf-8 -*-
"""Phase 3118 closeout (idempotent):
Ledger -> MEMO Phase 3118 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3118'
        r'\omega_p116_autoregressive_margin_'
        'trajectory')
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
assert res['verdict'] == \
    'belief_decays_in_generation|' \
    'temporal_compensation|' \
    'top_ablation_changes_behavior'
assert res['n_records'] == 2016
assert res['n_pairs'] == 672
assert res['smoke'] is False
tk = res['track']
assert abs(tk['auc0'] - 0.9809094210600907) < 1e-9
assert abs(tk['auc_last']
           - 0.6718218537414966) < 1e-9
assert abs(tk['decay'] - 0.3090875673185941) < 1e-9
ts = res['temporal_state']
assert abs(ts['ratio_mean']
           - 0.5235178059211297) < 1e-9
assert ts['gate'] == 'temporal_compensation'
cl = res['closed_loop']
assert abs(cl['L31']['ratio']
           - 1.7277062340789917) < 1e-9
assert abs(cl['L26']['ratio']
           - 1.0777458644936844) < 1e-9
bh = res['behavior']
assert abs(bh['yes_rate']['clean']
           - 0.5885416666666666) < 1e-9
assert abs(bh['yes_rate']['abl_L26']
           - 0.6644345238095238) < 1e-9
assert abs(bh['agree_P_A1']['clean']
           - 0.028273809523809524) < 1e-9
assert abs(bh['max_diff']
           - 0.0758928571428572) < 1e-9
sm = res['sampled']
assert abs(sm['seq_agree']
           - 0.33666666666666667) < 1e-9
assert abs(sm['state_ratio']
           - 0.4809525247666324) < 1e-9
assert sm['n_seq'] == 1200
chk = res['greedy_recheck']
assert chk['cleanseq_clean'] > 0.99
assert chk['L26seq_L26'] > 0.99

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3118
           for m in led['measurements']):
    claim = (
        'Omega-P116 (3118, T4 first phase: '
        'autoregressive margin trajectory, qwen3-4b, '
        '4x1344 greedy generations + 10x1344 '
        'teacher-forced trajectory readouts + 1200 '
        'sampled generations + 4x1200 sampled '
        'trajectories, 3160s) - verdict '
        'belief_decays_in_generation|'
        'temporal_compensation|'
        'top_ablation_changes_behavior.  Method: '
        'm(t) = norm(h_NL[pos0+t]).(w_yes-w_no) read '
        'at EVERY generation step t=0..12 in ONE '
        'teacher-forced forward per (sequence, '
        'condition) - causal attention makes prefix '
        'states identical to step-wise decode states '
        '(greedy recheck 0.994 own-condition, 0.99+ '
        'own-replay).  (1) BELIEF DECAYS IN '
        'GENERATION: AUC(P vs A1 margin) falls 0.981 '
        '-> 0.672 over 12 greedy steps (decay 0.309 '
        '>> 0.10 gate) with a STRONG OSCILLATION '
        '(curve 0.981, 0.559, 0.797, 0.534, 0.760, '
        '0.879, 0.972, 0.853, 0.518, 0.645, 0.658, '
        '0.712, 0.672) - the belief is not a '
        'statically held state; it is REWRITTEN by '
        'each generated token, discrimination '
        'collapsing to near-chance at steps 1/3/8 '
        'and partially recovering at steps 2/6.  '
        '(2) TEMPORAL COMPENSATION vs CLOSED-LOOP '
        'AMPLIFICATION (the phase key decomposition): '
        'on FROZEN clean sequences the TOP3 ablation '
        'state effect SHRINKS along steps (tail/head '
        'ratio 0.447/0.531/0.593 for L26/L33/L31, '
        'mean 0.524 <= 0.7 gate) - downstream '
        'computation pulls the ablation-induced '
        'margin offset back over generation steps; '
        'but the CLOSED-LOOP total effect (ablated '
        'model on its OWN generations) AMPLIFIES '
        '(1.078/1.221/1.728, L31 +73%) - '
        'compensation acts on the state trajectory '
        'but NOT on the behavioral snowball of '
        'token-level divergence feeding back through '
        'context.  (3) TOP-CAUSAL ABLATION CHANGES '
        'BEHAVIOR: ablating L26 shifts greedy '
        'yes-family rate 58.85% -> 66.44% (+7.59pp, '
        'gate 5%) and makes P/A1 generations more '
        'similar (agree 0.028 -> 0.071) - the '
        'negative causal leader suppresses yes; '
        'contrast 3116 where ablating the L32 erase '
        'changed behavior ZERO - causal rank '
        'predicts behavioral leverage.  Sampled '
        'regime (300 pairs, K=2, temp 0.7): '
        'seq_agree clean-vs-abl_L26 0.337, yes rate '
        '59.42% -> 66.42%, state ratio 0.481 '
        '(consistent with greedy).  CAVEATS: (i) '
        '12-step window only, no long-horizon '
        'extrapolation; (ii) oscillation tokens not '
        'yet attributed (which generated tokens '
        'drive AUC recovery/collapse - 3119); (iii) '
        'closed-loop amplification conflates state '
        'x content interaction; (iv) single model.  '
        'NEXT 3119: token-level attribution of the '
        'AUC oscillation + time x layer write map '
        '(conditional ablation at each trajectory '
        'step) + one-step predictability of '
        'margin(t) from margin(t-1) and g_t.')
    meas = {
        'meas_id': 'meas3118_omega_p116_'
                   'autoregressive_margin_'
                   'trajectory',
        'phase': 3118,
        'claim': claim,
        'verdict': 'belief_decays_in_generation|'
                   'temporal_compensation|'
                   'top_ablation_changes_behavior',
        'anchors': 'design_seal.json frozen before '
                   'computation: track gate 0.10/0.03 '
                   'on AUC decay; temporal gate '
                   '0.7/1.3 on tail/head ratio; '
                   'behav gate 0.05/0.02 on yes-'
                   'rate shift; greedy recheck '
                   'reported (own-condition 0.99+)',
        'artifacts': {
            'result': 'phase3118/omega_p116_'
                      'autoregressive_margin_'
                      'trajectory/result.json',
            'seal': 'phase3118/omega_p116_'
                    'autoregressive_margin_'
                    'trajectory/design_seal.json',
            'readout': 'phase3118/omega_p116_'
                       'autoregressive_margin_'
                       'trajectory/traj_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 3160s: 4x1344 greedy gen '
                '+ 10x1344 trajectory forwards + '
                '1200 sampled gen + 4800 sampled '
                'trajectory forwards); teacher-'
                'forced one-forward trajectory '
                'readout; behavior yes_rate clean '
                '0.588542 bit-identical to 3116',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3118_omega_p116_autoregressive_margin_'
        'trajectory')
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

# ---------- MEMO Phase 3118 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3118:' not in memo:
    sec = u'''## Phase 3118: Ω-P116 自回归 margin 轨迹（T4 首Phase）——**信念随生成衰减：AUC(P/A1 margin) 0.981→0.672，强振荡（0.52–0.97），每步被生成 token 重写**；**状态效应时间补偿（tail/head=0.52）vs 闭环放大（1.08–1.73）：补偿作用于状态轨线、不作用于行为雪球**；**TOP1 消融改变行为：L26 消融 yes 率 +7.6pp（vs 3116 L32 零改变——因果秩预测行为杠杆）** [[NOW]]

**性质**：T4 第 1 Phase。qwen3-4b BF16，3160s。方法：m(t)=norm(h_NL[pos0+t])·(w_yes−w_no) 在**一次 teacher-forced 前向**中读出全部 13 个生成步（causal attention 下前缀状态=逐步解码状态；greedy recheck own 条件 0.994）。数据：4 条件（clean/L26/L33/L31 消融）× 1344 贪心生成 + 10×1344 轨迹前向（clean 序列 × 4 条件 + 各消融序列 × {own, clean}）+ 1200 采样生成（300 对 × P/A1 × {clean, abl_L26} × K=2，seed 同 3117 规则）+ 4800 采样轨迹前向。预注册三门（design_seal 先冻结）：track decay 0.10/0.03；temporal ratio 0.7/1.3；behav 0.05/0.02。

### 1. 三大发现（重复三遍）
1. **信念随生成衰减且强振荡**：AUC(P vs A1 的 margin 判别) 0.981→0.672（decay 0.309 ≥0.10 门→ belief_decays_in_generation）；曲线 0.981/0.559/0.797/0.534/0.760/0.879/0.972/0.853/0.518/0.645/0.658/0.712/0.672——第 1/3/8 步跌至近随机（0.52–0.56），第 2/6 步部分恢复（0.80/0.97）。**信念不是静态维持的状态：每个生成 token 都在重写它，判别信息在近随机与近完美之间振荡。margin=每步被重写的受控量。衰减但不清零。**
2. **状态补偿 vs 闭环雪球（本 Phase 关键分解）**：固定 clean 生成序列、仅切换消融条件（纯状态效应）——TOP3 消融的 |Δm| 随生成步**收缩**（tail/head ratio：L26 0.447、L33 0.531、L31 0.593，mean 0.524 ≤0.7 门→ temporal_compensation）：下游计算在后续步把消融偏移拉回。但**闭环总效应**（消融模型在自己的生成上）**放大**（1.078/1.221/1.728，L31 +73%）：消融改变生成 token → 不同上下文 → 更大效应。**补偿作用于状态轨线，不作用于行为分叉的雪球。时间维度补偿与空间维度补偿（3116/3117）互补且机制分离。**
3. **因果秩预测行为杠杆**：L26（因果 TOP1）消融使贪心 yes 率 58.85%→66.44%（+7.59pp ≥5% 门）、P/A1 生成趋同（agree 0.028→0.071）；L33 +1.0pp、L31 −3.3pp。**对比 3116：L32 擦除层消融行为零改变**——空间因果图上的排名预测行为杠杆，擦除层是分布调节器、L26 是行为通路。采样 regime 一致（abl_L26：yes 59.42%→66.42%，seq_agree 0.337，state_ratio 0.481）。

### 2. 关键数值
AUC 曲线见上；temporal ratio TOP3 = 0.447/0.531/0.593（mean 0.524）；closed-loop ratio = 1.078/1.221/1.728；yes_rate clean/L26/L33/L31 = 0.5885/0.6644/0.5982/0.5558（clean 与 3116 bit 级一致 0.588542）；agree P/A1 = 0.0283/0.0714/0.0446/0.0089；greedy recheck own 0.989–0.994、cross 0.862–0.967（消融放大 KV-cache 与全序列前向的核差异，预期内）；采样 state_ratio 0.481。

### 3. 硬伤
① 12 步窗口，无长程外推（雪球是否饱和未知）；② AUC 振荡的 token 归因未做（哪些生成 token 导致恢复/骤降——3119）；③ 闭环放大混合状态×内容交互，未分解；④ 采样仅 abl_L26 单条件；⑤ 单模型。

### 4. 机制拼图更新
内部响应图谱新增：**生成步维度的信念轨迹地图**（13 点 AUC 振荡曲线 + 消融效应 tail/head 分解）+ **状态/行为双通路口**（补偿网络作用于状态；行为通过 token 反馈雪球放大）。RDC 更新：内部信念状态是**每步重写的受控量**——"设定值跟踪"发生在状态空间（tail/head 0.52），而自由生成中受控量的设定值本身被输出 token 调制（振荡）；行为杠杆由因果写入层的排名预测（L26 ≫ L32）。对 AGI 理论：语言生成的自洽性不靠静态信念保持，靠**每步重写 + 状态空间补偿**；行为漂移通过上下文反馈形成雪球，这是与状态补偿正交的第二动力学。

### 5. 3119 预注册（T4 继续，观测前冻结框架）
① **振荡 token 归因**：按生成步 t 分解 AUC 振荡来源（g_t 的 yes/no 语义与 token 身份），检验"yes/no token 步驱动恢复、内容 token 步驱动塌缩"假说；② **时间 × 层写入地图**：对轨迹各步做 L14–L35 条件化消融扫描（写入链的时间结构）；③ **单步可预测性**：margin(t) 由 margin(t−1) 与 g_t 的 embedding 线性预测的精度（重写算子的线性度）；具体门在 3119 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3118/omega_p116_autoregressive_margin_trajectory/`（result.json、design_seal.json、run_log.txt、traj_readout.npz）；脚本 `tests/glm5/phase3118_omega_p116_autoregressive_margin_trajectory.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3118)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3118 Omega-P116 (T4 first phase: '
          'autoregressive margin trajectory, qwen3-4b, '
          '3160s): verdict belief_decays_in_generation|'
          'temporal_compensation|'
          'top_ablation_changes_behavior. One '
          'teacher-forced forward reads the full '
          '13-point m(t) trajectory (greedy recheck '
          '0.994). (1) AUC(P/A1 margin) 0.981->0.672 '
          'over 12 greedy steps with STRONG '
          'oscillation (0.518..0.972; near-chance at '
          'steps 1/3/8, recovery at 2/6) - belief is '
          'REWRITTEN by every generated token. (2) '
          'State effect on frozen clean sequences '
          'SHRINKS along steps (TOP3 tail/head 0.52 '
          '-> temporal_compensation) but closed-loop '
          'total effect on ablated model OWN '
          'generations AMPLIFIES (1.08/1.22/1.73) - '
          'compensation acts on the state trajectory, '
          'not on the token-feedback snowball. (3) '
          'L26 ablation shifts greedy yes-rate '
          '58.85->66.44pp +7.6 and P/A1 agree '
          '0.028->0.071 (vs 3116 L32 ZERO change) - '
          'causal rank predicts behavioral leverage. '
          'Sampled regime consistent (yes +7.0pp, '
          'state_ratio 0.481, seq_agree 0.337). NEXT '
          '3119: oscillation token attribution + '
          'time x layer write map + one-step '
          'predictability of margin(t).\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3118 Omega-P116' not in prev:
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
if 'max=3118' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3117）\n'
        '- 3117：成对消融 buffered|采样传导确认。'
        '对消对 add_err_rel=1.627（联合比线性预测'
        '更负，cann_eff 0.02–0.995）；同负亚可加 '
        '0.805 vs 同正超可加 1.149→**平衡不可加'
        '组合，缓冲不对称**。**行为传导方向不对称：'
        'L32 消融 A1 首 token 20.2% 分叉 vs P '
        '0.15%，yes-swap=0**；采样 seq_agree 0.444、'
        'yes −4pp→熵调节=regime 依赖传导。'
        'Overlap 4 项=0。\n',
        '## 机制链状态（3118）\n'
        '- 3118（T4）：自回归轨迹。**AUC(m 判别) '
        '0.981→0.672 强振荡（0.52–0.97，每步被'
        '生成 token 重写）**；**状态效应收缩 '
        'tail/head 0.52（时间补偿）vs 闭环放大 '
        '1.08–1.73（token 反馈雪球）——补偿作用于'
        '状态、不作用于行为**；L26 消融 yes +7.6pp '
        '（vs L32 零改变）→**因果秩预测行为杠杆**。\n'
        '- 3117：成对消融 buffered（add_err_rel '
        '1.627、cann_eff 0.02–0.995）；L32 消融 A1 '
        '首 token 20.2% 分叉；采样传导确认。\n')
    mem_new = mem_new.replace(
        'max=3117', 'max=3118').replace(
        '下一 3118（T4）：**自回归 margin 轨迹——'
        '逐生成步读出平衡态，贪心+采样双 regime，'
        '含/不含 L26/L33/L31 消融**。',
        '下一 3119：**振荡 token 归因 + 时间×层'
        '写入地图 + margin(t) 单步可预测性**。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
