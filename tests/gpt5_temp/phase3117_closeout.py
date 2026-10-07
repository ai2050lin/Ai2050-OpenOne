# -*- coding: utf-8 -*-
"""Phase 3117 closeout (idempotent):
Ledger -> MEMO Phase 3117 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3117'
        r'\omega_p115_pair_cancellation_sampling')
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
    'cancellation_pairwise_buffered|' \
    'sampled_consequence_confirmed'
assert res['n_records'] == 2016
assert res['n_samp_pairs'] == 300
assert res['smoke'] is False
cs = res['cancel_summary']
assert abs(cs['cancel_add_err_rel_mean']
           - 1.6267) < 1e-3
assert abs(cs['same_neg_add_frac_mean']
           - 0.8053) < 1e-3
assert abs(cs['same_pos_add_frac_mean']
           - 1.1487) < 1e-3
for k in ('L24', 'L35'):
    assert res['overlap_check'][k]['abs_diff'] == 0.0
assert res['js_overlap_check']['base_diff'] == 0.0
assert res['js_overlap_check']['abl_diff'] == 0.0
ga = res['greedy_control']
assert abs(ga['full_agree_P'] - 0.67) < 1e-9
assert abs(ga['full_agree_A1'] - 0.603333) < 1e-5
assert ga['div_yes_swap_rate_P'] == 0.0
assert ga['div_yes_swap_rate_A1'] == 0.0
gfa = res['greedy_first_all']
assert abs(gfa['P'] - 0.998512) < 1e-5
assert abs(gfa['A1'] - 0.797619) < 1e-5
sa = res['sampled']
assert abs(sa['seq_agree'] - 0.444333) < 1e-5
assert sa['n_seq'] == 3000
assert abs(sa['yes_rate_clean']
           - 0.585667) < 1e-5
assert abs(sa['yes_rate_abl'] - 0.546) < 1e-9
assert max(res['selfcheck_rel'].values()) < 0.01

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3117
           for m in led['measurements']):
    claim = (
        'Omega-P115 (3117, pair-cancellation ablation '
        'test of the equilibrium model + sampled '
        'behavioral transmission, qwen3-4b, 15x2016 '
        'forwards + JS 2x1344 + 1200 greedy + 6000 '
        'sampled generations (temp 0.7, K=5, seed '
        'crc32(pk|dir|rep) shared across conditions), '
        '3460s) - verdict '
        'cancellation_pairwise_buffered|'
        'sampled_consequence_confirmed.  Overlap '
        'reproduction BIT-EXACT on 4 anchors: L24/L35 '
        'dmp_rel vs 3116 diff 0.00e+00, JS(P||A1) '
        'base/abl means vs 3115 diff 0.00e+00 '
        '(float64 full-vocab log_softmax pipeline '
        'exactly reproducible).  (1) PAIR LEVEL IS '
        'BUFFERED, NOT ADDITIVE: 7 cancel pairs '
        '(one negative + one positive single from '
        'the 3116 map) give mean add_err_rel=1.627 '
        '>> 0.50 gate - joint effects deviate from '
        'the sum of frozen singles by 163% on '
        'average, usually MORE NEGATIVE than '
        'predicted (L26+L24: joint -0.462 vs sum '
        '-0.118, add_err_rel 2.93; L31+L35: 3.78); '
        'cann_eff varies 0.022..0.995 per pair - '
        'removing a (neg,pos) regulator pair '
        'partially collapses the remaining network '
        'compensation instead of leaving the '
        'predicted net.  Same-sign pairs: negative '
        'pairs sub-additive (add_frac mean 0.805), '
        'the single positive pair SUPER-additive '
        '(L24+L35: joint +0.830 vs sum +0.722, '
        'add_frac 1.149) - asymmetric buffering.  '
        '(2) DIRECTION-ASYMMETRIC BEHAVIORAL '
        'TRANSMISSION: ablating the L32 erase '
        'changes A1-direction greedy first tokens '
        'for 20.2% of all 672 pairs (P direction '
        'only 0.15%); on the 300-pair generation '
        'sample full 12-token greedy sequences '
        'agree clean-vs-abl only 67.0% (P) / 60.3% '
        '(A1) while yes-swap rate is exactly 0 '
        '(divergence is NOT yes-family surface '
        'substitution) - 3116 zero-change behavior '
        'statistics were masking real token-level '
        'divergence in the erase-active direction.  '
        '(3) SAMPLING CONFIRMS TRANSMISSION: '
        'same-seed temperature-0.7 sampled 12-token '
        'sequences agree clean vs abl in only '
        '44.4% (1333/3000), first-token empirical '
        'L1 distance 0.087, yes-family rate drops '
        '58.57%->54.60% (-4.0pp) - the '
        'distribution-shape change reaches sampled '
        'behavior; greedy yes/no semantics stay '
        'robust while surface tokens move.  '
        'Selfchecks 1.6e-3..2.1e-3.  CAVEATS: (i) '
        'add_err_rel uses frozen singles measured '
        'under full compensation - deviations '
        'conflate released buffering with true '
        'pair interactions; (ii) 12 selected pairs, '
        'not all 276; (iii) sampling statistics on '
        '300 pairs / K=5; (iv) single model.  NEXT '
        '3118 (T4): autoregressive margin trajectory '
        '- read the m=margin control variable at '
        'every generation step under greedy and '
        'temp-0.7 sampling, with/without top causal '
        'layers ablated; preregister gates in '
        '3118 seal.')
    meas = {
        'meas_id': 'meas3117_omega_p115_'
                   'pair_cancellation_sampling',
        'phase': 3117,
        'claim': claim,
        'verdict': 'cancellation_pairwise_buffered|'
                   'sampled_consequence_confirmed',
        'anchors': 'design_seal.json frozen before '
                   'computation: canc gate '
                   '0.20/0.50 on mean add_err_rel; '
                   'overlap L24/L35 vs 3116 and JS '
                   'anchors vs 3115 within 1e-4 '
                   '(achieved 0.0); samp gate 0.60/'
                   '0.98 on same-seed sequence '
                   'agreement; frozen singles from '
                   '3116 result.json sweep',
        'artifacts': {
            'result': 'phase3117/omega_p115_'
                      'pair_cancellation_sampling/'
                      'result.json',
            'seal': 'phase3117/omega_p115_'
                    'pair_cancellation_sampling/'
                    'design_seal.json',
            'readout': 'phase3117/omega_p115_'
                       'pair_cancellation_'
                       'sampling/pair_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 3460s: Part A 15x2016 '
                'forwards + JS 2x1344 + 1200 greedy '
                '+ 6000 sampled generations KV '
                'cache); selfchecks 1.6e-3..2.1e-3; '
                'overlap bit-exact 0.0 on all four '
                'anchors (2 dmp_rel + 2 JS)',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3117_omega_p115_pair_cancellation_'
        'sampling')
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

# ---------- MEMO Phase 3117 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3117:' not in memo:
    sec = u'''## Phase 3117: Ω-P115 成对消融+采样行为验证——**成对层面 buffered：对消对联合效应偏离单点之和平均 163%（add_err_rel=1.627），cann_eff 0.02–0.995 逐对变异**；**行为传导方向不对称：L32 消融改变 A1 贪心首 token 20.2% 而 P 仅 0.15%（yes-swap=0，非表面替换）**；**采样确认传导：同 seed 序列一致率 0.444、yes 率 −4.0pp** [[NOW]]

**性质**：T3 第 11 Phase。qwen3-4b BF16，3460s。Part A：15 条件（baseline + L24/L35 单点 overlap + 12 层对：7 对消 + 4 同号负 + 1 同号正）× 2016 记录；Part B：JS 2 条件 × 1344 P/A1 记录（float64 全词表）+ 1200 贪心 + 6000 温度采样生成（300 对 × P/A1 × clean/abl_L32 × K=5，temp 0.7，seed=crc32(pk|dir|rep) 无 cond 项——clean 与 abl 同 seed，序列差异纯由消融引起）。预注册（design_seal.json 先于计算）：canc 门 mean add_err_rel ≤0.20 → additive / ≥0.50 → buffered；overlap L24/L35 vs 3116 与 JS anchors vs 3115 ≤1e-4；samp 门 seq_agree ≤0.60 → confirmed / ≥0.98 → invariant。

### 1. 方法与自检
MLP 置零 hook 语义同 3114–3116；自检 6 条件 rel L2 1.6e-03–2.1e-03。**Overlap 四项全 bit-exact：L24/L35 dmp_rel vs 3116 diff 0.00e+00；JS base/abl vs 3115 diff 0.00e+00**——JS 管线（log_softmax float64 全词表 + 在线配对）跨 Phase 精确复现，管线确定性再次全域证实。

### 2. 成对消融（12 对，672 对材料）

| 对 | 类 | sum(单点) | 联合 d_pair | add_err_rel | cann_eff/add_frac |
| --- | --- | --- | --- | --- | --- |
| L26+L24 | 对消 | −0.118 | **−0.462** | 2.93 | 0.429 |
| L33+L35 | 对消 | −0.071 | −0.233 | 2.27 | 0.718 |
| L31+L24 | 对消 | −0.070 | −0.004 | 0.94 | 0.995 |
| L26+L35 | 对消 | −0.087 | −0.031 | 0.65 | 0.963 |
| L31+L35 | 对消 | −0.040 | −0.191 | 3.78 | 0.760 |
| L14+L35 | 对消 | +0.140 | +0.174 | 0.25 | 0.716 |
| L33+L32 | 对消 | −0.345 | **−0.539** | 0.56 | 0.022 |
| L26+L33 | 同负 | −0.911 | −0.671 | 0.26 | 0.737 |
| L26+L31 | 同负 | −0.880 | −0.823 | 0.06 | 0.936 |
| L20+L26 | 同负 | −0.821 | −0.664 | 0.19 | 0.808 |
| L17+L33 | 同负 | −0.826 | −0.612 | 0.26 | 0.741 |
| L24+L35 | 同正 | +0.722 | **+0.830** | 0.15 | **1.149** |

**对消对 mean add_err_rel = 1.627（门 0.50）→ cancellation_pairwise_buffered**；cann_eff mean 0.657 但逐对变异 0.022–0.995。同负对 add_frac mean 0.805（亚可加），同正对 1.149（**超可加**）——缓冲不对称。

### 3. 行为传导（贪心对照 + 采样）
**贪心对照（300 对，clean vs abl_L32 同文本直接比较——3116 未测）**：12 token 全序列一致率 P=0.670 / A1=0.603；首 token 一致率 P=0.9967 / A1=0.9167；全 672 对首 token 一致率 **P=0.9985 / A1=0.7976**；分叉对中 yes 族内部替换率 **0.0**（非表面替换，闭合 3116 硬伤③）。**采样（temp 0.7，K=5，3000 序列对）**：同 seed 序列一致率 **0.4443**（1333/3000 → confirmed 门）；首 token 经验分布 L1 距离 0.087；yes 族出现率 58.57%→54.60%（**−4.0pp**）。

### 4. 三大发现（重复三遍）
1. **成对层面 buffered，平衡不可加组合**：一负一正层对联合移除的效应偏离单点之和平均 163%（1.627），且通常比线性预测**更负**（L26+L24 联合 −0.462 vs 预测 −0.118）——移除一对调节器部分瓦解剩余网络的补偿结构，而非留下预测的净差。**成对消融 buffered。平衡结构不可加。cann_eff 逐对变异 0.02–0.995。**
2. **行为传导方向不对称**：L32 擦除消融改变 **A1 方向（擦除活跃语境）**贪心首 token 20.2%、全序列 39.7%，而 P 方向仅 0.15%/33%；yes-swap=0 排除表面替换。**3116 的"零行为改变"是 P/A1 合并统计的掩盖——擦除机制在其活跃方向有真实 token 级行为后果，但不改变 yes/no 语义。**
3. **采样确认传导**：温度 0.7 下同 seed 序列一致率仅 44.4%，yes 率显著下降 4.0pp——**分布形状改变（JS +34%）在采样 regime 传导为行为差异**；"分布熵调节"精化为"**regime 依赖的行为传导**"：贪心=语义鲁棒表面可变，采样=统计可分。

### 5. 硬伤
① add_err_rel 的基准（冻结单点）本身含补偿路径激活，偏离量混合"缓冲释放"与"真对交互"，不能唯一归因；② 12 对是选择样本（全 276 对未扫）；③ 采样统计 300 对 × K=5（yes 率 4.0pp 的二项 SE ~0.9pp，显著但精度有限）；④ 贪心对照的 12 token 窗口内语义等价性未人工核验；⑤ 单模型。

### 6. 机制拼图更新
内部响应图谱新增：**成对消融矩阵**（对消/同号层的联合响应结构，cann_eff 谱 0.02–0.995）+ **行为传导 regime 图**（贪心/采样 × P/A1 方向四象限）。RDC 更新：① 平衡态模型升级为**补偿网络模型**——margin 由多层调节器维持，移除调节器子集时补偿结构部分失效产生超线性下跌，且正负调节器的缓冲不对称（正对超可加、负对亚可加）；② 行为传导是**方向 × regime 依赖**的：擦除机制在 A1 语境活跃并影响该方向的 token 选择，但 yes/no 语义由更深层的冗余平衡保护。对 AGI 理论：语义鲁棒性=冗余平衡的吸引子性质，表面 token 分布=受控涨落。

### 7. 3118 预注册（T4 多步自回归，观测前冻结框架）
① **自回归 margin 轨迹**：P/A1 记录在自由生成（贪心与 temp 0.7 采样双 regime）的每个生成步读出 m=margin，测量平衡态跨生成步的维持/衰减/重置——检验"受控量"的时间结构；② **TOP3 消融下的生成轨迹**：L26/L33/L31 单独消融时 margin 轨迹与生成内容的演化（成对消融已证空间补偿非线性→生成步维度检验补偿的时间结构）；③ 具体门在 3118 design_seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3117/omega_p115_pair_cancellation_sampling/`（result.json、design_seal.json、run_log.txt、pair_readout.npz）；脚本 `tests/glm5/phase3117_omega_p115_pair_cancellation_sampling.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3117)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3117 Omega-P115 (pair-cancellation '
          'ablation + sampled behavioral transmission, '
          'qwen3-4b, 3460s): verdict '
          'cancellation_pairwise_buffered|'
          'sampled_consequence_confirmed. Overlap '
          'BIT-EXACT 4/4 (L24/L35 dmp_rel vs 3116, JS '
          'base/abl vs 3115, all diff 0.0). (1) Cancel '
          'pairs (neg+pos): mean add_err_rel 1.627, '
          'joint usually MORE negative than predicted '
          '(L26+L24 -0.462 vs sum -0.118); cann_eff '
          '0.022-0.995 per pair; same-neg sub-additive '
          '0.805, same-pos SUPER-additive 1.149 -> '
          'equilibrium not pairwise-composable, '
          'asymmetric buffering. (2) L32 erase ablation '
          'changes A1 greedy first tokens 20.2% vs P '
          '0.15%, yes-swap 0 (not surface '
          'substitution) - 3116 zero-change stats were '
          'direction-mixing. (3) Sampled temp 0.7 '
          'same-seed agreement 0.444, yes rate -4.0pp '
          '-> distribution-shape change DOES reach '
          'sampled behavior; entropy regulation '
          'refined to regime-dependent transmission. '
          'NEXT 3118 (T4): autoregressive margin '
          'trajectory per generation step, greedy + '
          'sampled, with/without L26/L33/L31 '
          'ablation.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3117 Omega-P115' not in prev:
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
if 'max=3117' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3116）\n'
        '- 3116：全层扫描 interaction_dominant|'
        'no_behavioral_decoupling。因果 TOP3='
        'L26/L33/L31（L28 仅 −0.164）；正负对抗链'
        '全深度；L35 +0.376；**sum 单点=2.28×联合→'
        'margin=多层调节平衡态**（36 层 MLP 全移除'
        '仍留 5.4%）。**行为否定解耦：P/A1 首 token '
        '分叉 97.2%、L32 消融零改变**→分布熵调节。'
        'Overlap=0。',
        '## 机制链状态（3117）\n'
        '- 3117：成对消融 buffered|采样传导确认。'
        '对消对 add_err_rel=1.627（联合比线性预测'
        '更负，cann_eff 0.02–0.995）；同负亚可加 '
        '0.805 vs 同正超可加 1.149→**平衡不可加'
        '组合，缓冲不对称**。**行为传导方向不对称：'
        'L32 消融 A1 首 token 20.2% 分叉 vs P '
        '0.15%，yes-swap=0**；采样 seq_agree 0.444、'
        'yes −4pp→熵调节=regime 依赖传导。'
        'Overlap 4 项=0。\n'
        '- 3116：TOP3=L26/L33/L31；sum 单点=2.28×'
        '联合→margin=多层调节平衡态（全移除仍留 '
        '5.4%）；L32 贪心统计零改变→分布熵调节'
        '（3117 精化为方向×regime 传导）。')
    mem_new = mem_new.replace(
        '- 3112：出现层判决 emerge_L6|few_channels|'
        'replicated。3105 九层单坐标中位 L0 0.521→'
        'L4 0.772→L6 0.939（emerge）→峰 L7 0.947→'
        'L8 0.922：广播沿深度渐进形成。坐标共变 '
        'few_channels：λ1 share 0.479、PR 4.1、'
        'PC1-truth AUC 0.994。3106 复现 L8 0.743'
        '（同形但更低，平台 ~0.78）。写入端窗口收窄'
        '至 L4-L6。\n\n- 3111：广播判决 '
        'mixed_broadcast_plus_distribution。mean(h) '
        'AUC 0.944（一阶矩广播存在）但中心化/归一化'
        '后 d_min 恒 5（全阶广播）；决定性：单坐标 '
        'AUC 中位 0.922、55% 坐标单独 >0.9——每个'
        '坐标≈真值完整读出器（全息冗余）。终图景：'
        '真值=记录级全局信念状态广播；3107-3110 简并'
        '全部是该载体的必然后果。伪迹分离待 3112。\n\n'
        '- 3110：K-sweep 最小端口维度。真值 d_min=5'
        '（sharp_core：任意 5 维随机子空间 AUC 0.994，'
        '平台平坦）；crit_rel d_min=400（80 倍，平滑）'
        '→ truth_specific_diffuseness：弥散冗余是'
        '真值特有，truth=广播式 / 关系身份=局部内容'
        '码。候选载体=全局一阶矩信号（per-record 激活'
        '水平/整合度），3111 判决。\n',
        '- 3110–3112：真值=记录级一阶矩广播（AUC '
        '0.99）；最小端口 d_min=5 全息冗余（任意 5 '
        '维子空间 0.994）；L6 emerge、λ1 0.479；'
        '3112 判决 emerge_L6|few_channels。\n')
    mem_new = mem_new.replace(
        'max=3116', 'max=3117').replace(
        '下一 3117：**正负层对成对消融检验平衡结构 + '
        '采样行为验证**→ 之后 T4 多步自回归。',
        '下一 3118（T4）：**自回归 margin 轨迹——'
        '逐生成步读出平衡态，贪心+采样双 regime，'
        '含/不含 L26/L33/L31 消融**。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
