# -*- coding: utf-8 -*-
"""Phase 3116 closeout (idempotent):
Ledger -> MEMO Phase 3116 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3116'
        r'\omega_p114_full_mlp_sweep_decouple')
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
    'interaction_dominant|no_behavioral_decoupling'
assert res['n_records'] == 2016
assert res['smoke'] is False
assert res['coverage']['cov'] > 0.20
assert abs(res['coverage']['d_all']
           - (-0.94633)) < 1e-4
for k in ('L20', 'L24', 'L28', 'L32'):
    assert res['overlap_check'][k]['abs_diff'] == 0.0
assert res['decouple']['diff'] == 0.0
assert abs(res['decouple']['yes_rate_clean']
           - 0.588542) < 1e-5
assert max(res['selfcheck_rel'].values()) < 0.01

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3116
           for m in led['measurements']):
    claim = (
        'Omega-P114 (3116, full-layer MLP ablation sweep '
        'L12-L35 + all-layer anchor + greedy-generation '
        'behavioral test, qwen3-4b, 26x2016 forwards + '
        '2688 generations, 2537s) - verdict '
        'interaction_dominant|no_behavioral_decoupling. '
        'Overlap reproduction BIT-EXACT: L20 vs 3115 and '
        'L24/L28/L32 vs 3114 dmp_rel diff 0.00e+00 '
        '(hook semantics + material rebuild + readout '
        'pipeline fully deterministic across phases).  '
        '(1) FULL CAUSAL MAP: largest single-point '
        'effects at L26 (-0.464), L33 (-0.448), L31 '
        '(-0.416) - NONE is the 3113 correlational peak '
        'L28 (causal -0.164); sign-alternating '
        'adversarial chain through all depth: L14-23 '
        'negative block (L17 -0.378, L18 -0.342, L20 '
        '-0.358, L22 -0.363, L23 -0.356), L24-25 '
        'POSITIVE (+0.346, +0.197), L26-28 negative, '
        'L31/33 negative with L32 positive (+0.103), '
        'L35 STRONGLY POSITIVE (+0.376) - the last '
        'layer MLP rebuilds the margin.  (2) MARGIN IS '
        'A BALANCED CONTROLLED STATE: sum of 24 '
        'single-point effects -3.102 is 2.28x the '
        'all-layer joint effect -0.946 (coverage gate '
        '0.20 -> interaction_dominant) - single-point '
        'effects are heavily buffered/compensated by '
        'the remaining layers; even with ALL 36 MLPs '
        'removed the margin keeps ~5.4% (attention/'
        'embedding path).  The margin is not an '
        'accumulated write but a multi-regulator '
        'equilibrium.  (3) BEHAVIORAL TEST KILLS THE '
        'DECOUPLING HYPOTHESIS: P vs A1 greedy '
        'generations diverge at token 1 for 97.2% of '
        'pairs (yes-family rate 58.85% - the model '
        'just answers the question), and ablating the '
        'L32 erase changes NOTHING behaviorally '
        '(agree_clean = agree_abl = 0.0283, diff '
        '0.0000; yes_rate identical 0.588542) - the '
        '3115 JS increase is a distribution-shape '
        '(entropy) effect that does NOT reach greedy '
        'behavior; 3115 "belief-generation decoupling" '
        'naming is WITHDRAWN, replaced by '
        '"distribution-entropy regulation".  '
        'Self-checks 1.9e-3..2.3e-3 (incl. abl_all_'
        'mlp).  CAVEATS: (i) coverage 2.28 is a lower-'
        'bound indicator of non-linearity, not an '
        'exact decomposition; (ii) greedy-only '
        'behavior (sampled generation untested); '
        '(iii) identical clean/abl behavior stats '
        'could hide token substitutions (yes->Yes); '
        '(iv) layers 0-11 not swept (outside '
        'preregistration); (v) single model.  NEXT '
        '3117: paired-ablation test of the equilibrium '
        'structure (negative+positive layer pairs '
        'removed together should cancel if the '
        'balance reading is right) + sampled-'
        'generation behavioral check; then T4.')
    meas = {
        'meas_id': 'meas3116_omega_p114_'
                   'full_mlp_sweep_decouple',
        'phase': 3116,
        'claim': claim,
        'verdict': 'interaction_dominant|'
                   'no_behavioral_decoupling',
        'anchors': 'design_seal.json frozen before '
                   'computation: coverage gate 0.20; '
                   'decouple gate 0.05/0.02 on '
                   'first-8-token agreement; overlap '
                   'must match 3114/3115 within 1e-4 '
                   '(achieved 0.0); frozen singles '
                   'from 3114/3115 result.json',
        'artifacts': {
            'result': 'phase3116/omega_p114_'
                      'full_mlp_sweep_decouple/'
                      'result.json',
            'seal': 'phase3116/omega_p114_'
                    'full_mlp_sweep_decouple/'
                    'design_seal.json',
            'readout': 'phase3116/omega_p114_'
                       'full_mlp_sweep_decouple/'
                       'sweep_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 2537s: 26x2016 forwards + '
                '2688 greedy generations with KV '
                'cache); selfchecks 1.9e-3..2.3e-3; '
                'overlap bit-exact 0.0 on all four '
                'frozen layers',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3116_omega_p114_full_mlp_sweep_decouple')
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

# ---------- MEMO Phase 3116 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3116:' not in memo:
    sec = u'''## Phase 3116: Ω-P114 全层 MLP 扫描+行为验证——因果地图=L26/L33/L31（相关峰 L28 仅 −0.164）；正负对抗链贯穿全深度（L35 强正 +0.376 重建 margin）；**sum 单点=2.28×联合（interaction_dominant）→ margin 是多层调节的平衡态**；**行为验证否定解耦假说：P/A1 生成首 token 即分叉（97.2%），L32 消融对行为统计零改变** [[NOW]]

**性质**：T3 第 10 Phase。qwen3-4b BF16，26 条件（baseline + L12–L35 单点 24 + abl_all_mlp 锚点）× 2016 记录 + 2688 贪心生成（12 token，KV cache），2537s。预注册（design_seal.json 先于计算）：coverage 门 |sum_single−d_all|/max(|d_all|,0.01) ≤0.20 → single_sum_explains_joint；decouple 门 agree_clean−agree_abl ≥0.05 → behavioral_decoupling_confirmed / ≤0.02 → no；overlap 与 3114/3115 一致 ≤1e-4。

### 1. 方法与自检
MLP 置零 hook（24 扫描层 + 全层锚点）；自检在 4 个仪器化层（20/24/28/32）+ abl_all_mlp，rel L2 1.9e-03–2.3e-03。**Overlap bit-exact 复现：L20（vs 3115）、L24/L28/L32（vs 3114）diff 全部 0.00e+00**——hook 语义、材料重建、读出管线跨 Phase 完全确定性。

### 2. 全层因果地图（2016 记录，672 对）

| 段 | 层 | dmp_rel |
| --- | --- | --- |
| 早期 | L12/13/15/19 | −0.006/−0.039/+0.035/−0.057（近零）|
| 早期主写 | L14/16/17/18 | −0.237/−0.286/**−0.378**/−0.342 |
| 写入窗 | L20/21/22/23 | −0.358/−0.187/−0.363/−0.356 |
| 对抗 | L24/25 | **+0.346**/+0.197 |
| 后段 | L26/27/28 | **−0.464**/−0.154/−0.164 |
| 晚段 | L29/30/31/32/33/34/35 | +0.050/−0.030/**−0.416**/+0.103/**−0.448**/+0.075/**+0.376** |

TOP3 因果层 = **L26（−0.464）、L33（−0.448）、L31（−0.416）**——无一在 3113 相关峰 L28（其因果效应仅 −0.164）。**L35（最后一层 MLP）强正 +0.376**：终点层在大幅重建 margin。abl_all_mlp（36 层 MLP 全移除）d_all = **−0.946**（margin 仍保留 ~5.4%，来自 attention/embedding 路径）。

### 3. 三大发现（重复三遍）
1. **相关地图≠因果地图（全域证实）**：24 层扫描中因果贡献最大的层（L26/L33/L31）都不在相关峰；正负交替的**对抗链**贯穿全深度。**相关地图≠因果地图。因果主力在相关盲区。**
2. **margin 是多层调节的平衡态**：sum(24 单点) = −3.102 = **2.28×** 联合效应 −0.946（interaction_dominant）——单点效应被其余层严重缓冲补偿；只有全部移除才掉 94.6%。**margin 不是写入的累计值，而是被多层正负调节器维持的平衡状态（受控量）。margin=平衡态。单点效应是补偿后的残差。**
3. **行为验证否定解耦假说**：P vs A1 贪心生成 **97.2% 在第一个 token 分叉**（yes 族出现率 58.85%——模型就是在回答问题，truth 直接驱动生成），且 L32 擦除消融对行为统计**零改变**（agree 0.0283→0.0283，yes_rate 0.588542→0.588542，bit 级相同）。**3115 的"信念-生成解耦"命名撤回**，修正为**"分布熵调节"**：擦除改变分布形状（JS +34%）但不触及贪心决策——分布空间的连续操控与生成的离散决策是分离的。**解耦假说否定。擦除=熵调节，非行为开关。**

### 4. 硬伤
① coverage 2.28 是非线性的下界指示，不是精确分解（单点效应含补偿路径激活）；② 行为验证仅贪心（温度>0 采样下 JS 差异可能表现为行为差异）；③ clean/abl 行为统计完全相同可能隐藏 token 替换（' yes'→'Yes'）；④ 层 0–11 未扫描（预注册范围外）；⑤ 单模型。

### 5. 机制拼图更新
内部响应图谱新增：**全深度因果对抗链地图**（正负写入交替、终点层重建）+ **平衡态本体论**（margin=受控量，非累计量）。RDC 更新：LLM 用多层正负调节器维持内部信念状态，类似控制系统的设定值跟踪；分布形状（熵/JS）与 argmax 行为是两个可分别操控的层面。对 AGI 理论：内部状态的鲁棒性来自冗余平衡调节，而非单点写入的强度。

### 6. 3117 预注册（观测前冻结）
① **平衡结构成对消融检验**：一负一正层对（如 L26+L24、L33+L35）联合移除——若平衡读出正确，联合效应 ≈ 单点之和且远小于各自绝对值之和的贡献（对消）；并做正层单独移除的符号验证；② **采样行为验证**（温度 0.7 重测 JS→行为传导）；③ 之后 T4 多步自回归。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3116/omega_p114_full_mlp_sweep_decouple/`（result.json、design_seal.json、run_log.txt、sweep_readout.npz）；脚本 `tests/glm5/phase3116_omega_p114_full_mlp_sweep_decouple.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3116)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3116 Omega-P114 (full-layer MLP '
          'sweep L12-L35 + all-anchor + greedy-gen '
          'behavioral test, qwen3-4b, 2537s): verdict '
          'interaction_dominant|no_behavioral_'
          'decoupling. Overlap BIT-EXACT (L20/L24/L28/'
          'L32 vs 3115/3114 diff 0.0). (1) Causal map: '
          'top layers L26 -0.464 / L33 -0.448 / L31 '
          '-0.416, NONE at the 3113 correlational '
          'peak L28 (-0.164); sign-alternating '
          'adversarial chain; L35 STRONGLY POSITIVE '
          '+0.376 rebuilds margin; all-36-MLP removal '
          'keeps 5.4%. (2) sum_single -3.102 = 2.28x '
          'joint -0.946 -> MARGIN IS A MULTI-'
          'REGULATOR BALANCED STATE, not accumulated '
          'write. (3) P/A1 generations diverge at '
          'token 1 (97.2%, yes-rate 58.85%); ablating '
          'L32 erase changes behavior ZERO (agree '
          '0.0283->0.0283, yes_rate bit-identical) -> '
          '3115 decoupling naming WITHDRAWN -> '
          '"distribution-entropy regulation"; '
          'distribution shape and greedy decision '
          'are separable layers. NEXT 3117: paired '
          '(neg+pos) ablation test of equilibrium + '
          'sampled-generation check; then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3116 Omega-P114' not in prev:
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
if 'max=3116' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3115）\n'
        '- 3115：联合消融 write_joint_partial|additive|'
        'erase_serves_generation。**L20 单点 −0.358='
        '因果最大层**（3113 ds 却最小→相关因果沿深度'
        '反转）；L20-28 三层 MLP 联合仅 −0.188='
        '**写入深度分布式（~81% 在别处）**且 additive；'
        '**L32 擦除目的性强证实：JS 0.159→0.213 '
        '(sign 0.976)=信念-生成解耦**。\n'
        '- 3114：head 消融不变（+0.0101，head=相关'
        '影子）；L28 MLP −0.164 部分；L24 单点 '
        '+0.346=条件性对抗写入。',
        '## 机制链状态（3116）\n'
        '- 3116：全层扫描 interaction_dominant|'
        'no_behavioral_decoupling。因果 TOP3='
        'L26/L33/L31（L28 仅 −0.164）；正负对抗链'
        '全深度；L35 +0.376；**sum 单点=2.28×联合→'
        'margin=多层调节平衡态**（36 层 MLP 全移除'
        '仍留 5.4%）。**行为否定解耦：P/A1 首 token '
        '分叉 97.2%、L32 消融零改变**→分布熵调节。'
        'Overlap=0。\n'
        '- 3115：三层联合 −0.188 additive；L32 '
        'JS +34%=熵调节（3116 修正）。\n'
        '- 3114：head=相关影子；L24 +0.346 条件性'
        '对抗。')
    mem_new = mem_new.replace(
        'max=3115', 'max=3116').replace(
        '下一 3116：**全层 MLP 消融扫描 L12–L36（定位 '
        '~81% 剩余贡献）+ 自由生成行为验证解耦**→ '
        '之后 T4 多步自回归。',
        '下一 3117：**正负层对成对消融检验平衡结构 + '
        '采样行为验证**→ 之后 T4 多步自回归。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
