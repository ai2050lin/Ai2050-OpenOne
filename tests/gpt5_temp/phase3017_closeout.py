# -*- coding: utf-8 -*-
"""Phase 3017 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3017'
     r'\omega_p2k_deep_absorption_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['final_verdict']
assert verdict == 'absorption_mixed_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['g7'] == 7
assert t2['med_js_g7e'] == 0.003332
assert t2['ident_rel_max'] == 0.04996
assert t2['gates_ok'] is True
assert t2['sig_layers'] == [5, 6, 7, 8, 9, 10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 27]
assert t2['sig_late_layers'] == [24, 25, 27]
assert t2['shrink_e35_e4'] == 10.1293
assert t2['reldecay_35_4'] == 0.2719
assert t2['med_c_by_layer']['10'] == -0.49607
assert t2['med_c_by_layer']['24'] == -0.10941
assert t2['med_c_by_layer']['34'] == 0.02199
assert t2['p_maxT_by_layer']['10'] == 0.0002
assert t2['p_maxT_by_layer']['24'] == 0.0002
t2b = res['T2b']
assert t2b['med_norm_e']['4'] == 3.6017
assert t2b['med_norm_e']['24'] == 15.8619
assert t2b['med_norm_e']['35'] == 35.4745
t2c = res['T2c']
assert t2c['component_split']['best_layer'] == 10
assert t2c['component_split']['c_total_best'] == \
    -0.49607
assert t2c['component_split']['c_attn_best'] == \
    -0.33801
assert t2c['component_split']['c_mlp_best'] == \
    -0.38824
assert t2c['med_js_sham_g7e'] == 0.000143
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a20_3016'] is True
assert 'run3: ' in res['correction_note']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3017
           for m in led['measurements']):
    claim = (
        'Omega-P2k (plan v5 P2) - deep-layer '
        'absorption mechanism: WHO digests the g7 '
        'K-routing perturbation in the deep layers '
        '(3016: lens JS at L8 is 29.1x final yet '
        'L32/35 converge to 0.76/0.56)?  Exact '
        'residual-recursion identity D_l = e_{l+1} - '
        'e_l (true residual stream via decoder-layer '
        'pre-hooks; e = erased-minus-baseline '
        'residual at the step-2 position); '
        'anti-alignment c(l) = cos(D_l, e_l), '
        '31-layer family l=4..34, circular-shift '
        'permutation null (shared shift, N=10000) '
        'with maxT family correction; ACTIVE = sig '
        'anti-alignment at l>=24 AND error shrink '
        '<= 0.7; PASSIVE = no sig layer AND '
        'relative-error decay <= 0.5.  Verdict '
        'absorption_mixed_qwen.  RESULTS: (i) the '
        'significant anti-alignment band is MID-'
        'STACK and CONTIGUOUS, L5-L25 + L27 (maxT '
        'p = 0.0002 at the gates), best layer L10: '
        'med c_total = -0.496 with BOTH components '
        'anti-aligned (c_attn -0.338, c_mlp '
        '-0.388) - not a late-layer phenomenon; '
        '(ii) the error norm does NOT shrink - it '
        'GROWS 10.1x along depth (med ||e||: 3.60 '
        'at L4, 15.86 at L24, 35.47 at L35; '
        'shrink gate 0.7 failed by 14x) - writes '
        'carry the error into larger-norm states '
        'rather than cancelling it; (iii) the '
        'RELATIVE error decays 3.7x (||e||/||res||: '
        '0.215 at L4 -> 0.057 at L35; reldecay '
        '0.272 <= 0.5) - the 3016 deep lens '
        'convergence is carried by residual-norm '
        'GROWTH (dilution) on top of a partial '
        'mid-stack active cancellation; (iv) '
        'identity gate idf = 0.04996 (single-chain '
        'recursion vs write norm - bf16 noise '
        'level); (v) calibration clean (sham JS '
        '0.000143), T3 drift 49.5123 bit-identical '
        '(a13 diff 0.0).  CONCLUSION: deep '
        '"convergence" is a MIXED mechanism - no '
        'dedicated late-layer compensator exists; '
        'mid-stack writes (L5-L25) are significantly '
        'anti-aligned with the error direction '
        '(partial active cancellation) while the '
        'absolute error norm grows 10x along depth, '
        'and the lens convergence that 3016 '
        'observed is a RELATIVE effect carried by '
        'residual-norm growth - matching the '
        'thesis that the L3 gate effect is decided '
        'by distributed deep readout, not by a '
        'dedicated absorption circuit.')
    meas = {
        'meas_id': 'meas3017_omega_p2k_deep_'
                   'absorption_qwen',
        'phase': 3017,
        'claim': claim,
        'verdict': verdict,
        'anchors': '20/20 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013; a18 3014; a19 '
                   '3015; a20 3016)',
        'artifacts': {
            'result': 'phase3017/omega_p2k_'
                      'deep_absorption_qwen/'
                      'result.json',
            'npz': 'phase3017/omega_p2k_'
                   'deep_absorption_qwen/'
                   'omega_p2k_deep_absorption_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run3 authoritative; run1 verdict void '
                '(self_attn pre-hook captures the '
                'POST-layernorm value, not the residual '
                'stream - identity gate idf 0.79 '
                'caught it); run2 void (cross-chain '
                'gate divided by the small difference '
                'med||dT|| - bf16 noise dominated, '
                'idf 0.152); fixes: decoder-layer '
                'pre-hook true-residual capture + '
                'single-chain identity gate ('
                'denominator = write norm); all '
                'registered in correction_note.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 156
    l14['connects'].append({
        'meas_id': 'meas3017_omega_p2k_deep_'
                   'absorption_qwen',
        'phase': 3017,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2k: deep absorption '
                        'of the g7 K-routing '
                        'perturbation is MIXED - '
                        'anti-alignment band is '
                        'MID-STACK contiguous L5-25+27 '
                        '(best L10: c_total -0.496, '
                        'c_attn -0.338, c_mlp -0.388; '
                        'maxT p 0.0002), NOT late-'
                        'layer; error norm GROWS 10.1x '
                        'along depth (3.60 -> 35.47) '
                        'while relative error decays '
                        '3.7x (0.215 -> 0.057) - the '
                        '3016 lens convergence is '
                        'carried by residual-norm '
                        'growth (dilution) + partial '
                        'mid-stack cancellation; no '
                        'dedicated late compensator'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted n=%d l14=%d'
             % (len(led['measurements']),
                len(l14['connects'])))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3017:' not in memo:
    sec = u'''## Phase 3017: Ω-P2k 深层吸收机制——中带主动抵消 + 稀释混合 [%(created)s]

**判决：`absorption_mixed_qwen`**（run3 权威，128.1s，锚 20/20：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 l\\*=3、a16-a20 完整性）

### 设计（3016 机器 verbatim + 残差递推恒等式）
锚链/几何/生成/位置选择/two-step 协议全继承 3016（SEED_RND=3009 显式重建）。**关键工具**：真残差流（decoder-layer forward_pre_hook 捕获——注意 self_attn pre-hook 捕获的是 post-input_layernorm 值而非残差流）。每 logic 位：基线 vs g7 擦除双链，e_l = res_l^er − res_l^base；**精确恒等式 D_l = e_(l+1) − e_l**（残差递推线性推出）；c(l)=cos(D_l, e_l)，l=4..34 共 31 层家族；circular-shift 置换 null（每置换共享移位，N=10000）+ **maxT 家族校正**（单侧下尾）。判决映射：晚层(l≥24)显著反平行+误差收缩≤0.7 → active_late；无显著层+相对误差衰减≤0.5 → passive_dilution；否则 mixed。恒等门：单链递推 res[l+1]−res[l]−attn−mlp vs 写入范数（与 bf16 噪声同尺度）。

### 核心结果（重复三遍）
**① 反平行带=中带连续而非晚层**：显著层 L5-L25 连续 + L27（maxT p=0.0002），最佳 L10 med c_total = **−0.496**，且 **attn（−0.338）与 mlp（−0.388）双组件都反平行**——中带写入系统性地顶着误差方向写；**② 误差范数不收缩反放大 10.1×**：‖e‖ 3.60(L4) → 15.86(L24) → 35.47(L35)，shrink 门 0.7 被 14× 击穿——写入把误差带向更大范数的状态而非消除它；**③ 相对误差被稀释 3.7×**：‖e‖/‖res‖ 0.215 → 0.057（reldecay 0.272 ≤ 0.5 门）——3016 观察到的深层 lens 收敛（L32/35 到 final 之下）主要由**残差范数增长的稀释**承载，叠加中带部分主动抵消；**④ 恒等门 idf=0.04996**（bf16 噪声水平）；**⑤ 校准干净**（sham 0.000143）、T3 漂移 49.5123 与 3008-3016 逐位一致。

### 结论（谁在消化）
深层"收敛"=**混合机制，没有晚层专用补偿器**：中带（L5-25）写入与误差方向显著反平行（部分主动抵消），但绝对误差沿深度增长 10×，最终 lens 收敛是**相对效应**（被残差增长稀释）。3016 的"深层吸收"由此定标：不是清除，而是**稀释+部分抵消+范数增长**。再次呼应：L3 门控效应的命运由分布式深层读出决定，单点操作化关闭。

### 硬伤（2 笔崩溃级 + 1 笔门重设计，correction_note 全登记）
run1 判决 void：**self_attn pre-hook 捕获的是 post-input_layernorm 值而非残差流**——恒等门 idf=0.79 抓住（结构错位量级）；run2 判决 void：真残差修复后 idf=0.152——**跨链恒等门分母 med‖D_l‖ 是小差分，bf16 量化噪声（~2⁻⁹）在差分下主导**；修复=恒等门改单链形式（分母=写入范数‖a+m‖，与噪声同尺度；跨链恒等由两条单链恒等线性推出）；run3 权威。教训入 MEMORY：递推恒等检查必须用真残差（decoder-layer 级 hook），门分母须与噪声同尺度。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3017/omega_p2k_deep_absorption_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3018 = A（主选）**稀释定标**——残差范数增长的层剖面与源头（哪些层贡献增长：attn vs mlp 范数预算），定量分割 lens 收敛中稀释 vs 抵消的份额；B L31 次峰定位；C 情景性检验（同词异位 K,V 相似度 vs 异词）；D 重定向终点测量（K 擦除后注意力质量流向哪些位置/是否为内容词）。
''' % {'created': created,
           'script8': exe['script_sha256_8'],
           'result8': seal['result_sha256_8'],
           'npz8': seal['npz_sha256_8'],
           'exec8': seal['exec_sha256_8'],
           'n': len(led['measurements']),
           'l14': len(l14['connects'])}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-20.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3017' not in prev:
    line = ('- Phase 3017 Omega-P2k: verdict '
            'absorption_mixed_qwen; deep absorption '
            'of the g7 K-routing perturbation is '
            'MIXED: anti-alignment band mid-stack '
            'contiguous L5-25+27 (best L10 c_total '
            '-0.496, attn -0.338, mlp -0.388, maxT '
            'p 0.0002), NOT late-layer; error norm '
            'GROWS 10.1x (3.60->35.47) while '
            'relative error decays 3.7x (0.215->'
            '0.057) - 3016 lens convergence = '
            'dilution + partial mid-stack '
            'cancellation; 2 void runs (post-LN '
            'capture gauge; bf16 noise in cross-'
            'chain gate) + single-chain identity '
            'gate redesign; ledger 156/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
