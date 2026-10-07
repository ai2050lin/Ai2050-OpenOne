# -*- coding: utf-8 -*-
"""Phase 3014 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3014'
     r'\omega_p2h_reverse_dose_law_qwen')
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
assert verdict == 'gate_destruction_fragile_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['l3'] == 3
assert t2['med_js_joint0'] == 0.013895
assert t2['med_js_k0'] == 0.01384
assert t2['med_js_v0'] == 0.012472
assert t2['med_retain_joint_05'] == 1.0112
assert t2['med_retain_joint']['0.25'] == 1.0121
assert t2['med_retain_joint']['0.75'] == 0.5028
assert t2['med_retain_konly']['0.75'] == 0.2409
assert t2['med_retain_vonly']['0.25'] == 0.37
assert t2['med_retain_vonly']['0.75'] == 0.0539
assert t2['med_d_kv_s0'] == -0.000124
assert t2['p_kv_s0'] == 0.75322
assert t2['gates_ok'] is True
assert res['T2b']['linear']['max_abs_resid'] == 0.2167
assert res['T2b']['quadratic']['max_abs_resid'] == 0.4538
assert res['T2c']['med_js_sham_joint']['0.0'] == 0.000317
assert res['T2c']['med_js_content_joint0'] == 0.000167
assert res['T2c']['n_sham'] == 11
assert res['T2c']['n_content'] == 22
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a13_t3_drift_diff'] == 0.0
assert res['anchors']['a17_3013'] is True

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3014
           for m in led['measurements']):
    claim = (
        'Omega-P2h (plan v5 P2) - reverse-surgery dose '
        'law at the L3 logic-position gate: HOW steep '
        'is the destruction side (3013: forgery '
        'impossible), and does the dose curve separate '
        'capacity from information?  Arms at L3, JS '
        'two-step readout, partial K,V erasure scale s '
        'in 0.0/0.25/0.5/0.75 for JOINT / KONLY / VONLY '
        'arms; retain(arm,s) = JS(arm,s)/JS(arm,0) is '
        'the DESTRUCTION-RETAINED share; PRIMARY stat '
        'med retain_joint(0.5).  Verdict '
        'gate_destruction_fragile_qwen.  RESULTS: '
        '(i) PRIMARY med retain_joint(0.5) = 1.0112 '
        '>= 0.75 gate: HALF-erasure destroys as much '
        'distribution as FULL erasure - the dose '
        'curve is NON-MONOTONE (retain 1.012/1.011/'
        '0.503 at s=0.25/0.5/0.75), a half-amplitude '
        'KV signal is at least as toxic as zero '
        'signal; (ii) K/V ARM DISSOCIATION: KONLY '
        'non-monotone (1.018/0.971/0.241) while '
        'VONLY monotone decreasing (0.370/0.177/'
        '0.054) - destruction is carried by the K '
        'routing channel (K enters softmax: '
        'competitive, non-monotone) whereas V content '
        'writes degrade linearly (V enters the value '
        'aggregate linearly); at full erasure K vs V '
        'show no median difference (dKV -0.0001, '
        'p=0.753) - the two channels cancel in the '
        'zero limit but differ in shape; (iii) both '
        'monotone fits are poor (linear resid 0.217 '
        'beats quadratic 0.454) because the curve is '
        'non-monotone; (iv) calibration clean: sham '
        'JS 0.0002-0.0004 across the grid (3011 '
        'level), content joint-zero JS 0.000167 - '
        'the fragile response is logic-position '
        'specific; (v) baseline determinism: T3 '
        'drift 49.5123 bit-identical to 3008-3013 '
        'and npz8 3c1c82c9 bit-identical across '
        'run4/run5 (a13 diff 0.0).  CONCLUSION: the '
        'L3 logic gate is fragile on the destruction '
        'side - scaling K to half amplitude already '
        'erases the gate effect; destruction needs '
        'no precision (any strong perturbation '
        'works, matching the 3009 token-level '
        'saturation), forgery needs the exact '
        'episodic K,V (3012/3013); white-box '
        'surgery asymmetry confirmed and now dose-'
        'calibrated.')
    meas = {
        'meas_id': 'meas3014_omega_p2h_reverse_'
                   'dose_law_qwen',
        'phase': 3014,
        'claim': claim,
        'verdict': verdict,
        'anchors': '17/17 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013)',
        'artifacts': {
            'result': 'phase3014/omega_p2h_reverse_'
                      'dose_law_qwen/result.json',
            'npz': 'phase3014/omega_p2h_reverse_'
                   'dose_law_qwen/'
                   'omega_p2h_reverse_dose_law_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run5 authoritative; run1 crashed at '
                'retain computation (np.array on a '
                'dict keyed by scale); run2 crashed at '
                'med_retain_05 (float vs str keys); '
                'run3 ran to completion but the JOINT '
                'arm was silently inert (lowercase '
                'joint in kv_scale_arm vs JOINT '
                'callers) producing all-zero JS and '
                'NaN retains - degenerate verdict '
                'VOID by protocol; run4 full pass but '
                'correction_note incomplete; all '
                'registered in correction_note.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 153
    l14['connects'].append({
        'meas_id': 'meas3014_omega_p2h_reverse_'
                   'dose_law_qwen',
        'phase': 3014,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2h: L3 logic gate '
                        'destruction dose law - '
                        'NON-MONOTONE (retain_joint '
                        '1.012/1.011/0.503 at s=0.25/'
                        '0.5/0.75): half-amplitude KV '
                        'as toxic as zero; K routing '
                        'channel carries the '
                        'destruction (KONLY non-'
                        'monotone 1.018/0.971/0.241) '
                        'while V content degrades '
                        'monotonically (0.370/0.177/'
                        '0.054); gate fragile to any '
                        'perturbation (matches 3009 '
                        'saturation), forgery needs '
                        'episodic K,V (3012/3013)'})
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
if '## Phase 3014:' not in memo:
    sec = u'''## Phase 3014: Ω-P2h 反向手术剂量律——非单调破坏曲线 + K/V 通道分岔 [%(created)s]

**判决：`gate_destruction_fragile_qwen`**（run5 权威，143.7s，锚 17/17：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 l\*=3、a16/a17 完整性）

### 设计（3013 机器 verbatim + 剂量网格）
锚链/几何/生成/位置选择/two-step 协议全继承 3013（SEED_RND=3009 显式重建）。T2a PRIMARY：logic 位 L3 K,V 部分擦除 s∈0,0.25,0.5,0.75 × 三臂 JOINT/KONLY/VONLY；retain(arm,s)=JS(arm,s)/JS(arm,0) 为**破坏保留份额**（retain 高 = 半擦除即丢大部分门控 = 门控脆弱）。T2b 剂量曲线形状拟合（线性 vs 二次）；T2c sham/content 校准。

### 核心结果（重复三遍）
**① 剂量曲线非单调、门控极度脆弱**：med retain_joint(0.5) = **1.0112** ≥ 0.75 门——半擦除（s=0.5）对分布的破坏 ≥ 全擦除（0.25/0.5/0.75 → 1.012/1.011/0.503）：半幅 KV 的"错误幅度信号"比零信号更毒（零=明确缺失，注意力均匀化；半幅=与均匀注意力竞争的毒信号）；**② K/V 通道完全分岔**：KONLY 非单调（1.018/0.971/0.241——K 进 softmax，指数竞争），VONLY 单调衰减（0.370/0.177/0.054——V 线性入值聚合）——**破坏由 K 路由通道承载，与 attention 机制精确对应**；全擦除时 K vs V 中位差为零（dKV −0.0001 p=0.753），两通道在零极限互抵但形状迥异；**③ 校准干净**：sham 全网格 JS 0.0002-0.0004（3011 水平）、content joint-zero 0.000167——脆弱响应是 logic 位特异；**④ 基线确定性**：T3 漂移 49.5123 与 3008-3013 逐位一致，npz8 3c1c82c9 跨 run4/run5 位级重现。

### 结论（白盒手术不对称的剂量定标）
L3 logic 门控破坏侧**无需精度**：K 缩到半幅即抹除门控效应（呼应 3009 token 级饱和——任何强扰动都破坏）；伪造侧**需要精确情景 K,V**（3012/3013 三重否定）。手术不对称确立且已定标：**破坏=粗粒度易行，伪造=bit 级不可能**。K/V 分岔给出破坏的机制分解——路由（K/softmax）比内容（V/线性）更毒。

### 硬伤（4 笔，correction_note 全登记）
run1 retain 计算崩（对 dict 键容器 np.array 得 0 维）；run2 med_rj float/str 键不一致 KeyError；run3 跑完但 **kv_scale_arm 小写 joint vs 调用方 JOINT 大小写 bug**——JOINT 臂静默无效（JS 全 0、retain 全 NaN）落入 else 分支得退化 verdict，**协议性作废**（退化统计量纪律的又一次实证：gates_ok 的 JS>0 门已捕获但 verdict 映射无独立分支）；run4 数据有效但 correction_note 缺 run2/run3 登记；run5 权威。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3014/omega_p2h_reverse_dose_law_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3015 = A（主选）K 路由通道机制定位——L3 logic 位 K 对后续 token attention 分配的直接测量（哪一层/哪个头消费该 K，softmax 熵变化）；B L31 次峰定位（晚层读出端）；C 情景性检验（同词异位 K,V 相似度 vs 异词）；D Ω-A2 GLM4 家族 Base 对照（需 GLM4-9B-Base 资产）。
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
if 'Phase 3014' not in prev:
    line = ('- Phase 3014 Omega-P2h: verdict '
            'gate_destruction_fragile_qwen; L3 gate '
            'destruction dose law NON-MONOTONE (retain '
            'joint 1.012/1.011/0.503 at s=0.25/0.5/0.75) '
            '- half-amplitude KV as toxic as zero; K '
            'routing carries destruction (KONLY non-'
            'monotone) vs V content monotone; fragile '
            'to any perturbation, forgery needs '
            'episodic KV; ledger 153/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
