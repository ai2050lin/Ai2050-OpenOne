# -*- coding: utf-8 -*-
"""Phase 3012 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3012'
     r'\omega_p2f_gate_surgery_qwen')
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
assert verdict == 'gate_mixed_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['l3'] == 3
assert t2['med_js_zero'] == 0.013895
assert t2['med_js_mean'] == 0.013723
assert t2['med_js_rand'] == 0.014344
assert t2['med_recov_mean'] == -0.0257
assert t2['med_recov_rand'] == 0.2557
assert t2['p_recov_mean'] == 0.50695
assert t2['med_js_sham_zero'] == 0.000317
assert t2['med_js_content_mean'] == 0.000247
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a15_3011'] is True
assert res['anchors']['a13_t3_drift_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3012
           for m in led['measurements']):
    claim = (
        'Omega-P2f (plan v5 P2) - L3 gate surgery '
        'operability: at L3 (=3011 l_star, asserted), '
        'logic-position K,V replaced by (ZERO = 3011 '
        'erasure / MEAN = per-dim mean over other '
        'prefill positions / RAND = per-dim Gaussian '
        'matched to empirical mean+std), JS readout '
        'two-step protocol, per-position paired '
        'recovery recov = 1 - JS_arm/JS_zero, '
        'sign-flip permutation N=10000.  Verdict '
        'gate_mixed_qwen.  RESULTS: (i) MEAN-REFILL '
        'FAILS: med recov_mean = -0.026 (p=0.507) - '
        'the per-dim average K/V of other positions '
        'does NOT restore the gate, so the gate is '
        'not carried by generic content; (ii) '
        'NOISE-REFILL PARTIAL: med recov_rand = '
        '0.256 (< 0.5 capacity gate) - scale-'
        'matched noise restores about a quarter, '
        'suggesting an amplitude/norm component; '
        '(iii) neither frozen gate met => MIXED: '
        'gate ~= 1/4 capacity (KV norm/scale) + '
        'content-specific remainder that neither '
        'mean nor noise can substitute (logic-token '
        'K/V content is not statistically '
        'replaceable); (iv) surgery calibration: '
        'refill itself harmless on sham positions '
        '(sham zero JS 0.000317, mean 0.000154, '
        'rand 0.00085) and content positions (mean-'
        'refill JS 0.000247) - content-position KV '
        'IS statistically interchangeable while '
        'logic-position KV is not; (v) dose: '
        'mean-refill on top of s-scaled KV is '
        'destabilizing at s=0.75 (recov -1.127); '
        '(vi) baseline determinism: T3 drift '
        '49.5123 bit-identical to 3008/3009/3010/'
        '3011 (a13 diff 0.0).  CONCLUSION: the L3 '
        'logic gate is a mixed informational/'
        'capacity object; full surgical restoration '
        'requires the original logic-token K/V - '
        'content-level KV patching alone cannot '
        'recreate it.')
    meas = {
        'meas_id': 'meas3012_omega_p2f_gate_'
                   'surgery_qwen',
        'phase': 3012,
        'claim': claim,
        'verdict': verdict,
        'anchors': '16/16 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3)',
        'artifacts': {
            'result': 'phase3012/omega_p2f_gate_'
                      'surgery_qwen/result.json',
            'npz': 'phase3012/omega_p2f_gate_'
                   'surgery_qwen/'
                   'omega_p2f_gate_surgery_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative; run1 crashed in '
                'kv_refill RAND arm (BFloat16 not '
                'numpy-convertible); rand noise source '
                'moved numpy->torch.Generator, seed '
                'semantics unchanged, registered in '
                'correction_note before run2.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 151
    l14['connects'].append({
        'meas_id': 'meas3012_omega_p2f_gate_'
                   'surgery_qwen',
        'phase': 3012,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2f: L3 logic gate is '
                        'MIXED - mean-refill fails '
                        '(recov -0.026 p=0.507), noise-'
                        'refill partial (0.256<0.5); '
                        '~1/4 capacity (KV norm) + '
                        'content-specific remainder; '
                        'content-position KV is '
                        'statistically interchangeable '
                        '(mean-refill JS 0.00025) while '
                        'logic-position KV is not - '
                        'surgical restoration needs the '
                        'original logic-token K/V'})
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
if '## Phase 3012:' not in memo:
    sec = u'''## Phase 3012: Ω-P2f L3 门控手术可操作性——混合门控（容量 ~1/4 + 内容特异主体） [%(created)s]

**判决：`gate_mixed_qwen`**（run2 权威，140.6s，锚 16/16：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 3011 完整性 + l*=3 断言）

### 设计（3011 机器 verbatim + 换干预性质）
锚链/几何/生成/位置选择/two-step 协议全继承 3011（SEED_RND=3009 显式重建，逐位可比）；L3 从 3011 result 读取并断言。T2 换为**回填对照**：logic 位 K,V 三臂——ZERO（=3011 擦除）/ MEAN（同 prompt prefill 其他位置的逐维均值）/ RAND（逐维 Gaussian 匹配经验均值+std，seed 固定）；配对恢复率 recov = 1 − JS_arm/JS_zero，sign-flip 置换 N=10000。T2c 剂量（scale s 后 mean 回填）描述性。

### 核心结果（重复三遍）
**① MEAN 回填不恢复门控**：med recov_mean = **−0.026**（p=0.507）——其他位置 K,V 的统计均值完全不能替代 logic 位内容；per-position 恢复率散布 −0.57~+0.12，无一致恢复。**② RAND 噪声回填部分恢复**：med recov_rand = **0.256**（< 0.5 容量门）——保尺度的噪声恢复约 1/4，提示存在幅度/范数成分。**③ 门控定性 = 混合**：≈1/4 容量（KV 范数/尺度）+ 内容特异主体（logic token 写入的特定 K,V 不可统计替代）；**④ 手术校准完美**：回填操作本身在 sham 位（zero JS 0.000317 / mean 0.000154 / rand 0.00085）与 content 位（mean 回填 JS 0.000247）几乎无扰动——**content 位 K,V 统计可互换，logic 位 K,V 不可**，双重差分确认 logic 位特殊性；**⑤ 剂量不稳定**：s=0.75 时 mean 回填叠加致 JS 翻倍（recov −1.127）；**⑥ 基线确定性五连位级**：T3 漂移 49.5123 与 3008/3009/3010/3011 逐位一致（a13 diff=0.0）。

### 结论（手术可操作性回答）
3011 的"L3 白盒手术靶点"在此获得可操作性边界：**完全恢复需要原始 logic-token K/V 内容**——均值补丁无效、噪声补丁只恢复尺度成分；但反向手术（破坏 logic 位门控）只需动 L3 KV 尺度（部分）或内容（完全）。"精准手术"愿景在门控侧 = 可破坏、不可伪造。

### 硬伤（1 笔，correction_note 权威重跑前登记）
run1 崩于 kv_refill RAND 臂（KV cache BFloat16 不可 numpy 转换）；rand 噪声源 numpy→torch.Generator（seed 语义不变 per-position，PREREG 唯一偏差在册）；run2 权威。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3012/omega_p2f_gate_surgery_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3013 = A（主选）logic 位 K,V 内容分解——K/V 各秩贡献 + logic-token K,V 的低秩投影结构（"不可统计替代"的内容到底是什么：单一方向还是分布式）；B L31 次峰定位；C 反向手术剂量律（L3 KV 部分擦除 s 网格下的 JS 剂量曲线，信息/容量成分分离定量）；D Ω-A2 GLM4 家族 Base 对照（需 GLM4-9B-Base 资产）。
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
if 'Phase 3012' not in prev:
    line = ('- Phase 3012 Omega-P2f: verdict '
            'gate_mixed_qwen; L3 logic gate is MIXED - '
            'mean-refill fails (recov -0.026 p=0.507), '
            'noise-refill partial (0.256 < 0.5 capacity '
            'gate); ~1/4 capacity (KV norm/scale) + '
            'content-specific remainder; content-'
            'position KV statistically interchangeable '
            '(mean-refill JS 0.00025) while logic-'
            'position KV is not; surgical restoration '
            'needs original logic-token K/V; ledger '
            '151/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
