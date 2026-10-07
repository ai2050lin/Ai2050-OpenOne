# -*- coding: utf-8 -*-
"""Phase 3013 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3013'
     r'\omega_p2g_kv_content_decomposition_qwen')
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
assert verdict == 'position_specific_gate_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['l3'] == 3
assert t2['kv_dim'] == 1024
assert t2['med_js_zero'] == 0.013895
assert t2['med_js_pool'] == 0.014391
assert t2['med_recov_pool'] == -0.0357
assert t2['med_recov_mixed'] == -0.0257
assert t2['med_recov_rand'] == 0.2557
assert t2['p_recov_pool'] == 0.75192
assert res['T2b']['k_star'] is None
assert res['T2b']['per_k']['8']['med_recov'] == 0.397
assert res['T2c']['med_cos_logic_to_pool'] == 0.9136
assert res['T2c']['med_cos_content_to_pool'] == 0.8473
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a16_3012'] is True
assert res['anchors']['a13_t3_drift_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3013
           for m in led['measurements']):
    claim = (
        'Omega-P2g (plan v5 P2) - L3 logic-position KV '
        'content decomposition: is the non-'
        'replaceable gate content (3012) a SHARED '
        'logic-class component (cross-prompt pool) or '
        'position-specific distributed content?  Arms '
        'at L3, JS two-step readout, per-position '
        'paired recovery: ZERO (3011 erasure) / '
        'MEAN_MIXED (3012 verbatim re-run) / '
        'MEAN_LOGIC (per-dim mean over ALL logic '
        'positions from ALL prompts, leave-p-out) / '
        'RAND (3012 verbatim) / LOO rank-k refill '
        '(K,V replaced by own top-k projection under '
        'PCA fit on the other logic positions; k in '
        '1,2,4,8).  Verdict position_specific_gate_'
        'qwen.  RESULTS: (i) logic-pool mean refill '
        'FAILS: med recov_pool = -0.036 (p=0.752) - '
        'the cross-prompt logic-class average K/V '
        'does NOT restore the gate either; (ii) LOO '
        'rank-k refill rises slowly and does not '
        'reach 0.5 within the grid (med recov -0.066/'
        '0.126/0.099/0.397 at k=1/2/4/8, k_star=null '
        'at n_pool=11) - the gate content is NOT '
        'carried by a few shared directions; (iii) '
        'rank-8 restoring ~0.40 while rank-2 ~0.13 '
        'suggests a broad high-dimensional structure '
        'plus the known ~1/4 scale component; (iv) '
        'class tightness: logic K cosine to pool mean '
        '0.914 vs content 0.847 - the logic class IS '
        'tighter, yet its mean is still not the gate '
        '(tight cluster around a mean that is not '
        'sufficient); (v) noise-refill (0.256) beats '
        'both mean refills - amplitude/norm carries '
        'more of the gate than the shared mean '
        'direction; (vi) baseline determinism: T3 '
        'drift 49.5123 bit-identical to '
        '3008/3009/3010/3011/3012 (a13 diff 0.0).  '
        'CONCLUSION: the L3 logic gate content is '
        'position-specific and high-dimensional - '
        'neither the class-mean direction nor low-'
        'rank projections substitute for it; the '
        'gate is individually written per logic '
        'token occurrence (episodic, not a reusable '
        'class code); surgical restoration requires '
        'the original K/V, destruction needs only '
        'scale.')
    meas = {
        'meas_id': 'meas3013_omega_p2g_kv_content_'
                   'decomposition_qwen',
        'phase': 3013,
        'claim': claim,
        'verdict': verdict,
        'anchors': '16/16 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012)',
        'artifacts': {
            'result': 'phase3013/omega_p2g_kv_content_'
                      'decomposition_qwen/result.json',
            'npz': 'phase3013/omega_p2g_kv_content_'
                   'decomposition_qwen/'
                   'omega_p2g_kv_content_'
                   'decomposition_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run3 authoritative; run1 crashed at '
                'T2a tag parse (P-prefixed prompt '
                'field); run2 crashed at pool_mean arm '
                '(kv_replace view fix was a phantom '
                'edit, 1-D flat vector vs [1,8,128]); '
                'both registered in correction_note.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 152
    l14['connects'].append({
        'meas_id': 'meas3013_omega_p2g_kv_content_'
                   'decomposition_qwen',
        'phase': 3013,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2g: L3 logic gate '
                        'content is position-specific '
                        'and high-dimensional - cross-'
                        'prompt logic-pool mean refill '
                        'fails (recov -0.036 p=0.752), '
                        'LOO rank-k refill does not '
                        'reach 0.5 by k=8 (0.397, '
                        'k_star null); noise (0.256) '
                        'beats class mean; logic class '
                        'tighter (cos 0.914 vs 0.847) '
                        'but its mean is not the gate - '
                        'episodic per-occurrence KV '
                        'content, not a reusable class '
                        'code'})
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
if '## Phase 3013:' not in memo:
    sec = u'''## Phase 3013: Ω-P2g logic 位 KV 内容分解——位置特异高维门控（情景式，非类码） [%(created)s]

**判决：`position_specific_gate_qwen`**（run3 权威，144.1s，锚 16/16：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 3011 完整性 + l\*=3、a16 3012 完整性）

### 设计（3012 机器 verbatim + 内容分解三臂）
锚链/几何/生成/位置选择/two-step 协议全继承 3012（SEED_RND=3009 显式重建）；T1 捕获各 prompt L3 KV（heads 展平，kvdim=1024）于全部 logic/content/sham 位。T2a 回填对照四臂：ZERO / MEAN_MIXED（3012 复跑）/ **MEAN_LOGIC（跨 prompt 全 logic 位逐维均值，leave-p-out）** / RAND。T2b **LOO 秩-k 回填**：logic p 的 K,V 换成自身在其他 logic 位 PCA 基下的 top-k 投影（k∈1,2,4,8）；T2c 类紧度对照。

### 核心结果（重复三遍）
**① 跨 prompt 类均值回填同样失败**：med recov_pool = **−0.036**（p=0.752）——logic 类的共享均值方向不承载门控（3012 的混类均值失败不是稀释假象，是真的没有类均值成分）；**② LOO 秩-k 回填缓慢爬升且不达门**：k=1/2/4/8 → −0.066/0.126/0.099/0.397，k\*=null（n_pool=11 内未达 0.5）——门控内容不在少数共享方向上，秩 8 也只恢复 0.40；**③ 噪声回填 0.256 击败两个均值臂**——幅度/范数成分比类均值方向承载更多门控；**④ 类紧度悖论**：logic K 对池均值 cos **0.914** vs content **0.847**——logic 类确实更紧，但"紧"不等于"均值即内容"：每个 logic token 出现写入的是围绕类中心的**个体特异高频结构**；**⑤ 基线确定性六连位级**：T3 漂移 49.5123 与 3008-3012 逐位一致。

### 结论（门控内容的本体论定性）
L3 logic 门控 = **情景式（episodic）逐次写入**：每个 logic token 出现在上下文中时写入各自的 K,V 内容，类中心只是云的聚点而非门控码本。三重否定（混类均值 3012 / 类均值 3013 / 低秩投影 3013）确立：**可破坏（动尺度）但不可伪造（须原文 K,V）**；"复用类码做白盒注入"路线在 KV 门控上被否证——门控不是参数化的类知识，而是运行时的情景记忆。

### 硬伤（2 笔，correction_note 全登记）
run1 崩于 T2a tag 解析（P 前缀 int 转换）；run2 崩于 pool_mean 臂——kv_replace 的 view 修复是 **Edit 幻影**（报成功未落盘，1-D 展平向量撞 [1,8,128] 形状错），改 Python 直接写盘补丁后 run3 权威（本会话第三次遭遇幻影，工程纪律持续有效）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3013/omega_p2g_kv_content_decomposition_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3014 = A（主选）反向手术剂量律——L3 KV 部分擦除 s 网格的 JS 剂量曲线 + 信息/容量成分分离定量（破坏侧可操作性定标）；B L31 次峰定位（晚层读出端）；C logic 位 K,V 的逐 token 情景性检验（同词不同位置 K,V 相似度 vs 不同词）；D Ω-A2 GLM4 家族 Base 对照（需 GLM4-9B-Base 资产）。
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
if 'Phase 3013' not in prev:
    line = ('- Phase 3013 Omega-P2g: verdict '
            'position_specific_gate_qwen; L3 logic gate '
            'content is position-specific high-'
            'dimensional - logic-pool mean refill fails '
            '(-0.036 p=0.752), LOO rank-k does not reach '
            '0.5 by k=8 (0.397), noise (0.256) beats '
            'class mean; logic class tighter (cos 0.914 '
            'vs 0.847) but mean is not the gate - '
            'episodic per-occurrence KV content, not a '
            'reusable class code; ledger 152/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
