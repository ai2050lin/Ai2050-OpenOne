# -*- coding: utf-8 -*-
"""Phase 3016 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3016'
     r'\omega_p2j_amplification_trace_qwen')
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
assert verdict == 'amp_distributed_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['g7'] == 7
assert t2['med_js_g7e'] == 0.003332
assert t2['med_restoration'] == 0.0851
assert t2['restoration_per_pos'][4] == -0.8321
assert t2['restoration_per_pos'][3] == 0.3856
assert t2['l_star_counts'] == {'4': 2, '7': 2, '8': 1,
                               '9': 1, '12': 1, '13': 3,
                               '24': 1}
assert len(t2['qh_star_counts']) == 9
assert t2['med_d_star'] == 1.8258
assert t2['gates_ok'] is True
t2b = res['T2b']
assert t2b['ratio_vs_final']['8'] == 29.0858
assert t2b['ratio_vs_final']['35'] == 0.5589
assert t2b['ratio_vs_final']['32'] == 0.7587
assert t2b['med_lens_js']['8'] == 0.096913
assert t2b['med_js_final'] == 0.003332
t2c = res['T2c']
assert t2c['concentration']['top_layer'] == 4
assert t2c['concentration']['top1_share'] == 0.1279
assert t2c['concentration']['participation'] == 2.94
assert t2c['med_js_sham_g7e'] == 0.000143
assert t2c['med_max_dnorm_sham'] == 0.865
assert t2c['n_sham'] == 11
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a19_3015'] is True
assert 'run2: authoritative' in res['correction_note']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3016
           for m in led['measurements']):
    claim = (
        'Omega-P2j (plan v5 P2) - downstream '
        'amplification trace at the L3 logic gate: '
        'WHO amplifies the g7 K-routing destruction '
        '(3015 leading consumer) into the full '
        'distribution JS?  Per logic position: '
        'baseline vs g7-erased (L3 KV head 7 K '
        'zeroed at p) two-step chains capturing '
        'every layer o_proj INPUT at the step-2 '
        'position (4096 = 32 query heads x 128); '
        'carrier = argmax normalized per-head delta '
        'energy over l in 4..35 (quasi-post-hoc '
        'labeled); causal restoration surgery - '
        'replace the carrier o_proj-input slice '
        'with its baseline value in the g7-erased '
        'chain; restoration = 1 - '
        'JS_restored/JS_g7e; logit-lens build-up '
        'profile.  Verdict amp_distributed_qwen.  '
        'RESULTS: (i) PRIMARY med restoration = '
        '0.0851 <= 0.2 gate - undoing the single '
        'largest-delta (layer, query-head) recovers '
        'only 8.5% of the JS; restorations scatter '
        'from -0.832 to +0.386 (largest delta is '
        'not the effect carrier - delta energy and '
        'distribution impact are DECOUPLED); '
        '(ii) no consistent carrier: per-position '
        'l* spread over L4-L24 (13 x3, 4 x2, 7 x2 '
        '...), qh* over 9 different query heads; '
        'per-layer delta energy is itself '
        'dispersed (top1 share 0.128 at the best '
        'layer, participation 2.94/32); '
        '(iii) logit-lens build-up is NON-MONOTONE: '
        'med lens JS peaks mid-stack (L8 0.0969 = '
        '29.1x the final JS) then CONVERGES below '
        'final at L32-L35 (0.759 / 0.559) - the '
        'KV perturbation is LARGE at mid-layer '
        'readouts but the deep layers partially '
        'absorb it; the final distribution JS is '
        'the RESIDUE of a mid-stack divergence '
        'that deep layers mostly clean up; '
        '(iv) sham positions show the same large '
        'mid-stack delta (max 0.865) with almost '
        'no distribution effect (JS 0.000143) - '
        'delta energy is generic, the distribution '
        'readout is logic-position specific; '
        '(v) calibration clean and T3 drift '
        '49.5123 bit-identical (a13 diff 0.0).  '
        'CONCLUSION: the amplification of the g7 '
        'K-routing destruction is DISTRIBUTED and '
        'DEEP-CONVERGING - no single downstream '
        'head carries it (matching the 3015 rho '
        '0.45 impact-vs-attention gap), mid-stack '
        'divergence is absorbed by deep layers, '
        'and the gate effect survives as a small '
        'residue; the L3 gate thus acts as a '
        'seed whose fate is decided by a '
        'distributed deep readout, not by a '
        'dedicated amplifier circuit.')
    meas = {
        'meas_id': 'meas3016_omega_p2j_amplification_'
                   'trace_qwen',
        'phase': 3016,
        'claim': claim,
        'verdict': verdict,
        'anchors': '19/19 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013; a18 3014; a19 '
                   '3015)',
        'artifacts': {
            'result': 'phase3016/omega_p2j_'
                      'amplification_trace_qwen/'
                      'result.json',
            'npz': 'phase3016/omega_p2j_'
                   'amplification_trace_qwen/'
                   'omega_p2j_amplification_trace_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative; run1 crashed at '
                'the logit-lens readout (model.norm '
                'fp32 output vs bf16 lm_head weights, '
                'F.linear dtype mismatch) - fixed by '
                'casting to lm_head.weight.dtype; '
                'crash occurred before any verdict, '
                'no data validity impact; registered '
                'in correction_note.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 155
    l14['connects'].append({
        'meas_id': 'meas3016_omega_p2j_amplification_'
                   'trace_qwen',
        'phase': 3016,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2j: downstream '
                        'amplification of the g7 '
                        'K-routing destruction is '
                        'DISTRIBUTED + DEEP-CONVERGING '
                        '- single-carrier restoration '
                        'only 8.5% (scatter -0.83..'
                        '+0.39: delta energy decoupled '
                        'from impact); l* spread L4-24, '
                        'qh* 9 heads, per-layer top1 '
                        'share 0.128; lens NON-MONOTONE: '
                        'mid-stack 29.1x final at L8 '
                        'then converges BELOW final '
                        '(0.759/0.559 at L32/L35) - '
                        'deep layers absorb the '
                        'perturbation; sham delta '
                        '0.865 with JS 0.000143 - '
                        'readout is logic-specific'})
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
if '## Phase 3016:' not in memo:
    sec = u'''## Phase 3016: Ω-P2j 下游放大追踪——分布式放大 + 深层收敛 [%(created)s]

**判决：`amp_distributed_qwen`**（run2 权威，138.6s，锚 19/19：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 l\*=3、a16-a19 完整性）

### 设计（3015 机器 verbatim + 载体复原手术）
锚链/几何/生成/位置选择/two-step 协议全继承 3015（SEED_RND=3009 显式重建）。每 logic 位：基线 vs g7 擦除（L3 KV 头 7 K 零化——3015 领先消费者）双链，捕获 step-2 位每层 **o_proj 输入**（4096=32 query 头×128）；dnorm(l,qh)=‖Δ‖/‖基线‖；载体=argmax（l∈4..35，quasi-post-hoc 已标注）；**因果复原手术**：g7 擦除链中把载体头 o_proj 输入切片改回基线值（in-place o_proj pre-hook 手术）；restoration=1−JS_rest/JS_g7e。T2b logit-lens 逐层构建剖面；T2c 集中度+sham 校准。

### 核心结果（重复三遍）
**① 放大=分布式**：PRIMARY med restoration = **0.0851** ≤ 0.2 门——撤销最大 delta 头只恢复 **8.5%%** 的 JS；各位置 restoration 散布 **−0.832 ~ +0.386**（最大 delta 头甚至可能反向）——**delta 能量与分布效应解耦**；**② 无一致载体**：l\* 散布 L4-L24（13×3、4×2、7×2…）、qh\* 散布 **9 个不同 query 头**；层内 delta 也分散（最佳层 top1 份额仅 **0.128**、参与率 2.94/32）；**③ lens 构建非单调**：中层读出分歧巨大（L8 med lens JS 0.0969 = **29.1× final**）但 **L32-L35 收敛到 final 之下**（0.759/0.559）——KV 扰动在中层读出很大、被深层部分吸收，最终分布 JS 只是中层分歧的**残余**；**④ sham 位同样有大 delta（max 0.865）却几乎无分布效应（JS 0.000143）**——delta 能量是泛发的，分布读出是 logic 位特异的；**⑤ 校准干净**、T3 漂移 49.5123 与 3008-3015 逐位一致。

### 结论（谁在放大）
g7 K 路由破坏的下游放大**没有专用放大器**：单一 (层,头) 复原仅 8.5%%，载体散布层与头，中层分歧被深层收敛——**L3 门控是种子，其命运由分布式深层读出决定**。这与 3015 的影响-注意力 rho 0.45 缺口互补：影响的"放大"不是某个头的功劳，而是分布式深层处理+终层收敛的网络属性。呼应核心命题：头级重要性=关系属性，单点操作化关闭。

### 硬伤（1 笔，correction_note 登记）
run1 在 logit-lens 读出崩（model.norm 输出 fp32 vs lm_head 权重 bf16，F.linear dtype 不匹配 double≠BFloat16）——修复：cast 到 lm_head.weight.dtype；崩点在首个 lens 调用（T2a 位置已算完、判决前），无数据有效性影响；run2 权威。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3016/omega_p2j_amplification_trace_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3017 = A（主选）**深层收敛机制**——L28-35 谁消化了 KV 扰动（晚层头/MLP 对扰动的响应剖面 + 补偿方向测量）；B L31 次峰定位（晚层读出端）；C 情景性检验（同词异位 K,V 相似度 vs 异词）；D 重定向终点测量（K 擦除后注意力质量流向哪些位置/是否为内容词）。
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
if 'Phase 3016' not in prev:
    line = ('- Phase 3016 Omega-P2j: verdict '
            'amp_distributed_qwen; downstream '
            'amplification of g7 K-routing '
            'destruction is DISTRIBUTED + DEEP-'
            'CONVERGING: single-carrier restoration '
            '8.5%% (scatter -0.83..+0.39, delta '
            'energy decoupled from impact); l* '
            'spread L4-24, qh* 9 heads, per-layer '
            'top1 0.128; lens mid-stack 29.1x final '
            'at L8 then converges BELOW final '
            '(0.759/0.559 at L32/35); sham delta '
            '0.865 with JS 0.000143; run1 lens '
            'dtype crash (fp32 vs bf16); ledger '
            '155/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
