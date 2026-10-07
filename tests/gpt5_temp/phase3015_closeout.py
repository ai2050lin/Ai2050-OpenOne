# -*- coding: utf-8 -*-
"""Phase 3015 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3015'
     r'\omega_p2i_k_consumer_heads_qwen')
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
assert verdict == 'k_consumer_mixed_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['nh'] == 8
assert t2['med_js_all'] == 0.01384
assert t2['top1_med_share'] == 0.39
assert t2['p_perm'] == 1.0
assert t2['top_head_counts'] == {'1': 2, '7': 9}
assert t2['med_share_top8']['7'] == 0.39
assert t2['med_share_top8']['1'] == 0.1205
assert t2['top4_heads'] == [7, 1, 5, 6]
assert t2['med_n_eff_005'] == 3.0
assert t2['med_participation'] == 2.01
assert t2['gates_ok'] is True
assert res['T2b']['med_js_v_top']['7'] == 0.00526
assert res['T2b']['med_js_v_top']['1'] == 0.000724
assert res['T2c']['med_js_sham_K'] == 0.000202
assert res['T2c']['med_js_content_K'] == 0.000312
assert res['T2c']['n_sham'] == 11
assert res['T2c']['n_content'] == 22
td = res['T2d']
assert td['att_ok'] is True
assert td['gqa_groups'] == 8
assert td['spearman_impact_vs_attp'] == 0.4524
assert td['med_entropy_base'] == 0.8364
assert td['med_entropy_erased'] == 0.7
assert td['med_d_attp_top1_grp'] == 0.42466
assert td['med_attp_query_head_top8']['28'] == 0.98438
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a18_3014'] is True
assert 'run3: authoritative' in res['correction_note']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3015
           for m in led['measurements']):
    claim = (
        'Omega-P2i (plan v5 P2) - K-consumer head '
        'localization at the L3 logic-position gate: '
        'WHO consumes the destroyed K (3014: '
        'destruction carried by the K routing '
        'channel)?  The L3 key cache at p is read by '
        'layer-3 queries at positions > p; per-KV-'
        'head K erasure (Qwen3-4B GQA: 32 query '
        'heads share 8 KV heads; erasure unit = KV '
        'head = group of 4 query heads) maps the '
        'functional consumers; share(g,p) = '
        'JS_g(p)/JS_all(p); top1_med = med over '
        'positions of max share; within-position '
        'permutation null N=10000.  Verdict '
        'k_consumer_mixed_qwen.  RESULTS: '
        '(i) top1_med = 0.390 with permutation '
        'p = 1.0: NOT single-head concentrated '
        '(top1 < 0.5 gate) yet NOT flat - KV head 7 '
        'leads at 9/11 positions (med share 0.390) '
        'with minor partner head 1 (0.121), tail '
        'heads ~0.01-0.02, med effective heads '
        '(share>0.05) = 3.0/8, participation ratio '
        '2.01/8 - a LEADING-CONSUMER + EPISODIC '
        'BACKGROUND structure; (ii) K specificity at '
        'head level: V-side erasure of the same '
        'top-4 KV heads gives med JS only 0.0053/'
        '0.0007/0.0001/0.0002 (~2 orders below the '
        'K shares) - the consumption map is K-'
        'routing specific, matching 3014; (iii) '
        'per-head dose curves (top-4) are strongly '
        'position-heterogeneous with retain values '
        'above 1 at partial dose (5.84, 4.01, 1.83) '
        '- paradoxical amplification episodes, '
        'consistent with episodic competitive '
        'routing; (iv) attention readout (descriptive, '
        'non-gating): the leading consumer group g7 '
        'holds med attention mass 0.425 on p (query '
        'head 28: 0.984, head 7: 0.758), Spearman '
        'rho(impact, attention-on-p) = 0.45; K '
        'erasure LOWERS attention on p by 0.425 and '
        'SHARPENS the distribution (entropy 0.836 -> '
        '0.700) - mass is REDIRECTED to specific '
        'competitors, not flattened, refining the '
        '3014 zero-vs-half narrative at the '
        'attention level; (v) calibration clean '
        '(sham 0.000202, content 0.000312) and T3 '
        'drift 49.5123 bit-identical (a13 diff 0.0); '
        'npz8 9e68957c bit-identical across run2/'
        'run3.  CONCLUSION: the destroyed K is '
        'consumed by a mixed economy - one dominant '
        'KV-head group (g7, attention-anchored) plus '
        'a sparse episodic background; functional '
        'impact only moderately tracks attention '
        'mass (rho 0.45), so impact is partly '
        'downstream-amplified beyond raw attention; '
        'head-level K consumption is K-specific and '
        'episodic, mirroring the gate itself.')
    meas = {
        'meas_id': 'meas3015_omega_p2i_k_consumer_'
                   'heads_qwen',
        'phase': 3015,
        'claim': claim,
        'verdict': verdict,
        'anchors': '18/18 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013; a18 3014)',
        'artifacts': {
            'result': 'phase3015/omega_p2i_k_consumer_'
                      'heads_qwen/result.json',
            'npz': 'phase3015/omega_p2i_k_consumer_'
                   'heads_qwen/'
                   'omega_p2i_k_consumer_heads_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run3 authoritative; run1 verdict VOID '
                'by protocol - NH_EXPECT frozen at 32 '
                'but the L3 cache is GQA with '
                'num_kv_heads=8 (gates fail; T2d '
                'broadcast crash 32 vs 8); run2 full '
                'pass but its correction_note edit was '
                'a phantom (old text persisted on '
                'disk) - re-registered; all in '
                'correction_note.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 154
    l14['connects'].append({
        'meas_id': 'meas3015_omega_p2i_k_consumer_'
                   'heads_qwen',
        'phase': 3015,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2i: K-consumer map at '
                        'the L3 gate - MIXED economy '
                        '(top1 share 0.390, perm p=1.0, '
                        '3.0/8 effective KV heads, PR '
                        '2.01): leading consumer g7 '
                        '(9/11 positions, med share '
                        '0.390, attention on p 0.425; '
                        'query head 28 att 0.984) + '
                        'minor g1 + episodic '
                        'background; K-specific at '
                        'head level (V erasure of top '
                        'heads ~1e-4-5e-3); K erasure '
                        'REDIRECTS attention (entropy '
                        '0.836->0.700, mass leaves p) '
                        'not flatten; impact vs '
                        'attention rho 0.45 - partly '
                        'downstream-amplified'})
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
if '## Phase 3015:' not in memo:
    sec = u'''## Phase 3015: Ω-P2i K 消费头定位——混合消费经济 + 注意力重定向 [%(created)s]

**判决：`k_consumer_mixed_qwen`**（run3 权威，149.7s，锚 18/18：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 l\*=3、a16/a17/a18 完整性）

### 设计（3014 机器 verbatim + 逐 KV 头 K 擦除）
锚链/几何/生成/位置选择/two-step 协议全继承 3014（SEED_RND=3009 显式重建）。机制关键：L3 层位置 p 的 K 缓存只被 L3 自己的 query（位置>p）读取；Qwen3-4B 是 **GQA（32 query 头共享 8 KV 头，kv 头 g 服务 query 头 4g..4g+3）**，故擦除单位=KV 头。T2a PRIMARY：逐 KV 头 K 零擦除 × 11 logic 位；share(g,p)=JS_g/JS_all；top1_med + 位置内置换 null（N=10000）。T2b top-4 头剂量曲线 + V 擦除特异性对照；T2c sham/content 校准；T2d 注意力读出（query 头粒度+GQA 组聚合，descriptive）。

### 核心结果（重复三遍）
**① 消费经济=领先者+情景背景（mixed）**：top1_med=**0.390**、置换 p=**1.0**——非单头集中（<0.5 门）也非平坦；KV 头 7 在 9/11 位领先（med share 0.390）+次要伙伴头 1（0.121），尾部头 ~0.01-0.02；有效头（share>0.05）med **3.0/8**、参与率 **2.01/8**；**② 头级 K 特异性**：top-4 头 V 擦除 med JS 仅 0.0053/0.0007/0.0001/0.0002（比 K share 低约两个数量级）——消费图谱是 K 路由特异，与 3014 呼应；**③ 剂量强位异质**：top-4 头 retain 在部分剂量出现 **>1 悖论放大**（5.84/4.01/1.83）——情景性竞争路由的直接证据；**④ 注意力重定向而非均匀化**：领先组 g7 对 p 的注意力质量 0.425（query 头 28 达 **0.984**、头 7 0.758），impact vs 注意力 Spearman **0.45**（中度——影响部分在下游放大）；K 擦除后 p 上注意力 **降 0.425** 且分布**变锐**（熵 0.836→0.700）——质量流向特定竞争者，在注意力层精化了 3014 的"零=均匀化"叙事；**⑤ 校准干净**：sham 0.000202、content 0.000312；T3 漂移 49.5123 与 3008-3014 逐位一致，npz8 9e68957c 跨 run2/run3 位级重现。

### 结论（谁在消费被破坏的 K）
L3 logic 位 K 的消费不是单头集中也不是分布式——**一个注意力锚定的主导 KV 头组（g7）+ 稀疏情景背景（~3 有效头）**。功能影响只中度追踪原始注意力质量（rho 0.45），提示头输出下游存在放大。头级消费的 K 特异性 + 情景性精确镜像门控本身（3013）：消费结构=关系属性，非静态通路。

### 硬伤（2 笔，correction_note 全登记）
run1 跑完但**判决协议性作废（gates fail）**——NH_EXPECT 冻结为 32 而实测 KV 缓存是 GQA 8 头（nh_ok=False、T2b 跳过、T2d 广播崩 32 vs 8 被捕获）；修复 NH_EXPECT=8 + GQA 组映射 + T2d 重写 query 头粒度；run2 全程有效但 **correction_note 编辑幻影**（工具报成功而磁盘留旧文——环境缺陷对策：Python 补丁落盘 + Grep 复核）；run3 权威。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3015/omega_p2i_k_consumer_heads_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3016 = A（主选）**消费头下游放大定位**——g7（及 query 头 28）的 o_proj 输出对 L4+ 层的贡献追踪（谁把 K 路由破坏放大为全分布 JS）；B L31 次峰定位（晚层读出端）；C 情景性检验（同词异位 K,V 相似度 vs 异词）；D 重定向终点测量（K 擦除后注意力质量流向哪些位置/是否为内容词）。
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
if 'Phase 3015' not in prev:
    line = ('- Phase 3015 Omega-P2i: verdict '
            'k_consumer_mixed_qwen; K-consumer map = '
            'leading KV head g7 (9/11 pos, share '
            '0.390, att-on-p 0.425, qh28 0.984) + '
            'minor g1 + episodic background (3.0/8 '
            'eff, PR 2.01, perm p=1.0); K-specific '
            '(V erasure of top heads ~1e-4); K '
            'erasure REDIRECTS attention (entropy '
            '0.836->0.700) not flatten; impact vs '
            'att rho 0.45; run1 VOID (GQA nh 8 vs '
            '32), run2 note phantom edit; ledger '
            '154/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
