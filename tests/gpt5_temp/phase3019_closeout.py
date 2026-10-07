# -*- coding: utf-8 -*-
"""Phase 3019 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3019'
     r'\omega_p2m_mlp_band_identity_qwen')
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
assert verdict == 'mlp_band_distributed_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_negmass'] == 11
assert t2['g7'] == 7
assert t2['l_prim'] == 10
assert t2['topk'] == 32
assert t2['med_js_g7e'] == 0.003332
assert t2['bf16_ident_med'] == 5e-05
assert t2['negmass_med'] == 0.8465
assert t2['conc_med'] == 0.0667
assert t2['p_perm'] == 0.6018
t2b = res['T2b']
assert t2b['med_negmass']['10'] == 0.8465
assert t2b['med_negmass']['20'] == 0.5936
assert t2b['med_conc_top32']['8'] == 0.1116
assert t2b['jaccard_top32_med'] == 0.0323
assert t2b['jaccard_null_med'] == 0.0
t2c = res['T2c']
assert t2c['med_negmass_content'] == 0.9505
assert t2c['med_negmass_sham'] == 0.9132
assert t2c['med_negmass_logic'] == 0.8465
assert t2c['spec_ratio'] == 1.1229
assert t2c['med_js_sham_g7e'] == 0.000143
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a22_3018'] is True
assert 'run1: crashed in T2a' in res['correction_note']
assert 'run2: crashed at the u matmul' in \
    res['correction_note']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3019
           for m in led['measurements']):
    claim = (
        'Omega-P2m (plan v5 P2) - neuron-level '
        'identity of the MLP cancellation band '
        '(3018: cancellation credit 2.51 nats, '
        'MLP-carried, band L8-20, best L10 c_mlp '
        '-0.469).  SwiGLU decomposition: dM = '
        'W_down @ dh with h the down_proj input, '
        'so the per-neuron contribution to the '
        'cross term is s_j = 2 dh_j (w_j . e) / '
        '||e||^2 and sum_j s_j == c_mlp exactly '
        '(bf16-consistency gate |sum_s - ref| / '
        '(2 ||dM|| ||e|| / ||e||^2) med < 0.05).  '
        'PRIMARY = med over logic tags of the '
        'top-32 share of negmass at L10 with a '
        'within-tag permutation null (N=10000); '
        'specificity: same pipeline at content '
        'and sham positions.  Verdict '
        'mlp_band_distributed_qwen.  RESULTS: '
        '(i) the band is DISTRIBUTED - top-32 '
        'neurons (0.33 pct of 9728) carry only '
        '6.7 pct of the negative mass (per-tag '
        '0.059-0.095), permutation p = 0.60, and '
        'the top-32 sets barely overlap across '
        'tags (Jaccard 0.032 vs random null '
        '0.0) - no suppression circuit, a '
        'different neuron coalition per position; '
        '(ii) the band is GENERIC - negmass at '
        'content positions 0.951 and sham 0.913 '
        'vs logic 0.846 (spec_ratio 1.12), i.e. '
        'the same-strength anti-parallel MLP '
        'field fires wherever the g7 K routing '
        'is destroyed, not only at logic '
        'positions (their downstream JS differs: '
        'content 0.000118, sham 0.000143 vs '
        'logic 0.003332 - the readout, not the '
        'suppression, is logic-specific); '
        '(iii) bf16 identity gate 5e-05 - the '
        'neuron attribution is exact accounting; '
        '(iv) band profile: negmass decays '
        '0.91 (L8) to 0.59 (L20) while '
        'concentration stays flat 0.07-0.11.  '
        'CONCLUSION: the 3018 cancellation '
        'credit is carried by a DISTRIBUTED, '
        'GENERIC mid-stack MLP error-suppression '
        'field - the suppression machinery is '
        'always-on and position-agnostic; what '
        'makes the logic gate special is only '
        'what reads it out.  Third independent '
        'convergence on: head-level importance '
        'and mechanism are RELATIONAL properties, '
        'single-point operationalization closed.')
    meas = {
        'meas_id': 'meas3019_omega_p2m_mlp_band_'
                   'identity_qwen',
        'phase': 3019,
        'claim': claim,
        'verdict': verdict,
        'anchors': '23/23 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013; a18 3014; a19 '
                   '3015; a20 3016; a21 3017; a22 '
                   '3018)',
        'artifacts': {
            'result': 'phase3019/omega_p2m_'
                      'mlp_band_identity_qwen/'
                      'result.json',
            'npz': 'phase3019/omega_p2m_'
                   'mlp_band_identity_qwen/'
                   'omega_p2m_mlp_band_identity_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run3 authoritative; run1 crashed in '
                'T2a (NameError: Wf - dead-code '
                'cleanup removed the W_down cache '
                'block with the placeholder); run2 '
                'crashed at the u matmul (double vs '
                'float dtype); both registered in '
                'correction_note.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 158
    l14['connects'].append({
        'meas_id': 'meas3019_omega_p2m_mlp_band_'
                   'identity_qwen',
        'phase': 3019,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2m: SwiGLU neuron '
                        'attribution of the MLP '
                        'cancellation band - '
                        'DISTRIBUTED (top-32 of 9728 '
                        'neurons carry 6.7 pct of '
                        'negmass, perm p 0.60, cross-'
                        'tag top-32 Jaccard 0.032 vs '
                        'null 0.0) and GENERIC (negmass '
                        'content 0.951 / sham 0.913 vs '
                        'logic 0.846, spec_ratio 1.12) '
                        '- the anti-parallel MLP field '
                        'is an always-on position-'
                        'agnostic error-suppression '
                        'medium; only the downstream '
                        'readout is logic-specific '
                        '(JS 0.0033 vs 0.0001)'})
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
if '## Phase 3019:' not in memo:
    sec = u'''## Phase 3019: Ω-P2m MLP 抵消带身份——分布式泛化误差抑制场 [%(created)s]

**判决：`mlp_band_distributed_qwen`**（run3 权威，151.1s，锚 23/23：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 l\\*=3、a16-a22 完整性）

### 设计（3018 机器 verbatim + SwiGLU 逐神经元精确归因）
锚链/几何/生成/位置选择/two-step 协议全继承 3018（SEED_RND=3009 显式重建）。**SwiGLU 分解**：mlp 输出 M = W_down·h（h=down_proj 输入，inter=9728），故 dM = W_down·dh，**逐神经元叉积贡献 s_j = 2·dh_j·(w_j·e)/‖e‖²，Σs_j ≡ c_mlp 恒等**（bf16 一致性门：|Σs−ref|/(2‖dM‖‖e‖/‖e‖²) 中位 < 0.05）。PRIMARY = L10（3018 带峰，quasi-a-priori）top-32 神经元负质量份额 med，置换 null（tag 内置换 N=10000）；特异性对照：content/sham 位同管线（baseline+erased 双链）。判决映射：conc≥0.5 且 p≤0.01 且 spec<0.5 → specific_concentrated；spec≥0.5 → generic_concentrated；否则 distributed。

### 核心结果（重复三遍）
**① 抵消带=分布式，无抑制回路**：top-32（占 9728 神经元 0.33pct）仅承载负质量 **6.7pct**（11 tag 全在 0.059-0.095），置换 **p=0.60**；跨 tag top-32 集合 **Jaccard 0.032**（随机 null 0.0）——每个位一个不同神经元联盟，无专用 circuit；**② 抵消带=通用，非 logic 特异**：negmass content **0.951** / sham **0.913** vs logic **0.846**（spec_ratio **1.12**）——g7 K 路由毁在何处，反平行 MLP 场就在何处同样强度点火；差异在**下游读出**（JS：logic 0.00333 vs content 0.000118 / sham 0.000143）；**③ 归因是精确记账**：bf16 恒等门 **5e-05**（Σs≡c_mlp 逐位闭合）；**④ 带剖面**：negmass 沿带衰减（L8 0.911 → L20 0.594）而集中度平坦（0.07-0.11）——整个中带是均匀稀释的抑制介质。

### 结论（3018 credit 的承载者）
3018 的抵消信用（2.51 nats）由一个**分布式、泛化的中带 MLP 误差抑制场**承载——常开、位置无关、无核心神经元。逻辑门控的特殊性只在**读出端**：同样的抑制场在 content/sham 位点火却不产生可读分布效应。这是第三个独立收敛：**头级重要性与机制=关系属性，单点操作化关闭**。

### 硬伤（2 笔崩溃，correction_note 全登记）
run1 在 T2a 崩（NameError: Wf）——死代码清理补丁把 W_down 缓存块与未用占位函数一并删除（3018 Ls 教训复发：切片删除跨界）；run2 在 u 矩阵乘崩（double vs float dtype——e 是 float64 而 W 缓存 float32）；e 转 float32 后 run3 权威。教训：补丁删除块前必须列出块内全部定义的消费者；torch matmul 两侧 dtype 显式对齐。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3019/omega_p2m_mlp_band_identity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3020 = A（主选）**读出端特异性的定位**——同一抑制场下，什么使 logic 位的下游 JS 大 30×（L14+ 读出头对 g7 载体的敏感性 vs content 位）；B L31 次峰定位；C 情景性检验（同词异位 K,V 相似度 vs 异词）；D 重定向终点测量（K 擦除后注意力质量流向哪些位置/是否为内容词）。
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
if 'Phase 3019' not in prev:
    line = ('- Phase 3019 Omega-P2m: verdict '
            'mlp_band_distributed_qwen; SwiGLU neuron '
            'attribution of the MLP cancellation band '
            '(s_j = 2 dh_j (w_j.e)/e2, sum == c_mlp '
            'identity bf16 gate 5e-05); DISTRIBUTED: '
            'top-32 of 9728 neurons carry 6.7 pct of '
            'negmass, perm p 0.60, cross-tag top-32 '
            'Jaccard 0.032 vs null 0.0; GENERIC: '
            'negmass content 0.951 / sham 0.913 vs '
            'logic 0.846 (spec_ratio 1.12) - always-on '
            'position-agnostic suppression field, '
            'only the readout is logic-specific (JS '
            '0.0033 vs 0.0001); 2 crash runs (Wf '
            'NameError; double/float matmul dtype); '
            'ledger 158/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
