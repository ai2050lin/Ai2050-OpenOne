# -*- coding: utf-8 -*-
"""Phase 3010 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3010'
     r'\omega_p2d_logitlens_causal_qwen')
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
assert verdict == 'logitlens_logic_specific_qwen', verdict
assert res['anchor_all_ok'] is True
t2 = res['T2']
p00 = t2['per_scale']['0.00']
assert p00['D'] == 0.029677 and p00['p_fam'] == 0.0135
assert p00['n_logic'] == 11 and p00['n_content'] == 22
assert p00['med_js_logic'] == 0.032381
assert p00['med_js_content'] == 0.002704
assert p00['med_js_sham'] == 0.001738
for s in ('0.00', '0.25', '0.50', '0.75'):
    e = t2['per_scale'][s]
    assert e['D'] > 0 and e['p_fam'] < 0.05, (s, e)
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a12_3009'] is True
assert res['anchors']['a13_t3_drift_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3010
           for m in led['measurements']):
    claim = (
        'Omega-P2d (plan v5 P2) - READOUT-SWAP causal '
        'probe per the 3009 conclusion: keep the '
        'single-position KV intervention, replace '
        'token-divergence with the first-decode-step '
        'next-token distribution JS distance (nats, '
        'two-step protocol both arms).  Position '
        'selection chain identical to 3009 (SEED_RND='
        '3009, explicit rebuild).  Verdict '
        'logitlens_logic_specific_qwen.  RESULTS: '
        '(i) HEADROOM APPEARS WITH THE NEW READOUT: '
        'sham med JS 0.0017 vs logic 0.0324 at s=0 - '
        'the saturation that killed token-divergence '
        'is a property of that readout, not of the '
        'manifold; (ii) LOGIC-POSITION CAUSAL '
        'SPECIFICITY ESTABLISHED for the first time: '
        'PRIMARY s=0 D=medJS_L-medJS_C=0.0297 '
        'p_fam=0.0135 (nL=11 nC=22), and D>0 with '
        'p_fam<0.05 at ALL FOUR scales (0.0297/0.0229/'
        '0.0189/0.0217, p=0.0135/0.0209/0.0231/0.0222) '
        '- KV ablation at logic positions shifts the '
        'next-token distribution ~11x more than at '
        'matched content positions (vs sham ~19x); '
        '(iii) READOUT DISSOCIATION with 3009: c8 '
        'coordinate displacement ordered logic < '
        'content < sham while distribution JS orders '
        'logic >> content > sham - ablating a logic '
        'position keeps the trajectory near the '
        'baseline manifold yet bends its output '
        'distribution: logic tokens gate the '
        'distribution boundary, not the manifold '
        'location; argmax-flip rate logic 0.27 vs '
        'content 0.18 at s=0 concurs; (iv) baseline '
        'determinism: T3 drift 49.5123 bit-identical '
        'to 3008/3009 (a13 diff 0.0).  CONCLUSION: '
        'logic positions are causally load-bearing for '
        'the next-token distribution - the 3009 '
        'boundary-trigger hypothesis is confirmed '
        'under a continuous readout.')
    meas = {
        'meas_id': 'meas3010_omega_p2d_logitlens_'
                   'causal_qwen',
        'phase': 3010,
        'claim': claim,
        'verdict': verdict,
        'anchors': '13/13 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0)',
        'artifacts': {
            'result': 'phase3010/omega_p2d_logitlens_'
                      'causal_qwen/result.json',
            'npz': 'phase3010/omega_p2d_logitlens_'
                   'causal_qwen/'
                   'omega_p2d_logitlens_causal_qwen.'
                   'npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative, first pass, zero '
                'defects.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 149
    l14['connects'].append({
        'meas_id': 'meas3010_omega_p2d_logitlens_'
                   'causal_qwen',
        'phase': 3010,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2d: readout-swap (JS '
                        'distance of next-token '
                        'distribution) restores causal '
                        'headroom (sham 0.0017 vs logic '
                        '0.0324); logic-position '
                        'specificity established '
                        '(D=0.0297 p=0.0135 at s=0, '
                        'D>0 p<0.05 at 4/4 scales); '
                        'readout dissociation with 3009 '
                        '(c8: logic<content<sham vs JS: '
                        'logic>>content) - logic tokens '
                        'gate the distribution boundary '
                        'not the manifold location'})
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
if '## Phase 3010:' not in memo:
    sec = u'''## Phase 3010: Ω-P2d logit-lens 分布距离因果读出——换读出保干预，逻辑位因果特异性确立 [%(created)s]

**判决：`logitlens_logic_specific_qwen`**（run1 权威一次通过零硬伤，141.6s，锚 13/13：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a12 3009 完整性）

### 设计（3009 机器 verbatim + 换读出）
干预不变（单位置 K/V 缩放，36 层全层，s∈{0,0.25,0.5,0.75}），读出改为 two-step 协议下的下一 token 分布 JS 距离（nats，连续量无饱和地板）：两臂同为 prefill 后再喂 ids[-1] 一步读分布（对照公平），干预臂在两次 forward 之间缩放 K/V；位置选择链 3009 verbatim（SEED_RND=3009 显式重建，逐位可比）。

### 核心结果（重复三遍）
**① headroom 随读出出现**：sham med JS **0.0017** vs logic **0.0324**（s=0）——杀死 token 分歧读出的饱和是**该读出的性质**，不是流形的性质。**② 逻辑位因果特异性首次确立**：PRIMARY s=0 **D=0.0297，p_fam=0.0135**（nL=11/nC=22 达门），且**四尺度 D 全正 p 全<0.05**（0.0297/0.0229/0.0189/0.0217，p=0.0135/0.0209/0.0231/0.0222）——逻辑位 KV 消融使下一 token 分布偏移 ~11× 于匹配内容位（vs sham ~19×）。**③ 与 3009 读出解离**：c8 坐标位移序 logic < content < sham，而分布 JS 序 logic ≫ content > sham——消融逻辑位使轨迹贴近基线流形**但**弯曲其输出分布：**逻辑 token 门控分布边界而非流形位置**；argmax 翻转率 logic 0.27 vs content 0.18 同证。**④ 基线确定性跨 run 位级**：T3 漂移 49.5123 与 3008/3009 逐位一致（a13 diff=0.0）。3009 边界触发器假说在连续读出下获证。

### 结论
3009 方法论结论（换读出非换强度）正确且立刻兑现：逻辑位对下一 token 分布有因果承重。下一步定位层×尺度（哪一层/哪个子空间承载分布门控）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3010/omega_p2d_logitlens_causal_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3011 = A（主选）层×尺度 JS 剖面定位（K/V 分臂 × 层子集 × s 网格，找分布门控的承载层）；B steering-vector 响应读出（xdir/u35/w2 注入下的 JS 剂量）；C Ω-A2 GLM4 家族 Base 对照（需 GLM4-9B-Base 资产）。
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
if 'Phase 3010' not in prev:
    line = ('- Phase 3010 Omega-P2d: verdict '
            'logitlens_logic_specific_qwen; readout-'
            'swap (JS distance) restores headroom '
            '(sham 0.0017 vs logic 0.0324); logic-'
            'position causal specificity established '
            '(D=0.0297 p=0.0135 at s=0; D>0 p<0.05 at '
            '4/4 scales); readout dissociation vs 3009 '
            '(c8 vs JS opposite order) - logic tokens '
            'gate distribution boundary, not manifold '
            'location; T3 drift bit-identical to '
            '3008/3009; ledger 149/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
