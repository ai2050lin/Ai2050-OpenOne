# -*- coding: utf-8 -*-
"""Phase 3009 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3009'
     r'\omega_p2c_kv_scale_causal_qwen')
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
assert verdict == 'kv_scale_saturated_qwen', verdict
assert res['anchor_all_ok'] is True
assert res['verdict_scale'] is None
t2 = res['T2']
p02 = t2['per_scale']['0.2']
p06 = t2['per_scale']['0.6']
p08 = t2['per_scale']['0.8']
assert p02['D'] == 4.5 and p02['p_fam'] == 0.0068
assert p06['D'] == 4.0 and p06['p_fam'] == 0.0344
assert p08['p_fam'] == 0.14069
for s in ('0.2', '0.4', '0.6', '0.8'):
    e = t2['per_scale'][s]
    assert e['headroom'] is False
    assert e['med_sham'] >= 22, (s, e['med_sham'])
assert p02['med_dc8']['logic'] < p02['med_dc8']['content']
assert p02['med_dc8']['content'] < p02['med_dc8']['sham']
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a11_3008'] is True

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3009
           for m in led['measurements']):
    claim = (
        'Omega-P2c (plan v5 P2) - graded KV-scale '
        'causal scan to restore the headroom lost to '
        'saturation in 3008: scale K,V at a prompt '
        'position (all 36 layers) by s in {0.2, 0.4, '
        '0.6, 0.8} after prefill, decode 32; logic '
        '(<=2/prompt) vs 2 rng content vs 1 sham '
        'positions, 3008 selection protocol verbatim. '
        'Verdict kv_scale_saturated_qwen. RESULTS: '
        '(i) SATURATION IS INTENSITY-INDEPENDENT: even '
        'at s=0.2 the SHAM positions diverge med 26/32 '
        'tokens (headroom gate <=4 never met at any '
        'scale) - 32-step greedy decode is chaotic to '
        'the EXISTENCE of a single-position KV '
        'perturbation, not to its strength; verdict '
        'scale = None, no logic-position causal claim '
        'possible on token-divergence; (ii) but logic '
        'vs content D>0 replicates at 3/4 scales '
        '(s0.2 D=4.5 p_fam=0.0068; s0.4 D=3.5 '
        'p=0.0343; s0.6 D=4.0 p=0.0344; s0.8 D=3.0 '
        'p=0.141 n.s.) - registered as '
        'saturation-confounded descriptive signal '
        'only; (iii) COORDINATE DISSOCIATION: med '
        'c8-distance logic < content < sham (65.56 < '
        '74.37 < 79.43 at s0.2) - ablating a logic '
        'position causes MORE token divergence but '
        'SMALLER manifold displacement: logic tokens '
        'act as boundary/argmax-flip triggers whose '
        'removal keeps the trajectory closer to the '
        'baseline manifold; (iv) T3 baseline '
        'determinism cross-run: drift med 49.5123 '
        'bit-identical to 3008. CONCLUSION: token-'
        'divergence under single-position KV '
        'intervention is a dead causal instrument at '
        'every strength; next causal probes must '
        'change the READOUT (logit-lens next-token '
        'distribution distance, single-layer/singular '
        'KV scaling, or steering-vector responses) '
        'rather than the intervention strength.')
    meas = {
        'meas_id': 'meas3009_omega_p2c_kv_scale_'
                   'causal_qwen',
        'phase': 3009,
        'claim': claim,
        'verdict': verdict,
        'anchors': '12/12 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det '
                   '0.0; a11 3008 integrity)',
        'artifacts': {
            'result': 'phase3009/omega_p2c_kv_scale_'
                      'causal_qwen/result.json',
            'npz': 'phase3009/omega_p2c_kv_scale_'
                   'causal_qwen/'
                   'omega_p2c_kv_scale_causal_qwen.'
                   'npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative, first pass, zero '
                'defects.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 148
    l14['connects'].append({
        'meas_id': 'meas3009_omega_p2c_kv_scale_'
                   'causal_qwen',
        'phase': 3009,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2c: KV-scale scan '
                        '(0.2-0.8) shows saturation is '
                        'intensity-independent (sham 26/'
                        '32 at s=0.2, headroom never '
                        'met) - single-position KV '
                        'token-divergence is dead as '
                        'causal instrument; logic D>0 '
                        'at 3/4 scales registered as '
                        'confounded signal; logic-'
                        'ablation coordinate '
                        'displacement SMALLER than '
                        'content/sham (boundary-'
                        'trigger dissociation)'})
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
if '## Phase 3009:' not in memo:
    sec = u'''## Phase 3009: Ω-P2c KV 缩放因果扫描——饱和与强度无关，token 分歧作为因果工具退役 [%(created)s]

**判决：`kv_scale_saturated_qwen`**（run1 权威一次通过零硬伤，319.4s，锚 12/12：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、a11 3008 完整性）

### 设计（3008 机器 verbatim + 零化→缩放）
12 prompt × 256 token 贪心基线；干预=s 缩放目标位 K/V（36 层全层，prefill 后），s∈{0.2,0.4,0.6,0.8}；位置选择协议 3008 verbatim（logic ≤2/prompt、2 rng content、1 sham）；headroom 门=sham 中位分歧 ≤4/32；判决尺度级联（0.6→0.4→0.2→0.8）。

### 核心结果（重复三遍）
**① 饱和与干预强度无关**：即使 s=0.2（仅 20%% KV），sham 位置中位分歧仍 **26/32**（四尺度 sham 26/26/25/22，headroom 门 ≤4 从未达成）——32 步贪心解码对**单位置 KV 扰动的存在本身**混沌，与强度无关；判决尺度=None，token 分歧读出下任何逻辑位因果主张都不可能。**② 逻辑位正向信号在 3/4 尺度一致**（D=4.5/3.5/4.0/3.0，p_fam=0.0068/0.0343/0.0344/0.141）——但被饱和混杂覆盖，只登记为描述性信号。**③ 坐标解离（新现象）**：med c8 偏差 **logic < content < sham**（65.56 < 74.37 < 79.43 @s0.2，四尺度同序）——消融逻辑位导致**更多 token 分歧但更小流形偏移**：逻辑 token 是边界/argmax 翻转触发器，其移除使轨迹更贴近基线流形。**④ 基线确定性跨 run 位级**：T3 漂移 49.5123 与 3008 完全一致。

### 结论
单位置 KV 干预 + token 分歧读出 = 死因果工具（与强度无关）；下一步因果探针必须换**读出**（logit-lens 下一 token 分布距离、单层/单奇异值 KV 缩放、steering-vector 响应）而非换强度。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3009/omega_p2c_kv_scale_causal_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3010 = A（主选）logit-lens 分布距离因果读出（JS/KL 于下一 token 分布，换读出保干预）；B 单层×单奇异 KV 缩放定位（哪一层/哪个子空间承载边界触发）；C Ω-A2 GLM4 家族 Base 对照（需 GLM4-9B-Base 资产）。
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
if 'Phase 3009' not in prev:
    line = ('- Phase 3009 Omega-P2c: verdict '
            'kv_scale_saturated_qwen; KV saturation is '
            'intensity-independent (sham 26/32 at s=0.2, '
            'headroom never met) - token divergence dead '
            'as causal instrument; logic D>0 at 3/4 '
            'scales (confounded, registered); coordinate '
            'dissociation logic<content<sham (boundary-'
            'trigger); ledger 148/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
