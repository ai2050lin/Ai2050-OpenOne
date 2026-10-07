# -*- coding: utf-8 -*-
"""Phase 3008 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3008'
     r'\omega_p2b_logic_kv_causal_qwen')
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
assert verdict == 'logic_sig_gen_only_qwen', verdict
assert res['anchor_all_ok'] is True
t1 = res['T1']
assert t1['sig_gen'] is True
assert t1['axes']['w2']['p_fam'] == 0.0002
assert t1['axes']['w1024']['p_fam'] == 0.0002
assert t1['axes']['w2']['n_logic'] == 114
assert t1['axes']['w2']['n_content'] == 1280
assert t1['axes']['w2']['sig'] is True
assert t1['axes']['w1024']['sig'] is True
t2 = res['T2']
assert t2['n_logic_pos'] == 11 and t2['n_content_pos'] == 22
assert t2['D'] == 2.0 and t2['p_fam'] == 0.40666
assert t2['kv_dominant'] is False
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3008
           for m in led['measurements']):
    claim = (
        'Omega-P2b (plan v5 P2) - 2993 logic-signature '
        'machinery ported to GENERATION state + first '
        'KV-ablation causal probe: 12 prompts x 256 '
        'greedy tokens, per-decode-step L34 residual '
        'projected on the held-out w2/w1024 axes '
        '(a1 identity 0.0 bit-level; a10 generation '
        'determinism rel 0.0). Verdict '
        'logic_sig_gen_only_qwen. RESULTS: (i) T1 '
        'GENERATION-STATE SIGNATURE HOLDS: logic tokens '
        'project HIGHER on the logic pole than content '
        'tokens on BOTH axes (med 20046.7 vs 16080.9 '
        'w2; 15445.0 vs 11589.8 w1024; D~3900, perm '
        'p_fam 0.0002 x2, n 114/1280) - the 2993 '
        'static-prompt axis carries its class split '
        'into autoregressive generation, positive-'
        'pole direction preserved; step-locking on w '
        'med|dw| logic 2118.1 < content 2361.0 '
        '(descriptive); (ii) T2 KV ABLATION NO '
        'ASYMMETRY: zeroing K/V at a logic prompt '
        'position vs matched content positions gives '
        'med div 30 vs 28 of 32 steps, D=2.0, perm '
        'p_fam 0.407 (nL 11 / nC 22) - NOT dominant; '
        'KEY CONFOUND REGISTERED: divergence is '
        'SATURATED (almost every position 22-31/32 '
        'diff tokens, sham ranges 0-31) - 32-step '
        'greedy decode is chaotic under any KV '
        'removal, so the metric has no headroom; KV '
        'causality needs a graded/softer intervention '
        'before any logic-position claim; (iii) T3 '
        '256-token drift: med |s(end)-s_pre| 49.51 '
        '(vs 45.97 at 64 tok), late std 25.80 < early '
        '34.84 (contraction persists at 4x length). '
        'CONCLUSION: logic signature is a genuine '
        'generation-state property (signature '
        'generalizes), but single-position KV '
        'ablation at full strength is too blunt a '
        'causal instrument.')
    meas = {
        'meas_id': 'meas3008_omega_p2b_logic_kv_'
                   'causal_qwen',
        'phase': 3008,
        'claim': claim,
        'verdict': verdict,
        'anchors': '11/11 (a0 2993 integrity; a1 axis '
                   'identity 0.0; a2 2.17e-08; a3 0.0; '
                   'a4 7.2e-06; a5 6.3e-06; a6 rel '
                   '0.0; a7 9.95e-14; a8 23/23 '
                   'single-token; a9 3007 integrity; '
                   'a10 gen det 0.0)',
        'artifacts': {
            'result': 'phase3008/omega_p2b_logic_kv_'
                      'causal_qwen/result.json',
            'npz': 'phase3008/omega_p2b_logic_kv_'
                   'causal_qwen/'
                   'omega_p2b_logic_kv_causal_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative (run1 anchor_fail: '
                'a1 recomputed w2/w1024 with unit() but '
                '2993 stores UNNORMALIZED mean-diff '
                'vectors; fixed to raw + relative gate, '
                'correction_note registered BEFORE '
                'run2).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 147
    l14['connects'].append({
        'meas_id': 'meas3008_omega_p2b_logic_kv_'
                   'causal_qwen',
        'phase': 3008,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2b: 2993 held-out logic '
                        'axis SEPARATES classes in '
                        'generation state (med proj '
                        'logic>content both axes, '
                        'p_fam 0.0002, n 114/1280) - '
                        'signature generalizes from '
                        'static prompts to '
                        'autoregressive decode; KV '
                        'ablation saturated (any '
                        'position 22-31/32 div) => no '
                        'logic-position causal claim, '
                        'graded intervention required'})
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
if '## Phase 3008:' not in memo:
    sec = u'''## Phase 3008: Ω-P2b 生成态逻辑签名 + KV 因果探测——签名泛化成立，全强度消融饱和无分辨力 [%(created)s]

**判决：`logic_sig_gen_only_qwen`**（run2 权威，175.3s，锚 11/11：a1 轴身份 **0.00e+00** 位级、a10 生成确定性 rel 0.0、a2/a3/a4/a5 全过）

### 设计（3007 生成机器 verbatim + 2993 轴位级加载）
12 prompt × 256 token 贪心 KV 生成；逐 decode 步捕获 **L34 decoder-layer 输入残差**（2993 res34 口径）投影到 held-out w2/w1024 轴（a1 从 2993 float32 res34 重算 = 0.0）+ final-norm s/c8；T1 生成态签名（置换 p_fam×2）；T2 KV 消融因果（transformers 5.x DynamicCache layers[i].keys/values 位零化，36 层全消）；T3 256-token 漂移。

### 核心结果（重复三遍）
**① 生成态逻辑签名强成立**：逻辑 token 在逻辑极上的投影**高于**内容 token（w2 med 20046.7 vs 16080.9；w1024 15445.0 vs 11589.8；D≈3900，置换 p_fam 均 **0.0002**，n 114/1280）——2993 静态 prompt 轴的类分离**带符号方向不变地迁移到自回归生成态**；w 轴步进 med|Δw| logic 2118 < content 2361（描述性）。**② KV 消融无不对称 + 测度饱和混杂**：逻辑词位 vs 匹配内容位消融 med 分歧 30 vs 28（/32 步），D=2.0，p_fam=0.407——不显著；但**几乎任意位置消融都致 22-31/32 步全分歧**（sham 0-31），32 步贪心解码对任意 KV 移除是混沌的，指标无 headroom——逻辑位因果主张需要**分级/软化干预**（KV 缩放而非零化）才有分辨力。**③ 256-token 漂移**：med |s(end)-s_pre| 49.51（64-token 时 45.97），晚窗 std 25.80 < 早窗 34.84——4× 长度下收缩保持，无失控。

### 结论
逻辑签名是真实的生成态属性（跨状态泛化确立）；但单位置全强度 KV 消融作为因果工具过钝——下一步 KV 缩放扫描（scale 0.2-0.8）恢复 headroom 后再问"逻辑位是否特异承重"。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3008/omega_p2b_logic_kv_causal_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3009 = A（主选）KV 缩放因果扫描（scale 网格恢复 headroom + 逻辑位特异性重测）；B P2b 扰动-恢复全网格；C Ω-A2 GLM4 家族 Base 对照。
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
if 'Phase 3008' not in prev:
    line = ('- Phase 3008 Omega-P2b: verdict '
            'logic_sig_gen_only_qwen; 2993 held-out logic '
            'axis separates logic vs content tokens in '
            'GENERATION state (p_fam 0.0002 x2, n '
            '114/1280); KV ablation saturated (any pos '
            '22-31/32 div, D=2.0 p=0.407) => graded '
            'intervention needed; drift contraction '
            'persists at 256 tok; ledger 147/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
