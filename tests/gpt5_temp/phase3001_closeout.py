# -*- coding: utf-8 -*-
"""Phase 3001 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3001'
     r'\omega_g1_robustness_source_glm4')
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
assert verdict == 'word_carry_general_erasure_glm4', \
    verdict
assert res['anchor_all_ok'] is True
t1 = res['T1']
assert t1['carry_bool'] is True
assert t1['carry_index'] == 0.8643, t1['carry_index']
assert t1['word_eff'] == 74.1 and t1['ctx_eff'] == 19.73
t2 = res['T2']
assert t2['selective'] is False
assert t2['L19']['sub_rnd_median'] == 0.0094
assert t2['L19']['xdir']['ratio'] == 0.0166
t3 = res['T3']['trk']
assert t3['L19']['eraser_layer'] == 20
assert t3['L4']['eraser_layer'] is None
a9 = res['anchors']['a9_parts']
assert a9['ratio19'] == 0.0, a9

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3001
           for m in led['measurements']):
    claim = (
        'Omega-G1 - GLM4 robustness source localized '
        '(word_carry_general_erasure_glm4). T1 word-'
        'token swap, additive 2x2 decomposition of the '
        'partner-word context arms: word_eff 74.1 = '
        '86.4 pct of sep_f 85.7 vs ctx_eff 19.7 (23 '
        'pct) - class separation is predominantly '
        'CARRIED BY THE WORD TOKEN (null0 61.4 '
        'confirmed; same-word ctx halves it to 30.6). '
        'T2 direction selectivity at L17/L19: per-cell '
        'random span(Vt8) vectors orthogonal to xdir '
        'propagate LESS than xdir (median 0.009 vs '
        '0.017) - NO selective defense; mid-depth '
        'erasure is GENERAL to all pos-1 displacements '
        '(xdir survives ~2x better than matched '
        'random). T3 tracking dissociation: L19 '
        'injection killed in ONE layer (trk 1.96 -> '
        '0.32 at L20, xdir_proj 3.83 -> -0.02); L4 '
        'injection persists in norm through L39 (trk '
        '-> 1.99) but its xdir component is '
        'orthogonalized away (proj 0.12) - early '
        'displacements are absorbed, not erased. '
        'Anchors 10/10 incl. a9 2999 ratio19 '
        'bit-level (diff 0.0) and a6 sep_f bit-level.')
    meas = {
        'meas_id': 'meas3001_omega_g1_robust_source',
        'phase': 3001,
        'claim': claim,
        'verdict': verdict,
        'anchors': '10/10 (a1 0.0 x3; a2 0.0; a6 0.0 '
                   '2999 bit-level; a9 ratio19 0.0 '
                   '2999 bit-level, sep_n0 6.9e-4, '
                   'mirror 4.9e-3/3.1e-4; a5 0.0)',
        'artifacts': {
            'result': 'phase3001/'
                      'omega_g1_robustness_source_glm4/'
                      'result.json',
            'npz': 'phase3001/'
                   'omega_g1_robustness_source_glm4/'
                   'omega_g1_robustness_source_glm4.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run5 authoritative. 5 prereg/engineering '
                'defects registered in correction_note '
                '(xdir per-cell shape; vacuous ctx-label '
                'identity banned pre-verdict; phantom '
                'edit; rounded-vs-full-precision anchor '
                'comparison). T3 is descriptive by '
                'prereg - no verdict branch.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 140
    l14['connects'].append({
        'meas_id': 'meas3001_omega_g1_robust_source',
        'phase': 3001,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-G1 opened: GLM4 '
                        'robustness = word-token carry '
                        '(86 pct) + general mid-depth '
                        'erasure (non-selective); L20 '
                        'one-layer kill vs L4 '
                        'orthogonalize-and-keep'})
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
if '## Phase 3001:' not in memo:
    sec = u'''## Phase 3001: Ω-G1 GLM4 鲁棒性来源——词 token 携带 + 一般性中途洗消 [%(created)s]

**判决：`word_carry_general_erasure_glm4`**（run5 权威，150.0s，锚 10/10；a6/a9 与 2999 bit 级 diff=0.0）

### 设计（预注册冻结）
2999 机器 verbatim（98 cells、dirs_g bit 锚 2996、Vt8/u39、xdir per-cell (98,4096)、单层 attn_in pos-1 注入）。**T1 词 token swap**：种子化随机伙伴词上下文（SEED_PARTNER=3002，禁严格反类配对——那会使 ctx-labels sep ≡ −sep_D 永真）+ 加法 2×2 分解 m(word,ctx)：word_eff=列均值差、ctx_eff=行均值差，carry 门 word_eff≥0.7·sep_f。**T2 方向选择性**：L17/L19 s=2 K2，每细胞 span(Vt8)⊥xdir 随机方向 ×2（主对照）+全空间随机 ×1；selective 门 sub_rnd(L19)≥0.2 且 ≥2×xdir。**T3（描述项）逐层追踪**：注入层捕获修改后 pos-1 attn_in，trk_ratio(l)=med‖Δ_l‖/s。

### 结果
- **T1 carry=True**：word_eff **74.1（86.4%%）** vs ctx_eff 19.7（23%%）；m 表 58.1/37.0/−17.3/−35.7 全非退化；swap sep 75.7、same-word ctx 30.6（词自身重复反而腰斩）；
- **T2 selective=False**：随机子空间方向传播比 xdir **更弱**（med 0.0087–0.0094 vs xdir 0.0166–0.0195）——**无选择性防御**，中带洗消对所有 pos-1 位移一般有效（xdir 反而 ~2× 更耐）；
- **T3 分解注记**：L19 注入**下一层即杀**（trk 1.96→0.32@L20，xdir_proj 3.83→−0.02）；L4 注入**幅度存续至 L39**（trk→1.99）但 xdir 分量被正交化（proj→0.12）——早层"吸收不改向"，中带"主动即刻洗消"，机制分界清晰。

### 结论
GLM4 对 xdir 注入的鲁棒性 = **词 token 携带（86%%，上下文本就无可移之物）+ 一般性中途洗消（非方向选择性）**。2998 的"类分离由词 token 携带"获加法分解定量确认；2999 的"全局强洗消"细化为 L20 级单层即刻杀灭 + 早层正交化存续。Ω-G 后续：qwen 侧同款 swap 分解（跨模型镜像）。

### 硬伤（5 笔，均如实登记 correction_note）
run1 xdir 形状误用（per-cell (98,4096)）+ 判决前重设计 T1（严格反类配对→永真恒等式禁用）；run2 括号 SyntaxError 未执行；run3 Edit 幻影残留旧代码崩溃（Python 补丁修复）；run4 a9 用 4dp 舍入值对全精度参考（diff 4.66e-5=纯舍入 vs 门 1e-6）→判决协议性作废；run5 权威（a9 diff 0.0 确证舍入解释）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3001/omega_g1_robustness_source_glm4/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 140 / L14 %(l14)d。

**接续**：候选 3002：A（主选）qwen 同款词 token swap 加法分解（Ω-G 跨模型镜像，57 词协议适配）；B L4 存续 vs L20 即杀的机制分界定位；C 2989 T3 加密复测+k 剂量；D Ω-E 错误吸引子操作化。
''' % {'created': created,
           'script8': exe['script_sha256_8'],
           'result8': seal['result_sha256_8'],
           'npz8': seal['npz_sha256_8'],
           'exec8': seal['exec_sha256_8'],
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
if 'Phase 3001' not in prev:
    line = ('- Phase 3001 Omega-G1 GLM4 robustness '
            'source: verdict '
            'word_carry_general_erasure_glm4 (word_eff '
            '86 pct vs ctx 23 pct; erasure non-'
            'selective; L20 one-layer kill vs L4 '
            'orthogonalize-and-keep); ledger 140/L14 '
            '%d.\n' % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
