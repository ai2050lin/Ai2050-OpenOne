# -*- coding: utf-8 -*-
"""Phase 3018 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3018'
     r'\omega_p2l_dilution_decomposition_qwen')
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
assert verdict == 'decomp_cancellation_dominant_qwen', \
    verdict
assert res['anchor_all_ok'] is True
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_eff'] == 10
assert t2['g7'] == 7
assert t2['med_js_g7e'] == 0.003332
assert t2['ident_rel_med'] == 0.0
assert t2['med_E'] == 2.3154
assert t2['med_R'] == 3.5987
assert t2['med_credit'] == 2.511
assert t2['cancel_share_med'] == 1.8604
assert t2['cancel_ci95'] == [1.6619, 2.3645]
assert t2['gates_ok'] is True
t2b = res['T2b']
assert t2b['med_c_attn_x2']['10'] == -0.23051
assert t2b['med_c_mlp_x2']['10'] == -0.46931
assert t2b['med_c_mlp_x2']['8'] == -0.33141
assert t2b['med_c_mlp_x2']['20'] == -0.18162
t2c = res['T2c']
assert t2c['component_split']['best_layer'] == 10
assert t2c['component_split']['c_attn_x2_best'] == \
    -0.23051
assert t2c['component_split']['c_mlp_x2_best'] == \
    -0.46931
assert t2c['med_js_sham_g7e'] == 0.000143
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a21_3017'] is True
assert 'run1: crashed in T2c' in res['correction_note']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3018
           for m in led['measurements']):
    claim = (
        'Omega-P2l (plan v5 P2) - dilution vs '
        'cancellation decomposition of the log '
        'relative-error decay (3017: rel decays 3.7x '
        'while ||e|| grows 10.1x).  Exact telescoping '
        'identity on same-float arrays: log(rel_35/'
        'rel_4) = E - R = (R - E_nc) + credit, where '
        'E = sum log(||e_{l+1}||/||e_l||), R = sum '
        'log(||res_{l+1}||/||res_l||) (dilution budget), '
        'and credit = E_nc - E is the no-cancel '
        'counterfactual (drop the negative cross term '
        '2 D.e from the error growth).  PRIMARY = med '
        'cancel_share = credit/(R-E) over tags with '
        'R-E > 0.01; bootstrap 95pct CI resampling '
        'tags.  Verdict decomp_cancellation_dominant_'
        'qwen.  RESULTS: (i) med cancel_share = 1.8604 '
        '(bootstrap CI 1.66-2.36; all 10 effective '
        'tags in 1.51-2.72) - cancellation credit is '
        '2.511 nats vs the 1.283 nats total decay: '
        'the anti-parallel writes suppress MORE error '
        'growth than the entire observed decay, i.e. '
        'the no-cancel counterfactual has E_nc = 4.83 '
        '> R = 3.60 so WITHOUT the mid-stack '
        'anti-parallel band the relative error would '
        'GROW 3.4x instead of decaying 3.7x - the '
        'band FLIPS the sign of the outcome; (ii) '
        'component split: the MLP channel carries the '
        'larger anti-parallel cross term (med 2dM.e/'
        '||e||^2 at L10 = -0.469, band L8-L20; attn '
        'spike -0.231 at L10 only) - cancellation is '
        'MLP-dominated; (iii) telescoping identity '
        'gate idf = 0.0 (exact, same-float arrays); '
        '(iv) calibration clean (sham JS 0.000143), '
        'T3 drift 49.5123 bit-identical (a13 diff '
        '0.0).  CONCLUSION: the 3017 "dilution-'
        'dominated" reading is REVERSED by exact '
        'accounting - dilution (R = 3.60 nats) is '
        'the larger raw budget, but the net decay '
        '(1.28 nats) exists only because active '
        'mid-stack cancellation (credit 2.51 nats, '
        'MLP-carried) suppresses error growth '
        'beyond it; deep readout stability is an '
        'ACTIVE achievement of the mid-stack MLP '
        'writes, not passive residual growth.')
    meas = {
        'meas_id': 'meas3018_omega_p2l_dilution_'
                   'decomposition_qwen',
        'phase': 3018,
        'claim': claim,
        'verdict': verdict,
        'anchors': '21/21 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013; a18 3014; a19 '
                   '3015; a20 3016; a21 3017)',
        'artifacts': {
            'result': 'phase3018/omega_p2l_'
                      'dilution_decomposition_qwen/'
                      'result.json',
            'npz': 'phase3018/omega_p2l_'
                   'dilution_decomposition_qwen/'
                   'omega_p2l_dilution_'
                   'decomposition_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative; run1 crashed in '
                'T2c (NameError: Ls) after T2a/T2b '
                'completed - dead-code cleanup removed '
                'the Ls definition still used by the '
                'T2c best-layer loop; Ls restored, '
                'registered in correction_note.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 157
    l14['connects'].append({
        'meas_id': 'meas3018_omega_p2l_dilution_'
                   'decomposition_qwen',
        'phase': 3018,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2l: exact additive '
                        'decomposition of the log '
                        'relative-error decay - '
                        'cancellation credit 2.51 nats '
                        'EXCEEDS the total decay 1.28 '
                        'nats (med cancel_share 1.86, '
                        'CI 1.66-2.36); no-cancel '
                        'counterfactual E_nc 4.83 > '
                        'dilution R 3.60 so without '
                        'the mid-stack anti-parallel '
                        'band rel error would GROW '
                        '3.4x; cancellation is '
                        'MLP-carried (2dM.e/e2 at L10 '
                        '-0.469, band L8-20; attn '
                        'spike -0.231 at L10) - deep '
                        'readout stability is an '
                        'ACTIVE mid-stack MLP '
                        'achievement'})
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
if '## Phase 3018:' not in memo:
    sec = u'''## Phase 3018: Ω-P2l 稀释-抵消精确分解——抵消主导且翻转结局符号 [%(created)s]

**判决：`decomp_cancellation_dominant_qwen`**（run2 权威，144.9s，锚 21/21：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a15 l\\*=3、a16-a21 完整性）

### 设计（3017 机器 verbatim + 望远镜恒等式分解）
锚链/几何/生成/位置选择/two-step 协议全继承 3017（SEED_RND=3009 显式重建，真残差流 decoder-layer hook）。**精确加法分解**（同 float 数组望远镜闭合）：log(rel35/rel4) = E − R = (R − E_nc) + credit，E=Σlog(‖e_(l+1)‖/‖e_l‖)（误差增长预算），R=Σlog(‖res_(l+1)‖/‖res_l‖)（稀释预算），credit_l = ½log((1+q)/(1+2c+q))（c<0 时；**no-cancel 反事实**——去掉负叉积 2D·e 的误差增长）。PRIMARY = med cancel_share = credit/(R−E)（R−E>0.01 的 tag）；bootstrap 95pct CI（重采样 tag，N=10000，描述性）；恒等门 idf<0.05；门 nL≥8、n_eff≥8、JS 全正。判决映射：share≥2/3 → cancellation_dominant；<1/3 → dilution_dominant；否则 mixed。

### 核心结果（重复三遍）
**① 抵消信用超过全部衰减**：med cancel_share = **1.8604**（CI [1.66, 2.36]，10/10 有效 tag 全在 1.51-2.72）——credit **2.511** nats > 总衰减 **1.283** nats：反平行写入压制的误差增长比观测到的全部衰减还多；**no-cancel 反事实 E_nc = 4.83 > R = 3.60——没有中带反平行写入，相对误差会反向增长 3.4× 而非衰减 3.7×**：中带抵消翻转了结局的符号；**② 抵消由 MLP 通道承载**：med 2dM·e/‖e‖² 在 L10 = **−0.469**（反平行带 L8-L20 连续），attn 仅 L10 尖峰 −0.231、L4/L28-34 近零——分解把 3017 的"双组件反平行"精化为 **MLP 主导、attn 突触式单点**；**③ 恒等门 idf = 0.0**（望远镜逐位闭合，同 float 数组）；**④ 校准干净**（sham 0.000143）、T3 漂移 49.5123 与 3008-3017 逐位一致。

### 结论（3017 叙事被精确记账反转）
3017 的"稀释主导"读法被推翻：稀释（R=3.60 nats）确实是更大的原始预算，但**净衰减（1.28 nats）之所以存在，只因主动的中带抵消（credit 2.51 nats，MLP 承载）压过了误差增长**。深层读出稳定性是**中带 MLP 写入的主动成就**，不是被动残差增长的副产品。分布式深层读出命题获得机制载体：LLP 的 MLP 反平行带。

### 硬伤（1 笔，correction_note 全登记）
run1 在 T2c 崩（NameError: Ls）——patch 清理死代码时把 T2c best-layer 循环仍引用的 Ls 定义一并删除；T2a/T2b 数据已完整产出且与 run2 一致。Ls 恢复后 run2 权威。教训：删代码块前 Grep 其定义的全部消费者。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3018/omega_p2l_dilution_decomposition_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3019 = A（主选）**MLP 抵消带的身份**——L8-20 反平行 MLP 写入的下游读出（哪些 MLP 神经元/方向承载 credit，是否为通用误差抑制回路 vs logic 位特异）；B L31 次峰定位；C 情景性检验（同词异位 K,V 相似度 vs 异词）；D 重定向终点测量（K 擦除后注意力质量流向哪些位置/是否为内容词）。
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
if 'Phase 3018' not in prev:
    line = ('- Phase 3018 Omega-P2l: verdict '
            'decomp_cancellation_dominant_qwen; '
            'exact telescoping decomposition of the '
            'log rel-error decay: cancellation credit '
            '2.511 nats > total decay 1.283 nats (med '
            'cancel_share 1.8604, CI 1.66-2.36); '
            'no-cancel counterfactual E_nc 4.83 > R '
            '3.60 - without the mid-stack anti-'
            'parallel band rel error would GROW 3.4x; '
            'cancellation MLP-carried (2dM.e/e2 L10 '
            '-0.469, band L8-20); identity gate 0.0; '
            '1 crash run (Ls NameError after dead-'
            'code cleanup); ledger 157/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
