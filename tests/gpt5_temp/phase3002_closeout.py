# -*- coding: utf-8 -*-
"""Phase 3002 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3002'
     r'\omega_g2_robustness_source_qwen')
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
assert verdict == 'context_entangled_qwen', verdict
assert res['anchor_all_ok'] is True
t1 = res['T1']
assert t1['carry_bool'] is False
assert t1['carry_index'] == 0.6342, t1['carry_index']
assert t1['word_eff'] == 117.77 and t1['ctx_eff'] == 14.3
t2 = res['T2']
assert t2['selective'] is False
assert t2['L17']['xdir']['ratio'] == 1.5004
assert t2['L17']['sub_rnd_median'] == 0.0098
assert t2['L17']['full_rnd_median'] == 0.0094
t3 = res['T3']['trk']
assert t3['L4']['eraser_layer'] == 5
assert t3['L17']['eraser_layer'] is None
a9 = res['anchors']['a9_diffs']
assert a9['15'] < 1e-4 and a9['17'] < 1e-4, a9
assert t3['L4']['profile']['5']['trk_ratio'] == 0.456
assert t3['L17']['profile']['35']['trk_ratio'] \
    == 10.2551

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3002
           for m in led['measurements']):
    claim = (
        'Omega-G2 - qwen mirror of 3001: verdict '
        'context_entangled_qwen - the cross-model gap '
        'is a DOUBLE REVERSAL. (i) Signal source: qwen '
        'word carry only 63.4 pct of sep_f 185.7 '
        '(word_eff 117.8 vs ctx_eff 14.3; carry gate '
        '0.7 MISSED at 0.634) vs GLM4 86.4 pct; '
        'same-word context alone cuts qwen separation '
        'by 59 pct (185.7 -> 76.8) - qwen separation '
        'is context-sensitive. (ii) Propagation band '
        'position reversed: qwen L4 injection killed '
        'next layer (trk 4.72 -> 0.46 @L5) then '
        're-amplifies direction-free norm (trk 1.7 '
        '@L20, proj ~0) and re-gains xdir component '
        'deep (proj 2.3 @L35); L17 injection '
        'propagates ALL THE WAY (eraser=None, trk '
        '1.5 -> 10.3 @L35, proj 0.14 -> 3.24) - the '
        'exact mirror of GLM4 (L4 keep, L20 one-layer '
        'kill). (iii) qwen mid-band propagation is '
        'EXTREMELY direction-specific: xdir ratio '
        '1.21/1.50 (L15/L17, 2945 reproduced to '
        '4.3e-5/5.2e-6) vs per-cell span(Vt8) '
        'orthogonal random 0.011/0.0098 and '
        'full-space random 0.0099/0.0094 - ~150x '
        'specificity; GLM4 erasure was non-selective '
        '(everything ~0.01). Interpretation: GLM4 '
        'robustness = word-token carry + general '
        'mid-depth erasure; qwen sensitivity = '
        'context-entangled signal + a mid-band xdir-'
        'specific amplification band. Anchors 12/12 '
        '(a4/a5 2935 bit-level 7.2e-6/6.3e-6; a9 '
        '2945 raw-ratio repro).')
    meas = {
        'meas_id': 'meas3002_omega_g2_robust_source_qwen',
        'phase': 3002,
        'claim': claim,
        'verdict': verdict,
        'anchors': '12/12 (a1 2.2e-8; a2 0.0; a3 0.0; '
                   'a4 7.2e-6 2935; a5 6.3e-6 2935; '
                   'a7 9.9e-14; a9 2945 4.3e-5/5.2e-6 '
                   'raw ratio; a10 3001 integrity; '
                   'a11 0.0)',
        'artifacts': {
            'result': 'phase3002/'
                      'omega_g2_robustness_source_qwen/'
                      'result.json',
            'npz': 'phase3002/'
                   'omega_g2_robustness_source_qwen/'
                   'omega_g2_robustness_source_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative (run1 crashed: '
                'inj vec never armed in arm(); patched '
                'via Python, rerun from scratch). T3 is '
                'descriptive by prereg - no verdict '
                'branch. selective=False here means '
                'random directions do NOT propagate '
                '(sub 0.0098 << xdir 1.50), i.e. the '
                'qwen mid-band is xdir-specific '
                'amplification, NOT non-specific '
                'propagation.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 141
    l14['connects'].append({
        'meas_id':
            'meas3002_omega_g2_robust_source_qwen',
        'phase': 3002,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-G2 closed the qwen '
                        'mirror: double reversal vs '
                        'GLM4 - context-entangled signal '
                        '(carry 63 pct, same-word ctx '
                        'cuts 59 pct) + mid-band xdir-'
                        'specific amplification (150x '
                        'over random; L17 propagates to '
                        'L35) vs GLM4 word-carry + '
                        'general erasure'})
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
if '## Phase 3002:' not in memo:
    sec = u'''## Phase 3002: Ω-G2 qwen 鲁棒性来源镜像——上下文纠缠 + 中带 xdir 特异放大带 [%(created)s]

**判决：`context_entangled_qwen`**（run2 权威，13.4s，锚 12/12；a4/a5 与 2935 bit 级、a9 与 2945 raw ratio 复现 diff 4.3e-5/5.2e-6）

### 设计（预注册冻结）
3000/2945 机器 verbatim 的 3001 完全镜像（57 词 2887、dirs_word a1 vs 2927 2.2e-8、Vt8 a3 vs 2939 0.0、u35、dcks 2939、xdir=dcks_S@Vt8_S per-cell (57,2560)、单层 attn_in pos-1 注入）。T1 同款种子化随机伙伴 + 加法 2×2 分解（carry 门 0.7）；T2 L15/L17 s=2 K2 方向选择性（span(Vt8)⊥xdir 每细胞随机 ×2 + 全空间随机 ×1）；T3 描述项追踪 L4/L17（含注入层修改后捕获）。a9 用 raw ratio（3001 run4 教训制度化）。

### 结果（与 GLM4 3001 逐项对照）
- **T1 carry=False**：word_eff 117.8 = **63.4%%** of sep_f 185.7（门 0.7 未达；GLM4 86.4%%）vs ctx_eff 14.3（7.7%%）；m 表 132.1/98.7/−4.8/0.06——非 en 词侧投影≈0，分离几乎全由 en 词侧贡献，但换 ctx 使 en 词投影掉 25%%；**同词上下文单独即砍 59%%**（185.7→76.8，GLM4 砍 64%% 但其 null0 基线本就 72%% 承载）。qwen null0/f=41.6%% vs GLM4 72%%。
- **T2 极端方向特异性**：xdir ratio 1.2149/1.5004（L15/L17，2945 bit 级复现）vs span(Vt8)⊥xdir 每细胞随机 **0.011/0.0098**、全空间随机 0.0099/0.0094——**~150× 特异性**。GLM4 侧对照：xdir 0.0167 vs sub 0.009（2×，且全部≈0=全灭非选择）。判据 selective=False 语义=随机方向不传播（反向极端），非 GLM4 的"洗消非选择"。
- **T3 传播带位置镜像反转**：qwen **L4 注入下一层即杀**（trk 4.72→0.46@L5，proj 22.3→−0.07）→中带幅度无向恢复（trk 1.7@L20 而 proj≈0）→深层重获 xdir 分量（proj 2.3@L35）；**L17 注入全程传播 eraser=None**（trk 1.5→10.3@L35，proj 0.14→3.24）。GLM4 恰好相反：L4 幅度存续正交化、L19 下一层即杀（eraser=20）。**两模型都把强效应放在中带，但符号相反：qwen=放大带，GLM4=洗消带。**

### 结论（Ω-G 收官级）
跨模型鲁棒性差异 = **双重反转**：①信号源——GLM4 词 token 携带 86%%（上下文无关，注入无可移之物），qwen 上下文纠缠（63%%，同词上下文即砍 59%%）；②中带——GLM4 一般性即刻洗消（方向无选择），qwen xdir 特异放大带（150× over 随机、一路传到 L35）。"qwen 敏感"不是脆弱而是放大：注入方向被主动传播增强；"GLM4 鲁棒"=词携带硬承载+残差流中带主动清场。2945/3000 的 ratio 剖面（1.2-1.5）由此获得机制解释——它是放大带而非防御缺失。

### 硬伤（2 笔，correction_note 登记）
run1 崩溃：arm() 接收 vec_t 但从未装配 inj['vec']（3000 原脚本的 inj['vec']=xdir_t 装配行被移植时遗漏）——Python 补丁修复后 run2 从零重跑权威。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3002/omega_g2_robustness_source_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 141 / L14 %(l14)d。

**接续**：候选 3003：A（主选）qwen 放大带剂量律+镜像（-xdir 是否对称放大→放大带是方向性机制还是一般性增益）；B 3001/3002 差异的头级定位（h12 类比：qwen 中带放大载体头普查）；C 2989 T3 加密复测+k 剂量；D Ω-E 错误吸引子操作化。
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
if 'Phase 3002' not in prev:
    line = ('- Phase 3002 Omega-G2 qwen mirror: verdict '
            'context_entangled_qwen - double reversal '
            'vs GLM4 (carry 63 pct vs 86 pct; mid-band '
            'xdir-specific amplification ~150x over '
            'random, L17 propagates to L35 vs GLM4 L20 '
            'one-layer kill); ledger 141/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
