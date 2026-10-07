# -*- coding: utf-8 -*-
"""Phase 2999 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2999'
     r'\omega_f2b_sensitivity_band_glm4')
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
assert verdict == 'weak_partial_band_glm4', verdict
assert res['anchor_all_ok'] is True
t1 = res['T1']
assert t1['argmax_layer'] == 4 and t1['band_dead'] == [2, 4], t1
assert res['T3']['direction_specific'] is True
assert res['scan']['4']['ratio'] == 0.1285

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 2999
           for m in led['measurements']):
    claim = (
        'Omega-F2b GLM4 sensitivity band depth scan: '
        'weak_partial_band_glm4. Single-layer xdir '
        'injection s=2 K2 at every layer L2..L33 (2998 '
        'machine verbatim, dirs_g bit-anchored to 2996 '
        'npz). RESULT: NO strong band anywhere - max '
        'ratio 0.128 at L4, ratio>=0.1 only at L2/L4 '
        '(L3 dips to 0.069), all '
        'middle/late layers ride the 85.7 baseline '
        '(ratio <= 0.07). Dose stage at L4/L2: monotone '
        'sep decline but s_c (half decay) NOT reached '
        'within s<=6 (L4 sep 47.2 vs SEP_REF 42.9; L2 '
        '56.9) - reaching qwen-equivalent propagation '
        'needs ~3-6x the qwen scale AND only at the '
        'earliest layers. Mirror -xdir at L4: sep 85.3 '
        'vs plus 76.0, direction specific. VERDICT '
        'STRUCTURE: defense is GLOBAL + QUANTITATIVELY '
        'STRONGER (not layer-shifted: no ratio>=0.5 '
        'band; not pure dead zone: L2-4 weak band + '
        'monotone dose curves). Consistent with 2996 '
        'K2 registry at L7/10/13: GLM4 language '
        'mechanism consolidates EARLIER (sensitive '
        'band L2-4 < registry L7-13 < qwen L15-17), '
        'and the null0-carried word-token separation '
        '(2998) leaves little for context injection '
        'to move. Anchors 10/10 incl. a9 machine-'
        'transmits guard (L19 ref ratio 0.0166) and '
        'cross-run bit determinism (run2/run3 '
        'identical npz hash).')
    meas = {
        'meas_id': 'meas2999_omega_f2b_band',
        'phase': 2999,
        'claim': claim,
        'verdict': verdict,
        'anchors': '10/10 (a0 order; a8 collision; a1 '
                   'dirs bit-equal 0.0 x3; a7 unit '
                   '1e-16; a2 det 0.0; a6 sep_f '
                   '85.7>0; a3/a4 identity 2.5e-14; '
                   'a5 K-det 1.4e-14; a9 L19 ref '
                   'ratio 0.0166>=0.01)',
        'artifacts': {
            'result': 'phase2999/'
                      'omega_f2b_sensitivity_band_glm4/'
                      'result.json',
            'npz': 'phase2999/'
                   'omega_f2b_sensitivity_band_glm4/'
                   'omega_f2b_sensitivity_band_glm4.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': '3 runs: run1 crashed at a log format '
                'string (2-tuple into single %s) '
                'before any arm; run2 protocol-'
                'faithful; run3 authoritative '
                'replication (identical npz hash '
                '97bceae5). s_c non-existence within '
                's<=6 registered as-is (dose curve '
                'stored in result.T2).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 138
    l14['connects'].append({
        'meas_id': 'meas2999_omega_f2b_band',
        'phase': 2999,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'GLM4 defense GLOBAL+stronger: '
                        'max ratio 0.128 @L4 '
                        '(ratio>=0.1 only L2/L4); '
                        's_c > 6 at all '
                        'scanned layers; direction '
                        'specific'})
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
if '## Phase 2999:' not in memo:
    sec = u'''## Phase 2999: Ω-F2b GLM4 敏感带深度扫描——全局强洗消+早期弱带 [%(created)s]

**判决：`weak_partial_band_glm4`**（run3 权威，485.7s，锚 10/10；run2/run3 npz hash 相同=跨 run bit 级确定性）

### 设计（预注册冻结）
2998 机器 verbatim（98 cells、dirs_g 重建 a1 bit 级 0.00、Vt8/u39、in-session dcks、xdir 单层 attn_in pos-1 注入）。**T1 扫描**：s=2.0、K2、单层注入 L2–L33（相对深度 0.05–0.825），逐层 sep_med/ratio_med。**T2 剂量**：按 ratio 取 top-2 层，s∈{0.5..6.0}×K3，s_c=首个 sep_med<0.5×sep_f（插值）。**T3 镜像**：argmax 层与 L19 参考臂 −xdir。新增 a9 机器传播守卫（L19 ref ratio≥0.01，2998 实测 ~0.02）。

### 结果（判别三候选）
- **T1**：max ratio=**0.128 @L4**，ratio≥0.1 仅 **L2/L4** 两点（L3 回落到 0.069）；L5–33 全部贴基线（sep 85–86，ratio≤0.076）——**无 ratio≥0.5 的层位偏移带**；
- **T2**：L4/L2 剂量曲线单调下降（L4: 85.2→47.2；L2: 85.0→56.9）但 **s_c 在 s≤6 内不存在**（SEP_REF=42.9 未达）——达到 qwen 等效传播需 ~3–6× 剂量且只在最早期层；ratio 峰值 L2 1.058@s6、L4 0.648@s6；
- **T3**：L4 镜像 −xdir sep=85.3 vs +xdir 76.0（基线 85.7）——方向特异 ✓（非幅度伪影）。

### 解读（三候选判别完成）
GLM4 防御既非"层位偏移"（无强带），也非"纯死区"（L2/L4 弱带+单调剂量），而是 **全局强洗消+定量更强**：xdir 在任何层注入的传播都 ≤0.13（s=2）， Language 机制定型更早（弱带 L2–4 < 注册表 L7–13（2996 K2）< qwen 敏感带 L15–17——三卡交叉一致）；叠加 2998 的 null0 词 token 携带结构（上下文注入本无可移之物）。Ω-F2 跨模型章节收官：2945 机器是 qwen 特异的深度定位，GLM4 需 L2–4 早期注入+高剂量才可比较。

### 硬伤（1 笔）
run1 在 inj-vec log 行崩溃（`'%%s' %% tuple(shape)` 单占位符撞二元组 TypeError，实验臂未启动、无判决）；run2 协议保真；run3 权威复现。correction_note 登记。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase2999/omega_f2b_sensitivity_band_glm4/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 138 / L14 %(l14)d。

**接续**：候选 3000：A（主选）qwen↔GLM4 逐层传播比剖面（同 xdir 双模型定量防御带，收官 Ω-F 跨模型刻度）；B 2989 T3 加密复测+k 剂量；C Ω-E 错误吸引子操作化；D GLM4 L2–4 早期带定位细扫（s 网格+层内宽度）。
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
if 'Phase 2999' not in prev:
    line = ('- Phase 2999 Omega-F2b sensitivity band '
            'glm4: verdict weak_partial_band_glm4 '
            '(global strong erasure + weak early band '
            'L2-4, max ratio 0.128 @L4, s_c>6, anchors '
            '10/10); ledger 138/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
