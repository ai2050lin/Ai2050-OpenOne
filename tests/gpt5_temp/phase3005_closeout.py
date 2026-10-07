# -*- coding: utf-8 -*-
"""Phase 3005 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3005'
     r'\omega_p5_operator_structure_glm4')
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
assert verdict == 'cell_scatter_operator_glm4', verdict
assert res['anchor_all_ok'] is True
band = res['A']['L19_band']
assert band['med_r1'] == 0.188855
assert band['med_medcos'] == 0.129813
assert band['med_trk_B'] == 0.75215
assert res['A']['null_r1_rowgauss'] == 0.013433
a12a = res['anchors']['a12a_diffs']
assert a12a['L4'] == 0.0 and a12a['L19'] == 0.0
t3c = res['T3_corrected_GLM4']
assert t3c['L19']['20']['trk_ratio'] == 1.2928
assert t3c['L19']['39']['xdir_proj'] == -0.0822
assert t3c['L4']['5']['trk_ratio'] == 4.1106

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3005
           for m in led['measurements']):
    claim = (
        'Omega-P5 (plan v5 P4 closure) - layer transfer '
        'operator structure, GLM4: verdict '
        'cell_scatter_operator_glm4. 3001 machine '
        'verbatim + 3004 T4/A/E machine (per-cell '
        'per-layer Delta 98x4096 from xdir s=2 '
        'injection at L4/L19, full SVD per layer, '
        'degeneracy amp-floor gate first). PRIMARY '
        '(L19 eraser band, layers 20-39): corrected '
        'amplitude med trk_B 0.752 (above the 0.05 '
        'floor, structure decodable); r1 med 0.189 and '
        'medcos med 0.130 vs row-gauss null r1 0.0134 '
        '(14x) - per-cell SCATTER, not shared low-rank '
        '(qwen band: 0.715/0.826/32x); r1/medcos decay '
        'monotonically from L20 (0.413/0.517) toward '
        'the scatter floor. T3_corrected_GLM4: L19 '
        'injection 15.07@19 -> 1.29@20 (91.4 pct '
        'amplitude loss) -> 0.74@39 plateau, xdir '
        'projection 227 -> -1.3@20 -> -0.08@39 (full '
        'directional clearance); L4 injection 73 pct '
        'loss at L5 then amplitude persists to L39 '
        'with cleared xdir component. CROSS-MODEL '
        'OPERATOR CONTRAST CLOSED: qwen mid-band = '
        'shared-direction redirect+amplification '
        '(constructive rewrite), GLM4 mid-band = '
        'per-cell scatter+clearance (destructive '
        'rewrite / whitenoise-ization); 3001 '
        'non-selective erasure mechanized. a12a '
        'bug-identity 0.0/0.0 - the 3004 axis-bug '
        'registration extends bit-level to GLM4; T3 '
        'descriptive, 3001 verdict stands.')
    meas = {
        'meas_id': 'meas3005_omega_p5_operator_'
                   'structure',
        'phase': 3005,
        'claim': claim,
        'verdict': verdict,
        'anchors': '11/11 (a1 0.0 x3 bit; a2 0.0; a3 '
                   '2.5e-14; a6 0.0 bit vs 2999; a7 '
                   '1.1e-16; a9 ratio19 0.0 bit raw; '
                   'a10 3001 integrity; a11 0.0; a12a '
                   '0.0/0.0 bug-identity)',
        'artifacts': {
            'result': 'phase3005/omega_p5_operator_'
                      'structure_glm4/result.json',
            'npz': 'phase3005/omega_p5_operator_'
                   'structure_glm4/'
                   'omega_p5_operator_structure_glm4'
                   '.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative, no defects.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 144
    l14['connects'].append({
        'meas_id': 'meas3005_omega_p5_operator_'
                   'structure',
        'phase': 3005,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P5: GLM4 eraser band = '
                        'per-cell scatter operator (r1 '
                        '0.189/medcos 0.130 vs null '
                        '0.0134), monotone decay from '
                        'L20 head (0.413/0.517); vs '
                        'qwen shared low-rank redirect '
                        '(0.715/0.826) - mid-band '
                        'operator structure is the '
                        'cross-model discriminator; '
                        'GLM4 T3 axis-bug corrected '
                        'bit-level (a12a 0.0)'})
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
if '## Phase 3005:' not in memo:
    sec = u'''## Phase 3005: Ω-P5 GLM4 传递算子结构——洗消=逐细胞散射，跨模型算子对照闭环 [%(created)s]

**判决：`cell_scatter_operator_glm4`**（run1 权威，113.4s，锚 11/11；a6 sep_f 与 a9 ratio19 均 **0.00e+00 bit 级** vs 2999；a12a bug-identity 0.0/0.0——3004 轴序 bug 登记扩展至 GLM4 bit 级）

### 设计
3001 机器 verbatim（98 cells、dirs_g a1 0.0×3、in-session dcks、xdir pos-1 注入、final-norm readout）+ 3004 T4/A/E 机器（L4/L19 注入、per-cell per-layer Δ 全量捕获 98×4096、逐层 SVD、能量分账）。判决门新增**幅度塌缩地板**（degeneracy 门先行）：band med trk_B < 0.05 → 结构不可判。实测 med trk_B=0.752 远超地板，可判。

### 结果（跨模型算子对照闭环，重复三遍）
**① GLM4 洗消带（L19 注入→l20-39）= 逐细胞散射算子**：r1 med **0.189**、medcos med **0.130** vs 行归一高斯 null r1 **0.0134**（14×）——对比 qwen 放大带的共享低秩（0.715/0.826，32×）；r1/medcos 从 L20 头部（0.413/0.517）**单调衰减**至散射地板，洗消=把 per-cell 注入位移打散到细胞特异方向。**② T3_corrected_GLM4**：L19 注入 15.07@19→**1.29@20（91.4%% 幅度损失）**→0.74@39 平台，xdir 投影 227→**−1.3@20→−0.08@39（方向完全清除，残余轻微负补偿）**；L4 注入 73%% 损失@L5 后幅度存续至 L39 而 xdir 分量已清。**③ 跨模型机制统一**：两模型中带都是重写区但算子结构相反——qwen=共享方向重定向+放大（**构造性重写**），GLM4=逐细胞散射+清除（**破坏性重写/白噪声化**）；3001 的"非选择性洗消"由此机制化：无共享擦除方向，因为每个细胞的位移被独立散射。**

### 结论（对方案 v5-P4 的最终回答）
层间传递算子存在但**跨模型异质**："转换矩阵"不是普适对象——qwen 中带可压缩为单一共享重定向方向（r1 0.715），GLM4 中带不可（r1 0.189 仅超 null 14×）。白盒手术的可行域因此分化：qwen 侧可做方向级干预（改共享重定向方向），GLM4 侧只能做幅度级干预（洗消带无方向结构可劫持）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3005/omega_p5_operator_structure_glm4/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3006 = A（主选）Base 对齐三件套（v5-P1 主攻，Qwen3-4B-Base 资产三 sha 已验证）：Base 侧语言轴扫描 + xdir 注入剂量 + 算子结构 → 对齐 delta 剖面（对齐训练雕刻了什么）；B 生成轨迹记录器（v5-P2a，自回归动力学第一块）；C L20 头部共享性衰减曲线精细扫描（GLM4 洗消启动时刻定位）。
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
if 'Phase 3005' not in prev:
    line = ('- Phase 3005 Omega-P5 operator structure '
            'GLM4: verdict cell_scatter_operator_glm4 '
            '(L19 band r1 0.189/medcos 0.130 vs null '
            '0.0134, monotone decay from L20 head; '
            'vs qwen shared low-rank 0.715/0.826 = '
            'cross-model operator contrast closed; '
            'T3_corrected_GLM4: L19 inj 91.4 pct loss '
            '@L20, xdir proj fully cleared; a12a '
            '0.0/0.0 extends axis bug to GLM4); '
            'anchors 11/11 bit-level; ledger 144/L14 '
            '%d.\n' % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
