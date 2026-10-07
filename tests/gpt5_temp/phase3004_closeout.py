# -*- coding: utf-8 -*-
"""Phase 3004 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3004'
     r'\omega_p4_operator_structure_qwen')
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
assert verdict == 'shared_lowrank_operator_qwen', \
    verdict
assert res['anchor_all_ok'] is True
band = res['A']['L17_band']
assert band['med_r1'] == 0.714665
assert band['med_medcos'] == 0.825766
assert res['A']['null_r1_rowgauss'] == 0.022317
a12a = res['anchors']['a12a_diffs']
assert a12a['L4'] == 0.0 and a12a['L17'] == 0.0
t3c = res['T3_corrected']
assert t3c['L4']['5']['trk_ratio'] == 3.9603
assert t3c['L17']['31']['trk_ratio'] == 91.8927

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3004
           for m in led['measurements']):
    claim = (
        'Omega-P4 (plan v5 P4) - layer transfer '
        'operator structure, qwen: verdict '
        'shared_lowrank_operator_qwen. Per-cell '
        'per-layer Delta (57x2560) from xdir s=2 '
        'injection at L4/L17, full SVD per layer. '
        'PRIMARY (L17 band, layers 18-25): the '
        'per-cell Delta matrix is strongly LOW-RANK '
        'SHARED - top-1 singular-value energy share '
        'r1 med 0.715 (range 0.66-0.74) and median '
        'per-cell |cos(delta, v1)| 0.826, vs '
        'row-normalized gaussian null r1 0.0223 '
        '(~32x); the L4-inj post-kill band is '
        'weaker-shared (r1 med ~0.47). So the '
        'amplification band is carried by a SHARED '
        'REDIRECT direction, not per-cell scatter - '
        'a meaningful conversion operator exists, in '
        'the 2985-corrected sense (low-rank shared '
        'but NOT orthogonal and NOT fixed: energy '
        'account shows the shared direction is NOT '
        'the injected xdir - aligned energy fraction '
        'with the source direction only 0.0075@L18 '
        'rising to 0.039@L31 while amplitude ratio '
        'grows 0.28->1.14, i.e. redirect with '
        'amplification, not preservation). MAJOR '
        'CORRECTION REGISTERED: 3001/3002 T3 stored '
        'trk_ratio/xdir_proj are a mis-declared '
        'quantity (forward_trk yields (1,n,hid); '
        'norm/sum axis=1 crossed CELLS per hidden '
        'dim); proven bit-level by a12a '
        'bug-identity anchor (3002 formula '
        'reproduces 3002 stored profiles at 0.0 '
        'from this run Dl); T3 was descriptive so '
        'both verdicts stand, but the numeric '
        'narratives are superseded by '
        'T3_corrected: qwen L4 injection loses '
        '~93 pct amplitude at L5 (55.5->3.96), '
        'L17 injection drops ~70 pct at L18 '
        '(55.5->16.7) then regrows to 1.66x '
        'injection by L31 (91.9) with growing '
        'xdir-aligned component (proj 3082->861); '
        'TRK_DROP eraser gate retired; GLM4 T3 '
        'recomputation = next phase.')
    meas = {
        'meas_id': 'meas3004_omega_p4_operator_'
                   'structure',
        'phase': 3004,
        'claim': claim,
        'verdict': verdict,
        'anchors': '13/13 (a1 2.2e-8; a2 0.0; a3 0.0; '
                   'a4 7.2e-6; a5 6.3e-6; a7 9.9e-14; '
                   'a9 4.3e-5/5.2e-6 raw; a10 3002 '
                   'integrity; a11 0.0; a12a 0.0/0.0 '
                   'bug-identity)',
        'artifacts': {
            'result': 'phase3004/omega_p4_operator_'
                      'structure_qwen/result.json',
            'npz': 'phase3004/omega_p4_operator_'
                   'structure_qwen/omega_p4_operator_'
                   'structure_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run4 authoritative (run1 S_IDX '
                'NameError; run2 3-D SVD TypeError; '
                'run3 a12 exposed the 3002/3001 T3 '
                'axis bug, verdict void by protocol; '
                'full history in correction_note).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 143
    l14['connects'].append({
        'meas_id': 'meas3004_omega_p4_operator_'
                   'structure',
        'phase': 3004,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P4: qwen mid-band '
                        'amplification carried by a '
                        'SHARED LOW-RANK redirect '
                        '(r1 0.715/medcos 0.826 vs null '
                        '0.022), shared direction is '
                        'NOT the injected xdir '
                        '(aligned frac 0.0075) = '
                        'redirect-with-amplification; '
                        '3001/3002 T3 axis bug '
                        'corrected in registry'})
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
if '## Phase 3004:' not in memo:
    sec = u'''## Phase 3004: Ω-P4 层间传递算子结构（qwen）+ 3001/3002 T3 轴序 bug 修正 [%(created)s]

**判决：`shared_lowrank_operator_qwen`**（run4 权威，14.2s，锚 13/13；a12a bug-identity 锚 0.0/0.0 bit 级）

### 设计（预注册冻结 + 两处显式修正）
3002 机器 verbatim（57 词、dirs a1、Vt8 0.0、xdir per-cell、pos-1 注入 s=2）。**修正一（run 前登记）**：方案 v5-P4 的"零 forward 重分析"前提不成立——3001/3002 npz 只存了均值 trk（1-D），per-cell per-layer Δ 未存盘，P4 须扩展重捕获。**T4 扩展捕获**：L4/L17 注入，逐层全量 Δ (57×2560) 存档。**A 算子结构**：每层 M_l=per-cell Δ 矩阵 SVD → r1=S1²/ΣS²（能量集中度）、erank、medcos=per-cell |cos(Δ,v1)|（共享方向对齐度）。**E 能量分账**：源=注入层 Δ，aligned_frac=(Δ·û_src)²/‖Δ‖²、amp_ratio=‖Δ‖/‖src‖。**修正二（run3 后准事后登记）**：a12 重定义为 a12a。

### 重大副产物：3001/3002 T3 轴序 bug（bit 级证实）
`forward_trk` 单 batch 捕获 stack 成 **(1, n, hid)**，而 T3 的 `norm(Dl, axis=1)`/`sum(Dl*xdir, axis=1)` 实际**沿细胞维**运算（per-hidden-dim 跨细胞范数/投影和）——存档的 trk_ratio/xdir_proj 是误声明量。**证据链**：run2 同形垃圾量 a12=0.0（真空通过）；run3 真量 a12=3.06e+03≈med‖xdir‖²(3079)；专用探针：3002 公式从本 run Dl 复现 3002 存档值 **0.0**（4.9e-5=纯 4dp 舍入），声明量差 3.06e+03。**影响**：T3 在 3001/3002 均为 descriptive，两判决（T1 决定）不受影响；但"即杀/放大带"数值叙事被 T3_corrected 取代；TRK_DROP=0.5 eraser 门基于错单位，退役；**GLM4 侧 T3 重算 = Phase 3005**。

### 结果
- **判决（主问题=L17 带 l18-25）**：r1 med **0.715**（0.66-0.74）、medcos **0.826**（0.76-0.86）vs 行归一高斯 null r1 **0.0223**（~32×）→ **共享低秩重定向**，非逐细胞散射；L4 注入后段 r1 med ~0.47（弱共享）；
- **共享方向 ≠ 注入 xdir**：aligned_frac 仅 0.0075@L18→0.039@L31，而 amp_ratio 0.28→1.14——**重定向伴随放大**，非保持（"守恒律"最简形式再度否证，与 2985/3001 T3 正交化证据一致）；
- **T3_corrected（qwen）**：L4 注入 55.5→3.96@L5（**93%% 次层衰减**，即杀存活且更强）；L17 注入 55.5→16.7@L18（先降 70%%）→…→**91.9@L31（1.66× 注入）**；xdir 对齐分量 3082→40.7→861 随幅度增长——放大带定性更锐：**共享重定向方向上的受控放大**。

### 结论（对方案 v5 的回答）
附件空白四的"层间转换矩阵"以 **2985 修正版**形式成立：传递算子**低秩、跨细胞共享**（r1 0.715/medcos 0.826），但**非正交、非固定、非守恒**——它是"把语言子空间位移重定向到共享目标方向并放大"的算子。单点操作化关闭的边界进一步收窄：**子空间级共享算子存在，方向级不守恒**。

### 硬伤（4 笔，correction_note 全登记）
run1 漏 S_IDX 常量；run2 3-D SVD 崩 + a12=0.0 真空通过；run3 a12=3.06e3 揭出 3001/3002 T3 轴序 bug（判决协议性作废）；run4 权威。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3004/omega_p4_operator_structure_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 143 / L14 %(l14)d。P1 前置：Qwen3-4B-Base 下载中（hub 元数据头被沙箱代理剥离 → 自写 urllib 流式下载器）。

**接续**：3005 = A（主选）GLM4 T3 轴序 bug 重算（3071 机器 + [0] 修复 + 双剖面），完成跨模型算子对照；B Base 下载完成后对齐三件套；C 生成轨迹记录器（v5-P2a）。
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
if 'Phase 3004' not in prev:
    line = ('- Phase 3004 Omega-P4 operator structure: '
            'verdict shared_lowrank_operator_qwen (L17 '
            'band r1 0.715/medcos 0.826 vs null 0.022; '
            'shared dir != xdir, aligned 0.0075 = '
            'redirect-with-amplification); MAJOR: '
            '3001/3002 T3 axis bug proven bit-level '
            '(a12a 0.0), T3_corrected registered, GLM4 '
            'recompute = 3005; ledger 143/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
