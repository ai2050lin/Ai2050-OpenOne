# -*- coding: utf-8 -*-
"""Phase 3003 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3003'
     r'\omega_g3_dose_mirror_qwen')
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
assert verdict == 'asymmetric_amplification_qwen', \
    verdict
assert res['anchor_all_ok'] is True
t1 = res['T1']
assert t1['linear'] is True
assert t1['dose']['L15']['lin_max_rel'] == 0.1293
assert t1['dose']['L17']['lin_max_rel'] == 0.1152
assert t1['sub_rnd0_dose_L17'] == {'1.00': 0.0094,
                                   '2.00': 0.0101}
t2 = res['T2']
assert t2['symmetric'] is False
assert t2['rectifying'] is False
assert t2['mirror']['L17']['ratio'] == 0.8288
a9 = res['anchors']['a9_diffs']
assert a9['15'] < 1e-4 and a9['17'] < 1e-4, a9

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3003
           for m in led['measurements']):
    claim = (
        'Omega-G3 - qwen amplification band ontology: '
        'verdict asymmetric_amplification_qwen. The '
        'mid-band xdir channel is (i) DOSE-LINEAR: '
        'through-origin fit max relative residual '
        '0.129 (L15) / 0.115 (L17) over s in '
        '{0.5,1,2,3}, ratio up to 1.90 at s=3 - a '
        'near-linear gain, not a threshold circuit; '
        '(ii) xdir-SPECIFIC: the span(Vt8)-orthogonal '
        'random direction does NOT grow with dose '
        '(0.0094 at s=1 vs 0.0101 at s=2) while xdir '
        'doubles - the 150x specificity of 3002 is '
        'dose-stable; (iii) SIGN-ASYMMETRIC: -xdir '
        'propagates at 0.819/0.829 (L15/L17) vs +xdir '
        '1.215/1.500 (34-45 pct weaker, sym_parts '
        '0.326/0.448 > 0.2 gate) - neither '
        'linear_symmetric (passive no-erase) nor '
        'rectifying (one-sign gate) fits. Both signs '
        'COMPRESS the separation (mirror sep 92.3/77.4 '
        'vs sep_f 185.7; +xdir drives it to 14.8/-6.4 '
        'at s=2, -14.4 at s=3): the mid band rewrites '
        'pos-1 language-subspace displacements toward '
        'an intermediate state, +xdir at full gain, '
        '-xdir at partial gain. Spectrum: GLM4 mid '
        'band = both-sign erasure (ratio ~0.01); '
        'qwen mid band = asymmetric partial-gain '
        'compression. An empty-language-subspace '
        'control stays flat under dose, so the gain '
        'is a property of the xdir channel, not of '
        'general mid-band nonlinearity. Anchors 12/12 '
        '(a4/a5 2935 bit-level; a9 2945 raw-ratio '
        'repro 4.3e-5/5.2e-6 from the s=2 dose '
        'medians).')
    meas = {
        'meas_id': 'meas3003_omega_g3_dose_mirror',
        'phase': 3003,
        'claim': claim,
        'verdict': verdict,
        'anchors': '12/12 (a1 2.2e-8; a2 0.0; a3 0.0; '
                   'a4 7.2e-6; a5 6.3e-6; a7 9.9e-14; '
                   'a9 4.3e-5/5.2e-6 raw; a10 3002 '
                   'integrity; a11 2.8e-14)',
        'artifacts': {
            'result': 'phase3003/omega_g3_dose_mirror_'
                      'qwen/result.json',
            'npz': 'phase3003/omega_g3_dose_mirror_'
                   'qwen/omega_g3_dose_mirror_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative. T3 descriptive '
                'field defect registered: glm4_3001.'
                'xdir_ratio_L19 mistakenly holds '
                'word_eff 74.1 (field mislabel, no '
                'verdict/anchor impact). Dose-median '
                'a9 confirms 3002 K=2 values to '
                '5.2e-6.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 142
    l14['connects'].append({
        'meas_id': 'meas3003_omega_g3_dose_mirror',
        'phase': 3003,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-G3: qwen mid band = '
                        'xdir-specific dose-linear '
                        'SIGN-ASYMMETRIC compression '
                        'channel (both signs compress '
                        'separation; -xdir 34-45 pct '
                        'weaker) - excludes passive '
                        'symmetric no-erase AND '
                        'rectifier; GLM4-vs-qwen '
                        'mid-band spectrum completed'})
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
if '## Phase 3003:' not in memo:
    sec = u'''## Phase 3003: Ω-G3 qwen 放大带本体——剂量线性 + xdir 特异 + 符号不对称压缩通道 [%(created)s]

**判决：`asymmetric_amplification_qwen`**（run1 权威，14.0s，锚 12/12；a9 2945 raw ratio 复现 4.3e-5/5.2e-6 且由 K=3 剂量中位数独立确认 3002 值）

### 设计（预注册冻结）
3002 机器 verbatim（57 词 2887、dirs a1 2.2e-8、Vt8 0.0、u35、xdir per-cell、pos-1 注入）。**T1 剂量律**：L15/L17 × s∈{0.5,1,2,3} × K3，过原点线性拟合最大相对残差门 0.15；sub_rnd0 剂量对照（正交随机方向随剂量是否增长）。**T2 镜像对称**：−xdir s=2 K2，对称门 |r−/r+−1|≤0.2、整流门 r−/r+≤0.3。**T3 描述项**：跨模型对照表（3001 sealed 引用）。

### 结果
- **剂量线性两层皆达**：lin_max_rel 0.129/0.115（≤0.15）；ratio 随 s 近线性升至 1.90/1.87@s=3；sep 单调压缩（L15: 176.6→8.2；L17: 146.0→−14.4，大剂量翻转分离符号）；
- **sub_rnd0 剂量对照平坦**：0.0094@s=1 vs 0.0101@s=2——随机方向不随剂量增长，**3002 的 ~150× 特异性是剂量稳定的通道属性**，非一般性非线性；
- **镜像不对称**：−xdir ratio 0.819/0.829 vs +xdir 1.215/1.500（弱 34-45%%，sym_parts 0.326/0.448>0.2）；但非整流（0.674/0.553>0.3 门）。**两个方向都压缩分离**（mirror sep 92.3/77.4 vs sep_f 185.7）。

### 结论
qwen 中带 = **xdir 特异、剂量近线性、符号不对称的压缩通道**：把 pos-1 语言子空间位移向中间态重写，+xdir 全增益、−xdir 部分增益。排除"被动对称无洗消"（线性对称分支）与"主动整流电路"（0.3 门）。**GLM4-qwen 中带谱系完成**：GLM4=双向全杀（ratio~0.01，3001）↔ qwen=不对称部分增益压缩（0.82-1.90）。"qwen 敏感"的最终定性：中带对 xdir 通道的线性增益不对称——不是放大器，是**不对称压缩重写器**（与 2985 重写非旋转、2992 重定向谱系一致）。

### 硬伤（1 笔，correction_note 登记）
T3 描述性字段错位：glm4_3001.xdir_ratio_L19 误填 word_eff 74.1（字段标签错，无判决/锚影响，如实登记）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3003/omega_g3_dose_mirror_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 142 / L14 %(l14)d。

**接续**：方案 v5 已发布（`research/gpt5/docs/plan_v5_dynamic_manifold_control.md`，含附件五大空白逐条判定与三处事实校正）。候选 3004：A（主选，方案 v5-P1）Qwen3-4B-Base 对照三件套（下载前置）；B（方案 v5-P4，零 forward 可即行）层间传递算子 Dl 低秩+能量分账（3001/3002 trk 剖面重分析）；C 方案 v5-P2a 生成轨迹记录器；D 方案 v5-P3 跨语言同构 Procrustes。
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
if 'Phase 3003' not in prev:
    line = ('- Phase 3003 Omega-G3 dose+mirror: verdict '
            'asymmetric_amplification_qwen (dose-linear '
            'lin_rel 0.13/0.12; sub_rnd dose-flat = xdir-'
            'specific; mirror 34-45 pct weaker, both '
            'signs compress sep = asymmetric compression '
            'channel); plan_v5 published; ledger 142/'
            'L14 %d.\n' % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
