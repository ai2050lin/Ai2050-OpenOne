# -*- coding: utf-8 -*-
"""Phase 3000 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3000'
     r'\omega_f_cross_model_profile')
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
assert verdict == 'cross_model_scale_gap_confirmed', verdict
assert res['anchor_all_ok'] is True
t4 = res['T4']
assert t4['pass'] is True and t4['area_ratio'] == 23.2, t4
assert res['T1']['band_n'] == 25
assert res['anchors']['a9_diffs'] == {'15': 0.0, '17': 0.0}

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3000
           for m in led['measurements']):
    claim = (
        'Omega-F CLOSE - cross-model propagation profile '
        '(cross_model_scale_gap_confirmed). Qwen 2945 '
        'machine verbatim scan: xdir single-layer '
        'injection s=2 K2 at L2..L29. Qwen propagates '
        'EVERYWHERE: band_n 25/28 layers ratio>=0.1, '
        'area 0.829, argmax L3 ratio 2.153 (sep goes '
        'NEGATIVE -13.0); a9 reproduced 2945 stored '
        'L15/L17 ratios bit-level (diff 0.0). GLM4 '
        '(2999 sealed npz, a10 hash-verified): band_n '
        '2 (L2/L4), area 0.036, max 0.1285. '
        'DIMENSIONLESS GAP: area ratio 23.2x, band '
        '25 vs 2 - qwen language readout is fragile '
        'to early/mid perturbation at BOTH scan arms, '
        'GLM4 is robust to all single-layer s=2 '
        'injections. Caveat registered: T3 mirror at '
        'qwen argmax L3 is NOT direction-specific '
        '(both signs collapse sep 185.7 -> -13.0/+15.8) '
        '- the very-early qwen band is partly '
        'magnitude-driven; the directional machine '
        'region is mid-depth L13-22 (2945 L15/17 '
        'lives there, a9 bit-level). Anchors 10/10.')
    meas = {
        'meas_id': 'meas3000_omega_f_close',
        'phase': 3000,
        'claim': claim,
        'verdict': verdict,
        'anchors': '10/10 (a0 57 words; a1 dirs 2.17e-08; '
                   'a2 0.0; a3 Vt8 0.0; a4/a5 ~7e-06; '
                   'a6 sep_f 185.70; a7 9.9e-14; a8 '
                   '2.8e-14; a9 2945 repro diff 0.0 x2; '
                   'a10 2999 hash True)',
        'artifacts': {
            'result': 'phase3000/'
                      'omega_f_cross_model_profile/'
                      'result.json',
            'npz': 'phase3000/'
                   'omega_f_cross_model_profile/'
                   'omega_f_cross_model_profile.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 protocol-faithful authoritative '
                '(no rerun needed; all anchors passed '
                'first pass). GLM4 side reused from '
                '2999 sealed npz (hash-verified), no '
                'recompute.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 139
    l14['connects'].append({
        'meas_id': 'meas3000_omega_f_close',
        'phase': 3000,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-F CLOSED: dimensionless '
                        'cross-model gap area 23.2x '
                        '(qwen 0.829 vs glm4 0.036), '
                        'band 25 vs 2 layers'})
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
if '## Phase 3000:' not in memo:
    sec = u'''## Phase 3000: Ω-F 收官——跨模型传播剖面，刻度差 23.2× [%(created)s]

**判决：`cross_model_scale_gap_confirmed`**（run1 权威一次通过，21.7s，锚 10/10；GLM4 侧复用 2999 sealed npz，a10 hash 校验）

### 设计（预注册冻结）
qwen 2945 机器 verbatim：57 词（2887）、dirs 重建（a1 vs 2927 **2.17e-08 正则值**）、Vt8（a3 vs 2939 **0.00**）、u35 读出、dcks=coords(null0)−coords(func)（2939）、xdir=dcks_S@Vt8_S、单层 attn_in pos-1 注入。**T1 扫描**：s=2 K2 全层 L2–L29（相对深度 0.083–0.833，对位 GLM4 L2–L33）。T2 剂量 top-2（s_c 用 qwen 本征 sep<100，仅内部描述量）；T3 镜像 argmax；**T4 跨模型无量纲对比**：max_ratio/band_n(ratio≥0.1 层数)/area(扫描均值)。

### 结果
- **qwen 处处传播**：band_n=25/28（ratio≥0.1），area=0.829，argmax L3 ratio=2.153 且 sep 转负（−13.0）；L13–22 中带 0.49–1.5（2945 L15/17 区）；
- **a9 跨卡复现**：L15/L17 s=2 ratio 与 2945 存储值 diff **0.0/0.0（bit 级）**——协议跨 Phase 冻结复现；
- **GLM4（2999）**：band_n=2、area=0.036、max=0.1285；
- **刻度差（无量纲）**：area 比 **23.2×**、band **25 vs 2** → T4 pass，Ω-F 章节以数字收官：qwen 语言读出对早/中带扰动脆弱，GLM4 对全部单层 s=2 注入鲁棒。

### 诚实注记（T3 非方向特异）
qwen argmax L3 镜像 −xdir sep=15.8 vs +xdir −13.0（基线 185.7）——**双符号均塌缩**：极早层 band 部分是幅度性扰动（大扰动破坏一切），方向特异机器区在中带 L13–22（a9 bit 级锁定）。T3 在预注册中仅为描述项，不影响判决分支；已入 correction/claim 如实登记。

### 硬伤（0 笔）
run1 即协议保真权威（全部锚首过，无重跑）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3000/omega_f_cross_model_profile/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 139 / L14 %(l14)d。

**接续**：候选 3001：A（主选）Ω-G 开题——GLM4 鲁棒性来源定位（L2–4 早期带 vs 词 token 携带的分离结构，哪个承重：词 token swap 实验）；B 2989 T3 加密复测+k 剂量；C Ω-E 错误吸引子操作化；D qwen L3 幅度性塌缩 vs L15 方向特异的分界定位。
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
if 'Phase 3000' not in prev:
    line = ('- Phase 3000 Omega-F close cross-model '
            'profile: verdict '
            'cross_model_scale_gap_confirmed (area '
            '23.2x, band 25 vs 2, a9 2945 repro bit-'
            'level, T3 non-specific caveat '
            'registered); ledger 139/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
