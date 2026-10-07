# -*- coding: utf-8 -*-
"""Phase 3091 closeout (idempotent):
Ledger(measurements + L14) -> MEMO append ->
audit 52 addendum -> wlog -> MEMORY.md ->
closeout_log."""
import hashlib
import io
import json
import os
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3091'
     r'\omega_p88_continuum_l38_sensitivity')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
z = np.load(R + r'\omega_p88_continuum_l38_'
            r'sensitivity.npz',
            allow_pickle=False)
verdict = res['verdict']
assert verdict == 'continuum_robust_l38', verdict
assert res['forward_free'] is True
assert int(res['forwards']) == 0
assert bool(z['ANCH_86_RHOT_BIT'])
assert bool(z['ANCH_GLM4_L37_BIT'])
assert bool(z['ANCH_STOUFFER_BIT'])
assert str(z['VERDICT']) == verdict
rho_a = float(z['RHO_T_REPL'])
p_a = float(z['P_T_REPL'])
rho_b = float(z['RHO_T_N15'])
p_b = float(z['P_T_N15'])
st_s = float(z['STOUFFER_SIGNED'])
st_u = float(z['STOUFFER_UNSIGNED'])
o.append('loaded result/seal/npz ok')

# ---------- Ledger ----------
meas = {
    'meas_id': 'meas3091_omega_p88_'
               'continuum_l38_sensitivity',
    'phase': 3091,
    'claim':
        'Omega-P88 (plan branch A of the '
        '3089 same-cell verdict) - '
        'continuum L38 layer sensitivity, '
        'forward-free, three preregistered '
        'arms frozen in execution.json '
        'before any rho computation.  '
        'Arm A (n=12 replacement): GLM4 '
        'units s_lo/T_med/U_med/MIG moved '
        'from L37 to 3089 L38 values '
        '(s_lo 0.8281/0.8281/0.9031 from '
        'top3 0.8281/0.9031/0.9117; T_med '
        '+0.2465/+0.2379/+0.0528) -> '
        'rho_T +0.7413 p=0.0077 SIG, '
        'rho_smean +0.6643 p=0.0233 SIG, '
        'rho_U +0.3497 ns, rho_MIG +0.3706 '
        'ns.  Arm B (n=15, L37 units kept '
        '+ L38 appended): rho_T +0.7821 '
        'p=0.00085 SIG.  VERDICT '
        'continuum_robust_l38 - the 3088 '
        'spectrum->migration continuum is '
        'layer-robust; the L37/L38 '
        'criterion split is fully closed '
        'at the behavior level.  '
        'Attenuation not reversal: L38 '
        'moves GLM4 s_lo toward the trunk '
        'side while T_med BC drops to '
        '0.053, pulling rho 0.902->0.741.  '
        'Arm C (descriptive): unsigned '
        'Stouffer reproduced bit-level '
        '(3.1501); sign-aware variant '
        '+1.1012 ns - the preregistered '
        'unsigned form is '
        'direction-agnostic and '
        'overstates aggregate evidence; '
        'the sign-aware G_DS count gate '
        '(1/6) remains the decision gate.  '
        'CAVEATS: single-model sensitivity '
        '(GLM4 only); frozen argsort '
        'spearman is tie-order sensitive '
        '(4 tied s_lo groups) - unit '
        'order fixed to the 3088 '
        'sorted-key convention and '
        'recorded in execution.json '
        '(0.8951-vs-0.9021 lesson).',
    'verdict': verdict,
    'anchors':
        'z86 rho_T recompute bit-exact '
        '(0.9020979021, sorted-key unit '
        'order); GLM4 L37 provenance '
        'bit-exact (z87-derived == z86 '
        'stored units); z89 sanity (20922 '
        'fw, L_INJ=38, mixed_absent); '
        'unsigned Stouffer bit-exact vs '
        'z89 - see npz ANCH_* keys',
    'artifacts': {
        'result': 'phase3091/'
                  'omega_p88_continuum_l38_'
                  'sensitivity/result.json',
        'npz': 'phase3091/'
               'omega_p88_continuum_l38_'
               'sensitivity/'
               'omega_p88_continuum_l38_'
               'sensitivity.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'forward-free, 0.1s runtime, '
            'no model loads; unit order = '
            '3088 sorted-key convention '
            '(frozen argsort spearman '
            'tie-order sensitivity fixed as '
            'a convention)',
}
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3091
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 229
    l14['connects'].append({
        'meas_id': 'meas3091_omega_p88_'
                   'continuum_l38_'
                   'sensitivity',
        'phase': 3091,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change':
            'Omega-P88: continuum L38 '
            'sensitivity ROBUST - '
            'replacement arm rho_T +0.7413 '
            '(p=0.0077), n=15 arm +0.7821 '
            '(p=0.00085); criterion split '
            'fully closed.  Stouffer signed '
            '+1.10 vs unsigned 3.15 caveat '
            '(unsigned form '
            'direction-agnostic; G_DS count '
            'gate is the decision gate).  '
            'Next: 3092 A G_DS gate '
            'sensitivity; B 4B trunk '
            'anatomy; C qwen3-14b fifth '
            'point; D R1 reuse topology '
            'panorama'})
    assert len(l14['connects']) == 197
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already has 3091')

# ---------- MEMO append ----------
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3091:' not in memo:
    entry = (
        '\n## Phase 3091: Ω-P88 连续统 L38 '
        '敏感性检验（免前向）——判决 '
        'continuum_robust_l38，判据分歧完全闭合 '
        '[%s]\n\n'
        '**状态**: 已执行（免前向 0.1s；三臂预注册，'
        'execution.json 先于任何 rho 计算冻结）。\n\n'
        '### 1. 目标与原理\n'
        '3088 连续统（rho +0.9021, n=12）的 GLM4 '
        '单元取自 L37；3089 证明 L38 全管线同格。'
        '问题：把 GLM4 单元从 L37 值换成 L38 值后，'
        '连续统结论是否存活？\n'
        '$$\\rho_T = sp(s\\_lo, T\\_med),\\quad '
        's\\_lo^{L38}(m,p)=\\min(top3^{L38}_{fa},'
        'top3^{L38}_{fb})$$\n'
        '三臂：A 替换（n=12）、B 增补（n=15）、'
        'C Stouffer 符号敏感性（描述性）。'
        '冻结 spearman/perm_p 从 3088 源码逐字提取'
        '（N_PERM=20000, SEED=3091）。\n\n'
        '### 2. 锚（全过）\n'
        '- z86 重算锚：sorted-key 单元序下 '
        'rho_T=0.9020979021 与存储值 bit 级相等；\n'
        '- GLM4 L37 溯源锚：z86 存储 GLM4 单元 == '
        'z87 推导值 bit 级；\n'
        '- z89 sanity：20922 前向、L38、'
        'mixed_absent、SMOKE=False；\n'
        '- Stouffer 无符号重算 3.1501 与 z89 '
        '存储值 bit 级相等。\n\n'
        '### 3. 结果\n'
        '| 臂 | rho_T | p | 显著 |\n'
        '|---|-------|---|------|\n'
        '| A 替换 n=12 | +0.7413 | 0.00770 | 是 |\n'
        '| A s_mean 变体 | +0.6643 | 0.02325 | 是 |\n'
        '| B 增补 n=15 | +0.7821 | 0.00085 | 是 |\n'
        '（rho_U +0.3497 ns、rho_MIG +0.3706 ns '
        '两臂一致）\n'
        'L38 单元值：s_lo 0.8281/0.8281/0.9031；'
        'T_med +0.2465/+0.2379/+0.0528。\n'
        'C 臂：无符号 Stouffer 3.1501（bit 级复现）'
        'vs 符号感知 +1.1012（ns）。\n\n'
        '### 4. 分析\n'
        '- **判决 continuum_robust_l38**：两臂独立'
        '显著为正——连续统"谱位→迁移强度"规律对 '
        'GLM4 层位选择稳健，L37(n_neg)/L38(幅值) '
        '判据分歧在行为层完全闭合。\n'
        '- **衰减而非反转**：L38 使 GLM4 单元 '
        's_lo 上移（0.70→0.83-0.90，靠向 trunk 侧）'
        '而 T_med BC 骤降（0.108→0.053），rho '
        '0.902→0.741——方向符合"单元沿连续统移动"'
        '的预测，幅度衰减但排序结构存活。\n'
        '- **方法学发现（重要）**：frozen spearman '
        '用 argsort 定秩、无平均秩校正，s_lo 存在 '
        '4 组并列值时 rho 依赖单元排列顺序'
        '（0.8951 vs 0.9021，sum(d^2) 30 vs 28）'
        '——3088 的计算序是 sorted(units) 键序；'
        '本轮将其固化为约定并写入 execution.json。'
        '任何复用该 spearman 的后续 Phase 必须'
        '固定单元序。\n'
        '- **Stouffer 口径**：3089 预注册的无符号'
        '形式（z=3.150）把小 p 一律计为正向证据、'
        '与 sp 符号无关，属高估口径；符号感知 '
        'z=+1.10（ns）。判决树实际依赖的 G_DS '
        '计数门（符号感知，1/6）未受影响。\n\n'
        '### 5. 硬伤与边界\n'
        '- 单模型敏感性（仅 GLM4 换层）；其他模型'
        '无第二层位数据。\n'
        '- perm_p 的 obs 用 |rho| 双侧置换，'
        '单元序同样进入 obs 与置换分布——本判决在'
        '固定约定序下成立。\n'
        '- L38 单点谱值在 run_log 已被观察后才冻结'
        '设计（预注册诚实注记已写入 execution.json）。\n\n'
        '### 6. 结论与接续\n'
        '连续统结论 layer-robust；trunk+locked '
        '迁移仍为 qwen3-4b 独有；mixed_absent 获'
        '双层位支持。接续：3092（A G_DS 门敏感性'
        '正式化；B 4B 主干解剖；C qwen3-14b 第五'
        '谱点 ~21k 前向；D R1 复用拓扑全景——底册 '
        'reuse_inventory.json 已备）。\n\n'
        '资源消耗：免前向，CPU 0.1s；产物 sealed'
        '（npz8=%s result8=%s exec8=%s '
        'script8=%s）。\n'
        % (ts, seal['npz_sha256_8'],
           seal['result_sha256_8'],
           seal['exec8'],
           seal['script_sha256_8']))
    i3090 = memo.rindex('## Phase 3090:')
    assert i3090 > memo.rindex('## Phase 3089:')
    memo = memo.rstrip('\n') + '\n' + entry
    with io.open(MEMO, 'w',
                 encoding='utf-8') as f:
        f.write(memo)
    o.append('memo appended ts=%s' % ts)
else:
    o.append('memo already has 3091')

# ---------- audit 52 ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '五十二' not in aud:
    aud = aud.rstrip('\n') + (
        '\n\n---\n\n'
        '## 五十二、3091 增补：连续统 L38 敏感性'
        '——continuum_robust_l38\n'
        '1. **判决**：GLM4 单元 L37→L38 值替换'
        '（n=12）rho_T +0.7413（p=0.0077）、增补'
        '（n=15）+0.7821（p=0.00085），两臂显著为正'
        '——3088 连续统结论层位稳健，判据分歧完全'
        '闭合。\n'
        '2. **衰减方向符合预测**：L38 谱值上移'
        '（s_lo 0.83-0.90）而 T_med BC 骤降至 '
        '0.053，rho 0.902→0.741 为衰减而非反转；'
        'Stouffer 无符号 3.150 bit 级复现、符号感知'
        '仅 +1.10——无符号口径属高估，判决门以'
        '符号感知 G_DS 计数为准。\n'
        '3. **HDMCC 更新**：方法学新约束——frozen '
        'argsort spearman 并列值顺序敏感（4 组并列 '
        's_lo），单元序必须固定并写入 execution.json'
        '（本轮 0.8951 vs 0.9021 教训）；'
        'mixed_absent 获双层位支持，谱连续统为 '
        '4 架构 × 双层位稳健规律候选。\n')
    with io.open(AUDIT, 'w',
                 encoding='utf-8') as f:
        f.write(aud)
    o.append('audit 52 appended')
else:
    o.append('audit 52 already')

# ---------- wlog ----------
wl = os.path.join(WLOG_DIR, '2026-09-22.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if verdict not in prev:
    line = ('- Phase 3091 Omega-P88 continuum '
            'L38 sensitivity (forward-free): '
            'verdict %s (A repl rho +0.7413 '
            'p=0.0077; B n15 rho +0.7821 '
            'p=0.00085).  Criterion split '
            'fully CLOSED - continuum '
            'layer-robust.  Audit 52; ledger '
            '229/L14 197.\n' % verdict)
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
mem_cur = io.open(MEMO_W,
                  encoding='utf-8').read()
if 'max=3091' not in mem_cur:
    idx = mem_cur.find('## 下一步')
    assert idx > 0
    mem_cur = mem_cur[:idx] + (
        '## 下一步\n'
        '- max=3091，下一个 3092（A G_DS 门'
        '敏感性；B 4B 主干解剖；C qwen3-14b '
        '第五谱点 ~21k 前向；D R1 复用拓扑全景'
        '——reuse_inventory 底册已备）。\n')
    mem_cur = mem_cur.replace(
        '→ 3089 L38 复制 '
        'fourth_l38_mixed_absent**',
        '→ 3089 L38 复制 '
        'fourth_l38_mixed_absent → 3091 '
        '连续统 L38 敏感性 '
        'continuum_robust_l38（A +0.7413 / '
        'B +0.7821 均显著）**')
    mem_cur = mem_cur.replace(
        'repro 键族 L37→L38 是 REPS '
        '必改项）。',
        'repro 键族 L37→L38 是 REPS '
        '必改项）、单元序锚（3091：'
        'sorted-key 序下 z86 rho bit 级'
        '复现；frozen spearman 并列顺序'
        '敏感——单元序入 execution.json）。')
    assert len(mem_cur) < 3000, len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3091')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
