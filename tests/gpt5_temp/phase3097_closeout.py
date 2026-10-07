# -*- coding: utf-8 -*-
"""Idempotent single-arm closeout for
Phase 3097 (omega_p95_cliff_layer_
decomposition).  Ledger meas3097 + L14
+1 -> MEMO Phase 3097 -> audit
(五十八) -> wlog (case-matched tag) ->
MEMORY (2 anchors, budgeted)."""
import io
import json
import os
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
OD = (R13 + r'\phase3097'
      r'\omega_p95_cliff_layer_'
      r'decomposition')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
AUDIT = (ROOT + r'\research\gpt5\docs'
         r'\hdmcc_knowledge_map_review_'
         r'20260921.md')
WLOG_DIR = (ROOT + r'\.workbuddy'
            r'\memory')
o = []

res = json.load(io.open(
    OD + r'\result.json',
    encoding='utf-8'))
seal = json.load(io.open(
    OD + r'\seal.json',
    encoding='utf-8'))
z = np.load(OD + r'\omega_p95_cliff_'
            r'layer_decomposition.npz',
            allow_pickle=False)
v = res['verdict']
assert v == 'fifth_cliff_single_step', v
assert str(z['VERDICT']) == v
assert bool(z['SMOKE']) is False
S = res['stats']
A = res['anchors']
assert A['ok'] is True
assert A['c1_diff'] <= 1e-9
assert A['c2_diff'] <= 1e-9
assert A['c3_diff'] <= 1e-12
assert A['c4_ok'] is True
D = S['decision']
assert D['H_C1'] is True
assert D['H_C2'] is False
assert D['H_C3'] is False
o.append('3097 loaded: verdict=%s '
         'anchors_ok=True' % v)

KEYCi = ['%s_ci%d' % (key, ci)
         for key in ('AB', 'AC', 'BC')
         for ci in (1, 2, 3)]
rows = []
for ck in KEYCi:
    for side in ('14B', '4B'):
        rows.append(
            '| %s %s | %.3f | %.3f | L%d |'
            % (side, ck,
               S['E1']['max_last4'][
                   '%s_%s' % (side, ck)],
               S['E1']['max_prev'][
                   '%s_%s' % (side, ck)],
               S['E1']['cliff_at'][
                   '%s_%s' % (side, ck)]))
tbl = '\n'.join(rows)


def step(side, ck, L):
    return S['E1']['steps'][
        '%s_%s' % (side, ck)][L]


s_ac = step('14B', 'AC_ci1', 39)
s_bc = step('14B', 'BC_ci1', 39)
s_ab2 = step('14B', 'AB_ci2', 39)
s_ac2 = step('14B', 'AC_ci2', 39)

# ---------- Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
has3097 = any(
    isinstance(m, dict)
    and m.get('phase') == 3097
    for m in led['measurements'])
if not has3097:
    meas = {
        'meas_id':
            'meas3097_omega_p95_cliff_'
            'layer_decomposition',
        'phase': 3097,
        'claim':
            'Omega-P95 (forward-free) - '
            'cliff layer decomposition '
            'on the 3096-sealed F2P '
            'full-layer profiles.  '
            'Anchors: c1/c2 final '
            'columns vs 3079/3093 '
            'sealed F2_CTT '
            '1.2e-12/9.9e-13 <=1e-9; '
            'c3 grid medians re-render '
            '3096 curves bit-0; c4 '
            '3096 npz sha8 matches '
            'seal.json.  Verdict '
            'fifth_cliff_single_step: '
            'ALL 9 (pair,ci) groups '
            'place the max |step| at '
            'L39->L40; since both '
            'lens(L=39)=head(norm(h)) '
            'and native(L=40)=head(h_'
            'postnorm) share the same '
            'readout path, that step '
            'IS the last decoder block '
            '(block39) computation.  '
            'It rewrites TT directions '
            'conditionally: formal '
            'AC/BC destroyed in ONE '
            'step (-0.310/-0.212 after '
            '9 steps of only '
            '+/-0.02..0.08) while '
            'Shakespearean amplified '
            '(AB_ci2 +0.300, AC_ci2 '
            '+0.106).  4B control: '
            'same last-block rewrite '
            'of ci2 (+0.33/+0.42@L35) '
            'but NO destruction of '
            'formal (last4 max 0.086 '
            'vs prev 0.079) - the '
            'rewriter mechanism is '
            'shared, the POLICY '
            'differs by scale.  '
            'H_C1 True (0.310>=0.15 '
            'and >=2x prev 0.027); '
            'H_C2 False; H_C3 False '
            '(amplification also '
            'concentrated at L39).',
        'verdict': v,
        'inputs': ['phase3096 npz',
                   'phase3096 seal/result',
                   'phase3079 npz',
                   'phase3093 npz'],
        'outputs': [OD]}
    led['measurements'].append(meas)
    for l in led['linkage']:
        if (isinstance(l, dict)
                and l.get('link_id')
                == 'L14_readout_spectrum_'
                'cross_model'):
            cs = l['connects']
            cs.append(
                'meas3097_omega_p95_'
                'cliff_layer_'
                'decomposition')
            break
    old = led.pop('ledger_sha256_8', None)
    body = json.dumps(
        led, sort_keys=True,
        ensure_ascii=False)
    import hashlib
    led['ledger_sha256_8'] = (
        hashlib.sha256(
            body.encode('utf-8'))
        .hexdigest()[:8])
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f,
                  ensure_ascii=False,
                  indent=1)
    o.append('ledger meas3097 appended '
             '(n=%d)' % len(
                 led['measurements']))
else:
    o.append('ledger already')
n_meas = len([
    m for m in led['measurements']
    if isinstance(m, dict)])
l14 = [l for l in led['linkage']
       if isinstance(l, dict)
       and l.get('link_id')
       == 'L14_readout_spectrum_cross_'
       'model'][0]['connects']
o.append('ledger n=%d l14=%d'
         % (n_meas, len(l14)))

# ---------- MEMO ----------
memo = io.open(MEMO,
               encoding='utf-8').read()
if '## Phase 3097:' not in memo:
    entry = (
        '\n## Phase 3097: Ω-P95 断崖层位分'
        '解（免前向）——14B f2 坍缩 = 最后'
        '一个 block（L39→40）单步效应，条'
        '件选择性重写器'
        '（fifth_cliff_single_step） '
        '[%s]\n\n'
        '**状态**: 已执行（0 前向；输入 = '
        '3096 sealed F2P 全层剖面 + '
        '3079/3093 sealed npz；锚 c1/c2 '
        '终列 vs sealed F2_CTT '
        '1.2e-12 / 9.9e-13 ≤1e-9，c3 网格'
        '中位重渲 3096 曲线 bit 0.0，c4 '
        '3096 npz sha8 与 seal.json 一'
        '致）。execution.json 先冻结。\n\n'
        '### 1. 问题与设计\n'
        '3096 把断崖折叠在 d=0.9→1.0（'
        'L36..L40 四步）。本 Phase 用 '
        'sealed F2P（24 对 × 全层）做步降'
        '分解：step(L→L+1) = 相邻层 lens '
        'f2 之差。关键路径事实：lens(L=39)'
        '=head(norm(h)) 与 native(L=40)='
        'head(h_postnorm) 走**同一条 '
        'norm+head 读出路径**——L39→40 步'
        '的内容干净地等于**最后一个 decoder'
        ' block（block39）的计算**（无 '
        'final-norm 混淆，比 3096 的折叠'
        '叙述更精确）。判决门：H_C1 单步断'
        '崖（last4 max ≥0.15 且 ≥2× prev '
        'max）、H_C2 渐进（全 last10 步 '
        '<0.15）、H_C3 读出放大（ci2 组 '
        'last4 步 ≥3/4 为正）。\n\n'
        '### 2. E1 步降分解（last4 / prev '
        'max |step|，断崖层）\n'
        '| 组 | last4 max | prev max | '
        'argmax |\n|---|---|---|---|\n'
        '%s\n\n'
        '**9/9 组断崖全部定位在 L39→40**。'
        '14B 步向量（L30→39）例：AC_ci1 '
        '前 9 步仅 ±0.03 内波动，最后一步 '
        '%+.3f；BC_ci1 %+.3f。同时 '
        'AB_ci2 %+.3f、AC_ci2 %+.3f'
        '——同一步既摧毁 formal 又放大 '
        'Shakespearean。\n\n'
        '### 3. 判决逻辑\n'
        'H_C1=True（AC_ci1 0.310 ≥ 0.15 '
        '且 ≥ 2×0.027；BC_ci1 0.212 ≥ '
        '2×0.020）；H_C2=False；H_C3='
        'False（放大也集中在 L39 单步，非'
        '多步渐进）→ '
        '**fifth_cliff_single_step**。\n\n'
        '### 4. 分析（关键洞察）\n'
        '**14B 的 f2 角度坍缩由最后一个 '
        'decoder block 单步造成：block39 '
        '是条件选择性的 TT 方向重写器**——'
        '它对 formal 组的跨族对齐做摧毁性'
        '改写（-0.31/-0.21），对 '
        'Shakespearean 组做放大改写'
        '（+0.30/+0.11）。**4B 对照显示'
        '"末块重写"机制本身两模型共有**'
        '（4B ci2 在 L35→36 也有 '
        '+0.33/+0.42 大步），差异在重写策'
        '略：4B 不摧毁 formal（last4 max '
        '0.086 ≈ prev 0.079），14B 摧毁。'
        '3093 的 rescue 层带（L37/L38）正'
        '位于重写器上游——注入的结构随后被'
        'block39 读取，解释了为何救援有'
        '效。3094"角度律失因子"与 3095'
        '"prefix 条件主导"统一为：**规模'
        '改变的是最后一个 block 的重写策'
        '略，而非隐藏场的方向结构**。\n\n'
        '### 5. 硬伤与边界\n'
        '- 步降捆绑"block 计算 + 重 '
        'norm"，但 norm 两侧路径相同，'
        'L39→40 步无 final-norm 混淆——'
        '唯一残余归因风险是 RMSNorm 非线'
        '性对不同输入的缩放差（同函数复'
        '合，影响次要）；\n'
        '- block39 内部 attn vs MLP 未分'
        '解（需 block 级前向 hook）；\n'
        '- lens 探针 caveat 沿袭 3096；\n'
        '- n=8 对/ci 组中位数；H_C3 门在'
        '"放大集中于单步"的事实下判 False'
        '，属门定义与现象粒度错配（如实记'
        '录，不影响 H_C1 判决）。\n\n'
        '### 6. 结论与接续\n'
        'RDC 条件化齿轮获得迄今最具体的候'
        '选件：**末位 block 读出前置重写器'
        '（readout-preconditioning final '
        'block）**。接续 3098：(i) block39 '
        '内部分解（attn/MLP 子步 hook 前'
        '向）；(ii) block39 干预实验（跳过'
        '该 block / 替换其输出为输入，检验'
        '下游 TT 与行为因果性）；(C) R1 复'
        '用拓扑全景（底册已备）。\n\n'
        '资源消耗：0 前向 / 数秒级；产物 '
        'sealed（npz8=%s result8=%s）。\n'
        % (datetime.now().strftime(
               '%Y-%m-%d %H:%M'),
           tbl, s_ac, s_bc, s_ab2, s_ac2,
           seal['npz_sha256_8'],
           seal['result_sha256_8']))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(entry)
    o.append('memo 3097 appended')
else:
    o.append('memo already')

# ---------- audit ----------
aud = io.open(AUDIT,
              encoding='utf-8').read()
if '五十八' not in aud:
    add = (
        '\n## 五十八、3097 追加（Ω-P95）\n'
        '断崖层位分解（免前向，3096 sealed '
        'F2P）：9/9 组断崖定位在 L39→40 单'
        '步——该步读出路径两侧相同，内容即'
        '最后一个 decoder block 的计算。'
        'block39 条件选择性重写 TT 方向：'
        'formal AC/BC 单步 -0.310/-0.212'
        '（此前 9 步仅 ±0.02-0.08），'
        'Shakespearean 放大 +0.300/+0.106'
        '。4B 末块同样重写 ci2（+0.33/'
        '+0.42）但不摧毁 formal——机制共'
        '有、策略随规模改变。判决 '
        'fifth_cliff_single_step（H_C1：'
        '0.310≥0.15 且 ≥2×prev）。锚 '
        'c1/c2 ≤1e-9、c3 bit 0。\n')
    with io.open(AUDIT, 'a',
                 encoding='utf-8') as f:
        f.write(add)
    o.append('audit 58 appended')
else:
    o.append('audit already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
WLOG_TAG = ('cliff layer decomposition '
            'anatomy-closure')
try:
    prev = io.open(wlf,
                   encoding='utf-8').read()
except IOError:
    prev = ''
if WLOG_TAG not in prev:
    line = ('- Phase 3097 Omega-P95 '
            'cliff layer decomposition '
            'anatomy-closure: '
            'fifth_cliff_single_step '
            '(cliff IS block39, 9/9 '
            'groups @L39; conditional '
            'rewriter: formal destroyed '
            '-0.31/-0.21, Shakespearean '
            'amplified +0.30; 4B shares '
            'the rewriter, different '
            'policy).  Anchors c1-c4 '
            'pass; ledger %d/L14 %d.\n'
            % (n_meas, len(l14)))
    with io.open(wlf, 'a',
                 encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
mem_cur = io.open(MEMO_W,
                  encoding='utf-8').read()
if 'max=3097' not in mem_cur:
    anchor1 = ('- max=3096（Ω-P94 '
               'fifth_lens_late_assembly：'
               'f2 坍缩=读出端断崖）——3097：'
               '细化层位；C R1。')
    assert anchor1 in mem_cur, 'anchor1'
    mem_cur = mem_cur.replace(
        anchor1,
        '- max=3097（Ω-P95 '
        'cliff_single_step：block39 重'
        '写器）——3098：分解；C R1。')
    anchor3 = ('3096 lens_late**。')
    assert anchor3 in mem_cur, 'anchor3'
    mem_cur = mem_cur.replace(
        anchor3,
        '3096 lens_late → 3097 '
        'cliff**。')
    assert len(mem_cur) < 3000, \
        len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3097')

io.open(OD + r'\closeout_log.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
