# -*- coding: utf-8 -*-
"""Phase 3089 closeout (idempotent, verdict-
branched): Ledger -> L14 -> MEMO append -> HDMCC
audit addendum -> wlog -> MEMORY.md."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3089'
     r'\omega_p87_glm4_l38_full_arbitration')
R_L37 = (ROOT + r'\tests\glm5\result'
         r'\rdc_query_construction_20260913'
         r'\phase3087'
         r'\omega_p85_glm4_l37_full_arbitration')
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
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['verdict']
assert verdict.startswith('fourth_l38_'), verdict
assert seal['setup_ok'] is True
z = np.load(R + r'\omega_p87_glm4_l38_full_'
            r'arbitration.npz',
            allow_pickle=False)
assert bool(z['SMOKE']) is False
assert int(z['FORWARDS']) == res['forwards']
assert int(z['L_INJ']) == 38
assert int(z['L_POST']) == 39
assert bool(z['SETUP_OK'])
assert bool(z['REPRO_OK'])
fw = res['forwards']
el = res['elapsed']

top3 = {fk: float(z['E3_TOP3_CS_' + fk])
        for fk in 'ABC'}
top3h = {fk: float(z['E3_TOP3_CS1H_' + fk])
         for fk in 'ABC'}
s_lo = {'AB': min(top3['A'], top3['B']),
        'AC': min(top3['A'], top3['C']),
        'BC': min(top3['B'], top3['C'])}
tmed = {k: float(np.median(z['T_' + k]))
        for k in ('AB', 'AC', 'BC')}
umed = {k: float(np.median(z['U_' + k]))
        for k in ('AB', 'AC', 'BC')}
mig = {k: float(z['MIG_' + k])
       for k in ('AB', 'AC', 'BC')}
nneg = {fk: int(z['N_NEG_' + fk])
        for fk in 'ABC'}
medc = {fk: float(z['MED_C_' + fk])
        for fk in 'ABC'}
rall = {fk: float(z['R_ALL_' + fk])
        for fk in 'ABC'}
st_z = float(z['STOUFFER_Z'])
gds_cnt = int(z['GDS_COUNT'])
gds_min = float(z['GDS_MIN_SP'])
spec_class = str(z['SPEC_CLASS'])

# 3087 L37 reference values (for the
# comparison table)
z37 = np.load(R_L37 + r'\omega_p85_glm4_l37_'
              r'full_arbitration.npz',
              allow_pickle=False)
top3_37 = {fk: float(z37['E3_TOP3_CS_' + fk])
           for fk in 'ABC'}
medc_37 = {fk: float(z37['MED_C_' + fk])
           for fk in 'ABC'}
rall_37 = {fk: float(z37['R_ALL_' + fk])
           for fk in 'ABC'}
nneg_37 = {fk: int(z37['N_NEG_' + fk])
           for fk in 'ABC'}
tmed_37 = {k: float(np.median(z37['T_' + k]))
           for k in ('AB', 'AC', 'BC')}

same_cell = (verdict == 'fourth_l38_mixed_'
             'absent')
degenerate = (verdict == 'fourth_l38_top8_'
              'degenerate')
if same_cell:
    head_txt = ('L38 复制同格 fourth_l38_'
                'mixed_absent——判据分歧闭合，'
                'GLM4 结论与 3088 连续统层位'
                '稳健')
    core3 = (
        '**① L38 复制同格、判据分歧闭合（一）**：'
        'E1 幅值判据选出的 L38（med_c '
        '%(mc)s，比 L37 强 4-6 倍）跑完整 '
        '3083/3085/3087 管线，四格判决与 '
        'L37（n_neg 判据）**完全同格** '
        'fourth_l38_mixed_absent——装载'
        '幅值深度翻 4-6 倍不改变定性判决，'
        'mixed_absent 在更强干预下依然'
        '成立（absent 结论被强化而非'
        '削弱）。'
        '\n\n**② 3088 连续统结论层位稳健'
        '（二）**：L38 谱值 CS top3 '
        '%(spec)s（L37：0.699/0.786/'
        '0.744），%(spec_note)s'
        '\n\n**③ repro 锚 L38 键 bit 级'
        '通过（三）**：A2 vs 3087 A1 L38——'
        'n_neg/top8 三族精确相等，'
        'med_c/CS1H/R_ALL diff ≤%(rd).1e'
        '（容差 1e-9）；%(fw)d 前向 '
        '%(el).1f 秒，b 锚全 bit-0。'
        % {'mc': ' / '.join(
               '%.3f' % medc[fk]
               for fk in 'ABC'),
           'spec': ' / '.join(
               '%.4f' % top3[fk]
               for fk in 'ABC'),
           'spec_note': (
               '单元值移动幅度见对照表——'
               '3088 判决 cross_architecture'
               '_confirmed 的 GLM4 输入需按 '
               'L38 值做敏感性确认（3090 '
               'B，免前向）'
               if max(abs(top3[fk]
                          - top3_37[fk])
                      for fk in 'ABC') > 1e-6
               else '与 L37 bit 级一致，'
                    '3088 连续统 GLM4 单元值'
                    '无需更新'),
           'rd': max(max(float(z['REPRO_MEDC_'
                                  'DIFF_' + fk]),
                          float(z['REPRO_CS1H_'
                                 'DIFF_' + fk]),
                          float(z['REPRO_RALL_'
                                 'DIFF_' + fk]))
                      for fk in 'ABC'),
           'fw': fw, 'el': el})
elif degenerate:
    head_txt = ('L38 复制 fourth_l38_top8_'
                'degenerate——A1 描述性未'
                '传递到全管线，L38 不可作'
                '仲裁层')
    core3 = (
        '**① L38 top8 退化（一）**：A1 描述'
        '性 n_neg 15/12/10 未在全管线'
        '（K3=24 完整 24 对）下传递，'
        '预注册退化出口触发——L38 不可'
        '作 GLM4 仲裁层，L37 判据选择'
        '被间接支持。\n\n**② 判据分歧部分'
        '闭合（二）**：幅值判据候选层在'
        '完整管线下不可行，L37 成为唯一'
        '可行仲裁层；3088 连续统结论无需'
        '敏感性重跑（其输入层未被'
        '推翻）。\n\n**③ repro/工程锚（三）**：'
        'b 锚全 bit-0；repro 锚按预注册'
        '在退化出口前执行情况见 run_log；'
        '%(fw)d 前向 %(el).1f 秒。'
        % {'fw': fw, 'el': el})
else:
    head_txt = ('L38 复制判决 %s——层位敏感'
                '实证，3089 B 免前向重跑'
                '连续统' % verdict)
    core3 = (
        '**① 层位敏感实证（一）**：L38 '
        '（幅值强 4-6 倍）给出与 L37 不同'
        '的四格判决 %(v)s——GLM4 因果装载'
        '的四格定性依赖层位选择判据，'
        '判据分歧升级为实质分歧。'
        '\n\n**② 3088 连续统需敏感性重跑'
        '（二）**：L38 谱值 CS top3 '
        '%(spec)s ≠ L37（0.699/0.786/'
        '0.744），GLM4 三单元 s_lo/T_med/'
        'U_med 全部移动——下一 Phase 免'
        '前向重跑 3088 框架（GLM4-L38 '
        '单元），检验 cross_architecture'
        '_confirmed 是否维持。'
        '\n\n**③ repro 锚 bit 级（三）**：'
        'vs 3087 A1 L38 锚通过；%(fw)d '
        '前向 %(el).1f 秒。'
        % {'v': verdict,
           'spec': ' / '.join(
               '%.4f' % top3[fk]
               for fk in 'ABC'),
           'fw': fw, 'el': el})

# L37 vs L38 comparison table rows
cmp_rows = []
for fk in 'ABC':
    cmp_rows.append(
        '| %s | %d/%d | %.3f/%.3f | '
        '%+.3f/%+.3f | %.4f/%.4f |'
        % (fk, nneg_37[fk], nneg[fk],
           medc_37[fk], medc[fk],
           rall_37[fk], rall[fk],
           top3_37[fk], top3[fk]))
cmp_table = '\n'.join(cmp_rows)

unit_rows = []
for k in ('AB', 'AC', 'BC'):
    unit_rows.append(
        '| %s | %.4f/%.4f | %+.4f/%+.4f '
        '| %+.4f/%+.4f |'
        % (k, s_lo[k], min(top3_37['A'],
                           top3_37['B'])
           if k == 'AB' else
           (min(top3_37['A'],
                top3_37['C'])
            if k == 'AC' else
            min(top3_37['B'],
                top3_37['C'])),
           tmed[k], tmed_37[k],
           umed[k],
           float(np.median(
               z37['U_' + k]))))
unit_table = '\n'.join(unit_rows)

f2_txt = ' / '.join(
    '%s %+.4f (p=%.3f)'
    % (k, float(z['E3_F2_CTT_T_' + k]),
       float(z['E3P_F2_CTT_T_' + k]))
    for k in ('AB', 'AC', 'BC'))
meas = {
    'meas_id': 'meas3089_omega_p87_'
               'glm4_l38_replica',
    'phase': 3089,
    'claim': (
        'Omega-P87 (plan 3089 A) - GLM4 '
        'L38 arbitration replica, the '
        'criterion-split discriminant from '
        'the 3087 A1 scan (L38 by the E1 '
        'magnitude criterion, med_c '
        '4-6x L37; L37 was the n_neg '
        'choice).  Full 3083/3085/3087 '
        'pipeline at L_INJ=38/L_POST=39, '
        'seed 3089, %(fw)d forwards / '
        '%(el).1fs, bf16 sysmem fallback.  '
        'REPRO anchor vs the SAME 3087 A1 '
        'npz L38 keys: n_neg/top8 exact '
        'per family, med_c/CS1H/R_ALL '
        'diffs <= 1e-9.  L37-vs-L38: '
        'n_neg %(n37)s -> %(n38)s, med_c '
        '%(mc37)s -> %(mc38)s, R_ALL '
        '%(ra37)s -> %(ra38)s, CS top3 '
        '%(s37)s -> %(s38)s.  Spectrum '
        'class %(sc)s; T med %(tm)s; '
        'G_DS count=%(gc)d/6 (min_sp='
        '%(gm).4f); Stouffer z=%(z).3f; '
        'f2~T: %(f2)s.  VERDICT %(v)s - '
        '%(read)s  CAVEATS: replica phase '
        '(verdict space fixed in 3087, '
        'A1 L38 descriptives known); '
        'single-layer replica (no '
        'intermediate L36/L39); n=24 '
        'pairs partial orientation.'
        % {'fw': fw, 'el': el,
           'n37': '/'.join(str(nneg_37[fk])
                           for fk in 'ABC'),
           'n38': '/'.join(str(nneg[fk])
                           for fk in 'ABC'),
           'mc37': '/'.join('%.3f'
                            % medc_37[fk]
                            for fk in 'ABC'),
           'mc38': '/'.join('%.3f' % medc[fk]
                            for fk in 'ABC'),
           'ra37': '/'.join('%+.3f'
                            % rall_37[fk]
                            for fk in 'ABC'),
           'ra38': '/'.join('%+.3f' % rall[fk]
                            for fk in 'ABC'),
           's37': '/'.join('%.4f'
                           % top3_37[fk]
                           for fk in 'ABC'),
           's38': '/'.join('%.4f' % top3[fk]
                           for fk in 'ABC'),
           'sc': spec_class,
           'tm': '/'.join('%+.3f' % tmed[k]
                          for k in
                          ('AB', 'AC', 'BC')),
           'gc': gds_cnt, 'gm': gds_min,
           'z': st_z, 'f2': f2_txt,
           'v': verdict,
           'read': (
               'criterion split CLOSED - '
               'the GLM4 four-way verdict '
               'is layer-robust; the 3088 '
               'n=12 continuum verdict '
               'stands (spectrum units '
               'shift check recorded).'
               if same_cell else
               'criterion split escalated '
               'to a substantive divergence '
               '- the continuum needs the '
               'L38-unit sensitivity re-run.'
               if not degenerate else
               'the magnitude-criterion '
               'layer is NOT viable under '
               'the full pipeline - L37 '
               'remains the only viable '
               'arbitration layer.')}),
    'verdict': verdict,
    'anchors': 'b0/b1/b3/b4/b6/b7a/b8 '
               'bit-0 per family + repro '
               'anchor vs 3087 A1 L38 keys '
               '(n_neg/top8 exact, '
               'med_c/CS1H/R_ALL <= 1e-9) '
               '- see npz REPRO_* keys',
    'artifacts': {
        'result': 'phase3089/'
                  'omega_p87_glm4_l38_full_'
                  'arbitration/result.json',
        'npz': 'phase3089/'
               'omega_p87_glm4_l38_full_'
               'arbitration/'
               'omega_p87_glm4_l38_full_'
               'arbitration.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'L38 replica of 3087 A2 '
            '(criterion-split discriminant); '
            'generator: p3089_patch.py '
            'GSUBS+REPS on the 3087 source '
            'with bad/want lists; verdict '
            'names suffixed _l38',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3089
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 228
    l14['connects'].append({
        'meas_id': 'meas3089_omega_p87_'
                   'glm4_l38_replica',
        'phase': 3089,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P87: GLM4 '
                        'L38 replica - '
                        + ('criterion '
                           'split CLOSED '
                           '(same cell '
                           'fourth_l38_'
                           'mixed_'
                           'absent); '
                           'GLM4 verdict '
                           'and 3088 '
                           'continuum '
                           'layer-'
                           'robust.'
                           if same_cell else
                           'criterion '
                           'split open '
                           '(%s) - '
                           'continuum '
                           'sensitivity '
                           're-run '
                           'queued.'
                           % verdict)
                        + '  Next: 3090 '
                          'B continuum '
                          'sensitivity '
                          '(forward-free); '
                          'C G_DS gate '
                          'sensitivity; D '
                          '4B trunk '
                          'anatomy; E '
                          'qwen3-14b '
                          'fifth point'})
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
    o.append('ledger already upserted n=%d l14=%d'
             % (len(led['measurements']),
                len(l14['connects'])))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3089:' not in memo:
    sec = u'''## Phase 3089: Ω-P87 GLM4 L38 仲裁复制（判据分歧判别）——%(head)s [%(created)s]

**判决：`%(verdict)s`**。L_INJ=38/L_POST=39（E1 幅值判据层，med_c 比 L37 强 4-6 倍），完整 3083/3085/3087 管线复制，seed 3089，**%(fw)d 前向 / %(el).1f 秒**，repro 锚改用同一 3087 A1 npz 的 **L38 键**。生成器：对 3087 源做 GSUBS（fourth_→fourth_l38_、omega_p85→omega_p87）+ 14 条精确 REPS + docstring/PREREG 整块重写，bad 19 项清零、want 21 项核对、compile 通过。

### 核心结果（重复三遍）
%(core3)s

### L37 vs L38 对照（n_neg | med_c | R_ALL | CS top3）
| 族 | n_neg L37/L38 | med_c L37/L38 | R_ALL L37/L38 | CS top3 L37/L38 |
| --- | --- | --- | --- | --- |
%(cmp)s

### GLM4 连续统单元值（L37/L38）
| 对 | s_lo | T_med | U_med |
| --- | --- | --- | --- |
%(unit)s

### 理论更新（第一性原理）
%(theory)s

### 硬伤与边界
- 单层复制：只测了 L38 一个候选层，L36/L39 未探测——"判据分歧闭合"限于 {L37, L38} 两点；rescue 带边界仍未映射。
- 复制相位非盲：判决空间 3087 冻结、A1 L38 描述性先行已知——本 Phase 检验的是**稳健性**，不是盲发现。
- L38 最弱族 n_neg=10（rescue 带内最低），top8 子集与 L37（24）差异大——迁移门读数对子集构成的敏感性未分离。
- bf16 sysmem fallback（18.84GB>17.09GB VRAM）沿 3087 先例，repro 键 bit 级通过证明无数值扰动。
- n=24 对部分方向性协议、同一 3076 文本集。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3089/omega_p87_glm4_l38_full_arbitration/`：script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；%(fw)d forwards / %(el).1fs。

**接续 3090 菜单**——%(menu)s''' % {
        'head': head_txt,
        'created': created,
        'verdict': verdict,
        'fw': fw, 'el': el,
        'core3': core3,
        'cmp': cmp_table,
        'unit': unit_table,
        'theory': (
            '- **判据分歧闭合=层位稳健性成立**'
            '（same-cell 时）：两个独立选择判据'
            '（负头覆盖广度 vs 装载幅值深度）'
            '指向同一四格判决——GLM4 mixed_'
            'absent 不是单层伪影；更深的装载'
            '（4-6 倍幅值）不产生迁移，"谱位'
            '定容量"图景获得层位维度支持。\n'
            '- **连续统的层位稳定性**：3088 '
            'cross_architecture_confirmed 的 '
            'GLM4 输入来自 L37；L38 谱值/单元'
            '值移动幅度决定 3088 是否需要数值'
            '级更新（定性判决同格已保障方向'
            '稳健）。\n'
            '- **（migrate 分支）**层位敏感'
            '实证：装载深度改变四格定性，'
            '"absent" 是 L37 特有还是真稳健'
            '需连续统重跑判别。\n'
            '- **（degenerate 分支）**描述性'
            '筛选与全管线可行性解耦：A1 的 '
            'n_neg 是 K3=24 完整协议的描述量，'
            '但 top8 子集质量在完整管线下另'
            '有标准——层位选择判据应升级为'
            '"描述性 + 可行性"双门。'),
        'script8': seal['script_sha256_8'],
        'result8': seal['result_sha256_8'],
        'npz8': seal['npz_sha256_8'],
        'exec8': seal['exec_sha256_8'],
        'n': len(led['measurements']),
        'l14': len(l14['connects']),
        'menu': (
            'A（主选·免前向）**连续统 L38 敏感性'
            '确认**（GLM4-L38 三单元纳入 3088 '
            '框架重跑，检验 cross_architecture'
            '_confirmed 在层位扰动下是否维持）。'
            'B G_DS 门敏感性（免前向）。C 4B '
            '主干解剖（免前向）。D 层位×谱型。'
            'E qwen3-14b 第五谱点。'
            if same_cell else
            'A（主选·免前向）**连续统 L38 重跑**'
            '（GLM4-L38 三单元替换进 3088 框架，'
            '判决 cross_architecture_confirmed '
            '是否维持——层位敏感后的第一优先）。'
            'B G_DS 门敏感性。C 4B 主干解剖。'
            'D 层位×谱型。E qwen3-14b 第五谱点。'
            if not degenerate else
            'A（主选·免前向）**3088 结论确认与'
            '登记**（L38 退化出口使 L37 成为唯一'
            '可行仲裁层，3088 连续统输入无需'
            '更新——补登记层位选择双门判据）。'
            'B G_DS 门敏感性。C 4B 主干解剖。'
            'D 层位×谱型。E qwen3-14b 第五谱点。')}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 五十一、3089' not in aud:
    add = u'''
---
## 五十一、3089 增补：GLM4 L38 仲裁复制——判据分歧判别 %(hd)s
1. **判决**：L_INJ=38（幅值判据层）全管线复制，%(fw)d 前向，repro 锚 vs 3087 A1 L38 键通过——VERDICT %(v)s。
2. **L37-vs-L38**：n_neg %(n37)s→%(n38)s、med_c %(mc37)s→%(mc38)s（4-6 倍）、CS top3 %(s37)s→%(s38)s；%(cmpread)s
3. **HDMCC 更新**：%(hdmcc)s
''' % {'hd': ('同格、闭合'
              if same_cell else
              ('退化出口' if degenerate
               else '层位敏感')),
       'fw': fw, 'v': verdict,
       'n37': '/'.join(str(nneg_37[fk])
                       for fk in 'ABC'),
       'n38': '/'.join(str(nneg[fk])
                       for fk in 'ABC'),
       'mc37': '/'.join('%.3f' % medc_37[fk]
                        for fk in 'ABC'),
       'mc38': '/'.join('%.3f' % medc[fk]
                        for fk in 'ABC'),
       's37': '/'.join('%.4f' % top3_37[fk]
                       for fk in 'ABC'),
       's38': '/'.join('%.4f' % top3[fk]
                       for fk in 'ABC'),
       'cmpread': (
           '判据分歧闭合——GLM4 四格判决层位'
           '稳健，3088 连续统方向结论不受层位'
           '判据影响。'
           if same_cell else
           '层位敏感实证——连续统需 L38 单元'
           '重跑（3090 A）。'
           if not degenerate else
           '幅值判据层在完整管线下不可行——'
           'L37 为唯一可行仲裁层，3088 输入'
           '无需更新。'),
       'hdmcc': (
           'mixed_absent 获第二层位支持；'
           '层位选择升级为"描述性+可行性"双门'
           '候选判据。'
           if same_cell else
           '四格定性依赖层位——single-layer '
           '仲裁结论需带层位标注。'
           if not degenerate else
           'top8 退化出口首次在复制相位触发——'
           '负结果一等公民登记。')}
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-22.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3089' not in prev:
    line = ('- Phase 3089 Omega-P87 GLM4 L38 '
            'arbitration replica (criterion-'
            'split discriminant): verdict %s '
            '(%d forwards).  %s  Audit 51; '
            'ledger 228/L14 196.\n'
            % (verdict, fw,
               'Criterion split CLOSED - '
               'layer-robust.'
               if same_cell else
               ('Degenerate exit - L37 remains '
                'the only viable layer.'
                if degenerate else
                'Layer sensitivity - continuum '
                're-run queued.')))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W,
                      encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3089' not in mem_cur:
    mem_cur = mem_cur.replace(
        'max=3088，下一个 3089', 'max=3089')
    # replace the 下一步 section wholesale
    idx = mem_cur.find('## 下一步')
    if idx > 0:
        mem_cur = mem_cur[:idx] + (
            '## 下一步\n- max=3089，下一个 '
            '3090（视 L38 判决：同格→A 免前向'
            '连续统 L38 敏感性确认；异格→A 免'
            '前向连续统 L38 重跑为第一优先；'
            '另有 B G_DS 门敏感性、C 4B 主干'
            '解剖、D 层位×谱型、E qwen3-14b '
            '第五谱点）。\n')
    mem_cur = mem_cur.replace(
        '**3088 cross_architecture_confirmed'
        '（连续统 n=12 跨架构存活，rho '
        '+0.9021；谱位定容量、trunk 定锁定'
        '双格局）**',
        '**3088 cross_architecture_confirmed'
        '（连续统 n=12 跨架构存活，rho '
        '+0.9021；谱位定容量、trunk 定锁定'
        '双格局）→ 3089 L38 复制 %s**'
        % verdict)
    mem_cur = mem_cur.replace(
        '连续统检验锚（3086）、第四谱点双臂锚'
        '（3087）、**跨架构连续统锚（3088：'
        'qwen-only n=9 子集 bit 级复现 3086 '
        '+0.8667；统计块逐字节同一性断言）**',
        '连续统检验锚（3086）、第四谱点双臂锚'
        '（3087）、**跨架构连续统锚（3088：'
        'qwen-only n=9 子集 bit 级复现 3086 '
        '+0.8667；统计块逐字节同一性断言）**、'
        'L38 复制锚（3089：repro 锚改用 A1 '
        'L38 键；repro 键族 L37→L38 是 REPS '
        '必改项）')
    assert len(mem_cur) < 3000, len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3089')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
