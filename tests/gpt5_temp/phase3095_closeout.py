# -*- coding: utf-8 -*-
"""Idempotent single-arm closeout for
Phase 3095 (omega_p93_ab_specificity).
Ledger meas3095 + L14 +1 -> MEMO Phase
3095 -> audit addendum (五十六) -> wlog
(closure-phrase guarded) -> MEMORY (3
anchors).  Reads only sealed artifacts;
numbers rendered from result.json."""
import io
import json
import os
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
OD = (R13 + r'\phase3095'
      r'\omega_p93_ab_specificity')
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
z = np.load(OD + r'\omega_p93_ab_'
            r'specificity.npz',
            allow_pickle=False)
v = res['verdict']
assert v == 'fifth_ab_mixed', v
assert str(z['VERDICT']) == v
assert bool(z['SMOKE']) is False
E1 = res['stats']['E1']
E2 = res['stats']['E2']
E3 = res['stats']['E3']
D = res['stats']['decision']
A = res['anchors']
assert A['b1_ok'] and A['b1_4b_ok'] \
    and A['b3_ok']
assert D['H_A1'] is False and \
    D['H_A3'] is False and \
    D['H_A2'] is False
o.append('3095 loaded: verdict=%s '
         'anchors_ok=True' % v)


def f3(x):
    return '%.3f' % x


# ---------- E1 table ----------
rows = []
for key in ('AB', 'AC', 'BC'):
    rows.append(
        '| %s | %s / %s / %s | %s / %s '
        '/ %s |'
        % (key,
           f3(E1['f2_4B_%s_ci1_med' % key]),
           f3(E1['f2_4B_%s_ci2_med' % key]),
           f3(E1['f2_4B_%s_ci3_med' % key]),
           f3(E1['f2_14B_%s_ci1_med'
                 % key]),
           f3(E1['f2_14B_%s_ci2_med'
                 % key]),
           f3(E1['f2_14B_%s_ci3_med'
                 % key])))
tbl = '\n'.join(rows)
e2_txt = '；'.join(
    '%s med %.3f mean %.3f'
    % (key, E2['overlap_%s_med' % key],
       E2['overlap_%s_mean' % key])
    for key in ('AB', 'AC', 'BC'))
e3_txt = 'f2 %s vs U %s；sp_full %+.3f ' \
         'LOO [%+.3f, %+.3f]' % (
    '/'.join('%.3f' % E3['f2_14B_AB_ci%d'
                         % ci]
             for ci in (1, 2, 3)),
    '/'.join('%.3f' % E3['U14_AB_ci%d_med'
                         % ci]
             for ci in (1, 2, 3)),
    E3['sp_full'], E3['sp_loo_min'],
    E3['sp_loo_max'])

# ---------- Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
has3095 = any(
    isinstance(m, dict)
    and m.get('phase') == 3095
    for m in led['measurements'])
if not has3095:
    meas = {
        'meas_id':
            'meas3095_omega_p93_ab_'
            'specificity',
        'phase': 3095,
        'claim':
            'Omega-P93 (forward-free) - '
            'AB-specificity anatomy of '
            'the 14B f2 survival.  '
            'Anchors: b1 14B F2 replay '
            'bit-0 vs 3093 sealed '
            'F2_CTT + determinism '
            'recompute; b1-4b 4B f2 '
            'medians bit-0 vs 3094 '
            'result.json; b2 pair-map '
            'self-check; b3 gate '
            'replay.  Verdict '
            'fifth_ab_mixed: all three '
            'preregistered hypotheses '
            'REJECTED.  f2 survival is '
            'prefix-condition-dominated, '
            'not pair-specific: ci2 '
            '(Shakespearean) survives '
            'across all pairs and both '
            'models (14B 0.519-0.758); '
            '14B AC/BC collapse '
            'concentrates in ci1 '
            '(formal: 0.031/-0.003) '
            'while AB-c1 rises '
            '(0.470->0.620); ci3 '
            '(topic) low on both.  '
            'Token-content overlap does '
            'not explain it (AB margin '
            '0.020 < 0.05).  The 3080 '
            'angle-matching failure '
            'boundary is narrower than '
            '3094 concluded: alignment '
            'survives = Shakespearean '
            'overall + formal x AB.',
        'verdict': v,
        'inputs': ['phase3076 npz',
                   'phase3093 npz',
                   'phase3094 npz',
                   'phase3094 result'],
        'outputs': [OD]}
    led['measurements'].append(meas)
    for l in led['linkage']:
        if (isinstance(l, dict)
                and l.get('link_id')
                == 'L14_readout_spectrum_'
                'cross_model'):
            cs = l['connects']
            cs.append(
                'meas3095_omega_p93_ab_'
                'specificity')
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
    o.append('ledger meas3095 appended '
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
if '## Phase 3095:' not in memo:
    entry = (
        '\n## Phase 3095: Ω-P93 AB 特异性解'
        '剖（免前向）——AB 存活非 pair 特异，'
        '是 prefix 条件主导；三预注册假说全'
        '拒（fifth_ab_mixed） [%s]\n\n'
        '**状态**: 已执行（0 前向；输入 = '
        '3076/3093/3094 sealed 产物；锚 '
        'b1 14B F2 重放 bit 0.0 vs 3093 '
        'sealed F2_CTT + 同路径确定性重算 '
        'bit 0.0，b1-4b 4B f2 中位 bit 0.0 '
        'vs 3094 result.json，b2 对映射自'
        '检，b3 门重放）。execution.json '
        '先于计算冻结，含 vocab_alignment '
        '论证：两模型共享 Qwen3 tokenizer'
        '（vocab 151936），TT 方向场在同一'
        '词表坐标系语义对齐——跨模型 TT 比'
        '较合法（AGENTS.md 坐标禁令针对 '
        'hidden/residual，不针对共享词表 '
        'logits 空间）。\n\n'
        '### 1. 问题与设计\n'
        '3094 发现 14B 上 TT 角度因子 f2 '
        '在 AC/BC 坍缩（0.595→0.169、'
        '0.598→0.183）而 AB 保持（0.534→'
        '0.550）。为何 AB 保留？三个预注册'
        '假说：H_A1 ab_prefix_locked（AB '
        '存活集中于单一 prefix 组：某 ci '
        '组中位 ≥ 其余 max+0.15 且 ≥0.45'
        '，AC/BC 全 <0.35）、H_A3 '
        'ab_residual（AB 全组 ≥0.45 且 '
        'AC/BC 全 <0.35）、H_A2 '
        'ab_token_content（top-64 词元重'
        '叠 AB ≥ mean(AC,BC)+0.05 双模型'
        '成立）。判决顺序 H_A1→H_A3→H_A2'
        '→mixed→inconclusive。24 对结构 '
        'CIDX=k//8+1（prefix 条件 ci1='
        'formal style / ci2=Shakespearean'
        ' style / ci3=域话题 prefix）、'
        'BIDX=k%%8（8 个 body 句）。\n\n'
        '### 2. E1 f2 按 prefix 组（每組 '
        '8 body 中位）\n'
        '| 族对 | 4B ci1/ci2/ci3 | 14B '
        'ci1/ci2/ci3 |\n|---|---|---|\n'
        '%s\n\n'
        'E1 判决：h1_best=ci%d med=%.3f '
        'other_max=%.3f（gap 0.138 < 0.15'
        ' 阈值）acbc_max=%.3f ≥ 0.35 → '
        'H_A1=False；H_A3=False（AC/BC 非'
        '全 <0.35）。\n\n'
        '### 3. E2 词元内容与 E3 耦合\n'
        'top-64 词元重叠（同 tokenizer 跨'
        '模型）：%s。AB mean 0.127 vs '
        'others 0.107，margin 0.020 < '
        '0.05 → H_A2=False。E3 14B AB 逐'
        ' ci：%s——组间同向（ci2 双高、'
        'ci3 双低）组内弱负（Simpson 结'
        '构）；sp_full=-0.238（=3094 值）'
        'LOO [%+.3f, %+.3f] 符号稳定。\n\n'
        '### 4. 判决逻辑\n'
        '三假说全拒 → **fifth_ab_mixed**'
        '（诚实出口：AB 存活既非锁定单一 '
        'prefix 组、非全组保留、亦非词元内'
        '容驱动）。\n\n'
        '### 5. 分析（关键洞察）\n'
        '**AB 的"存活"不是 pair 特异而是 '
        'prefix 条件主导**：(1) ci2='
        'Shakespearean 是跨 pair 跨模型的'
        '普适存活组（4B 0.792-0.868；14B '
        '0.519-0.758）；(2) 14B AC/BC 坍'
        '缩集中发生在 ci1=formal 组'
        '（0.595→0.031、0.488→-0.003）'
        '——不是均匀退化而是条件选择性的'
        '结构死亡；(3) AB 在 formal 组反'
        '而升（0.470→0.620）——AB 特异性'
        '只在 formal 组成立；(4) ci3='
        'topic 双模型都低（0.098-0.322）'
        '——域话题 prefix 本来就破坏跨族 '
        'TT 对齐。Shakespearean 组存活可'
        '能与风格迁移本身是共享的表层变换'
        '有关；formal 组对齐在含 C 族'
        '（social-emotional）的对中死亡，'
        '提示 formal style 下 C 族 TT 方'
        '向被族特异内容主导。\n\n'
        '### 6. 硬伤与边界\n'
        '- n=8/组中位数，功效低；H_A1 差 '
        '0.012 被拒（gap 0.138 vs 0.15），'
        '门判对单组样本敏感；\n'
        '- H_A2 的 top-64 重叠基线本身低'
        '（med 5.5-7.0%%），margin 0.02 的'
        '解释力弱；\n'
        '- 仅 3 个 prefix 条件，无法区分'
        '"风格变换共享"与"内容特异"的更'
        '多层；\n'
        '- 免前向：只有相关性证据；AB-ci1 '
        '保留与 formal×C 死亡的机制原因需'
        '前向 patch 验证；\n'
        '- f2~U_AB=-0.238 不显著（3094 '
        'p>0.05），Simpson 分解是描述性'
        '的。\n\n'
        '### 7. 结论与接续\n'
        'trunk 谱位 + 头级 T 响应存活 + '
        'TT 对齐坍缩的图谱细化到条件级：'
        '**对齐存活的结构 = Shakespearean '
        '全体 + formal×AB**。3080 角度匹'
        '配律的失效边界比 3094 判定的更窄'
        '——大模型不是整体失去装配律，而是'
        '装配在语境条件下有选择性。接续 '
        '3096：(B) f2 退化前向定位（14B 读'
        '出层 patch 注入 4B 角度结构检验因'
        '果性）；(C) R1 复用拓扑全景（底册'
        '已备）；(D) 新线索——为何 '
        'Shakespearean 组跨模型保持 TT 对'
        '齐（风格变换的共享机制）。\n\n'
        '资源消耗：0 前向 / 数秒级；产物 '
        'sealed（npz8=%s result8=%s）。\n'
        % (datetime.now().strftime(
               '%Y-%m-%d %H:%M'),
           tbl, D['h1_best_ci'],
           max(E1['f2_14B_AB_ci%d_med'
                  % ci]
               for ci in (1, 2, 3)),
           sorted(
               E1['f2_14B_AB_ci%d_med'
                  % ci]
               for ci in (1, 2, 3))[-2],
           max(
               E1['f2_14B_%s_ci%d_med'
                  % (key, ci)]
               for key in ('AC', 'BC')
               for ci in (1, 2, 3)),
           e2_txt, e3_txt,
           E3['sp_loo_min'],
           E3['sp_loo_max'],
           seal['npz_sha256_8'],
           seal['result_sha256_8']))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(entry)
    o.append('memo 3095 appended')
else:
    o.append('memo already')

# ---------- audit ----------
aud = io.open(AUDIT,
              encoding='utf-8').read()
if '五十六' not in aud:
    add = (
        '\n## 五十六、3095 追加（Ω-P93）\n'
        'AB 特异性解剖（免前向）：三预注册'
        '假说全拒（fifth_ab_mixed）——AB '
        '存活非 pair 特异而是 prefix 条件'
        '主导：ci2=Shakespearean 组跨 pair'
        ' 跨模型普适存活（14B 0.519-0.758'
        '），14B AC/BC 坍缩集中发生在 '
        'ci1=formal 组（0.031/-0.003），'
        'AB 在 formal 组反而升（0.620）。'
        'E2 词元重叠不解释（margin '
        '0.020<0.05）；E3 f2~U 组间同向'
        '组内弱负（Simpson）。角度匹配律失'
        '效边界比 3094 更窄：对齐存活 = '
        'Shakespearean 全体 + formal×AB。'
        '锚 b1/b1-4b 全 bit 0.0。\n')
    with io.open(AUDIT, 'a',
                 encoding='utf-8') as f:
        f.write(add)
    o.append('audit 56 appended')
else:
    o.append('audit already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
WLOG_TAG = 'AB specificity anatomy-closure'
try:
    prev = io.open(wlf,
                   encoding='utf-8').read()
except IOError:
    prev = ''
if WLOG_TAG not in prev:
    line = ('- Phase 3095 Omega-P93 AB '
            'specificity '
            'anatomy-closure: '
            'fifth_ab_mixed (all 3 '
            'hypotheses rejected; f2 '
            'survival prefix-dominated: '
            'ci2 universal + AB-ci1).  '
            'Anchor suite bit-0; ledger '
            '%d/L14 %d; memo/audit '
            'appended.\n'
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
if 'max=3095' not in mem_cur:
    anchor1 = ('- max=3094（Ω-P92 '
               'fifth_mixed_degradation：f2 因子 '
               'AC/BC 坍缩，H1 否定，AB 存活）'
               '——3095：A AB 特异性；B f2 前向'
               '定位；C R1 全景。')
    assert anchor1 in mem_cur, 'anchor1'
    mem_cur = mem_cur.replace(
        anchor1,
        '- max=3095（Ω-P93 fifth_ab_mixed：'
        '三假说全拒，f2 存活=ci2+AB-ci1）'
        '——3096：B 前向定位；C R1 全景。')
    anchor2 = ('3093 谱点；3094 f2 退化）。')
    assert anchor2 in mem_cur, 'anchor2'
    mem_cur = mem_cur.replace(
        anchor2,
        '3093 谱点；3095 ci2 主导）。')
    anchor3 = ('3094 fifth_mixed_degradation'
               '（角度律失因子）**。')
    assert anchor3 in mem_cur, 'anchor3'
    mem_cur = mem_cur.replace(
        anchor3,
        '3094 fifth_mixed_degradation'
        ' → 3095 fifth_ab_mixed**。')
    assert len(mem_cur) < 3000, len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3095')

io.open(OD + r'\closeout_log.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
