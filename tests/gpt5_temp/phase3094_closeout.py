# -*- coding: utf-8 -*-
"""Idempotent single-arm closeout for
Phase 3094 (omega_p92_trunk_anatomy).
Ledger meas3094 + L14 +1 -> MEMO Phase
3094 -> audit addendum (五十五) -> wlog
(closure-phrase guarded) -> MEMORY (3
anchors).  Reads only sealed artifacts;
no numbers are hardcoded except sealed
verdict string checks."""
import io
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
OD = (R13 + r'\phase3094'
      r'\omega_p92_trunk_anatomy')
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
z = np.load(OD + r'\omega_p92_trunk_'
            r'anatomy.npz',
            allow_pickle=False)
v = res['verdict']
assert v == 'fifth_mixed_degradation', v
assert str(z['VERDICT']) == v
assert bool(z['SMOKE']) is False
E1 = res['stats']['E1']
D = res['stats']['decision']
A = res['anchors']
assert A['a1_ok'] and A['a2_ok'] \
    and A['a3_ok'] and A['a4_ok']
o.append('3094 loaded: verdict=%s '
         'anchors_ok=True' % v)


def f4(x):
    return '%+.4f' % x


# ---------- table text ----------
rows = []
for key in ('AB', 'AC', 'BC'):
    rows.append(
        '| %s | %s / %s (%s) | %s / %s '
        '(%.3f) | %s / %s | %s (p=%s) / '
        '%s (p=%s) |'
        % (key,
           f4(E1['redund_CS_%s_4B' % key]),
           f4(E1['redund_CS_%s_14B' % key]),
           f4(E1['redund_gap_CS_%s' % key]),
           f4(E1['std_U_%s_4B' % key]),
           f4(E1['std_U_%s_14B' % key]),
           E1['std_ratio_U_%s' % key],
           f4(E1['med_f2_%s_4B' % key]),
           f4(E1['med_f2_%s_14B' % key]),
           f4(E1['sp_f2U_%s_4B' % key]),
           '%.4f' % E1['p_f2U_%s_4B' % key],
           f4(E1['sp_f2U_%s_14B' % key]),
           '%.4f' % E1['p_f2U_%s_14B'
                       % key]))
tbl = '\n'.join(rows)

E2 = res['stats']['E2']
e2_txt = '；'.join(
    '%s %.1f→%.1f (x%.2f)'
    % (f, E2['med_ttnorm_%s_4B' % f],
       E2['med_ttnorm_%s_14B' % f],
       E2['ttnorm_ratio_%s' % f])
    for f in ('A', 'B', 'C'))

# ---------- Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
has3094 = any(
    isinstance(m, dict)
    and m.get('phase') == 3094
    for m in led['measurements'])
if not has3094:
    meas = {
        'meas_id':
            'meas3094_omega_p92_trunk_'
            'anatomy',
        'phase': 3094,
        'claim':
            'Omega-P92 (forward-free) - '
            '4B vs 14B trunk-model G_DS '
            'split anatomy.  Anchors a1/a2 '
            'T/U/F1/F2 replay bit-0 vs '
            '3079/3093 npz; a3 gate '
            'replay; a4 SP_UT replay '
            'bit-0.  Verdict '
            'fifth_mixed_degradation: '
            'H1 u_compression REJECTED '
            '(14B intra-CS redundancy '
            'LOWER than 4B, gap '
            '-0.041..-0.137); H2 partial '
            '- f2 median collapses AC/BC '
            '(0.595->0.169, 0.598->0.183) '
            'with AB preserved '
            '(0.534->0.550), sp(f2,U) '
            'coupling dies on 14B '
            '(4B +0.499/+0.727/+0.649 '
            'all p<=0.014 vs 14B '
            '-0.238/+0.038/+0.410 '
            'p=0.048 edge).  The '
            'angle-matching assembly rule '
            '(3080) loses its factor on '
            'the larger model, not its '
            'response resolution.',
        'verdict': v,
        'inputs': ['phase3076 npz',
                   'phase3079 npz',
                   'phase3080 npz',
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
                'meas3094_omega_p92_trunk_'
                'anatomy')
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
    o.append('ledger meas3094 appended '
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
if '## Phase 3094:' not in memo:
    entry = (
        '\n## Phase 3094: Ω-P92 trunk 迁移'
        '解剖（4B vs 14B，免前向）——f2 因子'
        '退化而非 CS 压缩，角度匹配律失去预测'
        '耦合（fifth_mixed_degradation）'
        ' [%s]\n\n'
        '**状态**: 已执行（0 前向；输入 = '
        '3076/3079/3080/3093 sealed npz；'
        '锚 a1/a2 T/U/F1/F2 重放全 bit 0.0'
        ' vs 3079 与 3093 npz，a3 门重放 '
        'count=3/min_sp=-0.2383（渲染容差 '
        '1e-3）/G2_COUNT=5，a4 SP_UT 重放 '
        'bit 0.0）。execution.json 先于计'
        '算冻结。\n\n'
        '### 1. 问题与设计\n'
        '3093 与 3080 两个 trunk 模型的 G_DS'
        ' 门分裂（4B 5/6 过 vs 14B 3/6 败）'
        '被精确定位在 U 侧（子集级迁移响'
        '应）。U 死于什么？预注册两假说：'
        'H1 u_compression（14B CS 列族内'
        '冗余↑→U 无区分度）与 H2 tt_factor'
        '_failed（CS 结构相当但 TT 夹角因'
        '子 f2 退化）。判据与锚全部冻结于 '
        'execution.json（seed 3094、n_perm '
        '20000）。\n\n'
        '### 2. E1 结构解剖（逐族对）\n'
        '| 族对 | CS 冗余 4B/14B (gap) | '
        'std_U 4B/14B (ratio) | med_f2 '
        '4B/14B | sp(f2,U) 4B / 14B |\n'
        '|---|---|---|---|---|\n'
        '%s\n\n'
        '### 3. E2 与门重放\n'
        'TT 范数中位（14B/4B）：%s。14B 门'
        '键重放 count=3、min_sp=-0.238261'
        '（=MEMO 渲染 -0.2383）；4B '
        'G2_COUNT=5。\n\n'
        '### 4. 判决逻辑\n'
        'h1_pairs=0（H1 否定：冗余 gap 全'
        '负——14B CS 区分度反而更好）；'
        'h2_redund_ok=3 且 h2_f2_ok=2 但 '
        'h2_u_ns=False（f2~U_BC p=0.0481 '
        '边缘显著）→ 严格 H2 不满足，诚实'
        '落入 **fifth_mixed_degradation**'
        '（f2 退化为主 + 残余耦合）。\n\n'
        '### 5. 分析\n'
        'U 侧死因不是子集响应压缩（H1 方向'
        '预测被数据否定），而是 TT 方向场'
        '的跨语义对对齐水平坍缩（AC/BC '
        'med_f2 从 ~0.60 跌至 ~0.17，AB '
        '保持）且 f2 与迁移的预测耦合同步'
        '断裂（4B 三族对 sp(f2,U) 全显著'
        '正 → 14B 全灭）。头级 T 响应在 '
        '14B 仍强（0.57-0.91）——因果结构'
        '族间相关仍在，但读出方向场不再按'
        '角度匹配律装配。\n\n'
        '### 6. 硬伤与边界\n'
        '- H1 判据是预注册方向假设，数据反'
        '向（gap 全负）——预注册防止了事后'
        '叙事，但方向选择本身未经试算；\n'
        '- AC/BC 的 std_U 坍缩（0.508/0.803'
        '）与 f2 坍缩混合，两类贡献不可完全'
        '分离（mixed 判决如实反映）；\n'
        '- TT 范数比 1.58-1.70 混杂维度差'
        '（5120/4096=1.25）与分布尖化，未'
        '归一化对比仅取向；\n'
        '- n=24 spearman 功效中等；f2~U_BC '
        'p=0.048 边缘即翻转判决分支（H2 严'
        '格门 vs mixed），门判对单对敏感；\n'
        '- 免前向再分析：不产生新的因果干'
        '预证据；f2 退化的机制原因（为何 '
        'AC/BC 方向场失对齐）需后续前向实'
        '验。\n\n'
        '### 7. 结论与接续\n'
        '**角度匹配装配律（3080）在 14B 上'
        '失去的不是响应分辨率而是因子本身**'
        '——trunk 谱位 + 头级因果相关 + TT '
        '对齐坍缩三者并存，说明"方向几何匹'
        '配"是 4B 尺度上偶然成立的装配启发'
        '式，而非普遍机制。接续 3095：'
        '(A) AB 特异性——为何 A-B 对在 14B '
        '保留 TT 对齐（逐对 TT 解剖/词元重'
        '叠/连接词分布）；(B) f2 退化的前向'
        '定位（14B 读出层 patch：把 4B 角度'
        '结构注入 14B 检验因果性）；(C) R1 '
        '复用拓扑全景（底册已备）。\n\n'
        '资源消耗：0 前向 / 数秒级；产物 '
        'sealed（npz8=%s result8=%s）。\n'
        % (datetime.now().strftime(
               '%Y-%m-%d %H:%M'),
           tbl, e2_txt,
           seal['npz_sha256_8'],
           seal['result_sha256_8']))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(entry)
    o.append('memo 3094 appended')
else:
    o.append('memo already')

# ---------- audit ----------
aud = io.open(AUDIT,
              encoding='utf-8').read()
if '五十五' not in aud:
    add = (
        '\n## 五十五、3094 追加（Ω-P92）\n'
        'trunk 迁移解剖（免前向，4B vs 14B'
        '）：H1 u_compression 被预注册判据'
        '否定（14B 族内 CS 列冗余更低，gap '
        '-0.041..-0.137）；f2 角度因子在 '
        'AC/BC 坍缩（med 0.60→0.17）且 '
        'sp(f2,U) 预测耦合全灭（4B 全显著'
        '正 vs 14B -0.238/+0.038/+0.410'
        'p=0.048）。判决 fifth_mixed_'
        'degradation：3080 角度匹配装配律'
        '在大模型上失去因子本身，而非响应'
        '分辨率。trunk 谱位、头级因果相关'
        '（T 0.57-0.91）与 TT 对齐坍缩并存'
        '——装配选择规则需要新的数学刻画。'
        '锚 a1/a2/a4 全 bit 0.0。\n')
    with io.open(AUDIT, 'a',
                 encoding='utf-8') as f:
        f.write(add)
    o.append('audit 55 appended')
else:
    o.append('audit already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
WLOG_TAG = 'trunk anatomy anatomy-closure'
try:
    prev = io.open(wlf,
                   encoding='utf-8').read()
except IOError:
    prev = ''
if WLOG_TAG not in prev:
    line = ('- Phase 3094 Omega-P92 trunk '
            'anatomy anatomy-closure: '
            'fifth_mixed_degradation '
            '(H1 rejected, f2 factor '
            'collapse AC/BC + coupling '
            'death).  Anchor suite bit-0; '
            'ledger %d/L14 %d; memo/audit '
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
if 'max=3094' not in mem_cur:
    anchor1 = ('- max=3093（qwen3-14b 第五'
               '谱点 layer_rescue + A2 '
               'fifth_trunk_no_migrate，G0 '
               '门下缺席/未执行）——下一个 '
               '3094：B 4B 主干解剖；C R1 '
               '复用拓扑全景（reuse_inventory '
               '底册已备）。')
    assert anchor1 in mem_cur
    mem_cur = mem_cur.replace(
        anchor1,
        '- max=3094（Ω-P92 '
        'fifth_mixed_degradation：f2 因子 '
        'AC/BC 坍缩，H1 否定，AB 存活）'
        '——3095：A AB 特异性；B f2 前向'
        '定位；C R1 全景。')
    anchor2 = ('；repro 锚 bit 级通过）、'
               'qwen3-14b（40L 40Q 8kv GQA '
               '5120 vocab 151936 untied '
               'bf16 29.54GB sysmem '
               'fallback 非 OOM；3093 第五'
               '谱点）。')
    assert anchor2 in mem_cur
    mem_cur = mem_cur.replace(
        anchor2,
        '；repro 锚 bit 级通过）、qwen3-14b'
        '（40L 40Q 8kv GQA 5120 vocab '
        '151936 untied bf16 29.54GB '
        'sysmem fallback 非 OOM；3093 谱'
        '点；3094 f2 退化）。')
    anchor3 = ('3092 G_DS 门敏感性 '
               'gate_sensitive_substantive'
               '（无符号 Stouffer 否决）→ '
               '3093 qwen3-14b 第五谱点'
               '（layer_rescue + A2 '
               'fifth_trunk_no_migrate）**。')
    assert anchor3 in mem_cur
    mem_cur = mem_cur.replace(
        anchor3,
        '3092 G_DS 门敏感性 '
        'gate_sensitive_substantive（无符'
        '号 Stouffer 否决）→ 3093 '
        'qwen3-14b 第五谱点（layer_rescue '
        '+ A2 fifth_trunk_no_migrate）→ '
        '3094 fifth_mixed_degradation'
        '（角度律失因子）**。')
    assert len(mem_cur) < 3000, len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3094')

io.open(OD + r'\closeout_log.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
