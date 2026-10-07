# -*- coding: utf-8 -*-
"""Idempotent single-arm closeout for
Phase 3098 (omega_p96_lastblock_
substep_anatomy).  Ledger meas3098 +
L14 +1 -> MEMO Phase 3098 -> audit
(五十九) -> wlog (case-matched tag) ->
MEMORY (anchors + net compression,
budgeted <3000)."""
import io
import json
import os
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
OD = (R13 + r'\phase3098'
      r'\omega_p96_lastblock_substep_'
      r'anatomy')
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
z = np.load(OD + r'\omega_p96_lastblock_'
            r'substep_anatomy.npz',
            allow_pickle=False)
v = res['verdict']
assert v == 'fifth_lastblock_mlp_' \
    'rewrite', v
assert str(z['VERDICT']) == v
assert bool(z['SMOKE']) is False
A = res['anchors']
assert A['ok'] is True
for k in ('c0', 'c1', 'c5', 'c2b',
          'c3'):
    for side in ('4B', '14B'):
        assert A['%s_%s' % (k, side)] \
            == 0.0, (k, side)
for side in ('4B', '14B'):
    assert A['c2_%s' % side] <= 1e-9
assert A['c4_ok'] is True
assert res['forwards'] == 768
S = res['stats']
assert S['attrib']['4B'] == 'mlp'
assert S['attrib']['14B'] == 'mlp'
assert S['hd3']['4B'] is True
assert S['hd3']['14B'] is True
assert S['frac']['4B_med_frac_mlp'] \
    >= 0.7
assert S['frac']['14B_med_frac_mlp'] \
    >= 0.7
o.append('3098 loaded: verdict=%s '
         'anchors_ok=True forwards=768'
         % v)

gmed = {}
for side in ('4B', '14B'):
    for key in ('AB', 'AC', 'BC'):
        for ci in (1, 2, 3):
            g = z['GMED_%s_%s_ci%d'
                  % (side, key, ci)]
            gmed['%s_%s_ci%d'
                 % (side, key, ci)] = g
rows = []
for key in ('AB', 'AC', 'BC'):
    for ci in (1, 2, 3):
        g4 = gmed['4B_%s_ci%d'
                  % (key, ci)]
        g14 = gmed['14B_%s_ci%d'
                   % (key, ci)]
        rows.append(
            '| %s_ci%d | %+.3f | %+.3f '
            '| %+.3f | %+.3f |'
            % (key, ci, g4[1] - g4[0],
               g4[2] - g4[1],
               g14[1] - g14[0],
               g14[2] - g14[1]))
tbl = '\n'.join(rows)
s_ac_mlp = gmed['14B_AC_ci1'][2] \
    - gmed['14B_AC_ci1'][1]
s_bc_mlp = gmed['14B_BC_ci1'][2] \
    - gmed['14B_BC_ci1'][1]
s_ab1_mlp = gmed['14B_AB_ci1'][2] \
    - gmed['14B_AB_ci1'][1]
s_ab2_mlp = gmed['14B_AB_ci2'][2] \
    - gmed['14B_AB_ci2'][1]
f4 = S['frac']['4B_med_frac_mlp']
f14 = S['frac']['14B_med_frac_mlp']
ag = S['agg']

# ---------- Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
has3098 = any(
    isinstance(m, dict)
    and m.get('phase') == 3098
    for m in led['measurements'])
if not has3098:
    meas = {
        'meas_id':
            'meas3098_omega_p96_'
            'lastblock_substep_anatomy',
        'phase': 3098,
        'claim':
            'Omega-P96 - last-block '
            'substep anatomy + '
            'intervention (768 forwards: '
            '32 prompts x 3 families x 4 '
            'hook configs x 2 arms; 4B '
            'block35 / 14B block39).  '
            'Anchors: c0 (h0+a)+m == '
            'block-out bit 0; c1 '
            'head(norm(h2)) vs native '
            'logits bit 0; c5 skip == '
            'head(norm(h0)) bit 0; c2b '
            'F2S post == F2I native bit '
            '0; c2 vs sealed F2_CTT '
            '9.9e-13/1.2e-12; c3 pre vs '
            '3096 F2P[:,NL-1] bit 0; c4 '
            '3096 npz sha match.  '
            'Verdict '
            'fifth_lastblock_mlp_'
            'rewrite: the conditional '
            'TT rewriter sits in the '
            'MLP substep of the last '
            'block - med frac_mlp '
            '0.947/0.949 over active '
            'groups; 14B formal AC/BC '
            'destroyed by MLP in ONE '
            'step (-0.306/-0.194) while '
            'Shakespearean amplified '
            '(+0.379/+0.314), attn step '
            'within +/-0.04; substep '
            'sums reproduce the 3097 '
            'cliff steps (AC_ci1 '
            '-0.310 = -0.004 + -0.306). '
            'H_D3 True/True: KL '
            '(native||no_mlp) 0.148/'
            '0.150 > (native||no_attn) '
            '0.062/0.064, top1 agree '
            '0.813/0.823 < 0.906/0.948. '
            'TT-norm ratio post/pre >1 '
            'in ALL groups (x1.1-3.0): '
            'the rewrite is amplifying; '
            'MLP increment cross-family '
            'cos f2dm high for ci2 '
            '(0.57-0.99, family-common '
            'write) but negative for '
            '14B formal AC/BC (-0.11/'
            '-0.17, family-idiosyncratic '
            'write) - the family '
            'commonality of the MLP '
            'write flips with condition, '
            'which IS the f2 collapse/'
            'amplification mechanism.  '
            'run1 archived '
            '(discarded_run1: H_D3 '
            'column mislabel + forwards '
            'metadata); rerun npz sha8 '
            'identical eeba6c18.',
        'verdict': v,
        'inputs': ['phase3096 npz',
                   'phase3096 seal',
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
                'meas3098_omega_p96_'
                'lastblock_substep_'
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
    o.append('ledger meas3098 appended '
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
if '## Phase 3098:' not in memo:
    entry = (
        '\n## Phase 3098: Ω-P96 末位 '
        'block 子步解剖 + 干预实验——条'
        '件选择性 TT 重写器定位在最后 '
        'block 的 MLP 子步，attn 子步对'
        '读出不可见'
        '（fifth_lastblock_mlp_rewrite'
        '） [%s]\n\n'
        '**状态**: 已执行（768 前向 = '
        '32 prompt × 3 族 × 4 hook 配置 '
        '× 2 臂；4B block35 / 14B '
        'block39）。锚 c0（(h0+a)+m == '
        'hook block 输出）bit 0、c1（'
        'head(norm(h2)) vs native '
        'logits）bit 0、c5（skip == '
        'head(norm(h0))）bit 0、c2b（'
        'F2S post == F2I native）bit '
        '0、c3（pre vs 3096 F2P 第 '
        'NL-1 层列）bit 0、c2（vs '
        'sealed F2_CTT）9.9e-13/1.2e-12 '
        '≤1e-9、c4（3096 npz sha）一'
        '致。execution.json 先冻结。'
        'run1 事故：H_D3 列错位'
        '（dom=2/oth=1 取了 skip/'
        'no_mlp 而非 no_mlp/no_attn）'
        '+ forwards 元数据错（256，实'
        '为 768）——修脚本归档 '
        'discarded_run1 后重跑；npz '
        'sha8 两次同为 eeba6c18（判决'
        '与全部主统计逐位一致），仅 '
        'result.json agg 段两键受影'
        '响，H_D3 在错列下亦为 True'
        '（判决稳健）。\n\n'
        '### 1. 问题与设计\n'
        '3097 把断崖定位到 block39 整'
        '块。本 Phase 问：块内哪个子步'
        '执行重写（attn vs MLP），及其'
        '对输出分布是否因果材料。每 '
        'prompt 4 次前向，只差最后 '
        'block 的 hook：native（记录 '
        'h0=hs[NL-1]、attn 出 a、mlp '
        '出 m、块出 h2）/ no_attn（a=0'
        '）/ no_mlp（m=0）/ skip（块返'
        '回输入）。子步读出 Z_s=head('
        'norm(h_s))，s∈{pre,att,post}'
        '；TT_s[k]=Z_s[pref_k]−'
        'Z_s[base_k]；F2S=同模型跨族 '
        'cos（3096 定义）。断崖分解：'
        'attn_step=med(att)−med(pre)，'
        'mlp_step=med(post)−med(att)，'
        'frac_mlp=|mlp_step|/(|attn_'
        'step|+|mlp_step|)。判决门：'
        'active 组 |med(post)−med(pre)'
        '|≥0.10；med frac_mlp≥0.7→'
        'mlp、≤0.3→attn；H_D3 因果材'
        '料性：med KL(native||no_dom)'
        '>med KL(native||no_oth) 且 '
        'top1 一致率更低。\n\n'
        '### 2. E1 子步分解（组中位 '
        'step）\n'
        '| 组 | 4B attn | 4B mlp | '
        '14B attn | 14B mlp |\n'
        '|---|---|---|---|---|\n'
        '%s\n\n'
        '**子步分解与 3097 断崖步逐位'
        '衔接**：14B AC_ci1 −0.310 = '
        'attn −0.004 + mlp %+.3f；'
        'BC_ci1 −0.212 = −0.018 + '
        '%+.3f；AB_ci2 +0.300 = −0.013 '
        '+ %+.3f；4B BC_ci2 +0.42 = '
        '+0.009 + +0.414。attn 子步全'
        '部 9 组步幅仅 ±0.09 内（多数 '
        '±0.04 内）。\n\n'
        '### 3. 干预的因果材料性（H_D3'
        '）\n'
        'KL(native||cfg) 组中位 / top1 '
        '一致率：4B no_attn 0.062/'
        '0.906、no_mlp 0.148/0.813、'
        'skip 0.346/0.656；14B no_attn '
        '0.064/0.948、no_mlp 0.150/'
        '0.823、skip 0.257/0.688。移'
        '除 MLP 子步的扰动显著大于移'
        '除 attn 子步（KL ×2.3-2.4，'
        'top1 一致率 −0.09/−0.13）→ '
        'H_D3 True/True。skip 超加性'
        '（0.062+0.148=0.21 < 0.346）'
        '——attn 与 mlp 的读出贡献相互'
        '依赖：attn 经改变 MLP 输入'
        '（ln2(h1) vs ln2(h0)）间接作'
        '用，其直接读出足迹近乎不可'
        '见。\n\n'
        '### 4. 重写的形态：放大 + 增'
        '量方向家族共性随条件反转\n'
        'TT 范数比 post/pre 全部 9 组 '
        '>1（4B ×1.11-2.15，14B '
        '×1.36-3.05）——**末块 MLP 是'
        '条件效应的净放大器**，14B '
        'formal 的"f2 摧毁"发生在 TT '
        '整体放大的背景下。MLP 增量向'
        '量（DTT_mlp=TT_post−TT_att）'
        '的跨族 cos（f2dm）：14B ci2 '
        '组 0.57-0.89、4B ci2 组 '
        '0.98-0.99（**写家族共同方向**'
        '→跨族对齐放大），而 14B '
        'formal AC/BC −0.11/−0.17'
        '（**写族特异方向**→跨族对齐'
        '摧毁）。attn 增量 f2da 中等'
        '（0.04-0.61）无条件模式。家'
        '族共性随条件反转就是 f2 坍缩'
        '/放大的机制本体。\n\n'
        '### 5. 判决逻辑\n'
        'active 组：4B 4 个（全 ci2 + '
        'BC_ci3）、14B 6 个（formal '
        'AB/AC/BC_ci1 + 全 ci2）。'
        'med frac_mlp：4B %.3f、14B '
        '%.3f ≥0.7 → 双臂 mlp 主导；'
        'H_D3 True/True → '
        '**fifth_lastblock_mlp_'
        'rewrite**。\n\n'
        '### 6. 分析（关键洞察）\n'
        '**条件选择性 TT 重写器在末位 '
        'block 的 MLP 子步**。同一结论'
        '三重证据：(1) 分解——frac_mlp '
        '0.947/0.949，14B formal '
        'AC/BC 摧毁（−0.306/−0.194）'
        '与 Shakespearean 放大'
        '（+0.379/+0.314）同由 MLP 一'
        '步完成；(2) 因果——H_D3 门双 '
        'True（KL 0.148>0.062、'
        '0.150>0.064）；(3) 跨模型——'
        '4B 同构（ci2 放大 +0.32/+0.30/'
        '+0.41 全 MLP），机制共有、策'
        '略随规模。RDC 机制链更新：'
        'upstream（L37/38 rescue 带）→ '
        'block39 attn（准备输入，直接'
        '读出足迹可忽略）→ **block39 '
        'MLP（条件选择性 TT 重写：族'
        '共同写入=放大，族特异写入=摧'
        '毁）** → final norm → 读出。'
        '条件化齿轮的最深候选件更名：'
        '**readout-preconditioning MLP '
        'in final block**。\n\n'
        '### 7. 硬伤与边界\n'
        '- 清零消融非最小干预：no_attn '
        '使 mlp 输入变 ln2(h0)，attn '
        '的直接/间接贡献未完全解耦；'
        'attn"读出不可见"缺 bf16 ULP '
        '级定量（F2I no_mlp vs skip '
        '24/24 组全异，幅度 0.005-1.0'
        '）；\n'
        '- last position 单 token，行'
        '为层仅 KL/top1，自由生成未'
        '测；\n'
        '- n=8 体/ci 组；causal-'
        'connective 范式；\n'
        '- run1 H_D3 列错位事故（如实'
        '登记，见状态）。\n\n'
        '### 8. 结论与接续\n'
        '接续 3099：(i) block39 MLP 内'
        '部再分解——top-k 中间神经元'
        '对 TT 重写向量的贡献（对接 '
        'P53/P54）；(ii) MLP 增量方向'
        '的可预测性——3093 L37/38 注入'
        '结构能否预测 3098 增量（上游'
        '-重写器对接）；(C) R1 复用拓'
        '扑全景（底册已备）。\n\n'
        '资源消耗：768 前向约 5.3 分钟'
        '；产物 sealed（npz8=%s '
        'result8=%s；run1 归档于 '
        'discarded_run1）。\n'
        % (datetime.now().strftime(
               '%Y-%m-%d %H:%M'),
           tbl, s_ac_mlp, s_bc_mlp,
           s_ab2_mlp, f4, f14,
           seal['npz_sha256_8'],
           seal['result_sha256_8']))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(entry)
    o.append('memo 3098 appended')
else:
    o.append('memo already')

# ---------- audit ----------
aud = io.open(AUDIT,
              encoding='utf-8').read()
if '五十九' not in aud:
    add = (
        '\n## 五十九、3098 追加（Ω-P96）\n'
        '末位 block 子步解剖 + 干预'
        '（768 前向，4B block35/14B '
        'block39，每 prompt native/'
        'no_attn/no_mlp/skip）：条件选'
        '择性 TT 重写器定位在最后 '
        'block 的 **MLP 子步**——'
        'frac_mlp 0.947/0.949，14B '
        'formal AC/BC 单步摧毁 '
        '-0.306/-0.194 与 Shakespearean '
        '放大 +0.379/+0.314 全由 MLP '
        '完成，attn 步幅 ±0.09 内；子'
        '步分解与 3097 断崖步逐位衔接'
        '（AC_ci1 −0.310 = −0.004 + '
        '−0.306）。H_D3：KL no_mlp '
        '0.148/0.150 > no_attn 0.062/'
        '0.064，top1 0.813/0.823 < '
        '0.906/0.948。TT 范数比全组 '
        '>1（×1.1-3.0，放大式重写）；'
        'MLP 增量跨族 cos ci2 高'
        '（0.57-0.99，族共同写入）而 '
        '14B formal AC/BC 负（-0.11/'
        '-0.17，族特异写入）——家族共'
        '性随条件反转即 f2 坍缩/放大机'
        '制。判决 '
        'fifth_lastblock_mlp_rewrite。'
        '锚 c0/c1/c5/c2b/c3 bit 0、c2 '
        '≤1.2e-12；run1 H_D3 列错位归'
        '档重跑（npz sha8 两次同为 '
        'eeba6c18）。\n')
    with io.open(AUDIT, 'a',
                 encoding='utf-8') as f:
        f.write(add)
    o.append('audit 59 appended')
else:
    o.append('audit already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
WLOG_TAG = ('lastblock substep '
            'anatomy-closure')
try:
    prev = io.open(wlf,
                   encoding='utf-8').read()
except IOError:
    prev = ''
if WLOG_TAG not in prev:
    line = ('- Phase 3098 Omega-P96 '
            'lastblock substep '
            'anatomy-closure: '
            'fifth_lastblock_mlp_rewrite '
            '(rewriter = last-block MLP '
            'substep: frac_mlp 0.947/'
            '0.949; 14B formal destroyed '
            '-0.31/-0.19 and Shakespearean '
            'amplified +0.38/+0.31 all by '
            'MLP; attn readout-invisible; '
            'H_D3 KL 0.148>0.062 / '
            '0.150>0.064).  Anchors c0-c5 '
            'pass; run1 column bug '
            'archived, rerun bit-identical '
            '(npz8 eeba6c18); ledger '
            '%d/L14 %d.\n'
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
if 'max=3098' not in mem_cur:
    a1 = ('- max=3097（Ω-P95 '
          'cliff_single_step：block39 '
          '重写器）——3098：分解；C R1。')
    assert a1 in mem_cur, 'anchor1'
    mem_cur = mem_cur.replace(
        a1,
        '- max=3098（Ω-P96 '
        'lastblock_mlp_rewrite：重写器'
        '=末块MLP子步）——3099：MLP内'
        '分解；C R1。')
    a3 = ('3096 lens_late → 3097 '
          'cliff**。')
    assert a3 in mem_cur, 'anchor3'
    mem_cur = mem_cur.replace(
        a3,
        '3096 lens_late → 3097 '
        'cliff → 3098 mlp_substep**。')
    c1 = ('（GlmForCausalLM 40L 32H '
          '2kv 4096 inter 13696 vocab '
          '151552 tied=False bf16 '
          '18.84GB>17.09GB VRAM 靠 '
          'sysmem fallback 非 OOM；'
          'repro 锚 bit 级通过）')
    assert c1 in mem_cur, 'comp1'
    mem_cur = mem_cur.replace(
        c1,
        '（Glm4 40L 32H 2kv 4096 '
        'vocab 151552 tied=False bf16 '
        'sysmem fallback；repro 锚 '
        'bit 级过）')
    c2 = ('qwen3-14b（40L 40Q 8kv GQA '
          '5120 vocab 151936 untied '
          'bf16 29.54GB sysmem fallback '
          '非 OOM；3093 谱点；3095 '
          'ci2 主导）')
    assert c2 in mem_cur, 'comp2'
    mem_cur = mem_cur.replace(
        c2,
        'qwen3-14b（40L 40Q 8kv 5120 '
        'vocab 151936 untied bf16 '
        'sysmem fallback；3093 谱点；'
        '3095 ci2 主导）')
    assert len(mem_cur) < 3000, \
        len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3098')

io.open(OD + r'\closeout_log.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
