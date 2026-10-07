# -*- coding: utf-8 -*-
"""Idempotent single-arm closeout for
Phase 3096 (omega_p94_f2_lens_layer_
profile).  Ledger meas3096 + L14 +1 ->
MEMO Phase 3096 -> audit addendum
(五十七) -> wlog (closure-phrase
guarded, tag case-matched) -> MEMORY
(2 anchors this round, char-budgeted).
Reads only sealed artifacts."""
import io
import json
import os
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
OD = (R13 + r'\phase3096'
      r'\omega_p94_f2_lens_layer_profile')
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
z = np.load(OD + r'\omega_p94_f2_lens_'
            r'layer_profile.npz',
            allow_pickle=False)
v = res['verdict']
assert v == 'fifth_lens_late_assembly', v
assert str(z['VERDICT']) == v
assert bool(z['SMOKE']) is False
S = res['stats']
A = res['anchors']
assert A['ok'] is True
assert A['a1_diff_4B'] == 0.0
assert A['a1_diff_14B'] == 0.0
assert A['a2_diff_4B'] <= 1e-9
assert A['a2_diff_14B'] <= 1e-9
assert S['decision']['H_B1'] is True
assert S['decision']['H_B2'] is False
assert S['decision']['H_B3'] is False
o.append('3096 loaded: verdict=%s '
         'anchors_ok=True' % v)

KEYCi = ['%s_ci%d' % (key, ci)
         for key in ('AB', 'AC', 'BC')
         for ci in (1, 2, 3)]
SHOWD = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
GIDX = [int(round(d * 10)) for d in SHOWD]


def f2(x):
    return '%.2f' % x


rows = []
for ck in KEYCi:
    cv = S['curves'][ck]
    rows.append(
        '| %s | %s | %s |'
        % (ck,
           ' / '.join(f2(cv['m4B'][i])
                      for i in GIDX),
           ' / '.join(f2(cv['m14B'][i])
                      for i in GIDX)))
tbl = '\n'.join(rows)
dd_txt = '；'.join(
    '%s=%.1f' % (ck, S['d_div'][ck])
    for ck in KEYCi)
auc_ci2 = sum(
    S['auc']['14B_%s_ci2' % key]
    for key in ('AB', 'AC', 'BC')) / 3.0
auc_ci1 = sum(
    S['auc']['14B_%s_ci1' % key]
    for key in ('AB', 'AC', 'BC')) / 3.0

# ---------- Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
has3096 = any(
    isinstance(m, dict)
    and m.get('phase') == 3096
    for m in led['measurements'])
if not has3096:
    meas = {
        'meas_id':
            'meas3096_omega_p94_f2_lens_'
            'layer_profile',
        'phase': 3096,
        'claim':
            'Omega-P94 (192 forwards, '
            'dual-arm 4B+14B sequential) '
            '- logit-lens layer profile '
            'of the TT angle factor f2 '
            'on the frozen 3076 texts '
            '(96 prompts/arm, all '
            'last-position hidden '
            'states, Z_L=lm_head(norm'
            '(h_L)), L=NL post-norm '
            'native path).  Anchors: a1 '
            'lens(final) vs native '
            'logits BIT-0 both arms; '
            'a2 native final f2 vs '
            'sealed F2_CTT 9.9e-13/1.2e-'
            '12 <=1e-9; a2b lens-path '
            'equal.  Verdict '
            'fifth_lens_late_assembly: '
            'early-layer cross-family '
            'alignment exists on BOTH '
            'models (f2 0.3-0.8 at '
            'd<=0.3); the 14B AC/BC '
            'collapse concentrates in '
            'the final ~4 layers '
            '(L37-39 + final norm): '
            'd=0.9->1.0 lens f2 AC_ci1 '
            '0.30->0.03, BC_ci1 0.20->'
            '-0.00, while Shakespearean '
            '(ci2) is AMPLIFIED by the '
            'final readout (BC 0.27->'
            '0.52, AB 0.36->0.76) and '
            '4B is preserved (AC_ci1 '
            '0.40->0.60).  d_div 8x1.0 '
            '+ BC_ci1 0.9 -> H_B1 True '
            '(min 0.9>=0.6); H_B2 '
            'False; H_B3 False (med '
            'gap 0.05<0.15).  The 3080 '
            'angle-matching ASSEMBLY '
            'step sits at the unembed '
            'readout; 14B lost the '
            'assembly policy, not the '
            'hidden-field direction '
            'structure.  The collapse '
            'band coincides with the '
            '3093 rescue layers '
            '(L37/L38).',
        'verdict': v,
        'inputs': ['phase3079 npz',
                   'phase3093 npz',
                   'frozen 3076 texts'],
        'outputs': [OD]}
    led['measurements'].append(meas)
    for l in led['linkage']:
        if (isinstance(l, dict)
                and l.get('link_id')
                == 'L14_readout_spectrum_'
                'cross_model'):
            cs = l['connects']
            cs.append(
                'meas3096_omega_p94_f2_'
                'lens_layer_profile')
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
    o.append('ledger meas3096 appended '
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
if '## Phase 3096:' not in memo:
    entry = (
        '\n## Phase 3096: Ω-P94 f2 逐层 '
        'logit-lens 层位剖面（4B+14B 双臂，'
        '192 前向）——14B 角度坍缩是读出装'
        '配的晚发事件，断崖在最后 ~4 层'
        '（fifth_lens_late_assembly） '
        '[%s]\n\n'
        '**状态**: 已执行（192 前向；'
        '4B 先跑完释放显存再 14B；输入 = '
        'frozen 3076 文本 + 3079/3093 '
        'sealed npz 参照；锚 a1 '
        'lens(final) vs 原生 logits 双臂 '
        'bit 0.0，a2 原生终层 f2 vs '
        'sealed F2_CTT 9.9e-13 / 1.2e-12 '
        '≤1e-9，a2b lens 路径相等）。'
        'execution.json 先冻结（判决门 + '
        'd 网格 + lens 管线定义）。'
        'hidden_states[NL] 为 final norm '
        '后状态——L=NL 直接 head（双'
        '重 norm 陷阱在 smoke 期被 a1 '
        'bit 锚抓获并修正）。\n\n'
        '### 1. 问题与设计\n'
        '3094/3095 确立 14B 的 TT 角度因'
        '子 f2 在 AC/BC 坍缩且存活由 '
        'prefix 条件主导。悬而未决：坍缩'
        '发生在堆栈哪一层——表示层（早）'
        '还是读出装配（晚）？管线：3076-'
        'identical 96 prompt/臂，逐条前向'
        '采全部层最后位置隐状态（'
        'output_hidden_states），logit '
        'lens Z_L=lm_head(norm(h_L))，'
        'TT_L[k]=Z_L[pref_k]−Z_L[base_k]'
        '，f2_L=cos；归一化深度 d=L/NL，'
        '11 点共同网格（36L vs 40L 最近层'
        '对齐）。d_div=Δf2(4B−14B)≥0.2 的'
        '首达深度（尾均值≥0.15，无则 1.0'
        '）；判决门 H_B1（min d_div AC/BC '
        'ci1 ≥0.6）→ H_B2（max ≤0.3）→ '
        'H_B3（ci2 与 ci1 中位差 ≥0.15）'
        '→ mixed。\n\n'
        '### 2. E1 曲线（med f2，d=0.1 / '
        '0.3 / 0.5 / 0.7 / 0.9 / 1.0）\n'
        '| 组 | 4B | 14B |\n|---|---|---|\n'
        '%s\n\n'
        'd_div：%s。判决 H_B1=True（min '
        '0.9 ≥ 0.6）、H_B2=False、H_B3='
        'False（med_ci2 1.00 vs med_ci1 '
        '0.95，差 0.05 < 0.15）→ '
        '**fifth_lens_late_assembly**。\n\n'
        '### 3. 分析（关键洞察）\n'
        '**14B 的 f2 坍缩是读出装配的晚发'
        '事件：隐藏场早层的跨族方向对齐在'
        '两个模型中同样存在，坍缩集中在最'
        '后 ~4 层（L37-39 + final norm）'
        '的读出断崖。**（1）早层普遍对齐：'
        'd≤0.3 两模型全部组 f2 达 0.3-0.8'
        '——prefix−base 差方向的跨族对齐'
        '在表示层早段即形成，不是 14B 的'
        '表示缺陷；（2）断崖定位：14B '
        'AC_ci1 d=0.9→1.0 为 0.30→0.03、'
        'BC_ci1 0.20→-0.00，4B 同位反而保'
        '持/回升（0.40→0.60）——坍缩发生'
        '在最后 4 层带内（全层 F2P 已存 '
        'npz 可免前向细化），恰与 3093 '
        'rescue 层带（L37/L38）重合——同'
        '一段堆栈既是 G_DS 迁移可救援段也'
        '是 f2 坍缩发生段；（3）条件选择性'
        '读出：Shakespearean（ci2）最后一步'
        '被读出放大（BC 0.27→0.52、AB '
        '0.36→0.76、AC 0.49→0.63），'
        'formal 的 AC/BC 被抹除——3095 的'
        '"对齐存活=Shakespearean 全体+'
        'formal×AB"落到层位上就是 final '
        'readout 的条件选择性策略；（4）'
        'AUC（14B，f2 深度积分）ci2 组均 '
        '%.3f > ci1 组均 %.3f——'
        'Shakespearean 全深度更健康，与终'
        '层排序一致。'
        '理论定位：**3080 角度匹配装配律的'
        '"装配"步骤精确落在 unembedding '
        '读出端；14B 失去的是读出装配策略'
        '，不是隐藏场的方向结构**。\n\n'
        '### 4. 硬伤与边界\n'
        '- logit lens 是探针不是模型真实'
        '计算路径：中间层读出把 final '
        'norm+head 外推到中间态，"早层对'
        '齐"是 lens 意义的对齐；\n'
        '- d=0.9→1.0 折叠了 4 层（L36→'
        'L40）——断崖层位粒度粗，但全层 '
        'F2P（24 对 × 41 层）已 sealed，'
        '细化可免前向；\n'
        '- d_half 统计被 d=0 嵌入层伪影污'
        '染（f2(L=0)=0 → 全 0.0），本轮弃'
        '用，以 d_div/AUC 为准；\n'
        '- n=8 body/ci 组中位数；SMOKE '
        '(ci1-only) 仅验管线；\n'
        '- 14B 显存 fallback（17GB 卡 / '
        '29.5GB 权重）单次 OOM 警告非致命'
        '，lens 批量 96×151936 走 '
        'sysmem。\n\n'
        '### 5. 结论与接续\n'
        '3094 的"f2 因子消失"被精确定位为'
        '读出装配事件：**同一段堆栈（'
        'L37-39+final norm）既承载 3093 '
        '的 rescue 效应，也是角度结构被条'
        '件选择性压缩的位置**。隐藏场的跨'
        '族方向结构在两个规模上都存在——'
        '规模差异在"读出如何装配这些结构"'
        '。接续 3097：(i) 免前向细化断崖层'
        '位（sealed F2P 全层 41 点逐层表'
        '）；(ii) 14B 读出端干预（final '
        'norm 前后 patch / head 子空间投'
        '影，检验因果性）；(C) R1 复用拓扑'
        '全景（底册已备）。\n\n'
        '资源消耗：192 前向 / ~2.5 分钟；'
        '产物 sealed（npz8=%s '
        'result8=%s）。\n'
        % (datetime.now().strftime(
               '%Y-%m-%d %H:%M'),
           tbl, dd_txt, auc_ci2, auc_ci1,
           seal['npz_sha256_8'],
           seal['result_sha256_8']))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(entry)
    o.append('memo 3096 appended')
else:
    o.append('memo already')

# ---------- audit ----------
aud = io.open(AUDIT,
              encoding='utf-8').read()
if '五十七' not in aud:
    add = (
        '\n## 五十七、3096 追加（Ω-P94）\n'
        'f2 逐层 logit-lens 层位剖面（双臂 '
        '4B+14B，192 前向）：14B 的 f2 角'
        '度坍缩定位为读出装配晚发事件——早'
        '层（d≤0.3）两模型跨族对齐均高'
        '（0.3-0.8），断崖集中在最后 ~4 层'
        '（L37-39+final norm，与 3093 '
        'rescue 层带重合）；条件选择性：'
        'Shakespearean 组被终层读出放大'
        '（BC 0.27→0.52），formal 的 AC/BC '
        '被抹除（0.20→-0.00）。判决 '
        'fifth_lens_late_assembly（H_B1 '
        'min d_div 0.9 ≥ 0.6）。3080 装配'
        '律的"装配"落在 unembedding 端：'
        '14B 失去的是读出装配策略而非隐藏'
        '场方向结构。锚 a1 双臂 bit 0.0。\n')
    with io.open(AUDIT, 'a',
                 encoding='utf-8') as f:
        f.write(add)
    o.append('audit 57 appended')
else:
    o.append('audit already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
WLOG_TAG = 'f2 lens profile anatomy-closure'
try:
    prev = io.open(wlf,
                   encoding='utf-8').read()
except IOError:
    prev = ''
if WLOG_TAG not in prev:
    line = ('- Phase 3096 Omega-P94 f2 '
            'lens profile '
            'anatomy-closure: '
            'fifth_lens_late_assembly '
            '(collapse is a readout-'
            'assembly event, final ~4 '
            'layers; early-layer '
            'alignment present on both '
            'models).  Anchors a1 bit-0 '
            'x2; ledger %d/L14 %d; '
            'memo/audit appended.\n'
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
if 'max=3096' not in mem_cur:
    anchor1 = ('- max=3095（Ω-P93 '
               'fifth_ab_mixed：三假说全拒，'
               'f2 存活=ci2+AB-ci1）——3096：'
               'B 前向定位；C R1 全景。')
    assert anchor1 in mem_cur, 'anchor1'
    mem_cur = mem_cur.replace(
        anchor1,
        '- max=3096（Ω-P94 '
        'fifth_lens_late_assembly：f2 坍'
        '缩=读出端断崖）——3097：细化层'
        '位；C R1。')
    anchor3 = ('3095 fifth_ab_mixed**。')
    assert anchor3 in mem_cur, 'anchor3'
    mem_cur = mem_cur.replace(
        anchor3,
        '3095 fifth_ab_mixed → 3096 '
        'lens_late**。')
    assert len(mem_cur) < 3000, \
        len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3096')

io.open(OD + r'\closeout_log.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
