# -*- coding: utf-8 -*-
"""Phase 3099 closeout (Omega-P97
mlp neuron anatomy).  Five idempotent
writes: ledger meas3099 + L14 -> MEMO
Phase 3099 -> audit 60 -> wlog ->
MEMORY max=3099 (with net compression
to stay <3000 chars)."""
import io
import json
import os
from datetime import datetime

import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P99 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3099'
       r'\omega_p97_mlp_neuron_anatomy')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
AUDIT = (ROOT + r'\research\gpt5\docs'
         r'\hdmcc_knowledge_map_review_'
         r'20260921.md')
WLOG_DIR = (ROOT + r'\.workbuddy'
            r'\memory')
OD = P99

res = json.load(io.open(
    P99 + r'\result.json',
    encoding='utf-8'))
seal = json.load(io.open(
    P99 + r'\seal.json',
    encoding='utf-8'))
assert res['verdict'] == \
    'fifth_mlp_neuron_diffuse', \
    res['verdict']
assert not res['smoke']
S = res['stats']
G = S['gates']
v = res['verdict']

n_act = sum(
    1 for k in S['active'].values()
    if k['active'])


def rng(key_prefix, ci):
    vals = [S[key_prefix][
        '%s_%s_ci%d' % (side, key, ci)]
        for side in ('4B', '14B')
        for key in ('AB', 'AC', 'BC')]
    return min(vals), max(vals)


sh1 = rng('share', 1)
sh2 = rng('share', 2)
sh3 = rng('share', 3)
nf1 = rng('nf2', 1)
nf2v = rng('nf2', 2)
nf3 = rng('nf2', 3)
br1 = rng('bridge', 1)
br2 = rng('bridge', 2)
br3 = rng('bridge', 3)

o = []

# ---------- Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
has3099 = any(
    isinstance(m, dict)
    and m.get('phase') == 3099
    for m in led['measurements'])
MID = 'meas3099_omega_p97_mlp_' \
      'neuron_anatomy'
if not has3099:
    claim = (
        'Omega-P97 - inside the '
        'last-block MLP: neuron-level '
        'anatomy of the conditional TT '
        'rewrite (192 native-only '
        'forwards = 32 prompts x 3 '
        'families x 2 arms; 4B block35 '
        'inter 9728 / 14B block39 inter '
        '17408 -- config.json '
        'intermediate_size is 17408; '
        'earlier 13824 notes were '
        'wrong).  Anchors: d1 manual '
        'WHOLE-sequence MLP replay '
        '(x=ln2(h0+a); '
        'act=act_fn(gate(x))*up(x); '
        'm_man=down(act)) == hooked m '
        'bit 0 - whole-sequence GEMM '
        'shapes REQUIRED, per-position '
        'GEMV replay differed by up to '
        '0.25 in bf16 (smoke finding); '
        'd2 (h0+a)+m_man == hooked h2 '
        'bit 0; d3 recomputed F2I '
        'native vs 3098 sealed bit 0; '
        'd4 3098 npz sha8 == eeba6c18; '
        'd5 head(norm(h2)) vs native '
        'logits bit 0.  Verdict '
        'fifth_mlp_neuron_diffuse: '
        'H_E1 False - pooled median '
        'top-256 L1 share of '
        '|Delta-act| = 0.4457 < 0.5 '
        '(but far above the uniform '
        'baseline 1.5-2.6 pct: '
        'moderately concentrated, a '
        'few hundred mid-magnitude '
        'neurons, no giant units; 14B '
        'ci2 groups highest 0.4965-'
        '0.5255); H_E2a False - active '
        'ci2 cross-family top-64 '
        'Jaccard 0.5901 vs 14B formal '
        'AC/BC ci1 control 0.5647 '
        '(ratio 1.05 << 3): no '
        'style-specific shared neuron '
        'SET (all J64 0.49-0.64, '
        'baseline dilution); H_E3 '
        'False - pooled bridge '
        'cos(head(norm(Delta-m)), '
        'TT_logit) = 0.7826 < 0.8 '
        '(ci2 highest 0.84-0.94, ci3 '
        'lowest 0.55-0.73).  KEY: nf2 '
        '(cross-family cos of '
        'Delta-act) reproduces the '
        '3098 f2dm condition flip in '
        'neuron space - 4B ci2 0.80-'
        '0.85 (style writes '
        'family-common directions) vs '
        '14B ci1 AC/BC 0.13-0.22 '
        '(formal writes '
        'family-idiosyncratic) - the '
        'family commonality is encoded '
        'in continuous DIRECTIONS, not '
        'in discrete neuron SETS.  '
        'Top-1024 contribution rebuild '
        '(|Delta-act_j|*||W_down[:,j]|'
        '| ranking): SHM 0.965-0.984, '
        'COSM 0.987-0.996 - Delta-m is '
        'almost fully reconstructed '
        'linearly by the top-1024 '
        'neurons.')
    meas = {
        'meas_id': MID,
        'phase': 3099,
        'claim': claim,
        'verdict': v,
        'inputs': ['phase3098 npz',
                   'phase3098 seal'],
        'outputs': [P99]}
    led['measurements'].append(meas)
    for l in led['linkage']:
        if (isinstance(l, dict)
                and l.get('link_id')
                == 'L14_readout_spectrum_'
                'cross_model'):
            l['connects'].append(MID)
            break
    old = led.pop('ledger_sha256_8', None)
    body = json.dumps(
        led, sort_keys=True,
        ensure_ascii=False)
    led['ledger_sha256_8'] = (
        hashlib.sha256(
            body.encode('utf-8'))
        .hexdigest()[:8])
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f,
                  ensure_ascii=False,
                  indent=1)
    o.append('ledger meas3099 appended '
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
if '## Phase 3099:' not in memo:
    nf_tbl = (
        '| 组 | 4B nf2 | 14B nf2 |\n'
        '|---|---|---|\n')
    for key in ('AB', 'AC', 'BC'):
        for ci in (1, 2, 3):
            nf_tbl += (
                '| %s_ci%d | %.3f | %.3f '
                '|\n'
                % (key, ci,
                   S['nf2']['4B_%s_ci%d'
                            % (key, ci)],
                   S['nf2']['14B_%s_ci%d'
                            % (key, ci)]))
    entry = (
        '\n## Phase 3099: Ω-P97 末位 '
        'block MLP 神经元分解——重写弥散'
        '于宽激活增量、方向编码而非集合'
        '编码（fifth_mlp_neuron_diffuse'
        '） [%s]\n\n'
        '**状态**: 已执行（192 前向 '
        'native-only = 32 prompt × 3 '
        '族 × 2 臂；4B block35 inter '
        '9728 / 14B block39 inter '
        '17408——config.json 实测 '
        '17408，早前 13824 记录有误）。'
        '锚 d1（手工**全序列** MLP 重放'
        ' act=act_fn(gate(ln2(h0+a)))*'
        'up(x)、m_man=down(act) == '
        'hook m）bit 0、d2（(h0+a)+'
        'm_man == hook h2）bit 0、d3'
        '（重算 F2I native vs 3098 '
        'sealed）bit 0、d4（3098 npz '
        'sha8 == eeba6c18）、d5（head'
        '(norm(h2)) vs native logits'
        '）bit 0。execution.json 先冻'
        '结。smoke 重要发现：**手工重放'
        '必须与内部同 GEMM shape**——单'
        '位置 GEMV 重放与全序列 GEMM 在 '
        'bf16 下逐位差可达 0.25（cuBLAS'
        ' 累加序随 shape 变化），d1 锚由'
        '此要求全序列重放。\n\n'
        '### 1. 问题与设计\n'
        '3098 判决重写器=末块 MLP 子步'
        '后，打开 MLP 内部：Q1 重写由集'
        '中神经元组还是弥散承载；Q2 风'
        '格条件（ci2）是否跨族共享神经'
        '元集合；Q3 MLP 增量 Δm 是否一'
        '阶桥接 logit TT。每 prompt 1 '
        '次 native 前向 + 手工全序列重'
        '放；Δact=act[pref]−act[base]'
        '（激活空间条件效应）；nf2=跨族 '
        'cos(Δact)；shareK=top-K |Δact| '
        'L1 份额（K∈{1,8,64,256,1024}'
        '）；consensus top-64=组内 8 body '
        'median |Δact| stable argsort；'
        'J64=跨族 top-64 Jaccard；JCROSS='
        '同族跨条件；Δm top-1024 贡献重'
        '构（排序键 |Δact_j|·||W_down'
        '[:,j]||，SHM=||r||/||Δm||、'
        'COSM=cos(r,Δm)）；bridge=cos('
        'head(norm(Δm)), TT_logit)。预注'
        '册门：H_E1（pooled 组中位 '
        'share256 ≥0.5→集中）、H_E2a'
        '（active ci2 跨族 J64 ≥3× 14B '
        'formal AC/BC ci1 对照→风格共享'
        '集）、H_E3（pooled bridge ≥0.8'
        '→一阶桥接）；阶梯 diffuse/'
        'idiosyncratic/nobridge/style_'
        'common。\n\n'
        '### 2. nf2：神经元方向空间重'
        '现 3098 条件反转\n'
        '%s\n'
        '4B ci2 全对 0.80-0.85（风格写'
        '族共同方向），4B ci1 0.56-0.57、'
        'ci3 0.25-0.34；14B ci1 AC/BC '
        '0.13/0.22（formal 写族特异方'
        '向）、14B ci2 0.54-0.64、14B '
        'ci3 0.27-0.43。nf2 结构与 3098 '
        '读出空间 f2dm（ci2 高 0.57-0.99 '
        'vs 14B formal AC/BC 负）同构——'
        '**家族共性写在连续方向上，不写'
        '在离散神经元集合上**。\n\n'
        '### 3. 集中度：中等集中、无巨'
        '神经元\n'
        'share256 组中位：ci1 %.3f-%.3f、'
        'ci2 %.3f-%.3f（最高）、ci3 '
        '%.3f-%.3f；pooled 中位 %.4f '
        '<0.5 → H_E1 False。但均匀基线'
        '仅 256/inter=2.6%%(4B)/1.5%%'
        '(14B)——实际分布远比均匀集中：'
        '几百个中等幅度神经元承载一半份'
        '额，无单点巨神经元。top-1024 贡'
        '献重构：SHM 0.965-0.984、COSM '
        '0.987-0.996——**Δm 几乎被 top-'
        '1024 神经元线性完全重构**。\n\n'
        '### 4. 集合与桥接：双双未过门\n'
        'J64 全表 0.49-0.64 无条件分化'
        '（底座稀释：magnitude 排序被无'
        '条件高基线神经元主导）；active '
        'ci2 组 0.5901 vs 14B formal '
        'AC/BC ci1 对照 0.5647（1.05 倍 '
        '<< 3 倍）→ H_E2a False。bridge '
        '组中位 ci2 %.3f-%.3f（最高）> '
        'ci1 %.3f-%.3f > ci3 %.3f-%.3f'
        '；pooled %.4f <0.8（差 0.017）'
        '→ H_E3 False——风格写入最"直"'
        '，话题写入经读出几何扭曲最多。\n\n'
        '### 5. 判决逻辑\n'
        'active 组（从 3098 sealed F2S '
        '判）：%d/18。H_E1 False 直接短'
        '路 → **fifth_mlp_neuron_'
        'diffuse**。\n\n'
        '### 6. 分析（关键洞察）\n'
        '**（i）重写弥散但可分解**：'
        'top-256 L1 份额 44.6%%（<0.5 门'
        '）但 >>均匀基线 1.5-2.6%%；top-'
        '1024 线性重构 cos>0.99——宽分'
        '布中等幅度增量、几百神经元共同'
        '承载、无专职小组。**（ii）方向'
        '编码而非集合编码**：Jaccard 集'
        '合口径无风格特异（H_E2a False'
        '），但 nf2 方向口径精确重现 '
        '3098 f2dm 条件反转——家族共性'
        '是连续方向性质，不是"哪些神经元'
        '"的离散拓扑性质。**（iii）桥接'
        '条件分层**：ci2 bridge 0.84-'
        '0.94 接近一阶近似成立 > ci1 > '
        'ci3 0.55-0.73——末位 MLP 是读出'
        '预条件器，风格条件下写入方向与'
        '读出需求最对齐。RDC 齿轮候选件'
        '细化：readout-preconditioning '
        'MLP = 宽分布式线性可分解写入 + '
        '方向域条件共性 + 风格条件下近一'
        '阶读出直通。\n\n'
        '### 7. 硬伤与边界\n'
        '- Jaccard 底座稀释：magnitude '
        'top-64 集被无条件基线主导，集合'
        '口径对条件特异神经元不敏感——需'
        '基线校正（减跨条件 median 底座'
        '）后重测集合重叠；\n'
        '- H_E3 差 0.017 未过；且 head('
        'norm(Δm)) 把增量当独立状态过 '
        'RMSNorm，切向分量被放大——严格'
        '一阶检验应做 norm 的 Jacobian '
        '线性化；\n'
        '- share 用 L1 口径、重构用 L2 '
        '口径，二者并存报告；\n'
        '- n=8 body/组；last position 单 '
        'token；causal-connective 单范式'
        '；14B inter 17408 修正记录。\n\n'
        '### 8. 结论与接续\n'
        '接续 3100：(ii) MLP 增量方向的'
        '可预测性——3093 L37/38 注入结构'
        '能否预测 3098/3099 增量（上游-'
        '重写器对接）；基线校正后重测集'
        '合重叠（3099b 候选）；(C) R1 复'
        '用拓扑全景（底册已备）。\n\n'
        '资源消耗：192 前向约 2.5 分钟；'
        '产物 sealed（npz8=%s result8=%s'
        '）。\n'
        % (datetime.now().strftime(
               '%Y-%m-%d %H:%M'),
           nf_tbl,
           sh1[0], sh1[1],
           sh2[0], sh2[1],
           sh3[0], sh3[1],
           G['H_E1_val'],
           br2[0], br2[1],
           br1[0], br1[1],
           br3[0], br3[1],
           G['H_E3_val'],
           n_act,
           seal['npz_sha256_8'],
           seal['result_sha256_8']))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(entry)
    o.append('memo 3099 appended')
else:
    o.append('memo already')

# ---------- audit ----------
aud = io.open(AUDIT,
              encoding='utf-8').read()
if '六十' not in aud:
    add = (
        '\n## 六十、3099 追加（Ω-P97）\n'
        '末位 block MLP 神经元分解'
        '（192 native 前向，4B block35 '
        'inter 9728 / 14B block39 inter '
        '17408——config 实测 17408，早'
        '前 13824 记录有误）：判决 '
        'fifth_mlp_neuron_diffuse。'
        'H_E1 False：top-256 L1 份额 '
        '0.4457 <0.5（但远超均匀基线 '
        '1.5-2.6%，中等集中、无巨神经'
        '元）；H_E2a False：active ci2 '
        '跨族 top-64 Jaccard 0.5901 vs '
        '14B formal AC/BC ci1 对照 '
        '0.5647（1.05 倍 <<3 倍）——无'
        '风格特异共享神经元集合（J64 全'
        '表 0.49-0.64 底座稀释）；H_E3 '
        'False：pooled bridge 0.7826 '
        '<0.8（ci2 最高 0.84-0.94、ci3 '
        '最低 0.55-0.73）。关键：nf2'
        '（Δact 跨族 cos）在神经元方向'
        '空间精确重现 3098 f2dm 条件反'
        '转——4B ci2 0.80-0.85（风格写'
        '族共同方向）vs 14B ci1 AC/BC '
        '0.13-0.22（formal 写族特异方'
        '向）——家族共性编码在连续方向'
        '而非离散神经元集合。top-1024 '
        '贡献重构 COSM 0.987-0.996。方'
        '法论：手工重放必须同 GEMM '
        'shape（GEMV 重放 bf16 逐位差'
        '达 0.25）。锚 d1-d5 全过。\n')
    with io.open(AUDIT, 'a',
                 encoding='utf-8') as f:
        f.write(add)
    o.append('audit 60 appended')
else:
    o.append('audit already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
WLOG_TAG = 'mlp neuron anatomy-closure'
try:
    prev = io.open(wlf,
                   encoding='utf-8').read()
except IOError:
    prev = ''
if WLOG_TAG not in prev:
    line = (
        '- Phase 3099 Omega-P97 mlp '
        'neuron anatomy-closure: '
        'fifth_mlp_neuron_diffuse (no '
        'concentrated set: top-256 L1 '
        'share 0.4457 < 0.5 but >> '
        'uniform 1.5-2.6 pct; no '
        'style-shared J64 set 0.5901 '
        'vs ctrl 0.5647; bridge 0.7826 '
        '< 0.8; nf2 reproduces 3098 '
        'f2dm flip in DIRECTION space - '
        '4B ci2 0.80-0.85 vs 14B ci1 '
        'AC/BC 0.13-0.22; top-1024 '
        'rebuild COSM >0.99).  Anchors '
        'd1-d5 pass (whole-sequence '
        'GEMM replay bit-0; GEMV '
        'replay would differ 0.25); '
        'ledger %d/L14 %d.\n'
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
if 'max=3099' not in mem_cur:
    a1 = ('- max=3098（Ω-P96 '
          'lastblock_mlp_rewrite：重写器'
          '=末块MLP子步）——3099：MLP内'
          '分解；C R1。')
    assert a1 in mem_cur, 'anchor1'
    mem_cur = mem_cur.replace(
        a1,
        '- max=3099（Ω-P97 '
        'mlp_neuron_diffuse：宽增量方向'
        '编码非集合编码）——3100：增量'
        '可预测性；C R1。')
    a3 = ('3096 lens_late → 3097 '
          'cliff → 3098 mlp_substep**。')
    assert a3 in mem_cur, 'anchor3'
    mem_cur = mem_cur.replace(
        a3,
        '3096 lens_late → 3097 '
        'cliff → 3098 mlp_substep → '
        '3099 neuron_diffuse**。')
    c1 = ('5. verify 锚键先 Grep 主脚本 '
          'npz save 段逐一核对（布尔锚无 '
          'DIFF 分量）。')
    assert c1 in mem_cur, 'comp1'
    mem_cur = mem_cur.replace(
        c1,
        '5. verify 锚键先 Grep 主脚本 '
        'npz save 段逐一核对。')
    c2 = ('11. **verify 重算必须复现主'
          '脚本 sorted(units) 枚举序**'
          '（3088 教训）：3077-series 手'
          '工 spearman 用序数秩、不平均'
          '并列值，存在 tie 时 rho 依赖'
          '单元枚举顺序（n=12 两序差 '
          '0.007：0.8951 vs 0.9021）；'
          '主脚本排序键是 dict key 字典'
          '序（\'3B_\'<\'4B_\'<\'DS7B'
          '_\'<\'GLM4_\'）。')
    assert c2 in mem_cur, 'comp2'
    mem_cur = mem_cur.replace(
        c2,
        '11. **verify 重算必须复现主'
        '脚本 sorted-key 枚举序**'
        '（3088：tie 时 rho 依赖单元'
        '序、序数秩不平均并列，两序差 '
        '0.007；序=dict 键字典序）。')
    c3 = ('3. 重跑先删旧 execution.json '
          '与 result.json（脚本自删亦可'
          '）；')
    assert c3 in mem_cur, 'comp3'
    mem_cur = mem_cur.replace(
        c3,
        '3. 重跑先删旧 execution.json '
        '与 result.json；')
    c4 = ('、L38 复制锚（3089：repro 锚'
          '改用 A1 L38 键；repro 键族 '
          'L37→L38 是 REPS 必改项）、'
          '单元序锚（3091：sorted-key 序'
          '下 z86 rho bit 级复现；frozen '
          'spearman 并列顺序敏感——单元'
          '序入 execution.json）。')
    assert c4 in mem_cur, 'comp4'
    mem_cur = mem_cur.replace(
        c4,
        '、L38 复制锚（3089：repro 键'
        '族 L37→L38 是 REPS 必改项）、'
        '单元序锚（3091：sorted-key 序 '
        'z86 rho bit 复现；单元序入 '
        'execution.json）。')
    g1 = ('## 标准锚与精度\n')
    assert g1 in mem_cur, 'g1'
    mem_cur = mem_cur.replace(
        g1,
        '## 标准锚与精度\n'
        '- 3099 GEMM shape 锚：手工重'
        '放必须与内部同 GEMM shape（单'
        '位置 GEMV vs 全序列 GEMM bf16 '
        '逐位差可达 0.25）；d1 型锚要求'
        '全序列重放。\n')
    assert len(mem_cur) < 3000, \
        len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3099')

io.open(OD + r'\closeout_log.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
