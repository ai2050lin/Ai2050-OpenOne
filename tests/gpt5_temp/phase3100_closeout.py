# -*- coding: utf-8 -*-
"""Phase 3100 closeout (Omega-P98
upstream-rewriter coupling).  Five
idempotent writes: ledger meas3100 + L14
-> MEMO Phase 3100 -> audit 61 -> wlog
-> MEMORY max=3100 (with net compression
to stay <3000 chars)."""
import io
import json
import os
from datetime import datetime

import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P00 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3100'
       r'\omega_p98_upstream_predict')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
AUDIT = (ROOT + r'\research\gpt5\docs'
         r'\hdmcc_knowledge_map_review_'
         r'20260921.md')
WLOG_DIR = (ROOT + r'\.workbuddy'
            r'\memory')
OD = P00

res = json.load(io.open(
    P00 + r'\result.json',
    encoding='utf-8'))
seal = json.load(io.open(
    P00 + r'\seal.json',
    encoding='utf-8'))
assert res['verdict'] == \
    'sixth_upstream_residual', \
    res['verdict']
assert not res['smoke']
G = res['gates']
MD = res['medians']
AN = res['anchors']
v = res['verdict']

o = []

# ---------- Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
has3100 = any(
    isinstance(m, dict)
    and m.get('phase') == 3100
    for m in led['measurements'])
MID = 'meas3100_omega_p98_upstream_' \
      'predict'
if not has3100:
    claim = (
        'Omega-P98 - upstream-to-'
        'rewriter coupling (192 native-'
        'only forwards; 4B block35 / '
        '14B block39; f64 CPU exact '
        'linear algebra for the '
        'Jacobian statistics).  Q1 '
        'first-order MLP: Delta-'
        'act_pred = silu\'(g_b)*(u_b*'
        'Dg + g_b*Du) with Dg/Du = '
        'bank differences (exact '
        'linear upstream, only silu '
        'first-order), Delta-m_pred = '
        'W_down @ Delta-act_pred - '
        'H_F1 True, pooled median '
        'pred_cos 0.9404 (4B 0.9273 / '
        '14B 0.9533; act_cos 0.96-'
        '0.99; RMSNorm Jacobian '
        'quality xpcos 0.997+ - ln2 '
        'is transparent to '
        'increments).  Q2 channel '
        'split through the RMSNorm '
        'Jacobian (gamma included) '
        'into J@Delta-a vs J@Delta-h0 '
        'energy shares of Delta-'
        'm_pred - H_F2 False: '
        'attention channel only '
        '0.0350 pooled (4B 0.0438 / '
        '14B 0.0238) - the rewriter '
        'input arrives ~96.5 pct '
        'through the h0 RESIDUAL '
        'channel (upstream-layer '
        'propagation), not through '
        'the same-block attention.  '
        'Q3 natural carrier (14B): '
        'L37 per-head increments '
        '(post-o_proj 5120 vector, '
        '40x128 coordinate blocks, '
        '3093 convention) top-8 vs '
        '3093 causal focal TOP8 - '
        'H_F3 False, overlap 1/0/0 '
        '(median 0), Spearman(share, '
        'r1_nh) -0.093/0.257/-0.015 - '
        'causal focal heads are NOT '
        'the largest natural carriers '
        '(caveat: diff vs swap-'
        'injection asymmetry).  NEW '
        'ANCHOR d6: 3100 14B forward '
        '(different script, different '
        'day, output_hidden_states='
        'True) reproduces the 3093 '
        'sealed LG bank BIT-EXACT '
        '(lg_diff 0.0, tt_diff 2.4e-7 '
        '= f32 storage quantization '
        'only) - cross-script '
        'deterministic forward '
        'confirmed.  Anchors d1/d2/d3/'
        'd5 bit 0 both arms; d4 3098 '
        'sha8 eeba6c18 + 3093 sha8 '
        'match seal.')
    meas = {
        'meas_id': MID,
        'phase': 3100,
        'claim': claim,
        'verdict': v,
        'inputs': ['phase3098 npz',
                   'phase3098 seal',
                   'phase3093 npz',
                   'phase3093 seal'],
        'outputs': [P00]}
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
    o.append('ledger meas3100 appended '
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
if '## Phase 3100:' not in memo:
    med_tbl = (
        '| 族 | pred 4B | pred 14B | '
        'act 4B | act 14B | xpcos 4B | '
        'xpcos 14B | ashare 4B | '
        'ashare 14B |\n'
        '|---|---|---|---|---|---|---|'
        '---|---|\n')
    for fa in ('A', 'B', 'C'):
        m = MD[fa]
        med_tbl += (
            '| %s | %.4f | %.4f | %.4f '
            '| %.4f | %.4f | %.4f | '
            '%.4f | %.4f |\n'
            % (fa, m['pred_cos_4b'],
               m['pred_cos_14b'],
               m['act_cos_4b'],
               m['act_cos_14b'],
               m['xpcos_4b'],
               m['xpcos_14b'],
               m['ashare_4b'],
               m['ashare_14b']))
    q3_tbl = (
        '| 族 | top8_nat(14B) | '
        'top8_3093 | overlap | '
        'spearman |\n'
        '|---|---|---|---|---|\n')
    for fa in ('A', 'B', 'C'):
        q3_tbl += (
            '| %s | %s | %s | %d | '
            '%.3f |\n'
            % (fa,
               G['top8_nat_14b'][fa],
               G['top8_3093'][fa],
               G['overlap_14b'][
                   'ABC'.index(fa)],
               G['spearman_14b'][fa]))
    entry = (
        '\n## Phase 3100: Ω-P98 上游-重'
        '写器对接——末块 MLP 增量一阶可预'
        '测（cos 0.94）、输入 96%% 走 h0 '
        '残差通道、L37 因果焦点与自然载'
        '体解耦（sixth_upstream_residual'
        '） [%s]\n\n'
        '**状态**: 已执行（192 前向 '
        'native-only = 32 prompt × 3 族 '
        '× 2 臂；4B block35 / 14B '
        'block39；Jacobian 统计全部 f64 '
        'CPU 精确线性代数，绕开 bf16 '
        'GEMM shape 陷阱）。锚 d1/d2/d3'
        '/d5 bit 0 双臂、d4（3098 '
        'sha8=eeba6c18 + 3093 sha8 '
        'match seal）、**d6 新锚**：'
        '3100 的 14B 前向（不同脚本、不'
        '同天、output_hidden_states='
        'True）与 3093 sealed LG bank '
        '**逐位相同（lg_diff=0.0）**、'
        'TT diff 2.4e-7 = f32 存储量化'
        '级——跨脚本确定性前向 bit 级复'
        '现确认。execution.json 先冻结'
        '。smoke 一次通过。\n\n'
        '### 1. 问题与设计\n'
        '3098/3099 确证重写器=末块 MLP'
        ' 且方向编码；3093 确证 14B L37 '
        '的 8 个 focal head（干预意义'
        '下）携带族重定向。缺口：自然前'
        '向中上游增量如何进入重写器。'
        'Q1 一阶 MLP：Δact_pred = '
        'silu\'(g_b)⊙(u_b⊙Δg + g_b⊙Δu)'
        '（Δg/Δu 取 bank 差=精确线性上'
        '游，仅 silu 一阶近似），Δm_pred '
        '= W_down@Δact_pred；Q2 通道分'
        '解：RMSNorm Jacobian（含 γ）'
        'J@v = γ⊙(v/rms − h(h·v)/(rms³'
        '·d)) 把 Δx 一阶分为 J@Δa 与 '
        'J@Δh0，经 MLP Jacobian 得 '
        'Δm_pred_a/Δm_pred_h0 能量份额'
        '；Q3 自然载体（仅 14B）：L37 '
        'post-o_proj 末位置 5120 向量切'
        '成 40×128 坐标块（3093 HEAD_IDX '
        '约定），ΔBH37 每块 L2 份额 '
        'top-8 vs 3093 TOP8。预注册门：'
        'H_F1（pooled pred_cos ≥0.7）、'
        'H_F2（a-通道能量份额 ≥0.5）、'
        'H_F3（三族 median overlap ≥3/8'
        '）；阶梯 nonlinearity/residual'
        '/mismatch/coupled + '
        'setup_failed 出口。\n\n'
        '### 2. 结果\n'
        '%s\n'
        'pooled: pred_cos %.4f（H_F1 '
        'True）、ashare %.4f（H_F2 '
        'False）。\n\n'
        '### 3. Q3 自然载体 vs 因果焦点\n'
        '%s\n'
        'overlap 1/0/0（median 0 → H_F3 '
        'False）；Spearman(share, r1_nh) '
        '≈ 0。\n\n'
        '### 4. 判决逻辑\n'
        'setup_ok=True（全锚过）。H_F1 '
        'True → H_F2 False 短路 → **'
        'sixth_upstream_residual**（4B '
        '子判决同）。\n\n'
        '### 5. 分析（关键洞察）\n'
        '**（i）末块 MLP 是条件增量的近'
        '一阶放大器**：pred_cos 0.94、'
        'act_cos 0.96-0.99——silu 非线'
        '性在工作点附近弱，Δm 由 Δx 的一'
        '阶 Jacobian 预测到 0.94，残差 '
        '6%% 为高阶项。**（ii）重写器输'
        '入 96.5%% 走 h0 残差通道**：末'
        '块 attention 对条件增量的直接贡'
        '献仅 ~3.5%%（14B 更低 2.4%%）'
        '——上游语义写入（含 3093 L37 '
        'head）经中间层传播汇入 h0(L39) '
        '再进 MLP；同块 attention 不是条'
        '件通道。xpcos 0.997+：RMSNorm '
        '对增量几乎透明，传播中不混合。'
        '**（iii）因果焦点 ≠ 自然载体**'
        '：L37 上 swap 会破坏恢复的 8 个'
        ' head 与自然条件化增量最大的 8 '
        '个 head 几乎不重叠——干预焦点'
        '（必要性）与自然幅度载体（差分'
        '幅度）是两类对象；方向精确的小增'
        '量 head 可以因果关键。\n\n'
        'RDC 更新：条件化齿轮组传动链现'
        '完整到通道级——上游写入 → 残差'
        '流 h0 传播 → 末块 MLP 近一阶重'
        '写（3098/3099）→ 近一阶读出桥接'
        '（bridge 0.78-0.94）。“末块 '
        'attention 直写”路线在 causal-'
        'connective 范式内被否定。\n\n'
        '### 6. 硬伤与边界\n'
        '- Q3 对比对象不同构：ΔBH37 是 '
        'pref−base 全差（含 prefix 引入'
        '的一切变化），3093 swap 是单 '
        'head 输出替换——公平检验需受控'
        '注入下的自然对应实验；\n'
        '- ashare 是一阶框架内的能量分'
        '解，Δm_pred 本身 cos 0.94 近似'
        '（但 xpcos/act_cos 支持近似可信'
        '）；\n'
        '- 末块 attention 通道份额小 ≠ '
        '末块 attention 无作用（仅指条件'
        '增量通道）；\n'
        '- 单末位置；causal-connective '
        '单范式；f64 统计与 bf16 前向的'
        '混合精度口径（锚 bit 级、统计 '
        'f64）。\n\n'
        '### 7. 结论与接续\n'
        '接续 3101：(A) 受控注入自然对应'
        '（per-head swap 后 ΔBH37 变化'
        '量 → 公平载体检验）；(B) h0 通'
        '道回溯（L38/L37 输入增量的逐层'
        '份额分解 → 上游写入层定位）；'
        '(C) R1 复用拓扑全景（底册已'
        '备）。\n\n'
        '资源消耗：192 前向约 4 分钟；产'
        '物 sealed（npz8=%s result8=%s'
        '）。\n'
        % (datetime.now().strftime(
               '%Y-%m-%d %H:%M'),
           med_tbl,
           G['med_pred'],
           G['med_ashare'],
           q3_tbl,
           seal['npz_sha256_8'],
           seal['result_sha256_8']))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(entry)
    o.append('memo 3100 appended')
else:
    o.append('memo already')

# ---------- audit ----------
aud = io.open(AUDIT,
              encoding='utf-8').read()
if '六十一' not in aud:
    add = (
        '\n## 六十一、3100 追加（Ω-P98）\n'
        '上游-重写器对接（192 native 前'
        '向，4B block35 / 14B block39，'
        'f64 CPU 精确 Jacobian 统计）：'
        '判决 sixth_upstream_residual。'
        'H_F1 True：Δm 一阶可预测 '
        'pred_cos 0.9404（4B 0.9273/14B '
        '0.9533；act_cos 0.96-0.99、'
        'xpcos 0.997+）——末块 MLP 是条'
        '件增量近一阶放大器。H_F2 False'
        '：attention 通道仅 3.5%（14B '
        '2.4%）——重写器输入 96.5% 走 '
        'h0 残差通道，同块 attention 非'
        '条件通道。H_F3 False：L37 自然'
        '载体 top8 与 3093 因果 focal '
        'TOP8 重叠 1/0/0、Spearman≈0——'
        '因果焦点≠自然载体（差分 vs 替换'
        '不对称已记录）。d6 新锚：跨脚本'
        '跨天前向 logits bit 级复现 '
        '3093（lg_diff=0.0）。锚 d1-d5 '
        '全过。\n')
    with io.open(AUDIT, 'a',
                 encoding='utf-8') as f:
        f.write(add)
    o.append('audit 61 appended')
else:
    o.append('audit already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
WLOG_TAG = 'upstream-rewriter ' \
           'coupling-closure'
try:
    prev = io.open(wlf,
                   encoding='utf-8').read()
except IOError:
    prev = ''
if WLOG_TAG not in prev:
    line = (
        '- Phase 3100 Omega-P98 '
        'upstream-rewriter '
        'coupling-closure: '
        'sixth_upstream_residual '
        '(H_F1 True pred_cos 0.9404 - '
        'last-block MLP is a near-'
        'first-order amplifier of the '
        'conditional increment; H_F2 '
        'False ashare 0.0350 - '
        'rewriter input arrives 96.5 '
        'pct via the h0 RESIDUAL '
        'channel, same-block attention '
        'is NOT the conditional '
        'channel; H_F3 False overlap '
        '1/0/0 - L37 causal focal '
        'heads are not the natural '
        'carriers; NEW d6 anchor - '
        'cross-script cross-day bit-'
        'exact forward reproduction of '
        '3093 lg_diff 0.0).  Anchors '
        'd1-d6 pass; ledger %d/L14 '
        '%d.\n'
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
if 'max=3100' not in mem_cur:
    a1 = ('- max=3099（Ω-P97 '
          'mlp_neuron_diffuse：宽增量方向'
          '编码非集合编码）——3100：增量'
          '可预测性；C R1。')
    assert a1 in mem_cur, 'anchor1'
    mem_cur = mem_cur.replace(
        a1,
        '- max=3100（Ω-P98 '
        'upstream_residual：重写器输入'
        '96%走h0残差通道、因果焦点≠自然'
        '载体）——3101：C R1。')
    a3 = ('→ 3098 mlp_substep → '
          '3099 neuron_diffuse**。')
    assert a3 in mem_cur, 'anchor3'
    mem_cur = mem_cur.replace(
        a3,
        '→ 3098 mlp_substep → '
        '3099 neuron_diffuse → '
        '3100 upstream_residual**。')
    c1 = ('- 3099 GEMM shape 锚：手工重'
          '放必须与内部同 GEMM shape（单'
          '位置 GEMV vs 全序列 GEMM bf16 '
          '逐位差可达 0.25）；d1 型锚要求'
          '全序列重放。')
    assert c1 in mem_cur, 'comp1'
    mem_cur = mem_cur.replace(
        c1,
        '- 3099 GEMM 锚：手工重放须同'
        '内部 GEMM shape（GEMV vs 全序'
        '列 bf16 差达 0.25）。\n'
        '- 3100 复现锚：跨脚本跨天同模'
        '型同文本前向 logits bit 0'
        '（3100 vs 3093 lg_diff=0.0）；'
        'output_hidden_states 不改 '
        'logits。')
    c2 = ('glm4-9b-chat-hf（Glm4 40L '
          '32H 2kv 4096 vocab 151552 '
          'tied=False bf16 sysmem '
          'fallback；repro 锚 bit 级过）')
    assert c2 in mem_cur, 'comp2'
    mem_cur = mem_cur.replace(
        c2,
        'glm4-9b-chat-hf（Glm4 40L '
        '4096 vocab 151552 untied '
        'bf16 sysmem fallback；repro '
        'bit 级过）')
    c3 = ('- 脚本 tests\\glm5\\phase{N}_'
          '*.py；closeout/verify '
          'tests\\gpt5_temp\\；产物 ...'
          '\\phase{N}\\{arm}\\（smoke 在 '
          'smoke\\ 子目录）。')
    assert c3 in mem_cur, 'comp3'
    mem_cur = mem_cur.replace(
        c3,
        '- 脚本 tests\\glm5\\phase{N}_'
        '*.py；closeout tests\\gpt5_'
        'temp\\；产物 ...\\phase{N}\\'
        '{arm}\\（smoke 在 smoke\\）。')
    c4 = ('10. **小源脚本可直写**（3088 '
          '教训）：源 ≤600 行且有全文时，'
          '直接 Write 新脚本 + 静态同一性'
          '断言（frozen 统计块逐字节比对）'
          '+ bad/want 清单，比生成器快且'
          '等价可靠；WANT 计数须先数真实'
          '文件（PREREG 跨字符串段拆分会'
          '少计字面 token）。')
    assert c4 in mem_cur, 'comp4'
    mem_cur = mem_cur.replace(
        c4,
        '10. **小源脚本可直写**（3088）：'
        '源 ≤600 行有全文时直接 Write + '
        '静态同一性断言 + bad/want 清单；'
        'WANT 计数先数真实文件（PREREG '
        '跨段拆分少计 token）。')
    assert len(mem_cur) < 3000, \
        len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3100')

io.open(OD + r'\closeout_log.txt', 'w',
        encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
