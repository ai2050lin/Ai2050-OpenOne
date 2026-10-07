# -*- coding: utf-8 -*-
"""Phase 3070 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3070'
     r'\omega_p67_l34_suppression_anatomy')
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
assert verdict == 'suppression_same_block_attn', \
    verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a4_diff'] == 0.0 and an['a4_ok'] is True
assert an['b8_diff'] == 0.0 and an['b8_ok'] is True
assert an['aref_diff'] == 0.0 and an['aref_ok'] is True
assert an['a2lens_max'] == 0.062412261962890625
assert an['a2lens_ok'] is True
assert an['b0_diff'] == 0.0 and an['b0_ok'] is True
assert an['b1_diff'] == 0.0 and an['b1_ok'] is True
assert an['b3_ok'] is True
assert an['b4_diff'] == 0.0 and an['b4_ok'] is True
assert an['b5_diff'] == 0.0
assert an['b6_diff'] == 0.0 and an['b6_ok'] is True
assert an['b7a_diff'] == 0.0 and an['b7c_diff'] == 0.0
assert an['b7_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_34'] == 0.1487826048372403
assert st['pa34'] == 0.29685845971107483
assert st['pf34'] == 0.2371114194393158
assert st['d34_cproj']['x'] == 0.0
assert abs(st['d34_cproj']['attn']
           - 639.8104587682893) < 1e-9
assert abs(st['d34_cproj']['mlp']
           + 95.91242909117048) < 1e-9
assert abs(st['d34_cproj']['out']
           - 735.8797832320352) < 1e-9
assert abs(st['d35_cproj']['attn']
           + 14.567961614698053) < 1e-9
assert abs(st['d35_cproj']['mlp']
           + 924.1991208657355) < 1e-9
assert abs(st['d35_cproj']['out']
           + 191.78616219035163) < 1e-9
assert abs(st['norm_med']['a34']
           - 260.1191951417096) < 1e-9
assert abs(st['norm_med']['m34']
           - 88.3442006058601) < 1e-9
assert abs(st['norm_med']['a35']
           - 20.99257786087932) < 1e-9
assert abs(st['norm_med']['m35']
           - 175.2872691722519) < 1e-9
assert abs(st['lincheck_34']
           - 0.9999932519350134) < 1e-12
assert abs(st['c_perm']['g1_L34_MLP_ALL']
           - 0.10923343158032986) < 1e-12
assert abs(st['c_perm']['g2_L34_ATN_ALL']
           + 0.4229695021531773) < 1e-12
assert abs(st['c_perm']['g4_L35_ATN_ALL']
           - 0.10411487603305267) < 1e-12
assert abs(st['c_perm']['g6_L34MLP_L35MLP']
           - 0.7354897888565006) < 1e-12
assert abs(st['c_perm']['g7_L34_MLP_RAND']
           - 0.16028886046704904) < 1e-12
assert abs(st['recov']['g1_L34_MLP_ALL']
           + 0.03954917325691043) < 1e-12
assert abs(st['recov']['g2_L34_ATN_ALL']
           + 0.5717521069904176) < 1e-12
assert abs(st['recov']['g4_L35_ATN_ALL']
           + 0.04466772880418762) < 1e-12
assert abs(st['recov']['g6_L34MLP_L35MLP']
           - 0.5867071840192604) < 1e-12
assert abs(st['recov']['g7_L34_MLP_RAND']
           - 0.011506255629808754) < 1e-12
assert res['forwards'] == 231

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3070
           for m in led['measurements']):
    claim = (
        'Omega-P67 (plan 3070 A) - qwen3-4b bf16 '
        'L34 suppression anatomy: where does the '
        'suppressed negative MLP write go? (18.2s, '
        '231 forwards; anchors: a1 row 34 bit-exact '
        'vs 3066 npz diff 0.0; aref PA34/PF34 '
        'bit-exact vs 3069 per-layer authoritative '
        'diff 0.0; a4 PERM(L34,MLP ALL) bit-exact '
        'vs 3069 npz PERM_L[4,2] diff 0.0; b8 '
        'dzX_35=dzP_34 bit 0.0; med_c_34 reference '
        'assert; b0/b1/b4/b5/b6/b7 bit 0.0; b3 '
        'finite). RESULTS: verdict '
        'suppression_same_block_attn. (1) '
        'OBSERVED DECOMPOSITION (inj@34, 24 pairs, '
        'exact linear chain dzX_34=0 -> dzP_34 = '
        'dzA_34 + dzM_34 -> dzX_35 = dzP_34 -> '
        'dzP_35 = dzX_35 + dzA_35 + dzM_35; '
        'lincheck 1.0000): L34 attention diff '
        'writes +640 TT projection, L34 MLP diff '
        'writes -96 (matching 3069), block output '
        'stays +736 POSITIVE; norm a34 260 vs m34 '
        '88 (~3x). The L34 MLP negative write is '
        'dominated by its OWN block attention '
        'positive write, NOT cancelled to zero '
        'and NOT suppressed downstream. (2) '
        'CAUSAL CONFIRMATION: swap-to-base of the '
        'L34 attention output (all 4096 o_proj '
        'input dims) drops c by -0.572 (vs '
        'med_c_34 +0.149, i.e. c -> -0.42); '
        'swapping the L34 MLP moves c only '
        '-0.040 (a4 bit-anchored); L35 attention '
        'swap only -0.045. (3) Both layers MLP '
        'write the SAME negative direction (m34 '
        '-96, m35 -924): the MLP polarity flip '
        'happens one layer before the block-level '
        'readout flip because at L34 the block '
        'attention still writes strongly positive '
        '(L35 attention is near zero, -15). (4) '
        'Random control g7 +0.012 vs g1 -0.040 '
        '(3.4x specificity); removing BOTH MLP '
        'negative writers (L34+L35) raises c by '
        '+0.587 - the two-layer negative side '
        'total. Conclusion: 3069 suppressed-'
        'negative-writer L34 resolved - the '
        'suppressor is the SAME-BLOCK attention '
        'path; layer polarity in this model is a '
        'BLOCK-LEVEL property emerging from '
        'opposite-sign attention/MLP writes, not '
        'a per-component property.')
    meas = {
        'meas_id': 'meas3070_omega_p67_l34_'
                   'suppression_anatomy',
        'phase': 3070,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 row 34 bit 0.0 vs 3066 '
                   'npz (hard); aref PA34/PF34 '
                   'bit 0.0 vs 3069 per-layer '
                   '(hard); a4 PERM(L34,MLP ALL) '
                   'bit 0.0 vs 3069 npz PERM_L[4,'
                   '2] (hard); b8 dzX_35=dzP_34 '
                   'bit 0.0 (hard); med_c_34 '
                   'reference assert; b0/b1/b4/'
                   'b5/b6/b7 bit 0.0; b3 finite',
        'artifacts': {
            'result': 'phase3070/omega_p67_'
                      'l34_suppression_anatomy/'
                      'result.json',
            'npz': 'phase3070/omega_p67_'
                   'l34_suppression_anatomy/'
                   'omega_p67_l34_suppression_'
                   'anatomy.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16; smoke caught '
                '1 design error before the '
                'authoritative run - the 3067-'
                'style a2b calibration only '
                'holds at the L35 readout layer, '
                'replaced by bit-level PA34/PF34 '
                'reference anchors vs 3069)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 209
    l14['connects'].append({
        'meas_id': 'meas3070_omega_p67_l34_'
                   'suppression_anatomy',
        'phase': 3070,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P67: L34 suppression '
                        'anatomy - the suppressed '
                        'negative writer is dominated '
                        'by its OWN block attention: '
                        'observed chain dzP_34 = '
                        'dzA_34(+640) + dzM_34(-96) '
                        '-> block output +736 positive; '
                        'causal L34-attention swap '
                        'drops c -0.572 vs L34-MLP '
                        'swap -0.040 (a4 bit-anchored '
                        'vs 3069); L34 and L35 MLPs '
                        'write the SAME negative '
                        'direction (-96/-924) - the '
                        'MLP polarity flip at L33->34 '
                        'is real but block-level '
                        'polarity stays positive at '
                        'L34 because block attention '
                        'still writes strongly '
                        'positive (L35 attention '
                        'near zero). Layer polarity '
                        '= block-level emergent '
                        'property of opposite-sign '
                        'component writes. Opens '
                        '3071: A attention-head '
                        'decomposition of the L34 '
                        'positive write; B DS7B '
                        'last-layer control; C '
                        'input-displacement lineage; '
                        'D neuron identity; E cross-'
                        'prompt-family generalization'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
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
if '## Phase 3070:' not in memo:
    sec = u'''## Phase 3070: Ω-P67 L34 压制解剖——压制者是同块 attention（suppression_same_block_attn） [%(created)s]

**判决：`suppression_same_block_attn`**（qwen3-4b 单模型 bf16 18.2s，231 次前向；锚全过：**a1：row 34 与 3066 npz diff=0.0；aref：PA34/PF34 中位与 3069 per-layer 权威值 diff=0.0；a4：PERM(L34,MLP ALL) 与 3069 npz PERM_L[4,2] diff=0.0——第三个跨 phase 因果 bit 锚**；b8：dzX_35=dzP_34 bit 0.0；med_c_34 参考断言；b0/b1/b4/b5/b6/b7 全 bit 0.0）。调试史如实入册：smoke 抓住 1 个设计错误——3067 式 a2b 校准断言（|PF−med_c|≤0.01）只在 L35 读出层成立，L34 上权威也会挂（差 0.088）；已替换为 aref（PA34/PF34 vs 3069 bit 锚，更强）。

### 问题与设计
**问题**（3070 A，3069 菜单）：L34 MLP 写负（obs -96）但因果足迹近零（capture 0.054）——负写入去哪了？H2 同块 attention 相消 / H1 下游 L35 attention 压制 / H3 无效应。设计：E1 inj@34×24 对（协议逐字一致）捕获 L34/L35 两层的 x/a/m/p/act；E2 观察差分链分解 dzX_34≈0→dzP_34=dzA_34+dzM_34（线性精确，lincheck 1.0000）→dzX_35=dzP_34（b8）→dzP_35=dzX_35+dzA_35+dzM_35，各分量 TT 投影+范数；E3 七组因果置换（ATTN 换回=o_proj 输入 H 末位 forward-pre-hook 全 4096 维）：g1 L34-MLP-ALL（a4 锚）、g2 L34-ATTN-ALL、g3 L34 双 ALL、g4 L35-ATTN-ALL、g5 L34-MLP+L35-ATTN、g6 L34-MLP+L35-MLP、g7 随机对照（3069 npz S_R_L[4]）。

### 核心结果（重复三遍）
**① 观察分解（一）**：inj@34 下 L34 块内 attention 差分写 **+640**（TT 投影中位），MLP 差分写 **-96**（与 3069 一致），块输出仍 **+736 强正**；范数 a=260 vs m=88（约 3 倍）——**L34 MLP 的负写入被它自己块内的 attention 正写入支配**，不是相消到零，也不是下游压制。**② 因果证实（二）**：换回 L34 attention（全 4096 维 o_proj 输入）使 c 从 +0.149 崩到 **-0.423**（recov=**-0.572**）；换回 L34 MLP 只动 **-0.040**（a4 bit 锚确认=3069）；换回 L35 attention 只动 -0.045——压制者在同块。**③ 同向 MLP（三遍）**：L34 与 L35 的 MLP 写**同一负方向**（m34=-96、m35=-924）——3069 发现的 MLP 极性翻转（L33→34）是真实的组件级事实，但 **L34 的块级极性仍为正，因为该块 attention 仍强正**（L35 attention 近零 -15）；层极性是 attention/MLP 反号写入的**块级涌现属性**，不是逐组件属性。附加：g6（移除两层 MLP 负写者）c 升 +0.587——两层负侧总量；g7 随机对照 +0.012（g1 特异性 3.4 倍）；g3 与 g2 中位相同（g1 效应被 g2 淹没，npz 留原始数组）。

### 机制综合（Ω-P67 拼图）
3069 的"被压制负写者"resolution：**压制者=同块 attention 通路**。层级极性图景更新为两层结构：**组件级极性**（MLP 从 L34 起转负、attention 在 L34 仍强正/L35 近零）与**块级极性**（L34 块=正 because attention≫|MLP|；L35 块=负 because MLP≫attention）。3066 的"正带=分布式下游正化"现在有了组件归属：L34 层级的正贡献主要来自它的 attention 块内写入（+640），不是 MLP。结合 3067/3068（L35 MLP 池）与 3069（幅度分层 L35≫L33），末两层构型完整：**L34=attention 正写主导的过渡块，L35=MLP 负写主导的判决块**。

### 硬伤与边界
- obs cproj 是 fp32 unembed 无 final norm 的声明近似——+640/-96 只做相对比较；recov 经 final_norm 非线性（overshoot 已知）。
- ATTN 换回是"全部 4096 维"——未分解到头级；+640 的头分布（3066 说 heads_dist）未查。
- 单 prompt 族、单模型、单注入层（L34）；g3 与 g2 中位相同（g1 效应淹没），组合效应用 npz 原始数组可查但未单独分析。
- L34 attention 正写入的来源（注入 V 签名经哪些头）未分解——3071 A 的对象。

### 方法论入册
- **a2b 校准断言只对 L35 读出层有效**：PF 探针值与 med_c 的一致性是末层特殊性（final 位置=读出位置），中间层不成立——改用跨 phase 参考值 bit 断言（aref）。
- **ATTN 换回 = o_proj 输入 H 的 forward-pre-hook**（与 act 换回同一 hook 模式，stateATN[li]）；o_proj 输入维= NQ*HDIM=4096。
- **差分链观察分解**：dzP=dzA+dzM 线性精确（+lincheck 验证 act→m），注入层 dzX=0（b4）→块内归因不需要额外前向，纯捕获算术。
- SMOKE 第 6 phase 抓 1 设计错误（a2b 层错配）——若直跑权威将得到 setup_failed_suppression 的伪判决。

### 智能理论洞察（第一性原理）
"层"不是计算的基本单位——**（组件 × 层）才是**：同一个 L34 里 attention 写 +640、MLP 写 -96，块级读出的正负是两者竞争的涌现。这与 3067 的"符号=（神经元集×输入方向）对的属性"同构：**块级符号=（组件写入×相对幅度）对的属性**。语言能力的"条件化齿轮组"纵向图景补全：正负写入在不同组件间分配，深层网络的"层"只是竞争平衡的记账单位；决定读出符号的是末两层（L34 attention+L35 MLP）的联合，而 L35 MLP 是其中唯一池级集中的判决组件（3067/3068）。下一步缝隙：**L34 attention 的 +640 是哪些头写的**（3066 说分布式——头级分解后可与 L35 attention 近零对照，回答"为什么 L34 的 attention 响应注入而 L35 的不响应"）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3070/omega_p67_l34_suppression_anatomy/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3071 菜单**——A（主选）**L34 attention 头级分解**：inj@34 下 32 头逐一（或分组）换回，定位 +640 正写的头集合；与 L35 attention 近零对照；检验 3066 heads_dist 的头级结构。B **DS7B 末层对照**：DS7B 末层块解剖复刻（attention/MLP 分解+头级，跨模型检验"块级涌现极性"图景）。C **输入方向谱系**：dz_A 是否=注入 V 签名的线性像、dz_B 传播路径分解。D **神经元身份**：S_TOP 各层 up/gate 权重结构、W_U 关联。E **跨 prompt 族泛化**：新 prompt 族复测。"好的，继续"即进 3071 A。
''' % {'created': created,
           'script8': seal['script_sha256_8'],
           'result8': seal['result_sha256_8'],
           'npz8': seal['npz_sha256_8'],
           'exec8': seal['exec_sha256_8'],
           'n': len(led['measurements']),
           'l14': len(l14['connects'])}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 三十二、3070 增补' not in aud:
    add = u'''

---

## 三十二、3070 增补：L34 压制解剖——压制者是同块 attention（Omega-P67，判决 suppression_same_block_attn）

1. **观察分解**：inj@34 下 L34 块内 attention 差分写 +640（TT 投影中位）、MLP 差分写 -96（=3069）、块输出 +736 强正；范数 a=260 vs m=88；差分链 dzP_34=dzA_34+dzM_34 线性精确（lincheck 1.0000）、dzX_35=dzP_34 bit 0.0（b8）。
2. **因果证实**：L34 attention 换回（o_proj 输入全 4096 维）使 c +0.149→-0.423（recov -0.572）；L34 MLP 换回仅 -0.040（a4 与 3069 npz bit 互证）；L35 attention 仅 -0.045——压制者在同块，非下游。
3. **组件级 vs 块级极性**：L34/L35 的 MLP 写同一负方向（-96/-924），但 L34 块级极性为正（attention 强正）、L35 为负（attention 近零 -15）——层极性是反号组件写入的块级涌现，不是逐组件属性；3066"正带"的 L34 正贡献归属到 attention 块内写入。
4. HDMCC 修正：a2b 型校准断言只在末层读出位置有效，中间层须用跨 phase 参考 bit 锚（aref）；ATTN 换回 hook（o_proj 输入）与 act 换回同模式；块内归因可用纯捕获算术（差分链+线性验证）完成，零额外前向。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-21.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3070' not in prev:
    line = ('- Phase 3070 Omega-P67 L34 '
            'suppression anatomy (qwen3-4b bf16 '
            'single, 18.2s, 231 forwards): verdict '
            'suppression_same_block_attn. Anchors '
            'bit-exact: a1 row 34 vs 3066, aref '
            'PA34/PF34 vs 3069, a4 PERM(L34,MLP '
            'ALL) vs 3069 npz, b8 dzX_35=dzP_34. '
            'OBSERVED: L34 block attention writes '
            '+640 TT projection vs MLP -96, block '
            'output +736 positive (norms 260 vs '
            '88); CAUSAL: L34-attention swap '
            'drops c -0.572 vs L34-MLP swap '
            '-0.040; L35-attention only -0.045. '
            'L34/L35 MLPs write the SAME negative '
            'direction (-96/-924): layer polarity '
            'is a block-level emergent property '
            'of opposite-sign component writes. '
            'Smoke caught 1 design error (a2b '
            'calibration only valid at the L35 '
            'readout layer -> replaced by aref '
            'bit anchors). Audit 32; ledger '
            '209/L14 177.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W, encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3070' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃/smoke 推翻均如实登记；verdict 单分支赋值。
4. 统计纪律：阈值预注册；重叠算随机期望；capture 设效应量 floor；SMOKE 判决仅供管线验证。

## 标准锚与精度
- 跨脚本 bit 级锚家族：标量行（a1）、全层多行（a1ext）、集合相等（a2）、因果置换（a3/a4）、跨 phase 参考数（l35ctl/aref）、块链恒等（b8：dzX_35=dzP_34）。
- **a2b 型校准断言只在末层读出层有效**；中间层用 aref（跨 phase per-layer 值 bit 断言）。SMOKE 跳过的锚在 setup_ok 中视为通过。
- **3067**：forward-pre-hook 换模块输入；base 自换=恒等免 matched base；权重副本 .detach().float()。
- **3068**：m=W_down·act 线性→partition 恒等分解；ALL=上界锚。
- **3069**：注入层自捕获测层自响应；每层 stateACT[li] dict；recov 参照=同注入条件 med_c_l。
- **3070**：ATTN 换回=o_proj 输入 H 末位 pre-hook（stateATN[li]，4096 维）；差分链块内归因（dzP=dzA+dzM 线性精确，dzX_注入层=0）零额外前向。

## 机制解释审计链（命名前依次检查）
…→3067 神经元级解析→3068 一池两工况→3069 跨层双重分层（极性 L30-33 正/L34-35 负；幅度 L35 206 percent≫L33 30 percent；集中度随效应量）→**3070 L34 压制解剖：压制者=同块 attention（obs +640 vs -96，因果 g2 -0.572 vs g1 -0.040）；L34/L35 MLP 同写负（-96/-924）；层极性=反号组件写入的块级涌现（L34 块=正因 attention 主导、L35 块=负因 MLP 主导）→（组件×层）才是基本单位，层只是竞争平衡的记账**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→脚本自带 run_log 用 Read 读；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3070）
Ω-P2（3011-3070）：…3068 competition_splitpool；3069 cross_layer_distributed；**3070 suppression_same_block_attn（L34=attention 正写主导过渡块、L35=MLP 负写主导判决块；块级极性=组件竞争涌现）**。

## 下一步
- max=3070，下一个 3071（A 主选 **L34 attention 头级分解**：定位 +640 正写的头集合、与 L35 attention 近零对照；B DS7B 末层对照；C 输入方向谱系；D 神经元身份；E 跨 prompt 族泛化）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3070')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
