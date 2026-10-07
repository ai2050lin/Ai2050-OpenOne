# -*- coding: utf-8 -*-
"""Phase 3071 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3071'
     r'\omega_p68_attn_head_decomp')
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
assert verdict == 'attn_heads_focal', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a4_diff'] == 0.0 and an['a4_ok'] is True
assert an['a5_diff'] == 0.0 and an['a5_ok'] is True
assert an['a6_diff'] == 0.0 and an['a6_ok'] is True
assert an['a7_diff'] == 0.0 and an['a7_ok'] is True
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
assert abs(st['c_perm']['gA_L34_ATN_ALL']
           + 0.4229695021531773) < 1e-12
assert abs(st['c_perm']['gB_L35_ATN_ALL']
           - 0.10411487603305267) < 1e-12
assert abs(st['recov']['g1_L34_MLP_ALL']
           + 0.03954917325691043) < 1e-12
assert abs(st['recov']['gA_L34_ATN_ALL']
           + 0.5717521069904176) < 1e-12
assert abs(st['recov']['gB_L35_ATN_ALL']
           + 0.04466772880418762) < 1e-12
hd = st['head']
assert hd['n_neg'] == 16
assert abs(hd['capture8']
           - 0.8355652268128613) < 1e-12
assert hd['top8'] == [20, 7, 1, 14, 26,
                      0, 2, 24]
assert abs(hd['r34'][20]
           + 0.22156993894701427) < 1e-12
assert abs(hd['r34'][7]
           + 0.21310123529223413) < 1e-12
assert abs(hd['r34'][1]
           + 0.20872026764068408) < 1e-12
assert abs(hd['r34'][14]
           + 0.16739848114763328) < 1e-12
assert abs(hd['r34'][26]
           + 0.1574770002625661) < 1e-12
assert abs(hd['l35_max']
           - 0.04555673004619386) < 1e-12
assert hd['l35_flat'] is True
assert abs(hd['rank_corr']
           + 0.23313782991202345) < 1e-12
assert abs(hd['headlin34_med']
           - 0.0018137704767155584) < 1e-12
assert abs(hd['headsum34']
           - 1056.7015853447947) < 1e-9
assert abs(hd['r_quarters'][0]
           + 0.3343023417835643) < 1e-12
assert abs(hd['r_quarters'][1]
           - 0.450906761459989) < 1e-12
assert abs(hd['r_quarters'][2]
           + 0.10244623290529631) < 1e-12
assert abs(hd['r_quarters'][3]
           + 0.22896925428307716) < 1e-12
assert hd['has_o_proj_bias'] is False
assert res['forwards'] == 1767

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3071
           for m in led['measurements']):
    claim = (
        'Omega-P68 (plan 3071 A) - qwen3-4b bf16 '
        'L34 attention head decomposition: WHICH '
        'heads carry the +640 positive write? '
        '(108.5s, 1767 forwards; anchors: a1 row '
        '34 bit-exact vs 3066 npz; a7 E2 cproj '
        'medians bit-exact vs 3070 refs; a4 '
        'PERM(L34,MLP ALL) bit-exact vs 3069 npz '
        'PERM_L[4,2]; a5 gA(L34_ATN_ALL) median '
        'bit-exact vs 3070 c_perm g2 -0.4229695; '
        'a6 gB(L35_ATN_ALL) median bit-exact vs '
        '3070 c_perm g4 0.1041149; aref PA34/'
        'PF34 bit-exact vs 3069; b8 dzX_35=dzP_'
        '34 bit 0.0; b0/b1/b4/b5/b6/b7 bit 0.0). '
        'RESULTS: verdict attn_heads_focal '
        '(capture8 0.836 >= 0.6 threshold, '
        'preregistered). (1) PER-HEAD CAUSAL '
        'SWEEPS (32 heads x 24 pairs, o_proj-'
        'input head-slice swap-to-base at the '
        'last position): 16 heads have negative '
        'recov (their inj value carries a TT-'
        'positive write); the top-8 (h20 -0.222, '
        'h7 -0.213, h1 -0.209, h14 -0.167, h26 '
        '-0.157, h0 -0.098, h2 -0.092, h24 '
        '-0.060) capture 83.6 percent of the '
        'total negative recov mass - the full-'
        'swap recovery -0.572 is carried by a '
        'FOCAL head set, not head-distributed '
        '(3066 heads_dist was an observation-'
        'level claim; causally the effect '
        'concentrates). (2) L35 CONTRAST FLAT: '
        'all 32 L35 head swaps max |recov| '
        '0.046 < 0.05 - the L35 attention block '
        'carries no causal head-level footprint '
        '(consistent with its full-swap -0.045): '
        'responding to the injection is not a '
        'generic property of attention layers '
        'but a specialization of specific heads '
        'at L34. (3) OBS-CAUSAL DECOUPLING: '
        'per-head obs TT projection vs causal '
        'recov Spearman -0.233 (weak): h20 obs '
        'only -14.5 but the strongest causal '
        'head; h2 obs -76.6 yet causal -0.092; '
        'per-pair o_proj linearity is exact '
        '(headlin rel 1.8e-3, no o_proj bias) '
        'so the mismatch is downstream '
        'transmission (L34 MLP + L35 '
        'nonlinearity), not measurement error. '
        '(4) Five heads have POSITIVE recov '
        '(h12 +0.115, h5 +0.090, h13 +0.072, '
        'h25 +0.053, h4 +0.052) - the L34 '
        'attention block itself contains '
        'competing TT-negative writers. '
        'Quarters recov [-0.334, +0.451, -0.102, '
        '-0.229]: q1 (heads 8-15) is net TT-'
        'negative as a group, and per-head '
        'medians are non-additive (top-5 median '
        'sum -0.968 vs full swap -0.572) - head '
        'interactions matter. Conclusion: the '
        'L34 same-block attention suppressor '
        'resolves to a focal 5-head core (h20/7/'
        '1/14/26) plus a long tail; head-level '
        'causal effect = (direct TT-aligned '
        'write) x (downstream transmission), '
        'decoupled from the raw write '
        'projection.')
    meas = {
        'meas_id': 'meas3071_omega_p68_attn_'
                   'head_decomp',
        'phase': 3071,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 row 34 bit 0.0 vs 3066 '
                   'npz (hard); a7 E2 cproj '
                   'medians bit 0.0 vs 3070 '
                   'refs (hard); a4 PERM(L34,'
                   'MLP ALL) bit 0.0 vs 3069 '
                   'npz PERM_L[4,2] (hard); '
                   'a5 gA median bit 0.0 vs '
                   '3070 c_perm g2 (hard; '
                   'validates the head-slice '
                   'hook machinery); a6 gB '
                   'median bit 0.0 vs 3070 '
                   'c_perm g4 (hard); aref '
                   'PA34/PF34 bit 0.0 vs 3069 '
                   '(hard); b8 dzX_35=dzP_34 '
                   'bit 0.0 (hard); med_c_34 '
                   'reference assert; b0/b1/'
                   'b4/b5/b6/b7 bit 0.0; b3 '
                   'finite',
        'artifacts': {
            'result': 'phase3071/omega_p68_'
                      'attn_head_decomp/'
                      'result.json',
            'npz': 'phase3071/omega_p68_'
                   'attn_head_decomp/'
                   'omega_p68_attn_head_'
                   'decomp.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16; smoke passed '
                'clean but with a hidden median-'
                'axis bug that only crashed in '
                'the authoritative run (head 26 '
                'entered top8): np.median over '
                'the wrong axis gave per-pair '
                'instead of per-head medians - '
                'smoke NH_USE=8 masked it '
                'because both axes had size 8; '
                'fixed axis=0 and rerun',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 210
    l14['connects'].append({
        'meas_id': 'meas3071_omega_p68_attn_'
                   'head_decomp',
        'phase': 3071,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P68: L34 attention '
                        'head decomposition - the '
                        'same-block attention '
                        'suppressor resolves to a '
                        'FOCAL head set: top-8 '
                        'heads (h20/7/1/14/26/0/2/'
                        '24) capture 83.6 percent '
                        'of the total negative '
                        'recov; L35 contrast flat '
                        '(max 0.046) - responding '
                        'to the injection is a '
                        'head-level specialization '
                        'at L34, not a generic '
                        'layer property. 3066 '
                        'heads_dist overturned at '
                        'the causal level (it was '
                        'observation-level). '
                        'Obs-causal decoupling '
                        '(Spearman -0.233): causal '
                        'head effect = direct '
                        'write x downstream '
                        'transmission. Five '
                        'competing TT-negative '
                        'heads inside the same '
                        'block (h12/5/13/25/4). '
                        'Opens 3072: A focal-head '
                        'deep anatomy (attend '
                        'positions + V lineage + '
                        'OV circuit); B head '
                        'interaction matrix; C '
                        'cross-prompt-family '
                        'stability of the focal '
                        'set; D DS7B head-level '
                        'control; E neuron '
                        'identity'})
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
if '## Phase 3071:' not in memo:
    sec = u'''## Phase 3071: Ω-P68 L34 attention 头级分解——+640 正写集中于焦点头集合（attn_heads_focal，capture8 0.836） [%(created)s]

**判决：`attn_heads_focal`**（qwen3-4b 单模型 bf16 108.5s，1767 次前向；六重跨 phase bit 锚全过：**a1：row 34 与 3066 npz diff=0.0；a7：E2 cproj 六个中位与 3070 权威值 diff=0.0；a4：PERM(L34,MLP ALL) 与 3069 npz PERM_L[4,2] diff=0.0；a5：全头换回 gA 中位与 3070 c_perm g2 diff=0.0——头级 hook 机制与块级机制 bit 互证；a6：gB 与 3070 c_perm g4 diff=0.0**；b8 bit 0.0；aref bit 0.0；b0/b1/b4/b5/b6/b7 全 bit 0.0）。调试史如实入册：smoke 全绿但暗藏 1 个 median 轴错误（np.median 用了 axis=1，得到每对跨头中位而非每头跨对中位）——smoke 里 NH_USE=8 恰好两轴同尺寸、top8 全落在 0-7，侥幸通过；权威运行 head 26 进入 top8 即 IndexError 崩溃；修复 axis=0 重跑。教训：**smoke 的维度裁剪会把两类数组尺寸巧合对齐，轴错误在 smoke 里不可见**。

### 问题与设计
**问题**（3071 A，3070 菜单）：L34 同块 attention 写 +640（obs）、全换回 recov -0.572（因果）——是哪些头写的？3066 说 heads_dist（观察级）。设计：E1 inj@34×24 对（协议逐字一致）+ 捕获 zH（o_proj 输入 4096 维）于 L34/L35；E2 复刻 3070 差分链（a7 锚绑死 3070 值）；E2H 头级观察分解——dzH 切成 32 头×128 维切片，out_h = dzH_h @ W_O[:, h*128:(h+1)*128].T（o_proj 线性无 bias，diff 中 bias 相消；headlin rel med 1.8e-3 证实 per-pair 线性精确）；E3 因果头级扫描——per-head 换回 = 同一个 hook_pre_last，mask 换成头切片索引（零新 hook 代码）：32 头×24 对 @L34 + 32 头×24 对 @L35 对照 + 四分 组（8 头/组）×24 对。

### 核心结果（重复三遍）
**① 因果焦点化（一）**：32 头中 16 头 recov 为负（其注入值携带 TT 正写入），**top-8 头 [h20, h7, h1, h14, h26, h0, h2, h24] 捕获全部负 recov 质量的 83.6 percent**（capture8=0.836 ≥ 预注册 0.6 阈值），top-5 主力（h20 -0.222、h7 -0.213、h1 -0.209、h14 -0.167、h26 -0.157）每个都达全换回效应（-0.572）的三成上下——**+640 的因果承载是焦点头集合，3066 的 heads_dist 在因果层面被推翻**（它是观察级说法：响应头广泛，但因果权重集中）。**② L35 对照平坦（二）**：L35 全部 32 头换回 max|recov|=0.046 < 0.05（l35_flat=True），与 L35 全换回 -0.045 一致——**"响应注入"不是 attention 层的普遍属性，而是 L34 特定头集合的特化**；这直接回答了 3070 留下的问题："为什么 L34 的 attention 响应而 L35 的不响应"——因为承载响应的头只在 L34。**③ 观察-因果脱钩（三遍）**：per-head obs TT 投影与因果 recov 的 Spearman 仅 **-0.233**（弱）：h20 的 obs 只有 -14.5 却是最强因果头；h2 obs -76.6（负！）因果却 -0.092；per-pair 线性精确（headlin 1.8e-3）排除测量误差——脱钩来自**下游传输**（同块 MLP + L35 非线性）：头级因果效应 =（直接 TT 对齐写入）×（下游传输系数），不是头单独的属性。附加：L34 attention 内部也有 5 个 TT 负写入头（h12 +0.115、h5 +0.090、h13 +0.072、h25 +0.053、h4 +0.052——换回它们 c 反而升）；四分组 recov [-0.334, +0.451, -0.102, -0.229]（q1=头 8-15 整组净 TT 负）；per-head 中位不可加（top-5 中位和 -0.968 vs 全换回 -0.572）——头间交互真实存在。

### 机制综合（Ω-P68 拼图）
L34 attention 的压制者从"整个块"细化到 **5 头核心（h20/7/1/14/26）+ 长尾**。三层同构完成：3067（符号=（神经元集×输入方向）对的属性）→ 3070（块级符号=（组件写入×相对幅度）对的属性）→ **3071（头级效应=（直接写入×下游传输）对的属性）**。3066 heads_dist 的修正：观察分布宽（32 头都有 dzH 响应，obs top [h6 +295, h24 +183, h23 +148]）但因果集中（capture8 0.836）——"谁在写"和"谁的写到达读出"是两个不同的问题，中间隔着非线性传输链。L35 attention 全头平坦 + L34 焦点头特化 → 条件化机制在头级就有分工，这是"条件化齿轮组"的头级分辨率图景。

### 硬伤与边界
- 单 prompt 族、单模型、单注入层（L34）；焦点头集合的跨 prompt 族稳定性未测（3072 C）。
- per-head 因果中位不可加 + 头间交互真实（quarters 证据）；成对交互矩阵未做（3072 B）。
- obs cproj 是 fp32 无 final norm 的声明近似；headlin L35 rel 1.0e-2 偏大（dzA35 范数小 ~21，bf16 舍入相对误差放大）——L35 头级 obs 只做定性参考。
- 头索引 = o_proj 输入切片顺序（Q-head concat 顺序），GQA 下 swap 定义良定；但"头"与 attention 内部 Q/K/V head 的对应是索引级的，未查 W_Q/W_K 行身份。
- 头级 recov 是"单头换回"边际效应，非线性系统中的边际≠份额。

### 方法论入册
- **per-head 换回零新代码**：mask = 头切片索引 arange(h*128,(h+1)*128)，复用 3070 的 hook_pre_last。
- **a5/a6 hook 互证锚**：全头换回（32 切片同时）必须 bit 等于 3070 的块级 g2/g4——这是头级机制正确性的最强验证，比"换个方法重算"强。
- **median 轴陷阱**：smoke 维度裁剪使 (8,32)/(32,8) 两轴同尺寸，轴错误不可见——轴选择应让"语义轴"（头）的尺寸与 smoke 裁剪维度解耦，或 smoke 断言数组形状。
- 前向预算：1767 前向（32 银行+4 重捕+1 sham+24 E1+2 b7+3×24 组锚+768×2 头扫+96 四分组）108.5s——头级扫描的成本公式：NH×NP×2 层。

### 智能理论洞察（第一性原理）
"条件化齿轮组"现在有了**头级分辨率**：L34 对注入 V 签名的应答由 5 个主力头承载（h20/7/1/14/26），它们是这台机器的"应答器齿轮"；但齿轮的输出要经过下游非线性传输才变成读出效应（obs-causal 脱钩，rank_corr -0.233）——**单独一个头的"写入方向"不决定它的功能地位，（写入×传输）才决定**。这与大脑皮层的图景同构：单个神经元的投射不等于它的功能贡献，功能由（投射×下游回路增益）决定。语言能力的极小机制单元在三级分辨率上都是"关系对"而非"单个零件"：神经元级（神经元集×输入方向）、组件级（组件写入×相对幅度）、头级（直接写入×传输链）。下一步缝隙：**焦点头的输入端**——h20/7/1/14/26 在注入后 attend 到哪些位置、它们的 dzH 是否=注入 V 签名经 W_V 的线性像、OV 电路（W_O 切片×W_U）直接读哪个词——把"应答器齿轮"的输入-输出两端都打开。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3071/omega_p68_attn_head_decomp/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3072 菜单**——A（主选）**焦点头深解剖**：对 top-5 头（h20/7/1/14/26）做（i）attention 位置分布（注入后 attend 哪些位置）、（ii）V 签名谱系（dzH_h 与注入 V 经 W_V/W_O 线性像的余弦）、（iii）OV 电路方向（W_O 切片×W_U：头的输出直接读哪个词表方向）——把应答器齿轮的输入-输出两端打开。B **头间交互矩阵**：top-8 头成对联合换回，分解交互项。C **焦点头集合跨 prompt 族稳定性**：新 prompt 族复测头级 recov 谱。D **DS7B 头级对照**：跨模型检验焦点化图景。E **神经元身份**：S_TOP 的 up/gate 权重结构。"好的，继续"即进 3072 A。
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
if '## 三十三、3071 增补' not in aud:
    add = u'''

---

## 三十三、3071 增补：L34 attention 头级分解——因果焦点化（Omega-P68，判决 attn_heads_focal）

1. **因果焦点化**：32 头逐一换回（o_proj 输入头切片，24 对/头），16 头 recov 为负，top-8（h20/7/1/14/26/0/2/24）捕获全部负 recov 质量量 83.6 percent——3066 的 heads_dist 在因果层面被推翻（观察级"响应广泛"与因果级"效应集中"不矛盾：obs top 是 h6 +295/h24 +183，因果 top 是 h20/7/1）。
2. **L35 对照平坦**：L35 全部 32 头 max|recov| 0.046——"响应注入"是 L34 特定头集合的特化，不是 attention 层的普遍属性；回答了 3070 的"为什么 L34 响应而 L35 不响应"。
3. **观察-因果脱钩**：per-head obs TT 投影与因果 recov 的 Spearman 仅 -0.233；per-pair o_proj 线性精确（headlin 1.8e-3，无 bias）排除测量误差——脱钩=下游传输（同块 MLP+L35 非线性）；头级因果效应=（直接写入×下游传输）对的属性，第三层同构（3067 神经元级→3070 组件级→3071 头级）。
4. HDMCC 修正：per-head 换回零新代码（mask=头切片索引）；a5/a6 全头换回 bit 锚=头级机制与块级机制互证的最强验证；**median 轴陷阱**——smoke 维度裁剪可使两轴尺寸巧合对齐、轴错误不可见，权威运行才崩溃（head 26 进入 top8）；头级 recov 是非线性系统的边际效应，不可加（top-5 中位和 -0.968 vs 全换回 -0.572）。
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
if 'Phase 3071' not in prev:
    line = ('- Phase 3071 Omega-P68 L34 attention '
            'head decomposition (qwen3-4b bf16 '
            'single, 108.5s, 1767 forwards): '
            'verdict attn_heads_focal (capture8 '
            '0.836). Six cross-phase bit anchors '
            'exact: a1 vs 3066, a7 E2 medians vs '
            '3070, a4 vs 3069 PERM_L[4,2], a5 '
            'gA vs 3070 g2 (head-slice hook '
            'machinery validated), a6 gB vs 3070 '
            'g4, b8. Top-8 heads (h20/7/1/14/26/'
            '0/2/24) capture 83.6 percent of '
            'negative recov mass; L35 contrast '
            'flat (max 0.046); obs-causal '
            'Spearman -0.233 (causal head '
            'effect = direct write x downstream '
            'transmission); 5 competing TT-'
            'negative heads inside L34 attention '
            '(h12/5/13/25/4); quarters show '
            'non-additivity (q1 +0.451). 3066 '
            'heads_dist overturned at the causal '
            'level. Smoke passed with a hidden '
            'median-axis bug (authoritative '
            'crashed at head 26; fixed axis=0). '
            'Audit 33; ledger 210/L14 178.\n')
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
if 'max=3071' not in mem_cur:
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
- 跨脚本 bit 级锚家族：标量行（a1）、全层多行（a1ext）、集合相等（a2）、因果置换（a3/a4）、跨 phase 参考数（l35ctl/aref）、块链恒等（b8）、**hook 互证（a5/a6：全头换回 bit=3070 块级 g2/g4）**。
- **a2b 型校准断言只在末层读出层有效**；中间层用 aref。SMOKE 跳过的锚在 setup_ok 中视为通过。
- **3067**：forward-pre-hook 换模块输入；base 自换=恒等。**3068**：m=W_down·act partition 恒等。**3069**：每层 stateACT[li] dict。**3070**：ATTN 换回=o_proj 输入 H 末位 pre-hook（stateATN[li]）；差分链块内归因零额外前向。**3071**：per-head 换回=mask 用头切片索引（零新 hook）；o_proj 线性无 bias→per-pair 头级分解精确（headlin）。

## 机制解释审计链（命名前依次检查）
…→3070 L34 压制解剖（压制者=同块 attention；块级极性=组件竞争涌现）→**3071 头级分解：+640 因果集中于焦点头集合（top-8 capture8 0.836；5 主力 h20/7/1/14/26）；L35 全头平坦（max 0.046）——响应注入是头级特化非层属性；obs-causal 脱钩（Spearman -0.233）：头级因果效应=（直接写入×下游传输）对的属性；3066 heads_dist 因果层面被推翻**。三层同构：神经元级（神经元集×输入方向）→组件级（写入×相对幅度）→头级（直接写入×传输链）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→脚本自带 run_log 用 Read 读；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查。
- **median 轴陷阱（3071）**：smoke 维度裁剪可使 (8,32)/(32,8) 两轴同尺寸、轴错误不可见，权威才崩——语义轴尺寸应与 smoke 裁剪解耦或断言形状。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3071）
Ω-P2（3011-3071）：…3069 cross_layer_distributed；3070 suppression_same_block_attn；**3071 attn_heads_focal（L34 attention 压制者=5 头核心 h20/7/1/14/26+长尾；L35 attention 无头级足迹；头级效应=写入×传输）**。

## 下一步
- max=3071，下一个 3072（A 主选 **焦点头深解剖**：top-5 头的 attend 位置分布+V 签名谱系+OV 电路方向；B 头间交互矩阵；C 焦点头跨 prompt 族稳定性；D DS7B 头级对照；E 神经元身份）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3071')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
