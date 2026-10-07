# -*- coding: utf-8 -*-
"""Phase 3073 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3073'
     r'\omega_p70_head_interaction')
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
assert verdict == 'higher_order_required', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
for nm in ('a1', 'a8', 'a9', 'a10', 'b8', 'b0',
           'b1', 'b4', 'b6'):
    assert an[nm + '_diff'] == 0.0 \
        and an[nm + '_ok'] is True, nm
assert an['aref_diff'] == 0.0 and an['aref_ok'] is True
assert an['a2lens_max'] == 0.062412261962890625
assert an['a2lens_ok'] is True
assert an['b3_ok'] is True
assert an['b5_diff'] == 0.0
assert an['b7a_diff'] == 0.0 and an['b7c_diff'] == 0.0
assert an['b7_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_34'] == 0.1487826048372403
assert st['pa34'] == 0.29685845971107483
assert st['pf34'] == 0.2371114194393158
assert st['hl34_med'] == 0.0018137704767155584
assert st['top8'] == [20, 7, 1, 14, 26, 0, 2, 24]
assert st['r1'][0] == -0.22156993894701427
assert st['r1'][1] == -0.21310123529223413
assert st['r1'][2] == -0.20872026764068408
assert st['r1'][7] == -0.060069811227881575
assert st['i_pair_max'] == 0.10387255949605928
assert st['i_argmax_pair'] == [2, 3]
assert st['n_ipos'] == 24
assert st['n_ineg'] == 4
assert st['r_u8'] == -0.5322981028358011
assert st['r_u8_pred'] == -0.21182383293868212
assert st['pred_err'] == 0.320474269897119
assert st['t3'] == [0, 1, 2]
assert st['r_t3'] == -0.37527143927508755
assert st['r_t3_pred'] == -0.42269733769576967
assert st['err3'] == 0.04742589842068212
assert st['inter_significant'] is True
assert abs(st['sp_I_r1prod']
           - 0.8215654077723045) < 1e-12
assert abs(st['sp_I_dahprod']
           + 0.6464148877941983) < 1e-12
assert res['forwards'] == 975

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3073
           for m in led['measurements']):
    claim = (
        'Omega-P70 (plan 3073 A) - qwen3-4b '
        'bf16 head interaction matrix: WHERE '
        'does the 3071 non-additivity (top-5 '
        'sum -0.968 vs full swap -0.572) come '
        'from?  (61.2s, 975 forwards; anchors: '
        'a1 row 34 bit-exact vs 3066 npz; a8 '
        'ZH34/ZH35 bit-exact vs 3071 npz; a9 '
        'DAH medians bit-exact vs 3071 npz; '
        'a10 single-head resweep bit-exact vs '
        '3071 r34[top8] - cross-phase causal '
        'reproducibility perfect; aref bit vs '
        '3069; b8/b0/b1/b4/b6/b7 bit 0.0). '
        'RESULTS: verdict higher_order_'
        'required. (1) STRONG SUB-ADDITIVITY: '
        'of 28 pairwise joint swaps of the '
        'focal top-8, 24 interactions positive '
        '(median bookkeeping I(AB) = R(AB) - '
        'R(A) - R(B)); max +0.104 (h01+h14). '
        'Example: h20 alone -0.222, h07 alone '
        '-0.213, sum -0.435, joint only '
        '-0.195 - two heads together behave '
        'like roughly ONE head.  (2) SECOND-'
        'ORDER EXPANSION FAILS: sum singles '
        '-1.219 + sum pairwise I +1.007 '
        'predicts -0.212 but the measured '
        'top-8 joint is -0.532 (PRED_ERR '
        '0.320, gate 0.05) - pairwise '
        'interactions are themselves not '
        'additive; saturation is a collective '
        'property.  Triplet (h20/7/1): pred '
        '-0.423 vs measured -0.375, ERR3 '
        '0.047 > gate 0.02 - close but over.  '
        'Non-monotonicity observed: adding '
        'h07 to the best pair (h20+h01 -0.389) '
        'makes the triplet SHALLOWER (-0.375). '
        '(3) INTERACTION MAGNITUDE IS '
        'MAGNITUDE-DRIVEN: spearman(|r_a|*|r_'
        'b|, |I|) = 0.822 - interaction size '
        'is set by the two heads effect '
        'sizes, NOT by position structure '
        '(GQA-group/quarter means 0.033 vs '
        '0.037, no difference).  (4) TOP-8 '
        'JOINT = -0.532 vs full 32-head swap '
        '-0.572: the focal set carries 93 '
        'percent of the full recovery; the '
        '24-head tail adds only 0.04.  (5) '
        'Weak-head pairs flip: h02+h24 joint '
        '+0.018, h00+h24 -0.011 - near-zero '
        'heads are not consistent sign.  '
        'Conclusion: the focal heads write a '
        'SHARED TT channel with capacity-'
        'limited (saturating) joint effect; '
        'the head-level causal effect is a '
        'SET function with strong diminishing '
        'returns, not a sum of per-head '
        'terms.  The saturation curve R(S) '
        'over swap-set size (1/2/3/8/32 heads '
        'data now available) is the next '
        'quantitative target.')
    meas = {
        'meas_id': 'meas3073_omega_p70_head_'
                   'interaction',
        'phase': 3073,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 row 34 bit 0.0 vs 3066 '
                   'npz (hard); a8 ZH34/ZH35 '
                   'bit 0.0 vs 3071 npz (hard); '
                   'a9 DAH34/35 medians bit 0.0 '
                   'vs 3071 npz (hard); a10 '
                   'single-head recov medians '
                   'bit 0.0 vs 3071 result '
                   'r34[top8] (hard; cross-'
                   'phase causal reproducibility)'
                   '; aref PA34/PF34 bit 0.0 vs '
                   '3069 (hard); b8 dzX_35='
                   'dzP_34 bit 0.0 (hard); '
                   'med_c_34 reference assert; '
                   'b0/b1/b4/b6/b7 bit 0.0; b3 '
                   'finite',
        'artifacts': {
            'result': 'phase3073/omega_p70_'
                      'head_interaction/'
                      'result.json',
            'npz': 'phase3073/omega_p70_'
                   'head_interaction/'
                   'omega_p70_head_'
                   'interaction.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16).  Design '
                'history: smoke exposed that '
                'sample-level interaction terms '
                'summed over 28 pairs amplify '
                'cos noise ~sqrt(28) and predict '
                'garbage (pred flipped positive); '
                'pre-authoritative redesign to '
                'MEDIAN-SCALE bookkeeping (I(AB) '
                '= R(AB)-R(A)-R(B) on 24-pair '
                'medians, the same scale as the '
                '3071 evidence), prereg frozen '
                'before the authoritative run; '
                'also fixed a baseline off-by-one '
                '(sum of recovs subtracts N*med_c_'
                '34) and reran smoke before '
                'sealing',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 212
    l14['connects'].append({
        'meas_id': 'meas3073_omega_p70_head_'
                   'interaction',
        'phase': 3073,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P70: head '
                        'interaction matrix - '
                        'the 3071 non-additivity '
                        'is STRONG SUB-ADDITIVITY '
                        '(24/28 pairs positive I, '
                        'max +0.104): two focal '
                        'heads swapped together '
                        'behave like roughly one '
                        'head (h20+h07 sum -0.435 '
                        'vs joint -0.195).  '
                        'Second-order expansion '
                        'FAILS quantitatively '
                        '(pred -0.212 vs measured '
                        'top-8 joint -0.532, err '
                        '0.320): saturation is '
                        'collective, not pairwise '
                        'decomposable.  '
                        'Interaction magnitude is '
                        'driven by effect sizes '
                        '(spearman(|r_a||r_b|,|I|)'
                        '=0.822), not by GQA/'
                        'quarter position '
                        'structure.  Top-8 joint '
                        '-0.532 = 93 percent of '
                        'the full 32-head -0.572. '
                        'Weak-head pairs flip sign '
                        '(h02+h24 +0.018).  Model: '
                        'focal heads write a '
                        'SHARED TT channel with '
                        'capacity-limited joint '
                        'effect - head causal '
                        'effect is a SET function '
                        'with diminishing returns. '
                        'Opens 3074: A saturation '
                        'curve R(S) fitting '
                        '(1/2/3/8/32 data); B '
                        'focal-set cross-prompt '
                        'stability; C DS7B '
                        'control; D natural-'
                        'generation lineage '
                        'drift; E neuron '
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
if '## Phase 3073:' not in memo:
    sec = u'''## Phase 3073: Ω-P70 头间交互矩阵——强次可加与二阶展开不完备（higher_order_required） [%(created)s]

**判决：`higher_order_required`**（qwen3-4b 单模型 bf16 61.2s，975 次前向；11 重 bit 锚全过：**a1：row 34 与 3066 npz diff=0.0；a8/a9：ZH 与 DAH 中位与 3071 npz diff=0.0；a10：top-8 单头重扫 recov 与 3071 result r34 diff=0.0——跨 phase 因果复现性完美**；aref diff=0.0；b8/b0/b1/b4/b6/b7 bit 0.0）。设计史如实入册：smoke 暴露样本级交互求和的噪声放大（28 项 I 求和使 cos 噪声放大 ~sqrt(28) 倍，预测值翻正、彻底失效）——权威运行前重设计为**中位簿记分解**（I(AB)=R(AB)−R(A)−R(B) 定义在 24 对中位上，与 3071 证据同尺度），预注册冻结后才跑权威；同时修正基线差一错误（recov 和减 N·med_c_34）。

### 问题与设计（3073 A，3072 菜单主选）
**问题**：3071 证明头级因果中位不可加（top-5 单独中位和 -0.968 vs 联合换回 -0.572）——交互项从哪来、结构是什么？**设计**：对 3071 焦点集 top8=[h20,7,1,14,26,0,2,24] 做（i）单头重扫（a10 锚：bit 复现 3071）；（ii）**28 对成对联合换回**（每对 24 样本）；（iii）top-8 全联合；（iv）三元组检验（3 个最负单头 h20/7/1，约束三阶残差）。所有效应为中位 recov（R(x)=median cos−med_c_34）；交互簿记 I(AB)=R(AB)−R(A)−R(B)；完备性：R_PRED=ΣR_i+ΣI_p vs 实测 R_U8。预注册门：交互显著 |I|≥0.02；二阶完备 PRED_ERR<0.05；三阶残差 ERR3<0.02。

### 核心结果（重复三遍）
**① 强次可加（一）**：28 对中 **24 对交互为正**（max +0.104 在 h01+h14 对）；典型：h20 单独 -0.222、h07 单独 -0.213，和 -0.435，**联合只有 -0.195——两个头一起换回的行为近似一个头**。**② 二阶展开定量失败（二）**：Σ singles=-1.219 + Σ I=+1.007 → 预测 -0.212，但 top-8 联合实测 **-0.532**（PRED_ERR=0.320，门的 6.4 倍）——**成对交互本身不可加：饱和是集体性质**，pair 级簿记无法外推到 8 头。三元组（h20/7/1）预测 -0.423 vs 实测 -0.375，ERR3=0.047>0.02 也越门——但已接近。还观察到**非单调性**：最佳对 h20+h01 联合 -0.389，加入 h07 的三元组反而变浅（-0.375）。**③ 交互由效应量驱动（三遍）**：spearman(|r_a|·|r_b|, |I|)=**0.822**——交互幅度由两头的效应量乘积决定，与位置结构无关（GQA 组内/组间均值 0.033/0.037，quarter 内/间同值——无结构偏好）。**④ 焦点集垄断确认**：top-8 联合 -0.532 ≈ 全 32 头换回 -0.572 的 **93 percent**——24 头长尾只贡献 0.04。**⑤ 弱头对翻转**：h02+h24 联合 +0.018、h00+h24 -0.011——近零头的符号不稳定，长尾写入方向不完全共线。

### 机制综合（Ω-P70 拼图）
头级因果效应不是每头项的和，而是**换回头集合 S 的集合函数 R(S)，具强边际递减**。现在有 R 的 5 个尺度点：单头 max 0.222 → 最佳对 0.389 → 三元组 0.375（非单调）→ top-8 0.532 → 全 32 头 0.572——**饱和曲线**：效应随头数增长迅速衰减。结合 3072（每个头线性读取注入 V 签名、OV 读语义族方向）与 3071（写入×传输、竞争写入）：焦点头们**写的是同一条 TT 通道**（写入向量共线），联合效应受通道容量约束——单独换回时每个头都"独占"通道（效应大），联合换回时互相挤占（次可加），饱和由通道的非线性（下游 MLP/L35 传输，3071 的脱钩证据）施加。E4 的 0.822 相关进一步说明通道内没有位置/结构分工——**分工只按效应量（写入幅度）排列，没有"专属频段"**。

### 硬伤与边界
- 中位簿记的 I(AB) 是"边际中位之差"，不是样本级交互的中位（两者都记录了：I_BOOK vs I_SMED，量级 0.08 vs 0.01——样本级交互被中位数的非线性放大，解释见 PREREG）；簿记分解只在 24 对中位尺度上自洽。
- PRED_ERR 0.320 的解释不唯一：二阶展开不完备、或 pair 级 I 的估计误差（24 对中位的 SE ~0.02-0.04，28 对累积仍 ~0.1-0.2）——三元组数据（ERR3=0.047）倾向前者但未闭合。
- 单 prompt 族、单模型、单注入层；R(S) 曲线只有 5 个点，函数形式（对数/幂/有理饱和）未拟合（3074 A）。
- 非单调性（三元组浅于最佳对）幅度 0.014 在 SE 边缘，只做观察。
- a10 完美复现依赖同代码同种子同 24 对——是管线一致性验证，不是新统计证据。

### 方法论入册
- **样本级交互求和陷阱**：cos 的样本级噪声 ~0.1-0.3，28 项求和噪声放大 sqrt(28)≈5.3 倍、SNR<1——交互分析必须在效应量同尺度（中位簿记）上进行；样本级交互中位只作观察。
- **饱和效应分析的集合函数范式**：R(S) 是集合函数（submodular 候选）——机器学习中的次模函数工具自然适用；pair 级展开是它的 2 阶截断。
- 跨 phase 因果复现锚 a10：同协议重扫 + bit 断言 = 管线一致性的最强检查。
- smoke 的价值再次验证：样本级求和的崩溃只在 smoke 就暴露（K3=4 时更极端），避免了权威数据浪费。

### 智能理论洞察（第一性原理）
**"条件化齿轮组"的齿轮共享同一条传动轴。** 3072 说每个头是线性读头（读哪里×读什么×写什么），3073 说这些读头的**写入端不是独立通道而是共享通道**：单独看每个头都"像"在起作用（边际效应 -0.22~-0.06），联合看效应不叠加（2 头≈1 头，8 头才 2.4×最强单头）——**功能分工由写入幅度排列，不存在结构化的"专属频段"**（GQA/位置无差）。这与大脑的功能冗余同构：单个神经元的损失被回路补偿（边际效应大），成片损失才显性（集落效应）——**因果作用是集合函数而非零件函数**。对 AGI 理论的推论：语言能力的机制单元不能按"头/神经元"逐个计数解释，必须按**通道容量与写入分布**解释——有限参数支撑无限组合的预算分配，可能正是通过这种"共线写入+容量饱和"实现鲁棒性。下一步缝隙：**R(S) 饱和曲线的函数形式**（对数？幂律？次模？）——它是"写入分布→通道容量"的第一性定律的雏形；以及通道身份的直接测量（焦点头的写入方向互相余弦多少——3072 OV 数据已可初步回答）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3073/omega_p70_head_interaction/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3074 菜单**——A（主选）**饱和曲线 R(S) 拟合**：补 4 头/6 头等中间规模联合换回（每档 24 前向），拟合 R(·) 函数形式（对数/幂/次模），检验"通道容量定律"。B **写入方向共线性**：focal5 的 out_h/dAh 方向互相余弦，直接测"共线写入"假设（可用 3072 npz 免前向）。C **焦点头集合跨 prompt 族稳定性**：新 prompt 族复测头级 recov 谱与 R(S)。D **DS7B 头级对照**：跨模型检验焦点化+饱和图景。E **神经元身份**：S_TOP 的 up/gate 权重结构。"好的，继续"即进 3074 A。
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
if '## 三十五、3073 增补' not in aud:
    add = u'''

---

## 三十五、3073 增补：头间交互矩阵——强次可加与饱和（Omega-P70，判决 higher_order_required）

1. **强次可加**：28 对成对联合换回中 24 对交互为正（中位簿记，max +0.104）；典型 h20+h07：单独和 -0.435、联合只 -0.195——两头联合 ≈ 一头。头级因果效应是换回集合 S 的集合函数，边际递减强。
2. **二阶展开不完备**：Σ singles -1.219 + Σ I +1.007 预测 -0.212，实测 top-8 联合 -0.532（PRED_ERR 0.320，门 0.05）——饱和是集体性质，pair 级簿记不可外推；三元组 ERR3 0.047 也越门但接近；观察到非单调（三元组 -0.375 浅于最佳对 -0.389）。
3. **交互由效应量驱动**：spearman(|r_a||r_b|,|I|)=0.822，GQA 组/quarter 结构无差（0.033 vs 0.037）——通道内无"专属频段"，分工按写入幅度排列。top-8 联合 = 全换回的 93 percent；弱头对符号翻转（h02+h24 +0.018）。
4. HDMCC 修正：**样本级交互求和陷阱**（28 项 cos 噪声求和放大 sqrt(28) 倍，预测翻正）——交互分析必须在效应量同尺度（中位簿记）上进行；R(S) 集合函数/次模范式入册；跨 phase 因果复现锚 a10（同协议重扫 bit 0.0）= 管线一致性最强检查。
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
if 'Phase 3073' not in prev:
    line = ('- Phase 3073 Omega-P70 head '
            'interaction matrix (qwen3-4b bf16 '
            'single, 61.2s, 975 forwards): '
            'verdict higher_order_required. '
            'Eleven bit anchors exact (a1/a8/a9/'
            'a10/aref/b8/b0/b1/b4/b6/b7; a10 = '
            'single-head resweep bit-exact vs '
            '3071 r34). Strong sub-additivity: '
            '24/28 pairwise interactions '
            'positive (max +0.104); h20+h07 sum '
            '-0.435 vs joint -0.195 (two heads '
            '~= one head). Second-order '
            'expansion fails: pred -0.212 vs '
            'measured top-8 joint -0.532 '
            '(PRED_ERR 0.320); triplet ERR3 '
            '0.047; non-monotonic (triplet '
            'shallower than best pair). '
            'Interaction driven by effect size '
            '(spearman 0.822), not position '
            'structure (GQA/quarter flat). '
            'Top-8 joint = 93 percent of full '
            '-0.572. Design history: smoke '
            'exposed sample-level interaction '
            'summing noise trap; redesigned to '
            'median-scale bookkeeping before '
            'the authoritative run. Audit 35; '
            'ledger 212/L14 180.\n')
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
if 'max=3073' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model；旧条目可能混有 str，verify 需 isinstance 防御）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃/smoke 推翻均如实登记；verdict 单分支赋值。
4. 统计纪律：阈值预注册；**交互/组合分析必须在效应量同尺度（中位簿记）**；样本级噪声项求和放大 sqrt(n) 倍（3073 教训）；SMOKE 判决仅供管线验证。

## 标准锚与精度
- bit 级锚家族：标量行（a1）、因果置换（a3/a4）、跨 phase 参考数（aref）、块链恒等（b8）、hook 互证（a5/a6）、probe 协议（b9/b10）、**跨 phase 因果复现（a10：同协议重扫 bit 0.0 vs 上一 phase result）**。
- a2b 型校准断言只在末层读出层有效；中间层用 aref。SMOKE 跳过的锚在 setup_ok 中视为通过。
- **3070**：ATTN 换回=o_proj 输入 H 末位 pre-hook。**3071**：per-head 换回=mask 头切片。**3072**：V-clamp 推锚法；谱系恒等式 dzH_h=sum_p w_h(p)·dV_p[g(h)]（GQA g=h//4）cos=1.0；per-head 数组勿存 float64。**3073**：中位簿记 I(AB)=R(AB)-R(A)-R(B)（24 对中位尺度）；recov 和的基线减 N*med_c_34（勿差一）。

## 机制解释审计链（命名前依次检查）
…→3071 头级分解（焦点头集合 h20/7/1/14/26，效应=写入×传输）→3072 焦点头谱系（响应=V 签名纯线性读取 cos 1.0；focal5 倾听注入位置 97.8-99.4 percent；OV 读语义族）→**3073 头间交互矩阵（判决 higher_order_required）：强次可加（24/28 对 I>0，两头≈一头）；二阶展开不完备（PRED_ERR 0.320）；交互由效应量驱动（spearman 0.822）非位置结构；top8 联合=全换回 93 percent——焦点头写共享 TT 通道，因果效应是集合函数 R(S) 具边际递减**。齿轮=线性读头，齿轮箱=非线性传动，**传动轴共享**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→脚本自带 run_log 用 Read 读；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查。
- median 轴陷阱（3071）：smoke 维度裁剪可隐藏轴错误——断言形状。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3073）
Ω-P2（3011-3073）：…3071 attn_heads_focal；3072 focal_lineage_full；**3073 higher_order_required（共享通道+集合函数因果效应）**。

## 下一步
- max=3073，下一个 3074（A 主选 **饱和曲线 R(S) 拟合**：补中间规模联合换回，拟合对数/幂/次模形式；B 写入方向共线性（3072 npz 可免前向）；C 焦点头跨 prompt 族稳定性；D DS7B 头级对照；E 神经元身份）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3073')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
