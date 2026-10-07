# -*- coding: utf-8 -*-
"""Phase 3075 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3075'
     r'\omega_p72_supermodular_structure')
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
assert verdict == 'supermodular_diffuse', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a2_ok'] is True
assert an['a3_ok'] is True
assert an['a4_ok'] is True
assert an['a5_ok'] is True
assert an['a6_diff'] == 0.0 and an['a6_ok'] is True
assert an['b0_diff'] \
    == 3.3306690738754696e-16
assert an['b0_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['n_viol_re'] == 4359
assert st['viol_rate_re'] == 0.26463088878096164
assert st['n_pos_mu'] == 88
assert abs(st['pos_sum']
           - 7.367974765012477) < 1e-12
assert st['c10'] == 0.3292310776105696
assert st['max_mu_hi'] == 0.34525139007733663
assert st['mu2_min'] == -0.10387255949605928
assert st['mu2_max'] == 0.005628104014067381
assert st['sp_m_viol'] == 0.2142857142857143
assert st['sp_size_sm'] == 0.5055599285568148
assert st['sm_count'] == 467
assert st['sp_dah_m'] == -0.7380952380952381
assert st['sp_dah_v'] == -0.5952380952380953
assert st['w_pair_total'] == 774
assert st['w_gqa'] == 106 and st['w_qtr'] == 188
assert st['base_gqa'] == 4 and st['base_qtr'] == 7
assert st['spectrum']['2']['n_pos'] == 0
assert st['spectrum']['3']['n_pos'] == 5
assert st['spectrum']['4']['n_pos'] == 46
assert st['spectrum']['6']['n_pos'] == 19
assert st['spectrum']['6']['max'] \
    == 0.34525139007733663
assert st['spectrum']['8']['max'] \
    == 0.30888883323097094
assert st['worst10'][0] == {
    'delta': 0.33719686075775934,
    'S_mask': 251, 'x': 2, 'T_mask': 186}
assert res['forwards'] == 0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3075
           for m in led['measurements']):
    claim = (
        'Omega-P72 (plan 3075 A) - qwen3-4b '
        'supermodular structure localization '
        'via the full Moebius spectrum of the '
        'measured set function A(S) (NO '
        'forwards: pure re-analysis of the '
        'frozen 3074 npz A_S over all 255 '
        'subsets, plus 3073 r1/r2).  (0.1s; '
        'anchors: a1 deterministic enumeration '
        'replay of all 16472 marginal diffs '
        'bit 0.0 vs the 3074 npz PAIRS4; a2 '
        'n_viol=4359 and viol_rate bit vs '
        '3074 result; a3 A(255) bit; a4 npz '
        'R_ALL bit vs 3071 gA; a5 3073 file '
        'anchor; a6 mu2 pairs bit 0.0 vs the '
        '3073 r1/r2 chain with the '
        'brute-force summation order matched; '
        'b0 fast-vs-brute Moebius 3.3e-16 '
        '<= 1e-12 - algorithmic equivalence '
        'across different fp summation '
        'orders).  RESULTS: verdict '
        'supermodular_diffuse.  (1) PAIRS '
        'ARE ALL SUBMODULAR: mu2 has 0 '
        'positive coefficients (max +0.0056 '
        'vs tol 0.02, min -0.1039 = -max '
        'I_BOOK of 3073) - the 3073 '
        'pair-level competition picture is '
        'confirmed exactly.  (2) POSITIVE '
        'MASS LIVES AT HIGH ORDERS: order '
        'census of positive coefficients '
        '(>0.02): mu2 0/28, mu3 5/56 (max '
        '+0.052), mu4 46/70 (max +0.157), '
        'mu5 17/56 (max +0.116), mu6 19/28 '
        '(max +0.345 - the largest), mu7 '
        '0/8, mu8 = +0.309 (the full set '
        'itself is supermodular).  (3) '
        'VIOLATIONS ARE LARGE-BASE: rate by '
        'base size 0.010 (|S|=2) -> 0.051 '
        '(3) -> 0.086 (4) -> 0.256 (5) -> '
        '0.510 (6) -> 0.750 (7); by budget '
        'bucket 0.030 (lowest quartile) -> '
        '0.473 (highest).  All ten worst '
        'inequalities are h01 completing a '
        '6-head base (worst +0.337 = the 7-'
        'head set minus h01 vs a 4-head '
        'subset).  (4) DIFFUSE, NOT '
        'LOCALIZED: 88 positive coefficients '
        'summing 7.368, concentration c10 = '
        '0.329 < 0.5; the top-12 pair '
        'weights inside positive masks are '
        'FLAT (28-30 occurrences each, mean '
        '27.6) - there are NO privileged '
        'synergy cliques.  (5) NO POSITION '
        'STRUCTURE: GQA-same rate inside '
        'positive masks 0.137 vs baseline '
        '0.143 (4/28), quarter 0.243 vs '
        '0.250 (7/28); spearman(m_i, '
        'viol_sum_x) = 0.214 (weak); '
        'pooled spearman(|S|, SM(i,j|S)) = '
        '0.506 - synergy GROWS with base '
        'size; SM median negative for most '
        'pairs but SM max positive for all '
        '28 (467/1792 profiles above tol) - '
        'small-base competition, large-base '
        'completion.  (6) |DAH34_MED| vs '
        'm_i spearman = -0.738 (recorded: '
        'the strongest causal heads have '
        'the SMALLEST direct OV projections '
        '- their effect routes through '
        'transfer, consistent with 3071 '
        'write x transfer).  Conclusion: '
        'the amplitude A(S) is TWO-REGIME - '
        'submodular competition at small '
        'bases (where the 3074 Hill '
        'capacity law is fitted and '
        'validated) plus a diffuse '
        'completion/completeness effect at '
        'large bases (the last heads '
        'complete the write budget '
        'synergetically), which explains '
        'why the fitted Hill asymptote '
        '0.714 exceeds the measured '
        'all-32 recovery 0.572.')
    meas = {
        'meas_id': 'meas3075_omega_p72_supermodular_'
                   'structure',
        'phase': 3075,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 enumeration replay 16472 '
                   'marginals bit 0.0 vs 3074 '
                   'npz PAIRS4 (hard); a2 '
                   'n_viol/viol_rate bit vs '
                   '3074 result (hard); a3 '
                   'A(255) bit (hard); a4 R_ALL '
                   'bit vs 3071 gA (hard); a5 '
                   '3073 r1[0] file anchor; a6 '
                   'mu2 bit 0.0 vs 3073 r1/r2 '
                   'chain (summation order '
                   'matched); b0 fast-vs-brute '
                   'Moebius <=1e-12 (3.3e-16 '
                   'observed; different fp '
                   'summation orders)',
        'artifacts': {
            'result': 'phase3075/omega_p72_'
                      'supermodular_structure/'
                      'result.json',
            'npz': 'phase3075/omega_p72_'
                   'supermodular_structure/'
                   'omega_p72_supermodular_'
                   'structure.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (no '
                'forwards; frozen 3074 npz '
                're-analysis).  Moebius '
                'spectrum = set-function '
                'Fourier transform; a6 kept '
                'bit-exact by reordering the '
                'reference sum to match the '
                'brute-force enumeration '
                'order (the underlying A '
                'values are bit-identical to '
                '-(R1/R2) via the 3074 a10 '
                'family anchor); smoke also '
                'corrected the b0 gate to '
                '1e-12 with PREREG updated '
                'before the authoritative '
                'run',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 214
    l14['connects'].append({
        'meas_id': 'meas3075_omega_p72_supermodular_'
                   'structure',
        'phase': 3075,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P72: Moebius '
                        'spectrum of the '
                        'focal-head set '
                        'function (no '
                        'forwards).  mu2 all '
                        'nonpositive (pair '
                        'competition '
                        'confirmed); '
                        'positive Mobius '
                        'mass is HIGH-ORDER '
                        'and DIFFUSE (c10 '
                        '0.329, top masks '
                        'are 6-head bases '
                        'and the full set, '
                        'mu8 +0.309); '
                        'violation rate '
                        'rises monotonically '
                        'with base size '
                        '(0.010 at |S|=2 to '
                        '0.750 at |S|=7) '
                        'and with budget '
                        '(0.030 to 0.473); '
                        'all worst violations '
                        'are h01 completing a '
                        '6-head base; pair '
                        'weights flat (28-30, '
                        'no cliques); GQA/'
                        'quarter rates at '
                        'baseline (0.137/0.243 '
                        'vs 0.143/0.250); '
                        'spearman(|S|,SM)='
                        '0.506.  Model: '
                        'TWO-REGIME amplitude '
                        '- submodular '
                        'competition (Hill '
                        'region, small '
                        'bases) + diffuse '
                        'large-base completion '
                        '(completeness '
                        'effect), explaining '
                        'the 0.714 vs 0.572 '
                        'asymptote gap.  '
                        'Opens 3076: A '
                        'focal-set cross-'
                        'prompt stability '
                        '(Hill + spectrum '
                        'replication); B '
                        'write-direction '
                        'collinearity (3072 '
                        'npz, no forwards); '
                        'C DS7B head-level '
                        'control; D neuron '
                        'identity; E '
                        'completion-effect '
                        'downstream '
                        'localization from '
                        '3074 L35 data'})
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
if '## Phase 3075:' not in memo:
    sec = u'''## Phase 3075: Ω-P72 Möbius 谱与超模结构定位——弥散的大基座补全效应（supermodular_diffuse） [%(created)s]

**判决：`supermodular_diffuse`**（**免前向**：对冻结的 3074 npz 纯重分析，0.1s；7 重锚全过：**a1：确定性枚举重放 16472 条边际差与 3074 npz PAIRS4 diff=0.0；a2：n_viol=4359 与 viol_rate 与 3074 result bit 一致；a3：A(255) bit；a4：R_ALL 与 3071 gA bit；a5：3073 文件锚；a6：μ2 配对系数与 3073 r1/r2 链 diff=0.0**——通过把参考表达式重排为暴力枚举的求和顺序实现 bit 锚（底层 A 值经 3074 a10 家族锚与 −(R1/R2) bit 一致）；b0：快速 Möbius vs 暴力求和 3.3e-16 ≤ 1e-12（不同浮点求和顺序的算法等价性，PREREG 已注明非同路径 bit 锚））。勘误登记（追加说明）：3074 脚本 docstring 头注释的 E4 计数误写为 17496，实际枚举（T 为真子集含空集）与 PREREG/execution/result/run_log 一致均为 16472——冻结证据链无影响，script sha 不变。

### 问题与设计（3075 A，3074 菜单主选）
**问题**：3074 测得 26.5 percent 的次模违反——**违反住在哪？是特定"协同头对/小团体"还是弥散结构？** **设计**：集合函数的次模性 ⟺ 其 **Möbius 变换（集合傅里叶变换）μ(S)=Σ_{T⊆S}(−1)^(|S|−|T|)·A(T) 的全部 ≥2 阶系数 ≤ 0**——因此对 255 个子集振幅做完整 Möbius 谱分解（3^8=6561 项暴力+快速变换互验），正系数直接给出协同组合与所在阶数；辅以四层定位（按新增头、按基座大小、按预算桶、worst-10 不等式）、28 对×64 基座的条件协同剖面 SM(i,j|S)=A(S+ij)−A(S+i)−A(S+j)+A(S)、GQA（h//4）/quarter（h//8）位置对照与单头 OV 谱（DAH34_MED，3074 npz 自带）对照。预注册门：TOL_MOB=0.02（与 3074 TOL_SUB 同噪声尺度）；集中度门 c10=top-10 正质量和/正质量总和 ≥0.5 → localized。

### 核心结果（重复三遍）
**① pair 全部次模（一）**：μ2 的 28 个系数 **0 个为正**（max +0.0056 < 门，min −0.1039 = −3073 max I_BOOK）——3073 的"两头竞争"图景被精确确认。**② 正质量住在高阶（二）**：正系数阶分布 μ2: 0/28 → μ3: 5/56 (max +0.052) → μ4: 46/70 (max +0.157) → μ5: 17/56 → **μ6: 19/28 (max +0.345，全谱最大)** → μ7: 0/8 → **μ8（全集本身）= +0.309**——超模质量不在 pair 而在 4-6 头大基座与全集。**③ 违反随基座单调上升（三）**：违反率 |S|=2 → 1.0 percent、|S|=3 → 5.1、|S|=4 → 8.6、|S|=5 → 25.6、|S|=6 → 51.0、**|S|=7 → 75.0**；按预算桶 3.0 → 47.3 percent；**worst-10 不等式全部是 h01 向 6 头基座补全**（worst +0.337）。**④ 弥散而非局域**：88 个正系数总和 7.368，c10=0.329 < 0.5；正 mask 内部 pair 权重完全平坦（top-12 每对 28-30 次，均值 27.6）——**不存在专属协同小团体**。**⑤ 无位置结构**：正 mask 内 GQA 同组率 0.137 vs 基线 0.143、quarter 0.243 vs 0.250——超模与 GQA/quarter 无关；spearman(m_i, viol_sum_x)=0.214（弱）；pooled spearman(|S|, SM)=**0.506**——协同随基座增大而增大；28 对的 SM 中位大多为负（小基座竞争）但 **SM max 全部为正**（467/1792 剖面越门）——**小基座竞争、大基座补全的双区结构**。附带发现：spearman(|DAH34_MED|, m_i)=−0.738——单头因果效应最强的头其 OV 直接投射反而最小，效应经由传输路径放大（与 3071 写入×传输一致），记录待解。

### 数学公式
- Möbius 变换：μ(S) = Σ_{T⊆S} (−1)^(|S|−|T|) · A(T)；反演 A(S) = Σ_{T⊆S} μ(T)；快速变换 O(8·256) 与暴力 3^8 互验；
- 次模判据：A 次模 ⟺ ∀|S|≥2: μ(S) ≤ 0；26.5 percent 边际违反的来源 = 正 μ4-μ6 + μ8；
- 条件协同：SM(i,j|S) = A(S∪{i,j}) − A(S∪{i}) − A(S∪{j}) + A(S)（S 基座 |S|=1 时 SM = μ3）；
- 双区结构经验律：|S|≤3 区域 μ≤0（Hill 容量定律有效区）；|S|≥5 区域 μ 大量正（补全区）。

### 硬伤与边界
- TOL_MOB=0.02 下 μ3-μ5 的部分正系数（尤其 0.02-0.05 区间）可能混入 24 对中位的估计噪声（SE ~0.02-0.04）；但 μ6 max 0.345、μ8 0.309、worst Δ 0.337 远超噪声，高阶结论稳健。
- 单 prompt 族、单注入层；Möbius 系数对 A 的中位估计误差随阶数累积（μ6 由 64 个 A 值组合），阶越高噪声越敏感——μ7 全负与 μ8 正的对比因此更显可信（同阶噪声下符号相反）。
- "补全效应"的机制来源未定位（下游 L35/MLP 非线性？写入空间几何？）——3074 npz 的 L35 数据可做免前向追踪（3076 E）。
- c10 门 0.5 的选择无外部依据（预注册约定）；若门取 0.25 结论不变（0.329>0.25 仍 diffuse，因 88 个正系数分散）。

### 方法论入册
- **Möbius 谱 = 集合函数的傅里叶诊断**：26.5 percent 的边际违反率一句话分解为"哪一阶、哪些组合"——边际不等式只给违反总量，谱给结构。
- **跨路径 bit 锚的求和顺序匹配**：不同浮点路径 bit 一致的前提是同一操作序列；a6 通过显式重排参考表达式实现（底层值经 a10 家族 bit 同源）；不同算法（fast vs brute）只能等价性门 1e-12。
- 免前向 phase 的价值：零协议漂移风险（数据冻结），全部结论是已有数据的确定性重算。

### 智能理论洞察（第一性原理）
**写入通道呈"竞争-补全"双区结构。** 小基座（≤3 头）：写入器竞争共享通道，边际递减，Hill 容量定律定量有效；大基座（≥5 头）：出现弥散的**完备性效应**——集合越接近完整写入预算，每个剩余头的边际价值越大（μ6 最大 + μ8 正），如同"最后一块拼图"效应。这不是局部协同（无专属 pair 团、无位置偏好），而是**全局预算接近饱和时的集体非线性**——与 3074 的渐近 0.714 > 全换回 0.572 缝合：全换回受限于"集合必须完整才释放的补全收益"，预算参数化（只数幅度之和）看不到"哪块缺了"。对 AGI 理论的推论：**功能集团的因果作用分析必须区分"竞争区"与"完备区"两个 regime**——在竞争区可约化为标量守恒量（容量定律），在完备区必须保留集合结构；这为"有限参数实现无限组合"提供了第二重机制：不是所有单元都在所有时候等价可交换，**系统的响应依赖写入配置的完整性**，这可能是冗余与鲁棒之间平衡的数学形式。下一步缝隙：完备性效应的下游定位（3074 L35 数据免前向可查）与跨 prompt 稳定性（谱形状是否 prompt 无关）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3075/omega_p72_supermodular_structure/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3076 菜单**——A（主选）**焦点头集合跨 prompt 族稳定性**：新 prompt 族复测单头谱 + Hill 参数 + Möbius 谱形状，检验"容量定律+双区结构"是否 prompt 无关（~1000 前向）。B **写入方向共线性**：focal5 的 out_h/dAh 方向互相余弦（3072 npz 免前向）。C **完备性效应下游定位**：用 3074 npz 的 ZH35/DAH35 检查补全效应在 L35 传输区的表达（免前向）。D **DS7B 头级对照**：跨模型检验焦点化+容量定律+双区图景。E **神经元身份**：S_TOP 的 up/gate 权重结构。"好的，继续"即进 3076 A。
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
if '## 三十七、3075 增补' not in aud:
    add = u'''
---
## 三十七、3075 增补：Möbius 谱与超模结构定位（Omega-P72，判决 supermodular_diffuse）
1. **pair 全次模**：μ2 的 28 个系数 0 正（min −0.104 = −3073 max I_BOOK）——两头竞争精确确认；正 Möbius 质量在高阶（μ4 46/70、μ6 19/28 max +0.345、μ8 +0.309）。
2. **违反是大基座现象**：违反率随基座大小单调 1.0 → 75.0 percent（|S|=2 → 7），随预算 3.0 → 47.3 percent；worst-10 全部是 h01 补全 6 头基座（max +0.337）。
3. **弥散无局域结构**：c10=0.329 < 0.5；正 mask 内 pair 权重平坦（28-30 次/对）；GQA/quarter 同组率等于基线（0.137/0.243 vs 0.143/0.250）；spearman(|S|,SM)=0.506——小基座竞争、大基座补全的双区结构，解释 3074 渐近 0.714 > 全换回 0.572。
4. HDMCC 修正：**Möbius 谱 = 集合函数傅里叶诊断**入册（边际违反率→阶与组合定位）；跨路径 bit 锚的求和顺序匹配技术；免前向 phase（冻结数据确定性重算）零协议漂移。附记录：|DAH34_MED| 与 m_i spearman −0.738（强因果头 OV 直接投射反而小，效应经传输放大，待解）。
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
if 'Phase 3075' not in prev:
    line = ('- Phase 3075 Omega-P72 Moebius '
            'spectrum (NO forwards, 0.1s, '
            'frozen 3074 npz re-analysis): '
            'verdict supermodular_diffuse. '
            'Seven anchors exact (a1 replay '
            'of 16472 marginals bit 0.0; a6 '
            'mu2 bit 0.0 vs 3073 chain via '
            'summation-order matching; b0 '
            'fast-vs-brute 3.3e-16 <= 1e-12). '
            'mu2 all nonpositive (pair '
            'competition confirmed); positive '
            'mass high-order (mu6 max +0.345, '
            'mu8 +0.309); violation rate '
            'monotone in base size (0.010 -> '
            '0.750) and budget (0.030 -> '
            '0.473); worst-10 all h01 '
            'completing 6-head bases; pair '
            'weights flat (28-30, no '
            'cliques); GQA/quarter at '
            'baseline; spearman(|S|,SM)=0.506. '
            'Two-regime model: submodular '
            'competition (Hill region) + '
            'diffuse large-base completion. '
            '3074 docstring miscount erratum '
            'logged (PREREG/results correct at '
            '16472). Audit 37; ledger '
            '214/L14 182.\n')
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
if 'max=3075' not in mem_cur:
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
4. 统计纪律：阈值预注册；交互/组合分析在效应量同尺度（中位簿记）；样本级噪声求和放大 sqrt(n) 倍。

## 标准锚与精度
- bit 锚家族：标量行（a1）、因果置换（a3/a4）、跨 phase 参考数（aref）、块链恒等（b8）、hook 互证（a5/a6）、跨 phase 因果复现（a10）、嵌入式家族锚（3074：上 phase 全部条件作为扫描子集 bit 复现）、**枚举重放锚（3075 a1：确定性索引重放 vs 冻结 npz 数组 bit 0.0）**。
- 跨路径 bit 锚需匹配浮点求和顺序（3075 a6：重排参考表达式）；不同算法等价性用 ≤1e-12 门（3075 b0）并 PREREG 注明。
- **3070** ATTN 换回=o_proj 输入末位 pre-hook；**3071** per-head=mask 头切片；**3072** 谱系恒等式 cos=1.0；**3073** 中位簿记 I(AB)=R(AB)-R(A)-R(B)、基线减 N*med_c_34；**3074** 预算参数化+双门外推；L35 head_decomp 需补捕获 ZA35；**3075** Möbius 谱（暴力 3^8 + 快速变换互验）；免前向 phase 读上 phase npz（A_S/PAIRS4/TOP8/DAH34）。

## 机制解释审计链（命名前依次检查）
…→3073 共享 TT 通道+集合函数→3074 容量定律（Hill a=0.714 b=0.454 γ=1.186 双门外推；非严格次模 26.5 percent）→**3075 Möbius 谱（supermodular_diffuse）：μ2 全负（pair 竞争）；正质量高阶弥散（μ6 max +0.345、μ8 +0.309）；违反率随基座 1→75 percent；无协同小团体、无位置结构；双区结构=小基座竞争（Hill 区）+大基座补全（完备性效应），解释渐近 0.714>全换回 0.572**。齿轮=线性读头，传动轴共享，传动轴有容量定律+完备性效应。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→run_log 用 Read；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核（3074 曾 8 连 Edit 仅 2 处落盘）；改后必编译检查。
- median 轴陷阱：smoke 维度裁剪可隐藏轴错误——断言形状。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3075）
Ω-P2（3011-3075）：…3071 attn_heads_focal；3072 focal_lineage_full；3073 higher_order_required；3074 capacity_law_hill；**3075 supermodular_diffuse（双区结构：竞争+补全）**。

## 下一步
- max=3075，下一个 3076（A 主选 **焦点头跨 prompt 族稳定性**：新 prompt 族复测单头谱+Hill+谱形状，~1000 前向；B 写入方向共线性（3072 npz 免前向）；C 完备性效应下游定位（3074 L35 数据免前向）；D DS7B 头级对照；E 神经元身份）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3075')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
