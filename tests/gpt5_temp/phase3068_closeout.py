# -*- coding: utf-8 -*-
"""Phase 3068 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3068'
     r'\omega_p65_sb_symmetric_competition')
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
assert verdict == 'competition_splitpool_' \
    'top_negative_both', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a2set_diff'] == 0 and an['a2set_ok'] is True
assert an['a3_diff'] == 0.0 and an['a3_ok'] is True
assert an['a2lens_max'] == 0.062412261962890625
assert an['a2lens_ok'] is True
assert an['a2b_diff'] < 0.0002
assert an['a2b_ok'] is True
assert an['b0_diff'] == 0.0 and an['b0_ok'] is True
assert an['b1_diff'] == 0.0 and an['b1_ok'] is True
assert an['b3_ok'] is True
assert an['b4_diff'] == 0.0 and an['b4_ok'] is True
assert an['b5_diff'] == 0.0
assert an['b6_diff'] == 0.0 and an['b6_ok'] is True
assert an['b7_diff'] == 0.0 and an['b7_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_34'] == 0.1487826048372403
assert st['med_c_35'] == -0.35579100779974404
assert st['pa34'] == 0.29685845971107483
assert st['pa35'] == 0.19866131991147995
assert st['pf34'] == 0.2371114194393158
assert st['pf35'] == -0.3558848798274994
assert st['ovl'] == 76
assert st['ovl_curve'] == {'32': 18, '64': 37,
                           '128': 76, '256': 153,
                           '512': 331, '1024': 645,
                           '2048': 1152}
assert st['partition_sizes'] == {
    'P1': 52, 'P2': 52, 'P3': 76, 'P4': 9548}
assert abs(st['lincheck_a']
           - 0.9999253246389584) < 1e-12
assert abs(st['lincheck_b']
           - 0.9999884032877095) < 1e-12
assert abs(st['share']['P3']['a']
           - 0.7345814806227212) < 1e-12
assert abs(st['share']['P3']['b']
           - 0.7465158223390831) < 1e-12
assert abs(st['cproj']['P3']['a']
           + 445.17287174254955) < 1e-9
assert abs(st['cproj']['P3']['b']
           + 978.720743464625) < 1e-9
assert abs(st['cosu']['P3']['a']
           + 0.570057004529458) < 1e-12
assert abs(st['cosu']['P3']['b']
           + 0.5647079186254024) < 1e-12
assert st['c_perm']['A35_SA'] == 0.35042549175552035
assert st['c_perm']['A35_SB'] == 0.30045811862474414
assert st['c_perm']['A35_SR'] == -0.3593714306364453
assert st['c_perm']['A35_ALL'] == 0.37592020929499254
assert st['c_perm']['B34_SA'] == 0.5379552249419755
assert st['c_perm']['B34_SB'] == 0.5652126070846724
assert st['c_perm']['B34_SR'] == 0.15358778427566655
assert st['c_perm']['B34_ALL'] == 0.586834466800143
assert abs(st['recov']['A_Sa']
           - 0.7062164995552644) < 1e-12
assert abs(st['recov']['A_Sb']
           - 0.6562491264244882) < 1e-12
assert abs(st['recov']['A_Sr']
           + 0.003580422836701236) < 1e-12
assert abs(st['recov']['A_all']
           - 0.7317112170947366) < 1e-12
assert abs(st['recov']['B_Sa']
           - 0.3891726201047352) < 1e-12
assert abs(st['recov']['B_Sb']
           - 0.4164300022474321) < 1e-12
assert abs(st['recov']['B_Sr']
           - 0.00480517943842626) < 1e-12
assert abs(st['recov']['B_all']
           - 0.4380518619629027) < 1e-12
assert res['forwards'] == 278

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3068
           for m in led['measurements']):
    claim = (
        'Omega-P65 (plan 3068 A) - qwen3-4b bf16 '
        'S_B symmetric selection + exact linear '
        'competition decomposition (21.4s, 278 '
        'forwards; anchors: a1 ladder rows 34/35 '
        'bit-exact vs 3066 npz diff 0.0; a2 S_A '
        'set equality vs 3067 npz diff 0; a3 '
        'PERM(A35,S_A)/PERM(B34,S_A) bit-exact '
        'vs 3067 npz diff 0.0; med_c reference '
        'assert; a2b 9.4e-05; b0/b1/b4/b5/b6/b7 '
        'bit 0.0; b3 finite; a2 fp32 lens within '
        '1 bf16 ulp). RESULTS: verdict '
        'competition_splitpool_top_negative_'
        'both. (1) SYMMETRY: S_B (top-128 by '
        'score_B) recovers recov_B_Sb=+0.416 '
        'vs random +0.005; cross-recovery '
        'near-symmetric (S_B under state A '
        '0.656 vs S_A 0.706; S_A under state B '
        '0.389 vs S_B 0.416) - the two top-128 '
        'sets are near-interchangeable. '
        '(2) OVERLAP: |S_A n S_B|=76/128 (59.4 '
        'pct) vs random expectation ~1.7 '
        '(~45x enrichment); overlap curve '
        'm=2048: 1152 vs ~431 expected - '
        'enrichment concentrated at the top. '
        'Preregistered verdict splitpool (76 < '
        '96 threshold) but the two sets are '
        'two heavily-overlapping samples of '
        'one writer pool, not two sub-'
        'populations. (3) EXACT LINEAR '
        'DECOMPOSITION (m = W_down act is '
        'linear, partition P1=S_A-S_B P2=S_B-'
        'S_A P3=both P4=rest): P3 dominates '
        'norm share in BOTH states (0.735 A / '
        '0.747 B) and carries the most-'
        'negative TT projection (cproj -445 A '
        '/ -979 B) with nearly identical '
        'direction cosines (cosu -0.570 / '
        '-0.565) - the SHARED neurons are the '
        'negative writers in both states, '
        'along the same readout direction. '
        '(4) ALL-swap upper anchor: full-9728 '
        'swap recovers 0.732/0.438, i.e. '
        'top-128 already captures 96.5/95.0 '
        'pct of the maximal recoverable. '
        'Conclusion: the L35 conditional '
        'reversal is carried by ONE neuron '
        'pool serving both state conditions '
        '(pool reuse at parameter level); '
        'state-dependence = input-direction '
        'selection (3067) + gain amplitude '
        '(score_B max 39.8 vs score_A max '
        '13.4) - no dedicated positive-writer '
        'sub-population exists.')
    meas = {
        'meas_id': 'meas3068_omega_p65_sb_'
                   'symmetric_competition',
        'phase': 3068,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 rows 34/35 bit 0.0 vs '
                   '3066 npz (hard); a2 S_A set '
                   'equality vs 3067 npz (hard); '
                   'a3 PERM bit 0.0 vs 3067 npz '
                   '(hard); med_c reference '
                   'assert; a2b 9.4e-05; '
                   'b0/b1/b4/b5/b6/b7 bit 0.0; '
                   'b3 finite; a2 fp32 lens '
                   '<= 1 bf16 ulp',
        'artifacts': {
            'result': 'phase3068/omega_p65_'
                      'sb_symmetric_competition/'
                      'result.json',
            'npz': 'phase3068/omega_p65_'
                   'sb_symmetric_competition/'
                   'omega_p65_sb_symmetric_'
                   'competition.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16; smoke '
                'zero-crash; docstring escape '
                'warning fixed pre-smoke)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 207
    l14['connects'].append({
        'meas_id': 'meas3068_omega_p65_sb_'
                   'symmetric_competition',
        'phase': 3068,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P65: S_B symmetric '
                        'selection + exact linear '
                        'competition decomposition '
                        '- ONE pool, TWO states: '
                        '|S_A n S_B|=76/128 (~45x '
                        'random enrichment); cross-'
                        'recovery near-symmetric '
                        '(0.656/0.389 vs 0.706/0.416); '
                        'shared partition P3 dominates '
                        'norm share (0.735/0.747) and '
                        'carries the most-negative TT '
                        'projection (-445/-979) with '
                        'identical direction (-0.570/'
                        '-0.565); top-128 captures '
                        '96.5/95.0 pct of the ALL-swap '
                        'upper bound. No dedicated '
                        'positive-writer sub-population; '
                        'state-dependence = input '
                        'direction (3067) + gain. '
                        'Opens 3069: A cross-layer '
                        'L30-34 same protocol; B DS7B '
                        'last-layer control; C input-'
                        'displacement lineage dz_A/dz_B; '
                        'D neuron identity (up/gate '
                        'weights, W_U alignment); E '
                        'cross-prompt-family '
                        'generalization'})
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
if '## Phase 3068:' not in memo:
    sec = u'''## Phase 3068: Ω-P65 S_B 对称选择+精确线性竞争分解——一个池子服务两个状态（competition_splitpool_top_negative_both） [%(created)s]

**判决：`competition_splitpool_top_negative_both`**（qwen3-4b 单模型 bf16 21.4s，278 次前向；锚：**a1：阶梯 34/35 两行与 3066 npz diff=0.000e+00**；**a2：S_A 集合与 3067 npz 完全相等（diff=0）——跨 phase 评分复现 bit 级**；**a3：PERM(A35,S_A)/PERM(B34,S_A) 与 3067 npz diff=0.000e+00**；med_c 参考断言通过；a2b lens 校准 9.4e-05；b0/b1/b4/b5/b6/b7 全 bit 0.0；b3 finite；a2 fp32 lens ≤1 bf16 ulp）。SMOKE 零崩溃（权威启动前修 1 个 docstring 转义警告）。

### 问题与设计
**问题**（3068 A，3067 菜单）：L35 MLP 条件反转的载体是**一个共享神经元池（状态依赖增益）**，还是**两个部分不同的亚群（S_A 负写者 vs S_B 正写者）**？设计四步：E1 双状态捕获（3067 逐字一致→a1 锚）；E1.5 对称评分 `score_s[j]=median_k(|dact_s[k][j]|·||W_down[:,j]||)`，S_A=top-128（a2 锚）、S_B=score_B top-128、OVL 曲线 m∈{32..2048}、S_R=seed 3024 从 S_A∪S_B 补集抽 128；E2 **精确线性分解**——m=W_down·act 是线性的，partition P1=S_A∖S_B、P2=S_B∖S_A、P3=S_A∩S_B、P4=rest，v_i=W_down[:,P_i]·dact[P_i]，Σv_i=Δm（lincheck 0.9999/1.0000），per-pair 范数份额 + 有符号 TT 投影 cproj_i（fp32 unembed 无 final norm，声明近似）；E3 八组因果置换 {A35,B34}×{S_A,S_B,S_R,ALL}（ALL=全 9728 换 base→m=m_base bit-exact，rest-of-system 上界锚；a3 锚）。

### 核心结果（重复三遍）
**① 对称有效（一）**：S_B 在状态 B 换回 base，c 从 +0.149→+0.565（recov_B_Sb=**+0.416**，随机对照 **+0.005**）——3067 的评分协议对称成立。交叉恢复近乎对称：S_B 在状态 A 恢复 **0.656**（S_A 自己 0.706），S_A 在状态 B 恢复 **0.389**（S_B 自己 0.416）——两个 top-128 集几乎可互换。**② 重叠（二）**：|S_A∩S_B|=**76/128（59.4 percent）**，随机期望约 1.7（=128×128/9728），富集约 **45 倍**；重叠曲线 m=2048 时 1152 vs 期望约 431（2.7 倍）——富集集中于 top 端。预注册判决 splitpool（76<96 阈值），忠实执行；但证据指向**同一写者池的两次高度重叠采样**，非两个亚群。**③ 共享子池主导（三遍）**：精确线性分解中 P3（共享）在两状态都是范数份额最大（**0.735/0.747**）且 TT 投影最负（cproj **-445/-979**），方向余弦两状态几乎相同（cosu **-0.570/-0.565**）——**共享神经元在两个状态下沿同一读出方向写负**；P4 长尾份额可观（0.36/0.24）但投影小（-50/-72）——背景有质量、无方向性。**④ 上界锚**：全 9728 换 base 恢复 0.732/0.438，top-128 已捕获上界的 **96.5/95.0 percent**。附加：score_B max=39.8 vs score_A max=13.4（约 3 倍）——状态 B 的位移幅度更大。

### 机制综合（Ω-P65 拼图）
3067 的"（神经元集 × 输入方向 × 竞争平衡）"现在有了竞争项的定量分解：**S_A 与 S_B 不是分工，是同一个负写入者池的两次采样**。不存在专门的"正写者亚群"——其余系统（P4+attention+更早层）以小投影总量构成正侧，S 池以大投影负侧，总符号=两侧平衡，而 S 池 membership 由输入位移方向门控（3067 的 dz_A/dz_B 近正交）。"复用与差异"落到参数级：**池复用**（76/128 重叠、交叉恢复对称）、**增益差异**（score_B max 3 倍）。条件化齿轮组的第二张图纸：齿轮（J）不变、传动（输入方向）选路、同一组齿在多个工况下复用，仅输出幅度随工况变。

### 硬伤与边界
- splitpool/sharedpool 的 96 阈值预注册时偏严（任意值）；实测 76 已远超随机——判决忠实于预注册，但"两个亚群"解释按证据降级为候选。
- cproj 有符号未归一（无 final norm，声明近似）——只做相对比较，不读绝对份额。
- recov 经 final_norm 非线性读出（overshoot 已知）——份额是定性的。
- 单 prompt 族（style/connector）、单模型、单层（L35）、单位置；跨层/跨族/跨模型泛化未测。
- ALL 换回使 m=m_base bit-exact 是差分中和，不是绝对功能消融。

### 方法论入册
- **精确线性分解**：m=W_down·act 线性 → partition 分解是恒等式（lincheck 0.9999 验证数值），份额/投影可解释；非线性模块无此待遇。
- **跨 phase bit 锚升级**：a2（集合相等）/a3（置换结果 bit 0.0）——"新脚本=旧机制"验证从标量扩展到集合与因果结果。
- **随机对照要算期望**：|S_A∩S_B| 的 null=m²/inter（约 1.7），富集倍数才是可比量。
- SMOKE 连续第四 phase 零崩溃达成。

### 智能理论洞察（第一性原理）
有限参数实现条件化的证据链闭合到池级：**J 固定（3067）× 输入方向选路（3067）× 同一池多工况复用（3068）**。"无限组合"的微缩模型更新为：有限参数组 × 有限输入位移方向谱系 × 幅度增益 → 任意（方向, 符号, 幅度）响应。关键转向：个体神经元的身份不再重要，重要的是**池的 membership 函数**——什么输入方向把哪些神经元拉进负写入 coalition。下一关键问题（3067/3068 共同指向）：**输入位移谱系**——dz_A 是否=注入 V 签名经 ln1/attn/ln2 的线性像？dz_B 沿哪条路径传播？方向由什么决定？这决定"传动轴"是否可预测、可组合。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3068/omega_p65_sb_symmetric_competition/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3069 菜单**——A（主选）**跨层推广**：swap-to-base 同协议应用到 L30-34 正带 MLP（每层 top-128 + 随机对照 + ALL 上界），检验"同一池多工况复用"是否逐层普遍；B **DS7B 末层对照**：DS7B 末层 MLP 解剖复刻（跨模型检验，3065 菜单 C 遗留）；C **输入方向谱系**：dz_A 是否=注入 V 签名的线性像（ln1/attn/ln2 逐级传播分解）、dz_B 路径分解——"传动轴"可预测性；D **S_A/S_B 神经元身份**：top-128 的 up/gate 权重结构、与 W_U 语义方向关联、跨 prompt 族 dact 稳定性；E **跨 prompt 族泛化**：style/connector 之外的新 prompt 族复测池重叠。"好的，继续"即进 3069 A。
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
if '## 三十、3068 增补' not in aud:
    add = u'''

---

## 三十、3068 增补：S_B 对称选择+精确线性竞争分解——一个池子服务两个状态（Omega-P65，判决 competition_splitpool_top_negative_both）

1. **对称有效**：S_B（score_B top-128）在状态 B 换回 base 使 c +0.149→+0.565（recov_B_Sb=+0.416，随机 +0.005）；交叉恢复近乎对称（S_B@A 0.656 vs S_A@A 0.706；S_A@B 0.389 vs S_B@B 0.416）——两个 top-128 集几乎可互换。
2. **重叠=同一池**：|S_A∩S_B|=76/128（59.4 pct），随机期望约 1.7（约 45 倍富集）；预注册判决 splitpool（76<96）忠实执行，但证据降级"两个亚群"解释为候选——实为同一写者池的两次高度重叠采样。
3. **精确线性分解**：m=W_down·act 线性 → partition 恒等式分解（lincheck 0.9999）；共享 P3 在两状态范数份额最大（0.735/0.747）、TT 投影最负（-445/-979）、方向几乎相同（cosu -0.570/-0.565）——共享神经元在两状态沿同一方向写负；P4 长尾有质量无方向性；top-128 捕获 ALL 上界的 96.5/95.0 pct。
4. HDMCC 修正：无专门"正写者亚群"；条件化=输入方向选路（3067）+ 同一池复用 + 增益幅度（score_B max 3 倍于 A）。符号定位四件套升级为五件套（+精确线性分解/重叠期望对照）。
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
if 'Phase 3068' not in prev:
    line = ('- Phase 3068 Omega-P65 S_B symmetric '
            'selection + exact linear competition '
            'decomposition (qwen3-4b bf16 single, '
            '21.4s, 278 forwards): verdict '
            'competition_splitpool_top_negative_'
            'both. Anchors all bit-exact (a1 vs '
            '3066, a2 S_A set + a3 PERM vs 3067). '
            'ONE pool TWO states: |S_A n S_B|='
            '76/128 (~45x random enrichment); '
            'cross-recovery near-symmetric (0.656/'
            '0.389 vs 0.706/0.416); shared P3 '
            'dominates norm share (0.735/0.747) '
            'and carries most-negative TT '
            'projection (-445/-979) with same '
            'direction (-0.570/-0.565); top-128 '
            'captures 96.5/95.0 pct of ALL-swap '
            'upper bound. No dedicated positive-'
            'writer sub-population; state-'
            'dependence = input direction (3067) '
            '+ gain (score_B max 3x). Audit 30; '
            'ledger 207/L14 175.\n')
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
if 'max=3068' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃如实登记；verdict 单分支赋值。
4. 统计纪律：obs/null 同量纲；负对照带符号解读；阈值预注册；随机对照判特异性；重叠类指标必须算随机期望（如 m^2/inter）。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨脚本 bit 级锚是"新脚本=旧机制"最强验证：a1 标量行（vs 3066）、a2 集合相等、a3 因果置换结果（vs 3067）——锚已从标量扩展到集合与因果结果。
- **3066**：per-k 聚合显式累积后 median；cos 探针 fp32 权重副本；消融配 matched base；SMOKE 先行。
- **3067**：改模块输入用 forward-pre-hook；base 自替换=恒等→免 matched base（b7）；权重副本 .detach().float()；符号定位四件套=评分选择→换回 base 置换→随机对照→雅可比检验。
- **3068**：m=W_down·act 线性→partition 恒等分解（P1/P2/P3/P4 份额+有符号 TT 投影+lincheck）；ALL 置换=m_base bit-exact 上界锚；top-k 捕获率=recov_top/recov_ALL。

## 机制解释审计链（命名前依次检查）
…→3063 竞争轴→3064 跨模型（拓扑普适/符号特异）→3065 符号溯源→3066 翻转解剖（编排单位=（组件,状态条件）对）→3067 神经元级解析（定位 top-128、两状态共享、线性雅可比、输入位移近正交 0.028）→**3068 竞争分解：一个池子服务两个状态（重叠 76/128 约 45 倍富集、交叉恢复对称、P3 共享子池主导且方向一致）→（J 固定 × 输入方向选路 × 同一池多工况复用 + 增益幅度）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径用 D:/ 风格（/d/ 失败）；管道工具（tail 等）缺失→脚本自带 run_log 用 Read 读；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查（含 docstring 转义警告）。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3068）
Ω-P2（3011-3068）：…3065 sign_decoupled_all3；3066 sign_flip_downstream_distributed；3067 mlp_reversal_localized_shared；**3068 competition_splitpool_top_negative_both（一个负写入者池×两个状态工况；J 固定=齿、输入方向=传动、池复用+增益=工况）**。

## 下一步
- max=3068，下一个 3069（A 主选 **跨层推广 L30-34 同协议**；B DS7B 末层对照；C 输入方向谱系 dz_A/dz_B；D S_A/S_B 神经元身份；E 跨 prompt 族泛化）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3068')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
