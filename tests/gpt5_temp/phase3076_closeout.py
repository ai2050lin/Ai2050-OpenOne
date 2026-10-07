# -*- coding: utf-8 -*-
"""Phase 3076 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3076'
     r'\omega_p73_cross_prompt_family')
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
assert verdict == 'cross_prompt_unstable', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
assert res['forwards'] == 20925
st = res['stats']
fam = st['families']
assert fam['A']['stable'] is True
assert fam['B']['stable'] is False
assert fam['C']['stable'] is False
assert fam['A']['top8'] == [20, 7, 1, 14, 26, 0, 2, 24]
assert fam['B']['top8'] == [14, 7, 10, 24, 26, 2, 15, 11]
assert fam['C']['top8'] == [1, 20, 13, 25, 0, 19, 4, 14]
assert fam['A']['viol_rate'] == 0.26463088878096164
assert fam['B']['viol_rate'] == 0.5300509956289461
assert fam['C']['viol_rate'] == 0.3904808159300631
assert fam['A']['r_all32'] == -0.5717521069904176
assert fam['B']['r_all32'] == -0.32319844598763103
assert fam['C']['r_all32'] == -0.7423764830266624
assert fam['A']['spectrum']['2']['n_pos'] == 0
assert fam['B']['spectrum']['2']['n_pos'] == 0
assert fam['C']['spectrum']['2']['n_pos'] == 11
assert fam['C']['spectrum']['8']['max'] == -0.01630199735526322
assert fam['C']['a_best_subset']['mask'] == 191
assert fam['C']['a_best_subset']['amp'] == 0.5225939164469149
cross = st['cross']
assert cross['ov_ab'] == 5 and cross['ov_ac'] == 4
assert cross['ov_bc'] == 1
assert cross['n_stable'] == 1
assert cross['sp_dah_ab'] == 0.8251466275659823
assert cross['sp_r1_bc'] == -0.08834310850439882
for fk in ('A', 'B', 'C'):
    fa = fam[fk]
    assert fa['b_anchors']['setup_ok'] is True
    assert fa['b_anchors']['b8_ok'] is True
    assert fa['top8_sel_ok'] is True
    for ak in ('a1', 'aref', 'a8', 'a9', 'a10s',
               'a10p', 'a11', 'a12', 'a13', 'a14',
               'a15', 'a16'):
        assert fa['anchors'][ak]['ok'] is True, (fk, ak)
assert fam['A']['anchors']['a1']['diff'] == 0.0
assert fam['A']['anchors']['a16']['diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3076
           for m in led['measurements']):
    claim = (
        'Omega-P73 (plan 3076 A) - qwen3-4b '
        'focal-head cross-prompt-family '
        'stability (20925 forwards, 19.8 min): '
        'the full 3074-protocol (banks + '
        'b-anchors + E1 24-pair ladder inj@34 '
        '+ E2H head decomp + E3 32-head '
        'single scan + per-family focal top8 '
        '+ E4 ALL 255-subset sweep + E5 16472 '
        'submodular inequalities + E6 '
        'capacity-law fit2 double gates + E7 '
        'full Moebius spectrum + E8 '
        'cross-family) run on THREE prompt '
        'families sharing the syntactic '
        'causal-connective frame but '
        'differing in semantic domain: A = '
        'everyday-causal (3074 texts rerun, '
        'bit-anchor family), B = '
        'science-causal, C = social-emotional-'
        'causal.  VERDICT '
        'cross_prompt_unstable.  (1) ZERO '
        'DRIFT PROVEN: family A reproduces '
        'ALL 16 cross-phase anchors bit-exact '
        '(a1 3066 ladder row 34; aref 3069 '
        'PA34/PF34; a8/a9 3071 ZH/DAH; a10s '
        '3071 r34 all-32; a10p 3073 '
        'r2/r_t3/r_u8; a11 3071 gA; a12/a13 '
        '3074 A_S/PAIRS4; a14 3075 MU; a15 '
        'top8; a16 hill) - the instability '
        'verdict is a REAL prompt dependence, '
        'not protocol drift.  (2) '
        'OBSERVATION/CAUSATION SPLIT: '
        'spearman(|DAH34_MED|) across '
        'families = 0.825 (A-B) / 0.685 (A-C) '
        '/ 0.529 (B-C) - the observational '
        'single-head readout spectrum is '
        'cross-prompt STABLE; spearman(r1_'
        'all32) = 0.676 / 0.130 / -0.088 - '
        'the causal single-head swap spectrum '
        'is prompt-SPECIFIC.  (3) FOCAL-8 '
        'DRIFT: A=[20,7,1,14,26,0,2,24] '
        'B=[14,7,10,24,26,2,15,11] '
        'C=[1,20,13,25,0,19,4,14]; overlaps '
        '5/4/1 of 8; the ONLY head in all '
        'three focal sets is h14; capture8 '
        'stays high in all families '
        '(0.836/0.735/0.822).  (4) STRUCTURE '
        'SIGNATURE UNSTABLE: structure_stable '
        '= [A True, B False, C False].  A: '
        'hill double-gate pass (a=0.714 '
        'b=0.454 g=1.186 bit = 3074), mu2 0 '
        'positive, max mu(>=4) +0.345 - '
        '3074/3075 capacity law + dual-zone '
        'EXACT replication.  B: hill u8 pass '
        'but all-32 FAIL (rel err 1.415 >> '
        '0.05); mu2 0 positive; viol 53.0 '
        'percent (highest); STRONG-COMPETITION '
        'failure mode (sum of single amps '
        'x_all32 0.890 >> measured all-32 y '
        '0.323, ratio 2.8x).  C: hill u8 FAIL '
        '(rel 0.353); mu2 11/28 POSITIVE '
        '(pair competition GONE); mu8 -0.016; '
        'best subset mask 191 amp 0.523 > '
        'A(255) 0.474 (partial set BEATS '
        'full-8); near-ADDITIVE failure mode '
        '(x_all32 0.797 ~ y_all32 0.742); '
        'all-32 recovery strongest of all '
        'families (R_ALL -0.742).  (5) WHAT '
        'SURVIVES: significant swap effects '
        'everywhere (R_ALL all negative), '
        'high capture8, submodular violations '
        'in every family (26.5/53.0/39.0 '
        'percent), positive high-order Moebius '
        'mass in every family (c10 '
        '0.329/0.356/0.343 all < 0.5 '
        'diffuse).  Hill params across '
        'families a=0.714/1.893/1.971 '
        'b=0.454/1.172/1.270 '
        'g=1.186/1.286/1.100.  The '
        'QUALITATIVE picture (competing '
        'writers + diffuse synergy) is '
        'cross-prompt robust; the '
        'QUANTITATIVE laws (Hill constants, '
        'mu2 all-negative, the specific focal '
        'set) are family-A specifics.  Model: '
        'READOUT FIXED (stable observational '
        'structure incl. h14) + WRITE DYNAMIC '
        '(which heads execute the swap effect '
        'is prompt-family-routed).')
    meas = {
        'meas_id': 'meas3076_omega_p73_cross_'
                   'prompt_family',
        'phase': 3076,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'family A 16 cross-phase bit '
                   'anchors all diff=0.0 (a1 3066 '
                   'ladder row 34; aref 3069 '
                   'PA/PF; a8/a9 3071 ZH34/ZH35/'
                   'DAH34/DAH35; a10s 3071 r34 '
                   'all 32; a10p 3073 r2/r_t3/'
                   'r_u8; a11 3071 gA; a12/a13 '
                   '3074 A_S/PAIRS4; a14 3075 '
                   'MU; a15 top8; a16 3074 hill '
                   'le2); per-family b-anchors '
                   '(b0 recapture, b1 sham, b3 '
                   'finite, b4 delta-x, b5 silu, '
                   'b6 assoc, b7 identity, b8 '
                   'block chain) all bit 0.0; '
                   'b0m fast-vs-brute Moebius '
                   '1.7e-16 <= 1e-12 per family',
        'artifacts': {
            'result': 'phase3076/omega_p73_'
                      'cross_prompt_family/'
                      'result.json',
            'npz': 'phase3076/omega_p73_'
                   'cross_prompt_family/'
                   'omega_p73_cross_prompt_'
                   'family.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (families '
                'sequential in one process, big '
                'banks freed between families; '
                'smoke run first into smoke/ '
                'dir).  C bodies lengthened in '
                'smoke (repV requires base >= 4 '
                'tokens) BEFORE freezing; '
                'PREREG records the signed top8 '
                'criterion (argsort ascending, '
                'NOT |r1|) and the degenerate-'
                'family handling.  Verdict tree: '
                'n_stable=1 <2 -> unstable.  '
                'Scale caveat: absolute 0.05 '
                'gates are scale-dependent (rel '
                'errors recorded); B and C fail '
                'the structure gates in '
                'DIFFERENT ways (B strong '
                'competition, C near-additive) - '
                'the merged verdict hides two '
                'failure modes',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 215
    l14['connects'].append({
        'meas_id': 'meas3076_omega_p73_cross_'
                   'prompt_family',
        'phase': 3076,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P73: cross-prompt '
                        'family replication of the '
                        'focal-head capacity law + '
                        'Moebius structure (20925 '
                        'forwards).  ZERO protocol '
                        'drift (family A 16 anchors '
                        'bit 0.0).  VERDICT '
                        'cross_prompt_unstable: '
                        'observational head spectrum '
                        'stable across families '
                        '(spearman |DAH34_MED| '
                        '0.83/0.68/0.53) but causal '
                        'swap spectrum prompt-'
                        'specific (spearman r1 '
                        '0.68/0.13/-0.09); focal-8 '
                        'overlaps 5/4/1 of 8, ONLY '
                        'h14 common to all three '
                        'families; structure_stable '
                        '= [True, False, False]: A '
                        'replicates 3074/3075 exactly '
                        '(hill 0.714/0.454/1.186, mu2 '
                        '0 pos, max mu6 +0.345), B '
                        'fails hill all-32 (strong '
                        'competition: x 0.890 >> y '
                        '0.323), C fails hill u8 AND '
                        'mu2 all-neg (11/28 positive '
                        'pairs; near-additive: x '
                        '0.797 ~ y 0.742; best subset '
                        '191 beats full-8 0.523 vs '
                        '0.474).  SURVIVES: negative '
                        'R_ALL everywhere, capture8 '
                        '0.73-0.84, violations '
                        '26.5/53.0/39.0 percent, '
                        'diffuse positive Moebius '
                        'mass (c10 < 0.5 all).  '
                        'Model: READOUT FIXED (stable '
                        'observational structure incl. '
                        'h14) + WRITE DYNAMIC (which '
                        'heads execute the swap effect '
                        'is prompt-family-routed).  '
                        'Opens 3077: A write-routing '
                        'function (what decides which '
                        'heads become focal in a '
                        'family; 3076 npz r1x3 + 3072 '
                        'OV spectra, no forwards); B '
                        'h14 deep anatomy (the unique '
                        'cross-family core); C B/C '
                        'failure-mode dissection '
                        '(free); D DS7B cross-model '
                        'head control; E intra-family '
                        'observation-causation '
                        'relation (spearman(|DAH|,'
                        'r1))'})
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
if '## Phase 3076:' not in memo:
    sec = u'''## Phase 3076: Ω-P73 焦点头跨 prompt 族稳定性——读出固定、写入漂移（cross_prompt_unstable） [%(created)s]

**判决：`cross_prompt_unstable`**（**20925 前向** / 19.8 分钟；三 prompt 族顺序执行完整 3074 协议（顺序 A/B/C 冻结）；三族 b 锚（b0 重捕获/b1 替身/b3 有限性/b4 delta-x/b5 silu/b6 结合律/b7 恒等自换/b8 块链）全 bit 0.0；族 A **16 重跨 phase 锚全 bit 0.0**——a1 3066 阶梯行 34、aref 3069 PA34/PF34、a8/a9 3071 ZH34/ZH35/DAH34_MED/DAH35_MED、a10s 3071 全 32 头 r34、a10p 3073 r2/r_t3/r_u8、a11 3071 gA、a12/a13 3074 A_S/PAIRS4、a14 3075 MU、a15 3071 top8、a16 3074 Hill le2——**协议零漂移被证明："不稳定"是真实的 prompt 依赖，不是协议漂移**；b0m 快速 vs 暴力 Möbius ≤1.7e-16 逐族）。

### 问题与设计（3076 A，3075 菜单主选）
**问题**：3074/3075 的"容量定律+双区结构"是 prompt 无关的普适规律，还是日常因果文本（族 A）的特异结构？**设计**：三个共享语法框架（8 个因果连接词 so/because/therefore/however/while/yet/although/thus × 4 前缀，仅域槽不同）但语义域不同的 prompt 族——**A=日常因果**（3074 原文重跑，作 bit 锚族）、**B=科学因果**、**C=社会情感因果**——逐族执行完整协议：banks+b 锚 → E1 24 对阶梯注入@L34（repV 3065 同构）→ E2H 头级分解 → E3 全 32 头单头扫 → 每族焦点 top8（3071 signed 判据：argsort 升序取最负，**非** |r1|）→ E4 全 255 子集联合换回扫（24 对/子集）→ E5 16472 次模不等式（tol 0.02）→ E6 容量定律 fit2（|S|≤2 的 36 点外推 S8 与 all-32，双门 0.05）→ E7 完整 Möbius 谱（暴力 3^8+快速互验）→ E8 跨族对比。预注册判决树：structure_stable(f) = hill 双门 ∧ μ2 无正 ∧ max μ(≥4)>0.02；n_stable=3 且最小 top8 重叠 ≥5/8 → cross_prompt_stable；3/3 → stable_head_shift；2/3 → partial；否则 **unstable**。C 族文本在 smoke 阶段已加长（repV 的 ridx=0..3 要求基座 ≥4 token），冻结先于权威运行。

### 核心结果（重复三遍）
**① 协议零漂移、不稳定真实（一）**：族 A 全部 16 重跨 phase 锚 diff=0.0 bit 精确——3074 容量定律（Hill [0.7143, 0.4541, 1.1857]）、3075 Möbius 谱、3071 top8 [20,7,1,14,26,0,2,24] 全部跨运行复现到 bit；因此 B/C 的差异是**真实的 prompt 依赖**。**② 观察侧稳定、因果侧漂移（二）**：观察侧单头读出谱 spearman(|DAH34_MED|) 跨族 = 0.825 (A-B) / 0.685 (A-C) / 0.529 (B-C)——稳定；因果侧单头置换谱 spearman(r1_all32) = 0.676 / 0.130 / −0.088——只有 A-B 中等，A-C/B-C 接近零或负。**读出结构是模型全局性质，因果归因是语境条件的**。**③ 焦点集合漂移、h14 唯一核心（三）**：top8 A=[20,7,1,14,26,0,2,24]、B=[14,7,10,24,26,2,15,11]、C=[1,20,13,25,0,19,4,14]；重叠 AB=5/8、AC=4/8、BC=1/8；**三族交集 = {14}——h14 是唯一跨 prompt 稳定的核心因果头**；但 capture8 三族都高（0.836/0.735/0.822）——"top8 捕获大部分负效应"定性跨族成立，具体头身份漂移。**④ 结构签名不稳定、失效模式两种**：structure_stable=[A True, B False, C False]。A：hill 双门过 + μ2 0 正 + max μ6 +0.345（3074/3075 精确复现）；B：hill u8 过但 all-32 失败（rel 1.415），μ2 0 正，viol 53.0 percent（最高），**强竞争失效**（单头幅度和 x_all32=0.890 ≫ 全换回 y_all32=0.323，2.8 倍）；C：hill u8 失败（rel 0.353）+ μ2 **11/28 正**（pair 竞争消失）+ μ8=−0.016 + 最优子集 mask 191（amp 0.523）**超过全 8 换回**（0.474），**近加性失效**（x_all32=0.797 ≈ y_all32=0.742），且全换回效应反而最强（R_ALL=−0.742）。**⑤ 幸存的不变量**：R_ALL 三族全部显著负（−0.572/−0.323/−0.742）；viol 率都高（26.5/53.0/39.0 percent）；正 Möbius 质量都弥散（c10=0.329/0.356/0.343，全部 <0.5）；Hill 参数跨族发散（a=0.714/1.893/1.971、b=0.454/1.172/1.270、γ=1.186/1.286/1.100）——**定性图景（竞争写入+弥散协同）跨族稳健，定量定律（Hill 常数、μ2 全负、特定头集合）是族 A 特异的**。

### 数学公式
- structure_stable(f) = [pass_u8(f) ∧ pass_all32(f)] ∧ [Σ_{|S|=2} 1[μ(S)>0] = 0] ∧ [max_{|S|≥4} μ(S) > TOL_MOB=0.02]；
- 判决：n_stable=1 < 2 → cross_prompt_unstable（预注册切点：min overlap≥5/8 才 stable）；
- 观察/因果分离：sp_obs = spearman(|DAH34_MED^f|, |DAH34_MED^g|) ∈ [0.53, 0.83] vs sp_cau = spearman(r1^f, r1^g) ∈ [−0.09, 0.68]；
- 失效模式：B 型 x(S)=Σm_i ≫ A(S)（竞争超线性）；C 型 x(S) ≈ A(S)（近加性）且 max_S A(S) > A(S_full)（部分集合反超全集）。

### 硬伤与边界
- 预注册门是严格切点：A-B 重叠 5/8 恰达界；若门放宽为"定性相似"，A-C（4/8）也接近——stable/unstable 是连续谱上的预注册切分，不是二元事实。
- 判决树把 B、C 两种不同失效模式合并为一个 verdict：B=强竞争（竞争超线性），C=近加性+pair 竞争消失——机制上完全不同，合并掩盖差异；3077 应分别解剖。
- C 族 L34 载体弱（PA34=0.043 vs A 0.297、B 0.357）——C 的因果链路可能更分散（更多非 L34 通路），单层 L34 注入范式对 C 的代表性较低；C 的 μ2 正系数（max +0.068）与中位估计噪声（SE ~0.02-0.04）部分重叠，μ2 全负失败的证据强度中等。
- B 的 hill all-32 失败部分是尺度效应：0.05 绝对门对小尺度族（y=0.323）更严格；rel 误差已记录（1.415）。
- 三族共享语法框架（因果连接词）——比较隔离的是语义域，不是语法；每族 32 prompt 的族内多样性有限。
- 单模型 qwen3-4b；跨模型普适性未测（DS7B 对照未做）。

### 方法论入册
- **bit 锚族设计**：原文族重跑 + 全部跨 phase 锚 bit 复现 = 协议零漂移证明；此后新族的任何差异都是真实差异——这是"现象是否 prompt 无关"的标准检验设计。
- **观察侧/因果侧双谱诊断**：同一批头级数据同时算观察谱（|DAH|）与因果谱（r1），二者分离直接定位"读出固定 vs 写入动态"。
- **signed 判据考古必要性**：top8 判据是 argsort(r1) 升序（最负优先）而非 argsort(−|r1|)——h12 案例（|r1| 排第 6 但 r1>0 不入选）证明两种判据选出不同集合；跨 phase 比较必须复刻原判据。
- 失效模式区分意识：同一名义判决（unstable）下先查失效方向（x≫y 还是 x≈y），再下机制结论。

### 智能理论洞察（第一性原理）
**读出固定、写入路由——"有限参数实现无限组合"的又一重机制。** 跨 prompt 族检验否定了"焦点头集合是固定回路"的假设，但保留了更深一层的结构：(1) 写入通道存在且三族都显著（R_ALL 全负、capture8 0.73-0.84、违反率 26-53 percent、弥散协同 c10<0.5）；(2) 观察侧头谱跨族稳定（0.53-0.83）+ 唯一跨族核心 h14——存在稳定的"侦测/读出"结构；(3) 但哪个头执行写入按语义域动态路由（B/C 的 top8 几乎换血、Hill 参数发散、C 连 pair 竞争都消失）。这不是"容量定律错了"，而是**容量定律描述的是路由选定写入器之后的竞争结构，路由本身由 prompt 族决定**。对 AGI 理论：功能组织不是"头→功能"的静态分配，而是**固定读出底座 + 动态写入路由**的两层架构——参数无需为每个语义域复制一套写入器，同一批候选头按语境重新组合；这与 3062-3065 的"语义模式族条件化"和 3075 的"完备性效应"拼合：路由决定谁上场，容量定律约束上场的怎么竞争，完备性效应约束收尾的协同。下一步缝隙：**路由函数本身**——什么条件（头的 OV 方向与域词汇的几何关系？观察谱强度？位置？）决定某个头在某族成为焦点头，这是"条件化齿轮组"的路由定律，也是当前最大的未解结构。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3076/omega_p73_cross_prompt_family/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3077 菜单**——A（主选）**写入路由函数**：三族 32 头 r1 + 族标签 + 3072 OV 谱 + 头位置放进统一预测框架，什么特征决定"哪个头在某族成为焦点头"（3076 npz 免前向为主）。B **h14 深解剖**：唯一跨族核心头——OV 方向、与 3072 写入方向共线性、DS7B 对应头。C **B/C 失效模式解剖**：B 强竞争 vs C 近加性的来源（3076 npz A_S/MU 免前向）。D **DS7B 头级对照**：跨模型检验"读出固定+写入路由"架构。E **族内观察-因果关系**：spearman(|DAH|, r1) 逐族——观察谱为什么预测不了因果谱（3076 npz 免前向）。"好的，继续"即进 3077 A。
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
if '## 三十八、3076 增补' not in aud:
    add = u'''
---
## 三十八、3076 增补：焦点头跨 prompt 族稳定性（Omega-P73，判决 cross_prompt_unstable）
1. **协议零漂移 + 真实不稳定**：族 A 16 重跨 phase 锚 bit 0.0（3074 Hill/3075 谱/3071 top8 全复现）；族 B/C 结构门失败——B hill all-32 失败（强竞争：单头和 0.890 ≫ 全换回 0.323），C hill u8 失败且 μ2 11/28 正（近加性：x 0.797≈y 0.742；最优子集 191 反超全 8：0.523 vs 0.474）。
2. **读出固定、写入漂移**：观察侧 |DAH34_MED| 谱跨族 spearman 0.83/0.68/0.53 稳定；因果侧 r1 谱 0.68/0.13/−0.09 漂移；top8 重叠 5/4/1，三族唯一共同头 h14；capture8 0.73-0.84、R_ALL 全负、viol 26.5/53.0/39.0 percent、c10<0.5 跨族幸存。
3. HDMCC 修正：**容量定律是"路由选定写入器之后"的竞争结构定律，路由本身 prompt 族条件化**——"读出固定+写入动态路由"两层架构入册；bit 锚族（原文族重跑）= 检验现象 prompt 无关性的标准设计；signed top8 判据（argsort 升序）必须考古复刻。
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
if 'Phase 3076' not in prev:
    line = ('- Phase 3076 Omega-P73 cross-prompt '
            'family (20925 forwards, 19.8 min, '
            'families A/B/C sequential full 3074 '
            'protocol): verdict '
            'cross_prompt_unstable.  Family A: '
            'ALL 16 cross-phase anchors bit 0.0 '
            '(protocol zero-drift proven; 3074 '
            'hill/3075 spectrum/3071 top8 exact). '
            'B: top8=[14,7,10,24,26,2,15,11], hill '
            'all-32 FAIL (strong competition x '
            '0.890 >> y 0.323), mu2 0 pos, viol '
            '53.0 percent.  C: top8=[1,20,13,25,'
            '0,19,4,14], hill u8 FAIL, mu2 11/28 '
            'POSITIVE (pair competition gone), '
            'near-additive (x 0.797 ~ y 0.742), '
            'best subset 191 beats full-8 (0.523 '
            'vs 0.474).  Observation/causation '
            'split: spearman(|DAH|) '
            '0.83/0.68/0.53 stable vs '
            'spearman(r1) 0.68/0.13/-0.09 drift; '
            'top8 overlaps 5/4/1, only h14 '
            'common; capture8 0.73-0.84, R_ALL '
            'all negative, c10<0.5 survive.  '
            'Model: READOUT FIXED + WRITE ROUTED '
            'by prompt family.  Audit 38; ledger '
            '215/L14 183.\n')
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
if 'max=3076' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（link_id=L14_readout_spectrum_cross_model；verify 需 isinstance 防御）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃/smoke 推翻均如实登记；verdict 单分支赋值。
4. 统计纪律：阈值预注册；交互/组合分析在效应量同尺度（中位簿记）；绝对门有尺度依赖，rel 误差须记录。

## 标准锚与精度
- bit 锚家族：标量行（a1）、因果置换、跨 phase 参考数（aref）、块链恒等（b8）、hook 互证、跨 phase 因果复现（a10）、嵌入式家族锚、枚举重放锚（3075）、**bit 锚族（3076：原文族重跑 16 锚 bit=协议零漂移证明）**。
- 跨路径 bit 锚需匹配浮点求和顺序；不同算法等价性 ≤1e-12 门（b0/b0m）并 PREREG 注明。
- 3070 ATTN 换回=pre-hook 末位；3071 per-head=头切片；3072 谱系恒等式；3073 中位簿记；3074 预算参数化+双门外推；3075 Möbius 谱；**3076 跨 prompt 族协议（三族顺序冻结；每族 top8 独立选择；signed argsort 升序判据，非 |r1|）**；repV ridx=0..3 → 基座必须 ≥4 token。

## 机制解释审计链（命名前依次检查）
…→3073 共享 TT 通道+集合函数→3074 容量定律（Hill 0.714/0.454/1.186；26.5 percent 违反）→3075 超模弥散（μ2 全负 pair 竞争；μ6/μ8 高阶补全；双区）→**3076 cross_prompt_unstable：族 A 16 锚 bit 复现（协议零漂移）；观察侧 |DAH| 谱跨族稳定（spearman 0.53-0.83）、因果侧 r1 谱漂移（0.68/0.13/−0.09）；top8 重叠 5/4/1，唯一跨族核心=h14；B 强竞争（x 0.890≫y 0.323）、C 近加性（x 0.797≈y 0.742）且 μ2 11/28 正、最优子集反超全 8；幸存：R_ALL 全负、capture8 0.73-0.84、viol 26.5/53.0/39.0 percent、c10<0.5。模型：读出固定+写入按 prompt 族动态路由**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→run_log 用 Read；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查。
- smoke 小样本可隐藏退化（n_neg<8 → NT<NM_ALL）→ MASKS/枚举用 NM_ALL=1<<NT 泛化；基座 token 数断言 ≥4。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3076）
Ω-P2（3011-3076）：…3071 attn_heads_focal；3072 focal_lineage_full；3073 higher_order_required；3074 capacity_law_hill；3075 supermodular_diffuse；**3076 cross_prompt_unstable（读出固定+写入路由）**。

## 下一步
- max=3076，下一个 3077（A 主选 **写入路由函数**：什么决定哪个头在某族成为焦点头——3076 npz r1_all32×3 + 3072 OV 谱免前向；B h14 深解剖（唯一跨族核心头）；C B/C 失效模式解剖（免前向）；D DS7B 头级对照；E 族内观察-因果关系 spearman(|DAH|,r1) 免前向）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3076')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
