# -*- coding: utf-8 -*-
"""Phase 3077 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3077'
     r'\omega_p74_write_routing')
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
assert verdict == ('observation_causation_'
                   'decoupled'), verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
assert res['forwards'] == 0
st = res['stats']
an = st['anchors']
for k in ('a1', 'a2', 'a3', 'a4', 'a5', 'a6',
          'a7'):
    assert an[k + '_ok'] is True, k
    assert an[k + '_diff'] == 0.0, k
assert st['n_neg'] == {'A': 16, 'B': 20, 'C': 17}
ic = st['icc']['R']
assert abs(ic['icc_head'] - 0.4515411352114762) \
    < 1e-12, ic
assert ic['v_head'] > 0 and ic['p_perm'] <= 0.01
assert abs(st['icc']['absD34']['icc_head']
           - 0.8654809849797939) < 1e-12
sf = st['sign_flip']
assert abs(sf['delta'] - 0.046373717740786566) \
    < 1e-12
assert sf['p_perm'] < 0.05
fstats = st['features']
assert len(fstats) == 12
byid = {e['id']: e for e in fstats}
assert abs(byid['g5']['mean_abs_sp']
           - 0.4034090909090909) < 1e-12
assert byid['g5']['mean_abs_sp'] < 0.6
assert byid['g5']['mean_ov'] == 4.5
for e in fstats:
    assert e['mean_abs_sp'] < 0.6, e['id']
assert st['best_feature']['id'] == 'g5'
g = st['gates']
assert g['G1'] is False and g['G2'] is False
assert g['G3'] is True and g['G4'] is False
assert abs(g['mean_intra_sp']
           - (-0.1520039100684262)) < 1e-12
assert st['alpha_top8'] == [1, 7, 14, 20, 26, 0,
                            24, 2]
assert st['alpha_overlap'] == {'A': 8, 'B': 5,
                               'C': 4}
assert st['eps_overlap']['A']['overlap'] == 7
assert st['eps_overlap']['B']['overlap'] == 2
assert st['eps_overlap']['C']['overlap'] == 4
assert abs(st['sp_alpha_meand34']
           - (-0.1689882697947214)) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3077
           for m in led['measurements']):
    claim = (
        'Omega-P74 (plan 3077 A) - qwen3-4b '
        'write-routing function search (NO '
        'forwards, 0.9s): what decides which '
        'heads become FOCAL (top-8 most-negative '
        'causal swap effect) in a prompt '
        'family?  Frozen-npz re-analysis of '
        '3076 (R1_ALL32_A/B/C 3x32 causal swap '
        'spectra) + 3071 (DOH/DHH/R34/R35/C34H) '
        '+ 3072 (M34_MED/COS_MED_H/ATT34B/'
        'ATT34I) + 3073 (I_PAIR involvement); 7 '
        'cross-source bit anchors (a1 R1_A=R34; '
        'a2/a3 DAH34_MED_A = 3071 = 3072; a4 '
        'CS1H_A=C34H; a5 R1 = median(CS1H)-'
        'med_c replay bit; a6 top8 replay; a7 '
        'result.json consistency).  VERDICT '
        'observation_causation_decoupled.  (1) '
        'ALL 12 PREREGISTERED FEATURES FAIL: '
        'observational spectra (|DAH34_MED|/|'
        'DAH35_MED| cross-family mean/max/min), '
        '3071/3072 geometry (|DOH34|, |DHH34|, '
        '|M34|), head position (idx, GQA group, '
        'quarter) - mean |spearman| vs the '
        'causal spectrum <= 0.20 everywhere '
        '(top8 overlap 1.67-4.0 vs random 2.0); '
        'best = g5 (family-A causal prior '
        'R1_A, source-family-excluded '
        'evaluation): mean|sp| = 0.403 (B '
        '0.676, C 0.130) < 0.6 gate, mean top8 '
        'overlap 4.5 < 5 - NO global head '
        'feature explains focality.  (2) '
        'OBSERVATION-CAUSATION DECOUPLING: '
        'intra-family spearman(|DAH34_MED|, '
        'R1) = -0.291 (p 0.104) / -0.130 (0.479)'
        ' / -0.035 (0.852) - all weak-negative '
        'and nonsignificant; 3071/3072 '
        'geometry on family A equally weak '
        '(|M34| -0.281, involvement -0.289, |'
        'DHH34| -0.199, |DOH34| -0.180, '
        'COS_MED -0.192); attention features '
        'negligible (injection-region attention '
        'vs R1_A +0.120).  (3) VARIANCE '
        'DECOMPOSITION: two-way ANOVA on the '
        '3x32 R matrix gives ICC_head = 0.452 '
        '(permutation p 0.0001) vs ICC_head(|'
        'DAH34_MED|) = 0.865 - the OBSERVATIONAL '
        'spectrum is 87 percent stable head '
        'identity, the CAUSAL spectrum only 45 '
        'percent (rest is head x family '
        'interaction); sign-flip test: focal '
        'heads in their OWN family have R1 '
        '-0.075 vs -0.029 in other families '
        '(delta 0.046, permutation p 0.0096).  '
        '(4) FOCALITY LIVES IN THE INTERACTION: '
        'head main-effect top8 [1,7,14,20,26,0,'
        '24,2] overlaps family top8 A 8/8 (A '
        'amplitude dominates the mean) but B '
        '5/8 C 4/8; after removing the head '
        'main effect the interaction-residual '
        'top8 overlaps A 7, B 2, C 4 - family '
        'A focality is mostly main effect + '
        'interaction, B/C focality is '
        'interaction-driven.  (5) MIGRATION '
        'ASYMMETRY: R1_A predicts R1_B at '
        'spearman 0.676 but R1_C at 0.130 - '
        'everyday and science causality share '
        'routing structure, social-emotional '
        'does not.  SMOKE correction frozen '
        'before the authoritative run: '
        'source-family features (g4-g6, g11-'
        'g12) evaluated on non-source families '
        'only (g5 on A would be the identity; '
        'uncorrected smoke had spuriously '
        'passed G1/G2 via the 1.0 '
        'self-prediction).  Model: routing is '
        'NOT a lookup of static head geometry; '
        'a real but weak cross-family stable '
        'writer component exists (45 percent '
        'of causal variance, perm p 0.0001) '
        'but does NOT determine focality; '
        'which heads execute the write is '
        'decided by context-level dynamics '
        'outside the searched feature space.')
    meas = {
        'meas_id': 'meas3077_omega_p74_write_'
                   'routing',
        'phase': 3077,
        'claim': claim,
        'verdict': verdict,
        'anchors': '7 cross-source bit anchors '
                   'diff=0.0 (a1 R1_ALL32_A vs '
                   '3071 R34; a2 DAH34_MED_A vs '
                   '3071; a3 vs 3072; a4 CS1H_A '
                   'vs 3071 C34H; a5 R1 = '
                   'median(CS1H)-med_c replay '
                   'all 3 families; a6 top8 '
                   'replay all 3; a7 3076 '
                   'result.json consistency); '
                   'ANOVA decomposition identity '
                   'reconstruction; permutation '
                   'rng default_rng(3020) '
                   'frozen; hypergeometric '
                   'top8-overlap tails',
        'artifacts': {
            'result': 'phase3077/omega_p74_'
                      'write_routing/'
                      'result.json',
            'npz': 'phase3077/omega_p74_'
                   'write_routing/omega_p74_'
                   'write_routing.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (NO '
                'forwards; frozen npz '
                're-analysis, 0.9s).  Smoke '
                'found and fixed the '
                'self-prediction leak BEFORE '
                'freezing: source-family '
                'features are excluded from '
                'their own family evaluation '
                '(evaluation_rule in PREREG).  '
                'ATT last-position feature is '
                'all-NaN padding (0 valid '
                'heads) and reported NA; '
                'injection-region attention '
                'carries no routing signal '
                '(+0.120).  Caveat: 3075 '
                'spearman -0.738 was computed '
                'WITHIN the focal-8 on m_i, '
                'not comparable to the full-32 '
                'intra-family correlations '
                'here (different population, '
                'no contradiction)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 216
    l14['connects'].append({
        'meas_id': 'meas3077_omega_p74_write_'
                   'routing',
        'phase': 3077,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P74: write-routing '
                        'function search (no '
                        'forwards).  ALL 12 '
                        'preregistered head '
                        'features fail to '
                        'predict focality '
                        '(mean|sp| <= 0.20; '
                        'best causal prior g5 '
                        '0.403 with migration '
                        'asymmetry B 0.676 vs C '
                        '0.130).  '
                        'Observation-causation '
                        'decoupling quantified: '
                        'ICC_head 0.865 '
                        '(observational |DAH|) '
                        'vs 0.452 (causal R1, '
                        'perm p 0.0001); '
                        'sign-flip of focal '
                        'heads significant '
                        '(delta 0.046, p '
                        '0.0096); focality '
                        'lives in head x family '
                        'interaction (eps-top8 '
                        'overlaps A 7/B 2/C 4).  '
                        'Model: READOUT FIXED '
                        '(head identity) + '
                        'WRITE ROUTED (context '
                        'dynamics outside '
                        'static geometry); a '
                        'weak stable background '
                        'writer exists but '
                        'does not decide '
                        'focality.  Opens 3078: '
                        'A routing-decision '
                        'timing (layer-wise '
                        'head-level capture '
                        'L20->L34, ~2400 '
                        'forwards); B A_S '
                        'spectral-shape '
                        'similarity network (no '
                        'forwards); C h14 '
                        'cross-family anatomy; '
                        'D DS7B head-level '
                        'control'})
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
if '## Phase 3077:' not in memo:
    sec = u'''## Phase 3077: Ω-P74 写入路由函数——观察-因果解耦，焦点性住在头×族交互（observation_causation_decoupled） [%(created)s]

**判决：`observation_causation_decoupled`**（**免前向**：对冻结 npz 纯重分析，0.9 秒；**7 重跨源锚全 bit 0.0**——a1 R1_ALL32_A = 3071 R34、a2/a3 DAH34_MED_A = 3071 = 3072、a4 CS1H_A = 3071 C34H、a5 R1 = median(CS1H)−med_c 三族重放 bit、a6 top8 三族重放、a7 与 3076 result.json 一致性；ANOVA 分解恒等式重构；置换检验 rng(3020) 冻结、top8 重叠用超几何尾 p）。勘误（smoke 发现、权威前冻结修正）：**来源族特征不得在来源族上评估**——g5（族 A 因果先验）在族 A 上是恒等自预测（sp=1.0），未修正的 smoke 曾以 1.0 自预测污染 mean|sp|=0.602 虚假压线过 G1/G2；修正后 g5 只在 B/C 评估（evaluation_rule 入 PREREG）。

### 问题与设计（3077 A，3076 菜单主选）
**问题**：3076 证明焦点头集合跨 prompt 族漂移（top8 重叠 5/4/1、唯一共同头 h14）——那么**什么决定"哪个头在某族成为焦点头"**？是头的静态几何（观察谱/OV 范数/位置），还是语境级动力学？**设计**：对 96 个（族×头）因果观测，四路免前向分析：E2 族内观察-因果相关（spearman+10000 次置换 p）；E3 双因子（头×族）无重复方差分解 ICC + 符号翻转检验（焦点头 own vs other 族）；E4 **12 个预注册特征竞标**（g1-g3/g10 观察谱跨族统计、g4-g6/g11-g12 族 A 测量几何 [DOH/DHH/M34/因果先验]、g7-g9 头位置；方向预注册；来源族特征只在非来源族评估）；E5 稳定分量（α 头主效应）与交互项（ε）的焦点性归因；E6 族 A 注意力特征（ATT34I/B，NaN 防御）。判决门（预注册）：G1 best 特征 mean|sp|≥0.6；G2 mean top8 重叠≥5/8；G3 族内 sp(|D34|,R1) 的 |mean|<0.3（解耦）；G4 ICC≥0.5。

### 核心结果（重复三遍）
**① 12 特征竞标全部失败（一）**：观察谱（|D34|/|D35| 的跨族 mean/max/min）、3071/3072 几何（|DOH34|、|DHH34|、|M34|）、头位置（idx/GQA/quarter）预测因果谱的 mean|spearman| **全部 ≤0.20**（top8 重叠 1.67-4.0，随机基线 2.0）；最佳 g5（族 A 因果先验 R1_A，剔除自预测）mean|sp|=0.403 <0.6、mean 重叠 4.5 <5——**"哪个头在某族成为焦点头"不能由任何静态头特征解释**。**② 观察-因果解耦定量化（二）**：族内 sp(|DAH34_MED|, R1) = −0.291 (p=0.104) / −0.130 (p=0.479) / −0.035 (p=0.852)——全部弱负且不显著；族 A 的 3071/3072 几何同样弱（|M34| −0.281、头参与度 −0.289、|DHH34| −0.199、|DOH34| −0.180、COS_MED −0.192）；注意力特征无路由信号（注入区注意力 vs R1_A +0.120）。**③ 方差分解：读出固定、写入动态的定量版（三）**：R 矩阵 ICC_head = **0.452**（置换 p=0.0001）vs |DAH34_MED| 的 ICC_head = **0.865**——观察谱 87 percent 方差是头固有分量，因果谱只有 45 percent（其余是头×族交互）；符号翻转检验：焦点头在自己族 R1 均值 −0.075 vs 其他族 −0.029（delta=0.046，p=0.0096）。**④ 焦点性住在交互项**：头主效应 α_top8=[1,7,14,20,26,0,24,2] 与族 A top8 重叠 8/8（A 幅度主导均值）但 B 5/C 4；去掉主效应后的交互残差 ε_top8 与真实 top8 重叠 A 7/B 2/C 4——**A 族焦点性主要在主效应+交互，B/C 族焦点性主要在交互**。**⑤ 迁移不对称**：R1_A 预测 R1_B spearman=0.676（中等）但 R1_C=0.130（无）——日常因果与科学因果共享路由结构，社会情感不共享。

### 数学公式
- 双因子分解：R[f,i] = μ + α_i + β_f + ε_fi；ICC_head = σ²(α)/(σ²(α)+σ²(ε))；R 的 ICC=0.452（p=0.0001）、|D34| 的 ICC=0.865；
- 符号翻转统计：Δ = mean_other(R1) − mean_own(R1) = 0.046（头内置换 p=0.0096）；
- 竞标评估：mean|sp| 在 eval_f（来源族特征=非来源族）上；门 G1≥0.6、G2≥5/8；
- 超几何尾：P(overlap≥k)，N=32、K=n=8。

### 硬伤与边界
- 3 族样本的 ICC 族自由度仅 2，CI 宽；0.45 vs 0.5 门是预注册切点。
- best-of-12 的置换 p 有选择乐观偏差（但 G1/G2 均未过，无 selection 风险）。
- 族 A-only 特征（g4-g6/g11-g12）是"族 A 的 24 对上测量的头级中位"——对 B/C 的检验是泛化假设，不是头的固有几何的独立测量。
- g5 在 B 的 0.676 与 3076 的 sp_r1_AB 是同一数字——迁移不是新信息，竞标中它作"因果先验"上界参照。
- **3075 的 −0.738 与本 phase 的 −0.29 不矛盾**：口径不同——3075 是焦点 8 头内部 |DAH| vs m_i（选择后小总体），本 phase 是全部 32 头 |DAH| vs R1；"焦点头内部 OV 投射越小因果越强"未被推翻，是不同总体上的不同统计量。
- ATT 特征 50 percent NaN（右 padding），last 位无有效头（NA），仅注入区可用且弱；语义域仅 3 个，"域→路由"映射规律需更多域。

### 方法论入册
- **来源族自预测排除规则**：跨族竞标中，在族 X 上测量的特征不得在 X 上评估（否则恒等自预测污染门）——smoke 阶段发现、权威前冻结。
- **ICC 分解 = "读出固定 vs 写入动态"的定量诊断**：对同一批头同时算观察谱与因果谱的双因子方差分解，两个 ICC 之差（0.865 vs 0.452）直接量化解耦。
- **符号翻转检验**：焦点头 own vs other 族的头内置换——把"路由"操作化为可检验的显著量。
- 免前向 phase 的第 4 次使用（3075/3077 等）：零协议漂移、全锚 bit、秒级完成。

### 智能理论洞察（第一性原理）
**路由不在参数几何里——它是语境级动力学。** 12 个静态特征（观察谱、OV 范数、输入范数、头参与度、注意力、位置）全部无法预测"谁上场"（mean|sp|≤0.20）；唯一的迁移来自因果谱自身（A→B 0.676、A→C 0.130，且不对称）。结合 3076："读出固定+写入路由"获得定量内容：**读出结构是头的固有性质（ICC 0.87），写入配置是头×语境的交互（55 percent 交互+残差方差；符号翻转显著 p=0.0096）**。同时存在真实但弱的"背景写入者"分量（45 percent 头主效应方差，置换 p=0.0001）——它决定不焦点性，但说明写入通道有常驻底座；焦点写入者像"按项目借调"，背景写入者像"常驻员工"。对 AGI 理论：条件化齿轮组的**装配函数不是静态查表**——语境通过某种尚未定位的信号实时选择写入头；这个信号不在头的参数几何里，暗示它住在**上游表示的内容**中（L34 之前各层的信息流已"决定"了头的响应模式）。下一步缝隙：路由决定的时间/层定位——路由信息在哪一层变得可读？这需要新的前向捕获（逐层头级输入/输出演变）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3077/omega_p74_write_routing/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3078 菜单**——A（主选）**路由决定时序定位**：三族逐层（L20→L34）头级输入/输出捕获，检验路由信息在哪一层变得可读（头级因果贡献的族特异性出现层；~2400 前向）。B **A_S 谱形相似性网络**：三族 255 维谱形两两+对 3074，域距离 vs 结构相似（免前向）。C **h14 全域解剖**：唯一跨族核心头的逐族谱形/OV top tokens/DOH/DHH（免前向为主）。D **DS7B 头级对照**：跨模型检验"读出固定+写入路由"。"好的，继续"即进 3078 A。
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
if '## 三十九、3077 增补' not in aud:
    add = u'''
---
## 三十九、3077 增补：写入路由函数搜索（Omega-P74，判决 observation_causation_decoupled）
1. **12 特征竞标全败**：观察谱/3071-3072 几何/位置/注意力的 mean|sp| 全部 ≤0.20（top8 重叠≈随机）；最佳 g5（族 A 因果先验，剔除自预测）mean|sp|=0.403<0.6、迁移不对称 B 0.676 vs C 0.130——"谁成为焦点头"不由任何静态头特征解释。
2. **解耦定量化**：ICC_head(|DAH|)=0.865 vs ICC_head(R1)=0.452（置换 p=0.0001）；符号翻转显著（own −0.075 vs other −0.029，p=0.0096）；焦点性住在头×族交互（ε-top8 重叠 A 7/B 2/C 4）；存在弱背景写入者分量（45 percent，不决定焦点性）。
3. HDMCC 修正：**来源族自预测排除规则**入册（跨族竞标纪律）；**ICC 双谱分解**=读出固定 vs 写入动态的定量诊断；3075 的 −0.738（焦点 8 头内 m_i 口径）与 3077 的 −0.29（全 32 头 R1 口径）不矛盾——总体不同。
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
if 'Phase 3077' not in prev:
    line = ('- Phase 3077 Omega-P74 write-routing '
            'search (NO forwards, 0.9s, frozen npz '
            're-analysis): verdict '
            'observation_causation_decoupled.  7 '
            'cross-source anchors bit 0.0.  ALL 12 '
            'preregistered head features fail '
            '(mean|sp| <= 0.20; best g5 family-A '
            'causal prior 0.403 after '
            'source-family self-prediction '
            'exclusion - smoke leak fixed before '
            'freezing; migration asymmetry B 0.676 '
            'vs C 0.130).  ICC_head(|DAH|)=0.865 '
            'vs ICC_head(R1)=0.452 (perm p '
            '0.0001); sign-flip delta 0.046 p '
            '0.0096; focality lives in head x '
            'family interaction (eps-top8 overlap '
            'A 7/B 2/C 4); weak background-writer '
            'component real (45 percent) but does '
            'not decide focality.  Model: READOUT '
            'FIXED (head identity) + WRITE ROUTED '
            '(context dynamics outside static '
            'geometry).  Audit 39; ledger '
            '216/L14 184.\n')
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
if 'max=3077' not in mem_cur:
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
4. 统计纪律：阈值预注册；绝对门有尺度依赖，rel 误差须记录；**跨族竞标来源族特征不得自评（3077 规则）**。

## 标准锚与精度
- bit 锚家族：标量行（a1）、因果置换、跨 phase 参考数（aref）、块链恒等（b8）、hook 互证、跨 phase 因果复现（a10）、嵌入式家族锚、枚举重放锚（3075）、bit 锚族（3076 原文族重跑 16 锚）、**跨源一致性锚（3077：R1=median(CS1H)−med_c 重放、同数据多 phase 副本 bit 互证）**。
- 跨路径 bit 锚需匹配浮点求和顺序；不同算法等价性 ≤1e-12 门；置换检验 rng(3020) 冻结。
- 3071 per-head=头切片；3073 中位簿记；3074 预算参数化+双门；3075 Möbius 谱；3076 跨 prompt 族协议（signed argsort 判据）；**3077 双因子 ANOVA/ICC 分解+符号翻转检验+特征竞标协议**；repV 基座 ≥4 token。

## 机制解释审计链（命名前依次检查）
…→3074 容量定律→3075 超模弥散→3076 cross_prompt_unstable（读出固定+写入路由假说）→**3077 observation_causation_decoupled：12 静态特征全败（mean|sp|≤0.20）；ICC(|DAH|)=0.865 vs ICC(R1)=0.452（p=0.0001）；符号翻转显著（p=0.0096）；焦点性在头×族交互；弱背景写入者存在（45 percent）但不决定焦点性；迁移不对称 A→B 0.676 / A→C 0.130。路由=语境级动力学，不在静态参数几何；下一步=路由决定时序/层定位**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→run_log 用 Read；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查。
- NaN 数据（ATT padding）：nanmedian 全 NaN 列→argsort 污染 spearman，必须有效掩码过滤后计算。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3077）
Ω-P2（3011-3077）：…3071 attn_heads_focal；3072 focal_lineage_full；3073 higher_order_required；3074 capacity_law_hill；3075 supermodular_diffuse；3076 cross_prompt_unstable；**3077 observation_causation_decoupled（路由=语境动力学）**。

## 下一步
- max=3077，下一个 3078（A 主选 **路由决定时序定位**：三族逐层 L20→L34 头级捕获，路由信息哪层可读，~2400 前向；B A_S 谱形相似性网络（免前向）；C h14 全域解剖（免前向为主）；D DS7B 头级对照）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3077')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
