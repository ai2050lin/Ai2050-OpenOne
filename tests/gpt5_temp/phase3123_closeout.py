# -*- coding: utf-8 -*-
"""Phase 3123 closeout (idempotent):
result asserts -> Ledger -> MEMO Phase 3123 ->
workspace logs (x2 entries) -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3123'
        r'\omega_p121_dirfit_anchor_l35loc_'
        'syntax_trace')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
o = []

V = ('dirfit_failed|anchor_failed|'
     'pit_calibrated|pit_calibrated|'
     'anchor_separation_present|anchor_unreliable|'
     'final_brake_global|'
     'assertion_write_global_positive|'
     'replay_bit_exact|syntax_emerges_L21|'
     'syntax_emerges_L21|content_emerges_L20|'
     'content_emerges_L20')

# ---------- 1. result.json asserts ----------
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
_n = [0]


def chk(cond):
    assert cond, 'assert #%d failed' % len(_n)
    _n.append(1)


chk(res['phase'] == 3123)
chk(res['name'] == 'omega_p121_dirfit_'
    'anchor_l35loc_syntax_trace')
chk(res['verdict'] == V)
chk(res['smoke'] is False)
chk(res['n_pairs'] == 672)
chk(res['np_a'] == 672)
chk(abs(res['runtime_s'] - 230.4) < 0.05)
pa = res['part_a']
dfP = pa['dirfit']['P']
chk(abs(dfP['S'] - (-0.601611613690753)) < 1e-12)
chk(abs(dfP['MS'] - (-4.955594255793292)) < 1e-12)
chk(abs(dfP['resid_std']
        - 2.9407542416065002) < 1e-12)
chk(abs(dfP['anchor_mean']
        - (-4.9555942557933035)) < 1e-12)
chk(abs(dfP['anchor_std']
        - 1.1980504827646983) < 1e-12)
chk(abs(dfP['rel_r']
        - 0.15372006425624055) < 1e-12)
dfA = pa['dirfit']['A1']
chk(abs(dfA['S'] - (-0.5540798866844862)) < 1e-12)
chk(abs(dfA['MS'] - (-7.154421990932929)) < 1e-12)
chk(abs(dfA['resid_std']
        - 3.581853375490055) < 1e-12)
chk(abs(dfA['anchor_mean']
        - (-7.154421990932932)) < 1e-12)
chk(abs(dfA['anchor_std']
        - 1.217143626768722) < 1e-12)
chk(abs(dfA['rel_r']
        - 0.10384567447126661) < 1e-12)
an = pa['anchors']
chk(abs(an['gap'] - 2.198827735139629) < 1e-12)
chk(an['verdict_sep'] == 'anchor_separation_present')
chk(an['verdict_rel'] == 'anchor_unreliable')
sm = pa['sim']
chk(abs(sm['r_dir'] - 0.2285152966483294) < 1e-12)
chk(sm['dirfit_verdict'] == 'dirfit_failed')
chk(abs(sm['r_anchor']
        - 0.23528388458123164) < 1e-12)
chk(sm['anchor_verdict'] == 'anchor_failed')
chk(abs(sm['pit_ks_dir']
        - 0.03650545634920632) < 1e-12)
chk(sm['pit_dir_verdict'] == 'pit_calibrated')
chk(abs(sm['pit_ks_anchor']
        - 0.03262400793650794) < 1e-12)
chk(sm['pit_anchor_verdict'] == 'pit_calibrated')
ad = sm['auc_sim_dir']
chk(abs(ad[0] - 0.9809094210600907) < 1e-12)
chk(abs(ad[1] - 0.8437170075777707) < 1e-12)
chk(abs(ad[4] - 0.6342085000243587) < 1e-12)
chk(abs(ad[12] - 0.6635588242718963) < 1e-12)
aa = sm['auc_sim_anchor']
chk(abs(aa[1] - 0.8417776765230832) < 1e-12)
chk(abs(aa[12] - 0.6573411321038832) < 1e-12)
chk(sm['n_reps'] == 200)
pb = res['part_b']
l35 = pb['l35']
chk(abs(l35['ans_mean_pooled']
        - (-9.701520158776216)) < 1e-12)
chk(abs(l35['oth_mean_pooled']
        - (-11.597647604782505)) < 1e-12)
chk(l35['verdict'] == 'final_brake_global')
p35 = l35['per_dir']['P']
chk(abs(p35['ans_mean']
        - (-11.373681399084273)) < 1e-12)
chk(abs(p35['oth_mean']
        - (-11.479271332374513)) < 1e-12)
chk(p35['n_ans'] == 672 and p35['n_oth'] == 7392)
a35 = l35['per_dir']['A1']
chk(abs(a35['ans_mean']
        - (-8.029358918468157)) < 1e-12)
chk(abs(a35['oth_mean']
        - (-11.716023877190498)) < 1e-12)
lq = pb['l30q']
chk(abs(lq['q_mean_pooled']
        - 1.482873646914959) < 1e-12)
chk(abs(lq['nq_mean_pooled']
        - 1.941596696649045) < 1e-12)
chk(lq['verdict']
    == 'assertion_write_global_positive')
L30 = lq['per_layer']['L30']
chk(abs(L30['q_sum'] - 53.924607276916504) < 1e-9)
chk(L30['q_n'] == 70 and L30['nq_n'] == 16058)
chk(abs(L30['nq_sum'] - 30501.84459859878) < 1e-6)
chk(abs(L30['q_mean_per_dir'][0]
        - 0.7324254729531028) < 1e-12)
chk(abs(L30['q_mean_per_dir'][1]
        - 1.3961315155029297) < 1e-12)
chk(abs(L30['nq_mean_per_dir'][0]
        - 2.2729591248465213) < 1e-12)
chk(abs(L30['nq_mean_per_dir'][1]
        - 1.5288731412005336) < 1e-12)
L32 = lq['per_layer']['L32']
chk(abs(L32['q_sum'] - 153.67770329117775) < 1e-9)
chk(abs(L32['nq_sum'] - 31854.474910981953) < 1e-6)
chk(abs(L32['q_mean_per_dir'][0]
        - 2.2533089527578065) < 1e-12)
chk(abs(L32['q_mean_per_dir'][1]
        - 1.2398281022906303) < 1e-12)
chk(abs(L32['nq_mean_per_dir'][0]
        - 1.9923724996144472) < 1e-12)
chk(abs(L32['nq_mean_per_dir'][1]
        - 1.9751215457897773) < 1e-12)
pc = res['part_c']
chk(pc['repro']['max_diff'] == 0.0)
chk(pc['repro']['verdict'] == 'replay_bit_exact')
chk(pc['n_span'] == {'P': 305, 'A1': 320})
chk(pc['lstar'] == {'P': {'syn_L': 21,
                          'cont_L': 20},
                    'A1': {'syn_L': 21,
                           'cont_L': 20}})
cv = pc['curves']
chk(abs(cv['E_syn_P'][17]
        - 0.7791171511372582) < 1e-12)
chk(abs(cv['E_syn_P'][21]
        - (-0.16813003433532403)) < 1e-12)
chk(abs(cv['E_syn_P'][32]
        - (-1.7999306065723555)) < 1e-12)
chk(abs(cv['E_syn_P'][36]
        - (-2.2558336193932855)) < 1e-12)
chk(abs(cv['E_syn_A1'][21]
        - (-0.27427328491117814)) < 1e-12)
chk(abs(cv['E_syn_A1'][36]
        - (-1.7065305865835405)) < 1e-12)
chk(abs(cv['E_cont_P'][0]
        - 17.43467748180765) < 1e-9)
chk(abs(cv['E_cont_P'][15]
        - (-2.3295825219496358)) < 1e-12)
chk(abs(cv['E_cont_P'][20]
        - (-1.421478146881353)) < 1e-12)
chk(abs(cv['E_cont_P'][28]
        - (-4.157009344511345)) < 1e-12)
chk(abs(cv['E_cont_P'][36]
        - (-3.6768157212265207)) < 1e-12)
chk(abs(cv['E_cont_A1'][20]
        - (-1.6591162761952727)) < 1e-12)
chk(abs(cv['E_cont_A1'][28]
        - (-4.237328166845254)) < 1e-12)
chk(abs(cv['E_cont_A1'][36]
        - (-2.2313470710720873)) < 1e-12)
chk(pc['verdicts'] == {
    'P': {'syn': 'syntax_emerges_L21',
          'cont': 'content_emerges_L20'},
    'A1': {'syn': 'syntax_emerges_L21',
           'cont': 'content_emerges_L20'}})
# (cross-phase consistency embedded: E_syn_P[36]/E_cont_[36] asserts equal 3122 final values)
o.append('asserts ok (%d checks)' % len(_n))

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3123
           for m in led['measurements']):
    claim = (
        'Omega-P121 (3123, T4 sixth phase: '
        'per-direction operator refit + '
        'trajectory-anchor search + L35 final-'
        'write localization/binning + syntax-'
        'readability layer tracing, qwen3-4b, '
        '230.4s) - verdict ' + V + '.  Part A '
        '(offline, full 672 pairs, 200 reps '
        'seed 3123): per-direction refit gives '
        'S -0.6016 (P) / -0.5541 (A1), MS '
        '-4.956 / -7.154 (direction fixed '
        'points SEPARATED, anchor gap 2.199 '
        'present), and per-trajectory anchors '
        'a_i = mean_t[m - dm_res/S] with '
        'anchor std 1.20/1.22; BUT (1) dirfit '
        'simulation r(auc_sim, auc18) = 0.229 '
        'FAILED (gate 0.7/0.5), (2) pool-'
        'drawn static-anchor simulation r = '
        '0.235 FAILED, (3) split-half anchor '
        'reliability 0.154/0.104 -> '
        'anchor_unreliable: the implied per-'
        'step anchor m - dm_res/S is noise-'
        'dominated (resid_std/|S| = 4.8-6.5 '
        'per step vs anchor cross-trajectory '
        'std 1.2).  KEY ANALYTIC FINDING: the '
        'AR(1) stationary distributions '
        '(sigma = resid_std/sqrt(1-(1+S)^2) = '
        '3.2 (P) / 4.0 (A1), gap 2.2) predict '
        'Mann-Whitney AUC Phi(2.2/sqrt(3.2^2+'
        '4.0^2)) = Phi(0.43) = 0.666, matching '
        'the simulated plateau 0.6636 almost '
        'exactly (3122 shared operator '
        'plateaued at 0.49) -> per-direction '
        'operators raise the LEVEL but the '
        'residual SHAPE failure is an i.i.d.-'
        'bootstrap DIFFUSION ARTIFACT: if the '
        '+/-3-3.6 step residual were '
        'independent across trajectories the '
        'population would diffuse and AUC '
        'would collapse to 0.66, but empirical '
        'auc18 stays 0.975+ -> the residual '
        'must be COMMON-MODE/systematic (step-'
        'type or time-structured), not '
        'independent noise; one-step '
        'calibration IMPROVED (rank PIT KS '
        '0.0365/0.0326 calibrated vs 3122 '
        '0.0532 marginal).  Static persistent-'
        'anchor hypothesis (simplest upgrade '
        'of 3122 conclusion) REFUTED in its '
        'pool-draw form; anchor estimates need '
        'denoising (Kalman/RLS) before '
        'persistence can be tested with power.'
        '  Part B (offline, 3122 wrec_pd full '
        '672): L35 final write pooled ans '
        '-9.70 vs oth -11.60 -> '
        'final_brake_global BUT direction-'
        'asymmetric: A1 answer steps RELEASE '
        '(-8.03 vs -11.72, +3.69) while P flat '
        '(-11.37 vs -11.48) -> the final-layer '
        'brake is not purely content-'
        'nonspecific, it carries a direction x '
        'position-specific component; L30/L32 '
        'assertion write global positive (q '
        '+1.48 vs nq +1.94) with P-query '
        'layer-OPPOSITE modulation (L30: 0.73 '
        'vs 2.27 suppressed; L32: 2.25 vs 1.99 '
        'boosted; q n=70 per layer, small-'
        'sample caution).  Part C (GPU, 4 '
        'conditions x 672 x 2, all-37 hidden-'
        'state logit-lens w_dn readout, '
        'replay BIT-EXACT 0.0 vs 3122 sb_s0): '
        'E_syn(D_s1-D_s2) first sustained <= '
        '-th at L21 BOTH directions (P th '
        '-0.05, A1 -0.025); E_cont(D_s1-D_s3) '
        'at L20 BOTH; n_span 305/320; E_syn '
        'curve = early small negatives, '
        'POSITIVE bump L16-19 (+0.56..+0.90), '
        'then monotone deepening L21 -> final '
        '-2.256 (P) / -1.707 (A1) == 3122 '
        'final-layer values (internal '
        'consistency); E_cont early dips L8-15 '
        'outside the seal gate window (L>=20), '
        'sustained from L20, max depth -4.16/'
        '-4.24 at L28 -> SYNTAX GATING AND '
        'CONTENT READABILITY EMERGE AT L20-21, '
        'UPSTREAM OF THE WRITE CHAIN (L26+), '
        'syntax effect deepening in parallel '
        'with the assertion write chain.  '
        'NEXT 3124: residual common-mode '
        'decomposition (var(resid) = (cls,t) '
        'cell-mean share vs iid remainder) + '
        'cell-mean-drift simulation with '
        'analytic plateau prediction; Kalman/'
        'RLS anchor denoising + split-half '
        'retry; L35 release mechanism (final-'
        'norm gain vs semantic suppression); '
        'GLM4 cross-model replication of the '
        'sentence paradigm.')
    meas = {
        'meas_id': 'meas3123_omega_p121_dirfit_'
                   'anchor_l35loc_syntax_trace',
        'phase': 3123,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen '
                   'before computation: A_dirfit '
                   'r 0.7/0.5, A_anchor r 0.7/0.5, '
                   'A_pit KS 0.05/0.15 rank-based '
                   'same simulation, A_anchor_sep '
                   'gap>0, A_anchor_rel split-half '
                   '0.8 both dirs, B_l35 sign '
                   'pattern, B_l30q sign pattern, '
                   'C_repro ==0.0/<1e-6 FATAL, '
                   'C_syn/cont_trace L*=min L>=20 '
                   'E<=-th (P -0.05/A1 -0.025), '
                   'MC 200 reps seed 3123 '
                   'empirical-residual bootstrap',
        'artifacts': {
            'result': 'phase3123/omega_p121_'
                      'dirfit_anchor_l35loc_'
                      'syntax_trace/result.json',
            'seal': 'phase3123/omega_p121_'
                    'dirfit_anchor_l35loc_'
                    'syntax_trace/design_seal'
                    '.json',
            'readout': 'phase3123/omega_p121_'
                       'dirfit_anchor_l35loc_'
                       'syntax_trace/'
                       'p121_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, '
                'eager, batch 1, 230.4s); Parts '
                'A/B offline on frozen data; '
                'Part C replay bit-exact 0.0; '
                'SMOKE passed on FIRST attempt '
                '(10.8s, C-REPRO already bit-'
                'exact 0.0 at smoke); patch1 '
                '(build_prompt definition + '
                'exact sum/count pooling for '
                'L35/L30Q gates) applied before '
                'smoke; no pre-run bugs',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3123_omega_p121_dirfit_anchor_'
        'l35loc_syntax_trace')
    led.pop('ledger_sha256_8', None)
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
    o.append('ledger already upserted')

# ---------- 3. MEMO Phase 3123 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3123:' not in memo:
    sec = u'''## Phase 3123: Ω-P121 分方向算子 refit + 轨迹锚点搜索 + L35 负写定位分箱 + 语法可读性层追踪（T4 第6Phase）——**静态持久锚点假设否定：分方向算子（S −0.60/−0.55、MS −4.96/−7.15 分离）与池抽取锚点均未恢复 AUC 形状（r 0.229/0.235）但模拟平台 0.50→0.664=方向分离被部分捕获；解析定位：AR(1) 平稳分布预测 Φ(2.2/5.1)=0.666 与模拟平台 0.6636 吻合——AUC 坍缩是 i.i.d. bootstrap 扩散伪影，残差 ±3–3.6 必为共模/系统结构而非独立噪声；单步校准反而改善（PIT 0.0365/0.0326）；L35 刹车全局但方向不对称（A1 答案步释压 +3.7）；语法门控 L21、内容效应 L20——写入链上游** [[NOW]]

**性质**：T4 第 6 Phase，3122 MEMO 第 5 节预注册、门在 seal 观测前冻结（design_seal.json）。qwen3-4b BF16，230.4s（Part C GPU 5376 forwards ≈220s）。Part A offline：3118 冻结轨迹分方向 content 残差 lstsq refit → (S_dir, MS_dir)、锚点 a_i=mean_t[m−dm_res/S_dir]、split-half 可靠性、双模拟（dirfit 算子 / 池抽取锚点）+ 同模拟秩基 PIT。Part B offline：3122 wrec_pd 全量按 ann 类分箱（L35 ans/oth 门、L30+L32 query 极性门、spec_dn/spec_rel (36,7)×2）。Part C GPU：4 条件×672×2 全部 37 个 hidden_states logit-lens（model.model.norm 后 float32 numpy @ w_dn，与 3122 路径一致）→ E_syn(L)/E_cont(L) 逐层曲线 → L*=min L≥20 且 E≤−th。SMOKE 一次通过（10.8s，C-REPRO 已 bit-exact 0.0）；patch1（build_prompt 定义 + L35/L30Q 精确 sum/count 池化）smoke 前应用，无 pre-run bug。

### 1. 三大发现（重复三遍）
1. **静态持久锚点假设否定 + AUC 坍缩=扩散伪影（解析定位）**。分方向 refit 本身成立：(S,MS)=P(−0.6016, −4.956)/A1(−0.5541, −7.154)，方向不动点分离 gap=2.199 present；但 (a) dirfit 模拟 r(auc_sim, auc18)=0.2285 failed（门 0.7/0.5），(b) 池抽取静态锚点模拟 r=0.2353 failed，(c) split-half 锚点可靠性 0.154/0.104 → anchor_unreliable——每步隐含锚点 m−dm_res/S 被噪声淹没（resid_std/|S|≈4.8–6.5/步 vs 锚点跨轨迹 std 1.2）。**关键解析发现：AR(1) 平稳分布（σ=resid_std/√(1−(1+S)²)=3.2(P)/4.0(A1)、gap 2.2）预测 Mann-Whitney AUC Φ(2.2/√(3.2²+4.0²))=Φ(0.43)=0.666，与模拟平台 0.6636 几乎精确吻合（3122 共享算子平台 0.49）**——分方向算子抬升了水平但形状失败的本质是 **i.i.d. bootstrap 扩散伪影：若每步 ±3–3.6 残差跨轨迹独立，总体必然扩散、AUC 必然坍到 0.66；经验 auc18 却保持 0.975+ → 残差必为共模/系统结构（步型或时间结构化），不是独立噪声**。单步校准反而改善：PIT KS 0.0365/0.0326 双双 calibrated（3122 为 0.0532 marginal）。
2. **L35 刹车全局但方向不对称——A1 答案步释压 +3.7**。池化门：ans −9.70 vs oth −11.60 双负 → final_brake_global；但分方向揭示结构：**P 侧 ans≈oth（−11.37 vs −11.48，无差），A1 侧答案步显著释压（−8.03 vs −11.72，+3.69）→ 末层大负写不是纯内容非特异，含方向×位置特异成分——A1 答案步的部分释压是"no-答案成形"通道候选**。L30/L32 断言写全局正（q +1.48 vs nq +1.94 → assertion_write_global_positive，query 特异性再次否定），但 P-query 呈层相反调制：L30 抑制（0.73 vs nq 2.27）、L32 增强（2.25 vs 1.99）——query 步在断言写链内有层特异的重新分配（q 每层仅 70 步，小样本谨慎）。
3. **语法门控与内容可读性的层定位：L20–21，写入链上游**。37 层 logit-lens 全场（replay bit-exact 0.0，n_span 305/320）：**E_syn=D_s1−D_s2 首次持续 ≤−th 在 L21（P −0.05/A1 −0.025 门，双方向）；E_cont=D_s1−D_s3 在 L20（双方向）——语法效应与内容效应都恰在写入链（L26–35）上游涌现**。E_syn 曲线形状：早期小幅负（L1–2 ≈−0.34）→ 中层正凸 L16–19（+0.56…+0.90）→ L21 起转负且单调深化至终层 −2.256(P)/−1.707(A1)（=3122 终层值，内部一致性 bit 级吻合）；E_cont：L0 +17.4（embedding 平凡差）→ L8–15 多次越阈值（seal 门窗口 L≥20 之外）→ L20 起持续、L28 最深 −4.16(P)/−4.24(A1) → 终层 −3.677/−2.231。**语法效应沿断言写入链并行深化——"语法使内容可读"（3122）从终层现象升级为 L21 起的逐层累积过程**。

### 2. 关键数值
Part A：S_P=−0.601611613690753、MS_P=−4.955594255793292、resid_std 2.9408、锚点 −4.9556±1.1981、rel_r 0.1537；S_A1=−0.5540798866844862、MS_A1=−7.154421990932929、resid_std 3.5819、锚点 −7.1544±1.2171、rel_r 0.1038；gap=2.198827735139629；r_dir=0.2285152966483294、r_anchor=0.23528388458123164；PIT KS 0.03650545634920632/0.03262400793650794；auc_sim_dir=[0.9809, 0.8437, 0.7553, 0.7133, 0.6342, 0.6797, 0.6691, 0.6414, 0.6735, 0.6659, 0.6377, 0.6318, 0.6636]；auc_sim_anchor 终值 0.6573。Part B：L35 pooled ans −9.701520158776216/oth −11.597647604782505；P ans −11.3737/oth −11.4793，A1 ans −8.0294/oth −11.7160（n_ans 672/n_oth 7392 每方向）；L30Q pooled q 1.482873646914959/nq 1.941596696649045；L30 q_pd [0.7324, 1.3961]/nq_pd [2.2730, 1.5289]，L32 q_pd [2.2533, 1.2398]/nq_pd [1.9924, 1.9751]。Part C：repro 0.0；n_span 305/320；L* syn 21/21、cont 20/20；E_syn_P L17 +0.7791/L21 −0.1681/L32 −1.7999/L36 −2.2558336193932855；E_syn_A1 L21 −0.2743/L36 −1.7065；E_cont_P L0 +17.4347/L15 −2.3296/L20 −1.4215/L28 −4.1570/L36 −3.6768；E_cont_A1 L20 −1.6591/L28 −4.2373/L36 −2.2313。N_REPS 200 seed 3123。

### 3. 硬伤
① 静态锚点否定受限于估计噪声：resid_std/|S|≈4.8–6.5/步 vs 锚点 std 1.2——split-half 0.15 可能是检验力不足而非真不持久（Kalman/RLS/更长窗口去噪未做，属 3124）；② 锚点模拟用池抽取而非轨迹自身锚点——"自身锚点是否充分"的确定性诊断未做（诊断性用途但 seal 未含）；③ auc_sim（672×200 模拟池）与 auc18（672 经验）噪声水平不同，r 0.23 的形状相关部分受池方差差异影响；④ L35 分箱 ans n=672（每轨迹 1 步）vs oth n=7392 非对称；q 类步每层仅 70，P-query 层相反调制是小样本结论；⑤ E_cont L0 +17.4 为 embedding 层平凡差异（dots vs 词 token），曲线解释须从 L≥1 起；E_syn L1–2 小幅负（−0.34）与 L16–19 正凸并存——L21 是"持续越限"点而非首次非零；⑥ logit lens=norm 后线性读出，≠真实 logit（final RMSNorm 增益未逐位置分解，L35 释压的 norm 增益 vs 语义成分未分离）；⑦ 单模型单材料族，方向不对称（A1 释压）与层定位（L20/21）待跨模型；⑧ PIT 校准只证单步条件分布，不证多步生成（与 3122 同理）；扩散伪影论证依赖 AR(1) 平稳近似，真实轨迹远离平稳态。

### 4. 机制拼图更新
内部响应图谱：① 分方向算子表 (S_dir, MS_dir) + 锚点分布（P −4.96±1.20 / A1 −7.15±1.22，gap 2.20）；② 37 层×4 条件×672×2 margin 全场（ml npz，logit lens）；③ **E_syn/E_cont 逐层曲线（新结构：语法/内容效应涌现层 L21/L20 + E_syn 中层正凸 L16–19）**；④ L35×方向×ans/oth 分箱 + spec_dn/spec_rel (36,7)×2（全类×层写入谱）。RDC 更新：① **静态持久锚点假设否定（3122 提出的最简升级形式失败）——动态模型必须转向：残差共模结构（步型/时间项）或时变锚点或高阶 Markov；AUC 坍缩被解析定位为 i.i.d. bootstrap 扩散伪影（Φ(0.43)=0.666==0.6636），残差 ±3–3.6 必为共模——这是对"重构失败"原因的第一次解析级定位**；② **"语法使内容可读"获得层定位：L20–21 起效、写入链上游、沿写入链单调深化——语法系统是信息通路门控的层级证据成立**；③ **L35 末层刹车不对称：A1 答案步 +3.7 释压——末层负写含方向×位置特异成分，"no-答案成形"通道候选**。

### 5. 3124 预注册（T4 继续，观测前冻结框架）
① **残差共模分解（offline，冻结 npz）**：var(resid) 分解为 (cls,k) cell 均值份额 vs 独立余项；cell-mean-drift + 缩减残差重模拟，解析预测新平台值并对照——直接检验"扩散伪影"论证；② **锚点去噪复查**：Kalman/递归最小二乘锚点估计 + split-half 复测（检验力修正后的持久性判定）；③ **L35 释压机制分解**：final-norm 增益交互 vs 语义抑制（norm 谱×位置×方向，离线 wrec_pn 可部分分离）；④ **跨模型**：GLM4 复刻句级替换范式（s0–s3）+ 层追踪（GPU，逐模型防 OOM）——三图谱跨模型要求。具体门在 3124 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3123/omega_p121_dirfit_anchor_l35loc_syntax_trace/`（result.json、design_seal.json、run_log.txt、p121_readout.npz）；脚本 `tests/glm5/phase3123_omega_p121_dirfit_anchor_l35loc_syntax_trace.py`；补丁 `tests/gpt5_temp/p3123_patch1.py`。
'''
    sec = sec.replace('[[NOW]]', '[' + NOW + ']')
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    _memo_delta = len(sec)
    o.append('memo +%d chars (Phase 3123)'
             % len(sec))
else:
    _memo_delta = 0
    o.append('memo already appended')

# ---------- 4. workspace logs (x2) ----------
line_exp = ('- Phase 3123 Omega-P121 (T4 sixth '
            'phase: per-direction operator refit '
            '+ trajectory-anchor search + L35 '
            'final-write localization/binning + '
            'syntax-readability layer tracing, '
            'qwen3-4b, 230.4s): verdict ' + V + '. '
            '(A) Per-direction refit (S '
            '-0.6016/-0.5541, MS -4.956/-7.154, '
            'anchor gap 2.20 present) + pool-'
            'drawn static anchors BOTH FAIL to '
            'restore the auc18 shape (r 0.229/'
            '0.235 vs gate 0.7/0.5) BUT simulated '
            'plateau rises 0.49->0.664 and PIT '
            'improves to calibrated (KS 0.0365/'
            '0.0326); split-half anchor '
            'reliability 0.154/0.104 -> estimates '
            'noise-dominated (resid_std/|S| '
            '~4.8-6.5/step vs anchor std 1.2); '
            'ANALYTIC: AR(1) stationary '
            'distributions predict Phi(2.2/5.1)='
            '0.666 == simulated plateau -> AUC '
            'collapse is an i.i.d.-bootstrap '
            'DIFFUSION ARTIFACT, residual +/-3-3.6 '
            'must be COMMON-MODE/systematic, not '
            'independent noise. (B) L35 brake '
            'pooled GLOBAL (ans -9.70 vs oth '
            '-11.60) but direction-asymmetric: '
            'A1 answer steps RELEASE -8.03 vs '
            '-11.72 (+3.7), P flat (-11.37 vs '
            '-11.48); L30/L32 assertion write '
            'global positive (q +1.48 vs nq '
            '+1.94) with P-query layer-opposite '
            'modulation (L30 0.73 vs 2.27; L32 '
            '2.25 vs 1.99). (C) Layer tracing '
            '(37-layer logit lens, replay bit-'
            'exact 0.0, n_span 305/320): E_syn '
            'sustained <= -th at L21 BOTH dirs, '
            'E_cont at L20 BOTH -> syntax gating '
            '+ content readability emerge at '
            'WRITE-CHAIN UPSTREAM (L26+); E_syn '
            'positive bump L16-19 then monotone '
            'deepening to -2.256/-1.707 final '
            '(== 3122 values); E_cont early dips '
            'L8-15 outside gate window, max '
            '-4.16/-4.24 at L28. NEXT 3124: '
            'residual common-mode decomposition + '
            'cell-mean-drift simulation with '
            'analytic plateau prediction; Kalman '
            'anchor denoising; L35 norm-gain vs '
            'semantic decomposition; GLM4 '
            'cross-model sentence paradigm.\n')
line_clo = ('- Phase 3123 closeout finished: '
            'five-write chain ok (ledger n=260 '
            'l14=228 sha=%SHA8%, MEMO +%MEMOC% '
            'chars, dual wlog, MEMORY.md update); '
            'disk verify next. No pre-run bugs: '
            'SMOKE passed on first attempt '
            '(10.8s, C-REPRO already bit-exact '
            '0.0 at smoke); patch1 (build_prompt '
            '+ exact pooling) applied before '
            'smoke.\n')
try:
    led2 = json.load(io.open(LEDGER,
                             encoding='utf-8'))
    _sha8 = led2['ledger_sha256_8']
except Exception:
    _sha8 = 'unknown'
line_clo = line_clo.replace(
    '%SHA8%', _sha8).replace(
    '%MEMOC%', str(_memo_delta))
for wdir in (WLOG_D, WLOG_C):
    for tag, line in (('exp', line_exp),
                      ('clo', line_clo)):
        wl = wdir + '\\' + '2026-09-23.md'
        try:
            prev = io.open(wl,
                           encoding='utf-8').read()
        except IOError:
            prev = ''
        marker = ('Phase 3123 Omega-P121' if tag
                  == 'exp'
                  else 'Phase 3123 closeout')
        if marker not in prev:
            try:
                with io.open(wl, 'a',
                             encoding='utf-8') as f:
                    f.write(line)
                o.append('wlog %s appended %s'
                         % (tag, wl))
            except Exception as e:
                o.append('wlog %s fail %s: %r'
                         % (tag, wl, e))
        else:
            o.append('wlog %s already %s'
                     % (tag, wl))

# ---------- 5. MEMORY.md ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3122' in mem_old:
    sha8 = '?'
    try:
        led2 = json.load(io.open(LEDGER,
                                 encoding='utf-8'))
        sha8 = led2['ledger_sha256_8']
    except Exception:
        pass
    NEW_3121 = (u'- 3121（T4）：token 级替换范式失效'
                u'（振荡淹没均值、双反向）；擦除链写入='
                u'断言牵引、消融释放 no 提高判别；MC '
                u'覆盖 0.967=振荡是分布现象。')
    NEW_3122 = (u'- 3122（T4）：写入谱 L28–34 正写+'
                u'L35 大负写 −11.4；句级替换 E_cont '
                u'−3.68/−2.23、语法混排恢复一半=**语法'
                u'使内容可读**；单步校准但迭代坍缩=缺持'
                u'久锚点。')
    NEW_3123 = (u'- 3123（T4）：分方向算子（S −0.60/'
                u'−0.55、MS −4.96/−7.15 分离）+池抽锚点'
                u'均未恢复 AUC 形状（r 0.23）但平台 '
                u'0.50→0.66、PIT 0.033 校准=**i.i.d. '
                u'扩散伪影、残差必为共模**；L35 刹车全局'
                u'但 A1 答案步释压 +3.7；**语法门控 '
                u'L21/内容 L20 涌现=写入链上游**。')
    NEW_3120 = (u'- 3120（T4）：重述步上推（断言侵蚀）、'
                u'标点步恢复 gap；L30/L32 反向于 L26；'
                u'Δm 线性（slope −0.68）。')
    NEW_3118 = (u'- 3118（T4）：AUC 0.981→0.672 强振荡；'
                u'状态补偿 0.52 vs 闭环放大 1.08–1.73；'
                u'L26 消融 +7.6pp。')
    NEW_3114_17 = (u'- 3114–3117：head=相关影子、L24 条件'
                   u'性对抗；三层联合 additive、L32=熵调节；'
                   u'TOP3=L26/L33/L31；成对消融 buffered、'
                   u'A1 首 token 20.2% 分叉。')
    NEW_NEXT = (u'- max=3123，下一 3124：**残差共模分解+'
                u'cell-mean-drift 重模拟（解析预测平台）+ '
                u'Kalman 锚点去噪复查 + L35 norm 增益×语义'
                u'分解 + GLM4 跨模型句级范式**。')
    lines = mem_old.splitlines()
    out = []
    skip = 0
    for ln in lines:
        if skip > 0:
            skip -= 1
            continue
        if ln.startswith(u'## 机制链状态'):
            out.append(u'## 机制链状态（3123）')
        elif ln.startswith(u'- 3121'):
            out.append(NEW_3121)
        elif ln.startswith(u'- 3122'):
            out.append(NEW_3122)
            out.append(NEW_3123)
        elif ln.startswith(u'- 3120'):
            out.append(NEW_3120)
        elif ln.startswith(u'- 3118'):
            out.append(NEW_3118)
        elif ln.startswith(u'- 3117'):
            out.append(NEW_3114_17)
            skip = 3
        elif ln.startswith(u'- max=3122'):
            out.append(NEW_NEXT)
        else:
            out.append(ln)
    mem_new = u'\n'.join(out) + u'\n'
    assert mem_new.count(
        u'## 机制链状态（3123）') == 1
    assert mem_new.count(u'- 3123（T4）') == 1
    assert mem_new.count(u'max=3123') == 1
    assert mem_new.count(u'- 3116：') == 0
    assert mem_new.count(u'- 3115：') == 0
    assert mem_new.count(u'- 3114：') == 0
    assert mem_new.count(
        u'- 3114–3117：') == 1
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars (sha8=%s)'
             % (len(mem_new), sha8))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
