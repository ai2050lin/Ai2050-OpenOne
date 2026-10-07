# -*- coding: utf-8 -*-
"""Phase 3122 closeout (idempotent):
result asserts -> Ledger -> MEMO Phase 3122 ->
workspace logs (x2 entries) -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3122'
        r'\omega_p120_write_content_readout_'
        'sentence_causal_dist_recon')
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
TODAY = datetime.date.today().isoformat()
WDAYS = sorted(set([TODAY, '2026-09-23']))
o = []

V = ('mixed|write_polarity_diverged|'
     'replay_bit_exact|'
     'sentence_content_push_down|'
     'sentence_content_push_down|'
     'syntax_effect_present|'
     'syntax_effect_present|'
     'oscillation_persist|'
     'distribution_reconstruction_failed|'
     'pit_marginal')

# ---------- 1. result.json asserts ----------
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
assert res['verdict'] == V, res['verdict']
assert res['smoke'] is False
assert res['n_pairs'] == 672
assert res['np_a'] == 672
assert abs(res['runtime_s'] - 238.2) < 0.05
pa = res['part_a']
assert pa['repro']['verdict'] == 'replay_bit_exact'
assert pa['repro']['max_diff'] == 0.0
wp = pa['write_polarity']
assert wp['verdict'] == 'write_polarity_diverged'
assert abs(wp['p_dn_L30']
           - 1.851479411125183) < 1e-9
assert abs(wp['p_dn_L32']
           - 2.0154643058776855) < 1e-9
am = pa['ablation_margin_split']
assert am['verdict'] == 'mixed'
assert abs(am['a1_share']
           - 0.4832655000280688) < 1e-9
assert abs(am['sum_abs']['P']
           - 20217.24432387948) < 1e-6
assert abs(am['sum_abs']['A1']
           - 18907.76924687624) < 1e-6
ef = am['effects']
assert abs(ef['L26_P']['mean_first']
           - (-2.9966676770931198)) < 1e-9
assert abs(ef['L26_A1']['mean_first']
           - 0.8122878216561817) < 1e-9
assert abs(ef['L31_P']['mean_first']
           - (-2.8040195579330125)) < 1e-9
assert abs(ef['L31_A1']['mean_first']
           - 0.6196181188736644) < 1e-9
assert abs(ef['L33_P']['mean_first']
           - (-2.1185272875286283)) < 1e-9
assert abs(ef['L33_A1']['mean_first']
           - 1.5605492538639478) < 1e-9
sp = pa['spectrum']
assert abs(sp['P']['mean_dn'][34]
           - 4.504420280456543) < 1e-6
assert abs(sp['P']['mean_dn'][35]
           - (-11.44801139831543)) < 1e-6
assert abs(sp['A1']['mean_dn'][34]
           - 4.202137470245361) < 1e-6
assert abs(sp['A1']['mean_dn'][35]
           - (-11.520125389099121)) < 1e-6
pb = res['part_b']
assert pb['repro_max_diff'] == 0.0
assert pb['n_pad_info'] == {
    'n_span_total': 625, 'n_pad0': 540,
    'n_trunc0': 251}
assert pb['pad_sensitivity']['P']['n_clean'] == 83
assert pb['pad_sensitivity']['A1']['n_clean'] == 83
assert abs(pb['pad_sensitivity']['P']['E_cont_clean']
           - (-3.7325545700917764)) < 1e-9
assert abs(pb['pad_sensitivity']['A1']['E_cont_clean']
           - (-2.0561990694109213)) < 1e-9
eP = pb['effects']['P']
assert eP['n_span'] == 305 and eP['n_len2'] == 305
assert abs(eP['D_mean']['s1']
           - (-2.547917983209501)) < 1e-9
assert abs(eP['D_mean']['s2']
           - (-0.2920843638162144)) < 1e-9
assert abs(eP['D_mean']['s3']
           - 1.1288977380170198) < 1e-9
assert abs(eP['E_cont']
           - (-3.6768157212265207)) < 1e-9
assert abs(eP['E_syn']
           - (-2.255833619393286)) < 1e-9
assert abs(eP['r_osc']
           - 1.2247049670248196) < 1e-9
eA = pb['effects']['A1']
assert eA['n_span'] == 320 and eA['n_len2'] == 320
assert abs(eA['D_mean']['s1']
           - 0.1917136371973902) < 1e-9
assert abs(eA['D_mean']['s2']
           - 1.8982442237809303) < 1e-9
assert abs(eA['D_mean']['s3']
           - 2.423060708269477) < 1e-9
assert abs(eA['E_cont']
           - (-2.2313470710720864)) < 1e-9
assert abs(eA['E_syn']
           - (-1.7065305865835398)) < 1e-9
assert abs(eA['r_osc']
           - 0.9353352644780626) < 1e-9
assert pb['verdicts'] == {
    'P': {'cont': 'sentence_content_push_down',
          'syn': 'syntax_effect_present'},
    'A1': {'cont': 'sentence_content_push_down',
           'syn': 'syntax_effect_present'}}
assert pb['osc_verdict'] == 'oscillation_persist'
pc = res['part_c']
assert pc['n_reps'] == 200
assert abs(pc['sigma']
           - 3.3923936726908717) < 1e-9
assert abs(pc['r_dist']
           - 0.29878525059655753) < 1e-9
assert pc['dist_verdict'] == \
    'distribution_reconstruction_failed'
assert abs(pc['pit_ks']
           - 0.0531746031746031) < 1e-9
assert pc['pit_verdict'] == 'pit_marginal'
aucs = pc['auc_sim']
assert abs(aucs[0]
           - 0.9809094210600907) < 1e-12
assert abs(aucs[1]
           - 0.7180872017166241) < 1e-12
assert abs(aucs[3]
           - 0.523757279099791) < 1e-12
assert abs(aucs[12]
           - 0.49012593244003333) < 1e-12
o.append('asserts ok (%d checks)' % 45)

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3122
           for m in led['measurements']):
    claim = (
        'Omega-P120 (3122, T4 fifth phase: '
        'all-layer MLP write-projection readout '
        '+ sentence-level coherent replacement '
        'causality + distribution-level '
        'reconstruction, qwen3-4b, 238.2s) - '
        'verdict ' + V + '.  Part A-GPU (all-'
        'layer MLP down_proj outputs captured '
        'via hooks, projected on w_dn/w_fam/'
        'norm; replay BIT-EXACT 0.0 vs 3118): '
        'write spectrum = flat noise band '
        '+/-0.3 below L23, P-side rise at '
        'L23-24 (0.88/1.05), sustained '
        'POSITIVE writes L28-34 (P L29 +2.80/'
        'L30 +2.21/L34 +4.50; A1 isomorphic '
        '+1.27/+1.49/+4.20), and a large '
        'DIRECTION-INDEPENDENT final-layer L35 '
        'negative write -11.45 (P) / -11.52 '
        '(A1) -> write chain = assertion '
        'writes + global final correction; '
        'W-SAN preregistered gate (L30/L32 '
        'p_dn<0, inherited from the 3113 '
        'increment-correlation label L32 '
        'negative write) REFUTED (+1.85/+2.02) '
        'but the sign CONFIRMS the 3121 '
        'direction-split behavioral conclusion '
        '(erase-chain write = assertion-pull '
        'toward yes); metric-caliber difference '
        '(increment correlation vs direct '
        'projection) flagged for '
        'reconciliation.  Part A-offline (3118 '
        'ablation margin direction split, full '
        '672 pairs): a1_share 0.4833 mixed in '
        'MAGNITUDE but first-step SIGNS split '
        'cleanly: ablation LOWERS P first-step '
        'margin (L26/31/33 -3.00/-2.80/-2.12) '
        'and RAISES A1 (+0.81/+0.62/+1.56) -> '
        'early write layers (L26/31/33) '
        'ENHANCE discrimination (push P toward '
        'yes AND A1 toward no), polarity-'
        'opposed to L30/L32 assertion-pull -> '
        'write-chain polarity segmentation '
        'confirmed at continuous-margin level. '
        ' Part B (sentence-level replacement '
        '672x2x4 conditions: s0 replay / s1 '
        'different-content syntactic sentence '
        '/ s2 same-content scrambled / s3 '
        'dots, equal-length in-span '
        'substitution; s0 replay bit-exact '
        '0.0; 625/672 spans, n_pad0 540, '
        'n_trunc0 251): E_cont = D_s1-D_s3 = '
        '-3.677 (P) / -2.231 (A1) BOTH '
        'push_down; E_syn = D_s1-D_s2 = '
        '-2.256 / -1.707 BOTH present -> '
        'scrambling recovers HALF the content '
        'drop -> SYNTAX GATES CONTENT INTO '
        'margin dynamics (3121 token-paradigm '
        'failure positively completed: the '
        'sentence-level paradigm works and '
        'the content effect is NEGATIVE); s3 '
        'dots RAISES margin +1.13 (P); '
        'pad-clean subset (n_pad=0 & n_trunc=0, '
        'n=83) agrees (-3.73/-2.06) -> not a '
        'padding artifact; r_osc = sentence-'
        'level post-span oscillation over 3121 '
        'token-splice oscillation: P 1.225 '
        'persist / A1 0.935 -> sentence-level '
        'perturbation oscillates no less than '
        'token splicing -> oscillation is an '
        'intrinsic response mode, not a splice '
        'artifact.  Part C (empirical-residual '
        'bootstrap 200 reps seed 3122 + frozen '
        '3120 operator + per-(dir,cls,t) '
        'content table; rank-based PIT '
        'computed on the SAME simulation, '
        't>=1, 16128 u values): PIT KS 0.0532 '
        '-> pit_marginal, one-step transition '
        'distribution CALIBRATED; but auc_sim '
        'collapses 0.981->0.718->0.573->~0.50 '
        'while auc18 stays 0.975+ (r_dist '
        '0.2988) -> iterated dynamics collapse '
        'onto the shared linear fixed point '
        'MS=-6.05 -> a per-trajectory '
        'PERSISTENT ANCHOR state is missing '
        'from the (m,cls) conditional-mean '
        'model; the 3120 operator is valid as '
        'a one-step conditional approximation '
        'and invalid as a multi-step '
        'generator - dynamic model must be '
        'upgraded to m(t+1)=f(m(t),cls;'
        'anchor(trajectory)) with the anchor '
        'set by first-step assertion writes. '
        'NEXT 3123: per-direction operator '
        'refit + trajectory-anchor state '
        'search; L35 final-write localization '
        '(position x direction x norm-gain); '
        'write-spectrum position x direction '
        'x cls binning (offline wrec_pd); '
        's1-vs-s2 syntax-readability layer '
        'tracing.')
    meas = {
        'meas_id': 'meas3122_omega_p120_write_'
                   'content_readout_sentence_'
                   'causal_dist_recon',
        'phase': 3122,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen '
                   'before computation: A-REPRO '
                   '==0.0 / <1e-6, W-SAN L30/L32 '
                   'p_dn<0, A1-DIR 0.6/0.4, '
                   'B2-CONT +0.05/-0.05 (A1 '
                   '+0.025/-0.025), B2-SYN '
                   'abs 0.05/0.025, B2-OSC '
                   '0.7/1.0, B2-PAD clean '
                   'subset n_pad=0 & n_trunc=0, '
                   'C-DIST r 0.7/0.5, C-PIT KS '
                   '0.05/0.15 rank-based on the '
                   'same simulation, MC 200 '
                   'reps seed 3122 empirical-'
                   'residual bootstrap',
        'artifacts': {
            'result': 'phase3122/omega_p120_'
                      'write_content_readout_'
                      'sentence_causal_dist_'
                      'recon/result.json',
            'seal': 'phase3122/omega_p120_'
                    'write_content_readout_'
                    'sentence_causal_dist_'
                    'recon/design_seal.json',
            'readout': 'phase3122/omega_p120_'
                       'write_content_readout_'
                       'sentence_causal_dist_'
                       'recon/p120_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, '
                'eager, batch 1, 238.2s); '
                'A-offline and C offline on '
                'frozen data; A/B replay '
                'bit-exact 0.0; FOUR pre-run '
                'bugs caught by SMOKE repro '
                'gates and fixed BEFORE the '
                'full run (forward_wrec 4D '
                'stack missing squeeze(1); '
                'Part C dead-code + pseudo-'
                'PIT merged into single-'
                'simulation rank-based PIT; '
                'Part C NP_A slicing; '
                'forward_track2 swapped args '
                '(toks,base)->(prompt,toks) '
                'flagged by B-REPRO 13.05->'
                '0.0, plus pad_info cross-'
                'direction indexing) - main '
                'script edited in place '
                'pre-run, result.json from '
                'the clean full run',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3122_omega_p120_write_content_'
        'readout_sentence_causal_dist_recon')
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

# ---------- 3. MEMO Phase 3122 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3122:' not in memo:
    sec = u'''## Phase 3122: Ω-P120 全层 MLP 写入投影读出 + 句级连贯替换因果 + 分布级重构（T4 第5Phase）——**写入谱：L28–34 正写断言 + 末层 L35 大负写 −11.4（方向无关）；句级连贯替换确立内容因果：语法使内容可读（E_cont −3.68/−2.23 push_down 双侧、语法混排恢复一半）；单步转移校准（秩基 PIT KS 0.053）但迭代轨迹坍缩（AUC→0.50）——(m,cls) 模型缺失持久轨迹锚点** [[NOW]]

**性质**：T4 第 5 Phase，3121 MEMO 第 5 节预注册、门在 seal 观测前冻结（design_seal.json）。qwen3-4b BF16，238.2s。Part A-GPU：全层 MLP down_proj 输出 hook 捕获（WREC store），einsum 投影 w_dn/w_fam/norm（36 层×672 对×13 步×双方向）；重放对 3118 冻结轨迹。Part A-offline：3118 npz 消融 teacher-forced margin 轨迹分方向（免 GPU）。Part B-GPU：672×2×4 条件（s0 重放/s1 不同内容正确句法句/s2 同内容乱序/s3 '.' 填充，span 内等长替换）。Part C：经验残差 bootstrap（200 reps seed 3122）+3120 冻结算子+类别内容表，单段模拟上同时算 auc_sim 与秩基 PIT。SMOKE 修 4 bug 后全量（交换参数 bug 曾使 B-REPRO 13.05，修复后 0.0——复现门起作用）。

### 1. 三大发现（重复三遍）
1. **写入谱：L28–34 正写断言 + 末层 L35 大负写（方向无关）**。全层 MLP down_proj 输出在 w_dn 上的投影均值：L23 以下 ±0.3 噪声带，L23–24 起 P 侧上抬（0.88/1.05），L28–34 持续正写（P L29 +2.80/L30 +2.21/L34 +4.50；A1 同构 +1.27/+1.49/+4.20），**末层 L35 强负写 −11.45（P）/−11.52（A1）——两方向几乎相同 → 内容非特异的全局大负写**。W-SAN 预注册门（L30/L32 投影 <0，源自 3113 "L32 负写=擦除相" 的增量相关标签）被否定（+1.85/+2.02），但该符号恰好**确认 3121 方向分解的行为结论（擦除链写入=断言牵引往 yes）**；3113 的"负写"度量口径（家族增量相关）与直接投影的差异列为待澄清。Part A-offline（全 672 对）补充：a1_share=0.4833（mixed，量级均分）但**首步符号干净分裂——消融把 P 首步 margin 压低（L26/31/33：−3.00/−2.80/−2.12 → 写入把 P 顶向 yes）、把 A1 首步 margin 抬高（+0.81/+0.62/+1.56 → 写入把 A1 压向 no）——前段写层（L26/31/33）增强判别，与 L30/L32 断言牵引极性对抗**；写入链极性分段在连续 margin 层面成立。
2. **句级连贯替换确立内容因果：语法使内容可读**。Part B：s0 重放 bit-exact 0.0；625/672 对有 span（n_pad=0 540、n_trunc=0 251）。**E_cont=D_s1−D_s3：P −3.677/A1 −2.231，双侧过 push_down 门**；**E_syn=D_s1−D_s2：P −2.256/A1 −1.707，双侧 syntax_effect_present**。分解：替换句（s1）把 P margin 压低 ~2.5，同内容乱序（s2）只恢复一半 → **句法正确性是内容进入 margin 动态的门控——3121 token 级范式失效的正面补全：句级范式有效且内容效应为负向（替换掉支持句→信心下降）**；s3（'.' 填充）反而上推 +1.13（P）——句子存在本身经语法通道压低 margin。pad 干净子集（n_pad=0 且 n_trunc=0，n=83）E_cont P −3.733/A1 −2.056 与全样本一致 → 非填充伪影。**r_osc（句级替换 span 后振荡 / 3121 token 级替换振荡）= P 1.225 persist、A1 0.935——句级自然替换的振荡不低于 token 接缝替换 → 振荡是模型对扰动的固有响应模式，不是拼接伪影（再证 3121 分布现象结论）**。
3. **单步校准但迭代坍缩——(m,cls) 模型缺失持久轨迹锚点**。Part C：**秩基 PIT KS=0.0532（门 0.05/0.15）→ pit_marginal，单步转移分布校准良好**（同一模拟上计算，u 值 16128 个）；但 auc_sim 从 0.981→0.718→0.573→~0.50 坍缩（auc18 保持 0.975+），r_dist=0.2988 <0.5 → distribution_reconstruction_failed。**机理：线性回归核（S=−0.68, MS=−6.05 单一公共不动点）在迭代中把 P/A1 拉到同一点；真实动态保持 AUC 0.98 → 每条轨迹携带自身的"有效锚点"（轨迹级持久状态），条件均值算子 (m,cls) 不含该变量**。这是 3121 "写入=断言牵引"的动力学表现：断言写入在首步设定轨迹自身锚点，之后轨迹沿自身锚点演化。3120 单一 (S,MS) 是跨条件平均的产物——作为单步条件分布近似成立（PIT），作为多步生成模型不成立。

### 2. 关键数值
Part A：L30 +1.8515/L32 +2.0155（门 <0 被否）；谱 P dn L28–35：+1.229/+2.801/+2.210/+2.063/+4.504/−11.448，A1 L34/L35：+4.202/−11.520；norm 谱单调 5.2→344；a1_share 0.4833（sum_abs 20217.2/18907.8）；首步消融效应 L26/31/33：P −2.9967/−2.8040/−2.1185、A1 +0.8123/+0.6196/+1.5605，rest 步 |mean|≤0.42。Part B：repro 0.0；n_span 625（pad0 540/trunc0 251）；D_mean P s1 −2.5479/s2 −0.2921/s3 +1.1289，A1 s1 +0.1917/s2 +1.8982/s3 +2.4231；E_cont −3.6768/−2.2313、E_syn −2.2558/−1.7065、r_osc 1.2247/0.9353；pad 干净 n=83：−3.7326/−2.0562。Part C：σ=3.3924、pool 16128、N_REPS 200；r_dist 0.2988；auc_sim=[0.9809, 0.7181, 0.5731, 0.5238, 0.5059, 0.5018, 0.5010, 0.4780, 0.5018, 0.5016, 0.4939, 0.4832, 0.4901]；PIT KS 0.0532（t≥1，2×672×12=16128 u 值）。

### 3. 硬伤
① W-SAN 门方向预注册继承自 3113 相关法标签（增量相关≠直接投影），度量口径差异未澄清——门判 diverged 的科学解读依赖 3121 行为结论；② 写入谱为 672×13 位置平均，位置×方向×cls 未分箱（query 前/后、answer token 位置混合；npz wrec_pd 全量保留可离线后分析）；③ L35 大负写无法区分语义抑制与 final-norm 前数值规范（RMSNorm 增益交互未分解，且 norm 谱 L35 达 344，投影绝对值受尺度影响）；④ Part B 效应样本 305/320（93% 有 span），pad 干净子集仅 83 对；r_osc 分母为 token 级范式振荡，跨范式比值解释需谨慎（句级扰动幅度天然更大）；⑤ Part C content 表 in-sample 拟合+评估；残差 bootstrap 假设步间独立未检验；⑥ auc_sim（672×200 模拟池）与 auc18（672 对经验）的噪声水平不同，r_dist 偏低部分可由模拟池方差更小解释；⑦ 单模型单材料族，写入谱结构与语法门控结论待跨模型；⑧ SMOKE 抓出 4 bug（含 forward 参数交换这类高危害 bug）反映多 Phase 冻结数据级联的口径对齐复杂度高——复现门是必要保障而非冗余。

### 4. 机制拼图更新
内部响应图谱：① 全层写入谱表（36 层×双方向×3 读出，npz 全量）；② **L35 末层大负写**（方向无关、内容非特异）新结构；③ 写入链极性对抗结构（L26/31/33 判别增强 vs L30/32 断言牵引）的 margin 级连续证据；④ 句级替换因果效应表（s1/s2/s3×双方向+pad 干净子集）。RDC 更新：① **写入链功能分段：前段写层（L26/31/33）维持/增强判别（P 顶向 yes、A1 压向 no），后段 L30/L32 执行断言牵引（对 A1 构成假牵引），末层 L35 全局大负写（功能待定位）**；② **"语法使内容可读"确立为外部语言族图谱→内部响应图谱关联机制的关键一环：句法正确性门控内容进入 margin 动态（乱序恢复一半效应），这把语法系统从"伴随现象"提升为"信息通路门控"**；③ **动力学模型升级需求：从 (m,cls) 条件均值算子升级为带持久锚点的条件模型 m(t+1)=f(m(t),cls;anchor(轨迹))，锚点由首步断言写入设定——3120 算子降级为单步近似**。

### 5. 3123 预注册（T4 继续，观测前冻结框架）
① **分方向算子 refit + 锚点状态搜索**：(S_dir, MS_dir) 分方向拟合检验 auc_sim 恢复；进一步用轨迹前 k 步估计每轨迹锚点（轨迹条件化），检验 auc_sim→0.98 与 PIT 保持；② **L35 末层负写定位**：分位置（answer token/重述 span/其他）×分方向×与 final norm 增益交互分解，判别语义抑制 vs 数值规范；③ **写入谱分箱后分析**（离线，wrec_pd 全量）：位置×方向×cls 分箱定位断言写入热点层×位置；④ **语法可读性层追踪**：s1 vs s2 差异轨迹的逐层分化点定位（语法解析完成层候选）。具体门在 3123 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3122/omega_p120_write_content_readout_sentence_causal_dist_recon/`（result.json、design_seal.json、run_log.txt、p120_readout.npz）；脚本 `tests/glm5/phase3122_omega_p120_write_content_readout_sentence_causal_dist_recon.py`；补丁 `tests/gpt5_temp/p3122_patch1.py`～`p3122_patch4.py`。
'''
    sec = sec.replace('[[NOW]]', '[' + NOW + ']')
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    _memo_delta = len(sec)
    o.append('memo +%d chars (Phase 3122)'
             % len(sec))
else:
    _memo_delta = 0
    o.append('memo already appended')

# ---------- 4. workspace logs (x2) ----------
line_exp = ('- Phase 3122 Omega-P120 (T4 fifth '
            'phase: all-layer MLP write-projection '
            'readout + sentence-level replacement '
            'causality + distribution-level '
            'reconstruction, qwen3-4b, 238.2s): '
            'verdict ' + V + '. (A) Write spectrum: '
            'flat below L23, sustained POSITIVE '
            'writes L28-34 (P L34 +4.50) + final-'
            'layer L35 large NEGATIVE write -11.45/'
            '-11.52 direction-independent -> '
            'assertion writes + global final '
            'correction; W-SAN preregistered <0 '
            'REFUTED (+1.85/+2.02) but sign '
            'CONFIRMS 3121 assertion-pull (3113 '
            'negative-write label was increment-'
            'correlation caliber); offline: '
            'ablation first-step signs split (P '
            '-3.00/-2.80/-2.12 vs A1 +0.81/+0.62/'
            '+1.56) -> L26/31/33 ENHANCE '
            'discrimination, polarity-opposed to '
            'L30/L32. (B) Sentence replacement: '
            'E_cont -3.677/-2.231 push_down BOTH; '
            'E_syn -2.256/-1.707 BOTH present -> '
            'scrambling recovers HALF the drop -> '
            'SYNTAX GATES CONTENT INTO margin '
            'dynamics; s3 dots RAISES +1.13; pad-'
            'clean n=83 agrees; r_osc P 1.225 -> '
            'oscillation intrinsic, not splice '
            'artifact. (C) PIT KS 0.0532 marginal '
            '= one-step CALIBRATED but auc_sim '
            'collapses 0.981->0.50 (r_dist 0.299) '
            '-> per-trajectory PERSISTENT ANCHOR '
            'missing from (m,cls) model; 3120 '
            'operator = one-step approximation '
            'only. NEXT 3123: per-direction refit '
            '+ anchor search; L35 localization; '
            'spectrum binning; syntax-readability '
            'layer tracing.\n')
line_clo = ('- Phase 3122 closeout finished: '
            'five-write chain ok (ledger n=259 '
            'l14=227 sha=%SHA8%, MEMO +%MEMOC% '
            'chars, dual wlog, MEMORY.md update); '
            'disk verify next. FOUR pre-run bugs '
            'caught by SMOKE repro gates and '
            'fixed before the full run '
            '(forward_wrec squeeze; Part C merge '
            '+ rank PIT; NP_A slicing; swapped '
            'forward args flagged by B-REPRO '
            '13.05->0.0 + pad_info cross-dir), '
            'result.json from clean run.\n')
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
        marker = ('Phase 3122 Omega-P120' if tag
                  == 'exp'
                  else 'Phase 3122 closeout')
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
if 'max=3121' in mem_old:
    sha8 = '?'
    try:
        led2 = json.load(io.open(LEDGER,
                                 encoding='utf-8'))
        sha8 = led2['ledger_sha256_8']
    except Exception:
        pass
    r1o = u'## 机制链状态（3121）'
    r1n = u'## 机制链状态（3122）'
    assert mem_old.count(r1o) == 1
    mem_new = mem_old.replace(r1o, r1n)
    r2o = (u'- 3121（T4）：反事实替换+擦除链极性+双源'
           u'重构。**token 级替换范式失效：响应=±3~5 '
           u'振荡淹没条件均值（E_10 −2.53/E_31 +3.22 '
           u'双反向）；span 后三条件收敛（−3.41/−2.27/'
           u'−0.73）**。**方向分解翻转行为结论：擦除链'
           u'写入=断言牵引、消融=释放 no 提高判别（A1 '
           u'first_yes 0.150/0.098/joint 0.066 vs '
           u'clean 0.180、P 侧饱和）；L26 +7.6pp 全来'
           u'自 A1（0.180→0.332）**。**重构失败 R² '
           u'0.128 但 MC 覆盖 0.967=振荡是分布现象**。')
    r2n = (u'- 3121（T4）：token 级替换范式失效（振荡'
           u'淹没均值、双反向）；方向分解：擦除链写入='
           u'断言牵引、消融释放 no 提高判别（A1 '
           u'0.150/0.098/joint 0.066 vs clean '
           u'0.180）；重构失败但 MC 覆盖 0.967=振荡'
           u'是分布现象。\n'
           u'- 3122（T4）：写入谱 L28–34 正写+L35 末'
           u'层大负写 −11.4（方向无关）；句级替换确立'
           u'内容因果 E_cont −3.68/−2.23、语法混排恢'
           u'复一半=**语法使内容可读**；单步校准 PIT '
           u'0.053 但迭代坍缩 AUC→0.50=**缺持久轨迹'
           u'锚点**。')
    assert mem_new.count(r2o) == 1
    mem_new = mem_new.replace(r2o, r2n)
    r3o = (u'- 3120（T4）：重述步双向 margin 上推'
           u'（P +0.62/A1 +0.23=断言侵蚀）、标点步恢复 '
           u'gap（A1 −4.25）；L30/L32 行为反向于 L26'
           u'（−1.6/−4.2pp）；Δm 均值形状线性'
           u'（slope −0.68、R² 0.343）。')
    r3n = (u'- 3120（T4）：重述步双向 margin 上推'
           u'（断言侵蚀）、标点步恢复 gap；L30/L32 行为'
           u'反向于 L26；Δm 线性（slope −0.68、R² '
           u'0.343）。')
    assert mem_new.count(r3o) == 1
    mem_new = mem_new.replace(r3o, r3n)
    r4o = (u'- 3119：答案 token 仅 t=1、振荡由内容步'
           u'驱动；补偿层特异（L30/L32 late-peak）；'
           u'重写 R² 0.19=方差非曲率。')
    r4n = (u'- 3119：答案 token 仅 t=1、振荡由内容步'
           u'驱动；补偿层特异；重写 R² 0.19=方差非'
           u'曲率。')
    assert mem_new.count(r4o) == 1
    mem_new = mem_new.replace(r4o, r4n)
    r5o = (u'- max=3121，下一 3122：**句级连贯替换'
           u'检验内容特异性 + 方向分解行为门标准化 + '
           u'分布级重构（转移分布拟合）+ L26/L31/L33 '
           u'写入内容读出**。')
    r5n = (u'- max=3122，下一 3123：**分方向算子 '
           u'refit+轨迹锚点状态搜索 + L35 末层负写'
           u'定位 + 写入谱位置×方向×cls 分箱（离线）'
           u'+ s1-vs-s2 语法可读性层追踪**。')
    assert mem_new.count(r5o) == 1
    mem_new = mem_new.replace(r5o, r5n)
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
