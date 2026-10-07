# -*- coding: utf-8 -*-
"""Phase 3022 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3022'
     r'\omega_p2p_l3_relay_neurons_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
MEMO_W = WLOG_DIR + r'\MEMORY.md'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['final_verdict']
assert verdict == 'relay_dedicated_coalition_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_content'] == 11
assert t2['n_sham'] == 11
assert t2['inter'] == 9728
assert t2['bf16_ident_med'] == 9.9714e-05
assert t2['conc_dom_med'] == 0.8236
assert t2['med_negmass'] == 0.1346
assert t2['med_posmass'] == 1.5067
assert t2['direction_counts'] == {'pos': 11}
assert t2['med_p_m_recomputed'] == 0.686
assert t2['med_e4_norm'] == 3.6017
assert t2['med_js_final_logic'] == 0.003332
assert t2['n_mass'] == 11
t2b = res['T2b']
assert t2b['jac_tag_med'] == 0.3617
assert t2b['jac_cross_med'] == 0.0
assert t2b['jac_same_tag_med'] == 0.0
assert t2b['jac_null_med'] == 0.0
assert t2b['p_tag'] == 0.0
assert t2b['p_cross'] == 1.0
assert t2b['n_pairs_cross'] == 121
t2c = res['T2c']
assert t2c['med_negmass']['content'] == 0.0575
assert t2c['med_posmass']['logic'] == 1.5067
assert t2c['med_posmass']['content'] == 0.4831
assert t2c['spec_ratio'] == 0.4274
assert t2c['med_e4_norm_content'] == 0.0268
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a26_3021'] is True
assert res['anchors']['a25_e4_chain_diff'] == 0.0
assert res['anchors']['a27_pm_chain_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3022
           for m in led['measurements']):
    claim = (
        'Omega-P2p (plan v5 P2) - L3 MLP relay '
        'neuron identity via SwiGLU attribution '
        '(3021: the injection is MLP-relay '
        'dominant, p_m 0.686).  Per logic tag, '
        'baseline vs g7-K-erased two-step chains '
        'capture the L3 down_proj input and MLP '
        'output; s_j = 2 dh_j (w_j.e4)/||e4||^2 '
        'with e4 the residual error entering L4 '
        '(e_all[3]==0 exactly so e4 = da_3 + dM_3); '
        'identity sum_j s_j == 2 dM.e4/||e4||^2 '
        '(bf16 gate med 9.97e-05 < 0.05); chain '
        'anchors a25 (med ||e4|| == 3.6017) and a27 '
        '(med p_m == 0.686) BOTH diff 0.0 bit-level '
        'vs sealed 3020/3021.  The within-tag '
        'permutation null is banned (permutation-'
        'invariant statistic, 3021 lesson); '
        'identity-sensitive coalition tests '
        'instead.  Verdict '
        'relay_dedicated_coalition_qwen (frozen '
        'map: jac_cross 0.0 not > 2*jac_null 0.0, '
        'jac_tag 0.3617 > 0 and p_tag 0.0 <= '
        '0.01).  RESULTS: (i) the relay is a '
        'SPARSE COALITION - top-32 of 9728 neurons '
        '(0.33 pct) carry 82.4 pct of the total '
        'attribution mass (conc_dom med 0.8236); '
        '(ii) the coalition is POSITIVE-DIRECTED '
        'in 11/11 tags (med posmass 1.5067 vs '
        'negmass 0.1346) - the L3 relay WRITES '
        'along e4, it does not suppress; (iii) '
        'cross-tag Jaccard of the coalitions '
        '0.3617 (55 pairs) vs random-32 null '
        'median 0.0, p = 0.0 - the SAME neurons '
        'are reused across prompts/positions '
        '(tag-consistent dedicated coalition); '
        '(iv) cross-phase vs the sealed 3019 '
        'mid-band suppression field top-32 at '
        'L10: Jaccard EXACTLY 0.0 in all 121 '
        'pairs (p_cross 1.0) AND opposite sign - '
        'the L3 positive relay coalition and the '
        'L8-20 negative suppression field are '
        'IDENTITY-DISJOINT INDEPENDENT CIRCUITS.  '
        'Specificity: relay mass scales with the '
        'input error (posmass logic 1.5067 vs '
        'content 0.4831, e4 norms 3.60 vs 0.027) '
        '- error-driven amplification, not a '
        'static feature.  CONCLUSION: gate '
        'architecture upgraded to seed (heads 31 '
        'pct) -> L3 positive sparse relay '
        'coalition (0.33 pct neurons, 82 pct of '
        'the write) -> generic negative '
        'equalizer field - three anatomically '
        'distinct stages.')
    meas = {
        'meas_id': 'meas3022_omega_p2p_l3_relay_'
                   'neurons_qwen',
        'phase': 3022,
        'claim': claim,
        'verdict': verdict,
        'anchors': '27/27 (a0-a24 as 3021; a25 e4 '
                   'chain diff 0.0; a26 3021 '
                   'integrity; a27 p_m chain diff '
                   '0.0)',
        'artifacts': {
            'result': 'phase3022/omega_p2p_'
                      'l3_relay_neurons_qwen/'
                      'result.json',
            'npz': 'phase3022/omega_p2p_'
                   'l3_relay_neurons_qwen/'
                   'omega_p2p_l3_relay_neurons_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative, no crashes; '
                'bf16 identity 9.97e-05 (gate 0.05); '
                'degenerate permutation null '
                'pre-banned; null median 0.0 noted - '
                'p values are the primary evidence.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 161
    l14['connects'].append({
        'meas_id': 'meas3022_omega_p2p_l3_relay_'
                   'neurons_qwen',
        'phase': 3022,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2p: L3 relay = '
                        'SPARSE POSITIVE DEDICATED '
                        'COALITION (top-32/9728 = '
                        '0.33 pct neurons carry '
                        '82.4 pct; 11/11 tags '
                        'positive; cross-tag '
                        'Jaccard .362 p=0; vs 3019 '
                        'suppression field Jaccard '
                        '0.0 disjoint AND opposite '
                        'sign) - seed -> positive '
                        'relay -> negative '
                        'equalizer, three distinct '
                        'stages'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
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
if '## Phase 3022:' not in memo:
    sec = u'''## Phase 3022: Ω-P2p L3 中继神经元身份——正性专属稀疏联盟，与抑制场完全不相交 [%(created)s]

**判决：`relay_dedicated_coalition_qwen`**（run1 权威一次通过，142.4s，锚 27/27：a1 轴身份 0.0 位级、a10 rel 0.0、a13 diff 0.0、a26 3021 完整性、**a25 ‖e4‖ diff 0.0 + a27 p_m diff 0.0 双链身份位级**；correction_note 空）

### 设计（3021 机器 verbatim + L3 SwiGLU 神经元归因）
3021 协议全继承（SEED_RND=3009 显式重建）。逐 logic tag 双链（baseline vs g7-K 擦除）捕获 **L3 down_proj 输入 h 与 MLP 输出**；因两链进入 L3 的残差相同（e_all[3]≡0），**e4 = Δa_3 + ΔM_3 精确**，配对 **s_j = 2·dh_j·(w_j·e4)/‖e4‖²**（w_j = W_down(L3) 第 j 列，inter=9728）；恒等式 Σs_j ≡ 2·ΔM·e4/‖e4‖²（bf16 门实测 **9.97e-05**）。**统计纪律升级：置换不变量 null 预先禁用**（3021 教训前移），改用身份敏感的 top-32 联盟集合检验：跨 tag Jaccard（55 对）+ 跨相位 Jaccard vs sealed 3019 L10 top-32（121 对），null=随机 32 子集（N=2000，seed 5918）。a25（med‖e4‖=3.6017）与 a27（med p_m=0.686）双链锚均 diff 0.0 位级——本相位与 3020/3021 的链完全 bit 级一致。

### 核心结果（重复三遍）
**① 中继是稀疏联盟**：9728 个神经元中 **top-32（0.33pct）承载 82.4pct 的归因质量**（conc_dom med 0.8236；11 tag 范围 0.695-0.895）；**② 中继是正性写入**：11/11 tag 主方向为 pos（med posmass **1.5067** vs negmass 0.1346）——L3 中继沿 e4 **正向放大**，不是抑制；**③ 联盟跨情景复用**：跨 tag Jaccard **0.3617**（null 中位 0.0，p_tag=**0.0**）——同一批神经元在全部 11 个（prompt,位置）组合中重复承担中继；**④ 与 3019 抑制场完全不相交且符号相反**：跨相位 Jaccard **精确 0.0**（121/121 对无一个共同神经元，p_cross=1.0）——L3 正性中继联盟与 L8-20 负性抑制场是**两个身份不相交的独立回路**。特异性：中继质量随输入误差缩放（posmass logic 1.51 vs content 0.48，e4 范数 3.60 vs 0.027）= 误差驱动的放大器，非静态特征。

### 判决与统计说明
判决按冻结映射 = dedicated_coalition（jac_cross 0.0 未过 shares 门；jac_tag 0.3617 > 0 且 p_tag 0.0 ≤ 0.01）。统计说明：大 inter 下随机 32 子集 Jaccard null 中位数=0，"2×null"门平凡通过——**主证据是 p 值**（2000 次 null 抽样无一 ≥ 0.3617），已在 ledger note 登记。conc_top32 属描述性浓度。

### 结论
门控三级架构的解剖学完成：**种子（头直写 31pct，GQA 限 group7）→ L3 正性稀疏中继联盟（0.33pct 神经元承载 82pct 写入）→ 通用负性均衡场（L8-20 分布式）**——三个阶段在神经元身份与符号上均两两可分。与"无单点头载体、机制=关系属性"的总结论一致：联盟是关系性的（误差驱动、跨情景复用同一组神经元），不是静态词特征。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3022/omega_p2p_l3_relay_neurons_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3023 = A（主选）**中继联盟因果承重**——L3 top-32 联盟神经元消融 vs 随机 32 神经元对照，JS 塌缩检验（功能主张的消融腿，三层分账纪律）；B 联盟读出解码（W_down 列方向 / 下游受众）；C L31 次峰定位；D 情景性检验。
''' % {'created': created,
           'script8': exe['script_sha256_8'],
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

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-20.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3022' not in prev:
    line = ('- Phase 3022 Omega-P2p: verdict '
            'relay_dedicated_coalition_qwen; L3 MLP '
            'relay SwiGLU neuron attribution (s_j = '
            '2 dh_j (w_j.e4)/||e4||^2, identity gate '
            '9.97e-05; a25/a27 chain anchors 0.0); '
            'relay = SPARSE POSITIVE DEDICATED '
            'COALITION: top-32/9728 (0.33 pct) carry '
            '82.4 pct (conc .824), 11/11 tags '
            'positive (posmass 1.51 vs negmass .13), '
            'cross-tag Jaccard .362 p=0 (same '
            'neurons reused); vs 3019 suppression '
            'field Jaccard EXACTLY 0.0 (121/121 '
            'disjoint) AND opposite sign = '
            'independent circuits; permutation null '
            'pre-banned (3021 lesson); gate = seed '
            '(heads 31) -> L3 positive relay -> '
            'generic negative equalizer; ledger '
            '161/L14 %d.\n'
            % len(l14['connects']))
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md rewrite (<=3000 chars) ----------
mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；产物 ...\\phase{N}\\{arm}\\；临时 gpt5_temp\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结（PREREG/锚/判决）→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）。
3. 重跑先删旧产物；负结果如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格（{key} 不替换，2971）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；大内积阈 1e-8；跨相位锚 max|Δ|；链身份锚可多点（‖e4‖=3.6017 与 p_m=.686，3022 a25/a27=0.0）。

## 统计判据纪律
- 判据可达性先检（永真禁用）；top-k/部分和类=置换不变→null 退化 p≡1（3021）；大 inter 下随机子集 Jaccard null 中位可=0，"2×null"门平凡通过→以 p 值为主证据（3022）；margin n≳40+粒度；quasi-post-hoc 标注；大 family maxT；显著集重叠 null 校准；镜像 −dirs 必配；功能主张三层分账报效应量；跨产物对照先查类目对齐；分组结构零先检。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签（禁混用）→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；args 空用 kwargs；单样本保 batch 维；真残差流=decoder-layer pre-hook（3017）；恒等门分母与噪声同尺度；attn 输出 tuple；norm pre-hook 捕 final-norm 输入；np.stack 单元素降维（3004）；transpose(1,3,0,2)（2970）；跨 Phase 常量重建；logit-lens 全 bf16 对齐（3020）；einsum 标签逐一核对（3021）；CUDA 张量与权重同 device（W_down 勿 .cpu()，3022）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改（chr(96)）。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁；replace 未命中→先 Grep；Edit 误插裸文本=语法雷，改后必编译检查（3020）。
- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接。
- numpy 标量入 json 转 int()/float()；GQA：KV 缓存 8 头，头级干预单位=KV 头（3015）。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3022）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/主调制×词类特化/fr=重写/双轴注入/剂量窗/h12/正交重定向/局部化/perp=重写；2992-3010：符号率/签名稳健/轴防御/Ω-F 洗消；GLM4 86 vs qwen 63pct；3007 锁定 .457；3008 held-out 分离；3009 KV 饱和；3010 换读出→logic 位因果特异。
Ω-P2（3011-3022）：3011 **门控=L3 KV**（D=.0137 p 1e-4，次峰 L31）；3012 混合；3013 情景码 LOO .397；3014 剂量非单调（半≥全）=K 路由；3015 K 消费=领先 g7+情景（.390）；3016 放大分布式；3017 吸收混合（反平行中带）；3018 **抵消主导**（credit 2.51 nats，MLP 承载 L8-20）；3019 抵消带=通用抑制场（top32 6.7pct；spec 1.12）；3020 **读出特异=注入特异**（L4 即时 JS 944×→下游稀释 28.3×=均衡器）；3021 **注入=MLP 中继主导 69pct**（头 31pct GQA 限 group7、28/29 主导、跨组精确 0）；3022 Ω-P2p **中继=专属正性稀疏联盟**——top32/9728 承载 82pct（conc .824），11/11 正向（pos 1.51 vs neg .13），跨 tag Jaccard .362 p≈0（跨情景复用）；vs 3019 抑制场 Jaccard **0.0 全不相交且符号相反**=独立回路；a25/a27 链身份 0.0。**门控三级=种子（头 31pct）→L3 正性中继联盟→通用负性均衡场**。核心：null 重编码全层分布式涌现；头级重要性=关系属性。

## 下一步
- max=3022，下一个 3023（A 主选 **中继联盟因果承重**——L3 top32 联盟消融 vs 随机 32 神经元对照的 JS 塌缩，功能分账消融腿；B 联盟读出解码；C L31 次峰；D 情景性检验）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
