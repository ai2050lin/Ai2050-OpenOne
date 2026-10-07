# -*- coding: utf-8 -*-
"""Phase 3021 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3021'
     r'\omega_p2o_injection_anatomy_qwen')
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
assert verdict == 'injection_mixed_qwen', verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_content'] == 11
assert t2['n_sham'] == 11
assert t2['ident_med'] == 0.01409873
assert t2['conc_top3_med'] == 0.3141
assert t2['p_perm_med'] == 1.0
assert t2['p_m_med'] == 0.686
assert t2['med_js_final_logic'] == 0.003332
assert t2['med_e4_norm'] == 3.6017
assert t2['top3_per_tag'][0] == [28, 29, 26]
t2b = res['T2b']
assert t2b['kv_group_med_p']['7'] == 0.3132
assert t2b['kv_group_med_p']['0'] == 0.0
assert t2b['spearman_3015'] == 0.214
assert t2b['top1_kv_group'] == 7
assert t2b['a25_e4_diff'] == 0.0
t2c = res['T2c']
assert t2c['med_p_m']['logic'] == 0.686
assert t2c['act_ratio_logic_content'] == 52.0
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a24_3020'] is True
assert res['anchors']['a25_e4_chain_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3021
           for m in led['measurements']):
    claim = (
        'Omega-P2o (plan v5 P2) - injection-end '
        'anatomy: query-head decomposition of the L4 '
        'injection error at L3 (3020: the 30x gap is '
        'injection-born, L4 immediate JS 944x).  At '
        'the gate layer L3 the o_proj input per '
        'query head (32x128) and the MLP output are '
        'captured baseline vs g7-K-erased; exact '
        'identity e4 = W_o @ dConcat + dM (bf16 '
        'gate med 0.0141 < 0.05); shares p_q = '
        '(W_o^q da_q).e4/||e4||^2, p_m likewise, '
        'sum == 1.  Verdict injection_mixed_qwen '
        '(frozen map: conc 0.314 < 0.5 AND '
        'p_perm 1.0 > 0.01).  RESULTS: (i) the '
        'injection is MLP-RELAY DOMINANT - the L3 '
        'MLP channel carries 69 pct of the error '
        '(p_m med 0.686), the direct attention '
        'rewrite only 31 pct, i.e. the K-erasure '
        'effect is immediately amplified 2.2x '
        'through the gate-layer MLP; (ii) the head '
        'channel is GQA-STRUCTURALLY CONFINED - '
        'only group-7 query heads (28-31) read '
        'g7-K, cross-group shares are EXACTLY 0.0 '
        '(structural validation), within group 28/'
        '29 dominate (top3 = 28,29 + one of 26/30/'
        '31 in all 11 tags) with all-positive '
        'shares (no cancellation at injection); '
        '(iii) injection activity logic/content = '
        '52x; (iv) Spearman vs 3015 consumer '
        'ranking 0.214 - the linkage is '
        'category-confused (3015 measures the '
        'effect of erasing EACH KV head; 3021 '
        'measures the response TO g7 erasure, '
        'structurally confined to group 7) and '
        'only confirms the structural zero.  '
        'REGISTERED DEFECTS: (1) the permutation '
        'null is DEGENERATE - top-3 partial sums '
        'are permutation-invariant, p == 1.0 is '
        'unreachable below the 0.01 gate (the '
        'criterion-reachability lesson recurring); '
        '(2) the 3015 linkage category error.  '
        'CONCLUSION: seed composition = 31 pct '
        'structurally-confined head direct write + '
        '69 pct immediate MLP relay - even the '
        'injection itself is mostly MLP-mediated; '
        'gate architecture = position-specific '
        'seed (heads 31 pct) -> L3 MLP relay (69 '
        'pct) -> generic equalizer field.  Fifth '
        'independent convergence: no single-point '
        'head carrier, mechanism is a RELATIONAL '
        'property.')
    meas = {
        'meas_id': 'meas3021_omega_p2o_injection_'
                   'anatomy_qwen',
        'phase': 3021,
        'claim': claim,
        'verdict': verdict,
        'anchors': '25/25 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013; a18 3014; a19 '
                   '3015; a20 3016; a21 3017; a22 '
                   '3018; a23 3019; a24 3020; a25 e4 '
                   'chain diff 0.0)',
        'artifacts': {
            'result': 'phase3021/omega_p2o_'
                      'injection_anatomy_qwen/'
                      'result.json',
            'npz': 'phase3021/omega_p2o_'
                   'injection_anatomy_qwen/'
                   'omega_p2o_injection_anatomy_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative, no crashes; '
                'identity gate 0.0141 (gate 0.05); '
                'a25 chain identity vs 3020 med e4 '
                'diff 0.0 (bit level); degenerate '
                'permutation null and 3015-linkage '
                'category error registered in claim.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 160
    l14['connects'].append({
        'meas_id': 'meas3021_omega_p2o_injection_'
                   'anatomy_qwen',
        'phase': 3021,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2o: query-head '
                        'decomposition of the L4 '
                        'injection - MLP-RELAY '
                        'DOMINANT (L3 MLP carries 69 '
                        'pct of the seed, direct '
                        'attention write 31 pct '
                        'confined by GQA to group-7 '
                        'heads 28/29, cross-group '
                        'exactly 0.0) - gate = seed '
                        '(31) -> MLP relay (69) -> '
                        'generic equalizer; '
                        'permutation-null degeneracy '
                        'registered'})
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
if '## Phase 3021:' not in memo:
    sec = u'''## Phase 3021: Ω-P2o 注入端解剖——种子=L3 MLP 中继主导，头通道被 GQA 结构性限定 [%(created)s]

**判决：`injection_mixed_qwen`**（run1 权威一次通过，145.1s，锚 25/25：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、a13 diff 0.0、a24 3020 完整性、**a25 链身份 vs 3020 med‖e4‖ diff 0.0 位级**；correction_note 空）

### 设计（3020 机器 verbatim + L3 注入的 query 头分解）
3020 协议全继承（SEED_RND=3009 显式重建）。L3（门控层）捕获 **o_proj 输入逐 query 头（32×128，o_proj pre-hook）+ MLP 输出**（step-2 位）；因两链进入 L3 的残差相同，恒等式 **e4 = W_o·Δconcat + ΔM** 精确成立（bf16 门 med<0.05，实测 **1.41e-02**）；份额 **p_q = (W_o^q·Δa_q)·e4/‖e4‖²，p_m 同理，Σp_q+p_m ≡ 1**。PRIMARY conc_top3（top-3 p_q 和）+ tag 内置换 null（N=10000）；T2b GQA 分组（q//4）聚合 vs 3015 sealed med_share_top8 的 Spearman（descriptive）；T2c content/sham 同管线。判决映射（冻结）：conc≥0.5 且 p≤0.01 → concentrated；conc<0.5 且 p≤0.01 → distributed；否则 mixed。

### 核心结果（重复三遍）
**① 注入是 MLP 中继主导**：L3 误差的 **69pct 由 MLP 通道承载（p_m med 0.686）**，头直写仅 31pct——g7-K 擦除的直接注意力改写被门控层 MLP **即刻放大 2.2×**；**② 头通道被 GQA 结构性限定**：只有 group 7 的 4 个 query 头（28-31）读 g7 的 K，跨组份额 med **精确 0.0**（结构验证恒等），组内 **28/29 主导**（11 tag 的 top3 恒为 28,29,+26/30/31 之一），份额全正、注入处零抵消（Σ|p|≈|Σp|）；**③ 注入活动 logic/content 比 52×**（conc 0.314 vs 0.006，p_m content 亦 0.44=MLP 主导同构）；**④ Spearman vs 3015 = 0.214**——该对照属**类目错位**（3015 测"逐个擦各 KV 头的效应"，3021 测"对 g7 擦除的响应"，后者结构上必限 group 7），仅确证结构零。

### 判决与硬伤（2 笔统计缺陷，如实登记）
判决按冻结映射 = mixed（conc 0.314<0.5 且 p_perm 1.0>0.01）。**缺陷 1：置换 null 退化**——conc_top3 是 top-3 部分和=**置换不变量**，null 分布≡obs，p≡1.0、0.01 门不可达："判据可达性先检"纪律复发（top-k/部分和类统计量须先检置换不变性并改用非退化 null），已写入 MEMORY；**缺陷 2：T2b 对照类目错位**（上述④）。两者不动摇实质结论：69/31 分账、结构零、a25=0.0 均为精确记账。

### 结论
种子的构成 = **31pct 结构性头直写（group 7 专用，28/29 主导）+ 69pct L3 MLP 即刻中继**——连注入本身都主要是 MLP 介导的。与 3019"通用抑制场"、3016"分布式放大"完全同构：**门控三级架构 = 位置特异种子（头 31pct）→ L3 MLP 中继（69pct）→ 通用均衡场**。第五个独立收敛：无单点头载体，机制=关系属性。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3021/omega_p2o_injection_anatomy_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3022 = A（主选）**L3 MLP 中继的 SwiGLU 神经元归因**——ΔM 的神经元分解（3019 方法 verbatim 用在 L3 注入步），中继联盟 vs 3019 中带抑制场 top-32 集合的 Jaccard（同一联盟还是新联盟）；B q28/29 头身份（把 g7-K 读成什么方向）；C L31 次峰定位；D 情景性检验。
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
if 'Phase 3021' not in prev:
    line = ('- Phase 3021 Omega-P2o: verdict '
            'injection_mixed_qwen; query-head '
            'decomposition of the L4 injection at L3 '
            '(identity e4 = W_o dConcat + dM, gate '
            '0.0141); MLP-RELAY DOMINANT: L3 MLP '
            'carries 69 pct of the seed (p_m 0.686), '
            'head write 31 pct GQA-confined to group '
            '7 (q28/29 dominant, cross-group exactly '
            '0.0, no cancellation); act ratio 52x; '
            'a25 chain identity vs 3020 diff 0.0; '
            'defects registered: degenerate '
            'permutation null (top-3 sum is '
            'permutation-invariant, p=1.0) + 3015 '
            'linkage category error; gate = seed(31) '
            '-> MLP relay(69) -> equalizer; ledger '
            '160/L14 %d.\n'
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
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；大内积阈 1e-8；跨相位锚 max|Δ| vs 上 Phase。

## 统计判据纪律
- 判据可达性先检（永真禁用）；**top-k/部分和类统计量=置换不变→null 退化 p≡1（3021）——先检置换不变性，改非退化 null**；margin n≳40+粒度；quasi-post-hoc 标注；大 family maxT；显著集重叠 null 校准；镜像 −dirs 必配；功能主张三层分账报效应量；**跨产物对照先查类目对齐（3021 vs 3015 类目错位）**；分组结构零先检（GQA 组外必 0）。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签（禁混用）→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；args 空用 kwargs；单样本保 batch 维；真残差流=decoder-layer pre-hook（3017）；恒等门分母与噪声同尺度；attn 输出 tuple；norm pre-hook 捕 final-norm 输入；np.stack 单元素降维（3004）；per-(l,h) transpose(1,3,0,2)（2970）；跨 Phase 常量重建；logit-lens=residual→final RMSNorm→lm_head 全 bf16 对齐（3020 cons 0.0）；**einsum 标签逐一核对（3021）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改（chr(96)）。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁；replace 未命中→先 Grep；Edit 误插裸文本=语法雷，改后必编译检查（3020）。
- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接。
- numpy 标量入 json 转 int()/float()；GQA：KV 缓存 8 头，头级干预单位=KV 头（3015）。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3021）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁；主调制×词类特化/fr=重写/双轴注入/剂量窗/h12/正交重定向/局部化/perp=重写；2992-3010：符号率快照/签名稳健/轴防御/Ω-F xdir 洗消；GLM4 词携带 86pct vs qwen 63pct（构造 vs 破坏）；中带同构=预训练涌现；3007 锁定 .457；3008 held-out 分离；3009 KV 饱和；3010 换读出→logic 位因果特异。
Ω-P2（3011-3021）：3011 **门控=L3 KV**（D=.0137 p 1e-4，次峰 L31）；3012 混合；3013 情景码 LOO .397；3014 剂量非单调（半≥全）K/V 分岔=K 路由；3015 K 消费=领先 g7+情景（share .390）；3016 放大=分布式+深层收敛（门控=种子）；3017 吸收混合（反平行中带 L5-25）；3018 **抵消主导**（credit 2.51>衰减 1.28 nats，MLP 承载带 L8-20）；3019 **抵消带=分布式通用抑制场**（top32 6.7pct p .60；spec 1.12）；3020 Ω-P2n **读出特异=注入特异**——L4 即时 JS 944×，下游把 944× 稀释回 28.3×（均衡器非放大器）；3021 Ω-P2o **注入=L3 MLP 中继主导 69pct**（p_m .686；头 31pct 被 GQA 限 group7，28/29 主导、无抵消、跨组精确 0=结构验证；act 比 52×；vs 3015 Spearman .214=类目错位登记；a25 链身份 0.0）；判决 mixed（置换退化腿登记）。**门控三级架构=种子（头 31pct）→L3 MLP 中继（69pct）→通用均衡场**。核心：null 重编码全层分布式涌现；头级重要性=关系属性。

## 下一步
- max=3021，下一个 3022（A 主选 **L3 MLP 中继 SwiGLU 神经元归因**——ΔM 神经元分解，中继联盟 vs 3019 抑制场 top32 Jaccard；B q28/29 身份；C L31 次峰；D 情景性检验）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
