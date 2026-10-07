# -*- coding: utf-8 -*-
"""Phase 3027 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3027'
     r'\omega_p2u_consumer_heads_qwen')
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
assert verdict == 'consumption_mixed_qwen', verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['conc8_med'] == 0.5828
assert t2['null_med_med'] == 0.5611
assert t2['diff_med'] == 0.017
assert t2['n_pos'] == 7
assert t2['n_neg'] == 4
assert t2['p_binom_pos'] == 0.2744
assert t2['p_binom_neg'] == 0.8867
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a30_capture_self_diff'] == 0.0
assert res['anchors']['a35_abl_chain_diff'] == 0.0
assert res['anchors']['a36_content_co_diff'] == 0.0
t2b = res['T2b']
assert t2b['spearman_gqa_vs_3015'] == 0.1429
assert t2b['top8_pair_jaccard_med'] == 0.3333
assert t2b['spearman_abl_vs_erase_med'] == 0.8079
t2c = res['T2c']
assert t2c['e4cos_abs_med'] == 0.0224
assert t2c['e4cos_max'] == 0.1415
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a33_3025'] is True

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3027
           for m in led['measurements']):
    claim = (
        'Omega-P2u (plan v5 P2) - localizes the '
        'downstream consumers of the L3 coalition '
        'baseline output (3023: 55.5 pct of the L3 '
        'MLP output norm; 3024: causally load-'
        'bearing) at L4 = L3+1 per query head (32x'
        '128, o_proj input side): 3023 machine '
        'verbatim run_chain/hook_abl; per logic '
        'tag capture_base + ERASE (a28) + '
        'ABL_COAL zero-ablation (a35 vs sealed '
        '3023 npz js_abl_only bit-level 0.0) + 4 '
        'random-32 non-coalition null chains; per '
        'content position co chain (a36 vs 3023 '
        'js_coal_content bit-level 0.0).  T2a '
        'PRIMARY: conc8 (top-8 per-head norm '
        'share of the L4 delta) vs random-null '
        'median, exact binomial sign test.  '
        'Verdict consumption_mixed_qwen (frozen '
        'map: neither sign branch significant).  '
        'RESULTS: (i) coalition consumption is '
        'NOT head-specialized: conc8 med 0.5828 '
        'vs random-32 null 0.5611 (diff +0.017, '
        'n_pos 7 / n_neg 4, p 0.27) - and the '
        'null itself is far above uniform '
        '(0.25): ANY L3 MLP perturbation pulse '
        'is read out by the same structured '
        'head mode (8/32 heads carry ~56 pct), '
        'the concentration is a generic readout '
        'property, not a coalition channel; '
        '(ii) the consumption profile matches '
        'the erase-response profile per tag '
        '(Spearman med 0.808) - the same heads '
        'consume the baseline and transmit the '
        'erase response; top-8 head sets are '
        'tag-stable (pair Jaccard 0.333); vs '
        '3015 K-consumer GQA shares rho 0.143 '
        '(category mismatch confirmed third '
        'time); (iii) per-head residual '
        'contributions are near-orthogonal to '
        'e4 (|cos| med 0.022, max 0.142) - '
        'consistent with 3020 cos(e35,u35) '
        '0.047: the consumed signal does not '
        'travel along the erase direction; '
        '(iv) PROTOCOL LIMIT registered: the '
        'content ab chain is an exact rerun of '
        'the abl chain (with erase=False the '
        'p_pos argument does not enter the '
        'chain), so the content conc8 carries '
        'no independent information - L3 MLP '
        'ablation only acts at the step-2 '
        'position, an independent content-'
        'position consumption profile is not '
        'reachable under the two-step protocol; '
        '(v) quadruple chain identity: a28/a30/'
        'a35/a36 all bit-level 0.0.  '
        'INTERPRETATION: head-level readout is '
        'a context property, not a pathway '
        'property - seventh convergence (no '
        'head-specialized consumer circuit for '
        'the coalition; readout mode is set by '
        'prompt/context and reused across '
        'perturbation types).  NEXT: '
        'amplification dose symmetry, L31 '
        'secondary peak, situational '
        'specificity, or rho_er head-level '
        'identity.')
    meas = {
        'meas_id': 'meas3027_omega_p2u_consumer_'
                   'heads_qwen',
        'phase': 3027,
        'claim': claim,
        'verdict': verdict,
        'anchors': '30/30 core (a0-a27 as 3026; '
                   'a28 erase vs 3022 bit-level '
                   '0.0; a29 3023; a30 capture '
                   'self 0.0; a31 3024; a33 3025 '
                   'integrity; a35 ABL_COAL vs '
                   '3023 npz js_abl_only bit-'
                   'level 0.0; a36 content co vs '
                   '3023 js_coal_content bit-'
                   'level 0.0)',
        'artifacts': {
            'result': 'phase3027/omega_p2u_'
                      'consumer_heads_qwen/'
                      'result.json',
            'npz': 'phase3027/omega_p2u_'
                   'consumer_heads_qwen/'
                   'omega_p2u_consumer_heads_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run6 authoritative (147.0s; run1-5 '
                'crashes: pre_attn hook omitted, '
                'step-2 single-token index, l4_cap '
                'wiped by clear_cap, 1-d shares_of, '
                'missing conc8_null_all - all '
                'writing-phase defects, verdict map '
                'frozen); consumption is generic-'
                'readout concentrated (conc8 0.583 '
                'vs null 0.561, both >> 0.25 '
                'uniform), coalition-identity-'
                'null; consumption vs erase-'
                'response profile Spearman 0.808.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 166
    l14['connects'].append({
        'meas_id': 'meas3027_omega_p2u_consumer_'
                   'heads_qwen',
        'phase': 3027,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2u: coalition '
                        'baseline consumption is '
                        'NOT head-specialized - '
                        'conc8 0.583 vs random-32 '
                        'null 0.561 (p 0.27), both '
                        '>> uniform 0.25: any L3 '
                        'perturbation is read by '
                        'the same structured head '
                        'mode; consumption profile '
                        '= erase-response profile '
                        '(Spearman 0.808); '
                        'contributions near-'
                        'orthogonal to e4 (0.022); '
                        'head readout = context '
                        'property (seventh '
                        'convergence)'})
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
if '## Phase 3027:' not in memo:
    sec = u'''## Phase 3027: Ω-P2u 联盟基线功能的下游消费头定位——消费=通用读出结构 [%(created)s]

**判决：`consumption_mixed_qwen`**（run6 权威 147.0s，锚 **30/30**：a28 ERASE vs 3022、a30 捕获自洽、**a35 ABL_COAL vs 3023 npz js_abl_only、a36 content co vs 3023 js_coal_content 四重位级 0.0**；run1-5 崩溃如实登记：pre_attn hook 漏挂、step-2 单 token 索引越界、clear_cap 清空 l4_cap、shares_of 1 维、conc8_null_all 漏声明——全部写作期缺陷，判决映射未改）

### 设计（3023 机器 verbatim run_chain/hook_abl + 3026 锚链）
每 logic tag 四链：capture_base + ERASE（a28 + e4）+ ABL_COAL（联盟 32 神经元零消融、无擦除，a35 链身份）+ 4 条随机 32 非联盟 null 链（SEED_RND+28）。在 L4=L3+1 的 o_proj 输入侧（32×128）取 step-2 位置 delta，p_h = 逐头范数份额，conc8 = top-8 份额。content 位另跑 ab/co 链（a36）。

### 核心结果（重复三遍）
**① 联盟消费不头特异**：conc8 med **0.5828** vs 随机 32 null **0.5611**（diff +0.017，n_pos 7 / n_neg 4，p 0.27，mixed 分支）——但 null 本身远高于均匀值 0.25：**任何 L3 MLP 扰动脉冲都被同一结构化头模式读出（8/32 头承载约 56%%），集中是通用读出性质，不含联盟特异通道**。**② 消费画像 = 擦除响应画像**：逐 tag Spearman 中位 **0.808**——同一批头既消费联盟基线输出又传导擦除响应；top-8 头集合跨 tag Jaccard **0.333**（稳定）；vs 3015 K 消费者 GQA 份额 rho 0.143（类目错位第三次确认）。**③ 消费不沿擦除方向**：逐头残差贡献 vs e4 的 |cos| 中位 **0.022**（max 0.142）——与 3020 cos(e35,u35)=0.047 收敛，被消费的信号不走语言轴。

### 协议局限（如实登记）
content ab 链与 abl 链**参数完全等价**（erase=False 时 p_pos 不进入链）→ content conc8 是 abl 链的精确重复、无独立信息；L3 MLP 消融只作用于 step-2 位置，**content 位置的独立消费画像在两步协议下不可得**（3023 content 结果有效因其干预是 prefill KV 擦除）。

### 机制结论
**头级读出 = 上下文属性，非通路属性**（第七次收敛）：读出模式由 prompt/上下文设定并在不同干预类型间复用；联盟的身份性在其输出被消费的方式中不可见。三级架构的读出端补全：种子→中继→均衡场之后，**下游头读出对来源不敏感**——机制的整体图景是"上下文设定的固定读出模式 × 来源无关的扰动传播"。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3027/omega_p2u_consumer_heads_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3028 = A（主选）**放大式干预剂量对称性**——把 3024 恢复式 patch 的联盟变化加倍（2× patch），测 JS 响应的线性/非线性（剂量第三点，补全 3014 非单调剂量律在联盟通道的形状）；B L31 次峰定位；C 情景性检验；D rho_er=0.808 的头级身份（消费/响应共同头的 GQA 归属）。
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
if 'Phase 3027' not in prev:
    line = ('- Phase 3027 Omega-P2u: verdict '
            'consumption_mixed_qwen (run6 '
            'authoritative 147.0s after 5 writing-'
            'phase crash fixes, anchors 30/30, '
            'a28/a30/a35/a36 all bit-level 0.0, '
            'a35 = ABL_COAL vs sealed 3023 npz '
            'js_abl_only, a36 = content co vs '
            'js_coal_content); coalition baseline '
            'consumption is NOT head-specialized: '
            'conc8 0.5828 vs random-32 null 0.5611 '
            '(diff +0.017, p 0.27) and the null is '
            'far above uniform 0.25 = ANY L3 '
            'perturbation is read by the same '
            'structured head mode (generic readout, '
            'no coalition channel); consumption '
            'profile = erase-response profile '
            '(Spearman med 0.808), top-8 sets tag-'
            'stable (Jaccard 0.333), vs 3015 GQA '
            'rho 0.143 (category mismatch x3); '
            'per-head contributions near-orthogonal '
            'to e4 (|cos| med 0.022 vs 3020 0.047); '
            'PROTOCOL LIMIT: content ab chain = '
            'exact rerun of abl chain (p_pos does '
            'not enter with erase=False), content '
            'conc8 non-independent; head readout = '
            'context property, seventh convergence; '
            'ledger 166/L14 134.\n')
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
3. 重跑先删旧 execution/result/npz；负结果如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格（{key} 不替换，2971）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；链身份锚多点（js 序列、js(pb,p0)=0.0、vs 3023 js_abl_only / js_coal_content、3024/3025 npz，均 0.0）。

## 统计判据纪律
- 判据可达性先检：置换不变 null p≡1（3021）；消融类预检毒性门（3023）→基线恢复 patch（3024）；宽 patch bf16 噪声底线（3025）；null 后处理抽取写作期预检（3026）；**干预参数在链中真正生效才作对照（3027：erase=False 时 p_pos 不进链，content ab=abl 重复）**；中位数不可加；margin n≳40；maxT；镜像 −dirs 必配。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+竞争重平衡→功能局域≠几何符号身份→**读出集中度须对照"任意扰动"null：conc8 高≠通路特异（3027 null 0.561 vs 均匀 0.25）**。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；真残差流=decoder-layer pre-hook；npz dict→0-d 读回 .item()；hook 改输出用返回值+active 门；权重列 .detach()；bf16 hook 配 bf16 列；**step-2 前向单 token：捕获取 [0,-1]；clear_cap 每链清缓存→链后立即提取捕获（3027）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；**triple-quote 内行尾 \\ 是行继续转义会吃掉换行→补丁锚串禁行尾反斜杠（3027）**；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3027）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3027）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018 抵消主导；3019 抵消带=通用抑制场；3020 读出特异=注入特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 正性专属稀疏联盟（top32=82pct）；3023 零消融有毒；3024 联盟因果承重（collapse 0.657）；3025 非联盟变化保护性（B1 边缘带）；3026 B1 rank 特异但符号 null-like；3027 **消费=通用读出结构**（conc8 0.583 vs null 0.561 p=0.27，均>>均匀 0.25；消费画像=擦除响应画像 Spearman 0.808；贡献 ⊥ e4 0.022；头读出=上下文属性，第七次收敛）。核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性。

## 下一步
- max=3027，下一个 3028（A 主选 **放大式干预剂量对称性**——恢复式 patch 联盟变化 2×，测 JS 响应线性/非线性，补 3014 剂量律在联盟通道的第三点；B L31 次峰定位；C 情景性检验；D rho_er 头级身份）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
