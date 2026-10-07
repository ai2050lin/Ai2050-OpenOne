# -*- coding: utf-8 -*-
"""Phase 3026 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3026'
     r'\omega_p2t_b1_edge_band_qwen')
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
assert verdict == 'b1_protective_null_like_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['share_b1_med'] == 0.4383
assert t2['null_med_med'] == 0.4475
assert t2['diff_med'] == -0.0067
assert t2['n_pos'] == 4
assert t2['n_neg'] == 7
assert t2['p_binom_pos'] == 0.8867
assert t2['p_binom_neg'] == 0.2744
assert t2['b1_rel_med'] == 0.126397
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a30_capture_self_diff'] == 0.0
assert res['anchors']['a34_b1_restore_diff'] == 0.0
t2b = res['T2b']
assert t2b['b1_pair_jaccard_med'] == 0.261
assert t2b['rand500_pair_jaccard_med'] == 0.0256
assert t2b['j19_same_med'] == 0.0057
assert t2b['j19_all_max'] == 0.0133
assert t2b['b1_coalition_overlap_max'] == 0.0
t2c = res['T2c']
assert t2c['b1_protect_hi_frac'] == 0.024
assert t2c['b1_protect_lo_frac'] == 0.618
assert t2c['b1_protect_mid_frac'] == 0.358
assert t2c['coal_pos_hi_frac'] == 0.1875
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a30_capture_self_diff'] == 0.0
assert res['anchors']['a33_3025'] is True
assert res['anchors']['a34_b1_restore_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3026
           for m in led['measurements']):
    claim = (
        'Omega-P2t (plan v5 P2) - characterizes the '
        'B1 protective edge band localized in 3025 '
        '(rank 33-532 of the non-coalition by 3022 '
        '|s|, 500 neurons): per logic tag, capture '
        'no-erase baseline + ERASE (hcap) + '
        'RESTORE_B1; per-neuron e4 projection '
        'dproj = (h_er - h_base) * (W_down.T @ '
        'e4_unit); T2a PRIMARY: B1 protective mass '
        'share vs random 500-subset null of the '
        'non-coalition (200 draws, post-hoc on '
        'captured dproj - no extra model runs); '
        'T2b identity: B1 cross-tag Jaccard + vs '
        '3019 suppression top-32 (3022 convention '
        'most-negative s_prim); T2c per-neuron '
        'sign consistency.  Verdict '
        'b1_protective_null_like_qwen (frozen '
        'map: diff med -0.0067, neither sign '
        'branch significant).  RESULTS: '
        '(i) B1 is NOT sign-specialized: its '
        'protective mass share 0.4383 is null-'
        'like vs random 500-subsets (null med '
        '0.4475; diff med -0.0067, n_pos 4 / '
        'n_neg 7, p_pos 0.89 / p_neg 0.27) - '
        'the 3025 functional localization of '
        'the rebalancing in the B1 band does '
        'NOT come with a geometric sign '
        'signature concentrated there; (ii) B1 '
        'IS a stable population: cross-tag '
        'pairwise Jaccard 0.261 vs random-500 '
        '0.0256 (10.2x) - the high-|s| fringe '
        'neurons re-select across tags, rank-'
        'special not sign-special; (iii) B1 is '
        'distinct from the 3019 suppression '
        'top-32 (same-tag Jaccard 0.0057, all-'
        '121-pairs max 0.0133, near-random) '
        'and disjoint from the 3022 coalition '
        '(0 by construction, asserted); '
        '(iv) per-neuron sign consistency: '
        'within B1 only 2.4 pct of neurons '
        'protective in >=10/11 tags while '
        '61.8 pct protective in <=1/11 - the '
        'protective sign structure is '
        'population-level and tag-dependent, '
        'NOT carried by stable per-neuron '
        'sign specialists (coalition '
        'concurrent consistency also modest '
        'at 18.75 pct); (v) triple chain '
        'identity: a28 ERASE vs 3022 js bit-'
        'level 0.0, a30 capture self-'
        'consistency 0.0, a34 RESTORE_B1 vs '
        'sealed 3025 npz js_b1 bit-level 0.0; '
        'a33 3025 integrity.  INTERPRETATION: '
        'functional localization != geometric '
        'sign identity - the L3 push-pull '
        'rebalancing (3025) is carried by a '
        'stable high-|s| population whose '
        'protective direction is diffuse and '
        'tag-dependent; sixth convergence: no '
        'per-neuron sign carriers, '
        'population-level properties only.  '
        'NEXT: coalition baseline function / '
        'downstream consumers, amplification '
        'dose symmetry, L31 secondary peak, '
        'or situational specificity.')
    meas = {
        'meas_id': 'meas3026_omega_p2t_b1_edge_'
                   'band_qwen',
        'phase': 3026,
        'claim': claim,
        'verdict': verdict,
        'anchors': '30/30 core (a0-a27 as 3025; '
                   'a28 erase vs 3022 bit-level '
                   '0.0; a29 3023; a30 capture '
                   'self 0.0; a31 3024; a33 3025 '
                   'integrity; a34 RESTORE_B1 vs '
                   '3025 npz js_b1 bit-level '
                   '0.0)',
        'artifacts': {
            'result': 'phase3026/omega_p2t_'
                      'b1_edge_band_qwen/'
                      'result.json',
            'npz': 'phase3026/omega_p2t_'
                   'b1_edge_band_qwen/'
                   'omega_p2t_b1_edge_band_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative one-pass (149.2s, '
                'no crash); B1 band is rank-'
                'special and stable (cross-tag '
                'Jaccard 0.261 vs 0.026 random) '
                'but sign-null-like (share 0.438 '
                'vs null 0.448) - functional '
                'localization without geometric '
                'sign identity; protection is '
                'population-level, per-neuron '
                'sign flips across tags.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 165
    l14['connects'].append({
        'meas_id': 'meas3026_omega_p2t_b1_edge_'
                   'band_qwen',
        'phase': 3026,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2t: B1 protective '
                        'fringe is rank-special but '
                        'sign-null-like - protective '
                        'mass share 0.438 vs random-'
                        '500 null 0.448 (p 0.27), '
                        'cross-tag Jaccard 0.261 (10x '
                        'random), distinct from 3019 '
                        'suppression (<=0.013) and '
                        'coalition (0); per-neuron '
                        'sign flips across tags '
                        '(61.8 pct protective in <=1/'
                        '11) - rebalancing is '
                        'population-level, no '
                        'per-neuron sign carriers'})
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
if '## Phase 3026:' not in memo:
    sec = u'''## Phase 3026: Ω-P2t B1 保护性边缘带刻画——带稳定但符号不特异 [%(created)s]

**判决：`b1_protective_null_like_qwen`**（run1 权威一次通过，149.2s，无崩溃，锚 **30/30**：a28 ERASE vs 3022、a30 捕获自洽、**a34 RESTORE_B1 vs 3025 npz js_b1 三重位级 0.0**，a33 3025 完整性；correction_note 空）

### 设计（3025 机器 verbatim，null 判据可达性写作期预检）
每 logic tag 三链：capture_base（h_base）+ ERASE（hcap）+ RESTORE_B1（链身份锚；B1 = 3022 |s| 排名 33-532 非联盟 500 神经元，verbatim 重建）。逐神经元 e4 投影 dproj = (h_er − h_base) · (W_down.T @ e4_unit)。**T2a 主检验**：B1 保护性质量份额 vs 随机 500 非联盟子集 null（200 次抽取，纯后处理零额外模型开销）；T2b 身份：B1 跨 tag Jaccard + vs 3019 抑制 top-32（3022 约定最负 s_prim）；T2c 逐神经元符号一致性。

### 核心结果（重复三遍）
**① B1 不符号特异**：B1 保护性质量份额 **0.4383** vs 随机 500 null 中位 **0.4475**（diff −0.0067，n_pos 4 / n_neg 7，p 0.89 / 0.27）——3025 的功能局域化**不伴随**几何符号签名在带内集中。**② B1 是稳定群体**：跨 tag 成对 Jaccard **0.261** vs 随机 500 **0.0256**（10.2×）——高 |s| 边缘带神经元跨 tag 重复入选，**rank 特异而非符号特异**。**③ 身份三重分离**：vs 3019 抑制 top-32 同 tag Jaccard 0.0057（121 对最大 0.0133，近随机水平）、vs 3022 联盟构造性为 0（断言）——B1 是独立于两个已登记回路的第三群体。**④ 逐神经元符号翻转**：B1 内仅 **2.4%%** 神经元在 ≥10/11 tag 保护、**61.8%%** 在 ≤1/11 tag 保护——保护性符号是**群体级、tag 依赖**的，无稳定逐神经元符号载体（联盟同向一致性也只有 18.75%%）。

### 机制结论
**功能局域 ≠ 几何符号身份**：L3 推拉重平衡（3025）由稳定的高 |s| 群体承载，但其保护方向弥散且随 tag 翻转。第六次独立收敛：**无逐神经元符号载体，保护性=群体级性质**。三级架构的解剖学补充：种子（group7 头）→ 正性中继联盟（32，e4 对齐 0.854）→ 稳定高 |s| 群体（B1，功能保护但符号弥散）→ 负性均衡场（L8-20 分布式）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3026/omega_p2t_b1_edge_band_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3027 = A（主选）**联盟基线功能与下游消费头定位**——3023 已测联盟承载 L3 MLP 输出范数 55.5%%，刻画其基线输出被谁消费（下游头对联盟输出的敏感性分账，衔接 3015/3016）；B 放大式干预（联盟变化加倍——剂量对称性）；C L31 次峰定位；D 情景性检验（同词异位 K,V 相似度）。
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
if 'Phase 3026' not in prev:
    line = ('- Phase 3026 Omega-P2t: verdict '
            'b1_protective_null_like_qwen (run1 '
            'authoritative one-pass 149.2s no crash, '
            'anchors 30/30, a28/a30/a34 all bit-level '
            '0.0, a34 = RESTORE_B1 vs sealed 3025 '
            'npz js_b1); B1 fringe (rank 33-532, '
            '500 neurons) is rank-special but NOT '
            'sign-specialized: protective mass '
            'share 0.4383 vs random-500 null 0.4475 '
            '(diff -0.0067, p 0.27); cross-tag '
            'Jaccard 0.261 vs 0.0256 random (10.2x) '
            '= stable population; vs 3019 '
            'suppression top-32 <= 0.013 (near-'
            'random), vs coalition 0 by '
            'construction; per-neuron sign flips '
            'across tags (61.8 pct protective in '
            '<=1/11, only 2.4 pct in >=10/11) = '
            'protection is population-level, no '
            'per-neuron sign carriers (sixth '
            'convergence); functional localization '
            '!= geometric sign identity; next = '
            'coalition baseline consumers; ledger '
            '165/L14 133.\n')
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
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；链身份锚多点（js 序列、js(pb,p0)=0.0、RESTORE_COAL vs 3024、RESTORE_B1 vs 3025 npz，均 0.0）。

## 统计判据纪律
- 判据可达性先检：置换不变 null p≡1（3021）；Jaccard null 中位=0→以 p 值为主（3022）；**消融类预检 ablation-only 副作用 << 信号（毒性门，3023）**；大基线功能集合禁零消融→**基线恢复 patch to no-erase value（3024）**；**宽 patch bf16 噪声底线，只用于强信号位（3025）**；null 用后处理零开销抽取须写作期可达性预检（3026 share-null）；中位数不可加；margin n≳40；maxT；镜像 −dirs 必配。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+竞争重平衡（3024/3025 层内证实）→**功能局域 ≠ 几何符号身份：符号结构功能可见、几何不可见、逐神经元可翻转（3025/3026）**。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；args 空用 kwargs；单样本保 batch 维；真残差流=decoder-layer pre-hook；npz 存 dict→0-d 读回 .item()；hook 改输出用返回值+active 门；**权重列 no_grad 外 .detach()（3025）；bf16 hook 内配 bf16 列、fp32 分析配 .float()（3026 run 前修）**；hcap detach().clone() 区分 base/erase。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影（3026 复发：Edit 报成功未落盘）→Python 补丁 assert count==1；replace 未命中→先 Grep；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3026）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3026）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018 抵消主导；3019 抵消带=通用抑制场；3020 读出特异=注入特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 正性专属稀疏联盟（top32=82pct，vs 3019 Jaccard 0.0）；3023 零消融有毒；3024 基线恢复：联盟因果承重（collapse 0.657 p=0.0005）；3025 非联盟变化保护性（R_nc −0.0013 p=0.033；B1 |s| 边缘带承载）；3026 **B1 rank 特异但符号 null-like**（share 0.438 vs null 0.448；跨 tag Jaccard 0.261=10×；vs 3019 ≤0.013、vs 联盟 0；逐神经元符号翻转 61.8pct ≤1/11——保护=群体级，无逐神经元符号载体）。门控三级=种子→L3 正性中继联盟→负性均衡场+稳定高|s|群体推拉；核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性。

## 下一步
- max=3026，下一个 3027（A 主选 **联盟基线功能与下游消费头定位**——55.5% 范数被谁消费、下游头对联盟输出敏感性分账衔接 3015/3016；B 放大式干预剂量对称性；C L31 次峰定位；D 情景性检验）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
