# -*- coding: utf-8 -*-
"""Phase 3035 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3035'
     r'\omega_p32_fingerprint_competition_qwen')
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
assert verdict == 'fp_logistic_readout_qwen', verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == 'none'
t1 = res['T1_symmetry']
assert abs(t1['med_A_idx_logic']
           - 0.20187744590919185) < 1e-12
t2 = res['T2_inflection']
assert t2['n_match'] == 10 and t2['n_total'] == 11
assert t2['n_match_sham'] == 2
t3 = res['T3_competition']
assert t3['mono_count'] == 11
assert abs(t3['med_kappa'] - 0.44799394030826) < 1e-9
assert abs(t3['med_atten_early_vs_late']
           - 0.03731274829112813) < 1e-9
spec = res['T2b_specificity']
assert abs(spec['spec_ratio']
           - 13.18865323604452) < 1e-9
an = res['anchors']
assert an['a47_dup_base_bit'] == 0.0
assert an['a48_gate0_bit'] == 0.0
assert an['a49_dup_bit'] == 0.0
assert an['a49_ratio_err'] < 0.02
assert an['a51_top2_ok'] is True
assert an['a51_maxdiff'] < 0.15

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3035
           for m in led['measurements']):
    claim = (
        'Omega-P32 (plan v5 P32) - fingerprint '
        'competition anatomy: directed readout '
        'intervention along d_fp = (w_A - w_B)/||'
        'w_A - w_B|| (A,B = base top-2 tokens, '
        'lm_head rows) at decoder-layer input '
        'sites L8/L35, dose m in {-0.05..0.05} '
        'as fraction of ||h_last||; random-'
        'direction control (rank-100/101 token '
        'pair).  Anchors: a47 duplicate base '
        'chain logits bit-level 0.0; a48 gate-on '
        'm=0 vs gate-off bit-level 0.0; a49 '
        'duplicate +0.05 chain rs bit-level 0.0 '
        '+ injection ratio err 5.05e-4 (gate '
        '2e-2); a51 manual final-norm+lm_head '
        'recompute top-2 identity, max|dlogit| '
        '0.065 (gate 0.15, bf16 family).  '
        'PRIMARY T2 inflection: even part of '
        'dP(A) under (+m,-m) pairs matches the '
        'logistic prediction (sign + iff '
        'P0(A)<0.5) in 10/11 logic prompts '
        '(sham 2/3 chance; sole mismatch P11:3 '
        'P0=0.4436 near inflection) - the '
        'readout migration obeys the sigmoid '
        'second-derivative structure.  T1 '
        'symmetry: med A_idx 0.2019 - migration '
        'is odd-dominant with even part '
        'organized as predicted.  T3 '
        'specificity: spec_ratio 13.19 (med '
        'ratio top-2 vs random direction, min '
        'over m in {0.01,0.02}) - competition '
        'is fingerprint-specific, not any-'
        'perturbation.  Coupling: med kappa '
        '0.448 - dP(B) recovers only ~45 pct '
        'of dP(A): multi-fingerprint field, '
        'not a two-body duel.  Attenuation: L8 '
        'vs L35 med ratio 0.037 - mid-band '
        'damping field suppresses injected '
        'error ~27x (3018-3019 confirmation).  '
        'Monotonicity 11/11.  verdict '
        'fp_logistic_readout_qwen.  NEXT: '
        'fingerprint curvature map (H2 '
        'judgment), situational specificity, '
        'cross-model replication.')
    meas = {
        'meas_id': 'meas3035_omega_p32_fingerprint_'
                   'competition_qwen',
        'phase': 3035,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a47/a48/a49 bit-level 0.0; '
                   'a49 ratio 5.05e-4; a51 top2 '
                   'identity + 0.065; T2 10/11; '
                   'spec 13.19',
        'artifacts': {
            'result': 'phase3035/omega_p32_'
                      'fingerprint_competition_'
                      'qwen/result.json',
            'npz': 'phase3035/omega_p32_'
                   'fingerprint_competition_'
                   'qwen/omega_p32_fingerprint_'
                   'competition_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run6 authoritative (24.3s); run4 '
                'registered fp_contaminated_void - '
                'preregistered sham gate mis-'
                'specified (measured intervention '
                'efficacy m*||h|| at L35 input, '
                'not contamination; contamination '
                'covered by a47/a48 bit '
                'determinism); run5 crashed pre-'
                'observation (a49 M_GRID.index '
                'stale); run5 corrections: '
                'perturbative dose grid '
                '{-0.05..0.05}, random-direction '
                'specificity control, T2 pair '
                '0.01, T3 at 0.05',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 174
    l14['connects'].append({
        'meas_id': 'meas3035_omega_p32_fingerprint_'
                   'competition_qwen',
        'phase': 3035,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P32: fingerprint '
                        'competition anatomy - '
                        'directed w_A-w_B readout '
                        'injection: even-curvature '
                        'sign follows logistic '
                        'inflection prediction '
                        '10/11 (sham 2/3); '
                        'fingerprint-specific '
                        '13.2x vs random direction; '
                        'odd-dominant migration med '
                        'A_idx 0.202; kappa 0.448 = '
                        'multi-fingerprint field '
                        'not two-body; L8 damping '
                        '27x; fp_logistic_readout_'
                        'qwen'})
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
if '## Phase 3035:' not in memo:
    sec = u'''## Phase 3035: Ω-P32 指纹竞争解剖——logistic 读出确认（inflection 10/11、指纹特异 13.2×、κ=0.45 多体场）[%(created)s]

**判决：`fp_logistic_readout_qwen`**（run6 权威 24.3s；run4 `fp_contaminated_void` 如实登记=预注册 sham 门误设（测的是干预效力 m·‖h‖ 而非污染，污染由 a47/a48 位级确定性覆盖），run5 观测前崩溃（a49 残留旧剂量 0.1），run6 修正门+加对照；锚全过）

### 设计（定向读出干预 + 随机方向对照）
A,B = 基线 step-2 softmax top-2 token；d_fp = (w_A−w_B)/‖·‖（lm_head 行，float32）；在 decoder-layer 输入位 **L8（中带场）/L35（读出前）** 注入 h_last += m·‖h_last‖·d_fp，m∈{−0.05,−0.02,−0.01,0,±0.01,±0.02,±0.05}；pre-hook 返回 (new_args,new_kwargs) 且**注册先于捕获 hook**（捕获看到注入值，a49 位级验证）；对照臂 d_rand = rank-100/101 token 对的 ŵ 差方向。

### 核心结果（重复三遍）
**① logistic 曲率符号 10/11**：even 部分（(ΔP(+m)+ΔP(−m))/2，m=0.01）符号符合 logistic 二阶导预测（P0(A)<0.5 凸 / >0.5 凹）**10/11**（sham 2/3=机会；唯一失配 P11:3 P0=0.4436 贴近拐点 |f''|→0）——**读出迁移服从 sigmoid 形状：3038-3030 的"凸性"不是普适属性而是 sigmoid 左支属性，操作点过 0.5 翻转为凹**。**② 指纹特异 13.2×**：spec_ratio=13.19（m=0.01）/16.99（m=0.02）——top-2 指纹方向效应是随机方向对照的 13–17 倍，竞争是**指纹特异的**而非任意扰动（回应 3027 审计链"读出集中须对照任意扰动 null"）。**③ 多体场而非二体决斗**：med κ=0.448——ΔP(B) 只回收 ΔP(A) 的 ~45%%，其余质量流向词表其他 token；单调 11/11；med A_idx=0.2019（迁移奇分量主导）。**④ 中带阻尼场再确认**：L8/L35 效应比 med=0.037（~27×衰减）——注入误差在上游被 3018-3019 抑制场主动抵消，读出竞争只在近读出位有效表达。

### 机制解读
附件"指纹竞争"假设的**可证伪核心被证实并精化**：softmax 读出确实是指纹间的 sigmoid 竞争（曲率符号随操作点翻转、方向特异 13×），但竞争发生在**多指纹场**中（κ<1）且上游被阻尼场屏蔽。统一机制链更新：**种子→L3 联盟中继→阻尼场均衡→logistic 读出（操作点依赖曲率）**。

### 缺陷与修正登记（如实）
run4 判决 fp_contaminated_void：预注册 sham 门 θ=0.02 设在干预响应幅度上=测量效力而非污染（sham dPA(0.05) 达 0.48），按 3030 a38 先例登记后修正；run5 a49 块残留 M_GRID.index(0.1) 观测前崩溃；run6 权威。**教训：null 门必须设在"不接收干预的量"或"错配方向"上，不能设在"接收干预的量"上。**

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3035/omega_p32_fingerprint_competition_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3036 菜单——A（主选）**指纹曲率地图**（逐指纹 ±δ 二阶差分曲率：裁决附件"高曲率指纹可逆袭" vs 3030"各层 lens 均匀放大"；副产 logic vs 随机词 ŵ 夹角补"正交性"）；B 情景性检验（同词异位 K,V 相似度）；C 深峰头簇公共核心解剖；D 跨模型复刻（β<1、γ 增长、logistic 读出）。
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

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3035' not in prev:
    line = ('- Phase 3035 Omega-P32: verdict '
            'fp_logistic_readout_qwen (run6 '
            '24.3s; run4 fp_contaminated_void '
            'registered - sham gate mis-specified '
            '(efficacy not contamination); run5 '
            'pre-observation crash; anchors '
            'a47/a48/a49 bit-level 0.0, a49 ratio '
            '5.05e-4, a51 top2 identity + 0.065); '
            'directed w_A-w_B injection: logistic '
            'inflection sign 10/11 (sham 2/3), '
            'fingerprint-specific 13.2x vs random '
            'direction, kappa 0.448 = multi-'
            'fingerprint field, L8 damping 27x; '
            'ledger 174/L14 142.\n')
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
4. MEMO 占位符一律 %(key)s 风格；MEMO 文本内裸百分号写 %%（3033 教训）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08；bit 级仅限同文件链/上游全精度；跨相位 npz 锚均 0.0；剂量两端点锚死。
- GPU 复刻相位：verbatim 拷贝+外科补丁（assert count==1），全套旧锚新 run 复过（3034）。
- 干预相位新锚族（3035）：a47 重复基链位级 0.0；a48 门开 m=0 vs 门关 0.0；a49 重复臂 rs 0.0+注入比率门；a51 手工 norm+lm_head 重算 top2 恒等+0.15 门。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）。
- **null 门不得设在接收干预的量上**（3035 run4 教训：sham 门设在干预响应=测效力；应设在错配方向/不接收量上）。
- 比值/凸超额报 (log 基线, gamma) 二元组；集合统计用精确超几何 null；定向干预剂量报 m·‖h‖ 绝对量。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向（3035 spec 13.2×）→曲率符号须报操作点 P0（logistic 拐点）→消融差分=直接+重平衡→读出集中对照任意扰动 null。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；hook 改输出用返回值+active 门；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；**step-2 前向污染 KV cache→每链重新 prefill**；**pre-hook with_kwargs 改输入须返回 (new_args,new_kwargs) 且注册先于捕获 hook**；lm_head 行数=config.vocab_size 非 tok.vocab_size。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；`cmd &` 孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3035）
Ω-P2（3011-3035）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3018-3019 抵消=通用抑制场；3020 注入特异 944×；3021 MLP 中继 69pct；3022 稀疏联盟 top32=82pct；3023 零消融有毒；3024 联盟承重 0.657；3027 消费=通用读出；3028 剂量凸增长；3029-3030 凸=读出本征；3031/3033 异质性=tag 特异+比值伪影主因（β 0.649 CI<1）；3032 深峰=头集中 89pct；3034 头集合=部分共享通路（J 0.188 p 0.0034，按 ldp 层聚类）；3035 **指纹竞争=logistic 读出（定向 wA−wB 注入：inflection 符号 10/11 随操作点翻转、指纹特异 13.2×、κ 0.448 多体场、L8 阻尼 27×）**。核心：重要性=关系属性；统一链=种子→L3 联盟中继→阻尼场均衡→logistic 读出。

## 下一步
- max=3035，下一个 3036（A 主选 **指纹曲率地图**——逐指纹二阶差分曲率裁决附件"高曲率逆袭" vs 3030 均匀放大，副产 ŵ 夹角；B 情景性检验；C 深峰头簇公共核心；D 跨模型复刻；附件审计 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
