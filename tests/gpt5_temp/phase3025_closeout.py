# -*- coding: utf-8 -*-
"""Phase 3025 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3025'
     r'\omega_p2s_rebalance_decomp_qwen')
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
assert verdict == 'rebalance_noncoal_protective_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_content'] == 11
assert t2['med_js_erase'] == 0.003332
assert t2['med_js_noncoal'] == 0.004591
assert t2['med_js_coal'] == 0.00117
assert t2['R_nc_med'] == -0.001291
assert t2['R_coal_med'] == 0.001779
assert t2['R_all_3024_med'] == 0.001929
assert t2['additivity_resid_med'] == -0.000264
assert t2['n_neg'] == 9
assert t2['n_pos'] == 2
assert t2['p_binom_neg'] == 0.0327
assert t2['p_binom_pos'] == 0.9941
assert t2['a28_erase_chain_diff'] == 0.0
assert t2['a30_capture_self_diff'] == 0.0
assert t2['a32_coal_restore_diff'] == 0.0
t2b = res['T2b']
assert t2b['R_b1_med'] == -0.00109
assert t2b['R_b2_med'] == 2.6e-05
assert t2b['R_b3_med'] == -0.000155
assert t2b['sizes'] == {'coal': 32, 'b1': 500,
                        'b2': 1500, 'b3': 7696}
t2c = res['T2c']
assert t2c['cos_coal_e4_med'] == 0.8544
assert t2c['cos_noncoal_e4_med'] == 0.0496
assert t2c['protective_share_med'] == 0.4633
assert t2c['e4_norm_med'] == 3.6017
assert t2c['collapse_content'] == -26.1995
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a27_3022'] is True
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a29_3023'] is True
assert res['anchors']['a30_capture_self_diff'] == 0.0
assert res['anchors']['a31_3024'] is True
assert res['anchors']['a32_coal_restore_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3025
           for m in led['measurements']):
    claim = (
        'Omega-P2s (plan v5 P2) - localizes the '
        'non-coalition rebalancing discovered in '
        '3024 (RESTORE_ALL 0.511 < RESTORE_COAL '
        '0.657): per logic tag, ERASE / '
        'RESTORE_COAL / RESTORE_NONCOAL '
        '(complement 9696 neurons patched to no-'
        'erase values) / RESTORE_B1/B2/B3 (non-'
        'coalition ranked by 3022 |s|: next 500 / '
        'next 1500 / rest); R_X = js_erase - js_X '
        '(positive = concurrent, negative = '
        'protective); e4 projection sign '
        'accounting (e4 = res36[4] erase - base, '
        '3021 convention).  Verdict '
        'rebalance_noncoal_protective_qwen '
        '(frozen map: R_nc_med -0.001291 < 0 AND '
        'p_binom 0.0327 <= 0.05).  RESULTS: '
        '(i) the non-coalition erase-induced '
        'relay change is PROTECTIVE - reverting '
        'it RAISES the erase JS (med 0.003332 -> '
        '0.004591; R_nc negative in 9/11 tags, '
        'exact binomial p 0.0327) while the '
        'coalition change is concurrent (R_coal '
        '+0.001779): inside L3 the two changes '
        'PULL AGAINST EACH OTHER, direct '
        'functional confirmation of the 3024 '
        'RESTORE_ALL < RESTORE_COAL anomaly; '
        '(ii) the protective change lives in '
        'the HIGH-|s| FRINGE around the '
        'coalition: B1 (rank 33-532) R med '
        '-0.00109 carries it, B2 (533-2032) ~0, '
        'B3 (7696) -0.000155 - the rebalancing '
        'is localized, not diffuse; (iii) e4 '
        'geometry: coalition change strongly '
        'e4-aligned (cos 0.854 - the relay '
        'direction), non-coalition NET '
        'near-orthogonal (cos 0.050) with '
        'protective mass share 0.463 - '
        'concurrent and protective components '
        'cancel in projection, so the sign '
        'structure is invisible in the mean '
        'vector and only the functional patch '
        'reveals it; (iv) near-additivity: '
        'per-tag residual median R_coal+R_nc-'
        'R_all_3024 = -0.000264 (small vs R '
        'terms; per-tag medians do not compose '
        '- registered as descriptive only); '
        '(v) triple chain identity: a28 ERASE '
        'vs 3022 js bit-level 0.0, a30 capture '
        'self-consistency 0.0, a32 RESTORE_COAL '
        'vs sealed 3024 npz js bit-level 0.0.  '
        'REGISTERED CAVEAT: content-side '
        'RESTORE_NONCOAL collapse (-26.2) is at '
        'the bf16 patch noise floor - patching '
        '9696 neurons (subtract + re-add large '
        'bf16 quantities) injects rounding '
        'noise comparable to content-side JS '
        'effects (content er 1.2e-4 -> nc '
        '2.2e-3); the 32-neuron patch of 3024 '
        'was clean (-0.027), logic-side '
        'conclusions (effects 10-40x above '
        'noise) unaffected.  NEXT: characterize '
        'the B1 protective fringe (identity vs '
        '3019 suppression field, per-neuron e4 '
        'sign split) or coalition baseline '
        'consumers.')
    meas = {
        'meas_id': 'meas3025_omega_p2s_rebalance_'
                   'decomp_qwen',
        'phase': 3025,
        'claim': claim,
        'verdict': verdict,
        'anchors': '32/32 (a0-a27 as 3024; a28 erase '
                   'vs 3022 bit-level 0.0; a29 3023; '
                   'a30 capture self 0.0; a31 3024 '
                   'integrity; a32 RESTORE_COAL vs '
                   '3024 npz bit-level 0.0)',
        'artifacts': {
            'result': 'phase3025/omega_p2s_'
                      'rebalance_decomp_qwen/'
                      'result.json',
            'npz': 'phase3025/omega_p2s_'
                   'rebalance_decomp_qwen/'
                   'omega_p2s_rebalance_decomp_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'runs 1-2 crashed (bf16/fp32 dtype '
                'mismatch; requires-grad slice '
                'outside no_grad), run3 '
                'authoritative; non-coalition '
                'relay change is PROTECTIVE and '
                'lives in the high-|s| fringe '
                '(B1 rank 33-532); content-side '
                'wide patch at bf16 noise floor '
                '(registered caveat).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 164
    l14['connects'].append({
        'meas_id': 'meas3025_omega_p2s_rebalance_'
                   'decomp_qwen',
        'phase': 3025,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2s: non-coalition '
                        'erase-induced L3 relay '
                        'change is PROTECTIVE '
                        '(reverting raises erase '
                        'JS, R_nc med -0.0013, 9/11 '
                        'tags, p 0.033) and '
                        'localized in the high-|s| '
                        'fringe (B1 rank 33-532 '
                        'carries it); coalition '
                        'change e4-aligned (cos '
                        '0.854), non-coalition net '
                        'orthogonal (0.05) - '
                        'in-layer push-pull '
                        'confirmed functionally'})
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
if '## Phase 3025:' not in memo:
    sec = u'''## Phase 3025: Ω-P2s 非联盟重平衡分解——保护性变化定位在高 |s| 边缘带 [%(created)s]

**判决：`rebalance_noncoal_protective_qwen`**（run3 权威，151.1s，锚 **32/32**：a28 ERASE vs 3022、a30 捕获自洽、**a32 RESTORE_COAL vs 3024 npz 三重位级 0.0**，a31 3024 完整性；correction_note 空。run1-2 崩溃：bf16/fp32 dtype 失配、no_grad 外切参数切片 requires-grad）

### 设计（3024 机器 verbatim + 补集恢复与带分解）
每 logic tag 六臂：ERASE（捕 h_er）/ RESTORE_COAL（链身份锚）/ **RESTORE_NONCOAL（补集 9696 神经元 patch 到无擦除值）** / RESTORE_B1/B2/B3（非联盟按 3022 |s| 排名：33-532 / 533-2032 / 其余 7696）。R_X = js_erase − js_X（正=与联盟同向推进，负=保护性/重平衡）；e4 投影符号分账（e4 = res36[4] 擦除−基线，3021 口径）。

### 核心结果（重复三遍）
**① 非联盟擦除诱导变化是保护性的**：回退它使擦除 JS **反升**（med 0.003332 → 0.004591；R_nc med **−0.001291**，9/11 tag 为负，精确二项 p=0.0327），而联盟变化同向推进（R_coal +0.001779）——**L3 层内两种擦除诱导变化互相拉扯**，3024 的 RESTORE_ALL < RESTORE_COAL 反常获得直接功能确认。**② 保护性变化定位在高 |s| 边缘带**：B1（|s| 排名 33-532）R med **−0.00109** 独自承载，B2 ≈ 0、B3 −0.000155——重平衡是**局域化**的（围绕联盟的边缘），不是弥散的。**③ e4 几何**：联盟变化强 e4 对齐（cos **0.854**=中继方向），非联盟**净投影近正交**（cos 0.050，保护性质量占比 0.463）——同向与保护分量在均值向量里相互抵消，**符号结构只在功能干预下显形**，均值几何不可见。**④ 近可加性**：每 tag 残差中位 −0.000264（R 量级的小项；中位数不可加，仅描述性登记）。

### 测量缺陷（如实登记）
**content 侧 RESTORE_NONCOAL collapse −26.2 处于 bf16 patch 噪声底线**：patch 9696 神经元（减去再加回大 bf16 量）注入的舍入噪声与 content 位 JS 效应同量级（content er 1.2e-4 → nc 2.2e-3）；3024 的 32 神经元窄 patch 干净（−0.027）。logic 侧结论（效应高出噪声 10-40×）不受影响；**宽 patch 只能用于强信号位**。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3025/omega_p2s_rebalance_decomp_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3026 = A（主选）**B1 保护性边缘带刻画**——这 500 个神经元的身份（与 3019 抑制场 top-32 的 Jaccard、逐神经元 e4 符号分裂：保护亚群 vs 同向亚群）；B 联盟基线功能刻画/下游消费头定位；C 放大式干预（把联盟变化加倍而非回退——剂量对称性）；D L31 次峰定位。
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
if 'Phase 3025' not in prev:
    line = ('- Phase 3025 Omega-P2s: verdict '
            'rebalance_noncoal_protective_qwen '
            '(runs 1-2 crashed: dtype mismatch, '
            'requires-grad slice; run3 '
            'authoritative, anchors 32/32, a28/a30/'
            'a32 all bit-level 0.0); '
            'RESTORE_NONCOAL (complement 9696 '
            'neurons patched to no-erase values): '
            'reverting the non-coalition erase-'
            'induced relay change RAISES erase JS '
            '(med 0.003332 -> 0.004591, R_nc med '
            '-0.0013, 9/11 tags, p 0.0327) = '
            'PROTECTIVE/rebalancing, functional '
            'confirmation of the 3024 RESTORE_ALL '
            '< RESTORE_COAL anomaly; localized in '
            'B1 high-|s| fringe (rank 33-532, R '
            '-0.00109; B2 ~0, B3 -0.00016); '
            'coalition change e4-aligned cos '
            '0.854, non-coalition net orthogonal '
            '0.050 (protective share 0.463) - '
            'sign structure invisible in mean '
            'geometry; CAVEAT: content-side wide '
            'patch at bf16 noise floor (collapse '
            '-26.2 spurious, 3024 32-neuron patch '
            'clean); next = B1 fringe '
            'characterization; ledger 164/L14 '
            '132.\n')
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
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；链身份锚多点（‖e4‖=3.6017、js 序列、js(pb,p0)=0.0、RESTORE_COAL vs 3024 npz=0.0，均 0.0）。

## 统计判据纪律
- 判据可达性先检：置换不变 null p≡1（3021）；Jaccard null 中位=0→以 p 值为主（3022）；**消融类预检 ablation-only 副作用 << 信号（毒性门，3023 第三次复发）**；大基线功能集合禁零消融→**基线恢复 patch to no-erase value（3024 闭环）**；**宽 patch（数百+神经元）有 bf16 舍入噪声底线，只用于强信号位；content 等近零信号位会被污染（3025 collapse −26.2 假象）**；中位数不可加，加性检验用逐 tag 残差；margin n≳40；maxT；镜像 −dirs 必配。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+竞争重平衡（3024/3025 层内证实：非联盟变化保护性、B1 边缘带承载、净投影正交但功能反向——**均值几何看不见符号结构，必须功能干预**）。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；args 空用 kwargs；单样本保 batch 维；真残差流=decoder-layer pre-hook；恒等门分母与噪声同尺度；npz 存 dict→0-d 对象数组读回 .item()；hook 修改输出用返回值，active 门 prefill 期关；**权重列在 no_grad 外用必须 .detach()（3025 run2）；fp32 激活配 .float() 权重列（3025 run1）**；恢复捕获 h_base/h_er 用 detach().clone() 存 GPU bf16，hcap 标志区分 'base'/'erase'。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁；replace 未命中→先 Grep；改后必编译检查。
- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接。
- numpy 标量入 json 转 int()/float()；GQA：KV 缓存 8 头。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3025）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/GLM4 86 vs qwen 63pct/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3025）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018 抵消主导（2.51 nats）；3019 抵消带=通用抑制场；3020 读出特异=注入特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 中继=正性专属稀疏联盟（top32=82pct，vs 3019 Jaccard 0.0 反号）；3023 零消融有毒；3024 **基线恢复：联盟因果承重成立**（collapse 0.657，11/11 p=0.0005；RESTORE_ALL<COAL=层内重平衡）；3025 **非联盟变化保护性**（回退反升 JS，R_nc −0.0013 p=0.033；B1 |s| 边缘带 33-532 承载；联盟 cos_e4 0.854 vs 非联盟净 0.05——符号结构功能可见、几何不可见）。门控三级=种子→L3 正性中继联盟→负性均衡场；L3 内部=推进-保护推拉回路；核心：null 重编码全层分布式涌现；头级重要性=关系属性。

## 下一步
- max=3025，下一个 3026（A 主选 **B1 保护性边缘带刻画**——500 神经元身份 vs 3019 抑制场 Jaccard、逐神经元 e4 符号分裂=保护/同向亚群；B 联盟基线功能/下游消费头；C 放大式干预剂量对称性；D L31 次峰）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
