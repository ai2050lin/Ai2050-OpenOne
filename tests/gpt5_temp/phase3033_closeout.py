# -*- coding: utf-8 -*-
"""Phase 3033 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3033'
     r'\omega_p30_logscale_heterogeneity_qwen')
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
assert verdict == 'hetero_scale_artifact_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2a = res['T2a']
assert t2a['rho_gamma12_logje'] == -0.527273
assert t2a['p_perm'] == 0.101034
assert t2a['med_gamma12'] == 1.915738
assert t2a['med_gamma05'] == 0.739879
t2b = res['T2b']
assert t2b['beta'] == 0.648598
assert t2b['ci_lo'] == 0.303767
assert t2b['ci_hi'] == 0.965277
assert t2b['r2_loglog'] == 0.753429
assert t2b['mad_log_ratio'] == 0.50893
assert t2b['mad_rel_dev'] == 1.166178
assert t2b['med_log_ratio'] == 1.327888
t2c = res['T2c']
assert t2c['family'] == ['erase_mag',
                         'restore_asym',
                         'coal_share', 'coal_jac',
                         'mag2', 'recruit_rho',
                         'growth_g', 'band_mid',
                         'readout_lstar',
                         'log_je']
assert t2c['dropped'] == ['pos_frac:zero_var']
assert t2c['eligible'] == []
assert t2c['winner'] is None
assert t2c['rho_obs']['readout_lstar'] == -0.62244
assert t2c['rho_obs']['band_mid'] == 0.615036
assert t2c['rho_obs']['log_je'] == -0.527273
assert t2c['p_maxT']['readout_lstar'] == 0.263364
assert t2c['p_maxT']['band_mid'] == 0.280254
assert t2c['p_maxT']['log_je'] == 0.479243
an = res['anchors']
assert an['a0_source_integrity'] is True
assert an['a1_reldev_identity'] == 0.0
assert an['a2_dose_js2_cross'] == 0.0
assert an['a3_erase_cross'] == 0.0
assert an['a4_alpha0_cross'] == 0.0
assert an['a5_ratio2_vs_3031'] == 0.0
assert an['a6_ratio_reldev_identity'] < 1e-12
assert an['a7_conc_med_diff'] < 5e-5
assert an['a8_tags_identity'] is True
assert an['a9_med_reldev_diff'] < 5e-5

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3033
           for m in led['measurements']):
    claim = (
        'Omega-P30 (plan v5 P30) - log-scale '
        'reparameterization of the 3028 dose-'
        'response heterogeneity: REANALYSIS phase '
        '(CPU only, no model run) over sealed '
        '3028/3029/3022/3020/3031 npz.  T2a '
        'PRIMARY Spearman(gamma_12, log je) '
        '-0.5273 p .101 (n=11, 200k perm, seed '
        '3033; ns); T2b OLS log js2 ~ log je '
        'beta 0.6486 bootstrap CI [0.3038, '
        '0.9653] (200k, seed 30331) - CI '
        'EXCLUDES 1 on the low side, R2 0.753; '
        'T2c family re-test in scale-free space '
        '(outcome log js_ratio2, 10 candidates '
        'incl log_je, maxT 0.05, 200k seed '
        '30332): 0/10 survive.  Anchors 10/10: '
        'a0 five-source seal integrity; a1-a5 '
        'bit-level 0.0 (rel_dev identity, 3028-'
        'vs-3029 js2/js_erase/alpha0, ratio2 vs '
        '3031); a6 ratio identity 1.78e-15 '
        '(<1e-12 float rounding, not bit-'
        'exact); a7 3022 conc recompute 2.09e-'
        '05; a8 tags identity; a9 3.47e-07.  '
        'Verdict hetero_scale_artifact_qwen '
        '(frozen tree: ci_hi < 1 branch).  '
        'RESULTS: (i) the convexity exponent '
        'gamma_12 = log2(js2/js_erase) is '
        'SYSTEMATICALLY SMALLER for large-'
        'baseline tags - log-log slope 0.65 < 1 '
        'means doubling the erase baseline '
        'yields only 2^0.65=1.57x the alpha=2 '
        'response, so a substantial part of the '
        '3028 rel_dev spread (-0.697..+3.874) '
        'is a ratio-parameterization artifact '
        'of small baselines; (ii) yet scale is '
        'NOT the whole story - R2 0.75 leaves '
        '25pct residual shape variance, '
        'median-subtracted MAD shrinks 1.166 -> '
        '0.509 in log space but does not '
        'vanish, and the PRIMARY Spearman is '
        'ns at n=11; (iii) within-tag convexity '
        'in alpha is genuine and universal: '
        'gamma_05 med 0.740 < gamma_12 med '
        '1.916 (exponent GROWS with dose in '
        'every tag), consistent with 3028 '
        'superlinear; (iv) no preregistered '
        'feature survives maxT in scale-free '
        'space either - heterogeneity source '
        'remains unidentified beyond scale.  '
        'NEXT: deep-peak head-set identity, '
        'fingerprint competition anatomy '
        '(directional w_A - w_B intervention), '
        'or situational specificity.')
    meas = {
        'meas_id': 'meas3033_omega_p30_logscale_'
                   'heterogeneity_qwen',
        'phase': 3033,
        'claim': claim,
        'verdict': verdict,
        'anchors': '10/10 (a0 integrity x5; '
                   'a1-a5 bit-level 0.0; a6 '
                   '1.78e-15; a7 2.09e-05; a8 '
                   'identity; a9 3.47e-07)',
        'artifacts': {
            'result': 'phase3033/omega_p30_'
                      'logscale_heterogeneity_'
                      'qwen/result.json',
            'npz': 'phase3033/omega_p30_'
                   'logscale_heterogeneity_'
                   'qwen/omega_p30_logscale_'
                   'heterogeneity_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative (5.2s; run1 '
                'broadcast bug in T2a permutation '
                'null, fixed, no prereg change); '
                'verdict via preregistered '
                'ci_hi<1 branch; primary Spearman '
                'ns (n=11) - scale-artifact '
                'conclusion rests on the beta CI; '
                'within-tag alpha convexity '
                'universal (gamma05 0.74 < gamma12 '
                '1.92).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 172
    l14['connects'].append({
        'meas_id': 'meas3033_omega_p30_logscale_'
                   'heterogeneity_qwen',
        'phase': 3033,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P30: log-log '
                        'reparameterization - beta '
                        '0.649 (CI [0.304, 0.965], '
                        'excludes 1) R2 0.75: '
                        'convexity exponent shrinks '
                        'with erase baseline => part '
                        'of 3028 rel_dev heterogeneity '
                        'is ratio-scale artifact; 25pct '
                        'residual shape variance '
                        'remains (MAD 1.166->0.509); '
                        'primary Spearman(gamma12, '
                        'log je) -0.527 ns; within-tag '
                        'alpha convexity universal '
                        '(gamma05 0.74 < gamma12 '
                        '1.92); T2c 0/10 candidates '
                        'survive maxT in scale-free '
                        'space; anchors 10/10'})
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
if '## Phase 3033:' not in memo:
    sec = u'''## Phase 3033: Ω-P30 rel_dev 对数尺度重参数化——凸指数随基线收缩（β<1），异质性部分为比值伪影、25%% 形状残差留存 [%(created)s]

**判决：`hetero_scale_artifact_qwen`**（run2 权威 5.2s，run1 置换 null 广播 bug 修复后无预注册变更；锚 **10/10**：a0 五源封印完整性 + a1–a5 **位级 0.0**（rel_dev 恒等、3028-vs-3029 js2/js_erase/α0、ratio2-vs-3031）+ a6 比值恒等 1.78e-15（浮点舍入，<1e-12 门）+ a7 3022 conc 重算 2.09e-05 + a8 tags 恒等 + a9 3.47e-07）

### 设计（纯重分析相位，无模型运行）
跟进 3031 的尺度依赖探索性发现（erase_mag −0.527，P0/P7 大基线=仅有的负 rel_dev tag）。产出改为**无尺度量** log js_ratio2 = log js2 − log je（gamma_12 = log2(js2/js_erase) 是存储 rel_dev 的精确单调变换），预测量 log je。三项统计：T2a Spearman(gamma_12, log je)（200k 置换 seed 3033）；T2b log-log OLS β + pairs bootstrap CI（200k，seed 30331）；T2c 无尺度空间家族重检（10 候选含 log_je，maxT 0.05，seed 30332）。

### 核心结果（重复三遍）
**① 凸指数随基线系统性收缩**：β = **0.6486**，bootstrap CI **[0.3038, 0.9653] 上界 <1**（触发预注册 scale_artifact 分支），R² = **0.753**——擦除基线翻倍只换来 2^0.65≈1.57× 的 α=2 响应：**3028 的 rel_dev 散布（−0.697..+3.874）相当一部分是小基线 tag 的比值膨胀伪影**。**② 但尺度不是全部**：主检验 Spearman(γ₁₂, log je) = −0.527、p=.101（n=11 不显著）；R² 0.75 留 25%% 形状残差；中位去势 MAD 从 rel_dev 1.166 → log 空间 0.509，收缩但不消失。**③ tag 内 α 凸性普适且真实**：γ(0.5→1) med **0.740** < γ(1→2) med **1.916**——每个 tag 的指数都随剂量增长，3028 superlinear 的 tag 内结论不受伪影影响。**④ 无尺度空间候选家族依旧 0/10 存活**（最强 readout_lstar −0.622 p .263、band_mid +0.615 p .280、log_je −0.527 p .479）——异质性来源在尺度校正后仍未识别。

### 机制解读
3031/3033 合并结论：rel_dev 的 tag 间异质性 = **比值参数化伪影（主）+ 未知上下文形状属性（残差 25%%）**。报告凸超额时必须同时给出 (log je, gamma) 二元组而非单一 rel_dev。γ 随 α 增长的普适性把 3028 的"凸增长"精化为：**响应指数随剂量单调上升的幂律族**，而非固定指数——与 3030 的"凸性=读出映射本征"相容（读出非线性在任意深度都对大扰动放大更多）。

### 缺陷与修正登记（如实）
run1 T2a 置换 null 广播维度 bug（ry_a[perms]×rx_a[:,None]），run2 修复（rx_a[None,:]·ry_a[perms] sum axis=1），预注册未变。a6 恒等差 1.78e-15 为浮点舍入（ratio2 vs 2(1+rel_dev) 非位级精确），门 1e-12 通过、如实登记非 0.0。n=11 全程探索性边界维持。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3033/omega_p30_logscale_heterogeneity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3034 菜单——A（主选）**深峰头集合身份**（3032 top-8 头跨 tag Jaccard / GQA 归属：头载体是"通路属性"还是第三种上下文属性）；B **指纹竞争解剖**（对 top-2 竞争指纹做定向 ±δ·(w_A−w_B) 读出干预，直接检验概率迁移与竞争结构）；C 情景性检验（同词异位 K,V 相似度）；D 联盟读出解码。附件"指纹竞争"理论审计与综合方案见 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md。
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
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3033' not in prev:
    line = ('- Phase 3033 Omega-P30: verdict '
            'hetero_scale_artifact_qwen (reanalysis '
            'run2 5.2s; run1 broadcast bug fixed, no '
            'prereg change; anchors 10/10 incl a1-a5 '
            'bit-level 0.0); log-log beta 0.649 CI '
            '[0.304,0.965] excludes 1, R2 0.75: part '
            'of 3028 rel_dev heterogeneity is ratio-'
            'scale artifact; 25pct shape residual; '
            'primary Spearman -0.527 ns (n=11); '
            'within-tag alpha convexity universal '
            '(gamma05 0.74 < gamma12 1.92); T2c 0/10 '
            'survive maxT; ledger 172/L14 140.\n')
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
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；跨相位 npz 锚 js2/js_erase/α0/ratio2 均 0.0；剂量两端点锚死 α=1=擦除/α=0=恢复。
- 重分析相位锚=对源 npz/result 记账值重算恒等（门 5e-5）+ 源 seal.json 完整性（a0）；数组对齐按 key/索引数组（traj=dict 序）。
- 置换 null 广播：null=(rx[None,:]·ry[perms]).sum(axis=1)（3033 run1 教训）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）；镜像 −dirs 必配。
- **相对量/比值有尺度依赖（3031/3033）：报凸超额必须同时报 (log je, gamma) 二元组；log-log β<1=比值伪影信号；tag 内指数随剂量增长（γ05 0.74<γ12 1.92）是普适真信号**。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位（读出映射本征）→异质性归因先候选家族 maxT（0 存活=tag 特异）+ **log-log 尺度重参数化（3033：β 0.649 CI 上界<1=伪影主因，25%% 形状残差）**→消融差分=直接+竞争重平衡→功能局域≠几何符号身份→读出集中须对照任意扰动 null。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；真残差流=decoder-layer pre-hook；npz dict→0-d 读回 .item()；hook 改输出用返回值+active 门；权重列 .detach()；bf16 hook 配 bf16 列；step-2 单 token 取 [0,-1]；clear_cap 每链清→链后立即提取；位级锚 α 分支逐字复刻原表达式；**step-2 前向污染 KV cache→run_chain 每链重新 prefill（3032）**；hook 捕获门与 patch 门必须独立（state_hcap 期间关否则中和 α patch）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail/行尾反斜杠被篡改；rm shim 损坏→Python os.remove；`cmd &` 孤儿进程随会话死（3032）→一律 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3033）
Ω-P2（3011-3033）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018-3019 抵消=通用抑制场；3020 读出特异=注入特异（944×）；3021 注入=MLP 中继 69pct；3022 正性稀疏联盟（top32=82pct）；3023 零消融有毒；3024 联盟因果承重（0.657）；3025-3026 非联盟保护/B1 null-like；3027 消费=通用读出结构；3028 剂量凸增长；3029 凸=读出本征；3030 凸=分布式读出（lens 放大 2.6-2.9×）；3031 异质性 tag 特异（0/9）；3032 深峰=头集中（head8 89pct vs mlp32 17pct，载体地图：头=每深度读出载体，L3=唯一 MLP 中继深度）；3033 **异质性=比值伪影主因（β 0.649 CI[0.304,0.965]）+25%% 形状残差；tag 内 α 凸性普适（γ 随剂量长）**。核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性；连异质性也主要关系于尺度。

## 下一步
- max=3033，下一个 3034（A 主选 **深峰头集合身份**——3032 top-8 头跨 tag Jaccard/GQA 归属；B 指纹竞争解剖 ±δ·(wA−wB) 定向读出干预；C 情景性检验；D 联盟读出解码；附件理论审计见 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
