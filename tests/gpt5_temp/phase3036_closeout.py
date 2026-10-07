# -*- coding: utf-8 -*-
"""Phase 3036 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3036'
     r'\omega_p33_fingerprint_curvature_map_qwen')
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
assert verdict == 'fp_curvature_logistic_uniform_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == 'none'
t1 = res['T1_sign']
assert t1['n_eligible'] == 87 and t1['n_match'] == 86
assert abs(t1['frac_match']
           - 0.9885057471264368) < 1e-12
assert t1['sham_match'] == 21 and t1['sham_eligible'] == 24
t2 = res['T2_h2']
assert abs(t2['med_max_z']
           - 0.5643270428024156) < 1e-12
assert abs(t2['med_z'] - 0.1271283759298929) < 1e-12
assert abs(t2['spec_curv_ratio']
           - 0.8708228008426072) < 1e-12
assert t2['h2_supported'] is False
t3 = res['T3_ortho']
assert abs(t3['med_abs_cos_top8']
           - 0.12279830127954483) < 1e-12
assert abs(t3['med_abs_cos_random']
           - 0.05067892372608185) < 1e-12
an = res['anchors']
assert an['a52_dup_base_bit'] == 0.0
assert an['a53_gate0_bit'] == 0.0
assert an['a54_dup_bit'] == 0.0
assert an['a54_ratio_err'] < 0.02
assert an['a55_wu_row_bit'] == 0.0
assert an['a56_kappa_recompute_bit'] == 0.0
assert an['a57_source_seals'] is True
assert an['a51_top2_ok'] is True
assert an['a51_maxdiff'] < 0.15

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3036
           for m in led['measurements']):
    claim = (
        'Omega-P33 (plan v5 P33) - fingerprint '
        'curvature map (H2 decision): per prompt '
        'top-8 tokens probed along their OWN '
        'fingerprint direction d_t = W_U[t] row '
        'normalized at L35 decoder-layer input, m '
        'in {+-0.01, +-0.02} fraction of ||h||; '
        'second difference kappa_t = [P(+d)+P(-d)-'
        '2P(0)]/d^2 vs logistic operating-point '
        'prediction kappa_pred = b^2 (1-2P)/(P(1-P)) '
        '(b = empirical odd slope, no free curvature '
        'parameter); random-direction null (10 '
        'Gaussian unit vectors, seed 9036) sets the '
        'noise floor theta.  Anchors: a52/a53/a54-'
        'dup/a55/a56 bit-level 0.0; a54 injection '
        'ratio err 4.06e-3 (gate 2e-2); a57 source '
        'seals 3030/3032/3035 verified; a51 manual '
        'recompute top-2 identity max|dlogit| '
        '0.065.  PRIMARY T2 H2 REJECTED: med over '
        '11 logic prompts of per-prompt max_t z_t = '
        '0.564 << gate 3.0 (all logic rows <= 1.37; '
        'med z 0.127) - no fingerprint-specific '
        'curvature beyond the operating point; the '
        'attachment claim that a high-curvature '
        'fingerprint can win from behind is '
        'falsified at the Qwen3-4B readout.  T1 '
        'sign consistency 86/87 eligible (frac '
        '0.9885; sham 21/24, sole sham outlier = '
        'saturated P=0.9915 ill-conditioned case).  '
        'Curvature specificity spec_curv 0.87: '
        'own-direction curvature magnitude equals '
        'random-direction floor - specificity lives '
        'in the competition DIFFERENCE direction '
        '(3035: 13.2x), not the own direction.  T3 '
        'orthogonality: med |cos| within top-8 '
        'fingerprints 0.123 vs random row pairs '
        '0.051 - low-overlap but not orthogonal.  '
        'verdict fp_curvature_logistic_uniform_qwen.'
        '  NEXT: situational specificity, deep-head '
        'core anatomy, cross-model replication.')
    meas = {
        'meas_id': 'meas3036_omega_p33_fingerprint_'
                   'curvature_map_qwen',
        'phase': 3036,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a52/a53/a54/a55/a56 bit-level '
                   '0.0; a54 ratio 4.06e-3; a57 '
                   'seals ok; a51 0.065; H2 '
                   'rejected med_max_z 0.564<3; '
                   'sign 86/87',
        'artifacts': {
            'result': 'phase3036/omega_p33_'
                      'fingerprint_curvature_'
                      'map_qwen/result.json',
            'npz': 'phase3036/omega_p33_'
                   'fingerprint_curvature_'
                   'map_qwen/omega_p33_'
                   'fingerprint_curvature_map_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (62.7s) one-pass; '
                'late site only by design (mid '
                'attenuation quantified in 3035); '
                'sham0 maxz 14.0 registered as '
                'saturation ill-conditioning (P->1 '
                'makes kappa_pred explode) - chance '
                'reference only, logic rows '
                'unaffected',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 175
    l14['connects'].append({
        'meas_id': 'meas3036_omega_p33_fingerprint_'
                   'curvature_map_qwen',
        'phase': 3036,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P33: fingerprint '
                        'curvature map - H2 (high-'
                        'curvature fingerprint wins '
                        'from behind) REJECTED: '
                        'med_max_z 0.564 << 3.0, all '
                        'logic rows <= 1.37; curvature '
                        '= operating-point property '
                        '(sign 86/87 = 0.9885, sham '
                        '0.875); own-direction '
                        'curvature = random floor '
                        '(0.87x) vs difference-'
                        'direction specificity 13.2x '
                        '(3035); fingerprints low-'
                        'overlap cos 0.123 vs 0.051 '
                        'random; '
                        'fp_curvature_logistic_'
                        'uniform_qwen'})
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
if '## Phase 3036:' not in memo:
    sec = u'''## Phase 3036: Ω-P33 指纹曲率地图——H2 拒绝：曲率=操作点属性，无指纹特异曲率 [%(created)s]

**判决：`fp_curvature_logistic_uniform_qwen`**（run1 权威 62.7s 一次通过，锚全套通过；late 位 only by design——中带衰减已由 3035 atten=0.037 定量）

### 设计（逐指纹曲率探针 + 随机方向噪声底线）
每 prompt 取基线 top-8 token，沿**自身指纹方向** d_t = W_U[t]/‖W_U[t]‖（非 3035 的差方向）在 L35 输入位注入 ±m·‖h‖，m∈{±0.01,±0.02}；二阶差分 κ_t = [P(+d)+P(−d)−2P(0)]/δ²；logistic 操作点预测 **κ_pred = b_t²·(1−2P)/(P(1−P))**（b_t = 奇斜率经验值，**无自由曲率参数**）；噪声底线 θ = 每行 10 个高斯随机单位方向（seed 9036）的 |κ_rand| 95 分位；z_t = |κ_obs−κ_pred|/θ。

### 核心结果（重复三遍）
**① H2 被拒绝**：med_max_z = **0.564 ≪ 门 3.0**（11/11 logic tag 全部 ≤1.37；med z 0.127）——**不存在超越 logistic 操作点预测的指纹特异曲率**；附件"高曲率指纹可从落后逆袭"在 Qwen3-4B 读出位**被证伪**。**② 曲率符号 86/87**（frac 0.9885，n_elig=87 = 11×top-8 排除拐点窗 |P−0.5|<0.05；sham 21/24）——top-8 全员服从 sigmoid f'' 符号律。**③ 特异性在差方向不在自身方向**：spec_curv = 0.87——自身指纹方向的曲率幅度 ≈ 随机方向底线，而 3035 差方向特异 13.2×——**读出竞争的定向性由 (ŵ_A−ŵ_B) 承载，曲率由操作点决定，二者正交分解**。**④ 正交性副产**：top-8 指纹对 med|cos| = 0.123 vs 随机行对 0.051——低重叠（2.4× 随机）但非正交，附件"指纹近似正交"部分成立。

### 机制解读
统一机制链**定型**：种子 → L3 联盟中继 → 阻尼场均衡 → **logistic 读出（曲率=操作点普适函数，无指纹特异曲率；竞争=差方向特异、多体场 κ≈0.45）**。附件三断言终审：指纹=动态盆地 ✅；竞争=非线性 ✅（但多体+阻尼）；高曲率逆袭 ❌（曲率均一）。指纹在几何上低重叠（cos 0.12）但曲率均质——"逆袭"无几何基础。

### 缺陷与修正登记（如实）
sham0 maxz=14.0 登记为**饱和病态例**（P=0.9915 → P(1−P)→0 使 κ_pred 爆炸、θ 极小 11.0）——仅作 sham 机会参考，不影响 logic 判决；教训入册：**二阶差分检验须报操作点并排除饱和区**。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3036/omega_p33_fingerprint_curvature_map_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3037 菜单——A（主选）**情景性检验**（同词异位 K,V 相似度，直接检验 3013 情景式 KV）；B 深峰头簇公共核心解剖（3034 小公共核心是否跨 ldp 簇共享）；C 跨模型复刻（DS7B/GLM4 四件套：β<1、γ 增长、logistic 读出、曲率均一）；D 多体场质量流分解（κ=0.45 的去向谱）。
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
if 'Phase 3036' not in prev:
    line = ('- Phase 3036 Omega-P33: verdict '
            'fp_curvature_logistic_uniform_qwen '
            '(run1 62.7s one-pass; anchors '
            'a52/a53/a54/a55/a56 bit-level 0.0, '
            'a54 ratio 4.06e-3, a57 source seals, '
            'a51 0.065); H2 REJECTED: med_max_z '
            '0.564 << 3.0 (all logic rows <= 1.37), '
            'curvature = operating-point property, '
            'sign 86/87 = 0.9885; own-direction '
            'curvature = random floor (0.87x) vs '
            'difference-direction 13.2x (3035); '
            'fingerprints low-overlap cos 0.123 vs '
            '0.051; sham0 maxz 14.0 = saturation '
            'ill-conditioning registered; ledger '
            '175/L14 143.\n')
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
- 干预相位锚族（3035/3036）：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；a51 手工 norm+lm_head 重算 top2 恒等+0.15 门；a57 源封印校验。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）。
- **null 门不得设在接收干预的量上**（3035 run4 教训）；**二阶差分/曲率检验须报操作点 P0 并排除饱和区**（3036 sham0：P→1 使 κ_pred 病态）。
- 比值/凸超额报 (log 基线, gamma) 二元组；集合统计用精确超几何 null；定向干预剂量报 m·‖h‖ 绝对量。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向（3035 spec 13.2×）→曲率符号报操作点（logistic 拐点）→曲率残差对照随机方向底线（3036）→消融差分=直接+重平衡→读出集中对照任意扰动 null。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；hook 改输出用返回值+active 门；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；**step-2 前向污染 KV cache→每链重新 prefill**；**pre-hook with_kwargs 改输入须返回 (new_args,new_kwargs) 且注册先于捕获 hook**；lm_head 行数=config.vocab_size 非 tok.vocab_size。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；`cmd &` 孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3036）
Ω-P2（3011-3036）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3018-3019 抵消=通用抑制场；3020 注入特异 944×；3021 MLP 中继 69pct；3022 稀疏联盟 top32=82pct；3023 零消融有毒；3024 联盟承重 0.657；3027 消费=通用读出；3028 剂量凸增长；3029-3030 凸=读出本征；3031/3033 异质性=tag 特异+比值伪影主因（β 0.649 CI<1）；3032 深峰=头集中 89pct；3034 头集合=部分共享通路（J 0.188 p 0.0034，按 ldp 层聚类）；3035 指纹竞争=logistic 读出（inflection 10/11 随操作点翻转、差方向特异 13.2×、κ 0.448 多体场、L8 阻尼 27×）；3036 **曲率地图=H2 拒绝（med_max_z 0.564<3：无指纹特异曲率，曲率=操作点属性；符号 86/87；自身方向曲率=随机底线 0.87×；指纹低重叠 cos 0.123 vs 0.051）**。核心：重要性=关系属性；统一链=种子→L3 联盟中继→阻尼场均衡→logistic 读出（曲率均一、竞争差方向特异）。

## 下一步
- max=3036，下一个 3037（A 主选 **情景性检验**——同词异位 K,V 相似度，直接检验 3013 情景式 KV；B 深峰头簇公共核心；C 跨模型复刻 DS7B/GLM4 四件套：β<1、γ 增长、logistic 读出、曲率均一；D 多体场质量流分解 κ=0.45 去向谱；附件审计 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
