# -*- coding: utf-8 -*-
"""Phase 3034 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3034'
     r'\omega_p31_headset_identity_qwen')
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
assert verdict == 'headset_pathway_qwen', verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2a = res['T2a']
assert t2a['n_valid'] == 10
assert t2a['J_obs'] == 0.188407
assert t2a['p_T1'] == 0.00337
assert t2a['null1_med'] == 0.149588
assert t2a['null_expect'] == 0.142857
assert t2a['nperm'] == 200000
assert t2a['gates_ok'] is True
assert t2a['sets_per_tag']['P1:4'] == \
    t2a['sets_per_tag']['P11:3']
t2b = res['T2b']
assert t2b['J_deep_mid'] == 0.126886
assert t2b['p_T2'] == 0.780856
assert t2b['n_pairs2'] == 10
assert t2b['group7_count_obs'] == 3
assert t2b['group7_expect'] == 10.0
assert t2b['p_T3'] == 0.999265
assert t2b['med_spearman_dd_ph'] == -0.071298
assert t2b['p_T4'] == 0.346988
assert res['T2c']['med_js_sham'] == 0.000143
assert res['T2c']['a41_ident_med'] == 0.0001
an = res['anchors']
assert an['a28_erase_chain_diff'] == 0.0
assert an['a30_capture_self_diff'] == 0.0
assert an['a32_dose_alpha0_diff'] == 0.0
assert an['a38_lens_terminal_rel'] < 1e-4
assert an['a39_traj_3030_diff'] == 0.0
assert an['a40_dnorm_3029_diff'] == 0.0
assert an['a41_s_relay_diff'] < 1e-6
assert an['a42_headshare_recompute'] == 0.0
assert an['a43_E_matrix_3032'] == 0.0
assert an['a44_ldp_recompute'] == 0.0
assert an['a45_p3027_sanity'] < 1e-6
assert an['a46_source_seals'] is True

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3034
           for m in led['measurements']):
    claim = (
        'Omega-P31 (plan v5 P31) - deep-peak '
        'head-SET identity: GPU rerun of the '
        'verbatim 3032 machine (anchors a0-a41 '
        'all re-passed in-run: a2 dirs 2.17e-08; '
        'a28/a30/a32/a39/a40 bit-level 0.0; a38 '
        '1.58e-06; a41 2.33e-08) with per-head '
        'dd = squared d-o norm stashed at each '
        'tag deep-peak (ldp) and mid-peak '
        'layers; new anchors a42 head_top8 '
        'share recompute bit-level 0.0, a43 E '
        'matrix bit-level 0.0, a44 ldp '
        'recompute 0.0, a45 3027 p_h row-sum '
        '2.22e-16, a46 source seals.  PRIMARY '
        'T1: mean pairwise Jaccard of per-tag '
        'top-8 head sets J_obs 0.1884 vs '
        'hypergeometric null med 0.1496 (exact '
        'expectation 2/14=0.1429), p 0.00337 '
        '(200k, seed 30341) - ABOVE null => '
        'headset_pathway_qwen, but far below '
        '1.0: only ~2.5/8 heads shared per '
        'pair on average (small cross-tag core '
        '+ tag-specific periphery).  T2: '
        'deep-vs-mid-peak same-tag set Jaccard '
        '0.1269, p 0.781 - NOT above null: '
        'different depths recruit different '
        'heads.  T3: GQA group-7 count 3 vs '
        'expected 10, p 0.999 - deep-peak '
        'heads are NOT the L3 consumption '
        'group.  T4: Spearman(dd, 3027 p_h) '
        'med -0.071, p 0.347 - unrelated to '
        'consumption profile.  Descriptive: '
        'sets cluster by ldp layer (L22-24 / '
        'L25 / L30-31); P1:4 and P11:3 sets '
        'identical (J=1).  Interpretation: '
        'the deep readout carrier is a '
        'layer-organized, partially shared '
        'pathway - NOT the L3 relay coalition '
        'route (T3/T4) and NOT purely '
        'contextual (T1 above null).  NEXT: '
        'fingerprint competition anatomy '
        '(directional w_A - w_B readout '
        'intervention), fingerprint curvature '
        'map (H2 test), or situational '
        'specificity.')
    meas = {
        'meas_id': 'meas3034_omega_p31_headset_'
                   'identity_qwen',
        'phase': 3034,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'full 3032 suite + a42/a43/'
                   'a44 bit-level 0.0; a45 '
                   '2.22e-16; a46 seals ok; '
                   'T1 p 0.00337',
        'artifacts': {
            'result': 'phase3034/omega_p31_'
                      'headset_identity_qwen/'
                      'result.json',
            'npz': 'phase3034/omega_p31_'
                   'headset_identity_qwen/'
                   'omega_p31_headset_identity_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative (158.6s; '
                'run1 NameError io.open in new '
                'stats block, fixed to open(), '
                'no prereg change); verdict via '
                'preregistered T1 branch; '
                'partially-shared pathway: J '
                '0.188 vs null 0.150 - shared '
                'core ~2.5/8 heads per pair; '
                'layer-cluster organization '
                '(L22-24/L25/L30-31); NOT '
                'group7, NOT consumption-'
                'aligned.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 173
    l14['connects'].append({
        'meas_id': 'meas3034_omega_p31_headset_'
                   'identity_qwen',
        'phase': 3034,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P31: deep-peak '
                        'head-SET identity - '
                        'cross-tag Jaccard 0.188 '
                        'vs hypergeometric null '
                        '0.150 (p 0.0034): '
                        'partially shared pathway '
                        '(~2.5/8 heads per pair), '
                        'organized by ldp layer '
                        'clusters (L22-24/L25/'
                        'L30-31, P1/P11 sets '
                        'identical); deep-vs-mid '
                        'sets differ (p 0.78); '
                        'NOT GQA group7 (K 3 vs '
                        '10, p ~1) and NOT 3027 '
                        'consumption-aligned '
                        '(rho -0.07, p 0.35); '
                        'anchors full 3032 suite '
                        '+ a42-a46 all pass'})
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
if '## Phase 3034:' not in memo:
    sec = u'''## Phase 3034: Ω-P31 深峰头集合身份——部分共享的通路（J 0.188>随机 0.150，按 ldp 层聚类组织；非 g7、非消费画像）[%(created)s]

**判决：`headset_pathway_qwen`**（run2 权威 158.6s，verbatim 复刻 3032 机器；run1 io.open NameError 修复无预注册变更；锚全套通过：3032 原有 a0–a41 全部复过（a2 dirs 2.17e-08、**a28/a30/a32/a39/a40 位级 0.0**、a38 1.58e-06、a41 2.33e-08）+ 新锚 **a42 share 重算 / a43 E 矩阵 / a44 ldp 重算 位级 0.0** + a45 2.22e-16 + a46 源封印）

### 设计（verbatim 3032 机器 + dd stash）
GPU 重跑 3032 全链（逐 tag：capture_base + α=0/1/2 擦除链 + lens 轨迹 + sham），在 deep_share_head 内 stash 逐头 dd=‖Δo‖²；ldp 处 top-8 头集合（稳定 argsort）用于集合统计。T1 主检验：跨 tag 平均两两 Jaccard vs 超几何精确 null（32 选 8 重叠分布，200k，seed 30341）；T2 深峰-vs-中峰同 tag 集合 Jaccard（seed 30342）；T3 GQA group7 富集（seed 30343）；T4 Spearman(dd, 3027 p_h 消费画像)（seed 30344）。

### 核心结果（重复三遍）
**① 头集合是部分共享的通路**：J_obs = **0.1884** vs null med 0.1496（精确期望 2/14=0.1429），**p=0.00337**——高于随机，但远低于 1：平均每对 tag 只共享 ~2.5/8 个头（小的跨 tag 公共核心 + tag 特异外围）。**② 组织轴=层而非 tag 语义**：ldp 聚成 L22–24 / L25 / L30–31 三簇，同簇 tag 集合高度重叠（**P1:4 与 P11:3 集合完全相同 J=1**，P2/P11 共享 7/8）；T2 深峰-vs-中峰集合 J=0.1269（p=.78，不高于随机）——**不同深度招募不同头**。**③ 排除两个"预定载体"假设**：深峰头**不在 GQA group7 富集**（K_obs=3 vs 期望 10，p=.999——g7 是 L3 消费组但非深读出通路）；与 3027 消费头画像**无关**（med ρ=−0.071，p=.347）。头载体既非通路共享全程、也非纯上下文——是**按深度组织的第三类结构**。

### 机制解读
载体地图终版：L3 = MLP 稀疏联盟中继（82%% 集中于 32 神经元）；深层读出 = 头承载（89%%）但头集合按 ldp 层分簇、跨 tag 部分共享。这与 3027"消费=通用读出结构"和 3032"头=每深度读出载体"合并成：**上游写入通道（L3 KV）与下游读出通道（深头簇）是分离的两套结构，中间由分布式阻尼场衔接**——头簇像"按深度索引的读出总线"，每条总线服务落入该深度带的峰位。

### 缺陷与修正登记（如实）
run1 新统计块 io.open NameError（3032 模板用 open），改 open 后 run2；无预注册变更。P7 无深峰（nan）按预注册排除（n_valid=10）。T1 null 用超几何精确分布（集合均匀抽样下重叠精确服从 Hypergeom(32,8,8)），无渐近近似。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3034/omega_p31_headset_identity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3035 菜单——A（主选）**指纹竞争解剖**（对 top-2 竞争 token 定向 ±δ·(ŵA−ŵB) 读出干预，测 P(A)/P(B) 迁移对称性，操作化附件"竞争"假设）；B 指纹曲率地图（H2 判决：逐指纹曲率 vs 3030 均匀放大）；C 情景性检验（同词异位 K,V 相似度）；D 深峰头簇公共核心解剖（①的小核心是否跨 ldp 簇共享）。
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
if 'Phase 3034' not in prev:
    line = ('- Phase 3034 Omega-P31: verdict '
            'headset_pathway_qwen (GPU verbatim '
            '3032 machine run2 158.6s; run1 '
            'io.open NameError fixed; full '
            'anchor suite incl a28/a30/a32/a39/'
            'a40 bit-level 0.0 + new a42-a44 '
            'bit-level 0.0); T1 cross-tag top-8 '
            'head-set Jaccard 0.188 vs null '
            '0.150 (p 0.0034): partially shared '
            'pathway organized by ldp layer '
            'clusters (P1/P11 sets identical); '
            'deep-vs-mid sets differ (p 0.78); '
            'NOT GQA group7 (K 3 vs 10) and '
            'NOT consumption-aligned (rho '
            '-0.07); ledger 173/L14 141.\n')
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
4. MEMO 占位符一律 %(key)s 风格（{key} 不替换，2971）；MEMO 文本内裸百分号写 %%（3033 教训）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；跨相位 npz 锚（js2/js_erase/α0/ratio2/E/ldp）均 0.0；剂量两端点锚死 α=1=擦除/α=0=恢复。
- GPU 复刻相位：verbatim 拷贝旧脚本+外科补丁（assert count==1），全套旧锚在新 run 内复过（3034）。
- 重分析相位锚=源 npz/result 记账值重算恒等（门 5e-5）+ seal.json 完整性；置换 null 广播写法 (rx[None,:]·ry[perms]).sum(axis=1)。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）；镜像 −dirs 必配。
- 比值/凸超额必须报 (log 基线, gamma) 二元组（3031/3033：β 0.649 CI 上界<1=伪影主因；tag 内 γ 随剂量长 0.74→1.92=真信号）。
- 集合统计用精确超几何 null（32 选 8 重叠=Hypergeom(32,8,8)，期望 J=2/14）。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因（家族 maxT + log-log 尺度）→集合身份对照随机 null（3034 J 0.188 vs 0.150）→消融差分=直接+重平衡→功能局域≠几何符号身份→读出集中须对照任意扰动 null。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；hook 改输出用返回值+active 门；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；**step-2 前向污染 KV cache→run_chain 每链重新 prefill**；捕获门与 patch 门独立；3027 模板用 open() 非 io.open。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；`cmd &` 孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3034）
Ω-P2（3011-3034）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费领先 g7；3018-3019 抵消=通用抑制场；3020 注入特异 944×；3021 MLP 中继 69pct；3022 稀疏联盟 top32=82pct；3023 零消融有毒；3024 联盟承重 0.657；3025-3026 非联盟保护/B1 null-like；3027 消费=通用读出；3028 剂量凸增长；3029 凸=读出本征；3030 凸=分布式读出（lens 2.6-2.9×）；3031 异质性 tag 特异（0/9）；3032 深峰=头集中（89pct）；3033 异质性=比值伪影主因+25pct 形状残差；3034 **深峰头集合=部分共享通路（J 0.188 vs null 0.150 p 0.0034；按 ldp 层聚类 L22-24/L25/L30-31 组织；P1/P11 集合恒等；非 g7、非消费画像、深-vs-中集合不同）**。核心：null 重编码全层分布式涌现；重要性=关系属性；载体地图=L3 MLP 中继 + 深度分簇头读出总线 + 阻尼场衔接。

## 下一步
- max=3034，下一个 3035（A 主选 **指纹竞争解剖**——top-2 竞争 token 定向 ±δ·(wA−wB) 读出干预测迁移对称性；B 指纹曲率地图 H2 判决；C 情景性检验；D 深峰头簇公共核心解剖；附件审计见 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
