# -*- coding: utf-8 -*-
"""Phase 3041 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3041'
     r'\omega_p38_situational_axis_qwen')
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
assert verdict == 'sitaxis_multirank_qwen', verdict
assert res['anchor_all_ok'] is True
an = res['anchors']
assert an['a79_dup_prefill_bit'] == 0.0
assert an['a80_dup_all_bit'] == 0.0
assert an['a81_top2_ok'] is True
assert an['a81_maxdiff'] <= 0.15
assert an['a82_basis_orth_max'] <= 1e-5
assert an['a83_source_seals'] is True
assert an['a84_cross_phase_bit'] == 0.0
assert an['a84_matched'] == 25
assert an['a85_v3_bit'] == 0.0
assert an['a85_v20_bit'] == 0.0
assert an['a85_sit3_diff'] <= 1e-12
w3 = res['W3']
assert w3['n_words'] == 13
t1 = res['T1_rank_spectrum']
assert abs(t1['obs_med_PR']
           - 1.9415513453833941) < 1e-12
assert abs(t1['null_med_PR']
           - 1.817652676604078) < 1e-12
assert abs(t1['p_t1'] - 0.96678) < 1e-12
t2 = res['T2_prefix_pairs']
assert t2['n_sw_pairs'] == 87
assert abs(t2['slope'] - 0.3108) < 5e-4
assert t2['ci_slope'][0] > 0 and t2['ci_slope'][1] > 0
assert t2['n_high'] == 3
assert abs(t2['stat_subset'] - 1.4499) < 5e-4
assert abs(t2['p_subset'] - 0.00235) < 1e-12
t3 = res['T3_axis_identity']
assert abs(t3['obs_med_abs_cos']
           - 0.1015) < 5e-4
assert abs(t3['p_t3'] - 0.00025) < 1e-12
t4 = res['T4_layer20']
assert t4['n_words_ok'] == 13
assert abs(t4['p_t4'] - 0.94996) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3041
           for m in led['measurements']):
    claim = (
        'Omega-P38 (plan 3041 A) - rank-1 situational '
        'axis anatomy on the 3040 residual: bank + '
        'extraction + decomposition VERBATIM 3040 '
        'run3 (92 occ x 26 types, degenerate rows '
        'excluded, 60 ok rows); word set W3 = 13 '
        'words with >=3 non-degenerate occurrences. '
        'Anchors: a79/a80 bit 0.0; a81 0.0453; a82 '
        'orth 1.8e-15; a83 seals (incl 3040); a84 '
        'cross-phase vs 3037 npz 25/25 bit 0.0; a85 '
        'full-bank chain anchor vs 3040 npz V3/V20 '
        'bit 0.0 + SIT3 diff 0.0.  T1 per-word '
        'residual spectrum: median participation '
        'ratio 1.9416 vs construction-matched null '
        '1.8177, p(P(null<=obs)) 0.96678 - residuals '
        'are significantly MORE multi-rank than the '
        'matched null: the 3040 rank-1 anti-parallel '
        'median signature does NOT imply a single '
        'axis per word (per-word PR 1.6-2.7 of max '
        '3; sole exception: yet PR=1.0 EXACTLY - '
        'perfectly collinear residuals, e1=1.0).  '
        'T2 same-prefix minimal pairs: OLS slope of '
        'cos_dev ~ Lshare +0.311 CI [+0.212,+0.507] '
        'excludes 0; 3 pairs with Lshare>=2 have '
        'subset stat +1.4499 (med cos ~ +1.0 vs '
        'global med -0.45), exact random-subset p '
        '0.00235 - identical left-prefix contexts '
        'REPRODUCE the situational residual (3040 '
        'was cos+1.0 controlled generalization).  '
        'T3 axis identity: median cross-word '
        '|cos| 0.1015 vs size-matched Gaussian null '
        '0.0672, p 0.00025 - words SHARE a weak '
        'common situational direction (1.5x random; '
        'not orthogonal, echoing 3036).  T4: L20 '
        'spectrum same multirank (obs 1.971, p '
        '0.94996) - rank structure NOT L3-specific, '
        'unlike the context lock.  CONCLUSION: '
        'situational residual = multi-dimensional '
        'context code: strong local (left-prefix) '
        'determinism + weak shared axis across '
        'words + per-word 2-3 dim geometry; 3040 '
        "rank-1 label downgraded to dominant-axis + "
        'secondary spread; the anti-parallel median '
        'and the spectrum are reconciled (dominant '
        'axis + secondary components).  NEXT: style-'
        'field probe (global prompt-identity vs '
        'local prefix in the situational code), '
        'cross-lingual shared subspace, nested-'
        'subspace orthogonality quantification, '
        'cross-model replication.')
    meas = {
        'meas_id': 'meas3041_omega_p38_situational_'
                   'axis_qwen',
        'phase': 3041,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a79/a80/a84/a85 bit 0.0; a81 '
                   '0.0453; a82 1.8e-15; a83 seals; '
                   'T1 PR 1.942 vs null 1.818 p 0.967 '
                   '(multirank); T2 slope +0.311 CI '
                   'excl 0, subset stat 1.45 p 0.0024; '
                   'T3 |cos| 0.102 vs 0.067 p 0.00025; '
                   'T4 L20 p 0.95',
        'artifacts': {
            'result': 'phase3041/omega_p38_'
                      'situational_axis_qwen/'
                      'result.json',
            'npz': 'phase3041/omega_p38_'
                   'situational_axis_qwen/'
                   'omega_p38_situational_axis_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (29.7s); '
                'construction-matched null (3040 T2 '
                'machinery) reused for the spectrum '
                'test; Lshare>=1 fallback not needed '
                '(3 pairs at Lshare>=2)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 180
    l14['connects'].append({
        'meas_id': 'meas3041_omega_p38_situational_'
                   'axis_qwen',
        'phase': 3041,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P38: situational axis '
                        'anatomy - per-word residuals '
                        'MORE multi-rank than matched '
                        'null (med PR 1.942 vs 1.818, '
                        'p 0.967; yet exactly rank-1 '
                        'PR=1.0); same-prefix minimal '
                        'pairs reproduce residual '
                        '(slope +0.311 CI excl 0; '
                        'Lshare>=2 subset cos ~ +1.0, '
                        'p 0.0024); weak shared axis '
                        'across words (|cos| 0.102 vs '
                        '0.067 null, p 0.00025); L20 '
                        'same multirank - situational '
                        'code = multidimensional, '
                        'left-prefix deterministic, '
                        'weakly shared; 3040 rank-1 '
                        'label downgraded to '
                        'dominant-axis + secondary '
                        'spread; sitaxis_multirank_'
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
if '## Phase 3041:' not in memo:
    sec = u'''## Phase 3041: Ω-P38 秩1情境轴解剖——谱水平推翻单词秩1轴（PR 1.94 显著高于匹配 null 0.967）；同前缀最小对 cos≈+1.0 复现（p=0.0024）；跨词弱共享轴 1.5× 随机（p=0.00025） [%(created)s]

**判决：`sitaxis_multirank_qwen`**（run1 权威 29.7s；七锚全过，a79/a80/a84/a85 位级 0.0）

### 设计（3040 遗留 A：秩1情境轴解剖）
提取/分解/退化行剔除 verbatim 3040 run3（92 出现 × 26 类型，60 非退化行）；W3 = 13 个 ≥3 非退化出现词。T1 逐词残差谱：gram 特征值参与率 PR=(Σλ)²/Σλ²，统计量=词中位 PR，零假设=构造匹配置换（伪组均值重投影，3040 T2 机械复用，50k）；T2 同前缀最小对：Lshare=出现位置前公共左 token 前缀长，cos~Lshare OLS+bootstrap CI，Lshare≥2 子集对全体的差 + 精确随机子集零假设（预注册 Lshare≥1 兜底，实际 3 对≥2 未触发）；T3 轴身份跨词：主轴 |cos| 中位 vs 尺寸匹配高斯 null（20k）；T4 L20 谱对照。新锚 a85：全库链锚 vs 3040 npz——V3/V20 全 92 行位级 0.0 + SIT3 差 0.0。

### 核心结果（重复三遍）
**① 谱水平推翻"单词秩1轴"**：obs 词中位 PR=**1.9416** vs 构造匹配 null **1.8177**，p(P(null≤obs))=**0.96678**——真实残差显著**更**多维（PR 1.6–2.7，上限 3）；3040 的"秩1反平行几何"降级为**主轴+次级分量**（反平行中位数签名与多维谱兼容：主轴主导+次级展开）。唯一例外 **'yet' PR=1.0 精确**（三残差完全共线，e1=1.0）——完美秩1单词存在但是个案。**② 同前缀最小对受控推广成功**：cos~Lshare 斜率 **+0.311** CI[+0.212,+0.507] 不含 0；Lshare≥2 的 3 对子集 stat=**+1.4499**（子集 cos≈+1.0 vs 全体 med −0.45），精确随机子集 p=**0.00235**——同左前缀语境**复现**情境残差（3040 'was' cos+1.0 的受控推广成立）。**③ 跨词弱共享轴**：词间主轴 |cos| med **0.1015** vs 匹配高斯 null **0.0672**，p=**0.00025**（1.5× 随机；非正交，呼应 3036）。**④ L20 谱同样多维**（obs 1.971，p=0.95）——秩结构非 L3 特异，与语境锁定（L3 特异）形成对照。

### 机制链更新
情境残差=**多维上下文码**：强局部门控确定性（左前缀复现 cos≈+1）+ 跨词弱共享公共方向 + 逐词 2–3 维几何。L3 KV 写入最终四分量图像：公共中继（3037）+ 词身份（98.3pct，32/92 完全刻板）+ 多维情境码（3041：局部确定、弱共享、非秩1）+ 层内位置梯度（3040 T3）。**3040"秩1情境残差"表述已在账本中降级**；'yet' 个案证明完美秩1通道可对个别词出现。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3041/omega_p38_situational_axis_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3042 菜单**——A（主选）**风格场探针**：情境码的全局性检验（全局 prompt 身份/风格前缀 vs 局部前缀——加风格前缀句测量 V 写入位移，直接检验附件"全局引力场"主张）；B 跨语言共享子空间（同词 EN/ZH 语境深层 V 对齐，检验附件"语义引力井同构"）；C 嵌套子空间正交性量化（hypernym 对 apple/fruit/food 的 W_up/W_down 子空间，直接检验附件"俄罗斯套娃+正交"）；D 跨模型复刻。
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
if 'Phase 3041' not in prev:
    line = ('- Phase 3041 Omega-P38: verdict '
            'sitaxis_multirank_qwen (run1 29.7s, 7 '
            'anchors ok, a85 full-bank chain anchor '
            'vs 3040 bit 0.0); per-word residuals '
            'MORE multirank than construction-matched '
            'null (med PR 1.942 vs 1.818 p 0.967; yet '
            'exactly rank-1 PR=1.0); same-prefix '
            'minimal pairs reproduce residual (slope '
            '+0.311 CI excl 0, Lshare>=2 subset cos '
            '~+1.0 p 0.0024); weak shared axis across '
            'words (|cos| 0.102 vs 0.067 p 0.00025); '
            'L20 same multirank; 3040 rank-1 label '
            'downgraded to dominant-axis + secondary '
            'spread; ledger 180/L14 148.\n')
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
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结（PREREG/锚/判决）→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）。
3. 重跑先删旧 execution/result/npz；负结果与判据作废如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格；文本内裸百分号写 %%（3033 教训）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08；bit 级仅限同文件链/上游全精度；跨相位 npz 锚均 0.0；剂量两端点锚死。
- 干预锚族：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；手工 norm+lm_head 重算+0.15 门；源封印；跨相位 npz 位级锚（3040 a78 vs 3037 25/25；**3041 a85 全库链锚 vs 上相位 npz：V3/V20 位级 0.0 + 派生量 SIT 容差 1e-12**——派生量与提取量门限分离）。重复臂方向必须与原臂同对象（3039）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；margin n≳40（n=11 探索性）。
- null 门不得设在接收干预的量上（3035）；曲率检验排除饱和区（3036）；跨相位锚定前核对读出协议量纲（3037）。
- **余弦守卫零值毒化中位数/置换统计→退化行剔除并报计数（3040）**；**小 n 组内去均值→构造匹配置换（伪组均值重投影，可复用于谱检验 3041）**；**中位数签名与谱结构可分歧（反平行 med ≠ 单轴）——几何标签命名前必查谱（3041）**；子集零假设用等尺寸随机子集精确化。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→曲率符号报操作点→KV 相似性三分量（3037）→读出协议条件性标注（3038/3039）→残差检验防居中基线与守卫零伪影（3040）→**谱水平复核几何标签（3041）**→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；pre-hook with_kwargs 返回 (args,kwargs)；**SVD 行空间基底取 Vt[:r].T 而非 U**（3040）；gram 特征值算 PR/主轴（3041）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3041）
Ω-P2（3011-3041）：3011 门控=L3 KV；3018-3019 通用抑制场；3020 注入特异 944×；3021-3022 L3 联盟中继 82pct；3024 承重 0.657；3028 剂量凸增长；3031/3033 异质性=比值伪影；3032 深峰=头集中 89pct；3035 指纹 logistic（特异 13.2×）；3036 曲率=操作点属性+指纹非正交（2.4× 随机）；3037 KV=词身份主导+中继+语境调制；3038/3039 协议分层（曲率=再入条件；特异/多体/阻尼协议鲁棒）；3040 情景分量：词身份 98.3pct+32/92 刻板；残差语境锁 p=0.012 L3 特异；同词反平行超匹配 null；3041 **谱推翻单词秩1轴（PR 1.94 vs null 1.82 p=0.967，yet 个案 PR=1.0）；同前缀复现 cos≈+1（p=0.0024）；跨词弱共享轴 1.5× 随机（p=0.00025）→ 情境码=多维、局部确定、弱共享；L3 KV 四分量定版（中继+词身份+多维情境码+位置梯度）**。

## 下一步
- max=3041，下一个 3042（A 主选 **风格场探针**——情境码全局性检验：风格/主题前缀句 vs 局部前缀，V 写入位移，检验附件 HDMCC"全局引力场"；B 跨语言共享子空间；C 嵌套子空间正交性量化 apple/fruit/food；D 跨模型复刻）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
