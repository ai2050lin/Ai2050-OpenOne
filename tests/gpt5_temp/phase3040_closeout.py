# -*- coding: utf-8 -*-
"""Phase 3040 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3040'
     r'\omega_p37_situational_component_qwen')
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
assert verdict == 'sitcomp_context_pure_qwen', verdict
assert res['anchor_all_ok'] is True
an = res['anchors']
assert an['a73_dup_prefill_bit'] == 0.0
assert an['a74_dup_all_bit'] == 0.0
assert an['a75_top2_ok'] is True
assert an['a75_maxdiff'] <= 0.15
assert an['a76_basis_orth_max'] <= 1e-5
assert an['a77_source_seals'] is True
assert an['a78_cross_phase_bit'] == 0.0
assert an['a78_matched'] == 25
dec = res['decomposition']
assert dec['n_occ'] == 92 and dec['n_types'] == 26
assert dec['degenerate_L3'] == 32
assert abs(dec['med_energy_share_L3']
           - 0.9833626311036598) < 1e-12
t1 = res['T1_context_lock']
assert t1['n_same_prompt_diff_word'] == 50
assert t1['n_cross_prompt_diff_word'] == 1633
assert abs(t1['d1'] - 0.056267196783178) < 1e-12
assert abs(t1['p1'] - 0.01236) < 1e-12
t2 = res['T2_word_residual']
assert t2['n_same_word_cross'] == 87
assert abs(t2['d2']
           - (-0.4452205876820161)) < 1e-12
assert abs(t2['p2'] - 0.99979) < 1e-12
t3 = res['T3_decay']
assert t3['ci_dpos'][0] < 0 and t3['ci_dpos'][1] < 0
assert t3['ci_dpfrac'][0] < 0
assert t3['ci_dpfrac'][1] < 0
t4 = res['T4_layer20']
assert abs(t4['p1'] - 0.38134) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3040
           for m in led['measurements']):
    claim = (
        'Omega-P37 (plan 3040 A) - situational '
        'component extraction of the L3 kv7 V write: '
        '3037 bank + protocol verbatim, extended '
        'alphabetic-token bank (92 occurrences x 26 '
        'types, >=2 occurrences each) supplying the '
        'same-prompt different-word pairs the 8-word '
        'target set cannot; per-layer word-identity '
        'subspace = row space of the 26 word-mean V '
        'vectors (SVD basis B=Vt[:r].T), residual SIT '
        '= V - P(V); degenerate rows (residual norm '
        '<1e-6, exactly stereotyped writes) excluded '
        'from all pair statistics.  Anchors: a73/a74 '
        'bit 0.0; a75 manual recompute 0.0453; a76 '
        'basis orth 1.8e-15; a77 seals; a78 cross-'
        'phase vs 3037 npz 25/25 bit 0.0.  '
        'Decomposition: word-identity subspace '
        'captures 98.3pct of V energy (med); 32/92 '
        'writes EXACTLY stereotyped at fp32 (The x12, '
        'He x8, She x4, high/price/tired/We x2) - '
        'their L3 kv7 V write is a deterministic '
        'function of word identity.  T1: residual '
        'context lock SIGNIFICANT - same-prompt '
        'different-word residual cos med +0.0516 vs '
        'cross-prompt -0.0047 (d1 +0.0563, exact '
        'permutation p 0.0124, 50/1633 pairs), and '
        'L3-SPECIFIC (L20 d1 +0.007, p 0.38).  T2: NO '
        'positive word-residual code - same-word '
        'cross-prompt residual med -0.4499 (d2 '
        '-0.4452), construction-matched reconstruction '
        'permutation p 0.9998: same-word deviations '
        'are anti-parallel BEYOND the matched null = '
        'rank-1 deviation geometry (per-word deviation '
        'axis, context flips sign/magnitude); direct '
        'evidence: was at identical prefix (prompts 8 '
        'vs 22) gives residual cos +1.0 exact - the '
        'residual is a deterministic function of '
        'context.  T3: position-gradient decay (dpos '
        'slope -0.0069 CI [-0.0129,-0.0010]; dpfrac '
        'slope -0.0557 CI [-0.0933,-0.0171]).  '
        'CONCLUSION: three-component structure of the '
        'L3 KV write - shared relay component (3037) '
        '+ word identity (98.3pct) + rank-1 '
        'situational residual orthogonal to both; '
        '3013 episodic-KV realized at component '
        'level.  Run history: run1 crash pre-verdict '
        '(SVD basis taken from U instead of Vt - '
        'wrong space); run2 verdict sitcomp_null '
        'VOIDED by probe3040 (exact-0.0 cosine guard '
        'poisoned 2416/4186 pairs via the 32 '
        'degenerate rows, degenerating medians and '
        'the permutation null); run3 fixes registered '
        '(degenerate-row exclusion + construction-'
        'matched null).  NEXT: rank-1 situational '
        'axis anatomy (per-word deviation SVD spectra '
        '+ controlled same-prefix minimal pairs), '
        'context-lock direction localization, cross-'
        'model replication.')
    meas = {
        'meas_id': 'meas3040_omega_p37_situational_'
                   'component_qwen',
        'phase': 3040,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a73/a74 bit 0.0; a75 0.0453; a76 '
                   '1.8e-15; a77 seals; a78 25/25 bit '
                   '0.0; T1 d1 +0.056 p 0.0124 (L20 p '
                   '0.38); T2 d2 -0.445 p 0.9998 '
                   '(matched null); T3 dpos/dpfrac CI '
                   'exclude 0',
        'artifacts': {
            'result': 'phase3040/omega_p37_'
                      'situational_component_qwen/'
                      'result.json',
            'npz': 'phase3040/omega_p37_'
                   'situational_component_qwen/'
                   'omega_p37_situational_component_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run3 authoritative (192.4s); run1 '
                'crash pre-verdict (SVD basis space '
                'bug); run2 verdict VOIDED (guard-zero '
                'poisoning 2416/4186 pairs, probe3040 '
                'evidence); corrections registered in '
                'PREREG',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 179
    l14['connects'].append({
        'meas_id': 'meas3040_omega_p37_situational_'
                   'component_qwen',
        'phase': 3040,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P37: situational '
                        'component extraction - '
                        'word-identity subspace '
                        'captures 98.3pct of L3 kv7 V '
                        'energy, 32/92 writes exactly '
                        'stereotyped (fp32-zero '
                        'residual); residual context '
                        'lock SIGNIFICANT (d1 +0.056, '
                        'p 0.0124) and L3-specific '
                        '(L20 p 0.38); NO positive '
                        'word-residual code (d2 -0.445, '
                        'p 0.9998 vs construction-'
                        'matched null) - same-word '
                        'deviations anti-parallel '
                        'BEYOND matched null = rank-1 '
                        'deviation geometry; identical '
                        'contexts give residual cos '
                        '+1.0 (deterministic context '
                        'function); position-gradient '
                        'decay; sitcomp_context_pure_'
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
if '## Phase 3040:' not in memo:
    sec = u'''## Phase 3040: Ω-P37 情景分量提取——词身份子空间 98.3pct 能量 + 32/92 写入完全刻板；残差语境锁定显著（p=0.012）且 L3 特异；同词偏离=秩1反平行 [%(created)s]

**判决：`sitcomp_context_pure_qwen`**（run3 权威 192.4s；run1 崩溃前修正 + run2 判决被探针证伪作废，均如实登记）

### 设计（3037 遗留 B：情景分量直接提取）
3037 同库同协议 verbatim（28 prompts、8 目标词、纯 prefill、L3/L20 kv7 V 为主判据）；扩展字母 token 库（出现 ≥2 次的 token id，92 出现 × 26 类型——补足 same-prompt 异词对，8 目标词库内无同 prompt 共现）。逐层词身份子空间 = 26 词均值行空间（SVD 正交基，B=Vt[:r].T），残差 SIT = V − P(V)；**退化行剔除**（残差范数 <1e-6 的刻板写入，cosine 无定义）。T1 语境锁定（同/跨 prompt 异词残差 cos 差 + prompt 标签精确置换 100k）；T2 词残差码（**构造匹配置换零假设**：伪组均值重投影，精确复现组内 sum-to-zero 与 size-2 强制反平行——普通对子集置换不复现居中结构）；T3 衰减解剖（dpos/dpfrac OLS+bootstrap CI）；T4 L20 对照。锚：a73/a74 位级 0.0；a75 手工重算 0.0453；a76 基底正交 1.8e-15；a77 封印；a78 跨相位 vs 3037 npz 25/25 位级 0.0。

### 核心结果（重复三遍）
**① 词身份子空间截获 98.3pct 能量（med）**，且 **32/92 写入完全刻板**（残差 fp32 精确 0：The×12、He×8、She×4、high/price/tired/We×2）——大写功能词与部分内容词的 L3 kv7 V 写入是词身份的确定性函数。**② 残差语境锁定显著且 L3 特异**：同 prompt 异词残差 cos med **+0.0516** vs 跨 prompt **−0.0047**（d1=+0.0563，精确置换 **p=0.0124**）；L20 不复刻（d1=+0.007，p=0.38）——情景分量是 L3 中继时刻写入的，非普遍层属性。**③ 无正向词残差码，但同词偏离=秩1反平行几何**：同词跨语境残差 med **−0.4499**（d2=−0.4452），构造匹配零假设下 p=**0.9998**——反平行**超过**匹配零假设（同词三点近共线 vs 一般三角形）：每词偏离沿单词特异轴、语境改符号/幅度；直接证据：'was' 于同前缀句（prompt 8 vs 22 'The price was high, yet…'）残差 **cos=+1.0 精确复现**——残差是上下文的确定性函数。**④ 位置梯度**：slope_dpos=−0.0069 CI[−0.0129,−0.0010]、slope_dpfrac=−0.0557 CI[−0.0933,−0.0171] 均不含 0。

### 缺陷与修正登记（如实）
run1：make_sub SVD 基底取 U（类型空间）而非 Vt（行空间），判决前崩溃。run2：完整跑出 sitcomp_null_qwen 但被 probe3040 证伪作废——32 退化行使 coss 守卫返回精确 0.0×2416/4186 对，中位数/置换零分布全退化（std=0），p≡1 为守卫伪影；探针同时暴露同词残差 −0.476 ≈ −1/(n−1) 居中基线。run3 修正 = ①退化行剔除 ②T2 升级构造匹配置换。**教训入册：① 余弦守卫零值会毒化中位数与置换统计（必须剔除退化行并报告计数）；② 小 n 组内去均值强制 −1/(n−1) 反平行基线，同构检验须用构造匹配零假设。**

### 机制链更新
L3 kv7 V 写入三分量定版：**公共中继分量（3037，异词间 cos 0.55）+ 词身份分量（98.3pct 能量，部分完全刻板）+ 秩1情境残差（与两者正交；同 prompt 弱对齐、跨 prompt 正交、同词反号）**——3013"情景式 KV"获得分量级落实：语境调制不在词方向内，而在正交补空间的低秩情境轴上；残差对上下文确定性（同前缀 cos+1.0）指向可解码的情境码。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3040/omega_p37_situational_component_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3041 菜单——A（主选）**秩1情境轴解剖**（逐词偏离 SVD 谱 + 同前缀最小对受控设计：偏离符号/幅度 vs 上下文特征回归，'was' cos+1.0 的受控推广）；B 语境锁定方向定位（same-prompt 对齐方向的子空间/通道归属，L3 特异性来源）；C 阻尼场通道分解（再入 vs 直读衰减谱差）；D 跨模型复刻（KV 三分量 + 情景分量协议上 DS7B/GLM4）。
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
if 'Phase 3040' not in prev:
    line = ('- Phase 3040 Omega-P37: verdict '
            'sitcomp_context_pure_qwen (run3 192.4s; '
            'run1 crash pre-verdict SVD basis space '
            'bug; run2 VOIDED - guard-zero poisoned '
            '2416/4186 pairs via 32 degenerate rows, '
            'probe3040); word-identity subspace 98.3pct '
            'energy, 32/92 writes exactly stereotyped '
            '(The x12/He x8/She x4...); residual '
            'context lock significant d1 +0.056 p '
            '0.0124 and L3-specific (L20 p 0.38); no '
            'positive word-residual code - same-word '
            'deviations anti-parallel beyond '
            'construction-matched null = rank-1 '
            'deviation geometry; was identical-prefix '
            'residual cos +1.0; position-gradient '
            'decay CIs exclude 0; ledger 179/L14 147.\n')
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
- 干预锚族：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；手工 norm+lm_head 重算+0.15 门；源封印；跨相位 npz 位级锚（3040 a78 vs 3037 25/25）。**重复臂方向必须与原臂同对象**（3039）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）。
- null 门不得设在接收干预的量上（3035）；曲率检验排除饱和区 |P0−0.5|<0.05（3036）；**跨相位锚定前核对读出协议量纲：再入 step-2 ≠ 直接 prefill**（3037）。
- **余弦守卫零值毒化中位数/置换统计→退化行（范数<1e-6）剔除并报计数（3040）**；**小 n 组内去均值强制 −1/(n−1) 反平行基线→同构检验用构造匹配置换（伪组均值重投影）**（3040）。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→曲率符号报操作点→KV 相似性三分量（3037）→**读出协议条件性标注（3038/3039）**→**残差检验防居中基线与守卫零伪影（3040）**→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；每链重新 prefill；pre-hook with_kwargs 返回 (args,kwargs)；fp_inject 可移入 prefill 末位（直读协议）；**SVD 行空间基底取 Vt[:r].T 而非 U**（3040）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3040）
Ω-P2（3011-3040）：3011 门控=L3 KV；3018-3019 通用抑制场；3020 注入特异 944×；3021-3022 L3 联盟中继 82pct；3024 承重 0.657；3028 剂量凸增长；3031/3033 异质性=比值伪影（β 0.649）；3032 深峰=头集中 89pct；3034 头集合部分共享；3035 指纹竞争 logistic（特异 13.2×）；3036 曲率=操作点属性；3037 L3 KV=词身份主导+公共中继+语境调制（ratio 1.718）；3038 再入读出偏平（gain 0.868）；3039 直读复刻：曲率符号律崩溃 4/9=再入协议属性、特异 16×/奇主导/κ 协议鲁棒；3040 **情景分量提取：词身份子空间 98.3pct 能量、32/92 完全刻板；残差语境锁 p=0.012 且 L3 特异（L20 null）；无正向词残差码、同词偏离秩1反平行超匹配 null（was 同前缀 cos+1.0 确定性）→ L3 KV 三分量定版（公共中继+词身份+秩1情境残差）**。

## 下一步
- max=3040，下一个 3041（A 主选 **秩1情境轴解剖**——逐词偏离 SVD 谱+同前缀最小对受控设计，偏离符号/幅度 vs 上下文回归；B 语境锁定方向定位；C 阻尼场通道分解；D 跨模型复刻 KV 三分量+情景协议）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
