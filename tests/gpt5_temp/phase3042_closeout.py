# -*- coding: utf-8 -*-
"""Phase 3042 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite ->
HDMCC audit addendum."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3042'
     r'\omega_p39_style_field_probe_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
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
assert verdict == 'stylefield_global_identity_qwen', \
    verdict
assert res['anchor_all_ok'] is True
an = res['anchors']
assert an['a86_dup_prefill_bit'] == 0.0
assert an['a87_dup_all_bit'] == 0.0
assert an['a88_top2_ok'] is True
assert an['a88_maxdiff'] <= 0.15
assert an['a89_basis_orth_max'] <= 1e-5
assert an['a90_source_seals'] is True
assert an['a91_cross_phase_bit'] == 0.0
assert an['a91_matched'] == 8
assert an['a92_sit3_diff'] == 0.0
assert an['a92_sit20_diff'] == 0.0
dsp = res['displacements']
assert dsp['n'] == 24
assert dsp['degenerate_L3'] == 0
t1 = res['T1_global_field']
assert t1['n_pairs'] == 276
assert abs(t1['obs_med_abs_cos']
           - 0.2693297625254841) < 1e-12
assert abs(t1['null_med']
           - 0.05993130135299696) < 1e-12
assert t1['p_t1'] <= 1e-5
t1b = res['T1b_complement_field']
assert abs(t1b['obs_med_abs_cos']
           - 0.28099028324766717) < 1e-12
assert t1b['p_t1b'] <= 1e-5
t2 = res['T2_prefix_identity']
assert abs(t2['med_same']
           - 0.36100067099670197) < 1e-12
assert abs(t2['med_diff']
           - 0.24653887147664538) < 1e-12
assert abs(t2['d2']
           - 0.11446179952005658) < 1e-12
assert t2['p_t2'] <= 1e-5
t4 = res['T4_channel']
assert abs(t4['med_ew_L3']
           - 0.27266825537425143) < 1e-12
assert t4['complement_dominant'] is False
t3 = res['T3_dose']
assert t3['ci_slope'][0] < 0 and t3['ci_slope'][1] > 0
t5 = res['T5_layer20']
assert abs(t5['obs_t1']
           - 0.10942951807041752) < 1e-12
assert abs(t5['p_t2'] - 4e-05) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3042
           for m in led['measurements']):
    claim = (
        'Omega-P39 (plan 3042 A) - style-field probe, '
        'direct test of the HDMCC attachment claim '
        '"style prompt = global gravity field on '
        'attention routing": 8 body prompts (one '
        'logic target each) x 4 conditions (base + '
        'formal-style + Shakespearean-style + topic '
        'prefixes); Delta = V(cond) - V(base) at the '
        'SAME target occurrence, L3 kv7 V primary; '
        'channel basis B3/B20 reconstructed verbatim '
        'from the 3041 npz (anchor a92 SIT diff 0.0). '
        'Anchors: a86/a87 bit 0.0; a88 0.0453; a89 '
        '1.8e-15; a90 seals (5 phases); a91 base-'
        'condition targets vs 3037 npz 8/8 bit 0.0; '
        'a92 0.0.  T1 GLOBAL FIELD CONFIRMED: '
        'same-prefix cross-body displacement '
        'alignment med |cos| 0.2693 vs size-matched '
        'Gaussian null 0.0599 (4.5x), p < 1e-5 (0/20k '
        'draws exceed) - a prefix shifts the V write '
        'of DIFFERENT downstream body sentences in a '
        'SHARED direction.  T1b: complement-projected '
        'field 0.2810 vs 0.0672 (4.2x) - the field '
        'lives in the situational complement channel. '
        ' T2 PREFIX IDENTITY: same-prefix med signed '
        'cos 0.3610 vs diff-prefix 0.2465 (d2 '
        '+0.1145), exact label permutation p < 1e-5 - '
        'each prefix imprints its own direction ON '
        'TOP of the shared field.  T4 CHANNEL: med '
        'energy fraction in the word-identity '
        'subspace 0.2727 vs Gaussian expectation '
        '0.2031 - the displacement ALSO leaks into '
        'the identity channel above chance '
        '(complement_dominant FALSE), correcting the '
        'attachment claim that style leaves word '
        'identity untouched.  T3 DOSE: flat (slope '
        'CI [-0.098,+0.018] includes 0) - magnitude '
        'not explained by prefix length; topic '
        'prefix largest (med |Delta| 0.322; semantic-'
        'overlap hypothesis registered).  T5: L20 '
        'field present but weaker (0.109 vs 0.269; '
        'p_t2 4e-05) - L3-amplified, not '
        'L3-exclusive.  CONCLUSION: style field '
        'global + prefix-identity + identity-channel '
        'leak at the L3 KV write; attachment claim '
        'upgraded untested -> CONFIRMED WITH '
        'CORRECTIONS.  NEXT: field variance '
        'decomposition (shared vs prefix-specific '
        'components, content interaction), '
        'cross-lingual shared subspace, nested-'
        'subspace orthogonality, cross-model.')
    meas = {
        'meas_id': 'meas3042_omega_p39_style_field_'
                   'probe_qwen',
        'phase': 3042,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a86/a87/a91/a92 bit 0.0; a88 '
                   '0.0453; a89 1.8e-15; a90 seals; '
                   'T1 0.269 vs 0.060 p<1e-5; T1b '
                   '0.281 vs 0.067; T2 d2 +0.114 '
                   'p<1e-5; T4 ew 0.273>0.203; T5 '
                   'L20 0.109',
        'artifacts': {
            'result': 'phase3042/omega_p39_'
                      'style_field_probe_qwen/'
                      'result.json',
            'npz': 'phase3042/omega_p39_'
                   'style_field_probe_qwen/'
                   'omega_p39_style_field_probe_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run2 authoritative (13.0s); run1 '
                'crashed PRE-verdict (log format '
                'missing argument, no statistic '
                'observed); correction registered in '
                'PREREG',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 181
    l14['connects'].append({
        'meas_id': 'meas3042_omega_p39_style_field_'
                   'probe_qwen',
        'phase': 3042,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P39: style-field probe '
                        '- prefix displacements of '
                        'DIFFERENT body sentences '
                        'share a direction (med |cos| '
                        '0.269 vs Gaussian 0.060, '
                        '4.5x, p<1e-5) = GLOBAL FIELD '
                        'confirmed at L3 kv7 V write; '
                        'field lives in the situational '
                        'complement (T1b 0.281 vs '
                        '0.067) AND leaks into the '
                        'word-identity channel above '
                        'chance (ew 0.273 vs 0.203) - '
                        'identity NOT untouched; each '
                        'prefix adds its own direction '
                        '(d2 +0.114, p<1e-5); dose '
                        'flat in prefix length; L20 '
                        'weaker (0.109); HDMCC style-'
                        'field claim CONFIRMED WITH '
                        'CORRECTIONS; '
                        'stylefield_global_identity_'
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
if '## Phase 3042:' not in memo:
    sec = u'''## Phase 3042: Ω-P39 风格场探针——前缀位移跨句共享 4.5× 高斯（p<1e-5）= 全局引力场证实；场居情境补空间且泄漏词身份通道（ew 0.273>0.203）；前缀身份叠加其上（d2 +0.114） [%(created)s]

**判决：`stylefield_global_identity_qwen`**（run2 权威 13.0s；run1 判决前崩溃——log 格式串缺实参，无统计量观测，correction 已登记）

### 设计（3042 A 主选：情境码全局性检验，直接裁决 HDMCC 附件"全局引力场"）
8 基体句（3037 库 GEN 中各含恰一个 logic 目标词：so/because/therefore/however/while/yet/although/thus）× 4 条件（无前缀 / formal 风格 / Shakespearean 风格 / 主题前缀），组装句断言目标 token 恰出现一次；Δ = V(条件) − V(base) 于同一目标出现位置（L3 kv7 V 主判据 + L20 对照）；通道基 B3/B20 从 3041 npz verbatim 重建（a92：SIT 重算差 0.0）。T1 同前缀跨基体位移对齐（med |cos|）vs 尺寸匹配高斯 null（20k）；T1b 补空间投影版；T2 前缀身份（符号 cos，前缀标签精确置换 50k）；T3 剂量描述；T4 词身份子空间能量占比 vs 高斯期望 r3/HDIM。八锚：a86/a87/a91/a92 位级 0.0。

### 核心结果（重复三遍）
**① 全局引力场证实**：同前缀跨基体位移 med |cos| = **0.2693** vs 高斯 null **0.0599**（**4.5×**），p=**0.0**（20k 抽样零超出）——一个前缀把**不同**下游句子的 V 写入推向**共享方向**；附件"风格=全局引力场扭曲路由"在 L3 KV 写入层面主干成立。**② 场的通道定位**：补空间投影后 0.2810 vs 0.0672（4.2×）——场主要居于情境补空间；**但词身份通道能量占比 0.2727 > 高斯期望 0.2031**（complement_dominant=False）——位移同时泄漏进词身份通道，**修正附件"风格不触碰词身份/语义基底"**。**③ 前缀身份**：同前缀 med 符号 cos 0.3610 vs 异前缀 0.2465（d2=+0.1145，精确置换 p<1e-5）——每个前缀在共享场之上叠加自己的方向。**④ 剂量平坦**：|Δ|~前缀长度 slope CI[−0.098,+0.018] 含 0；主题前缀最大（0.322）——与 body0 语义重叠（weather×weather），内容交互假设入册。**⑤ L20 场弱一半**（0.109 vs 0.269，p=4e-05）——L3 放大、非 L3 独占。

### 机制链更新
L3 KV 写入第五分量候选：**前缀场**（跨句共享方向 + 前缀特异方向叠加；居情境补空间为主、泄漏词身份；剂量与长度无关、与内容相似性候选相关）。附件 HDMCC 风格主张从"未检验"升级为"**证实带修正**"：全局场 ✅、场在 KV 写入层 ✅；修正：非纯路由偏置（词身份通道同被推移）、非"不改语义基底"。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3042/omega_p39_style_field_probe_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3043 菜单**——A（主选）**场方差分解**：Δ 的共享分量 vs 前缀特异分量占比（逐前缀对齐矩阵 + 方差分解），内容交互检验（主题前缀×语义相关 body 的 |Δ| 增益）；B 跨语言共享子空间（EN/ZH 同词深层对齐）；C 嵌套子空间正交性量化（apple/fruit/food）；D 跨模型复刻（KV 五分量 + 风格场协议上 DS7B/GLM4）。
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

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 四、3042 增补' not in aud:
    add = u'''

---

## 四、3042 增补：风格场探针结果（Omega-P39，判决 stylefield_global_identity_qwen）

审计第三节"未检验但可检验"之 A 项已完成（3042 A 主选，run2 权威，八锚全过含 a91/a92 位级 0.0）：

1. **"全局引力场" 主干证实**：同前缀跨句位移 med |cos| 0.2693 vs 高斯 null 0.0599（4.5×），p<1e-5——前缀把不同下游句子的 L3 kv7 V 写入推向共享方向。附件该主张从 ❓未检验 升级为 ✅（KV 写入层面）。
2. **修正一（通道泄漏）**：位移在词身份子空间能量占比 0.2727 > 高斯期望 0.2031——风格场**同时推移词身份通道**，附件"不改变词身份/语义基底、只扭曲路由"过于干净。
3. **修正二（前缀身份）**：同前缀 vs 异前缀符号 cos 差 +0.114（p<1e-5）——场 = 共享分量 + 前缀特异分量叠加，非单一均匀场。
4. **修正三（剂量平坦）**：|Δ| 与前缀长度无关（CI 含 0）；主题前缀（weather×weather 语义重叠）位移最大——"引力强度"更似内容相似性驱动而非长度驱动。

据此 3043 菜单 A 定为场方差分解（共享 vs 前缀特异分量 + 内容交互检验）。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3042' not in prev:
    line = ('- Phase 3042 Omega-P39 style-field '
            'probe: verdict '
            'stylefield_global_identity_qwen (run2 '
            '13.0s; run1 crash pre-verdict log-format '
            'arg; 8 anchors ok, a91/a92 bit 0.0); '
            'prefix displacements across DIFFERENT '
            'body sentences share direction (med '
            '|cos| 0.269 vs Gaussian 0.060, 4.5x, '
            'p<1e-5) = HDMCC global style field '
            'CONFIRMED at L3 kv7 V write; field in '
            'situational complement (0.281 vs 0.067) '
            'AND leaks into word-identity channel '
            '(ew 0.273 > 0.203) - identity NOT '
            'untouched; prefix identity d2 +0.114 '
            'p<1e-5; dose flat; L20 weaker 0.109; '
            'audit doc addendum written; ledger '
            '181/L14 149.\n')
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
4. MEMO 占位符一律 %(key)s 风格；文本内裸百分号写 %%（3033 教训）；log 格式串占位符数=实参数（3042 run1 教训）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08；bit 级仅限同文件链/上游全精度；跨相位 npz 锚均 0.0；剂量两端点锚死。
- 干预锚族：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；手工 norm+lm_head 重算+0.15 门；源封印；跨相位 npz 位级锚（3040 a78 25/25；3041 a85 全库链锚；3042 a91 基条件 8/8 + **a92 通道基从上相位 npz verbatim 重建（SIT 重算差 0.0）——通道定义跨相位恒等**）。重复臂方向必须与原臂同对象（3039）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；margin n≳40（探索性标注）。
- null 门不得设在接收干预的量上（3035）；曲率检验排除饱和区（3036）；跨相位锚定前核对读出协议量纲（3037）。
- 余弦守卫零值→退化行剔除（3040）；小 n 组内去均值→构造匹配置换（3040/3041）；**中位数签名与谱结构可分歧——几何标签命名前必查谱（3041）**；对齐用 |cos|、身份用符号 cos（3042）。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→曲率符号报操作点→KV 相似性三分量（3037）→读出协议条件性标注（3038/3039）→残差检验防居中基线与守卫零伪影（3040）→谱水平复核几何标签（3041）→**位移场双 null（高斯对齐+标签置换身份）与通道泄漏检查（3042）**→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；pre-hook with_kwargs 返回 (args,kwargs)；SVD 行空间基底取 Vt[:r].T 而非 U（3040）；gram 特征值算 PR/主轴（3041）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁脚本自身先保证引号合法；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3042）
Ω-P2（3011-3042）：3011 门控=L3 KV；3018-3019 通用抑制场；3020 注入特异 944×；3021-3022 L3 联盟中继 82pct；3024 承重 0.657；3028 剂量凸增长；3031/3033 异质性=比值伪影；3032 深峰=头集中 89pct；3035 指纹 logistic（特异 13.2×）；3036 曲率=操作点属性+指纹非正交；3037 KV=词身份主导+中继+语境调制；3038/3039 协议分层；3040 情景分量（词身份 98.3pct+32/92 刻板+语境锁 p=0.012）；3041 谱推翻单词秩1轴（PR 1.94 vs 1.82；yet 个案 PR=1.0）；同前缀复现 cos≈+1；跨词弱共享轴；3042 **风格场探针：前缀位移跨句共享 4.5× 高斯（p<1e-5）=全局引力场证实；居情境补空间且泄漏词身份通道（ew 0.273>0.203）；前缀身份叠加（d2 +0.114）；剂量平坦；L20 弱一半 → L3 KV 五分量候选（中继+词身份+多维情境码+位置梯度+前缀场）**。HDMCC 附件审计：三大定律兼容；"正交嵌套子空间" ❌；风格场 ✅带修正（审计文档 hdmcc_knowledge_map_review_20260921.md 含 3042 增补）。

## 下一步
- max=3042，下一个 3043（A 主选 **场方差分解**——Δ 共享 vs 前缀特异分量占比+内容交互检验（weather×weather 高 |Δ|）；B 跨语言共享子空间 EN/ZH；C 嵌套子空间正交性量化 apple/fruit/food；D 跨模型复刻五分量+风格场）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
