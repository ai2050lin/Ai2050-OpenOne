# -*- coding: utf-8 -*-
"""Phase 3043 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log ->
MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3043'
     r'\omega_p40_field_variance_qwen')
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
assert verdict == 'fieldvar_prefix_plus_body_qwen', \
    verdict
assert res['anchor_all_ok'] is True
an = res['anchors']
assert an['a93_dup_prefill_bit'] == 0.0
assert an['a94_dup_all_bit'] == 0.0
assert an['a95_top2_ok'] is True
assert an['a95_maxdiff'] <= 0.15
assert an['a96_source_seals'] is True
assert an['a97_cross_phase_bit'] == 0.0
assert an['a97_matched'] == 8
assert an['a98_d3_bit'] == 0.0
assert an['a98_d20_bit'] == 0.0
assert an['a99_sit3_diff'] == 0.0
assert an['a99_sit20_diff'] == 0.0
t1 = res['T1_variance']
assert abs(t1['f_prefix']
           - 0.14808504909678993) < 1e-12
assert t1['p_prefix'] <= 1e-5
assert abs(t1['f_body']
           - 0.643312653533292) < 1e-12
assert t1['p_body'] <= 1e-5
assert abs(t1['f_resid']
           - 0.20860229736991798) < 1e-12
t2 = res['T2_content_outlier']
assert abs(t2['obs_norm']
           - 0.3155283913732225) < 1e-12
assert t2['rank'] == 9
assert abs(t2['p_t2'] - 0.62708) < 1e-12
t3 = res['T3_field_commonality']
assert abs(t3['obs_med_abs_cos']
           - 0.6469309674738258) < 1e-12
assert abs(t3['null_med']
           - 0.05987410862399565) < 1e-12
assert t3['p_t3'] <= 1e-5
t4 = res['T4_generalization']
assert t4['n_pairs'] == 12
assert abs(t4['obs_med_abs_cos']
           - 0.683332767418094) < 1e-12
assert t4['p_t4'] <= 1e-5
t5 = res['T5_layer20']
assert abs(t5['f_prefix']
           - 0.14516565972901058) < 1e-12
assert abs(t5['f_body']
           - 0.6087598084595457) < 1e-12
assert abs(t5['t4_obs']
           - 0.4371931608732015) < 1e-12
assert t5['t4_p'] <= 1e-5
fl = res['flags']
assert fl['prefixfx'] and fl['bodyfx']
assert fl['contenthit'] is False
assert fl['fieldcommon'] and fl['generalizes']

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3043
           for m in led['measurements']):
    claim = (
        'Omega-P40 (plan 3043 A) - field variance '
        'decomposition of the 3042 prefix field: 8 '
        'old bodies x 4 conditions verbatim 3042 '
        '(a98 displacement-chain anchor: our D3/D20 '
        'vs 3042 npz D3/D20 BIT 0.0) + 4 NEW '
        'out-of-bank bodies x 4 conditions (48 '
        'prompts); a97 vs 3037 npz 8/8 bit 0.0; a99 '
        'channel basis vs 3041 npz SIT diff 0.0.  '
        'T1 two-way additive decomposition: '
        'f_prefix 0.1481 (p<2e-5, exact label '
        'permutation 50k), f_body 0.6433 (p<2e-5), '
        'residual 0.2086 - CONTENT INTERACTION '
        'DOMINATES the displacement variance (64pct '
        'body main effect vs 15pct prefix main '
        'effect); L20 same (0.145/0.609).  T2 '
        'semantic-overlap outlier REFUTED: the '
        '(topic-prefix, weather-body) cell norm '
        '0.3155 is rank 9/24, within-block '
        'body-label permutation p 0.627 - the '
        '3042 topic-prefix largeness was block-wide '
        '(c3 rows generally larger, max 0.402 at '
        'c3_b3), not weather-specific.  T3 FIELD '
        'AXIS COMMONALITY: the three prefix-mean '
        'directions alpha_c align at med |cos| '
        '0.6469 vs Gaussian 0.0599 (10.8x, p<1e-5) '
        '- ONE dominant shared field axis plus '
        'per-prefix modulation (reconciles with '
        '3042 T2 identity: means align, individual '
        'displacements carry prefix identity).  T4 '
        'OUT-OF-BANK TRANSFER: 4 NEW body sentences '
        'x 3 prefixes align with the old-body field '
        'axes at med |cos| 0.6833 vs 0.0597 (11.4x, '
        'p<1e-5); L20 0.4372 - the field is a '
        'portable prefix property, not a bank '
        'artifact.  CONCLUSION: displacement field '
        '= body-specific content component (64pct) '
        '+ shared portable prefix axis (15pct, '
        'commonality 0.65, transfers to new '
        'sentences) + prefix identity + residual '
        'interaction; HDMCC global-field claim now '
        'quantitatively structured.  NEXT: causal '
        'injection along the shared field axis '
        '(connect to the 3035 fingerprint readout), '
        'cross-lingual shared subspace, nested-'
        'subspace orthogonality, cross-model.')
    meas = {
        'meas_id': 'meas3043_omega_p40_field_'
                   'variance_qwen',
        'phase': 3043,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a93/a94/a97/a98/a99 bit 0.0; a95 '
                   '0.0453; a96 seals; T1 f_prefix '
                   '0.148 p<2e-5 / f_body 0.643 '
                   'p<2e-5; T2 rank 9/24 p 0.627; T3 '
                   '0.647 vs 0.060; T4 0.683 vs 0.060 '
                   '(12 new pairs); T5 L20 same',
        'artifacts': {
            'result': 'phase3043/omega_p40_'
                      'field_variance_qwen/'
                      'result.json',
            'npz': 'phase3043/omega_p40_'
                   'field_variance_qwen/'
                   'omega_p40_field_variance_qwen'
                   '.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run3 authoritative (33.2s); run1 '
                'crash pre-verdict (a98 npz-key bug); '
                'run2 crash mid-T2 (empty cell under '
                'prefix-label permutation, T2 null '
                'redefined to within-block body-label '
                'permutation); T1 observed in run2, '
                'design unchanged; corrections '
                'registered in PREREG',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 182
    l14['connects'].append({
        'meas_id': 'meas3043_omega_p40_field_'
                   'variance_qwen',
        'phase': 3043,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P40: field variance '
                        'decomposition - body main '
                        'effect DOMINATES (64pct vs '
                        'prefix 15pct, both p<2e-5); '
                        'semantic-overlap outlier '
                        'REFUTED (rank 9/24, p 0.63); '
                        'prefix-mean directions share '
                        'ONE axis (0.647 vs 0.060, '
                        '10.8x); field TRANSFERS to 4 '
                        'new out-of-bank bodies (0.683 '
                        'vs 0.060, 11.4x); field = '
                        'portable prefix axis + body '
                        'content modulation; '
                        'fieldvar_prefix_plus_body_'
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
if '## Phase 3043:' not in memo:
    sec = u'''## Phase 3043: Ω-P40 场方差分解——体主效应主导（64pct vs 前缀 15pct）；语义重叠特异 refuted（rank 9/24 p=0.63）；场轴共性 0.647（10.8×）；场外推 4 新句 0.683（11.4×） [%(created)s]

**判决：`fieldvar_prefix_plus_body_qwen`**（run3 权威 33.2s；run1 a98 npz 键 bug 判决前崩、run2 T2 空格崩溃→T2 零假设改块内 body 置换，均登记；九锚全过含 a98 位移链锚 D3/D20 位级 0.0）

### 设计（3043 A 主选：场方差分解）
8 旧基体 × 4 条件 verbatim 3042（a98：我们的 D3/D20 vs 3042 npz **位级 0.0**——派生量链锚）+ **4 个库外新基体句** × 4 条件（48 prompts）。T1 双向加性方差分解（截距+前缀哑元+基体哑元 OLS，边际 SS，标签精确置换 50k×2）；T2 语义重叠格（topic×weather）块内置换离群检验；T3 三前缀均值方向 α_c 两两 |cos| vs 高斯；T4 新句位移 vs α_c 对齐（库外迁移）；T5 L20 对照。a97 vs 3037 npz 8/8、a99 vs 3041 npz SIT 差 0.0。

### 核心结果（重复三遍）
**① 体主效应主导方差**：f_prefix=**0.1481**（p<2e-5）、f_body=**0.6433**（p<2e-5）、残差 0.2086——位移场方差 64pct 来自基体内容（内容交互），前缀共享场仅 15pct；L20 同构（0.145/0.609）。**② 语义重叠特异 refuted**：(topic 前缀 × weather 基体) 格范数 0.3155 = rank **9/24**，块内置换 p=**0.627**——3042 观察到的 topic 前缀大位移是**块内普遍**（c3 行普遍偏大，最大 0.402 在 c3_b3），非 weather 特异；3042 的"内容交互候选"按预注册检验判负。**③ 场轴共性极强**：三前缀均值方向两两 med |cos|=**0.6469** vs 高斯 0.0599（**10.8×**，p<1e-5）——存在**单一主导共享场轴**+逐前缀调制（与 3042 T2 前缀身份调和：均值对齐、个体位移带前缀身份）。**④ 库外迁移成立**：4 个全新基体句 × 3 前缀的位移与旧库场轴 med |cos|=**0.6833** vs 0.0597（**11.4×**，p<1e-5）；L20 0.4372——场是**可移植的前缀属性**，非库伪影。

### 机制链更新
前缀场定量化：**位移 = 基体内容分量（64pct）+ 共享可移植前缀轴（15pct，轴共性 0.65，跨句迁移 0.68）+ 前缀身份调制 + 残差交互**。HDMCC"全局引力场"获得结构化量化：场存在、有公共轴、可迁移，但强度被基体内容主导调制——"引力场"更像"地形调制下的主导风向"。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3043/omega_p40_field_variance_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3044 菜单**——A（主选）**场轴因果注入**：沿共享场轴 ±m·ᾱ 定向干预 L3 KV 写入，测下游 logits/指纹竞争偏移与剂量响应（连接 3035 fp_inject 机械，把前缀场从相关推到因果）；B 跨语言共享子空间（EN/ZH 同词深层对齐）；C 嵌套子空间正交性量化（apple/fruit/food）；D 跨模型复刻（五分量+场协议上 DS7B/GLM4）。
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
if '## 五、3043 增补' not in aud:
    add = u'''

---

## 五、3043 增补：场方差分解结果（Omega-P40，判决 fieldvar_prefix_plus_body_qwen）

对第四节风格场的结构化量化（run3 权威，九锚全过，a98 位移链锚 vs 3042 npz 位级 0.0）：

1. **方差结构**：体主效应（内容交互）主导位移方差（64pct），前缀主效应（共享场）15pct，残差 21pct——"全局引力场"存在但强度被基体内容主导调制。
2. **语义重叠特异 refuted**：weather×weather 格 rank 9/24（p=0.63），3042 的 topic 前缀大位移是块内普遍现象——附件式"概念-概念特异性共振"叙事在本规模上无证据。
3. **场轴共性 0.647**（10.8× 高斯）：单一主导共享场轴 + 逐前缀调制。
4. **库外迁移 0.683**（11.4× 高斯，4 个全新句子）：场是可移植前缀属性——这是"全局"主张的最强证据形态。

据此 3044 菜单 A 定为场轴因果注入（沿 ᾱ ±m 干预 → 读出指纹竞争偏移，连接 3035 机械）。
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
if 'Phase 3043' not in prev:
    line = ('- Phase 3043 Omega-P40 field variance '
            'decomposition: verdict '
            'fieldvar_prefix_plus_body_qwen (run3 '
            '33.2s; run1 a98 npz-key bug pre-verdict, '
            'run2 empty-cell mid-T2 -> within-block '
            'body permutation; 9 anchors ok, a98 '
            'displacement chain vs 3042 npz bit 0.0); '
            'f_prefix 0.148 vs f_body 0.643 (content '
            'dominates); weather-overlap outlier '
            'REFUTED (rank 9/24 p 0.63); field-axis '
            'commonality 0.647 (10.8x); out-of-bank '
            'transfer to 4 new bodies 0.683 (11.4x); '
            'audit addendum written; ledger 182/L14 '
            '150.\n')
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
4. MEMO 占位符一律 %(key)s 风格；文本内裸百分号写 %%（3033）；log 格式串占位符数=实参数（3042 run1）；**置换格须保证存在（空格→块内置换方案，3043 run2）**；**锚引用 npz 键前先探针确认键名（3043 run1）**。

## 标准锚与精度
- a1 dirs 重建 2.17e-08；bit 级仅限同文件链/上游全精度；跨相位 npz 锚均 0.0；剂量两端点锚死。
- 干预锚族：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；手工 norm+lm_head 重算+0.15 门；源封印；跨相位锚（3040 a78 25/25；3041 a85 全库链锚；3042 a91 8/8+a92 通道基重建；**3043 a98 位移链锚：派生量 D3/D20 vs 上相位 npz 位级 0.0**）。重复臂方向必须与原臂同对象（3039）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；margin n≳40（探索性标注）。
- null 门不设在接收干预的量上（3035）；曲率排除饱和区（3036）；跨相位锚前核对量纲（3037）；余弦守卫零→退化行剔除（3040）；小 n 组内去均值→构造匹配置换（3040/3041）；几何标签命名前查谱（3041）；对齐 |cos|/身份符号 cos（3042）；**方差分解用边际 SS（全拟合−缩减拟合）+标签精确置换（3043）**。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→曲率符号报操作点→KV 三分量（3037）→读出协议条件性（3038/3039）→残差防居中/守卫零（3040）→谱水平复核（3041）→位移场双 null+通道泄漏（3042）→**方差分解分主效应+离群格块内置换（3043）**→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；pre-hook with_kwargs 返回 (args,kwargs)；SVD 行空间基底 Vt[:r].T（3040）；gram 特征值算 PR/主轴（3041）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁脚本引号先自查；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3043）
Ω-P2（3011-3043）：3011 门控=L3 KV；3018-3019 通用抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3031-3036 异质性伪影/头集中/指纹 logistic/曲率操作点/非正交；3037 KV 三分量；3038/3039 协议分层；3040 情景分量（98.3pct+刻板）；3041 谱推翻秩1标签（PR 1.94）；同前缀复现 cos≈+1；3042 **风格场探针：前缀位移跨句共享 4.5×（p<1e-5）=全局场证实；泄漏词身份通道；前缀身份叠加**；3043 **场方差分解：体主效应 64pct 主导、前缀 15pct；weather 重叠特异 refuted（rank 9/24）；场轴共性 0.647（10.8×）；库外迁移 0.683（11.4×）→ 前缀场=可移植公共轴+内容调制**。HDMCC 审计：三大定律兼容；正交嵌套 ❌；风格场 ✅带修正（审计文档含四/五节增补）。

## 下一步
- max=3043，下一个 3044（A 主选 **场轴因果注入**——沿 ᾱ ±m 干预 L3 KV 写入测下游指纹竞争偏移与剂量响应，连接 3035；B 跨语言共享子空间 EN/ZH；C 嵌套子空间正交性 apple/fruit/food；D 跨模型复刻）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
