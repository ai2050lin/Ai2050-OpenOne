# -*- coding: utf-8 -*-
"""Phase 3052 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3052'
     r'\omega_p49_kvhead_identity_qwen')
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
verdict = res['verdict']
assert verdict == 'kvhead_generic_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a150_recapture_diff'] == 0.0
assert an['a151_diag_diffs'] == {
    'h7': 0.0, 'h6': 0.0, 'h1': 0.0}
assert an['a152_sham_diff'] == 0.0
assert an['a153_fail'] == 0
assert an['a153_checked'] == 24
assert abs(an['a154_max_dlg']
           - 378.46999805889294) < 1e-6
t7 = st['T2_transfer']['h7']
assert abs(t7['med_A_diag']
           - 0.6804954382830863) < 1e-12
assert abs(t7['med_A_off']
           - 0.6724197789995571) < 1e-12
assert abs(t7['med_B_off']
           - 0.6598334689080656) < 1e-12
assert abs(t7['med_D_off']
           - -0.015035495821663136) < 1e-12
assert abs(t7['p_perm']
           - 0.8465767116441779) < 1e-12
assert abs(t7['ctgt_off_med']
           - 0.26831689973448103) < 1e-12
t6 = st['T2_transfer']['h6']
assert abs(t6['med_A_off']
           - 0.5841666189698019) < 1e-12
assert abs(t6['med_B_off']
           - 0.6380329480370981) < 1e-12
assert abs(t6['med_D_off']
           - 0.020605053153434916) < 1e-12
assert abs(t6['p_perm']
           - 0.04697651174412794) < 1e-12
t1 = st['T2_transfer']['h1']
assert abs(t1['med_A_off']
           - -0.16947309894670673) < 1e-12
assert abs(t1['med_B_off']
           - -0.14844517699595547) < 1e-12
t3 = st['T3_static']
assert abs(t3['med_cos_ov_per_head'][7]
           - 0.585428250672426) < 1e-12
assert abs(t3['med_cos_ov_per_head'][6]
           - 0.15335301249305677) < 1e-12
assert abs(t3['rank_corr_ov_vs_causal']
           - 0.3333333333333334) < 1e-12
assert abs(t3['med_cos_coalition_per_head'][7]
           - -0.018532553449200313) < 1e-12
assert abs(max(abs(x) for x in
               t3['cos_coal_h7_per_tag'])
           - 0.04827514799906157) < 1e-12
assert abs(min(t3['ratio_coal_h7_per_tag'])
           - 0.9213842570374168) < 1e-12
assert abs(max(t3['ratio_coal_h7_per_tag'])
           - 1.0113496260569281) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3052
           for m in led['measurements']):
    claim = (
        'Omega-P49 (plan 3052 A) - KV head '
        'identity (h7/h6), run1 fp32 '
        'authoritative (321.5s, all anchors '
        'passed first try). Capture bank '
        'LOADED from the z48 npz (a150 '
        're-capture bit 0.0); chain anchors: '
        'a151 diagonal repro vs the z51 COS_H '
        'diff 0.0 for h7/h6/h1; a152 sham bit '
        '0.0; a153 integrity fails 0 (24 '
        'checked). RESULTS: (1) T2 cross-pair '
        'transfer matrices (insert pair j '
        'head-h fields into base k, A = cos '
        'with destination target t_k, B = cos '
        'with source target t_j): h7 A_off '
        '0.6724 ~= diag 0.6805 - ANY pair '
        'fields work, aligned with the '
        'DESTINATION target; D = B - A med '
        '-0.0150, p_perm 0.847 -> h7 is a '
        'GENERIC slot (content-'
        'interchangeable); the direction is '
        'assembled from the destination '
        'context, not carried by the values. '
        '(2) h6 dissociates: D med +0.0206, '
        'p_perm 0.047, B_off 0.6380 > A_off '
        '0.5842 -> h6 is a WEAK content '
        'carrier; h1 negative, no transfer '
        '(A_off -0.1695). Off-diag target-'
        'similarity baseline ctgt med 0.2683; '
        'corr(A, ctgt) 0.475 / corr(B, ctgt) '
        '0.496 for h7. (3) T3 static: h7 '
        'uniform-attention OV write aligns '
        'with t at 0.5854 (only strong head; '
        'h6 causal 0.6367 but OV only 0.1534 '
        '- h6 causal effect does NOT flow '
        'through its OV write, pure K '
        'routing); rank corr(OV med, causal '
        'med) 0.333; top written tokens show '
        'NO lexical specialization '
        '(high-frequency / junk tokens). '
        '(4) 3022 coalition linkage: '
        'cos(w32_coalition, ubar_h) med per '
        'head -0.019..0.015 (|cos| <= 0.048 '
        'per tag), w32/w_all ratio 0.92-1.01 '
        '- NO linkage between the L35 carrier '
        'heads and the L3 MLP relay '
        'coalition; the injection chain and '
        'the natural prefix chain are '
        'write-orthogonal. Verdict '
        'kvhead_generic_qwen (h7 primary).')
    meas = {
        'meas_id': 'meas3052_omega_p49_kvhead_'
                   'identity_qwen',
        'phase': 3052,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a150 re-capture bit 0.0 vs '
                   'z48 + TT diff 0.0; a151 diag '
                   'repro diff 0.0 vs z51 COS_H '
                   '(h7/h6/h1); a152 sham bit '
                   '0.0; a153 integrity fails 0 '
                   '(24 checked); a154 max dlg '
                   '378.5',
        'artifacts': {
            'result': 'phase3052/omega_p49_'
                      'kvhead_identity_qwen/'
                      'result.json',
            'npz': 'phase3052/omega_p49_'
                   'kvhead_identity_qwen/'
                   'omega_p49_kvhead_identity_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (321.5s, '
                'fp32); 24x24 cross-pair matrices '
                'for h7/h6/h1 (1728 forwards); '
                'paired sign-flip permutation '
                'R=2000 on the off-diag D = B - A; '
                '3022 linkage via w32 = '
                'down_proj[:, top32] @ s_relay',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 191
    l14['connects'].append({
        'meas_id': 'meas3052_omega_p49_kvhead_'
                   'identity_qwen',
        'phase': 3052,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P49: head '
                        'identity - h7 GENERIC '
                        'slot (any pair fields '
                        'work, A_off 0.6724 ~= '
                        'diag 0.6805, aligned '
                        'with the DESTINATION '
                        'target, p_perm 0.847); '
                        'h6 weak content carrier '
                        '(B>A, p 0.047); no '
                        'lexical specialization '
                        'in the OV write; NO '
                        'linkage with the 3022 '
                        'L3 relay coalition '
                        '(|cos| <= 0.048) - '
                        'injection chain and '
                        'natural prefix chain '
                        'write-orthogonal '
                        '(kvhead_generic_qwen)'})
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
if '## Phase 3052:' not in memo:
    sec = u'''## Phase 3052: Ω-P49 KV 头身份解剖——h7 通用门槽 + h6 弱内容载体 + 与 L3 联盟写入正交（kvhead_generic_qwen） [%(created)s]

**判决：`kvhead_generic_qwen`**（run1 fp32 权威 321.5s，锚核心一次全过、无崩溃）。捕获库复用 z48 npz（a150 重捕获 bit 0.0 + TT diff 0.0）；链锚 a151 对角复现 z51 COS_H[h] diff 0.0（h7/h6/h1）；a152 sham bit 0.0；a153 完整性 24 前向全过。

### 设计
跨对迁移矩阵（主检验）：h7/h6/h1 三头 × 24×24 全有序对——把 pair j 的头 h 场（rotK_j+srcV_j，头块复刻 3051 T3 协议）插入 base k 的 L35×FRONT 行，A[k,j]=cos(r,t_k)（目的对齐）、B[k,j]=cos(r,t_j)（源对齐）；off-diag 配对差 D=B−A 符号翻转置换（R=2000 单侧）。T3 静态：均匀注意力 OV 写出 u_h=M_h·v̄_h → lm_head 词表空间对齐 + 3022 联盟跨链比对（w32_τ=Σ_top32 s·dh_j）。

### 核心结果（重复三遍）
**① h7 是通用门槽（内容可互换）**：A_off=**0.6724 ≈ diag 0.6805**——插入**任意** pair 的 h7 场都产生与**目的对**目标方向 0.67 的对齐；D=−0.0150，p_perm=**0.847** → 方向不由插入值承载，而由**目的语境下游组装**（与 3027/3028 读出=上下文属性闭环）。**② h6 解离为弱内容载体**：D=+0.0206，p_perm=**0.047**，B_off 0.6380 > A_off 0.5842——h6 的值携带弱源特异内容；h1 负头无迁移（A_off −0.1695）。双头双身份：**h7=门，h6=弱内容**。**③ OV 写出无词表语义身份**：h7 OV-目标对齐 0.5854（唯一强头；h6 因果 0.6367 但 OV 仅 0.1534 → h6 效应纯 K 路由不过 OV）；rank corr(OV, 因果)=0.333；h7/h6/h1 top 写出 token 全是高频/杂 token——无词级专化。**④ 与 3022 联盟写入正交**：cos(w32_联盟, ū_h) 每头 med −0.019..0.015（逐 tag |cos|≤0.048），w32/w_all 比 0.92-1.01——**L35 承载头与 L3 MLP 中继联盟无写入方向关联，注入链与自然前缀链是两条独立写通道**。

### 机制链定版
**KV 头身份完成：h7=通用门槽（KV 块内容可任意替换，方向由目的语境组装）+ h6=弱内容载体（纯 K 路由）+ 无词表身份 + 与 L3 注入联盟写入正交**。KV 载荷链收口：体位均匀（3049）→层维末层（3050）→通道维 K 场双头（3051）→**头身份：门而非载体（3052）**。承载头的作用是"开门"（使前缀效应通路成立），方向信息在目的语境+下游读出侧组装——"哪个头"与"什么方向"完全解耦。

### 方法论入册
- **跨对迁移矩阵纪律（3052）**：A/B 双目标对齐矩阵 + off-diag 配对符号翻转置换是"内容 vs 通用槽"的标准判别；对角即上游相位协议 bit 链锚。
- **门-载体判别（3052）**：A_off≈diag 且 p_perm 不显著 → 通用门；B>A 且 p<0.05 → 内容载体。两判据不可混用单侧叙述。
- **跨链正交检验（3052）**：注入链联盟写入方向（s_relay·down_proj 列重构）与自然链头 OV 写出的 cos 是跨机制链关联的标准零检验；|cos|<0.05 即正交。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3052/omega_p49_kvhead_identity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3053 菜单**——A（主选）**门控源解剖**：h7 通用门的效应来源分解——K 路由重分布 vs V 写入的跨对迁移分拆（K-only 矩阵预期同样通用）+ L35 MLP/读出侧组装定位（门开后方向从残流何处汇入）；B V 负成分机制（V-only −0.34）；C h6 弱内容载体深挖（body 特异 vs prefix 特异）；D 跨模型 DS7B 复刻全链。
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
if '## 十四、3052 增补' not in aud:
    add = u'''
    
---

## 十四、3052 增补：KV 头身份——h7 通用门槽、h6 弱内容载体、注入链与自然链写入正交（Omega-P49，判决 kvhead_generic_qwen）

1. **h7 = 通用门槽**：跨对迁移 A_off 0.6724 ≈ 对角 0.6805（任意 pair 场皆可、与目的对目标对齐，p_perm 0.847）——承载头的内容可任意替换，方向由目的语境下游组装；"哪个头"与"什么方向"完全解耦，是 3027"读出=上下文属性"在通道维的落地。
2. **h6 解离**：B>A 显著（p 0.047）但效应纯 K 路由（OV 对齐仅 0.1534）——弱内容载体；h7 OV 对齐 0.5854 但内容可互换——OV 写出对齐与内容承载亦是独立自由度。无词级专化（top token 全为高频/杂 token）。
3. **跨链正交**：L35 承载头 OV 写出与 3022 L3 中继联盟写入方向 |cos|≤0.048、w32/w_all≈1——注入实验的 L3 MLP 联盟与自然前缀效应的 L35 K 场是**两条独立写通道**；HDMCC 图谱中"注入可达"≠"自然承重"，两图必须分开画。
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
if 'Phase 3052' not in prev:
    line = ('- Phase 3052 Omega-P49 KV head '
            'identity: verdict kvhead_generic_'
            'qwen (run1 fp32 321.5s, all anchors '
            'passed first try). RESULTS: h7 '
            'GENERIC slot (cross-pair A_off '
            '0.6724 ~= diag 0.6805, aligned with '
            'DESTINATION target, D -0.015 p_perm '
            '0.847) - direction assembled from '
            'destination context; h6 weak '
            'content carrier (B>A p 0.047, but '
            'OV alignment only 0.1534 = pure K '
            'routing); h7 OV-target 0.5854 only '
            'strong head; no lexical '
            'specialization; NO linkage with '
            '3022 L3 relay coalition (|cos| '
            '<= 0.048, w32/w_all ~ 1) - '
            'injection chain and natural prefix '
            'chain write-orthogonal. Audit '
            'addendum 14; ledger 191/L14 159.\n')
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
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧产物；负结果/锚失败/崩溃如实登记；verdict 单分支赋值。
4. 统计量纪律：obs 与 null 同量纲同范围；null 限制在 exact 场同一行集（3050）；loo 全谱必报、禁可加归因（3051）；跨对迁移 A/B 双目标+配对符号翻转（3052）。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 捕获库跨相位复用：上游 npz+抽样重捕获 bit 锚+派生统计量逐对复现锚。
- 二维输出索引纪律（3050）：hook out[0] 剥 batch 维后 lm_head 输出 (n,V)——末 token 是 lg[-1]；末层 vs LG bit 锚必须在统计前抓退化。
- transformers 5.14（3050）：decoder layer 返回裸 tensor；output_hidden_states=True 不可信——层捕获一律自建 hook。
- 头块切片纪律（3051）：1024=8×128 头主序；头级替换=基场 copy+头块覆写；loo 把替换块外头块写回基场。

## 统计判据纪律
- 判据可达性先检；构造匹配置换；maxT；镜像必配。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯（3045-3048）→载荷定位（3049 体位均匀）→层定位（3050 末层 L35）→通道定位（3051 K 场 95.4pct、头 7/6、loo 非可加）→**头身份（3052：跨对迁移 A/B——h7 通用门槽 A_off≈diag、h6 弱内容载体、OV 对齐与承载独立、与 L3 联盟写入正交 |cos|≤0.048）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj 输出 (1,s,1024)；RoPE NeoX 配对、attention_scaling==1；fp32 logit 级测量。
- 头 h 的 OV 写出：M_h=Σ₄ o_proj 列组（4 个 Q 头块求和）@v̄；3022 联盟写入方向=s_relay·down_proj[:,j] 重构。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3052）
Ω-P2（3011-3052）：3011 门控=L3 KV；3018-3019 抑制场；3020 注入特异；3021-3024 联盟中继/承重；3045-3048 KV 五级阶梯；3049 载荷定位（体位均匀）；3050 层定位（末层 L35）；3051 通道定位（K 场+h7/h6）；**3052 头身份：h7 通用门槽+h6 弱内容载体+注入链/自然链写入正交（kvhead_generic_qwen）**。

## 下一步
- max=3052，下一个 3053（A 主选 **门控源解剖**——h7 门效应 K 路由 vs V 写入跨对分拆+L35 MLP/读出侧组装定位；B V 负成分机制；C h6 弱内容深挖；D 跨模型 DS7B 复刻）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
