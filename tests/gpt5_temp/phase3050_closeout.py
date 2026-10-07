# -*- coding: utf-8 -*-
"""Phase 3050 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3050'
     r'\omega_p47_kvdeep_dissection_qwen')
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
assert verdict == 'kvdeep_local_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a139_recapture_diff'] == 0.0
assert an['a140a_front_diff'] == 0.0
assert an['a140b_l30_diff'] == 0.0
assert an['a141_sham_diff'] == 0.0
assert an['a142_fail'] == 0
assert an['a142_checked'] == 24
assert an['a143_lens_diff'] == 0.0
assert abs(an['a144_max_dlg']
           - 2415.2599794127195) < 1e-6
t2 = st['T2_single_layer']
assert t2['peak_layer'] == 35
assert abs(t2['med_at_peak']
           - 0.702645386234352) < 1e-12
assert abs(t2['med_L30']
           - 0.2803912339087099) < 1e-12
assert abs(t2['med_L33']
           - 0.4467319090676518) < 1e-12
t3 = st['T3_suffix']
assert abs(t3['obs_deep']
           - 0.6953780087991128) < 1e-12
assert t3['onset_layer'] == 28
assert abs(t3['med_cos'][7]
           - 0.702645386234352) < 1e-12
assert abs(t3['med_cos'][6]
           - 0.5745394904985781) < 1e-12
t4a = st['T4a_null_deep']
assert abs(t4a['p_deep']
           - 0.004975124378109453) < 1e-12
assert abs(t4a['med_null']
           - 0.5479070783492456) < 1e-12
assert abs(t4a['max_null']
           - 0.6460919273852599) < 1e-12
t4b = st['T4b_null_front']
assert abs(t4b['p_front_restricted']
           - 0.009950248756218905) < 1e-12
assert abs(t4b['med_null']
           - 0.45592692832380066) < 1e-12
t5 = st['T5_lens']
assert t5['argmax_layer'] == 35
assert t5['onset_layer'] == 35
assert abs(t5['med_L35'] - 1.0) < 1e-12
assert abs(t5['med_L30']
           - 0.49641945075353044) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3050
           for m in led['measurements']):
    claim = (
        'Omega-P47 (plan 3050 A) - KV deep-band '
        '(LB30-35) causal dissection, run4 fp32 '
        'authoritative. RUN1 crashed pre-anchor '
        '(lens tensor shape / infeasible full '
        'storage). RUN2 crashed pre-anchor '
        '(decoder layer returns a bare tensor; '
        'hook out[0] already stripped the batch '
        'dim). RUN3 completed but FAILED the '
        'a143 anchor: lg[0,-1] on the 2-D lm_head '
        'output (n,NVOC) is a SCALAR, so lens '
        'rows were constant vectors; fixed to '
        'lg[-1]. Capture bank LOADED from the '
        'z48 npz (a139 re-capture bit 0.0) with '
        'per-pair chain anchors vs the z49 npz '
        '(a140a FRONT diff 0.0, a140b L30-35 '
        'diff 0.0) and the a143 lens final-layer '
        'vs LG bit 0.0. RESULTS: single-layer '
        'profile peaks at the LAST layer L35 '
        '(med cos 0.7026, vs L31 0.6311, L33 '
        '0.4467, L30 0.2804; mid-layer outlier '
        'L12 0.5170); L35 ALONE reproduces the '
        'whole deep-band signal (0.7026 vs '
        'LB30-35 0.6954) and even the ALL-band '
        'signal (0.6067); suffix bands are '
        'non-monotonic (L34-35 0.5745 < L35 '
        '0.7026 - adding L34 dilutes). PRIMARY '
        'restricted null on L30-35 x FRONT: obs '
        '0.6954 vs null med 0.5479 max 0.6461, '
        'p = 0.00498. FRONT-band restricted null '
        'retro-validates 3049: p = 0.00995 '
        '(significant; the 3049 all-True-mask '
        'null randomized ALL positions, '
        'null med 0.3654 < restricted 0.4559, '
        'so the 3049 verdict was conservative). '
        'Logit-lens: direction forms gradually '
        'in readout space (med cos 0.42 at L26, '
        '~0.50 L30-34, 1.0 at L35 by '
        'construction). Verdict kvdeep_local_'
        'qwen - the carrying KV payload is '
        'concentrated at the LAST layer, '
        'single-layer sufficient; deep-band KV '
        'carrying (0.70) exceeds the lens '
        'direction maturity (~0.50) -> active '
        'direction writing through L31-35, not '
        'passive relay.')
    meas = {
        'meas_id': 'meas3050_omega_p47_kvdeep_'
                   'dissection_qwen',
        'phase': 3050,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a139 re-capture bit 0.0 vs '
                   'z48; a140a FRONT diff 0.0 and '
                   'a140b L30-35 diff 0.0 vs z49 '
                   'per pair; a141 sham bit 0.0; '
                   'a142 integrity fails 0 (24 '
                   'checked); a143 lens vs LG bit '
                   '0.0; a144 max dlg 2415',
        'artifacts': {
            'result': 'phase3050/omega_p47_'
                      'kvdeep_dissection_qwen/'
                      'result.json',
            'npz': 'phase3050/omega_p47_'
                   'kvdeep_dissection_qwen/'
                   'omega_p47_kvdeep_dissection_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run4 authoritative (655.6s, fp32); '
                'norm-matched nulls RESTRICTED to '
                'the exact-field rows (3049 '
                'retro-validation T4b); lens '
                'captured via decoder-layer hooks '
                '(output_hidden_states unreliable '
                'on transformers 5.14)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 189
    l14['connects'].append({
        'meas_id': 'meas3050_omega_p47_kvdeep_'
                   'dissection_qwen',
        'phase': 3050,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P47: deep-band '
                        'dissection - single-layer '
                        'profile peaks at L35 '
                        '(0.7026), L35 alone >= '
                        'whole deep band 0.6954 '
                        'and ALL band 0.6067; '
                        'restricted deep-band null '
                        'p 0.00498; FRONT '
                        'restricted null '
                        'retro-validates 3049 '
                        '(p 0.00995); suffix bands '
                        'non-monotonic (L34 '
                        'dilutes); lens direction '
                        'maturity ~0.50 at L30-34 '
                        '< KV carrying 0.70 -> '
                        'active writing; '
                        'kvdeep_local_qwen'})
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
if '## Phase 3050:' not in memo:
    sec = u'''## Phase 3050: Ω-P47 深层带因果解剖——承载集中于末层 L35、单层复现全带信号（kvdeep_local_qwen） [%(created)s]

**判决：`kvdeep_local_qwen`**（run4 fp32 权威 655.6s；run1-2 判决前崩溃 + run3 完成但 a143 锚失败——**lg[0,-1] 在二维 lm_head 输出 (n,V) 上是标量**（batch 维已被 hook out[0] 剥掉），lens 行全为常数向量，剖面呈块状退化的假象；修复 lg[-1] 后全矩阵探针 lm_head(norm(cap35)) vs logits 逐位置 bit 0.0。三次崩溃/失败全部如实入册）

### 设计
逐层单独替换剖面（36 层×FRONT 带）+ 后缀累积带 L(l..35) l=28..35（l=30 带=3049 LB30-35 复现臂）+ **行限制性 norm 匹配 null**（深带 L30-35×FRONT 主检验 R=200；FRONT 带限制性 null 回溯验证 3049）+ logit-lens 方向剖面（decoder 层 hook 自捕获，a143 末层 vs LG bit 锚）。锚：a139 重捕获 / a140a FRONT / a140b L30-35 逐对 bit 0.0 vs z49 / a141 sham / a142 完整性 24 forwards / a144。

### 核心结果（重复三遍）
**① 承载集中于末层 L35**：单层剖面峰 **L35=0.7026**（L31=0.6311 次峰、L33=0.4467、L30=0.2804；中层孤峰 L12=0.5170）——**L35 单独复现整个深带（0.6954）乃至全带（0.6067）的信号**，单层即足。**② 后缀带非单调**：L34-35=0.5745 < L35=0.7026（加入 L34 反而稀释），L29-35=0.6984 最高；onset=L28。**③ 深带限制性 null 显著**：obs 0.6954 vs null med 0.5479 / max 0.6461，**p=0.00498**。**④ 3049 回溯验证成立**：FRONT 带严格限制性 null p=**0.00995** 仍显著（3049 因全 True mask 实际随机化了全部位置——null med 0.3654 < 带内 0.4559，原判决偏保守方向，结论稳健）。**⑤ logit-lens 对齐**：方向在读出空间渐成形（L26 0.42 → L30-34 ~0.50 → L35 按构造=1.0）——深带 KV 替换承载（0.70）**超过** lens 方向成熟度（~0.50）→ L31-35 深层在**主动写入**方向分量，非被动中继。

### 机制链定版
**KV 载荷层定位完成：末层 L35 主承载（单层充分）+ L31 次峰 + L12 中层孤峰**。与 3048 单层剖面 argmax L33、3049 深带 LB30-35 主导合并：承载沿层维向读出端集中，末端一层承担近乎全部 KV 通路信号；方向成形（lens）与承载写入（替换）分离——深层 KV 是写入器而非转发器。

### 方法论入册
- **二维输出索引纪律（3050）**：hook out[0] 剥掉 batch 维后，lm_head 输出 (n,V) 二维——末 token 是 lg[-1]；lg[0,-1] 是标量，广播成常数向量产生块状假剖面。**锚体系必须在统计前抓住此类退化**（a143 正是为此设的）。
- **行限制性 null 纪律（3050）**：norm 匹配 null 必须只随机化 exact 场的同一行集（band×position），其余行保持自替换；全位随机 null 会低估 null 分布（3049 回溯验证：0.3654 vs 0.4559）。
- **transformers 5.14**：decoder layer 返回裸 tensor（hook out[0] 已剥 batch）；output_hidden_states=True 不可信（收集机制与源码不符且破坏前向）——层捕获一律自建 hook。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3050/omega_p47_kvdeep_dissection_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3051 菜单**——A（主选）**L35 末层解剖**：K vs V 分拆（3045/3046 协议）+ 头级分解（8 KV 头逐个替换）+ 与 L35 attention 读出权重对齐；B L12 中层孤峰定位（哪条独立通路）；C 跨模型 DS7B 复刻（KV 阶梯+定位协议全链）；D 后缀带非单调解剖（L34 稀释效应成分分解）。
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
if '## 十二、3050 增补' not in aud:
    add = u'''
    
---

## 十二、3050 增补：KV 载荷层定位定版——末层 L35 单层充分承载（Omega-P47，判决 kvdeep_local_qwen）

1. **层定位收敛到末层**：逐层单独替换剖面峰 L35=0.703，单独一层复现深带（0.695）与全带（0.607）信号；L31 次峰 0.631、中层孤峰 L12=0.517（独立通路候选）。后缀累积带非单调（L34 稀释）——层维承载不是单调叠加而是稀疏集中。
2. **深带限制性 null p=0.00498**；**3049 回溯验证**：FRONT 带严格限制性 null p=0.00995（3049 的 null 因 mask 修复实际随机化了全部位置，带内 null med 0.4559 > 全位 0.3654——原判决偏保守，结论稳健）。
3. **写入器而非转发器**：logit-lens 方向成熟度在 L30-34 仅 ~0.50，而深带 KV 替换承载 0.70——深层 KV 在方向完全成形前就主动写入分量；最终对齐发生在 L35 读出端。
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
if 'Phase 3050' not in prev:
    line = ('- Phase 3050 Omega-P47 KV deep-band '
            'dissection: verdict kvdeep_local_'
            'qwen (run4 fp32 655.6s; run1-2 pre-'
            'anchor crashes + run3 a143 anchor '
            'failure registered: 2-D lm_head '
            'output lg[0,-1] is a SCALAR -> lens '
            'rows constant vectors; fix lg[-1]). '
            'RESULTS: single-layer profile peaks '
            'L35 0.7026 (alone reproduces deep '
            'band 0.6954 / ALL band 0.6067); L31 '
            '0.6311 secondary, L12 0.5170 mid-'
            'layer outlier; suffix bands non-'
            'monotonic (L34-35 0.5745 < L35); '
            'restricted deep null p 0.00498; '
            'FRONT restricted null retro-'
            'validates 3049 p 0.00995 (3049 all-'
            'True mask randomized ALL positions); '
            'lens maturity ~0.50 at L30-34 < KV '
            'carrying 0.70 -> active writing. '
            'Audit addendum 12; ledger 189/L14 '
            '157.\n')
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
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%；log 占位符数=实参数。
3. 重跑先删旧产物；负结果/锚失败/崩溃如实登记（含"完成但锚失败"）；verdict 单分支赋值。
4. 统计量纪律：obs 与 null 同量纲同范围；**null 必须限制在 exact 场同一行集**（3050）；logit 级因果读出 fp32。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨精度 cos 门 ≥0.999。
- 精确 KV 复放：post-norm K 捕获+RoPE 偏移旋转；repl/mask 行数不变式（全长 repl+全 True mask）。
- 捕获库跨相位复用：上游 npz+抽样重捕获 bit 锚+派生统计量逐对复现锚。
- **二维输出索引纪律（3050）**：hook out[0] 剥 batch 维后 lm_head 输出 (n,V)——末 token 是 lg[-1]，lg[0,-1] 是标量（广播成常数向量→块状假剖面）；末层 vs LG bit 锚（a143 式）必须在统计前抓此类退化。
- **transformers 5.14（3050）**：decoder layer 返回裸 tensor（hook out[0] 已剥 batch，dim==3 再 [0]）；output_hidden_states=True 不可信且破坏前向——层捕获一律自建 hook。

## 统计判据纪律
- 判据可达性先检；构造匹配置换；maxT；镜像必配。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯（3045-3048）→载荷定位（3049：体位均匀+TGT 负）→**层定位（3050：末层 L35 单层充分 0.703、L31 次峰、L12 孤峰、后缀带非单调 L34 稀释、深带限制 null p=0.005、lens 成熟度 0.50<承载 0.70=主动写入）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj 输出 (1,s,1024)（repl 传平铺）；RoPE NeoX 配对、attention_scaling==1；fp32 logit 级测量。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3050）
Ω-P2（3011-3050）：3011 门控=L3 KV；3018-3019 抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3037 KV 多分量；3045-3046 单层 V/K；3047 目标位 null；3048 全位置精确复放首次显著；3049 载荷定位（体位均匀+深带集中）；**3050 层定位：末层 L35 单层充分承载+主动写入（kvdeep_local_qwen）**。

## 下一步
- max=3050，下一个 3051（A 主选 **L35 末层解剖**——K/V 分拆+8 KV 头逐个替换+attention 读出权重对齐；B L12 中层孤峰定位；C 跨模型 DS7B 复刻；D L34 稀释效应分解）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
