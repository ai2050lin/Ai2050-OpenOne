# -*- coding: utf-8 -*-
"""Phase 3049 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3049'
     r'\omega_p46_kvload_localization_qwen')
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
assert verdict == 'kvload_uniform_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a132_recapture_diff'] == 0.0
assert an['a133_allband_diff'] == 0.0
assert an['a134_scr_diff'] == 0.0
assert an['a135_sham_diff'] == 0.0
assert an['a136_fail'] == 0
assert abs(an['a137_max_dlg']
           - 2316.0675531044194) < 1e-6
tb = st['T2_bands']
assert abs(tb['ALL']['med_cos']
           - 0.6066816593084429) < 1e-12
assert abs(tb['FRONT']['med_cos']
           - 0.6227623885915632) < 1e-12
assert abs(tb['TGT']['med_cos']
           - (-0.16030627137642653)) < 1e-12
assert abs(tb['REST']['med_cos']
           - 0.6395750823763471) < 1e-12
nu = st['null_front']
assert abs(nu['p_front']
           - 0.004975124378109453) < 1e-12
assert abs(nu['med_cos']
           - 0.3653646100871494) < 1e-12
assert abs(nu['max_cos']
           - 0.5454302694402624) < 1e-12
assert nu['R'] == 200 and nu['seed'] == 9901
t3 = st['T3_layerbands']
assert t3['argmax_band'] == 5
assert abs(t3['med_cos_at_argmax']
           - 0.6953780087991128) < 1e-12
assert abs(t3['med_cos'][0]
           - 0.19456256663229493) < 1e-12
t4 = st['T4_SCR']
assert abs(t4['med_kill_frac']
           - 2.2128508596306338) < 1e-12
assert abs(t4['med_kill_cos']
           - 0.07918879803189052) < 1e-12
assert abs(t4['front_kill_cos']
           - 0.7078680188193239) < 1e-12
assert abs(t4['back_kill_cos']
           - 0.6365750363565446) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3049
           for m in led['measurements']):
    claim = (
        'Omega-P46 (plan 3049 A) - KV payload '
        'localization (position-band x layer-band), '
        'run6 fp32 authoritative. RUN1 crashed '
        'pre-anchor: this phase assembles only the '
        '32 old-body prompts (no NEW_BODIES bank) '
        'but the copied assertion still required '
        'n_pr==48 and one a132 index pointed at a '
        'new-body row. RUN2 crashed at the first '
        'V-replacement: the copied forward_run '
        'applied the K-side reshape to replV, but '
        'v_proj output is (1,s,1024) and 3048 '
        'passes replV flat. RUN3/RUN4 crashed '
        'pre-anchor on the band_rep K source slice '
        'reshapes ((NL,1024) must reshape to '
        '(NL,8,128) and rot_apply output to '
        '(NL,1024)). RUN5 crashed pre-anchor: '
        'full-length repl field paired with a '
        'partial band mask (hook requires repl '
        'rows == mask.sum()); fixed by an all-True '
        'mask (non-band rows already hold the '
        'self-replacement values). Capture bank '
        'LOADED from the z48 npz with a132 sampled '
        're-capture bit 0.0, a133 ALL-band per-pair '
        'cos/frac vs z48 diff 0.0, a134 SCR vs z48 '
        'diff 0.0 (cross-phase reproduction '
        'anchors). RESULTS: FRONT band (prefix-'
        'adjacent positions 0..3) exact replacement '
        'med cos 0.6228 vs norm-matched random '
        'null med 0.3654 max 0.5454, p = 0.00498 '
        '(significant); REST band 0.6396 >= FRONT '
        '0.6228, TGT-only -0.1603 (negative); '
        'verdict kvload_uniform_qwen - carrying '
        'payload is UNIFORMLY distributed over '
        'body positions, NOT prefix-localized. '
        'Layer-band x FRONT matrix: deep block '
        'LB30-35 dominates 0.6954 vs 0.11-0.28 '
        'elsewhere. SCR split: front-half scramble '
        'kill_cos 0.7079, back-half 0.6366 (both '
        'halves of the prefix carry direction '
        'through their KV). CONCLUSION: the '
        'carrying KV payload is body-wide with a '
        'DEEP-LAYER concentration (LB30-35), '
        'target-position KV is not a carrier '
        '(negative cos, 3046/3047 consistent); '
        'fraction bounds: FRONT band alone '
        'reproduces ~100pct of the ALL-band '
        'direction signal.')
    meas = {
        'meas_id': 'meas3049_omega_p46_kvload_'
                   'localization_qwen',
        'phase': 3049,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a132 re-capture bit 0.0 vs z48; '
                   'a133 ALL-band cos/frac diff 0.0 '
                   'per pair; a134 SCR diff 0.0; '
                   'a135 sham bit 0.0; a136 bit-exact '
                   'fails 0; a137 max dlg 2316',
        'artifacts': {
            'result': 'phase3049/omega_p46_'
                      'kvload_localization_qwen/'
                      'result.json',
            'npz': 'phase3049/omega_p46_'
                   'kvload_localization_qwen/'
                   'omega_p46_kvload_localization_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run6 authoritative (316.9s, fp32); '
                'capture bank reused from the z48 '
                'npz with bit-level re-capture and '
                'per-pair statistics reproduction '
                'anchors; full-length repl + all-'
                'True mask pattern replaces the '
                'partial-mask pattern',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 188
    l14['connects'].append({
        'meas_id': 'meas3049_omega_p46_kvload_'
                   'localization_qwen',
        'phase': 3049,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P46: KV payload '
                        'localization - FRONT band '
                        'exact replacement significant '
                        '(cos 0.6228 vs null max '
                        '0.5454, p 0.00498) but REST '
                        'band 0.6396 >= FRONT -> '
                        'uniform body-wide carrying, '
                        'not prefix-localized; TGT-only '
                        'negative (-0.16); layer-band '
                        'matrix: deep LB30-35 dominates '
                        '(0.695 vs 0.11-0.28); SCR '
                        'split front 0.708 / back 0.637 '
                        '(both prefix halves carry '
                        'direction); cross-phase '
                        'anchors a132/a133/a134 bit '
                        '0.0; kvload_uniform_qwen'})
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
if '## Phase 3049:' not in memo:
    sec = u'''## Phase 3049: Ω-P46 KV 载荷定位——承载均匀分布于全体位置、深层带集中（kvload_uniform_qwen） [%(created)s]

**判决：`kvload_uniform_qwen`**（run6 fp32 权威 316.9s；run1-5 五次判决前崩溃全部如实入册：n_pr 断言残留 48/V 侧多余 reshape/K 源切片与赋值端 reshape/全长 repl 配部分 mask 行数失配——mask 改全 True（非 band 位已是自替换值，语义等价）。捕获库直接复用 z48 npz，a132 抽样重捕获、a133 ALL 带逐对 cos/frac、a134 SCR 三重跨相位锚全部 bit 0.0）

### 设计
位置带精确替换（post-norm K + RoPE 偏移旋转 + V verbatim，其余位自替换）：FRONT（体前 4 位，prefix 邻近）/ TGT（仅目标位）/ REST（去目标位）/ ALL（3048 复现臂）；FRONT 带 norm 匹配随机替换 null（R=200，预注册 24-pair 中位数）；层带×FRONT 矩阵（6 块）；SCR 复现+前后半分拆。

### 核心结果（重复三遍）
**① FRONT 带显著但非集中**：FRONT med cos=**0.6228** vs null med 0.3654 / max 0.5454，**p=0.00498**；但 REST=**0.6396** ≥ FRONT，TGT=**−0.1603**（负）——承载均匀弥散于全体体位，非前段集中；单独 FRONT 带已复现 ALL 带 (~100%%) 的方向信号。**② 层带矩阵深层主导**：LB30-35=**0.6954** vs 其余 0.11-0.28——载荷集中在最深 6 层。**③ SCR 前后分拆**：前半打乱 kill_cos=**0.7079**、后半 **0.6366**——前缀两半的 KV 都承载方向（与 3048 全打乱 kill_cos 0.079 对照：分位打乱是"替换为随机"，保留幅度；全打乱跨位置换模式破坏方向信息）。**④ 目标位 KV 非载体**：TGT-only 负 cos 且 3046（单层 0.3pct）/3047（联合 null）一致。

### 机制链定版
**KV 载荷定位完成：体位均匀 + 层深集中（LB30-35）+ 目标位排除**。前缀效应的 KV 通路=全部体位深层联合写入（深带单独 0.70 显著方向分量），非局部化"路由节点"；叠加 3048 非忠实性（cos 0.61、超调），最终图景=**深层 KV 联合承载部分方向 + 多通路（残流/MLP）补足**。

### 方法论入册
- **repl/mask 行数不变式**：hook 要求 repl 行数==mask.sum()；全长 repl+全 True mask（自替换值预填）优于部分字段+部分 mask。
- **捕获库跨相位复用配方**：z48 npz 加载 + 抽样重捕获 bit 锚 + 派生统计量逐对复现锚（a132/a133/a134）——避免 48 次捕获且链恒等最强。
- 五连崩溃教训：复刻上游脚本时 reshape/断言/返回值签名逐处核对，不凭记忆。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3049/omega_p46_kvload_localization_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3050 菜单**——A（主选）**深层带（LB30-35）因果解剖**：逐层替换剖面（36 层单独）+ 深带 vs 随机深带 null + 与 logit-lens 读出层对齐检验；B 阻尼场通道分解；C 跨模型 DS7B 复刻（KV 阶梯+定位协议）；D 跨语言共享子空间。
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
if '## 十一、3049 增补' not in aud:
    add = u'''
    
---

## 十一、3049 增补：KV 载荷定位——均匀弥散 + 深层集中，"语法路由节点"不存在（Omega-P46，判决 kvload_uniform_qwen）

1. **位置定位判负**：FRONT（prefix 邻近）带显著（cos 0.623，p=0.005）但 REST 0.640 ≥ FRONT——承载均匀弥散于全部体位；目标位 KV 单独为负（−0.16）。HDMCC 的"Attention 按语法重新抽取特征"隐含的**局部化路由节点在 KV 层面不存在**。
2. **层定位**：深带 LB30-35 单独承载 cos 0.695 vs 其余层带 0.11-0.28——KV 载荷集中最深 6 层（近读出端），与 3048 单层剖面 argmax L33 一致。
3. **SCR 前后分拆**（kill_cos 前 0.708 / 后 0.637）修正 3048 解释：全前缀打乱失效（0.079）是跨位置模式破坏，非"前缀位 KV 不承载方向"——前缀 KV 确实承载方向，但需要位置结构保持。
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
if 'Phase 3049' not in prev:
    line = ('- Phase 3049 Omega-P46 KV payload '
            'localization: verdict kvload_uniform_'
            'qwen (run6 fp32 316.9s; five pre-'
            'verdict crashes registered: stale '
            'n_pr assert / spurious V reshape / '
            'K slice + assignment reshapes / '
            'full-repl vs partial-mask row-count '
            'mismatch -> all-True mask). Capture '
            'bank reused from z48 npz (a132 '
            're-capture bit 0.0, a133 ALL-band '
            'per-pair diff 0.0, a134 SCR diff '
            '0.0). RESULTS: FRONT band cos 0.6228 '
            'vs null med 0.3654/max 0.5454 p '
            '0.00498 but REST 0.6396 >= FRONT, '
            'TGT -0.1603 -> uniform body-wide '
            'carrying, not prefix-localized; '
            'layer-band matrix LB30-35 dominates '
            '0.695 vs 0.11-0.28; SCR split front '
            '0.708 / back 0.637 (both halves '
            'carry direction). Deep-layer KV '
            'joint carrying + multi-path; audit '
            'addendum 11; ledger 188/L14 156.\n')
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
3. 重跑先删旧产物；负结果与判据作废如实登记；verdict 单分支赋值。
4. **统计量纪律（3044-3049）**：obs 与 null 同量纲同范围（null 循环严禁复用外层残留变量）；统计量先量纲自检；logit 级因果读出必须 fp32；改函数返回值后全文件查解包。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨精度 cos 门 ≥0.999。
- **干预点纪律（3046）**：先探针 norm 管线（qk-norm 在 k_proj 与 RoPE 之间）；自然尺度/场轴定义在干预点所在空间；捕获顺序先改后录。
- **完整性检查纪律（3047）**：禁用 (x+d)−x==d；位级 mod == orig+delta / where(mask,repl,orig) 散射式。
- **精确 KV 复放配方（3048）**：post-norm（k_norm 输出 (1,s,8,128)）捕获+替换；RoPE 相对旋转必须在 post-norm 点施加；a129 式 past-key 实证验证先于统计。
- **BPE 对齐纪律（3048）**：尾部对齐+首 token strip 等值，禁朴素子序列搜索。
- **repl/mask 不变式（3049）**：hook 要求 repl 行数==mask.sum()；全长 repl+全 True mask（自替换值预填）优于部分字段+部分 mask。
- **捕获库跨相位复用（3049）**：上游 npz 加载+抽样重捕获 bit 锚+派生统计量逐对复现锚；复刻上游脚本时 reshape/断言/签名逐处核对。

## 统计判据纪律
- 判据可达性先检；退化行先剔；构造匹配置换；池内标签置换 MC。

## 机制解释审计链（命名前依次检查）
…→KV 多分量→谱水平复核→场方差分解→相关可测层≠因果作用层→**KV 五级阶梯**：单层 V（3045）→单层 K（3046）→全层联合目标位（3047）→全位置精确复放（3048 显著 cos 0.607、超调 3.2）→**载荷定位（3049：体位均匀 REST≥FRONT、TGT 负、层带 LB30-35 深层主导 0.695、SCR 前后分拆 0.71/0.64）**→深层 KV 联合承载+多通路。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj 输出 (1,s,1024)（repl 传平铺）；RoPE NeoX 配对、attention_scaling==1。
- output_attentions detach().cpu()；fp32 用于 logit 级因果测量。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3049）
Ω-P2（3011-3049）：3011 门控=L3 KV；3018-3019 抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3031-3036 异质性伪影/头集中/指纹 logistic/非正交；3037 KV 多分量；3038/3039 协议分层；3040-3041 情景分量+谱复核；3042-3043 风格场；3044 作废；3045 fp32 V 无特权+bf16 噪声地板；3046 qk-norm 发现；3047 全层联合目标位 null；3048 全位置精确复放首次显著；**3049 载荷定位：体位均匀+深层 LB30-35 集中+目标位排除（kvload_uniform_qwen）**。

## 下一步
- max=3049，下一个 3050（A 主选 **深层带 LB30-35 因果解剖**——逐层单独替换剖面+深带 null+与 logit-lens 读出层对齐；B 阻尼场通道分解；C 跨模型 DS7B 复刻；D 跨语言共享子空间）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
