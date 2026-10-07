# -*- coding: utf-8 -*-
"""Phase 3046 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3046'
     r'\omega_p43_kfield_injection_qwen')
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
assert verdict == 'kfield_route_specific_qwen', verdict
assert res['anchor_core_ok'] is True
an = res['anchors']
assert an['a114_dup_prefill_bit'] == 0.0
assert an['a115_dup_all_bit'] == 0.0
assert an['a116_source_seals'] is True
assert abs(an['a117_cross_precision_min_cos']
           - 0.9997792619950878) < 1e-12
assert an['a117_matched'] == 16
assert an['a118_integrity_ok'] is True
assert an['n_integ_fail'] == 0
assert an['a119_chain_entry_ok'] is True
assert abs(an['max_dlg_t2']
           - 9.037592265194522) < 1e-9
t1p = res['T1pre_field']
assert abs(t1p['obs_L3']
           - 0.3796115350862224) < 1e-9
assert t1p['p_L3'] == 0.0
assert abs(t1p['obs_L20']
           - 0.2882847586438275) < 1e-9
assert t1p['p_L20'] == 0.0
assert t1p['n_pairs'] == 84
t1o = res['T1post_field']
assert abs(t1o['obs_L3']
           - 0.7841612266042324) < 1e-9
assert abs(t1o['obs_L20']
           - 0.3349760128042324) < 1e-9
t2 = res['T2_efficiency']
assert abs(t2['obs_L3']
           - 0.8769420632268898) < 1e-9
assert abs(t2['p_L3'] - 0.7151) < 1e-12
assert abs(t2['obs_L20']
           - 1.0761790583025406) < 1e-9
assert abs(t2['p_L20'] - 0.33935) < 1e-12
t3a = res['T3a_axis_replay']
assert abs(t3a['p_L3'] - 0.83365) < 1e-12
assert abs(t3a['p_L20'] - 0.2212) < 1e-12
t3b = res['T3b_exact_replay']
assert abs(t3b['obs_L20']
           - 0.0455720811820145) < 1e-9
assert abs(t3b['p_L20'] - 0.0161) < 1e-12
assert abs(t3b['p_L3'] - 0.94635) < 1e-12
assert abs(t3b['frac_med_L3']
           - 0.0007976089701822217) < 1e-12
assert abs(t3b['frac_med_L20']
           - 0.0029855174287045936) < 1e-12
t4 = res['T4_attention']
assert abs(t4['L3']['spearman'] - 1.0) < 1e-12
assert abs(t4['L3']['med_axis_dlg']
           - 0.045471705220890196) < 1e-9
assert abs(t4['L3']['med_rand_dlg']
           - 0.21458062364991803) < 1e-9
t5 = res['T5_ladder']
assert abs(t5['L3'][-1][1]
           - 0.3645632843741782) < 1e-9
assert abs(t5['L20'][-1][1]
           - 0.32064115970645507) < 1e-9
fl = res['flags']
assert fl['fieldK'] is True
assert fl['causK'] is False
assert fl['routK'] is True
assert abs(fl['med_post_ratio']
           - 7.318483970940876) < 1e-9

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3046
           for m in led['measurements']):
    claim = (
        'Omega-P43 (plan 3046 A) - K-side field '
        'injection, run5 fp32 authoritative (pre-'
        'norm redesign). RUN1-4 REGISTERED: run1 '
        'assembly slip (NEW else-branch BODIES[bi]); '
        'run2 cuda attention crash; run3 a118 208/'
        '208 + flat T5 dose -> probe root-cause: '
        '(i) capture-order bug, (ii) QWEN3 QK-NORM '
        '- k_norm (weight mean 1.7633) sits between '
        'k_proj and RoPE; pre-norm kv7 key norm '
        '1.708 vs post-norm+RoPE 20.16, so the '
        'post-norm displacement scale gK 15.7 '
        'injected pre-norm OVERDRIVES ~9x '
        '(saturating attention, washing out '
        'direction effects); run4 T5 unpack slip. '
        'RUN5 RESULTS (pre-norm natural scale '
        'gpre, L3 med 0.238 on keys of norm 1.68; '
        'L20 med 3.45 on keys of norm 9.75; '
        'anchors a114/a115 bit 0.0, a117 16/16 '
        'cos 0.99978, a118v2 fail 0, a119 pass): '
        '(T1pre) the K field is REAL at the '
        'intervention point: same-prefix cross-'
        'body med |cos| L3 0.3796 and L20 0.2883 '
        'vs Gaussian null 0.060 (6.3x / 4.8x, '
        'both p 0.00000, 84 pairs); qk-norm '
        'AMPLIFIES field alignment (T1post L3 '
        '0.7842 = 13x); (T2) NO efficiency '
        'privilege: 0.877 (p 0.72) / 1.076 (p '
        '0.34); (T3a) no axis replay; (T3b) EXACT '
        'single-layer replay of the prefix key '
        'change: L20 cos 0.0456 vs matched random '
        '(p 0.0161) marginally significant, but '
        'the replayed key change carries only '
        '0.08 pct (L3) / 0.30 pct (L20) of the '
        'prefix logit displacement norm; (T4) K '
        'to attention to logit chain mechanically '
        'clean (Spearman rho 1.0 at L3) yet the '
        'field axis moves attention LESS than '
        'random (0.0062 vs 0.0279); (T5) healthy '
        'monotone dose curves. CONCLUSION: '
        'kfield_route_specific_qwen - the prefix '
        'field exists in K at both the pre- and '
        'post-norm measurement levels, but a '
        'single-layer key change is causally '
        'negligible for the readout (0.3 pct); '
        'together with 3045 (V side) the prefix '
        'logit effect is NOT carried by any '
        'single-layer KV write and must be '
        'distributed across layers/components. '
        'NEXT: multi-layer joint key replay '
        '(all-layer bound), damping anatomy, '
        'cross-model, cross-lingual.')
    meas = {
        'meas_id': 'meas3046_omega_p43_kfield_'
                   'injection_qwen',
        'phase': 3046,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a114/a115 bit 0.0; a116 seals; '
                   'a117 cos 0.99978 (16/16); '
                   'a118v2 fail 0 (dpre exact, '
                   'non-target bit, post bound 60, '
                   'med ratio 7.32 descriptive); '
                   'a119 max dlg 9.04; T1pre 6.3x/'
                   '4.8x p 0.0; T2 0.877/1.076 no '
                   'spec; T3b L20 p 0.0161 frac '
                   '0.30 pct; T4 rho 1.0',
        'artifacts': {
            'result': 'phase3046/omega_p43_'
                      'kfield_injection_qwen/'
                      'result.json',
            'npz': 'phase3046/omega_p43_'
                   'kfield_injection_qwen/'
                   'omega_p43_kfield_injection_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run5 authoritative (115.2s, fp32 '
                'pre-norm redesign); run1 assembly '
                'slip, run2 cuda attention crash, '
                'run3 qk-norm overdrive diagnosis '
                '(a118 208/208), run4 T5 unpack '
                'slip; T1pre/T1post/T2/T3a/T3b/T4 '
                'reproduced bit-identically across '
                'run4/run5 under frozen seeds',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 185
    l14['connects'].append({
        'meas_id': 'meas3046_omega_p43_kfield_'
                   'injection_qwen',
        'phase': 3046,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P43: K-side causal '
                        'test - QWEN3 QK-NORM '
                        'discovered as an '
                        'intervention-point hazard '
                        '(pre-norm key norm 1.71 vs '
                        'post-norm 20.16; post-norm-'
                        'scale injection overdrives '
                        '9x and saturates attention); '
                        'at the architecture-correct '
                        'pre-norm scale the K field '
                        'is REAL (6.3x/4.8x, p 0.0) '
                        'but carries NO efficiency '
                        'privilege and the EXACT '
                        'single-layer key replay '
                        'carries only 0.3 pct of the '
                        'prefix logit displacement; '
                        'single-layer KV writes (V '
                        'per 3045, K per 3046) are '
                        'causally negligible for '
                        'readout; kfield_route_'
                        'specific_qwen'})
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
if '## Phase 3046:' not in memo:
    sec = u'''## Phase 3046: Ω-P43 K 侧场注入——**Qwen3 qk-norm 架构发现（干预点陷阱）+ pre-norm 重设计**；K 场真实但单层 key 因果贡献仅 0.3pct（kfield_route_specific_qwen） [%(created)s]

**判决：`kfield_route_specific_qwen`**（run5 fp32 权威 115.2s；run1 组装转录滑误、run2 cuda attention 崩、run3 a118 全败+T5 平坦→探针裁决、run4 T5 解包滑误，均登记；a114/a115 位级 0.0、a117 16/16 cos 0.99978、a118v2 fail=0、a119 过；T1-T4 跨 run4/run5 冻结种子逐位复现）

### 架构发现（重复三遍）
**Qwen3 有 qk-norm**：k_norm（weight mean 1.7633）位于 k_proj 输出与 RoPE 之间；pre-norm kv7 key 范数 **1.708**，post-norm+RoPE 后 **20.16**——按 post-norm 位移尺度 gK=15.7 在 pre-norm 点注入 = **过驱动约 9 倍**：注意力饱和（T5 剂量 16× 范围全平坦 0.611）、方向效应被淹没（T2≈1.00）、post-norm ratio 门 [0.9,1.1] 架构性失效（实测 1.7325）。**干预点必须定义在 k_norm 之前的 pre-norm 空间，自然尺度取 pre-norm 位移范数**（L3 med 0.238 于 key 范数 1.68；L20 med 3.45 于 9.75）。

### run5 核心结果
**① K 场在干预点真实存在**：pre-norm 同前缀跨基体对齐 L3 **0.3796**（6.3× 高斯 null 0.060）、L20 **0.2883**（4.8×），均 p=0.00000（84 对）；qk-norm **放大**场对齐（T1post L3 0.7842=13×）。**② 无效率特权**：pre-norm 场轴效率 0.877（p 0.72）/ 1.076（p 0.34）。**③ 精确单层复放（T3b，k_norm(x0+d)=k_norm(xc) 逐点成立）**：L20 cos 0.0456 vs 匹配随机（p=0.0161）边缘显著，但复放的 key 变化仅携带前缀 logit 位移范数的 **0.08pct（L3）/ 0.30pct（L20）**。**④ K→attention→logit 链机械完好**：L3 Spearman rho=**1.0**，但场轴移动注意力**小于**随机（0.0062 vs 0.0279）。**⑤ 剂量曲线健康**：L3 单调升至 0.365（4×），L20 饱和于 0.32。

### 机制链定版
前缀场在 K 侧（pre- 与 post-norm）皆为真实相关结构，但**单层 key 变化对读出因果可忽略（0.3pct）**；结合 3045（V 侧无特权），**前缀的 logit 效应由任何单层 KV 写入都不承载，必分布于多层/多成分的联合变化**。KV 注入纪律新增：干预点先探针 norm 管线（qk-norm/RoPE 位置）、捕获顺序（先改后录）、返回值元数与解包一致。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3046/omega_p43_kfield_injection_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3047 菜单**——A（主选）**多层联合 K 复放**：同时替换 L0-35 全部层的 kv7 key 位移（all-layer replay 上界：前缀效应最多有多少经 key 路由承载）；B 阻尼场通道分解；C 跨模型复刻（V/K 协议上 DS7B）；D 跨语言共享子空间。
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
if '## 八、3046 增补' not in aud:
    add = u'''
    
---

## 八、3046 增补：K 侧（attention 路由）因果终审（Omega-P43，判决 kfield_route_specific_qwen）

1. **架构警示**：Qwen3 的 qk-norm（k_norm，weight mean 1.763）位于 k_proj 与 RoPE 之间；在错误空间（post-norm 尺度打在 pre-norm 点）注入会过驱动约 9 倍并饱和注意力——干预实验必须先探针 norm 管线。
2. **K 场真实**：pre-norm 干预点上同前缀跨基体对齐 6.3×（L3）/4.8×（L20），p=0.00000；qk-norm 把对齐放大到 13×（post-norm）。
3. **但因果贡献极小**：精确单层 key 复放仅携带前缀 logit 位移的 0.3pct（L20，p=0.016）；场轴无效率特权（0.877/1.076）。K→attention→logit 链机械完好（rho=1.0）——通路在、方向不带特权信息。
4. **对附件的终审**："Attention 重新绑定语法路由 / 全局引力场扭曲指纹竞争"：KV 单层写入（V 或 K）都不是承载者；前缀效应若经 KV 路由，必为多层联合效应。3047 A（全层联合 key 复放）将给出上界。
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
if 'Phase 3046' not in prev:
    line = ('- Phase 3046 Omega-P43 K-side field '
            'injection: verdict kfield_route_'
            'specific_qwen (run5 fp32 pre-norm '
            '115.2s; run1 assembly slip, run2 cuda '
            'attention crash, run3 a118 208/208 + '
            'flat T5, run4 T5 unpack slip - all '
            'registered). MAJOR: Qwen3 qk-norm '
            'discovered (k_norm weight mean 1.763 '
            'between k_proj and RoPE; pre-norm key '
            '1.71 vs post-norm 20.16; post-norm-'
            'scale injection overdrives 9x, '
            'saturating attention). Clean results: '
            'K field REAL at intervention point '
            '(pre-norm 6.3x/4.8x p 0.0; qk-norm '
            'amplifies to 13x) but NO efficiency '
            'privilege (0.877/1.076) and EXACT '
            'single-layer key replay carries only '
            '0.3 pct of prefix logit displacement '
            '(L20 p 0.016); K-attention-logit '
            'chain mechanically clean (rho 1.0); '
            'with 3045 (V side) single-layer KV '
            'writes are causally negligible; '
            'audit addendum 8; ledger 185/L14 153.\n')
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
4. **统计量纪律（3044-3046 血泪）**：obs 与 null 同量纲；统计量先量纲自检（cos>1、超界即 bug）；**logit 级因果读出必须 fp32**；改函数返回值后全文件查解包（3046 run4）。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨精度 cos 门 ≥0.999（a110 类）。
- **干预点纪律（3046 qk-norm 事故）**：注入前先探针 norm 管线（Qwen3 qk-norm 在 k_proj 与 RoPE 之间：pre-norm key 1.71 vs post-norm 20.16，post-norm 尺度打 pre-norm 点=过驱动 9× 注意力饱和）；自然尺度/场轴定义在**干预点所在空间**；捕获顺序先改后录；post-norm ratio 门须按架构放宽（med ratio 7.32 仅 descriptive）。
- 注入 readback：dpre 精确==delta + 非目标位 bit 0.0 + 链锚派生量位级比对。

## 统计判据纪律
- 判据可达性先检；退化行先剔；构造匹配置换；池内标签置换 MC；margin n≳40 标注探索性。

## 机制解释审计链（命名前依次检查）
…→KV 多分量→谱水平复核→场方差分解→**相关可测层≠因果作用层**→V/K 单层注入均无因果特权（3045/3046）→单层 key 复放仅承载 0.3pct 前缀 logit 位移→前缀效应=多层联合。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv（kv7↔q28-31）；KV 注入=v/k_proj 输出切片 hook；**fp32 用于 logit 级因果测量**；qk-norm：k_norm 在 k_proj 后、RoPE 前。
- output_attentions=True 的 attn 张量须 detach().cpu()（3046 run2）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3046）
Ω-P2（3011-3046）：3011 门控=L3 KV；3018-3019 抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3031-3036 异质性伪影/头集中/指纹 logistic/非正交；3037 KV 多分量；3038/3039 协议分层；3040-3041 情景分量+谱复核；3042-3043 风格场（4.5×共享、体主效应 64pct、轴共性 0.647、库外迁移）；3044 场轴注入（统计作废）；3045 fp32 终审：V 场轴无因果特权+3044 撤回+bf16 噪声地板；3046 **K 侧：qk-norm 架构发现、K 场真实（6.3×/4.8×）但单层 key 复放仅 0.3pct、K→attn→logit 链 rho=1.0**。单层 KV 写入因果可忽略。

## 下一步
- max=3046，下一个 3047（A 主选 **多层联合 K 复放**——全层 key 替换给前缀路由效应上界；B 阻尼场通道分解；C 跨模型 DS7B 复刻；D 跨语言共享子空间）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
