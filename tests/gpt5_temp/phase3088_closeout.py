# -*- coding: utf-8 -*-
"""Phase 3088 closeout (idempotent): Ledger -> L14
-> MEMO append -> HDMCC audit addendum -> wlog
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3088'
     r'\omega_p86_continuum_n12')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
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
assert verdict == 'cross_architecture_confirmed', \
    verdict
assert seal['setup_ok'] is True
z = np.load(R + r'\omega_p86_continuum_n12.npz',
            allow_pickle=False)
assert bool(z['SMOKE']) is False
assert int(z['FORWARDS']) == 0
assert int(z['FORWARDS']) == res['forwards']
assert int(z['SEED']) == 3088
assert int(z['N_PERM']) == 20000
assert int(z['N_UNITS']) == 12
assert bool(z['SETUP_OK'])
assert bool(z['ANCH_A1'])
assert bool(z['ANCH_A2'])
assert bool(z['ANCH_A3'])
rho_t = float(z['RHO_T'])
p_t = float(z['P_T'])
rho_u = float(z['RHO_U'])
p_u = float(z['P_U'])
rho_m = float(z['RHO_MIG'])
p_m = float(z['P_MIG'])
smt = float(z['SP_MEAN_T'])
smu = float(z['SP_MEAN_U'])
smm = float(z['SP_MEAN_MIG'])
assert abs(rho_t - 0.9021) < 5e-4, rho_t
assert p_t < 5e-5, p_t
tags = ['M4B_AB', 'M4B_AC', 'M4B_BC',
        'MDS7B_AB', 'MDS7B_AC', 'MDS7B_BC',
        'M3B_AB', 'M3B_AC', 'M3B_BC',
        'MGLM4_AB', 'MGLM4_AC', 'MGLM4_BC']
glm_slo = [float(z['S_LO_' + t])
           for t in tags[9:]]
glm_t = [float(z['T_MED_' + t])
         for t in tags[9:]]
t3 = [float(v) for v in z['TOP3_GLM4']]
glm_slo_exact = [min(t3[0], t3[1]),
                 min(t3[0], t3[2]),
                 min(t3[1], t3[2])]
assert all(abs(a - b) < 1e-12 for a, b in
           zip(glm_slo, glm_slo_exact)), \
    (glm_slo, glm_slo_exact)
assert all(abs(a - b) < 1e-4 for a, b in
           zip(glm_slo,
               [0.699220, 0.699220, 0.744174]))
el = res['elapsed']

meas = {
    'meas_id': 'meas3088_omega_p86_'
               'continuum_n12',
    'phase': 3088,
    'claim': (
        'Omega-P86 (plan 3088 A) - '
        'continuum discriminative test at '
        'n=12, forward-free numpy analysis '
        '(seed 3088, 20000 perm, '
        '%(el).1fs).  GLM4-9B three units '
        '(s_lo 0.6992/0.6992/0.7442 from '
        'the 3087 A2 npz sha 32d3bf08) '
        'added to the 3086 preregistered '
        'framework; statistics block bit-'
        'identical to 3086 (static '
        'identity check, 1164 chars).  '
        'MAIN rho_T = sp(s_lo, T_med) = '
        '%(rt).4f, p=%(pt).6f (cnt=0, '
        'p<5e-5) - UP from +0.8667 at '
        'n=9; GLM4 T_med +0.1175/+0.0709/'
        '+0.1080 interpolates strictly '
        'between DS7B (approx 0) and 3B '
        '(+0.29..+0.45).  rho_U=%(ru).4f '
        '(p=%(pu).6f, significant); '
        'rho_MIG=%(rm).4f (p=%(pm).6f, '
        'ns); s_mean secondary sp(T)='
        '%(smt).4f.  Internal consistency: '
        'qwen-only n=9 subset sp=+0.8667 '
        'reproduces the 3086 reference '
        'bit-level.  VERDICT '
        'cross_architecture_confirmed - '
        'the spectrum->migration '
        'continuum is not Qwen-lineage-'
        'internal; candidate upgrade to '
        'a cross-architecture regularity '
        '(4 architectures, 1 non-Qwen '
        'point).  CAVEATS: GLM4 is a '
        'single non-Qwen point; L37/L38 '
        'criterion split unresolved (an '
        'L38 replica would shift GLM4 '
        'unit values); p at permutation '
        'resolution floor; historical '
        'data preregistration covers '
        'test definition + verdict tree '
        'only.'
        % {'el': el, 'rt': rho_t,
           'pt': p_t, 'ru': rho_u,
           'pu': p_u, 'rm': rho_m,
           'pm': p_m, 'smt': smt}),
    'verdict': verdict,
    'anchors': 'a1 sha chain (3082 INPUT_'
               'SHA79/81 + sha8(S87)=='
               '32d3bf08); a2 four-model '
               'spectrum bands; a3 '
               'shapes24 + forwards S81 '
               '20634/S85 19770/S87 20922 '
               '- all True',
    'artifacts': {
        'result': 'phase3088/'
                  'omega_p86_continuum_n12/'
                  'result.json',
        'npz': 'phase3088/'
               'omega_p86_continuum_n12/'
               'omega_p86_continuum_n12'
               '.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'forward-free; stats block '
            'bit-identical to 3086; '
            'qwen-only subset reproduces '
            '3086 +0.8667 exactly; '
            '0.7s runtime, no model loads',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3088
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 227
    l14['connects'].append({
        'meas_id': 'meas3088_omega_p86_'
                   'continuum_n12',
        'phase': 3088,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P86: '
                        'continuum survives '
                        'the n=12 '
                        'discriminative test '
                        '- rho_T +0.8667 -> '
                        '+0.9021 (p<5e-5) '
                        'after adding GLM4 '
                        '(first non-Qwen-'
                        'lineage point); '
                        'GLM4 units '
                        'interpolate strictly '
                        'between DS7B and '
                        '3B; rho_U +0.7483 '
                        'sig, rho_MIG +0.4545 '
                        'ns.  Next: 3089 A '
                        'L38 arbitration '
                        'replica (criterion '
                        'split discriminant); '
                        'B G_DS gate '
                        'sensitivity; C 4B '
                        'trunk anatomy; D '
                        'layer x spectrum; E '
                        'qwen3-14b fifth '
                        'point'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
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
if '## Phase 3088:' not in memo:
    sec = u'''## Phase 3088: Ω-P86 连续统判别时刻 n=12——GLM4 三单元纳入，rho +0.8667→+0.9021 上升，判决 cross_architecture_confirmed [%(created)s]

**判决：`cross_architecture_confirmed`**。GLM4-9B 三个单元（s_lo 0.6992/0.6992/0.7442，来自 3087 A2 npz sha 32d3bf08）纳入 3086 预注册框架重跑主检验 rho(s_lo, T_med)；统计函数（spearman + 向量化置换）与 3086 源**逐字节同一**（静态同一性断言 1164 字符），seed 3088、20000 置换、免前向、elapsed %(el).1fs。

### 核心结果（重复三遍）
**① 连续统跨架构存活（一）**：主检验 rho_T = **+0.9021（p<5e-5，cnt=0/20000）**，比 n=9 的 +0.8667（p=0.00465）**上升**；GLM4 三单元 T_med（+0.1175/+0.0709/+0.1080）严格内插在 DS7B（≈0）与 3B（+0.29~+0.45）之间的预测位置——**"谱低维位置→迁移强度"不是 Qwen 系内部规律**。

**② n=9 子集 bit 级自洽（二）**：本次运行内 qwen-only 子集 sp=+0.8667 与 3086 参考值**完全一致**（spearman 为确定性统计量，统计管线同一性获证）；锚 a1（3082 sha 链 + sha8(S87)=32d3bf08）、a2（四模型谱带全过）、a3（shapes24 + forwards 20634/19770/20922）全 True。

**③ U_med 显著、MIG 方向一致但 ns（三）**：rho_U = **+0.7483（p=0.00605）显著**——头级与子集级响应一致；rho_MIG = +0.4545（p=0.139）ns——族级 R1 迁移参考聚合损失信息，不能声称族级迁移也跨架构单调。次预测器 s_mean：sp(T)=+0.9161 / sp(U)=+0.7552 / sp(MIG)=+0.4615。

### n=12 连续统阶梯（按 s_lo 排序）
| 模型 | s_lo 范围 | T_med 范围 | U_med 范围 | 谱类 |
| --- | --- | --- | --- | --- |
| DS7B | 0.266−0.293 | −0.021~+0.044 | −0.002~+0.034 | dispersed |
| GLM4 | 0.699−0.744 | +0.071~+0.118 | +0.006~+0.137 | mixed |
| 3B | 0.827−0.870 | +0.287~+0.447 | +0.485~+0.643 | mixed |
| 4B | 0.957−0.957 | +0.430~+0.448 | +0.415~+0.525 | trunk |

### 理论更新（第一性原理）
- **连续统升级为跨架构规律候选**（截至 4 架构样本）：谱位决定迁移容量——低 top3（读出被少数方向垄断）→弱头级迁移、高 top3→强迁移。GLM4 第一个非 Qwen 系谱点严格落带，候选解释从"Qwen 训练配方伪影"升级为"Transformer 语言编码的一般约束"；机制解释（为何低维谱限制迁移）待解剖类任务。
- **trunk 稀有 + 连续统跨架构 = 双重格局**：trunk+locked 迁移仍是 qwen3-4b 孤例（3082/3087），但强迁移不要求 trunk（3B mixed 也 +0.29~+0.45）——**谱形态决定"迁移容量"，trunk 决定"迁移锁定"**，两个自由度分离。
- **mixed 类内分辨力成立**：GLM4 与 3B 同类但谱位差 0.13，T_med 差 ~0.25——连续统对类内差异敏感，谱阈值两点校准的粗糙性被连续数值关系部分补偿。
- **MIG 与 T_med 解耦**：头级 CS1H 分解（T_med）显著而族级聚合（MIG）ns——迁移强度的高分辨读出在头级分解，族级单值掩盖结构。

### 硬伤与边界
- **GLM4 是单个非 Qwen 点**："跨架构"目前 = "1 个非 Qwen 系谱点符合"；本地无 Llama 等第三系模型，qwen3-14b 仍属 Qwen 系（只能作系内规模点）——普适化声明需保守分层。
- **L37/L38 判据分歧未决**（3087 遗留）：本检验消费 L37 n_neg 判据产物；若 L38（幅值判据）才是真 rescue 层，GLM4 单元值将变——3089 A（L38 仲裁复制）是本结论的前置压力测试。
- p=0.000000 为置换分辨率下限（cnt=0/20000，p<5e-5）；rho_MIG ns；24 对部分方向性协议、同一 3076 文本集（model independence 非 text independence）。
- 历史数据预注册：n=9 方向 3086 已知、GLM4 响应值 3087 冻结——预注册覆盖检验定义与判决树，非盲数据。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3088/omega_p86_continuum_n12/`：script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；免前向，elapsed %(el).1fs；ledger %(n)d / L14 %(l14)d。

**接续 3089 菜单**——A（主选·压力测试）**L38 仲裁复制**（GLM4 A2 管线 L_INJ=38，~21k 前向，判据分歧判别：L38 判决若≠L37，GLM4 单元值更新 + 连续统敏感性重跑）。B **G_DS 门敏感性**（免前向，3085 部分锁定 vs 3087 全灭的门结构对比）。C **4B 主干解剖**（免前向，CS top-1 成分的头/子集分解，服务"为何低维谱限制迁移"机制层）。D **层位×谱型**（GLM4 L34 补 E4 谱，中量前向）。E **qwen3-14b 第五谱点**（Qwen 系内规模点，新采集 ~20k 前向，检验连续统随规模的走向）。"好的，继续"即进 3089 A。
''' % {'created': created,
       'el': el,
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
if '## 五十、3088' not in aud:
    add = u'''
---
## 五十、3088 增补：连续统判别时刻 n=12——cross_architecture_confirmed
1. **判决**：GLM4 三单元纳入 3086 预注册框架（免前向、统计函数与 3086 逐字节同一），rho(s_lo, T_med) +0.8667（n=9）→ **+0.9021**（n=12，p<5e-5）——跨架构存活。
2. **GLM4 单元严格内插**：s_lo 0.699−0.744 → T_med +0.07~+0.12，恰在 DS7B（≈0）与 3B（+0.29~+0.45）之间；mixed 类内分辨力成立；qwen-only n=9 子集 bit 级复现 3086（+0.8667）。
3. **HDMCC 更新**："谱位→迁移强度"连续统升级为跨架构规律候选（4 架构、1 非 Qwen 点）；与 trunk 稀有性构成双格局（谱位决定容量、trunk 决定锁定）；rho_U +0.7483 显著、rho_MIG +0.4545 ns（族级聚合损失信息）。压力项：GLM4 单点外推保守 + L37/L38 判据分歧未决（3089 A 判别）。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-22.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3088' not in prev:
    line = ('- Phase 3088 Omega-P86 '
            'continuum n=12 discriminative '
            'test (forward-free): verdict '
            'cross_architecture_confirmed - '
            'rho_T +0.8667 (n=9) -> +0.9021 '
            '(n=12, p<5e-5); GLM4 units '
            'interpolate strictly between '
            'DS7B and 3B; qwen-only n=9 '
            'subset reproduces 3086 '
            'bit-level (+0.8667); rho_U '
            '+0.7483 sig, rho_MIG +0.4545 '
            'ns.  Audit 50; ledger '
            '227/L14 195.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W,
                      encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3088' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L 4kv 3584 bf16）、qwen2.5-3b-instruct（Qwen2 36L 16H 2kv 2048 bf16 tied）、glm4-9b-chat-hf（GlmForCausalLM 40L 32H 2kv 4096 inter 13696 vocab 151552 tied=False bf16 18.84GB>17.09GB VRAM 靠 sysmem fallback 非 OOM；repro 锚 bit 级通过）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout/verify tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\（smoke 在 smoke\\ 子目录）。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（**含旧 str 条目，必须 isinstance(c, dict) 防御**）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→独立 verify→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json（脚本自删亦可）；负结果与预注册退化/混合出口如实登记为一等公民——门未过禁止放宽（防 post-hoc）。
4. 统计纪律：阈值预注册；置换/偏相关 p 必须与主脚本 bit 一致；泛化声明分层（跨架构声明目前=4 架构 1 非 Qwen 点）。
5. verify 锚键先 Grep 主脚本 npz save 段逐一核对（布尔锚无 DIFF 分量）。
6. **确定性复现锚（3085 标准件）**：同模型同文本同层跨 Phase 重跑，n_neg/top8 精确相等、浮点差 ≤1e-9；mismatch→setup_failed。
7. **字符串 replace 先长后短**（'DS7B' 含 '4B' 子串）。
8. **patch 生成器/直写脚本 bad 清单必须含 OUT 路径 phase 号**；SMOKE 后必查产物目录名（3087 教训）。
9. **Edit 假成功频发**：改后必须 Grep 复核，可疑即重试。
10. **小源脚本可直写**（3088 教训）：源 ≤600 行且有全文时，直接 Write 新脚本 + 静态同一性断言（frozen 统计块逐字节比对）+ bad/want 清单，比生成器快且等价可靠；WANT 计数须先数真实文件（PREREG 跨字符串段拆分会少计字面 token）。

## 标准锚与精度
- bit 锚家族：…跨模型 setup 锚（3081）、免前向冻结重放锚（3082）、退化出口锚（3083）、层位扫描锚（3084）、复现锚+四格混合出口（3085）、连续统检验锚（3086）、第四谱点双臂锚（3087）、**跨架构连续统锚（3088：qwen-only n=9 子集 bit 级复现 3086 +0.8667；统计块逐字节同一性断言）**。

## 机制解释审计链
…→3082 trunk_migrates→3083 third_top8_degenerate→3084 layer_rescue→3085 third_mixed_absent→3086 continuum_confirmed→3087 A1 layer_rescue（判据分歧）+ A2 fourth_mixed_absent→**3088 cross_architecture_confirmed（连续统 n=12 跨架构存活，rho +0.9021；谱位定容量、trunk 定锁定双格局）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail/wc 坏→python + 写文件；日志用 Read；-c stdout 丢→写文件再 Read（偶发透传，勿依赖）。
- result.json 无 smoke 键，npz 里 SMOKE 标量为准。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 下一步
- max=3088，下一个 3089（A 主选 **L38 仲裁复制**——判据分歧判别、GLM4 单元值压力测试，~21k 前向；B G_DS 门敏感性免前向；C 4B 主干解剖免前向；D 层位×谱型；E qwen3-14b 第五谱点 Qwen 系内规模点）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars'
             % len(mem_new))
else:
    o.append('memory already max=3088')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
