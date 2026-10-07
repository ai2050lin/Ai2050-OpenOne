# -*- coding: utf-8 -*-
"""Phase 3066 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3066'
     r'\omega_p63_last_layer_flip_anatomy')
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
assert verdict == ('sign_flip_downstream_'
                   'distributed_heads_dist'), \
    verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a2b_diff'] < 0.0002
assert an['a2b_ok'] is True
assert an['b0_diff'] == 0.0 and an['b0_ok'] is True
assert an['b1_diff'] == 0.0 and an['b1_ok'] is True
assert an['b3_ok'] is True
assert an['b4_diff'] == 0.0 and an['b4_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_30_35'] == [
    0.5312391051160289, 0.36067533326696655,
    0.10106477801389738, 0.44953101303168347,
    0.1487826048372403, -0.35579100779974404]
assert st['c_raw_30_35'] == [
    -0.12307487428188324, -0.34938569366931915,
    -0.3021078258752823, -0.40910688042640686,
    0.29685845971107483, 0.19866131991147995]
assert st['c_full_diag_30_35'] == [
    0.02859259769320488, -0.3170311897993088,
    0.00018262215598952025,
    -0.02663557231426239, 0.2371114194393158,
    -0.3558848798274994]
assert st['raw_pos_med'] == -0.3021078258752823
assert st['med_c_34'] == 0.1487826048372403
assert st['c_sil35'] == 0.1098812109126209
assert abs(st['share35']
           - 0.08627682309235352) < 1e-12
assert abs(st['sil_cum']['25']
           + 0.39457944324116356) < 1e-12
assert abs(st['sil_cum']['30']
           - 0.4777513834139916) < 1e-12
assert abs(st['c_unnorm_35']
           + 0.29030826233179635) < 1e-12
assert abs(st['head_top1']
           - 0.15261208429917472) < 1e-12
assert st['head_arg_layer'] == 34
assert st['head_arg_h'] == 29

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3066
           for m in led['measurements']):
    claim = (
        'Omega-P63 (plan 3066 A) - qwen3-4b bf16 '
        'last-layer flip anatomy (110.3s, ~1813 '
        'forwards; anchors: a1 full ladder bit-'
        'exact vs 3065 npz diff 0.0; a2b lens '
        'calibration 9.4e-05; b0/b1/b4 bit 0.0; '
        'b3 finite; a2 fp32 lens within 1 bf16 '
        'ulp). Debug history: smoke caught 3 '
        'script bugs before the authoritative '
        'run (HDIM order; LOG path under SMOKE; '
        'PROP rows overwritten per pair inside '
        'the k-loop - single-pair value instead '
        'of the 24-pair median) plus one '
        'precision bug (bf16 linear output '
        'rounding pollutes lens cosines at '
        '~0.25 level; fixed by resident fp32 '
        'W32 = Wemb.float()). RESULTS: verdict '
        'sign_flip_downstream_distributed_heads_'
        'dist. (1) Raw V-readout polarity C_RAW '
        'swings the FULL range with no fixed '
        'sign (-0.41..+0.44); in the positive '
        'band L30-34 it is NEGATIVE in 4/5 '
        'layers (median -0.302) - the positive '
        'ladder values are NOT raw readout '
        'polarity. (2) Lens chain (x -> h1=x+a '
        '-> h2, final_norm + fp32 unembed): '
        'full_diag at the injection-layer output '
        'is ~0/negative for L30-33 (+0.03/-0.32/'
        '0.00/-0.03) while the final med_c is '
        '+0.53/+0.36/+0.10/+0.45 - the '
        'positivization is MANUFACTURED '
        'DOWNSTREAM, distributed over the '
        'propagation path. (3) L35 direct '
        'injection: raw +0.199 (attention writes '
        'a POSITIVELY aligned displacement) -> '
        'final -0.356: the endpoint negative '
        'sign is written INSIDE the L35 block, '
        'mainly by the MLP value channel (E4 '
        'c_unnorm -0.290 shows the final norm '
        'only deepens it) - and the SAME L35 '
        'processor PRESERVES the L34-propagated '
        'positive displacement (+0.237 -> '
        '+0.149): the MLP response is STATE-'
        'DEPENDENT, opposite signs for opposite '
        'input conditions. (4) Causal silencing: '
        'V0@35 (zero the V rows of positions '
        '0..3 at L35) removes only 8.6 pct of '
        'med_c(34) (0.149 -> 0.110) - the L35 '
        'readout is a minor carrier; for the '
        'negative band l=25 one-at-a-time '
        'silencing is pass-through but '
        'CUMULATIVE silencing STRENGTHENS the '
        'negative (-0.113 -> -0.395): downstream '
        'attention readouts partially COUNTERACT '
        'the negative band. (5) Head map: top1 '
        'share 0.153 (L34 h29) / 0.065 (L35 h31) '
        '- distributed. Conclusion: the sign '
        'orchestration decomposes as raw '
        'per-layer polarity (layer-specific, '
        'no fixed sign) TIMES state-dependent '
        'downstream response field; there is '
        'no single sign-carrying component - '
        'the unit of orchestration is the '
        '(component, state-condition) pair.')
    meas = {
        'meas_id': 'meas3066_omega_p63_last_'
                   'layer_flip_anatomy',
        'phase': 3066,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 ladder bit 0.0 vs 3065 '
                   'npz (hard); a2 fp32 lens '
                   '<= 1 bf16 ulp (bit attempt '
                   'recorded); a2b 9.4e-05; '
                   'b0/b1/b4 bit 0.0; b3 '
                   'finite; b5 V0@35-on-base '
                   '||dlg||=127.0 recorded',
        'artifacts': {
            'result': 'phase3066/omega_p63_'
                      'last_layer_flip_anatomy/'
                      'result.json',
            'npz': 'phase3066/omega_p63_'
                   'last_layer_flip_anatomy/'
                   'omega_p63_last_layer_flip_'
                   'anatomy.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16; smoke mode '
                'caught 3 script bugs + 1 '
                'precision bug before the '
                'authoritative launch)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 205
    l14['connects'].append({
        'meas_id': 'meas3066_omega_p63_last_'
                   'layer_flip_anatomy',
        'phase': 3066,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P63: last-layer '
                        'flip anatomy - the '
                        'positive band L30-34 is '
                        'manufactured downstream '
                        '(raw polarity negative in '
                        '4/5 band layers, median '
                        '-0.302; L35 readout '
                        'share only 8.6 pct) and '
                        'the L35 endpoint negative '
                        'is written inside the L35 '
                        'block by a STATE-DEPENDENT '
                        'MLP response (+0.199 raw '
                        '-> -0.356 final for '
                        'direct injection, but '
                        'preserves the L34-'
                        'propagated +0.237 -> '
                        '+0.149); no single sign '
                        'carrier - orchestration '
                        'unit = (component, '
                        'state-condition) pair. '
                        'Opens 3067 A: L35 MLP '
                        'conditional reversal '
                        'anatomy'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
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
if '## Phase 3066:' not in memo:
    sec = u'''## Phase 3066: Ω-P63 末层翻转机制解剖——正带=分布式下游正化，L35 负号=块内 MLP 条件反转（sign_flip_downstream_distributed_heads_dist） [%(created)s]

**判决：`sign_flip_downstream_distributed_heads_dist`**（qwen3-4b 单模型 bf16 110.3s，约 1813 次前向；锚：**a1：36×24 全阶梯与 3065 npz diff=0.000e+00——跨脚本 bit 级一致**；a2b lens 校准 9.4e-05；b0/b1/b4 bit 0.0；b3 finite；a2 fp32 lens ≤1 bf16 ulp）。调试史如实入册：smoke 模式在权威启动前抓住 3 个脚本 bug（HDIM 定义顺序；SMOKE 下 LOG 路径错位；**PROP 矩阵在 k 循环内被逐对覆盖——存成了第 24 对单对值而非 24 对中位**）+ 1 个精度 bug（**bf16 linear 输出舍入把 lens cos 污染 ~0.25 级**——常驻 W32=Wemb.float() fp32 消除）。无崩溃，smoke→权威一次通过。

### 问题与设计
**问题**（3066 A，3065 菜单）：qwen3-4b 阶梯正带 L30-34（+0.10..+0.53）vs L35 −0.356 的"末层翻转"写在哪里——注入层原始读出极性、下游 attention 对注入位置的读出、MLP 重塑、还是 final norm？设计：E1 全 36 层阶梯复跑（协议 3065 逐字一致→a1 bit 锚），每前向捕获末位四元组（块输入 x、attn 输出 a、mlp 输出 m、块输出 h2），h1=x+a（bf16 加，bit 等于模型内部），lens 探针（final_norm bf16 + fp32 unembed 矩阵乘）把三点映射 logit 空间：**C_RAW[l]=PROP_ATTN[l,l]（原始 V 读出极性）**、PROP_FULL[l,lp]（逐层 lens 对齐矩阵）；E2 因果静默 V0（下游层位置 0..3 的 V 行置零，l∈{25,30,34}，逐层+累积，每变体配 matched base-V0 对照，fp64 cosv）；E3 头级贡献图（o_proj 输入捕获，描述性）；E4 norm 通道（无 norm 的 c_unnorm 对照）。

### 核心结果（重复三遍）
**① 原始读出极性无固定符号（一）**：C_RAW 全谱摆动 −0.41..+0.44；正带 L30-34 内 **4/5 层为负**（L30 −0.123 / L31 −0.349 / L32 −0.302 / L33 −0.409 / L34 +0.297，中位 **−0.302**）——**正带的"正"不是原始极性**。**② 正化在下游分布式制造（二）**：注入层输出处的即时对齐 full_diag 在 L30-33 ≈0/负（+0.03/−0.32/0.00/−0.03），而最终 med_c=+0.53/+0.36/+0.10/+0.45——**正效应由 l→35 的传播逐步制造**；因果份额：V0@35 静默 L35 读出仅移除 med_c(34) 的 **8.6 pct**（0.149→0.110）——L35 读出是小载体，正效应主体在 L34 及更早已写入末位残差。**③ L35 负号=块内 MLP 条件反转（三遍）**：L35 直接注入 raw=**+0.199**（attn 写入正对齐位移）→ MLP 后 **−0.356**；而同一 L35 处理器对 L34 传播来的 +0.237 位移**保持正**（→+0.149）——**同一 MLP 对不同状态条件输出相反符号**；E4 c_unnorm(35)=−0.290（无 norm 亦负）——final norm 只加深负性、翻转主体在 MLP 值通道。附加：负带 l=25 逐层静默≈透传（−0.11..−0.20）而**累积静默使负效应更强（−0.113→−0.395）**——下游 attn 读出对负带是部分抑制性的；头图 top1 share 0.153（L34 h29）/0.065（L35 h31）分布式。

### 机制综合（Ω-P63 拼图）
符号编排 = **逐层原始极性（层特异、无固定符号）× 状态依赖下游响应场** 的合成。不存在单一"符号载体"组件：L35 的 MLP 是本 phase 最接近"翻转点"的组件，但它的符号作用是**相对于输入状态**的（对直接注入位移反转、对传播位移保持）。这把 3065 的"符号住计算流"具体化为：**编排的最小单位不是组件，而是（组件, 状态条件）对**——条件齿轮组的"齿"是组件-条件耦合项，不是零件本身。

### 硬伤与边界
- lens 是探针：末层=精确（a2b 9.4e-05 背书），中间层是 logit-lens 解释，非模型实际计算路径；C_RAW/PROP 为 fp32 GPU 量（h1 重构在 bf16）。
- share35 的分母含 lens probe 值（raw_pos_med）；E2 累积静默连锁改变下游输入分布——份额读作"界"而非精确分解。
- 单 prompt 族（style/connector）；单模型；E3 头图描述性（无因果消融）；COS_LAD 的 24 对中位含 L24/L3 等弱层的高方差对。

### 方法论入册
- **内层循环累积后 median**：逐对覆盖矩阵行是本 phase 最深 bug——任何"per-k 计算+聚合"都必须显式累积（COS_LAD 模式）。
- **bf16 linear 的输出舍入会以 ~0.25 级污染 cos 探针**：探针级矩阵乘用 fp32 权重副本（Wemb.float() 常驻 +1.55 GB），模型自身路径保持 bf16。
- **matched 对照**：消融变体（V0）必须配同图案 base 前向，否则消融效应与注入效应混叠。
- **SMOKE 快速链路验证**（子集层+全锚）在权威启动前抓住全部 4 个 bug——零崩溃达成。

### 智能理论洞察（第一性原理）
3065 问"编排由什么自由度承载"，3066 给出第一个否定性答案：**不是任何单一组件**（头分布式、读出份额 8.6 pct、MLP 符号作用状态依赖）。正面发现：**L35 MLP 的条件反转**是"相对编码"的直接证据——同一参数组（L35 MLP 权重）在不同残差状态下对同类扰动（V 签名位移）输出相反方向的读出效应。这支持把功能定义为 f(组件, 状态) 而非 f(组件)：**语言能力的"机制"在组件层面必然欠定，只有在组件×状态耦合层面才充分**。下一层剖析对象由此明确：L35 MLP 的门控/下游投影如何读出"输入状态"来决定符号作用——这是"条件齿轮"的第一个具体参数化问题。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3066/omega_p63_last_layer_flip_anatomy/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3067 菜单**——A（主选）**L35 MLP 条件反转解剖**：定位 L35 MLP 对"直接注入位移 vs 传播位移"符号相反的参数载体（下游投影行分解/门控激活对比/两状态下 MLP 输出方向的夹角）；B **正化路径全谱追踪**：PROP 36×36 矩阵分析，定位 31-35 中把 ≈0/负位移拉正的主贡献层（逐层 lens 增量归因）；C **DS7B 传播翻转溯源**（3065 菜单 C 遗留：late_med_flip=1.0 的层定位，与 qwen L35 MLP 反转对照）；D **头级因果消融**：对 h29/h31 做 V0-silence 验证描述性头图；E **跨 prompt 族泛化**：新语义族复测阶梯，检验符号编排的域稳定性。"好的，继续"即进 3067 A。
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
if '## 二十八、3066 增补' not in aud:
    add = u'''

---

## 二十八、3066 增补：末层翻转解剖——正带=下游分布式正化，L35 负号=块内 MLP 条件反转（Omega-P63，判决 sign_flip_downstream_distributed_heads_dist）

1. **原始极性无固定符号**：C_RAW（注入层 V 读出的 lens 对齐）全谱 −0.41..+0.44；正带 L30-34 内 4/5 层为负（中位 −0.302）——阶梯正带的"正"不是注入层写入的极性，而是下游制造的。
2. **分布式正化 + L35 读出小份额**：注入层输出处 full_diag 在 L30-33 ≈0/负而最终为正；V0@35 因果静默仅移除 med_c(34) 的 8.6 pct——正效应主体在 L34 及更早已写入末位残差。
3. **L35 负号=MLP 条件反转**：L35 直接注入 raw +0.199 → 最终 −0.356（E4 证 final norm 只加深、翻转在 MLP 值通道）；同一 L35 对 L34 传播位移 +0.237 保持正（→+0.149）——同一组件对不同状态条件输出相反符号，"相对编码"的组件级直接证据。
4. HDMCC 修正：符号类结论的归因单元从"组件"升级为"（组件, 状态条件）对"；单一组件消融（头/MLP 二分）不足以裁决符号载体，必须做状态依赖响应分析。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-21.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3066' not in prev:
    line = ('- Phase 3066 Omega-P63 last-layer '
            'flip anatomy (qwen3-4b bf16 single, '
            '110.3s): verdict '
            'sign_flip_downstream_distributed_'
            'heads_dist. a1 full ladder bit-exact '
            'vs 3065 npz diff 0.0; a2b lens calib '
            '9.4e-05. C_RAW swings -0.41..+0.44 '
            'with no fixed sign; positive band '
            'L30-34 raw is negative in 4/5 layers '
            '(median -0.302) - positiveness '
            'manufactured downstream (V0@35 share '
            'only 8.6 pct). L35 endpoint negative '
            '= block-internal MLP conditional '
            'reversal (raw +0.199 -> -0.356 for '
            'direct injection; preserves L34-'
            'propagated +0.237 -> +0.149) - same '
            'component, opposite signs for '
            'opposite state conditions. Negative '
            'band l=25: cumulative downstream '
            'silencing strengthens it (-0.113 -> '
            '-0.395). Head map distributed (top1 '
            '0.153). Smoke mode caught 3 script '
            'bugs (PROP per-pair overwrite '
            'deepest) + 1 precision bug (bf16 '
            'linear rounding pollutes lens cos '
            '~0.25; fixed by fp32 W32 resident). '
            'Audit 28; ledger 205/L14 173.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W, encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3066' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃如实登记；verdict 单分支赋值。
4. 统计纪律：obs/null 同量纲；负对照带符号解读；阈值预注册；随机对照判特异性。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨脚本 bit 级锚（a4/a1：两份独立实现 diff=0.0）是"新脚本=旧机制"最强验证。
- **3066**：per-k 聚合必须显式累积后 median（内层循环覆盖=单对值 bug）；cos 探针矩阵乘用 fp32 权重副本（bf16 输出舍入污染 ~0.25 级）；消融变体必须配 matched base；SMOKE 子集先验证链路再权威跑。

## 机制解释审计链（命名前依次检查）
…→KV 阶梯→门位易感→γ 管道→PC1→写入分解→3063 竞争轴→3064 跨模型（拓扑普适/符号特异）→3065 符号溯源（符号住传播动力学）→**3066 翻转解剖：C_RAW 无固定符号（band 中位 −0.302）；正带=下游分布式正化（V0@35 份额 8.6 pct）；L35 负号=MLP 条件反转（raw +0.199→−0.356；对传播位移保持正）；编排单位=（组件,状态条件）对**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核；幻影 Edit 会再现——Python 补丁 assert count==1 唯一可靠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3066）
Ω-P2（3011-3066）：3045-3048 KV 阶梯；3050 末层；3051 K 场+头；3052-3053 门槽；3054-3057 γ 重整/预对齐/反方差；3060 PC1 身份；3061 写入分解；3063 竞争轴；3064 chain_fragmented（拓扑普适/符号特异）；3065 sign_decoupled_all3（符号住传播动力学）；**3066 sign_flip_downstream_distributed_heads_dist（正带=分布式正化；L35 MLP 条件反转；无单一符号载体）**。

## 下一步
- max=3066，下一个 3067（A 主选 **L35 MLP 条件反转解剖**：符号相反的参数载体；B 正化路径全谱追踪；C DS7B 传播翻转溯源；D 头级因果消融；E 跨 prompt 族泛化）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3066')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
