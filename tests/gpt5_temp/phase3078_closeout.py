# -*- coding: utf-8 -*-
"""Phase 3078 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3078'
     r'\omega_p75_routing_timing')
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
assert verdict == 'routing_signal_absent', \
    verdict
assert res['forwards'] == 99
assert res['smoke'] is False
an = res['anchors']
assert an['a2_ok'] is True
assert an['a2_diff'] == 0.0
for fk in ('A', 'B', 'C'):
    assert an['a1_ok'][fk] is True, fk
    assert an['a1_diff'][fk] == 0.0, fk
    assert an['a1m_diff'][fk] == 0.0, fk
    assert an['a0_ok'][fk] is True, fk
    assert an['b0_ok'][fk] is True, fk
    assert an['a0_lg'][fk] == 0.0, fk
st = res['stats']
si = st['sp_intra']
assert abs(si[6][2]
           - 0.22067448680351906) < 1e-15
assert abs(si[7][0]
           - 0.1748533724340176) < 1e-15
assert abs(si[7][1]
           - 0.16972140762463342) < 1e-15
assert abs(si[7][2]
           - (-0.06048387096774194)) < 1e-15
sc = st['sp_cross']
assert abs(sc[7][0]
           - 0.9450146627565983) < 1e-15
assert abs(sc[7][1]
           - 0.8673020527859238) < 1e-15
assert abs(sc[7][2]
           - 0.8782991202346041) < 1e-15
icc = st['icc']
assert abs(icc[0]
           - 0.6653109170636378) < 1e-15
assert abs(icc[6]
           - 0.9450708435694475) < 1e-15
assert abs(st['sp_d34b'][1]
           - (-0.36107038123167157)) < 1e-15
pp = st['pp_intra']
assert abs(min(min(r) for r in pp)
           - 0.22625) < 1e-12
g = res['gates']
assert g['S_layers'] == []
assert g['early_layers'] == []
assert abs(g['global_max_abs_sp']
           - 0.22067448680351906) < 1e-15

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3078
           for m in led['measurements']):
    claim = (
        'Omega-P75 (plan 3078 A) - qwen3-4b '
        'routing-decision TIMING scan (99 base '
        'forwards, 18.8s, NO injections): at '
        'which layer does the routing signal '
        '(the head x family causal pattern R1, '
        '3077-quantified as interaction-driven '
        'with ICC 0.45 vs 0.87 observational) '
        'become readable from the BASE '
        'representation?  Design: 3 families x '
        '32 base prompts forwarded; last-'
        'position o_proj INPUT zH_l (per-head '
        'concat) captured at L24/26/28/30/31/32/'
        '33/34/35; per head h and pair k the '
        'head write is projected on the family '
        'TT direction, dAh_l[k,h] = ((zh_l[h] @ '
        'WoT_h) @ W32 @ TT64[k]) / ||TT64[k]|| '
        '(3076 head_decomp math bit-copied); '
        'readout spectrum Dm_l[f] = median_k; '
        'TT64/LG64 rebuilt float64 from fresh '
        'forwards (3076 npz stores only a '
        'float32 TT copy).  VERDICT '
        'routing_signal_absent.  (1) ALL-'
        'LAYER ABSENCE: intra-family '
        'spearman(|Dm_l|, R1) over 9 layers x 3 '
        'families has global max |sp| = 0.221 '
        '(L33 family C), all permutation p >= '
        '0.226 (20000 perms, seed 3078) - '
        'nothing approaches the preregistered '
        '0.35/0.05 gates anywhere from L24 to '
        'L35; the 3077 observation-causation '
        'decoupling now covers the ENTIRE time '
        'course.  (2) AMPLITUDE CONTROL ALSO '
        'NULL: spearman(median head write '
        'norm, R1) max |sp| = 0.249 - absence '
        'is not a mask-by-amplitude artifact.  '
        '(3) SPECTRA ARE RICH BUT ORTHOGONAL: '
        'the base readout spectra are highly '
        'cross-family similar (sp_cross '
        '|Dm_l[f]| vs |Dm_l[g]| = 0.36-0.95, '
        'peaking at L34 0.945/0.867/0.878) '
        'with layerwise ICC_head rising 0.665 '
        '-> 0.945 (L33) - the heads have a '
        'stable, family-SHARED write-profile '
        'structure that is simply NOT what '
        'causal focality is made of; visible '
        'signal is abundant yet orthogonal to '
        'the causal variable.  (4) BASE vs '
        'DELTA: base spectrum vs injection-'
        'delta spectrum (|Dm_34| vs |DAH34_MED'
        '|) sp = -0.201/-0.361/+0.074 - even '
        'the two observational views disagree. '
        'ANCHORS: a1 per-k DAH34/35 recomputed '
        'from (3076-npz ZH34/35) - (fresh base '
        'zH at L34/35) vs 3076 npz rows: bit '
        '0.0 in all 3 families (proves BOTH '
        'that 3078 base forwards are bit-'
        'identical to 3076 banks AND that the '
        '3078 headproj math is bit-identical); '
        'a1m medians bit 0.0; a2 R1_ALL32_A vs '
        '3071 r34 bit 0.0; a0 provenance LG64-'
        'LG32_npz = 0.0 (logits are fp32-'
        'valued, lossless) and TT64-TT32 ~2.4e-'
        '07 (pure float32 storage rounding); '
        'b0 in-process recapture bit 0.0.  '
        'SMOKE story: first smoke FAILED '
        'family C anchors (a0 LG diff 7.1, a1 '
        'diff 3.5e2) - the C bodies had been '
        'typed from memory instead of the '
        '3076 source; bit anchors caught it.  '
        'Also fixed B prefix #4 (Regarding the '
        'experiment) which smoke could NOT '
        'catch (smoke forwards only ci<=1 '
        'prompts) - lesson: provenance anchors '
        'must cover every sample that enters '
        'the statistics.  Model: the routing '
        'signal does NOT pre-exist in the base '
        'representation in linearly readable '
        'form at ANY layer; focality is '
        'generated at injection time by the '
        'injected state interacting with head '
        'computation (3073 higher-order '
        'conjecture now the leading candidate), '
        'not read out from pre-existing '
        'content.')
    meas = {
        'meas_id': 'meas3078_omega_p75_'
                   'routing_timing',
        'phase': 3078,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'bit anchors all 3 families: '
                   'a1 per-k DAH34/DAH35 '
                   'recompute vs 3076 npz rows '
                   '(frozen ZH34/35 minus fresh '
                   'base zH, headproj math '
                   '3076-identical) diff 0.0; '
                   'a1m medians diff 0.0; a2 '
                   'R1_ALL32_A vs 3071 r34 diff '
                   '0.0; a0 provenance LG64 vs '
                   'LG32_npz diff 0.0 and TT64 '
                   'vs TT32_npz <= 2.4e-07; b0 '
                   'in-process recapture 0.0',
        'artifacts': {
            'result': 'phase3078/omega_p75_'
                      'routing_timing/'
                      'result.json',
            'npz': 'phase3078/omega_p75_'
                   'routing_timing/'
                   'omega_p75_routing_timing.'
                   'npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (99 base '
                'forwards, 18.8s, no '
                'injections).  Smoke run 1 '
                'caught fabricated family-C '
                'texts via a0/a1 bit anchors '
                '(LG diff 7.1); B prefix-4 '
                'error found by source '
                're-read and was invisible to '
                'smoke (ci<=1 only) - '
                'provenance anchors must '
                'cover all statistical '
                'samples.  Linear-readout '
                'caveat: only TT-direction '
                'projections and norms tested; '
                'a nonlinear routing signal '
                'would be invisible by '
                'design.  Families share the '
                'syntactic frame, so the '
                'high cross-family spectrum '
                'similarity may be syntax-'
                'driven.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 217
    l14['connects'].append({
        'meas_id': 'meas3078_omega_p75_'
                   'routing_timing',
        'phase': 3078,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P75: routing-'
                        'decision timing scan '
                        '(99 base forwards, no '
                        'injections).  Base '
                        'per-head TT-projection '
                        'spectra Dm_l tested vs '
                        'R1 at L24/26/28/30/31/'
                        '32/33/34/35: global max '
                        '|sp| 0.221, all perm p '
                        '>= 0.226 - NO layer '
                        'carries the routing '
                        'signal (gates 0.35/0.05 '
                        'preregistered).  '
                        'Amplitude control null '
                        '(0.249); spectra cross-'
                        'family similar (L34 '
                        '0.945/0.867/0.878) with '
                        'ICC rising to 0.945 - '
                        'rich SHARED structure, '
                        'orthogonal to causality. '
                        'Base-vs-delta spectra '
                        'also disagree (-0.201/-'
                        '0.361/+0.074).  '
                        'Observation-causation '
                        'decoupling now spans '
                        'the full time course: '
                        'focality is generated at '
                        'injection time, not '
                        'pre-existing readable '
                        'content.  Bit anchors: '
                        'a1 per-k DAH recompute '
                        '0.0 (fresh base forwards '
                        '== 3076 banks + '
                        'headproj math), a1m/a2 '
                        '0.0.  Opens 3079: A '
                        'spectral-shape similarity '
                        '-> migration strength '
                        '(no forwards); B h14 '
                        'anatomy; C DS7B cross-'
                        'model control; D '
                        'injected-state higher-'
                        'order interaction '
                        'spectrum'})
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
if '## Phase 3078:' not in memo:
    sec = u'''## Phase 3078: Ω-P75 路由决定时序定位——基座全层线性读出不携带路由信号（routing_signal_absent） [%(created)s]

**判决：`routing_signal_absent`**（**99 次基座前向、18.8 秒、零注入**：注入在 L34 内部 ⇒ L34 之前基座=注入态 ⇒ 路由信号若先于注入存在必在基座表示中——本 phase 把 3077 的"观察-因果解耦"从单层 L34 扩展到**全时序 L24→L35**。锚：**a1 逐 k DAH34/35 重算三族 bit 0.0**（用 3076 npz 冻结 ZH34/35 减 3078 新基座 zH、3076 headproj 数学逐位复刻——同时证明 3078 基座前向与 3076 banks bit 一致 + 数学实现一致）、a1m 中位 bit 0.0、a2 R1_ALL32_A=3071 r34 bit 0.0、a0 溯源（LG64−LG32_npz=0.0：logits 本是 fp32 值无损存储；TT64−TT32≈2.4e-07 纯 float32 存储舍入）、b0 进程内重捕 bit 0.0）。

### 问题与设计（3078 A，3077 菜单主选）
**问题**：路由信号（R1 的头×族特异模式，ICC=0.45、12 静态特征全败）**在哪一层变得可读**？早层可读 ⇒ 路由预先写在基座表示中；仅末层可读 ⇒ 随注入出现；全层不可读 ⇒ 路由是注入时刻的局部计算。**设计**：三族 × 32 基座前向，捕获 L24/26/28/30/31/32/33/34/35 共 9 层 last 位置 o_proj 输入 zH_l（每头 128 维 concat）；对每对 k 每头 h：dAh_l[k,h] = ((zh_l[h] @ WoT_h) @ W32 @ TT64[k]) / ‖TT64[k]‖（3076 head_decomp 数学逐位复刻）；族读出谱 Dm_l[f] = median_k。**TT64/LG64 必须从新前向重建 float64**——3076 npz 只存了 float32 TT 副本，而其 E2H 计算用的是 float64 TT，bit 锚要求精确复刻。统计：族内 sp(|Dm_l|, R1) 逐层曲线 + 20000 次置换 p（seed 3078）；跨族谱相似性；逐层 ICC；幅度对照（头写入范数 vs R1）；基座谱 vs 注入差异谱关系。判决门（预注册）：S = {l : min_f |sp|≥0.35 且 max_f pp<0.05}；early（S 含 l≤32）/ late（S 全≥33）/ absent（S 空 且全局 max|sp|<0.35）/ partial。

### 核心结果（重复三遍）
**① 全层无路由信号（一）**：9 层 × 3 族的族内 spearman(|Dm_l|, R1) **全局最大 |sp| = 0.221**（L33 族 C），全部置换 p ≥ 0.226——离预注册门（0.35/0.05）很远；L24 [0.207, −0.071, −0.070]、L34 [0.175, 0.170, −0.060]、L35 [−0.005, −0.088, 0.056]，无任何层位出现信号。**② 幅度对照同样为零（二）**：sp(头写入范数中位, R1) 全局 max|sp| = 0.249（L34 族 A）——"信号被幅度淹没"的假阴性解释排除。**③ 谱结构丰富但与因果正交（三）**：基座读出谱跨族高度相似（sp_cross = 0.36–0.95，**L34 达 0.945/0.867/0.878**），逐层 ICC_head 从 0.665 升至 0.945（L33）——头间存在稳定、**跨族共享**的写入轮廓结构，但它与因果焦点性无关；**可见信号丰富却与因果量正交——观察-因果解耦的最强形式：不是信号弱看不见，而是看得见的都无关**。**④ 基座谱 vs 注入差异谱也不一致**：sp(|Dm_34|, |DAH34_MED|) = −0.201/−0.361/+0.074——两种观察视角彼此都不对齐。

### 数学公式
- 头级读出投影：dAh_l[k,h] = ((zh_l[h] @ WoT_h) @ W32 @ TT64[k]) / ‖TT64[k]‖，WoT = o_proj.W^T reshape (32,128,2560)（每头 W_O 块），W32 = 输出嵌入 fp32；
- 族读出谱：Dm_l[f][h] = median_k dAh_l[k,h]；可读性：sp_l,f = spearman(|Dm_l[f]|, R1_f)；
- 判决量：gmax = max_{l,f} |sp_l,f| = 0.221（L33，C）；S 层集 = ∅。

### 硬伤与边界
- **线性读出只测线性可读信号**：TT 方向投影与范数是线性量——若路由信号是非线性的（注入态×头权重的高阶交互），本设计按构造不可见（PREREG limitation 预注册）。
- 三族共享句法框架（8 body × 4 prefix 同构），跨族谱相似 0.95 可能由句法/位置结构主导——族特异内容信号若存在也可能被句法共享稀释。
- n=32 头 spearman 功效有限（0.35 门对应 p≈0.05）；9×3 检验未做多重校正（min-over-families 要求为主守门）。
- TT 方向本身是族内 pref−base logits 差；基座对它的投影不必然携带 pref 信息——这正是被检验的假设，结论是"不携带"。

### 方法论入册
- **bit 锚抓住文本伪造**：smoke 第一轮族 C 全锚失败（a0 LG 差 7.1、a1 差 3.5e2）——族 C 文本是凭记忆打字而非 3076 原文；bit 锚系统精确定位。
- **smoke 覆盖盲区教训**：B 族第 4 prefix 错误 smoke 抓不到（smoke 只前向 ci≤1 的 prompt，a0 溯源只覆盖前向子集）——**溯源锚必须覆盖所有进入统计的样本**；权威运行前逐字核对源文本。
- **float64/float32 双版本陷阱**：上游 npz 只存 float32 降采样副本时，bit 锚必须从 float64 原量重建（TT64/LG64 从新前向重建，logits 本为 fp32 值故 LG 无损、TT 差有 float32 舍入）。

### 智能理论洞察（第一性原理）
**因果角色是关系性质，不是内在性质。** 基座表示在全部 9 层上都有丰富、稳定、跨族共享的头级写入结构（ICC 至 0.945、跨族相似至 0.945），但其中不含有 R1 的任何线性可读痕迹（max 0.221）——"谁上场"不由"头上写了什么"决定。结合 3077（ICC 观察 0.865 vs 因果 0.452、12 特征全败、符号翻转 p=0.0096），三层证据汇成同一个结论：**LLM 用参数编码"如何读"（读出结构固有且共享），用"注入态×头计算的非线性交互"编码"谁上场"（路由=语境时刻局部生成）**。这类似于神经科学的增益调制（gain modulation）：通道内容不变，调制信号实时决定哪个通道被放大——但 LLM 的调制信号不驻留在任何单层基座表示中，它是注入事件本身与权重的乘积。对 AGI 理论：条件化齿轮组的**装配函数是事件驱动的，不是状态查表的**——这把搜索空间从"表示内容"（L0-L35 全部排除线性可读部分）压缩到"交互项"（3073 higher_order_required 假说升格为首候选）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3078/omega_p75_routing_timing/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；99 forwards / 18.8s。

**接续 3079 菜单**——A（主选）**谱形相似性→迁移强度**：三族 255 子集谱形（3076 npz A_S/R_S/CS/MASKS）两两 + 对 3074 谱形相似，检验"族间谱形相似度是否预测 R1 迁移强度（A→B 0.676 vs A→C 0.130）"——把迁移不对称的来源定位到谱形结构（免前向）。B **h14 全域解剖**：唯一跨族核心头的逐族谱形/OV top tokens/DOH/DHH（免前向为主）。C **DS7B 跨模型对照**：routing_signal_absent + 读出固定写入路由的跨模型复现检验。D **注入态高阶交互谱**：路由信号最后藏身处的直接检验（3073 假说延伸，需前向）。"好的，继续"即进 3079 A。
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
if '## 四十、3078 增补' not in aud:
    add = u'''
---
## 四十、3078 增补：路由决定时序定位（Omega-P75，判决 routing_signal_absent）
1. **全层零信号**：三族基座 per-head TT 投影谱（L24-L35 共 9 层）对因果谱 R1 的族内 spearman 全局 max|sp|=0.221、置换 p 全≥0.226（门 0.35/0.05）——路由信号在基座表示的任何层都线性不可读；3077 的观察-因果解耦扩展到全时序。
2. **结构丰富但正交**：谱跨族相似高达 0.945（L34）、ICC 升至 0.945（L33），幅度对照也零（0.249）——可见信号丰富却与因果量正交，排除"信号弱"解释。
3. **bit 锚体系**：a1 逐 k DAH34/35 重算三族 0.0（冻结 ZH34/35 − 新基座 zH + 逐位复刻 headproj）；a0 溯源 LG64−LG32=0.0（fp32 logits 无损）；smoke 抓住凭记忆打字的族 C 文本伪造；教训=溯源锚必须覆盖所有进入统计的样本（B prefix-4 错误 smoke 盲区）。
4. HDMCC 更新：路由=事件驱动装配（注入态×头交互），非状态查表；3073 higher-order 假说升格为首候选；线性读出边界预注册。
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
if 'Phase 3078' not in prev:
    line = ('- Phase 3078 Omega-P75 routing-timing '
            'scan (99 base forwards, 18.8s, no '
            'injections): verdict '
            'routing_signal_absent.  Base '
            'per-head TT-projection spectra at '
            'L24-L35 vs R1: global max|sp|=0.221, '
            'all perm p>=0.226 - no layer '
            'carries the routing signal; '
            'amplitude control null (0.249); '
            'spectra cross-family similar up to '
            '0.945 with ICC to 0.945 (rich '
            'shared structure, orthogonal to '
            'causality); base-vs-delta spectra '
            'disagree (-0.201/-0.361/+0.074).  '
            'Focality is generated at injection '
            'time (injected state x head '
            'interaction), not pre-existing '
            'readable content.  Bit anchors: a1 '
            'per-k DAH recompute 0.0 (3 '
            'families), a1m/a2 0.0, a0 '
            'provenance LG diff 0.0; smoke '
            'caught fabricated family-C texts '
            '(LG diff 7.1); B prefix-4 error was '
            'a smoke blind spot (ci<=1 only).  '
            'Audit 40; ledger 217/L14 185.\n')
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
if 'max=3078' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（link_id=L14_readout_spectrum_cross_model；verify 需 isinstance 防御）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃/smoke 推翻均如实登记；verdict 单分支赋值。
4. 统计纪律：阈值预注册；**跨族竞标来源族特征不得自评（3077 规则）**；**溯源锚必须覆盖所有进入统计的样本（3078 规则：B prefix 错误为 smoke 盲区）**。

## 标准锚与精度
- bit 锚家族：标量行、因果置换、跨 phase 参考数、块链恒等、hook 互证、跨 phase 因果复现、枚举重放（3075）、bit 锚族（3076）、跨源一致性（3077）、**前向重建锚（3078：a1 逐 k DAH 重算=冻结 ZH34 − 新基座 zH，同时证基座前向 bit 一致+数学一致；a0 溯源 LG64−LG32=0.0 因 fp32 logits 无损、TT 舍入 ~1e-7）**。
- 上游 npz 只有 float32 降采样时，bit 锚必须重建 float64 原量（TT64/LG64）。
- 跨路径 bit 锚需匹配浮点求和顺序；置换检验 rng 冻结（3078 用 seed 3078）。

## 机制解释审计链（命名前依次检查）
…→3076 cross_prompt_unstable→3077 observation_causation_decoupled（12 静态特征全败；ICC 0.865 vs 0.452）→**3078 routing_signal_absent：基座 L24-L35 全层线性读出与 R1 无关（max|sp|=0.221，p≥0.226）；谱结构跨族共享至 0.945 但与因果正交；幅度对照零；基座谱 vs 差异谱也不一致（−0.20/−0.36/+0.07）→路由=注入时刻事件驱动装配（注入态×头交互），3073 higher-order 升格首候选**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep 坏→删除目录用 python shutil.rmtree；日志用 Read；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查。
- NaN 数据：argsort 污染 spearman，须有效掩码过滤。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3078）
Ω-P2（3011-3078）：…3075 supermodular_diffuse；3076 cross_prompt_unstable；3077 observation_causation_decoupled；**3078 routing_signal_absent（路由=事件驱动，全层线性读出排除）**。

## 下一步
- max=3078，下一个 3079（A 主选 **谱形相似性→迁移强度**：三族 255 子集谱形 vs 迁移 spearman，免前向；B h14 全域解剖（免前向为主）；C DS7B 跨模型对照；D 注入态高阶交互谱（需前向））。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3078')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
