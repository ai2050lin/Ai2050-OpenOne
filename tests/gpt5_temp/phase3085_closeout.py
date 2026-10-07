# -*- coding: utf-8 -*-
"""Phase 3085 closeout (idempotent): Ledger -> L14
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
     r'\phase3085'
     r'\omega_p82_l34_full_arbitration')
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
assert verdict == 'third_mixed_absent', \
    verdict
assert res['forwards'] == 19770
assert seal['setup_ok'] is True
fam = res['stats']['families']
assert fam['A']['n_neg'] == 12
assert fam['B']['n_neg'] == 12
assert fam['C']['n_neg'] == 11
assert fam['C']['top8_sel_ok']
assert res['stats']['repro'] is not None
z = np.load(R + r'\omega_p82_l34_full_'
            r'arbitration.npz',
            allow_pickle=False)
spec_cls = str(z['SPEC_CLASS'])
assert spec_cls == 'mixed', spec_cls
t3 = {fk: float(z['E3_TOP3_CS_' + fk])
      for fk in ('A', 'B', 'C')}
pr = {fk: float(z['E3_PR_CS_' + fk])
      for fk in ('A', 'B', 'C')}
rmin = {fk: float(np.min(z['R_S_' + fk]))
        for fk in ('A', 'B', 'C')}
rall = {fk: float(z['R_ALL_' + fk])
        for fk in ('A', 'B', 'C')}
mc = {fk: float(z['MED_C_' + fk])
      for fk in ('A', 'B', 'C')}
mig = {k: float(z['MIG_' + k])
       for k in ('AB', 'AC', 'BC')}
rdiff = {fk: float(max(
    z['REPRO_MEDC_DIFF_' + fk],
    z['REPRO_CS1H_DIFF_' + fk],
    z['REPRO_RALL_DIFF_' + fk]))
    for fk in ('A', 'B', 'C')}
gds = res['stats']['cross']
assert gds['g_ds_count'] == 1
el = res['elapsed']
fw = res['forwards']

meas = {
    'meas_id': 'meas3085_omega_p82_'
               'l34_full_arbitration',
    'phase': 3085,
    'claim': (
        'Omega-P82 (plan 3085 A) - the 3083 '
        'four-way trunk/dispersed x '
        'migrates arbitration RE-OPENED at '
        'L_INJ=34 (the 3084-confirmed '
        'strongest rescue layer) on '
        'qwen2.5-3b-instruct: full 3083 '
        'pipeline (E1 repV ladder + E3 '
        '16-head scan + focal top8 + E4 '
        '255-subset sweep + spectrum + '
        'G_DS/f2~T_AB migration gate + '
        'cross statistics; Qwen2ForCausalLM, '
        '36L/16H/2kv, hidden 2048, bf16, '
        'tied embeddings recorded; seed '
        '3085, %d forwards, %.1fs; '
        'b-anchors all bit 0).  NEW '
        'deterministic reproduction anchor '
        'vs the 3084 L34 npz PASSED on all '
        'three families (n_neg/top8 exact; '
        'med_c/CS1H/R_ALL max diff '
        '%.1e/%.1e/%.1e - pure fp32-'
        'roundtrip noise, tolerance 1e-9), '
        'confirming bit-deterministic '
        'activation collection.  VERDICT '
        'third_mixed_absent (preregistered '
        'mixed exit): all three families '
        'fully healthy at L34 (n_neg '
        '12/12/11, capture8 0.923/0.896/'
        '0.945, med_c %.3f/%.3f/%.3f, '
        'R(S) min %.3f/%.3f/%.3f) - the '
        '3083 L33 degeneration is a layer '
        'effect, NOT a family absence.  CS '
        'spectrum top3 A=%.4f (PR %.2f) '
        'B=%.4f (PR %.2f) C=%.4f (PR %.2f) '
        '-> MIXED band; three-model '
        'continuum: 4B 0.957-0.964 (trunk), '
        '3B 0.827-0.875 (mixed), DS7B '
        '0.27-0.32 (dispersed).  Migration '
        'gate NOT passed: G_DS count=1/6 '
        '(f2~T_AB +0.430 p=0.035 singly '
        'significant; f2~U_BC -0.540 '
        'p=0.007 opposite-sign), min_sp '
        '<0, Stouffer z=1.643 (edge, '
        '0.05 bar 1.645); mig(R1) '
        'AB=%.3f AC=%.3f BC=%.3f.  '
        'ARBITRATION FINAL: 4B '
        'trunk x migrates, DS7B dispersed '
        'x no_migrate, 3B mixed x absent - '
        'the 3082 spectral prediction is '
        'direction-consistent at all three '
        'points and NOT falsified; the '
        'continuum reading (loading '
        'low-dimensionality decays with '
        'spectral position; migration gate '
        'fails conservatively in the '
        'middle) gains its first '
        'quantitative support.  CAVEAT: '
        '"absent" is not "no migration" - '
        'Stouffer is 1-perm from the bar '
        'and mig_AB=0.859/T_AB significant '
        'suggest partial AB locking '
        '(recorded, not gated).'
        % (fw, el,
           rdiff['A'], rdiff['B'],
           rdiff['C'],
           mc['A'], mc['B'], mc['C'],
           rmin['A'], rmin['B'], rmin['C'],
           t3['A'], pr['A'],
           t3['B'], pr['B'],
           t3['C'], pr['C'],
           mig['AB'], mig['AC'],
           mig['BC'])),
    'verdict': verdict,
    'anchors': 'b0/b1/b3/b4/b6/b7a/b8 all '
               'bit 0.0 on all three families '
               '(setup_ok=true); repro anchor '
               'vs 3084 L34 npz passed (exact '
               'n_neg/top8, fp32-noise-level '
               'float diffs); spectrum '
               'replayed bit-exact from npz '
               'CS matrices',
    'artifacts': {
        'result': 'phase3085/omega_p82_'
                  'l34_full_arbitration/'
                  'result.json',
        'npz': 'phase3085/omega_p82_'
               'l34_full_arbitration/'
               'omega_p82_l34_full_'
               'arbitration.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'four-way arbitration completed '
            'at the 3084-selected layer; '
            'mixed x absent is the '
            'preregistered exit for the '
            'middle band; Stouffer edge '
            '(1.643 vs 1.645) recorded; '
            'f2~U_BC opposite-sign '
            'significant is unexplained '
            '(post-hoc candidates only); '
            'same 3076 texts; tied '
            'embeddings recorded.',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3085
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 224
    l14['connects'].append({
        'meas_id': 'meas3085_omega_p82_'
                   'l34_full_arbitration',
        'phase': 3085,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P82: four-way '
                        'arbitration re-opened '
                        'at L_INJ=34 (full 3083 '
                        'pipeline, 19770 '
                        'forwards).  VERDICT '
                        'third_mixed_absent: '
                        'all families healthy '
                        '(n_neg 12/12/11, '
                        'capture8 0.92/0.90/'
                        '0.95); CS top3 '
                        '0.870/0.875/0.827 '
                        'MIXED band (continuum '
                        '4B 0.957+ / 3B mid / '
                        'DS7B 0.32-); G_DS '
                        'count=1/6, Stouffer '
                        'z=1.643 edge, mig_AB='
                        '0.859 partial-AB '
                        'caveat.  3082 spectral '
                        'prediction direction-'
                        'consistent at three '
                        'points (trunk->migrates, '
                        'dispersed->no, '
                        'mixed->absent); '
                        'deterministic repro '
                        'anchor vs 3084 passed '
                        '(fp32-noise level).  '
                        'Opens 3086: A spectrum-'
                        'migration continuum '
                        'test; B L28 four-way '
                        'replication; C layer x '
                        'spectrum; D 4B trunk '
                        'anatomy; E gate '
                        'sensitivity'})
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
if '## Phase 3085:' not in memo:
    sec = u'''## Phase 3085: Ω-P82 L34 完整管线四格仲裁——三族健康、谱落混合带、判决 third_mixed_absent [%(created)s]

**判决：`third_mixed_absent`**（预注册 mixed 出口：谱落 0.5–0.9 混合带 + 迁移门未过。qwen2.5-3b-instruct bf16，L_INJ=34（3084 确认的最强救援层）完整 3083 管线：E1 repV 阶梯 + E3 16 头扫描 + 焦点 top8 + E4 255 子集全扫 + 谱结构 + G_DS/f2~T_AB 迁移门 + 跨族统计，seed 3085，**19770 前向 / %(el).1f 秒**，b-anchor 全 bit 0，**确定性复现锚三族全过**）。

### 核心结果（重复三遍）
**① 三族全健康 + 复现锚（一）**：n_neg 12/12/11（L33 处是 14/13/**2**）、capture8 0.923/0.896/0.945、med_c 0.310/0.222/0.360；对 3084 L34 npz 的确定性复现锚全过——n_neg/top8 精确相等、med_c/CS1H/R_ALL 最大差 %(rd1).1e/%(rd2).1e/%(rd3).1e（纯 fp32 往返噪声，容差 1e-9）→ **激活收集 bit 确定性成立；3083 退化=层位效应再次确认**。**② 谱混合带（二）**：CS top3 A=0.8704（PR 1.88）、B=0.8747（PR 2.09）、C=0.8273（PR 2.99）→ **mixed**；三模型连续统完整成形：4B 0.957–0.964（trunk）/ **3B 0.827–0.875（mixed）** / DS7B 0.27–0.32（dispersed）。**③ 迁移门未过（三）**：G_DS count=1/6（f2~T_AB +0.430 p=0.035 单显著、f2~U_BC −0.540 p=0.007 反号显著）、min_sp<0、Stouffer z=1.643（**边缘**，距 0.05 门限 1.645 一步之遥）；mig(R1) AB=0.859 / AC=0.450 / BC=0.294 → 判决 absent（保守）。

### 四格仲裁最终格局（3082 问题闭合）
| 模型 | 谱型 | 迁移 | 出处 |
| --- | --- | --- | --- |
| qwen3-4b | trunk（0.957–0.964） | migrates | 3080 |
| DS7B | dispersed（0.27–0.32） | no_migrate | 3081 |
| qwen2.5-3b@L34 | **mixed（0.827–0.875）** | **absent（门未过）** | 3085 |

**3082 的谱型→迁移预测在三个数据点上方向全部一致，未被证伪**；3B 混合带 + absent 支持"连续统"版本：因果载入的低维共享性随谱位置衰减，混合带模型的迁移门在门限附近保守失效。

### 理论更新（第一性原理）
- **PR/top3 诊断量获得完整第三点**：C 族修复后谱值 0.827 也落带内——3083 的 C 无效谱不是带外反例；混合带真实存在且跨族（A/B/C 内部一致）。
- **谱-迁移耦合升级为三点方向一致**：trunk↔迁移存在、dispersed↔无迁移、mixed↔保守 absent。第一个定量支持，但仍是非参数方向证据，非定律。
- **层位维度并入理论**：机制可用性=模型×族×层三元函数（L33 退化/L34 健康）；同模型不同层的谱型差异（L33 vs L34）未测，是下一个自然问题。
- **确定性复现锚成为新标准件**：同模型同文本同层的跨 Phase 激活 bit 复现（差异 ≤3.6e-13）——以后所有单模型 Phase 都可挂此锚防实现漂移。

### 硬伤与边界
- **"absent"≠"无迁移"**：Stouffer 1.643 距门限 0.002（置换粒度级），mig_AB=0.859 + f2~T_AB 单显著提示 AB 对存在部分锁定；预注册判决如实记录，理论解读必须携带边缘性。
- f2~U_BC 反号显著（−0.540）无预注册解释——post-hoc 候选（族间异质/幅度混杂），仅登记不采信。
- 层位单点（仅 L34）；L28 的四格未测；谱阈值 0.9/0.5 仍是两点校准。
- 3B 与 4B 同 Qwen 系架构；n=24 对、一阶 partial 为方向性证据。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3085/omega_p82_l34_full_arbitration/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；%(fw)d forwards / %(el).1fs。

**接续 3086 菜单**——A（主选）**谱-迁移连续统定量检验**（三模型数据汇总 + 谱位置与迁移强度的预注册关系检验设计，免前向为主）。B **L28 四格复制**（第二救援带完整管线，~20k 前向，检验层位×判决稳定性）。C **层位×谱型**（L31/L33 补 E4 谱，检验谱型是否随层变化；中量前向）。D **4B 主干解剖**（免前向，CS top-1 成分的头/子集分解）。E **G_DS 门敏感性分析**（免前向：Stouffer 边缘性、count 门 vs 连续 z 门、f2~U_BC 反号诊断）。"好的，继续"即进 3086 A。
''' % {'created': created,
       'el': el,
       'rd1': rdiff['A'],
       'rd2': rdiff['B'],
       'rd3': rdiff['C'],
       'script8': seal['script_sha256_8'],
       'result8': seal['result_sha256_8'],
       'npz8': seal['npz_sha256_8'],
       'exec8': seal['exec_sha256_8'],
       'n': len(led['measurements']),
       'l14': len(l14['connects']),
       'fw': fw}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 四十七、3085' not in aud:
    add = u'''
---
## 四十七、3085 增补：L34 完整管线四格仲裁——判决 third_mixed_absent（Omega-P82）
1. **判决**：L_INJ=34 完整 3083 管线（19770 前向，锚全 bit 0，确定性复现锚三族全过）——三族全健康（n_neg 12/12/11、capture8 0.92/0.90/0.95）；CS top3 0.870/0.875/0.827 → mixed；G_DS count=1/6、Stouffer z=1.643（边缘）→ 迁移 absent（保守）。
2. **四格仲裁闭合**：4B trunk×migrates、DS7B dispersed×no_migrate、3B mixed×absent——3082 谱型→迁移预测三点方向一致，未被证伪；连续统解读（载入低维性随谱位置衰减）获首个定量支持。
3. **HDMCC 更新**：PR/top3 连续统（4B 0.957+ / 3B 0.827–0.875 / DS7B 0.32−）成形；"absent"携带 Stouffer 边缘性 + mig_AB=0.859 部分锁定警示；层位=机制可用性第三维；确定性复现锚升级为标准防呆件。f2~U_BC 反号显著未解释（仅登记）。
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
if 'Phase 3085' not in prev:
    line = ('- Phase 3085 Omega-P82 L34 full '
            'pipeline four-way arbitration '
            '(qwen2.5-3b-instruct, L_INJ=34, '
            '19770 forwards): verdict '
            'third_mixed_absent - all families '
            'healthy (n_neg 12/12/11), CS top3 '
            '0.870/0.875/0.827 mixed band, '
            'G_DS count=1/6 Stouffer z=1.643 '
            'edge.  3082 spectral prediction '
            'direction-consistent at three '
            'points (4B trunk->migrates, DS7B '
            'dispersed->no, 3B mixed->absent).  '
            'Deterministic repro anchor vs '
            '3084 passed.  Audit 47; ledger '
            '224/L14 192.\n')
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
if 'max=3085' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L 4kv 3584 bf16）、qwen2.5-3b-instruct（Qwen2 36L 16H 2kv 2048 bf16 **tied embeddings**；层位扫描 rescue=[28,34]，L34=四格仲裁层）；glm4-9b-chat-hf 备用。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout/verify tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（**含旧 str 条目，必须 isinstance(c, dict) 防御**）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→独立 verify→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json（脚本自删亦可）；负结果与预注册退化/混合出口如实登记为一等公民——门未过禁止放宽（防 post-hoc）。
4. 统计纪律：阈值预注册；置换/偏相关 p 必须与主脚本 bit 一致；泛化声明分层。
5. verify 锚键先 Grep 主脚本 npz save 段逐一核对（B3 类布尔锚无 DIFF 分量）。
6. **确定性复现锚（3085 新标准件）**：同模型同文本同层跨 Phase 重跑，n_neg/top8 须精确相等、浮点差 ≤1e-9（fp32 往返噪声 ~3e-12）；mismatch→setup_failed。

## 标准锚与精度
- bit 锚家族：…跨模型 setup 锚（3081）、免前向冻结重放锚（3082）、退化出口锚（3083）、层位扫描锚（3084）、**复现锚+四格混合出口（3085）**。
- b8 语义：注入前向内部块链连续性；置换 rng seed=phase 号。

## 机制解释审计链
…→3082 ds7b_decorrelated→3083 third_top8_degenerate（L33 C 族退化）→3084 layer_rescue（L28/L34 双救援）→**3085 third_mixed_absent（L34 四格仲裁：三族健康、谱 mixed 0.827–0.875、G_DS 1/6 Stouffer 1.643 边缘；3082 预测三点方向一致：4B trunk→migrates、DS7B dispersed→no、3B mixed→absent）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail 坏→python + 写文件；日志用 Read；-c stdout 丢→写文件再 Read；Glob/Grep 对部分目录失效→python os.listdir 为准。
- 关键写入后必须 Grep/Read 复核；改后必编译检查；result.json 无 smoke 键，npz 里 SMOKE 标量为准。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 下一步
- max=3085，下一个 3086（A 主选 **谱-迁移连续统定量检验**；B L28 四格复制；C 层位×谱型 L31/L33 补 E4；D 4B 主干解剖免前向；E G_DS 门敏感性分析）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars'
             % len(mem_new))
else:
    o.append('memory already max=3085')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
