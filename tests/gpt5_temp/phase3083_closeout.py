# -*- coding: utf-8 -*-
"""Phase 3083 closeout (idempotent): Ledger -> L14
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
     r'\phase3083'
     r'\omega_p80_third_model_arbitration')
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
assert verdict == 'third_top8_degenerate', \
    verdict
assert res['forwards'] == 13722
assert seal['setup_ok'] is True
fam = res['stats']['families']
assert fam['C']['n_neg'] == 2
assert fam['A']['n_neg'] == 14
assert fam['B']['n_neg'] == 13
assert not fam['C']['top8_sel_ok']
assert fam['A']['top8_sel_ok']
assert fam['B']['top8_sel_ok']
z = np.load(R + r'\omega_p80_third_model_'
            r'arbitration.npz',
            allow_pickle=False)
spec_cls = str(z['SPEC_CLASS'])
assert spec_cls == 'mixed', spec_cls
t3 = {fk: float(z['E3_TOP3_CS_' + fk])
      for fk in ('A', 'B', 'C')}
pr = {fk: float(z['E3_PR_CS_' + fk])
      for fk in ('A', 'B', 'C')}
keff = {fk: float(z['E3_KEFF_CS_' + fk])
        for fk in ('A', 'B', 'C')}
rmin = {fk: float(np.min(z['R_S_' + fk]))
        for fk in ('A', 'B', 'C')}
rall = {fk: float(z['R_ALL_' + fk])
        for fk in ('A', 'B', 'C')}
el = res['elapsed']
fw = res['forwards']

meas = {
    'meas_id': 'meas3083_omega_p80_'
               'third_model_arbitration',
    'phase': 3083,
    'claim': (
        'Omega-P80 (plan 3083 A) - third-model '
        'arbitration of the 3082 PR spectrum '
        'prediction on qwen2.5-3b-instruct '
        '(Qwen2ForCausalLM, 36L/16H/2kv, '
        'hidden 2048, vocab 151936, bf16, '
        'tied embeddings recorded; '
        'L_INJ=33/L_POST=34; full 3081 '
        'pipeline, %d forwards, %.1fs; '
        'b-anchors all bit 0).  VERDICT '
        'third_top8_degenerate - the '
        'preregistered degenerate exit: '
        'family C focal heads n_neg=2/16 '
        '(top8=[11,13], capture8=1.0) - '
        'single-head causal swap responses '
        'at/above the med_c baseline; '
        'families A/B fully healthy (n_neg '
        '14/16 and 13/16, capture8 '
        '%.3f/%.3f, R(S) min %.3f/%.3f) so '
        'the degeneration is family-'
        'dependent, not an implementation '
        'fault.  CS spectrum top3 A=%.4f '
        '(PR %.2f) B=%.4f (PR %.2f) - in '
        'the MIXED BAND between 4B '
        '(0.957-0.964) and DS7B (0.27-0.32); '
        'C spectrum invalid (CS reduced to '
        '3 rows).  Full 16-head swap '
        'recovery near zero (%+.4f/%+.4f/'
        '%+.4f) - causal response '
        'magnitudes are small on 3B.  '
        'Freeze discipline: cross-model '
        'statistics SKIPPED after '
        'degeneration (no post-hoc gate '
        'relaxation), so the four-way '
        'trunk/dispersed x migrates/no '
        'arbitration is NOT answered yet.  '
        'Candidate explanations (all '
        'unproven): (a) layer position - '
        'L_INJ=33 (3rd from last) may not '
        'be a causal loading layer on 3B; '
        '(b) capacity/magnitude - causal '
        'effects smaller than the single-'
        'head resolution in family C; '
        '(c) genuine family-specific '
        'absence.  THEORY: PR/top3 gains a '
        'partial third data point - A/B '
        'spectra suggest a continuum '
        'between trunk and dispersed rather '
        'than a binary split; focal-head '
        'structure existence is model x '
        'family dependent.'
        % (fw, el,
           fam['A']['capture8'],
           fam['B']['capture8'],
           rmin['A'], rmin['B'],
           t3['A'], pr['A'],
           t3['B'], pr['B'],
           rall['A'], rall['B'],
           rall['C'])),
    'verdict': verdict,
    'anchors': 'b0/b1/b3/b4/b6/b7a/b8 all '
               'bit 0.0 on all three families '
               '(setup_ok=true); spectrum '
               'replayed bit-exact from npz CS '
               'matrices; degeneration itself '
               'is data (C n_neg=2/16)',
    'artifacts': {
        'result': 'phase3083/omega_p80_'
                  'third_model_arbitration/'
                  'result.json',
        'npz': 'phase3083/omega_p80_'
               'third_model_arbitration/'
               'omega_p80_third_model_'
               'arbitration.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'degenerate preregistered exit; '
            'tied embeddings (recorded, '
            'forwards/logits protocol '
            'unaffected); L_INJ=33 single '
            'layer - layer confound '
            'untested; cross statistics '
            'absent by freeze discipline; '
            'C-family CS spectrum computed '
            'on a degenerate 3-row matrix '
            '(recorded, not interpretable).',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3083
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 222
    l14['connects'].append({
        'meas_id': 'meas3083_omega_p80_'
                   'third_model_arbitration',
        'phase': 3083,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P80: third-'
                        'model arbitration '
                        '(qwen2.5-3b-instruct, '
                        '36L/16H/2kv, L_INJ=33, '
                        'full 3081 pipeline).  '
                        'VERDICT '
                        'third_top8_degenerate '
                        '(preregistered exit): '
                        'family C n_neg=2/16 '
                        '(top8=[11,13]) vs '
                        'healthy A/B (n_neg '
                        '14/13, capture8 '
                        '0.845/0.896).  CS '
                        'top3 A=0.854 B=0.891 '
                        'in the MIXED BAND '
                        'between 4B (0.957+) '
                        'and DS7B (0.32-); '
                        'full-head swap near '
                        'zero - 3B causal '
                        'response magnitudes '
                        'small.  Arbitration '
                        'four-way table NOT '
                        'answered (cross '
                        'stats frozen out).  '
                        'Candidates: layer '
                        'position, capacity/'
                        'magnitude, genuine '
                        'family-specific '
                        'absence.  Opens 3084: '
                        'A 3B L_INJ layer scan; '
                        'B relaxed-K3 prereg '
                        'rerun; C 4B trunk '
                        'anatomy (no-forward); '
                        'D f2~U third '
                        'replication design; E '
                        'continuum spectrum-'
                        'migration test'})
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
if '## Phase 3083:' not in memo:
    sec = u'''## Phase 3083: Ω-P80 第三模型仲裁（qwen2.5-3b）——C 族焦点头退化（n_neg 2/16），判决 third_top8_degenerate；A/B 族谱落混合带 [%(created)s]

**判决：`third_top8_degenerate`**（预注册第三合法出口：任一族 n_neg<8 → 跨族统计冻结跳过。qwen2.5-3b-instruct bf16 单模型全 3081 管线（Qwen2ForCausalLM，36L/16H/2kv，hidden 2048，vocab 151936，tied embeddings 已记录），L_INJ=33/L_POST=34，**13722 前向 / %(el).1f 秒**，b-anchor 全 bit 0）。

### 核心结果（重复三遍）
**① C 族焦点头退化（一）**：social-emotional-causal 族的 16 头中仅 2 个负 r1（top8=[11,13]，capture8=1.0）——单头因果置换响应几乎全部不低于 med_c 基线。**② A/B 族完全健康（二）**：n_neg 14/16、13/16；top8 capture8 0.845/0.896；R(S) min −0.108/−0.180——管线在该模型上正常工作，**退化是族依赖的，不是实现故障**。**③ 谱结构中间带（三）**：CS top3 A=0.854（PR 2.80）、B=0.891（PR 2.15），落在 4B（0.957–0.964）与 DS7B（0.27–0.32）之间的**混合带**；C 族 CS 仅 3 行（退化）谱无效。全 16 头 swap 恢复近零（+0.011/+0.023/−0.041）——3B 因果响应幅度整体偏小。

### 仲裁问题状态
四格判决（trunk/dispersed × migrates/no）**未被回答**——预注册纪律禁止退化后放宽门限跑跨族统计（防 post-hoc）。3082 的"主干型↔迁移存在"预测在 qwen2.5-3b 上**暂不可检验**，仲裁推迟。

### 三个候选解释（全部为候选，无机制证明）
(a) **层位假说**：L_INJ=33（36 层倒数第 3）可能不是 3B 的因果载入层——4B 用倒数 2/1 层、DS7B 用倒数 3/2 层；3B 的容量/深度比不同，载入层位可能系统性不同。(b) **容量/幅度假说**：R(S) 幅度小 + 全头 swap 近零提示 C 族 24 对置换的平均效应低于单头分辨率。(c) **真实阴性**：qwen2.5-3b 的 C 族因果响应确实无焦点结构（A/B 有、C 无——族依赖）。

### 理论更新（第一性原理）
**PR/top3 诊断量获得第三个（部分）数据点**：A/B 族的 0.854/0.891 提示谱结构可能是**连续统**而非 4B/DS7B 式二分——若 3084 修好 C 族并跑通跨族统计，可检验"中间谱→中间迁移"的连续性版本。**退化本身是一等公民数据：焦点头结构的存在性是模型×族依赖的**——不是所有 instruct 模型都在所有语义族上维持可定位的因果焦点头。

### 硬伤与边界
- C 族退化 → 跨族统计缺失，仲裁未完成；单层位（L_INJ=33）单模型，层位混杂未排除。
- CS 谱的 A/B 值在 NT=8 健康条件下算出；C 值在 3 行退化矩阵上算出（已记录、不可解释）。
- tied embeddings 模型（协议只经 forwards/logits，不受影响，已记录为适配项）。
- 3B 响应幅度小可能整体压低置换效应的可检测性。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3083/omega_p80_third_model_arbitration/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；%(fw)d forwards / %(el).1fs。

**接续 3084 菜单**——A（主选）**3B 层位扫描**（L_INJ ∈ {28, 31, 33, 34} 小规模 E3 焦点扫描，定位 3B 因果载入层、检验 n_neg 族分布随层变化；中量前向）。B **3B 放宽预注册重跑**（新预注册放宽 K3/topk 门 + C 族专项）。C **4B 主干解剖**（免前向为主，CS top-1 成分的头/子集分解）。D **f2~U_AB 第三复现设计**。E **连续谱-迁移假设**（汇总 4B/DS7B/3B 三点，设计连续性检验）。"好的，继续"即进 3084 A。
''' % {'created': created,
       'el': el,
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
if '## 四十五、3083' not in aud:
    add = u'''
---
## 四十五、3083 增补：第三模型仲裁（qwen2.5-3b）——C 族焦点头退化、判决 third_top8_degenerate（Omega-P80）
1. **判决**：预注册退化出口——C 族（social-emotional-causal）n_neg=2/16（top8=[11,13]），跨族统计冻结跳过；A/B 族健康（n_neg 14/13、capture8 0.845/0.896）——退化是族依赖的，非实现故障。
2. **谱结构**：CS top3 A=0.854/B=0.891 落在 4B（0.957–0.964）与 DS7B（0.27–0.32）之间的**混合带**；C 无效（3 行退化矩阵）。全头 swap 近零（+0.011/+0.023/−0.041）——3B 因果响应幅度小。
3. **HDMCC 更新**：焦点头结构存在性是模型×族依赖；PR/top3 获第三个（部分）数据点——可能是连续统而非二分；四格仲裁推迟至 3084 层位扫描。候选解释：层位（L_INJ=33 非 3B 载入层）/容量幅度/真实族依赖阴性。
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
if 'Phase 3083' not in prev:
    line = ('- Phase 3083 Omega-P80 third-'
            'model arbitration (qwen2.5-3b-'
            'instruct, full 3081 pipeline, '
            'L_INJ=33, 13722 forwards): '
            'verdict third_top8_degenerate - '
            'family C focal heads n_neg=2/16 '
            'vs healthy A/B (14/13, capture8 '
            '0.845/0.896).  CS top3 A=0.854 '
            'B=0.891 in the mixed band '
            'between 4B (0.957+) and DS7B '
            '(0.32-); full-head swap near '
            'zero.  Cross stats frozen out; '
            'arbitration postponed.  Audit '
            '45; ledger 222/L14 190.\n')
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
if 'max=3083' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L 4kv 3584 bf16）、qwen2.5-3b-instruct（Qwen2 36L 16H 2kv 2048 bf16 **tied embeddings**，L_INJ=33）；glm4-9b-chat-hf 备用。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout/verify tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（**L14.connects 含旧 str 条目，必须 isinstance(c, dict) 防御**）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→独立 verify→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json（脚本自删亦可）；负结果与**预注册退化出口**（3083 third_top8_degenerate）如实登记为一等公民——退化后禁止放宽门限跑跨族统计（防 post-hoc）。
4. 统计纪律：阈值预注册；置换/偏相关 p 的实现必须与主脚本 bit 一致（partial p 主脚本=cnt/n_perm 无 +1 平滑）；泛化声明分层。

## 标准锚与精度
- bit 锚家族：…跨模型 setup 锚（3081）、免前向输入冻结+重放锚（3082）、**退化出口锚（3083：n_neg/top8_len=min(8,n_neg)/capture8 记录一致）**。
- b8 语义：注入前向内部块链连续性；置换 rng seed=phase 号。

## 机制解释审计链（命名前依次检查）
…→3080 ab_cos_locked→3081 ds7b_cos_absent→3082 ds7b_decorrelated（CS PR/top3=可迁移性诊断）→**3083 third_top8_degenerate（qwen2.5-3b：C 族 n_neg 2/16 退化；A/B 谱 top3 0.854/0.891 落混合带；四格仲裁推迟；候选=层位/容量幅度/族依赖阴性）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail 坏→python + 写文件；日志用 Read；-c stdout 丢→写文件再 Read；Glob 对部分目录失效→python os.listdir 为准。
- 关键写入后必须 Grep/Read 复核；改后必编译检查；result.json 无 smoke 键，npz 里 SMOKE 标量为准。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3083）
Ω-P2（3011-3083）：…3081 ds7b_cos_absent；3082 ds7b_decorrelated；**3083 third_top8_degenerate（qwen2.5-3b C 族退化，仲裁推迟）**。

## 下一步
- max=3083，下一个 3084（A 主选 **3B 层位扫描** L_INJ∈{28,31,33,34} 定位载入层；B 放宽预注册重跑；C 4B 主干解剖（免前向）；D f2~U 第三复现设计；E 连续谱-迁移假设检验）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars'
             % len(mem_new))
else:
    o.append('memory already max=3083')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
