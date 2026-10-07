# -*- coding: utf-8 -*-
"""Phase 3084 closeout (idempotent): Ledger -> L14
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
     r'\phase3084'
     r'\omega_p81_3b_layer_scan')
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
assert verdict == 'layer_rescue', verdict
assert res['forwards'] == 5298
assert seal['setup_ok'] is True
st = res['stats']
assert st['rescue_layers'] == [28, 34]
assert st['rescue_best'] == 28
fl = st['families_layers']
assert fl['C']['33']['n_neg'] == 2
assert fl['C']['28']['n_neg'] == 15
assert fl['C']['34']['n_neg'] == 11
assert fl['C']['31']['n_neg'] == 6
assert fl['A']['33']['n_neg'] == 14
assert fl['B']['33']['n_neg'] == 13
z = np.load(R + r'\omega_p81_3b_layer_scan.npz',
            allow_pickle=False)
mc34 = {fk: float(z['MED_C_L34_' + fk])
        for fk in ('A', 'B', 'C')}
ra33 = {fk: float(z['R_ALL_L33_' + fk])
        for fk in ('A', 'B', 'C')}
ra34 = {fk: float(z['R_ALL_L34_' + fk])
        for fk in ('A', 'B', 'C')}
el = res['elapsed']
fw = res['forwards']

meas = {
    'meas_id': 'meas3084_omega_p81_'
               '3b_layer_scan',
    'phase': 3084,
    'claim': (
        'Omega-P81 (plan 3084 A) - '
        'descriptive layer-position scan '
        'of the 3071/3083 focal-head '
        'machinery on qwen2.5-3b-instruct '
        '(Qwen2ForCausalLM, 36L/16H/2kv, '
        'hidden 2048, bf16, tied '
        'embeddings recorded; banks '
        'layer-independent per family, '
        'L_INJ in (28,31,33,34), '
        'L_POST=L_INJ+1; seed 3084, %d '
        'forwards, %.1fs; family anchors '
        'b0/b1/b3/b6/b7a and per-layer '
        'b4/b8 all bit 0).  VERDICT '
        'layer_rescue: L28 and L34 both '
        'rescue ALL families (n_neg>=8) '
        'while L33 (the 3083 position) '
        'reproduces the 3083 degeneration '
        '(family C n_neg=2/16, med_c '
        'lowest 0.072-0.086, R_ALL near '
        'zero %+.3f/%+.3f/%+.3f) - the '
        '3083 family-C degeneration IS A '
        'LAYER-POSITION EFFECT, not an '
        'implementation fault and not a '
        'genuine family-specific absence.  '
        'L34 (2nd from last, the qwen3-4b '
        'protocol position) is the '
        'STRONGEST layer overall: med_c '
        '%.3f/%.3f/%.3f (highest in '
        'scan), R_ALL %+.3f/%+.3f/%+.3f '
        '(strongest), capture8 '
        '0.923/0.896/0.945.  L28 (depth '
        '0.78) is a second rescue band '
        '(n_neg 12/11/15, R_ALL '
        '-0.17/-0.17/-0.12).  L31 shows '
        'partial C-family degradation '
        '(n_neg=6).  rescue_layers=[28,'
        '34]; rescue_best=28 by '
        'min-family-n_neg tie-break '
        '(11==11), while L34 dominates '
        'on magnitudes.  THEORY: causal '
        'loading on 3B is a TWO-BAND '
        'structure (mid-deep L28 + '
        'deep-readout L34) with a valley '
        'at L31/L33; two of three 3083 '
        'candidate explanations are now '
        'disfavored (capacity/magnitude: '
        'L34 med_c is 4B-scale; genuine '
        'family absence: C is healthy at '
        'L28/L34).  The four-way '
        'trunk/dispersed x migrates '
        'arbitration is RE-OPENED at '
        'L_INJ=34.'
        % (fw, el,
           ra33['A'], ra33['B'], ra33['C'],
           mc34['A'], mc34['B'], mc34['C'],
           ra34['A'], ra34['B'], ra34['C'])),
    'verdict': verdict,
    'anchors': 'family anchors b0/b1/b3/b6/'
               'b7a bit 0.0 on all three '
               'families; per-layer b4/b8 '
               'bit 0.0 on all 12 (layer, '
               'family) cells (setup_ok='
               'true)',
    'artifacts': {
        'result': 'phase3084/omega_p81_'
                  '3b_layer_scan/'
                  'result.json',
        'npz': 'phase3084/omega_p81_'
               '3b_layer_scan/'
               'omega_p81_3b_layer_scan.'
               'npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'descriptive preregistered '
            'scan (no permutation tests); '
            'coarse 4-layer grid; '
            'rescue_best=28 is a tie-break '
            'on min-family n_neg (11==11), '
            'L34 stronger on magnitudes; '
            'same 3076 texts; tied '
            'embeddings recorded.',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3084
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 223
    l14['connects'].append({
        'meas_id': 'meas3084_omega_p81_'
                   '3b_layer_scan',
        'phase': 3084,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P81: 3B '
                        'layer-position scan '
                        '(qwen2.5-3b-instruct, '
                        'L_INJ in 28/31/33/34, '
                        '5298 forwards, '
                        'descriptive).  VERDICT '
                        'layer_rescue: L28 and '
                        'L34 rescue all three '
                        'families (n_neg '
                        '12/11/15 and 12/12/11) '
                        'while L33 reproduces '
                        'the 3083 C-family '
                        'degeneration (n_neg=2) '
                        '- 3083 degeneration is '
                        'a LAYER-POSITION '
                        'EFFECT.  L34 (2nd from '
                        'last, 4B protocol '
                        'position) strongest: '
                        'med_c 0.22-0.36, '
                        'R_ALL -0.23..-0.30, '
                        'capture8 0.90-0.95; '
                        'L28 mid-deep second '
                        'band; L31 partial '
                        'degradation (C n_neg='
                        '6).  Two of three 3083 '
                        'explanations disfavored '
                        '(capacity/magnitude, '
                        'genuine family absence); '
                        'loading = TWO-BAND '
                        'structure with valley at '
                        'L31/L33.  Opens 3085: A '
                        'L34 full 3083 pipeline '
                        '(four-way arbitration '
                        're-opened); B L28 '
                        'replication; C fine '
                        'layer profile; D 4B '
                        'layer-control scan; E '
                        'two-band theory '
                        'analysis'})
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
if '## Phase 3084:' not in memo:
    sec = u'''## Phase 3084: Ω-P81 3B 层位扫描——L28/L34 双救援（判决 layer_rescue），3083 C 族退化定位为层位效应 [%(created)s]

**判决：`layer_rescue`**（预注册判决树：某层三族 n_neg 全 ≥8 → layer_rescue。qwen2.5-3b-instruct bf16 描述性层位扫描，银行层无关每族采一次，L_INJ ∈ {28, 31, 33, 34}、L_POST=L_INJ+1，seed 3084，**5298 前向 / %(el).1f 秒**，家族锚 b0/b1/b3/b6/b7a + 每层 b4/b8 全 bit 0）。

### 核心结果（重复三遍）
**① 层位效应确认（一）**：L_INJ=33（3083 位置）完整复现 3083 退化——med_c 全扫描最低（0.072–0.086）、R_ALL 近零（+0.011/+0.023/−0.041）、C 族 n_neg=2/16；而 **L28 与 L34 双救援**（三族 n_neg 12/11/15 与 12/12/11，全部 ≥8）→ **3083 的 C 族退化是层位效应，不是实现故障，也不是真实族阴性**。**② L34 信号最强（二）**：med_c 0.310/0.222/0.360（全扫描最高，4B 量级）、R_ALL −0.281/−0.227/−0.302（全扫描最强）、capture8 0.923/0.896/0.945——与 qwen3-4B 协议位置（倒数第 2 层）一致，载入机制偏好深层读出附近。**③ 双带结构（三）**：L28（深度 0.78）是第二救援带（R_ALL −0.166/−0.172/−0.118）；L31 是 C 族部分退化（n_neg=6）——救援带不连续，L31/L33 构成低谷。rescue_layers=[28, 34]；rescue_best=28 由最小族 n_neg 平局决胜（11==11）取先者，**幅度上 L34 全面占优**。

### 3083 三候选解释的裁决
(a) **层位假说——成立**：同一族同一管线换层即从退化（C n_neg=2）变为健康（C n_neg=15@L28、11@L34）。(b) **容量/幅度假说——削弱**：L34 的 med_c/R_ALL 达 4B 量级，3B 并非整体响应幅度不足，只是 L33 处坍缩。(c) **真实族阴性——否定**：C 族在 L28/L34 完全健康。

### 理论更新（第一性原理）
**3B 的因果载入呈双带结构**：mid-deep 带（L28，深度 0.78）+ deep-readout 带（L34，倒数第 2），中间 L31/L33 低谷——候选解读：载入可能存在两类通道（深层语义整合期注入 vs 读出前注入），或层间存在功能分化带；此为描述性观察，机制归因未证明。**3083 的仲裁问题恢复可检验性**：在 L_INJ=34 重跑 3083 完整管线（E4 谱结构 + G_DS 门 + 跨族统计）即可完成被推迟的四格 trunk/dispersed × migrates/no 仲裁。层位敏感性本身是新的一等公民数据：**焦点头机器的可用性是（模型 × 族 × 层）三元依赖**。

### 硬伤与边界
- 粗粒度 4 层网格：真实载入带边界未定位（29/30/32 未测；"双带"与"单宽带加两个异常点"不可区分）。
- 描述性扫描：无置换检验、无幅度门（n_neg≥8 为 3071 焦点框架要求，预注册复用）；E4 谱未按层收集。
- rescue_best=28 是 tie-break，非优势证据；层选择应结合幅度（L34）。
- 同一 3076 文本集；tied embeddings 模型（已记录）；单模型（4B/DS7B 对照未做）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3084/omega_p81_3b_layer_scan/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；%(fw)d forwards / %(el).1fs。

**接续 3085 菜单**——A（主选）**L34 完整 3083 管线四格仲裁**（L_INJ=34 全管线：E4 谱结构 + G_DS 门 + 跨族统计，完成被推迟的 trunk/dispersed × migrates/no 仲裁；中量前向 ~14k）。B **L28 复制对照**（第二救援带上重跑，检验救援带间判决一致性）。C **细粒度层位剖面**（29/30/32 补测 + 全层 med_c/R_ALL 剖面，定位低谷结构；免置换轻量）。D **4B 层位对照**（qwen3-4b 同扫描，检验双带是否模型普遍）。E **免前向理论**：双带载入的结构解释 + 谱-迁移连续统检验设计。"好的，继续"即进 3085 A。
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
if '## 四十六、3084' not in aud:
    add = u'''
---
## 四十六、3084 增补：3B 层位扫描——L28/L34 双救援、判决 layer_rescue（Omega-P81）
1. **判决**：描述性层位扫描（L_INJ ∈ {28,31,33,34}，5298 前向，锚全 bit 0）——L28 与 L34 三族 n_neg 全 ≥8（12/11/15 与 12/12/11），L33 复现 3083 退化（C n_neg=2、med_c 最低、R_ALL 近零）→ **3083 C 族退化=层位效应**；rescue_best=28（最小族 n_neg tie-break 11==11）。
2. **信号结构**：L34 全扫描最强（med_c 0.22–0.36、R_ALL −0.23~−0.30、capture8 0.90–0.95，与 4B 协议位置一致）；L28 mid-deep 第二带；L31 C 族部分退化（n_neg=6）——救援带间存在低谷。
3. **HDMCC 更新**：3B 因果载入呈候选双带结构（mid-deep + deep-readout）；3083 候选解释中容量/幅度假说削弱、真实族阴性否定、层位假说成立；焦点头机器可用性=模型×族×层三元依赖；四格仲裁在 L_INJ=34 恢复可检验——3085 主选 L34 完整管线。
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
if 'Phase 3084' not in prev:
    line = ('- Phase 3084 Omega-P81 3B '
            'layer scan (qwen2.5-3b-'
            'instruct, L_INJ in 28/31/33/'
            '34, 5298 forwards, '
            'descriptive): verdict '
            'layer_rescue - L28 and L34 '
            'rescue all families (n_neg '
            '12/11/15 and 12/12/11) while '
            'L33 reproduces the 3083 C '
            'degeneration (n_neg=2) - '
            'layer-position effect '
            'confirmed.  L34 strongest '
            '(med_c 0.22-0.36, R_ALL '
            '-0.23..-0.30); two-band '
            'loading candidate; 3085 '
            'opens four-way arbitration '
            'at L34.  Audit 46; ledger '
            '223/L14 191.\n')
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
if 'max=3084' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L 4kv 3584 bf16）、qwen2.5-3b-instruct（Qwen2 36L 16H 2kv 2048 bf16 **tied embeddings**，3083 位 L_INJ=33；3084 层位扫描 rescue=[28,34]）；glm4-9b-chat-hf 备用。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout/verify tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（**L14.connects 含旧 str 条目，必须 isinstance(c, dict) 防御**）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→独立 verify→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json（脚本自删亦可）；负结果与**预注册退化出口**如实登记为一等公民——退化后禁止放宽门限（防 post-hoc）。
4. 统计纪律：阈值预注册；置换/偏相关 p 的实现必须与主脚本 bit 一致（partial p 主脚本=cnt/n_perm 无 +1 平滑）；泛化声明分层。

## 标准锚与精度
- bit 锚家族：…跨模型 setup 锚（3081）、免前向输入冻结+重放锚（3082）、退化出口锚（3083）、**层位扫描锚（3084：家族锚 b0/b1/b3/b6/b7a + 每层 b4/b8，FAM_ANCH 只收集 endswith _ok 后缀的键防 0.0 falsy）**。
- b8 语义：注入前向内部块链连续性；置换 rng seed=phase 号。

## 机制解释审计链（命名前依次检查）
…→3080 ab_cos_locked→3081 ds7b_cos_absent→3082 ds7b_decorrelated→3083 third_top8_degenerate（qwen2.5-3b C 族 L33 退化）→**3084 layer_rescue（层位扫描：L28/L34 双救援、L34 幅度最强；3083 退化=层位效应；载入候选双带结构；四格仲裁恢复可检验）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail 坏→python + 写文件；日志用 Read；-c stdout 丢→写文件再 Read；Glob 对部分目录失效→python os.listdir 为准。
- 关键写入后必须 Grep/Read 复核；改后必编译检查；result.json 无 smoke 键，npz 里 SMOKE 标量为准。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3084）
Ω-P2（3011-3084）：…3082 ds7b_decorrelated；3083 third_top8_degenerate；**3084 layer_rescue（3B 载入双带候选：L28+L34，L31/L33 低谷）**。

## 下一步
- max=3084，下一个 3085（A 主选 **L34 完整 3083 管线四格仲裁**：E4 谱+G_DS+跨族统计；B L28 复制；C 细粒度层位剖面 29/30/32；D 4B 层位对照；E 免前向双带理论分析）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars'
             % len(mem_new))
else:
    o.append('memory already max=3084')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
