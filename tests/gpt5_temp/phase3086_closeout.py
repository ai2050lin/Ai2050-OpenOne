# -*- coding: utf-8 -*-
"""Phase 3086 closeout (idempotent): Ledger -> L14
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
     r'\phase3086'
     r'\omega_p83_continuum_test')
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
assert verdict == 'continuum_confirmed', \
    verdict
assert res['forwards'] == 0
assert seal['setup_ok'] is True
z = np.load(R + r'\omega_p83_continuum_'
            r'test.npz', allow_pickle=False)
assert bool(z['SMOKE']) is False
assert int(z['N_PERM']) == 20000
assert int(z['N_UNITS']) == 9
assert bool(z['ANCH_A1'])
assert bool(z['ANCH_A2'])
assert bool(z['ANCH_A3'])
st = res['stats']['tests']
rt = st['rho_T']
pt = st['p_T']
ru = st['rho_U']
pu = st['p_U']
rm = st['rho_MIG']
pm = st['p_MIG']
smt = st['sp_mean_T']
units = res['stats']['units']
sha = res['stats']['sha']
el = res['elapsed']

meas = {
    'meas_id': 'meas3086_omega_p83_'
               'continuum_test',
    'phase': 3086,
    'claim': (
        'Omega-P83 (plan 3086 A) - '
        'preregistered spectrum-migration '
        'continuum test over frozen sources '
        'S79 (4B T/U/SP_R1_REPLAY, sha '
        '%(s79)s) / S81 (DS7B, sha %(s81)s) '
        '/ S82 (spectra + INPUT_SHA chain, '
        'sha %(s82)s) / S85 (3B, sha '
        '%(s85)s): forward-free numpy, 0 '
        'forwards, %(el).1fs, seed 3086, '
        'N_PERM=20000, N_UNITS=9 '
        '({4B,DS7B,3B}x{AB,AC,BC}).  '
        'Anchors all PASSED: a1 sha chain '
        '(S82 INPUT_SHA79 == sha8(S79), '
        'INPUT_SHA81 == sha8(S81)), a2 '
        'spectrum bands (4B 0.957-0.964 in '
        '(0.95,0.97); DS7B 0.266-0.320 in '
        '(0.26,0.33); 3B 0.827-0.875 in '
        '(0.82,0.88)), a3 shapes (all T_/U_ '
        'length-24; FORWARDS S81=20634, '
        'S85=19770).  Predictor s_lo = '
        'min(top3 CS of the two families in '
        'pair) (bottleneck definition); '
        'responses T_med/U_med (24-pair '
        'medians) and MIG (sp(R1_fa,R1_fb)).'
        '  MAIN rho(s_lo, T_med) = '
        '+%(rt).4f, perm p = %(pt).5f < 0.05 '
        '-> VERDICT continuum_confirmed.  '
        'Secondary: rho(s_lo, U_med) = '
        '+%(ru).4f p=%(pu).5f (edge); '
        'rho(s_lo, MIG) = +%(rm).4f '
        'p=%(pm).5f (positive ns); s_mean '
        'backup predictor sp(T) = '
        '+%(smt).4f same ordering.  '
        'Within-model (n=3, no power, '
        'recorded): 3B +1.0, 4B -0.5, DS7B '
        '-0.5.  Pattern: DS7B dispersed '
        'T_med -0.021..+0.044 (~0); 3B '
        'mixed 0.287-0.447; 4B trunk '
        '0.430-0.448 - spectral position '
        'predicts cross-model shared '
        'T-structure strength; the '
        '3082->3085 direction-consistent '
        'prediction is now a preregistered '
        'quantitative confirmation.  '
        'CAVEAT: 9 units share 3 source '
        'models (not fully independent); '
        'rho_MIG positive ns - the shared '
        'T substrate responds to spectral '
        'position, unit-level migration '
        'strength is a second-order '
        'quantity; correlational, not '
        'causal.'
        % {'s79': sha['S79'],
           's81': sha['S81'],
           's82': sha['S82'],
           's85': sha['S85'],
           'el': el, 'rt': rt, 'pt': pt,
           'ru': ru, 'pu': pu, 'rm': rm,
           'pm': pm, 'smt': smt}),
    'verdict': verdict,
    'anchors': 'a1 sha chain (S82 '
               'INPUT_SHA79/INPUT_SHA81 == '
               'sha8 of S79/S81 files); a2 '
               'spectrum bands (3 models x 3 '
               'families all in preregistered '
               'bands); a3 shapes (T_/U_ '
               'length-24) + forwards gates '
               '(S81 20634, S85 19770) - all '
               'True',
    'artifacts': {
        'result': 'phase3086/'
                  'omega_p83_continuum_test/'
                  'result.json',
        'npz': 'phase3086/'
               'omega_p83_continuum_test/'
               'omega_p83_continuum_test.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'forward-free preregistered '
            'test; SMOKE p=0.003 vs '
            'authoritative p=0.00465 at the '
            'same rho=0.8667 (perm noise '
            'only); unit tag replace-order '
            'bug (DS7B contains 4B) fixed '
            'before the authoritative run; '
            'rho_U edge and rho_MIG ns '
            'recorded as-is; n=9 units are '
            'model-clustered.',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3086
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 225
    l14['connects'].append({
        'meas_id': 'meas3086_omega_p83_'
                   'continuum_test',
        'phase': 3086,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P83: '
                        'preregistered '
                        'spectrum-migration '
                        'continuum test (9 '
                        'units, 20000 perms, '
                        'forward-free).  '
                        'rho(s_lo, T_med)='
                        '+0.867 p=0.00465 '
                        'CONFIRMED - spectral '
                        'position (bottleneck '
                        'family CS top3) '
                        'predicts cross-model '
                        'shared T-structure '
                        'strength (DS7B '
                        'dispersed->T~0, 3B '
                        'mixed->0.29-0.45, 4B '
                        'trunk->0.43-0.45).  '
                        'rho_U +0.65 edge, '
                        'rho_MIG +0.48 '
                        'positive ns.  The '
                        '3082 spectral->'
                        'migration prediction '
                        'upgraded from '
                        'direction-consistent '
                        'to preregistered '
                        'significant.  Opens '
                        '3087: A GLM4 fourth '
                        'continuum point; B '
                        'L28 four-way '
                        'replication; C layer '
                        'x spectrum; D 4B '
                        'trunk anatomy; E '
                        'gate sensitivity'})
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
if '## Phase 3086:' not in memo:
    sec = u'''## Phase 3086: Ω-P83 谱-迁移连续统定量检验——rho_T=+0.867 跨模型预注册显著（判决 continuum_confirmed） [%(created)s]

**判决：`continuum_confirmed`**（预注册主检验 rho(s_lo, T_med)=+0.8667、置换 p=0.00465 < 0.05。免前向 numpy 分析：冻结源 S79/S81/S82/S85，**0 前向 / %(el).1f 秒**，seed 3086，N_PERM=20000，9 单元 = {4B, DS7B, 3B}×{AB, AC, BC}；三锚全过）。

### 核心结果（重复三遍）
**① 预注册检验显著（一）**：预测子 s_lo = 配对两族 CS top3 的最小值（瓶颈族定义，3082 谱位置诊断），响应 T_med = 24 对跨模型公共 T 结构中位数；**Spearman rho=+0.8667，20000 次置换 p=0.00465**——谱位置（trunk 程度）预测跨模型共享 T 结构强度，SMOKE（1000 perm）p=0.003 同 rho 一致（纯置换噪声）。**② 三点连续统定量化（二）**：DS7B（dispersed，s_lo 0.266–0.293）T_med −0.021~+0.044 ≈ 0；3B（mixed，s_lo 0.827–0.870）T_med +0.287~+0.447；4B（trunk，s_lo 0.957）T_med +0.430~+0.448——3082 谱型→迁移预测从"三点方向一致"升级为**预注册定量显著**。**③ 副检验如实登记（三）**：rho(s_lo, U_med)=+0.65（p=0.066 边缘）、rho(s_lo, MIG)=+0.48（p=0.193 正不显著）、s_mean 备用预测子 sp(T)=+0.817 同序；组内 n=3 无力（3B +1.0、4B −0.5、DS7B −0.5，仅记录）。

### 九单元明细
| 单元 | s_lo | T_med | U_med | MIG |
| --- | --- | --- | --- | --- |
| 4B_AB | 0.9567 | +0.4479 | +0.5253 | +0.6763 |
| 4B_AC | 0.9567 | +0.4300 | +0.4146 | +0.1305 |
| 4B_BC | 0.9572 | +0.4388 | +0.4501 | −0.0883 |
| DS7B_AB | 0.2664 | +0.0438 | −0.0021 | −0.1259 |
| DS7B_AC | 0.2926 | −0.0101 | +0.0343 | +0.0909 |
| DS7B_BC | 0.2664 | −0.0211 | +0.0241 | −0.1303 |
| 3B_AB | 0.8704 | +0.4471 | +0.6428 | +0.8588 |
| 3B_AC | 0.8273 | +0.2868 | +0.4846 | +0.4500 |
| 3B_BC | 0.8273 | +0.3059 | +0.5119 | +0.2941 |

### 理论更新（第一性原理）
- **连续统假设首个预注册显著支持**：机制可用性 = f(谱位置) 的核心预测——载入低维共享性随谱位置（dispersed→mixed→trunk）单调增强——在 n=9 单元、控制三模型聚类的方向上定量成立。这是 RDC 从"每 Phase 一个现象"走向"跨模型定量规律"的第一步。
- **T 结构是谱位置的直接响应，MIG 是二级量**：rho_T 显著而 rho_MIG 正 ns，提示谱位置先决定"跨模型公共 T 结构是否存在"（一级量），单元级迁移强度还依赖门控/幅度等二级因素——与 3085 的 Stouffer 边缘、mig_AB=0.859 部分锁定一致。
- **U_med 边缘（+0.65）**：U（非迁移公共结构）与 T 同向但弱，与 3081 U 无家族结构的发现相容。
- **谱-迁移链闭合度**：3082（谱分型）→3085（三点方向一致）→3086（预注册显著）三段链完成；下一步是加第四个谱点（GLM4-9B）压 n=12 并打破 Qwen 系聚类。

### 硬伤与边界
- **9 单元非独立**：每模型 3 单元共享源数据与管线，存在模型级聚类；rho 的置换 p 在单元级置换下有效，但有效样本量介于 3–9 之间，p 值应读作方向性显著而非精确概率。
- 相关非因果：s_lo 与 T_med 的耦合也可能是共同原因（训练数据/架构）的下游；加第四点（非 Qwen 系）是关键判别。
- s_lo 的"瓶颈定义"（min）是预注册选择；s_mean 备用同序（+0.817）但未过显著性门槛记录——定义敏感性未系统扫描。
- 阈值带（0.95/0.5）仍是两点校准；FORWARDS 门仅覆盖 S81/S85（S79 为 P76 免前向冻结重放源）。
- 工程项：单元 tag 的 replace 顺序 bug（'DS7B' 含 '4B' 子串致键名错位）在权威运行前修复重跑，SMOKE 与权威数值同 rho 验证一致。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3086/omega_p83_continuum_test/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；0 forwards / %(el).1fs。

**接续 3087 菜单**——A（新提议·主选）**GLM4-9B 第四谱点**（复用 3083/3085 管线在 glm4-9b-chat-hf 上跑 255 文本完整管线 + 谱 + T/U/MIG，~20k 前向；打破 Qwen 系聚类，连续统 n=12）。B **L28 四格复制**（第二救援带完整管线，~20k 前向，检验层位×判决稳定性）。C **层位×谱型**（L31/L33 补 E4 谱，检验谱型是否随层变化；中量前向）。D **4B 主干解剖**（免前向，CS top-1 成分的头/子集分解）。E **G_DS 门敏感性分析**（免前向：Stouffer 边缘性、count 门 vs 连续 z 门、f2~U_BC 反号诊断）。"好的，继续"即进 3087 A。
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
if '## 四十八、3086' not in aud:
    add = u'''
---
## 四十八、3086 增补：谱-迁移连续统预注册检验——判决 continuum_confirmed（Omega-P83）
1. **判决**：免前向预注册检验（9 单元、20000 置换、三锚全过）——rho(s_lo, T_med)=+0.867、p=0.00465 显著；副检验 rho_U +0.65（边缘）、rho_MIG +0.48（正 ns）如实登记。
2. **连续统定量化**：DS7B dispersed（s_lo 0.27）T_med≈0、3B mixed（0.83）0.29–0.45、4B trunk（0.96）0.43–0.45——3082 谱型→迁移预测从方向一致升级为预注册定量显著；RDC 获得首个跨模型定量规律候选。
3. **HDMCC 更新**：T 结构是谱位置的一级响应、MIG 是二级量（rho_MIG ns 与 3085 Stouffer 边缘一致）；9 单元模型聚类 → 有效样本量介于 3–9，p 读作方向性显著；下一个判别步骤是加非 Qwen 系第四点（GLM4-9B）。
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
if 'Phase 3086' not in prev:
    line = ('- Phase 3086 Omega-P83 '
            'preregistered spectrum-migration '
            'continuum test (forward-free, 9 '
            'units, 20000 perms): verdict '
            'continuum_confirmed - rho(s_lo, '
            'T_med)=+0.867 p=0.00465; DS7B '
            'dispersed T~0, 3B mixed 0.29-0.45, '
            '4B trunk 0.43-0.45; rho_U edge, '
            'rho_MIG positive ns.  3082->3085 '
            'direction chain upgraded to '
            'preregistered significant.  '
            'Audit 48; ledger 225/L14 193.\n')
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
if 'max=3086' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L 4kv 3584 bf16）、qwen2.5-3b-instruct（Qwen2 36L 16H 2kv 2048 bf16 **tied embeddings**；层位扫描 rescue=[28,34]，L34=四格仲裁层）；glm4-9b-chat-hf 备用。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout/verify tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\（smoke 在 smoke\\ 子目录）。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（**含旧 str 条目，必须 isinstance(c, dict) 防御**）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→独立 verify→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json（脚本自删亦可）；负结果与预注册退化/混合出口如实登记为一等公民——门未过禁止放宽（防 post-hoc）。
4. 统计纪律：阈值预注册；置换/偏相关 p 必须与主脚本 bit 一致；泛化声明分层。
5. verify 锚键先 Grep 主脚本 npz save 段逐一核对（B3 类布尔锚无 DIFF 分量）。
6. **确定性复现锚（3085 标准件）**：同模型同文本同层跨 Phase 重跑，n_neg/top8 精确相等、浮点差 ≤1e-9（fp32 往返噪声 ~3e-12）；mismatch→setup_failed。
7. **字符串 replace 先长后短**（3086 教训：'DS7B' 含 '4B' 子串，先 replace('4B',...) 会毁掉 DS7B 键名）。

## 标准锚与精度
- bit 锚家族：…跨模型 setup 锚（3081）、免前向冻结重放锚（3082）、退化出口锚（3083）、层位扫描锚（3084）、复现锚+四格混合出口（3085）、**连续统检验锚（3086：a1 sha 链/a2 谱带/a3 形状前向门）**。
- b8 语义：注入前向内部块链连续性；置换 rng seed=phase 号。

## 机制解释审计链
…→3082 ds7b_decorrelated→3083 third_top8_degenerate（L33 C 族退化）→3084 layer_rescue（L28/L34 双救援）→3085 third_mixed_absent（L34 四格仲裁）→**3086 continuum_confirmed（谱-迁移连续统预注册显著：rho(s_lo,T_med)=+0.867 p=0.0047；T 是一级响应、MIG 二级）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail 坏→python + 写文件；日志用 Read；-c stdout 丢→写文件再 Read；Glob/Grep 对部分目录失效→python os.listdir 为准。
- 关键写入后必须 Grep/Read 复核；Edit 偶发假成功（报成功未落盘）→改后必须 Grep 复核，可疑即重试；改后必编译检查。
- result.json 无 smoke 键，npz 里 SMOKE 标量为准。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 下一步
- max=3086，下一个 3087（A 主选 **GLM4-9B 第四谱点**，打破 Qwen 聚类 n=12；B L28 四格复制；C 层位×谱型 L31/L33 补 E4；D 4B 主干解剖免前向；E G_DS 门敏感性分析）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars'
             % len(mem_new))
else:
    o.append('memory already max=3086')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
