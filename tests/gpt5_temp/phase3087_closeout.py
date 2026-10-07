# -*- coding: utf-8 -*-
"""Phase 3087 closeout (idempotent): Ledger -> L14
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
     r'\phase3087'
     r'\omega_p85_glm4_l37_full_arbitration')
R_A1 = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3087'
        r'\omega_p84_glm4_layer_scan')
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
assert verdict == 'fourth_mixed_absent', verdict
assert seal['setup_ok'] is True
z = np.load(R + r'\omega_p85_glm4_l37_full_'
            r'arbitration.npz',
            allow_pickle=False)
assert bool(z['SMOKE']) is False
assert int(z['FORWARDS']) == res['forwards']
assert int(z['FORWARDS']) == 20922
assert int(z['L_INJ']) == 37
assert int(z['L_POST']) == 38
assert bool(z['SETUP_OK'])
assert bool(z['REPRO_OK'])
assert str(z['SPEC_CLASS']) == 'mixed'
za1 = np.load(R_A1 + r'\omega_p84_glm4_layer_'
              r'scan.npz', allow_pickle=False)
assert str(za1['VERDICT']) == 'layer_rescue'

top3 = {fk: float(z['E3_TOP3_CS_' + fk])
        for fk in 'ABC'}
top3h = {fk: float(z['E3_TOP3_CS1H_' + fk])
         for fk in 'ABC'}
s_lo = {'AB': min(top3['A'], top3['B']),
        'AC': min(top3['A'], top3['C']),
        'BC': min(top3['B'], top3['C'])}
tmed = {k: float(np.median(z['T_' + k]))
        for k in ('AB', 'AC', 'BC')}
umed = {k: float(np.median(z['U_' + k]))
        for k in ('AB', 'AC', 'BC')}
mig = {k: float(z['MIG_' + k])
       for k in ('AB', 'AC', 'BC')}
nneg = {fk: int(z['N_NEG_' + fk])
        for fk in 'ABC'}
medc = {fk: float(z['MED_C_' + fk])
        for fk in 'ABC'}
rall = {fk: float(z['R_ALL_' + fk])
        for fk in 'ABC'}
st_z = float(z['STOUFFER_Z'])
gds_cnt = int(z['GDS_COUNT'])
gds_min = float(z['GDS_MIN_SP'])
el = res['elapsed']
fw = res['forwards']

# A1 per-layer table strings
a1_rows = []
for L in (31, 34, 37, 38):
    nn = '/'.join(str(int(za1['N_NEG_L%d_%s'
                           % (L, fk)]))
                  for fk in 'ABC')
    mc = '/'.join(('%.3f' % float(
        za1['MED_C_L%d_%s' % (L, fk)]))
        for fk in 'ABC')
    ra = '/'.join(('%.3f' % float(
        za1['R_ALL_L%d_%s' % (L, fk)]))
        for fk in 'ABC')
    a1_rows.append('| L%d | %s | %s | %s |'
                   % (L, nn, mc, ra))
a1_table = '\n'.join(a1_rows)

meas = {
    'meas_id': 'meas3087_omega_p84_p85_'
               'glm4_fourth_point',
    'phase': 3087,
    'claim': (
        'Omega-P84/P85 (plan 3087 A) - '
        'GLM4-9B fourth spectrum point, '
        'two arms: A1 layer scan '
        '(omega_p84_glm4_layer_scan, L31/'
        '34/37/38, 9906 forwards, verdict '
        'layer_rescue, rescue=[34,37,38], '
        'best=L37 by min-family n_neg 12) '
        'and A2 full four-way arbitration '
        '(omega_p85_glm4_l37_full_'
        'arbitration, L_INJ=37/L_POST=38, '
        '%(fw)d forwards / %(el).1fs, seed '
        '3087, 32 heads, vocab 151552, '
        'tied=False, bf16 with driver '
        'sysmem fallback).  REPRO anchor '
        'vs A1 L37 PASSED per family '
        '(B/C med_c and R_ALL bit-0, A '
        '<=4.9e-14/4.5e-13/1.8e-13 vs tol '
        '1e-9) - GLM4 pipeline '
        'determinism established.  '
        'Spectrum class MIXED (CS top3 '
        'A=0.6992 B=0.7858 C=0.7442; '
        'CS1H 0.5597/0.6721/0.5441) '
        'between DS7B dispersed (0.27-'
        '0.32) and 3B mixed (0.827-'
        '0.870).  Migration gate G_DS '
        'count=0/6 (min_sp=%(gm).4f), '
        'Stouffer z=+%(z).3f one-sided '
        'edge; f2~T all positive ns (AB '
        '+0.3435 p=0.103 / AC +0.3470 '
        'p=0.098 / BC +0.2496 p=0.241); '
        'mig_fg 0.250/0.164/-0.061.  '
        'VERDICT fourth_mixed_absent - '
        'SAME CELL as 3085 '
        'third_mixed_absent (qwen2.5-3b). '
        'Trunk+locked migration remains '
        'observed ONLY at qwen3-4b '
        '(3082) after breaking the Qwen '
        'lineage.  Continuum n=9->12 '
        'data ready: GLM4 units s_lo '
        'AB=0.6992 AC=0.6992 BC=0.7442. '
        'CAVEATS: A2 layer choice used '
        'the n_neg criterion (3084/3085 '
        'precedent); the A1 criterion '
        'split (L37 n_neg vs L38 E1 '
        'magnitude, 4-6x) is unresolved '
        '- an L38 arbitration replica is '
        'the discriminant; spectrum '
        'thresholds two-point calibrated; '
        'n=24 partial orientation only.'
        % {'fw': fw, 'el': el,
           'gm': gds_min, 'z': st_z}),
    'verdict': verdict,
    'anchors': 'A1: per-layer b0/b1/b3/'
               'b4/b6/b7a/b8 all bit-0 per '
               'family; A2: same anchor set '
               'at L_INJ=37 bit-0 per family '
               '+ repro anchor vs A1 L37 npz '
               '(n_neg/top8 exact, med_c/'
               'CS1H/R_ALL <= 1e-9) all '
               'True',
    'artifacts': {
        'result': 'phase3087/'
                  'omega_p85_glm4_l37_full_'
                  'arbitration/result.json',
        'npz': 'phase3087/'
               'omega_p85_glm4_l37_full_'
               'arbitration/'
               'omega_p85_glm4_l37_full_'
               'arbitration.npz',
        'a1_scan': 'phase3087/'
                   'omega_p84_glm4_layer_scan/'
                   'omega_p84_glm4_layer_scan'
                   '.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8'],
        'a1_npz_sha256_8': '912ec7df'},
    'note': 'two-arm phase (A1 scan -> A2 '
            'arbitration); SMOKE all anchors '
            'bit-0; generator bad-list now '
            'includes the OUT phase-directory '
            'token (phase3085->phase3087 miss '
            'caught by the smoke artifact '
            'location check); legacy '
            '"28-head" log text replaced by '
            'model-agnostic "full attn swap"',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3087
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 226
    l14['connects'].append({
        'meas_id': 'meas3087_omega_p84_p85_'
                   'glm4_fourth_point',
        'phase': 3087,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P84/P85: '
                        'GLM4-9B fourth '
                        'spectrum point.  '
                        'A1 layer_rescue '
                        '(best=L37; criterion '
                        'split L37 n_neg vs '
                        'L38 magnitude).  A2 '
                        'fourth_mixed_absent '
                        '- same cell as 3085 '
                        'third_mixed_absent; '
                        'repro anchor bit-0 '
                        '(B/C).  Trunk+locked '
                        'migration still '
                        'exclusive to '
                        'qwen3-4b after '
                        'breaking the Qwen '
                        'lineage.  Continuum '
                        'n=12 extension data '
                        'ready (s_lo '
                        '0.6992/0.6992/'
                        '0.7442) - opens '
                        '3088: A continuum '
                        'n=12 discriminant '
                        'test; B L38 '
                        'arbitration replica; '
                        'C G_DS gate '
                        'sensitivity; D 4B '
                        'trunk anatomy; E '
                        'layer x spectrum'})
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
if '## Phase 3087:' not in memo:
    sec = u'''## Phase 3087: Ω-P84/P85 GLM4-9B 第四谱点——A1 层位扫描 layer_rescue（best=L37、判据分歧）+ A2 四格仲裁 fourth_mixed_absent（与 3085 同格） [%(created)s]

**判决：A1（Ω-P84）`layer_rescue` + A2（Ω-P85）`fourth_mixed_absent`**。两臂：A1 层位扫描（40 层等比映射 L31/34/37/38，9906 前向）→ A2 完整 255-subset 仲裁管线（L_INJ=37/L_POST=38，**20922 前向 / %(el).1f 秒**，seed 3087，32 头 / vocab 151552 / tied=False / bf16 sysmem fallback）。

### 核心结果（重复三遍）
**① A1 判决 layer_rescue 且出现判据分歧（一）**：B 族 L31 退化（n_neg=4）出局，rescue 集 [34,37,38]；n_neg 判据 best=**L37**（12/24/14），但 E1 幅值判据 best=**L38**（med_c 0.192/0.200/0.132、R_ALL −0.12~−0.18，比 L37 强 4-6 倍）——**负头覆盖广度与装载幅值深度在 GLM4 救援带内解耦**（3084 的 3B 上两判据同点 L34）。

**② A2 判决 fourth_mixed_absent、与 3085 同格（二）**：谱类 mixed（CS top3 A=0.6992/B=0.7858/C=0.7442；CS1H 0.560/0.672/0.544），落在 DS7B dispersed（0.27−0.32）与 3B mixed（0.827−0.870）之间；迁移门 G_DS **count=0/6**（min_sp=−0.2443）、Stouffer z=+1.564（单侧 p≈0.059 边缘）；f2~T 三对全正但不显著（AB +0.3435 p=0.103 / AC +0.3470 p=0.098 / BC +0.2496 p=0.241）；mig_fg 0.250/0.164/−0.061。**打破 Qwen 系聚类后，trunk+locked 迁移仍只在 qwen3-4b（3082）观察到**——qwen2.5-3b（P3）与 GLM4-9B（P4）两个独立谱点收敛于 mixed_absent。

**③ repro 锚 bit 级通过、GLM4 确定性确立（三）**：A2 vs A1 L37——B/C 族 med_c 与 R_ALL **diff=0（bit 级相同）**、A 族 ≤4.9e-14/4.5e-13/1.8e-13，远低于 1e-9 容差；n_neg/top8 三族精确相等。bf16 sysmem fallback（18.84GB alloc > 17.09GB VRAM）无数值扰动。

### A1 四层位明细（n_neg A/B/C | med_c | R_ALL）
| 层 | n_neg | med_c | R_ALL |
| --- | --- | --- | --- |
%(a1_table)s

### 四谱点判例全景
| 谱点 | 模型 | 谱类（CS top3） | 判决 |
| --- | --- | --- | --- |
| P1 | qwen3-4b | trunk（0.957−0.964） | trunk_migrates/locked（3082） |
| P2 | DS7B-7B | dispersed（0.266−0.320） | ds7b_cos_absent（3081） |
| P3 | qwen2.5-3b | mixed（0.827−0.870） | third_mixed_absent（3085） |
| P4 | GLM4-9B | mixed（0.699−0.786） | fourth_mixed_absent（3087） |

### 理论更新（第一性原理）
- **trunk 稀有性获得第四点支持**：非 Qwen 系加入后，"谱 trunk + 头迁移锁定"仍是孤例（qwen3-4b）；混合谱 + 无锁定迁移在两个独立架构上复现——trunk 形态不是 Qwen 系普遍属性，候选解释收窄到 qwen3 系特征（架构/训练配方）。
- **mixed 不是均匀类**：GLM4（s_lo 0.699）比 3B（0.827）更靠近 dispersed 侧但同得 absent——按 3086 连续统预测（s_lo→T_med 单调），GLM4 的 T_med 应落在 DS7B（≈0）与 3B（+0.29~+0.45）之间；n=12 判别时刻到来：rho 保持→跨架构规律，崩溃→Qwen 系内规律。
- **判据分歧=层内分工候选**：L37（负头覆盖广）与 L38（幅值深）解耦提示 GLM4 装载机制沿层存在"覆盖-强度"分工；A2 若在 L38 重跑可能得不同判决——3088 B 判别。
- **MIG 与连续统方向一致**：GLM4 mig_fg（0.250/0.164/−0.061）低于 3B（0.859/0.450/0.294）高于 DS7B（≈0），与低谱位→低迁移的 3086 图景相容。

### 硬伤与边界
- **层位选择判据敏感性未解决**：A2 沿用 3084/3085 的 n_neg 判据选 L37；若按 E1 幅值选 L38，四格判决可能不同（L38 三族 n_neg 15/12/10 均 ≥8，管线可行）。
- 谱阈值（0.9/0.5）两点校准；mixed 类内谱位跨度大（0.70 vs 0.83）——类内分辨力只能靠连续统数值预测检验。
- f2~T 三对全正但 ns：n=24 下方向一致（Stouffer 边缘）与"无效应"在功效上不可区分。
- 部分相关 n=24 仅方向性；3076 文本为 model independence（非 text independence）。
- 工程：生成器 bad 清单补入 OUT 路径 phase 号（phase3085→phase3087 漏改被 SMOKE 产物位置检查捕获）；遗留 "28-head" log 文本改为模型无关 "full attn swap"（swap 本体 NQW=4096 坐标全置换，计算正确）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3087/`：A1 `omega_p84_glm4_layer_scan/`（npz8 912ec7df）+ A2 `omega_p85_glm4_l37_full_arbitration/`（script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s）；ledger %(n)d / L14 %(l14)d；A2 20922 forwards / %(el).1fs。

**接续 3088 菜单**——A（新提议·主选）**连续统 n=12 判别时刻**（GLM4 三单元 s_lo/T_med/U_med/MIG 纳入 3086 预注册框架重跑，免前向；rho 保持→跨架构规律、崩溃→Qwen 系内规律；同时检验 mixed 类内分辨力）。B **L38 仲裁复制**（A2 管线 L_INJ=38，~21k 前向，判据分歧判别）。C **G_DS 门敏感性分析**（免前向：3085 部分锁定 vs 3087 全灭的门结构对比）。D **4B 主干解剖**（免前向，CS top-1 成分的头/子集分解）。E **层位×谱型**（GLM4 L34 补 E4 谱，中量前向）。"好的，继续"即进 3088 A。
''' % {'created': created,
       'el': el,
       'a1_table': a1_table,
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
if '## 四十九、3087' not in aud:
    add = u'''
---
## 四十九、3087 增补：GLM4-9B 第四谱点——A1 layer_rescue（判据分歧）+ A2 fourth_mixed_absent（与 3085 同格）
1. **判决**：A1 层位扫描 layer_rescue（rescue=[34,37,38]，best=L37 按 n_neg；L38 幅值强 4-6 倍——覆盖/强度解耦）；A2 四格仲裁 fourth_mixed_absent（谱 mixed 0.699−0.786、G_DS 0/6、f2~T 全正 ns、Stouffer z=1.564 边缘）。
2. **trunk 稀有性**：打破 Qwen 系聚类后 trunk+locked 迁移仍只在 qwen3-4b；qwen2.5-3b 与 GLM4-9B 两独立谱点同格 mixed_absent——trunk 候选解释收窄到 qwen3 系。
3. **HDMCC 更新**：mixed 非均匀类（GLM4 s_lo 0.699 vs 3B 0.827），连续统 n=12 判别时刻（GLM4 T_med 预期落在 DS7B≈0 与 3B +0.29~0.45 之间）；repro 锚 bit 级通过确立 GLM4 确定性；层位选择判据敏感性（L37 vs L38）是未决判别项（3088 B）。
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
if 'Phase 3087' not in prev:
    line = ('- Phase 3087 Omega-P84/P85 '
            'GLM4-9B fourth spectrum point, '
            'two arms: A1 layer scan verdict '
            'layer_rescue (best=L37, '
            'criterion split L37 n_neg vs '
            'L38 magnitude) + A2 full '
            'arbitration verdict '
            'fourth_mixed_absent (same cell '
            'as 3085; spectrum mixed '
            '0.699-0.786; G_DS 0/6; repro '
            'anchor bit-0 B/C).  Trunk+'
            'locked migration still '
            'exclusive to qwen3-4b.  '
            'Continuum n=12 data ready.  '
            'Audit 49; ledger 226/L14 194; '
            'A2 20922 forwards.\n')
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
if 'max=3087' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L 4kv 3584 bf16）、qwen2.5-3b-instruct（Qwen2 36L 16H 2kv 2048 bf16 tied）、**glm4-9b-chat-hf（GlmForCausalLM 40L 32H 2kv 4096 inter 13696 vocab 151552 tied=False bf16 18.84GB>17.09GB VRAM 靠 sysmem fallback 非 OOM；repro 锚 bit 级通过）**。
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
7. **字符串 replace 先长后短**（'DS7B' 含 '4B' 子串）。
8. **patch 生成器 bad 清单必须含 OUT 路径 phase 号**（3087 教训：phase3085→phase3087 漏改 OUT，SMOKE 产物写进 phase3085/ 目录——SMOKE 后必查产物目录名）。
9. **Edit 假成功频发**：改后必须 Grep 复核，可疑即重试（3087 又复发 2 次）。

## 标准锚与精度
- bit 锚家族：…跨模型 setup 锚（3081）、免前向冻结重放锚（3082）、退化出口锚（3083）、层位扫描锚（3084）、复现锚+四格混合出口（3085）、连续统检验锚（3086）、**第四谱点双臂锚（3087：A1 层位扫描+判据分歧、A2 repro 锚 bit0）**。

## 机制解释审计链
…→3082 trunk_migrates→3083 third_top8_degenerate→3084 layer_rescue→3085 third_mixed_absent→3086 continuum_confirmed→**3087 A1 layer_rescue（GLM4 判据分歧 L37/L38）+ A2 fourth_mixed_absent（与 3085 同格；trunk 仍 qwen3-4b 独有；连续统 n=12 数据就绪）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail/wc 坏→python + 写文件；日志用 Read；-c stdout 丢→写文件再 Read。
- result.json 无 smoke 键，npz 里 SMOKE 标量为准。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 下一步
- max=3087，下一个 3088（A 主选 **连续统 n=12 判别时刻**，免前向，GLM4 三单元纳入 3086 框架；B L38 仲裁复制判据分歧判别；C G_DS 门敏感性；D 4B 主干解剖免前向；E 层位×谱型）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars'
             % len(mem_new))
else:
    o.append('memory already max=3087')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
