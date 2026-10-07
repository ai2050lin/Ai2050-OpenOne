# -*- coding: utf-8 -*-
"""Phase 3082 closeout (idempotent): Ledger -> L14
-> MEMO append -> HDMCC audit addendum -> wlog
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3082'
     r'\omega_p79_ds7b_negative_anatomy')
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
assert verdict == 'ds7b_decorrelated', verdict
assert res['forwards'] == 0
st = res['stats']
tr = st['tree']
assert tr == {'D': True, 'R1': True,
              'R2': True, 'R3': True}, tr
assert st['E6']['both_significant'] is True
# key magnitudes cited in the claim
assert st['E3']['4B_CS_A']['top3'] > 0.95
assert st['E3']['DS7B_CS_A']['top3'] < 0.35
assert st['E2']['4B_AB']['sf_p'] < 0.05
assert st['E2']['DS7B_AB']['sf_p'] > 0.5
assert st['E4']['DS7B_U_AB']['snr'] < 0.5
assert st['E4']['4B_U_AB']['snr'] > 2.0
el = res['elapsed']

e3 = st['E3']
claim = (
    'Omega-P79 (plan 3082 A) - no-forward '
    'anatomy of the 3081 DS7B negative result '
    '(inputs: sealed 3081/3076/3079/3080 npz, '
    'sha8-verified; 0 forwards, %.1fs).  '
    'PREREGISTERED verdict tree D/R1/R2/R3 '
    'evaluated post-freeze; ALL FOUR true; '
    'D dominates.  VERDICT '
    'ds7b_decorrelated - the migration '
    'absence is a SPECTRUM-STRUCTURE '
    'phenomenon: 4B causal spectra are '
    'low-rank shared (CS 255x24 column-'
    'centered SVD: participation ratio '
    '1.31-1.99, k_eff 1.78-2.70, top-3 '
    'energy %.3f-%.3f), DS7B spectra are '
    'high-rank dispersed (PR %.1f-%.1f, '
    'k_eff %.1f-%.1f, top-3 %.3f-%.3f) - '
    'qwen3-4b has ONE shared causal trunk '
    'per family, DS7B has 24 pair-specific '
    'modes and no trunk.  Corroborating: '
    '(a) E2 focal top8 overlap 4B AB=%d '
    '(hypergeom SF p=%.4f) / AC=%d / BC=%d '
    'vs DS7B %d/%d/%d all at chance '
    '(p=%.2f-%.2f, expect ~%.1f);  (b) '
    'sp(A_S) across families 4B '
    '%.3f/%.3f/%.3f vs DS7B '
    '%.3f/%.3f/%.3f (replayed bit-exact);  '
    '(c) E4 dispersion: T/U spectrum std '
    '4B 0.45-0.54 (SNR 2.2-2.6) vs DS7B '
    'T 0.16-0.17 (SNR 0.76-0.82) and U '
    '0.06-0.08 (SNR %.2f-%.2f) - DS7B '
    'subset-level response is BELOW the '
    'spearman noise floor;  (d) E5 TT '
    'direction alignment itself is NOT '
    'low (med f2 %.3f-%.3f both models) '
    'but rank-direction coupling sp(f1,f2) '
    '4B %.2f-%.2f vs DS7B %.2f-%.2f - '
    'decoupled on DS7B.  CONSERVATIVE '
    'CROSS-MODEL REGULARITY (E6): f2~U_AB '
    'significant on BOTH models (4B '
    '%+.4f p=%.5f; DS7B %+.4f p=%.5f) - '
    'TT alignment conditions subset '
    'transfer even at zero mean; second '
    'model replication achieved, '
    'architecture-level candidate.  '
    'THEORY: the 3080 angle-matching rule '
    'now has a mechanism-level precondition '
    '- a shared low-rank causal trunk '
    '(top3>0.95); PR/top3 of CS is a '
    'single-model, no-intervention '
    'TRANSFERABILITY DIAGNOSTIC; '
    'falsifiable prediction: a third model '
    'with trunk-like spectra should show '
    'cross-family migration, dispersed-'
    'spectra models should not.'
    % (el,
       e3['4B_CS_A']['top3'],
       e3['4B_CS_C']['top3'],
       e3['DS7B_CS_A']['PR'],
       e3['DS7B_CS_B']['PR'],
       e3['DS7B_CS_A']['keff'],
       e3['DS7B_CS_B']['keff'],
       e3['DS7B_CS_A']['top3'],
       e3['DS7B_CS_B']['top3'],
       st['E2']['4B_AB']['overlap'],
       st['E2']['4B_AB']['sf_p'],
       st['E2']['4B_AC']['overlap'],
       st['E2']['4B_BC']['overlap'],
       st['E2']['DS7B_AB']['overlap'],
       st['E2']['DS7B_AC']['overlap'],
       st['E2']['DS7B_BC']['overlap'],
       st['E2']['DS7B_AC']['sf_p'],
       st['E2']['DS7B_BC']['sf_p'],
       st['E2']['DS7B_AB']['expect'],
       st['SP_AS']['4B']['AB'],
       st['SP_AS']['4B']['AC'],
       st['SP_AS']['4B']['BC'],
       st['SP_AS']['DS7B']['AB'],
       st['SP_AS']['DS7B']['AC'],
       st['SP_AS']['DS7B']['BC'],
       st['E4']['DS7B_U_AB']['snr'],
       st['E4']['DS7B_U_BC']['snr'],
       st['E5']['med_f2_4B_AB'],
       st['E5']['med_f2_DS7B_BC'],
       0.67, 0.79, 0.06, 0.43,
       st['E6']['f2u_AB_4B']['sp'],
       st['E6']['f2u_AB_4B']['p'],
       st['E6']['f2u_AB_DS7B']['sp'],
       st['E6']['f2u_AB_DS7B']['p']))

meas = {
    'meas_id': 'meas3082_omega_p79_'
               'ds7b_negative_anatomy',
    'phase': 3082,
    'claim': claim,
    'verdict': verdict,
    'anchors': 'no-forward phase: inputs are '
               'sealed authoritative npz of '
               '3081/3076/3079/3080 (sha8 '
               'recomputed at freeze); internal '
               'replay anchors all bit-exact: '
               'top8 from R1 vectors matches '
               'npz TOP8 (3 families x 2 '
               'models), SP_AS spearman replay '
               'diff 0.0e+00 vs npz (6 pairs), '
               'n_neg replay matches; '
               'deterministic analytic stats '
               '(hypergeometric SF, SVD), no '
               'RNG paths',
    'artifacts': {
        'result': 'phase3082/omega_p79_'
                  'ds7b_negative_anatomy/'
                  'result.json',
        'npz': 'phase3082/omega_p79_'
               'ds7b_negative_anatomy/'
               'omega_p79_ds7b_negative_'
               'anatomy.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8':
            seal['result_sha256_8'],
        'script_sha256_8':
            seal['script_sha256_8']},
    'note': 'no-forward re-analysis on sealed '
            'artifacts.  Caveats: two models '
            'only; layer positions differ (4B '
            'L34/35 vs DS7B L25/26) - PR '
            'layer-robustness untested; E3 is '
            'descriptive, not a mechanism '
            'proof; capture8 baselines differ '
            '(DS7B n_neg 21-27/28 vs 4B '
            '16-20/32); f2~U_AB at DS7B U-SNR '
            '0.27 rests on exact permutation '
            'p (n=24, 20000 perms).',
}
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3082
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 221
    l14['connects'].append({
        'meas_id': 'meas3082_omega_p79_'
                   'ds7b_negative_anatomy',
        'phase': 3082,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P79: DS7B '
                        'negative-anatomy '
                        '(no-forward, sealed '
                        'inputs).  Shape of '
                        'the absence: causal '
                        'spectra high-rank '
                        'DISPERSED (PR 13.6-'
                        '17.1/24, top3 0.27-'
                        '0.32) vs 4B low-rank '
                        'shared trunk (PR 1.3-'
                        '2.0, top3 0.957-'
                        '0.964); focal top8 '
                        'overlap at chance '
                        '(2/1/2, SF p>=0.76) '
                        'vs 4B AB 5 (p=0.012); '
                        'sp(A_S) across '
                        'families ~0.03-0.08 '
                        'vs 4B 0.58-0.83; '
                        'DS7B T/U SNR 0.27-'
                        '0.82 below noise '
                        'floor.  NEW DIAGNOS'
                        'TIC: CS PR/top3 = '
                        'single-model '
                        'transferability '
                        'predictor.  '
                        'Conservative cross-'
                        'model regularity: '
                        'f2~U_AB significant '
                        'on both (4B +0.499 '
                        'p=0.014, DS7B '
                        '+0.563 p=0.0056).  '
                        'Opens 3083: A third-'
                        'model arbitration '
                        'with PR pre-screen; '
                        'B PR layer scan; C '
                        '4B trunk anatomy; D '
                        'f2~U upgrade design'})
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
if '## Phase 3082:' not in memo:
    sec = u'''## Phase 3082: Ω-P79 DS7B 阴性结果解剖——因果谱高秩弥散，无共享主干；PR/top3 成为单模型可迁移性诊断量（ds7b_decorrelated） [%(created)s]

**判决：`ds7b_decorrelated`**（免前向 npz 再分析：输入=3081/3076/3079/3080 四个 sealed 权威 npz（sha8 冻结复核），0 前向、%(el).1f 秒；预注册判决树 D/R1/R2/R3 **全部为真**，D 优先——DS7B 的跨族迁移缺失是**谱结构现象**：**4B 的因果谱低秩共享（CS 255×24 列中心化 SVD：PR 1.31–1.99、k_eff 1.78–2.70、top-3 能量 0.957–0.964），DS7B 高秩弥散（PR 13.6–17.1、k_eff 15.2–19.2、top-3 0.27–0.32）——qwen3-4b 每族有一条共享因果主干，DS7B 的 24 对各有各的模式、无主干**）。

### 核心结果（重复三遍）
**① 谱结构决定性对照（一）**：4B 三族的 24 对 swap 响应由 1–2 个公共成分主导（top3 > 95%%），DS7B 的 24 对各自独立（PR≈14–17/24，接近满秩）——**"共享低秩因果主干"是 4B 跨族迁移（mig +0.676、sp(A_S) 0.58–0.83）存在的结构前提；DS7B 无主干，故迁移全面缺失（mig≈0、sp(A_S) 0.03–0.08）**。**② 焦点路由层面（二）**：4B top8 族间重叠 AB=5（超几何 SF p=0.0115 显著）、AC=4（p=0.082 边缘）；DS7B 2/1/2 全部处于随机（p=0.76/0.96/0.76，期望 2.29）——头级复用同样缺失。**③ 响应谱信噪比（三）**：DS7B 的 T 谱 std 0.16–0.17（SNR 0.76–0.82）、U 谱 std 0.06–0.08（**SNR 0.27–0.37，低于 spearman 噪声底 0.2085**）；4B 全部 SNR 2.2–2.6——3081 的门失败不是边缘失败，是信号本身在噪声水平。另 E5：TT 方向对齐本身不低（med f2 两模型 0.53–0.80），但"秩形-方向"耦合 sp(f1,f2) 4B 0.67–0.79 vs DS7B 0.06–0.43 解耦。**跨模型保守规律（E6）：f2~U_AB 在两模型独立显著**（4B +0.499 p=0.0143、DS7B +0.563 p=0.0056）——"TT 对齐条件化子集迁移"完成第二模型复现，升格架构级候选。

### 理论更新（第一性原理）
**3080 角度匹配规则获得机制级前提解释**：角度匹配之所以能在 4B 上实现装配泛化，是因为其因果响应被 1–2 个公共成分主导——**单主干是"夹角可匹配"的几何前提**；DS7B 无主干，夹角失去可匹配的对象。**新诊断量：CS 的 PR/top3 = 单模型、免干预的可迁移性预测器**——比较模型时无需跑跨模型管线，先算谱结构即可预测跨族/跨模型迁移能力。可证伪预测：第三模型若呈主干型谱（top3 高）应表现跨族迁移，弥散型谱不应表现。

### 硬伤与边界
- 仅两模型；注入层位不同（4B L34/35 vs DS7B L25/26）——PR 的层位稳健性未测。
- E3 是描述统计不是机制证明；PR 差异可能与宽度/头数/训练配方混杂。
- f2~U_AB 在 DS7B 的 U-SNR 0.27 下依赖精确置换 p（n=24、20000 置换、分辨率 1/20001）。
- capture8 基线不同（DS7B n_neg 21–27/28 vs 4B 16–20/32）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3082/omega_p79_ds7b_negative_anatomy/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；0 forwards / %(el).1fs。

**接续 3083 菜单**——A（主选）**第三模型仲裁 + PR 预检**（glm4-9b 或 qwen2.5-3b 跑 3076 管线，同轮算 CS PR/top3，检验"主干型↔迁移存在"预测；重，需前向防 OOM，先 SMOKE 显存适配）。B **PR 层位扫描**（4B 多 L_INJ 重算 CS 谱结构，检验 PR 稳健性；中量前向）。C **4B 主干解剖**（CS top-1 成分的头/子集结构分解，识别共享主干是哪些头承载；免前向为主）。D **f2~U_AB 升格设计**（第三模型上的预注册第三复现）。"好的，继续"即进 3083 A。
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
if '## 四十四、3082' not in aud:
    add = u'''
---
## 四十四、3082 增补：DS7B 阴性结果解剖——高秩弥散谱、无共享主干；可迁移性诊断量（Omega-P79，判决 ds7b_decorrelated）
1. **形态判定**：3081 迁移缺失 = 谱结构现象——4B CS 谱低秩共享（PR 1.31–1.99、top3 0.957–0.964），DS7B 高秩弥散（PR 13.6–17.1、top3 0.27–0.32）；预注册 D/R1/R2/R3 全真，D 优先。
2. **旁证**：焦点 top8 族间重叠 4B AB=5 显著（p=0.0115）vs DS7B 全随机；sp(A_S) 跨族 4B 0.58–0.83 vs DS7B 0.03–0.08；DS7B T/U 谱 SNR 0.27–0.82 低于噪声底。
3. **跨模型保守规律**：f2~U_AB 两模型独立显著（4B +0.499 p=0.014、DS7B +0.563 p=0.006）——TT 对齐条件化子集迁移，第二模型复现完成，升格架构级候选。
4. HDMCC 更新：**CS PR/top3 = 单模型免干预可迁移性诊断量**（主干型↔可迁移，弥散型↔不可迁移）；角度匹配规则的前提=共享低秩因果主干。仲裁实验=第三模型 PR 预检 + 同管线。
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
if 'Phase 3082' not in prev:
    line = ('- Phase 3082 Omega-P79 DS7B '
            'negative-anatomy (no-forward, '
            'sealed 3081/3076/3079/3080 npz '
            'inputs): verdict '
            'ds7b_decorrelated.  Shape: 4B '
            'causal spectra low-rank shared '
            '(CS PR 1.3-2.0, top3 0.957-0.964) '
            'vs DS7B high-rank dispersed (PR '
            '13.6-17.1, top3 0.27-0.32); top8 '
            'overlap 4B AB p=0.012 vs DS7B '
            'chance; sp(A_S) 0.58-0.83 vs '
            '0.03-0.08; DS7B T/U SNR 0.27-0.82 '
            'below noise floor.  NEW: CS '
            'PR/top3 = single-model '
            'transferability diagnostic; '
            'f2~U_AB conservative cross-model '
            '(both significant).  Audit 44; '
            'ledger 221/L14 189.\n')
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
if 'max=3082' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16 15.23GB，16GB 卡省 W32/E2H）、glm4-9b-chat-hf 备用。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout/verify tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（link_id=L14_readout_spectrum_cross_model；**L14.connects 含旧 str 条目，verify 必须 isinstance(c, dict) 防御**）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→独立 verify→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json（脚本自删亦可）；负结果如实登记为一等公民。
4. 统计纪律：阈值预注册；**置换/偏相关 p 的实现必须与主脚本 bit 一致（3081 verify 教训：partial p 主脚本=cnt/n_perm 无 +1 平滑）**；泛化声明分层：架构级/模型级/族级（3081）。

## 标准锚与精度
- bit 锚家族：标量行、因果置换、块链恒等、hook 互证、bit 锚族（3076）、跨源一致性（3077）、前向重建锚（3078）、免前向重放锚（3079）、跨 npz 锚（3080）、跨模型 setup 锚（3081）、**免前向输入冻结+重放锚（3082：输入 npz sha8、top8/SP_AS/n_neg 重放 bit 0）**。
- b8 语义：注入前向内部块链连续性；置换 rng seed=phase 号。

## 机制解释审计链（命名前依次检查）
…→3079 migration_tt_locked→3080 ab_cos_locked（角度匹配，模型内）→3081 ds7b_cos_absent（跨族迁移缺失）→**3082 ds7b_decorrelated：缺失形态=因果谱高秩弥散（DS7B CS PR 13.6-17.1/top3 0.27-0.32 vs 4B PR 1.3-2.0/top3 0.957-0.964，无共享主干）；CS PR/top3=单模型可迁移性诊断；f2~U_AB 跨模型保守（双显著）→仲裁=第三模型 PR 预检**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail 坏→python + 写文件；日志用 Read；-c stdout 丢→写文件再 Read；Glob 对部分目录失效→python os.listdir 为准。
- 关键写入后必须 Grep/Read 复核；改后必编译检查；result.json 无 smoke 键（3081），npz 里 SMOKE 标量为准。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3082）
Ω-P2（3011-3082）：…3080 ab_cos_locked；3081 ds7b_cos_absent；**3082 ds7b_decorrelated（高秩弥散、无主干；PR/top3 诊断量）**。

## 下一步
- max=3082，下一个 3083（A 主选 **第三模型仲裁+PR 预检**：glm4-9b/qwen2.5-3b 跑 3076 管线+CS PR，检验"主干型↔迁移存在"；B PR 层位扫描；C 4B 主干解剖（免前向为主）；D f2~U 升格设计）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars'
             % len(mem_new))
else:
    o.append('memory already max=3082')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
