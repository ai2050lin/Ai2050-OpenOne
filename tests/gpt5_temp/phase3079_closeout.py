# -*- coding: utf-8 -*-
"""Phase 3079 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3079'
     r'\omega_p76_migration_lock')
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
assert verdict == 'migration_tt_locked', \
    verdict
assert res['forwards'] == 0
assert res['smoke'] is False
an = res['anchors']
for k in ('a1', 'a2', 'a3', 'a4', 'a5', 'a6',
          'a7'):
    assert an['oks'][k] is True, k
    assert an['diffs'][k] == 0, (k,
                                 an['diffs'][k])
st = res['stats']
e3 = st['e3']
assert abs(e3['f1_sTT~T_AC']['sp']
           - 0.5060869565217392) < 1e-15
assert abs(e3['f1_sTT~T_AC']['p']
           - 0.01385) < 1e-12
assert abs(e3['f1_sTT~T_BC']['sp']
           - 0.47391304347826085) < 1e-15
assert abs(e3['f1_sTT~U_AC']['sp']
           - 0.5026086956521739) < 1e-15
assert abs(e3['f2_cTT~T_AC']['sp']
           - 0.7713043478260869) < 1e-15
assert abs(e3['f2_cTT~T_AC']['p']
           - 5e-05) < 1e-12
assert abs(e3['f1_sTT~T_AB']['sp']
           - 0.21391304347826087) < 1e-15
assert abs(st['sp_U_T']['AB']
           - 0.6895652173913044) < 1e-15
assert abs(st['sp_U_T']['BC']
           - (-0.09913043478260869)) < 1e-15
g = res['gates']
assert g['G1'] is True
assert abs(g['G1_sp']
           - 0.5060869565217392) < 1e-15
assert g['G1_resp'] == 'T'
assert g['G1_pair'] == 'AC'
assert abs(g['max_abs_sp_all']
           - 0.7713043478260869) < 1e-15

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3079
           for m in led['measurements']):
    claim = (
        'Omega-P76 (plan 3079 A) - qwen3-4b '
        'migration-lock analysis (0 forwards, '
        '4.1s, frozen 3076-npz re-analysis): '
        'WHAT LOCKS the asymmetric R1 migration '
        'across prompt families (spearman A->B '
        '0.676, A->C 0.130, B->C -0.088)?  '
        'Design: E2 family-level 255-subset '
        'amplitude-spectral-shape similarity '
        '(n=3, orientation only); E3 per-pair '
        '(n=24 per family pair) migration '
        'responses T_fg[k]=sp(CS1H_f[:,k], '
        'CS1H_g[:,k]) (head level, 32 heads) '
        'and U_fg[k]=sp(CS_f[:,k], CS_g[:,k]) '
        '(subset level, 255 subsets) vs five '
        'preregistered predictors: f1 sTT = '
        'sp(TT_f[k], TT_g[k]) rank-shape '
        '(MAIN, only predictor gated), f2 '
        'cTT = cos, f3/f4 logits shape at '
        'pref/base prompts, f5 TT norm ratio '
        '(control); 30 tests, perm 20000, '
        'seed 3079.  VERDICT migration_tt_'
        'locked.  (1) FAMILY-LEVEL ANTI-'
        'ORDERING: sp(sp(A_S_f,A_S_g), '
        'migration) = -1.0 - BC is the MOST '
        'amplitude-similar family pair (0.832) '
        'yet migrates WORST (-0.088); static '
        'amplitude spectra do not carry '
        'migration.  (2) G1 PASSES: per-pair '
        'TT rank-shape similarity positively '
        'locks migration - f1_sTT~T_AC '
        'sp=+0.5061 p=0.0138, T_BC +0.4739 '
        'p=0.0203, U_AB +0.4878 p=0.0153, '
        'U_AC +0.5026 p=0.0135 - BUT f1 '
        'FAILS on T_AB (+0.2139 p=0.32): the '
        'AB pair, which migrates STRONGEST '
        '(0.676), is NOT explained by rank '
        'shape.  (3) EXPLORATORY f2 cTT is '
        'stronger everywhere: T_AC +0.7713 '
        '(p=1e-4), U_AC +0.7270, U_BC '
        '+0.6487, T_AB +0.5452 - continuous '
        'direction geometry beats rank '
        'shape (upgrade candidate, not '
        'gated).  (4) OUTPUT SIMILARITY '
        'NULL: logits shape similarity '
        '(f3/f4) does not predict migration '
        'anywhere (max +0.559 on U_BC; '
        'elsewhere ns) - behavioral/output '
        'similarity does NOT imply '
        'mechanistic similarity.  (5) E4: '
        'head-level vs subset-level '
        'migration is coupled on AB (sp '
        'U,T = 0.690) and AC (0.487) but '
        'DECOUPLED on BC (-0.099).  '
        'ANCHORS: 7 bit anchors all 0.0 '
        '(a1 A_S_A vs 3074; a2 R1 vs 3071 '
        'r34; a3 SP_R1 replay - required '
        'the bit-exact 3076 corrcoef-path '
        'spearman76 because the manual '
        '3077-series spearman differs by '
        '1.4e-17 = 1 ulp, a float-path '
        'lesson now on record; a4 MASKS '
        'identical; a5/a6/a7 median/top8 '
        'replays).  Model: the causal head '
        'configuration migrates across '
        'prompt families along the per-pair '
        'TT DIRECTION geometry (the '
        'pref-minus-base readout direction '
        'spectrum), not along static '
        'amplitude spectra and not along '
        'output distributions.')
    meas = {
        'meas_id': 'meas3079_omega_p76_'
                   'migration_lock',
        'phase': 3079,
        'claim': claim,
        'verdict': verdict,
        'anchors': '7 bit anchors all 0.0: '
                   'a1 A_S_A vs 3074 npz; a2 '
                   'R1_ALL32_A vs 3071 r34; '
                   'a3 SP_R1_AB/AC/BC replay '
                   'vs 3076 npz (bit-exact '
                   'corrcoef-path spearman76; '
                   'manual spearman agrees to '
                   '1.4e-17 = 1 ulp only); a4 '
                   'MASKS identical across '
                   'families; a5 R1 = '
                   'median(CS1H)-med_c; a6 '
                   'median(DAH34) vs '
                   'DAH34_MED; a7 top8 = '
                   'argsort(R1)[:8]',
        'artifacts': {
            'result': 'phase3079/omega_p76_'
                      'migration_lock/'
                      'result.json',
            'npz': 'phase3079/omega_p76_'
                   'migration_lock/'
                   'omega_p76_migration_'
                   'lock.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (0 forwards, '
                '4.1s).  Frozen 3076-npz '
                're-analysis; no injections.  '
                'Caveats: n=24 per family pair '
                '(moderate spearman power); 30 '
                'tests without FDR correction '
                '(G1 preregistered on f1 only, '
                '0.5/0.05); G1 achieved via '
                'max-over-pairs (T_AC just '
                'crosses 0.5) and AB remains '
                'unexplained by f1; f2 cTT is '
                'exploratory (upgrade candidate '
                'for 3080); single model, '
                'causal-connective paradigm with '
                'shared syntactic frame; T/U '
                'migration defined on the focal '
                'top-8 subset frame.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 218
    l14['connects'].append({
        'meas_id': 'meas3079_omega_p76_'
                   'migration_lock',
        'phase': 3079,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P76: migration-'
                        'lock analysis (0 '
                        'forwards, frozen 3076 '
                        'npz).  G1 PASSED: '
                        'per-pair TT rank-shape '
                        'similarity positively '
                        'locks R1 migration '
                        '(f1~T_AC +0.5061 '
                        'p=0.0138; T_BC/U_AB/'
                        'U_AC ~+0.50 all '
                        'significant) - BUT T_AB '
                        'fails (+0.214 ns) while '
                        'AB migrates strongest '
                        '(0.676).  Family-level '
                        'amplitude-spectral '
                        'similarity ANTI-orders '
                        'with migration '
                        '(sp=-1.0: BC most '
                        'similar 0.832, migrates '
                        'worst -0.088).  Logits '
                        'shape similarity null - '
                        'output similarity does '
                        'not imply mechanism '
                        'similarity.  Exploratory '
                        'cos(TT) stronger '
                        'everywhere (T_AC +0.771 '
                        'p=1e-4) - direction '
                        'geometry beats rank '
                        'shape.  Migration '
                        'carrier = per-pair TT '
                        'DIRECTION geometry.  7 '
                        'bit anchors 0.0; a3 '
                        'float-path lesson: '
                        'cross-phase bit replay '
                        'must match the upstream '
                        'spearman implementation '
                        '(corrcoef vs manual, 1 '
                        'ulp).  Opens 3080: A '
                        'AB-anomaly anatomy + f2 '
                        'preregistered upgrade '
                        '(no forwards); B h14 '
                        'anatomy; C DS7B cross-'
                        'model replication; D '
                        'injected-state higher-'
                        'order spectrum'})
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
if '## Phase 3079:' not in memo:
    sec = u'''## Phase 3079: Ω-P76 谱形相似性锁定迁移强度——TT 方向几何是路由迁移的载体（migration_tt_locked） [%(created)s]

**判决：`migration_tt_locked`**（**免前向**：3076 npz 冻结数据再分析，4.1 秒。G1 预注册门通过：主检验因子 f1 逐对 TT 秩形相似对迁移的最大 spearman = **+0.5061（p=0.0138，T_AC）**，正号 ⇒ 锁定。锚 a1–a7 全 bit 0.0：a1 A_S_A vs 3074、a2 R1 vs 3071 r34、a3 SP_R1 重放（**必须用 3076 corrcoef 路径版 spearman76 才 bit 0.0**；3077 系手写版差 1.4e-17=1 ulp——浮点路径教训入册）、a4 MASKS 三族相同、a5/a6/a7 median/top8 重放）。

### 问题与设计（3079 A，3078 菜单主选）
**问题**：R1 因果谱跨族迁移不对称（A→B 0.676、A→C 0.130、B→C −0.088）——**什么锁定了迁移强度**？候选：族间谱形相似性。**设计**：E2 族级 255 子集幅度谱形相似（n=3，仅取向）；E3 逐对（每族对 n=24）迁移响应 T_fg[k]=sp(CS1H_f[:,k], CS1H_g[:,k])（头级 32 头）与 U_fg[k]=sp(CS_f[:,k], CS_g[:,k])（子集级 255 子集），对 5 个预注册因子：f1 sTT=sp(TT_f[k],TT_g[k]) 秩形（**主检验，唯一进门因子**）、f2 cTT=cos(TT_f[k],TT_g[k])、f3/f4 logits 形状（pref/base prompt）、f5 TT 范数比（对照）；30 检验 = 2 响应 × 5 因子 × 3 族对，置换 20000 次（seed 3079）。判决门 G1：max |sp(f1, resp)| ≥ 0.5 且 p<0.05，符号定 locked/anti。

### 核心结果（重复三遍）
**① 族级反序（一）**：sp(sp(A_S_f,A_S_g), 迁移) = **−1.0**——BC 幅度谱最像（0.832）却迁移最差（−0.088）；静态幅度谱形不携带迁移。**② 逐对 TT 方向秩形锁定迁移（二）**：f1 显著正——T_AC +0.5061（p=0.0138）、T_BC +0.4739（p=0.0203）、U_AB +0.4878（p=0.0153）、U_AC +0.5026（p=0.0135）；**但 T_AB 失效（+0.2139，p=0.32）——迁移最强的 AB 反而不能用秩形解释（未解反例）**。**③ 输出相似 ≠ 机制相似（三）**：logits 形状相似（f3/f4）几乎全不显著（最大 f3~U_BC +0.559，其余 ns）——**表面输出分布相似不能预测内部路由结构迁移**。探索性 f2 cos(TT) 全面强于 f1：T_AC +0.7713（p=1e-4）、U_AC +0.7270、U_BC +0.6487、T_AB +0.5452（p=0.0055）——连续方向几何强于秩形（升格候选，未进门）。E4：头级与子集级迁移在 AB（sp(U,T)=0.690）、AC（0.487）耦合，**BC 解耦（−0.099）**。

### 数学公式
- 迁移响应：T_fg[k] = sp(CS1H_f[:,k], CS1H_g[:,k])（头级）、U_fg[k] = sp(CS_f[:,k], CS_g[:,k])（子集级）；
- 预测因子：f1[k] = sp(TT_f[k], TT_g[k])、f2[k] = cos(TT_f[k], TT_g[k])（TT = pref−base logits 差方向，每族对 24 条）；
- G1 = [ max_{resp∈{T,U}, pair} |sp(f1, resp)| ≥ 0.5 ] ∧ [ p < 0.05 ]；实测 = 0.5061 / 0.01385（T_AC，正号）。

### 硬伤与边界
- **G1 由 max-over-pairs 达成**：T_AC 的 +0.506 刚过 0.5 门，且 AB 反例（迁移 0.676 最强、f1 不显著）未被解释——"锁定"是部分的，AB 迁移必有别的来源（f2 cos 0.545、f5 amp 0.449 在 AB 上显著，是候选）。
- **f2 cos 全面强于 f1 但属探索性**（未预注册进门）——升格检验留给 3080；不能事后把 f2 当结论。
- n=24 spearman 功效中等；30 检验未做 FDR（G1 仅 f1 预注册守门，f3~U_BC +0.559 显著可能是噪声）。
- 单模型（qwen3-4b）、因果连接词范式、共享句法框架；T/U 迁移定义依赖 top-8 焦点子集框架；族级 n=3 仅取向。

### 方法论入册
- **a3 浮点路径教训**：跨 phase bit 重放必须匹配上游实现的浮点求和路径——3076 用 np.corrcoef 算 spearman，3077 系手写公式数学等价但差 1 ulp（1.4e-17）；修复 = 新增 spearman76（逐位复刻 3076）专用于 a3，主统计保持 3077 系（与 perm_p 一致）。
- 主脚本 E4 键覆盖 bug（dict 每循环覆盖 + npz/result 段旧键引用）在自查 + SMOKE 前静态复核中修复；本次两次 Edit 出现"报成功未落盘"，Grep 复核兜底。

### 智能理论洞察（第一性原理）
**迁移的载体是"方向几何"，不是"幅度"、更不是"行为"。** 三层证据收窄同一问题——什么在语境间迁移：①静态幅度谱（255 子集响应大小）不迁移（族级反序 −1.0）；②输出分布（logits 形状）相似不伴随机制迁移（f3/f4 全败——**行为相似≠机制相似的内部直接证据**）；③逐对 TT 方向（pref−base 读出差方向的谱形）相似正锁迁移（f1 通过预注册门，f2 cos 更强）。结合 3078（路由信号不在基座表示、装配是注入时刻事件驱动的），3079 给出**条件化齿轮组的第一条装配选择规则：事件驱动的装配，其跨语境泛化由 TT 方向几何相似性引导**——两个语境若对同一对 prompt 产生形状相似的读出方向，头配置的因果结构就跟着走。这把"齿轮组如何泛化"从表示内容问题（3078 已排除）转为**方向几何匹配问题**：泛化单位是 24 维方向谱形，不是 255 维幅度谱，也不是 151936 维输出分布。对 AGI 理论：机制级泛化的正确抽象层级是"方向谱形"（低维、关系性、跨语境可对齐），行为与幅度都是它的影子。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3079/omega_p76_migration_lock/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；0 forwards / 4.1s。

**接续 3080 菜单**——A（主选）**AB 反例解剖 + f2 升格预注册**：AB 迁移最强（0.676）但 f1 不显著（+0.214）——解剖 AB 迁移来源（候选：f2 cos +0.545*、f5 amp +0.449*、A_S 相似、基座谱共享 3078 cross 0.945），并把 f2_cTT 升格为预注册主检验在冻结数据上权威复验（免前向）。B **h14 全域解剖**（3078 菜单遗留）：唯一跨族核心头的逐族谱形/OV top tokens/DOH/DHH（免前向为主）。C **DS7B 跨模型对照**：migration_tt_locked 复现检验（需重跑 3076 类管线）。D **注入态高阶交互谱**：路由信号最后藏身处的直接检验（3073 假说，需前向）。"好的，继续"即进 3080 A。
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
if '## 四十一、3079' not in aud:
    add = u'''
---
## 四十一、3079 增补：谱形相似性锁定迁移强度（Omega-P76，判决 migration_tt_locked）
1. **G1 通过（f1 预注册门）**：逐对 TT 秩形相似正锁迁移——f1~T_AC +0.5061（p=0.0138）、T_BC/U_AB/U_AC 均 ~+0.50 显著；但 T_AB 失效（+0.214 ns）而 AB 迁移最强（0.676）——部分锁定，AB 反例未解。
2. **族级反序**：sp(255 幅度谱相似, 迁移) = −1.0——BC 幅度最像（0.832）迁移最差（−0.088）；静态幅度谱不携带迁移。
3. **输出相似 ≠ 机制相似**：logits 形状相似（f3/f4）不预测迁移（几乎全 ns）；探索性 cos(TT) 全面强于秩形（T_AC +0.7713, p=1e-4）——方向几何是迁移载体，升格候选。
4. **bit 锚 7/7**；a3 浮点路径教训：跨 phase 重放必须匹配上游 spearman 实现（corrcoef vs 手写差 1 ulp）→ spearman76。HDMCC 更新：装配选择规则第一条=TT 方向几何匹配；泛化单位=24 维方向谱形。
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
if 'Phase 3079' not in prev:
    line = ('- Phase 3079 Omega-P76 migration-lock '
            '(0 forwards, 4.1s, frozen 3076 npz): '
            'verdict migration_tt_locked.  G1 '
            'passed: per-pair TT rank-shape '
            'similarity positively locks R1 '
            'migration (f1~T_AC +0.5061 p=0.0138; '
            'T_BC/U_AB/U_AC ~+0.50 sig); T_AB '
            'fails (+0.214 ns) while AB migrates '
            'strongest - partial lock, AB anomaly '
            'open.  Family-level amplitude-spectral '
            'similarity ANTI-orders with migration '
            '(sp=-1.0); logits shape similarity '
            'null (output similarity != mechanism '
            'similarity); exploratory cos(TT) '
            'stronger everywhere (T_AC +0.7713 '
            'p=1e-4).  7 bit anchors 0.0; a3 '
            'float-path lesson -> spearman76.  '
            'Audit 41; ledger 218/L14 186.\n')
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
if 'max=3079' not in mem_cur:
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
4. 统计纪律：阈值预注册；**跨族竞标来源族特征不得自评（3077 规则）**；**溯源锚必须覆盖所有进入统计的样本（3078 规则）**；**跨 phase bit 重放必须匹配上游实现的浮点路径（3079 规则：corrcoef vs 手写差 1 ulp→spearman76）**。

## 标准锚与精度
- bit 锚家族：标量行、因果置换、跨 phase 参考数、块链恒等、hook 互证、跨 phase 因果复现、枚举重放（3075）、bit 锚族（3076）、跨源一致性（3077）、前向重建锚（3078：a1=冻结 ZH34−新基座 zH；a0 溯源 LG64−LG32=0.0）、**免前向重放锚组（3079：a1-a7 全 bit，A_S/R1/SP_R1/MASKS/median/top8）**。
- 上游 npz 只有 float32 降采样时，bit 锚必须重建 float64 原量（3078 TT64/LG64）。
- 跨路径 bit 锚需匹配浮点求和顺序；置换检验 rng 冻结（seed=phase 号）。

## 机制解释审计链（命名前依次检查）
…→3076 cross_prompt_unstable→3077 observation_causation_decoupled→3078 routing_signal_absent（基座 L24-L35 全层线性读出 max|sp|=0.221；路由=注入时刻事件驱动装配）→**3079 migration_tt_locked：逐对 TT 方向秩形相似正锁迁移（f1~T_AC +0.506 p=0.014；G1 过）；族级幅度谱反序（−1.0）；logits 形状全败（行为相似≠机制相似）；cos(TT) 更强（+0.771，升格候选）；AB 反例未解（迁移 0.676 最强但 f1 ns）→装配选择规则第一条=TT 方向几何匹配；泛化单位=24 维方向谱形**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep 坏→删除目录用 python shutil.rmtree；日志用 Read；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核（Edit 报成功未落盘会发生）；改后必编译检查。
- NaN 数据：argsort 污染 spearman，须有效掩码过滤。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3079）
Ω-P2（3011-3079）：…3075 supermodular_diffuse；3076 cross_prompt_unstable；3077 observation_causation_decoupled；3078 routing_signal_absent；**3079 migration_tt_locked（迁移载体=TT 方向几何）**。

## 下一步
- max=3079，下一个 3080（A 主选 **AB 反例解剖 + f2 升格预注册**：AB 迁移最强但 f1 ns，解剖来源并把 f2_cTT 升格主检验权威复验，免前向；B h14 全域解剖；C DS7B 跨模型对照；D 注入态高阶交互谱）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3079')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
