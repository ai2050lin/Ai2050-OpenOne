# -*- coding: utf-8 -*-
"""Phase 3080 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3080'
     r'\omega_p77_ab_anatomy')
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
assert verdict == 'ab_cos_locked', \
    verdict
assert res['forwards'] == 0
assert res['smoke'] is False
an = res['anchors']
for k in ('A1', 'A2', 'A3', 'A4', 'A5'):
    assert an['oks'][k] is True, k
    assert an['diffs'][k] == 0, (k,
                                 an['diffs'][k])
st = res['stats']
assert abs(st['f2_tests']['T_AB']['sp']
           - 0.5452173913043479) < 1e-15
assert abs(st['f2_tests']['T_AB']['p']
           - 0.0073) < 1e-12
assert abs(st['f2_tests']['T_AC']['sp']
           - 0.7713043478260869) < 1e-15
assert abs(st['f2_tests']['U_AC']['sp']
           - 0.7269565217391304) < 1e-15
assert abs(st['f2_tests']['U_BC']['sp']
           - 0.648695652173913) < 1e-15
assert abs(st['partial']['f2g1_T_AB']
           - 0.556046582156676) < 1e-15
assert abs(st['sp_f1f2']['AB']
           - 0.6747826086956522) < 1e-15
g = res['gates']
assert g['G2'] is True
assert g['count_sig_pos'] == 5
assert abs(g['min_sp']
           - 0.3591304347826087) < 1e-15
assert abs(g['stouffer_z']
           - 8.091847859848315) < 1e-12
assert abs(g['f2_TAB_sp']
           - 0.5452173913043479) < 1e-15
assert g['f2_TAB_p'] < 0.05

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3080
           for m in led['measurements']):
    claim = (
        'Omega-P77 (plan 3080 A) - qwen3-4b AB-'
        'anomaly anatomy + preregistered '
        'confirmatory re-check of f2 cos(TT) '
        '(0 forwards, 4.7s, frozen 3076/3078/'
        '3079 npz).  Question: AB migrates '
        'strongest (0.676) but f1 rank-shape '
        'fails (+0.214 ns) - what locks AB?  '
        'HONESTY: f2 was observed in 3079 on '
        'the SAME frozen data; E2 is a '
        'preregistered gate fixed before this '
        'run - a confirmatory RE-CHECK, not '
        'independent confirmation.  VERDICT '
        'ab_cos_locked.  (1) G2 PASSES: 5/6 '
        'f2-response tests significant positive '
        '(f2~T_AB +0.5452 p=0.0073 Bonf, T_AC '
        '+0.7713 p=1e-5 Bonf, U_AB +0.4991 '
        'p=0.013, U_AC +0.7270 p=2e-4 Bonf, '
        'U_BC +0.6487 p=7e-4 Bonf; T_BC +0.359 '
        'p=0.086 ns), min sp 0.359 > 0, '
        'Stouffer one-sided z = 8.09 - the '
        'direction-angle cos locks migration '
        'INCLUDING the anomalous AB pair.  '
        '(2) AB ANOMALY RESOLVED AS PROXY '
        'COARSENING: sp(f1, f2) coupling per '
        'pair = AB 0.675 < AC 0.770 < BC 0.790 '
        '- rank shape is a coarsened proxy of '
        'the angle; AB is where the proxy '
        'degrades MOST, so f1 fails exactly '
        'there while the true carrier '
        'continues to lock.  (3) PARTIAL '
        'SPEARMAN (exploratory): sp(f2 | f1) = '
        '+0.556 (p=0.0066) on T_AB, +0.694 '
        '(p=0.0003) on T_AC, +0.616 (p=0.0016) '
        'on U_AC - the angle carries '
        'significant signal BEYOND rank '
        'shape; sp(f5 | f2) ~ 0 (U_AC -0.496 '
        'marginal) - amplitude adds nothing '
        'once direction is controlled.  (4) '
        'PER-PAIR STRUCTURE: ci=3-prefix pairs '
        '(k8-15) dominate AB migration (T up '
        'to 0.966) with high f2 (0.68-0.93); '
        'the single worst pair k16 (bi=0, ci=3) '
        'is the ONLY negative-cos pair (f2=-'
        '0.360) and has the most negative '
        'migration (T=-0.705).  (5) FAMILY '
        'LEVEL (n=3 orientation): base '
        'readout-spectrum similarity (3078 '
        'DM_L34 cross 0.945/0.867/0.878) vs '
        'migration sp=+0.5 weak positive; '
        'static amplitude spectra ANTI-order '
        '(-1.0, 3079) - of three family-level '
        'similarity measures only the per-pair '
        'direction geometry carries the '
        'migration.  ANCHORS: 5 bit anchors '
        'all 0.0 (a1 T/U/F1-F5 recompute vs '
        '3079 npz 25 arrays; a2 SP_R1 '
        'spearman76; a3 R1 vs 3071 r34; a4 '
        'DM_L34 cross vs 3078 result; a5 A_S '
        'vs 3074).  Model: the assembly '
        'generalization rule is ANGLE '
        'MATCHING cos(TT_f[k], TT_g[k]), not '
        'rank matching and not amplitude - '
        'rank shape is a lossy proxy of the '
        'angle.')
    meas = {
        'meas_id': 'meas3080_omega_p77_'
                   'ab_anatomy',
        'phase': 3080,
        'claim': claim,
        'verdict': verdict,
        'anchors': '5 bit anchors all 0.0: a1 '
                   'T/U/F1-F5 recompute vs '
                   '3079 npz (25 arrays); a2 '
                   'SP_R1 replay spearman76 vs '
                   '3076; a3 R1_ALL32_A vs '
                   '3071 r34; a4 DM_L34 '
                   'cross-family spearman vs '
                   '3078 result.json; a5 A_S_A '
                   'vs 3074 npz',
        'artifacts': {
            'result': 'phase3080/omega_p77_'
                      'ab_anatomy/'
                      'result.json',
            'npz': 'phase3080/omega_p77_'
                   'ab_anatomy/'
                   'omega_p77_ab_anatomy.'
                   'npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (0 forwards, '
                '4.7s).  Confirmatory RE-CHECK on '
                'the SAME frozen data as 3079 - '
                'NOT independent; honest '
                'confirmation requires new data.  '
                'Caveats: n=24 per pair; partial '
                'spearman first-order on n=24 is '
                'orientation only; T_BC remains '
                'ns (5/6 not 6/6); f1-f2 coupling '
                '0.675-0.790 means part of the '
                'f2 signal overlaps rank shape; '
                'single model, shared syntactic '
                'frame; migration defined on the '
                'focal top-8 subset frame.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 219
    l14['connects'].append({
        'meas_id': 'meas3080_omega_p77_'
                   'ab_anatomy',
        'phase': 3080,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P77: AB-anomaly '
                        'anatomy + f2 confirmatory '
                        're-check (0 forwards, '
                        'same frozen data - '
                        'declared non-independent). '
                        'G2 passed 5/6 (Stouffer '
                        'z=8.09): direction-angle '
                        'cos locks migration '
                        'including anomalous AB '
                        '(f2~T_AB +0.545 p=0.007). '
                        'AB anomaly resolved as '
                        'PROXY COARSENING: f1-f2 '
                        'coupling AB 0.675 lowest - '
                        'rank shape degrades most '
                        'exactly where the angle '
                        'still locks.  Partial '
                        'sp(f2|f1)=+0.556 '
                        '(p=0.0066) T_AB; amplitude '
                        'adds nothing (f5|f2 ~ 0). '
                        'Worst pair k16 is the only '
                        'negative-cos pair and has '
                        'most negative migration.  '
                        'Family level: base-spectrum '
                        'similarity +0.5 weak, '
                        'amplitude ANTI-orders -1.0 '
                        '- only per-pair direction '
                        'geometry carries '
                        'migration.  Assembly rule '
                        'sharpened: ANGLE MATCHING, '
                        'not rank matching.  5 bit '
                        'anchors 0.0.  Opens 3081: '
                        'A DS7B cross-model '
                        'replication (the honest '
                        'independent test); B '
                        'new-family D control; C '
                        'h14 anatomy; D injected-'
                        'state higher-order '
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
if '## Phase 3080:' not in memo:
    sec = u'''## Phase 3080: Ω-P77 AB 反例解剖——方向夹角锁定迁移，秩形只是粗化代理（ab_cos_locked） [%(created)s]

**判决：`ab_cos_locked`**（**免前向**：3076/3078/3079 冻结 npz 再分析，4.7 秒。G2 预注册门通过：**f2 cos(TT) 对迁移 5/6 检验显著正**（Bonferroni 0.05/6 下 4/6），min sp=0.359>0，Stouffer 单侧 z=**8.09**；关键的是 **f2~T_AB = +0.5452（p=0.0073）**——3079 上 f1 失效的 AB 反例，被方向夹角完整锁定。锚 A1–A5 全 bit 0.0：a1 T/U/F1–F5 重算 vs 3079 npz（25 数组）、a2 SP_R1（spearman76）、a3 R1 vs 3071 r34、a4 DM_L34 cross 重放 vs 3078 result、a5 A_S vs 3074）。**诚实声明（预注册）**：f2 数值在 3079 同一批冻结数据上已被观察过，本轮是**门先行固定的确认性重验**，不是独立确认——独立确认需要新数据（DS7B 或新族）。

### 问题与设计（3080 A，3079 菜单主选）
**问题**：AB 迁移最强（0.676）但 f1 秩形失效（+0.214 ns）——AB 迁移由什么锁定？f2 cos 是否为统一载体？**设计**：E2 f2 确认性复验（6 检验 = 2 响应 × 3 族对，门 G2：count(sp>0 ∧ p<0.05)≥4 ∧ min(sp)>0；seed 3080、20000 置换；Bonferroni 参考 + Stouffer 合并）；E3 AB 解剖（逐对表、f1-f2 耦合、一阶偏 spearman、族级三种相似度量对照）。

### 核心结果（重复三遍）
**① 方向夹角锁定迁移，包括反例 AB（一）**：f2 六检验 T_AB +0.5452（p=0.0073）、T_AC +0.7713（p=1e-5）、U_AB +0.4991、U_AC +0.7270（p=2e-4）、U_BC +0.6487（p=7e-4）显著正，T_BC +0.359（p=0.086）不显著——5/6。**② AB 反例的机制=代理粗化（二）**：f1-f2 耦合 **AB 0.675 < AC 0.770 < BC 0.790**——秩形是夹角的粗化代理，**AB 正是代理退化最严重处**，所以 f1 恰好在 AB 失效而真载体（cos）继续锁定；偏 spearman：sp(f2|f1) T_AB **+0.556**（p=0.0066）、T_AC +0.694（p=0.0003）、U_AC +0.616（p=0.0016）——控制秩形后夹角仍有显著增量；sp(f5|f2)≈0（U_AC −0.496 边际）——**控制方向后幅度无独立贡献**。**③ 逐对结构一致（三）**：AB 的 ci=3-prefix 8 对（k8–15）主导迁移（T 至 0.966）且 f2 高（0.68–0.93）；唯一负 cos 的对 k16（bi=0, ci=3，f2=−0.360）恰是迁移最负的对（T=−0.705）——单对级别的 cos↔迁移一致。族级（n=3 取向）：基座读出谱相似（3078 DM_L34 cross 0.945/0.867/0.878）对迁移 sp=+0.5 弱正、静态幅度谱反序 −1.0——三种族间相似度量中**只有逐对方向几何携带迁移**。

### 数学公式
- 载体（升级确认）：f2[k] = cos(TT_f[k], TT_g[k])，TT = pref−base logits 差方向（每族对 24 条）；
- 门：G2 = [ #{(resp,pair): f2 sp>0 ∧ perm p<0.05} ≥ 4 ] ∧ [ min f2 sp > 0 ]；实测 5/6、0.359；
- 偏相关：sp(X,Y|Z) = (ρ_xy − ρ_xz ρ_yz)/√((1−ρ_xz²)(1−ρ_yz²))（秩上 pearson；响应列置换）。

### 硬伤与边界
- **同数据重验**：f2 六个数值 3079 已见——本轮门先行固定但数据非独立；把 cos 锁定当结论前必须过 DS7B/新族独立检验。
- f1-f2 耦合 0.675–0.790：f2 的"增量"与秩形部分重叠（同一信号的不同投影）；偏相关一阶、n=24，仅取向。
- T_BC 仍 ns（5/6 非 6/6）；BC 的 U 上 f3 logits 形状 +0.559（3079）仍是孤例噪声。
- 单模型 qwen3-4b、因果连接词范式、共享句法框架；迁移定义依赖 top-8 焦点框架；族级 n=3 仅取向。

### 方法论入册
- **确认性重验纪律**：探索性发现升格时，门必须在重验前固定并声明"非独立"；真正的独立检验只能来自新数据/新模型。
- 锚体系新增**跨 npz 数据一致锚**（a1：25 个数组对 3079 npz bit 重放）——冻结链上每个新 phase 自动继承上游全部统计量的 bit 一致性。

### 智能理论洞察（第一性原理）
**AB 反例解决：装配选择规则的精确形式是"方向夹角匹配"（cos），不是"秩形匹配"（sp），更不是幅度。** 秩形是夹角的粗化代理（耦合 0.675–0.790），粗化最严重的 AB 上代理失效而真信号继续锁定——这解释了 3079 的全部遗留。重复三遍：**迁移由逐对 TT 方向夹角锁定；秩形与幅度都不是载体；f1 的 AB 失效=代理粗化**。几何层级现在完整：151936 维输出分布 →（pref−base 差分）→ 24 维方向谱形 →（丢秩保角）→ 标量夹角——**泛化规律在最高压缩层级最干净**。结合 3078（装配=注入时刻事件驱动）：条件化齿轮组的泛化不是"表示相似"，而是"事件产生的方向场在另一语境中被相同角度对齐"——齿轮啮合的判据是**角度**，不是内容、不是排序、不是大小。对 AGI 理论：机制级泛化的正确抽象是**连续方向几何中的角度匹配**，这给出可计算的装配预测器 f2——下一个模型/新语境上的可证伪预测：cos 锁定应复现，秩形锁定应在其代理退化处系统性失效。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3080/omega_p77_ab_anatomy/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d；0 forwards / 4.7s。

**接续 3081 菜单**——A（主选）**DS7B 跨模型对照**：对 DS7B 重跑 3076 类管线（三族 × 8 body × 4 prefix 前向 + L34 注入 + top8/255 子集扫描 + TT/LG），检验 migration_tt_locked 与 ab_cos_locked 在独立模型上复现——cos 锁定的唯一诚实独立检验（重，需前向，注意逐一测试防 OOM）。B **新族 D 对照**：qwen3-4b 上构造第 4 族（新语义域 body），f2 锁定在新数据上复验（比 DS7B 轻，仍需前向）。C **h14 全域解剖**（遗留，免前向为主）。D **注入态高阶交互谱**（3073 假说，需前向）。"好的，继续"即进 3081 A。
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
if '## 四十二、3080' not in aud:
    add = u'''
---
## 四十二、3080 增补：AB 反例解剖——方向夹角锁定迁移（Omega-P77，判决 ab_cos_locked）
1. **G2 通过（5/6，Stouffer z=8.09）**：f2 cos(TT) 对迁移 5/6 检验显著正（含 f2~T_AB +0.5452 p=0.0073）——方向夹角锁定迁移**包括 3079 的 AB 反例**；同数据确认性重验（预注册声明非独立）。
2. **AB 反例机制=代理粗化**：f1-f2 耦合 AB 0.675 < AC 0.770 < BC 0.790——秩形是夹角的粗化代理，AB 正是代理退化最严重处；偏 spearman sp(f2|f1)=+0.556（p=0.0066）——控制秩形后夹角仍有增量；幅度无独立贡献（f5|f2≈0）。
3. **逐对一致**：唯一负 cos 对（k16，f2=−0.360）恰是迁移最负对（T=−0.705）；族级基座谱相似弱正（+0.5）、幅度谱反序（−1.0）——只有逐对方向几何携带迁移。
4. HDMCC 更新：装配泛化规则精确化=**角度匹配 cos(TT_f[k],TT_g[k])**，非秩形、非幅度；泛化规律在最高压缩层级（标量夹角）最干净；独立检验=DS7B/新族（3081 主选）。
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
if 'Phase 3080' not in prev:
    line = ('- Phase 3080 Omega-P77 AB-anomaly '
            'anatomy + f2 confirmatory re-check '
            '(0 forwards, 4.7s, same frozen data '
            '- declared non-independent): verdict '
            'ab_cos_locked.  G2 passed 5/6 '
            '(Stouffer z=8.09): direction-angle '
            'cos locks migration including AB '
            '(f2~T_AB +0.545 p=0.007); AB anomaly '
            'resolved as proxy coarsening (f1-f2 '
            'coupling AB 0.675 lowest; partial '
            'sp(f2|f1)=+0.556 p=0.0066; amplitude '
            'adds nothing).  Only negative-cos '
            'pair k16 has most negative '
            'migration.  Assembly rule sharpened: '
            'ANGLE MATCHING, not rank matching.  '
            '5 bit anchors 0.0 (incl. 25-array '
            'cross-npz replay).  Audit 42; '
            'ledger 219/L14 187.\n')
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
if 'max=3080' not in mem_cur:
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
4. 统计纪律：阈值预注册；**跨族竞标来源族特征不得自评（3077）**；**溯源锚必须覆盖所有进入统计的样本（3078）**；**跨 phase bit 重放必须匹配上游浮点路径（3079：corrcoef vs 手写 1 ulp→spearman76）**；**探索性发现升格=门先行固定的确认性重验并声明非独立（3080）**。

## 标准锚与精度
- bit 锚家族：标量行、因果置换、跨 phase 参考数、块链恒等、hook 互证、枚举重放（3075）、bit 锚族（3076）、跨源一致性（3077）、前向重建锚（3078）、免前向重放锚组（3079 a1-a7）、**跨 npz 数据一致锚（3080 a1：25 数组对 3079 npz bit 重放；a4 DM_L34 vs 3078 result）**。
- 上游 npz 只有 float32 降采样时，bit 锚必须重建 float64 原量；置换 rng seed=phase 号。

## 机制解释审计链（命名前依次检查）
…→3076 cross_prompt_unstable→3077 observation_causation_decoupled→3078 routing_signal_absent（路由=注入时刻事件驱动装配）→3079 migration_tt_locked（逐对 TT 秩形正锁迁移；族级幅度反序；logits 形状全败）→**3080 ab_cos_locked：方向夹角 cos 锁定迁移含反例 AB（5/6，Stouffer z=8.09）；AB 反例=代理粗化（f1-f2 耦合 AB 0.675 最低；偏 sp(f2|f1)=+0.556）；幅度无独立贡献→装配泛化规则=角度匹配 cos(TT_f[k],TT_g[k])，泛化规律在最高压缩层级（标量夹角）最干净；独立检验=DS7B/新族**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep 坏→删除目录用 python shutil.rmtree；日志用 Read；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核（Edit 报成功未落盘会发生）；改后必编译检查；跨行属性访问用括号。
- NaN 数据：argsort 污染 spearman，须有效掩码过滤。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3080）
Ω-P2（3011-3080）：…3075 supermodular_diffuse；3076 cross_prompt_unstable；3077 observation_causation_decoupled；3078 routing_signal_absent；3079 migration_tt_locked；**3080 ab_cos_locked（装配泛化=角度匹配）**。

## 下一步
- max=3080，下一个 3081（A 主选 **DS7B 跨模型对照**：重跑 3076 类管线检验 migration_tt_locked + ab_cos_locked 复现，需前向防 OOM 逐一测试；B 新族 D 对照；C h14 解剖（免前向）；D 注入态高阶交互谱）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3080')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
