# -*- coding: utf-8 -*-
"""Phase 3101+3102 closeout (idempotent):
Ledger -> MEMO (Phase 3101 + Phase 3102) -> workspace logs
-> MEMORY.md rewrite."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3101'
     r'\omega_p99_upstream_writeup')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = R + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(R + r'\result.json', encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json', encoding='utf-8'))
assert res['verdict'] == 'seventh_carrier_absent', res['verdict']
assert res['verdict_4b'] == 'seventh_profile_only'
g = res['gates']
assert g['H_G1'] is False and g['H_G2'] is False
assert g['overlap_pre'] == {'A': 2, 'B': 3, 'C': 2}
assert abs(g['g2_ratios'][0] - 0.7984496657331223) < 1e-12
assert abs(g['late3_14b'] - 0.32521795358904454) < 1e-12
an = res['anchors']
assert an['d1_max'] == 0.0 and an['d1b_max'] == 0.0
assert an['d1v_max'] == 0.0 and an['d2_max'] == 0.0
assert an['d2b_max'] == 0.0
assert an['d4'] == {'p98_ok': True, 'p93_ok': True}
npz8 = res['npz_sha256_8']
assert npz8 == '15ec0e8c'
result8 = seal['result_sha256_8']
assert result8 == 'e38d1707'
try:
    script8 = seal['script_sha256_8']
except KeyError:
    script8 = 'n/a(seal result-only)'

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3101
           for m in led['measurements']):
    claim = (
        'Omega-P99 (3101) - L37 carrier arbitration '
        'under the CORRECT pre-o_proj per-head '
        'convention. First: 3093 intervention '
        'semantics fully reverse-engineered - it is '
        'a FULL V-STREAM replacement (all 48 layers '
        'v_proj outputs clamped to the base natural '
        'bank; only L37 rows [0,FRONT=4) carry the '
        'prefix V), bit-level replicated (d2b=0 '
        'across A/B/C; smoke+formal). Then three '
        'pre-registered gates: H_G1 natural-carrier '
        'top8 (pre-o_proj nat share) vs 3093 focal '
        'top8 overlap 2/3/2 median 2 < 3 (random '
        'baseline 1.6) FALSE; H_G2 focal identity-'
        'recovery increment vs nonfocal ratio '
        '0.798/0.912/0.801 (need >=2) FALSE; H_G3 '
        'late-3-block nat share 0.297(4B)/0.325(14B) '
        'n=72 (need >=0.5) FALSE. VERDICT '
        'seventh_carrier_absent (4B '
        'seventh_profile_only): the 3093 head-swap '
        'recovery is NOT returning work to natural '
        'carriers - it is a distributed rebalancing '
        'within the L37 attention block; 3100 Q3 '
        'overlap 1/0/0 was a post-o_proj pseudo '
        'per-head artifact, correct value 2/3/2. '
        'NEW: top8_nat_pre is cross-family stable '
        '(heads 21/12/14 top-3 in all of A/B/C) '
        'while 3093 focal top8 is family-specific - '
        'L37 has a condition-stable natural writer '
        'group carrying the shared component, '
        'disjoint from condition-specific causal '
        'focal heads. TOP8_R1 sanity overlap 0/8 '
        '(identity-recovery ranking vs prefix-'
        'swap-in ranking fully disjoint - '
        'registered observation). Anchors all '
        'bit-0: d1/d1b/d1v/d2/d2b/d3a/d5/d6=0, '
        'd3b=4.8e-7, d4 sha ok. Review claims '
        'R44/R48/R51/R52/R53/R54/R55 all VERIFIED '
        '(7/7) in the same session - see Phase '
        '3102. NEXT: 3103 RDC formula evidence-'
        'grade audit (no-forward); 3104+ true '
        'relation-combination dose design.')
    meas = {
        'meas_id': 'meas3101_omega_p99_'
                   'upstream_writeup',
        'phase': 3101,
        'claim': claim,
        'verdict': 'seventh_carrier_absent',
        'anchors': 'd1/d1b/d1v/d2/d2b/d3a/d5/d6 '
                   'bit-0; d3b 4.8e-7; d4 sha ok; '
                   'H_G1 2/3/2; H_G2 0.80/0.91/0.80; '
                   'H_G3 0.297/0.325',
        'artifacts': {
            'result': 'phase3101/omega_p99_'
                      'upstream_writeup/'
                      'result.json',
            'npz': 'phase3101/omega_p99_'
                   'upstream_writeup/'
                   'omega_p99_upstream_writeup.npz'},
        'hashes': {
            'npz_sha256_8': npz8,
            'result_sha256_8': result8},
        'note': 'full-V-stream semantics reverse-'
                'engineered from 3093 source + '
                'sealed run_log + mtime forensics + '
                'variant probe (V2 bit-level 1e-12); '
                'script = phase3101_omega_p99_'
                'upstream_writeup.py; 54 min 14B '
                'formal (3051 fwd) + 4B profile',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3101_omega_p99_upstream_writeup')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    led_n = len(led['measurements'])
    o.append('ledger already upserted n=%d' % led_n)

# ---------- MEMO append (Phase 3101 + 3102) ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3101:' not in memo:
    sec = u'''## Phase 3101: Ω-P99 L37 载体仲裁——3093 干预语义逆向闭合（全 V 流替换，bit 级复刻 d2b=0）+ pre-o_proj 重测：自然载体与因果焦点解耦（seventh_carrier_absent）[%(now)s]

**判决：`seventh_carrier_absent`（4B 子判决 `seventh_profile_only`）**。setup_ok=True，全锚 bit-0（d1/d1b/d1v/d2/d2b/d3a/d5/d6=0，d3b=4.8e-7，d4 sha 双过）。

### 1. 破案：3093 干预语义的完全逆向（本 Phase 前置成果）
3093 E3 的"repV+head swap"联合干预此前被误读为"仅替换 L37 的 V"。本 Phase 用源码取证（hook 结构）+ sealed run_log（FRONT=4、KVW=1024、E1 med_c=0.5104）+ mtime 链（源码 10:57 < execution 11:07 < sealed 15:10）+ **变体探针三重判别**（V1 仅 L37 替换差 7.7e-2；V2 全层复刻差 **1.0e-12 bit 级**；V3 prefix 取 [0,4) 差 1.56）定案：**repV 是全 V 流替换——所有层 v_proj 输出钳回 base 自然 bank，仅 L37 行 [0,4) 携带 prefix V**；干预恰好局域化于 L37 的读出位。3101 脚本按此语义重构后 **d2b=0.000e+00（vs 3093 sealed COS_LAD bit 级一致），E1 med_c_new=0.5104 与 3093 完全相同**。"仅替换一层是 bit 等价"的旧论证被证伪（下游层 V 会从被改状态重算）。

### 2. 设计：pre-o_proj 正确约定下的载体仲裁
3100 Q3 用 post-o_proj 输出按头 reshape（数学上无意义——o_proj 已混合各头，审查 R55 证实）。3101 改用 **o_proj 输入侧 forward_pre hook（BH_PRE，per-head 纯净块）**，swap 为恒等恢复式（repV 注入后把某头输出换回 base 自然块），r1_nh_new=median(CS1H_new[h])−med_c_new。预注册三门：H_G1 自然载体 top8（pre-o_proj 自然份额）vs 3093 焦点 top8 overlap≥3；H_G2 focal 恒等恢复增量 ≥2× nonfocal；H_G3 末三块自然份额 ≥0.5。14B 正式 3051 前向（54 min）+ 4B profile（144）。

### 3. 核心结果（重复三遍）
**① 自然载体与因果焦点解耦**：overlap_pre = **2/3/2**（median 2 < 3 门；随机基线 1.6）→ H_G1 False。**② 恒等恢复不聚焦于焦点头**：g2_ratios = **0.798/0.912/0.801**（不仅 <2，甚至 <1——focal 头恢复增量不高于非 focal）→ H_G2 False。**③ 自然份额不集中于末三块**：late-3 share = **0.297(4B)/0.325(14B)**（n=72）→ H_G3 False。**判决：3093 的 head swap 恢复不是"把工作归还给自然载体"——它是 L37 attention 块内的分布式再平衡**。3100 Q3 的 1/0/0 是 post-o_proj 伪分解伪象，正确 pre-o_proj 值为 2/3/2（与审查报告引用的 GLM2751 同基修订 2/3/2 独立一致）。
**④ 新结构发现**：top8_nat_pre **跨族高度稳定**——head **21/12/14** 在 A/B/C 三族全部进入前 3（A=[21,12,14,26,11,10,23,30]，B=[21,12,14,11,24,10,30,35]，C=[12,21,14,26,4,30,9,3]），而 3093 焦点 top8 是族特异的。**L37 存在一个跨条件稳定的"自然写入头组"（承载公共分量），与条件特异的"因果焦点头"是两套组织**——这与 3037"词身份分量+公共中继分量"的三分解在 L37 读出位形成呼应。⑤ 登记：TOP8_R1 sanity overlap=0/8——恒等恢复视角与 prefix 换入视角的头排序完全不相交（两个干预问的是不同问题，排序不可互推）。

### 4. 审查核查结论（同会话完成，详见 Phase 3102）
R44/R48/R51/R52/R53/R54/R55 **7/7 全部证实**：R53 的 26.55–41.18% 与 3100 sealed RESID 中位数端点逐位吻合；R44 的 467/1792 由本会话独立重算逐位复现。

### 5. 硬伤与边界
- 判决语义：carrier_absent 否定的是"swap 恢复=自然载体归还"这一特定机制解释，不否定 3093 swap 本身的效果（其因果有效性 bit 级复真实测存在）；
- TOP8_R1 0/8 提示 r1_nh（恒等恢复）与 3093 r1_allnh（prefix 换入）的排序差异未做统计检验，登记为观察；
- 单末位置、causal-connective 单范式、bf16 前向+f64 统计口径同前。

### 6. 结论与接续
RDC 修正：条件齿轮的"复用"须区分**公共分量载体**（跨条件稳定写入头组）与**条件竞争焦点**（族特异、由干预揭露）——两者解耦是 L37 层的实证结构。接续 3103（RDC 公式证据分级审计，免前向）→ 3104+（真关系组合剂量设计，用 3093 全 V 流范式）。

资源消耗：14B 正式 3051 前向约 54 min + 4B 144 前向；产物 sealed（npz8=15ec0e8c result8=e38d1707）。

---

## Phase 3102: Ω-P100 审查核查与理论收紧——GPT 综合审查 7/7 证实，四项大任务系统性方案（review_verification，无实验 Phase）[[NOW]]

**性质**：对用户提交的 GPT 综合审查报告（R01–R57，覆盖 Phase 2750–3100）的可核查论断逐项取证核实，并据核实结果收紧理论、制定系统性方案。核查证据全部落盘 `tests/gpt5_temp/p3101_review_verify1–5.*`。

### 1. 核查结论：7/7 全部证实（无一冤枉）
| 编号 | 审查主张 | 判定 | 决定性证据 |
|---|---|---|---|
| R44 | 3075"次模⟺高阶 Möbius 全非正"等价式错误；条件二阶差分 467/1792 越 0.02 | **完全证实** | 本会话从 sealed A_S+MASKS 独立重算：**467/1792>0.02 逐位一致**（max D=0.2625）；μ₄ 51/70 正系数 ≠ 次模违反数——次模约束的是相关 μ 项之和 |
| R48 | 3082–3092 跨模型强统计撤回（ρ 0.93→0.46） | **证实** | 3091 源码 L70 自述 "spearman ranks ties by argsort position (**no average-rank correction**)" |
| R51 | 3099 H_E2a 门 1.6941>1 数学不可达 | **证实** | 3×0.5647=1.6941（精确）；Jaccard≤1，判据不可通过，False 不能作"方向编码"证据（GLM MEMO 已有同结论记录） |
| R52 | 3099 head(norm(Δm)) 一阶直通桥解读错误 | **证实** | 源码 L623 `bv=head(norm_mod(dmt))`——增量单独过 norm，**基点是 0 而非工作点 h**；与 J_norm(h)·Δm 是不同量；降级为描述性相似度 |
| R53 | 3100 SwiGLU du 项导数错误；"残差 6%"实为幅度误差 26.55–41.18% | **完全证实** | 源码 L788 `dap=sp*(up_*dgp+gp*dup)` 第二项应为 `silu(gp)*dup`；sealed npz RESID 中位数 **0.2655(14B B)–0.4118(4B C)** 与审查区间两端逐位吻合；1−cos 仅是角度残差（等范数 L2=√(2(1−cos))≈0.35 自洽） |
| R54 | 3100 "96.5% 走 h0 残差通道"取补不合法 | **证实** | ASH/HSH 为能量比 e_a/e_t、e_h/e_t，‖A+H‖² 含交叉项 2⟨A,H⟩，ASH+HSH≠1；xpcos 0.997 不支持"RMSNorm 不混合"的一般化 |
| R55 | 3100 Q3 post-o_proj 伪 per-head | **证实** | 源码 L19/L337 自述 'post-o_proj'；o_proj 输出已混合各头，按头 reshape 无意义；3101 正确重测 2/3/2（第七载体判决已入册） |

另：审查对 3100"GLM2751 同基修订 2/3/2"的引用与 3101 独立重测一致，交叉验证成立。

### 2. 理论收紧五原则（即刻生效，写入审计链）
1. **度量双报**：cos 与 relative-L2 并报；"解释率"语言只允许用于声明了度量的量；份额/分解必须声明正交性或互补性条件，能量份额默认含交叉项。
2. **判据可达性预检**：任何门设计先做可达域数学验证（门值必须落在判据统计量的可达区间内——1.6941>1 教训）。
3. **统计方法修正**：秩相关用平均秩（ties correction）；跨模型/跨族合并统计须做块级独立性与伪重复校正；双侧 p→z 换算用标准公式；修统计方法后的"强确认"一律降级重验后才可恢复。
4. **机制解读审计**：JVP/Jacobian 预测准 ≠ 映射不混合；描述性相似度（基点任选的 cos）≠ 机制桥；任何"直通/透明"语言必须指明基点与工作点。
5. **集合函数判据**：次模性的权威判据是条件二阶差分（全部 1792 条），Möbius 谱只作交互谱描述，不再作为结构性判决依据。

### 3. 对既有记录的更正登记（append-only，原文不改）
- **3100 节**：（i）"残差 6% 为高阶项"→ 更正为"角度残差 6%；幅度相对误差（RESD）26.55–41.18%"；（ii）dap 公式 du 项更正为 silu(gp)·dup；（iii）"重写器输入 96.5% 走 h0 残差通道"→ 更正为"h0 通道能量份额 96.5%（含交叉项，非互补份额）"；（iv）Q3 overlap 1/0/0 作废 → 以 3101 pre-o_proj 2/3/2 为权威。
- **3099 节**：H_E2a 判决语义更正（门不可达，False 非证据）；bridge 表述降级。
- **3075 节**：supermodular_diffuse 的结构定位降级为"高阶交互谱描述"；次模违反 467/1792 为权威数字。
- **3091 节**：跨模型 ρ 强统计确认撤回（ties 未修正），保留为候选相关。

### 4. 系统性方案（四项大任务 → Phase 3103+）
| 大任务 | 落地 Phase | 核心工作 | 成本 |
|---|---|---|---|
| T1 清理理论依赖链 | **3103** | RDC Unified Theory 主公式逐条标注证据等级（bit 级重算/统计支持/降级/撤回），产出 evidence_grade 总表 + 依赖图（哪条公式依赖哪些 Phase 的哪个数字）；免前向 | 低（纯分析） |
| T2 真关系组合 | **3104–3106** | 从单关系探针升级到关系组合剂量：双属性（如 属于+颜色）、语法组合（否定+转折）、跨层组合的 3093 全 V 流干预剂量实验；先 4B 小规模再 14B | 中 |
| T3 可读关系↔计算来源 | **3107–3109** | 外部语言模式族 ↔ 内部条件齿轮映射：以 3101 发现的跨条件稳定自然写入头组（21/12/14…）为源追踪起点，建立"模式族 → 写入头组 → 读出"三层查询表 | 中高 |
| T4 完整自回归预测 | **3110+** | 从单步读出扩展到多步生成预测器：条件化状态转移（含 3100 一阶预测器的多步传播与误差累积量化） | 高 |

**瓶颈判断**：当前三类硬伤（度量口径、干预语义、统计方法）已在本 Phase 闭环修复；下一阶段主战场是**组合**——单关系图谱（知识/语法/角色）已足够厚，破解"无限组合"必须直接攻组合机制，T2 是主线，T1 是其前置质检。

### 5. 产物
核查脚本与输出：`tests/gpt5_temp/p3101_review_verify1–5.py/.txt`；3101 主产物 `tests/glm5/result/rdc_query_construction_20260913/phase3101/omega_p99_upstream_writeup/`（npz8=15ec0e8c result8=e38d1707）；审查报告原文 `tests/glm5/result/gpt5_comprehensive_review_20260923/review.md`。
'''.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3101+3102)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3101 Omega-P99: 3093 semantics '
          'cracked (FULL V-STREAM replacement, bit-'
          'level d2b=0, med_c_new=0.5104 == 3093); '
          'pre-o_proj arbitration verdict '
          'seventh_carrier_absent (H_G1 2/3/2, H_G2 '
          '0.80/0.91/0.80, H_G3 0.297/0.325): swap '
          'recovery = distributed rebalancing, NOT '
          'natural-carrier return; NEW cross-family '
          'stable natural writer group 21/12/14 vs '
          'family-specific focal heads; Phase 3102: '
          'review R44/R48/R51/R52/R53/R54/R55 '
          'verified 7/7 (467/1792 recomputed '
          'exactly; RESID 26.55-41.18 pct matches '
          'npz; 1.6941=3x0.5647 unreachable; '
          'head(norm(dm)) basepoint-0; ASH+HSH!=1 '
          'cross-term; post-o_proj pseudo per-head); '
          'theory tightened (5 principles); '
          'systematic plan T1-T4 -> Phase 3103-3110; '
          'ledger n=[[N]].\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3101' not in prev:
        try:
            with io.open(wl, 'a', encoding='utf-8') as f:
                f.write(line_d.replace(
                    '[[N]]', str(len(json.load(
                        io.open(LEDGER,
                                encoding='utf-8'))
                        ['measurements']))))
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md rewrite (<=3000 chars) ----------
mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）+ qwen3-14b。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；产物 ...\\phase{N}\\{arm}\\；临时 gpt5_temp\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json；L14 connects 现为 meas_id 字符串列表。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`；纠错 append 不回改。

## 标准锚与精度
- bit-0 锚族：d1/d1b/d1v/d2/d2b/d3/d5/d6；d4 sha；跨相位 sealed 对比 0.0。
- **干预语义必须先逆向再复刻**（3101 教训）：3093=全 V 流替换（所有层 v_proj 钳回自然 bank，仅 L37 [0,4) 携带 prefix V）；"仅替换一层"与全流不等价（下游 V 重算）。
- per-head 分解只在 o_proj 输入侧合法（post-o_proj reshape=伪象，R55）。

## 统计与度量纪律（3102 收紧五原则）
1. cos 与 relative-L2 双报；能量份额含交叉项非互补（96.5% 取补教训）。
2. 门设计先验证可达域（3×0.5647=1.6941>1 教训）。
3. 秩相关用平均秩；跨模型合并须块校正（R48 撤回教训）。
4. JVP 准≠映射不混合；描述性 cos≠机制桥（head(norm(Δm)) 基点=0 教训）。
5. 次模权威判据=条件二阶差分（467/1792@tol0.02）；Möbius 谱仅描述。

## 机制链状态（3093-3102）
3093 L37 swap 恢复 bit 级复刻；3098 末块子步解剖；3099 MLP 重写弥散；3100 末块 MLP 近一阶（cos 0.94，RESD 26.55-41.18%）+ h0 通道主导；3101 **seventh_carrier_absent：swap 恢复=分布式再平衡非自然载体归还；L37 跨条件稳定自然写入头组 21/12/14 与族特异焦点头解耦**；3102 审查 7/7 证实+五原则收紧。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail/wc 坏→python + 写文件；-c stdout 丢→写文件再 Read；反引号被命令替换→chr(96)。
- 关键写入后必须 Grep/Read 复核磁盘；GPU 测试逐模型防 OOM；残留进程查 psutil。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出；关键发现重复 3 次。

## 下一步
- max=3102，下一 3103：**RDC 公式证据分级审计**（免前向，T1）→ 3104-3106 真关系组合剂量（T2 主线）→ 3107-3109 模式族↔写入头组映射（T3）→ 3110+ 多步自回归预测（T4）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
