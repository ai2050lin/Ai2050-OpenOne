# -*- coding: utf-8 -*-
"""Phase 3104 closeout (idempotent):
Ledger -> MEMO Phase 3104 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3104\omega_p102_relation_vs_endpoint')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
cc = json.load(io.open(OUTD + r'\confound_check.json',
                       encoding='utf-8'))
mat_sha = res['material_sha8']
assert mat_sha == '8447a471', mat_sha
g = res['gates']
verb = res['verdict']

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3104
           for m in led['measurements']):
    claim = (
        'Omega-P102 (3104, 4B, 2064 prompts) - '
        'relation-vs-endpoint minimal-pair material '
        '(8 relations x 28 entities, 672 pair-unique '
        'triples, true/false differ in exactly 2 '
        'predicate tokens).  VALID POSITIVE: asserted '
        'relation is bound into endpoint states - '
        'object-end decode 100pct TEST / 95.9pct TEST-E '
        'at hs[12] (73.9pct at hs[4]) vs endpoint-'
        'embedding baseline 13.1pct = chance; frozen '
        'extractors extrapolate to new pairs and '
        '(mostly) new entities; bow surface ceiling '
        '72.1pct.  NEGATIVE + CORRECTION: truth gate '
        '(K3) was ill-posed - truth = external graph '
        'membership never observable in context '
        '(differing from 2754 where truth is chain-'
        'derivable); yes/no margin at chance (AUC '
        '0.497/0.490, pair-strict 0.324, mean '
        'reversed) = correct model behavior, not '
        'shortcut evidence; T2 probe AUC 0.75-0.82 '
        'CONFOUNDED by d_rel frequency artifact (cue '
        'AUC alone 0.667/0.710; true conditions '
        'always contain queried relation in '
        'distractors).  Script verdict '
        'endpoint_shortcut_dominant_candidate is '
        'DOWNGRADED to: K1/K2 valid positive, K3 '
        'gates invalid-as-designed.  NEXT 3105: '
        'in-context-derivable truth (2754 chain '
        'style) + condition-balanced distractor '
        'relations; keep binding measurement.')
    meas = {
        'meas_id': 'meas3104_omega_p102_'
                   'relation_vs_endpoint',
        'phase': 3104,
        'claim': claim,
        'verdict': 'K1_K2_relation_binding_valid;'
                   '_K3_gate_ill_posed_confounded',
        'anchors': 'material sha8=8447a471; A1 '
                   'determinism=0; A3 splits '
                   'disjoint; A4 672:1344 balanced; '
                   'A6 multiset 20 pairs; A7 perm '
                   '0.135=chance',
        'artifacts': {
            'result': 'phase3104/omega_p102_'
                      'relation_vs_endpoint/'
                      'result.json',
            'material': 'phase3104/omega_p102_'
                        'relation_vs_endpoint/'
                        'material.json',
            'confound': 'phase3104/omega_p102_'
                        'relation_vs_endpoint/'
                        'confound_check.json',
            'capture': 'phase3104/omega_p102_'
                       'relation_vs_endpoint/'
                       'capture.npz'},
        'hashes': {'material_sha256_8': mat_sha},
        'note': '4B BF16 eager batch1; capture '
                '2064x6pos x9slots x2560 fp16; '
                'probes ridge L2 lam in '
                '(0.01,0.1,1.0) val-selected',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3104_omega_p102_relation_vs_endpoint')
    led.pop('ledger_sha256_8', None)
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
    o.append('ledger already upserted')

# ---------- MEMO Phase 3104 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3104:' not in memo:
    sec = u'''## Phase 3104: Ω-P102 关系编码 vs 端点捷径——K1/K2 关系绑定正结果成立，K3 真值门不适定与 d_rel 混杂自纠（relation_vs_endpoint，4B 先导）[[NOW]]

**性质**：T2 主线首 Phase（3102 方案 → 审查第 8 节任务 B）。合成事实三元组网络最小对素材，检验"模型内部编码关系本身还是端点捷径"。**本 Phase 最重要的产出是方法论自纠**：K3 真值门设计不适定 + d_rel 频率混杂的量化证实——探针的高 AUC 不能归因于真值编码。产物 `tests/glm5/result/rdc_query_construction_20260913/phase3104/omega_p102_relation_vs_endpoint/`（result.json、material.json sha8=8447a471、confound_check.json、capture.npz 2064×6×9×2560 fp16、design_seal.json 预注册）。

### 1. 素材与锚（观测前冻结）
8 关系×28 实体，每（关系,主语）出度 3 → 672 真三元组；**pair-uniqueness**：每实体对至多属于一个真关系。每对 3 条件：true = (s,r,o)；false1/false2 = (s,ri,o)；**真/假 prompt 除 2 个关系词 token 外逐字节同构**（A6 multiset 断言 20 对全过：diff 恰为 {r:2}/{ri:2}）。干扰块 D（d_rel 同关系干扰 + d_subj 同主语干扰 + 5 中性）跨条件恒同；事实行 8 行，查询行逐字重复第 k 行（k ~ U(0..7) 冻结）。切分 pair-blocked：train 365 / val 48 / test 74 / testE 185 对（4 个 held-out 实体）。48 个链补全诱饵（表面熟悉但假）。锚全过：A1 确定性=0（bit 级）；A3 切分对级不相交；A4 平衡 672:1344、关系边缘均匀（228–258）；A5 谓词单词符；A7 标签置换 0.135≈chance 0.125。

### 2. T1/T2 正结果：关系绑定进端点状态（有效，不受混杂影响）
 ridge 探针（L2，λ∈{0.01,0.1,1.0} VAL 选）解 8-way "关键事实断言关系"：

| 特征 | TEST acc | TEST-E acc（新实体） | 基线 |
| --- | ---: | ---: | --- |
| crit_pred（谓词位，hs[4]..FN 全层） | 1.000 | 1.000 | 表面天花板 |
| **crit_obj（宾语末位）L0=hs[4]** | **1.000** | **0.739** | endpoint 0.131 / additive 0.108 / pos0 0.122 / bow 0.721 |
| **crit_obj L2=hs[12]** | **1.000** | **0.959** | 同上 |

- **端点嵌入基线 13.1%≈chance 12.5%**（pair-blocked 设计按构造摧毁对→关系先验）→ crit_obj 100%/95.9% 的信息只能来自谓词的上下文整合 = **断言关系在前 12 层内被绑定进宾语端点表示**，泛化到新对（100%）与新实体（95.9%）。
- **词袋 72.1% = 表面天花板**（谓词 embedding 稀释 1/28 后仍可读）；上下文状态 100% 显著超越。K1 门过（1.000 ≥ 0.80 且 ≥ 基线+0.30，bootstrap CI [0.824,0.914]）；K2 门过（冻结提取器 testE 1.000 / T2-AUC testE 0.825）。
- 与 2754 衔接：2754 在无分叉链上发现 L16 角色位置读出基础谓词方向；3104 在三元组网络上把绑定曲线前移定位到**宾语末位 hs[4]→hs[12]**（新实体需更深），并证明这是计算出来的（非 embedding 存储）。

### 3. K3 负结果的自纠：不适定门 + 混杂量化（本 Phase 核心方法论产出）
预注册 K3 门：真值探针 AUC≥0.80、零参数 yes/no margin 对内严格率≥0.65、端点基线≤0.60。实测：**margin AUC test 0.497 / testE 0.490 = chance；对内严格率 0.324；均值反向（true 0.158 < false 0.270）**；探针 AUC 0.75–0.82（差 0.00005 摸门）。逐层审查后发现两个设计缺陷，**结论从"shortcut 主导候选"降级**：

1. **真值在上下文不可观察（门不适定）**：真值标签 = 外部图 pair2rel 成员关系，而素材刻意排除 (s,o) 对的其他出现——模型在上下文中**无法知道**私有图的成员关系。2754 范式中真值由链内事实可推导，3104 不是。margin≈chance 是**正确行为**（模型不理会不可观察的私有标签），不能读作捷径证据。
2. **d_rel 频率混杂（探针信号的构造性解释）**：d_rel 恒为该对的真关系 → 真条件中查询关系在事实里出现 2 次、假条件仅 1 次（中性碰撞率 57.7%）。量化（confound_check.json）：**线索 AUC 单独达 test 0.667 / testE 0.710**（真条件线索率 1.000），可解释探针 0.75–0.82 的大部分；另有"事实中不同关系数"等相关的表面统计量同族。残余（约 0.09–0.13 AUC）在混杂排除前**不可归因于真值编码**。
3. 诱饵诊断（描述性）：链补全诱饵探针分 0.025 介于 true 0.294 与 false −0.154 之间（70.8% 低于 true 均值）——部分熟悉性响应；诱饵 margin −6.46 为 7 行格式伪迹，不作真值解读。

### 4. 硬伤清单（诚实登记）
① K3 门预注册时未检验"真值是否 in-context 可观察"这一前提（2754 与 3104 的真值语义不同源）；② d_rel 构造引入系统性关系频率差；③ 查询行逐字复制第 k 行使"真=句子复现"结构存在（虽因两条件都复制而不构成真值线索，但削弱与 2754 多跳范式的可比性）；④ 3104 无法区分"关系抽象绑定"与"谓词 token 局部拷贝到宾语位"——crit_obj 曲线只证明信息到了端点状态，未证明抽象化。

### 5. 3105 预注册（修正设计，观测前冻结）
**3105 = Ω-P103 in-context 真值一致性检验**：① 真值改为上下文可推导（2754 链式范式：呈现链事实 + 查询链上/链外三元组）；② 干扰关系在条件间精确平衡（查询关系在事实中的出现次数对 true/false 恒同）；③ 保留 3104 的绑定测量机制（T1 对 in-context 内容解码，无混杂）；④ 新增测量：绑定曲线（crit_obj 各层）与真值一致性信号的层间对齐——检验"关系绑定位置"与"真值计算位置"是否同层。门：G1 链真值 margin AUC≥0.70（in-context 可推导时模型应显著超 chance）；G2 绑定曲线复现（crit_obj testE≥0.90）；G3 平衡后探针 AUC 相对线索 AUC 的增量≥0.10 才可声称真值编码。

产物 sha8：material=8447a471；脚本 `tests/glm5/phase3104_omega_p102_relation_vs_endpoint.py`；SMOKE 5 轮迭代修复（w_dn.cuda、位置索引、基线 config、bootstrap None、NaN 守卫）后 bit 级确定性通过。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3104)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3104 Omega-P102 (4B, 2064 '
          'prompts): relation-vs-endpoint minimal '
          'pairs. VALID: relation bound into endpoint '
          'states (crit_obj 100pct test / 95.9pct '
          'testE at hs[12] vs endpoint 13.1pct '
          'chance; bow ceiling 72.1pct). CORRECTION: '
          'K3 truth gate ill-posed (truth = external '
          'graph, not in-context observable; margin '
          'AUC 0.497 = correct behavior) + d_rel '
          'frequency confound quantified (cue AUC '
          '0.667/0.710 explains probe 0.75-0.82 '
          'mostly). Verdict downgraded from '
          'endpoint_shortcut_dominant_candidate. '
          'NEXT 3105 Omega-P103: in-context-'
          'derivable truth + balanced distractor '
          'relations. material sha8=8447a471.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3104 Omega-P102' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md rewrite ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3104' not in mem_old:
    mem_new = mem_old.replace(
        '## 下一步',
        '## 机制链状态（3104）\n'
        '- 3104：关系绑定进端点状态成立（crit_obj '
        '100%/95.9% vs endpoint 13.1% chance）；K3 '
        '真值门不适定（外部图真值上下文不可观察）'
        '+ d_rel 频率混杂（线索 AUC 0.667/0.710）。\n'
        '- 3105 预注册：in-context 可推导真值 + '
        '条件平衡干扰。\n'
        '\n## 下一步')
    mem_new = mem_new.replace(
        'max=3103', 'max=3104').replace(
        '下一 3104：**T2 真关系组合剂量**（3093 全 V 流范式，4B 先导）→',
        '下一 3105：**Ω-P103 in-context 真值一致性**（3104 修正设计）→')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
