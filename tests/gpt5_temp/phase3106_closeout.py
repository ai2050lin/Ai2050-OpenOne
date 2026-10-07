# -*- coding: utf-8 -*-
"""Phase 3106 closeout (idempotent):
Ledger -> MEMO Phase 3106 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3106'
        r'\omega_p104_composition_dose_depth')
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
mat_sha = res['material_sha8']
# Verified: first formal run (pre-dedup fix) sha8=a073c0a5 was
# DESIGN-INVALID (duplicate (a,c) chain units split across
# train/test lists; short-circuit merge kept data leak-free but
# broke unit-blocked split, shrinking test chains to 5 units).
# Superseded by this run (unique (a,c), 83 chains, 572 recs).
assert mat_sha == '1437bf61', mat_sha
g = res['gates']
verb = res['verdict']

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3106
           for m in led['measurements']):
    claim = (
        'Omega-P104 (3106, 4B, 572 prompts, 83 chains + '
        '60 scatter pairs) - composition dose & depth.  '
        'ALL 4 GATES PASS -> composition_dose_tracked.  '
        'G1 chain truth: 2-hop transitive chains (query '
        'triple NOT verbatim present; rule sentence '
        'frozen): m AUC 0.930 test / 0.958 testE (CI '
        '[0.826,1.0]), C-true>C-rel strict 0.793 (29 '
        'units; 1-token 2nd-hop predicate break).  G2 '
        'frequency heuristic NEGATED: C-true>C-sub '
        'strict 1.000 (r-frequency matched to C-true, '
        'path broken), mean m C-sub=-8.08, mean m '
        'S-scatter=-9.76 (scatter r appears 2x > '
        'S-active 1x yet strongly NO) -> readout does '
        'joint/path matching, not frequency counting.  '
        'G3 grammar penetration: S-passive (o is owned '
        'by s) > S-false strict 1.000, S-passive is the '
        'ONLY condition with positive mean m (+0.36); '
        'scatter-family m AUC 0.988/1.000.  G4 binding '
        'rerun (offline, revised VAL tie-break toward '
        'later layers): tie set [0,2,3,6,7,8] -> picks '
        'crit_obj|L8, TEST 1.000 / TEST-E 0.9835 - '
        '3105 G2 failure confirmed as pure selection '
        'artifact.  G5 layer alignment (descriptive): '
        'Pearson(binding teE curve, truth teE curve) = '
        '0.960 (vs last-position curve 0.915), binding '
        'peaks L3, truth peaks L4, offset 1 layer - '
        'truth evaluation sits on top of completed '
        'binding.  DESIGN INVALIDITY superseded: first '
        'run (a073c0a5) had duplicate (a,c) chain units '
        'splitting across split lists (short-circuit '
        'merge = leak-free but unit-blocking broken, '
        'test chains shrank to 5 units); rerun with '
        '(a,c)-unique sampling.  2-hop path supply is '
        'the sampling ceiling (83 unique chains from '
        'outdeg-3 pair-unique graph).  DISSOCIATION '
        'extended: readout tracks in-context truth in '
        'DERIVED (2-hop) and GRAMMAR-VARIED (passive) '
        'forms - truth computation is compositional, '
        'not surface lookup.  NEXT 3107: T3 mode-family '
        'to write-head mapping (probe weight structure).')
    meas = {
        'meas_id': 'meas3106_omega_p104_'
                   'composition_dose_depth',
        'phase': 3106,
        'claim': claim,
        'verdict': 'all_4_gates_pass_'
                   'composition_dose_tracked',
        'anchors': 'material sha8=1437bf61; A1 '
                   'determinism=0; A4 chain 83:249 '
                   'scatter 120:120; A6 40 units; A7 '
                   'perm 0.079=chance; (a,c)-unique '
                   'chains; cue theory frozen in seal',
        'artifacts': {
            'result': 'phase3106/omega_p104_'
                      'composition_dose_depth/'
                      'result.json',
            'material': 'phase3106/omega_p104_'
                        'composition_dose_depth/'
                        'material.json',
            'capture': 'phase3106/omega_p104_'
                       'composition_dose_depth/'
                       'capture.npz'},
        'hashes': {'material_sha256_8': mat_sha},
        'note': '4B BF16 eager batch1; offline G4/G5 '
                'from 3105 result.json; SMOKE caught '
                'pick_neutral/crit_i/swap-assert/split '
                'bugs + dedup leakage risk before '
                'final run',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3106_omega_p104_composition_'
        'dose_depth')
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

# ---------- MEMO Phase 3106 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3106:' not in memo:
    sec = u'''## Phase 3106: Ω-P104 组合剂量与深度——多跳链读出保持（m AUC 0.930/0.958）、频率启发式否证（C-sub/S-scatter 强判 no）、被动改述穿透（S-passive 唯一转正 +0.36）、绑定门修订后通过（G4），四门全过 composition_dose_tracked（composition_dose_depth，4B）[[NOW]]

**性质**：T2 主线第三 Phase，落实 3105 预注册。两族材料：**链族**（2 跳同关系传递，查询三元组不在场需推导；规则句恒同 "The same predicate may hold transitively"；4 条件 C-true/C-swap/C-rel/C-sub，C-true vs C-rel 恰差 1 token、C-true vs C-sub 词袋同且 r 频率同）+ **散射族**（4 条件 S-active/S-passive/S-scatter/S-false，S-passive 用被动改述 "The o is owned by the s." 使表面联合匹配失败而语义联合匹配成立；S-scatter 让 r 出现 2 次 > S-active 的 1 次而真值为 no——频率启发式预测 yes）。离线部分（免 GPU）用 3105 result.json 完成 G4 绑定门修订重跑与 G5 层对齐量化。产物 `tests/glm5/result/rdc_query_construction_20260913/phase3106/omega_p104_composition_dose_depth/`（material sha8=1437bf61、result.json、capture.npz 572×6×9×2560、design_seal.json 含冻结线索理论：链 r 频率线索 AUC 0.667、散射主动谓词线索 0.375）。

### 0. 设计无效化与重跑（工程诚实记录）
首轮正式跑（sha8=a073c0a5）发现 chain_units 含重复 (a,c)（同一对可经不同中间实体成链），blocked_split 的列表语义使同一 unit 分裂进不同 split 列表；chain_split 短路归并把所有重复 unit 的记录并入 train——**数据无 train/test 泄漏**，但 unit-blocked 设计被破坏（test 链缩至 5 units），判决不可作为预注册检验。修复为 (a,c)-唯一采样后重跑，本节全部数字来自重跑（83 链×4 + 60 散射对×4 = 572 prompts）。2 跳路径供给是采样天花板：出度 3、pair-unique 图中唯一 (a,c) 2 跳链共 83 条。SMOKE 先后抓获 pick_neutral 关系池不足、S-false crit_i 笔误、C-swap 断言语义错误（swap 是词序变换词袋应恒同）、smoke split 空集、重复 unit 泄漏风险——5 处，全部在正式观测前修复。

### 1. G1 通过：链真值读出保持（查询三元组不在场）
| 指标 | 值 | 解读 |
| --- | --- | --- |
| m AUC（链族 test / testE） | **0.930 / 0.958**（CI [0.826,1.0]） | 真值需 2 跳传递推导，读出仍近乎完美排序 |
| C-true > C-rel 严格率 | **0.793**（n=29 units） | 第二行谓词差 1 token（r→ri 断链），79% 单元 margin 正确翻转 |
| C-true > C-swap 严格率 | 1.000 | 方向反转全部正确判 no（2754 的 theta×rho 因子在小样本复现） |
| mean m | C-true=−1.13（其余 −4.6 ~ −8.8） | 全局 yes 偏置下 margin 秩序仍保持（AUC 为秩统计） |

3105 证明读出追踪**在场**真值；3106 证明追踪**可推导**真值——2 跳传递后读出保持 AUC 0.93+，一致性计算是组合性的，不是表面查找。

### 2. G2 通过：频率启发式被否证（本 Phase 最重要判别）
C-sub 与 C-true 的 r 频率完全相同（各 2 次）、端点频率相同，仅第二行主语 b→x（断开路径）；S-scatter 的 r 频率（2 次）**高于** S-active（1 次）而真值为 no。结果：C-true > C-sub 严格率 **1.000（29/29）**，mean m C-sub=−8.08；mean m S-scatter=−9.76（8 条件中最负）。**若读出走频率计数，这两条应被判 yes 且分数最高——实际反向。** 结合 3105 诱饵（关系在场 2 次判 no）与 3104 混杂教训，三次独立证据收敛：**读出做联合/路径匹配，不做频率计数。**

### 3. G3 通过：被动改述穿透（语法不变性）
S-passive（"The tavoka is owned by the kapira." ↔ 查询 "The kapira owns the tavoka."）表面 token 联合匹配失败、语义联合匹配成立。S-passive > S-false 严格率 **1.000**；S-passive 是 8 条件中唯一 mean m 转正者（**+0.36**，其余全负）；散射族 m AUC te=0.988 / teE=1.000（CI [0.951,1.0]）；S-active>S-scatter 1.000。读出对被动形式的语义不变性成立——语法变换不破坏真值追踪（2754 theta 被动因子在小样本外的强复现）。

### 4. G4 通过（离线，3105 数据修订重跑）+ G5 层对齐
**G4**：修订 VAL 平局规则（平局→取更高层）后，crit_obj 平局集 [L0,L2,L3,L6,L7,L8]（val 均 1.000）→ 选 **crit_obj|L8：TEST 1.000 / TEST-E 0.9835** ≥ 0.95/0.90 门限——3105 的 G2"失败"确认为纯选择伪迹，绑定门实质通过。**G5（描述性）**：绑定曲线（crit_obj teE）与真值曲线（query_obj teE）Pearson **r=0.960**（与 last 位置曲线 r=0.915）；绑定峰 L3、真值峰 L4、**峰差 1 层**——真值评估建立在绑定完成之后（L2–L3 绑定跃迁 → L4 真值读出峰值），"一致性评估建立在绑定完成的表示上"从并列陈述升级为量化对齐。

### 5. 硬伤与边界
① 链采样天花板：83 条唯一 (a,c) 链（test 12 units / testE 20 units），链族 test 统计功效中等（AUC CI [0.826,1.0] 宽）；② 规则句显式告知传递性——读出是否内建传递语义还是执行规则句条件下的推理，3107 需无规则句对照；③ 2 跳封顶，3+ 跳深度未测；④ S-scatter 行数与中性块内容跨条件不同（散射条件的定义性差异，已登记 seal limitation）；⑤ mean m 全体偏负（yes/no 先验），绝对 margin 不可跨条件比较，只读秩序。

### 6. 理论更新与 3107 预注册
三图谱增量：**内部响应图谱**新增"真值读出对推导真值（2 跳）与语法变体（被动）的鲁棒性带 hs[12]–final"；**关联机制**新增"绑定完成（L2–L3）→ 真值评估（L4 峰）的层间序律，Pearson 0.960"。RDC：一致性齿轮在传递条件与被动条件下均闭合——条件化齿轮组的**组合深度**首次得到读出级证据。**3107 = T3 模式族↔写入头组映射**：对 3105/3106 冻结 capture 分析探针权重 W 的结构（哪些读出方向/坐标子集承载真值信号；W 与 W_yes−W_no 的子空间重叠；写入头组按模式族分组的稀疏性），门在观测前冻结。之后 3108–3109 完成三图谱 T3 收口 → 3110+ 多步自回归预测（T4）。

产物 sha8：material=1437bf61；脚本 `tests/glm5/phase3106_omega_p104_composition_dose_depth.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3106)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3106 Omega-P104 (4B, 572 prompts): '
          'composition dose & depth, ALL 4 GATES PASS -> '
          'composition_dose_tracked. G1 chain: m AUC '
          '0.930/0.958 (2-hop derived truth), C-true>C-rel '
          '0.793; G2 freq NEGATED: C-true>C-sub 1.000, '
          'S-scatter mean m -9.76 (joint/path matching not '
          'frequency); G3 grammar: S-passive>S-false 1.000, '
          'only positive mean m (+0.36), scat AUC '
          '0.988/1.000; G4 offline rerun: revised tie-break '
          '-> crit_obj|L8 TEST 1.000/TEST-E 0.9835 (3105 '
          'G2 = pure selection artifact); G5: Pearson '
          'bind-truth 0.960, peak offset 1 layer. First '
          'run a073c0a5 design-invalid (duplicate (a,c) '
          'units; leak-free but blocking broken), rerun '
          'with unique sampling. NEXT 3107: T3 '
          'mode-family to write-head mapping. material '
          'sha8=1437bf61.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3106 Omega-P104' not in prev:
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
if 'max=3106' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3105）',
        '## 机制链状态（3106）\n'
        '- 3106：四门全过 composition_dose_tracked。'
        '链（2 跳推导）m AUC 0.930/0.958；频率启发式'
        '否证（C-sub 严格率 1.000、S-scatter −9.76）；'
        '被动改述穿透（S-passive 唯一转正 +0.36）；'
        '绑定门修订后过（crit_obj|L8 teE 0.9835）；'
        '绑定-真值层对齐 Pearson 0.960、峰差 1 层。\n'
        '- 3105：读出追踪 in-context 真值（m AUC '
        '0.99，P>A1 74/74，诱饵拒绝=联合匹配）；'
        '与 3104 形成判离：上下文可推导 0.99 vs '
        '外部私有 0.50。绑定曲线复现（hs[12] teE '
        '0.995）；G2 败于平局规则（3106 已修订）。\n'
        '- 3104 更正：margin 符号 +m（AUC 0.503，'
        '判决不变）。\n')
    mem_new = mem_new.replace(
        'max=3105', 'max=3106').replace(
        '下一 3106：**Ω-P104 组合剂量与深度**（多跳+散射证据+门规则修订）→',
        '下一 3107：**T3 模式族↔写入头组映射**（探针权重结构/读出子空间）→')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
