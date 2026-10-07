# -*- coding: utf-8 -*-
"""Phase 3110 closeout (idempotent):
Ledger -> MEMO Phase 3110 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3110'
        r'\omega_p108_ksweep_minport')
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
assert res['curve_shape_truth'] == 'sharp_core'
assert res['specificity'] == \
    'truth_specific_diffuseness'
assert res['d_min_truth'] == 5

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3110
           for m in led['measurements']):
    claim = (
        'Omega-P108 (3110, offline T3 on frozen 3105 '
        'capture, no GPU, 27s) - K-sweep minimum '
        'readout-port dimension + cross-variable '
        'specificity.  Verdict sharp_core | '
        'truth_specific_diffuseness.  K-SWEEP: random '
        'K-dim subspace ridge refit AUC (median of 10 '
        'seeds) for truth = 0.9942 (K=5) / 0.9978 (10) / '
        '0.9989 (20) / 0.9998 (50) / 0.9997 (100) / '
        '0.9998 (200) / 0.9999 (400) / 0.9999 (800) / '
        '0.9999 (1600) / 0.9999 (2560): d_min_truth = 5 '
        '(0.2pct of coordinates!) -> sharp_core; ANY 5-'
        'dim random subspace reads the truth variable at '
        'AUC 0.994.  Same subsets, crit_rel target '
        '(8-way relation identity, one-vs-rest): 0.6014 '
        '(K=5, near chance) / 0.6552 / 0.7125 / 0.7973 '
        '/ 0.8577 / 0.9161 / 0.9469 (K=400, first >= '
        '0.95 x full 0.9829) / 0.9704 / 0.9802 / '
        '0.9829: d_min_rel = 400, 80x larger, smooth '
        'growth - concentrated/low-redundancy encoding.  '
        'SPECIFICITY: truth_specific_diffuseness - the '
        'diffuse redundancy is a property of the TRUTH '
        'variable, not a universal property of the '
        'residual stream; two variable classes have '
        'qualitatively different encoding topologies '
        '(truth = broadcast; relation identity = '
        'localized content code).  IMPLICATION: truth '
        'information cannot be carried by any single '
        'direction or low-dim manifold (a random 5-dim '
        'subspace aligns with any fixed direction only '
        'at expected ~K/2560 energy); it is broadcast '
        'across nearly all coordinates.  CANDIDATE '
        'PHYSICAL CARRIER (new alternative to test): a '
        'global first-moment signal - per-record overall '
        'activation level / integration state correlated '
        'with truth - would be readable from ANY '
        'subspace and is consistent with the flat '
        'spectrum (3108 M3 Gini 0.10); this does NOT '
        'contradict the semantic nature of the readout '
        '(3105/3106 conditional-balanced designs), it '
        'identifies the broadcast as confidence-like '
        'global state.  NEXT 3111 (verdict '
        'experiment): (a) direct test - per-record '
        '||h||, mean(h), median(h) as single scalar '
        'truth features, AUC each; (b) per-record '
        'centering (subtract each record cross-'
        'coordinate mean) then rerun K-sweep -> d_min '
        'explosion = broadcast_first_moment, persistence '
        '= distributed_higher_order; (c) per-coordinate '
        'single-dim truth AUC histogram - if most '
        'coordinates individually reach AUC 0.7+, every '
        'coordinate carries a truth component.  After '
        '3111: T4 multi-step autoregression + write-in '
        'side.')
    meas = {
        'meas_id': 'meas3110_omega_p108_'
                   'ksweep_minport',
        'phase': 3110,
        'claim': claim,
        'verdict': 'sharp_core_'
                   'truth_specific_diffuseness',
        'anchors': 'design_seal.json frozen before '
                   'computation: K list, 10 seeds, '
                   'd_min rule 0.95 x full, shared '
                   'subsets for both targets, bands '
                   '20/200, specificity rule 3x/0.9; '
                   'lambda 0.01 recorded; '
                   'standardization frozen from full '
                   'train',
        'artifacts': {
            'result': 'phase3110/omega_p108_'
                      'ksweep_minport/'
                      'result.json',
            'seal': 'phase3110/omega_p108_'
                    'ksweep_minport/'
                    'design_seal.json'},
        'hashes': {},
        'note': 'offline only; SMOKE clean (first '
                'try); full-dim truth 0.9999 and '
                'crit_rel OVR 0.9829 baselines '
                'recorded',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3110_omega_p108_ksweep_minport')
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

# ---------- MEMO Phase 3110 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3110:' not in memo:
    sec = u'''## Phase 3110: Ω-P108 K-sweep 最小读出端口维度——真值 d_min=5（sharp_core，任意 5 维随机子空间 AUC 0.994），crit_rel d_min=400（80 倍，平滑增长）→ truth_specific_diffuseness：弥散冗余是真值变量特有，两类变量编码拓扑定性不同；提出全局一阶矩广播信号为候选物理载体，3111 判决 [[NOW]]

**性质**：T3 第 4 Phase（读出端三部曲收官对照），纯离线（3105 冻结 capture，免 GPU，27s，SMOKE 一次通过）。预注册（design_seal.json 先于一切统计冻结）：K ∈ {5,10,20,50,100,200,400,800,1600,2560} × 10 seeds，同一批随机子集喂两个目标；d_min = 中位 AUC ≥ 0.95×全维的最小 K；带宽 20/200；特异性规则 3×/0.9。λ=0.01 记录值，标准化冻结自全 train。

### 1. K-sweep 曲线：两条定性不同的曲线
| K | 5 | 10 | 20 | 50 | 100 | 200 | 400 | 800 | 1600 | 2560 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| truth AUC（10 seeds 中位） | **0.9942** | 0.9978 | 0.9989 | 0.9998 | 0.9997 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| crit_rel OVR AUC | 0.6014 | 0.6552 | 0.7125 | 0.7973 | 0.8577 | 0.9161 | **0.9469** | 0.9704 | 0.9802 | 0.9829 |

真值：**d_min = 5**（0.2% 坐标，sharp_core）——任意 5 维随机子空间读出 AUC 0.994，且几乎无 K 依赖（平台从 K=5 就开始）。crit_rel：d_min = **400**（80 倍），随 K 平滑增长——信息相对集中、低冗余。

### 2. 特异性判决：truth_specific_diffuseness
弥散冗余是**真值变量的属性**，不是残差流的普遍属性。两类变量编码拓扑定性不同：真值 = 广播式（broadcast）；关系身份（crit_rel） = 局部化内容码（localized content code）。这为"不同语义变量具有不同编码拓扑"（400b 系列发现）提供了读出端的定量版本。

### 3. 理论推论与候选物理载体
若真值信息由单方向或低维流形承载，任意 5 维子空间与之对齐的期望能量 ~5/2560——不可能 AUC 0.994。因此真值信息**广播在几乎所有坐标中**。候选物理载体（新替代解释，3111 判决）：**全局一阶矩信号**——per-record 的总体激活水平/整合状态与真值相关，任何子空间都能读到它；这与 3108 M3 平坦谱（Gini 0.10）一致。注意这不否定读出的语义性（3105/3106 条件平衡设计已控表面线索），而是把"广播"具体化为类置信度的全局状态——与 m margin（模型自有读数）的语义内容相容。

### 4. 硬伤
① 单配置（3105 last\\|L8）；② K=5 处 10 seeds 的方差未见异常但未做正式置信区间；③ crit_rel 全维基线 0.9829 是 8 类 OVR 均值，其 d_min=400 依赖该基线（若基线更低判据更严）；④ 3105 材料的真值条件设计可能引入全局状态差异（正确/错误事实的整体加工差异）——这正是候选载体假说的来源，需 3111 显式判决而非默认。

### 5. 3111 预注册（观测前冻结于 3111 seal）
① **直接检验**：per-record 的 \\|\\|h\\|\\|、mean(h)、median(h) 作为单标量真值特征，各自 AUC——若 \\|\\|h\\|\\| AUC ≥ 0.9 → 广播信号实锤；② **per-record 中心化**（每记录减去其跨坐标均值）后重跑 K-sweep → d_min 暴涨 = broadcast_first_moment；仍 sharp_core = distributed_higher_order；③ **单坐标真值 AUC 直方图**：若多数坐标单独达 AUC 0.7+，则每坐标都携带真值分量。之后 T4 多步自回归 + 写入端条件化结构。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3110/omega_p108_ksweep_minport/`（result.json、design_seal.json、run_log.txt）；脚本 `tests/glm5/phase3110_omega_p108_ksweep_minport.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3110)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3110 Omega-P108 (offline T3, no GPU): '
          'K-sweep min port + specificity -> sharp_core | '
          'truth_specific_diffuseness. truth: d_min=5 '
          '(0.2pct coords, AUC 0.994 at K=5, flat '
          'plateau) - ANY 5-dim random subspace reads '
          'truth; crit_rel: d_min=400 (80x), smooth '
          'growth 0.60->0.98. Diffuse redundancy is '
          'TRUTH-SPECIFIC: two variable classes have '
          'qualitatively different encoding topologies '
          '(truth=broadcast, relation identity=localized '
          'content code). Candidate carrier: global '
          'first-moment signal (per-record activation '
          'level / integration state) - readable from '
          'any subspace, consistent with flat spectrum. '
          'NEXT 3111: ||h||/mean AUC direct test + '
          'per-record centering K-sweep rerun + '
          'per-coordinate AUC histogram -> '
          'broadcast_first_moment vs '
          'distributed_higher_order; then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3110 Omega-P108' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md update (append-style) ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3110' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3109）',
        '## 机制链状态（3110）\n'
        '- 3110：K-sweep 最小端口维度。真值 d_min=5'
        '（sharp_core：任意 5 维随机子空间 AUC 0.994，'
        '平台平坦）；crit_rel d_min=400（80 倍，平滑）'
        '→ truth_specific_diffuseness：弥散冗余是真值'
        '特有，truth=广播式 / 关系身份=局部内容码。候选'
        '载体=全局一阶矩信号（per-record 激活水平/整合'
        '度），3111 判决。\n')
    mem_new = mem_new.replace(
        'max=3109', 'max=3110').replace(
        '下一 3110：**K-sweep 最小读出端口维度 + 跨变量特异性对照**（弥散度定量）→',
        '下一 3111：**广播信号判决**（‖h‖/mean 直接检验+中心化 K-sweep+单坐标直方图）→ 之后 T4 + 写入端')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
