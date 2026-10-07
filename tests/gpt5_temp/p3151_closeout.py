# -*- coding: utf-8 -*-
"""Phase 3151 closeout: rev-3151b verdict correction +
five writes (ledger + MEMO + daily + MEMORY). Idempotent."""

import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                   'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTD = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913',
                    'phase3151', 'g1p1_combo_additive_vs_interaction')
RES = os.path.join(OUTD, 'result.json')
RES_V2 = os.path.join(OUTD, 'result_v2fix.json')
RES_B = os.path.join(OUTD, 'result_rev3151b.json')

MAIN_DISK = 'dab41389'
MAIN_RES = '3b4344c6'
MAIN_SEAL = 'a566dc96'
V2_RES = '2a26190f'
V2_DISK = 'd8cef721'
V2_SEAL = '44f17187'
CORR = ('a_3150_ok|interaction_pair_generalizes_at_k3_only|'
        'gain_above_capacity_k3|v3_n2h1_confirmed|k1_model_1_of_3')

raw = open(RES, 'rb').read()
assert hashlib.sha256(raw).hexdigest()[:8] == MAIN_DISK
r = json.loads(raw.decode('utf-8'))
assert r['seal_sha8'] == MAIN_SEAL
raw2 = open(RES_V2, 'rb').read()
assert hashlib.sha256(raw2).hexdigest()[:8] == V2_DISK
print('result verify ok')

now = time.strftime('%H:%M')

# 0) rev-3151b correction record
if not os.path.exists(RES_B):
    rb = dict(
        phase=3151, rev='3151b', kind='verdict_correction',
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        sign_convention=('margin = err_cand - err_base; margin<0 = '
                         'candidate better; gate: margin <= -2*MDE '
                         'for 3/3 seeds (S1 s7/s8/s9)'),
        corrected_verdict=CORR,
        evidence=dict(
            main_result_disk_sha8=MAIN_DISK, main_res_sha8=MAIN_RES,
            main_seal=MAIN_SEAL, v2fix_res_sha8=V2_RES,
            v2fix_seal=V2_SEAL,
            gates_k3_M1_vs_B4=[[-0.0289, 0.0068], [-0.0323, 0.0075],
                               [-0.0248, 0.0059]],
            gates_k39_M1_vs_B4_worse=[[0.7709, 0.1797],
                                      [1.0504, 0.3015],
                                      [0.8985, 0.1525]],
            M1_vs_B5_k3=[[-0.1687, 0.0918], [-0.1468, 0.0558],
                         [-0.1665, 0.0637]],
            M2_fail_all='margins +0.21..+0.24 >= 2MDE (worse) at k3/k39',
            v3=dict(b4_worst_fold_k39='shuiguo(1.2226)',
                    m2_fruit_margin_k3_k39=[0.348, 0.289],
                    confirmed=True),
            v5=dict(class_subspace_energy_k3=0.8813,
                    class_subspace_energy_k39=0.1244),
            v4_s3_b4_mean=0.6114,
            v0_material_ok=True, median_true_margin=1.1095),
        errors_corrected=[
            'rev-3151a: V2 was at k39 (M1 overfit) -> k3 independent recompute',
            'rev-3151b: gate sign inverted in code (m>0 as better) -> '
            'reinterpreted; script patched',
            'v3 prediction inequality inverted -> confirmed'])
    blob = json.dumps(rb, ensure_ascii=False, indent=1,
                      sort_keys=True).encode('utf-8')
    rb['res_sha8'] = hashlib.sha256(blob).hexdigest()[:8]
    open(RES_B, 'w', encoding='utf-8').write(
        json.dumps(rb, ensure_ascii=False, indent=1, sort_keys=True))
    seal_b = hashlib.sha256(open(RES_B, 'rb').read()).hexdigest()[:8]
    rb['seal_sha8'] = seal_b
    open(RES_B, 'w', encoding='utf-8').write(
        json.dumps(rb, ensure_ascii=False, indent=1, sort_keys=True))
    print('rev3151b written seal', seal_b)
else:
    print('rev3151b exists')

# 1) ledger append (n=288)
led = json.load(open(LED, encoding='utf-8'))
if not any(m.get('phase') == 3151
           for m in led['measurements']):
    led['measurements'].append({
        'phase': 3151, 'name': 'g1p1_combo_additive_vs_interaction',
        'seal_sha8': MAIN_SEAL, 'result_sha8': MAIN_DISK,
        'evidence_level': 'statistical',
        'model_scope': 'glm4-9b', 'n_rows': 738,
        'prereg_id': 'p3151', 'superseded_by': None,
        'verdict': CORR,
        'rev_note': ('rev-3151a v2fix k3 (res %s seal %s); '
                     'rev-3151b sign correction (res_rev '
                     'see result_rev3151b.json)' % (V2_RES, V2_SEAL)),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': r['runtime_s']})
    with open(LED, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    print('ledger -> n=%d' % len(led['measurements']))
else:
    print('ledger already has 3151')

# 2) MEMO append
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 3151:' not in memo:
    section = u'''
## Phase 3151: G1组合判决（T4 第34 Phase）[%s]

### 执行与修订链
面板 41实体×6类×3模板=**738行**（≥672 硬约束✓）；GLM4-9B；execution 冻结 e48acffd（含 UNIFIED_REVIEW 修正 A1-A3 观测前冻结）。SMOKE 79s 贯通（6 轮修：幻影编辑×4、dual 核矩阵缺项、numpy2 solve 语义、one-hot 精确共线 λ1e-3、v5 索引/右奇异）。正式跑 **1098.1s**。**rev-3151a**：V2 原固定在 k=39（M1 于中后层过拟合）→ k=3 独立重算（`result_v2fix.json`，与主跑 gates 数值位级一致=内部复现）；**rev-3151b**：配对门符号约定在代码中反转（m>0 误当"更优"）→ 重新解释判决并修补脚本；V3 预言不等式同步修正。**度量字典补丁（P0 F1 rev）**：margin=err_cand−err_base，<0 为候选优；门=margin≤−2×MDE 且 3/3 seeds。

### 第二篇长文裁决（全文 `UNIFIED_REVIEW_ADJUDICATION_v1.md`）
数值引用 **100%% 命中磁盘记录**（E1/E2/N1/N2/N2-h1 全表核对，含 288头/0.266/3.0%%/−6.014/B-A 0.997-1.010/水果0.05/末层接口cos0.864）；阶段A-E 与已冻结 TESTPLAN v1 逐段重合；"编号冲突"=幻觉（3150=Ω-P148=同一 Phase）；采纳增量 A1 三切分/A2 B5 容量对照/A3 失败样例/A4 8族表入 G2/A5 单一主验收结论。

### 主判决（修正后）：interaction_pair_generalizes_at_k3_only（×3）
1. **门层 k=3（承诺层）M1 胜 B4**：rank-5 网格 ALS 配对 margin **−0.0289/−0.0323/−0.0248**，全部 ≥2×MDE（0.0136/0.0151/0.0119），3/3 seeds；**k=39 处 M1 反而显著更差**（+0.77..+1.05，过拟合）→ 交互结构存在且**仅在承诺层泛化**；B4 曲线 k3 0.046→k15 0.50→k39 0.39。
2. **容量对照（k=3）**：M1 胜 B5(ELM-64) **−0.169/−0.147/−0.167** ≥2×MDE → 收益是结构性的非容量。
3. **M2（嵌入中介双线性=RDC v⊗φ 的嵌入行实现）全层全切分失败**（+0.21..+0.24 ≥2×MDE 更差）→ 交互**不由输入嵌入几何中介**；泛化交互只在自由网格分解下成立。
4. **V3 n2h1 预言 confirmed**：S2 留一类，k39 B4 最差折=水果（1.2226，n2h1b 硬类复现）；M2 水果折不优于 B4。
5. **V5 类子空间链接**：B4 残差能量在 5 维类子空间内份额 **88.1%%@k3 vs 12.4%%@k39** → k=3 的泛化交互≈类别子空间本身（N2-h1 U 的组合侧因果化）。
6. V4 新实体域：S3 B4 mean 0.611（加性只解释 ~39%%）；交互模型结构上不可用于新因子值。
7. **K1：模型 1/3 无触发**；但 B4@k39=0.39>5%% 且候选不胜任 → K1 压力实录，3152 双模型定夺。

### 锚
res disk sha8=**dab41389**（inner 3b4344c6，seal **a566dc96**）；v2fix res **2a26190f** seal **44f17187**；rev3151b 独立 seal；ledger n=**288**。产物 `phase3151\\g1p1_combo_additive_vs_interaction\\`（result.json/result_v2fix.json/result_rev3151b.json/collect.npz/execution.json）。

### 预注册 Phase 3152：G1-P2 K1 双模型复现+层位定位（TESTPLAN §4.2+K1）
①qwen3-4b + qwen3-14b 全管线复现（层位按深度分数 k/NL 对齐）；②非线性指数逐层曲线（GLM4 版已得：B4 rel-L2 曲线）+ M1 过拟合解剖（rank∈{1,2,5,10}×ridge∈{1e-3,1e-2,1e-1}@k∈{15,23,31,39}）；③**K1 判决层位冻结**：机制层 k*（承诺层，深度~8%%）与读出层（末前层）都报；K1 误差门适用候选声称作用层位（k*）——若 3 模型 k* 处 B4 误差>5%% 且 M1 不胜 B4 → 弃算子代数；④A3 worst-20 失败样例（并 3153 失败模态）。
''' % now
    with open(MEMO, 'a', encoding='utf-8') as f:
        f.write(section)
    print('MEMO appended')

# 3) daily log append
daily = open(DAILY, encoding='utf-8').read()
if 'Phase 3151 (gpt5 线)' not in daily:
    with open(DAILY, 'a', encoding='utf-8') as f:
        f.write(u'''
## Phase 3151 (gpt5 线) [%s]
- 第二篇长文（研究总评）裁决：数值引用 100%% 命中；阶段A-E=TESTPLAN 确认；编号冲突=幻觉；采纳 A1-A5（UNIFIED_REVIEW_ADJUDICATION_v1.md）。
- Phase 3151 G1-P1（738 行面板，GLM4-9B）：**interaction_pair_generalizes_at_k3_only**——M1(rank5 ALS) k=3 胜 B4 全加性（−0.029/−0.032/−0.025 全≥2×MDE）且胜 B5 容量对照；k=39 过拟合更差；M2 嵌入中介全线失败（RDC v⊗φ 嵌入行实现否证）；V5 残差 88%%@k3 落在 5 维类子空间=N2-h1 组合侧因果化；V3 水果硬类预言 confirmed；K1 模型 1/3 无触发。
- rev-3151a（V2 层位修正）/rev-3151b（配对门符号约定修正+度量字典补丁）。
- 锚：main disk dab41389 seal a566dc96；v2fix 2a26190f/44f17187；ledger n=288。3152=K1 双模型复现+层位定位。
''' % now)
    print('daily appended')

# 4) MEMORY.md update
mem = open(MEM, encoding='utf-8').read()
if '3151：G1-P1' not in mem:
    mem = mem.replace(
        u'- Ledger research\\gpt5\\atlas\\atlas_ledger.json（n=286 @3149）。',
        u'- Ledger research\\gpt5\\atlas\\atlas_ledger.json（n=288 @3151）。')
    mem = mem.replace(
        u'## 路线排期（TESTPLAN v1，2026-09-30 冻结）',
        u'''## 裁决速查 2（G1 主线）
- 3151：G1-P1 interaction_pair_generalizes_at_k3_only——M1(rank5 ALS) 承诺层 k=3 胜全加性 B4（3 seeds ≥2×MDE）且胜 B5 容量对照；k=39 过拟合；**M2 嵌入中介全线失败=RDC v⊗φ 嵌入行实现否证**；残差 88%@k3 在 5 维类子空间（N2-h1 组合侧因果化）；V3 水果硬类 confirmed；K1 模型 1/3 无触发。main disk dab41389 seal a566dc96；v2fix 2a26190f/44f17187。配对门符号约定：margin=err_cand−err_base<0 为优（rev-3151b）。

## 路线排期（TESTPLAN v1，2026-09-30 冻结）''')
    mem = mem.replace(
        u'- 排期：3150 P0 ✅（2026-10-01）→ 3151-3153 G1 组合 → 3154-3156 G2',
        u'- 排期：3150 P0 ✅、3151 G1-P1 ✅（2026-10-01）→ 3152-3153 G1 → 3154-3156 G2')
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    print('MEMORY updated')

print('CLOSEOUT DONE')
