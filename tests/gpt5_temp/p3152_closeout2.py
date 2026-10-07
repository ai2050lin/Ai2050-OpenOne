# -*- coding: utf-8 -*-
"""Phase 3152 closeout v2: MEMO/daily/MEMORY writes via token replace.
Idempotent. Ledger already n=289 (skip guarded)."""

import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
BASE = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913',
                    'phase3152', 'g1p2_tri_model_k1')
R4B = os.path.join(BASE, 'qwen3-4b', 'result.json')
R14B = os.path.join(BASE, 'qwen3-14b', 'result.json')
RG4 = os.path.join(BASE, 'glm4k1', 'result.json')
RSUM = os.path.join(BASE, 'summary', 'result_summary.json')


def shas(path):
    raw = open(path, 'rb').read()
    disk = hashlib.sha256(raw).hexdigest()[:8]
    r = json.loads(raw.decode('utf-8'))
    return disk, r


disk4b, r4b = shas(R4B)
disk14b, r14b = shas(R14B)
diskg4, rg4 = shas(RG4)
disksum, rsum = shas(RSUM)
assert r14b['res_sha8'] == '8a29eace' and r14b['seal_sha8'] == '2631b5fd'
assert rg4['res_sha8'] == '52d05025'
assert rsum['k1_triggered'] is False
print('disk sha: 4b=%s 14b=%s glm4k1=%s summary=%s' %
      (disk4b, disk14b, diskg4, disksum))

now = time.strftime('%H:%M')

# 2) MEMO append
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 3152:' not in memo:
    T = u'''
## Phase 3152: K1三模型定夺（T4 第35 Phase）[@NOW@]

### 执行
四模式单脚本 `phase3152_g1p2_tri_model_k1.py`：qwen3-4b（NL36/D2560，k*=3/readout=35）与 qwen3-14b（NL40/D5120，k*=3/readout=39）全管线（738 行面板×S1 三 seed×S2×S3×全层曲线×M1 曲线 10 深度点×门×M1 网格 48 fits×A3 worst20×V5）+ glm4k1（零 GPU：读 3151 collect.npz 重算 B4@k3/k39 三 seed、M1@k3 门、M1 网格、worst20）+ summary（K1 三模型汇总）。层位按深度分数对齐 k*=round(0.075NL)。SMOKE 47.3s 贯通；正式 4b 1339.3s / 14b 3554.7s / glm4k1 1385.9s。修 1 次：glm4k1 冻结 D=5120 错（glm4-9b 实为 4096）→ patch2 从 npz 动态读 D 并重冻结 execution（0446d74c）。

### 双锚位级复核（glm4k1）
B4@k39 三 seed mean=0.389835 与 3151 锚 drift=**0.00e+00**；M1@k3 margins **−0.0289/−0.0323/−0.0248** 与 3151 gates 位级一致（ALS seed 公式不变）→ 重算管线=3151 管线位级复现。

### 主判决：k1_not_triggered_b4_additive_at_kstar_operator_line_kept（×3）
K1 三模型 k* 报告（判决层位=机制层 k*≈深度 7.5%，预注册冻结）：

| 模型 | B4@k* | B4@读出层 | above 5pct 门 | M1@k* 三seed margin |
|---|---|---|---|---|
| glm4-9b | 0.0461 | 0.3898 | 否 | −0.0289/−0.0323/−0.0248 pass（2-6×MDE）|
| qwen3-4b | **0.0079** | 0.3316 | 否 | −0.0040/−0.0019/−0.0024 不过门 |
| qwen3-14b | **0.0807** | 0.3986 | **是** | −0.0432/−0.0448/−0.0445 pass（~5×MDE）|

- **K1 弃线条件（3 模型全 above 且 M1 全败）不成立**：above=1/3、m1_pass=2/3 → **算子代数线保住**；qwen3-14b 的 k* 处组合非线性被 rank-5 交互完全修复（0.081→0.036，改善 56pct）。
- M2（嵌入中介）三模型 k* 全败（margin +0.07/+0.27/+0.24）——再次否证嵌入行实现。
- **层位定律（新增，三模型一致）**：B4 组合误差从 k*=3 到读出层单调恶化 4.9×-42×（4b 42×、glm4 8.5×、14b 4.9×）；argmin 全在 k=0（embedding 处 one-hot 完全可加）。4b 曲线呈"k0-6 加性近完备（<0.01）→ k7 单层跳变 0.006→0.197 → 中层平台 0.43 → 缓降 0.33"；14b/glm4 平滑上升。
- **交互修复窗**：M1 仅 k≤10-11 改善（4b k3-10、14b k3-11、glm4 k3），k≥15 全线过拟合（读出层 margin +0.75/+1.08/+1.08）→ 交互结构=承诺层现象，读出层组合非线性不可由低秩交互修复。
- **M1 过拟合解剖**（rank{1,2,5,10}×ridge{1e-3,1e-2,1e-1}×4 深度层×3 模型=144 fits）：读出层 optimal rank=10（网格顶格）×3 模型；强正则 rank10 读出层仅微弱改善（glm4 0.365 vs B4 0.390；14b 0.400 vs 0.410；4b 0.294 vs 0.332）→ **读出层残差=高秩散布结构**，与 k* 残差=rank-5 类子空间（V5：4b 92.3pct@k* vs 14.9pct@读出；14b 51.9pct vs 18.0pct；glm4 88.1pct vs 12.4pct[3151]）形成对照。
- V3 留一类全 confirmed（worst：4b=金属、14b=颜色、glm4=水果）；V4 留一实例 B4 0.719/0.687/0.611（新实体误差 ~2× S1）；V0 材料 margin 中位 0.693/0.844/1.110（qwen logit 尺度低，记录型）。
- **A3 worst-20 跨模型**：qwen3-4b/14b 高重叠（床/沙发/黑/绿/黄=家具+颜色类），glm4 偏家具+错配（"狗是一种金属""窗帘是一种水果"）→ 失败模态两类：难类（颜色/家具）与错配组合；并 3153 解剖。

### 锚
qwen3-4b res **199b145d**（disk @D4B@）；qwen3-14b res **8a29eace** seal **2631b5fd**（disk @D14B@）；glm4k1 res **52d05025** seal **e302664b**（disk @DG4@）；summary res **1d17b42e**（disk @DSUM@，seal @SSUM@）；ledger n=**289**。产物 `phase3152\\g1p2_tri_model_k1\\{qwen3-4b,qwen3-14b,glm4k1,summary}\\`（result*.json/collect.npz×2/execution.json×4）。

### 预注册 Phase 3153：G1-P3 失败模态解剖（TESTPLAN 4.2+UNIFIED_REVIEW A3）
(1)三模型 worst-20 并/交分析：失败样本级 Jaccard + 失败模态分类表（难类 vs 错配 vs 高秩散布），glm4 错配子集（如 狗→金属）单独成列；(2)读出层高秩残差谱解剖：B4@读出层残差的 PCA 谱 + 类内/类外能量分解 + 残差与（实体轴×类轴）外积空间的子空间角——回答"读出层组合非线性的载体"；(3)k* 类子空间（5 维）与 M1 rank-5 因子网格的对齐检验（Pc 列 vs U/V 因子的主角谱）；(4)验收：失败模态分类覆盖率>=80pct + 高秩残差谱指纹跨模型一致（谱形状相关>=0.8）+ worst 交叉验证；(5)零 GPU 假设检验为主，GPU 仅补残差采集复用 3152 collect.npz。若高秩残差谱指纹跨模型不一致 → G1 关闭"交互=类子空间"叙事，3153 只交付描述分类。
'''
    T = (T.replace('@NOW@', now).replace('@D4B@', disk4b)
          .replace('@D14B@', disk14b).replace('@DG4@', diskg4)
          .replace('@DSUM@', disksum).replace('@SSUM@', rsum['seal_sha8']))
    with open(MEMO, 'a', encoding='utf-8') as f:
        f.write(T)
    print('MEMO appended')
else:
    print('MEMO already has 3152')

# 3) daily log append
if not os.path.exists(DAILY):
    open(DAILY, 'w', encoding='utf-8').write('# 2026-10-01\n')
daily = open(DAILY, encoding='utf-8').read()
if 'Phase 3152 (gpt5 线)' not in daily:
    D = u'''
## Phase 3152 (gpt5 线) [@NOW@]
- Phase 3152 G1-P2 K1 三模型定夺：**k1_not_triggered_b4_additive_at_kstar_operator_line_kept**（above=1/3、m1_pass=2/3）→ 算子代数线保住。
- 层位定律：B4 组合误差 k*→读出层恶化 4.9×-42×（argmin 全 k=0）；M1 交互修复窗仅 k≤10-11；读出层残差高秩（optimal rank=10 顶格×3 模型）vs k*=rank-5 类子空间（V5 92/52/88pct@k*）。
- glm4k1 双锚位级复核：B4@k39 drift 0.00e+00；M1@k3 margins 与 3151 gates 位级一致。
- 修 1 次：glm4k1 D=5120→4096（patch2 从 npz 动态读）。
- 锚：4b res 199b145d disk @D4B@；14b res 8a29eace seal 2631b5fd disk @D14B@；glm4k1 res 52d05025 seal e302664b disk @DG4@；summary res 1d17b42e seal @SSUM@ disk @DSUM@；ledger n=289。3153=G1-P3 失败模态解剖（预注册已写入 MEMO）。
'''
    D = (D.replace('@NOW@', now).replace('@D4B@', disk4b)
          .replace('@D14B@', disk14b).replace('@DG4@', diskg4)
          .replace('@DSUM@', disksum).replace('@SSUM@', rsum['seal_sha8']))
    with open(DAILY, 'a', encoding='utf-8') as f:
        f.write(D)
    print('daily appended')

# 4) MEMORY.md update
mem = open(MEM, encoding='utf-8').read()
changed = False
old_led = u'（n=288 @3151）。'
if old_led in mem:
    mem = mem.replace(old_led, u'（n=289 @3152）。')
    changed = True
old_sch = u'- 排期：3150 P0 ✅、3151 G1-P1 ✅（2026-10-01）→ 3152-3153 G1 → 3154-3156 G2 关联 → 3157-3159 G3 自回归 → 3160 整合 v5.4。每主线第一 Phase 必含否证判决。'
new_sch = (u'- 3152：G1-P2 K1 三模型定夺 k1_not_triggered_b4_additive_at_kstar——above=1/3（仅 14b 0.0807>5pct）、m1_pass=2/3（glm4 −0.029/−0.032/−0.025、14b −0.043/−0.045/−0.045 全过）；4b k* 加性近完备 0.0079。层位定律：B4 误差 k*→读出层恶化 4.9×-42×；M1 修复窗仅 k≤10-11；读出层残差高秩（optimal rank=10 顶格×3）vs k*=rank-5 类子空间。glm4k1 双锚位级复现（B4@k39 drift 0；M1@k3 margins=3151 gates）。锚：4b 199b145d、14b 8a29eace/2631b5fd、glm4k1 52d05025/e302664b、summary 1d17b42e。\n'
           u'- 排期：3150 P0 ✅、3151 G1-P1 ✅、3152 G1-P2 ✅（2026-10-01）→ 3153 G1-P3 失败模态解剖 → 3154-3156 G2 关联 → 3157-3159 G3 自回归 → 3160 整合 v5.4。每主线第一 Phase 必含否证判决。')
if old_sch in mem:
    mem = mem.replace(old_sch, new_sch)
    changed = True
if changed:
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    print('MEMORY updated')
else:
    print('MEMORY anchors not found (check manually)')

print('CLOSEOUT V2 DONE')
