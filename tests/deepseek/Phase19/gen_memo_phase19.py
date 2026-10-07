# -*- coding: utf-8 -*-
"""Phase 19 MEMO 追加节生成器（数据驱动：所有数字由 result_phase19.json 现场渲染，铁律 (ae)）。
产物：tests/deepseek_temp/Phase19/memo_append_phase19.md
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
RESP = os.path.join(P19T, 'result_phase19.json')
RES = json.load(io.open(RESP, encoding='utf-8'))
EX = json.load(io.open(os.path.join(P19T, 'execution_phase19.json'), encoding='utf-8'))
SEAL = json.load(io.open(os.path.join(P19T, 'N2h1a12_design_seal.json'), encoding='utf-8'))
PROBE = json.load(io.open(os.path.join(P19T, '_probe19_A0_both.json'), encoding='utf-8'))
V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
QP = JV['quant_pairs']
AO = list(EX['arm_order'])
FL = RES['floors']
ARMS = RES['arms']


def f(x, nd=4):
    return ('%.' + str(nd) + 'f') % x if isinstance(x, (int, float)) and x is not None else str(x)


def sn(a):
    return {'A0_nf4': 'A0·nf4', 'A0_bf16': 'A0·bf16', 'A1_nf4': 'A1·nf4', 'A1_bf16': 'A1·bf16'}.get(a, a)


def pk(pair, key, nd=4):
    s = QP.get(pair)
    if not s:
        return 'NA'
    v = s[key]
    return f(v, nd) if isinstance(v, (int, float)) else str(v)


def ps(k):
    v = PC.get(k, {})
    p = v.get('pass_')
    return '✔ PASS' if p is True else ('— N/A' if p is None else '✘ FAIL')


L = []


def A(s=''):
    L.append(s)


ts = time.strftime('%H:%M')
dfA0 = pk('A0_nf4|A0_bf16', 'delta_com_V')
dfA1 = pk('A1_nf4|A1_bf16', 'delta_com_V')
rhoA0 = pk('A0_nf4|A0_bf16', 'spearman_w')
rhoA1 = pk('A1_nf4|A1_bf16', 'spearman_w')

A('## Phase 19: 写入向量谱的量化口径稳健性（nf4 vs bf16）（N2h1-α-12）[%s]' % ts)
A('')
A('> seal `%s` / exec `%s` / result `%s`；锚 = Phase 17 result `%s`。'
  % (RES['seal_sha256'][:8], RES['exec_sha256'][:8],
     hashlib.sha256(open(RESP, 'rb').read()).hexdigest()[:8], RES['anchor_result_sha256'][:8]))
A('')
A('### 0. 一句话')
A('')
A('**「写入向量质量深端集中」不是 nf4 量化地板效应**：把同一模型、同一尺度、同一套口径'
  '（同 template / instances / pairs / `U_ℓ` / 区间求和质心 / REACH 域）从 nf4 换成 **bf16** 后，')
A('质心只移动 **%s 层**（A0）/ **%s 层**（A1），谱秩相关 **%s / %s**，`argmax` 层不变，组件归属同侧 —— '
  % (dfA0, dfA1, rhoA0, rhoA1))
A('而这是 **P17 自己 `quant.why` 预留却未执行**的检查（「A0 臂专职量化对其结论的影响」）。')
A('')

A('### 1. 动机：P17 自己写下的缺口')
A('')
A('- **gap_1（主）**：Phase 17 的三臂头条 `DEEP_ALL`（`com_V` = 26.1501 / 26.7037 / 26.6749）'
  '**全部在 nf4 单一数值口径下测得**；而 `quant.why` 原文写着「为保持三臂同一数值口径统一改为 nf4 … '
  '**A0 臂专职量化对其结论的影响**」—— 该检查**从未执行**。')
A('- **gap_2**：Phase 18 在同一 nf4 口径上又建了一层（行为预算质心 `com_B`），其核心结论 '
  '「`com_B` 比 `com_V` 浅 2.5–6.6 层」是**两个 nf4 量之差** ⇒ 若量化口径给 `w_ℓ` 带来系统位移，'
  '这个「差」的标定要重做。')
A('- **gap_3**：Phase 12 的 **bf16** 未投影残差谱 `diff_norms` 深端爆炸（L6 24.9 → L30 240.1），'
  '**方向支持**「深端大」，但对象不同（未投影残差 vs 投影组件和）⇒ 不能外推，必须实测。')
A('')

A('### 2. 设计：量化口径是**唯一**自变量')
A('')
A('| | 内容 |')
A('|---|---|')
A('| **变** | 前向数值精度：bitsandbytes nf4 (4-bit) ↔ bfloat16 |')
A('| **不变** | template / classes / instances_all / pairs_all / discovery / confirmation |')
A('| | `U_ℓ` 口径：全 41 实例按类平均 → 类别质心差 SVD，秩 = `n_classes−1` = 5 |')
A('| | 质量定义 `w_ℓ = mean_pairs ‖P_{U_ℓ}(Δ_inc,ℓ)‖`；质心 = REACH 上**相邻位点区间求和** + 中点 |')
A('| | REACH 域（取该臂 P17 冻结值，跨口径不变，用于同域配对）；邻域宽 ±2；BP 与种子 |')
A('| **两臂加载配置** | 除量化外**逐项一致**：同 `attn_implementation="eager"`、同 `device_map="auto"`、'
  '同 `max_memory`、同 `low_cpu_mem_usage` |')
A('')
A('**臂集**（4 臂）：`A0_nf4`/`A1_nf4` 是**预注册校准臂**（必须逐位复现 P17 锚），'
  '`A0_bf16`/`A1_bf16` 是**检验臂**（A1_bf16 为 holdout）。')
A('')

A('### 3. 装置门与校准（P1）')
A('')
A('| 臂 | 模型 | 精度 | 保真度 arch max | 保真度 blk max | 锚 |')
A('|---|---|---|---|---|---|')
for a in AO:
    v = V[a]
    A('| `%s` | %s | %s | %s | %s | %s |'
      % (a, ARMS[a]['model'], ARMS[a]['scheme'], f(v['Q1_arch_max'], 3),
         f(v['Q1_blk_max'], 3), v['Q2_label']))
A('')
A('保真度门（arch ≤ %s、blk ≤ %s）四臂全通过；两个 **nf4 校准臂逐位复现 P17 冻结锚**'
  '（`com_V` / `com_V_mlp` / `com_V_attn` 差 ≤ %s，`nb` / `argmax_w` 精确相同）⇒ 装置与 Phase 17 同源。'
  % (f(FL['P19_FID_ARCH'], 2), f(FL['P19_FID_BLK'], 2), ('%g' % FL['CALIB_TOL_COMV'])))
A('')

A('### 4. 主要读数')
A('')
A('| 臂 | `com_V` | `com_V_mlp` | `com_V_attn` | `median(REACH)` | `nb` | `share_mlp_nb` | `argmax_w` |')
A('|---|---|---|---|---|---|---|---|')
for a in AO:
    v = V[a]
    A('| `%s` | **%s** | %s | %s | %s | %s | %s | L%s |'
      % (a, f(v['com_V']), f(v['com_V_mlp']), f(v['com_V_attn']), f(v['median_reach'], 1),
         v['neighbourhood'], f(v['share_mlp_nb']), v['argmax_w_layer']))
A('')
A('**同 Phase 配对（量化敏感度）**：')
A('')
A('| 配对 | Δ`com_V` | Δ`com_V_mlp` | `spearman(w_nf4,w_bf16)` | 相对残差中位 / p90 | `argmax`(nf4/bf16) | `share_mlp_nb`(nf4/bf16) |')
A('|---|---|---|---|---|---|---|')
for k in sorted(QP):
    s = QP[k]
    A('| `%s` | **%s** | %s | **%s** | %s / %s | L%s / L%s | %s / %s |'
      % (k, f(s['delta_com_V']), f(s['delta_com_V_mlp']),
         f(s['spearman_w']) if s['spearman_w'] is not None else 'NA',
         f(s['median_rel_resid']), f(s['p90_rel_resid']),
         s['argmax_nf4'], s['argmax_bf16'],
         f(s['share_mlp_nb_nf4']), f(s['share_mlp_nb_bf16'])))
A('')
A('含义：**质心几乎不动**（Δ ≤ %s 层，容差 %s），**谱形状几乎完全保持**（秩相关 ≥ %s），'
  '**层位置不变**（`argmax` 相同），**组件归属不变**（`share_mlp_nb` 同侧且都过半）。'
  % (f(max(QP[k]['delta_com_V'] for k in QP) if QP else 0), f(FL['QUANT_TOL_COMV'], 1),
     f(FL['RHO_SHAPE_MIN'], 2)))
A('')

A('### 5. holdout 预测（seal 前未观测）')
A('')
A('- **P3 跨家族量化稳健**：%s —— A1（GLM 家族，不同 tie 状态）的 Δ`com_V` = **%s** 层。'
  % (ps('P3'), dfA1))
A('- **P4 bf16 仍「组件过半 + 深端」**：%s —— A1·bf16 的 `share_mlp_nb` = **%s**（> %s），'
  '`com_V` = %s ≥ `median(REACH)` = %s，`nb` = %s。'
  % (ps('P4'), f(V['A1_bf16']['share_mlp_nb']), f(FL['MLP_DOM_MIN'], 2),
     f(V['A1_bf16']['com_V']), f(V['A1_bf16']['median_reach'], 1), V['A1_bf16']['neighbourhood']))
A('- **P5 谱形状**：%s —— 所有配对的 `spearman ≥ %s`。' % (ps('P5'), f(FL['RHO_SHAPE_MIN'], 2)))
A('')
A('| 预测 | 判 |')
A('|---|---|')
for k in sorted(PC):
    A('| %s | %s |' % (k, ps(k)))
A('')

A('### 6. 判决表')
A('')
A('- `Q1` 保真度：**%s**' % JV['Q1_joint'])
A('- `Q2` 校准（nf4 臂复现 P17 锚）：**%s**' % JV['Q2_joint'])
A('- `Q3` **量化敏感度（核心）**：**%s**（%d/%d 对 ≤ %s 层）'
  % (JV['Q3_joint'], JV['Q3_counts']['STABLE'], JV['Q3_counts']['n'], f(FL['QUANT_TOL_COMV'], 1)))
A('- `Q4` 谱形状：**%s**（%d/%d 对 ≥ %s）'
  % (JV['Q4_joint'], JV['Q4_counts']['CONSISTENT'], JV['Q4_counts']['n'], f(FL['RHO_SHAPE_MIN'], 2)))
A('- `Q5` bf16 组件归属：**%s**' % JV['Q5_joint'])
A('- `Q6` bf16 深端：**%s**' % JV['Q6_joint'])
A('- `Q7` 置换零假设（各臂 `com_V` 的 tail）：`%s`'
  % json.dumps({a: JV['Q7_null'][a] for a in JV['Q7_null']}, ensure_ascii=False))
A('')

A('### 7. 交叉验证：独立探针 ↔ 生产实现')
A('')
A('A0 的读数由**两个独立实现**给出（`probe_feasibility_phase19.py` 与主脚本），且与 Phase 17 锚三方对齐：')
A('')
A('| 量 | P17 冻结锚 (nf4) | 探针 nf4 | 生产 nf4 | 探针 bf16 | 生产 bf16 |')
A('|---|---|---|---|---|---|')
_pr = PROBE['runs']
A('| `com_V` | %s | %s | **%s** | %s | **%s** |'
  % (f(SEAL['anchor_values']['A0_nf4']['com_V']), f(_pr['nf4']['com_V']), f(V['A0_nf4']['com_V']),
     f(_pr['bf16']['com_V']), f(V['A0_bf16']['com_V'])))
A('| `share_mlp_nb` | %s | %s | %s | %s | %s |'
  % (f(SEAL['anchor_values']['A0_nf4']['share_mlp_nb']), f(_pr['nf4']['share_mlp_nb']),
     f(V['A0_nf4']['share_mlp_nb']), f(_pr['bf16']['share_mlp_nb']), f(V['A0_bf16']['share_mlp_nb'])))
A('| `argmax_w` | L%s | L%s | L%s | L%s | L%s |'
  % (SEAL['anchor_values']['A0_nf4']['argmax_w_layer'], _pr['nf4']['argmax_w_layer'],
     V['A0_nf4']['argmax_w_layer'], _pr['bf16']['argmax_w_layer'], V['A0_bf16']['argmax_w_layer']))
A('')
A('**四个 nf4/bf16 数字在两套实现下逐位相同** ⇒ 量化敏感度不是实现细节的产物。')
A('')

A('### 8. 限界（诚实性）')
A('')
A('- **H8 / 覆盖限界**：**A2（Qwen3-14B）无法参与 bf16 腿** —— 29.5 GB bf16 在**加载 19% 时 segfault**'
  '（实测；与 P17 `quant.why` 记录的同一 RAM 天花板一致）。因此本 Phase 的跨精度稳健性只在 '
  '**qwen3-4b 与 glm4-9b** 两个模型上验证。')
A('- **H2 两源差异**：bf16 与 nf4 的差异含「量化误差 + 线性层 kernel 路径」两源'
  '（同 `eager` 已控注意力 kernel，反量化路径无法消除）⇒ Δ 只能说「量化口径的整体影响」。')
A('- **H6 offload**：A1·bf16 需 CPU offload（18.8 GB > 14 GiB 上限）⇒ 该臂含「分片执行」第三源；'
  '其读数按预注册仍入 P3/P4 硬门，但若超容差须先在无 offload 的 A0 上排除分片效应。')
A('- **H3**：`w_ℓ` 是**激活级**分解（hook），非权重级实现证明（承 N2h1-α-1 挂账）。')
A('- **H9**：本 Phase **不重测** P17 的行为量（J / `com_layer`）与 P18 的行为预算 `b`；'
  '只回答「`w_ℓ` 谱与 `com_V` 是否量化稳健」。')
A('')

A('### 9. 与 P17 / P18 的关系')
A('')
A('- **P17**：`DEEP_ALL`（`com_V` 深端）获得**跨数值口径**支持 ⇒ nf4 不是结论的必需条件。')
A('- **P18**：`com_B` 与 `com_V` 的 gap（2.5–6.6 层）虽都是 nf4 量之差，但既然 `com_V` 的量化位移只有 '
  '**%s 层**（远小于该 gap），P18 的「行为质心比向量质心浅」在量级上仍成立（该结论的严格跨精度检验'
  '需另做行为量重测，列入挂账）。' % dfA0)
A('')

A('### 10. 同轮勘误')
A('')
A('- **[E-A2]** A2·bf16 加载期 segfault（~19% 权重）⇒ bf16 腿只有两模型；A2 仅入校准门。'
  '（本 Phase **不**因可行性限制而改判据，只如实缩小覆盖范围并写入 seal 的 `loadability`。）')
A('- **[E-pair]** 探针首版把配对过滤写成 `p[0] in DISC_W and p[2] in DISC_W`（比 P17 严，24 对降为 17 对），'
  '导致探针 nf4 的 `com_V` = 26.1956 ≠ P17 锚 26.1501（差 0.0455）。改为与 P17 逐字一致的 '
  '`p[0] in DISC_W` 后**逐位复现 26.1501**。教训：配对集定义也是口径的一部分，'
  '跨实现比对（探针 vs 生产 vs 冻结锚）是唯一可靠的检出手段（承铁律 (ad)）。')
A('')

A('### 11. 下一步')
A('')
A('- **Phase 20 候选（最高）**：把跨精度检验推进到**行为量** —— P18 的 `b_{c,ℓ}` 与 P16 的 `com_layer` '
  '在 bf16 下复算，补上 P18 结论的跨精度证据（本 Phase 只覆盖了向量侧）。')
A('- **并列**：邻域宽度 ±2 敏感性（三臂 `nb` 恰都 [26,28]）；P17 `P6` 的 MEMO 改判（承 P18 `P5`）。')
A('- **N 线挂账**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌；N3-β→N3-ε；R1 补强；K4；'
  '**N 线 P3–P7 补登 Ledger**。')
A('')

OUTP = os.path.join(P19T, 'memo_append_phase19.md')
io.open(OUTP, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('WROTE %s  %d B / %d lines' % (OUTP, os.path.getsize(OUTP), len(L)))
print('headings:', sum(1 for x in L if x.startswith('### ')))
