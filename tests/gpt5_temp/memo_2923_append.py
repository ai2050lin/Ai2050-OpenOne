# -*- coding: utf-8 -*-
"""memo_2923_append.py -- append Phase 2923 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T12:36:51 -> [2026-09-19 12:36]).
Idempotent: refuses if a 2923 title already exists.
"""
import hashlib
import json
import os
import re

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
LEDGER = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas"
          r"\atlas_ledger.json")
ROADMAP = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs"
           r"\lpf_multiaxis_gating_roadmap_v1.md")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2923_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2923_polarity_sign_anatomy" in m_ids, \
    "run ledger_2923_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2923_polarity_sign_anatomy" in l14["connects"]
assert len(L["measurements"]) == 62 and len(l14["connects"]) == 30
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2923:" not in txt, "already appended"

SEC = """
## Phase 2923: 极性符号结构解剖——polarity_sign_layer_structured [2026-09-19 12:36]

### 目的（2922 接续候选 A / 纲领 afa60f18 §4 2923 行）
2922 发现 13 个属性事件全部极性对齐但驱动符号分裂（8 LOW / 5 HIGH）。本 Phase 零前向检验驱动符号的可预测性：H1 词构成说（top-|r| 词极性组成预测符号，主检验预注册）、H2 层位说（quasi-post-hoc）、H3 事件特异。zero-forward（1.0 s）。

### 设计（冻结口径）
- 数据：2921 npz（B/labels）+ 2922 result/npz 作登记参照（d_pole 13 值、4 条 size 联动边）。
- P1 主检验（判据在运行前冻结）：f_k = top-k |r| 词的极性均值（+1 HIGH / -1 LOW），k=5 主 / k=10 副；T_obs = #{sign(f_k) == sign(d_pole)}（f_k=0 计不一致）；null = 2000 次全局保组大小 pole 置换（fresh default_rng(2898)，置换下 d_pole 与 f_k 同重算、top-|r| 词集不变）；判据 p_perm <= 0.05 且 T_obs >= 10（k=5）。
- P2 层位（quasi-post-hoc——2922 输出已并排显示 d_pole 与 peak，方向预期已知，只作描述性权重）：符号 x peak 中位分割 Fisher 精确（单尾，LOW 偏浅）+ 点二列。
- P3 分解（描述）：m_hi/m_lo 原始均值符号组合——contrast 型（反号）vs magnitude 型（同号）。
- P4 同头符号对（quasi-post-hoc）；P5 联动边符号同质（2922 的 4 条显著边，精确二项，quasi-post-hoc）。
- 判决映射：polarity_sign_topword_predicted / polarity_sign_layer_structured / polarity_sign_event_specific。

### 锚（3/3，全过）
- a1：2921 npz sign_M 重算（2922 a1 verbatim）median ~6e-10、max 0。
- a2：13 事件 d_pole 重算（2922 公式 verbatim）vs 2922 result P1 全部 |diff| < 1e-3。
- a3：2922 npz 边 (21,7)-(18,7) rho 0.6764 / p 0.029485 精确 + sign_M_ref == 2921 npz sign_M。

### 结果
- **判决：polarity_sign_layer_structured**。
- P1 主检验**否定**：T5 = 7/13（p_perm 0.4818，null p50 6 / max 11；T10 = 5/13，p 0.6012）——top 词极性组成不能预测驱动符号，且**多事件符号反向**：(22,2) d=-1.20 但 top5 全 HIGH（f5=+1.0）、(8,15) d=-1.60 但 f5=+0.6、(14,12) d=-0.87 但 f5=+0.6、(18,16) d=+1.36 但 f5=-0.2。响应最强的词是谁与格子往哪端偏是两回事。
- P2 层位结构：Fisher {low_shallow 7, low_deep 1, high_shallow 0, high_deep 5}，**p = 0.0047**、点二列 r = 0.623——**LOW 驱动集中浅层（7/8），HIGH 驱动全部深层（5/5，peak 16-23）**。
- P3 分解：**13/13 全部 contrast 型**（m_hi 与 m_lo 严格反号，零 magnitude 型）——每个属性事件格是双极对比检测器（hi/lo 极间均值反号），不是单边检测器。典型对称对 (18,16)：m_hi +0.0091 / m_lo -0.0091。
- P4：h11 同头 size 内反号 ((11,23) +0.84 / (11,27) -0.79，且同 peak 层 23——头内符号翻转存在)；h18 跨类别反号 (+1.36 / -1.44)。
- P5：4 条显著联动边 **4/4 符号同质**（精确二项 p = 0.0625，quasi-post-hoc）——联动发生在同号事件之间。

### 解读
1. **驱动符号由层深组织，不由词构成组织**：浅层格对比方向偏 LOW（lo 端响应高），深层格偏 HIGH——与 2922 的层深分离（attr peak L15 vs lang L6）合成"深度推进的极性编码"图景：同一 pole 维度的读出对比方向随深度翻转/演进。
2. **双极对比检测器**：13/13 contrast 型——事件格对 pole 两端均值做反号对比（类似差分读出），2919 dirs = unit(mean HIGH - mean LOW) 的锚侧约定与格侧读出符号无必然一致——**格侧读出符号是层依赖的自由参数，锚方向不外推到格**。
3. 反向案例的方法论意义：(22,2) 格最强响应词全是 HIGH 词但格子 LOW 驱动——"谁响应强"（幅度结构）与"往哪端偏"（均值对比方向）是独立自由度；P1 否定说明词级幅度结构不决定极性读出方向。
4. 联动同号（P5 4/4）+ 同头反号（P4 h11）——同头不同层可实现相反读出方向，size 4 事件分量是同向（全 LOW 驱动）协作通道。
5. 理论对应：Cmp(o,r,v) 候选竞争图景下，属性维度的"比较方向"（哪端为正）在层间演进——支持"读出方向是路由性质而非内容性质"。

### 方法论常数（新增）
- **quasi-post-hoc 标注制度**：上游 Phase 输出已并排展示过的量做下游检验时必须标注（P2/P4/P5），判决权重只给真预注册检验（P1）。
- **P5 显著边口径**：联动边符号同质性只用 maxT 显著边（4 条），不得混入全 36 对（run1 实现偏差即此，已修复重跑）。
- 双极 contrast 检验（m_hi/m_lo 反号判定）入事件解剖标配。

### 执行史
- run1 通过但 P5 实现偏差（预注册 4 条显著边，代码算了全部 36 对 size 边 -> "16/4" 格式错乱即根因）-> 修复为 res22 P2 edges 口径；删产物重跑 run2 通过（1.0 s）。另清理死函数 t_count（无效语法残留，未执行到）。
- 零前向：无模型加载，锚从 npz + 上游 result 自洽复算。

### 硬伤
- P2/P4/P5 为 quasi-post-hoc（2922 输出已含 d_pole 与 peak 并排）——判决链依赖 P2 的 p=0.0047 但其权重按预注册制度降为描述性；结论"层位结构"需 2924 独立样本复验。
- n=13 事件、3 轴混池——层位结构的轴内功效未单独评估。
- P2 用 margin peak 层作层位代理；d_pole 所在层 l 与 peak 层不同（如 (11,27) 事件层 27、peak 23）——用事件层 l 重算未做（2924 候选）。
- h11 头内反号（同 peak 层）是层位说的个体反例（已如实登记）。
- n=1 run（npz 自洽锚缓解）；单模型。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2923_polarity_sign_anatomy.py: 31dff1a8
- execution.json: 91a00fc0（created 2026-09-19T12:36:51）
- result.json: 87f9b769（final_verdict=polarity_sign_layer_structured，runtime 1.0 s）
- polarity_sign_anatomy.npz: 6713376e（d_pole13/f5_13/f10_13/t5_perm/t10_perm/m_hi13/m_lo13/event_ids/peak_layers）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2923/polarity_sign_anatomy/
- Ledger：M2923_polarity_sign_anatomy 入账，measurements 61->62，L14 connects 29->30，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2923 行完成）

### 接续（2924 候选）
- A（主选）：**深度分级极性编码形式化**——d_pole vs 事件层 l（非 peak）回归 + 全格 per-layer 极性符号普查（sign_M 显著格之外的 m_hi/m_lo 对比方向层剖面），零前向；检验"浅 LOW / 深 HIGH"是否为全格级连续梯度。
- B：**探针相对性检验**——lang 词表换词级 lang 探针（dirs_word）重测事件集（一次前向 ~1 min）。
- C（零前向）：h4 L1<->L19 复用子空间主角度分析（2918 唯一真复用通道，roadmap 遗留项）。
- D：**contrast 不对称解剖**——|m_hi|/|m_lo| 比值结构（13 事件对比强度不对称性 vs 层位/轴）。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2923: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2923 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2923")
