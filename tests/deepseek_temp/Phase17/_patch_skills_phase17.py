# -*- coding: utf-8 -*-
"""同步两个技能到 Phase 17。"""
import io

MAIN = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
CLOSE = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'

# ============================ main-axis
s = io.open(MAIN, encoding='utf-8').read()

o = "## 4 已实测的坑（56 条，逐条对应数值）"
n = "## 4 已实测的坑（57 条，逐条对应数值）"
assert s.count(o) == 1
s = s.replace(o, n)

ANCH = "\n## 5 代码骨架要点\n"
assert s.count(ANCH) == 1, s.count(ANCH)
PIT57 = """
57. **「口径歧义」不会自己暴露 —— 必须用**第二份独立实现**逐位对齐来捕捉；且「臂内已算好的量」MERGE 不会重算 ⇒ 改口径必须重跑臂（Phase 17 实证：E4 = 24.466 → 26.150）**：Phase 17 把 Phase 8 的单层向量预算推广到逐层（`w_ℓ = mean_pairs‖P_{U_ℓ}(Δ_inc,ℓ)‖`），质心 `com_V` 的 seal 定义是**区间求和** `W_j = Σ_{ℓ∈[s_j,s_{j+1})} w_ℓ`。首版生产实现却取了 `w_{s_j}`（**位点单层**）—— 两种口径**都能跑通、都给出 20–27 层的「合理」质心**，单看任何一个都不会发现错。
    - **检出唯一手段 = 第二份独立实现**：可行性探针（独立脚本）算得 A0 `com_V = 26.15`，生产实现给出 `24.466`（差 **1.68 层**）⇒ 立刻定位口径不一致。改成区间求和后两者**逐位相同**（`26.150056332` vs `26.150056332`）。**纪律**：凡 seal 给了公式，实现必须**逐字照抄**（「区间求和」不能念成「位点取值」）；交付前用**独立实现**（探针 / `disk_verify_*`）逐位对齐 —— 「数值合理」不足为凭。
    - **MERGE 不重算臂内量**：`com_V` 在**臂记录**里就算好了，`MERGE=1` 只读 partial 组装 ⇒ 只重跑 MERGE 会**沿用旧值**（本轮首修只重跑 MERGE，A0 仍是 24.466，直到**重跑三臂**才变 26.150）。**纪律**：改口径 / 改统计量 ⇒ **必须重跑臂**；报告要能区分「重跑臂」与「只重跑 MERGE」。
    - **独立复核从原始量重算**：`disk_verify_phase17.py`（A–L，**178 项 0 FAIL**）从**臂记录的 `w_all`/`reach`** 用**区间求和**重算 `com_V` 族、`median`、`share_*`、`spearman`，**同 seed 重跑置换 null**，从 P16 锚的 `xhalf`/`J` 序列重算 `span_k`/`com_layer` 与 Q7 双读数；**不从 verdict 里读 `Q3_com_V` 自证**。
    - **数据驱动渲染须覆盖「元数据 + 散文」两处**：Ledger 的 `rev_note` 与 MEMO 的 §1/§5/§6/§10 都出现过**手工转录的旧数字**（Ledger 的 P3/P4/P5 三臂数、MEMO §6「向量质心（24.5）只差 1.46 层」而表格已 26.150/3.14）⇒ 两处都要**由 result 现场渲染**并**逐句对照数据**；**修正前的历史值从留痕文件读**，不写死。
    - **附属范式（值得复用）**：「可加性**由构造成立** + 另设**保真度门**」—— 把 `Δ_inc := Δ_attn + Δ_mlp`、`Δ_attn := Σ_h Δ_head_h` 写成**定义**（精确可加），再用 arch 恒等式 `‖(h_{ℓ+1}−h_ℓ)−(attn_out_ℓ+m_ℓ)‖/‖h_{ℓ+1}−h_ℓ‖` 与分块可加性 `‖Σ_h head_block − o_proj(v)‖/‖o_proj(v)‖` 检验**实现与模型计算一致**（nf4 实测地板 1.62e-2 / 3.59e-3）。它把「分解对不对」变成**可测量的门**，而不是假设。
"""
s = s.replace(ANCH, PIT57 + ANCH)
io.open(MAIN, 'w', encoding='utf-8', newline='\n').write(s)
t = io.open(MAIN, encoding='utf-8').read()
assert '57. **「口径歧义」不会自己暴露' in t and '（57 条，逐条对应数值）' in t
print('main-axis OK: now %d chars' % len(t))

# ============================ closeout
s = io.open(CLOSE, encoding='utf-8').read()

o = "## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 16，2026-10-01/02）"
n = "## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 17，2026-10-01/02）"
if s.count(o) == 1:
    s = s.replace(o, n)
else:
    print('warn: closeout header pattern count=%d' % s.count(o))

ANCH2 = "\n## 参照实现（Phase 3125"
assert s.count(ANCH2) == 1, s.count(ANCH2)
L28 = """
28. **改口径只重跑 MERGE = 沿用旧值 ⇒ 必须重跑臂；`already` 分支必须「幂等刷新」而非「跳过」；数据驱动渲染要覆盖 Ledger `rev_note` 与 MEMO 散文两处（Phase 17 实证）**：
    - **(a) 口径歧义靠「独立实现逐位对齐」检出**：Phase 17 的 `com_V` seal 定义是**区间求和** `W_j = Σ_{ℓ∈[s_j,s_{j+1})} w_ℓ`，首版实现取了 `w_{s_j}`（**位点单层**）—— 两种口径都能跑通、都给出「合理」质心。探针（独立脚本）A0 `26.15` vs 生产 `24.466`（差 1.68 层）⇒ 立刻定位；修正后**逐位相同**（`26.150056332`）。**纪律**：seal 给了公式 ⇒ 交付前必须**独立实现逐位对齐**。
    - **(b) MERGE 不重算臂内量**：`com_V` 在**臂记录**里算好，`MERGE=1` 只组装 partial ⇒ 只重跑 MERGE 会**沿用旧值**（首修时 A0 仍 24.466，直到**重跑三臂**）。**纪律**：改口径/统计量 ⇒ **必须重跑臂**。
    - **(c) `already` 分支要「就地刷新」**：`closeout_*` 的幂等分支不能只「跳过」—— result 重生成后要**就地更新** `result_sha8 / rev_note / verdict / n_rows / n_forwards_per_arm` 并**重算 `ledger_sha256_8`**（Phase 17 首版仅跳过 ⇒ Ledger 的 `result_sha8` 停留**前一版**快照，与最终 `ee27627a` 不一致）。
    - **(d) 渲染数据驱动要覆盖两处**：Ledger `rev_note` 与 MEMO §1/§5/§6/§10 都出现过**手工转录旧数字** ⇒ 两处都要**由 result 现场渲染**；修正前的历史值**从留痕文件读**（不写死）。
    - **(e) 锚点预检的「假报警」其实是它在工作**：`do_append_*` 报缺 `qwen3-14b` —— 实为正文写作 `Qwen3-14B` ⇒ 改锚点，**不要删锚点**。
    - **(f) MEMO 收尾需含「下一步（死线）」**：本轮渲染器首版漏了该节（项目要求每 Phase MEMO 记录接续）⇒ 补 §11 并由 result 渲染候选与挂账。
"""
s = s.replace(ANCH2, L28 + ANCH2)

REF_OLD = "- N 线独立复核（Phase 12）：`tests/deepseek/Phase12/disk_verify_phase12.py`（从冻结 `result_phase12.json` 确定性重算 `xhalf`/`J_swap`/`recover`、**同 seed 复现 bootstrap 带与置换 null**、G 族布尔、F11/F12/F13/F10）。"
REF_NEW = REF_OLD + "\n- N 线独立复核（Phase 17）：`tests/deepseek/Phase17/disk_verify_phase17.py`（**A–L 分区，178 项 0 FAIL**；从**臂记录** `w_all`/`reach` 用**区间求和**重算 `com_V` 族/`median`/`share_*`/`spearman`、**同 seed 重跑置换 null**、从 P16 锚序列重算 `span_k`/`com_layer`、Q7 双读数、**探针 vs 生产跨实现逐位对齐**、MEMO 前缀锚与 `_infra` 基线链）。"
if s.count(REF_OLD) == 1:
    s = s.replace(REF_OLD, REF_NEW)
else:
    print('warn: closeout ref Phase12 count=%d' % s.count(REF_OLD))

io.open(CLOSE, 'w', encoding='utf-8', newline='\n').write(s)
t = io.open(CLOSE, encoding='utf-8').read()
assert '28. **改口径只重跑 MERGE' in t and 'Phase 8 → Phase 17' in t and 'disk_verify_phase17.py' in t
print('closeout OK: now %d chars' % len(t))
