# -*- coding: utf-8 -*-
"""技能同步（Phase 19）：
  rdc-main-axis-probe: 新增坑 59（量化口径稳健性 + offload meta 占位 + 配对集定义也是口径）；计数 56/57 -> 59。
  rdc-phase-closeout : 新增教训 30；N 线实证表头 Phase 17 -> Phase 19；跑通次数 七 -> 十二。
"""
import io
import os

SK = r'C:\Users\Admin\.workbuddy\skills'
MA = os.path.join(SK, 'rdc-main-axis-probe', 'SKILL.md')
CO = os.path.join(SK, 'rdc-phase-closeout', 'SKILL.md')

# ---------------- rdc-main-axis-probe ----------------
s = io.open(MA, encoding='utf-8').read()


def rep(old, new, n=1, text=None):
    global s
    assert text.count(old) == n, 'count=%d for %r' % (text.count(old), old[:70])
    s = text.replace(old, new)


PIT59 = """59. **换数值精度（量化口径）= 引入一个新自由度 ⇒ 必须预注册「同模型 × 原口径」校准臂；另两个装置坑：CPU-offload 的权重以 **meta 占位**承载，且**配对集定义本身也是口径**（Phase 19 nf4↔bf16 实证）**：Phase 17 的 `w_ℓ = mean_pairs ‖P_{U_ℓ}(Δ_inc,ℓ)‖` 与质心 `com_V` 全在 bitsandbytes **nf4** 下测得 —— 「深端集中」可能是**量化地板效应**。Phase 19 把同一模型、同一尺度（同 template/classes/instances/pairs/`U_ℓ`=类别质心差 SVD 秩 `n_classes−1`/区间求和质心/REACH 域）换成 **bf16** 复算，**除精度外逐项冻结**。
    - **(a) 校准臂不可省**：bf16 臂**按设计不复现** nf4 锚 ⇒ 必须另设 **nf4 校准臂**（逐位复现上一 Phase 冻结锚：`com_V`/`com_V_mlp`/`com_V_attn` 差 ≤1e-3，`nb`/`argmax_w` 精确相同）来证明「装置与上一 Phase 同源」；否则 bf16 与锚的任何差异都分不清是**口径**还是**装置漂移**。
    - **(b) 结果**：Δ`com_V` = **0.0930**（qwen3-4b）/ **0.0696**（glm4-9b）层（容差 2.0）；`spearman(w_nf4,w_bf16)` = **0.9924 / 0.9988**；`argmax_w` 层不变（L30/L38）；邻域 `share_mlp_nb` 同侧且都过半（0.740→0.774 / 0.975→0.976）⇒ **「深端集中」与「MLP 主导」都不是量化地板效应**。
    - **(c) 装置坑 1（offload 的 meta 占位）**：`device_map="auto"` + `max_memory` 超限时，accelerate 把部分权重放在 **meta 占位参数**上（真设备在 `m._hf_hook.execution_device`）。凡自己 schedule 输入张量的探针（如 fidelity 重算 `OPR[l](x)`）若按 `next(p.device)` 建输入 ⇒ `NotImplementedError: Cannot copy out of meta tensor; no data!`。**对策**：`mod_dev(m)` 优先返回 `getattr(m,'_hf_hook',None).execution_device`；**改实现后必须重跑全部臂**，保持臂间同一实现版本。
    - **(d) 装置坑 2（配对集定义也是口径）**：探针首版把配对过滤写成 `p[0] in DISC_W and p[2] in DISC_W`（比上一 Phase 严）⇒ 24 对降 17 对，`com_V` = 26.1956 ≠ 锚 26.1501（差 0.0455）；改回与上一 Phase 逐字一致的 `p[0] in DISC_W` 后**逐位复现 26.1501**。**纪律**：跨实现/跨 Phase 复现时，**配对集、支撑域、位点约定**都属于口径，必须逐字一致；唯一可靠检出手段是**探针 vs 生产 vs 冻结锚三方逐位比对**（承坑 57/58）。
    - **(e) 可行性限界如实记录，不因可行性改判据**：Qwen3-14B（29.5 GB）bf16 **加载 19% 时 segfault** ⇒ bf16 腿只有两模型；A2 仅入校准门。覆盖限界写入 seal 的 `loadability` 与 honesty（H8）。
    - **(f) 与 `rdc-phase-closeout` 教训 30 配套**（臂集设计 / `loadability` 记录 / 同 Phase 配对 / 独立复核的 `interval_centroid`+`perm_null` 重写）。

"""

rep('## 5 代码骨架要点\n\n- 层访问要兼容多模态包装',
    PIT59 + '## 5 代码骨架要点\n\n- 层访问要兼容多模态包装', text=s)
rep('## 4 已实测的坑（57 条，逐条对应数值）', '## 4 已实测的坑（59 条，逐条对应数值）', text=s)
rep('以及 56 条已实测的坑。', '以及 59 条已实测的坑。', text=s)
io.open(MA, 'w', encoding='utf-8', newline='\n').write(s)
s2 = io.open(MA, encoding='utf-8').read()
assert '59. **换数值精度' in s2 and '（59 条，逐条对应数值）' in s2 and '59 条已实测的坑' in s2
print('MAIN-AXIS OK  bytes=%d' % len(s2.encode('utf-8')))

# ---------------- rdc-phase-closeout ----------------
t = io.open(CO, encoding='utf-8').read()


def rep2(old, new, n=1):
    global t
    assert t.count(old) == n, 'count=%d for %r' % (t.count(old), old[:70])
    t = t.replace(old, new)


LES30 = """30. **换数值精度（量化口径）的 Phase：必须预注册「同模型 × 原口径」校准臂 + 把「可行性限界」只做记录；CPU-offload 用 meta 占位（Phase 19 实证，收尾链第十二次）**：
    - **(a) 校准臂不可省**：检验臂（bf16）**不应**复现原口径（nf4）锚 ⇒ 若只跑检验臂，任何差异都无法归因（**口径** vs **装置漂移**）。臂集模板 = 2 校准（原口径，逐位复现上一 Phase 冻结锚）+ 2 检验（新口径，其一为 holdout）；联合门 `CALIB_ALL_OK`。
    - **(b) 「可行性」只记录、不改判据**：`loadability`（A2/Qwen3-14B 29.5 GB bf16 **segfault@19%**）写进 seal 与 honesty（H8），**不进 verdict**；覆盖限界如实缩小（「只在两模型上验证」）。
    - **(c) offload 臂的输入设备**：`mod_dev(m)` 必须优先 `getattr(m,'_hf_hook',None).execution_device`，否则在 meta 张量上建输入 → `Cannot copy out of meta tensor`；**改实现后重跑全部臂**（否则臂间实现版本不一致，MERGE 出来的 result 混版）。
    - **(d) 用「同 Phase 配对」而非「跨 Phase 比对」**：量化敏感度取**同一 result 内** nf4/bf16 两臂配对（Δ`com_V` / `spearman(w)` / 相对残差中位·p90 / `argmax` 同层 / `share` 同侧），避免跨 Phase 基线漂移混入。
    - **(e) 独立复核的「重算→导出标签」形态仍然成立，但作用域断言要写对**：`disk_verify_*` 另写一份 `interval_centroid`（区间求和+中点）/ `perm_null`（同 seed 重放 RNG）/ 平均秩 `spearman`，逐条与落盘比对（本轮 **108 PASS / 0 FAIL**）。**本轮新增的复核坑**：Ledger 混有 **N 线 / G 线 / 无 `phase`** 三类条目（302 条里只 198 个唯一 `phase`）⇒ 「phase 唯一」断言必须**限定作用域**（N 线 = `name.startswith('n2h1a')`，应为 8..19 共 12 条），否则产出假 FAIL。

"""

rep2('## 参照实现（Phase 3125', LES30 + '## 参照实现（Phase 3125')
rep2('## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 17，2026-10-01/02）',
     '## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 19，2026-10-01/02）')
rep2('这条线的收尾链已跑通七次（Phase 8/9/10/11/12/13/14）',
     '这条线的收尾链已跑通十二次（Phase 8–19）')
io.open(CO, 'w', encoding='utf-8', newline='\n').write(t)
t2 = io.open(CO, encoding='utf-8').read()
assert '30. **换数值精度' in t2 and 'Phase 8 → Phase 19' in t2 and '已跑通十二次' in t2
print('CLOSEOUT OK  bytes=%d' % len(t2.encode('utf-8')))
