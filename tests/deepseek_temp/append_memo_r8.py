# -*- coding: utf-8 -*-
"""R8：把 A 闸门关闭记录作为 Phase 35 追加进唯一研究日志（append-only，CRLF+BOM）。"""
import os, json, hashlib, datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MEMO = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
CLOSE= os.path.join(ROOT, r"research\deepseek\atlas\a_gate_closure_v1.json")
DL   = os.path.join(ROOT, r"tests\deepseek\result\deadline_dual_track_v1.json")
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_r8_memo_report.txt")

EXPECT_PREFIX_SHA8 = "b257f54f"
NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

cl = json.loads(open(CLOSE, "rb").read().decode("utf-8-sig"))
dl = json.loads(open(DL, "rb").read().decode("utf-8-sig"))
k1 = dl["k1_recompute"]; per = k1["per_model"]; ml = k1["model_level"]; agg = k1["aggregate"]
cv = cl["k1_reverdict"]

MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]

L = []
def A(s): L.append(s)

A("## Phase 35: A 闸门 seal 执行与关闭 —— Q08=甲（K1 改判触发）、C1–C6 全接受、I1/I9 冻结确认（R8；治理，非实验）[%s]" % NOW)
A("")
A("**seal 原文（逐字）**：`seal: Q08=甲 | C=全接受 | I1=确认 | I9=确认`")
A("")
A("### 0 一句话")
A("把 A 闸门（Q01 / Q02 / Q08 / Q09 / Q12）从「待 seal」推到「**已 seal 并关闭**」；其中唯一影响主线存续的 Q08 判为**甲**：K1 的判定层取**行为读出层**，K1 因此**触发**，命题「条件齿轮组 = 算子代数」由 mechanism 降级为 descriptive。")
A("")
A("### 1 Q08=甲：K1 判定层 = 行为读出层 ⇒ K1 触发")
A("")
A("同一把 5% 门，挂在不同层位读数完全不同——这正是判据自指（用假设自选的层去检验该假设）必须被消掉的地方：")
A("")
A("| 模型 | k\\* | 读出层 | B4 误差@k\\* | B4 误差@读出 | k\\* 层 margin（负=候选优） | 过门 | 读出层 margin | 过门 |")
A("|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    p = per[m]; l = ml[m]
    def pas(b): return "是" if b else "否"
    A("| `%s` | %d | %d | %.7f | %.7f | %+.6f（MDE %.6f） | %s | %+.6f（MDE %.6f） | %s |" % (
        m, p["kstar"], p["readout"], p["b4_kstar"], p["b4_readout"],
        l["kstar_margin_model"], l["kstar_mde_model"], pas(l["kstar_pass"]),
        l["readout_margin_model"], l["readout_mde_model"], pas(l["readout_pass"])))
A("")
A("**聚合（现场从 `deadline_dual_track_v1.json` 渲染）**：")
A("")
A("- 读出层：池化 margin = **%+.6f**（2×MDE = %.6f）⇒ 未达负向门 ⇒ **3/3 模型否决**；`E_read` 池化 = **%.6f**（5%% 门的 **%.1f×**）；逐模型 `E_read` = %s。"
  % (agg["readout_pooled_margin"], 2 * agg["readout_pooled_mde"], agg["E_read_pooled"],
     agg["E_read_pooled"] / 0.05, " / ".join("%.4f" % e for e in agg["E_read_per_model"])))
A("- k\\* 层：池化 margin = **%+.6f**（2×MDE = %.6f）⇒ 达到负向门（轨 A 不触发）；但 `qwen3-4b` 单模型不达标 ⇒ **轨 B 否决**。"
  % (agg["kstar_pooled_margin"], 2 * agg["kstar_pooled_mde"]))
A("- 轨 A（读出层）触发 = %s；轨 B 否决模型（k\\*）= %s；读出层否决模型 = %s。"
  % (k1["verdict"]["readout_trackA_fired"], k1["verdict"]["kstar_veto_models"], k1["verdict"]["readout_veto_models"]))
A("")
A("**判决**：判定层 = 行为读出层 ⇒ `K1 = fired_all_models`（3/3 否决 + 轨 A 触发）；k\\* 层 = `model_specific`（`qwen3-4b` 否决，不得升机制）。")
A("")
A("**后果（重复三次，防走样）**：")
A("")
A("1. 命题「条件齿轮组 = 算子代数」**降级为 descriptive**——只能作「功能性端口类描述」。")
A("2. 命题「条件齿轮组 = 算子代数」**降级为 descriptive**——不得再作机制主张（mechanism claim）。")
A("3. 命题「条件齿轮组 = 算子代数」**降级为 descriptive**——其 mechanism 字样的任何后续引用到此为止。")
A("")
A("### 2 C1–C6：全接受，逐条落盘（erratum，不改写归档正文）")
A("")
A("| id | 目标 | 原值 / 原文 | 更正为 | 本线执行方式 | 状态 |")
A("|---|---|---|---|---|---|")
_for = {
 "C1": "本线记录（count_mode=component）。原正文在 G 线备忘录，跨线不改。",
 "C2": "本线记录并采用 canonical=41.5%。",
 "C3": "TESTPLAN 已作归档全文并入本备忘录（Phase 23，逐字保留）；更正在此声明。",
 "C4": "目标为跨线共享账本 ⇒ **未施加**；补丁规格见 `ledger_corrections_v1.json`。",
 "C5": "以外部 manifest 记录真值（哈希自指失效 ⇒ 不写入被哈希文件本体）。",
 "C6": "目标为跨线共享账本 ⇒ **未施加**；补丁规格见 `ledger_corrections_v1.json`。",
}
for c in cl["corrections_C1_C6"]:
    rv = c["resolved_value"]
    if c["id"] == "C1":
        cur, new = "未标口径", "`count_mode=component`（条数 %d、分量和 %d；A5/B34/C14/D21/E20）" % (rv["n_atomic"], rv["sum_components"])
    elif c["id"] == "C2":
        cur, new = "约 55%（量纲混淆：分量数÷条数）", "canonical `component` **%.1f%%**(39/94)；备选 `atomic_best` %.1f%%(39/62)" % (rv["canonical_pct"], rv["alternate_pct"])
    elif c["id"] == "C3":
        cur, new = "`A=5 / B=6 / C=10 / D=21 / E=20`", "`A=5 / B=34 / C=14 / D=21 / E=20`（component，条数 62）"
    elif c["id"] == "C4":
        cur, new = "`%s`" % rv["recorded"], "`%s`（content_excluding_self；本对话独立重算复现）" % rv["correct_self_excluding"]
    elif c["id"] == "C5":
        cur, new = "`%s`" % rv["recorded_in_memo3103"], "`%s`" % rv["actual"]
    else:
        cur, new = "两套并存（270 / 199 / 兼有 165）；evidence_level 常量", "统一含 `meas_id`+`phase`；`evidence_level` 三值枚举"
    st = "已接受" if c["status"] == "accepted" else "已接受·待跨线施加"
    A("| **%s** | %s | %s | %s | %s | %s |" % (c["id"], c["target"], cur, new, _for[c["id"]], st))
A("")
A("**C5 目标可达性核实**：`proposition_ledger.json` 实际 sha8 = `%s`（本对话现场复核），与更正值一致。"
  % cl["corrections_C1_C6"][4]["resolved_value"]["verified_here"])
A("")
A("### 3 K1 改判登记 + 跨线账本补丁（未施加，跨线纪律）")
A("")
A("**账本归属取证**（本轮实测）：`research/gpt5/atlas/atlas_ledger.json`（`%s`，n=%d）**同时**含 deepseek 线 Phase 8–21 与 G 线 Phase 2902–3153 的测量 ⇒ 属**跨线共享**，本对话规范要求「避免与其他路线混合」⇒ **本轮不改**。"
  % (json.loads(open(os.path.join(ROOT, r"research\deepseek\atlas\a_gate_closure_v1.json"), "rb").read().decode("utf-8-sig"))["protected_fingerprints_unchanged"]["atlas_ledger.json"],
     cl["corrections_C1_C6"][5]["resolved_value"]["n_total"]))
A("")
A("- **C4/C6 补丁** 落 `research/deepseek/atlas/ledger_corrections_v1.json`（状态 `SEALED_BUT_NOT_APPLIED`，含目标 sha256 前值 + 4 条 op + 已复现的正确值）⇒ 施加需用户一句确认或交账本所有者。")
A("- **K1 改判登记**（`phase: A-gate`，`verdict: k1_fired_at_readout__kpecific__k2_k3_not_measurable__A_gate_closed`）同样只落本线，跨线账本追加在补丁 spec 内。")
A("- 本线权威登记 = 本 Phase 35 + `research/deepseek/atlas/a_gate_closure_v1.json`。")
A("")
A("### 4 A 闸门关闭宣告")
A("")
A("| 队列项 | 标题 | 决定 | 产物 |")
A("|---|---|---|---|")
for qid, v in cl["items"].items():
    A("| **%s** | %s | %s | `%s` |" % (qid, v["title"], v.get("decision", v.get("k1_verdict", "—")), v.get("artifact", "a_gate_closure_v1.json")))
A("")
A("**闸门状态**：A 闸门 = **CLOSED**（5/5 sealed）。`phase_queue_v1.json`：Q01/Q02/Q08/Q09/Q12 → `sealed`，其余 25 项仍 `pending`。")
A("")
A("**I1 / I9 冻结确认**：")
A("")
A("- **I1**（§1 唯一全局 KPI）：只有 `E_read` / `E_ar(k)` / `C_steer` 可作全局 KPI；一个 Phase 若未改善其中任一，账本登记为 **catalog**，**不得**登记为 **advance**。")
A("- **I9**（§7 议程）：`phase_queue_v1.json` 为唯一议程来源；**禁止**由「残差 → 自动派生下一 Phase」。")
A("")
A("### 5 机制拼图与限界")
A("")
A("**新确立（本轮）**：")
A("")
A("- **死线免疫的完整病理** = 「合取（K1/K3）+ 两条根本没有装置（K2/K3）」。三条死线里 **两条从未被测量**，一条的触发条件写成**合取**。")
A("- **K1 双轨重算与旧结论相反**：读出层 `fired_all_models`（池化 margin %+.4f、`E_read` 池化 %.4f、3/3 否决）；k\\* 层 `model_specific`（`qwen3-4b` 否决）。"
  % (agg["readout_pooled_margin"], agg["E_read_pooled"]))
A("- **Q12 无实质违规**，但暴露两项制度缺陷：`RULE-UNDECIDABLE`（%s 条复合等级使「E 级禁入」不可机械判定）与 `NS-COLLISION`（账本 id 非唯一命名空间 ⇒ 纯 id grep 必假阳性，审计须走文本指纹）。"
  % "26/38")
A("")
A("**限界（新增/强化）**：")
A("")
A("- **K2/K3 至今没有装置** ⇒ K1 的触发**不代表**三条死线都被检验过；「算子代数线保住」的旧说法**双错**（层位自选 + 合取）。")
A("- **Q08 的层位选择本身是判据的一部分** —— 同一数据、同一门，层位一换结论翻转 ⇒ 今后任何「误差门」判据**必须预先冻结判定层**。")
A("- **C4/C6 未施加**（跨线）⇒ 共享账本 `ledger_sha256_8` 仍记录陈旧值 `%s`；任何外部读者引用账本自声明哈希时须以补丁 spec 为准。" % cl["corrections_C1_C6"][3]["resolved_value"]["recorded"])
A("")
A("### 6 第一性原理")
A("")
A("**判据若允许「选择测量位置」，它就不是判据，而是后验拟合。** K1 的教训不是「误差大」，而是「误差可以小到 0.8%，也可以大到 40%，取决于你把探针插在假设自己挑的那一层，还是插在模型真正说话的那一层」。")
A("")
A("**死线必须可测量且单值。** 合取（所有模型都必须…）把死线变成几乎不可能触发；两条没有装置的 K2/K3 则把死线变成装饰。凡写入合同的死线，必须同时写入(a)装置、(b)判定层、(c)聚合口径。")
A("")
A("### 7 后续死线")
A("")
A("- **B 闸门（需 GPU，本轮已探明可用：RTX 5080 / 16 GB / torch 2.13+cu130）**：Q03 `E_read` 统一基线复算 → Q04/Q05 `E_ar(k)` 装置与测量 → Q06 `C_steer` 基座。")
A("- **C 闸门（未识别）**：Q13–Q19 高秩散布因子恢复。")
A("- **D/E 闸门**：Q10（A 级可识别性回溯）、Q11（面板功效硬约束）、Q20–Q25（机制）。")
A("- **F 制度**：Q26（30-Phase KPI 复盘）、Q27（backlog 冻结）、Q30（队列收束）。")
A("- **待用户再确认**：跨线账本补丁（C4/C6）是否由本对话施加。")
A("")
A("### 8 一句话 ×3")
A("")
A("1. **A 闸门已关闭**：Q01/Q02/Q08/Q09/Q12 全部 sealed，主线存续问题定案为 Q08=甲。")
A("2. **K1 触发**：判定层取行为读出层 ⇒「条件齿轮组 = 算子代数」降级 descriptive；k\\* 层 `model_specific`。")
A("3. **全程未改跨线文件**：`AGI_GPT5_MEMO.md` / `atlas_ledger.json` / `proposition_ledger.json` 指纹逐字节未变；C4/C6 以补丁 spec 形式挂账。")

new_block = "\n".join(L)

# ---- 写入 ----
raw = open(MEMO, "rb").read()
pre_sha8 = hashlib.sha256(raw).hexdigest()[:8]
pre_bytes = len(raw)
assert pre_sha8 == EXPECT_PREFIX_SHA8, "前置哈希不符：%s（期望 %s）——先复核，勿盲目追加" % (pre_sha8, EXPECT_PREFIX_SHA8)
assert raw.startswith(b"\xef\xbb\xbf"), "缺 BOM"

old_txt = raw.decode("utf-8-sig")
base_lf = old_txt.replace("\r\n", "\n").rstrip("\n")
new_lf = base_lf + "\n\n" + new_block
payload = b"\xef\xbb\xbf" + new_lf.replace("\n", "\r\n").encode("utf-8")
open(MEMO, "wb").write(payload)

# ---- 回读 ----
raw2 = open(MEMO, "rb").read()
assert raw2[:len(raw)] == raw, "前缀被破坏！"
t2 = raw2.decode("utf-8-sig")
assert t2.count("\n") - t2.count("\r\n") == 0, "出现 bare LF"
assert "## Phase 35:" in t2 and "seal: Q08=甲" in t2 and "fired_all_models" in t2
n_phase = sum(1 for x in t2.replace("\r\n", "\n").split("\n") if x.startswith("## Phase "))

rep = []
rep.append("memo 前: %d B  %s" % (pre_bytes, pre_sha8))
rep.append("memo 后: %d B  %s" % (len(raw2), hashlib.sha256(raw2).hexdigest()[:8]))
rep.append("前缀逐字节不变 = %s" % (raw2[:len(raw)] == raw))
rep.append("bare_lf = %d" % (t2.count("\n") - t2.count("\r\n")))
rep.append("Phase 标题数 = %d（期望 35：Phase 1–34 + 35）" % n_phase)
rep.append("新增块行数 = %d" % len(L))
rep.append("新增字节 = %d" % (len(raw2) - pre_bytes))
txt = "\n".join(rep)
open(OUT, "w", encoding="utf-8").write(txt)
print(txt)
assert n_phase == 35, "Phase 标题数异常：%d" % n_phase
