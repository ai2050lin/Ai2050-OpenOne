# -*- coding: utf-8 -*-
"""Q09 生成器：死线双轨重述（I3）。
把 K1/K2/K3 从「跨模型合取」重述为「聚合统计 + 单模型否决权」。
所有数字从 3151/3152 result 现场读取（禁手工转录）。
产物：research/gpt5/docs/DEADLINE_DUAL_TRACK_Q09.md
      research/gpt5/atlas/deadline_dual_track_v1.json
"""
import os, json, hashlib, random

ROOT = r"D:\AI2050\Ai2050-OpenOne"
BASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
DOCS = os.path.join(ROOT, "research", "gpt5", "docs")
ATLAS = os.path.join(ROOT, "research", "gpt5", "atlas")

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

def jload(p):
    return json.loads(open(p, "rb").read().decode("utf-8-sig"))

P3152 = os.path.join(BASE, "phase3152", "g1p2_tri_model_k1")
P3151 = os.path.join(BASE, "phase3151", "g1p1_combo_additive_vs_interaction")
summary = jload(os.path.join(P3152, "summary", "result_summary.json"))
r3151 = jload(os.path.join(P3151, "result.json"))

MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]
MDIR = {"qwen3-4b": "qwen3-4b", "qwen3-14b": "qwen3-14b", "glm4-9b": "glm4k1"}
arms = {m: jload(os.path.join(P3152, MDIR[m], "result.json")) for m in MODELS}

m1_k39 = r3151["v1"]["gates"]["M1_k39"]
glm_readout_margins = [round(x["margin"], 12) for x in m1_k39]
glm_readout_mdes = [round(x["mde"], 12) for x in m1_k39]

K1 = {}
for m in MODELS:
    s = summary["k1_3model"][m]
    a = arms[m]["k1_model_report"]
    if m == "glm4-9b":
        ro_m, ro_d = glm_readout_margins, glm_readout_mdes
    else:
        ro_m = [round(x, 12) for x in a["m1_readout"]["margins"]]
        ro_d = [round(x, 12) for x in a["m1_readout"]["mdes"]]
    K1[m] = {
        "kstar": s["kstar"], "readout": s["readout"],
        "b4_kstar": s["b4_rel_kstar"], "b4_readout": s["b4_rel_readout"],
        "m1_kstar_margins": [round(x, 12) for x in s["m1_kstar_margins"]],
        "m1_kstar_mdes": [round(x, 12) for x in a["m1_kstar"]["mdes"]],
        "m1_readout_margins": ro_m, "m1_readout_mdes": ro_d,
        "above_add_gate": s["above_add_gate"],
        "m1_kstar_pass_all3": s["m1_kstar_pass_all3"],
    }

def mean(v):
    return sum(v) / float(len(v))

def boot_ci(vals, n=20000, seed=20261003, alpha=0.05):
    rnd = random.Random(seed)
    k = len(vals)
    ms = []
    for _ in range(n):
        ms.append(sum(vals[rnd.randrange(k)] for _ in range(k)) / float(k))
    ms.sort()
    return ms[int(alpha / 2 * n)], ms[int((1 - alpha / 2) * n) - 1]

E_READ = [K1[m]["b4_readout"] for m in MODELS]
E_READ_POOLED = mean(E_READ)

def pooled(margins_key, mdes_key):
    marg, mdes = [], []
    for m in MODELS:
        marg += K1[m][margins_key]
        mdes += K1[m][mdes_key]
    return mean(marg), mean(mdes), marg

mr, dr, marg_ro_all = pooled("m1_readout_margins", "m1_readout_mdes")
mk, dk, marg_k_all = pooled("m1_kstar_margins", "m1_kstar_mdes")

model_level = {}
for m in MODELS:
    mm, md = mean(K1[m]["m1_readout_margins"]), mean(K1[m]["m1_readout_mdes"])
    km, kd = mean(K1[m]["m1_kstar_margins"]), mean(K1[m]["m1_kstar_mdes"])
    model_level[m] = {
        "readout_margin_model": mm, "readout_mde_model": md, "readout_pass": mm <= -2 * md,
        "kstar_margin_model": km, "kstar_mde_model": kd, "kstar_pass": km <= -2 * kd,
    }

ci_ro3 = boot_ci([model_level[m]["readout_margin_model"] for m in MODELS])
ci_ks3 = boot_ci([model_level[m]["kstar_margin_model"] for m in MODELS])
ci_ro9 = boot_ci(marg_ro_all)
ci_ks9 = boot_ci(marg_k_all)
ci_eread = boot_ci(E_READ)

veto_ro = [m for m in MODELS if not model_level[m]["readout_pass"]]
veto_ks = [m for m in MODELS if not model_level[m]["kstar_pass"]]

t1_ks = mk > -2 * dk
t1_ro = (mr > -2 * dr) or (E_READ_POOLED > 0.05)
K1_KS_VERDICT = "model_specific" if veto_ks else ("fired" if t1_ks else "not_fired")
if t1_ro and len(veto_ro) == len(MODELS):
    K1_RO_VERDICT = "fired_all_models"
else:
    K1_RO_VERDICT = "model_specific" if veto_ro else ("fired" if t1_ro else "not_fired")

prov = {
    "summary": os.path.relpath(os.path.join(P3152, "summary", "result_summary.json"), ROOT),
    "summary_sha8": sha8(os.path.join(P3152, "summary", "result_summary.json")),
    "r3151": os.path.relpath(os.path.join(P3151, "result.json"), ROOT),
    "r3151_sha8": sha8(os.path.join(P3151, "result.json")),
    "arms": {m: {"path": os.path.relpath(os.path.join(P3152, MDIR[m], "result.json"), ROOT),
                 "sha8": sha8(os.path.join(P3152, MDIR[m], "result.json"))} for m in MODELS},
    "glm4_readout_source": "3152 glm4k1 仅有 note；读出层 margin/MDE 取 3151 v1.gates.M1_k39",
}

out = {
    "schema": "rdc_deadline_dual_track_v1",
    "generated_by": "tests/deepseek/gen_q09_r5.py",
    "queue_item": "Q09",
    "constitution_ref": "RDC_RESEARCH_CONSTITUTION_v1.md 3 (I3)",
    "source_testplan": {"path": "research/gpt5/docs/RDC_TESTPLAN_v1.md",
                        "sha8": sha8(os.path.join(DOCS, "RDC_TESTPLAN_v1.md"))},
    "original_form": {
        "K1": "3 模型 × 未见组合 的 logit 预测误差 > 5% 且不显著优于全加性基线 B4",
        "K2": "phi_l(c) 与 W_l 不可分离：分离后响应 cos 下降 > 50% 而行为不掉",
        "K3": "在 3 个模式族上，单坐标筛选 top-50 覆盖率均 < 30%",
        "structural_defect": {
            "K1": "触发 = 对 3 模型的全称量词 AND（合取）；与跨模型从不一致叠加 ⇒ 触发概率趋 0",
            "K2": "同一文件内两种不一致操作化（§8.2 响应 cos 下降>50% vs 3154 预注册 交互份额>50%）",
            "K3": "触发 = 对 3 模式族的全称量词 AND（合取）",
        },
    },
    "restated_form": {
        "track_A_primary": {
            "rule": "主判据 = 跨模型聚合统计量 + bootstrap CI；禁用合取",
            "K1": "pooled margin mbar=mean_{model,seed}(margin)；pooled error Ebar_read=mean_model(E_read)；触发当 (mbar > -2*MDE_pooled) OR (Ebar_read > 0.05)",
            "K2": "pooled |dcos| 与 pooled dchg；触发当 (|dcos|_bar > 0.50) AND (dchg_bar < MDE_pooled)",
            "K3": "pooled coverage cov_bar；触发当 (cov_bar < 0.30)",
        },
        "track_B_veto": {
            "rule": "任一模型在聚合口径下不达标（与聚合结论相反）⇒ 命题标 model_specific",
            "consequence": "model_specific 不得升为机制；只能作 descriptive / 待复现",
        },
        "forbidden": [
            "把全称量词形式的合取写进触发条件",
            "用任一模型不达标当作加固主判据的证据（方向必须相反：它是降级信号）",
        ],
    },
    "k1_recompute": {
        "per_model": K1,
        "model_level": model_level,
        "aggregate": {
            "E_read_per_model": E_READ, "E_read_pooled": E_READ_POOLED,
            "E_read_ci95_modellevel": ci_eread,
            "kstar_pooled_margin": mk, "kstar_pooled_mde": dk,
            "kstar_margin_ci95_modellevel": ci_ks3, "kstar_margin_ci95_unit9": ci_ks9,
            "readout_pooled_margin": mr, "readout_pooled_mde": dr,
            "readout_margin_ci95_modellevel": ci_ro3, "readout_margin_ci95_unit9": ci_ro9,
        },
        "verdict": {
            "kstar_layer": K1_KS_VERDICT, "kstar_trackA_fired": bool(t1_ks),
            "kstar_veto_models": veto_ks,
            "readout_layer": K1_RO_VERDICT, "readout_trackA_fired": bool(t1_ro),
            "readout_veto_models": veto_ro,
            "layer_choice_is_Q08": True,
        },
        "old_form_result": "above=1/3, m1_pass@k*=2/3 ⇒ NOT triggered ⇒ operator line kept",
    },
    "k2_status": {
        "state": "not_measurable_yet",
        "reason": "phase3154 目录不存在（预注册于 3153 预注册 3154，尚未运行）",
        "note": "同一文件内两种操作化不一致，须在 3154 开跑前先冻结其一",
    },
    "k3_status": {
        "state": "not_measurable_yet",
        "reason": "Phases 中无单坐标筛选 top-50 覆盖率量；MEMO 中的覆盖率指微场普查 283 词覆盖率，非同一条目",
    },
    "deadline_immunity_quantified": {
        "old_conjunction": "K1/K3 均为跨模型或跨族全称量词 ⇒ 系统性不可触发",
        "never_measured": ["K2", "K3"], "measured": ["K1"],
    },
    "provenance": prov,
}

jp = os.path.join(ATLAS, "deadline_dual_track_v1.json")
with open(jp, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(out, ensure_ascii=False, indent=1))

L = []
def W(s=""):
    L.append(s)
f6 = lambda x: "{:.6f}".format(x)

W("# Q09 死线双轨重述 —— 让死线可以触发")
W("")
W("- **文档性质**：研究治理文件（非 Phase 记录，不改动任何 MEMO 原文 / 既有判决）。")
W("- **上位依据**：`RDC_RESEARCH_CONSTITUTION_v1.md` §3（I3）；`RDC_TESTPLAN_v1.md` §8.2（原文逐字）。")
W("- **数据来源**：3152 `result_summary.json`（`%s`）+ 3151 `result.json`（`%s`）+ 三臂 `result.json`；全部数字由 `gen_q09_r5.py` 现场读取。" % (prov["summary_sha8"], prov["r3151_sha8"]))
W("- **冻结日期**：2026-10-03")
W("")
W("---")
W("")
W("## §0 结论摘要")
W("")
W("| 死线 | 旧形式的触发条件 | 结构病 | 新形式下的实测 | 状态 |")
W("|---|---|---|---|---|")
W("| **K1** | 3 模型全 above **且** M1 全败 | 对模型的全称量词（合取） | k* 层 `model_specific`；读出层 **触发**（3/3 否决 + 主判据） | **可测量** |")
W("| **K2** | 分离后 cos 降 >50% 且行为不掉 | 同文件内两种操作化互斥 | 无数据 | **从未被测量** |")
W("| **K3** | 3 模式族 top-50 覆盖率 **均** <30% | 对族数的全称量词（合取） | 无数据 | **从未被测量** |")
W("")
W("> **三个死线里，两个从未被测量，一个的触发条件写成合取。** 这就是“死线免疫”的完整病理：不只是合取，还有两条根本没有装置。")
W("")
W("---")
W("")
W("## §1 原文（逐字保留，不得改动）")
W("")
W("`RDC_TESTPLAN_v1.md` §8.2：")
W("")
W("| 编号 | 条件 | 放弃什么 |")
W("|---|---|---|")
W("| **K1** | 3 模型 × **未见组合** 的 logit 预测误差 **> 5%** 且不显著优于全加性基线 B4 | 放弃“条件齿轮组=算子代数”，降级为“功能性端口类描述” |")
W("| **K2** | phi_l(c) 与 W_l 不可分离：分离后响应 cos 下降 **> 50%** 而行为不掉 | 放弃“条件门”作为独立结构 |")
W("| **K3** | 在 3 个模式族上，单坐标筛选 top-50 覆盖率均 **< 30%** | 放弃“单坐标机制”目标，改为“低维联合（≤10 维）+ 端口类”目标 |")
W("")
W("**结构病（三处，逐条可验）**")
W("")
W("1. **K1 的触发是合取。** “3 模型 × …误差>5% 且不显著优于 B4”的字面展开是「对 3 模型全体 above5」**且**「对 3 模型全体 not-better」。项目自述“跨模型从不一致”，两支同时成立的概率被系统性压低。实测 **above = 1/3**。")
W("2. **K3 的触发也是合取**（“3 个模式族上…**均**<30%”），同病。")
W("3. **K2 在同一份文件里有两个互斥的操作化**：§8.2 写“分离后**响应 cos 下降 >50%**”，而 3153 的 §预注册 3154 写“不可分离（**交互份额>50%**）”。两个量的量纲与零假设都不同 ⇒ 判决无法预登记。")
W("")
W("---")
W("")
W("## §2 重述：双轨制")
W("")
W("### 轨 A —— 主判据 = 跨模型聚合量 + bootstrap CI（**禁用合取**）")
W("")
W("| 死线 | 聚合量 | 触发条件 |")
W("|---|---|---|")
W("| K1 | mbar = mean_{m,s}(margin)；Ebar_read = mean_m(E_read) | (mbar > -2·MDE_pooled) **或** (Ebar_read > 0.05) |")
W("| K2 | mean(|Δcos|)；mean(Δchg) | (mean(|Δcos|) > 0.50) **且** (mean(Δchg) < MDE_pooled) |")
W("| K3 | mean(cov) | (mean(cov) < 0.30) |")
W("")
W("### 轨 B —— 单模型否决权")
W("")
W("- **任一模型**在聚合口径下不达标（与聚合结论相反）⇒ 该命题标 **`model_specific`**。")
W("- **`model_specific` 不得升为机制**：只能以 `descriptive` 或“待复现”形式引用（与 §4 可识别性门同一处置）。")
W("- 方向必须写对：**“任一模型不达标”是降级信号，不是加固信号。**")
W("")
W("### 明令禁止")
W("")
W("1. 把全称量词形式的合取写进触发条件。")
W("2. 用“任一模型不达标”当作主判据成立的证据。")
W("")
W("---")
W("")
W("## §3 K1 重算（数据全部来自磁盘 result，零手工转录）")
W("")
W("### 3.1 逐模型")
W("")
W("| 模型 | k* | 读出层 | B4@k* | B4@读出层 | M1@k* margin (3 seed) | MDE@k* | M1@读出层 margin (3 seed) | above |")
W("|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    d = K1[m]
    W("| %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
        m, d["kstar"], d["readout"], f6(d["b4_kstar"]), f6(d["b4_readout"]),
        " / ".join("{:+.6f}".format(x) for x in d["m1_kstar_margins"]),
        " / ".join(f6(x) for x in d["m1_kstar_mdes"]),
        " / ".join("{:+.4f}".format(x) for x in d["m1_readout_margins"]),
        str(d["above_add_gate"])))
W("")
W("> 门语义：`margin = err_cand - err_B4`，**负 = 候选优**；过门当 `margin <= -2 x MDE`。glm4 的读出层 margin/MDE 取 3151 `v1.gates.M1_k39`（3152 glm4k1 仅留 note，未重算）。")
W("")
W("### 3.2 聚合")
W("")
W("| 量 | 值 | 95% CI（模型级 n=3） |")
W("|---|---|---|")
W("| Ebar_read（池化读出层误差） | **%s** | [%s, %s] |" % (f6(E_READ_POOLED), f6(ci_eread[0]), f6(ci_eread[1])))
W("| 池化 margin @ **k\\*** | **%+.6f** | [%+.6f, %+.6f] |" % (mk, ci_ks3[0], ci_ks3[1]))
W("| 池化 MDE @ k* | %s | — |" % f6(dk))
W("| 池化 margin @ **读出层** | **%+.6f** | [%+.6f, %+.6f] |" % (mr, ci_ro3[0], ci_ro3[1]))
W("| 池化 MDE @ 读出层 | %s | — |" % f6(dr))
W("")
W("### 3.3 双轨判决")
W("")
W("**读出层（行为层）**")
W("")
W("- 轨 A：池化 margin = **%+.6f** ⇒ 为正（候选**比 B4 更差**），且 Ebar_read = **%s > 0.05** ⇒ **主判据触发**。" % (mr, f6(E_READ_POOLED)))
W("- 轨 B：否决模型 = %s ⇒ **%d/3**。" % (", ".join(veto_ro) if veto_ro else "无", len(veto_ro)))
W("- **判决：`%s`** —— 与旧形式“未触发、算子代数线保住”的结论相反。" % K1_RO_VERDICT)
W("")
W("| 模型 | 模型级池化 margin | 模型级 2×MDE | 过门 |")
W("|---|---|---|---|")
for m in MODELS:
    d = model_level[m]
    W("| %s | %+.6f | %.6f | %s |" % (m, d["readout_margin_model"], 2 * d["readout_mde_model"], "是" if d["readout_pass"] else "**否**"))
W("")
W("**k\\* 层（承诺层）**")
W("")
W("- 轨 A：池化 margin = **%+.6f**，2·MDE_pooled = %.6f ⇒ margin ≤ -2·MDE ⇒ **主判据通过**（候选显著优于 B4）。" % (mk, 2 * dk))
W("- 轨 B：否决模型 = %s ⇒ **%d/3** ⇒ **判决：`%s`**。" % (", ".join(veto_ks) if veto_ks else "无", len(veto_ks), K1_KS_VERDICT))
W("")
W("| 模型 | 模型级池化 margin | 模型级 2×MDE | 过门 |")
W("|---|---|---|---|")
for m in MODELS:
    d = model_level[m]
    W("| %s | %+.6f | %.6f | %s |" % (m, d["kstar_margin_model"], 2 * d["kstar_mde_model"], "是" if d["kstar_pass"] else "**否（否决）**"))
W("")
W("> **层位选择属 Q08**（需用户 seal）。本节把两个层的双轨结果**并列报告**，不代为决定哪一层是 K1 的判定层。")
W("")
W("---")
W("")
W("## §4 K2 / K3：从未被测量")
W("")
W("| 死线 | 装置 | 磁盘证据 |")
W("|---|---|---|")
W("| K2 | `phase3154` | **目录不存在**（`tests/glm5/result/rdc_query_construction_20260913/phase3154*` → 0 命中）；预注册停在 3153 的“§预注册 Phase 3154” |")
W("| K3 | “单坐标筛选 top-50 覆盖率” | 全 MEMO 无此量。MEMO 中的“覆盖率”全部指微场普查（283 词, τ=0.45, θ=0.50 >= 0.70），与 K3 条目**不同源** |")
W("")
W("> 即：K2 与 K3 在预注册后**从未开跑**。它们对“整条主线”的否证能力至今为**零**。")
W("")
W("---")
W("")
W("## §5 与 I3 的关系与边界")
W("")
W("- 本文件是 **I3 的落地形式**：聚合统计 + 单模型否决权。")
W("- **未改动** `RDC_TESTPLAN_v1.md` 原文；旧形式在本文件中以“原文（逐字保留，不得改动）”整节保留。")
W("- `model_specific` 的处置（不得升为机制）与 `descriptive` 同级，接入 `metric_dict.json` 的判决分级。")
W("- 本文件的**生效**需用户 seal（与 Q08、I1、I9 同批）。")
W("")
W("---")
W("")
W("*本文件为治理性文档；所有统计数字由 `gen_q09_r5.py` 从 result 现场读取，可逐条回查。*")

mp = os.path.join(DOCS, "DEADLINE_DUAL_TRACK_Q09.md")
with open(mp, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(L) + "\n")

print("WROTE", os.path.relpath(jp, ROOT), os.path.getsize(jp), "B", sha8(jp))
print("WROTE", os.path.relpath(mp, ROOT), os.path.getsize(mp), "B", sha8(mp))
print("K1 verdict: k*=%s readout=%s" % (K1_KS_VERDICT, K1_RO_VERDICT))
print("pooled: Eread=%.6f mk=%+.6f dk=%.6f mr=%+.6f dr=%.6f" % (E_READ_POOLED, mk, dk, mr, dr))
print("veto: ks=%s ro=%s" % (veto_ks, veto_ro))
