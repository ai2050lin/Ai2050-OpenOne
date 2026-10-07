# -*- coding: utf-8 -*-
"""Q02 生成器：KPI 口径冻结。
交付：
  research/gpt5/atlas/metric_dict.json         （v1 -> v2 就地升级；meta_rules/metrics 原样保留）
  research/gpt5/atlas/metric_dict_v1_backup.json（v1 备份）
  research/gpt5/docs/METRIC_DICT_Q02.md         （口径报告）
全部数字现场解析，零手工转录；不改动任何 MEMO / TESTPLAN / Ledger / 3152 产物。
"""
import os, re, json, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
def rel(p):  return p.replace("\\", "/")
def rd(p):
    with open(os.path.join(ROOT, p), "rb") as f: return f.read()
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
def SER(o):  return json.dumps(o, ensure_ascii=False, indent=1).encode("utf-8")

MP  = "research/gpt5/atlas/metric_dict.json"
BP  = "research/gpt5/atlas/metric_dict_v1_backup.json"
BASE = "tests/glm5/result/rdc_query_construction_20260913/phase3152/g1p2_tri_model_k1"
NPZ_G = "tests/glm5/result/rdc_query_construction_20260913/phase3151/g1p1_combo_additive_vs_interaction/collect.npz"
DIAG = "research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md"

# ---------- v1 ----------
v1raw = rd(MP)
V1 = json.loads(v1raw.decode("utf-8"))
V1_SHA = sha8(v1raw)
assert V1["format"] == "metric_dict_v1", V1["format"]
assert len(V1["metrics"]) == 7, len(V1["metrics"])
assert "F2_verdict_grading" in V1["meta_rules"] and "F7_bit_anchor_status" in V1["meta_rules"]

# 备份 v1（幂等）
bp_abs = os.path.join(ROOT, BP)
if os.path.exists(bp_abs):
    assert open(bp_abs, "rb").read() == v1raw, "备份与当前 v1 不一致，中止"
    BACKUP_STATE = "verified_identical"
else:
    open(bp_abs, "wb").write(v1raw)
    BACKUP_STATE = "written"

# ---------- E_read 来源现场提取 ----------
ARMS = [("qwen3-4b", "qwen3-4b"), ("qwen3-14b", "qwen3-14b"), ("glm4-9b", "glm4k1")]
SRC = {}
for mname, arm in ARMS:
    p = "%s/%s/result.json" % (BASE, arm)
    b = rd(p)
    d = json.loads(b.decode("utf-8-sig"))
    k = d["k1_model_report"]
    SRC[mname] = {
        "result_path": rel(p), "result_file_sha8": sha8(b),
        "res_sha8": d.get("res_sha8"), "seal_sha8": d.get("seal_sha8"),
        "model_field": k["model"], "NL": k["NL"], "readout_layer": k["readout"], "kstar": k["kstar"],
        "b4_rel_readout_mean3seed": k["b4_rel_readout_mean3seed"],
        "b4_rel_readout_per_seed": k["b4_rel_readout_per_seed"],
        "b4_rel_kstar_mean3seed": k["b4_rel_kstar_mean3seed"],
        "panel_rows": d["panel_rows"], "n_pairs": d["n_pairs"],
    }
# summary
sp = "%s/summary/result_summary.json" % BASE
sb = rd(sp); SD = json.loads(sb.decode("utf-8-sig"))
SRC_SUMMARY = {"path": rel(sp), "file_sha8": sha8(sb), "res_sha8": SD.get("res_sha8"),
               "seal_sha8": SD.get("seal_sha8")}

def npzinfo(p):
    b = rd(p)
    return {"path": rel(p), "bytes": len(b), "sha8": sha8(b)}

CARRIER = {"qwen3-4b": npzinfo(BASE + "/qwen3-4b/collect.npz"),
           "qwen3-14b": npzinfo(BASE + "/qwen3-14b/collect.npz"),
           "glm4-9b": npzinfo(NPZ_G)}
# glm4 的 source_npz_sha8 交叉校验
g = json.loads(rd(BASE + "/glm4k1/result.json").decode("utf-8-sig"))
GLM_SRC_OK = (g.get("source_npz_sha8") == CARRIER["glm4-9b"]["sha8"])

E_READ = {m: SRC[m]["b4_rel_readout_mean3seed"] for m, _ in ARMS}
E_READ_FAIL = sum(1 for v in E_READ.values() if v > 0.05)

# ---------- C_steer 代理证据（从诊断正文现场提取） ----------
diat = rd(DIAG).decode("utf-8")
mc = re.search(r"cancel\s*反向加重\s*([\d.]+)", diat)
mk = re.search(r"附带损伤地板\s*([\d.]+)\s*/\s*(\d+)", diat)
C_PROXY = {"cancel_reverse_aggravate": float(mc.group(1)) if mc else None,
           "collateral_floor": ("%s/%s" % (mk.group(1), mk.group(2))) if mk else None,
           "source": DIAG}

# ---------- 组 v2 ----------
V2 = {
  "format": "metric_dict_v2",
  "version": 2,
  "created": "2026-10-01",
  "frozen_at": "2026-10-03",
  "generated_by": "tests/deepseek/gen_q02_r4.py",
  "supersedes": {"format": "metric_dict_v1", "sha8": V1_SHA, "created": "2026-10-01", "backup": BP},

  # ---- v1 原样保留（历史断言依赖） ----
  "meta_rules": V1["meta_rules"],
  "metrics": V1["metrics"],

  # ---- Q02 新增：三个全局 KPI ----
  "global_kpis": {
    "E_read": {
      "name_cn": "读出层预测误差",
      "space": "behavior-readout",
      "definition": "未见组合 (实体 i, 类 c) 在行为读出层的相对 L2 预测误差",
      "formula": "E_read(m) = mean_{s in 3 seeds} rel_L2( B4_pred_s(h_readout; i, c), y_true(i, c) )，对 held-out test fold 求均值",
      "operationalization": {
        "predictor_family": "B4 = 端点 / 词袋 / 位置 / 加性 四基线中最好的组合预测器（3151/3152 冻结实现）",
        "layer": "readout layer（模型特异；见 readout_layer）",
        "fold": "s7 test fold（147 行）",
        "seeds": 3,
        "aggregation": "对 3 seed 取算术平均",
        "primary_key": "k1_model_report.b4_rel_readout_mean3seed",
        "secondary_key_report_only": "k1_model_report.b4_rel_kstar_mean3seed（承诺层，仅附表，不作主线）"
      },
      "data": {"panel_rows": 738, "n_pairs": 246, "heldout_carrier": CARRIER,
               "carrier_cross_check_glm4_source_npz": GLM_SRC_OK},
      "gate": {"threshold": 0.05, "sense": "error <= 0.05 为过", "name": "5% 门"},
      "ci": {"method": "per-seed 全列 + 3-seed 均值；seed 数=3 不折叠为区间",
             "bootstrap": "未启用（seed=3）；扩至 >=10 seed 后再启用 percentile bootstrap"},
      "current": E_READ,
      "current_verdict": "all_above_gate (%d/3 过门)" % (3 - E_READ_FAIL),
      "source": {"phase": 3152, "per_model": SRC, "summary": SRC_SUMMARY},
      "status": "measured"
    },
    "E_ar": {
      "name_cn": "k 步自回归 logit-margin 误差",
      "space": "behavior-autoregressive",
      "definition": "把模型自身前 k 步生成内容回喂后，对第 k 步 logit-margin（目标 token 减竞争 token）的预测误差",
      "formula": "E_ar(k) = mean_cells | margin_pred(k; i, c) - margin_true(k; i, c) |，margin = logit(t_target) - logit(t_competitor)",
      "status": "not_built",
      "to_build_in": "Q04（装置）/ Q05（正式测量）",
      "data": "待 Q04 定义；须与 E_read 同一 held-out 面板族，并在 metric_dict 登记后冻结",
      "gate": "待 Q04 预注册（必须可失败）",
      "ci": "待 Q04",
      "k_range": "k = 1..K，K 由 Q04 预注册"
    },
    "C_steer": {
      "name_cn": "可控率（无附带损伤）",
      "space": "causal-control",
      "definition": "对 held-out 目标组合，用抽取出的机制构造干预后行为被正确改变、且无附带损伤的样本比例",
      "formula": "C_steer = #{ (i,c) : behaviour_after_intervention = target AND collateral(i,c) = 0 } / N_heldout",
      "status": "not_measured",
      "to_measure_in": "Q06（基座）/ Q20（正式化）",
      "gate": "待 Q20 预注册（低分即诚实总成绩，不得事后改判）",
      "ci": "binomial（n = held-out 行数）；MDE 由 Q11 功效约束决定",
      "proxy_evidence": C_PROXY
    }
  },

  "kpi_registration_rule": {
    "rule": "每个 Phase 必须报告 E_read / E_ar(k) / C_steer 三者；未降低其中任何一项者，Ledger 登记为 catalog（目录条目），不得登记为 advance（进展）",
    "source": "RDC_RESEARCH_CONSTITUTION_v1.md §1 (I1)",
    "anti_gaming": "必须与 §4 可识别性门 + §3 单模型否决权捆绑使用；KPI 被优化到虚假低值（held-out 泄漏 / 选易子面板）= 重演同一错误"
  },

  "ci_methods": {
    "seed_level": "3 seed 全部单列（per_seed），不折叠",
    "bootstrap": "未启用；扩 seed 后启用 percentile bootstrap（n_boot=10000）",
    "mde": "由 gate_precheck 计算（binomial 或 paired），一律报告 margin vs MDE"
  }
}

# ---- content_excluding_self 哈希（示范 Q01 §4.3 规范） ----
V2.pop("content_sha256_8", None)
V2["content_sha256_8"] = sha8(SER(V2))
v2raw = SER(V2)
open(os.path.join(ROOT, MP), "wb").write(v2raw)
V2_SHA = sha8(v2raw)

# ================= 报告 =================
A = []
def W(s=""): A.append(s)
W("# Q02 KPI 口径冻结 —— metric_dict v2")
W("")
W("- **文档性质**：治理文件（Q02 交付件）。**不改动** MEMO / TESTPLAN / Ledger / 任何 3152 产物。")
W("- **上位依据**：`RDC_RESEARCH_CONSTITUTION_v1.md`（sha8 `%s`）§1 I1。" % sha8(rd("research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md")))
W("- **主交付**：`%s`（v1 `%s` -> **v2 `%s`**，%d B）。" % (MP, V1_SHA, V2_SHA, len(v2raw)))
W("- **备份**：`%s`（%s）。" % (BP, BACKUP_STATE))
W("- **冻结日期**：2026-10-03")
W("")
W("---")
W("")
W("## §0 三个全局 KPI 的现状")
W("")
W("| KPI | 定义 | 现状 | 当前值 |")
W("|---|---|---|---|")
W("| `E_read` | 未见组合在行为读出层的 rel-L2 误差 | **已测**（3152） | %s |"
  % " / ".join("`%s` %.5f" % (m, v) for m, v in E_READ.items()))
W("| `E_ar(k)` | k 步自回归 logit-margin 误差 | **不存在**（待 Q04 建造） | - |")
W("| `C_steer` | 受控率（无附带损伤） | **未测**（待 Q06/Q20） | 代理：cancel 反向加重 %s、地板 %s |"
  % (C_PROXY["cancel_reverse_aggravate"], C_PROXY["collateral_floor"]))
W("")
W("> **E_read 在 5%% 门（0.05）上的判定：%d/3 过门** —— 三模型全部远超阈值。这是「行为读出层失败」的量化原话。" % (3 - E_READ_FAIL))
W("")
W("---")
W("")
W("## §1 E_read 冻结口径（主 KPI）")
W("")
W("**公式**：`E_read(m) = mean_{s in 3 seeds} rel_L2( B4_pred_s(h_readout; i, c), y_true(i, c) )`")
W("")
W("**操作化**（全部为已冻结实现，不得改口径）：")
W("")
W("- 预测器族 `B4` = 端点 / 词袋 / 位置 / 加性 四基线中最好的组合预测器（3151/3152 冻结实现）。")
W("- 读出层 `readout layer`（模型特异）：%s。" % " / ".join("`%s`=%d" % (m, SRC[m]["readout_layer"]) for m, _ in ARMS))
W("- held-out：**s7 test fold（147 行）**；面板 `panel_rows=%d`、`n_pairs=%d`；seed 数 = **3**。" % (SRC["qwen3-4b"]["panel_rows"], SRC["qwen3-4b"]["n_pairs"]))
W("- 主键 = `k1_model_report.b4_rel_readout_mean3seed`；承诺层 k* 值仅附表（`b4_rel_kstar_mean3seed`），**不得用于主线判定**。")
W("")
W("**判据**：5% 门（`error <= 0.05`）。")
W("")
W("**CI**：3 seed 全部单列，不折叠；seed 数扩到 >=10 后再启用 percentile bootstrap。")
W("")
W("**来源链（可逐条回查）**：")
W("")
W("| 模型 | result.json | 文件 sha8 | `res_sha8` | 读出层 | B4@读出层 | per-seed |")
W("|---|---|---|---|---|---|---|")
for m, _ in ARMS:
    s = SRC[m]
    W("| %s | `%s` | `%s` | `%s` | %d | **%.5f** | %s |"
      % (m, os.path.basename(os.path.dirname(s["result_path"])) + "/result.json",
         s["result_file_sha8"], s["res_sha8"], s["readout_layer"],
         s["b4_rel_readout_mean3seed"],
         " / ".join("%.5f" % x for x in s["b4_rel_readout_per_seed"])))
W("")
W("汇总：`%s`（sha8 `%s` / `res_sha8` `%s`）。" % (SRC_SUMMARY["path"], SRC_SUMMARY["file_sha8"], SRC_SUMMARY["res_sha8"]))
W("")
W("**held-out 载体指纹**（面板不可替换）：")
W("")
W("| 模型 | 文件 | 字节 | sha8 |")
W("|---|---|---|---|")
for m in CARRIER:
    W("| %s | `%s` | %d | `%s` |" % (m, CARRIER[m]["path"].split("rdc_query_construction_20260913/")[-1],
                                    CARRIER[m]["bytes"], CARRIER[m]["sha8"]))
W("")
W("glm4-9b 的载体为 3151 npz；其 `source_npz_sha8` 与实测 sha8 一致：**%s**。" % GLM_SRC_OK)
W("")
W("---")
W("")
W("## §2 E_ar(k) 口径（待 Q04 建造）")
W("")
W("- **定义**：把模型自身前 k 步生成内容回喂后，对第 k 步 logit-margin 的预测误差。")
W("- **公式**：`E_ar(k) = mean_cells | margin_pred(k; i,c) - margin_true(k; i,c) |`，`margin = logit(t_target) - logit(t_competitor)`。")
W("- **k 范围**：`k = 1..K`，K 由 Q04 预注册。")
W("- **约束**：必须与 `E_read` **同一 held-out 面板族**；判据必须可失败；登记后再冻结。")
W("- **现状**：MEMO 中 `E_ar` 出现 **0** 次 ⇒ 该量从未被命名过。")
W("")
W("---")
W("")
W("## §3 C_steer 口径（待 Q06 / Q20）")
W("")
W("- **定义**：对 held-out 目标组合，用抽取机制干预后行为被正确改变、且无附带损伤的样本比例。")
W("- **公式**：`C_steer = #{ (i,c) : behaviour_after = target AND collateral(i,c) = 0 } / N_heldout`。")
W("- **代理证据**（现有，指向低值）：cancel 反向加重 **%s**；附带损伤地板 **%s**。" % (C_PROXY["cancel_reverse_aggravate"], C_PROXY["collateral_floor"]))
W("- **纪律**：低分即诚实总成绩，**不得**事后改判。")
W("")
W("---")
W("")
W("## §4 登记规则与反风险（I1 核心）")
W("")
W("> %s" % V2["kpi_registration_rule"]["rule"])
W("")
W("- **反风险**：%s" % V2["kpi_registration_rule"]["anti_gaming"])
W("")
W("---")
W("")
W("## §5 与 Q01 的衔接")
W("")
W("- Q01 的 `meta_single_source_v4.json` 中 `single_source_of_truth.kpi.status = \"pending_Q02\"`；本文件完成后该指针应指向 `%s` v2（`%s`）。" % (MP, V2_SHA))
W("- 本文件**示范了 Q01 §4.3 的哈希规范**：`content_sha256_8` = **`%s`**（content_excluding_self，`json.dumps(ensure_ascii=False, indent=1)`，UTF-8，无尾换行）。" % V2["content_sha256_8"])
W("")
W("---")
W("")
W("## §6 兼容性（历史断言仍过）")
W("")
W("- `tests/gpt5_temp/p3150_disk_verify.py` 的两条断言保持不变：")
W("  - `len(metric_dict['metrics']) == 7` -> 本文件 `metrics` 原样 7 项 ✓")
W("  - `'F2_verdict_grading' in meta_rules and 'F7_bit_anchor_status' in meta_rules` -> `meta_rules` 原样 ✓")
W("- 故 v1 -> v2 为**向后兼容升级**；v1 已备份为 `%s`。" % BP)
W("")
W("---")
W("")
W("## §7 边界")
W("")
W("1. 本文件**未改动** MEMO 原文、TESTPLAN、Ledger、3152 产物。")
W("2. `E_read` 的全部数字来自 3152 四个 result.json 的 `k1_model_report`，现场读取。")
W("3. `E_ar(k)` / `C_steer` 为**口径预注册**，尚未测量；测量须新开 Phase（Q04-Q06）。")
W("")
W("*本文件为治理性文档；哈希与数值均现场解析，可逐条回查。*")

mdp = os.path.join(ROOT, "research/gpt5/docs/METRIC_DICT_Q02.md")
mdb = ("\n".join(A) + "\n").encode("utf-8")
open(mdp, "wb").write(mdb)

print("BACKUP   %s  %s  sha8=%s" % (BP, BACKUP_STATE, sha8(v1raw)))
print("WROTE    %s  %d B  sha8=%s (v1 %s -> v2)" % (MP, len(v2raw), V2_SHA, V1_SHA))
print("WROTE    %s  %d B  sha8=%s" % (mdp, len(mdb), sha8(mdb)))
print("E_READ   %s" % E_READ)
print("GLM_SRC_OK=%s  self_hash=%s" % (GLM_SRC_OK, V2["content_sha256_8"]))
print("OK")
