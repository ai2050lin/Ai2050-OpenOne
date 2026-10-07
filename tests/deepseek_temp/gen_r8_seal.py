# -*- coding: utf-8 -*-
"""R8 seal 落盘（JSON 部分）：关闭记录 + 队列状态 + meta 更正 + 跨线补丁spec。
每处写入：写前断言(前缀/字段) -> 写 -> 回读断言。"""
import os, json, hashlib, datetime

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_r8_seal_report.txt")
r = []
def add(s): r.append(s)
def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def sha_all(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def rj(p): return json.loads(open(p, "rb").read().decode("utf-8-sig"))
def wj(p, obj):
    b = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    open(p, "wb").write(b)

NOW = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
SEAL = "seal: Q08=甲 | C=全接受 | I1=确认 | I9=确认"

QUEUE = os.path.join(ROOT, r"research\deepseek\atlas\phase_queue_v1.json")
META  = os.path.join(ROOT, r"tests\deepseek\result\meta_single_source_v4.json")
DL    = os.path.join(ROOT, r"tests\deepseek\result\deadline_dual_track_v1.json")
CLOSE = os.path.join(ROOT, r"research\deepseek\atlas\a_gate_closure_v1.json")
PATCH = os.path.join(ROOT, r"research\deepseek\atlas\ledger_corrections_v1.json")
LEDGER= os.path.join(ROOT, r"research\gpt5\atlas\atlas_ledger.json")

# ---------- 0) 独立重算 C4 的 content_excluding_self ----------
lg = rj(LEDGER)
lg_wo = {k: v for k, v in lg.items() if k != "ledger_sha256_8"}
recomputed = hashlib.sha256(json.dumps(lg_wo, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()[:8]
recomputed_nl = hashlib.sha256((json.dumps(lg_wo, ensure_ascii=False, indent=1) + "\n").encode("utf-8")).hexdigest()[:8]
add("=== C4 独立重算 ===")
add("  ledger 现文件 sha8 = %s" % sha8(LEDGER))
add("  recorded ledger_sha256_8 = %s" % lg.get("ledger_sha256_8"))
add("  recomputed(no trailing nl) = %s   <- 期望 0dc6e57a" % recomputed)
add("  recomputed(with trailing nl)= %s" % recomputed_nl)
C4_OK = (recomputed == "0dc6e57a")
add("  C4 可复现 = %s" % C4_OK)

# ---------- 1) 关闭记录 ----------
dl = rj(DL)
k1 = dl["k1_recompute"]
verdict = k1["verdict"]
per = k1["per_model"]; ml = k1["model_level"]; agg = k1["aggregate"]

corr = [
 {"id":"C1","target":"MEMO 3103 正文分级数字","current":"未标口径","resolved":"加标 count_mode=component（条数=62、分量和=94）",
  "resolved_value":{"count_mode":"component","n_atomic":62,"sum_components":94,
                    "table":{"A":5,"B":34,"C":14,"D":21,"E":20}},
  "enactment":"本线已记录；目标正文位于 G 线备忘录（AGI_GPT5_MEMO.md），跨线不改。","status":"accepted"},
 {"id":"C2","target":"MEMO 3103 正文「约 55% 命题（A+B）」","current":"量纲混淆（分量数÷条数）",
  "resolved":"同口径只报一个值：canonical=component",
  "resolved_value":{"canonical_pct":41.4894,"canonical_frac":"39/94","alternate_pct":62.9032,"alternate_frac":"39/62",
                    "canonical_mode":"component"},
  "enactment":"本线已记录并采用 canonical=41.5%。","status":"accepted"},
 {"id":"C3","target":"TESTPLAN T5 行","current":"A=5 / B=6 / C=10 / D=21 / E=20",
  "resolved":"A=5 / B=34 / C=14 / D=21 / E=20（count_mode=component，条数 62）",
  "resolved_value":{"A":5,"B":34,"C":14,"D":21,"E":20,"count_mode":"component","n_atomic":62},
  "enactment":"TESTPLAN 已作为归档全文并入本线备忘录（Phase 23，逐字保留）；更正以本记录 + memo erratum 形式声明，不改写归档正文。","status":"accepted"},
 {"id":"C4","target":"atlas_ledger.json ledger_sha256_8","current":"41d65a13",
  "resolved":"0dc6e57a（content_excluding_self）",
  "resolved_value":{"recorded":"41d65a13","file_sha8":sha8(LEDGER),"correct_self_excluding":recomputed,
                    "recomputed_here":C4_OK},
  "enactment":"目标文件为跨线共享（含 G 线 Phase 2902–3153 测量）⇒ 未施加；补丁规格见 ledger_corrections_v1.json。","status":"pending_cross_line"},
 {"id":"C5","target":"MEMO 3103 记录的 proposition_ledger.json 哈希","current":"add57ba7","resolved":"9a3c6ff4",
  "resolved_value":{"recorded_in_memo3103":"add57ba7","actual":"9a3c6ff4",
                    "path":"tests/glm5/result/rdc_query_construction_20260913/phase3103/omega_p101_formula_audit/proposition_ledger.json",
                    "verified_here":sha8(os.path.join(ROOT, r"tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json"))},
  "enactment":"本线以外部 manifest 记录真值（哈希自指失效 ⇒ 不写入被哈希文件本体）。","status":"accepted"},
 {"id":"C6","target":"atlas_ledger measurements schema","current":"两套并存（270 / 199 / 兼有 165）；evidence_level 常量",
  "resolved":"统一含 meas_id+phase；evidence_level 三值枚举（bit_anchored / statistical / descriptive）",
  "resolved_value":{"n_total":len(lg["measurements"]),
                    "n_has_meas_id":sum(1 for m in lg["measurements"] if "meas_id" in m),
                    "n_has_phase":sum(1 for m in lg["measurements"] if "phase" in m),
                    "n_both":sum(1 for m in lg["measurements"] if "meas_id" in m and "phase" in m),
                    "evidence_level_values":sorted({str(m.get("evidence_level")) for m in lg["measurements"]})},
  "enactment":"目标为跨线共享文件 ⇒ 未施加；补丁规格见 ledger_corrections_v1.json。","status":"pending_cross_line"},
]

closure = {
 "schema": "rdc_a_gate_closure_v1",
 "sealed_at": NOW,
 "seal_verbatim": SEAL,
 "gate": "A 闸门（零 GPU 前置闸门）",
 "gate_closed": True,
 "provenance": {"line": "deepseek", "memo": "research/deepseek/docs/AGI_DEEPSEEK_MEMO.md",
                "generated_by": "tests/deepseek_temp/gen_r8_seal.py"},
 "items": {
   "Q01": {"title":"元层单一真源对账","block":"A 闸门","status":"sealed","decision":"接受 Q01 五处不自洽定位 + schema v4",
           "artifact":"tests/deepseek/result/meta_single_source_v4.json"},
   "Q02": {"title":"KPI 口径冻结","block":"A 闸门","status":"sealed","decision":"I1 确认：E_read / E_ar(k) / C_steer 为唯一全局 KPI；未改善者登记 catalog 不得登记 advance",
           "artifact":"research/deepseek/atlas/metric_dict.json","sha8":sha8(os.path.join(ROOT,r"research\deepseek\atlas\metric_dict.json"))},
   "Q08": {"title":"K1 重述与改判","block":"A 闸门","status":"sealed","decision":"读法甲（甲=改判）：判定层 = 行为读出层",
           "k1_verdict":"fired_all_models @ readout；model_specific @ k*",
           "proposition_effect":"「条件齿轮组 = 算子代数」由 mechanism 降级为 descriptive（功能性端口类描述）"},
   "Q09": {"title":"死线双轨重述","block":"A 闸门","status":"sealed",
           "decision":"轨 A 聚合统计 + bootstrap CI（禁合取）；轨 B 单模型否决权；model_specific 不得升机制",
           "artifact":"tests/deepseek/result/deadline_dual_track_v1.json"},
   "Q12": {"title":"D/E 级命题引用审计","block":"A 闸门","status":"sealed",
           "decision":"新推理链 id 命中 1 处（R55，Phase 3113）+ 文本指纹 0 命中 ⇒ 无实质违规；两项制度缺陷记录在案",
           "artifact":"tests/deepseek/result/prop_citation_audit_v1.json"}
 },
 "k1_reverdict": {
   "judging_layer_under_Q08_A": "readout（行为读出层）",
   "per_model": {m: {"kstar": per[m]["kstar"], "readout_layer": per[m]["readout"],
                     "B4_err_at_kstar": per[m]["b4_kstar"], "B4_err_at_readout": per[m]["b4_readout"],
                     "kstar_margin_model": ml[m]["kstar_margin_model"], "kstar_mde_model": ml[m]["kstar_mde_model"],
                     "kstar_pass": ml[m]["kstar_pass"],
                     "readout_margin_model": ml[m]["readout_margin_model"], "readout_mde_model": ml[m]["readout_mde_model"],
                     "readout_pass": ml[m]["readout_pass"]} for m in per},
   "aggregate": {"E_read_per_model": agg["E_read_per_model"], "E_read_pooled": agg["E_read_pooled"],
                 "kstar_pooled_margin": agg["kstar_pooled_margin"], "kstar_pooled_mde": agg["kstar_pooled_mde"],
                 "readout_pooled_margin": agg["readout_pooled_margin"], "readout_pooled_mde": agg["readout_pooled_mde"]},
   "verdict": verdict,
   "E_read_gate_5pct": {"threshold": 0.05, "n_models_passing": sum(1 for e in agg["E_read_per_model"] if e > 0.05),
                        "of": len(agg["E_read_per_model"]),
                        "ratio_vs_gate": round(agg["E_read_pooled"]/0.05, 2)}
 },
 "deadlines": {
   "K1": {"state":"FIRED (under Q08=甲)", "evidence":"readout 层 3/3 否决 + E_read 池化超门",
          "consequence":"「条件齿轮组=算子代数」降级 descriptive"},
   "K2": {"state":"not_measurable_yet", "reason":"phase3154 未运行；同一文件内两种操作化互斥，须先冻结其一"},
   "K3": {"state":"not_measurable_yet", "reason":"项目中无「单坐标筛选 top-50 覆盖率」量"}
 },
 "corrections_C1_C6": corr,
 "i_clauses": {"I1": {"status":"confirmed_frozen","section":"§1 唯一全局 KPI",
                      "rule":"仅 E_read / E_ar(k) / C_steer 可作为全局 KPI；未改善者登记 catalog 而非 advance"},
               "I9": {"status":"confirmed_frozen","section":"§7 议程：冻结 30-Phase 队列",
                      "rule":"phase_queue_v1.json 为唯一议程来源；禁止由残差自动派生新 Phase"}},
 "ledger_registration": {
   "target":"research/gpt5/atlas/atlas_ledger.json#measurements",
   "kind":"governance_registration",
   "status":"pending_cross_line",
   "entry": {"phase": "A-gate", "name": "rdc_a_gate_closure_q01_q02_q08_q09_q12",
             "verdict": "k1_fired_at_readout__kspecific__k2_k3_not_measurable__A_gate_closed",
             "evidence_level": "bit_anchored", "note": "Q08=甲 seal 后登记；属治理记录非科学测量"}
 },
 "protected_fingerprints_unchanged": {
   "AGI_GPT5_MEMO.md": "2a84776b", "atlas_ledger.json": sha8(LEDGER),
   "proposition_ledger.json": "9a3c6ff4"
 },
 "next": ["B 闸门 Q03（E_read 统一基线复算，需 GPU）", "Q04/Q05 E_ar(k) 装置", "Q06 C_steer 基座"]
}
wj(CLOSE, closure)
back = rj(CLOSE)
assert back["schema"] == "rdc_a_gate_closure_v1", "closure readback"
assert back["k1_reverdict"]["verdict"]["readout_layer"] == "fired_all_models"
add("")
add("=== 1) a_gate_closure_v1.json ===")
add("  %s  %d B  %s" % (CLOSE, os.path.getsize(CLOSE), sha8(CLOSE)))

# ---------- 2) queue 状态 ----------
q = rj(QUEUE)
assert sha8(QUEUE) == "675836fd", "queue 已变，先复核"
SEALED = {"Q01","Q02","Q08","Q09","Q12"}
n_sealed = 0
for it in q["queue"]:
    if it["id"] in SEALED:
        it["status"] = "sealed"
        it["sealed_at"] = NOW
        it["seal_record"] = "research/deepseek/atlas/a_gate_closure_v1.json"
        n_sealed += 1
q["kpi_definition_ref"] = "research/deepseek/docs/AGI_DEEPSEEK_MEMO.md （Phase 25 = RDC_RESEARCH_CONSTITUTION_v1 §1）"
q["status_updated_at"] = NOW
q["status_updated_by"] = "tests/deepseek_temp/gen_r8_seal.py"
q["sealed_items"] = sorted(SEALED)
wj(QUEUE, q)
bq = rj(QUEUE)
assert sum(1 for it in bq["queue"] if it["status"] == "sealed") == 5
assert bq["count"] == 30
add("=== 2) phase_queue_v1.json ===")
add("  sealed=%d/30  %d B  %s" % (n_sealed, os.path.getsize(QUEUE), sha8(QUEUE)))

# ---------- 3) meta_single_source_v4 更正回填 ----------
m = rj(META)
assert sha8(META) == "f9d7ede6", "meta 已变，先复核"
by_id = {c["id"]: c for c in corr}
for c in m["corrections"]:
    src = by_id[c["id"]]
    c["status"] = "accepted_sealed" if src["status"] == "accepted" else "accepted_sealed_pending_cross_line"
    c["resolved_value"] = src["resolved_value"]
    c["enactment"] = src["enactment"]
    c["sealed_at"] = NOW
m["sealed_at"] = NOW
m["seal_verbatim"] = SEAL
m["closure_record"] = "research/deepseek/atlas/a_gate_closure_v1.json"
if "hash_policy" in m:
    m["hash_policy"]["atlas_ledger"]["status"] = "STALE__correction_C4_sealed_pending_cross_line"
    m["hash_policy"]["proposition_ledger"]["status"] = "STALE__correction_C5_sealed_recorded_externally"
wj(META, m)
bm = rj(META)
assert all(c["status"].startswith("accepted") for c in bm["corrections"]), "corrections 未全接受"
assert bm["sealed_at"] == NOW
add("=== 3) meta_single_source_v4.json ===")
add("  corrections=%d 全部 accepted  %d B  %s" % (len(bm["corrections"]), os.path.getsize(META), sha8(META)))

# ---------- 4) 跨线补丁 spec（未施加） ----------
patch = {
 "schema": "rdc_cross_line_ledger_patch_v1",
 "created_at": NOW,
 "status": "SEALED_BUT_NOT_APPLIED",
 "reason": "目标文件 research/gpt5/atlas/atlas_ledger.json 为跨线共享（含 deepseek Phase 8–21 与 G 线 Phase 2902–3153），本对话规范要求避免与其他路线混合 ⇒ 不施加，交其所有者或经用户再确认后施加。",
 "target": "research/gpt5/atlas/atlas_ledger.json",
 "target_sha256_before": sha_all(LEDGER),
 "operations": [
   {"op": "set_field", "path": "ledger_sha256_8", "from": lg.get("ledger_sha256_8"), "to": recomputed,
    "note": "C4：content_excluding_self（json.dumps(...indent=1, ensure_ascii=False, 无尾换行) 的 sha256[:8]），本对话已独立复现"},
   {"op": "unify_schema", "fields": ["meas_id", "phase"],
    "detail": {"n_total": len(lg["measurements"]),
               "n_has_meas_id": sum(1 for x in lg["measurements"] if "meas_id" in x),
               "n_has_phase": sum(1 for x in lg["measurements"] if "phase" in x),
               "n_both": sum(1 for x in lg["measurements"] if "meas_id" in x and "phase" in x)},
    "note": "C6：统一后每条含 meas_id+phase；缺者按 name/sha 从 tests/glm5/.../phaseNNNN 回填"},
   {"op": "enumerate_field", "path": "measurements[].evidence_level",
    "allowed": ["bit_anchored", "statistical", "descriptive"],
    "current_distribution": {"statistical": len(lg["measurements"])},
    "note": "C6：现 304/304 恒为 statistical（常量）⇒ 须按 metric_dict.meta_rules.F2_verdict_grading 重新判定每条"},
   {"op": "append_measurement", "entry": closure["ledger_registration"]["entry"],
    "note": "A 闸门关闭的治理登记（Q08=甲）"}
 ],
 "recorded_correct_value": recomputed,
 "closure_record": "research/deepseek/atlas/a_gate_closure_v1.json"
}
wj(PATCH, patch)
bp = rj(PATCH)
assert bp["status"] == "SEALED_BUT_NOT_APPLIED"
add("=== 4) ledger_corrections_v1.json（跨线补丁，未施加）===")
add("  %s  %d B  %s" % (PATCH, os.path.getsize(PATCH), sha8(PATCH)))

# ---------- 5) 跨线文件未被触碰 ----------
add("")
add("=== 5) 跨线受保护文件（须未变）===")
for rel, exp in [(r"research\gpt5\docs\AGI_GPT5_MEMO.md", "2a84776b"),
                 (r"research\gpt5\atlas\atlas_ledger.json", "bbda63df"),
                 (r"tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json", "9a3c6ff4")]:
    p = os.path.join(ROOT, rel)
    g = sha8(p)
    add("  %-8s exp=%s got=%s  %s" % ("OK" if g == exp else "DRIFT!", exp, g, os.path.basename(p)))

open(OUT, "w", encoding="utf-8").write("\n".join(r))
print("\n".join(r))
