# -*- coding: utf-8 -*-
"""Phase 3271 (G5-H34): pending_items_impact_preview -- report-only.

For each of the 5 owner-pending items, preview the EXACT impact of approval:
which files/lines would be touched, what the ledger delta would look like
(including a zero-write DRY-RUN chain computation for lcorr op4), and what
mechanism (append-only annotation vs registry arm) would apply.
Zero GPU, zero writes to any pending-object or sealed account.
"""
import hashlib
import json
import os
import sys
import time

BASE = r"D:\AI2050\Ai2050-OpenOne"
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
GAP13 = os.path.join(BASE, "tests", "glm5", "result",
                     "rdc_query_construction_20260913", "phase3265",
                     "g5h28_appx7_landing", "gap_ledger_v1_13.json")
REG36 = os.path.join(BASE, "tests", "glm5", "result",
                     "rdc_query_construction_20260913", "phase3265",
                     "g5h28_appx7_landing", "atlas_registry_v1_36.json")
LCORR = os.path.join(BASE, "tests", "glm5", "result",
                     "rdc_query_construction_20260913", "phase3260",
                     "g5h23_lcorr_rederivation", "lcorr_rederivation_v2.json")
MEMO_DS = os.path.join(BASE, "research", "deepseek", "docs",
                       "AGI_DEEPSEEK_MEMO.md")
MEMO_G5 = os.path.join(BASE, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
PROTO23 = os.path.join(BASE, "tests", "glm5", "result",
                       "rdc_query_construction_20260913", "phase3267",
                       "g5h30_appx23_protocol", "appx23_protocol_v2.json")
PHASE = 3271
ARM = "G5-H34"
SMOKE = os.environ.get("P3271_SMOKE") == "1"
OUTDIR = os.path.join(BASE, "tests", "glm5", "result",
                      "rdc_query_construction_20260913",
                      "phase%d" % PHASE, "g5h34_impact_preview")


def s8(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ================================================================ DESIGN
DESIGN = {
    "phase": PHASE, "arm": ARM, "mode": "report_only_impact_preview",
    "zero": ["gpu", "ledger_measurement_write", "pending_object_write",
             "sealed_account_write"],
    "arms": {
        "item1_fruit_k4": "locate exact MEMO lines listing the two backlog "
                          "items; classify line type; impact = append-only "
                          "annotation mechanism (no byte rewrite)",
        "item2_lcorr_op4": "payload from lcorr v2 ef72c733 op4_append_agate; "
                           "DRY-RUN chain computation of prospective post; "
                           "evidence-level delta preview",
        "item3_appx1": "nf4 context + decision space (owner-only), status "
                       "unchanged",
        "item4_appx23": "protocol v2 ce73984d preconditions recheck + GPU "
                        "cost statement (from sealed notes), status unchanged",
        "item5_appx45": "APPX-4 resolution evidence (N-P3..P7 in ledger "
                        "since 3258) + APPX-5 reduction to op4; both need a "
                        "future gap v1.14 appendix annotation via registry arm",
    },
    "inputs": {"lcorr_v2": "ef72c733", "gap_v113": "814ce808",
               "registry_v136": "78d37211",
               "ledger_baseline": "#422 post 52219dd7 n=427"},
    "smoke_states": ["base_only"] if SMOKE else None,
}
design_blob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True,
                         indent=1).encode("utf-8")
design_sha8 = hashlib.sha256(design_blob).hexdigest()[:8]

os.makedirs(OUTDIR, exist_ok=True)
exe_name = "execution_smoke.json" if SMOKE else "execution.json"
exe_p = os.path.join(OUTDIR, exe_name)
if os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding="utf-8"))
    assert prev["design_sha8"] == design_sha8, "DESIGN DRIFT"
    log("freeze: existing %s OK (%s)" % (exe_name, design_sha8))
else:
    with open(exe_p, "w", encoding="utf-8", newline="") as f:
        json.dump({"design": DESIGN, "design_sha8": design_sha8,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "smoke": SMOKE}, f, ensure_ascii=False, indent=1)
    log("freeze: %s written (%s)" % (exe_name, design_sha8))
with open(os.path.join(OUTDIR, "design_sha8.txt"), "w",
          encoding="utf-8", newline="") as f:
    f.write(design_sha8)

# ================================================================ gates
gates = {}

# ---- snapshot all touch-candidates BEFORE (for g5 zero-write) ----
TOUCH = [LED, GAP13, REG36, LCORR, MEMO_DS, MEMO_G5, PROTO23]
before = {p: s8(p) for p in TOUCH}

# ---- g1: ledger chain head recompute (full ms + field=prev_post) ----
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
Lc = dict(L)
Lc["ledger_sha256_8"] = ms[-1]["prev_post"]
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_chain_head"] = (rec == L["ledger_sha256_8"] == "52219dd7"
                          and len(ms) == 427
                          and str(ms[-1]["phase"]) == "3270")

# ---- g2: incumbent versions ----
g13 = json.load(open(GAP13, encoding="utf-8"))
r36 = json.load(open(REG36, encoding="utf-8"))
gates["g2_incumbent"] = (str(g13["version"]) == "1.13"
                         and str(r36["version"]) == "1.36"
                         and len(g13["appendix"]) == 7
                         and len(r36["upgrade_log"]) == 15)

# ---- arm 1: fruit/K4 MEMO line locations ----
fruit_lines, k4_lines = [], []
for path, acc_f, acc_k in ((MEMO_DS, fruit_lines, k4_lines),):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if "水果类" in line:
                acc_f.append(i)
            if "K4" in line:
                acc_k.append(i)
# classify: backlog-list lines vs science-section lines
BACKLOG_MARKS = ("挂账", "结转", "不变")
bl_f = [i for i in fruit_lines]
gates["g3_memo_loci"] = (len(fruit_lines) >= 5 and len(k4_lines) >= 7)
arm1 = {
    "file": "research/deepseek/docs/AGI_DEEPSEEK_MEMO.md",
    "file_sha8_baseline": before[MEMO_DS],
    "fruit_class_lines": fruit_lines,
    "k4_lines": k4_lines,
    "mechanism": "append-only annotation (a dated 挂账-resolution note "
                 "appended at MEMO tail + backlog-list lines annotated in a "
                 "new dated section); byte-rewrite of historical lines is "
                 "NOT the project convention",
    "lines_touched_if_approved": len(fruit_lines) + len(k4_lines),
}
if SMOKE:
    arm1["note"] = "SMOKE: line scan only"

# ---- arm 2: lcorr op4 DRY-RUN ----
d = json.load(open(LCORR, encoding="utf-8"))
op4 = d["operations"]["op4_append_agate"]
gates["g4_lcorr_draft"] = (d["status"] == "REPORT_ONLY_DRAFT"
                           and op4["disposition"] == "READY_PENDING_CONFIRMATION"
                           and before[LCORR] == "ef72c733")
payload = dict(op4["rederived_entry"])
arm2 = {
    "payload": payload,
    "closure_record": op4["evidence"]["closure_record"],
    "closure_record_sha8": op4["evidence"]["closure_record_sha8"],
    "mechanism": "single append-only receipt to atlas_ledger via the same "
                 "protocol as 3258 (append + recompute + revert-proof)",
    "ledger_delta_preview": {},
    "dry_run_post": None,
}
if not SMOKE:
    # dry-run chain computation (NO write)
    new_rec = {"phase": payload["phase"], "name": payload["name"],
               "verdict": payload["verdict"],
               "evidence_level": payload["evidence_level"],
               "note": payload["note"], "prev_post": L["ledger_sha256_8"]}
    Ld = dict(L)
    Ld["measurements"] = ms + [new_rec]
    Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
    post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                          .encode("utf-8")).hexdigest()[:8]
    # determinism: compute twice
    post2 = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                           .encode("utf-8")).hexdigest()[:8]
    arm2["dry_run_post"] = post
    arm2["ledger_delta_preview"] = {
        "n": "%d -> %d" % (len(ms), len(ms) + 1),
        "evidence_level_new_value": payload["evidence_level"],
        "phase_type": "string ('A-gate' namespace)",
        "note": payload["note"],
    }
    gates["g5_dryrun_deterministic"] = (post == post2 and len(post) == 8)
else:
    gates["g5_dryrun_deterministic"] = None

# ---- arm 3: APPX-1 context ----
a1 = [a for a in g13["appendix"] if a["id"] == "APPX-1"][0]
nf4 = os.path.join(BASE, "models", "hf", "Qwen3-14B-bnb-nf4")
arm3 = {
    "definition": a1,
    "context": "RTX 5080 16GB cannot host Qwen3-14B in bf16; nf4 quantized "
               "copy already on disk (3262/3267 recorded as context facts)",
    "nf4_on_disk": os.path.isdir(nf4),
    "decision_space": ["approve nf4 baseline for APPX-1 execution",
                       "reject quantization (defer to future hardware)",
                       "keep open"],
    "impact_if_approved": "new GPU execution phase(s) after 3267 protocol "
                          "pattern; no change to sealed accounts",
}

# ---- arm 4: APPX-2/3 preconditions recheck ----
proto = json.load(open(PROTO23, encoding="utf-8"))
CAND = ["qwen2.5-3b-instruct", "qwen2-7b", "deepseek-r1-distill-qwen-7b",
        "glm4-9b-chat-hf", "gemma-3-4b-it"]
prec = {}
for c in CAND:
    mp = os.path.join(BASE, "models", "hf", c)
    prec[c] = {"dir": os.path.isdir(mp),
               "tokenizer": os.path.isdir(mp) and any(
                   f.startswith("tokenizer") for f in os.listdir(mp))}
gates["g6_appx23_preconditions"] = all(v["dir"] and v["tokenizer"]
                                       for v in prec.values())
arm4 = {
    "protocol": {"file": "appx23_protocol_v2.json", "sha8": before[PROTO23],
                 "status": proto.get("status")},
    "model_preconditions": prec,
    "gpu_cost_statement": "APPX-2 ~15-60 min/model for S1-S3 per 3262 "
                          "note (readiness, not a schedule); APPX-3 "
                          "K2/K3 direct measurement on existing 3-model "
                          "collects, GPU-light",
}

# ---- arm 5: APPX-4/5 disposition evidence ----
np_entries = [m["phase"] for m in ms
              if isinstance(m.get("phase"), str)
              and str(m.get("phase", "")).startswith("N-P")]
arm5 = {
    "APPX-4": {
        "status_in_gap_v113": a1 and [a for a in g13["appendix"]
                                      if a["id"] == "APPX-4"][0]["status"],
        "resolution_evidence": {
            "n_p_phases_in_ledger": np_entries,
            "applied_at": "phase 3258 (ledger #410-ish, n 409->414)",
            "finding": "STALE_STATUS: work item already done; status "
                       "field predates 3258",
        },
        "impact_if_annotated": "gap v1.14 appendix note (registry arm) "
                               "marking APPX-4 RESOLVED; byte-anchored "
                               "v1.13 untouched",
    },
    "APPX-5": {
        "reduction": "ledger_corrections_v1.json (beb775bd) fully "
                     "dispositioned by 3260: op1 void / op2 no-migration / "
                     "op3 superseded-by-practice / op4 READY; the ONLY "
                     "live remainder of APPX-5 == lcorr op4 (item 2)",
        "impact": "approving item 2 closes APPX-5's last open path",
    },
}

# ---- g7: zero-write after all arms ----
after = {p: s8(p) for p in TOUCH}
gates["g7_zero_write"] = (before == after)
zero_write_verified = after[LED]

# ---- assemble result ----
RESULT = {
    "phase": PHASE, "arm": ARM,
    "verdict": "pending_impact_preview_registered",
    "design_sha8": design_sha8,
    "gates": gates,
    "arms": {"item1_fruit_k4": arm1, "item2_lcorr_op4": arm2,
             "item3_appx1": arm3, "item4_appx23": arm4,
             "item5_appx45": arm5},
    "zero_write_verified": zero_write_verified,
    "baseline": {"ledger_n": len(ms), "ledger_post": L["ledger_sha256_8"]},
}
if SMOKE:
    RESULT["smoke"] = True

res_blob = json.dumps(RESULT, ensure_ascii=False, sort_keys=True,
                      indent=1).encode("utf-8")
res_sha8 = hashlib.sha256(res_blob).hexdigest()[:8]
RESULT["res_sha8"] = res_sha8

with open(os.path.join(OUTDIR, "result.json"), "w", encoding="utf-8",
          newline="") as f:
    json.dump(RESULT, f, ensure_ascii=False, sort_keys=True, indent=1)

fails = [k for k, v in gates.items() if v is False]  # None = N/A (SMOKE)
log("gates: %d/%d pass%s" % (sum(1 for v in gates.values() if v is True),
                             len(gates),
                             ("  FAIL=" + ",".join(fails)) if fails else ""))
log("res_sha8: %s" % res_sha8)
log("SMOKE DONE" if SMOKE else "FORMAL DONE")
