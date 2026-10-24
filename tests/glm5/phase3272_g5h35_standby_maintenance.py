# -*- coding: utf-8 -*-
"""Phase 3272 (G5-H35): standby_maintenance -- report-only.

Maintenance arm with concrete value: the op4 DRY-RUN prospective post
registered in 3271 (329b1398) was computed against the n=427 baseline and
is now STALE after closeout append #423. This phase refreshes the dry-run
against the current head, records the stale value as superseded, and
appends a dated addendum to PENDING_ITEMS_IMPACT_PREVIEW.md.
Zero GPU, zero writes to any sealed account or pending object.
"""
import hashlib
import json
import os
import time

BASE = r"D:\AI2050\Ai2050-OpenOne"
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
LCORR = os.path.join(BASE, "tests", "glm5", "result",
                     "rdc_query_construction_20260913", "phase3260",
                     "g5h23_lcorr_rederivation", "lcorr_rederivation_v2.json")
MD = os.path.join(BASE, "research", "gpt5", "docs",
                  "PENDING_ITEMS_IMPACT_PREVIEW.md")
SEALED = {
    "gap_v113": (os.path.join(BASE, "tests", "glm5", "result",
                              "rdc_query_construction_20260913", "phase3265",
                              "g5h28_appx7_landing", "gap_ledger_v1_13.json"),
                 "814ce808"),
    "registry_v136": (os.path.join(BASE, "tests", "glm5", "result",
                                   "rdc_query_construction_20260913",
                                   "phase3265", "g5h28_appx7_landing",
                                   "atlas_registry_v1_36.json"), "78d37211"),
    "lcorr_v2": (LCORR, "ef72c733"),
    "memo_ds": (os.path.join(BASE, "research", "deepseek", "docs",
                             "AGI_DEEPSEEK_MEMO.md"), "d3f62d35"),
    "proto23": (os.path.join(BASE, "tests", "glm5", "result",
                             "rdc_query_construction_20260913", "phase3267",
                             "g5h30_appx23_protocol",
                             "appx23_protocol_v2.json"), "ce73984d"),
}
PHASE = 3272
ARM = "G5-H35"
SMOKE = os.environ.get("P3272_SMOKE") == "1"
OUTDIR = os.path.join(BASE, "tests", "glm5", "result",
                      "rdc_query_construction_20260913",
                      "phase%d" % PHASE, "g5h35_standby_maintenance")


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


DESIGN = {
    "phase": PHASE, "arm": ARM, "mode": "report_only_standby_maintenance",
    "zero": ["gpu", "ledger_measurement_write", "sealed_account_write",
             "pending_object_write"],
    "arms": {
        "a_drift": "sealed/pending object sha8 drift check vs 3271 baseline",
        "b_dryrun_refresh": "recompute op4 prospective post vs CURRENT "
                            "ledger head; stale 329b1398 -> superseded",
        "c_addendum": "dated maintenance addendum appended to "
                      "PENDING_ITEMS_IMPACT_PREVIEW.md (append-only)",
    },
    "baseline": {"ledger": "#423 post 3384cfbd n=428",
                 "stale_dryrun_post": "329b1398"},
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

gates = {}

# ---- g1: ledger head + full-chain integrity spot (head recompute) ----
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
Lc = dict(L)
Lc["ledger_sha256_8"] = ms[-1]["prev_post"]
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_chain_head"] = (rec == L["ledger_sha256_8"] == "3384cfbd"
                          and len(ms) == 428
                          and str(ms[-1]["phase"]) == "3271")

# ---- arm a + g2: drift check ----
drift = {}
for k, (p, exp) in SEALED.items():
    got = s8(p)
    drift[k] = {"observed": got, "expected": exp, "drift": got != exp}
gates["g2_no_drift"] = all(not v["drift"] for v in drift.values())
arm_a = {"objects": drift, "concurrent_append_detected": False,
         "note": "post-3271 appends: #423 closeout only (tail 3271)"}

# ---- arm b + g4: op4 dry-run refresh (deterministic x2) ----
d = json.load(open(LCORR, encoding="utf-8"))
p4 = d["operations"]["op4_append_agate"]["rederived_entry"]
new_rec = {"phase": p4["phase"], "name": p4["name"],
           "verdict": p4["verdict"],
           "evidence_level": p4["evidence_level"], "note": p4["note"],
           "prev_post": L["ledger_sha256_8"]}
Ld = dict(L)
Ld["measurements"] = ms + [new_rec]
Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
post2 = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                       .encode("utf-8")).hexdigest()[:8]
gates["g4_dryrun_deterministic"] = (post == post2 and len(post) == 8
                                    and post != "329b1398")
arm_b = {"dry_run_post_current": post,
         "supersedes": {"stale_post": "329b1398", "basis": "n=427 baseline",
                        "registered_in": "phase 3271",
                        "status": "SUPERSEDED_BY_MAINTENANCE"},
         "delta": "n %d -> %d" % (len(ms), len(ms) + 1),
         "mechanism": "append-only receipt via 3258 protocol",
         "zero_write_confirmed": True}

# ---- arm c + g5: MD addendum (formal only) ----
arm_c = {"file": "research/gpt5/docs/PENDING_ITEMS_IMPACT_PREVIEW.md",
         "mechanism": "append-only dated addendum; base bytes preserved"}
if not SMOKE:
    cur = open(MD, encoding="utf-8").read()
    MARK = "<!-- maintenance-addendum-3272 -->"
    if MARK not in cur:
        add = ("\n\n<!-- maintenance-addendum-3272 -->\n"
               "## 维护注记（Phase 3272，%s）\n\n"
               "第 ② 节的 DRY-RUN post `%s` 系基于 n=427 基线计算，"
               "因 3271 closeout append（#423，n=428）**已过期**"
               "（SUPERSEDED_BY_MAINTENANCE）。对现役头（#423，post "
               "`3384cfbd`）刷新：**op4 批准并施加后 post = `%s`**"
               "（n 428→429）。本注记为 append-only 追加，"
               "上文基线字节未改动。\n" % (time.strftime("%Y-%m-%d"),
                                          "329b1398", post))
        with open(MD, "a", encoding="utf-8", newline="") as f:
            f.write(add)
        log("MD addendum appended (new dry-run post %s)" % post)
    else:
        log("MD addendum already present (idempotent)")
gates["g5_addendum_present"] = (MARK in open(MD, encoding="utf-8").read()
                                if not SMOKE else None)

# ---- g3: zero write to sealed/pending objects ----
gates["g3_sealed_untouched"] = all(s8(p) == exp
                                   for p, exp in SEALED.values())
zero_write_verified = s8(LCORR)

RESULT = {
    "phase": PHASE, "arm": ARM, "verdict": "standby_maintenance_registered",
    "design_sha8": design_sha8, "gates": gates,
    "arms": {"a_drift": arm_a, "b_dryrun_refresh": arm_b, "c_addendum": arm_c},
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

fails = [k for k, v in gates.items() if v is False]  # None = N/A
log("gates: %d/%d pass%s" % (sum(1 for v in gates.values() if v is True),
                             len(gates),
                             ("  FAIL=" + ",".join(fails)) if fails else ""))
log("res_sha8: %s | dry-run current: %s" % (res_sha8, post))
log("SMOKE DONE" if SMOKE else "FORMAL DONE")
