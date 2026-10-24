# -*- coding: utf-8 -*-
"""Phase 3273 (G5-H36): minimal_standby_refresh -- report-only, MINIMAL.

Repetitive by design (declared in 3272): drift check + chain head only.
No MD, no new analysis. Zero GPU, zero writes to sealed/pending objects.
"""
import hashlib
import json
import os
import time

BASE = r"D:\AI2050\Ai2050-OpenOne"
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
SEALED = {
    "gap_v113": (os.path.join(BASE, "tests", "glm5", "result",
                              "rdc_query_construction_20260913", "phase3265",
                              "g5h28_appx7_landing", "gap_ledger_v1_13.json"),
                 "814ce808"),
    "registry_v136": (os.path.join(BASE, "tests", "glm5", "result",
                                   "rdc_query_construction_20260913",
                                   "phase3265", "g5h28_appx7_landing",
                                   "atlas_registry_v1_36.json"), "78d37211"),
    "lcorr_v2": (os.path.join(BASE, "tests", "glm5", "result",
                              "rdc_query_construction_20260913", "phase3260",
                              "g5h23_lcorr_rederivation",
                              "lcorr_rederivation_v2.json"), "ef72c733"),
    "memo_ds": (os.path.join(BASE, "research", "deepseek", "docs",
                             "AGI_DEEPSEEK_MEMO.md"), "d3f62d35"),
    "proto23": (os.path.join(BASE, "tests", "glm5", "result",
                             "rdc_query_construction_20260913", "phase3267",
                             "g5h30_appx23_protocol",
                             "appx23_protocol_v2.json"), "ce73984d"),
}
PHASE, ARM = 3273, "G5-H36"
SMOKE = os.environ.get("P3273_SMOKE") == "1"
OUTDIR = os.path.join(BASE, "tests", "glm5", "result",
                      "rdc_query_construction_20260913",
                      "phase%d" % PHASE, "g5h36_minimal_refresh")


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


DESIGN = {"phase": PHASE, "arm": ARM, "mode": "minimal_standby_refresh",
          "repetitive_declared": True,
          "zero": ["gpu", "ledger_measurement_write", "sealed_account_write",
                   "pending_object_write"],
          "baseline": "#424 post 3f210467 n=429"}
design_sha8 = hashlib.sha256(json.dumps(
    DESIGN, ensure_ascii=False, sort_keys=True, indent=1)
    .encode("utf-8")).hexdigest()[:8]
os.makedirs(OUTDIR, exist_ok=True)
exe_name = "execution_smoke.json" if SMOKE else "execution.json"
exe_p = os.path.join(OUTDIR, exe_name)
if os.path.exists(exe_p):
    assert json.load(open(exe_p, encoding="utf-8"))["design_sha8"] == \
        design_sha8, "DESIGN DRIFT"
else:
    with open(exe_p, "w", encoding="utf-8", newline="") as f:
        json.dump({"design": DESIGN, "design_sha8": design_sha8,
                   "smoke": SMOKE,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, ensure_ascii=False, indent=1)
with open(os.path.join(OUTDIR, "design_sha8.txt"), "w",
          encoding="utf-8", newline="") as f:
    f.write(design_sha8)

gates = {}
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
Lc = dict(L)
Lc["ledger_sha256_8"] = ms[-1]["prev_post"]
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
gates["g1_chain_head"] = (rec == L["ledger_sha256_8"] == "3f210467"
                          and len(ms) == 429
                          and str(ms[-1]["phase"]) == "3272")
gates["g2_no_drift"] = all(s8(p) == exp for p, exp in SEALED.values())
gates["g3_no_concurrent"] = str(ms[-1]["phase"]) == "3272"

RESULT = {"phase": PHASE, "arm": ARM,
          "verdict": "minimal_standby_refresh_registered",
          "design_sha8": design_sha8, "gates": gates,
          "zero_write_verified": s8(SEALED["lcorr_v2"][0]),
          "baseline": {"ledger_n": len(ms),
                       "ledger_post": L["ledger_sha256_8"]}}
if SMOKE:
    RESULT["smoke"] = True
res_blob = json.dumps(RESULT, ensure_ascii=False, sort_keys=True,
                      indent=1).encode("utf-8")
RESULT["res_sha8"] = hashlib.sha256(res_blob).hexdigest()[:8]
with open(os.path.join(OUTDIR, "result.json"), "w", encoding="utf-8",
          newline="") as f:
    json.dump(RESULT, f, ensure_ascii=False, sort_keys=True, indent=1)
fails = [k for k, v in gates.items() if v is False]
print("gates: %d/%d%s | freeze %s | res %s | %s"
      % (sum(1 for v in gates.values() if v is True), len(gates),
         (" FAIL=" + ",".join(fails)) if fails else "",
         design_sha8, RESULT["res_sha8"],
         "SMOKE" if SMOKE else "FORMAL"))
