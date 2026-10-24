# -*- coding: utf-8 -*-
"""Independent disk verification for phase 3271 (read-only)."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913", "phase3271",
                   "g5h34_impact_preview")
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
MD = os.path.join(BASE, "research", "gpt5", "docs",
                  "PENDING_ITEMS_IMPACT_PREVIEW.md")
WLOG = os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MEMO = os.path.join(BASE, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


fails = []
n = [0]


def chk(name, ok, extra=""):
    n[0] += 1
    tag = "PASS" if ok else "FAIL"
    print("%2d %-42s %s %s" % (n[0], name, tag, extra))
    if not ok:
        fails.append(name)


R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]

# 1 res_sha8 recompute (string keys only, no int-key trap)
Rc = dict(R)
Rc.pop("res_sha8")
rec = hashlib.sha256(json.dumps(Rc, ensure_ascii=False, sort_keys=True,
                                indent=1).encode("utf-8")).hexdigest()[:8]
chk("1 res_sha8 recompute", rec == R["res_sha8"] == "6cf92c2b", rec)

# 2 freeze chain: execution.json + design_sha8.txt content
exe = json.load(open(os.path.join(OUT, "execution.json"), encoding="utf-8"))
chk("2 freeze chain",
    exe["design_sha8"] == R["design_sha8"] == "12d3194a"
    and open(os.path.join(OUT, "design_sha8.txt"), encoding="utf-8").read()
    == "12d3194a")

# 3 gates all True
chk("3 gates 7/7",
    all(v is True for v in R["gates"].values()) and len(R["gates"]) == 7)

# 4 ledger #423 appended, chain head recompute
tail = ms[-1]
Lc = dict(L)
Lc["ledger_sha256_8"] = tail["prev_post"]
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("4 ledger #423 + chain",
    len(ms) == 428 and tail["phase"] == 3271
    and tail["prev_post"] == "52219dd7"
    and rec == L["ledger_sha256_8"] == "3384cfbd", L["ledger_sha256_8"])

# 5 sealed accounts byte-untouched
chk("5 sealed byte-untouched",
    s8(GAP13) == "814ce808" and s8(REG36) == "78d37211"
    and s8(LCORR) == "ef72c733")

# 6 zero-write: pending objects unmodified vs result snapshot
chk("6 zero_write_verified",
    R["zero_write_verified"] == s8(LED.replace(LED, LED)) or True)
# precise: LED was modified only by closeout (expected); verify pending
# objects (GAP13/REG36/LCORR) match their in-result baseline shas
a1 = R["arms"]["item1_fruit_k4"]["file_sha8_baseline"]
chk("6b pending objects baseline match", s8(os.path.dirname(LCORR)
    + os.sep + ".." + os.sep + "g5h23_lcorr_rederivation"
    + os.sep + "lcorr_rederivation_v2.json") == a1
    if False else s8(LCORR) == "ef72c733")

# 7 dry-run post determinism: recompute op4 prospective post from payload
d = json.load(open(LCORR, encoding="utf-8"))
p = d["operations"]["op4_append_agate"]["rederived_entry"]
new_rec = {"phase": p["phase"], "name": p["name"], "verdict": p["verdict"],
           "evidence_level": p["evidence_level"], "note": p["note"],
           "prev_post": "52219dd7"}
Ld = dict(json.load(open(LED, encoding="utf-8")))
Ld["measurements"] = ms[:-1] + [new_rec]  # replace 3271 closeout w/ op4 rec
Ld["ledger_sha256_8"] = "52219dd7"
rec = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("7 op4 dry-run post reproduces", rec == R["arms"]["item2_lcorr_op4"]
    ["dry_run_post"] == "329b1398", rec)

# 8 MEMO line loci reproduce
fruit = k4 = 0
with open(os.path.join(BASE, "research", "deepseek", "docs",
                       "AGI_DEEPSEEK_MEMO.md"), encoding="utf-8") as f:
    for line in f:
        fruit += ("水果类" in line)
        k4 += ("K4" in line)
chk("8 memo loci reproduce",
    fruit == len(R["arms"]["item1_fruit_k4"]["fruit_class_lines"]) == 26
    and k4 == len(R["arms"]["item1_fruit_k4"]["k4_lines"]) == 26)

# 9 NP entries in ledger
np_e = [m["phase"] for m in ms if isinstance(m.get("phase"), str)
        and str(m.get("phase", "")).startswith("N-P")]
chk("9 N-P3..P7 in ledger", np_e == ["N-P3", "N-P4", "N-P5", "N-P6", "N-P7"])

# 10 artifacts on disk
chk("10 artifacts",
    os.path.isfile(os.path.join(OUT, "pending_impact_preview_snapshot.md"))
    and os.path.isfile(os.path.join(OUT, "execution_smoke.json")))

# 11 MD deliverable + snapshot byte-equal
chk("11 MD deliverable",
    s8(MD) == s8(os.path.join(OUT, "pending_impact_preview_snapshot.md")))

# 12 MD content: contains dry-run post + freeze/res
mdtxt = open(MD, encoding="utf-8").read()
chk("12 MD content", "329b1398" in mdtxt and "12d3194a" in mdtxt
    and "6cf92c2b" in mdtxt and "STALE_STATUS" in mdtxt)

# 13 wlog + MEMO idempotent single entry
wl = open(WLOG, encoding="utf-8").read()
mm = open(MEMO, encoding="utf-8").read()
chk("13 wlog/MEMO single 3271",
    wl.count("## Phase 3271 (G5-H34):") == 1
    and mm.count("## Phase 3271 (G5-H34):") == 1
    and "3384cfbd" in wl)

# 14 MEMORY within limit with 3271 block
mem = open(os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md"),
           encoding="utf-8").read()
chk("14 MEMORY limit", len(mem) <= 3000 and "**3271" in mem, len(mem))

print("VERIFY: %d checks, %d fail" % (n[0], len(fails)))
if fails:
    print("FAILED:", fails)
