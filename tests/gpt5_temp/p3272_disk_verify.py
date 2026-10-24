# -*- coding: utf-8 -*-
"""Independent disk verification for phase 3272 (read-only)."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913", "phase3272",
                   "g5h35_standby_maintenance")
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
MD = os.path.join(BASE, "research", "gpt5", "docs",
                  "PENDING_ITEMS_IMPACT_PREVIEW.md")
WLOG = os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MEMO = os.path.join(BASE, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
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


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


fails = []
n = [0]


def chk(name, ok, extra=""):
    n[0] += 1
    print("%2d %-40s %s %s" % (n[0], name, "PASS" if ok else "FAIL", extra))
    if not ok:
        fails.append(name)


R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]

# 1 res_sha8 recompute
Rc = dict(R)
Rc.pop("res_sha8")
rec = hashlib.sha256(json.dumps(Rc, ensure_ascii=False, sort_keys=True,
                                indent=1).encode("utf-8")).hexdigest()[:8]
chk("1 res_sha8 recompute", rec == R["res_sha8"] == "06293923", rec)

# 2 freeze chain
exe = json.load(open(os.path.join(OUT, "execution.json"), encoding="utf-8"))
chk("2 freeze chain",
    exe["design_sha8"] == R["design_sha8"] == "b66ca350"
    and open(os.path.join(OUT, "design_sha8.txt"),
             encoding="utf-8").read() == "b66ca350")

# 3 gates 5/5
chk("3 gates 5/5",
    all(v is True for v in R["gates"].values()) and len(R["gates"]) == 5)

# 4 ledger #424 + chain head recompute
tail = ms[-1]
Lc = dict(L)
Lc["ledger_sha256_8"] = tail["prev_post"]
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("4 ledger #424 + chain",
    len(ms) == 429 and tail["phase"] == 3272
    and tail["prev_post"] == "3384cfbd"
    and rec == L["ledger_sha256_8"] == "3f210467", L["ledger_sha256_8"])

# 5 sealed/pending objects byte-untouched
chk("5 sealed untouched",
    all(s8(p) == exp for p, exp in SEALED.values()))

# 6 dry-run refresh reproduces against current head
d = json.load(open(SEALED["lcorr_v2"][0], encoding="utf-8"))
p4 = d["operations"]["op4_append_agate"]["rederived_entry"]
new_rec = {"phase": p4["phase"], "name": p4["name"],
           "verdict": p4["verdict"],
           "evidence_level": p4["evidence_level"], "note": p4["note"],
           "prev_post": "3384cfbd"}
Ld = dict(L)
Ld["measurements"] = ms[:-1] + [new_rec]  # swap 3272 rec for op4 rec
Ld["ledger_sha256_8"] = "3384cfbd"
rec = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("6 dry-run refresh reproduces",
    rec == R["arms"]["b_dryrun_refresh"]["dry_run_post_current"]
    == "d165ee30", rec)

# 7 MD addendum single, append-only (base head bytes intact)
mdtxt = open(MD, encoding="utf-8").read()
chk("7 MD addendum",
    mdtxt.count("maintenance-addendum-3272") == 1
    and "d165ee30" in mdtxt and "SUPERSEDED_BY_MAINTENANCE" in mdtxt
    and mdtxt.index("d165ee30") > mdtxt.index("329b1398"))
# base head (3271 content) still starts the file
chk("7b MD base intact",
    mdtxt.startswith("# 待决项影响预览（Phase 3271，report-only）"))

# 8 stale value recorded as superseded in result
chk("8 stale recorded",
    R["arms"]["b_dryrun_refresh"]["supersedes"]["stale_post"] == "329b1398")

# 9 wlog/MEMO single 3272 entries
wl = open(WLOG, encoding="utf-8").read()
mm = open(MEMO, encoding="utf-8").read()
chk("9 wlog/MEMO single 3272",
    wl.count("## Phase 3272 (G5-H35):") == 1
    and mm.count("## Phase 3272 (G5-H35):") == 1
    and "3f210467" in wl)

# 10 MEMORY limit
mem = open(os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md"),
           encoding="utf-8").read()
chk("10 MEMORY limit", len(mem) <= 3000 and "**3272" in mem, len(mem))

print("VERIFY: %d checks, %d fail" % (n[0], len(fails)))
if fails:
    print("FAILED:", fails)
