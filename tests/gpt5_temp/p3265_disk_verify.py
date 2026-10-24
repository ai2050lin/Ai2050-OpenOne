# -*- coding: utf-8 -*-
"""Independent read-only disk verification for phase 3265 closeout."""
import hashlib
import json
import os

ROOT = r"D:\AI2050\Ai2050-OpenOne"
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3265", "g5h28_appx7_landing")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
G12 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")
R35 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "atlas_registry_v1_35.json")
P3263_RES = os.path.join(RBASE, "phase3263", "g5h26_p_traj", "result.json")
P3264_RES = os.path.join(RBASE, "phase3264", "g5h27_traj_second_order", "result.json")

FREEZE = "84fd1381"
RES_SHA = "18437872"
POST = "ec34b29b"
CHAIN_PREV = "113e6af3"
G13_SHA = "814ce808"
R36_SHA = "78d37211"

fails = []
oks = 0


def chk(name, cond, extra=""):
    global oks
    if cond:
        oks += 1
        print("PASS", name)
    else:
        fails.append(name)
        print("FAIL", name, extra)


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))

# 1 result verdict/hashes/gates
chk("1 result", R["verdict"] == "appx7_landed" and R["gate_all"] is True
    and R["design_sha8"] == FREEZE and R["res_sha8"] == RES_SHA
    and R["design_sha8"] == open(os.path.join(OUT, "design_sha8.txt"),
                                 encoding="utf-8").read()
    and len(R["gates"]) == 6 and all(R["gates"].values()))

# 2 res_sha8 recompute
r2 = {k: v for k, v in R.items() if k != "res_sha8"}
rec = hashlib.sha256(json.dumps(r2, sort_keys=True, ensure_ascii=False)
                     .encode("utf-8")).hexdigest()[:8]
chk("2 res_sha8 recompute", rec == RES_SHA, rec)

# 3 landed files: version chain + APPX-7 entry + upgrade_log
g13 = json.load(open(os.path.join(OUT, "gap_ledger_v1_13.json"), encoding="utf-8"))
r36 = json.load(open(os.path.join(OUT, "atlas_registry_v1_36.json"), encoding="utf-8"))
chk("3 landed files", s8(os.path.join(OUT, "gap_ledger_v1_13.json")) == G13_SHA
    and s8(os.path.join(OUT, "atlas_registry_v1_36.json")) == R36_SHA
    and g13["version"] == "1.13" and len(g13["appendix"]) == 7
    and g13["appendix"][-1]["id"] == "APPX-7"
    and r36["version"] == "1.36" and len(r36["upgrade_log"]) == 15
    and r36["upgrade_log"][-1]["phase"] == 3265
    and "updated" not in r36)

# 4 APPX-7 entry content + anchors (independent read)
a7 = g13["appendix"][-1]
chk("4 APPX-7 content", a7["anchor"] == "p3263+p3264"
    and a7["anchor_sha8"] == "f223a9a0"
    and a7["status"] == "appendix_recorded"
    and "15.98" in a7["note"] and "30–40×" in a7["note"])

# 5 pointer-diff: gaps/features/failures/census/taxonomy untouched
g12 = json.load(open(G12, encoding="utf-8"))
r35 = json.load(open(R35, encoding="utf-8"))
chk("5 pointer diff", g13["gaps"] == g12["gaps"]
    and g13["appendix"][:-1] == g12["appendix"]
    and g13["provenance"] == g12["provenance"]
    and r36["features"] == r35["features"]
    and r36["failures"] == r35["failures"]
    and r36["provenance"]["anchor_files"] == r35["provenance"]["anchor_files"]
    and r36["census_3238"] == r35["census_3238"]
    and r36["evidence_level_taxonomy"] == r35["evidence_level_taxonomy"])

# 6 sealed predecessors byte-identical
chk("6 sealed untouched", s8(G12) == "aa8cff21" and s8(R35) == "e8335331"
    and s8(P3263_RES) == hashlib.sha256(
        open(P3263_RES, "rb").read()).hexdigest()[:8]
    and json.load(open(P3263_RES, encoding="utf-8"))["res_sha8"] == "f0f95c27"
    and json.load(open(P3264_RES, encoding="utf-8"))["res_sha8"] == "f223a9a0")

# 7 ledger #417 chained
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
e = ms[-1]
chk("7 ledger #417", len(ms) == 422 and e["phase"] == 3265
    and e["verdict"] == "appx7_landed" and e["res_sha8"] == RES_SHA
    and led["ledger_sha256_8"] == POST)
cp = json.loads(json.dumps(led))
cp["ledger_sha256_8"] = CHAIN_PREV
rec2 = hashlib.sha256(json.dumps(cp, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
chk("8 ledger chain recompute", rec2 == POST, rec2)

# 9 MEMO section
memo = open(MEMO, encoding="utf-8").read()
chk("9 MEMO section", "## Phase 3265 (G5-H28):" in memo
    and "814ce808" in memo and "78d37211" in memo)

# 10 dual wlogs
for i, p in enumerate([os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
                       os.path.join(ROOT, "research", "deepseek", "docs",
                                    "WORKLOG_gpt5_mirror.md")]):
    t = open(p, encoding="utf-8").read()
    chk("1%d wlog %s" % (0 + i, os.path.basename(os.path.dirname(p))),
        "3265 G5-H28" in t and POST in t)

# 11 MEMORY: 3265 block + incumbent version header updated
mem = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
           encoding="utf-8").read()
chk("11 MEMORY", "**3265" in mem and "v1.36 78d37211" in mem
    and "v1.13 814ce808" in mem and len(mem) <= 3000, len(mem))

print("VERIFY %d/%d" % (oks, oks + len(fails)))
if fails:
    print("FAILS:", fails)
    raise SystemExit(1)
print("ALL OK")
