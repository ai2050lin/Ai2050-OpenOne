# -*- coding: utf-8 -*-
"""Independent read-only disk verification for phase 3266 closeout."""
import asyncio
import hashlib
import json
import os
import sys

ROOT = r"D:\AI2050\Ai2050-OpenOne"
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3266", "g5h29_closure_integration")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
MV1 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal",
                   "nline_handoff_manifest.json")
G12 = os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")
R35 = os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "atlas_registry_v1_35.json")

FREEZE = "6cc272b4"
RES_SHA = "d6b15de7"
POST = "9b6ac3d1"
CHAIN_PREV = "ec34b29b"
MV2_SHA = "57531ec3"

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
chk("1 result", R["verdict"] == "closure_integration_registered"
    and R["gate_all"] is True and R["design_sha8"] == FREEZE
    and R["res_sha8"] == RES_SHA
    and R["design_sha8"] == open(os.path.join(OUT, "design_sha8.txt"),
                                 encoding="utf-8").read()
    and len(R["gates"]) == 6 and all(R["gates"].values()))

# 2 res_sha8 recompute
r2 = {k: v for k, v in R.items() if k != "res_sha8"}
rec = hashlib.sha256(json.dumps(r2, sort_keys=True, ensure_ascii=False)
                     .encode("utf-8")).hexdigest()[:8]
chk("2 res_sha8 recompute", rec == RES_SHA, rec)

# 3 manifest v2 on disk: schema/snapshot fields
MV2 = os.path.join(OUT, "nline_handoff_manifest_v2.json")
m2 = json.load(open(MV2, encoding="utf-8"))
chk("3 manifest v2", s8(MV2) == MV2_SHA and m2["schema"] == "nline_handoff_v2"
    and m2["phase"] == 3266
    and m2["sealed_artifacts"]["registry_current"]["sha8_file"] == "78d37211"
    and m2["sealed_artifacts"]["gap_ledger_current"]["sha8_file"] == "814ce808"
    and m2["sealed_artifacts"]["gap_ledger_current"]["appendix_count"] == 7
    and m2["sealed_artifacts"]["ledger"]["entries"] == 422
    and m2["sealed_artifacts"]["ledger"]["post_sha8"] == "ec34b29b")

# 4 manifest v2 content: backlog + trilogy + v1 verbatim fields
m1 = json.load(open(MV1, encoding="utf-8"))
chk("4 manifest v2 content",
    any(x.startswith("RESOLVED") for x in m2["gpu_backlog"])
    and set(m2["u_star_trilogy"]) == {"3251_lexical", "3252_layer",
                                      "3253_subspace", "3263_trajectory",
                                      "3264_second_order", "3265_appx7"}
    and m2["u_star_trilogy"]["3263_trajectory"]["res_sha8"] == "f0f95c27"
    and m2["boundary_conditions"] == m1["boundary_conditions"]
    and m2["apply_protocol"] == m1["apply_protocol"]
    and m2["closure_facts"] == m1["closure_facts"])

# 5 sealed triple byte-identical
chk("5 sealed untouched", s8(MV1) == "bc212d52" and s8(G12) == "aa8cff21"
    and s8(R35) == "e8335331")

# 6 independent endpoint coroutine re-read (same in-process口径 as phase)
sys.path.insert(0, ROOT)
from server.ai_rnd_service import atlas_gaps, atlas_registry  # noqa: E402

doc_g = asyncio.run(atlas_gaps())
chk("6 gaps live v1.13", doc_g["version"] == "1.13"
    and len(doc_g["appendix"]) == 7
    and doc_g["appendix"][-1]["id"] == "APPX-7"
    and sorted(g["id"] for g in doc_g["gaps"])
    == ["GAP-1", "GAP-2", "GAP-3", "GAP-4", "GAP-5"])
doc_r = asyncio.run(atlas_registry())
chk("7 registry live v1.36", doc_r["version"] == "1.36"
    and len(doc_r["upgrade_log"]) == 15
    and doc_r["upgrade_log"][-1]["phase"] == 3265)
doc_r35 = asyncio.run(atlas_registry(version="1.35"))
chk("8 registry v1.35 regression", doc_r35["version"] == "1.35"
    and len(doc_r35["upgrade_log"]) == 14)

# 9 ledger #418 chained
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
e = ms[-1]
chk("9 ledger #418", len(ms) == 423 and e["phase"] == 3266
    and e["verdict"] == "closure_integration_registered"
    and e["res_sha8"] == RES_SHA and led["ledger_sha256_8"] == POST)
cp = json.loads(json.dumps(led))
cp["ledger_sha256_8"] = CHAIN_PREV
rec2 = hashlib.sha256(json.dumps(cp, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
chk("10 ledger chain recompute", rec2 == POST, rec2)

# 11 MEMO + wlogs + MEMORY
memo = open(MEMO, encoding="utf-8").read()
chk("11 MEMO section", "## Phase 3266 (G5-H29):" in memo
    and "57531ec3" in memo)
for i, p in enumerate([os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
                       os.path.join(ROOT, "research", "deepseek", "docs",
                                    "WORKLOG_gpt5_mirror.md")]):
    t = open(p, encoding="utf-8").read()
    chk("1%d wlog %s" % (2 + i, os.path.basename(os.path.dirname(p))),
        "3266 G5-H29" in t and POST in t)
mem = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
           encoding="utf-8").read()
chk("14 MEMORY", "**3266" in mem and len(mem) <= 3000, len(mem))

print("VERIFY %d/%d" % (oks, oks + len(fails)))
if fails:
    print("FAILS:", fails)
    raise SystemExit(1)
print("ALL OK")
