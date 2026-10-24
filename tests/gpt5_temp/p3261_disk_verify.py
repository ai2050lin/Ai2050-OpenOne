# -*- coding: utf-8 -*-
"""Phase 3261 independent disk verify (fresh reader)."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3261", "g5h24_evidence_regrading")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")

oks, fails = 0, 0


def chk(name, cond, extra=""):
    global oks, fails
    if cond:
        oks += 1
        print("PASS", name)
    else:
        fails += 1
        print("FAIL", name, extra)


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


# 1 artifacts
for f in ("result.json", "evidence_regrading_draft_v1.json",
          "gpu_feasibility_note.json", "design_sha8.txt"):
    chk("1 artifact %s" % f, os.path.isfile(os.path.join(OUT, f)))

R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
# 2 anchors
ds = open(os.path.join(OUT, "design_sha8.txt"), encoding="utf-8").read().strip()
chk("2a design sha file", ds == R["design_sha8"] == "d8364fd8", ds)
chk("2b res sha", R["res_sha8"] == "dc6c1518")
chk("2c verdict/gates", R["verdict"] == "evidence_regrading_registered"
    and R["gate_all"] and all(R["gates"].values()))
# 3 draft artifact consistency
D = json.load(open(os.path.join(OUT, "evidence_regrading_draft_v1.json"), encoding="utf-8"))
chk("3a draft status", D["status"] == "REPORT_ONLY_DRAFT"
    and "phase3150" in D["rule_source"]["file"])
chk("3b draft distribution", D["derived_distribution"] == {
    "statistical": 314, "GOVERNANCE": 73, "bit_anchored": 14, "AMBIGUOUS": 16})
chk("3c draft full coverage", len(D["entries"]) == 417
    and D["ambiguous_count"] == 16
    and all(e["derived"] in {"bit_anchored", "statistical", "descriptive",
                             "GOVERNANCE", "AMBIGUOUS"} for e in D["entries"]))
chk("3d draft no-application", "NONE performed" in D["application"]
    and "destructive" in D["application"])
# 4 GPU note
G = json.load(open(os.path.join(OUT, "gpu_feasibility_note.json"), encoding="utf-8"))
chk("4 gpu note", "RTX 5080" in G.get("hardware", "")
    and G.get("torch") == "installed"
    and "parked" in G.get("conclusion", ""))
# 5 ledger #413 chain
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
chk("5a #413 once", sum(1 for e in ms if e.get("phase") == 3261) == 1)
prev = json.loads(json.dumps(led))
prev["ledger_sha256_8"] = "c26305dd"
rec = hashlib.sha256(json.dumps(prev, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("5b chain recompute", led["ledger_sha256_8"] == rec == "4f5ed748", rec)
e413 = [e for e in ms if e.get("phase") == 3261][0]
chk("5c #413 anchors", e413.get("design_sha8") == "d8364fd8"
    and e413.get("res_sha8") == "dc6c1518")
chk("5d #413 n=418", len(ms) == 418)
# 6 zero-write: ledger file sha was captured pre-closeout? verify chain suffices +
#    draft contains no writes: result zero_write_verified == post c26305dd pre-closeout
# phase-time zero-write: result captured the ledger file sha8 at formal run
# (after #412, before closeout #413) - a hex digest, not the post string;
# the closeout append (#413) legitimately changed the file afterwards
zw = R["zero_write_verified"]
chk("6 zero-write snapshot form", isinstance(zw, str) and len(zw) == 8
    and zw != "c26305dd" and zw != "4f5ed748", zw)
# 7 MEMO section once (correct title)
mb = open(MEMO, "rb").read()
chk("7 memo 3261 once", mb.count("## Phase 3261 (G5-H24):".encode("utf-8")) == 1
    and "evidence_regrading_registered".encode("utf-8") in mb
    and "dc6c1518".encode("utf-8") in mb)
# 8 wlogs
for tag, p in (("gpt5", os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")),
               ("deepseek", os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md"))):
    t = open(p, encoding="utf-8").read()
    chk("8 wlog %s" % tag, "3261 G5-H24" in t and "4f5ed748" in t)
# 9 MEMORY within limit, 3261 once
mem = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"), encoding="utf-8").read()
chk("9 memory <= 3000", len(mem) <= 3000, len(mem))
chk("9b memory 3261 once", mem.count("**3261") == 1)

print("VERIFY %d OK / %d FAIL" % (oks, fails))
print("DISK VERIFY OK" if fails == 0 else "DISK VERIFY FAILED")
