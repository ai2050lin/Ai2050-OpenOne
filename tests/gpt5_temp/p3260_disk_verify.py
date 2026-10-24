# -*- coding: utf-8 -*-
"""Phase 3260 independent disk verify (fresh reader, no writer state)."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3260", "g5h23_lcorr_rederivation")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
LCORR = os.path.join(ROOT, "research", "deepseek", "atlas", "ledger_corrections_v1.json")
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
for f in ("result.json", "lcorr_rederivation_v2.json", "backlog_inventory_v3.json",
          "user_confirmation_proposal.md", "design_sha8.txt"):
    chk("1 artifact %s" % f, os.path.isfile(os.path.join(OUT, f)))

R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
# 2 design/res anchors
ds = open(os.path.join(OUT, "design_sha8.txt"), encoding="utf-8").read().strip()
chk("2a design sha file", ds == R["design_sha8"] == "f962faf0", ds)
chk("2b res sha", R["res_sha8"] == "209fe48d")
chk("2c verdict/gates", R["verdict"] == "lcorr_rederivation_registered" and R["gate_all"]
    and all(R["gates"].values()))
# 3 re-derivation dispositions vs v2 artifact
v2 = json.load(open(os.path.join(OUT, "lcorr_rederivation_v2.json"), encoding="utf-8"))
disp = {k: v["disposition"] for k, v in v2["operations"].items()}
chk("3a v2 dispositions", disp == {
    "op1_set_field": "OBSOLETE_SUPERSEDED",
    "op2_unify_schema": "REQUIRES_DESTRUCTIVE_MIGRATION",
    "op3_enumerate_field": "SUPERSEDED_BY_PRACTICE",
    "op4_append_agate": "READY_PENDING_CONFIRMATION"})
chk("3b v2 status", v2["status"] == "REPORT_ONLY_DRAFT"
    and v2["supersedes_draft"].startswith("ledger_corrections_v1.json"))
chk("3c v2 target state", v2["target_state_at_derivation"] == {"n": 416, "post": "7b6e6230"})
# 4 op evidence re-measured independently
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
chk("4a n=417 (post-#412)", len(ms) == 417, len(ms))
mid = sum(1 for e in ms if "meas_id" in e)
chk("4b meas_id 270", mid == 270)
agate = any(e.get("phase") == "A-gate" for e in ms)
chk("4c agate absent", agate is False and
    v2["operations"]["op4_append_agate"]["agate_in_ledger"] is False)
# 5 ledger #412 chain
chk("5a #412 once", sum(1 for e in ms if e.get("phase") == 3260) == 1)
e412 = [e for e in ms if e.get("phase") == 3260][0]
prev = json.loads(json.dumps(led))
prev["ledger_sha256_8"] = "7b6e6230"
rec = hashlib.sha256(json.dumps(prev, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("5b chain recompute", led["ledger_sha256_8"] == rec == "c26305dd", rec)
chk("5c #412 anchors", e412.get("design_sha8") == "f962faf0"
    and e412.get("res_sha8") == "209fe48d")
# 6 zero-write sealed files
chk("6a lcorr sealed", s8(LCORR) == "beb775bd")
smoke_files = sorted(os.listdir(os.path.join(OUT, "smoke"))) \
    if os.path.isdir(os.path.join(OUT, "smoke")) else []
chk("6b smoke dir conventional", all(
    f.endswith(".json") for f in smoke_files), str(smoke_files))
# 7 backlog v3
bl = json.load(open(os.path.join(OUT, "backlog_inventory_v3.json"), encoding="utf-8"))
chk("7a backlog closed x2", len(bl["closed_since_v2"]) == 2)
chk("7b backlog pending x2", len(bl["pending_user_confirmation"]) == 2)
# 8 proposal page readable
prop = open(os.path.join(OUT, "user_confirmation_proposal.md"), encoding="utf-8").read()
chk("8 proposal tokens", all(t in prop for t in
    ("水果类、K4", "lcorr v2 施加", "op4", "REPORT_ONLY", "3258")))
# 9 MEMO section once
mb = open(MEMO, "rb").read()
chk("9 memo 3260 once", mb.count("## Phase 3260 (G5-H23):".encode("utf-8")) == 1
    and "lcorr_rederivation_registered".encode("utf-8") in mb
    and "209fe48d".encode("utf-8") in mb)
# 10 wlogs
for tag, p in (("gpt5", os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")),
               ("deepseek", os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md"))):
    t = open(p, encoding="utf-8").read()
    chk("10 wlog %s" % tag, "3260 G5-H23" in t and "c26305dd" in t)
# 11 MEMORY within limit
mem = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"), encoding="utf-8").read()
chk("11 memory <= 3000", len(mem) <= 3000, len(mem))
chk("11b memory 3260 once", mem.count("**3260") == 1)

print("VERIFY %d OK / %d FAIL" % (oks, fails))
print("DISK VERIFY OK" if fails == 0 else "DISK VERIFY FAILED")
