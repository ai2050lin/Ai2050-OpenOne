# -*- coding: utf-8 -*-
"""Phase 3262 independent disk verification (read-only)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3262", "g5h25_gpu_arm_readiness")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
R3261 = os.path.join(RBASE, "phase3261", "g5h24_evidence_regrading", "result.json")

fails = []


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def chk(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail)
    if not ok:
        fails.append(name)


# 1 result.json on disk matches sealed values
R = json.load(open(RES := os.path.join(OUT, "result.json"), encoding="utf-8"))
rec = hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
chk("1 result res_sha8 recompute", rec == R["res_sha8"] == "acf1b81c", rec)
chk("2 result verdict/gates", R["verdict"] == "gpu_arm_readiness_registered"
    and R["gate_all"] and all(R["gates"].values()))
chk("3 result freeze", R["design_sha8"] == "5c311013"
    and open(os.path.join(OUT, "design_sha8.txt"),
             encoding="utf-8").read().strip() == "5c311013")

# 2 artifacts sha8
chk("4 audit artifact sha8",
    s8(os.path.join(OUT, "checkpoint_presence_audit_v1.json")) == "9bbc3664")
chk("5 protocol artifact sha8",
    s8(os.path.join(OUT, "gpu_trajectory_protocol_draft_v1.json")) == "b569ce24")

# 3 ledger #414 chained
led = json.load(open(LED, encoding="utf-8"))
e414 = [e for e in led["measurements"] if e.get("phase") == 3262]
chk("6 ledger #414 present exactly once", len(e414) == 1)
chk("7 ledger n/post", len(led["measurements"]) == 419
    and led["ledger_sha256_8"] == "ec0dc42c", led["ledger_sha256_8"])
cp = dict(led)
cp["ledger_sha256_8"] = "4f5ed748"
rec2 = hashlib.sha256(json.dumps(cp, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
chk("8 ledger chain recompute", rec2 == "ec0dc42c", rec2)
ent = e414[0]
chk("9 #414 anchors", ent.get("design_sha8") == "5c311013"
    and ent.get("res_sha8") == "acf1b81c"
    and ent.get("verdict") == "gpu_arm_readiness_registered")

# 4 sealed 3261 untouched
R61 = json.load(open(R3261, encoding="utf-8"))
rec61 = hashlib.sha256(json.dumps(
    {k: v for k, v in R61.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
chk("10 sealed 3261 result untouched", rec61 == R61["res_sha8"] == "dc6c1518", rec61)

# 5 MEMO + wlogs + MEMORY
memo = open(MEMO, encoding="utf-8").read()
chk("11 MEMO 3262 section exactly once", memo.count("## Phase 3262 (G5-H25):") == 1)
for p in WLOGS:
    t = open(p, encoding="utf-8").read()
    chk("12 wlog " + os.path.basename(os.path.dirname(p)),
        t.count("3262 G5-H25") == 1 and "ec0dc42c" in t)
mem = open(MEMORY, encoding="utf-8").read()
chk("13 MEMORY 3262 block + <=3000", "**3262" in mem and len(mem) <= 3000,
    str(len(mem)))

# 6 presence spot-checks independent of phase script
chk("14 delta files spot", all(os.path.isfile(os.path.join(
    RBASE, "phase2755", "qwen25_3b", r, "delta_%03d.npz" % s))
    for r in ["true_token_2747", "surface_class_mass_2748"] for s in [1, 8, 32, 128]))
chk("15 archive absent", not os.path.exists("C:\\AI2050-RDC-Archive"))

print("VERIFY", "FAIL(" + str(len(fails)) + ")" if fails else "OK 15/15",
      fails if fails else "")
raise SystemExit(1 if fails else 0)
