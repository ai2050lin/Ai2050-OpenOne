# -*- coding: utf-8 -*-
"""Phase 3263 independent disk verification (read-only)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = os.path.join(RBASE, "phase3263", "g5h26_p_traj")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
WLOGS = [os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
         os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md")]
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
R3262 = os.path.join(RBASE, "phase3262", "g5h25_gpu_arm_readiness", "result.json")

fails = []


def destring_int_keys(o):
    """Round-trip fix: the sealed blob was computed in-memory where top1_curve
    etc. have INT keys (sorted numerically by json sort_keys); after load they
    are digit strings (sorted lexicographically). Restore int keys recursively
    so the blob is reproducible from the file alone."""
    if isinstance(o, dict):
        if o and all(isinstance(k, str) and k.lstrip("-").isdigit() for k in o):
            return {int(k): destring_int_keys(v) for k, v in o.items()}
        return {k: destring_int_keys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [destring_int_keys(v) for v in o]
    return o


def result_sha8(path):
    d = json.load(open(path, encoding="utf-8"))
    d.pop("res_sha8", None)
    blob = json.dumps(destring_int_keys(d), ensure_ascii=False,
                      sort_keys=True, indent=1).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def chk(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail)
    if not ok:
        fails.append(name)


R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
rec = hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
chk("1 result res_sha8 recompute (int-key normalized)",
    result_sha8(os.path.join(OUT, "result.json")) == R["res_sha8"] == "f0f95c27")
chk("2 result verdict + 7 gates", R["verdict"] == "p_traj_executed"
    and R["gate_all"] and all(R["gates"].values()) and len(R["gates"]) == 7)
chk("3 result freeze", R["design_sha8"] == "bdcd77ef"
    and open(os.path.join(OUT, "execution.json"), encoding="utf-8").read()
    .count("bdcd77ef") >= 1)
chk("4 25 states + smoke file separate",
    len(R["states"]) == 25
    and os.path.isfile(os.path.join(OUT, "result_smoke.json")))
chk("5 trajectory npz", os.path.isfile(os.path.join(OUT, "trajectory.npz")))
caches = [f for f in os.listdir(OUT) if f.startswith("collect_")]
chk("6 collect caches >= 26 (25 states + recheck)", len(caches) >= 26, str(len(caches)))
chk("7 g6 restore bitwise", R["gates"]["g6_restore_bitwise"] is True)

led = json.load(open(LED, encoding="utf-8"))
e415 = [e for e in led["measurements"] if e.get("phase") == 3263]
chk("8 ledger #415 exactly once", len(e415) == 1)
chk("9 ledger n/post", len(led["measurements"]) == 420
    and led["ledger_sha256_8"] == "b740bf76", led["ledger_sha256_8"])
cp = dict(led)
cp["ledger_sha256_8"] = "ec0dc42c"
rec2 = hashlib.sha256(json.dumps(cp, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
chk("10 ledger chain recompute", rec2 == "b740bf76", rec2)
ent = e415[0]
chk("11 #415 anchors", ent.get("design_sha8") == "bdcd77ef"
    and ent.get("res_sha8") == "f0f95c27"
    and ent.get("verdict") == "p_traj_executed")

R62 = json.load(open(R3262, encoding="utf-8"))
rec62 = hashlib.sha256(json.dumps(
    {k: v for k, v in R62.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
chk("12 sealed 3262 result untouched", rec62 == R62["res_sha8"] == "acf1b81c", rec62)

memo = open(MEMO, encoding="utf-8").read()
chk("13 MEMO 3262+3263 sections once each",
    memo.count("## Phase 3262 (G5-H25):") == 1
    and memo.count("## Phase 3263 (G5-H26):") == 1)
for p in WLOGS:
    t = open(p, encoding="utf-8").read()
    chk("14 wlog " + os.path.basename(os.path.dirname(p)),
        t.count("3263 G5-H26") == 1 and "b740bf76" in t)
mem = open(MEMORY, encoding="utf-8").read()
chk("15 MEMORY 3263 block + <=3000", "**3263" in mem and len(mem) <= 3000,
    str(len(mem)))

print("VERIFY", "FAIL(" + str(len(fails)) + ")" if fails else "OK 15/15",
      fails if fails else "")
raise SystemExit(1 if fails else 0)
