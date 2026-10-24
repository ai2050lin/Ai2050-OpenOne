# -*- coding: utf-8 -*-
"""Phase 3259 independent disk verify (fresh read, no writer state)."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
OUT = []
BAD = []


def chk(name, cond, extra=""):
    (OUT if cond else BAD).append("%s %s %s" % ("PASS" if cond else "FAIL", name, extra))


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


# 1 result.json self-consistent
rp = os.path.join(RBASE, "phase3259", "g5h22_postapply_consistency", "result.json")
R = json.load(open(rp, encoding="utf-8"))
raw = open(rp, "rb").read()
recomputed = hashlib.sha256(json.dumps(
    {k: v for k, v in json.loads(raw.decode("utf-8")).items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
chk("1a res_sha8 recompute", recomputed == R["res_sha8"], R["res_sha8"])
chk("1b verdict/gates", R["verdict"] == "postapply_consistency_confirmed"
    and R["gate_all"] is True and all(
        v for k, v in R["gates"].items() if v is not None))
chk("1c design sha matches file", R["design_sha8"] == open(
    os.path.join(RBASE, "phase3259", "g5h22_postapply_consistency", "design_sha8.txt"),
    encoding="utf-8").read().strip())

# 2 ledger final state
led_raw = open(os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json"), "rb").read()
led = json.loads(led_raw.decode("utf-8"))
chk("2a n=416", len(led["measurements"]) == 416, str(len(led["measurements"])))
chk("2b post declared 7b6e6230", led["ledger_sha256_8"] == "7b6e6230", led["ledger_sha256_8"])
chk_c = json.loads(led_raw.decode("utf-8"))
chk_c["ledger_sha256_8"] = "8de77a63"
rec = hashlib.sha256(json.dumps(chk_c, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("2c chain recompute", rec == "7b6e6230", rec)
chk("2d #411 present once", sum(1 for e in led["measurements"]
                                if e.get("phase") == 3259) == 1)
bak = json.load(open(os.path.join(ROOT, "research", "gpt5", "atlas",
                                  "atlas_ledger.json.bak_3259_closeout"), encoding="utf-8"))
rev = json.loads(led_raw.decode("utf-8"))
rev["measurements"] = rev["measurements"][:-1]
rev["ledger_sha256_8"] = "8de77a63"
chk("2e revert == pre-closeout backup", json.dumps(rev, ensure_ascii=False, indent=1)
    == json.dumps(bak, ensure_ascii=False, indent=1))
e411 = led["measurements"][-1]
chk("2f #411 name/verdict", e411["name"] == "g5h22_postapply_consistency"
    and e411["verdict"].startswith("postapply_consistency_confirmed"))

# 3 sealed accounts untouched
SEAL = {"registry_v1_35": ("e8335331", os.path.join(RBASE, "phase3256", "g5h19_closure_seal", "atlas_registry_v1_35.json")),
        "registry_v1_34": ("3deabf25", os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "atlas_registry_v1_34.json")),
        "gap_v1_12": ("aa8cff21", os.path.join(RBASE, "phase3254", "g5h17_registry_landing", "gap_ledger_v1_12.json")),
        "draft": ("f29da2de", os.path.join(RBASE, "phase3208", "g5g7_nline_backfill_draft", "n_line_backfill_draft_v1.json")),
        "lcorr": ("beb775bd", os.path.join(ROOT, "research", "deepseek", "atlas", "ledger_corrections_v1.json"))}
ok3 = all(s8(p) == want for want, p in SEAL.values())
chk("3 sealed set untouched (5 files)", ok3,
    str({k: s8(p) for k, (want, p) in SEAL.items()}))

# 4 deepseek MEMO receipt
ds = open(os.path.join(ROOT, "research", "deepseek", "docs", "AGI_DEEPSEEK_MEMO.md"),
          encoding="utf-8").read()
chk("4a receipt once", ds.count("## 跨线施加回执（G 线 Phase 3258）") == 1)
chk("4b receipt facts", "f29da2de" in ds[ds.find("跨线施加回执"):]
    and "8de77a63" in ds[ds.find("跨线施加回执"):]
    and "STALE_AS_SEALED" in ds[ds.find("跨线施加回执"):])

# 5 endpoint output file
ep = open(os.path.join(ROOT, "tests", "gpt5_temp", "p3259_endpoint_out.txt"),
          encoding="utf-8").read()
chk("5 endpoint 11 PASS / 0 FAIL",
    sum(1 for l in ep.splitlines() if l.startswith("PASS")) == 11
    and "ENDPOINT CHECK OK" in ep and not any(l.startswith("FAIL") for l in ep.splitlines()))

# 6 MEMO section once
mb = open(os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md"),
          "rb").read()
chk("6 memo 3259 once", mb.count("## Phase 3259:".encode("utf-8")) == 1
    and "postapply_consistency_confirmed".encode("utf-8") in mb)

# 7 wlogs
for p, tag in ((os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"), "gpt5"),
               (os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md"), "deepseek")):
    t = open(p, encoding="utf-8").read()
    chk("7 wlog " + tag, "3259 G5-H22" in t
        and "freeze 3de50b8d (SMOKE 49925445)" in t)

print("\n".join(OUT))
if BAD:
    print("\n".join(BAD))
print("TOTAL %d OK / %d FAIL" % (len(OUT), len(BAD)))
