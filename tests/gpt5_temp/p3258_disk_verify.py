# -*- coding: utf-8 -*-
"""Phase 3258 independent disk verify (fresh read, no writer state)."""
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
rp = os.path.join(RBASE, "phase3258", "g5h21_nline_apply", "result.json")
R = json.load(open(rp, encoding="utf-8"))
raw = open(rp, "rb").read()
recomputed = hashlib.sha256(json.dumps(
    {k: v for k, v in json.loads(raw.decode("utf-8")).items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
chk("1a res_sha8 recompute", recomputed == R["res_sha8"], R["res_sha8"])
chk("1b verdict/gates", R["verdict"] == "nline_apply_registered"
    and R["gate_all"] is True and all(
        v for k, v in R["gates"].items() if v is not None))
chk("1c design sha matches file", R["design_sha8"] == open(
    os.path.join(RBASE, "phase3258", "g5h21_nline_apply", "design_sha8.txt"),
    encoding="utf-8").read().strip())

# 2 ledger final state
led_raw = open(os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json"), "rb").read()
led = json.loads(led_raw.decode("utf-8"))
chk("2a n=415", len(led["measurements"]) == 415, str(len(led["measurements"])))
chk("2b post declared", led["ledger_sha256_8"] == "8de77a63", led["ledger_sha256_8"])
chk_c = json.loads(led_raw.decode("utf-8"))
chk_c["ledger_sha256_8"] = "a59da52b"
rec = hashlib.sha256(json.dumps(chk_c, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
chk("2c chain recompute", rec == "8de77a63", rec)
chk("2d #410 present once", sum(1 for e in led["measurements"]
                                if e.get("phase") == 3258) == 1)
# append-only vs backup: revert = drop last entry + restore field
bak = json.load(open(os.path.join(ROOT, "research", "gpt5", "atlas",
                                  "atlas_ledger.json.bak_3258_closeout"), encoding="utf-8"))
rev = json.loads(led_raw.decode("utf-8"))
rev["measurements"] = rev["measurements"][:-1]
rev["ledger_sha256_8"] = "a59da52b"
chk("2e revert == pre-closeout backup", json.dumps(rev, ensure_ascii=False, indent=1)
    == json.dumps(bak, ensure_ascii=False, indent=1))
# 410 entry content
e410 = led["measurements"][-1]
chk("2f #410 name/verdict", e410["name"] == "g5h21_nline_apply"
    and e410["verdict"].startswith("nline_apply_registered")
    and e410["prereg_id"] == "G5-H21 (3257 tail backlog order)")

# 3 N-P entries content vs draft (verbatim fields)
draft = json.load(open(os.path.join(RBASE, "phase3208", "g5g7_nline_backfill_draft",
                                    "n_line_backfill_draft_v1.json"), encoding="utf-8"))
np_entries = [e for e in led["measurements"] if str(e.get("phase", "")).startswith("N-P")]
chk("3a five N-P entries", len(np_entries) == 5, str(len(np_entries)))
ok_verbatim = True
for de, le in zip(draft["entries"], np_entries):
    for k in ("name", "phase", "sources", "status", "title"):
        if de[k] != le.get(k):
            ok_verbatim = False
    if le.get("backfill", {}).get("applied_by_phase") != 3258:
        ok_verbatim = False
    if le.get("line") != "N":
        ok_verbatim = False
chk("3b draft fields verbatim + provenance", ok_verbatim)
chk("3c draft file untouched", s8(os.path.join(RBASE, "phase3208", "g5g7_nline_backfill_draft",
                                               "n_line_backfill_draft_v1.json")) == "f29da2de")

# 4 lcorr untouched + stale facts in result
lp = os.path.join(ROOT, "research", "deepseek", "atlas", "ledger_corrections_v1.json")
chk("4a lcorr untouched", s8(lp) == "beb775bd")
lc = json.load(open(lp, encoding="utf-8"))
chk("4b lcorr status unchanged", lc["status"] == "SEALED_BUT_NOT_APPLIED")
chk("4c disposition recorded", R["arm_b"]["disposition"].startswith("STALE_AS_SEALED")
    and R["arm_b"]["writes"] == 0)

# 5 ledger backups exist
for b in ("atlas_ledger.json.bak_3258", "atlas_ledger.json.bak_3258_closeout"):
    chk("5 backup " + b, os.path.isfile(os.path.join(ROOT, "research", "gpt5", "atlas", b)))

# 6 MEMO section once
mb = open(os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md"),
          "rb").read()
chk("6 memo 3258 once", mb.count("## Phase 3258:".encode("utf-8")) == 1
    and "nline_apply_registered".encode("utf-8") in mb
    and "a59da52b".encode("utf-8") in mb)

# 7 wlogs
for p, tag in ((os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"), "gpt5"),
               (os.path.join(ROOT, "research", "deepseek", "docs", "WORKLOG_gpt5_mirror.md"), "deepseek")):
    t = open(p, encoding="utf-8").read()
    chk("7 wlog " + tag, "3258 G5-H21" in t
        and "freeze db2255eb (SMOKE ae1e34cd)" in t)

print("\n".join(OUT))
if BAD:
    print("\n".join(BAD))
print("TOTAL %d OK / %d FAIL" % (len(OUT), len(BAD)))
