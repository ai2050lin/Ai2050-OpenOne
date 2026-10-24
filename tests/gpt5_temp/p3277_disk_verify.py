# -*- coding: utf-8 -*-
"""Phase 3277 independent disk verify (separate from closeout). 11 checks."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")
P3277 = os.path.join(RBASE, "phase3277", "g5h40_appx23_landing")
P3274 = os.path.join(RBASE, "phase3274", "g5h37_lcorr_op4_apply")
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), name)


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


# 1 chain head recompute: full state with field=prev_post(76783ddd-level)
raw = open(LED, "rb").read()
led = json.loads(raw.decode("utf-8"))
ms = led["measurements"]
probe = json.loads(raw.decode("utf-8"))
probe["ledger_sha256_8"] = "ee5b19f3"
rec = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
check("v1 chain head n=435 field=ee5b19f3 -> 76783ddd",
      len(ms) == 435 and led["ledger_sha256_8"] == "76783ddd" and rec == "76783ddd")

# 2 prev-level: drop last entry, field=923806e0 -> ee5b19f3
prev = json.loads(raw.decode("utf-8"))
prev["measurements"] = ms[:-1]
prev["ledger_sha256_8"] = "923806e0"
rec2 = hashlib.sha256(json.dumps(prev, ensure_ascii=False, indent=1)
                      .encode("utf-8")).hexdigest()[:8]
check("v2 prev-level n=434 field=923806e0 -> ee5b19f3", rec2 == "ee5b19f3")

# 3 #429 record content
r429 = ms[-1]
check("v3 #429 rec phase/verdict/res/prev",
      str(r429["phase"]) == "3277" and r429["verdict"] == "appx23_landed"
      and r429["res_sha8"] == "3ddd53b8" and r429["prev_post"] == "ee5b19f3"
      and "58094dab" in r429["note"] and "f8772d65" in r429["note"])

# 4 result.json res_sha8 content recompute
rj = json.load(open(os.path.join(P3277, "result.json"), encoding="utf-8"))
rc = json.loads(json.dumps(rj, ensure_ascii=False))
declared = rc.pop("res_sha8")
rr = hashlib.sha256(json.dumps(rc, sort_keys=True, ensure_ascii=False)
                    .encode("utf-8")).hexdigest()[:8]
check("v4 result res_sha8 content recompute 3ddd53b8",
      declared == "3ddd53b8" and rr == "3ddd53b8")

# 5 landed files byte sha
check("v5 gap v1.15 file sha 58094dab",
      fsha(os.path.join(P3277, "gap_ledger_v1_15.json")) == "58094dab")
check("v6 registry v1.38 file sha f8772d65",
      fsha(os.path.join(P3277, "atlas_registry_v1_38.json")) == "f8772d65")

# 7 structure
g15 = json.load(open(os.path.join(P3277, "gap_ledger_v1_15.json"), encoding="utf-8"))
r38 = json.load(open(os.path.join(P3277, "atlas_registry_v1_38.json"), encoding="utf-8"))
check("v7 structure v1.15/v1.38",
      g15["version"] == "1.15" and len(g15["appendix"]) == 9
      and g15["appendix"][-1]["id"] == "APPX-8"
      and g15["appendix"][-1]["anchor_sha8"] == "8d5bba36"
      and r38["version"] == "1.38" and len(r38["upgrade_log"]) == 17
      and r38["upgrade_log"][-1]["phase"] == 3277
      and "updated" not in r38)

# 8 REVERT bitwise (content sha) to sealed predecessors
g14d = json.load(open(os.path.join(P3274, "gap_ledger_v1_14.json"), encoding="utf-8"))
r37d = json.load(open(os.path.join(P3274, "atlas_registry_v1_37.json"), encoding="utf-8"))
rev_g = json.loads(json.dumps(g15))
rev_g["version"] = "1.14"
rev_g["appendix"] = g14d["appendix"]
rev_g["supersedes"] = g14d["supersedes"]
rev_g["updated"] = g14d["updated"]
rev_r = json.loads(json.dumps(r38))
rev_r["version"] = "1.37"
rev_r["upgrade_log"] = r37d["upgrade_log"]
rev_r["supersedes"] = r37d["supersedes"]
check("v8 revert gap -> c6d97133",
      hashlib.sha256(json.dumps(rev_g, ensure_ascii=False, indent=1,
                                sort_keys=False).encode("utf-8")).hexdigest()[:8]
      == "c6d97133")
check("v9 revert registry -> bfb7d7e5",
      hashlib.sha256(json.dumps(rev_r, ensure_ascii=False, indent=1,
                                sort_keys=False).encode("utf-8")).hexdigest()[:8]
      == "bfb7d7e5")

# 10 sealed predecessors byte-untouched (file sha, 3274 landing convention)
check("v10 sealed v1.14/v1.37 bytes intact",
      fsha(os.path.join(P3274, "gap_ledger_v1_14.json")) == "c6d97133"
      and fsha(os.path.join(P3274, "atlas_registry_v1_37.json")) == "bfb7d7e5")

# 11 five-write artifacts present
wl = open(os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
          encoding="utf-8").read()
gm = open(os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md"),
          encoding="utf-8").read()
dl = open(os.path.join(ROOT, ".workbuddy", "memory", "2026-10-21.md"),
          encoding="utf-8").read()
pv = open(os.path.join(ROOT, "research", "gpt5", "docs",
                       "PENDING_ITEMS_IMPACT_PREVIEW.md"), encoding="utf-8").read()
mem = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
           encoding="utf-8").read()
check("v11 five-write present + MEMORY<=3000",
      "G5-H40" in wl and "G5-H40" in gm and "3277 G5-H40" in dl
      and "addendum-3277" in pv
      and "58094dab" in mem and len(mem) <= 3000)

n_pass = sum(1 for _, v in ok if v)
print("VERIFY %d/%d" % (n_pass, len(ok)))
print("VERIFY PASS" if n_pass == len(ok) else "VERIFY FAIL")
