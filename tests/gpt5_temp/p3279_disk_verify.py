# -*- coding: utf-8 -*-
"""Phase 3279 independent disk verify (separate pass, fresh read)."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
OUT = os.path.join(ROOT, "tests", "glm5", "result",
                   "rdc_query_construction_20260913",
                   "phase3279", "g5h42_appx1_landing")
G15 = os.path.join(ROOT, "tests", "glm5", "result",
                   "rdc_query_construction_20260913",
                   "phase3277", "g5h40_appx23_landing", "gap_ledger_v1_15.json")
R38 = os.path.join(ROOT, "tests", "glm5", "result",
                   "rdc_query_construction_20260913",
                   "phase3277", "g5h40_appx23_landing", "atlas_registry_v1_38.json")
P3278 = os.path.join(ROOT, "tests", "glm5", "result",
                     "rdc_query_construction_20260913",
                     "phase3278", "g5h41_appx1_exec", "result.json")


def s8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS" if cond else "FAIL"), name)


# 1 landed files on disk
g16p = os.path.join(OUT, "gap_ledger_v1_16.json")
r39p = os.path.join(OUT, "atlas_registry_v1_39.json")
g16 = json.load(open(g16p, encoding="utf-8"))
r39 = json.load(open(r39p, encoding="utf-8"))
check("v1 gap v1.16 structure (appendix 10, last APPX-9)",
      g16["version"] == "1.16" and len(g16["appendix"]) == 10
      and g16["appendix"][-1]["id"] == "APPX-9"
      and "blocked" in g16["appendix"][-1]["status"])
check("v2 registry v1.39 structure (upgrade_log 18)",
      r39["version"] == "1.39" and len(r39["upgrade_log"]) == 18
      and "updated" not in r39)

# 2 result.json anchors
res = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
res_re = hashlib.sha256(json.dumps(
    {k: v for k, v in res.items() if k != "res_sha8"},
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
check("v3 res_sha8 content-recompute == 0d702e47", res_re == "0d702e47")
check("v4 verdict + gates 6/6",
      res["verdict"] == "appx1_landed_blocked_verbatim"
      and res["gate_all"] is True
      and all(res["gates"].values()))
check("v5 appx1 standing item NOT closed",
      "NOT closed" in res["appx1_standing_item"])

# 3 sealed predecessors byte-identical
check("v6 gap v1.15 sealed 58094dab", s8(G15) == "58094dab")
check("v7 registry v1.38 sealed f8772d65", s8(R38) == "f8772d65")

# 4 revert bitwise from v1.16/v1.39
g15d = json.load(open(G15, encoding="utf-8"))
r38d = json.load(open(R38, encoding="utf-8"))
rg = json.loads(json.dumps(g16))
rg["version"] = "1.15"
rg["appendix"] = g15d["appendix"]
rg["supersedes"] = g15d["supersedes"]
rg["updated"] = g15d["updated"]
rr = json.loads(json.dumps(r39))
rr["version"] = "1.38"
rr["upgrade_log"] = r38d["upgrade_log"]
rr["supersedes"] = r38d["supersedes"]
hg = hashlib.sha256(json.dumps(rg, ensure_ascii=False, indent=1,
                               sort_keys=False).encode("utf-8")).hexdigest()[:8]
hr = hashlib.sha256(json.dumps(rr, ensure_ascii=False, indent=1,
                               sort_keys=False).encode("utf-8")).hexdigest()[:8]
check("v8 revert gap -> 58094dab", hg == "58094dab")
check("v9 revert registry -> f8772d65", hr == "f8772d65")

# 5 pointer-diff audit
check("v10 gaps/features/failures untouched",
      g16["gaps"] == g15d["gaps"]
      and r39["features"] == r38d["features"]
      and r39["failures"] == r38d["failures"])

# 6 ledger chain head recompute (full state, field=prev_post)
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
probe = json.loads(json.dumps(led))
probe["ledger_sha256_8"] = "c4d25687"
rec = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                     .encode("utf-8")).hexdigest()[:8]
check("v11 chain head n=437 post=0ee33c4c recompute",
      len(ms) == 437 and led["ledger_sha256_8"] == "0ee33c4c" and rec == "0ee33c4c")
check("v12 last rec anchors", str(ms[-1]["phase"]) == "3279"
      and ms[-1]["res_sha8"] == "0d702e47"
      and ms[-1]["verdict"] == "appx1_landed_blocked_verbatim")

# 7 p3278 sealed untouched
r278 = json.load(open(P3278, encoding="utf-8"))
check("v13 p3278 res sealed 588d36c9", r278["res_sha8"] == "588d36c9")

# 8 five-write presence
wlog = open(os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md"),
            encoding="utf-8").read()
memo = open(os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md"),
            encoding="utf-8").read()
daily = open(os.path.join(ROOT, ".workbuddy", "memory", "2026-10-21.md"),
             encoding="utf-8").read()
memory = open(os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md"),
              encoding="utf-8").read()
check("v14 five-write presence", "g5h42_appx1_landing" in wlog
      and "Phase 3279 (G5-H42) APPX-1 落册相" in memo
      and "3279 G5-H42" in daily
      and "APPX-1 落册" in memory)
check("v15 MEMORY <=3000 chars", len(memory) <= 3000)

print("VERIFY", "PASS" if all(ok) else "FAIL", "%d/%d" % (sum(ok), len(ok)))
