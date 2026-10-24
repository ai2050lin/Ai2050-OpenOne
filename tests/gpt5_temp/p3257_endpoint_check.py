# -*- coding: utf-8 -*-
"""Phase 3257 endpoint check: registry v1.35 client visibility (G-line final audit arm a).

Live-verified against :5001:
  1. /atlas/registry           -> version 1.35, version chain contains 1.35,
                                  upgrade_log[-1] == closure seal annotation (3256)
  2. /atlas/registry?version=1.34 -> still resolvable (chain integrity, supersedes points to 3deabf25 file)
  3. /atlas/gaps               -> v1.12 (regression: appendix 6, APPX-6 anchor verbatim)
  4. three-account cross check  -> ledger #408 tail == registry seal reason == gap appendix count
READ-ONLY: no writes anywhere.
"""
import hashlib
import json
import sys
import urllib.request

BASE = "http://localhost:5001/api/ai-rnd"
PROBE = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with PROBE.open(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


OUT = []
ok = True


def check(name, cond, detail=""):
    global ok
    if not cond:
        ok = False
    OUT.append("%s %s %s" % ("PASS" if cond else "FAIL", name, detail))


# 1. registry v1.35 visible, chain complete
r = get("/atlas/registry")
vers = [v["version"] for v in r["versions"]]
check("1a version==1.35", r["version"] == "1.35", r["version"])
check("1b chain has 1.35..1.1", "1.35" in vers and "1.1" in vers,
      "n=%d head=%s" % (len(vers), vers[:3]))
v35 = [v for v in r["versions"] if v["version"] == "1.35"][0]
check("1c v1.35 file", v35["file"] == "atlas_registry_v1_35.json", v35["file"])
check("1d v1.35 upgrades=14", v35["n_upgrades"] == 14, str(v35["n_upgrades"]))
ul = r["upgrade_log"]
check("1e seal annotation tail", "closure seal" in ul[-1]["change"]
      and "3256" in ul[-1]["reason"], ul[-1]["change"][:60])

# 2. chain integrity: v1.34 still resolvable
r34 = get("/atlas/registry?version=1.34")
check("2a v1.34 resolvable", r34["version"] == "1.34", r34["version"])
check("2b v1.34 upgrades=13", len(r34["upgrade_log"]) == 13,
      str(len(r34["upgrade_log"])))

# 3. gaps v1.12 regression
g = get("/atlas/gaps")
check("3a gaps v1.12", g["version"] == "1.12", g["version"])
check("3b appendix 6", len(g["appendix"]) == 6, str(len(g["appendix"])))
appx6 = [a for a in g["appendix"] if a["id"] == "APPX-6"][0]
check("3c APPX-6 anchor", appx6.get("anchor") == "p3250"
      and appx6.get("anchor_sha8") == "9d0f1825",
      "%s/%s" % (appx6.get("anchor"), appx6.get("anchor_sha8")))

# 4. three-account cross check (ledger read from disk, read-only)
LED = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json"
led = json.load(open(LED, encoding="utf-8"))
e = led["measurements"][-1]
check("4a ledger tail #408", e["phase"] == 3256
      and led["ledger_sha256_8"] == "5601b3f5"
      and len(led["measurements"]) == 408, str(e["phase"]))
check("4b ledger verdict", e["verdict"].startswith("closure_sealed_registered"))
seal_change = ul[-1]["change"]
check("4c registry seal cites incumbent", "XC_p5_b8" in seal_change
      and "-0.449039" in seal_change and "0.7436" in seal_change)
check("4d counts consistent", len(led["measurements"]) == 408
      and len(ul) == 14 and len(g["appendix"]) == 6)

print("\n".join(OUT))
print("ENDPOINT CHECK OK" if ok else "ENDPOINT CHECK FAILED")
sys.exit(0 if ok else 1)
