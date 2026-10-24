# -*- coding: utf-8 -*-
"""Phase 3259 endpoint check: post-apply client visibility of N-P entries (live)."""
import json
import urllib.request

BASE = "http://localhost:5001/api/ai-rnd"
OUT = []


def chk(name, cond, extra=""):
    OUT.append("%s %s %s" % ("PASS" if cond else "FAIL", name, extra))
    return cond


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


# 1 ledger head: total / post
d = get("/atlas/ledger?limit=1")
chk("1a ledger total 415", d["total"] == 415, str(d["total"]))
chk("1b ledger post 8de77a63", d["ledger_sha256_8"] == "8de77a63", d["ledger_sha256_8"])

# 2 line=N filter: exactly the 5 backfill entries
dn = get("/atlas/ledger?line=N&limit=50")
chk("2a line=N count 5", dn["count"] == 5, str(dn["count"]))
phases = [m["phase"] for m in dn["measurements"]]
chk("2b phases N-P3..N-P7", phases == ["N-P7", "N-P6", "N-P5", "N-P4", "N-P3"], str(phases))
names = [m["name"] for m in dn["measurements"]]
chk("2c names verbatim", set(names) == {
    "e2_conditioned_compare_layer_scan", "n1_main_axis_layer_scan",
    "n2_reconstruction_source", "n2h1_permutation_ablation",
    "n3_device_transfer"}, str(names))
chk("2d prereg_id draft anchor", all("f29da2de" in (m.get("prereg_id") or "")
                                     for m in dn["measurements"]))
chk("2e evidence descriptive backfill", all(
    str(m.get("evidence_level", "")).startswith("descriptive")
    for m in dn["measurements"]))
# newest-first page: top entry should be #410 (phase 3258) on unfiltered head
chk("2f head top is 3258", d["measurements"][0]["phase"] == 3258
    and d["measurements"][0]["name"] == "g5h21_nline_apply",
    str(d["measurements"][0]["phase"]))

# 3 sealed accounts still visible unchanged
dr = get("/atlas/registry")
chk("3a registry default v1.35", str(dr.get("version")) == "1.35", str(dr.get("version")))
dg = get("/atlas/gaps")
chk("3b gaps v1.12 + appendix 6", str(dg.get("version")) == "1.12"
    and len(dg.get("appendix", [])) == 6,
    "%s/%d" % (dg.get("version"), len(dg.get("appendix", []))))

# 4 G-line filter unaffected
dgl = get("/atlas/ledger?line=G&limit=5")
chk("4a line=G top 3258", dgl["measurements"][0]["phase"] == 3258
    and all(m.get("line") == "G" for m in dgl["measurements"]))

print("\n".join(OUT))
bad = [l for l in OUT if l.startswith("FAIL")]
print("ENDPOINT CHECK %s: %d PASS / %d FAIL"
      % ("OK" if not bad else "DIRTY", len(OUT) - len(bad), len(bad)))
