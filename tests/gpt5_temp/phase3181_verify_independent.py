# -*- coding: utf-8 -*-
"""Phase 3181 独立复核（独立进程；verify FAIL 只修 verify 不动封存产物）"""
import json, hashlib, re, os

R = r"D:/AI2050/Ai2050-OpenOne"
D = os.path.join(R, r"tests\glm5\result\rdc_query_construction_20260913\phase3181\g5b6_ftr04_glm4_attribution")
V17 = os.path.join(R, r"tests\glm5\result\rdc_query_construction_20260913\phase3180\g5b5_ftr15_heldout\atlas_registry_v1_7.json")
V18 = os.path.join(D, "atlas_registry_v1_8.json")
HTML = os.path.join(D, "atlas_v1_9.html")
FIN = os.path.join(D, "final_result.json")
LED = os.path.join(R, r"research\gpt5\atlas\atlas_ledger.json")
BAK = os.path.join(R, r"tests\gpt5_temp\atlas_ledger_backup_pre_phase3181.json")

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

sha8 = lambda b: hashlib.sha256(b).hexdigest()[:8]

# 1 sha asserts
ck("sha reg18 d08e9064", sha8(open(V18, "rb").read()) == "d08e9064")
ck("sha html 79acbb6c", sha8(open(HTML, "rb").read()) == "79acbb6c")
ck("sha final seal 4ad066e0", json.loads(open(FIN, encoding="utf-8").read())["seal_sha8"] == "4ad066e0")

v17 = json.loads(open(V17, encoding="utf-8").read())
v18 = json.loads(open(V18, encoding="utf-8").read())
ck("v18 version 1.8", v18["version"] == "1.8")
ck("features count 23", len(v18["features"]) == 23)

# 2 legacy preservation: v17 features byte-equal in v18 except FTR-04 counter_evidence append-only
def norm(f): return json.dumps(f, ensure_ascii=False, sort_keys=True)
m17 = {f["id"]: f for f in v17["features"]}
m18 = {f["id"]: f for f in v18["features"]}
legacy_ok, ce_ok = True, True
for fid, f17 in m17.items():
    f18 = m18[fid]
    if fid == "FTR-04":
        if len(f18["counter_evidence"]) != len(f17["counter_evidence"]) + 1:
            ce_ok = False
        if f18["counter_evidence"][:len(f17["counter_evidence"])] != f17["counter_evidence"]:
            ce_ok = False
        rest18 = {k: v for k, v in f18.items() if k not in ("counter_evidence", "anchors")}
        rest17 = {k: v for k, v in f17.items() if k not in ("counter_evidence", "anchors")}
        if norm(rest18) != norm(rest17):
            ce_ok = False
    else:
        if norm(f18) != norm(f17):
            legacy_ok = False
ck("22 legacy features byte-preserved", legacy_ok)
ck("FTR-04 counter_evidence append-only", ce_ok)
for k in ("failures", "upgrade_log", "lineage"):
    ck("v17 %s untouched" % k, norm(v18.get(k)) == norm(v17.get(k)))

# 3 atlas data-k readback: every data-k present exactly once
html = open(HTML, encoding="utf-8").read()
ks = re.findall(r'data-k="([^"]+)"', html)
ck("atlas data-k count %d (expect 708)" % len(ks), len(ks) == 708)
ck("atlas data-k dups 0", len(ks) == len(set(ks)))
ck("atlas no missing placeholder", "data-k-missing" not in html)

# 4 ledger append-only
led = json.loads(open(LED, encoding="utf-8").read())
bak = json.loads(open(BAK, encoding="utf-8").read())
ck("ledger n=333", len(led["measurements"]) == 333)
ck("ledger #333 phase 3181", led["measurements"][-1]["phase"] == 3181)
pre = [json.dumps(m, sort_keys=True) for m in bak["measurements"]]
cur = [json.dumps(m, sort_keys=True) for m in led["measurements"][:332]]
ck("ledger prefix equivalent", pre == cur)
ck("ledger other keys untouched", {k: v for k, v in led.items() if k != "measurements"} ==
   {k: v for k, v in bak.items() if k != "measurements"})

print("RESULT: %s" % ("ALL PASS" if not fails else "FAIL: %s" % fails))
