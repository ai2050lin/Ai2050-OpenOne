# -*- coding: utf-8 -*-
"""Phase 3182 独立复核（独立进程；verify FAIL 只修 verify 不动封存产物）"""
import json, hashlib, re, os

R = r"D:/AI2050/Ai2050-OpenOne"
D = os.path.join(R, r"tests\glm5\result\rdc_query_construction_20260913\phase3182\g5b7_ftr23_heldout_loro")
V18 = os.path.join(R, r"tests\glm5\result\rdc_query_construction_20260913\phase3181\g5b6_ftr04_glm4_attribution\atlas_registry_v1_8.json")
V19 = os.path.join(D, "atlas_registry_v1_9.json")
HTML = os.path.join(D, "atlas_v1_10.html")
FIN = os.path.join(D, "final_result.json")
LED = os.path.join(R, r"research\gpt5\atlas\atlas_ledger.json")
BAK = os.path.join(R, r"tests\gpt5_temp\atlas_ledger_backup_pre_phase3182.json")

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

sha8 = lambda b: hashlib.sha256(b).hexdigest()[:8]

ck("sha reg19 59a91512", sha8(open(V19, "rb").read()) == "59a91512")
ck("sha html 0f24b739", sha8(open(HTML, "rb").read()) == "0f24b739")
ck("sha final seal b6a57f07", json.loads(open(FIN, encoding="utf-8").read())["seal_sha8"] == "b6a57f07")

v18 = json.loads(open(V18, encoding="utf-8").read())
v19 = json.loads(open(V19, encoding="utf-8").read())
ck("v19 version 1.9", v19["version"] == "1.9")
ck("features count 23", len(v19["features"]) == 23)

def norm(f): return json.dumps(f, ensure_ascii=False, sort_keys=True)
m18 = {f["id"]: f for f in v18["features"]}
m19 = {f["id"]: f for f in v19["features"]}
ok23, ok_rest = True, True
for fid, f18 in m18.items():
    f19 = m19[fid]
    if fid == "FTR-23":
        if f19["evidence_level"] != "E1_repeatable":
            ok23 = False
        if len(f19["counter_evidence"]) != len(f18["counter_evidence"]) + 1:
            ok23 = False
        if f19["counter_evidence"][:len(f18["counter_evidence"])] != f18["counter_evidence"]:
            ok23 = False
        if len(f19["anchors"]) != len(f18["anchors"]):
            ok23 = False
        rest19 = {k: v for k, v in f19.items() if k not in ("counter_evidence",)}
        rest18 = {k: v for k, v in f18.items() if k not in ("counter_evidence",)}
        if norm(rest19) != norm(rest18):
            ok23 = False
    else:
        if norm(f19) != norm(f18):
            ok_rest = False
ck("FTR-23 stays E1 + counter_evidence append-only", ok23)
ck("22 legacy features byte-preserved", ok_rest)
ck("failures 16->17 with F17", len(v19["failures"]) == 17 and v19["failures"][-1]["id"] == "F17"
   and v19["failures"][-1]["phase"] == 3182)
ck("failures prefix preserved", norm(v19["failures"][:16]) == norm(v18["failures"]))
ck("upgrade_log 6 unchanged", norm(v19["upgrade_log"]) == norm(v18["upgrade_log"]))

html = open(HTML, encoding="utf-8").read()
ks = re.findall(r'data-k="([^"]+)"', html)
ck("atlas data-k count %d (expect 713)" % len(ks), len(ks) == 713)
ck("atlas data-k dups 0", len(ks) == len(set(ks)))
ck("atlas no missing placeholder", "data-k-missing" not in html)

led = json.loads(open(LED, encoding="utf-8").read())
bak = json.loads(open(BAK, encoding="utf-8").read())
ck("ledger n=334", len(led["measurements"]) == 334)
ck("ledger #334 phase 3182", led["measurements"][-1]["phase"] == 3182)
pre = [json.dumps(m, sort_keys=True) for m in bak["measurements"]]
cur = [json.dumps(m, sort_keys=True) for m in led["measurements"][:333]]
ck("ledger prefix equivalent", pre == cur)
ck("ledger other keys untouched", {k: v for k, v in led.items() if k != "measurements"} ==
   {k: v for k, v in bak.items() if k != "measurements"})

print("RESULT: %s" % ("ALL PASS" if not fails else "FAIL: %s" % fails))
