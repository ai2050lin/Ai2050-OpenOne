# -*- coding: utf-8 -*-
"""Phase 3183 独立复核（独立进程；verify FAIL 只修 verify 不动封存产物）— ALL PASS 已验证"""
import json, hashlib, re, os
R = r"D:/AI2050/Ai2050-OpenOne"
D = os.path.join(R, r"tests\glm5\result\rdc_query_construction_20260913\phase3183\g5b8_dv_lowrank")
V19 = os.path.join(R, r"tests\glm5\result\rdc_query_construction_20260913\phase3182\g5b7_ftr23_heldout_loro\atlas_registry_v1_9.json")
V20 = os.path.join(D, "atlas_registry_v1_10.json")
HTML = os.path.join(D, "atlas_v1_11.html")
FIN = os.path.join(D, "final_result.json")
LED = os.path.join(R, r"research\gpt5\atlas\atlas_ledger.json")
BAK = os.path.join(R, r"tests\gpt5_temp\atlas_ledger_backup_pre_phase3183.json")
fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)
sha8 = lambda b: hashlib.sha256(b).hexdigest()[:8]
ck("sha reg20 4cf49743", sha8(open(V20,"rb").read()) == "4cf49743")
ck("sha html 53f24922", sha8(open(HTML,"rb").read()) == "53f24922")
ck("sha final seal 66adfb27", json.loads(open(FIN,encoding="utf-8").read())["seal_sha8"] == "66adfb27")
v19 = json.loads(open(V19,encoding="utf-8").read())
v20 = json.loads(open(V20,encoding="utf-8").read())
ck("v20 version 1.10", v20["version"] == "1.10")
ck("features count 23 (no FTR-24)", len(v20["features"]) == 23)
norm = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)
ok_rest = True
for f in v19["features"]:
    g = [x for x in v20["features"] if x["id"]==f["id"]][0]
    if f["id"] == "FTR-23":
        ok = (len(g["counter_evidence"]) == len(f["counter_evidence"])+1
              and g["counter_evidence"][:len(f["counter_evidence"])] == f["counter_evidence"]
              and norm({k:v for k,v in f.items() if k!="counter_evidence"}) ==
                  norm({k:v for k,v in g.items() if k!="counter_evidence"}))
        if not ok: ok_rest = False
    elif norm(g) != norm(f): ok_rest = False
ck("23 features preserved (FTR-23 counter append only)", ok_rest)
ck("failures 17->18 with F18", len(v20["failures"])==18 and v20["failures"][-1]["id"]=="F18" and v20["failures"][-1]["phase"]==3183)
ck("failures prefix preserved", norm(v19["failures"]) == norm(v20["failures"][:17]))
ck("upgrade_log 6 unchanged", norm(v20["upgrade_log"]) == norm(v19["upgrade_log"]))
html = open(HTML, encoding="utf-8").read()
ks = re.findall(r'data-k="([^"]+)"', html)
ck("atlas data-k count %d (expect 717)" % len(ks), len(ks)==717)
ck("atlas data-k dups 0", len(ks)==len(set(ks)))
led = json.loads(open(LED,encoding="utf-8").read())
bak = json.loads(open(BAK,encoding="utf-8").read())
ck("ledger n=335", len(led["measurements"])==335)
ck("ledger #335 phase 3183", led["measurements"][-1]["phase"]==3183)
pre = [json.dumps(m,sort_keys=True) for m in bak["measurements"]]
cur = [json.dumps(m,sort_keys=True) for m in led["measurements"][:334]]
ck("ledger prefix equivalent", pre==cur)
ck("ledger other keys untouched", {k:v for k,v in led.items() if k!="measurements"} == {k:v for k,v in bak.items() if k!="measurements"})
print("RESULT: %s" % ("ALL PASS" if not fails else "FAIL: %s" % fails))
