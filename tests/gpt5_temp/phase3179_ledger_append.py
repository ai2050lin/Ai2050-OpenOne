# -*- coding: utf-8 -*-
"""Phase 3179 closeout: atlas_ledger.json append-only（G 线 measurement #331）"""
import json, io, hashlib, shutil, sys

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json"
BAK = r"D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/atlas_ledger_backup_pre_phase3179.json"

def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

raw = open(P, "rb").read()
pre8 = sha8(raw)
assert pre8 == "8b6cfe43", "pre-append file sha changed: %s" % pre8
shutil.copyfile(P, BAK)

d = json.loads(raw.decode("utf-8"))
assert len(d["measurements"]) == 330, len(d["measurements"])
assert d["measurements"][-1]["phase"] == 3178

entry = {
 "phase": 3179,
 "name": "g5b4_ftr23_registration",
 "line": "G",
 "date": "2026-10-10",
 "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
 "verdict": "g5b4_ftr23_registration|ftr23_registered_E1_repeatable|features_23|failures_16|upgrades_5|registry_v16|html_fields_698_ok",
 "evidence_level": "E1_repeatable",
 "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
 "prereg_id": "G5-B4",
 "superseded_by": None,
 "detail": ("G5-B4 FTR-23 registration (zero GPU; 'failure into atlas' positivized): T_C relation-specific "
            "residual promoted from FTR-14 counter_evidence string to standalone feature FTR-23 "
            "(family=context, E1_repeatable, cross_model immediate; held_out upgrade deferred -- for a "
            "relation-specific feature 'held-out relation' is itself the measurement construction). "
            "Registration bar met: two-phase independent measurement (3177 held-out arm + 3178 sampling-null "
            "attribution), 3-model same direction r(m)=0.0918/0.0951/0.1289 >= 0.05 bar x3, complete prereg "
            "gate history (F16 gate fail -> 3178 quantitative attribution). Device: 12 source files sha-asserted; "
            "sealed content double-locked p3178 4f521d9b/1f166645 + p3177 d451030f/7b746eb2 + p3157 collect "
            "95a25965/9552086d/23dd74eb; recompute vs sealed 3178 drift 0.00e+00 (tc_full/exch/p_mean x3 models); "
            "range asserts vs MEMO draft all green; registry v1.5 -> v1.6 features-append-only (22 legacy features "
            "byte-for-byte, failures/upgrade_log/lineage untouched, FTR-23 appended; bebdf38d, SMOKE/FULL "
            "byte-identical); atlas_v1_7.html 698 fields 0/0/0/0 (b2bcec66). Independent verify (separate process, "
            "this closeout): sha re-assert pass, 22-feature preservation pass, field-level diff vs sealed v1.6 html "
            "dropped=0 / changed+new only FTR-23+meta+node views. res 18d6697f / seal e1b5e3a2. "
            "ledger_sha256_8 NOT recomputed (recipe unknown, stale since Phase 8 per migration_history); "
            "pre-append file sha8 8b6cfe43."),
 "created": "2026-10-10T04:20:00"
}
d["measurements"].append(entry)
out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
open(P, "wb").write(out.encode("utf-8"))
print("appended: measurements 330 ->", len(d["measurements"]))
print("pre sha8", pre8, "| post sha8", sha8(open(P, "rb").read()))
# 回读断言
d2 = json.loads(open(P, "rb").read().decode("utf-8"))
assert len(d2["measurements"]) == 331 and d2["measurements"][-1]["phase"] == 3179
assert all(m["phase"] <= 3178 for m in d2["measurements"][:330])
print("read-back OK")
