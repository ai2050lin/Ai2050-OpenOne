# -*- coding: utf-8 -*-
"""Phase 3181 closeout: atlas_ledger.json append-only（G 线 measurement #333）"""
import json, hashlib, shutil

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json"
BAK = r"D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/atlas_ledger_backup_pre_phase3181.json"

def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

raw = open(P, "rb").read()
pre8 = sha8(raw)
assert pre8 == "8d03febc", "pre-append file sha changed: %s" % pre8
shutil.copyfile(P, BAK)

d = json.loads(raw.decode("utf-8"))
assert len(d["measurements"]) == 332, len(d["measurements"])
assert d["measurements"][-1]["phase"] == 3180

entry = {
 "phase": 3181,
 "name": "g5b6_ftr04_glm4_attribution",
 "line": "G",
 "date": "2026-10-10",
 "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
 "verdict": "g5b6_ftr04_glm4_attribution|noise_level_not_attributable|gates_Ga_F_Gb_F|registry_v18|html_fields_708_ok",
 "evidence_level": "E1_repeatable_unchanged",
 "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
 "prereg_id": "G5-B6",
 "superseded_by": None,
 "detail": ("G5-B6 FTR-04 glm4 weak-specificity attribution (zero GPU; two-arm decide-one). "
            "Geo arm on S_class (3166 recipe verbatim, per-model W_U): kappa 4.81e7/4.60e7/4.50e7 "
            "(glm4 NOT >= 1.25x max qwen), e90 8/8/8 (no drop), PR 8.78/8.90/8.92 (glm4 NOT <= 0.8x min) "
            "-> G-a FALSE. Readout arm on sealed 3169 seen rows: proj_ratio 0.0763/0.0328/0.0294 "
            "(glm4 0.0294 NOT <= 0.75x min qwen 0.0246), LOEO mean_margin 2.6614/1.4164/5.2326 "
            "(glm4 largest, NOT <= 0.75x min) -> G-b FALSE. Verdict noise_level: FTR-04 weak specificity "
            "(glm4 LOEO 0.4800 < 0.5 bar) attributable to NEITHER S_class geometry NOR readout pathway; "
            "E-level unchanged, held_out closure of 3176 maintained. Fidelity: LOEO acc re-run "
            "0.6019/0.5981/0.4800 bit-identical vs sealed 3176 arm_a. 8 source files sha-asserted; content "
            "double-lock p3169 49430a39/018c6024 + ratio_B anchor 2.5388114997805062; registry v1.7 -> v1.8 "
            "append-only (23 legacy features byte-preserved, FTR-04 counter_evidence quantified append + "
            "attribution anchor; d08e9064); atlas_v1_9.html 708 fields 0/0/0/0 (79acbb6c). "
            "res 1e95a19d / seal 4ad066e0. ledger_sha256_8 NOT recomputed (recipe unknown, stale since "
            "Phase 8); pre-append file sha8 8d03febc."),
 "created": "2026-10-10T06:21:00"
}
d["measurements"].append(entry)
out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
open(P, "wb").write(out.encode("utf-8"))
print("appended: measurements 332 ->", len(d["measurements"]))
print("pre sha8", pre8, "| post sha8", sha8(open(P, "rb").read()))
d2 = json.loads(open(P, "rb").read().decode("utf-8"))
assert len(d2["measurements"]) == 333 and d2["measurements"][-1]["phase"] == 3181
print("read-back OK")
