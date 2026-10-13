# -*- coding: utf-8 -*-
"""Phase 3186 closeout: atlas_ledger.json append-only（G 线 measurement #338）"""
import json, hashlib, shutil

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json"
BAK = r"D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/atlas_ledger_backup_pre_phase3186.json"

def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

raw = open(P, "rb").read()
pre8 = sha8(raw)
assert pre8 == "4fff0f4a", "pre-append file sha changed: %s" % pre8
shutil.copyfile(P, BAK)

d = json.loads(raw.decode("utf-8"))
assert len(d["measurements"]) == 337, len(d["measurements"])
assert d["measurements"][-1]["phase"] == 3185

entry = {
 "phase": 3186,
 "name": "g5b11_second_order",
 "line": "G",
 "date": "2026-10-10",
 "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
 "verdict": "g5b11_second_order|meanlevel_contraction|top2_energy_0.2458_0.3197_0.3004|r80_9-13_directions|loo_flips_none|ftr24_counter_evidence_plus_contraction|registry_v113|html_fields_737_ok",
 "evidence_level": "E1_repeatable_unchanged",
 "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
 "prereg_id": "G5-B11",
 "superseded_by": None,
 "detail": ("G5-B11 relation-offset second-order structure test, T2 revision arm (zero "
            "GPU; sealed 3157 collect + 3177/3178/3182/3183/3185 results). Arm A "
            "(spectrum): entity-level deviation delta_e = u_e - vh_t (sum-zero 15-dim, "
            "identity verified <1e-6) per relation, 16 vectors, SVD energy: top2 share "
            "0.2458/0.3197/0.3004 (4b/14b/glm4, mean of isa/hasa; top1 0.1259-0.2323), "
            "rank for 80% energy needs 9-10 directions, for 90% needs 12-13 -- "
            "FULL-RANK DIFFUSE x3 (gate 0.5 all fail). Arm B (outlier LOO, 3184 "
            "protocol): 16 folds drop one entity (vh_t -> 30-cell mean, evaluation on "
            "remaining 15), recompute r_g/R2_px per 3185 fit-free recipes: flip "
            "entities (post-removal BOTH r_g>=0.5 AND R2_px>=0.5) = NONE x3 models x2 "
            "relations; 4b isa r_g LOO 0.3744..0.6171 (removal of e15 lifts 0.47->0.62 "
            "but R2 stays <=0.33), 14b hasa r_g LOO -0.3338..-0.0396 (e12 removal does "
            "NOT flip -- 3184 outlier is not the cause here), glm4 isa "
            "0.5194..0.7442. => GATE FAIL: no plane parameterization; mean-level "
            "contraction pinned: R_rel keeps relation-level Delta-v only (FTR-24 "
            "direction statement), entity-level variation is individual diffuse noise "
            "-- F19 failure is SYSTEMIC, not outlier-carried. Substantive reading: the "
            "relation-offset direction (FTR-24, aggregate) and the entity-level "
            "deviation spectrum are decoupled -- aggregate low-rank does not imply "
            "entity-level low-rank; T2 closes with relation-level parameterization "
            "(Delta-M_rel = T_C + alpha*Delta-v at mean level, alpha fit next phase, "
            "no entity-level prediction claims). No new failure id (F19 already "
            "covers entity-level linear rejection; prereg pinned no-F-on-diffuse). "
            "Fidelity: 16 source files sha-asserted; 3157 anchors double-locked; "
            "sealed 3177/3178/3182/3183/3185 results double-locked; replication "
            "gates all <1e-9 incl. NEW 3185 full-panel r_A/R2_B recompute drift "
            "0.00e+00; device tc drift <1e-9; SMOKE/FULL byte-identical. Registry "
            "v1.12 -> v1.13 (730729f5): FTR-24 counter_evidence append + contraction "
            "statement append (both append-only, anchors 3 unchanged), 23 legacy "
            "features byte-preserved, failures 19 / upgrade_log 6 unchanged; "
            "atlas_v1_14.html 56c45e50 737 fields 0/0/0/0. res 1bd47cee / seal "
            "731169a0; design 2c4906b7. pre-append file sha8 4fff0f4a."),
 "created": "2026-10-10T13:05:00"
}
d["measurements"].append(entry)
out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
open(P, "wb").write(out.encode("utf-8"))
print("appended: measurements 337 ->", len(d["measurements"]))
print("pre sha8", pre8, "| post sha8", sha8(open(P, "rb").read()))
d2 = json.loads(open(P, "rb").read().decode("utf-8"))
assert len(d2["measurements"]) == 338 and d2["measurements"][-1]["phase"] == 3186
pre = json.loads(open(BAK, "rb").read().decode("utf-8"))
assert json.dumps(pre["measurements"], ensure_ascii=False, sort_keys=True) == \
    json.dumps(d2["measurements"][:337], ensure_ascii=False, sort_keys=True)
print("prefix equivalence OK")
