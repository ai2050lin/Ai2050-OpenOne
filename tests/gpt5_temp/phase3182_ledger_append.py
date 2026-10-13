# -*- coding: utf-8 -*-
"""Phase 3182 closeout: atlas_ledger.json append-only（G 线 measurement #334）"""
import json, hashlib, shutil

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json"
BAK = r"D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/atlas_ledger_backup_pre_phase3182.json"

def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

raw = open(P, "rb").read()
pre8 = sha8(raw)
assert pre8 == "72567879", "pre-append file sha changed: %s" % pre8
shutil.copyfile(P, BAK)

d = json.loads(raw.decode("utf-8"))
assert len(d["measurements"]) == 333, len(d["measurements"])
assert d["measurements"][-1]["phase"] == 3181

entry = {
 "phase": 3182,
 "name": "g5b7_ftr23_heldout_loro",
 "line": "G",
 "date": "2026-10-10",
 "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
 "verdict": "g5b7_ftr23_heldout_loro|held_out_gate_fail_F17|delta_0.1070_0.1048_0.1368|ftr23_stays_E1|registry_v19|html_fields_713_ok",
 "evidence_level": "E1_repeatable_unchanged",
 "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
 "prereg_id": "G5-B7",
 "superseded_by": None,
 "detail": ("G5-B7 FTR-23 held_out leave-one-relation-out cross-null arm (zero GPU; sealed "
            "3157 collect). r_full recomputed with the 3178 verbatim recipe: drift 0.00e+00 "
            "vs sealed 3178 x3 models (0.0918/0.0951/0.1289). NEW cross-relation null "
            "(calib = other relation, 30-cell entity-LOO; read = held relation 2 cells): "
            "p_x(m,rel) 0.1299/0.1342 (4b isa/hasa), 0.1507/0.1035 (14b), 0.1579/0.1564 "
            "(glm4) -- 5-8x the within-relation sampling penalty (~0.02), fully absorbing "
            "the relation gap g. r_held -0.0153/-0.0097/-0.0079; delta vs r_full "
            "0.1070/0.1048/0.1368 all >= 0.1 gate => FAIL: held_out upgrade denied, F17 "
            "registered, FTR-23 stays E1_repeatable, counter_evidence quantified append. "
            "Substantive reading: the relation-specific residual does NOT replicate as a "
            "residual under cross-relation calibration -- it reappears as a symmetric "
            "cross-relation readout penalty (isa<->hasa direction, p_x isa~hasa within "
            "0.02), consistent with a relation-offset structure rather than "
            "relation-asymmetric noise. Fidelity: 12 source files sha-asserted; 3157 "
            "anchors double-locked; 3177 d451030f/7b746eb2 + 3178 4f521d9b/1f166645 "
            "double-locked; 96/96 null + 96/96 xnull sanity |p|<0.5; device drift < 1e-9. "
            "Pinning note: prereg text said null calibrated from ALL relations; non-leaky "
            "version (other relation only) pinned at freeze, deviation registered. "
            "Registry v1.8 -> v1.9 (59a91512): FTR-23 counter_evidence append + F17 "
            "failures append, 22 legacy features byte-preserved; atlas_v1_10.html "
            "0f24b739 713 fields 0/0/0/0. res 5f08426f / seal b6a57f07; design ea0a1e7a. "
            "ledger_sha256_8 NOT recomputed (recipe unknown, stale since Phase 8); "
            "pre-append file sha8 72567879."),
 "created": "2026-10-10T11:10:00"
}
d["measurements"].append(entry)
out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
open(P, "wb").write(out.encode("utf-8"))
print("appended: measurements 333 ->", len(d["measurements"]))
print("pre sha8", pre8, "| post sha8", sha8(open(P, "rb").read()))
d2 = json.loads(open(P, "rb").read().decode("utf-8"))
assert len(d2["measurements"]) == 334 and d2["measurements"][-1]["phase"] == 3182
print("read-back OK")
