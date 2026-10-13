# -*- coding: utf-8 -*-
"""Phase 3187 closeout: atlas_ledger.json append-only（G 线 measurement #339）"""
import json, hashlib, shutil

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json"
BAK = r"D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/atlas_ledger_backup_pre_phase3187.json"

def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

raw = open(P, "rb").read()
pre8 = sha8(raw)
assert pre8 == "17c4f3e6", "pre-append file sha changed: %s" % pre8
shutil.copyfile(P, BAK)

d = json.loads(raw.decode("utf-8"))
assert len(d["measurements"]) == 338, len(d["measurements"])
assert d["measurements"][-1]["phase"] == 3186

entry = {
 "phase": 3187,
 "name": "g5b12_alpha_calibration",
 "line": "G",
 "date": "2026-10-10",
 "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
 "verdict": "g5b12_alpha_calibration|alpha_unstable_direction_level_downgrade|alpha_hat_+1.1672_-0.0131_+1.7448|14b_sign_flip_+0.76_-0.79|band_ratio_133.28|explained_rate_equals_rA2_x6|registry_v114|html_fields_736_ok",
 "evidence_level": "E1_repeatable_unchanged",
 "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
 "prereg_id": "G5-B12",
 "superseded_by": None,
 "detail": ("G5-B12 R_rel mean-level alpha calibration, T2 closing arm (zero GPU; "
            "sealed 3157 collect + 3177/3178/3182/3183/3185 results). One-parameter "
            "LS on the 3185 prediction-A form g_e = gbar - kappa*alpha*gam_e per "
            "model per held-relation fold: alpha_hat 4b +1.1348/+1.1997 (isa/hasa, "
            "explained 0.2235/0.4113), 14b +0.7602/-0.7864 (SIGN FLIP, explained "
            "0.0219/0.0477), glm4 +2.1793/+1.3102 (explained 0.4195/0.3420); "
            "alpha_m +1.1672/-0.0131/+1.7448; cross-model band ratio 133.28 (gate "
            "<=3) -- 14b alpha_m ~0 kills the band. GATE UNSTABLE => T2 downgraded "
            "to DIRECTION-LEVEL statement: FTR-24 keeps the Delta-v direction claim "
            "only, amplitude parameter alpha withdrawn (entity-level claims already "
            "withdrawn at 3186). LS algebra identity verified: explained_rate == "
            "sealed 3185 r_A^2 x6 <1e-9 (replication of 3185 through an independent "
            "route). p_x-side one-parameter alpha_px report-only: 4b +1.36/+1.19, "
            "14b +0.59/-1.03, glm4 +2.08/+1.20 (same 14b inconsistency). "
            "Substantive reading: 4b and glm4 give consistent alpha_hat ~1.2-1.7 "
            "(Delta-v amplitude at the scale of the relation gap -- the mean-level "
            "linear form IS meaningful for 2 of 3 models), but 14b's antisymmetric "
            "fold structure (consistent with its 3185 hasa r_A=-0.2185) breaks "
            "cross-model calibration; per the pinned gate (x3 required) the "
            "direction-level downgrade is the honest disposition -- alpha may be "
            "revisited per-model-family but no cross-model scalar is claimable. "
            "Fidelity: 15 source files sha-asserted; 3157 anchors double-locked; "
            "sealed 3177/3178/3182/3183/3185 double-locked; replication gates all "
            "<1e-9 (3177 folds, 3178 r_full, 3182 p_x, 3183 dv); device tc drift "
            "<1e-9; SMOKE/FULL per_model readings byte-identical. FULL-1 crash "
            "registered: finish_registry FAIL-branch format referenced "
            "summary['band_ratio'] instead of summary['gates']['band_ratio'], "
            "crashed BEFORE any registry write; fixed and FULL re-run, arm "
            "res/seal byte-identical across runs (10140b94/cb1bce5b) -- no "
            "observation leakage. Registry v1.13 -> v1.14 (95fe55f5): FTR-24 "
            "statement append only (direction-level downgrade; anchors 3 and "
            "counter_evidence unchanged), 23 legacy features byte-preserved, "
            "failures 19 / upgrade_log 6 unchanged; atlas_v1_15.html a29022d5 "
            "736 fields 0/0/0/0. res 10140b94 / seal cb1bce5b; design 760b886d. "
            "pre-append file sha8 17c4f3e6."),
 "created": "2026-10-10T13:15:00"
}
d["measurements"].append(entry)
out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
open(P, "wb").write(out.encode("utf-8"))
print("appended: measurements 338 ->", len(d["measurements"]))
print("pre sha8", pre8, "| post sha8", sha8(open(P, "rb").read()))
d2 = json.loads(open(P, "rb").read().decode("utf-8"))
assert len(d2["measurements"]) == 339 and d2["measurements"][-1]["phase"] == 3187
pre = json.loads(open(BAK, "rb").read().decode("utf-8"))
assert json.dumps(pre["measurements"], ensure_ascii=False, sort_keys=True) == \
    json.dumps(d2["measurements"][:338], ensure_ascii=False, sort_keys=True)
print("prefix equivalence OK")
