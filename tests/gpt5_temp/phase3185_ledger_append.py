# -*- coding: utf-8 -*-
"""Phase 3185 closeout: atlas_ledger.json append-only（G 线 measurement #337）"""
import json, hashlib, shutil

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json"
BAK = r"D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/atlas_ledger_backup_pre_phase3185.json"

def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

raw = open(P, "rb").read()
pre8 = sha8(raw)
assert pre8 == "ad566a5c", "pre-append file sha changed: %s" % pre8
shutil.copyfile(P, BAK)

d = json.loads(raw.decode("utf-8"))
assert len(d["measurements"]) == 336, len(d["measurements"])
assert d["measurements"][-1]["phase"] == 3184

entry = {
 "phase": 3185,
 "name": "g5b10_rrel_predictive",
 "line": "G",
 "date": "2026-10-10",
 "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
 "verdict": "g5b10_rrel_predictive|linear_superposition_rejected_F19|r_g_0.5571_-0.0353_0.6162|r2_px_0.2241_-0.0692_0.2858|ftr24_counter_evidence_append|registry_v112|html_fields_736_ok",
 "evidence_level": "E1_repeatable_unchanged",
 "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
 "prereg_id": "G5-B10",
 "superseded_by": None,
 "detail": ("G5-B10 R_rel low-rank model quantitative predictive test, T2 main arm (zero "
            "GPU; sealed 3157 collect + 3177/3178/3182/3183/3184 results). Fit-free "
            "analytic Delta-v parameterization (dv = v_isa - v_hasa, 3183 verbatim): "
            "(A) 3177 relation gap entity-level g_e = cos_train - u_e@v_c predicted by "
            "g_pred = gbar - kap*gam_e (gam_e = (u_e - vh_t)@dvu, kap = dvu@v_c; exact "
            "identities mean(g_e)=gbar, mean(gam_e)=0 verified <1e-6 float32); r_g "
            "0.5571/-0.0353/0.6162 (4b/14b/glm4; fold-wise 4b 0.4728/0.6413, 14b "
            "0.1478/-0.2185, glm4 0.6477/0.5848). (B) 3182 LORO cross-null p_x "
            "entity-level distribution predicted by p_pred = b_e - gam_e*kap_x(e) "
            "(fit-free); R2_px 0.2241/-0.0692/0.2858 (r_B 0.6166/0.6351 4b, "
            "0.3559/-0.2105 14b, 0.6876/0.5750 glm4). GATE (r_g>=0.5 AND r2_px>=0.5 x3) "
            "FAIL: r_g passes 2/3 (14b fails, hasa fold r=-0.2185), R2_px fails 3/3 "
            "(fit-free; even charitable OLS r^2 reading gives 4b 0.39/0.40, 14b neg, "
            "glm4 0.33-0.47, still <0.5). => F19 rrel_linear_superposition_rejected "
            "registered; FTR-24 counter_evidence append (stays E1); T2 route revised: "
            "R_rel low-rank direction exists (FTR-24) but superposition is NONLINEAR "
            "-- dv-component predicts gap/p_x sign-correlation (r_B up to 0.69) yet "
            "explains <30% variance; the entity-level deviation spectrum is NOT "
            "captured by a single relation-offset direction. Residual spectrum: "
            "max-abs entities e15 (4b/glm4/14b-isa) and e12 (14b-hasa, consistent "
            "with 3184 outlier e=12). Fidelity: 15 source files sha-asserted; 3157 "
            "anchors double-locked; sealed 3177 d451030f/7b746eb2 + 3178 4f521d9b/"
            "1f166645 + 3182 33d73196/ed51ca22 + 3183 c55e4e5e/7196fa8a + 3184 "
            "54e2b9d9/55706292 double-locked; replication gates all 0.00e+00 (3177 "
            "folds, 3178 r_full, 3182 p_x, 3183 dv quantities); device tc drift "
            "<1e-9; SMOKE/FULL byte-identical numbers. SMOKE-1 fix registered: "
            "exact-identity asserts loosened 1e-12 -> 1e-6 (float32 accumulation), "
            "FULL before any FULL observation. Protocol note: prereg '3177 relation "
            "gap g_e' operationalized entity-level as g_e = cos_train - u_e@v_c "
            "(3177 verbatim components), pinned at freeze. Registry v1.11 -> v1.12 "
            "(269a37c0): F19 failures append + FTR-24 counter_evidence append, 23 "
            "legacy features byte-preserved, upgrade_log 6 unchanged; atlas_v1_13.html "
            "a3bd2206 736 fields 0/0/0/0. res df958cc9 / seal 7559b27b; design "
            "65ee0dcc. pre-append file sha8 ad566a5c."),
 "created": "2026-10-10T12:40:00"
}
d["measurements"].append(entry)
out = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
open(P, "wb").write(out.encode("utf-8"))
print("appended: measurements 336 ->", len(d["measurements"]))
print("pre sha8", pre8, "| post sha8", sha8(open(P, "rb").read()))
d2 = json.loads(open(P, "rb").read().decode("utf-8"))
assert len(d2["measurements"]) == 337 and d2["measurements"][-1]["phase"] == 3185
# prefix equivalence
pre = json.loads(open(BAK, "rb").read().decode("utf-8"))
assert json.dumps(pre["measurements"], ensure_ascii=False, sort_keys=True) == \
    json.dumps(d2["measurements"][:336], ensure_ascii=False, sort_keys=True)
print("prefix equivalence OK")
