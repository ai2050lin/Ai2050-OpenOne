# -*- coding: utf-8 -*-
"""Phase 3201 ledger append: measurement #353 (G5-E3 E-line closing audit), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 352
assert ms[-1]['phase'] == 3200
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m353 = {
    "phase": 3201, "name": "g5e3_eline_audit", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5e3_eline_audit|E_LINE_CLOSED|A1_domain_counts_ok_E2held3_E3_3_E1_3_E2pred17|A2a_held_out_anchors_FTR23_25_26|A2b_primary_intervention_FTR10_11_16|A2c_counter_evidence_FTR04_14_24|A3_upgrade_chain_9_pinned|0_contradictions|ftr24_scope_limits_methodology_append|registry_v1.23_5f5b9e73_to_v1.24_0d74b628|atlas_v1_26_fbc064d4_783_fields_clean",
    "evidence_level": "E_LINE_CLOSED",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-E3", "superseded_by": None,
    "detail": ("G5-E3 E-line closing audit arm (registration arm on sealed "
               "registry v1.23, 3176 batch precedent; zero GPU). Pinned audit "
               "A1-A3 over all 26 features: A1 domain (every evidence_level "
               "in {E0_candidate,E1_repeatable,E2_predictive,E2_held_out,"
               "E3_causal_scoped}) + counts E2_held_out==3 / E3==3 / E1==3 / "
               "E2_predictive==17; A2a FTR-23/25/26 each carry role=held_out "
               "anchor; A2b FTR-10/11/16 each carry role=primary_intervention; "
               "A2c FTR-04/14/24 each have >=1 counter_evidence; A3 upgrade_log "
               "node sequence == 9 pinned entries (N04->FTR-13, N05->FTR-12, "
               "N13->FTR-16, FTR-06, FTR-08, FTR-15, FTR-25, FTR-26, FTR-23). "
               "Observed: AUDIT CLEAN, 0 contradictions => E_LINE_CLOSED. "
               "A4 closing: FTR-24 scope_limits append ONLY (E-line methodology "
               "verdict: held-out predictability geometric divide - class-mean/"
               "within-relation structure carries it (FTR-25/26 OOV med|cos|>=0.3, "
               "FTR-23 within-relation entity holdout med cos 0.74-0.79), "
               "between-relation offsets and relation directions do not (dvu "
               "med|cos| 0.035-0.060); E1 residual falsifiable records complete; "
               "taxonomy note: E2_held_out is the held-out strengthened tier of "
               "E2_predictive, legacy values byte-preserved); all other features "
               "+ upgrade_log + failures F1-F19 byte-preserved (25 legacy "
               "features assert-equal). Appendix inventory: all 5 items GPU/"
               "crossline dependent, zero-GPU candidate set empty. Control "
               "anchors: sealed 3178 scalars 4dp re-read drift==0 x3 (ANCH_3178 "
               "carried from 3200) + 3183 dv bitwise + 3157 tc drift<1e-9 + "
               "registry chain v1.20-v1.23 sha refs. Honest log: generation "
               "script P17 global reg23->reg24 rename over-reached into newly "
               "inserted code (preservation asserts self-comparing); repaired "
               "pre-FULL via phase3201_fix_rename.py (10 targeted restores, "
               "assert count==1 each); HTML sub-line initially dropped v1.20 "
               "span vs flatten key (missing=1), span restored, 783/783 clean. "
               "Registry v1.23 5f5b9e73 -> v1.24 0d74b628 (append-only, FTR-24 "
               "scope_limits only); atlas_v1_26.html fbc064d4 (783/783, 0/0/0/0). "
               "design 38fbdd1a, res 1dc22531/seal 30b08ff8, final c0350500/"
               "25cad189."),
    "chain": {"design_sha8": "38fbdd1a", "res_sha8": "1dc22531", "seal_sha8": "30b08ff8",
              "final_res_sha8": "c0350500", "final_seal_sha8": "25cad189",
              "registry_v124": "0d74b628", "registry_v123": "5f5b9e73",
              "atlas_v1_26": "fbc064d4"},
}
d['measurements'] = ms + [m353]
assert len(d['measurements']) == 353
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3201')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 353
assert json.dumps(d2['measurements'][:352], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #353 appended: post %s' % post)
