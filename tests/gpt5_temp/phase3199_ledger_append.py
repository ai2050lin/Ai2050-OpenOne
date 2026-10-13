# -*- coding: utf-8 -*-
"""Phase 3199 ledger append: measurement #351 (G5-E1), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 350
assert ms[-1]['phase'] == 3198
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m351 = {
    "phase": 3199, "name": "g5e1_ftr24_heldout_upgrade", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5e1_ftr24_heldout_upgrade|NOT_PREDICTIVE|0of3|med_cos_held_dvu_min_0.0404_0.0598_0.0355_all_below_0.3|full_panel_med_0.0610_0.0734_0.0535|dvu_no_heldout_oov_predictability|consistent_with_3188_DECOUPLED_and_3189_class_mean_dominance|ftr24_counter_evidence_append_E1_unchanged|registry_v1.21_47cf01e0_to_v1.22_9fd87433|atlas_v1_24_74a11492_775_fields_clean",
    "evidence_level": "E1_repeatable",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-E1", "superseded_by": None,
    "detail": ("G5-E1 FTR-24 held_out upgrade arm (3194 protocol; zero GPU; "
               "sealed 3169/3183/3189/3190/3191/3192/3193 reuse + sealed 3197 "
               "file lock; prereg target corrected FTR-15 -> FTR-24 since "
               "FTR-15 was already upgraded E2_held_out in Phase 3180). "
               "Protocol: per model, 4 folds hold out one OOV class (219 row "
               "directions from the 876-row 3169 B-protocol residual chain); "
               "P1 = FTR-24 sealed direction dvu = unit(v_isa - v_hasa), "
               "3183/3157 verbatim (build_dvu bitwise anchors); gate (pinned): "
               "median |cos(row, dvu)| >= 0.3 in ALL 4 folds x ALL 3 models "
               "=> E1->E2_held_out + upgrade_log #9. Observed: NOT_PREDICTIVE "
               "0/3 - min med|cos_held,dvu| 0.0404/0.0598/0.0355 (4b/14b/"
               "glm4, all 12 folds far below 0.3); full-panel med|cos| "
               "0.0610/0.0734/0.0535. Interpretation: the relation-offset "
               "direction dvu (isa/hasa low-rank geometry, FTR-24 structural "
               "evidence unchanged) carries NO cross-class held-out "
               "predictability on OOV rows - held-out row directions are "
               "carried by class-mean structure (u*/v_c, FTR-25/26), NOT by "
               "the relation direction. Consistent with 3188 DECOUPLED "
               "(dvu x spectral-collapse amplitude share ~ noise) and 3189 "
               "class-mean dominance. FTR-24 counter_evidence appended (4th "
               "entry), evidence_level UNCHANGED E1_repeatable, upgrade_log "
               "unchanged 8, failures F1-F19 unchanged; 25 legacy features "
               "byte-preserved. Registry v1.21 47cf01e0 -> v1.22 9fd87433 "
               "(append-only); atlas_v1_24.html 74a11492 (775/775 fields, 0 "
               "missing/extra/mismatch/dups). Device anchors: drift69/89/90/"
               "91/93 all == 0 bitwise x3, dv_drift == 0, tc drift < 1e-9; "
               "3197 signed_weights/ls_resid 4dp re-read drift == 0 x3 "
               "(ANCH_3197 pinned pre-freeze, carried from 3198). Honest "
               "log: SMOKE/FULL payloads bitwise identical (zero GPU). "
               "design cfbe183c, res 94442e48/seal b908731e, final "
               "06644f1f/82009c4a."),
    "chain": {"design_sha8": "cfbe183c", "res_sha8": "94442e48", "seal_sha8": "b908731e",
              "final_res_sha8": "06644f1f", "final_seal_sha8": "82009c4a",
              "registry_v122": "9fd87433", "registry_v121": "47cf01e0",
              "atlas_v1_24": "74a11492"},
}
d['measurements'] = ms + [m351]
assert len(d['measurements']) == 351
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3199')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 351
assert json.dumps(d2['measurements'][:350], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #351 appended: post %s' % post)
