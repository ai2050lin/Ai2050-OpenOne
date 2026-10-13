# -*- coding: utf-8 -*-
"""Phase 3200 ledger append: measurement #352 (G5-E2), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 351
assert ms[-1]['phase'] == 3199
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m352 = {
    "phase": 3200, "name": "g5e2_ftr23_heldout_upgrade", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5e2_ftr23_heldout_upgrade|HOLDOUT_PREDICTIVE|6of6_units|med_cos_out_isa_0.7696_0.7564_0.7762|hasa_0.7429_0.7855_0.7885|gate_ge_0.5_all_units|min_fold_0.6091|ftr23_E1_to_E2_held_out_upgrade_log_9|registry_v1.22_9fd87433_to_v1.23_5f5b9e73|atlas_v1_25_7dc08f1e_782_fields_clean",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-E2", "superseded_by": None,
    "detail": ("G5-E2 FTR-23 held_out upgrade arm (registration arm on sealed "
               "scalars, 3176 batch-upgrade precedent; zero GPU; sealed p3178 "
               "result abae296e file-locked + content double lock 4f521d9b/"
               "1f166645 + sealed 3157 npz x3 + sealed 3183 result). Protocol "
               "refined PRE-freeze and registered: FTR-23 has no sealed "
               "direction vector in the 3169 OOV row space (scalar phenomenon "
               "r_m/p_m on the 3157 context-transform panel), so the held-out "
               "arm evaluates the SEALED 3178 entity-LOO null folds cos_out "
               "(held entity's 2 cells vs within-relation mean direction "
               "rebuilt from the other 15 entities, 16 folds x 2 relations x "
               "3 models, 3178 protocol verbatim) - the panel-internal "
               "held-out reading. Gate (pinned): median cos_out >= 0.5 in ALL "
               "6 (model, relation) units. Observed: HOLDOUT_PREDICTIVE 6/6 - "
               "med isa 0.7696/0.7564/0.7762, hasa 0.7429/0.7855/0.7885; min "
               "fold 0.6880/0.6224/0.6918 (isa), 0.7128/0.6091/0.6941 (hasa). "
               "Interpretation: within a relation, the context-shift geometry "
               "is entity-held-out predictable (median cos ~0.74-0.79) - "
               "FTR-23's relation-specific residual is a BETWEEN-relation "
               "offset, NOT within-relation entity noise; together with 3199 "
               "(dvu NOT predictive on OOV rows) this completes the E-line "
               "picture: held-out predictability lives in class-mean/"
               "relation-internal structure, while between-relation offsets "
               "and relation directions carry no cross-panel prediction. "
               "FTR-23 E1_repeatable -> E2_held_out (upgrade_log #9), 25 "
               "legacy features byte-preserved, failures F1-F19 unchanged, "
               "counter_evidence 5 entries preserved. Registry v1.22 "
               "9fd87433 -> v1.23 5f5b9e73 (append-only); atlas_v1_25.html "
               "7dc08f1e (782/782 fields, 0 missing/extra/mismatch/dups). "
               "Device/control anchors: 3157 tc + 3183 dv scalars bitwise x3 "
               "(build_dvu); sealed 3178 scalars re-read vs pinned 4dp "
               "constants drift == 0 x3 (medians probed from sealed p3178 "
               "BEFORE freeze - measured-anchor discipline; exch_mean "
               "1.2728174525072664 bitwise). Honest log: prereg protocol "
               "refined pre-freeze (OOV-row projection -> sealed-null-fold "
               "evaluation) since FTR-23 has no OOV-panel direction; gate "
               "bar 0.5 pinned after pre-freeze probe of sealed medians "
               "(conservative: half the observed band floor, above the 0.3 "
               "OOV-panel precedent). SMOKE/FULL payloads bitwise identical. "
               "design 2ea6e1f6, res f9a280fc/seal cba9d56c, final "
               "ae7a5837/dfd3b880."),
    "chain": {"design_sha8": "2ea6e1f6", "res_sha8": "f9a280fc", "seal_sha8": "cba9d56c",
              "final_res_sha8": "ae7a5837", "final_seal_sha8": "dfd3b880",
              "registry_v123": "5f5b9e73", "registry_v122": "9fd87433",
              "atlas_v1_25": "7dc08f1e"},
}
d['measurements'] = ms + [m352]
assert len(d['measurements']) == 352
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3200')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 352
assert json.dumps(d2['measurements'][:351], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #352 appended: post %s' % post)
