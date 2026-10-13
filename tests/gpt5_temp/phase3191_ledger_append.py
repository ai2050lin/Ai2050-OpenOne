# -*- coding: utf-8 -*-
"""Phase 3191 ledger append: measurement #343 (G5-C4), prefix-equivalence asserted."""
import hashlib
import io
import json
import shutil

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
raw = open(P, 'rb').read()
d = json.loads(raw.decode('utf-8'))
ms = d['measurements']
assert len(ms) == 342
assert ms[-1]['phase'] == 3190
old_blob = json.dumps(ms, ensure_ascii=False, sort_keys=True)

m343 = {
    "phase": 3191, "name": "g5c4_ftr25_heldout_upgrade", "line": "G",
    "date": "2026-10-10", "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5c4_ftr25_heldout_upgrade|heldout_predictive_x3|min_med_abs_cos_0.3906_0.4174_0.3717|FTR25_E1_repeatable_to_E2_held_out|upgrade_log_7|registry_v118",
    "evidence_level": "E2_held_out",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-C4", "superseded_by": None,
    "detail": ("G5-C4 FTR-25 held_out upgrade arm (C line, zero GPU; sealed "
               "3169 collect x3 + 3169/3189/3190 results). Residual chain = "
               "3190 verbatim; device anchors: 3189 sealed scalars drift == 0 "
               "bitwise, 3190 sealed scalars (top1/cross_med/within/loo_max_delta/"
               "loo_top1) drift == 0 bitwise, 3169 per-seed E replication 0 drift "
               "- chain identical across 3189/3190/3191. Gate (pinned "
               "pre-observation): HOLDOUT_PREDICTIVE iff median |cos(row, "
               "unit-mean of other 3 classes' directions)| >= 0.3 in ALL 4 folds "
               "x >= 2/3 models. Observed 3/3: med|cos_held| 4b "
               "0.4432/0.3906/0.4931/0.4547 (min 0.3906), 14b "
               "0.4405/0.4174/0.5001/0.4296 (min 0.4174), glm4 "
               "0.4233/0.3717/0.4330/0.3807 (min 0.3717) - 3-class mean direction "
               "PREDICTS held-out class row directions cross-model: FTR-25 "
               "E1_repeatable -> E2_held_out (18th E2; upgrade_log #7). "
               "Report-only KS(|cos_held| vs within-class pairwise cos) = 0.17-"
               "0.47 (held-class rows spread wider than class-internal cos, "
               "consistent with top1 ~0.30 shared core + within-class dispersion). "
               "GAP-4 residual mechanism now at E2: shared OOV collapse direction "
               "is cross-model held-out PREDICTIVE geometry. Honest log: SMOKE-1 "
               "CONTENT_3190 anchor mis-set to final shas (fc8be7db/f7d8571a) "
               "instead of arm-stage content anchors embedded in p3190 "
               "result.json (24461856/0c975f5b) - content sha != file sha lesson "
               "re-confirmed; FULL-1 crash on FTR-25 e-level field name "
               "(e_level vs evidence_level legacy convention), crashed before "
               "final write, arm res/seal bitwise identical across reruns "
               "836fe933/a66c0b3d, registry v1.18 rewritten c08cbcc2. registry "
               "v1.17->v1.18 (c08cbcc2, append-only), atlas_v1_19.html "
               "(d5157665, 754 fields 0/0/0/0), design c4673cc3, final res "
               "28a852b0/seal 6eb89727."),
    "chain": {"design_sha8": "c4673cc3", "res_sha8": "28a852b0", "seal_sha8": "6eb89727",
              "arm_res_sha8": "836fe933", "arm_seal_sha8": "a66c0b3d",
              "registry_v118": "c08cbcc2", "atlas_v1_19": "d5157665"},
}
d['measurements'] = ms + [m343]
assert len(d['measurements']) == 343
new_blob = json.dumps(d['measurements'], ensure_ascii=False, sort_keys=True)
assert new_blob.startswith(old_blob[:-1]), 'prefix not preserved'
shutil.copyfile(P, P + '.bak_3191')
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(d, ensure_ascii=False, indent=1))
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
d2 = json.load(io.open(P, encoding='utf-8'))
assert len(d2['measurements']) == 343
assert json.dumps(d2['measurements'][:342], ensure_ascii=False, sort_keys=True) == old_blob
print('ledger #343 appended: post %s' % post)
